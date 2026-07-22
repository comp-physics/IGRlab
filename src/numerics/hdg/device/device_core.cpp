#include "numerics/hdg/device/device_core.hpp"

#include "utils/profiling/observability.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>

namespace dg::hdg::device
{
namespace
{
using core::Status;
using ScopedRegion = dg_igr::observability::ScopedRegion<
    dg_igr::observability::DefaultBackend>;

constexpr double pivot_scale = 64.0 * 2.2204460492503131e-16;

bool Multiply(const std::size_t left, const std::size_t right,
              std::size_t& product) noexcept
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
  {
    return false;
  }
  product = left * right;
  return true;
}

bool FitsMfemSize(const std::size_t count) noexcept
{
  return count <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

template <typename Scalar>
void Pack(const StridedBlockView<const Scalar> input, Scalar* packed) noexcept
{
  std::size_t entry = 0;
  for (LocalIndex batch = 0; batch < input.batch_count; ++batch)
    for (LocalIndex row = 0; row < input.rows; ++row)
      for (LocalIndex column = 0; column < input.columns; ++column)
      {
        packed[entry++] = input(batch, row, column);
      }
}

void CopyOut(const double* packed, const std::size_t batches,
             const std::size_t rows, const std::size_t columns,
             const StridedBlockView<double> output) noexcept
{
  std::size_t entry = 0;
  for (std::size_t batch = 0; batch < batches; ++batch)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t column = 0; column < columns; ++column)
      {
        output(static_cast<LocalIndex>(batch), static_cast<LocalIndex>(row),
               static_cast<LocalIndex>(column)) = packed[entry++];
      }
}

bool Matches(const double* packed, const std::size_t batches,
             const std::size_t rows, const std::size_t columns,
             const StridedBlockView<double> reference) noexcept
{
  constexpr double tolerance = 1024.0 * std::numeric_limits<double>::epsilon();
  std::size_t entry = 0;
  for (std::size_t batch = 0; batch < batches; ++batch)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t column = 0; column < columns; ++column)
      {
        const double device_value = packed[entry++];
        const double reference_value =
            reference(static_cast<LocalIndex>(batch),
                      static_cast<LocalIndex>(row),
                      static_cast<LocalIndex>(column));
        if (!std::isfinite(device_value) ||
            std::abs(device_value - reference_value) >
                tolerance * std::max(1.0, std::abs(reference_value)))
        {
          return false;
        }
      }
  return true;
}

MFEM_HOST_DEVICE void SolvePivoted(const double* lu, const double* pivots,
                                   const int ni, double* values,
                                   const int columns)
{
  for (int row = 0; row < ni; ++row)
  {
    const int pivot = static_cast<int>(pivots[row]);
    if (pivot != row)
      for (int column = 0; column < columns; ++column)
      {
        const double temporary = values[row * columns + column];
        values[row * columns + column] = values[pivot * columns + column];
        values[pivot * columns + column] = temporary;
      }
    for (int below = row + 1; below < ni; ++below)
      for (int column = 0; column < columns; ++column)
      {
        values[below * columns + column] -=
            lu[below * ni + row] * values[row * columns + column];
      }
  }
  for (int row = ni; row-- > 0;)
    for (int column = 0; column < columns; ++column)
    {
      for (int right = row + 1; right < ni; ++right)
      {
        values[row * columns + column] -=
            lu[row * ni + right] * values[right * columns + column];
      }
      values[row * columns + column] /= lu[row * ni + row];
    }
}
}  // namespace

bool DeviceCoreEngine::DeviceEligible() const noexcept
{
  // Do not cause a backend query in a normal CPU process.  A configured MFEM
  // device is an explicit caller decision; a multi-device configuration falls
  // back because this bounded core is single-GPU only.
  return mfem::Device::IsConfigured() &&
         mfem::Device::Allows(mfem::Backend::DEVICE_MASK) &&
         mfem::Device::GetDeviceCount() == 1;
}

void DeviceCoreEngine::RecordHostToDevice(const std::size_t bytes) noexcept
{
  accounting_.host_to_device_bytes += bytes;
}

void DeviceCoreEngine::RecordDeviceToHost(const std::size_t bytes) noexcept
{
  accounting_.device_to_host_bytes += bytes;
}

bool DeviceCoreEngine::ResizePersistent(const std::size_t ni,
                                        const std::size_t nt,
                                        const std::size_t batches) noexcept
{
  std::size_t ni2 = 0, ni_nt = 0, nt_ni = 0, nt2 = 0, solve_per_batch = 0;
  if (!Multiply(ni, ni, ni2) || !Multiply(ni, nt, ni_nt) ||
      !Multiply(nt, ni, nt_ni) || !Multiply(nt, nt, nt2) ||
      !Multiply(ni, nt + 1, solve_per_batch))
  {
    return false;
  }
  const std::size_t counts[] = {ni2, ni, ni_nt, nt_ni, nt2, ni,
                                solve_per_batch, nt2, nt, nt, nt, ni};
  std::size_t total_bytes = 0;
  for (const std::size_t per_batch : counts)
  {
    std::size_t count = 0;
    if (!Multiply(batches, per_batch, count) || !FitsMfemSize(count) ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        total_bytes > std::numeric_limits<std::size_t>::max() -
                          count * sizeof(double))
    {
      return false;
    }
    total_bytes += count * sizeof(double);
  }
  std::array<mfem::Vector*, 12> vectors{{
      &lu_, &pivots_, &b_, &c_, &d_, &f_, &solve_, &schur_, &rhs_, &trace_,
      &action_, &interior_}};
  std::array<int, 12> old_capacities{};
  for (std::size_t index = 0; index < vectors.size(); ++index)
  {
    old_capacities[index] = vectors[index]->Capacity();
  }
  try
  {
    const auto resize = [](mfem::Vector& vector, const std::size_t count)
    {
      vector.SetSize(static_cast<int>(count));
      vector.UseDevice(true);
    };
    resize(lu_, batches * ni2);
    resize(pivots_, batches * ni);
    resize(b_, batches * ni_nt);
    resize(c_, batches * nt_ni);
    resize(d_, batches * nt2);
    resize(f_, batches * ni);
    resize(solve_, batches * solve_per_batch);
    resize(schur_, batches * nt2);
    resize(rhs_, batches * nt);
    resize(trace_, batches * nt);
    resize(action_, batches * nt);
    resize(interior_, batches * ni);
  }
  catch (const std::bad_alloc&)
  {
    return false;
  }
  catch (...)
  {
    return false;
  }
  std::size_t persistent_bytes = 0;
  for (std::size_t index = 0; index < vectors.size(); ++index)
  {
    const std::size_t capacity =
        static_cast<std::size_t>(vectors[index]->Capacity());
    if (capacity > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        persistent_bytes > std::numeric_limits<std::size_t>::max() -
                               capacity * sizeof(double))
    {
      return false;
    }
    persistent_bytes += capacity * sizeof(double);
    if (vectors[index]->Capacity() != old_capacities[index])
    {
      ++accounting_.allocation_events;
    }
  }
  accounting_.persistent_bytes = persistent_bytes;
  return true;
}

Status DeviceCoreEngine::Condense(const core::LocalSystemBatchView& input,
                                  const core::CondensedBatchView& output) noexcept
{
  ScopedRegion condensation(
      dg_igr::observability::RegionId::hdg_condensation);
  const Status reference_status = reference_.Condense(input, output);
  if (reference_status != Status::success)
  {
    return reference_status;
  }
  ready_ = true;
  shape_ = input.shape;
  batch_count_ = static_cast<std::size_t>(input.a.batch_count);
  if (!DeviceEligible())
  {
    path_ = ExecutionPath::host_reference;
    ++accounting_.host_reference_fallbacks;
    return Status::success;
  }

  const std::size_t ni = static_cast<std::size_t>(shape_.InteriorUnknowns());
  const std::size_t nt = static_cast<std::size_t>(shape_.TraceUnknowns());
  if (!ResizePersistent(ni, nt, batch_count_))
  {
    path_ = ExecutionPath::host_reference;
    ++accounting_.host_reference_fallbacks;
    return Status::success;
  }
  {
    ScopedRegion packing(
        dg_igr::observability::RegionId::hdg_packing);
    Pack(input.a, lu_.HostWrite());
    Pack(input.b, b_.HostWrite());
    Pack(input.c, c_.HostWrite());
    Pack(input.d, d_.HostWrite());
    Pack(input.f, f_.HostWrite());
    Pack(input.g, rhs_.HostWrite());
  }
  std::size_t upload_scalars = 0;
  for (const std::size_t count : {ni * ni, ni * nt, nt * ni, nt * nt, ni, nt})
  {
    upload_scalars += batch_count_ * count;
  }
  RecordHostToDevice(upload_scalars * sizeof(double));

  const int batches = static_cast<int>(batch_count_);
  const int ni_i = static_cast<int>(ni);
  const int nt_i = static_cast<int>(nt);
  double* lu = lu_.ReadWrite();
  double* pivots = pivots_.Write();
  const double* b = b_.Read();
  const double* c = c_.Read();
  const double* d = d_.Read();
  const double* f = f_.Read();
  double* solve = solve_.Write();
  double* schur = schur_.Write();
  double* rhs = rhs_.ReadWrite();
  mfem::forall(batches, [=] MFEM_HOST_DEVICE (int batch)
  {
    double* local_lu = lu + batch * ni_i * ni_i;
    double* local_pivots = pivots + batch * ni_i;
    double scale = 0.0;
    for (int entry = 0; entry < ni_i * ni_i; ++entry)
    {
      scale = fabs(local_lu[entry]) > scale ? fabs(local_lu[entry]) : scale;
    }
    for (int column = 0; column < ni_i; ++column)
    {
      int pivot = column;
      for (int row = column + 1; row < ni_i; ++row)
      {
        if (fabs(local_lu[row * ni_i + column]) >
            fabs(local_lu[pivot * ni_i + column]))
        {
          pivot = row;
        }
      }
      local_pivots[column] = static_cast<double>(pivot);
      if (pivot != column)
        for (int entry = 0; entry < ni_i; ++entry)
        {
          const double temporary = local_lu[column * ni_i + entry];
          local_lu[column * ni_i + entry] = local_lu[pivot * ni_i + entry];
          local_lu[pivot * ni_i + entry] = temporary;
        }
      if (fabs(local_lu[column * ni_i + column]) <= pivot_scale * scale)
      {
        return;
      }
      for (int row = column + 1; row < ni_i; ++row)
      {
        local_lu[row * ni_i + column] /= local_lu[column * ni_i + column];
        for (int entry = column + 1; entry < ni_i; ++entry)
        {
          local_lu[row * ni_i + entry] -=
              local_lu[row * ni_i + column] * local_lu[column * ni_i + entry];
        }
      }
    }
    double* local_solve = solve + batch * ni_i * (nt_i + 1);
    for (int row = 0; row < ni_i; ++row)
    {
      for (int column = 0; column < nt_i; ++column)
      {
        local_solve[row * (nt_i + 1) + column] =
            b[batch * ni_i * nt_i + row * nt_i + column];
      }
      local_solve[row * (nt_i + 1) + nt_i] = f[batch * ni_i + row];
    }
    SolvePivoted(local_lu, local_pivots, ni_i, local_solve, nt_i + 1);
    for (int row = 0; row < nt_i; ++row)
    {
      double condensed_rhs = rhs[batch * nt_i + row];
      for (int interior = 0; interior < ni_i; ++interior)
      {
        condensed_rhs -= c[batch * nt_i * ni_i + row * ni_i + interior] *
                         local_solve[interior * (nt_i + 1) + nt_i];
      }
      rhs[batch * nt_i + row] = condensed_rhs;
      for (int column = 0; column < nt_i; ++column)
      {
        double value = d[batch * nt_i * nt_i + row * nt_i + column];
        for (int interior = 0; interior < ni_i; ++interior)
        {
          value -= c[batch * nt_i * ni_i + row * ni_i + interior] *
                   local_solve[interior * (nt_i + 1) + column];
        }
        schur[batch * nt_i * nt_i + row * nt_i + column] = value;
      }
    }
  });

  const double* host_schur = schur_.HostRead();
  const double* host_rhs = rhs_.HostRead();
  RecordDeviceToHost(batch_count_ * (nt * nt + nt) * sizeof(double));
  if (!Matches(host_schur, batch_count_, nt, nt, output.schur) ||
      !Matches(host_rhs, batch_count_, nt, 1, output.right_hand_side))
  {
    path_ = ExecutionPath::host_reference;
    ++accounting_.host_reference_fallbacks;
    return Status::success;
  }
  CopyOut(host_schur, batch_count_, nt, nt, output.schur);
  CopyOut(host_rhs, batch_count_, nt, 1, output.right_hand_side);
  path_ = ExecutionPath::mfem_device;
  return Status::success;
}

Status DeviceCoreEngine::ApplySchur(const StridedBlockView<const double> trace,
                                    const StridedBlockView<double> result) noexcept
{
  ScopedRegion action_region(
      dg_igr::observability::RegionId::hdg_schur_action);
  const Status reference_status = reference_.ApplySchur(trace, result);
  if (reference_status != Status::success || path_ != ExecutionPath::mfem_device)
  {
    if (reference_status == Status::success) { ++accounting_.host_reference_fallbacks; }
    return reference_status;
  }
  const std::size_t ni = static_cast<std::size_t>(shape_.InteriorUnknowns());
  const std::size_t nt = static_cast<std::size_t>(shape_.TraceUnknowns());
  Pack(trace, trace_.HostWrite());
  RecordHostToDevice(batch_count_ * nt * sizeof(double));
  const int batches = static_cast<int>(batch_count_);
  const int ni_i = static_cast<int>(ni);
  const int nt_i = static_cast<int>(nt);
  const double* lu = lu_.Read();
  const double* pivots = pivots_.Read();
  const double* b = b_.Read();
  const double* c = c_.Read();
  const double* d = d_.Read();
  const double* device_trace = trace_.Read();
  double* solve = solve_.Write();
  double* action = action_.Write();
  mfem::forall(batches, [=] MFEM_HOST_DEVICE (int batch)
  {
    double* local = solve + batch * ni_i * (nt_i + 1);
    for (int row = 0; row < ni_i; ++row)
    {
      double value = 0.0;
      for (int column = 0; column < nt_i; ++column)
      {
        value += b[batch * ni_i * nt_i + row * nt_i + column] *
                 device_trace[batch * nt_i + column];
      }
      local[row] = value;
    }
    SolvePivoted(lu + batch * ni_i * ni_i, pivots + batch * ni_i, ni_i,
                 local, 1);
    for (int row = 0; row < nt_i; ++row)
    {
      double value = 0.0;
      for (int column = 0; column < nt_i; ++column)
      {
        value += d[batch * nt_i * nt_i + row * nt_i + column] *
                 device_trace[batch * nt_i + column];
      }
      for (int interior = 0; interior < ni_i; ++interior)
      {
        value -= c[batch * nt_i * ni_i + row * ni_i + interior] * local[interior];
      }
      action[batch * nt_i + row] = value;
    }
  });
  const double* host_action = action_.HostRead();
  RecordDeviceToHost(batch_count_ * nt * sizeof(double));
  if (!Matches(host_action, batch_count_, nt, 1, result))
  {
    path_ = ExecutionPath::host_reference;
    ++accounting_.host_reference_fallbacks;
    return Status::success;
  }
  CopyOut(host_action, batch_count_, nt, 1, result);
  return Status::success;
}

Status DeviceCoreEngine::Recover(const StridedBlockView<const double> trace,
                                 const StridedBlockView<double> interior) noexcept
{
  ScopedRegion recovery_region(
      dg_igr::observability::RegionId::hdg_interior_recovery);
  const Status reference_status = reference_.Recover(trace, interior);
  if (reference_status != Status::success || path_ != ExecutionPath::mfem_device)
  {
    if (reference_status == Status::success) { ++accounting_.host_reference_fallbacks; }
    return reference_status;
  }
  const std::size_t ni = static_cast<std::size_t>(shape_.InteriorUnknowns());
  const std::size_t nt = static_cast<std::size_t>(shape_.TraceUnknowns());
  Pack(trace, trace_.HostWrite());
  RecordHostToDevice(batch_count_ * nt * sizeof(double));
  const int batches = static_cast<int>(batch_count_);
  const int ni_i = static_cast<int>(ni);
  const int nt_i = static_cast<int>(nt);
  const double* lu = lu_.Read();
  const double* pivots = pivots_.Read();
  const double* b = b_.Read();
  const double* f = f_.Read();
  const double* device_trace = trace_.Read();
  double* device_interior = interior_.Write();
  mfem::forall(batches, [=] MFEM_HOST_DEVICE (int batch)
  {
    double* local = device_interior + batch * ni_i;
    for (int row = 0; row < ni_i; ++row)
    {
      double value = f[batch * ni_i + row];
      for (int column = 0; column < nt_i; ++column)
      {
        value -= b[batch * ni_i * nt_i + row * nt_i + column] *
                 device_trace[batch * nt_i + column];
      }
      local[row] = value;
    }
    SolvePivoted(lu + batch * ni_i * ni_i, pivots + batch * ni_i, ni_i,
                 local, 1);
  });
  const double* host_interior = interior_.HostRead();
  RecordDeviceToHost(batch_count_ * ni * sizeof(double));
  if (!Matches(host_interior, batch_count_, ni, 1, interior))
  {
    path_ = ExecutionPath::host_reference;
    ++accounting_.host_reference_fallbacks;
    return Status::success;
  }
  CopyOut(host_interior, batch_count_, ni, 1, interior);
  return Status::success;
}
}  // namespace dg::hdg::device

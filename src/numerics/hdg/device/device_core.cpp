#include "numerics/hdg/device/device_core.hpp"

#include "pivoted_solve.hpp"

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
bool ValidVectorView(const StridedBlockView<Scalar> view,
                     const std::size_t batches,
                     const std::size_t rows) noexcept
{
  if (view.data == nullptr || view.batch_count != static_cast<LocalIndex>(batches) ||
      view.rows != static_cast<LocalIndex>(rows) || view.columns != 1 ||
      view.batch_stride <= 0 || view.row_stride <= 0 || view.column_stride <= 0)
  {
    return false;
  }
  const auto maximum = std::numeric_limits<std::ptrdiff_t>::max();
  const auto fits = [maximum](const std::size_t count,
                              const std::ptrdiff_t stride) noexcept
  { return count > 0 && count - 1 <= static_cast<std::size_t>(maximum / stride); };
  if (!fits(batches, view.batch_stride) || !fits(rows, view.row_stride))
  {
    return false;
  }
  const auto batch_offset = (batches - 1) * view.batch_stride;
  const auto row_offset = (rows - 1) * view.row_stride;
  return batch_offset <= maximum - row_offset;
}

template <typename Scalar>
bool AllFinite(const StridedBlockView<Scalar> view) noexcept
{
  for (LocalIndex batch = 0; batch < view.batch_count; ++batch)
    for (LocalIndex row = 0; row < view.rows; ++row)
      if (!std::isfinite(view(batch, row, 0))) { return false; }
  return true;
}

template <typename Left, typename Right>
bool Overlaps(const StridedBlockView<Left> left,
              const StridedBlockView<Right> right) noexcept
{
  for (LocalIndex left_batch = 0; left_batch < left.batch_count; ++left_batch)
    for (LocalIndex left_row = 0; left_row < left.rows; ++left_row)
      for (LocalIndex right_batch = 0; right_batch < right.batch_count;
           ++right_batch)
        for (LocalIndex right_row = 0; right_row < right.rows; ++right_row)
          if (static_cast<const void*>(&left(left_batch, left_row, 0)) ==
              static_cast<const void*>(&right(right_batch, right_row, 0)))
          {
            return true;
          }
  return false;
}

template <typename Scalar>
bool HasDistinctAddresses(const StridedBlockView<Scalar> view) noexcept
{
  for (LocalIndex first_batch = 0; first_batch < view.batch_count;
       ++first_batch)
    for (LocalIndex first_row = 0; first_row < view.rows; ++first_row)
      for (LocalIndex second_batch = first_batch;
           second_batch < view.batch_count; ++second_batch)
        for (LocalIndex second_row =
                 second_batch == first_batch ? first_row + 1 : 0;
             second_row < view.rows; ++second_row)
          if (static_cast<const void*>(&view(first_batch, first_row, 0)) ==
              static_cast<const void*>(&view(second_batch, second_row, 0)))
          {
            return false;
          }
  return true;
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

bool PackedFinite(const double* values, const std::size_t count) noexcept
{
  for (std::size_t entry = 0; entry < count; ++entry)
    if (!std::isfinite(values[entry])) { return false; }
  return true;
}

}  // namespace

bool MakePersistentStorageLayout(const std::size_t ni, const std::size_t nt,
                                 const std::size_t batches,
                                 PersistentStorageLayout& layout) noexcept
{
  std::size_t ni2 = 0, ni_nt = 0, nt_ni = 0, nt2 = 0, solve_per_batch = 0;
  if (ni == 0 || nt == 0 || batches == 0 ||
      nt == std::numeric_limits<std::size_t>::max() || !Multiply(ni, ni, ni2) ||
      !Multiply(ni, nt, ni_nt) || !Multiply(nt, ni, nt_ni) ||
      !Multiply(nt, nt, nt2) || !Multiply(ni, nt + 1, solve_per_batch))
  {
    return false;
  }
  const std::size_t per_batch[] = {ni2, ni, ni_nt, nt_ni, nt2, ni,
                                   solve_per_batch, nt2, nt, nt, nt, ni};
  PersistentStorageLayout next{};
  for (std::size_t index = 0; index < 12; ++index)
  {
    if (!Multiply(batches, per_batch[index], next.counts[index]) ||
        !FitsMfemSize(next.counts[index]) ||
        next.counts[index] >
            std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        next.bytes > std::numeric_limits<std::size_t>::max() -
                         next.counts[index] * sizeof(double))
    {
      return false;
    }
    next.bytes += next.counts[index] * sizeof(double);
  }
  layout = next;
  return true;
}

bool DeviceCoreEngine::DeviceEligible() const noexcept
{
  // MFEM already selects a single device id.  Do not reject that selection
  // merely because the process can see several GPUs, and do not mistake the
  // host-only DEBUG_DEVICE backend for hardware GPU execution.
  return IsGpuExecutionEligible(
      {mfem::Device::IsConfigured(),
       mfem::Device::Allows(mfem::Backend::CUDA_MASK),
       mfem::Device::Allows(mfem::Backend::HIP_MASK),
       mfem::Device::Allows(mfem::Backend::DEBUG_DEVICE), 0});
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
                                        const std::size_t batches)
{
  PersistentStorageLayout layout{};
  if (!MakePersistentStorageLayout(ni, nt, batches, layout)) { return false; }
  try
  {
    // MFEM may make device allocation/registration lazy.  Establish every
    // staged vector on both sides before swapping it into the engine.  A C++
    // allocation exception leaves the live engine untouched.  Condense is
    // intentionally not noexcept: MFEM fatal allocation policy is not
    // recoverable through a false Status claim.
    std::array<mfem::Vector, 12> staged{};
    for (std::size_t index = 0; index < staged.size(); ++index)
    {
      staged[index].SetSize(static_cast<int>(layout.counts[index]));
      staged[index].UseDevice(true);
      std::fill_n(staged[index].HostWrite(), layout.counts[index], 0.0);
      (void)staged[index].Read();
    }
    std::array<mfem::Vector*, 12> live{{
        &lu_, &pivots_, &b_, &c_, &d_, &f_, &solve_, &schur_, &rhs_, &trace_,
        &action_, &interior_}};
    for (std::size_t index = 0; index < live.size(); ++index)
    {
      live[index]->Swap(staged[index]);
    }
  }
  catch (const std::bad_alloc&)
  {
    return false;
  }
  catch (...)
  {
    return false;
  }
  accounting_.persistent_bytes = layout.bytes;
  accounting_.allocation_events += 12;
  RecordHostToDevice(layout.bytes);
  return true;
}

Status DeviceCoreEngine::Condense(const core::LocalSystemBatchView& input,
                                  const core::CondensedBatchView& output)
{
  ScopedRegion condensation(
      dg_igr::observability::RegionId::hdg_condensation);
  const Status reference_status = reference_.Condense(input, output);
  if (reference_status != Status::success)
  {
    return reference_status;
  }
  const std::size_t next_batches = static_cast<std::size_t>(input.a.batch_count);
  if (!DeviceEligible())
  {
    ready_ = true;
    shape_ = input.shape;
    batch_count_ = next_batches;
    path_ = ExecutionPath::host_reference;
    ++accounting_.host_reference_fallbacks;
    return Status::success;
  }

  const std::size_t ni =
      static_cast<std::size_t>(input.shape.InteriorUnknowns());
  const std::size_t nt =
      static_cast<std::size_t>(input.shape.TraceUnknowns());
  PersistentStorageLayout layout{};
  const bool valid_layout = MakePersistentStorageLayout(ni, nt, next_batches,
                                                        layout);
  const std::array<const mfem::Vector*, 12> vectors{{
      &lu_, &pivots_, &b_, &c_, &d_, &f_, &solve_, &schur_, &rhs_, &trace_,
      &action_, &interior_}};
  const bool same_shape = ready_ && path_ == ExecutionPath::mfem_device &&
                          shape_.interior_dofs == input.shape.interior_dofs &&
                          shape_.trace_dofs == input.shape.trace_dofs &&
                          shape_.interior_components ==
                              input.shape.interior_components &&
                          shape_.trace_components == input.shape.trace_components &&
                          batch_count_ == next_batches;
  bool reusable = valid_layout && same_shape;
  for (std::size_t index = 0; reusable && index < vectors.size(); ++index)
  {
    reusable = vectors[index]->Size() == static_cast<int>(layout.counts[index]);
  }
  if (!reusable && (!valid_layout || !ResizePersistent(ni, nt, next_batches)))
  {
    ready_ = true;
    shape_ = input.shape;
    batch_count_ = next_batches;
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
    upload_scalars += next_batches * count;
  }
  RecordHostToDevice(upload_scalars * sizeof(double));

  const int batches = static_cast<int>(next_batches);
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
    detail::SolvePivoted(local_lu, local_pivots, ni_i, local_solve, nt_i + 1);
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
  RecordDeviceToHost(next_batches * (nt * nt + nt) * sizeof(double));
  if (!Matches(host_schur, next_batches, nt, nt, output.schur) ||
      !Matches(host_rhs, next_batches, nt, 1, output.right_hand_side))
  {
    ready_ = true;
    shape_ = input.shape;
    batch_count_ = next_batches;
    path_ = ExecutionPath::host_reference;
    ++accounting_.host_reference_fallbacks;
    return Status::success;
  }
  CopyOut(host_schur, next_batches, nt, nt, output.schur);
  CopyOut(host_rhs, next_batches, nt, 1, output.right_hand_side);
  ready_ = true;
  shape_ = input.shape;
  batch_count_ = next_batches;
  path_ = ExecutionPath::mfem_device;
  return Status::success;
}

Status DeviceCoreEngine::ApplySchur(const StridedBlockView<const double> trace,
                                    const StridedBlockView<double> result) noexcept
{
  ScopedRegion action_region(
      dg_igr::observability::RegionId::hdg_schur_action);
  if (!ready_) { return Status::not_ready; }
  if (path_ != ExecutionPath::mfem_device)
  {
    const Status reference_status = reference_.ApplySchur(trace, result);
    if (reference_status == Status::success) { ++accounting_.host_reference_fallbacks; }
    return reference_status;
  }
  const std::size_t ni = static_cast<std::size_t>(shape_.InteriorUnknowns());
  const std::size_t nt = static_cast<std::size_t>(shape_.TraceUnknowns());
  if (!ValidVectorView(trace, batch_count_, nt) ||
      !ValidVectorView(result, batch_count_, nt))
  {
    return Status::invalid_view;
  }
  if (!HasDistinctAddresses(trace) || !HasDistinctAddresses(result) ||
      Overlaps(trace, result))
  {
    return Status::unsupported_alias;
  }
  if (!AllFinite(trace)) { return Status::non_finite; }
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
    detail::SolvePivoted(lu + batch * ni_i * ni_i,
                         pivots + batch * ni_i, ni_i, local, 1);
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
  if (!PackedFinite(host_action, batch_count_ * nt))
  {
    return Status::non_finite;
  }
  CopyOut(host_action, batch_count_, nt, 1, result);
  return Status::success;
}

Status DeviceCoreEngine::Recover(const StridedBlockView<const double> trace,
                                 const StridedBlockView<double> interior) noexcept
{
  ScopedRegion recovery_region(
      dg_igr::observability::RegionId::hdg_interior_recovery);
  if (!ready_) { return Status::not_ready; }
  if (path_ != ExecutionPath::mfem_device)
  {
    const Status reference_status = reference_.Recover(trace, interior);
    if (reference_status == Status::success) { ++accounting_.host_reference_fallbacks; }
    return reference_status;
  }
  const std::size_t ni = static_cast<std::size_t>(shape_.InteriorUnknowns());
  const std::size_t nt = static_cast<std::size_t>(shape_.TraceUnknowns());
  if (!ValidVectorView(trace, batch_count_, nt) ||
      !ValidVectorView(interior, batch_count_, ni))
  {
    return Status::invalid_view;
  }
  if (!HasDistinctAddresses(trace) || !HasDistinctAddresses(interior) ||
      Overlaps(trace, interior))
  {
    return Status::unsupported_alias;
  }
  if (!AllFinite(trace)) { return Status::non_finite; }
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
    detail::SolvePivoted(lu + batch * ni_i * ni_i,
                         pivots + batch * ni_i, ni_i, local, 1);
  });
  const double* host_interior = interior_.HostRead();
  RecordDeviceToHost(batch_count_ * ni * sizeof(double));
  if (!PackedFinite(host_interior, batch_count_ * ni))
  {
    return Status::non_finite;
  }
  CopyOut(host_interior, batch_count_, ni, 1, interior);
  return Status::success;
}
}  // namespace dg::hdg::device

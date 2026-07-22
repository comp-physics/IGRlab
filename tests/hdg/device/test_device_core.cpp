#include "numerics/hdg/device/device_core.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace
{
using dg::hdg::LocalIndex;
using dg::hdg::StridedBlockView;
using dg::hdg::core::CondensedBatchView;
using dg::hdg::core::LocalSystemBatchView;
using dg::hdg::core::Status;
using dg::hdg::device::DeviceCoreEngine;
using dg::hdg::device::ExecutionPath;
using dg::hdg::device::BackendSelection;

template <typename T>
StridedBlockView<T> View(T* data, const LocalIndex batches,
                         const LocalIndex rows, const LocalIndex columns)
{
  return {data, batches, rows, columns, rows * columns, columns, 1};
}

StridedBlockView<const double> ConstView(double* data, const LocalIndex batches,
                                         const LocalIndex rows,
                                         const LocalIndex columns)
{
  return dg::hdg::AsConst(View(data, batches, rows, columns));
}

template <typename T>
StridedBlockView<T> StridedView(T* data, const LocalIndex batches,
                                const LocalIndex rows,
                                const LocalIndex columns,
                                const std::ptrdiff_t batch_stride,
                                const std::ptrdiff_t row_stride,
                                const std::ptrdiff_t column_stride = 1)
{
  return {data, batches, rows, columns, batch_stride, row_stride,
          column_stride};
}

bool Near(const double left, const double right)
{
  return std::abs(left - right) < 2.0e-12;
}

int TestReferenceFallbackParity()
{
  constexpr LocalIndex batches = 2;
  constexpr LocalIndex ni = 2;
  constexpr LocalIndex nt = 2;
  std::array<double, 8> a{{4., 1., 2., 3., 3., -1., 1., 2.}};
  std::array<double, 8> b{{1., 0., 0., 1., 2., 1., 1., -1.}};
  std::array<double, 8> c{{2., -1., 1., 2., 1., 3., -2., 1.}};
  std::array<double, 8> d{{5., .5, .5, 4., 6., 1., 1., 3.}};
  std::array<double, 4> f{{1., 2., 4., -1.}};
  std::array<double, 4> g{{3., 4., 2., 5.}};
  std::array<double, 8> schur{};
  std::array<double, 4> rhs{};
  const LocalSystemBatchView input{
      {ni, nt, 1, 1}, ConstView(a.data(), batches, ni, ni),
      ConstView(b.data(), batches, ni, nt), ConstView(c.data(), batches, nt, ni),
      ConstView(d.data(), batches, nt, nt), ConstView(f.data(), batches, ni, 1),
      ConstView(g.data(), batches, nt, 1)};
  DeviceCoreEngine engine;
  if (engine.Condense(input, {View(schur.data(), batches, nt, nt),
                              View(rhs.data(), batches, nt, 1)}) != Status::success ||
      !engine.ready())
  {
    return 1;
  }
  std::array<double, 8> expected_s{{4.2, 1.1, .6, 3.3,
                                     34. / 7., 18. / 7., 16. / 7., 27. / 7.}};
  std::array<double, 4> expected_rhs{{3.4, 2.7, 4., 8.}};
  for (std::size_t entry = 0; entry < schur.size(); ++entry)
  {
    if (!Near(schur[entry], expected_s[entry])) { return 2; }
  }
  for (std::size_t entry = 0; entry < rhs.size(); ++entry)
  {
    if (!Near(rhs[entry], expected_rhs[entry])) { return 3; }
  }
  std::array<double, 4> trace{{.25, -.75, 1.5, .5}};
  std::array<double, 4> action{};
  std::array<double, 4> interior{};
  if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                        View(action.data(), batches, nt, 1)) != Status::success ||
      engine.Recover(ConstView(trace.data(), batches, nt, 1),
                     View(interior.data(), batches, ni, 1)) != Status::success)
  {
    return 4;
  }
  for (LocalIndex batch = 0; batch < batches; ++batch)
    for (LocalIndex row = 0; row < nt; ++row)
    {
      double expected = 0.0;
      for (LocalIndex column = 0; column < nt; ++column)
      {
        expected += schur[batch * nt * nt + row * nt + column] *
                    trace[batch * nt + column];
      }
      if (!Near(action[batch * nt + row], expected)) { return 5; }
    }
  const auto& accounting = engine.accounting();
#ifdef MFEM_USE_CUDA
  if (engine.execution_path() != ExecutionPath::mfem_device ||
      !engine.using_device() || accounting.persistent_bytes == 0 ||
      accounting.allocation_events == 0 || accounting.host_to_device_bytes == 0 ||
      accounting.device_to_host_bytes == 0)
  {
    return 6;
  }
#else
  if (engine.execution_path() != ExecutionPath::host_reference ||
      accounting.host_reference_fallbacks < 3)
  {
    return 6;
  }
#endif
  const std::size_t allocations = accounting.allocation_events;
  for (int repeat = 0; repeat < 3; ++repeat)
  {
    if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                          View(action.data(), batches, nt, 1)) != Status::success ||
        engine.Recover(ConstView(trace.data(), batches, nt, 1),
                       View(interior.data(), batches, ni, 1)) != Status::success ||
        engine.accounting().allocation_events != allocations ||
        engine.accounting().steady_state_allocation_events != 0)
    {
      return 7;
    }
  }
  return 0;
}

int TestBackendSelectionPolicy()
{
  using dg::hdg::device::IsGpuExecutionEligible;
  if (!IsGpuExecutionEligible({true, true, false, false}) ||
      !IsGpuExecutionEligible({true, false, true, false}) ||
      IsGpuExecutionEligible({true, false, false, true}) ||
      IsGpuExecutionEligible({true, true, false, true}) ||
      IsGpuExecutionEligible({false, true, false, false}))
  {
    return 10;
  }
  // MFEM owns selection by device id; visibility cardinality is intentionally
  // absent from the pure policy, so one selected GPU remains eligible even
  // when a process can see additional GPUs.
  const BackendSelection selected_cuda{true, true, false, false, 8};
  if (!IsGpuExecutionEligible(selected_cuda)) { return 11; }
  return 0;
}

int TestPivotPaddedAndFailures()
{
  constexpr LocalIndex one = 1;
  constexpr LocalIndex ni = 2;
  constexpr LocalIndex nt = 2;
  std::array<double, 8> a{{0., 2., 99., 3., 4., 98., 97., 96.}};
  std::array<double, 8> b{{1., 0., 95., 0., 1., 94., 93., 92.}};
  std::array<double, 8> c{{1., 0., 91., 0., 1., 90., 89., 88.}};
  std::array<double, 8> d{{0., 0., 87., 0., 0., 86., 85., 84.}};
  std::array<double, 3> f{{0., 0., 83.}};
  std::array<double, 3> g{{0., 0., 82.}};
  std::array<double, 8> schur{{-7., -7., 77., -7., -7., 76., 75., 74.}};
  std::array<double, 3> rhs{{-7., -7., 73.}};
  const LocalSystemBatchView input{
      {ni, nt, 1, 1},
      dg::hdg::AsConst(StridedView(a.data(), one, ni, ni, 8, 3)),
      dg::hdg::AsConst(StridedView(b.data(), one, ni, nt, 8, 3)),
      dg::hdg::AsConst(StridedView(c.data(), one, nt, ni, 8, 3)),
      dg::hdg::AsConst(StridedView(d.data(), one, nt, nt, 8, 3)),
      dg::hdg::AsConst(StridedView(f.data(), one, ni, 1, 3, 1)),
      dg::hdg::AsConst(StridedView(g.data(), one, nt, 1, 3, 1))};
  DeviceCoreEngine engine;
  const CondensedBatchView output{
      StridedView(schur.data(), one, nt, nt, 8, 3),
      StridedView(rhs.data(), one, nt, 1, 3, 1)};
  if (engine.Condense(input, output) != Status::success ||
      !Near(schur[0], 2. / 3.) || !Near(schur[1], -1. / 3.) ||
      !Near(schur[3], -0.5) || !Near(schur[4], 0.0) ||
      !Near(rhs[0], 0.0) || !Near(rhs[1], 0.0))
  {
    return 12;
  }
  const auto successful_schur = schur;
  const auto successful_rhs = rhs;

  std::array<double, 4> overlap{};
  const CondensedBatchView aliased{View(overlap.data(), one, nt, nt),
                                   View(overlap.data(), one, nt, 1)};
  if (engine.Condense(input, aliased) != Status::unsupported_alias) { return 13; }
  LocalSystemBatchView invalid = input;
  invalid.a.data = nullptr;
  if (engine.Condense(invalid, output) != Status::invalid_view) { return 14; }
  LocalSystemBatchView singular = input;
  std::array<double, 8> zero_a{};
  singular.a = dg::hdg::AsConst(StridedView(zero_a.data(), one, ni, ni, 8, 3));
  if (engine.Condense(singular, output) != Status::singular) { return 15; }
  LocalSystemBatchView non_finite = input;
  std::array<double, 8> inf_a = a;
  inf_a[0] = std::numeric_limits<double>::infinity();
  non_finite.a = dg::hdg::AsConst(StridedView(inf_a.data(), one, ni, ni, 8, 3));
  if (engine.Condense(non_finite, output) != Status::non_finite) { return 16; }
  if (schur != successful_schur || rhs != successful_rhs) { return 17; }

  dg::hdg::device::PersistentStorageLayout layout{};
  layout.bytes = 17;
  if (dg::hdg::device::MakePersistentStorageLayout(
          std::numeric_limits<std::size_t>::max(), 2, 1, layout) ||
      layout.bytes != 17)
  {
    return 18;
  }
  return 0;
}
}  // namespace

int main()
{
#ifdef MFEM_USE_CUDA
  // The CMake resource lock serializes this process on the selected GPU.
  mfem::Device device("cuda");
  (void)device;
#endif
  const int policy = TestBackendSelectionPolicy();
  if (policy != 0) { return policy; }
  const int parity = TestReferenceFallbackParity();
  if (parity != 0) { return parity; }
  return TestPivotPaddedAndFailures();
}

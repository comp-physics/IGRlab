#include "numerics/hdg/device/device_core.hpp"

#include "../../../src/numerics/hdg/device/pivoted_solve.hpp"

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
using dg::hdg::device::BackendSelection;
using dg::hdg::device::DeviceCoreEngine;
using dg::hdg::device::ExecutionPath;

#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
constexpr bool expect_actual_device = true;
#else
constexpr bool expect_actual_device = false;
#endif

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

template <std::size_t Size>
bool Matches(const std::array<double, Size>& actual,
             const std::array<double, Size>& expected)
{
  for (std::size_t entry = 0; entry < Size; ++entry)
    if (!Near(actual[entry], expected[entry])) { return false; }
  return true;
}

bool ExpectedExecution(const DeviceCoreEngine& engine,
                       const std::size_t allocations) noexcept
{
  if constexpr (expect_actual_device)
  {
    return engine.ready() && engine.execution_path() == ExecutionPath::mfem_device &&
           engine.using_device() && engine.accounting().host_reference_fallbacks == 0 &&
           engine.accounting().allocation_events == allocations;
  }
  return engine.ready() && engine.execution_path() == ExecutionPath::host_reference &&
         !engine.using_device() && engine.accounting().allocation_events == allocations;
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
  // MFEM selects a device id.  Visibility of eight GPUs must not turn the
  // selected CUDA backend into a host fallback.
  const BackendSelection selected_cuda{true, true, false, false, 8};
  if (!IsGpuExecutionEligible(selected_cuda)) { return 11; }
  return 0;
}

int TestCondenseActionRecoverParityAndContracts()
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
                              View(rhs.data(), batches, nt, 1)}) != Status::success)
  {
    return 20;
  }
  const std::array<double, 8> expected_s{{4.2, 1.1, .6, 3.3,
                                            34. / 7., 18. / 7., 16. / 7., 27. / 7.}};
  const std::array<double, 4> expected_rhs{{3.4, 2.7, 4., 8.}};
  if (!Matches(schur, expected_s) || !Matches(rhs, expected_rhs)) { return 21; }

  std::array<double, 4> trace{{.25, -.75, 1.5, .5}};
  const std::array<double, 4> expected_action{{.225, -2.325, 60. / 7., 75. / 14.}};
  const std::array<double, 4> expected_interior{{-.05, .95, -1. / 7., -13. / 14.}};
  std::array<double, 4> action{};
  std::array<double, 4> interior{};
  if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                        View(action.data(), batches, nt, 1)) != Status::success ||
      engine.Recover(ConstView(trace.data(), batches, nt, 1),
                     View(interior.data(), batches, ni, 1)) != Status::success ||
      !Matches(action, expected_action) || !Matches(interior, expected_interior))
  {
    return 22;
  }
  const auto& accounting = engine.accounting();
  if constexpr (expect_actual_device)
  {
    if (!ExpectedExecution(engine, accounting.allocation_events) ||
        accounting.persistent_bytes == 0 || accounting.allocation_events == 0 ||
        accounting.host_to_device_bytes == 0 || accounting.device_to_host_bytes == 0)
    {
      return 23;
    }
  }
  else if (!ExpectedExecution(engine, accounting.allocation_events) ||
           accounting.host_reference_fallbacks < 3)
  {
    return 23;
  }

  const std::size_t allocations = accounting.allocation_events;
  for (int repeat = 0; repeat < 3; ++repeat)
  {
    action.fill(-31.0);
    interior.fill(-37.0);
    if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                          View(action.data(), batches, nt, 1)) != Status::success ||
        engine.Recover(ConstView(trace.data(), batches, nt, 1),
                       View(interior.data(), batches, ni, 1)) != Status::success ||
        !Matches(action, expected_action) || !Matches(interior, expected_interior) ||
        !ExpectedExecution(engine, allocations) ||
        engine.accounting().steady_state_allocation_events != 0)
    {
      return 24;
    }
  }

  if constexpr (!expect_actual_device) { return 0; }

  // Device-only negative contracts: each returned failure must preserve the
  // caller's output and the established device engine state.
  const auto stable = [&engine, allocations]()
  { return ExpectedExecution(engine, allocations) &&
           engine.accounting().steady_state_allocation_events == 0; };
  action.fill(41.0);
  const auto saved_action = action;
  if (engine.ApplySchur(ConstView(trace.data(), batches, 1, 1),
                        View(action.data(), batches, nt, 1)) != Status::invalid_view ||
      action != saved_action || !stable()) { return 25; }
  if (engine.ApplySchur(dg::hdg::AsConst(
                            StridedView(trace.data(), batches, nt, 1, 1, 1)),
                        View(action.data(), batches, nt, 1)) != Status::unsupported_alias ||
      action != saved_action || !stable()) { return 26; }
  if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                        View(action.data(), batches, 1, 1)) != Status::invalid_view ||
      action != saved_action || !stable()) { return 27; }
  if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                        StridedView(action.data(), batches, nt, 1, 0, 1)) !=
          Status::invalid_view ||
      action != saved_action || !stable()) { return 28; }
  if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                        StridedView(action.data(), batches, nt, 1, 1, 1)) !=
          Status::unsupported_alias ||
      action != saved_action || !stable()) { return 29; }
  if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                        View(trace.data(), batches, nt, 1)) != Status::unsupported_alias ||
      action != saved_action || !stable()) { return 30; }
  std::array<double, 4> bad_trace = trace;
  bad_trace[1] = std::numeric_limits<double>::infinity();
  if (engine.ApplySchur(ConstView(bad_trace.data(), batches, nt, 1),
                        View(action.data(), batches, nt, 1)) != Status::non_finite ||
      action != saved_action || !stable()) { return 31; }

  interior.fill(43.0);
  const auto saved_interior = interior;
  if (engine.Recover(ConstView(trace.data(), batches, 1, 1),
                     View(interior.data(), batches, ni, 1)) != Status::invalid_view ||
      interior != saved_interior || !stable()) { return 32; }
  if (engine.Recover(dg::hdg::AsConst(
                         StridedView(trace.data(), batches, nt, 1, 1, 1)),
                     View(interior.data(), batches, ni, 1)) != Status::unsupported_alias ||
      interior != saved_interior || !stable()) { return 33; }
  if (engine.Recover(ConstView(trace.data(), batches, nt, 1),
                     View(interior.data(), batches, 1, 1)) != Status::invalid_view ||
      interior != saved_interior || !stable()) { return 34; }
  if (engine.Recover(ConstView(trace.data(), batches, nt, 1),
                     StridedView(interior.data(), batches, ni, 1, 0, 1)) !=
          Status::invalid_view ||
      interior != saved_interior || !stable()) { return 35; }
  if (engine.Recover(ConstView(trace.data(), batches, nt, 1),
                     StridedView(interior.data(), batches, ni, 1, 1, 1)) !=
          Status::unsupported_alias ||
      interior != saved_interior || !stable()) { return 36; }
  if (engine.Recover(ConstView(trace.data(), batches, nt, 1),
                     View(trace.data(), batches, ni, 1)) != Status::unsupported_alias ||
      interior != saved_interior || !stable()) { return 37; }
  if (engine.Recover(ConstView(bad_trace.data(), batches, nt, 1),
                     View(interior.data(), batches, ni, 1)) != Status::non_finite ||
      interior != saved_interior || !stable()) { return 38; }
  if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                        View(action.data(), batches, nt, 1)) != Status::success ||
      engine.Recover(ConstView(trace.data(), batches, nt, 1),
                     View(interior.data(), batches, ni, 1)) != Status::success ||
      !Matches(action, expected_action) || !Matches(interior, expected_interior) ||
      !stable())
  {
    return 39;
  }
  return 0;
}

int TestPaddedTwoBatchPivotAndFailures()
{
  constexpr LocalIndex batches = 2;
  constexpr LocalIndex ni = 2;
  constexpr LocalIndex nt = 2;
  std::array<double, 16> a{}, b{}, c{}, d{}, f{}, g{}, schur{}, rhs{};
  a.fill(91.0); b.fill(92.0); c.fill(93.0); d.fill(94.0);
  f.fill(95.0); g.fill(96.0); schur.fill(-7.0); rhs.fill(-8.0);
  // Each matrix has row stride 3 and batch stride 8.  Both A blocks require
  // pivot swaps, so logical batch offsets cannot be hidden by a contiguous
  // single-batch path.
  a[0] = 0.; a[1] = 2.; a[3] = 3.; a[4] = 4.;
  a[8] = 1.; a[9] = 2.; a[11] = 4.; a[12] = 1.;
  for (const std::size_t base : {std::size_t{0}, std::size_t{8}})
  {
    b[base] = 1.; b[base + 1] = 0.; b[base + 3] = 0.; b[base + 4] = 1.;
    c[base] = 1.; c[base + 1] = 0.; c[base + 3] = 0.; c[base + 4] = 1.;
    d[base] = 0.; d[base + 1] = 0.; d[base + 3] = 0.; d[base + 4] = 0.;
    f[base] = 0.; f[base + 3] = 0.; g[base] = 0.; g[base + 3] = 0.;
  }
  const LocalSystemBatchView input{
      {ni, nt, 1, 1},
      dg::hdg::AsConst(StridedView(a.data(), batches, ni, ni, 8, 3)),
      dg::hdg::AsConst(StridedView(b.data(), batches, ni, nt, 8, 3)),
      dg::hdg::AsConst(StridedView(c.data(), batches, nt, ni, 8, 3)),
      dg::hdg::AsConst(StridedView(d.data(), batches, nt, nt, 8, 3)),
      dg::hdg::AsConst(StridedView(f.data(), batches, ni, 1, 8, 3)),
      dg::hdg::AsConst(StridedView(g.data(), batches, nt, 1, 8, 3))};
  DeviceCoreEngine engine;
  const CondensedBatchView output{
      StridedView(schur.data(), batches, nt, nt, 8, 3),
      StridedView(rhs.data(), batches, nt, 1, 8, 3)};
  if (engine.Condense(input, output) != Status::success ||
      !Near(schur[0], 2. / 3.) || !Near(schur[1], -1. / 3.) ||
      !Near(schur[3], -0.5) || !Near(schur[4], 0.0) ||
      !Near(schur[8], 1. / 7.) || !Near(schur[9], -2. / 7.) ||
      !Near(schur[11], -4. / 7.) || !Near(schur[12], 1. / 7.) ||
      !Near(rhs[0], 0.0) || !Near(rhs[3], 0.0) ||
      !Near(rhs[8], 0.0) || !Near(rhs[11], 0.0))
  {
    return 40;
  }
  if constexpr (expect_actual_device)
  {
    if (engine.execution_path() != ExecutionPath::mfem_device ||
        !engine.using_device() || engine.accounting().host_reference_fallbacks != 0)
    {
      return 41;
    }
  }
  else if (engine.execution_path() != ExecutionPath::host_reference)
  {
    return 41;
  }

  const auto successful_schur = schur;
  const auto successful_rhs = rhs;
  std::array<double, 8> overlap{};
  const CondensedBatchView aliased{View(overlap.data(), batches, nt, nt),
                                   View(overlap.data(), batches, nt, 1)};
  if (engine.Condense(input, aliased) != Status::unsupported_alias) { return 42; }
  LocalSystemBatchView invalid = input;
  invalid.a.data = nullptr;
  if (engine.Condense(invalid, output) != Status::invalid_view) { return 43; }
  LocalSystemBatchView singular = input;
  std::array<double, 16> zero_a{};
  singular.a = dg::hdg::AsConst(StridedView(zero_a.data(), batches, ni, ni, 8, 3));
  if (engine.Condense(singular, output) != Status::singular) { return 44; }
  LocalSystemBatchView non_finite = input;
  std::array<double, 16> inf_a = a;
  inf_a[0] = std::numeric_limits<double>::infinity();
  non_finite.a = dg::hdg::AsConst(StridedView(inf_a.data(), batches, ni, ni, 8, 3));
  if (engine.Condense(non_finite, output) != Status::non_finite) { return 45; }
  if (schur != successful_schur || rhs != successful_rhs) { return 46; }

  dg::hdg::device::PersistentStorageLayout layout{};
  layout.bytes = 17;
  if (dg::hdg::device::MakePersistentStorageLayout(
          std::numeric_limits<std::size_t>::max(), 2, 1, layout) ||
      layout.bytes != 17)
  {
    return 47;
  }
  return 0;
}

int TestDelayedPivotSharedSolveAndProductionPath()
{
  constexpr LocalIndex batches = 1;
  constexpr LocalIndex ni = 3;
  constexpr LocalIndex nt = 2;

  // Exact retained factorization of A.  Column zero is not swapped; column
  // one swaps rows with distinct retained column-zero multipliers.
  std::array<double, 9> a{{4., 2., 0., 1., 1., 5., 2., 3., 1.}};
  const std::array<double, 9> lu{{4., 2., 0., .5, 2., 1., .25, .25, 19. / 4.}};
  const std::array<double, 3> pivots{{0., 2., 2.}};
  std::array<double, 6> b{{1., -2., 3., .5, -1., 4.}};
  std::array<double, 3> f{{7., -2., 5.}};
  std::array<double, 6> inverse_b = b;
  std::array<double, 3> inverse_f = f;
  const std::array<double, 6> expected_inverse_b{{
      15. / 19., -67. / 38., -41. / 38., 48. / 19., 25. / 38., -1. / 19.}};
  const std::array<double, 3> expected_inverse_f{{
      22. / 19., 45. / 38., -33. / 38.}};
  dg::hdg::device::detail::SolvePivoted(lu.data(), pivots.data(), ni,
                                        inverse_b.data(), nt);
  dg::hdg::device::detail::SolvePivoted(lu.data(), pivots.data(), ni,
                                        inverse_f.data(), 1);
  if (!Matches(inverse_b, expected_inverse_b) ||
      !Matches(inverse_f, expected_inverse_f))
  {
    return 50;
  }

  std::array<double, 6> c{{2., -1., .5, -3., 2., 1.}};
  std::array<double, 4> d{{6., .25, -.75, 5.}};
  std::array<double, 2> g{{1.5, -4.}};
  std::array<double, 4> schur{};
  std::array<double, 2> condensed_rhs{};
  const LocalSystemBatchView input{
      {ni, nt, 1, 1}, ConstView(a.data(), batches, ni, ni),
      ConstView(b.data(), batches, ni, nt), ConstView(c.data(), batches, nt, ni),
      ConstView(d.data(), batches, nt, nt), ConstView(f.data(), batches, ni, 1),
      ConstView(g.data(), batches, nt, 1)};
  DeviceCoreEngine engine;
  if (engine.Condense(input, {View(schur.data(), batches, nt, nt),
                              View(condensed_rhs.data(), batches, nt, 1)}) !=
      Status::success)
  {
    return 51;
  }
  const std::array<double, 4> expected_schur{{
      229. / 76., 481. / 76., 237. / 76., -201. / 38.}};
  const std::array<double, 2> expected_condensed_rhs{{61. / 76., -77. / 38.}};
  if (!Matches(schur, expected_schur) ||
      !Matches(condensed_rhs, expected_condensed_rhs))
  {
    return 52;
  }

  const std::size_t allocations = engine.accounting().allocation_events;
  if (!ExpectedExecution(engine, allocations)) { return 53; }
  if constexpr (expect_actual_device)
  {
    if (allocations == 0 || engine.accounting().host_reference_fallbacks != 0)
    {
      return 54;
    }
  }

  std::array<double, 2> trace{{.75, -.5}};
  std::array<double, 2> action{};
  std::array<double, 3> recovered{};
  if (engine.ApplySchur(ConstView(trace.data(), batches, nt, 1),
                        View(action.data(), batches, nt, 1)) != Status::success ||
      engine.Recover(ConstView(trace.data(), batches, nt, 1),
                     View(recovered.data(), batches, ni, 1)) != Status::success)
  {
    return 55;
  }
  const std::array<double, 2> expected_action{{-275. / 304., 1515. / 304.}};
  const std::array<double, 3> expected_recovered{{
      -6. / 19., 495. / 152., -211. / 152.}};
  const std::array<double, 3> expected_recovery_rhs{{21. / 4., -4., 31. / 4.}};
  if (!Matches(action, expected_action) ||
      !Matches(recovered, expected_recovered) ||
      !ExpectedExecution(engine, allocations) ||
      engine.accounting().steady_state_allocation_events != 0)
  {
    return 56;
  }
  if constexpr (expect_actual_device)
  {
    if (engine.accounting().host_reference_fallbacks != 0) { return 57; }
  }

  for (LocalIndex row = 0; row < ni; ++row)
  {
    double recovery_rhs = f[row];
    for (LocalIndex column = 0; column < nt; ++column)
    {
      recovery_rhs -= b[row * nt + column] * trace[column];
    }
    if (!Near(recovery_rhs, expected_recovery_rhs[row])) { return 58; }
    double product = 0.0;
    for (LocalIndex column = 0; column < ni; ++column)
    {
      product += a[row * ni + column] * recovered[column];
    }
    if (!Near(product, expected_recovery_rhs[row])) { return 59; }
  }
  return 0;
}
}  // namespace

int main()
{
#ifdef MFEM_USE_CUDA
  // CUDA is runtime-validated by this test on the SM120 lock.  HIP is only
  // structurally gated below on this CUDA host and is not claimed validated.
  mfem::Device device("cuda");
  (void)device;
#elif defined(MFEM_USE_HIP)
  mfem::Device device("hip");
  (void)device;
#endif
  const int policy = TestBackendSelectionPolicy();
  if (policy != 0) { return policy; }
  const int parity = TestCondenseActionRecoverParityAndContracts();
  if (parity != 0) { return parity; }
  const int padded = TestPaddedTwoBatchPivotAndFailures();
  if (padded != 0) { return padded; }
  return TestDelayedPivotSharedSolveAndProductionPath();
}

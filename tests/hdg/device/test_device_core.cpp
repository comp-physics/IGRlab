#include "numerics/hdg/device/device_core.hpp"

#include <array>
#include <cmath>

namespace
{
using dg::hdg::LocalIndex;
using dg::hdg::StridedBlockView;
using dg::hdg::core::CondensedBatchView;
using dg::hdg::core::LocalSystemBatchView;
using dg::hdg::core::Status;
using dg::hdg::device::DeviceCoreEngine;
using dg::hdg::device::ExecutionPath;

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

bool Near(const double left, const double right)
{
  return std::abs(left - right) < 2.0e-12;
}

int TestReferenceFallbackParity()
{
#ifdef MFEM_USE_CUDA
  // This test is registered only under the GPU resource lock when the
  // selected MFEM was built with CUDA.  CPU builds remain pure fallback tests.
  mfem::Device device("cuda");
#endif
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
  return 0;
}
}  // namespace

int main()
{
  return TestReferenceFallbackParity();
}

#include "generic_euler_convergence_fixture.hpp"

#include "numerics/hdg/runtime/cpu/generic_euler.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace
{

using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerHdgCpuSession;
using dg::hdg::test::generic_euler_convergence::CollectiveAll;

bool AdvanceCpu(const EulerHdgRequest& request, const int steps,
                const double dt, mfem::ParGridFunction& committed)
{
  GenericEulerHdgCpuSession session;
  if (session.Initialize(request).status !=
      GenericEulerCpuSessionStatus::initialized)
  {
    return false;
  }
  for (int step = 0; step < steps; ++step)
  {
    if (session.Advance(dt).status != GenericEulerCpuSessionStatus::advanced)
      return false;
  }
  const auto coordinates = session.coordinates();
  const auto diagnostics = session.diagnostics();
  const double expected_time = static_cast<double>(steps) * dt;
  const double time_scale = std::max(1.0, std::abs(expected_time));
  const bool local = session.ExportCommittedInterior(committed) &&
      coordinates.step == static_cast<std::uint64_t>(steps) &&
      coordinates.generation == static_cast<std::uint64_t>(steps) &&
      std::abs(coordinates.time - expected_time) <=
          64.0 * std::numeric_limits<double>::epsilon() * time_scale &&
      diagnostics.available &&
      diagnostics.wave_generation == coordinates.generation &&
      diagnostics.coordinates.step == coordinates.step &&
      std::isfinite(diagnostics.maximum_wave_speed) &&
      diagnostics.maximum_wave_speed > 0.0;
  return CollectiveAll(local, request.communicator);
}

struct CpuRunner
{
  bool Begin(
      const dg::hdg::test::generic_euler_convergence::ExecutionSelection
          execution) const noexcept
  {
    return execution.backend == dg::hdg::generic::BackendKind::cpu;
  }

  bool Advance(const EulerHdgRequest& request, const int steps,
               const double dt, mfem::ParGridFunction& committed) const
  {
    return AdvanceCpu(request, steps, dt, committed);
  }
};

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dg::hdg::test::generic_euler_convergence::CaseSelection test_case{};
  const bool parsed =
      dg::hdg::test::generic_euler_convergence::ParseCaseSelection(
          argc, argv, test_case);
  const std::array<
      dg::hdg::test::generic_euler_convergence::ExecutionSelection, 1>
      executions{{
          {dg::hdg::generic::BackendKind::cpu,
           dg::hdg::generic::CudaEngineKind::ordinary, "CPU"}}};
  CpuRunner runner;
  const bool passed = parsed &&
      dg::hdg::test::generic_euler_convergence::RunComparisonCase(
          MPI_COMM_WORLD, test_case, executions, runner);
  MPI_Finalize();
  return passed ? 0 : 1;
}

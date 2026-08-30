#include "generic_euler_convergence_fixture.hpp"

#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace dg::hdg::runtime::cpu::detail
{

struct CpuQualificationTestAccess
{
  static bool AuditLastConservation(
      const CpuNumericsCore& core, double& maximum_error,
      double& maximum_relative_error, bool& gate) noexcept
  {
    return core.AuditLastBorrowedEulerConservation(
        maximum_error, maximum_relative_error, gate);
  }
};

} // namespace dg::hdg::runtime::cpu::detail

namespace
{

using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerHdgCpuSession;
using dg::hdg::test::generic_euler_convergence::CollectiveAll;
using dg::hdg::test::generic_euler_convergence::QuadratureTrajectoryHealth;

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

bool AdvanceCpuQualified(
    const EulerHdgRequest& request, const int steps, const double dt,
    mfem::ParGridFunction& committed, QuadratureTrajectoryHealth& health)
{
  health = {};
  dg::hdg::runtime::cpu::detail::CpuNumericsCore core;
  if (steps <= 0 || !(dt > 0.0) || !std::isfinite(dt) ||
      !core.InitializeBorrowedEuler(request))
  {
    return false;
  }

  dg::hdg::runtime::StateCoordinates coordinates{};
  bool valid = true;
  for (int step = 0; step < steps; ++step)
  {
    std::uint64_t generation = 0;
    double maximum_wave_speed = 0.0;
    double backend_stable_dt = 0.0;
    if (!core.PrepareBorrowedEuler(dt, coordinates, generation) ||
        generation == 0 ||
        !core.CommitBorrowedEuler(generation, coordinates,
                                  maximum_wave_speed, backend_stable_dt))
    {
      valid = false;
      break;
    }

    double conservation_error = std::numeric_limits<double>::infinity();
    double conservation_relative_error =
        std::numeric_limits<double>::infinity();
    bool conservation_gate = false;
    const bool conservation_audited =
        dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
            AuditLastConservation(
                core, conservation_error, conservation_relative_error,
                conservation_gate);
    const auto& attempt = core.evidence().last_attempt;
    const bool linear_accounting = attempt.linear_iterations >= 0 &&
        attempt.linear_solve_attempts >= 0 &&
        attempt.newton_correction_linear_iterations >= 0 &&
        attempt.final_coherent_linear_iterations >= 0 &&
        attempt.true_residual_tightenings >= 0 &&
        attempt.linear_iterations ==
            attempt.newton_correction_linear_iterations +
                attempt.final_coherent_linear_iterations &&
        attempt.linear_solve_attempts ==
            static_cast<int>(attempt.work.complete_trace_solves) +
                attempt.true_residual_tightenings;
    const bool lifecycle_work =
        attempt.work.complete_factor_setups ==
            attempt.work.complete_trace_solves &&
        attempt.work.complete_trace_solves ==
            attempt.work.complete_recoveries &&
        attempt.work.complete_trace_solves ==
            attempt.work.accepted_newton_updates + 1;
    const bool step_valid = conservation_audited && conservation_gate &&
        attempt.status ==
            dg::hdg::runtime::cpu::CpuAttemptStatus::success &&
        attempt.committed && attempt.finite_gate &&
        attempt.admissibility_gate && attempt.recovery_gate &&
        attempt.nonlinear_gate && attempt.true_residual_gate &&
        linear_accounting && lifecycle_work &&
        attempt.nonlinear_updates >= 0 &&
        attempt.final_line_search_trials >= 0 &&
        std::isfinite(attempt.initial_nonlinear_residual) &&
        std::isfinite(attempt.final_nonlinear_residual) &&
        attempt.final_nonlinear_residual <=
            attempt.initial_nonlinear_residual &&
        std::isfinite(attempt.maximum_true_relative_residual) &&
        attempt.maximum_true_relative_residual <=
            request.numerical_policy.true_residual_relative_tolerance &&
        std::isfinite(attempt.minimum_density) &&
        attempt.minimum_density > 0.0 &&
        std::isfinite(attempt.minimum_pressure) &&
        attempt.minimum_pressure > 0.0 &&
        std::isfinite(conservation_error) && conservation_error >= 0.0 &&
        std::isfinite(conservation_relative_error) &&
        conservation_relative_error >= 0.0 &&
        conservation_relative_error <=
            request.numerical_policy.conservation_relative_tolerance &&
        std::isfinite(maximum_wave_speed) && maximum_wave_speed > 0.0 &&
        std::isfinite(backend_stable_dt) && backend_stable_dt > 0.0;
    valid = valid && step_valid;

    health.accepted_steps += 1;
    health.nonlinear_updates += attempt.nonlinear_updates;
    health.linear_iterations += attempt.linear_iterations;
    health.linear_solve_attempts += attempt.linear_solve_attempts;
    health.line_search_trials += attempt.final_line_search_trials;
    health.true_residual_tightenings +=
        attempt.true_residual_tightenings;
    health.trace_solves += static_cast<std::int64_t>(
        attempt.work.complete_trace_solves);
    health.maximum_true_relative_residual = std::max(
        health.maximum_true_relative_residual,
        attempt.maximum_true_relative_residual);
    health.maximum_conservation_error = std::max(
        health.maximum_conservation_error, conservation_error);
    health.maximum_conservation_relative_error = std::max(
        health.maximum_conservation_relative_error,
        conservation_relative_error);
    health.maximum_final_nonlinear_residual = std::max(
        health.maximum_final_nonlinear_residual,
        attempt.final_nonlinear_residual);
    health.minimum_density =
        std::min(health.minimum_density, attempt.minimum_density);
    health.minimum_pressure =
        std::min(health.minimum_pressure, attempt.minimum_pressure);
    health.maximum_wave_speed =
        std::max(health.maximum_wave_speed, maximum_wave_speed);
  }

  const double expected_time = static_cast<double>(steps) * dt;
  const double time_scale = std::max(1.0, std::abs(expected_time));
  const bool exported = valid && core.ExportCommittedInterior(committed) &&
      coordinates.step == static_cast<std::uint64_t>(steps) &&
      coordinates.generation == static_cast<std::uint64_t>(steps) &&
      std::abs(coordinates.time - expected_time) <=
          64.0 * std::numeric_limits<double>::epsilon() * time_scale &&
      health.accepted_steps == steps;
  health.passed = exported;
  return CollectiveAll(exported, request.communicator);
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

  bool AdvanceQualified(
      const EulerHdgRequest& request, const int steps, const double dt,
      mfem::ParGridFunction& committed,
      QuadratureTrajectoryHealth& health) const
  {
    return AdvanceCpuQualified(request, steps, dt, committed, health);
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
  if (!parsed)
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      std::cerr << "Usage: " << argv[0]
                << " --order 1|2|3 --regularization conventional|igr"
                   " [--igr-volume-quadrature-order Q"
                   " --igr-face-quadrature-order Q]"
                   " [--newton-linear-rtol R]"
                   " [--final-linear-rtol R]"
                   " [--true-residual-tightening-linear-rtol R]"
                   " [--quadrature-equivalence]\n";
    MPI_Finalize();
    return 2;
  }
  const std::array<
      dg::hdg::test::generic_euler_convergence::ExecutionSelection, 1>
      executions{{
          {dg::hdg::generic::BackendKind::cpu,
           dg::hdg::generic::CudaEngineKind::ordinary, "CPU"}}};
  CpuRunner runner;
  const bool passed = test_case.quadrature_equivalence
           ? dg::hdg::test::generic_euler_convergence::
                 RunQuadratureEquivalenceCase(
                     MPI_COMM_WORLD, test_case, executions[0], runner)
           : dg::hdg::test::generic_euler_convergence::RunComparisonCase(
                 MPI_COMM_WORLD, test_case, executions, runner);
  MPI_Finalize();
  return passed ? 0 : 1;
}

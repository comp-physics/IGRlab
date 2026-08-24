#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"

#include <mfem.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>

namespace
{
using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::cuda_w8a;

constexpr bool TrueResidualAuthorityPolicy() noexcept
{
  dg::hdg::runtime::cuda_w8a::detail::CudaSolveDiagnostics diagnostics{};
  diagnostics.device_policy = true;
  diagnostics.configured_preconditioner = true;
  diagnostics.configured_ilu = true;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_ilu0;
  diagnostics.finite = true;
  diagnostics.iterations = 0;
  diagnostics.reported_residual = 1.1286157960703033e-12;
  diagnostics.reported_tolerance_met = false;
  const bool reported_tolerance_is_not_a_gate =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.ilu_accounting_kind = CudaIluAccountingKind::unavailable;
  diagnostics.ilu_factor_reuse_policy =
      CudaIluFactorReusePolicy::not_applicable;
  diagnostics.ilu_factor_setups =
      std::numeric_limits<std::uint64_t>::max();
  diagnostics.ilu_factor_reuses =
      std::numeric_limits<std::uint64_t>::max();
  diagnostics.ilu_preconditioner_callbacks_lower_bound = 9;
  diagnostics.ilu_preconditioner_callbacks_upper_bound = 1;
  diagnostics.ilu_inner_iterations_lower_bound = 9;
  diagnostics.ilu_inner_iterations_upper_bound = 1;
  diagnostics.ilu_jacobi_factor_spmv_updates_lower_bound = 72;
  diagnostics.ilu_jacobi_factor_spmv_updates_upper_bound = 8;
  diagnostics.ilu_factor_age_generations =
      std::numeric_limits<std::uint64_t>::max();
  diagnostics.ilu_accounting_complete = false;
  const bool malformed_aggregate_is_not_a_gate =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.configured_ilu = false;
  diagnostics.configured_diagonal = true;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_diag_scale;
  const bool diagonal_is_valid =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.configured_preconditioner = false;
  diagnostics.configured_diagonal = false;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::unconfigured;
  const bool neither_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.configured_preconditioner = true;
  diagnostics.configured_ilu = true;
  diagnostics.configured_diagonal = true;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_ilu0;
  const bool both_are_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.configured_diagonal = false;
  diagnostics.iterations = 8;
  const bool cap_is_inclusive =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.iterations = -1;
  const bool negative_iteration_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.iterations = 9;
  const bool over_cap_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  return reported_tolerance_is_not_a_gate &&
      malformed_aggregate_is_not_a_gate && diagonal_is_valid &&
      neither_is_rejected && both_are_rejected && cap_is_inclusive &&
      negative_iteration_is_rejected && over_cap_is_rejected;
}

static_assert(TrueResidualAuthorityPolicy(),
              "the rebuilt true residual must remain authoritative");

constexpr bool IluSetupOrderPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  using dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupTransition;
  dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupState rejected{};
  if (rejected.Advance(CudaHypreSetupTransition::set_operator) ||
      rejected.Advance(CudaHypreSetupTransition::bind_preconditioner))
    return false;
  dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupState duplicate{};
  if (!duplicate.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      duplicate.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      duplicate.Advance(CudaHypreSetupTransition::set_operator))
    return false;
  dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupState skipped{};
  if (!skipped.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      !skipped.Advance(CudaHypreSetupTransition::set_operator) ||
      !skipped.Advance(CudaHypreSetupTransition::ilu_type) ||
      skipped.Advance(CudaHypreSetupTransition::ilu_reordering) ||
      skipped.Advance(CudaHypreSetupTransition::ilu_fill))
    return false;
  dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupState evidence{};
  evidence.matrix_generation = 23;
  evidence.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_ilu0;
  evidence.ilu_type = 0;
  evidence.ilu_level_of_fill = 0;
  evidence.ilu_local_reordering = 1;
  evidence.ilu_application_iterations = 1;
  evidence.ilu_triangular_solve_kind =
      CudaIluTriangularSolveKind::iterative_jacobi;
  evidence.ilu_lower_jacobi_iterations = 5;
  evidence.ilu_upper_jacobi_iterations = 5;
  if (!evidence.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      !evidence.Advance(CudaHypreSetupTransition::set_operator) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_type) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_fill) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_reordering) ||
      !evidence.Advance(
          CudaHypreSetupTransition::ilu_tri_solve_configured) ||
      !evidence.Advance(
          CudaHypreSetupTransition::ilu_iterative_setup_off) ||
      !evidence.Advance(
          CudaHypreSetupTransition::ilu_application_iterations) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_tol0) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_print0) ||
      !evidence.Advance(CudaHypreSetupTransition::rebind_preconditioner) ||
      !evidence.Advance(CudaHypreSetupTransition::gmres_settings))
    return false;
  auto wrong_fill = evidence;
  wrong_fill.ilu_level_of_fill = 1;
  auto direct = evidence;
  direct.ilu_triangular_solve_kind = CudaIluTriangularSolveKind::direct;
  direct.ilu_lower_jacobi_iterations = 0;
  direct.ilu_upper_jacobi_iterations = 0;
  auto ordinary_conventional = direct;
  ordinary_conventional.ilu_local_reordering = 0;
  if (!evidence.ConfigurationExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      wrong_fill.ConfigurationExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      evidence.ConfigurationExactForGeneration(
          23, CudaEngineKind::ordinary,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      !direct.ConfigurationExactForGeneration(
          23, CudaEngineKind::ordinary,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      !ordinary_conventional.ConfigurationExactForGeneration(
          23, CudaEngineKind::ordinary,
          CudaTracePreconditionerKind::hypre_ilu0, false) ||
      direct.ConfigurationExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      evidence.ConfigurationExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, false) ||
      !evidence.Advance(CudaHypreSetupTransition::mult))
    return false;
  auto extra_terminal = evidence;
  if (extra_terminal.Advance(CudaHypreSetupTransition::mult) ||
      extra_terminal.SolveExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true))
    return false;
  return evidence.SolveExactForGeneration(
             23, CudaEngineKind::tensor_core,
             CudaTracePreconditionerKind::hypre_ilu0, true) &&
      !evidence.SolveExactForGeneration(
          24, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true);
}

static_assert(IluSetupOrderPolicy(),
              "ILU setup and application work must be generation-exact");

constexpr bool DiagScaleSetupOrderPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  using dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupTransition;
  dg::hdg::runtime::cuda_w8a::detail::CudaDiagScaleSetupState evidence{};
  evidence.matrix_generation = 17;
  evidence.selected_engine = CudaEngineKind::tensor_core;
  evidence.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_diag_scale;
  auto rejected = evidence;
  if (rejected.Advance(CudaHypreSetupTransition::set_operator) ||
      rejected.Advance(CudaHypreSetupTransition::bind_preconditioner))
    return false;
  auto duplicate = evidence;
  if (!duplicate.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      duplicate.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      duplicate.Advance(CudaHypreSetupTransition::set_operator))
    return false;
  auto skipped = evidence;
  if (!skipped.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      skipped.Advance(CudaHypreSetupTransition::rebind_preconditioner) ||
      skipped.Advance(CudaHypreSetupTransition::set_operator))
    return false;
  if (!evidence.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      !evidence.Advance(CudaHypreSetupTransition::set_operator) ||
      !evidence.Advance(CudaHypreSetupTransition::rebind_preconditioner) ||
      !evidence.Advance(CudaHypreSetupTransition::gmres_settings) ||
      !evidence.ConfigurationExactForGeneration(
          17, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      evidence.ConfigurationExactForGeneration(
          17, CudaEngineKind::ordinary,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      evidence.ConfigurationExactForGeneration(
          17, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, true) ||
      evidence.SolveExactForGeneration(
          17, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      !evidence.Advance(CudaHypreSetupTransition::mult))
    return false;
  auto extra_terminal = evidence;
  if (extra_terminal.Advance(CudaHypreSetupTransition::mult) ||
      extra_terminal.SolveExactForGeneration(
          17, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, false))
    return false;
  return evidence.SolveExactForGeneration(
             17, CudaEngineKind::tensor_core,
             CudaTracePreconditionerKind::hypre_diag_scale, false) &&
      !evidence.SolveExactForGeneration(
          18, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, false);
}

static_assert(DiagScaleSetupOrderPolicy(),
              "diagonal scaling must be rebound for each matrix generation");

constexpr bool TracePreconditionerPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  return CudaTracePreconditionerForEngine(
             CudaEngineKind::ordinary, false) ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      CudaTracePreconditionerForEngine(CudaEngineKind::ordinary, true) ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      CudaTracePreconditionerForEngine(CudaEngineKind::tensor_core, false) ==
          CudaTracePreconditionerKind::hypre_diag_scale &&
      CudaTracePreconditionerForEngine(CudaEngineKind::tensor_core, true) ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      CudaTracePreconditionerIsIlu(
          CudaTracePreconditionerKind::hypre_ilu0) &&
      !CudaTracePreconditionerIsIlu(
          CudaTracePreconditionerKind::hypre_diag_scale) &&
      CudaIluType(CudaTracePreconditionerKind::hypre_ilu0) == 0 &&
      CudaIluType(CudaTracePreconditionerKind::hypre_diag_scale) == -1 &&
      CudaIluLevelOfFill(CudaTracePreconditionerKind::hypre_ilu0) == 0 &&
      CudaIluLevelOfFill(
          CudaTracePreconditionerKind::hypre_diag_scale) == -1 &&
      CudaIluLocalReordering(
          CudaTracePreconditionerKind::hypre_ilu0, false) == 0 &&
      CudaIluLocalReordering(
          CudaTracePreconditionerKind::hypre_ilu0, true) == 1 &&
      CudaIluApplicationIterations(
          CudaTracePreconditionerKind::hypre_ilu0) == 1 &&
      CudaIluApplicationIterations(
          CudaTracePreconditionerKind::hypre_diag_scale) == 0 &&
      CudaIluApplicationIterations(CudaEngineKind::ordinary, false) == 1 &&
      CudaIluApplicationIterations(CudaEngineKind::ordinary, true) == 1 &&
      CudaIluApplicationIterations(CudaEngineKind::tensor_core, true) == 1 &&
      CudaIluFactorReusePolicyForEngine(CudaEngineKind::ordinary, false) ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve &&
      CudaIluFactorReusePolicyForEngine(CudaEngineKind::ordinary, true) ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve &&
      CudaIluFactorReusePolicyForEngine(CudaEngineKind::tensor_core, false) ==
          CudaIluFactorReusePolicy::not_applicable &&
      CudaIluFactorReusePolicyForEngine(CudaEngineKind::tensor_core, true) ==
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
      CudaIluTriangularSolveKindForEngine(
          CudaEngineKind::ordinary, false) ==
          CudaIluTriangularSolveKind::direct &&
      CudaIluTriangularSolveKindForEngine(
          CudaEngineKind::ordinary, true) ==
          CudaIluTriangularSolveKind::direct &&
      CudaIluTriangularSolveKindForEngine(
          CudaEngineKind::tensor_core, false) ==
          CudaIluTriangularSolveKind::not_applicable &&
      CudaIluTriangularSolveKindForEngine(
          CudaEngineKind::tensor_core, true) ==
          CudaIluTriangularSolveKind::iterative_jacobi &&
      CudaIluTriangularSolve(CudaEngineKind::ordinary, true) == 1 &&
      CudaIluTriangularSolve(CudaEngineKind::tensor_core, false) == -1 &&
      CudaIluTriangularSolve(CudaEngineKind::tensor_core, true) == 0 &&
      CudaIluLowerJacobiIterations(CudaEngineKind::ordinary, true) == 0 &&
      CudaIluUpperJacobiIterations(CudaEngineKind::ordinary, true) == 0 &&
      CudaIluLowerJacobiIterations(CudaEngineKind::tensor_core, true) == 5 &&
      CudaIluUpperJacobiIterations(CudaEngineKind::tensor_core, true) == 5 &&
      std::string_view(CudaTracePreconditionerName(
          CudaTracePreconditionerKind::hypre_ilu0)) == "hypre_ilu0" &&
      std::string_view(CudaIluTriangularSolveKindName(
          CudaIluTriangularSolveKind::iterative_jacobi)) ==
          "iterative_jacobi" &&
      std::string_view(CudaIluFactorReusePolicyName(
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt)) ==
          "first_trace_solve_per_attempt";
}

static_assert(TracePreconditionerPolicy(),
              "trace preconditioner selection must be engine-and-mode exact");

constexpr bool IluDerivedBoundsPolicy() noexcept
{
  using dg::hdg::runtime::cuda_w8a::detail::
      CudaDeriveIluAccountingBoundsV1;
  const auto zero = CudaDeriveIluAccountingBoundsV1(0, 20, 1);
  const auto restart_boundary =
      CudaDeriveIluAccountingBoundsV1(20, 20, 1);
  const auto restart_boundary_jacobi =
      CudaDeriveIluAccountingBoundsV1(20, 20, 1, 8);
  const auto restart_tail =
      CudaDeriveIluAccountingBoundsV1(21, 20, 2, 8);
  const auto zero_krylov = CudaDeriveIluAccountingBoundsV1(1, 0, 1);
  const auto zero_application = CudaDeriveIluAccountingBoundsV1(1, 20, 0);
  const auto addition_overflow = CudaDeriveIluAccountingBoundsV1(
      std::numeric_limits<std::uint64_t>::max(), 1, 1);
  const auto multiplication_overflow = CudaDeriveIluAccountingBoundsV1(
      std::numeric_limits<std::uint64_t>::max()/4+1,
      std::numeric_limits<std::uint64_t>::max(), 2);
  const auto jacobi_multiplication_overflow =
      CudaDeriveIluAccountingBoundsV1(
          std::numeric_limits<std::uint64_t>::max()/4,
          std::numeric_limits<std::uint64_t>::max(), 1, 8);
  return zero.ilu_accounting_complete &&
      zero.ilu_preconditioner_callbacks_lower_bound == 0 &&
      zero.ilu_preconditioner_callbacks_upper_bound == 0 &&
      zero.ilu_inner_iterations_lower_bound == 0 &&
      zero.ilu_inner_iterations_upper_bound == 0 &&
      zero.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
      zero.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
      restart_boundary.ilu_accounting_complete &&
      restart_boundary.ilu_preconditioner_callbacks_lower_bound == 21 &&
      restart_boundary.ilu_preconditioner_callbacks_upper_bound == 40 &&
      restart_boundary.ilu_inner_iterations_lower_bound == 21 &&
      restart_boundary.ilu_inner_iterations_upper_bound == 40 &&
      restart_boundary.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
      restart_boundary.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
      restart_boundary_jacobi.ilu_accounting_complete &&
      restart_boundary_jacobi.
          ilu_jacobi_factor_spmv_updates_lower_bound == 168 &&
      restart_boundary_jacobi.
          ilu_jacobi_factor_spmv_updates_upper_bound == 320 &&
      restart_tail.ilu_accounting_complete &&
      restart_tail.ilu_preconditioner_callbacks_lower_bound == 23 &&
      restart_tail.ilu_preconditioner_callbacks_upper_bound == 42 &&
      restart_tail.ilu_inner_iterations_lower_bound == 46 &&
      restart_tail.ilu_inner_iterations_upper_bound == 84 &&
      restart_tail.ilu_jacobi_factor_spmv_updates_lower_bound == 368 &&
      restart_tail.ilu_jacobi_factor_spmv_updates_upper_bound == 672 &&
      !zero_krylov.ilu_accounting_complete &&
      !zero_application.ilu_accounting_complete &&
      !addition_overflow.ilu_accounting_complete &&
      !multiplication_overflow.ilu_accounting_complete &&
      !jacobi_multiplication_overflow.ilu_accounting_complete;
}

static_assert(IluDerivedBoundsPolicy(),
              "derived ILU bounds must cover zero, restart, tail, and "
              "overflow cases");

constexpr dg::hdg::runtime::cuda_w8a::detail::CudaSolveDiagnostics
IluDiagnostics(const bool igr_active,
               const dg::hdg::generic::CudaEngineKind engine,
               const CudaIluFactorReusePolicy factor_reuse_policy,
               const std::uint64_t factor_setups,
               const std::uint64_t factor_reuses,
               const std::uint64_t factor_age,
               const int iterations) noexcept
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  CudaSolveDiagnostics diagnostics{};
  diagnostics.device_policy = true;
  diagnostics.configured_preconditioner = true;
  diagnostics.configured_ilu = true;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_ilu0;
  diagnostics.ilu_type = 0;
  diagnostics.ilu_level_of_fill = 0;
  diagnostics.ilu_local_reordering = igr_active ? 1 : 0;
  diagnostics.ilu_application_iterations = 1;
  diagnostics.ilu_triangular_solve_kind =
      CudaIluTriangularSolveKindForEngine(engine, igr_active);
  diagnostics.ilu_lower_jacobi_iterations =
      CudaIluLowerJacobiIterations(engine, igr_active);
  diagnostics.ilu_upper_jacobi_iterations =
      CudaIluUpperJacobiIterations(engine, igr_active);
  diagnostics.krylov_dimension = 4;
  diagnostics.igr_active = igr_active;
  diagnostics.ilu_accounting_kind =
      CudaIluAccountingKind::derived_bounds_v1;
  diagnostics.ilu_factor_reuse_policy = factor_reuse_policy;
  diagnostics.ilu_factor_setups = factor_setups;
  diagnostics.ilu_factor_reuses = factor_reuses;
  diagnostics.ilu_factor_age_generations = factor_age;
  diagnostics.iterations = iterations;
  const std::uint64_t factor_spmv_updates_per_application =
      diagnostics.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::iterative_jacobi
      ? 8 : 0;
  const auto bounds = CudaDeriveIluAccountingBoundsV1(
      static_cast<std::uint64_t>(iterations), 4, 1,
      factor_spmv_updates_per_application);
  diagnostics.ilu_preconditioner_callbacks_lower_bound =
      bounds.ilu_preconditioner_callbacks_lower_bound;
  diagnostics.ilu_preconditioner_callbacks_upper_bound =
      bounds.ilu_preconditioner_callbacks_upper_bound;
  diagnostics.ilu_inner_iterations_lower_bound =
      bounds.ilu_inner_iterations_lower_bound;
  diagnostics.ilu_inner_iterations_upper_bound =
      bounds.ilu_inner_iterations_upper_bound;
  diagnostics.ilu_jacobi_factor_spmv_updates_lower_bound =
      bounds.ilu_jacobi_factor_spmv_updates_lower_bound;
  diagnostics.ilu_jacobi_factor_spmv_updates_upper_bound =
      bounds.ilu_jacobi_factor_spmv_updates_upper_bound;
  diagnostics.ilu_accounting_complete = bounds.ilu_accounting_complete;
  return diagnostics;
}

constexpr bool IluWorkAccumulationPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  using namespace dg::hdg::runtime::cuda_w8a::detail;

  CudaTensorCoreEvidence invalid_evidence{};
  invalid_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (CudaInitializeIluEvidence(
          invalid_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, false) ||
      invalid_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence diagonal_evidence{};
  diagonal_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (!CudaInitializeIluEvidence(
          diagonal_evidence,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      diagonal_evidence.ilu_factor_reuse_policy !=
          CudaIluFactorReusePolicy::not_applicable ||
      diagonal_evidence.ilu_triangular_solve_kind !=
          CudaIluTriangularSolveKind::not_applicable ||
      diagonal_evidence.ilu_lower_jacobi_iterations != 0 ||
      diagonal_evidence.ilu_upper_jacobi_iterations != 0 ||
      diagonal_evidence.ilu_factor_setups != 0 ||
      diagonal_evidence.ilu_factor_reuses != 0 ||
      diagonal_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 0 ||
      diagonal_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 0 ||
      diagonal_evidence.ilu_factor_age_generations != 0 ||
      !diagonal_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence ordinary_evidence{};
  ordinary_evidence.selected_engine = CudaEngineKind::ordinary;
  if (!CudaInitializeIluEvidence(
          ordinary_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, false))
    return false;
  const auto refresh = IluDiagnostics(
      false, CudaEngineKind::ordinary,
      CudaIluFactorReusePolicy::refresh_each_trace_solve,
      1, 0, 0, 9);
  if (!CudaAccumulateIluSolveEvidence(ordinary_evidence, refresh) ||
      !CudaAccumulateIluSolveEvidence(ordinary_evidence, refresh) ||
      ordinary_evidence.ilu_factor_reuse_policy !=
          CudaIluFactorReusePolicy::refresh_each_trace_solve ||
      ordinary_evidence.ilu_factor_setups != 2 ||
      ordinary_evidence.ilu_factor_reuses != 0 ||
      ordinary_evidence.ilu_factor_age_generations != 0 ||
      ordinary_evidence.ilu_triangular_solve_kind !=
          CudaIluTriangularSolveKind::direct ||
      ordinary_evidence.ilu_lower_jacobi_iterations != 0 ||
      ordinary_evidence.ilu_upper_jacobi_iterations != 0 ||
      ordinary_evidence.ilu_preconditioner_callbacks_lower_bound != 24 ||
      ordinary_evidence.ilu_preconditioner_callbacks_upper_bound != 36 ||
      ordinary_evidence.ilu_inner_iterations_lower_bound != 24 ||
      ordinary_evidence.ilu_inner_iterations_upper_bound != 36 ||
      ordinary_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 0 ||
      ordinary_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 0 ||
      !ordinary_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence tensor_igr_evidence{};
  tensor_igr_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (!CudaInitializeIluEvidence(
          tensor_igr_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, true))
    return false;
  const auto first = IluDiagnostics(
      true, CudaEngineKind::tensor_core,
      CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
      1, 0, 0, 9);
  const auto reuse = IluDiagnostics(
      true, CudaEngineKind::tensor_core,
      CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
      0, 1, 1, 0);
  if (!CudaAccumulateIluSolveEvidence(tensor_igr_evidence, first) ||
      !CudaAccumulateIluSolveEvidence(tensor_igr_evidence, reuse) ||
      tensor_igr_evidence.ilu_factor_reuse_policy !=
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt ||
      tensor_igr_evidence.ilu_factor_setups != 1 ||
      tensor_igr_evidence.ilu_factor_reuses != 1 ||
      tensor_igr_evidence.ilu_factor_age_generations != 1 ||
      tensor_igr_evidence.ilu_triangular_solve_kind !=
          CudaIluTriangularSolveKind::iterative_jacobi ||
      tensor_igr_evidence.ilu_lower_jacobi_iterations != 5 ||
      tensor_igr_evidence.ilu_upper_jacobi_iterations != 5 ||
      tensor_igr_evidence.ilu_preconditioner_callbacks_lower_bound != 12 ||
      tensor_igr_evidence.ilu_preconditioner_callbacks_upper_bound != 18 ||
      tensor_igr_evidence.ilu_inner_iterations_lower_bound != 12 ||
      tensor_igr_evidence.ilu_inner_iterations_upper_bound != 18 ||
      tensor_igr_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 96 ||
      tensor_igr_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 144 ||
      !tensor_igr_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence incomplete_evidence{};
  incomplete_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (!CudaInitializeIluEvidence(
          incomplete_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, true))
    return false;
  auto malformed_bounds = first;
  malformed_bounds.ilu_preconditioner_callbacks_lower_bound = 11;
  malformed_bounds.ilu_inner_iterations_lower_bound = 11;
  malformed_bounds.ilu_accounting_complete = false;
  const bool malformed_numerically_ready =
      CudaTraceSolveReadyForTrueResidualVerification(malformed_bounds, 9);
  if (!CudaAccumulateIluSolveEvidence(
          incomplete_evidence, malformed_bounds) ||
      !malformed_numerically_ready ||
      incomplete_evidence.ilu_factor_setups != 1 ||
      incomplete_evidence.ilu_factor_reuses != 0 ||
      incomplete_evidence.ilu_factor_age_generations != 0 ||
      incomplete_evidence.ilu_preconditioner_callbacks_lower_bound != 0 ||
      incomplete_evidence.ilu_preconditioner_callbacks_upper_bound != 0 ||
      incomplete_evidence.ilu_inner_iterations_lower_bound != 0 ||
      incomplete_evidence.ilu_inner_iterations_upper_bound != 0 ||
      incomplete_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 0 ||
      incomplete_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 0 ||
      incomplete_evidence.ilu_accounting_complete)
    return false;
  if (!CudaAccumulateIluSolveEvidence(incomplete_evidence, reuse) ||
      incomplete_evidence.ilu_factor_setups != 1 ||
      incomplete_evidence.ilu_factor_reuses != 1 ||
      incomplete_evidence.ilu_factor_age_generations != 1 ||
      incomplete_evidence.ilu_preconditioner_callbacks_lower_bound != 0 ||
      incomplete_evidence.ilu_preconditioner_callbacks_upper_bound != 0 ||
      incomplete_evidence.ilu_inner_iterations_lower_bound != 0 ||
      incomplete_evidence.ilu_inner_iterations_upper_bound != 0 ||
      incomplete_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 0 ||
      incomplete_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 0 ||
      incomplete_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence hard_failure_evidence{};
  hard_failure_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (!CudaInitializeIluEvidence(
          hard_failure_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      CudaAccumulateIluSolveEvidence(hard_failure_evidence, reuse))
    return false;
  return hard_failure_evidence.ilu_factor_setups == 0 &&
      hard_failure_evidence.ilu_factor_reuses == 0 &&
      hard_failure_evidence.ilu_factor_age_generations == 0 &&
      hard_failure_evidence.ilu_preconditioner_callbacks_lower_bound == 0 &&
      hard_failure_evidence.ilu_preconditioner_callbacks_upper_bound == 0 &&
      hard_failure_evidence.ilu_inner_iterations_lower_bound == 0 &&
      hard_failure_evidence.ilu_inner_iterations_upper_bound == 0 &&
      hard_failure_evidence.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
      hard_failure_evidence.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
      !hard_failure_evidence.ilu_accounting_complete;
}

static_assert(IluWorkAccumulationPolicy(),
              "ILU work accumulation must preserve derived bounded work");

constexpr bool IluRefreshConsensusPolicy() noexcept
{
  using dg::hdg::runtime::cuda_w8a::detail::CudaIluRefreshDecisionRequiresConsensus;
  return !CudaIluRefreshDecisionRequiresConsensus(
             CudaIluFactorReusePolicy::not_applicable) &&
      !CudaIluRefreshDecisionRequiresConsensus(
          CudaIluFactorReusePolicy::refresh_each_trace_solve) &&
      CudaIluRefreshDecisionRequiresConsensus(
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt);
}

static_assert(IluRefreshConsensusPolicy(),
              "only retained attempt factors require branch consensus");

bool IluFactorStateLifecyclePolicy() noexcept
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  AttemptIdentity first{};
  first.sequence = 7;
  first.committed_step = 3;
  first.committed_generation = 5;
  first.dt = 1.0e-4;
  AttemptIdentity retry = first;
  retry.sequence = 8;
  retry.retry_index = 1;

  CudaIluAttemptFactorState state{};
  state.active = true;
  state.factor_ready = true;
  if (CudaBeginIluFactorAttempt(
          state, first,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
          101, 201, 0, 8, 3) || state.active || state.factor_ready)
    return false;
  if (!CudaBeginIluFactorAttempt(
          state, first,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
          101, 201, 301, 8, 3) ||
      !CudaIluFactorAttemptMatches(
          state, first,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt) ||
      CudaIluFactorAttemptMatches(
          state, retry,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt) ||
      CudaIluFactorAttemptMatches(
          state, first,
          CudaIluFactorReusePolicy::refresh_each_trace_solve) ||
      !CudaIluFactorRefreshRequired(state))
    return false;
  state.factor_ready = true;
  state.hypre_matrix_token = 401;
  state.ilu_handle_token = 501;
  state.factor_matrix_token = 601;
  state.factor_values_token = 701;
  state.factor_permutation_token = 801;
  state.jacobi_diagonal_vector_token = 901;
  state.jacobi_diagonal_values_token = 1001;
  state.factor_generation = 17;
  state.last_solve_generation = 17;
  if (CudaIluFactorRefreshRequired(state)) return false;
  {
    CudaIluFactorAttemptEndGuard end_factor_attempt(state);
    if (!state.active || !state.factor_ready) return false;
  }
  const AttemptIdentity empty_attempt{};
  if (state.active || state.factor_ready ||
      state.policy != CudaIluFactorReusePolicy::not_applicable ||
      !::dg::hdg::runtime::detail::SameAttempt(
          state.attempt, empty_attempt) ||
      state.matrix_object_token != 0 || state.hypre_matrix_token != 0 ||
      state.diag_values_token != 0 || state.offd_values_token != 0 ||
      state.ilu_handle_token != 0 || state.factor_matrix_token != 0 ||
      state.factor_values_token != 0 ||
      state.factor_permutation_token != 0 ||
      state.jacobi_diagonal_vector_token != 0 ||
      state.jacobi_diagonal_values_token != 0 ||
      state.diag_nonzeros != 0 || state.offd_nonzeros != 0 ||
      state.factor_generation != 0 || state.last_solve_generation != 0)
    return false;

  CudaIluAttemptFactorState refresh_each{};
  if (!CudaBeginIluFactorAttempt(
          refresh_each, first,
          CudaIluFactorReusePolicy::refresh_each_trace_solve,
          102, 202, 0, 9, 0))
    return false;
  refresh_each.factor_ready = true;
  if (!CudaIluFactorRefreshRequired(refresh_each)) return false;

  if (!CudaBeginIluFactorAttempt(
          state, retry,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
          101, 201, 301, 8, 3))
    return false;
  return CudaIluFactorAttemptMatches(
             state, retry,
             CudaIluFactorReusePolicy::first_trace_solve_per_attempt) &&
      !CudaIluFactorAttemptMatches(
          state, first,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt) &&
      CudaIluFactorRefreshRequired(state) && !state.factor_ready &&
      state.hypre_matrix_token == 0 && state.ilu_handle_token == 0 &&
      state.factor_matrix_token == 0 && state.factor_values_token == 0 &&
      state.factor_permutation_token == 0 &&
      state.jacobi_diagonal_vector_token == 0 &&
      state.jacobi_diagonal_values_token == 0 &&
      state.factor_generation == 0 && state.last_solve_generation == 0;
}

struct ModelTag { inline static constexpr std::string_view id = "euler.compressible.2d"; };
struct FormulationTag { inline static constexpr std::string_view id = "hdg.conventional"; };
struct TraceTag { inline static constexpr std::string_view id = "mfem.dg_interface.trace"; };
struct StabilizationTag { inline static constexpr std::string_view id = "tau.fixed"; };
struct TimeTag { inline static constexpr std::string_view id = "backward_euler"; };
struct RankOneTag { inline static constexpr std::string_view id = "cuda.rank1.one_gpu"; };
struct RankTwoTag { inline static constexpr std::string_view id = "cuda.rank2.one_gpu.host_staged"; };
struct PrecisionTag { inline static constexpr std::string_view id = "fp64"; };

template <typename Backend>
using Selection = StaticRunSelection<ModelTag, FormulationTag, TraceTag,
    StabilizationTag, TimeTag, Backend, PrecisionTag>;
template <typename Backend>
using Engine = PersistentCudaSmoothEulerEngine<Selection<Backend>,
    UmpireDeviceAllocatorPolicy, CudaTransportPolicy, CudaMpiCollectivePolicy>;

bool All(const bool local) noexcept
{
  int value = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && result == 1;
}

bool CompactRange(const CudaStepEvidence& evidence,
                  const CudaTransferLedgerRangeV1& range) noexcept
{
  const CudaTransferLedgerV1 ledger = CudaTransportPolicy::Ledger();
  return range.begin_ordinal != 0 && range.end_ordinal >= range.begin_ordinal &&
      range.ordered_digest != 0 && range.complete &&
      evidence.transfer_range.complete && evidence.transfers.complete &&
      evidence.transfers.event_count != 0 && ledger.totals.complete &&
      !ledger.overflow && ledger.observed_count >= range.end_ordinal;
}

bool SetupValid(const CudaRunEvidence& setup, const int ranks,
                const bool igr_active) noexcept
{
  return setup.status == CudaRunStatus::ready && setup.ranks == ranks &&
      setup.tensor_core.trace_preconditioner_configured &&
      setup.tensor_core.trace_preconditioner ==
          CudaTracePreconditionerForEngine(
              setup.tensor_core.selected_engine, igr_active) &&
      setup.mfem_device_valid && setup.hypre_cuda_umpire_device &&
      setup.same_physical_gpu && setup.no_host_solver_fallback &&
      setup.reuse.persistent_graph && setup.reuse.persistent_slots &&
      setup.reuse.persistent_workspace && (ranks == 1 || setup.host_staged_rank2);
}

CudaConfig Config(const int ranks, const bool functional = false) noexcept
{
  CudaConfig config{};
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  if (functional)
  {
    config.order = functional_polynomial_order;
    config.mesh_n = functional_elements_per_direction;
  }
  return config;
}

template <typename Backend>
bool Initialize(Engine<Backend>& engine, CudaRunEvidence& setup,
                const int ranks) noexcept
{
  const bool initialized = engine.Initialize(setup);
  return All(initialized && SetupValid(setup, ranks, false));
}

template <typename Backend>
bool Commit(Engine<Backend>& engine, const bool retry,
            CudaSerializedCheckpointV1& checkpoint) noexcept
{
  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence evidence{};
  const bool prepared = engine.PrepareStep(retry, snapshot, evidence);
  if (!All(prepared && snapshot.numerical_candidate_ready &&
           snapshot.products.CoupledAndOrdered() && evidence.no_fallback &&
           evidence.device_resident))
    return false;
  const bool staged = engine.StagePreparedCheckpoint(checkpoint, evidence);
  if (!All(staged)) { return false; }
  const bool marked = engine.MarkOutputStaged(checkpoint);
  if (!All(marked)) { return false; }
  const bool commit_prepared = engine.PrepareCommit(checkpoint, evidence);
  if (!All(commit_prepared)) { return false; }
  const bool provisional = engine.CommitPrepared(checkpoint, evidence);
  if (!All(provisional)) { return false; }
  const bool finalized = engine.FinalizeCommit(checkpoint, evidence);
  const bool valid = finalized && checkpoint.successful_commit &&
      evidence.committed && evidence.status == CudaRunStatus::committed &&
      evidence.after.step == snapshot.target.step &&
      evidence.after_interior_hash == checkpoint.descriptor.interior_hash &&
      evidence.after_trace_hash == checkpoint.descriptor.trace_hash &&
      CompactRange(evidence, checkpoint.staging_range);
  return All(valid);
}

template <typename Backend>
bool RestoreChecked(Engine<Backend>& engine,
                    const CudaSerializedCheckpointV1& checkpoint,
                    CudaStepEvidence& evidence) noexcept
{
  const bool restored = engine.Restore(checkpoint, evidence);
  const bool valid = restored && evidence.committed &&
      evidence.status == CudaRunStatus::committed &&
      evidence.contract_reason == FailureReason::none &&
      dg::hdg::runtime::detail::SameCoordinates(
          evidence.after, checkpoint.descriptor.coordinates) &&
      evidence.after_interior_hash == checkpoint.descriptor.interior_hash &&
      evidence.after_trace_hash == checkpoint.descriptor.trace_hash &&
      CompactRange(evidence, evidence.transfer_range);
  return All(valid);
}

bool SameCheckpointIdentity(const CudaSerializedCheckpointV1& left,
                            const CudaSerializedCheckpointV1& right) noexcept
{
  return left.local_mesh_hash == right.local_mesh_hash &&
      left.local_partition_hash == right.local_partition_hash &&
      left.local_topology_hash == right.local_topology_hash &&
      left.descriptor.mesh_hash == right.descriptor.mesh_hash &&
      left.descriptor.partition_hash == right.descriptor.partition_hash &&
      left.descriptor.topology_hash == right.descriptor.topology_hash;
}

bool SameCheckpointState(const CudaSerializedCheckpointV1& left,
                         const CudaSerializedCheckpointV1& right) noexcept
{
  return dg::hdg::runtime::detail::SameCoordinates(
             left.descriptor.coordinates, right.descriptor.coordinates) &&
      left.descriptor.interior_hash == right.descriptor.interior_hash &&
      left.descriptor.trace_hash == right.descriptor.trace_hash &&
      left.local_interior_hash == right.local_interior_hash &&
      left.local_trace_hash == right.local_trace_hash &&
      SameCheckpointIdentity(left, right);
}

template <typename Backend>
bool FreshCleanCommit(const int ranks) noexcept
{
  auto engine = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence setup{};
  CudaSerializedCheckpointV1 checkpoint{};
  const bool initialized = Initialize(*engine, setup, ranks);
  if (!initialized) { return false; }
  return Commit(*engine, false, checkpoint);
}

template <typename Backend>
bool LifecycleAndRecovery(const int ranks) noexcept
{
  auto engine = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence setup{};
  if (!Initialize(*engine, setup, ranks)) { return false; }
  CudaSerializedCheckpointV1 initial{};
  if (!Commit(*engine, false, initial)) { return false; }
  const auto* contract = engine->contract();
  if (!All(contract != nullptr)) { return false; }
  const StateCoordinates after_first = contract->coordinates();

  CudaPrecommitSnapshot prepared_snapshot{};
  CudaStepEvidence prepared_evidence{};
  CudaSerializedCheckpointV1 prepared{};
  const bool prepared_step = engine->PrepareStep(false, prepared_snapshot,
                                                  prepared_evidence);
  if (!All(prepared_step && prepared_snapshot.numerical_candidate_ready))
    return false;
  const bool prepared_stage = engine->StagePreparedCheckpoint(prepared,
                                                               prepared_evidence);
  if (!All(prepared_stage)) { return false; }
  const bool prepared_mark = engine->MarkOutputStaged(prepared);
  if (!All(prepared_mark)) { return false; }
  const bool prepared_commit = engine->PrepareCommit(prepared, prepared_evidence);
  if (!All(prepared_commit)) { return false; }
  const bool aborted = engine->AbortPrepared(prepared,
      FailureReason::backend_failure, prepared_evidence);
  const bool abort_valid = aborted && !prepared.successful_commit &&
      !prepared.candidate_frozen && !prepared.checkpoint_staged &&
      !prepared.output_staged && prepared_evidence.retry_available &&
      prepared_evidence.coordinates_preserved &&
      prepared_evidence.committed_bytes_preserved &&
      dg::hdg::runtime::detail::SameCoordinates(contract->coordinates(),
                                                 after_first);
  if (!All(abort_valid)) { return false; }

  CudaSerializedCheckpointV1 after_abort{};
  if (!Commit(*engine, true, after_abort)) { return false; }
  const StateCoordinates after_abort_commit = contract->coordinates();

  CudaPrecommitSnapshot provisional_snapshot{};
  CudaStepEvidence provisional_evidence{};
  CudaSerializedCheckpointV1 provisional{};
  const bool provisional_step = engine->PrepareStep(false, provisional_snapshot,
                                                     provisional_evidence);
  if (!All(provisional_step && provisional_snapshot.numerical_candidate_ready))
    return false;
  const bool provisional_stage = engine->StagePreparedCheckpoint(
      provisional, provisional_evidence);
  if (!All(provisional_stage)) { return false; }
  const bool provisional_mark = engine->MarkOutputStaged(provisional);
  if (!All(provisional_mark)) { return false; }
  const bool provisional_prepare = engine->PrepareCommit(provisional,
                                                          provisional_evidence);
  if (!All(provisional_prepare)) { return false; }
  const bool provisioned = engine->CommitPrepared(provisional,
                                                   provisional_evidence);
  if (!All(provisioned)) { return false; }
  const bool reverted = engine->RevertCommit(provisional,
      FailureReason::backend_failure, provisional_evidence);
  const bool revert_valid = reverted && !provisional.successful_commit &&
      !provisional.candidate_frozen && !provisional.checkpoint_staged &&
      !provisional.output_staged && provisional_evidence.retry_available &&
      provisional_evidence.coordinates_preserved &&
      provisional_evidence.committed_bytes_preserved &&
      dg::hdg::runtime::detail::SameCoordinates(contract->coordinates(),
                                                 after_abort_commit);
  if (!All(revert_valid)) { return false; }

  CudaSerializedCheckpointV1 final{};
  if (!Commit(*engine, true, final)) { return false; }
  engine.reset();

  // A checksum-rejected restore must leave the physical committed slot usable.
  auto reference = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence reference_setup{};
  CudaSerializedCheckpointV1 clean_reference{};
  if (!Initialize(*reference, reference_setup, ranks)) { return false; }
  if (!Commit(*reference, false, clean_reference)) { return false; }
  reference.reset();

  auto corrupt_reader = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence corrupt_setup{};
  if (!Initialize(*corrupt_reader, corrupt_setup, ranks)) { return false; }
  CudaSerializedCheckpointV1 corrupt = final;
  corrupt.interior[0] += 1.0 + (corrupt.interior[0] < 0.0
      ? -corrupt.interior[0] : corrupt.interior[0]);
  CudaStepEvidence corrupt_evidence{};
  const bool corrupt_return = corrupt_reader->Restore(corrupt, corrupt_evidence);
  const bool corrupt_rejected = !corrupt_return &&
      corrupt_evidence.status == CudaRunStatus::runtime_failure &&
      corrupt_evidence.contract_reason == FailureReason::restart_incompatible;
  if (!All(corrupt_rejected)) { return false; }
  CudaSerializedCheckpointV1 after_corrupt{};
  if (!Commit(*corrupt_reader, false, after_corrupt)) { return false; }
  const bool corrupt_bytes_preserved = SameCheckpointState(after_corrupt,
                                                            clean_reference);
  if (!All(corrupt_bytes_preserved)) { return false; }
  corrupt_reader.reset();

  // A consumed checkpoint is likewise rejected without modifying publication.
  auto subject = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence subject_setup{};
  if (!Initialize(*subject, subject_setup, ranks)) { return false; }
  CudaStepEvidence subject_restore{};
  if (!RestoreChecked(*subject, final, subject_restore)) { return false; }
  CudaStepEvidence consumed{};
  const bool consumed_return = subject->Restore(final, consumed);
  const bool consumed_rejected = !consumed_return &&
      consumed.status == CudaRunStatus::runtime_failure &&
      consumed.contract_reason == FailureReason::restart_incompatible;
  if (!All(consumed_rejected)) { return false; }
  CudaSerializedCheckpointV1 subject_next{};
  if (!Commit(*subject, false, subject_next)) { return false; }
  subject.reset();

  auto control = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence control_setup{};
  if (!Initialize(*control, control_setup, ranks)) { return false; }
  CudaStepEvidence control_restore{};
  if (!RestoreChecked(*control, final, control_restore)) { return false; }
  CudaSerializedCheckpointV1 control_next{};
  if (!Commit(*control, false, control_next)) { return false; }
  return All(SameCheckpointState(subject_next, control_next));
}

template <typename Backend>
bool TokenFaults(const int ranks) noexcept
{
  const auto run = [&](const CudaTokenFault fault,
                       const FailureReason expected) noexcept
  {
    CudaConfig config = Config(ranks);
    config.injection.token_fault = fault;
    config.injection.source_rank = ranks - 1;
    auto engine = std::make_unique<Engine<Backend>>(config);
    CudaRunEvidence setup{};
    if (!Initialize(*engine, setup, ranks)) { return false; }
    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence evidence{};
    CudaSerializedCheckpointV1 checkpoint{};
    const bool prepared = engine->PrepareStep(false, snapshot, evidence);
    if (fault != CudaTokenFault::issue_matrix_before_residual)
    {
      if (!All(prepared && snapshot.numerical_candidate_ready &&
               !snapshot.products.CoupledAndOrdered()))
        return false;
      const bool staged = engine->StagePreparedCheckpoint(checkpoint, evidence);
      if (!All(staged)) { return false; }
      const bool marked = engine->MarkOutputStaged(checkpoint);
      if (!All(marked)) { return false; }
      const bool commit_prepared = engine->PrepareCommit(checkpoint, evidence);
      const auto* contract = engine->contract();
      const bool rejected = commit_prepared && contract != nullptr &&
          evidence.status == CudaRunStatus::retry_available &&
          evidence.contract_reason == expected && evidence.retry_available &&
          !evidence.committed && evidence.coordinates_preserved &&
          evidence.committed_bytes_preserved &&
          !checkpoint.candidate_frozen && !checkpoint.checkpoint_staged &&
          !checkpoint.output_staged && !checkpoint.successful_commit &&
          contract->telemetry().commits == 0;
      if (!All(rejected)) { return false; }
      engine.reset();
      return FreshCleanCommit<Backend>(ranks);
    }
    if (!All(prepared && snapshot.numerical_candidate_ready &&
             snapshot.products.CoupledAndOrdered()))
      return false;
    const bool staged = engine->StagePreparedCheckpoint(checkpoint, evidence);
    if (!All(staged)) { return false; }
    const bool marked = engine->MarkOutputStaged(checkpoint);
    if (!All(marked)) { return false; }
    const bool commit_prepared = engine->PrepareCommit(checkpoint, evidence);
    if (!All(commit_prepared)) { return false; }
    const bool provisioned = engine->CommitPrepared(checkpoint, evidence);
    if (!All(provisioned)) { return false; }
    const bool finalized = engine->FinalizeCommit(checkpoint, evidence);
    const auto* contract = engine->contract();
    const bool recovered = finalized && checkpoint.successful_commit &&
        contract != nullptr && contract->telemetry().commits == 1 &&
        contract->telemetry().out_of_order_product_issues_rejected == 1 &&
        contract->telemetry().last_attempt.validated_product_mask == 0x0fU &&
        CompactRange(evidence, checkpoint.staging_range);
    return All(recovered);
  };
  if (!run(CudaTokenFault::omit_solution, FailureReason::missing_product))
    return false;
  if (!run(CudaTokenFault::stale_attempt, FailureReason::missing_product))
    return false;
  return run(CudaTokenFault::issue_matrix_before_residual, FailureReason::none);
}

template <typename Backend>
bool FunctionalRestart(const int ranks) noexcept
{
  auto reference = std::make_unique<Engine<Backend>>(Config(ranks, true));
  CudaRunEvidence reference_setup{};
  if (!Initialize(*reference, reference_setup, ranks)) { return false; }
  CudaSerializedCheckpointV1 reference_final{};
  CudaSerializedCheckpointV1 bridge{};
  for (int step = 1; step <= functional_accepted_steps; ++step)
  {
    CudaSerializedCheckpointV1 checkpoint{};
    if (!Commit(*reference, false, checkpoint)) { return false; }
    if (step == functional_checkpoint_step) { bridge = checkpoint; }
    reference_final = checkpoint;
  }
  const auto* reference_contract = reference->contract();
  if (!All(reference_contract != nullptr &&
           reference_contract->telemetry().commits == functional_accepted_steps))
    return false;
  reference.reset();

  auto writer = std::make_unique<Engine<Backend>>(Config(ranks, true));
  CudaRunEvidence writer_setup{};
  if (!Initialize(*writer, writer_setup, ranks)) { return false; }
  CudaSerializedCheckpointV1 split{};
  for (int step = 1; step <= functional_checkpoint_step; ++step)
    if (!Commit(*writer, false, split)) { return false; }
  const auto* writer_contract = writer->contract();
  const bool checkpoint_agrees = writer_contract != nullptr &&
      writer_contract->telemetry().commits == functional_checkpoint_step &&
      SameCheckpointState(split, bridge);
  if (!All(checkpoint_agrees)) { return false; }
  writer.reset();

  auto reader = std::make_unique<Engine<Backend>>(Config(ranks, true));
  CudaRunEvidence reader_setup{};
  if (!Initialize(*reader, reader_setup, ranks)) { return false; }
  CudaStepEvidence restored{};
  if (!RestoreChecked(*reader, split, restored)) { return false; }
  CudaSerializedCheckpointV1 resumed_final{};
  for (int step = functional_checkpoint_step + 1;
       step <= functional_accepted_steps; ++step)
    if (!Commit(*reader, false, resumed_final)) { return false; }
  const auto* reader_contract = reader->contract();
  const bool same_final = reader_contract != nullptr &&
      reader_contract->telemetry().commits == functional_checkpoint_step &&
      SameCheckpointState(reference_final, resumed_final) &&
      reference_final.descriptor.coordinates.step == functional_accepted_steps &&
      reference_final.descriptor.coordinates.generation == functional_accepted_steps;
  return All(same_final);
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const std::string_view mode = argc == 2 ? argv[1] : "";
  const bool known_mode = mode.empty() || mode == "--token-negatives" ||
      mode == "--functional";
  const bool pass = IluFactorStateLifecyclePolicy() && known_mode && (ranks == 1
      ? (mode == "--functional" ? FunctionalRestart<RankOneTag>(ranks)
         : mode == "--token-negatives" ? TokenFaults<RankOneTag>(ranks)
         : LifecycleAndRecovery<RankOneTag>(ranks))
      : ranks == 2
      ? (mode == "--functional" ? FunctionalRestart<RankTwoTag>(ranks)
         : mode == "--token-negatives" ? TokenFaults<RankTwoTag>(ranks)
         : LifecycleAndRecovery<RankTwoTag>(ranks))
      : false);
  if (!pass && rank == 0)
    std::cerr << "compact CUDA W8A audit failed\n";
  return pass ? 0 : 1;
}

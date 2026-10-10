#include "igrlab/environment.hpp"
#include "generic_euler_convergence_fixture.hpp"

#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "sim/hdg/euler_output_fields.hpp"
#include "sim/hdg/parallel_vtk.hpp"
#include "sim/hdg/regularization_scaling.hpp"
#include "sim/hdg/transactional_time_adapter.hpp"

#include <cuda_runtime.h>
#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

enum class CudaQualificationRobustBranch
{
  not_applicable,
  any_qualified,
  lower,
  coupled
};

// One reduced record per prepared step for the bounded architecture screen.
// This remains test-only and deliberately retains no stage or iteration
// history.  The friend accessor below copies the already-computed aggregate
// before the affine prepared authority is consumed.
struct CudaArchitectureStepAggregate
{
  double final_residual = std::numeric_limits<double>::infinity();
  double true_relative_residual = std::numeric_limits<double>::infinity();
  double maximum_conservation_error =
      std::numeric_limits<double>::infinity();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::infinity();
  double temporal_error_norm = std::numeric_limits<double>::infinity();
  double final_algebraic_residual =
      std::numeric_limits<double>::infinity();
  double final_algebraic_correction_norm =
      std::numeric_limits<double>::infinity();
  std::uint64_t complete_assemblies = 0;
  std::uint64_t complete_condensations = 0;
  std::uint64_t complete_recoveries = 0;
  std::uint64_t common_shift_operator_builds = 0;
  std::uint64_t common_shift_operator_reuses = 0;
  std::uint64_t condensed_rhs_refreshes = 0;
  std::uint64_t common_shift_trace_solves = 0;
  std::uint64_t common_shift_preconditioner_setups = 0;
  std::uint64_t common_shift_preconditioner_reuses = 0;
  std::uint64_t native_trace_solve_calls = 0;
  std::uint64_t hypre_trace_solve_calls = 0;
  std::uint64_t patch_setups = 0;
  std::uint64_t preconditioner_applications = 0;
  std::uint64_t matrix_free_operator_applications = 0;
  int nonlinear_updates = 0;
  int linear_iterations = 0;
  int linear_solve_attempts = 0;
  int trace_solves = 0;
  int scalar_closure_solves = 0;
  int scalar_closure_linear_iterations = 0;
  int true_residual_tightenings = 0;
  bool finite = false;
  bool admissible = false;
  bool recovery = false;
  bool true_residual = false;
  bool conservation = false;
  bool success = false;
  bool no_fallback = false;
  bool trace_policy_exact = false;
  bool common_shift_contract = false;
};

// This finite qualification accessor binds one explicit trace-policy
// ablation and checks the already-retained prepared aggregate.  It adds no
// production selector, callback, history, or diagnostic surface.
struct CudaQualificationTestAccess
{
  static dg::hdg::generic::HdgEulerSessionResult InitializeTracePolicy(
      cuda::GenericEulerHdgCudaSession& session,
      const dg::hdg::generic::EulerHdgRequest& request,
      const CudaTracePolicyKind trace_policy) noexcept
  {
    return session.InitializeRequest(request, trace_policy);
  }

  static bool PreparedTracePolicyValid(
      cuda::GenericEulerHdgCudaSession& session,
      const dg::hdg::generic::EulerHdgRequest& request,
      const CudaTracePolicyKind requested_trace_policy,
      const CudaQualificationRobustBranch robust_branch) noexcept
  {
    return session.Dispatch([&](auto& engine)
    {
    const auto& pending = engine.pending_;
    const auto& numerical = pending.numerical;
    const auto& tensor = pending.evidence.tensor_core;
    const bool igr_active =
        request.numerical_policy.stabilization.kind ==
            dg::hdg::generic::RegularizationKind::igr;
    const CudaTracePolicyDescriptor expected = ResolveCudaTracePolicy(
        request.backend.cuda_engine, igr_active, 1,
        request.polynomial_order, requested_trace_policy);
    if (numerical.trace_solves <= 0 ||
        numerical.linear_solve_attempts < numerical.trace_solves)
      return false;
    const std::uint64_t solves =
        static_cast<std::uint64_t>(numerical.trace_solves);
    const std::uint64_t attempts =
        static_cast<std::uint64_t>(numerical.linear_solve_attempts);
    const bool robust =
        CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
            expected.resolved_selection) &&
        CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(
            expected.trace_preconditioner);
    const bool robust_counts_bounded =
        tensor.robust_lower_patch_setups <= solves &&
        tensor.robust_coupled_patch_setups <= solves &&
        tensor.robust_lower_trace_solve_calls <= attempts &&
        tensor.robust_coupled_trace_solve_calls <= attempts &&
        tensor.robust_lower_trace_solve_calls <=
            std::numeric_limits<std::uint64_t>::max() -
                tensor.robust_coupled_trace_solve_calls &&
        tensor.robust_lower_patch_setups <=
            std::numeric_limits<std::uint64_t>::max() -
                tensor.robust_coupled_patch_setups &&
        attempts <= std::numeric_limits<std::uint64_t>::max() -
            tensor.robust_coupled_patch_setups;
    const std::uint64_t physical_calls = robust_counts_bounded
        ? tensor.robust_lower_trace_solve_calls +
              tensor.robust_coupled_trace_solve_calls
        : 0;
    const std::uint64_t physical_setups = robust_counts_bounded
        ? tensor.robust_lower_patch_setups +
              tensor.robust_coupled_patch_setups
        : 0;
    const bool robust_accounting = robust
        ? robust_counts_bounded &&
              tensor.robust_lower_specialization_supported &&
              tensor.robust_coupled_specialization_supported &&
              tensor.robust_branch_accounting_complete &&
              tensor.robust_lower_patch_setups == solves &&
              tensor.element_schwarz_patch_setups == physical_setups &&
              tensor.native_rank1_trace_solve_calls == physical_calls &&
              physical_calls >= attempts &&
              physical_calls <=
                  attempts + tensor.robust_coupled_patch_setups &&
              tensor.robust_coupled_trace_solve_calls >=
                  tensor.robust_coupled_patch_setups &&
              tensor.robust_lower_trace_solve_calls >=
                  solves - tensor.robust_coupled_patch_setups &&
              ((robust_branch ==
                    CudaQualificationRobustBranch::any_qualified) ||
               (robust_branch == CudaQualificationRobustBranch::lower &&
                tensor.robust_lower_trace_solve_calls == attempts &&
               tensor.robust_coupled_patch_setups == 0 &&
               tensor.robust_coupled_trace_solve_calls == 0) ||
               (robust_branch == CudaQualificationRobustBranch::coupled &&
                tensor.robust_coupled_patch_setups > 0 &&
                tensor.robust_coupled_trace_solve_calls >=
                    tensor.robust_coupled_patch_setups &&
                physical_calls > attempts))
        : robust_branch == CudaQualificationRobustBranch::not_applicable &&
              !tensor.robust_lower_specialization_supported &&
              !tensor.robust_coupled_specialization_supported &&
              tensor.robust_branch_accounting_complete &&
              tensor.robust_lower_patch_setups == 0 &&
              tensor.robust_lower_trace_solve_calls == 0 &&
              tensor.robust_coupled_patch_setups == 0 &&
              tensor.robust_coupled_trace_solve_calls == 0;
    const bool exact_preconditioner_leaf = robust
        ? CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(
              expected.trace_preconditioner)
        : CudaTracePreconditionerIsNativeElementSchwarzWarpCoupled(
              expected.trace_preconditioner);
    const bool valid = CudaTracePolicyDescriptorIsExact(expected) &&
        pending.active && pending.staging && numerical.candidate_ready &&
        numerical.finite && numerical.admissible && numerical.recovery &&
        numerical.true_residual && numerical.conservation &&
        numerical.success && pending.evidence.no_fallback &&
        tensor.selected_engine == request.backend.cuda_engine &&
        tensor.requested_trace_policy == requested_trace_policy &&
        tensor.resolved_trace_policy == expected.resolved_selection &&
        tensor.trace_policy_rank_count == 1 &&
        tensor.trace_policy_order == request.polynomial_order &&
        tensor.trace_preconditioner_configured &&
        tensor.trace_preconditioner == expected.trace_preconditioner &&
        exact_preconditioner_leaf && robust_accounting &&
        tensor.native_rank1_trace_solver_configured &&
        tensor.native_rank1_trace_solve_calls ==
            (robust ? physical_calls : attempts) &&
        tensor.hypre_trace_solve_calls == 0 &&
        tensor.trace_solver_accounting_complete &&
        tensor.native_matrix_free_trace_operator_configured &&
        tensor.native_element_schwarz_configured &&
        tensor.fallback_count == 0 &&
        tensor.fallback_accounting_complete;
    if (!valid)
    {
      std::cerr << "trace-policy qualification failed"
                << " active=" << pending.active
                << " staging=" << static_cast<bool>(pending.staging)
                << " candidate=" << numerical.candidate_ready
                << " finite=" << numerical.finite
                << " admissible=" << numerical.admissible
                << " recovery=" << numerical.recovery
                << " true_residual=" << numerical.true_residual
                << " conservation=" << numerical.conservation
                << " numerical_success=" << numerical.success
                << " trace_solves=" << numerical.trace_solves
                << " linear_attempts=" << numerical.linear_solve_attempts
                << " robust=" << robust
                << " robust_accounting=" << robust_accounting
                << " lower_supported="
                << tensor.robust_lower_specialization_supported
                << " coupled_supported="
                << tensor.robust_coupled_specialization_supported
                << " branch_accounting="
                << tensor.robust_branch_accounting_complete
                << " lower_setups=" << tensor.robust_lower_patch_setups
                << " lower_solves="
                << tensor.robust_lower_trace_solve_calls
                << " coupled_setups="
                << tensor.robust_coupled_patch_setups
                << " coupled_solves="
                << tensor.robust_coupled_trace_solve_calls
                << " exact_leaf=" << exact_preconditioner_leaf
                << " native_calls="
                << tensor.native_rank1_trace_solve_calls
                << " hypre_calls=" << tensor.hypre_trace_solve_calls
                << " trace_accounting="
                << tensor.trace_solver_accounting_complete
                << " matrix_free="
                << tensor.native_matrix_free_trace_operator_configured
                << " schwarz="
                << tensor.native_element_schwarz_configured
                << " no_fallback=" << pending.evidence.no_fallback
                << " fallback_count=" << tensor.fallback_count
                << " fallback_accounting="
                << tensor.fallback_accounting_complete << '\n';
    }
    return valid;
    });
  }

  static bool ReadPreparedArchitectureAggregate(
      cuda::GenericEulerHdgCudaSession& session,
      const dg::hdg::generic::EulerHdgRequest& request,
      const CudaTracePolicyKind requested_trace_policy,
      CudaArchitectureStepAggregate& output) noexcept
  {
    output = {};
    return session.Dispatch([&](auto& engine)
    {
      const auto& pending = engine.pending_;
      const auto& numerical = pending.numerical;
      const auto& tensor = numerical.tensor_core;
      const bool igr_active =
          request.numerical_policy.stabilization.kind ==
              dg::hdg::generic::RegularizationKind::igr;
      const CudaTracePolicyDescriptor expected = ResolveCudaTracePolicy(
          request.backend.cuda_engine, igr_active, 1,
          request.polynomial_order, requested_trace_policy);
      output.final_residual = numerical.final_residual;
      output.true_relative_residual = numerical.true_relative_residual;
      output.maximum_conservation_error =
          numerical.maximum_conservation_error;
      output.maximum_conservation_relative_error =
          numerical.maximum_conservation_relative_error;
      output.temporal_error_norm = numerical.temporal_error_norm;
      output.final_algebraic_residual = numerical.final_algebraic_residual;
      output.final_algebraic_correction_norm =
          numerical.final_algebraic_correction_norm;
      output.complete_assemblies = numerical.complete_assemblies;
      output.complete_condensations = numerical.complete_condensations;
      output.complete_recoveries = numerical.complete_recoveries;
      output.common_shift_operator_builds =
          numerical.common_shift_operator_builds;
      output.common_shift_operator_reuses =
          numerical.common_shift_operator_reuses;
      output.condensed_rhs_refreshes = numerical.condensed_rhs_refreshes;
      output.common_shift_trace_solves =
          numerical.common_shift_trace_solves;
      output.common_shift_preconditioner_setups =
          numerical.common_shift_preconditioner_setups;
      output.common_shift_preconditioner_reuses =
          numerical.common_shift_preconditioner_reuses;
      output.native_trace_solve_calls =
          tensor.native_rank1_trace_solve_calls;
      output.hypre_trace_solve_calls = tensor.hypre_trace_solve_calls;
      output.patch_setups = tensor.element_schwarz_patch_setups;
      output.preconditioner_applications =
          tensor.element_schwarz_preconditioner_applications;
      output.matrix_free_operator_applications =
          tensor.native_matrix_free_trace_operator_applications;
      output.nonlinear_updates = numerical.nonlinear_updates;
      output.linear_iterations = numerical.linear_iterations;
      output.linear_solve_attempts = numerical.linear_solve_attempts;
      output.trace_solves = numerical.trace_solves;
      output.scalar_closure_solves = numerical.scalar_closure_solves;
      output.scalar_closure_linear_iterations =
          numerical.scalar_closure_linear_iterations;
      output.true_residual_tightenings =
          numerical.true_residual_tightenings;
      output.finite = numerical.finite;
      output.admissible = numerical.admissible;
      output.recovery = numerical.recovery;
      output.true_residual = numerical.true_residual;
      output.conservation = numerical.conservation;
      output.success = numerical.success;
      output.no_fallback = pending.evidence.no_fallback;
      output.trace_policy_exact = CudaTracePolicyDescriptorIsExact(expected) &&
          tensor.selected_engine == request.backend.cuda_engine &&
          tensor.requested_trace_policy == requested_trace_policy &&
          tensor.resolved_trace_policy == expected.resolved_selection &&
          tensor.trace_policy_rank_count == 1 &&
          tensor.trace_policy_order == request.polynomial_order &&
          tensor.trace_preconditioner_configured &&
          tensor.trace_preconditioner == expected.trace_preconditioner &&
          tensor.trace_solver_accounting_complete &&
          tensor.fallback_accounting_complete;
      const bool rodas3 = request.time_treatment ==
          dg::hdg::generic::EulerHdgTimeTreatment::rodas3;
      output.common_shift_contract = !rodas3 ||
          (numerical.common_shift_operator_builds == 1 &&
           numerical.common_shift_operator_reuses == 3 &&
           numerical.condensed_rhs_refreshes == 3 &&
           numerical.common_shift_trace_solves == 4 &&
           numerical.common_shift_preconditioner_setups >= 1 &&
           numerical.common_shift_preconditioner_reuses == 3 &&
           tensor.native_matrix_free_trace_operator_configured &&
           tensor.native_element_schwarz_configured &&
           tensor.native_rank1_trace_solver_configured &&
           tensor.hypre_trace_solve_calls == 0);
      return pending.active && pending.staging &&
          numerical.candidate_ready && output.finite &&
          output.admissible && output.recovery && output.true_residual &&
          output.conservation && output.success && output.no_fallback &&
          output.trace_policy_exact && output.common_shift_contract &&
          std::isfinite(output.final_residual) &&
          std::isfinite(output.true_relative_residual) &&
          std::isfinite(output.maximum_conservation_error) &&
          std::isfinite(output.maximum_conservation_relative_error) &&
          std::isfinite(output.temporal_error_norm) &&
          std::isfinite(output.final_algebraic_residual) &&
          std::isfinite(output.final_algebraic_correction_norm) &&
          output.trace_solves > 0 && output.linear_solve_attempts >=
              output.trace_solves;
    });
  }
};

} // namespace dg::hdg::runtime::cuda_w8a::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerHdgTimeTreatment;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerHdgCpuSession;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyDescriptor;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyDescriptorIsExact;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKind;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKindName;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyUsesNativeElementMatrixFreeWarpCoupled;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled;
using dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationRobustBranch;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;
using dg::hdg::test::generic_euler_convergence::CollectiveAll;
using dg::hdg::test::generic_euler_convergence::ExecutionSelection;

enum class ScreenMemoryMethod
{
  none,
  rodas3,
  monolithic_backward_euler,
  partitioned_backward_euler
};

struct TestSelection
{
  CudaTracePolicyKind tensor_igr_trace_policy =
      CudaTracePolicyKind::engine_default;
  EulerHdgTimeTreatment time_treatment =
      EulerHdgTimeTreatment::backward_euler;
  bool zero_gradient_only = false;
  bool architecture_selection_screen = false;
  bool adaptive_rodas3_screen = false;
  bool screen_probe_only = false;
  int screen_elements_per_direction = 12;
  double screen_final_time = 0.08;
  double screen_time_atol = 0.0;
  double screen_time_rtol = 0.0;
  double screen_initial_dt = 0.0;
  double screen_dt_min = 0.0;
  ScreenMemoryMethod screen_memory_method = ScreenMemoryMethod::none;
  int screen_memory_steps = 0;
  std::string screen_output_root;
};

bool ParseScreenElements(const std::string_view text, int& value) noexcept
{
  value = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end &&
      value >= 4 && value <= 32;
}

bool ParseScreenMemorySteps(const std::string_view text, int& value) noexcept
{
  value = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end &&
      value >= 1 && value <= 512;
}

bool ParseScreenMemoryMethod(
    const std::string_view text, ScreenMemoryMethod& method) noexcept
{
  if (text == "rodas3")
  {
    method = ScreenMemoryMethod::rodas3;
    return true;
  }
  if (text == "monolithic_backward_euler")
  {
    method = ScreenMemoryMethod::monolithic_backward_euler;
    return true;
  }
  if (text == "partitioned_backward_euler")
  {
    method = ScreenMemoryMethod::partitioned_backward_euler;
    return true;
  }
  return false;
}

bool ParseTimeTreatment(
    const std::string_view text,
    EulerHdgTimeTreatment& treatment) noexcept
{
  if (text == "backward-euler")
  {
    treatment = EulerHdgTimeTreatment::backward_euler;
    return true;
  }
  if (text == "partitioned-backward-euler")
  {
    treatment = EulerHdgTimeTreatment::partitioned_backward_euler;
    return true;
  }
  return false;
}

bool ParseExplicitTracePolicy(
    const std::string_view text, CudaTracePolicyKind& policy) noexcept
{
  constexpr std::array<CudaTracePolicyKind, 4> policies{{
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_fp64,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi2_fp64,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi10_fp64,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64}};
  for (const CudaTracePolicyKind candidate : policies)
    if (text == CudaTracePolicyKindName(candidate))
    {
      policy = candidate;
      return true;
    }
  return false;
}

bool ParseTestSelection(
    const int argc, char* argv[],
    dg::hdg::test::generic_euler_convergence::CaseSelection& test_case,
    TestSelection& selection)
{
  selection = {};
  bool have_trace_policy = false;
  bool have_time_treatment = false;
  bool have_zero_gradient_only = false;
  bool have_architecture_selection_screen = false;
  bool have_adaptive_rodas3_screen = false;
  bool have_screen_probe_only = false;
  bool have_screen_elements = false;
  bool have_screen_final_time = false;
  bool have_screen_time_atol = false;
  bool have_screen_time_rtol = false;
  bool have_screen_initial_dt = false;
  bool have_screen_dt_min = false;
  bool have_screen_memory_method = false;
  bool have_screen_memory_steps = false;
  bool have_screen_output_root = false;
  std::vector<char*> common_arguments;
  common_arguments.reserve(static_cast<std::size_t>(argc));
  common_arguments.push_back(argv[0]);
  for (int argument = 1; argument < argc; ++argument)
  {
    const std::string_view option{argv[argument]};
    if (option == "--tensor-igr-trace-policy" &&
        argument + 1 < argc && !have_trace_policy)
    {
      have_trace_policy = ParseExplicitTracePolicy(
          argv[++argument], selection.tensor_igr_trace_policy);
      if (!have_trace_policy) return false;
    }
    else if (option == "--time-treatment" &&
             argument + 1 < argc && !have_time_treatment)
    {
      have_time_treatment = ParseTimeTreatment(
          argv[++argument], selection.time_treatment);
      if (!have_time_treatment) return false;
    }
    else if (option == "--zero-gradient-only" &&
             !have_zero_gradient_only)
    {
      selection.zero_gradient_only = true;
      have_zero_gradient_only = true;
    }
    else if (option == "--architecture-selection-screen" &&
             !have_architecture_selection_screen)
    {
      selection.architecture_selection_screen = true;
      have_architecture_selection_screen = true;
    }
    else if (option == "--adaptive-rodas3-screen" &&
             !have_adaptive_rodas3_screen)
    {
      selection.adaptive_rodas3_screen = true;
      have_adaptive_rodas3_screen = true;
    }
    else if (option == "--screen-probe-only" && !have_screen_probe_only)
    {
      selection.screen_probe_only = true;
      have_screen_probe_only = true;
    }
    else if (option == "--screen-elements" && argument + 1 < argc &&
             !have_screen_elements)
    {
      have_screen_elements = ParseScreenElements(
          argv[++argument], selection.screen_elements_per_direction);
      if (!have_screen_elements) return false;
    }
    else if (option == "--screen-final-time" && argument + 1 < argc &&
             !have_screen_final_time)
    {
      have_screen_final_time =
          dg::hdg::test::generic_euler_convergence::ParsePositiveDouble(
              argv[++argument], selection.screen_final_time);
      if (!have_screen_final_time || selection.screen_final_time > 0.25)
        return false;
    }
    else if (option == "--screen-time-atol" && argument + 1 < argc &&
             !have_screen_time_atol)
    {
      have_screen_time_atol =
          dg::hdg::test::generic_euler_convergence::ParsePositiveDouble(
              argv[++argument], selection.screen_time_atol);
      if (!have_screen_time_atol) return false;
    }
    else if (option == "--screen-time-rtol" && argument + 1 < argc &&
             !have_screen_time_rtol)
    {
      have_screen_time_rtol =
          dg::hdg::test::generic_euler_convergence::ParsePositiveDouble(
              argv[++argument], selection.screen_time_rtol);
      if (!have_screen_time_rtol) return false;
    }
    else if (option == "--screen-initial-dt" && argument + 1 < argc &&
             !have_screen_initial_dt)
    {
      have_screen_initial_dt =
          dg::hdg::test::generic_euler_convergence::ParsePositiveDouble(
              argv[++argument], selection.screen_initial_dt);
      if (!have_screen_initial_dt) return false;
    }
    else if (option == "--screen-dt-min" && argument + 1 < argc &&
             !have_screen_dt_min)
    {
      have_screen_dt_min =
          dg::hdg::test::generic_euler_convergence::ParsePositiveDouble(
              argv[++argument], selection.screen_dt_min);
      if (!have_screen_dt_min) return false;
    }
    else if (option == "--screen-memory-method" &&
             argument + 1 < argc && !have_screen_memory_method)
    {
      have_screen_memory_method = ParseScreenMemoryMethod(
          argv[++argument], selection.screen_memory_method);
      if (!have_screen_memory_method) return false;
    }
    else if (option == "--screen-memory-steps" &&
             argument + 1 < argc && !have_screen_memory_steps)
    {
      have_screen_memory_steps = ParseScreenMemorySteps(
          argv[++argument], selection.screen_memory_steps);
      if (!have_screen_memory_steps) return false;
    }
    else if (option == "--screen-output-root" && argument + 1 < argc &&
             !have_screen_output_root)
    {
      selection.screen_output_root = argv[++argument];
      have_screen_output_root = !selection.screen_output_root.empty();
      if (!have_screen_output_root) return false;
    }
    else
    {
      common_arguments.push_back(argv[argument]);
    }
  }
  if (!dg::hdg::test::generic_euler_convergence::ParseCaseSelection(
          static_cast<int>(common_arguments.size()),
          common_arguments.data(), test_case))
    return false;
  const bool explicit_policy = selection.tensor_igr_trace_policy !=
      CudaTracePolicyKind::engine_default;
  const bool igr = test_case.regularization ==
      dg::hdg::generic::RegularizationKind::igr;
  const bool memory_screen = selection.screen_memory_method !=
      ScreenMemoryMethod::none;
  const bool rodas_memory_screen = selection.screen_memory_method ==
      ScreenMemoryMethod::rodas3;
  const bool fixed_memory_screen = memory_screen && !rodas_memory_screen;
  if (explicit_policy != have_trace_policy ||
      (explicit_policy && !igr) ||
      (explicit_policy && selection.time_treatment !=
          EulerHdgTimeTreatment::backward_euler) ||
      (selection.zero_gradient_only && !igr) ||
      (selection.architecture_selection_screen &&
       (have_trace_policy || have_time_treatment ||
        selection.zero_gradient_only || test_case.quadrature_equivalence)) ||
      (selection.screen_probe_only &&
       !selection.architecture_selection_screen) ||
      (selection.screen_probe_only && have_screen_output_root) ||
      (memory_screen && selection.screen_probe_only) ||
      (rodas_memory_screen && !selection.adaptive_rodas3_screen) ||
      (fixed_memory_screen && selection.adaptive_rodas3_screen) ||
      (have_screen_memory_steps && !fixed_memory_screen) ||
      (fixed_memory_screen && !have_screen_memory_steps) ||
      (selection.adaptive_rodas3_screen &&
       (!selection.architecture_selection_screen ||
        selection.screen_probe_only ||
        !have_screen_time_atol || !have_screen_time_rtol ||
        !have_screen_initial_dt || !have_screen_dt_min)) ||
      (!selection.adaptive_rodas3_screen &&
       (have_screen_time_atol || have_screen_time_rtol ||
        have_screen_initial_dt || have_screen_dt_min)) ||
      (selection.adaptive_rodas3_screen &&
       (selection.screen_initial_dt > selection.screen_final_time ||
        selection.screen_dt_min > selection.screen_initial_dt)) ||
      (!selection.architecture_selection_screen &&
       (have_adaptive_rodas3_screen || have_screen_probe_only ||
        have_screen_elements ||
        have_screen_final_time || have_screen_memory_method ||
        have_screen_memory_steps || have_screen_output_root)))
    return false;
  if (selection.architecture_selection_screen) return true;
  if (!explicit_policy && !selection.zero_gradient_only) return true;
  const CudaTracePolicyDescriptor descriptor = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, igr, 1, test_case.order,
      selection.tensor_igr_trace_policy);
  return CudaTracePolicyDescriptorIsExact(descriptor) &&
      (CudaTracePolicyUsesNativeElementMatrixFreeWarpCoupled(
           descriptor.resolved_selection) ||
       CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
           descriptor.resolved_selection));
}

template <typename Coordinates, typename Diagnostics>
bool ValidCommittedMetadata(const Coordinates coordinates,
                            const Diagnostics diagnostics,
                            const int steps, const double dt) noexcept
{
  const double expected_time = static_cast<double>(steps) * dt;
  const double time_scale = std::max(1.0, std::abs(expected_time));
  return coordinates.step == static_cast<std::uint64_t>(steps) &&
      coordinates.generation == static_cast<std::uint64_t>(steps) &&
      std::abs(coordinates.time - expected_time) <=
          64.0 * std::numeric_limits<double>::epsilon() * time_scale &&
      diagnostics.available &&
      diagnostics.wave_generation == coordinates.generation &&
      diagnostics.coordinates.step == coordinates.step &&
      std::isfinite(diagnostics.maximum_wave_speed) &&
      diagnostics.maximum_wave_speed > 0.0;
}

template <typename Coordinates>
bool SameSessionCoordinates(const Coordinates& left,
                            const Coordinates& right) noexcept
{
  return left.time == right.time && left.dt == right.dt &&
      left.step == right.step && left.generation == right.generation;
}

template <typename Coordinates, typename Diagnostics>
bool ValidAdaptiveCommittedMetadata(
    const Coordinates coordinates, const Diagnostics diagnostics,
    const std::uint64_t accepted_steps,
    const double final_time) noexcept
{
  return coordinates.step == accepted_steps &&
      coordinates.generation == accepted_steps &&
      coordinates.time == final_time && diagnostics.available &&
      diagnostics.wave_generation == coordinates.generation &&
      diagnostics.coordinates.step == coordinates.step &&
      diagnostics.coordinates.time == coordinates.time &&
      std::isfinite(diagnostics.maximum_wave_speed) &&
      diagnostics.maximum_wave_speed > 0.0;
}

bool ExactDoubleConsensus(const double value,
                          MPI_Comm communicator) noexcept
{
  const std::uint64_t local = dg::hdg::generic::DoubleBits(value);
  std::uint64_t minimum = 0;
  std::uint64_t maximum = 0;
  return MPI_Allreduce(
             &local, &minimum, 1, MPI_UINT64_T, MPI_MIN,
             communicator) == MPI_SUCCESS &&
      MPI_Allreduce(
          &local, &maximum, 1, MPI_UINT64_T, MPI_MAX,
          communicator) == MPI_SUCCESS &&
      minimum == maximum;
}

bool ExactBoolConsensus(const bool value,
                        MPI_Comm communicator) noexcept
{
  const int local = value ? 1 : 0;
  int minimum = 0;
  int maximum = 0;
  return MPI_Allreduce(
             &local, &minimum, 1, MPI_INT, MPI_MIN,
             communicator) == MPI_SUCCESS &&
      MPI_Allreduce(
          &local, &maximum, 1, MPI_INT, MPI_MAX,
          communicator) == MPI_SUCCESS &&
      minimum == maximum;
}

struct ArchitectureWorkAggregate
{
  std::int64_t accepted_steps = 0;
  std::int64_t rejected_attempts = 0;
  std::int64_t nonlinear_updates = 0;
  std::int64_t linear_iterations = 0;
  std::int64_t linear_solve_attempts = 0;
  std::int64_t trace_solves = 0;
  std::int64_t scalar_closure_solves = 0;
  std::int64_t scalar_closure_linear_iterations = 0;
  std::int64_t true_residual_tightenings = 0;
  std::uint64_t complete_assemblies = 0;
  std::uint64_t complete_condensations = 0;
  std::uint64_t complete_recoveries = 0;
  std::uint64_t common_shift_operator_builds = 0;
  std::uint64_t common_shift_operator_reuses = 0;
  std::uint64_t condensed_rhs_refreshes = 0;
  std::uint64_t common_shift_trace_solves = 0;
  std::uint64_t common_shift_preconditioner_setups = 0;
  std::uint64_t common_shift_preconditioner_reuses = 0;
  std::uint64_t native_trace_solve_calls = 0;
  std::uint64_t hypre_trace_solve_calls = 0;
  std::uint64_t patch_setups = 0;
  std::uint64_t preconditioner_applications = 0;
  std::uint64_t matrix_free_operator_applications = 0;
  double maximum_final_residual = 0.0;
  double maximum_true_relative_residual = 0.0;
  double maximum_conservation_error = 0.0;
  double maximum_conservation_relative_error = 0.0;
  double maximum_temporal_error_norm = 0.0;
  double maximum_final_algebraic_residual = 0.0;
  double maximum_final_algebraic_correction_norm = 0.0;
  bool finite_gate = true;
  bool admissibility_gate = true;
  bool recovery_gate = true;
  bool true_residual_gate = true;
  bool conservation_gate = true;
  bool no_fallback_gate = true;
  bool trace_policy_gate = true;
  bool common_shift_gate = true;
};

struct ArchitectureTrajectoryResult
{
  std::string_view method{};
  EulerHdgTimeTreatment time_treatment =
      EulerHdgTimeTreatment::backward_euler;
  int steps = 0;
  double dt = 0.0;
  double time_absolute_tolerance = 0.0;
  double time_relative_tolerance = 0.0;
  double adaptive_initial_dt = 0.0;
  double adaptive_minimum_dt = 0.0;
  double measured_wall_seconds = std::numeric_limits<double>::infinity();
  double comparable_wall_seconds = std::numeric_limits<double>::infinity();
  double measured_u4_l2_error = std::numeric_limits<double>::infinity();
  double sigma_coefficient_rms = std::numeric_limits<double>::infinity();
  dg::hdg::test::generic_euler_convergence::GlobalFieldMetrics field{};
  ArchitectureWorkAggregate work{};
  double maximum_wave_speed = 0.0;
  std::size_t device_total_bytes = 0;
  std::size_t device_free_bytes_before = 0;
  std::size_t minimum_observed_device_free_bytes = 0;
  std::size_t device_free_bytes_after = 0;
  std::size_t observed_device_free_memory_delta_bytes = 0;
  std::size_t internal_cuda_used_floor_bytes = 0;
  std::uint64_t controller_revision = 0;
  std::uint64_t output_publications = 0;
  int timing_repetitions = 1;
  bool adaptive = false;
  bool temporal_reference_error = false;
  bool midpoint_output_landed = false;
  bool final_output_landed = false;
  bool prepared_assessment_authority = false;
  bool passed = false;
};

struct AdaptiveArchitectureControl
{
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  double initial_dt = 0.0;
  double minimum_dt = 0.0;
};

double GlobalCoefficientRms(
    const std::vector<double>& values, const MPI_Comm communicator)
{
  double local_square_sum = 0.0;
  bool local_valid = !values.empty();
  for (const double value : values)
  {
    local_valid = local_valid && std::isfinite(value);
    local_square_sum = std::fma(value, value, local_square_sum);
  }
  local_valid = local_valid && std::isfinite(local_square_sum);
  if (!dg::hdg::test::generic_euler_convergence::CollectiveAll(
          local_valid, communicator))
    return std::numeric_limits<double>::infinity();

  double global_square_sum = 0.0;
  const unsigned long long local_count =
      static_cast<unsigned long long>(values.size());
  unsigned long long global_count = 0;
  if (MPI_Allreduce(&local_square_sum, &global_square_sum, 1, MPI_DOUBLE,
                    MPI_SUM, communicator) != MPI_SUCCESS ||
      MPI_Allreduce(&local_count, &global_count, 1,
                    MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                    communicator) != MPI_SUCCESS ||
      global_count == 0 || !std::isfinite(global_square_sum) ||
      global_square_sum <= 0.0)
    return std::numeric_limits<double>::infinity();
  return std::sqrt(
      global_square_sum / static_cast<double>(global_count));
}

bool AccumulateArchitectureStep(
    const dg::hdg::runtime::cuda_w8a::detail::
        CudaArchitectureStepAggregate& step,
    ArchitectureWorkAggregate& total,
    const bool accepted = true) noexcept
{
  const bool nonnegative = step.nonlinear_updates >= 0 &&
      step.linear_iterations >= 0 && step.linear_solve_attempts >= 0 &&
      step.trace_solves > 0 && step.scalar_closure_solves >= 0 &&
      step.scalar_closure_linear_iterations >= 0 &&
      step.true_residual_tightenings >= 0;
  const std::int64_t disposition_count = accepted
      ? total.accepted_steps : total.rejected_attempts;
  if (!nonnegative || disposition_count ==
          std::numeric_limits<std::int64_t>::max())
    return false;
  if (accepted)
    ++total.accepted_steps;
  else
    ++total.rejected_attempts;
  total.nonlinear_updates += step.nonlinear_updates;
  total.linear_iterations += step.linear_iterations;
  total.linear_solve_attempts += step.linear_solve_attempts;
  total.trace_solves += step.trace_solves;
  total.scalar_closure_solves += step.scalar_closure_solves;
  total.scalar_closure_linear_iterations +=
      step.scalar_closure_linear_iterations;
  total.true_residual_tightenings += step.true_residual_tightenings;
  total.complete_assemblies += step.complete_assemblies;
  total.complete_condensations += step.complete_condensations;
  total.complete_recoveries += step.complete_recoveries;
  total.common_shift_operator_builds +=
      step.common_shift_operator_builds;
  total.common_shift_operator_reuses +=
      step.common_shift_operator_reuses;
  total.condensed_rhs_refreshes += step.condensed_rhs_refreshes;
  total.common_shift_trace_solves += step.common_shift_trace_solves;
  total.common_shift_preconditioner_setups +=
      step.common_shift_preconditioner_setups;
  total.common_shift_preconditioner_reuses +=
      step.common_shift_preconditioner_reuses;
  total.native_trace_solve_calls += step.native_trace_solve_calls;
  total.hypre_trace_solve_calls += step.hypre_trace_solve_calls;
  total.patch_setups += step.patch_setups;
  total.preconditioner_applications += step.preconditioner_applications;
  total.matrix_free_operator_applications +=
      step.matrix_free_operator_applications;
  total.maximum_final_residual = std::max(
      total.maximum_final_residual, step.final_residual);
  total.maximum_true_relative_residual = std::max(
      total.maximum_true_relative_residual, step.true_relative_residual);
  total.maximum_conservation_error = std::max(
      total.maximum_conservation_error, step.maximum_conservation_error);
  total.maximum_conservation_relative_error = std::max(
      total.maximum_conservation_relative_error,
      step.maximum_conservation_relative_error);
  total.maximum_temporal_error_norm = std::max(
      total.maximum_temporal_error_norm, step.temporal_error_norm);
  total.maximum_final_algebraic_residual = std::max(
      total.maximum_final_algebraic_residual,
      step.final_algebraic_residual);
  total.maximum_final_algebraic_correction_norm = std::max(
      total.maximum_final_algebraic_correction_norm,
      step.final_algebraic_correction_norm);
  total.finite_gate = total.finite_gate && step.finite;
  total.admissibility_gate = total.admissibility_gate && step.admissible;
  total.recovery_gate = total.recovery_gate && step.recovery;
  total.true_residual_gate =
      total.true_residual_gate && step.true_residual;
  total.conservation_gate = total.conservation_gate && step.conservation;
  total.no_fallback_gate = total.no_fallback_gate && step.no_fallback;
  total.trace_policy_gate =
      total.trace_policy_gate && step.trace_policy_exact;
  total.common_shift_gate =
      total.common_shift_gate && step.common_shift_contract;
  return true;
}

bool RunTensorArchitectureTrajectory(
    const EulerHdgRequest& request,
    mfem::ParMesh& mesh,
    const std::string_view method,
    const EulerHdgTimeTreatment time_treatment,
    const CudaTracePolicyKind trace_policy,
    const int steps, const double final_time,
    const double common_setup_seconds,
    mfem::L2_FECollection& collection,
    mfem::ParGridFunction& committed,
    ArchitectureTrajectoryResult& result,
    const std::string_view output_root = {},
    const std::string_view output_tag = {},
    const bool publish_midpoint = false)
{
  using clock = std::chrono::steady_clock;
  using dg::hdg::generic::HdgEulerSessionCapability;
  using dg::hdg::generic::HdgEulerSessionCapabilityBit;
  using dg::hdg::generic::HasHdgEulerSessionCapabilities;
  using dg::hdg::runtime::cuda_w8a::detail::
      CudaArchitectureStepAggregate;
  using dg::hdg::test::generic_euler_convergence::MeasureGlobalField;

  result = {};
  result.method = method;
  result.time_treatment = time_treatment;
  result.steps = steps;
  result.dt = final_time / static_cast<double>(steps);
  const bool publish_output = !output_root.empty();
  if (publish_output != !output_tag.empty() || request.mesh == nullptr ||
      (publish_midpoint && (!publish_output || steps % 2 != 0)))
    return false;
  EulerHdgRequest selected_request = request;
  selected_request.time_treatment = time_treatment;
  if (MPI_Barrier(request.communicator) != MPI_SUCCESS) return false;
  const auto start = clock::now();
  std::size_t free_before = 0;
  std::size_t total_before = 0;
  const bool initial_memory_sample =
      cudaMemGetInfo(&free_before, &total_before) == cudaSuccess;
  result.device_total_bytes = total_before;
  result.device_free_bytes_before = free_before;
  result.minimum_observed_device_free_bytes = free_before;
  const bool local = [&]()
  {
    if (!initial_memory_sample || HYPRE_ClearAllErrors() != 0 ||
        HYPRE_GetError() != 0)
      return false;
    const auto sample_memory = [&]()
    {
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      const bool sampled =
          cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
          total_bytes == result.device_total_bytes;
      if (sampled)
        result.minimum_observed_device_free_bytes = std::min(
            result.minimum_observed_device_free_bytes, free_bytes);
      return sampled;
    };
    GenericEulerHdgCudaSession session;
    const auto initialization =
        CudaQualificationTestAccess::InitializeTracePolicy(
            session, selected_request, trace_policy);
    if (initialization.status != HdgEulerSessionStatus::initialized)
    {
      std::cerr << "architecture-screen initialization failed method="
                << method << " steps=" << steps << " status="
                << static_cast<int>(initialization.status) << '\n';
      return false;
    }
    std::unique_ptr<dg_sim::hdg::HdgEulerParaViewFields> output_fields;
    std::unique_ptr<dg_sim::hdg::HDGParaViewSeries> output;
    const auto publish = [&](const int cycle, const int step,
                             const double time)
    {
      dg::hdg::generic::HdgEulerHostInteriorFields destination{};
      if (!output_fields || !output ||
          !output_fields->FreshHostOutput(destination) ||
          !session.ExportCommittedFields(destination) ||
          !output_fields->UpdateDerived())
        return false;
      try
      {
        output->Write({cycle, step, time});
        return true;
      }
      catch (...) { return false; }
    };
    if (publish_output)
    {
      try
      {
        output_fields =
            std::make_unique<dg_sim::hdg::HdgEulerParaViewFields>(
                mesh, collection, committed,
                selected_request.numerical_policy.stabilization.kind ==
                    dg::hdg::generic::RegularizationKind::igr,
                1.4, 1.0e-16);
        output = std::make_unique<dg_sim::hdg::HDGParaViewSeries>(
            dg_sim::hdg::HDGParaViewRequest{
                mesh, output_fields->Fields(),
                {std::string{output_root}, std::string{output_tag}}});
      }
      catch (...) { return false; }
      if (!publish(0, 0, 0.0)) return false;
      result.output_publications = 1;
    }
    if (!sample_memory()) return false;
    const bool prepared_assessment_required =
        time_treatment == EulerHdgTimeTreatment::rodas3;
    const bool prepared_assessment_capability =
        HasHdgEulerSessionCapabilities(
            session.capabilities(), HdgEulerSessionCapabilityBit(
                HdgEulerSessionCapability::prepared_embedded_assessment));
    if (prepared_assessment_required &&
        !prepared_assessment_capability)
    {
      std::cerr << "architecture-screen missing prepared assessment "
                   "capability method="
                << method << " steps=" << steps << '\n';
      return false;
    }
    int authorized_prepared_assessments = 0;
    constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
        prepared_assessment_policy{1.0e-8, 1.0e-5, 0.25};
    for (int step_index = 0; step_index < steps; ++step_index)
    {
      GenericEulerCudaPreparedStep authority;
      GenericEulerHdgCudaSession::prepared_output_type output{};
      const auto prepared = session.Prepare(result.dt, authority, output);
      if (prepared.status != HdgEulerSessionStatus::prepared)
      {
        ++result.work.rejected_attempts;
        std::cerr << "architecture-screen prepare failed method=" << method
                  << " steps=" << steps << " step=" << step_index
                  << " status=" << static_cast<int>(prepared.status)
                  << '\n';
        return false;
      }
      CudaArchitectureStepAggregate step{};
      const bool evidence =
          CudaQualificationTestAccess::ReadPreparedArchitectureAggregate(
              session, selected_request, trace_policy, step);
      if (!evidence)
      {
        ++result.work.rejected_attempts;
        (void)session.Abort(std::move(authority), output);
        std::cerr << "architecture-screen prepared aggregate failed method="
                  << method << " steps=" << steps << " step="
                  << step_index
                  << " finite=" << step.finite
                  << " admissible=" << step.admissible
                  << " recovery=" << step.recovery
                  << " true_residual=" << step.true_residual
                  << " conservation=" << step.conservation
                  << " success=" << step.success
                  << " no_fallback=" << step.no_fallback
                  << " trace_policy_exact=" << step.trace_policy_exact
                  << " common_shift_contract="
                  << step.common_shift_contract
                  << " final_residual=" << step.final_residual
                  << " true_relative_residual="
                  << step.true_relative_residual
                  << " conservation_error="
                  << step.maximum_conservation_error
                  << " conservation_relative="
                  << step.maximum_conservation_relative_error
                  << " temporal_error=" << step.temporal_error_norm
                  << " algebraic_residual="
                  << step.final_algebraic_residual
                  << " algebraic_correction="
                  << step.final_algebraic_correction_norm
                  << " trace_solves=" << step.trace_solves
                  << " linear_attempts=" << step.linear_solve_attempts
                  << " common_shift_builds="
                  << step.common_shift_operator_builds
                  << " common_shift_reuses="
                  << step.common_shift_operator_reuses
                  << " rhs_refreshes=" << step.condensed_rhs_refreshes
                  << " common_shift_solves="
                  << step.common_shift_trace_solves
                  << " preconditioner_setups="
                  << step.common_shift_preconditioner_setups
                  << " preconditioner_reuses="
                  << step.common_shift_preconditioner_reuses
                  << " native_calls=" << step.native_trace_solve_calls
                  << " hypre_calls=" << step.hypre_trace_solve_calls
                  << '\n';
        return false;
      }
      if (prepared_assessment_required)
      {
        dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
        const bool assessed = session.AssessPrepared(
            authority, output, prepared_assessment_policy, assessment);
        const bool assessment_authorized = assessed &&
            dg::hdg::generic::IsValidHdgEulerPreparedAssessment(
                assessment) &&
            assessment.finite_gate && assessment.solver_gate &&
            assessment.residual_gate &&
            assessment.admissibility_gate &&
            assessment.algebraic_closure_gate &&
            assessment.algebraic_correction_gate;
        if (!assessment_authorized)
        {
          ++result.work.rejected_attempts;
          (void)session.Abort(std::move(authority), output);
          std::cerr << "architecture-screen prepared assessment failed "
                       "method="
                    << method << " steps=" << steps << " step="
                    << step_index << " assessed=" << assessed
                    << " normalized_differential_error="
                    << assessment.normalized_differential_error
                    << " normalized_algebraic_correction="
                    << assessment.normalized_algebraic_correction
                    << " finite_gate=" << assessment.finite_gate
                    << " solver_gate=" << assessment.solver_gate
                    << " residual_gate=" << assessment.residual_gate
                    << " admissibility_gate="
                    << assessment.admissibility_gate
                    << " algebraic_closure_gate="
                    << assessment.algebraic_closure_gate
                    << " algebraic_correction_gate="
                    << assessment.algebraic_correction_gate << '\n';
          return false;
        }
        ++authorized_prepared_assessments;
      }
      if (!session.AcknowledgePreparedOutput(authority, output))
      {
        ++result.work.rejected_attempts;
        (void)session.Abort(std::move(authority), output);
        return false;
      }
      const auto committed_step =
          session.Commit(std::move(authority), output);
      if (committed_step.status != HdgEulerSessionStatus::committed ||
          !AccumulateArchitectureStep(step, result.work))
      {
        ++result.work.rejected_attempts;
        return false;
      }
      if (!sample_memory()) return false;
      if (publish_midpoint && step_index + 1 == steps / 2 &&
          !publish(1, step_index + 1, 0.5 * final_time))
        return false;
      if (publish_midpoint && step_index + 1 == steps / 2)
      {
        ++result.output_publications;
        result.midpoint_output_landed = true;
      }
    }
    result.prepared_assessment_authority =
        prepared_assessment_required && steps > 0 &&
        authorized_prepared_assessments == steps;
    const int size = committed.Size();
    constexpr int conserved_components =
        dg::hdg::generic::euler_2d_conserved_components;
    const bool extent_valid = size > 0 && size % conserved_components == 0;
    const std::size_t scalar_count = extent_valid
        ? static_cast<std::size_t>(size / conserved_components) : 0;
    std::vector<double> sigma(scalar_count);
    if (!extent_valid || !session.ExportCommittedFields({
            committed.HostWrite(), sigma.data(), scalar_count}) ||
        !ValidCommittedMetadata(
            session.coordinates(), session.diagnostics(), steps, result.dt))
      return false;
    if (publish_output && !publish(
            publish_midpoint ? 2 : 1, steps, final_time))
      return false;
    if (publish_output)
    {
      ++result.output_publications;
      result.final_output_landed = true;
    }
    result.maximum_wave_speed = session.diagnostics().maximum_wave_speed;
    result.field = MeasureGlobalField(committed);
    result.sigma_coefficient_rms = GlobalCoefficientRms(
        sigma, request.communicator);
    return result.field.valid && result.field.minimum_density > 0.0 &&
        result.field.minimum_pressure > 0.0 &&
        std::isfinite(result.sigma_coefficient_rms) &&
        result.sigma_coefficient_rms > 0.0 &&
        result.work.accepted_steps == steps &&
        result.work.rejected_attempts == 0 &&
        result.work.finite_gate && result.work.admissibility_gate &&
        result.work.recovery_gate && result.work.true_residual_gate &&
        result.work.conservation_gate && result.work.no_fallback_gate &&
        result.work.trace_policy_gate && result.work.common_shift_gate &&
        (!prepared_assessment_required ||
         result.prepared_assessment_authority);
  }();
  std::size_t free_after = 0;
  std::size_t total_after = 0;
  const bool final_memory_sample =
      cudaMemGetInfo(&free_after, &total_after) == cudaSuccess &&
      total_after == result.device_total_bytes;
  result.device_free_bytes_after = free_after;
  result.observed_device_free_memory_delta_bytes =
      result.device_free_bytes_before >=
          result.minimum_observed_device_free_bytes
      ? result.device_free_bytes_before -
          result.minimum_observed_device_free_bytes
      : 0;
  result.internal_cuda_used_floor_bytes =
      result.device_total_bytes >= result.minimum_observed_device_free_bytes
      ? result.device_total_bytes -
          result.minimum_observed_device_free_bytes
      : 0;
  const bool synchronized =
      MPI_Barrier(request.communicator) == MPI_SUCCESS;
  const double local_seconds =
      std::chrono::duration<double>(clock::now() - start).count();
  double maximum_seconds = 0.0;
  const bool time_reduced = MPI_Allreduce(
      &local_seconds, &maximum_seconds, 1, MPI_DOUBLE, MPI_MAX,
      request.communicator) == MPI_SUCCESS;
  result.measured_wall_seconds = maximum_seconds;
  result.comparable_wall_seconds = common_setup_seconds + maximum_seconds;
  result.passed = CollectiveAll(
      local && synchronized && time_reduced && final_memory_sample &&
          std::isfinite(result.measured_wall_seconds) &&
          result.measured_wall_seconds > 0.0 &&
          std::isfinite(result.comparable_wall_seconds) &&
          result.comparable_wall_seconds >= result.measured_wall_seconds,
      request.communicator);
  return result.passed;
}

bool RunTensorAdaptiveArchitectureTrajectory(
    const EulerHdgRequest& request,
    mfem::ParMesh& mesh,
    const std::string_view method,
    const CudaTracePolicyKind trace_policy,
    const AdaptiveArchitectureControl control,
    const double final_time,
    const double common_setup_seconds,
    mfem::L2_FECollection& collection,
    mfem::ParGridFunction& committed,
    ArchitectureTrajectoryResult& result,
    const std::string_view output_root = {},
    const std::string_view output_tag = {})
{
  using clock = std::chrono::steady_clock;
  using dg::hdg::generic::HdgEulerSessionCapability;
  using dg::hdg::generic::HdgEulerSessionCapabilityBit;
  using dg::hdg::generic::HasHdgEulerSessionCapabilities;
  using dg::hdg::runtime::cuda_w8a::detail::
      CudaArchitectureStepAggregate;
  using dg::hdg::test::generic_euler_convergence::MeasureGlobalField;
  using dg_sim::hdg::HdgTimeControlMode;
  using dg_sim::hdg::HdgTransactionalTimeAdapter;

  result = {};
  result.method = method;
  result.time_treatment = EulerHdgTimeTreatment::rodas3;
  result.dt = control.initial_dt;
  result.time_absolute_tolerance = control.absolute_tolerance;
  result.time_relative_tolerance = control.relative_tolerance;
  result.adaptive_initial_dt = control.initial_dt;
  result.adaptive_minimum_dt = control.minimum_dt;
  result.adaptive = true;
  const bool publish_output = !output_root.empty();
  const dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
      assessment_policy{control.absolute_tolerance,
                        control.relative_tolerance, 0.25};
  if (publish_output != !output_tag.empty() || request.mesh == nullptr ||
      !dg::hdg::generic::IsValidHdgEulerPreparedAssessmentPolicy(
          assessment_policy) ||
      !std::isfinite(control.initial_dt) || control.initial_dt <= 0.0 ||
      control.initial_dt > final_time ||
      !std::isfinite(control.minimum_dt) || control.minimum_dt <= 0.0 ||
      control.minimum_dt > control.initial_dt ||
      !std::isfinite(final_time) || final_time <= 0.0)
    return false;

  EulerHdgRequest selected_request = request;
  selected_request.time_treatment = EulerHdgTimeTreatment::rodas3;
  if (MPI_Barrier(request.communicator) != MPI_SUCCESS) return false;
  const auto start = clock::now();
  std::size_t free_before = 0;
  std::size_t total_before = 0;
  const bool initial_memory_sample =
      cudaMemGetInfo(&free_before, &total_before) == cudaSuccess;
  result.device_total_bytes = total_before;
  result.device_free_bytes_before = free_before;
  result.minimum_observed_device_free_bytes = free_before;

  const bool local = [&]()
  {
    if (!initial_memory_sample || HYPRE_ClearAllErrors() != 0 ||
        HYPRE_GetError() != 0)
      return false;
    const auto sample_memory = [&]()
    {
      std::size_t free_bytes = 0;
      std::size_t total_bytes = 0;
      const bool sampled =
          cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
          total_bytes == result.device_total_bytes;
      if (sampled)
        result.minimum_observed_device_free_bytes = std::min(
            result.minimum_observed_device_free_bytes, free_bytes);
      return sampled;
    };

    GenericEulerHdgCudaSession session;
    const auto initialization =
        CudaQualificationTestAccess::InitializeTracePolicy(
            session, selected_request, trace_policy);
    const bool assessment_capability =
        HasHdgEulerSessionCapabilities(
            session.capabilities(), HdgEulerSessionCapabilityBit(
                HdgEulerSessionCapability::prepared_embedded_assessment));
    if (initialization.status != HdgEulerSessionStatus::initialized ||
        !assessment_capability)
    {
      std::cerr << "adaptive architecture-screen initialization failed "
                   "method="
                << method << " status="
                << static_cast<int>(initialization.status)
                << " assessment_capability=" << assessment_capability
                << '\n';
      return false;
    }

    std::unique_ptr<dg_sim::hdg::HdgEulerParaViewFields> output_fields;
    std::unique_ptr<dg_sim::hdg::HDGParaViewSeries> output;
    const auto publish = [&](const std::uint64_t cycle,
                             const std::uint64_t step,
                             const double time)
    {
      dg::hdg::generic::HdgEulerHostInteriorFields destination{};
      if (cycle > static_cast<std::uint64_t>(
              std::numeric_limits<int>::max()) ||
          step > static_cast<std::uint64_t>(
              std::numeric_limits<int>::max()) ||
          !output_fields || !output ||
          !output_fields->FreshHostOutput(destination) ||
          !session.ExportCommittedFields(destination) ||
          !output_fields->UpdateDerived())
        return false;
      try
      {
        output->Write({static_cast<int>(cycle),
                       static_cast<int>(step), time});
        return true;
      }
      catch (...) { return false; }
    };

    HdgTransactionalTimeAdapter::AdaptiveControllerConfig
        controller_config;
    controller_config.minimum_dt = control.minimum_dt;
    controller_config.maximum_dt = final_time;
    std::unique_ptr<HdgTransactionalTimeAdapter> time_driver;
    try
    {
      const auto time_control =
          HdgTransactionalTimeAdapter::TimeControl::Adaptive(
              0.0, final_time, control.initial_dt);
      const auto output_schedule = publish_output
          ? HdgTransactionalTimeAdapter::OutputSchedule::FixedInterval(
                0.5 * final_time,
                dg_sim::time::backend::InitialOutputPolicy::Emit,
                dg_sim::time::backend::FinalOutputPolicy::Emit)
          : HdgTransactionalTimeAdapter::OutputSchedule::Disabled(
                dg_sim::time::backend::InitialOutputPolicy::Suppress,
                dg_sim::time::backend::FinalOutputPolicy::Suppress);
      time_driver = std::make_unique<HdgTransactionalTimeAdapter>(
          time_control, output_schedule, controller_config,
          HdgTimeControlMode::AdaptiveRodas3);
      if (publish_output)
      {
        output_fields =
            std::make_unique<dg_sim::hdg::HdgEulerParaViewFields>(
                mesh, collection, committed,
                selected_request.numerical_policy.stabilization.kind ==
                    dg::hdg::generic::RegularizationKind::igr,
                1.4, 1.0e-16);
        output = std::make_unique<dg_sim::hdg::HDGParaViewSeries>(
            dg_sim::hdg::HDGParaViewRequest{
                mesh, output_fields->Fields(),
                {std::string{output_root}, std::string{output_tag}}});
        const auto initial = time_driver->InitialOutput();
        if (!initial || !initial->is_initial || initial->is_final ||
            initial->cycle != 0 || initial->time != 0.0 ||
            !publish(initial->cycle, 0, initial->time))
          return false;
        result.output_publications = 1;
      }
      else if (time_driver->InitialOutput())
      {
        return false;
      }
    }
    catch (...) { return false; }

    if (!sample_memory()) return false;
    constexpr std::uint64_t maximum_attempts = 512;
    std::uint64_t attempted_steps = 0;
    std::uint64_t authorized_assessments = 0;
    std::uint64_t failed_prepares = 0;
    while (!time_driver->Finished())
    {
      if (attempted_steps == maximum_attempts) return false;
      ++attempted_steps;
      HdgTransactionalTimeAdapter::StepPlan plan{};
      try { plan = time_driver->BeginAdaptiveStep(); }
      catch (...) { return false; }
      if (!std::isfinite(plan.dt) || plan.dt <= 0.0 ||
          plan.end_time > final_time ||
          !ExactDoubleConsensus(plan.dt, request.communicator))
        return false;

      const auto coordinates_before = session.coordinates();
      const double time_before = time_driver->time();
      const std::uint64_t step_before = time_driver->solver_step();
      GenericEulerCudaPreparedStep authority;
      GenericEulerHdgCudaSession::prepared_output_type native_output{};
      const auto prepared = session.Prepare(
          plan.dt, authority, native_output);
      if (prepared.status != HdgEulerSessionStatus::prepared)
      {
        ++failed_prepares;
        const bool abandoned = session.RejectFailedPrepare().status ==
            HdgEulerSessionStatus::aborted;
        std::cerr << "adaptive architecture-screen prepare failed method="
                  << method << " attempted_dt=" << plan.dt
                  << " abandoned=" << abandoned << '\n';
        return false;
      }

      CudaArchitectureStepAggregate step{};
      if (!CudaQualificationTestAccess::ReadPreparedArchitectureAggregate(
              session, selected_request, trace_policy, step))
      {
        (void)session.Abort(std::move(authority), native_output);
        return false;
      }

      dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
      const bool assessed = session.AssessPrepared(
          authority, native_output, assessment_policy, assessment);
      const bool assessment_values_valid = CollectiveAll(
          dg::hdg::generic::IsValidHdgEulerPreparedAssessment(assessment),
          request.communicator);
      const bool assessment_values_collective =
          ExactDoubleConsensus(
              assessment.normalized_differential_error,
              request.communicator) &&
          ExactDoubleConsensus(
              assessment.normalized_algebraic_correction,
              request.communicator);
      const bool assessment_ready = CollectiveAll(
          assessed, request.communicator);
      if (!assessment_values_valid || !assessment_values_collective ||
          !assessment_ready)
      {
        (void)session.RejectPrepared(
            std::move(authority), native_output);
        return false;
      }
      ++authorized_assessments;

      const HdgTransactionalTimeAdapter::AdaptiveAttempt attempt{
          assessment.normalized_differential_error,
          assessment.finite_gate && assessment.solver_gate,
          assessment.residual_gate,
          assessment.admissibility_gate,
          assessment.algebraic_closure_gate &&
              assessment.algebraic_correction_gate};
      std::optional<HdgTransactionalTimeAdapter::AdaptiveDecision>
          decision;
      try { decision.emplace(
          time_driver->PreviewAdaptiveDecision(attempt)); }
      catch (...)
      {
        (void)session.Abort(std::move(authority), native_output);
        return false;
      }
      const bool decision_collective =
          ExactBoolConsensus(
              decision->Accepted(), request.communicator) &&
          ExactDoubleConsensus(
              decision->next_dt(), request.communicator);
      if (!decision_collective)
      {
        (void)session.Abort(std::move(authority), native_output);
        return false;
      }

      if (!decision->Accepted())
      {
        // A failed admissibility/closure/correction gate is a controlled
        // adaptive rejection just like an estimator value above one.  The
        // production CPU and CUDA drivers feed both classes through this
        // same transactional controller path.
        const bool controlled_rejection = decision->error_is_valid() &&
            (!decision->collective_gates_satisfied() ||
             decision->normalized_error() > 1.0);
        const bool rejected = session.RejectPrepared(
            std::move(authority), native_output).status ==
            HdgEulerSessionStatus::aborted;
        const bool rollback = rejected && SameSessionCoordinates(
            coordinates_before, session.coordinates());
        if (!CollectiveAll(rollback, request.communicator)) return false;
        try { time_driver->FinalizeAdaptiveRejection(*decision); }
        catch (...) { return false; }
        const bool time_rollback = time_driver->time() == time_before &&
            time_driver->solver_step() == step_before &&
            !time_driver->HasPendingStep();
        if (!CollectiveAll(
                controlled_rejection && time_rollback &&
                    AccumulateArchitectureStep(step, result.work, false),
                request.communicator) ||
            decision->minimum_dt_reached() || !sample_memory())
          return false;
        continue;
      }

      std::optional<HdgTransactionalTimeAdapter::OutputEvent> preview;
      try { preview = time_driver->PreviewFinalizedOutput(); }
      catch (...)
      {
        (void)session.Abort(std::move(authority), native_output);
        return false;
      }
      if (!ExactBoolConsensus(preview.has_value(), request.communicator) ||
          (preview.has_value() != publish_output &&
           (plan.end_time == 0.5 * final_time ||
            plan.end_time == final_time)))
      {
        (void)session.Abort(std::move(authority), native_output);
        return false;
      }
      if (!session.AcknowledgePreparedOutput(authority, native_output))
      {
        (void)session.Abort(std::move(authority), native_output);
        return false;
      }

      std::optional<HdgTransactionalTimeAdapter::OutputEvent> preflight;
      try { preflight = time_driver->PreflightAdaptiveSuccess(*decision); }
      catch (...)
      {
        (void)session.Abort(std::move(authority), native_output);
        return false;
      }
      const bool same_preflight =
          preflight.has_value() == preview.has_value() &&
          (!preview || (preflight->cycle == preview->cycle &&
                        preflight->time == preview->time &&
                        preflight->is_initial == preview->is_initial &&
                        preflight->is_final == preview->is_final));
      if (!CollectiveAll(same_preflight, request.communicator))
      {
        (void)session.Abort(std::move(authority), native_output);
        return false;
      }

      const auto committed_step =
          session.Commit(std::move(authority), native_output);
      if (committed_step.status != HdgEulerSessionStatus::committed ||
          !AccumulateArchitectureStep(step, result.work))
        return false;
      std::optional<HdgTransactionalTimeAdapter::OutputEvent> finalized;
      try { finalized =
          time_driver->FinalizeAdaptiveSuccess(*decision); }
      catch (...) { return false; }
      const bool same_finalized =
          finalized.has_value() == preview.has_value() &&
          (!preview || (finalized->cycle == preview->cycle &&
                        finalized->time == preview->time &&
                        finalized->is_initial == preview->is_initial &&
                        finalized->is_final == preview->is_final));
      const auto committed_coordinates = session.coordinates();
      const bool committed_coordinates_match =
          committed_coordinates.time == time_driver->time() &&
          committed_coordinates.step == time_driver->solver_step() &&
          committed_coordinates.generation ==
              time_driver->solver_step();
      if (!CollectiveAll(
              same_finalized && committed_coordinates_match,
              request.communicator))
        return false;
      if (finalized)
      {
        const bool midpoint = finalized->time == 0.5 * final_time &&
            !finalized->is_initial && !finalized->is_final;
        const bool final = finalized->time == final_time &&
            !finalized->is_initial && finalized->is_final;
        if ((!midpoint && !final) ||
            !publish(finalized->cycle, time_driver->solver_step(),
                     finalized->time))
          return false;
        ++result.output_publications;
        result.midpoint_output_landed =
            result.midpoint_output_landed || midpoint;
        result.final_output_landed = result.final_output_landed || final;
      }
      if (!sample_memory()) return false;
    }

    HdgTransactionalTimeAdapter::AdaptiveControllerState controller{};
    try { controller = time_driver->AdaptiveCheckpointState(); }
    catch (...) { return false; }
    if (controller.accepted_steps >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      return false;
    result.steps = static_cast<int>(controller.accepted_steps);
    result.controller_revision = controller.revision;
    result.prepared_assessment_authority = attempted_steps > 0 &&
        authorized_assessments == attempted_steps;

    const bool revision_accounting = failed_prepares == 0 &&
        controller.accepted_steps ==
            static_cast<std::uint64_t>(result.work.accepted_steps) &&
        controller.rejected_attempts ==
            static_cast<std::uint64_t>(result.work.rejected_attempts) &&
        controller.revision == controller.accepted_steps +
            controller.rejected_attempts &&
        controller.revision == attempted_steps &&
        time_driver->solver_step() == controller.accepted_steps &&
        controller.rejected_attempts > 0 &&
        controller.consecutive_rejections == 0 &&
        controller.last_step_accepted == 1U &&
        !time_driver->HasPendingStep() && time_driver->Finished() &&
        time_driver->time() == final_time;
    const bool output_landing = publish_output
        ? result.output_publications == 3 &&
            result.midpoint_output_landed && result.final_output_landed &&
            time_driver->output_cycle() == 2
        : result.output_publications == 0 &&
            !result.midpoint_output_landed &&
            !result.final_output_landed &&
            time_driver->output_cycle() == 0;

    const int size = committed.Size();
    constexpr int conserved_components =
        dg::hdg::generic::euler_2d_conserved_components;
    const bool extent_valid =
        size > 0 && size % conserved_components == 0;
    const std::size_t scalar_count = extent_valid
        ? static_cast<std::size_t>(size / conserved_components) : 0;
    std::vector<double> sigma(scalar_count);
    if (!extent_valid || !session.ExportCommittedFields({
            committed.HostWrite(), sigma.data(), scalar_count}) ||
        !ValidAdaptiveCommittedMetadata(
            session.coordinates(), session.diagnostics(),
            controller.accepted_steps, final_time))
      return false;

    result.maximum_wave_speed = session.diagnostics().maximum_wave_speed;
    result.field = MeasureGlobalField(committed);
    result.sigma_coefficient_rms = GlobalCoefficientRms(
        sigma, request.communicator);
    return revision_accounting && output_landing &&
        result.prepared_assessment_authority && result.field.valid &&
        result.field.minimum_density > 0.0 &&
        result.field.minimum_pressure > 0.0 &&
        std::isfinite(result.sigma_coefficient_rms) &&
        result.sigma_coefficient_rms > 0.0 &&
        result.work.finite_gate && result.work.admissibility_gate &&
        result.work.recovery_gate && result.work.true_residual_gate &&
        result.work.conservation_gate && result.work.no_fallback_gate &&
        result.work.trace_policy_gate && result.work.common_shift_gate;
  }();

  std::size_t free_after = 0;
  std::size_t total_after = 0;
  const bool final_memory_sample =
      cudaMemGetInfo(&free_after, &total_after) == cudaSuccess &&
      total_after == result.device_total_bytes;
  result.device_free_bytes_after = free_after;
  result.observed_device_free_memory_delta_bytes =
      result.device_free_bytes_before >=
          result.minimum_observed_device_free_bytes
      ? result.device_free_bytes_before -
          result.minimum_observed_device_free_bytes
      : 0;
  result.internal_cuda_used_floor_bytes =
      result.device_total_bytes >= result.minimum_observed_device_free_bytes
      ? result.device_total_bytes -
          result.minimum_observed_device_free_bytes
      : 0;
  const bool synchronized =
      MPI_Barrier(request.communicator) == MPI_SUCCESS;
  const double local_seconds =
      std::chrono::duration<double>(clock::now() - start).count();
  double maximum_seconds = 0.0;
  const bool time_reduced = MPI_Allreduce(
      &local_seconds, &maximum_seconds, 1, MPI_DOUBLE, MPI_MAX,
      request.communicator) == MPI_SUCCESS;
  result.measured_wall_seconds = maximum_seconds;
  result.comparable_wall_seconds = common_setup_seconds + maximum_seconds;
  result.passed = CollectiveAll(
      local && synchronized && time_reduced && final_memory_sample &&
          std::isfinite(result.measured_wall_seconds) &&
          result.measured_wall_seconds > 0.0 &&
          std::isfinite(result.comparable_wall_seconds) &&
          result.comparable_wall_seconds >= result.measured_wall_seconds,
      request.communicator);
  return result.passed;
}

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
  const bool local = session.ExportCommittedInterior(committed) &&
      ValidCommittedMetadata(
          session.coordinates(), session.diagnostics(), steps, dt);
  return CollectiveAll(local, request.communicator);
}

bool AdvanceCuda(const EulerHdgRequest& request, const int steps,
                 const double dt, mfem::ParGridFunction& committed,
                 const CudaTracePolicyKind trace_policy,
                 const CudaQualificationRobustBranch robust_branch =
                     CudaQualificationRobustBranch::not_applicable)
{
  GenericEulerHdgCudaSession session;
  const auto initialization = CudaQualificationTestAccess::InitializeTracePolicy(
      session, request, trace_policy);
  if (initialization.status != HdgEulerSessionStatus::initialized)
  {
    std::cerr << "trace-policy initialization failed status="
              << static_cast<int>(initialization.status) << '\n';
    return false;
  }
  for (int step = 0; step < steps; ++step)
  {
    GenericEulerCudaPreparedStep authority;
    GenericEulerHdgCudaSession::prepared_output_type output{};
    const auto prepared = session.Prepare(dt, authority, output);
    if (prepared.status != HdgEulerSessionStatus::prepared)
    {
      std::cerr << "trace-policy prepare failed step=" << step
                << " status=" << static_cast<int>(prepared.status) << '\n';
      return false;
    }
    const bool qualify_trace_policy =
        trace_policy != CudaTracePolicyKind::engine_default ||
        robust_branch != CudaQualificationRobustBranch::not_applicable;
    if (qualify_trace_policy &&
        !CudaQualificationTestAccess::PreparedTracePolicyValid(
            session, request, trace_policy, robust_branch))
    {
      std::cerr << "trace-policy prepared evidence failed step=" << step
                << '\n';
      return false;
    }
    if (!session.AcknowledgePreparedOutput(authority, output))
    {
      std::cerr << "trace-policy output acknowledgement failed step="
                << step << '\n';
      return false;
    }
    const auto committed = session.Commit(std::move(authority), output);
    if (committed.status != HdgEulerSessionStatus::committed)
    {
      std::cerr << "trace-policy commit failed step=" << step
                << " status=" << static_cast<int>(committed.status) << '\n';
      return false;
    }
  }
  const int size = committed.Size();
  constexpr int components =
      dg::hdg::generic::euler_2d_conserved_components;
  const bool extent_valid = size > 0 && size % components == 0;
  const bool local = extent_valid && session.ExportCommittedInterior({
      committed.HostWrite(),
      static_cast<std::size_t>(size / components), components}) &&
      ValidCommittedMetadata(
          session.coordinates(), session.diagnostics(), steps, dt);
  return CollectiveAll(local, request.communicator);
}

class ThreeEngineRunner
{
public:
  explicit ThreeEngineRunner(
      const CudaTracePolicyKind tensor_igr_trace_policy,
      const EulerHdgTimeTreatment time_treatment) noexcept
      : tensor_igr_trace_policy_(tensor_igr_trace_policy),
        time_treatment_(time_treatment)
  {
  }

  bool Begin(const ExecutionSelection execution)
  {
    if (execution.backend == BackendKind::cpu)
    {
      if (phase_ != 0 || device_) return false;
      phase_ = 1;
      return true;
    }
    if (execution.backend != BackendKind::cuda) return false;
    if (execution.cuda_engine == CudaEngineKind::ordinary)
    {
      if (phase_ != 1 || device_) return false;
      if (HYPRE_ClearAllErrors() != 0 || HYPRE_GetError() != 0)
        return false;
      device_ = std::make_unique<mfem::Device>("cuda");
      phase_ = 2;
      return true;
    }
    if (execution.cuda_engine == CudaEngineKind::tensor_core)
    {
      if (phase_ != 2 || !device_) return false;
      if (HYPRE_ClearAllErrors() != 0 || HYPRE_GetError() != 0)
        return false;
      phase_ = 3;
      return true;
    }
    return false;
  }

  bool Advance(const EulerHdgRequest& request, const int steps,
               const double dt, mfem::ParGridFunction& committed) const
  {
    // Keep one spatial/temporal fixture for both typed Backward-Euler
    // architectures; bind the selection identically for all three engines.
    EulerHdgRequest selected_request = request;
    selected_request.time_treatment = time_treatment_;
    if (selected_request.backend.kind == BackendKind::cpu)
      return phase_ == 1 &&
          AdvanceCpu(selected_request, steps, dt, committed);
    if (selected_request.backend.kind != BackendKind::cuda || phase_ < 2)
      return false;
    const CudaTracePolicyKind trace_policy =
        selected_request.backend.cuda_engine == CudaEngineKind::tensor_core &&
                selected_request.numerical_policy.stabilization.kind ==
                    dg::hdg::generic::RegularizationKind::igr
            ? tensor_igr_trace_policy_
            : CudaTracePolicyKind::engine_default;
    const bool tensor_igr =
        selected_request.backend.cuda_engine == CudaEngineKind::tensor_core &&
        selected_request.numerical_policy.stabilization.kind ==
            dg::hdg::generic::RegularizationKind::igr;
    const CudaTracePolicyDescriptor descriptor = ResolveCudaTracePolicy(
        selected_request.backend.cuda_engine, tensor_igr, 1,
        selected_request.polynomial_order, trace_policy);
    const CudaQualificationRobustBranch robust_branch =
        time_treatment_ == EulerHdgTimeTreatment::backward_euler &&
            CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
                descriptor.resolved_selection)
        ? CudaQualificationRobustBranch::any_qualified
        : CudaQualificationRobustBranch::not_applicable;
    return AdvanceCuda(
        selected_request, steps, dt, committed, trace_policy, robust_branch);
  }

private:
  std::unique_ptr<mfem::Device> device_;
  CudaTracePolicyKind tensor_igr_trace_policy_ =
      CudaTracePolicyKind::engine_default;
  EulerHdgTimeTreatment time_treatment_ =
      EulerHdgTimeTreatment::backward_euler;
  int phase_ = 0;
};

bool RunZeroGradientTensorIgrCase(
    const dg::hdg::test::generic_euler_convergence::CaseSelection& test_case,
    const CudaTracePolicyKind trace_policy,
    const EulerHdgTimeTreatment time_treatment)
{
  using namespace dg::hdg::test::generic_euler_convergence;
  const CudaTracePolicyDescriptor descriptor = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true, 1, test_case.order, trace_policy);
  if (!CudaTracePolicyDescriptorIsExact(descriptor) ||
      (!CudaTracePolicyUsesNativeElementMatrixFreeWarpCoupled(
           descriptor.resolved_selection) &&
       !CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
           descriptor.resolved_selection)) ||
      HYPRE_ClearAllErrors() != 0 || HYPRE_GetError() != 0)
    return false;

  mfem::Device device("cuda");
  mfem::ParMesh mesh = MakePeriodicMesh(3, MPI_COMM_WORLD);
  mfem::L2_FECollection collection(
      test_case.order, dimension, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction initial(&space);
  mfem::ParGridFunction source(&space);
  mfem::ParGridFunction committed(&space);
  source = 0.0;
  const EntropyWave wave{{{0.5, 0.0}}, {{0.0, 1.0}}, 0.1, 1.0};
  BoundaryAudit boundary{};
  if (!ProjectInitialState(wave, initial)) return false;
  const ExecutionSelection execution{
      BackendKind::cuda, CudaEngineKind::tensor_core, "tensor CUDA"};
  EulerHdgRequest request = MakeRequest(
      mesh, initial, source, boundary, execution, test_case,
      test_case.igr_assembly_quadrature_orders);
  request.time_treatment = time_treatment;
  constexpr double dt = 0.02;
  const CudaQualificationRobustBranch robust_branch =
      time_treatment == EulerHdgTimeTreatment::backward_euler &&
          CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
              descriptor.resolved_selection)
      ? CudaQualificationRobustBranch::coupled
      : CudaQualificationRobustBranch::not_applicable;
  const bool advanced = AdvanceCuda(
      request, 1, dt, committed, trace_policy, robust_branch);
  const double norm = advanced ? GlobalL2Norm(committed)
                               : std::numeric_limits<double>::infinity();
  const bool local = advanced && boundary.calls == 0 &&
      std::isfinite(norm) && norm > 0.0;
  const bool passed = CollectiveAll(local, MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::cout << "tensor IGR p" << test_case.order
              << " zero-gradient coupled trace gate policy="
              << CudaTracePolicyKindName(trace_policy)
              << " passed=" << passed << '\n';
  return passed;
}

struct ArchitectureTrajectorySample
{
  ArchitectureTrajectoryResult result{};
  std::unique_ptr<mfem::ParGridFunction> field;
};

double MedianThree(std::array<double, 3> values) noexcept
{
  std::sort(values.begin(), values.end());
  return values[1];
}

void PrintArchitectureTrajectory(
    const ArchitectureTrajectoryResult& result,
    const std::string_view regularization,
    const int order, const int elements_per_direction,
    const double final_time)
{
  std::cout << std::setprecision(17)
            << "ARCHITECTURE_SELECTION_POINT"
            << " method=" << result.method
            << " regularization=" << regularization
            << " order=" << order
            << " elements_per_direction=" << elements_per_direction
            << " final_time=" << final_time
            << " steps=" << result.steps
            << " dt=" << result.dt
            << " adaptive=" << result.adaptive
            << " time_atol=" << result.time_absolute_tolerance
            << " time_rtol=" << result.time_relative_tolerance
            << " adaptive_initial_dt=" << result.adaptive_initial_dt
            << " adaptive_minimum_dt=" << result.adaptive_minimum_dt
            << " controller_revision=" << result.controller_revision
            << " timing_repetitions=" << result.timing_repetitions
            << " measured_session_wall_seconds="
            << result.measured_wall_seconds
            << " common_setup_inclusive_wall_seconds="
            << result.comparable_wall_seconds
            << " measured_u4_global_l2_error="
            << result.measured_u4_l2_error
            << " temporal_reference_error="
            << result.temporal_reference_error
            << " sigma_coefficient_rms="
            << result.sigma_coefficient_rms
            << " accepted_steps=" << result.work.accepted_steps
            << " rejected_attempts=" << result.work.rejected_attempts
            << " nonlinear_updates=" << result.work.nonlinear_updates
            << " linear_iterations=" << result.work.linear_iterations
            << " linear_solve_attempts="
            << result.work.linear_solve_attempts
            << " trace_solves=" << result.work.trace_solves
            << " scalar_closure_solves="
            << result.work.scalar_closure_solves
            << " scalar_closure_linear_iterations="
            << result.work.scalar_closure_linear_iterations
            << " true_residual_tightenings="
            << result.work.true_residual_tightenings
            << " complete_assemblies=" << result.work.complete_assemblies
            << " complete_condensations="
            << result.work.complete_condensations
            << " complete_recoveries=" << result.work.complete_recoveries
            << " common_shift_operator_builds="
            << result.work.common_shift_operator_builds
            << " common_shift_operator_reuses="
            << result.work.common_shift_operator_reuses
            << " condensed_rhs_refreshes="
            << result.work.condensed_rhs_refreshes
            << " common_shift_trace_solves="
            << result.work.common_shift_trace_solves
            << " common_shift_preconditioner_setups="
            << result.work.common_shift_preconditioner_setups
            << " common_shift_preconditioner_reuses="
            << result.work.common_shift_preconditioner_reuses
            << " native_trace_solve_calls="
            << result.work.native_trace_solve_calls
            << " hypre_trace_solve_calls="
            << result.work.hypre_trace_solve_calls
            << " patch_setups=" << result.work.patch_setups
            << " preconditioner_applications="
            << result.work.preconditioner_applications
            << " matrix_free_operator_applications="
            << result.work.matrix_free_operator_applications
            << " maximum_final_residual="
            << result.work.maximum_final_residual
            << " maximum_true_relative_residual="
            << result.work.maximum_true_relative_residual
            << " maximum_conservation_error="
            << result.work.maximum_conservation_error
            << " maximum_conservation_relative_error="
            << result.work.maximum_conservation_relative_error
            << " maximum_embedded_temporal_error_norm="
            << result.work.maximum_temporal_error_norm
            << " maximum_final_algebraic_residual="
            << result.work.maximum_final_algebraic_residual
            << " maximum_final_algebraic_correction_norm="
            << result.work.maximum_final_algebraic_correction_norm
            << " minimum_density=" << result.field.minimum_density
            << " minimum_pressure=" << result.field.minimum_pressure
            << " maximum_wave_speed=" << result.maximum_wave_speed
            << " observed_device_free_memory_delta_bytes="
            << result.observed_device_free_memory_delta_bytes
            << " memory_metric=device_free_delta_not_allocator_peak"
            << " internal_cuda_used_floor_bytes="
            << result.internal_cuda_used_floor_bytes
            << " output_publications=" << result.output_publications
            << " midpoint_output_landed="
            << result.midpoint_output_landed
            << " final_output_landed=" << result.final_output_landed
            << " prepared_assessment_authority="
            << result.prepared_assessment_authority
            << " finite_gate=" << result.work.finite_gate
            << " admissibility_gate=" << result.work.admissibility_gate
            << " recovery_gate=" << result.work.recovery_gate
            << " true_residual_gate=" << result.work.true_residual_gate
            << " conservation_gate=" << result.work.conservation_gate
            << " no_fallback_gate=" << result.work.no_fallback_gate
            << " trace_policy_gate=" << result.work.trace_policy_gate
            << " common_shift_gate=" << result.work.common_shift_gate
            << " passed=" << result.passed << '\n';
}

bool RunArchitectureSelectionScreen(
    const dg::hdg::test::generic_euler_convergence::CaseSelection& test_case,
    const TestSelection& selection)
{
  using namespace dg::hdg::test::generic_euler_convergence;
  using clock = std::chrono::steady_clock;
  // The independent algebraic-correction gate intentionally rejects the
  // very coarse 1/2/4-step trajectories for corrected C_alpha=1 IGR.  Keep
  // the bounded decision ladder inside the admitted regime and use the
  // already-required 64-step numerical reference as its fourth order level;
  // the dedicated CPU manufactured case supplies the broader qualification.
  constexpr std::array<int, 3> rodas_steps{{8, 16, 32}};
  constexpr std::array<int, 6> backward_euler_steps{{2, 4, 8, 16, 32, 64}};
  constexpr std::array<double, 4> adaptive_tolerance_scales{{
      1.0, 0.25, 0.0625, 0.015625}};
  constexpr double floor_separation_factor = 4.0;
  constexpr double rodas_order_floor = 2.2;
  constexpr double rodas_order_ceiling = 3.8;
  constexpr int required_qualifying_orders = 2;

  if (test_case.order != 3 ||
      test_case.regularization !=
          dg::hdg::generic::RegularizationKind::igr ||
      !(selection.screen_final_time > 0.0) ||
      selection.screen_elements_per_direction < 4 ||
      HYPRE_ClearAllErrors() != 0 || HYPRE_GetError() != 0 ||
      MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
    return false;

  const auto common_start = clock::now();
  mfem::Device device("cuda");
  const bool context_ready = cudaFree(nullptr) == cudaSuccess &&
      cudaDeviceSynchronize() == cudaSuccess &&
      HYPRE_ClearAllErrors() == 0 && HYPRE_GetError() == 0;
  mfem::ParMesh mesh = MakePeriodicMesh(
      selection.screen_elements_per_direction, MPI_COMM_WORLD);
  mfem::L2_FECollection collection(
      test_case.order, dimension, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction initial(&space);
  mfem::ParGridFunction source(&space);
  initial.UseDevice(false);
  source.UseDevice(false);
  source = 0.0;
  const ActiveIgrWave wave{};
  const bool projected = ProjectInitialState(wave, initial);
  const double initial_u4_l2_norm = projected
      ? GlobalL2Norm(initial)
      : std::numeric_limits<double>::infinity();
  const GlobalFieldMetrics initial_metrics = projected
      ? MeasureGlobalField(initial) : GlobalFieldMetrics{};
  dg_sim::hdg::ResolvedHdgRegularization resolved_regularization{};
  bool regularization_resolved = false;
  try
  {
    resolved_regularization = dg_sim::hdg::ResolveHdgRegularization(
        mesh, igrlab::IGRAlphaLengthScalePolicy::Average,
        dg_sim::hdg::HdgRegularizationRequest{
            dg::hdg::generic::RegularizationKind::igr, 1.0, 1.0});
    regularization_resolved = true;
  }
  catch (...) {}
  const bool regularization_exact = regularization_resolved &&
      resolved_regularization.selection.kind ==
          dg::hdg::generic::RegularizationKind::igr &&
      resolved_regularization.alpha_info.c_alpha == 1.0 &&
      resolved_regularization.alpha_info.policy ==
          igrlab::IGRAlphaLengthScalePolicy::Average &&
      std::isfinite(resolved_regularization.alpha_info.dx_eff) &&
      resolved_regularization.alpha_info.dx_eff > 0.0 &&
      std::isfinite(resolved_regularization.selection.alpha) &&
      resolved_regularization.selection.alpha > 0.0 &&
      resolved_regularization.selection.beta == 1.0;
  const bool common_ready = CollectiveAll(
      context_ready && projected && regularization_exact &&
          initial_metrics.valid &&
          initial_metrics.minimum_density > 0.0 &&
          initial_metrics.minimum_pressure > 0.0 &&
          std::isfinite(initial_u4_l2_norm) &&
          initial_u4_l2_norm > 0.0,
      MPI_COMM_WORLD);
  if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) return false;
  const double local_common_seconds =
      std::chrono::duration<double>(clock::now() - common_start).count();
  double common_setup_seconds = 0.0;
  const bool common_time_ready = MPI_Allreduce(
      &local_common_seconds, &common_setup_seconds, 1, MPI_DOUBLE, MPI_MAX,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!common_ready || !common_time_ready ||
      !std::isfinite(common_setup_seconds) || common_setup_seconds <= 0.0)
    return false;

  const ExecutionSelection execution{
      BackendKind::cuda, CudaEngineKind::tensor_core, "tensor CUDA"};
  // Compare the RODAS candidate with the same production trace-policy
  // selection that the promoted BE/PBE modes receive.  Explicit selectors
  // remain available through the existing command-line option for bounded
  // ablations, but the default equal-error decision must not silently swap in
  // a benchmark-only solver policy.
  const CudaTracePolicyKind rodas_trace_policy =
      selection.tensor_igr_trace_policy;
  const CudaTracePolicyDescriptor rodas_descriptor = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true, 1, test_case.order,
      rodas_trace_policy);
  if (!CudaTracePolicyDescriptorIsExact(rodas_descriptor)) return false;

  const mfem::ParGridFunction* temporal_reference = nullptr;
  const auto finish_sample = [&](ArchitectureTrajectorySample& sample,
                                 const bool advanced,
                                 const BoundaryAudit& boundary)
  {
    const auto validation_start = clock::now();
    sample.result.temporal_reference_error = temporal_reference != nullptr;
    sample.result.measured_u4_l2_error = advanced
        ? GlobalDifferenceL2(
              *sample.field,
              temporal_reference == nullptr ? initial : *temporal_reference)
        : std::numeric_limits<double>::infinity();
    // Coupled RODAS3 and monolithic BE solve the full Euler-IGR DAE and
    // therefore correctly report no *separate* scalar-closure solves.  Their
    // initialization still performs the required scalar bootstrap.  A
    // finite, nonzero exported sigma is the architecture-neutral proof that
    // the IGR closure is active; per-step method-specific accounting is
    // already required by ReadPreparedArchitectureAggregate.
    const bool active_igr =
        std::isfinite(sample.result.sigma_coefficient_rms) &&
        sample.result.sigma_coefficient_rms > 0.0;
    const double local_validation_seconds =
        std::chrono::duration<double>(
            clock::now() - validation_start).count();
    double validation_seconds = 0.0;
    const bool validation_time_reduced = MPI_Allreduce(
        &local_validation_seconds, &validation_seconds, 1, MPI_DOUBLE,
        MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS;
    if (validation_time_reduced)
    {
      sample.result.measured_wall_seconds += validation_seconds;
      sample.result.comparable_wall_seconds += validation_seconds;
    }
    const bool passed = CollectiveAll(
        advanced && sample.result.passed && boundary.calls == 0 &&
            validation_time_reduced &&
            std::isfinite(sample.result.measured_u4_l2_error) &&
            sample.result.measured_u4_l2_error > 0.0 && active_igr,
        MPI_COMM_WORLD);
    sample.result.passed = passed;
    return passed;
  };

  const auto make_screen_request = [&](BoundaryAudit& boundary)
  {
    EulerHdgRequest request = MakeRequest(
        mesh, initial, source, boundary, execution, test_case,
        test_case.igr_assembly_quadrature_orders);
    request.numerical_policy.stabilization =
        resolved_regularization.selection;
    return request;
  };

  auto run_sample = [&](const std::string_view method,
                        const EulerHdgTimeTreatment treatment,
                        const CudaTracePolicyKind trace_policy,
                        const int steps,
                        ArchitectureTrajectorySample& sample,
                        const std::string_view output_root = {},
                        const std::string_view output_tag = {},
                        const bool publish_midpoint = false)
  {
    sample.field = std::make_unique<mfem::ParGridFunction>(&space);
    sample.field->UseDevice(false);
    BoundaryAudit boundary{};
    EulerHdgRequest request = make_screen_request(boundary);
    const bool advanced = RunTensorArchitectureTrajectory(
        request, mesh, method, treatment, trace_policy, steps,
        selection.screen_final_time, common_setup_seconds, collection,
        *sample.field, sample.result, output_root, output_tag,
        publish_midpoint);
    return finish_sample(sample, advanced, boundary);
  };

  auto run_adaptive_sample = [&] (
      const AdaptiveArchitectureControl control,
      const std::string_view method,
      ArchitectureTrajectorySample& sample,
      const std::string_view output_root = {},
      const std::string_view output_tag = {})
  {
    sample.field = std::make_unique<mfem::ParGridFunction>(&space);
    sample.field->UseDevice(false);
    BoundaryAudit boundary{};
    EulerHdgRequest request = make_screen_request(boundary);
    const bool advanced = RunTensorAdaptiveArchitectureTrajectory(
        request, mesh, method, rodas_trace_policy,
        control, selection.screen_final_time, common_setup_seconds,
        collection, *sample.field, sample.result, output_root, output_tag);
    return finish_sample(sample, advanced, boundary);
  };

  auto run_output_sample = [&] (
      const ArchitectureTrajectoryResult& reference,
      const CudaTracePolicyKind trace_policy,
      const std::string_view output_tag,
      ArchitectureTrajectorySample& sample)
  {
    sample.field = std::make_unique<mfem::ParGridFunction>(&space);
    sample.field->UseDevice(false);
    BoundaryAudit boundary{};
    EulerHdgRequest request = make_screen_request(boundary);
    bool advanced = false;
    if (reference.adaptive)
    {
      const AdaptiveArchitectureControl control{
          reference.time_absolute_tolerance,
          reference.time_relative_tolerance,
          reference.adaptive_initial_dt,
          reference.adaptive_minimum_dt};
      advanced = RunTensorAdaptiveArchitectureTrajectory(
          request, mesh, reference.method, trace_policy, control,
          selection.screen_final_time, common_setup_seconds, collection,
          *sample.field, sample.result, selection.screen_output_root,
          output_tag);
    }
    else
    {
      advanced = RunTensorArchitectureTrajectory(
          request, mesh, reference.method, reference.time_treatment,
          trace_policy, reference.steps, selection.screen_final_time,
          common_setup_seconds, collection, *sample.field, sample.result,
          selection.screen_output_root, output_tag,
          selection.adaptive_rodas3_screen);
    }
    return finish_sample(sample, advanced, boundary);
  };

  if (selection.screen_memory_method != ScreenMemoryMethod::none)
  {
    // A whole-device sampler launches this seam in a fresh process.  Run the
    // exact selected method once, retain the full trajectory gates, and emit
    // one parseable point.  RODAS controls are the exact selected tuple in
    // this mode; unlike the equal-error screen, they are not ladder bases.
    ArchitectureTrajectorySample memory_sample;
    bool advanced = false;
    const bool publish_output = !selection.screen_output_root.empty();
    const std::string_view output_tag = publish_output
        ? std::string_view{"selected-method"} : std::string_view{};
    switch (selection.screen_memory_method)
    {
    case ScreenMemoryMethod::rodas3:
      advanced = run_adaptive_sample(
          AdaptiveArchitectureControl{
              selection.screen_time_atol,
              selection.screen_time_rtol,
              selection.screen_initial_dt,
              selection.screen_dt_min},
          "rodas3", memory_sample, selection.screen_output_root,
          output_tag);
      break;
    case ScreenMemoryMethod::monolithic_backward_euler:
      advanced = run_sample(
          "monolithic_backward_euler",
          EulerHdgTimeTreatment::backward_euler,
          CudaTracePolicyKind::engine_default,
          selection.screen_memory_steps, memory_sample,
          selection.screen_output_root, output_tag, publish_output);
      break;
    case ScreenMemoryMethod::partitioned_backward_euler:
      advanced = run_sample(
          "partitioned_backward_euler",
          EulerHdgTimeTreatment::partitioned_backward_euler,
          CudaTracePolicyKind::engine_default,
          selection.screen_memory_steps, memory_sample,
          selection.screen_output_root, output_tag, publish_output);
      break;
    case ScreenMemoryMethod::none:
      return false;
    }

    auto& result = memory_sample.result;
    const bool rodas = selection.screen_memory_method ==
        ScreenMemoryMethod::rodas3;
    const bool method_contract = rodas
        ? result.method == "rodas3" && result.adaptive &&
            result.time_treatment == EulerHdgTimeTreatment::rodas3 &&
            result.time_absolute_tolerance == selection.screen_time_atol &&
            result.time_relative_tolerance == selection.screen_time_rtol &&
            result.adaptive_initial_dt == selection.screen_initial_dt &&
            result.adaptive_minimum_dt == selection.screen_dt_min &&
            result.steps == result.work.accepted_steps &&
            result.work.rejected_attempts > 0 &&
            result.controller_revision == static_cast<std::uint64_t>(
                result.work.accepted_steps +
                result.work.rejected_attempts) &&
            result.prepared_assessment_authority
        : !result.adaptive && result.steps == selection.screen_memory_steps &&
            result.work.accepted_steps == selection.screen_memory_steps &&
            result.work.rejected_attempts == 0 &&
            ((selection.screen_memory_method == ScreenMemoryMethod::
                  monolithic_backward_euler &&
              result.method == "monolithic_backward_euler" &&
              result.time_treatment ==
                  EulerHdgTimeTreatment::backward_euler) ||
             (selection.screen_memory_method == ScreenMemoryMethod::
                  partitioned_backward_euler &&
              result.method == "partitioned_backward_euler" &&
              result.time_treatment == EulerHdgTimeTreatment::
                  partitioned_backward_euler));
    const bool retained_numerical_gates =
        result.work.finite_gate && result.work.admissibility_gate &&
        result.work.recovery_gate && result.work.true_residual_gate &&
        result.work.conservation_gate && result.work.no_fallback_gate &&
        result.work.trace_policy_gate && result.work.common_shift_gate &&
        result.field.valid && result.field.minimum_density > 0.0 &&
        result.field.minimum_pressure > 0.0 &&
        std::isfinite(result.measured_u4_l2_error) &&
        result.measured_u4_l2_error > 0.0 &&
        std::isfinite(result.sigma_coefficient_rms) &&
        result.sigma_coefficient_rms > 0.0 &&
        std::isfinite(result.work.maximum_final_residual) &&
        std::isfinite(result.work.maximum_true_relative_residual) &&
        std::isfinite(result.work.maximum_conservation_error) &&
        std::isfinite(result.work.maximum_conservation_relative_error) &&
        std::isfinite(result.work.maximum_temporal_error_norm) &&
        std::isfinite(result.work.maximum_final_algebraic_residual) &&
        std::isfinite(
            result.work.maximum_final_algebraic_correction_norm) &&
        std::isfinite(result.maximum_wave_speed) &&
        result.maximum_wave_speed > 0.0;
    const bool memory_floor_gate = result.device_total_bytes > 0 &&
        result.minimum_observed_device_free_bytes <=
            result.device_total_bytes &&
        result.internal_cuda_used_floor_bytes ==
            result.device_total_bytes -
                result.minimum_observed_device_free_bytes;
    const bool output_gate = publish_output
        ? result.output_publications == 3 &&
            result.midpoint_output_landed && result.final_output_landed
        : result.output_publications == 0 &&
            !result.midpoint_output_landed && !result.final_output_landed;
    const bool single_run_gate = result.timing_repetitions == 1 &&
        output_gate &&
        std::isfinite(result.measured_wall_seconds) &&
        result.measured_wall_seconds > 0.0 &&
        std::isfinite(result.comparable_wall_seconds) &&
        result.comparable_wall_seconds >= result.measured_wall_seconds;
    const bool passed = CollectiveAll(
        advanced && result.passed && method_contract &&
            retained_numerical_gates && memory_floor_gate &&
            single_run_gate,
        MPI_COMM_WORLD);
    result.passed = passed;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      PrintArchitectureTrajectory(
          result, RegularizationName(test_case.regularization),
          test_case.order, selection.screen_elements_per_direction,
          selection.screen_final_time);
    return passed;
  }

  if (selection.screen_probe_only)
  {
    ArchitectureTrajectorySample probe;
    const bool advanced = run_sample(
        "rodas3", EulerHdgTimeTreatment::rodas3, rodas_trace_policy,
        rodas_steps.front(), probe);
    const bool passed = CollectiveAll(advanced, MPI_COMM_WORLD);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
      const std::string_view regularization =
          RegularizationName(test_case.regularization);
      std::cout << std::setprecision(17)
                << "ARCHITECTURE_SELECTION_COMMON_SETUP"
                << " regularization=" << regularization
                << " order=" << test_case.order
                << " elements_per_direction="
                << selection.screen_elements_per_direction
                << " final_time=" << selection.screen_final_time
                << " common_mesh_projection_device_setup_seconds="
                << common_setup_seconds
                << " initial_u4_l2_norm="
                << initial_u4_l2_norm
                << " igr_c_alpha=1"
                << " igr_alpha_policy=average"
                << " resolved_igr_alpha="
                << resolved_regularization.selection.alpha
                << " igr_trace_penalty="
                << resolved_regularization.selection.beta
                << " output_enabled=0 probe_only=1\n";
      PrintArchitectureTrajectory(
          probe.result, regularization, test_case.order,
          selection.screen_elements_per_direction,
          selection.screen_final_time);
      std::cout << "ARCHITECTURE_SELECTION_PROBE_GATE passed="
                << passed << '\n';
    }
    return passed;
  }

  // The active IGR state has no manufactured exact evolution.  Build one
  // bounded, same-space numerical reference with the candidate method before
  // timing the decision ladders.  The 32/64-step difference supplies a
  // third-order Richardson uncertainty estimate, and selection requires a
  // four-times separation from it.  No spatial projection error is mixed
  // into the temporal comparison because every trajectory uses this exact
  // mesh and finite-element space.
  ArchitectureTrajectorySample reference_coarse;
  ArchitectureTrajectorySample reference_fine;
  const bool reference_coarse_advanced = run_sample(
      "rodas3_reference_32", EulerHdgTimeTreatment::rodas3,
      rodas_trace_policy, 32, reference_coarse);
  if (!reference_coarse_advanced)
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
      std::cerr << "architecture-screen numerical reference failed "
                   "level=coarse\n";
      PrintArchitectureTrajectory(
          reference_coarse.result,
          RegularizationName(test_case.regularization), test_case.order,
          selection.screen_elements_per_direction,
          selection.screen_final_time);
    }
    return false;
  }
  const bool reference_fine_advanced = run_sample(
      "rodas3_reference_64", EulerHdgTimeTreatment::rodas3,
      rodas_trace_policy, 64, reference_fine);
  if (!reference_fine_advanced)
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
      std::cerr << "architecture-screen numerical reference failed "
                   "level=fine\n";
      PrintArchitectureTrajectory(
          reference_fine.result,
          RegularizationName(test_case.regularization), test_case.order,
          selection.screen_elements_per_direction,
          selection.screen_final_time);
    }
    return false;
  }
  const double reference_difference = GlobalDifferenceL2(
      *reference_coarse.field, *reference_fine.field);
  const double reference_uncertainty = reference_difference / 7.0;
  const bool reference_gate = CollectiveAll(
      reference_coarse.result.passed && reference_fine.result.passed &&
          std::isfinite(reference_coarse.result.sigma_coefficient_rms) &&
          reference_coarse.result.sigma_coefficient_rms > 0.0 &&
          std::isfinite(reference_fine.result.sigma_coefficient_rms) &&
          reference_fine.result.sigma_coefficient_rms > 0.0 &&
          std::isfinite(reference_difference) &&
          reference_difference > 0.0 &&
          std::isfinite(reference_uncertainty) &&
          reference_uncertainty > 0.0,
      MPI_COMM_WORLD);
  if (!reference_gate) return false;
  temporal_reference = reference_fine.field.get();

  std::vector<ArchitectureTrajectorySample> rodas;
  std::vector<ArchitectureTrajectorySample> backward_euler;
  std::vector<ArchitectureTrajectorySample> partitioned_backward_euler;
  rodas.reserve(std::max(
      rodas_steps.size(), adaptive_tolerance_scales.size()));
  backward_euler.reserve(backward_euler_steps.size());
  partitioned_backward_euler.reserve(backward_euler_steps.size());
  if (selection.adaptive_rodas3_screen)
  {
    for (const double scale : adaptive_tolerance_scales)
    {
      rodas.emplace_back();
      const AdaptiveArchitectureControl control{
          scale * selection.screen_time_atol,
          scale * selection.screen_time_rtol,
          selection.screen_initial_dt,
          selection.screen_dt_min};
      if (!run_adaptive_sample(
              control, "adaptive_rodas3", rodas.back()))
      {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
        {
          std::cerr << "architecture-screen adaptive trajectory failed "
                       "tolerance_scale="
                    << scale << '\n';
          PrintArchitectureTrajectory(
              rodas.back().result,
              RegularizationName(test_case.regularization),
              test_case.order, selection.screen_elements_per_direction,
              selection.screen_final_time);
        }
        return false;
      }
    }
  }
  else
  {
    for (const int steps : rodas_steps)
    {
      rodas.emplace_back();
      if (!run_sample("rodas3", EulerHdgTimeTreatment::rodas3,
                      rodas_trace_policy, steps, rodas.back()))
        return false;
    }
  }
  for (const int steps : backward_euler_steps)
  {
    backward_euler.emplace_back();
    if (!run_sample("monolithic_backward_euler",
                    EulerHdgTimeTreatment::backward_euler,
                    CudaTracePolicyKind::engine_default, steps,
                    backward_euler.back()))
      return false;
  }
  for (const int steps : backward_euler_steps)
  {
    partitioned_backward_euler.emplace_back();
    if (!run_sample("partitioned_backward_euler",
                    EulerHdgTimeTreatment::partitioned_backward_euler,
                    CudaTracePolicyKind::engine_default, steps,
                    partitioned_backward_euler.back()))
      return false;
  }

  if (rodas.size() < 2) return false;
  std::vector<double> rodas_self_differences(rodas.size() - 1);
  for (std::size_t level = 0; level < rodas_self_differences.size(); ++level)
    rodas_self_differences[level] = GlobalDifferenceL2(
        *rodas[level].field, *rodas[level + 1].field);
  if (!selection.adaptive_rodas3_screen)
    rodas_self_differences.push_back(reference_difference);
  std::vector<double> rodas_observed_orders;
  if (!selection.adaptive_rodas3_screen)
  {
    rodas_observed_orders.resize(rodas_self_differences.size() - 1);
    for (std::size_t level = 0;
         level < rodas_observed_orders.size(); ++level)
      rodas_observed_orders[level] = HalvingOrder(
          rodas_self_differences[level],
          rodas_self_differences[level + 1]);
  }
  const int qualifying_orders = static_cast<int>(std::count_if(
      rodas_observed_orders.begin(), rodas_observed_orders.end(),
      [](const double value)
      {
        return std::isfinite(value) && value >= rodas_order_floor &&
            value <= rodas_order_ceiling;
      }));
  const bool positive_self_differences =
      std::all_of(rodas_self_differences.begin(),
                  rodas_self_differences.end(),
                  [](const double value)
                  { return std::isfinite(value) && value > 0.0; });
  const bool rodas_order_pass = positive_self_differences &&
      qualifying_orders >= required_qualifying_orders;
  bool strict_adaptive_tolerance_ladder =
      selection.adaptive_rodas3_screen;
  for (std::size_t level = 1; level < rodas.size(); ++level)
    strict_adaptive_tolerance_ladder =
        strict_adaptive_tolerance_ladder &&
        rodas[level].result.time_absolute_tolerance <
            rodas[level - 1].result.time_absolute_tolerance &&
        rodas[level].result.time_relative_tolerance <
            rodas[level - 1].result.time_relative_tolerance;
  const bool adaptive_controller_ladder_pass =
      selection.adaptive_rodas3_screen &&
      strict_adaptive_tolerance_ladder && positive_self_differences &&
      std::all_of(
          rodas.begin(), rodas.end(),
          [](const ArchitectureTrajectorySample& sample)
          {
            const auto& result = sample.result;
            return result.passed && result.adaptive &&
                result.prepared_assessment_authority &&
                result.work.accepted_steps > 0 &&
                result.work.rejected_attempts > 0 &&
                result.controller_revision ==
                    static_cast<std::uint64_t>(
                        result.work.accepted_steps +
                        result.work.rejected_attempts);
          });
  const double finest_self_difference = rodas_self_differences.back();
  bool strictly_decreasing_reference_errors = true;
  for (std::size_t level = 1; level < rodas.size(); ++level)
    strictly_decreasing_reference_errors =
        strictly_decreasing_reference_errors &&
        std::isfinite(rodas[level].result.measured_u4_l2_error) &&
        rodas[level].result.measured_u4_l2_error <
            rodas[level - 1].result.measured_u4_l2_error;
  const bool reference_sentinel_valid = reference_gate &&
      std::isfinite(reference_uncertainty) &&
      reference_uncertainty > 0.0;
  const bool rodas_method_pass = strictly_decreasing_reference_errors &&
      (selection.adaptive_rodas3_screen
           ? adaptive_controller_ladder_pass : rodas_order_pass);
  const std::size_t rodas_decision_levels = rodas.size();

  const auto fastest_within = [](auto& samples, const double tolerance,
                                 const std::size_t count)
      -> ArchitectureTrajectorySample*
  {
    ArchitectureTrajectorySample* selected = nullptr;
    const std::size_t bounded_count = std::min(count, samples.size());
    for (std::size_t index = 0; index < bounded_count; ++index)
    {
      auto& candidate = samples[index];
      if (!candidate.result.passed ||
          candidate.result.measured_u4_l2_error > tolerance)
        continue;
      if (selected == nullptr ||
          candidate.result.comparable_wall_seconds <
              selected->result.comparable_wall_seconds)
        selected = &candidate;
    }
    return selected;
  };

  std::array<ArchitectureTrajectorySample*, 3> selected{{
      nullptr, nullptr, nullptr}};
  double common_error_tolerance = std::numeric_limits<double>::infinity();
  // Select the strictest measured, floor-separated tolerance reached by all
  // architectures.  Restricting candidates to RODAS3 error levels makes the
  // screen fail precisely when RODAS3 outruns the bounded control ladders: a
  // finer RODAS3 result is still a valid result at the coarser shared error
  // tolerance.  Drawing candidates from every trajectory keeps the rule
  // architecture-neutral while fastest_within preserves equal-tolerance
  // comparison.
  std::vector<double> measured_tolerances;
  measured_tolerances.reserve(
      rodas_decision_levels + backward_euler.size() +
      partitioned_backward_euler.size());
  const auto append_tolerances = [&measured_tolerances](
      const auto& samples, const std::size_t count)
  {
    const std::size_t bounded_count = std::min(count, samples.size());
    for (std::size_t index = 0; index < bounded_count; ++index)
      if (std::isfinite(samples[index].result.measured_u4_l2_error) &&
          samples[index].result.measured_u4_l2_error > 0.0)
        measured_tolerances.push_back(
            samples[index].result.measured_u4_l2_error *
            (1.0 + 64.0 * std::numeric_limits<double>::epsilon()));
  };
  append_tolerances(rodas, rodas_decision_levels);
  append_tolerances(backward_euler, backward_euler.size());
  append_tolerances(
      partitioned_backward_euler, partitioned_backward_euler.size());
  std::sort(measured_tolerances.begin(), measured_tolerances.end());
  for (const double tolerance : measured_tolerances)
  {
    if (!(tolerance >
          floor_separation_factor * reference_uncertainty))
      continue;
    auto* const selected_rodas = fastest_within(
        rodas, tolerance, rodas_decision_levels);
    auto* const selected_backward_euler = fastest_within(
        backward_euler, tolerance, backward_euler.size());
    auto* const selected_partitioned = fastest_within(
        partitioned_backward_euler, tolerance,
        partitioned_backward_euler.size());
    if (selected_rodas != nullptr && selected_backward_euler != nullptr &&
        selected_partitioned != nullptr)
    {
      common_error_tolerance = tolerance;
      selected = {{selected_backward_euler, selected_partitioned,
                   selected_rodas}};
      break;
    }
  }
  const bool equal_error_selected = std::all_of(
      selected.begin(), selected.end(),
      [](const ArchitectureTrajectorySample* sample)
      { return sample != nullptr; });

  if (equal_error_selected)
  {
    std::array<std::array<double, 3>, 3> measured_times{};
    constexpr std::array<std::array<std::size_t, 3>, 3> repeat_order{{
        {{2, 0, 1}}, // RODAS3, monolithic BE, partitioned BE.
        {{0, 1, 2}}, // Monolithic BE, partitioned BE, RODAS3.
        {{1, 2, 0}}  // Partitioned BE, RODAS3, monolithic BE.
    }};
    for (std::size_t repetition = 0;
         repetition < repeat_order.size(); ++repetition)
      for (const std::size_t method : repeat_order[repetition])
      {
        const ArchitectureTrajectoryResult reference =
            selected[method]->result;
        ArchitectureTrajectorySample repeated;
        const CudaTracePolicyKind trace_policy =
            reference.time_treatment == EulerHdgTimeTreatment::rodas3
            ? rodas_trace_policy : CudaTracePolicyKind::engine_default;
        const bool repeated_ok = reference.adaptive
            ? run_adaptive_sample(
                  AdaptiveArchitectureControl{
                      reference.time_absolute_tolerance,
                      reference.time_relative_tolerance,
                      reference.adaptive_initial_dt,
                      reference.adaptive_minimum_dt},
                  reference.method,
                  repeated)
            : run_sample(reference.method, reference.time_treatment,
                         trace_policy, reference.steps, repeated);
        if (!repeated_ok)
          return false;
        const double error_scale = std::max(
            reference.measured_u4_l2_error,
            repeated.result.measured_u4_l2_error);
        const bool same_adaptive_path = !reference.adaptive ||
            (repeated.result.adaptive &&
             repeated.result.steps == reference.steps &&
             repeated.result.work.rejected_attempts ==
                 reference.work.rejected_attempts &&
             repeated.result.controller_revision ==
                 reference.controller_revision);
        if (!same_adaptive_path ||
            std::abs(reference.measured_u4_l2_error -
                     repeated.result.measured_u4_l2_error) >
                1.0e-13 + 1.0e-9 * error_scale ||
            repeated.result.measured_u4_l2_error >
                common_error_tolerance)
          return false;
        measured_times[method][repetition] =
            repeated.result.measured_wall_seconds;
        selected[method]->result.observed_device_free_memory_delta_bytes =
            std::max(
                selected[method]->result.
                    observed_device_free_memory_delta_bytes,
                repeated.result.observed_device_free_memory_delta_bytes);
      }
    for (std::size_t method = 0; method < selected.size(); ++method)
    {
      const double median = MedianThree(measured_times[method]);
      selected[method]->result.measured_wall_seconds = median;
      selected[method]->result.comparable_wall_seconds =
          common_setup_seconds + median;
      selected[method]->result.timing_repetitions = 3;
    }
  }

  std::array<double, 3> output_inclusive_walls{{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()}};
  bool output_screen_pass = selection.screen_output_root.empty();
  if (!selection.screen_output_root.empty() && equal_error_selected)
  {
    std::array<std::array<double, 3>, 3> measured_times{};
    std::array<std::optional<ArchitectureTrajectoryResult>, 3>
        output_trajectory_references{};
    constexpr std::array<std::array<std::size_t, 3>, 3> paired_order{{
        {{2, 0, 1}}, // RODAS3, monolithic BE, partitioned BE.
        {{0, 1, 2}}, // Monolithic BE, partitioned BE, RODAS3.
        {{1, 2, 0}}  // Partitioned BE, RODAS3, monolithic BE.
    }};
    output_screen_pass = true;
    for (std::size_t repetition = 0;
         output_screen_pass && repetition < paired_order.size(); ++repetition)
      for (const std::size_t method : paired_order[repetition])
      {
        const ArchitectureTrajectoryResult reference =
            selected[method]->result;
        const CudaTracePolicyKind trace_policy =
            reference.time_treatment == EulerHdgTimeTreatment::rodas3
            ? rodas_trace_policy : CudaTracePolicyKind::engine_default;
        const std::string output_tag =
            "equal_error_" + std::to_string(method) + "_repeat_" +
            std::to_string(repetition);
        ArchitectureTrajectorySample repeated;
        if (!run_output_sample(
                reference, trace_policy, output_tag, repeated))
        {
          output_screen_pass = false;
          break;
        }
        // Output-time landing is part of the timed workload and may shorten
        // an adaptive step.  Qualify repeatability against the first output
        // trajectory for this method, rather than requiring its controller
        // path to equal the otherwise-equivalent no-output trajectory.
        const auto& output_reference =
            output_trajectory_references[method];
        const double error_scale = output_reference
            ? std::max(output_reference->measured_u4_l2_error,
                       repeated.result.measured_u4_l2_error)
            : repeated.result.measured_u4_l2_error;
        const bool same_error = !output_reference || std::abs(
            output_reference->measured_u4_l2_error -
            repeated.result.measured_u4_l2_error) <=
            1.0e-13 + 1.0e-9 * error_scale;
        const std::uint64_t expected_publications =
            selection.adaptive_rodas3_screen ? 3 : 2;
        const bool landing = repeated.result.output_publications ==
                expected_publications &&
            repeated.result.final_output_landed &&
            (selection.adaptive_rodas3_screen
                 ? repeated.result.midpoint_output_landed
                 : !repeated.result.midpoint_output_landed);
        const bool same_output_path = !output_reference ||
            (repeated.result.adaptive == output_reference->adaptive &&
             repeated.result.steps == output_reference->steps &&
             repeated.result.work.rejected_attempts ==
                 output_reference->work.rejected_attempts &&
             repeated.result.controller_revision ==
                 output_reference->controller_revision);
        const bool output_trajectory_pass =
            same_error && same_output_path && landing &&
            repeated.result.measured_u4_l2_error <=
                common_error_tolerance &&
            std::isfinite(repeated.result.comparable_wall_seconds) &&
            repeated.result.comparable_wall_seconds > 0.0;
        if (!output_trajectory_pass)
        {
          output_screen_pass = false;
          break;
        }
        if (!output_reference)
          output_trajectory_references[method] = repeated.result;
        measured_times[method][repetition] =
            repeated.result.comparable_wall_seconds;
      }
    if (output_screen_pass)
      for (std::size_t method = 0; method < selected.size(); ++method)
      {
        output_inclusive_walls[method] =
            MedianThree(measured_times[method]);
        output_screen_pass = output_screen_pass &&
            std::isfinite(output_inclusive_walls[method]) &&
            output_inclusive_walls[method] > 0.0;
      }
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
  {
    const std::string_view regularization =
        RegularizationName(test_case.regularization);
    std::cout << std::setprecision(17)
              << "ARCHITECTURE_SELECTION_COMMON_SETUP"
              << " regularization=" << regularization
              << " order=" << test_case.order
              << " elements_per_direction="
              << selection.screen_elements_per_direction
              << " final_time=" << selection.screen_final_time
              << " common_mesh_projection_device_setup_seconds="
              << common_setup_seconds
              << " initial_u4_l2_norm=" << initial_u4_l2_norm
              << " igr_c_alpha=1"
              << " igr_alpha_policy=average"
              << " resolved_igr_dx="
              << resolved_regularization.alpha_info.dx_eff
              << " resolved_igr_alpha="
              << resolved_regularization.selection.alpha
              << " igr_trace_penalty="
              << resolved_regularization.selection.beta
              << " adaptive_rodas3="
              << selection.adaptive_rodas3_screen
              << " output_enabled=0\n";
    for (const auto& sample : rodas)
      PrintArchitectureTrajectory(
          sample.result, regularization, test_case.order,
          selection.screen_elements_per_direction,
          selection.screen_final_time);
    for (const auto& sample : backward_euler)
      PrintArchitectureTrajectory(
          sample.result, regularization, test_case.order,
          selection.screen_elements_per_direction,
          selection.screen_final_time);
    for (const auto& sample : partitioned_backward_euler)
      PrintArchitectureTrajectory(
          sample.result, regularization, test_case.order,
          selection.screen_elements_per_direction,
          selection.screen_final_time);
    std::cout << (selection.adaptive_rodas3_screen
                      ? "ARCHITECTURE_SELECTION_ADAPTIVE_RODAS_LADDER"
                      : "ARCHITECTURE_SELECTION_RODAS_ORDER")
              << " self_differences=";
    for (const double value : rodas_self_differences)
      std::cout << value << ',';
    if (selection.adaptive_rodas3_screen)
    {
      std::cout << " tolerance_scales=";
      for (const double value : adaptive_tolerance_scales)
        std::cout << value << ',';
      std::cout << " strict_tolerance_ladder="
                << strict_adaptive_tolerance_ladder
                << " strict_reference_error_ladder="
                << strictly_decreasing_reference_errors
                << " estimator_rejection_required=1"
                << " exact_revision_accounting_required=1"
                << " passed=" << rodas_method_pass << '\n';
    }
    else
    {
      std::cout << " observed_orders=";
      for (const double value : rodas_observed_orders)
        std::cout << value << ',';
      std::cout << " required_order_interval=[" << rodas_order_floor << ','
                << rodas_order_ceiling << ']'
                << " qualifying_orders=" << qualifying_orders
                << " required_qualifying_orders="
                << required_qualifying_orders
                << " strict_reference_error_ladder="
                << strictly_decreasing_reference_errors
                << " passed=" << rodas_method_pass << '\n';
    }
    std::cout
              << "ARCHITECTURE_SELECTION_NUMERICAL_REFERENCE"
              << " coarse_steps=" << reference_coarse.result.steps
              << " fine_steps=" << reference_fine.result.steps
              << " same_mesh=1 same_space=1"
              << " reference_method=rodas3"
              << " reference_difference=" << reference_difference
              << " richardson_order=3"
              << " reference_uncertainty=" << reference_uncertainty
              << " coarse_sigma_coefficient_rms="
              << reference_coarse.result.sigma_coefficient_rms
              << " fine_sigma_coefficient_rms="
              << reference_fine.result.sigma_coefficient_rms
              << " coarse_scalar_closure_solves="
              << reference_coarse.result.work.scalar_closure_solves
              << " fine_scalar_closure_solves="
              << reference_fine.result.work.scalar_closure_solves
              << " active_igr_gate=" << reference_gate
              << " passed=" << reference_sentinel_valid << '\n';
    std::cout
              << "ARCHITECTURE_SELECTION_REFERENCE_FLOOR_SENTINEL"
              << " finest_decision_steps=" << rodas.back().result.steps
              << " finest_decision_u4_l2_error="
              << rodas.back().result.measured_u4_l2_error
              << " finest_decision_self_difference="
              << finest_self_difference
              << " numerical_reference_uncertainty="
              << reference_uncertainty
              << " common_tolerance=" << common_error_tolerance
              << " floor_separation_factor=" << floor_separation_factor
              << " floor_limited="
              << (!equal_error_selected ||
                  !(common_error_tolerance >
                    floor_separation_factor * reference_uncertainty))
              << " passed=" << reference_sentinel_valid << '\n';
    if (equal_error_selected)
    {
      const double be_wall =
          selected[0]->result.comparable_wall_seconds;
      const double pbe_wall =
          selected[1]->result.comparable_wall_seconds;
      const double rodas_wall =
          selected[2]->result.comparable_wall_seconds;
      std::cout << "ARCHITECTURE_SELECTION_EQUAL_ERROR"
                << " common_u4_l2_tolerance=" << common_error_tolerance
                << " monolithic_steps=" << selected[0]->result.steps
                << " monolithic_error="
                << selected[0]->result.measured_u4_l2_error
                << " monolithic_wall_seconds=" << be_wall
                << " partitioned_steps=" << selected[1]->result.steps
                << " partitioned_error="
                << selected[1]->result.measured_u4_l2_error
                << " partitioned_wall_seconds=" << pbe_wall
                << " rodas3_steps=" << selected[2]->result.steps
                << " rodas3_error="
                << selected[2]->result.measured_u4_l2_error
                << " rodas3_adaptive=" << selected[2]->result.adaptive
                << " rodas3_time_atol="
                << selected[2]->result.time_absolute_tolerance
                << " rodas3_time_rtol="
                << selected[2]->result.time_relative_tolerance
                << " rodas3_initial_dt="
                << selected[2]->result.adaptive_initial_dt
                << " rodas3_dt_min="
                << selected[2]->result.adaptive_minimum_dt
                << " rodas3_rejected_attempts="
                << selected[2]->result.work.rejected_attempts
                << " rodas3_controller_revision="
                << selected[2]->result.controller_revision
                << " rodas3_wall_seconds=" << rodas_wall
                << " rodas3_speedup_vs_monolithic="
                << be_wall / rodas_wall
                << " rodas3_speedup_vs_partitioned="
                << pbe_wall / rodas_wall
                << " timing_repetitions=3"
                << " paired_repeat_order="
                   "repeat1_rodas_be_pbe,repeat2_be_pbe_rodas,"
                   "repeat3_pbe_rodas_be"
                << " common_setup_included=1"
                << " session_setup_included=1"
                << " validation_and_synchronization_included=1"
                << " process_startup_included=0"
                << " rodas3_prepared_assessment_authority="
                << selected[2]->result.prepared_assessment_authority
                << '\n';
      if (!selection.screen_output_root.empty())
        std::cout << "ARCHITECTURE_SELECTION_EQUAL_ERROR_WITH_OUTPUT"
                  << " common_u4_l2_tolerance=" << common_error_tolerance
                  << " monolithic_steps=" << selected[0]->result.steps
                  << " monolithic_wall_seconds="
                  << output_inclusive_walls[0]
                  << " partitioned_steps=" << selected[1]->result.steps
                  << " partitioned_wall_seconds="
                  << output_inclusive_walls[1]
                  << " rodas3_steps=" << selected[2]->result.steps
                  << " rodas3_wall_seconds="
                  << output_inclusive_walls[2]
                  << " rodas3_speedup_vs_monolithic="
                  << output_inclusive_walls[0] / output_inclusive_walls[2]
                  << " rodas3_speedup_vs_partitioned="
                  << output_inclusive_walls[1] / output_inclusive_walls[2]
                  << " timing_repetitions=3"
                  << " publications_per_trajectory="
                  << (selection.adaptive_rodas3_screen ? 3 : 2)
                  << " publication_cycles="
                  << (selection.adaptive_rodas3_screen
                          ? "initial,midpoint,final" : "initial,final")
                  << " common_setup_included=1"
                  << " session_setup_included=1"
                  << " validation_and_synchronization_included=1"
                  << " process_startup_included=0"
                  << " output_gate=" << output_screen_pass << '\n';
    }
  }

  const bool floor_separated = equal_error_selected &&
      common_error_tolerance >
          floor_separation_factor * reference_uncertainty;
  const bool passed = CollectiveAll(
      rodas_method_pass && reference_sentinel_valid && floor_separated &&
          output_screen_pass,
      MPI_COMM_WORLD);
  if (rank == 0)
    std::cout << "ARCHITECTURE_SELECTION_GATE passed=" << passed << '\n';
  return passed;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  dg::hdg::test::generic_euler_convergence::CaseSelection test_case{};
  TestSelection selection{};
  const bool parsed = ParseTestSelection(
      argc, argv, test_case, selection);
  if (!parsed || ranks != 1)
  {
    if (ranks == 1)
      std::cerr << "Usage: " << argv[0]
                << " --order 1|2|3 --regularization conventional|igr"
                   " [--igr-volume-quadrature-order Q"
                   " --igr-face-quadrature-order Q]"
                   " [--newton-linear-rtol R]"
                   " [--final-linear-rtol R]"
                   " [--true-residual-tightening-linear-rtol R]"
                   " [--architecture-selection-screen"
                   " --adaptive-rodas3-screen"
                   " --screen-probe-only"
                   " --screen-elements 4..32"
                   " --screen-final-time T"
                   " --screen-time-atol A"
                   " --screen-time-rtol R"
                   " --screen-initial-dt DT"
                   " --screen-dt-min DT_MIN"
                   " --screen-memory-method rodas3|"
                   "monolithic_backward_euler|"
                   "partitioned_backward_euler"
                   " --screen-memory-steps N"
                   " --screen-output-root PATH]"
                   " [--time-treatment backward-euler|"
                   "partitioned-backward-euler]"
                   " [--tensor-igr-trace-policy "
                   "native-element-schwarz-matrix-free-warp-coupled-fp64|"
                   "native-element-schwarz-matrix-free-warp-coupled-sigma-"
                   "jacobi2-fp64|native-element-schwarz-matrix-free-warp-"
                   "coupled-sigma-jacobi10-fp64|"
                   "native-element-schwarz-matrix-free-warp-robust-"
                   "coupled-sigma-jacobi10-fp64]"
                   " [--zero-gradient-only]\n";
    else
      std::cerr << "Three-engine convergence qualification is rank-one; "
                   "rank-two CUDA correctness has separate coverage\n";
    return 2;
  }

  if (selection.zero_gradient_only)
    return RunZeroGradientTensorIgrCase(
        test_case, selection.tensor_igr_trace_policy,
        selection.time_treatment) ? 0 : 1;

  if (selection.architecture_selection_screen)
  {
    const char* const enabled =
        ::igrlab::GetEnv("DG_IGR_RUN_ARCHITECTURE_SELECTION_SCREEN");
    if (enabled == nullptr || std::string_view{enabled} != "1")
    {
      std::cout << "ARCHITECTURE_SELECTION_GATE skipped=1 reason="
                   "explicit_environment_opt_in_required\n";
      return 77;
    }
    return RunArchitectureSelectionScreen(test_case, selection) ? 0 : 1;
  }

  const std::array<ExecutionSelection, 3> executions{{
      {BackendKind::cpu, CudaEngineKind::ordinary, "CPU"},
      {BackendKind::cuda, CudaEngineKind::ordinary, "ordinary CUDA"},
      {BackendKind::cuda, CudaEngineKind::tensor_core, "tensor CUDA"}}};
  ThreeEngineRunner runner(
      selection.tensor_igr_trace_policy, selection.time_treatment);
  const bool passed =
      dg::hdg::test::generic_euler_convergence::RunComparisonCase(
          MPI_COMM_WORLD, test_case, executions, runner);
  return passed ? 0 : 1;
}

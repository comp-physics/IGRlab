#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>

namespace
{

using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::app;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "application contract test must compile in strict C++17");
static_assert(!std::is_default_constructible<ValidatedApplicationConfig>::value &&
                  !std::is_aggregate<ValidatedApplicationConfig>::value,
              "validated configuration has no construction bypass");
static_assert(smooth_periodic_euler_v1_polynomial_order == 1 &&
                  smooth_periodic_euler_v1_dt == 0.05 &&
                  smooth_periodic_euler_v1_minimum_steps == 20 &&
                  smooth_periodic_euler_v1_checkpoint_step == 10 &&
                  smooth_periodic_euler_v1_failure_step == 14 &&
                  smooth_periodic_euler_v1_failure_phase ==
                      FailurePhase::AfterRecoveryBeforeCommit &&
                  smooth_periodic_euler_v1_retry_limit == 1 &&
                  smooth_periodic_euler_v1_source_rank == 0,
              "smooth periodic Euler v1 constants are frozen");
static_assert(IsFrozenApplicationBackend<CpuSerialBackendTag>::value &&
                  IsFrozenApplicationBackend<CpuRank2BackendTag>::value &&
                  IsFrozenApplicationBackend<
                      CudaRank1OneGpuBackendTag>::value &&
                  IsFrozenApplicationBackend<
                      CudaRank2OneGpuHostStagedBackendTag>::value &&
                  !IsFrozenApplicationBackend<int>::value,
              "the application boundary is closed over four backends");

bool Check(const bool condition, const char* name)
{
  if (!condition) { std::cerr << "FAILED: " << name << '\n'; }
  return condition;
}

ApplicationOptionsView OptionsFor(const std::string_view backend)
{
  ApplicationOptionsView options;
  options.backend_id = backend;
  if (backend == CpuRank2BackendTag::id ||
      backend == CudaRank2OneGpuHostStagedBackendTag::id)
  {
    options.observed.mpi_rank_count = 2;
  }
  if (backend == CudaRank1OneGpuBackendTag::id ||
      backend == CudaRank2OneGpuHostStagedBackendTag::id)
  {
    options.observed.participating_gpu_count = 1;
    options.observed.selected_gpu_ordinal = 0;
    options.observed.selected_gpu_identity = "GPU-frozen-observed-identity";
    options.observed.all_ranks_share_one_gpu = true;
  }
  return options;
}

bool CheckCompiledAvailabilityAndFourShapes()
{
  struct Case
  {
    std::string_view backend;
    ApplicationBackend kind;
  };
  const std::array<Case, 4> cases{{
      {CpuSerialBackendTag::id, ApplicationBackend::cpu_serial},
      {CpuRank2BackendTag::id, ApplicationBackend::cpu_rank2},
      {CudaRank1OneGpuBackendTag::id,
       ApplicationBackend::cuda_rank1_one_gpu},
      {CudaRank2OneGpuHostStagedBackendTag::id,
       ApplicationBackend::cuda_rank2_one_gpu_host_staged}}};
  bool passed = true;
  for (const Case& item : cases)
  {
    ApplicationOptionsView options = OptionsFor(item.backend);
    const ApplicationValidation validation =
        ValidateApplicationOptions(options);
    if (CompiledBackendAvailable(item.kind))
    {
      const ValidatedApplicationConfig& config = validation.config();
      passed = Check(
                   validation.accepted() &&
                       validation.rejection() == ApplicationRejection::none &&
                       config.backend() == item.kind &&
                       config.tuple().backend.view() == item.backend &&
                       config.interior_space() == InteriorSpace::l2 &&
                       config.polynomial_order() == 1 && config.mesh_n() == 6 &&
                       config.step_bound() >= 20 && config.dt() == 0.05 &&
                       config.output_steps() ==
                           std::array<std::uint64_t, 3>{{0, 10, 20}} &&
                       config.card_id().view() ==
                           smooth_periodic_euler_v1_card_id &&
                       config.case_id().view() ==
                           smooth_periodic_euler_v1_case_id &&
                       config.checkpoint_id().view() ==
                           smooth_periodic_euler_v1_checkpoint_id &&
                       config.checkpoint_step() == 10 &&
                       config.restart_mode() ==
                           RestartMode::same_backend_rank_partition &&
                       config.restart_trajectory() ==
                           RestartTrajectoryModeV1::
                               uninterrupted_and_split_same_tuple &&
                       config.injected_failure_step() == 14 &&
                       config.injected_failure_phase() ==
                           FailurePhase::AfterRecoveryBeforeCommit &&
                       config.rollback_obligation() ==
                           RollbackObligationV1::
                               collective_exact_complete_state &&
                       config.source_rank_rule() ==
                           CheckpointSourceRankRuleV1::
                               rank_zero_publishes_manifest &&
                       config.checkpoint_source_rank() == 0 &&
                       config.retry_limit() == 1 &&
                       config.mpi_rank_count() ==
                           options.observed.mpi_rank_count &&
                       IsSharedFourShapeMesh(config.mesh_n()),
                   "compiled_N6_four_shape_contract") &&
          passed;
    }
    else
    {
      passed = Check(!validation.accepted() &&
                         validation.rejection() ==
                             ApplicationRejection::unbuilt_backend,
                     "known_unbuilt_backend_rejected") &&
          passed;
    }
  }
  passed = Check(CompiledBackendAvailable(ApplicationBackend::cpu_serial) ==
                         (DG_IGR_HDG_APP_HAS_CPU_SERIAL == 1) &&
                     CompiledBackendAvailable(ApplicationBackend::cpu_rank2) ==
                         (DG_IGR_HDG_APP_HAS_CPU_RANK2 == 1) &&
                     CompiledBackendAvailable(
                         ApplicationBackend::cuda_rank1_one_gpu) ==
                         (DG_IGR_HDG_APP_HAS_CUDA_RANK1 == 1) &&
                     CompiledBackendAvailable(
                         ApplicationBackend::
                             cuda_rank2_one_gpu_host_staged) ==
                         (DG_IGR_HDG_APP_HAS_CUDA_RANK2 == 1),
                 "availability_is_build_owned") &&
      passed;
  return passed;
}

bool CheckMeshMatrix()
{
  bool passed = true;
  for (const std::string_view backend :
       {CpuSerialBackendTag::id, CpuRank2BackendTag::id})
  {
    for (const std::uint32_t mesh : {3U, 6U, 12U})
    {
      ApplicationOptionsView options = OptionsFor(backend);
      options.mesh_n = mesh;
      passed = Check(ValidateApplicationOptions(options).accepted(),
                     "CPU_mesh_3_6_12_admitted") &&
          passed;
    }
    ApplicationOptionsView rejected = OptionsFor(backend);
    rejected.mesh_n = 24;
    passed = Check(ValidateApplicationOptions(rejected).rejection() ==
                       ApplicationRejection::invalid_mesh,
                   "CPU_other_mesh_rejected") &&
        passed;
  }
  for (const std::string_view backend :
       {CudaRank1OneGpuBackendTag::id,
        CudaRank2OneGpuHostStagedBackendTag::id})
  {
    if (!CompiledBackendAvailable(ParseApplicationBackend(backend)))
    {
      continue;
    }
    for (const std::uint32_t mesh : {3U, 6U})
    {
      ApplicationOptionsView options = OptionsFor(backend);
      options.mesh_n = mesh;
      passed = Check(ValidateApplicationOptions(options).accepted(),
                     "CUDA_mesh_3_6_admitted") &&
          passed;
    }
    ApplicationOptionsView rejected = OptionsFor(backend);
    rejected.mesh_n = 12;
    passed = Check(ValidateApplicationOptions(rejected).rejection() ==
                       ApplicationRejection::invalid_mesh,
                   "CUDA_mesh_12_rejected") &&
        passed;
  }
  return passed;
}

bool CheckPreExecutionFactsAndSelection()
{
  bool passed = true;
  ApplicationOptionsView options;
  options.interior_space = InteriorSpace::h1;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::unsupported_h1,
                 "H1_rejected_pre_execution") &&
      passed;
  options = {};
  options.interior_space = static_cast<InteriorSpace>(255);
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_interior_space,
                 "unknown_interior_rejected") &&
      passed;
  options = {};
  options.formulation_id = "hdg.unknown";
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::unknown_tuple,
                 "unknown_tuple_rejected") &&
      passed;
  options = {};
  options.backend_id = "backend.unknown";
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::unknown_backend,
                 "unknown_backend_rejected") &&
      passed;

  options = OptionsFor(CpuRank2BackendTag::id);
  options.observed.mpi_rank_count = 1;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_mpi_facts,
                 "rank2_requires_observed_two_ranks") &&
      passed;
  options = OptionsFor(CpuSerialBackendTag::id);
  options.observed.participating_gpu_count = 1;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_gpu_facts,
                 "CPU_rejects_selected_GPU_facts") &&
      passed;
  if (CompiledBackendAvailable(ApplicationBackend::cuda_rank1_one_gpu))
  {
    options = OptionsFor(CudaRank1OneGpuBackendTag::id);
    options.observed.selected_gpu_identity = {};
    passed = Check(ValidateApplicationOptions(options).rejection() ==
                       ApplicationRejection::invalid_gpu_facts,
                   "CUDA_requires_observed_GPU_identity") &&
        passed;
  }
  return passed;
}

bool CheckEveryFrozenFieldRejectsMutation()
{
  bool passed = true;
  ApplicationOptionsView options;
  options.card_id = "card.unknown";
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::unknown_card,
                 "card_identity_bound") &&
      passed;
  options = {};
  options.case_id = "case.unknown";
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::unknown_case,
                 "case_identity_bound") &&
      passed;
  options = {};
  options.polynomial_order = 2;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_polynomial_order,
                 "p1_bound") &&
      passed;
  options = {};
  options.step_bound = 19;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::insufficient_step_bound,
                 "minimum_steps_bound") &&
      passed;
  options = {};
  options.step_bound = 21;
  passed = Check(ValidateApplicationOptions(options).accepted(),
                 "longer_trajectory_admitted") &&
      passed;
  options = {};
  options.dt = 0.1;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_dt,
                 "dt_bound") &&
      passed;
  options = {};
  options.output_steps = {{0, 5, 20}};
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_output_schedule,
                 "output_schedule_bound") &&
      passed;
  options = {};
  options.checkpoint_step = 9;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_checkpoint_schedule,
                 "checkpoint_step_bound") &&
      passed;
  options = {};
  options.checkpoint_id = "checkpoint.wrong";
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_checkpoint_identity,
                 "checkpoint_identity_bound") &&
      passed;
  options = {};
  options.restart_mode = static_cast<RestartMode>(255);
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_restart_mode,
                 "same_tuple_restart_bound") &&
      passed;
  options = {};
  options.restart_trajectory = static_cast<RestartTrajectoryModeV1>(255);
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_restart_trajectory,
                 "split_and_uninterrupted_bound") &&
      passed;
  options = {};
  options.injected_failure_step = 13;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_failure_injection,
                 "failure_step_bound") &&
      passed;
  options = {};
  options.injected_failure_phase = FailurePhase::Recovery;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_failure_phase,
                 "failure_phase_bound") &&
      passed;
  options = {};
  options.rollback_obligation = static_cast<RollbackObligationV1>(255);
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_rollback_obligation,
                 "collective_rollback_bound") &&
      passed;
  options = {};
  options.checkpoint_source_rank = 1;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_source_rank_rule,
                 "source_rank_zero_bound") &&
      passed;
  options = {};
  options.retry_limit = 2;
  passed = Check(ValidateApplicationOptions(options).rejection() ==
                     ApplicationRejection::invalid_retry_limit,
                 "one_retry_bound") &&
      passed;
  passed = Check(ApplicationRunResult{}.reason ==
                     FailureReason::invalid_resource_descriptor,
                 "application_result_uses_existing_failure_reason") &&
      passed;
  return passed;
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckCompiledAvailabilityAndFourShapes() && passed;
  passed = CheckMeshMatrix() && passed;
  passed = CheckPreExecutionFactsAndSelection() && passed;
  passed = CheckEveryFrozenFieldRejectsMutation() && passed;
  if (passed)
  {
    std::cout << "HDG W8A application contract checks passed\n";
    return 0;
  }
  return 1;
}

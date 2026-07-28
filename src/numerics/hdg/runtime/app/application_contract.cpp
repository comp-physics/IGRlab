#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifndef DG_IGR_HDG_APP_HAS_CPU_SERIAL
#error "build-owned CPU serial availability definition is required"
#endif
#ifndef DG_IGR_HDG_APP_HAS_CPU_RANK2
#error "build-owned CPU rank-2 availability definition is required"
#endif
#ifndef DG_IGR_HDG_APP_HAS_CUDA_RANK1
#error "build-owned CUDA rank-1 availability definition is required"
#endif
#ifndef DG_IGR_HDG_APP_HAS_CUDA_RANK2
#error "build-owned CUDA rank-2 availability definition is required"
#endif

namespace dg::hdg::runtime::app
{
namespace
{

bool NonzeroHash(const StateHash& hash) noexcept
{
  return std::any_of(hash.bytes.begin(), hash.bytes.end(),
                     [](const std::uint8_t value) { return value != 0; });
}

bool ExactFrozenTuple(const ApplicationOptionsView& options) noexcept
{
  return options.model_id == EulerCompressible2dTag::id &&
      options.formulation_id == ConventionalHdgTag::id &&
      options.trace_id == MfemDgInterfaceTraceTag::id &&
      options.stabilization_id == FixedScalarTauTag::id &&
      options.time_treatment_id == BackwardEulerRuntimeTag::id &&
      options.precision_id == Fp64RuntimeTag::id;
}

bool ExactScientificIdentities(const ApplicationOptionsView& options) noexcept
{
  return options.mesh_family_id ==
          smooth_periodic_euler_v1_mesh_family_id &&
      options.source_id == smooth_periodic_euler_v1_source_id &&
      options.flux_id == smooth_periodic_euler_v1_flux_id &&
      options.time_rule_id == smooth_periodic_euler_v1_time_rule_id &&
      options.layout_id == smooth_periodic_euler_v1_layout_id &&
      options.gamma == smooth_periodic_euler_v1_gamma &&
      options.tau == smooth_periodic_euler_v1_tau;
}

RuntimeTupleDescriptor TupleForBackend(
    const ApplicationBackend backend) noexcept
{
  switch (backend)
  {
    case ApplicationBackend::cpu_serial:
      return CpuSerialRuntimeSelection::Descriptor();
    case ApplicationBackend::cpu_rank2:
      return CpuRank2RuntimeSelection::Descriptor();
    case ApplicationBackend::cuda_rank1_one_gpu:
      return CudaRank1OneGpuRuntimeSelection::Descriptor();
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
      return CudaRank2OneGpuHostStagedRuntimeSelection::Descriptor();
    case ApplicationBackend::unknown: return {};
  }
  return {};
}

DeferredRuntimePredicatesV1 DeferredForBackend(
    const ApplicationBackend backend) noexcept
{
  switch (backend)
  {
    case ApplicationBackend::cpu_serial:
      return {1, DeferredGpuRuleV1::no_cuda_device,
              DeferredSourceRankRuleV1::
                  rank_zero_for_rank1_rank_one_for_rank2,
              0};
    case ApplicationBackend::cpu_rank2:
      return {2, DeferredGpuRuleV1::no_cuda_device,
              DeferredSourceRankRuleV1::
                  rank_zero_for_rank1_rank_one_for_rank2,
              1};
    case ApplicationBackend::cuda_rank1_one_gpu:
      return {1, DeferredGpuRuleV1::one_physical_gpu,
              DeferredSourceRankRuleV1::
                  rank_zero_for_rank1_rank_one_for_rank2,
              0};
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
      return {2, DeferredGpuRuleV1::one_shared_physical_gpu,
              DeferredSourceRankRuleV1::
                  rank_zero_for_rank1_rank_one_for_rank2,
              1};
    case ApplicationBackend::unknown: return {};
  }
  return {};
}

bool ExactEndTime(const ApplicationOptionsView& options) noexcept
{
  if (!std::isfinite(options.dt) || !std::isfinite(options.final_time))
  {
    return false;
  }
  const double reconstructed =
      static_cast<double>(options.accepted_steps) * options.dt;
  const double scale = std::max(1.0, std::abs(options.final_time));
  return std::abs(options.final_time - reconstructed) <=
      8.0 * std::numeric_limits<double>::epsilon() * scale;
}

ApplicationRejection ValidateSmoke(
    const ApplicationOptionsView& options) noexcept
{
  if (options.polynomial_order !=
          smooth_periodic_euler_v1_smoke_polynomial_order ||
      options.mesh_n != smooth_periodic_euler_v1_smoke_mesh_n)
  {
    return ApplicationRejection::invalid_polynomial_mesh_pair;
  }
  if (options.q_a != smooth_periodic_euler_v1_smoke_q_a ||
      options.q_e != smooth_periodic_euler_v1_smoke_q_e ||
      options.q_e4 != smooth_periodic_euler_v1_smoke_q_e4 ||
      options.q_o != smooth_periodic_euler_v1_smoke_q_o)
  {
    return ApplicationRejection::invalid_quadrature;
  }
  if (options.accepted_steps != smooth_periodic_euler_v1_smoke_steps)
  {
    return ApplicationRejection::invalid_step_count;
  }
  if (options.final_time != smooth_periodic_euler_v1_smoke_final_time ||
      !ExactEndTime(options))
  {
    return ApplicationRejection::invalid_final_time;
  }
  if (options.output_steps != smooth_periodic_euler_v1_smoke_output_steps)
  {
    return ApplicationRejection::invalid_output_schedule;
  }
  if (options.checkpoint_step !=
      smooth_periodic_euler_v1_smoke_checkpoint_step)
  {
    return ApplicationRejection::invalid_checkpoint_schedule;
  }
  if (options.checkpoint_id !=
      smooth_periodic_euler_v1_smoke_checkpoint_id)
  {
    return ApplicationRejection::invalid_checkpoint_identity;
  }
  if (options.injected_failure_committed_step !=
          smooth_periodic_euler_v1_smoke_failure_committed_step ||
      options.injected_failure_target_step !=
          smooth_periodic_euler_v1_smoke_failure_target_step)
  {
    return ApplicationRejection::invalid_failure_injection;
  }
  if (options.seed != smooth_periodic_euler_v1_smoke_seed)
  {
    return ApplicationRejection::invalid_seed;
  }
  return ApplicationRejection::none;
}

ApplicationRejection ValidateFunctional(
    const ApplicationOptionsView& options,
    const ApplicationBackend backend) noexcept
{
  if (options.polynomial_order !=
      smooth_periodic_euler_v1_functional_polynomial_order)
  {
    return ApplicationRejection::invalid_polynomial_mesh_pair;
  }
  const bool serial_ladder = backend == ApplicationBackend::cpu_serial &&
      (options.mesh_n == 3 || options.mesh_n == 6 || options.mesh_n == 12);
  const bool principal_other_backend =
      backend != ApplicationBackend::cpu_serial &&
      options.mesh_n == smooth_periodic_euler_v1_functional_mesh_n;
  if (!serial_ladder && !principal_other_backend)
  {
    return ApplicationRejection::invalid_polynomial_mesh_pair;
  }
  if (options.q_a != smooth_periodic_euler_v1_functional_q_a ||
      options.q_e != smooth_periodic_euler_v1_functional_q_e ||
      options.q_e4 != smooth_periodic_euler_v1_functional_q_e4 ||
      options.q_o != smooth_periodic_euler_v1_functional_q_o)
  {
    return ApplicationRejection::invalid_quadrature;
  }
  if (options.accepted_steps != smooth_periodic_euler_v1_functional_steps)
  {
    return ApplicationRejection::invalid_step_count;
  }
  if (options.final_time != smooth_periodic_euler_v1_functional_final_time ||
      !ExactEndTime(options))
  {
    return ApplicationRejection::invalid_final_time;
  }
  if (options.output_steps !=
      smooth_periodic_euler_v1_functional_output_steps)
  {
    return ApplicationRejection::invalid_output_schedule;
  }
  if (options.checkpoint_step !=
      smooth_periodic_euler_v1_functional_checkpoint_step)
  {
    return ApplicationRejection::invalid_checkpoint_schedule;
  }
  if (options.checkpoint_id !=
      smooth_periodic_euler_v1_functional_checkpoint_id)
  {
    return ApplicationRejection::invalid_checkpoint_identity;
  }
  if (options.injected_failure_committed_step !=
          smooth_periodic_euler_v1_functional_failure_committed_step ||
      options.injected_failure_target_step !=
          smooth_periodic_euler_v1_functional_failure_target_step)
  {
    return ApplicationRejection::invalid_failure_injection;
  }
  if (options.seed != smooth_periodic_euler_v1_functional_seed)
  {
    return ApplicationRejection::invalid_seed;
  }
  return ApplicationRejection::none;
}

} // namespace

ValidatedApplicationConfig::ValidatedApplicationConfig(
    const ApplicationOptionsView& options,
    const ApplicationScenarioV1 scenario,
    const ApplicationBackend backend,
    const RuntimeTupleDescriptor tuple,
    const DeferredRuntimePredicatesV1 deferred) noexcept
    : scenario_(scenario),
      backend_(backend),
      tuple_(tuple),
      card_id_(StableIdentity::Copy(options.card_id)),
      case_id_(StableIdentity::Copy(options.case_id)),
      scenario_id_(StableIdentity::Copy(options.scenario_id)),
      checkpoint_id_(StableIdentity::Copy(options.checkpoint_id)),
      mesh_family_id_(StableIdentity::Copy(options.mesh_family_id)),
      source_id_(StableIdentity::Copy(options.source_id)),
      flux_id_(StableIdentity::Copy(options.flux_id)),
      time_rule_id_(StableIdentity::Copy(options.time_rule_id)),
      layout_id_(StableIdentity::Copy(options.layout_id)),
      interior_space_(options.interior_space),
      polynomial_order_(options.polynomial_order),
      mesh_n_(options.mesh_n),
      q_a_(options.q_a),
      q_e_(options.q_e),
      q_e4_(options.q_e4),
      q_o_(options.q_o),
      gamma_(options.gamma),
      tau_(options.tau),
      dt_(options.dt),
      accepted_steps_(options.accepted_steps),
      final_time_(options.final_time),
      output_steps_(options.output_steps),
      checkpoint_step_(options.checkpoint_step),
      restart_mode_(options.restart_mode),
      restart_trajectory_(options.restart_trajectory),
      injected_failure_committed_step_(
          options.injected_failure_committed_step),
      injected_failure_target_step_(options.injected_failure_target_step),
      injected_failure_phase_(options.injected_failure_phase),
      rollback_obligation_(options.rollback_obligation),
      source_rank_rule_(options.source_rank_rule),
      retry_limit_(options.retry_limit),
      seed_(options.seed),
      scientific_config_sha256_(options.scientific_config_sha256),
      run_config_sha256_(options.run_config_sha256),
      deferred_runtime_(deferred)
{}

ApplicationScenarioV1 ParseApplicationScenario(
    const std::string_view id) noexcept
{
  if (id == smooth_periodic_euler_v1_smoke_id)
  {
    return ApplicationScenarioV1::smoke;
  }
  if (id == smooth_periodic_euler_v1_functional_id)
  {
    return ApplicationScenarioV1::functional;
  }
  return ApplicationScenarioV1::unknown;
}

ApplicationBackend ParseApplicationBackend(const std::string_view id) noexcept
{
  if (id == CpuSerialBackendTag::id)
  {
    return ApplicationBackend::cpu_serial;
  }
  if (id == CpuRank2BackendTag::id)
  {
    return ApplicationBackend::cpu_rank2;
  }
  if (id == CudaRank1OneGpuBackendTag::id)
  {
    return ApplicationBackend::cuda_rank1_one_gpu;
  }
  if (id == CudaRank2OneGpuHostStagedBackendTag::id)
  {
    return ApplicationBackend::cuda_rank2_one_gpu_host_staged;
  }
  return ApplicationBackend::unknown;
}

bool CompiledBackendAvailable(const ApplicationBackend backend) noexcept
{
  switch (backend)
  {
    case ApplicationBackend::cpu_serial:
      return DG_IGR_HDG_APP_HAS_CPU_SERIAL == 1;
    case ApplicationBackend::cpu_rank2:
      return DG_IGR_HDG_APP_HAS_CPU_RANK2 == 1;
    case ApplicationBackend::cuda_rank1_one_gpu:
      return DG_IGR_HDG_APP_HAS_CUDA_RANK1 == 1;
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
      return DG_IGR_HDG_APP_HAS_CUDA_RANK2 == 1;
    case ApplicationBackend::unknown: return false;
  }
  return false;
}

ApplicationValidation ValidateApplicationOptions(
    const ApplicationOptionsView& options) noexcept
{
  const auto reject = [](const ApplicationRejection reason) noexcept {
    return ApplicationValidation(reason);
  };
  if (options.card_id != smooth_periodic_euler_v1_card_id)
  {
    return reject(ApplicationRejection::unknown_card);
  }
  if (options.case_id != smooth_periodic_euler_v1_case_id)
  {
    return reject(ApplicationRejection::unknown_case);
  }
  const ApplicationScenarioV1 scenario =
      ParseApplicationScenario(options.scenario_id);
  if (scenario == ApplicationScenarioV1::unknown)
  {
    return reject(ApplicationRejection::unknown_scenario);
  }
  if (options.interior_space == InteriorSpace::h1)
  {
    return reject(ApplicationRejection::unsupported_h1);
  }
  if (options.interior_space != InteriorSpace::l2)
  {
    return reject(ApplicationRejection::invalid_interior_space);
  }
  if (!ExactFrozenTuple(options))
  {
    return reject(ApplicationRejection::unknown_tuple);
  }
  const ApplicationBackend backend =
      ParseApplicationBackend(options.backend_id);
  if (backend == ApplicationBackend::unknown)
  {
    return reject(ApplicationRejection::unknown_backend);
  }
  if (!CompiledBackendAvailable(backend))
  {
    return reject(ApplicationRejection::unbuilt_backend);
  }
  if (!ExactScientificIdentities(options))
  {
    return reject(ApplicationRejection::invalid_scientific_identity);
  }
  if (!NonzeroHash(options.scientific_config_sha256) ||
      !NonzeroHash(options.run_config_sha256))
  {
    return reject(ApplicationRejection::invalid_config_hash);
  }
  if (options.dt != smooth_periodic_euler_v1_dt)
  {
    return reject(ApplicationRejection::invalid_dt);
  }
  const ApplicationRejection scenario_rejection =
      scenario == ApplicationScenarioV1::smoke
      ? ValidateSmoke(options)
      : ValidateFunctional(options, backend);
  if (scenario_rejection != ApplicationRejection::none)
  {
    return reject(scenario_rejection);
  }
  if (options.restart_mode != RestartMode::same_backend_rank_partition)
  {
    return reject(ApplicationRejection::invalid_restart_mode);
  }
  if (options.restart_trajectory !=
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple)
  {
    return reject(ApplicationRejection::invalid_restart_trajectory);
  }
  if (options.injected_failure_phase !=
      smooth_periodic_euler_v1_failure_phase)
  {
    return reject(ApplicationRejection::invalid_failure_phase);
  }
  if (options.rollback_obligation !=
      RollbackObligationV1::collective_exact_complete_state)
  {
    return reject(ApplicationRejection::invalid_rollback_obligation);
  }
  if (options.source_rank_rule != DeferredSourceRankRuleV1::
          rank_zero_for_rank1_rank_one_for_rank2)
  {
    return reject(ApplicationRejection::invalid_source_rank_rule);
  }
  if (options.retry_limit != smooth_periodic_euler_v1_retry_limit)
  {
    return reject(ApplicationRejection::invalid_retry_limit);
  }
  const RuntimeTupleDescriptor tuple = TupleForBackend(backend);
  if (!tuple.Valid())
  {
    return reject(ApplicationRejection::unknown_tuple);
  }
  return ApplicationValidation(ValidatedApplicationConfig(
      options, scenario, backend, tuple, DeferredForBackend(backend)));
}

} // namespace dg::hdg::runtime::app

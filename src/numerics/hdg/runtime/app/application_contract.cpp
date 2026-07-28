#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <cmath>

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

bool ExactFrozenTuple(const ApplicationOptionsView& options) noexcept
{
  return options.model_id == EulerCompressible2dTag::id &&
      options.formulation_id == ConventionalHdgTag::id &&
      options.trace_id == MfemDgInterfaceTraceTag::id &&
      options.stabilization_id == FixedScalarTauTag::id &&
      options.time_treatment_id == BackwardEulerRuntimeTag::id &&
      options.precision_id == Fp64RuntimeTag::id;
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

bool ExpectedMpiFacts(const ApplicationBackend backend,
                      const ObservedRuntimeFactsView& facts) noexcept
{
  if (!facts.mpi_initialized ||
      facts.local_mpi_rank >= facts.mpi_rank_count)
  {
    return false;
  }
  const bool rank1 = backend == ApplicationBackend::cpu_serial ||
      backend == ApplicationBackend::cuda_rank1_one_gpu;
  return facts.mpi_rank_count == (rank1 ? 1U : 2U);
}

bool ExpectedGpuFacts(const ApplicationBackend backend,
                      const ObservedRuntimeFactsView& facts) noexcept
{
  const bool cuda = backend == ApplicationBackend::cuda_rank1_one_gpu ||
      backend == ApplicationBackend::cuda_rank2_one_gpu_host_staged;
  if (!cuda)
  {
    return facts.participating_gpu_count == 0 &&
        facts.selected_gpu_ordinal == -1 &&
        facts.selected_gpu_identity.empty() &&
        facts.gpu_identity_consistent_across_ranks &&
        !facts.all_ranks_share_one_gpu;
  }
  return facts.participating_gpu_count == 1 &&
      facts.selected_gpu_ordinal >= 0 &&
      !facts.selected_gpu_identity.empty() &&
      facts.gpu_identity_consistent_across_ranks &&
      facts.all_ranks_share_one_gpu;
}

} // namespace

ValidatedApplicationConfig::ValidatedApplicationConfig(
    const ApplicationOptionsView& options,
    const ApplicationBackend backend,
    const RuntimeTupleDescriptor tuple) noexcept
    : backend_(backend),
      tuple_(tuple),
      interior_space_(options.interior_space),
      polynomial_order_(options.polynomial_order),
      mesh_n_(options.mesh_n),
      step_bound_(options.step_bound),
      dt_(options.dt),
      output_steps_(options.output_steps),
      checkpoint_step_(options.checkpoint_step),
      card_id_(StableIdentity::Copy(options.card_id)),
      case_id_(StableIdentity::Copy(options.case_id)),
      checkpoint_id_(StableIdentity::Copy(options.checkpoint_id)),
      restart_mode_(options.restart_mode),
      restart_trajectory_(options.restart_trajectory),
      injected_failure_step_(options.injected_failure_step),
      injected_failure_phase_(options.injected_failure_phase),
      rollback_obligation_(options.rollback_obligation),
      source_rank_rule_(options.source_rank_rule),
      checkpoint_source_rank_(options.checkpoint_source_rank),
      retry_limit_(options.retry_limit),
      mpi_rank_count_(options.observed.mpi_rank_count),
      local_mpi_rank_(options.observed.local_mpi_rank),
      participating_gpu_count_(options.observed.participating_gpu_count),
      selected_gpu_ordinal_(options.observed.selected_gpu_ordinal),
      selected_gpu_identity_(
          StableIdentity::Copy(options.observed.selected_gpu_identity)),
      gpu_identity_consistent_across_ranks_(
          options.observed.gpu_identity_consistent_across_ranks),
      all_ranks_share_one_gpu_(options.observed.all_ranks_share_one_gpu)
{}

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

bool IsAdmittedMesh(const ApplicationBackend backend,
                    const std::uint32_t mesh_n) noexcept
{
  switch (backend)
  {
    case ApplicationBackend::cpu_serial:
    case ApplicationBackend::cpu_rank2:
      return mesh_n == 3 || mesh_n == 6 || mesh_n == 12;
    case ApplicationBackend::cuda_rank1_one_gpu:
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
      return mesh_n == 3 || mesh_n == 6;
    case ApplicationBackend::unknown: return false;
  }
  return false;
}

ApplicationValidation ValidateApplicationOptions(
    const ApplicationOptionsView& options) noexcept
{
  const auto reject = [](const ApplicationRejection reason) noexcept {
    return ApplicationValidation(false, reason, ValidatedApplicationConfig{});
  };
  if (options.card_id != smooth_periodic_euler_v1_card_id)
  {
    return reject(ApplicationRejection::unknown_card);
  }
  if (options.case_id != smooth_periodic_euler_v1_case_id)
  {
    return reject(ApplicationRejection::unknown_case);
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
  if (!ExpectedMpiFacts(backend, options.observed))
  {
    return reject(ApplicationRejection::invalid_mpi_facts);
  }
  if (!ExpectedGpuFacts(backend, options.observed))
  {
    return reject(ApplicationRejection::invalid_gpu_facts);
  }
  if (options.polynomial_order != smooth_periodic_euler_v1_polynomial_order)
  {
    return reject(ApplicationRejection::invalid_polynomial_order);
  }
  if (!IsAdmittedMesh(backend, options.mesh_n))
  {
    return reject(ApplicationRejection::invalid_mesh);
  }
  if (options.step_bound < smooth_periodic_euler_v1_minimum_steps)
  {
    return reject(ApplicationRejection::insufficient_step_bound);
  }
  if (!std::isfinite(options.dt) ||
      options.dt != smooth_periodic_euler_v1_dt)
  {
    return reject(ApplicationRejection::invalid_dt);
  }
  if (options.output_steps != smooth_periodic_euler_v1_output_steps)
  {
    return reject(ApplicationRejection::invalid_output_schedule);
  }
  if (options.checkpoint_step != smooth_periodic_euler_v1_checkpoint_step ||
      options.checkpoint_step > options.step_bound)
  {
    return reject(ApplicationRejection::invalid_checkpoint_schedule);
  }
  if (options.checkpoint_id != smooth_periodic_euler_v1_checkpoint_id)
  {
    return reject(ApplicationRejection::invalid_checkpoint_identity);
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
  if (options.injected_failure_step !=
          smooth_periodic_euler_v1_failure_step ||
      options.injected_failure_step > options.step_bound)
  {
    return reject(ApplicationRejection::invalid_failure_injection);
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
  if (options.source_rank_rule !=
          CheckpointSourceRankRuleV1::rank_zero_publishes_manifest ||
      options.checkpoint_source_rank != smooth_periodic_euler_v1_source_rank)
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
  return ApplicationValidation(
      true, ApplicationRejection::none,
      ValidatedApplicationConfig(options, backend, tuple));
}

} // namespace dg::hdg::runtime::app

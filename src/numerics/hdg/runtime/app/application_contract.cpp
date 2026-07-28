#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <cmath>

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

ApplicationValidation Reject(const ApplicationRejection reason) noexcept
{
  return {false, reason, {}};
}

} // namespace

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

bool IsBackendBuilt(const ApplicationBackend backend,
                    const BuiltBackendSet& built) noexcept
{
  switch (backend)
  {
    case ApplicationBackend::cpu_serial: return built.cpu_serial;
    case ApplicationBackend::cpu_rank2: return built.cpu_rank2;
    case ApplicationBackend::cuda_rank1_one_gpu:
      return built.cuda_rank1_one_gpu;
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
      return built.cuda_rank2_one_gpu_host_staged;
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
    const ApplicationOptionsView& options,
    const BuiltBackendSet& built) noexcept
{
  if (options.case_id != smooth_periodic_euler_v1_case_id)
  {
    return Reject(ApplicationRejection::unknown_case);
  }
  if (options.interior_space == InteriorSpace::h1)
  {
    return Reject(ApplicationRejection::unsupported_h1);
  }
  if (options.interior_space != InteriorSpace::l2)
  {
    return Reject(ApplicationRejection::invalid_interior_space);
  }
  if (!ExactFrozenTuple(options))
  {
    return Reject(ApplicationRejection::unknown_tuple);
  }

  const ApplicationBackend backend =
      ParseApplicationBackend(options.backend_id);
  if (backend == ApplicationBackend::unknown)
  {
    return Reject(ApplicationRejection::unknown_backend);
  }
  if (!IsBackendBuilt(backend, built))
  {
    return Reject(ApplicationRejection::unbuilt_backend);
  }
  if (options.polynomial_order != smooth_periodic_euler_v1_polynomial_order)
  {
    return Reject(ApplicationRejection::invalid_polynomial_order);
  }
  if (!IsAdmittedMesh(backend, options.mesh_n))
  {
    return Reject(ApplicationRejection::invalid_mesh);
  }
  if (options.step_bound < smooth_periodic_euler_v1_minimum_steps)
  {
    return Reject(ApplicationRejection::insufficient_step_bound);
  }
  if (!std::isfinite(options.dt) ||
      options.dt != smooth_periodic_euler_v1_dt)
  {
    return Reject(ApplicationRejection::invalid_dt);
  }
  if (options.output_steps != smooth_periodic_euler_v1_output_steps)
  {
    return Reject(ApplicationRejection::invalid_output_schedule);
  }
  if (options.checkpoint_step != smooth_periodic_euler_v1_checkpoint_step ||
      options.checkpoint_step > options.step_bound)
  {
    return Reject(ApplicationRejection::invalid_checkpoint_schedule);
  }
  if (options.restart_mode != RestartMode::same_backend_rank_partition)
  {
    return Reject(ApplicationRejection::invalid_restart_mode);
  }
  if (options.injected_failure_step !=
          smooth_periodic_euler_v1_failure_step ||
      options.injected_failure_step > options.step_bound)
  {
    return Reject(ApplicationRejection::invalid_failure_injection);
  }
  if (options.retry_limit != smooth_periodic_euler_v1_retry_limit)
  {
    return Reject(ApplicationRejection::invalid_retry_limit);
  }

  const RuntimeTupleDescriptor tuple = TupleForBackend(backend);
  if (!tuple.Valid())
  {
    return Reject(ApplicationRejection::unknown_tuple);
  }
  const ValidatedApplicationConfig config{backend,
                                          tuple,
                                          options.interior_space,
                                          options.polynomial_order,
                                          options.mesh_n,
                                          options.step_bound,
                                          options.dt,
                                          options.output_steps,
                                          options.checkpoint_step,
                                          options.restart_mode,
                                          options.injected_failure_step,
                                          options.retry_limit};
  return {true, ApplicationRejection::none, config};
}

} // namespace dg::hdg::runtime::app

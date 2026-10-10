#include "numerics/hdg/runtime/app/solver.hpp"

#include "solver_entry_detail.hpp"

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

ApplicationRunResult RejectStaticEntryMismatch() noexcept
{
  ApplicationRunResult result{};
  result.status = RunStatus::invalid_request;
  result.reason = FailureReason::invalid_resource_descriptor;
  result.failure_class = ApplicationRunResult::FailureClass::configuration;
  return result;
}

bool IsConventionalStaticEntry(
    const ApplicationOptionsView& request,
    const std::string_view expected_backend) noexcept
{
  return request.backend_id == expected_backend &&
      request.card_id == smooth_periodic_euler_v1_card_id;
}

HdgMethodResultV1 RejectMethodRequest(
    const HdgMethodCapabilitiesV1 capabilities) noexcept
{
  HdgMethodResultV1 result{};
  result.capabilities = capabilities;
  result.run = RejectStaticEntryMismatch();
  return result;
}

} // namespace

MpiApplicationContextV1::MpiApplicationContextV1(
    const MPI_Comm communicator) noexcept : communicator_(communicator)
{
  if (communicator_ != MPI_COMM_NULL)
  {
    (void)MPI_Comm_rank(communicator_, &rank_);
    (void)MPI_Comm_size(communicator_, &ranks_);
  }
}

CudaApplicationContextV1::CudaApplicationContextV1(
    const MPI_Comm communicator) noexcept : communicator_(communicator)
{
  if (communicator_ != MPI_COMM_NULL)
  {
    (void)MPI_Comm_rank(communicator_, &rank_);
    (void)MPI_Comm_size(communicator_, &ranks_);
  }
}

HdgMethodResultV1 RunHdgMethodV1(const HdgMethodRequestV1& request)
{
  if (request.options == nullptr || request.config == nullptr)
  {
    return RejectMethodRequest({});
  }
  const ApplicationValidation validation =
      ValidateApplicationOptions(*request.options);
  if (!validation.accepted() ||
      validation.config()->scientific_config_sha256() !=
          request.config->scientific_config_sha256() ||
      validation.config()->run_config_sha256() != request.config->run_config_sha256())
  {
    return RejectMethodRequest(QueryHdgMethodCapabilitiesV1(*request.options));
  }

  HdgMethodResultV1 result{};
  result.capabilities = CapabilitiesForV1(*request.config);
  result.checkpoint_identity = CheckpointIdentityForV1(*request.config);
  if (!result.capabilities.supported)
  {
    result.run = RejectStaticEntryMismatch();
    return result;
  }

  switch (request.config->backend())
  {
    case ApplicationBackend::cpu_serial:
      if (request.cpu_context == nullptr) return RejectMethodRequest(result.capabilities);
      result.run = RunSmoothPeriodicEulerV1(
          CpuSerialBackendTag{}, *request.options, *request.cpu_context);
      return result;
    case ApplicationBackend::cpu_rank2:
      if (request.cpu_context == nullptr) return RejectMethodRequest(result.capabilities);
      result.run = RunSmoothPeriodicEulerV1(
          CpuRank2BackendTag{}, *request.options, *request.cpu_context);
      return result;
    case ApplicationBackend::cuda_rank1_one_gpu:
#if DG_IGR_HDG_APP_HAS_CUDA_RANK1
      if (request.cuda_context == nullptr) return RejectMethodRequest(result.capabilities);
      result.run = RunSmoothPeriodicEulerV1(
          CudaRank1OneGpuBackendTag{}, *request.options, *request.cuda_context);
      return result;
#else
      return RejectMethodRequest(result.capabilities);
#endif
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
#if DG_IGR_HDG_APP_HAS_CUDA_RANK2
      if (request.cuda_context == nullptr) return RejectMethodRequest(result.capabilities);
      result.run = RunSmoothPeriodicEulerV1(
          CudaRank2OneGpuHostStagedBackendTag{}, *request.options,
          *request.cuda_context);
      return result;
#else
      return RejectMethodRequest(result.capabilities);
#endif
    case ApplicationBackend::unknown: return RejectMethodRequest(result.capabilities);
  }
  return RejectMethodRequest(result.capabilities);
}

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuSerialBackendTag tag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  if (!IsConventionalStaticEntry(request, CpuSerialBackendTag::id))
    return RejectStaticEntryMismatch();
  return detail::RunSmoothPeriodicEulerV1Downstream(tag, request, context);
}

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuRank2BackendTag tag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  if (!IsConventionalStaticEntry(request, CpuRank2BackendTag::id))
    return RejectStaticEntryMismatch();
  return detail::RunSmoothPeriodicEulerV1Downstream(tag, request, context);
}

#if DG_IGR_HDG_APP_HAS_CUDA_RANK1
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank1OneGpuBackendTag tag, const ApplicationOptionsView& request,
    CudaApplicationContextV1& context)
{
  if (!IsConventionalStaticEntry(
          request, CudaRank1OneGpuBackendTag::id))
    return RejectStaticEntryMismatch();
  return detail::RunSmoothPeriodicEulerV1Downstream(tag, request, context);
}
#endif

#if DG_IGR_HDG_APP_HAS_CUDA_RANK2
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank2OneGpuHostStagedBackendTag tag,
    const ApplicationOptionsView& request,
    CudaApplicationContextV1& context)
{
  if (!IsConventionalStaticEntry(
          request, CudaRank2OneGpuHostStagedBackendTag::id))
    return RejectStaticEntryMismatch();
  return detail::RunSmoothPeriodicEulerV1Downstream(tag, request, context);
}
#endif

} // namespace dg::hdg::runtime::app

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

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuSerialBackendTag tag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  if (request.backend_id != CpuSerialBackendTag::id)
    return RejectStaticEntryMismatch();
  return detail::RunSmoothPeriodicEulerV1Downstream(tag, request, context);
}

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuRank2BackendTag tag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  if (request.backend_id != CpuRank2BackendTag::id)
    return RejectStaticEntryMismatch();
  return detail::RunSmoothPeriodicEulerV1Downstream(tag, request, context);
}

#if DG_IGR_HDG_APP_HAS_CUDA_RANK1
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank1OneGpuBackendTag tag, const ApplicationOptionsView& request,
    CudaApplicationContextV1& context)
{
  if (request.backend_id != CudaRank1OneGpuBackendTag::id)
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
  if (request.backend_id != CudaRank2OneGpuHostStagedBackendTag::id)
    return RejectStaticEntryMismatch();
  return detail::RunSmoothPeriodicEulerV1Downstream(tag, request, context);
}
#endif

} // namespace dg::hdg::runtime::app

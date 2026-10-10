#pragma once

#include "numerics/hdg/runtime/app/application_contract.hpp"

namespace dg::hdg::runtime::app::detail
{

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CpuSerialBackendTag,
    const ApplicationOptionsView& request,
    MpiApplicationContextV1& context);
ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CpuRank2BackendTag,
    const ApplicationOptionsView& request,
    MpiApplicationContextV1& context);
ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CudaRank1OneGpuBackendTag,
    const ApplicationOptionsView& request,
    CudaApplicationContextV1& context);
ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CudaRank2OneGpuHostStagedBackendTag,
    const ApplicationOptionsView& request,
    CudaApplicationContextV1& context);

} // namespace dg::hdg::runtime::app::detail

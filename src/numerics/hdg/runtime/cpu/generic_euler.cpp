#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

namespace dg::hdg::runtime::cpu
{
bool ExecuteGenericEulerHdgCpu(const generic::EulerHdgRequest& request) noexcept
{
  const generic::ContractStatus admission =
      generic::ValidateEulerHdgRequest(request);
  if (admission != generic::ContractStatus::accepted)
  {
    return false;
  }
  if (request.backend.kind != generic::BackendKind::cpu || request.mesh == nullptr)
  {
    return false;
  }

  detail::CpuNumericsCore core;
  return core.RunBorrowedEuler(request);
}

bool RestartGenericEulerHdgCpu(const generic::EulerHdgRequest&) noexcept
{
  // No checkpoint representation is accepted by this borrowed, one-shot API.
  // Leaving the caller's output untouched makes rejection failure-atomic.
  return false;
}

} // namespace dg::hdg::runtime::cpu

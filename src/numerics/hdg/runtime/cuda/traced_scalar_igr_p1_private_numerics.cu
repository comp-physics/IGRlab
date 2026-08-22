#include "numerics/hdg/runtime/cuda/traced_scalar_igr_p1_private_numerics.cuh"

namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics
{
const PackedQuadratureContract& ReferencePlan() noexcept
{
  static constexpr PackedQuadratureContract plan{12};
  return plan;
}
} // namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics

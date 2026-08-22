#include "numerics/hdg/runtime/cuda/traced_scalar_igr_p1_private_numerics.cuh"

namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics
{
const PackedQ1ReferencePlan& ReferencePlan() noexcept
{
  constexpr double a = 0.21132486540518711775;
  constexpr double b = 0.78867513459481288225;
  static constexpr PackedQ1ReferencePlan plan{{
      a, a, 0.25,
      b, a, 0.25,
      a, b, 0.25,
      b, b, 0.25},
      {a, 0.5, b, 0.5}};
  return plan;
}
} // namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics

#pragma once

namespace dg::hdg::generic
{

// Shared bounds for the dynamic Euler/IGR local algebra and CUDA trace plans.
// Keep this header independent of MFEM and CUDA for the topology planners.
inline constexpr int minimum_polynomial_order = 1;
inline constexpr int maximum_polynomial_order = 10;

} // namespace dg::hdg::generic

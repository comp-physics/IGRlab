#pragma once

// Immutable reference geometry for the private packed Q1 borrowing ABI.
// This is deliberately a non-owning value plan: it creates no mesh, finite
// element space, communicator, arena, HYPRE object, or solver lifecycle.

#include <array>

namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics
{
struct PackedQ1ReferencePlan final
{
  static constexpr int volume_points = 4;
  static constexpr int segment_points = 2;
  std::array<double, 3 * volume_points> volume_xyw{};
  std::array<double, 2 * segment_points> segment_xw{};
};

// One immutable affine-Q1 Gauss plan shared by the SC25 owners. Callers own
// any device copies and must admit their topology before consuming it.
const PackedQ1ReferencePlan& ReferencePlan() noexcept;
} // namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics

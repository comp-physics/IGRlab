#pragma once

// Immutable quadrature contract for the private packed borrowing ABI. This is
// deliberately a non-owning value plan: it creates no mesh, finite
// element space, communicator, arena, HYPRE object, or solver lifecycle.

#include <array>

namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics
{
struct PackedQuadratureContract final
{
  int integration_order = 12;
};

// One immutable contract shared by SC25 owners. They source the actual
// order-12 MFEM rules from their admitted finite-element topology and own any
// corresponding device copies.
const PackedQuadratureContract& ReferencePlan() noexcept;
} // namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics

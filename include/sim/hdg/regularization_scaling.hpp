#pragma once

/**
 * @file sim/hdg/regularization_scaling.hpp
 * @brief Typed lowering of mesh-independent HDG IGR controls.
 */

#include <cmath>
#include <stdexcept>

#include "numerics/hdg/generic/contract.hpp"
#include "sim/context/mesh_build.hpp"

namespace dg_sim::hdg
{

/** User-facing IGR controls before the mesh supplies the physical length. */
struct HdgRegularizationRequest
{
  dg::hdg::generic::RegularizationKind kind =
      dg::hdg::generic::RegularizationKind::none;
  double c_alpha = 0.0;
  double euler_trace_penalty = 0.0;
};

/** Mesh-resolved IGR coefficient and the diagnostics used to obtain it. */
struct ResolvedHdgRegularization
{
  dg::hdg::generic::RegularizationSelection selection{
      dg::hdg::generic::RegularizationKind::none, 0.0, 0.0};
  AlphaInfo alpha_info{};
};

inline bool ValidHdgRegularizationRequest(
    const HdgRegularizationRequest& request) noexcept
{
  using dg::hdg::generic::DoubleBits;
  using dg::hdg::generic::RegularizationKind;
  if (request.kind == RegularizationKind::none)
  {
    return DoubleBits(request.c_alpha) == 0 &&
        DoubleBits(request.euler_trace_penalty) == 0;
  }
  return request.kind == RegularizationKind::igr &&
      std::isfinite(request.c_alpha) && request.c_alpha > 0.0 &&
      std::isfinite(request.euler_trace_penalty) &&
      request.euler_trace_penalty > 0.0;
}

/**
 * Convert the dimensionless IGR scalar to the physical squared-length
 * coefficient used by the singular-mass HDG equations.
 *
 * This is collective because ComputeAlphaInfo globally reduces the selected
 * mesh length. The HDG implementation currently carries one constant alpha;
 * the legacy Local policy instead requires an element-varying coefficient and
 * is therefore rejected rather than silently lowered to a different model.
 */
inline ResolvedHdgRegularization ResolveHdgRegularization(
    const mfem::ParMesh& mesh,
    const dg_igr::IGRAlphaLengthScalePolicy policy,
    const HdgRegularizationRequest& request)
{
  using dg::hdg::generic::RegularizationKind;
  if (!ValidHdgRegularizationRequest(request))
  {
    throw std::invalid_argument(
        "HDG IGR c_alpha and Euler trace penalty must be finite and positive");
  }
  if (request.kind == RegularizationKind::none)
  {
    return {};
  }
  if (policy == dg_igr::IGRAlphaLengthScalePolicy::Local)
  {
    throw std::invalid_argument(
        "HDG IGR does not support the element-local alpha length policy");
  }

  ResolvedHdgRegularization resolved;
  resolved.alpha_info = ComputeAlphaInfo(mesh, request.c_alpha, policy);
  if (!std::isfinite(resolved.alpha_info.alpha) ||
      resolved.alpha_info.alpha <= 0.0 ||
      !std::isfinite(resolved.alpha_info.dx_eff) ||
      resolved.alpha_info.dx_eff <= 0.0)
  {
    throw std::runtime_error(
        "HDG IGR mesh-scaled alpha resolution produced an invalid coefficient");
  }
  resolved.selection = {RegularizationKind::igr,
                        resolved.alpha_info.alpha,
                        request.euler_trace_penalty};
  return resolved;
}

} // namespace dg_sim::hdg

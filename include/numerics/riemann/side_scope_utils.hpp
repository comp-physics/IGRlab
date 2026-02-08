#pragma once

#include "numerics/riemann/face_side.hpp"

/**
 * @file numerics/riemann/side_scope_utils.hpp
 * @brief Shared explicit-side helpers for side-aware numerical flux calls.
 * @ingroup numerics_riemann
 */

namespace mfem
{
namespace riemann_detail
{

inline real_t ComputeFluxDotNForSide(const FaceSideExplicitFlux *explicit_flux,
                                     const FluxFunction &flux_fn,
                                     FaceSide side,
                                     const Vector &state,
                                     const Vector &normal,
                                     FaceElementTransformations &Tr,
                                     Vector &flux_dot_n)
{
  if (explicit_flux)
  {
    return explicit_flux->ComputeFluxDotNAtSide(state, normal, Tr, side, flux_dot_n);
  }
  return flux_fn.ComputeFluxDotN(state, normal, Tr, flux_dot_n);
}

inline real_t ComputeFluxDotNForSide(const FluxFunction &flux_fn,
                                     FaceSide side,
                                     const Vector &state,
                                     const Vector &normal,
                                     FaceElementTransformations &Tr,
                                     Vector &flux_dot_n)
{
  const auto *explicit_flux = dynamic_cast<const FaceSideExplicitFlux*>(&flux_fn);
  return ComputeFluxDotNForSide(explicit_flux, flux_fn, side, state, normal, Tr,
                                flux_dot_n);
}

inline real_t ComputeAvgFluxDotNForSide(const FaceSideExplicitFlux *explicit_flux,
                                        const FluxFunction &flux_fn,
                                        FaceSide side,
                                        const Vector &state1,
                                        const Vector &state2,
                                        const Vector &normal,
                                        FaceElementTransformations &Tr,
                                        Vector &flux_dot_n)
{
  if (explicit_flux)
  {
    return explicit_flux->ComputeAvgFluxDotNAtSide(state1, state2, normal, Tr,
                                                   side, flux_dot_n);
  }
  return flux_fn.ComputeAvgFluxDotN(state1, state2, normal, Tr, flux_dot_n);
}

inline real_t ComputeAvgFluxDotNForSide(const FluxFunction &flux_fn,
                                        FaceSide side,
                                        const Vector &state1,
                                        const Vector &state2,
                                        const Vector &normal,
                                        FaceElementTransformations &Tr,
                                        Vector &flux_dot_n)
{
  const auto *explicit_flux = dynamic_cast<const FaceSideExplicitFlux*>(&flux_fn);
  return ComputeAvgFluxDotNForSide(explicit_flux, flux_fn, side, state1, state2,
                                   normal, Tr, flux_dot_n);
}

} // namespace riemann_detail
} // namespace mfem

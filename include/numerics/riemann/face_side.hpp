#pragma once

#include "mfem.hpp"

/**
 * @file numerics/riemann/face_side.hpp
 * @brief Face-side context helpers for numerical fluxes.
 * @ingroup numerics_riemann
 */

namespace mfem
{

/// @brief Side identifier for face-based flux evaluation.
/// @ingroup numerics_riemann
enum class FaceSide { Left, Right };

/// @brief Optional side-explicit flux interface that avoids mutable side state.
/// @ingroup numerics_riemann
class FaceSideExplicitFlux
{
public:
  virtual ~FaceSideExplicitFlux() = default;

  virtual real_t ComputeFluxDotNAtSide(const Vector &state,
                                       const Vector &normal,
                                       FaceElementTransformations &Tr,
                                       FaceSide side,
                                       Vector &fluxDotN) const = 0;

  virtual real_t ComputeAvgFluxDotNAtSide(const Vector &state1,
                                          const Vector &state2,
                                          const Vector &normal,
                                          FaceElementTransformations &Tr,
                                          FaceSide side,
                                          Vector &fluxDotN) const = 0;
};

} // namespace mfem

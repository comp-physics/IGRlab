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

/// @brief Interface for fluxes that need face-side context.
/// @ingroup numerics_riemann
class FaceSideAwareFlux
{
public:
  virtual ~FaceSideAwareFlux() = default;
  virtual void SetFaceSide(FaceSide side) = 0;
};

} // namespace mfem

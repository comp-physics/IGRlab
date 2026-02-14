#pragma once

#include "mfem.hpp"
#include "utils/config/dg_igr_defaults.hpp"

/**
 * @file numerics/limiters/venkat.hpp
 * @brief Venkatakrishnan limiters (characteristic, face-normal).
 * @ingroup numerics_limiters
 */

namespace mfem
{

/**
 * @brief Options for Venkatakrishnan limiter.
 */
struct VenkatOptions
{
  int dim = mfem::dg_igr::EulerDefaults::kDim; ///< 2D supported here
  double gamma = mfem::dg_igr::EulerDefaults::kGamma;
  int slope_qord = 0;     ///< 0 => auto: 2*order+2
  double eps_kappa = 1.0; ///< Smoothness scale
};

/**
 * @brief Regular characteristic Venkatakrishnan limiter (x/y directions).
 */
void ApplyCharacteristicVenkatXY(mfem::ParFiniteElementSpace& vfes,
                                 mfem::ParGridFunction& U,
                                 const VenkatOptions& opt);

/**
 * @brief Face-normal characteristic Venkatakrishnan limiter (general meshes).
 */
void ApplyCharacteristicVenkatFaceNormal(mfem::ParFiniteElementSpace& vfes,
                                         mfem::ParGridFunction& U,
                                         const VenkatOptions& opt);

} // namespace mfem

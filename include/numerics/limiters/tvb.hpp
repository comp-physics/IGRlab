#pragma once

#include "igrlab/compat.hpp"

#include "mfem.hpp"
#include "utils/config/dg_igr_defaults.hpp"

/**
 * @file numerics/limiters/tvb.hpp
 * @brief Characteristic TVB limiters (including face-normal variant).
 * @ingroup numerics_limiters
 */

namespace mfem
{

/**
 * @brief Options for the characteristic TVB limiter with KXRCF shock flag.
 */
struct CharTVBOptions
{
  int dim = mfem::igrlab::EulerDefaults::
    kDim; ///< Spatial dimension (implemented for 2D)
  double gamma =
    mfem::igrlab::EulerDefaults::kGamma; ///< Ratio of specific heats
  double theta = 1.5;                    ///< Shu TVB parameter (approx 1-2)
  double M_tvb = 50.0;    ///< TVB threshold constant (tune per problem)
  double tau_flag = 0.05; ///< KXRCF-like density jump flag threshold
  int slope_qord = 0; ///< Quadrature for slope weights (0 => auto 2*order+2)
};

/**
 * @brief Characteristic TVB limiter with a simple KXRCF-style density sensor.
 *
 * @note Assumes Ordering::byNODES and DG (L2) space with nodal DOFs.
 */
void ApplyCharacteristicTVBLimiter(mfem::ParFiniteElementSpace& vfes,
                                   mfem::ParGridFunction& U,
                                   const CharTVBOptions& opt);

/**
 * @brief Face-normal TVB limiter (direction-invariant, general meshes).
 */
void ApplyCharacteristicTVBFaceNormal(mfem::ParFiniteElementSpace& vfes,
                                      mfem::ParGridFunction& U,
                                      const CharTVBOptions& opt);

} // namespace mfem

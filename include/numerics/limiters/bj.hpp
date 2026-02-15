#pragma once

#include "mfem.hpp"
#include "utils/config/dg_igr_defaults.hpp"

/**
 * @file numerics/limiters/bj.hpp
 * @brief Barth-Jespersen limiters (component-wise and characteristic).
 * @ingroup numerics_limiters
 */

namespace mfem
{

/**
 * @brief Options for a Barth–Jespersen (BJ) slope limiter (component-wise).
 */
struct BJOptions
{
  int dim =
    mfem::dg_igr::EulerDefaults::kDim; ///< Spatial dimension (2D implemented)
  int qord_avg =
    0; ///< Quadrature order to compute cell averages (0=>2*order+2)
  int slope_qord = 0;    ///< Quadrature order for slope weights (0=>2*order+2)
  bool limit_rho = true; ///< Limit density slope
  bool limit_momentum = true; ///< Limit momentum slopes
  bool limit_energy = true;   ///< Limit energy slope
  bool characteristic =
    false; ///< If true, limit in characteristic vars (not implemented here)
};

/**
 * @brief Apply Barth–Jespersen slope limiting (component-wise) in 2D.
 *
 * @note Assumes Ordering::byNODES and 2D.
 */
void ApplyBarthJespersenLimiter(mfem::ParFiniteElementSpace& vfes,
                                mfem::ParGridFunction& U,
                                const BJOptions& opt);

/**
 * @brief Options for characteristic Barth–Jespersen (slope) limiter.
 */
struct CharBJOptions
{
  int dim = mfem::dg_igr::EulerDefaults::
    kDim; ///< Spatial dimension (implemented for 2D)
  double gamma =
    mfem::dg_igr::EulerDefaults::kGamma; ///< Ratio of specific heats
  int slope_qord = 0;  ///< Quadrature for slope weights (0 => auto 2*order+2)
  bool limit_x = true; ///< limit in x-direction
  bool limit_y = true; ///< limit in y-direction
};

/**
 * @brief Characteristic Barth–Jespersen limiter (2D Euler, linear reconstruction).
 *
 * @note Degrades locally to first order in troubled cells, by design.
 * @note Assumes Ordering::byNODES and DG (L2) space with nodal DOFs.
 */
void ApplyCharacteristicBJLimiter(mfem::ParFiniteElementSpace& vfes,
                                  mfem::ParGridFunction& U,
                                  const CharBJOptions& opt);

/**
 * @brief Options for the face-normal characteristic BJ limiter.
 */
struct FaceBJOptions
{
  int dim = mfem::dg_igr::EulerDefaults::
    kDim; ///< Spatial dimension (implemented for 2D)
  double gamma =
    mfem::dg_igr::EulerDefaults::kGamma; ///< Ratio of specific heats
  int slope_qord = 0; ///< Quadrature for slope weights (0 => auto 2*order+2)
  double eps_rho = 1e-14; ///< Density floor used in characteristic transforms
};

/**
 * @brief Apply the face-normal characteristic Barth–Jespersen limiter.
 */
void ApplyFaceNormalCharacteristicBJLimiter(mfem::ParFiniteElementSpace& vfes,
                                            mfem::ParGridFunction& U,
                                            const FaceBJOptions& opt);

} // namespace mfem

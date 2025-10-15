#pragma once
#include "mfem.hpp"
#include <vector>

/**
 * @file limiters.hpp
 * @brief Positivity-preserving, characteristic TVB, and Barth–Jespersen slope
 *        limiters for DG Euler in MFEM (2D).
 */

// =========================== Zhang–Shu positivity limiter ===========================

/**
 * @brief Zhang–Shu (a.k.a. Shu) positivity-preserving limiter for DG Euler.
 *
 * Enforces non-negativity of density and pressure at a set of intra-element
 * “positivity points” by blending each cell’s polynomial solution toward its
 * cell average:
 *
 *   U_lim(x) = Ū + θ (U(x) − Ū),  with θ ∈ [0, 1] chosen per element,
 *
 * so that for all positivity points in the element:
 *   ρ ≥ rho_min,   p ≥ p_min,  with  p = (γ−1)\*(E − |m|²/(2ρ)).
 *
 * The procedure is conservative (preserves the cell average) unless the
 * average itself is inadmissible, in which case the element is set to a
 * constant admissible state with p = p_min.
 *
 * @note Assumes @p vfes uses @c Ordering::byNODES.
 * @note Operates locally on each element (no communication). In parallel,
 *       call @c sol.ExchangeFaceNbrData() after limiting to refresh ghosts.
 *
 * @param[in]  vfes            Vector FE space of @p U (vdim = dim+2).
 * @param[in,out] U            Conservative state (DG) to be limited in-place.
 * @param[in]  dim             Spatial dimension (2 or 3).
 * @param[in]  gamma           Specific heat ratio γ.
 * @param[in]  rho_min         Density floor (e.g., 1e-12 in your units).
 * @param[in]  p_min           Pressure floor (e.g., 1e-12 in your units).
 * @param[in]  quad_order      Quadrature order for positivity checks
 *                             (suggestion: max(2*p, 4)).
 * @param[in]  p_bisect_iters  Bisection iterations when enforcing p ≥ p_min
 *                             (default ~12 is plenty).
 */
void ApplyZhangShuPositivityLimiter(mfem::ParFiniteElementSpace &vfes,
                                    mfem::ParGridFunction &U,
                                    int dim, double gamma,
                                    double rho_min, double p_min,
                                    int quad_order,
                                    int p_bisect_iters = 12);

/** Utility coefficient to evaluate pressure p(U) for visualization/debug. */
struct PressureCoefficient : public mfem::Coefficient
{
  const mfem::ParGridFunction *U = nullptr;
  int dim = 2;
  double gamma = 1.4;
  mutable mfem::Vector state; // reuse buffer

  PressureCoefficient() = default;
  PressureCoefficient(const mfem::ParGridFunction &U_, int dim_, double g_)
    : U(&U_), dim(dim_), gamma(g_), state(dim_+2) {}

  void SetState(const mfem::ParGridFunction &U_) { U = &U_; }
  void SetDimGamma(int d, double g) { dim = d; gamma = g; state.SetSize(d+2); }

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override;
};

// ======================== Characteristic TVB slope limiter ==========================

/**
 * @brief Options for the characteristic TVB limiter with KXRCF shock flag.
 */
struct CharTVBOptions {
  int    dim        = 2;      ///< Spatial dimension (implemented for 2D)
  double gamma      = 1.4;    ///< Ratio of specific heats
  double theta      = 1.5;    ///< Shu TVB parameter (≈1–2)
  double M_tvb      = 50.0;   ///< TVB threshold constant (tune per problem)
  double tau_flag   = 0.05;   ///< KXRCF-like density jump flag threshold
  int    slope_qord = 0;      ///< Quadrature for slope weights (0 => auto 2*order+2)
};

/**
 * @brief Characteristic TVB limiter with a simple KXRCF-style density sensor.
 *
 * See implementation comments in limiters.cpp for details.
 *
 * @note Assumes @c Ordering::byNODES and DG (L2) space with nodal DOFs.
 */
void ApplyCharacteristicTVBLimiter(mfem::ParFiniteElementSpace &vfes,
                                   mfem::ParGridFunction &U,
                                   const CharTVBOptions &opt);

// ======================== Barth–Jespersen slope limiter =============================

/**
 * @brief Options for a Barth–Jespersen (BJ) slope limiter (component-wise).
 *
 * This is a classic vertex/pointwise monotonicity limiter: it limits the
 * *linear part* (slopes) so that the polynomial remains within the range of
 * neighboring cell averages at a prescribed set of intra-element evaluation
 * points (here we use FE nodes). We reconstruct a linear polynomial
 * U(x) = Ū + s_x·(x−x_c) + s_y·(y−y_c) with a single θ∈[0,1] applied per
 * component (minimum over points) to preserve local monotonicity.
 *
 * - Conservative by construction (mean Ū is preserved).
 * - Typically applied per conserved component; can be restricted to a subset.
 * - For p>1, this acts like a “linear rec” limiter (order reduction locally).
 */
struct BJOptions {
  int  dim            = 2;     ///< Spatial dimension (2D implemented)
  int  qord_avg       = 0;     ///< Quadrature order to compute cell averages (0=>2*order+2)
  int  slope_qord     = 0;     ///< Quadrature order for slope weights (0=>2*order+2)
  bool limit_rho      = true;  ///< Limit density slope
  bool limit_momentum = true;  ///< Limit momentum slopes
  bool limit_energy   = true;  ///< Limit energy slope
  bool characteristic = false; ///< If true, limit in characteristic vars (not implemented here)
};

/**
 * @brief Apply Barth–Jespersen slope limiting (component-wise) in 2D.
 *
 * - Builds neighbor-based min/max bounds from element means.
 * - Computes candidate DG slopes via ∫ grad φ_j (like TVB helper).
 * - Computes a single θ per component so that nodal values stay within bounds.
 * - Reconstructs a linear polynomial at the DOFs.
 *
 * @note Currently component-wise in conservative variables; characteristic
 *       BJ can be added later similarly to the TVB pathway.
 * @note Assumes @c Ordering::byNODES and 2D.
 */
void ApplyBarthJespersenLimiter(mfem::ParFiniteElementSpace &vfes,
                                mfem::ParGridFunction &U,
                                const BJOptions &opt);


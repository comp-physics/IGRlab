#pragma once
#include "mfem.hpp"
#include "dg_igr_defaults.hpp"
#include <vector>

/**
 * @file limiters.hpp
 * @brief Positivity-preserving, characteristic TVB, and BJ slope limiters.
 * @ingroup numerics_limiters
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
  int dim = mfem::dg_igr::EulerDefaults::kDim;
  double gamma = mfem::dg_igr::EulerDefaults::kGamma;
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
  int    dim        = mfem::dg_igr::EulerDefaults::kDim;    ///< Spatial dimension (implemented for 2D)
  double gamma      = mfem::dg_igr::EulerDefaults::kGamma;  ///< Ratio of specific heats
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

// ===================== Face-normal characteristic TVB (direction-invariant) =====================

/**
 * TVB limiter applied per *face normal* (works on general meshes).
 * For each face f of element K with unit outward normal n_f:
 *   - project DG slope to s_DG,n = n_f·(s_DGx, s_DGy)
 *   - build one-sided mean slopes d^-_n, d^+_n from neighbor means along n_f
 *   - do TVB minmod in characteristic space of A_n (approx with L_n ≈ n_x L_x + n_y L_y, etc.)
 *   - map back and accumulate a vector gradient:  sU += (R_n W_n) ⊗ n_f
 * Finally reconstruct a linear polynomial in K: U = Ū + sUx*(x-xc) + sUy*(y-yc).
 */
void ApplyCharacteristicTVBFaceNormal( mfem::ParFiniteElementSpace &vfes,
                                       mfem::ParGridFunction &U,
                                       const CharTVBOptions &opt );


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
  int  dim            = mfem::dg_igr::EulerDefaults::kDim;  ///< Spatial dimension (2D implemented)
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

// ===================== Characteristic Barth–Jespersen limiter =======================

/**
 * @brief Options for characteristic Barth–Jespersen (slope) limiter.
 *
 * The limiter scales each characteristic-direction slope (per wave) so that
 * pointwise values at face centers stay within the range of the cell mean and
 * the across-face neighbor mean, both projected to the same characteristic
 * basis (one per direction).
 */
struct CharBJOptions {
  int    dim        = mfem::dg_igr::EulerDefaults::kDim;    ///< Spatial dimension (implemented for 2D)
  double gamma      = mfem::dg_igr::EulerDefaults::kGamma;  ///< Ratio of specific heats
  int    slope_qord = 0;      ///< Quadrature for slope weights (0 => auto 2*order+2)
  bool   limit_x    = true;   ///< limit in x-direction
  bool   limit_y    = true;   ///< limit in y-direction
};

/**
 * @brief Characteristic Barth–Jespersen limiter (2D Euler, linear reconstruction).
 *
 * Steps per element:
 *  1) Compute cell mean, one-sided mean differences, and DG candidate slopes.
 *  2) Build characteristic transforms L_x/R_x and L_y/R_y at the mean state.
 *  3) For dir∈{x,y} and wave k, predict value at the two face centers and
 *     constrain with neighbor-based min/max to compute θ ∈ [0,1].
 *  4) Scale the characteristic slopes by θ and map back to conserved slopes.
 *  5) Overwrite the element DOFs with the linear polynomial Ū + s_x·dx + s_y·dy.
 *
 * @note Degrades locally to first order in troubled cells, by design.
 * @note Assumes @c Ordering::byNODES and DG (L2) space with nodal DOFs.
 */
void ApplyCharacteristicBJLimiter(mfem::ParFiniteElementSpace &vfes,
                                  mfem::ParGridFunction &U,
                                  const CharBJOptions &opt);

// ==================== Face-normal characteristic Barth–Jespersen ====================

/**
 * @brief Options for the face-normal characteristic BJ limiter.
 *
 * Works on arbitrary meshes: for each face, build the *normal/tangent* frame,
 * rotate the local state (and neighbor means) so that the normal aligns with x,
 * apply 1D (normal) BJ limiting in characteristic space, and compute a single
 * element-wise θ ∈ [0,1] that scales the element gradient.
 *
 * Note: This limiter is *monotonicity preserving* (prevents new extrema) but is
 *       not positivity-preserving. Keep your Zhang–Shu PP limiter in the
 *       pipeline after this one.
 */
struct FaceBJOptions {
  int    dim        = mfem::dg_igr::EulerDefaults::kDim;    ///< Spatial dimension (implemented for 2D)
  double gamma      = mfem::dg_igr::EulerDefaults::kGamma;  ///< Ratio of specific heats
  int    slope_qord = 0;      ///< Quadrature for slope weights (0 => auto 2*order+2)
  double eps_rho    = 1e-14;  ///< Density floor used in characteristic transforms
};

/**
 * @brief Apply the face-normal characteristic Barth–Jespersen limiter.
 *
 * For each element:
 *   - Compute cell mean Ū and candidate slopes sDGx, sDGy.
 *   - For each incident face:
 *       • compute face center x_f and outward *approximate* normal n ∝ (x_f - x_c),
 *       • rotate (m_x, m_y) to (m_n, m_t) using (n, t),
 *       • predict U at x_f via Ū + sDGx*dx + sDGy*dy, rotate it,
 *       • project onto 1D Euler characteristics along n (R/L of A_n),
 *       • enforce BJ bound vs neighbor mean (also rotated) to get a θ_f ∈ [0,1].
 *   - Use θ = min_f θ_f, and rescale both slopes: sDGx ← θ sDGx, sDGy ← θ sDGy.
 *   - Reconstruct linear polynomial at DOFs.
 *
 * This is rotation- and mesh-orientation-invariant (uses face normals).
 */
void ApplyFaceNormalCharacteristicBJLimiter(mfem::ParFiniteElementSpace &vfes,
                                            mfem::ParGridFunction &U,
                                            const FaceBJOptions &opt);

// ======================= Venkatakrishnan limiter options & APIs =======================

/** Options for Venkatakrishnan limiter. Reuses your slope quadrature idea; eps_kappa
 *  controls smoothness: ε = eps_kappa * h^3 (regular) or * (h_n)^3 (face-normal).
 */
struct VenkatOptions {
  int    dim        = mfem::dg_igr::EulerDefaults::kDim;    ///< 2D supported here
  double gamma      = mfem::dg_igr::EulerDefaults::kGamma;
  int    slope_qord = 0;      ///< 0 => auto: 2*order+2
  double eps_kappa  = 1.0;    ///< Smoothness scale
};

/** Regular characteristic Venkatakrishnan limiter (x/y directions). */
void ApplyCharacteristicVenkatXY(mfem::ParFiniteElementSpace &vfes,
                                 mfem::ParGridFunction &U,
                                 const VenkatOptions &opt);

/** Face-normal characteristic Venkatakrishnan limiter (general meshes). */
void ApplyCharacteristicVenkatFaceNormal(mfem::ParFiniteElementSpace &vfes,
                                         mfem::ParGridFunction &U,
                                         const VenkatOptions &opt);

// =================== Krivodonova / BDS hierarchical modal limiter ===================

/**
 * @brief Options for the hierarchical modal limiter (Krivodonova / BDS style).
 *
 * Works element-locally: projects the DG polynomial onto a hierarchical
 * total-degree basis on the reference element, isolates the highest total-degree
 * block (degree p), and scales only that block so that, at face quadrature
 * points, the solution stays within neighbor-based bounds. Preserves the cell
 * mean. Designed for 2D quads and tris; implemented here for quads (tensor meshes).
 */
struct KrivoOptions {
  int    dim        = 2;        ///< spatial dimension (2)
  double gamma      = 1.4;      ///< EOS, only used if you later switch to characteristic space
  int    face_qord  = 0;        ///< quadrature on each face (0 => auto: p+2)
  double inflate    = 1.0;      ///< widen neighbor bounds: [umin-ε, umax+ε], ε = inflate * |range|
  bool   conservative_only = true; ///< if true, limit component-wise (robust). If false, you can wire characteristic transform.
};

/**
 * @brief Element-local hierarchical modal limiter (Krivodonova/BDS) for quads.
 *
 * Assumptions:
 *  - L2 (DG) space with byNODES ordering (your setup).
 *  - Quadrilateral reference element [-1,1]^2; nodal DOFs provide samples in ref element.
 *  - Limits only the highest total-degree block (degree p) by scaling factor α∈[0,1],
 *    chosen to avoid overshoot at face quadrature points compared to neighbor
 *    min/max envelopes (optionally inflated).
 *  - Preserves the element-wise cell average exactly.
 *
 * Notes:
 *  - This is a robust, *accuracy-preserving* limiter for p>1: smooth regions keep
 *    their high modes; only near discontinuities the p-th order content is clipped.
 *  - Implemented component-wise by default (characteristic variant can be added later).
 */
void ApplyKrivodonovaHierarchicalLimiter(mfem::ParFiniteElementSpace &vfes,
                                         mfem::ParGridFunction &U,
                                         const KrivoOptions &opt);

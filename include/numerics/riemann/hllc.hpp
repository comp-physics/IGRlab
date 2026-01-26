#pragma once

#include <algorithm>
#include <cmath>
#include "mfem.hpp"

/**
 * @file numerics/riemann/hllc.hpp
 * @brief HLLC numerical flux for Euler equations.
 * @ingroup numerics_riemann
 */

namespace mfem
{

/**
 * @brief HLLC flux for compressible Euler, computed in the face-normal direction.
 *
 * Uses unit-normal HLLC formulas internally, then scales by |nor| to match MFEM's
 * convention where fluxN terms are evaluated with the (possibly non-unit) normal.
 *
 * Assumptions:
 *  - state vector is conservative [rho, rho*u(0),...,rho*u(dim-1), rho*E]
 *  - gamma is provided in the constructor
 *  - fluxFunction corresponds to Euler and returns F·n via ComputeFluxDotN
 *
 * API mimics RusanovFlux to minimize changes at call sites.
 */
class HLLCFlux : public NumericalFlux
{
public:
  /**
   * @brief Constructor for a flux function
   * @param fluxFunction flux function F(u,x) (e.g., EulerFlux)
   * @param gamma_ratio specific heat ratio gamma
   */
  HLLCFlux(const FluxFunction &fluxFunction, real_t gamma_ratio);

  /**
   * @brief  Normal numerical flux Fhat(u^-,u^+,x) n via HLLC
   *
   * @param[in] state1 state value (u^-) at a point from the first element
   * @param[in] state2 state value (u^+) at a point from the second element
   * @param[in] nor normal vector (not a unit vector)
   * @param[in] Tr face element transformation
   * @param[out] flux HLLC normal flux (scaled by |nor|)
   * @return speed scale for CFL, here max(|S_L|, |S_R|)
   */
  real_t Eval(const Vector &state1, const Vector &state2,
              const Vector &nor, FaceElementTransformations &Tr,
              Vector &flux) const override;

  /**
   * @brief  Jacobian of normal numerical flux, diagonal-stabilized
   * @note We use the same diagonal augmentation pattern as Rusanov, replacing
   *       lambda with max(|S_L|,|S_R|). Exact HLLC Jacobians are lengthy; this
   *       robust approximation works well for explicit DG.
   */
  void Grad(int side, const Vector &state1, const Vector &state2,
            const Vector &nor, FaceElementTransformations &Tr,
            DenseMatrix &grad) const override;

  /**
   * @brief  Average normal numerical flux variant
   * @note For robustness and simplicity we reuse Eval(); this matches the
   *       dominant code path in explicit DG runs.
   */
  real_t Average(const Vector &state1, const Vector &state2,
                 const Vector &nor, FaceElementTransformations &Tr,
                 Vector &flux) const override;

  /// @brief Approximate Jacobian for Average; same as Grad().
  void AverageGrad(int side, const Vector &state1, const Vector &state2,
                   const Vector &nor, FaceElementTransformations &Tr,
                   DenseMatrix &grad) const override;

protected:
#ifndef MFEM_THREAD_SAFE
  mutable Vector fluxN1, fluxN2;   ///< parity with Rusanov; not strictly needed here
  mutable DenseMatrix JDotN;
#endif

  const real_t gamma;              ///< specific heat ratio

  // Helpers
  inline int DimFromState(const Vector &U) const
  {
    const int neq = U.Size();
    MFEM_ASSERT(neq >= 3, "Euler requires neq >= 3.");
    return neq - 2;
  }

  static inline void UnitNormalAndMag(const Vector &nor, Vector &nhat, real_t &nmag)
  {
    nmag = nor.Norml2();
    if (nmag > 0.0) { nhat = nor; nhat /= nmag; }
    else { nhat.SetSize(nor.Size()); nhat = 0.0; }
  }

  struct Prim
  {
    real_t rho, p, a, un; // density, pressure, sound speed, normal velocity
    Vector u;             // velocity (size dim)
  };

  // Convert conservative U -> primitive W; un = u·n_hat
  void ConsToPrim(const Vector &U, const Vector &nhat, Prim &W) const;

  // F(U)·n_hat (unit normal)
  void FluxDotNhat(const Vector &U, const Prim &W, const Vector &nhat, Vector &Fn_hat) const;

  // Einfeldt/Toro wave speeds
  void HLLC_WaveSpeeds(const Prim &WL, const Prim &WR,
                       real_t &SL, real_t &Sstar, real_t &SR) const;

  static inline real_t SpeedScale(real_t SL, real_t SR)
  {
    return std::max(std::abs(SL), std::abs(SR));
  }
};

} // namespace mfem

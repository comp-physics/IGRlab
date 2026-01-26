#pragma once

#include <limits>
#include "mfem.hpp"

/**
 * @file numerics/riemann/diffusive_rusanov.hpp
 * @brief Diffusive Rusanov (Lax-Friedrichs) numerical flux.
 * @ingroup numerics_riemann
 */

namespace mfem
{

/**
 * @brief Diffusive Rusanov flux, a Rusanov (local Lax–Friedrichs) variant with
 * tunable extra dissipation:
 *
 *    Fhat n = 1/2(F(u^+,x)n + F(u^-,x)n) - 1/2 lambda_tilde (u^+ - u^-),
 *
 * where the inflated speed lambda_tilde is built from the baseline
 *   lambda = max(|dF(u^+,x)/du^+·n|, |dF(u^-,x)/du^-·n|)
 * using
 *   lambda_tilde = min( cap,
 *             beta * (1 + s(u^-,u^+)) * lambda + delta ),
 * with beta >= 1 a global multiplier, delta >= 0 a floor, an optional bounded
 * jump sensor s in [0, sensor_max], and an optional cap to avoid blow-ups.
 *
 * @note Assumes monotonous |dF(u,x)/du·n| in u (same assumption as RusanovFlux).
 */
class DiffusiveRusanovFlux : public NumericalFlux
{
public:
  /**
   * @brief Constructor for a flux function with tunable diffusion.
   * @param fluxFunction flux function F(u,x)
   * @param beta   >= 1.0 multiplicative safety factor on lambda (default 1.5)
   * @param delta  >= 0.0 additive floor on lambda (default 0.0)
   * @param cap    upper bound on lambda_tilde (default +infinity)
   * @param sensor_k   >= 0.0 strength of jump-based boost (0 disables sensor)
   * @param sensor_max > 0.0 clamp for sensor contribution
   *
   * The jump sensor uses a relative L1 jump:
   *   s = clamp( sensor_k * ||u^+ - u^-||_1 / (||u^+||_1 + ||u^-||_1 + eps), 0, sensor_max ).
   */
  DiffusiveRusanovFlux(const FluxFunction &fluxFunction,
                       real_t beta = 1.5,
                       real_t delta = 0.0,
                       real_t cap   = std::numeric_limits<real_t>::infinity(),
                       real_t sensor_k = 0.0,
                       real_t sensor_max = 1.0);

  /**
   * @brief  Normal numerical flux Fhat(u^-,u^+,x) n with inflated lambda_tilde.
   * @note Systems of equations are treated component-wise.
   *
   * @param[in] state1 state value (u^-) from the first element (num_equations)
   * @param[in] state2 state value (u^+) from the second element (num_equations)
   * @param[in] nor normal vector (not a unit vector) (dim)
   * @param[in] Tr face element transformation
   * @param[out] flux Fhat n = 1/2(F(u^+,x)n + F(u^-,x)n) - 1/2 lambda_tilde(u^+ - u^-)
   * @return max(|dF(u^+,x)/du^+·n|, |dF(u^-,x)/du^-·n|)  (baseline lambda, for CFL accounting)
   */
  real_t Eval(const Vector &state1, const Vector &state2,
              const Vector &nor, FaceElementTransformations &Tr,
              Vector &flux) const override;

  /**
   * @brief  Jacobian of normal numerical flux Fhat(u^-,u^+,x) n using lambda_tilde.
   * @note The Jacobian of flux J n must be implemented in
   * FluxFunction::ComputeFluxJacobianDotN().
   *
   * @param[in] side gradient w.r.t the first (u^-) or second argument (u^+)
   * @param[in] state1 state value (u^-) (num_equations)
   * @param[in] state2 state value (u^+) (num_equations)
   * @param[in] nor normal vector (not a unit vector) (dim)
   * @param[in] Tr face element transformation
   * @param[out] grad Jacobian of Fhat(u^-,u^+,x) n
   * side = 1:  1/2 J(u^-,x)n + 1/2 lambda_tilde
   * side = 2:  1/2 J(u^+,x)n - 1/2 lambda_tilde
   */
  void Grad(int side, const Vector &state1, const Vector &state2,
            const Vector &nor, FaceElementTransformations &Tr,
            DenseMatrix &grad) const override;

  /**
   * @brief  Average normal numerical flux over [u^-, u^+] in the second argument:
   * Fhat(u^-,u,x) n with lambda_tilde halved as in Rusanov's averaged form.
   * @note Requires FluxFunction::ComputeAvgFluxDotN().
   * @note Systems of equations are treated component-wise.
   *
   * @param[in] state1 state value (u^-) (num_equations)
   * @param[in] state2 state value (u^+) (num_equations)
   * @param[in] nor normal vector (not a unit vector) (dim)
   * @param[in] Tr face element transformation
   * @param[out] flux Fhat n (averaged form)
   * @return max eigenvalue for CFL accounting
   */
  real_t Average(const Vector &state1, const Vector &state2,
                 const Vector &nor, FaceElementTransformations &Tr,
                 Vector &flux) const override;

  /// @brief AverageGrad for the averaged form.
  void AverageGrad(int side, const Vector &state1, const Vector &state2,
                   const Vector &nor, FaceElementTransformations &Tr,
                   DenseMatrix &grad) const override;

private:
  /// Compute inflated lambda_tilde (without |n| scaling) from speeds and states.
  real_t EffectiveLambda(real_t speed1, real_t speed2,
                         const Vector &state1, const Vector &state2) const;

  // Parameters
  real_t beta_, delta_, cap_;
  real_t sensor_k_, sensor_max_;

#ifndef MFEM_THREAD_SAFE
  mutable Vector fluxN1, fluxN2;
  mutable DenseMatrix JDotN;
#endif
};

} // namespace mfem

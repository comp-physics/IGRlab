#ifndef MFEM_FLUX_FCNS
#define MFEM_FLUX_FCNS

#include "mfem.hpp"
// #include "hyperbolic.hpp"

namespace mfem
{

/**
 * @brief Diffusive Rusanov flux, a Rusanov (local Lax–Friedrichs) variant with
 * tunable extra dissipation:
 *
 *    F̂ n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½ λ̃ (u⁺ - u⁻),
 *
 * where the inflated speed λ̃ is built from the baseline
 *   λ = max(|dF(u⁺,x)/du⁺⋅n|, |dF(u⁻,x)/du⁻⋅n|)
 * using
 *   λ̃ = min( cap,
 *             β * (1 + s(u⁻,u⁺)) * λ + δ ),
 * with β ≥ 1 a global multiplier, δ ≥ 0 a floor, an optional bounded
 * jump sensor s ∈ [0, sensor_max], and an optional cap to avoid blow-ups.
 *
 * @note Assumes monotonous |dF(u,x)/du⋅n| in u (same assumption as RusanovFlux).
 */
class DiffusiveRusanovFlux : public NumericalFlux
{
public:
   /**
    * @brief Constructor for a flux function with tunable diffusion.
    * @param fluxFunction flux function F(u,x)
    * @param beta   ≥ 1.0 multiplicative safety factor on λ (default 1.5)
    * @param delta  ≥ 0.0 additive floor on λ (default 0.0)
    * @param cap    upper bound on λ̃ (default +∞)
    * @param sensor_k   ≥ 0.0 strength of jump-based boost (0 disables sensor)
    * @param sensor_max > 0.0 clamp for sensor contribution
    *
    * The jump sensor uses a relative L1 jump:
    *   s = clamp( sensor_k * ||u⁺-u⁻||₁ / (||u⁺||₁ + ||u⁻||₁ + eps), 0, sensor_max ).
    */
   DiffusiveRusanovFlux(const FluxFunction &fluxFunction,
                        real_t beta = 1.5,
                        real_t delta = 0.0,
                        real_t cap   = std::numeric_limits<real_t>::infinity(),
                        real_t sensor_k = 0.0,
                        real_t sensor_max = 1.0);

   /**
    * @brief  Normal numerical flux F̂(u⁻,u⁺,x) n with inflated λ̃.
    * @note Systems of equations are treated component-wise.
    *
    * @param[in] state1 state value (u⁻) from the first element (num_equations)
    * @param[in] state2 state value (u⁺) from the second element (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] flux F̂ n = ½(F(u⁺,x)n + F(u⁻,x)n) - ½λ̃(u⁺ - u⁻)
    * @return max(|dF(u⁺,x)/du⁺⋅n|, |dF(u⁻,x)/du⁻⋅n|)  (baseline λ, for CFL accounting)
    */
   real_t Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, FaceElementTransformations &Tr,
               Vector &flux) const override;

   /**
    * @brief  Jacobian of normal numerical flux F̂(u⁻,u⁺,x) n using λ̃.
    * @note The Jacobian of flux J n must be implemented in
    * FluxFunction::ComputeFluxJacobianDotN().
    *
    * @param[in] side gradient w.r.t the first (u⁻) or second argument (u⁺)
    * @param[in] state1 state value (u⁻) (num_equations)
    * @param[in] state2 state value (u⁺) (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] grad Jacobian of F̂(u⁻,u⁺,x) n
    * side = 1:  ½J(u⁻,x)n + ½λ̃
    * side = 2:  ½J(u⁺,x)n - ½λ̃
    */
   void Grad(int side, const Vector &state1, const Vector &state2,
             const Vector &nor, FaceElementTransformations &Tr,
             DenseMatrix &grad) const override;

   /**
    * @brief  Average normal numerical flux over [u⁻, u⁺] in the second argument:
    * F̂(u⁻,u,x) n with λ̃ halved as in Rusanov’s averaged form.
    * @note Requires FluxFunction::ComputeAvgFluxDotN().
    * @note Systems of equations are treated component-wise.
    *
    * @param[in] state1 state value (u⁻) (num_equations)
    * @param[in] state2 state value (u⁺) (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] flux ½(F̄(u⁻,u⁺,x)n + F(u⁻,x)n) - ¼λ̃(u⁺ - u⁻)
    * @return max(|dF(u⁺,x)/du⁺⋅n|, |dF(u⁻,x)/du⁻⋅n|)
    */
   real_t Average(const Vector &state1, const Vector &state2,
                  const Vector &nor, FaceElementTransformations &Tr,
                  Vector &flux) const override;

   /**
    * @brief  Jacobian of average normal numerical flux over [u⁻, u⁺] in the
    * second argument F̂(u⁻,u,x) n with λ̃.
    * @note Requires FluxFunction::ComputeAvgFluxDotN() and
    * FluxFunction::ComputeFluxJacobianDotN().
    * @note Only diagonal terms of J n are considered (decoupled components).
    *
    * @param[in] side gradient w.r.t the first (u⁻) or second argument (u⁺)
    * @param[in] state1 state value (u⁻) (num_equations)
    * @param[in] state2 state value (u⁺) (num_equations)
    * @param[in] nor normal vector (not a unit vector) (dim)
    * @param[in] Tr face element transformation
    * @param[out] grad Jacobian of F̄(u⁻,u⁺,x) n
    * side = 1:
    *    ½(F̄(u⁻,u⁺,x)n - F(u⁻,x)n) / (u⁺ - u⁻) - ½J(u⁻,x)n + ¼λ̃
    * side = 2:
    *    ½(F(u⁺,x)n - F̄(u⁻,u⁺,x)n) / (u⁺ - u⁻) - ¼λ̃
    */
   void AverageGrad(int side, const Vector &state1, const Vector &state2,
                    const Vector &nor, FaceElementTransformations &Tr,
                    DenseMatrix &grad) const override;

   // --- Tunable parameters (runtime adjustable) ---
   void SetBeta(real_t beta)             { beta_ = beta; }              ///< β ≥ 1
   void SetDelta(real_t delta)           { delta_ = delta; }            ///< δ ≥ 0
   void SetCap(real_t cap)               { cap_ = cap; }                ///< cap
   void SetSensor(real_t k, real_t maxv) { sensor_k_ = k; sensor_max_ = maxv; }

protected:
#ifndef MFEM_THREAD_SAFE
   mutable Vector fluxN1, fluxN2;
   mutable DenseMatrix JDotN;
#endif

   /// Compute inflated λ̃ (without |n| scaling) from speeds and states.
   real_t EffectiveLambda(real_t speed1, real_t speed2,
                          const Vector &state1, const Vector &state2) const
   {
      real_t lam = std::max(speed1, speed2); // baseline λ from FluxFunction
      if (sensor_k_ > 0.0)
      {
         // Relative L1 jump sensor in [0, sensor_max_]
         double num = 0.0, den = 0.0, eps = 1e-14;
         const int m = state1.Size();
         for (int i = 0; i < m; ++i)
         {
            const double a = state1(i), b = state2(i);
            num += std::abs(a - b);
            den += std::abs(a) + std::abs(b);
         }
         const double rel = num / (den + eps);
         const double s = std::min(sensor_k_ * rel, (double)sensor_max_);
         lam *= (1.0 + s);
      }
      lam = beta_ * lam + delta_;
      if (lam > cap_) { lam = cap_; }
      return lam;
   }

   // Parameters
   real_t beta_, delta_, cap_;
   real_t sensor_k_, sensor_max_;
};

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
    * @param gamma_ratio specific heat ratio γ
    */
   HLLCFlux(const FluxFunction &fluxFunction, real_t gamma_ratio);

   /**
    * @brief  Normal numerical flux F̂(u⁻,u⁺,x) n via HLLC
    *
    * @param[in] state1 state value (u⁻) at a point from the first element
    * @param[in] state2 state value (u⁺) at a point from the second element
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
    *       λ with max(|S_L|,|S_R|). Exact HLLC Jacobians are lengthy; this
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
   mutable Vector fluxN1, fluxN2;   ///< for parity with Rusanov; not strictly needed here
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

   // Convert conservative U -> primitive W; un = u·n̂
   void ConsToPrim(const Vector &U, const Vector &nhat, Prim &W) const;

   // F(U)·n̂ (unit normal)
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

#endif // MFEM_FLUX_FCNS

// Implementation of hyperbolic conservation laws based on MFEM

#include "mfem.hpp"
#include "BdrHyperbolicFluxIntegrator.hpp"
#include "fluxfunctions.hpp"
// #include "hyperbolic.hpp"
// #include "nonlinearform.hpp"
// #include "pnonlinearform.hpp"

namespace mfem
{

DiffusiveRusanovFlux::DiffusiveRusanovFlux(const FluxFunction &fluxFunction,
                                           real_t beta,
                                           real_t delta,
                                           real_t cap,
                                           real_t sensor_k,
                                           real_t sensor_max)
   : NumericalFlux(fluxFunction),
     beta_(beta), delta_(delta), cap_(cap),
     sensor_k_(sensor_k), sensor_max_(sensor_max)
{
#ifndef MFEM_THREAD_SAFE
   fluxN1.SetSize(fluxFunction.num_equations);
   fluxN2.SetSize(fluxFunction.num_equations);
   // JDotN.SetSize(fluxFunction.num_equations);
#endif
}

real_t DiffusiveRusanovFlux::Eval(const Vector &state1, const Vector &state2,
                                  const Vector &nor, FaceElementTransformations &Tr,
                                  Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
   const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

   // Inflate λ and then scale by |n|
   const real_t lam     = EffectiveLambda(speed1, speed2, state1, state2);
   const real_t scaled  = lam * nor.Norml2();

   const int m = fluxFunction.num_equations;
   for (int i = 0; i < m; i++)
   {
      flux(i) = 0.5*(scaled*(state1(i) - state2(i)) + (fluxN1(i) + fluxN2(i)));
   }

   // Return the baseline max eigenvalue (as in RusanovFlux) for CFL accounting
   return std::max(speed1, speed2);
}

void DiffusiveRusanovFlux::Grad(int side, const Vector &state1, const Vector &state2,
                                const Vector &nor, FaceElementTransformations &Tr,
                                DenseMatrix &grad) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
   const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

   const real_t lam    = EffectiveLambda(speed1, speed2, state1, state2);
   const real_t scaled = lam * nor.Norml2();

   const int m = fluxFunction.num_equations;
   if (side == 1)
   {
      fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, grad);
      for (int i = 0; i < m; i++) { grad(i,i) += 0.5 * scaled; }
   }
   else
   {
      fluxFunction.ComputeFluxJacobianDotN(state2, nor, Tr, grad);
      for (int i = 0; i < m; i++) { grad(i,i) -= 0.5 * scaled; }
   }
}

real_t DiffusiveRusanovFlux::Average(const Vector &state1, const Vector &state2,
                                     const Vector &nor, FaceElementTransformations &Tr,
                                     Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
   const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
   const real_t speed2 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN2);

   const real_t lam    = EffectiveLambda(speed1, speed2, state1, state2);
   const real_t scaled = lam * nor.Norml2() * 0.5; // MFEM halves in Average

   const int m = fluxFunction.num_equations;
   for (int i = 0; i < m; i++)
   {
      flux(i) = 0.5*(scaled*(state1(i) - state2(i)) + (fluxN1(i) + fluxN2(i)));
   }

   // Return baseline max eigenvalue
   return std::max(speed1, speed2);
}

void DiffusiveRusanovFlux::AverageGrad(int side, const Vector &state1,
                                       const Vector &state2,
                                       const Vector &nor, FaceElementTransformations &Tr,
                                       DenseMatrix &grad) const
{
#ifdef MFEM_THREAD_SAFE
   Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif

#if defined(MFEM_USE_DOUBLE)
   constexpr real_t tol = 1e-12;
#elif defined(MFEM_USE_SINGLE)
   constexpr real_t tol = 4e-6;
#else
#error "Only single and double precision are supported!"
   constexpr real_t tol = 1.;
#endif

   auto equal_check = [=](real_t a, real_t b) -> bool
   {
      return std::abs(a - b) <= tol * std::abs(a + b);
   };

   if (side == 1)
   {
#ifdef MFEM_THREAD_SAFE
      DenseMatrix JDotN(fluxFunction.num_equations);
#else
      JDotN.SetSize(fluxFunction.num_equations);
#endif
      const real_t speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
      const real_t speed2 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN2);
      fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, JDotN);

      const real_t lam    = EffectiveLambda(speed1, speed2, state1, state2);
      const real_t scaled = lam * nor.Norml2() * 0.5;

      grad = 0.0;

      const int m = fluxFunction.num_equations;
      for (int i = 0; i < m; i++)
      {
         // Only diagonal terms of J considered; skip if states are equal
         if (equal_check(state1(i), state2(i))) { continue; }
         grad(i,i) = 0.5 * ((fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i))
                            - JDotN(i,i) + scaled);
      }
   }
   else
   {
      const real_t speed1 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN1);
      const real_t speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

      const real_t lam    = EffectiveLambda(speed1, speed2, state1, state2);
      const real_t scaled = lam * nor.Norml2() * 0.5;

      grad = 0.0;

      const int m = fluxFunction.num_equations;
      for (int i = 0; i < m; i++)
      {
         if (equal_check(state1(i), state2(i))) { continue; }
         grad(i,i) = 0.5 * ((fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i)) - scaled);
      }
   }
}

HLLCFlux::HLLCFlux(const FluxFunction &fluxFunction, real_t gamma_ratio)
   : NumericalFlux(fluxFunction), gamma(gamma_ratio)
{
#ifndef MFEM_THREAD_SAFE
   fluxN1.SetSize(fluxFunction.num_equations);
   fluxN2.SetSize(fluxFunction.num_equations);
   JDotN.SetSize(fluxFunction.num_equations);
#endif
}

void HLLCFlux::ConsToPrim(const Vector &U, const Vector &nhat, Prim &W) const
{
   const int neq = U.Size();
   const int dim = neq - 2;

   const real_t rho = U(0);
   MFEM_ASSERT(rho > 0.0, "Non-positive density in ConsToPrim.");
   W.rho = rho;

   if (W.u.Size() != dim) { W.u.SetSize(dim); }
   real_t u2 = 0.0;
   for (int d = 0; d < dim; ++d)
   {
      const real_t md = U(1 + d);
      const real_t ud = md / rho;
      W.u(d) = ud;
      u2 += ud * ud;
   }

   const real_t E = U(1 + dim);
   real_t p = (gamma - 1.0) * (E - 0.5 * rho * u2);
   if (p < 0.0) p = 0.0;                 // tiny clip for robustness
   W.p = p;
   W.a = std::sqrt(std::max(gamma * p / rho, (real_t)0.0));

   real_t un = 0.0;
   for (int d = 0; d < dim; ++d) { un += W.u(d) * nhat(d); }
   W.un = un;
}

void HLLCFlux::FluxDotNhat(const Vector &U, const Prim &W, const Vector &nhat, Vector &Fn_hat) const
{
   const int neq = U.Size();
   const int dim = neq - 2;
   if (Fn_hat.Size() != neq) { Fn_hat.SetSize(neq); }

   const real_t rho = W.rho;
   const real_t p   = W.p;
   const real_t un  = W.un;

   // mass
   Fn_hat(0) = rho * un;

   // momentum: rho*u_i*un + p*n_i
   for (int d = 0; d < dim; ++d)
   {
      Fn_hat(1 + d) = rho * W.u(d) * un + p * nhat(d);
   }

   // energy: (E + p)*un
   const real_t E = U(1 + dim);
   Fn_hat(1 + dim) = (E + p) * un;
}

void HLLCFlux::HLLC_WaveSpeeds(const Prim &WL, const Prim &WR,
                               real_t &SL, real_t &Sstar, real_t &SR) const
{
   // Einfeldt bounds
   SL = std::min(WL.un - WL.a, WR.un - WR.a);
   SR = std::max(WL.un + WL.a, WR.un + WR.a);

   // Toro's contact speed (PVRS-like)
   const real_t num = (WR.p - WL.p)
                    + WL.rho * WL.un * (SL - WL.un)
                    - WR.rho * WR.un * (SR - WR.un);
   const real_t den = WL.rho * (SL - WL.un) - WR.rho * (SR - WR.un);
   if (std::abs(den) > 1e-14) { Sstar = num / den; }
   else { Sstar = 0.5 * (SL + SR); }
}

real_t HLLCFlux::Eval(const Vector &state1, const Vector &state2,
                      const Vector &nor, FaceElementTransformations &Tr,
                      Vector &flux) const
{
   const int neq = fluxFunction.num_equations;
   const int dim = neq - 2;
   if (flux.Size() != neq) { flux.SetSize(neq); }

   // n̂ and |n|
#ifdef MFEM_THREAD_SAFE
   Vector nhat(nor.Size());
#else
   static thread_local Vector nhat; nhat.SetSize(nor.Size());
#endif
   real_t nmag = 0.0;
   UnitNormalAndMag(nor, nhat, nmag);

   // Left/right primitive states
   Prim WL, WR;
   ConsToPrim(state1, nhat, WL);
   ConsToPrim(state2, nhat, WR);

   // Wave speeds
   real_t SL=0.0, Sstar=0.0, SR=0.0;
   HLLC_WaveSpeeds(WL, WR, SL, Sstar, SR);

   // Fluxes with unit normal
#ifdef MFEM_THREAD_SAFE
   Vector FnL_hat(neq), FnR_hat(neq), Fstar_hat(neq);
#else
   static thread_local Vector FnL_hat, FnR_hat, Fstar_hat;
   FnL_hat.SetSize(neq); FnR_hat.SetSize(neq); Fstar_hat.SetSize(neq);
#endif
   FluxDotNhat(state1, WL, nhat, FnL_hat);
   FluxDotNhat(state2, WR, nhat, FnR_hat);

   // Upwind branches
   if (SL >= 0.0)
   {
      for (int i = 0; i < neq; ++i) { flux(i) = FnL_hat(i) * nmag; }
      return SpeedScale(SL, SR);
   }
   if (SR <= 0.0)
   {
      for (int i = 0; i < neq; ++i) { flux(i) = FnR_hat(i) * nmag; }
      return SpeedScale(SL, SR);
   }

   // Star states U*_L and U*_R
#ifdef MFEM_THREAD_SAFE
   Vector ULstar(neq), URstar(neq);
#else
   static thread_local Vector ULstar, URstar;
   ULstar.SetSize(neq); URstar.SetSize(neq);
#endif

   // rho*_k
   const real_t rhoL_star = WL.rho * (SL - WL.un) / (SL - Sstar);
   const real_t rhoR_star = WR.rho * (SR - WR.un) / (SR - Sstar);
   ULstar(0) = rhoL_star;
   URstar(0) = rhoR_star;

   // momentum* = m + rho* (S* - un) n̂
   const real_t coeffL = rhoL_star * (Sstar - WL.un);
   const real_t coeffR = rhoR_star * (Sstar - WR.un);
   for (int d = 0; d < dim; ++d)
   {
      ULstar(1 + d) = state1(1 + d) + coeffL * nhat(d);
      URstar(1 + d) = state2(1 + d) + coeffR * nhat(d);
   }

   // p* from side k
   const real_t pL_star = WL.p + WL.rho * (SL - WL.un) * (Sstar - WL.un);
   const real_t pR_star = WR.p + WR.rho * (SR - WR.un) * (Sstar - WR.un);

   // E* using Toro's conservative update
   const real_t EL = state1(1 + dim);
   const real_t ER = state2(1 + dim);
   const real_t denomL = (SL - Sstar);
   const real_t denomR = (SR - Sstar);
   real_t ELstar = EL, ERstar = ER;
   if (std::abs(denomL) > 1e-14)
   {
      ELstar = (EL * (SL - WL.un) - WL.p * WL.un + pL_star * Sstar) / denomL;
   }
   if (std::abs(denomR) > 1e-14)
   {
      ERstar = (ER * (SR - WR.un) - WR.p * WR.un + pR_star * Sstar) / denomR;
   }
   ULstar(1 + dim) = ELstar;
   URstar(1 + dim) = ERstar;

   // HLLC flux in unit-normal direction
   if (Sstar >= 0.0)
   {
      // F* = F_L + S_L (U*_L - U_L)
      for (int i = 0; i < neq; ++i)
      {
         Fstar_hat(i) = FnL_hat(i) + SL * (ULstar(i) - state1(i));
      }
   }
   else
   {
      // F* = F_R + S_R (U*_R - U_R)
      for (int i = 0; i < neq; ++i)
      {
         Fstar_hat(i) = FnR_hat(i) + SR * (URstar(i) - state2(i));
      }
   }

   // Scale back to non-unit normal
   for (int i = 0; i < neq; ++i) { flux(i) = Fstar_hat(i) * nmag; }

   return SpeedScale(SL, SR);
}

void HLLCFlux::Grad(int side, const Vector &state1, const Vector &state2,
                    const Vector &nor, FaceElementTransformations &Tr,
                    DenseMatrix &grad) const
{
   const int neq = fluxFunction.num_equations;
   if (grad.Height() != neq || grad.Width() != neq) { grad.SetSize(neq); }

   // Build unit normal and wave-speed scale
#ifdef MFEM_THREAD_SAFE
   Vector nhat(nor.Size());
#else
   static thread_local Vector nhat; nhat.SetSize(nor.Size());
#endif
   real_t nmag = 0.0;
   UnitNormalAndMag(nor, nhat, nmag);

   Prim WL, WR;
   ConsToPrim(state1, nhat, WL);
   ConsToPrim(state2, nhat, WR);
   real_t SL=0.0, Sstar=0.0, SR=0.0;
   HLLC_WaveSpeeds(WL, WR, SL, Sstar, SR);

   const real_t scaled = SpeedScale(SL, SR) * nmag;

   if (side == 1)
   {
      fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, grad);
      for (int i = 0; i < neq; ++i) { grad(i,i) += 0.5 * scaled; }
   }
   else
   {
      fluxFunction.ComputeFluxJacobianDotN(state2, nor, Tr, grad);
      for (int i = 0; i < neq; ++i) { grad(i,i) -= 0.5 * scaled; }
   }
}

real_t HLLCFlux::Average(const Vector &state1, const Vector &state2,
                         const Vector &nor, FaceElementTransformations &Tr,
                         Vector &flux) const
{
   // Robust default: use full HLLC evaluation
   return Eval(state1, state2, nor, Tr, flux);
}

void HLLCFlux::AverageGrad(int side, const Vector &state1, const Vector &state2,
                           const Vector &nor, FaceElementTransformations &Tr,
                           DenseMatrix &grad) const
{
   // Same diagonal-stabilized Jacobian pattern
   Grad(side, state1, state2, nor, Tr, grad);
}

real_t FluxWithIGR::ComputeFlux(const Vector &U, ElementTransformation &Tr,
                                DenseMatrix &F) const
{
  // Base tensor flux + base max characteristic speed
  const real_t mcs = base_.ComputeFlux(U, Tr, F);
//   if (!sigma_) return mcs;

  const IntegrationPoint &ip = Tr.GetIntPoint(); // MFEM sets this before call
  double sigma = sigma_->GetValue(Tr, ip);

  AddTensorSigmaTerms(U, sigma, F);
  return mcs; // keep dissipation unchanged
}

real_t FluxWithIGR::ComputeFluxDotN(const Vector &U, const Vector &n,
                                    FaceElementTransformations &Tr,
                                    Vector &Fn) const
{
  // Base normal flux + base (normal) max characteristic speed
  const real_t mcs = base_.ComputeFluxDotN(U, n, Tr, Fn);
//   if (!sigma_) return mcs;

  // Face-average σ on interior faces, interior σ on boundary faces
  double sigma_face = 0.0;
  if (Tr.Elem2) // interior face
  {
    ElementTransformation &T1 = *Tr.Elem1;
    ElementTransformation &T2 = *Tr.Elem2;
    const IntegrationPoint &ip1 = Tr.GetElement1IntPoint();
    const IntegrationPoint &ip2 = Tr.GetElement2IntPoint();
    T1.SetIntPoint(&ip1);
    T2.SetIntPoint(&ip2);
    const double sL = sigma_->GetValue(T1, ip1);
    const double sR = sigma_->GetValue(T2, ip2);
   //  sigma_face = 0.5 * (sL + sR);
   sigma_face = std::max(sL, sR);
  }
  else // boundary face
  {
    ElementTransformation &T1 = *Tr.Elem1;
    const IntegrationPoint &ip1 = Tr.GetElement1IntPoint();
    T1.SetIntPoint(&ip1);
    sigma_face = sigma_->GetValue(T1, ip1);
  }

  AddDotNSigmaTerms(U, n, sigma_face, Fn);
  return mcs;
}


} // namespace mfem

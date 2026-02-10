#include "numerics/riemann/hllc.hpp"
#include "numerics/riemann/side_scope_utils.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{

HLLCFlux::HLLCFlux(const FluxFunction &fluxFunction, real_t gamma_ratio)
  : NumericalFlux(fluxFunction),
    side_explicit_flux_(dynamic_cast<const FaceSideExplicitFlux*>(&fluxFunction)),
    gamma(gamma_ratio)
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
  W.p = p;

  real_t un = 0.0;
  for (int d = 0; d < dim; ++d) { un += W.u(d) * nhat(d); }
  W.un = un;

  W.a = std::sqrt(std::max(real_t(0.0), gamma * p / rho));
}

void HLLCFlux::FluxDotNhat(const Vector &U, const Prim &W, const Vector &nhat, Vector &Fn_hat) const
{
  const int dim = W.u.Size();

  Fn_hat.SetSize(U.Size());
  Fn_hat(0) = W.rho * W.un;

  for (int d = 0; d < dim; ++d)
  {
    Fn_hat(1 + d) = W.rho * W.u(d) * W.un + W.p * nhat(d);
  }

  const real_t E = U(1 + dim);
  Fn_hat(1 + dim) = (E + W.p) * W.un;
}

void HLLCFlux::HLLC_WaveSpeeds(const Prim &WL, const Prim &WR,
                               real_t &SL, real_t &Sstar, real_t &SR) const
{
  const real_t sqrt_rho_L = std::sqrt(WL.rho);
  const real_t sqrt_rho_R = std::sqrt(WR.rho);
  const real_t denom = sqrt_rho_L + sqrt_rho_R;

  const real_t u_tilde = (sqrt_rho_L * WL.un + sqrt_rho_R * WR.un) / denom;
  const real_t u2_L = InnerProduct(WL.u, WL.u);
  const real_t u2_R = InnerProduct(WR.u, WR.u);
  const real_t H_L = (WL.p / (gamma - 1.0) + 0.5 * WL.rho * u2_L + WL.p) / WL.rho;
  const real_t H_R = (WR.p / (gamma - 1.0) + 0.5 * WR.rho * u2_R + WR.p) / WR.rho;
  const real_t H_tilde = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom;

  const real_t a_tilde = std::sqrt(std::max(real_t(0.0), (gamma - 1.0) * (H_tilde - 0.5 * u_tilde * u_tilde)));

  SL = std::min(WL.un - WL.a, u_tilde - a_tilde);
  SR = std::max(WR.un + WR.a, u_tilde + a_tilde);

  // Contact speed
  const real_t num = WR.p - WL.p + WL.rho * WL.un * (SL - WL.un) - WR.rho * WR.un * (SR - WR.un);
  const real_t den = WL.rho * (SL - WL.un) - WR.rho * (SR - WR.un);
  Sstar = num / den;
}

real_t HLLCFlux::Eval(const Vector &state1, const Vector &state2,
                      const Vector &nor, FaceElementTransformations &Tr,
                      Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
  Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
  const int neq = fluxFunction.num_equations;

  Vector nhat(nor.Size());
  real_t nmag = 0.0;
  UnitNormalAndMag(nor, nhat, nmag);

  Prim WL, WR;
  ConsToPrim(state1, nhat, WL);
  ConsToPrim(state2, nhat, WR);

  real_t SL = 0.0, Sstar = 0.0, SR = 0.0;
  HLLC_WaveSpeeds(WL, WR, SL, Sstar, SR);

  // Fluxes on left and right; side-explicit calls preserve IGR sigma-side
  // semantics while reusing the common helper.
  riemann_detail::ComputeFluxDotNForSide(
      side_explicit_flux_, fluxFunction, FaceSide::Left, state1, nhat, Tr, fluxN1);
  riemann_detail::ComputeFluxDotNForSide(
      side_explicit_flux_, fluxFunction, FaceSide::Right, state2, nhat, Tr, fluxN2);

  Vector flux_hat(neq);

  if (SL >= 0.0)
  {
    flux_hat = fluxN1;
  }
  else if (SL <= 0.0 && Sstar >= 0.0)
  {
    // Left star state
    const real_t rho_star = WL.rho * (SL - WL.un) / (SL - Sstar);

    flux_hat(0) = fluxN1(0) + SL * (rho_star - WL.rho);
    for (int d = 0; d < WL.u.Size(); ++d)
    {
      const real_t mom = rho_star * (WL.u(d) + (Sstar - WL.un) * nhat(d));
      flux_hat(1 + d) = fluxN1(1 + d) + SL * (mom - state1(1 + d));
    }
    const real_t E = state1(1 + WL.u.Size());
    const real_t E_star = (SL - WL.un) / (SL - Sstar)
                          * (E + (Sstar - WL.un) * (rho_star * Sstar + WL.p / (SL - WL.un)));
    flux_hat(1 + WL.u.Size()) = fluxN1(1 + WL.u.Size()) + SL * (E_star - E);
  }
  else if (Sstar <= 0.0 && SR >= 0.0)
  {
    // Right star state
    const real_t rho_star = WR.rho * (SR - WR.un) / (SR - Sstar);

    flux_hat(0) = fluxN2(0) + SR * (rho_star - WR.rho);
    for (int d = 0; d < WR.u.Size(); ++d)
    {
      const real_t mom = rho_star * (WR.u(d) + (Sstar - WR.un) * nhat(d));
      flux_hat(1 + d) = fluxN2(1 + d) + SR * (mom - state2(1 + d));
    }
    const real_t E = state2(1 + WR.u.Size());
    const real_t E_star = (SR - WR.un) / (SR - Sstar)
                          * (E + (Sstar - WR.un) * (rho_star * Sstar + WR.p / (SR - WR.un)));
    flux_hat(1 + WR.u.Size()) = fluxN2(1 + WR.u.Size()) + SR * (E_star - E);
  }
  else
  {
    flux_hat = fluxN2;
  }

  // Scale by |n|
  flux = flux_hat;
  flux *= nmag;

  return SpeedScale(SL, SR);
}

void HLLCFlux::Grad(int side, const Vector &state1, const Vector &state2,
                    const Vector &nor, FaceElementTransformations &Tr,
                    DenseMatrix &grad) const
{
#ifdef MFEM_THREAD_SAFE
  Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
  const int neq = fluxFunction.num_equations;

  Vector nhat(nor.Size());
  real_t nmag = 0.0;
  UnitNormalAndMag(nor, nhat, nmag);

  Prim WL, WR;
  ConsToPrim(state1, nhat, WL);
  ConsToPrim(state2, nhat, WR);
  real_t SL = 0.0, Sstar = 0.0, SR = 0.0;
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

} // namespace mfem

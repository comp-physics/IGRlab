#include "numerics/riemann/diffusive_rusanov.hpp"

#include "numerics/riemann/side_scope_utils.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{

DiffusiveRusanovFlux::DiffusiveRusanovFlux(const FluxFunction& fluxFunction,
                                           real_t beta,
                                           real_t delta,
                                           real_t cap,
                                           real_t sensor_k,
                                           real_t sensor_max)
  : NumericalFlux(fluxFunction),
    side_explicit_flux_(
      dynamic_cast<const FaceSideExplicitFlux*>(&fluxFunction)),
    beta_(beta),
    delta_(delta),
    cap_(cap),
    sensor_k_(sensor_k),
    sensor_max_(sensor_max)
{
#ifndef MFEM_THREAD_SAFE
  fluxN1.SetSize(fluxFunction.num_equations);
  fluxN2.SetSize(fluxFunction.num_equations);
#endif
}

real_t DiffusiveRusanovFlux::Eval(const Vector& state1,
                                  const Vector& state2,
                                  const Vector& nor,
                                  FaceElementTransformations& Tr,
                                  Vector& flux) const
{
#ifdef MFEM_THREAD_SAFE
  Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
  real_t speed1, speed2;
  // Shared side-scoping helper keeps behavior aligned with RusanovFluxWithSides.
  speed1 = riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                                  fluxFunction,
                                                  FaceSide::Left,
                                                  state1,
                                                  nor,
                                                  Tr,
                                                  fluxN1);
  speed2 = riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                                  fluxFunction,
                                                  FaceSide::Right,
                                                  state2,
                                                  nor,
                                                  Tr,
                                                  fluxN2);

  // Inflate lambda and then scale by |n|
  const real_t lam = EffectiveLambda(speed1, speed2, state1, state2);
  const real_t scaled = lam * nor.Norml2();

  const int m = fluxFunction.num_equations;
  for (int i = 0; i < m; i++)
  {
    flux(i) =
      0.5 * (scaled * (state1(i) - state2(i)) + (fluxN1(i) + fluxN2(i)));
  }

  // Return the baseline max eigenvalue (as in RusanovFlux) for CFL accounting
  return std::max(speed1, speed2);
}

void DiffusiveRusanovFlux::Grad(int side,
                                const Vector& state1,
                                const Vector& state2,
                                const Vector& nor,
                                FaceElementTransformations& Tr,
                                DenseMatrix& grad) const
{
#ifdef MFEM_THREAD_SAFE
  Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
  real_t speed1, speed2;
  speed1 = riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                                  fluxFunction,
                                                  FaceSide::Left,
                                                  state1,
                                                  nor,
                                                  Tr,
                                                  fluxN1);
  speed2 = riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                                  fluxFunction,
                                                  FaceSide::Right,
                                                  state2,
                                                  nor,
                                                  Tr,
                                                  fluxN2);

  const real_t lam = EffectiveLambda(speed1, speed2, state1, state2);
  const real_t scaled = lam * nor.Norml2();

  const int m = fluxFunction.num_equations;
  if (side == 1)
  {
    fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, grad);
    for (int i = 0; i < m; i++)
    {
      grad(i, i) += 0.5 * scaled;
    }
  }
  else
  {
    fluxFunction.ComputeFluxJacobianDotN(state2, nor, Tr, grad);
    for (int i = 0; i < m; i++)
    {
      grad(i, i) -= 0.5 * scaled;
    }
  }
}

real_t DiffusiveRusanovFlux::Average(const Vector& state1,
                                     const Vector& state2,
                                     const Vector& nor,
                                     FaceElementTransformations& Tr,
                                     Vector& flux) const
{
#ifdef MFEM_THREAD_SAFE
  Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif
  real_t speed1, speed2;
  speed1 = riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                                  fluxFunction,
                                                  FaceSide::Left,
                                                  state1,
                                                  nor,
                                                  Tr,
                                                  fluxN1);
  speed2 = riemann_detail::ComputeAvgFluxDotNForSide(side_explicit_flux_,
                                                     fluxFunction,
                                                     FaceSide::Right,
                                                     state1,
                                                     state2,
                                                     nor,
                                                     Tr,
                                                     fluxN2);

  const real_t lam = EffectiveLambda(speed1, speed2, state1, state2);
  const real_t scaled = lam * nor.Norml2() * 0.5; // MFEM halves in Average

  const int m = fluxFunction.num_equations;
  for (int i = 0; i < m; i++)
  {
    flux(i) =
      0.5 * (scaled * (state1(i) - state2(i)) + (fluxN1(i) + fluxN2(i)));
  }

  // Return baseline max eigenvalue
  return std::max(speed1, speed2);
}

void DiffusiveRusanovFlux::AverageGrad(int side,
                                       const Vector& state1,
                                       const Vector& state2,
                                       const Vector& nor,
                                       FaceElementTransformations& Tr,
                                       DenseMatrix& grad) const
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
    real_t speed1, speed2;
    speed1 = riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                                    fluxFunction,
                                                    FaceSide::Left,
                                                    state1,
                                                    nor,
                                                    Tr,
                                                    fluxN1);
    speed2 = riemann_detail::ComputeAvgFluxDotNForSide(side_explicit_flux_,
                                                       fluxFunction,
                                                       FaceSide::Right,
                                                       state1,
                                                       state2,
                                                       nor,
                                                       Tr,
                                                       fluxN2);
    fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, JDotN);

    const real_t lam = EffectiveLambda(speed1, speed2, state1, state2);
    const real_t scaled = lam * nor.Norml2() * 0.5;

    grad = 0.0;

    const int m = fluxFunction.num_equations;
    for (int i = 0; i < m; i++)
    {
      // Only diagonal terms of J considered; skip if states are equal
      if (equal_check(state1(i), state2(i)))
      {
        continue;
      }
      grad(i, i) = 0.5 * ((fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i)) -
                          JDotN(i, i) + scaled);
    }
  }
  else
  {
    real_t speed1, speed2;
    speed1 = riemann_detail::ComputeAvgFluxDotNForSide(side_explicit_flux_,
                                                       fluxFunction,
                                                       FaceSide::Left,
                                                       state1,
                                                       state2,
                                                       nor,
                                                       Tr,
                                                       fluxN1);
    speed2 = riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                                    fluxFunction,
                                                    FaceSide::Right,
                                                    state2,
                                                    nor,
                                                    Tr,
                                                    fluxN2);

    const real_t lam = EffectiveLambda(speed1, speed2, state1, state2);
    const real_t scaled = lam * nor.Norml2() * 0.5;

    grad = 0.0;

    const int m = fluxFunction.num_equations;
    for (int i = 0; i < m; i++)
    {
      if (equal_check(state1(i), state2(i)))
      {
        continue;
      }
      grad(i, i) =
        0.5 * ((fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i)) - scaled);
    }
  }
}

real_t DiffusiveRusanovFlux::EffectiveLambda(real_t speed1,
                                             real_t speed2,
                                             const Vector& state1,
                                             const Vector& state2) const
{
  real_t lam = std::max(speed1, speed2); // baseline lambda from FluxFunction
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
  if (lam > cap_)
  {
    lam = cap_;
  }
  return lam;
}

} // namespace mfem

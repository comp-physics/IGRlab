#include "numerics/riemann/rusanov.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{

RusanovFluxWithSides::RusanovFluxWithSides(const FluxFunction &fluxFunction)
  : NumericalFlux(fluxFunction)
{
#ifndef MFEM_THREAD_SAFE
  fluxN1.SetSize(fluxFunction.num_equations);
  fluxN2.SetSize(fluxFunction.num_equations);
#endif
}

real_t RusanovFluxWithSides::Eval(const Vector &state1, const Vector &state2,
                                  const Vector &nor, FaceElementTransformations &Tr,
                                  Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
  Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif

  real_t speed1, speed2;

  // Minimal addition: if the flux supports side info, set it around each call.
  if (auto *side_aware =
          dynamic_cast<FaceSideAwareFlux*>(const_cast<FluxFunction*>(&fluxFunction)))
  {
    side_aware->SetFaceSide(FaceSide::Left);
    speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);

    side_aware->SetFaceSide(FaceSide::Right);
    speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

    // Optional: restore to a default (not required)
    side_aware->SetFaceSide(FaceSide::Left);
  }
  else
  {
    speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
    speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);
  }

  // NOTE: nor in general is not a unit normal
  const real_t maxE = std::max(speed1, speed2);
  // here, nor.Norml2() is multiplied to match the scale with fluxN
  const real_t scaledMaxE = maxE * nor.Norml2();

  for (int i = 0; i < fluxFunction.num_equations; i++)
  {
    flux(i) = 0.5*(scaledMaxE*(state1(i) - state2(i)) + (fluxN1(i) + fluxN2(i)));
  }
  return maxE;
}

void RusanovFluxWithSides::Grad(int side, const Vector &state1, const Vector &state2,
                                const Vector &nor, FaceElementTransformations &Tr,
                                DenseMatrix &grad) const
{
#ifdef MFEM_THREAD_SAFE
  Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif

  real_t speed1, speed2;

  if (auto *side_aware =
          dynamic_cast<FaceSideAwareFlux*>(const_cast<FluxFunction*>(&fluxFunction)))
  {
    side_aware->SetFaceSide(FaceSide::Left);
    speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);

    side_aware->SetFaceSide(FaceSide::Right);
    speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

    side_aware->SetFaceSide(FaceSide::Left);
  }
  else
  {
    speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
    speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);
  }

  const real_t maxE = std::max(speed1, speed2);
  const real_t scaledMaxE = maxE * nor.Norml2();

  const int m = fluxFunction.num_equations;
  if (side == 1)
  {
    fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, grad);
    for (int i = 0; i < m; i++) { grad(i,i) += 0.5 * scaledMaxE; }
  }
  else
  {
    fluxFunction.ComputeFluxJacobianDotN(state2, nor, Tr, grad);
    for (int i = 0; i < m; i++) { grad(i,i) -= 0.5 * scaledMaxE; }
  }
}

real_t RusanovFluxWithSides::Average(const Vector &state1, const Vector &state2,
                                     const Vector &nor, FaceElementTransformations &Tr,
                                     Vector &flux) const
{
#ifdef MFEM_THREAD_SAFE
  Vector fluxN1(fluxFunction.num_equations), fluxN2(fluxFunction.num_equations);
#endif

  real_t speed1, speed2;

  if (auto *side_aware =
          dynamic_cast<FaceSideAwareFlux*>(const_cast<FluxFunction*>(&fluxFunction)))
  {
    side_aware->SetFaceSide(FaceSide::Left);
    speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);

    side_aware->SetFaceSide(FaceSide::Right);
    speed2 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN2);

    side_aware->SetFaceSide(FaceSide::Left);
  }
  else
  {
    speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
    speed2 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN2);
  }

  const real_t maxE = std::max(speed1, speed2);
  const real_t scaledMaxE = maxE * nor.Norml2() * 0.5;

  for (int i = 0; i < fluxFunction.num_equations; i++)
  {
    flux(i) = 0.5*(scaledMaxE*(state1(i) - state2(i)) + (fluxN1(i) + fluxN2(i)));
  }
  return maxE;
}

void RusanovFluxWithSides::AverageGrad(int side, const Vector &state1,
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
    real_t speed1 = 0.0, speed2 = 0.0;

    if (auto *side_aware =
            dynamic_cast<FaceSideAwareFlux*>(const_cast<FluxFunction*>(&fluxFunction)))
    {
      side_aware->SetFaceSide(FaceSide::Left);
      speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);

      side_aware->SetFaceSide(FaceSide::Right);
      speed2 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN2);

      side_aware->SetFaceSide(FaceSide::Left);
    }
    else
    {
      speed1 = fluxFunction.ComputeFluxDotN(state1, nor, Tr, fluxN1);
      speed2 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN2);
    }

    fluxFunction.ComputeFluxJacobianDotN(state1, nor, Tr, JDotN);

    const real_t maxE = std::max(speed1, speed2);
    const real_t scaledMaxE = maxE * nor.Norml2() * 0.5;

    grad = 0.0;

    const int m = fluxFunction.num_equations;
    for (int i = 0; i < m; i++)
    {
      if (equal_check(state1(i), state2(i))) { continue; }
      grad(i,i) = 0.5 * ((fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i))
                         - JDotN(i,i) + scaledMaxE);
    }
  }
  else
  {
    real_t speed1 = 0.0, speed2 = 0.0;

    if (auto *side_aware =
            dynamic_cast<FaceSideAwareFlux*>(const_cast<FluxFunction*>(&fluxFunction)))
    {
      side_aware->SetFaceSide(FaceSide::Left);
      speed1 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN1);

      side_aware->SetFaceSide(FaceSide::Right);
      speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);

      side_aware->SetFaceSide(FaceSide::Left);
    }
    else
    {
      speed1 = fluxFunction.ComputeAvgFluxDotN(state1, state2, nor, Tr, fluxN1);
      speed2 = fluxFunction.ComputeFluxDotN(state2, nor, Tr, fluxN2);
    }

    const real_t maxE = std::max(speed1, speed2);
    const real_t scaledMaxE = maxE * nor.Norml2() * 0.5;

    grad = 0.0;

    const int m = fluxFunction.num_equations;
    for (int i = 0; i < m; i++)
    {
      if (equal_check(state1(i), state2(i))) { continue; }
      grad(i,i) = 0.5 * ((fluxN2(i) - fluxN1(i)) / (state2(i) - state1(i)) - scaledMaxE);
    }
  }
}

} // namespace mfem

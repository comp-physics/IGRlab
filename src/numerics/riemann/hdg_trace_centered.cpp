#include "numerics/riemann/hdg_trace_centered.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "numerics/riemann/side_scope_utils.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "physics/igr/igr_flux.hpp"

namespace mfem
{

namespace
{

real_t InvariantDomainSpeedAtTrace(const FluxWithIGR& igr_flux,
                                   const Vector& state,
                                   const Vector& normal,
                                   real_t sigma,
                                   real_t base_speed)
{
  const int dim = state.Size() - 2;
  const real_t rho_floor = igr_flux.GetVelocityRhoFloor();
  const real_t density = state(0);
  const real_t safe_density = std::max(density, rho_floor);
  real_t momentum_squared = 0.0;
  real_t momentum_dot_normal = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t momentum = state(1 + d);
    momentum_squared += momentum * momentum;
    momentum_dot_normal += momentum * normal(d);
  }
  const real_t gamma = igr_flux.GetGamma();
  const real_t pressure =
    (gamma - 1.0) *
    (state(1 + dim) - 0.5 * momentum_squared / safe_density);
  const real_t normal_norm = normal.Norml2();
  const real_t transport_speed =
    normal_norm > 0.0
      ? std::abs(momentum_dot_normal) / (safe_density * normal_norm)
      : 0.0;
  return IGRFluxInvariantDomainSpeed(base_speed,
                                     density,
                                     pressure,
                                     transport_speed,
                                     sigma,
                                     gamma,
                                     rho_floor,
                                     igr_flux.GetInvariantDomainPressureFloor());
}

} // namespace

HdgTraceCenteredFlux::HdgTraceCenteredFlux(
  const FluxFunction& flux_function)
  : NumericalFlux(flux_function),
    flux_fn_(flux_function),
    side_explicit_flux_(
      dynamic_cast<const FaceSideExplicitFlux*>(&flux_function)),
    igr_flux_(dynamic_cast<const FluxWithIGR*>(&flux_function))
{
#ifndef MFEM_THREAD_SAFE
  state_hat_.SetSize(flux_function.num_equations);
  flux_n_left_.SetSize(flux_function.num_equations);
  flux_n_right_.SetSize(flux_function.num_equations);
  flux_n_hat_.SetSize(flux_function.num_equations);
#endif
}

real_t HdgTraceCenteredFlux::TraceFluxAndSpeeds_(
  const Vector& state_hat,
  real_t sigma_hat,
  const Vector& nor,
  FaceElementTransformations& Tr,
  Vector& flux_n_hat,
  real_t& dissipation_speed) const
{
  if (!igr_flux_)
  {
    const real_t speed =
      flux_fn_.ComputeFluxDotN(state_hat, nor, Tr, flux_n_hat);
    dissipation_speed = speed;
    return speed;
  }

  const real_t base_speed = igr_flux_->GetBaseFlux().ComputeFluxDotN(
    state_hat, nor, Tr, flux_n_hat);
  if (!igr_flux_->GetSigmaField())
  {
    dissipation_speed = base_speed;
    return base_speed;
  }

  ::dg_igr::frozen_scalar_igr_algebra::AddNormalSigmaFlux(
    flux_fn_.dim,
    state_hat.GetData(),
    1,
    nor.GetData(),
    1,
    sigma_hat,
    igr_flux_->GetVelocityRhoFloor(),
    flux_n_hat.GetData(),
    1);

  if (igr_flux_->GetInvariantDomainFlux())
  {
    const real_t speed = InvariantDomainSpeedAtTrace(
      *igr_flux_, state_hat, nor, sigma_hat, base_speed);
    dissipation_speed = speed;
    return speed;
  }

  const real_t cfl_speed =
    igr_flux_->GetSigmaAwareCFL()
      ? IGRFluxCharacteristicSpeed(base_speed,
                                   state_hat(0),
                                   sigma_hat,
                                   igr_flux_->GetVelocityRhoFloor())
      : base_speed;
  dissipation_speed = igr_flux_->SelectRusanovDissipationSpeed(
    cfl_speed, state_hat(0), sigma_hat);
  return cfl_speed;
}

real_t HdgTraceCenteredFlux::Eval(const Vector& state1,
                                  const Vector& state2,
                                  const Vector& nor,
                                  FaceElementTransformations& Tr,
                                  Vector& flux) const
{
  MFEM_VERIFY(dynamic_cast<const EulerFlux*>(&flux_fn_) != nullptr ||
                (igr_flux_ && igr_flux_->HasEulerBase()),
              "The HDG-derived trace-centered flux currently supports only "
              "Euler and Euler+IGR physical fluxes.");
  MFEM_VERIFY(state1.Size() == flux_fn_.num_equations &&
                state2.Size() == flux_fn_.num_equations,
              "Trace-centered flux received an incompatible state size.");
#ifdef MFEM_THREAD_SAFE
  Vector state_hat(flux_fn_.num_equations);
  Vector flux_n_left(flux_fn_.num_equations);
  Vector flux_n_right(flux_fn_.num_equations);
  Vector flux_n_hat(flux_fn_.num_equations);
#else
  Vector& state_hat = state_hat_;
  Vector& flux_n_left = flux_n_left_;
  Vector& flux_n_right = flux_n_right_;
  Vector& flux_n_hat = flux_n_hat_;
#endif

  real_t cfl_left = riemann_detail::ComputeFluxDotNForSide(
    side_explicit_flux_, flux_fn_, FaceSide::Left, state1, nor, Tr, flux_n_left);
  real_t cfl_right = riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                                            flux_fn_,
                                                            FaceSide::Right,
                                                            state2,
                                                            nor,
                                                            Tr,
                                                            flux_n_right);
  real_t dissipation_left = cfl_left;
  real_t dissipation_right = cfl_right;
  real_t sigma_left = 0.0;
  real_t sigma_right = 0.0;
  if (igr_flux_)
  {
    sigma_left = igr_flux_->GetSigmaAtSide(Tr, FaceSide::Left);
    sigma_right = igr_flux_->GetSigmaAtSide(Tr, FaceSide::Right);
    dissipation_left = igr_flux_->SelectRusanovDissipationSpeed(
      cfl_left, state1(0), sigma_left);
    dissipation_right = igr_flux_->SelectRusanovDissipationSpeed(
      cfl_right, state2(0), sigma_right);
  }

  for (int equation = 0; equation < flux_fn_.num_equations; ++equation)
  {
    state_hat(equation) = 0.5 * (state1(equation) + state2(equation));
  }
  const real_t sigma_hat = 0.5 * (sigma_left + sigma_right);
  real_t dissipation_hat = 0.0;
  const real_t cfl_hat = TraceFluxAndSpeeds_(
    state_hat, sigma_hat, nor, Tr, flux_n_hat, dissipation_hat);

  const real_t tau =
    std::max(dissipation_hat,
             std::max(dissipation_left, dissipation_right));
  const real_t scaled_tau = tau * nor.Norml2();
  flux.SetSize(flux_fn_.num_equations);
  for (int equation = 0; equation < flux_fn_.num_equations; ++equation)
  {
    flux(equation) = flux_n_hat(equation) +
                     0.5 * scaled_tau *
                       (state1(equation) - state2(equation));
  }
  return std::max(cfl_hat, std::max(cfl_left, cfl_right));
}

void HdgTraceCenteredFlux::Grad(int side,
                                const Vector& state1,
                                const Vector& state2,
                                const Vector& nor,
                                FaceElementTransformations& Tr,
                                DenseMatrix& grad) const
{
  MFEM_VERIFY(side == 1 || side == 2,
              "Numerical-flux side must be one or two.");
  const int equations = flux_fn_.num_equations;
  grad.SetSize(equations);
  Vector perturbed1(state1);
  Vector perturbed2(state2);
  Vector plus(equations);
  Vector minus(equations);
  const real_t epsilon = std::sqrt(std::numeric_limits<real_t>::epsilon());
  for (int state_component = 0; state_component < equations; ++state_component)
  {
    Vector& selected = side == 1 ? perturbed1 : perturbed2;
    const real_t original = selected(state_component);
    const real_t delta = epsilon * std::max<real_t>(1.0, std::abs(original));
    selected(state_component) = original + delta;
    Eval(perturbed1, perturbed2, nor, Tr, plus);
    selected(state_component) = original - delta;
    Eval(perturbed1, perturbed2, nor, Tr, minus);
    selected(state_component) = original;
    for (int flux_equation = 0; flux_equation < equations; ++flux_equation)
    {
      grad(flux_equation, state_component) =
        (plus(flux_equation) - minus(flux_equation)) / (2.0 * delta);
    }
  }
}

real_t HdgTraceCenteredFlux::Average(const Vector& state1,
                                     const Vector& state2,
                                     const Vector& nor,
                                     FaceElementTransformations& Tr,
                                     Vector& flux) const
{
  return Eval(state1, state2, nor, Tr, flux);
}

void HdgTraceCenteredFlux::AverageGrad(int side,
                                       const Vector& state1,
                                       const Vector& state2,
                                       const Vector& nor,
                                       FaceElementTransformations& Tr,
                                       DenseMatrix& grad) const
{
  Grad(side, state1, state2, nor, Tr, grad);
}

} // namespace mfem

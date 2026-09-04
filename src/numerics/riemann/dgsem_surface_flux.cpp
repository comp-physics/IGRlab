#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "numerics/riemann/side_scope_utils.hpp"

#include <algorithm>

namespace mfem
{

namespace
{

inline const char* SurfaceFluxTypeName(DGSEMSurfaceFluxType type)
{
  switch (type)
  {
  case DGSEMSurfaceFluxType::Rusanov:
    return "rusanov";
  case DGSEMSurfaceFluxType::DiffusiveRusanov:
    return "diffusive-rusanov";
  case DGSEMSurfaceFluxType::EntropyStableWinters:
    return "es-winters";
  case DGSEMSurfaceFluxType::EntropyStableLFEntropyVars:
    return "es-lf-entropy";
  }
  MFEM_ABORT(
    "Unhandled DGSEMSurfaceFluxType enum value: " << static_cast<int>(type));
  return "invalid";
}

} // namespace

DGSEMSurfaceFlux::DGSEMSurfaceFlux(const FluxFunction& fluxFunction)
  : NumericalFlux(fluxFunction),
    flux_fn_(fluxFunction),
    side_explicit_flux_(
      dynamic_cast<const FaceSideExplicitFlux*>(&fluxFunction)),
    rusanov_(fluxFunction),
    diffusive_rusanov_(fluxFunction, /*beta=*/2.0)
{
}

real_t DGSEMSurfaceFlux::Eval(const Vector& state1,
                              const Vector& state2,
                              const Vector& nor,
                              FaceElementTransformations& Tr,
                              Vector& flux) const
{
  if (type_ == DGSEMSurfaceFluxType::Rusanov)
  {
    return rusanov_.Eval(state1, state2, nor, Tr, flux);
  }
  if (type_ == DGSEMSurfaceFluxType::DiffusiveRusanov)
  {
    return diffusive_rusanov_.Eval(state1, state2, nor, Tr, flux);
  }

  const FluxWithIGR* flux_igr = dynamic_cast<const FluxWithIGR*>(&flux_fn_);
  double sigma_l = 0.0;
  double sigma_r = 0.0;
  if (flux_igr)
  {
    sigma_l = flux_igr->GetSigmaAtSide(Tr, FaceSide::Left);
    sigma_r = flux_igr->GetSigmaAtSide(Tr, FaceSide::Right);
  }

  Vector flux_tmp(state1.Size());
  const real_t speed1 =
    riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                           flux_fn_,
                                           FaceSide::Left,
                                           state1,
                                           nor,
                                           Tr,
                                           flux_tmp);
  const real_t speed2 =
    riemann_detail::ComputeFluxDotNForSide(side_explicit_flux_,
                                           flux_fn_,
                                           FaceSide::Right,
                                           state2,
                                           nor,
                                           Tr,
                                           flux_tmp);

  const real_t max_speed = std::max(speed1, speed2);

  Vector flux_ec(state1.Size());
  dgsem::EntropyConservativeFluxDotN(state1,
                                     state2,
                                     nor,
                                     gamma_,
                                     entropy_type_,
                                     flux_ec,
                                     sigma_l,
                                     sigma_r);
  flux = flux_ec;

  if (type_ == DGSEMSurfaceFluxType::EntropyStableWinters)
  {
    Vector diss(state1.Size());
    dgsem::WintersDissipation(state1,
                              state2,
                              nor,
                              gamma_,
                              diss,
                              sigma_l,
                              sigma_r);
    flux += diss;
  }
  else
  {
    const double scaled = max_speed * nor.Norml2();
    Vector diss(state1.Size());
    dgsem::EntropyVariableLFDissipation(state1,
                                        state2,
                                        gamma_,
                                        scaled,
                                        diss,
                                        sigma_l,
                                        sigma_r,
                                        entropy_jacobian_mode_);
    flux += diss;
  }

  return max_speed;
}

void DGSEMSurfaceFlux::Grad(int side,
                            const Vector& state1,
                            const Vector& state2,
                            const Vector& nor,
                            FaceElementTransformations& Tr,
                            DenseMatrix& grad) const
{
  if (type_ == DGSEMSurfaceFluxType::Rusanov)
  {
    rusanov_.Grad(side, state1, state2, nor, Tr, grad);
    return;
  }
  if (type_ == DGSEMSurfaceFluxType::DiffusiveRusanov)
  {
    diffusive_rusanov_.Grad(side, state1, state2, nor, Tr, grad);
    return;
  }
  MFEM_VERIFY(
    false,
    "DGSEMSurfaceFlux::Grad is unsupported for entropy-stable surface flux '"
      << SurfaceFluxTypeName(type_)
      << "'. Use rusanov/diffusive-rusanov for Jacobian paths.");
}

real_t DGSEMSurfaceFlux::Average(const Vector& state1,
                                 const Vector& state2,
                                 const Vector& nor,
                                 FaceElementTransformations& Tr,
                                 Vector& flux) const
{
  if (type_ == DGSEMSurfaceFluxType::Rusanov)
  {
    return rusanov_.Average(state1, state2, nor, Tr, flux);
  }
  if (type_ == DGSEMSurfaceFluxType::DiffusiveRusanov)
  {
    return diffusive_rusanov_.Average(state1, state2, nor, Tr, flux);
  }
  return Eval(state1, state2, nor, Tr, flux);
}

void DGSEMSurfaceFlux::AverageGrad(int side,
                                   const Vector& state1,
                                   const Vector& state2,
                                   const Vector& nor,
                                   FaceElementTransformations& Tr,
                                   DenseMatrix& grad) const
{
  if (type_ == DGSEMSurfaceFluxType::Rusanov)
  {
    rusanov_.AverageGrad(side, state1, state2, nor, Tr, grad);
    return;
  }
  if (type_ == DGSEMSurfaceFluxType::DiffusiveRusanov)
  {
    diffusive_rusanov_.AverageGrad(side, state1, state2, nor, Tr, grad);
    return;
  }
  MFEM_VERIFY(false,
              "DGSEMSurfaceFlux::AverageGrad is unsupported for entropy-stable "
              "surface flux '"
                << SurfaceFluxTypeName(type_)
                << "'. Use rusanov/diffusive-rusanov for Jacobian paths.");
}

} // namespace mfem

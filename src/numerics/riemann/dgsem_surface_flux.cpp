#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "numerics/riemann/side_scope_utils.hpp"

#include <cmath>
#include <iostream>
#include <limits>

namespace mfem
{

namespace
{

inline const char* SurfaceFluxTypeName(DGSEMSurfaceFluxType type)
{
  switch (type)
  {
    case DGSEMSurfaceFluxType::Rusanov: return "rusanov";
    case DGSEMSurfaceFluxType::DiffusiveRusanov: return "diffusive-rusanov";
    case DGSEMSurfaceFluxType::EntropyStableWinters: return "es-winters";
    case DGSEMSurfaceFluxType::EntropyStableLFEntropyVars: return "es-lf-entropy";
  }
  return "unknown";
}

inline double FiniteDiffStep(double value)
{
  const double rel = std::sqrt(std::numeric_limits<double>::epsilon());
  return rel * std::max(1.0, std::abs(value));
}

template <typename FluxEval>
void ComputeFaceFluxJacobianFD(int side,
                               const Vector &state1,
                               const Vector &state2,
                               DenseMatrix &grad,
                               FluxEval &&eval_flux)
{
  MFEM_VERIFY(state1.Size() == state2.Size(),
              "DGSEMSurfaceFlux Jacobian fallback expects same-sized states.");
  MFEM_VERIFY(side == 1 || side == 2,
              "DGSEMSurfaceFlux Jacobian fallback expects side=1 or side=2.");

  const int n = state1.Size();
  grad.SetSize(n, n);
  grad = 0.0;

  Vector base_flux(n), pert_flux(n);
  eval_flux(state1, state2, base_flux);

  Vector pert_state1(state1);
  Vector pert_state2(state2);
  Vector &target = (side == 1) ? pert_state1 : pert_state2;

  for (int j = 0; j < n; ++j)
  {
    const double h = FiniteDiffStep(target(j));
    target(j) += h;
    eval_flux(pert_state1, pert_state2, pert_flux);
    target(j) -= h;
    for (int i = 0; i < n; ++i)
    {
      grad(i, j) = (pert_flux(i) - base_flux(i)) / h;
    }
  }
}

} // namespace

DGSEMSurfaceFlux::DGSEMSurfaceFlux(const FluxFunction &fluxFunction)
  : NumericalFlux(fluxFunction),
    flux_fn_(fluxFunction),
    rusanov_(fluxFunction),
    diffusive_rusanov_(fluxFunction, /*beta=*/2.0)
{
}

real_t DGSEMSurfaceFlux::Eval(const Vector &state1, const Vector &state2,
                              const Vector &nor, FaceElementTransformations &Tr,
                              Vector &flux) const
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
  const real_t speed1 = riemann_detail::ComputeFluxDotNForSide(
      flux_fn_, FaceSide::Left, state1, nor, Tr, flux_tmp);
  const real_t speed2 = riemann_detail::ComputeFluxDotNForSide(
      flux_fn_, FaceSide::Right, state2, nor, Tr, flux_tmp);

  const real_t max_speed = std::max(speed1, speed2);

  Vector flux_ec(state1.Size());
  dgsem::EntropyConservativeFluxDotN(state1, state2, nor, gamma_,
                                     entropy_type_, flux_ec,
                                     sigma_l, sigma_r);
  flux = flux_ec;

  if (type_ == DGSEMSurfaceFluxType::EntropyStableWinters)
  {
    Vector diss(state1.Size());
    dgsem::WintersDissipation(state1, state2, nor, gamma_, diss,
                              sigma_l, sigma_r);
    flux += diss;
  }
  else
  {
    const double scaled = max_speed * nor.Norml2();
    Vector diss(state1.Size());
    dgsem::EntropyVariableLFDissipation(state1, state2, gamma_, scaled, diss,
                                        sigma_l, sigma_r,
                                        entropy_jacobian_mode_);
    flux += diss;
  }

  return max_speed;
}

void DGSEMSurfaceFlux::Grad(int side, const Vector &state1, const Vector &state2,
                            const Vector &nor, FaceElementTransformations &Tr,
                            DenseMatrix &grad) const
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

  static bool warned = false;
  if (!warned)
  {
    warned = true;
    if (mfem::Mpi::Root())
    {
      std::cout << "[dgsem-surface-flux] event=jacobian-fallback method=Grad"
                << " flux_type=" << SurfaceFluxTypeName(type_)
                << " detail=\"finite-difference Jacobian used for entropy-stable surface flux\""
                << "\n";
    }
  }

  auto eval_flux = [&](const Vector &u_l, const Vector &u_r, Vector &flux)
  {
    Eval(u_l, u_r, nor, Tr, flux);
  };
  ComputeFaceFluxJacobianFD(side, state1, state2, grad, eval_flux);
}

real_t DGSEMSurfaceFlux::Average(const Vector &state1, const Vector &state2,
                                 const Vector &nor, FaceElementTransformations &Tr,
                                 Vector &flux) const
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

void DGSEMSurfaceFlux::AverageGrad(int side, const Vector &state1,
                                   const Vector &state2, const Vector &nor,
                                   FaceElementTransformations &Tr,
                                   DenseMatrix &grad) const
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

  static bool warned = false;
  if (!warned)
  {
    warned = true;
    if (mfem::Mpi::Root())
    {
      std::cout << "[dgsem-surface-flux] event=jacobian-fallback method=AverageGrad"
                << " flux_type=" << SurfaceFluxTypeName(type_)
                << " detail=\"finite-difference averaged Jacobian used for entropy-stable surface flux\""
                << "\n";
    }
  }

  auto eval_flux = [&](const Vector &u_l, const Vector &u_r, Vector &flux)
  {
    Average(u_l, u_r, nor, Tr, flux);
  };
  ComputeFaceFluxJacobianFD(side, state1, state2, grad, eval_flux);
}

} // namespace mfem

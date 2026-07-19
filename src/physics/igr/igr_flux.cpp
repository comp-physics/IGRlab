#include "physics/igr/igr_flux.hpp"

#include <algorithm>

namespace mfem
{

real_t FluxWithIGR::ComputeFlux(const Vector& U,
                                ElementTransformation& Tr,
                                DenseMatrix& F) const
{
  // Base tensor flux + base max characteristic speed
  const real_t mcs = base_.ComputeFlux(U, Tr, F);
  if (!sigma_) return mcs;

  const IntegrationPoint& ip = Tr.GetIntPoint(); // MFEM sets this before call
  const double sigma = sigma_->GetValue(Tr, ip);

  AddTensorSigmaTerms(U, sigma, F);
  return sigma_aware_cfl_
           ? IGRFluxCharacteristicSpeed(mcs, U(0), sigma, rho_vel_floor_)
           : mcs;
}

real_t FluxWithIGR::ComputeFluxDotN(const Vector& U,
                                    const Vector& n,
                                    FaceElementTransformations& Tr,
                                    Vector& Fn) const
{
  return ComputeFluxDotNAtSide(U, n, Tr, FaceSide::Left, Fn);
}

real_t FluxWithIGR::ComputeFluxDotNAtSide(const Vector& U,
                                          const Vector& n,
                                          FaceElementTransformations& Tr,
                                          FaceSide side,
                                          Vector& Fn) const
{
  const real_t mcs = base_.ComputeFluxDotN(U, n, Tr, Fn);
  if (!sigma_)
  {
    return mcs;
  }

  const double sigma_face = GetSigmaAtSide(Tr, side);
  AddDotNSigmaTerms(U, n, sigma_face, Fn);
  return sigma_aware_cfl_
           ? IGRFluxCharacteristicSpeed(
               mcs, U(0), sigma_face, rho_vel_floor_)
           : mcs;
}

real_t FluxWithIGR::ComputeAvgFluxDotNAtSide(const Vector& U1,
                                             const Vector& U2,
                                             const Vector& n,
                                             FaceElementTransformations& Tr,
                                             FaceSide side,
                                             Vector& Fn) const
{
  const real_t mcs = base_.ComputeAvgFluxDotN(U1, U2, n, Tr, Fn);
  if (!sigma_)
  {
    return mcs;
  }

  const double sigma_face = GetSigmaAtSide(Tr, side);
  MFEM_VERIFY(U1(0) > 0.0 && U2(0) > 0.0,
              "FluxWithIGR::ComputeAvgFluxDotNAtSide requires positive "
              "density endpoints for exact averaged sigma correction.");
  ::dg_igr::frozen_scalar_igr_algebra::AddTwoStateAverageNormalSigmaFlux(
    dim,
    U1.GetData(),
    1,
    U2.GetData(),
    1,
    n.GetData(),
    1,
    sigma_face,
    Fn.GetData(),
    1);
  return sigma_aware_cfl_
           ? IGRFluxCharacteristicSpeed(
               mcs, std::min(U1(0), U2(0)), sigma_face, rho_vel_floor_)
           : mcs;
}

double FluxWithIGR::GetSigmaAtPoint(ElementTransformation& Tr) const
{
  if (!sigma_)
  {
    return 0.0;
  }
  const IntegrationPoint& ip = Tr.GetIntPoint();
  return sigma_->GetValue(Tr, ip);
}

double FluxWithIGR::GetSigmaAtSide(FaceElementTransformations& Tr,
                                   FaceSide side) const
{
  if (!sigma_)
  {
    return 0.0;
  }

  if (Tr.Elem2) // interior face
  {
    if (side == FaceSide::Left)
    {
      ElementTransformation& T1 = *Tr.Elem1;
      const IntegrationPoint& ip1 = Tr.GetElement1IntPoint();
      T1.SetIntPoint(&ip1);
      return sigma_->GetValue(T1, ip1);
    }
    ElementTransformation& T2 = *Tr.Elem2;
    const IntegrationPoint& ip2 = Tr.GetElement2IntPoint();
    T2.SetIntPoint(&ip2);
    return sigma_->GetValue(T2, ip2);
  }

  ElementTransformation& T1 = *Tr.Elem1;
  const IntegrationPoint& ip1 = Tr.GetElement1IntPoint();
  T1.SetIntPoint(&ip1);
  return sigma_->GetValue(T1, ip1);
}

} // namespace mfem

#include "physics/igr/igr_flux.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{

namespace
{

double AverageUnOverRhoAlongSegment(const Vector& U1,
                                    const Vector& U2,
                                    const Vector& n,
                                    int dim)
{
  const double rho1 = U1(0);
  const double rho2 = U2(0);
  MFEM_VERIFY(rho1 > 0.0 && rho2 > 0.0,
              "FluxWithIGR::ComputeAvgFluxDotNAtSide requires positive "
              "density endpoints for exact averaged sigma correction.");

  double mn1 = 0.0;
  double mn2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    mn1 += U1(1 + d) * n(d);
    mn2 += U2(1 + d) * n(d);
  }
  const double dmn = mn2 - mn1;
  const double drho = rho2 - rho1;
  const double eps = 1.0e-14 * std::max(1.0, std::abs(rho1));
  if (std::abs(drho) <= eps)
  {
    return (mn1 + 0.5 * dmn) / rho1;
  }

  const double inv_drho = 1.0 / drho;
  const double inv_drho2 = inv_drho * inv_drho;
  const double log_ratio = std::log1p(drho / rho1);
  return dmn * inv_drho + (mn1 * drho - dmn * rho1) * inv_drho2 * log_ratio;
}

} // namespace

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
  return mcs; // keep dissipation unchanged
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
  return mcs;
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
  for (int d = 0; d < dim; ++d)
  {
    Fn(1 + d) += sigma_face * n(d);
  }
  Fn(1 + dim) += sigma_face * AverageUnOverRhoAlongSegment(U1, U2, n, dim);
  return mcs;
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

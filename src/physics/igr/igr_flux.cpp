#include "physics/igr/igr_flux.hpp"

#include <algorithm>

namespace mfem
{

real_t FluxWithIGR::ComputeFlux(const Vector &U, ElementTransformation &Tr,
                                DenseMatrix &F) const
{
  // Base tensor flux + base max characteristic speed
  const real_t mcs = base_.ComputeFlux(U, Tr, F);
  if (!sigma_) return mcs;

  const IntegrationPoint &ip = Tr.GetIntPoint(); // MFEM sets this before call
  const double sigma = sigma_->GetValue(Tr, ip);

  AddTensorSigmaTerms(U, sigma, F);
  return mcs; // keep dissipation unchanged
}

real_t FluxWithIGR::ComputeFluxDotN(const Vector &U, const Vector &n,
                                    FaceElementTransformations &Tr,
                                    Vector &Fn) const
{
  // Base normal flux + base (normal) max characteristic speed
  const real_t mcs = base_.ComputeFluxDotN(U, n, Tr, Fn);
  if (!sigma_) return mcs;

  // Select Sigma from the SAME side as the state U we were given.
  double sigma_face = 0.0;

  if (Tr.Elem2) // interior face
  {
    if (curr_side_ == FaceSide::Left)
    {
      ElementTransformation &T1 = *Tr.Elem1;
      const IntegrationPoint &ip1 = Tr.GetElement1IntPoint();
      T1.SetIntPoint(&ip1);
      sigma_face = sigma_->GetValue(T1, ip1);
    }
    else // Right
    {
      ElementTransformation &T2 = *Tr.Elem2;
      const IntegrationPoint &ip2 = Tr.GetElement2IntPoint();
      T2.SetIntPoint(&ip2);
      sigma_face = sigma_->GetValue(T2, ip2);
    }
  }
  else // boundary face -> only interior exists; use that Sigma
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

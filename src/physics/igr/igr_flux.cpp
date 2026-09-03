#include "physics/igr/igr_flux.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{

void FluxWithIGR::RecordFrozenSigmaSample(real_t pressure,
                                          real_t sigma) const
{
  IGRFrozenSigmaTelemetry& telemetry = frozen_sigma_telemetry_;
  telemetry.enabled = true;
  ++telemetry.sample_count;
  const double q_sigma = static_cast<double>(gamma_) * pressure +
                         (static_cast<double>(gamma_) - 1.0) * sigma;
  const double ratio =
    q_sigma / (static_cast<double>(gamma_) * pressure);
  const bool q_finite = std::isfinite(q_sigma);
  const bool ratio_finite = std::isfinite(ratio);
  if (!q_finite || !ratio_finite)
  {
    ++telemetry.nonfinite_count;
  }
  if (q_finite)
  {
    if (q_sigma <= 0.0)
    {
      ++telemetry.nonpositive_count;
    }
    telemetry.minimum_raw_q_sigma =
      std::min(telemetry.minimum_raw_q_sigma, q_sigma);
  }
  if (ratio_finite)
  {
    telemetry.minimum_q_sigma_over_gamma_p =
      std::min(telemetry.minimum_q_sigma_over_gamma_p, ratio);
  }
}

real_t FluxWithIGR::InvariantDomainSpeed(const Vector& U,
                                         const Vector* normal,
                                         real_t sigma,
                                         real_t base_speed) const
{
  const real_t density = U(0);
  const real_t safe_density = std::max(density, rho_vel_floor_);
  real_t momentum_sq = 0.0;
  real_t momentum_dot_normal = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const real_t momentum = U(1 + d);
    momentum_sq += momentum * momentum;
    if (normal)
    {
      momentum_dot_normal += momentum * (*normal)(d);
    }
  }
  const real_t pressure =
    (gamma_ - 1.0) * (U(1 + dim) - 0.5 * momentum_sq / safe_density);
  RecordFrozenSigmaSample(pressure, sigma);
  const real_t normal_norm = normal ? normal->Norml2() : 0.0;
  const real_t transport_speed = normal
                                   ? (normal_norm > 0.0
                                        ? std::abs(momentum_dot_normal) /
                                            (safe_density * normal_norm)
                                        : 0.0)
                                   : std::sqrt(momentum_sq) / safe_density;
  return IGRFluxInvariantDomainSpeed(base_speed,
                                     density,
                                     pressure,
                                     transport_speed,
                                     sigma,
                                     gamma_,
                                     rho_vel_floor_,
                                     invariant_domain_pressure_floor_);
}

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
  if (invariant_domain_flux_)
  {
    return InvariantDomainSpeed(U, nullptr, sigma, mcs);
  }
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
  if (invariant_domain_flux_)
  {
    return InvariantDomainSpeed(U, &n, sigma_face, mcs);
  }
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
  if (invariant_domain_flux_)
  {
    return std::max(InvariantDomainSpeed(U1, &n, sigma_face, mcs),
                    InvariantDomainSpeed(U2, &n, sigma_face, mcs));
  }
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

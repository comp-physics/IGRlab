#pragma once

#include <algorithm>
#include "mfem.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "utils/config/dg_igr_defaults.hpp"
#include "numerics/riemann/face_side.hpp"

/**
 * @file physics/igr/igr_flux.hpp
 * @brief IGR-aware flux decorator for Euler systems.
 * @ingroup physics_igr
 */

namespace mfem
{

/** Conservative explicit speed bound for the pressure-like IGR flux.
 *
 * Sigma has pressure units, so sqrt(|Sigma|/rho) is an additional velocity
 * scale that can be used independently for numerical-flux dissipation and
 * CFL accounting.
 */
MFEM_HOST_DEVICE inline real_t IGRFluxCharacteristicSpeed(real_t base_speed,
                                                           real_t rho,
                                                           real_t sigma,
                                                           real_t rho_floor)
{
  return ::dg_igr::frozen_scalar_igr_algebra::FluxCharacteristicSpeed(
    base_speed,
    rho,
    sigma,
    rho_floor);
}

/** Frozen-stage invariant-domain threshold for the IGR flux.
 *
 * For normal velocity u_n and total pressure P = p + sigma, positivity of
 * both Lax-Friedrichs split states U +/- F_IGR(U)/a requires
 *
 *   a >= |u_n| + |P| sqrt((gamma - 1)/(2 rho p)).
 *
 * The ordinary Euler characteristic speed already bounds this expression
 * when sigma is zero. During one explicit stage sigma is fixed by the stage
 * state, so the bound applies to that stage's face flux and CFL estimate.
 */
MFEM_HOST_DEVICE inline real_t IGRFluxInvariantDomainSpeed(
  real_t base_speed,
  real_t rho,
  real_t pressure,
  real_t transport_speed,
  real_t sigma,
  real_t gamma,
  real_t rho_floor,
  real_t pressure_floor = dg_igr::LimiterDefaults::kPMin)
{
  return ::dg_igr::frozen_scalar_igr_algebra::FluxInvariantDomainSpeed(
    base_speed,
    rho,
    pressure,
    transport_speed,
    sigma,
    gamma,
    rho_floor,
    pressure_floor);
}

/**
 * @brief Decorator that adds Information Geometric Regularization (IGR) to a base FluxFunction.
 *
 * Applies p -> p + Sigma consistently:
 *  - Tensor flux F(U):           momentum rows +Sigma I, energy row +Sigma u.
 *  - Normal flux (F(U)·n):       momentum +Sigma n,   energy +Sigma (u·n).
 *
 * Notes
 *  - The returned max characteristic speed includes the pressure-like IGR
 *    scale sqrt(|Sigma|/rho) when Sigma is active and sigma-aware CFL is on.
 *  - On faces, Sigma is taken from the explicit side argument in
 *    ComputeFluxDotNAtSide/ComputeAvgFluxDotNAtSide.
 *    On boundary faces (Elem2 == nullptr) the interior (Left) value is used.
 */
class FluxWithIGR : public FluxFunction, public FaceSideExplicitFlux
{
public:
  /// Construct an IGR-decorated flux around an existing base FluxFunction.
  explicit FluxWithIGR(const FluxFunction& base)
    : FluxFunction(base.num_equations, base.dim),
      base_(base)
  {
  }

  /// Provide the scalar Sigma field (same mesh as the solution). Pass nullptr to disable IGR.
  void SetSigmaField(const mfem::ParGridFunction* sigma) { sigma_ = sigma; }

  /// Density floor used only for u = m / rho in the IGR terms (default FluxDefaults::kRhoVelFloor).
  void SetVelocityRhoFloor(double f) { rho_vel_floor_ = f; }

  /// Include the sigma characteristic scale in the speed returned for CFL accounting.
  /// Disabled by default; enable it explicitly for sigma-aware CFL diagnostics.
  void SetSigmaAwareCFL(bool enabled) { sigma_aware_cfl_ = enabled; }

  /// Include the sigma characteristic scale in Rusanov dissipation.
  /// Disabled by default; enable it explicitly when requested by the case.
  void SetSigmaAwareRusanov(bool enabled) { sigma_aware_rusanov_ = enabled; }

  bool GetSigmaAwareCFL() const { return sigma_aware_cfl_; }
  bool GetSigmaAwareRusanov() const { return sigma_aware_rusanov_; }

  /// Select the Rusanov speed independently from the speed reported for CFL.
  real_t SelectRusanovDissipationSpeed(real_t reported_speed,
                                       real_t rho,
                                       real_t sigma) const
  {
    return ::dg_igr::frozen_scalar_igr_algebra::SelectRusanovDissipationSpeed(
      reported_speed,
      rho,
      sigma,
      rho_vel_floor_,
      sigma_aware_cfl_,
      sigma_aware_rusanov_);
  }

  /// Side-explicit normal flux path used by side-aware numerical fluxes.
  real_t ComputeFluxDotNAtSide(const Vector& state,
                               const Vector& normal,
                               FaceElementTransformations& Tr,
                               FaceSide side,
                               Vector& fluxDotN) const override;

  /// Side-explicit average normal flux path with exact line-average sigma term.
  real_t ComputeAvgFluxDotNAtSide(const Vector& state1,
                                  const Vector& state2,
                                  const Vector& normal,
                                  FaceElementTransformations& Tr,
                                  FaceSide side,
                                  Vector& fluxDotN) const override;

  /// Compute tensor flux F(U), including the IGR flux and configured CFL speed.
  real_t ComputeFlux(const Vector& state,
                     ElementTransformation& Tr,
                     DenseMatrix& flux) const override;

  /// Compute normal flux F(U)·n, including the IGR flux and configured CFL speed.
  real_t ComputeFluxDotN(const Vector& state,
                         const Vector& normal,
                         FaceElementTransformations& Tr,
                         Vector& fluxN) const override;

  /** Jacobians delegate the physical base term to the wrapped flux, then add
   * the frozen-Sigma pressure-like energy-row increment. The wrapped flux
   * must implement the requested Jacobian virtual.
   */
  void ComputeFluxJacobian(const Vector& state,
                           ElementTransformation& Tr,
                           DenseTensor& J_) const override
  {
    base_.ComputeFluxJacobian(state, Tr, J_);
    if (!sigma_)
    {
      return;
    }
    AddTensorSigmaJacobianTerms(state, GetSigmaAtPoint(Tr), J_);
  }

  void ComputeFluxJacobianDotN(const Vector& state,
                               const Vector& normal,
                               ElementTransformation& Tr,
                               DenseMatrix& JDotN) const override
  {
    base_.ComputeFluxJacobianDotN(state, normal, Tr, JDotN);
    if (!sigma_)
    {
      return;
    }
    AddDotNSigmaJacobianTerms(state, normal, GetSigmaAtPoint(Tr), JDotN);
  }

  /// Get Sigma at the current element integration point (0 if IGR is off).
  double GetSigmaAtPoint(ElementTransformation& Tr) const;

  /// Get Sigma for a face side at the current face integration point (0 if off).
  double GetSigmaAtSide(FaceElementTransformations& Tr, FaceSide side) const;

  /// Return true when the wrapped base flux is MFEM EulerFlux.
  bool HasEulerBase() const { return dynamic_cast<const EulerFlux*>(&base_) != nullptr; }

  /// Host-side identity/preflight access to the wrapped physical flux.
  const FluxFunction& GetBaseFlux() const { return base_; }

  /// Access the current Sigma field pointer (may be null).
  const mfem::ParGridFunction* GetSigmaField() const { return sigma_; }

  /// Access the density floor used only in the IGR energy sigma term.
  double GetVelocityRhoFloor() const { return rho_vel_floor_; }

private:
  const FluxFunction& base_;
  const mfem::ParGridFunction* sigma_ =
    nullptr; // non-owning; owned by IGR solver/state wiring
  double rho_vel_floor_ = dg_igr::FluxDefaults::kRhoVelFloor;
  bool sigma_aware_cfl_ = false;
  bool sigma_aware_rusanov_ = false;

  // helpers

  inline void
  AddTensorSigmaTerms(const Vector& U, double sigma, DenseMatrix& F) const
  {
    ::dg_igr::frozen_scalar_igr_algebra::AddTensorSigmaFlux(dim,
                                                            U.GetData(),
                                                            1,
                                                            sigma,
                                                            rho_vel_floor_,
                                                            F.GetData(),
                                                            1,
                                                            F.Height());
  }

  inline void AddDotNSigmaTerms(const Vector& U,
                                const Vector& n,
                                double sigma_face,
                                Vector& Fn) const
  {
    ::dg_igr::frozen_scalar_igr_algebra::AddNormalSigmaFlux(dim,
                                                            U.GetData(),
                                                            1,
                                                            n.GetData(),
                                                            1,
                                                            sigma_face,
                                                            rho_vel_floor_,
                                                            Fn.GetData(),
                                                            1);
  }

  inline void AddTensorSigmaJacobianTerms(const Vector& U,
                                          double sigma,
                                          DenseTensor& J_) const
  {
    ::dg_igr::frozen_scalar_igr_algebra::AddTensorSigmaFluxJacobian(
      dim,
      U.GetData(),
      1,
      sigma,
      rho_vel_floor_,
      J_.Data(),
      1,
      num_equations,
      num_equations * num_equations);
  }

  inline void AddDotNSigmaJacobianTerms(const Vector& U,
                                        const Vector& n,
                                        double sigma_face,
                                        DenseMatrix& JDotN) const
  {
    ::dg_igr::frozen_scalar_igr_algebra::AddNormalSigmaFluxJacobian(
      dim,
      U.GetData(),
      1,
      n.GetData(),
      1,
      sigma_face,
      rho_vel_floor_,
      JDotN.GetData(),
      1,
      JDotN.Height());
  }
};

} // namespace mfem

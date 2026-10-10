#pragma once

#include "igrlab/compat.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
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

/** Bounded aggregate over frozen-sigma characteristic-state samples.
 *
 * A sample is counted whenever exact invariant-domain mode evaluates a state
 * for its dissipation speed or independent q_sigma qualification telemetry.
 * `nonfinite_count` counts a sample when either q_sigma or its
 * normalized ratio q_sigma/(gamma*p) is non-finite.  Minima ignore their
 * respective non-finite values.  The DG operator combines this host aggregate
 * with its packed-device aggregates at end of run.
 */
struct IGRFrozenSigmaTelemetry
{
  bool enabled = false;
  std::uint64_t sample_count = 0;
  std::uint64_t nonpositive_count = 0;
  std::uint64_t nonfinite_count = 0;
  double minimum_raw_q_sigma = std::numeric_limits<double>::infinity();
  double minimum_q_sigma_over_gamma_p =
    std::numeric_limits<double>::infinity();

  void Merge(const IGRFrozenSigmaTelemetry& other)
  {
    enabled = enabled || other.enabled;
    sample_count += other.sample_count;
    nonpositive_count += other.nonpositive_count;
    nonfinite_count += other.nonfinite_count;
    minimum_raw_q_sigma =
      std::min(minimum_raw_q_sigma, other.minimum_raw_q_sigma);
    minimum_q_sigma_over_gamma_p =
      std::min(minimum_q_sigma_over_gamma_p,
               other.minimum_q_sigma_over_gamma_p);
  }
};

/// Read-only, communicator-wide aggregate supplied by an Euler operator.
class IGRFrozenSigmaTelemetryProvider
{
public:
  virtual ~IGRFrozenSigmaTelemetryProvider() = default;
  virtual IGRFrozenSigmaTelemetry GetGlobalFrozenSigmaTelemetry() const = 0;
};

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
  return ::igrlab::frozen_scalar_igr_algebra::FluxCharacteristicSpeed(
    base_speed,
    rho,
    sigma,
    rho_floor);
}

/** Frozen-stage invariant-domain and characteristic bound for the IGR flux.
 *
 * For normal velocity u_n and total pressure P = p + sigma, non-negativity of
 * the density and internal energy in both Lax-Friedrichs split states
 * U +/- F_IGR(U)/a requires
 *
 *   a >= |u_n| + |P| sqrt((gamma - 1)/(2 rho p)).
 *
 * The returned speed is the maximum of that split threshold, the ordinary
 * Euler speed, and a modulus bound for the frozen-sigma local roots.  The
 * latter are strictly hyperbolic only when
 * gamma*p + (gamma - 1)*sigma > 0; a finite modulus bound when this quantity
 * is nonpositive is not a hyperbolicity or stability claim.  This local
 * statement also does not characterize the nonlocal reduced DAE Jacobian.
 */
MFEM_HOST_DEVICE inline real_t IGRFluxInvariantDomainSpeed(
  real_t base_speed,
  real_t rho,
  real_t pressure,
  real_t transport_speed,
  real_t sigma,
  real_t gamma,
  real_t rho_floor,
  real_t pressure_floor = igrlab::LimiterDefaults::kPMin)
{
  return ::igrlab::frozen_scalar_igr_algebra::FluxInvariantDomainSpeed(
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
 *  - The typed frozen-sigma spectral mode reports the exact local acoustic
 *    radius |u_n|+sqrt((gamma*p+(gamma-1)*Sigma)/rho) on admitted states and
 *    uses it for Rusanov dissipation. The two legacy sigma-aware toggles retain
 *    their independent heuristic behavior. The invariant-domain mode instead
 *    reports the combined frozen-stage split/characteristic bound.
 *  - On faces, Sigma is taken from the explicit side argument in
 *    ComputeFluxDotNAtSide/ComputeAvgFluxDotNAtSide.
 *    On boundary faces (Elem2 == nullptr) the interior (Left) value is used.
 */
class FluxWithIGR : public FluxFunction, public FaceSideExplicitFlux
{
public:
  /// Construct an IGR-decorated flux around an existing base FluxFunction.
  explicit FluxWithIGR(
    const FluxFunction& base,
    real_t gamma = igrlab::EulerDefaults::kGamma)
    : FluxFunction(base.num_equations, base.dim),
      base_(base),
      gamma_(gamma)
  {
  }

  /// Provide the scalar Sigma field (same mesh as the solution). Pass nullptr to disable IGR.
  void SetSigmaField(const mfem::ParGridFunction* sigma) { sigma_ = sigma; }

  /// Density floor used only for u = m / rho in the IGR terms (default FluxDefaults::kRhoVelFloor).
  void SetVelocityRhoFloor(double f) { rho_vel_floor_ = f; }

  /** Floors defining the admitted state set for invariant-domain mode.
   *
   * The density floor is also used by the frozen-sigma energy flux.  Keeping
   * both values on the flux object lets host and packed-device paths share the
   * limiter's configured admissibility contract.
   */
  void SetInvariantDomainFloors(double density_floor,
                                double pressure_floor)
  {
    MFEM_VERIFY(std::isfinite(density_floor) && density_floor > 0.0,
                "The IGR invariant-domain density floor must be finite and "
                "positive.");
    MFEM_VERIFY(std::isfinite(pressure_floor) && pressure_floor > 0.0,
                "The IGR invariant-domain pressure floor must be finite and "
                "positive.");
    rho_vel_floor_ = density_floor;
    invariant_domain_pressure_floor_ = pressure_floor;
  }

  /// Include the sigma characteristic scale in the speed returned for CFL accounting.
  /// Disabled by default; enable it explicitly for sigma-aware CFL diagnostics.
  void SetSigmaAwareCFL(bool enabled) { sigma_aware_cfl_ = enabled; }

  /// Include the sigma characteristic scale in Rusanov dissipation.
  /// Disabled by default; enable it explicitly when requested by the case.
  void SetSigmaAwareRusanov(bool enabled) { sigma_aware_rusanov_ = enabled; }

  /** Couple CFL reporting and Rusanov dissipation to the exact local
   * frozen-sigma spectrum. The extrapolated DGSEM production method owns this
   * typed policy; q_sigma admission is enforced separately at every stage and
   * exact endpoint.
   */
  void SetFrozenSigmaSpectralSpeed(bool enabled)
  {
    MFEM_VERIFY(!enabled || HasEulerBase(),
                "The frozen-sigma spectral speed requires an Euler base flux.");
    frozen_sigma_spectral_speed_ = enabled;
  }

  /** Use the frozen-stage IGR split/characteristic speed for both CFL
   * accounting and Rusanov dissipation. This typed mode takes precedence over
   * the two independent legacy sigma-speed heuristics. "Invariant-domain"
   * names the exact LF split condition; it is not by itself a complete DG
   * positivity theorem or a reduced-DAE hyperbolicity claim.
   */
  void SetInvariantDomainFlux(bool enabled)
  {
    MFEM_VERIFY(!enabled || HasEulerBase(),
                "The IGR invariant-domain flux requires an Euler base flux.");
    invariant_domain_flux_ = enabled;
  }

  bool GetSigmaAwareCFL() const { return sigma_aware_cfl_; }
  bool GetSigmaAwareRusanov() const { return sigma_aware_rusanov_; }
  bool GetFrozenSigmaSpectralSpeed() const
  {
    return frozen_sigma_spectral_speed_;
  }
  bool GetInvariantDomainFlux() const { return invariant_domain_flux_; }
  real_t GetGamma() const { return gamma_; }

  /** Host-path aggregate since construction.
   *
   * This includes every attempted exact-mode state sample, including samples
   * from subsequently rejected time-step attempts.  It stores no event history.
   */
  IGRFrozenSigmaTelemetry GetFrozenSigmaTelemetry() const
  {
    return frozen_sigma_telemetry_;
  }

  /// Select the legacy independent Rusanov speed, or the exact coupled speed.
  real_t SelectRusanovDissipationSpeed(real_t reported_speed,
                                       real_t rho,
                                       real_t sigma) const
  {
    if (invariant_domain_flux_)
    {
      return reported_speed;
    }
    if (frozen_sigma_spectral_speed_)
    {
      return reported_speed;
    }
    return ::igrlab::frozen_scalar_igr_algebra::SelectRusanovDissipationSpeed(
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

  /// Access the pressure floor defining the invariant-domain admitted set.
  double GetInvariantDomainPressureFloor() const
  {
    return invariant_domain_pressure_floor_;
  }

private:
  const FluxFunction& base_;
  real_t gamma_ = igrlab::EulerDefaults::kGamma;
  const mfem::ParGridFunction* sigma_ =
    nullptr; // non-owning; owned by IGR solver/state wiring
  double rho_vel_floor_ = igrlab::FluxDefaults::kRhoVelFloor;
  double invariant_domain_pressure_floor_ = igrlab::LimiterDefaults::kPMin;
  bool sigma_aware_cfl_ = false;
  bool sigma_aware_rusanov_ = false;
  bool frozen_sigma_spectral_speed_ = false;
  bool invariant_domain_flux_ = false;
  mutable IGRFrozenSigmaTelemetry frozen_sigma_telemetry_;

  // helpers

  real_t InvariantDomainSpeed(const Vector& U,
                              const Vector* normal,
                              real_t sigma,
                              real_t base_speed) const;

  real_t FrozenSigmaSpectralSpeed(const Vector& U,
                                  const Vector* normal,
                                  real_t sigma) const;

  void RecordFrozenSigmaSample(real_t pressure, real_t sigma) const;

  inline void
  AddTensorSigmaTerms(const Vector& U, double sigma, DenseMatrix& F) const
  {
    ::igrlab::frozen_scalar_igr_algebra::AddTensorSigmaFlux(dim,
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
    ::igrlab::frozen_scalar_igr_algebra::AddNormalSigmaFlux(dim,
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
    ::igrlab::frozen_scalar_igr_algebra::AddTensorSigmaFluxJacobian(
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
    ::igrlab::frozen_scalar_igr_algebra::AddNormalSigmaFluxJacobian(
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

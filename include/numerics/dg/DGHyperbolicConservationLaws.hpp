#pragma once
/**
 * @file numerics/dg/DGHyperbolicConservationLaws.hpp
 * @brief Time-dependent DG operator for hyperbolic conservation laws.
 * @ingroup numerics_dg
 *
 * This code provides example problems for the Euler equations and implements
 * the time-dependent DG operator given by the equation:
 *
 *            (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u, n), [[v]]>_F = 0.
 *
 * This operator is designed for explicit time stepping methods. Specifically,
 * the function DGHyperbolicConservationLaws::Mult implements the following
 * transformation:
 *
 *                             u ↦ M⁻¹(-DF(u) + NF(u))
 *
 * where M is the mass matrix, DF is the weak divergence of flux, and NF is the
 * interface flux. The inverse of the mass matrix is computed element-wise by
 * leveraging the block-diagonal structure of the DG mass matrix. Additionally,
 * the flux-related terms are computed using the HyperbolicFormIntegrator.
 *
 * The maximum characteristic speed is determined for each time step. For more
 * details, refer to the documentation of DGHyperbolicConservationLaws::Mult.
 */

#include <functional>
#include <cstdint>
#include "mfem.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dgsem/dgsem_production.hpp"
#include "numerics/riemann/fluxfunctions.hpp"
#include "physics/igr/igr_flux.hpp"

namespace mfem
{
namespace dgsem
{
class DGSEMOperator;
}

/// Choose which method to compute the *volume* contribution:
///  - WeakDivergence: (F(u), ∇v)_T via preassembled weak divergence or matrix-free
///  - DGSEMFluxDiff : flux-difference (“pairwise flux”) volume term in DGSEM style
enum class VolumeMode
{
  WeakDivergence,
  DGSEMFluxDiff
};

/// Optional DGSEM stabilization hook selection (placeholder).
enum class DGSEMStabilization
{
  None,
  SplitForm,
  Filter
};

/// DGSEM volume flux selection for entropy-conservative variants (Euler-only).
enum class DGSEMVolumeFlux
{
  Central,
  Chandrashekar,
  Ranocha
};

/// DGSEM volume dissipation selection (added to flux-differencing pair flux).
enum class DGSEMVolumeDissipation
{
  None,
  LLF,
  EntropyVarsLF
};

/// Optional one-shot provider for an operator's physical-boundary flux.
///
/// A caller arms the provider immediately before the RHS Mult() it wants to
/// observe. The completed vector is the communicator-wide integral of the
/// outward numerical flux used by that Mult(); unarmed calls preserve it.
class PhysicalBoundaryFluxIntegralProvider
{
public:
  virtual ~PhysicalBoundaryFluxIntegralProvider() = default;
  virtual bool SupportsPhysicalBoundaryFluxIntegral() const = 0;
  virtual void ArmPhysicalBoundaryFluxIntegral() const = 0;
  virtual bool HasLastPhysicalBoundaryFluxIntegral() const = 0;
  virtual const Vector& GetLastPhysicalBoundaryFluxIntegral() const = 0;
};

/// ---------------------------------------------------------------------------
/// @brief Time-dependent DG operator for hyperbolic conservation laws.
/// @ingroup numerics_dg
///
/// Implements the semi-discrete form:
///      (u_t, v)_T = (F(u), ∇v)_T - < F̂(u, n), [[v]] >_F
///
/// Provides y = M⁻¹( (F(u), ∇v) - <F̂, [[v]]> ) suitable for explicit timestepping.
/// The volume term strategy can be switched at runtime between:
///  - WeakDivergence (default): preassembled (I_h F(u), ∇v) via a local matrix
///  - DGSEMFluxDiff: DGSEM flux-difference approach at nodal points
/// ---------------------------------------------------------------------------
class DGHyperbolicConservationLaws
  : public TimeDependentOperator,
    public PhysicalBoundaryFluxIntegralProvider,
    public IGRFrozenSigmaTelemetryProvider
{
public:
  /**
    * @brief Construct a DG operator with a given flux integrator.
    *
    * @param vfes_                     Vector finite element space; tested for DG [P_p]^n.
    * @param formIntegrator_           Face/domain flux integrator (provides ComputeFlux + numerical flux).
    * @param preassembleWeakDivergence If true, preassemble local weak-divergence matrices.
    */
  DGHyperbolicConservationLaws(
    FiniteElementSpace& vfes_,
    std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_,
    bool preassembleWeakDivergence = true);

  DGHyperbolicConservationLaws(const DGHyperbolicConservationLaws&) = delete;
  DGHyperbolicConservationLaws&
  operator=(const DGHyperbolicConservationLaws&) = delete;
  DGHyperbolicConservationLaws(DGHyperbolicConservationLaws&&);
  DGHyperbolicConservationLaws&
  operator=(DGHyperbolicConservationLaws&&) = delete;

  /// Select volume-contribution strategy (WeakDivergence or DGSEMFluxDiff).
  void SetVolumeMode(VolumeMode m) { vol_mode_ = m; }

  /// Get the current volume-contribution strategy.
  VolumeMode GetVolumeMode() const { return vol_mode_; }

  /// Select optional DGSEM stabilization hook (not implemented yet).
  void SetDGSEMStabilization(DGSEMStabilization s) { dgsem_stab_ = s; }

  /// Get current DGSEM stabilization hook selection.
  DGSEMStabilization GetDGSEMStabilization() const { return dgsem_stab_; }

  /// Set the numerical flux used for DGSEM surface evaluation (non-owning).
  void SetNumericalFlux(const NumericalFlux* flux);

  /// Select DGSEM volume flux (central or entropy-conservative).
  void SetDGSEMVolumeFlux(DGSEMVolumeFlux f) { dgsem_vol_flux_ = f; }

  /// Get current DGSEM volume flux.
  DGSEMVolumeFlux GetDGSEMVolumeFlux() const { return dgsem_vol_flux_; }

  /// Set Euler gamma used by entropy-conservative DGSEM fluxes.
  void SetDGSEMGamma(double gamma) { dgsem_gamma_ = gamma; }

  /// Set face quadrature order offset (2*p + offset) for DG face integration.
  void SetFaceQuadratureOrderOffset(int offset)
  {
    face_int_order_offset_ = offset;
    face_data_dirty_ = true;
  }

  /// Get Euler gamma used by DGSEM entropy-conservative fluxes.
  double GetDGSEMGamma() const { return dgsem_gamma_; }

  /// Select DGSEM volume dissipation mode.
  void SetDGSEMVolumeDissipation(DGSEMVolumeDissipation d)
  {
    dgsem_vol_diss_ = d;
  }

  /// Get DGSEM volume dissipation mode.
  DGSEMVolumeDissipation GetDGSEMVolumeDissipation() const
  {
    return dgsem_vol_diss_;
  }

  /// Select entropy Jacobian mode for entropy-variable LF volume dissipation.
  void SetDGSEMEntropyJacobianAnalytic(bool analytic)
  {
    dgsem_entropy_jacobian_analytic_ = analytic;
  }

  /// Query entropy Jacobian mode for entropy-variable LF volume dissipation.
  bool GetDGSEMEntropyJacobianAnalytic() const
  {
    return dgsem_entropy_jacobian_analytic_;
  }

  /**
   * @brief Set a custom face integration rule for surface flux evaluation.
   *
   * Applies the rule to the interior-face integrator and caches it so boundary
   * integrators added later inherit the same rule.
   */
  void SetFaceIntegrationRule(const IntegrationRule& ir);

  /**
    * @brief Apply the operator: y = M⁻¹( -<F̂, [[v]]> + (F(u), ∇v) ).
    *
    * If WeakDivergence mode is active and weak divergence is preassembled,
    * the local volume term is added by multiplying the “weakdiv” matrix by nodal fluxes.
    * If DGSEMFluxDiff mode is active, the DGSEM-style volume flux-difference is added.
    */
  void Mult(const Vector& x, Vector& y) const override;

  /// Recompute data if mesh/fe space changes (mass matrix, weak divergence, etc.).
  void Update();

  /// Max characteristic speed seen during the last Mult() (for CFL).
  real_t GetMaxCharSpeed() { return max_char_speed; }

  /** Request volume-node CFL accounting for exact Euler+IGR fluxes.
   *
   * Disabled by default so fixed-step runs and legacy fluxes retain their
   * original volume-residual hot path. Even when requested, volume speeds are
   * evaluated only when an active FluxWithIGR uses invariant-domain fluxes;
   * all other configurations continue to report face/boundary speeds only.
   * The time loop should enable this only while adaptive CFL selection needs
   * the additional volume-node bound.
   */
  void SetInvariantDomainVolumeSpeedTracking(bool enabled)
  {
    invariant_domain_volume_speed_tracking_ = enabled;
  }
  bool GetInvariantDomainVolumeSpeedTracking() const
  {
    return invariant_domain_volume_speed_tracking_;
  }

  /** Communicator-wide aggregate over every exact-mode q_sigma sample.
   *
   * Counts include CFL probes, accepted-stage evaluations, and work from
   * rejected attempts. MPI sums rank-local execution counts, so the two
   * executions of a shared-face sample are deliberately two work samples,
   * not one deduplicated physical point. Reading this bounded aggregate
   * synchronizes packed-device telemetry only once at the caller-selected
   * summary point.
   */
  IGRFrozenSigmaTelemetry GetGlobalFrozenSigmaTelemetry() const override;

  /// Install a time-independent source already projected into this DG space.
  /// An identically zero source is discarded, leaving the RHS hot path free of
  /// source vector work.
  void SetStaticSource(const Vector& source);
  bool HasStaticSource() const { return static_source_.Size() != 0; }
  bool StaticSourceRequiresRebind() const
  {
    return static_source_requires_rebind_;
  }

  /// Whether the active DG volume/boundary path can capture the complete flux.
  bool SupportsPhysicalBoundaryFluxIntegral() const override;

  /**
   * @brief Arm a one-shot outward physical-boundary flux integral.
   *
   * The next Mult() consumes the request. Unarmed Mult() calls, including CFL
   * probes, neither accumulate into nor overwrite the last completed capture.
   * On a parallel finite-element space this request must be armed collectively
   * because the completed integral is summed over the space communicator.
   */
  void ArmPhysicalBoundaryFluxIntegral() const override
  {
    physical_boundary_flux_integral_armed_ = true;
    physical_boundary_flux_integral_available_ = false;
  }

  /// Whether a requested Mult() completed a supported flux capture.
  bool HasLastPhysicalBoundaryFluxIntegral() const override
  {
    return physical_boundary_flux_integral_available_;
  }

  /// Last completed communicator-wide outward numerical-flux integral.
  const Vector& GetLastPhysicalBoundaryFluxIntegral() const override
  {
    MFEM_VERIFY(physical_boundary_flux_integral_available_,
                "No completed physical-boundary flux integral is available.");
    return physical_boundary_flux_integral_;
  }

  /// Read-only production selection from the persistent Mult bridge.
  dgsem::DGSEMProductionSelection GetDGSEMProductionSelection() const;

  /// Read-only worker cache counters reached through production Mult.
  dgsem::DGSEMProductionCacheCounters
  GetDGSEMProductionCacheCounters() const;

  /// Read-only packed-worker workspace identity reached through Mult.
  dgsem::DGSEMProductionWorkspaceEvidence
  GetDGSEMProductionWorkspaceEvidence() const;

  /// Coordinator bridge lifetime/workspace evidence. Evidence-only: taking
  /// the snapshot synchronizes MFEM memory and may materialize host/device
  /// views; it is not an allocation-free hot-path observation.
  dgsem::DGSEMProductionBridgeEvidence
  GetDGSEMProductionBridgeEvidence() const;

  /**
   * @brief Add a boundary-face integrator with a marker (MFEM-style).
   *
   * Ownership semantics: the operator stores the integrator and a heap copy of the marker,
   * then registers them into the underlying NonlinearForm with UseExternalIntegrators().
   */
  void AddBdrFaceIntegrator(std::unique_ptr<NonlinearFormIntegrator> nfi,
                            const Array<int>& bdr_marker);

  /// Compatibility overload; ownership is transferred to the unique_ptr API.
  void AddBdrFaceIntegrator(NonlinearFormIntegrator* nfi,
                            const Array<int>& bdr_marker)
  {
    AddBdrFaceIntegrator(std::unique_ptr<NonlinearFormIntegrator>(nfi),
                         bdr_marker);
  }

  ~DGHyperbolicConservationLaws() override;

private:
  friend class dgsem::DGSEMOperator;

  // --- Problem description and finite element spaces ---
  const int num_equations;  ///< number of conservative equations
  const int dim;            ///< space dimension
  FiniteElementSpace& vfes; ///< vector FE space (vdim = num_equations)

  // --- Nonlinear assembly / integrators ---
  std::unique_ptr<HyperbolicFormIntegrator> formIntegrator;
  std::unique_ptr<NonlinearForm> nonlinearForm;

  // --- Local per-element operators ---
  std::vector<DenseMatrix> invmass; ///< inverse local mass matrices (dof × dof)
  std::vector<DenseMatrix>
    weakdiv; ///< local weak-divergence (dof × (dof*dim); columns grouped by node then dim)

  // --- Runtime state ---
  mutable real_t max_char_speed =
    0.0;            ///< updated during Mult() by the form integrator
  mutable Vector z; ///< scratch for the assembled RHS before M⁻¹
  Vector static_source_; ///< optional projected source in conservative ordering
  bool static_source_requires_rebind_ = false;
  bool invariant_domain_volume_speed_tracking_ = false;
  mutable bool physical_boundary_flux_integral_armed_ = false;
  mutable bool physical_boundary_flux_integral_available_ = false;
  mutable Vector physical_boundary_flux_integral_;

  mutable std::unique_ptr<dgsem::DGSEMOperator> dgsem_production_operator_;
  mutable Vector dgsem_x_local_workspace_;
  mutable Vector dgsem_z_local_workspace_;
  mutable Vector dgsem_y_local_workspace_;
  mutable bool dgsem_production_dispatch_observed_ = false;
  mutable bool dgsem_production_dispatch_supported_ = false;
  mutable bool dgsem_production_dispatch_strict_ = false;
  mutable bool dgsem_production_dispatch_cuda_ = false;
  mutable dgsem::DGSEMProductionReason dgsem_production_dispatch_reason_ =
    dgsem::DGSEMProductionReason::None;
  mutable std::int64_t dgsem_production_operator_constructions_ = 0;
  mutable std::int64_t dgsem_production_operator_rebinds_ = 0;
  mutable std::int64_t dgsem_production_mult_calls_ = 0;
  mutable std::int64_t dgsem_production_fallback_calls_ = 0;

  // Keep external boundary integrators alive + stable copies of their markers.
  std::vector<std::unique_ptr<NonlinearFormIntegrator>> ext_bdr_int_;
  std::vector<std::unique_ptr<Array<int>>> ext_bdr_markers_;
  std::unique_ptr<IntegrationRule> face_int_rule_owned_;
  const IntegrationRule* face_int_rule_ = nullptr;

  // Volume strategy
  VolumeMode vol_mode_ = VolumeMode::WeakDivergence;
  const NumericalFlux* num_flux_ = nullptr;
  DGSEMStabilization dgsem_stab_ = DGSEMStabilization::None;
  DGSEMVolumeFlux dgsem_vol_flux_ = DGSEMVolumeFlux::Central;
  DGSEMVolumeDissipation dgsem_vol_diss_ = DGSEMVolumeDissipation::None;
  double dgsem_gamma_ = 1.4;
  bool dgsem_entropy_jacobian_analytic_ = false;
  mutable bool dgsem_timing_enabled_ = false;
  mutable bool dgsem_timing_init_ = false;

  struct DGSEMTimingStats
  {
    std::int64_t calls = 0;
    double surface_seconds = 0.0;
    double volume_seconds = 0.0;
    double invmass_seconds = 0.0;
    double total_seconds = 0.0;
  };
  mutable DGSEMTimingStats dgsem_timing_stats_;

  // --- Helpers ---
  /// Compute per-element inverse mass matrices (block-diagonal DG mass).
  void ComputeInvMass();

  /// Preassemble the weak-divergence operator on each element (optional).
  /// The layout is (test dof) × (trial node × dim), i.e. ByDim grouped per node.
  void ComputeWeakDivergence();

  /// Build packed data required by the inverse-mass apply path.
  void BuildInvMassApplyData_();

  /// Build packed data required by the preassembled weak-divergence apply path.
  void BuildWeakDivApplyData_();

  /// Apply y = M^{-1} z using the best available path for the active backend.
  void ApplyInverseMass_(const Vector& z_in, Vector& y_out) const;

  /// Add the preassembled weak-divergence volume contribution into z.
  real_t AddPreassembledWeakDivergence_(const Vector& x_in,
                                        Vector& z_inout) const;

  /// Build interpolation data for sigma(e, state-node) sampling on device.
  bool BuildWeakDivSigmaData_(const ParGridFunction& sigma) const;

  /// Build packed interior/boundary face data for device DG surface assembly.
  void BuildFaceApplyData_();

  /// Build interpolation data for sigma(e, face-qpoint) sampling on device.
  bool BuildFaceSigmaData_(const ParGridFunction& sigma) const;

  /// Build boundary-attribute policy data for device boundary-face kernels.
  bool BuildBoundaryPolicyData_();

  /// Find a boundary face integrator matching the given boundary attribute.
  const BdrHyperbolicFluxIntegrator* FindBdrIntegrator_(int attr) const;

  /// Return true when the custom device face residual path is available.
  bool UseDeviceFaceResidual_() const;

  /// Add DG face contributions with a packed Euler/Rusanov device path.
  bool AddDeviceFaceContributionsEulerRusanov_(const Vector& x_in,
                                               Vector& z_inout,
                                               real_t& max_face_speed,
                                               bool capture_boundary_flux) const;

  /// Reduce a per-face speed vector to max speed with device support.
  real_t ReduceFaceMaxSpeed_(const Vector& face_speed,
                             Array<real_t>& workspace) const;

  /// Device weak-divergence path for Euler-family fluxes.
  real_t AddPreassembledWeakDivergenceDeviceEuler_(
    const Vector& x_in,
    Vector& z_inout,
    bool use_sigma,
    const ParGridFunction* sigma_field,
    real_t sigma_rho_floor,
    real_t invariant_domain_pressure_floor,
    bool invariant_domain_flux,
    bool track_volume_speed) const;

  /// Packed inverse-mass apply (element-major, equation-minor map).
  void ApplyInverseMassPacked_(const Vector& z_in, Vector& y_out) const;

  /// Host fallback inverse-mass apply (packed element map, no subvector staging).
  void ApplyInverseMassHost_(const Vector& z_in, Vector& y_out) const;

  /// Return true when packed inverse-mass path should run on device.
  bool UsePackedInverseMassDevice_() const;

  /// Build MFEM-native inverse-mass solver for accelerated backends.
  void BuildNativeMassInverseSolver_();

  /// Return true when MFEM-native inverse-mass solver is available.
  bool UseNativeMassInverseSolver_() const;

  bool IsDGSEMTimingEnabled_() const;
  void RecordDGSEMTiming_(double surface_seconds,
                          double volume_seconds,
                          double invmass_seconds,
                          double total_seconds) const;
  void PrintDGSEMTimingSummary_() const;

  /// Return the persistent worker, repairing ownership after a parent move.
  dgsem::DGSEMOperator& GetDGSEMProductionOperator_() const;

  /// True only on rank zero of the active FE-space communicator.
  bool IsDGSEMProductionLogRoot_() const;


  /// Per-element scalar dof count used by packed inverse-mass apply.
  int invmass_elem_dof_ = 0;
  /// Flattened per-element inverse mass blocks (column-major dof x dof).
  Vector invmass_data_flat_;
  /// Per-element vector-dof map in node-major, equation-minor layout.
  Array<int> invmass_elem_vdof_map_;
  /// Whether packed inverse-mass data is available.
  bool invmass_apply_ready_ = false;
  /// Ensures inverse-mass path selection is logged only once.
  mutable bool invmass_path_logged_ = false;
  /// Ensures DG face/domain primary path is logged only once.
  mutable bool dg_face_path_logged_ = false;
  /// Marks packed face data as stale after topology/configuration changes.
  mutable bool face_data_dirty_ = true;
  /// Ensures weak-divergence volume path selection is logged only once.
  mutable bool weakdiv_volume_path_logged_ = false;

  /// Default face quadrature order offset for custom DG face assembly.
  int face_int_order_offset_ = 1;
  /// Packed DG face apply is ready when all required metadata exists.
  bool face_apply_ready_ = false;
  /// Packed interior face scalar dof count.
  int face_elem_dof_ = 0;
  /// Interior-face quadrature point count.
  int face_int_nq_ = 0;
  /// Boundary-face quadrature point count.
  int face_bdr_nq_ = 0;
  /// Shared-face quadrature point count (MPI face-neighbor interfaces).
  int face_shared_nq_ = 0;
  /// Packed interior face -> element 1 index map.
  Array<int> face_int_e1_;
  /// Packed interior face -> element 2 index map.
  Array<int> face_int_e2_;
  /// Packed interior face -> mesh face index map.
  Array<int> face_int_mesh_id_;
  /// Packed interior face shape values on side 1 (face, q, dof).
  Vector face_int_shape1_;
  /// Packed interior face shape values on side 2 (face, q, dof).
  Vector face_int_shape2_;
  /// Packed interior face normals (face, q, dim).
  Vector face_int_normals_;
  /// Packed interior face quadrature weights (face, q).
  Vector face_int_weights_;
  /// Packed interior side-1 reference coordinates (face, q, dim).
  Vector face_int_ip1_ref_;
  /// Packed interior side-2 reference coordinates (face, q, dim).
  Vector face_int_ip2_ref_;
  /// Per-interior-face max speed scratch.
  mutable Vector face_int_speed_;
  /// Per-interior-face quadrature-point speed scratch (face, q).
  mutable Vector face_int_speed_q_;

  /// Packed boundary face -> element 1 index map.
  Array<int> face_bdr_e1_;
  /// Packed boundary face -> boundary attribute map.
  Array<int> face_bdr_attr_;
  /// Packed boundary face -> boundary-face index map.
  Array<int> face_bdr_mesh_id_;
  /// Packed boundary face shape values on side 1 (face, q, dof).
  Vector face_bdr_shape1_;
  /// Packed boundary face normals (face, q, dim).
  Vector face_bdr_normals_;
  /// Packed boundary face quadrature weights (face, q).
  Vector face_bdr_weights_;
  /// Packed boundary side-1 reference coordinates (face, q, dim).
  Vector face_bdr_ip1_ref_;
  /// Per-boundary-face max speed scratch.
  mutable Vector face_bdr_speed_;
  /// Per-boundary-face quadrature-point speed scratch (face, q).
  mutable Vector face_bdr_speed_q_;
  /// Packed shared face -> local element index map.
  Array<int> face_shared_e1_;
  /// Packed shared-face neighbor vector-dof map (face, node, eq).
  Array<int> face_shared_nbr_vdof_map_;
  /// Packed shared face shape values on local side 1 (face, q, dof).
  Vector face_shared_shape1_;
  /// Packed shared face shape values on neighbor side 2 (face, q, dof).
  Vector face_shared_shape2_;
  /// Packed shared face normals (face, q, dim).
  Vector face_shared_normals_;
  /// Packed shared face quadrature weights (face, q).
  Vector face_shared_weights_;
  /// Packed shared side-1 reference coordinates (face, q, dim).
  Vector face_shared_ip1_ref_;
  /// Packed shared side-2 reference coordinates (face, q, dim).
  Vector face_shared_ip2_ref_;
  /// Per-shared-face max speed scratch.
  mutable Vector face_shared_speed_;
  /// Per-shared-face quadrature-point speed scratch (face, q).
  mutable Vector face_shared_speed_q_;

#ifdef MFEM_USE_MPI
  /// Reused alias for state L-vector neighbor exchange (avoids per-step temp
  /// ParGridFunction construction in packed shared-face assembly).
  mutable std::unique_ptr<ParGridFunction> face_state_face_nbr_alias_;
#endif

  /// Max boundary attribute value represented in policy arrays.
  int face_bdr_max_attr_ = 0;
  /// Boundary policy type by attribute (enum value stored as int).
  Array<int> face_bdr_type_by_attr_;
  /// Boundary policy gamma by attribute.
  Vector face_bdr_gamma_by_attr_;
  /// Boundary policy outlet pressure by attribute (if applicable).
  Vector face_bdr_pressure_by_attr_;
  /// Boundary policy reference state by attribute (flattened attr-major).
  Vector face_bdr_state_by_attr_;
  /// Boundary policy metadata is complete for all active boundary attributes.
  bool face_bdr_policy_ready_ = false;

  /// Per-element sigma scalar dof count for face interpolation.
  mutable int face_sigma_ldof_ = 0;
  /// Per-element sigma scalar-dof map for face interpolation.
  mutable Array<int> face_sigma_elem_dof_map_;
  /// Interior-face sigma shape values on side 1 (face, q, sigma_ldof).
  mutable Vector face_int_sigma_shape1_;
  /// Interior-face sigma shape values on side 2 (face, q, sigma_ldof).
  mutable Vector face_int_sigma_shape2_;
  /// Boundary-face sigma shape values on side 1 (face, q, sigma_ldof).
  mutable Vector face_bdr_sigma_shape1_;
  /// Shared-face sigma shape values on side 1 (face, q, sigma_ldof).
  mutable Vector face_shared_sigma_shape1_;
  /// Shared-face sigma shape values on side 2 (face, q, sigma_ldof).
  mutable Vector face_shared_sigma_shape2_;
  /// Shared-face sigma neighbor-dof map (face, sigma_ldof).
  mutable Array<int> face_shared_sigma_nbr_dof_map_;
  /// Face sigma interpolation metadata is valid for current sigma space.
  mutable bool face_sigma_ready_ = false;
  /// Sigma FE-space used to build face interpolation metadata.
  mutable const FiniteElementSpace* face_sigma_fes_ = nullptr;
  /// Sigma FE-space sequence used to build face interpolation metadata.
  mutable long face_sigma_fes_sequence_ = -1;
  /// Reduction workspace for interior-face max-speed reduction.
  mutable Array<real_t> face_int_speed_reduce_workspace_;
  /// Reduction workspace for boundary-face max-speed reduction.
  mutable Array<real_t> face_bdr_speed_reduce_workspace_;
  /// Reduction workspace for shared-face max-speed reduction.
  mutable Array<real_t> face_shared_speed_reduce_workspace_;

  /// Per-element scalar dof count used by packed weak-divergence apply.
  int weakdiv_elem_dof_ = 0;
  /// Flattened per-element weak-divergence blocks (row-major dof x (dof*dim)).
  Vector weakdiv_data_flat_;
  /// Per-element maximum volume-node characteristic speed scratch.
  mutable Vector weakdiv_volume_speed_;
  /// Persistent per-volume-node q_sigma aggregates (five FP64 scalars/slot).
  mutable Array<double> weakdiv_q_sigma_telemetry_;
  /// Reduction workspace for the volume-node maximum speed.
  mutable Array<real_t> weakdiv_volume_speed_reduce_workspace_;
  /// Whether packed weak-divergence data is available.
  bool weakdiv_apply_ready_ = false;

  /// Per-element scalar dof count for sigma interpolation.
  mutable int weakdiv_sigma_ldof_ = 0;
  /// Per-element sigma scalar-dof map (for packed interpolation).
  mutable Array<int> weakdiv_sigma_elem_dof_map_;
  /// Interpolation weights from sigma dofs to state nodes per element.
  mutable Vector weakdiv_sigma_interp_;
  /// Whether packed sigma interpolation data is available.
  mutable bool weakdiv_sigma_ready_ = false;
  /// Sigma space pointer used to build interpolation data.
  mutable const FiniteElementSpace* weakdiv_sigma_fes_ = nullptr;
  /// Sigma FE-space sequence used to build interpolation data.
  mutable long weakdiv_sigma_fes_sequence_ = -1;

  /// Persistent per-face-quadrature-point q_sigma aggregates.
  mutable Array<double> face_int_q_sigma_telemetry_;
  mutable Array<double> face_bdr_q_sigma_telemetry_;
  mutable Array<double> face_shared_q_sigma_telemetry_;

  /// MFEM-native vector-mass form (serial).
  std::unique_ptr<BilinearForm> mass_form_;
#ifdef MFEM_USE_MPI
  /// MFEM-native vector-mass form (parallel).
  std::unique_ptr<ParBilinearForm> par_mass_form_;
#endif
  /// Jacobi preconditioner for native inverse-mass solve.
  std::unique_ptr<OperatorJacobiSmoother> mass_inv_prec_;
  /// Krylov solver for native inverse-mass solve.
  std::unique_ptr<CGSolver> mass_inv_solver_;
  /// Whether the native inverse-mass solver path is available.
  bool mass_inv_solver_ready_ = false;
  /// Ensures accelerated mass-inverse fallback warning is emitted once.
  mutable bool mass_inv_solver_fallback_warned_ = false;
};

} // namespace mfem

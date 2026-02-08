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
#include "mfem.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/riemann/fluxfunctions.hpp"

namespace mfem {
namespace dgsem { class DGSEMOperator; }

/// Choose which method to compute the *volume* contribution:
///  - WeakDivergence: (F(u), ∇v)_T via preassembled weak divergence or matrix-free
///  - DGSEMFluxDiff : flux-difference (“pairwise flux”) volume term in DGSEM style
enum class VolumeMode { WeakDivergence, DGSEMFluxDiff };

/// Optional DGSEM stabilization hook selection (placeholder).
enum class DGSEMStabilization { None, SplitForm, Filter };

/// DGSEM volume flux selection for entropy-conservative variants (Euler-only).
enum class DGSEMVolumeFlux { Central, Chandrashekar, Ranocha };

/// DGSEM volume dissipation selection (added to flux-differencing pair flux).
enum class DGSEMVolumeDissipation { None, LLF, EntropyVarsLF };

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
class DGHyperbolicConservationLaws : public TimeDependentOperator
{
public:
   /**
    * @brief Construct a DG operator with a given flux integrator.
    *
    * @param vfes_                     Vector finite element space; tested for DG [P_p]^n.
    * @param formIntegrator_           Face/domain flux integrator (provides ComputeFlux + numerical flux).
    * @param preassembleWeakDivergence If true, preassemble local weak-divergence matrices.
    */
   DGHyperbolicConservationLaws(FiniteElementSpace &vfes_,
                                std::unique_ptr<HyperbolicFormIntegrator> formIntegrator_,
                                bool preassembleWeakDivergence = true);

   DGHyperbolicConservationLaws(const DGHyperbolicConservationLaws&) = delete;
   DGHyperbolicConservationLaws& operator=(const DGHyperbolicConservationLaws&) = delete;
   DGHyperbolicConservationLaws(DGHyperbolicConservationLaws&&) = default;
   DGHyperbolicConservationLaws& operator=(DGHyperbolicConservationLaws&&) = default;

   /// Select volume-contribution strategy (WeakDivergence or DGSEMFluxDiff).
   void SetVolumeMode(VolumeMode m) { vol_mode_ = m; }

   /// Get the current volume-contribution strategy.
   VolumeMode GetVolumeMode() const { return vol_mode_; }

   /// Select optional DGSEM stabilization hook (not implemented yet).
   void SetDGSEMStabilization(DGSEMStabilization s) { dgsem_stab_ = s; }

   /// Get current DGSEM stabilization hook selection.
   DGSEMStabilization GetDGSEMStabilization() const { return dgsem_stab_; }

   /// Set the numerical flux used for DGSEM surface evaluation (non-owning).
   void SetNumericalFlux(const NumericalFlux *flux) { num_flux_ = flux; }

   /// Select DGSEM volume flux (central or entropy-conservative).
   void SetDGSEMVolumeFlux(DGSEMVolumeFlux f) { dgsem_vol_flux_ = f; }

   /// Get current DGSEM volume flux.
   DGSEMVolumeFlux GetDGSEMVolumeFlux() const { return dgsem_vol_flux_; }

   /// Set Euler gamma used by entropy-conservative DGSEM fluxes.
   void SetDGSEMGamma(double gamma) { dgsem_gamma_ = gamma; }

   /// Get Euler gamma used by DGSEM entropy-conservative fluxes.
   double GetDGSEMGamma() const { return dgsem_gamma_; }

   /// Select DGSEM volume dissipation mode.
   void SetDGSEMVolumeDissipation(DGSEMVolumeDissipation d)
   { dgsem_vol_diss_ = d; }

   /// Get DGSEM volume dissipation mode.
   DGSEMVolumeDissipation GetDGSEMVolumeDissipation() const
   { return dgsem_vol_diss_; }

   /// Select entropy Jacobian mode for entropy-variable LF volume dissipation.
   void SetDGSEMEntropyJacobianAnalytic(bool analytic)
   { dgsem_entropy_jacobian_analytic_ = analytic; }

   /// Query entropy Jacobian mode for entropy-variable LF volume dissipation.
   bool GetDGSEMEntropyJacobianAnalytic() const
   { return dgsem_entropy_jacobian_analytic_; }

   /**
   * @brief Set a custom face integration rule for surface flux evaluation.
   *
   * Applies the rule to the interior-face integrator and caches it so boundary
   * integrators added later inherit the same rule.
   */
   void SetFaceIntegrationRule(const IntegrationRule &ir)
   {
      face_int_rule_owned_ = std::make_unique<IntegrationRule>(ir);
      face_int_rule_ = face_int_rule_owned_.get();
      formIntegrator->SetIntegrationRule(*face_int_rule_);
   }
   
   /**
    * @brief Apply the operator: y = M⁻¹( -<F̂, [[v]]> + (F(u), ∇v) ).
    *
    * If WeakDivergence mode is active and weak divergence is preassembled,
    * the local volume term is added by multiplying the “weakdiv” matrix by nodal fluxes.
    * If DGSEMFluxDiff mode is active, the DGSEM-style volume flux-difference is added.
    */
   void Mult(const Vector &x, Vector &y) const override;

   /// Recompute data if mesh/fe space changes (mass matrix, weak divergence, etc.).
   void Update();

   /// Max characteristic speed seen during the last Mult() (for CFL).
   real_t GetMaxCharSpeed() { return max_char_speed; }

   /**
   * @brief Add a boundary-face integrator with a marker (MFEM-style).
   *
   * Ownership semantics: the operator stores the integrator and a heap copy of the marker,
   * then registers them into the underlying NonlinearForm with UseExternalIntegrators().
   */
   void AddBdrFaceIntegrator(std::unique_ptr<NonlinearFormIntegrator> nfi,
                             const Array<int> &bdr_marker);

   /// Compatibility overload; ownership is transferred to the unique_ptr API.
   void AddBdrFaceIntegrator(NonlinearFormIntegrator *nfi,
                             const Array<int> &bdr_marker)
   {
      AddBdrFaceIntegrator(std::unique_ptr<NonlinearFormIntegrator>(nfi),
                           bdr_marker);
   }

   ~DGHyperbolicConservationLaws() override = default;

private:
   friend class dgsem::DGSEMOperator;

   // --- Problem description and finite element spaces ---
   const int num_equations;          ///< number of conservative equations
   const int dim;                    ///< space dimension
   FiniteElementSpace &vfes;         ///< vector FE space (vdim = num_equations)

   // --- Nonlinear assembly / integrators ---
   std::unique_ptr<HyperbolicFormIntegrator> formIntegrator;
   std::unique_ptr<NonlinearForm> nonlinearForm;

   // --- Local per-element operators ---
   std::vector<DenseMatrix> invmass; ///< inverse local mass matrices (dof × dof)
   std::vector<DenseMatrix> weakdiv; ///< local weak-divergence (dof × (dof*dim); columns grouped by node then dim)

   // --- Runtime state ---
   mutable real_t max_char_speed = 0.0; ///< updated during Mult() by the form integrator
   mutable Vector z;                    ///< scratch for the assembled RHS before M⁻¹

   // Keep external boundary integrators alive + stable copies of their markers.
   std::vector<std::unique_ptr<NonlinearFormIntegrator>> ext_bdr_int_;
   std::vector<std::unique_ptr<Array<int>>> ext_bdr_markers_;
   std::unique_ptr<IntegrationRule> face_int_rule_owned_;
   const IntegrationRule *face_int_rule_ = nullptr;

   // Volume strategy
   VolumeMode vol_mode_ = VolumeMode::WeakDivergence;
   const NumericalFlux *num_flux_ = nullptr;
   DGSEMStabilization dgsem_stab_ = DGSEMStabilization::None;
   DGSEMVolumeFlux dgsem_vol_flux_ = DGSEMVolumeFlux::Central;
   DGSEMVolumeDissipation dgsem_vol_diss_ = DGSEMVolumeDissipation::None;
   double dgsem_gamma_ = 1.4;
   bool dgsem_entropy_jacobian_analytic_ = false;

   // --- Helpers ---
   /// Compute per-element inverse mass matrices (block-diagonal DG mass).
   void ComputeInvMass();

   /// Preassemble the weak-divergence operator on each element (optional).
   /// The layout is (test dof) × (trial node × dim), i.e. ByDim grouped per node.
   void ComputeWeakDivergence();
};

} // namespace mfem

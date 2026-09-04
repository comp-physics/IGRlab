#pragma once
#include "mfem.hpp"
#include "utils/config/igr_config.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <string>

/**
 * @file physics/igr/IGR.hpp
 * @brief Information Geometric Regularization (IGR) solver for sigma coupled to Euler.
 * @ingroup physics_igr
 *
 * Given the current conservative state U = [ρ, ρu, E], this class computes Σ by
 * solving the scalar elliptic PDE
 *
 *   (1/ρ) Σ - α ∇·( (1/ρ) ∇Σ )
 *     = α ( tr((∇u)^2) + (∇·u)^2 )
 *
 * Discretization:
 *  - Mass term: ( (1/ρ) Σ, v )
 *  - Diffusion term: ( (α/ρ) ∇Σ, ∇v ) [+ IPDG face terms if Σ-space is DG]
 *  - RHS: ( α( tr((∇u)^2) + (∇·u)^2 ), v )
 *
 * Solve options:
 *  - Legacy: a few preconditioned Richardson smoother sweeps (fast/approx)
 *  - Recommended: MFEM Krylov solve to tolerance (CG/GMRES + preconditioner)
 */
namespace dg_igr
{

enum class IGRSolvePurpose
{
  ProductionStage,
  ProductionStage0Prefetch,
  OutputOnly
};

/// @brief Information Geometric Regularization (IGR) solver for sigma.
/// @ingroup physics_igr
class IGRSolver
{
public:
  /// Type aliases for stable enum/config types.
  using SmootherType = IGRSmootherType;
  using GradBackend = IGRGradBackend;
  using IPKind = IGRIPKind;
  using SolveMode = IGRSolveMode;
  using LinearSolverType = IGRLinearSolverType;
  using PrecondType = IGRPrecondType;
  using NonconvergencePolicy = IGRNonconvergencePolicy;
  using AlphaLengthScalePolicy = IGRAlphaLengthScalePolicy;
  using Config = IGRConfig;

  /**
   * @brief Construct an IGR solver.
   * @param sigma_fes   Scalar FE space for Σ (H1 or DG).
   * @param state_fes   Vector FE space for conservative state U (vdim=num_eq).
   * @param dim         Spatial dimension (2 or 3).
   * @param num_eq      Number of conservative variables (e.g., 4 in 2D, 5 in 3D).
   * @param alpha       Regularization parameter α.
   */
  IGRSolver(mfem::ParFiniteElementSpace& sigma_fes,
            mfem::ParFiniteElementSpace& state_fes,
            int dim,
            int num_eq,
            double alpha);

  /// Set regularization parameter α.
  void SetAlpha(double a);
  void SetAlphaLengthScalePolicy(AlphaLengthScalePolicy policy);

  /// Clamp for rho to avoid division by very small numbers (default IGRDefaults::kRhoFloor).
  void SetRhoFloor(double rho_floor);

  /// Set essential (Dirichlet) boundary attributes for Σ; default is none (natural Neumann).
  void SetDirichletBC(const mfem::Array<int>& ess_bdr);

  /// Select which gradient backend to use for the Σ RHS.
  void SetGradBackend(GradBackend b) { backend_ = b; }
  void SetIPKind(IPKind k);
  void SetExactDGFaceOperator(bool enabled);

  // ---- Legacy smoother-sweep knobs (still supported) ----
  void SetSmoother(SmootherType t);
  void SetSweeps(int regular_sweeps, int warm_start_sweeps);

  // ---- Recommended Krylov-solve knobs ----
  void SetSolveMode(SolveMode m) { solve_mode_ = m; }
  SolveMode GetSolveMode() const { return solve_mode_; }
  void SetLinearSolver(LinearSolverType s) { lin_solver_type_ = s; }
  void SetPreconditioner(PrecondType p);
  void SetNonconvergencePolicy(NonconvergencePolicy policy)
  {
    nonconvergence_policy_ = policy;
  }
  void SetAMGParams(int coarsen_type,
                    int relax_type,
                    int max_levels,
                    double strong_threshold)
  {
    amg_coarsen_type_ = coarsen_type;
    amg_relax_type_ = relax_type;
    amg_max_levels_ = max_levels;
    amg_strong_threshold_ = strong_threshold;
  }
  void SetSigmaUpdatePolicy(IGRSigmaUpdateCadence cadence, int interval)
  {
    sigma_cadence_ = cadence;
    sigma_update_interval_ = std::max(1, interval);
  }
  void SetStageRtolScale(double scale)
  {
    stage_rtol_scale_ = std::max(1.0, scale);
  }
  /// Legacy three-stage context retained for direct callers and SSPRK(3,3).
  void SetRKStageIndex(int stage)
  {
    rk_stage_index_ = stage;
    rk_stage_count_ = stage >= 0 ? 3 : -1;
  }
  void SetRKStageContext(int stage, int stage_count)
  {
    MFEM_VERIFY(stage >= 0 && stage_count > 0 && stage < stage_count,
                "Invalid explicit RK stage context.");
    rk_stage_index_ = stage;
    rk_stage_count_ = stage_count;
  }
  void SetRKStepIndex(int step) { rk_step_index_ = step; }
  void SetCurrentTime(double time) { current_time_ = time; }
  void SetCurrentTimeStep(double dt) { current_dt_ = dt; }
  IGRSigmaUpdateCadence GetSigmaUpdateCadence() const
  {
    return sigma_cadence_;
  }
  int GetSigmaUpdateInterval() const { return sigma_update_interval_; }

  /// Set solve tolerances and iteration cap (used in Krylov mode).
  void SetLinearSolveParams(double rel_tol,
                            double abs_tol,
                            int max_it,
                            int print_level = 0)
  {
    lin_rtol_ = rel_tol;
    lin_atol_ = abs_tol;
    lin_maxit_ = max_it;
    lin_print_ = print_level;
  }

  /// Apply a configuration bundle without changing existing defaults.
  void ApplyConfig(const Config& cfg)
  {
    SetAlpha(cfg.alpha);
    SetAlphaLengthScalePolicy(cfg.alpha_length_scale_policy);
    SetRhoFloor(cfg.rho_floor);
    SetGradBackend(cfg.backend);
    SetIPKind(cfg.ip_kind);

    SetSmoother(cfg.smoot);
    SetSweeps(cfg.regular_sweeps, cfg.warm_start_sweeps);

    SetSolveMode(cfg.solve_mode);
    SetLinearSolver(cfg.lin_solver_type);
    SetPreconditioner(cfg.prec_type);
    SetNonconvergencePolicy(cfg.nonconvergence_policy);
    SetAMGParams(cfg.amg_coarsen_type,
                 cfg.amg_relax_type,
                 cfg.amg_max_levels,
                 cfg.amg_strong_threshold);
    SetSigmaUpdatePolicy(cfg.sigma_cadence, cfg.sigma_update_interval);
    SetStageRtolScale(cfg.stage_rtol_scale);
    SetExactDGFaceOperator(cfg.exact_dg_face_operator);
    SetLinearSolveParams(cfg.lin_rtol,
                         cfg.lin_atol,
                         cfg.lin_maxit,
                         cfg.lin_print);
  }

  /// Update Σ from the current state U.
  void UpdateFromState(
    const mfem::ParGridFunction& U,
    bool warm_start,
    IGRSolvePurpose purpose = IGRSolvePurpose::ProductionStage,
    bool certify_residual = false);
  void ConsumePendingStage0PrefetchAccounting();
  bool ReclassifyPendingStage0PrefetchAsOutputOnly(
    bool restore_warm_start = true);
  bool HasPendingStage0PrefetchAccounting() const
  {
    return pending_stage0_prefetch_accounting_.valid;
  }

  /// Access the current Σ field.
  const mfem::ParGridFunction& GetSigma() const { return sigma_; }
  mfem::ParGridFunction& GetSigma() { return sigma_; }

  /// Last linear solve diagnostics (updated by Krylov solve path).
  int GetLastLinearIters() const { return last_linear_iters_; }
  bool GetLastLinearConverged() const { return last_linear_converged_; }
  double GetLastLinearFinalNorm() const { return last_linear_final_norm_; }
  double GetLastLinearInitialNorm() const { return last_linear_initial_norm_; }
  double GetLastLinearTrueResidualAbs() const
  {
    return last_linear_true_residual_abs_;
  }
  double GetLastLinearTrueResidualRelRHS() const
  {
    return last_linear_true_residual_rel_rhs_;
  }
  double GetLastIndependentCertificateRelativeLimit() const
  {
    return last_independent_certificate_relative_limit_;
  }
  double GetLastIndependentCertificateAbsoluteLimit() const
  {
    return last_independent_certificate_absolute_limit_;
  }
  bool GetLastWarmStartEffective() const
  {
    return last_warm_start_effective_;
  }
  bool GetLastWarmStartGuardRejected() const
  {
    return last_warm_start_guard_rejected_;
  }
  double GetLastWarmStartResidualNorm() const
  {
    return last_warm_start_residual_norm_;
  }
  double GetLastZeroStartResidualNorm() const
  {
    return last_zero_start_residual_norm_;
  }
  double GetLastLinearTrueResidualRel() const
  {
    return last_linear_true_residual_rel_initial_;
  }
  double GetLastLinearTrueInitialResidualAbs() const
  {
    return last_linear_true_initial_residual_abs_;
  }
  double GetLastLinearRecomputedPreconditionedInitialNorm() const
  {
    return last_linear_recomputed_preconditioned_initial_norm_;
  }
  double GetLastLinearRecomputedPreconditionedFinalNorm() const
  {
    return last_linear_recomputed_preconditioned_final_norm_;
  }
  double GetLastLinearSolveSeconds() const
  {
    return last_linear_solve_seconds_;
  }
  double GetLastAssembleSeconds() const { return last_assemble_seconds_; }
  double GetLastFormSystemSeconds() const { return last_form_system_seconds_; }
  double GetLastPCSetupSeconds() const { return last_pc_setup_seconds_; }
  double GetLastChebyshevDiagonalSeconds() const
  {
    return last_chebyshev_diagonal_seconds_;
  }
  double GetLastChebyshevSetupSeconds() const
  {
    return last_chebyshev_setup_seconds_;
  }
  bool GetLastChebyshevEigenvalueReused() const
  {
    return last_chebyshev_eigenvalue_reused_;
  }
  bool GetLastChebyshevSmootherReused() const
  {
    return last_chebyshev_smoother_reused_;
  }
  double GetLastChebyshevEigenvalueEstimate() const
  {
    return last_chebyshev_eigenvalue_estimate_;
  }
  double GetLastKrylovOperatorApplySeconds() const
  {
    return last_krylov_operator_apply_seconds_;
  }
  double GetLastKrylovPreconditionerApplySeconds() const
  {
    return last_krylov_preconditioner_apply_seconds_;
  }
  double GetLastKrylovPreconditionerOperatorApplySeconds() const
  {
    return last_krylov_preconditioner_operator_apply_seconds_;
  }
  long long GetLastKrylovOperatorApplyCount() const
  {
    return last_krylov_operator_apply_count_;
  }
  long long GetLastKrylovPreconditionerApplyCount() const
  {
    return last_krylov_preconditioner_apply_count_;
  }
  long long GetLastKrylovPreconditionerOperatorApplyCount() const
  {
    return last_krylov_preconditioner_operator_apply_count_;
  }
  double GetTotalLinearSolveSeconds() const
  {
    return total_linear_solve_seconds_;
  }
  double GetTotalAssembleSeconds() const { return total_assemble_seconds_; }
  double GetTotalFormSystemSeconds() const
  {
    return total_form_system_seconds_;
  }
  double GetTotalPCSetupSeconds() const { return total_pc_setup_seconds_; }
  double GetTotalChebyshevDiagonalSeconds() const
  {
    return total_chebyshev_diagonal_seconds_;
  }
  double GetTotalChebyshevSetupSeconds() const
  {
    return total_chebyshev_setup_seconds_;
  }
  int GetTotalChebyshevEigenvalueReuses() const
  {
    return total_chebyshev_eigenvalue_reuses_;
  }
  int GetTotalChebyshevSmootherReuses() const
  {
    return total_chebyshev_smoother_reuses_;
  }
  int GetTotalChebyshevSmootherBuilds() const
  {
    return total_chebyshev_smoother_builds_;
  }
  double GetTotalKrylovOperatorApplySeconds() const
  {
    return total_krylov_operator_apply_seconds_;
  }
  double GetTotalKrylovPreconditionerApplySeconds() const
  {
    return total_krylov_preconditioner_apply_seconds_;
  }
  double GetTotalKrylovPreconditionerOperatorApplySeconds() const
  {
    return total_krylov_preconditioner_operator_apply_seconds_;
  }
  long long GetTotalKrylovOperatorApplyCount() const
  {
    return total_krylov_operator_apply_count_;
  }
  long long GetTotalKrylovPreconditionerApplyCount() const
  {
    return total_krylov_preconditioner_apply_count_;
  }
  long long GetTotalKrylovPreconditionerOperatorApplyCount() const
  {
    return total_krylov_preconditioner_operator_apply_count_;
  }
  double GetTotalIGRWorkSeconds() const
  {
    return total_assemble_seconds_ + total_form_system_seconds_ +
           total_pc_setup_seconds_ + total_linear_solve_seconds_;
  }
  double GetLastUpdateCallSeconds() const { return last_update_call_seconds_; }
  double GetTotalUpdateCallSeconds() const { return total_update_call_seconds_; }
  int GetLinearSolveCalls() const { return linear_solve_calls_; }
  int GetTotalLinearIterations() const { return total_linear_iters_; }
  int GetMaxLinearIterations() const { return max_linear_iters_; }
  int GetTotalLinearConvergedCalls() const
  {
    return total_linear_converged_calls_;
  }
  double GetAverageLinearIterations() const
  {
    return (linear_solve_calls_ > 0)
             ? static_cast<double>(total_linear_iters_) /
                 static_cast<double>(linear_solve_calls_)
             : 0.0;
  }
  int GetOutputOnlyLinearSolveCalls() const
  {
    return output_only_linear_solve_calls_;
  }
  int GetProductionLinearSolveCalls() const
  {
    return linear_solve_calls_ - output_only_linear_solve_calls_;
  }
  int GetProductionLinearIterations() const
  {
    return total_linear_iters_ - output_only_linear_iters_;
  }
  int GetProductionLinearConvergedCalls() const
  {
    return total_linear_converged_calls_ -
           output_only_linear_converged_calls_;
  }
  double GetProductionLinearSolveSeconds() const
  {
    return total_linear_solve_seconds_ - output_only_linear_solve_seconds_;
  }
  double GetProductionAssembleSeconds() const
  {
    return total_assemble_seconds_ - output_only_assemble_seconds_;
  }
  double GetProductionFormSystemSeconds() const
  {
    return total_form_system_seconds_ - output_only_form_system_seconds_;
  }
  double GetProductionPCSetupSeconds() const
  {
    return total_pc_setup_seconds_ - output_only_pc_setup_seconds_;
  }
  double GetProductionIGRWorkSeconds() const
  {
    return GetProductionAssembleSeconds() + GetProductionFormSystemSeconds() +
           GetProductionPCSetupSeconds() + GetProductionLinearSolveSeconds();
  }
  double GetAverageProductionLinearIterations() const
  {
    const int calls = GetProductionLinearSolveCalls();
    return calls > 0 ? static_cast<double>(GetProductionLinearIterations()) /
                         static_cast<double>(calls)
                     : 0.0;
  }
  int GetOutputOnlyLinearIterations() const
  {
    return output_only_linear_iters_;
  }
  int GetOutputOnlyLinearConvergedCalls() const
  {
    return output_only_linear_converged_calls_;
  }
  double GetOutputOnlyLinearSolveSeconds() const
  {
    return output_only_linear_solve_seconds_;
  }
  double GetOutputOnlyIGRWorkSeconds() const
  {
    return output_only_assemble_seconds_ + output_only_form_system_seconds_ +
           output_only_pc_setup_seconds_ + output_only_linear_solve_seconds_;
  }
  const std::string& GetLastLinearSolverName() const
  {
    return last_linear_solver_name_;
  }
  const std::string& GetLastPrecondName() const { return last_precond_name_; }
  /// Access the currently assembled linear operator (for diagnostics/tests).
  const mfem::HypreParMatrix* GetAssembledOperator() const
  {
    return Ah_.Is<mfem::HypreParMatrix>();
  }
  /// Access the currently formed linear operator (for diagnostics/tests).
  const mfem::Operator* GetLinearOperator() const
  {
    return ActiveLinearOperator_();
  }
  /// Access exact-face operator components when active (diagnostics/tests).
  const mfem::Operator* GetLinearBaseOperator() const;
  const mfem::Operator* GetLinearFaceOperator() const;

  /// Access the currently formed linear RHS (for diagnostics/tests).
  const mfem::Vector& GetLinearRHS() const { return B_; }
  int GetLastExactFaceLocalFaceCount() const
  {
    return last_exact_face_local_nf_;
  }
  int GetLastExactFaceSharedFaceCount() const
  {
    return last_exact_face_shared_nf_;
  }
  bool GetLastExactFaceRequested() const
  {
    return last_exact_face_requested_;
  }
  bool GetLastExactFaceEffective() const
  {
    return last_exact_face_effective_;
  }
  const std::string& GetLastExactFaceReason() const
  {
    return last_exact_face_reason_;
  }
  long long GetExactFaceOperatorUseCount() const
  {
    return exact_face_operator_uses_;
  }
  long long GetLastExactFaceOperatorMultCalls() const
  {
    return last_exact_face_operator_mult_calls_;
  }
  bool GetLastDirectIdentitySystem() const
  {
    return direct_identity_system_;
  }
  int GetDirectIdentitySystemUses() const
  {
    return direct_identity_system_uses_;
  }
  const std::string& GetLastDirectIdentityReason() const
  {
    return last_direct_identity_reason_;
  }

  /// Spatial dimension.
  int Dim() const { return dim_; }

  /// Number of equations (vdim of state U).
  int NumEquations() const { return num_eq_; }

private:
  struct SolveAccountingDelta
  {
    bool valid = false;
    int calls = 0;
    int iterations = 0;
    int converged_calls = 0;
    double solve_seconds = 0.0;
    double assemble_seconds = 0.0;
    double form_system_seconds = 0.0;
    double pc_setup_seconds = 0.0;
  };

  void AddOutputOnlyAccounting_(const SolveAccountingDelta& delta);

  // --- QuadratureFunction coefficient path for PA/device operators. ---
  // On CUDA this is the default host-sync-free path. On CPU it is explicit
  // opt-in for operator-equivalence diagnostics and PA reference runs.
  bool UseDeviceCoefficients_() const;
  void EnsureDeviceQuadrature_();
  void EnsureDeviceElementAlpha_();
  void UpdateDeviceCoefficients_(const mfem::ParGridFunction& U);

  // ---------- Coefficients that depend on the current state U ----------
  /// c_M(x) = 1 / max(ρ, rho_floor_)
  class InvRhoCoefficient : public mfem::Coefficient
  {
  public:
    InvRhoCoefficient(const mfem::ParGridFunction& U, double rho_floor);
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override;
    void SetState(const mfem::ParGridFunction& U) { U_ = &U; }
    void SetRhoFloor(double floor) { rho_floor_ = floor; }

  private:
    const mfem::ParGridFunction* U_ = nullptr; // non-owning; owned by caller
    double rho_floor_;
  };

  /// Vector coefficient u = m/ρ assembled from conservative U.
  class VelocityFromConsCoefficient : public mfem::VectorCoefficient
  {
  public:
    VelocityFromConsCoefficient(int dim,
                                const mfem::ParGridFunction* U,
                                double rho_floor);
    void SetState(const mfem::ParGridFunction* U) { U_ = U; }
    void SetRhoFloor(double f) { rho_floor_ = f; }
    void Eval(mfem::Vector& V,
              mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override;

  private:
    const mfem::ParGridFunction* U_ = nullptr; // non-owning; owned by caller
    double rho_floor_;
  };

  /// f(x) = α ( ||∇u||_F^2 + (∇·u)^2 ), with u = m / ρ (componentwise).
  class SigmaSourceCoefficient : public mfem::Coefficient
  {
  public:
    SigmaSourceCoefficient(const mfem::ParGridFunction& U,
                           int dim,
                           int num_eq,
                           double alpha,
                           double rho_floor);
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override;
    void SetState(const mfem::ParGridFunction& U) { U_ = &U; }
    void SetAlpha(double a) { alpha_ = a; }
    void SetAlphaCoefficient(mfem::Coefficient* coeff) { alpha_coeff_ = coeff; }
    void SetRhoFloor(double floor) { rho_floor_ = floor; }
    // Optional continuous velocity field (vdim=dim); if set, its gradient is used.
    void SetVelField(const mfem::ParGridFunction* v) { velH1_ = v; }

  private:
    const mfem::ParGridFunction* U_ = nullptr; // non-owning; owned by caller
    const mfem::ParGridFunction* velH1_ =
      nullptr; // non-owning optional continuous velocity
    int dim_, num_eq_;
    double alpha_, rho_floor_;
    mfem::Coefficient* alpha_coeff_ = nullptr; // non-owning
    mfem::Vector state_;      // [rho, m(0:dim-1), E]
    mfem::DenseMatrix gradU_; // num_eq_ x dim_
    mfem::DenseMatrix gradu_; // dim_ x dim_
  };

  // Device/QF coefficient cache.
  bool using_device_coeffs_ = false;
  bool device_coeff_logged_ = false;
  long device_mesh_sequence_ = -1;
  long device_sfes_sequence_ = -1;
  long device_vfes_sequence_ = -1;
  int device_qorder_ = 0;
  int device_mass_qorder_ = 0;
  int device_diff_qorder_ = 0;
  int device_rhs_qorder_ = 0;
  int device_face_qorder_ = 0;

  std::unique_ptr<mfem::QuadratureSpace> mass_qs_;
  std::unique_ptr<mfem::QuadratureSpace> sigma_qs_;
  std::unique_ptr<mfem::QuadratureSpace> rhs_qs_;
  std::unique_ptr<mfem::FaceQuadratureSpace> sigma_fqs_;
  std::unique_ptr<mfem::QuadratureFunction> u_mass_qf_;
  std::unique_ptr<mfem::QuadratureFunction> u_qf_;
  std::unique_ptr<mfem::QuadratureFunction> u_rhs_qf_;
  std::unique_ptr<mfem::QuadratureFunction> u_face_qf_;
  std::unique_ptr<mfem::QuadratureFunction> grad_u_qf_;
  std::unique_ptr<mfem::QuadratureFunction> grad_u_rhs_qf_;
  std::unique_ptr<mfem::QuadratureFunction> invrho_qf_;
  std::unique_ptr<mfem::QuadratureFunction> ck_qf_;
  std::unique_ptr<mfem::QuadratureFunction> src_qf_;
  std::unique_ptr<mfem::QuadratureFunction> ck_face_qf_;
  std::unique_ptr<mfem::QuadratureFunctionCoefficient> invrho_qf_coef_;
  std::unique_ptr<mfem::QuadratureFunctionCoefficient> ck_qf_coef_;
  std::unique_ptr<mfem::QuadratureFunctionCoefficient> src_qf_coef_;
  std::unique_ptr<mfem::QuadratureFunctionCoefficient> ck_face_qf_coef_;
  mfem::Vector device_qinterp_evec_;
  mfem::Vector device_face_restr_evec_;
  // For the local length policy, alpha_ stores C_alpha while this cache stores
  // the physical per-element alpha_e = C_alpha h_e^2.  It is reused until the
  // mesh sequence or C_alpha changes.
  mfem::Vector device_element_alpha_;
  long device_element_alpha_mesh_sequence_ = -1;
  double device_element_alpha_c_alpha_ =
    std::numeric_limits<double>::quiet_NaN();

  /// α(x) as a coefficient derived from the selected length-scale policy.
  class AlphaPolicyCoefficient : public mfem::Coefficient
  {
  public:
    AlphaPolicyCoefficient(const mfem::ParMesh& mesh,
                           const double& a_ref,
                           const AlphaLengthScalePolicy& policy_ref)
      : mesh_(mesh),
        a_ref_(a_ref),
        policy_ref_(policy_ref)
    {
    }
    double Eval(mfem::ElementTransformation&,
                const mfem::IntegrationPoint&) override
    ;

  private:
    const mfem::ParMesh& mesh_;
    const double& a_ref_;
    const AlphaLengthScalePolicy& policy_ref_;
  };

  // ---------- Core assemble/apply helpers ----------
  void
  AssembleSystem(const mfem::ParGridFunction& U); // fill A and b for current U
  void DoSmootherSweeps(int sweeps, bool warm_start); // legacy
  void SolveEllipticSystem_(bool warm_start);         // new recommended path
  void EnsureH1VelocitySpace_();
  void BuildBilinearForm_();
  bool ShouldUsePartialAssembly_() const;
  bool UseExactDGFaceOperator_() const;
  bool ShouldUseDirectIdentitySystem_(bool use_device_qf,
                                      std::string& reason) const;
  mfem::Operator* ActiveLinearOperator_() const;

  // ---------- Data ----------
  mfem::ParFiniteElementSpace& sfes_; // scalar Σ space (H1 or DG)
  mfem::ParFiniteElementSpace& vfes_; // vector state U space (vdim=num_eq)
  int dim_, num_eq_;
  double alpha_ = mfem::dg_igr::IGRDefaults::kAlpha;
  double rho_floor_ = mfem::dg_igr::IGRDefaults::kRhoFloor;
  AlphaLengthScalePolicy alpha_length_scale_policy_ =
    AlphaLengthScalePolicy::Average;
  SmootherType smoot_ = SmootherType::L1Jacobi;
  GradBackend backend_ = GradBackend::H1Projection;
  IPKind ip_kind_ = IPKind::SIPG; //default SPD

  // Legacy sweeps
  int regular_sweeps_ = mfem::dg_igr::IGRDefaults::kRegularSweeps;
  int warm_start_sweeps_ = mfem::dg_igr::IGRDefaults::kWarmStartSweeps;
  bool first_call_ = true;

  // New solve controls
  SolveMode solve_mode_ = SolveMode::Krylov;
  LinearSolverType lin_solver_type_ = LinearSolverType::Auto;
  PrecondType prec_type_ = PrecondType::Auto;
  NonconvergencePolicy nonconvergence_policy_ =
    NonconvergencePolicy::Accept;
  int amg_coarsen_type_ = -1;
  int amg_relax_type_ = -1;
  int amg_max_levels_ = -1;
  double amg_strong_threshold_ = -1.0;
  IGRSigmaUpdateCadence sigma_cadence_ = IGRSigmaUpdateCadence::EveryStage;
  int sigma_update_interval_ = 1;
  double stage_rtol_scale_ = 1.0;
  int rk_stage_index_ = -1;
  int rk_stage_count_ = -1;
  int rk_step_index_ = -1;
  double current_time_ = std::numeric_limits<double>::quiet_NaN();
  double current_dt_ = std::numeric_limits<double>::quiet_NaN();

  double lin_rtol_ = mfem::dg_igr::IGRDefaults::kLinRtol;
  double lin_atol_ = mfem::dg_igr::IGRDefaults::kLinAtol;
  int lin_maxit_ = mfem::dg_igr::IGRDefaults::kLinMaxIters;
  int lin_print_ = mfem::dg_igr::IGRDefaults::kLinPrint;

  // Σ field
  mfem::ParGridFunction
    state_scratch_; // instance-owned scratch used for coeff bootstrap
  const mfem::ParGridFunction* diagnostic_state_ = nullptr;
  mfem::ParGridFunction sigma_;

  // FE forms (reused each call; values change, sparsity fixed)
  std::unique_ptr<mfem::ParBilinearForm> a_;
  std::unique_ptr<mfem::ParLinearForm> b_;
  bool use_partial_assembly_ = false;

  // Coefficients that refer to U
  std::unique_ptr<InvRhoCoefficient> invrho_coef_;
  std::unique_ptr<SigmaSourceCoefficient> src_coef_;
  std::unique_ptr<AlphaPolicyCoefficient> alpha_coef_; // α(x) from policy
  std::unique_ptr<mfem::ProductCoefficient> ck_coef_; // ck(x) = α/ρ

  // Linear system objects
  mfem::Array<int> ess_bdr_;
  mfem::Array<int> ess_tdof_list_;
  mfem::OperatorHandle Ah_; // holds the assembled operator (Hypre matrix)
  // Owns the current stage's exact-face operator while Ah_ owns the sum.
  std::unique_ptr<mfem::Operator> exact_face_operator_;
  mfem::Vector X_, B_;      // true-dof vectors

  // H1 projection backend storage
  std::unique_ptr<mfem::H1_FECollection> h1_vec_fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> h1_vec_fes_; // vdim = dim
  std::unique_ptr<mfem::ParGridFunction> vel_h1_; // projected velocity

  // Last Krylov solve telemetry.
  int last_linear_iters_ = 0;
  bool last_linear_converged_ = false;
  double last_linear_final_norm_ = std::numeric_limits<double>::quiet_NaN();
  double last_linear_initial_norm_ = std::numeric_limits<double>::quiet_NaN();
  double last_linear_true_residual_abs_ =
    std::numeric_limits<double>::quiet_NaN();
  double last_linear_true_residual_rel_rhs_ =
    std::numeric_limits<double>::quiet_NaN();
  double last_independent_certificate_relative_limit_ =
    std::numeric_limits<double>::quiet_NaN();
  double last_independent_certificate_absolute_limit_ =
    std::numeric_limits<double>::quiet_NaN();
  double last_linear_true_initial_residual_abs_ =
    std::numeric_limits<double>::quiet_NaN();
  double last_linear_true_residual_rel_initial_ =
    std::numeric_limits<double>::quiet_NaN();
  double last_linear_recomputed_preconditioned_initial_norm_ =
    std::numeric_limits<double>::quiet_NaN();
  double last_linear_recomputed_preconditioned_final_norm_ =
    std::numeric_limits<double>::quiet_NaN();
  bool last_warm_start_effective_ = false;
  bool last_warm_start_guard_rejected_ = false;
  double last_warm_start_residual_norm_ =
    std::numeric_limits<double>::quiet_NaN();
  double last_zero_start_residual_norm_ =
    std::numeric_limits<double>::quiet_NaN();
  double last_linear_solve_seconds_ = 0.0;
  double last_assemble_seconds_ = 0.0;
  double last_form_system_seconds_ = 0.0;
  double last_pc_setup_seconds_ = 0.0;
  double last_chebyshev_diagonal_seconds_ = 0.0;
  double last_chebyshev_setup_seconds_ = 0.0;
  bool last_chebyshev_eigenvalue_reused_ = false;
  bool last_chebyshev_smoother_reused_ = false;
  double last_chebyshev_eigenvalue_estimate_ = 0.0;
  double last_krylov_operator_apply_seconds_ = 0.0;
  double last_krylov_preconditioner_apply_seconds_ = 0.0;
  double last_krylov_preconditioner_operator_apply_seconds_ = 0.0;
  long long last_krylov_operator_apply_count_ = 0;
  long long last_krylov_preconditioner_apply_count_ = 0;
  long long last_krylov_preconditioner_operator_apply_count_ = 0;
  double total_linear_solve_seconds_ = 0.0;
  double total_assemble_seconds_ = 0.0;
  double total_form_system_seconds_ = 0.0;
  double total_pc_setup_seconds_ = 0.0;
  double total_chebyshev_diagonal_seconds_ = 0.0;
  double total_chebyshev_setup_seconds_ = 0.0;
  int total_chebyshev_eigenvalue_reuses_ = 0;
  int total_chebyshev_smoother_reuses_ = 0;
  int total_chebyshev_smoother_builds_ = 0;
  double total_krylov_operator_apply_seconds_ = 0.0;
  double total_krylov_preconditioner_apply_seconds_ = 0.0;
  double total_krylov_preconditioner_operator_apply_seconds_ = 0.0;
  long long total_krylov_operator_apply_count_ = 0;
  long long total_krylov_preconditioner_apply_count_ = 0;
  long long total_krylov_preconditioner_operator_apply_count_ = 0;
  double last_update_call_seconds_ = 0.0;
  double total_update_call_seconds_ = 0.0;
  int linear_solve_calls_ = 0;
  int total_linear_iters_ = 0;
  int max_linear_iters_ = 0;
  int total_linear_converged_calls_ = 0;
  IGRSolvePurpose current_solve_purpose_ = IGRSolvePurpose::ProductionStage;
  bool certify_current_solve_ = false;
  int output_only_linear_solve_calls_ = 0;
  int output_only_linear_iters_ = 0;
  int output_only_linear_converged_calls_ = 0;
  double output_only_linear_solve_seconds_ = 0.0;
  double output_only_assemble_seconds_ = 0.0;
  double output_only_form_system_seconds_ = 0.0;
  double output_only_pc_setup_seconds_ = 0.0;
  SolveAccountingDelta pending_stage0_prefetch_accounting_;
  mfem::Vector pending_stage0_prefetch_sigma_checkpoint_;
  bool pending_stage0_prefetch_first_call_ = true;
  bool pending_stage0_prefetch_last_converged_ = false;
  double pending_stage0_prefetch_last_final_norm_ =
    std::numeric_limits<double>::quiet_NaN();
  bool direct_identity_system_ = false;
  int direct_identity_system_uses_ = 0;
  std::string last_direct_identity_reason_ = "not-evaluated";
  bool exact_dg_face_operator_ = false;
  bool negative_quadratic_capture_written_ = false;
  bool linear_event_capture_written_ = false;
  bool last_exact_face_requested_ = false;
  bool last_exact_face_effective_ = false;
  std::string last_exact_face_reason_ = "not-requested";
  int last_exact_face_local_nf_ = 0;
  int last_exact_face_shared_nf_ = 0;
  long long exact_face_operator_uses_ = 0;
  long long last_exact_face_operator_mult_calls_ = 0;
  std::string last_linear_solver_name_ = "none";
  std::string last_precond_name_ = "none";
};

} // namespace dg_igr

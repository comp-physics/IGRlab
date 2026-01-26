#pragma once
#include "mfem.hpp"
#include "dg_igr_defaults.hpp"
#include <memory>

/**
 * @file IGR.hpp
 * @brief Information Geometric Regularization (IGR) solver for sigma coupled to Euler.
 * @ingroup physics_igr
 *
 * Given the current conservative state U = [ρ, ρu, E], this class computes Σ by
 * solving the scalar elliptic PDE
 *
 *   (1/ρ) Σ - α ∇·( (1/ρ) ∇Σ ) = α ( ||∇u||_F^2 + (∇·u)^2 )
 *
 * Discretization:
 *  - Mass term: ( (1/ρ) Σ, v )
 *  - Diffusion term: ( (α/ρ) ∇Σ, ∇v ) [+ IPDG face terms if Σ-space is DG]
 *  - RHS: ( α( ||∇u||_F^2 + (∇·u)^2 ), v )
 *
 * Solve options:
 *  - Legacy: a few preconditioned Richardson smoother sweeps (fast/approx)
 *  - Recommended: MFEM Krylov solve to tolerance (CG/GMRES + preconditioner)
 */
/// @brief Information Geometric Regularization (IGR) solver for sigma.
/// @ingroup physics_igr
class IGRSolver
{
public:
  /// Available smoother types used either for legacy sweeps or as a smoother-preconditioner.
  enum class SmootherType { Jacobi, L1Jacobi, L1GS, L1GStr, GS };

  /// Gradient construction backend for the RHS (∇u terms).
  enum class GradBackend { H1Projection, BrokenDG /*, WeakRT, BR2 */ };

  /// Interior-penalty DG flavor for the diffusion operator on DG spaces.
  /// Values map to DGDiffusionIntegrator's kappa parameter: -1 (SIPG), 0 (IIPG), +1 (NIPG).
  enum class IPKind { SIPG = -1, IIPG = 0, NIPG = +1 };

  /// How to solve A x = b each update.
  enum class SolveMode { SmootherSweeps, Krylov };

  /// Krylov solver choice.
  enum class LinearSolverType { Auto, CG, GMRES };

  /// Preconditioner choice.
  enum class PrecondType { Auto, BoomerAMG, HypreSmoother, None };

  /// Configuration bundle for IGR defaults and runtime tuning.
  struct Config
  {
    double alpha = mfem::dg_igr::IGRDefaults::kAlpha;
    double rho_floor = mfem::dg_igr::IGRDefaults::kRhoFloor;

    SmootherType smoot = SmootherType::L1Jacobi;
    GradBackend  backend = GradBackend::H1Projection;
    IPKind       ip_kind = IPKind::SIPG;

    int  regular_sweeps = mfem::dg_igr::IGRDefaults::kRegularSweeps;
    int  warm_start_sweeps = mfem::dg_igr::IGRDefaults::kWarmStartSweeps;

    SolveMode        solve_mode = SolveMode::Krylov;
    LinearSolverType lin_solver_type = LinearSolverType::Auto;
    PrecondType      prec_type = PrecondType::Auto;

    double lin_rtol = mfem::dg_igr::IGRDefaults::kLinRtol;
    double lin_atol = mfem::dg_igr::IGRDefaults::kLinAtol;
    int    lin_maxit = mfem::dg_igr::IGRDefaults::kLinMaxIters;
    int    lin_print = mfem::dg_igr::IGRDefaults::kLinPrint;
  };

  /**
   * @brief Construct an IGR solver.
   * @param sigma_fes   Scalar FE space for Σ (H1 or DG).
   * @param state_fes   Vector FE space for conservative state U (vdim=num_eq).
   * @param dim         Spatial dimension (2 or 3).
   * @param num_eq      Number of conservative variables (e.g., 4 in 2D, 5 in 3D).
   * @param alpha       Regularization parameter α.
   */
  IGRSolver(const mfem::ParFiniteElementSpace& sigma_fes,
            const mfem::ParFiniteElementSpace& state_fes,
            int dim, int num_eq, double alpha);

  /// Set regularization parameter α.
  void SetAlpha(double a);

  /// Clamp for rho to avoid division by very small numbers (default IGRDefaults::kRhoFloor).
  void SetRhoFloor(double rho_floor);

  /// Set essential (Dirichlet) boundary attributes for Σ; default is none (natural Neumann).
  void SetDirichletBC(const mfem::Array<int>& ess_bdr);

  /// Select which gradient backend to use for the Σ RHS.
  void SetGradBackend(GradBackend b) { backend_ = b; }
  void SetIPKind(IPKind k) { ip_kind_ = k; }

  // ---- Legacy smoother-sweep knobs (still supported) ----
  void SetSmoother(SmootherType t);
  void SetSweeps(int regular_sweeps, int warm_start_sweeps);

  // ---- Recommended Krylov-solve knobs ----
  void SetSolveMode(SolveMode m) { solve_mode_ = m; }
  void SetLinearSolver(LinearSolverType s) { lin_solver_type_ = s; }
  void SetPreconditioner(PrecondType p) { prec_type_ = p; }

  /// Set solve tolerances and iteration cap (used in Krylov mode).
  void SetLinearSolveParams(double rel_tol, double abs_tol, int max_it, int print_level = 0)
  {
    lin_rtol_  = rel_tol;
    lin_atol_  = abs_tol;
    lin_maxit_ = max_it;
    lin_print_ = print_level;
  }

  /// Apply a configuration bundle without changing existing defaults.
  void ApplyConfig(const Config &cfg)
  {
    SetAlpha(cfg.alpha);
    SetRhoFloor(cfg.rho_floor);
    SetGradBackend(cfg.backend);
    SetIPKind(cfg.ip_kind);

    SetSmoother(cfg.smoot);
    SetSweeps(cfg.regular_sweeps, cfg.warm_start_sweeps);

    SetSolveMode(cfg.solve_mode);
    SetLinearSolver(cfg.lin_solver_type);
    SetPreconditioner(cfg.prec_type);
    SetLinearSolveParams(cfg.lin_rtol, cfg.lin_atol, cfg.lin_maxit, cfg.lin_print);
  }

  /// Update Σ from the current state U.
  void UpdateFromState(const mfem::ParGridFunction& U, bool warm_start);

  /// Access the current Σ field.
  const mfem::ParGridFunction& GetSigma() const { return sigma_; }
  mfem::ParGridFunction&       GetSigma()       { return sigma_; }

  /// Spatial dimension.
  int Dim() const { return dim_; }

  /// Number of equations (vdim of state U).
  int NumEquations() const { return num_eq_; }

private:
  // ---------- Coefficients that depend on the current state U ----------
  /// c_M(x) = 1 / max(ρ, rho_floor_)
  class InvRhoCoefficient : public mfem::Coefficient
  {
  public:
    InvRhoCoefficient(const mfem::ParGridFunction& U, double rho_floor);
    double Eval(mfem::ElementTransformation &T,
                const mfem::IntegrationPoint &ip) override;
    void SetState(const mfem::ParGridFunction& U) { U_ = &U; }
    void SetRhoFloor(double floor) { rho_floor_ = floor; }
  private:
    const mfem::ParGridFunction* U_ = nullptr;
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
    void Eval(mfem::Vector &V,
              mfem::ElementTransformation &T,
              const mfem::IntegrationPoint &ip) override;
  private:
    const mfem::ParGridFunction* U_;
    double rho_floor_;
  };

  /// f(x) = α ( ||∇u||_F^2 + (∇·u)^2 ), with u = m / ρ (componentwise).
  class SigmaSourceCoefficient : public mfem::Coefficient
  {
  public:
    SigmaSourceCoefficient(const mfem::ParGridFunction& U,
                           int dim, int num_eq, double alpha, double rho_floor);
    double Eval(mfem::ElementTransformation &T,
                const mfem::IntegrationPoint &ip) override;
    void SetState(const mfem::ParGridFunction& U) { U_ = &U; }
    void SetAlpha(double a) { alpha_ = a; }
    void SetRhoFloor(double floor) { rho_floor_ = floor; }
    // Optional continuous velocity field (vdim=dim); if set, its gradient is used.
    void SetVelField(const mfem::ParGridFunction* v) { velH1_ = v; }
  private:
    const mfem::ParGridFunction* U_ = nullptr;
    const mfem::ParGridFunction* velH1_ = nullptr; // optional continuous velocity
    int dim_, num_eq_;
    double alpha_, rho_floor_;
    mfem::Vector      state_;   // [rho, m(0:dim-1), E]
    mfem::DenseMatrix gradU_;   // num_eq_ x dim_
    mfem::DenseMatrix gradu_;   // dim_ x dim_
  };

  /// α as a coefficient that references a double.
  class AlphaRefCoefficient : public mfem::Coefficient
  {
  public:
    explicit AlphaRefCoefficient(const double &a_ref) : a_ref_(a_ref) {}
    double Eval(mfem::ElementTransformation &,
                const mfem::IntegrationPoint &) override { return a_ref_; }
  private:
    const double &a_ref_;
  };

  // ---------- Core assemble/apply helpers ----------
  void AssembleSystem(const mfem::ParGridFunction& U); // fill A and b for current U
  void DoSmootherSweeps(int sweeps, bool warm_start);   // legacy
  void SolveEllipticSystem_(bool warm_start);           // new recommended path
  void EnsureH1VelocitySpace_();

  // ---------- Data ----------
  const mfem::ParFiniteElementSpace& sfes_;  // scalar Σ space (H1 or DG)
  const mfem::ParFiniteElementSpace& vfes_;  // vector state U space (vdim=num_eq)
  int dim_, num_eq_;
  double alpha_       = mfem::dg_igr::IGRDefaults::kAlpha;
  double rho_floor_   = mfem::dg_igr::IGRDefaults::kRhoFloor;
  SmootherType smoot_ = SmootherType::L1Jacobi;
  GradBackend  backend_ = GradBackend::H1Projection;
  IPKind       ip_kind_ = IPKind::SIPG; //default SPD

  // Legacy sweeps
  int  regular_sweeps_    = mfem::dg_igr::IGRDefaults::kRegularSweeps;
  int  warm_start_sweeps_ = mfem::dg_igr::IGRDefaults::kWarmStartSweeps;
  bool first_call_        = true;

  // New solve controls
  SolveMode        solve_mode_      = SolveMode::Krylov;
  LinearSolverType lin_solver_type_ = LinearSolverType::Auto;
  PrecondType      prec_type_       = PrecondType::Auto;

  double lin_rtol_ = mfem::dg_igr::IGRDefaults::kLinRtol;
  double lin_atol_ = mfem::dg_igr::IGRDefaults::kLinAtol;
  int    lin_maxit_ = mfem::dg_igr::IGRDefaults::kLinMaxIters;
  int    lin_print_ = mfem::dg_igr::IGRDefaults::kLinPrint;

  // Σ field
  mfem::ParGridFunction sigma_;

  // FE forms (reused each call; values change, sparsity fixed)
  std::unique_ptr<mfem::ParBilinearForm> a_;
  std::unique_ptr<mfem::ParLinearForm>   b_;

  // Coefficients that refer to U
  std::unique_ptr<InvRhoCoefficient>        invrho_coef_;
  std::unique_ptr<SigmaSourceCoefficient>   src_coef_;
  std::unique_ptr<AlphaRefCoefficient>      alpha_coef_;  // α(x) ≡ alpha_
  std::unique_ptr<mfem::ProductCoefficient> ck_coef_;     // ck(x) = α/ρ

  // Linear system objects
  mfem::Array<int>        ess_bdr_;
  mfem::Array<int>        ess_tdof_list_;
  mfem::OperatorHandle    Ah_;     // holds the assembled operator (Hypre matrix)
  mfem::Vector            X_, B_;  // true-dof vectors

  // H1 projection backend storage
  std::unique_ptr<mfem::H1_FECollection>        h1_vec_fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace>  h1_vec_fes_;  // vdim = dim
  std::unique_ptr<mfem::ParGridFunction>        vel_h1_;      // projected velocity
};

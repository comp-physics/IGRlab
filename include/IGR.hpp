#pragma once
#include "mfem.hpp"

/**
 * @file IGR.hpp
 * @brief Information Geometric Regularization (IGR) solver for Σ coupled to Euler.
 *
 * Given the current conservative state U = [ρ, ρu, E], this class computes Σ by
 * approximately solving the scalar elliptic PDE
 *
 *   (1/ρ) Σ - α ∇·( (1/ρ) ∇Σ ) = α ( ||∇u||_F^2 + (∇·u)^2 )
 *
 * using a small, fixed number of Jacobi-like smoothing sweeps with a warm start
 * from the previously computed Σ. The result is then used by the hyperbolic
 * operator via p -> p + Σ in momentum/energy fluxes.
 *
 * Usage (per stage):
 *   1) IGRSolver igr(sfes, vfes, dim, num_eq, alpha);
 *   2) igr.UpdateFromState(U, //warm_start=//true);
 *   3) const mfem::ParGridFunction& sigma = igr.GetSigma();
 */
class IGRSolver
{
public:
  /// Available smoother types for the fixed-iteration “few sweeps” update.
  enum class SmootherType { Jacobi, L1Jacobi };
  // enum class SmootherType { Jacobi, L1Jacobi, Chebyshev };

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

  /// Select smoother type (default: L1Jacobi for parallel robustness).
  void SetSmoother(SmootherType t);

  /// Set sweep counts: regular per-call sweeps and warm-start sweeps for the first call.
  void SetSweeps(int regular_sweeps, int warm_start_sweeps);

  /// Clamp for ρ to avoid division by very small numbers (default 1e-12).
  void SetRhoFloor(double rho_floor);

  /// Set essential (Dirichlet) boundary attributes for Σ; default is none (natural Neumann).
  void SetDirichletBC(const mfem::Array<int>& ess_bdr);

  /**
   * @brief Assemble and apply ≤N smoother sweeps to update Σ from current state U.
   *
   * Forms the linear system A(Σ)=b where:
   *   A = ( (1/ρ)·,· ) + α ( (1/ρ)∇·,∇· )
   *   b = α ( ||∇u||_F^2 + (∇·u)^2 , · )
   *
   * Uses the previously computed Σ as initial guess if warm_start=true,
   * then performs a small, fixed number of smoother iterations.
   *
   * @param U           Conservative state as a vector ParGridFunction (vdim=num_eq).
   * @param warm_start  If true, use the current Σ as initial guess and take
   *                    warm_start_sweeps on the first call; otherwise use regular_sweeps.
   */
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
  private:
    const mfem::ParGridFunction* U_ = nullptr;
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
  void DoSmootherSweeps(int sweeps, bool warm_start);  // ≤N sweeps with chosen smoother

  // ---------- Data ----------
  const mfem::ParFiniteElementSpace& sfes_;  // scalar Σ space
  const mfem::ParFiniteElementSpace& vfes_;  // vector state U space (vdim=num_eq)
  int dim_, num_eq_;
  double alpha_       = 0.0;
  double rho_floor_   = 1e-12;
  SmootherType smoot_ = SmootherType::L1Jacobi;

  int  regular_sweeps_     = 3;
  int  warm_start_sweeps_  = 5;
  bool first_call_         = true;

  // Σ field
  mfem::ParGridFunction sigma_;

  // FE forms (reused each call; values change, sparsity fixed)
  std::unique_ptr<mfem::ParBilinearForm> a_;
  std::unique_ptr<mfem::ParLinearForm>   b_;

  // Coefficients that refer to U
  std::unique_ptr<InvRhoCoefficient>       invrho_coef_;
  std::unique_ptr<SigmaSourceCoefficient>  src_coef_;
  std::unique_ptr<AlphaRefCoefficient>     alpha_coef_;  // α(x) ≡ alpha_
  std::unique_ptr<mfem::ProductCoefficient> ck_coef_;    // ck(x) = α/ρ

  // Linear system objects
  mfem::Array<int>        ess_bdr_;
  mfem::Array<int>        ess_tdof_list_;
  mfem::OperatorHandle    Ah_;     // holds the assembled operator (Hypre matrix)
  mfem::Vector            X_, B_;  // true-dof vectors
};

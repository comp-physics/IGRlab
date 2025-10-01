#include "IGR.hpp"
#include <cmath>

using namespace mfem;

// ---------------- IGRSolver::InvRhoCoefficient ----------------
IGRSolver::InvRhoCoefficient::InvRhoCoefficient(const ParGridFunction& U, double rho_floor)
  : U_(&U), rho_floor_(rho_floor) {}

double IGRSolver::InvRhoCoefficient::Eval(ElementTransformation &T,
                                          const IntegrationPoint &ip)
{
  MFEM_VERIFY(U_ != nullptr, "InvRhoCoefficient requires a valid state U.");
  T.SetIntPoint(&ip);

  // Conservative state value at (T, ip): state.Size() == num_eq
  Vector state;
  U_->GetVectorValue(T, ip, state);
  const double rho = std::max(state(0), rho_floor_);
  return 1.0 / rho;
}

// ---------------- IGRSolver::SigmaSourceCoefficient ----------------
IGRSolver::SigmaSourceCoefficient::SigmaSourceCoefficient(const ParGridFunction& U,
                                                          int dim, int num_eq,
                                                          double alpha, double rho_floor)
  : U_(&U), dim_(dim), num_eq_(num_eq),
    alpha_(alpha), rho_floor_(rho_floor),
    state_(num_eq), gradU_(num_eq, dim), gradu_(dim, dim)
{ }

double IGRSolver::SigmaSourceCoefficient::Eval(ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
  MFEM_VERIFY(U_ != nullptr, "SigmaSourceCoefficient requires a valid state U.");

  T.SetIntPoint(&ip);
  U_->GetVectorValue(T, ip, state_);        // [rho, m(0..dim-1), E]
  U_->GetVectorGradient(T, gradU_);         // rows: components, cols: spatial derivs

  const double rho = std::max(state_(0), rho_floor_);

  // momentum components:
  Vector mom(&state_(1), dim_);

  // grad of rho: row 0 in gradU_
  Vector grad_rho(dim_);
  for (int j = 0; j < dim_; ++j) { grad_rho(j) = gradU_(0, j); }

  // Build grad(u) via quotient rule: du_i/dx_j = (rho*dm_i/dx_j - m_i*drho/dx_j)/rho^2
  const double rho2 = rho * rho;
  for (int i = 0; i < dim_; ++i)
  {
    for (int j = 0; j < dim_; ++j)
    {
      const double dmij = gradU_(1 + i, j);
      gradu_(i, j) = (rho * dmij - mom(i) * grad_rho(j)) / rho2;
    }
  }

  // divergence
  double divu = 0.0;
  for (int i = 0; i < dim_; ++i) { divu += gradu_(i,i); }

  // tr( (∇u)^2 ) = sum_{i,j} (∂_j u_i) (∂_i u_j)
  double trA2 = 0.0;
  for (int i = 0; i < dim_; ++i)
  {
    for (int j = 0; j < dim_; ++j)
    {
      trA2 += gradu_(i,j) * gradu_(j,i);
    }
  }

  // Paper RHS: α ( tr((∇u)^2) + (tr ∇u)^2 )
  return alpha_ * (trA2 + divu*divu);
}

// ---------------- IGRSolver core ----------------
IGRSolver::IGRSolver(const ParFiniteElementSpace& sigma_fes,
                     const ParFiniteElementSpace& state_fes,
                     int dim, int num_eq, double alpha)
  : sfes_(sigma_fes),
    vfes_(state_fes),
    dim_(dim),
    num_eq_(num_eq),
    alpha_(alpha),
    sigma_(const_cast<ParFiniteElementSpace*>(&sfes_))
{
  // Allocate forms once
  a_ = std::make_unique<ParBilinearForm>(const_cast<ParFiniteElementSpace*>(&sfes_));
  b_ = std::make_unique<ParLinearForm>(const_cast<ParFiniteElementSpace*>(&sfes_));

  // Create coefficients once (they hold references; we’ll update their state each call)
  // NOTE: we pass a dummy U for construction; we'll immediately SetState(U) later.
  static ParGridFunction dummy(const_cast<ParFiniteElementSpace*>(&vfes_));
  dummy = 1.0;

  invrho_coef_ = std::make_unique<InvRhoCoefficient>(dummy, rho_floor_);
  src_coef_    = std::make_unique<SigmaSourceCoefficient>(dummy, dim_, num_eq_, alpha_, rho_floor_);
  alpha_coef_  = std::make_unique<AlphaRefCoefficient>(alpha_);
  ck_coef_     = std::make_unique<ProductCoefficient>(*alpha_coef_, *invrho_coef_);

  // Attach integrators ONCE.
  a_->AddDomainIntegrator(new MassIntegrator(*invrho_coef_));
  a_->AddDomainIntegrator(new DiffusionIntegrator(*ck_coef_));
  b_->AddDomainIntegrator(new DomainLFIntegrator(*src_coef_));

  // Default BCs
  ess_bdr_.SetSize(0);

  // Initial Σ
  sigma_ = 0.0;

  // Initialize operator handle type we expect (parallel CSR)
  Ah_.SetType(Operator::Hypre_ParCSR);
}

void IGRSolver::SetAlpha(double a)
{
  alpha_ = a;              // alpha_coef reads this by reference (see above)
  if (src_coef_) { src_coef_->SetAlpha(a); }
}

void IGRSolver::SetSmoother(SmootherType t)
{
  smoot_ = t;
}

void IGRSolver::SetSweeps(int regular_sweeps, int warm_start_sweeps)
{
  regular_sweeps_    = std::max(0, regular_sweeps);
  warm_start_sweeps_ = std::max(regular_sweeps_, warm_start_sweeps);
}

void IGRSolver::SetRhoFloor(double rho_floor)
{
  rho_floor_ = rho_floor;
  if (invrho_coef_) { invrho_coef_->SetRhoFloor(rho_floor_); }
  if (src_coef_)    { src_coef_->SetRhoFloor(rho_floor_); }
}

void IGRSolver::SetDirichletBC(const Array<int>& ess_bdr)
{
  ess_bdr_.SetSize(ess_bdr.Size());
  ess_bdr_ = ess_bdr;
}

void IGRSolver::AssembleSystem(const ParGridFunction& U)
{
  // Refresh coefficient views of the current state and params
  invrho_coef_->SetState(U);
  invrho_coef_->SetRhoFloor(rho_floor_);

  src_coef_->SetState(U);
  src_coef_->SetAlpha(alpha_);
  src_coef_->SetRhoFloor(rho_floor_);

  // --- IMPORTANT: tell MFEM we will re-use the same ParBilinearForm again ---
  // Forms keep internal parallel structures (P, R, etc.); Update() is required
  // before re-assembly when the form has already been used once.
  a_->Update();
  b_->Update();

  // Re-assemble values (sparsity pattern unchanged)
  a_->Assemble();
  b_->Assemble();

  // Rebuild essential dofs if user set Dirichlet BCs
  ess_tdof_list_.SetSize(0);
  if (ess_bdr_.Size() > 0)
  {
    Array<int> dbdr(ess_bdr_);
    sfes_.GetEssentialTrueDofs(dbdr, ess_tdof_list_);
  }

  // We are going to rebuild the system operator; clear the handle first.
  Ah_.Clear();
  Ah_.SetType(Operator::Hypre_ParCSR); // keep explicit; harmless if redundant

  // Warm-start: FormLinearSystem pulls current sigma_ into X_
  // Use the OperatorHandle-based overload (parallel)
  a_->FormLinearSystem(ess_tdof_list_, sigma_, *b_, Ah_, X_, B_);
}

void IGRSolver::DoSmootherSweeps(int sweeps, bool /*warm_start*/)
{
  // We expect a Hypre matrix here
  HypreParMatrix *A_hypre = Ah_.As<HypreParMatrix>();
  MFEM_VERIFY(A_hypre != nullptr, "IGR: Assembled operator is not HypreParMatrix.");

  HypreSmoother S(*A_hypre);
  switch (smoot_)
  {
    case SmootherType::Jacobi:    S.SetType(HypreSmoother::Jacobi);    break;
    case SmootherType::L1Jacobi:  S.SetType(HypreSmoother::l1Jacobi);  break;
    // case SmootherType::Chebyshev: S.SetType(HypreSmoother::Chebyshev); break;
  }
  S.SetPositiveDiagonal(true);

  Vector r(B_.Size()), z(X_.Size());
  for (int it = 0; it < sweeps; ++it)
  {
    // r = b - A x
    r = B_;
    A_hypre->Mult(X_, r);
    r.Neg(); r += B_;

    // z ≈ M^{-1} r  (one application of the smoother/preconditioner)
    z = 0.0;
    S.Mult(r, z);

    // x <- x + z  (Richardson step)
    X_ += z;
  }

  // Scatter back to FE space
  a_->RecoverFEMSolution(X_, *b_, sigma_);
}

void IGRSolver::UpdateFromState(const ParGridFunction& U, bool warm_start)
{
  // Assemble A and b for current state U (warm-start handled by FormLinearSystem)
  AssembleSystem(U);

  // Decide sweep count
  const int sweeps = (first_call_ && warm_start)
                     ? std::max(warm_start_sweeps_, regular_sweeps_)
                     : regular_sweeps_;

  DoSmootherSweeps(sweeps, warm_start);
  first_call_ = false;
}

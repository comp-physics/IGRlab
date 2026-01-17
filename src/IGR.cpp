#include "IGR.hpp"
#include <cmath>
#include <algorithm>

using namespace mfem;

// ----------------- small helpers (file-local) -----------------
namespace {

inline bool IsDGSpace(const mfem::FiniteElementSpace &fes)
{
  // Identify DG/L2 spaces by their FECollection type
  const mfem::FiniteElementCollection *fec = fes.FEColl(); // FE collection accessor
  return dynamic_cast<const mfem::L2_FECollection*>(fec) != nullptr
      || dynamic_cast<const mfem::DG_FECollection*>(fec) != nullptr;
}

inline int SpaceOrder(const mfem::FiniteElementSpace &fes)
{
  MFEM_VERIFY(fes.GetNE() > 0, "Empty FES.");
  return fes.GetFE(0)->GetOrder();
}

inline double IPDGPenalty(const mfem::FiniteElementSpace &fes, double Cbase = 4.0)
{
  const int p = SpaceOrder(fes);
  return Cbase * (p + 1) * (p + 1);
}

inline bool IsSPD_DefaultAssumption(const mfem::FiniteElementSpace &sfes,
                                   IGRSolver::IPKind ip_kind)
{
  // Safe heuristic:
  //  - H1 space with symmetric bilinear form => SPD
  //  - DG diffusion is SPD under SIPG (kappa = -1), but can be nonsymmetric for NIPG.
  if (IsDGSpace(sfes))
  {
    return (ip_kind == IGRSolver::IPKind::SIPG);
  }
  return true; // H1: mass + diffusion -> SPD
}

} // namespace

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
  U_->GetVectorValue(T, ip, state); // [rho, m(0..dim-1), E]
  const double rho = std::max(state(0), rho_floor_);
  return 1.0 / rho;
}

// ---------------- IGRSolver::VelocityFromConsCoefficient ---------------
IGRSolver::VelocityFromConsCoefficient::VelocityFromConsCoefficient(
    int dim, const ParGridFunction* U, double rho_floor)
  : VectorCoefficient(dim), U_(U), rho_floor_(rho_floor)
{ }

void IGRSolver::VelocityFromConsCoefficient::Eval(Vector &V,
                                                  ElementTransformation &T,
                                                  const IntegrationPoint &ip)
{
  MFEM_VERIFY(U_ != nullptr, "VelocityFromConsCoefficient requires a valid state U.");
  T.SetIntPoint(&ip);
  Vector cons;
  U_->GetVectorValue(T, ip, cons); // [rho, m(0..dim-1), E]
  const double rho = std::max(cons(0), rho_floor_);
  V.SetSize(vdim);
  for (int d = 0; d < vdim; ++d)
  {
    const double md = cons(1 + d);
    V(d) = md / rho;
  }
}

// ---------------- IGRSolver::SigmaSourceCoefficient ----------------
IGRSolver::SigmaSourceCoefficient::SigmaSourceCoefficient(const ParGridFunction& U,
                                                          int dim, int num_eq,
                                                          double alpha, double rho_floor)
  : U_(&U), velH1_(nullptr), dim_(dim), num_eq_(num_eq),
    alpha_(alpha), rho_floor_(rho_floor),
    state_(num_eq), gradU_(num_eq, dim), gradu_(dim, dim)
{ }

double IGRSolver::SigmaSourceCoefficient::Eval(ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
  MFEM_VERIFY(U_ != nullptr, "SigmaSourceCoefficient requires a valid state U.");
  T.SetIntPoint(&ip);

  if (velH1_)
  {
    // H1Projection backend: use ∇(projected velocity)
    DenseMatrix gradv(dim_, dim_);
    velH1_->GetVectorGradient(T, gradv);   // gradv(i,j) = ∂_j u_i

    // div u
    double divu = 0.0;
    for (int i = 0; i < dim_; ++i) { divu += gradv(i,i); }

    // tr((∇u)^2) = sum_ij (∂_j u_i)(∂_i u_j)
    double trA2 = 0.0;
    for (int i = 0; i < dim_; ++i)
      for (int j = 0; j < dim_; ++j)
        trA2 += gradv(i,j) * gradv(j,i);

    return alpha_ * (trA2 + divu*divu);
  }

  // BrokenDG backend: in-cell quotient rule from conservative gradients
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
  static ParGridFunction dummy(const_cast<ParFiniteElementSpace*>(&vfes_));
  dummy = 1.0;

  invrho_coef_ = std::make_unique<InvRhoCoefficient>(dummy, rho_floor_);
  src_coef_    = std::make_unique<SigmaSourceCoefficient>(dummy, dim_, num_eq_, alpha_, rho_floor_);
  alpha_coef_  = std::make_unique<AlphaRefCoefficient>(alpha_);
  ck_coef_     = std::make_unique<ProductCoefficient>(*alpha_coef_, *invrho_coef_);

  // ---- Operator: mass + diffusion (IPDG if Σ-space is DG) ----
  // Reaction/mass: (1/ρ σ, v)
  a_->AddDomainIntegrator(new MassIntegrator(*invrho_coef_));

  // Volume diffusion term: ( (α/ρ) ∇σ, ∇v )
  a_->AddDomainIntegrator(new DiffusionIntegrator(*ck_coef_));

  // If Σ-space is discontinuous, add IPDG face terms (interior + boundary)
  if (IsDGSpace(sfes_))
  {
    const double sigma_ip = IPDGPenalty(sfes_);                   // penalty scaling
    const int    kappa    = static_cast<int>(ip_kind_);           // -1(SIPG),0, +1
    a_->AddInteriorFaceIntegrator(
        new DGDiffusionIntegrator(*ck_coef_, sigma_ip, kappa));
    a_->AddBdrFaceIntegrator(
        new DGDiffusionIntegrator(*ck_coef_, sigma_ip, kappa));
  }

  // RHS: α ( ||∇u||_F^2 + (∇·u)^2 , v )
  b_->AddDomainIntegrator(new DomainLFIntegrator(*src_coef_));

  // Default BCs: natural Neumann (no flux). For Nitsche Dirichlet, add linear terms later.

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

static int GetSpaceOrder_(const ParFiniteElementSpace& fes)
{
  MFEM_VERIFY(fes.GetNE() > 0, "Empty FES.");
  return fes.GetFE(0)->GetOrder();
}

void IGRSolver::EnsureH1VelocitySpace_()
{
  if (backend_ != GradBackend::H1Projection) { return; }

  ParMesh* pm = vfes_.GetParMesh();
  MFEM_VERIFY(pm, "Expected ParMesh for state space.");

  // Choose H1 order sensibly: ≥ 1, and not below the state space order.
  const int p_state = GetSpaceOrder_(vfes_);
  const int p_h1 = std::max(1, p_state);

  // Recreate only if absent or order/mesh changed
  bool need_make = (!h1_vec_fec_ || !h1_vec_fes_ || !vel_h1_);
  if (!need_make)
  {
    if (h1_vec_fec_->GetOrder() != p_h1) need_make = true;
    if (h1_vec_fes_->GetParMesh() != pm) need_make = true;
  }
  if (need_make)
  {
    h1_vec_fec_ = std::make_unique<H1_FECollection>(p_h1, pm->Dimension());
    h1_vec_fes_ = std::make_unique<ParFiniteElementSpace>(pm, h1_vec_fec_.get(), /*vdim=*/dim_);
    vel_h1_     = std::make_unique<ParGridFunction>(h1_vec_fes_.get());
    *vel_h1_    = 0.0;
  }
}

void IGRSolver::AssembleSystem(const ParGridFunction& U)
{
  // Needed for DG/IPDG assembly: variable coeffs evaluated on shared faces
  // will read U on neighbor elements.
  const_cast<ParGridFunction&>(U).ExchangeFaceNbrData();

  // Refresh coefficient views of the current state and params
  invrho_coef_->SetState(U);
  invrho_coef_->SetRhoFloor(rho_floor_);

  // Build gradient backend inputs
  if (backend_ == GradBackend::H1Projection)
  {
    EnsureH1VelocitySpace_();

    // Project velocity u = m/ρ into H1^dim
    VelocityFromConsCoefficient vel_coeff(dim_, &U, rho_floor_);
    vel_coeff.SetRhoFloor(rho_floor_);
    vel_coeff.SetState(&U);

    vel_h1_->ProjectCoefficient(vel_coeff);

    // Tell RHS coefficient to use this continuous velocity field
    src_coef_->SetVelField(vel_h1_.get());
  }
  else
  {
    // BrokenDG backend: remove any projected velocity
    src_coef_->SetVelField(nullptr);
  }

  src_coef_->SetState(U);
  src_coef_->SetAlpha(alpha_);
  src_coef_->SetRhoFloor(rho_floor_);

  // --- IMPORTANT: tell MFEM we will re-use the same ParBilinearForm again ---
  a_->Update();
  b_->Update();

  // Re-assemble values (sparsity pattern unchanged)
  a_->Assemble();
  b_->Assemble();

  // Rebuild essential dofs if user set Dirichlet BCs (for H1 spaces).
  ess_tdof_list_.SetSize(0);
  if (ess_bdr_.Size() > 0)
  {
    Array<int> dbdr(ess_bdr_);
    sfes_.GetEssentialTrueDofs(dbdr, ess_tdof_list_);
  }

  // Rebuild the system operator; clear handle first.
  Ah_.Clear();
  Ah_.SetType(Operator::Hypre_ParCSR);

  // Warm-start: FormLinearSystem pulls current sigma_ into X_
  a_->FormLinearSystem(ess_tdof_list_, sigma_, *b_, Ah_, X_, B_);
}

void IGRSolver::DoSmootherSweeps(int sweeps, bool /*warm_start*/)
{
  HypreParMatrix *A_hypre = Ah_.As<HypreParMatrix>();
  MFEM_VERIFY(A_hypre != nullptr, "IGR: Assembled operator is not HypreParMatrix.");

  HypreSmoother S(*A_hypre);
  switch (smoot_)
  {
    case SmootherType::Jacobi:    S.SetType(HypreSmoother::Jacobi);   break;
    case SmootherType::L1Jacobi:  S.SetType(HypreSmoother::l1Jacobi); break;
    case SmootherType::L1GS:      S.SetType(HypreSmoother::l1GS);     break;
    case SmootherType::L1GStr:    S.SetType(HypreSmoother::l1GStr);   break;
    case SmootherType::GS:        S.SetType(HypreSmoother::GS);       break;
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

  a_->RecoverFEMSolution(X_, *b_, sigma_);
}

void IGRSolver::SolveEllipticSystem_(bool warm_start)
{
  HypreParMatrix *A = Ah_.As<HypreParMatrix>();
  MFEM_VERIFY(A != nullptr, "IGR: Assembled operator is not HypreParMatrix.");

  MPI_Comm comm = sfes_.GetComm();
  const bool is_dg  = IsDGSpace(sfes_);
  const bool spd_ok = IsSPD_DefaultAssumption(sfes_, ip_kind_);

  // --- Choose solver (Auto -> CG if SPD, else GMRES) ---
  LinearSolverType solver_choice = lin_solver_type_;
  if (solver_choice == LinearSolverType::Auto)
  {
    solver_choice = spd_ok ? LinearSolverType::CG : LinearSolverType::GMRES;
  }

  // --- Choose preconditioner (Auto -> AMG for H1, smoother for DG) ---
  PrecondType prec_choice = prec_type_;
  if (prec_choice == PrecondType::Auto)
  {
    prec_choice = (!is_dg) ? PrecondType::BoomerAMG : PrecondType::HypreSmoother;
  }

  std::unique_ptr<Solver> prec;
  if (prec_choice == PrecondType::BoomerAMG)
  {
    auto amg = std::make_unique<HypreBoomerAMG>(*A);
    // Typical AMG-as-preconditioner settings
    amg->SetPrintLevel(0);
    amg->SetMaxIter(1);
    amg->SetTol(0.0);
    prec = std::move(amg);
  }
  else if (prec_choice == PrecondType::HypreSmoother)
  {
    auto sm = std::make_unique<HypreSmoother>(*A);
    switch (smoot_)
    {
      case SmootherType::Jacobi:   sm->SetType(HypreSmoother::Jacobi);   break;
      case SmootherType::L1Jacobi: sm->SetType(HypreSmoother::l1Jacobi); break;
      case SmootherType::L1GS:     sm->SetType(HypreSmoother::l1GS);     break;
      case SmootherType::L1GStr:   sm->SetType(HypreSmoother::l1GStr);   break;
      case SmootherType::GS:       sm->SetType(HypreSmoother::GS);       break;
    }
    sm->SetPositiveDiagonal(true);
    prec = std::move(sm);
  }
  // else None -> leave prec null

  if (solver_choice == LinearSolverType::CG)
  {
    CGSolver solver(comm);
    solver.SetOperator(*A);
    if (prec) { solver.SetPreconditioner(*prec); }

    solver.SetRelTol(lin_rtol_);
    solver.SetAbsTol(lin_atol_);
    solver.SetMaxIter(lin_maxit_);
    solver.SetPrintLevel(lin_print_);

    // IMPORTANT: use warm-start guess X_ from FormLinearSystem if requested
    // solver.SetInitialGuess(warm_start);

    solver.Mult(B_, X_);
  }
  else // GMRES
  {
    GMRESSolver solver(comm);
    solver.SetOperator(*A);
    if (prec) { solver.SetPreconditioner(*prec); }

    solver.SetRelTol(lin_rtol_);
    solver.SetAbsTol(lin_atol_);
    solver.SetMaxIter(lin_maxit_);
    solver.SetPrintLevel(lin_print_);
    // solver.iterative_mode = warm_start;

    solver.Mult(B_, X_);
  }

  a_->RecoverFEMSolution(X_, *b_, sigma_);
}

void IGRSolver::UpdateFromState(const ParGridFunction& U, bool warm_start)
{
  // Assemble A and b for current state U (warm-start handled by FormLinearSystem)
  AssembleSystem(U);

  if (solve_mode_ == SolveMode::SmootherSweeps)
  {
    const int sweeps = (first_call_ && warm_start)
                       ? std::max(warm_start_sweeps_, regular_sweeps_)
                       : regular_sweeps_;
    DoSmootherSweeps(sweeps, warm_start);
  }
  else
  {
    SolveEllipticSystem_(warm_start);
  }

  first_call_ = false;
}

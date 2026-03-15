#include "physics/igr/IGR.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace dg_igr
{

// ----------------- small helpers (file-local) -----------------
namespace
{

inline bool IsDGSpace(const mfem::FiniteElementSpace& fes)
{
  // Identify DG/L2 spaces by their FECollection type
  const mfem::FiniteElementCollection* fec =
    fes.FEColl(); // FE collection accessor
  return dynamic_cast<const mfem::L2_FECollection*>(fec) != nullptr ||
         dynamic_cast<const mfem::DG_FECollection*>(fec) != nullptr;
}

inline int SpaceOrder(const mfem::FiniteElementSpace& fes)
{
  MFEM_VERIFY(fes.GetNE() > 0, "Empty FES.");
  return fes.GetFE(0)->GetOrder();
}

inline double IPDGPenalty(const mfem::FiniteElementSpace& fes,
                          double Cbase = 4.0)
{
  const int p = SpaceOrder(fes);
  return Cbase * (p + 1) * (p + 1);
}

inline bool IsSPD_DefaultAssumption(const mfem::FiniteElementSpace& sfes,
                                    IGRSolver::IPKind ip_kind)
{
  // Structural policy:
  //  - H1 operator is assembled symmetrically -> SPD-friendly.
  //  - DG/L2 with SIPG is SPD-by-construction when DGDiffusionIntegrator
  //    is wired as (sigma=-1, kappa>0 penalty). IIPG/NIPG are non-SPD.
  if (IsDGSpace(sfes))
  {
    return ip_kind == IGRSolver::IPKind::SIPG;
  }
  return true; // H1: mass + diffusion -> SPD
}

inline const char* LinearSolverTypeName(IGRSolver::LinearSolverType t)
{
  switch (t)
  {
  case IGRSolver::LinearSolverType::Auto:
    return "auto";
  case IGRSolver::LinearSolverType::CG:
    return "cg";
  case IGRSolver::LinearSolverType::GMRES:
    return "gmres";
  case IGRSolver::LinearSolverType::FGMRES:
    return "fgmres";
  case IGRSolver::LinearSolverType::BiCGSTAB:
    return "bicgstab";
  case IGRSolver::LinearSolverType::MINRES:
    return "minres";
  }
  return "auto";
}

inline const char* PrecondTypeName(IGRSolver::PrecondType t)
{
  switch (t)
  {
  case IGRSolver::PrecondType::Auto:
    return "auto";
  case IGRSolver::PrecondType::BoomerAMG:
    return "boomeramg";
  case IGRSolver::PrecondType::HypreILU:
    return "hypreilu";
  case IGRSolver::PrecondType::HypreEuclid:
    return "hypreeuclid";
  case IGRSolver::PrecondType::ParaSails:
    return "parasails";
  case IGRSolver::PrecondType::HypreSmoother:
    return "hypresmoother";
  case IGRSolver::PrecondType::None:
    return "none";
  }
  return "auto";
}

inline bool RequiresSPD(IGRSolver::LinearSolverType solver)
{
  return solver == IGRSolver::LinearSolverType::CG ||
         solver == IGRSolver::LinearSolverType::MINRES;
}

inline bool IsSymmetricPC(IGRSolver::PrecondType pc)
{
  return pc == IGRSolver::PrecondType::BoomerAMG ||
         pc == IGRSolver::PrecondType::HypreSmoother ||
         pc == IGRSolver::PrecondType::None;
}

inline bool VectorIsFinite(const mfem::Vector& v)
{
  const int n = v.Size();
  for (int i = 0; i < n; ++i)
  {
    if (!std::isfinite(v(i)))
    {
      return false;
    }
  }
  return true;
}

inline void LogIGRFallback(const char* reason,
                           IGRSolver::LinearSolverType requested_solver,
                           IGRSolver::LinearSolverType selected_solver,
                           IGRSolver::PrecondType requested_prec,
                           IGRSolver::PrecondType selected_prec)
{
  if (!mfem::Mpi::Root())
  {
    return;
  }
  std::cout << "[igr-linear] event=fallback" << " reason=\"" << reason << "\""
            << " requested_solver=" << LinearSolverTypeName(requested_solver)
            << " selected_solver=" << LinearSolverTypeName(selected_solver)
            << " requested_precond=" << PrecondTypeName(requested_prec)
            << " selected_precond=" << PrecondTypeName(selected_prec) << "\n";
}

inline void LogIGRSelection(bool warm_start,
                            bool is_dg,
                            bool spd_ok,
                            IGRSolver::LinearSolverType solver_choice,
                            IGRSolver::PrecondType prec_choice,
                            double lin_rtol,
                            double lin_atol,
                            int lin_maxit)
{
  if (!mfem::Mpi::Root())
  {
    return;
  }
  std::cout << "[igr-linear] event=select"
            << " warm_start=" << (warm_start ? 1 : 0)
            << " space=" << (is_dg ? "dg" : "h1")
            << " spd_assumed=" << (spd_ok ? 1 : 0)
            << " solver=" << LinearSolverTypeName(solver_choice)
            << " precond=" << PrecondTypeName(prec_choice)
            << " rtol=" << lin_rtol << " atol=" << lin_atol
            << " maxit=" << lin_maxit << "\n";
}

} // namespace

// ---------------- IGRSolver::InvRhoCoefficient ----------------
IGRSolver::InvRhoCoefficient::InvRhoCoefficient(const mfem::ParGridFunction& U,
                                                double rho_floor)
  : U_(&U),
    rho_floor_(rho_floor)
{
}

double IGRSolver::InvRhoCoefficient::Eval(mfem::ElementTransformation& T,
                                          const mfem::IntegrationPoint& ip)
{
  MFEM_VERIFY(U_ != nullptr, "InvRhoCoefficient requires a valid state U.");
  T.SetIntPoint(&ip);

  // Conservative state value at (T, ip): state.Size() == num_eq
  mfem::Vector state;
  U_->GetVectorValue(T, ip, state); // [rho, m(0..dim-1), E]
  const double rho = std::max(state(0), rho_floor_);
  return 1.0 / rho;
}

// ---------------- IGRSolver::VelocityFromConsCoefficient ---------------
IGRSolver::VelocityFromConsCoefficient::VelocityFromConsCoefficient(
  int dim,
  const mfem::ParGridFunction* U,
  double rho_floor)
  : mfem::VectorCoefficient(dim),
    U_(U),
    rho_floor_(rho_floor)
{
}

void IGRSolver::VelocityFromConsCoefficient::Eval(
  mfem::Vector& V,
  mfem::ElementTransformation& T,
  const mfem::IntegrationPoint& ip)
{
  MFEM_VERIFY(U_ != nullptr,
              "VelocityFromConsCoefficient requires a valid state U.");
  T.SetIntPoint(&ip);
  mfem::Vector cons;
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
IGRSolver::SigmaSourceCoefficient::SigmaSourceCoefficient(
  const mfem::ParGridFunction& U,
  int dim,
  int num_eq,
  double alpha,
  double rho_floor)
  : U_(&U),
    velH1_(nullptr),
    dim_(dim),
    num_eq_(num_eq),
    alpha_(alpha),
    rho_floor_(rho_floor),
    state_(num_eq),
    gradU_(num_eq, dim),
    gradu_(dim, dim)
{
}

double IGRSolver::SigmaSourceCoefficient::Eval(mfem::ElementTransformation& T,
                                               const mfem::IntegrationPoint& ip)
{
  MFEM_VERIFY(U_ != nullptr,
              "SigmaSourceCoefficient requires a valid state U.");
  T.SetIntPoint(&ip);

  if (velH1_)
  {
    // H1Projection backend: use ∇(projected velocity)
    mfem::DenseMatrix gradv(dim_, dim_);
    velH1_->GetVectorGradient(T, gradv); // gradv(i,j) = ∂_j u_i

    // div u
    double divu = 0.0;
    for (int i = 0; i < dim_; ++i)
    {
      divu += gradv(i, i);
    }

    // tr((∇u)^2) = sum_ij (∂_j u_i)(∂_i u_j)
    double trA2 = 0.0;
    for (int i = 0; i < dim_; ++i)
      for (int j = 0; j < dim_; ++j)
        trA2 += gradv(i, j) * gradv(j, i);

    const double alpha =
      alpha_coeff_ ? alpha_coeff_->Eval(T, ip) : alpha_;
    return alpha * (trA2 + divu * divu);
  }

  // BrokenDG backend: in-cell quotient rule from conservative gradients
  U_->GetVectorValue(T, ip, state_); // [rho, m(0..dim-1), E]
  U_->GetVectorGradient(T, gradU_);  // rows: components, cols: spatial derivs

  const double rho = std::max(state_(0), rho_floor_);

  // momentum components:
  mfem::Vector mom(&state_(1), dim_);

  // grad of rho: row 0 in gradU_
  mfem::Vector grad_rho(dim_);
  for (int j = 0; j < dim_; ++j)
  {
    grad_rho(j) = gradU_(0, j);
  }

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
  for (int i = 0; i < dim_; ++i)
  {
    divu += gradu_(i, i);
  }

  // tr( (∇u)^2 ) = sum_{i,j} (∂_j u_i) (∂_i u_j)
  double trA2 = 0.0;
  for (int i = 0; i < dim_; ++i)
  {
    for (int j = 0; j < dim_; ++j)
    {
      trA2 += gradu_(i, j) * gradu_(j, i);
    }
  }

  // Paper RHS: α ( tr((∇u)^2) + (tr ∇u)^2 )
  const double alpha = alpha_coeff_ ? alpha_coeff_->Eval(T, ip) : alpha_;
  return alpha * (trA2 + divu * divu);
}

double IGRSolver::AlphaPolicyCoefficient::Eval(
  mfem::ElementTransformation& T,
  const mfem::IntegrationPoint&)
{
  if (policy_ref_ != AlphaLengthScalePolicy::Local || T.ElementNo < 0)
  {
    return a_ref_;
  }
  const double h = mesh_.GetElementSize(&T, 1);
  return a_ref_ * h * h;
}

// ---------------- IGRSolver core ----------------
IGRSolver::IGRSolver(mfem::ParFiniteElementSpace& sigma_fes,
                     mfem::ParFiniteElementSpace& state_fes,
                     int dim,
                     int num_eq,
                     double alpha)
  : sfes_(sigma_fes),
    vfes_(state_fes),
    dim_(dim),
    num_eq_(num_eq),
    alpha_(alpha),
    state_scratch_(&vfes_),
    sigma_(&sfes_)
{
  // Allocate RHS form once; bilinear form is built through BuildBilinearForm_().
  b_ = std::make_unique<mfem::ParLinearForm>(&sfes_);

  // Create coefficients once (they hold references; we’ll update their state each call)
  state_scratch_ = 1.0;

  invrho_coef_ =
    std::make_unique<InvRhoCoefficient>(state_scratch_, rho_floor_);
  src_coef_ = std::make_unique<SigmaSourceCoefficient>(state_scratch_,
                                                       dim_,
                                                       num_eq_,
                                                       alpha_,
                                                       rho_floor_);
  alpha_coef_ = std::make_unique<AlphaPolicyCoefficient>(
    *sfes_.GetParMesh(), alpha_, alpha_length_scale_policy_);
  src_coef_->SetAlphaCoefficient(alpha_coef_.get());
  ck_coef_ =
    std::make_unique<mfem::ProductCoefficient>(*alpha_coef_, *invrho_coef_);

  BuildBilinearForm_();

  // RHS: α ( ||∇u||_F^2 + (∇·u)^2 , v )
  b_->AddDomainIntegrator(new mfem::DomainLFIntegrator(*src_coef_));

  // Default BCs: natural Neumann (no flux). For Nitsche Dirichlet, add linear terms later.

  // Initial Σ
  sigma_ = 0.0;

  // Initialize operator handle type we expect (parallel CSR)
  Ah_.SetType(mfem::Operator::Hypre_ParCSR);
}

void IGRSolver::SetAlpha(double a)
{
  alpha_ = a; // alpha_coef reads this by reference (see above)
  if (src_coef_)
  {
    src_coef_->SetAlpha(a);
  }
}

void IGRSolver::SetAlphaLengthScalePolicy(AlphaLengthScalePolicy policy)
{
  alpha_length_scale_policy_ = policy;
}

void IGRSolver::SetSmoother(SmootherType t)
{
  smoot_ = t;
}

void IGRSolver::SetIPKind(IPKind k)
{
  if (ip_kind_ == k)
  {
    return;
  }
  ip_kind_ = k;
  if (a_)
  {
    BuildBilinearForm_();
  }
}

void IGRSolver::SetSweeps(int regular_sweeps, int warm_start_sweeps)
{
  regular_sweeps_ = std::max(0, regular_sweeps);
  warm_start_sweeps_ = std::max(regular_sweeps_, warm_start_sweeps);
}

void IGRSolver::SetRhoFloor(double rho_floor)
{
  rho_floor_ = rho_floor;
  if (invrho_coef_)
  {
    invrho_coef_->SetRhoFloor(rho_floor_);
  }
  if (src_coef_)
  {
    src_coef_->SetRhoFloor(rho_floor_);
  }
}

void IGRSolver::SetDirichletBC(const mfem::Array<int>& ess_bdr)
{
  ess_bdr_.SetSize(ess_bdr.Size());
  ess_bdr_ = ess_bdr;
}

static int GetSpaceOrder_(const mfem::ParFiniteElementSpace& fes)
{
  MFEM_VERIFY(fes.GetNE() > 0, "Empty FES.");
  return fes.GetFE(0)->GetOrder();
}

void IGRSolver::EnsureH1VelocitySpace_()
{
  if (backend_ != GradBackend::H1Projection)
  {
    return;
  }

  mfem::ParMesh* pm = vfes_.GetParMesh();
  MFEM_VERIFY(pm, "Expected mfem::ParMesh for state space.");

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
    h1_vec_fec_ =
      std::make_unique<mfem::H1_FECollection>(p_h1, pm->Dimension());
    h1_vec_fes_ =
      std::make_unique<mfem::ParFiniteElementSpace>(pm,
                                                    h1_vec_fec_.get(),
                                                    /*vdim=*/dim_);
    vel_h1_ = std::make_unique<mfem::ParGridFunction>(h1_vec_fes_.get());
    *vel_h1_ = 0.0;
  }
}

void IGRSolver::BuildBilinearForm_()
{
  a_ = std::make_unique<mfem::ParBilinearForm>(&sfes_);

  // Reaction/mass: (1/ρ σ, v)
  a_->AddDomainIntegrator(new mfem::MassIntegrator(*invrho_coef_));

  // Volume diffusion term: ( (α/ρ) ∇σ, ∇v )
  a_->AddDomainIntegrator(new mfem::DiffusionIntegrator(*ck_coef_));

  // If Σ-space is discontinuous, add IPDG face terms (interior only).
  if (IsDGSpace(sfes_))
  {
    const double sigma = static_cast<double>(ip_kind_); // -1(SIPG), 0, +1
    const double kappa = IPDGPenalty(sfes_);            // penalty scaling
    a_->AddInteriorFaceIntegrator(
      new mfem::DGDiffusionIntegrator(*ck_coef_, sigma, kappa));
    // For the IGR sigma equation, default behavior is natural boundary
    // conditions. Do not add DG boundary diffusion terms unless a specific
    // weak boundary closure is implemented.
  }
}

void IGRSolver::AssembleSystem(const mfem::ParGridFunction& U)
{
  const auto assemble_start = std::chrono::steady_clock::now();

  // Needed for DG/IPDG assembly: variable coeffs evaluated on shared faces
  // will read U on neighbor elements.
  const_cast<mfem::ParGridFunction&>(U).ExchangeFaceNbrData();
  MFEM_VERIFY(VectorIsFinite(U),
              "IGR state contains non-finite values before linear assembly.");

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

    // DG state-derived velocity is generally discontinuous; use arithmetic
    // averaging at shared H1 dofs for partition-independent projection.
    vel_h1_->ProjectDiscCoefficient(vel_coeff, mfem::GridFunction::ARITHMETIC);

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
  src_coef_->SetAlphaCoefficient(alpha_coef_.get());
  src_coef_->SetRhoFloor(rho_floor_);

  // --- IMPORTANT: tell MFEM we will re-use the same mfem::ParBilinearForm again ---
  a_->Update();
  b_->Update();

  // Re-assemble values (sparsity pattern unchanged)
  a_->Assemble();
  b_->Assemble();

  // Rebuild essential dofs if user set Dirichlet BCs (for H1 spaces).
  ess_tdof_list_.SetSize(0);
  if (ess_bdr_.Size() > 0)
  {
    mfem::Array<int> dbdr(ess_bdr_);
    sfes_.GetEssentialTrueDofs(dbdr, ess_tdof_list_);
  }

  const auto assemble_stop = std::chrono::steady_clock::now();
  last_assemble_seconds_ =
    std::chrono::duration<double>(assemble_stop - assemble_start).count();
  total_assemble_seconds_ += last_assemble_seconds_;

  // Rebuild the system operator; clear handle first.
  Ah_.Clear();
  Ah_.SetType(mfem::Operator::Hypre_ParCSR);

  // Warm-start: FormLinearSystem pulls current sigma_ into X_
  const auto form_start = std::chrono::steady_clock::now();
  a_->FormLinearSystem(ess_tdof_list_, sigma_, *b_, Ah_, X_, B_);
  const auto form_stop = std::chrono::steady_clock::now();
  last_form_system_seconds_ =
    std::chrono::duration<double>(form_stop - form_start).count();
  total_form_system_seconds_ += last_form_system_seconds_;
}

void IGRSolver::DoSmootherSweeps(int sweeps, bool /*warm_start*/)
{
  mfem::HypreParMatrix* A_hypre = Ah_.As<mfem::HypreParMatrix>();
  MFEM_VERIFY(A_hypre != nullptr,
              "IGR: Assembled operator is not mfem::HypreParMatrix.");

  mfem::HypreSmoother S(*A_hypre);
  switch (smoot_)
  {
  case SmootherType::Jacobi:
    S.SetType(mfem::HypreSmoother::Jacobi);
    break;
  case SmootherType::L1Jacobi:
    S.SetType(mfem::HypreSmoother::l1Jacobi);
    break;
  case SmootherType::L1GS:
    S.SetType(mfem::HypreSmoother::l1GS);
    break;
  case SmootherType::L1GStr:
    S.SetType(mfem::HypreSmoother::l1GStr);
    break;
  case SmootherType::GS:
    S.SetType(mfem::HypreSmoother::GS);
    break;
  }
  S.SetPositiveDiagonal(true);

  mfem::Vector r(B_.Size()), z(X_.Size());
  for (int it = 0; it < sweeps; ++it)
  {
    // r = b - A x
    r = B_;
    A_hypre->Mult(X_, r);
    r.Neg();
    r += B_;

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
  mfem::HypreParMatrix* A = Ah_.As<mfem::HypreParMatrix>();
  MFEM_VERIFY(A != nullptr,
              "IGR: Assembled operator is not mfem::HypreParMatrix.");

  MPI_Comm comm = sfes_.GetComm();
  const bool is_dg = IsDGSpace(sfes_);
  const bool spd_ok = IsSPD_DefaultAssumption(sfes_, ip_kind_);
  const bool prev_linear_converged = last_linear_converged_;
  const double prev_linear_final_norm = last_linear_final_norm_;

  last_linear_iters_ = 0;
  last_linear_converged_ = false;
  last_linear_final_norm_ = std::numeric_limits<double>::quiet_NaN();
  last_linear_initial_norm_ = std::numeric_limits<double>::quiet_NaN();
  last_linear_solve_seconds_ = 0.0;
  last_pc_setup_seconds_ = 0.0;
  last_linear_solver_name_ = "none";
  last_precond_name_ = "none";

  // --- Choose solver (Auto policy is space-aware) ---
  const LinearSolverType requested_solver = lin_solver_type_;
  LinearSolverType solver_choice = lin_solver_type_;
  if (solver_choice == LinearSolverType::Auto)
  {
    solver_choice = spd_ok ? LinearSolverType::CG : LinearSolverType::GMRES;
  }

  // Guardrail: do not use SPD-only Krylov methods when SPD assumptions fail.
  if (RequiresSPD(solver_choice) && !spd_ok)
  {
    const LinearSolverType prev = solver_choice;
    solver_choice = LinearSolverType::GMRES;
    LogIGRFallback("operator-not-spd",
                   prev,
                   solver_choice,
                   prec_type_,
                   prec_type_);
  }

  // --- Choose preconditioner (Auto policy is space-aware) ---
  const PrecondType requested_prec = prec_type_;
  PrecondType prec_choice = prec_type_;
  if (prec_choice == PrecondType::Auto)
  {
    // H1 and DG/L2: BoomerAMG is the robust default for this operator family.
    // Keep other preconditioners available via runtime override.
    (void)is_dg;
    prec_choice = PrecondType::BoomerAMG;
  }

  // Guardrail: CG/MINRES should use symmetric/no preconditioners.
  if (RequiresSPD(solver_choice) && !IsSymmetricPC(prec_choice))
  {
    const PrecondType prev = prec_choice;
    prec_choice = PrecondType::BoomerAMG;
    LogIGRFallback("spd-solver-with-nonsymmetric-preconditioner",
                   requested_solver,
                   solver_choice,
                   prev,
                   prec_choice);
  }

  const auto pc_setup_start = std::chrono::steady_clock::now();
  std::unique_ptr<mfem::Solver> prec;
  if (prec_choice == PrecondType::BoomerAMG)
  {
    auto amg = std::make_unique<mfem::HypreBoomerAMG>(*A);
    // Typical AMG-as-preconditioner settings
    amg->SetPrintLevel(0);
    amg->SetMaxIter(1);
    amg->SetTol(0.0);
    if (amg_coarsen_type_ >= 0)
    {
      amg->SetCoarsening(amg_coarsen_type_);
    }
    if (amg_relax_type_ >= 0)
    {
      amg->SetRelaxType(amg_relax_type_);
    }
    if (amg_max_levels_ >= 0)
    {
      amg->SetMaxLevels(amg_max_levels_);
    }
    if (amg_strong_threshold_ >= 0.0)
    {
      amg->SetStrengthThresh(amg_strong_threshold_);
    }
    prec = std::move(amg);
  }
  else if (prec_choice == PrecondType::HypreILU)
  {
    auto ilu = std::make_unique<mfem::HypreILU>();
    ilu->SetPrintLevel(0);
    ilu->SetMaxIter(1);
    ilu->SetTol(0.0);
    ilu->SetOperator(*A);
    prec = std::move(ilu);
  }
  else if (prec_choice == PrecondType::HypreEuclid)
  {
    auto euclid = std::make_unique<mfem::HypreEuclid>(*A);
    euclid->SetStats(0);
    prec = std::move(euclid);
  }
  else if (prec_choice == PrecondType::ParaSails)
  {
    auto parasails = std::make_unique<mfem::HypreParaSails>(*A);
    parasails->SetLogging(0);
    prec = std::move(parasails);
  }
  else if (prec_choice == PrecondType::HypreSmoother)
  {
    auto sm = std::make_unique<mfem::HypreSmoother>(*A);
    switch (smoot_)
    {
    case SmootherType::Jacobi:
      sm->SetType(mfem::HypreSmoother::Jacobi);
      break;
    case SmootherType::L1Jacobi:
      sm->SetType(mfem::HypreSmoother::l1Jacobi);
      break;
    case SmootherType::L1GS:
      sm->SetType(mfem::HypreSmoother::l1GS);
      break;
    case SmootherType::L1GStr:
      sm->SetType(mfem::HypreSmoother::l1GStr);
      break;
    case SmootherType::GS:
      sm->SetType(mfem::HypreSmoother::GS);
      break;
    }
    sm->SetPositiveDiagonal(true);
    prec = std::move(sm);
  }
  // else None -> leave prec null
  const auto pc_setup_stop = std::chrono::steady_clock::now();
  last_pc_setup_seconds_ =
    std::chrono::duration<double>(pc_setup_stop - pc_setup_start).count();
  total_pc_setup_seconds_ += last_pc_setup_seconds_;

  if (lin_print_ > 0)
  {
    LogIGRSelection(warm_start,
                    is_dg,
                    spd_ok,
                    solver_choice,
                    prec_choice,
                    lin_rtol_,
                    lin_atol_,
                    lin_maxit_);
  }

  // Warm start is only safe if the prior solve produced a usable iterate.
  // If not, fall back to a cold-start solve to avoid poisoning Krylov with
  // stale/non-finite data.
  bool enable_warm_start = warm_start;
  if (enable_warm_start && linear_solve_calls_ > 0)
  {
    if (!prev_linear_converged || !std::isfinite(prev_linear_final_norm))
    {
      enable_warm_start = false;
      if (lin_print_ > 0)
      {
        LogIGRFallback("warm-start-disabled-after-unsuccessful-previous-solve",
                       solver_choice,
                       solver_choice,
                       prec_choice,
                       prec_choice);
      }
    }
  }
  if (!enable_warm_start || !VectorIsFinite(X_))
  {
    X_ = 0.0;
    enable_warm_start = false;
  }
  MFEM_VERIFY(VectorIsFinite(B_),
              "IGR RHS contains non-finite values before Krylov solve.");

  const bool is_intermediate_stage = (rk_stage_index_ >= 0 && rk_stage_index_ < 2);
  const double effective_rtol =
    is_intermediate_stage ? (lin_rtol_ * stage_rtol_scale_) : lin_rtol_;

  auto configure_solver = [&](mfem::IterativeSolver& solver)
  {
    solver.SetOperator(*A);
    if (prec)
    {
      solver.SetPreconditioner(*prec);
    }
    solver.SetRelTol(effective_rtol);
    solver.SetAbsTol(lin_atol_);
    solver.SetMaxIter(lin_maxit_);
    solver.SetPrintLevel(lin_print_);
    solver.iterative_mode = enable_warm_start;
  };

  const auto solve_start = std::chrono::steady_clock::now();

  if (solver_choice == LinearSolverType::CG)
  {
    mfem::CGSolver solver(comm);
    configure_solver(solver);
    solver.Mult(B_, X_);
    last_linear_iters_ = solver.GetNumIterations();
    last_linear_converged_ = solver.GetConverged();
    last_linear_final_norm_ = solver.GetFinalNorm();
    last_linear_initial_norm_ = solver.GetInitialNorm();
  }
  else if (solver_choice == LinearSolverType::GMRES)
  {
    mfem::GMRESSolver solver(comm);
    configure_solver(solver);
    solver.Mult(B_, X_);
    last_linear_iters_ = solver.GetNumIterations();
    last_linear_converged_ = solver.GetConverged();
    last_linear_final_norm_ = solver.GetFinalNorm();
    last_linear_initial_norm_ = solver.GetInitialNorm();
  }
  else if (solver_choice == LinearSolverType::FGMRES)
  {
    mfem::FGMRESSolver solver(comm);
    configure_solver(solver);
    solver.Mult(B_, X_);
    last_linear_iters_ = solver.GetNumIterations();
    last_linear_converged_ = solver.GetConverged();
    last_linear_final_norm_ = solver.GetFinalNorm();
    last_linear_initial_norm_ = solver.GetInitialNorm();
  }
  else if (solver_choice == LinearSolverType::BiCGSTAB)
  {
    mfem::BiCGSTABSolver solver(comm);
    configure_solver(solver);
    solver.Mult(B_, X_);
    last_linear_iters_ = solver.GetNumIterations();
    last_linear_converged_ = solver.GetConverged();
    last_linear_final_norm_ = solver.GetFinalNorm();
    last_linear_initial_norm_ = solver.GetInitialNorm();
  }
  else if (solver_choice == LinearSolverType::MINRES)
  {
    mfem::MINRESSolver solver(comm);
    configure_solver(solver);
    solver.Mult(B_, X_);
    last_linear_iters_ = solver.GetNumIterations();
    last_linear_converged_ = solver.GetConverged();
    last_linear_final_norm_ = solver.GetFinalNorm();
    last_linear_initial_norm_ = solver.GetInitialNorm();
  }
  else
  {
    MFEM_ABORT("Unsupported IGR linear solver selection.");
  }

  const auto solve_stop = std::chrono::steady_clock::now();
  last_linear_solve_seconds_ =
    std::chrono::duration<double>(solve_stop - solve_start).count();
  total_linear_solve_seconds_ += last_linear_solve_seconds_;
  ++linear_solve_calls_;
  total_linear_iters_ += last_linear_iters_;
  if (last_linear_converged_)
  {
    ++total_linear_converged_calls_;
  }

  last_linear_solver_name_ = LinearSolverTypeName(solver_choice);
  last_precond_name_ = PrecondTypeName(prec_choice);

  a_->RecoverFEMSolution(X_, *b_, sigma_);
}

void IGRSolver::UpdateFromState(const mfem::ParGridFunction& U, bool warm_start)
{
  // Assemble A and b for current state U (warm-start handled by FormLinearSystem)
  AssembleSystem(U);

  if (solve_mode_ == SolveMode::SmootherSweeps)
  {
    last_linear_iters_ = 0;
    last_linear_converged_ = true;
    last_linear_initial_norm_ = std::numeric_limits<double>::quiet_NaN();
    last_linear_final_norm_ = std::numeric_limits<double>::quiet_NaN();
    last_linear_solve_seconds_ = 0.0;
    last_pc_setup_seconds_ = 0.0;
    last_linear_solver_name_ = "smoother";
    last_precond_name_ = PrecondTypeName(PrecondType::HypreSmoother);

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

} // namespace dg_igr

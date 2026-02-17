#include "physics/igr/IGR.hpp"
#include "utils/config/env_flags.hpp"
#include "IGR_internal_utils.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

namespace dg_igr
{
using namespace igr_internal;
void IGRSolver::BuildBilinearForm_()
{
  const bool debug_qf =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=build-bilinear-start" << "\n";
  }
  a_ = std::make_unique<mfem::ParBilinearForm>(&sfes_);
  use_partial_assembly_ = ShouldUsePartialAssembly_();
  if (use_partial_assembly_)
  {
    a_->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL);
  }

  const bool use_device_qf = UseDeviceCoefficients_();
  using_device_coeffs_ = use_device_qf;
  if (use_device_qf)
  {
    EnsureDeviceQuadrature_();
  }

  mfem::Coefficient* invrho = nullptr;
  mfem::Coefficient* ck = nullptr;
  mfem::Coefficient* src = nullptr;
  if (use_device_qf)
  {
    invrho = invrho_qf_coef_.get();
    ck = ck_qf_coef_.get();
    src = src_qf_coef_.get();
  }
  else
  {
    invrho = invrho_coef_.get();
    ck = ck_coef_.get();
    src = src_coef_.get();
  }

  // Reaction/mass: (1/ρ σ, v)
  auto* mass_integ = new mfem::MassIntegrator(*invrho);
  if (use_device_qf && sigma_qs_)
  {
    mass_integ->SetIntRule(&sigma_qs_->GetIntRule(0));
  }
  a_->AddDomainIntegrator(mass_integ);

  // Volume diffusion term: ( (α/ρ) ∇σ, ∇v )
  auto* diff_integ = new mfem::DiffusionIntegrator(*ck);
  if (use_device_qf && sigma_qs_)
  {
    diff_integ->SetIntRule(&sigma_qs_->GetIntRule(0));
  }
  a_->AddDomainIntegrator(diff_integ);

  // If Σ-space is discontinuous, add IPDG face terms (interior only).
  if (IsDGSpace(sfes_))
  {
    if (debug_qf && mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=build-bilinear-dg-face"
                << " sigma_fqs=" << (sigma_fqs_ ? 1 : 0)
                << " ck_face_qf_coef=" << (ck_face_qf_coef_ ? 1 : 0)
                << std::endl;
    }
    const double sigma = static_cast<double>(ip_kind_); // -1(SIPG), 0, +1
    const double kappa = IPDGPenalty(sfes_);            // penalty scaling
    mfem::Coefficient* ck_face =
      use_device_qf ? ck_face_qf_coef_.get() : ck;
    auto* dg_diff = new mfem::DGDiffusionIntegrator(*ck_face, sigma, kappa);
    if (use_device_qf && sigma_fqs_)
    {
      dg_diff->SetIntRule(&sigma_fqs_->GetIntRule(0));
    }
    a_->AddInteriorFaceIntegrator(dg_diff);
    // For the IGR sigma equation, default behavior is natural boundary
    // conditions. Do not add DG boundary diffusion terms unless a specific
    // weak boundary closure is implemented.
  }

  b_ = std::make_unique<mfem::ParLinearForm>(&sfes_);
  auto* rhs_integ = new mfem::DomainLFIntegrator(*src);
  if (use_device_qf && sigma_qs_)
  {
    rhs_integ->SetIntRule(&sigma_qs_->GetIntRule(0));
  }
  b_->AddDomainIntegrator(rhs_integ);
  b_->UseFastAssembly(use_partial_assembly_);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=build-bilinear-done" << "\n";
  }
}

bool IGRSolver::ShouldUsePartialAssembly_() const
{
  const bool device_backend =
    mfem::Device::Allows(mfem::Backend::DEVICE_MASK);
  if (device_backend)
  {
    const bool requested =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_PARTIAL_ASSEMBLY", true);
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=partial-assembly-policy"
                << " requested=" << (requested ? "on" : "off")
                << " effective=on"
                << " reason=device-required-partial-assembly"
                << " precond=" << PrecondTypeName(prec_type_) << "\n";
    }
    return true;
  }

  bool use_partial =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_PARTIAL_ASSEMBLY", false);
  if (use_partial && PreconditionerRequiresAssembledMatrix(prec_type_))
  {
    if (mfem::Mpi::Root())
    {
      std::cout << "[igr-linear] event=partial-assembly-policy"
                << " requested=on"
                << " effective=off"
                << " reason=matrix-preconditioner"
                << " precond=" << PrecondTypeName(prec_type_) << "\n";
    }
    use_partial = false;
  }
  return use_partial;
}

void IGRSolver::AssembleSystem(const mfem::ParGridFunction& U)
{
  const auto assemble_start = std::chrono::steady_clock::now();

  // Needed for DG/IPDG assembly: variable coeffs evaluated on shared faces
  // will read U on neighbor elements.
  // For H1 sigma spaces, no DG/IP face terms are assembled, so this exchange is
  // unnecessary. In serial runs, neighbor exchange is also a no-op with extra
  // overhead; skip it in that case as well.
  if (IsDGSpace(sfes_) && mfem::Mpi::WorldSize() > 1)
  {
    const_cast<mfem::ParGridFunction&>(U).ExchangeFaceNbrData();
  }
  MFEM_VERIFY(VectorIsFinite(U),
              "IGR state contains non-finite values before linear assembly.");

  const bool device_backend = mfem::Device::Allows(mfem::Backend::DEVICE_MASK);
  const bool use_device_qf = UseDeviceCoefficients_();
  const bool debug_qf =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_DEBUG_QF", false);
  if (debug_qf && mfem::Mpi::Root())
  {
    std::cout << "[igr-linear] event=assemble-use-device-qf"
              << " enabled=" << (use_device_qf ? 1 : 0) << "\n";
  }
  if (use_device_qf != using_device_coeffs_)
  {
    BuildBilinearForm_();
  }

  if (use_device_qf)
  {
    UpdateDeviceCoefficients_(U);
  }
  else
  {
    // Ensure host-visible state for coefficient evaluation.
    // The IGR coefficients call GridFunction::GetVectorValue(), which is host-only.
    // On device backends, default to syncing unless explicitly disabled.
    const bool hostsync_default = device_backend;
    if (::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_FORCE_STATE_HOSTSYNC",
                                        hostsync_default))
    {
      (void)U.HostRead();
    }

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
  }

  // Keep MFEM form internals consistent before re-assembly across policy modes.
  a_->Update();
  b_->Update();
  b_->UseFastAssembly(use_partial_assembly_);

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
  Ah_.SetType(use_partial_assembly_ ? mfem::Operator::ANY_TYPE
                                    : mfem::Operator::Hypre_ParCSR);

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
  mfem::Operator* A_op = Ah_.Ptr();
  MFEM_VERIFY(A_op != nullptr,
              "IGR: linear operator is not assembled before smoother sweeps.");
  mfem::HypreParMatrix* A_hypre = Ah_.Is<mfem::HypreParMatrix>();

  std::unique_ptr<mfem::Solver> smoother;
  if (A_hypre)
  {
    auto hypre_sm = std::make_unique<mfem::HypreSmoother>(*A_hypre);
    switch (smoot_)
    {
    case SmootherType::Jacobi:
      hypre_sm->SetType(mfem::HypreSmoother::Jacobi);
      break;
    case SmootherType::L1Jacobi:
      hypre_sm->SetType(mfem::HypreSmoother::l1Jacobi);
      break;
    case SmootherType::L1GS:
      hypre_sm->SetType(mfem::HypreSmoother::l1GS);
      break;
    case SmootherType::L1GStr:
      hypre_sm->SetType(mfem::HypreSmoother::l1GStr);
      break;
    case SmootherType::GS:
      hypre_sm->SetType(mfem::HypreSmoother::GS);
      break;
    }
    hypre_sm->SetPositiveDiagonal(true);
    smoother = std::move(hypre_sm);
  }
  else
  {
    auto jac = std::make_unique<mfem::OperatorJacobiSmoother>(*a_,
                                                               ess_tdof_list_);
    jac->SetPositiveDiagonal(true);
    smoother = std::move(jac);
  }

  mfem::Vector r(B_.Size()), z(X_.Size());
  for (int it = 0; it < sweeps; ++it)
  {
    // r = b - A x
    r = B_;
    A_op->Mult(X_, r);
    r.Neg();
    r += B_;

    // z ≈ M^{-1} r  (one application of the smoother/preconditioner)
    z = 0.0;
    smoother->Mult(r, z);

    // x <- x + z  (Richardson step)
    X_ += z;
  }

  a_->RecoverFEMSolution(X_, *b_, sigma_);
}

void IGRSolver::SolveEllipticSystem_(bool warm_start)
{
  mfem::Operator* A_op = Ah_.Ptr();
  MFEM_VERIFY(A_op != nullptr,
              "IGR: linear operator is not assembled before Krylov solve.");
  mfem::HypreParMatrix* A_hypre = Ah_.Is<mfem::HypreParMatrix>();
  const bool has_hypre_matrix = (A_hypre != nullptr);

  MPI_Comm comm = sfes_.GetComm();
  const bool is_dg = IsDGSpace(sfes_);
  const bool spd_ok = IsSPD_DefaultAssumption(sfes_, ip_kind_);
  const bool using_cuda = IsCudaBackendActive();
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
  PrecondType prec_choice = prec_type_;
  if (prec_choice == PrecondType::Auto)
  {
    // CPU: keep BoomerAMG as robust default.
    // CUDA: prefer HypreSmoother for lower setup/solve overhead on current
    // inoue-style runs; fallback policy remains active for unsupported paths.
    (void)is_dg;
    prec_choice =
      using_cuda ? PrecondType::HypreSmoother : PrecondType::BoomerAMG;
  }

  // Guardrail: CG/MINRES should use symmetric/no preconditioners.
  if (RequiresSPD(solver_choice) && !IsSymmetricPC(prec_choice))
  {
    if (using_cuda)
    {
      MFEM_VERIFY(false,
                  "IGR device-closure requires symmetric preconditioning for "
                  "SPD Krylov solvers; requested preconditioner is incompatible.");
    }
    const PrecondType prev = prec_choice;
    prec_choice = PrecondType::BoomerAMG;
    LogIGRFallback("spd-solver-with-nonsymmetric-preconditioner",
                   requested_solver,
                   solver_choice,
                   prev,
                   prec_choice);
  }

  if (using_cuda)
  {
    MFEM_VERIFY(!has_hypre_matrix,
                "IGR device-closure requires partial-assembly operator path "
                "(no assembled Hypre matrix).");
    MFEM_VERIFY(prec_choice == PrecondType::HypreSmoother ||
                  prec_choice == PrecondType::None,
                "IGR device-closure on CUDA supports only "
                "--igr-preconditioner hypresmoother|none.");
  }
  else if (!has_hypre_matrix &&
           (prec_choice == PrecondType::BoomerAMG ||
            prec_choice == PrecondType::HypreILU ||
            prec_choice == PrecondType::HypreEuclid ||
            prec_choice == PrecondType::ParaSails))
  {
    const PrecondType prev = prec_choice;
    prec_choice = PrecondType::HypreSmoother;
    LogIGRFallback("operator-path-without-hypre-matrix",
                   requested_solver,
                   solver_choice,
                   prev,
                   prec_choice);
  }

  const auto pc_setup_start = std::chrono::steady_clock::now();
  std::unique_ptr<mfem::Solver> prec;
  bool using_operator_jacobi = false;
  if (prec_choice == PrecondType::BoomerAMG)
  {
    MFEM_VERIFY(A_hypre != nullptr,
                "BoomerAMG preconditioner requires a Hypre matrix operator.");
    auto amg = std::make_unique<mfem::HypreBoomerAMG>(*A_hypre);
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
    MFEM_VERIFY(A_hypre != nullptr,
                "HypreILU preconditioner requires a Hypre matrix operator.");
    auto ilu = std::make_unique<mfem::HypreILU>();
    ilu->SetPrintLevel(0);
    ilu->SetMaxIter(1);
    ilu->SetTol(0.0);
    ilu->SetOperator(*A_hypre);
    prec = std::move(ilu);
  }
  else if (prec_choice == PrecondType::HypreEuclid)
  {
    MFEM_VERIFY(A_hypre != nullptr,
                "HypreEuclid preconditioner requires a Hypre matrix operator.");
    auto euclid = std::make_unique<mfem::HypreEuclid>(*A_hypre);
    euclid->SetStats(0);
    prec = std::move(euclid);
  }
  else if (prec_choice == PrecondType::ParaSails)
  {
    MFEM_VERIFY(A_hypre != nullptr,
                "ParaSails preconditioner requires a Hypre matrix operator.");
    auto parasails = std::make_unique<mfem::HypreParaSails>(*A_hypre);
    parasails->SetLogging(0);
    prec = std::move(parasails);
  }
  else if (prec_choice == PrecondType::HypreSmoother)
  {
    if (A_hypre)
    {
      auto sm = std::make_unique<mfem::HypreSmoother>(*A_hypre);
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
    else
    {
      auto jac = std::make_unique<mfem::OperatorJacobiSmoother>(*a_,
                                                                 ess_tdof_list_);
      jac->SetPositiveDiagonal(true);
      prec = std::move(jac);
      using_operator_jacobi = true;
    }
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
                    using_cuda,
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
    solver.SetOperator(*A_op);
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
  last_precond_name_ =
    using_operator_jacobi ? "operatorjacobi" : PrecondTypeName(prec_choice);

  a_->RecoverFEMSolution(X_, *b_, sigma_);
}

} // namespace dg_igr

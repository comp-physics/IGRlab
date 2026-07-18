#include "physics/igr/IGR.hpp"
#include "utils/config/env_flags.hpp"
#include "IGR_internal_utils.hpp"
#include "utils/profiling/nvtx.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

namespace dg_igr
{
using namespace igr_internal;

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

    const double alpha = alpha_coeff_ ? alpha_coeff_->Eval(T, ip) : alpha_;
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

void IGRSolver::SetPreconditioner(PrecondType p)
{
  if (prec_type_ == p)
  {
    return;
  }
  prec_type_ = p;
  // Assembly level policy depends on the selected preconditioner.
  if (a_)
  {
    BuildBilinearForm_();
  }
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

void IGRSolver::SetExactDGFaceOperator(bool enabled)
{
  if (exact_dg_face_operator_ == enabled)
  {
    return;
  }
  exact_dg_face_operator_ = enabled;
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

bool IGRSolver::UseDeviceCoefficients_() const
{
  const bool device_backend =
    mfem::Device::Allows(mfem::Backend::DEVICE_MASK);
  const bool qf_enabled =
    ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_DEVICE_QF", device_backend);
  if (!qf_enabled)
  {
    return false;
  }
  if (!device_backend && !IsDGSpace(sfes_))
  {
    // The CPU opt-in exists to build an L2/DG PA reference that avoids
    // face-time GetVectorValue() on discontinuous state fields.
    return false;
  }
  return true;
}

void IGRSolver::AddOutputOnlyAccounting_(const SolveAccountingDelta& delta)
{
  output_only_linear_solve_calls_ += delta.calls;
  output_only_linear_iters_ += delta.iterations;
  output_only_linear_converged_calls_ += delta.converged_calls;
  output_only_linear_solve_seconds_ += delta.solve_seconds;
  output_only_assemble_seconds_ += delta.assemble_seconds;
  output_only_form_system_seconds_ += delta.form_system_seconds;
  output_only_pc_setup_seconds_ += delta.pc_setup_seconds;
}

void IGRSolver::ConsumePendingStage0PrefetchAccounting()
{
  pending_stage0_prefetch_accounting_ = {};
}

bool IGRSolver::ReclassifyPendingStage0PrefetchAsOutputOnly(
  bool restore_warm_start)
{
  if (!pending_stage0_prefetch_accounting_.valid)
  {
    return false;
  }
  if (restore_warm_start)
  {
    MFEM_VERIFY(
      pending_stage0_prefetch_sigma_checkpoint_.Size() == sigma_.Size(),
      "Pending stage-0 prefetch has no valid sigma checkpoint.");
    sigma_ = pending_stage0_prefetch_sigma_checkpoint_;
    first_call_ = pending_stage0_prefetch_first_call_;
    last_linear_converged_ = pending_stage0_prefetch_last_converged_;
    last_linear_final_norm_ = pending_stage0_prefetch_last_final_norm_;
  }
  AddOutputOnlyAccounting_(pending_stage0_prefetch_accounting_);
  pending_stage0_prefetch_accounting_ = {};
  return true;
}

void IGRSolver::UpdateFromState(const mfem::ParGridFunction& U,
                                bool warm_start,
                                IGRSolvePurpose purpose,
                                bool certify_residual)
{
  MFEM_VERIFY(!pending_stage0_prefetch_accounting_.valid,
              "A stage-0 sigma prefetch must be consumed or reclassified "
              "before another IGR solve.");
  if (purpose == IGRSolvePurpose::ProductionStage0Prefetch)
  {
    pending_stage0_prefetch_sigma_checkpoint_ = sigma_;
    pending_stage0_prefetch_first_call_ = first_call_;
    pending_stage0_prefetch_last_converged_ = last_linear_converged_;
    pending_stage0_prefetch_last_final_norm_ = last_linear_final_norm_;
  }
  ::dg_igr::NVTXRange range("IGR/update-from-state");
  const auto update_start = std::chrono::steady_clock::now();
  const int calls_before = linear_solve_calls_;
  const int iters_before = total_linear_iters_;
  const int converged_before = total_linear_converged_calls_;
  const double solve_before = total_linear_solve_seconds_;
  const double assemble_before = total_assemble_seconds_;
  const double form_before = total_form_system_seconds_;
  const double pc_before = total_pc_setup_seconds_;
  current_solve_purpose_ = purpose;
  certify_current_solve_ = certify_residual;
  diagnostic_state_ = &U;

  // Assemble A and b for current state U (warm-start handled by FormLinearSystem)
  AssembleSystem(U);

  if (solve_mode_ == SolveMode::SmootherSweeps)
  {
    MFEM_VERIFY(!certify_current_solve_,
                "Independent restart residual certification requires a "
                "Krylov IGR solve.");
    last_linear_iters_ = 0;
    last_linear_converged_ = true;
    last_linear_initial_norm_ = std::numeric_limits<double>::quiet_NaN();
    last_linear_final_norm_ = std::numeric_limits<double>::quiet_NaN();
    last_linear_true_residual_abs_ =
      std::numeric_limits<double>::quiet_NaN();
    last_linear_true_residual_rel_rhs_ =
      std::numeric_limits<double>::quiet_NaN();
    last_linear_true_initial_residual_abs_ =
      std::numeric_limits<double>::quiet_NaN();
    last_linear_true_residual_rel_initial_ =
      std::numeric_limits<double>::quiet_NaN();
    last_linear_recomputed_preconditioned_initial_norm_ =
      std::numeric_limits<double>::quiet_NaN();
    last_linear_recomputed_preconditioned_final_norm_ =
      std::numeric_limits<double>::quiet_NaN();
    last_linear_solve_seconds_ = 0.0;
    last_pc_setup_seconds_ = 0.0;
    last_chebyshev_diagonal_seconds_ = 0.0;
    last_chebyshev_setup_seconds_ = 0.0;
    last_chebyshev_eigenvalue_reused_ = false;
    last_chebyshev_smoother_reused_ = false;
    last_chebyshev_eigenvalue_estimate_ = 0.0;
    last_krylov_operator_apply_seconds_ = 0.0;
    last_krylov_preconditioner_apply_seconds_ = 0.0;
    last_krylov_preconditioner_operator_apply_seconds_ = 0.0;
    last_krylov_operator_apply_count_ = 0;
    last_krylov_preconditioner_apply_count_ = 0;
    last_krylov_preconditioner_operator_apply_count_ = 0;
    last_linear_solver_name_ = "smoother";
    last_precond_name_ = Ah_.Is<mfem::HypreParMatrix>() ? "hypresmoother"
                                                         : "operatorjacobi";

    const int sweeps = (first_call_ && warm_start)
                         ? std::max(warm_start_sweeps_, regular_sweeps_)
                         : regular_sweeps_;
    DoSmootherSweeps(sweeps, warm_start);
  }
  else
  {
    SolveEllipticSystem_(warm_start);
  }

  certify_current_solve_ = false;
  first_call_ = false;
  diagnostic_state_ = nullptr;

  const auto update_stop = std::chrono::steady_clock::now();
  last_update_call_seconds_ =
    std::chrono::duration<double>(update_stop - update_start).count();
  total_update_call_seconds_ += last_update_call_seconds_;

  SolveAccountingDelta delta;
  delta.valid = true;
  delta.calls = linear_solve_calls_ - calls_before;
  delta.iterations = total_linear_iters_ - iters_before;
  delta.converged_calls = total_linear_converged_calls_ - converged_before;
  delta.solve_seconds = total_linear_solve_seconds_ - solve_before;
  delta.assemble_seconds = total_assemble_seconds_ - assemble_before;
  delta.form_system_seconds = total_form_system_seconds_ - form_before;
  delta.pc_setup_seconds = total_pc_setup_seconds_ - pc_before;

  if (purpose == IGRSolvePurpose::OutputOnly)
  {
    AddOutputOnlyAccounting_(delta);
  }
  else if (purpose == IGRSolvePurpose::ProductionStage0Prefetch)
  {
    pending_stage0_prefetch_accounting_ = delta;
  }
}

} // namespace dg_igr

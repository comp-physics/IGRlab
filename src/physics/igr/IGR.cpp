#include "igrlab/compat.hpp"
#include "physics/igr/IGR.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "utils/config/env_flags.hpp"
#include "IGR_internal_utils.hpp"
#include "IGR_nodal_source.hpp"
#include "utils/profiling/nvtx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace igrlab
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
  return frozen_scalar_igr_algebra::SafeInverseDensity(state(0), rho_floor_);
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
  V.SetSize(vdim);
  for (int d = 0; d < vdim; ++d)
  {
    const double md = cons(1 + d);
    V(d) = frozen_scalar_igr_algebra::ConservativeVelocityComponent(md,
                                                                    cons(0),
                                                                    rho_floor_);
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

    const double alpha = alpha_coeff_ ? alpha_coeff_->Eval(T, ip) : alpha_;
    return frozen_scalar_igr_algebra::SourceFromVelocityGradient(
      dim_,
      gradv.GetData(),
      1,
      gradv.Height(),
      alpha);
  }

  // BrokenDG backend: in-cell quotient rule from conservative gradients
  U_->GetVectorValue(T, ip, state_); // [rho, m(0..dim-1), E]
  U_->GetVectorGradient(T, gradU_);  // rows: components, cols: spatial derivs

  const double alpha = alpha_coeff_ ? alpha_coeff_->Eval(T, ip) : alpha_;
  return frozen_scalar_igr_algebra::SourceFromConservativeStateGradient(
    dim_,
    state_.GetData(),
    1,
    gradU_.GetData(),
    1,
    gradU_.Height(),
    alpha,
    rho_floor_);
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
  MFEM_VERIFY(((!element_jacobi_tensor_blocks_ || !element_jacobi_blocks_) &&
                 !local_tensor_preconditioner_) || a == alpha_,
              "Cannot change alpha with live tensor element-Jacobi factors.");
  alpha_ = a; // alpha_coef reads this by reference (see above)
  if (src_coef_)
  {
    src_coef_->SetAlpha(a);
  }
}

void IGRSolver::SetAlphaLengthScalePolicy(AlphaLengthScalePolicy policy)
{
  MFEM_VERIFY(((!element_jacobi_tensor_blocks_ || !element_jacobi_blocks_) &&
                 !local_tensor_preconditioner_) ||
                policy == alpha_length_scale_policy_,
              "Cannot change alpha length-scale policy with live tensor element-Jacobi factors.");
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

void IGRSolver::SetSubcellQuadratureMinimumOrder(int order)
{
  MFEM_VERIFY(order == 0 || order == 4,
              "Subcell IGR quadrature minimum order must be zero or four.");
  MFEM_VERIFY(order == 0 || (mfem::Device::Allows(mfem::Backend::CUDA_MASK) && UseDeviceCoefficients_()),
              "Subcell IGR quadrature floor requires CUDA current coefficient QFunctions.");
  if (order == subcell_quadrature_minimum_order_) return;
  MFEM_VERIFY(first_call_, "Configure subcell quadrature before the first IGR state update.");
  subcell_quadrature_minimum_order_ = order;
  BuildBilinearForm_();
}

void IGRSolver::SetSweeps(int regular_sweeps, int warm_start_sweeps)
{
  regular_sweeps_ = std::max(0, regular_sweeps);
  warm_start_sweeps_ = std::max(regular_sweeps_, warm_start_sweeps);
}

void IGRSolver::SetElementJacobiDamping(double omega)
{
  MFEM_VERIFY(std::isfinite(omega) && omega > 0.0 && omega <= 1.0,
              "Element-block Jacobi damping must be finite and in (0, 1].");
  element_jacobi_omega_ = omega;
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
  const bool nodal_source = GetNodalSourceMode() != NodalSourceMode::Baseline;
  MFEM_VERIFY(!NodalSourceMirrorWalls() || GetNodalSourceMode() == NodalSourceMode::Central,
              "DG_IGR_IGR_NODAL_WALL_MIRROR requires the central nodal source.");
  MFEM_VERIFY((!nodal_source && NodalSourceQuadratureMinimumOrder() == 0 &&
                OperatorQuadratureMinimumOrder() == 0) || IsDGSpace(sfes_),
              "Nodal IGR source / coefficient-quadrature override requires an L2 scalar space.");
  const bool device_backend =
    mfem::Device::Allows(mfem::Backend::DEVICE_MASK);
  // The exact side-aware DG face operator consumes the element coefficient
  // QFunctions on every backend; it is therefore a typed dependency, not a
  // hidden environment prerequisite.  The environment switch remains an
  // opt-in/opt-out control only for closure paths that do not request exact
  // faces.
  const bool qf_enabled = nodal_source || NodalSourceQuadratureMinimumOrder() > 0 ||
                          OperatorQuadratureMinimumOrder() > 0 || UseExactDGFaceOperator_() ||
                          ::igrlab::EnvFlagEnabledDefault(
                            "DG_IGR_IGR_DEVICE_QF", device_backend);
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

void IGRSolver::BeginFixedElementJacobiRKStep_()
{
  auto& saved = fixed_element_jacobi_rk_snapshot_;
  MFEM_VERIFY(solve_mode_ == SolveMode::FixedElementJacobi && !saved.active,
              "Invalid fixed element-Jacobi RK transaction begin.");
  saved.active = true;
  saved.history_saved = false;
  saved.history_valid = element_jacobi_history_valid_;
  saved.first_call = first_call_;
  saved.last_converged = last_linear_converged_;
  saved.last_final_norm = last_linear_final_norm_;
  saved.sigma.UseDevice(true);
  saved.sigma = sigma_;
  saved.blocks_present = bool(element_jacobi_blocks_);
  saved.tensor_key = element_jacobi_tensor_key_;
  saved.prefetch_accounting = pending_stage0_prefetch_accounting_;
  saved.original_prefetch_pending = saved.prefetch_accounting.valid;
  saved.original_prefetch_reclassified = false;
  if (saved.prefetch_accounting.valid)
  {
    saved.prefetch_sigma_checkpoint.UseDevice(true);
    saved.prefetch_sigma_checkpoint = pending_stage0_prefetch_sigma_checkpoint_;
    saved.prefetch_first_call = pending_stage0_prefetch_first_call_;
    saved.prefetch_last_converged = pending_stage0_prefetch_last_converged_;
    saved.prefetch_last_final_norm = pending_stage0_prefetch_last_final_norm_;
  }
}

void IGRSolver::AcceptFixedElementJacobiRKStep_()
{
  auto& saved = fixed_element_jacobi_rk_snapshot_;
  MFEM_VERIFY(saved.active, "Fixed element-Jacobi RK accept has no snapshot.");
  saved.abandoned_blocks.reset();
  saved.active = false;
}

bool IGRSolver::AbortFixedElementJacobiRKStep_()
{
  auto& saved = fixed_element_jacobi_rk_snapshot_;
  MFEM_VERIFY(saved.active, "Fixed element-Jacobi RK abort has no snapshot.");
  if (saved.history_saved)
  {
    element_jacobi_history_.Swap(saved.history);
  }
  element_jacobi_history_valid_ = saved.history_valid;
  first_call_ = saved.first_call;
  sigma_ = saved.sigma;
  last_linear_converged_ = saved.last_converged;
  last_linear_final_norm_ = saved.last_final_norm;
  if (element_jacobi_reference_blocks_ || element_jacobi_tensor_blocks_)
  {
    if (!saved.blocks_present)
    {
      // A failed first initialization must not choose the reference operator
      // for a later accepted state, even if its fifty sweeps finished.
      element_jacobi_blocks_.reset();
    }
    else if (saved.abandoned_blocks)
    {
      element_jacobi_blocks_ = std::move(saved.abandoned_blocks);
    }
    element_jacobi_tensor_key_ = saved.tensor_key;
  }
  if (saved.original_prefetch_reclassified)
  {
    // This solve will again be available to stage zero. Undo only its
    // temporary classification; all work done by the rejected attempt stays
    // in the measured totals and retains its own classification.
    auto undo = saved.prefetch_accounting;
    undo.calls = -undo.calls;
    undo.iterations = -undo.iterations;
    undo.converged_calls = -undo.converged_calls;
    undo.solve_seconds = -undo.solve_seconds;
    undo.assemble_seconds = -undo.assemble_seconds;
    undo.form_system_seconds = -undo.form_system_seconds;
    undo.pc_setup_seconds = -undo.pc_setup_seconds;
    AddOutputOnlyAccounting_(undo);
  }
  pending_stage0_prefetch_accounting_ = saved.prefetch_accounting;
  if (saved.prefetch_accounting.valid)
  {
    pending_stage0_prefetch_sigma_checkpoint_ = saved.prefetch_sigma_checkpoint;
    pending_stage0_prefetch_first_call_ = saved.prefetch_first_call;
    pending_stage0_prefetch_last_converged_ = saved.prefetch_last_converged;
    pending_stage0_prefetch_last_final_norm_ = saved.prefetch_last_final_norm;
  }
  saved.active = false;
  return saved.original_prefetch_reclassified;
}

void IGRSolver::CommitElementJacobiHistory_()
{
  auto& saved = fixed_element_jacobi_rk_snapshot_;
  if (saved.active && !saved.history_saved)
  {
    element_jacobi_history_.Swap(saved.history);
    saved.history_saved = true;
  }
  element_jacobi_history_.UseDevice(true);
  element_jacobi_history_ = sigma_;
  element_jacobi_history_valid_ = true;
}

void IGRSolver::DiscardUncommittedElementJacobiBlocks_()
{
  auto& saved = fixed_element_jacobi_rk_snapshot_;
  if (saved.active && saved.blocks_present && !saved.abandoned_blocks)
  {
    // Entry limiting can invalidate a prefetched first initialization. Keep
    // its factors alive until the attempt is accepted or rolled back.
    saved.abandoned_blocks = std::move(element_jacobi_blocks_);
  }
  else
  {
    element_jacobi_blocks_.reset();
  }
}

void IGRSolver::ConsumePendingStage0PrefetchAccounting()
{
  if (pending_stage0_prefetch_accounting_.valid && local_tensor_preconditioner_)
  {
    local_tensor_preconditioner_committed_ = true;
  }
  if (pending_stage0_prefetch_accounting_.valid &&
      solve_mode_ == SolveMode::FixedElementJacobi)
  {
    CommitElementJacobiHistory_();
  }
  fixed_element_jacobi_rk_snapshot_.original_prefetch_pending = false;
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
  auto& saved = fixed_element_jacobi_rk_snapshot_;
  if (saved.active && saved.original_prefetch_pending)
  {
    saved.original_prefetch_reclassified = true;
    saved.original_prefetch_pending = false;
  }
  if (solve_mode_ == SolveMode::FixedElementJacobi &&
      (element_jacobi_reference_blocks_ || element_jacobi_tensor_blocks_) &&
      !element_jacobi_history_valid_)
  {
    // An abandoned initialization must not choose the frozen inverse or
    // normalized tensor structure for the first production solve.
    DiscardUncommittedElementJacobiBlocks_();
  }
  if (!local_tensor_preconditioner_committed_)
  {
    local_tensor_preconditioner_.reset();
    local_tensor_preconditioner_key_.clear();
  }
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
  MFEM_VERIFY(!(element_jacobi_reference_blocks_ && element_jacobi_tensor_blocks_),
              "Reference and tensor element-Jacobi blocks are mutually exclusive.");
  MFEM_VERIFY(!(element_jacobi_reference_blocks_ || element_jacobi_tensor_blocks_) ||
                solve_mode_ == SolveMode::FixedElementJacobi,
              "Reference/tensor element blocks require fixed element-block Jacobi.");
  const bool local_tensor_requested = ::igrlab::EnvFlagEnabledDefault(
    "DG_IGR_IGR_LOCAL_TENSOR_PRECONDITIONER", false);
  MFEM_VERIFY(!local_tensor_requested || solve_mode_ == SolveMode::Krylov,
              "IGRLAB_IGR_LOCAL_TENSOR_PRECONDITIONER requires a Krylov solve.");
  MFEM_VERIFY(!local_tensor_preconditioner_ || local_tensor_requested,
              "Cannot disable the local tensor preconditioner on a live solver.");
  if (purpose == IGRSolvePurpose::ProductionStage0Prefetch)
  {
    pending_stage0_prefetch_sigma_checkpoint_ = sigma_;
    pending_stage0_prefetch_first_call_ = first_call_;
    pending_stage0_prefetch_last_converged_ = last_linear_converged_;
    pending_stage0_prefetch_last_final_norm_ = last_linear_final_norm_;
  }
  ::igrlab::NVTXRange range("IGR/update-from-state");
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
  if (purpose == IGRSolvePurpose::ProductionStage && local_tensor_preconditioner_)
  {
    local_tensor_preconditioner_committed_ = true;
  }
  if (solve_mode_ == SolveMode::FixedElementJacobi &&
      purpose == IGRSolvePurpose::ProductionStage)
  {
    CommitElementJacobiHistory_();
  }
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
    if (solve_mode_ == SolveMode::FixedElementJacobi &&
        (element_jacobi_reference_blocks_ || element_jacobi_tensor_blocks_) &&
        !element_jacobi_history_valid_)
    {
      DiscardUncommittedElementJacobiBlocks_();
    }
    if (!local_tensor_preconditioner_committed_)
    {
      local_tensor_preconditioner_.reset();
      local_tensor_preconditioner_key_.clear();
    }
  }
  else if (purpose == IGRSolvePurpose::ProductionStage0Prefetch)
  {
    pending_stage0_prefetch_accounting_ = delta;
  }
}

} // namespace igrlab

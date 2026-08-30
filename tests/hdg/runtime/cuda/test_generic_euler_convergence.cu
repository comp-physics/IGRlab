#include "generic_euler_convergence_fixture.hpp"

#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

enum class CudaQualificationRobustBranch
{
  not_applicable,
  any_qualified,
  lower,
  coupled
};

// This finite qualification accessor binds one explicit trace-policy
// ablation and checks the already-retained prepared aggregate.  It adds no
// production selector, callback, history, or diagnostic surface.
struct CudaQualificationTestAccess
{
  static dg::hdg::generic::HdgEulerSessionResult InitializeTracePolicy(
      cuda::GenericEulerHdgCudaSession& session,
      const dg::hdg::generic::EulerHdgRequest& request,
      const CudaTracePolicyKind trace_policy) noexcept
  {
    return session.InitializeRequest(request, trace_policy);
  }

  static bool PreparedTracePolicyValid(
      cuda::GenericEulerHdgCudaSession& session,
      const dg::hdg::generic::EulerHdgRequest& request,
      const CudaTracePolicyKind requested_trace_policy,
      const CudaQualificationRobustBranch robust_branch) noexcept
  {
    if (!session.borrowed_rank1_) return false;
    const auto& engine = *session.borrowed_rank1_;
    const auto& pending = engine.pending_;
    const auto& numerical = pending.numerical;
    const auto& tensor = pending.evidence.tensor_core;
    const bool igr_active =
        request.numerical_policy.stabilization.kind ==
            dg::hdg::generic::RegularizationKind::igr;
    const CudaTracePolicyDescriptor expected = ResolveCudaTracePolicy(
        request.backend.cuda_engine, igr_active, 1,
        request.polynomial_order, requested_trace_policy);
    if (numerical.trace_solves <= 0 ||
        numerical.linear_solve_attempts < numerical.trace_solves)
      return false;
    const std::uint64_t solves =
        static_cast<std::uint64_t>(numerical.trace_solves);
    const std::uint64_t attempts =
        static_cast<std::uint64_t>(numerical.linear_solve_attempts);
    const bool robust =
        CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
            expected.resolved_selection) &&
        CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(
            expected.trace_preconditioner);
    const bool robust_counts_bounded =
        tensor.robust_lower_patch_setups <= solves &&
        tensor.robust_coupled_patch_setups <= solves &&
        tensor.robust_lower_trace_solve_calls <= attempts &&
        tensor.robust_coupled_trace_solve_calls <= attempts &&
        tensor.robust_lower_trace_solve_calls <=
            std::numeric_limits<std::uint64_t>::max() -
                tensor.robust_coupled_trace_solve_calls &&
        tensor.robust_lower_patch_setups <=
            std::numeric_limits<std::uint64_t>::max() -
                tensor.robust_coupled_patch_setups &&
        attempts <= std::numeric_limits<std::uint64_t>::max() -
            tensor.robust_coupled_patch_setups;
    const std::uint64_t physical_calls = robust_counts_bounded
        ? tensor.robust_lower_trace_solve_calls +
              tensor.robust_coupled_trace_solve_calls
        : 0;
    const std::uint64_t physical_setups = robust_counts_bounded
        ? tensor.robust_lower_patch_setups +
              tensor.robust_coupled_patch_setups
        : 0;
    const bool robust_accounting = robust
        ? robust_counts_bounded &&
              tensor.robust_lower_specialization_supported &&
              tensor.robust_coupled_specialization_supported &&
              tensor.robust_branch_accounting_complete &&
              tensor.robust_lower_patch_setups == solves &&
              tensor.element_schwarz_patch_setups == physical_setups &&
              tensor.native_rank1_trace_solve_calls == physical_calls &&
              physical_calls >= attempts &&
              physical_calls <=
                  attempts + tensor.robust_coupled_patch_setups &&
              tensor.robust_coupled_trace_solve_calls >=
                  tensor.robust_coupled_patch_setups &&
              tensor.robust_lower_trace_solve_calls >=
                  solves - tensor.robust_coupled_patch_setups &&
              ((robust_branch ==
                    CudaQualificationRobustBranch::any_qualified) ||
               (robust_branch == CudaQualificationRobustBranch::lower &&
                tensor.robust_lower_trace_solve_calls == attempts &&
               tensor.robust_coupled_patch_setups == 0 &&
               tensor.robust_coupled_trace_solve_calls == 0) ||
               (robust_branch == CudaQualificationRobustBranch::coupled &&
                tensor.robust_coupled_patch_setups > 0 &&
                tensor.robust_coupled_trace_solve_calls >=
                    tensor.robust_coupled_patch_setups &&
                physical_calls > attempts))
        : robust_branch == CudaQualificationRobustBranch::not_applicable &&
              !tensor.robust_lower_specialization_supported &&
              !tensor.robust_coupled_specialization_supported &&
              tensor.robust_branch_accounting_complete &&
              tensor.robust_lower_patch_setups == 0 &&
              tensor.robust_lower_trace_solve_calls == 0 &&
              tensor.robust_coupled_patch_setups == 0 &&
              tensor.robust_coupled_trace_solve_calls == 0;
    const bool exact_preconditioner_leaf = robust
        ? CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(
              expected.trace_preconditioner)
        : CudaTracePreconditionerIsNativeElementSchwarzWarpCoupled(
              expected.trace_preconditioner);
    const bool valid = CudaTracePolicyDescriptorIsExact(expected) &&
        pending.active && pending.staging && numerical.candidate_ready &&
        numerical.finite && numerical.admissible && numerical.recovery &&
        numerical.true_residual && numerical.conservation &&
        numerical.success && pending.evidence.no_fallback &&
        tensor.selected_engine == request.backend.cuda_engine &&
        tensor.requested_trace_policy == requested_trace_policy &&
        tensor.resolved_trace_policy == expected.resolved_selection &&
        tensor.trace_policy_rank_count == 1 &&
        tensor.trace_policy_order == request.polynomial_order &&
        tensor.trace_preconditioner_configured &&
        tensor.trace_preconditioner == expected.trace_preconditioner &&
        exact_preconditioner_leaf && robust_accounting &&
        tensor.native_rank1_trace_solver_configured &&
        tensor.native_rank1_trace_solve_calls ==
            (robust ? physical_calls : attempts) &&
        tensor.hypre_trace_solve_calls == 0 &&
        tensor.trace_solver_accounting_complete &&
        tensor.native_matrix_free_trace_operator_configured &&
        tensor.native_element_schwarz_configured &&
        tensor.fallback_count == 0 &&
        tensor.fallback_accounting_complete;
    if (!valid)
    {
      std::cerr << "trace-policy qualification failed"
                << " active=" << pending.active
                << " staging=" << static_cast<bool>(pending.staging)
                << " candidate=" << numerical.candidate_ready
                << " finite=" << numerical.finite
                << " admissible=" << numerical.admissible
                << " recovery=" << numerical.recovery
                << " true_residual=" << numerical.true_residual
                << " conservation=" << numerical.conservation
                << " numerical_success=" << numerical.success
                << " trace_solves=" << numerical.trace_solves
                << " linear_attempts=" << numerical.linear_solve_attempts
                << " robust=" << robust
                << " robust_accounting=" << robust_accounting
                << " lower_supported="
                << tensor.robust_lower_specialization_supported
                << " coupled_supported="
                << tensor.robust_coupled_specialization_supported
                << " branch_accounting="
                << tensor.robust_branch_accounting_complete
                << " lower_setups=" << tensor.robust_lower_patch_setups
                << " lower_solves="
                << tensor.robust_lower_trace_solve_calls
                << " coupled_setups="
                << tensor.robust_coupled_patch_setups
                << " coupled_solves="
                << tensor.robust_coupled_trace_solve_calls
                << " exact_leaf=" << exact_preconditioner_leaf
                << " native_calls="
                << tensor.native_rank1_trace_solve_calls
                << " hypre_calls=" << tensor.hypre_trace_solve_calls
                << " trace_accounting="
                << tensor.trace_solver_accounting_complete
                << " matrix_free="
                << tensor.native_matrix_free_trace_operator_configured
                << " schwarz="
                << tensor.native_element_schwarz_configured
                << " no_fallback=" << pending.evidence.no_fallback
                << " fallback_count=" << tensor.fallback_count
                << " fallback_accounting="
                << tensor.fallback_accounting_complete << '\n';
    }
    return valid;
  }
};

} // namespace dg::hdg::runtime::cuda_w8a::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerHdgCpuSession;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyDescriptor;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyDescriptorIsExact;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKind;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKindName;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyUsesNativeElementMatrixFreeWarpCoupled;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled;
using dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationRobustBranch;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;
using dg::hdg::test::generic_euler_convergence::CollectiveAll;
using dg::hdg::test::generic_euler_convergence::ExecutionSelection;

struct TestSelection
{
  CudaTracePolicyKind tensor_igr_trace_policy =
      CudaTracePolicyKind::engine_default;
  bool zero_gradient_only = false;
};

bool ParseExplicitTracePolicy(
    const std::string_view text, CudaTracePolicyKind& policy) noexcept
{
  constexpr std::array<CudaTracePolicyKind, 4> policies{{
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_fp64,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi2_fp64,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi10_fp64,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64}};
  for (const CudaTracePolicyKind candidate : policies)
    if (text == CudaTracePolicyKindName(candidate))
    {
      policy = candidate;
      return true;
    }
  return false;
}

bool ParseTestSelection(
    const int argc, char* argv[],
    dg::hdg::test::generic_euler_convergence::CaseSelection& test_case,
    TestSelection& selection)
{
  selection = {};
  bool have_trace_policy = false;
  bool have_zero_gradient_only = false;
  std::vector<char*> common_arguments;
  common_arguments.reserve(static_cast<std::size_t>(argc));
  common_arguments.push_back(argv[0]);
  for (int argument = 1; argument < argc; ++argument)
  {
    const std::string_view option{argv[argument]};
    if (option == "--tensor-igr-trace-policy" &&
        argument + 1 < argc && !have_trace_policy)
    {
      have_trace_policy = ParseExplicitTracePolicy(
          argv[++argument], selection.tensor_igr_trace_policy);
      if (!have_trace_policy) return false;
    }
    else if (option == "--zero-gradient-only" &&
             !have_zero_gradient_only)
    {
      selection.zero_gradient_only = true;
      have_zero_gradient_only = true;
    }
    else
    {
      common_arguments.push_back(argv[argument]);
    }
  }
  if (!dg::hdg::test::generic_euler_convergence::ParseCaseSelection(
          static_cast<int>(common_arguments.size()),
          common_arguments.data(), test_case))
    return false;
  const bool explicit_policy = selection.tensor_igr_trace_policy !=
      CudaTracePolicyKind::engine_default;
  const bool igr = test_case.regularization ==
      dg::hdg::generic::RegularizationKind::igr;
  if (explicit_policy != have_trace_policy ||
      (explicit_policy && !igr) ||
      (selection.zero_gradient_only && !igr))
    return false;
  if (!explicit_policy && !selection.zero_gradient_only) return true;
  const CudaTracePolicyDescriptor descriptor = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, igr, 1, test_case.order,
      selection.tensor_igr_trace_policy);
  return CudaTracePolicyDescriptorIsExact(descriptor) &&
      (CudaTracePolicyUsesNativeElementMatrixFreeWarpCoupled(
           descriptor.resolved_selection) ||
       CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
           descriptor.resolved_selection));
}

template <typename Coordinates, typename Diagnostics>
bool ValidCommittedMetadata(const Coordinates coordinates,
                            const Diagnostics diagnostics,
                            const int steps, const double dt) noexcept
{
  const double expected_time = static_cast<double>(steps) * dt;
  const double time_scale = std::max(1.0, std::abs(expected_time));
  return coordinates.step == static_cast<std::uint64_t>(steps) &&
      coordinates.generation == static_cast<std::uint64_t>(steps) &&
      std::abs(coordinates.time - expected_time) <=
          64.0 * std::numeric_limits<double>::epsilon() * time_scale &&
      diagnostics.available &&
      diagnostics.wave_generation == coordinates.generation &&
      diagnostics.coordinates.step == coordinates.step &&
      std::isfinite(diagnostics.maximum_wave_speed) &&
      diagnostics.maximum_wave_speed > 0.0;
}

bool AdvanceCpu(const EulerHdgRequest& request, const int steps,
                const double dt, mfem::ParGridFunction& committed)
{
  GenericEulerHdgCpuSession session;
  if (session.Initialize(request).status !=
      GenericEulerCpuSessionStatus::initialized)
  {
    return false;
  }
  for (int step = 0; step < steps; ++step)
  {
    if (session.Advance(dt).status != GenericEulerCpuSessionStatus::advanced)
      return false;
  }
  const bool local = session.ExportCommittedInterior(committed) &&
      ValidCommittedMetadata(
          session.coordinates(), session.diagnostics(), steps, dt);
  return CollectiveAll(local, request.communicator);
}

bool AdvanceCuda(const EulerHdgRequest& request, const int steps,
                 const double dt, mfem::ParGridFunction& committed,
                 const CudaTracePolicyKind trace_policy,
                 const CudaQualificationRobustBranch robust_branch =
                     CudaQualificationRobustBranch::not_applicable)
{
  GenericEulerHdgCudaSession session;
  const auto initialization = CudaQualificationTestAccess::InitializeTracePolicy(
      session, request, trace_policy);
  if (initialization.status != HdgEulerSessionStatus::initialized)
  {
    std::cerr << "trace-policy initialization failed status="
              << static_cast<int>(initialization.status) << '\n';
    return false;
  }
  for (int step = 0; step < steps; ++step)
  {
    GenericEulerCudaPreparedStep authority;
    GenericEulerHdgCudaSession::prepared_output_type output{};
    const auto prepared = session.Prepare(dt, authority, output);
    if (prepared.status != HdgEulerSessionStatus::prepared)
    {
      std::cerr << "trace-policy prepare failed step=" << step
                << " status=" << static_cast<int>(prepared.status) << '\n';
      return false;
    }
    const bool qualify_trace_policy =
        trace_policy != CudaTracePolicyKind::engine_default ||
        robust_branch != CudaQualificationRobustBranch::not_applicable;
    if (qualify_trace_policy &&
        !CudaQualificationTestAccess::PreparedTracePolicyValid(
            session, request, trace_policy, robust_branch))
    {
      std::cerr << "trace-policy prepared evidence failed step=" << step
                << '\n';
      return false;
    }
    if (!session.AcknowledgePreparedOutput(authority, output))
    {
      std::cerr << "trace-policy output acknowledgement failed step="
                << step << '\n';
      return false;
    }
    const auto committed = session.Commit(std::move(authority), output);
    if (committed.status != HdgEulerSessionStatus::committed)
    {
      std::cerr << "trace-policy commit failed step=" << step
                << " status=" << static_cast<int>(committed.status) << '\n';
      return false;
    }
  }
  const int size = committed.Size();
  constexpr int components =
      dg::hdg::generic::euler_2d_conserved_components;
  const bool extent_valid = size > 0 && size % components == 0;
  const bool local = extent_valid && session.ExportCommittedInterior({
      committed.HostWrite(),
      static_cast<std::size_t>(size / components), components}) &&
      ValidCommittedMetadata(
          session.coordinates(), session.diagnostics(), steps, dt);
  return CollectiveAll(local, request.communicator);
}

class ThreeEngineRunner
{
public:
  explicit ThreeEngineRunner(
      const CudaTracePolicyKind tensor_igr_trace_policy) noexcept
      : tensor_igr_trace_policy_(tensor_igr_trace_policy)
  {
  }

  bool Begin(const ExecutionSelection execution)
  {
    if (execution.backend == BackendKind::cpu)
    {
      if (phase_ != 0 || device_) return false;
      phase_ = 1;
      return true;
    }
    if (execution.backend != BackendKind::cuda) return false;
    if (execution.cuda_engine == CudaEngineKind::ordinary)
    {
      if (phase_ != 1 || device_) return false;
      if (HYPRE_ClearAllErrors() != 0 || HYPRE_GetError() != 0)
        return false;
      device_ = std::make_unique<mfem::Device>("cuda");
      phase_ = 2;
      return true;
    }
    if (execution.cuda_engine == CudaEngineKind::tensor_core)
    {
      if (phase_ != 2 || !device_) return false;
      if (HYPRE_ClearAllErrors() != 0 || HYPRE_GetError() != 0)
        return false;
      phase_ = 3;
      return true;
    }
    return false;
  }

  bool Advance(const EulerHdgRequest& request, const int steps,
               const double dt, mfem::ParGridFunction& committed) const
  {
    if (request.backend.kind == BackendKind::cpu)
      return phase_ == 1 && AdvanceCpu(request, steps, dt, committed);
    if (request.backend.kind != BackendKind::cuda || phase_ < 2)
      return false;
    const CudaTracePolicyKind trace_policy =
        request.backend.cuda_engine == CudaEngineKind::tensor_core &&
                request.numerical_policy.stabilization.kind ==
                    dg::hdg::generic::RegularizationKind::igr
            ? tensor_igr_trace_policy_
            : CudaTracePolicyKind::engine_default;
    const bool tensor_igr =
        request.backend.cuda_engine == CudaEngineKind::tensor_core &&
        request.numerical_policy.stabilization.kind ==
            dg::hdg::generic::RegularizationKind::igr;
    const CudaTracePolicyDescriptor descriptor = ResolveCudaTracePolicy(
        request.backend.cuda_engine, tensor_igr, 1,
        request.polynomial_order, trace_policy);
    const CudaQualificationRobustBranch robust_branch =
        CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
            descriptor.resolved_selection)
        ? CudaQualificationRobustBranch::any_qualified
        : CudaQualificationRobustBranch::not_applicable;
    return AdvanceCuda(
        request, steps, dt, committed, trace_policy, robust_branch);
  }

private:
  std::unique_ptr<mfem::Device> device_;
  CudaTracePolicyKind tensor_igr_trace_policy_ =
      CudaTracePolicyKind::engine_default;
  int phase_ = 0;
};

bool RunZeroGradientTensorIgrCase(
    const dg::hdg::test::generic_euler_convergence::CaseSelection& test_case,
    const CudaTracePolicyKind trace_policy)
{
  using namespace dg::hdg::test::generic_euler_convergence;
  const CudaTracePolicyDescriptor descriptor = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true, 1, test_case.order, trace_policy);
  if (!CudaTracePolicyDescriptorIsExact(descriptor) ||
      (!CudaTracePolicyUsesNativeElementMatrixFreeWarpCoupled(
           descriptor.resolved_selection) &&
       !CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
           descriptor.resolved_selection)) ||
      HYPRE_ClearAllErrors() != 0 || HYPRE_GetError() != 0)
    return false;

  mfem::Device device("cuda");
  mfem::ParMesh mesh = MakePeriodicMesh(3, MPI_COMM_WORLD);
  mfem::L2_FECollection collection(
      test_case.order, dimension, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction initial(&space);
  mfem::ParGridFunction source(&space);
  mfem::ParGridFunction committed(&space);
  source = 0.0;
  const EntropyWave wave{{{0.5, 0.0}}, {{0.0, 1.0}}, 0.1, 1.0};
  BoundaryAudit boundary{};
  if (!ProjectInitialState(wave, initial)) return false;
  const ExecutionSelection execution{
      BackendKind::cuda, CudaEngineKind::tensor_core, "tensor CUDA"};
  const EulerHdgRequest request = MakeRequest(
      mesh, initial, source, boundary, execution, test_case,
      test_case.igr_assembly_quadrature_orders);
  constexpr double dt = 0.02;
  const CudaQualificationRobustBranch robust_branch =
      CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
          descriptor.resolved_selection)
      ? CudaQualificationRobustBranch::coupled
      : CudaQualificationRobustBranch::not_applicable;
  const bool advanced = AdvanceCuda(
      request, 1, dt, committed, trace_policy, robust_branch);
  const double norm = advanced ? GlobalL2Norm(committed)
                               : std::numeric_limits<double>::infinity();
  const bool local = advanced && boundary.calls == 0 &&
      std::isfinite(norm) && norm > 0.0;
  const bool passed = CollectiveAll(local, MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    std::cout << "tensor IGR p" << test_case.order
              << " zero-gradient coupled trace gate policy="
              << CudaTracePolicyKindName(trace_policy)
              << " passed=" << passed << '\n';
  return passed;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  dg::hdg::test::generic_euler_convergence::CaseSelection test_case{};
  TestSelection selection{};
  const bool parsed = ParseTestSelection(
      argc, argv, test_case, selection);
  if (!parsed || ranks != 1)
  {
    if (ranks == 1)
      std::cerr << "Usage: " << argv[0]
                << " --order 1|2|3 --regularization conventional|igr"
                   " [--igr-volume-quadrature-order Q"
                   " --igr-face-quadrature-order Q]"
                   " [--newton-linear-rtol R]"
                   " [--final-linear-rtol R]"
                   " [--true-residual-tightening-linear-rtol R]"
                   " [--tensor-igr-trace-policy "
                   "native-element-schwarz-matrix-free-warp-coupled-fp64|"
                   "native-element-schwarz-matrix-free-warp-coupled-sigma-"
                   "jacobi2-fp64|native-element-schwarz-matrix-free-warp-"
                   "coupled-sigma-jacobi10-fp64|"
                   "native-element-schwarz-matrix-free-warp-robust-"
                   "coupled-sigma-jacobi10-fp64]"
                   " [--zero-gradient-only]\n";
    else
      std::cerr << "Three-engine convergence qualification is rank-one; "
                   "rank-two CUDA correctness has separate coverage\n";
    return 2;
  }

  if (selection.zero_gradient_only)
    return RunZeroGradientTensorIgrCase(
        test_case, selection.tensor_igr_trace_policy) ? 0 : 1;

  const std::array<ExecutionSelection, 3> executions{{
      {BackendKind::cpu, CudaEngineKind::ordinary, "CPU"},
      {BackendKind::cuda, CudaEngineKind::ordinary, "ordinary CUDA"},
      {BackendKind::cuda, CudaEngineKind::tensor_core, "tensor CUDA"}}};
  ThreeEngineRunner runner(selection.tensor_igr_trace_policy);
  const bool passed =
      dg::hdg::test::generic_euler_convergence::RunComparisonCase(
          MPI_COMM_WORLD, test_case, executions, runner);
  return passed ? 0 : 1;
}

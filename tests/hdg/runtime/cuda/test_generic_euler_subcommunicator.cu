#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "sim/hdg/state_io.hpp"
#include "test_cuda_trace_solver_routing.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

constexpr bool CudaPreparedIluAccountingValid(
    const CudaTensorCoreEvidence& tensor, const int trace_solves,
    const int linear_iterations, const int linear_solve_attempts,
    const std::uint64_t matrix_generations,
    const generic::CudaEngineKind engine, const bool igr_active) noexcept
{
  const auto policy = ResolveCudaTracePolicy(
      engine, igr_active, tensor.trace_policy_rank_count,
      tensor.trace_policy_order, tensor.requested_trace_policy);
  if (!CudaTracePolicyDescriptorIsExact(policy) ||
      tensor.resolved_trace_policy != policy.resolved_selection)
    return false;
  const auto preconditioner = policy.trace_preconditioner;
  const bool ilu = CudaTracePreconditionerIsIlu(preconditioner);
  const auto factor_reuse_policy = policy.ilu_factor_reuse_policy;
  const auto triangular_solve_kind = policy.ilu_triangular_solve_kind;
  const int lower_jacobi_iterations = policy.ilu_lower_jacobi_iterations;
  const int upper_jacobi_iterations = policy.ilu_upper_jacobi_iterations;
  if (!ilu)
    return (preconditioner == CudaTracePreconditionerKind::hypre_diag_scale ||
            CudaTracePreconditionerIsNativeElementSchwarz(preconditioner)) &&
        tensor.ilu_type == -1 && tensor.ilu_level_of_fill == -1 &&
        tensor.ilu_local_reordering == -1 &&
        tensor.ilu_application_iterations == 0 &&
        triangular_solve_kind ==
            CudaIluTriangularSolveKind::not_applicable &&
        tensor.ilu_triangular_solve_kind == triangular_solve_kind &&
        tensor.ilu_lower_jacobi_iterations == 0 &&
        tensor.ilu_upper_jacobi_iterations == 0 &&
        tensor.ilu_accounting_kind == CudaIluAccountingKind::unavailable &&
        factor_reuse_policy == CudaIluFactorReusePolicy::not_applicable &&
        tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
        tensor.ilu_factor_setups == 0 &&
        tensor.ilu_factor_reuses == 0 &&
        tensor.ilu_preconditioner_callbacks_lower_bound == 0 &&
        tensor.ilu_preconditioner_callbacks_upper_bound == 0 &&
        tensor.ilu_inner_iterations_lower_bound == 0 &&
        tensor.ilu_inner_iterations_upper_bound == 0 &&
        tensor.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
        tensor.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
        tensor.ilu_factor_age_generations == 0 &&
        tensor.ilu_accounting_complete;
  const int expected_fill = 0;
  const int expected_reordering = igr_active ? 1 : 0;
  if (trace_solves <= 0 || linear_iterations < 0 ||
      linear_solve_attempts < trace_solves || tensor.ilu_type != 0 ||
      tensor.ilu_level_of_fill != expected_fill ||
      tensor.ilu_local_reordering != expected_reordering ||
      tensor.ilu_application_iterations != 1 ||
      tensor.ilu_triangular_solve_kind != triangular_solve_kind ||
      tensor.ilu_lower_jacobi_iterations != lower_jacobi_iterations ||
      tensor.ilu_upper_jacobi_iterations != upper_jacobi_iterations ||
      (triangular_solve_kind == CudaIluTriangularSolveKind::direct &&
       (lower_jacobi_iterations != 0 || upper_jacobi_iterations != 0)) ||
      (triangular_solve_kind ==
           CudaIluTriangularSolveKind::iterative_jacobi &&
       (lower_jacobi_iterations != 5 || upper_jacobi_iterations != 5)) ||
      tensor.ilu_accounting_kind !=
          CudaIluAccountingKind::derived_bounds_v1 ||
      !tensor.ilu_accounting_complete ||
      tensor.ilu_preconditioner_callbacks_lower_bound >
          tensor.ilu_preconditioner_callbacks_upper_bound)
    return false;
  const std::uint64_t attempts =
      static_cast<std::uint64_t>(linear_solve_attempts);
  const bool factor_work = factor_reuse_policy ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve
      ? tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
            tensor.ilu_factor_setups == attempts &&
            tensor.ilu_factor_reuses == 0 &&
            tensor.ilu_factor_age_generations == 0
      : factor_reuse_policy ==
              CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
            matrix_generations > 0 &&
            matrix_generations <=
                static_cast<std::uint64_t>(trace_solves) &&
            tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
            tensor.ilu_factor_setups == 1 &&
            tensor.ilu_factor_reuses == attempts - 1 &&
            // Factor age counts distinct matrix generations, not causal
            // solves.  RODAS3's last three stages reuse the common shifted
            // matrix and therefore do not age its retained ILU factor.
            tensor.ilu_factor_age_generations == matrix_generations - 1;
  if (!factor_work) return false;
  const std::uint64_t iterations =
      static_cast<std::uint64_t>(linear_iterations);
  const std::uint64_t lower =
      tensor.ilu_preconditioner_callbacks_lower_bound;
  const std::uint64_t upper =
      tensor.ilu_preconditioner_callbacks_upper_bound;
  const bool callback_bounds = iterations == 0
      ? lower == 0 && upper == 0
      : iterations <= std::numeric_limits<std::uint64_t>::max()/2 &&
            iterations < lower && upper == 2*iterations;
  const std::uint64_t multiplier = static_cast<std::uint64_t>(
      tensor.ilu_application_iterations);
  const std::uint64_t factor_spmv_multiplier =
      triangular_solve_kind == CudaIluTriangularSolveKind::iterative_jacobi
      ? 8 : 0;
  return callback_bounds &&
      lower <= std::numeric_limits<std::uint64_t>::max()/multiplier &&
      upper <= std::numeric_limits<std::uint64_t>::max()/multiplier &&
      tensor.ilu_inner_iterations_lower_bound == lower*multiplier &&
      tensor.ilu_inner_iterations_upper_bound == upper*multiplier &&
      (factor_spmv_multiplier == 0 ||
       (tensor.ilu_inner_iterations_lower_bound <=
            std::numeric_limits<std::uint64_t>::max()/
                factor_spmv_multiplier &&
        tensor.ilu_inner_iterations_upper_bound <=
            std::numeric_limits<std::uint64_t>::max()/
                factor_spmv_multiplier)) &&
      tensor.ilu_jacobi_factor_spmv_updates_lower_bound ==
          tensor.ilu_inner_iterations_lower_bound*factor_spmv_multiplier &&
      tensor.ilu_jacobi_factor_spmv_updates_upper_bound ==
          tensor.ilu_inner_iterations_upper_bound*factor_spmv_multiplier;
}

// Test-only friend access keeps engine-selection evidence attached to the
// existing subcommunicator lifecycle instead of adding a production API.
struct CudaQualificationTestAccess
{
  template <typename Engine>
  static bool VerifyPreparedCudaEngine(
      Engine& engine, const generic::CudaEngineKind expected,
      const generic::EulerHdgTimeTreatment expected_time_treatment,
      const MPI_Comm expected_communicator) noexcept
  {
    if (!engine.initialized_ || !engine.pending_.active ||
        !engine.pending_.staging)
      return false;
    const auto& evidence = engine.pending_.evidence;
    const auto& numerical = engine.pending_.numerical;
    const auto& tensor = evidence.tensor_core;
    const bool igr_active = engine.numerics_.shape_.igr_active;
    const bool partitioned_igr = igr_active &&
        expected_time_treatment ==
            generic::EulerHdgTimeTreatment::partitioned_backward_euler;
    const bool rodas3 = expected_time_treatment ==
        generic::EulerHdgTimeTreatment::rodas3;
    const int final_trace_solves = partitioned_igr ? 2 : 1;
    const auto trace_policy = ResolveCudaTracePolicy(
        expected, igr_active, engine.config_.expected_rank_count,
        engine.config_.order, CudaTracePolicyKind::engine_default);
    const bool tensor_rank_one =
        expected == generic::CudaEngineKind::tensor_core &&
        engine.config_.expected_rank_count == 1;
    const bool retained_ilu = trace_policy.ilu_factor_reuse_policy ==
        CudaIluFactorReusePolicy::first_trace_solve_per_attempt;
    const bool rodas3_preconditioner_census = tensor_rank_one
        ? evidence.common_shift_preconditioner_setups >= 1 &&
              evidence.common_shift_preconditioner_reuses == 3
        : retained_ilu
            ? evidence.common_shift_preconditioner_setups == 1 &&
                  evidence.common_shift_preconditioner_reuses == 3
            : evidence.common_shift_preconditioner_setups == 4 &&
                  evidence.common_shift_preconditioner_reuses == 0;
    const bool common_linear_accounting = evidence.linear_iterations >= 0 &&
        evidence.newton_correction_linear_iterations >= 0 &&
        evidence.final_coherent_linear_iterations >= 0 &&
        static_cast<long long>(evidence.linear_iterations) ==
            static_cast<long long>(
                evidence.newton_correction_linear_iterations) +
                evidence.final_coherent_linear_iterations &&
        evidence.linear_solve_attempts >= evidence.trace_solves &&
        evidence.true_residual_tightenings >= 0 &&
        static_cast<long long>(evidence.linear_solve_attempts) ==
            static_cast<long long>(evidence.trace_solves) +
                evidence.true_residual_tightenings &&
        evidence.line_search_trials >= 0 &&
        evidence.rejected_line_search_trials >= 0;
    const bool backward_euler_work = !rodas3 &&
        evidence.nonlinear_updates >= 0 &&
        evidence.trace_solves ==
            evidence.nonlinear_updates + final_trace_solves &&
        evidence.line_search_trials == evidence.nonlinear_updates +
            evidence.rejected_line_search_trials;
    const bool rodas3_work = rodas3 && evidence.nonlinear_updates >= 0 &&
        evidence.trace_solves == evidence.nonlinear_updates + 4 &&
        evidence.line_search_trials == 0 &&
        evidence.rejected_line_search_trials == 0 &&
        evidence.common_shift_operator_builds == 1 &&
        evidence.common_shift_operator_reuses == 3 &&
        evidence.condensed_rhs_refreshes == 3 &&
        evidence.common_shift_trace_solves == 4 &&
        rodas3_preconditioner_census &&
        std::isfinite(evidence.temporal_error_norm) &&
        std::isfinite(evidence.final_algebraic_residual) &&
        std::isfinite(evidence.final_algebraic_correction_norm);
    int communicator_comparison = MPI_UNEQUAL;
    int world_comparison = MPI_IDENT;
    const bool communicator_bound = expected_communicator != MPI_COMM_NULL &&
        MPI_Comm_compare(engine.communicator_, expected_communicator,
                         &communicator_comparison) == MPI_SUCCESS &&
        MPI_Comm_compare(engine.communicator_, MPI_COMM_WORLD,
                         &world_comparison) == MPI_SUCCESS &&
        (communicator_comparison == MPI_IDENT ||
         communicator_comparison == MPI_CONGRUENT) &&
        world_comparison == MPI_UNEQUAL;
    const bool work_accounting = common_linear_accounting &&
        (backward_euler_work || rodas3_work);
    const bool numerical_gates = numerical.candidate_ready &&
        numerical.finite && numerical.admissible && numerical.recovery &&
        numerical.true_residual && numerical.conservation &&
        numerical.telemetry && numerical.success &&
        std::isfinite(numerical.maximum_conservation_error) &&
        std::isfinite(numerical.maximum_conservation_relative_error) &&
        numerical.maximum_conservation_relative_error <=
            engine.config_.numerical_policy.conservation_relative_tolerance;
    const bool host_staged_rank_two =
        engine.config_.expected_rank_count != 2 ||
        (tensor.hypre_host_staged_rank2_trace_solver_configured &&
         evidence.transfers.mpi_staging_bytes > 0);
    const bool time_contract =
        engine.config_.time_treatment == expected_time_treatment &&
        engine.numerics_.partitioned_backward_euler_ == partitioned_igr;
    const bool policy_contract =
        tensor.selected_engine == expected &&
        CudaTracePolicyDescriptorIsExact(trace_policy) &&
        tensor.requested_trace_policy ==
            CudaTracePolicyKind::engine_default &&
        tensor.resolved_trace_policy == trace_policy.resolved_selection &&
        tensor.trace_policy_rank_count == engine.config_.expected_rank_count &&
        tensor.trace_policy_order == engine.config_.order &&
        tensor.trace_preconditioner_configured &&
        tensor.trace_preconditioner == trace_policy.trace_preconditioner;
    const std::uint64_t matrix_generations = rodas3
        ? evidence.complete_condensations
        : evidence.trace_solves > 0
            ? static_cast<std::uint64_t>(evidence.trace_solves) : 0;
    const bool ilu_accounting = CudaPreparedIluAccountingValid(
        tensor, evidence.trace_solves, evidence.linear_iterations,
        evidence.linear_solve_attempts, matrix_generations, expected,
        igr_active);
    const bool routing = CudaPreparedTraceSolverRoutingValid(
        tensor, rodas3 ? evidence.linear_solve_attempts
                       : evidence.trace_solves,
        expected,
        engine.config_.expected_rank_count);
    const bool exact_rodas3_launch_census = rodas3 &&
        evidence.complete_assemblies > 0 &&
        evidence.complete_condensations > 0 &&
        tensor.volume_assembly_launches == evidence.complete_assemblies &&
        tensor.condensation_launches == evidence.complete_condensations;
    const bool volume = rodas3
        ? exact_rodas3_launch_census &&
              (expected == generic::CudaEngineKind::ordinary
                   ? tensor.volume_mma_tiles == 0
                   : CudaPreparedTensorVolumeEvidenceValid(
                         tensor, engine.numerics_.shape_, expected) &&
                         tensor.volume_mma_tiles > 0)
        : CudaPreparedTensorVolumeEvidenceValid(
              tensor, engine.numerics_.shape_, expected);
    const bool no_fallback = tensor.fallback_accounting_complete &&
        tensor.fallback_count == 0 && evidence.no_fallback;
    const bool backend_work = rodas3
        ? exact_rodas3_launch_census &&
              (expected == generic::CudaEngineKind::ordinary
                   ? tensor.mma_tiles == 0
                   : tensor.hardware_supported && tensor.mma_tiles > 0)
        : expected == generic::CudaEngineKind::ordinary
            ? tensor.condensation_launches == 0 && tensor.mma_tiles == 0
            : tensor.hardware_supported &&
                  tensor.condensation_launches > 0 && tensor.mma_tiles > 0;
    const bool valid = communicator_bound && time_contract &&
        work_accounting && numerical_gates && host_staged_rank_two &&
        policy_contract && ilu_accounting && routing && volume &&
        no_fallback && backend_work;
    int rank = -1;
    if (!valid && expected_communicator != MPI_COMM_NULL &&
        MPI_Comm_rank(expected_communicator, &rank) == MPI_SUCCESS &&
        rank == 0)
    {
      std::cerr << "subcommunicator prepared-evidence mismatch"
                << " communicator/time/work/numerical/host/policy/ilu/"
                   "routing/volume/fallback/backend="
                << communicator_bound << '/' << time_contract << '/'
                << work_accounting << '/' << numerical_gates << '/'
                << host_staged_rank_two << '/' << policy_contract << '/'
                << ilu_accounting << '/' << routing << '/' << volume << '/'
                << no_fallback << '/' << backend_work
                << " trace/linear/attempts/tightenings="
                << evidence.trace_solves << '/'
                << evidence.linear_iterations << '/'
                << evidence.linear_solve_attempts << '/'
                << evidence.true_residual_tightenings
                << " common_shift="
                << evidence.common_shift_operator_builds << '/'
                << evidence.common_shift_operator_reuses << '/'
                << evidence.condensed_rhs_refreshes << '/'
                << evidence.common_shift_trace_solves
                << " preconditioner="
                << evidence.common_shift_preconditioner_setups << '/'
                << evidence.common_shift_preconditioner_reuses
                << " assembly=" << tensor.volume_assembly_launches << '/'
                << evidence.complete_assemblies << '/'
                << tensor.volume_mma_tiles
                << " condensation=" << tensor.condensation_launches << '/'
                << evidence.complete_condensations << '/'
                << tensor.mma_tiles << " hardware="
                << tensor.hardware_supported << '\n';
    }
    return valid;
  }

  static bool VerifyPreparedCudaEngine(
      cuda::GenericEulerHdgCudaSession& session,
      const generic::CudaEngineKind expected,
      const generic::EulerHdgTimeTreatment expected_time_treatment,
      const MPI_Comm expected_communicator) noexcept
  {
    return session.Dispatch([&](auto& engine)
    { return VerifyPreparedCudaEngine(
          engine, expected, expected_time_treatment,
          expected_communicator); });
  }

  template <typename Engine>
  static bool CaptureCommittedHashes(
      Engine& engine, StateHash& interior, StateHash& trace) noexcept
  {
    if (!engine.initialized_ || engine.pending_.active) return false;
    const auto committed = engine.numerics_.CommittedSlot();
    if (committed.interior.count == 0 || committed.trace.count == 0)
      return false;
    interior = committed.interior.hash;
    trace = committed.trace.hash;
    return true;
  }

  static bool CaptureCommittedHashes(
      cuda::GenericEulerHdgCudaSession& session,
      StateHash& interior, StateHash& trace) noexcept
  {
    return session.Dispatch([&](auto& engine)
    { return CaptureCommittedHashes(engine, interior, trace); });
  }
};

} // namespace dg::hdg::runtime::cuda_w8a::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::EulerHdgTimeTreatment;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::StateHash;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::CudaPreparedOutputMetadataV1;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;

constexpr int components = dg::hdg::generic::euler_2d_conserved_components;
constexpr std::uint32_t cuda_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cuda) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

bool All(const MPI_Comm communicator, const bool local) noexcept
{
  const int input = local ? 1 : 0;
  int output = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator) ==
          MPI_SUCCESS &&
      output == 1;
}

bool SameCoordinates(
    const dg::hdg::generic::HdgEulerSessionCoordinates& left,
    const dg::hdg::generic::HdgEulerSessionCoordinates& right) noexcept
{
  return left.time == right.time && left.dt == right.dt &&
      left.step == right.step && left.generation == right.generation;
}

bool AssessPreparedRodas3(
    GenericEulerHdgCudaSession& session,
    const GenericEulerCudaPreparedStep& authority,
    const CudaPreparedOutputMetadataV1& output,
    const EulerHdgTimeTreatment time_treatment) noexcept
{
  if (time_treatment != EulerHdgTimeTreatment::rodas3) return true;
  constexpr dg::hdg::generic::HdgEulerPreparedAssessmentPolicy policy{
      1.0e-8, 1.0e-5, 0.25};
  dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
  return session.AssessPrepared(authority, output, policy, assessment) &&
      dg::hdg::generic::IsValidHdgEulerPreparedAssessment(assessment) &&
      assessment.finite_gate && assessment.solver_gate &&
      assessment.residual_gate && assessment.admissibility_gate &&
      assessment.algebraic_closure_gate &&
      assessment.algebraic_correction_gate;
}

bool ExteriorState(const void*,
                   const dg::hdg::generic::BoundaryPointView&,
                   const double* interior, double* exterior) noexcept
{
  if (interior == nullptr || exterior == nullptr) return false;
  for (int component = 0; component < components; ++component)
    exterior[component] = interior[component];
  return true;
}

mfem::Mesh PeriodicMesh()
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0] = 0.0;
  translations[1] = 0.0;
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  return mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
}

struct BorrowedFields
{
  std::vector<double> initial;
  std::vector<double> initial_true_trace;
};

BorrowedFields MakeFields(mfem::ParMesh& mesh)
{
  constexpr int order = 1;
  BorrowedFields fields;
  mfem::L2_FECollection volume_collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace volume_space(
      &mesh, &volume_collection, components, mfem::Ordering::byNODES);
  const std::size_t volume_scalars = static_cast<std::size_t>(
      volume_space.GetVSize() / components);
  fields.initial.resize(static_cast<std::size_t>(volume_space.GetVSize()));
  for (std::size_t scalar = 0; scalar < volume_scalars; ++scalar)
  {
    fields.initial[scalar] = 1.0;
    fields.initial[volume_scalars + scalar] = 0.05;
    fields.initial[2 * volume_scalars + scalar] = -0.02;
    fields.initial[3 * volume_scalars + scalar] = 2.60;
  }

  mfem::DG_Interface_FECollection trace_collection(
      order, 2, mfem::FiniteElement::VALUE,
      mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &trace_collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction local_trace(&trace_space);
  mfem::VectorFunctionCoefficient coefficient(
      components, [](const mfem::Vector&, mfem::Vector& state)
      {
        state.SetSize(components);
        state[0] = 1.0;
        state[1] = 0.05;
        state[2] = -0.02;
        state[3] = 2.60;
      });
  mfem::Vector true_trace;
  const auto policy = dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  if (!dg_sim::hdg::ProjectEulerTrueTraceCoefficientL2(
          trace_space, local_trace, true_trace, coefficient,
          policy.true_trace_projection_quadrature_order))
  {
    return {};
  }
  if (true_trace.Size() > 0)
  {
    const double* const values = true_trace.HostRead();
    fields.initial_true_trace.assign(values, values + true_trace.Size());
  }
  return fields;
}

EulerHdgRequest MakeRequest(const mfem::ParMesh& mesh,
                            BorrowedFields& fields,
                            const CudaEngineKind engine,
                            const EulerHdgTimeTreatment time_treatment)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = 1;
  request.backend = {BackendKind::cuda, 0, engine};
  request.capabilities = cuda_capabilities;
  request.initial_state = {
      fields.initial.empty() ? nullptr : fields.initial.data(),
      fields.initial.size(), components};
  request.initial_true_trace = {
      fields.initial_true_trace.empty() ? nullptr
                                        : fields.initial_true_trace.data(),
      fields.initial_true_trace.size(), components};
  request.source = {nullptr, 0, components};
  request.boundary = {
      nullptr, ExteriorState, {nullptr, 0, 2, 1.4}};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(1);
  request.numerical_policy.stabilization = {
      RegularizationKind::igr, 0.2, 0.8};
  request.time_treatment = time_treatment;
  return request;
}

bool CheckActiveSubcommunicator(const MPI_Comm communicator,
                                const int active_ranks,
                                const CudaEngineKind engine,
                                const EulerHdgTimeTreatment time_treatment)
{
  int rank = -1;
  int ranks = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS ||
      ranks != active_ranks)
    return false;
  const auto fail = [&](const std::string_view phase) noexcept
  {
    if (rank == 0)
      std::cerr << "subcommunicator RODAS lifecycle failed phase="
                << phase << " active_ranks=" << active_ranks
                << " engine=" << static_cast<int>(engine) << '\n';
    return false;
  };
  const double dt = time_treatment == EulerHdgTimeTreatment::rodas3
      ? 1.0e-6 : 0.025;

  mfem::Mesh serial = PeriodicMesh();
  mfem::ParMesh mesh(communicator, serial);
  BorrowedFields fields = MakeFields(mesh);
  if (!All(communicator, HYPRE_GetError() == 0)) return fail("fields");
  EulerHdgRequest request = MakeRequest(
      mesh, fields, engine, time_treatment);

  if (ranks == 2)
  {
    GenericEulerHdgCudaSession rejected_session;
    EulerHdgRequest asymmetric = request;
    if (rank == 0) asymmetric.mesh = nullptr;
    const bool rejected = rejected_session.Initialize(asymmetric).status ==
        HdgEulerSessionStatus::rejected;
    if (!All(communicator, rejected)) return fail("asymmetric-rejection");
  }

  GenericEulerHdgCudaSession session;
  const bool initialized = session.Initialize(request).status ==
      HdgEulerSessionStatus::initialized;
  if (!All(communicator, initialized)) return fail("initialize");

  StateHash initial_interior{};
  StateHash initial_trace{};
  const auto initial_coordinates = session.coordinates();
  const bool initial_captured =
      CudaQualificationTestAccess::CaptureCommittedHashes(
          session, initial_interior, initial_trace);
  if (!All(communicator, initial_captured))
    return fail("initial-hash-capture");

  GenericEulerCudaPreparedStep rejected_authority;
  CudaPreparedOutputMetadataV1 rejected_output{};
  const bool first_prepared = session.Prepare(
      dt, rejected_authority, rejected_output).status ==
      HdgEulerSessionStatus::prepared;
  if (!All(communicator, first_prepared)) return fail("first-prepare");
  const bool first_assessed = AssessPreparedRodas3(
      session, rejected_authority, rejected_output, time_treatment);
  const bool first_engine_evidence =
      CudaQualificationTestAccess::VerifyPreparedCudaEngine(
          session, engine, time_treatment, communicator);
  if (!All(communicator, first_assessed && first_engine_evidence))
    return fail(first_assessed ? "first-evidence" : "first-assessment");
  const bool rejected = session.RejectPrepared(
      std::move(rejected_authority), rejected_output).status ==
      HdgEulerSessionStatus::aborted;
  if (!All(communicator, rejected)) return fail("reject");
  StateHash after_reject_interior{};
  StateHash after_reject_trace{};
  const bool rollback =
      CudaQualificationTestAccess::CaptureCommittedHashes(
          session, after_reject_interior, after_reject_trace) &&
      after_reject_interior == initial_interior &&
      after_reject_trace == initial_trace &&
      SameCoordinates(session.coordinates(), initial_coordinates);
  if (!All(communicator, rollback)) return fail("rollback");

  GenericEulerCudaPreparedStep authority;
  CudaPreparedOutputMetadataV1 output{};
  const bool prepared = session.Prepare(dt, authority, output).status ==
      HdgEulerSessionStatus::prepared;
  if (!All(communicator, prepared)) return fail("second-prepare");
  const bool assessed = AssessPreparedRodas3(
      session, authority, output, time_treatment);
  const bool engine_evidence =
      CudaQualificationTestAccess::VerifyPreparedCudaEngine(
          session, engine, time_treatment, communicator);
  if (!All(communicator, assessed && engine_evidence))
    return fail(assessed ? "second-evidence" : "second-assessment");
  const bool acknowledged = session.AcknowledgePreparedOutput(
      authority, output);
  if (!All(communicator, acknowledged)) return fail("acknowledge");
  const bool committed = session.Commit(std::move(authority), output).status ==
      HdgEulerSessionStatus::committed;
  if (!All(communicator, committed && session.coordinates().step == 1))
    return fail("commit");
  return true;
}

struct Options
{
  int active_ranks = 0;
  CudaEngineKind engine = CudaEngineKind::ordinary;
  EulerHdgTimeTreatment time_treatment =
      EulerHdgTimeTreatment::backward_euler;
};

bool ParseOptions(const int argc, char* argv[], Options& options) noexcept
{
  for (int index = 1; index < argc; ++index)
  {
    const std::string_view argument(argv[index]);
    if (argument == "--active-ranks")
    {
      if (++index >= argc) return false;
      options.active_ranks = std::atoi(argv[index]);
      continue;
    }
    if (argument == "--engine")
    {
      if (++index >= argc) return false;
      const std::string_view engine(argv[index]);
      if (engine == "ordinary")
        options.engine = CudaEngineKind::ordinary;
      else if (engine == "tensor")
        options.engine = CudaEngineKind::tensor_core;
      else
        return false;
      continue;
    }
    if (argument == "--time-treatment")
    {
      if (++index >= argc) return false;
      const std::string_view treatment(argv[index]);
      if (treatment == "backward-euler")
        options.time_treatment = EulerHdgTimeTreatment::backward_euler;
      else if (treatment == "partitioned-backward-euler")
        options.time_treatment =
            EulerHdgTimeTreatment::partitioned_backward_euler;
      else if (treatment == "rodas3")
        options.time_treatment = EulerHdgTimeTreatment::rodas3;
      else
        return false;
      continue;
    }
    return false;
  }
  return options.active_ranks == 1 || options.active_ranks == 2;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  Options options{};
  const bool options_ready = ParseOptions(argc, argv, options);
  const int active_ranks = options.active_ranks;

  int world_rank = -1;
  int world_ranks = 0;
  const bool world_ready =
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) == MPI_SUCCESS &&
      MPI_Comm_size(MPI_COMM_WORLD, &world_ranks) == MPI_SUCCESS;
  bool local = options_ready && world_ready &&
      world_ranks == active_ranks + 1;

  MPI_Comm active = MPI_COMM_NULL;
  if (local)
  {
    const int color = world_rank < active_ranks ? 0 : MPI_UNDEFINED;
    local = MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &active) ==
        MPI_SUCCESS;
  }
  if (local && active != MPI_COMM_NULL)
  {
    try
    {
      local = CheckActiveSubcommunicator(
          active, active_ranks, options.engine, options.time_treatment);
    }
    catch (...) { local = false; }
  }
  if (active != MPI_COMM_NULL)
    local = MPI_Comm_free(&active) == MPI_SUCCESS && local;

  const int input = local ? 1 : 0;
  int output = 0;
  const bool reduced = MPI_Allreduce(
      &input, &output, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS;
  return reduced && output == 1 ? 0 : 1;
}

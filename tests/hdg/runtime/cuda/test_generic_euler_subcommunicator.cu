#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "sim/hdg/state_io.hpp"
#include "test_cuda_trace_solver_routing.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

constexpr bool CudaPreparedIluAccountingValid(
    const CudaTensorCoreEvidence& tensor, const int trace_solves,
    const int linear_iterations, const generic::CudaEngineKind engine,
    const bool igr_active) noexcept
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
  if (trace_solves <= 0 || linear_iterations < 0 || tensor.ilu_type != 0 ||
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
  const std::uint64_t solves = static_cast<std::uint64_t>(trace_solves);
  const bool factor_work = factor_reuse_policy ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve
      ? tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
            tensor.ilu_factor_setups == solves &&
            tensor.ilu_factor_reuses == 0 &&
            tensor.ilu_factor_age_generations == 0
      : factor_reuse_policy ==
              CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
            tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
            tensor.ilu_factor_setups == 1 &&
            tensor.ilu_factor_reuses == solves - 1 &&
            tensor.ilu_factor_age_generations == solves - 1;
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
      Engine& engine, const generic::CudaEngineKind expected) noexcept
  {
    if (!engine.initialized_ || !engine.pending_.active ||
        !engine.pending_.staging)
      return false;
    const auto& evidence = engine.pending_.evidence;
    const auto& tensor = evidence.tensor_core;
    const bool igr_active = engine.numerics_.shape_.igr_active;
    const auto trace_policy = ResolveCudaTracePolicy(
        expected, igr_active, engine.config_.expected_rank_count,
        engine.config_.order, CudaTracePolicyKind::engine_default);
    return tensor.selected_engine == expected &&
        CudaTracePolicyDescriptorIsExact(trace_policy) &&
        tensor.requested_trace_policy ==
            CudaTracePolicyKind::engine_default &&
        tensor.resolved_trace_policy == trace_policy.resolved_selection &&
        tensor.trace_policy_rank_count == engine.config_.expected_rank_count &&
        tensor.trace_policy_order == engine.config_.order &&
        tensor.trace_preconditioner_configured &&
        tensor.trace_preconditioner == trace_policy.trace_preconditioner &&
        CudaPreparedIluAccountingValid(
            tensor, evidence.trace_solves, evidence.linear_iterations,
            expected, igr_active) &&
        CudaPreparedTraceSolverRoutingValid(
            tensor, evidence.trace_solves, expected,
            engine.config_.expected_rank_count) &&
        CudaPreparedTensorVolumeEvidenceValid(
            tensor, engine.numerics_.shape_, expected) &&
        tensor.fallback_count == 0 && evidence.no_fallback &&
        (expected == generic::CudaEngineKind::ordinary
             ? tensor.condensation_launches == 0 && tensor.mma_tiles == 0
             : tensor.hardware_supported &&
                   tensor.condensation_launches > 0 && tensor.mma_tiles > 0);
  }

  static bool VerifyPreparedCudaEngine(
      cuda::GenericEulerHdgCudaSession& session,
      const generic::CudaEngineKind expected) noexcept
  {
    if (session.borrowed_rank1_)
      return VerifyPreparedCudaEngine(*session.borrowed_rank1_, expected);
    if (session.borrowed_rank2_)
      return VerifyPreparedCudaEngine(*session.borrowed_rank2_, expected);
    return false;
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
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
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
                            const CudaEngineKind engine)
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
  return request;
}

bool CheckActiveSubcommunicator(const MPI_Comm communicator,
                                const int active_ranks,
                                const CudaEngineKind engine)
{
  int rank = -1;
  int ranks = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS ||
      ranks != active_ranks)
    return false;

  mfem::Mesh serial = PeriodicMesh();
  mfem::ParMesh mesh(communicator, serial);
  BorrowedFields fields = MakeFields(mesh);
  if (!All(communicator, HYPRE_GetError() == 0)) return false;
  EulerHdgRequest request = MakeRequest(mesh, fields, engine);

  if (ranks == 2)
  {
    GenericEulerHdgCudaSession rejected_session;
    EulerHdgRequest asymmetric = request;
    if (rank == 0) asymmetric.mesh = nullptr;
    const bool rejected = rejected_session.Initialize(asymmetric).status ==
        HdgEulerSessionStatus::rejected;
    if (!All(communicator, rejected)) return false;
  }

  GenericEulerHdgCudaSession session;
  const bool initialized = session.Initialize(request).status ==
      HdgEulerSessionStatus::initialized;
  if (!All(communicator, initialized)) return false;

  GenericEulerCudaPreparedStep authority;
  CudaPreparedOutputMetadataV1 output{};
  const bool prepared = session.Prepare(0.025, authority, output).status ==
      HdgEulerSessionStatus::prepared;
  if (!All(communicator, prepared)) return false;
  const bool engine_evidence =
      CudaQualificationTestAccess::VerifyPreparedCudaEngine(session, engine);
  if (!All(communicator, engine_evidence)) return false;
  const bool acknowledged = session.AcknowledgePreparedOutput(authority, output);
  if (!All(communicator, acknowledged)) return false;
  const bool committed = session.Commit(std::move(authority), output).status ==
      HdgEulerSessionStatus::committed;
  return All(communicator, committed && session.coordinates().step == 1);
}

struct Options
{
  int active_ranks = 0;
  CudaEngineKind engine = CudaEngineKind::ordinary;
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
          active, active_ranks, options.engine);
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

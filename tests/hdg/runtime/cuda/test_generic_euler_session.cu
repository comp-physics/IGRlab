#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "test_cuda_trace_solver_routing.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <string_view>
#include <type_traits>
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
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::HdgEulerSessionCapability;
using dg::hdg::generic::HdgEulerSessionCapabilityBit;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::runtime::BorrowedCudaRank1OneGpuRuntimeSelection;
using dg::hdg::runtime::BorrowedCudaRank2OneGpuHostStagedRuntimeSelection;
using dg::hdg::runtime::FailureReason;
using dg::hdg::runtime::StableIdentity;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;
using namespace dg::hdg::runtime::cuda_w8a;
using GenericPreparedOutput = CudaPreparedOutputMetadataV1;

inline constexpr std::string_view nonperiodic_topology_identity =
    "cuda.generic.quad.physical-boundary.v1";

CudaEngineKind selected_cuda_engine = CudaEngineKind::ordinary;

bool All(const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

mfem::Mesh PeriodicSquareMesh(const int n)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      n, n, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  return mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
}

mfem::Mesh NonperiodicSquareMesh(const int n)
{
  return mfem::Mesh::MakeCartesian2D(
      n, n, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
}

CudaConfig BorrowedConfig(mfem::ParMesh& mesh, const int ranks,
                          const int order, const int n)
{
  // Each returned config receives one unique stable slot.  The bounded pool
  // fails closed rather than reusing/overwriting a queued config's input.
  using storage_type = std::array<double, maximum_rank_elements *
      maximum_interior_dofs_per_element>;
  static std::array<storage_type, 128> fixture_storage{};
  static std::size_t fixture_count = 0;
  if (fixture_count == fixture_storage.size()) throw std::bad_alloc();
  storage_type& initial = fixture_storage[fixture_count++];
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, conserved_components, mfem::Ordering::byNODES);
  const std::size_t scalar_count = static_cast<std::size_t>(space.GetVSize() /
      conserved_components);
  if (scalar_count * conserved_components > initial.size())
    throw std::bad_alloc();
  for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
  {
    const double perturbation = 1.0e-3 * static_cast<double>(scalar + 1);
    initial[scalar] = 1.0 + perturbation;
    initial[scalar_count + scalar] = 0.20 + 0.25 * perturbation;
    initial[2 * scalar_count + scalar] = -0.10 + 0.15 * perturbation;
    initial[3 * scalar_count + scalar] = 2.60 + 0.10 * perturbation;
  }
  CudaConfig config{};
  config.order = order;
  config.mesh_n = n;
  config.requested_device = 0;
  config.cuda_engine = selected_cuda_engine;
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.use_borrowed_topology = true;
  config.borrowed_mesh = &mesh;
  config.borrowed_communicator = MPI_COMM_WORLD;
  config.use_borrowed_host_inputs = true;
  config.borrowed_initial = {initial.data(), scalar_count,
                             conserved_components};
  config.borrowed_source = {nullptr, 0, conserved_components};
  if (selected_cuda_engine == CudaEngineKind::tensor_core)
  {
    config.use_generic_numerical_policy = true;
    config.numerical_policy =
        dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  }
  return config;
}

struct HostFields
{
  std::vector<double> initial;
  std::vector<double> source;
  std::size_t scalar_count = 0;
};

HostFields NonuniformHostFields(mfem::ParMesh& mesh, const int order,
                                const bool nonzero_source)
{
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, conserved_components, mfem::Ordering::byNODES);
  HostFields fields;
  fields.scalar_count = static_cast<std::size_t>(space.GetVSize() /
      conserved_components);
  fields.initial.resize(fields.scalar_count * conserved_components);
  if (nonzero_source)
    fields.source.resize(fields.initial.size());
  for (std::size_t scalar = 0; scalar < fields.scalar_count; ++scalar)
  {
    const double value = 1.0e-3 * static_cast<double>(scalar + 1);
    fields.initial[scalar] = 1.0 + value;
    fields.initial[fields.scalar_count + scalar] = 0.20 + 0.25 * value;
    fields.initial[2 * fields.scalar_count + scalar] = -0.10 + 0.15 * value;
    fields.initial[3 * fields.scalar_count + scalar] = 2.60 + 0.10 * value;
    if (nonzero_source)
    {
      fields.source[scalar] = 0.01 * value;
      fields.source[fields.scalar_count + scalar] = -0.02 * value;
      fields.source[2 * fields.scalar_count + scalar] = 0.03 * value;
      fields.source[3 * fields.scalar_count + scalar] = 0.04 * value;
    }
  }
  return fields;
}

HostFields HighWaveHostFields(mfem::ParMesh& mesh, const int order)
{
  HostFields fields = NonuniformHostFields(mesh, order, false);
  for (std::size_t scalar = 0; scalar < fields.scalar_count; ++scalar)
  {
    const double rho = 1.0 + 1.0e-3 * static_cast<double>(scalar + 1);
    const double u = 2.0;
    const double v = 0.0;
    const double pressure = 1.0;
    fields.initial[scalar] = rho;
    fields.initial[fields.scalar_count + scalar] = rho * u;
    fields.initial[2 * fields.scalar_count + scalar] = rho * v;
    fields.initial[3 * fields.scalar_count + scalar] = pressure / 0.4 +
        0.5 * rho * (u * u + v * v);
  }
  return fields;
}

CudaConfig BorrowedDataConfig(mfem::ParMesh& mesh, const int ranks,
                              const int order, const int n,
                              const HostFields& fields)
{
  CudaConfig config{};
  config.order = order;
  config.mesh_n = n;
  config.requested_device = 0;
  config.cuda_engine = selected_cuda_engine;
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.use_borrowed_topology = true;
  config.borrowed_mesh = &mesh;
  config.borrowed_communicator = MPI_COMM_WORLD;
  config.use_borrowed_host_inputs = true;
  config.borrowed_initial = {fields.initial.data(), fields.scalar_count,
                             conserved_components};
  config.borrowed_source = fields.source.empty()
      ? CudaBorrowedHostL2Input{nullptr, 0, conserved_components}
      : CudaBorrowedHostL2Input{fields.source.data(), fields.scalar_count,
                                conserved_components};
  if (selected_cuda_engine == CudaEngineKind::tensor_core)
  {
    config.use_generic_numerical_policy = true;
    config.numerical_policy =
        dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  }
  return config;
}

CudaConfig LegacyBorrowedConfig(mfem::ParMesh& mesh, const int ranks,
                                const int order, const int n)
{
  CudaConfig config = BorrowedConfig(mesh, ranks, order, n);
  config.cuda_engine = CudaEngineKind::ordinary;
  config.use_borrowed_host_inputs = false;
  config.borrowed_initial = {nullptr, 0, conserved_components};
  config.borrowed_source = {nullptr, 0, conserved_components};
  return config;
}

bool SameValues(const std::vector<double>& left,
                const std::vector<double>& right) noexcept
{
  return left == right;
}

bool FullyFiniteOverwritten(const std::vector<double>& values,
                            const double sentinel) noexcept
{
  for (const double value : values)
    if (!std::isfinite(value) || value == sentinel) return false;
  return true;
}

bool SameCoordinates(const dg::hdg::generic::HdgEulerSessionCoordinates& left,
                     const dg::hdg::runtime::StateCoordinates& right) noexcept
{
  return left.time == right.time && left.dt == right.dt &&
      left.step == right.step && left.generation == right.generation;
}

bool SameNativeExport(const CudaPreparedOutputMetadataV1& left,
                      const CudaPreparedOutputMetadataV1& right) noexcept
{
  const auto same_attempt = [](const dg::hdg::runtime::AttemptIdentity& first,
                               const dg::hdg::runtime::AttemptIdentity& second)
  {
    return first.sequence == second.sequence &&
        first.committed_step == second.committed_step &&
        first.committed_generation == second.committed_generation &&
        first.retry_index == second.retry_index &&
        dg::hdg::runtime::detail::DoubleBits(first.dt) ==
            dg::hdg::runtime::detail::DoubleBits(second.dt);
  };
  const auto same_range = [](const CudaTransferLedgerRangeV1& first,
                             const CudaTransferLedgerRangeV1& second)
  {
    return first.begin_ordinal == second.begin_ordinal &&
        first.end_ordinal == second.end_ordinal &&
        first.ordered_digest == second.ordered_digest &&
        first.complete == second.complete;
  };
  return left.schema == right.schema && left.source_rank == right.source_rank &&
      left.rank_count == right.rank_count && same_attempt(left.attempt, right.attempt) &&
      left.facade_incarnation == right.facade_incarnation &&
      dg::hdg::runtime::detail::SameCheckpoint(left.descriptor, right.descriptor) &&
      left.local_mesh_hash == right.local_mesh_hash &&
      left.local_partition_hash == right.local_partition_hash &&
      left.local_topology_hash == right.local_topology_hash &&
      left.local_interior_hash == right.local_interior_hash &&
      left.local_trace_hash == right.local_trace_hash &&
      left.interior_count == right.interior_count &&
      left.owned_true_trace_count == right.owned_true_trace_count &&
      left.interior_source_token == right.interior_source_token &&
      left.trace_source_token == right.trace_source_token &&
      left.staging_generation == right.staging_generation &&
      left.staging_host_to_device_bytes == right.staging_host_to_device_bytes &&
      left.staging_device_to_host_bytes == right.staging_device_to_host_bytes &&
      left.staging_device_to_device_bytes == right.staging_device_to_device_bytes &&
      same_range(left.staging_range, right.staging_range) &&
      left.candidate_frozen == right.candidate_frozen &&
      left.checkpoint_staged == right.checkpoint_staged &&
      left.output_staged == right.output_staged &&
      left.successful_commit == right.successful_commit;
}

bool CheckTransaction(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  GenericEulerHdgCudaSession session;
  CudaConfig periodic_config = BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  // Periodic borrowed callers retain the empty physical-policy and topology
  // identity spellings; only an active static physical table needs an ID.
  const bool periodic_legacy_spelling =
      periodic_config.borrowed_boundary_policies.policies == nullptr &&
      periodic_config.borrowed_boundary_policies.count == 0 &&
      periodic_config.borrowed_global_topology.empty();
  const auto initialized = session.Initialize(periodic_config);
  const std::uint32_t expected_capabilities =
      HdgEulerSessionCapabilityBit(HdgEulerSessionCapability::prepared_commit) |
      HdgEulerSessionCapabilityBit(HdgEulerSessionCapability::prepared_output) |
      HdgEulerSessionCapabilityBit(
          HdgEulerSessionCapability::committed_cfl_diagnostics) |
      HdgEulerSessionCapabilityBit(
          HdgEulerSessionCapability::generic_borrowed_inputs);
  if (!All(periodic_legacy_spelling &&
           initialized.status == HdgEulerSessionStatus::initialized &&
           session.capabilities() == expected_capabilities &&
           (session.capabilities() & HdgEulerSessionCapabilityBit(
               HdgEulerSessionCapability::committed_cfl_diagnostics)) != 0 &&
           dg::hdg::generic::IsValidHdgEulerCommittedDiagnostics(
               session.diagnostics(), session.capabilities()) &&
           (session.capabilities() & HdgEulerSessionCapabilityBit(
               HdgEulerSessionCapability::generic_borrowed_inputs)) != 0))
    return false;

  GenericEulerCudaPreparedStep prepared{};
  GenericPreparedOutput native_export{};
  const auto staged = session.Prepare(0.025, prepared, native_export);
  const auto before_commit = session.coordinates();
  const bool native_staged = staged.status == HdgEulerSessionStatus::prepared &&
      native_export.candidate_frozen && native_export.checkpoint_staged &&
      native_export.staging_range.complete &&
      native_export.staging_device_to_device_bytes != 0 &&
      native_export.staging_host_to_device_bytes == 0 &&
      native_export.staging_device_to_host_bytes == 0 &&
      !native_export.output_staged && !native_export.successful_commit;
  const GenericPreparedOutput before_duplicate = native_export;
  GenericEulerCudaPreparedStep duplicate{};
  const auto duplicate_prepare = session.Prepare(duplicate, native_export);
  const bool duplicate_preserved =
      duplicate_prepare.status == HdgEulerSessionStatus::rejected &&
      SameNativeExport(native_export, before_duplicate);
  const std::uint64_t facade_incarnation = native_export.facade_incarnation;
  ++native_export.facade_incarnation;
  const bool aba_rejected = !session.AcknowledgePreparedOutput(
      prepared, native_export);
  native_export.facade_incarnation = facade_incarnation;
  const auto no_ack_commit = session.Commit(std::move(prepared), native_export);
  const bool acknowledged = session.AcknowledgePreparedOutput(
      prepared, native_export);
  if (!All(native_staged &&
           duplicate_preserved && aba_rejected &&
           no_ack_commit.status == HdgEulerSessionStatus::rejected &&
           session.coordinates().step == before_commit.step &&
           session.coordinates().generation == before_commit.generation &&
           acknowledged))
    return false;

  const auto committed = session.Commit(std::move(prepared), native_export);
  const auto after_commit = session.coordinates();
  if (!All(committed.status == HdgEulerSessionStatus::committed &&
           native_export.successful_commit &&
           SameCoordinates(after_commit, native_export.descriptor.coordinates) &&
           after_commit.step == before_commit.step + 1 &&
           after_commit.generation == before_commit.generation + 1))
    return false;

  GenericEulerCudaPreparedStep abort_authority{};
  GenericPreparedOutput aborted_export{};
  const auto diagnostics_before_abort = session.diagnostics();
  const auto abort_prepared = session.Prepare(0.013, abort_authority, aborted_export);
  GenericEulerHdgCudaSession foreign;
  const bool foreign_rejected = !foreign.AcknowledgePreparedOutput(
      abort_authority, aborted_export);
  const auto aborted = session.Abort(std::move(abort_authority), aborted_export);
  const bool stale_rejected = !session.AcknowledgePreparedOutput(
      abort_authority, aborted_export);
  const auto double_abort = session.Abort(std::move(abort_authority),
                                          aborted_export);
  if (!All(abort_prepared.status == HdgEulerSessionStatus::prepared &&
             foreign_rejected && aborted.status == HdgEulerSessionStatus::aborted &&
             session.coordinates().step == after_commit.step &&
             session.coordinates().generation == after_commit.generation &&
             session.diagnostics().coordinates.generation ==
                 diagnostics_before_abort.coordinates.generation &&
             session.diagnostics().backend_stable_dt ==
                 diagnostics_before_abort.backend_stable_dt &&
             !aborted_export.candidate_frozen &&
             !aborted_export.checkpoint_staged && !aborted_export.output_staged &&
             !aborted_export.successful_commit && stale_rejected &&
             double_abort.status == HdgEulerSessionStatus::rejected))
    return false;

  GenericEulerCudaPreparedStep retry_authority{};
  GenericPreparedOutput retry_export{};
  const auto retry_prepared = session.Prepare(0.013, retry_authority, retry_export);
  const bool retry_acknowledged = session.AcknowledgePreparedOutput(
      retry_authority, retry_export);
  const auto retry_committed = session.Commit(std::move(retry_authority),
                                              retry_export);
  return All(retry_prepared.status == HdgEulerSessionStatus::prepared &&
             retry_acknowledged &&
             retry_committed.status == HdgEulerSessionStatus::committed &&
             retry_export.successful_commit &&
             session.coordinates().step == after_commit.step + 1 &&
             session.coordinates().generation == after_commit.generation + 1);
}

bool CheckRankTwoCollectiveAdmission(const int ranks)
{
  if (ranks != 2) return true;
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GenericEulerHdgCudaSession session;
  const CudaConfig config = BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  const auto initialized = session.Initialize(config);
  if (initialized.status !=
      HdgEulerSessionStatus::initialized)
    return false;
  const std::size_t scalar_count = config.borrowed_initial.scalar_count;
  std::vector<double> legacy_committed(
      conserved_components * scalar_count, -11.0);
  std::vector<double> fields_committed(
      conserved_components * scalar_count, -13.0);
  const bool crossed_committed_export = rank == 0
      ? session.ExportCommittedInterior(
          {legacy_committed.data(), scalar_count, conserved_components})
      : session.ExportCommittedFields(
          {fields_committed.data(), nullptr, scalar_count});
  if (!All(!crossed_committed_export &&
           std::all_of(legacy_committed.begin(), legacy_committed.end(),
                       [](const double value) { return value == -11.0; }) &&
           std::all_of(fields_committed.begin(), fields_committed.end(),
                       [](const double value) { return value == -13.0; })))
    return false;
  GenericEulerCudaPreparedStep authority{};
  GenericPreparedOutput native_export{};
  const auto staged = session.Prepare(authority, native_export);
  std::vector<double> legacy_prepared(
      conserved_components * scalar_count, -17.0);
  std::vector<double> fields_prepared(
      conserved_components * scalar_count, -19.0);
  const bool crossed_prepared_export = rank == 0
      ? session.ExportPreparedInterior(
          authority, native_export,
          {legacy_prepared.data(), scalar_count, conserved_components})
      : session.ExportPreparedFields(
          authority, native_export,
          {fields_prepared.data(), nullptr, scalar_count});
  if (!All(staged.status == HdgEulerSessionStatus::prepared &&
           !crossed_prepared_export &&
           std::all_of(legacy_prepared.begin(), legacy_prepared.end(),
                       [](const double value) { return value == -17.0; }) &&
           std::all_of(fields_prepared.begin(), fields_prepared.end(),
                       [](const double value) { return value == -19.0; })))
    return false;
  const GenericPreparedOutput before_tamper = native_export;
  // This passes facade authority matching but makes one rank's checkpoint
  // descriptor stale, exercising engine-side collective atomicity.
  if (rank == 0) ++native_export.descriptor.coordinates.generation;
  const bool tamper_rejected = !session.AcknowledgePreparedOutput(
      authority, native_export);
  native_export.descriptor.coordinates = before_tamper.descriptor.coordinates;
  const bool tamper_atomic = tamper_rejected &&
      SameNativeExport(native_export, before_tamper) &&
      !native_export.output_staged && !native_export.successful_commit;
  const bool acknowledged = session.AcknowledgePreparedOutput(authority,
                                                               native_export);
  if (staged.status != HdgEulerSessionStatus::prepared || !tamper_atomic ||
      !acknowledged)
    return false;
  const auto mismatched = rank == 0
      ? session.Commit(std::move(authority), native_export)
      : session.Abort(std::move(authority), native_export);
  if (!All(mismatched.status == HdgEulerSessionStatus::rejected &&
           !native_export.successful_commit && session.coordinates().step == 0))
    return false;
  const auto initial_abort = session.Abort(std::move(authority), native_export);
  if (initial_abort.status !=
      HdgEulerSessionStatus::aborted)
    return false;

  GenericEulerCudaPreparedStep retry_authority{};
  GenericPreparedOutput retry_export{};
  const auto retry_prepared = session.Prepare(retry_authority, retry_export);
  if (retry_prepared.status !=
      HdgEulerSessionStatus::prepared)
    return false;
  GenericEulerCudaPreparedStep stale{};
  const auto stale_result = rank == 0
      ? session.Abort(std::move(stale), retry_export)
      : session.Abort(std::move(retry_authority), retry_export);
  if (!All(stale_result.status == HdgEulerSessionStatus::rejected &&
           session.coordinates().step == 0 && !retry_export.successful_commit))
    return false;
  const bool retry_acknowledged = session.AcknowledgePreparedOutput(
      retry_authority, retry_export);
  if (!retry_acknowledged)
    return false;
  const auto committed = session.Commit(std::move(retry_authority), retry_export);
  return All(committed.status == HdgEulerSessionStatus::committed &&
             retry_export.successful_commit && session.coordinates().step == 1 &&
             session.coordinates().generation == 1);
}

bool CheckRankTwoSessionIdentityAdmission(const int ranks)
{
  if (ranks != 2) return true;
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // This is collective-identity discrimination only.  The global CUDA
  // allocation baseline permits the last initialized engine (B) to advance;
  // A remains intentionally untouched after B establishes its baseline.
  GenericEulerHdgCudaSession a;
  GenericEulerHdgCudaSession b;
  const auto a_initialized = a.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  const auto b_initialized = b.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));

  GenericEulerCudaPreparedStep a_authority{};
  GenericEulerCudaPreparedStep b_authority{};
  GenericPreparedOutput a_export{};
  GenericPreparedOutput b_export{};
  const auto crossed_prepare = rank == 0
      ? a.Prepare(a_authority, a_export)
      : b.Prepare(b_authority, b_export);
  const bool a_untouched = a.coordinates().step == 0 &&
      a.coordinates().generation == 0 && !a_export.candidate_frozen &&
      !a_export.checkpoint_staged && !a_export.output_staged &&
      !a_export.successful_commit;
  const bool b_untouched = b.coordinates().step == 0 &&
      b.coordinates().generation == 0 && !b_export.candidate_frozen &&
      !b_export.checkpoint_staged && !b_export.output_staged &&
      !b_export.successful_commit;

  const auto b_prepared = b.Prepare(b_authority, b_export);
  const auto b_aborted = b.Abort(std::move(b_authority), b_export);
  return All(a_initialized.status == HdgEulerSessionStatus::initialized &&
             b_initialized.status == HdgEulerSessionStatus::initialized &&
             crossed_prepare.status == HdgEulerSessionStatus::rejected &&
             a_untouched && b_untouched &&
             b_prepared.status == HdgEulerSessionStatus::prepared &&
             b_aborted.status == HdgEulerSessionStatus::aborted &&
             b.coordinates().step == 0 && b.coordinates().generation == 0);
}

bool CheckForeignDestinationAuthority(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  GenericEulerHdgCudaSession owner;
  GenericEulerHdgCudaSession recipient;
  const auto recipient_initialized = recipient.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  const auto owner_initialized = owner.Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  if (!All(owner_initialized.status == HdgEulerSessionStatus::initialized &&
           recipient_initialized.status == HdgEulerSessionStatus::initialized))
    return false;

  GenericEulerCudaPreparedStep owner_authority{};
  GenericPreparedOutput owner_export{};
  const auto owner_prepared = owner.Prepare(owner_authority, owner_export);
  const GenericPreparedOutput before_rejection = owner_export;
  GenericPreparedOutput recipient_export{};
  const GenericPreparedOutput recipient_before_rejection = recipient_export;
  const auto rejected = recipient.Prepare(owner_authority, recipient_export);
  const bool preserved = SameNativeExport(owner_export, before_rejection) &&
      SameNativeExport(recipient_export, recipient_before_rejection);
  const auto aborted = owner.Abort(std::move(owner_authority), owner_export);

  GenericEulerCudaPreparedStep retry_authority{};
  GenericPreparedOutput retry_export{};
  const auto retry_prepared = owner.Prepare(retry_authority, retry_export);
  const GenericPreparedOutput retry_before_rejection = retry_export;
  const auto retry_rejected = recipient.Prepare(retry_authority, recipient_export);
  const bool retry_preserved =
      SameNativeExport(retry_export, retry_before_rejection) &&
      SameNativeExport(recipient_export, recipient_before_rejection);
  const bool retry_acknowledged = owner.AcknowledgePreparedOutput(
      retry_authority, retry_export);
  const auto committed = owner.Commit(std::move(retry_authority), retry_export);
  return All(owner_prepared.status == HdgEulerSessionStatus::prepared &&
             rejected.status == HdgEulerSessionStatus::rejected && preserved &&
             aborted.status == HdgEulerSessionStatus::aborted &&
             retry_prepared.status == HdgEulerSessionStatus::prepared &&
             retry_rejected.status == HdgEulerSessionStatus::rejected &&
             retry_preserved &&
             retry_acknowledged &&
             committed.status == HdgEulerSessionStatus::committed &&
             retry_export.successful_commit &&
             recipient.coordinates().step == 0 &&
             recipient.coordinates().generation == 0);
}

bool CheckAddressReuseAba(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  using storage_type = std::aligned_storage_t<sizeof(GenericEulerHdgCudaSession),
                                              alignof(GenericEulerHdgCudaSession)>;
  storage_type storage{};
  auto* retired = new (&storage) GenericEulerHdgCudaSession;
  const auto retired_initialized = retired->Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  if (!All(retired_initialized.status == HdgEulerSessionStatus::initialized))
  {
    retired->~GenericEulerHdgCudaSession();
    return false;
  }
  GenericEulerCudaPreparedStep retired_authority{};
  GenericPreparedOutput shared_export{};
  const auto retired_prepared = retired->Prepare(retired_authority, shared_export);
  if (!All(retired_prepared.status == HdgEulerSessionStatus::prepared))
  {
    retired->~GenericEulerHdgCudaSession();
    return false;
  }
  retired->~GenericEulerHdgCudaSession();

  auto* replacement = new (&storage) GenericEulerHdgCudaSession;
  const auto replacement_initialized = replacement->Initialize(BorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  GenericEulerCudaPreparedStep current_authority{};
  shared_export = {};
  const auto current_prepared = replacement->Prepare(current_authority,
                                                      shared_export);
  // Session and export addresses are both reused.  Rejection must therefore
  // come from the retired facade incarnation carried by the authority.
  const bool stale_acknowledged = replacement->AcknowledgePreparedOutput(
      retired_authority, shared_export);
  const auto current_aborted = replacement->Abort(std::move(current_authority),
                                                   shared_export);
  const bool passed = All(replacement_initialized.status ==
                              HdgEulerSessionStatus::initialized &&
                          current_prepared.status == HdgEulerSessionStatus::prepared &&
                          !stale_acknowledged &&
                          current_aborted.status == HdgEulerSessionStatus::aborted);
  replacement->~GenericEulerHdgCudaSession();
  return passed;
}

bool CheckP2Admission(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(functional_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  GenericEulerHdgCudaSession session;
  const auto result = session.Initialize(BorrowedConfig(
      mesh, ranks, functional_polynomial_order,
      functional_elements_per_direction));
  return All(result.status == HdgEulerSessionStatus::initialized &&
             session.coordinates().step == 0 && session.coordinates().dt == 0.0);
}

bool CheckRefinedDynamicCapacity(const int ranks)
{
  constexpr int refined_elements_per_direction = 6;
  mfem::Mesh serial = PeriodicSquareMesh(refined_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mesh.UniformRefinement();

  mfem::DG_Interface_FECollection trace_collection(
      functional_polynomial_order, dimension, mfem::FiniteElement::VALUE,
      mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace trace_space(
      &mesh, &trace_collection, conserved_components, mfem::Ordering::byNODES);
  const bool exceeds_fixed_capacity = mesh.GetNE() > maximum_rank_elements &&
      trace_space.GetTrueVSize() > maximum_true_trace_dofs;

  HostFields fields = NonuniformHostFields(
      mesh, functional_polynomial_order, true);
  const std::vector<double> initial_before = fields.initial;
  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(BorrowedDataConfig(
      mesh, ranks, functional_polynomial_order,
      refined_elements_per_direction, fields));
  const auto coordinates_before_export = session.coordinates();

  // Generic initialization owns persistent copies, so the caller can release
  // both field buffers before observing the committed initial interior.
  std::fill(fields.initial.begin(), fields.initial.end(), -21.0);
  std::fill(fields.source.begin(), fields.source.end(), -22.0);
  fields.initial.clear();
  fields.source.clear();
  fields.initial.shrink_to_fit();
  fields.source.shrink_to_fit();

  std::vector<double> committed(initial_before.size(), -23.0);
  const bool exported = session.ExportCommittedInterior(
      {committed.data(), fields.scalar_count, conserved_components});
  const auto coordinates_after_export = session.coordinates();
  return All(exceeds_fixed_capacity &&
             initialized.status == HdgEulerSessionStatus::initialized &&
             exported && FullyFiniteOverwritten(committed, -23.0) &&
             SameValues(committed, initial_before) &&
             coordinates_after_export.step == coordinates_before_export.step &&
             coordinates_after_export.generation ==
                 coordinates_before_export.generation &&
             coordinates_after_export.time == coordinates_before_export.time);
}

bool CheckLegacyFacadeAdmission(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(LegacyBorrowedConfig(
      mesh, ranks, smoke_polynomial_order, smoke_elements_per_direction));
  GenericEulerCudaPreparedStep authority{};
  GenericEulerHdgCudaSession::native_export_type native_export{};
  const auto coordinates_before_rejected = session.coordinates();
  const auto nonfixed = initialized.status == HdgEulerSessionStatus::initialized
      ? session.Prepare(0.025, authority, native_export)
      : dg::hdg::generic::HdgEulerSessionResult{};
  const auto prepared = initialized.status == HdgEulerSessionStatus::initialized
      ? session.Prepare(authority, native_export)
      : dg::hdg::generic::HdgEulerSessionResult{};
  const auto aborted = prepared.status == HdgEulerSessionStatus::prepared
      ? session.Abort(std::move(authority), native_export)
      : dg::hdg::generic::HdgEulerSessionResult{};
  return All(initialized.status == HdgEulerSessionStatus::initialized &&
             nonfixed.status == HdgEulerSessionStatus::rejected &&
             session.coordinates().step == 0 &&
             session.coordinates().generation == 0 &&
             coordinates_before_rejected.time == 0.0 &&
             prepared.status == HdgEulerSessionStatus::prepared &&
             native_export.interior_count != 0 &&
             native_export.owned_true_trace_count != 0 &&
             native_export.staging_range.complete &&
             aborted.status == HdgEulerSessionStatus::aborted &&
             session.coordinates().step == 0 &&
             session.coordinates().generation == 0);
}

bool CheckBorrowedDataBridge(const int ranks, const int order, const int n)
{
  std::vector<double> candidate;
  {
    mfem::Mesh serial = PeriodicSquareMesh(n);
    mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
    HostFields zero = NonuniformHostFields(mesh, order, false);
    const std::vector<double> initial_before = zero.initial;
    GenericEulerHdgCudaSession session;
    const auto initialized = session.Initialize(
        BorrowedDataConfig(mesh, ranks, order, n, zero));
    // Initialize has copied input/source bytes.  The caller may immediately
    // overwrite and release both buffers without changing the engine state.
    std::fill(zero.initial.begin(), zero.initial.end(), -3.0);
    zero.initial.clear();
    zero.initial.shrink_to_fit();
    std::vector<double> committed(initial_before.size(), -7.0);
    const bool committed_export = session.ExportCommittedInterior(
        {committed.data(), zero.scalar_count, conserved_components});
    std::vector<double> invalid_destination(initial_before.size(), -13.0);
    const bool invalid_export = session.ExportCommittedInterior(
        {invalid_destination.data(), zero.scalar_count - 1,
         conserved_components});
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> divergent_destination(initial_before.size(), -15.0);
    const bool divergent_export = ranks != 2 ||
        session.ExportCommittedInterior(
            {divergent_destination.data(), rank == 0 ? zero.scalar_count :
             zero.scalar_count - 1, conserved_components});
    if (!All(initialized.status == HdgEulerSessionStatus::initialized &&
             committed_export && SameValues(committed, initial_before) &&
             !invalid_export &&
             std::all_of(invalid_destination.begin(),
                         invalid_destination.end(),
                         [](const double value) { return value == -13.0; }) &&
             (ranks != 2 || (!divergent_export && std::all_of(
                 divergent_destination.begin(), divergent_destination.end(),
                 [](const double value) { return value == -15.0; })))))
      return false;

    GenericEulerCudaPreparedStep authority{};
    GenericPreparedOutput native_export{};
    const auto prepared = session.Prepare(authority, native_export);
    const bool engine_evidence =
        CudaQualificationTestAccess::VerifyPreparedCudaEngine(
            session, selected_cuda_engine);
    candidate.assign(initial_before.size(), -9.0);
    const bool prepared_export = session.ExportPreparedInterior(
        authority, native_export,
        {candidate.data(), zero.scalar_count, conserved_components});
    const bool acknowledged = session.AcknowledgePreparedOutput(authority,
                                                                 native_export);
    const auto committed_step = session.Commit(std::move(authority),
                                                native_export);
    std::vector<double> committed_after(initial_before.size(), -17.0);
    const bool committed_after_export = session.ExportCommittedInterior(
        {committed_after.data(), zero.scalar_count, conserved_components});
    if (!All(prepared.status == HdgEulerSessionStatus::prepared &&
             engine_evidence &&
             prepared_export && FullyFiniteOverwritten(candidate, -9.0) &&
             acknowledged &&
             committed_step.status == HdgEulerSessionStatus::committed &&
             committed_after_export && SameValues(committed_after, candidate)))
      return false;
  }

  // The persistent CUDA runtime owns process-global allocation and transport
  // ledgers.  Retain only exported bytes across the source comparison; do not
  // overlap two live numerical engines.

  // A static source has its own persistent source slice: with identical ICs,
  // the prepared physical candidate must not collapse to the zero-source one.
  mfem::Mesh source_serial = PeriodicSquareMesh(n);
  mfem::ParMesh source_mesh(MPI_COMM_WORLD, source_serial);
  HostFields forced = NonuniformHostFields(source_mesh, order, true);
  GenericEulerHdgCudaSession forced_session;
  const auto forced_initialized = forced_session.Initialize(
      BorrowedDataConfig(source_mesh, ranks, order, n, forced));
  std::fill(forced.initial.begin(), forced.initial.end(), -5.0);
  std::fill(forced.source.begin(), forced.source.end(), -6.0);
  forced.initial.clear();
  forced.source.clear();
  forced.initial.shrink_to_fit();
  forced.source.shrink_to_fit();
  GenericEulerCudaPreparedStep forced_authority{};
  GenericPreparedOutput forced_export{};
  const auto forced_prepared = forced_session.Prepare(forced_authority,
                                                       forced_export);
  const bool forced_engine_evidence =
      CudaQualificationTestAccess::VerifyPreparedCudaEngine(
          forced_session, selected_cuda_engine);
  std::vector<double> forced_candidate(candidate.size(), -9.0);
  const bool forced_output = forced_session.ExportPreparedInterior(
      forced_authority, forced_export,
      {forced_candidate.data(), forced.scalar_count, conserved_components});
  const auto forced_aborted = forced_session.Abort(std::move(forced_authority),
                                                    forced_export);
  return All(forced_initialized.status == HdgEulerSessionStatus::initialized &&
             forced_prepared.status == HdgEulerSessionStatus::prepared &&
             forced_engine_evidence &&
             forced_output && FullyFiniteOverwritten(forced_candidate, -9.0) &&
             forced_aborted.status ==
                 HdgEulerSessionStatus::aborted &&
             !SameValues(candidate, forced_candidate));
}

bool CheckBorrowedLaggedLlfHighWave(const int ranks, const int order,
                                    const int n)
{
  mfem::Mesh serial = PeriodicSquareMesh(n);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  HostFields fields = HighWaveHostFields(mesh, order);
  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(
      BorrowedDataConfig(mesh, ranks, order, n, fields));
  if (!All(initialized.status == HdgEulerSessionStatus::initialized))
    return false;

  std::vector<double> prepared_one(fields.initial.size(), -11.0);
  std::vector<double> prepared_two(fields.initial.size(), -12.0);
  GenericEulerCudaPreparedStep first{};
  GenericPreparedOutput first_export{};
  const auto first_prepared = session.Prepare(first, first_export);
  const bool first_bytes = session.ExportPreparedInterior(
      first, first_export, {prepared_one.data(), fields.scalar_count,
                            conserved_components});
  const bool first_ack = session.AcknowledgePreparedOutput(first, first_export);
  const auto first_committed = session.Commit(std::move(first), first_export);
  std::vector<double> committed_one(fields.initial.size(), -13.0);
  const bool first_committed_bytes = session.ExportCommittedInterior(
      {committed_one.data(), fields.scalar_count, conserved_components});

  GenericEulerCudaPreparedStep second{};
  GenericPreparedOutput second_export{};
  const auto second_prepared = session.Prepare(second, second_export);
  const bool second_bytes = session.ExportPreparedInterior(
      second, second_export, {prepared_two.data(), fields.scalar_count,
                               conserved_components});
  const bool second_ack = session.AcknowledgePreparedOutput(second, second_export);
  const auto second_committed = session.Commit(std::move(second), second_export);
  std::vector<double> committed_two(fields.initial.size(), -14.0);
  const bool second_committed_bytes = session.ExportCommittedInterior(
      {committed_two.data(), fields.scalar_count, conserved_components});

  return All(first_prepared.status == HdgEulerSessionStatus::prepared &&
             first_bytes && FullyFiniteOverwritten(prepared_one, -11.0) &&
             first_ack && first_committed.status == HdgEulerSessionStatus::committed &&
             first_committed_bytes && SameValues(committed_one, prepared_one) &&
             second_prepared.status == HdgEulerSessionStatus::prepared &&
             second_bytes && FullyFiniteOverwritten(prepared_two, -12.0) &&
             second_ack && second_committed.status == HdgEulerSessionStatus::committed &&
             second_committed_bytes && SameValues(committed_two, prepared_two) &&
             session.coordinates().step == 2 &&
             session.coordinates().generation == 2 &&
             session.coordinates().time == 2.0 * fixed_dt);
}

bool CheckPreparedDtAdmission(const int ranks)
{
  mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  GenericEulerHdgCudaSession session;
  if (!All(session.Initialize(BorrowedConfig(mesh, ranks,
      smoke_polynomial_order, smoke_elements_per_direction)).status ==
      HdgEulerSessionStatus::initialized)) return false;
  GenericEulerCudaPreparedStep authority{};
  GenericPreparedOutput output{};
  const bool zero_rejected = session.Prepare(0.0, authority, output).status ==
      HdgEulerSessionStatus::rejected;
  const bool nan_rejected = session.Prepare(std::numeric_limits<double>::quiet_NaN(),
      authority, output).status == HdgEulerSessionStatus::rejected;
  const bool infinity_rejected = session.Prepare(
      std::numeric_limits<double>::infinity(), authority, output).status ==
      HdgEulerSessionStatus::rejected;
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool divergent_rejected = true;
  if (ranks == 2)
  {
    const double divergent_dt = rank == 0 ? 0.0125 : 0.025;
    divergent_rejected = session.Prepare(divergent_dt, authority, output).status ==
        HdgEulerSessionStatus::rejected;
  }
  const auto prepared = session.Prepare(0.0125, authority, output);
  const auto aborted = prepared.status == HdgEulerSessionStatus::prepared
      ? session.Abort(std::move(authority), output)
      : dg::hdg::generic::HdgEulerSessionResult{};
  return All(zero_rejected && nan_rejected && infinity_rejected &&
      divergent_rejected && prepared.status == HdgEulerSessionStatus::prepared &&
      aborted.status == HdgEulerSessionStatus::aborted);
}

bool Rejected(GenericEulerHdgCudaSession& session,
              const CudaConfig& config) noexcept
{
  const auto result = session.Initialize(config);
  return result.status == HdgEulerSessionStatus::rejected &&
      result.coordinates.step == 0 && result.coordinates.generation == 0;
}

bool DirectFixedOracleRejected(const CudaConfig& config, const int ranks)
{
  CudaRunEvidence evidence{};
  if (ranks == 1)
  {
    auto engine = std::make_unique<
        PersistentCudaSmoothEulerEngine<
            BorrowedCudaRank1OneGpuRuntimeSelection>>(config);
    return !engine->Initialize(evidence) &&
        evidence.status == CudaRunStatus::setup_failure &&
        evidence.setup_origin ==
            dg::hdg::runtime::FailureReason::collective_protocol &&
        evidence.device == -1;
  }
  auto engine = std::make_unique<
      PersistentCudaSmoothEulerEngine<
          BorrowedCudaRank2OneGpuHostStagedRuntimeSelection>>(config);
  return !engine->Initialize(evidence) &&
      evidence.status == CudaRunStatus::setup_failure &&
      evidence.setup_origin ==
          dg::hdg::runtime::FailureReason::collective_protocol &&
      evidence.device == -1;
}

bool DirectBorrowedRejectedBeforeDevice(const CudaConfig& config,
                                        const int ranks,
                                        const FailureReason expected_origin =
                                            FailureReason::collective_protocol)
{
  CudaRunEvidence evidence{};
  if (ranks == 1)
  {
    auto engine = std::make_unique<
        PersistentCudaSmoothEulerEngine<
            BorrowedCudaRank1OneGpuRuntimeSelection>>(config);
    return !engine->Initialize(evidence) && evidence.device == -1 &&
        evidence.status == CudaRunStatus::setup_failure &&
        evidence.setup_origin == expected_origin;
  }
  auto engine = std::make_unique<
      PersistentCudaSmoothEulerEngine<
          BorrowedCudaRank2OneGpuHostStagedRuntimeSelection>>(config);
  return !engine->Initialize(evidence) && evidence.device == -1 &&
      evidence.status == CudaRunStatus::setup_failure &&
      evidence.setup_origin == expected_origin;
}

std::array<CudaStaticBoundaryPolicy, 4> StaticPhysicalPolicies()
{
  const std::array<double, conserved_components> reference{{1.0, 2.0, 0.0,
      4.5}};
  const std::array<double, conserved_components> unused{{0.0, 0.0, 0.0,
      0.0}};
  // MFEM Cartesian boundary attributes are bottom/right/top/left = 1/2/3/4.
  // Exercise a horizontal left-in/right-out channel with slip walls.
  return {{{1, CudaStaticBoundaryPolicyKind::slip_wall, unused},
           {2, CudaStaticBoundaryPolicyKind::supersonic_outlet, unused},
           {3, CudaStaticBoundaryPolicyKind::slip_wall, unused},
           {4, CudaStaticBoundaryPolicyKind::supersonic_inlet, reference}}};
}

bool BoundaryPolicyViolationsRejectedBeforeDevice(mfem::ParMesh& mesh,
                                                  const int ranks,
                                                  const int order,
                                                  const int n)
{
  HostFields fields = NonuniformHostFields(mesh, order, false);
  CudaConfig base = BorrowedDataConfig(mesh, ranks, order, n, fields);
  base.borrowed_global_topology =
      StableIdentity::Copy(nonperiodic_topology_identity);

  std::array<CudaStaticBoundaryPolicy, 4> unknown = StaticPhysicalPolicies();
  for (std::size_t policy = 0; policy < unknown.size(); ++policy)
    unknown[policy].attribute = static_cast<int>(policy) + 5;
  CudaConfig unknown_config = base;
  unknown_config.borrowed_boundary_policies = {unknown.data(), unknown.size()};
  GenericEulerHdgCudaSession unknown_session;
  const bool unknown_rejected_pre_device = Rejected(unknown_session,
                                                    unknown_config);

  std::array<CudaStaticBoundaryPolicy, 4> invalid_far_field_reference =
      StaticPhysicalPolicies();
  invalid_far_field_reference[0].kind =
      CudaStaticBoundaryPolicyKind::far_field;
  CudaConfig invalid_far_field_reference_config = base;
  invalid_far_field_reference_config.borrowed_boundary_policies = {
      invalid_far_field_reference.data(), invalid_far_field_reference.size()};

  std::array<CudaStaticBoundaryPolicy, 4> nonfinite = StaticPhysicalPolicies();
  nonfinite[3].reference_state[0] = std::numeric_limits<double>::quiet_NaN();
  CudaConfig nonfinite_config = base;
  nonfinite_config.borrowed_boundary_policies = {nonfinite.data(), nonfinite.size()};

  std::array<CudaStaticBoundaryPolicy, 4> noncanonical = StaticPhysicalPolicies();
  noncanonical[1].reference_state[0] = 1.0;
  CudaConfig noncanonical_config = base;
  noncanonical_config.borrowed_boundary_policies =
      {noncanonical.data(), noncanonical.size()};

  std::array<CudaStaticBoundaryPolicy, 4> signed_zero = StaticPhysicalPolicies();
  signed_zero[1].reference_state[0] = -0.0;
  CudaConfig signed_zero_config = base;
  signed_zero_config.borrowed_boundary_policies =
      {signed_zero.data(), signed_zero.size()};

  std::array<CudaStaticBoundaryPolicy, 4> invalid_outlet_pressure =
      StaticPhysicalPolicies();
  invalid_outlet_pressure[1].kind =
      CudaStaticBoundaryPolicyKind::subsonic_outlet_pressure;
  CudaConfig invalid_outlet_pressure_config = base;
  invalid_outlet_pressure_config.borrowed_boundary_policies = {
      invalid_outlet_pressure.data(), invalid_outlet_pressure.size()};

  std::array<CudaStaticBoundaryPolicy, 4> unused_pressure =
      StaticPhysicalPolicies();
  unused_pressure[1].pressure_out = 0.9;
  CudaConfig unused_pressure_config = base;
  unused_pressure_config.borrowed_boundary_policies = {
      unused_pressure.data(), unused_pressure.size()};

  std::array<CudaStaticBoundaryPolicy, 4> signed_zero_pressure =
      StaticPhysicalPolicies();
  signed_zero_pressure[1].pressure_out = -0.0;
  CudaConfig signed_zero_pressure_config = base;
  signed_zero_pressure_config.borrowed_boundary_policies = {
      signed_zero_pressure.data(), signed_zero_pressure.size()};

  std::array<CudaStaticBoundaryPolicy, 5> unused_extra{};
  const std::array<CudaStaticBoundaryPolicy, 4> valid =
      StaticPhysicalPolicies();
  std::copy(valid.begin(), valid.end(), unused_extra.begin());
  unused_extra[4] = {5, CudaStaticBoundaryPolicyKind::slip_wall,
                     {{0.0, 0.0, 0.0, 0.0}}};
  CudaConfig unused_extra_config = base;
  unused_extra_config.borrowed_boundary_policies =
      {unused_extra.data(), unused_extra.size()};
  GenericEulerHdgCudaSession unused_extra_session;
  const bool unused_extra_session_rejected =
      Rejected(unused_extra_session, unused_extra_config);
  const bool unused_extra_direct_rejected =
      DirectBorrowedRejectedBeforeDevice(
          unused_extra_config, ranks, FailureReason::backend_failure);
  // Every direct initialization performs collective host preflight.  Preserve
  // the same call sequence on every rank even when an earlier case fails.
  const bool invalid_far_field_reference_rejected =
      DirectBorrowedRejectedBeforeDevice(
          invalid_far_field_reference_config, ranks);
  const bool nonfinite_rejected =
      DirectBorrowedRejectedBeforeDevice(nonfinite_config, ranks);
  const bool noncanonical_rejected =
      DirectBorrowedRejectedBeforeDevice(noncanonical_config, ranks);
  const bool signed_zero_rejected =
      DirectBorrowedRejectedBeforeDevice(signed_zero_config, ranks);
  const bool invalid_outlet_pressure_rejected =
      DirectBorrowedRejectedBeforeDevice(
          invalid_outlet_pressure_config, ranks);
  const bool unused_pressure_rejected =
      DirectBorrowedRejectedBeforeDevice(unused_pressure_config, ranks);
  const bool signed_zero_pressure_rejected =
      DirectBorrowedRejectedBeforeDevice(
          signed_zero_pressure_config, ranks);

  return unknown_rejected_pre_device &&
      invalid_far_field_reference_rejected && nonfinite_rejected &&
      noncanonical_rejected && signed_zero_rejected &&
      invalid_outlet_pressure_rejected && unused_pressure_rejected &&
      signed_zero_pressure_rejected && unused_extra_session_rejected &&
      unused_extra_direct_rejected;
}

bool CheckStaticPhysicalBoundaryTransaction(const int ranks, const int order,
                                            const int n)
{
  mfem::Mesh serial = NonperiodicSquareMesh(n);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  HostFields fields = HighWaveHostFields(mesh, order);
  const std::array<CudaStaticBoundaryPolicy, 4> fixture =
      StaticPhysicalPolicies();
  std::vector<CudaStaticBoundaryPolicy> policies(fixture.begin(), fixture.end());
  CudaConfig config = BorrowedDataConfig(mesh, ranks, order, n, fields);
  config.borrowed_global_topology =
      StableIdentity::Copy(nonperiodic_topology_identity);
  config.borrowed_boundary_policies = {policies.data(), policies.size()};
  GenericEulerHdgCudaSession session;
  const auto initialized = session.Initialize(config);
  // Initialize owns a copied host/device policy projection.  Destroy the
  // caller table before the first numerical attempt to catch any retained
  // table pointer in the session or kernel launch path.
  policies.clear();
  policies.shrink_to_fit();
  const bool initialized_everywhere = All(
      initialized.status == HdgEulerSessionStatus::initialized);
  if (!initialized_everywhere) return false;
  std::vector<double> before(fields.initial.size(), -9.0);
  const bool initial_export = session.ExportCommittedInterior(
      {before.data(), fields.scalar_count, conserved_components});
  if (!All(initial_export)) return false;
  GenericEulerCudaPreparedStep authority{};
  GenericPreparedOutput output{};
  const auto prepared = session.Prepare(0.0125, authority, output);
  const bool prepared_everywhere = All(
      prepared.status == HdgEulerSessionStatus::prepared);
  if (!prepared_everywhere) return false;
  const auto coordinates_before_abort = session.coordinates();
  const auto aborted = session.Abort(std::move(authority), output);
  const bool aborted_everywhere = All(
      aborted.status == HdgEulerSessionStatus::aborted);
  if (!aborted_everywhere) return false;
  std::vector<double> after_abort(fields.initial.size(), -7.0);
  const bool abort_export = session.ExportCommittedInterior(
      {after_abort.data(), fields.scalar_count, conserved_components});
  if (!All(abort_export)) return false;
  GenericEulerCudaPreparedStep retry{};
  GenericPreparedOutput retry_output{};
  const auto retry_prepared = session.Prepare(0.0125, retry, retry_output);
  const bool retry_prepared_everywhere = All(
      retry_prepared.status == HdgEulerSessionStatus::prepared);
  if (!retry_prepared_everywhere) return false;
  const bool acknowledged = session.AcknowledgePreparedOutput(
      retry, retry_output);
  if (!All(acknowledged)) return false;
  const auto committed = session.Commit(std::move(retry), retry_output);
  return All(initial_export && abort_export && SameValues(before, after_abort) &&
      session.coordinates().step == coordinates_before_abort.step + 1 &&
      committed.status == HdgEulerSessionStatus::committed &&
      retry_output.successful_commit);
}

bool CheckNegatives(const int ranks)
{
  mfem::Mesh periodic_serial = PeriodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh periodic(MPI_COMM_WORLD, periodic_serial);
  mfem::Mesh boundary_serial = NonperiodicSquareMesh(smoke_elements_per_direction);
  mfem::ParMesh boundary(MPI_COMM_WORLD, boundary_serial);

  CudaConfig nonperiodic = BorrowedConfig(
      boundary, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  CudaConfig invalid_mesh_n = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  invalid_mesh_n.mesh_n = -1;
  CudaConfig unsupported_order = BorrowedConfig(
      periodic, ranks, dg::hdg::generic::maximum_polynomial_order + 1,
      smoke_elements_per_direction);
  CudaConfig wrong_device = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  wrong_device.requested_device = 1;
  CudaConfig wrong_ranks = BorrowedConfig(
      periodic, 3, smoke_polynomial_order, smoke_elements_per_direction);
  CudaConfig invalid_config_dt = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  invalid_config_dt.dt = 0.0;
  CudaConfig capture_oracle = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  capture_oracle.capture_oracle = true;
  CudaConfig injection = BorrowedConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
  injection.injection.enabled = true;

  HostFields fields = NonuniformHostFields(periodic, smoke_polynomial_order,
                                           true);
  const std::vector<double> fields_initial_before = fields.initial;
  const std::vector<double> fields_source_before = fields.source;
  CudaConfig null_initial = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  null_initial.borrowed_initial.coefficients = nullptr;
  CudaConfig wrong_initial_count = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  --wrong_initial_count.borrowed_initial.scalar_count;
  CudaConfig wrong_initial_components = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  wrong_initial_components.borrowed_initial.components = 3;
  CudaConfig null_source = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  null_source.borrowed_source.coefficients = nullptr;
  HostFields initial_nan = fields;
  initial_nan.initial[0] = std::numeric_limits<double>::quiet_NaN();
  CudaConfig nonfinite_initial = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      initial_nan);
  HostFields source_infinite = fields;
  source_infinite.source[0] = std::numeric_limits<double>::infinity();
  CudaConfig nonfinite_source = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      source_infinite);
  HostFields nonphysical = fields;
  // Finite coefficients with negative pressure must fail pre-device borrowed
  // input admission; this is distinct from the existing NaN/Inf checks.
  nonphysical.initial[3 * nonphysical.scalar_count] = 0.01;
  CudaConfig nonphysical_initial = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      nonphysical);
  std::array<CudaStaticBoundaryPolicy, 4> policies = StaticPhysicalPolicies();
  HostFields boundary_fields = NonuniformHostFields(
      boundary, smoke_polynomial_order, false);
  CudaConfig unknown_boundary_policy = BorrowedDataConfig(
      boundary, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      boundary_fields);
  unknown_boundary_policy.borrowed_global_topology =
      StableIdentity::Copy(nonperiodic_topology_identity);
  for (std::size_t policy = 0; policy < policies.size(); ++policy)
    policies[policy].attribute = static_cast<int>(policy) + 5;
  unknown_boundary_policy.borrowed_boundary_policies =
      {policies.data(), policies.size()};
  std::array<CudaStaticBoundaryPolicy, 4> invalid_far_field_reference =
      StaticPhysicalPolicies();
  invalid_far_field_reference[0].kind =
      CudaStaticBoundaryPolicyKind::far_field;
  CudaConfig invalid_far_field_reference_policy = unknown_boundary_policy;
  invalid_far_field_reference_policy.borrowed_boundary_policies = {
      invalid_far_field_reference.data(), invalid_far_field_reference.size()};
  std::array<CudaStaticBoundaryPolicy, 4> nonfinite = StaticPhysicalPolicies();
  nonfinite[3].reference_state[0] = std::numeric_limits<double>::quiet_NaN();
  CudaConfig nonfinite_boundary_policy = unknown_boundary_policy;
  nonfinite_boundary_policy.borrowed_boundary_policies =
      {nonfinite.data(), nonfinite.size()};
  std::array<CudaStaticBoundaryPolicy, 4> noncanonical = StaticPhysicalPolicies();
  noncanonical[1].reference_state[0] = 1.0;
  CudaConfig noncanonical_boundary_policy = unknown_boundary_policy;
  noncanonical_boundary_policy.borrowed_boundary_policies =
      {noncanonical.data(), noncanonical.size()};
  std::array<CudaStaticBoundaryPolicy, 4> signed_zero = StaticPhysicalPolicies();
  signed_zero[1].reference_state[0] = -0.0;
  CudaConfig signed_zero_boundary_policy = unknown_boundary_policy;
  signed_zero_boundary_policy.borrowed_boundary_policies =
      {signed_zero.data(), signed_zero.size()};
  const bool signed_zero_table_rejected =
      !ValidateCudaStaticBoundaryPolicyTable(
          signed_zero_boundary_policy.borrowed_boundary_policies);
  std::array<CudaStaticBoundaryPolicy, 4> invalid_outlet_pressure =
      StaticPhysicalPolicies();
  invalid_outlet_pressure[1].kind =
      CudaStaticBoundaryPolicyKind::subsonic_outlet_pressure;
  CudaConfig invalid_outlet_pressure_policy = unknown_boundary_policy;
  invalid_outlet_pressure_policy.borrowed_boundary_policies = {
      invalid_outlet_pressure.data(), invalid_outlet_pressure.size()};
  std::array<CudaStaticBoundaryPolicy, 4> unused_pressure =
      StaticPhysicalPolicies();
  unused_pressure[1].pressure_out = 0.9;
  CudaConfig unused_pressure_policy = unknown_boundary_policy;
  unused_pressure_policy.borrowed_boundary_policies = {
      unused_pressure.data(), unused_pressure.size()};
  std::array<CudaStaticBoundaryPolicy, 4> signed_zero_pressure =
      StaticPhysicalPolicies();
  signed_zero_pressure[1].pressure_out = -0.0;
  CudaConfig signed_zero_pressure_policy = unknown_boundary_policy;
  signed_zero_pressure_policy.borrowed_boundary_policies = {
      signed_zero_pressure.data(), signed_zero_pressure.size()};
  CudaConfig noncanonical_periodic_identity = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  noncanonical_periodic_identity.borrowed_global_topology =
      StableIdentity::Copy(nonperiodic_topology_identity);
  std::array<CudaStaticBoundaryPolicy, 4> unused_periodic_policies =
      StaticPhysicalPolicies();
  CudaConfig unused_periodic_policy = BorrowedDataConfig(
      periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
      fields);
  unused_periodic_policy.borrowed_global_topology =
      StableIdentity::Copy(nonperiodic_topology_identity);
  unused_periodic_policy.borrowed_boundary_policies =
      {unused_periodic_policies.data(), unused_periodic_policies.size()};

  GenericEulerHdgCudaSession one, two, three, four, five, six, seven, eight;
  const bool nonperiodic_rejected = Rejected(one, nonperiodic);
  const bool invalid_mesh_n_rejected = Rejected(two, invalid_mesh_n);
  const bool unsupported_order_rejected = Rejected(three, unsupported_order);
  const bool wrong_device_rejected = Rejected(four, wrong_device);
  const bool wrong_ranks_rejected = Rejected(five, wrong_ranks);
  const bool invalid_config_dt_rejected = Rejected(six, invalid_config_dt);
  const bool capture_oracle_rejected = Rejected(seven, capture_oracle);
  const bool direct_capture_oracle_rejected = DirectFixedOracleRejected(
      capture_oracle, ranks);
  const bool injection_rejected = Rejected(eight, injection);
  GenericEulerHdgCudaSession nine, ten, eleven, twelve, thirteen, fourteen,
      fifteen;
  const bool null_initial_rejected = Rejected(nine, null_initial);
  const bool wrong_initial_count_rejected = Rejected(ten, wrong_initial_count);
  const bool wrong_initial_components_rejected = Rejected(
      eleven, wrong_initial_components);
  const bool null_source_rejected = Rejected(twelve, null_source);
  const bool nonfinite_initial_rejected = Rejected(thirteen, nonfinite_initial);
  const bool nonfinite_source_rejected = Rejected(fourteen, nonfinite_source);
  const bool nonphysical_initial_rejected = Rejected(fifteen,
                                                      nonphysical_initial);
  GenericEulerHdgCudaSession eighteen, nineteen, twenty, twenty_one, twenty_two,
      twenty_five, twenty_six, twenty_seven;
  const bool unknown_boundary_rejected = Rejected(eighteen, unknown_boundary_policy);
  const bool invalid_far_field_reference_rejected = Rejected(
      nineteen, invalid_far_field_reference_policy);
  const bool nonfinite_boundary_rejected = Rejected(
      twenty, nonfinite_boundary_policy);
  const bool noncanonical_boundary_rejected = Rejected(
      twenty_one, noncanonical_boundary_policy);
  const bool signed_zero_boundary_rejected = Rejected(
      twenty_two, signed_zero_boundary_policy);
  const bool invalid_outlet_pressure_rejected = Rejected(
      twenty_five, invalid_outlet_pressure_policy);
  const bool unused_pressure_rejected = Rejected(
      twenty_six, unused_pressure_policy);
  const bool signed_zero_pressure_rejected = Rejected(
      twenty_seven, signed_zero_pressure_policy);
  GenericEulerHdgCudaSession twenty_three;
  const bool noncanonical_periodic_identity_rejected = Rejected(
      twenty_three, noncanonical_periodic_identity);
  GenericEulerHdgCudaSession twenty_four;
  const bool unused_periodic_policy_session_rejected = Rejected(
      twenty_four, unused_periodic_policy);
  const bool unused_periodic_policy_direct_rejected =
      DirectBorrowedRejectedBeforeDevice(
          unused_periodic_policy, ranks, FailureReason::backend_failure);
  // Each call enters the same collective pre-device gate; never let a local
  // failure suppress a later case on only one rank.
  const bool invalid_far_field_reference_direct_rejected =
      DirectBorrowedRejectedBeforeDevice(
          invalid_far_field_reference_policy, ranks);
  const bool nonfinite_boundary_direct_rejected =
      DirectBorrowedRejectedBeforeDevice(nonfinite_boundary_policy, ranks);
  const bool noncanonical_boundary_direct_rejected =
      DirectBorrowedRejectedBeforeDevice(noncanonical_boundary_policy, ranks);
  const bool signed_zero_boundary_direct_rejected =
      DirectBorrowedRejectedBeforeDevice(signed_zero_boundary_policy, ranks);
  const bool invalid_outlet_pressure_direct_rejected =
      DirectBorrowedRejectedBeforeDevice(
          invalid_outlet_pressure_policy, ranks);
  const bool unused_pressure_direct_rejected =
      DirectBorrowedRejectedBeforeDevice(unused_pressure_policy, ranks);
  const bool signed_zero_pressure_direct_rejected =
      DirectBorrowedRejectedBeforeDevice(signed_zero_pressure_policy, ranks);
  const bool boundary_pre_device_rejected =
      invalid_far_field_reference_direct_rejected &&
      nonfinite_boundary_direct_rejected &&
      noncanonical_boundary_direct_rejected &&
      signed_zero_boundary_direct_rejected &&
      invalid_outlet_pressure_direct_rejected &&
      unused_pressure_direct_rejected && signed_zero_pressure_direct_rejected;
  // The same stage-specific rejection coverage applies to functional p2 on
  // both supported rank selections; no CUDA device may be configured first.
  const bool boundary_p2_pre_device_rejected =
      BoundaryPolicyViolationsRejectedBeforeDevice(
          boundary, ranks, functional_polynomial_order,
          smoke_elements_per_direction);
  bool rejected = nonperiodic_rejected && invalid_mesh_n_rejected &&
      unsupported_order_rejected &&
      wrong_device_rejected && wrong_ranks_rejected && invalid_config_dt_rejected &&
      capture_oracle_rejected && direct_capture_oracle_rejected &&
      injection_rejected && null_initial_rejected &&
      wrong_initial_count_rejected && wrong_initial_components_rejected &&
      null_source_rejected && nonfinite_initial_rejected &&
      nonfinite_source_rejected && nonphysical_initial_rejected &&
      unknown_boundary_rejected && invalid_far_field_reference_rejected &&
      nonfinite_boundary_rejected && noncanonical_boundary_rejected &&
      signed_zero_table_rejected && signed_zero_boundary_rejected &&
      invalid_outlet_pressure_rejected && unused_pressure_rejected &&
      signed_zero_pressure_rejected &&
      noncanonical_periodic_identity_rejected &&
      unused_periodic_policy_session_rejected &&
      unused_periodic_policy_direct_rejected &&
      boundary_pre_device_rejected &&
      boundary_p2_pre_device_rejected &&
      SameValues(fields.initial, fields_initial_before) &&
      SameValues(fields.source, fields_source_before);

  if (ranks == 2)
  {
    mfem::Mesh serial = PeriodicSquareMesh(smoke_elements_per_direction);
    std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()), 0);
    mfem::ParMesh empty_local(MPI_COMM_WORLD, serial, partition.data());
    CudaConfig empty = BorrowedConfig(
        empty_local, ranks, smoke_polynomial_order,
        smoke_elements_per_direction);
    GenericEulerHdgCudaSession sixteen;
    const bool empty_rejected = Rejected(sixteen, empty);

    CudaConfig divergent = BorrowedConfig(
        periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) divergent.dt = 0.01;
    GenericEulerHdgCudaSession seventeen;
    const bool divergent_rejected = Rejected(seventeen, divergent);

    CudaConfig mode_mismatch = BorrowedDataConfig(
        periodic, ranks, smoke_polynomial_order, smoke_elements_per_direction,
        fields);
    if (rank == 1)
    {
      mode_mismatch.use_borrowed_host_inputs = false;
      mode_mismatch.borrowed_initial = {nullptr, 0, conserved_components};
      mode_mismatch.borrowed_source = {nullptr, 0, conserved_components};
    }
    GenericEulerHdgCudaSession mode_session;
    const bool mode_mismatch_session_rejected =
        Rejected(mode_session, mode_mismatch);
    const bool mode_mismatch_direct_rejected =
        DirectBorrowedRejectedBeforeDevice(mode_mismatch, ranks);
    const bool mode_mismatch_rejected = mode_mismatch_session_rejected &&
        mode_mismatch_direct_rejected;

    // Each rank deliberately borrows a different mesh object while declaring
    // the other mesh's rank-common identity.  The identity min/max admission
    // must reject before engine allocation or any ParFESpace exchange.
    mfem::Mesh serial_a = NonperiodicSquareMesh(smoke_elements_per_direction);
    mfem::Mesh serial_b = NonperiodicSquareMesh(smoke_elements_per_direction);
    mfem::ParMesh mesh_a(MPI_COMM_WORLD, serial_a);
    mfem::ParMesh mesh_b(MPI_COMM_WORLD, serial_b);
    HostFields fields_a = NonuniformHostFields(
        mesh_a, smoke_polynomial_order, false);
    HostFields fields_b = NonuniformHostFields(
        mesh_b, smoke_polynomial_order, false);
    CudaConfig crossed_identity = rank == 0
        ? BorrowedDataConfig(mesh_a, ranks, smoke_polynomial_order,
                             smoke_elements_per_direction, fields_a)
        : BorrowedDataConfig(mesh_b, ranks, smoke_polynomial_order,
                             smoke_elements_per_direction, fields_b);
    crossed_identity.borrowed_global_topology = rank == 0
        ? StableIdentity::Copy("cuda.generic.quad.physical.mesh-b.v1")
        : StableIdentity::Copy("cuda.generic.quad.physical.mesh-a.v1");
    std::array<CudaStaticBoundaryPolicy, 4> crossed_policies =
        StaticPhysicalPolicies();
    crossed_identity.borrowed_boundary_policies = {crossed_policies.data(),
                                                   crossed_policies.size()};
    GenericEulerHdgCudaSession twenty_four;
    const bool crossed_identity_session_rejected =
        Rejected(twenty_four, crossed_identity);
    const bool crossed_identity_direct_rejected =
        DirectBorrowedRejectedBeforeDevice(crossed_identity, ranks);
    const bool crossed_identity_rejected =
        crossed_identity_session_rejected && crossed_identity_direct_rejected;
    rejected = rejected && empty_rejected && divergent_rejected &&
        mode_mismatch_rejected &&
        crossed_identity_rejected;
  }
  return All(rejected);
}
enum class TestMode
{
  smoke,
  p2,
  negatives
};

struct Options
{
  TestMode mode = TestMode::smoke;
  CudaEngineKind engine = CudaEngineKind::ordinary;
};

bool ParseOptions(const int argc, char** argv, Options& options) noexcept
{
  bool mode_selected = false;
  bool engine_selected = false;
  for (int index = 1; index < argc; ++index)
  {
    const std::string_view argument(argv[index]);
    if (argument == "--p2" || argument == "--negatives")
    {
      if (mode_selected) return false;
      options.mode = argument == "--p2" ? TestMode::p2
                                          : TestMode::negatives;
      mode_selected = true;
      continue;
    }
    if (argument == "--engine")
    {
      if (engine_selected || ++index >= argc) return false;
      const std::string_view engine(argv[index]);
      if (engine == "ordinary")
        options.engine = CudaEngineKind::ordinary;
      else if (engine == "tensor")
        options.engine = CudaEngineKind::tensor_core;
      else
        return false;
      engine_selected = true;
      continue;
    }
    return false;
  }
  return options.mode != TestMode::negatives ||
      options.engine == CudaEngineKind::ordinary;
}

}  // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  Options options;
  const bool known_mode = ParseOptions(argc, argv, options);
  selected_cuda_engine = options.engine;
  bool pass = false;
  if (known_mode && options.mode == TestMode::negatives)
  {
    const bool device_unconfigured_before = !mfem::Device::IsConfigured();
    const bool negatives = CheckNegatives(ranks);
    const bool device_unconfigured_after = !mfem::Device::IsConfigured();
    pass = known_mode && device_unconfigured_before && negatives &&
        device_unconfigured_after;
  }
  else if (known_mode)
  {
    mfem::Device device("cuda");
    (void)device;
    if (options.mode == TestMode::p2)
    {
      const bool admission = CheckP2Admission(ranks);
      const bool host_data = CheckBorrowedDataBridge(
          ranks, functional_polynomial_order,
          functional_elements_per_direction);
      const bool high_wave = CheckBorrowedLaggedLlfHighWave(
          ranks, functional_polynomial_order,
          functional_elements_per_direction);
      const bool refined_capacity = CheckRefinedDynamicCapacity(ranks);
      const bool physical_boundary = CheckStaticPhysicalBoundaryTransaction(
          ranks, functional_polynomial_order,
          functional_elements_per_direction);
      pass = admission && host_data && high_wave && refined_capacity &&
          physical_boundary;
    }
    else
    {
      const bool transaction = CheckTransaction(ranks);
      const bool foreign_authority = CheckForeignDestinationAuthority(ranks);
      const bool address_reuse = CheckAddressReuseAba(ranks);
      const bool session_identity = CheckRankTwoSessionIdentityAdmission(ranks);
      const bool rank_two = CheckRankTwoCollectiveAdmission(ranks);
      const bool legacy_facade =
          selected_cuda_engine == CudaEngineKind::ordinary
          ? CheckLegacyFacadeAdmission(ranks) : true;
      const bool host_data = CheckBorrowedDataBridge(
          ranks, smoke_polynomial_order, smoke_elements_per_direction);
      const bool high_wave = CheckBorrowedLaggedLlfHighWave(
          ranks, smoke_polynomial_order, smoke_elements_per_direction);
      const bool prepared_dt = CheckPreparedDtAdmission(ranks);
      const bool refined_capacity = CheckRefinedDynamicCapacity(ranks);
      const bool physical_boundary = CheckStaticPhysicalBoundaryTransaction(
          ranks, smoke_polynomial_order, smoke_elements_per_direction);
      pass = transaction && foreign_authority && address_reuse &&
          session_identity && rank_two && legacy_facade && host_data && high_wave &&
          prepared_dt && refined_capacity && physical_boundary;
    }
  }
  if (!pass && ranks > 0)
    std::cerr << "CUDA generic Euler session audit failed engine="
              << (selected_cuda_engine == CudaEngineKind::tensor_core
                      ? "tensor" : "ordinary") << '\n';
  return pass ? 0 : 1;
}

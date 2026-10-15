#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cpu
{
namespace
{

constexpr double rodas3_algebraic_correction_budget_fraction = 0.25;
constexpr std::string_view generic_cpu_mfem_pin =
    "d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec";
constexpr std::string_view generic_cpu_igr_layout =
    "l2-interior.true-trace-owned.component-major.igr-extension.7x5.v1";

StateCoordinates ToCoreCoordinates(
    const GenericEulerCpuSessionCoordinates& coordinates) noexcept
{
  return {coordinates.time, coordinates.step, coordinates.generation,
          coordinates.step, coordinates.dt};
}

GenericEulerCpuSessionCoordinates FromCoreCoordinates(
    const StateCoordinates& coordinates) noexcept
{
  return {coordinates.time, coordinates.step, coordinates.generation,
          coordinates.dt};
}

bool SameCoordinates(const GenericEulerCpuSessionCoordinates& left,
                     const GenericEulerCpuSessionCoordinates& right) noexcept
{
  return generic::DoubleBits(left.time) == generic::DoubleBits(right.time) &&
      left.step == right.step && left.generation == right.generation &&
      generic::DoubleBits(left.dt) == generic::DoubleBits(right.dt);
}

bool WorldCollectiveCompatible(const MPI_Comm communicator) noexcept
{
  int relation = MPI_UNEQUAL;
  return communicator != MPI_COMM_NULL &&
      MPI_Comm_compare(communicator, MPI_COMM_WORLD, &relation) ==
          MPI_SUCCESS &&
      (relation == MPI_IDENT || relation == MPI_CONGRUENT);
}

bool SameNumericalPolicy(const generic::EulerHdgNumericalPolicy& left,
                         const generic::EulerHdgNumericalPolicy& right)
    noexcept
{
  return left.volume_projection_quadrature_order ==
          right.volume_projection_quadrature_order &&
      left.true_trace_projection_quadrature_order ==
          right.true_trace_projection_quadrature_order &&
      left.conventional_assembly_quadrature_order ==
          right.conventional_assembly_quadrature_order &&
      left.igr_assembly_quadrature_orders.volume_order ==
          right.igr_assembly_quadrature_orders.volume_order &&
      left.igr_assembly_quadrature_orders.face_order ==
          right.igr_assembly_quadrature_orders.face_order &&
      left.maximum_nonlinear_updates == right.maximum_nonlinear_updates &&
      left.maximum_linear_iterations == right.maximum_linear_iterations &&
      left.maximum_line_search_trials ==
          right.maximum_line_search_trials &&
      generic::DoubleBits(left.nonlinear_relative_tolerance) ==
          generic::DoubleBits(right.nonlinear_relative_tolerance) &&
      generic::DoubleBits(
          left.newton_correction_linear_relative_tolerance) ==
          generic::DoubleBits(
              right.newton_correction_linear_relative_tolerance) &&
      generic::DoubleBits(left.final_coherent_linear_relative_tolerance) ==
          generic::DoubleBits(
              right.final_coherent_linear_relative_tolerance) &&
      generic::DoubleBits(
          left.true_residual_tightening_linear_relative_tolerance) ==
          generic::DoubleBits(
              right.true_residual_tightening_linear_relative_tolerance) &&
      generic::DoubleBits(left.true_residual_relative_tolerance) ==
          generic::DoubleBits(right.true_residual_relative_tolerance) &&
      generic::DoubleBits(left.conservation_relative_tolerance) ==
          generic::DoubleBits(right.conservation_relative_tolerance) &&
      generic::DoubleBits(left.armijo_constant) ==
          generic::DoubleBits(right.armijo_constant) &&
      left.stabilization.kind == right.stabilization.kind &&
      generic::DoubleBits(left.stabilization.alpha) ==
          generic::DoubleBits(right.stabilization.alpha) &&
      generic::DoubleBits(left.stabilization.beta) ==
          generic::DoubleBits(right.stabilization.beta) &&
      left.positivity_limiter.kind == right.positivity_limiter.kind &&
      left.positivity_limiter.point_set ==
          right.positivity_limiter.point_set &&
      generic::DoubleBits(left.positivity_limiter.density_floor) ==
          generic::DoubleBits(right.positivity_limiter.density_floor) &&
      generic::DoubleBits(left.positivity_limiter.pressure_floor) ==
          generic::DoubleBits(right.positivity_limiter.pressure_floor) &&
      left.positivity_limiter.pressure_bisection_iterations ==
          right.positivity_limiter.pressure_bisection_iterations;
}

bool SameBackend(const generic::BackendSelection& left,
                 const generic::BackendSelection& right) noexcept
{
  return left.kind == right.kind &&
      left.device_ordinal == right.device_ordinal &&
      left.cuda_engine == right.cuda_engine;
}

bool SameBoundaryPolicy(const generic::EulerBoundaryPolicy& left,
                        const generic::EulerBoundaryPolicy& right) noexcept
{
  if (left.kind != right.kind ||
      generic::DoubleBits(left.pressure_out) !=
          generic::DoubleBits(right.pressure_out))
  {
    return false;
  }
  for (std::size_t component = 0; component < left.reference_state.size();
       ++component)
  {
    if (generic::DoubleBits(left.reference_state[component]) !=
        generic::DoubleBits(right.reference_state[component]))
    {
      return false;
    }
  }
  return true;
}

RuntimeTupleDescriptor GenericCpuTuple(
    const generic::EulerHdgTimeTreatment time_treatment,
    const generic::RegularizationKind regularization) noexcept
{
  const std::string_view time =
      time_treatment == generic::EulerHdgTimeTreatment::rodas3
      ? std::string_view{"rodas3-common-gamma.v1"}
      : time_treatment ==
              generic::EulerHdgTimeTreatment::partitioned_backward_euler
      ? std::string_view{"partitioned-backward-euler.v1"}
      : std::string_view{"backward-euler.v1"};
  return {StableIdentity::Copy("euler.conservative.2d.v1"),
          StableIdentity::Copy("hdg.conventional.v1"),
          StableIdentity::Copy("trace.dg.interface.v1"),
          StableIdentity::Copy(
              regularization == generic::RegularizationKind::igr
                  ? "igr.scalar.traced.alpha-beta.v1"
                  : "tau.llf.max-interior-trace.frozen-jacobian.v1"),
          StableIdentity::Copy(time),
          StableIdentity::Copy("cpu.host.v1"),
          StableIdentity::Copy("fp64.v1")};
}

template <generic::EulerHdgTimeTreatment TimeTreatment,
          generic::RegularizationKind Regularization>
struct GenericCpuCheckpointSelection
{
  static RuntimeTupleDescriptor Descriptor() noexcept
  {
    return GenericCpuTuple(TimeTreatment, Regularization);
  }
};

template <typename Selection>
RestoreOutcome ValidateRestoreWithContract(
    CpuCheckpoint& initial_committed, CpuCheckpoint& initial_candidate,
    const CpuCheckpoint& loaded, const PersistentResourceDescriptor& resources,
    RunTelemetry& telemetry) noexcept
{
  using Contract = PersistentRunContract<
      Selection, CpuInteriorState, CpuTraceState, CpuStateStoragePolicy,
      CpuWorldCollectivePolicy>;
  Contract contract(std::move(initial_committed.state),
                    std::move(initial_candidate.state), StateCoordinates{},
                    resources);
  const RestoreOutcome outcome = contract.Restore(
      loaded.descriptor, loaded.state);
  telemetry = contract.telemetry();
  return outcome;
}

RestoreOutcome ValidateGenericCpuRestore(
    const generic::EulerHdgRequest& request,
    CpuCheckpoint& initial_committed, CpuCheckpoint& initial_candidate,
    const CpuCheckpoint& loaded, const PersistentResourceDescriptor& resources,
    RunTelemetry& telemetry) noexcept
{
  const bool igr = request.numerical_policy.stabilization.kind ==
      generic::RegularizationKind::igr;
  switch (request.time_treatment)
  {
    case generic::EulerHdgTimeTreatment::backward_euler:
      return igr
          ? ValidateRestoreWithContract<GenericCpuCheckpointSelection<
                generic::EulerHdgTimeTreatment::backward_euler,
                generic::RegularizationKind::igr>>(
                initial_committed, initial_candidate, loaded, resources,
                telemetry)
          : ValidateRestoreWithContract<GenericCpuCheckpointSelection<
                generic::EulerHdgTimeTreatment::backward_euler,
                generic::RegularizationKind::none>>(
                initial_committed, initial_candidate, loaded, resources,
                telemetry);
    case generic::EulerHdgTimeTreatment::partitioned_backward_euler:
      return igr
          ? ValidateRestoreWithContract<GenericCpuCheckpointSelection<
                generic::EulerHdgTimeTreatment::partitioned_backward_euler,
                generic::RegularizationKind::igr>>(
                initial_committed, initial_candidate, loaded, resources,
                telemetry)
          : ValidateRestoreWithContract<GenericCpuCheckpointSelection<
                generic::EulerHdgTimeTreatment::partitioned_backward_euler,
                generic::RegularizationKind::none>>(
                initial_committed, initial_candidate, loaded, resources,
                telemetry);
    case generic::EulerHdgTimeTreatment::rodas3:
      return igr
          ? ValidateRestoreWithContract<GenericCpuCheckpointSelection<
                generic::EulerHdgTimeTreatment::rodas3,
                generic::RegularizationKind::igr>>(
                initial_committed, initial_candidate, loaded, resources,
                telemetry)
          : ValidateRestoreWithContract<GenericCpuCheckpointSelection<
                generic::EulerHdgTimeTreatment::rodas3,
                generic::RegularizationKind::none>>(
                initial_committed, initial_candidate, loaded, resources,
                telemetry);
  }
  return {false, RestartCompatibility::tuple_mismatch};
}

} // namespace

struct GenericEulerCpuCheckpoint::Storage
{
  CpuCheckpoint checkpoint{};
  GenericEulerCpuSessionCoordinates coordinates{};
  generic::EulerHdgNumericalPolicy numerical_policy{};
  generic::BackendSelection backend{};
  generic::EulerHdgTimeTreatment time_treatment =
      generic::EulerHdgTimeTreatment::backward_euler;
  std::uint32_t capabilities = 0;
  int source_components = 0;
  std::vector<double> source{};
  generic::NumericalBoundaryState boundary_operator = nullptr;
  int boundary_space_dimension = 0;
  double boundary_gamma = 0.0;
  std::vector<generic::EulerBoundaryPolicy> boundary_policies{};
};

GenericEulerCpuCheckpoint::GenericEulerCpuCheckpoint() noexcept = default;
GenericEulerCpuCheckpoint::~GenericEulerCpuCheckpoint() = default;
GenericEulerCpuCheckpoint::GenericEulerCpuCheckpoint(
    GenericEulerCpuCheckpoint&&) noexcept = default;
GenericEulerCpuCheckpoint& GenericEulerCpuCheckpoint::operator=(
    GenericEulerCpuCheckpoint&&) noexcept = default;

bool GenericEulerCpuCheckpoint::valid() const noexcept
{
  return storage_ && storage_->checkpoint.valid;
}

GenericEulerCpuSessionCoordinates
GenericEulerCpuCheckpoint::coordinates() const noexcept
{
  return valid() ? storage_->coordinates
                 : GenericEulerCpuSessionCoordinates{};
}

bool GenericEulerCpuCheckpoint::SameStateForTests(
    const GenericEulerCpuCheckpoint& other) const noexcept
{
  if (!valid() || !other.valid() ||
      !::dg::hdg::runtime::detail::SameCheckpoint(
          storage_->checkpoint.descriptor,
          other.storage_->checkpoint.descriptor) ||
      !SameCoordinates(storage_->coordinates, other.storage_->coordinates))
  {
    return false;
  }
  const CpuCoefficientView left_interior = storage_->checkpoint.interior();
  const CpuCoefficientView right_interior =
      other.storage_->checkpoint.interior();
  const CpuCoefficientView left_trace = storage_->checkpoint.trace();
  const CpuCoefficientView right_trace = other.storage_->checkpoint.trace();
  return left_interior.size == right_interior.size &&
      left_trace.size == right_trace.size &&
      (left_interior.size == 0 ||
       std::equal(left_interior.data,
                  left_interior.data + left_interior.size,
                  right_interior.data)) &&
      (left_trace.size == 0 ||
       std::equal(left_trace.data, left_trace.data + left_trace.size,
                  right_trace.data));
}

GenericEulerHdgCpuSession::GenericEulerHdgCpuSession()
    : core_(std::make_unique<detail::CpuNumericsCore>())
{}

GenericEulerHdgCpuSession::~GenericEulerHdgCpuSession() = default;

GenericEulerCpuSessionCoordinates
GenericEulerHdgCpuSession::coordinates() const noexcept
{
  return coordinates_;
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Initialize(
    const generic::EulerHdgRequest& request) noexcept
{
  if (initialization_attempted_ || !core_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  initialization_attempted_ = true;
  const generic::ContractStatus admission =
      generic::ValidateEulerHdgRequest(request);
  if (admission != generic::ContractStatus::accepted ||
      request.backend.kind != generic::BackendKind::cpu ||
      request.mesh == nullptr)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  request_ = request;
  initialized_ = core_->InitializeBorrowedEuler(request_);
  if (!initialized_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  session_epoch_ = core_->incarnation();
  if (session_epoch_ == 0)
  {
    initialized_ = false;
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  double maximum_wave_speed = 0.0;
  double backend_stable_dt = 0.0;
  if (!core_->CommittedBorrowedEulerDiagnostics(maximum_wave_speed,
                                                backend_stable_dt))
  {
    initialized_ = false;
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  diagnostics_.available = true;
  diagnostics_.coordinates = {coordinates_.time, coordinates_.dt,
                              coordinates_.step, coordinates_.generation};
  diagnostics_.wave_generation = coordinates_.generation;
  diagnostics_.maximum_wave_speed = maximum_wave_speed;
  diagnostics_.backend_stable_dt = backend_stable_dt;
  return {GenericEulerCpuSessionStatus::initialized, coordinates_};
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Prepare(
    const double dt, GenericEulerCpuPreparedStep& token) noexcept
{
  if (!initialized_ || !core_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::prepare))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  const bool token_empty = !token.valid_;
  if (!core_->CollectiveAll(token_empty))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  StateCoordinates core_coordinates = ToCoreCoordinates(coordinates_);
  std::uint64_t generation = 0;
  if (!core_->PrepareBorrowedEuler(dt, core_coordinates, generation))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  token.owner_ = this;
  token.epoch_ = session_epoch_;
  token.generation_ = generation;
  token.valid_ = true;
  prepared_generation_ = generation;
  assessed_generation_ = 0;
  return {GenericEulerCpuSessionStatus::prepared, coordinates_};
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Commit(
    GenericEulerCpuPreparedStep&& token) noexcept
{
  if (!initialized_ || !core_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::commit))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  const bool assessment_required =
      request_.time_treatment == generic::EulerHdgTimeTreatment::rodas3;
  const bool local_token_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_ &&
      (!assessment_required ||
       assessed_generation_ == token.generation_);
  StateCoordinates core_coordinates = ToCoreCoordinates(coordinates_);
  double maximum_wave_speed = 0.0;
  double backend_stable_dt = 0.0;
  if (!core_->CommitBorrowedEuler(local_token_valid ? token.generation_ : 0,
                                  core_coordinates, maximum_wave_speed,
                                  backend_stable_dt))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  token.valid_ = false;
  prepared_generation_ = 0;
  assessed_generation_ = 0;
  coordinates_ = FromCoreCoordinates(core_coordinates);
  diagnostics_.coordinates = {coordinates_.time, coordinates_.dt,
                              coordinates_.step, coordinates_.generation};
  diagnostics_.wave_generation = coordinates_.generation;
  diagnostics_.maximum_wave_speed = maximum_wave_speed;
  diagnostics_.backend_stable_dt = backend_stable_dt;
  return {GenericEulerCpuSessionStatus::committed, coordinates_};
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Abort(
    GenericEulerCpuPreparedStep&& token) noexcept
{
  if (!initialized_ || !core_)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::abort))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  const bool local_token_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_;
  const StateCoordinates core_coordinates = ToCoreCoordinates(coordinates_);
  if (!core_->AbortBorrowedEuler(local_token_valid ? token.generation_ : 0,
                                 core_coordinates))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  token.valid_ = false;
  prepared_generation_ = 0;
  assessed_generation_ = 0;
  return {GenericEulerCpuSessionStatus::aborted, coordinates_};
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Advance(
    const double dt) noexcept
{
  GenericEulerCpuPreparedStep token;
  if (Prepare(dt, token).status != GenericEulerCpuSessionStatus::prepared)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }
  if (request_.time_treatment == generic::EulerHdgTimeTreatment::rodas3)
  {
    constexpr generic::HdgEulerPreparedAssessmentPolicy fixed_policy{
        1.0e-8, 1.0e-5,
        rodas3_algebraic_correction_budget_fraction};
    generic::HdgEulerPreparedAssessment assessment{};
    if (!AssessPrepared(token, fixed_policy, assessment) ||
        !assessment.algebraic_correction_gate)
    {
      static_cast<void>(Abort(std::move(token)));
      return {GenericEulerCpuSessionStatus::rejected, coordinates_};
    }
  }
  const GenericEulerCpuSessionResult committed = Commit(std::move(token));
  if (committed.status == GenericEulerCpuSessionStatus::committed)
  {
    return {GenericEulerCpuSessionStatus::advanced, committed.coordinates};
  }
  // Commit can fail only before publication.  Its retained authority is still
  // valid in that case, so clean the prepared candidate collectively rather
  // than leaving this compatibility session wedged.
  static_cast<void>(Abort(std::move(token)));
  return {GenericEulerCpuSessionStatus::rejected, coordinates_};
}

bool GenericEulerHdgCpuSession::CaptureCheckpoint(
    GenericEulerCpuCheckpoint& output) noexcept
{
  const bool local_ready = initialized_ && core_ && !output.storage_ &&
      prepared_generation_ == 0 && assessed_generation_ == 0 &&
      coordinates_.step > 0 && coordinates_.generation > 0 &&
      std::isfinite(coordinates_.time) && coordinates_.time > 0.0 &&
      std::isfinite(coordinates_.dt) && coordinates_.dt > 0.0 &&
      WorldCollectiveCompatible(request_.communicator);
  if (!core_ || !core_->CollectiveAll(local_ready)) return false;

  std::unique_ptr<GenericEulerCpuCheckpoint::Storage> next;
  bool storage_ready = false;
  try
  {
    next = std::make_unique<GenericEulerCpuCheckpoint::Storage>();
    next->coordinates = coordinates_;
    next->numerical_policy = request_.numerical_policy;
    next->backend = request_.backend;
    next->time_treatment = request_.time_treatment;
    next->capabilities = request_.capabilities;
    next->source_components = request_.source.components;
    if (request_.source.scalar_count != 0)
    {
      next->source.assign(
          request_.source.data,
          request_.source.data + request_.source.scalar_count);
    }
    next->boundary_operator = request_.boundary.exterior_state;
    next->boundary_space_dimension = request_.boundary.policies.space_dimension;
    next->boundary_gamma = request_.boundary.policies.gamma;
    if (request_.boundary.policies.count != 0)
    {
      next->boundary_policies.assign(
          request_.boundary.policies.policies,
          request_.boundary.policies.policies +
              request_.boundary.policies.count);
    }
    storage_ready = next != nullptr;
  }
  catch (...)
  {
    storage_ready = false;
  }
  if (!core_->CollectiveAll(storage_ready)) return false;

  const CpuRunEvidence& evidence = core_->evidence();
  const StateCoordinates core_coordinates = ToCoreCoordinates(coordinates_);
  const bool igr = request_.numerical_policy.stabilization.kind ==
      generic::RegularizationKind::igr;
  CheckpointDescriptor descriptor{
      checkpoint_schema_version,
      true,
      RestartMode::same_backend_rank_partition,
      evidence.tuple,
      StableIdentity::Copy(generic_cpu_mfem_pin),
      evidence.mesh_hash,
      evidence.partition_hash,
      evidence.topology_hash,
      static_cast<std::uint32_t>(evidence.ranks),
      evidence.polynomial_order,
      StableIdentity::Copy(
          igr ? generic_cpu_igr_layout
              : detail::cpu_runtime_layout_identity),
      evidence.tuple.precision,
      core_coordinates.cursor,
      core_coordinates.dt,
      core_coordinates,
      ::dg::hdg::runtime::detail::CoordinateFingerprint(core_coordinates),
      {},
      {}};
  const bool captured = core_->CaptureCheckpoint(
      descriptor, next->checkpoint);
  if (!core_->CollectiveAll(captured)) return false;

  // Capture owns an independent complete slot.  Bind its rank-common hashes
  // only after every rank has successfully allocated and copied both fields,
  // so the ordered hash collectives cannot diverge.
  next->checkpoint.descriptor.interior_hash =
      CpuStateStoragePolicy::HashInterior(
          next->checkpoint.state.interior);
  next->checkpoint.descriptor.trace_hash =
      CpuStateStoragePolicy::HashTrace(next->checkpoint.state.trace);
  if (!core_->CollectiveAll(
          next->checkpoint.descriptor.interior_hash != StateHash{} &&
          next->checkpoint.descriptor.trace_hash != StateHash{}))
  {
    return false;
  }
  output.storage_ = std::move(next);
  return true;
}

GenericEulerCpuSessionResult GenericEulerHdgCpuSession::Restore(
    const GenericEulerCpuCheckpoint& source) noexcept
{
  const bool source_ready = source.storage_ &&
      source.storage_->checkpoint.valid;
  bool configuration_matches = false;
  if (source_ready)
  {
    const auto& stored = *source.storage_;
    const bool source_field_matches =
        stored.source_components == request_.source.components &&
        stored.source.size() == request_.source.scalar_count &&
        (stored.source.empty() ||
         std::equal(stored.source.begin(), stored.source.end(),
                    request_.source.data));
    const auto& table = request_.boundary.policies;
    bool boundary_matches =
        stored.boundary_operator == request_.boundary.exterior_state &&
        stored.boundary_space_dimension == table.space_dimension &&
        generic::DoubleBits(stored.boundary_gamma) ==
            generic::DoubleBits(table.gamma) &&
        stored.boundary_policies.size() == table.count;
    for (std::size_t index = 0;
         index < stored.boundary_policies.size() && boundary_matches; ++index)
    {
      boundary_matches = table.policies != nullptr && SameBoundaryPolicy(
          stored.boundary_policies[index], table.policies[index]);
    }
    configuration_matches =
        SameNumericalPolicy(stored.numerical_policy,
                            request_.numerical_policy) &&
        SameBackend(stored.backend, request_.backend) &&
        stored.time_treatment == request_.time_treatment &&
        stored.capabilities == request_.capabilities &&
        source_field_matches && boundary_matches &&
        SameCoordinates(stored.coordinates,
                        FromCoreCoordinates(
                            stored.checkpoint.descriptor.coordinates));
  }

  const bool fresh = initialized_ && core_ && prepared_generation_ == 0 &&
      assessed_generation_ == 0 && coordinates_.time == 0.0 &&
      coordinates_.step == 0 && coordinates_.generation == 0 &&
      coordinates_.dt == 0.0;
  const bool local_ready = fresh && source_ready && configuration_matches &&
      WorldCollectiveCompatible(request_.communicator);
  if (!core_ || !core_->CollectiveAll(local_ready))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }

  const CpuCheckpoint& retained = source.storage_->checkpoint;
  const CpuCoefficientView retained_interior = retained.interior();
  const CpuCoefficientView retained_trace = retained.trace();
  CpuCheckpoint imported;
  if (!ImportCpuCheckpoint(
          retained.descriptor, retained_interior.data,
          retained_interior.size, retained_trace.data, retained_trace.size,
          imported))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }

  // The established restore contract validates an independently imported
  // payload against two disjoint slots shaped by this fresh core.  The core's
  // compact checkpoint copier is used here solely to obtain those exact live
  // rank-local extents; neither dummy descriptor becomes externally visible.
  CheckpointDescriptor initial_descriptor{};
  initial_descriptor.successful_commit = true;
  CpuCheckpoint initial_committed;
  CpuCheckpoint initial_candidate;
  CpuCheckpoint initial_backup;
  const bool initial_slots_ready = core_->CaptureCheckpoint(
          initial_descriptor, initial_committed) &&
      core_->CaptureCheckpoint(initial_descriptor, initial_candidate) &&
      core_->CaptureCheckpoint(initial_descriptor, initial_backup);
  if (!core_->CollectiveAll(initial_slots_ready) ||
      !core_->ValidateRestoredScalarClosure(imported.state))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }

  const CpuRunEvidence& evidence = core_->evidence();
  const bool igr = request_.numerical_policy.stabilization.kind ==
      generic::RegularizationKind::igr;
  PersistentResourceDescriptor resources{};
  resources.mfem_pin = StableIdentity::Copy(generic_cpu_mfem_pin);
  resources.mesh_hash = evidence.mesh_hash;
  resources.partition_hash = evidence.partition_hash;
  resources.topology_hash = evidence.topology_hash;
  resources.rank_count = static_cast<std::uint32_t>(evidence.ranks);
  resources.polynomial_order = evidence.polynomial_order;
  resources.layout = StableIdentity::Copy(
      igr ? generic_cpu_igr_layout : detail::cpu_runtime_layout_identity);
  resources.reuse = {true, true, true, false, 1, 1, 1, 1};

  RunTelemetry restore_telemetry{};
  const RestoreOutcome restored = ValidateGenericCpuRestore(
      request_, initial_committed, initial_candidate, imported, resources,
      restore_telemetry);
  if (!restored.restored ||
      restored.compatibility != RestartCompatibility::compatible)
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }

  CpuStateSnapshot live = core_->Snapshot(ToCoreCoordinates(coordinates_));
  const CpuCoefficientView loaded_interior = imported.interior();
  const CpuCoefficientView loaded_trace = imported.trace();
  const CpuCoefficientView backup_interior = initial_backup.interior();
  const CpuCoefficientView backup_trace = initial_backup.trace();
  const bool publication_ready = live.interior.size == loaded_interior.size &&
      live.trace.size == loaded_trace.size &&
      backup_interior.size == loaded_interior.size &&
      backup_trace.size == loaded_trace.size &&
      (live.interior.data || live.interior.size == 0) &&
      (live.trace.data || live.trace.size == 0);
  if (!core_->CollectiveAll(publication_ready))
  {
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }

  // All allocation, compatibility, closure, and hash checks are complete.
  // Publishing the complete state is now two nofail byte copies into the
  // fresh core's committed backing.  The candidate is refreshed from this
  // committed slot by the next Prepare operation.
  double* const live_interior = const_cast<double*>(live.interior.data);
  double* const live_trace = const_cast<double*>(live.trace.data);
  if (loaded_interior.size != 0)
  {
    std::memcpy(live_interior, loaded_interior.data,
                loaded_interior.size * sizeof(double));
  }
  if (loaded_trace.size != 0)
  {
    std::memcpy(live_trace, loaded_trace.data,
                loaded_trace.size * sizeof(double));
  }

  double maximum_wave_speed = 0.0;
  double backend_stable_dt = 0.0;
  if (!core_->CommittedBorrowedEulerDiagnostics(maximum_wave_speed,
                                                backend_stable_dt))
  {
    if (backup_interior.size != 0)
    {
      std::memcpy(live_interior, backup_interior.data,
                  backup_interior.size * sizeof(double));
    }
    if (backup_trace.size != 0)
    {
      std::memcpy(live_trace, backup_trace.data,
                  backup_trace.size * sizeof(double));
    }
    return {GenericEulerCpuSessionStatus::rejected, coordinates_};
  }

  coordinates_ = source.storage_->coordinates;
  const StateCoordinates restored_coordinates =
      ToCoreCoordinates(coordinates_);
  core_->CompleteRestore(restored, restored_coordinates, restore_telemetry);
  prepared_generation_ = 0;
  assessed_generation_ = 0;
  diagnostics_.available = true;
  diagnostics_.coordinates = {coordinates_.time, coordinates_.dt,
                              coordinates_.step, coordinates_.generation};
  diagnostics_.wave_generation = coordinates_.generation;
  diagnostics_.maximum_wave_speed = maximum_wave_speed;
  diagnostics_.backend_stable_dt = backend_stable_dt;
  return {GenericEulerCpuSessionStatus::restored, coordinates_};
}

bool GenericEulerHdgCpuSession::AssessPrepared(
    const GenericEulerCpuPreparedStep& token,
    const generic::HdgEulerPreparedAssessmentPolicy policy,
    generic::HdgEulerPreparedAssessment& assessment) noexcept
{
  assessment = {};
  assessed_generation_ = 0;
  if (!initialized_ || !core_) return false;
  // Policy bytes are caller-owned and may differ by rank.  Include local
  // policy validity in the same collective preflight as the prepared token;
  // otherwise an invalid rank could return while its peers enter the core's
  // policy-consensus collectives.
  const bool local_request_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_ && token.generation_ != 0 &&
      generic::IsValidHdgEulerPreparedAssessmentPolicy(policy);
  if (!core_->CollectiveAll(local_request_valid)) return false;
  const bool assessed = core_->AssessPreparedBorrowedEuler(
      token.generation_, policy, assessment);
  if (assessed && assessment.algebraic_correction_gate)
  {
    assessed_generation_ = token.generation_;
  }
  return assessed;
}

std::uint32_t GenericEulerHdgCpuSession::capabilities() const noexcept
{
  const std::uint32_t base = generic::HdgEulerSessionCapabilityBit(
             generic::HdgEulerSessionCapability::prepared_commit) |
      generic::HdgEulerSessionCapabilityBit(
          generic::HdgEulerSessionCapability::prepared_output) |
      generic::HdgEulerSessionCapabilityBit(
          generic::HdgEulerSessionCapability::committed_cfl_diagnostics) |
      generic::HdgEulerSessionCapabilityBit(
          generic::HdgEulerSessionCapability::generic_borrowed_inputs);
  return request_.time_treatment ==
          generic::EulerHdgTimeTreatment::rodas3
      ? base | generic::HdgEulerSessionCapabilityBit(
                   generic::HdgEulerSessionCapability::
                       prepared_embedded_assessment)
      : base;
}

generic::HdgEulerCommittedDiagnostics
GenericEulerHdgCpuSession::diagnostics() const noexcept
{
  return diagnostics_;
}

bool GenericEulerHdgCpuSession::ExportCommittedInterior(
    const GenericEulerCpuInteriorBuffer destination) const noexcept
{
  return initialized_ && core_ &&
      core_->ExportCommittedInterior(destination.data, destination.size);
}

bool GenericEulerHdgCpuSession::ExportCommittedInterior(
    mfem::ParGridFunction& destination) const noexcept
{
  return initialized_ && core_ && core_->ExportCommittedInterior(destination);
}

bool GenericEulerHdgCpuSession::ExportPreparedInterior(
    const GenericEulerCpuPreparedStep& token,
    const GenericEulerCpuInteriorBuffer destination) const noexcept
{
  if (!initialized_ || !core_) { return false; }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::export_prepared_buffer))
  {
    return false;
  }
  const bool local_token_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_;
  return core_->ExportPreparedInterior(local_token_valid ? token.generation_ : 0,
                                       destination.data, destination.size);
}

bool GenericEulerHdgCpuSession::ExportPreparedInterior(
    const GenericEulerCpuPreparedStep& token,
    mfem::ParGridFunction& destination) const noexcept
{
  if (!initialized_ || !core_) { return false; }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::export_prepared_grid_function))
  {
    return false;
  }
  const bool local_token_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_;
  return core_->ExportPreparedInterior(local_token_valid ? token.generation_ : 0,
                                       destination);
}

bool GenericEulerHdgCpuSession::ExportCommittedFields(
    const generic::HdgEulerHostInteriorFields destination) const noexcept
{
  return initialized_ && core_ && core_->ExportCommittedFields(destination);
}

bool GenericEulerHdgCpuSession::ExportPreparedFields(
    const GenericEulerCpuPreparedStep& token,
    const generic::HdgEulerHostInteriorFields destination) const noexcept
{
  if (!initialized_ || !core_) { return false; }
  if (!core_->AdmitBorrowedEulerOperation(
          detail::CpuBorrowedEulerOperation::export_prepared_fields))
  {
    return false;
  }
  const bool local_token_valid = token.valid_ && token.owner_ == this &&
      token.epoch_ == session_epoch_;
  return core_->ExportPreparedFields(
      local_token_valid ? token.generation_ : 0, destination);
}

bool ExecuteGenericEulerHdgCpu(const generic::EulerHdgRequest& request) noexcept
{
  GenericEulerHdgCpuSession session;
  return session.Initialize(request).status ==
             GenericEulerCpuSessionStatus::initialized &&
      session.Advance(cpu_backward_euler_dt).status ==
          GenericEulerCpuSessionStatus::advanced;
}

bool RestartGenericEulerHdgCpu(const generic::EulerHdgRequest&) noexcept
{
  // No checkpoint representation is accepted by this borrowed, one-shot API.
  // Leaving the caller's output untouched makes rejection failure-atomic.
  return false;
}

} // namespace dg::hdg::runtime::cpu

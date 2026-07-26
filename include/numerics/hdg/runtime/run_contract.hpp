#pragma once

/**
 * @file numerics/hdg/runtime/run_contract.hpp
 * @brief Static, storage-agnostic persistent HDG run lifecycle contract.
 *
 * Backend, formulation, trace, stabilization, time, and precision choices are
 * template parameters.  The contract owns two complete state slots and never
 * publishes its candidate slot.  Backend policies provide only static,
 * non-polymorphic state and collective operations.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

namespace dg::hdg::runtime
{

enum class InteriorSpace : unsigned char
{
  l2,
  h1
};

enum class RunStatus : unsigned char
{
  ready,
  attempt_active,
  retry_available,
  committed,
  terminal_failure,
  unsupported_h1,
  invalid_request
};

enum class FailureReason : unsigned char
{
  none,
  unsupported_h1,
  invalid_interior_space,
  invalid_dt,
  invalid_resource_descriptor,
  coordinate_overflow,
  candidate_preparation,
  stale_attempt,
  collective_identity,
  finite_gate,
  admissibility_gate,
  recovery_gate,
  true_residual_gate,
  telemetry_gate,
  success_gate,
  backend_failure,
  retry_exhausted
};

enum class DerivedProductKind : unsigned char
{
  residual,
  matrix,
  factor,
  solution
};

enum class RestartMode : unsigned char
{
  same_backend_rank_partition
};

enum class RestartCompatibility : unsigned char
{
  compatible,
  no_successful_commit,
  tuple_mismatch,
  backend_mismatch,
  mfem_pin_mismatch,
  mesh_mismatch,
  partition_mismatch,
  rank_count_mismatch,
  order_mismatch,
  layout_mismatch,
  precision_mismatch,
  policy_mismatch
};

struct StateHash
{
  std::array<std::uint8_t, 32> bytes{};
};

constexpr bool operator==(const StateHash& left,
                          const StateHash& right) noexcept
{
  for (std::size_t i = 0; i < left.bytes.size(); ++i)
  {
    if (left.bytes[i] != right.bytes[i]) { return false; }
  }
  return true;
}

constexpr bool operator!=(const StateHash& left,
                          const StateHash& right) noexcept
{
  return !(left == right);
}

struct StateCoordinates
{
  double time = 0.0;
  std::uint64_t step = 0;
  std::uint64_t generation = 0;
  std::uint64_t cursor = 0;
  double dt = 0.0;
};

constexpr bool operator==(const StateCoordinates& left,
                          const StateCoordinates& right) noexcept
{
  return left.time == right.time && left.step == right.step &&
      left.generation == right.generation && left.cursor == right.cursor &&
      left.dt == right.dt;
}

constexpr bool operator!=(const StateCoordinates& left,
                          const StateCoordinates& right) noexcept
{
  return !(left == right);
}

struct AttemptIdentity
{
  std::uint64_t sequence = 0;
  std::uint64_t committed_step = 0;
  std::uint64_t committed_generation = 0;
  unsigned char retry_index = 0;
  double dt = 0.0;
};

constexpr bool operator==(const AttemptIdentity& left,
                          const AttemptIdentity& right) noexcept
{
  return left.sequence == right.sequence &&
      left.committed_step == right.committed_step &&
      left.committed_generation == right.committed_generation &&
      left.retry_index == right.retry_index && left.dt == right.dt;
}

constexpr bool operator!=(const AttemptIdentity& left,
                          const AttemptIdentity& right) noexcept
{
  return !(left == right);
}

template <DerivedProductKind Kind, typename Payload>
struct AttemptProduct
{
  static constexpr DerivedProductKind kind = Kind;
  AttemptIdentity attempt{};
  Payload payload{};
};

struct GateReport
{
  bool finite = false;
  bool admissible = false;
  bool recovery = false;
  bool true_residual = false;
  bool telemetry = false;
  bool success = false;

  constexpr bool All() const noexcept
  {
    return finite && admissible && recovery && true_residual && telemetry &&
        success;
  }
};

struct CollectiveGateReport
{
  GateReport gates{};
  std::uint32_t participating_ranks = 0;
  bool identity_consistent = false;
};

struct TransferAccounting
{
  std::uint64_t host_to_device_bytes = 0;
  std::uint64_t device_to_host_bytes = 0;
  std::uint64_t mpi_staging_bytes = 0;
  std::uint64_t allocation_events = 0;
  std::uint64_t synchronization_events = 0;
  bool complete = false;
};

struct ProhibitedActionAccounting
{
  std::uint64_t dt_change_events = 0;
  std::uint64_t clipping_events = 0;
  std::uint64_t solver_substitution_events = 0;
  std::uint64_t fallback_events = 0;
  std::uint64_t hidden_insertion_events = 0;

  constexpr bool Clean() const noexcept
  {
    return dt_change_events == 0 && clipping_events == 0 &&
        solver_substitution_events == 0 && fallback_events == 0 &&
        hidden_insertion_events == 0;
  }
};

struct AttemptTelemetry
{
  AttemptIdentity identity{};
  TransferAccounting transfers{};
  ProhibitedActionAccounting prohibited{};
  FailureReason failure_reason = FailureReason::none;
  std::uint64_t residual_products = 0;
  std::uint64_t matrix_products = 0;
  std::uint64_t factor_products = 0;
  std::uint64_t solution_products = 0;
  double true_relative_residual =
      std::numeric_limits<double>::infinity();
};

struct AcceptedTelemetry
{
  bool valid = false;
  StateCoordinates coordinates{};
  AttemptTelemetry attempt{};
};

struct RunTelemetry
{
  std::uint64_t attempts_started = 0;
  std::uint64_t attempts_failed = 0;
  std::uint64_t rollbacks = 0;
  std::uint64_t retries_started = 0;
  std::uint64_t commits = 0;
  std::uint64_t stale_consumers_rejected = 0;
  TransferAccounting attempted_transfers{0, 0, 0, 0, 0, true};
  TransferAccounting rolled_back_transfers{0, 0, 0, 0, 0, true};
  TransferAccounting committed_transfers{0, 0, 0, 0, 0, true};
  AttemptTelemetry last_attempt{};
  AcceptedTelemetry accepted{};
};

struct ReuseDescriptor
{
  bool persistent_topology = false;
  bool persistent_complete_state_slots = false;
  bool persistent_workspace = false;
  bool per_step_allocation_required = true;
  std::uint64_t topology_generation = 0;
  std::uint64_t storage_generation = 0;
  std::size_t persistent_bytes = 0;
  std::size_t workspace_bytes = 0;

  constexpr bool SupportsPersistentRun() const noexcept
  {
    return persistent_topology && persistent_complete_state_slots &&
        persistent_workspace && !per_step_allocation_required;
  }
};

struct PersistentResourceDescriptor
{
  std::string_view mfem_pin{};
  StateHash mesh_hash{};
  StateHash partition_hash{};
  StateHash topology_hash{};
  std::uint32_t rank_count = 0;
  int polynomial_order = -1;
  std::string_view layout{};
  ReuseDescriptor reuse{};

  constexpr bool Valid() const noexcept
  {
    return !mfem_pin.empty() && rank_count != 0 && polynomial_order >= 0 &&
        !layout.empty() && reuse.SupportsPersistentRun();
  }
};

struct RuntimeTupleDescriptor
{
  std::string_view model{};
  std::string_view formulation{};
  std::string_view trace{};
  std::string_view stabilization{};
  std::string_view time_treatment{};
  std::string_view backend{};
  std::string_view precision{};
};

constexpr bool operator==(const RuntimeTupleDescriptor& left,
                          const RuntimeTupleDescriptor& right) noexcept
{
  return left.model == right.model &&
      left.formulation == right.formulation && left.trace == right.trace &&
      left.stabilization == right.stabilization &&
      left.time_treatment == right.time_treatment &&
      left.backend == right.backend && left.precision == right.precision;
}

constexpr bool operator!=(const RuntimeTupleDescriptor& left,
                          const RuntimeTupleDescriptor& right) noexcept
{
  return !(left == right);
}

struct CheckpointDescriptor
{
  bool successful_commit = false;
  RestartMode restart_mode = RestartMode::same_backend_rank_partition;
  RuntimeTupleDescriptor tuple{};
  std::string_view mfem_pin{};
  StateHash mesh_hash{};
  StateHash partition_hash{};
  std::uint32_t rank_count = 0;
  int polynomial_order = -1;
  std::string_view layout{};
  std::string_view precision{};
  std::uint64_t cursor = 0;
  double dt = 0.0;
  StateCoordinates coordinates{};
  StateHash interior_hash{};
  StateHash trace_hash{};
};

/** Threshold values remain unset until a later scenario card supplies them. */
template <typename Scalar>
struct AdmissionThreshold
{
  Scalar value{};
  bool specified = false;
};

template <typename Scalar>
struct SmoothAdmissionThresholds
{
  AdmissionThreshold<Scalar> minimum_rate{};
  AdmissionThreshold<Scalar> maximum_conservation_error{};
};

/** Typed, statically dispatched hooks; this contract assigns no thresholds. */
template <typename RateHook, typename ConservationHook>
struct SmoothAdmissionHooks
{
  static_assert(!std::is_polymorphic<RateHook>::value,
                "HDG run rate hook must be non-polymorphic");
  static_assert(!std::is_polymorphic<ConservationHook>::value,
                "HDG run conservation hook must be non-polymorphic");
  RateHook rate{};
  ConservationHook conservation{};
};

template <typename ModelTag,
          typename FormulationTag,
          typename TraceTag,
          typename StabilizationTag,
          typename TimeTag,
          typename BackendTag,
          typename PrecisionTag>
struct StaticRunSelection
{
  static_assert(!std::is_polymorphic<ModelTag>::value,
                "HDG run model selection must be static");
  static_assert(!std::is_polymorphic<FormulationTag>::value,
                "HDG run formulation selection must be static");
  static_assert(!std::is_polymorphic<TraceTag>::value,
                "HDG run trace selection must be static");
  static_assert(!std::is_polymorphic<StabilizationTag>::value,
                "HDG run stabilization selection must be static");
  static_assert(!std::is_polymorphic<TimeTag>::value,
                "HDG run time selection must be static");
  static_assert(!std::is_polymorphic<BackendTag>::value,
                "HDG run backend selection must be static");
  static_assert(!std::is_polymorphic<PrecisionTag>::value,
                "HDG run precision selection must be static");

  using model_tag = ModelTag;
  using formulation_tag = FormulationTag;
  using trace_tag = TraceTag;
  using stabilization_tag = StabilizationTag;
  using time_tag = TimeTag;
  using backend_tag = BackendTag;
  using precision_tag = PrecisionTag;

  static constexpr InteriorSpace interior_space = InteriorSpace::l2;

  static constexpr RuntimeTupleDescriptor Descriptor() noexcept
  {
    return {ModelTag::id,
            FormulationTag::id,
            TraceTag::id,
            StabilizationTag::id,
            TimeTag::id,
            BackendTag::id,
            PrecisionTag::id};
  }
};

template <typename InteriorState, typename TraceState>
struct CompleteStateSlot
{
  InteriorState interior{};
  TraceState trace{};
};

template <typename InteriorState, typename TraceState>
struct CommittedStateView
{
  const InteriorState& interior;
  const TraceState& trace;
  StateCoordinates coordinates{};
};

struct StepRequest
{
  InteriorSpace interior_space = InteriorSpace::l2;
  double dt = 0.0;
};

struct BeginOutcome
{
  bool started = false;
  AttemptIdentity attempt{};
  FailureReason reason = FailureReason::none;
};

struct AttemptReport
{
  GateReport gates{};
  AttemptTelemetry telemetry{};
  std::uint64_t candidate_cursor = 0;
  FailureReason originating_failure = FailureReason::none;
};

struct AttemptOutcome
{
  RunStatus status = RunStatus::invalid_request;
  FailureReason reason = FailureReason::none;
  bool committed = false;
  bool retry_available = false;
};

namespace detail
{

constexpr std::uint64_t SaturatingAdd(const std::uint64_t left,
                                      const std::uint64_t right) noexcept
{
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  return right > maximum - left ? maximum : left + right;
}

inline void AddTransfers(TransferAccounting& total,
                         const TransferAccounting& value) noexcept
{
  total.host_to_device_bytes =
      SaturatingAdd(total.host_to_device_bytes, value.host_to_device_bytes);
  total.device_to_host_bytes =
      SaturatingAdd(total.device_to_host_bytes, value.device_to_host_bytes);
  total.mpi_staging_bytes =
      SaturatingAdd(total.mpi_staging_bytes, value.mpi_staging_bytes);
  total.allocation_events =
      SaturatingAdd(total.allocation_events, value.allocation_events);
  total.synchronization_events = SaturatingAdd(
      total.synchronization_events, value.synchronization_events);
  total.complete = total.complete && value.complete;
}

constexpr FailureReason FirstFailedGate(const GateReport& gates) noexcept
{
  if (!gates.finite) { return FailureReason::finite_gate; }
  if (!gates.admissible) { return FailureReason::admissibility_gate; }
  if (!gates.recovery) { return FailureReason::recovery_gate; }
  if (!gates.true_residual) { return FailureReason::true_residual_gate; }
  if (!gates.telemetry) { return FailureReason::telemetry_gate; }
  if (!gates.success) { return FailureReason::success_gate; }
  return FailureReason::none;
}

} // namespace detail

/**
 * Persistent two-slot lifecycle.
 *
 * StoragePolicy must provide static PrepareCandidate, DiscardCandidate,
 * Commit, HashInterior, and HashTrace operations. CollectivePolicy must
 * provide a static Reduce operation. Commit and discard are required to be
 * noexcept so the published interior, trace, metadata, and accepted telemetry
 * transition as one contract operation.
 */
template <typename Selection,
          typename InteriorState,
          typename TraceState,
          typename StoragePolicy,
          typename CollectivePolicy>
class PersistentRunContract
{
public:
  using selection_type = Selection;
  using state_slot = CompleteStateSlot<InteriorState, TraceState>;

private:
  using prepare_result = decltype(StoragePolicy::PrepareCandidate(
      std::declval<const state_slot&>(), std::declval<state_slot&>()));
  using discard_result = decltype(
      StoragePolicy::DiscardCandidate(std::declval<state_slot&>()));
  using commit_result = decltype(StoragePolicy::Commit(
      std::declval<state_slot&>(), std::declval<state_slot&>()));
  using interior_hash_result = decltype(
      StoragePolicy::HashInterior(std::declval<const InteriorState&>()));
  using trace_hash_result = decltype(
      StoragePolicy::HashTrace(std::declval<const TraceState&>()));
  using collective_result = decltype(CollectivePolicy::Reduce(
      std::declval<const AttemptIdentity&>(), std::declval<const GateReport&>(),
      std::declval<std::uint32_t>()));

  static_assert(!std::is_polymorphic<StoragePolicy>::value,
                "HDG run storage policy must be non-polymorphic");
  static_assert(!std::is_polymorphic<CollectivePolicy>::value,
                "HDG run collective policy must be non-polymorphic");
  static_assert(std::is_same<prepare_result, bool>::value,
                "PrepareCandidate must return bool");
  static_assert(std::is_same<discard_result, void>::value,
                "DiscardCandidate must return void");
  static_assert(std::is_same<commit_result, void>::value,
                "Commit must return void");
  static_assert(std::is_same<interior_hash_result, StateHash>::value,
                "HashInterior must return StateHash");
  static_assert(std::is_same<trace_hash_result, StateHash>::value,
                "HashTrace must return StateHash");
  static_assert(std::is_same<collective_result, CollectiveGateReport>::value,
                "Collective Reduce must return CollectiveGateReport");
  static_assert(noexcept(StoragePolicy::PrepareCandidate(
                    std::declval<const state_slot&>(),
                    std::declval<state_slot&>())),
                "PrepareCandidate must be noexcept");
  static_assert(noexcept(StoragePolicy::DiscardCandidate(
                    std::declval<state_slot&>())),
                "DiscardCandidate must be noexcept");
  static_assert(noexcept(StoragePolicy::Commit(
                    std::declval<state_slot&>(), std::declval<state_slot&>())),
                "Commit must be noexcept");
  static_assert(noexcept(CollectivePolicy::Reduce(
                    std::declval<const AttemptIdentity&>(),
                    std::declval<const GateReport&>(),
                    std::declval<std::uint32_t>())),
                "Collective Reduce must be noexcept");
  static_assert(noexcept(StoragePolicy::HashInterior(
                    std::declval<const InteriorState&>())),
                "HashInterior must be noexcept");
  static_assert(noexcept(StoragePolicy::HashTrace(
                    std::declval<const TraceState&>())),
                "HashTrace must be noexcept");

public:
  PersistentRunContract(state_slot committed,
                        state_slot candidate,
                        const StateCoordinates coordinates,
                        const PersistentResourceDescriptor resources)
      : committed_(std::move(committed)),
        candidate_(std::move(candidate)),
        coordinates_(coordinates),
        resources_(resources)
  {}

  RunStatus status() const noexcept { return status_; }
  FailureReason last_failure() const noexcept { return last_failure_; }
  const StateCoordinates& coordinates() const noexcept { return coordinates_; }
  const PersistentResourceDescriptor& resources() const noexcept
  {
    return resources_;
  }
  const RunTelemetry& telemetry() const noexcept { return telemetry_; }

  CommittedStateView<InteriorState, TraceState> Old() const noexcept
  {
    return {committed_.interior, committed_.trace, coordinates_};
  }

  CommittedStateView<InteriorState, TraceState> Published() const noexcept
  {
    return Old();
  }

  const state_slot* Candidate(const AttemptIdentity& attempt) const noexcept
  {
    return IsCurrentAttempt(attempt) ? &candidate_ : nullptr;
  }

  state_slot* Candidate(const AttemptIdentity& attempt) noexcept
  {
    return IsCurrentAttempt(attempt) ? &candidate_ : nullptr;
  }

  bool IsCurrentAttempt(const AttemptIdentity& attempt) const noexcept
  {
    return active_ && attempt == active_attempt_;
  }

  template <DerivedProductKind Kind, typename Payload>
  bool TryTagProduct(const AttemptIdentity& attempt,
                     Payload payload,
                     AttemptProduct<Kind, Payload>& product) const
  {
    if (!IsCurrentAttempt(attempt)) { return false; }
    product = {attempt, std::move(payload)};
    return true;
  }

  template <DerivedProductKind Kind, typename Payload>
  bool Accepts(const AttemptProduct<Kind, Payload>& product) noexcept
  {
    if (IsCurrentAttempt(product.attempt)) { return true; }
    telemetry_.stale_consumers_rejected = detail::SaturatingAdd(
        telemetry_.stale_consumers_rejected, 1);
    return false;
  }

  BeginOutcome BeginStep(const StepRequest& request) noexcept
  {
    if (status_ != RunStatus::ready && status_ != RunStatus::committed)
    {
      return {false, {}, FailureReason::stale_attempt};
    }
    if (request.interior_space == InteriorSpace::h1)
    {
      status_ = RunStatus::unsupported_h1;
      last_failure_ = FailureReason::unsupported_h1;
      return {false, {}, last_failure_};
    }
    if (request.interior_space != InteriorSpace::l2)
    {
      status_ = RunStatus::invalid_request;
      last_failure_ = FailureReason::invalid_interior_space;
      return {false, {}, last_failure_};
    }
    if (!resources_.Valid())
    {
      status_ = RunStatus::invalid_request;
      last_failure_ = FailureReason::invalid_resource_descriptor;
      return {false, {}, last_failure_};
    }
    if (!std::isfinite(request.dt) || request.dt <= 0.0 ||
        !std::isfinite(coordinates_.time))
    {
      status_ = RunStatus::invalid_request;
      last_failure_ = FailureReason::invalid_dt;
      return {false, {}, last_failure_};
    }
    const double target_time = coordinates_.time + request.dt;
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (!std::isfinite(target_time) || coordinates_.step == maximum ||
        coordinates_.generation == maximum || attempt_sequence_ == maximum)
    {
      status_ = RunStatus::invalid_request;
      last_failure_ = FailureReason::coordinate_overflow;
      return {false, {}, last_failure_};
    }
    if (!StoragePolicy::PrepareCandidate(committed_, candidate_))
    {
      StoragePolicy::DiscardCandidate(candidate_);
      status_ = RunStatus::terminal_failure;
      last_failure_ = FailureReason::candidate_preparation;
      return {false, {}, last_failure_};
    }

    pending_dt_ = request.dt;
    pending_target_time_ = target_time;
    retry_used_ = false;
    active_attempt_ = {++attempt_sequence_,
                       coordinates_.step,
                       coordinates_.generation,
                       0,
                       pending_dt_};
    active_ = true;
    status_ = RunStatus::attempt_active;
    last_failure_ = FailureReason::none;
    telemetry_.attempts_started =
        detail::SaturatingAdd(telemetry_.attempts_started, 1);
    return {true, active_attempt_, FailureReason::none};
  }

  BeginOutcome Retry() noexcept
  {
    if (status_ != RunStatus::retry_available || retry_used_ || active_)
    {
      return {false, {}, FailureReason::retry_exhausted};
    }
    if (attempt_sequence_ == std::numeric_limits<std::uint64_t>::max())
    {
      status_ = RunStatus::terminal_failure;
      last_failure_ = FailureReason::coordinate_overflow;
      return {false, {}, last_failure_};
    }
    if (!StoragePolicy::PrepareCandidate(committed_, candidate_))
    {
      StoragePolicy::DiscardCandidate(candidate_);
      status_ = RunStatus::terminal_failure;
      last_failure_ = FailureReason::candidate_preparation;
      return {false, {}, last_failure_};
    }

    retry_used_ = true;
    active_attempt_ = {++attempt_sequence_,
                       coordinates_.step,
                       coordinates_.generation,
                       1,
                       pending_dt_};
    active_ = true;
    status_ = RunStatus::attempt_active;
    last_failure_ = FailureReason::none;
    telemetry_.attempts_started =
        detail::SaturatingAdd(telemetry_.attempts_started, 1);
    telemetry_.retries_started =
        detail::SaturatingAdd(telemetry_.retries_started, 1);
    return {true, active_attempt_, FailureReason::none};
  }

  AttemptOutcome FinishAttempt(const AttemptIdentity& attempt,
                               const AttemptReport& report) noexcept
  {
    if (!IsCurrentAttempt(attempt))
    {
      telemetry_.stale_consumers_rejected = detail::SaturatingAdd(
          telemetry_.stale_consumers_rejected, 1);
      return {status_, FailureReason::stale_attempt, false, false};
    }

    GateReport effective_gates = report.gates;
    if (!report.telemetry.transfers.complete ||
        !report.telemetry.prohibited.Clean() ||
        (resources_.reuse.per_step_allocation_required == false &&
         report.telemetry.transfers.allocation_events != 0))
    {
      effective_gates.telemetry = false;
    }
    const CollectiveGateReport collective = CollectivePolicy::Reduce(
        attempt, effective_gates, resources_.rank_count);
    const GateReport collective_gates{
        effective_gates.finite && collective.gates.finite,
        effective_gates.admissible && collective.gates.admissible,
        effective_gates.recovery && collective.gates.recovery,
        effective_gates.true_residual && collective.gates.true_residual,
        effective_gates.telemetry && collective.gates.telemetry,
        effective_gates.success && collective.gates.success};
    const bool collective_identity = collective.identity_consistent &&
        collective.participating_ranks == resources_.rank_count;

    FailureReason reason = FailureReason::none;
    if (!collective_identity)
    {
      reason = FailureReason::collective_identity;
    }
    else
    {
      reason = detail::FirstFailedGate(collective_gates);
      if (reason == FailureReason::success_gate &&
          report.originating_failure != FailureReason::none)
      {
        reason = report.originating_failure;
      }
    }

    AttemptTelemetry attempt_telemetry = report.telemetry;
    attempt_telemetry.identity = attempt;
    attempt_telemetry.failure_reason = reason;
    telemetry_.last_attempt = attempt_telemetry;
    detail::AddTransfers(telemetry_.attempted_transfers,
                         report.telemetry.transfers);

    if (!collective_identity || !collective_gates.All())
    {
      detail::AddTransfers(telemetry_.rolled_back_transfers,
                           report.telemetry.transfers);
      telemetry_.attempts_failed =
          detail::SaturatingAdd(telemetry_.attempts_failed, 1);
      telemetry_.rollbacks = detail::SaturatingAdd(telemetry_.rollbacks, 1);
      StoragePolicy::DiscardCandidate(candidate_);
      active_ = false;
      last_failure_ = reason == FailureReason::none
          ? FailureReason::backend_failure
          : reason;
      if (attempt.retry_index == 0)
      {
        status_ = RunStatus::retry_available;
        return {status_, last_failure_, false, true};
      }
      status_ = RunStatus::terminal_failure;
      return {status_, last_failure_, false, false};
    }

    StoragePolicy::Commit(committed_, candidate_);
    coordinates_.time = pending_target_time_;
    ++coordinates_.step;
    ++coordinates_.generation;
    coordinates_.cursor = report.candidate_cursor;
    coordinates_.dt = pending_dt_;
    attempt_telemetry.failure_reason = FailureReason::none;
    telemetry_.last_attempt = attempt_telemetry;
    telemetry_.accepted = {true, coordinates_, attempt_telemetry};
    detail::AddTransfers(telemetry_.committed_transfers,
                         report.telemetry.transfers);
    telemetry_.commits = detail::SaturatingAdd(telemetry_.commits, 1);
    active_ = false;
    status_ = RunStatus::committed;
    last_failure_ = FailureReason::none;
    return {status_, FailureReason::none, true, false};
  }

  bool TryMakeCheckpoint(CheckpointDescriptor& checkpoint) const noexcept
  {
    if (active_ || !telemetry_.accepted.valid ||
        telemetry_.accepted.coordinates != coordinates_)
    {
      return false;
    }
    const RuntimeTupleDescriptor tuple = Selection::Descriptor();
    checkpoint = {true,
                  RestartMode::same_backend_rank_partition,
                  tuple,
                  resources_.mfem_pin,
                  resources_.mesh_hash,
                  resources_.partition_hash,
                  resources_.rank_count,
                  resources_.polynomial_order,
                  resources_.layout,
                  tuple.precision,
                  coordinates_.cursor,
                  coordinates_.dt,
                  coordinates_,
                  StoragePolicy::HashInterior(committed_.interior),
                  StoragePolicy::HashTrace(committed_.trace)};
    return true;
  }

  RestartCompatibility CheckRestart(
      const CheckpointDescriptor& checkpoint) const noexcept
  {
    if (!checkpoint.successful_commit)
    {
      return RestartCompatibility::no_successful_commit;
    }
    if (checkpoint.restart_mode != RestartMode::same_backend_rank_partition)
    {
      return RestartCompatibility::policy_mismatch;
    }
    const RuntimeTupleDescriptor tuple = Selection::Descriptor();
    if (checkpoint.tuple.backend != tuple.backend)
    {
      return RestartCompatibility::backend_mismatch;
    }
    if (checkpoint.tuple != tuple)
    {
      return RestartCompatibility::tuple_mismatch;
    }
    if (checkpoint.mfem_pin != resources_.mfem_pin)
    {
      return RestartCompatibility::mfem_pin_mismatch;
    }
    if (checkpoint.rank_count != resources_.rank_count)
    {
      return RestartCompatibility::rank_count_mismatch;
    }
    if (checkpoint.partition_hash != resources_.partition_hash)
    {
      return RestartCompatibility::partition_mismatch;
    }
    if (checkpoint.mesh_hash != resources_.mesh_hash)
    {
      return RestartCompatibility::mesh_mismatch;
    }
    if (checkpoint.polynomial_order != resources_.polynomial_order)
    {
      return RestartCompatibility::order_mismatch;
    }
    if (checkpoint.layout != resources_.layout)
    {
      return RestartCompatibility::layout_mismatch;
    }
    if (checkpoint.precision != tuple.precision)
    {
      return RestartCompatibility::precision_mismatch;
    }
    return RestartCompatibility::compatible;
  }

private:
  state_slot committed_{};
  state_slot candidate_{};
  StateCoordinates coordinates_{};
  PersistentResourceDescriptor resources_{};
  RunTelemetry telemetry_{};
  RunStatus status_ = RunStatus::ready;
  FailureReason last_failure_ = FailureReason::none;
  AttemptIdentity active_attempt_{};
  std::uint64_t attempt_sequence_ = 0;
  double pending_dt_ = 0.0;
  double pending_target_time_ = 0.0;
  bool active_ = false;
  bool retry_used_ = false;
};

} // namespace dg::hdg::runtime

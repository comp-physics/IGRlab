#pragma once

/**
 * @file numerics/hdg/runtime/run_contract.hpp
 * @brief Static, storage-agnostic persistent HDG run lifecycle contract.
 *
 * Backend, formulation, trace, stabilization, time, and precision choices are
 * compile-time tags. The contract owns two disjoint complete state slots,
 * publishes only committed state, and makes every candidate capability and
 * derived-product token attempt- and contract-incarnation-bound. Backend
 * policies provide only static, non-polymorphic storage and collective
 * operations.
 *
 * Adapter proof obligations intentionally outside this storage-agnostic slice:
 * (1) the selected backend tag must name the concrete storage/collective
 * policies used to instantiate the contract; and (2) issued product tokens
 * must be coupled by that adapter to the actual residual, matrix, factor, and
 * solution payloads in the required numerical order. Tokens prove contract
 * issuance, order, ownership, and consumption, not numerical payload content;
 * and (3) StoragePolicy::Publish is a trusted adapter proof obligation whose
 * semantic atomicity/deep-copy behavior cannot be proved by the C++ type
 * system. This contract enforces its void-noexcept type, orders every
 * detectable precondition before it, and performs no recoverable check after
 * it; and (4) the optional two-phase API exists only for policies that exactly
 * attest a reversible complete-slot exchange and provide one unambiguous
 * void-noexcept RevertPublish inverse.
 */

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
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
  incarnation_exhausted,
  coordinate_overflow,
  candidate_preparation,
  shared_backing_store,
  publication_precondition,
  stale_attempt,
  collective_protocol,
  restart_incompatible,
  missing_product,
  invalid_product,
  stale_product,
  finite_gate,
  admissibility_gate,
  recovery_gate,
  true_residual_gate,
  telemetry_gate,
  success_gate,
  backend_failure,
  retry_exhausted,
  count
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
  contract_busy,
  no_successful_commit,
  schema_mismatch,
  time_mismatch,
  coordinate_mismatch,
  cursor_mismatch,
  dt_mismatch,
  tuple_mismatch,
  backend_mismatch,
  mfem_pin_mismatch,
  mesh_mismatch,
  partition_mismatch,
  topology_mismatch,
  rank_count_mismatch,
  order_mismatch,
  layout_mismatch,
  precision_mismatch,
  interior_hash_mismatch,
  trace_hash_mismatch,
  candidate_preparation,
  shared_backing_store,
  publication_precondition,
  collective_protocol,
  policy_mismatch,
  count
};

/** Fixed-capacity owned identity. No descriptor retains caller text storage. */
template <std::size_t Capacity = 96>
class StableId
{
public:
  constexpr StableId() noexcept = default;

  template <std::size_t Size>
  constexpr StableId(const char (&text)[Size]) noexcept
  {
    Assign(std::string_view{text, Size - 1});
  }

  static constexpr StableId Copy(const std::string_view text) noexcept
  {
    StableId result;
    result.Assign(text);
    return result;
  }

  constexpr bool valid() const noexcept { return valid_; }
  constexpr bool empty() const noexcept { return length_ == 0; }
  constexpr std::size_t size() const noexcept { return length_; }
  constexpr const char* data() const noexcept { return bytes_.data(); }
  constexpr std::string_view view() const noexcept
  {
    return {bytes_.data(), length_};
  }

private:
  constexpr void Assign(const std::string_view text) noexcept
  {
    if (text.size() > Capacity)
    {
      valid_ = false;
      length_ = 0;
      return;
    }
    length_ = text.size();
    for (std::size_t i = 0; i < length_; ++i) { bytes_[i] = text[i]; }
    valid_ = true;
  }

  std::array<char, Capacity> bytes_{};
  std::size_t length_ = 0;
  bool valid_ = true;

  template <std::size_t LeftCapacity, std::size_t RightCapacity>
  friend constexpr bool operator==(const StableId<LeftCapacity>&,
                                   const StableId<RightCapacity>&) noexcept;
};

template <std::size_t LeftCapacity, std::size_t RightCapacity>
constexpr bool operator==(const StableId<LeftCapacity>& left,
                          const StableId<RightCapacity>& right) noexcept
{
  if (!left.valid_ || !right.valid_ || left.length_ != right.length_)
  {
    return false;
  }
  for (std::size_t i = 0; i < left.length_; ++i)
  {
    if (left.bytes_[i] != right.bytes_[i]) { return false; }
  }
  return true;
}

template <std::size_t LeftCapacity, std::size_t RightCapacity>
constexpr bool operator!=(const StableId<LeftCapacity>& left,
                          const StableId<RightCapacity>& right) noexcept
{
  return !(left == right);
}

using StableIdentity = StableId<96>;

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

struct AttemptIdentity
{
  std::uint64_t sequence = 0;
  std::uint64_t committed_step = 0;
  std::uint64_t committed_generation = 0;
  unsigned char retry_index = 0;
  double dt = 0.0;
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

struct AttemptCollectiveContext
{
  AttemptIdentity attempt{};
  StateCoordinates committed{};
  StateCoordinates target{};
  std::uint64_t candidate_cursor = 0;
};

struct CollectiveAdmissionReport
{
  AttemptCollectiveContext context{};
  std::uint32_t participating_ranks = 0;
  bool identity_consistent = false;
  bool failure_consistent = false;
  bool all_prepared = false;
  bool any_failure = false;
  std::uint64_t failure_mask = 0;
  FailureReason resolved_reason = FailureReason::none;
};

struct CollectiveGateReport
{
  AttemptCollectiveContext context{};
  GateReport gates{};
  std::uint32_t participating_ranks = 0;
  bool identity_consistent = false;
  bool failure_consistent = false;
  bool any_failure = false;
  std::uint64_t failure_mask = 0;
  FailureReason resolved_reason = FailureReason::none;
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
  std::uint8_t validated_product_mask = 0;
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
  // Summary event counters deliberately saturate at uint64_t max. Transfer
  // totals do not saturate: capacity is preflighted and overflow is rejected
  // without mutating any transfer total.
  std::uint64_t admissions_started = 0;
  std::uint64_t admissions_failed = 0;
  std::uint64_t attempts_started = 0;
  std::uint64_t attempts_failed = 0;
  std::uint64_t rollbacks = 0;
  std::uint64_t retries_started = 0;
  std::uint64_t commits = 0;
  std::uint64_t restores = 0;
  std::uint64_t stale_consumers_rejected = 0;
  std::uint64_t duplicate_product_issues_rejected = 0;
  std::uint64_t out_of_order_product_issues_rejected = 0;
  std::uint64_t transfer_accounting_overflow_rejections = 0;
  TransferAccounting attempted_transfers{0, 0, 0, 0, 0, true};
  TransferAccounting rolled_back_transfers{0, 0, 0, 0, 0, true};
  TransferAccounting committed_transfers{0, 0, 0, 0, 0, true};
  AttemptTelemetry last_attempt{};
  AcceptedTelemetry accepted{};
  FailureReason first_failure = FailureReason::none;
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
  StableIdentity mfem_pin{};
  StateHash mesh_hash{};
  StateHash partition_hash{};
  StateHash topology_hash{};
  std::uint32_t rank_count = 0;
  int polynomial_order = -1;
  StableIdentity layout{};
  ReuseDescriptor reuse{};

  constexpr bool Valid() const noexcept
  {
    return mfem_pin.valid() && !mfem_pin.empty() && rank_count != 0 &&
        polynomial_order >= 0 && layout.valid() && !layout.empty() &&
        reuse.SupportsPersistentRun();
  }
};

struct RuntimeTupleDescriptor
{
  StableIdentity model{};
  StableIdentity formulation{};
  StableIdentity trace{};
  StableIdentity stabilization{};
  StableIdentity time_treatment{};
  StableIdentity backend{};
  StableIdentity precision{};

  constexpr bool Valid() const noexcept
  {
    return model.valid() && !model.empty() && formulation.valid() &&
        !formulation.empty() && trace.valid() && !trace.empty() &&
        stabilization.valid() && !stabilization.empty() &&
        time_treatment.valid() && !time_treatment.empty() && backend.valid() &&
        !backend.empty() && precision.valid() && !precision.empty();
  }
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

inline constexpr std::uint32_t checkpoint_schema_version = 1;

struct CheckpointDescriptor
{
  std::uint32_t schema_version = 0;
  bool successful_commit = false;
  RestartMode restart_mode = RestartMode::same_backend_rank_partition;
  RuntimeTupleDescriptor tuple{};
  StableIdentity mfem_pin{};
  StateHash mesh_hash{};
  StateHash partition_hash{};
  StateHash topology_hash{};
  std::uint32_t rank_count = 0;
  int polynomial_order = -1;
  StableIdentity layout{};
  StableIdentity precision{};
  std::uint64_t cursor = 0;
  double dt = 0.0;
  StateCoordinates coordinates{};
  std::uint64_t coordinate_fingerprint = 0;
  StateHash interior_hash{};
  StateHash trace_hash{};
};

struct RestoreOutcome
{
  bool restored = false;
  RestartCompatibility compatibility = RestartCompatibility::policy_mismatch;
};

// Telemetry boundary: an exact RestartCompatibility remains observable in
// RestoreOutcome. Any non-compatible result records the generic write-once
// FailureReason::restart_incompatible, except collective contradictions, which
// record FailureReason::collective_protocol. A later successful restore does
// not erase that first-failure record.

struct RestoreCollectiveContext
{
  CheckpointDescriptor checkpoint{};
  StateHash loaded_interior_hash{};
  StateHash loaded_trace_hash{};
  StateHash prepared_interior_hash{};
  StateHash prepared_trace_hash{};
  bool preparation_attempted = false;
  bool candidate_prepared = false;
};

struct CollectiveRestoreReport
{
  RestoreCollectiveContext context{};
  std::uint32_t participating_ranks = 0;
  bool identity_consistent = false;
  bool failure_consistent = false;
  bool any_failure = false;
  std::uint64_t failure_mask = 0;
  std::uint64_t compatibility_mask = 0;
  FailureReason resolved_reason = FailureReason::none;
  RestartCompatibility resolved_compatibility =
      RestartCompatibility::compatible;
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
    return {StableIdentity::Copy(ModelTag::id),
            StableIdentity::Copy(FormulationTag::id),
            StableIdentity::Copy(TraceTag::id),
            StableIdentity::Copy(StabilizationTag::id),
            StableIdentity::Copy(TimeTag::id),
            StableIdentity::Copy(BackendTag::id),
            StableIdentity::Copy(PrecisionTag::id)};
  }
};

template <typename InteriorState, typename TraceState>
struct CompleteStateSlot
{
  InteriorState interior{};
  TraceState trace{};
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

static_assert(static_cast<unsigned int>(FailureReason::none) == 0,
              "failure truth masks reserve zero for no failure");
static_assert(static_cast<unsigned int>(FailureReason::count) < 64,
              "failure truth mask must represent every FailureReason");
static_assert(static_cast<unsigned int>(RestartCompatibility::compatible) == 0,
              "restore truth masks reserve zero for compatibility");
static_assert(static_cast<unsigned int>(RestartCompatibility::count) < 64,
              "restore truth mask must represent every incompatibility");

inline std::atomic<std::uint64_t> next_contract_incarnation{1};

inline std::uint64_t AcquireContractIncarnation() noexcept
{
  std::uint64_t current = next_contract_incarnation.load(
      std::memory_order_relaxed);
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  while (current != maximum)
  {
    if (next_contract_incarnation.compare_exchange_weak(
            current, current + 1, std::memory_order_relaxed,
            std::memory_order_relaxed))
    {
      return current;
    }
  }
  return 0;
}

inline std::uint64_t DoubleBits(const double value) noexcept
{
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline bool SameCoordinates(const StateCoordinates& left,
                            const StateCoordinates& right) noexcept
{
  return DoubleBits(left.time) == DoubleBits(right.time) &&
      left.step == right.step && left.generation == right.generation &&
      left.cursor == right.cursor &&
      DoubleBits(left.dt) == DoubleBits(right.dt);
}

inline bool SameAttempt(const AttemptIdentity& left,
                        const AttemptIdentity& right) noexcept
{
  return left.sequence == right.sequence &&
      left.committed_step == right.committed_step &&
      left.committed_generation == right.committed_generation &&
      left.retry_index == right.retry_index &&
      DoubleBits(left.dt) == DoubleBits(right.dt);
}

inline bool SameContext(const AttemptCollectiveContext& left,
                        const AttemptCollectiveContext& right) noexcept
{
  return SameAttempt(left.attempt, right.attempt) &&
      SameCoordinates(left.committed, right.committed) &&
      SameCoordinates(left.target, right.target) &&
      left.candidate_cursor == right.candidate_cursor;
}

constexpr bool IsNamedFailureReason(const FailureReason reason) noexcept
{
  return static_cast<unsigned int>(reason) <
      static_cast<unsigned int>(FailureReason::count);
}

constexpr bool IsNamedRestartCompatibility(
    const RestartCompatibility compatibility) noexcept
{
  return static_cast<unsigned int>(compatibility) <
      static_cast<unsigned int>(RestartCompatibility::count);
}

constexpr std::uint64_t FailureBit(const FailureReason reason) noexcept
{
  const unsigned int value = static_cast<unsigned int>(reason);
  return IsNamedFailureReason(reason) && value != 0 && value < 64
      ? (std::uint64_t{1} << value)
      : 0;
}

constexpr std::uint64_t CompatibilityBit(
    const RestartCompatibility compatibility) noexcept
{
  const unsigned int value = static_cast<unsigned int>(compatibility);
  return IsNamedRestartCompatibility(compatibility) && value != 0 &&
          value < 64
      ? (std::uint64_t{1} << value)
      : 0;
}

constexpr std::uint64_t NamedFailureMask() noexcept
{
  std::uint64_t mask = 0;
  for (unsigned int value = 1;
       value < static_cast<unsigned int>(FailureReason::count) && value < 64;
       ++value)
  {
    mask |= std::uint64_t{1} << value;
  }
  return mask;
}

constexpr std::uint64_t NamedCompatibilityMask() noexcept
{
  std::uint64_t mask = 0;
  for (unsigned int value = 1;
       value < static_cast<unsigned int>(RestartCompatibility::count) &&
           value < 64;
       ++value)
  {
    mask |= std::uint64_t{1} << value;
  }
  return mask;
}

constexpr bool HasOnlyNamedFailureBits(const std::uint64_t mask) noexcept
{
  return (mask & ~NamedFailureMask()) == 0;
}

constexpr bool HasOnlyNamedCompatibilityBits(
    const std::uint64_t mask) noexcept
{
  return (mask & ~NamedCompatibilityMask()) == 0;
}

inline bool SameCheckpoint(const CheckpointDescriptor& left,
                           const CheckpointDescriptor& right) noexcept
{
  return left.schema_version == right.schema_version &&
      left.successful_commit == right.successful_commit &&
      left.restart_mode == right.restart_mode && left.tuple == right.tuple &&
      left.mfem_pin == right.mfem_pin &&
      left.mesh_hash == right.mesh_hash &&
      left.partition_hash == right.partition_hash &&
      left.topology_hash == right.topology_hash &&
      left.rank_count == right.rank_count &&
      left.polynomial_order == right.polynomial_order &&
      left.layout == right.layout && left.precision == right.precision &&
      left.cursor == right.cursor &&
      DoubleBits(left.dt) == DoubleBits(right.dt) &&
      SameCoordinates(left.coordinates, right.coordinates) &&
      left.coordinate_fingerprint == right.coordinate_fingerprint &&
      left.interior_hash == right.interior_hash &&
      left.trace_hash == right.trace_hash;
}

inline bool SameRestoreContext(const RestoreCollectiveContext& left,
                               const RestoreCollectiveContext& right) noexcept
{
  return SameCheckpoint(left.checkpoint, right.checkpoint) &&
      left.loaded_interior_hash == right.loaded_interior_hash &&
      left.loaded_trace_hash == right.loaded_trace_hash &&
      left.prepared_interior_hash == right.prepared_interior_hash &&
      left.prepared_trace_hash == right.prepared_trace_hash &&
      left.preparation_attempted == right.preparation_attempted &&
      left.candidate_prepared == right.candidate_prepared;
}

constexpr bool WouldOverflow(const std::uint64_t left,
                             const std::uint64_t right) noexcept
{
  return right > std::numeric_limits<std::uint64_t>::max() - left;
}

constexpr std::uint64_t SaturatingAdd(const std::uint64_t left,
                                      const std::uint64_t right) noexcept
{
  return WouldOverflow(left, right)
      ? std::numeric_limits<std::uint64_t>::max()
      : left + right;
}

inline bool CanAddTransfersExactly(const TransferAccounting& total,
                                   const TransferAccounting& value) noexcept
{
  return !WouldOverflow(total.host_to_device_bytes,
                        value.host_to_device_bytes) &&
      !WouldOverflow(total.device_to_host_bytes,
                     value.device_to_host_bytes) &&
      !WouldOverflow(total.mpi_staging_bytes, value.mpi_staging_bytes) &&
      !WouldOverflow(total.allocation_events, value.allocation_events) &&
      !WouldOverflow(total.synchronization_events,
                     value.synchronization_events);
}

/** Caller proves CanAddTransfersExactly before this non-saturating mutation. */
inline void AddTransfersExact(TransferAccounting& total,
                              const TransferAccounting& value) noexcept
{
  total.host_to_device_bytes += value.host_to_device_bytes;
  total.device_to_host_bytes += value.device_to_host_bytes;
  total.mpi_staging_bytes += value.mpi_staging_bytes;
  total.allocation_events += value.allocation_events;
  total.synchronization_events += value.synchronization_events;
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

inline std::uint64_t CoordinateFingerprint(
    const StateCoordinates& coordinates) noexcept
{
  const std::array<std::uint64_t, 5> fields{{DoubleBits(coordinates.time),
                                             coordinates.step,
                                             coordinates.generation,
                                             coordinates.cursor,
                                             DoubleBits(coordinates.dt)}};
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::uint64_t field : fields)
  {
    for (unsigned int shift = 0; shift < 64; shift += 8)
    {
      hash ^= static_cast<std::uint8_t>(field >> shift);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

constexpr std::size_t ProductIndex(const DerivedProductKind kind) noexcept
{
  switch (kind)
  {
    case DerivedProductKind::residual: return 0;
    case DerivedProductKind::matrix: return 1;
    case DerivedProductKind::factor: return 2;
    case DerivedProductKind::solution: return 3;
  }
  return 4;
}

inline bool GateReductionContainsLocal(const GateReport& local,
                                       const GateReport& reduced) noexcept
{
  return !(!local.finite && reduced.finite) &&
      !(!local.admissible && reduced.admissible) &&
      !(!local.recovery && reduced.recovery) &&
      !(!local.true_residual && reduced.true_residual) &&
      !(!local.telemetry && reduced.telemetry) &&
      !(!local.success && reduced.success);
}

/**
 * Exact opt-in for the reversible slot-exchange protocol.  Keeping this a
 * detection trait (rather than class-level static assertions) is deliberate:
 * legacy policies and near-miss policies retain the complete one-phase API.
 */
template <typename Policy, typename Slot, typename = void>
struct SupportsTwoPhaseSlotExchange : std::false_type
{};

template <typename Policy, typename Slot>
struct SupportsTwoPhaseSlotExchange<
    Policy,
    Slot,
    std::void_t<
        decltype(Policy::publish_is_complete_slot_exchange),
        decltype(Policy::candidate_retains_exact_old_state_after_publish),
        decltype(Policy::revert_publish_is_exact_byte_inverse),
        decltype(Policy::publish_and_revert_preserve_disjoint_backing),
        decltype(&Policy::RevertPublish)>>
    : std::integral_constant<
          bool,
          std::is_same<
              decltype(Policy::publish_is_complete_slot_exchange),
              const bool>::value &&
              Policy::publish_is_complete_slot_exchange &&
              std::is_same<
                  decltype(
                      Policy::candidate_retains_exact_old_state_after_publish),
                  const bool>::value &&
              Policy::candidate_retains_exact_old_state_after_publish &&
              std::is_same<
                  decltype(Policy::revert_publish_is_exact_byte_inverse),
                  const bool>::value &&
              Policy::revert_publish_is_exact_byte_inverse &&
              std::is_same<
                  decltype(
                      Policy::publish_and_revert_preserve_disjoint_backing),
                  const bool>::value &&
              Policy::publish_and_revert_preserve_disjoint_backing &&
              std::is_same<decltype(&Policy::RevertPublish),
                           void (*)(Slot&, Slot&) noexcept>::value>
{};

} // namespace detail

/**
 * Persistent two-slot lifecycle.
 *
 * StoragePolicy supplies static, noexcept preparation, private-copy,
 * disjointness, hash, publication-preflight, and publication operations.
 * Publish is the sole forward mutation of committed storage; an exactly
 * detected two-phase policy may additionally supply its nofail RevertPublish
 * inverse. Their exchange, inverse, byte-retention, and disjointness semantics
 * are explicit trusted adapter attestations. The contract mechanically
 * enforces the exact function types, orders every detectable fallible check
 * before exchange, and performs no recoverable check after it.
 * CollectivePolicy supplies static admission, finish, and restore reductions.
 * Neither policy uses dynamic dispatch.
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
  static constexpr double true_relative_residual_limit = 1.0e-10;

private:
  enum class AttemptPhase : unsigned char
  {
    idle,
    mutable_candidate,
    frozen,
    prepared,
    provisional
  };

  using prepare_result = decltype(StoragePolicy::PrepareCandidate(
      std::declval<const state_slot&>(), std::declval<state_slot&>()));
  using restore_prepare_result = decltype(StoragePolicy::PrepareRestore(
      std::declval<const state_slot&>(), std::declval<state_slot&>()));
  using store_result = decltype(StoragePolicy::StoreCandidate(
      std::declval<state_slot&>(), std::declval<const state_slot&>()));
  using copy_result = decltype(StoragePolicy::CopyOut(
      std::declval<const state_slot&>(), std::declval<state_slot&>()));
  using discard_result = decltype(
      StoragePolicy::DiscardCandidate(std::declval<state_slot&>()));
  using prepublish_result = decltype(StoragePolicy::CanPublish(
      std::declval<const state_slot&>(), std::declval<const state_slot&>()));
  using publish_result = decltype(StoragePolicy::Publish(
      std::declval<state_slot&>(), std::declval<state_slot&>()));
  using disjoint_result = decltype(StoragePolicy::Disjoint(
      std::declval<const state_slot&>(), std::declval<const state_slot&>()));
  using interior_hash_result = decltype(
      StoragePolicy::HashInterior(std::declval<const InteriorState&>()));
  using trace_hash_result = decltype(
      StoragePolicy::HashTrace(std::declval<const TraceState&>()));
  using admission_result = decltype(CollectivePolicy::Admit(
      std::declval<const AttemptCollectiveContext&>(), bool{},
      std::declval<FailureReason>(), std::declval<std::uint32_t>()));
  using collective_result = decltype(CollectivePolicy::Reduce(
      std::declval<const AttemptCollectiveContext&>(),
      std::declval<const GateReport&>(), std::declval<FailureReason>(),
      std::declval<std::uint32_t>()));
  using restore_collective_result = decltype(CollectivePolicy::ReduceRestore(
      std::declval<const RestoreCollectiveContext&>(),
      std::declval<RestartCompatibility>(),
      std::declval<FailureReason>(), std::declval<std::uint32_t>()));

  static_assert(!std::is_polymorphic<StoragePolicy>::value,
                "HDG run storage policy must be non-polymorphic");
  static_assert(!std::is_polymorphic<CollectivePolicy>::value,
                "HDG run collective policy must be non-polymorphic");
  static_assert(StoragePolicy::preparation_preserves_committed,
                "preparation must preserve committed bytes");
  static_assert(StoragePolicy::private_copy_is_independent,
                "candidate/copy operations must create independent storage");
  static_assert(StoragePolicy::publish_is_atomic,
                "storage policy must declare atomic publication");
  static_assert(StoragePolicy::publish_preserves_disjoint_backing,
                "publication must preserve disjoint slot backing");
  static_assert(std::is_same<prepare_result, bool>::value,
                "PrepareCandidate must return bool");
  static_assert(std::is_same<restore_prepare_result, bool>::value,
                "PrepareRestore must return bool");
  static_assert(std::is_same<store_result, bool>::value,
                "StoreCandidate must return bool");
  static_assert(std::is_same<copy_result, bool>::value,
                "CopyOut must return bool");
  static_assert(std::is_same<discard_result, void>::value,
                "DiscardCandidate must return void");
  static_assert(std::is_same<prepublish_result, bool>::value,
                "CanPublish must return bool");
  static_assert(std::is_same<publish_result, void>::value,
                "Publish must return void");
  static_assert(std::is_same<disjoint_result, bool>::value,
                "Disjoint must return bool");
  static_assert(std::is_same<interior_hash_result, StateHash>::value,
                "HashInterior must return StateHash");
  static_assert(std::is_same<trace_hash_result, StateHash>::value,
                "HashTrace must return StateHash");
  static_assert(std::is_same<admission_result,
                             CollectiveAdmissionReport>::value,
                "collective Admit must return CollectiveAdmissionReport");
  static_assert(std::is_same<collective_result,
                             CollectiveGateReport>::value,
                "collective Reduce must return CollectiveGateReport");
  static_assert(std::is_same<restore_collective_result,
                             CollectiveRestoreReport>::value,
                "collective ReduceRestore must return CollectiveRestoreReport");
  static_assert(noexcept(StoragePolicy::PrepareCandidate(
                    std::declval<const state_slot&>(),
                    std::declval<state_slot&>())),
                "PrepareCandidate must be noexcept");
  static_assert(noexcept(StoragePolicy::PrepareRestore(
                    std::declval<const state_slot&>(),
                    std::declval<state_slot&>())),
                "PrepareRestore must be noexcept");
  static_assert(noexcept(StoragePolicy::StoreCandidate(
                    std::declval<state_slot&>(),
                    std::declval<const state_slot&>())),
                "StoreCandidate must be noexcept");
  static_assert(noexcept(StoragePolicy::CopyOut(
                    std::declval<const state_slot&>(),
                    std::declval<state_slot&>())),
                "CopyOut must be noexcept");
  static_assert(noexcept(StoragePolicy::DiscardCandidate(
                    std::declval<state_slot&>())),
                "DiscardCandidate must be noexcept");
  static_assert(noexcept(StoragePolicy::CanPublish(
                    std::declval<const state_slot&>(),
                    std::declval<const state_slot&>())),
                "CanPublish must be noexcept");
  static_assert(noexcept(StoragePolicy::Publish(
                    std::declval<state_slot&>(),
                    std::declval<state_slot&>())),
                "Publish must be noexcept");
  static_assert(noexcept(StoragePolicy::Disjoint(
                    std::declval<const state_slot&>(),
                    std::declval<const state_slot&>())),
                "Disjoint must be noexcept");
  static_assert(noexcept(CollectivePolicy::Admit(
                    std::declval<const AttemptCollectiveContext&>(), bool{},
                    std::declval<FailureReason>(),
                    std::declval<std::uint32_t>())),
                "collective Admit must be noexcept");
  static_assert(noexcept(CollectivePolicy::Reduce(
                    std::declval<const AttemptCollectiveContext&>(),
                    std::declval<const GateReport&>(),
                    std::declval<FailureReason>(),
                    std::declval<std::uint32_t>())),
                "collective Reduce must be noexcept");
  static_assert(noexcept(CollectivePolicy::ReduceRestore(
                    std::declval<const RestoreCollectiveContext&>(),
                    std::declval<RestartCompatibility>(),
                    std::declval<FailureReason>(),
                    std::declval<std::uint32_t>())),
                "collective ReduceRestore must be noexcept");
  static_assert(noexcept(StoragePolicy::HashInterior(
                    std::declval<const InteriorState&>())),
                "HashInterior must be noexcept");
  static_assert(noexcept(StoragePolicy::HashTrace(
                    std::declval<const TraceState&>())),
                "HashTrace must be noexcept");

public:
  template <DerivedProductKind Kind>
  class ProductToken
  {
  public:
    ProductToken() = delete;
    ProductToken(const ProductToken&) = delete;
    ProductToken& operator=(const ProductToken&) = delete;

    ProductToken(ProductToken&& other) noexcept
        : incarnation_(other.incarnation_),
          attempt_(other.attempt_),
          serial_(other.serial_),
          valid_(other.valid_)
    {
      other.valid_ = false;
    }

    ProductToken& operator=(ProductToken&& other) noexcept
    {
      if (this != &other)
      {
        incarnation_ = other.incarnation_;
        attempt_ = other.attempt_;
        serial_ = other.serial_;
        valid_ = other.valid_;
        other.valid_ = false;
      }
      return *this;
    }

  private:
    ProductToken(const std::uint64_t incarnation,
                 const AttemptIdentity attempt,
                 const std::uint64_t serial) noexcept
        : incarnation_(incarnation),
          attempt_(attempt),
          serial_(serial),
          valid_(true)
    {}

    std::uint64_t incarnation_ = 0;
    AttemptIdentity attempt_{};
    std::uint64_t serial_ = 0;
    bool valid_ = false;

    friend class PersistentRunContract;
  };

  struct ProductBundle
  {
    std::optional<ProductToken<DerivedProductKind::residual>> residual{};
    std::optional<ProductToken<DerivedProductKind::matrix>> matrix{};
    std::optional<ProductToken<DerivedProductKind::factor>> factor{};
    std::optional<ProductToken<DerivedProductKind::solution>> solution{};
  };

  /** Read-only authority over one exactly frozen candidate generation. */
  class StagingToken
  {
  public:
    StagingToken() = delete;
    StagingToken(const StagingToken&) = delete;
    StagingToken& operator=(const StagingToken&) = delete;
    StagingToken& operator=(StagingToken&&) = delete;

    StagingToken(StagingToken&& other) noexcept
        : owner_(other.owner_),
          incarnation_(other.incarnation_),
          attempt_(other.attempt_),
          candidate_serial_(other.candidate_serial_),
          bound_phase_(other.bound_phase_),
          entry_(other.entry_),
          target_(other.target_),
          interior_hash_(other.interior_hash_),
          trace_hash_(other.trace_hash_),
          checkpoint_(other.checkpoint_),
          valid_(other.valid_)
    {
      other.valid_ = false;
    }

    ~StagingToken() noexcept = default;

    const StateCoordinates& entry_coordinates() const noexcept
    {
      return entry_;
    }
    const StateCoordinates& target_coordinates() const noexcept
    {
      return target_;
    }
    const StateHash& interior_hash() const noexcept { return interior_hash_; }
    const StateHash& trace_hash() const noexcept { return trace_hash_; }
    const CheckpointDescriptor& checkpoint_descriptor() const noexcept
    {
      return checkpoint_;
    }
    const CheckpointDescriptor& future_checkpoint() const noexcept
    {
      return checkpoint_;
    }

  private:
    StagingToken(const PersistentRunContract* owner,
                 const std::uint64_t incarnation,
                 const AttemptIdentity attempt,
                 const std::uint64_t candidate_serial,
                 const StateCoordinates entry,
                 const StateCoordinates target,
                 const StateHash interior_hash,
                 const StateHash trace_hash,
                 const CheckpointDescriptor checkpoint) noexcept
        : owner_(owner),
          incarnation_(incarnation),
          attempt_(attempt),
          candidate_serial_(candidate_serial),
          bound_phase_(AttemptPhase::frozen),
          entry_(entry),
          target_(target),
          interior_hash_(interior_hash),
          trace_hash_(trace_hash),
          checkpoint_(checkpoint),
          valid_(true)
    {}

    const PersistentRunContract* owner_ = nullptr;
    std::uint64_t incarnation_ = 0;
    AttemptIdentity attempt_{};
    std::uint64_t candidate_serial_ = 0;
    AttemptPhase bound_phase_ = AttemptPhase::idle;
    StateCoordinates entry_{};
    StateCoordinates target_{};
    StateHash interior_hash_{};
    StateHash trace_hash_{};
    CheckpointDescriptor checkpoint_{};
    bool valid_ = false;

    friend class PersistentRunContract;
  };

  /** Single-use authority to perform the nofail reversible slot exchange. */
  class PreparedCommitToken
  {
  public:
    PreparedCommitToken() = delete;
    PreparedCommitToken(const PreparedCommitToken&) = delete;
    PreparedCommitToken& operator=(const PreparedCommitToken&) = delete;
    PreparedCommitToken& operator=(PreparedCommitToken&&) = delete;

    PreparedCommitToken(PreparedCommitToken&& other) noexcept
        : owner_(other.owner_),
          incarnation_(other.incarnation_),
          attempt_(other.attempt_),
          candidate_serial_(other.candidate_serial_),
          bound_phase_(other.bound_phase_),
          checkpoint_(other.checkpoint_),
          valid_(other.valid_)
    {
      other.valid_ = false;
    }

    ~PreparedCommitToken() noexcept = default;

    const CheckpointDescriptor& checkpoint_descriptor() const noexcept
    {
      return checkpoint_;
    }
    const CheckpointDescriptor& future_checkpoint() const noexcept
    {
      return checkpoint_;
    }

  private:
    PreparedCommitToken(const PersistentRunContract* owner,
                        const std::uint64_t incarnation,
                        const AttemptIdentity attempt,
                        const std::uint64_t candidate_serial,
                        const CheckpointDescriptor checkpoint) noexcept
        : owner_(owner),
          incarnation_(incarnation),
          attempt_(attempt),
          candidate_serial_(candidate_serial),
          bound_phase_(AttemptPhase::prepared),
          checkpoint_(checkpoint),
          valid_(true)
    {}

    const PersistentRunContract* owner_ = nullptr;
    std::uint64_t incarnation_ = 0;
    AttemptIdentity attempt_{};
    std::uint64_t candidate_serial_ = 0;
    AttemptPhase bound_phase_ = AttemptPhase::idle;
    CheckpointDescriptor checkpoint_{};
    bool valid_ = false;

    friend class PersistentRunContract;
  };

  /** Single-use authority to finalize or exactly revert one exchanged slot. */
  class ProvisionalCommitToken
  {
  public:
    ProvisionalCommitToken() = delete;
    ProvisionalCommitToken(const ProvisionalCommitToken&) = delete;
    ProvisionalCommitToken& operator=(const ProvisionalCommitToken&) = delete;
    ProvisionalCommitToken& operator=(ProvisionalCommitToken&&) = delete;

    ProvisionalCommitToken(ProvisionalCommitToken&& other) noexcept
        : owner_(other.owner_),
          incarnation_(other.incarnation_),
          attempt_(other.attempt_),
          candidate_serial_(other.candidate_serial_),
          bound_phase_(other.bound_phase_),
          valid_(other.valid_)
    {
      other.valid_ = false;
    }

    ~ProvisionalCommitToken() noexcept = default;

  private:
    ProvisionalCommitToken(const PersistentRunContract* owner,
                           const std::uint64_t incarnation,
                           const AttemptIdentity attempt,
                           const std::uint64_t candidate_serial) noexcept
        : owner_(owner),
          incarnation_(incarnation),
          attempt_(attempt),
          candidate_serial_(candidate_serial),
          bound_phase_(AttemptPhase::provisional),
          valid_(true)
    {}

    const PersistentRunContract* owner_ = nullptr;
    std::uint64_t incarnation_ = 0;
    AttemptIdentity attempt_{};
    std::uint64_t candidate_serial_ = 0;
    AttemptPhase bound_phase_ = AttemptPhase::idle;
    bool valid_ = false;

    friend class PersistentRunContract;
  };

  /** Pointer-free value; all use is revalidated by a live contract. */
  class CandidateCapability
  {
  public:
    CandidateCapability() noexcept = default;

  private:
    CandidateCapability(const std::uint64_t incarnation,
                        const AttemptIdentity attempt,
                        const std::uint64_t serial) noexcept
        : incarnation_(incarnation), attempt_(attempt), serial_(serial)
    {}

    std::uint64_t incarnation_ = 0;
    AttemptIdentity attempt_{};
    std::uint64_t serial_ = 0;
    friend class PersistentRunContract;
  };

  /** Pointer-free publication coordinate; stale publications cannot inspect. */
  class PublishedCapability
  {
  public:
    PublishedCapability() noexcept = default;

  private:
    PublishedCapability(const std::uint64_t incarnation,
                        const std::uint64_t serial) noexcept
        : incarnation_(incarnation), serial_(serial)
    {}

    std::uint64_t incarnation_ = 0;
    std::uint64_t serial_ = 0;
    friend class PersistentRunContract;
  };

  PersistentRunContract(state_slot committed,
                        state_slot candidate,
                        const StateCoordinates coordinates,
                        const PersistentResourceDescriptor resources,
                        const std::uint64_t initial_attempt_sequence = 0)
      : committed_(std::move(committed)),
        candidate_(std::move(candidate)),
        coordinates_(coordinates),
        resources_(resources),
        attempt_sequence_(initial_attempt_sequence),
        incarnation_(detail::AcquireContractIncarnation())
  {
    if (incarnation_ == 0)
    {
      constructor_failure_ = FailureReason::incarnation_exhausted;
    }
    else if (!resources_.Valid())
    {
      constructor_failure_ = FailureReason::invalid_resource_descriptor;
    }
    else if (!std::isfinite(coordinates_.time) ||
             !std::isfinite(coordinates_.dt))
    {
      constructor_failure_ = FailureReason::invalid_dt;
    }
    else if (!StoragePolicy::Disjoint(committed_, candidate_))
    {
      constructor_failure_ = FailureReason::shared_backing_store;
      storage_valid_ = false;
    }
    if (constructor_failure_ != FailureReason::none)
    {
      status_ = RunStatus::invalid_request;
      last_failure_ = constructor_failure_;
      RememberFirstFailure(constructor_failure_);
    }
  }

  PersistentRunContract(const PersistentRunContract&) = delete;
  PersistentRunContract& operator=(const PersistentRunContract&) = delete;
  PersistentRunContract(PersistentRunContract&&) = delete;
  PersistentRunContract& operator=(PersistentRunContract&&) = delete;

  RunStatus status() const noexcept { return status_; }
  FailureReason last_failure() const noexcept { return last_failure_; }
  StateCoordinates coordinates() const noexcept { return coordinates_; }
  const PersistentResourceDescriptor& resources() const noexcept
  {
    return resources_;
  }
  const RunTelemetry& telemetry() const noexcept { return telemetry_; }

  /**
   * Adapter preflight before any attempt transfers occur. A false result means
   * FinishAttempt or PrepareCommit will reject that report without mutating
   * any overflowing transfer total.
   */
  bool CanAccountAttemptTransfers(
      const TransferAccounting& transfers) const noexcept
  {
    return detail::CanAddTransfersExactly(telemetry_.attempted_transfers,
                                          transfers) &&
        detail::CanAddTransfersExactly(telemetry_.rolled_back_transfers,
                                       transfers) &&
        detail::CanAddTransfersExactly(telemetry_.committed_transfers,
                                       transfers);
  }

  PublishedCapability Old() const noexcept
  {
    return {incarnation_, publication_serial_};
  }
  PublishedCapability Published() const noexcept { return Old(); }

  bool PublishedCoordinates(const PublishedCapability& capability,
                            StateCoordinates& copy) const noexcept
  {
    if (!ValidPublished(capability)) { return false; }
    copy = coordinates_;
    return ValidPublished(capability);
  }

  bool CopyPublished(const PublishedCapability& capability,
                     state_slot& copy) const noexcept
  {
    if (!ValidPublished(capability)) { return false; }
    const state_slot& published = PublishedSlot();
    const bool copied = StoragePolicy::CopyOut(published, copy);
    return copied && StoragePolicy::Disjoint(published, copy) &&
        ValidPublished(capability);
  }

  std::optional<CandidateCapability> AcquireCandidate(
      const AttemptIdentity& attempt) noexcept
  {
    if (!IsCurrentAttempt(attempt))
    {
      RejectStaleConsumer();
      return std::nullopt;
    }
    return CandidateCapability{incarnation_, attempt, candidate_serial_};
  }

  bool StoreCandidate(const CandidateCapability& capability,
                      const state_slot& value) noexcept
  {
    if (!ValidCandidate(capability))
    {
      RejectStaleConsumer();
      return false;
    }
    const bool stored = StoragePolicy::StoreCandidate(candidate_, value);
    const bool disjoint = StoragePolicy::Disjoint(committed_, candidate_) &&
        StoragePolicy::Disjoint(candidate_, value);
    if (!stored)
    {
      candidate_storage_failure_ = FailureReason::candidate_preparation;
      return false;
    }
    if (!disjoint)
    {
      candidate_storage_failure_ = FailureReason::shared_backing_store;
      return false;
    }
    return ValidCandidate(capability);
  }

  bool CopyCandidate(const CandidateCapability& capability,
                     state_slot& copy) const noexcept
  {
    if (!ValidCandidate(capability)) { return false; }
    const bool copied = StoragePolicy::CopyOut(candidate_, copy);
    return copied && StoragePolicy::Disjoint(candidate_, copy) &&
        ValidCandidate(capability);
  }

  bool IsCurrentAttempt(const AttemptIdentity& attempt) const noexcept
  {
    return phase_ == AttemptPhase::mutable_candidate &&
        SameLiveAttempt(attempt);
  }

  template <DerivedProductKind Kind>
  std::optional<ProductToken<Kind>> IssueProduct(
      const AttemptIdentity& attempt) noexcept
  {
    if (!IsCurrentAttempt(attempt))
    {
      RejectStaleConsumer();
      return std::nullopt;
    }
    constexpr std::size_t index = detail::ProductIndex(Kind);
    static_assert(index < 4, "unknown required HDG product kind");
    const std::uint8_t bit = static_cast<std::uint8_t>(1U << index);
    if ((issued_product_mask_ & bit) != 0)
    {
      telemetry_.duplicate_product_issues_rejected = detail::SaturatingAdd(
          telemetry_.duplicate_product_issues_rejected, 1);
      return std::nullopt;
    }
    if (index != issued_product_count_)
    {
      telemetry_.out_of_order_product_issues_rejected = detail::SaturatingAdd(
          telemetry_.out_of_order_product_issues_rejected, 1);
      return std::nullopt;
    }
    if (product_serial_ == std::numeric_limits<std::uint64_t>::max())
    {
      return std::nullopt;
    }
    issued_product_mask_ = static_cast<std::uint8_t>(issued_product_mask_ | bit);
    ++issued_product_count_;
    issued_product_serials_[index] = ++product_serial_;
    return ProductToken<Kind>{incarnation_, attempt,
                              issued_product_serials_[index]};
  }

  template <
      typename Policy = StoragePolicy,
      std::enable_if_t<
          std::is_same<Policy, StoragePolicy>::value &&
              detail::SupportsTwoPhaseSlotExchange<Policy,
                                                   state_slot>::value,
          int> = 0>
  std::optional<StagingToken> FreezeCandidate(
      const AttemptIdentity& attempt,
      const std::uint64_t candidate_cursor) noexcept
  {
    if (phase_ != AttemptPhase::mutable_candidate ||
        !SameLiveAttempt(attempt) ||
        candidate_storage_failure_ != FailureReason::none ||
        !StoragePolicy::Disjoint(committed_, candidate_))
    {
      RejectStaleConsumer();
      return std::nullopt;
    }
    const RuntimeTupleDescriptor tuple = Selection::Descriptor();
    if (!tuple.Valid()) { return std::nullopt; }

    const StateCoordinates entry = coordinates_;
    const StateCoordinates target = PendingTarget(candidate_cursor);
    const StateHash interior_hash =
        StoragePolicy::HashInterior(candidate_.interior);
    const StateHash trace_hash = StoragePolicy::HashTrace(candidate_.trace);
    const CheckpointDescriptor checkpoint{
        checkpoint_schema_version,
        true,
        RestartMode::same_backend_rank_partition,
        tuple,
        resources_.mfem_pin,
        resources_.mesh_hash,
        resources_.partition_hash,
        resources_.topology_hash,
        resources_.rank_count,
        resources_.polynomial_order,
        resources_.layout,
        tuple.precision,
        candidate_cursor,
        pending_dt_,
        target,
        detail::CoordinateFingerprint(target),
        interior_hash,
        trace_hash};

    phase_ = AttemptPhase::frozen;
    return StagingToken{this,
                        incarnation_,
                        active_attempt_,
                        candidate_serial_,
                        entry,
                        target,
                        interior_hash,
                        trace_hash,
                        checkpoint};
  }

  template <
      typename Policy = StoragePolicy,
      std::enable_if_t<
          std::is_same<Policy, StoragePolicy>::value &&
              detail::SupportsTwoPhaseSlotExchange<Policy,
                                                   state_slot>::value,
          int> = 0>
  bool CopyPrepared(const StagingToken& token,
                    state_slot& copy) const noexcept
  {
    if (!ValidStagingToken(token) || !FrozenCandidateMatches(token))
    {
      return false;
    }
    const bool copied = StoragePolicy::CopyOut(candidate_, copy);
    return copied && StoragePolicy::Disjoint(candidate_, copy) &&
        StoragePolicy::Disjoint(committed_, copy) &&
        ValidStagingToken(token) && FrozenCandidateMatches(token);
  }

  template <
      typename Policy = StoragePolicy,
      std::enable_if_t<
          std::is_same<Policy, StoragePolicy>::value &&
              detail::SupportsTwoPhaseSlotExchange<Policy,
                                                   state_slot>::value,
          int> = 0>
  std::optional<PreparedCommitToken> PrepareCommit(
      StagingToken&& token,
      const AttemptReport& report,
      ProductBundle&& products) noexcept
  {
    if (!ValidStagingToken(token))
    {
      RejectStaleConsumer();
      return std::nullopt;
    }

    token.valid_ = false;
    const AttemptIdentity attempt = token.attempt_;
    const bool current = SameLiveAttempt(attempt);
    const bool originating_reason_valid =
        detail::IsNamedFailureReason(report.originating_failure);
    FailureReason product_reason = FailureReason::stale_attempt;
    if (current)
    {
      product_reason = ValidateAndConsumeProducts(attempt, products);
    }
    else
    {
      InvalidateProducts(products);
      RejectStaleConsumer();
    }

    GateReport local_gates = report.gates;
    const bool exact_transfers =
        CanAccountAttemptTransfers(report.telemetry.transfers);
    if (!report.telemetry.transfers.complete || !exact_transfers ||
        !report.telemetry.prohibited.Clean() ||
        (!resources_.reuse.per_step_allocation_required &&
         report.telemetry.transfers.allocation_events != 0))
    {
      local_gates.telemetry = false;
    }
    if (!std::isfinite(report.telemetry.true_relative_residual) ||
        report.telemetry.true_relative_residual < 0.0 ||
        report.telemetry.true_relative_residual >
            true_relative_residual_limit)
    {
      local_gates.true_residual = false;
    }

    FailureReason storage_reason = candidate_storage_failure_;
    if (current && storage_reason == FailureReason::none &&
        (!detail::SameCoordinates(coordinates_, token.entry_) ||
         !detail::SameCoordinates(PendingTarget(token.target_.cursor),
                                  token.target_) ||
         report.candidate_cursor != token.target_.cursor))
    {
      storage_reason = FailureReason::stale_attempt;
    }
    if (current && storage_reason == FailureReason::none &&
        !StoragePolicy::Disjoint(committed_, candidate_))
    {
      storage_reason = FailureReason::shared_backing_store;
    }
    if (current && storage_reason == FailureReason::none &&
        (StoragePolicy::HashInterior(candidate_.interior) !=
             token.interior_hash_ ||
         StoragePolicy::HashTrace(candidate_.trace) != token.trace_hash_))
    {
      storage_reason = FailureReason::candidate_preparation;
    }
    if (current && storage_reason == FailureReason::none &&
        !StoragePolicy::CanPublish(committed_, candidate_))
    {
      storage_reason = FailureReason::publication_precondition;
    }
    if (current && storage_reason == FailureReason::none &&
        publication_serial_ == std::numeric_limits<std::uint64_t>::max())
    {
      storage_reason = FailureReason::coordinate_overflow;
    }

    FailureReason local_reason = FailureReason::none;
    if (!originating_reason_valid)
    {
      local_reason = FailureReason::collective_protocol;
    }
    else if (!current)
    {
      local_reason = FailureReason::stale_attempt;
    }
    else if (report.originating_failure != FailureReason::none)
    {
      local_reason = report.originating_failure;
    }
    else if (product_reason != FailureReason::none)
    {
      local_reason = product_reason;
    }
    else if (storage_reason != FailureReason::none)
    {
      local_reason = storage_reason;
    }
    else
    {
      local_reason = detail::FirstFailedGate(local_gates);
    }
    if (local_reason != FailureReason::none) { local_gates.success = false; }

    const AttemptCollectiveContext local_context{
        attempt, token.entry_, token.target_, token.target_.cursor};
    const CollectiveGateReport collective = CollectivePolicy::Reduce(
        local_context, local_gates, local_reason, resources_.rank_count);
    const FailureReason reason = ResolveGateCollective(
        local_context, local_gates, local_reason, collective);

    AttemptTelemetry attempt_telemetry = report.telemetry;
    attempt_telemetry.identity = current ? active_attempt_ : attempt;
    attempt_telemetry.failure_reason = reason;
    attempt_telemetry.validated_product_mask =
        product_reason == FailureReason::none ? required_product_mask : 0;
    telemetry_.last_attempt = attempt_telemetry;
    if (exact_transfers)
    {
      detail::AddTransfersExact(telemetry_.attempted_transfers,
                                report.telemetry.transfers);
    }
    else
    {
      telemetry_.transfer_accounting_overflow_rejections =
          detail::SaturatingAdd(
              telemetry_.transfer_accounting_overflow_rejections, 1);
    }

    if (reason != FailureReason::none)
    {
      RememberFirstFailure(reason);
      Rollback(active_attempt_, reason, report.telemetry.transfers,
               exact_transfers);
      return std::nullopt;
    }

    prepared_target_ = token.target_;
    prepared_telemetry_ = attempt_telemetry;
    prepared_transfers_ = report.telemetry.transfers;
    prepared_checkpoint_ = token.checkpoint_;
    phase_ = AttemptPhase::prepared;
    return PreparedCommitToken{
        this,
        incarnation_,
        active_attempt_,
        candidate_serial_,
        prepared_checkpoint_};
  }

  template <
      typename Policy = StoragePolicy,
      std::enable_if_t<
          std::is_same<Policy, StoragePolicy>::value &&
              detail::SupportsTwoPhaseSlotExchange<Policy,
                                                   state_slot>::value,
          int> = 0>
  std::optional<ProvisionalCommitToken> CommitPrepared(
      PreparedCommitToken&& token) noexcept
  {
    if (!ValidPreparedToken(token))
    {
      RejectStaleConsumer();
      return std::nullopt;
    }
    token.valid_ = false;
    StoragePolicy::Publish(committed_, candidate_);
    phase_ = AttemptPhase::provisional;
    return ProvisionalCommitToken{
        this, incarnation_, active_attempt_, candidate_serial_};
  }

  template <
      typename Policy = StoragePolicy,
      std::enable_if_t<
          std::is_same<Policy, StoragePolicy>::value &&
              detail::SupportsTwoPhaseSlotExchange<Policy,
                                                   state_slot>::value,
          int> = 0>
  AttemptOutcome FinalizeCommit(ProvisionalCommitToken&& token) noexcept
  {
    if (!ValidProvisionalToken(token))
    {
      RejectStaleConsumer();
      return {status_, FailureReason::stale_attempt, false, false};
    }
    token.valid_ = false;

    coordinates_ = prepared_target_;
    ++publication_serial_;
    prepared_telemetry_.failure_reason = FailureReason::none;
    telemetry_.last_attempt = prepared_telemetry_;
    telemetry_.accepted = {true, coordinates_, prepared_telemetry_};
    detail::AddTransfersExact(telemetry_.committed_transfers,
                              prepared_transfers_);
    telemetry_.commits = detail::SaturatingAdd(telemetry_.commits, 1);
    active_ = false;
    phase_ = AttemptPhase::idle;
    candidate_storage_failure_ = FailureReason::none;
    status_ = RunStatus::committed;
    last_failure_ = FailureReason::none;
    ClearPreparedCommit();
    return {status_, FailureReason::none, true, false};
  }

  template <
      typename Policy = StoragePolicy,
      std::enable_if_t<
          std::is_same<Policy, StoragePolicy>::value &&
              detail::SupportsTwoPhaseSlotExchange<Policy,
                                                   state_slot>::value,
          int> = 0>
  AttemptOutcome AbortPrepared(
      PreparedCommitToken&& token,
      FailureReason reason = FailureReason::backend_failure) noexcept
  {
    if (!ValidPreparedToken(token))
    {
      RejectStaleConsumer();
      return {status_, FailureReason::stale_attempt, false, false};
    }
    token.valid_ = false;
    reason = NormalizeRollbackReason(reason);
    RememberFirstFailure(reason);
    telemetry_.last_attempt.failure_reason = reason;
    return Rollback(active_attempt_, reason, prepared_transfers_, true);
  }

  template <
      typename Policy = StoragePolicy,
      std::enable_if_t<
          std::is_same<Policy, StoragePolicy>::value &&
              detail::SupportsTwoPhaseSlotExchange<Policy,
                                                   state_slot>::value,
          int> = 0>
  AttemptOutcome RevertCommit(
      ProvisionalCommitToken&& token,
      FailureReason reason = FailureReason::backend_failure) noexcept
  {
    if (!ValidProvisionalToken(token))
    {
      RejectStaleConsumer();
      return {status_, FailureReason::stale_attempt, false, false};
    }
    token.valid_ = false;
    Policy::RevertPublish(committed_, candidate_);
    phase_ = AttemptPhase::prepared;
    reason = NormalizeRollbackReason(reason);
    RememberFirstFailure(reason);
    telemetry_.last_attempt.failure_reason = reason;
    return Rollback(active_attempt_, reason, prepared_transfers_, true);
  }

  BeginOutcome BeginStep(const StepRequest& request) noexcept
  {
    if (TwoPhaseCommitBusy())
    {
      RejectStaleConsumer();
      return {false, active_attempt_, FailureReason::stale_attempt};
    }
    return PrepareAndAdmit(request, false);
  }

  BeginOutcome Retry() noexcept
  {
    if (TwoPhaseCommitBusy())
    {
      RejectStaleConsumer();
      return {false, active_attempt_, FailureReason::retry_exhausted};
    }
    return PrepareAndAdmit({InteriorSpace::l2, pending_dt_}, true);
  }

  AttemptOutcome FinishAttempt(const AttemptIdentity& attempt,
                               const AttemptReport& report,
                               ProductBundle&& products) noexcept
  {
    if (TwoPhaseCommitBusy())
    {
      InvalidateProducts(products);
      RejectStaleConsumer();
      return {status_, FailureReason::stale_attempt, false, false};
    }
    const bool current = IsCurrentAttempt(attempt);
    const bool originating_reason_valid =
        detail::IsNamedFailureReason(report.originating_failure);
    FailureReason product_reason = FailureReason::stale_attempt;
    if (current)
    {
      product_reason = ValidateAndConsumeProducts(attempt, products);
    }
    else
    {
      InvalidateProducts(products);
      RejectStaleConsumer();
    }

    GateReport local_gates = report.gates;
    const bool exact_transfers =
        CanAccountAttemptTransfers(report.telemetry.transfers);
    if (!report.telemetry.transfers.complete || !exact_transfers ||
        !report.telemetry.prohibited.Clean() ||
        (!resources_.reuse.per_step_allocation_required &&
         report.telemetry.transfers.allocation_events != 0))
    {
      local_gates.telemetry = false;
    }
    if (!std::isfinite(report.telemetry.true_relative_residual) ||
        report.telemetry.true_relative_residual < 0.0 ||
        report.telemetry.true_relative_residual >
            true_relative_residual_limit)
    {
      local_gates.true_residual = false;
    }

    FailureReason storage_reason = candidate_storage_failure_;
    if (current && storage_reason == FailureReason::none &&
        !StoragePolicy::Disjoint(committed_, candidate_))
    {
      storage_reason = FailureReason::shared_backing_store;
    }
    if (current && storage_reason == FailureReason::none &&
        !StoragePolicy::CanPublish(committed_, candidate_))
    {
      storage_reason = FailureReason::publication_precondition;
    }
    if (current && storage_reason == FailureReason::none &&
        publication_serial_ == std::numeric_limits<std::uint64_t>::max())
    {
      storage_reason = FailureReason::coordinate_overflow;
    }

    FailureReason local_reason = FailureReason::none;
    if (!originating_reason_valid)
    {
      local_reason = FailureReason::collective_protocol;
    }
    else if (!current)
    {
      local_reason = FailureReason::stale_attempt;
    }
    else if (report.originating_failure != FailureReason::none)
    {
      local_reason = report.originating_failure;
    }
    else if (product_reason != FailureReason::none)
    {
      local_reason = product_reason;
    }
    else if (storage_reason != FailureReason::none)
    {
      local_reason = storage_reason;
    }
    else
    {
      local_reason = detail::FirstFailedGate(local_gates);
    }
    if (local_reason != FailureReason::none) { local_gates.success = false; }

    const StateCoordinates target{
        pending_target_time_,
        coordinates_.step == std::numeric_limits<std::uint64_t>::max()
            ? coordinates_.step
            : coordinates_.step + 1,
        coordinates_.generation == std::numeric_limits<std::uint64_t>::max()
            ? coordinates_.generation
            : coordinates_.generation + 1,
        report.candidate_cursor,
        pending_dt_};
    const AttemptCollectiveContext local_context{
        attempt, coordinates_, target, report.candidate_cursor};
    const CollectiveGateReport collective = CollectivePolicy::Reduce(
        local_context, local_gates, local_reason, resources_.rank_count);
    const FailureReason reason = ResolveGateCollective(
        local_context, local_gates, local_reason, collective);

    if (!current)
    {
      RememberFirstFailure(reason);
      if (active_ && reason != FailureReason::stale_attempt)
      {
        // A mixed stale/current identity or any contradictory collective
        // truth aborts the live attempt, but the stale caller's telemetry and
        // transfers never become attributes of that newer attempt.
        AttemptTelemetry active_failure{};
        active_failure.identity = active_attempt_;
        active_failure.transfers.complete = true;
        active_failure.failure_reason = reason;
        telemetry_.last_attempt = active_failure;
        return Rollback(active_attempt_, reason, TransferAccounting{}, false);
      }
      return {status_, reason, false,
              status_ == RunStatus::retry_available};
    }

    AttemptTelemetry attempt_telemetry = report.telemetry;
    attempt_telemetry.identity = current ? active_attempt_ : attempt;
    attempt_telemetry.failure_reason = reason;
    attempt_telemetry.validated_product_mask =
        product_reason == FailureReason::none ? required_product_mask : 0;
    telemetry_.last_attempt = attempt_telemetry;
    if (exact_transfers)
    {
      detail::AddTransfersExact(telemetry_.attempted_transfers,
                                report.telemetry.transfers);
    }
    else
    {
      telemetry_.transfer_accounting_overflow_rejections =
          detail::SaturatingAdd(
              telemetry_.transfer_accounting_overflow_rejections, 1);
    }

    if (reason != FailureReason::none)
    {
      RememberFirstFailure(reason);
      return Rollback(active_attempt_, reason, report.telemetry.transfers,
                      exact_transfers);
    }

    // Every fallible validation is complete. Publish is statically noexcept;
    // its atomic/disjoint semantics are the trusted adapter proof obligation.
    // Nothing below may reject the commit.
    StoragePolicy::Publish(committed_, candidate_);
    coordinates_ = target;
    ++publication_serial_;
    attempt_telemetry.failure_reason = FailureReason::none;
    telemetry_.last_attempt = attempt_telemetry;
    telemetry_.accepted = {true, coordinates_, attempt_telemetry};
    detail::AddTransfersExact(telemetry_.committed_transfers,
                              report.telemetry.transfers);
    telemetry_.commits = detail::SaturatingAdd(telemetry_.commits, 1);
    active_ = false;
    phase_ = AttemptPhase::idle;
    candidate_storage_failure_ = FailureReason::none;
    status_ = RunStatus::committed;
    last_failure_ = FailureReason::none;
    return {status_, FailureReason::none, true, false};
  }

  bool TryMakeCheckpoint(CheckpointDescriptor& checkpoint) const noexcept
  {
    if (active_ || !storage_valid_ || !telemetry_.accepted.valid ||
        !detail::SameCoordinates(telemetry_.accepted.coordinates,
                                 coordinates_) ||
        !std::isfinite(coordinates_.time) ||
        !std::isfinite(coordinates_.dt) || coordinates_.dt <= 0.0)
    {
      return false;
    }
    const RuntimeTupleDescriptor tuple = Selection::Descriptor();
    if (!tuple.Valid()) { return false; }
    checkpoint = {checkpoint_schema_version,
                  true,
                  RestartMode::same_backend_rank_partition,
                  tuple,
                  resources_.mfem_pin,
                  resources_.mesh_hash,
                  resources_.partition_hash,
                  resources_.topology_hash,
                  resources_.rank_count,
                  resources_.polynomial_order,
                  resources_.layout,
                  tuple.precision,
                  coordinates_.cursor,
                  coordinates_.dt,
                  coordinates_,
                  detail::CoordinateFingerprint(coordinates_),
                  StoragePolicy::HashInterior(committed_.interior),
                  StoragePolicy::HashTrace(committed_.trace)};
    return true;
  }

  RestoreOutcome Restore(const CheckpointDescriptor& checkpoint,
                         const state_slot& loaded) noexcept
  {
    const StateHash loaded_interior_hash =
        StoragePolicy::HashInterior(loaded.interior);
    const StateHash loaded_trace_hash = StoragePolicy::HashTrace(loaded.trace);
    RestartCompatibility local_compatibility = ValidateCheckpoint(
        checkpoint, loaded_interior_hash, loaded_trace_hash);
    bool preparation_attempted = false;
    bool candidate_prepared = false;
    StateHash prepared_interior_hash{};
    StateHash prepared_trace_hash{};
    if (local_compatibility == RestartCompatibility::compatible)
    {
      if (!StoragePolicy::Disjoint(committed_, candidate_))
      {
        local_compatibility = RestartCompatibility::shared_backing_store;
      }
      else
      {
        preparation_attempted = true;
        candidate_prepared = StoragePolicy::PrepareRestore(loaded, candidate_);
        if (!candidate_prepared)
        {
          local_compatibility = RestartCompatibility::candidate_preparation;
        }
        else
        {
          prepared_interior_hash =
              StoragePolicy::HashInterior(candidate_.interior);
          prepared_trace_hash = StoragePolicy::HashTrace(candidate_.trace);
          if (!StoragePolicy::Disjoint(committed_, candidate_) ||
              !StoragePolicy::Disjoint(candidate_, loaded))
          {
            local_compatibility = RestartCompatibility::shared_backing_store;
          }
          else if (prepared_interior_hash != checkpoint.interior_hash)
          {
            local_compatibility = RestartCompatibility::interior_hash_mismatch;
          }
          else if (prepared_trace_hash != checkpoint.trace_hash)
          {
            local_compatibility = RestartCompatibility::trace_hash_mismatch;
          }
          else if (!StoragePolicy::CanPublish(committed_, candidate_) ||
                   publication_serial_ ==
                       std::numeric_limits<std::uint64_t>::max())
          {
            local_compatibility =
                RestartCompatibility::publication_precondition;
          }
        }
      }
    }

    const FailureReason local_reason =
        local_compatibility == RestartCompatibility::compatible
        ? FailureReason::none
        : FailureReason::restart_incompatible;
    const RestoreCollectiveContext local_context{checkpoint,
                                                 loaded_interior_hash,
                                                 loaded_trace_hash,
                                                 prepared_interior_hash,
                                                 prepared_trace_hash,
                                                 preparation_attempted,
                                                 candidate_prepared};
    // Unconditional: corrupt descriptors, buffers, local storage, and busy
    // status all participate in exactly one restore reduction before publish.
    const CollectiveRestoreReport collective =
        CollectivePolicy::ReduceRestore(local_context,
                                        local_compatibility,
                                        local_reason,
                                        resources_.rank_count);
    const RestartCompatibility compatibility = ResolveRestoreCollective(
        local_context, local_compatibility, local_reason, collective);
    if (compatibility != RestartCompatibility::compatible)
    {
      if (preparation_attempted) { DiscardPrivateCandidate(); }
      const FailureReason reason =
          compatibility == RestartCompatibility::collective_protocol
          ? FailureReason::collective_protocol
          : FailureReason::restart_incompatible;
      RememberFirstFailure(reason);
      last_failure_ = reason;
      return {false, compatibility};
    }

    // Atomic verified restore: every local fallible check and the all-rank
    // identity/validity decision are complete. Publish is the trusted static
    // adapter proof obligation and no recoverable check follows it.
    StoragePolicy::Publish(committed_, candidate_);
    coordinates_ = checkpoint.coordinates;
    ++publication_serial_;
    active_ = false;
    phase_ = AttemptPhase::idle;
    retry_used_ = false;
    pending_dt_ = 0.0;
    pending_target_time_ = 0.0;
    candidate_storage_failure_ = FailureReason::none;
    AttemptTelemetry restored{};
    restored.identity = {0,
                         coordinates_.step,
                         coordinates_.generation,
                         0,
                         coordinates_.dt};
    restored.failure_reason = FailureReason::none;
    telemetry_.accepted = {true, coordinates_, restored};
    telemetry_.restores = detail::SaturatingAdd(telemetry_.restores, 1);
    status_ = RunStatus::committed;
    last_failure_ = FailureReason::none;
    return {true, RestartCompatibility::compatible};
  }

private:
  static constexpr std::uint8_t required_product_mask = 0x0fU;

  bool SameLiveAttempt(const AttemptIdentity& attempt) const noexcept
  {
    return active_ && detail::SameAttempt(attempt, active_attempt_);
  }

  bool TwoPhaseCommitBusy() const noexcept
  {
    return phase_ == AttemptPhase::frozen ||
        phase_ == AttemptPhase::prepared ||
        phase_ == AttemptPhase::provisional;
  }

  StateCoordinates PendingTarget(const std::uint64_t cursor) const noexcept
  {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return {pending_target_time_,
            coordinates_.step == maximum ? maximum : coordinates_.step + 1,
            coordinates_.generation == maximum
                ? maximum
                : coordinates_.generation + 1,
            cursor,
            pending_dt_};
  }

  const state_slot& PublishedSlot() const noexcept
  {
    return phase_ == AttemptPhase::provisional ? candidate_ : committed_;
  }

  bool ValidStagingToken(const StagingToken& token) const noexcept
  {
    return token.valid_ && token.owner_ == this &&
        token.incarnation_ == incarnation_ &&
        token.candidate_serial_ == candidate_serial_ &&
        token.bound_phase_ == AttemptPhase::frozen &&
        phase_ == AttemptPhase::frozen &&
        SameLiveAttempt(token.attempt_);
  }

  bool FrozenCandidateMatches(const StagingToken& token) const noexcept
  {
    return token.owner_ == this && token.incarnation_ == incarnation_ &&
        token.candidate_serial_ == candidate_serial_ &&
        token.bound_phase_ == AttemptPhase::frozen &&
        phase_ == AttemptPhase::frozen &&
        SameLiveAttempt(token.attempt_) &&
        detail::SameCoordinates(coordinates_, token.entry_) &&
        detail::SameCoordinates(PendingTarget(token.target_.cursor),
                                token.target_) &&
        StoragePolicy::Disjoint(committed_, candidate_) &&
        StoragePolicy::HashInterior(candidate_.interior) ==
            token.interior_hash_ &&
        StoragePolicy::HashTrace(candidate_.trace) == token.trace_hash_;
  }

  bool ValidPreparedToken(const PreparedCommitToken& token) const noexcept
  {
    return token.valid_ && token.owner_ == this &&
        token.incarnation_ == incarnation_ &&
        token.candidate_serial_ == candidate_serial_ &&
        token.bound_phase_ == AttemptPhase::prepared &&
        phase_ == AttemptPhase::prepared &&
        SameLiveAttempt(token.attempt_);
  }

  bool ValidProvisionalToken(
      const ProvisionalCommitToken& token) const noexcept
  {
    return token.valid_ && token.owner_ == this &&
        token.incarnation_ == incarnation_ &&
        token.candidate_serial_ == candidate_serial_ &&
        token.bound_phase_ == AttemptPhase::provisional &&
        phase_ == AttemptPhase::provisional &&
        SameLiveAttempt(token.attempt_);
  }

  static FailureReason NormalizeRollbackReason(
      const FailureReason reason) noexcept
  {
    if (!detail::IsNamedFailureReason(reason))
    {
      return FailureReason::collective_protocol;
    }
    return reason == FailureReason::none ? FailureReason::backend_failure
                                         : reason;
  }

  void ClearPreparedCommit() noexcept
  {
    prepared_target_ = {};
    prepared_telemetry_ = {};
    prepared_transfers_ = {};
    prepared_checkpoint_ = {};
  }

  bool ValidCandidate(const CandidateCapability& capability) const noexcept
  {
    return capability.incarnation_ == incarnation_ &&
        capability.serial_ == candidate_serial_ &&
        IsCurrentAttempt(capability.attempt_);
  }

  bool ValidPublished(const PublishedCapability& capability) const noexcept
  {
    return capability.incarnation_ == incarnation_ &&
        capability.serial_ == publication_serial_;
  }

  void RejectStaleConsumer() noexcept
  {
    telemetry_.stale_consumers_rejected = detail::SaturatingAdd(
        telemetry_.stale_consumers_rejected, 1);
  }

  StateCoordinates AdmissionTarget(const double dt) const noexcept
  {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return {coordinates_.time + dt,
            coordinates_.step == maximum ? maximum : coordinates_.step + 1,
            coordinates_.generation == maximum
                ? maximum
                : coordinates_.generation + 1,
            coordinates_.cursor,
            dt};
  }

  FailureReason ValidateAdmissionRequest(const StepRequest& request,
                                         const bool retry,
                                         const StateCoordinates& target) const
      noexcept
  {
    if (retry)
    {
      if (status_ != RunStatus::retry_available || retry_used_ || active_)
      {
        return FailureReason::retry_exhausted;
      }
    }
    else if (status_ != RunStatus::ready && status_ != RunStatus::committed)
    {
      return status_ == RunStatus::unsupported_h1
          ? FailureReason::unsupported_h1
          : (constructor_failure_ != FailureReason::none
                 ? constructor_failure_
                 : FailureReason::stale_attempt);
    }
    if (request.interior_space == InteriorSpace::h1)
    {
      return FailureReason::unsupported_h1;
    }
    if (request.interior_space != InteriorSpace::l2)
    {
      return FailureReason::invalid_interior_space;
    }
    if (!storage_valid_ || !resources_.Valid())
    {
      return storage_valid_ ? FailureReason::invalid_resource_descriptor
                            : FailureReason::shared_backing_store;
    }
    if (!std::isfinite(request.dt) || request.dt <= 0.0 ||
        !std::isfinite(coordinates_.time))
    {
      return FailureReason::invalid_dt;
    }
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (!std::isfinite(target.time) || coordinates_.step == maximum ||
        coordinates_.generation == maximum || attempt_sequence_ == maximum ||
        candidate_serial_ == maximum)
    {
      return FailureReason::coordinate_overflow;
    }
    if (!StoragePolicy::Disjoint(committed_, candidate_))
    {
      return FailureReason::shared_backing_store;
    }
    return FailureReason::none;
  }

  static RunStatus FailureStatus(const FailureReason reason) noexcept
  {
    if (reason == FailureReason::unsupported_h1)
    {
      return RunStatus::unsupported_h1;
    }
    if (reason == FailureReason::invalid_interior_space ||
        reason == FailureReason::invalid_dt ||
        reason == FailureReason::invalid_resource_descriptor ||
        reason == FailureReason::incarnation_exhausted)
    {
      return RunStatus::invalid_request;
    }
    return RunStatus::terminal_failure;
  }

  BeginOutcome PrepareAndAdmit(const StepRequest& request,
                               const bool retry) noexcept
  {
    const bool preserve_live_attempt = active_;
    telemetry_.admissions_started = detail::SaturatingAdd(
        telemetry_.admissions_started, 1);
    const std::uint64_t proposed_sequence =
        attempt_sequence_ == std::numeric_limits<std::uint64_t>::max()
        ? attempt_sequence_
        : attempt_sequence_ + 1;
    const AttemptIdentity proposed{proposed_sequence,
                                   coordinates_.step,
                                   coordinates_.generation,
                                   static_cast<unsigned char>(retry ? 1 : 0),
                                   request.dt};
    const StateCoordinates target = AdmissionTarget(request.dt);
    FailureReason local_reason =
        ValidateAdmissionRequest(request, retry, target);
    bool local_prepared = false;
    bool preparation_attempted = false;
    if (local_reason == FailureReason::none)
    {
      preparation_attempted = true;
      local_prepared = StoragePolicy::PrepareCandidate(committed_, candidate_);
      if (!local_prepared)
      {
        local_reason = FailureReason::candidate_preparation;
      }
      else if (!StoragePolicy::Disjoint(committed_, candidate_))
      {
        local_reason = FailureReason::shared_backing_store;
      }
      else if (!StoragePolicy::CanPublish(committed_, candidate_))
      {
        local_reason = FailureReason::publication_precondition;
      }
    }
    local_prepared = local_prepared && local_reason == FailureReason::none;

    // This is deliberately unconditional: every participant reaches exactly
    // one admission collective even when any local validation/preparation fails.
    const AttemptCollectiveContext local_context{
        proposed, coordinates_, target, coordinates_.cursor};
    const CollectiveAdmissionReport admission = CollectivePolicy::Admit(
        local_context, local_prepared, local_reason, resources_.rank_count);
    const FailureReason reason = ResolveAdmissionCollective(
        local_context, local_prepared, local_reason, admission);
    if (reason != FailureReason::none)
    {
      if (preparation_attempted &&
          StoragePolicy::Disjoint(committed_, candidate_))
      {
        StoragePolicy::DiscardCandidate(candidate_);
      }
      else if (preparation_attempted)
      {
        storage_valid_ = false;
      }
      telemetry_.admissions_failed = detail::SaturatingAdd(
          telemetry_.admissions_failed, 1);
      RememberFirstFailure(reason);
      if (preserve_live_attempt)
      {
        return {false, proposed, reason};
      }
      active_ = false;
      phase_ = AttemptPhase::idle;
      status_ = FailureStatus(reason);
      last_failure_ = reason;
      return {false, proposed, reason};
    }

    pending_dt_ = request.dt;
    pending_target_time_ = target.time;
    retry_used_ = retry;
    active_attempt_ = proposed;
    attempt_sequence_ = proposed.sequence;
    ++candidate_serial_;
    issued_product_mask_ = 0;
    issued_product_count_ = 0;
    issued_product_serials_.fill(0);
    candidate_storage_failure_ = FailureReason::none;
    active_ = true;
    phase_ = AttemptPhase::mutable_candidate;
    status_ = RunStatus::attempt_active;
    last_failure_ = FailureReason::none;
    telemetry_.attempts_started =
        detail::SaturatingAdd(telemetry_.attempts_started, 1);
    if (retry)
    {
      telemetry_.retries_started =
          detail::SaturatingAdd(telemetry_.retries_started, 1);
    }
    return {true, active_attempt_, FailureReason::none};
  }

  FailureReason ResolveAdmissionCollective(
      const AttemptCollectiveContext& local_context,
      const bool local_prepared,
      const FailureReason local_reason,
      const CollectiveAdmissionReport& report) const noexcept
  {
    const bool domains = detail::IsNamedFailureReason(local_reason) &&
        detail::IsNamedFailureReason(report.resolved_reason) &&
        detail::HasOnlyNamedFailureBits(report.failure_mask);
    if (!domains) { return FailureReason::collective_protocol; }
    const bool identity =
        report.participating_ranks == resources_.rank_count &&
        report.identity_consistent && report.failure_consistent &&
        detail::SameContext(report.context, local_context);
    const bool truth =
        report.any_failure ==
            (report.resolved_reason != FailureReason::none) &&
        report.any_failure == (report.failure_mask != 0) &&
        (local_reason == FailureReason::none ||
         (report.failure_mask & detail::FailureBit(local_reason)) != 0) &&
        (report.resolved_reason == FailureReason::none ||
         (report.failure_mask &
          detail::FailureBit(report.resolved_reason)) != 0) &&
        !(!local_prepared && report.all_prepared) &&
        !(local_reason != FailureReason::none && !report.any_failure) &&
        !(!report.all_prepared && !report.any_failure);
    if (!identity || !truth) { return FailureReason::collective_protocol; }
    if (report.any_failure) { return report.resolved_reason; }
    return report.all_prepared ? FailureReason::none
                               : FailureReason::collective_protocol;
  }

  FailureReason ResolveGateCollective(
      const AttemptCollectiveContext& local_context,
      const GateReport& local_gates,
      const FailureReason local_reason,
      const CollectiveGateReport& report) const noexcept
  {
    const bool domains = detail::IsNamedFailureReason(local_reason) &&
        detail::IsNamedFailureReason(report.resolved_reason) &&
        detail::HasOnlyNamedFailureBits(report.failure_mask);
    if (!domains) { return FailureReason::collective_protocol; }
    const bool identity =
        report.participating_ranks == resources_.rank_count &&
        report.identity_consistent && report.failure_consistent &&
        detail::SameContext(report.context, local_context);
    const bool truth =
        report.any_failure ==
            (report.resolved_reason != FailureReason::none) &&
        report.any_failure == (report.failure_mask != 0) &&
        (local_reason == FailureReason::none ||
         (report.failure_mask & detail::FailureBit(local_reason)) != 0) &&
        (report.resolved_reason == FailureReason::none ||
         (report.failure_mask &
          detail::FailureBit(report.resolved_reason)) != 0) &&
        !(local_reason != FailureReason::none && !report.any_failure) &&
        detail::GateReductionContainsLocal(local_gates, report.gates) &&
        !(detail::FirstFailedGate(report.gates) != FailureReason::none &&
          !report.any_failure);
    if (!identity || !truth) { return FailureReason::collective_protocol; }
    return report.any_failure ? report.resolved_reason : FailureReason::none;
  }

  RestartCompatibility ResolveRestoreCollective(
      const RestoreCollectiveContext& local_context,
      const RestartCompatibility local_compatibility,
      const FailureReason local_reason,
      const CollectiveRestoreReport& report) const noexcept
  {
    const bool domains =
        detail::IsNamedRestartCompatibility(local_compatibility) &&
        detail::IsNamedFailureReason(local_reason) &&
        detail::IsNamedRestartCompatibility(report.resolved_compatibility) &&
        detail::IsNamedFailureReason(report.resolved_reason) &&
        detail::HasOnlyNamedFailureBits(report.failure_mask) &&
        detail::HasOnlyNamedCompatibilityBits(report.compatibility_mask);
    if (!domains) { return RestartCompatibility::collective_protocol; }
    const bool identity =
        report.participating_ranks == resources_.rank_count &&
        report.identity_consistent && report.failure_consistent &&
        detail::SameRestoreContext(report.context, local_context);
    const bool truth =
        report.any_failure ==
            (report.resolved_reason != FailureReason::none) &&
        report.any_failure ==
            (report.resolved_compatibility !=
             RestartCompatibility::compatible) &&
        report.any_failure == (report.failure_mask != 0) &&
        report.any_failure == (report.compatibility_mask != 0) &&
        (local_reason == FailureReason::none ||
         (report.failure_mask & detail::FailureBit(local_reason)) != 0) &&
        (local_compatibility == RestartCompatibility::compatible ||
         (report.compatibility_mask &
          detail::CompatibilityBit(local_compatibility)) != 0) &&
        (report.resolved_reason == FailureReason::none ||
         (report.failure_mask &
          detail::FailureBit(report.resolved_reason)) != 0) &&
        (report.resolved_compatibility ==
             RestartCompatibility::compatible ||
         (report.compatibility_mask &
          detail::CompatibilityBit(report.resolved_compatibility)) != 0) &&
        (!report.any_failure ||
         report.resolved_reason == FailureReason::restart_incompatible);
    if (!identity || !truth)
    {
      return RestartCompatibility::collective_protocol;
    }
    return report.any_failure ? report.resolved_compatibility
                              : RestartCompatibility::compatible;
  }

  template <DerivedProductKind Kind>
  FailureReason ValidateToken(ProductToken<Kind>& token,
                              const AttemptIdentity& attempt) const noexcept
  {
    constexpr std::size_t index = detail::ProductIndex(Kind);
    if (!token.valid_ || token.incarnation_ != incarnation_)
    {
      return FailureReason::invalid_product;
    }
    if (!detail::SameAttempt(token.attempt_, attempt))
    {
      return FailureReason::stale_product;
    }
    if (token.serial_ == 0 ||
        token.serial_ != issued_product_serials_[index])
    {
      return FailureReason::invalid_product;
    }
    return FailureReason::none;
  }

  FailureReason ValidateAndConsumeProducts(const AttemptIdentity& attempt,
                                            ProductBundle& products) noexcept
  {
    if (!products.residual || !products.matrix || !products.factor ||
        !products.solution)
    {
      InvalidateProducts(products);
      return FailureReason::missing_product;
    }
    FailureReason reason = ValidateToken(*products.residual, attempt);
    if (reason == FailureReason::none)
    {
      reason = ValidateToken(*products.matrix, attempt);
    }
    if (reason == FailureReason::none)
    {
      reason = ValidateToken(*products.factor, attempt);
    }
    if (reason == FailureReason::none)
    {
      reason = ValidateToken(*products.solution, attempt);
    }
    if (reason == FailureReason::none &&
        (issued_product_mask_ != required_product_mask ||
         issued_product_count_ != 4))
    {
      reason = FailureReason::missing_product;
    }
    InvalidateProducts(products);
    return reason;
  }

  static void InvalidateProducts(ProductBundle& products) noexcept
  {
    if (products.residual) { products.residual->valid_ = false; }
    if (products.matrix) { products.matrix->valid_ = false; }
    if (products.factor) { products.factor->valid_ = false; }
    if (products.solution) { products.solution->valid_ = false; }
  }

  AttemptOutcome Rollback(const AttemptIdentity& attempt,
                          FailureReason reason,
                          const TransferAccounting& transfers,
                          const bool account_transfers) noexcept
  {
    if (account_transfers)
    {
      detail::AddTransfersExact(telemetry_.rolled_back_transfers, transfers);
    }
    telemetry_.attempts_failed =
        detail::SaturatingAdd(telemetry_.attempts_failed, 1);
    telemetry_.rollbacks = detail::SaturatingAdd(telemetry_.rollbacks, 1);
    if (StoragePolicy::Disjoint(committed_, candidate_))
    {
      StoragePolicy::DiscardCandidate(candidate_);
    }
    else
    {
      storage_valid_ = false;
    }
    active_ = false;
    phase_ = AttemptPhase::idle;
    candidate_storage_failure_ = FailureReason::none;
    ClearPreparedCommit();
    if (reason == FailureReason::none) { reason = FailureReason::backend_failure; }
    last_failure_ = reason;
    telemetry_.last_attempt.failure_reason = reason;
    if (attempt.retry_index == 0 && storage_valid_)
    {
      status_ = RunStatus::retry_available;
      return {status_, reason, false, true};
    }
    status_ = RunStatus::terminal_failure;
    return {status_, reason, false, false};
  }

  void RememberFirstFailure(const FailureReason reason) noexcept
  {
    const FailureReason named_reason = detail::IsNamedFailureReason(reason)
        ? reason
        : FailureReason::collective_protocol;
    if (named_reason != FailureReason::none &&
        telemetry_.first_failure == FailureReason::none)
    {
      telemetry_.first_failure = named_reason;
    }
  }

  void DiscardPrivateCandidate() noexcept
  {
    if (StoragePolicy::Disjoint(committed_, candidate_))
    {
      StoragePolicy::DiscardCandidate(candidate_);
    }
    else
    {
      storage_valid_ = false;
    }
  }

  RestartCompatibility ValidateCheckpoint(
      const CheckpointDescriptor& checkpoint,
      const StateHash& loaded_interior_hash,
      const StateHash& loaded_trace_hash) const noexcept
  {
    if (!detail::IsNamedFailureReason(constructor_failure_) ||
        constructor_failure_ != FailureReason::none || incarnation_ == 0 ||
        !storage_valid_ || !resources_.Valid() ||
        !Selection::Descriptor().Valid() ||
        !StoragePolicy::Disjoint(committed_, candidate_))
    {
      return RestartCompatibility::policy_mismatch;
    }
    if (active_) { return RestartCompatibility::contract_busy; }
    if (!checkpoint.successful_commit)
    {
      return RestartCompatibility::no_successful_commit;
    }
    if (checkpoint.schema_version != checkpoint_schema_version)
    {
      return RestartCompatibility::schema_mismatch;
    }
    if (checkpoint.restart_mode != RestartMode::same_backend_rank_partition)
    {
      return RestartCompatibility::policy_mismatch;
    }
    if (!std::isfinite(checkpoint.coordinates.time))
    {
      return RestartCompatibility::time_mismatch;
    }
    if (detail::CoordinateFingerprint(checkpoint.coordinates) !=
        checkpoint.coordinate_fingerprint)
    {
      return RestartCompatibility::coordinate_mismatch;
    }
    if (checkpoint.cursor != checkpoint.coordinates.cursor)
    {
      return RestartCompatibility::cursor_mismatch;
    }
    if (!std::isfinite(checkpoint.dt) || checkpoint.dt <= 0.0 ||
        !std::isfinite(checkpoint.coordinates.dt) ||
        checkpoint.coordinates.dt <= 0.0 ||
        detail::DoubleBits(checkpoint.dt) !=
            detail::DoubleBits(checkpoint.coordinates.dt))
    {
      return RestartCompatibility::dt_mismatch;
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
    if (checkpoint.topology_hash != resources_.topology_hash)
    {
      return RestartCompatibility::topology_mismatch;
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
    if (loaded_interior_hash != checkpoint.interior_hash)
    {
      return RestartCompatibility::interior_hash_mismatch;
    }
    if (loaded_trace_hash != checkpoint.trace_hash)
    {
      return RestartCompatibility::trace_hash_mismatch;
    }
    return RestartCompatibility::compatible;
  }

  state_slot committed_{};
  state_slot candidate_{};
  StateCoordinates coordinates_{};
  PersistentResourceDescriptor resources_{};
  RunTelemetry telemetry_{};
  StateCoordinates prepared_target_{};
  AttemptTelemetry prepared_telemetry_{};
  TransferAccounting prepared_transfers_{};
  CheckpointDescriptor prepared_checkpoint_{};
  RunStatus status_ = RunStatus::ready;
  AttemptPhase phase_ = AttemptPhase::idle;
  FailureReason last_failure_ = FailureReason::none;
  FailureReason constructor_failure_ = FailureReason::none;
  FailureReason candidate_storage_failure_ = FailureReason::none;
  AttemptIdentity active_attempt_{};
  std::array<std::uint64_t, 4> issued_product_serials_{};
  std::uint64_t attempt_sequence_ = 0;
  std::uint64_t product_serial_ = 0;
  std::uint64_t incarnation_ = 0;
  std::uint64_t candidate_serial_ = 0;
  std::uint64_t publication_serial_ = 1;
  double pending_dt_ = 0.0;
  double pending_target_time_ = 0.0;
  std::uint8_t issued_product_mask_ = 0;
  std::uint8_t issued_product_count_ = 0;
  bool active_ = false;
  bool retry_used_ = false;
  bool storage_valid_ = true;
};

} // namespace dg::hdg::runtime

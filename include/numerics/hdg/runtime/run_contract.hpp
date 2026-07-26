#pragma once

/**
 * @file numerics/hdg/runtime/run_contract.hpp
 * @brief Static, storage-agnostic persistent HDG run lifecycle contract.
 *
 * Backend, formulation, trace, stabilization, time, and precision choices are
 * compile-time tags. The contract owns two disjoint complete state slots,
 * publishes only committed state, and makes every candidate operation and
 * derived product attempt-bound. Backend policies provide only static,
 * non-polymorphic storage and collective operations.
 */

#include <array>
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
  coordinate_overflow,
  candidate_preparation,
  shared_backing_store,
  stale_attempt,
  collective_identity,
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
  schema_mismatch,
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
  policy_mismatch
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

constexpr bool operator==(const AttemptCollectiveContext& left,
                          const AttemptCollectiveContext& right) noexcept
{
  return left.attempt == right.attempt && left.committed == right.committed &&
      left.target == right.target &&
      left.candidate_cursor == right.candidate_cursor;
}

constexpr bool operator!=(const AttemptCollectiveContext& left,
                          const AttemptCollectiveContext& right) noexcept
{
  return !(left == right);
}

struct CollectiveAdmissionReport
{
  AttemptCollectiveContext context{};
  std::uint32_t participating_ranks = 0;
  bool all_prepared = false;
};

struct CollectiveGateReport
{
  AttemptCollectiveContext context{};
  GateReport gates{};
  std::uint32_t participating_ranks = 0;
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
  std::uint64_t admissions_started = 0;
  std::uint64_t admissions_failed = 0;
  std::uint64_t attempts_started = 0;
  std::uint64_t attempts_failed = 0;
  std::uint64_t rollbacks = 0;
  std::uint64_t retries_started = 0;
  std::uint64_t commits = 0;
  std::uint64_t stale_consumers_rejected = 0;
  std::uint64_t duplicate_product_issues_rejected = 0;
  TransferAccounting attempted_transfers{0, 0, 0, 0, 0, true};
  TransferAccounting rolled_back_transfers{0, 0, 0, 0, 0, true};
  TransferAccounting committed_transfers{0, 0, 0, 0, 0, true};
  AttemptTelemetry last_attempt{};
  AcceptedTelemetry accepted{};
  FailureReason first_originating_failure = FailureReason::none;
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

inline void AddTransfers(TransferAccounting& total,
                         const TransferAccounting& value) noexcept
{
  const bool exact = CanAddTransfersExactly(total, value);
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
  total.complete = total.complete && value.complete && exact;
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
  std::uint64_t time_bits = 0;
  std::uint64_t dt_bits = 0;
  std::memcpy(&time_bits, &coordinates.time, sizeof(time_bits));
  std::memcpy(&dt_bits, &coordinates.dt, sizeof(dt_bits));
  const std::array<std::uint64_t, 5> fields{{time_bits,
                                             coordinates.step,
                                             coordinates.generation,
                                             coordinates.cursor,
                                             dt_bits}};
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

} // namespace detail

/**
 * Persistent two-slot lifecycle.
 *
 * StoragePolicy supplies static PrepareCandidate, DiscardCandidate, Commit,
 * Disjoint, HashInterior, and HashTrace operations. CollectivePolicy supplies
 * static Admit and Reduce operations. Both policies are non-polymorphic.
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
  using prepare_result = decltype(StoragePolicy::PrepareCandidate(
      std::declval<const state_slot&>(), std::declval<state_slot&>()));
  using discard_result = decltype(
      StoragePolicy::DiscardCandidate(std::declval<state_slot&>()));
  using commit_result = decltype(StoragePolicy::Commit(
      std::declval<state_slot&>(), std::declval<state_slot&>()));
  using disjoint_result = decltype(StoragePolicy::Disjoint(
      std::declval<const state_slot&>(), std::declval<const state_slot&>()));
  using interior_hash_result = decltype(
      StoragePolicy::HashInterior(std::declval<const InteriorState&>()));
  using trace_hash_result = decltype(
      StoragePolicy::HashTrace(std::declval<const TraceState&>()));
  using admission_result = decltype(CollectivePolicy::Admit(
      std::declval<const AttemptCollectiveContext&>(), bool{},
      std::declval<std::uint32_t>()));
  using collective_result = decltype(CollectivePolicy::Reduce(
      std::declval<const AttemptCollectiveContext&>(),
      std::declval<const GateReport&>(), std::declval<std::uint32_t>()));

  static_assert(!std::is_polymorphic<StoragePolicy>::value,
                "HDG run storage policy must be non-polymorphic");
  static_assert(!std::is_polymorphic<CollectivePolicy>::value,
                "HDG run collective policy must be non-polymorphic");
  static_assert(StoragePolicy::commit_is_atomic,
                "storage policy must declare atomic commit");
  static_assert(StoragePolicy::commit_preserves_disjoint_backing,
                "storage policy must declare disjoint commit backing");
  static_assert(std::is_same<prepare_result, bool>::value,
                "PrepareCandidate must return bool");
  static_assert(std::is_same<discard_result, void>::value,
                "DiscardCandidate must return void");
  static_assert(std::is_same<commit_result, void>::value,
                "Commit must return void");
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
  static_assert(noexcept(StoragePolicy::PrepareCandidate(
                    std::declval<const state_slot&>(),
                    std::declval<state_slot&>())),
                "PrepareCandidate must be noexcept");
  static_assert(noexcept(StoragePolicy::DiscardCandidate(
                    std::declval<state_slot&>())),
                "DiscardCandidate must be noexcept");
  static_assert(noexcept(StoragePolicy::Commit(
                    std::declval<state_slot&>(),
                    std::declval<state_slot&>())),
                "Commit must be noexcept");
  static_assert(noexcept(StoragePolicy::Disjoint(
                    std::declval<const state_slot&>(),
                    std::declval<const state_slot&>())),
                "Disjoint must be noexcept");
  static_assert(noexcept(CollectivePolicy::Admit(
                    std::declval<const AttemptCollectiveContext&>(), bool{},
                    std::declval<std::uint32_t>())),
                "collective Admit must be noexcept");
  static_assert(noexcept(CollectivePolicy::Reduce(
                    std::declval<const AttemptCollectiveContext&>(),
                    std::declval<const GateReport&>(),
                    std::declval<std::uint32_t>())),
                "collective Reduce must be noexcept");
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
        : owner_(other.owner_),
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
        owner_ = other.owner_;
        attempt_ = other.attempt_;
        serial_ = other.serial_;
        valid_ = other.valid_;
        other.valid_ = false;
      }
      return *this;
    }

  private:
    ProductToken(const std::uintptr_t owner,
                 const AttemptIdentity attempt,
                 const std::uint64_t serial) noexcept
        : owner_(owner), attempt_(attempt), serial_(serial), valid_(true)
    {}

    std::uintptr_t owner_ = 0;
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

  class CandidateHandle
  {
  public:
    template <typename Function>
    bool Mutate(Function&& function)
    {
      return owner_ != nullptr &&
          owner_->WithCandidate(attempt_, std::forward<Function>(function));
    }

    bool Copy(state_slot& copy) const
    {
      return owner_ != nullptr && owner_->CopyCandidate(attempt_, copy);
    }

    AttemptIdentity attempt() const noexcept { return attempt_; }

  private:
    CandidateHandle(PersistentRunContract* owner,
                    const AttemptIdentity attempt) noexcept
        : owner_(owner), attempt_(attempt)
    {}

    PersistentRunContract* owner_ = nullptr;
    AttemptIdentity attempt_{};
    friend class PersistentRunContract;
  };

  class PublishedHandle
  {
  public:
    template <typename Function>
    bool Inspect(Function&& function) const
    {
      return owner_ != nullptr && owner_->WithPublished(
          coordinates_, std::forward<Function>(function));
    }

    bool Copy(state_slot& copy) const
    {
      return owner_ != nullptr && owner_->CopyPublished(coordinates_, copy);
    }

    StateCoordinates coordinates() const noexcept { return coordinates_; }

  private:
    PublishedHandle(const PersistentRunContract* owner,
                    const StateCoordinates coordinates) noexcept
        : owner_(owner), coordinates_(coordinates)
    {}

    const PersistentRunContract* owner_ = nullptr;
    StateCoordinates coordinates_{};
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
        attempt_sequence_(initial_attempt_sequence)
  {
    if (!resources_.Valid())
    {
      status_ = RunStatus::invalid_request;
      last_failure_ = FailureReason::invalid_resource_descriptor;
      storage_valid_ = false;
    }
    else if (!StoragePolicy::Disjoint(committed_, candidate_))
    {
      status_ = RunStatus::invalid_request;
      last_failure_ = FailureReason::shared_backing_store;
      storage_valid_ = false;
    }
  }

  PersistentRunContract(const PersistentRunContract&) = delete;
  PersistentRunContract& operator=(const PersistentRunContract&) = delete;
  PersistentRunContract(PersistentRunContract&&) = delete;
  PersistentRunContract& operator=(PersistentRunContract&&) = delete;

  RunStatus status() const noexcept { return status_; }
  FailureReason last_failure() const noexcept { return last_failure_; }
  const StateCoordinates& coordinates() const noexcept { return coordinates_; }
  const PersistentResourceDescriptor& resources() const noexcept
  {
    return resources_;
  }
  const RunTelemetry& telemetry() const noexcept { return telemetry_; }

  PublishedHandle Old() const noexcept { return {this, coordinates_}; }
  PublishedHandle Published() const noexcept { return Old(); }

  std::optional<CandidateHandle> AcquireCandidate(
      const AttemptIdentity& attempt) noexcept
  {
    if (!IsCurrentAttempt(attempt)) { return std::nullopt; }
    return CandidateHandle{this, attempt};
  }

  bool IsCurrentAttempt(const AttemptIdentity& attempt) const noexcept
  {
    return active_ && attempt == active_attempt_;
  }

  template <DerivedProductKind Kind>
  std::optional<ProductToken<Kind>> IssueProduct(
      const AttemptIdentity& attempt) noexcept
  {
    if (!IsCurrentAttempt(attempt))
    {
      telemetry_.stale_consumers_rejected = detail::SaturatingAdd(
          telemetry_.stale_consumers_rejected, 1);
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
    if (product_serial_ == std::numeric_limits<std::uint64_t>::max())
    {
      return std::nullopt;
    }
    issued_product_mask_ = static_cast<std::uint8_t>(issued_product_mask_ | bit);
    issued_product_serials_[index] = ++product_serial_;
    return ProductToken<Kind>{OwnerIdentity(), attempt,
                              issued_product_serials_[index]};
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
    if (!storage_valid_ || !resources_.Valid())
    {
      status_ = RunStatus::invalid_request;
      last_failure_ = storage_valid_
          ? FailureReason::invalid_resource_descriptor
          : last_failure_;
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

    pending_dt_ = request.dt;
    pending_target_time_ = target_time;
    retry_used_ = false;
    const AttemptIdentity proposed{attempt_sequence_ + 1,
                                   coordinates_.step,
                                   coordinates_.generation,
                                   0,
                                   pending_dt_};
    return PrepareAndAdmit(proposed, false);
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
    const AttemptIdentity proposed{attempt_sequence_ + 1,
                                   coordinates_.step,
                                   coordinates_.generation,
                                   1,
                                   pending_dt_};
    return PrepareAndAdmit(proposed, true);
  }

  AttemptOutcome FinishAttempt(const AttemptIdentity& attempt,
                               const AttemptReport& report,
                               ProductBundle&& products) noexcept
  {
    if (!IsCurrentAttempt(attempt))
    {
      telemetry_.stale_consumers_rejected = detail::SaturatingAdd(
          telemetry_.stale_consumers_rejected, 1);
      return {status_,
              FailureReason::stale_attempt,
              false,
              status_ == RunStatus::retry_available};
    }

    if (report.originating_failure != FailureReason::none &&
        telemetry_.first_originating_failure == FailureReason::none)
    {
      telemetry_.first_originating_failure = report.originating_failure;
    }

    const FailureReason product_reason = ValidateAndConsumeProducts(
        attempt, products);
    GateReport effective_gates = report.gates;
    const bool exact_transfers =
        detail::CanAddTransfersExactly(telemetry_.attempted_transfers,
                                       report.telemetry.transfers) &&
        detail::CanAddTransfersExactly(telemetry_.rolled_back_transfers,
                                       report.telemetry.transfers) &&
        detail::CanAddTransfersExactly(telemetry_.committed_transfers,
                                       report.telemetry.transfers);
    if (!report.telemetry.transfers.complete || !exact_transfers ||
        !report.telemetry.prohibited.Clean() ||
        (resources_.reuse.per_step_allocation_required == false &&
         report.telemetry.transfers.allocation_events != 0))
    {
      effective_gates.telemetry = false;
    }
    if (!std::isfinite(report.telemetry.true_relative_residual) ||
        report.telemetry.true_relative_residual >
            true_relative_residual_limit)
    {
      effective_gates.true_residual = false;
    }
    if (report.originating_failure != FailureReason::none ||
        product_reason != FailureReason::none)
    {
      effective_gates.success = false;
    }

    const StateCoordinates target{pending_target_time_,
                                  coordinates_.step + 1,
                                  coordinates_.generation + 1,
                                  report.candidate_cursor,
                                  pending_dt_};
    const AttemptCollectiveContext local_context{
        attempt, coordinates_, target, report.candidate_cursor};
    const CollectiveGateReport collective = CollectivePolicy::Reduce(
        local_context, effective_gates, resources_.rank_count);
    const bool collective_identity =
        collective.participating_ranks == resources_.rank_count &&
        collective.context == local_context;
    const GateReport collective_gates{
        effective_gates.finite && collective.gates.finite,
        effective_gates.admissible && collective.gates.admissible,
        effective_gates.recovery && collective.gates.recovery,
        effective_gates.true_residual && collective.gates.true_residual,
        effective_gates.telemetry && collective.gates.telemetry,
        effective_gates.success && collective.gates.success};

    FailureReason reason = FailureReason::none;
    if (report.originating_failure != FailureReason::none)
    {
      reason = report.originating_failure;
    }
    else if (product_reason != FailureReason::none)
    {
      reason = product_reason;
    }
    else if (!collective_identity)
    {
      reason = FailureReason::collective_identity;
    }
    else
    {
      reason = detail::FirstFailedGate(collective_gates);
    }

    AttemptTelemetry attempt_telemetry = report.telemetry;
    attempt_telemetry.identity = attempt;
    attempt_telemetry.failure_reason = reason;
    attempt_telemetry.validated_product_mask =
        product_reason == FailureReason::none ? required_product_mask : 0;
    telemetry_.last_attempt = attempt_telemetry;
    detail::AddTransfers(telemetry_.attempted_transfers,
                         report.telemetry.transfers);

    if (reason != FailureReason::none || !collective_gates.All())
    {
      return Rollback(attempt, reason, report.telemetry.transfers);
    }

    if (!StoragePolicy::Disjoint(committed_, candidate_))
    {
      return FailInvalidStorage(FailureReason::shared_backing_store);
    }
    StoragePolicy::Commit(committed_, candidate_);
    if (!StoragePolicy::Disjoint(committed_, candidate_))
    {
      storage_valid_ = false;
      return FailInvalidStorage(FailureReason::shared_backing_store);
    }

    coordinates_ = target;
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
    if (active_ || !storage_valid_ || !telemetry_.accepted.valid ||
        telemetry_.accepted.coordinates != coordinates_)
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

  RestartCompatibility CheckRestart(
      const CheckpointDescriptor& checkpoint,
      const InteriorState& loaded_interior,
      const TraceState& loaded_trace) const noexcept
  {
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
        checkpoint.dt != checkpoint.coordinates.dt)
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
    if (StoragePolicy::HashInterior(loaded_interior) !=
        checkpoint.interior_hash)
    {
      return RestartCompatibility::interior_hash_mismatch;
    }
    if (StoragePolicy::HashTrace(loaded_trace) != checkpoint.trace_hash)
    {
      return RestartCompatibility::trace_hash_mismatch;
    }
    return RestartCompatibility::compatible;
  }

private:
  static constexpr std::uint8_t required_product_mask = 0x0fU;

  std::uintptr_t OwnerIdentity() const noexcept
  {
    return reinterpret_cast<std::uintptr_t>(this);
  }

  template <typename Function>
  bool WithCandidate(const AttemptIdentity& attempt, Function&& function)
  {
    if (!IsCurrentAttempt(attempt))
    {
      telemetry_.stale_consumers_rejected = detail::SaturatingAdd(
          telemetry_.stale_consumers_rejected, 1);
      return false;
    }
    std::forward<Function>(function)(candidate_);
    if (!StoragePolicy::Disjoint(committed_, candidate_))
    {
      storage_valid_ = false;
      active_ = false;
      status_ = RunStatus::terminal_failure;
      last_failure_ = FailureReason::shared_backing_store;
      return false;
    }
    return IsCurrentAttempt(attempt);
  }

  bool CopyCandidate(const AttemptIdentity& attempt, state_slot& copy) const
  {
    if (!IsCurrentAttempt(attempt)) { return false; }
    copy = candidate_;
    return IsCurrentAttempt(attempt);
  }

  template <typename Function>
  bool WithPublished(const StateCoordinates& coordinates,
                     Function&& function) const
  {
    if (coordinates != coordinates_) { return false; }
    std::forward<Function>(function)(committed_, coordinates_);
    return coordinates == coordinates_;
  }

  bool CopyPublished(const StateCoordinates& coordinates,
                     state_slot& copy) const
  {
    if (coordinates != coordinates_) { return false; }
    copy = committed_;
    return coordinates == coordinates_;
  }

  StateCoordinates AdmissionTarget() const noexcept
  {
    return {pending_target_time_,
            coordinates_.step + 1,
            coordinates_.generation + 1,
            coordinates_.cursor,
            pending_dt_};
  }

  BeginOutcome PrepareAndAdmit(const AttemptIdentity& proposed,
                               const bool retry) noexcept
  {
    telemetry_.admissions_started = detail::SaturatingAdd(
        telemetry_.admissions_started, 1);
    attempt_sequence_ = proposed.sequence;
    const bool disjoint_before =
        StoragePolicy::Disjoint(committed_, candidate_);
    bool local_prepared = false;
    if (disjoint_before)
    {
      local_prepared = StoragePolicy::PrepareCandidate(committed_, candidate_);
    }
    const bool disjoint_after =
        StoragePolicy::Disjoint(committed_, candidate_);
    local_prepared = local_prepared && disjoint_after;

    const AttemptCollectiveContext local_context{
        proposed, coordinates_, AdmissionTarget(), coordinates_.cursor};
    const CollectiveAdmissionReport admission = CollectivePolicy::Admit(
        local_context, local_prepared, resources_.rank_count);
    const bool collective_identity =
        admission.participating_ranks == resources_.rank_count &&
        admission.context == local_context;
    if (!local_prepared || !admission.all_prepared || !collective_identity)
    {
      if (disjoint_after) { StoragePolicy::DiscardCandidate(candidate_); }
      telemetry_.admissions_failed = detail::SaturatingAdd(
          telemetry_.admissions_failed, 1);
      active_ = false;
      status_ = RunStatus::terminal_failure;
      if (!disjoint_before || !disjoint_after)
      {
        storage_valid_ = false;
        last_failure_ = FailureReason::shared_backing_store;
      }
      else if (!local_prepared || !admission.all_prepared)
      {
        last_failure_ = FailureReason::candidate_preparation;
      }
      else
      {
        last_failure_ = FailureReason::collective_identity;
      }
      return {false, proposed, last_failure_};
    }

    retry_used_ = retry;
    active_attempt_ = proposed;
    issued_product_mask_ = 0;
    issued_product_serials_.fill(0);
    active_ = true;
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

  template <DerivedProductKind Kind>
  FailureReason ValidateToken(ProductToken<Kind>& token,
                              const AttemptIdentity& attempt) const noexcept
  {
    constexpr std::size_t index = detail::ProductIndex(Kind);
    if (token.attempt_ != attempt) { return FailureReason::stale_product; }
    if (!token.valid_ || token.owner_ != OwnerIdentity() ||
        token.serial_ == 0 ||
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
        issued_product_mask_ != required_product_mask)
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
                          const TransferAccounting& transfers) noexcept
  {
    detail::AddTransfers(telemetry_.rolled_back_transfers, transfers);
    telemetry_.attempts_failed =
        detail::SaturatingAdd(telemetry_.attempts_failed, 1);
    telemetry_.rollbacks = detail::SaturatingAdd(telemetry_.rollbacks, 1);
    StoragePolicy::DiscardCandidate(candidate_);
    active_ = false;
    if (reason == FailureReason::none) { reason = FailureReason::backend_failure; }
    last_failure_ = reason;
    telemetry_.last_attempt.failure_reason = reason;
    if (attempt.retry_index == 0)
    {
      status_ = RunStatus::retry_available;
      return {status_, last_failure_, false, true};
    }
    status_ = RunStatus::terminal_failure;
    return {status_, last_failure_, false, false};
  }

  AttemptOutcome FailInvalidStorage(const FailureReason reason) noexcept
  {
    active_ = false;
    status_ = RunStatus::terminal_failure;
    last_failure_ = reason;
    telemetry_.last_attempt.failure_reason = reason;
    telemetry_.attempts_failed =
        detail::SaturatingAdd(telemetry_.attempts_failed, 1);
    telemetry_.rollbacks = detail::SaturatingAdd(telemetry_.rollbacks, 1);
    return {status_, reason, false, false};
  }

  state_slot committed_{};
  state_slot candidate_{};
  StateCoordinates coordinates_{};
  PersistentResourceDescriptor resources_{};
  RunTelemetry telemetry_{};
  RunStatus status_ = RunStatus::ready;
  FailureReason last_failure_ = FailureReason::none;
  AttemptIdentity active_attempt_{};
  std::array<std::uint64_t, 4> issued_product_serials_{};
  std::uint64_t attempt_sequence_ = 0;
  std::uint64_t product_serial_ = 0;
  double pending_dt_ = 0.0;
  double pending_target_time_ = 0.0;
  std::uint8_t issued_product_mask_ = 0;
  bool active_ = false;
  bool retry_used_ = false;
  bool storage_valid_ = true;
};

} // namespace dg::hdg::runtime

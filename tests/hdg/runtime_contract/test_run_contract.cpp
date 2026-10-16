#include "numerics/hdg/runtime/run_contract.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <new>
#include <string_view>
#include <type_traits>
#include <utility>

namespace
{

using namespace dg::hdg::runtime;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "runtime contract must compile in strict C++17 mode");

struct EulerModelTag
{
  static constexpr std::string_view id = "euler.compressible.2d";
};
struct ConventionalFormulationTag
{
  static constexpr std::string_view id = "hdg.conventional";
};
struct DgInterfaceTraceTag
{
  static constexpr std::string_view id = "mfem.dg_interface.trace";
};
struct FixedTauTag
{
  static constexpr std::string_view id = "tau.fixed";
};
struct BackwardEulerTag
{
  static constexpr std::string_view id = "backward_euler";
};
struct CpuBackendTag
{
  static constexpr std::string_view id = "cpu.serial";
};
struct Fp64Tag
{
  static constexpr std::string_view id = "fp64";
};

using Selection = StaticRunSelection<EulerModelTag,
                                     ConventionalFormulationTag,
                                     DgInterfaceTraceTag,
                                     FixedTauTag,
                                     BackwardEulerTag,
                                     CpuBackendTag,
                                     Fp64Tag>;
using Interior = std::array<double, 4>;
using Trace = std::array<double, 3>;
using Slot = CompleteStateSlot<Interior, Trace>;

StateHash Digest(const std::uint8_t seed) noexcept
{
  StateHash hash{};
  for (std::size_t i = 0; i < hash.bytes.size(); ++i)
  {
    hash.bytes[i] = static_cast<std::uint8_t>(seed + i);
  }
  return hash;
}

template <typename Values>
StateHash ValueHash(const Values& values, const int offset) noexcept
{
  StateHash hash{};
  for (std::size_t i = 0; i < hash.bytes.size(); ++i)
  {
    hash.bytes[i] = static_cast<std::uint8_t>(
        static_cast<int>(values[i % values.size()] * 3.0) +
        static_cast<int>(i) + offset);
  }
  return hash;
}

struct ArrayStoragePolicy
{
  static constexpr bool preparation_preserves_committed = true;
  static constexpr bool private_copy_is_independent = true;
  static constexpr bool publish_is_atomic = true;
  static constexpr bool publish_preserves_disjoint_backing = true;
  static inline std::uint64_t preparations = 0;
  static inline std::uint64_t restore_preparations = 0;
  static inline std::uint64_t stores = 0;
  static inline std::uint64_t copies = 0;
  static inline std::uint64_t discards = 0;
  static inline std::uint64_t prepublish_checks = 0;
  static inline std::uint64_t publications = 0;
  static inline std::uint64_t fail_prepare_call = 0;
  static inline std::uint64_t fail_restore_call = 0;
  static inline std::uint64_t fail_store_call = 0;
  static inline std::uint64_t fail_prepublish_call = 0;

  static void Reset() noexcept
  {
    preparations = 0;
    restore_preparations = 0;
    stores = 0;
    copies = 0;
    discards = 0;
    prepublish_checks = 0;
    publications = 0;
    fail_prepare_call = 0;
    fail_restore_call = 0;
    fail_store_call = 0;
    fail_prepublish_call = 0;
  }

  static bool Disjoint(const Slot& left, const Slot& right) noexcept
  {
    return left.interior.data() != right.interior.data() &&
        left.trace.data() != right.trace.data();
  }

  static bool PrepareCandidate(const Slot& committed,
                               Slot& candidate) noexcept
  {
    ++preparations;
    if (preparations == fail_prepare_call) { return false; }
    candidate = committed;
    return true;
  }

  static bool PrepareRestore(const Slot& loaded, Slot& candidate) noexcept
  {
    ++restore_preparations;
    if (restore_preparations == fail_restore_call) { return false; }
    candidate = loaded;
    return true;
  }

  static bool StoreCandidate(Slot& candidate, const Slot& value) noexcept
  {
    ++stores;
    if (stores == fail_store_call) { return false; }
    candidate = value;
    return true;
  }

  static bool CopyOut(const Slot& source, Slot& copy) noexcept
  {
    ++copies;
    copy = source;
    return true;
  }

  static void DiscardCandidate(Slot& candidate) noexcept
  {
    ++discards;
    candidate.interior.fill(-900.0);
    candidate.trace.fill(-901.0);
  }

  static bool CanPublish(const Slot& committed,
                         const Slot& candidate) noexcept
  {
    ++prepublish_checks;
    return prepublish_checks != fail_prepublish_call &&
        Disjoint(committed, candidate);
  }

  static void Publish(Slot& committed, Slot& candidate) noexcept
  {
    ++publications;
    using std::swap;
    swap(committed, candidate);
  }

  static StateHash HashInterior(const Interior& values) noexcept
  {
    return ValueHash(values, 0);
  }

  static StateHash HashTrace(const Trace& values) noexcept
  {
    return ValueHash(values, 100);
  }
};

enum class CollectiveFault : unsigned char
{
  none,
  admission_identity_false,
  admission_failure_consistency_false,
  admission_participants,
  admission_context,
  admission_remote_failure,
  admission_hide_local_failure,
  admission_change_local_failure,
  admission_invalid_resolved_reason,
  admission_unknown_failure_mask,
  finish_identity_false,
  finish_failure_consistency_false,
  finish_participants,
  finish_attempt,
  finish_committed_time,
  finish_committed_step,
  finish_committed_generation,
  finish_committed_cursor,
  finish_committed_dt,
  finish_target_time,
  finish_target_step,
  finish_target_generation,
  finish_target_cursor,
  finish_target_dt,
  finish_candidate_cursor,
  finish_remote_failure,
  finish_hide_local_failure,
  finish_change_local_failure,
  finish_invalid_resolved_reason,
  finish_unknown_failure_mask,
  restore_identity_false,
  restore_failure_consistency_false,
  restore_participants,
  restore_context,
  restore_remote_failure,
  restore_hide_local_failure,
  restore_change_local_failure,
  restore_invalid_resolved_reason,
  restore_invalid_resolved_compatibility,
  restore_unknown_failure_mask,
  restore_unknown_compatibility_mask
};

constexpr std::uint64_t FailureMask(const FailureReason reason) noexcept
{
  const unsigned int value = static_cast<unsigned int>(reason);
  return value != 0 &&
          value < static_cast<unsigned int>(FailureReason::count) && value < 64
      ? (std::uint64_t{1} << value)
      : 0;
}

constexpr std::uint64_t CompatibilityMask(
    const RestartCompatibility compatibility) noexcept
{
  const unsigned int value = static_cast<unsigned int>(compatibility);
  return value != 0 &&
          value < static_cast<unsigned int>(RestartCompatibility::count) &&
          value < 64
      ? (std::uint64_t{1} << value)
      : 0;
}

constexpr std::uint64_t unknown_collective_mask_bit =
    std::uint64_t{1} << 63;

void FailGateForReason(GateReport& gates,
                       const FailureReason reason) noexcept
{
  switch (reason)
  {
    case FailureReason::finite_gate: gates.finite = false; break;
    case FailureReason::admissibility_gate: gates.admissible = false; break;
    case FailureReason::recovery_gate: gates.recovery = false; break;
    case FailureReason::true_residual_gate:
      gates.true_residual = false;
      break;
    case FailureReason::telemetry_gate: gates.telemetry = false; break;
    case FailureReason::success_gate: gates.success = false; break;
    default: gates.success = false; break;
  }
}

struct TestCollectivePolicy
{
  static inline CollectiveFault fault = CollectiveFault::none;
  static inline FailureReason remote_reason = FailureReason::backend_failure;
  static inline RestartCompatibility remote_compatibility =
      RestartCompatibility::interior_hash_mismatch;
  static inline FailureReason injected_reason =
      static_cast<FailureReason>(255);
  static inline RestartCompatibility injected_compatibility =
      static_cast<RestartCompatibility>(255);
  static inline std::uint64_t admissions = 0;
  static inline std::uint64_t reductions = 0;
  static inline std::uint64_t restores = 0;

  static void Reset() noexcept
  {
    fault = CollectiveFault::none;
    remote_reason = FailureReason::backend_failure;
    remote_compatibility = RestartCompatibility::interior_hash_mismatch;
    injected_reason = static_cast<FailureReason>(255);
    injected_compatibility = static_cast<RestartCompatibility>(255);
    admissions = 0;
    reductions = 0;
    restores = 0;
  }

  static CollectiveAdmissionReport Admit(
      const AttemptCollectiveContext& local,
      const bool local_prepared,
      const FailureReason local_reason,
      const std::uint32_t expected_ranks) noexcept
  {
    ++admissions;
    CollectiveAdmissionReport report{local,
                                     expected_ranks,
                                     true,
                                     true,
                                     local_prepared,
                                     local_reason != FailureReason::none,
                                     FailureMask(local_reason),
                                     local_reason};
    switch (fault)
    {
      case CollectiveFault::admission_identity_false:
        report.identity_consistent = false;
        break;
      case CollectiveFault::admission_failure_consistency_false:
        report.failure_consistent = false;
        report.all_prepared = false;
        report.any_failure = true;
        report.failure_mask = FailureMask(remote_reason);
        report.resolved_reason = remote_reason;
        break;
      case CollectiveFault::admission_participants:
        ++report.participating_ranks;
        break;
      case CollectiveFault::admission_context:
        ++report.context.target.cursor;
        break;
      case CollectiveFault::admission_remote_failure:
        report.all_prepared = false;
        report.any_failure = true;
        report.failure_mask = FailureMask(remote_reason);
        report.resolved_reason = remote_reason;
        break;
      case CollectiveFault::admission_hide_local_failure:
        report.all_prepared = true;
        report.any_failure = false;
        report.failure_mask = 0;
        report.resolved_reason = FailureReason::none;
        break;
      case CollectiveFault::admission_change_local_failure:
        report.all_prepared = false;
        report.any_failure = true;
        report.failure_mask = FailureMask(remote_reason);
        report.resolved_reason = remote_reason;
        break;
      case CollectiveFault::admission_invalid_resolved_reason:
        report.all_prepared = false;
        report.any_failure = true;
        report.failure_mask = FailureMask(remote_reason);
        report.resolved_reason = injected_reason;
        break;
      case CollectiveFault::admission_unknown_failure_mask:
        report.all_prepared = false;
        report.any_failure = true;
        report.failure_mask =
            FailureMask(remote_reason) | unknown_collective_mask_bit;
        report.resolved_reason = remote_reason;
        break;
      default: break;
    }
    return report;
  }

  static CollectiveGateReport Reduce(
      const AttemptCollectiveContext& local,
      const GateReport& local_gates,
      const FailureReason local_reason,
      const std::uint32_t expected_ranks) noexcept
  {
    ++reductions;
    CollectiveGateReport report{local,
                                local_gates,
                                expected_ranks,
                                true,
                                true,
                                local_reason != FailureReason::none,
                                FailureMask(local_reason),
                                local_reason};
    switch (fault)
    {
      case CollectiveFault::finish_identity_false:
        report.identity_consistent = false;
        break;
      case CollectiveFault::finish_failure_consistency_false:
        report.failure_consistent = false;
        report.any_failure = true;
        report.failure_mask = FailureMask(remote_reason);
        report.resolved_reason = remote_reason;
        FailGateForReason(report.gates, remote_reason);
        break;
      case CollectiveFault::finish_participants:
        ++report.participating_ranks;
        break;
      case CollectiveFault::finish_attempt:
        ++report.context.attempt.sequence;
        break;
      case CollectiveFault::finish_committed_time:
        report.context.committed.time += 1.0;
        break;
      case CollectiveFault::finish_committed_step:
        ++report.context.committed.step;
        break;
      case CollectiveFault::finish_committed_generation:
        ++report.context.committed.generation;
        break;
      case CollectiveFault::finish_committed_cursor:
        ++report.context.committed.cursor;
        break;
      case CollectiveFault::finish_committed_dt:
        report.context.committed.dt += 1.0;
        break;
      case CollectiveFault::finish_target_time:
        report.context.target.time += 1.0;
        break;
      case CollectiveFault::finish_target_step:
        ++report.context.target.step;
        break;
      case CollectiveFault::finish_target_generation:
        ++report.context.target.generation;
        break;
      case CollectiveFault::finish_target_cursor:
        ++report.context.target.cursor;
        break;
      case CollectiveFault::finish_target_dt:
        report.context.target.dt += 1.0;
        break;
      case CollectiveFault::finish_candidate_cursor:
        ++report.context.candidate_cursor;
        break;
      case CollectiveFault::finish_remote_failure:
        report.any_failure = true;
        report.failure_mask = FailureMask(remote_reason);
        report.resolved_reason = remote_reason;
        FailGateForReason(report.gates, remote_reason);
        break;
      case CollectiveFault::finish_hide_local_failure:
        report.gates = {true, true, true, true, true, true};
        report.any_failure = false;
        report.failure_mask = 0;
        report.resolved_reason = FailureReason::none;
        break;
      case CollectiveFault::finish_change_local_failure:
        report.any_failure = true;
        report.failure_mask = FailureMask(remote_reason);
        report.resolved_reason = remote_reason;
        FailGateForReason(report.gates, remote_reason);
        break;
      case CollectiveFault::finish_invalid_resolved_reason:
        report.any_failure = true;
        report.failure_mask = FailureMask(remote_reason);
        report.resolved_reason = injected_reason;
        FailGateForReason(report.gates, remote_reason);
        break;
      case CollectiveFault::finish_unknown_failure_mask:
        report.any_failure = true;
        report.failure_mask =
            FailureMask(remote_reason) | unknown_collective_mask_bit;
        report.resolved_reason = remote_reason;
        FailGateForReason(report.gates, remote_reason);
        break;
      default: break;
    }
    return report;
  }

  static CollectiveRestoreReport ReduceRestore(
      const RestoreCollectiveContext& local,
      const RestartCompatibility local_compatibility,
      const FailureReason local_reason,
      const std::uint32_t expected_ranks) noexcept
  {
    ++restores;
    CollectiveRestoreReport report{
        local,
        expected_ranks,
        true,
        true,
        local_reason != FailureReason::none,
        FailureMask(local_reason),
        CompatibilityMask(local_compatibility),
        local_reason,
        local_compatibility};
    switch (fault)
    {
      case CollectiveFault::restore_identity_false:
        report.identity_consistent = false;
        break;
      case CollectiveFault::restore_failure_consistency_false:
        report.failure_consistent = false;
        report.any_failure = true;
        report.failure_mask = FailureMask(FailureReason::restart_incompatible);
        report.compatibility_mask = CompatibilityMask(remote_compatibility);
        report.resolved_reason = FailureReason::restart_incompatible;
        report.resolved_compatibility = remote_compatibility;
        break;
      case CollectiveFault::restore_participants:
        ++report.participating_ranks;
        break;
      case CollectiveFault::restore_context:
        report.context.loaded_interior_hash.bytes[0] ^= 1U;
        break;
      case CollectiveFault::restore_remote_failure:
        report.any_failure = true;
        report.failure_mask = FailureMask(FailureReason::restart_incompatible);
        report.compatibility_mask = CompatibilityMask(remote_compatibility);
        report.resolved_reason = FailureReason::restart_incompatible;
        report.resolved_compatibility = remote_compatibility;
        break;
      case CollectiveFault::restore_hide_local_failure:
        report.any_failure = false;
        report.failure_mask = 0;
        report.compatibility_mask = 0;
        report.resolved_reason = FailureReason::none;
        report.resolved_compatibility = RestartCompatibility::compatible;
        break;
      case CollectiveFault::restore_change_local_failure:
        report.any_failure = true;
        report.failure_mask = FailureMask(FailureReason::restart_incompatible);
        report.compatibility_mask = CompatibilityMask(remote_compatibility);
        report.resolved_reason = FailureReason::restart_incompatible;
        report.resolved_compatibility = remote_compatibility;
        break;
      case CollectiveFault::restore_invalid_resolved_reason:
        report.any_failure = true;
        report.failure_mask = FailureMask(FailureReason::restart_incompatible);
        report.compatibility_mask = CompatibilityMask(remote_compatibility);
        report.resolved_reason = injected_reason;
        report.resolved_compatibility = remote_compatibility;
        break;
      case CollectiveFault::restore_invalid_resolved_compatibility:
        report.any_failure = true;
        report.failure_mask = FailureMask(FailureReason::restart_incompatible);
        report.compatibility_mask = CompatibilityMask(remote_compatibility);
        report.resolved_reason = FailureReason::restart_incompatible;
        report.resolved_compatibility = injected_compatibility;
        break;
      case CollectiveFault::restore_unknown_failure_mask:
        report.any_failure = true;
        report.failure_mask = FailureMask(FailureReason::restart_incompatible) |
            unknown_collective_mask_bit;
        report.compatibility_mask = CompatibilityMask(remote_compatibility);
        report.resolved_reason = FailureReason::restart_incompatible;
        report.resolved_compatibility = remote_compatibility;
        break;
      case CollectiveFault::restore_unknown_compatibility_mask:
        report.any_failure = true;
        report.failure_mask = FailureMask(FailureReason::restart_incompatible);
        report.compatibility_mask = CompatibilityMask(remote_compatibility) |
            unknown_collective_mask_bit;
        report.resolved_reason = FailureReason::restart_incompatible;
        report.resolved_compatibility = remote_compatibility;
        break;
      default: break;
    }
    return report;
  }
};

using Contract = PersistentRunContract<Selection,
                                       Interior,
                                       Trace,
                                       ArrayStoragePolicy,
                                       TestCollectivePolicy>;

struct TwoPhaseArrayStoragePolicy : ArrayStoragePolicy
{
  static constexpr bool publish_is_complete_slot_exchange = true;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = true;
  static constexpr bool publish_and_revert_preserve_disjoint_backing = true;
  static inline std::uint64_t reversions = 0;
  static inline Slot* live_candidate = nullptr;
  static inline Slot reverted_committed{};
  static inline Slot reverted_candidate{};

  static void Reset() noexcept
  {
    ArrayStoragePolicy::Reset();
    reversions = 0;
    live_candidate = nullptr;
    reverted_committed = {};
    reverted_candidate = {};
  }

  static bool PrepareCandidate(const Slot& committed,
                               Slot& candidate) noexcept
  {
    live_candidate = &candidate;
    return ArrayStoragePolicy::PrepareCandidate(committed, candidate);
  }

  static bool StoreCandidate(Slot& candidate, const Slot& value) noexcept
  {
    live_candidate = &candidate;
    return ArrayStoragePolicy::StoreCandidate(candidate, value);
  }

  static void RevertPublish(Slot& committed, Slot& candidate) noexcept
  {
    ++reversions;
    using std::swap;
    swap(committed, candidate);
    reverted_committed = committed;
    reverted_candidate = candidate;
  }
};

struct FalseTwoPhaseAttestationPolicy : ArrayStoragePolicy
{
  static constexpr bool publish_is_complete_slot_exchange = true;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = false;
  static constexpr bool publish_and_revert_preserve_disjoint_backing = true;
  static void RevertPublish(Slot&, Slot&) noexcept {}
};

template <int FalseIndex>
struct FalseAttestationMatrixPolicy : ArrayStoragePolicy
{
  static constexpr bool publish_is_complete_slot_exchange = FalseIndex != 0;
  static constexpr bool candidate_retains_exact_old_state_after_publish =
      FalseIndex != 1;
  static constexpr bool revert_publish_is_exact_byte_inverse =
      FalseIndex != 2;
  static constexpr bool publish_and_revert_preserve_disjoint_backing =
      FalseIndex != 3;
  static void RevertPublish(Slot&, Slot&) noexcept {}
};

struct MissingRevertPolicy : ArrayStoragePolicy
{
  static constexpr bool publish_is_complete_slot_exchange = true;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = true;
  static constexpr bool publish_and_revert_preserve_disjoint_backing = true;
};

struct MissingAttestationPolicy : ArrayStoragePolicy
{
  static constexpr bool publish_is_complete_slot_exchange = true;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = true;
  static void RevertPublish(Slot&, Slot&) noexcept {}
};

struct ThrowingRevertPolicy : ArrayStoragePolicy
{
  static constexpr bool publish_is_complete_slot_exchange = true;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = true;
  static constexpr bool publish_and_revert_preserve_disjoint_backing = true;
  static void RevertPublish(Slot&, Slot&) {}
};

struct OverloadedRevertPolicy : ArrayStoragePolicy
{
  static constexpr bool publish_is_complete_slot_exchange = true;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = true;
  static constexpr bool publish_and_revert_preserve_disjoint_backing = true;
  static void RevertPublish(Slot&, Slot&) noexcept {}
  static void RevertPublish(Slot&, const Slot&) noexcept {}
};

struct WrongAttestationTypePolicy : ArrayStoragePolicy
{
  static constexpr int publish_is_complete_slot_exchange = 1;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = true;
  static constexpr bool publish_and_revert_preserve_disjoint_backing = true;
  static void RevertPublish(Slot&, Slot&) noexcept {}
};

using TwoPhaseContract = PersistentRunContract<Selection,
                                               Interior,
                                               Trace,
                                               TwoPhaseArrayStoragePolicy,
                                               TestCollectivePolicy>;
using FalseTwoPhaseContract = PersistentRunContract<
    Selection,
    Interior,
    Trace,
    FalseTwoPhaseAttestationPolicy,
    TestCollectivePolicy>;
using ThrowingRevertContract = PersistentRunContract<Selection,
                                                     Interior,
                                                     Trace,
                                                     ThrowingRevertPolicy,
                                                     TestCollectivePolicy>;
using OverloadedRevertContract = PersistentRunContract<Selection,
                                                       Interior,
                                                       Trace,
                                                       OverloadedRevertPolicy,
                                                       TestCollectivePolicy>;
using WrongAttestationContract = PersistentRunContract<
    Selection,
    Interior,
    Trace,
    WrongAttestationTypePolicy,
    TestCollectivePolicy>;
template <int FalseIndex>
using FalseAttestationMatrixContract = PersistentRunContract<
    Selection,
    Interior,
    Trace,
    FalseAttestationMatrixPolicy<FalseIndex>,
    TestCollectivePolicy>;
using MissingRevertContract = PersistentRunContract<Selection,
                                                    Interior,
                                                    Trace,
                                                    MissingRevertPolicy,
                                                    TestCollectivePolicy>;
using MissingAttestationContract = PersistentRunContract<
    Selection,
    Interior,
    Trace,
    MissingAttestationPolicy,
    TestCollectivePolicy>;

template <typename ContractType, typename = void>
struct HasTwoPhaseCommitApi : std::false_type
{};

template <typename ContractType>
struct HasTwoPhaseCommitApi<
    ContractType,
    std::void_t<
        decltype(std::declval<ContractType&>().FreezeCandidate(
            std::declval<const AttemptIdentity&>(), std::uint64_t{})),
        decltype(std::declval<const ContractType&>().CopyPrepared(
            std::declval<const typename ContractType::StagingToken&>(),
            std::declval<typename ContractType::state_slot&>())),
        decltype(std::declval<ContractType&>().PrepareCommit(
            std::declval<typename ContractType::StagingToken&&>(),
            std::declval<const AttemptReport&>(),
            std::declval<typename ContractType::ProductBundle&&>())),
        decltype(std::declval<ContractType&>().CommitPrepared(
            std::declval<typename ContractType::PreparedCommitToken&&>())),
        decltype(std::declval<ContractType&>().FinalizeCommit(
            std::declval<typename ContractType::ProvisionalCommitToken&&>())),
        decltype(std::declval<ContractType&>().AbortPrepared(
            std::declval<typename ContractType::PreparedCommitToken&&>())),
        decltype(std::declval<ContractType&>().RevertCommit(
            std::declval<typename ContractType::ProvisionalCommitToken&&>()))>>
    : std::true_type
{};

template <typename Token, typename SlotType, typename = void>
struct HasTokenSelfCopyPrepared : std::false_type
{};

template <typename Token, typename SlotType>
struct HasTokenSelfCopyPrepared<
    Token,
    SlotType,
    std::void_t<decltype(std::declval<const Token&>().CopyPrepared(
        std::declval<SlotType&>()))>> : std::true_type
{};

static_assert(Selection::interior_space == InteriorSpace::l2);
static_assert(!std::is_polymorphic<Selection>::value);
static_assert(!std::is_polymorphic<Contract>::value);
static_assert(!std::is_polymorphic<ArrayStoragePolicy>::value);
static_assert(!std::is_polymorphic<TestCollectivePolicy>::value);
static_assert(!std::is_copy_constructible<Contract>::value);
static_assert(!std::is_move_constructible<Contract>::value);
static_assert(!std::is_default_constructible<
                  Contract::ProductToken<DerivedProductKind::residual>>::value,
              "attempt products must not be forgeable by construction");
static_assert(!std::is_copy_constructible<
                  Contract::ProductToken<DerivedProductKind::residual>>::value,
              "attempt products must not be duplicable by copying");
static_assert(!std::is_convertible<Contract::CandidateCapability, Slot*>::value);
static_assert(!std::is_convertible<Contract::PublishedCapability, Slot*>::value);
static_assert(HasTwoPhaseCommitApi<TwoPhaseContract>::value);
static_assert(!HasTwoPhaseCommitApi<Contract>::value,
              "legacy storage policies expose no two-phase members");
static_assert(!HasTwoPhaseCommitApi<FalseTwoPhaseContract>::value);
static_assert(!HasTwoPhaseCommitApi<ThrowingRevertContract>::value);
static_assert(!HasTwoPhaseCommitApi<OverloadedRevertContract>::value);
static_assert(!HasTwoPhaseCommitApi<WrongAttestationContract>::value);
static_assert(!HasTwoPhaseCommitApi<FalseAttestationMatrixContract<0>>::value);
static_assert(!HasTwoPhaseCommitApi<FalseAttestationMatrixContract<1>>::value);
static_assert(!HasTwoPhaseCommitApi<FalseAttestationMatrixContract<2>>::value);
static_assert(!HasTwoPhaseCommitApi<FalseAttestationMatrixContract<3>>::value);
static_assert(!HasTwoPhaseCommitApi<MissingRevertContract>::value);
static_assert(!HasTwoPhaseCommitApi<MissingAttestationContract>::value);
static_assert(std::is_same<
                  decltype(std::declval<FalseTwoPhaseContract&>().BeginStep(
                      std::declval<const StepRequest&>())),
                  BeginOutcome>::value);
static_assert(std::is_same<
                  decltype(std::declval<ThrowingRevertContract&>().BeginStep(
                      std::declval<const StepRequest&>())),
                  BeginOutcome>::value);
static_assert(std::is_same<
                  decltype(std::declval<OverloadedRevertContract&>().BeginStep(
                      std::declval<const StepRequest&>())),
                  BeginOutcome>::value);
static_assert(std::is_same<
                  decltype(std::declval<WrongAttestationContract&>().BeginStep(
                      std::declval<const StepRequest&>())),
                  BeginOutcome>::value);
static_assert(detail::SupportsTwoPhaseSlotExchange<
                  TwoPhaseArrayStoragePolicy,
                  Slot>::value);
static_assert(!detail::SupportsTwoPhaseSlotExchange<ArrayStoragePolicy,
                                                    Slot>::value);
static_assert(!std::is_default_constructible<
                  TwoPhaseContract::StagingToken>::value);
static_assert(!std::is_copy_constructible<
                  TwoPhaseContract::StagingToken>::value);
static_assert(std::is_move_constructible<
                  TwoPhaseContract::StagingToken>::value);
static_assert(!std::is_move_assignable<
                  TwoPhaseContract::StagingToken>::value);
static_assert(!HasTokenSelfCopyPrepared<
                  TwoPhaseContract::StagingToken,
                  Slot>::value,
              "staging tokens must not dereference retained owners");
static_assert(!std::is_default_constructible<
                  TwoPhaseContract::PreparedCommitToken>::value);
static_assert(!std::is_copy_constructible<
                  TwoPhaseContract::PreparedCommitToken>::value);
static_assert(std::is_move_constructible<
                  TwoPhaseContract::PreparedCommitToken>::value);
static_assert(!std::is_move_assignable<
                  TwoPhaseContract::PreparedCommitToken>::value);
static_assert(!std::is_default_constructible<
                  TwoPhaseContract::ProvisionalCommitToken>::value);
static_assert(!std::is_copy_constructible<
                  TwoPhaseContract::ProvisionalCommitToken>::value);
static_assert(std::is_move_constructible<
                  TwoPhaseContract::ProvisionalCommitToken>::value);
static_assert(!std::is_move_assignable<
                  TwoPhaseContract::ProvisionalCommitToken>::value);
static_assert(static_cast<unsigned int>(FailureReason::count) < 64);
static_assert(static_cast<unsigned int>(RestartCompatibility::count) < 64);
static_assert(FailureMask(static_cast<FailureReason>(255)) == 0);
static_assert(FailureMask(FailureReason::count) == 0);
static_assert(FailureMask(static_cast<FailureReason>(
                  static_cast<unsigned int>(FailureReason::count) + 1U)) == 0);
static_assert(CompatibilityMask(static_cast<RestartCompatibility>(255)) == 0);
static_assert(CompatibilityMask(RestartCompatibility::count) == 0);
static_assert(CompatibilityMask(static_cast<RestartCompatibility>(
                  static_cast<unsigned int>(RestartCompatibility::count) +
                  1U)) == 0);
static_assert(detail::FailureBit(static_cast<FailureReason>(255)) == 0);
static_assert(detail::FailureBit(FailureReason::count) == 0);
static_assert(detail::FailureBit(static_cast<FailureReason>(
                  static_cast<unsigned int>(FailureReason::count) + 1U)) == 0);
static_assert(detail::CompatibilityBit(
                  static_cast<RestartCompatibility>(255)) == 0);
static_assert(
    detail::CompatibilityBit(RestartCompatibility::count) == 0);
static_assert(detail::CompatibilityBit(static_cast<RestartCompatibility>(
                  static_cast<unsigned int>(RestartCompatibility::count) +
                  1U)) == 0);
static_assert(!detail::HasOnlyNamedFailureBits(unknown_collective_mask_bit));
static_assert(
    !detail::HasOnlyNamedCompatibilityBits(unknown_collective_mask_bit));
static_assert(std::is_same<decltype(SmoothAdmissionThresholds<double>{}
                                        .minimum_rate.value),
                           double>::value);

Slot CommittedSlot()
{
  return {{{1.0, 2.0, 3.0, 4.0}}, {{5.0, 6.0, 7.0}}};
}

Slot ScratchSlot()
{
  return {{{-1.0, -2.0, -3.0, -4.0}}, {{-5.0, -6.0, -7.0}}};
}

Slot AcceptedSlot()
{
  return {{{11.0, 12.0, 13.0, 14.0}}, {{15.0, 16.0, 17.0}}};
}

Slot OtherSlot()
{
  return {{{31.0, 32.0, 33.0, 34.0}}, {{35.0, 36.0, 37.0}}};
}

StateCoordinates InitialCoordinates()
{
  return {1.0, 4, 7, 9, 0.25};
}

PersistentResourceDescriptor Resources()
{
  return {StableIdentity{"d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec"},
          Digest(10),
          Digest(20),
          Digest(30),
          1,
          3,
          StableIdentity{"mfem.bynodes.fp64.v1"},
          {true, true, true, false, 41, 53, 4096, 2048}};
}

Contract MakeContract(const StateCoordinates coordinates = InitialCoordinates(),
                      const PersistentResourceDescriptor resources = Resources(),
                      const std::uint64_t initial_sequence = 0,
                      const Slot committed = CommittedSlot(),
                      const Slot scratch = ScratchSlot())
{
  ArrayStoragePolicy::Reset();
  TestCollectivePolicy::Reset();
  return Contract{committed, scratch, coordinates, resources, initial_sequence};
}

TwoPhaseContract MakeTwoPhaseContract(
    const StateCoordinates coordinates = InitialCoordinates(),
    const PersistentResourceDescriptor resources = Resources(),
    const std::uint64_t initial_sequence = 0,
    const Slot committed = CommittedSlot(),
    const Slot scratch = ScratchSlot())
{
  TwoPhaseArrayStoragePolicy::Reset();
  TestCollectivePolicy::Reset();
  return TwoPhaseContract{
      committed, scratch, coordinates, resources, initial_sequence};
}

AttemptReport PassingReport(const std::uint64_t cursor)
{
  AttemptReport report{};
  report.gates = {true, true, true, true, true, true};
  report.telemetry.transfers = {30, 40, 5, 0, 2, true};
  report.telemetry.true_relative_residual = 5.0e-12;
  report.candidate_cursor = cursor;
  return report;
}

Contract::ProductBundle IssueAll(Contract& contract,
                                 const AttemptIdentity& attempt)
{
  Contract::ProductBundle products{};
  products.residual =
      contract.IssueProduct<DerivedProductKind::residual>(attempt);
  products.matrix = contract.IssueProduct<DerivedProductKind::matrix>(attempt);
  products.factor = contract.IssueProduct<DerivedProductKind::factor>(attempt);
  products.solution =
      contract.IssueProduct<DerivedProductKind::solution>(attempt);
  return products;
}

template <typename ContractType>
typename ContractType::ProductBundle IssueAllFor(
    ContractType& contract,
    const AttemptIdentity& attempt)
{
  typename ContractType::ProductBundle products{};
  products.residual =
      contract.template IssueProduct<DerivedProductKind::residual>(attempt);
  products.matrix =
      contract.template IssueProduct<DerivedProductKind::matrix>(attempt);
  products.factor =
      contract.template IssueProduct<DerivedProductKind::factor>(attempt);
  products.solution =
      contract.template IssueProduct<DerivedProductKind::solution>(attempt);
  return products;
}

bool StoreCandidate(Contract& contract,
                    const AttemptIdentity& attempt,
                    const Slot& value)
{
  const auto capability = contract.AcquireCandidate(attempt);
  return capability && contract.StoreCandidate(*capability, value);
}

template <typename ContractType>
bool StoreCandidateFor(ContractType& contract,
                       const AttemptIdentity& attempt,
                       const typename ContractType::state_slot& value)
{
  const auto capability = contract.AcquireCandidate(attempt);
  return capability && contract.StoreCandidate(*capability, value);
}

bool CopyPublished(const Contract& contract, Slot& copy)
{
  const auto capability = contract.Published();
  return contract.CopyPublished(capability, copy);
}

template <typename ContractType>
bool CopyPublishedFor(const ContractType& contract,
                      typename ContractType::state_slot& copy)
{
  const auto capability = contract.Published();
  return contract.CopyPublished(capability, copy);
}

bool SameCoordinates(const StateCoordinates& left,
                     const StateCoordinates& right)
{
  return std::memcmp(&left.time, &right.time, sizeof(double)) == 0 &&
      left.step == right.step && left.generation == right.generation &&
      left.cursor == right.cursor &&
      std::memcmp(&left.dt, &right.dt, sizeof(double)) == 0;
}

bool SameSlotBytes(const Slot& left, const Slot& right)
{
  return std::memcmp(&left, &right, sizeof(Slot)) == 0;
}

bool SameTransfers(const TransferAccounting& left,
                   const TransferAccounting& right)
{
  return left.host_to_device_bytes == right.host_to_device_bytes &&
      left.device_to_host_bytes == right.device_to_host_bytes &&
      left.mpi_staging_bytes == right.mpi_staging_bytes &&
      left.allocation_events == right.allocation_events &&
      left.synchronization_events == right.synchronization_events &&
      left.complete == right.complete;
}

bool SameAttempt(const AttemptIdentity& left,
                 const AttemptIdentity& right)
{
  return left.sequence == right.sequence &&
      left.committed_step == right.committed_step &&
      left.committed_generation == right.committed_generation &&
      left.retry_index == right.retry_index &&
      std::memcmp(&left.dt, &right.dt, sizeof(double)) == 0;
}

bool SameProhibited(const ProhibitedActionAccounting& left,
                    const ProhibitedActionAccounting& right)
{
  return left.dt_change_events == right.dt_change_events &&
      left.clipping_events == right.clipping_events &&
      left.solver_substitution_events == right.solver_substitution_events &&
      left.fallback_events == right.fallback_events &&
      left.hidden_insertion_events == right.hidden_insertion_events;
}

bool SameAccepted(const AcceptedTelemetry& left,
                  const AcceptedTelemetry& right)
{
  return left.valid == right.valid &&
      SameCoordinates(left.coordinates, right.coordinates) &&
      SameAttempt(left.attempt.identity, right.attempt.identity) &&
      SameTransfers(left.attempt.transfers, right.attempt.transfers) &&
      SameProhibited(left.attempt.prohibited, right.attempt.prohibited) &&
      left.attempt.failure_reason == right.attempt.failure_reason &&
      left.attempt.validated_product_mask ==
          right.attempt.validated_product_mask &&
      std::memcmp(&left.attempt.true_relative_residual,
                  &right.attempt.true_relative_residual,
                  sizeof(double)) == 0;
}

bool SameCommitAccounting(const RunTelemetry& left,
                          const RunTelemetry& right)
{
  return left.admissions_started == right.admissions_started &&
      left.admissions_failed == right.admissions_failed &&
      left.attempts_started == right.attempts_started &&
      left.attempts_failed == right.attempts_failed &&
      left.rollbacks == right.rollbacks &&
      left.retries_started == right.retries_started &&
      left.commits == right.commits && left.restores == right.restores &&
      left.duplicate_product_issues_rejected ==
          right.duplicate_product_issues_rejected &&
      left.out_of_order_product_issues_rejected ==
          right.out_of_order_product_issues_rejected &&
      left.transfer_accounting_overflow_rejections ==
          right.transfer_accounting_overflow_rejections &&
      SameTransfers(left.attempted_transfers, right.attempted_transfers) &&
      SameTransfers(left.rolled_back_transfers,
                    right.rolled_back_transfers) &&
      SameTransfers(left.committed_transfers, right.committed_transfers) &&
      SameAccepted(left.accepted, right.accepted) &&
      SameAttempt(left.last_attempt.identity, right.last_attempt.identity) &&
      SameTransfers(left.last_attempt.transfers,
                    right.last_attempt.transfers) &&
      SameProhibited(left.last_attempt.prohibited,
                     right.last_attempt.prohibited) &&
      left.last_attempt.failure_reason == right.last_attempt.failure_reason &&
      left.last_attempt.validated_product_mask ==
          right.last_attempt.validated_product_mask &&
      left.first_failure == right.first_failure &&
      std::memcmp(&left.last_attempt.true_relative_residual,
                  &right.last_attempt.true_relative_residual,
                  sizeof(double)) == 0;
}

std::uint64_t FrozenCoordinateFingerprint(
    const StateCoordinates& coordinates) noexcept
{
  std::array<std::uint64_t, 5> fields{};
  std::memcpy(&fields[0], &coordinates.time, sizeof(double));
  fields[1] = coordinates.step;
  fields[2] = coordinates.generation;
  fields[3] = coordinates.cursor;
  std::memcpy(&fields[4], &coordinates.dt, sizeof(double));
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

bool Check(const bool condition, const char* requirement)
{
  if (!condition)
  {
    std::cerr << "FAIL requirement=" << requirement << '\n';
  }
  return condition;
}

bool CheckStaticSelectionAndCollectiveAdmission()
{
  bool passed = true;
  const RuntimeTupleDescriptor tuple = Selection::Descriptor();
  passed = Check(tuple.Valid() &&
                     tuple.model == StableIdentity{"euler.compressible.2d"} &&
                     tuple.backend == StableIdentity{"cpu.serial"} &&
                     tuple.precision == StableIdentity{"fp64"},
                 "static_owned_outer_selection") && passed;

  std::array<char, 16> caller_text{{'t', 'e', 'm', 'p', '.', 'i', 'd'}};
  const StableIdentity owned = StableIdentity::Copy(
      std::string_view{caller_text.data(), 7});
  caller_text.fill('x');
  passed = Check(owned == StableIdentity{"temp.id"},
                 "text_identity_owns_source_lifetime") && passed;

  const SmoothAdmissionThresholds<double> thresholds{};
  passed = Check(!thresholds.minimum_rate.specified &&
                     !thresholds.maximum_conservation_error.specified,
                 "smooth_thresholds_are_typed_but_uninvented") && passed;

  struct InvalidCase
  {
    InteriorSpace space;
    double dt;
    FailureReason reason;
    const char* name;
  };
  const std::array<InvalidCase, 6> invalid_requests{{
      {InteriorSpace::h1,
       0.125,
       FailureReason::unsupported_h1,
       "h1_explicitly_unsupported_after_collective"},
      {static_cast<InteriorSpace>(255),
       0.125,
       FailureReason::invalid_interior_space,
       "invalid_space_collectively_rejected"},
      {InteriorSpace::l2,
       0.0,
       FailureReason::invalid_dt,
       "zero_dt_collectively_rejected"},
      {InteriorSpace::l2,
       -1.0,
       FailureReason::invalid_dt,
       "negative_dt_collectively_rejected"},
      {InteriorSpace::l2,
       std::numeric_limits<double>::infinity(),
       FailureReason::invalid_dt,
       "infinite_dt_collectively_rejected"},
      {InteriorSpace::l2,
       std::numeric_limits<double>::quiet_NaN(),
       FailureReason::invalid_dt,
       "nan_dt_collectively_rejected"}}};
  for (const InvalidCase& invalid : invalid_requests)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({invalid.space, invalid.dt});
    passed = Check(!begin.started && begin.reason == invalid.reason &&
                       TestCollectivePolicy::admissions == 1 &&
                       ArrayStoragePolicy::preparations == 0,
                   invalid.name) && passed;
  }

  {
    StateCoordinates coordinates = InitialCoordinates();
    coordinates.time = std::numeric_limits<double>::quiet_NaN();
    Contract contract = MakeContract(coordinates);
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started && begin.reason == FailureReason::invalid_dt &&
                       TestCollectivePolicy::admissions == 1,
                   "nonfinite_internal_time_enters_admission") && passed;
  }
  {
    PersistentResourceDescriptor resources = Resources();
    resources.reuse.persistent_workspace = false;
    Contract contract = MakeContract(InitialCoordinates(), resources);
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason ==
                           FailureReason::invalid_resource_descriptor &&
                       contract.telemetry().first_failure ==
                           FailureReason::invalid_resource_descriptor &&
                       TestCollectivePolicy::admissions == 1,
                   "constructor_failure_is_write_once_and_enters_admission") &&
        passed;
  }

  for (const int field : {0, 1})
  {
    StateCoordinates coordinates = InitialCoordinates();
    if (field == 0)
    {
      coordinates.step = std::numeric_limits<std::uint64_t>::max();
    }
    else
    {
      coordinates.generation = std::numeric_limits<std::uint64_t>::max();
    }
    Contract contract = MakeContract(coordinates);
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::coordinate_overflow &&
                       TestCollectivePolicy::admissions == 1,
                   "coordinate_overflow_enters_admission") && passed;
  }
  {
    Contract contract = MakeContract(
        InitialCoordinates(), Resources(),
        std::numeric_limits<std::uint64_t>::max());
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::coordinate_overflow &&
                       TestCollectivePolicy::admissions == 1,
                   "attempt_sequence_overflow_enters_admission") && passed;
  }
  {
    StateCoordinates coordinates = InitialCoordinates();
    coordinates.time = std::numeric_limits<double>::max();
    Contract contract = MakeContract(coordinates);
    const BeginOutcome begin = contract.BeginStep(
        {InteriorSpace::l2, std::numeric_limits<double>::max()});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::coordinate_overflow &&
                       TestCollectivePolicy::admissions == 1,
                   "target_time_overflow_enters_admission") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome first =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const BeginOutcome second =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(first.started && !second.started &&
                       second.reason == FailureReason::stale_attempt &&
                       TestCollectivePolicy::admissions == 2,
                   "invalid_local_status_still_enters_admission") && passed;
  }
  {
    Contract contract = MakeContract();
    ArrayStoragePolicy::fail_prepare_call = 1;
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    Slot published{};
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::candidate_preparation &&
                       TestCollectivePolicy::admissions == 1 &&
                       contract.telemetry().attempts_started == 0 &&
                       CopyPublished(contract, published) &&
                       published.interior == CommittedSlot().interior,
                   "local_prepare_failure_collectively_terminal") && passed;
  }

  const std::array<FailureReason, 9> remote_preadmission_failures{{
      FailureReason::stale_attempt,
      FailureReason::unsupported_h1,
      FailureReason::invalid_interior_space,
      FailureReason::invalid_dt,
      FailureReason::invalid_resource_descriptor,
      FailureReason::coordinate_overflow,
      FailureReason::shared_backing_store,
      FailureReason::candidate_preparation,
      FailureReason::publication_precondition}};
  for (const FailureReason reason : remote_preadmission_failures)
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault = CollectiveFault::admission_remote_failure;
    TestCollectivePolicy::remote_reason = reason;
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    Slot published{};
    passed = Check(!begin.started && begin.reason == reason &&
                       TestCollectivePolicy::admissions == 1 &&
                       contract.telemetry().attempts_started == 0 &&
                       CopyPublished(contract, published) &&
                       published.interior == CommittedSlot().interior,
                   "one_rank_preadmission_failure_is_atomic") && passed;
  }

  for (const CollectiveFault fault : {
           CollectiveFault::admission_identity_false,
           CollectiveFault::admission_failure_consistency_false,
           CollectiveFault::admission_participants,
           CollectiveFault::admission_context})
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault = fault;
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::collective_protocol &&
                       TestCollectivePolicy::admissions == 1,
                   "admission_identity_truth_or_census_mismatch") && passed;
  }
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault =
        CollectiveFault::admission_hide_local_failure;
    const BeginOutcome first =
        contract.BeginStep({InteriorSpace::h1, 0.125});
    passed = Check(!first.started &&
                       first.reason == FailureReason::collective_protocol &&
                       TestCollectivePolicy::admissions == 1,
                   "admission_cannot_hide_local_failure_truth") && passed;
  }
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault =
        CollectiveFault::admission_change_local_failure;
    TestCollectivePolicy::remote_reason = FailureReason::backend_failure;
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::h1, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::collective_protocol &&
                       TestCollectivePolicy::admissions == 1,
                   "admission_cannot_substitute_local_failure_bit") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport failed_report = PassingReport(12);
    failed_report.gates.finite = false;
    const AttemptOutcome failed = contract.FinishAttempt(
        begin.attempt, failed_report, std::move(products));
    ArrayStoragePolicy::fail_prepare_call = 2;
    const BeginOutcome retry = contract.Retry();
    passed = Check(failed.retry_available && !retry.started &&
                       retry.reason == FailureReason::candidate_preparation &&
                       TestCollectivePolicy::admissions == 2 &&
                       SameCoordinates(contract.coordinates(),
                                       InitialCoordinates()),
                   "retry_prepare_failure_collectively_terminal") && passed;
  }
  return passed;
}

struct BackingState
{
  std::uint64_t backing = 0;
  double value = 0.0;
};

using BackingSlot = CompleteStateSlot<BackingState, BackingState>;

struct BackingStoragePolicy
{
  static constexpr bool preparation_preserves_committed = true;
  static constexpr bool private_copy_is_independent = true;
  static constexpr bool publish_is_atomic = true;
  static constexpr bool publish_preserves_disjoint_backing = true;
  static inline bool alias_on_prepare = false;
  static inline bool alias_on_store = false;
  static inline std::uint64_t restore_preparations = 0;
  static inline std::uint64_t can_publish_calls = 0;
  static inline std::uint64_t reject_can_publish_call = 0;
  static inline std::uint64_t publish_calls = 0;

  static void Reset() noexcept
  {
    alias_on_prepare = false;
    alias_on_store = false;
    restore_preparations = 0;
    can_publish_calls = 0;
    reject_can_publish_call = 0;
    publish_calls = 0;
  }

  static bool Disjoint(const BackingSlot& left,
                       const BackingSlot& right) noexcept
  {
    return left.interior.backing != right.interior.backing &&
        left.trace.backing != right.trace.backing;
  }

  static bool PrepareCandidate(const BackingSlot& committed,
                               BackingSlot& candidate) noexcept
  {
    candidate.interior.value = committed.interior.value;
    candidate.trace.value = committed.trace.value;
    if (alias_on_prepare)
    {
      candidate.interior.backing = committed.interior.backing;
      candidate.trace.backing = committed.trace.backing;
    }
    return true;
  }

  static bool PrepareRestore(const BackingSlot& loaded,
                             BackingSlot& candidate) noexcept
  {
    ++restore_preparations;
    candidate.interior.value = loaded.interior.value;
    candidate.trace.value = loaded.trace.value;
    return true;
  }

  static bool StoreCandidate(BackingSlot& candidate,
                             const BackingSlot& value) noexcept
  {
    candidate.interior.value = value.interior.value;
    candidate.trace.value = value.trace.value;
    if (alias_on_store)
    {
      candidate.interior.backing = value.interior.backing;
      candidate.trace.backing = value.trace.backing;
    }
    return true;
  }

  static bool CopyOut(const BackingSlot& source, BackingSlot& copy) noexcept
  {
    copy.interior.value = source.interior.value;
    copy.trace.value = source.trace.value;
    return true;
  }

  static void DiscardCandidate(BackingSlot& candidate) noexcept
  {
    candidate.interior.value = -900.0;
    candidate.trace.value = -901.0;
  }

  static bool CanPublish(const BackingSlot& committed,
                         const BackingSlot& candidate) noexcept
  {
    ++can_publish_calls;
    return can_publish_calls != reject_can_publish_call &&
        Disjoint(committed, candidate);
  }

  static void Publish(BackingSlot& committed,
                      BackingSlot& candidate) noexcept
  {
    ++publish_calls;
    // If an invalid policy invocation reached here it would corrupt bytes and
    // alias backing. The contract test requires rejection before this call.
    committed.interior.value = 999.0;
    committed.trace.value = 998.0;
    candidate.interior.backing = committed.interior.backing;
    candidate.trace.backing = committed.trace.backing;
  }

  static StateHash HashInterior(const BackingState& state) noexcept
  {
    return Digest(static_cast<std::uint8_t>(state.value));
  }

  static StateHash HashTrace(const BackingState& state) noexcept
  {
    return Digest(static_cast<std::uint8_t>(state.value + 1.0));
  }
};

using BackingContract = PersistentRunContract<Selection,
                                              BackingState,
                                              BackingState,
                                              BackingStoragePolicy,
                                              TestCollectivePolicy>;

BackingContract MakeBackingContract(const bool shared = false)
{
  BackingStoragePolicy::Reset();
  TestCollectivePolicy::Reset();
  const BackingSlot committed{{1, 10.0}, {2, 20.0}};
  const BackingSlot candidate{
      {shared ? 1U : 3U, -10.0}, {shared ? 2U : 4U, -20.0}};
  return BackingContract{
      committed, candidate, InitialCoordinates(), Resources()};
}

bool CheckCapabilitiesAndFailureAtomicPublication()
{
  bool passed = true;
  {
    BackingContract contract = MakeBackingContract(true);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::shared_backing_store &&
                       TestCollectivePolicy::admissions == 1,
                   "shared_backing_construction_still_collects_admission") &&
        passed;
  }
  {
    BackingContract contract = MakeBackingContract();
    BackingStoragePolicy::alias_on_prepare = true;
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::shared_backing_store &&
                       TestCollectivePolicy::admissions == 1,
                   "shared_backing_after_preparation_rejected") && passed;
  }
  {
    BackingContract contract = MakeBackingContract();
    const auto published = contract.Published();
    BackingSlot before{{91, 0.0}, {92, 0.0}};
    const bool captured = contract.CopyPublished(published, before);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const auto candidate = contract.AcquireCandidate(begin.attempt);
    BackingStoragePolicy::alias_on_store = true;
    const BackingSlot aliased_value{{5, 30.0}, {6, 40.0}};
    const bool stored = candidate &&
        contract.StoreCandidate(*candidate, aliased_value);
    auto products = IssueAllFor(contract, begin.attempt);
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(14), std::move(products));
    BackingSlot after{{93, 0.0}, {94, 0.0}};
    passed = Check(captured && !stored && !outcome.committed &&
                       outcome.reason == FailureReason::shared_backing_store &&
                       BackingStoragePolicy::publish_calls == 0 &&
                       contract.CopyPublished(published, after) &&
                       before.interior.value == after.interior.value &&
                       before.trace.value == after.trace.value,
                   "candidate_input_alias_is_detected_before_publication") &&
        passed;
  }
  {
    Contract contract = MakeContract();
    const auto old = contract.Old();
    Slot old_copy{};
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const auto candidate = contract.AcquireCandidate(begin.attempt);
    Slot candidate_copy{};
    passed = Check(begin.started && candidate &&
                       contract.CopyPublished(old, old_copy) &&
                       contract.StoreCandidate(*candidate, AcceptedSlot()) &&
                       contract.CopyCandidate(*candidate, candidate_copy) &&
                       candidate_copy.interior == AcceptedSlot().interior,
                   "pointer_free_capabilities_require_live_contract") && passed;
    auto products = IssueAll(contract, begin.attempt);
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    Slot stale_copy{};
    Slot fresh_copy{};
    StateCoordinates stale_coordinates{};
    passed = Check(outcome.committed &&
                       !contract.CopyPublished(old, stale_copy) &&
                       !contract.PublishedCoordinates(old, stale_coordinates) &&
                       !contract.StoreCandidate(*candidate, OtherSlot()) &&
                       CopyPublished(contract, fresh_copy) &&
                       fresh_copy.interior == AcceptedSlot().interior &&
                       fresh_copy.trace == AcceptedSlot().trace,
                   "retained_candidate_and_published_capabilities_go_stale") &&
        passed;
    Contract other = MakeContract();
    passed = Check(!other.CopyPublished(old, stale_copy) &&
                       !other.StoreCandidate(*candidate, OtherSlot()),
                   "capabilities_reject_different_live_owner") && passed;
  }
  {
    BackingContract contract = MakeBackingContract();
    const auto published_before = contract.Published();
    BackingSlot committed_before{{91, 0.0}, {92, 0.0}};
    StateCoordinates coordinates_before{};
    const bool copied_before =
        contract.CopyPublished(published_before, committed_before) &&
        contract.PublishedCoordinates(published_before, coordinates_before);
    const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const auto candidate = contract.AcquireCandidate(begin.attempt);
    const BackingSlot value{{5, 30.0}, {6, 40.0}};
    const bool stored = candidate && contract.StoreCandidate(*candidate, value);
    auto products = IssueAllFor(contract, begin.attempt);
    BackingStoragePolicy::reject_can_publish_call = 2;
    const AttemptReport report = PassingReport(14);
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    BackingSlot committed_after{{93, 0.0}, {94, 0.0}};
    const bool copied_after =
        contract.CopyPublished(published_before, committed_after);
    passed = Check(copied_before && stored && !outcome.committed &&
                       outcome.reason ==
                           FailureReason::publication_precondition &&
                       outcome.retry_available &&
                       BackingStoragePolicy::publish_calls == 0 && copied_after &&
                       committed_after.interior.value ==
                           committed_before.interior.value &&
                       committed_after.trace.value ==
                           committed_before.trace.value &&
                       SameCoordinates(contract.coordinates(),
                                       coordinates_before) &&
                       SameAccepted(contract.telemetry().accepted,
                                    accepted_before) &&
                       SameTransfers(contract.telemetry().rolled_back_transfers,
                                     report.telemetry.transfers) &&
                       contract.telemetry().committed_transfers
                               .host_to_device_bytes == 0,
                   "malicious_publish_rejected_before_any_mutation") && passed;
  }
  return passed;
}

bool CheckProductProtocolAndIncarnationABA()
{
  bool passed = true;
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const auto matrix =
        contract.IssueProduct<DerivedProductKind::matrix>(begin.attempt);
    const auto residual =
        contract.IssueProduct<DerivedProductKind::residual>(begin.attempt);
    const auto duplicate =
        contract.IssueProduct<DerivedProductKind::residual>(begin.attempt);
    passed = Check(!matrix && residual && !duplicate &&
                       contract.telemetry()
                               .out_of_order_product_issues_rejected == 1 &&
                       contract.telemetry()
                               .duplicate_product_issues_rejected == 1,
                   "product_issuance_order_and_duplicates_rejected") && passed;
  }

  for (std::size_t missing = 0; missing < 4; ++missing)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    if (missing == 0) { products.residual.reset(); }
    if (missing == 1) { products.matrix.reset(); }
    if (missing == 2) { products.factor.reset(); }
    if (missing == 3) { products.solution.reset(); }
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    passed = Check(!outcome.committed && outcome.retry_available &&
                       outcome.reason == FailureReason::missing_product,
                   "each_absent_required_product_is_missing") && passed;
  }

  {
    Contract source = MakeContract();
    const BeginOutcome source_begin =
        source.BeginStep({InteriorSpace::l2, 0.125});
    auto foreign = IssueAll(source, source_begin.attempt);
    Contract destination = MakeContract();
    const BeginOutcome destination_begin =
        destination.BeginStep({InteriorSpace::l2, 0.125});
    const AttemptOutcome outcome = destination.FinishAttempt(
        destination_begin.attempt, PassingReport(12), std::move(foreign));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::invalid_product,
                   "foreign_full_bundle_is_invalid_not_missing") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto stale = IssueAll(contract, begin.attempt);
    Contract::ProductBundle missing{};
    const AttemptOutcome first = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(missing));
    const BeginOutcome retry = contract.Retry();
    const AttemptOutcome second = contract.FinishAttempt(
        retry.attempt, PassingReport(12), std::move(stale));
    passed = Check(first.retry_available && retry.started &&
                       !second.committed &&
                       second.reason == FailureReason::stale_product,
                   "prior_attempt_full_bundle_is_stale_not_missing") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport failed = PassingReport(12);
    failed.gates.finite = false;
    const AttemptOutcome first = contract.FinishAttempt(
        begin.attempt, failed, std::move(products));
    Contract::ProductBundle duplicate_use{};
    const std::uint64_t rollbacks = contract.telemetry().rollbacks;
    const AttemptOutcome duplicate = contract.FinishAttempt(
        begin.attempt, failed, std::move(duplicate_use));
    passed = Check(first.retry_available && !duplicate.committed &&
                       duplicate.reason == FailureReason::stale_attempt &&
                       duplicate.status == RunStatus::retry_available &&
                       duplicate.retry_available &&
                       contract.telemetry().rollbacks == rollbacks,
                   "duplicate_finish_is_stale_and_retry_consistent") && passed;
  }

  {
    alignas(Contract) std::array<unsigned char, sizeof(Contract)> storage{};
    ArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    Contract* first = new (storage.data()) Contract{
        CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
    const auto old_published = first->Published();
    const BeginOutcome first_begin =
        first->BeginStep({InteriorSpace::l2, 0.125});
    const auto old_candidate = first->AcquireCandidate(first_begin.attempt);
    auto old_products = IssueAll(*first, first_begin.attempt);
    first->~Contract();

    ArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    Contract* second = new (storage.data()) Contract{
        CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
    const BeginOutcome second_begin =
        second->BeginStep({InteriorSpace::l2, 0.125});
    Slot copy{};
    const bool old_candidate_rejected =
        old_candidate && !second->StoreCandidate(*old_candidate, AcceptedSlot());
    const bool old_published_rejected =
        !second->CopyPublished(old_published, copy);
    const AttemptOutcome outcome = second->FinishAttempt(
        second_begin.attempt, PassingReport(12), std::move(old_products));
    passed = Check(old_candidate_rejected && old_published_rejected &&
                       !outcome.committed &&
                       outcome.reason == FailureReason::invalid_product,
                   "same_address_reconstruction_rejects_all_old_capabilities") &&
        passed;
    second->~Contract();
  }
  return passed;
}

bool CheckAbandonedRetryAdmitsFreshTimestep()
{
  Contract contract = MakeContract();
  const BeginOutcome initial =
      contract.BeginStep({InteriorSpace::l2, 0.125});
  auto initial_products = IssueAll(contract, initial.attempt);
  AttemptReport rejected = PassingReport(12);
  rejected.gates.success = false;
  const AttemptOutcome rolled_back = contract.FinishAttempt(
      initial.attempt, rejected, std::move(initial_products));

  const StateCoordinates coordinates_before = contract.coordinates();
  Slot published_before{};
  const bool published_captured = CopyPublished(contract, published_before);
  const bool abandoned = contract.AbandonRetry();
  const bool duplicate_rejected = !contract.AbandonRetry();
  Slot published_after{};
  const bool published_preserved = CopyPublished(contract, published_after);

  constexpr double replacement_dt = 0.03125;
  const BeginOutcome replacement =
      contract.BeginStep({InteriorSpace::l2, replacement_dt});
  const bool stored = replacement.started &&
      StoreCandidate(contract, replacement.attempt, AcceptedSlot());
  auto replacement_products = replacement.started
      ? IssueAll(contract, replacement.attempt)
      : Contract::ProductBundle{};
  const AttemptOutcome committed = replacement.started
      ? contract.FinishAttempt(
            replacement.attempt, PassingReport(14),
            std::move(replacement_products))
      : AttemptOutcome{};
  const StateCoordinates expected{1.0 + replacement_dt, 5, 8, 14,
                                  replacement_dt};

  return Check(
      initial.started && rolled_back.retry_available &&
          published_captured && abandoned && duplicate_rejected &&
          published_preserved &&
          published_after.interior == published_before.interior &&
          published_after.trace == published_before.trace &&
          SameCoordinates(coordinates_before, InitialCoordinates()) &&
          replacement.started && replacement.attempt.retry_index == 0 &&
          replacement.attempt.dt == replacement_dt && stored &&
          committed.committed && SameCoordinates(contract.coordinates(), expected),
      "abandoned_exact_retry_admits_fresh_primary_timestep");
}

bool CheckStaleFinishIsolation()
{
  bool passed = true;
  {
    Contract contract = MakeContract();
    const BeginOutcome initial =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto initial_products = IssueAll(contract, initial.attempt);
    AttemptReport initial_failure = PassingReport(12);
    initial_failure.gates.finite = false;
    const AttemptOutcome failed = contract.FinishAttempt(
        initial.attempt, initial_failure, std::move(initial_products));
    const BeginOutcome retry = contract.Retry();
    const bool stored =
        StoreCandidate(contract, retry.attempt, AcceptedSlot());
    const auto retry_capability = contract.AcquireCandidate(retry.attempt);
    Slot candidate_before{};
    const bool candidate_captured = retry_capability &&
        contract.CopyCandidate(*retry_capability, candidate_before);
    auto retry_products = IssueAll(contract, retry.attempt);
    const auto published_capability = contract.Published();
    Slot published_before{};
    const bool published_captured =
        contract.CopyPublished(published_capability, published_before);
    const StateCoordinates coordinates_before = contract.coordinates();
    const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
    const TransferAccounting attempted_before =
        contract.telemetry().attempted_transfers;
    const TransferAccounting rolled_back_before =
        contract.telemetry().rolled_back_transfers;
    const TransferAccounting committed_before =
        contract.telemetry().committed_transfers;
    const std::uint64_t rollbacks_before = contract.telemetry().rollbacks;

    AttemptReport stale_report = PassingReport(99);
    stale_report.telemetry.transfers = {300, 400, 50, 0, 20, true};
    Contract::ProductBundle stale_products{};
    const AttemptOutcome stale = contract.FinishAttempt(
        initial.attempt, stale_report, std::move(stale_products));
    Slot candidate_after{};
    Slot published_after{};
    const bool candidate_preserved = retry_capability &&
        contract.CopyCandidate(*retry_capability, candidate_after);
    const bool retry_still_current =
        contract.IsCurrentAttempt(retry.attempt);
    const bool published_preserved =
        contract.CopyPublished(published_capability, published_after);

    AttemptReport retry_report = PassingReport(13);
    retry_report.telemetry.transfers = {7, 8, 9, 0, 3, true};
    const AttemptOutcome committed = contract.FinishAttempt(
        retry.attempt, retry_report, std::move(retry_products));
    const StateCoordinates expected_coordinates{1.125, 5, 8, 13, 0.125};
    passed = Check(
                 failed.retry_available && retry.started && stored &&
                     candidate_captured && published_captured &&
                     !stale.committed &&
                     stale.status == RunStatus::attempt_active &&
                     stale.reason == FailureReason::stale_attempt &&
                     !stale.retry_available &&
                     contract.telemetry().rollbacks == rollbacks_before &&
                     retry_still_current &&
                     candidate_preserved &&
                     candidate_after.interior == candidate_before.interior &&
                     candidate_after.trace == candidate_before.trace &&
                     published_preserved &&
                     published_after.interior == published_before.interior &&
                     published_after.trace == published_before.trace &&
                     SameCoordinates(coordinates_before,
                                     InitialCoordinates()) &&
                     SameAccepted(accepted_before, AcceptedTelemetry{}) &&
                     SameTransfers(attempted_before,
                                   initial_failure.telemetry.transfers) &&
                     SameTransfers(rolled_back_before,
                                   initial_failure.telemetry.transfers) &&
                     committed_before.host_to_device_bytes == 0 &&
                     committed.committed &&
                     SameCoordinates(contract.coordinates(),
                                     expected_coordinates) &&
                     contract.telemetry().attempted_transfers
                             .host_to_device_bytes == 37 &&
                     contract.telemetry().attempted_transfers
                             .device_to_host_bytes == 48 &&
                     contract.telemetry().attempted_transfers
                             .mpi_staging_bytes == 14 &&
                     contract.telemetry().attempted_transfers
                             .synchronization_events == 5 &&
                     SameTransfers(contract.telemetry().rolled_back_transfers,
                                   initial_failure.telemetry.transfers) &&
                     SameTransfers(contract.telemetry().committed_transfers,
                                   retry_report.telemetry.transfers),
                 "stale_finish_preserves_live_retry_and_excludes_stale_transfers") &&
        passed;
  }

  {
    Contract contract = MakeContract();
    const BeginOutcome initial =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto initial_products = IssueAll(contract, initial.attempt);
    AttemptReport initial_failure = PassingReport(12);
    initial_failure.gates.finite = false;
    const AttemptOutcome failed = contract.FinishAttempt(
        initial.attempt, initial_failure, std::move(initial_products));
    const BeginOutcome retry = contract.Retry();
    const bool stored = StoreCandidate(contract, retry.attempt, OtherSlot());
    const TransferAccounting attempted_before =
        contract.telemetry().attempted_transfers;
    const TransferAccounting rolled_back_before =
        contract.telemetry().rolled_back_transfers;
    const TransferAccounting committed_before =
        contract.telemetry().committed_transfers;
    const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
    const StateCoordinates coordinates_before = contract.coordinates();
    Slot published_before{};
    const bool published_captured = CopyPublished(contract, published_before);

    TestCollectivePolicy::fault = CollectiveFault::finish_identity_false;
    AttemptReport stale_report = PassingReport(99);
    stale_report.telemetry.transfers = {900, 800, 70, 0, 60, true};
    Contract::ProductBundle stale_products{};
    const AttemptOutcome mixed = contract.FinishAttempt(
        initial.attempt, stale_report, std::move(stale_products));
    Slot published_after{};
    passed = Check(
                 failed.retry_available && retry.started && stored &&
                     published_captured && !mixed.committed &&
                     mixed.status == RunStatus::terminal_failure &&
                     mixed.reason == FailureReason::collective_protocol &&
                     !mixed.retry_available &&
                     !contract.IsCurrentAttempt(retry.attempt) &&
                     SameAttempt(contract.telemetry().last_attempt.identity,
                                 retry.attempt) &&
                     contract.telemetry().last_attempt.failure_reason ==
                         FailureReason::collective_protocol &&
                     SameTransfers(contract.telemetry().attempted_transfers,
                                   attempted_before) &&
                     SameTransfers(contract.telemetry().rolled_back_transfers,
                                   rolled_back_before) &&
                     SameTransfers(contract.telemetry().committed_transfers,
                                   committed_before) &&
                     SameAccepted(contract.telemetry().accepted,
                                  accepted_before) &&
                     SameCoordinates(contract.coordinates(),
                                     coordinates_before) &&
                     CopyPublished(contract, published_after) &&
                     published_after.interior == published_before.interior &&
                     published_after.trace == published_before.trace,
                 "mixed_stale_current_finish_aborts_retry_without_transfer_misattribution") &&
        passed;
  }
  return passed;
}

enum class RetryFailureCase : unsigned char
{
  local_gate,
  remote_gate,
  product,
  backend,
  collective
};

bool CheckFirstFailureAcrossSuccessfulRetry()
{
  bool passed = true;
  const std::array<RetryFailureCase, 5> cases{{
      RetryFailureCase::local_gate,
      RetryFailureCase::remote_gate,
      RetryFailureCase::product,
      RetryFailureCase::backend,
      RetryFailureCase::collective}};
  for (const RetryFailureCase failure_case : cases)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidate(contract, begin.attempt, OtherSlot());
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    FailureReason expected = FailureReason::none;
    if (failure_case == RetryFailureCase::local_gate)
    {
      report.gates.finite = false;
      expected = FailureReason::finite_gate;
    }
    else if (failure_case == RetryFailureCase::remote_gate)
    {
      TestCollectivePolicy::fault = CollectiveFault::finish_remote_failure;
      TestCollectivePolicy::remote_reason = FailureReason::admissibility_gate;
      expected = FailureReason::admissibility_gate;
    }
    else if (failure_case == RetryFailureCase::product)
    {
      products.solution.reset();
      expected = FailureReason::missing_product;
    }
    else if (failure_case == RetryFailureCase::backend)
    {
      report.originating_failure = FailureReason::backend_failure;
      expected = FailureReason::backend_failure;
    }
    else
    {
      TestCollectivePolicy::fault = CollectiveFault::finish_identity_false;
      expected = FailureReason::collective_protocol;
    }
    const AttemptOutcome failed = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    TestCollectivePolicy::fault = CollectiveFault::none;
    const BeginOutcome retry = contract.Retry();
    const auto retry_candidate = contract.AcquireCandidate(retry.attempt);
    Slot retry_base{};
    const bool retry_reuses_committed = retry_candidate &&
        contract.CopyCandidate(*retry_candidate, retry_base) &&
        retry_base.interior == CommittedSlot().interior &&
        retry_base.trace == CommittedSlot().trace;
    const bool stored = StoreCandidate(contract, retry.attempt, AcceptedSlot());
    auto retry_products = IssueAll(contract, retry.attempt);
    const AttemptOutcome accepted = contract.FinishAttempt(
        retry.attempt, PassingReport(13), std::move(retry_products));
    const StateCoordinates expected_coordinates{1.125, 5, 8, 13, 0.125};
    Slot published{};
    passed = Check(failed.retry_available && failed.reason == expected &&
                       retry.started && stored && accepted.committed &&
                       retry_reuses_committed &&
                       contract.telemetry().first_failure == expected &&
                       contract.telemetry().attempts_started == 2 &&
                       contract.telemetry().rollbacks == 1 &&
                       contract.telemetry().commits == 1 &&
                       SameCoordinates(contract.coordinates(),
                                       expected_coordinates) &&
                       CopyPublished(contract, published) &&
                       published.interior == AcceptedSlot().interior,
                   "first_resolved_failure_survives_successful_retry") && passed;
  }

  for (const FailureReason remote : {
           FailureReason::invalid_product, FailureReason::backend_failure})
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    TestCollectivePolicy::fault = CollectiveFault::finish_remote_failure;
    TestCollectivePolicy::remote_reason = remote;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    passed = Check(!outcome.committed && outcome.reason == remote &&
                       TestCollectivePolicy::reductions == 1,
                   "one_rank_product_or_backend_failure_reduces_truth") && passed;
  }
  return passed;
}

bool CheckResidualTelemetryAndCollectiveTruth()
{
  bool passed = true;
  for (const double residual : {0.0, Contract::true_relative_residual_limit})
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidate(contract, begin.attempt, AcceptedSlot());
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.telemetry.true_relative_residual = residual;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(outcome.committed,
                   "closed_nonnegative_true_residual_domain_accepts_bounds") &&
        passed;
  }

  const std::array<double, 5> invalid_residuals{{
      -1.0e-12,
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
      Contract::true_relative_residual_limit * 1.01}};
  for (const double residual : invalid_residuals)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.telemetry.true_relative_residual = residual;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::true_residual_gate,
                   "negative_nonfinite_or_above_limit_residual_rejected") &&
        passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.gates.true_residual = false;
    report.telemetry.true_relative_residual = 0.0;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::true_residual_gate,
                   "oracle_true_residual_gate_is_still_required") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport first = PassingReport(12);
    first.gates.finite = false;
    first.telemetry.transfers.host_to_device_bytes =
        std::numeric_limits<std::uint64_t>::max();
    const bool boundary_fits =
        contract.CanAccountAttemptTransfers(first.telemetry.transfers);
    const AttemptOutcome failed = contract.FinishAttempt(
        begin.attempt, first, std::move(products));
    const BeginOutcome retry = contract.Retry();
    auto retry_products = IssueAll(contract, retry.attempt);
    AttemptReport overflow = PassingReport(12);
    overflow.telemetry.transfers = {1, 0, 0, 0, 0, true};
    const bool overflow_preflight =
        contract.CanAccountAttemptTransfers(overflow.telemetry.transfers);
    const AttemptOutcome outcome = contract.FinishAttempt(
        retry.attempt, overflow, std::move(retry_products));
    const BeginOutcome exhausted = contract.Retry();
    passed = Check(boundary_fits && failed.retry_available && retry.started &&
                       !overflow_preflight &&
                       !outcome.committed &&
                       outcome.reason == FailureReason::telemetry_gate &&
                       !exhausted.started &&
                       exhausted.reason == FailureReason::retry_exhausted &&
                       contract.telemetry().attempted_transfers
                               .host_to_device_bytes ==
                           std::numeric_limits<std::uint64_t>::max() &&
                       contract.telemetry().rolled_back_transfers
                               .host_to_device_bytes ==
                           std::numeric_limits<std::uint64_t>::max() &&
                       contract.telemetry().attempted_transfers.complete &&
                       contract.telemetry().rolled_back_transfers.complete &&
                       contract.telemetry()
                               .transfer_accounting_overflow_rejections == 1 &&
                       ArrayStoragePolicy::publications == 0,
                   "transfer_overflow_preflight_rejects_without_saturation") &&
        passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.originating_failure = FailureReason::backend_failure;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::backend_failure &&
                       report.gates.All(),
                   "non_none_origin_fails_closed_with_all_booleans_true") &&
        passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome first_begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidate(contract, first_begin.attempt, AcceptedSlot());
    auto first_products = IssueAll(contract, first_begin.attempt);
    const AttemptOutcome first = contract.FinishAttempt(
        first_begin.attempt, PassingReport(20), std::move(first_products));
    const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
    const StateCoordinates coordinates_before = contract.coordinates();
    Slot published_before{};
    const bool captured = CopyPublished(contract, published_before);
    const BeginOutcome second_begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidate(contract, second_begin.attempt, OtherSlot());
    auto second_products = IssueAll(contract, second_begin.attempt);
    AttemptReport failure = PassingReport(21);
    failure.gates.recovery = false;
    const AttemptOutcome second = contract.FinishAttempt(
        second_begin.attempt, failure, std::move(second_products));
    Slot published_after{};
    passed = Check(first.committed && captured && !second.committed &&
                       second.retry_available &&
                       SameCoordinates(contract.coordinates(),
                                       coordinates_before) &&
                       SameAccepted(contract.telemetry().accepted,
                                    accepted_before) &&
                       CopyPublished(contract, published_after) &&
                       published_after.interior == published_before.interior &&
                       published_after.trace == published_before.trace,
                   "failed_later_attempt_preserves_last_accepted_publication") &&
        passed;
  }

  for (const int prohibited_field : {0, 1, 2, 3, 4})
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    if (prohibited_field == 0)
    {
      report.telemetry.prohibited.dt_change_events = 1;
    }
    else if (prohibited_field == 1)
    {
      report.telemetry.prohibited.clipping_events = 1;
    }
    else if (prohibited_field == 2)
    {
      report.telemetry.prohibited.solver_substitution_events = 1;
    }
    else if (prohibited_field == 3)
    {
      report.telemetry.prohibited.fallback_events = 1;
    }
    else
    {
      report.telemetry.prohibited.hidden_insertion_events = 1;
    }
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::telemetry_gate,
                   "prohibited_recovery_actions_fail_telemetry_gate") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.telemetry.transfers.allocation_events = 1;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::telemetry_gate,
                   "persistent_run_rejects_per_step_allocation") && passed;
  }

  const std::array<CollectiveFault, 17> identity_faults{{
      CollectiveFault::finish_identity_false,
      CollectiveFault::finish_failure_consistency_false,
      CollectiveFault::finish_participants,
      CollectiveFault::finish_attempt,
      CollectiveFault::finish_committed_time,
      CollectiveFault::finish_committed_step,
      CollectiveFault::finish_committed_generation,
      CollectiveFault::finish_committed_cursor,
      CollectiveFault::finish_committed_dt,
      CollectiveFault::finish_target_time,
      CollectiveFault::finish_target_step,
      CollectiveFault::finish_target_generation,
      CollectiveFault::finish_target_cursor,
      CollectiveFault::finish_target_dt,
      CollectiveFault::finish_candidate_cursor,
      CollectiveFault::finish_hide_local_failure,
      CollectiveFault::finish_change_local_failure}};
  for (const CollectiveFault fault : identity_faults)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    if (fault == CollectiveFault::finish_hide_local_failure ||
        fault == CollectiveFault::finish_change_local_failure)
    {
      report.gates.finite = false;
    }
    TestCollectivePolicy::fault = fault;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::collective_protocol,
                   "complete_finish_context_census_and_truth_are_reduced") &&
        passed;
  }
  return passed;
}

bool CheckTwoPhaseDetectionLegacyAndReentrantAdmission()
{
  bool passed = true;
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const bool stored = StoreCandidate(contract, begin.attempt, AcceptedSlot());
    auto products = IssueAll(contract, begin.attempt);
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    passed = Check(begin.started && stored && outcome.committed &&
                       TestCollectivePolicy::reductions == 1 &&
                       ArrayStoragePolicy::publications == 1 &&
                       contract.telemetry().commits == 1,
                   "legacy_policy_keeps_one_reduce_immediate_finish") &&
        passed;
  }

  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const auto candidate = contract.AcquireCandidate(begin.attempt);
    const bool stored = candidate &&
        contract.StoreCandidate(*candidate, AcceptedSlot());
    const std::uint64_t discards_before = ArrayStoragePolicy::discards;
    const BeginOutcome nested_begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const BeginOutcome nested_retry = contract.Retry();
    CheckpointDescriptor ignored_checkpoint{};
    const RestoreOutcome nested_restore =
        contract.Restore(ignored_checkpoint, OtherSlot());
    Slot candidate_after{};
    const bool candidate_preserved = candidate &&
        contract.CopyCandidate(*candidate, candidate_after);
    auto products = IssueAll(contract, begin.attempt);
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    passed = Check(
                 begin.started && stored && !nested_begin.started &&
                     nested_begin.reason == FailureReason::stale_attempt &&
                     !nested_retry.started &&
                     nested_retry.reason == FailureReason::retry_exhausted &&
                     !nested_restore.restored &&
                     nested_restore.compatibility ==
                         RestartCompatibility::contract_busy &&
                     candidate_preserved &&
                     SameSlotBytes(candidate_after, AcceptedSlot()) &&
                     ArrayStoragePolicy::discards == discards_before &&
                     TestCollectivePolicy::admissions == 3 &&
                     TestCollectivePolicy::restores == 1 &&
                     outcome.committed &&
                     TestCollectivePolicy::reductions == 1,
                 "reentrant_admission_and_restore_preserve_live_mutable_candidate") &&
        passed;
  }
  return passed;
}

template <bool TwoPhase>
bool CheckDeclaredResidualLimitPath()
{
  using TestedContract = std::conditional_t<TwoPhase, TwoPhaseContract, Contract>;
  bool passed = true;
  for (const double limit : {1.0e-10, 1.0e-6, 1.0e-4})
  {
    for (const double residual : {
             0.0, limit * 0.5, limit,
             std::nextafter(limit, std::numeric_limits<double>::infinity()),
             -1.0, std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::quiet_NaN()})
    {
      TwoPhaseArrayStoragePolicy::Reset();
      TestCollectivePolicy::Reset();
      TestedContract contract{CommittedSlot(), ScratchSlot(),
          InitialCoordinates(), Resources(), 0, TestCollectivePolicy{}, limit};
      const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
      const bool stored = StoreCandidateFor(contract, begin.attempt, AcceptedSlot());
      auto products = IssueAllFor(contract, begin.attempt);
      AttemptReport report = PassingReport(12);
      report.telemetry.true_relative_residual = residual;
      const bool expected = std::isfinite(residual) && residual >= 0.0 &&
          residual <= limit;
      bool committed = false;
      if constexpr (TwoPhase)
      {
        auto staging = contract.FreezeCandidate(begin.attempt, 12);
        auto prepared = staging
            ? contract.PrepareCommit(std::move(*staging), report, std::move(products))
            : std::optional<typename TestedContract::PreparedCommitToken>{};
        if (prepared)
        {
          auto provisional = contract.CommitPrepared(std::move(*prepared));
          committed = provisional &&
              contract.FinalizeCommit(std::move(*provisional)).committed;
        }
      }
      else
      {
        committed = contract.FinishAttempt(
            begin.attempt, report, std::move(products)).committed;
      }
      passed = Check(begin.started && stored && committed == expected &&
          (expected
              ? contract.telemetry().accepted.attempt.true_relative_residual == residual
              : contract.last_failure() == FailureReason::true_residual_gate &&
                ArrayStoragePolicy::publications == 0),
          TwoPhase ? "declared_residual_limit_two_phase_unclamped_and_enforced"
                   : "declared_residual_limit_finish_unclamped_and_enforced") && passed;
    }
  }
  for (const double invalid : {0.0, -1.0,
           std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN()})
  {
    TwoPhaseArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    TestedContract contract{CommittedSlot(), ScratchSlot(),
        InitialCoordinates(), Resources(), 0, TestCollectivePolicy{}, invalid};
    passed = Check(contract.status() == RunStatus::invalid_request &&
        contract.last_failure() == FailureReason::invalid_resource_descriptor &&
        ArrayStoragePolicy::publications == 0,
        "invalid_declared_residual_limit_fails_closed") && passed;
  }
  return passed;
}

bool CheckDeclaredResidualLimits()
{
  static_assert(Contract::true_relative_residual_limit == 1.0e-10,
                "legacy publication ceiling remains unchanged");
  bool passed = CheckDeclaredResidualLimitPath<false>();
  passed = CheckDeclaredResidualLimitPath<true>() && passed;
  // The real prepared-commit route must retain its strict default even when
  // the producer reports every boolean gate as passing.
  TwoPhaseContract contract = MakeTwoPhaseContract();
  const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
  const bool stored = StoreCandidateFor(contract, begin.attempt, AcceptedSlot());
  auto products = IssueAllFor(contract, begin.attempt);
  auto staging = contract.FreezeCandidate(begin.attempt, 12);
  AttemptReport report = PassingReport(12);
  report.telemetry.true_relative_residual = 1.0e-8;
  auto prepared = staging
      ? contract.PrepareCommit(std::move(*staging), report, std::move(products))
      : std::optional<TwoPhaseContract::PreparedCommitToken>{};
  return Check(begin.started && stored && staging && !prepared &&
      contract.last_failure() == FailureReason::true_residual_gate &&
      ArrayStoragePolicy::publications == 0,
      "default_two_phase_limit_stays_strict") && passed;
}

bool CheckTwoPhaseFreezePrepareCommitFinalize()
{
  bool passed = true;
  TwoPhaseContract contract = MakeTwoPhaseContract();
  const auto old = contract.Published();
  Slot old_before{};
  StateCoordinates old_coordinates{};
  const bool old_captured = contract.CopyPublished(old, old_before) &&
      contract.PublishedCoordinates(old, old_coordinates);
  const BeginOutcome begin =
      contract.BeginStep({InteriorSpace::l2, 0.125});
  const auto mutable_capability = contract.AcquireCandidate(begin.attempt);
  const bool stored = mutable_capability &&
      contract.StoreCandidate(*mutable_capability, AcceptedSlot());
  auto products = IssueAllFor(contract, begin.attempt);

  const RunTelemetry telemetry_before_freeze = contract.telemetry();
  const StateCoordinates coordinates_before_freeze = contract.coordinates();
  const std::uint64_t reductions_before = TestCollectivePolicy::reductions;
  const std::uint64_t publications_before = ArrayStoragePolicy::publications;
  auto staging = contract.FreezeCandidate(begin.attempt, 12);
  const CheckpointDescriptor future = staging
      ? staging->future_checkpoint()
      : CheckpointDescriptor{};
  const StateCoordinates expected_target{1.125, 5, 8, 12, 0.125};
  Slot frozen_copy{};
  const bool copied_frozen = staging &&
      contract.CopyPrepared(*staging, frozen_copy);
  CheckpointDescriptor premature_checkpoint{};
  const bool premature_checkpoint_rejected =
      !contract.TryMakeCheckpoint(premature_checkpoint);

  const auto blocked_acquire = contract.AcquireCandidate(begin.attempt);
  const bool blocked_store = mutable_capability &&
      !contract.StoreCandidate(*mutable_capability, OtherSlot());
  Slot blocked_copy{};
  const bool blocked_candidate_copy = mutable_capability &&
      !contract.CopyCandidate(*mutable_capability, blocked_copy);
  const auto blocked_product =
      contract.IssueProduct<DerivedProductKind::residual>(begin.attempt);

  passed = Check(
               old_captured && begin.started && stored && staging &&
                   copied_frozen && SameSlotBytes(frozen_copy, AcceptedSlot()) &&
                   !blocked_acquire && blocked_store &&
                   blocked_candidate_copy && !blocked_product &&
                   premature_checkpoint_rejected &&
                   staging->entry_coordinates().step ==
                       coordinates_before_freeze.step &&
                   SameCoordinates(staging->target_coordinates(),
                                   expected_target) &&
                   staging->interior_hash() ==
                       ArrayStoragePolicy::HashInterior(
                           AcceptedSlot().interior) &&
                   staging->trace_hash() ==
                       ArrayStoragePolicy::HashTrace(AcceptedSlot().trace) &&
                   future.successful_commit && future.cursor == 12 &&
                   SameCoordinates(future.coordinates, expected_target) &&
                   TestCollectivePolicy::reductions == reductions_before &&
                   ArrayStoragePolicy::publications == publications_before &&
                   SameCoordinates(contract.coordinates(),
                                   coordinates_before_freeze) &&
                   contract.status() == RunStatus::attempt_active &&
                   SameTransfers(contract.telemetry().attempted_transfers,
                                 telemetry_before_freeze.attempted_transfers) &&
                   SameTransfers(contract.telemetry().rolled_back_transfers,
                                 telemetry_before_freeze.rolled_back_transfers) &&
                   SameTransfers(contract.telemetry().committed_transfers,
                                 telemetry_before_freeze.committed_transfers),
               "freeze_is_zero_effect_and_blocks_mutation_product_checkpoint") &&
      passed;

  if (!staging) { return passed; }
  TwoPhaseContract::StagingToken moved_staging{std::move(*staging)};
  Slot moved_from_copy{};
  const bool moved_from_rejected =
      !contract.CopyPrepared(*staging, moved_from_copy);
  const AttemptReport report = PassingReport(12);
  auto prepared = contract.PrepareCommit(
      std::move(moved_staging), report, std::move(products));
  const CheckpointDescriptor prepared_descriptor = prepared
      ? prepared->checkpoint_descriptor()
      : CheckpointDescriptor{};
  Slot old_after_prepare{};
  StateCoordinates coordinates_after_prepare{};
  const bool old_visible_after_prepare =
      contract.CopyPublished(old, old_after_prepare) &&
      contract.PublishedCoordinates(old, coordinates_after_prepare);
  passed = Check(
               moved_from_rejected && prepared &&
                   TestCollectivePolicy::reductions == reductions_before + 1 &&
                   ArrayStoragePolicy::publications == publications_before &&
                   old_visible_after_prepare &&
                   SameSlotBytes(old_before, old_after_prepare) &&
                   SameCoordinates(old_coordinates, coordinates_after_prepare) &&
                   SameCoordinates(contract.coordinates(), old_coordinates) &&
                   contract.status() == RunStatus::attempt_active &&
                   contract.telemetry().commits == 0 &&
                   !contract.telemetry().accepted.valid &&
                   SameTransfers(contract.telemetry().attempted_transfers,
                                 report.telemetry.transfers) &&
                   contract.telemetry().rolled_back_transfers
                           .host_to_device_bytes == 0 &&
                   contract.telemetry().committed_transfers
                           .host_to_device_bytes == 0,
               "prepare_reduces_accounts_once_without_publication_or_success") &&
      passed;
  if (!prepared) { return passed; }

  const std::uint64_t prepublish_checks_before =
      ArrayStoragePolicy::prepublish_checks;
  const std::uint64_t discards_before = ArrayStoragePolicy::discards;
  auto provisional = contract.CommitPrepared(std::move(*prepared));
  auto duplicate_exchange =
      contract.CommitPrepared(std::move(*prepared));
  Slot old_during_provisional{};
  StateCoordinates coordinates_during_provisional{};
  const bool old_visible_during_provisional =
      contract.CopyPublished(old, old_during_provisional) &&
      contract.PublishedCoordinates(old, coordinates_during_provisional);
  passed = Check(
               provisional && !duplicate_exchange &&
                   old_visible_during_provisional &&
                   SameSlotBytes(old_before, old_during_provisional) &&
                   SameCoordinates(old_coordinates,
                                   coordinates_during_provisional) &&
                   SameCoordinates(contract.coordinates(), old_coordinates) &&
                   ArrayStoragePolicy::publications == publications_before + 1 &&
                   ArrayStoragePolicy::prepublish_checks ==
                       prepublish_checks_before &&
                   ArrayStoragePolicy::discards == discards_before &&
                   contract.telemetry().commits == 0 &&
                   !contract.telemetry().accepted.valid &&
                   contract.telemetry().committed_transfers
                           .host_to_device_bytes == 0 &&
                   contract.status() == RunStatus::attempt_active &&
                   !contract.TryMakeCheckpoint(premature_checkpoint),
               "provisional_exchange_preserves_all_old_public_state") &&
      passed;
  if (!provisional) { return passed; }

  const AttemptOutcome finalized =
      contract.FinalizeCommit(std::move(*provisional));
  const std::uint64_t commits_after_finalize = contract.telemetry().commits;
  const TransferAccounting committed_after_finalize =
      contract.telemetry().committed_transfers;
  const AttemptOutcome duplicate_finalize =
      contract.FinalizeCommit(std::move(*provisional));
  Slot published_after{};
  Slot stale_old{};
  CheckpointDescriptor final_checkpoint{};
  const bool final_checkpoint_created =
      contract.TryMakeCheckpoint(final_checkpoint);
  passed = Check(
               finalized.committed &&
                   finalized.status == RunStatus::committed &&
                   duplicate_finalize.reason == FailureReason::stale_attempt &&
                   !duplicate_finalize.committed &&
                   contract.telemetry().commits == commits_after_finalize &&
                   commits_after_finalize == 1 &&
                   SameTransfers(committed_after_finalize,
                                 report.telemetry.transfers) &&
                   SameTransfers(contract.telemetry().committed_transfers,
                                 committed_after_finalize) &&
                   SameCoordinates(contract.coordinates(), expected_target) &&
                   !contract.CopyPublished(old, stale_old) &&
                   CopyPublishedFor(contract, published_after) &&
                   SameSlotBytes(published_after, AcceptedSlot()) &&
                   final_checkpoint_created &&
                   detail::SameCheckpoint(future, prepared_descriptor) &&
                   detail::SameCheckpoint(future, final_checkpoint),
               "finalize_is_sole_exactly_once_logical_success_and_descriptor_match") &&
      passed;
  return passed;
}

bool CheckTwoPhasePrepareFailuresAbortAndRevert()
{
  bool passed = true;
  for (const bool remote_failure : {false, true})
  {
    TwoPhaseContract contract = MakeTwoPhaseContract();
    const auto old = contract.Published();
    Slot old_before{};
    const bool captured = contract.CopyPublished(old, old_before);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const bool stored =
        StoreCandidateFor(contract, begin.attempt, AcceptedSlot());
    auto products = IssueAllFor(contract, begin.attempt);
    auto staging = contract.FreezeCandidate(begin.attempt, 12);
    AttemptReport report = PassingReport(12);
    FailureReason expected = FailureReason::backend_failure;
    if (remote_failure)
    {
      TestCollectivePolicy::fault = CollectiveFault::finish_remote_failure;
      TestCollectivePolicy::remote_reason =
          FailureReason::admissibility_gate;
      expected = FailureReason::admissibility_gate;
    }
    else
    {
      report.originating_failure = FailureReason::backend_failure;
    }
    auto prepared = staging
        ? contract.PrepareCommit(
              std::move(*staging), report, std::move(products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    Slot old_after{};
    passed = Check(
                 captured && begin.started && stored && staging && !prepared &&
                     contract.status() == RunStatus::retry_available &&
                     contract.last_failure() == expected &&
                     TestCollectivePolicy::reductions == 1 &&
                     ArrayStoragePolicy::publications == 0 &&
                     contract.telemetry().attempts_failed == 1 &&
                     contract.telemetry().rollbacks == 1 &&
                     SameTransfers(contract.telemetry().attempted_transfers,
                                   report.telemetry.transfers) &&
                     SameTransfers(contract.telemetry().rolled_back_transfers,
                                   report.telemetry.transfers) &&
                     contract.telemetry().committed_transfers
                             .host_to_device_bytes == 0 &&
                     contract.CopyPublished(old, old_after) &&
                     SameSlotBytes(old_before, old_after) &&
                     SameCoordinates(contract.coordinates(),
                                     InitialCoordinates()),
                 remote_failure
                     ? "remote_prepare_failure_uses_ordinary_rollback_retry"
                     : "originating_backend_prepare_failure_uses_ordinary_rollback_retry") &&
        passed;
  }

  {
    TwoPhaseContract contract = MakeTwoPhaseContract();
    const auto old = contract.Published();
    Slot old_before{};
    const bool captured = contract.CopyPublished(old, old_before);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const bool stored =
        StoreCandidateFor(contract, begin.attempt, AcceptedSlot());
    auto products = IssueAllFor(contract, begin.attempt);
    auto staging = contract.FreezeCandidate(begin.attempt, 12);
    const AttemptReport report = PassingReport(12);
    auto prepared = staging
        ? contract.PrepareCommit(
              std::move(*staging), report, std::move(products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    const AttemptOutcome aborted = prepared
        ? contract.AbortPrepared(
              std::move(*prepared), FailureReason::backend_failure)
        : AttemptOutcome{};
    const std::uint64_t rollbacks_after_abort =
        contract.telemetry().rollbacks;
    const AttemptOutcome duplicate_abort = prepared
        ? contract.AbortPrepared(std::move(*prepared))
        : AttemptOutcome{};
    Slot old_after{};
    const BeginOutcome retry = contract.Retry();
    const auto retry_candidate = contract.AcquireCandidate(retry.attempt);
    Slot retry_base{};
    const bool retry_uses_old = retry_candidate &&
        contract.CopyCandidate(*retry_candidate, retry_base);
    passed = Check(
                 captured && begin.started && stored && staging && prepared &&
                     aborted.retry_available && !aborted.committed &&
                     duplicate_abort.reason == FailureReason::stale_attempt &&
                     contract.telemetry().rollbacks == rollbacks_after_abort &&
                     rollbacks_after_abort == 1 &&
                     TestCollectivePolicy::reductions == 1 &&
                     ArrayStoragePolicy::publications == 0 &&
                     SameTransfers(contract.telemetry().attempted_transfers,
                                   report.telemetry.transfers) &&
                     SameTransfers(contract.telemetry().rolled_back_transfers,
                                   report.telemetry.transfers) &&
                     contract.telemetry().committed_transfers
                             .host_to_device_bytes == 0 &&
                     contract.CopyPublished(old, old_after) &&
                     SameSlotBytes(old_before, old_after) && retry.started &&
                     retry_uses_old && SameSlotBytes(retry_base, old_before),
                 "abort_prepared_rolls_back_accounts_once_and_retries_old") &&
        passed;
  }

  {
    TwoPhaseContract contract = MakeTwoPhaseContract();
    const auto old = contract.Published();
    Slot old_before{};
    const bool captured = contract.CopyPublished(old, old_before);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const bool stored =
        StoreCandidateFor(contract, begin.attempt, AcceptedSlot());
    auto products = IssueAllFor(contract, begin.attempt);
    auto staging = contract.FreezeCandidate(begin.attempt, 12);
    const AttemptReport report = PassingReport(12);
    auto prepared = staging
        ? contract.PrepareCommit(
              std::move(*staging), report, std::move(products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    auto provisional = prepared
        ? contract.CommitPrepared(std::move(*prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    Slot old_provisional{};
    const bool old_still_visible = provisional &&
        contract.CopyPublished(old, old_provisional);
    const AttemptOutcome reverted = provisional
        ? contract.RevertCommit(
              std::move(*provisional), FailureReason::backend_failure)
        : AttemptOutcome{};
    const std::uint64_t reversions_after =
        TwoPhaseArrayStoragePolicy::reversions;
    const std::uint64_t rollbacks_after = contract.telemetry().rollbacks;
    const AttemptOutcome duplicate_revert = provisional
        ? contract.RevertCommit(std::move(*provisional))
        : AttemptOutcome{};
    Slot old_after{};
    passed = Check(
                 captured && begin.started && stored && staging && prepared &&
                     provisional && old_still_visible &&
                     SameSlotBytes(old_before, old_provisional) &&
                     reverted.retry_available && !reverted.committed &&
                     duplicate_revert.reason == FailureReason::stale_attempt &&
                     TwoPhaseArrayStoragePolicy::reversions ==
                         reversions_after &&
                     reversions_after == 1 &&
                     contract.telemetry().rollbacks == rollbacks_after &&
                     rollbacks_after == 1 &&
                     ArrayStoragePolicy::publications == 1 &&
                     SameSlotBytes(
                         TwoPhaseArrayStoragePolicy::reverted_committed,
                         old_before) &&
                     SameSlotBytes(
                         TwoPhaseArrayStoragePolicy::reverted_candidate,
                         AcceptedSlot()) &&
                     contract.CopyPublished(old, old_after) &&
                     SameSlotBytes(old_before, old_after) &&
                     SameCoordinates(contract.coordinates(),
                                     InitialCoordinates()) &&
                     SameTransfers(contract.telemetry().attempted_transfers,
                                   report.telemetry.transfers) &&
                     SameTransfers(contract.telemetry().rolled_back_transfers,
                                   report.telemetry.transfers) &&
                     contract.telemetry().committed_transfers
                             .host_to_device_bytes == 0,
                 "revert_inverse_precedes_exact_once_ordinary_rollback") &&
        passed;
  }
  return passed;
}

bool CheckTwoPhaseBindingsTamperAndOverflow()
{
  bool passed = true;
  {
    TwoPhaseContract contract = MakeTwoPhaseContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(contract, begin.attempt, AcceptedSlot());
    auto products = IssueAllFor(contract, begin.attempt);
    auto staging = contract.FreezeCandidate(begin.attempt, 12);
    auto prepared = staging
        ? contract.PrepareCommit(
              std::move(*staging), PassingReport(13), std::move(products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    passed = Check(staging && !prepared &&
                       contract.last_failure() == FailureReason::stale_attempt &&
                       contract.status() == RunStatus::retry_available &&
                       TestCollectivePolicy::reductions == 1 &&
                       ArrayStoragePolicy::publications == 0,
                   "frozen_target_and_report_cursor_are_exactly_bound") &&
        passed;
  }

  {
    TwoPhaseContract contract = MakeTwoPhaseContract();
    const auto old = contract.Published();
    Slot old_before{};
    const bool captured = contract.CopyPublished(old, old_before);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(contract, begin.attempt, AcceptedSlot());
    auto products = IssueAllFor(contract, begin.attempt);
    auto staging = contract.FreezeCandidate(begin.attempt, 12);
    if (TwoPhaseArrayStoragePolicy::live_candidate)
    {
      TwoPhaseArrayStoragePolicy::live_candidate->interior[0] += 1.0;
    }
    Slot tampered_copy{};
    const bool copy_rejected = staging &&
        !contract.CopyPrepared(*staging, tampered_copy);
    auto prepared = staging
        ? contract.PrepareCommit(
              std::move(*staging), PassingReport(12), std::move(products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    Slot old_after{};
    passed = Check(captured && staging && copy_rejected && !prepared &&
                       contract.last_failure() ==
                           FailureReason::candidate_preparation &&
                       contract.status() == RunStatus::retry_available &&
                       ArrayStoragePolicy::publications == 0 &&
                       contract.CopyPublished(old, old_after) &&
                       SameSlotBytes(old_before, old_after),
                   "frozen_candidate_hash_tamper_fails_before_publication") &&
        passed;
  }

  {
    TwoPhaseContract contract = MakeTwoPhaseContract();
    const BeginOutcome first =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto first_products = IssueAllFor(contract, first.attempt);
    AttemptReport boundary = PassingReport(12);
    boundary.gates.finite = false;
    boundary.telemetry.transfers.host_to_device_bytes =
        std::numeric_limits<std::uint64_t>::max();
    const AttemptOutcome failed = contract.FinishAttempt(
        first.attempt, boundary, std::move(first_products));
    const BeginOutcome retry = contract.Retry();
    StoreCandidateFor(contract, retry.attempt, AcceptedSlot());
    auto retry_products = IssueAllFor(contract, retry.attempt);
    auto staging = contract.FreezeCandidate(retry.attempt, 12);
    AttemptReport overflow = PassingReport(12);
    overflow.telemetry.transfers = {1, 0, 0, 0, 0, true};
    auto prepared = staging
        ? contract.PrepareCommit(
              std::move(*staging), overflow, std::move(retry_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    passed = Check(failed.retry_available && retry.started && staging &&
                       !prepared &&
                       contract.last_failure() == FailureReason::telemetry_gate &&
                       contract.status() == RunStatus::terminal_failure &&
                       contract.telemetry().attempted_transfers
                               .host_to_device_bytes ==
                           std::numeric_limits<std::uint64_t>::max() &&
                       contract.telemetry().rolled_back_transfers
                               .host_to_device_bytes ==
                           std::numeric_limits<std::uint64_t>::max() &&
                       contract.telemetry()
                               .transfer_accounting_overflow_rejections == 1 &&
                       ArrayStoragePolicy::publications == 0,
                   "two_phase_transfer_overflow_fails_without_saturation") &&
        passed;
  }
  return passed;
}

bool CheckTwoPhaseTokenOwnershipABAAndBusyReentrancy()
{
  bool passed = true;
  {
    TwoPhaseContract first = MakeTwoPhaseContract();
    const BeginOutcome first_begin =
        first.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(first, first_begin.attempt, AcceptedSlot());
    auto first_staging = first.FreezeCandidate(first_begin.attempt, 12);

    TwoPhaseContract second = MakeTwoPhaseContract();
    const BeginOutcome second_begin =
        second.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(second, second_begin.attempt, AcceptedSlot());
    auto products = IssueAllFor(second, second_begin.attempt);
    auto second_staging = second.FreezeCandidate(second_begin.attempt, 12);
    Slot foreign_copy{};
    const bool foreign_copy_rejected = first_staging &&
        !second.CopyPrepared(*first_staging, foreign_copy);
    auto foreign_prepare = first_staging
        ? second.PrepareCommit(std::move(*first_staging),
                               PassingReport(12),
                               std::move(products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    Slot first_after_foreign{};
    const bool first_token_survives_foreign_owner = first_staging &&
        first.CopyPrepared(*first_staging, first_after_foreign);
    auto prepared = second_staging
        ? second.PrepareCommit(std::move(*second_staging),
                               PassingReport(12),
                               std::move(products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    const AttemptOutcome aborted = prepared
        ? second.AbortPrepared(std::move(*prepared))
        : AttemptOutcome{};
    passed = Check(
                 first_staging && second_staging && foreign_copy_rejected &&
                     !foreign_prepare && first_token_survives_foreign_owner &&
                     SameSlotBytes(first_after_foreign, AcceptedSlot()) &&
                     prepared && aborted.retry_available &&
                     TestCollectivePolicy::reductions == 1 &&
                     ArrayStoragePolicy::publications == 0,
                 "cross_owner_tokens_fail_closed_without_consuming_live_owner") &&
        passed;
  }

  {
    TwoPhaseContract first = MakeTwoPhaseContract();
    const BeginOutcome first_begin =
        first.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(first, first_begin.attempt, AcceptedSlot());
    auto first_products = IssueAllFor(first, first_begin.attempt);
    auto first_staging = first.FreezeCandidate(first_begin.attempt, 12);
    Slot first_candidate{};
    const bool first_candidate_captured = first_staging &&
        first.CopyPrepared(*first_staging, first_candidate);
    auto first_prepared = first_staging
        ? first.PrepareCommit(std::move(*first_staging),
                              PassingReport(12),
                              std::move(first_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};

    TwoPhaseContract second = MakeTwoPhaseContract();
    const BeginOutcome second_begin =
        second.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(second, second_begin.attempt, AcceptedSlot());
    auto second_products = IssueAllFor(second, second_begin.attempt);
    auto second_staging = second.FreezeCandidate(second_begin.attempt, 12);
    Slot second_candidate{};
    const bool second_candidate_captured = second_staging &&
        second.CopyPrepared(*second_staging, second_candidate);
    auto second_prepared = second_staging
        ? second.PrepareCommit(std::move(*second_staging),
                               PassingReport(12),
                               std::move(second_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    const bool exact_prepared_checkpoint = first_prepared && second_prepared &&
        detail::SameCheckpoint(first_prepared->checkpoint_descriptor(),
                               second_prepared->checkpoint_descriptor());
    const auto first_published = first.Published();
    const auto second_published = second.Published();
    Slot first_published_before{};
    Slot second_published_before{};
    const bool published_captured =
        first.CopyPublished(first_published, first_published_before) &&
        second.CopyPublished(second_published, second_published_before);

    const RunTelemetry first_accounting = first.telemetry();
    const RunTelemetry second_accounting = second.telemetry();
    const StateCoordinates first_coordinates = first.coordinates();
    const StateCoordinates second_coordinates = second.coordinates();
    const RunStatus first_status = first.status();
    const RunStatus second_status = second.status();
    const FailureReason first_failure = first.last_failure();
    const FailureReason second_failure = second.last_failure();
    const std::uint64_t publications = ArrayStoragePolicy::publications;
    const std::uint64_t discards = ArrayStoragePolicy::discards;
    const std::uint64_t stores = ArrayStoragePolicy::stores;
    const std::uint64_t preparations = ArrayStoragePolicy::preparations;
    const std::uint64_t restore_preparations =
        ArrayStoragePolicy::restore_preparations;
    const std::uint64_t copies = ArrayStoragePolicy::copies;
    const std::uint64_t prepublish_checks =
        ArrayStoragePolicy::prepublish_checks;
    const std::uint64_t reversions =
        TwoPhaseArrayStoragePolicy::reversions;
    auto rejected = first_prepared
        ? second.CommitPrepared(std::move(*first_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    const bool rejection_is_inert =
        !rejected && SameCommitAccounting(first.telemetry(), first_accounting) &&
        SameCommitAccounting(second.telemetry(), second_accounting) &&
        first.telemetry().stale_consumers_rejected ==
            first_accounting.stale_consumers_rejected &&
        second.telemetry().stale_consumers_rejected ==
            second_accounting.stale_consumers_rejected + 1 &&
        SameCoordinates(first.coordinates(), first_coordinates) &&
        SameCoordinates(second.coordinates(), second_coordinates) &&
        first.status() == first_status && second.status() == second_status &&
        first.last_failure() == first_failure &&
        second.last_failure() == second_failure &&
        ArrayStoragePolicy::publications == publications &&
        ArrayStoragePolicy::discards == discards &&
        ArrayStoragePolicy::stores == stores &&
        ArrayStoragePolicy::preparations == preparations &&
        ArrayStoragePolicy::restore_preparations == restore_preparations &&
        ArrayStoragePolicy::copies == copies &&
        ArrayStoragePolicy::prepublish_checks == prepublish_checks &&
        TwoPhaseArrayStoragePolicy::reversions == reversions;
    Slot first_published_after{};
    Slot second_published_after{};
    const bool published_is_unchanged =
        first.CopyPublished(first_published, first_published_after) &&
        second.CopyPublished(second_published, second_published_after) &&
        SameSlotBytes(first_published_before, first_published_after) &&
        SameSlotBytes(second_published_before, second_published_after);
    auto first_provisional = first_prepared
        ? first.CommitPrepared(std::move(*first_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    auto second_provisional = second_prepared
        ? second.CommitPrepared(std::move(*second_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    const AttemptOutcome first_reverted = first_provisional
        ? first.RevertCommit(std::move(*first_provisional))
        : AttemptOutcome{};
    const AttemptOutcome second_reverted = second_provisional
        ? second.RevertCommit(std::move(*second_provisional))
        : AttemptOutcome{};
    passed = Check(
                 first_prepared && second_prepared &&
                     first_candidate_captured && second_candidate_captured &&
                     SameSlotBytes(first_candidate, second_candidate) &&
                     SameAttempt(first_begin.attempt, second_begin.attempt) &&
                     exact_prepared_checkpoint && published_captured &&
                     rejection_is_inert && published_is_unchanged &&
                     first_provisional &&
                     second_provisional && first_reverted.retry_available &&
                     second_reverted.retry_available,
                 "prepared_cross_owner_rejects_without_mutation_or_authority_consumption") &&
        passed;
  }

  {
    TwoPhaseContract first = MakeTwoPhaseContract();
    const BeginOutcome first_begin =
        first.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(first, first_begin.attempt, AcceptedSlot());
    auto first_products = IssueAllFor(first, first_begin.attempt);
    auto first_staging = first.FreezeCandidate(first_begin.attempt, 12);
    auto first_prepared = first_staging
        ? first.PrepareCommit(std::move(*first_staging),
                              PassingReport(12),
                              std::move(first_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    const CheckpointDescriptor first_checkpoint = first_prepared
        ? first_prepared->checkpoint_descriptor()
        : CheckpointDescriptor{};
    auto first_provisional = first_prepared
        ? first.CommitPrepared(std::move(*first_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};

    TwoPhaseContract second = MakeTwoPhaseContract();
    const BeginOutcome second_begin =
        second.BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(second, second_begin.attempt, AcceptedSlot());
    auto second_products = IssueAllFor(second, second_begin.attempt);
    auto second_staging = second.FreezeCandidate(second_begin.attempt, 12);
    auto second_prepared = second_staging
        ? second.PrepareCommit(std::move(*second_staging),
                               PassingReport(12),
                               std::move(second_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    const CheckpointDescriptor second_checkpoint = second_prepared
        ? second_prepared->checkpoint_descriptor()
        : CheckpointDescriptor{};
    const bool exact_prepared_checkpoint = first_prepared && second_prepared &&
        detail::SameCheckpoint(first_checkpoint, second_checkpoint);
    auto second_provisional = second_prepared
        ? second.CommitPrepared(std::move(*second_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    const auto first_published = first.Published();
    const auto second_published = second.Published();
    Slot first_published_before{};
    Slot second_published_before{};
    const bool published_captured =
        first.CopyPublished(first_published, first_published_before) &&
        second.CopyPublished(second_published, second_published_before);

    const RunTelemetry first_accounting = first.telemetry();
    const RunTelemetry second_accounting = second.telemetry();
    const StateCoordinates first_coordinates = first.coordinates();
    const StateCoordinates second_coordinates = second.coordinates();
    const RunStatus first_status = first.status();
    const RunStatus second_status = second.status();
    const FailureReason first_failure = first.last_failure();
    const FailureReason second_failure = second.last_failure();
    const std::uint64_t publications = ArrayStoragePolicy::publications;
    const std::uint64_t discards = ArrayStoragePolicy::discards;
    const std::uint64_t stores = ArrayStoragePolicy::stores;
    const std::uint64_t preparations = ArrayStoragePolicy::preparations;
    const std::uint64_t restore_preparations =
        ArrayStoragePolicy::restore_preparations;
    const std::uint64_t copies = ArrayStoragePolicy::copies;
    const std::uint64_t prepublish_checks =
        ArrayStoragePolicy::prepublish_checks;
    const std::uint64_t reversions =
        TwoPhaseArrayStoragePolicy::reversions;
    const AttemptOutcome rejected = first_provisional
        ? second.FinalizeCommit(std::move(*first_provisional))
        : AttemptOutcome{};
    const bool rejection_is_inert =
        rejected.reason == FailureReason::stale_attempt &&
        SameCommitAccounting(first.telemetry(), first_accounting) &&
        SameCommitAccounting(second.telemetry(), second_accounting) &&
        first.telemetry().stale_consumers_rejected ==
            first_accounting.stale_consumers_rejected &&
        second.telemetry().stale_consumers_rejected ==
            second_accounting.stale_consumers_rejected + 1 &&
        SameCoordinates(first.coordinates(), first_coordinates) &&
        SameCoordinates(second.coordinates(), second_coordinates) &&
        first.status() == first_status && second.status() == second_status &&
        first.last_failure() == first_failure &&
        second.last_failure() == second_failure &&
        ArrayStoragePolicy::publications == publications &&
        ArrayStoragePolicy::discards == discards &&
        ArrayStoragePolicy::stores == stores &&
        ArrayStoragePolicy::preparations == preparations &&
        ArrayStoragePolicy::restore_preparations == restore_preparations &&
        ArrayStoragePolicy::copies == copies &&
        ArrayStoragePolicy::prepublish_checks == prepublish_checks &&
        TwoPhaseArrayStoragePolicy::reversions == reversions;
    Slot first_published_after{};
    Slot second_published_after{};
    const bool published_is_unchanged =
        first.CopyPublished(first_published, first_published_after) &&
        second.CopyPublished(second_published, second_published_after) &&
        SameSlotBytes(first_published_before, first_published_after) &&
        SameSlotBytes(second_published_before, second_published_after);
    const AttemptOutcome first_finalized = first_provisional
        ? first.FinalizeCommit(std::move(*first_provisional))
        : AttemptOutcome{};
    const AttemptOutcome second_finalized = second_provisional
        ? second.FinalizeCommit(std::move(*second_provisional))
        : AttemptOutcome{};
    passed = Check(first_provisional && second_provisional &&
                       SameAttempt(first_begin.attempt, second_begin.attempt) &&
                       exact_prepared_checkpoint && published_captured &&
                       rejection_is_inert && published_is_unchanged &&
                       first_finalized.committed &&
                       second_finalized.committed,
                   "provisional_cross_owner_rejects_without_mutation_or_authority_consumption") &&
        passed;
  }

  {
    alignas(TwoPhaseContract)
        std::array<unsigned char, sizeof(TwoPhaseContract)> storage{};
    TwoPhaseArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    TwoPhaseContract* first = new (storage.data()) TwoPhaseContract{
        CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
    const BeginOutcome first_begin =
        first->BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(*first, first_begin.attempt, AcceptedSlot());
    auto old_staging = first->FreezeCandidate(first_begin.attempt, 12);
    Slot old_frozen_copy{};
    const bool old_frozen_captured = old_staging &&
        first->CopyPrepared(*old_staging, old_frozen_copy);
    const StateCoordinates old_entry = old_staging
        ? old_staging->entry_coordinates()
        : StateCoordinates{};
    const StateCoordinates old_target = old_staging
        ? old_staging->target_coordinates()
        : StateCoordinates{};
    const CheckpointDescriptor old_checkpoint = old_staging
        ? old_staging->checkpoint_descriptor()
        : CheckpointDescriptor{};
    first->~TwoPhaseContract();

    TwoPhaseArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    TwoPhaseContract* second = new (storage.data()) TwoPhaseContract{
        CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
    const BeginOutcome second_begin =
        second->BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(*second, second_begin.attempt, AcceptedSlot());
    auto new_staging = second->FreezeCandidate(second_begin.attempt, 12);
    const bool exact_attempt = SameAttempt(first_begin.attempt,
                                           second_begin.attempt);
    const bool exact_entry = old_staging && new_staging &&
        SameCoordinates(old_entry, new_staging->entry_coordinates());
    const bool exact_target = old_staging && new_staging &&
        SameCoordinates(old_target, new_staging->target_coordinates());
    const bool exact_checkpoint = old_staging && new_staging &&
        detail::SameCheckpoint(old_checkpoint,
                               new_staging->checkpoint_descriptor());
    Slot stale_copy{};
    Slot fresh_copy{};
    const bool stale_rejected = old_staging &&
        !second->CopyPrepared(*old_staging, stale_copy);
    const bool fresh_accepted = new_staging &&
        second->CopyPrepared(*new_staging, fresh_copy);
    passed = Check(old_staging && new_staging && old_frozen_captured &&
                       exact_attempt && exact_entry && exact_target &&
                       exact_checkpoint && stale_rejected && fresh_accepted &&
                       SameSlotBytes(old_frozen_copy, fresh_copy) &&
                       SameSlotBytes(fresh_copy, AcceptedSlot()),
                   "same_address_incarnation_aba_rejects_old_staging_token") &&
        passed;
    second->~TwoPhaseContract();
  }

  {
    alignas(TwoPhaseContract)
        std::array<unsigned char, sizeof(TwoPhaseContract)> storage{};
    TwoPhaseArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    TwoPhaseContract* first = new (storage.data()) TwoPhaseContract{
        CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
    const BeginOutcome first_begin =
        first->BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(*first, first_begin.attempt, AcceptedSlot());
    auto first_products = IssueAllFor(*first, first_begin.attempt);
    auto first_staging = first->FreezeCandidate(first_begin.attempt, 12);
    Slot first_candidate{};
    const bool first_candidate_captured = first_staging &&
        first->CopyPrepared(*first_staging, first_candidate);
    const StateCoordinates first_entry = first_staging
        ? first_staging->entry_coordinates()
        : StateCoordinates{};
    const StateCoordinates first_target = first_staging
        ? first_staging->target_coordinates()
        : StateCoordinates{};
    const CheckpointDescriptor first_checkpoint = first_staging
        ? first_staging->checkpoint_descriptor()
        : CheckpointDescriptor{};
    auto old_prepared = first_staging
        ? first->PrepareCommit(std::move(*first_staging),
                               PassingReport(12),
                               std::move(first_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    first->~TwoPhaseContract();

    TwoPhaseArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    TwoPhaseContract* second = new (storage.data()) TwoPhaseContract{
        CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
    const BeginOutcome second_begin =
        second->BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(*second, second_begin.attempt, AcceptedSlot());
    auto second_products = IssueAllFor(*second, second_begin.attempt);
    auto second_staging = second->FreezeCandidate(second_begin.attempt, 12);
    Slot second_candidate{};
    const bool second_candidate_captured = second_staging &&
        second->CopyPrepared(*second_staging, second_candidate);
    const bool exact_entry = second_staging &&
        SameCoordinates(first_entry, second_staging->entry_coordinates());
    const bool exact_target = second_staging &&
        SameCoordinates(first_target, second_staging->target_coordinates());
    const bool exact_checkpoint = second_staging &&
        detail::SameCheckpoint(first_checkpoint,
                               second_staging->checkpoint_descriptor());
    auto new_prepared = second_staging
        ? second->PrepareCommit(std::move(*second_staging),
                                PassingReport(12),
                                std::move(second_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    const bool exact_prepared_checkpoint = new_prepared &&
        detail::SameCheckpoint(first_checkpoint,
                               new_prepared->checkpoint_descriptor());
    const auto published = second->Published();
    Slot published_before{};
    const bool published_captured =
        second->CopyPublished(published, published_before);

    const RunTelemetry accounting_before = second->telemetry();
    const StateCoordinates coordinates_before = second->coordinates();
    const RunStatus status_before = second->status();
    const FailureReason failure_before = second->last_failure();
    const std::uint64_t publications = ArrayStoragePolicy::publications;
    const std::uint64_t discards = ArrayStoragePolicy::discards;
    const std::uint64_t stores = ArrayStoragePolicy::stores;
    const std::uint64_t preparations = ArrayStoragePolicy::preparations;
    const std::uint64_t restore_preparations =
        ArrayStoragePolicy::restore_preparations;
    const std::uint64_t copies = ArrayStoragePolicy::copies;
    const std::uint64_t prepublish_checks =
        ArrayStoragePolicy::prepublish_checks;
    const std::uint64_t reversions =
        TwoPhaseArrayStoragePolicy::reversions;
    auto stale_exchange = old_prepared
        ? second->CommitPrepared(std::move(*old_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    const bool stale_rejection_is_inert =
        !stale_exchange &&
        SameCommitAccounting(second->telemetry(), accounting_before) &&
        second->telemetry().stale_consumers_rejected ==
            accounting_before.stale_consumers_rejected + 1 &&
        SameCoordinates(second->coordinates(), coordinates_before) &&
        second->status() == status_before &&
        second->last_failure() == failure_before &&
        ArrayStoragePolicy::publications == publications &&
        ArrayStoragePolicy::discards == discards &&
        ArrayStoragePolicy::stores == stores &&
        ArrayStoragePolicy::preparations == preparations &&
        ArrayStoragePolicy::restore_preparations == restore_preparations &&
        ArrayStoragePolicy::copies == copies &&
        ArrayStoragePolicy::prepublish_checks == prepublish_checks &&
        TwoPhaseArrayStoragePolicy::reversions == reversions;
    Slot published_after{};
    const bool published_is_unchanged =
        second->CopyPublished(published, published_after) &&
        SameSlotBytes(published_before, published_after);
    auto live_exchange = new_prepared
        ? second->CommitPrepared(std::move(*new_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    const AttemptOutcome cleanup = live_exchange
        ? second->RevertCommit(std::move(*live_exchange))
        : AttemptOutcome{};
    passed = Check(
                 old_prepared && new_prepared && first_candidate_captured &&
                     second_candidate_captured &&
                     SameSlotBytes(first_candidate, second_candidate) &&
                     SameAttempt(first_begin.attempt, second_begin.attempt) &&
                     exact_entry && exact_target && exact_checkpoint &&
                     exact_prepared_checkpoint && published_captured &&
                     stale_rejection_is_inert && published_is_unchanged &&
                     live_exchange &&
                     cleanup.retry_available,
                 "same_address_incarnation_aba_rejects_old_prepared_authority_only") &&
        passed;
    second->~TwoPhaseContract();
  }

  {
    alignas(TwoPhaseContract)
        std::array<unsigned char, sizeof(TwoPhaseContract)> storage{};
    TwoPhaseArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    TwoPhaseContract* first = new (storage.data()) TwoPhaseContract{
        CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
    const BeginOutcome first_begin =
        first->BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(*first, first_begin.attempt, AcceptedSlot());
    auto first_products = IssueAllFor(*first, first_begin.attempt);
    auto first_staging = first->FreezeCandidate(first_begin.attempt, 12);
    Slot first_candidate{};
    const bool first_candidate_captured = first_staging &&
        first->CopyPrepared(*first_staging, first_candidate);
    const StateCoordinates first_entry = first_staging
        ? first_staging->entry_coordinates()
        : StateCoordinates{};
    const StateCoordinates first_target = first_staging
        ? first_staging->target_coordinates()
        : StateCoordinates{};
    const CheckpointDescriptor first_checkpoint = first_staging
        ? first_staging->checkpoint_descriptor()
        : CheckpointDescriptor{};
    auto first_prepared = first_staging
        ? first->PrepareCommit(std::move(*first_staging),
                               PassingReport(12),
                               std::move(first_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    auto old_provisional = first_prepared
        ? first->CommitPrepared(std::move(*first_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    first->~TwoPhaseContract();

    TwoPhaseArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    TwoPhaseContract* second = new (storage.data()) TwoPhaseContract{
        CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
    const BeginOutcome second_begin =
        second->BeginStep({InteriorSpace::l2, 0.125});
    StoreCandidateFor(*second, second_begin.attempt, AcceptedSlot());
    auto second_products = IssueAllFor(*second, second_begin.attempt);
    auto second_staging = second->FreezeCandidate(second_begin.attempt, 12);
    Slot second_candidate{};
    const bool second_candidate_captured = second_staging &&
        second->CopyPrepared(*second_staging, second_candidate);
    const bool exact_entry = second_staging &&
        SameCoordinates(first_entry, second_staging->entry_coordinates());
    const bool exact_target = second_staging &&
        SameCoordinates(first_target, second_staging->target_coordinates());
    const bool exact_checkpoint = second_staging &&
        detail::SameCheckpoint(first_checkpoint,
                               second_staging->checkpoint_descriptor());
    auto second_prepared = second_staging
        ? second->PrepareCommit(std::move(*second_staging),
                                PassingReport(12),
                                std::move(second_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    const bool exact_prepared_checkpoint = second_prepared &&
        detail::SameCheckpoint(first_checkpoint,
                               second_prepared->checkpoint_descriptor());
    auto new_provisional = second_prepared
        ? second->CommitPrepared(std::move(*second_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    const auto published = second->Published();
    Slot published_before{};
    const bool published_captured =
        second->CopyPublished(published, published_before);

    const RunTelemetry accounting_before = second->telemetry();
    const StateCoordinates coordinates_before = second->coordinates();
    const RunStatus status_before = second->status();
    const FailureReason failure_before = second->last_failure();
    const std::uint64_t publications = ArrayStoragePolicy::publications;
    const std::uint64_t discards = ArrayStoragePolicy::discards;
    const std::uint64_t stores = ArrayStoragePolicy::stores;
    const std::uint64_t preparations = ArrayStoragePolicy::preparations;
    const std::uint64_t restore_preparations =
        ArrayStoragePolicy::restore_preparations;
    const std::uint64_t copies = ArrayStoragePolicy::copies;
    const std::uint64_t prepublish_checks =
        ArrayStoragePolicy::prepublish_checks;
    const std::uint64_t reversions =
        TwoPhaseArrayStoragePolicy::reversions;
    const AttemptOutcome stale_finalize = old_provisional
        ? second->FinalizeCommit(std::move(*old_provisional))
        : AttemptOutcome{};
    const bool stale_rejection_is_inert =
        stale_finalize.reason == FailureReason::stale_attempt &&
        SameCommitAccounting(second->telemetry(), accounting_before) &&
        second->telemetry().stale_consumers_rejected ==
            accounting_before.stale_consumers_rejected + 1 &&
        SameCoordinates(second->coordinates(), coordinates_before) &&
        second->status() == status_before &&
        second->last_failure() == failure_before &&
        ArrayStoragePolicy::publications == publications &&
        ArrayStoragePolicy::discards == discards &&
        ArrayStoragePolicy::stores == stores &&
        ArrayStoragePolicy::preparations == preparations &&
        ArrayStoragePolicy::restore_preparations == restore_preparations &&
        ArrayStoragePolicy::copies == copies &&
        ArrayStoragePolicy::prepublish_checks == prepublish_checks &&
        TwoPhaseArrayStoragePolicy::reversions == reversions;
    Slot published_after{};
    const bool published_is_unchanged =
        second->CopyPublished(published, published_after) &&
        SameSlotBytes(published_before, published_after);
    const AttemptOutcome live_finalize = new_provisional
        ? second->FinalizeCommit(std::move(*new_provisional))
        : AttemptOutcome{};
    passed = Check(
                 old_provisional && new_provisional &&
                     first_candidate_captured && second_candidate_captured &&
                     SameSlotBytes(first_candidate, second_candidate) &&
                     SameAttempt(first_begin.attempt, second_begin.attempt) &&
                     exact_entry && exact_target && exact_checkpoint &&
                     exact_prepared_checkpoint && published_captured &&
                     stale_rejection_is_inert && published_is_unchanged &&
                     live_finalize.committed,
                 "same_address_incarnation_aba_rejects_old_provisional_authority_only") &&
        passed;
    second->~TwoPhaseContract();
  }

  for (const bool provisional_phase : {false, true})
  {
    TwoPhaseContract contract = MakeTwoPhaseContract();
    const auto old = contract.Published();
    Slot old_before{};
    StateCoordinates coordinates_before{};
    const bool captured = contract.CopyPublished(old, old_before) &&
        contract.PublishedCoordinates(old, coordinates_before);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const auto retained_candidate = contract.AcquireCandidate(begin.attempt);
    const bool stored = retained_candidate &&
        contract.StoreCandidate(*retained_candidate, AcceptedSlot());
    auto products = IssueAllFor(contract, begin.attempt);
    auto staging = contract.FreezeCandidate(begin.attempt, 12);
    auto prepared = staging
        ? contract.PrepareCommit(std::move(*staging),
                                 PassingReport(12),
                                 std::move(products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    std::optional<TwoPhaseContract::ProvisionalCommitToken> provisional{};
    if (provisional_phase && prepared)
    {
      auto exchanged = contract.CommitPrepared(std::move(*prepared));
      if (exchanged) { provisional.emplace(std::move(*exchanged)); }
    }

    const std::uint64_t admissions_before =
        TestCollectivePolicy::admissions;
    const std::uint64_t reductions_before =
        TestCollectivePolicy::reductions;
    const std::uint64_t publications_before =
        ArrayStoragePolicy::publications;
    const std::uint64_t discards_before = ArrayStoragePolicy::discards;
    const std::uint64_t commits_before = contract.telemetry().commits;
    const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
    const TransferAccounting committed_before =
        contract.telemetry().committed_transfers;

    const BeginOutcome nested_begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const BeginOutcome nested_retry = contract.Retry();
    TwoPhaseContract::ProductBundle legacy_products{};
    const AttemptOutcome nested_finish = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(legacy_products));
    const bool blocked_store = retained_candidate &&
        !contract.StoreCandidate(*retained_candidate, OtherSlot());
    const auto blocked_acquire = contract.AcquireCandidate(begin.attempt);
    Slot blocked_candidate_copy{};
    const bool blocked_copy = retained_candidate &&
        !contract.CopyCandidate(*retained_candidate, blocked_candidate_copy);
    const auto blocked_product =
        contract.IssueProduct<DerivedProductKind::residual>(begin.attempt);
    CheckpointDescriptor busy_checkpoint{};
    const bool blocked_checkpoint =
        !contract.TryMakeCheckpoint(busy_checkpoint);
    const RestoreOutcome nested_restore =
        contract.Restore(CheckpointDescriptor{}, OtherSlot());
    Slot old_after{};
    StateCoordinates coordinates_after{};
    const bool old_preserved = contract.CopyPublished(old, old_after) &&
        contract.PublishedCoordinates(old, coordinates_after);

    passed = Check(
                 captured && begin.started && stored && staging && prepared &&
                     (!provisional_phase || provisional) &&
                     !nested_begin.started && !nested_retry.started &&
                     nested_finish.reason == FailureReason::stale_attempt &&
                     blocked_store && !blocked_acquire && blocked_copy &&
                     !blocked_product && blocked_checkpoint &&
                     !nested_restore.restored &&
                     nested_restore.compatibility ==
                         RestartCompatibility::contract_busy &&
                     old_preserved && SameSlotBytes(old_before, old_after) &&
                     SameCoordinates(coordinates_before, coordinates_after) &&
                     SameCoordinates(contract.coordinates(),
                                     coordinates_before) &&
                     TestCollectivePolicy::admissions == admissions_before &&
                     TestCollectivePolicy::reductions == reductions_before &&
                     ArrayStoragePolicy::publications == publications_before &&
                     ArrayStoragePolicy::discards == discards_before &&
                     ArrayStoragePolicy::restore_preparations == 0 &&
                     contract.status() == RunStatus::attempt_active &&
                     contract.telemetry().commits == commits_before &&
                     SameAccepted(contract.telemetry().accepted,
                                  accepted_before) &&
                     SameTransfers(contract.telemetry().committed_transfers,
                                   committed_before),
                 provisional_phase
                     ? "provisional_reentrant_apis_preserve_both_live_slots"
                     : "prepared_reentrant_apis_preserve_both_live_slots") &&
        passed;

    if (provisional_phase && provisional)
    {
      const AttemptOutcome reverted =
          contract.RevertCommit(std::move(*provisional));
      passed = Check(reverted.retry_available,
                     "provisional_reentrant_regression_can_still_revert") &&
          passed;
    }
    else if (prepared)
    {
      const AttemptOutcome aborted =
          contract.AbortPrepared(std::move(*prepared));
      passed = Check(aborted.retry_available,
                     "prepared_reentrant_regression_can_still_abort") &&
          passed;
    }
  }
  return passed;
}

StateHash FrozenAcceptedInteriorHash()
{
  StateHash hash{};
  hash.bytes = {{33, 37, 41, 45, 37, 41, 45, 49,
                 41, 45, 49, 53, 45, 49, 53, 57,
                 49, 53, 57, 61, 53, 57, 61, 65,
                 57, 61, 65, 69, 61, 65, 69, 73}};
  return hash;
}

StateHash FrozenAcceptedTraceHash()
{
  StateHash hash{};
  hash.bytes = {{145, 149, 153, 148, 152, 156, 151, 155,
                 159, 154, 158, 162, 157, 161, 165, 160,
                 164, 168, 163, 167, 171, 166, 170, 174,
                 169, 173, 177, 172, 176, 180, 175, 179}};
  return hash;
}

bool CreateCheckpoint(CheckpointDescriptor& checkpoint,
                      Slot& accepted_state)
{
  Contract contract = MakeContract();
  const BeginOutcome begin =
      contract.BeginStep({InteriorSpace::l2, 0.125});
  accepted_state = AcceptedSlot();
  if (!StoreCandidate(contract, begin.attempt, accepted_state)) { return false; }
  auto products = IssueAll(contract, begin.attempt);
  const AttemptOutcome outcome = contract.FinishAttempt(
      begin.attempt, PassingReport(44), std::move(products));
  return outcome.committed && contract.TryMakeCheckpoint(checkpoint);
}

bool CheckExternalEnumAndMaskDomains()
{
  bool passed = true;
  const std::array<FailureReason, 3> invalid_failure_reasons{{
      static_cast<FailureReason>(255),
      FailureReason::count,
      static_cast<FailureReason>(
          static_cast<unsigned int>(FailureReason::count) + 1U)}};
  const std::array<RestartCompatibility, 3> invalid_compatibilities{{
      static_cast<RestartCompatibility>(255),
      RestartCompatibility::count,
      static_cast<RestartCompatibility>(
          static_cast<unsigned int>(RestartCompatibility::count) + 1U)}};

  for (const FailureReason invalid : invalid_failure_reasons)
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault =
        CollectiveFault::admission_invalid_resolved_reason;
    TestCollectivePolicy::injected_reason = invalid;
    const BeginOutcome outcome =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!outcome.started &&
                       outcome.reason == FailureReason::collective_protocol &&
                       TestCollectivePolicy::admissions == 1 &&
                       contract.telemetry().first_failure ==
                           FailureReason::collective_protocol,
                   "admission_invalid_reduced_reason_fails_closed") && passed;
  }
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault =
        CollectiveFault::admission_unknown_failure_mask;
    const BeginOutcome outcome =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!outcome.started &&
                       outcome.reason == FailureReason::collective_protocol &&
                       TestCollectivePolicy::admissions == 1,
                   "admission_unknown_failure_mask_fails_closed") && passed;
  }

  for (const FailureReason invalid : invalid_failure_reasons)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.originating_failure = invalid;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::collective_protocol &&
                       outcome.retry_available &&
                       TestCollectivePolicy::reductions == 1,
                   "finish_invalid_originating_reason_fails_closed") && passed;
  }
  for (const FailureReason invalid : invalid_failure_reasons)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    TestCollectivePolicy::fault =
        CollectiveFault::finish_invalid_resolved_reason;
    TestCollectivePolicy::injected_reason = invalid;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::collective_protocol &&
                       TestCollectivePolicy::reductions == 1,
                   "finish_invalid_reduced_reason_fails_closed") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    TestCollectivePolicy::fault =
        CollectiveFault::finish_unknown_failure_mask;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::collective_protocol &&
                       TestCollectivePolicy::reductions == 1,
                   "finish_unknown_failure_mask_fails_closed") && passed;
  }

  CheckpointDescriptor checkpoint{};
  Slot loaded{};
  const bool checkpoint_created = CreateCheckpoint(checkpoint, loaded);
  passed = Check(checkpoint_created,
                 "enum_domain_checkpoint_fixture_created") && passed;
  if (!checkpoint_created) { return passed; }

  for (const FailureReason invalid : invalid_failure_reasons)
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault =
        CollectiveFault::restore_invalid_resolved_reason;
    TestCollectivePolicy::injected_reason = invalid;
    const RestoreOutcome outcome = contract.Restore(checkpoint, loaded);
    passed = Check(!outcome.restored &&
                       outcome.compatibility ==
                           RestartCompatibility::collective_protocol &&
                       TestCollectivePolicy::restores == 1 &&
                       ArrayStoragePolicy::publications == 0,
                   "restore_invalid_reduced_reason_fails_closed") && passed;
  }
  for (const RestartCompatibility invalid : invalid_compatibilities)
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault =
        CollectiveFault::restore_invalid_resolved_compatibility;
    TestCollectivePolicy::injected_compatibility = invalid;
    const RestoreOutcome outcome = contract.Restore(checkpoint, loaded);
    passed = Check(!outcome.restored &&
                       outcome.compatibility ==
                           RestartCompatibility::collective_protocol &&
                       TestCollectivePolicy::restores == 1 &&
                       ArrayStoragePolicy::publications == 0,
                   "restore_invalid_reduced_compatibility_fails_closed") &&
        passed;
  }
  for (const CollectiveFault fault : {
           CollectiveFault::restore_unknown_failure_mask,
           CollectiveFault::restore_unknown_compatibility_mask})
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault = fault;
    const RestoreOutcome outcome = contract.Restore(checkpoint, loaded);
    passed = Check(!outcome.restored &&
                       outcome.compatibility ==
                           RestartCompatibility::collective_protocol &&
                       TestCollectivePolicy::restores == 1 &&
                       ArrayStoragePolicy::publications == 0,
                   "restore_unknown_collective_mask_bit_fails_closed") &&
        passed;
  }
  return passed;
}

template <typename Mutator>
bool ExpectRestoreFailure(const CheckpointDescriptor& original,
                          const Slot& loaded,
                          const RestartCompatibility expected,
                          Mutator mutate,
                          const char* requirement)
{
  CheckpointDescriptor changed = original;
  mutate(changed);
  Contract contract = MakeContract();
  const auto published = contract.Published();
  Slot before{};
  StateCoordinates coordinates_before{};
  const bool captured = contract.CopyPublished(published, before) &&
      contract.PublishedCoordinates(published, coordinates_before);
  const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
  const RestoreOutcome outcome = contract.Restore(changed, loaded);
  Slot after{};
  const bool preserved = contract.CopyPublished(published, after);
  return Check(captured && !outcome.restored &&
                   outcome.compatibility == expected && preserved &&
                   before.interior == after.interior &&
                   before.trace == after.trace &&
                   SameCoordinates(contract.coordinates(), coordinates_before) &&
                   SameAccepted(contract.telemetry().accepted,
                                accepted_before) &&
                   TestCollectivePolicy::restores == 1,
               requirement);
}

bool CheckRestoreAttemptFrontierAndAuthorities(
    const CheckpointDescriptor& checkpoint,
    const Slot& loaded)
{
  bool passed = true;
  const std::uint64_t next_checkpoint_cursor = checkpoint.cursor + 1;

  {
    TwoPhaseContract contract = MakeTwoPhaseContract();
    const RestoreOutcome restored = contract.Restore(checkpoint, loaded);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, checkpoint.dt});
    const bool stored = begin.started &&
        StoreCandidateFor(contract, begin.attempt, OtherSlot());
    auto products = IssueAllFor(contract, begin.attempt);
    auto staging = contract.FreezeCandidate(begin.attempt,
                                             begin.attempt.sequence);
    const bool staging_target_is_next = staging &&
        staging->entry_coordinates().cursor == checkpoint.cursor &&
        staging->target_coordinates().cursor == next_checkpoint_cursor;
    auto prepared = staging
        ? contract.PrepareCommit(std::move(*staging),
                                 PassingReport(begin.attempt.sequence),
                                 std::move(products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    const bool prepared_target_is_next = prepared &&
        prepared->checkpoint_descriptor().cursor == next_checkpoint_cursor &&
        prepared->checkpoint_descriptor().coordinates.cursor ==
            next_checkpoint_cursor;
    const AttemptOutcome aborted = prepared
        ? contract.AbortPrepared(std::move(*prepared))
        : AttemptOutcome{};
    passed = Check(
                 restored.restored && begin.started && stored && staging &&
                     prepared &&
                     begin.attempt.sequence == next_checkpoint_cursor &&
                     staging_target_is_next && prepared_target_is_next &&
                     aborted.retry_available,
                 "restored_cursor_sets_next_attempt_and_prepare_target_frontier") &&
        passed;
  }

  {
    const std::uint64_t newer_frontier = checkpoint.cursor + 20;
    Contract contract = MakeContract(InitialCoordinates(),
                                     Resources(),
                                     newer_frontier);
    const RestoreOutcome restored = contract.Restore(checkpoint, loaded);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, checkpoint.dt});
    passed = Check(restored.restored && begin.started &&
                       begin.attempt.sequence == newer_frontier + 1 &&
                       contract.coordinates().cursor == checkpoint.cursor,
                   "older_restore_never_decreases_newer_attempt_frontier") &&
        passed;
  }

  {
    const std::uint64_t frontier_before_rejection = checkpoint.cursor + 20;
    Contract contract = MakeContract(InitialCoordinates(),
                                     Resources(),
                                     frontier_before_rejection);
    Slot corrupt = loaded;
    corrupt.trace[0] += 1.0;
    const RestoreOutcome rejected = contract.Restore(checkpoint, corrupt);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, checkpoint.dt});
    passed = Check(
                 !rejected.restored &&
                     rejected.compatibility ==
                         RestartCompatibility::trace_hash_mismatch &&
                     begin.started &&
                     begin.attempt.sequence == frontier_before_rejection + 1 &&
                     SameCoordinates(contract.coordinates(),
                                     InitialCoordinates()) &&
                     ArrayStoragePolicy::publications == 0 &&
                     TestCollectivePolicy::restores == 1,
                 "rejected_restore_does_not_mutate_attempt_frontier") &&
        passed;
  }

  {
    const std::uint64_t initial_frontier = checkpoint.cursor + 16;
    TwoPhaseContract contract = MakeTwoPhaseContract(
        InitialCoordinates(), Resources(), initial_frontier);
    const auto old_published = contract.Published();
    const BeginOutcome first =
        contract.BeginStep({InteriorSpace::l2, checkpoint.dt});
    auto stale_products = IssueAllFor(contract, first.attempt);
    TwoPhaseContract::ProductBundle missing_products{};
    const AttemptOutcome failed = contract.FinishAttempt(
        first.attempt,
        PassingReport(first.attempt.sequence),
        std::move(missing_products));
    const BeginOutcome retry = contract.Retry();
    const auto old_candidate = contract.AcquireCandidate(retry.attempt);
    const bool retry_stored = old_candidate &&
        contract.StoreCandidate(*old_candidate, OtherSlot());
    auto retry_products = IssueAllFor(contract, retry.attempt);
    auto old_staging = contract.FreezeCandidate(retry.attempt,
                                                 retry.attempt.sequence);
    auto old_prepared = old_staging
        ? contract.PrepareCommit(std::move(*old_staging),
                                 PassingReport(retry.attempt.sequence),
                                 std::move(retry_products))
        : std::optional<TwoPhaseContract::PreparedCommitToken>{};
    const AttemptOutcome aborted = old_prepared
        ? contract.AbortPrepared(std::move(*old_prepared))
        : AttemptOutcome{};
    const RestoreOutcome restored = contract.Restore(checkpoint, loaded);
    const auto restored_published = contract.Published();
    Slot restored_copy{};
    const bool restored_publication_valid =
        contract.CopyPublished(restored_published, restored_copy) &&
        SameSlotBytes(restored_copy, loaded);
    const StateCoordinates restored_coordinates = contract.coordinates();
    const RunStatus restored_status = contract.status();
    const FailureReason restored_failure = contract.last_failure();
    const std::uint64_t stale_rejections =
        contract.telemetry().stale_consumers_rejected;
    const std::uint64_t publications = ArrayStoragePolicy::publications;
    const std::uint64_t stores = ArrayStoragePolicy::stores;
    const std::uint64_t copies = ArrayStoragePolicy::copies;
    Slot stale_copy{};
    const bool stale_publication_rejected =
        !contract.CopyPublished(old_published, stale_copy);
    const bool moved_staging_rejected = old_staging &&
        !contract.CopyPrepared(*old_staging, stale_copy);
    const bool old_candidate_rejected = old_candidate &&
        !contract.StoreCandidate(*old_candidate, AcceptedSlot());
    auto stale_exchange = old_prepared
        ? contract.CommitPrepared(std::move(*old_prepared))
        : std::optional<TwoPhaseContract::ProvisionalCommitToken>{};
    const bool stale_authorities_are_inert =
        restored_publication_valid && stale_publication_rejected &&
        moved_staging_rejected && old_candidate_rejected && !stale_exchange &&
        SameCoordinates(contract.coordinates(), restored_coordinates) &&
        contract.status() == restored_status &&
        contract.last_failure() == restored_failure &&
        contract.telemetry().stale_consumers_rejected ==
            stale_rejections + 2 &&
        ArrayStoragePolicy::publications == publications &&
        ArrayStoragePolicy::stores == stores &&
        ArrayStoragePolicy::copies == copies;
    const BeginOutcome after_restore =
        contract.BeginStep({InteriorSpace::l2, checkpoint.dt});
    const auto fresh_candidate =
        contract.AcquireCandidate(after_restore.attempt);
    const bool fresh_stored = fresh_candidate &&
        contract.StoreCandidate(*fresh_candidate, OtherSlot());
    const AttemptOutcome stale_product = contract.FinishAttempt(
        after_restore.attempt,
        PassingReport(after_restore.attempt.sequence),
        std::move(stale_products));
    passed = Check(
                 first.started &&
                     first.attempt.sequence == initial_frontier + 1 &&
                     failed.retry_available &&
                     failed.reason == FailureReason::missing_product &&
                     retry.started && retry_stored && old_staging &&
                     old_prepared && !aborted.retry_available &&
                     restored.restored && stale_authorities_are_inert &&
                     after_restore.started && fresh_stored &&
                     after_restore.attempt.sequence ==
                         retry.attempt.sequence + 1 &&
                     !stale_product.committed &&
                     stale_product.reason == FailureReason::stale_product &&
                     ArrayStoragePolicy::publications == publications,
                 "restore_preserves_authority_staleness_without_aba") &&
        passed;
  }

  {
    const std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();
    CheckpointDescriptor maximum_cursor = checkpoint;
    maximum_cursor.cursor = maximum;
    maximum_cursor.coordinates.cursor = maximum;
    maximum_cursor.coordinate_fingerprint =
        FrozenCoordinateFingerprint(maximum_cursor.coordinates);
    Contract contract = MakeContract();
    const RestoreOutcome restored = contract.Restore(maximum_cursor, loaded);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, checkpoint.dt});
    passed = Check(restored.restored && !begin.started &&
                       begin.reason == FailureReason::coordinate_overflow &&
                       begin.attempt.sequence == maximum &&
                       contract.coordinates().cursor == maximum &&
                       TestCollectivePolicy::admissions == 1,
                   "maximum_restored_cursor_fails_closed_without_wraparound") &&
        passed;
  }

  {
    const std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();
    Contract contract = MakeContract(InitialCoordinates(),
                                     Resources(),
                                     maximum);
    const RestoreOutcome restored = contract.Restore(checkpoint, loaded);
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, checkpoint.dt});
    passed = Check(restored.restored && !begin.started &&
                       begin.reason == FailureReason::coordinate_overflow &&
                       begin.attempt.sequence == maximum &&
                       contract.coordinates().cursor == checkpoint.cursor,
                   "maximum_live_attempt_frontier_survives_older_restore") &&
        passed;
  }
  return passed;
}

bool CheckAtomicCheckpointRestore()
{
  bool passed = true;
  {
    Contract contract = MakeContract();
    CheckpointDescriptor checkpoint{};
    passed = Check(!contract.TryMakeCheckpoint(checkpoint),
                   "checkpoint_requires_successful_committed_state") && passed;
  }

  CheckpointDescriptor checkpoint{};
  Slot loaded{};
  passed = Check(CreateCheckpoint(checkpoint, loaded),
                 "checkpoint_source_commit_created") && passed;
  passed = Check(checkpoint.schema_version == checkpoint_schema_version &&
                     checkpoint.successful_commit &&
                     checkpoint.restart_mode ==
                         RestartMode::same_backend_rank_partition &&
                     checkpoint.cursor == 44 && checkpoint.dt == 0.125 &&
                     checkpoint.mfem_pin == Resources().mfem_pin &&
                     checkpoint.mesh_hash == Resources().mesh_hash &&
                     checkpoint.partition_hash == Resources().partition_hash &&
                     checkpoint.topology_hash == Resources().topology_hash &&
                     checkpoint.rank_count == Resources().rank_count &&
                     checkpoint.polynomial_order ==
                         Resources().polynomial_order &&
                     checkpoint.layout == Resources().layout &&
                     checkpoint.precision == Selection::Descriptor().precision &&
                     checkpoint.interior_hash ==
                         FrozenAcceptedInteriorHash() &&
                     checkpoint.trace_hash == FrozenAcceptedTraceHash() &&
                     checkpoint.coordinate_fingerprint ==
                         FrozenCoordinateFingerprint(checkpoint.coordinates),
                 "checkpoint_exact_tuple_topology_coordinates_and_frozen_hashes") &&
      passed;
  passed = CheckRestoreAttemptFrontierAndAuthorities(checkpoint, loaded) &&
      passed;

  for (const int invalid_field : {0, 1, 2, 3})
  {
    PersistentResourceDescriptor resources = Resources();
    CheckpointDescriptor matching = checkpoint;
    if (invalid_field == 0)
    {
      resources.rank_count = 0;
      matching.rank_count = 0;
    }
    else if (invalid_field == 1)
    {
      resources.mfem_pin = StableIdentity{""};
      matching.mfem_pin = resources.mfem_pin;
    }
    else if (invalid_field == 2)
    {
      resources.layout = StableIdentity{""};
      matching.layout = resources.layout;
    }
    else
    {
      resources.reuse.persistent_workspace = false;
    }
    Contract contract = MakeContract(InitialCoordinates(), resources);
    const auto published_before = contract.Published();
    Slot before{};
    const bool captured = contract.CopyPublished(published_before, before);
    const StateCoordinates coordinates_before = contract.coordinates();
    const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
    const RestoreOutcome outcome = contract.Restore(matching, loaded);
    Slot after{};
    passed = Check(
                 captured && !outcome.restored &&
                     outcome.compatibility ==
                         RestartCompatibility::policy_mismatch &&
                     contract.status() == RunStatus::invalid_request &&
                     contract.telemetry().first_failure ==
                         FailureReason::invalid_resource_descriptor &&
                     TestCollectivePolicy::restores == 1 &&
                     ArrayStoragePolicy::restore_preparations == 0 &&
                     ArrayStoragePolicy::publications == 0 &&
                     ArrayStoragePolicy::discards == 0 &&
                     SameCoordinates(contract.coordinates(),
                                     coordinates_before) &&
                     SameAccepted(contract.telemetry().accepted,
                                  accepted_before) &&
                     contract.CopyPublished(published_before, after) &&
                     after.interior == before.interior &&
                     after.trace == before.trace,
                 "invalid_resources_cannot_be_rehabilitated_by_matching_restore") &&
        passed;
  }

  {
    StateCoordinates invalid_coordinates = InitialCoordinates();
    invalid_coordinates.time = std::numeric_limits<double>::quiet_NaN();
    Contract contract = MakeContract(invalid_coordinates);
    const StateCoordinates coordinates_before = contract.coordinates();
    const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
    const RestoreOutcome outcome = contract.Restore(checkpoint, loaded);
    passed = Check(!outcome.restored &&
                       outcome.compatibility ==
                           RestartCompatibility::policy_mismatch &&
                       contract.status() == RunStatus::invalid_request &&
                       contract.telemetry().first_failure ==
                           FailureReason::invalid_dt &&
                       TestCollectivePolicy::restores == 1 &&
                       ArrayStoragePolicy::restore_preparations == 0 &&
                       ArrayStoragePolicy::publications == 0 &&
                       SameCoordinates(contract.coordinates(),
                                       coordinates_before) &&
                       SameAccepted(contract.telemetry().accepted,
                                    accepted_before),
                   "invalid_constructor_coordinates_cannot_be_restored") &&
        passed;
  }

  {
    const std::uint64_t saved_incarnation =
        detail::next_contract_incarnation.load(std::memory_order_relaxed);
    detail::next_contract_incarnation.store(
        std::numeric_limits<std::uint64_t>::max(),
        std::memory_order_relaxed);
    ArrayStoragePolicy::Reset();
    TestCollectivePolicy::Reset();
    Contract contract{
        CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
    detail::next_contract_incarnation.store(saved_incarnation,
                                            std::memory_order_relaxed);
    const StateCoordinates coordinates_before = contract.coordinates();
    const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
    const RestoreOutcome outcome = contract.Restore(checkpoint, loaded);
    passed = Check(!outcome.restored &&
                       outcome.compatibility ==
                           RestartCompatibility::policy_mismatch &&
                       contract.status() == RunStatus::invalid_request &&
                       contract.telemetry().first_failure ==
                           FailureReason::incarnation_exhausted &&
                       TestCollectivePolicy::restores == 1 &&
                       ArrayStoragePolicy::restore_preparations == 0 &&
                       ArrayStoragePolicy::publications == 0 &&
                       SameCoordinates(contract.coordinates(),
                                       coordinates_before) &&
                       SameAccepted(contract.telemetry().accepted,
                                    accepted_before),
                   "invalid_incarnation_cannot_be_restored") && passed;
  }

  {
    BackingContract contract = MakeBackingContract(true);
    BackingSlot backing_loaded{{5, 11.0}, {6, 15.0}};
    CheckpointDescriptor matching = checkpoint;
    matching.interior_hash =
        BackingStoragePolicy::HashInterior(backing_loaded.interior);
    matching.trace_hash =
        BackingStoragePolicy::HashTrace(backing_loaded.trace);
    const auto published_before = contract.Published();
    BackingSlot before{{91, 0.0}, {92, 0.0}};
    const bool captured = contract.CopyPublished(published_before, before);
    const StateCoordinates coordinates_before = contract.coordinates();
    const AcceptedTelemetry accepted_before = contract.telemetry().accepted;
    const RestoreOutcome outcome = contract.Restore(matching, backing_loaded);
    BackingSlot after{{93, 0.0}, {94, 0.0}};
    passed = Check(
                 captured && !outcome.restored &&
                     outcome.compatibility ==
                         RestartCompatibility::policy_mismatch &&
                     contract.status() == RunStatus::invalid_request &&
                     contract.telemetry().first_failure ==
                         FailureReason::shared_backing_store &&
                     TestCollectivePolicy::restores == 1 &&
                     BackingStoragePolicy::restore_preparations == 0 &&
                     BackingStoragePolicy::can_publish_calls == 0 &&
                     BackingStoragePolicy::publish_calls == 0 &&
                     SameCoordinates(contract.coordinates(),
                                     coordinates_before) &&
                     SameAccepted(contract.telemetry().accepted,
                                  accepted_before) &&
                     contract.CopyPublished(published_before, after) &&
                     after.interior.value == before.interior.value &&
                     after.trace.value == before.trace.value,
                 "invalid_shared_storage_construction_cannot_be_restored") &&
        passed;
  }

  {
    Contract contract = MakeContract();
    const auto published_before = contract.Published();
    Slot before{};
    const bool captured = contract.CopyPublished(published_before, before);
    TestCollectivePolicy::fault = CollectiveFault::restore_remote_failure;
    TestCollectivePolicy::remote_compatibility =
        RestartCompatibility::interior_hash_mismatch;
    const RestoreOutcome failed = contract.Restore(checkpoint, AcceptedSlot());
    Slot after{};
    const bool preserved = contract.CopyPublished(published_before, after);
    TestCollectivePolicy::fault = CollectiveFault::none;
    const RestoreOutcome accepted =
        contract.Restore(checkpoint, AcceptedSlot());
    passed = Check(captured && !failed.restored &&
                       failed.compatibility ==
                           RestartCompatibility::interior_hash_mismatch &&
                       preserved && before.interior == after.interior &&
                       before.trace == after.trace && accepted.restored &&
                       TestCollectivePolicy::restores == 2 &&
                       contract.telemetry().first_failure ==
                           FailureReason::restart_incompatible,
                   "remote_restore_failure_is_atomic_and_persists_first_failure") &&
        passed;
  }

  for (const int failure_point : {0, 1})
  {
    Contract contract = MakeContract();
    const auto published_before = contract.Published();
    Slot before{};
    const bool captured = contract.CopyPublished(published_before, before);
    if (failure_point == 0)
    {
      ArrayStoragePolicy::fail_restore_call = 1;
    }
    else
    {
      ArrayStoragePolicy::fail_prepublish_call = 1;
    }
    const RestoreOutcome outcome = contract.Restore(checkpoint, AcceptedSlot());
    Slot after{};
    const RestartCompatibility expected = failure_point == 0
        ? RestartCompatibility::candidate_preparation
        : RestartCompatibility::publication_precondition;
    passed = Check(captured && !outcome.restored &&
                       outcome.compatibility == expected &&
                       ArrayStoragePolicy::publications == 0 &&
                       ArrayStoragePolicy::discards == 1 &&
                       contract.CopyPublished(published_before, after) &&
                       before.interior == after.interior &&
                       before.trace == after.trace &&
                       TestCollectivePolicy::restores == 1,
                   "restore_prepare_and_publish_preflight_fail_before_mutation") &&
        passed;
  }

  for (const CollectiveFault fault : {
           CollectiveFault::restore_identity_false,
           CollectiveFault::restore_failure_consistency_false,
           CollectiveFault::restore_participants,
           CollectiveFault::restore_context})
  {
    Contract contract = MakeContract();
    const auto published_before = contract.Published();
    Slot before{};
    const bool captured = contract.CopyPublished(published_before, before);
    TestCollectivePolicy::fault = fault;
    const RestoreOutcome outcome = contract.Restore(checkpoint, AcceptedSlot());
    Slot after{};
    passed = Check(captured && !outcome.restored &&
                       outcome.compatibility ==
                           RestartCompatibility::collective_protocol &&
                       contract.CopyPublished(published_before, after) &&
                       before.interior == after.interior &&
                       before.trace == after.trace &&
                       contract.telemetry().first_failure ==
                           FailureReason::collective_protocol &&
                       TestCollectivePolicy::restores == 1,
                   "restore_context_reason_or_census_contradiction_fails_closed") &&
        passed;
  }

  for (const CollectiveFault fault : {
           CollectiveFault::restore_hide_local_failure,
           CollectiveFault::restore_change_local_failure})
  {
    Contract contract = MakeContract();
    Slot corrupt = AcceptedSlot();
    corrupt.interior[0] += 1.0;
    TestCollectivePolicy::fault = fault;
    TestCollectivePolicy::remote_compatibility =
        RestartCompatibility::trace_hash_mismatch;
    const RestoreOutcome outcome = contract.Restore(checkpoint, corrupt);
    passed = Check(!outcome.restored &&
                       outcome.compatibility ==
                           RestartCompatibility::collective_protocol &&
                       contract.telemetry().first_failure ==
                           FailureReason::collective_protocol &&
                       TestCollectivePolicy::restores == 1,
                   "restore_cannot_omit_or_substitute_local_incompatibility") &&
        passed;
  }

  {
    StateCoordinates wrong_coordinates{8.0, 80, 90, 100, 0.5};
    Contract restored = MakeContract(
        wrong_coordinates, Resources(), 0, OtherSlot(), ScratchSlot());
    const auto pre_restore = restored.Published();
    const RestoreOutcome outcome = restored.Restore(checkpoint, loaded);
    loaded = OtherSlot();
    Slot published{};
    Slot stale{};
    const bool copied = CopyPublished(restored, published);
    const bool stale_rejected = !restored.CopyPublished(pre_restore, stale);
    const BeginOutcome begin =
        restored.BeginStep({InteriorSpace::l2, checkpoint.dt});
    const auto candidate = restored.AcquireCandidate(begin.attempt);
    Slot candidate_copy{};
    const bool candidate_matches = candidate &&
        restored.CopyCandidate(*candidate, candidate_copy) &&
        candidate_copy.interior == AcceptedSlot().interior &&
        candidate_copy.trace == AcceptedSlot().trace;
    passed = Check(outcome.restored && copied && stale_rejected &&
                       published.interior == AcceptedSlot().interior &&
                       published.trace == AcceptedSlot().trace &&
                       SameCoordinates(restored.coordinates(),
                                       checkpoint.coordinates) &&
                       candidate_matches,
                   "restore_atomically_overwrites_wrong_private_state_and_detaches_input") &&
        passed;
  }

  Slot substituted = AcceptedSlot();
  substituted.interior[0] += 1.0;
  passed = ExpectRestoreFailure(
               checkpoint,
               substituted,
               RestartCompatibility::interior_hash_mismatch,
               [](CheckpointDescriptor&) {},
               "substituted_interior_buffer_rejected_atomically") &&
      passed;
  substituted = AcceptedSlot();
  substituted.trace[0] += 1.0;
  passed = ExpectRestoreFailure(
               checkpoint,
               substituted,
               RestartCompatibility::trace_hash_mismatch,
               [](CheckpointDescriptor&) {},
               "substituted_trace_buffer_rejected_atomically") &&
      passed;

  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::no_successful_commit,
               [](CheckpointDescriptor& value) {
                 value.successful_commit = false;
               },
               "altered_checkpoint_success_flag_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::schema_mismatch,
               [](CheckpointDescriptor& value) { ++value.schema_version; },
               "altered_checkpoint_schema_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::policy_mismatch,
               [](CheckpointDescriptor& value) {
                 value.restart_mode = static_cast<RestartMode>(255);
               },
               "altered_restart_policy_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::coordinate_mismatch,
               [](CheckpointDescriptor& value) { value.coordinates.time += 1.0; },
               "altered_coordinate_time_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::coordinate_mismatch,
               [](CheckpointDescriptor& value) { ++value.coordinates.step; },
               "altered_coordinate_step_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::coordinate_mismatch,
               [](CheckpointDescriptor& value) {
                 ++value.coordinates.generation;
               },
               "altered_coordinate_generation_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::coordinate_mismatch,
               [](CheckpointDescriptor& value) { ++value.coordinates.cursor; },
               "altered_coordinate_cursor_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::coordinate_mismatch,
               [](CheckpointDescriptor& value) { value.coordinates.dt *= 2.0; },
               "altered_coordinate_dt_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::coordinate_mismatch,
               [](CheckpointDescriptor& value) {
                 value.coordinate_fingerprint ^= 1U;
               },
               "altered_coordinate_fingerprint_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::cursor_mismatch,
               [](CheckpointDescriptor& value) { ++value.cursor; },
               "altered_checkpoint_cursor_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::dt_mismatch,
               [](CheckpointDescriptor& value) { value.dt *= 2.0; },
               "altered_checkpoint_dt_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::backend_mismatch,
               [](CheckpointDescriptor& value) {
                 value.tuple.backend = StableIdentity{"cuda.rank1"};
               },
               "altered_backend_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::tuple_mismatch,
               [](CheckpointDescriptor& value) {
                 value.tuple.model = StableIdentity{"other.model"};
               },
               "altered_tuple_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::tuple_mismatch,
               [](CheckpointDescriptor& value) {
                 value.tuple.formulation = StableIdentity{"other.formulation"};
               },
               "altered_formulation_tuple_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::tuple_mismatch,
               [](CheckpointDescriptor& value) {
                 value.tuple.trace = StableIdentity{"other.trace"};
               },
               "altered_trace_tuple_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::tuple_mismatch,
               [](CheckpointDescriptor& value) {
                 value.tuple.stabilization =
                     StableIdentity{"other.stabilization"};
               },
               "altered_stabilization_tuple_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::tuple_mismatch,
               [](CheckpointDescriptor& value) {
                 value.tuple.time_treatment = StableIdentity{"other.time"};
               },
               "altered_time_tuple_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::tuple_mismatch,
               [](CheckpointDescriptor& value) {
                 value.tuple.precision = StableIdentity{"fp32"};
               },
               "altered_tuple_precision_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::mfem_pin_mismatch,
               [](CheckpointDescriptor& value) {
                 value.mfem_pin = StableIdentity{"wrong-pin"};
               },
               "altered_mfem_pin_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::mesh_mismatch,
               [](CheckpointDescriptor& value) { value.mesh_hash.bytes[0] ^= 1U; },
               "altered_mesh_hash_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::partition_mismatch,
               [](CheckpointDescriptor& value) {
                 value.partition_hash.bytes[0] ^= 1U;
               },
               "altered_partition_hash_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::topology_mismatch,
               [](CheckpointDescriptor& value) {
                 value.topology_hash.bytes[0] ^= 1U;
               },
               "altered_topology_hash_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::rank_count_mismatch,
               [](CheckpointDescriptor& value) { ++value.rank_count; },
               "altered_rank_count_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::order_mismatch,
               [](CheckpointDescriptor& value) { ++value.polynomial_order; },
               "altered_order_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::layout_mismatch,
               [](CheckpointDescriptor& value) {
                 value.layout = StableIdentity{"other.layout"};
               },
               "altered_layout_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::precision_mismatch,
               [](CheckpointDescriptor& value) {
                 value.precision = StableIdentity{"fp32"};
               },
               "altered_precision_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::interior_hash_mismatch,
               [](CheckpointDescriptor& value) {
                 value.interior_hash.bytes[0] ^= 1U;
               },
               "altered_interior_hash_rejected") &&
      passed;
  passed = ExpectRestoreFailure(
               checkpoint,
               AcceptedSlot(),
               RestartCompatibility::trace_hash_mismatch,
               [](CheckpointDescriptor& value) {
                 value.trace_hash.bytes[0] ^= 1U;
               },
               "altered_trace_hash_rejected") &&
      passed;

  for (const double time : {
           std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity()})
  {
    passed = ExpectRestoreFailure(
                 checkpoint,
                 AcceptedSlot(),
                 RestartCompatibility::time_mismatch,
                 [time](CheckpointDescriptor& value) {
                   value.coordinates.time = time;
                   value.coordinate_fingerprint =
                       FrozenCoordinateFingerprint(value.coordinates);
                 },
                 "nonfinite_time_rejected_even_with_recomputed_fingerprint") &&
        passed;
  }
  for (const double dt : {
           std::numeric_limits<double>::quiet_NaN(),
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity(),
           0.0,
           -0.125})
  {
    passed = ExpectRestoreFailure(
                 checkpoint,
                 AcceptedSlot(),
                 RestartCompatibility::dt_mismatch,
                 [dt](CheckpointDescriptor& value) {
                   value.dt = dt;
                   value.coordinates.dt = dt;
                   value.coordinate_fingerprint =
                       FrozenCoordinateFingerprint(value.coordinates);
                 },
                 "checkpoint_dt_must_be_positive_and_finite") && passed;
  }

  {
    Contract contract = MakeContract();
    const BeginOutcome begin =
        contract.BeginStep({InteriorSpace::l2, 0.125});
    const RestoreOutcome busy = contract.Restore(checkpoint, AcceptedSlot());
    passed = Check(begin.started && !busy.restored &&
                       busy.compatibility == RestartCompatibility::contract_busy &&
                       TestCollectivePolicy::restores == 1 &&
                       contract.telemetry().first_failure ==
                           FailureReason::restart_incompatible,
                   "active_attempt_blocks_restore") && passed;
  }
  return passed;
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckStaticSelectionAndCollectiveAdmission() && passed;
  passed = CheckCapabilitiesAndFailureAtomicPublication() && passed;
  passed = CheckProductProtocolAndIncarnationABA() && passed;
  passed = CheckAbandonedRetryAdmitsFreshTimestep() && passed;
  passed = CheckStaleFinishIsolation() && passed;
  passed = CheckFirstFailureAcrossSuccessfulRetry() && passed;
  passed = CheckResidualTelemetryAndCollectiveTruth() && passed;
  passed = CheckDeclaredResidualLimits() && passed;
  passed = CheckTwoPhaseDetectionLegacyAndReentrantAdmission() && passed;
  passed = CheckTwoPhaseFreezePrepareCommitFinalize() && passed;
  passed = CheckTwoPhasePrepareFailuresAbortAndRevert() && passed;
  passed = CheckTwoPhaseBindingsTamperAndOverflow() && passed;
  passed = CheckTwoPhaseTokenOwnershipABAAndBusyReentrancy() && passed;
  passed = CheckExternalEnumAndMaskDomains() && passed;
  passed = CheckAtomicCheckpointRestore() && passed;
  return passed ? 0 : 1;
}

#include "numerics/hdg/runtime/run_contract.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
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

struct ArrayStoragePolicy
{
  static inline std::uint64_t preparations = 0;
  static inline std::uint64_t discards = 0;
  static inline std::uint64_t commits = 0;

  static void Reset() noexcept
  {
    preparations = 0;
    discards = 0;
    commits = 0;
  }

  static bool PrepareCandidate(const Slot& committed,
                               Slot& candidate) noexcept
  {
    ++preparations;
    candidate = committed;
    return true;
  }

  static void DiscardCandidate(Slot& candidate) noexcept
  {
    ++discards;
    candidate.interior.fill(-900.0);
    candidate.trace.fill(-901.0);
  }

  static void Commit(Slot& committed, Slot& candidate) noexcept
  {
    ++commits;
    using std::swap;
    swap(committed, candidate);
  }

  template <typename Values>
  static StateHash Hash(const Values& values) noexcept
  {
    StateHash hash{};
    std::uint64_t state = 1469598103934665603ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    for (std::size_t i = 0; i < sizeof(typename Values::value_type) * values.size();
         ++i)
    {
      state ^= bytes[i];
      state *= 1099511628211ULL;
    }
    for (std::size_t i = 0; i < hash.bytes.size(); ++i)
    {
      state ^= state >> 12;
      state ^= state << 25;
      state ^= state >> 27;
      hash.bytes[i] = static_cast<std::uint8_t>(state >> ((i % 8) * 8));
    }
    return hash;
  }

  static StateHash HashInterior(const Interior& values) noexcept
  {
    return Hash(values);
  }

  static StateHash HashTrace(const Trace& values) noexcept
  {
    return Hash(values);
  }
};

struct TestCollectivePolicy
{
  static inline std::uint64_t reductions = 0;
  static inline std::uint32_t last_expected_ranks = 0;
  static inline bool force_identity_failure = false;

  static void Reset() noexcept
  {
    reductions = 0;
    last_expected_ranks = 0;
    force_identity_failure = false;
  }

  static CollectiveGateReport Reduce(const AttemptIdentity&,
                                     const GateReport& local,
                                     const std::uint32_t expected_ranks) noexcept
  {
    ++reductions;
    last_expected_ranks = expected_ranks;
    return {local, expected_ranks, !force_identity_failure};
  }
};

using Contract = PersistentRunContract<Selection,
                                       Interior,
                                       Trace,
                                       ArrayStoragePolicy,
                                       TestCollectivePolicy>;

static_assert(Selection::interior_space == InteriorSpace::l2);
static_assert(!std::is_polymorphic<Selection>::value);
static_assert(!std::is_polymorphic<Contract>::value);
static_assert(!std::is_polymorphic<ArrayStoragePolicy>::value);
static_assert(!std::is_polymorphic<TestCollectivePolicy>::value);
static_assert(std::is_const<std::remove_reference_t<decltype(
                  (std::declval<const Contract&>().Old().interior))>>::value,
              "old interior must be immutable");
static_assert(std::is_const<std::remove_reference_t<decltype(
                  (std::declval<const Contract&>().Old().trace))>>::value,
              "old trace must be immutable");
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

StateCoordinates InitialCoordinates()
{
  return {1.0, 4, 7, 9, 0.25};
}

PersistentResourceDescriptor Resources()
{
  return {"d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec",
          Digest(10),
          Digest(20),
          Digest(30),
          1,
          3,
          "mfem.bynodes.fp64.v1",
          {true, true, true, false, 41, 53, 4096, 2048}};
}

Contract MakeContract()
{
  ArrayStoragePolicy::Reset();
  TestCollectivePolicy::Reset();
  return {CommittedSlot(), ScratchSlot(), InitialCoordinates(), Resources()};
}

AttemptReport PassingReport(const std::uint64_t cursor)
{
  AttemptReport report{};
  report.gates = {true, true, true, true, true, true};
  report.telemetry.transfers = {30, 40, 5, 0, 2, true};
  report.telemetry.residual_products = 1;
  report.telemetry.matrix_products = 1;
  report.telemetry.factor_products = 1;
  report.telemetry.solution_products = 1;
  report.telemetry.true_relative_residual = 5.0e-12;
  report.candidate_cursor = cursor;
  return report;
}

template <typename T>
std::array<unsigned char, sizeof(T)> ObjectBytes(const T& value)
{
  static_assert(std::is_trivially_copyable<T>::value,
                "byte preservation probes require trivial values");
  std::array<unsigned char, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  return bytes;
}

bool Check(const bool condition, const char* requirement)
{
  if (!condition)
  {
    std::cerr << "FAIL requirement=" << requirement << '\n';
  }
  return condition;
}

bool CheckStaticSelectionAndH1()
{
  bool passed = true;
  const RuntimeTupleDescriptor tuple = Selection::Descriptor();
  passed = Check(tuple.model == EulerModelTag::id &&
                     tuple.formulation == ConventionalFormulationTag::id &&
                     tuple.backend == CpuBackendTag::id &&
                     tuple.precision == Fp64Tag::id,
                 "static_outer_selection") && passed;

  const SmoothAdmissionThresholds<double> thresholds{};
  passed = Check(!thresholds.minimum_rate.specified &&
                     !thresholds.maximum_conservation_error.specified,
                 "smooth_thresholds_unset") && passed;

  Contract contract = MakeContract();
  const auto original_interior = contract.Published().interior;
  const auto original_trace = contract.Published().trace;
  const StateCoordinates original_coordinates = contract.coordinates();
  const BeginOutcome begin = contract.BeginStep({InteriorSpace::h1, 0.125});
  passed = Check(!begin.started &&
                     begin.reason == FailureReason::unsupported_h1 &&
                     contract.status() == RunStatus::unsupported_h1 &&
                     contract.Published().interior == original_interior &&
                     contract.Published().trace == original_trace &&
                     contract.coordinates() == original_coordinates &&
                     ArrayStoragePolicy::preparations == 0,
                 "h1_explicitly_unsupported") && passed;

  Contract invalid = MakeContract();
  const BeginOutcome invalid_begin = invalid.BeginStep(
      {static_cast<InteriorSpace>(255), 0.125});
  passed = Check(!invalid_begin.started &&
                     invalid.status() == RunStatus::invalid_request &&
                     invalid_begin.reason ==
                         FailureReason::invalid_interior_space,
                 "invalid_interior_space_fails_closed") && passed;
  return passed;
}

bool CheckAttemptProductsRetryAndCommit()
{
  Contract contract = MakeContract();
  const Slot original = CommittedSlot();
  const StateCoordinates initial = contract.coordinates();
  const PersistentResourceDescriptor resources = contract.resources();
  const BeginOutcome first = contract.BeginStep({InteriorSpace::l2, 0.125});
  bool passed = Check(first.started && first.attempt.retry_index == 0 &&
                          first.attempt.committed_generation == initial.generation &&
                          first.attempt.committed_step == initial.step &&
                          first.attempt.dt == 0.125,
                      "first_attempt_identity");
  Slot* candidate = contract.Candidate(first.attempt);
  passed = Check(candidate != nullptr && candidate->interior == original.interior &&
                     candidate->trace == original.trace,
                 "candidate_prepared_from_committed") && passed;
  if (candidate != nullptr)
  {
    candidate->interior = {{11.0, 12.0, 13.0, 14.0}};
    candidate->trace = {{15.0, 16.0, 17.0}};
  }

  AttemptProduct<DerivedProductKind::residual, int> residual{};
  AttemptProduct<DerivedProductKind::matrix, int> matrix{};
  AttemptProduct<DerivedProductKind::factor, int> factor{};
  AttemptProduct<DerivedProductKind::solution, int> solution{};
  passed = Check(contract.TryTagProduct(first.attempt, 1, residual) &&
                     contract.TryTagProduct(first.attempt, 2, matrix) &&
                     contract.TryTagProduct(first.attempt, 3, factor) &&
                     contract.TryTagProduct(first.attempt, 4, solution) &&
                     contract.Accepts(residual) && contract.Accepts(matrix) &&
                     contract.Accepts(factor) && contract.Accepts(solution),
                 "attempt_local_products") && passed;

  AttemptReport failed = PassingReport(99);
  failed.gates.finite = false;
  failed.telemetry.transfers = {10, 20, 3, 0, 1, true};
  const AttemptOutcome failed_outcome =
      contract.FinishAttempt(first.attempt, failed);
  passed = Check(!failed_outcome.committed && failed_outcome.retry_available &&
                     failed_outcome.reason == FailureReason::finite_gate &&
                     contract.status() == RunStatus::retry_available &&
                     contract.Published().interior == original.interior &&
                     contract.Published().trace == original.trace &&
                     contract.coordinates() == initial &&
                     contract.Candidate(first.attempt) == nullptr &&
                     ArrayStoragePolicy::discards == 1 &&
                     ArrayStoragePolicy::commits == 0,
                 "failed_attempt_rollback") && passed;
  passed = Check(!contract.Accepts(residual) && !contract.Accepts(matrix) &&
                     !contract.Accepts(factor) && !contract.Accepts(solution),
                 "discarded_products_rejected") && passed;

  const BeginOutcome retry = contract.Retry();
  passed = Check(retry.started && retry.attempt.sequence != first.attempt.sequence &&
                     retry.attempt.retry_index == 1 &&
                     retry.attempt.committed_generation ==
                         first.attempt.committed_generation &&
                     retry.attempt.committed_step == first.attempt.committed_step &&
                     retry.attempt.dt == first.attempt.dt,
                 "single_retry_identity_and_dt") && passed;
  candidate = contract.Candidate(retry.attempt);
  passed = Check(candidate != nullptr && candidate->interior == original.interior &&
                     candidate->trace == original.trace,
                 "retry_uses_identical_committed_state") && passed;
  if (candidate != nullptr)
  {
    candidate->interior = {{21.0, 22.0, 23.0, 24.0}};
    candidate->trace = {{25.0, 26.0, 27.0}};
  }

  const AttemptReport success = PassingReport(11);
  const AttemptOutcome committed = contract.FinishAttempt(retry.attempt, success);
  const StateCoordinates expected_coordinates{1.125, 5, 8, 11, 0.125};
  passed = Check(committed.committed && !committed.retry_available &&
                     contract.status() == RunStatus::committed &&
                     contract.Published().interior ==
                         Interior{{21.0, 22.0, 23.0, 24.0}} &&
                     contract.Published().trace == Trace{{25.0, 26.0, 27.0}} &&
                     contract.coordinates() == expected_coordinates &&
                     ArrayStoragePolicy::commits == 1,
                 "atomic_successful_commit") && passed;

  const RunTelemetry& telemetry = contract.telemetry();
  passed = Check(telemetry.attempts_started == 2 &&
                     telemetry.attempts_failed == 1 && telemetry.rollbacks == 1 &&
                     telemetry.retries_started == 1 && telemetry.commits == 1 &&
                     telemetry.attempted_transfers.host_to_device_bytes == 40 &&
                     telemetry.attempted_transfers.device_to_host_bytes == 60 &&
                     telemetry.rolled_back_transfers.host_to_device_bytes == 10 &&
                     telemetry.committed_transfers.host_to_device_bytes == 30 &&
                     telemetry.accepted.valid &&
                     telemetry.accepted.coordinates == expected_coordinates &&
                     telemetry.accepted.attempt.identity == retry.attempt &&
                     telemetry.accepted.attempt.failure_reason ==
                         FailureReason::none,
                 "status_and_transfer_telemetry") && passed;

  const auto committed_bytes = ObjectBytes(contract.Published().interior);
  const StateCoordinates once = contract.coordinates();
  const AttemptOutcome duplicate = contract.FinishAttempt(retry.attempt, success);
  passed = Check(!duplicate.committed &&
                     duplicate.reason == FailureReason::stale_attempt &&
                     contract.coordinates() == once &&
                     ObjectBytes(contract.Published().interior) == committed_bytes &&
                     ArrayStoragePolicy::commits == 1,
                 "commit_advances_exactly_once") && passed;
  const BeginOutcome second_retry = contract.Retry();
  passed = Check(!second_retry.started &&
                     second_retry.reason == FailureReason::retry_exhausted,
                 "at_most_one_retry") && passed;
  passed = Check(contract.resources().topology_hash == resources.topology_hash &&
                     contract.resources().reuse.topology_generation ==
                         resources.reuse.topology_generation &&
                     contract.resources().reuse.storage_generation ==
                         resources.reuse.storage_generation,
                 "persistent_descriptor_stable") && passed;
  return passed;
}

bool CheckEveryCommitGateAndCollective()
{
  const std::array<FailureReason, 6> expected{{
      FailureReason::finite_gate,
      FailureReason::admissibility_gate,
      FailureReason::recovery_gate,
      FailureReason::true_residual_gate,
      FailureReason::telemetry_gate,
      FailureReason::backend_failure}};
  bool passed = true;
  for (std::size_t gate = 0; gate < expected.size(); ++gate)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    AttemptReport report = PassingReport(12);
    bool* gates[] = {&report.gates.finite,
                     &report.gates.admissible,
                     &report.gates.recovery,
                     &report.gates.true_residual,
                     &report.gates.telemetry,
                     &report.gates.success};
    *gates[gate] = false;
    if (gate == 5)
    {
      report.originating_failure = FailureReason::backend_failure;
    }
    const AttemptOutcome outcome = contract.FinishAttempt(begin.attempt, report);
    passed = Check(begin.started && !outcome.committed &&
                       outcome.reason == expected[gate] &&
                       contract.coordinates() == InitialCoordinates() &&
                       TestCollectivePolicy::reductions == 1 &&
                       TestCollectivePolicy::last_expected_ranks == 1,
                   "all_commit_gates_required") && passed;
  }

  Contract collective_failure = MakeContract();
  const BeginOutcome begin =
      collective_failure.BeginStep({InteriorSpace::l2, 0.125});
  TestCollectivePolicy::force_identity_failure = true;
  const AttemptOutcome outcome =
      collective_failure.FinishAttempt(begin.attempt, PassingReport(12));
  passed = Check(!outcome.committed &&
                     outcome.reason == FailureReason::collective_identity &&
                     collective_failure.coordinates() == InitialCoordinates(),
                 "collective_identity_gate") && passed;
  return passed;
}

bool CheckForbiddenActionsAndAcceptedPreservation()
{
  bool passed = true;
  Contract forbidden = MakeContract();
  const BeginOutcome forbidden_begin =
      forbidden.BeginStep({InteriorSpace::l2, 0.125});
  AttemptReport forbidden_report = PassingReport(12);
  forbidden_report.telemetry.prohibited = {1, 1, 1, 1, 1};
  const AttemptOutcome forbidden_outcome =
      forbidden.FinishAttempt(forbidden_begin.attempt, forbidden_report);
  passed = Check(!forbidden_outcome.committed &&
                     forbidden_outcome.reason == FailureReason::telemetry_gate,
                 "no_dt_clipping_solver_fallback_or_insertion") && passed;

  Contract allocations = MakeContract();
  const BeginOutcome allocation_begin =
      allocations.BeginStep({InteriorSpace::l2, 0.125});
  AttemptReport allocation_report = PassingReport(12);
  allocation_report.telemetry.transfers.allocation_events = 1;
  const AttemptOutcome allocation_outcome =
      allocations.FinishAttempt(allocation_begin.attempt, allocation_report);
  passed = Check(!allocation_outcome.committed &&
                     allocation_outcome.reason == FailureReason::telemetry_gate,
                 "persistent_run_requires_no_step_allocation") && passed;

  Contract incomplete = MakeContract();
  const BeginOutcome incomplete_begin =
      incomplete.BeginStep({InteriorSpace::l2, 0.125});
  AttemptReport incomplete_report = PassingReport(12);
  incomplete_report.telemetry.transfers.complete = false;
  const AttemptOutcome incomplete_outcome =
      incomplete.FinishAttempt(incomplete_begin.attempt, incomplete_report);
  passed = Check(!incomplete_outcome.committed &&
                     incomplete_outcome.reason == FailureReason::telemetry_gate &&
                     !incomplete.telemetry().attempted_transfers.complete &&
                     !incomplete.telemetry().rolled_back_transfers.complete,
                 "incomplete_transfer_accounting_stays_incomplete") && passed;

  Contract contract = MakeContract();
  BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
  Slot* candidate = contract.Candidate(begin.attempt);
  if (candidate != nullptr)
  {
    candidate->interior = {{31.0, 32.0, 33.0, 34.0}};
    candidate->trace = {{35.0, 36.0, 37.0}};
  }
  const AttemptOutcome first_commit =
      contract.FinishAttempt(begin.attempt, PassingReport(14));
  const auto interior_bytes = ObjectBytes(contract.Published().interior);
  const auto trace_bytes = ObjectBytes(contract.Published().trace);
  const auto accepted_bytes = ObjectBytes(contract.telemetry().accepted);
  const StateCoordinates committed_coordinates = contract.coordinates();

  begin = contract.BeginStep({InteriorSpace::l2, 0.125});
  candidate = contract.Candidate(begin.attempt);
  if (candidate != nullptr)
  {
    candidate->interior.fill(1000.0);
    candidate->trace.fill(2000.0);
  }
  AttemptReport failed = PassingReport(999);
  failed.gates.recovery = false;
  const AttemptOutcome failure = contract.FinishAttempt(begin.attempt, failed);
  passed = Check(first_commit.committed && !failure.committed &&
                     ObjectBytes(contract.Published().interior) == interior_bytes &&
                     ObjectBytes(contract.Published().trace) == trace_bytes &&
                     ObjectBytes(contract.telemetry().accepted) == accepted_bytes &&
                     contract.coordinates() == committed_coordinates,
                 "failure_preserves_committed_and_accepted_bytes") && passed;

  const BeginOutcome retry = contract.Retry();
  AttemptReport retry_failure = PassingReport(1000);
  retry_failure.gates.true_residual = false;
  const AttemptOutcome terminal =
      contract.FinishAttempt(retry.attempt, retry_failure);
  passed = Check(retry.started && !terminal.committed &&
                     terminal.status == RunStatus::terminal_failure &&
                     terminal.reason == FailureReason::true_residual_gate &&
                     ObjectBytes(contract.Published().interior) == interior_bytes &&
                     ObjectBytes(contract.Published().trace) == trace_bytes &&
                     ObjectBytes(contract.telemetry().accepted) == accepted_bytes &&
                     contract.coordinates() == committed_coordinates,
                 "retry_failure_preserves_committed_state") && passed;
  return passed;
}

bool CheckCheckpointAndRestart()
{
  Contract contract = MakeContract();
  CheckpointDescriptor checkpoint{};
  bool passed = Check(!contract.TryMakeCheckpoint(checkpoint),
                      "checkpoint_rejects_unaccepted_initial_state");
  const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
  passed = Check(!contract.TryMakeCheckpoint(checkpoint),
                 "checkpoint_rejects_active_candidate") && passed;
  Slot* candidate = contract.Candidate(begin.attempt);
  if (candidate != nullptr)
  {
    candidate->interior = {{41.0, 42.0, 43.0, 44.0}};
    candidate->trace = {{45.0, 46.0, 47.0}};
  }
  const AttemptOutcome committed =
      contract.FinishAttempt(begin.attempt, PassingReport(77));
  passed = Check(committed.committed && contract.TryMakeCheckpoint(checkpoint) &&
                     checkpoint.successful_commit &&
                     checkpoint.restart_mode ==
                         RestartMode::same_backend_rank_partition &&
                     checkpoint.tuple == Selection::Descriptor() &&
                     checkpoint.mfem_pin == Resources().mfem_pin &&
                     checkpoint.mesh_hash == Resources().mesh_hash &&
                     checkpoint.partition_hash == Resources().partition_hash &&
                     checkpoint.rank_count == 1 &&
                     checkpoint.polynomial_order == 3 &&
                     checkpoint.layout == Resources().layout &&
                     checkpoint.precision == Fp64Tag::id &&
                     checkpoint.cursor == 77 && checkpoint.dt == 0.125 &&
                     checkpoint.interior_hash == ArrayStoragePolicy::HashInterior(
                         contract.Published().interior) &&
                     checkpoint.trace_hash == ArrayStoragePolicy::HashTrace(
                         contract.Published().trace),
                 "checkpoint_exact_success_tuple") && passed;
  passed = Check(contract.CheckRestart(checkpoint) ==
                     RestartCompatibility::compatible,
                 "same_backend_rank_partition_restart") && passed;

  CheckpointDescriptor changed = checkpoint;
  changed.tuple.backend = "cuda.rank1";
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::backend_mismatch,
                 "restart_backend_mismatch") && passed;
  changed = checkpoint;
  changed.tuple.formulation = "igr.scalar";
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::tuple_mismatch,
                 "restart_tuple_mismatch") && passed;
  changed = checkpoint;
  changed.mfem_pin = "different";
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::mfem_pin_mismatch,
                 "restart_mfem_mismatch") && passed;
  changed = checkpoint;
  changed.rank_count = 2;
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::rank_count_mismatch,
                 "restart_rank_mismatch") && passed;
  changed = checkpoint;
  changed.partition_hash = Digest(21);
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::partition_mismatch,
                 "restart_partition_mismatch") && passed;
  changed = checkpoint;
  changed.mesh_hash = Digest(11);
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::mesh_mismatch,
                 "restart_mesh_mismatch") && passed;
  changed = checkpoint;
  changed.polynomial_order = 4;
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::order_mismatch,
                 "restart_order_mismatch") && passed;
  changed = checkpoint;
  changed.layout = "other-layout";
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::layout_mismatch,
                 "restart_layout_mismatch") && passed;
  changed = checkpoint;
  changed.precision = "fp32";
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::precision_mismatch,
                 "restart_precision_mismatch") && passed;
  changed = checkpoint;
  changed.successful_commit = false;
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::no_successful_commit,
                 "restart_requires_successful_commit") && passed;
  changed = checkpoint;
  changed.restart_mode = static_cast<RestartMode>(255);
  passed = Check(contract.CheckRestart(changed) ==
                     RestartCompatibility::policy_mismatch,
                 "restart_policy_explicit") && passed;

  const BeginOutcome later = contract.BeginStep({InteriorSpace::l2, 0.125});
  passed = Check(!contract.TryMakeCheckpoint(changed),
                 "checkpoint_not_from_private_candidate") && passed;
  AttemptReport failed = PassingReport(88);
  failed.gates.admissible = false;
  contract.FinishAttempt(later.attempt, failed);
  CheckpointDescriptor preserved{};
  passed = Check(contract.TryMakeCheckpoint(preserved) &&
                     preserved.coordinates == checkpoint.coordinates &&
                     preserved.interior_hash == checkpoint.interior_hash &&
                     preserved.trace_hash == checkpoint.trace_hash,
                 "checkpoint_after_rollback_uses_committed_state") && passed;
  return passed;
}

bool CheckInvalidPersistentDescriptor()
{
  PersistentResourceDescriptor resources = Resources();
  resources.reuse.persistent_workspace = false;
  Contract contract{CommittedSlot(), ScratchSlot(), InitialCoordinates(), resources};
  const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
  return Check(!begin.started &&
                   begin.reason == FailureReason::invalid_resource_descriptor &&
                   contract.Published().interior == CommittedSlot().interior &&
                   contract.coordinates() == InitialCoordinates(),
               "persistent_descriptor_required");
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckStaticSelectionAndH1() && passed;
  passed = CheckAttemptProductsRetryAndCommit() && passed;
  passed = CheckEveryCommitGateAndCollective() && passed;
  passed = CheckForbiddenActionsAndAcceptedPreservation() && passed;
  passed = CheckCheckpointAndRestart() && passed;
  passed = CheckInvalidPersistentDescriptor() && passed;
  if (passed)
  {
    std::cout << "PASS hdg_runtime_contract_cxx17"
              << " attempts=2 retries=1 commits=1 checkpoint=exact"
              << " restart=same_backend_rank_partition\n";
  }
  return passed ? 0 : 1;
}

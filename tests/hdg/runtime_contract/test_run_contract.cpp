#include "numerics/hdg/runtime/run_contract.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
  static constexpr bool commit_is_atomic = true;
  static constexpr bool commit_preserves_disjoint_backing = true;
  static inline std::uint64_t preparations = 0;
  static inline std::uint64_t discards = 0;
  static inline std::uint64_t commits = 0;
  static inline std::uint64_t fail_prepare_call = 0;

  static void Reset() noexcept
  {
    preparations = 0;
    discards = 0;
    commits = 0;
    fail_prepare_call = 0;
  }

  static bool Disjoint(const Slot& committed, const Slot& candidate) noexcept
  {
    return committed.interior.data() != candidate.interior.data() &&
        committed.trace.data() != candidate.trace.data();
  }

  static bool PrepareCandidate(const Slot& committed,
                               Slot& candidate) noexcept
  {
    ++preparations;
    if (preparations == fail_prepare_call) { return false; }
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
  static StateHash Hash(const Values& values, const int offset) noexcept
  {
    StateHash hash{};
    for (std::size_t i = 0; i < hash.bytes.size(); ++i)
    {
      hash.bytes[i] = static_cast<std::uint8_t>(
          static_cast<int>(values[i % values.size()]) +
          static_cast<int>(i) + offset);
    }
    return hash;
  }

  static StateHash HashInterior(const Interior& values) noexcept
  {
    return Hash(values, 0);
  }

  static StateHash HashTrace(const Trace& values) noexcept
  {
    return Hash(values, 100);
  }
};

enum class CollectiveFault : unsigned char
{
  none,
  admission_remote_preparation,
  admission_participants,
  admission_committed_time,
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
  finish_participants,
  remote_finite,
  remote_admissible,
  remote_recovery,
  remote_true_residual,
  remote_telemetry,
  remote_success
};

struct TestCollectivePolicy
{
  static inline CollectiveFault fault = CollectiveFault::none;
  static inline std::uint64_t admissions = 0;
  static inline std::uint64_t reductions = 0;

  static void Reset() noexcept
  {
    fault = CollectiveFault::none;
    admissions = 0;
    reductions = 0;
  }

  static CollectiveAdmissionReport Admit(
      const AttemptCollectiveContext& local,
      const bool local_prepared,
      const std::uint32_t expected_ranks) noexcept
  {
    ++admissions;
    CollectiveAdmissionReport report{local, expected_ranks, local_prepared};
    if (fault == CollectiveFault::admission_remote_preparation)
    {
      report.all_prepared = false;
    }
    else if (fault == CollectiveFault::admission_participants)
    {
      report.participating_ranks = expected_ranks + 1;
    }
    else if (fault == CollectiveFault::admission_committed_time)
    {
      report.context.committed.time += 1.0;
    }
    return report;
  }

  static CollectiveGateReport Reduce(
      const AttemptCollectiveContext& local,
      const GateReport&,
      const std::uint32_t expected_ranks) noexcept
  {
    ++reductions;
    CollectiveGateReport report{
        local, {true, true, true, true, true, true}, expected_ranks};
    switch (fault)
    {
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
      case CollectiveFault::finish_participants:
        report.participating_ranks = expected_ranks + 1;
        break;
      case CollectiveFault::remote_finite:
        report.gates.finite = false;
        break;
      case CollectiveFault::remote_admissible:
        report.gates.admissible = false;
        break;
      case CollectiveFault::remote_recovery:
        report.gates.recovery = false;
        break;
      case CollectiveFault::remote_true_residual:
        report.gates.true_residual = false;
        break;
      case CollectiveFault::remote_telemetry:
        report.gates.telemetry = false;
        break;
      case CollectiveFault::remote_success:
        report.gates.success = false;
        break;
      default:
        break;
    }
    return report;
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
static_assert(!std::is_default_constructible<
                  Contract::ProductToken<DerivedProductKind::residual>>::value,
              "attempt products must not be forgeable by construction");
static_assert(!std::is_copy_constructible<
                  Contract::ProductToken<DerivedProductKind::residual>>::value,
              "attempt products must not be duplicable by copying");
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
                      const std::uint64_t initial_sequence = 0)
{
  ArrayStoragePolicy::Reset();
  TestCollectivePolicy::Reset();
  return Contract{
      CommittedSlot(), ScratchSlot(), coordinates, resources, initial_sequence};
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

bool MutateCandidate(Contract& contract,
                     const AttemptIdentity& attempt,
                     const Slot& value)
{
  auto handle = contract.AcquireCandidate(attempt);
  return handle && handle->Mutate(
      [&value](Slot& candidate) noexcept { candidate = value; });
}

bool CopyPublished(const Contract& contract, Slot& copy)
{
  return contract.Published().Copy(copy);
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

bool CheckStaticSelectionIdentityAndInvalidRequests()
{
  bool passed = true;
  const RuntimeTupleDescriptor tuple = Selection::Descriptor();
  passed = Check(tuple.Valid() &&
                     tuple.model == StableIdentity{"euler.compressible.2d"} &&
                     tuple.backend == StableIdentity{"cpu.serial"} &&
                     tuple.precision == StableIdentity{"fp64"},
                 "static_owned_outer_selection") && passed;

  std::array<char, 16> local_text{{'t', 'e', 'm', 'p', '.', 'i', 'd'}};
  const StableIdentity owned = StableIdentity::Copy(
      std::string_view{local_text.data(), 7});
  local_text.fill('x');
  passed = Check(owned == StableIdentity{"temp.id"},
                 "text_identity_owns_source_lifetime") && passed;

  const SmoothAdmissionThresholds<double> thresholds{};
  passed = Check(!thresholds.minimum_rate.specified &&
                     !thresholds.maximum_conservation_error.specified,
                 "smooth_thresholds_remain_unset") && passed;

  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::h1, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::unsupported_h1 &&
                       contract.status() == RunStatus::unsupported_h1 &&
                       ArrayStoragePolicy::preparations == 0 &&
                       TestCollectivePolicy::admissions == 0,
                   "h1_explicitly_unsupported") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep(
        {static_cast<InteriorSpace>(255), 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::invalid_interior_space,
                   "invalid_interior_space_fails_closed") && passed;
  }

  const std::array<double, 4> invalid_dt{{
      0.0,
      -1.0,
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()}};
  for (const double dt : invalid_dt)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, dt});
    passed = Check(!begin.started && begin.reason == FailureReason::invalid_dt &&
                       TestCollectivePolicy::admissions == 0,
                   "invalid_or_nonfinite_dt_rejected") && passed;
  }

  {
    StateCoordinates coordinates = InitialCoordinates();
    coordinates.step = std::numeric_limits<std::uint64_t>::max();
    Contract contract = MakeContract(coordinates);
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::coordinate_overflow,
                   "coordinate_overflow_rejected") && passed;
  }
  {
    Contract contract = MakeContract(
        InitialCoordinates(), Resources(),
        std::numeric_limits<std::uint64_t>::max());
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::coordinate_overflow,
                   "attempt_sequence_overflow_rejected") && passed;
  }
  {
    PersistentResourceDescriptor resources = Resources();
    resources.reuse.persistent_workspace = false;
    Contract contract = MakeContract(InitialCoordinates(), resources);
    passed = Check(contract.status() == RunStatus::invalid_request &&
                       contract.last_failure() ==
                           FailureReason::invalid_resource_descriptor,
                   "invalid_persistent_descriptor_rejected") && passed;
  }
  return passed;
}

bool CheckCollectiveAdmissionAndPreparation()
{
  bool passed = true;
  {
    Contract contract = MakeContract();
    ArrayStoragePolicy::fail_prepare_call = 1;
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    Slot published{};
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::candidate_preparation &&
                       contract.status() == RunStatus::terminal_failure &&
                       TestCollectivePolicy::admissions == 1 &&
                       contract.telemetry().admissions_failed == 1 &&
                       contract.telemetry().attempts_started == 0 &&
                       CopyPublished(contract, published) &&
                       published.interior == CommittedSlot().interior,
                   "initial_local_prepare_failure_collectively_admitted") &&
        passed;
  }
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault =
        CollectiveFault::admission_remote_preparation;
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::candidate_preparation &&
                       ArrayStoragePolicy::preparations == 1 &&
                       ArrayStoragePolicy::discards == 1 &&
                       TestCollectivePolicy::admissions == 1,
                   "remote_prepare_failure_atomic_before_activation") && passed;
  }
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault = CollectiveFault::admission_participants;
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::collective_identity &&
                       contract.telemetry().attempts_started == 0,
                   "admission_participant_mismatch") && passed;
  }
  {
    Contract contract = MakeContract();
    TestCollectivePolicy::fault = CollectiveFault::admission_committed_time;
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::collective_identity,
                   "admission_coordinate_mismatch") && passed;
  }
  {
    Contract contract = MakeContract();
    BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.gates.finite = false;
    const AttemptOutcome failed = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    ArrayStoragePolicy::fail_prepare_call = 2;
    const BeginOutcome retry = contract.Retry();
    passed = Check(begin.started && failed.retry_available && !retry.started &&
                       retry.reason == FailureReason::candidate_preparation &&
                       TestCollectivePolicy::admissions == 2 &&
                       contract.status() == RunStatus::terminal_failure &&
                       contract.coordinates() == InitialCoordinates(),
                   "retry_prepare_failure_collectively_admitted") && passed;
  }
  return passed;
}

struct BackingState
{
  int backing = 0;
  double value = 0.0;
};

using BackingSlot = CompleteStateSlot<BackingState, BackingState>;

struct BackingStoragePolicy
{
  static constexpr bool commit_is_atomic = true;
  static constexpr bool commit_preserves_disjoint_backing = true;
  static inline bool alias_on_prepare = false;
  static inline bool alias_on_commit = false;

  static void Reset() noexcept
  {
    alias_on_prepare = false;
    alias_on_commit = false;
  }

  static bool Disjoint(const BackingSlot& committed,
                       const BackingSlot& candidate) noexcept
  {
    return committed.interior.backing != candidate.interior.backing &&
        committed.trace.backing != candidate.trace.backing;
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

  static void DiscardCandidate(BackingSlot& candidate) noexcept
  {
    candidate.interior.value = -1.0;
    candidate.trace.value = -1.0;
  }

  static void Commit(BackingSlot& committed, BackingSlot& candidate) noexcept
  {
    if (alias_on_commit)
    {
      candidate.interior.backing = committed.interior.backing;
      candidate.trace.backing = committed.trace.backing;
      return;
    }
    using std::swap;
    swap(committed.interior.value, candidate.interior.value);
    swap(committed.trace.value, candidate.trace.value);
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
      {shared ? 1 : 3, -10.0}, {shared ? 2 : 4, -20.0}};
  return BackingContract{
      committed, candidate, InitialCoordinates(), Resources()};
}

bool CheckDisjointStorageAndBoundHandles()
{
  bool passed = true;
  {
    BackingContract contract = MakeBackingContract(true);
    passed = Check(contract.status() == RunStatus::invalid_request &&
                       contract.last_failure() ==
                           FailureReason::shared_backing_store,
                   "shared_backing_rejected_at_construction") && passed;
  }
  {
    BackingContract contract = MakeBackingContract();
    BackingStoragePolicy::alias_on_prepare = true;
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    passed = Check(!begin.started &&
                       begin.reason == FailureReason::shared_backing_store &&
                       TestCollectivePolicy::admissions == 1,
                   "shared_backing_rejected_after_preparation") && passed;
  }
  {
    BackingContract contract = MakeBackingContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto handle = contract.AcquireCandidate(begin.attempt);
    passed = Check(handle && handle->Mutate([](BackingSlot& slot) noexcept {
                     slot.interior.value = 30.0;
                     slot.trace.value = 40.0;
                   }),
                   "backing_candidate_handle_active") && passed;
    auto products = IssueAllFor(contract, begin.attempt);
    BackingStoragePolicy::alias_on_commit = true;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(15), std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::shared_backing_store &&
                       contract.status() == RunStatus::terminal_failure,
                   "shared_backing_rejected_after_commit_boundary") && passed;
  }
  {
    Contract contract = MakeContract();
    const auto old = contract.Old();
    Slot old_before{};
    BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto retained_candidate = contract.AcquireCandidate(begin.attempt);
    const Slot accepted{{{21.0, 22.0, 23.0, 24.0}}, {{25.0, 26.0, 27.0}}};
    passed = Check(old.Copy(old_before) && retained_candidate &&
                       retained_candidate->Mutate(
                           [&accepted](Slot& slot) noexcept { slot = accepted; }),
                   "handles_read_or_mutate_only_while_fresh") && passed;
    auto products = IssueAll(contract, begin.attempt);
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(11), std::move(products));
    Slot stale_copy{};
    Slot fresh_copy{};
    passed = Check(outcome.committed && !old.Copy(stale_copy) &&
                       !retained_candidate->Mutate([](Slot& slot) noexcept {
                         slot.interior.fill(999.0);
                       }) &&
                       contract.Published().Copy(fresh_copy) &&
                       fresh_copy.interior == accepted.interior &&
                       fresh_copy.trace == accepted.trace,
                   "retained_old_and_candidate_handles_fail_stale") && passed;
  }
  return passed;
}

bool CheckIssuedProductProtocol()
{
  bool passed = true;
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto residual =
        contract.IssueProduct<DerivedProductKind::residual>(begin.attempt);
    auto duplicate =
        contract.IssueProduct<DerivedProductKind::residual>(begin.attempt);
    passed = Check(residual && !duplicate &&
                       contract.telemetry().duplicate_product_issues_rejected ==
                           1,
                   "duplicate_product_issuance_rejected") && passed;
  }

  for (std::size_t missing = 0; missing < 4; ++missing)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    if (missing == 0) { products.residual.reset(); }
    if (missing == 1) { products.matrix.reset(); }
    if (missing == 2) { products.factor.reset(); }
    if (missing == 3) { products.solution.reset(); }
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    passed = Check(!outcome.committed && outcome.retry_available &&
                       outcome.reason == FailureReason::missing_product,
                   "each_missing_required_product_rejected") && passed;
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
                   "cross_contract_forged_products_rejected") && passed;
  }

  {
    Contract contract = MakeContract();
    const BeginOutcome first = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto stale_products = IssueAll(contract, first.attempt);
    Contract::ProductBundle missing{};
    const AttemptOutcome first_failure = contract.FinishAttempt(
        first.attempt, PassingReport(12), std::move(missing));
    const BeginOutcome retry = contract.Retry();
    const AttemptOutcome stale = contract.FinishAttempt(
        retry.attempt, PassingReport(12), std::move(stale_products));
    passed = Check(first_failure.retry_available && retry.started &&
                       !stale.committed &&
                       stale.reason == FailureReason::stale_product,
                   "stale_attempt_products_rejected") && passed;
  }

  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    const AttemptOutcome committed = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    const AttemptOutcome duplicate_use = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    passed = Check(committed.committed && !duplicate_use.committed &&
                       duplicate_use.reason == FailureReason::stale_attempt &&
                       contract.telemetry().commits == 1,
                   "duplicate_product_use_cannot_recommit") && passed;
  }
  return passed;
}

bool CheckResidualOriginAndTransferGates()
{
  bool passed = true;
  const std::array<double, 3> rejected{{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
      1.0001e-10}};
  for (const double residual : rejected)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.telemetry.true_relative_residual = residual;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::true_residual_gate,
                   "nonfinite_or_above_limit_true_residual_rejected") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.telemetry.true_relative_residual = 1.0e-10;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(outcome.committed,
                   "exact_frozen_true_residual_limit_admitted") && passed;
  }
  {
    Contract contract = MakeContract();
    BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.originating_failure = FailureReason::backend_failure;
    const AttemptOutcome failed = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    const AttemptOutcome stale_duplicate = contract.FinishAttempt(
        begin.attempt, report, Contract::ProductBundle{});
    const BeginOutcome retry = contract.Retry();
    auto retry_products = IssueAll(contract, retry.attempt);
    const AttemptOutcome committed = contract.FinishAttempt(
        retry.attempt, PassingReport(13), std::move(retry_products));
    passed = Check(!failed.committed && failed.retry_available &&
                       failed.reason == FailureReason::backend_failure &&
                       stale_duplicate.status == RunStatus::retry_available &&
                       stale_duplicate.retry_available &&
                       stale_duplicate.reason == FailureReason::stale_attempt &&
                       retry.started && committed.committed &&
                       contract.telemetry().first_originating_failure ==
                           FailureReason::backend_failure,
                   "originating_failure_and_retry_state_preserved") && passed;
  }
  {
    Contract contract = MakeContract();
    BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport huge = PassingReport(12);
    huge.telemetry.transfers.host_to_device_bytes =
        std::numeric_limits<std::uint64_t>::max();
    const AttemptOutcome first = contract.FinishAttempt(
        begin.attempt, huge, std::move(products));
    begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    products = IssueAll(contract, begin.attempt);
    AttemptReport overflow = PassingReport(13);
    overflow.telemetry.transfers.host_to_device_bytes = 1;
    const AttemptOutcome second = contract.FinishAttempt(
        begin.attempt, overflow, std::move(products));
    passed = Check(first.committed && !second.committed &&
                       second.reason == FailureReason::telemetry_gate &&
                       !contract.telemetry().attempted_transfers.complete &&
                       contract.telemetry().attempted_transfers
                               .host_to_device_bytes ==
                           std::numeric_limits<std::uint64_t>::max(),
                   "transfer_overflow_marks_accounting_inexact") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.telemetry.prohibited = {1, 1, 1, 1, 1};
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::telemetry_gate,
                   "dt_change_clipping_substitution_fallback_insertion_forbidden") &&
        passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.telemetry.transfers.allocation_events = 1;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::telemetry_gate,
                   "persistent_run_rejects_per_step_allocation") && passed;
  }
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    report.telemetry.transfers.complete = false;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed &&
                       outcome.reason == FailureReason::telemetry_gate &&
                       !contract.telemetry().attempted_transfers.complete &&
                       !contract.telemetry().rolled_back_transfers.complete,
                   "incomplete_transfer_accounting_stays_incomplete") && passed;
  }
  return passed;
}

bool CheckCollectiveIdentityAndRemoteGates()
{
  struct Case
  {
    CollectiveFault fault;
    FailureReason expected;
  };
  const std::array<Case, 19> cases{{
      {CollectiveFault::finish_attempt, FailureReason::collective_identity},
      {CollectiveFault::finish_committed_time,
       FailureReason::collective_identity},
      {CollectiveFault::finish_committed_step,
       FailureReason::collective_identity},
      {CollectiveFault::finish_committed_generation,
       FailureReason::collective_identity},
      {CollectiveFault::finish_committed_cursor,
       FailureReason::collective_identity},
      {CollectiveFault::finish_committed_dt,
       FailureReason::collective_identity},
      {CollectiveFault::finish_target_time, FailureReason::collective_identity},
      {CollectiveFault::finish_target_step, FailureReason::collective_identity},
      {CollectiveFault::finish_target_generation,
       FailureReason::collective_identity},
      {CollectiveFault::finish_target_cursor,
       FailureReason::collective_identity},
      {CollectiveFault::finish_target_dt, FailureReason::collective_identity},
      {CollectiveFault::finish_candidate_cursor,
       FailureReason::collective_identity},
      {CollectiveFault::finish_participants,
       FailureReason::collective_identity},
      {CollectiveFault::remote_finite, FailureReason::finite_gate},
      {CollectiveFault::remote_admissible,
       FailureReason::admissibility_gate},
      {CollectiveFault::remote_recovery, FailureReason::recovery_gate},
      {CollectiveFault::remote_true_residual,
       FailureReason::true_residual_gate},
      {CollectiveFault::remote_telemetry, FailureReason::telemetry_gate},
      {CollectiveFault::remote_success, FailureReason::success_gate}}};

  bool passed = true;
  for (const Case& test_case : cases)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    TestCollectivePolicy::fault = test_case.fault;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, PassingReport(12), std::move(products));
    passed = Check(begin.started && !outcome.committed &&
                       outcome.reason == test_case.expected &&
                       contract.coordinates() == InitialCoordinates() &&
                       TestCollectivePolicy::reductions == 1,
                   "independent_collective_context_or_gate_mismatch") && passed;
  }
  return passed;
}

bool CheckAtomicRetryCommitAndTelemetry()
{
  Contract contract = MakeContract();
  const Slot original = CommittedSlot();
  const StateCoordinates initial = contract.coordinates();
  const BeginOutcome first = contract.BeginStep({InteriorSpace::l2, 0.125});
  auto first_handle = contract.AcquireCandidate(first.attempt);
  const Slot rejected{{{11.0, 12.0, 13.0, 14.0}}, {{15.0, 16.0, 17.0}}};
  bool passed = Check(first.started && first_handle &&
                          first_handle->Mutate(
                              [&rejected](Slot& slot) noexcept {
                                slot = rejected;
                              }),
                      "first_candidate_prepared") ;
  auto products = IssueAll(contract, first.attempt);
  AttemptReport failed_report = PassingReport(99);
  failed_report.gates.finite = false;
  failed_report.telemetry.transfers = {10, 20, 3, 0, 1, true};
  const AttemptOutcome failed = contract.FinishAttempt(
      first.attempt, failed_report, std::move(products));
  Slot published{};
  passed = Check(!failed.committed && failed.retry_available &&
                     CopyPublished(contract, published) &&
                     published.interior == original.interior &&
                     published.trace == original.trace &&
                     contract.coordinates() == initial &&
                     !first_handle->Mutate([](Slot& slot) noexcept {
                       slot.interior.fill(1000.0);
                     }),
                 "failed_attempt_discards_private_candidate") && passed;

  const BeginOutcome retry = contract.Retry();
  Slot retry_copy{};
  auto retry_handle = contract.AcquireCandidate(retry.attempt);
  passed = Check(retry.started && retry.attempt.retry_index == 1 &&
                     retry.attempt.committed_step == first.attempt.committed_step &&
                     retry.attempt.committed_generation ==
                         first.attempt.committed_generation &&
                     retry.attempt.dt == first.attempt.dt && retry_handle &&
                     retry_handle->Copy(retry_copy) &&
                     retry_copy.interior == original.interior &&
                     retry_copy.trace == original.trace,
                 "retry_uses_identical_commit_and_dt") && passed;

  const Slot accepted{{{21.0, 22.0, 23.0, 24.0}}, {{25.0, 26.0, 27.0}}};
  if (retry_handle)
  {
    passed = Check(retry_handle->Mutate(
                       [&accepted](Slot& slot) noexcept { slot = accepted; }),
                   "retry_candidate_mutated") && passed;
  }
  products = IssueAll(contract, retry.attempt);
  const AttemptOutcome committed = contract.FinishAttempt(
      retry.attempt, PassingReport(11), std::move(products));
  const StateCoordinates expected{1.125, 5, 8, 11, 0.125};
  Slot final_state{};
  passed = Check(committed.committed && CopyPublished(contract, final_state) &&
                     final_state.interior == accepted.interior &&
                     final_state.trace == accepted.trace &&
                     contract.coordinates() == expected &&
                     contract.telemetry().admissions_started == 2 &&
                     contract.telemetry().attempts_started == 2 &&
                     contract.telemetry().attempts_failed == 1 &&
                     contract.telemetry().rollbacks == 1 &&
                     contract.telemetry().retries_started == 1 &&
                     contract.telemetry().commits == 1 &&
                     contract.telemetry().accepted.valid &&
                     contract.telemetry().accepted.coordinates == expected &&
                     contract.telemetry().accepted.attempt
                             .validated_product_mask == 0x0fU,
                 "atomic_retry_commit_and_separated_telemetry") && passed;
  const BeginOutcome second_retry = contract.Retry();
  passed = Check(!second_retry.started &&
                     second_retry.reason == FailureReason::retry_exhausted,
                 "at_most_one_retry") && passed;
  return passed;
}

StateHash FrozenInteriorHash()
{
  return {{{41, 43, 45, 47, 45, 47, 49, 51,
            49, 51, 53, 55, 53, 55, 57, 59,
            57, 59, 61, 63, 61, 63, 65, 67,
            65, 67, 69, 71, 69, 71, 73, 75}}};
}

StateHash FrozenTraceHash()
{
  return {{{145, 147, 149, 148, 150, 152, 151, 153,
            155, 154, 156, 158, 157, 159, 161, 160,
            162, 164, 163, 165, 167, 166, 168, 170,
            169, 171, 173, 172, 174, 176, 175, 177}}};
}

bool CheckCheckpointSchemaAndIntegrity()
{
  Contract contract = MakeContract();
  CheckpointDescriptor checkpoint{};
  bool passed = Check(!contract.TryMakeCheckpoint(checkpoint),
                      "checkpoint_requires_successful_commit");
  const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
  passed = Check(!contract.TryMakeCheckpoint(checkpoint),
                 "checkpoint_rejects_active_candidate") && passed;
  const Slot accepted{{{41.0, 42.0, 43.0, 44.0}}, {{45.0, 46.0, 47.0}}};
  passed = Check(MutateCandidate(contract, begin.attempt, accepted),
                 "checkpoint_candidate_mutated") && passed;
  auto products = IssueAll(contract, begin.attempt);
  const AttemptOutcome committed = contract.FinishAttempt(
      begin.attempt, PassingReport(77), std::move(products));
  passed = Check(committed.committed && contract.TryMakeCheckpoint(checkpoint) &&
                     checkpoint.schema_version == checkpoint_schema_version &&
                     checkpoint.successful_commit &&
                     checkpoint.topology_hash == Resources().topology_hash &&
                     checkpoint.cursor == checkpoint.coordinates.cursor &&
                     checkpoint.dt == checkpoint.coordinates.dt &&
                     checkpoint.interior_hash == FrozenInteriorHash() &&
                     checkpoint.trace_hash == FrozenTraceHash(),
                 "versioned_checkpoint_matches_independent_frozen_hashes") &&
      passed;
  passed = Check(contract.CheckRestart(
                         checkpoint, accepted.interior, accepted.trace) ==
                     RestartCompatibility::compatible,
                 "same_backend_rank_partition_restart_integrity") && passed;

  CheckpointDescriptor changed = checkpoint;
  changed.schema_version = checkpoint_schema_version + 1;
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::schema_mismatch,
                 "restart_schema_mismatch") && passed;

  const std::array<int, 5> coordinate_fields{{0, 1, 2, 3, 4}};
  for (const int field : coordinate_fields)
  {
    changed = checkpoint;
    if (field == 0) { changed.coordinates.time += 1.0; }
    if (field == 1) { ++changed.coordinates.step; }
    if (field == 2) { ++changed.coordinates.generation; }
    if (field == 3) { ++changed.coordinates.cursor; }
    if (field == 4) { changed.coordinates.dt += 1.0; }
    passed = Check(contract.CheckRestart(changed, accepted.interior,
                                         accepted.trace) ==
                       RestartCompatibility::coordinate_mismatch,
                   "restart_each_coordinate_field_mismatch") && passed;
  }

  changed = checkpoint;
  ++changed.cursor;
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::cursor_mismatch,
                 "restart_cursor_mirror_mismatch") && passed;
  changed = checkpoint;
  changed.dt += 1.0;
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::dt_mismatch,
                 "restart_dt_mirror_mismatch") && passed;
  changed = checkpoint;
  changed.topology_hash = Digest(31);
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::topology_mismatch,
                 "restart_topology_mismatch") && passed;
  changed = checkpoint;
  changed.tuple.backend = StableIdentity{"cuda.rank1"};
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::backend_mismatch,
                 "restart_backend_mismatch") && passed;
  changed = checkpoint;
  changed.tuple.formulation = StableIdentity{"igr.scalar"};
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::tuple_mismatch,
                 "restart_tuple_mismatch") && passed;
  changed = checkpoint;
  changed.mfem_pin = StableIdentity{"different"};
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::mfem_pin_mismatch,
                 "restart_mfem_mismatch") && passed;
  changed = checkpoint;
  changed.rank_count = 2;
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::rank_count_mismatch,
                 "restart_rank_mismatch") && passed;
  changed = checkpoint;
  changed.partition_hash = Digest(21);
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::partition_mismatch,
                 "restart_partition_mismatch") && passed;
  changed = checkpoint;
  changed.mesh_hash = Digest(11);
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::mesh_mismatch,
                 "restart_mesh_mismatch") && passed;
  changed = checkpoint;
  changed.polynomial_order = 4;
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::order_mismatch,
                 "restart_order_mismatch") && passed;
  changed = checkpoint;
  changed.layout = StableIdentity{"other-layout"};
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::layout_mismatch,
                 "restart_layout_mismatch") && passed;
  changed = checkpoint;
  changed.precision = StableIdentity{"fp32"};
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::precision_mismatch,
                 "restart_precision_mismatch") && passed;
  changed = checkpoint;
  changed.interior_hash.bytes[0] ^= 1U;
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::interior_hash_mismatch,
                 "restart_descriptor_interior_hash_mismatch") && passed;
  changed = checkpoint;
  changed.trace_hash.bytes[0] ^= 1U;
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::trace_hash_mismatch,
                 "restart_descriptor_trace_hash_mismatch") && passed;
  Interior altered_interior = accepted.interior;
  altered_interior[0] += 1.0;
  passed = Check(contract.CheckRestart(checkpoint, altered_interior,
                                       accepted.trace) ==
                     RestartCompatibility::interior_hash_mismatch,
                 "restart_loaded_interior_integrity") && passed;
  Trace altered_trace = accepted.trace;
  altered_trace[0] += 1.0;
  passed = Check(contract.CheckRestart(checkpoint, accepted.interior,
                                       altered_trace) ==
                     RestartCompatibility::trace_hash_mismatch,
                 "restart_loaded_trace_integrity") && passed;
  changed = checkpoint;
  changed.successful_commit = false;
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::no_successful_commit,
                 "restart_requires_successful_commit") && passed;
  changed = checkpoint;
  changed.restart_mode = static_cast<RestartMode>(255);
  passed = Check(contract.CheckRestart(changed, accepted.interior,
                                       accepted.trace) ==
                     RestartCompatibility::policy_mismatch,
                 "restart_policy_explicit") && passed;
  return passed;
}

bool CheckAllLocalGatesAndStatePreservation()
{
  const std::array<FailureReason, 6> expected{{
      FailureReason::finite_gate,
      FailureReason::admissibility_gate,
      FailureReason::recovery_gate,
      FailureReason::true_residual_gate,
      FailureReason::telemetry_gate,
      FailureReason::success_gate}};
  bool passed = true;
  for (std::size_t gate = 0; gate < expected.size(); ++gate)
  {
    Contract contract = MakeContract();
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
    auto products = IssueAll(contract, begin.attempt);
    AttemptReport report = PassingReport(12);
    bool* gates[] = {&report.gates.finite,
                     &report.gates.admissible,
                     &report.gates.recovery,
                     &report.gates.true_residual,
                     &report.gates.telemetry,
                     &report.gates.success};
    *gates[gate] = false;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, std::move(products));
    passed = Check(!outcome.committed && outcome.reason == expected[gate] &&
                       contract.coordinates() == InitialCoordinates(),
                   "every_local_commit_gate_required") && passed;
  }

  Contract contract = MakeContract();
  BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.125});
  const Slot accepted{{{31.0, 32.0, 33.0, 34.0}}, {{35.0, 36.0, 37.0}}};
  passed = Check(MutateCandidate(contract, begin.attempt, accepted),
                 "accepted_candidate_mutated") && passed;
  auto products = IssueAll(contract, begin.attempt);
  const AttemptOutcome first = contract.FinishAttempt(
      begin.attempt, PassingReport(14), std::move(products));
  Slot committed_copy{};
  contract.Published().Copy(committed_copy);
  const auto interior_bytes = ObjectBytes(committed_copy.interior);
  const auto trace_bytes = ObjectBytes(committed_copy.trace);
  const StateCoordinates committed_coordinates = contract.coordinates();

  begin = contract.BeginStep({InteriorSpace::l2, 0.125});
  const Slot rejected{{{1000.0, 1000.0, 1000.0, 1000.0}},
                      {{2000.0, 2000.0, 2000.0}}};
  passed = Check(MutateCandidate(contract, begin.attempt, rejected),
                 "rejected_candidate_mutated") && passed;
  products = IssueAll(contract, begin.attempt);
  AttemptReport report = PassingReport(99);
  report.gates.recovery = false;
  const AttemptOutcome failure = contract.FinishAttempt(
      begin.attempt, report, std::move(products));
  Slot after_failure{};
  contract.Published().Copy(after_failure);
  passed = Check(first.committed && !failure.committed &&
                     ObjectBytes(after_failure.interior) == interior_bytes &&
                     ObjectBytes(after_failure.trace) == trace_bytes &&
                     contract.coordinates() == committed_coordinates &&
                     contract.telemetry().accepted.coordinates ==
                         committed_coordinates,
                 "failed_attempt_preserves_committed_bytes_and_telemetry") &&
      passed;
  return passed;
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckStaticSelectionIdentityAndInvalidRequests() && passed;
  passed = CheckCollectiveAdmissionAndPreparation() && passed;
  passed = CheckDisjointStorageAndBoundHandles() && passed;
  passed = CheckIssuedProductProtocol() && passed;
  passed = CheckResidualOriginAndTransferGates() && passed;
  passed = CheckCollectiveIdentityAndRemoteGates() && passed;
  passed = CheckAtomicRetryCommitAndTelemetry() && passed;
  passed = CheckCheckpointSchemaAndIntegrity() && passed;
  passed = CheckAllLocalGatesAndStatePreservation() && passed;
  if (passed)
  {
    std::cout << "PASS hdg_runtime_contract_cxx17"
              << " admission=collective products=issued residual=1e-10"
              << " checkpoint=schema1 restart=integrity\n";
  }
  return passed ? 0 : 1;
}

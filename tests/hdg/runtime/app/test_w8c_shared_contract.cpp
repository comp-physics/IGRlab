#include "hdg_cpu_serial_lifecycle.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using namespace dg::hdg::runtime;
namespace w8c = dg::hdg::runtime::cpu::w8c;

struct NoRegularization final {};
struct TracedScalarIgrPolicy final {};

enum class Event : unsigned char
{
  prepare,
  stage,
  install,
  publish,
  acknowledge,
  finalize_numerical,
  release_durable,
  revert_durable,
  abort_prepared,
  revert_installed
};

struct Trace final
{
  std::vector<Event> events;
  bool durable_published = false;
};

template <typename Policy>
class Method final
{
public:
  struct Transaction final
  {
    StateCoordinates entry{};
    StateCoordinates target{};
    bool prepared = false;
    bool installed = false;
  };

  explicit Method(Trace& trace) noexcept : trace_(trace) {}

  bool Initialize() noexcept
  {
    status_ = RunStatus::ready;
    reason_ = FailureReason::none;
    return true;
  }
  bool Prepare(Transaction& transaction, const bool retry,
               const std::uint64_t target) noexcept
  {
    trace_.events.push_back(Event::prepare);
    retry_flags.push_back(retry);
    transaction.entry = coordinates_;
    transaction.target = {
        coordinates_.time + w8c::dt, coordinates_.step + 1,
        coordinates_.generation + 1, coordinates_.cursor + 1, w8c::dt};
    if (transaction.target.step != target) return false;
    transaction.prepared = true;
    prepared_targets.push_back(transaction.target.step);
    return true;
  }
  w8c::LifecycleAttemptOutcome PrepareFailure(
      const Transaction&) const noexcept
  {
    return {true, RetryAvailable(), false};
  }
  bool Install(Transaction& transaction) noexcept
  {
    trace_.events.push_back(Event::install);
    if (!transaction.prepared) return false;
    transaction.installed = true;
    coordinates_ = transaction.target;
    status_ = RunStatus::committed;
    return true;
  }
  w8c::LifecycleAttemptOutcome AbortPrepared(
      Transaction&& transaction, w8c::LifecycleFailure,
      const bool durable_reverted) noexcept
  {
    trace_.events.push_back(Event::abort_prepared);
    if (!transaction.prepared || transaction.installed)
      return {false, false, false};
    status_ = RunStatus::retry_available;
    reason_ = FailureReason::backend_failure;
    ++aborts;
    return {durable_reverted, true, false};
  }
  w8c::LifecycleAttemptOutcome RevertInstalled(
      Transaction&& transaction, const w8c::LifecycleFailure failure,
      const bool durable_reverted) noexcept
  {
    trace_.events.push_back(Event::revert_installed);
    if (!transaction.installed) return {false, false, false};
    coordinates_ = transaction.entry;
    status_ = RunStatus::retry_available;
    reason_ = FailureReason::backend_failure;
    ++reverts;
    const bool retryable =
        failure == w8c::LifecycleFailure::injected_after_recovery;
    return {durable_reverted, retryable, false};
  }
  void Finalize(Transaction&& transaction) noexcept
  {
    trace_.events.push_back(Event::finalize_numerical);
    if (!transaction.installed || !trace_.durable_published) std::abort();
    status_ = RunStatus::committed;
    reason_ = FailureReason::none;
    ++finalizes;
  }
  void RestoreStep32() noexcept
  {
    coordinates_ = {
        32.0 * w8c::dt, 32, 32, 32, w8c::dt};
    status_ = RunStatus::committed;
  }
  RunStatus Status() const noexcept { return status_; }
  FailureReason Reason() const noexcept { return reason_; }
  bool RetryAvailable() const noexcept
  {
    return status_ == RunStatus::retry_available;
  }
  StateCoordinates Coordinates() const noexcept { return coordinates_; }
  constexpr bool ForceTerminalFailure() const noexcept { return true; }
  void OnRetry(w8c::LifecycleFailure, std::uint32_t) const noexcept {}

  std::vector<std::uint64_t> prepared_targets;
  std::vector<bool> retry_flags;
  int aborts = 0;
  int reverts = 0;
  int finalizes = 0;

private:
  Trace& trace_;
  StateCoordinates coordinates_{};
  RunStatus status_ = RunStatus::invalid_request;
  FailureReason reason_ = FailureReason::none;
};

template <typename Policy>
class Publication final
{
public:
  explicit Publication(Trace& trace) noexcept : trace_(trace) {}

  bool Restore(Method<Policy>& method) noexcept
  {
    method.RestoreStep32();
    ++restores;
    return true;
  }
  bool PublishInitial(Method<Policy>&) noexcept
  {
    ++initial_publications;
    return true;
  }
  bool Stage(Method<Policy>&,
             const typename Method<Policy>::Transaction&,
             const std::uint64_t target, const bool output,
             const bool checkpoint, const bool terminal,
             const std::uint32_t) noexcept
  {
    trace_.events.push_back(Event::stage);
    staged_targets.push_back(target);
    if (output) output_targets.push_back(target);
    if (checkpoint) checkpoint_targets.push_back(target);
    if (terminal) terminal_targets.push_back(target);
    attempt_open = true;
    active = output || checkpoint || terminal;
    active_target = target;
    return active_target != staging_failure_target;
  }
  bool ReadyToPublish(
      Method<Policy>&, const typename Method<Policy>::Transaction&,
      const std::uint32_t) noexcept
  {
    return true;
  }
  bool Publish(
      Method<Policy>&,
      const typename Method<Policy>::Transaction&) noexcept
  {
    trace_.events.push_back(Event::publish);
    if (!attempt_open) return false;
    if (!publication_failure_consumed &&
        active_target == publication_failure_target)
    {
      publication_failure_consumed = true;
      return false;
    }
    published = true;
    trace_.durable_published = true;
    ++publications;
    return true;
  }
  bool Acknowledge(
      Method<Policy>&, const typename Method<Policy>::Transaction&,
      const std::uint32_t) noexcept
  {
    trace_.events.push_back(Event::acknowledge);
    return active_target != acknowledgement_failure_target;
  }
  bool Revert() noexcept
  {
    trace_.events.push_back(Event::revert_durable);
    if (!attempt_open) return false;
    attempt_open = false;
    active = false;
    published = false;
    trace_.durable_published = false;
    ++reverts;
    return true;
  }
  void Finalize() noexcept
  {
    trace_.events.push_back(Event::release_durable);
    if (!attempt_open || !published) std::abort();
    attempt_open = false;
    active = false;
    published = false;
    trace_.durable_published = false;
    ++releases;
  }

  std::vector<std::uint64_t> staged_targets;
  std::vector<std::uint64_t> output_targets;
  std::vector<std::uint64_t> checkpoint_targets;
  std::vector<std::uint64_t> terminal_targets;
  int restores = 0;
  int initial_publications = 0;
  int publications = 0;
  int reverts = 0;
  int releases = 0;
  std::uint64_t active_target = 0;
  std::uint64_t staging_failure_target = 0;
  std::uint64_t publication_failure_target = 0;
  std::uint64_t acknowledgement_failure_target = 0;
  bool active = false;
  bool attempt_open = false;
  bool published = false;
  bool publication_failure_consumed = false;

private:
  Trace& trace_;
};

struct Checks final
{
  void Require(const bool condition, const char* message)
  {
    if (condition) return;
    ++failures;
    std::cerr << message << '\n';
  }
  int failures = 0;
};

template <typename Policy, typename Case = w8c::ExactJumpV1Case>
void CheckUninterruptedAndRetry(Checks& checks)
{
  Trace trace;
  Method<Policy> method(trace);
  Publication<Policy> publication(trace);
  const w8c::LifecycleControl control{false, false, true, 1};
  const w8c::LifecycleReport report =
      w8c::RunCpuSerialLifecycle(method, publication, control, Case{});
  checks.Require(report.status == RunStatus::committed &&
                     report.reason == FailureReason::none &&
                     report.coordinates.step == 64 &&
                     report.coordinates.time == Case::final_time &&
                     report.retries == 1 && report.injected,
                 "focused W8C retry must reach the exact terminal cursor");
  checks.Require(method.reverts == 1 && method.finalizes == 64 &&
                     publication.reverts == 1 &&
                     publication.publications == 64 &&
                     publication.releases == 64 &&
                     publication.initial_publications == 1,
                 "engine must own one rollback and 64 publications");
  const std::vector<Event> first_transaction{
      Event::prepare, Event::stage, Event::install,
      Event::publish, Event::acknowledge,
      Event::finalize_numerical, Event::release_durable};
  checks.Require(trace.events.size() >= first_transaction.size() &&
                     std::equal(first_transaction.begin(),
                                first_transaction.end(),
                                trace.events.begin()),
                 "serial transaction order must retain durable authority through finalize");
  checks.Require(std::count(method.prepared_targets.begin(),
                            method.prepared_targets.end(), 7) == 2 &&
                     method.retry_flags.size() == 65 &&
                     method.retry_flags[7],
                 "step seven must be retried once at the same dt");
  checks.Require(publication.output_targets ==
                         std::vector<std::uint64_t>({32, 64}) &&
                     publication.checkpoint_targets ==
                         std::vector<std::uint64_t>({32}) &&
                     publication.terminal_targets ==
                         std::vector<std::uint64_t>({64}),
                 "engine output/checkpoint schedule must be 0,32,64/32");
}

template <typename Policy, typename Case = w8c::ExactJumpV1Case>
void CheckSplitRestart(Checks& checks)
{
  Trace writer_trace;
  Method<Policy> writer_method(writer_trace);
  Publication<Policy> writer_publication(writer_trace);
  const w8c::LifecycleReport writer = w8c::RunCpuSerialLifecycle(
      writer_method, writer_publication, {false, true, false, 0}, Case{});
  checks.Require(writer.status == RunStatus::committed &&
                     writer.coordinates.step == 32 &&
                     writer_publication.checkpoint_targets ==
                         std::vector<std::uint64_t>({32}),
                 "split writer must stop only after committed step 32");

  Trace reader_trace;
  Method<Policy> reader_method(reader_trace);
  Publication<Policy> reader_publication(reader_trace);
  const w8c::LifecycleReport reader = w8c::RunCpuSerialLifecycle(
      reader_method, reader_publication, {true, false, false, 0}, Case{});
  checks.Require(reader.status == RunStatus::committed &&
                     reader.coordinates.step == 64 &&
                     reader.coordinates.time == Case::final_time &&
                     reader_publication.restores == 1 &&
                     reader_publication.initial_publications == 0 &&
                     reader_publication.output_targets ==
                         std::vector<std::uint64_t>({64}),
                 "split reader must restore step 32 and continue to 64");
}

template <typename Policy, typename Case = w8c::ExactJumpV1Case>
void CheckPublicationFailureRollback(Checks& checks)
{
  Trace trace;
  Method<Policy> method(trace);
  Publication<Policy> publication(trace);
  publication.publication_failure_target = 4;
  const w8c::LifecycleReport report = w8c::RunCpuSerialLifecycle(
      method, publication, {false, false, false, 0}, Case{});
  checks.Require(report.status == RunStatus::terminal_failure &&
                     report.failure == w8c::LifecycleFailure::publication &&
                     report.coordinates.step == 3 &&
                     report.coordinates.time == 3.0 * w8c::dt &&
                     report.coordinates.generation == 3 &&
                     report.coordinates.cursor == 3 &&
                     report.coordinates.dt == w8c::dt,
                 "publication failure must preserve the prior committed cursor");
  checks.Require(method.reverts == 1 && publication.reverts == 1 &&
                     method.finalizes == 3 && publication.publications == 3,
                 "publication failure must revert durable and numerical transactions");
}

template <typename Policy, typename Case = w8c::ExactJumpV1Case>
void CheckPreparedAndInstalledRollbackAreDistinct(Checks& checks)
{
  Trace staged_trace;
  Method<Policy> staged_method(staged_trace);
  Publication<Policy> staged_publication(staged_trace);
  staged_publication.staging_failure_target = 4;
  const w8c::LifecycleReport staged = w8c::RunCpuSerialLifecycle(
      staged_method, staged_publication, {false, false, false, 0}, Case{});
  checks.Require(staged.failure == w8c::LifecycleFailure::staging &&
                     staged.coordinates.step == 3 &&
                     staged.coordinates.time == 3.0 * w8c::dt &&
                     staged.coordinates.generation == 3 &&
                     staged.coordinates.cursor == 3 &&
                     staged.coordinates.dt == w8c::dt &&
                     staged_method.aborts == 1 &&
                     staged_method.reverts == 0,
                 "staging failure must AbortPrepared before provisional install");
  const auto staged_durable = std::find(
      staged_trace.events.begin(), staged_trace.events.end(),
      Event::revert_durable);
  const auto staged_numerical = std::find(
      staged_trace.events.begin(), staged_trace.events.end(),
      Event::abort_prepared);
  checks.Require(staged_durable < staged_numerical,
                 "durable staging rollback must precede AbortPrepared");

  Trace acknowledged_trace;
  Method<Policy> acknowledged_method(acknowledged_trace);
  Publication<Policy> acknowledged_publication(acknowledged_trace);
  acknowledged_publication.acknowledgement_failure_target = 4;
  const w8c::LifecycleReport acknowledged = w8c::RunCpuSerialLifecycle(
      acknowledged_method, acknowledged_publication,
      {false, false, false, 0}, Case{});
  checks.Require(
      acknowledged.failure ==
              w8c::LifecycleFailure::post_publication_acknowledgement &&
          acknowledged.coordinates.step == 3 &&
          acknowledged.coordinates.time == 3.0 * w8c::dt &&
          acknowledged.coordinates.generation == 3 &&
          acknowledged.coordinates.cursor == 3 &&
          acknowledged.coordinates.dt == w8c::dt &&
          acknowledged_method.aborts == 0 &&
          acknowledged_method.reverts == 1 &&
          acknowledged_publication.reverts == 1,
      "acknowledgement failure must restore durable and installed state");
  const auto acknowledged_durable = std::find(
      acknowledged_trace.events.begin(), acknowledged_trace.events.end(),
      Event::revert_durable);
  const auto acknowledged_numerical = std::find(
      acknowledged_trace.events.begin(), acknowledged_trace.events.end(),
      Event::revert_installed);
  checks.Require(acknowledged_durable < acknowledged_numerical,
                 "durable marker restoration must precede RevertInstalled");
}

} // namespace

int main()
{
  using StaticMethod = Method<NoRegularization>;
  using StaticPublication = Publication<NoRegularization>;
  using StaticTransaction = StaticMethod::Transaction;
  static_assert(std::is_same<
                    decltype(std::declval<StaticMethod&>().Finalize(
                        std::declval<StaticTransaction&&>())),
                    void>::value,
                "numerical Finalize must return void");
  static_assert(noexcept(std::declval<StaticMethod&>().Finalize(
                    std::declval<StaticTransaction&&>())),
                "numerical Finalize must be noexcept");
  static_assert(std::is_same<
                    decltype(std::declval<StaticPublication&>().Finalize()),
                    void>::value,
                "durable Finalize must return void");
  static_assert(noexcept(std::declval<StaticPublication&>().Finalize()),
                "durable Finalize must be noexcept");
  static_assert(!std::is_polymorphic<Method<NoRegularization>>::value,
                "static none");
  static_assert(!std::is_polymorphic<Method<TracedScalarIgrPolicy>>::value,
                "static scalar IGR");
  Checks checks;
  CheckUninterruptedAndRetry<NoRegularization>(checks);
  CheckUninterruptedAndRetry<TracedScalarIgrPolicy>(checks);
  CheckSplitRestart<NoRegularization>(checks);
  CheckSplitRestart<TracedScalarIgrPolicy>(checks);
  CheckPublicationFailureRollback<NoRegularization>(checks);
  CheckPublicationFailureRollback<TracedScalarIgrPolicy>(checks);
  CheckPreparedAndInstalledRollbackAreDistinct<NoRegularization>(checks);
  CheckPreparedAndInstalledRollbackAreDistinct<TracedScalarIgrPolicy>(checks);
  CheckUninterruptedAndRetry<
      NoRegularization, w8c::SmoothTransitionV2Case>(checks);
  CheckUninterruptedAndRetry<
      TracedScalarIgrPolicy, w8c::SmoothTransitionV2Case>(checks);
  CheckSplitRestart<NoRegularization, w8c::SmoothTransitionV2Case>(checks);
  CheckSplitRestart<
      TracedScalarIgrPolicy, w8c::SmoothTransitionV2Case>(checks);
  CheckPublicationFailureRollback<
      NoRegularization, w8c::SmoothTransitionV2Case>(checks);
  CheckPublicationFailureRollback<
      TracedScalarIgrPolicy, w8c::SmoothTransitionV2Case>(checks);
  return checks.failures == 0 ? 0 : 1;
}

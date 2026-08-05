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

struct NonePolicy final {};
struct TracedScalarIgrPolicy final {};

template <typename Policy>
class Method final
{
public:
  struct Transaction final
  {
    StateCoordinates entry{};
    StateCoordinates target{};
    bool installed = false;
  };

  bool Initialize() noexcept
  {
    status_ = RunStatus::ready;
    reason_ = FailureReason::none;
    return true;
  }
  bool Prepare(Transaction& transaction, const bool retry) noexcept
  {
    retry_flags.push_back(retry);
    transaction.entry = coordinates_;
    transaction.target = {
        coordinates_.time + w8c::dt, coordinates_.step + 1,
        coordinates_.generation + 1, coordinates_.cursor + 1, w8c::dt};
    prepared_targets.push_back(transaction.target.step);
    return true;
  }
  bool Install(Transaction& transaction) noexcept
  {
    transaction.installed = true;
    coordinates_ = transaction.target;
    status_ = RunStatus::committed;
    return true;
  }
  bool Revert(Transaction&& transaction) noexcept
  {
    if (!transaction.installed) return false;
    coordinates_ = transaction.entry;
    status_ = RunStatus::retry_available;
    reason_ = FailureReason::backend_failure;
    ++reverts;
    return true;
  }
  void Finalize(Transaction&& transaction) noexcept
  {
    if (!transaction.installed) std::abort();
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

  std::vector<std::uint64_t> prepared_targets;
  std::vector<bool> retry_flags;
  int reverts = 0;
  int finalizes = 0;

private:
  StateCoordinates coordinates_{};
  RunStatus status_ = RunStatus::invalid_request;
  FailureReason reason_ = FailureReason::none;
};

template <typename Policy>
class Publication final
{
public:
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
             const bool checkpoint, const bool terminal) noexcept
  {
    staged_targets.push_back(target);
    if (output) output_targets.push_back(target);
    if (checkpoint) checkpoint_targets.push_back(target);
    if (terminal) terminal_targets.push_back(target);
    active = true;
    active_target = target;
    return true;
  }
  bool Publish() noexcept
  {
    if (!active) return false;
    if (!publication_failure_consumed &&
        active_target == publication_failure_target)
    {
      publication_failure_consumed = true;
      return false;
    }
    active = false;
    ++publications;
    return true;
  }
  bool Revert() noexcept
  {
    if (!active) return false;
    active = false;
    ++reverts;
    return true;
  }

  std::vector<std::uint64_t> staged_targets;
  std::vector<std::uint64_t> output_targets;
  std::vector<std::uint64_t> checkpoint_targets;
  std::vector<std::uint64_t> terminal_targets;
  int restores = 0;
  int initial_publications = 0;
  int publications = 0;
  int reverts = 0;
  std::uint64_t active_target = 0;
  std::uint64_t publication_failure_target = 0;
  bool active = false;
  bool publication_failure_consumed = false;
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

template <typename Policy>
void CheckUninterruptedAndRetry(Checks& checks)
{
  Method<Policy> method;
  Publication<Policy> publication;
  const w8c::LifecycleControl control{false, false, true, 1};
  const w8c::LifecycleReport report =
      w8c::RunCpuSerialLifecycle(method, publication, control);
  checks.Require(report.status == RunStatus::committed &&
                     report.reason == FailureReason::none &&
                     report.coordinates.step == 64 &&
                     report.coordinates.time == w8c::final_time &&
                     report.retries == 1 && report.injected,
                 "focused W8C retry must reach the exact terminal cursor");
  checks.Require(method.reverts == 1 && method.finalizes == 64 &&
                     publication.reverts == 1 &&
                     publication.publications == 64 &&
                     publication.initial_publications == 1,
                 "engine must own one rollback and 64 publications");
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

template <typename Policy>
void CheckSplitRestart(Checks& checks)
{
  Method<Policy> writer_method;
  Publication<Policy> writer_publication;
  const w8c::LifecycleReport writer = w8c::RunCpuSerialLifecycle(
      writer_method, writer_publication, {false, true, false, 0});
  checks.Require(writer.status == RunStatus::committed &&
                     writer.coordinates.step == 32 &&
                     writer_publication.checkpoint_targets ==
                         std::vector<std::uint64_t>({32}),
                 "split writer must stop only after committed step 32");

  Method<Policy> reader_method;
  Publication<Policy> reader_publication;
  const w8c::LifecycleReport reader = w8c::RunCpuSerialLifecycle(
      reader_method, reader_publication, {true, false, false, 0});
  checks.Require(reader.status == RunStatus::committed &&
                     reader.coordinates.step == 64 &&
                     reader.coordinates.time == w8c::final_time &&
                     reader_publication.restores == 1 &&
                     reader_publication.initial_publications == 0 &&
                     reader_publication.output_targets ==
                         std::vector<std::uint64_t>({64}),
                 "split reader must restore step 32 and continue to 64");
}

template <typename Policy>
void CheckPublicationFailureRollback(Checks& checks)
{
  Method<Policy> method;
  Publication<Policy> publication;
  publication.publication_failure_target = 4;
  const w8c::LifecycleReport report = w8c::RunCpuSerialLifecycle(
      method, publication, {false, false, false, 0});
  checks.Require(report.status == RunStatus::terminal_failure &&
                     report.failure == w8c::LifecycleFailure::publication &&
                     report.coordinates.step == 3,
                 "publication failure must preserve the prior committed cursor");
  checks.Require(method.reverts == 1 && publication.reverts == 1 &&
                     method.finalizes == 3 && publication.publications == 3,
                 "publication failure must revert durable and numerical transactions");
}

} // namespace

int main()
{
  static_assert(!std::is_polymorphic<Method<NonePolicy>>::value, "static none");
  static_assert(!std::is_polymorphic<Method<TracedScalarIgrPolicy>>::value,
                "static scalar IGR");
  Checks checks;
  CheckUninterruptedAndRetry<NonePolicy>(checks);
  CheckUninterruptedAndRetry<TracedScalarIgrPolicy>(checks);
  CheckSplitRestart<NonePolicy>(checks);
  CheckSplitRestart<TracedScalarIgrPolicy>(checks);
  CheckPublicationFailureRollback<NonePolicy>(checks);
  CheckPublicationFailureRollback<TracedScalarIgrPolicy>(checks);
  return checks.failures == 0 ? 0 : 1;
}

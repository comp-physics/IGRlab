#pragma once

#include "numerics/hdg/runtime/runtime_abi.hpp"

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

namespace dg::hdg::runtime::cpu::w8c
{

enum class CaseKind : unsigned char
{
  exact_jump_v1,
  smooth_transition_v2
};

struct ExactJumpV1Case final
{
  static constexpr CaseKind kind = CaseKind::exact_jump_v1;
  static constexpr std::string_view case_id = "periodic-double-riemann-v1";
  static constexpr std::string_view scenario_id = "W8C-N8-DT10";
  static constexpr std::string_view conventional_source_id =
      "source.euler.periodic-double-riemann.v1";
  static constexpr std::string_view scalar_source_id =
      "source.sc25.periodic-double-riemann.v3";
  static constexpr std::string_view conventional_checkpoint_id =
      "checkpoint.hdg.euler.periodic-double-riemann.n8-dt10.step-32.v1";
  static constexpr std::string_view scalar_checkpoint_id =
      "checkpoint.hdg.igr.sc25.periodic-double-riemann.n8-dt10.step-32.v1";
  static constexpr std::string_view experiment_id =
      "case.hdg.euler.periodic_double_riemann.w8c.v1";
  static constexpr std::uint64_t seed = UINT64_C(0x5738434452495631);
  static constexpr int polynomial_order = 1;
  static constexpr std::uint32_t mesh_n = 8;
  static constexpr int conventional_q_a = 11;
  static constexpr int scalar_q_a = 12;
  static constexpr int scalar_q_s = 14;
  static constexpr int observable_q_o = 18;
  static constexpr double dt = 1.0 / 1024.0;
  static constexpr std::uint64_t accepted_steps = 64;
  static constexpr double final_time = 1.0 / 16.0;
  static constexpr std::array<std::uint64_t, 3> output_steps{{0, 32, 64}};
  static constexpr std::uint64_t checkpoint_step = 32;
  static constexpr std::uint64_t failure_target_step = 7;
  static constexpr std::uint32_t retry_limit = 1;
};

struct SmoothTransitionV2Case final
{
  static constexpr CaseKind kind = CaseKind::smooth_transition_v2;
  static constexpr std::string_view case_id =
      "periodic-smooth-transition-v2";
  static constexpr std::string_view scenario_id = "W8C-V2-N8-DT10";
  static constexpr std::string_view conventional_source_id =
      "source.euler.periodic-smooth-transition.v2";
  static constexpr std::string_view scalar_source_id =
      "source.sc25.periodic-smooth-transition.v3";
  static constexpr std::string_view conventional_checkpoint_id =
      "checkpoint.hdg.euler.periodic-smooth-transition.n8-dt10.step-32.v2";
  static constexpr std::string_view scalar_checkpoint_id =
      "checkpoint.hdg.igr.sc25.periodic-smooth-transition.n8-dt10.step-32.v2";
  static constexpr std::string_view experiment_id =
      "case.hdg.euler.periodic_smooth_transition.w8c.v2";
  static constexpr std::uint64_t seed = UINT64_C(0x573843534d4f5632);
  static constexpr int polynomial_order = 1;
  static constexpr std::uint32_t mesh_n = 8;
  static constexpr int conventional_q_a = 11;
  static constexpr int scalar_q_a = 12;
  static constexpr int scalar_q_s = 14;
  static constexpr int projection_q = 14;
  static constexpr int observable_q_o = 18;
  static constexpr double dt = 1.0 / 1024.0;
  static constexpr std::uint64_t accepted_steps = 64;
  static constexpr double final_time = 1.0 / 16.0;
  static constexpr std::array<std::uint64_t, 3> output_steps{{0, 32, 64}};
  static constexpr std::uint64_t checkpoint_step = 32;
  static constexpr std::uint64_t failure_target_step = 7;
  static constexpr std::uint32_t retry_limit = 1;
  static constexpr double transition_width = 1.0 / 8.0;
};

// Exact-jump aliases retain the accepted W8C-v1 internal identities and keep
// its existing callers and regression bytes unchanged.
inline constexpr std::string_view case_id = ExactJumpV1Case::case_id;
inline constexpr std::string_view scenario_id = ExactJumpV1Case::scenario_id;
inline constexpr std::string_view conventional_source_id =
    ExactJumpV1Case::conventional_source_id;
inline constexpr std::string_view scalar_source_id =
    ExactJumpV1Case::scalar_source_id;
inline constexpr std::string_view conventional_checkpoint_id =
    ExactJumpV1Case::conventional_checkpoint_id;
inline constexpr std::string_view scalar_checkpoint_id =
    ExactJumpV1Case::scalar_checkpoint_id;
inline constexpr std::string_view experiment_id =
    ExactJumpV1Case::experiment_id;
inline constexpr std::uint64_t seed = ExactJumpV1Case::seed;
inline constexpr int polynomial_order = ExactJumpV1Case::polynomial_order;
inline constexpr std::uint32_t mesh_n = ExactJumpV1Case::mesh_n;
inline constexpr int conventional_q_a = ExactJumpV1Case::conventional_q_a;
inline constexpr int scalar_q_a = ExactJumpV1Case::scalar_q_a;
inline constexpr int scalar_q_s = ExactJumpV1Case::scalar_q_s;
inline constexpr int observable_q_o = ExactJumpV1Case::observable_q_o;
inline constexpr double dt = ExactJumpV1Case::dt;
inline constexpr std::uint64_t accepted_steps =
    ExactJumpV1Case::accepted_steps;
inline constexpr double final_time = ExactJumpV1Case::final_time;
inline constexpr auto output_steps = ExactJumpV1Case::output_steps;
inline constexpr std::uint64_t checkpoint_step =
    ExactJumpV1Case::checkpoint_step;
inline constexpr std::uint64_t failure_target_step =
    ExactJumpV1Case::failure_target_step;
inline constexpr std::uint32_t retry_limit = ExactJumpV1Case::retry_limit;
inline constexpr std::array<double, 4> right_state{{
    0.125, 0.0, 0.0, 0.25}};
inline constexpr std::array<double, 4> left_state{{
    1.0, 0.0, 0.0, 2.5}};
inline constexpr std::array<double, 4> interface_mean{{
    0.5625, 0.0, 0.0, 1.375}};

constexpr bool IsOutputStep(const std::uint64_t step) noexcept
{
  return step == output_steps[0] || step == output_steps[1] ||
      step == output_steps[2];
}

constexpr CaseKind SelectCaseKind(const std::string_view selected) noexcept
{
  return selected == SmoothTransitionV2Case::case_id
      ? CaseKind::smooth_transition_v2 : CaseKind::exact_jump_v1;
}

constexpr bool IsCaseId(const std::string_view selected) noexcept
{
  return selected == ExactJumpV1Case::case_id ||
      selected == SmoothTransitionV2Case::case_id;
}

enum class LifecycleFailure : unsigned char
{
  none,
  initialization,
  restart,
  initial_output,
  numerical_prepare,
  staging,
  injected_after_recovery,
  provisional_install,
  pre_publication,
  publication,
  post_publication_acknowledgement,
  rollback
};

struct LifecycleControl final
{
  bool restart = false;
  bool split_writer = false;
  bool inject_after_recovery = false;
  std::uint32_t maximum_retries = 0;
  std::uint32_t initial_retries = 0;
};

struct LifecycleAttemptOutcome final
{
  bool rollback_succeeded = true;
  bool retryable = false;
  bool retry_without_budget = false;
};

struct LifecycleReport final
{
  RunStatus status = RunStatus::invalid_request;
  FailureReason reason = FailureReason::invalid_resource_descriptor;
  StateCoordinates coordinates{};
  std::uint32_t retries = 0;
  LifecycleFailure failure = LifecycleFailure::none;
  bool injected = false;
};

/**
 * One closed CPU-serial HDG application lifecycle.
 *
 * Method is the statically selected mathematical component, Publication is
 * the application-owned durable output/checkpoint adapter, and Schedule is a
 * statically composed case/configuration view.  The engine alone controls
 * step targets, retry, transaction order, publication, rollback, and terminal
 * propagation.  Policy hooks may prepare numerical state or its durable
 * representation; they cannot advance the accepted-step cursor or choose a
 * retry.
 */
template <typename Method, typename Publication, typename Schedule>
LifecycleReport RunCpuSerialLifecycleWithSchedule(
    Method& method, Publication& publication,
    const LifecycleControl control, const Schedule& schedule) noexcept
{
  static_assert(!std::is_polymorphic<Method>::value,
                "CPU-serial method composition must use static dispatch");
  static_assert(!std::is_polymorphic<Publication>::value,
                "CPU-serial publication composition must use static dispatch");
  static_assert(!std::is_polymorphic<Schedule>::value,
                "CPU-serial schedule composition must use static dispatch");
  using Transaction = typename Method::Transaction;
  static_assert(
      std::is_same<decltype(std::declval<Method&>().Finalize(
                       std::declval<Transaction&&>())), void>::value,
      "CPU-serial finalization is a non-fallible invariant");
  static_assert(
      noexcept(std::declval<Method&>().Finalize(
          std::declval<Transaction&&>())),
      "CPU-serial finalization must be noexcept after acknowledgement");
  static_assert(
      std::is_same<decltype(std::declval<Publication&>().Finalize()),
                   void>::value,
      "durable authority release must be a non-fallible invariant");
  static_assert(noexcept(std::declval<Publication&>().Finalize()),
                "durable authority release must be noexcept");

  LifecycleReport report{};
  report.retries = control.initial_retries;
  const auto fail = [&](const LifecycleFailure failure,
                        const FailureReason fallback) noexcept {
    report.status = method.Status();
    if (method.ForceTerminalFailure() &&
        report.status != RunStatus::terminal_failure)
      report.status = RunStatus::terminal_failure;
    report.reason = method.Reason() == FailureReason::none
        ? fallback : method.Reason();
    report.coordinates = method.Coordinates();
    report.failure = failure;
    return report;
  };

  if (!method.Initialize())
    return fail(LifecycleFailure::initialization,
                FailureReason::backend_failure);
  report.status = RunStatus::ready;
  report.reason = FailureReason::none;
  report.coordinates = method.Coordinates();

  if (control.restart && !publication.Restore(method))
    return fail(LifecycleFailure::restart,
                FailureReason::restart_incompatible);
  report.coordinates = method.Coordinates();
  if (report.coordinates.step == 0 &&
      !publication.PublishInitial(method))
    return fail(LifecycleFailure::initial_output,
                FailureReason::backend_failure);

  const std::uint64_t stop_step = control.split_writer
      ? schedule.CheckpointStep() : schedule.AcceptedSteps();
  bool retry = false;
  while (method.Coordinates().step < stop_step)
  {
    const std::uint64_t target = method.Coordinates().step + 1;
    Transaction transaction;
    LifecycleFailure failure = LifecycleFailure::none;
    LifecycleAttemptOutcome outcome{};
    bool committed = false;

    if (!method.Prepare(transaction, retry, target))
    {
      failure = LifecycleFailure::numerical_prepare;
      outcome = method.PrepareFailure(transaction);
    }
    else
    {
      const bool staged = publication.Stage(
          method, transaction, target, schedule.OutputStep(target),
          schedule.CheckpointStep(target), target == stop_step,
          report.retries);
      if (!staged)
      {
        failure = LifecycleFailure::staging;
        const bool durable_reverted = publication.Revert();
        outcome = method.AbortPrepared(
            std::move(transaction), failure, durable_reverted);
      }
      else if (!method.Install(transaction))
      {
        failure = LifecycleFailure::provisional_install;
        const bool durable_reverted = publication.Revert();
        outcome = method.AbortPrepared(
            std::move(transaction), failure, durable_reverted);
      }
      else
      {
        const bool injected = control.inject_after_recovery &&
            !report.injected && target == schedule.FailureTargetStep();
        const bool ready_to_publish = !injected &&
            publication.ReadyToPublish(
                method, transaction, report.retries);
        if (!ready_to_publish)
        {
          failure = injected ? LifecycleFailure::injected_after_recovery
                             : LifecycleFailure::pre_publication;
          report.injected = report.injected || injected;
          const bool durable_reverted = publication.Revert();
          outcome = method.RevertInstalled(
              std::move(transaction), failure, durable_reverted);
        }
        else if (!publication.Publish(method, transaction))
        {
          failure = LifecycleFailure::publication;
          const bool durable_reverted = publication.Revert();
          outcome = method.RevertInstalled(
              std::move(transaction), failure, durable_reverted);
        }
        else if (!publication.Acknowledge(
                     method, transaction, report.retries))
        {
          failure = LifecycleFailure::post_publication_acknowledgement;
          const bool durable_reverted = publication.Revert();
          outcome = method.RevertInstalled(
              std::move(transaction), failure, durable_reverted);
        }
        else
        {
          // Publication retains exact prior-generation rollback authority
          // until the non-fallible numerical token consumption completes.
          method.Finalize(std::move(transaction));
          publication.Finalize();
          committed = true;
        }
      }
    }

    if (committed)
    {
      retry = false;
      report.status = RunStatus::committed;
      report.reason = FailureReason::none;
      report.coordinates = method.Coordinates();
      continue;
    }

    if (!outcome.rollback_succeeded)
      failure = LifecycleFailure::rollback;
    const bool retry_allowed = outcome.rollback_succeeded &&
        outcome.retryable &&
        (report.retries < control.maximum_retries ||
         (outcome.retry_without_budget && report.retries == 0));
    if (retry_allowed)
    {
      ++report.retries;
      method.OnRetry(failure, report.retries);
      retry = true;
      continue;
    }
    return fail(
        failure,
        failure == LifecycleFailure::restart
            ? FailureReason::restart_incompatible
            : (failure == LifecycleFailure::provisional_install
                   ? FailureReason::publication_precondition
                   : FailureReason::backend_failure));
  }

  report.status = RunStatus::committed;
  report.reason = FailureReason::none;
  report.coordinates = method.Coordinates();
  report.failure = LifecycleFailure::none;
  return report;
}

template <typename Case>
class FrozenCaseSchedule final
{
public:
  constexpr std::uint64_t AcceptedSteps() const noexcept
  { return Case::accepted_steps; }
  constexpr std::uint64_t CheckpointStep() const noexcept
  { return Case::checkpoint_step; }
  constexpr bool OutputStep(const std::uint64_t step) const noexcept
  {
    return step == Case::output_steps[0] ||
        step == Case::output_steps[1] || step == Case::output_steps[2];
  }
  constexpr bool CheckpointStep(const std::uint64_t step) const noexcept
  { return step == Case::checkpoint_step; }
  constexpr std::uint64_t FailureTargetStep() const noexcept
  { return Case::failure_target_step; }
};

template <typename Case, typename Method, typename Publication>
LifecycleReport RunCpuSerialLifecycle(
    Method& method, Publication& publication,
    const LifecycleControl control, Case) noexcept
{
  static_assert(std::is_same<Case, ExactJumpV1Case>::value ||
                    std::is_same<Case, SmoothTransitionV2Case>::value,
                "W8C lifecycle requires one frozen private case descriptor");
  return RunCpuSerialLifecycleWithSchedule(
      method, publication, control, FrozenCaseSchedule<Case>{});
}

template <typename Method, typename Publication>
LifecycleReport RunCpuSerialLifecycle(
    Method& method, Publication& publication,
    const LifecycleControl control) noexcept
{
  return RunCpuSerialLifecycle(
      method, publication, control, ExactJumpV1Case{});
}

} // namespace dg::hdg::runtime::cpu::w8c

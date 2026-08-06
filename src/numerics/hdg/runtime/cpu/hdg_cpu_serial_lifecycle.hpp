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
  provisional_install,
  staging,
  injected_after_recovery,
  publication,
  rollback
};

struct LifecycleControl final
{
  bool restart = false;
  bool split_writer = false;
  bool inject_after_recovery = false;
  std::uint32_t maximum_retries = 0;
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
 * Method is the statically selected mathematical component.  Publication is
 * the application-owned durable output/checkpoint adapter.  The engine alone
 * controls step targets, retry, provisional installation, durable staging,
 * rollback, finalization, and terminal failure propagation.  A method may
 * execute formulation-specific Newton/linear solve hooks while preparing its
 * typed transaction; it cannot publish, advance the accepted-step cursor, or
 * choose a retry.
 */
template <typename Case, typename Method, typename Publication>
LifecycleReport RunCpuSerialLifecycle(
    Method& method, Publication& publication,
    const LifecycleControl control, Case) noexcept
{
  static_assert(std::is_same<Case, ExactJumpV1Case>::value ||
                    std::is_same<Case, SmoothTransitionV2Case>::value,
                "W8C lifecycle requires one frozen private case descriptor");
  static_assert(!std::is_polymorphic<Method>::value,
                "W8C method composition must use static dispatch");
  static_assert(!std::is_polymorphic<Publication>::value,
                "W8C publication composition must use static dispatch");
  using Transaction = typename Method::Transaction;
  static_assert(
      std::is_same<decltype(std::declval<Method&>().Finalize(
                       std::declval<Transaction&&>())), void>::value,
      "W8C finalization is a non-fallible post-publication invariant");
  static_assert(
      noexcept(std::declval<Method&>().Finalize(
          std::declval<Transaction&&>())),
      "W8C finalization must be noexcept after durable publication");

  LifecycleReport report{};
  const auto fail = [&](const LifecycleFailure failure,
                        const FailureReason fallback) noexcept {
    report.status = method.Status();
    if (report.status != RunStatus::terminal_failure)
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
      ? Case::checkpoint_step : Case::accepted_steps;
  bool retry = false;
  while (method.Coordinates().step < stop_step)
  {
    const std::uint64_t target = method.Coordinates().step + 1;
    Transaction transaction;
    if (!method.Prepare(transaction, retry))
    {
      if (method.RetryAvailable() &&
          report.retries < control.maximum_retries)
      {
        ++report.retries;
        retry = true;
        continue;
      }
      return fail(LifecycleFailure::numerical_prepare,
                  FailureReason::backend_failure);
    }
    if (!method.Install(transaction))
      return fail(LifecycleFailure::provisional_install,
                  FailureReason::publication_precondition);

    const bool staged = publication.Stage(
        method, transaction, target,
        target == Case::output_steps[0] || target == Case::output_steps[1] ||
            target == Case::output_steps[2],
        target == Case::checkpoint_step, target == stop_step);
    const bool injected = control.inject_after_recovery && !report.injected &&
        target == Case::failure_target_step;
    if (!staged || injected)
    {
      const bool durable_reverted = publication.Revert();
      const bool state_reverted = method.Revert(std::move(transaction));
      report.injected = report.injected || injected;
      if (!durable_reverted || !state_reverted)
        return fail(LifecycleFailure::rollback,
                    FailureReason::backend_failure);
      if (report.retries < control.maximum_retries)
      {
        ++report.retries;
        retry = true;
        continue;
      }
      return fail(injected ? LifecycleFailure::injected_after_recovery
                           : LifecycleFailure::staging,
                  injected ? FailureReason::backend_failure
                           : FailureReason::backend_failure);
    }

    if (!publication.Publish())
    {
      const bool durable_reverted = publication.Revert();
      const bool state_reverted = method.Revert(std::move(transaction));
      if (!durable_reverted || !state_reverted)
        return fail(LifecycleFailure::rollback,
                    FailureReason::backend_failure);
      return fail(LifecycleFailure::publication,
                  FailureReason::backend_failure);
    }
    method.Finalize(std::move(transaction));
    retry = false;
    report.status = RunStatus::committed;
    report.reason = FailureReason::none;
    report.coordinates = method.Coordinates();
  }

  report.status = RunStatus::committed;
  report.reason = FailureReason::none;
  report.coordinates = method.Coordinates();
  report.failure = LifecycleFailure::none;
  return report;
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

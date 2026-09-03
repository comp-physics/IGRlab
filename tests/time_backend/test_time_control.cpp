#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "sim/hdg/transactional_time_adapter.hpp"
#include "sim/time/backend/time_control.hpp"

namespace
{

using dg_sim::time::backend::FinalOutputPolicy;
using dg_sim::time::backend::EmbeddedErrorAttempt;
using dg_sim::time::backend::AdaptiveStepDecision;
using dg_sim::time::backend::EmbeddedErrorControllerConfig;
using dg_sim::time::backend::EmbeddedErrorControllerState;
using dg_sim::time::backend::EmbeddedErrorTimeController;
using dg_sim::time::backend::InitialOutputPolicy;
using dg_sim::time::backend::IntegratorTag;
using dg_sim::time::backend::IntegratorTagName;
using dg_sim::time::backend::OutputEvent;
using dg_sim::time::backend::OutputSchedule;
using dg_sim::time::backend::StepPlan;
using dg_sim::time::backend::StepPlanCursor;
using dg_sim::time::backend::StepPlanCursorState;
using dg_sim::time::backend::StepControlKind;
using dg_sim::time::backend::TimeControl;
using dg_sim::hdg::HdgStepFailurePhase;
using dg_sim::hdg::HdgTimeControlMode;
using dg_sim::hdg::HdgTransactionalTimeAdapter;
using dg_sim::hdg::HdgTransactionalTimeAdapterState;

class Checks
{
public:
  void Expect(bool condition, const std::string& message)
  {
    if (!condition)
    {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  int failures() const { return failures_; }

private:
  int failures_ = 0;
};

bool Near(double lhs, double rhs)
{
  return std::abs(lhs - rhs) < 1.0e-12;
}

bool SameOutput(const std::optional<dg_sim::time::backend::OutputEvent>& lhs,
                const std::optional<dg_sim::time::backend::OutputEvent>& rhs)
{
  if (lhs.has_value() != rhs.has_value())
  {
    return false;
  }
  return !lhs || (lhs->cycle == rhs->cycle && Near(lhs->time, rhs->time) &&
                  lhs->is_initial == rhs->is_initial &&
                  lhs->is_final == rhs->is_final);
}

bool SamePlan(const StepPlan& lhs, const StepPlan& rhs)
{
  return lhs.sequence == rhs.sequence && lhs.start_time == rhs.start_time &&
      lhs.proposed_dt == rhs.proposed_dt && lhs.dt == rhs.dt &&
      lhs.end_time == rhs.end_time &&
      lhs.hits_scheduled_output == rhs.hits_scheduled_output &&
      lhs.hits_final == rhs.hits_final &&
      lhs.scheduled_cycle == rhs.scheduled_cycle;
}

bool SameTimeControl(const TimeControl& lhs, const TimeControl& rhs)
{
  return lhs.initial_time == rhs.initial_time &&
      lhs.final_time == rhs.final_time && lhs.kind == rhs.kind &&
      lhs.fixed_dt == rhs.fixed_dt && lhs.cfl == rhs.cfl &&
      lhs.adaptive_initial_dt == rhs.adaptive_initial_dt;
}

bool SameOutputSchedule(const OutputSchedule& lhs,
                        const OutputSchedule& rhs)
{
  return lhs.kind == rhs.kind && lhs.interval == rhs.interval &&
      lhs.explicit_times == rhs.explicit_times &&
      lhs.initial == rhs.initial && lhs.final == rhs.final;
}

bool SameCursorState(const StepPlanCursorState& lhs,
                     const StepPlanCursorState& rhs)
{
  return lhs.schema_version == rhs.schema_version &&
      SameTimeControl(lhs.control, rhs.control) &&
      SameOutputSchedule(lhs.output_schedule, rhs.output_schedule) &&
      lhs.integrator == rhs.integrator &&
      lhs.current_time == rhs.current_time &&
      lhs.committed_steps == rhs.committed_steps &&
      lhs.next_sequence == rhs.next_sequence &&
      lhs.next_fixed_cycle == rhs.next_fixed_cycle &&
      lhs.next_publication_cycle == rhs.next_publication_cycle &&
      lhs.next_explicit_index == rhs.next_explicit_index;
}

bool SameControllerState(const EmbeddedErrorControllerState& lhs,
                         const EmbeddedErrorControllerState& rhs)
{
  return lhs.schema_version == rhs.schema_version &&
      lhs.configured_estimator_order == rhs.configured_estimator_order &&
      lhs.configured_minimum_dt == rhs.configured_minimum_dt &&
      lhs.configured_maximum_dt == rhs.configured_maximum_dt &&
      lhs.configured_safety_factor == rhs.configured_safety_factor &&
      lhs.configured_minimum_scale == rhs.configured_minimum_scale &&
      lhs.configured_maximum_rejection_scale ==
          rhs.configured_maximum_rejection_scale &&
      lhs.configured_maximum_growth_scale ==
          rhs.configured_maximum_growth_scale &&
      lhs.consecutive_rejections == rhs.consecutive_rejections &&
      lhs.revision == rhs.revision &&
      lhs.accepted_steps == rhs.accepted_steps &&
      lhs.rejected_attempts == rhs.rejected_attempts &&
      lhs.next_dt == rhs.next_dt &&
      lhs.last_attempted_dt == rhs.last_attempted_dt &&
      lhs.last_normalized_error == rhs.last_normalized_error &&
      lhs.last_step_accepted == rhs.last_step_accepted;
}

bool SameAdapterState(const HdgTransactionalTimeAdapterState& lhs,
                      const HdgTransactionalTimeAdapterState& rhs)
{
  if (lhs.adaptive_controller.has_value() !=
      rhs.adaptive_controller.has_value())
  {
    return false;
  }
  return lhs.schema_version == rhs.schema_version &&
      SameCursorState(lhs.cursor, rhs.cursor) && lhs.mode == rhs.mode &&
      lhs.output_cycle == rhs.output_cycle &&
      (!lhs.adaptive_controller ||
       SameControllerState(*lhs.adaptive_controller,
                           *rhs.adaptive_controller));
}

template <typename Fn>
void ExpectInvalid(Checks& checks, Fn&& fn, const std::string& message)
{
  try
  {
    fn();
    checks.Expect(false, message);
  }
  catch (const std::invalid_argument&)
  {
  }
}

template <typename Fn>
void ExpectLogic(Checks& checks, Fn&& fn, const std::string& message)
{
  try
  {
    fn();
    checks.Expect(false, message);
  }
  catch (const std::logic_error&)
  {
  }
}

[[maybe_unused]] EmbeddedErrorAttempt PassingAttempt(
    const double normalized_error)
{
  return EmbeddedErrorAttempt{
      normalized_error, true, true, true, true};
}

static_assert(std::is_trivial<EmbeddedErrorControllerState>::value,
              "controller checkpoint must remain POD-like");
static_assert(!std::is_aggregate<AdaptiveStepDecision>::value,
              "adaptive decisions must not expose mutable aggregate state");
static_assert(!std::is_default_constructible<AdaptiveStepDecision>::value &&
              !std::is_copy_assignable<AdaptiveStepDecision>::value &&
              !std::is_move_assignable<AdaptiveStepDecision>::value,
              "adaptive decisions must be issued and remain immutable");
static_assert(static_cast<int>(IntegratorTag::BackendDefault) == 0 &&
              static_cast<int>(IntegratorTag::EmbeddedError) == 5 &&
              static_cast<int>(IntegratorTag::Rodas3) == 6,
              "integrator tag identities must remain append-only");
static_assert(static_cast<int>(StepControlKind::Fixed) == 0 &&
              static_cast<int>(StepControlKind::CFL) == 1 &&
              static_cast<int>(StepControlKind::Adaptive) == 2,
              "step-control identities must remain append-only");
static_assert(static_cast<int>(HdgTimeControlMode::FixedBackwardEuler) == 0 &&
              static_cast<int>(HdgTimeControlMode::EmbeddedError) == 5 &&
              static_cast<int>(HdgTimeControlMode::AdaptiveRodas3) == 7,
              "HDG time-control identities must remain append-only");

} // namespace

int main()
{
  Checks checks;

  checks.Expect(std::string(IntegratorTagName(IntegratorTag::BackendDefault)) ==
                  "backend_default" &&
                  std::string(IntegratorTagName(IntegratorTag::SSPRK33)) ==
                    "SSPRK33" &&
                  std::string(IntegratorTagName(IntegratorTag::BackwardEuler)) ==
                    "backward_euler",
                "implemented integrator selections retain stable tags");
  checks.Expect(std::string(IntegratorTagName(IntegratorTag::DIRK)) == "DIRK" &&
                  std::string(IntegratorTagName(IntegratorTag::IMEX)) == "IMEX" &&
                  std::string(IntegratorTagName(IntegratorTag::EmbeddedError)) ==
                    "embedded_error" &&
                  std::string(IntegratorTagName(IntegratorTag::Rodas3)) ==
                    "rodas3",
                "future integrator tags are reserved without implementation");

  {
    StepPlanCursor cursor(TimeControl::Fixed(0.0, 1.0, 0.6),
                          OutputSchedule::FixedInterval(0.25),
                          IntegratorTag::SSPRK33);
    const StepPlan& first = cursor.Plan(99.0);
    checks.Expect(Near(first.proposed_dt, 0.6),
                  "fixed control ignores the backend candidate");
    checks.Expect(Near(first.dt, 0.25) && first.hits_scheduled_output,
                  "fixed control clips at the first output boundary");
    const auto output = cursor.Commit(first);
    checks.Expect(output && output->cycle == 1 && Near(output->time, 0.25),
                  "first fixed output is cycle one at its exact boundary");

    const StepPlan& second = cursor.Plan(0.01);
    checks.Expect(Near(second.start_time, 0.25) && Near(second.dt, 0.25),
                  "integer interval cycles avoid accumulated output drift");
    cursor.Commit(second);
    const StepPlan& third = cursor.Plan(0.01);
    checks.Expect(Near(third.end_time, 0.75),
                  "third interval remains anchored to initial_time + cycle * dt");
  }

  {
    StepPlanCursor cursor(TimeControl::CFL(0.0, 1.0, 0.9),
                          OutputSchedule::Explicit({0.3, 0.3, 0.8}));
    const StepPlan& plan = cursor.Plan(0.7);
    checks.Expect(Near(plan.proposed_dt, 0.7) && Near(plan.dt, 0.3),
                  "CFL proposal is clipped to the next explicit output");
    const auto output = cursor.Commit(plan);
    checks.Expect(output && output->cycle == 1 && Near(output->time, 0.3),
                  "sorted explicit duplicates are deduplicated");
    const StepPlan& next = cursor.Plan(0.7);
    checks.Expect(Near(next.end_time, 0.8),
                  "deduplicated explicit schedule advances to its next entry");
  }

  {
    const double just_after_one = 1.0 + 1.0e-14;
    StepPlanCursor fixed(TimeControl::Fixed(0.0, 2.0, 1.0),
                         OutputSchedule::Explicit({just_after_one},
                                                  InitialOutputPolicy::Suppress,
                                                  FinalOutputPolicy::Suppress));
    const StepPlan& fixed_first = fixed.Plan(0.0);
    checks.Expect(fixed_first.dt == 1.0 && !fixed_first.hits_scheduled_output,
                  "a near-future explicit output never increases fixed dt");
    fixed.Commit(fixed_first);
    const StepPlan& fixed_second = fixed.Plan(0.0);
    checks.Expect(fixed_second.hits_scheduled_output &&
                    fixed_second.dt > 0.0 && fixed_second.dt <= 1.0,
                  "the near-future fixed output is clipped on its following step");

    StepPlanCursor cfl(TimeControl::CFL(0.0, 2.0, 0.5),
                       OutputSchedule::Explicit({just_after_one},
                                                InitialOutputPolicy::Suppress,
                                                FinalOutputPolicy::Suppress));
    const StepPlan& cfl_first = cfl.Plan(1.0);
    checks.Expect(cfl_first.dt == 1.0 && !cfl_first.hits_scheduled_output,
                  "a near-future explicit output never increases CFL dt");
  }

  {
    constexpr double output_interval = 0.025;
    constexpr double final_time = 0.2;
    StepPlanCursor cursor(TimeControl::CFL(0.0, final_time, 0.5),
                          OutputSchedule::FixedInterval(output_interval));
    for (std::uint64_t cycle = 1; cycle <= 5; ++cycle)
    {
      const auto output = cursor.Commit(cursor.Plan(0.03));
      checks.Expect(output && output->cycle == cycle,
                    "boundary-rounding setup advances the preceding output cycles");
    }
    cursor.Commit(cursor.Plan(0.02));

    const double output_time = 6.0 * output_interval;
    const double output_dt = output_time - cursor.time();
    const double just_below_output_dt =
        std::nextafter(output_dt, 0.0);
    checks.Expect(just_below_output_dt < output_dt &&
                      cursor.time() + just_below_output_dt == output_time,
                  "boundary-rounding regression setup straddles one ULP");

    const StepPlan first = cursor.Plan(just_below_output_dt);
    checks.Expect(first.dt == just_below_output_dt &&
                      first.end_time == output_time &&
                      first.hits_scheduled_output &&
                      first.scheduled_cycle == 6,
                  "a rounded endpoint advances its exact output boundary without increasing dt");
    const auto output = cursor.Commit(first);
    const StepPlan second = cursor.Plan(0.001);
    checks.Expect(output && output->cycle == 6 &&
                      output->time == output_time && second.dt > 0.0 &&
                      second.start_time == output_time,
                  "the output cursor advances after a one-ULP boundary landing");
  }

  {
    StepPlanCursor cursor(TimeControl::Fixed(0.0, 1.0, 1.0),
                          OutputSchedule::Explicit({0.0, 0.4, 0.8},
                                                   InitialOutputPolicy::Suppress,
                                                   FinalOutputPolicy::Suppress));
    const auto initial = cursor.InitialOutput();
    checks.Expect(initial && initial->is_initial && initial->cycle == 0,
                  "an explicit initial request emits despite suppressed force");
    const auto first = cursor.Commit(cursor.Plan(0.0));
    checks.Expect(first && first->cycle == 1 && Near(first->time, 0.4),
                  "explicit publications remain contiguous after cycle zero");
    const auto second = cursor.Commit(cursor.Plan(0.0));
    checks.Expect(second && second->cycle == 2 && Near(second->time, 0.8),
                  "explicit publication cycle is independent of vector index");
  }

  {
    StepPlanCursor cursor(TimeControl::Fixed(0.0, 1.0, 1.0),
                          OutputSchedule::Explicit({0.0, 1.0},
                                                   InitialOutputPolicy::Emit,
                                                   FinalOutputPolicy::Emit));
    const auto initial = cursor.InitialOutput();
    checks.Expect(initial && initial->is_initial && initial->cycle == 0,
                  "automatic and explicit initial outputs deduplicate at cycle zero");
    const auto final = cursor.Commit(cursor.Plan(0.0));
    checks.Expect(final && final->is_final && final->cycle == 1,
                  "scheduled and forced final outputs deduplicate at one cycle");
  }

  {
    StepPlanCursor cursor(TimeControl::Fixed(0.0, 1.0, 1.0),
                          OutputSchedule::Explicit({1.0},
                                                   InitialOutputPolicy::Suppress,
                                                   FinalOutputPolicy::Suppress));
    const auto final = cursor.Commit(cursor.Plan(0.0));
    checks.Expect(final && final->is_final && final->cycle == 1,
                  "a scheduled final output remains enabled when force is suppressed");
  }

  {
    StepPlanCursor cursor(TimeControl::Fixed(0.0, 1.0, 2.0),
                          OutputSchedule::Explicit({1.0 + 1.0e-14},
                                                   InitialOutputPolicy::Suppress,
                                                   FinalOutputPolicy::Suppress));
    const auto final = cursor.Commit(cursor.Plan(0.0));
    checks.Expect(final && final->cycle == 1 && final->is_final &&
                    Near(final->time, 1.0),
                  "near-outside final canonicalizes to its scheduled boundary");
  }

  {
    StepPlanCursor cursor(TimeControl::Fixed(0.0, 1.0, 1.0),
                          OutputSchedule::Explicit({-1.0e-14},
                                                   InitialOutputPolicy::Suppress,
                                                   FinalOutputPolicy::Suppress));
    const auto initial = cursor.InitialOutput();
    checks.Expect(initial && initial->cycle == 0 && initial->is_initial &&
                    !initial->is_final && Near(initial->time, 0.0),
                  "near-outside initial canonicalizes to cycle-zero schedule");
  }

  {
    constexpr double short_final = 1.0e-14;
    StepPlanCursor cursor(TimeControl::Fixed(0.0, short_final, 1.0),
                          OutputSchedule::Explicit({short_final},
                                                   InitialOutputPolicy::Suppress,
                                                   FinalOutputPolicy::Suppress));
    const auto final = cursor.Commit(cursor.Plan(0.0));
    checks.Expect(final && final->cycle == 1 && final->is_final &&
                    final->time == short_final,
                  "short-span explicit final remains scheduled when force is suppressed");
  }

  {
    constexpr double short_final = 1.0e-14;
    StepPlanCursor cursor(TimeControl::Fixed(0.0, short_final, 1.0),
                          OutputSchedule::Explicit({0.0, short_final},
                                                   InitialOutputPolicy::Suppress,
                                                   FinalOutputPolicy::Suppress));
    const auto initial = cursor.InitialOutput();
    const auto final = cursor.Commit(cursor.Plan(0.0));
    checks.Expect(initial && initial->cycle == 0 && initial->is_initial &&
                    final && final->cycle == 1 && final->is_final,
                  "short-span explicit initial and final remain distinct publications");
  }

  {
    StepPlanCursor cursor(TimeControl::CFL(0.0, 1.0, 0.5),
                          OutputSchedule::FixedInterval(0.4));
    const StepPlan& original = cursor.Plan(0.7);
    const StepPlan& retry = cursor.Plan(0.01);
    checks.Expect(original.sequence == retry.sequence && Near(original.dt, retry.dt) &&
                    Near(original.end_time, retry.end_time),
                  "a retry reuses the cached plan exactly");
    checks.Expect(Near(cursor.time(), 0.0) && cursor.committed_steps() == 0,
                  "planning alone does not advance the cursor");
    const StepPlan committed = original;
    cursor.Commit(original);
    checks.Expect(Near(cursor.time(), 0.4) && cursor.committed_steps() == 1,
                  "commit advances time and step count once");
    try
    {
      cursor.Commit(committed);
      checks.Expect(false, "a committed plan cannot be committed twice");
    }
    catch (const std::logic_error&)
    {
    }
  }

  {
    StepPlanCursor cursor(TimeControl::Fixed(0.0, 1.0, 0.6),
                          OutputSchedule::FixedInterval(0.25));
    const StepPlan previewed_plan = cursor.Plan(0.0);
    const auto preview = cursor.PreviewCommit(previewed_plan);
    const auto preflight = cursor.PreflightCommit(previewed_plan);
    checks.Expect(Near(cursor.time(), 0.0) && cursor.HasPendingPlan(),
                  "commit preview is pure and retains the pending transaction");
    const auto committed = cursor.Commit(previewed_plan);
    checks.Expect(SameOutput(preview, preflight) &&
                      SameOutput(preflight, committed),
                  "preflighted output exactly equals the later commit event");
  }

  {
    HdgTransactionalTimeAdapter adapter(
      TimeControl::Fixed(0.0, 1.0, 0.6), OutputSchedule::FixedInterval(0.25));
    const StepPlan first = adapter.BeginStep();
    const auto preview = adapter.PreviewFinalizedOutput();
    const auto preflight = adapter.PreflightSuccess();
    adapter.RecordFailure(HdgStepFailurePhase::Assembly);
    adapter.RecordFailure(HdgStepFailurePhase::NonlinearSolve);
    adapter.RecordFailure(HdgStepFailurePhase::LinearSolve);
    adapter.RecordFailure(HdgStepFailurePhase::OutputStaging);
    const StepPlan retry = adapter.RetryPendingStep();
    checks.Expect(first.sequence == retry.sequence && Near(first.dt, retry.dt) &&
                    Near(first.end_time, retry.end_time) &&
                    adapter.solver_step() == 0 && adapter.output_cycle() == 0 &&
                    Near(adapter.time(), 0.0),
                  "all HDG failure phases retain the exact pending transaction");
    const auto committed = adapter.FinalizeSuccess();
    checks.Expect(SameOutput(preview, preflight) &&
                    SameOutput(preflight, committed) &&
                    adapter.solver_step() == 1 &&
                    adapter.output_cycle() == 1 && Near(adapter.time(), 0.25),
                  "HDG finalized success advances solver step and output cycle once");
  }

  {
    HdgTransactionalTimeAdapter adapter(
        TimeControl::CFL(0.0, 0.7, 0.5),
        OutputSchedule::FixedInterval(0.25),
        HdgTimeControlMode::CFLBackwardEuler);
    const StepPlan first = adapter.BeginStep(0.8);
    adapter.RecordFailure(HdgStepFailurePhase::LinearSolve);
    const StepPlan retry = adapter.BeginStep(0.1);
    const auto first_output = adapter.FinalizeSuccess();
    const StepPlan second = adapter.BeginStep(0.4);
    adapter.FinalizeSuccess();
    const StepPlan final = adapter.BeginStep(0.4);
    checks.Expect(Near(first.proposed_dt, 0.8) && Near(first.dt, 0.25) &&
                    first.sequence == retry.sequence &&
                    Near(retry.proposed_dt, first.proposed_dt) &&
                    first_output && Near(first_output->time, 0.25) &&
                    Near(second.proposed_dt, 0.4) && Near(second.dt, 0.25) &&
                    Near(final.proposed_dt, 0.4) && Near(final.dt, 0.2) &&
                    final.hits_final,
                  "HDG CFL proposals are transactional and clip at output and final boundaries");
  }

  {
    HdgTransactionalTimeAdapter fixed(
        TimeControl::Fixed(0.0, 1.0, 0.2), OutputSchedule::Disabled(),
        HdgTimeControlMode::FixedPartitionedBackwardEuler);
    HdgTransactionalTimeAdapter cfl(
        TimeControl::CFL(0.0, 1.0, 0.5), OutputSchedule::Disabled(),
        HdgTimeControlMode::CFLPartitionedBackwardEuler);
    const StepPlan fixed_plan = fixed.BeginStep();
    const StepPlan cfl_plan = cfl.BeginStep(0.3, 0.1);
    checks.Expect(fixed.integrator() == IntegratorTag::BackwardEuler &&
                    cfl.integrator() == IntegratorTag::BackwardEuler &&
                    !fixed.UsesCflControl() && cfl.UsesCflControl() &&
                    Near(fixed_plan.dt, 0.2) && Near(cfl_plan.dt, 0.3),
                  "partitioned backward-euler retains typed fixed/CFL transactions");
  }

  {
    HdgTransactionalTimeAdapter fixed(
        TimeControl::Fixed(0.0, 1.0, 0.2), OutputSchedule::Disabled(),
        HdgTimeControlMode::FixedRodas3);
    const StepPlan plan = fixed.BeginStep();
    checks.Expect(fixed.integrator() == IntegratorTag::Rodas3 &&
                    !fixed.UsesCflControl() &&
                    !fixed.UsesAdaptiveControl() && Near(plan.dt, 0.2),
                  "fixed RODAS3 retains the established transaction path");
  }

  {
    EmbeddedErrorControllerConfig config;
    config.minimum_dt = 0.05;
    config.maximum_dt = 0.8;
    config.maximum_growth_scale = 2.0;
    HdgTransactionalTimeAdapter adapter(
        TimeControl::Adaptive(0.0, 1.0, 0.8),
        OutputSchedule::FixedInterval(0.3),
        config);

    const StepPlan first = adapter.BeginAdaptiveStep();
    ExpectLogic(checks,
                [&adapter]
                {
                  static_cast<void>(adapter.AdaptiveCheckpointState());
                },
                "adaptive checkpoints reject pending attempts");
    ExpectLogic(checks,
                [&adapter]
                {
                  static_cast<void>(adapter.CheckpointState());
                },
                "full time checkpoints reject pending attempts");
    const auto decision =
        adapter.PreviewAdaptiveDecision(PassingAttempt(0.125));
    const auto preview = adapter.PreviewFinalizedOutput();
    const auto preflight = adapter.PreflightAdaptiveSuccess(decision);
    const auto output = adapter.FinalizeAdaptiveSuccess(decision);
    const auto controller_checkpoint = adapter.AdaptiveCheckpointState();
    const auto checkpoint = adapter.CheckpointState();
    checks.Expect(adapter.integrator() == IntegratorTag::Rodas3 &&
                    adapter.UsesAdaptiveControl() &&
                    Near(first.proposed_dt, 0.8) && Near(first.dt, 0.3) &&
                    first.end_time == 0.3 && first.hits_scheduled_output &&
                    decision.Accepted() && Near(decision.scale(), 1.8) &&
                    Near(decision.next_dt(), 0.54) &&
                    SameOutput(preview, preflight) &&
                    SameOutput(preflight, output) && output &&
                    output->time == 0.3 &&
                    controller_checkpoint.revision == 1 &&
                    controller_checkpoint.accepted_steps == 1 &&
                    controller_checkpoint.rejected_attempts == 0 &&
                    Near(controller_checkpoint.next_dt, 0.54),
                  "adaptive RODAS3 accepts collectively and lands exactly on output");

    HdgTransactionalTimeAdapter restored(
        TimeControl::Adaptive(0.0, 1.0, 0.8),
        OutputSchedule::FixedInterval(0.3),
        config,
        checkpoint);
    const StepPlan restored_plan = restored.BeginAdaptiveStep();
    checks.Expect(Near(restored.adaptive_proposed_dt(), 0.54) &&
                    Near(restored_plan.proposed_dt, 0.54) &&
                    restored_plan.dt == 0.3 && restored_plan.end_time == 0.6 &&
                    restored_plan.hits_scheduled_output,
                  "restored controller resumes its proposal and exact output landing");
  }

  {
    EmbeddedErrorControllerConfig config;
    config.minimum_dt = 0.05;
    config.maximum_dt = 1.0;
    HdgTransactionalTimeAdapter adapter(
        TimeControl::Adaptive(0.0, 1.0, 0.4),
        OutputSchedule::Disabled(),
        config);

    const StepPlan first = adapter.BeginAdaptiveStep();
    const auto error_rejection =
        adapter.PreviewAdaptiveDecision(PassingAttempt(8.0));
    adapter.FinalizeAdaptiveRejection(error_rejection);
    const auto after_error = adapter.AdaptiveCheckpointState();
    const StepPlan second = adapter.BeginAdaptiveStep();

    EmbeddedErrorAttempt failed_gate = PassingAttempt(1.0e-3);
    failed_gate.collective_algebraic_closure_satisfied = false;
    const auto gate_rejection =
        adapter.PreviewAdaptiveDecision(failed_gate);
    adapter.FinalizeAdaptiveRejection(gate_rejection);
    const StepPlan third = adapter.BeginAdaptiveStep();
    const auto acceptance =
        adapter.PreviewAdaptiveDecision(PassingAttempt(1.0));
    adapter.FinalizeAdaptiveSuccess(acceptance);
    const auto final_state = adapter.AdaptiveCheckpointState();

    checks.Expect(!error_rejection.Accepted() &&
                    Near(error_rejection.scale(), 0.45) &&
                    Near(error_rejection.next_dt(), 0.18) &&
                    first.sequence != second.sequence &&
                    Near(adapter.time(), 0.09) && adapter.solver_step() == 1 &&
                    after_error.accepted_steps == 0 &&
                    after_error.rejected_attempts == 1 &&
                    after_error.consecutive_rejections == 1 &&
                    Near(second.dt, 0.18) && !gate_rejection.Accepted() &&
                    Near(gate_rejection.scale(), 0.5) &&
                    Near(gate_rejection.next_dt(), 0.09) &&
                    Near(third.dt, 0.09) && acceptance.Accepted() &&
                    final_state.accepted_steps == 1 &&
                    final_state.rejected_attempts == 2 &&
                    final_state.consecutive_rejections == 0,
                  "adaptive rejection rolls back and all collective gates remain mandatory");
  }

  {
    const TimeControl control = TimeControl::Adaptive(0.0, 1.0, 0.5);
    const OutputSchedule schedule = OutputSchedule::Explicit({0.3, 0.7});
    EmbeddedErrorControllerConfig config;
    config.minimum_dt = 0.01;
    config.maximum_dt = 0.5;
    config.maximum_growth_scale = 2.0;

    HdgTransactionalTimeAdapter uninterrupted(control, schedule, config);
    HdgTransactionalTimeAdapter split(control, schedule, config);
    const StepPlan uninterrupted_rejected =
        uninterrupted.BeginAdaptiveStep();
    const StepPlan split_rejected = split.BeginAdaptiveStep();
    const auto uninterrupted_rejection =
        uninterrupted.PreviewAdaptiveDecision(PassingAttempt(8.0));
    const auto split_rejection =
        split.PreviewAdaptiveDecision(PassingAttempt(8.0));
    uninterrupted.FinalizeAdaptiveRejection(uninterrupted_rejection);
    split.FinalizeAdaptiveRejection(split_rejection);

    const HdgTransactionalTimeAdapterState bridge = split.CheckpointState();
    const HdgTransactionalTimeAdapterState uninterrupted_bridge =
        uninterrupted.CheckpointState();
    HdgTransactionalTimeAdapter resumed(
        control, schedule, config, bridge);

    bool future_history_matches =
        SamePlan(uninterrupted_rejected, split_rejected) &&
        SameAdapterState(uninterrupted_bridge, bridge);
    std::vector<OutputEvent> future_outputs;
    int guard = 0;
    while (!uninterrupted.Finished() && !resumed.Finished() && guard++ < 32)
    {
      const StepPlan uninterrupted_plan =
          uninterrupted.BeginAdaptiveStep();
      const StepPlan resumed_plan = resumed.BeginAdaptiveStep();
      const auto uninterrupted_decision =
          uninterrupted.PreviewAdaptiveDecision(PassingAttempt(0.125));
      const auto resumed_decision =
          resumed.PreviewAdaptiveDecision(PassingAttempt(0.125));
      const auto uninterrupted_preview =
          uninterrupted.PreviewFinalizedOutput();
      const auto resumed_preview = resumed.PreviewFinalizedOutput();
      const auto uninterrupted_output =
          uninterrupted.FinalizeAdaptiveSuccess(uninterrupted_decision);
      const auto resumed_output =
          resumed.FinalizeAdaptiveSuccess(resumed_decision);
      future_history_matches = future_history_matches &&
          SamePlan(uninterrupted_plan, resumed_plan) &&
          uninterrupted_decision.Accepted() && resumed_decision.Accepted() &&
          uninterrupted_decision.next_dt() == resumed_decision.next_dt() &&
          SameOutput(uninterrupted_preview, resumed_preview) &&
          SameOutput(uninterrupted_output, resumed_output);
      if (uninterrupted_output)
      {
        future_outputs.push_back(*uninterrupted_output);
      }
    }

    const HdgTransactionalTimeAdapterState uninterrupted_final =
        uninterrupted.CheckpointState();
    const HdgTransactionalTimeAdapterState resumed_final =
        resumed.CheckpointState();
    checks.Expect(bridge.cursor.current_time == 0.0 &&
                    bridge.cursor.committed_steps == 0 &&
                    bridge.cursor.next_sequence == 2 &&
                    bridge.adaptive_controller &&
                    bridge.adaptive_controller->revision == 1 &&
                    bridge.adaptive_controller->rejected_attempts == 1 &&
                    bridge.adaptive_controller->accepted_steps == 0 &&
                    future_history_matches && uninterrupted.Finished() &&
                    resumed.Finished() && guard < 32 &&
                    future_outputs.size() == 3 &&
                    future_outputs[0].cycle == 1 &&
                    Near(future_outputs[0].time, 0.3) &&
                    future_outputs[1].cycle == 2 &&
                    Near(future_outputs[1].time, 0.7) &&
                    future_outputs[2].cycle == 3 &&
                    Near(future_outputs[2].time, 1.0) &&
                    future_outputs[2].is_final &&
                    SameAdapterState(uninterrupted_final, resumed_final),
                  "adaptive restart after rejection preserves future steps and output landing");
  }

  {
    const TimeControl control = TimeControl::Fixed(0.0, 1.0, 0.26);
    const OutputSchedule schedule = OutputSchedule::FixedInterval(0.4);
    HdgTransactionalTimeAdapter uninterrupted(
        control, schedule, HdgTimeControlMode::FixedRodas3);
    HdgTransactionalTimeAdapter split(
        control, schedule, HdgTimeControlMode::FixedRodas3);

    bool history_matches = true;
    for (int i = 0; i < 3; ++i)
    {
      const StepPlan uninterrupted_plan = uninterrupted.BeginStep();
      const StepPlan split_plan = split.BeginStep();
      const auto uninterrupted_output = uninterrupted.FinalizeSuccess();
      const auto split_output = split.FinalizeSuccess();
      history_matches = history_matches &&
          SamePlan(uninterrupted_plan, split_plan) &&
          SameOutput(uninterrupted_output, split_output);
    }

    const HdgTransactionalTimeAdapterState bridge = split.CheckpointState();
    HdgTransactionalTimeAdapter resumed(
        control, schedule, bridge, HdgTimeControlMode::FixedRodas3);
    std::vector<OutputEvent> future_outputs;
    int guard = 0;
    while (!uninterrupted.Finished() && !resumed.Finished() && guard++ < 16)
    {
      const StepPlan uninterrupted_plan = uninterrupted.BeginStep();
      const StepPlan resumed_plan = resumed.BeginStep();
      const auto uninterrupted_output = uninterrupted.FinalizeSuccess();
      const auto resumed_output = resumed.FinalizeSuccess();
      history_matches = history_matches &&
          SamePlan(uninterrupted_plan, resumed_plan) &&
          SameOutput(uninterrupted_output, resumed_output);
      if (uninterrupted_output)
      {
        future_outputs.push_back(*uninterrupted_output);
      }
    }

    checks.Expect(bridge.mode == HdgTimeControlMode::FixedRodas3 &&
                    !bridge.adaptive_controller &&
                    Near(bridge.cursor.current_time, 0.66) &&
                    bridge.cursor.committed_steps == 3 &&
                    bridge.cursor.next_sequence == 4 &&
                    bridge.cursor.next_fixed_cycle == 2 &&
                    bridge.cursor.next_publication_cycle == 2 &&
                    bridge.output_cycle == 1 && history_matches &&
                    uninterrupted.Finished() && resumed.Finished() &&
                    future_outputs.size() == 2 &&
                    future_outputs[0].cycle == 2 &&
                    Near(future_outputs[0].time, 0.8) &&
                    future_outputs[1].cycle == 3 &&
                    Near(future_outputs[1].time, 1.0) &&
                    future_outputs[1].is_final &&
                    SameAdapterState(uninterrupted.CheckpointState(),
                                     resumed.CheckpointState()),
                  "fixed restart preserves scheduled cursors and future output cycles");
  }

  {
    const TimeControl control = TimeControl::Fixed(0.0, 1.0, 0.5);
    const OutputSchedule schedule = OutputSchedule::Explicit(
        {0.0, 0.25, 0.75}, InitialOutputPolicy::Emit,
        FinalOutputPolicy::Emit);
    StepPlanCursor source(control, schedule, IntegratorTag::Rodas3);
    source.Commit(source.Plan(0.0));
    const StepPlanCursorState checkpoint = source.CheckpointState();

    StepPlanCursor target(control, schedule, IntegratorTag::Rodas3);
    const StepPlanCursorState target_before = target.CheckpointState();
    StepPlanCursorState corrupt = checkpoint;
    corrupt.schema_version = 0;
    ExpectInvalid(checks,
                  [&target, &corrupt] { target.Restore(corrupt); },
                  "cursor restore rejects an unsupported schema");
    corrupt = checkpoint;
    corrupt.current_time = std::numeric_limits<double>::quiet_NaN();
    ExpectInvalid(checks,
                  [&target, &corrupt] { target.Restore(corrupt); },
                  "cursor restore rejects a nonfinite time");
    corrupt = checkpoint;
    corrupt.next_sequence = 1;
    ExpectInvalid(checks,
                  [&target, &corrupt] { target.Restore(corrupt); },
                  "cursor restore rejects contradictory step counters");
    corrupt = checkpoint;
    corrupt.next_explicit_index = 1;
    ExpectInvalid(checks,
                  [&target, &corrupt] { target.Restore(corrupt); },
                  "cursor restore rejects contradictory output cursors");
    corrupt = checkpoint;
    corrupt.control.fixed_dt = 0.25;
    ExpectInvalid(checks,
                  [&target, &corrupt] { target.Restore(corrupt); },
                  "cursor restore binds the complete time control");
    corrupt = checkpoint;
    corrupt.output_schedule.explicit_times[1] = 0.3;
    ExpectInvalid(checks,
                  [&target, &corrupt] { target.Restore(corrupt); },
                  "cursor restore binds the complete output schedule");
    corrupt = checkpoint;
    corrupt.integrator = IntegratorTag::BackwardEuler;
    ExpectInvalid(checks,
                  [&target, &corrupt] { target.Restore(corrupt); },
                  "cursor restore binds the integrator identity");
    checks.Expect(SameCursorState(target.CheckpointState(), target_before),
                  "failed cursor restores leave the target unchanged");

    target.Restore(checkpoint);
    checks.Expect(SameCursorState(target.CheckpointState(), checkpoint),
                  "a valid cursor checkpoint restores every field");
    const StepPlan pending = target.Plan(0.0);
    ExpectLogic(checks,
                [&target, &checkpoint] { target.Restore(checkpoint); },
                "cursor restore cannot discard a pending plan");
    checks.Expect(SamePlan(pending, target.PendingPlan()),
                  "rejected pending restore preserves the pending plan");

    StepPlanCursor different_control(
        TimeControl::Fixed(0.0, 1.0, 0.25), schedule,
        IntegratorTag::Rodas3);
    ExpectInvalid(checks,
                  [&different_control, &checkpoint]
                  {
                    different_control.Restore(checkpoint);
                  },
                  "cursor restore rejects a caller configuration mismatch");
  }

  {
    const TimeControl control = TimeControl::Adaptive(0.0, 1.0, 0.5);
    const OutputSchedule schedule = OutputSchedule::Explicit({0.3, 0.7});
    EmbeddedErrorControllerConfig config;
    config.minimum_dt = 0.01;
    config.maximum_dt = 0.5;
    HdgTransactionalTimeAdapter adapter(control, schedule, config);
    static_cast<void>(adapter.BeginAdaptiveStep());
    const auto rejection =
        adapter.PreviewAdaptiveDecision(PassingAttempt(8.0));
    adapter.FinalizeAdaptiveRejection(rejection);
    const HdgTransactionalTimeAdapterState checkpoint =
        adapter.CheckpointState();

    HdgTransactionalTimeAdapterState corrupt = checkpoint;
    corrupt.schema_version = 0;
    ExpectInvalid(
        checks,
        [&control, &schedule, &config, &corrupt]
        {
          static_cast<void>(HdgTransactionalTimeAdapter(
              control, schedule, config, corrupt));
        },
        "adapter restore rejects an unsupported schema");
    corrupt = checkpoint;
    corrupt.mode = HdgTimeControlMode::FixedRodas3;
    ExpectInvalid(
        checks,
        [&control, &schedule, &config, &corrupt]
        {
          static_cast<void>(HdgTransactionalTimeAdapter(
              control, schedule, config, corrupt));
        },
        "adapter restore binds its time-control mode");
    corrupt = checkpoint;
    corrupt.output_cycle = 1;
    ExpectInvalid(
        checks,
        [&control, &schedule, &config, &corrupt]
        {
          static_cast<void>(HdgTransactionalTimeAdapter(
              control, schedule, config, corrupt));
        },
        "adapter restore rejects a contradictory output cycle");
    corrupt = checkpoint;
    corrupt.adaptive_controller.reset();
    ExpectInvalid(
        checks,
        [&control, &schedule, &config, &corrupt]
        {
          static_cast<void>(HdgTransactionalTimeAdapter(
              control, schedule, config, corrupt));
        },
        "adaptive adapter restore requires controller state");
    corrupt = checkpoint;
    corrupt.cursor.next_sequence = 3;
    ExpectInvalid(
        checks,
        [&control, &schedule, &config, &corrupt]
        {
          static_cast<void>(HdgTransactionalTimeAdapter(
              control, schedule, config, corrupt));
        },
        "adaptive adapter restore binds cursor and controller revisions");

    EmbeddedErrorControllerConfig changed_config = config;
    changed_config.safety_factor = 0.8;
    ExpectInvalid(
        checks,
        [&control, &schedule, &changed_config, &checkpoint]
        {
          static_cast<void>(HdgTransactionalTimeAdapter(
              control, schedule, changed_config, checkpoint));
        },
        "adapter restore preserves full controller configuration identity");
    ExpectInvalid(
        checks,
        [&control, &config, &checkpoint]
        {
          static_cast<void>(HdgTransactionalTimeAdapter(
              control, OutputSchedule::Explicit({0.4, 0.7}), config,
              checkpoint));
        },
        "adapter restore rejects an output-schedule mismatch");

    HdgTransactionalTimeAdapter fixed(
        TimeControl::Fixed(0.0, 1.0, 0.2), OutputSchedule::Disabled(),
        HdgTimeControlMode::FixedRodas3);
    HdgTransactionalTimeAdapterState fixed_corrupt = fixed.CheckpointState();
    fixed_corrupt.adaptive_controller = checkpoint.adaptive_controller;
    ExpectInvalid(
        checks,
        [&fixed_corrupt]
        {
          static_cast<void>(HdgTransactionalTimeAdapter(
              TimeControl::Fixed(0.0, 1.0, 0.2),
              OutputSchedule::Disabled(), fixed_corrupt,
              HdgTimeControlMode::FixedRodas3));
        },
        "fixed adapter restore rejects unexpected controller state");
    checks.Expect(SameAdapterState(adapter.CheckpointState(), checkpoint),
                  "failed adapter restores leave the source state unchanged");
  }

  {
    EmbeddedErrorControllerConfig config;
    config.minimum_dt = 0.1;
    config.maximum_dt = 1.0;
    EmbeddedErrorTimeController controller(config, 0.1);
    EmbeddedErrorAttempt nonfinite = PassingAttempt(
        std::numeric_limits<double>::infinity());
    const auto rejected = controller.PreviewDecision(0.1, nonfinite);
    controller.CommitDecision(rejected);
    const EmbeddedErrorControllerState checkpoint = controller.state();
    EmbeddedErrorTimeController restored(config, checkpoint);
    checks.Expect(!rejected.Accepted() && !rejected.error_is_valid() &&
                    Near(rejected.scale(), config.minimum_scale) &&
                    rejected.minimum_dt_reached() &&
                    checkpoint.rejected_attempts == 1 &&
                    checkpoint.consecutive_rejections == 1 &&
                    Near(restored.proposed_dt(), config.minimum_dt),
                  "nonfinite error rejects at the bounded minimum and restores exactly");
  }

  {
    EmbeddedErrorControllerConfig config;
    config.minimum_dt = 0.01;
    config.maximum_dt = 1.0;
    EmbeddedErrorTimeController first(config, 0.2);
    EmbeddedErrorTimeController second(config, 0.2);
    const auto foreign = first.PreviewDecision(0.2, PassingAttempt(0.5));
    const EmbeddedErrorControllerState before = second.state();
    ExpectLogic(checks,
                [&second, &foreign] { second.CommitDecision(foreign); },
                "adaptive decisions are bound to their issuing controller");
    checks.Expect(SameControllerState(second.state(), before),
                  "a foreign adaptive decision cannot mutate controller state");

    const auto accepted = second.PreviewDecision(0.2, PassingAttempt(0.5));
    const auto stale = second.PreviewDecision(0.2, PassingAttempt(0.25));
    second.CommitDecision(accepted);
    const EmbeddedErrorControllerState after_accept = second.state();
    ExpectLogic(checks,
                [&second, &stale] { second.CommitDecision(stale); },
                "adaptive decisions cannot be committed after their revision");
    checks.Expect(SameControllerState(second.state(), after_accept),
                  "a stale adaptive decision cannot mutate controller state");
  }

  {
    EmbeddedErrorControllerConfig config;
    config.minimum_dt = 0.01;
    config.maximum_dt = 1.0;
    const EmbeddedErrorControllerState accepted_with_rejections{
        EmbeddedErrorTimeController::state_schema_version,
        config.estimator_order,
        config.minimum_dt,
        config.maximum_dt,
        config.safety_factor,
        config.minimum_scale,
        config.maximum_rejection_scale,
        config.maximum_growth_scale,
        1,
        2,
        1,
        1,
        0.1,
        0.1,
        0.5,
        1U};
    ExpectInvalid(
        checks,
        [&config, &accepted_with_rejections]
        {
          EmbeddedErrorTimeController(config, accepted_with_rejections);
        },
        "restored acceptance rejects a nonzero consecutive-rejection count");

    const EmbeddedErrorControllerState rejection_without_history{
        EmbeddedErrorTimeController::state_schema_version,
        config.estimator_order,
        config.minimum_dt,
        config.maximum_dt,
        config.safety_factor,
        config.minimum_scale,
        config.maximum_rejection_scale,
        config.maximum_growth_scale,
        0,
        1,
        1,
        0,
        0.1,
        0.1,
        2.0,
        0U};
    ExpectInvalid(
        checks,
        [&config, &rejection_without_history]
        {
          EmbeddedErrorTimeController(config, rejection_without_history);
        },
        "restored rejection requires rejected-attempt history");

    EmbeddedErrorTimeController source(config, 0.1);
    const auto source_acceptance =
        source.PreviewDecision(0.1, PassingAttempt(0.5));
    source.CommitDecision(source_acceptance);
    EmbeddedErrorControllerConfig changed_config = config;
    changed_config.safety_factor = 0.8;
    ExpectInvalid(
        checks,
        [&changed_config, &source]
        {
          EmbeddedErrorTimeController(changed_config, source.state());
        },
        "adaptive restore rejects a controller configuration mismatch");
    EmbeddedErrorControllerState accepted_error_violation = source.state();
    accepted_error_violation.last_normalized_error = 2.0;
    ExpectInvalid(
        checks,
        [&config, &accepted_error_violation]
        {
          EmbeddedErrorTimeController(config, accepted_error_violation);
        },
        "restored acceptance must satisfy the normalized-error gate");

    EmbeddedErrorControllerState derived_proposal_violation = source.state();
    derived_proposal_violation.next_dt *= 0.5;
    ExpectInvalid(
        checks,
        [&config, &derived_proposal_violation]
        {
          EmbeddedErrorTimeController(config, derived_proposal_violation);
        },
        "restored proposal must be derived from the persisted transition");
  }

  {
    HdgTransactionalTimeAdapter adapter(
        TimeControl::CFL(0.0, 1.0, 0.5),
        OutputSchedule::FixedInterval(0.05),
        HdgTimeControlMode::CFLBackwardEuler);
    ExpectInvalid(checks,
                  [&adapter] { adapter.BeginStep(0.04, 0.05); },
                  "HDG applies dt_min to the nominal CFL proposal before output clipping");
    const StepPlan accepted = adapter.BeginStep(0.06, 0.05);
    checks.Expect(Near(accepted.proposed_dt, 0.06) && Near(accepted.dt, 0.05),
                  "HDG admits a nominal CFL proposal before its output clipping");
  }

  {
    HdgTransactionalTimeAdapter adapter(
      TimeControl::Fixed(0.0, 1.0, 0.6),
      OutputSchedule::Disabled(InitialOutputPolicy::Suppress,
                               FinalOutputPolicy::Suppress));
    adapter.BeginStep();
    adapter.FinalizeSuccess();
    const StepPlan final_plan = adapter.BeginStep();
    const auto final_preview = adapter.PreviewFinalizedOutput();
    const auto final = adapter.FinalizeSuccess();
    checks.Expect(Near(final_plan.dt, 0.4) && !final &&
                    SameOutput(final_preview, final) && adapter.solver_step() == 2 &&
                    adapter.output_cycle() == 0,
                  "HDG clips the final remainder without conflating steps and outputs");
  }

  {
    HdgTransactionalTimeAdapter adapter(
      TimeControl::Fixed(0.0, 1.0, 0.3), OutputSchedule::FixedInterval(0.4));
    const auto initial = adapter.InitialOutput();
    adapter.BeginStep();
    adapter.FinalizeSuccess();
    adapter.BeginStep();
    const auto first_output = adapter.FinalizeSuccess();
    adapter.BeginStep();
    adapter.FinalizeSuccess();
    adapter.BeginStep();
    const auto second_output = adapter.FinalizeSuccess();
    adapter.BeginStep();
    const auto final_output = adapter.FinalizeSuccess();
    checks.Expect(!initial && first_output && first_output->cycle == 1 &&
                    Near(first_output->time, 0.4) && second_output &&
                    second_output->cycle == 2 && Near(second_output->time, 0.8) &&
                    final_output && final_output->cycle == 3 && final_output->is_final &&
                    Near(final_output->time, 1.0),
                  "nonintegral save cadence emits deduplicated scheduled and final cycles");
  }

  {
    HdgTransactionalTimeAdapter adapter(
      TimeControl::Fixed(0.0, 1.0, 2.0),
      OutputSchedule::Explicit({0.0, 1.0}, InitialOutputPolicy::Emit,
                               FinalOutputPolicy::Emit));
    const auto initial = adapter.InitialOutput();
    adapter.BeginStep();
    const auto final = adapter.FinalizeSuccess();
    checks.Expect(initial && initial->cycle == 0 && initial->is_initial &&
                    final && final->cycle == 1 && final->is_final &&
                    adapter.output_cycle() == 1,
                  "HDG initial and final output requests deduplicate by cycle");
  }

  {
    StepPlanCursor cursor(TimeControl::Fixed(0.0, 1.0, 2.0),
                          OutputSchedule::Disabled(InitialOutputPolicy::Emit,
                                                   FinalOutputPolicy::Emit),
                          IntegratorTag::BackwardEuler);
    const auto initial = cursor.InitialOutput();
    checks.Expect(initial && initial->is_initial && initial->cycle == 0,
                  "initial output policy emits a cycle-zero event");
    const auto final = cursor.Commit(cursor.Plan(0.0));
    checks.Expect(final && final->is_final && Near(final->time, 1.0),
                  "final output policy emits once even when schedule is disabled");
  }

  {
    StepPlanCursor cursor(TimeControl::Fixed(1.0, 1.0, 1.0),
                          OutputSchedule::Disabled(InitialOutputPolicy::Suppress,
                                                   FinalOutputPolicy::Emit));
    const auto final = cursor.InitialOutput();
    checks.Expect(final && final->cycle == 0 && !final->is_initial &&
                    final->is_final && Near(final->time, 1.0),
                  "a zero-duration forced final is a sole cycle-zero publication");
  }

  {
    StepPlanCursor cursor(TimeControl::Fixed(0.0, 0.3, 1.0),
                          OutputSchedule::FixedInterval(0.1));
    cursor.Commit(cursor.Plan(0.0));
    cursor.Commit(cursor.Plan(0.0));
    const auto final = cursor.Commit(cursor.Plan(0.0));
    checks.Expect(final && final->is_final && final->cycle == 3 &&
                    Near(final->time, 0.3),
                  "floating interval boundary canonicalizes to final time");
  }

  ExpectInvalid(checks,
                [] { TimeControl::Fixed(1.0, 0.0, 0.1).Validate(); },
                "reversed time bounds are rejected");
  ExpectInvalid(checks,
                [] { OutputSchedule::FixedInterval(0.0).Validate(); },
                "nonpositive output intervals are rejected");
  ExpectInvalid(checks,
                [] { OutputSchedule::Explicit({0.5, 0.4}).Validate(); },
                "unsorted explicit output times are rejected");
  ExpectInvalid(checks,
                []
                {
                  StepPlanCursor(TimeControl::Fixed(0.0, 1.0, 0.1),
                                 OutputSchedule::Explicit({1.1}));
                },
                "explicit output beyond final time is rejected");
  ExpectInvalid(checks,
                []
                {
                  StepPlanCursor cursor(TimeControl::CFL(0.0, 1.0, 0.5),
                                        OutputSchedule::Disabled());
                  cursor.Plan(-0.1);
                },
                "nonpositive CFL backend proposals are rejected");
  ExpectInvalid(checks,
                []
                {
                  TimeControl control = TimeControl::Fixed(0.0, 1.0, 0.1);
                  control.kind = static_cast<dg_sim::time::backend::StepControlKind>(99);
                  control.Validate();
                },
                "invalid step control enum is rejected");
  ExpectInvalid(checks,
                []
                {
                  OutputSchedule schedule = OutputSchedule::Disabled();
                  schedule.kind =
                    static_cast<dg_sim::time::backend::OutputScheduleKind>(99);
                  schedule.Validate();
                },
                "invalid output schedule enum is rejected");
  ExpectInvalid(checks,
                []
                {
                  OutputSchedule schedule = OutputSchedule::Disabled();
                  schedule.initial =
                    static_cast<InitialOutputPolicy>(99);
                  schedule.Validate();
                },
                "invalid initial output policy enum is rejected");
  ExpectInvalid(checks,
                []
                {
                  OutputSchedule schedule = OutputSchedule::Disabled();
                  schedule.final = static_cast<FinalOutputPolicy>(99);
                  schedule.Validate();
                },
                "invalid final output policy enum is rejected");
  ExpectInvalid(checks,
                []
                {
                  StepPlanCursor(TimeControl::Fixed(0.0, 1.0, 0.1),
                                 OutputSchedule::Disabled(),
                                 static_cast<IntegratorTag>(99));
                },
                "invalid integrator tag is rejected at cursor construction");
  ExpectInvalid(checks,
                []
                {
                  HdgTransactionalTimeAdapter(
                    TimeControl::CFL(0.0, 1.0, 0.5), OutputSchedule::Disabled(),
                    HdgTimeControlMode::FixedBackwardEuler);
                },
                "HDG fixed Backward-Euler rejects CFL control");
  ExpectInvalid(checks,
                []
                {
                  HdgTransactionalTimeAdapter(
                    TimeControl::Fixed(0.0, 1.0, 0.1), OutputSchedule::Disabled(),
                    HdgTimeControlMode::CFLBackwardEuler);
                },
                "HDG CFL Backward-Euler rejects fixed control");
  ExpectInvalid(checks,
                []
                {
                  HdgTransactionalTimeAdapter(
                    TimeControl::CFL(0.0, 1.0, 0.5),
                    OutputSchedule::Disabled(),
                    HdgTimeControlMode::FixedPartitionedBackwardEuler);
                },
                "HDG fixed partitioned Backward-Euler rejects CFL control");
  ExpectInvalid(checks,
                []
                {
                  HdgTransactionalTimeAdapter(
                    TimeControl::Fixed(0.0, 1.0, 0.1),
                    OutputSchedule::Disabled(),
                    HdgTimeControlMode::IMEX);
                },
                "HDG IMEX control remains reserved until implemented");

  if (checks.failures() != 0)
  {
    return 1;
  }
  std::cout << "time backend contract tests passed\n";
  return 0;
}

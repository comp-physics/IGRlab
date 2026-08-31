#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sim/hdg/transactional_time_adapter.hpp"
#include "sim/time/backend/time_control.hpp"

namespace
{

using dg_sim::time::backend::FinalOutputPolicy;
using dg_sim::time::backend::InitialOutputPolicy;
using dg_sim::time::backend::IntegratorTag;
using dg_sim::time::backend::IntegratorTagName;
using dg_sim::time::backend::OutputSchedule;
using dg_sim::time::backend::StepPlan;
using dg_sim::time::backend::StepPlanCursor;
using dg_sim::time::backend::TimeControl;
using dg_sim::hdg::HdgStepFailurePhase;
using dg_sim::hdg::HdgTimeControlMode;
using dg_sim::hdg::HdgTransactionalTimeAdapter;

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
                    "embedded_error",
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
    checks.Expect(Near(cursor.time(), 0.0) && cursor.HasPendingPlan(),
                  "commit preview is pure and retains the pending transaction");
    const auto committed = cursor.Commit(previewed_plan);
    checks.Expect(SameOutput(preview, committed),
                  "previewed output exactly equals the later commit event");
  }

  {
    HdgTransactionalTimeAdapter adapter(
      TimeControl::Fixed(0.0, 1.0, 0.6), OutputSchedule::FixedInterval(0.25));
    const StepPlan first = adapter.BeginStep();
    const auto preview = adapter.PreviewFinalizedOutput();
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
    checks.Expect(SameOutput(preview, committed) && adapter.solver_step() == 1 &&
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

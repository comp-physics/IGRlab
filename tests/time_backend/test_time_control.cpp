#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

  if (checks.failures() != 0)
  {
    return 1;
  }
  std::cout << "time backend contract tests passed\n";
  return 0;
}

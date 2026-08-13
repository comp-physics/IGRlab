#pragma once

/**
 * @file sim/hdg/transactional_time_adapter.hpp
 * @brief Pure fixed-step Backward-Euler transaction adapter for HDG staging.
 *
 * No solver, MFEM, runtime backend, or output writer is included here.  The
 * caller owns those effects and uses the preview before FinalizeSuccess().
 */

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sim/time/backend/time_control.hpp"

namespace dg_sim::hdg
{

/// Reserved selection tags for future HDG time-control implementations.
enum class HdgTimeControlMode
{
  FixedBackwardEuler,
  CFLBackwardEuler,
  IMEX,
  EmbeddedError
};

/// The phase that failed outside this pure transaction coordinator.
enum class HdgStepFailurePhase
{
  Assembly,
  NonlinearSolve,
  LinearSolve,
  OutputStaging
};

/**
 * @brief Nonvirtual HDG step transaction coordinator.
 *
 * One cursor owns the pending StepPlan.  BeginStep(), retries, and failures
 * retain it byte-for-byte; only FinalizeSuccess() commits it.  Solver-step
 * counting is intentionally separate from the output publication cycle.
 */
class HdgTransactionalTimeAdapter
{
public:
  using OutputEvent = time::backend::OutputEvent;
  using OutputSchedule = time::backend::OutputSchedule;
  using StepPlan = time::backend::StepPlan;
  using StepPlanCursor = time::backend::StepPlanCursor;
  using TimeControl = time::backend::TimeControl;

  HdgTransactionalTimeAdapter(
    TimeControl control,
    OutputSchedule schedule,
    HdgTimeControlMode mode = HdgTimeControlMode::FixedBackwardEuler)
    : cursor_(ValidateAndMoveControl(std::move(control), mode),
              std::move(schedule),
              time::backend::IntegratorTag::BackwardEuler),
      mode_(mode)
  {
  }

  HdgTimeControlMode mode() const { return mode_; }
  double time() const { return cursor_.time(); }
  bool Finished() const { return cursor_.Finished(); }
  bool HasPendingStep() const { return cursor_.HasPendingPlan(); }

  /// The number of finalized solver steps, independent of output events.
  std::uint64_t solver_step() const { return cursor_.committed_steps(); }

  /// The last finalized output cycle; zero denotes cycle-zero/no output.
  std::uint64_t output_cycle() const { return output_cycle_; }

  /// Starts a fixed Backward-Euler transaction, or returns its pending retry.
  const StepPlan& BeginStep() { return cursor_.Plan(0.0); }

  /// Returns the same uncommitted plan after an external failure.
  const StepPlan& RetryPendingStep() const { return cursor_.PendingPlan(); }

  /// Pure preview of the event FinalizeSuccess() will later return.
  std::optional<OutputEvent> PreviewFinalizedOutput() const
  {
    return cursor_.PreviewCommit(cursor_.PendingPlan());
  }

  /// Records no side effects: failure retains the plan for a retry.
  void RecordFailure(HdgStepFailurePhase phase)
  {
    ValidateFailurePhase(phase);
    static_cast<void>(cursor_.PendingPlan());
  }

  /// The sole advancing operation: commit a successfully finalized attempt.
  std::optional<OutputEvent> FinalizeSuccess()
  {
    const std::optional<OutputEvent> output =
      cursor_.Commit(cursor_.PendingPlan());
    if (output)
    {
      output_cycle_ = output->cycle;
    }
    return output;
  }

  std::optional<OutputEvent> InitialOutput() const
  {
    return cursor_.InitialOutput();
  }

private:
  static TimeControl ValidateAndMoveControl(TimeControl control,
                                            HdgTimeControlMode mode)
  {
    if (mode != HdgTimeControlMode::FixedBackwardEuler)
    {
      throw std::invalid_argument(
        "HDG time-control mode is reserved but not implemented");
    }
    if (control.kind != time::backend::StepControlKind::Fixed)
    {
      throw std::invalid_argument(
        "HDG fixed Backward-Euler requires fixed time control");
    }
    return control;
  }

  static void ValidateFailurePhase(HdgStepFailurePhase phase)
  {
    switch (phase)
    {
      case HdgStepFailurePhase::Assembly:
      case HdgStepFailurePhase::NonlinearSolve:
      case HdgStepFailurePhase::LinearSolve:
      case HdgStepFailurePhase::OutputStaging:
        return;
      default:
        throw std::invalid_argument("unknown HDG step failure phase");
    }
  }

  StepPlanCursor cursor_;
  HdgTimeControlMode mode_;
  std::uint64_t output_cycle_ = 0;
};

} // namespace dg_sim::hdg

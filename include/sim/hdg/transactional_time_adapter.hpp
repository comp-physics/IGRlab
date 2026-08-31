#pragma once

/**
 * @file sim/hdg/transactional_time_adapter.hpp
 * @brief Pure fixed/CFL transaction adapter for typed HDG time treatments.
 *
 * No solver, MFEM, runtime backend, or output writer is included here.  The
 * caller owns those effects and uses the preview before FinalizeSuccess().
 */

#include <cstdint>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sim/time/backend/time_control.hpp"

namespace dg_sim::hdg
{

/// Exact time-treatment/control pairing for one HDG transaction cursor.
enum class HdgTimeControlMode
{
  FixedBackwardEuler,
  CFLBackwardEuler,
  FixedPartitionedBackwardEuler,
  CFLPartitionedBackwardEuler,
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
              IntegratorForMode(mode)),
      mode_(mode)
  {
  }

  HdgTimeControlMode mode() const { return mode_; }
  time::backend::IntegratorTag integrator() const
  {
    return cursor_.integrator();
  }
  bool UsesCflControl() const { return IsCflMode(mode_); }
  double time() const { return cursor_.time(); }
  bool Finished() const { return cursor_.Finished(); }
  bool HasPendingStep() const { return cursor_.HasPendingPlan(); }

  /// The number of finalized solver steps, independent of output events.
  std::uint64_t solver_step() const { return cursor_.committed_steps(); }

  /// The last finalized output cycle; zero denotes cycle-zero/no output.
  std::uint64_t output_cycle() const { return output_cycle_; }

  /**
   * Start one typed HDG transaction, or return its pending retry.
   *
   * In fixed mode @p proposed_dt is intentionally ignored by the underlying
   * planner.  In CFL mode it must be the backend's finite, positive,
   * communicator-reduced stable step before output/final-time clipping.
   */
  const StepPlan& BeginStep(const double proposed_dt)
  {
    return cursor_.Plan(proposed_dt);
  }

  /**
   * Start a transaction after enforcing a minimum on the nominal CFL
   * proposal.  The minimum is deliberately checked before cursor clipping at
   * an output or final-time boundary.
   */
  const StepPlan& BeginStep(const double proposed_dt,
                            const double minimum_nominal_dt)
  {
    if (!std::isfinite(minimum_nominal_dt) || minimum_nominal_dt < 0.0)
    {
      throw std::invalid_argument(
          "HDG minimum nominal dt must be finite and non-negative");
    }
    if (IsCflMode(mode_) &&
        proposed_dt < minimum_nominal_dt)
    {
      throw std::invalid_argument(
          "HDG CFL nominal dt is below the configured minimum");
    }
    return BeginStep(proposed_dt);
  }

  /// Compatibility spelling for a fixed-control transaction.
  const StepPlan& BeginStep() { return BeginStep(0.0); }

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
  static constexpr bool IsCflMode(const HdgTimeControlMode mode) noexcept
  {
    return mode == HdgTimeControlMode::CFLBackwardEuler ||
        mode == HdgTimeControlMode::CFLPartitionedBackwardEuler;
  }

  static time::backend::IntegratorTag IntegratorForMode(
      const HdgTimeControlMode mode)
  {
    switch (mode)
    {
    case HdgTimeControlMode::FixedBackwardEuler:
    case HdgTimeControlMode::CFLBackwardEuler:
    case HdgTimeControlMode::FixedPartitionedBackwardEuler:
    case HdgTimeControlMode::CFLPartitionedBackwardEuler:
      return time::backend::IntegratorTag::BackwardEuler;
    case HdgTimeControlMode::IMEX:
      return time::backend::IntegratorTag::IMEX;
    case HdgTimeControlMode::EmbeddedError:
      return time::backend::IntegratorTag::EmbeddedError;
    }
    throw std::invalid_argument("unknown HDG time-control mode");
  }

  static TimeControl ValidateAndMoveControl(TimeControl control,
                                            HdgTimeControlMode mode)
  {
    const bool fixed_mode = mode == HdgTimeControlMode::FixedBackwardEuler ||
        mode == HdgTimeControlMode::FixedPartitionedBackwardEuler;
    const bool cfl_mode = IsCflMode(mode);
    const bool fixed_pair = fixed_mode &&
        control.kind == time::backend::StepControlKind::Fixed;
    const bool cfl_pair = cfl_mode &&
        control.kind == time::backend::StepControlKind::CFL;
    if (!fixed_pair && !cfl_pair)
    {
      throw std::invalid_argument(
          "HDG time-control requires an exact integrator/control pairing");
    }
    control.Validate();
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

#pragma once

/**
 * @file sim/hdg/transactional_time_adapter.hpp
 * @brief Pure fixed/CFL/adaptive transactions for typed HDG time treatments.
 *
 * No solver, MFEM, runtime backend, or output writer is included here.  The
 * caller owns those effects and uses the preview before FinalizeSuccess().
 */

#include <cstdint>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "sim/time/backend/adaptive_time_control.hpp"
#include "sim/time/backend/time_control.hpp"

namespace dg_sim::hdg
{

/// Exact time-treatment/control pairing for one HDG transaction cursor.
enum class HdgTimeControlMode
{
  FixedBackwardEuler = 0,
  CFLBackwardEuler = 1,
  FixedPartitionedBackwardEuler = 2,
  CFLPartitionedBackwardEuler = 3,
  IMEX = 4,
  EmbeddedError = 5,
  FixedRodas3 = 6,
  AdaptiveRodas3 = 7
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
 * Fieldwise committed-boundary restart record for the HDG time adapter.
 * The optional controller state is present exactly for AdaptiveRodas3.
 */
struct HdgTransactionalTimeAdapterState
{
  std::uint32_t schema_version = 0;
  time::backend::StepPlanCursorState cursor;
  HdgTimeControlMode mode = HdgTimeControlMode::FixedBackwardEuler;
  std::uint64_t output_cycle = 0;
  std::optional<time::backend::EmbeddedErrorControllerState>
      adaptive_controller;
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
  static constexpr std::uint32_t state_schema_version = 1;

  using AdaptiveAttempt = time::backend::EmbeddedErrorAttempt;
  using AdaptiveControllerConfig =
      time::backend::EmbeddedErrorControllerConfig;
  using AdaptiveControllerState =
      time::backend::EmbeddedErrorControllerState;
  using AdaptiveDecision = time::backend::AdaptiveStepDecision;
  using OutputEvent = time::backend::OutputEvent;
  using OutputSchedule = time::backend::OutputSchedule;
  using RestartState = HdgTransactionalTimeAdapterState;
  using StepPlan = time::backend::StepPlan;
  using StepPlanCursor = time::backend::StepPlanCursor;
  using TimeControl = time::backend::TimeControl;

  HdgTransactionalTimeAdapter(
    TimeControl control,
    OutputSchedule schedule,
    HdgTimeControlMode mode = HdgTimeControlMode::FixedBackwardEuler)
    : cursor_(ValidateNonAdaptiveAndMoveControl(std::move(control), mode),
              std::move(schedule),
              IntegratorForMode(mode)),
      mode_(mode)
  {
  }

  HdgTransactionalTimeAdapter(
    TimeControl control,
    OutputSchedule schedule,
    const RestartState& restored_state,
    HdgTimeControlMode mode = HdgTimeControlMode::FixedBackwardEuler)
    : cursor_(ValidateNonAdaptiveAndMoveControl(std::move(control), mode),
              std::move(schedule),
              IntegratorForMode(mode)),
      mode_(mode)
  {
    ValidateRestartEnvelope(restored_state, mode_, false);
    RestoreCursorAndOutput(restored_state);
  }

  HdgTransactionalTimeAdapter(
    TimeControl control,
    OutputSchedule schedule,
    AdaptiveControllerConfig adaptive_config,
    HdgTimeControlMode mode = HdgTimeControlMode::AdaptiveRodas3)
    : cursor_(ValidateAdaptiveAndMoveControl(std::move(control), mode),
              std::move(schedule),
              IntegratorForMode(mode)),
      mode_(mode),
      adaptive_controller_(std::in_place,
                           adaptive_config,
                           cursor_.control().adaptive_initial_dt)
  {
  }

  HdgTransactionalTimeAdapter(
    TimeControl control,
    OutputSchedule schedule,
    AdaptiveControllerConfig adaptive_config,
    const RestartState& restored_state,
    HdgTimeControlMode mode = HdgTimeControlMode::AdaptiveRodas3)
    : cursor_(ValidateAdaptiveAndMoveControl(std::move(control), mode),
              std::move(schedule),
              IntegratorForMode(mode)),
      mode_(mode),
      adaptive_controller_(std::in_place,
                           adaptive_config,
                           RestoredAdaptiveControllerState(restored_state,
                                                           mode))
  {
    RestoreCursorAndOutput(restored_state);
  }

  HdgTimeControlMode mode() const { return mode_; }
  time::backend::IntegratorTag integrator() const
  {
    return cursor_.integrator();
  }
  bool UsesCflControl() const { return IsCflMode(mode_); }
  bool UsesAdaptiveControl() const { return IsAdaptiveMode(mode_); }
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
    if (UsesAdaptiveControl())
    {
      throw std::logic_error(
          "adaptive HDG steps use the controller proposal");
    }
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

  /// Compatibility spelling for fixed control and adaptive controller steps.
  const StepPlan& BeginStep()
  {
    return UsesAdaptiveControl() ? BeginAdaptiveStep() : BeginStep(0.0);
  }

  /** Begin or retry a RODAS3 step from the persisted controller proposal. */
  const StepPlan& BeginAdaptiveStep()
  {
    RequireAdaptiveController();
    return cursor_.Plan(adaptive_controller_->proposed_dt());
  }

  /// Returns the same uncommitted plan after an external failure.
  const StepPlan& RetryPendingStep() const { return cursor_.PendingPlan(); }

  /// Pure preview of the event FinalizeSuccess() will later return.
  std::optional<OutputEvent> PreviewFinalizedOutput() const
  {
    return cursor_.PreviewCommit(cursor_.PendingPlan());
  }

  /**
   * Validate a non-adaptive success before an external PDE commit.
   *
   * With no intervening time-side mutation, FinalizeSuccess() then has no
   * remaining fallible validation or counter-overflow condition.
   */
  std::optional<OutputEvent> PreflightSuccess() const
  {
    if (UsesAdaptiveControl())
    {
      throw std::logic_error(
          "adaptive HDG success requires its accepted controller decision");
    }
    return cursor_.PreflightCommit(cursor_.PendingPlan());
  }

  /** Validate an accepted adaptive success before an external PDE commit. */
  std::optional<OutputEvent> PreflightAdaptiveSuccess(
      const AdaptiveDecision& decision) const
  {
    ValidateAdaptiveDecisionForPending(decision, true);
    adaptive_controller_->PreflightCommitDecision(decision);
    return cursor_.PreflightCommit(cursor_.PendingPlan());
  }

  /// Records no side effects: failure retains the plan for a retry.
  void RecordFailure(HdgStepFailurePhase phase)
  {
    ValidateFailurePhase(phase);
    static_cast<void>(cursor_.PendingPlan());
  }

  /** Pure assessment of already-collective error and solver gates. */
  AdaptiveDecision PreviewAdaptiveDecision(
      const AdaptiveAttempt& attempt) const
  {
    RequireAdaptiveController();
    return adaptive_controller_->PreviewDecision(
        cursor_.PendingPlan().dt, attempt);
  }

  /**
   * Commit a rejected controller transition and cancel the solver attempt.
   * Time, committed-step, and output cursors remain unchanged.
   */
  void FinalizeAdaptiveRejection(const AdaptiveDecision& decision)
  {
    ValidateAdaptiveDecisionForPending(decision, false);
    adaptive_controller_->CommitDecision(decision);
    cursor_.Abort(cursor_.PendingPlan());
  }

  /// The sole advancing operation: commit a successfully finalized attempt.
  std::optional<OutputEvent> FinalizeSuccess()
  {
    if (UsesAdaptiveControl())
    {
      throw std::logic_error(
          "adaptive HDG success requires its accepted controller decision");
    }
    return CommitPendingStep();
  }

  /** Advance an accepted adaptive decision after solver/output commit. */
  std::optional<OutputEvent> FinalizeAdaptiveSuccess(
      const AdaptiveDecision& decision)
  {
    ValidateAdaptiveDecisionForPending(decision, true);
    adaptive_controller_->CommitDecision(decision);
    return CommitPendingStep();
  }

  /**
   * Controller checkpoint for a committed boundary.
   *
   * A pending attempt must first be accepted or rolled back; this prevents a
   * restart record from mixing committed PDE state with an attempted dt.
   */
  AdaptiveControllerState AdaptiveCheckpointState() const
  {
    RequireAdaptiveController();
    if (HasPendingStep())
    {
      throw std::logic_error(
          "adaptive controller cannot checkpoint a pending attempt");
    }
    return adaptive_controller_->state();
  }

  /** Capture the complete time-side restart state at a committed boundary. */
  RestartState CheckpointState() const
  {
    RestartState state;
    state.schema_version = state_schema_version;
    state.cursor = cursor_.CheckpointState();
    state.mode = mode_;
    state.output_cycle = output_cycle_;
    if (adaptive_controller_)
    {
      state.adaptive_controller = adaptive_controller_->state();
    }
    ValidateRestartRelationship(state);
    return state;
  }

  const AdaptiveControllerConfig& adaptive_controller_config() const
  {
    RequireAdaptiveController();
    return adaptive_controller_->config();
  }

  double adaptive_proposed_dt() const
  {
    RequireAdaptiveController();
    return adaptive_controller_->proposed_dt();
  }

  std::optional<OutputEvent> InitialOutput() const
  {
    return cursor_.InitialOutput();
  }

private:
  static void ValidateRestartEnvelope(const RestartState& state,
                                      const HdgTimeControlMode mode,
                                      const bool expect_adaptive)
  {
    if (state.schema_version != state_schema_version)
    {
      throw std::invalid_argument(
          "HDG time adapter state schema is unsupported");
    }
    if (state.mode != mode)
    {
      throw std::invalid_argument(
          "HDG time adapter state belongs to a different mode");
    }
    if (state.adaptive_controller.has_value() != expect_adaptive)
    {
      throw std::invalid_argument(
          "HDG time adapter controller state presence is inconsistent");
    }
  }

  static AdaptiveControllerState RestoredAdaptiveControllerState(
      const RestartState& state,
      const HdgTimeControlMode mode)
  {
    ValidateRestartEnvelope(state, mode, true);
    return *state.adaptive_controller;
  }

  void RestoreCursorAndOutput(const RestartState& state)
  {
    cursor_.Restore(state.cursor);
    ValidateRestartRelationship(state);
    output_cycle_ = state.output_cycle;
  }

  void ValidateRestartRelationship(const RestartState& state) const
  {
    if (state.output_cycle != cursor_.LastCommittedOutputCycle())
    {
      throw std::invalid_argument(
          "HDG time adapter output cycle is inconsistent");
    }

    if (UsesAdaptiveControl())
    {
      if (!state.adaptive_controller)
      {
        throw std::invalid_argument(
            "adaptive HDG time adapter state lacks its controller");
      }
      const AdaptiveControllerState& controller =
          *state.adaptive_controller;
      if (controller.accepted_steps != cursor_.committed_steps() ||
          controller.revision ==
              std::numeric_limits<std::uint64_t>::max() ||
          state.cursor.next_sequence != controller.revision + 1)
      {
        throw std::invalid_argument(
            "adaptive HDG cursor and controller histories diverge");
      }
      return;
    }

    if (state.adaptive_controller ||
        state.cursor.committed_steps ==
            std::numeric_limits<std::uint64_t>::max() ||
        state.cursor.next_sequence != state.cursor.committed_steps + 1)
    {
      throw std::invalid_argument(
          "non-adaptive HDG cursor history is inconsistent");
    }
  }

  std::optional<OutputEvent> CommitPendingStep()
  {
    const std::optional<OutputEvent> output =
      cursor_.Commit(cursor_.PendingPlan());
    if (output)
    {
      output_cycle_ = output->cycle;
    }
    return output;
  }
  static constexpr bool IsCflMode(const HdgTimeControlMode mode) noexcept
  {
    return mode == HdgTimeControlMode::CFLBackwardEuler ||
        mode == HdgTimeControlMode::CFLPartitionedBackwardEuler;
  }

  static constexpr bool IsAdaptiveMode(
      const HdgTimeControlMode mode) noexcept
  {
    return mode == HdgTimeControlMode::AdaptiveRodas3;
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
    case HdgTimeControlMode::FixedRodas3:
    case HdgTimeControlMode::AdaptiveRodas3:
      return time::backend::IntegratorTag::Rodas3;
    }
    throw std::invalid_argument("unknown HDG time-control mode");
  }

  static TimeControl ValidateAndMoveControl(TimeControl control,
                                            HdgTimeControlMode mode)
  {
    const bool fixed_mode = mode == HdgTimeControlMode::FixedBackwardEuler ||
        mode == HdgTimeControlMode::FixedPartitionedBackwardEuler ||
        mode == HdgTimeControlMode::FixedRodas3;
    const bool cfl_mode = IsCflMode(mode);
    const bool adaptive_mode = IsAdaptiveMode(mode);
    const bool fixed_pair = fixed_mode &&
        control.kind == time::backend::StepControlKind::Fixed;
    const bool cfl_pair = cfl_mode &&
        control.kind == time::backend::StepControlKind::CFL;
    const bool adaptive_pair = adaptive_mode &&
        control.kind == time::backend::StepControlKind::Adaptive;
    if (!fixed_pair && !cfl_pair && !adaptive_pair)
    {
      throw std::invalid_argument(
          "HDG time-control requires an exact integrator/control pairing");
    }
    control.Validate();
    return control;
  }

  static TimeControl ValidateAdaptiveAndMoveControl(
      TimeControl control,
      const HdgTimeControlMode mode)
  {
    if (!IsAdaptiveMode(mode))
    {
      throw std::invalid_argument(
          "adaptive controller constructor requires AdaptiveRodas3 mode");
    }
    return ValidateAndMoveControl(std::move(control), mode);
  }

  static TimeControl ValidateNonAdaptiveAndMoveControl(
      TimeControl control,
      const HdgTimeControlMode mode)
  {
    if (IsAdaptiveMode(mode))
    {
      throw std::invalid_argument(
          "AdaptiveRodas3 mode requires adaptive controller configuration");
    }
    return ValidateAndMoveControl(std::move(control), mode);
  }

  void RequireAdaptiveController() const
  {
    if (!UsesAdaptiveControl() || !adaptive_controller_)
    {
      throw std::logic_error(
          "HDG time adapter has no adaptive controller");
    }
  }

  void ValidateAdaptiveDecisionForPending(
      const AdaptiveDecision& decision,
      const bool expected_acceptance) const
  {
    RequireAdaptiveController();
    const StepPlan& pending = cursor_.PendingPlan();
    if (decision.Accepted() != expected_acceptance)
    {
      throw std::invalid_argument(
          "adaptive decision has the wrong disposition for finalization");
    }
    if (decision.attempted_dt() != pending.dt)
    {
      throw std::invalid_argument(
          "adaptive decision does not describe the pending attempt");
    }
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
  std::optional<time::backend::EmbeddedErrorTimeController>
      adaptive_controller_;
};

} // namespace dg_sim::hdg

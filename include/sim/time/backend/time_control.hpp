#pragma once

/**
 * @file sim/time/backend/time_control.hpp
 * @brief Pure host-side time-step and output-boundary planning contract.
 *
 * This deliberately has no dependency on MFEM, a particular integrator, or a
 * backend.  Runtime wiring owns translating its configuration and a backend
 * CFL estimate into this value contract.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dg_sim::time::backend
{

/// The selection tag is data only; no integrator implementation lives here.
enum class IntegratorTag
{
  BackendDefault = 0,
  SSPRK33 = 1,
  BackwardEuler = 2,
  // Reserved tags: a later backend may implement them without changing this
  // scheduling contract.
  DIRK = 3,
  IMEX = 4,
  EmbeddedError = 5,
  Rodas3 = 6
};

inline void ValidateIntegratorTag(IntegratorTag tag)
{
  switch (tag)
  {
    case IntegratorTag::BackendDefault:
    case IntegratorTag::SSPRK33:
    case IntegratorTag::BackwardEuler:
    case IntegratorTag::DIRK:
    case IntegratorTag::IMEX:
    case IntegratorTag::EmbeddedError:
    case IntegratorTag::Rodas3:
      return;
    default:
      throw std::invalid_argument("unknown integrator tag");
  }
}

inline const char* IntegratorTagName(IntegratorTag tag)
{
  switch (tag)
  {
    case IntegratorTag::BackendDefault:
      return "backend_default";
    case IntegratorTag::SSPRK33:
      return "SSPRK33";
    case IntegratorTag::BackwardEuler:
      return "backward_euler";
    case IntegratorTag::DIRK:
      return "DIRK";
    case IntegratorTag::IMEX:
      return "IMEX";
    case IntegratorTag::EmbeddedError:
      return "embedded_error";
    case IntegratorTag::Rodas3:
      return "rodas3";
  }
  throw std::invalid_argument("unknown integrator tag");
}

enum class StepControlKind
{
  Fixed = 0,
  CFL = 1,
  Adaptive = 2
};

/**
 * @brief Fixed/CFL/adaptive proposal source, independent of IntegratorTag.
 *
 * In fixed mode the backend proposal passed to StepPlanCursor::Plan is
 * ignored.  In CFL mode that proposal is the backend's already-reduced CFL
 * candidate.  In adaptive mode it is the pure controller's current proposal.
 * Both proposal modes are validated here before boundary clipping.
 */
struct TimeControl
{
  double initial_time = 0.0;
  double final_time = 0.0;
  StepControlKind kind = StepControlKind::Fixed;
  double fixed_dt = 0.0;
  double cfl = 0.0;
  double adaptive_initial_dt = 0.0;

  static TimeControl Fixed(double initial_time,
                           double final_time,
                           double fixed_dt)
  {
    return {initial_time,
            final_time,
            StepControlKind::Fixed,
            fixed_dt,
            0.0,
            0.0};
  }

  static TimeControl CFL(double initial_time, double final_time, double cfl)
  {
    return {initial_time,
            final_time,
            StepControlKind::CFL,
            0.0,
            cfl,
            0.0};
  }

  static TimeControl Adaptive(double initial_time,
                              double final_time,
                              double initial_dt)
  {
    return {initial_time,
            final_time,
            StepControlKind::Adaptive,
            0.0,
            0.0,
            initial_dt};
  }

  void Validate() const
  {
    if (!std::isfinite(initial_time) || !std::isfinite(final_time))
    {
      throw std::invalid_argument("time bounds must be finite");
    }
    if (final_time < initial_time)
    {
      throw std::invalid_argument("final time must not precede initial time");
    }
    switch (kind)
    {
      case StepControlKind::Fixed:
        if (!std::isfinite(fixed_dt) || fixed_dt <= 0.0)
        {
          throw std::invalid_argument("fixed dt must be finite and positive");
        }
        break;
      case StepControlKind::CFL:
        if (!std::isfinite(cfl) || cfl <= 0.0)
        {
          throw std::invalid_argument(
            "CFL number must be finite and positive");
        }
        break;
      case StepControlKind::Adaptive:
        if (!std::isfinite(adaptive_initial_dt) ||
            adaptive_initial_dt <= 0.0)
        {
          throw std::invalid_argument(
            "adaptive initial dt must be finite and positive");
        }
        break;
      default:
        throw std::invalid_argument("unknown step control kind");
    }
  }

  double ResolveDt(double backend_proposed_dt) const
  {
    Validate();
    if (kind == StepControlKind::Fixed)
    {
      return fixed_dt;
    }
    if (!std::isfinite(backend_proposed_dt) || backend_proposed_dt <= 0.0)
    {
      throw std::invalid_argument(kind == StepControlKind::CFL
          ? "backend CFL proposal must be finite and positive"
          : "adaptive controller proposal must be finite and positive");
    }
    return backend_proposed_dt;
  }
};

enum class OutputScheduleKind
{
  Disabled = 0,
  FixedInterval = 1,
  ExplicitTimes = 2
};

enum class InitialOutputPolicy
{
  Suppress = 0,
  Emit = 1
};

enum class FinalOutputPolicy
{
  Suppress = 0,
  Emit = 1
};

/**
 * @brief Output event policy and regular-boundary definition.
 *
 * The initial/final policies are additive force flags.  A scheduled explicit
 * entry at either boundary is always emitted; Emit adds that boundary only if
 * the schedule does not already do so.  Coincident scheduled and forced
 * events are published once.
 */
struct OutputSchedule
{
  OutputScheduleKind kind = OutputScheduleKind::Disabled;
  double interval = 0.0;
  std::vector<double> explicit_times;
  InitialOutputPolicy initial = InitialOutputPolicy::Suppress;
  FinalOutputPolicy final = FinalOutputPolicy::Emit;

  static OutputSchedule Disabled(InitialOutputPolicy initial =
                                   InitialOutputPolicy::Suppress,
                                 FinalOutputPolicy final =
                                   FinalOutputPolicy::Suppress)
  {
    return {OutputScheduleKind::Disabled, 0.0, {}, initial, final};
  }

  static OutputSchedule FixedInterval(double interval,
                                      InitialOutputPolicy initial =
                                        InitialOutputPolicy::Suppress,
                                      FinalOutputPolicy final =
                                        FinalOutputPolicy::Emit)
  {
    return {OutputScheduleKind::FixedInterval, interval, {}, initial, final};
  }

  static OutputSchedule Explicit(std::vector<double> times,
                                 InitialOutputPolicy initial =
                                   InitialOutputPolicy::Suppress,
                                 FinalOutputPolicy final =
                                   FinalOutputPolicy::Emit)
  {
    return {OutputScheduleKind::ExplicitTimes,
            0.0,
            std::move(times),
            initial,
            final};
  }

  void Validate() const
  {
    if (initial != InitialOutputPolicy::Suppress &&
        initial != InitialOutputPolicy::Emit)
    {
      throw std::invalid_argument("unknown initial output policy");
    }
    if (final != FinalOutputPolicy::Suppress &&
        final != FinalOutputPolicy::Emit)
    {
      throw std::invalid_argument("unknown final output policy");
    }
    switch (kind)
    {
      case OutputScheduleKind::Disabled:
        return;
      case OutputScheduleKind::FixedInterval:
        if (!std::isfinite(interval) || interval <= 0.0)
        {
          throw std::invalid_argument(
            "output interval must be finite and positive");
        }
        return;
      case OutputScheduleKind::ExplicitTimes:
        for (std::size_t i = 0; i < explicit_times.size(); ++i)
        {
          if (!std::isfinite(explicit_times[i]))
          {
            throw std::invalid_argument("explicit output time must be finite");
          }
          if (i > 0 && explicit_times[i] < explicit_times[i - 1])
          {
            throw std::invalid_argument(
              "explicit output times must be sorted");
          }
        }
        return;
      default:
        throw std::invalid_argument("unknown output schedule kind");
    }
  }
};

struct OutputEvent
{
  std::uint64_t cycle = 0;
  double time = 0.0;
  bool is_initial = false;
  bool is_final = false;
};

/** An immutable proposed step, valid until its owning cursor commits it. */
struct StepPlan
{
  std::uint64_t sequence = 0;
  double start_time = 0.0;
  double proposed_dt = 0.0;
  double dt = 0.0;
  double end_time = 0.0;
  bool hits_scheduled_output = false;
  bool hits_final = false;
  std::uint64_t scheduled_cycle = 0;
};

/**
 * @brief Fieldwise restart record for a committed StepPlanCursor boundary.
 *
 * Checkpoint codecs must serialize the named configuration and cursor fields,
 * including each explicit output time, rather than object bytes or padding.
 * A pending plan is deliberately absent: it belongs to an uncommitted solver
 * attempt and must be accepted or aborted before this record can be produced.
 */
struct StepPlanCursorState
{
  std::uint32_t schema_version = 0;
  TimeControl control;
  OutputSchedule output_schedule;
  IntegratorTag integrator = IntegratorTag::BackendDefault;
  double current_time = 0.0;
  std::uint64_t committed_steps = 0;
  std::uint64_t next_sequence = 0;
  std::uint64_t next_fixed_cycle = 0;
  std::uint64_t next_publication_cycle = 0;
  std::uint64_t next_explicit_index = 0;
};

/**
 * @brief Nonvirtual commit cursor for deterministic output-aligned stepping.
 *
 * Plan() caches its result.  Retrying an attempted backend step therefore
 * returns the exact same values even if a caller presents a different CFL
 * proposal.  Time and output cursors move only in Commit().
 */
class StepPlanCursor
{
public:
  static constexpr std::uint32_t state_schema_version = 1;

  StepPlanCursor(TimeControl control,
                 OutputSchedule schedule,
                 IntegratorTag integrator = IntegratorTag::BackendDefault)
    : control_(std::move(control)),
      schedule_(std::move(schedule)),
      integrator_(integrator),
      time_(control_.initial_time)
  {
    ValidateIntegratorTag(integrator_);
    control_.Validate();
    schedule_.Validate();
    CanonicalizeExplicitTimes();
    ValidateExplicitBounds();
    SkipInitialExplicitTimes();
  }

  const TimeControl& control() const { return control_; }
  const OutputSchedule& schedule() const { return schedule_; }
  IntegratorTag integrator() const { return integrator_; }
  double time() const { return time_; }
  std::uint64_t committed_steps() const { return committed_steps_; }
  bool Finished() const { return time_ == control_.final_time; }
  bool HasPendingPlan() const { return pending_.has_value(); }

  /** Capture the complete committed scheduling state. */
  StepPlanCursorState CheckpointState() const
  {
    if (pending_)
    {
      throw std::logic_error(
          "time cursor cannot checkpoint a pending plan");
    }
    return StepPlanCursorState{state_schema_version,
                               control_,
                               schedule_,
                               integrator_,
                               time_,
                               committed_steps_,
                               next_sequence_,
                               next_fixed_cycle_,
                               next_publication_cycle_,
                               static_cast<std::uint64_t>(
                                   next_explicit_index_)};
  }

  /**
   * Restore a matching committed scheduling state with a strong guarantee.
   *
   * All identity and cursor invariants are checked before any live member is
   * changed.  Restore never discards a pending plan.
   */
  void Restore(const StepPlanCursorState& state)
  {
    if (pending_)
    {
      throw std::logic_error(
          "time cursor cannot restore over a pending plan");
    }
    ValidateRestoredState(state);
    time_ = state.current_time;
    committed_steps_ = state.committed_steps;
    next_sequence_ = state.next_sequence;
    next_fixed_cycle_ = state.next_fixed_cycle;
    next_publication_cycle_ = state.next_publication_cycle;
    next_explicit_index_ =
        static_cast<std::size_t>(state.next_explicit_index);
  }

  /** The deterministic last output cycle implied by committed cursor state. */
  std::uint64_t LastCommittedOutputCycle() const
  {
    if (next_publication_cycle_ == 0)
    {
      throw std::logic_error("time cursor publication cycle is invalid");
    }
    if (committed_steps_ == 0)
    {
      return 0;
    }
    if (Finished() && schedule_.final == FinalOutputPolicy::Emit &&
        !FinalBoundaryWasScheduled())
    {
      return next_publication_cycle_;
    }
    return next_publication_cycle_ - 1;
  }

  /// Returns the transaction that Commit() may consume without advancing it.
  const StepPlan& PendingPlan() const
  {
    if (!pending_)
    {
      throw std::logic_error("no step plan is pending");
    }
    return *pending_;
  }

  /// Returns the cycle-zero boundary publication, if requested.  For a
  /// zero-duration run, initial/scheduled/final requests coalesce into one
  /// cycle-zero event with the corresponding flags.
  std::optional<OutputEvent> InitialOutput() const
  {
    const bool explicit_initial = HasExplicitInitialOutput();
    const bool emit_initial = schedule_.initial == InitialOutputPolicy::Emit ||
                              explicit_initial;
    const bool emit_final = Finished() &&
                            (schedule_.final == FinalOutputPolicy::Emit ||
                             explicit_initial);
    if (!emit_initial && !emit_final)
    {
      return std::nullopt;
    }
    return OutputEvent{0, control_.initial_time, emit_initial, emit_final};
  }

  const StepPlan& Plan(double backend_proposed_dt)
  {
    if (pending_)
    {
      return *pending_;
    }
    if (Finished())
    {
      throw std::logic_error("cannot plan after final time");
    }
    if (next_sequence_ == std::numeric_limits<std::uint64_t>::max())
    {
      throw std::overflow_error("time plan sequence overflow");
    }

    const double proposed_dt = control_.ResolveDt(backend_proposed_dt);
    const double final_boundary = control_.final_time;
    const std::optional<Boundary> output = NextOutputBoundary();
    const double remaining_dt = final_boundary - time_;
    double planned_dt = std::min(proposed_dt, remaining_dt);
    double end_time = planned_dt == remaining_dt ? final_boundary
                                                  : time_ + planned_dt;
    bool hit_output = false;
    std::uint64_t output_cycle = 0;
    if (output)
    {
      const double output_dt = output->time - time_;
      // A proposal can be one representable value below output_dt while its
      // rounded endpoint is exactly the output boundary.  Classify that as a
      // boundary hit without increasing the solver's proposed dt; otherwise
      // Commit() reaches the boundary without advancing the output cursor and
      // the following plan is forced to a zero-length step.
      const bool rounded_to_output = end_time == output->time;
      if (output_dt >= 0.0 &&
          (output_dt <= planned_dt || rounded_to_output))
      {
        if (output_dt <= planned_dt)
        {
          planned_dt = output_dt;
        }
        end_time = output->time;
        hit_output = true;
        output_cycle = output->cycle;
      }
    }
    const bool hit_final = end_time == final_boundary;
    if (!(end_time > time_))
    {
      throw std::logic_error("time plan did not make forward progress");
    }
    pending_ = StepPlan{next_sequence_++,
                        time_,
                        proposed_dt,
                        planned_dt,
                        end_time,
                        hit_output,
                        hit_final,
                        output_cycle};
    return *pending_;
  }

  /**
   * @brief Preview exactly the output event that Commit() would return.
   *
   * This is deliberately a validation-only operation: it leaves the time,
   * output, and pending-plan cursors unchanged.  Staging code can therefore
   * prepare an output transaction before accepting the associated solver
   * step, while a retry continues to observe the same plan and event.
   */
  std::optional<OutputEvent> PreviewCommit(const StepPlan& plan) const
  {
    if (!pending_ || !SamePlan(plan, *pending_))
    {
      throw std::logic_error("only the pending plan may be previewed");
    }
    return OutputForCommit(*pending_);
  }

  /**
   * Validate every fallible condition for a later Commit().
   *
   * A caller may use this immediately before committing an external solver
   * transaction.  If the cursor is not mutated in between, Commit() performs
   * only the already-validated, allocation-free state transition.
   */
  std::optional<OutputEvent> PreflightCommit(const StepPlan& plan) const
  {
    const std::optional<OutputEvent> output = PreviewCommit(plan);
    if (committed_steps_ == std::numeric_limits<std::uint64_t>::max())
    {
      throw std::overflow_error("time committed-step counter overflow");
    }
    if (pending_->hits_scheduled_output)
    {
      if (next_publication_cycle_ ==
          std::numeric_limits<std::uint64_t>::max())
      {
        throw std::overflow_error("time publication-cycle counter overflow");
      }
      if (schedule_.kind == OutputScheduleKind::FixedInterval &&
          next_fixed_cycle_ == std::numeric_limits<std::uint64_t>::max())
      {
        throw std::overflow_error("time fixed-output counter overflow");
      }
      if (schedule_.kind == OutputScheduleKind::ExplicitTimes &&
          next_explicit_index_ == std::numeric_limits<std::size_t>::max())
      {
        throw std::overflow_error("time explicit-output cursor overflow");
      }
    }
    return output;
  }

  std::optional<OutputEvent> Commit(const StepPlan& plan)
  {
    const std::optional<OutputEvent> output = PreflightCommit(plan);
    const StepPlan committed = *pending_;
    pending_.reset();
    time_ = committed.end_time;
    ++committed_steps_;
    if (committed.hits_scheduled_output)
    {
      AdvanceScheduledOutput();
    }

    return output;
  }

  /**
   * Reject the pending attempt without advancing time or output state.
   *
   * The next Plan() receives a new sequence and may use a smaller adaptive
   * proposal.  Fixed/CFL retry behavior remains unchanged unless a caller
   * explicitly invokes this operation.
   */
  void Abort(const StepPlan& plan)
  {
    if (!pending_ || !SamePlan(plan, *pending_))
    {
      throw std::logic_error("only the pending plan may be aborted once");
    }
    pending_.reset();
  }

private:
  struct Boundary
  {
    double time = 0.0;
    std::uint64_t cycle = 0;
  };

  static bool NearlyEqual(double lhs, double rhs)
  {
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <=
           64.0 * std::numeric_limits<double>::epsilon() * scale;
  }

  static bool SamePlan(const StepPlan& lhs, const StepPlan& rhs)
  {
    return lhs.sequence == rhs.sequence && lhs.start_time == rhs.start_time &&
           lhs.proposed_dt == rhs.proposed_dt && lhs.dt == rhs.dt &&
           lhs.end_time == rhs.end_time &&
           lhs.hits_scheduled_output == rhs.hits_scheduled_output &&
           lhs.hits_final == rhs.hits_final &&
           lhs.scheduled_cycle == rhs.scheduled_cycle;
  }

  static bool SameControl(const TimeControl& lhs, const TimeControl& rhs)
  {
    return lhs.initial_time == rhs.initial_time &&
           lhs.final_time == rhs.final_time && lhs.kind == rhs.kind &&
           lhs.fixed_dt == rhs.fixed_dt && lhs.cfl == rhs.cfl &&
           lhs.adaptive_initial_dt == rhs.adaptive_initial_dt;
  }

  static bool SameSchedule(const OutputSchedule& lhs,
                           const OutputSchedule& rhs)
  {
    return lhs.kind == rhs.kind && lhs.interval == rhs.interval &&
           lhs.explicit_times == rhs.explicit_times &&
           lhs.initial == rhs.initial && lhs.final == rhs.final;
  }

  double CanonicalFixedBoundary(const std::uint64_t cycle) const
  {
    const double target = control_.initial_time +
        static_cast<double>(cycle) * schedule_.interval;
    return std::isfinite(target) && NearlyEqual(target, control_.final_time)
        ? control_.final_time
        : target;
  }

  void ValidateScheduledState(const StepPlanCursorState& state) const
  {
    if (state.next_publication_cycle == 0 ||
        state.next_publication_cycle - 1 > state.committed_steps)
    {
      throw std::invalid_argument(
          "time cursor publication counters are inconsistent");
    }

    switch (schedule_.kind)
    {
      case OutputScheduleKind::Disabled:
        if (state.next_fixed_cycle != 1 ||
            state.next_publication_cycle != 1 ||
            state.next_explicit_index != 0)
        {
          throw std::invalid_argument(
              "disabled output cursor state is inconsistent");
        }
        return;

      case OutputScheduleKind::FixedInterval:
      {
        if (state.next_fixed_cycle == 0 ||
            state.next_fixed_cycle != state.next_publication_cycle ||
            state.next_explicit_index != 0)
        {
          throw std::invalid_argument(
              "fixed-interval output cursors are inconsistent");
        }
        if (state.next_fixed_cycle > 1)
        {
          const double previous =
              CanonicalFixedBoundary(state.next_fixed_cycle - 1);
          if (!std::isfinite(previous) ||
              (previous > control_.final_time &&
               !NearlyEqual(previous, control_.final_time)) ||
              previous > state.current_time)
          {
            throw std::invalid_argument(
                "fixed-interval output cursor skipped its prior boundary");
          }
        }
        const double next = CanonicalFixedBoundary(state.next_fixed_cycle);
        if (std::isfinite(next) &&
            !(next > control_.final_time &&
              !NearlyEqual(next, control_.final_time)) &&
            !(next > state.current_time))
        {
          throw std::invalid_argument(
              "fixed-interval output cursor did not advance past time");
        }
        return;
      }

      case OutputScheduleKind::ExplicitTimes:
      {
        if (state.next_fixed_cycle != 1 ||
            state.next_explicit_index >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()) ||
            state.next_explicit_index >
                static_cast<std::uint64_t>(schedule_.explicit_times.size()))
        {
          throw std::invalid_argument(
              "explicit output cursor index is invalid");
        }
        const std::uint64_t initial_skip =
            HasExplicitInitialOutput() ? 1U : 0U;
        if (state.next_explicit_index < initial_skip)
        {
          throw std::invalid_argument(
              "explicit output cursor precedes its initial boundary");
        }
        const std::uint64_t emitted =
            state.next_explicit_index - initial_skip;
        if (emitted != state.next_publication_cycle - 1)
        {
          throw std::invalid_argument(
              "explicit output cursors are inconsistent");
        }
        const std::size_t next_index =
            static_cast<std::size_t>(state.next_explicit_index);
        if (emitted != 0 &&
            schedule_.explicit_times[next_index - 1] > state.current_time)
        {
          throw std::invalid_argument(
              "explicit output cursor skipped its prior boundary");
        }
        if (next_index < schedule_.explicit_times.size() &&
            !(schedule_.explicit_times[next_index] > state.current_time))
        {
          throw std::invalid_argument(
              "explicit output cursor did not advance past time");
        }
        return;
      }
    }
    throw std::invalid_argument("unknown output schedule kind");
  }

  void ValidateRestoredState(const StepPlanCursorState& state) const
  {
    if (state.schema_version != state_schema_version)
    {
      throw std::invalid_argument(
          "time cursor state schema is unsupported");
    }
    state.control.Validate();
    state.output_schedule.Validate();
    ValidateIntegratorTag(state.integrator);
    if (!SameControl(state.control, control_) ||
        !SameSchedule(state.output_schedule, schedule_) ||
        state.integrator != integrator_)
    {
      throw std::invalid_argument(
          "time cursor state belongs to a different configuration");
    }
    if (!std::isfinite(state.current_time) ||
        state.current_time < control_.initial_time ||
        state.current_time > control_.final_time)
    {
      throw std::invalid_argument(
          "time cursor current time is outside its configured bounds");
    }
    const bool at_initial = state.current_time == control_.initial_time;
    if ((state.committed_steps == 0) != at_initial)
    {
      throw std::invalid_argument(
          "time cursor step count contradicts its current time");
    }
    if (state.next_sequence == 0 ||
        state.next_sequence - 1 < state.committed_steps ||
        (state.current_time != control_.final_time &&
         state.next_sequence ==
             std::numeric_limits<std::uint64_t>::max()))
    {
      throw std::invalid_argument(
          "time cursor sequence counter is inconsistent");
    }
    ValidateScheduledState(state);
  }

  void CanonicalizeExplicitTimes()
  {
    if (schedule_.kind != OutputScheduleKind::ExplicitTimes)
    {
      return;
    }
    std::vector<double> unique;
    unique.reserve(schedule_.explicit_times.size());
    for (double value : schedule_.explicit_times)
    {
      // Only tolerance-admitted values outside the interval normalize to a
      // bound.  In-range explicit values are caller-authored events and stay
      // distinct, even for spans smaller than the generic float tolerance.
      if (value < control_.initial_time &&
          NearlyEqual(value, control_.initial_time))
      {
        value = control_.initial_time;
      }
      else if (value > control_.final_time &&
               NearlyEqual(value, control_.final_time))
      {
        value = control_.final_time;
      }
      if (unique.empty() || unique.back() != value)
      {
        unique.push_back(value);
      }
    }
    schedule_.explicit_times = std::move(unique);
  }

  void ValidateExplicitBounds() const
  {
    if (schedule_.kind != OutputScheduleKind::ExplicitTimes)
    {
      return;
    }
    for (double value : schedule_.explicit_times)
    {
      if (value < control_.initial_time &&
          !NearlyEqual(value, control_.initial_time))
      {
        throw std::invalid_argument("explicit output precedes initial time");
      }
      if (value > control_.final_time && !NearlyEqual(value, control_.final_time))
      {
        throw std::invalid_argument("explicit output exceeds final time");
      }
    }
  }

  void SkipInitialExplicitTimes()
  {
    if (schedule_.kind == OutputScheduleKind::ExplicitTimes &&
        !schedule_.explicit_times.empty() &&
        schedule_.explicit_times.front() == control_.initial_time)
    {
      next_explicit_index_ = 1;
    }
  }

  bool FinalBoundaryWasScheduled() const
  {
    if (schedule_.kind == OutputScheduleKind::FixedInterval &&
        next_fixed_cycle_ > 1)
    {
      return CanonicalFixedBoundary(next_fixed_cycle_ - 1) ==
          control_.final_time;
    }
    if (schedule_.kind == OutputScheduleKind::ExplicitTimes &&
        next_explicit_index_ != 0)
    {
      return schedule_.explicit_times[next_explicit_index_ - 1] ==
          control_.final_time;
    }
    return false;
  }

  bool HasExplicitInitialOutput() const
  {
    return schedule_.kind == OutputScheduleKind::ExplicitTimes &&
           !schedule_.explicit_times.empty() &&
           schedule_.explicit_times.front() == control_.initial_time;
  }

  std::optional<Boundary> NextOutputBoundary() const
  {
    if (schedule_.kind == OutputScheduleKind::Disabled)
    {
      return std::nullopt;
    }
    if (schedule_.kind == OutputScheduleKind::FixedInterval)
    {
      const double target = control_.initial_time +
                            static_cast<double>(next_fixed_cycle_) *
                              schedule_.interval;
      if (!std::isfinite(target) ||
          (target > control_.final_time &&
           !NearlyEqual(target, control_.final_time)))
      {
        return std::nullopt;
      }
      return Boundary{NearlyEqual(target, control_.final_time)
                        ? control_.final_time
                        : target,
                      next_fixed_cycle_};
    }
    if (next_explicit_index_ >= schedule_.explicit_times.size())
    {
      return std::nullopt;
    }
    return Boundary{schedule_.explicit_times[next_explicit_index_],
                    next_publication_cycle_};
  }

  void AdvanceScheduledOutput()
  {
    if (schedule_.kind == OutputScheduleKind::FixedInterval)
    {
      ++next_fixed_cycle_;
    }
    else if (schedule_.kind == OutputScheduleKind::ExplicitTimes)
    {
      ++next_explicit_index_;
    }
    ++next_publication_cycle_;
  }

  std::uint64_t NextPolicyCycle() const
  {
    return next_publication_cycle_;
  }

  std::optional<OutputEvent> OutputForCommit(const StepPlan& committed) const
  {
    if (committed.hits_scheduled_output)
    {
      return OutputEvent{committed.scheduled_cycle,
                         committed.end_time,
                         false,
                         committed.hits_final};
    }
    if (committed.hits_final && schedule_.final == FinalOutputPolicy::Emit)
    {
      return OutputEvent{NextPolicyCycle(), committed.end_time, false, true};
    }
    return std::nullopt;
  }

  TimeControl control_;
  OutputSchedule schedule_;
  IntegratorTag integrator_;
  double time_;
  std::uint64_t committed_steps_ = 0;
  std::uint64_t next_sequence_ = 1;
  std::uint64_t next_fixed_cycle_ = 1;
  std::uint64_t next_publication_cycle_ = 1;
  std::size_t next_explicit_index_ = 0;
  std::optional<StepPlan> pending_;
};

} // namespace dg_sim::time::backend

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
  BackendDefault,
  SSPRK33,
  BackwardEuler,
  // Reserved tags: a later backend may implement them without changing this
  // scheduling contract.
  DIRK,
  IMEX,
  EmbeddedError
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
  }
  throw std::invalid_argument("unknown integrator tag");
}

enum class StepControlKind
{
  Fixed,
  CFL
};

/**
 * @brief Fixed/CFL step control, intentionally independent of IntegratorTag.
 *
 * In fixed mode the backend proposal passed to StepPlanCursor::Plan is
 * ignored.  In CFL mode that proposal is the backend's already-reduced CFL
 * candidate and is validated here before clipping.
 */
struct TimeControl
{
  double initial_time = 0.0;
  double final_time = 0.0;
  StepControlKind kind = StepControlKind::Fixed;
  double fixed_dt = 0.0;
  double cfl = 0.0;

  static TimeControl Fixed(double initial_time,
                           double final_time,
                           double fixed_dt)
  {
    return {initial_time, final_time, StepControlKind::Fixed, fixed_dt, 0.0};
  }

  static TimeControl CFL(double initial_time, double final_time, double cfl)
  {
    return {initial_time, final_time, StepControlKind::CFL, 0.0, cfl};
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
      throw std::invalid_argument(
        "backend CFL proposal must be finite and positive");
    }
    return backend_proposed_dt;
  }
};

enum class OutputScheduleKind
{
  Disabled,
  FixedInterval,
  ExplicitTimes
};

enum class InitialOutputPolicy
{
  Suppress,
  Emit
};

enum class FinalOutputPolicy
{
  Suppress,
  Emit
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
 * @brief Nonvirtual commit cursor for deterministic output-aligned stepping.
 *
 * Plan() caches its result.  Retrying an attempted backend step therefore
 * returns the exact same values even if a caller presents a different CFL
 * proposal.  Time and output cursors move only in Commit().
 */
class StepPlanCursor
{
public:
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
      if (output_dt >= 0.0 && output_dt <= planned_dt)
      {
        planned_dt = output_dt;
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

  std::optional<OutputEvent> Commit(const StepPlan& plan)
  {
    if (!pending_ || !SamePlan(plan, *pending_))
    {
      throw std::logic_error("only the pending plan may be committed once");
    }

    const StepPlan committed = *pending_;
    pending_.reset();
    time_ = committed.end_time;
    ++committed_steps_;
    if (committed.hits_scheduled_output)
    {
      AdvanceScheduledOutput();
    }

    if (committed.hits_scheduled_output)
    {
      return OutputEvent{committed.scheduled_cycle,
                         time_,
                         false,
                         committed.hits_final};
    }
    if (committed.hits_final && schedule_.final == FinalOutputPolicy::Emit)
    {
      return OutputEvent{NextPolicyCycle(), time_, false, true};
    }
    return std::nullopt;
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
      // Bounds admitted by ValidateExplicitBounds() through NearlyEqual() must
      // become exact cursor boundaries before deduplication.  Otherwise a
      // just-outside endpoint can pass validation yet be missed by clipping.
      if (NearlyEqual(value, control_.initial_time))
      {
        value = control_.initial_time;
      }
      else if (NearlyEqual(value, control_.final_time))
      {
        value = control_.final_time;
      }
      if (unique.empty() || !NearlyEqual(unique.back(), value))
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
        NearlyEqual(schedule_.explicit_times.front(), control_.initial_time))
    {
      next_explicit_index_ = 1;
    }
  }

  bool HasExplicitInitialOutput() const
  {
    return schedule_.kind == OutputScheduleKind::ExplicitTimes &&
           !schedule_.explicit_times.empty() &&
           NearlyEqual(schedule_.explicit_times.front(), control_.initial_time);
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
      if (target > control_.final_time && !NearlyEqual(target, control_.final_time))
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

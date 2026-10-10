#pragma once

/**
 * @file sim/time/backend/adaptive_time_control.hpp
 * @brief Pure embedded-error timestep policy and restart state.
 *
 * The caller supplies communicator-reduced error and acceptance gates.  This
 * policy deliberately performs no MPI calls and owns no numerical or output
 * state.  PreviewDecision() is pure; CommitDecision() is the only operation
 * that advances controller state.
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace dg_sim::time::backend
{

/** Bounded I-controller parameters for an embedded local-error estimate. */
struct EmbeddedErrorControllerConfig
{
  double minimum_dt = 1.0e-12;
  double maximum_dt = std::numeric_limits<double>::max();
  double safety_factor = 0.9;
  double minimum_scale = 0.2;
  double maximum_rejection_scale = 0.5;
  double maximum_growth_scale = 5.0;
  std::uint32_t estimator_order = 3;

  void Validate() const
  {
    if (!std::isfinite(minimum_dt) || minimum_dt <= 0.0 ||
        !std::isfinite(maximum_dt) || maximum_dt < minimum_dt)
    {
      throw std::invalid_argument(
          "adaptive dt bounds must be finite, positive, and ordered");
    }
    if (!std::isfinite(safety_factor) || safety_factor <= 0.0 ||
        safety_factor > 1.0)
    {
      throw std::invalid_argument(
          "adaptive safety factor must be finite and in (0, 1]");
    }
    if (!std::isfinite(minimum_scale) || minimum_scale <= 0.0 ||
        minimum_scale >= 1.0)
    {
      throw std::invalid_argument(
          "adaptive minimum scale must be finite and in (0, 1)");
    }
    if (!std::isfinite(maximum_rejection_scale) ||
        maximum_rejection_scale < minimum_scale ||
        maximum_rejection_scale >= 1.0)
    {
      throw std::invalid_argument(
          "adaptive rejection scale must be finite, bounded, and shrinking");
    }
    if (!std::isfinite(maximum_growth_scale) ||
        maximum_growth_scale < 1.0)
    {
      throw std::invalid_argument(
          "adaptive growth scale must be finite and at least one");
    }
    if (estimator_order == 0)
    {
      throw std::invalid_argument(
          "adaptive estimator order must be positive");
    }
  }
};

/**
 * Collective inputs for one attempted embedded step.
 *
 * collective_normalized_error is the already-reduced differential-state error
 * norm, scaled so one is the acceptance threshold.  Every boolean must also
 * be the communicator-wide decision produced by the caller.
 */
struct EmbeddedErrorAttempt
{
  double collective_normalized_error =
      std::numeric_limits<double>::infinity();
  bool collective_solver_success = false;
  bool collective_residuals_satisfied = false;
  bool collective_admissible = false;
  bool collective_algebraic_closure_satisfied = false;
};

enum class AdaptiveStepDisposition : std::uint8_t
{
  accepted = 0,
  rejected = 1
};

class EmbeddedErrorTimeController;

/** Immutable, controller-bound preview of one controller transition. */
class AdaptiveStepDecision
{
public:
  AdaptiveStepDecision(const AdaptiveStepDecision&) = default;
  AdaptiveStepDecision(AdaptiveStepDecision&&) = default;
  AdaptiveStepDecision& operator=(const AdaptiveStepDecision&) = delete;
  AdaptiveStepDecision& operator=(AdaptiveStepDecision&&) = delete;

  bool Accepted() const noexcept
  {
    return disposition_ == AdaptiveStepDisposition::accepted;
  }

  std::uint64_t source_revision() const noexcept { return source_revision_; }
  AdaptiveStepDisposition disposition() const noexcept { return disposition_; }
  double attempted_dt() const noexcept { return attempted_dt_; }
  double normalized_error() const noexcept { return normalized_error_; }
  double scale() const noexcept { return scale_; }
  double next_dt() const noexcept { return next_dt_; }
  bool error_is_valid() const noexcept { return error_is_valid_; }
  bool collective_gates_satisfied() const noexcept
  {
    return collective_gates_satisfied_;
  }
  bool minimum_dt_reached() const noexcept { return minimum_dt_reached_; }

private:
  friend class EmbeddedErrorTimeController;

  AdaptiveStepDecision(
      const std::uint64_t controller_authority,
      const std::uint64_t source_revision,
      const AdaptiveStepDisposition disposition,
      const double attempted_dt,
      const double normalized_error,
      const double scale,
      const double next_dt,
      const bool error_is_valid,
      const bool collective_gates_satisfied,
      const bool minimum_dt_reached) noexcept
    : controller_authority_(controller_authority),
      source_revision_(source_revision),
      disposition_(disposition),
      attempted_dt_(attempted_dt),
      normalized_error_(normalized_error),
      scale_(scale),
      next_dt_(next_dt),
      error_is_valid_(error_is_valid),
      collective_gates_satisfied_(collective_gates_satisfied),
      minimum_dt_reached_(minimum_dt_reached)
  {}

  std::uint64_t controller_authority_ = 0;
  std::uint64_t source_revision_ = 0;
  AdaptiveStepDisposition disposition_ = AdaptiveStepDisposition::rejected;
  double attempted_dt_ = 0.0;
  double normalized_error_ = 0.0;
  double scale_ = 0.0;
  double next_dt_ = 0.0;
  bool error_is_valid_ = false;
  bool collective_gates_satisfied_ = false;
  bool minimum_dt_reached_ = false;
};

/**
 * Fieldwise-persistable adaptive state.
 *
 * Checkpoints should serialize these named fields rather than object padding.
 * schema_version rejects incompatible layouts during restore.
 */
struct EmbeddedErrorControllerState
{
  std::uint32_t schema_version;
  std::uint32_t configured_estimator_order;
  double configured_minimum_dt;
  double configured_maximum_dt;
  double configured_safety_factor;
  double configured_minimum_scale;
  double configured_maximum_rejection_scale;
  double configured_maximum_growth_scale;
  std::uint32_t consecutive_rejections;
  std::uint64_t revision;
  std::uint64_t accepted_steps;
  std::uint64_t rejected_attempts;
  double next_dt;
  double last_attempted_dt;
  double last_normalized_error;
  std::uint8_t last_step_accepted;
};

static_assert(std::is_trivial<EmbeddedErrorControllerState>::value,
              "adaptive controller state must remain trivial");
static_assert(std::is_standard_layout<EmbeddedErrorControllerState>::value,
              "adaptive controller state must remain standard-layout");

class EmbeddedErrorTimeController
{
public:
  static constexpr std::uint32_t state_schema_version = 2;

  EmbeddedErrorTimeController(EmbeddedErrorControllerConfig config,
                              const double initial_dt)
    : config_(config), state_(InitialState(initial_dt)),
      controller_authority_(AcquireControllerAuthority())
  {
    config_.Validate();
  }

  EmbeddedErrorTimeController(EmbeddedErrorControllerConfig config,
                              EmbeddedErrorControllerState restored_state)
    : config_(config), state_(restored_state),
      controller_authority_(AcquireControllerAuthority())
  {
    config_.Validate();
    ValidateState(state_);
  }

  EmbeddedErrorTimeController(const EmbeddedErrorTimeController&) = delete;
  EmbeddedErrorTimeController& operator=(
      const EmbeddedErrorTimeController&) = delete;
  EmbeddedErrorTimeController(EmbeddedErrorTimeController&&) = delete;
  EmbeddedErrorTimeController& operator=(
      EmbeddedErrorTimeController&&) = delete;

  const EmbeddedErrorControllerConfig& config() const noexcept
  {
    return config_;
  }

  const EmbeddedErrorControllerState& state() const noexcept
  {
    return state_;
  }

  double proposed_dt() const noexcept { return state_.next_dt; }

  /** Compute a transition without changing the restart state. */
  AdaptiveStepDecision PreviewDecision(
      const double attempted_dt,
      const EmbeddedErrorAttempt& attempt) const
  {
    if (!std::isfinite(attempted_dt) || attempted_dt <= 0.0)
    {
      throw std::invalid_argument(
          "adaptive attempted dt must be finite and positive");
    }

    const bool error_is_valid =
        std::isfinite(attempt.collective_normalized_error) &&
        attempt.collective_normalized_error >= 0.0;
    const bool gates_satisfied = attempt.collective_solver_success &&
        attempt.collective_residuals_satisfied &&
        attempt.collective_admissible &&
        attempt.collective_algebraic_closure_satisfied;
    const bool accepted = error_is_valid && gates_satisfied &&
        attempt.collective_normalized_error <= 1.0;
    const double normalized_error = error_is_valid
        ? attempt.collective_normalized_error
        : std::numeric_limits<double>::max();
    const double raw_scale = RawScale(normalized_error, error_is_valid);
    const double scale = accepted
        ? std::clamp(raw_scale,
                     config_.minimum_scale,
                     config_.maximum_growth_scale)
        : std::clamp(raw_scale,
                     config_.minimum_scale,
                     config_.maximum_rejection_scale);
    const double next_dt = ScaledDt(attempted_dt, scale);
    return AdaptiveStepDecision{
        controller_authority_,
        state_.revision,
        accepted ? AdaptiveStepDisposition::accepted
                 : AdaptiveStepDisposition::rejected,
        attempted_dt,
        normalized_error,
        scale,
        next_dt,
        error_is_valid,
        gates_satisfied,
        !accepted && next_dt == config_.minimum_dt};
  }

  /** Commit exactly one previously previewed transition. */
  void PreflightCommitDecision(const AdaptiveStepDecision& decision) const
  {
    ValidateDecision(decision);
    if (state_.revision == std::numeric_limits<std::uint64_t>::max())
    {
      throw std::overflow_error("adaptive controller revision overflow");
    }
    if (decision.Accepted())
    {
      if (state_.accepted_steps == std::numeric_limits<std::uint64_t>::max())
      {
        throw std::overflow_error(
            "adaptive accepted-step counter overflow");
      }
    }
    else
    {
      if (state_.rejected_attempts ==
          std::numeric_limits<std::uint64_t>::max())
      {
        throw std::overflow_error(
          "adaptive rejected-attempt counter overflow");
      }
      if (state_.consecutive_rejections ==
          std::numeric_limits<std::uint32_t>::max())
      {
        throw std::overflow_error(
          "adaptive consecutive-rejection counter overflow");
      }
    }
  }

  /** Commit exactly one previously previewed transition. */
  void CommitDecision(const AdaptiveStepDecision& decision)
  {
    PreflightCommitDecision(decision);

    ++state_.revision;
    state_.next_dt = decision.next_dt();
    state_.last_attempted_dt = decision.attempted_dt();
    state_.last_normalized_error = decision.normalized_error();
    state_.last_step_accepted = decision.Accepted() ? 1U : 0U;
    if (decision.Accepted())
    {
      ++state_.accepted_steps;
      state_.consecutive_rejections = 0;
    }
    else
    {
      ++state_.rejected_attempts;
      ++state_.consecutive_rejections;
    }
  }

private:
  EmbeddedErrorControllerState InitialState(const double initial_dt) const
  {
    config_.Validate();
    if (!std::isfinite(initial_dt) || initial_dt <= 0.0)
    {
      throw std::invalid_argument(
          "adaptive initial dt must be finite and positive");
    }
    return EmbeddedErrorControllerState{
        state_schema_version,
        config_.estimator_order,
        config_.minimum_dt,
        config_.maximum_dt,
        config_.safety_factor,
        config_.minimum_scale,
        config_.maximum_rejection_scale,
        config_.maximum_growth_scale,
        0,
        0,
        0,
        0,
        std::clamp(initial_dt, config_.minimum_dt, config_.maximum_dt),
        0.0,
        0.0,
        0U};
  }

  void ValidateState(const EmbeddedErrorControllerState& state) const
  {
    if (state.schema_version != state_schema_version)
    {
      throw std::invalid_argument(
          "adaptive controller state schema is unsupported");
    }
    if (state.configured_estimator_order != config_.estimator_order ||
        state.configured_minimum_dt != config_.minimum_dt ||
        state.configured_maximum_dt != config_.maximum_dt ||
        state.configured_safety_factor != config_.safety_factor ||
        state.configured_minimum_scale != config_.minimum_scale ||
        state.configured_maximum_rejection_scale !=
            config_.maximum_rejection_scale ||
        state.configured_maximum_growth_scale !=
            config_.maximum_growth_scale)
    {
      throw std::invalid_argument(
          "adaptive controller state belongs to a different configuration");
    }
    if (state.accepted_steps >
          std::numeric_limits<std::uint64_t>::max() -
            state.rejected_attempts ||
        state.revision != state.accepted_steps + state.rejected_attempts ||
        state.consecutive_rejections > state.rejected_attempts)
    {
      throw std::invalid_argument(
          "adaptive controller counters are inconsistent");
    }
    if (!std::isfinite(state.next_dt) ||
        state.next_dt < config_.minimum_dt ||
        state.next_dt > config_.maximum_dt)
    {
      throw std::invalid_argument(
          "adaptive controller next dt is outside its configured bounds");
    }
    if (state.last_step_accepted > 1U ||
        !std::isfinite(state.last_attempted_dt) ||
        state.last_attempted_dt < 0.0 ||
        !std::isfinite(state.last_normalized_error) ||
        state.last_normalized_error < 0.0)
    {
      throw std::invalid_argument(
          "adaptive controller last-attempt state is invalid");
    }
    if ((state.revision == 0 &&
         (state.last_attempted_dt != 0.0 ||
          state.last_normalized_error != 0.0 ||
          state.last_step_accepted != 0U)) ||
        (state.revision != 0 && state.last_attempted_dt == 0.0))
    {
      throw std::invalid_argument(
          "adaptive controller history is inconsistent");
    }
    if (state.revision != 0 &&
        ((state.last_step_accepted == 1U &&
          (state.accepted_steps == 0 || state.consecutive_rejections != 0)) ||
         (state.last_step_accepted == 0U &&
          (state.rejected_attempts == 0 ||
           state.consecutive_rejections == 0))))
    {
      throw std::invalid_argument(
          "adaptive controller last disposition contradicts its counters");
    }
    if (state.revision != 0)
    {
      const bool accepted = state.last_step_accepted == 1U;
      if (accepted && state.last_normalized_error > 1.0)
      {
        throw std::invalid_argument(
            "adaptive controller accepted history exceeds its error gate");
      }
      const double raw_scale = RawScale(
          state.last_normalized_error, true);
      const double expected_scale = accepted
          ? std::clamp(raw_scale,
                       config_.minimum_scale,
                       config_.maximum_growth_scale)
          : std::clamp(raw_scale,
                       config_.minimum_scale,
                       config_.maximum_rejection_scale);
      if (state.next_dt != ScaledDt(
              state.last_attempted_dt, expected_scale))
      {
        throw std::invalid_argument(
            "adaptive controller restored proposal is inconsistent with its history");
      }
    }
  }

  double RawScale(const double normalized_error,
                  const bool error_is_valid) const
  {
    if (!error_is_valid)
    {
      return config_.minimum_scale;
    }
    if (normalized_error == 0.0)
    {
      return config_.maximum_growth_scale;
    }
    return config_.safety_factor *
        std::pow(normalized_error,
                 -1.0 / static_cast<double>(config_.estimator_order));
  }

  double ScaledDt(const double attempted_dt, const double scale) const
  {
    const double scaled = attempted_dt > config_.maximum_dt / scale
        ? config_.maximum_dt
        : attempted_dt * scale;
    return std::clamp(scaled, config_.minimum_dt, config_.maximum_dt);
  }

  void ValidateDecision(const AdaptiveStepDecision& decision) const
  {
    if (decision.controller_authority_ != controller_authority_)
    {
      throw std::logic_error(
          "adaptive controller decision belongs to another controller");
    }
    if (decision.source_revision() != state_.revision)
    {
      throw std::logic_error("adaptive controller decision is stale");
    }
    if (decision.disposition() != AdaptiveStepDisposition::accepted &&
        decision.disposition() != AdaptiveStepDisposition::rejected)
    {
      throw std::invalid_argument(
          "adaptive controller decision disposition is unknown");
    }
    if (!std::isfinite(decision.attempted_dt()) ||
        decision.attempted_dt() <= 0.0 ||
        !std::isfinite(decision.normalized_error()) ||
        decision.normalized_error() < 0.0 ||
        !std::isfinite(decision.scale()) || decision.scale() <= 0.0 ||
        !std::isfinite(decision.next_dt()) ||
        decision.next_dt() < config_.minimum_dt ||
        decision.next_dt() > config_.maximum_dt)
    {
      throw std::invalid_argument(
          "adaptive controller decision is numerically invalid");
    }
    const bool expected_acceptance = decision.error_is_valid() &&
        decision.collective_gates_satisfied() &&
        decision.normalized_error() <= 1.0;
    if (decision.Accepted() != expected_acceptance)
    {
      throw std::invalid_argument(
          "adaptive controller disposition violates its gates");
    }
    const double expected_raw_scale = RawScale(
        decision.normalized_error(), decision.error_is_valid());
    const double expected_scale = decision.Accepted()
        ? std::clamp(expected_raw_scale,
                     config_.minimum_scale,
                     config_.maximum_growth_scale)
        : std::clamp(expected_raw_scale,
                     config_.minimum_scale,
                     config_.maximum_rejection_scale);
    const double expected_next_dt = ScaledDt(
        decision.attempted_dt(), expected_scale);
    if (decision.scale() != expected_scale ||
        decision.next_dt() != expected_next_dt)
    {
      throw std::invalid_argument(
          "adaptive controller decision has invalid derived values");
    }
    if (!decision.Accepted() && decision.minimum_dt_reached() !=
        (decision.next_dt() == config_.minimum_dt))
    {
      throw std::invalid_argument(
          "adaptive controller minimum-dt marker is inconsistent");
    }
  }

  static std::uint64_t AcquireControllerAuthority()
  {
    std::uint64_t authority =
        next_controller_authority_.load(std::memory_order_relaxed);
    for (;;)
    {
      if (authority == 0 ||
          authority == std::numeric_limits<std::uint64_t>::max())
      {
        throw std::overflow_error("adaptive controller authority overflow");
      }
      if (next_controller_authority_.compare_exchange_weak(
              authority, authority + 1,
              std::memory_order_relaxed, std::memory_order_relaxed))
      {
        return authority;
      }
    }
  }

  EmbeddedErrorControllerConfig config_;
  EmbeddedErrorControllerState state_;
  std::uint64_t controller_authority_ = 0;
  inline static std::atomic<std::uint64_t> next_controller_authority_{1};
};

} // namespace dg_sim::time::backend

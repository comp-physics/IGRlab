#pragma once

#include <cstdint>
#include <stdexcept>

/**
 * @file numerics/timestepping/stage_aware_operator.hpp
 * @brief Optional interface for operators that need RK stage notifications.
 * @ingroup numerics_timestepping
 */

#include "mfem.hpp"

namespace mfem
{

/// Bounded reason code for a recoverable RK-stage admission failure.
enum class RKStageAdmissionFailureReason
{
  QSigmaNonpositive,
  QSigmaNonfinite,
  QSigmaNonpositiveAndNonfinite
};

inline const char* RKStageAdmissionFailureReasonName(
  RKStageAdmissionFailureReason reason)
{
  switch (reason)
  {
  case RKStageAdmissionFailureReason::QSigmaNonpositive:
    return "q-sigma-nonpositive";
  case RKStageAdmissionFailureReason::QSigmaNonfinite:
    return "q-sigma-nonfinite";
  case RKStageAdmissionFailureReason::QSigmaNonpositiveAndNonfinite:
    return "q-sigma-nonpositive-and-nonfinite";
  }
  return "unknown-stage-admission-failure";
}

/** Recoverable failure raised after an RHS violates its stage admission set.
 *
 * This is deliberately separate from limiter failure: no conservative or
 * algebraic field has been clamped.  The outer time-step transaction restores
 * the accepted state and asks the stage-aware operator to restore its
 * algebraic history before retrying with a smaller timestep.
 */
class RecoverableRKStageAdmissionFailure : public std::runtime_error
{
public:
  RecoverableRKStageAdmissionFailure(
    RKStageAdmissionFailureReason reason,
    std::uint64_t global_nonpositive_q_sigma_samples,
    std::uint64_t global_nonfinite_q_sigma_samples,
    int stage)
    : std::runtime_error("Recoverable RK-stage q_sigma admission failure."),
      reason_(reason),
      global_nonpositive_q_sigma_samples_(
        global_nonpositive_q_sigma_samples),
      global_nonfinite_q_sigma_samples_(global_nonfinite_q_sigma_samples),
      stage_(stage)
  {
  }

  RKStageAdmissionFailureReason Reason() const { return reason_; }
  const char* ReasonName() const
  {
    return RKStageAdmissionFailureReasonName(reason_);
  }
  std::uint64_t GlobalNonpositiveQSigmaSampleCount() const
  {
    return global_nonpositive_q_sigma_samples_;
  }
  std::uint64_t GlobalNonfiniteQSigmaSampleCount() const
  {
    return global_nonfinite_q_sigma_samples_;
  }
  int Stage() const { return stage_; }

private:
  RKStageAdmissionFailureReason reason_;
  std::uint64_t global_nonpositive_q_sigma_samples_ = 0;
  std::uint64_t global_nonfinite_q_sigma_samples_ = 0;
  int stage_ = -1;
};

/// @brief Optional interface for receiving SSPRK stage/step notifications.
class RKStageAwareOperator
{
public:
  virtual ~RKStageAwareOperator() = default;

  /// Whether this operator can request recoverable rejection during an RHS.
  virtual bool RequiresRKStepTransaction() const { return false; }

  /// Called at the beginning of each attempted RK time step.
  virtual void BeginRKStep(int step, real_t t, real_t dt) = 0;

  /// Called with the total number of RHS stages for the selected method.
  ///
  /// The compatibility implementation forwards to the historical three-
  /// argument callback. Stage-aware operators that distinguish an
  /// intermediate stage from the final RHS stage should override this form.
  virtual void BeginRKStep(int step,
                           real_t t,
                           real_t dt,
                           int stage_count)
  {
    BeginRKStep(step, t, dt);
    (void)stage_count;
  }

  /// Called before each RHS evaluation with index [0, stage_count).
  virtual void SetRKStage(int stage) = 0;

  /// Called before each RHS evaluation with the exact state generation.
  virtual void SetRKStageState(int stage, std::uint64_t state_generation)
  {
    SetRKStage(stage);
    (void)state_generation;
  }

  /// Called before each RHS evaluation with state provenance and stage time.
  ///
  /// The compatibility implementation preserves existing stage-aware
  /// operators. Methods that extrapolate an algebraic field in physical time
  /// override this form so non-monotone RK abscissae remain unambiguous.
  virtual void SetRKStageState(int stage,
                               std::uint64_t state_generation,
                               real_t stage_time)
  {
    SetRKStageState(stage, state_generation);
    (void)stage_time;
  }

  /// Called after the final limiter has produced the accepted step state.
  virtual void EndRKStep(const Vector& state,
                         real_t t,
                         std::uint64_t state_generation)
  {
    (void)state;
    (void)t;
    (void)state_generation;
  }

  /// Called after an outer transaction rejects the current RK attempt.
  ///
  /// The conservative state, time, and generation have already been restored
  /// by the caller. Stage-aware algebraic operators use this notification to
  /// restore cadence bookkeeping and prepare any state needed by a retry.
  virtual void AbortRKStep(const Vector& restored_state,
                           real_t restored_time,
                           std::uint64_t restored_state_generation,
                           real_t retry_dt)
  {
    (void)restored_state;
    (void)restored_time;
    (void)restored_state_generation;
    (void)retry_dt;
  }
};

} // namespace mfem

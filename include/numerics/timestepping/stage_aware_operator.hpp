#pragma once

#include <cstdint>

/**
 * @file numerics/timestepping/stage_aware_operator.hpp
 * @brief Optional interface for operators that need RK stage notifications.
 * @ingroup numerics_timestepping
 */

#include "mfem.hpp"

namespace mfem
{

/// @brief Optional interface for receiving SSPRK stage/step notifications.
class RKStageAwareOperator
{
public:
  virtual ~RKStageAwareOperator() = default;

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

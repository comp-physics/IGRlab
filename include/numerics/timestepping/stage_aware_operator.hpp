#pragma once

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

  /// Called at the beginning of each accepted RK time step.
  virtual void BeginRKStep(int step, real_t t, real_t dt) = 0;

  /// Called before each RHS evaluation with stage index [0, 1, 2].
  virtual void SetRKStage(int stage) = 0;
};

} // namespace mfem

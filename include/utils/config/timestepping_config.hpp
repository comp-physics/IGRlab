#pragma once

#include "igrlab/compat.hpp"

#include "utils/config/limiter_config.hpp"
#include "utils/config/dg_igr_defaults.hpp"

/**
 * @file utils/config/timestepping_config.hpp
 * @brief Configuration defaults for explicit time integration.
 * @ingroup utils_config
 */

namespace mfem
{

/// @brief Configuration bundle for RK3SSPWithLimiter.
/// @ingroup utils_config
struct RK3Config : public LimiterConfig
{
  bool limit_initial = igrlab::RK3Defaults::kLimitInitial;
  long max_limiter_steps = igrlab::RK3Defaults::kLimiterSteps;
};

} // namespace mfem

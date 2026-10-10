#pragma once

#include "igrlab/compat.hpp"

#include "utils/config/dg_igr_defaults.hpp"

/**
 * @file utils/config/limiter_config.hpp
 * @brief Common limiter configuration defaults.
 * @ingroup utils_config
 */

namespace mfem
{

/// @brief Shared limiter parameters (Euler + positivity settings).
/// @ingroup utils_config
struct LimiterConfig
{
  int dim = igrlab::EulerDefaults::kDim;        ///< Spatial dimension
  double gamma = igrlab::EulerDefaults::kGamma; ///< Ratio of specific heats
  double rho_min = igrlab::LimiterDefaults::kRhoMin;
  double p_min = igrlab::LimiterDefaults::kPMin;
  int quad_order = igrlab::LimiterDefaults::kQuadOrder;
  int p_bisect_iters = igrlab::LimiterDefaults::kBisectIters;
};

} // namespace mfem

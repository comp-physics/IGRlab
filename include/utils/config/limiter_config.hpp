#pragma once

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
  int    dim            = dg_igr::EulerDefaults::kDim;    ///< Spatial dimension
  double gamma          = dg_igr::EulerDefaults::kGamma;  ///< Ratio of specific heats
  double rho_min        = dg_igr::LimiterDefaults::kRhoMin;
  double p_min          = dg_igr::LimiterDefaults::kPMin;
  int    quad_order     = dg_igr::LimiterDefaults::kQuadOrder;
  int    p_bisect_iters = dg_igr::LimiterDefaults::kBisectIters;
};

} // namespace mfem

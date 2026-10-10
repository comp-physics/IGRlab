#pragma once

#include "igrlab/compat.hpp"

/**
 * @file utils/config/dg_igr_defaults.hpp
 * @brief Named defaults and config scaffolding for the dg_igr miniapp.
 * @ingroup utils_config
 */

namespace mfem
{
namespace igrlab
{

struct EulerDefaults
{
  static constexpr int kDim = 2;
  static constexpr double kGamma = 1.4;
};

struct IGRDefaults
{
  static constexpr double kAlpha = 0.0;
  static constexpr double kRhoFloor = 5.0e-7;
  static constexpr int kRegularSweeps = 3;
  static constexpr int kWarmStartSweeps = 5;
  static constexpr double kLinRtol = 1.0e-8;
  static constexpr double kLinAtol = 0.0;
  static constexpr int kLinMaxIters = 200;
  static constexpr int kLinPrint = 0;
};

struct FluxDefaults
{
  static constexpr double kRhoVelFloor = 1.0e-10;
};

struct LimiterDefaults
{
  static constexpr double kRhoMin = 1.0e-12;
  static constexpr double kPMin = 1.0e-12;
  static constexpr int kQuadOrder = 0;
  static constexpr int kBisectIters = 12;
};

struct RK3Defaults
{
  static constexpr bool kLimitInitial = false;
  static constexpr long kLimiterSteps = -1;
};

} // namespace igrlab
} // namespace mfem

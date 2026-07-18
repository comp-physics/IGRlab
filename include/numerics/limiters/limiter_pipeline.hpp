#pragma once

#include "mfem.hpp"
#include "numerics/limiters/bj.hpp"
#include "numerics/limiters/krivodonova.hpp"
#include "numerics/limiters/tvb.hpp"
#include "numerics/limiters/venkat.hpp"
#include "numerics/limiters/zhang_shu.hpp"

/**
 * @file numerics/limiters/limiter_pipeline.hpp
 * @brief Shared limiter pipeline configuration helpers.
 * @ingroup numerics_limiters
 */

namespace mfem
{

class RK3SSPWithLimiter;

/// @brief Barth-Jespersen limiter variants for pipeline configuration.
/// @ingroup numerics_limiters
enum class BJLimiterMode
{
  kNone,
  kComponent,
  kCharacteristic,
  kFaceNormalCharacteristic
};

/// @brief Characteristic TVB limiter variants for pipeline configuration.
/// @ingroup numerics_limiters
enum class TVBLimiterMode
{
  kNone,
  kCharacteristic,
  kFaceNormal
};

/// @brief Venkatakrishnan limiter variants for pipeline configuration.
/// @ingroup numerics_limiters
enum class VenkatLimiterMode
{
  kNone,
  kXY,
  kFaceNormal
};

/// @brief Configuration for composing limiter pipelines in RK3.
/// @ingroup numerics_limiters
struct LimiterPipelineConfig
{
  BJLimiterMode bj_mode = BJLimiterMode::kNone;
  BJOptions bj_opts;
  CharBJOptions char_bj_opts;
  FaceBJOptions face_bj_opts;

  TVBLimiterMode tvb_mode = TVBLimiterMode::kNone;
  CharTVBOptions tvb_opts;

  VenkatLimiterMode venkat_mode = VenkatLimiterMode::kNone;
  VenkatOptions venkat_opts;

  bool krivodonova = false;
  KrivoOptions krivo_opts;

  bool zhang_shu = false;
  ZhangShuPointSet zhang_shu_points = ZhangShuPointSet::kConsumer;
};

/**
 * @brief Configure an RK3 limiter pipeline from the provided config.
 *
 * The order of application is BJ (if enabled), TVB (if enabled), Venkat,
 * Krivodonova, then Zhang-Shu positivity.
 *
 * @param[in,out] rk3      RK3 stepper to configure.
 * @param[in]     config   Pipeline configuration.
 * @param[in]     vfes     DG finite element space (used for order defaults).
 * @param[in]     dim      Spatial dimension.
 * @param[in]     gamma    Specific heat ratio for characteristic limiters.
 */
void ConfigureLimiterPipeline(RK3SSPWithLimiter& rk3,
                              const LimiterPipelineConfig& config,
                              const ParFiniteElementSpace& vfes,
                              int dim,
                              double gamma);

} // namespace mfem

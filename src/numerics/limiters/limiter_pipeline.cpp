#include "numerics/limiters/limiter_pipeline.hpp"

#include "numerics/timestepping/RK3SSPWithLimiter.hpp"

namespace mfem
{

namespace
{
int DefaultSlopeQord(const ParFiniteElementSpace& vfes)
{
  return 2 * vfes.GetFE(0)->GetOrder() + 2;
}

int DefaultFaceQord(const ParFiniteElementSpace& vfes)
{
  return vfes.GetFE(0)->GetOrder() + 2;
}
} // namespace

void ConfigureLimiterPipeline(RK3SSPWithLimiter& rk3,
                              const LimiterPipelineConfig& config,
                              const ParFiniteElementSpace& vfes,
                              int dim,
                              double gamma)
{
  const int slope_qord = DefaultSlopeQord(vfes);

  if (config.bj_mode == BJLimiterMode::kComponent)
  {
    BJOptions opts = config.bj_opts;
    opts.dim = dim;
    if (opts.qord_avg == 0)
    {
      opts.qord_avg = slope_qord;
    }
    if (opts.slope_qord == 0)
    {
      opts.slope_qord = slope_qord;
    }
    rk3.AddBarthJespersenLimiter(opts);
  }
  else if (config.bj_mode == BJLimiterMode::kCharacteristic)
  {
    CharBJOptions opts = config.char_bj_opts;
    opts.dim = dim;
    opts.gamma = gamma;
    if (opts.slope_qord == 0)
    {
      opts.slope_qord = slope_qord;
    }
    rk3.AddCharBJLimiter(opts);
  }
  else if (config.bj_mode == BJLimiterMode::kFaceNormalCharacteristic)
  {
    FaceBJOptions opts = config.face_bj_opts;
    opts.dim = dim;
    opts.gamma = gamma;
    if (opts.slope_qord == 0)
    {
      opts.slope_qord = slope_qord;
    }
    rk3.AddFaceCharBJLimiter(opts);
  }

  if (config.tvb_mode == TVBLimiterMode::kCharacteristic)
  {
    CharTVBOptions opts = config.tvb_opts;
    opts.dim = dim;
    opts.gamma = gamma;
    if (opts.slope_qord == 0)
    {
      opts.slope_qord = slope_qord;
    }
    rk3.AddCharTVBLimiter(opts);
  }
  else if (config.tvb_mode == TVBLimiterMode::kFaceNormal)
  {
    CharTVBOptions opts = config.tvb_opts;
    opts.dim = dim;
    opts.gamma = gamma;
    if (opts.slope_qord == 0)
    {
      opts.slope_qord = slope_qord;
    }
    rk3.AddCharTVBFaceNormalLimiter(opts);
  }

  if (config.venkat_mode == VenkatLimiterMode::kXY ||
      config.venkat_mode == VenkatLimiterMode::kFaceNormal)
  {
    VenkatOptions opts = config.venkat_opts;
    opts.dim = dim;
    opts.gamma = gamma;
    if (opts.slope_qord == 0)
    {
      opts.slope_qord = slope_qord;
    }
    if (config.venkat_mode == VenkatLimiterMode::kXY)
    {
      rk3.AddCharVenkatXYLimiter(opts);
    }
    else
    {
      rk3.AddCharVenkatFaceNormalLimiter(opts);
    }
  }

  if (config.krivodonova)
  {
    KrivoOptions opts = config.krivo_opts;
    opts.dim = dim;
    opts.gamma = gamma;
    if (opts.face_qord == 0)
    {
      opts.face_qord = DefaultFaceQord(vfes);
    }
    rk3.AddKrivodonovaLimiter(opts);
  }

  if (config.zhang_shu)
  {
    rk3.AddZhangShuLimiter(config.zhang_shu_points);
  }
}

} // namespace mfem

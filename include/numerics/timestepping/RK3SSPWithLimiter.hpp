#pragma once
#include "mfem.hpp"
#include "utils/config/timestepping_config.hpp"
#include "numerics/limiters/limiters.hpp" // brings limiter option types
#include "numerics/timestepping/stage_aware_operator.hpp"
#include <functional>
#include <vector>

/**
 * @file numerics/timestepping/RK3SSPWithLimiter.hpp
 * @brief SSP RK3 with a stage-wise callback pipeline (e.g., BJ -> TVB -> PP).
 * @ingroup numerics_timestepping
 *
 * Mirrors MFEM's RK3SSPSolver but inserts a user-defined sequence of
 * element-local callbacks between stages. Each callback operates on a
 * ParGridFunction view of the stage vector and may use the current time.
 *
 * Nothing is hardcoded: add any number of callbacks with AddCallback(...).
 *
 * Convenience helpers AddZhangShuLimiter(), AddCharTVBLimiter(...),
 * and AddBarthJespersenLimiter(...) are provided; they push properly
 * configured callbacks that call the limiter routines from limiters.cpp.
 * Convenience helpers push properly configured callbacks that call the limiter
 * routines from limiters.cpp:
 *   - AddZhangShuLimiter()
 *   - AddCharTVBLimiter(...)
 *   - AddCharBJLimiter(...)
 *   - AddBarthJespersenLimiter(...)
 */

namespace mfem
{

/// SSPRK(3,3) with an ordered list of stage callbacks (limiters/filters).
/// @ingroup numerics_timestepping
class RK3SSPWithLimiter : public ODESolver
{
public:
  /// Configuration bundle for RK3 + limiter defaults.
  using Config = RK3Config;

  /// Construct with the FE space used by the conservative state U (vdim = dim+2).
  explicit RK3SSPWithLimiter(ParFiniteElementSpace& vfes);

  /// Configure Euler parameters (used by helper callbacks).
  void SetEulerParams(int dim, double gamma)
  {
    dim_ = dim;
    gamma_ = gamma;
  }

  /// Set default params used by the Zhang–Shu PP helper callback.
  void SetLimiterParams(double rho_min,
                        double p_min,
                        int quad_order,
                        int p_bisect_iters)
  {
    rho_min_ = rho_min;
    p_min_ = p_min;
    quad_order_ = quad_order;
    p_bisect_iters_ = p_bisect_iters;
  }

  /// If true, also apply the pipeline at the start of Step() to the input x.
  void LimitInitial(bool on) { limit_initial_ = on; }

  /// Limiters scheduling: optional callbacks are applied only for nsteps >= 0.
  ///   nsteps < 0  -> optional callbacks are always applied (no limit).
  ///   nsteps >= 0 -> optional callbacks applied while step_count_ < nsteps.
  void SetLimiterSteps(long nsteps) { max_limiter_steps_ = nsteps; }

  /// Enable per-stage NaN/inf scans (diagnostic; off by default).
  void EnableStageNanScan(bool on) { nan_scan_stages_ = on; }

  /// Reset the internal step counter (e.g., if restarting the time integrator).
  void ResetStepCount() { step_count_ = 0; }

  /// Apply a configuration bundle without changing existing defaults.
  void ApplyConfig(const Config& cfg)
  {
    SetEulerParams(cfg.dim, cfg.gamma);
    SetLimiterParams(cfg.rho_min,
                     cfg.p_min,
                     cfg.quad_order,
                     cfg.p_bisect_iters);
    LimitInitial(cfg.limit_initial);
    SetLimiterSteps(cfg.max_limiter_steps);
  }

  /// Apply only limiter-related parameters (Euler + positivity defaults).
  void ApplyLimiterConfig(const LimiterConfig& cfg)
  {
    SetEulerParams(cfg.dim, cfg.gamma);
    SetLimiterParams(cfg.rho_min,
                     cfg.p_min,
                     cfg.quad_order,
                     cfg.p_bisect_iters);
  }

  // ----------------- Ordered list of stage callbacks -----------------
  // Signature: void(ParFiniteElementSpace&, ParGridFunction&, double t)
  using StageCallback =
    std::function<void(ParFiniteElementSpace&, ParGridFunction&, double)>;

  /// Remove all callbacks (both always and optional).
  void ClearCallbacks()
  {
    always_callbacks_.clear();
    optional_callbacks_.clear();
  }

  /// Append a callback that always runs (e.g. Zhang–Shu positivity).
  void AddAlwaysCallback(StageCallback cb)
  {
    always_callbacks_.push_back(std::move(cb));
  }

  /// Append a callback that only runs for the first max_limiter_steps_ steps.
  void AddOptionalCallback(StageCallback cb)
  {
    optional_callbacks_.push_back(std::move(cb));
  }

  /// Backward-compatible: generic AddCallback() -> treat as optional.
  void AddCallback(StageCallback cb) { AddOptionalCallback(std::move(cb)); }

  // ---- Convenience: push common limiters as callbacks (defined in .cpp) ----
  /// Add a Zhang–Shu positivity limiter.
  /// @param always_on
  ///   - true  -> add as an always-on callback (default)
  ///   - false -> add as an optional callback (respecting max_limiter_steps_)
  void AddZhangShuLimiter(bool always_on = true);
  void AddCharTVBLimiter(const CharTVBOptions& opts); // characteristic TVB
  void AddCharTVBFaceNormalLimiter(const CharTVBOptions& opts);
  void AddCharBJLimiter(const CharBJOptions& opts); // characteristic BJ
  void AddFaceCharBJLimiter(const FaceBJOptions& opts);
  void AddBarthJespersenLimiter(const BJOptions& opts); // component-wise BJ
  void AddCharVenkatXYLimiter(const VenkatOptions& opts);
  void AddCharVenkatFaceNormalLimiter(const VenkatOptions& opts);
  void AddKrivodonovaLimiter(const KrivoOptions& opts);

  // --------------- ODESolver interface ---------------
  void Init(TimeDependentOperator& f_) override;
  void Step(Vector& x, real_t& t, real_t& dt) override;

private:
  // FE space for state
  ParFiniteElementSpace& vfes_;

  // Euler / helper-limiter params (only used by helper callbacks)
  int dim_ = dg_igr::EulerDefaults::kDim;
  double gamma_ = dg_igr::EulerDefaults::kGamma;
  double rho_min_ = dg_igr::LimiterDefaults::kRhoMin;
  double p_min_ = dg_igr::LimiterDefaults::kPMin;
  int quad_order_ = dg_igr::LimiterDefaults::kQuadOrder; // 0 => auto: 2*order+2
  int p_bisect_iters_ = dg_igr::LimiterDefaults::kBisectIters;
  bool limit_initial_ = dg_igr::RK3Defaults::kLimitInitial;

  // Callback pipelines
  std::vector<StageCallback> always_callbacks_;   ///< Always run
  std::vector<StageCallback> optional_callbacks_; ///< Only early steps

  // Optional limiter scheduling
  long step_count_ = 0; ///< number of completed Step() calls
  long max_limiter_steps_ =
    dg_igr::RK3Defaults::kLimiterSteps; ///< <0 => optional limiters always on
  bool nan_scan_stages_ = false;
  bool stage_nan_detected_ = false;

  // Work vectors
  Vector y_, k_;

  // Helpers
  /// Apply both always-on and (optionally) optional callbacks to vector v.
  void ApplyCallbacks(Vector& v, double t, bool apply_optional);
  void MaybeScanStage(Vector& v, double t, const char* label);
  int AutoQuadOrder() const;
};

} // namespace mfem

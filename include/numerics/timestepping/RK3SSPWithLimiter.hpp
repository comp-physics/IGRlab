#pragma once

#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "utils/config/timestepping_config.hpp"
#include "numerics/limiters/limiters.hpp" // brings limiter option types
#include "numerics/timestepping/ssprk_coefficients.hpp"
#include "numerics/timestepping/stage_aware_operator.hpp"
#include "utils/diagnostics/euler_admissibility.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

/**
 * @file numerics/timestepping/RK3SSPWithLimiter.hpp
 * @brief Explicit SSP RK with a stage-wise callback pipeline.
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

class PhysicalBoundaryFluxIntegralProvider;

/// Explicit RK with stage callbacks; research methods have SSP coefficient zero.
///
/// The historical class name is retained so existing limiter and application
/// plumbing remains source-compatible. SSPRK(3,3) remains the default method.
/// @ingroup numerics_timestepping
class RK3SSPWithLimiter : public ODESolver
{
public:
  /// Configuration bundle for RK3 + limiter defaults.
  using Config = RK3Config;

  /// Construct with the FE space used by the conservative state U (vdim = dim+2).
  explicit RK3SSPWithLimiter(
    ParFiniteElementSpace& vfes,
    SSPRKMethod method = SSPRKMethod::SSPRK33);

  void SetMethod(SSPRKMethod method) { method_ = method; }
  SSPRKMethod GetMethod() const { return method_; }
  int GetStageCount() const { return SSPRKStageCount(method_); }
  int GetOrder() const { return SSPRKOrder(method_); }
  real_t GetSSPCoefficient() const { return SSPRKCoefficient(method_); }
  long long GetRHSEvaluationCount() const { return rhs_evaluation_count_; }

  /** Bind the characteristic-speed value produced by the active operator.
   *
   * The accepted aggregate is the maximum over every RHS stage and the
   * post-EndRKStep operator state in the last successfully committed attempt.
   * The endpoint read lets an algebraic reconstruction tighten the next CFL
   * step without evaluating another Euler residual. A callback or endpoint
   * rejection never publishes the attempted aggregate, so an outer rollback
   * cannot contaminate the next adaptive-CFL decision.
   */
  void SetCharacteristicSpeedReader(std::function<real_t()> reader)
  {
    characteristic_speed_reader_ = std::move(reader);
  }
  bool HasAcceptedCharacteristicSpeed() const
  {
    return accepted_characteristic_speed_available_;
  }
  real_t GetAcceptedCharacteristicSpeed() const
  {
    MFEM_VERIFY(accepted_characteristic_speed_available_,
                "No accepted SSPRK characteristic-speed aggregate is "
                "available.");
    return accepted_characteristic_speed_;
  }

  /// Accumulate accepted-step boundary fluxes without retaining stage history.
  void EnablePhysicalBoundaryFluxTracking(bool enable = true)
  {
    physical_boundary_flux_tracking_enabled_ = enable;
  }
  bool IsPhysicalBoundaryFluxTrackingEnabled() const
  {
    return physical_boundary_flux_tracking_enabled_;
  }
  /// Whether the initialized RHS operator provides exact one-shot captures.
  bool SupportsPhysicalBoundaryFluxTracking() const;
  /// Whether at least one fully accepted tracked step has been committed.
  bool HasCumulativePhysicalBoundaryFluxIntegral() const
  {
    return cumulative_physical_boundary_flux_available_;
  }
  /// Sum over accepted steps of dt * sum_i b_i * boundary_flux_i.
  const Vector& GetCumulativePhysicalBoundaryFluxIntegral() const;
  /// Componentwise sum of dt * b_i * abs(boundary_flux_i).
  const Vector& GetCumulativePhysicalBoundaryFluxAbsoluteIntegral() const;

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

  /// Set additional face quadrature orders that the positivity limiter must
  /// enforce. These should match the actual DG face integration rule(s).
  void SetLimiterFaceQuadratureOrders(std::vector<int> qords)
  {
    face_quad_orders_ = std::move(qords);
  }

  /// Optional order-0 DG fields storing the latest Zhang-Shu limiter activity.
  void SetZhangShuDiagnosticsFields(ParGridFunction* active,
                                    ParGridFunction* theta_rho,
                                    ParGridFunction* theta_p)
  {
    zhang_shu_active_field_ = active;
    zhang_shu_theta_rho_field_ = theta_rho;
    zhang_shu_theta_p_field_ = theta_p;
  }

  /// If true, also apply the pipeline at the start of Step() to the input x.
  void LimitInitial(bool on) { limit_initial_ = on; }

  /// Limiters scheduling: optional callbacks are applied only for nsteps >= 0.
  ///   nsteps < 0  -> optional callbacks are always applied (no limit).
  ///   nsteps >= 0 -> optional callbacks applied while step_count_ < nsteps.
  void SetLimiterSteps(long nsteps) { max_limiter_steps_ = nsteps; }

  /// Enable per-stage NaN/inf scans (diagnostic; off by default).
  void EnableStageNanScan(bool on) { nan_scan_stages_ = on; }
  /// Enable per-stage Euler admissibility scans (diagnostic; off by default).
  void EnableStageAdmissibilityScan(bool on) { admissibility_scan_stages_ = on; }

  /// Reset the internal step counter (e.g., if restarting the time integrator).
  void ResetStepCount() { step_count_ = 0; }
  /// Restore the completed global step count for limiter scheduling on restart.
  void SetStepCount(long step)
  {
    MFEM_VERIFY(step >= 0, "RK restart step must be nonnegative.");
    step_count_ = step;
  }

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

  /** Append a callback that always runs after any optional callbacks.
   *
   * This group is the terminal stage/endpoint gate; Zhang--Shu positivity is
   * placed here so a slope limiter cannot subsequently invalidate admission.
   */
  void AddAlwaysCallback(StageCallback cb)
  {
    always_callbacks_.push_back(
      [callback = std::move(cb)](ParFiniteElementSpace& vfes,
                                 ParGridFunction& state,
                                 double time,
                                 bool report_state_mutation) mutable
      {
        callback(vfes, state, time);
        // The legacy callback contract has no mutation result.  When exact
        // provenance matters, conservatively assume that invoking it changed
        // the conservative state.  Intermediate/final stages already acquire
        // a new generation from their RK update and do not request a report.
        return report_state_mutation;
      });
  }

  /// Append a callback that only runs for the first max_limiter_steps_ steps.
  void AddOptionalCallback(StageCallback cb)
  {
    optional_callbacks_.push_back(
      [callback = std::move(cb)](ParFiniteElementSpace& vfes,
                                 ParGridFunction& state,
                                 double time,
                                 bool report_state_mutation) mutable
      {
        callback(vfes, state, time);
        return report_state_mutation;
      });
  }

  /// Backward-compatible: generic AddCallback() -> treat as optional.
  void AddCallback(StageCallback cb) { AddOptionalCallback(std::move(cb)); }

  /// Whether an outer time loop must retain a rollback copy for callbacks.
  /// Callback-free transport can still require a transaction when its
  /// stage-aware RHS performs a recoverable admission check.
  bool RequiresStepTransaction() const
  {
    return !always_callbacks_.empty() || !optional_callbacks_.empty() ||
           rhs_requires_step_transaction_;
  }

  // ---- Convenience: push common limiters as callbacks (defined in .cpp) ----
  /// Add a Zhang–Shu positivity limiter.
  /// @param point_set Positivity points enforced by the limiter.
  /// @param always_on
  ///   - true  -> add as an always-on callback (default)
  ///   - false -> add as an optional callback (respecting max_limiter_steps_)
  /// @param enforce_vtk_points Explicit refined-VTK-point policy; omission
  ///   preserves the low-level legacy environment fallback.
  void AddZhangShuLimiter(
    ZhangShuPointSet point_set = ZhangShuPointSet::kConsumer,
    bool always_on = true,
    std::optional<bool> enforce_vtk_points = std::nullopt);
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

  /// Generation of the accepted state vector, for auxiliary-field provenance.
  std::uint64_t GetStateGeneration() const { return state_generation_; }
  /// Restore provenance after an outer time-loop transaction rejects a step.
  void RestoreStateGeneration(std::uint64_t state_generation)
  {
    state_generation_ = state_generation;
  }
  long GetStepCount() const { return step_count_; }
  void Step(Vector& x, real_t& t, real_t& dt) override;

private:
  // FE space for state
  ParFiniteElementSpace& vfes_;
  SSPRKMethod method_ = SSPRKMethod::SSPRK33;

  // Euler / helper-limiter params (only used by helper callbacks)
  int dim_ = igrlab::EulerDefaults::kDim;
  double gamma_ = igrlab::EulerDefaults::kGamma;
  double rho_min_ = igrlab::LimiterDefaults::kRhoMin;
  double p_min_ = igrlab::LimiterDefaults::kPMin;
  int quad_order_ = igrlab::LimiterDefaults::kQuadOrder; // 0 => auto: 2*order+2
  int p_bisect_iters_ = igrlab::LimiterDefaults::kBisectIters;
  std::vector<int> face_quad_orders_;
  igrlab::detail::EulerAdmissibilityCache admissibility_cache_;
  ParGridFunction* zhang_shu_active_field_ = nullptr;
  ParGridFunction* zhang_shu_theta_rho_field_ = nullptr;
  ParGridFunction* zhang_shu_theta_p_field_ = nullptr;
  ZhangShuLimiterElementDiagnostics zhang_shu_diagnostics_;
  bool zhang_shu_diagnostics_pending_ = false;
  bool limit_initial_ = igrlab::RK3Defaults::kLimitInitial;

  // Internal callbacks optionally report whether they changed the
  // conservative state.  Public legacy callbacks remain conservative: when a
  // report is requested they are treated as mutating.  Built-in callbacks
  // with an exact mutation result can preserve accepted-state provenance.
  using InternalStageCallback = std::function<bool(ParFiniteElementSpace&,
                                                    ParGridFunction&,
                                                    double,
                                                    bool)>;
  std::vector<InternalStageCallback> always_callbacks_;   ///< Always run
  std::vector<InternalStageCallback> optional_callbacks_; ///< Only early steps

  // Optional limiter scheduling
  long step_count_ = 0; ///< number of completed Step() calls
  long max_limiter_steps_ =
    igrlab::RK3Defaults::kLimiterSteps; ///< <0 => optional limiters always on
  bool nan_scan_stages_ = false;
  bool stage_nan_detected_ = false;
  bool admissibility_scan_stages_ = false;
  bool stage_pair_capture_written_ = false;
  std::uint64_t state_generation_ = 0;
  long long rhs_evaluation_count_ = 0;
  std::function<real_t()> characteristic_speed_reader_;
  real_t attempted_characteristic_speed_ = 0.0;
  real_t accepted_characteristic_speed_ = 0.0;
  bool attempted_characteristic_speed_available_ = false;
  bool accepted_characteristic_speed_available_ = false;
  bool rhs_requires_step_transaction_ = false;
  const PhysicalBoundaryFluxIntegralProvider* boundary_flux_provider_ = nullptr;
  bool physical_boundary_flux_tracking_enabled_ = false;
  bool cumulative_physical_boundary_flux_available_ = false;

  // Work vectors
  Vector y_, k_;
  std::array<Vector, 6> research_stage_work_;
  Vector attempted_physical_boundary_flux_integral_;
  Vector attempted_physical_boundary_flux_absolute_integral_;
  Vector cumulative_physical_boundary_flux_integral_;
  Vector cumulative_physical_boundary_flux_absolute_integral_;

  // Helpers
  /** Apply both callback groups and optionally report state mutation.
   *
   * Mutation reporting is requested only for the accepted state at Step()
   * entry.  Every other callback boundary follows an RK state update and
   * therefore advances provenance independently of limiter activity.
   */
  bool ApplyCallbacks(Vector& v,
                      double t,
                      bool apply_optional,
                      bool report_state_mutation = false);
  void AdvanceStateGeneration();
  void MaybeScanVector(Vector& v, double t, const char* label);
  void MaybeScanAdmissibility(Vector& v, double t, const char* label);
  void MaybeCaptureStagePair(Vector& v, double t, const char* side);
  int AutoQuadOrder() const;
};

} // namespace mfem

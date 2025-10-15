#pragma once
#include "mfem.hpp"
#include "limiters.hpp"   // <-- brings CharTVBOptions and BJOptions in as complete types
#include <functional>
#include <vector>

/**
 * @file timesteppers.hpp
 * @brief SSP RK3 with a stage-wise callback pipeline (e.g., BJ -> TVB -> PP).
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
 */

namespace mfem {

/// SSPRK(3,3) with an ordered list of stage callbacks (limiters/filters).
class RK3SSPWithLimiter : public ODESolver
{
public:
  /// Construct with the FE space used by the conservative state U (vdim = dim+2).
  explicit RK3SSPWithLimiter(ParFiniteElementSpace &vfes);

  /// Configure Euler parameters (used by helper callbacks).
  void SetEulerParams(int dim, double gamma) { dim_ = dim; gamma_ = gamma; }

  /// Set default params used by the Zhang–Shu PP helper callback.
  void SetLimiterParams(double rho_min, double p_min,
                        int quad_order, int p_bisect_iters)
  { rho_min_ = rho_min; p_min_ = p_min; quad_order_ = quad_order; p_bisect_iters_ = p_bisect_iters; }

  /// If true, also apply the pipeline at the start of Step() to the input x.
  void LimitInitial(bool on) { limit_initial_ = on; }

  // ----------------- Ordered list of stage callbacks -----------------
  // Signature: void(ParFiniteElementSpace&, ParGridFunction&, double t)
  using StageCallback = std::function<void(ParFiniteElementSpace&, ParGridFunction&, double)>;

  /// Remove all callbacks.
  void ClearCallbacks() { callbacks_.clear(); }

  /// Append a callback to the pipeline (executed in registration order).
  void AddCallback(StageCallback cb) { callbacks_.push_back(std::move(cb)); }

  // ---- Convenience: push common limiters as callbacks (defined in .cpp) ----
  void AddZhangShuLimiter();                          // uses SetEulerParams + SetLimiterParams
  void AddCharTVBLimiter(const CharTVBOptions &opts); // characteristic TVB
  void AddBarthJespersenLimiter(const BJOptions &opts); // component-wise BJ slope limiter

  // --------------- ODESolver interface ---------------
  void Init(TimeDependentOperator &f_) override;
  void Step(Vector &x, real_t &t, real_t &dt) override;

private:
  // FE space for state
  ParFiniteElementSpace &vfes_;

  // Euler / helper-limiter params (only used by helper callbacks)
  int    dim_            = 2;
  double gamma_          = 1.4;
  double rho_min_        = 1e-12;
  double p_min_          = 1e-12;
  int    quad_order_     = 0;      // 0 => auto: 2*order+2
  int    p_bisect_iters_ = 12;
  bool   limit_initial_  = true;

  // Callback pipeline
  std::vector<StageCallback> callbacks_;

  // Work vectors
  Vector y_, k_;

  // Helpers
  void ApplyCallbacks(Vector &v, double t);
  int  AutoQuadOrder() const;
};

} // namespace mfem

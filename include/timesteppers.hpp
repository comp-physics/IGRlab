#pragma once
#include "mfem.hpp"
#include <functional>

/**
 * @file timesteppers.hpp
 * @brief SSP RK3 with stage-wise positivity limiting for hyperbolic DG.
 *
 * This time stepper mirrors MFEM's RK3SSPSolver but inserts a positivity
 * limiter between the three SSPRK stages:
 *
 *   1) y  = x + dt * f(t, x)                 [limit(y)]
 *   2) y  = 3/4 x + 1/4 ( y + dt * f(t+dt,y) )   [limit(y)]
 *   3) x  = 1/3 x + 2/3 ( y + dt * f(t+dt/2,y) ) [limit(x)]
 *
 * The limiter is expected to enforce ρ >= ρ_min and p >= p_min at a set of
 * positivity points, preserving element-wise averages (Zhang–Shu style).
 *
 * Usage:
 *   - Construct with the ParFES you use for the conservative state U.
 *   - Set γ, (ρ_min, p_min), quadrature order, and bisection iters.
 *   - Call Init(op) and use like any MFEM ODESolver.
 *
 * Notes:
 *   - We view stage vectors as ParGridFunctions on 'vfes' using external data,
 *     so no copying is done for limiting.
 *   - If you also use IGR, your operator can keep recomputing Σ inside Mult().
 */

namespace mfem {

/// Stage-wise positivity-limited SSPRK3
class RK3SSPWithLimiter : public ODESolver
{
public:
  /// Construct with the FE space used by the conservative state vector (vdim = dim+2).
  explicit RK3SSPWithLimiter(ParFiniteElementSpace &vfes);

  /// Configure Euler parameters and limiter thresholds.
  void SetEulerParams(int dim, double gamma) { dim_ = dim; gamma_ = gamma; }

  /// Set limiter thresholds and controls.
  /// quad_order: rule used for positivity points (recommend 2*order+2).
  /// p_bisect_iters: bisection iters to find theta for pressure.
  void SetLimiterParams(double rho_min, double p_min,
                        int quad_order, int p_bisect_iters)
  {
    rho_min_ = rho_min; p_min_ = p_min;
    quad_order_ = quad_order; p_bisect_iters_ = p_bisect_iters;
  }

  /// If true, also limit the *input* x at the start of Step() (default true).
  void LimitInitial(bool on) { limit_initial_ = on; }

  /// ODESolver interface
  void Init(TimeDependentOperator &f_) override;
  void Step(Vector &x, real_t &t, real_t &dt) override;

private:
  // FE space for state
  ParFiniteElementSpace &vfes_;

  // Euler / limiter parameters
  int    dim_            = 2;
  double gamma_          = 1.4;
  double rho_min_        = 1e-12;
  double p_min_          = 1e-12;
  int    quad_order_     = 0;      // 0 => auto: 2*order+2
  int    p_bisect_iters_ = 12;
  bool   limit_initial_  = true;

  // Work vectors
  Vector y_, k_;

  // Helpers
  void LimitVectorAsGridFunction(Vector &v);
  int  AutoQuadOrder() const;
};

} // namespace mfem

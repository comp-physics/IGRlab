#pragma once

#include "igrlab/compat.hpp"

/**
 * @file sim/solver/solver_helpers.hpp
 * @brief Helper utilities for solver setup (RK3/IGR).
 */

#include "mfem.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "physics/igr/IGR.hpp"

#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/context/mesh_build.hpp"

#include <cmath>
#include <limits>
#include <memory>

namespace dg_sim
{

inline void ApplyGradOption(const std::string& grad_opt, ::igrlab::IGRConfig& cfg)
{
  const std::string lower = ToLowerCopy(grad_opt);
  if (lower == "h1")
  {
    cfg.backend = ::igrlab::IGRGradBackend::H1Projection;
    return;
  }
  if (lower == "l2" || lower == "brokenl2" || lower == "brokendg" ||
      lower == "broken-dg")
  {
    cfg.backend = ::igrlab::IGRGradBackend::BrokenDG;
    return;
  }
  MFEM_VERIFY(false,
              "Invalid IGR gradient backend '" << grad_opt
                                               << "'. Use 'H1' or 'BrokenL2'.");
  cfg.backend = ::igrlab::IGRGradBackend::BrokenDG;
}

inline void ApplyIGRRuntimeOptionOverrides(const IGROptions& igr_opts,
                                           ::igrlab::IGRConfig& cfg)
{
  if (!igr_opts.alpha_length_policy.empty())
  {
    cfg.alpha_length_scale_policy =
      ParseIGRAlphaLengthScalePolicy(igr_opts.alpha_length_policy);
  }
  ApplyGradOption(igr_opts.grad, cfg);
  cfg.solve_mode = ParseIGRSolveMode(igr_opts.solve_mode);
  if (igr_opts.regular_sweeps != -1)
  {
    cfg.regular_sweeps = igr_opts.regular_sweeps;
  }
  if (igr_opts.initial_sweeps != -1)
  {
    cfg.warm_start_sweeps = igr_opts.initial_sweeps;
  }
  if (igr_opts.element_jacobi_omega != -1.0)
  {
    cfg.element_jacobi_omega = igr_opts.element_jacobi_omega;
  }
  cfg.lin_solver_type = ParseIGRLinearSolverType(igr_opts.linear_solver);
  cfg.prec_type = ParseIGRPrecondType(igr_opts.preconditioner);
  if (!igr_opts.nonconvergence_policy.empty())
  {
    cfg.nonconvergence_policy =
      ParseIGRNonconvergencePolicy(igr_opts.nonconvergence_policy);
  }
  cfg.ip_kind = ParseIGRIPKind(igr_opts.ip_kind);
  if (!igr_opts.sigma_cadence.empty())
  {
    cfg.sigma_cadence = ParseIGRSigmaUpdateCadence(igr_opts.sigma_cadence);
  }
  if (igr_opts.sigma_update_interval > 0)
  {
    cfg.sigma_update_interval = igr_opts.sigma_update_interval;
  }
  if (igr_opts.stage_rtol_scale >= 0.0)
  {
    cfg.stage_rtol_scale = igr_opts.stage_rtol_scale;
  }

  if (igr_opts.lin_rtol >= 0.0)
  {
    cfg.lin_rtol = igr_opts.lin_rtol;
  }
  if (igr_opts.lin_atol >= 0.0)
  {
    cfg.lin_atol = igr_opts.lin_atol;
  }
  if (igr_opts.lin_maxit >= 0)
  {
    cfg.lin_maxit = igr_opts.lin_maxit;
  }
  if (igr_opts.lin_print != std::numeric_limits<int>::min())
  {
    cfg.lin_print = igr_opts.lin_print;
  }
  if (igr_opts.amg_coarsen_type >= 0)
  {
    cfg.amg_coarsen_type = igr_opts.amg_coarsen_type;
  }
  if (igr_opts.amg_relax_type >= 0)
  {
    cfg.amg_relax_type = igr_opts.amg_relax_type;
  }
  if (igr_opts.amg_max_levels >= 0)
  {
    cfg.amg_max_levels = igr_opts.amg_max_levels;
  }
  if (igr_opts.amg_strong_threshold >= 0.0)
  {
    cfg.amg_strong_threshold = igr_opts.amg_strong_threshold;
  }
  cfg.exact_dg_face_operator = igr_opts.exact_dg_face_operator;
}

inline mfem::SSPRKMethod SSPRKMethodForTimeIntegrator(
  TimeIntegrator time_integrator)
{
  switch (time_integrator)
  {
  case TimeIntegrator::RK3SSP:
    return mfem::SSPRKMethod::SSPRK33;
  case TimeIntegrator::SSPRK104:
    return mfem::SSPRKMethod::SSPRK104;
  case TimeIntegrator::ResearchRK4:
    return mfem::SSPRKMethod::ResearchRK4;
  case TimeIntegrator::ResearchDOPRI5:
    return mfem::SSPRKMethod::ResearchDOPRI5;
  case TimeIntegrator::ResearchRadauPicard5:
    return mfem::SSPRKMethod::ResearchRadauPicard5;
  case TimeIntegrator::BackwardEuler:
  case TimeIntegrator::PartitionedBackwardEuler:
  case TimeIntegrator::Rodas3:
    break;
  }
  MFEM_VERIFY(false,
              "Selected time integrator is not an explicit RK method.");
  return mfem::SSPRKMethod::SSPRK33;
}

inline mfem::RK3SSPWithLimiter MakeRK3Stepper(
  mfem::ParFiniteElementSpace& vfes,
  int dim,
  const mfem::RK3Config& cfg,
  TimeIntegrator time_integrator = TimeIntegrator::RK3SSP)
{
  mfem::RK3SSPWithLimiter rk3(
    vfes, SSPRKMethodForTimeIntegrator(time_integrator));
  rk3.SetEulerParams(dim, cfg.gamma);
  rk3.SetLimiterParams(cfg.rho_min,
                       cfg.p_min,
                       cfg.quad_order,
                       cfg.p_bisect_iters);
  rk3.LimitInitial(cfg.limit_initial);
  rk3.SetLimiterSteps(cfg.max_limiter_steps);
  return rk3;
}

/** Resolve the RK configuration that the application actually executes.
 *
 * The extrapolated instantaneous-IGR route admits and limits the initialized
 * (or restarted) Euler state before constructing its first exact algebraic
 * anchor.  Reapplying the legacy per-step entry limiter after BeginRKStep
 * could mutate that state, force an extra anchor, and discard the two-point
 * extrapolation history.  Stage and accepted-endpoint limiting remain active;
 * only that redundant RK-entry callback is disabled for this typed route.
 */
inline mfem::RK3Config ResolveRK3StepperConfig(const RunConfig& run_cfg)
{
  mfem::RK3Config cfg = run_cfg.case_config.rk3_config;
  if (UsesFrozenSigmaSpectralSpeed(run_cfg))
  {
    cfg.limit_initial = false;
  }
  return cfg;
}

inline std::unique_ptr<::igrlab::IGRSolver> MakeIGRSolver(
  DGSpaces& spaces,
  int dim,
  int num_equations,
  const RunConfig& run_cfg,
  double rho_floor,
  const AlphaInfo& alpha_info,
  mfem::ParFiniteElementSpace* closure_state_fes = nullptr)
{
  MFEM_VERIFY(spaces.sigma_fes != nullptr, "IGR requires a sigma space.");
  ::igrlab::IGRConfig igr_cfg = run_cfg.case_config.igr_config;
  igr_cfg.alpha = alpha_info.alpha;
  igr_cfg.alpha_length_scale_policy = alpha_info.policy;
  igr_cfg.rho_floor = rho_floor;
  ApplyIGRRuntimeOptionOverrides(run_cfg.igr_opts, igr_cfg);
  // The common CLI policy also controls IGR's optional routine diagnostics.
  igr_cfg.nan_scan = run_cfg.opts.nan_scan;

  // Coefficients retain references to this solver's alpha/policy members.
  // Construct at its final address so returning from the factory moves only
  // the owning pointer, rather than leaving references into a moved solver.
  auto igr = std::make_unique<::igrlab::IGRSolver>(
    *spaces.sigma_fes,
    closure_state_fes ? *closure_state_fes : spaces.vfes,
    dim,
    num_equations,
    igr_cfg.alpha);
  igr->ApplyConfig(igr_cfg);
  return igr;
}

} // namespace dg_sim

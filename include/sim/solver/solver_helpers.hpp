#pragma once

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

namespace dg_sim
{

inline void ApplyGradOption(const std::string& grad_opt, dg_igr::IGRConfig& cfg)
{
  const std::string lower = ToLowerCopy(grad_opt);
  if (lower == "h1")
  {
    cfg.backend = dg_igr::IGRGradBackend::H1Projection;
    return;
  }
  if (lower == "l2" || lower == "brokenl2" || lower == "brokendg" ||
      lower == "broken-dg")
  {
    cfg.backend = dg_igr::IGRGradBackend::BrokenDG;
    return;
  }
  MFEM_VERIFY(false,
              "Invalid IGR gradient backend '" << grad_opt
                                               << "'. Use 'H1' or 'BrokenL2'.");
  cfg.backend = dg_igr::IGRGradBackend::BrokenDG;
}

inline void ApplyIGRRuntimeOptionOverrides(const IGROptions& igr_opts,
                                           dg_igr::IGRConfig& cfg)
{
  if (!igr_opts.alpha_length_policy.empty())
  {
    cfg.alpha_length_scale_policy =
      ParseIGRAlphaLengthScalePolicy(igr_opts.alpha_length_policy);
  }
  ApplyGradOption(igr_opts.grad, cfg);
  cfg.solve_mode = ParseIGRSolveMode(igr_opts.solve_mode);
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

inline mfem::RK3SSPWithLimiter MakeRK3Stepper(mfem::ParFiniteElementSpace& vfes,
                                              int dim,
                                              const mfem::RK3Config& cfg)
{
  mfem::RK3SSPWithLimiter rk3(vfes);
  rk3.SetEulerParams(dim, cfg.gamma);
  rk3.SetLimiterParams(cfg.rho_min,
                       cfg.p_min,
                       cfg.quad_order,
                       cfg.p_bisect_iters);
  rk3.LimitInitial(cfg.limit_initial);
  rk3.SetLimiterSteps(cfg.max_limiter_steps);
  return rk3;
}

inline dg_igr::IGRSolver MakeIGRSolver(DGSpaces& spaces,
                                       int dim,
                                       int num_equations,
                                       const RunConfig& run_cfg,
                                       double rho_floor,
                                       const AlphaInfo& alpha_info)
{
  MFEM_VERIFY(spaces.sigma_fes != nullptr, "IGR requires a sigma space.");
  dg_igr::IGRConfig igr_cfg = run_cfg.case_config.igr_config;
  igr_cfg.alpha = alpha_info.alpha;
  igr_cfg.alpha_length_scale_policy = alpha_info.policy;
  igr_cfg.rho_floor = rho_floor;
  ApplyIGRRuntimeOptionOverrides(run_cfg.igr_opts, igr_cfg);

  dg_igr::IGRSolver igr(*spaces.sigma_fes,
                        spaces.vfes,
                        dim,
                        num_equations,
                        igr_cfg.alpha);
  igr.ApplyConfig(igr_cfg);
  return igr;
}

} // namespace dg_sim

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
  cfg.backend = dg_igr::IGRGradBackend::BrokenDG;
}

inline mfem::RK3SSPWithLimiter MakeRK3Stepper(mfem::ParFiniteElementSpace& vfes,
                                              int dim,
                                              const mfem::RK3Config& cfg)
{
  mfem::RK3SSPWithLimiter rk3(vfes);
  rk3.SetEulerParams(dim, cfg.gamma);
  rk3.SetLimiterParams(cfg.rho_min, cfg.p_min, cfg.quad_order, cfg.p_bisect_iters);
  rk3.LimitInitial(cfg.limit_initial);
  rk3.SetLimiterSteps(cfg.max_limiter_steps);
  return rk3;
}

inline dg_igr::IGRSolver MakeIGRSolver(const DGSpaces& spaces,
                                       int dim,
                                       int num_equations,
                                       const RunConfig& run_cfg,
                                       double rho_floor,
                                       const AlphaInfo& alpha_info)
{
  MFEM_VERIFY(spaces.sigma_fes != nullptr, "IGR requires a sigma space.");
  dg_igr::IGRConfig igr_cfg = run_cfg.case_config.igr_config;
  igr_cfg.alpha = alpha_info.alpha;
  igr_cfg.rho_floor = rho_floor;
  ApplyGradOption(run_cfg.igr_opts.grad, igr_cfg);

  dg_igr::IGRSolver igr(*spaces.sigma_fes, spaces.vfes, dim, num_equations, igr_cfg.alpha);
  igr.ApplyConfig(igr_cfg);
  return igr;
}

} // namespace dg_sim

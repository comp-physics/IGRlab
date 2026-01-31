#pragma once

/**
 * @file sim/config/case_config.hpp
 * @brief Case/Run configuration structures for the sim-layer.
 */

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>

#include "mfem.hpp"
#include "numerics/limiters/limiter_pipeline.hpp"
#include "utils/config/igr_config.hpp"
#include "utils/config/io_config.hpp"
#include "utils/config/timestepping_config.hpp"

#include "sim/config/common_options.hpp"
#include "sim/time/time_loop.hpp"

namespace dg_igr_examples
{

/// @brief Options controlling the IGR discretization wiring in examples.
struct IGROptions
{
  double c_alpha_user = -1.0;
  std::string space = "H1";
  std::string grad = "H1";
};

/// @brief Per-case configuration defaults plus case identity metadata.
struct CaseConfig
{
  std::string case_name;
  std::string default_mesh_file;
  std::string case_dir;
  int int_order_offset = 1;
  mfem::real_t gamma = 1.4;
  CommonOptions defaults;
  mfem::RK3Config rk3_config;
  dg_igr::IGRConfig igr_config;
  mfem::LimiterPipelineConfig limiter_pipeline;
  mfem::io::VTKOutputConfig vtk_config;
};

/// @brief Resolved runtime configuration after CLI parsing.
struct RunConfig
{
  CaseConfig case_config;
  CommonOptions opts;
  IGROptions igr_opts;
  std::string mesh_path;
  int order = 1;
  int ser_ref_levels = 0;
  int par_ref_levels = 0;
  mfem::real_t t_final = 0.0;
  mfem::real_t dt = 0.0;
  mfem::real_t cfl = 0.0;
  bool visualization = false;
  bool vtk_save = false;
  mfem::real_t dt_save = 0.0;
  bool preassemble_weak_div = true;
  int vis_steps = 0;
  int int_order_offset = 1;
};

inline std::string ToLowerCopy(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline std::string CaseDirFromFile(const char* file)
{
  return std::filesystem::path(file).parent_path().string();
}

inline std::string ResolveCaseMesh(const CaseConfig& cfg, const std::string& override_mesh)
{
  if (!override_mesh.empty())
  {
    return override_mesh;
  }
  return (std::filesystem::path(cfg.case_dir) / cfg.default_mesh_file).string();
}

inline std::string DefaultCaseMeshPath(const CaseConfig& cfg)
{
  return ResolveCaseMesh(cfg, "");
}

/// @brief Apply global example defaults, overriding any struct defaults.
///
/// This function reinitializes config members with default-constructed values
/// and then applies overrides (e.g., RK3/IGR/limiter settings). If you are
/// looking for where defaults are set, check here first.
inline void ApplyExampleDefaults(CaseConfig& cfg)
{
  cfg.defaults = CommonOptions{};
  cfg.rk3_config = mfem::RK3Config{};
  cfg.igr_config = dg_igr::IGRConfig{};
  cfg.limiter_pipeline = mfem::LimiterPipelineConfig{};
  cfg.vtk_config = mfem::io::VTKOutputConfig{};

  cfg.limiter_pipeline.venkat_opts.eps_kappa = 100.0;

  cfg.rk3_config.gamma = cfg.gamma;
  cfg.rk3_config.rho_min = 5.0e-7;
  cfg.rk3_config.p_min = 1.0e-6;
  cfg.rk3_config.quad_order = 0;
  cfg.rk3_config.p_bisect_iters = 12;
  cfg.rk3_config.limit_initial = false;
  cfg.rk3_config.max_limiter_steps = 0;

  cfg.igr_config.lin_rtol = 1.0e-8;
  cfg.igr_config.lin_atol = 0.0;
  cfg.igr_config.lin_maxit = 1000;
  cfg.igr_config.lin_print = 0;
}

inline CaseConfig MakeCaseConfig(const std::string& case_name,
                                 const std::string& default_mesh_file,
                                 const char* case_file,
                                 mfem::real_t gamma = mfem::dg_igr::EulerDefaults::kGamma)
{
  CaseConfig cfg;
  cfg.case_name = case_name;
  cfg.default_mesh_file = default_mesh_file;
  cfg.case_dir = CaseDirFromFile(case_file);
  cfg.gamma = gamma;
  ApplyExampleDefaults(cfg);
  return cfg;
}

inline void AddIGROptions(mfem::OptionsParser& args, IGROptions& igr_opts)
{
  args.AddOption(&igr_opts.c_alpha_user, "-ca", "--c-alpha",
                 "Forward scaling constant C_alpha (default 1.0).");
  args.AddOption(&igr_opts.space, "-is", "--igr-space",
                 "Space to solve for sigma: 'H1' or 'L2'.");
  args.AddOption(&igr_opts.grad, "-ig", "--igr-grad",
                 "Gradient reconstruction type: 'H1' or 'BrokenL2'.");
}

inline RunConfig MakeRunConfig(const CaseConfig& case_cfg,
                               const CommonOptions& opts,
                               const IGROptions& igr_opts)
{
  RunConfig run_cfg;
  run_cfg.case_config = case_cfg;
  run_cfg.opts = opts;
  run_cfg.igr_opts = igr_opts;
  run_cfg.mesh_path = ResolveCaseMesh(case_cfg, opts.mesh_file);
  run_cfg.order = opts.order;
  run_cfg.ser_ref_levels = opts.ser_ref_levels;
  run_cfg.par_ref_levels = opts.par_ref_levels;
  run_cfg.t_final = opts.t_final;
  run_cfg.dt = opts.dt;
  run_cfg.cfl = opts.cfl;
  run_cfg.visualization = opts.visualization;
  run_cfg.vtk_save = opts.vtk_save;
  run_cfg.dt_save = opts.dt_save;
  run_cfg.preassemble_weak_div = opts.preassemble_weak_div;
  run_cfg.vis_steps = opts.vis_steps;
  run_cfg.int_order_offset = case_cfg.int_order_offset;
  return run_cfg;
}

inline TimeLoopConfig MakeTimeLoopConfig(const RunConfig& run_cfg)
{
  const CommonOptions& opts = run_cfg.opts;
  TimeLoopConfig cfg;
  cfg.t_final = opts.t_final;
  cfg.dt = opts.dt;
  cfg.cfl = opts.cfl;
  cfg.order = opts.order;
  cfg.vis_steps = opts.vis_steps;
  cfg.vtk_save = opts.vtk_save;
  cfg.dt_save = opts.dt_save;
  cfg.nan_scan = opts.nan_scan;
  if (opts.analysis_steps < 0)
  {
    cfg.analysis_steps = (opts.vis_steps > 0) ? opts.vis_steps * 10 : 0;
  }
  else
  {
    cfg.analysis_steps = opts.analysis_steps;
  }
  cfg.save_index_offset = 0;
  return cfg;
}

} // namespace dg_igr_examples

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

/// @brief DG backend selection.
enum class DGBackend
{
  DG,
  DGSEM
};

/// @brief Optional DGSEM stabilization hook selection.
enum class DGSEMStabilization
{
  None,
  SplitForm,
  Filter
};

/// @brief DGSEM volume flux selection for entropy-conservative options.
enum class DGSEMVolumeFlux
{
  Central,
  Chandrashekar,
  Ranocha
};

/// @brief DGSEM volume dissipation selection.
enum class DGSEMVolumeDissipation
{
  None,
  LLF,
  EntropyVarsLF
};

/// @brief DGSEM surface flux selection.
enum class DGSEMSurfaceFlux
{
  Rusanov,
  DiffusiveRusanov,
  EntropyStableWinters,
  EntropyStableLFEntropyVars
};

/// @brief Entropy Jacobian evaluation mode for ES LF surface flux.
enum class DGSEMEntropyJacobian
{
  FiniteDifference,
  Analytic
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
  DGBackend dg_backend = DGBackend::DG;
  DGSEMStabilization dgsem_stab = DGSEMStabilization::None;
  DGSEMVolumeFlux dgsem_volume_flux = DGSEMVolumeFlux::Central;
  DGSEMVolumeDissipation dgsem_volume_dissipation =
      DGSEMVolumeDissipation::None;
  DGSEMSurfaceFlux dgsem_surface_flux = DGSEMSurfaceFlux::Rusanov;
  DGSEMEntropyJacobian dgsem_entropy_jacobian = DGSEMEntropyJacobian::FiniteDifference;
  std::string mesh_path;
};

inline std::string ToLowerCopy(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline DGBackend ParseDGBackend(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "dg")
  {
    return DGBackend::DG;
  }
  if (lower == "dgsem")
  {
    return DGBackend::DGSEM;
  }
  MFEM_VERIFY(false, "Invalid DG backend '" << value << "'. Use 'dg' or 'dgsem'.");
  return DGBackend::DG;
}

inline DGSEMStabilization ParseDGSEMStabilization(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "none")
  {
    return DGSEMStabilization::None;
  }
  if (lower == "split" || lower == "split-form" || lower == "splitform")
  {
    return DGSEMStabilization::SplitForm;
  }
  if (lower == "filter" || lower == "filtering")
  {
    return DGSEMStabilization::Filter;
  }
  MFEM_VERIFY(false, "Invalid DGSEM stabilization '" << value
              << "'. Use 'none', 'split', or 'filter'.");
  return DGSEMStabilization::None;
}

inline const char* DGSEMStabilizationToString(DGSEMStabilization stab)
{
  switch (stab)
  {
    case DGSEMStabilization::None: return "none";
    case DGSEMStabilization::SplitForm: return "split";
    case DGSEMStabilization::Filter: return "filter";
  }
  return "none";
}

inline DGSEMVolumeFlux ParseDGSEMVolumeFlux(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "central")
  {
    return DGSEMVolumeFlux::Central;
  }
  if (lower == "chandrashekar" || lower == "chandra")
  {
    return DGSEMVolumeFlux::Chandrashekar;
  }
  if (lower == "ranocha")
  {
    return DGSEMVolumeFlux::Ranocha;
  }
  MFEM_VERIFY(false, "Invalid DGSEM volume flux '" << value
              << "'. Use 'central', 'chandrashekar', or 'ranocha'.");
  return DGSEMVolumeFlux::Central;
}

inline const char* DGSEMVolumeFluxToString(DGSEMVolumeFlux flux)
{
  switch (flux)
  {
    case DGSEMVolumeFlux::Central: return "central";
    case DGSEMVolumeFlux::Chandrashekar: return "chandrashekar";
    case DGSEMVolumeFlux::Ranocha: return "ranocha";
  }
  return "central";
}

inline DGSEMVolumeDissipation ParseDGSEMVolumeDissipation(
    const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "none")
  {
    return DGSEMVolumeDissipation::None;
  }
  if (lower == "llf" || lower == "lax-friedrichs" || lower == "rusanov")
  {
    return DGSEMVolumeDissipation::LLF;
  }
  if (lower == "entropy-lf" || lower == "entropy-vars-lf" ||
      lower == "entropy-variable-lf" || lower == "es-lf")
  {
    return DGSEMVolumeDissipation::EntropyVarsLF;
  }
  MFEM_VERIFY(false, "Invalid DGSEM volume dissipation '" << value
              << "'. Use 'none', 'llf', or 'entropy-lf'.");
  return DGSEMVolumeDissipation::None;
}

inline const char* DGSEMVolumeDissipationToString(DGSEMVolumeDissipation diss)
{
  switch (diss)
  {
    case DGSEMVolumeDissipation::None: return "none";
    case DGSEMVolumeDissipation::LLF: return "llf";
    case DGSEMVolumeDissipation::EntropyVarsLF: return "entropy-lf";
  }
  return "none";
}

inline DGSEMSurfaceFlux ParseDGSEMSurfaceFlux(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "rusanov" || lower == "lf" || lower == "lax-friedrichs")
  {
    return DGSEMSurfaceFlux::Rusanov;
  }
  if (lower == "diffusive-rusanov" || lower == "rusanov-diffusive" ||
      lower == "llf-diffusive" || lower == "diffusive-llf")
  {
    return DGSEMSurfaceFlux::DiffusiveRusanov;
  }
  if (lower == "es-winters" || lower == "entropy-winters" || lower == "winters")
  {
    return DGSEMSurfaceFlux::EntropyStableWinters;
  }
  if (lower == "es-lf-entropy" || lower == "entropy-lf" || lower == "es-lf")
  {
    return DGSEMSurfaceFlux::EntropyStableLFEntropyVars;
  }
  MFEM_VERIFY(false, "Invalid DGSEM surface flux '" << value
              << "'. Use 'rusanov', 'diffusive-rusanov', 'es-winters', or 'es-lf-entropy'.");
  return DGSEMSurfaceFlux::Rusanov;
}

inline DGSEMEntropyJacobian ParseDGSEMEntropyJacobian(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "fd" || lower == "finite" || lower == "finite-difference")
  {
    return DGSEMEntropyJacobian::FiniteDifference;
  }
  if (lower == "analytic" || lower == "exact")
  {
    return DGSEMEntropyJacobian::Analytic;
  }
  MFEM_VERIFY(false, "Invalid DGSEM entropy Jacobian '" << value
              << "'. Use 'fd' or 'analytic'.");
  return DGSEMEntropyJacobian::FiniteDifference;
}

inline const char* DGSEMEntropyJacobianToString(DGSEMEntropyJacobian mode)
{
  switch (mode)
  {
    case DGSEMEntropyJacobian::FiniteDifference: return "fd";
    case DGSEMEntropyJacobian::Analytic: return "analytic";
  }
  return "fd";
}

inline const char* DGSEMSurfaceFluxToString(DGSEMSurfaceFlux flux)
{
  switch (flux)
  {
    case DGSEMSurfaceFlux::Rusanov: return "rusanov";
    case DGSEMSurfaceFlux::DiffusiveRusanov: return "diffusive-rusanov";
    case DGSEMSurfaceFlux::EntropyStableWinters: return "es-winters";
    case DGSEMSurfaceFlux::EntropyStableLFEntropyVars: return "es-lf-entropy";
  }
  MFEM_ABORT("Unhandled DGSEMSurfaceFlux enum value in DGSEMSurfaceFluxToString.");
  return "rusanov";
}

inline const char* DGBackendToString(DGBackend backend)
{
  switch (backend)
  {
    case DGBackend::DG: return "dg";
    case DGBackend::DGSEM: return "dgsem";
  }
  return "dg";
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
  run_cfg.dg_backend = ParseDGBackend(opts.dg_backend);
  run_cfg.dgsem_stab = ParseDGSEMStabilization(opts.dgsem_stab);
  run_cfg.dgsem_volume_flux = ParseDGSEMVolumeFlux(opts.dgsem_volume_flux);
  run_cfg.dgsem_volume_dissipation =
      ParseDGSEMVolumeDissipation(opts.dgsem_volume_dissipation);
  run_cfg.dgsem_surface_flux = ParseDGSEMSurfaceFlux(opts.dgsem_surface_flux);
  run_cfg.dgsem_entropy_jacobian =
      ParseDGSEMEntropyJacobian(opts.dgsem_entropy_jacobian);
  run_cfg.mesh_path = ResolveCaseMesh(case_cfg, opts.mesh_file);
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

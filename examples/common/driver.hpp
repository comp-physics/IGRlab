#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>

#include "mfem.hpp"
#include "numerics/limiters/limiter_pipeline.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "physics/igr/IGR.hpp"
#include "utils/config/igr_config.hpp"
#include "utils/config/io_config.hpp"
#include "utils/config/timestepping_config.hpp"

#include "common_options.hpp"
#include "time_loop.hpp"

namespace dg_igr_examples
{

/// Options controlling the IGR discretization wiring in the examples.
struct IGROptions
{
  double c_alpha_user = -1.0;
  std::string space = "H1";
  std::string grad = "H1";
};

/// Per-case configuration defaults plus case identity metadata.
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
  vis::GLVisConfig glvis_config;
  mfem::io::VTKOutputConfig vtk_config;
};

/// Resolved runtime configuration after CLI parsing.
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

inline std::string ToLowerCopy(std::string value);

/// Aggregate DG spaces used by the Euler+IGR examples.
struct DGSpaces
{
  mfem::DG_FECollection fec;
  mfem::ParFiniteElementSpace fes;
  mfem::ParFiniteElementSpace dfes;
  mfem::ParFiniteElementSpace vfes;
  std::unique_ptr<mfem::FiniteElementCollection> sigma_fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> sigma_fes;
  std::unique_ptr<mfem::H1_FECollection> h1_fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> h1_vfes;

  DGSpaces(mfem::ParMesh& pmesh,
           int order,
           int dim,
           int num_equations,
           const std::string& sigma_space,
           bool need_sigma,
           bool need_h1)
    : fec(order, dim),
      fes(&pmesh, &fec),
      dfes(&pmesh, &fec, dim, mfem::Ordering::byNODES),
      vfes(&pmesh, &fec, num_equations, mfem::Ordering::byNODES)
  {
    if (need_sigma)
    {
      sigma_fec = BuildSigmaCollection(order, dim, sigma_space);
      sigma_fes = std::make_unique<mfem::ParFiniteElementSpace>(&pmesh, sigma_fec.get());
    }
    if (need_h1)
    {
      h1_fec = std::make_unique<mfem::H1_FECollection>(std::max(1, order), dim);
      h1_vfes = std::make_unique<mfem::ParFiniteElementSpace>(
          &pmesh, h1_fec.get(), dim, mfem::Ordering::byNODES);
    }
  }

private:
  static std::unique_ptr<mfem::FiniteElementCollection>
  BuildSigmaCollection(int order, int dim, const std::string& sigma_space)
  {
    const std::string lower = ToLowerCopy(sigma_space);
    if (lower == "h1")
    {
      return std::make_unique<mfem::H1_FECollection>(std::max(order, 3), dim);
    }
    return std::make_unique<mfem::DG_FECollection>(order, dim);
  }
};

/// Alpha diagnostics used in case naming and solver setup.
struct AlphaInfo
{
  mfem::real_t hmin = mfem::infinity();
  double c_alpha = 1.0;
  mfem::real_t alpha = 1.0;
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

inline void ApplyExampleDefaults(CaseConfig& cfg)
{
  cfg.defaults = CommonOptions{};
  cfg.rk3_config = mfem::RK3Config{};
  cfg.igr_config = dg_igr::IGRConfig{};
  cfg.limiter_pipeline = mfem::LimiterPipelineConfig{};
  cfg.glvis_config = vis::GLVisConfig{};
  cfg.vtk_config = mfem::io::VTKOutputConfig{};

  cfg.limiter_pipeline.venkat_opts.eps_kappa = 100.0;

  cfg.rk3_config.gamma = cfg.gamma;
  cfg.rk3_config.rho_min = 5.0e-7;
  cfg.rk3_config.p_min = 1.0e-6;
  cfg.rk3_config.quad_order = 0;
  cfg.rk3_config.p_bisect_iters = 12;
  cfg.rk3_config.limit_initial = true;
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

inline mfem::Mesh LoadSerialMesh(const std::string& mesh_path, int ser_ref_levels)
{
  mfem::Mesh mesh(mesh_path);
  for (int lev = 0; lev < ser_ref_levels; ++lev)
  {
    mesh.UniformRefinement();
  }
  return mesh;
}

inline void RefineParMesh(mfem::ParMesh& pmesh, int par_ref_levels)
{
  for (int lev = 0; lev < par_ref_levels; ++lev)
  {
    pmesh.UniformRefinement();
  }
}

inline AlphaInfo ComputeAlphaInfo(const mfem::ParMesh& pmesh, double c_alpha_user)
{
  AlphaInfo info;
  for (int i = 0; i < pmesh.GetNE(); ++i)
  {
    auto& mesh = const_cast<mfem::ParMesh&>(pmesh);
    info.hmin = std::min(info.hmin, mesh.GetElementSize(i, 1));
  }
  MPI_Allreduce(MPI_IN_PLACE, &info.hmin, 1,
                mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, pmesh.GetComm());

  info.c_alpha = (c_alpha_user > 0.0) ? c_alpha_user : 1.0;
  const mfem::real_t dx_eff = info.hmin;
  info.alpha = static_cast<mfem::real_t>(info.c_alpha * dx_eff * dx_eff);
  return info;
}

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

} // namespace dg_igr_examples

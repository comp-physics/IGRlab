#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <string>

#include "mfem.hpp"

#include "driver.hpp"
#include "driver_hooks.hpp"
#include "vis_helpers.hpp"

namespace dg_igr_examples
{

struct VisualizationConfig
{
  bool enabled = false;
  bool h1_diagnostics = false;
  int precision = 8;
};

struct VTKConfig
{
  bool enabled = true;
  bool print_dataset_name = false;
  double c_alpha_value = std::numeric_limits<double>::quiet_NaN();
};

struct EmptyCaseOptions
{};

template <typename CaseOptions, typename NumericalFlux>
struct CaseSpec
{
  using Options = CaseOptions;
  using Flux = NumericalFlux;

  std::string case_name;
  std::string default_mesh_path;
  const char* case_file = nullptr;
  double gamma = mfem::dg_igr::EulerDefaults::kGamma;
  bool use_igr = true;
  bool need_sigma = true;
  bool need_h1 = false;

  CaseOptions options;
  IGROptions igr_defaults;
  VisualizationConfig vis;
  VTKConfig vtk;

  std::function<void(CaseConfig&)> configure_case;
  std::function<void(mfem::OptionsParser&, CaseOptions&)> add_options;
  std::function<std::unique_ptr<mfem::VectorCoefficient>(
      const CaseOptions&, const ExampleContext&, double gamma)> make_ic;
  std::function<BoundaryMarkerMap(ExampleContext&)> build_markers;
  std::function<void(const CaseOptions&,
                     const ExampleContext&,
                     EulerBoundaryWiring<NumericalFlux>&,
                     const BoundaryMarkerMap&)> apply_bc;
  std::function<void(const CaseOptions&, ExampleContext&, EulerStateBundle&)> after_init;
  std::function<int(const CaseOptions&,
                    const ExampleContext&,
                    const EulerStateBundle&,
                    mfem::VectorCoefficient&)> postprocess;
};

template <typename CaseOptions, typename NumericalFlux>
struct RunSpec
{
  CaseOptions options;
  CommonOptions common_opts;
  IGROptions igr_opts;
  RunConfig run_cfg;
  ExampleContext* ctx = nullptr;
  EulerStateBundle* state = nullptr;
  const mfem::VectorCoefficient* ic = nullptr;
};

struct Solver
{
  template <typename CaseOptions, typename NumericalFlux>
  static int solve(const CaseSpec<CaseOptions, NumericalFlux>& spec,
                   int argc, char** argv)
  {
    mfem::Mpi::Init(argc, argv);
    mfem::Hypre::Init();

    CaseConfig case_cfg = MakeCaseConfig(
        spec.case_name, spec.default_mesh_path,
        spec.case_file ? spec.case_file : "", spec.gamma);
    if (spec.configure_case)
    {
      spec.configure_case(case_cfg);
    }
    case_cfg.rk3_config.gamma = case_cfg.gamma;
    case_cfg.defaults.visualization = spec.vis.enabled;
    case_cfg.defaults.vtk_save = spec.vtk.enabled;

    CaseOptions case_opts = spec.options;
    CommonOptions cli_opts = case_cfg.defaults;
    IGROptions cli_igr_opts = spec.igr_defaults;

    mfem::OptionsParser args(argc, argv);
    const std::string mesh_help = std::string("MFEM mesh file to use. Defaults to ")
                                  + DefaultCaseMeshPath(case_cfg) + ".";
    AddCommonOptions(args, cli_opts, mesh_help);
    AddIGROptions(args, cli_igr_opts);
    if (spec.add_options)
    {
      spec.add_options(args, case_opts);
    }
    args.ParseCheck();

    const RunConfig run_cfg = MakeRunConfig(case_cfg, cli_opts, cli_igr_opts);
    bool visualization = run_cfg.opts.visualization;
    std::cout.precision(spec.vis.precision);

    const bool need_sigma = spec.use_igr || spec.need_sigma;
    const bool need_h1 = spec.need_h1 || spec.vis.h1_diagnostics;
    ExampleContext ctx = BuildExampleContext(run_cfg, run_cfg.igr_opts.space,
                                             need_sigma, need_h1);
    if (need_sigma)
    {
      MFEM_VERIFY(ctx.spaces.sigma_fes != nullptr, "Sigma space was not created.");
    }
    if (need_h1 || spec.vis.h1_diagnostics)
    {
      MFEM_VERIFY(ctx.spaces.h1_vfes != nullptr, "H1 velocity space was not created.");
    }

    mfem::RK3SSPWithLimiter rk3 = MakeRK3Stepper(
        ctx.spaces.vfes, ctx.dim, case_cfg.rk3_config);
    ConfigureLimiterPipeline(rk3, case_cfg.limiter_pipeline,
                             ctx.spaces.vfes, ctx.dim, case_cfg.gamma);

    const double rho_floor = case_cfg.rk3_config.rho_min;
    std::unique_ptr<dg_igr::IGRSolver> igr;
    double vtk_c_alpha = spec.vtk.c_alpha_value;
    if (spec.use_igr)
    {
      const AlphaInfo alpha_info = ComputeAlphaInfo(ctx.Mesh(),
                                                    run_cfg.igr_opts.c_alpha_user);
      vtk_c_alpha = std::isnan(vtk_c_alpha) ? alpha_info.c_alpha : vtk_c_alpha;
      igr = std::make_unique<dg_igr::IGRSolver>(
          MakeIGRSolver(ctx.spaces, ctx.dim, ctx.num_equations,
                        run_cfg, rho_floor, alpha_info));
    }

    MFEM_ASSERT(ctx.spaces.fes.GetOrdering() == mfem::Ordering::byNODES, "");

    HYPRE_BigInt glob_size = ctx.spaces.vfes.GlobalTrueVSize();
    if (mfem::Mpi::Root())
    {
      std::cout << "Number of unknowns: " << glob_size << std::endl;
    }

    MFEM_VERIFY(spec.make_ic, "CaseSpec is missing an initial condition builder.");
    std::unique_ptr<mfem::VectorCoefficient> ic =
        spec.make_ic(case_opts, ctx, case_cfg.gamma);
    MFEM_VERIFY(ic, "CaseSpec initial condition builder returned null.");

    EulerStateBundle state_bundle(ctx, case_cfg.gamma);
    if (igr)
    {
      state_bundle.AttachIGR(*igr);
    }
    state_bundle.ProjectIC(*ic);

    std::unique_ptr<H1Diagnostics> h1_diag;
    if (spec.vis.h1_diagnostics && run_cfg.opts.visualization)
    {
      h1_diag = std::make_unique<H1Diagnostics>(
          *ctx.spaces.h1_vfes, state_bundle.sol, ctx.dim, rho_floor);
    }

    if (spec.after_init)
    {
      spec.after_init(case_opts, ctx, state_bundle);
    }

    SaveInitialEulerState(ctx, state_bundle.sol, spec.vis.precision);

    MFEM_VERIFY(spec.build_markers, "CaseSpec is missing a boundary marker builder.");
    const BoundaryMarkerMap bdr_markers = spec.build_markers(ctx);

    const auto on_open = [&]()
    {
      if (h1_diag)
      {
        h1_diag->Update();
      }
    };

    MFEM_VERIFY(spec.apply_bc, "CaseSpec is missing boundary wiring.");
    if (igr)
    {
      auto physics = MakeEulerIGRPhysics<NumericalFlux>(
          ctx, case_cfg.gamma, *igr, bdr_markers,
          [&](auto& bc, const auto& markers)
          {
            spec.apply_bc(case_opts, ctx, bc, markers);
          });

      if (!RunEulerIGRCaseWithIO(
              ctx, case_cfg, run_cfg, state_bundle, *igr, rk3, physics.op,
              visualization, spec.vis.precision, on_open,
              vtk_c_alpha, spec.vtk.print_dataset_name))
      {
        return 2;
      }
    }
    else
    {
      mfem::EulerFlux base_flux(ctx.dim, case_cfg.gamma);
      NumericalFlux numerical_flux(base_flux);
      mfem::DGHyperbolicConservationLaws op =
          MakeDGOperator(ctx, numerical_flux);
      EulerBoundaryWiring<NumericalFlux> bc(
          op, numerical_flux, ctx.run_cfg.int_order_offset, ctx.dim);
      spec.apply_bc(case_opts, ctx, bc, bdr_markers);

      if (!RunEulerCaseWithIO(
              ctx, case_cfg, run_cfg, state_bundle, rk3, op,
              visualization, spec.vis.precision,
              spec.vtk.print_dataset_name))
      {
        return 2;
      }
    }

    SaveFinalEulerState(ctx, state_bundle.sol, spec.vis.precision);

    if (spec.postprocess)
    {
      const int code = spec.postprocess(case_opts, ctx, state_bundle, *ic);
      if (code != 0)
      {
        return code;
      }
    }

    return 0;
  }
};

} // namespace dg_igr_examples

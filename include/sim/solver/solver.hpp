#pragma once

/**
 * @file sim/solver/solver.hpp
 * @brief Euler-specific solver orchestration for sim-layer cases.
 */

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>

#include "mfem.hpp"

#include "physics/igr/igr_coupling.hpp"

#include "sim/case/case_spec.hpp"
#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/context/mesh_build.hpp"
#include "sim/io/save_state.hpp"
#include "sim/io/vis_helpers.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/solver/solver_helpers.hpp"
#include "sim/time/time_defaults.hpp"
#include "sim/wiring/bc_wiring.hpp"
#include "sim/wiring/dg_operator.hpp"
#include "sim/wiring/dgsem_flux_config.hpp"
#include "sim/wiring/igr_physics.hpp"

namespace dg_igr_examples
{

namespace detail
{

inline void VerifyDGSEMElementGeometry(const mfem::Mesh& mesh)
{
  for (int i = 0; i < mesh.GetNE(); ++i)
  {
    const mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(i);
    if (geom == mfem::Geometry::SEGMENT || geom == mfem::Geometry::SQUARE)
    {
      continue;
    }
    MFEM_VERIFY(false,
                "DGSEM v1 supports tensor-product meshes only (segment/quad). "
                "Found geometry: " << mfem::Geometry::Name[geom] << ".");
  }
}

template <typename NumericalFlux, typename BcSetup, typename Func>
bool WithEulerOperator(ExampleContext& ctx,
                       double gamma,
                       dg_igr::IGRSolver* igr,
                       mfem::ParGridFunction& sol,
                       const BoundaryMarkerMap& markers,
                       BcSetup&& bc_setup,
                       Func&& func)
{
  if (igr)
  {
    auto physics = MakeEulerIGRPhysics<NumericalFlux>(
        ctx, gamma, *igr, markers, std::forward<BcSetup>(bc_setup));
    mfem::EulerWithIGR wrapped(physics.op, *igr, sol);
    return func(wrapped);
  }

  mfem::EulerFlux base_flux(ctx.dim, gamma);
  NumericalFlux numerical_flux(base_flux);
  ConfigureDGSEMSurfaceFlux(numerical_flux, ctx.run_cfg, gamma);
  mfem::DGHyperbolicConservationLaws op =
      MakeDGOperator(ctx, numerical_flux);
  EulerBoundaryWiring<NumericalFlux> bc(
      op, numerical_flux, ctx.run_cfg.int_order_offset, ctx.dim);
  bc_setup(bc, markers);
  return func(op);
}

} // namespace detail

/// @brief Run Euler (with or without IGR) using the shared IO/time-loop wiring.
template <typename OperatorType, typename OnOpenCallback>
inline bool RunEulerCase(ExampleContext& ctx,
                               const CaseConfig& case_cfg,
                               const RunConfig& run_cfg,
                               EulerStateBundle& bundle,
                               mfem::RK3SSPWithLimiter& rk3,
                               OperatorType& op,
                               bool& visualization,
                               const TimeLoopCallbacks& callbacks,
                               int precision,
                               OnOpenCallback&& on_open_callback,
                               double c_alpha_value =
                                  std::numeric_limits<double>::quiet_NaN(),
                               bool print_dataset_name = false)
{
  UpdateEulerOutputFields(bundle.field_list, bundle.sol, /*warm_start=*/false);

  EulerVisWindows glvis_windows;
  if (visualization)
  {
    const bool ok =
        bundle.field_list.igr
            ? OpenEulerIGRGLVisWindows(glvis_windows, bundle.field_list,
                                       ctx.Mesh(),
                                       mfem::Mpi::WorldSize(),
                                       mfem::Mpi::WorldRank(),
                                       precision)
            : OpenEulerGLVisWindows(glvis_windows, bundle.field_list,
                                    ctx.Mesh(),
                                    mfem::Mpi::WorldSize(),
                                    mfem::Mpi::WorldRank(),
                                    precision);
    if (ok)
    {
      on_open_callback();
    }
    else
    {
      visualization = false;
      if (mfem::Mpi::Root())
      {
        std::cout << "Unable to connect to GLVis. Visualization disabled.\n";
      }
    }
  }

  const std::string dc_name =
      MakeCaseVTKDatasetName(case_cfg, run_cfg, c_alpha_value);
  auto paraview_dc =
      std::make_unique<mfem::ParaViewDataCollection>(dc_name, &ctx.Mesh());
  if (run_cfg.opts.vtk_save)
  {
    RegisterEulerVTKFields(*paraview_dc, bundle.field_list);
    const mfem::io::VTKOutputConfig vtk_cfg = MakeCaseVTKConfig(case_cfg);
    ApplyCaseVTKConfig(*paraview_dc, vtk_cfg, run_cfg.order);
    paraview_dc->SetCycle(0);
    paraview_dc->SetTime(0.0);
    paraview_dc->Save();
    if (callbacks.on_vtk_save)
    {
      VTKSaveInfo info;
      info.cycle = 0;
      info.step = 0;
      info.t = 0.0;
      info.is_initial = true;
      info.is_final = false;
      callbacks.on_vtk_save(info);
    }
    if (mfem::Mpi::Root())
    {
      if (print_dataset_name)
      {
        std::cout << "VTK dataset: " << dc_name << "\n";
      }
    }
  }

  mfem::tic_toc.Clear();
  mfem::tic_toc.Start();

  TimeLoopIO io;
  io.glvis_windows = &glvis_windows;
  io.visualization = &visualization;
  io.paraview = paraview_dc.get();

  rk3.Init(op);
  TimeLoopConfig loop_cfg = MakeTimeLoopConfig(run_cfg);
  if (run_cfg.opts.vtk_save)
  {
    loop_cfg.save_index_offset = 1;
  }
  const int num_procs = mfem::Mpi::WorldSize();
  const int my_rank = mfem::Mpi::WorldRank();
  const TimeLoopHooks hooks =
      MakeEulerTimeLoopHooks(ctx, bundle, num_procs, my_rank, io, callbacks);
  const TimeLoopResult result =
      RunTimeLoop(op, ctx.Mesh(), bundle.sol, rk3, loop_cfg, hooks);
  if (result.non_finite_count > 0)
  {
    return false;
  }

  mfem::tic_toc.Stop();
  if (mfem::Mpi::Root())
  {
    const double elapsed = mfem::tic_toc.RealTime();
    std::cout << " done, " << elapsed << "s." << std::endl;
    if (callbacks.on_summary)
    {
      const RunSummary summary = BuildRunSummary(ctx, result, elapsed, num_procs);
      callbacks.on_summary(summary);
    }
  }
  return true;
}

template <typename OperatorType>
inline bool RunEulerCase(ExampleContext& ctx,
                               const CaseConfig& case_cfg,
                               const RunConfig& run_cfg,
                               EulerStateBundle& bundle,
                               mfem::RK3SSPWithLimiter& rk3,
                               OperatorType& op,
                               bool& visualization,
                               const TimeLoopCallbacks& callbacks,
                               int precision,
                               double c_alpha_value =
                                  std::numeric_limits<double>::quiet_NaN(),
                               bool print_dataset_name = false)
{
  return RunEulerCase(ctx, case_cfg, run_cfg, bundle, rk3, op,
                      visualization, callbacks, precision, []() {},
                      c_alpha_value, print_dataset_name);
}

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
    case_cfg.defaults.dg_backend = DGBackendToString(spec.dg_backend_default);

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
    if (run_cfg.dg_backend == DGBackend::DGSEM)
    {
      detail::VerifyDGSEMElementGeometry(ctx.Mesh());
    }
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
    rk3.EnableStageNanScan(run_cfg.opts.nan_scan);

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

    EulerStateBundle state_bundle(ctx, case_cfg.gamma, rho_floor);
    if (igr)
    {
      state_bundle.AttachIGR(*igr);
    }
    state_bundle.ProjectIC(*ic);

    TimeLoopCallbacks callbacks = spec.make_callbacks
        ? spec.make_callbacks(ctx, state_bundle)
        : MakeDefaultTimeLoopCallbacks(spec.callback_defaults);

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
    auto bc_setup = [&](auto& bc, const auto& markers)
    {
      spec.apply_bc(case_opts, ctx, bc, markers);
    };

    const bool ok = detail::WithEulerOperator<NumericalFlux>(
        ctx, case_cfg.gamma, igr.get(), state_bundle.sol, bdr_markers, bc_setup,
        [&](auto& op)
        {
          return RunEulerCase(
              ctx, case_cfg, run_cfg, state_bundle, rk3, op,
              visualization, callbacks, spec.vis.precision, on_open,
              vtk_c_alpha, spec.vtk.print_dataset_name);
        });

    if (!ok)
    {
      return 2;
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

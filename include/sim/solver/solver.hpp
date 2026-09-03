#pragma once

/**
 * @file sim/solver/solver.hpp
 * @brief Euler-specific solver orchestration for sim-layer cases.
 */

#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "mfem.hpp"

#include "physics/igr/igr_coupling.hpp"

#include "sim/case/case_spec.hpp"
#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/context/mesh_build.hpp"
#include "sim/io/save_state.hpp"
#include "sim/io/run_metadata.hpp"
#include "sim/io/vis_helpers.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/io/vtk_restart.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/solver/solver_helpers.hpp"
#include "sim/time/time_defaults.hpp"
#include "sim/wiring/bc_wiring.hpp"
#include "sim/wiring/dg_operator.hpp"
#include "sim/wiring/dgsem_flux_config.hpp"
#include "sim/wiring/igr_physics.hpp"

namespace dg_sim
{

namespace detail
{

inline std::string MakeEulerOutputDatasetName(
  const CaseConfig& case_cfg,
  const RunConfig& run_cfg,
  double c_alpha_value,
  const VTKRestartInfo* restart_info)
{
  std::string name =
    MakeCaseVTKDatasetName(case_cfg, run_cfg, c_alpha_value);
  if (restart_info)
  {
    name += "_restart_c" + std::to_string(restart_info->cycle) + "_s" +
            std::to_string(restart_info->step);
  }
  return name;
}

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
                "Found geometry: "
                  << mfem::Geometry::Name[geom] << ".");
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
    auto physics =
      MakeEulerIGRPhysics<NumericalFlux>(ctx,
                                         gamma,
                                         *igr,
                                         markers,
                                         std::forward<BcSetup>(bc_setup));
    mfem::EulerWithIGR wrapped(physics.op, *igr, sol);
    return func(wrapped);
  }

  mfem::EulerFlux base_flux(ctx.dim, gamma);
  NumericalFlux numerical_flux(base_flux);
  ConfigureDGSEMSurfaceFlux(numerical_flux, ctx.run_cfg, gamma);
  mfem::DGHyperbolicConservationLaws op = MakeDGOperator(ctx, numerical_flux);
  EulerBoundaryWiring<NumericalFlux> bc(
    op,
    numerical_flux,
    ctx.run_cfg.case_config.int_order_offset,
    ctx.dim,
    gamma);
  bc_setup(bc, markers);
  return func(op);
}

enum class SolverRequestAdmission
{
  Accepted,
  ResolutionFailure,
  Mismatch,
  CollectiveFailure
};

/// @brief Admit one identical, resolved request on every MPI rank.
inline SolverRequestAdmission
AdmitSolverRequest(const SolverRequestResolution& request)
{
  constexpr int tuple_size = 8;
  const int local[tuple_size] = {
    static_cast<int>(request.status),
    static_cast<int>(request.discretization),
    static_cast<int>(request.discretization_provenance),
    request.dg_backend ? static_cast<int>(*request.dg_backend) : -1,
    static_cast<int>(request.device_backend),
    static_cast<int>(request.hdg_cuda_engine),
    static_cast<int>(request.regularization),
    static_cast<int>(request.time_integrator)};
  int minimum[tuple_size] = {};
  int maximum[tuple_size] = {};
  const int minimum_status =
    MPI_Allreduce(local, minimum, tuple_size, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  const int maximum_status =
    MPI_Allreduce(local, maximum, tuple_size, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (minimum_status != MPI_SUCCESS || maximum_status != MPI_SUCCESS)
  {
    return SolverRequestAdmission::CollectiveFailure;
  }
  for (int i = 0; i < tuple_size; ++i)
  {
    if (minimum[i] != maximum[i])
    {
      return SolverRequestAdmission::Mismatch;
    }
  }
  return minimum[0] ==
           static_cast<int>(SolverRequestResolutionStatus::Resolved)
           ? SolverRequestAdmission::Accepted
           : SolverRequestAdmission::ResolutionFailure;
}

} // namespace detail

/// @brief Run Euler (with or without IGR) using the shared IO/time-loop wiring.
template <typename OperatorType, typename OnOpenCallback>
inline bool
RunEulerCase(ExampleContext& ctx,
             const CaseConfig& case_cfg,
             const RunConfig& run_cfg,
             EulerStateBundle& bundle,
             mfem::RK3SSPWithLimiter& rk3,
             OperatorType& op,
             bool& glvis_enabled,
             const TimeLoopCallbacks& callbacks,
             int precision,
             OnOpenCallback&& on_open_callback,
             RunSummary* summary_out = nullptr,
             double c_alpha_value = std::numeric_limits<double>::quiet_NaN(),
             bool print_dataset_name = false,
             const VTKRestartInfo* restart_info = nullptr)
{
  const mfem::real_t initial_time = restart_info ? restart_info->time : 0.0;
  MFEM_VERIFY(!restart_info ||
                restart_info->step <= std::numeric_limits<int>::max(),
              "Restart step exceeds the supported time-loop counter range.");
  const int initial_step = restart_info ? static_cast<int>(restart_info->step) : 0;
  const int initial_cycle = restart_info ? restart_info->cycle : 0;
  rk3.Init(op);
  rk3.SetStepCount(initial_step);
  auto* sigma_provider = dynamic_cast<mfem::IGRSigmaStateProvider*>(&op);

  EulerOutputFieldTiming initial_field_timing;
  double initial_sigma_prepare_s = 0.0;
  mfem::IGRSigmaPreparationResult initial_sigma_preparation;
  if (sigma_provider &&
      (restart_info || run_cfg.opts.vtk_save || glvis_enabled))
  {
    const auto prepare_start = std::chrono::steady_clock::now();
    initial_sigma_preparation = sigma_provider->PrepareSigmaForOutput(
      bundle.sol,
      rk3.GetStateGeneration(),
      initial_step,
      initial_time,
      /*reusable_stage0=*/true,
      /*cold_start=*/restart_info != nullptr,
      /*certify_residual=*/restart_info != nullptr);
    initial_sigma_prepare_s = SecondsSince(prepare_start);
  }
  if (restart_info && sigma_provider && mfem::Mpi::Root())
  {
    std::cout << "[vtk-restart] event=sigma-initialization"
              << " solved=" << (initial_sigma_preparation.solved ? 1 : 0)
              << " cold_start=1"
              << " reusable_stage0="
              << (initial_sigma_preparation.reusable_stage0 ? 1 : 0)
              << " independent_residual_certified="
              << (initial_sigma_preparation.independent_residual_certified ? 1
                                                                          : 0)
              << " independent_residual_rel_rhs="
              << initial_sigma_preparation.independent_residual_rel_rhs
              << "\n";
  }
  UpdateEulerOutputFields(bundle.field_list, &initial_field_timing);
  initial_field_timing.igr_update_s = initial_sigma_prepare_s;
  initial_field_timing.total_s += initial_sigma_prepare_s;
  const double initial_update_fields_s = initial_field_timing.total_s;
  PrintVTKOutputTiming("initial-fields",
                       initial_cycle,
                       initial_step,
                       initial_time,
                       /*is_initial=*/restart_info == nullptr,
                       /*is_final=*/false,
                       initial_update_fields_s,
                       /*paraview_save_s=*/0.0,
                       ctx.Mesh().GetComm());
  PrintVTKOutputFieldTiming("initial-fields",
                            initial_cycle,
                            initial_step,
                            initial_time,
                            /*is_initial=*/restart_info == nullptr,
                            /*is_final=*/false,
                            initial_field_timing,
                            ctx.Mesh().GetComm());

  EulerVisWindows glvis_windows;
  if (glvis_enabled)
  {
    const bool ok = bundle.field_list.igr
                      ? OpenEulerIGRGLVisWindows(glvis_windows,
                                                 bundle.field_list,
                                                 ctx.Mesh(),
                                                 mfem::Mpi::WorldSize(),
                                                 mfem::Mpi::WorldRank(),
                                                 precision)
                      : OpenEulerGLVisWindows(glvis_windows,
                                              bundle.field_list,
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
      glvis_enabled = false;
      if (mfem::Mpi::Root())
      {
        std::cout << "Unable to connect to GLVis. Visualization disabled.\n";
      }
    }
  }

  const std::string dc_name = detail::MakeEulerOutputDatasetName(
    case_cfg, run_cfg, c_alpha_value, restart_info);
  auto paraview_dc =
    std::make_unique<mfem::ParaViewDataCollection>(dc_name, &ctx.Mesh());
  const mfem::io::VTKOutputConfig vtk_cfg = MakeCaseVTKConfig(case_cfg);
  const std::string dataset_dir =
    (std::filesystem::path(vtk_cfg.prefix_path) / dc_name).string();
  if (run_cfg.opts.vtk_save)
  {
    RegisterEulerVTKFields(*paraview_dc, bundle.field_list);
    ApplyCaseVTKConfig(*paraview_dc, vtk_cfg, run_cfg.opts.order);
    if (!restart_info)
    {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      const auto initial_save_start = std::chrono::steady_clock::now();
      paraview_dc->Save();
      const double initial_save_s = SecondsSince(initial_save_start);
      WriteVTKRestartManifest(dataset_dir,
                              dc_name,
                              0,
                              0.0,
                              0,
                              run_cfg,
                              ctx,
                              bundle.field_list.sigma != nullptr,
                              paraview_dc->GetCompressionLevel());
      PrintVTKOutputTiming("initial-save",
                           /*cycle=*/0,
                           /*step=*/0,
                           /*t=*/0.0,
                           /*is_initial=*/true,
                           /*is_final=*/false,
                           /*update_fields_s=*/0.0,
                           initial_save_s,
                           ctx.Mesh().GetComm());
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
  io.glvis_enabled = &glvis_enabled;
  io.paraview = paraview_dc.get();
  if (run_cfg.opts.vtk_save)
  {
    io.after_vtk_save = [&run_cfg,
                         &ctx,
                         dataset_dir,
                         dc_name,
                         sigma_saved = bundle.field_list.sigma != nullptr,
                         vtk_compression_level =
                           paraview_dc->GetCompressionLevel()](
                          int step, mfem::real_t t, int cycle)
    {
      WriteVTKRestartManifest(dataset_dir,
                              dc_name,
                              cycle,
                              t,
                              step,
                              run_cfg,
                              ctx,
                              sigma_saved,
                              vtk_compression_level);
    };
  }
  if (sigma_provider)
  {
    io.prepare_sigma = [sigma_provider, &bundle, &rk3](int,
                                                       mfem::real_t t,
                                                       bool is_final)
    {
      const auto prepare_start = std::chrono::steady_clock::now();
      sigma_provider->PrepareSigmaForOutput(
        bundle.sol,
        rk3.GetStateGeneration(),
        static_cast<int>(rk3.GetStepCount()),
        t,
        /*reusable_stage0=*/!is_final);
      return SecondsSince(prepare_start);
    };
  }

  TimeLoopConfig loop_cfg = MakeTimeLoopConfig(run_cfg);
  if (run_cfg.opts.vtk_save)
  {
    loop_cfg.save_index_offset = restart_info ? restart_info->cycle + 1 : 1;
  }
  loop_cfg.initial_time = initial_time;
  loop_cfg.initial_step = initial_step;
  loop_cfg.next_save_time =
    (run_cfg.opts.vtk_save && run_cfg.opts.dt_save > 0.0)
      ? initial_time + run_cfg.opts.dt_save
      : -1.0;
  const int num_procs = mfem::Mpi::WorldSize();
  const int my_rank = mfem::Mpi::WorldRank();
  const TimeLoopHooks hooks =
    MakeEulerTimeLoopHooks(ctx,
                           bundle,
                           num_procs,
                           my_rank,
                           io,
                           callbacks,
                           initial_step);
  const TimeLoopResult result =
    RunTimeLoop(op, ctx.Mesh(), bundle.sol, rk3, loop_cfg, hooks);
  if (sigma_provider && mfem::Mpi::Root())
  {
    std::cout << "[igr-sigma-lifecycle] event=run-summary"
              << " prefetch_solves="
              << sigma_provider->GetSigmaPrefetchSolveCount()
              << " stage0_reuses="
              << sigma_provider->GetSigmaStage0ReuseCount()
              << " output_only_solves="
              << sigma_provider->GetSigmaOutputOnlySolveCount()
              << " prefetch_invalidations="
              << sigma_provider->GetSigmaPrefetchInvalidationCount() << "\n";
  }
  if (result.non_finite_count > 0 || result.dt_min_reached)
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
      const RunSummary summary =
        BuildRunSummary(ctx, result, elapsed, num_procs, bundle.field_list.igr);
      if (summary_out)
      {
        *summary_out = summary;
      }
      callbacks.on_summary(summary);
    }
    else if (summary_out)
    {
      *summary_out =
        BuildRunSummary(ctx, result, elapsed, num_procs, bundle.field_list.igr);
    }
  }
  return true;
}

template <typename OperatorType>
inline bool
RunEulerCase(ExampleContext& ctx,
             const CaseConfig& case_cfg,
             const RunConfig& run_cfg,
             EulerStateBundle& bundle,
             mfem::RK3SSPWithLimiter& rk3,
             OperatorType& op,
             bool& glvis_enabled,
             const TimeLoopCallbacks& callbacks,
             int precision,
             RunSummary* summary_out = nullptr,
             double c_alpha_value = std::numeric_limits<double>::quiet_NaN(),
             bool print_dataset_name = false,
             const VTKRestartInfo* restart_info = nullptr)
{
  return RunEulerCase(
    ctx,
    case_cfg,
    run_cfg,
    bundle,
    rk3,
    op,
    glvis_enabled,
    callbacks,
    precision,
    []() {},
    summary_out,
    c_alpha_value,
    print_dataset_name,
    restart_info);
}

struct Solver
{
  template <typename CaseOptions, typename NumericalFlux>
  static int
  solve(const CaseSpec<CaseOptions, NumericalFlux>& spec, int argc, char** argv)
  {
    mfem::Mpi::Init(argc, argv);
    mfem::Hypre::Init();

    CaseConfig case_cfg = MakeCaseConfig(spec.case_name,
                                         spec.default_mesh_path,
                                         spec.case_file ? spec.case_file : "",
                                         spec.gamma);
    // Apply CaseSpec defaults first; configure_case can then override them.
    case_cfg.defaults.glvis = spec.glvis.enabled;
    case_cfg.defaults.vtk_save = spec.vtk.enabled;
    case_cfg.defaults.dg_backend = DGBackendToString(spec.dg_backend_default);
    if (spec.configure_case)
    {
      spec.configure_case(case_cfg);
    }
    case_cfg.rk3_config.gamma = case_cfg.gamma;

    CaseOptions case_opts = spec.options;
    CommonOptions cli_opts = case_cfg.defaults;
    IGROptions cli_igr_opts = spec.igr_defaults;

    mfem::OptionsParser args(argc, argv);
    const std::string mesh_help =
      std::string("MFEM mesh file to use. Defaults to ") +
      DefaultCaseMeshPath(case_cfg) + ".";
    AddCommonOptions(args, cli_opts, mesh_help);
    AddIGROptions(args, cli_igr_opts);
    if (spec.add_options)
    {
      spec.add_options(args, case_opts);
    }
    args.ParseCheck();
    const SolverRequestResolution request_resolution = ResolveSolverRequest(
      case_cfg.defaults.dg_backend,
      spec.use_igr,
      cli_opts.discretization,
      cli_opts.dg_backend,
      cli_opts.regularization,
      cli_opts.time_integrator,
      HasDiscretizationOptionOverride(argc, argv),
      HasDGBackendOptionOverride(argc, argv),
      HasRegularizationOptionOverride(argc, argv),
      HasTimeIntegratorOptionOverride(argc, argv),
      cli_opts.device,
      cli_opts.hdg_cuda_engine);
    const detail::SolverRequestAdmission request_admission =
      detail::AdmitSolverRequest(request_resolution);
    if (request_admission != detail::SolverRequestAdmission::Accepted)
    {
      if (mfem::Mpi::Root())
      {
        if (request_admission ==
            detail::SolverRequestAdmission::ResolutionFailure)
        {
          std::cerr << request_resolution.message << "\n";
        }
        else if (request_admission == detail::SolverRequestAdmission::Mismatch)
        {
          std::cerr << "Solver request differs across MPI ranks; refusing "
                       "runtime construction.\n";
        }
        else
        {
          std::cerr << "Unable to collectively admit the solver request.\n";
        }
      }
      return 1;
    }
    cli_opts.discretization =
      DiscretizationToString(request_resolution.discretization);
    cli_opts.device = DeviceBackendToString(request_resolution.device_backend);
    cli_opts.hdg_cuda_engine =
      HdgCudaEngineToString(request_resolution.hdg_cuda_engine);
    cli_opts.regularization =
      RegularizationToString(request_resolution.regularization);
    cli_opts.time_integrator =
      TimeIntegratorToString(request_resolution.time_integrator);
    MFEM_VERIFY(!cli_opts.restart_vtk.empty() ||
                  (cli_opts.restart_cycle < 0 && cli_opts.restart_step < 0),
                "--restart-cycle/--restart-step require --restart-vtk.");
    if (request_resolution.discretization == Discretization::HDG)
    {
#if defined(DG_IGR_HAS_HDG_MAIN_BACKEND) && DG_IGR_HAS_HDG_MAIN_BACKEND
      const bool c_alpha_overridden = HasCAlphaOptionOverride(argc, argv);
      const bool hdg_igr_alpha_overridden =
        HasHdgIgrAlphaOptionOverride(argc, argv);
      if (c_alpha_overridden && hdg_igr_alpha_overridden &&
          cli_igr_opts.c_alpha_user != cli_opts.hdg_igr_alpha)
      {
        if (mfem::Mpi::Root())
        {
          std::cerr << "Conflicting HDG IGR C_alpha values: --c-alpha="
                    << cli_igr_opts.c_alpha_user
                    << " and --hdg-igr-alpha=" << cli_opts.hdg_igr_alpha
                    << ". Use one spelling or supply the same value to both.\n";
        }
        return 1;
      }
      if (c_alpha_overridden)
      {
        cli_opts.hdg_igr_alpha = cli_igr_opts.c_alpha_user;
      }
      MFEM_VERIFY(static_cast<bool>(spec.make_hdg_case_data),
                  "Case '" << case_cfg.case_name
                            << "' does not provide an HDG backend adapter.");
      if (!spec.make_hdg_case_data)
      {
        return 1;
      }
      const hdg::HdgEulerCaseDataFactory make_hdg_case_data =
        [&spec, &case_opts, &case_cfg, &cli_opts](const CaseContext& context)
      {
        return spec.make_hdg_case_data(case_opts, case_cfg, cli_opts, context);
      };
      const std::optional<dg_igr::IGRAlphaLengthScalePolicy>
        hdg_alpha_length_policy_override =
          cli_igr_opts.alpha_length_policy.empty()
            ? std::nullopt
            : std::optional<dg_igr::IGRAlphaLengthScalePolicy>{
                ParseIGRAlphaLengthScalePolicy(
                  cli_igr_opts.alpha_length_policy)};
      const hdg::HdgEulerMainRequest hdg_request{
        case_cfg,
        cli_opts,
        make_hdg_case_data,
        MPI_COMM_WORLD,
        hdg_alpha_length_policy_override};
#if defined(MFEM_USE_CUDA)
      // The CUDA HDG adapter returns before the legacy device construction
      // below.  Keep this device alive for its entire generic backend call so
      // it sees the same selected MFEM device lifetime as every other CUDA
      // execution route, without changing CPU/DG/DGSEM.
      if (request_resolution.device_backend == DeviceBackend::CUDA)
      {
        mfem::Device hdg_cuda_device("cuda");
        if (mfem::Mpi::Root())
        {
          std::cout << "[device] requested=cuda hdg_cuda_engine="
                    << cli_opts.hdg_cuda_engine << '\n';
          hdg_cuda_device.Print();
        }
        return hdg::RunEulerMainBackend(hdg_request);
      }
#endif
      return hdg::RunEulerMainBackend(hdg_request);
#else
      if (mfem::Mpi::Root())
      {
        std::cerr << "Case '" << case_cfg.case_name
                  << "' does not provide an HDG backend adapter.\n";
      }
      return 1;
#endif
    }

    MFEM_VERIFY(request_resolution.dg_backend,
                "DG/DGSEM selection requires a legacy DG backend mapping.");
    cli_opts.dg_backend =
      DGBackendToString(*request_resolution.dg_backend);
    ApplyIGRDerivedRuntimeDefaults(cli_opts,
                                   cli_igr_opts,
                                   HasExactDGFaceOptionOverride(argc, argv));

    const RunConfig run_cfg = MakeRunConfig(
      case_cfg,
      cli_opts,
      cli_igr_opts,
      request_resolution.discretization_provenance,
      request_resolution.regularization,
      request_resolution.time_integrator);
    const bool use_igr = run_cfg.regularization == Regularization::IGR;
    if (use_igr && mfem::Mpi::Root())
    {
      std::cout << "[igr-options] event=effective"
                << " exact_dg_face="
                << (run_cfg.igr_opts.exact_dg_face_operator ? "on" : "off")
                << " solver=" << run_cfg.igr_opts.linear_solver
                << " preconditioner=" << run_cfg.igr_opts.preconditioner
                << " nonconvergence_policy="
                << run_cfg.igr_opts.nonconvergence_policy
                << " sigma_cfl="
                << (run_cfg.igr_opts.sigma_aware_cfl ? "on" : "off")
                << " sigma_rusanov="
                << (run_cfg.igr_opts.sigma_aware_rusanov ? "on" : "off")
                << " stage_rtol_scale="
                << (run_cfg.igr_opts.stage_rtol_scale >= 0.0
                      ? run_cfg.igr_opts.stage_rtol_scale
                      : run_cfg.case_config.igr_config.stage_rtol_scale)
                << " lin_rtol="
                << (run_cfg.igr_opts.lin_rtol >= 0.0
                      ? run_cfg.igr_opts.lin_rtol
                      : run_cfg.case_config.igr_config.lin_rtol)
                << " lin_atol="
                << (run_cfg.igr_opts.lin_atol >= 0.0
                      ? run_cfg.igr_opts.lin_atol
                      : run_cfg.case_config.igr_config.lin_atol)
                << " lin_maxit="
                << (run_cfg.igr_opts.lin_maxit >= 0
                      ? run_cfg.igr_opts.lin_maxit
                      : run_cfg.case_config.igr_config.lin_maxit)
                << "\n";
    }
    if (run_cfg.device_backend == DeviceBackend::CUDA)
    {
#ifndef MFEM_USE_CUDA
      MFEM_VERIFY(false,
                  "Requested '--device cuda' but MFEM was built without CUDA "
                  "support.");
#endif
    }
    mfem::Device device(DeviceBackendToString(run_cfg.device_backend));
    if (mfem::Mpi::Root())
    {
      std::cout << "[device] requested="
                << DeviceBackendToString(run_cfg.device_backend) << "\n";
      device.Print();
    }
    bool glvis_enabled = run_cfg.opts.glvis;
    std::cout.precision(spec.glvis.precision);

    const bool need_sigma_space = use_igr || spec.need_sigma;
    const bool enable_h1_vis_diag =
      spec.diagnostics.h1_diagnostics &&
      (run_cfg.opts.glvis ||
       (run_cfg.opts.vtk_save && run_cfg.opts.vtk_h1_diagnostics));
    const bool enable_l2_diag =
      spec.diagnostics.l2_diagnostics && run_cfg.opts.vtk_save &&
      run_cfg.opts.vtk_l2_diagnostics;
    const bool enable_mpi_rank_diag =
      spec.diagnostics.mpi_rank_field && run_cfg.opts.vtk_save &&
      run_cfg.opts.vtk_mpi_rank_field;
    const bool enable_zhang_shu_diag =
      spec.diagnostics.zhang_shu_limiter &&
      run_cfg.opts.vtk_save &&
      run_cfg.opts.vtk_zhang_shu_diagnostics &&
      run_cfg.case_config.limiter_pipeline.zhang_shu;
    const bool need_h1 = spec.need_h1 || enable_h1_vis_diag;
    ExampleContext ctx = BuildExampleContext(run_cfg,
                                             run_cfg.igr_opts.space,
                                             need_sigma_space,
                                             need_h1);
    if (run_cfg.dg_backend == DGBackend::DGSEM)
    {
      detail::VerifyDGSEMElementGeometry(ctx.Mesh());
    }
    if (need_sigma_space)
    {
      MFEM_VERIFY(ctx.spaces.sigma_fes != nullptr,
                  "Sigma space was not created.");
    }
    if (need_h1)
    {
      MFEM_VERIFY(ctx.spaces.h1_vfes != nullptr,
                  "H1 velocity space was not created.");
    }

    mfem::RK3SSPWithLimiter rk3 =
      MakeRK3Stepper(ctx.spaces.vfes, ctx.dim,
                     run_cfg.case_config.rk3_config);
    if (run_cfg.dg_backend == DGBackend::DG)
    {
      const int dg_face_qord =
        2 * ctx.spaces.vfes.GetFE(0)->GetOrder() +
        run_cfg.case_config.int_order_offset;
      rk3.SetLimiterFaceQuadratureOrders({dg_face_qord});
    }
    ConfigureLimiterPipeline(rk3,
                             run_cfg.case_config.limiter_pipeline,
                             ctx.spaces.vfes,
                             ctx.dim,
                             run_cfg.case_config.gamma);
    rk3.EnableStageNanScan(run_cfg.opts.nan_scan);
    rk3.EnableStageAdmissibilityScan(run_cfg.opts.admissibility_scan);

    const double rho_floor = run_cfg.case_config.rk3_config.rho_min;
    std::unique_ptr<dg_igr::IGRSolver> igr;
    double vtk_c_alpha = spec.vtk.c_alpha_value;
    if (use_igr)
    {
      const auto alpha_policy = ResolveIGRAlphaLengthScalePolicy(run_cfg);
      const AlphaInfo alpha_info =
        ComputeAlphaInfo(ctx.Mesh(),
                         run_cfg.igr_opts.c_alpha_user,
                         alpha_policy);
      vtk_c_alpha = std::isnan(vtk_c_alpha) ? alpha_info.c_alpha : vtk_c_alpha;
      igr = std::make_unique<dg_igr::IGRSolver>(MakeIGRSolver(ctx.spaces,
                                                              ctx.dim,
                                                              ctx.num_equations,
                                                              run_cfg,
                                                              rho_floor,
                                                              alpha_info));
    }

    MFEM_ASSERT(ctx.spaces.fes.GetOrdering() == mfem::Ordering::byNODES, "");

    HYPRE_BigInt glob_size = ctx.spaces.vfes.GlobalTrueVSize();
    if (mfem::Mpi::Root())
    {
      std::cout << "Number of unknowns: " << glob_size << std::endl;
    }

    MFEM_VERIFY(spec.make_ic,
                "CaseSpec is missing an initial condition builder.");
    std::unique_ptr<mfem::VectorCoefficient> ic =
      spec.make_ic(case_opts, ctx, case_cfg.gamma);
    MFEM_VERIFY(ic, "CaseSpec initial condition builder returned null.");

    EulerStateBundle state_bundle(ctx, case_cfg.gamma, rho_floor);
    if (igr)
    {
      state_bundle.AttachIGR(*igr);
    }
    std::optional<VTKRestartInfo> restart_info;
    if (run_cfg.opts.restart_vtk.empty())
    {
      state_bundle.ProjectIC(*ic);
    }
    else
    {
      VTKRestartRequest request;
      request.path = run_cfg.opts.restart_vtk;
      request.cycle = run_cfg.opts.restart_cycle;
      request.legacy_step = run_cfg.opts.restart_step;
      restart_info = LoadEulerVTKRestart(
        request, run_cfg, ctx, state_bundle.sol, igr != nullptr);
      MFEM_VERIFY(run_cfg.opts.t_final > restart_info->time,
                  "-tf/--t-final must be greater than the restart time "
                    << restart_info->time << ".");
      if (mfem::Mpi::Root())
      {
        std::cout << "[vtk-restart] source=" << restart_info->pvd_path
                  << " cycle=" << restart_info->cycle
                  << " time=" << restart_info->time
                  << " step=" << restart_info->step
                  << " legacy=" << (restart_info->legacy ? 1 : 0)
                  << " ranks=" << mfem::Mpi::WorldSize()
                  << " reconstruction_max_abs="
                  << restart_info->reconstruction_max_abs
                  << " reconstruction_rel_l2="
                  << restart_info->reconstruction_rel_l2
                  << " mesh_point_max_abs="
                  << restart_info->mesh_point_max_abs
                  << " condition_inf="
                  << restart_info->interpolation_condition_inf
                  << " rho_min=" << restart_info->density_min
                  << " p_min=" << restart_info->pressure_min
                  << " sigma_policy=cold-recompute"
                  << " output_mode=new"
                  << " next_cycle=" << restart_info->cycle + 1
                  << " next_time=" << restart_info->time + run_cfg.opts.dt_save
                  << " load_seconds=" << restart_info->load_seconds << "\n";
      }
    }

    TimeLoopCallbacks callbacks =
      spec.make_callbacks
        ? spec.make_callbacks(ctx, state_bundle)
        : MakeDefaultTimeLoopCallbacks(spec.callback_defaults);
    RunSummary captured_summary{};
    bool captured_summary_valid = false;

    std::unique_ptr<H1Diagnostics> h1_diag;
    std::unique_ptr<L2Diagnostics> l2_diag;
    std::unique_ptr<MPIRankField> mpi_rank_field;
    std::unique_ptr<ZhangShuLimiterField> zhang_shu_field;
    if (enable_h1_vis_diag)
    {
      h1_diag = std::make_unique<H1Diagnostics>(*ctx.spaces.h1_vfes,
                                                state_bundle.sol,
                                                ctx.dim,
                                                rho_floor);
      state_bundle.field_list.h1_diag = h1_diag.get();
    }
    if (enable_l2_diag)
    {
      l2_diag = std::make_unique<L2Diagnostics>(ctx.spaces.fes,
                                                state_bundle.sol,
                                                ctx.dim,
                                                rho_floor);
      state_bundle.field_list.l2_diag = l2_diag.get();
    }
    if (enable_mpi_rank_diag)
    {
      mpi_rank_field = std::make_unique<MPIRankField>(ctx.Mesh());
      state_bundle.field_list.mpi_rank = mpi_rank_field.get();
    }
    if (enable_zhang_shu_diag)
    {
      zhang_shu_field = std::make_unique<ZhangShuLimiterField>(ctx.Mesh());
      state_bundle.field_list.zhang_shu = zhang_shu_field.get();
      rk3.SetZhangShuDiagnosticsFields(&zhang_shu_field->active,
                                       &zhang_shu_field->theta_rho,
                                       &zhang_shu_field->theta_p);
    }

    if (spec.after_init)
    {
      spec.after_init(case_opts, ctx, state_bundle);
    }

    const std::string output_dataset_name = detail::MakeEulerOutputDatasetName(
      case_cfg,
      run_cfg,
      vtk_c_alpha,
      restart_info ? &*restart_info : nullptr);

    std::string state_snapshot_dir;
    if (run_cfg.opts.save_state_files)
    {
      const mfem::io::VTKOutputConfig vtk_cfg = MakeCaseVTKConfig(case_cfg);
      state_snapshot_dir = (std::filesystem::path(vtk_cfg.prefix_path) /
                            output_dataset_name / "glvis_state")
                             .string();
      SaveInitialEulerState(ctx,
                            state_bundle.sol,
                            spec.glvis.precision,
                            "euler-mesh",
                            "euler",
                            state_snapshot_dir);
    }

    MFEM_VERIFY(spec.build_markers,
                "CaseSpec is missing a boundary marker builder.");
    const BoundaryMarkerMap bdr_markers = spec.build_markers(ctx);

    const auto on_open = [&h1_diag]()
    {
      if (h1_diag)
      {
        h1_diag->Update();
      }
    };

    MFEM_VERIFY(spec.apply_bc, "CaseSpec is missing boundary wiring.");
    auto bc_setup = [&spec, &case_opts, &ctx](auto& bc, const auto& markers)
    {
      spec.apply_bc(case_opts, ctx, bc, markers);
    };

    const bool ok = detail::WithEulerOperator<NumericalFlux>(
      ctx,
      case_cfg.gamma,
      igr.get(),
      state_bundle.sol,
      bdr_markers,
      bc_setup,
      [&ctx,
       &case_cfg,
       &run_cfg,
       &state_bundle,
       &rk3,
       &glvis_enabled,
       &callbacks,
       &spec,
       &on_open,
       &captured_summary,
       &restart_info,
       vtk_c_alpha](auto& op)
      {
        return RunEulerCase(ctx,
                            case_cfg,
                            run_cfg,
                            state_bundle,
                            rk3,
                            op,
                            glvis_enabled,
                            callbacks,
                            spec.glvis.precision,
                            on_open,
                            &captured_summary,
                            vtk_c_alpha,
                            spec.vtk.print_dataset_name,
                            restart_info ? &*restart_info : nullptr);
      });
    captured_summary_valid = ok;

    if (!ok)
    {
      WriteRunMetadataSidecar(case_cfg,
                              run_cfg,
                              captured_summary_valid ? &captured_summary
                                                     : nullptr,
                              igr.get(),
                              vtk_c_alpha,
                              argc,
                              argv,
                              /*success=*/false,
                              output_dataset_name,
                              restart_info ? &*restart_info : nullptr);
      return 2;
    }

    if (igr && igr->GetLinearSolveCalls() > 0 && mfem::Mpi::Root())
    {
      const double avg_linear_time =
        igr->GetTotalLinearSolveSeconds() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_assemble_time =
        igr->GetTotalAssembleSeconds() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_form_time =
        igr->GetTotalFormSystemSeconds() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_pc_setup_time =
        igr->GetTotalPCSetupSeconds() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const int total_linear_iters = igr->GetTotalLinearIterations();
      const double avg_krylov_operator_apply_count =
        igr->GetTotalKrylovOperatorApplyCount() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_krylov_preconditioner_apply_count =
        igr->GetTotalKrylovPreconditionerApplyCount() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_krylov_preconditioner_operator_apply_count =
        igr->GetTotalKrylovPreconditionerOperatorApplyCount() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double krylov_operator_apply_count_per_iter =
        (total_linear_iters > 0)
          ? (igr->GetTotalKrylovOperatorApplyCount() /
             static_cast<double>(total_linear_iters))
          : 0.0;
      const double krylov_preconditioner_apply_count_per_iter =
        (total_linear_iters > 0)
          ? (igr->GetTotalKrylovPreconditionerApplyCount() /
             static_cast<double>(total_linear_iters))
          : 0.0;
      const double krylov_preconditioner_operator_apply_count_per_iter =
        (total_linear_iters > 0)
          ? (igr->GetTotalKrylovPreconditionerOperatorApplyCount() /
             static_cast<double>(total_linear_iters))
          : 0.0;
      const double igr_total_pct =
        (captured_summary_valid && captured_summary.elapsed > 0.0)
          ? (100.0 * igr->GetTotalIGRWorkSeconds() / captured_summary.elapsed)
          : 0.0;
      const double igr_solve_pct =
        (captured_summary_valid && captured_summary.elapsed > 0.0)
          ? (100.0 * igr->GetTotalLinearSolveSeconds() /
             captured_summary.elapsed)
          : 0.0;
      const double igr_stage_total_pct =
        (igr->GetTotalUpdateCallSeconds() > 0.0)
          ? (100.0 * igr->GetTotalIGRWorkSeconds() /
             igr->GetTotalUpdateCallSeconds())
          : 0.0;
      const double igr_stage_solve_pct =
        (igr->GetTotalUpdateCallSeconds() > 0.0)
          ? (100.0 * igr->GetTotalLinearSolveSeconds() /
             igr->GetTotalUpdateCallSeconds())
          : 0.0;
      std::cout << "[igr-linear] event=run-summary"
                << " calls=" << igr->GetLinearSolveCalls()
                << " converged_calls=" << igr->GetTotalLinearConvergedCalls()
                << " production_calls="
                << igr->GetProductionLinearSolveCalls()
                << " production_converged_calls="
                << igr->GetProductionLinearConvergedCalls()
                << " output_only_calls="
                << igr->GetOutputOnlyLinearSolveCalls()
                << " output_only_converged_calls="
                << igr->GetOutputOnlyLinearConvergedCalls()
                << " output_only_iters="
                << igr->GetOutputOnlyLinearIterations()
                << " output_only_solve_s="
                << igr->GetOutputOnlyLinearSolveSeconds()
                << " output_only_igr_s="
                << igr->GetOutputOnlyIGRWorkSeconds()
                << " total_iters=" << total_linear_iters
                << " avg_iters=" << igr->GetAverageLinearIterations()
                << " max_iters=" << igr->GetMaxLinearIterations()
                << " avg_solve_iter_s="
                << ((total_linear_iters > 0)
                      ? (igr->GetTotalLinearSolveSeconds() /
                         static_cast<double>(total_linear_iters))
                      : 0.0)
                << " avg_krylov_operator_apply_count="
                << avg_krylov_operator_apply_count
                << " avg_krylov_preconditioner_apply_count="
                << avg_krylov_preconditioner_apply_count
                << " avg_krylov_preconditioner_operator_apply_count="
                << avg_krylov_preconditioner_operator_apply_count
                << " krylov_operator_apply_count_per_iter="
                << krylov_operator_apply_count_per_iter
                << " krylov_preconditioner_apply_count_per_iter="
                << krylov_preconditioner_apply_count_per_iter
                << " krylov_preconditioner_operator_apply_count_per_iter="
                << krylov_preconditioner_operator_apply_count_per_iter
                << " avg_assemble_s=" << avg_assemble_time
                << " avg_form_s=" << avg_form_time
                << " avg_pc_setup_s=" << avg_pc_setup_time
                << " avg_solve_s=" << avg_linear_time
                << " total_assemble_s=" << igr->GetTotalAssembleSeconds()
                << " total_form_s=" << igr->GetTotalFormSystemSeconds()
                << " total_pc_setup_s=" << igr->GetTotalPCSetupSeconds()
                << " total_cheb_diag_s="
                << igr->GetTotalChebyshevDiagonalSeconds()
                << " total_cheb_setup_s="
                << igr->GetTotalChebyshevSetupSeconds()
                << " total_cheb_eigen_reuses="
                << igr->GetTotalChebyshevEigenvalueReuses()
                << " total_cheb_smoother_reuses="
                << igr->GetTotalChebyshevSmootherReuses()
                << " total_cheb_smoother_builds="
                << igr->GetTotalChebyshevSmootherBuilds()
                << " total_krylov_operator_apply_s="
                << igr->GetTotalKrylovOperatorApplySeconds()
                << " total_krylov_preconditioner_apply_s="
                << igr->GetTotalKrylovPreconditionerApplySeconds()
                << " total_krylov_preconditioner_operator_apply_s="
                << igr->GetTotalKrylovPreconditionerOperatorApplySeconds()
                << " total_krylov_operator_apply_count="
                << igr->GetTotalKrylovOperatorApplyCount()
                << " total_krylov_preconditioner_apply_count="
                << igr->GetTotalKrylovPreconditionerApplyCount()
                << " total_krylov_preconditioner_operator_apply_count="
                << igr->GetTotalKrylovPreconditionerOperatorApplyCount()
                << " total_solve_s=" << igr->GetTotalLinearSolveSeconds()
                << " total_igr_s=" << igr->GetTotalIGRWorkSeconds()
                << " total_update_s=" << igr->GetTotalUpdateCallSeconds()
                << " solve_pct=" << igr_solve_pct
                << " total_igr_pct=" << igr_total_pct
                << " solve_pct_stage=" << igr_stage_solve_pct
                << " total_igr_pct_stage=" << igr_stage_total_pct
                << " last_solver=" << igr->GetLastLinearSolverName()
                << " last_precond=" << igr->GetLastPrecondName()
                << " last_cheb_diag_s="
                << igr->GetLastChebyshevDiagonalSeconds()
                << " last_cheb_setup_s="
                << igr->GetLastChebyshevSetupSeconds()
                << " last_cheb_eigen_reused="
                << (igr->GetLastChebyshevEigenvalueReused() ? 1 : 0)
                << " last_cheb_smoother_reused="
                << (igr->GetLastChebyshevSmootherReused() ? 1 : 0)
                << " last_cheb_eigen_estimate="
                << igr->GetLastChebyshevEigenvalueEstimate()
                << " last_krylov_operator_apply_s="
                << igr->GetLastKrylovOperatorApplySeconds()
                << " last_krylov_preconditioner_apply_s="
                << igr->GetLastKrylovPreconditionerApplySeconds()
                << " last_krylov_preconditioner_operator_apply_s="
                << igr->GetLastKrylovPreconditionerOperatorApplySeconds()
                << " last_krylov_operator_apply_count="
                << igr->GetLastKrylovOperatorApplyCount()
                << " last_krylov_preconditioner_apply_count="
                << igr->GetLastKrylovPreconditionerApplyCount()
                << " last_krylov_preconditioner_operator_apply_count="
                << igr->GetLastKrylovPreconditionerOperatorApplyCount()
                << " direct_identity_system="
                << (igr->GetLastDirectIdentitySystem() ? 1 : 0)
                << " direct_identity_reason="
                << igr->GetLastDirectIdentityReason()
                << " direct_identity_uses="
                << igr->GetDirectIdentitySystemUses()
                << " exact_face_requested="
                << (igr->GetLastExactFaceRequested() ? 1 : 0)
                << " exact_face_effective="
                << (igr->GetLastExactFaceEffective() ? 1 : 0)
                << " exact_face_reason=" << igr->GetLastExactFaceReason()
                << " exact_face_local_nf="
                << igr->GetLastExactFaceLocalFaceCount()
                << " exact_face_shared_nf="
                << igr->GetLastExactFaceSharedFaceCount()
                << " exact_face_operator_uses="
                << igr->GetExactFaceOperatorUseCount()
                << " exact_face_last_mult_calls="
                << igr->GetLastExactFaceOperatorMultCalls() << "\n";
    }

    if (run_cfg.opts.save_state_files)
    {
      SaveFinalEulerState(ctx,
                          state_bundle.sol,
                          spec.glvis.precision,
                          "euler-mesh-final",
                          "euler",
                          state_snapshot_dir);
    }

    WriteRunMetadataSidecar(case_cfg,
                            run_cfg,
                            captured_summary_valid ? &captured_summary
                                                   : nullptr,
                            igr.get(),
                            vtk_c_alpha,
                            argc,
                            argv,
                            /*success=*/true,
                            output_dataset_name,
                            restart_info ? &*restart_info : nullptr);

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

} // namespace dg_sim

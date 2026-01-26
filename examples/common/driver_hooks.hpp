#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mfem.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "physics/igr/igr_flux.hpp"
#include "physics/igr/igr_coupling.hpp"

#include "driver.hpp"
#include "boundary_helpers.hpp"
#include "time_loop.hpp"
#include "vis_helpers.hpp"

namespace dg_igr_examples
{

/// @brief Bundles mesh, DG spaces, and basic dimension metadata for examples.
struct ExampleContext
{
  const RunConfig& run_cfg;
  std::unique_ptr<mfem::ParMesh> pmesh;
  int dim = 0;
  int num_equations = 0;
  DGSpaces spaces;

  ExampleContext(const RunConfig& cfg,
                 std::unique_ptr<mfem::ParMesh> pmesh_in,
                 const std::string& sigma_space,
                 bool need_sigma,
                 bool need_h1)
    : run_cfg(cfg),
      pmesh(std::move(pmesh_in)),
      dim(pmesh->Dimension()),
      num_equations(dim + 2),
      spaces(*pmesh, cfg.order, dim, num_equations, sigma_space, need_sigma, need_h1)
  {}

  mfem::ParMesh& Mesh() { return *pmesh; }
  const mfem::ParMesh& Mesh() const { return *pmesh; }
};

/// @brief Build mesh + DG spaces with standard serial/parallel refinement.
inline ExampleContext BuildExampleContext(const RunConfig& run_cfg,
                                          const std::string& sigma_space,
                                          bool need_sigma,
                                          bool need_h1)
{
  mfem::Mesh mesh = LoadSerialMesh(run_cfg.mesh_path, run_cfg.opts.ser_ref_levels);
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mesh.Clear();
  RefineParMesh(*pmesh, run_cfg.opts.par_ref_levels);

  return ExampleContext(run_cfg, std::move(pmesh), sigma_space, need_sigma, need_h1);
}

/// @brief Common state/pressure bundle for Euler examples.
struct EulerStateBundle
{
  mfem::ParGridFunction sol;
  EulerStateViews state;
  EulerPressureView pressure_view;
  EulerFieldList field_list;

  EulerStateBundle(ExampleContext& ctx, double gamma)
    : sol(&ctx.spaces.vfes),
      state(ctx.spaces.fes, ctx.spaces.dfes, sol, ctx.dim),
      pressure_view(ctx.spaces.fes, sol, ctx.dim, gamma)
  {
    field_list.state = &state;
    field_list.pressure = &pressure_view;
  }

  void AttachIGR(dg_igr::IGRSolver& igr) { field_list.igr = &igr; }

  void ProjectIC(mfem::VectorCoefficient& ic)
  {
    sol.ProjectCoefficient(ic);
    pressure_view.Update();
  }
};

/// @brief Common inflow options for examples that expose inlet velocities.
struct InflowOptions
{
  double vx = 3.0;
  double vy = 0.0;
};

inline void AddInflowOptions(mfem::OptionsParser& args,
                             InflowOptions& inflow_opts)
{
  args.AddOption(&inflow_opts.vx, "-vx", "--velocity-x",
                 "Free-stream / inflow velocity x-component.");
  args.AddOption(&inflow_opts.vy, "-vy", "--velocity-y",
                 "Free-stream / inflow velocity y-component.");
}

/// @brief Save the initial mesh + conservative state in GLVis-compatible files.
inline void SaveInitialEulerState(ExampleContext& ctx,
                                  mfem::ParGridFunction& sol,
                                  int precision,
                                  const std::string& mesh_prefix = "euler-mesh",
                                  const std::string& field_prefix = "euler")
{
  std::ostringstream mesh_name;
  mesh_name << mesh_prefix << "." << std::setfill('0') << std::setw(6)
            << mfem::Mpi::WorldRank();
  std::ofstream mesh_ofs(mesh_name.str().c_str());
  mesh_ofs.precision(precision);
  mesh_ofs << ctx.Mesh();

  for (int k = 0; k < ctx.num_equations; k++)
  {
    mfem::ParGridFunction uk(&ctx.spaces.fes,
                             sol.GetData() + k * ctx.spaces.fes.GetNDofs());
    std::ostringstream sol_name;
    sol_name << field_prefix << "-" << k << "-init." << std::setfill('0')
             << std::setw(6) << mfem::Mpi::WorldRank();
    std::ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(precision);
    sol_ofs << uk;
  }
}

/// @brief Save the final mesh + conservative state in GLVis-compatible files.
inline void SaveFinalEulerState(ExampleContext& ctx,
                                mfem::ParGridFunction& sol,
                                int precision,
                                const std::string& mesh_prefix = "euler-mesh-final",
                                const std::string& field_prefix = "euler")
{
  std::ostringstream mesh_name;
  mesh_name << mesh_prefix << "." << std::setfill('0') << std::setw(6)
            << mfem::Mpi::WorldRank();
  std::ofstream mesh_ofs(mesh_name.str().c_str());
  mesh_ofs.precision(precision);
  mesh_ofs << ctx.Mesh();

  for (int k = 0; k < ctx.num_equations; k++)
  {
    mfem::ParGridFunction uk(&ctx.spaces.fes,
                             sol.GetData() + k * ctx.spaces.fes.GetNDofs());
    std::ostringstream sol_name;
    sol_name << field_prefix << "-" << k << "-final." << std::setfill('0')
             << std::setw(6) << mfem::Mpi::WorldRank();
    std::ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(precision);
    sol_ofs << uk;
  }
}

/// @brief Default VTK output directory for a case (prefix_path/case_name).
inline std::string DefaultCaseVTKOutputDir(const CaseConfig& case_cfg)
{
  return (std::filesystem::path(case_cfg.vtk_config.prefix_path)
          / case_cfg.case_name).string();
}

/// @brief Build a consistent dataset name for ParaView output.
inline std::string MakeCaseVTKDatasetName(const CaseConfig& case_cfg,
                                          const RunConfig& run_cfg,
                                          double c_alpha_value =
                                             std::numeric_limits<double>::quiet_NaN())
{
  const int total_ref = run_cfg.opts.ser_ref_levels + run_cfg.opts.par_ref_levels;
  const double c_alpha = std::isnan(c_alpha_value)
                             ? run_cfg.igr_opts.c_alpha_user
                             : c_alpha_value;
  std::ostringstream name;
  name << case_cfg.case_name
       << "_p_" << run_cfg.order
       << "_ref_" << total_ref
       << "_is_" << run_cfg.igr_opts.space
       << "_ig_" << run_cfg.igr_opts.grad
       << "_ca_" << c_alpha;
  return name.str();
}

/// @brief Apply case defaults to the VTK output config.
inline mfem::io::VTKOutputConfig MakeCaseVTKConfig(const CaseConfig& case_cfg)
{
  mfem::io::VTKOutputConfig cfg = case_cfg.vtk_config;
  cfg.prefix_path = DefaultCaseVTKOutputDir(case_cfg);
  return cfg;
}

/// @brief Flux bundle for Euler + IGR wiring with a chosen numerical flux.
template <typename NumericalFlux>
struct EulerIGRFluxBundle
{
  mfem::EulerFlux base_flux;
  mfem::FluxWithIGR igr_flux;
  NumericalFlux numerical_flux;

  EulerIGRFluxBundle(int dim, double gamma, dg_igr::IGRSolver& igr)
    : base_flux(dim, gamma),
      igr_flux(base_flux),
      numerical_flux(igr_flux)
  {
    igr_flux.SetSigmaField(&igr.GetSigma());
  }
};

/// @brief Convenience wrapper for adding Euler boundary integrators.
template <typename NumericalFlux>
class EulerBoundaryWiring
{
public:
  EulerBoundaryWiring(mfem::DGHyperbolicConservationLaws& op,
                      const NumericalFlux& numerical_flux,
                      int int_order_offset,
                      int dim)
    : op_(op),
      numerical_flux_(numerical_flux),
      int_order_offset_(int_order_offset),
      dim_(dim)
  {}

  void AddSlipWall(const BoundaryMarkerMap& markers,
                   BoundaryRole role = BoundaryRole::Slip) const
  {
    dg_igr_examples::AddSlipWall(op_, numerical_flux_, int_order_offset_,
                                 dim_, markers.At(role));
  }

  void AddSupersonicOutlet(const BoundaryMarkerMap& markers,
                           BoundaryRole role = BoundaryRole::Outlet) const
  {
    dg_igr_examples::AddSupersonicOutlet(op_, numerical_flux_, int_order_offset_,
                                         markers.At(role));
  }

  void AddSupersonicInlet(const BoundaryMarkerMap& markers,
                          const mfem::Vector& inlet_state,
                          BoundaryRole role = BoundaryRole::Inlet) const
  {
    dg_igr_examples::AddSupersonicInlet(op_, numerical_flux_, inlet_state,
                                        int_order_offset_, markers.At(role));
  }

  template <typename StateBuilder>
  void AddSupersonicInletFromBuilder(const BoundaryMarkerMap& markers,
                                     StateBuilder&& build_inlet_state,
                                     BoundaryRole role = BoundaryRole::Inlet) const
  {
    dg_igr_examples::AddSupersonicInletFromBuilder(
        op_, numerical_flux_, std::forward<StateBuilder>(build_inlet_state),
        int_order_offset_, markers.At(role));
  }

private:
  mfem::DGHyperbolicConservationLaws& op_;
  const NumericalFlux& numerical_flux_;
  int int_order_offset_;
  int dim_;
};

/// @brief Euler+IGR physics bundle with BC wiring applied.
template <typename NumericalFlux>
struct EulerIGRPhysics
{
  EulerIGRFluxBundle<NumericalFlux> flux;
  mfem::DGHyperbolicConservationLaws op;

  template <typename BcSetup>
  EulerIGRPhysics(ExampleContext& ctx,
                  double gamma,
                  dg_igr::IGRSolver& igr,
                  const BoundaryMarkerMap& markers,
                  BcSetup&& bc_setup)
    : flux(ctx.dim, gamma, igr),
      op(MakeDGOperator(ctx, flux.numerical_flux))
  {
    EulerBoundaryWiring<NumericalFlux> bc(
        op, flux.numerical_flux, ctx.run_cfg.int_order_offset, ctx.dim);
    bc_setup(bc, markers);
  }
};

/// @brief Build Euler+IGR physics with per-case BC wiring.
template <typename NumericalFlux, typename BcSetup>
inline EulerIGRPhysics<NumericalFlux> MakeEulerIGRPhysics(
    ExampleContext& ctx,
    double gamma,
    dg_igr::IGRSolver& igr,
    const BoundaryMarkerMap& markers,
    BcSetup&& bc_setup)
{
  return EulerIGRPhysics<NumericalFlux>(
      ctx, gamma, igr, markers, std::forward<BcSetup>(bc_setup));
}

/// @brief Build a DG operator with a numerical flux and shared settings.
template <typename NumericalFlux>
inline mfem::DGHyperbolicConservationLaws MakeDGOperator(
    ExampleContext& ctx,
    const NumericalFlux& numerical_flux)
{
  return mfem::DGHyperbolicConservationLaws(
      ctx.spaces.vfes,
      std::unique_ptr<mfem::HyperbolicFormIntegrator>(
          new mfem::HyperbolicFormIntegrator(numerical_flux,
                                             ctx.run_cfg.int_order_offset)),
      ctx.run_cfg.opts.preassemble_weak_div);
}

/// @brief Optional IO wiring for the shared time loop.
struct TimeLoopIO
{
  EulerVisWindows* glvis_windows = nullptr;
  bool* visualization = nullptr;
  mfem::ParaViewDataCollection* paraview = nullptr;
};

inline std::string FormatFixed(double value, int precision = 6)
{
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os << std::setprecision(precision) << value;
  return os.str();
}

inline std::string FormatSci(double value, int precision = 3)
{
  std::ostringstream os;
  os.setf(std::ios::scientific);
  os << std::setprecision(precision) << value;
  return os.str();
}

inline void PrintVTKSaveLog(int cycle, int step, double t,
                            bool is_final, bool is_initial)
{
  std::cout << "[vtk] cycle " << std::setw(6) << cycle
            << " | step " << std::setw(6) << step
            << " | t " << std::setw(12) << FormatFixed(t);
  if (is_initial)
  {
    std::cout << " | initial";
  }
  if (is_final)
  {
    std::cout << " | final";
  }
  std::cout << std::endl;
}

inline void PrintBox(const std::vector<std::string>& lines)
{
  if (lines.empty())
  {
    return;
  }
  size_t width = 0;
  for (const auto& line : lines)
  {
    width = std::max(width, line.size());
  }
  const std::string border = "+" + std::string(width + 2, '-') + "+";
  std::cout << border << "\n";
  for (const auto& line : lines)
  {
    std::cout << "| " << line
              << std::string(width - line.size(), ' ')
              << " |\n";
  }
  std::cout << border << std::endl;
}

inline std::pair<double, double> ComputeGlobalMinMax(const mfem::ParGridFunction& gf,
                                                     MPI_Comm comm)
{
  const int n = gf.Size();
  const double* data = gf.GetData();
  double local_min = std::numeric_limits<double>::infinity();
  double local_max = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i)
  {
    local_min = std::min(local_min, data[i]);
    local_max = std::max(local_max, data[i]);
  }
  double global_min = local_min;
  double global_max = local_max;
  MPI_Allreduce(MPI_IN_PLACE, &global_min, 1,
                mfem::MPITypeMap<double>::mpi_type, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &global_max, 1,
                mfem::MPITypeMap<double>::mpi_type, MPI_MAX, comm);
  return {global_min, global_max};
}

inline void PrintRunPerformanceSummary(const ExampleContext& ctx,
                                       const TimeLoopResult& result,
                                       double elapsed_seconds,
                                       int num_ranks)
{
  const double local_dofs =
      static_cast<double>(ctx.spaces.vfes.TrueVSize());
  const double global_dofs =
      static_cast<double>(ctx.spaces.vfes.GlobalTrueVSize());

  if (result.steps <= 0)
  {
    PrintBox({
        "Summary",
        "steps: 0",
        "elapsed: " + FormatFixed(elapsed_seconds) + " s",
        "dofs (local/global): " +
            std::to_string(static_cast<long long>(local_dofs)) + " / " +
            std::to_string(static_cast<long long>(global_dofs)),
        "ranks: " + std::to_string(num_ranks)
    });
    return;
  }

  const double steps = static_cast<double>(result.steps);
  const double avg_step = elapsed_seconds / steps;
  const double avg_step_per_dof_local =
      (local_dofs > 0.0) ? (avg_step / local_dofs) : 0.0;
  const double avg_step_per_dof_rank =
      (global_dofs > 0.0 && num_ranks > 0)
          ? (avg_step * static_cast<double>(num_ranks) / global_dofs)
          : 0.0;

  PrintBox({
      "Summary",
      "steps: " + std::to_string(result.steps) +
          " | elapsed: " + FormatFixed(elapsed_seconds) + " s",
      "avg/step: " + FormatSci(avg_step) + " s",
      "avg/step/dof (local): " + FormatSci(avg_step_per_dof_local) + " s",
      "avg/step/dof/rank: " + FormatSci(avg_step_per_dof_rank) + " s",
      "dofs (local/global): " +
          std::to_string(static_cast<long long>(local_dofs)) + " / " +
          std::to_string(static_cast<long long>(global_dofs)),
      "ranks: " + std::to_string(num_ranks)
  });
}

inline void PrintStepLog(int step,
                         double t,
                         double dt,
                         double avg_step,
                         double avg_step_per_dof_local,
                         double avg_step_per_dof_rank)
{
  std::cout << "[alive] step " << std::setw(6) << step
            << " | t " << std::setw(12) << FormatFixed(t)
            << " | dt " << std::setw(12) << FormatSci(dt)
            << " | avg/step " << std::setw(12) << FormatSci(avg_step) << " s"
            << " | avg/step/dof(local) " << std::setw(12)
            << FormatSci(avg_step_per_dof_local) << " s"
            << " | avg/step/dof/rank " << std::setw(12)
            << FormatSci(avg_step_per_dof_rank) << " s"
            << std::endl;
}

/// @brief Build standard visualization/save hooks for Euler+IGR examples.
inline TimeLoopHooks MakeEulerTimeLoopHooks(const ExampleContext& ctx,
                                            EulerStateBundle& bundle,
                                            int num_procs,
                                            int my_rank,
                                            TimeLoopIO io)
{
  const auto start_time = std::chrono::steady_clock::now();
  const double local_dofs =
      static_cast<double>(ctx.spaces.vfes.TrueVSize());
  const double global_dofs =
      static_cast<double>(ctx.spaces.vfes.GlobalTrueVSize());
  TimeLoopHooks hooks;
  hooks.on_visualize = [&](int step, mfem::real_t t, mfem::real_t dt_real, bool)
  {
    if (mfem::Mpi::Root())
    {
      const auto now = std::chrono::steady_clock::now();
      const double elapsed =
          std::chrono::duration<double>(now - start_time).count();
      const double avg_step = (step > 0) ? (elapsed / step) : 0.0;
      const double avg_step_per_dof_local =
          (step > 0 && local_dofs > 0.0) ? (avg_step / local_dofs) : 0.0;
      const double avg_step_per_dof_rank =
          (step > 0 && global_dofs > 0.0 && num_procs > 0)
              ? (avg_step * static_cast<double>(num_procs) / global_dofs)
              : 0.0;
      PrintStepLog(step, t, dt_real, avg_step,
                   avg_step_per_dof_local, avg_step_per_dof_rank);
    }
    if (io.visualization && *io.visualization && io.glvis_windows)
    {
      UpdateEulerGLVis(*io.glvis_windows, bundle.field_list, bundle.sol,
                       num_procs, my_rank, ctx.Mesh(), t,
                       /*warm_start=*/true);
    }
  };

  hooks.on_save = [&](int step, mfem::real_t t, int save_idx, bool is_final)
  {
    if (mfem::Mpi::Root())
    {
      PrintVTKSaveLog(save_idx, step, t, is_final, /*is_initial=*/false);
    }
    UpdateEulerOutputFields(bundle.field_list, bundle.sol, /*warm_start=*/true);
    if (io.paraview)
    {
      io.paraview->SetCycle(save_idx);
      io.paraview->SetTime(t);
      io.paraview->Save();
    }
  };

  hooks.on_analysis = [&](int step, mfem::real_t t, mfem::real_t dt_real, bool done)
  {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(now - start_time).count();
    const double avg_step = (step > 0) ? (elapsed / step) : 0.0;
    const double avg_step_per_dof_local =
        (step > 0 && local_dofs > 0.0) ? (avg_step / local_dofs) : 0.0;
    const double avg_step_per_dof_rank =
        (step > 0 && global_dofs > 0.0 && num_procs > 0)
            ? (avg_step * static_cast<double>(num_procs) / global_dofs)
            : 0.0;

    std::pair<double, double> rho_mm;
    std::pair<double, double> p_mm;
    bool have_rho = false;
    bool have_p = false;
    if (bundle.field_list.state)
    {
      rho_mm = ComputeGlobalMinMax(
          bundle.field_list.state->density, ctx.Mesh().GetComm());
      have_rho = true;
    }
    if (bundle.field_list.pressure)
    {
      bundle.field_list.pressure->Update();
      p_mm = ComputeGlobalMinMax(
          bundle.field_list.pressure->pressure, ctx.Mesh().GetComm());
      have_p = true;
    }

    if (!mfem::Mpi::Root())
    {
      return;
    }

    std::vector<std::string> lines;
    lines.push_back("Analysis");
    lines.push_back("step: " + std::to_string(step) +
                    " | t: " + FormatFixed(t) +
                    " | dt: " + FormatSci(dt_real) +
                    (done ? " | final" : ""));
    lines.push_back("elapsed: " + FormatFixed(elapsed) + " s");
    lines.push_back("avg/step: " + FormatSci(avg_step) + " s");
    lines.push_back("avg/step/dof (local): " +
                    FormatSci(avg_step_per_dof_local) + " s");
    lines.push_back("avg/step/dof/rank: " +
                    FormatSci(avg_step_per_dof_rank) + " s");
    lines.push_back("dofs (local/global): " +
                    std::to_string(static_cast<long long>(local_dofs)) + " / " +
                    std::to_string(static_cast<long long>(global_dofs)) +
                    " | ranks: " + std::to_string(num_procs));

    if (have_rho)
    {
      lines.push_back("rho min/max: " + FormatSci(rho_mm.first) + " / " +
                      FormatSci(rho_mm.second));
    }
    if (have_p)
    {
      lines.push_back("p   min/max: " + FormatSci(p_mm.first) + " / " +
                      FormatSci(p_mm.second));
    }

    PrintBox(lines);
  };

  return hooks;
}

/// @brief Run Euler+IGR with standard visualization + VTK setup in one call.
template <typename ExtraOnOpen>
inline bool RunEulerIGRCaseWithIO(ExampleContext& ctx,
                                  const CaseConfig& case_cfg,
                                  const RunConfig& run_cfg,
                                  EulerStateBundle& bundle,
                                  dg_igr::IGRSolver& igr,
                                  mfem::RK3SSPWithLimiter& rk3,
                                  mfem::DGHyperbolicConservationLaws& op,
                                  bool& visualization,
                                  int precision,
                                  ExtraOnOpen&& extra_on_open,
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
      extra_on_open();
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
    if (mfem::Mpi::Root())
    {
      PrintVTKSaveLog(0, 0, 0.0, /*is_final=*/false, /*is_initial=*/true);
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

  mfem::EulerWithIGR wrapped(op, igr, bundle.sol);
  rk3.Init(wrapped);
  TimeLoopConfig loop_cfg = MakeTimeLoopConfig(run_cfg);
  if (run_cfg.opts.vtk_save)
  {
    loop_cfg.save_index_offset = 1;
  }
  const int num_procs = mfem::Mpi::WorldSize();
  const int my_rank = mfem::Mpi::WorldRank();
  const TimeLoopHooks hooks =
      MakeEulerTimeLoopHooks(ctx, bundle, num_procs, my_rank, io);
  const TimeLoopResult result =
      RunTimeLoop(wrapped, ctx.Mesh(), bundle.sol, rk3, loop_cfg, hooks);
  if (result.non_finite_count > 0)
  {
    return false;
  }

  mfem::tic_toc.Stop();
  if (mfem::Mpi::Root())
  {
    const double elapsed = mfem::tic_toc.RealTime();
    std::cout << " done, " << elapsed << "s." << std::endl;
    PrintRunPerformanceSummary(ctx, result, elapsed, num_procs);
  }
  return true;
}

inline bool RunEulerIGRCaseWithIO(ExampleContext& ctx,
                                  const CaseConfig& case_cfg,
                                  const RunConfig& run_cfg,
                                  EulerStateBundle& bundle,
                                  dg_igr::IGRSolver& igr,
                                  mfem::RK3SSPWithLimiter& rk3,
                                  mfem::DGHyperbolicConservationLaws& op,
                                  bool& visualization,
                                  int precision,
                                  double c_alpha_value =
                                     std::numeric_limits<double>::quiet_NaN(),
                                  bool print_dataset_name = false)
{
  return RunEulerIGRCaseWithIO(ctx, case_cfg, run_cfg, bundle, igr, rk3, op,
                               visualization, precision, []() {},
                               c_alpha_value, print_dataset_name);
}

/// @brief Run Euler (no IGR) with standard visualization + VTK setup in one call.
template <typename ExtraOnOpen>
inline bool RunEulerCaseWithIO(ExampleContext& ctx,
                               const CaseConfig& case_cfg,
                               const RunConfig& run_cfg,
                               EulerStateBundle& bundle,
                               mfem::RK3SSPWithLimiter& rk3,
                               mfem::DGHyperbolicConservationLaws& op,
                               bool& visualization,
                               int precision,
                               ExtraOnOpen&& extra_on_open,
                               bool print_dataset_name = false)
{
  UpdateEulerOutputFields(bundle.field_list, bundle.sol, /*warm_start=*/false);

  EulerVisWindows glvis_windows;
  if (visualization)
  {
    const bool ok = OpenEulerGLVisWindows(glvis_windows, bundle.field_list,
                                          ctx.Mesh(),
                                          mfem::Mpi::WorldSize(),
                                          mfem::Mpi::WorldRank(),
                                          precision);
    if (ok)
    {
      extra_on_open();
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
      MakeCaseVTKDatasetName(case_cfg, run_cfg);
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
    if (mfem::Mpi::Root())
    {
      PrintVTKSaveLog(0, 0, 0.0, /*is_final=*/false, /*is_initial=*/true);
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
      MakeEulerTimeLoopHooks(ctx, bundle, num_procs, my_rank, io);
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
    PrintRunPerformanceSummary(ctx, result, elapsed, num_procs);
  }
  return true;
}

inline bool RunEulerCaseWithIO(ExampleContext& ctx,
                               const CaseConfig& case_cfg,
                               const RunConfig& run_cfg,
                               EulerStateBundle& bundle,
                               mfem::RK3SSPWithLimiter& rk3,
                               mfem::DGHyperbolicConservationLaws& op,
                               bool& visualization,
                               int precision,
                               bool print_dataset_name = false)
{
  return RunEulerCaseWithIO(ctx, case_cfg, run_cfg, bundle, rk3, op,
                            visualization, precision, []() {},
                            print_dataset_name);
}

} // namespace dg_igr_examples

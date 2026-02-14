#pragma once

/**
 * @file sim/time/time_defaults.hpp
 * @brief Default logging/analysis hooks for the shared time loop.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mfem.hpp"

#include "sim/context/example_context.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/io/vis_helpers.hpp"
#include "sim/time/time_callbacks.hpp"
#include "sim/time/time_loop.hpp"

namespace dg_sim
{

/// @brief Optional IO wiring for the shared time loop.
struct TimeLoopIO
{
  EulerVisWindows* glvis_windows = nullptr;
  bool* glvis_enabled = nullptr;
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

inline void
PrintVTKSaveLog(int cycle, int step, double t, bool is_final, bool is_initial)
{
  std::cout << "[vtk] cycle " << std::setw(6) << cycle << " | step "
            << std::setw(6) << step << " | t " << std::setw(12)
            << FormatFixed(t);
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
  auto repeat = [](const std::string& token, size_t count)
  {
    std::string out;
    out.reserve(token.size() * count);
    for (size_t i = 0; i < count; ++i)
    {
      out += token;
    }
    return out;
  };
  const std::string h = repeat("─", width + 2);
  const std::string top = "┌" + h + "┐";
  const std::string mid = "├" + h + "┤";
  const std::string bot = "└" + h + "┘";
  std::cout << top << "\n";
  for (size_t i = 0; i < lines.size(); ++i)
  {
    const auto& line = lines[i];
    std::cout << "│ " << line << std::string(width - line.size(), ' ')
              << " │\n";
    if (i == 0 && lines.size() > 1)
    {
      std::cout << mid << "\n";
    }
  }
  std::cout << bot << std::endl;
}

inline std::pair<double, double>
ComputeGlobalMinMax(const mfem::ParGridFunction& gf, MPI_Comm comm)
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
  MPI_Allreduce(MPI_IN_PLACE,
                &global_min,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MIN,
                comm);
  MPI_Allreduce(MPI_IN_PLACE,
                &global_max,
                1,
                mfem::MPITypeMap<double>::mpi_type,
                MPI_MAX,
                comm);
  return {global_min, global_max};
}

inline RunSummary BuildRunSummary(const ExampleContext& ctx,
                                  const TimeLoopResult& result,
                                  double elapsed_seconds,
                                  int num_ranks,
                                  const dg_igr::IGRSolver* igr = nullptr)
{
  RunSummary summary;
  summary.steps = result.steps;
  summary.elapsed = elapsed_seconds;
  summary.local_dofs = static_cast<int>(ctx.spaces.vfes.TrueVSize());
  summary.global_dofs =
    static_cast<long long>(ctx.spaces.vfes.GlobalTrueVSize());
  summary.ranks = num_ranks;
  if (igr)
  {
    summary.has_igr = true;
    summary.igr_linear_calls = igr->GetLinearSolveCalls();
    summary.igr_linear_converged_calls = igr->GetTotalLinearConvergedCalls();
    summary.igr_linear_avg_iters = igr->GetAverageLinearIterations();
    summary.igr_linear_solve_seconds = igr->GetTotalLinearSolveSeconds();
    summary.igr_linear_solve_fraction =
      (elapsed_seconds > 0.0)
        ? (summary.igr_linear_solve_seconds / elapsed_seconds)
        : 0.0;
  }

  if (result.steps <= 0 || summary.local_dofs <= 0 || summary.global_dofs <= 0)
  {
    return summary;
  }

  const double steps = static_cast<double>(result.steps);
  summary.avg_step = elapsed_seconds / steps;
  summary.avg_step_per_dof_local =
    summary.avg_step / static_cast<double>(summary.local_dofs);
  summary.avg_step_per_dof_rank =
    (num_ranks > 0) ? (summary.avg_step * static_cast<double>(num_ranks) /
                       static_cast<double>(summary.global_dofs))
                    : 0.0;
  return summary;
}

inline void FillIGRTimeInfo(TimeLoopInfo& info,
                            double elapsed,
                            const dg_igr::IGRSolver* igr_solver)
{
  if (!igr_solver)
  {
    return;
  }
  info.has_igr = true;
  info.igr_linear_calls = igr_solver->GetLinearSolveCalls();
  info.igr_linear_converged_calls = igr_solver->GetTotalLinearConvergedCalls();
  info.igr_linear_avg_iters = igr_solver->GetAverageLinearIterations();
  info.igr_linear_solve_seconds = igr_solver->GetTotalLinearSolveSeconds();
  info.igr_linear_solve_fraction =
    (elapsed > 0.0) ? (info.igr_linear_solve_seconds / elapsed) : 0.0;
}

inline void PrintSummaryBox(const RunSummary& summary)
{
  const auto igr_line = [&summary]() -> std::string
  {
    if (!summary.has_igr)
    {
      return "IGR linear solve: disabled";
    }
    std::ostringstream os;
    os << "IGR linear solve: " << FormatFixed(summary.igr_linear_solve_seconds)
       << " s (" << FormatFixed(100.0 * summary.igr_linear_solve_fraction, 2)
       << "%) | calls: " << summary.igr_linear_calls
       << " | conv: " << summary.igr_linear_converged_calls
       << " | avg iters: " << FormatFixed(summary.igr_linear_avg_iters, 2);
    return os.str();
  };

  if (summary.steps <= 0)
  {
    PrintBox({"Summary",
              "steps: 0",
              "elapsed: " + FormatFixed(summary.elapsed) + " s",
              igr_line(),
              "dofs (local/global): " +
                std::to_string(static_cast<long long>(summary.local_dofs)) +
                " / " +
                std::to_string(static_cast<long long>(summary.global_dofs)),
              "ranks: " + std::to_string(summary.ranks)});
    return;
  }

  PrintBox(
    {"Summary",
     "steps: " + std::to_string(summary.steps) +
       " | elapsed: " + FormatFixed(summary.elapsed) + " s",
     igr_line(),
     "avg/step: " + FormatSci(summary.avg_step) + " s",
     "avg/step/dof (local): " + FormatSci(summary.avg_step_per_dof_local) +
       " s",
     "avg/step/dof/rank: " + FormatSci(summary.avg_step_per_dof_rank) + " s",
     "dofs (local/global): " +
       std::to_string(static_cast<long long>(summary.local_dofs)) + " / " +
       std::to_string(static_cast<long long>(summary.global_dofs)),
     "ranks: " + std::to_string(summary.ranks)});
}

inline void PrintStepLog(int step,
                         double t,
                         double dt,
                         double avg_step,
                         double avg_step_per_dof_local,
                         double avg_step_per_dof_rank)
{
  std::cout << "[alive] step " << std::setw(6) << step << " | t "
            << std::setw(12) << FormatFixed(t) << " | dt " << std::setw(12)
            << FormatSci(dt) << " | avg/step " << std::setw(12)
            << FormatSci(avg_step) << " s" << " | avg/step/dof(local) "
            << std::setw(12) << FormatSci(avg_step_per_dof_local) << " s"
            << " | avg/step/dof/rank " << std::setw(12)
            << FormatSci(avg_step_per_dof_rank) << " s" << std::endl;
}

/// @brief Build default callbacks (alive/analysis/vtk/summary) from config.
inline TimeLoopCallbacks
MakeDefaultTimeLoopCallbacks(const DefaultCallbackConfig& cfg)
{
  TimeLoopCallbacks callbacks;
  callbacks.wants_minmax = cfg.enable_analysis && cfg.compute_minmax;

  if (cfg.enable_alive)
  {
    callbacks.on_alive = [cfg](const TimeLoopInfo& info)
    {
      if (!mfem::Mpi::Root())
      {
        return;
      }
      PrintStepLog(info.step,
                   info.t,
                   info.dt,
                   info.avg_step,
                   info.avg_step_per_dof_local,
                   info.avg_step_per_dof_rank);
    };
  }

  if (cfg.enable_vtk)
  {
    callbacks.on_vtk_save = [cfg](const VTKSaveInfo& info)
    {
      if (!mfem::Mpi::Root())
      {
        return;
      }
      PrintVTKSaveLog(info.cycle,
                      info.step,
                      info.t,
                      info.is_final,
                      info.is_initial);
    };
  }

  if (cfg.enable_analysis)
  {
    callbacks.on_analysis = [cfg](const AnalysisInfo& info)
    {
      if (!mfem::Mpi::Root())
      {
        return;
      }

      std::vector<std::string> lines;
      lines.push_back("Analysis");
      lines.push_back("step: " + std::to_string(info.base.step) +
                      " | t: " + FormatFixed(info.base.t) +
                      " | dt: " + FormatSci(info.base.dt) +
                      (info.base.is_final ? " | final" : ""));
      lines.push_back("elapsed: " + FormatFixed(info.base.elapsed) + " s");
      lines.push_back("avg/step: " + FormatSci(info.base.avg_step) + " s");
      lines.push_back("avg/step/dof (local): " +
                      FormatSci(info.base.avg_step_per_dof_local) + " s");
      lines.push_back("avg/step/dof/rank: " +
                      FormatSci(info.base.avg_step_per_dof_rank) + " s");
      if (info.base.has_igr)
      {
        lines.push_back(
          "IGR linear solve: " +
          FormatFixed(info.base.igr_linear_solve_seconds) + " s (" +
          FormatFixed(100.0 * info.base.igr_linear_solve_fraction, 2) +
          "%) | calls: " + std::to_string(info.base.igr_linear_calls) +
          " | conv: " + std::to_string(info.base.igr_linear_converged_calls) +
          " | avg iters: " + FormatFixed(info.base.igr_linear_avg_iters, 2));
      }
      lines.push_back(
        "dofs (local/global): " +
        std::to_string(static_cast<long long>(info.base.local_dofs)) + " / " +
        std::to_string(static_cast<long long>(info.base.global_dofs)) +
        " | ranks: " + std::to_string(info.base.ranks));

      if (info.has_rho)
      {
        lines.push_back("rho min/max: " + FormatSci(info.rho_min) + " / " +
                        FormatSci(info.rho_max));
      }
      if (info.has_p)
      {
        lines.push_back("p   min/max: " + FormatSci(info.p_min) + " / " +
                        FormatSci(info.p_max));
      }

      PrintBox(lines);
    };
  }

  if (cfg.enable_summary)
  {
    callbacks.on_summary = [cfg](const RunSummary& summary)
    {
      if (!mfem::Mpi::Root())
      {
        return;
      }
      PrintSummaryBox(summary);
    };
  }

  return callbacks;
}

/// @brief Build standard visualization/save hooks for Euler examples.
inline TimeLoopHooks MakeEulerTimeLoopHooks(const ExampleContext& ctx,
                                            EulerStateBundle& bundle,
                                            int num_procs,
                                            int my_rank,
                                            TimeLoopIO io,
                                            const TimeLoopCallbacks& callbacks)
{
  const auto start_time = std::chrono::steady_clock::now();
  const double local_dofs = static_cast<double>(ctx.spaces.vfes.TrueVSize());
  const double global_dofs =
    static_cast<double>(ctx.spaces.vfes.GlobalTrueVSize());
  const int num_procs_value = num_procs;
  const int my_rank_value = my_rank;
  EulerVisWindows* glvis_windows = io.glvis_windows;
  bool* glvis_enabled = io.glvis_enabled;
  mfem::ParaViewDataCollection* paraview = io.paraview;
  dg_igr::IGRSolver* igr_solver = bundle.field_list.igr;
  const ExampleContext* ctx_ptr = &ctx;
  EulerStateBundle* bundle_ptr = &bundle;
  TimeLoopHooks hooks;
  hooks.on_visualize =
    [start_time,
     local_dofs,
     global_dofs,
     glvis_windows,
     glvis_enabled,
     num_procs_value,
     my_rank_value,
     callbacks,
     igr_solver,
     ctx_ptr,
     bundle_ptr](int step, mfem::real_t t, mfem::real_t dt_real, bool)
  {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
      std::chrono::duration<double>(now - start_time).count();
    const double avg_step = (step > 0) ? (elapsed / step) : 0.0;
    const double avg_step_per_dof_local =
      (step > 0 && local_dofs > 0.0) ? (avg_step / local_dofs) : 0.0;
    const double avg_step_per_dof_rank =
      (step > 0 && global_dofs > 0.0 && num_procs_value > 0)
        ? (avg_step * static_cast<double>(num_procs_value) / global_dofs)
        : 0.0;

    if (callbacks.on_alive)
    {
      TimeLoopInfo info;
      info.step = step;
      info.t = t;
      info.dt = dt_real;
      info.elapsed = elapsed;
      info.local_dofs = static_cast<int>(local_dofs);
      info.global_dofs = static_cast<long long>(global_dofs);
      info.ranks = num_procs_value;
      info.is_final = false;
      info.avg_step = avg_step;
      info.avg_step_per_dof_local = avg_step_per_dof_local;
      info.avg_step_per_dof_rank = avg_step_per_dof_rank;
      FillIGRTimeInfo(info, elapsed, igr_solver);
      callbacks.on_alive(info);
    }
    if (glvis_enabled && *glvis_enabled && glvis_windows)
    {
      UpdateEulerGLVis(*glvis_windows,
                       bundle_ptr->field_list,
                       bundle_ptr->sol,
                       num_procs_value,
                       my_rank_value,
                       ctx_ptr->Mesh(),
                       t,
                       /*warm_start=*/true);
    }
  };

  hooks.on_save = [paraview, callbacks, bundle_ptr](int step,
                                                     mfem::real_t t,
                                                     int save_idx,
                                                     bool is_final)
  {
    if (callbacks.on_vtk_save)
    {
      VTKSaveInfo info;
      info.cycle = save_idx;
      info.step = step;
      info.t = t;
      info.is_initial = false;
      info.is_final = is_final;
      callbacks.on_vtk_save(info);
    }
    UpdateEulerOutputFields(bundle_ptr->field_list,
                            bundle_ptr->sol,
                            /*warm_start=*/true);
    if (paraview)
    {
      paraview->SetCycle(save_idx);
      paraview->SetTime(t);
      paraview->Save();
    }
  };

  hooks.on_analysis =
    [start_time,
     local_dofs,
     global_dofs,
     num_procs_value,
     callbacks,
     igr_solver,
     ctx_ptr,
     bundle_ptr](int step, mfem::real_t t, mfem::real_t dt_real, bool done)
  {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
      std::chrono::duration<double>(now - start_time).count();
    const double avg_step = (step > 0) ? (elapsed / step) : 0.0;
    const double avg_step_per_dof_local =
      (step > 0 && local_dofs > 0.0) ? (avg_step / local_dofs) : 0.0;
    const double avg_step_per_dof_rank =
      (step > 0 && global_dofs > 0.0 && num_procs_value > 0)
        ? (avg_step * static_cast<double>(num_procs_value) / global_dofs)
        : 0.0;

    if (callbacks.on_analysis)
    {
      AnalysisInfo info;
      info.base.step = step;
      info.base.t = t;
      info.base.dt = dt_real;
      info.base.elapsed = elapsed;
      info.base.local_dofs = static_cast<int>(local_dofs);
      info.base.global_dofs = static_cast<long long>(global_dofs);
      info.base.ranks = num_procs_value;
      info.base.is_final = done;
      info.base.avg_step = avg_step;
      info.base.avg_step_per_dof_local = avg_step_per_dof_local;
      info.base.avg_step_per_dof_rank = avg_step_per_dof_rank;
      FillIGRTimeInfo(info.base, elapsed, igr_solver);

      if (callbacks.wants_minmax)
      {
        if (bundle_ptr->field_list.state)
        {
          const auto rho_mm =
            ComputeGlobalMinMax(bundle_ptr->field_list.state->density,
                                ctx_ptr->Mesh().GetComm());
          info.has_rho = true;
          info.rho_min = rho_mm.first;
          info.rho_max = rho_mm.second;
        }
        if (bundle_ptr->field_list.pressure)
        {
          bundle_ptr->field_list.pressure->Update();
          const auto p_mm =
            ComputeGlobalMinMax(bundle_ptr->field_list.pressure->pressure,
                                ctx_ptr->Mesh().GetComm());
          info.has_p = true;
          info.p_min = p_mm.first;
          info.p_max = p_mm.second;
        }
      }

      callbacks.on_analysis(info);
    }
  };

  return hooks;
}

} // namespace dg_sim

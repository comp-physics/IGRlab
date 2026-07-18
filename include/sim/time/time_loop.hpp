#pragma once

/**
 * @file sim/time/time_loop.hpp
 * @brief Shared explicit time loop used by simulation examples.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <string>

#include "mfem.hpp"
#include "utils/diagnostics/non_finite.hpp"

namespace dg_sim
{

/**
 * @brief Configuration for the shared explicit time loop.
 */
struct TimeLoopConfig
{
  mfem::real_t t_final = 0.0;
  mfem::real_t dt = 0.0;
  mfem::real_t dt_min = 0.0;
  mfem::real_t cfl = 0.0;
  int order = 1;
  int vis_steps = 0;
  int analysis_steps = 0;
  bool vtk_save = true;
  mfem::real_t dt_save = 0.0;
  bool nan_scan = false;
  int save_index_offset = 0;
  mfem::real_t initial_time = 0.0;
  int initial_step = 0;
  mfem::real_t next_save_time = -1.0;
};

/**
 * @brief Optional hooks for visualization and saving during the time loop.
 */
struct TimeLoopHooks
{
  std::function<void(int step, mfem::real_t t, mfem::real_t dt_real, bool done)>
    on_visualize;
  std::function<void(int step, mfem::real_t t, mfem::real_t dt_real, bool done)>
    on_analysis;
  std::function<void(int step, mfem::real_t t, int save_idx, bool is_final)>
    on_save;
};

/**
 * @brief Result summary for the shared explicit time loop.
 */
struct TimeLoopResult
{
  int steps = 0;
  int segment_steps = 0;
  mfem::real_t initial_time = 0.0;
  mfem::real_t final_time = 0.0;
  int non_finite_count = 0;
  bool dt_min_reached = false;
  double non_finite_min = std::numeric_limits<double>::infinity();
  double non_finite_max = -std::numeric_limits<double>::infinity();
};

inline mfem::real_t ComputeGlobalHMin(mfem::ParMesh& pmesh)
{
  mfem::real_t hmin = mfem::infinity();
  for (int i = 0; i < pmesh.GetNE(); ++i)
  {
    hmin = std::min(hmin, pmesh.GetElementSize(i, 1));
  }
  MPI_Allreduce(MPI_IN_PLACE,
                &hmin,
                1,
                mfem::MPITypeMap<mfem::real_t>::mpi_type,
                MPI_MIN,
                pmesh.GetComm());
  return hmin;
}

template <typename OperatorType>
mfem::real_t ComputeCFLDt(OperatorType& op,
                          mfem::ParMesh& pmesh,
                          mfem::ParGridFunction& sol,
                          mfem::real_t cfl,
                          int order,
                          mfem::real_t hmin)
{
  mfem::Vector z(sol.Size());
  op.Mult(sol, z);

  mfem::real_t max_char_speed = op.GetMaxCharSpeed();
  MPI_Allreduce(MPI_IN_PLACE,
                &max_char_speed,
                1,
                mfem::MPITypeMap<mfem::real_t>::mpi_type,
                MPI_MAX,
                pmesh.GetComm());

  if (max_char_speed <= 0.0)
  {
    return cfl * hmin / (2 * order + 1);
  }
  return cfl * hmin / max_char_speed / (2 * order + 1);
}

template <typename OperatorType, typename StepperType>
TimeLoopResult RunTimeLoop(OperatorType& op,
                           mfem::ParMesh& pmesh,
                           mfem::ParGridFunction& sol,
                           StepperType& stepper,
                           const TimeLoopConfig& cfg,
                           const TimeLoopHooks& hooks = {})
{
  TimeLoopResult result;
  MFEM_VERIFY(cfg.initial_step >= 0, "Initial global step must be nonnegative.");
  MFEM_VERIFY(std::isfinite(cfg.initial_time), "Initial time must be finite.");
  MFEM_VERIFY(cfg.t_final >= cfg.initial_time,
              "Final time must not precede the initial/restart time.");
  mfem::real_t t = cfg.initial_time;
  result.steps = cfg.initial_step;
  result.initial_time = t;
  result.final_time = t;
  const mfem::real_t dt_min = std::max<mfem::real_t>(0.0, cfg.dt_min);
  const char* dt_trace_text = std::getenv("DG_IGR_TIME_DT_TRACE");
  const bool trace_nominal_dt =
    dt_trace_text != nullptr && std::string(dt_trace_text) != "0" &&
    std::string(dt_trace_text) != "false";
  auto TraceNominalDt = [&](const char* stage,
                            int step,
                            mfem::real_t time,
                            mfem::real_t value)
  {
    if (trace_nominal_dt && mfem::Mpi::Root())
    {
      std::cout << "[time-loop] event=nominal-dt"
                << " stage=" << stage
                << " step=" << step
                << " time=" << time
                << " dt=" << value << "\n";
    }
  };

  auto CheckDtMinimum = [&](mfem::real_t dt_value, const char* stage) -> bool
  {
    if (dt_min <= 0.0)
    {
      return false;
    }
    if (std::abs(dt_value) >= dt_min)
    {
      return false;
    }
    result.dt_min_reached = true;
    if (mfem::Mpi::Root())
    {
      std::cout << "[dt-check] dt below minimum threshold: dt=" << dt_value
                << " dt_min=" << dt_min << " stage=" << stage << " t=" << t
                << "\n";
    }
    return true;
  };

  mfem::real_t dt = cfg.dt;
  mfem::real_t hmin = mfem::infinity();
  if (cfg.cfl > 0.0)
  {
    hmin = ComputeGlobalHMin(pmesh);
    dt = ComputeCFLDt(op, pmesh, sol, cfg.cfl, cfg.order, hmin);
  }
  TraceNominalDt("initial", cfg.initial_step, t, dt);
  if (!std::isfinite(dt))
  {
    if (mfem::Mpi::Root())
    {
      std::cout << "[dt-check] Non-finite dt before time loop: " << dt << "\n";
    }
    const mfem::dg_igr::NonFiniteSummary summary =
      mfem::dg_igr::ScanNonFinite(sol);
    result.non_finite_count =
      summary.global_non_finite > 0 ? summary.global_non_finite : 1;
    result.non_finite_min = summary.global_min;
    result.non_finite_max = summary.global_max;
    return result;
  }
  if (CheckDtMinimum(dt, "before-time-loop"))
  {
    return result;
  }

  int save_idx = cfg.save_index_offset;
  mfem::real_t t_next_save =
    (cfg.vtk_save && cfg.dt_save > 0.0)
      ? ((cfg.next_save_time > cfg.initial_time) ? cfg.next_save_time
                                                 : cfg.initial_time + cfg.dt_save)
      : 0.0;

  bool done = false;
  for (int segment_step = 0; !done; ++segment_step)
  {
    mfem::real_t dt_real_base = std::min(dt, cfg.t_final - t);
    mfem::real_t dt_real = dt_real_base;

    bool hit_save_boundary = false;
    if (cfg.vtk_save && cfg.dt_save > 0.0)
    {
      const mfem::real_t gap = t_next_save - t;
      if (gap >= 0.0 && gap <= dt_real_base)
      {
        dt_real = gap;
        hit_save_boundary = true;
      }
    }

    if (!std::isfinite(dt_real))
    {
      if (mfem::Mpi::Root())
      {
        std::cout << "[dt-check] Non-finite dt during time loop: " << dt_real
                  << " at t=" << t << "\n";
      }
      const mfem::dg_igr::NonFiniteSummary summary =
        mfem::dg_igr::ScanNonFinite(sol);
      result.non_finite_count =
        summary.global_non_finite > 0 ? summary.global_non_finite : 1;
      result.non_finite_min = summary.global_min;
      result.non_finite_max = summary.global_max;
      break;
    }
    if (dt_real <= 0.0)
    {
      if (mfem::Mpi::Root())
      {
        std::cout << "[dt-check] non-positive dt_real encountered: dt_real="
                  << dt_real << " t=" << t << " dt=" << dt << "\n";
      }
      result.final_time = t;
      done = true;
      if (cfg.vtk_save && hooks.on_save)
      {
        hooks.on_save(result.steps, t, save_idx, /*is_final=*/true);
      }
      break;
    }
    stepper.Step(sol, t, dt_real);
    if (hit_save_boundary)
    {
      t = t_next_save;
    }

    if (cfg.cfl > 0.0)
    {
      mfem::real_t max_char_speed = op.GetMaxCharSpeed();
      MPI_Allreduce(MPI_IN_PLACE,
                    &max_char_speed,
                    1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_MAX,
                    pmesh.GetComm());
      if (max_char_speed > 0.0)
      {
        dt = cfg.cfl * hmin / max_char_speed / (2 * cfg.order + 1);
      }
      else
      {
        dt = cfg.cfl * hmin / (2 * cfg.order + 1);
      }
      TraceNominalDt("update", cfg.initial_step + segment_step + 1, t, dt);
      if (!std::isfinite(dt))
      {
        if (mfem::Mpi::Root())
        {
          std::cout << "[dt-check] Non-finite dt after CFL update: " << dt
                    << " at t=" << t << "\n";
        }
        const mfem::dg_igr::NonFiniteSummary summary =
          mfem::dg_igr::ScanNonFinite(sol);
        result.non_finite_count =
          summary.global_non_finite > 0 ? summary.global_non_finite : 1;
        result.non_finite_min = summary.global_min;
        result.non_finite_max = summary.global_max;
        break;
      }
      if (CheckDtMinimum(dt, "after-cfl-update"))
      {
        result.segment_steps = segment_step + 1;
        result.steps = cfg.initial_step + result.segment_steps;
        result.final_time = t;
        done = true;
        if (cfg.vtk_save && hooks.on_save)
        {
          hooks.on_save(result.steps, t, save_idx, /*is_final=*/true);
        }
        break;
      }
    }

    result.segment_steps = segment_step + 1;
    result.steps = cfg.initial_step + result.segment_steps;
    result.final_time = t;
    done = (t >= cfg.t_final - 1e-8 * std::abs(dt));

    if (cfg.nan_scan)
    {
      const mfem::dg_igr::NonFiniteSummary summary =
        mfem::dg_igr::ScanNonFinite(sol);
      if (summary.global_non_finite > 0)
      {
        result.non_finite_count = summary.global_non_finite;
        result.non_finite_min = summary.global_min;
        result.non_finite_max = summary.global_max;
        if (mfem::Mpi::Root())
        {
          std::cout << "[nan-scan] Detected " << summary.global_non_finite
                    << " non-finite entries at t=" << t
                    << " (finite min=" << summary.global_min
                    << ", finite max=" << summary.global_max << ")\n";
        }
        break;
      }
    }

    const bool should_visualize =
      (cfg.vis_steps > 0) && (done || (result.steps % cfg.vis_steps == 0));
    if (should_visualize && hooks.on_visualize)
    {
      hooks.on_visualize(result.steps, t, dt_real, done);
    }

    const bool should_analyze =
      (cfg.analysis_steps > 0) &&
      (done || (result.steps % cfg.analysis_steps == 0));
    if (should_analyze && hooks.on_analysis)
    {
      hooks.on_analysis(result.steps, t, dt_real, done);
    }

    bool saved_this_step = false;
    if (cfg.vtk_save && cfg.dt_save > 0.0 && hit_save_boundary)
    {
      if (hooks.on_save)
      {
        hooks.on_save(result.steps, t, save_idx, /*is_final=*/done);
      }
      saved_this_step = true;
      ++save_idx;
      t_next_save += cfg.dt_save;
    }

    if (done && cfg.vtk_save && hooks.on_save && !saved_this_step)
    {
      hooks.on_save(result.steps, t, save_idx, /*is_final=*/true);
    }
  }

  return result;
}

} // namespace dg_sim

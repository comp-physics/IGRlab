#pragma once

/**
 * @file sim/time/time_loop.hpp
 * @brief Shared explicit time loop used by simulation examples.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "mfem.hpp"
#include "numerics/limiters/zhang_shu.hpp"
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
  /// The caller already emitted the initial state into this output series.
  bool initial_state_saved = false;
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
  /// Called after conservative rollback and operator abort recovery complete.
  std::function<void(int step,
                     mfem::real_t t,
                     mfem::real_t attempted_dt,
                     mfem::real_t retry_dt)>
    on_rejected_attempt;
};

/**
 * @brief Result summary for the shared explicit time loop.
 */
struct TimeLoopResult
{
  int steps = 0;
  int segment_steps = 0;
  int accepted_attempts = 0;
  int rejected_attempts = 0;
  /// Sum of failed element admissions over rejected limiter attempts.
  long long rejected_limiter_elements = 0;
  long long rhs_evaluations = 0;
  mfem::real_t initial_time = 0.0;
  mfem::real_t final_time = 0.0;
  mfem::real_t minimum_attempted_dt =
    std::numeric_limits<mfem::real_t>::infinity();
  mfem::real_t minimum_accepted_dt =
    std::numeric_limits<mfem::real_t>::infinity();
  mfem::real_t maximum_attempted_dt = 0.0;
  mfem::real_t maximum_accepted_dt = 0.0;
  mfem::real_t minimum_characteristic_speed =
    std::numeric_limits<mfem::real_t>::infinity();
  mfem::real_t maximum_characteristic_speed = 0.0;
  int characteristic_speed_probes = 0;
  int non_finite_count = 0;
  bool dt_min_reached = false;
  bool success = false;
  std::string termination_reason = "not-started";
  double non_finite_min = std::numeric_limits<double>::infinity();
  double non_finite_max = -std::numeric_limits<double>::infinity();
};

namespace time_loop_detail
{

template <typename OperatorType, typename = void>
struct InvariantDomainVolumeSpeedTracking
{
  static void Configure(OperatorType&, bool) { }
};

template <typename OperatorType>
struct InvariantDomainVolumeSpeedTracking<
  OperatorType,
  std::void_t<decltype(std::declval<OperatorType&>()
                         .SetInvariantDomainVolumeSpeedTracking(true))>>
{
  static void Configure(OperatorType& op, bool enabled)
  {
    op.SetInvariantDomainVolumeSpeedTracking(enabled);
  }
};

template <typename StepperType, typename = void>
struct StateGenerationTransaction
{
  static std::uint64_t Capture(const StepperType&) { return 0; }
  static void Restore(StepperType&, std::uint64_t) { }
};

template <typename StepperType>
struct StateGenerationTransaction<
  StepperType,
  std::void_t<decltype(std::declval<const StepperType&>().GetStateGeneration()),
              decltype(std::declval<StepperType&>().RestoreStateGeneration(
                std::declval<std::uint64_t>()))>>
{
  static std::uint64_t Capture(const StepperType& stepper)
  {
    return static_cast<std::uint64_t>(stepper.GetStateGeneration());
  }

  static void Restore(StepperType& stepper, std::uint64_t generation)
  {
    stepper.RestoreStateGeneration(generation);
  }
};

template <typename StepperType, typename = void>
struct StepTransactionRequirement
{
  // Unknown steppers remain conservative: retain rollback support in case
  // they participate in the typed recoverable-failure contract.
  static bool Required(const StepperType&) { return true; }
};

template <typename StepperType>
struct StepTransactionRequirement<
  StepperType,
  std::void_t<decltype(
    std::declval<const StepperType&>().RequiresStepTransaction())>>
{
  static bool Required(const StepperType& stepper)
  {
    return static_cast<bool>(stepper.RequiresStepTransaction());
  }
};

template <typename StepperType, typename = void>
struct RHSEvaluationCounter
{
  static long long Read(const StepperType&) { return 0; }
};

template <typename StepperType>
struct RHSEvaluationCounter<
  StepperType,
  std::void_t<decltype(
    std::declval<const StepperType&>().GetRHSEvaluationCount())>>
{
  static long long Read(const StepperType& stepper)
  {
    return static_cast<long long>(stepper.GetRHSEvaluationCount());
  }
};

template <typename StepperType, typename OperatorType, typename = void>
struct AcceptedCharacteristicSpeed
{
  static void Bind(StepperType&, OperatorType&, bool) { }

  static mfem::real_t Read(const StepperType&, OperatorType& op)
  {
    return op.GetMaxCharSpeed();
  }
};

template <typename StepperType, typename OperatorType>
struct AcceptedCharacteristicSpeed<
  StepperType,
  OperatorType,
  std::void_t<decltype(std::declval<StepperType&>()
                         .SetCharacteristicSpeedReader(
                           std::declval<std::function<mfem::real_t()>>())),
              decltype(std::declval<const StepperType&>()
                         .HasAcceptedCharacteristicSpeed()),
              decltype(std::declval<const StepperType&>()
                         .GetAcceptedCharacteristicSpeed())>>
{
  static void Bind(StepperType& stepper, OperatorType& op, bool enabled)
  {
    if (enabled)
    {
      stepper.SetCharacteristicSpeedReader(
        [&op]() { return op.GetMaxCharSpeed(); });
    }
    else
    {
      stepper.SetCharacteristicSpeedReader(
        std::function<mfem::real_t()>{});
    }
  }

  static mfem::real_t Read(const StepperType& stepper, OperatorType& op)
  {
    return stepper.HasAcceptedCharacteristicSpeed()
             ? stepper.GetAcceptedCharacteristicSpeed()
             : op.GetMaxCharSpeed();
  }
};

template <typename OperatorType, typename = void>
struct OperatorStepTransaction
{
  static void Abort(OperatorType&,
                    const mfem::Vector&,
                    mfem::real_t,
                    std::uint64_t,
                    mfem::real_t)
  {
  }
};

template <typename OperatorType>
struct OperatorStepTransaction<
  OperatorType,
  std::void_t<decltype(std::declval<OperatorType&>().AbortRKStep(
    std::declval<const mfem::Vector&>(),
    std::declval<mfem::real_t>(),
    std::declval<std::uint64_t>(),
    std::declval<mfem::real_t>()))>>
{
  static void Abort(OperatorType& op,
                    const mfem::Vector& restored_state,
                    mfem::real_t restored_time,
                    std::uint64_t restored_generation,
                    mfem::real_t retry_dt)
  {
    op.AbortRKStep(
      restored_state, restored_time, restored_generation, retry_dt);
  }
};

} // namespace time_loop_detail

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
                          mfem::real_t hmin,
                          mfem::real_t* global_max_char_speed = nullptr)
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

  if (global_max_char_speed)
  {
    *global_max_char_speed = max_char_speed;
  }

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
  result.termination_reason = "running";
  MFEM_VERIFY(cfg.initial_step >= 0, "Initial global step must be nonnegative.");
  MFEM_VERIFY(std::isfinite(cfg.initial_time), "Initial time must be finite.");
  MFEM_VERIFY(cfg.t_final >= cfg.initial_time,
              "Final time must not precede the initial/restart time.");
  mfem::real_t t = cfg.initial_time;
  result.steps = cfg.initial_step;
  result.initial_time = t;
  result.final_time = t;
  int communicator_rank = 0;
  MPI_Comm_rank(pmesh.GetComm(), &communicator_rank);
  const bool communicator_root = communicator_rank == 0;
  time_loop_detail::InvariantDomainVolumeSpeedTracking<OperatorType>::Configure(
    op, cfg.cfl > 0.0);
  time_loop_detail::AcceptedCharacteristicSpeed<StepperType,
                                                OperatorType>::Bind(stepper,
                                                                    op,
                                                                    cfg.cfl > 0.0);
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
    if (trace_nominal_dt && communicator_root)
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
    if (communicator_root)
    {
      std::cout << "[dt-check] dt below minimum threshold: dt=" << dt_value
                << " dt_min=" << dt_min << " stage=" << stage << " t=" << t
                << "\n";
    }
    return true;
  };

  int save_idx = cfg.save_index_offset;
  bool has_last_save_time = cfg.initial_state_saved;
  mfem::real_t last_save_time = cfg.initial_time;
  const auto SaveTimeMatches = [](mfem::real_t lhs, mfem::real_t rhs)
  {
    return std::abs(lhs - rhs) <=
           32.0 * std::numeric_limits<mfem::real_t>::epsilon() *
             std::max({mfem::real_t{1.0}, std::abs(lhs), std::abs(rhs)});
  };
  auto EmitSave = [&](int step,
                      mfem::real_t time,
                      int cycle,
                      bool is_final,
                      bool suppress_duplicate_time = false) -> bool
  {
    if (!cfg.vtk_save || !hooks.on_save ||
        (suppress_duplicate_time && has_last_save_time &&
         SaveTimeMatches(time, last_save_time)))
    {
      return false;
    }
    hooks.on_save(step, time, cycle, is_final);
    has_last_save_time = true;
    last_save_time = time;
    return true;
  };

  mfem::real_t dt = cfg.dt;
  mfem::real_t hmin = mfem::infinity();
  const long long cfl_probe_rhs_evaluations = cfg.cfl > 0.0 ? 1 : 0;
  result.rhs_evaluations = cfl_probe_rhs_evaluations;
  if (cfg.cfl > 0.0)
  {
    hmin = ComputeGlobalHMin(pmesh);
    mfem::real_t max_char_speed = 0.0;
    dt = ComputeCFLDt(
      op, pmesh, sol, cfg.cfl, cfg.order, hmin, &max_char_speed);
    result.minimum_characteristic_speed = max_char_speed;
    result.maximum_characteristic_speed = max_char_speed;
    result.characteristic_speed_probes = 1;
  }
  TraceNominalDt("initial", cfg.initial_step, t, dt);
  if (!std::isfinite(dt))
  {
    if (communicator_root)
    {
      std::cout << "[dt-check] Non-finite dt before time loop: " << dt << "\n";
    }
    const mfem::dg_igr::NonFiniteSummary summary =
      mfem::dg_igr::ScanNonFinite(sol);
    result.non_finite_count =
      summary.global_non_finite > 0 ? summary.global_non_finite : 1;
    result.non_finite_min = summary.global_min;
    result.non_finite_max = summary.global_max;
    result.termination_reason = "nonfinite-initial-dt";
    if (summary.global_non_finite == 0)
    {
      EmitSave(result.steps,
               t,
               save_idx,
               /*is_final=*/true,
               /*suppress_duplicate_time=*/true);
    }
    return result;
  }
  if (CheckDtMinimum(dt, "before-time-loop"))
  {
    result.termination_reason = "initial-dt-min";
    EmitSave(result.steps,
             t,
             save_idx,
             /*is_final=*/true,
             /*suppress_duplicate_time=*/true);
    return result;
  }
  mfem::real_t t_next_save =
    (cfg.vtk_save && cfg.dt_save > 0.0)
      ? ((cfg.next_save_time > cfg.initial_time) ? cfg.next_save_time
                                                 : cfg.initial_time + cfg.dt_save)
      : 0.0;
  const bool requires_step_transaction =
    time_loop_detail::StepTransactionRequirement<StepperType>::Required(
      stepper);
  const long long initial_rhs_evaluations =
    time_loop_detail::RHSEvaluationCounter<StepperType>::Read(stepper);
  mfem::Vector pre_attempt_state;
  if (requires_step_transaction)
  {
    pre_attempt_state.SetSize(sol.Size(), sol.GetMemory().GetMemoryType());
    pre_attempt_state.UseDevice(sol.UseDevice());
  }

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
      if (communicator_root)
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
      result.termination_reason = "nonfinite-attempt-dt";
      if (summary.global_non_finite == 0)
      {
        EmitSave(result.steps,
                 t,
                 save_idx,
                 /*is_final=*/true,
                 /*suppress_duplicate_time=*/true);
      }
      break;
    }
    if (dt_real <= 0.0)
    {
      if (communicator_root)
      {
        std::cout << "[dt-check] non-positive dt_real encountered: dt_real="
                  << dt_real << " t=" << t << " dt=" << dt << "\n";
      }
      result.final_time = t;
      done = true;
      const mfem::real_t completion_tolerance =
        1.0e-12 * std::max<mfem::real_t>(1.0, std::abs(cfg.t_final));
      if (t >= cfg.t_final - completion_tolerance)
      {
        result.success = true;
        result.termination_reason = "target-reached";
      }
      else
      {
        result.termination_reason = "nonpositive-attempt-dt";
      }
      EmitSave(result.steps,
               t,
               save_idx,
               /*is_final=*/true,
               /*suppress_duplicate_time=*/true);
      break;
    }

    if (requires_step_transaction)
    {
      pre_attempt_state = sol;
    }
    const mfem::real_t pre_attempt_time = t;
    const std::uint64_t pre_attempt_generation =
      time_loop_detail::StateGenerationTransaction<StepperType>::Capture(
        stepper);
    bool accepted_attempt = false;
    bool terminal_rejection = false;
    constexpr int kMaximumRetriesPerStep = 64;
    int retries_this_step = 0;
    while (!accepted_attempt)
    {
      const mfem::real_t attempted_dt = dt_real;
      result.minimum_attempted_dt =
        std::min(result.minimum_attempted_dt, attempted_dt);
      result.maximum_attempted_dt =
        std::max(result.maximum_attempted_dt, attempted_dt);
      try
      {
        stepper.Step(sol, t, dt_real);
        accepted_attempt = true;
        ++result.accepted_attempts;
        result.minimum_accepted_dt =
          std::min(result.minimum_accepted_dt, attempted_dt);
        result.maximum_accepted_dt =
          std::max(result.maximum_accepted_dt, attempted_dt);
      }
      catch (const mfem::RecoverableLimiterFailure& failure)
      {
        MFEM_VERIFY(requires_step_transaction,
                    "A recoverable limiter rejection requires a pre-attempt "
                    "state transaction.");
        sol = pre_attempt_state;
        t = pre_attempt_time;
        time_loop_detail::StateGenerationTransaction<StepperType>::Restore(
          stepper, pre_attempt_generation);
        sol.ExchangeFaceNbrData();
        ++result.rejected_attempts;
        result.rejected_limiter_elements +=
          failure.GlobalRejectedElementCount();
        ++retries_this_step;

        mfem::real_t retry_dt = 0.0;
        if (retries_this_step < kMaximumRetriesPerStep)
        {
          if (dt_min > 0.0 && attempted_dt > dt_min)
          {
            retry_dt = std::max(dt_min, 0.5 * attempted_dt);
          }
          else if (dt_min <= 0.0)
          {
            retry_dt = 0.5 * attempted_dt;
          }
        }
        const bool retry_is_valid =
          std::isfinite(retry_dt) && retry_dt > 0.0 &&
          retry_dt < attempted_dt;

        time_loop_detail::OperatorStepTransaction<OperatorType>::Abort(
          op,
          sol,
          t,
          pre_attempt_generation,
          retry_is_valid ? retry_dt : 0.0);

        if (hooks.on_rejected_attempt)
        {
          hooks.on_rejected_attempt(result.steps,
                                    t,
                                    attempted_dt,
                                    retry_is_valid ? retry_dt : 0.0);
        }

        if (!retry_is_valid)
        {
          const bool limiter_dt_min =
            dt_min > 0.0 && attempted_dt <= dt_min;
          result.dt_min_reached = limiter_dt_min;
          result.termination_reason = limiter_dt_min
                                        ? "limiter-dt-min"
                                        : "limiter-retry-limit";
          terminal_rejection = true;
          done = true;
          break;
        }

        dt = retry_dt;
        dt_real = retry_dt;
        hit_save_boundary = false;
      }
    }

    if (terminal_rejection)
    {
      result.final_time = t;
      EmitSave(result.steps,
               t,
               save_idx,
               /*is_final=*/true,
               /*suppress_duplicate_time=*/true);
      break;
    }
    if (hit_save_boundary)
    {
      t = t_next_save;
    }

    const mfem::real_t completion_tolerance =
      1.0e-12 * std::max<mfem::real_t>(1.0, std::abs(cfg.t_final));
    const bool accepted_reached_target =
      t >= cfg.t_final - completion_tolerance;
    if (cfg.cfl > 0.0)
    {
      mfem::real_t max_char_speed =
        time_loop_detail::AcceptedCharacteristicSpeed<StepperType,
                                                      OperatorType>::Read(
          stepper, op);
      MPI_Allreduce(MPI_IN_PLACE,
                    &max_char_speed,
                    1,
                    mfem::MPITypeMap<mfem::real_t>::mpi_type,
                    MPI_MAX,
                    pmesh.GetComm());
      result.minimum_characteristic_speed =
        std::min(result.minimum_characteristic_speed, max_char_speed);
      result.maximum_characteristic_speed =
        std::max(result.maximum_characteristic_speed, max_char_speed);
      ++result.characteristic_speed_probes;
      if (!accepted_reached_target && max_char_speed > 0.0)
      {
        dt = cfg.cfl * hmin / max_char_speed / (2 * cfg.order + 1);
      }
      else if (!accepted_reached_target)
      {
        dt = cfg.cfl * hmin / (2 * cfg.order + 1);
      }
      if (!accepted_reached_target)
      {
        TraceNominalDt("update", cfg.initial_step + segment_step + 1, t, dt);
        if (!std::isfinite(dt))
        {
          if (communicator_root)
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
          result.segment_steps = segment_step + 1;
          result.steps = cfg.initial_step + result.segment_steps;
          result.final_time = t;
          result.termination_reason = "nonfinite-cfl-dt";
          if (summary.global_non_finite == 0)
          {
            EmitSave(result.steps,
                     t,
                     save_idx,
                     /*is_final=*/true,
                     /*suppress_duplicate_time=*/true);
          }
          break;
        }
        if (CheckDtMinimum(dt, "after-cfl-update"))
        {
          result.segment_steps = segment_step + 1;
          result.steps = cfg.initial_step + result.segment_steps;
          result.final_time = t;
          result.termination_reason = "cfl-dt-min";
          done = true;
          EmitSave(result.steps,
                   t,
                   save_idx,
                   /*is_final=*/true,
                   /*suppress_duplicate_time=*/true);
          break;
        }
      }
    }

    result.segment_steps = segment_step + 1;
    result.steps = cfg.initial_step + result.segment_steps;
    result.final_time = t;
    done = accepted_reached_target;

    if (cfg.nan_scan)
    {
      const mfem::dg_igr::NonFiniteSummary summary =
        mfem::dg_igr::ScanNonFinite(sol);
      if (summary.global_non_finite > 0)
      {
        result.non_finite_count = summary.global_non_finite;
        result.non_finite_min = summary.global_min;
        result.non_finite_max = summary.global_max;
        result.termination_reason = "nonfinite-state";
        if (communicator_root)
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
      saved_this_step = EmitSave(result.steps,
                                 t,
                                 save_idx,
                                 /*is_final=*/done);
      ++save_idx;
      t_next_save += cfg.dt_save;
    }

    if (done && !saved_this_step)
    {
      EmitSave(result.steps,
               t,
               save_idx,
               /*is_final=*/true,
               /*suppress_duplicate_time=*/true);
    }
  }

  result.rhs_evaluations =
    cfl_probe_rhs_evaluations +
    time_loop_detail::RHSEvaluationCounter<StepperType>::Read(stepper) -
    initial_rhs_evaluations;
  if (result.termination_reason == "running")
  {
    const mfem::real_t completion_tolerance =
      1.0e-12 * std::max<mfem::real_t>(1.0, std::abs(cfg.t_final));
    result.success = result.final_time >= cfg.t_final - completion_tolerance;
    result.termination_reason = result.success ? "target-reached"
                                               : "incomplete";
  }
  if (communicator_root)
  {
    const mfem::real_t minimum_attempted_dt =
      std::isfinite(result.minimum_attempted_dt)
        ? result.minimum_attempted_dt
        : 0.0;
    const mfem::real_t minimum_accepted_dt =
      std::isfinite(result.minimum_accepted_dt)
        ? result.minimum_accepted_dt
        : 0.0;
    const mfem::real_t minimum_characteristic_speed =
      std::isfinite(result.minimum_characteristic_speed)
        ? result.minimum_characteristic_speed
        : 0.0;
    std::cout << "[explicit-time] event=run-summary"
              << " steps=" << result.steps
              << " final_time=" << result.final_time
              << " rhs_evaluations=" << result.rhs_evaluations
              << " accepted_attempts=" << result.accepted_attempts
              << " rejected_attempts=" << result.rejected_attempts
              << " rejected_limiter_elements="
              << result.rejected_limiter_elements
              << " minimum_attempted_dt=" << minimum_attempted_dt
              << " maximum_attempted_dt=" << result.maximum_attempted_dt
              << " minimum_accepted_dt=" << minimum_accepted_dt
              << " maximum_accepted_dt=" << result.maximum_accepted_dt
              << " characteristic_speed_probes="
              << result.characteristic_speed_probes
              << " minimum_characteristic_speed="
              << minimum_characteristic_speed
              << " maximum_characteristic_speed="
              << result.maximum_characteristic_speed
              << " dt_min_reached=" << (result.dt_min_reached ? 1 : 0)
              << " success=" << (result.success ? 1 : 0)
              << " termination_reason=" << result.termination_reason
              << "\n";
  }
  return result;
}

} // namespace dg_sim

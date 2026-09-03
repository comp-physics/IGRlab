#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "mfem.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/timestepping/stage_aware_operator.hpp"
#include "physics/igr/IGR.hpp"

/**
 * @file physics/igr/igr_coupling.hpp
 * @brief Coupling helpers between IGR and DG Euler operators.
 * @ingroup physics_igr
 */

namespace mfem
{

struct IGRSigmaPreparationResult
{
  bool solved = false;
  bool reusable_stage0 = false;
  bool output_only = false;
  bool independent_residual_certified = false;
  double independent_residual_abs =
    std::numeric_limits<double>::quiet_NaN();
  double independent_residual_rel_rhs =
    std::numeric_limits<double>::quiet_NaN();
  double independent_residual_relative_limit =
    std::numeric_limits<double>::quiet_NaN();
  double independent_residual_absolute_limit =
    std::numeric_limits<double>::quiet_NaN();
};

/// Read-only sigma field plus lifecycle preparation for output orchestration.
class IGRSigmaStateProvider
{
public:
  virtual ~IGRSigmaStateProvider() = default;
  virtual const ParGridFunction& GetLiveSigma() const = 0;
  virtual IGRSigmaPreparationResult PrepareSigmaForOutput(
    const ParGridFunction& state,
    std::uint64_t state_generation,
    int next_rk_step,
    real_t time,
    bool reusable_stage0,
    bool cold_start = false,
    bool certify_residual = false) = 0;
  virtual std::uint64_t GetSigmaStateGeneration() const = 0;
  virtual long long GetSigmaPrefetchSolveCount() const = 0;
  virtual long long GetSigmaStage0ReuseCount() const = 0;
  virtual long long GetSigmaOutputOnlySolveCount() const = 0;
  virtual long long GetSigmaPrefetchInvalidationCount() const = 0;
  virtual bool SupportsIndependentResidualCertification() const = 0;
};

/// Wrapper operator: recompute sigma each Mult() and forward to DG.
/// @ingroup physics_igr
class EulerWithIGR : public TimeDependentOperator,
                     public RKStageAwareOperator,
                     public IGRSigmaStateProvider,
                     public PhysicalBoundaryFluxIntegralProvider,
                     public IGRFrozenSigmaTelemetryProvider
{
public:
  EulerWithIGR(DGHyperbolicConservationLaws& base,
               ::dg_igr::IGRSolver& igr,
               ParGridFunction& state)
    : TimeDependentOperator(base.Height()),
      base_(base),
      igr_(igr),
      state_(state)
  {
    sigma_cadence_ = igr_.GetSigmaUpdateCadence();
    sigma_interval_ = std::max(1, igr_.GetSigmaUpdateInterval());
  }

  void SetTime(double t) override
  {
    base_.SetTime(t);
    igr_.SetCurrentTime(t);
  }

  void BeginRKStep(int step, real_t t, real_t dt) override
  {
    BeginRKStep(step, t, dt, /*stage_count=*/3);
  }

  void BeginRKStep(int step,
                   real_t,
                   real_t dt,
                   int stage_count) override
  {
    MFEM_VERIFY(stage_count > 0,
                "EulerWithIGR requires a positive RK stage count.");
    MFEM_VERIFY(!rk_step_snapshot_.active,
                "EulerWithIGR cannot begin a new RK step before the prior "
                "step is accepted or aborted.");
    RestoreTemporaryOutputState();
    CaptureRKStepSnapshot(step);
    rk_step_ = step;
    rk_stage_ = -1;
    rk_stage_count_ = stage_count;
    igr_.SetRKStepIndex(step);
    igr_.SetCurrentTimeStep(dt);
  }

  void SetRKStage(int stage) override
  {
    MFEM_VERIFY(stage >= 0 &&
                  (rk_stage_count_ < 0 || stage < rk_stage_count_),
                "EulerWithIGR received an invalid RK stage index.");
    rk_stage_ = stage;
    rk_state_generation_ = kInvalidGeneration;
  }

  void SetRKStageState(int stage,
                       std::uint64_t state_generation) override
  {
    MFEM_VERIFY(stage >= 0 &&
                  (rk_stage_count_ < 0 || stage < rk_stage_count_),
                "EulerWithIGR received an invalid RK stage index.");
    rk_stage_ = stage;
    rk_state_generation_ = state_generation;
  }

  void EndRKStep(const Vector&,
                 real_t,
                 std::uint64_t state_generation) override
  {
    accepted_state_generation_ = state_generation;
    rk_stage_ = -1;
    rk_stage_count_ = -1;
    rk_state_generation_ = state_generation;
    rk_step_snapshot_.active = false;
  }

  void AbortRKStep(const Vector& restored_state,
                   real_t restored_time,
                   std::uint64_t restored_state_generation,
                   real_t retry_dt) override
  {
    MFEM_VERIFY(rk_step_snapshot_.active,
                "EulerWithIGR received an abort without an active RK step.");
    MFEM_VERIFY(restored_state.Size() == state_.Size(),
                "EulerWithIGR RK abort received an unexpected state size.");

    const int aborted_step = rk_step_;
    const bool update_was_due = rk_step_snapshot_.sigma_update_due;
    const bool will_retry = retry_dt > 0.0;
    if (!update_was_due)
    {
      // In a skipped EveryNSteps step no closure operation can have occurred.
      // Restoring the cadence snapshot is therefore both exact and essential:
      // forcing a current-state solve here would change the selected method.
      RestoreRKStepSnapshot();
      accepted_state_generation_ = restored_state_generation;
      base_.SetTime(restored_time);
      if (!will_retry && igr_.HasPendingStage0PrefetchAccounting())
      {
        AbandonPendingStage0Prefetch(/*restore_warm_start=*/false);
      }
      igr_.SetRKStepIndex(rk_step_);
      igr_.SetRKStageIndex(-1);
      igr_.SetCurrentTime(restored_time);
      igr_.SetCurrentTimeStep(std::numeric_limits<double>::quiet_NaN());
      rk_step_snapshot_.active = false;
      return;
    }

    bool pending_matches_restored = false;
    if (igr_.HasPendingStage0PrefetchAccounting())
    {
      pending_matches_restored =
        prefetched_stage0_generation_ == restored_state_generation &&
        prefetched_stage0_step_ == aborted_step &&
        PrefetchedStateMatches(restored_state);
    }
    bool sigma_matches_restored =
      sigma_initialized_ &&
      sigma_state_generation_ == restored_state_generation;
    if (pending_matches_restored)
    {
      // The pending solve itself is authoritative provenance for live sigma.
      sigma_matches_restored = true;
    }
    else if (igr_.HasPendingStage0PrefetchAccounting())
    {
      // A pending solve for a different state cannot survive the transaction.
      // Its measured work remains accounted as output-only, while its warm
      // start is restored before any retry preparation below.
      AbandonPendingStage0Prefetch();
      sigma_matches_restored = false;
    }

    RestoreRKStepSnapshot();
    accepted_state_generation_ = restored_state_generation;
    base_.SetTime(restored_time);
    igr_.SetCurrentTime(restored_time);

    if (!sigma_matches_restored)
    {
      // The failed attempt advanced sigma to another stage state. Mark that
      // provenance stale before preparing one retry-stage-0 solve from the
      // restored conservative state. Rejected solves remain in total work.
      sigma_state_generation_ = kInvalidGeneration;
      prefetched_stage0_generation_ = kInvalidGeneration;
      prefetched_stage0_step_ = -1;
      ParGridFunction restored_grid_state(state_.ParFESpace(),
                                          const_cast<Vector&>(restored_state),
                                          /*base_offset=*/0);
      if (mfem::Mpi::WorldSize() > 1)
      {
        restored_grid_state.ExchangeFaceNbrData();
      }
      PrepareSigmaForOutput(restored_grid_state,
                            restored_state_generation,
                            aborted_step,
                            restored_time,
                            /*reusable_stage0=*/will_retry);
    }
    else if (will_retry)
    {
      // Stage 0 may already have solved exactly this restored generation.
      // Preserve it as the retry prefetch, whether or not it still has pending
      // accounting in IGRSolver.
      sigma_initialized_ = true;
      sigma_state_generation_ = restored_state_generation;
      prefetched_stage0_generation_ = restored_state_generation;
      prefetched_stage0_step_ = aborted_step;
      prefetched_stage0_state_ = restored_state;
    }
    else
    {
      // A terminal rejection has no production stage 0 to consume a pending
      // solve. Preserve the matching field but classify any pending solve as
      // output-only and clear its retry provenance.
      sigma_initialized_ = true;
      sigma_state_generation_ = restored_state_generation;
      ParGridFunction restored_grid_state(state_.ParFESpace(),
                                          const_cast<Vector&>(restored_state),
                                          /*base_offset=*/0);
      PrepareSigmaForOutput(restored_grid_state,
                            restored_state_generation,
                            aborted_step,
                            restored_time,
                            /*reusable_stage0=*/false);
    }

    // BeginRKStep will set this context again. Keeping it coherent here also
    // supports bounded rejection diagnostics before the retry starts.
    igr_.SetRKStepIndex(aborted_step);
    igr_.SetRKStageIndex(will_retry ? 0 : -1);
    igr_.SetCurrentTimeStep(
      retry_dt > 0.0 ? retry_dt
                     : std::numeric_limits<double>::quiet_NaN());
    rk_step_snapshot_.active = false;
  }

  const ParGridFunction& GetLiveSigma() const override
  {
    return igr_.GetSigma();
  }

  IGRSigmaPreparationResult PrepareSigmaForOutput(
    const ParGridFunction& state,
    std::uint64_t state_generation,
    int next_rk_step,
    real_t time,
    bool reusable_stage0,
    bool cold_start = false,
    bool certify_residual = false) override
  {
    MFEM_VERIFY(state.Size() == state_.Size(),
                "Sigma output preparation received an unexpected state size.");
    if (accepted_state_generation_ != kInvalidGeneration)
    {
      MFEM_VERIFY(state_generation == accepted_state_generation_,
                  "Sigma output preparation received stale state provenance.");
    }

    const auto& output_snapshot = temporary_output_snapshot_;
    const bool sigma_update_due =
      output_snapshot.active
        ? SigmaUpdateDueForStep(output_snapshot.sigma_initialized,
                                output_snapshot.last_sigma_step,
                                next_rk_step)
        : SigmaUpdateDueForStep(next_rk_step);
    const bool reusable_for_transport = reusable_stage0 && sigma_update_due;

    IGRSigmaPreparationResult result;
    result.reusable_stage0 = reusable_for_transport;
    result.output_only = !reusable_for_transport;

    if (temporary_output_snapshot_.active)
    {
      const bool matching_temporary_output =
        temporary_output_snapshot_.output_state_generation ==
          state_generation &&
        sigma_initialized_ && sigma_state_generation_ == state_generation;
      if (matching_temporary_output && !reusable_for_transport &&
          !certify_residual)
      {
        return result;
      }
      RestoreTemporaryOutputState();
    }

    const bool temporary_output = reusable_stage0 && !sigma_update_due;
    if (temporary_output)
    {
      // A fresh output field before a cadence-skipped step is diagnostic only.
      // Settle any earlier prefetch, preserve the production closure, and keep
      // the output field live only until the next transport call.
      if (igr_.HasPendingStage0PrefetchAccounting())
      {
        AbandonPendingStage0Prefetch(/*restore_warm_start=*/true);
      }
      if (!sigma_initialized_ || sigma_state_generation_ != state_generation ||
          certify_residual)
      {
        CaptureTemporaryOutputState(state_generation);
        igr_.SetRKStepIndex(next_rk_step);
        igr_.SetRKStageIndex(-1);
        igr_.SetCurrentTime(time);
        igr_.SetCurrentTimeStep(std::numeric_limits<double>::quiet_NaN());
        igr_.UpdateFromState(state,
                             /*warm_start=*/!cold_start,
                             ::dg_igr::IGRSolvePurpose::OutputOnly,
                             certify_residual);
        result.independent_residual_rel_rhs =
          igr_.GetLastLinearTrueResidualRelRHS();
        result.independent_residual_abs = igr_.GetLastLinearTrueResidualAbs();
        result.independent_residual_relative_limit =
          igr_.GetLastIndependentCertificateRelativeLimit();
        result.independent_residual_absolute_limit =
          igr_.GetLastIndependentCertificateAbsoluteLimit();
        result.independent_residual_certified =
          certify_residual && igr_.GetLastLinearConverged() &&
          std::isfinite(result.independent_residual_rel_rhs);
        if (mfem::Mpi::WorldSize() > 1)
        {
          igr_.GetSigma().ExchangeFaceNbrData();
        }
        sigma_initialized_ = true;
        sigma_state_generation_ = state_generation;
        prefetched_stage0_generation_ = kInvalidGeneration;
        prefetched_stage0_step_ = -1;
        result.solved = true;
        ++sigma_output_only_solve_count_;
      }
      return result;
    }

    if (igr_.HasPendingStage0PrefetchAccounting() &&
        (!reusable_for_transport || certify_residual ||
         prefetched_stage0_generation_ != state_generation ||
         prefetched_stage0_step_ != next_rk_step))
    {
      AbandonPendingStage0Prefetch(
        /*restore_warm_start=*/reusable_for_transport);
    }

    if (!sigma_initialized_ || sigma_state_generation_ != state_generation ||
        certify_residual)
    {
      igr_.SetRKStepIndex(next_rk_step);
      // A reusable solve is the next production stage-0 solve and must honor
      // that stage's configured tolerance. A final output-only solve has no
      // production stage to replace and uses the base tolerance.
      igr_.SetRKStageIndex(reusable_for_transport ? 0 : -1);
      igr_.SetCurrentTime(time);
      igr_.SetCurrentTimeStep(std::numeric_limits<double>::quiet_NaN());
      igr_.UpdateFromState(
        state,
        /*warm_start=*/!cold_start,
        reusable_for_transport
          ? ::dg_igr::IGRSolvePurpose::ProductionStage0Prefetch
          : ::dg_igr::IGRSolvePurpose::OutputOnly,
        certify_residual);
      result.independent_residual_rel_rhs =
        igr_.GetLastLinearTrueResidualRelRHS();
      result.independent_residual_abs = igr_.GetLastLinearTrueResidualAbs();
      result.independent_residual_relative_limit =
        igr_.GetLastIndependentCertificateRelativeLimit();
      result.independent_residual_absolute_limit =
        igr_.GetLastIndependentCertificateAbsoluteLimit();
      result.independent_residual_certified =
        certify_residual && igr_.GetLastLinearConverged() &&
        std::isfinite(result.independent_residual_rel_rhs);
      if (mfem::Mpi::WorldSize() > 1)
      {
        igr_.GetSigma().ExchangeFaceNbrData();
      }
      sigma_initialized_ = true;
      sigma_state_generation_ = state_generation;
      result.solved = true;
      if (reusable_for_transport)
      {
        ++sigma_prefetch_solve_count_;
      }
      else
      {
        ++sigma_output_only_solve_count_;
      }
    }

    if (reusable_for_transport)
    {
      prefetched_stage0_generation_ = state_generation;
      prefetched_stage0_step_ = next_rk_step;
      prefetched_stage0_state_ = state;
    }
    else
    {
      prefetched_stage0_generation_ = kInvalidGeneration;
      prefetched_stage0_step_ = -1;
    }
    return result;
  }

  std::uint64_t GetSigmaStateGeneration() const override
  {
    return sigma_state_generation_;
  }
  long long GetSigmaPrefetchSolveCount() const override
  {
    return sigma_prefetch_solve_count_;
  }
  long long GetSigmaStage0ReuseCount() const override
  {
    return sigma_stage0_reuse_count_;
  }
  long long GetSigmaOutputOnlySolveCount() const override
  {
    return sigma_output_only_solve_count_;
  }
  long long GetSigmaPrefetchInvalidationCount() const override
  {
    return sigma_prefetch_invalidation_count_;
  }
  bool SupportsIndependentResidualCertification() const override
  {
    return igr_.GetSolveMode() == ::dg_igr::IGRSolveMode::Krylov;
  }

  bool SupportsPhysicalBoundaryFluxIntegral() const override
  {
    return base_.SupportsPhysicalBoundaryFluxIntegral();
  }

  void ArmPhysicalBoundaryFluxIntegral() const override
  {
    physical_boundary_flux_integral_armed_ = true;
    physical_boundary_flux_integral_available_ = false;
  }

  bool HasLastPhysicalBoundaryFluxIntegral() const override
  {
    return physical_boundary_flux_integral_available_;
  }

  const Vector& GetLastPhysicalBoundaryFluxIntegral() const override
  {
    MFEM_VERIFY(physical_boundary_flux_integral_available_,
                "No completed Euler+IGR physical-boundary flux integral is "
                "available.");
    return base_.GetLastPhysicalBoundaryFluxIntegral();
  }

  IGRFrozenSigmaTelemetry GetGlobalFrozenSigmaTelemetry() const override
  {
    return base_.GetGlobalFrozenSigmaTelemetry();
  }

  void Mult(const Vector& U, Vector& dUdt) const override
  {
    const bool boundary_flux_requested =
      physical_boundary_flux_integral_armed_;
    physical_boundary_flux_integral_armed_ = false;
    if (boundary_flux_requested)
    {
      physical_boundary_flux_integral_available_ = false;
    }

    // Output-only sigma for a cadence-skipped step must never leak into the
    // transport operator, including direct non-stage-aware Mult() callers.
    RestoreTemporaryOutputState();
    bool update_sigma = false;
    const bool have_rk_step = (rk_step_ >= 0);
    switch (sigma_cadence_)
    {
    case ::dg_igr::IGRSigmaUpdateCadence::EveryStage:
      update_sigma = true;
      break;
    case ::dg_igr::IGRSigmaUpdateCadence::EveryStep:
      update_sigma = !have_rk_step
                       ? true
                       : (!sigma_initialized_ || (last_sigma_step_ != rk_step_));
      break;
    case ::dg_igr::IGRSigmaUpdateCadence::EveryNSteps:
      if (!have_rk_step)
      {
        // Outside stage-aware steppers, prefer safety over cadence fidelity:
        // recompute sigma on every RHS evaluation to avoid stale coupling.
        update_sigma = true;
      }
      else if (!sigma_initialized_)
      {
        update_sigma = true;
      }
      else if (rk_step_ % sigma_interval_ == 0)
      {
        update_sigma = (last_sigma_step_ != rk_step_);
      }
      else
      {
        update_sigma = false;
      }
      break;
    }

    const bool stage0_candidate =
      update_sigma && sigma_initialized_ &&
      prefetched_stage0_generation_ != kInvalidGeneration &&
      sigma_state_generation_ == prefetched_stage0_generation_ &&
      ((rk_stage_ == 0 && rk_step_ == prefetched_stage0_step_ &&
        (rk_state_generation_ == prefetched_stage0_generation_ ||
         PrefetchedStateMatches(U))) ||
       (rk_stage_ < 0 && rk_step_ < 0 && &U == &state_));
    const bool reusable_stage0 = stage0_candidate;

    if (update_sigma && !reusable_stage0)
    {
      if (prefetched_stage0_generation_ != kInvalidGeneration)
      {
        AbandonPendingStage0Prefetch();
      }
      int igr_solve_stage = rk_stage_;
      if (sigma_cadence_ != ::dg_igr::IGRSigmaUpdateCadence::EveryStage)
      {
        // Stage-based rtol loosening is only meaningful when a later stage
        // solve will occur in the same RK step.
        igr_solve_stage = -1;
      }

      MFEM_VERIFY(U.Size() == state_.Size(),
                  "EulerWithIGR::Mult received state with unexpected size.");

      // Build a temporary ParGridFunction view over the stage vector so sigma
      // is updated from the same state used for RHS evaluation.
      ParGridFunction stage_state(state_.ParFESpace(),
                                  const_cast<Vector&>(U),
                                  /*base_offset=*/0);

      // Recompute sigma and expose current RK stage so staged tolerance
      // policies can use looser settings on intermediate stages.
      if (igr_solve_stage >= 0 && rk_stage_count_ > 0)
      {
        igr_.SetRKStageContext(igr_solve_stage, rk_stage_count_);
      }
      else
      {
        igr_.SetRKStageIndex(-1);
      }
      igr_.UpdateFromState(stage_state, /*warm_start=*/true);

      // Make sigma available on face neighbors across MPI subdomains.
      if (mfem::Mpi::WorldSize() > 1)
      {
        igr_.GetSigma().ExchangeFaceNbrData();
      }

      sigma_initialized_ = true;
      sigma_state_generation_ = rk_state_generation_;
      prefetched_stage0_generation_ = kInvalidGeneration;
      prefetched_stage0_step_ = -1;
      if (rk_step_ >= 0)
      {
        last_sigma_step_ = rk_step_;
      }
    }
    else if (reusable_stage0 && rk_stage_ == 0)
    {
      igr_.ConsumePendingStage0PrefetchAccounting();
      sigma_state_generation_ = rk_state_generation_;
      ++sigma_stage0_reuse_count_;
      prefetched_stage0_generation_ = kInvalidGeneration;
      prefetched_stage0_step_ = -1;
      last_sigma_step_ = rk_step_;
    }

    // Delay forwarding the arm until immediately before base_.Mult. If an IGR
    // closure fails first, no armed request can leak into a later CFL probe.
    if (boundary_flux_requested)
    {
      base_.ArmPhysicalBoundaryFluxIntegral();
    }
    base_.Mult(U, dUdt);
    if (boundary_flux_requested)
    {
      physical_boundary_flux_integral_available_ =
        base_.HasLastPhysicalBoundaryFluxIntegral();
    }
  }

  real_t GetMaxCharSpeed() { return base_.GetMaxCharSpeed(); }

  void SetInvariantDomainVolumeSpeedTracking(bool enabled)
  {
    base_.SetInvariantDomainVolumeSpeedTracking(enabled);
  }

  bool GetInvariantDomainVolumeSpeedTracking() const
  {
    return base_.GetInvariantDomainVolumeSpeedTracking();
  }

private:
  static constexpr std::uint64_t kInvalidGeneration =
    std::numeric_limits<std::uint64_t>::max();

  struct RKStepSnapshot
  {
    bool active = false;
    bool sigma_update_due = false;
    int rk_step = -1;
    int rk_stage = -1;
    int rk_stage_count = -1;
    int last_sigma_step = -1;
    bool sigma_initialized = false;
    std::uint64_t rk_state_generation = 0;
    std::uint64_t accepted_state_generation = kInvalidGeneration;
    std::uint64_t sigma_state_generation = kInvalidGeneration;
    std::uint64_t prefetched_stage0_generation = kInvalidGeneration;
    int prefetched_stage0_step = -1;
  };

  struct TemporaryOutputSnapshot
  {
    bool active = false;
    bool sigma_initialized = false;
    int last_sigma_step = -1;
    std::uint64_t sigma_state_generation = kInvalidGeneration;
    std::uint64_t prefetched_stage0_generation = kInvalidGeneration;
    int prefetched_stage0_step = -1;
    std::uint64_t output_state_generation = kInvalidGeneration;
    Vector sigma;
    Vector prefetched_stage0_state;
  };

  bool SigmaUpdateDueForStep(bool sigma_initialized,
                             int last_sigma_step,
                             int step) const
  {
    switch (sigma_cadence_)
    {
    case ::dg_igr::IGRSigmaUpdateCadence::EveryStage:
      return true;
    case ::dg_igr::IGRSigmaUpdateCadence::EveryStep:
      return !sigma_initialized || last_sigma_step != step;
    case ::dg_igr::IGRSigmaUpdateCadence::EveryNSteps:
      return !sigma_initialized ||
             (step % sigma_interval_ == 0 && last_sigma_step != step);
    }
    return true;
  }

  bool SigmaUpdateDueForStep(int step) const
  {
    return SigmaUpdateDueForStep(
      sigma_initialized_, last_sigma_step_, step);
  }

  void CaptureTemporaryOutputState(std::uint64_t output_state_generation)
  {
    MFEM_VERIFY(!temporary_output_snapshot_.active,
                "Nested temporary sigma output preparation is unsupported.");
    MFEM_VERIFY(!igr_.HasPendingStage0PrefetchAccounting(),
                "Temporary sigma output requires settled prefetch accounting.");
    temporary_output_snapshot_.active = true;
    temporary_output_snapshot_.sigma_initialized = sigma_initialized_;
    temporary_output_snapshot_.last_sigma_step = last_sigma_step_;
    temporary_output_snapshot_.sigma_state_generation =
      sigma_state_generation_;
    temporary_output_snapshot_.prefetched_stage0_generation =
      prefetched_stage0_generation_;
    temporary_output_snapshot_.prefetched_stage0_step =
      prefetched_stage0_step_;
    temporary_output_snapshot_.output_state_generation =
      output_state_generation;
    temporary_output_snapshot_.sigma = igr_.GetSigma();
    temporary_output_snapshot_.prefetched_stage0_state =
      prefetched_stage0_state_;
  }

  void RestoreTemporaryOutputState() const
  {
    if (!temporary_output_snapshot_.active)
    {
      return;
    }
    MFEM_VERIFY(!igr_.HasPendingStage0PrefetchAccounting(),
                "Temporary output sigma unexpectedly owns production "
                "prefetch accounting.");
    igr_.GetSigma() = temporary_output_snapshot_.sigma;
    if (mfem::Mpi::WorldSize() > 1)
    {
      igr_.GetSigma().ExchangeFaceNbrData();
    }
    sigma_initialized_ = temporary_output_snapshot_.sigma_initialized;
    last_sigma_step_ = temporary_output_snapshot_.last_sigma_step;
    sigma_state_generation_ =
      temporary_output_snapshot_.sigma_state_generation;
    prefetched_stage0_generation_ =
      temporary_output_snapshot_.prefetched_stage0_generation;
    prefetched_stage0_step_ =
      temporary_output_snapshot_.prefetched_stage0_step;
    prefetched_stage0_state_ =
      temporary_output_snapshot_.prefetched_stage0_state;
    temporary_output_snapshot_.active = false;
  }

  void CaptureRKStepSnapshot(int next_step)
  {
    rk_step_snapshot_.active = true;
    rk_step_snapshot_.sigma_update_due = SigmaUpdateDueForStep(next_step);
    rk_step_snapshot_.rk_step = rk_step_;
    rk_step_snapshot_.rk_stage = rk_stage_;
    rk_step_snapshot_.rk_stage_count = rk_stage_count_;
    rk_step_snapshot_.last_sigma_step = last_sigma_step_;
    rk_step_snapshot_.sigma_initialized = sigma_initialized_;
    rk_step_snapshot_.rk_state_generation = rk_state_generation_;
    rk_step_snapshot_.accepted_state_generation = accepted_state_generation_;
    rk_step_snapshot_.sigma_state_generation = sigma_state_generation_;
    rk_step_snapshot_.prefetched_stage0_generation =
      prefetched_stage0_generation_;
    rk_step_snapshot_.prefetched_stage0_step = prefetched_stage0_step_;
  }

  void RestoreRKStepSnapshot()
  {
    rk_step_ = rk_step_snapshot_.rk_step;
    rk_stage_ = rk_step_snapshot_.rk_stage;
    rk_stage_count_ = rk_step_snapshot_.rk_stage_count;
    last_sigma_step_ = rk_step_snapshot_.last_sigma_step;
    sigma_initialized_ = rk_step_snapshot_.sigma_initialized;
    rk_state_generation_ = rk_step_snapshot_.rk_state_generation;
    accepted_state_generation_ =
      rk_step_snapshot_.accepted_state_generation;
    sigma_state_generation_ = rk_step_snapshot_.sigma_state_generation;
    prefetched_stage0_generation_ =
      rk_step_snapshot_.prefetched_stage0_generation;
    prefetched_stage0_step_ = rk_step_snapshot_.prefetched_stage0_step;
  }

  void AbandonPendingStage0Prefetch(bool restore_warm_start = true) const
  {
    if (igr_.ReclassifyPendingStage0PrefetchAsOutputOnly(restore_warm_start))
    {
      ++sigma_prefetch_invalidation_count_;
      ++sigma_output_only_solve_count_;
      if (restore_warm_start)
      {
        sigma_state_generation_ = kInvalidGeneration;
      }
    }
    prefetched_stage0_generation_ = kInvalidGeneration;
    prefetched_stage0_step_ = -1;
  }

  bool PrefetchedStateMatches(const Vector& state) const
  {
    if (prefetched_stage0_state_.Size() != state.Size())
    {
      return false;
    }
    const real_t* current = state.HostRead();
    const real_t* prepared = prefetched_stage0_state_.HostRead();
    int local_match =
      std::memcmp(current,
                  prepared,
                  static_cast<size_t>(state.Size()) * sizeof(real_t)) == 0
        ? 1
        : 0;
#ifdef MFEM_USE_MPI
    MPI_Allreduce(MPI_IN_PLACE,
                  &local_match,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  state_.ParFESpace()->GetComm());
#endif
    return local_match == 1;
  }

  DGHyperbolicConservationLaws& base_;
  ::dg_igr::IGRSolver& igr_;
  ParGridFunction& state_; // view of current state (vdim=num_eq)
  ::dg_igr::IGRSigmaUpdateCadence sigma_cadence_ =
    ::dg_igr::IGRSigmaUpdateCadence::EveryStage;
  int sigma_interval_ = 1;
  mutable int rk_step_ = -1;
  mutable int rk_stage_ = -1;
  mutable int rk_stage_count_ = -1;
  mutable int last_sigma_step_ = -1;
  mutable bool sigma_initialized_ = false;
  mutable std::uint64_t rk_state_generation_ = 0;
  mutable std::uint64_t accepted_state_generation_ = kInvalidGeneration;
  mutable std::uint64_t sigma_state_generation_ = kInvalidGeneration;
  mutable std::uint64_t prefetched_stage0_generation_ = kInvalidGeneration;
  mutable int prefetched_stage0_step_ = -1;
  mutable Vector prefetched_stage0_state_;
  mutable long long sigma_prefetch_solve_count_ = 0;
  mutable long long sigma_stage0_reuse_count_ = 0;
  mutable long long sigma_output_only_solve_count_ = 0;
  mutable long long sigma_prefetch_invalidation_count_ = 0;
  mutable bool physical_boundary_flux_integral_armed_ = false;
  mutable bool physical_boundary_flux_integral_available_ = false;
  mutable RKStepSnapshot rk_step_snapshot_;
  mutable TemporaryOutputSnapshot temporary_output_snapshot_;
};

} // namespace mfem

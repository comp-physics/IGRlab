#pragma once

#include <algorithm>
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
    bool reusable_stage0) = 0;
  virtual std::uint64_t GetSigmaStateGeneration() const = 0;
  virtual long long GetSigmaPrefetchSolveCount() const = 0;
  virtual long long GetSigmaStage0ReuseCount() const = 0;
  virtual long long GetSigmaOutputOnlySolveCount() const = 0;
  virtual long long GetSigmaPrefetchInvalidationCount() const = 0;
};

/// Wrapper operator: recompute sigma each Mult() and forward to DG.
/// @ingroup physics_igr
class EulerWithIGR : public TimeDependentOperator,
                     public RKStageAwareOperator,
                     public IGRSigmaStateProvider
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

  void BeginRKStep(int step, real_t, real_t dt) override
  {
    rk_step_ = step;
    rk_stage_ = -1;
    igr_.SetRKStepIndex(step);
    igr_.SetCurrentTimeStep(dt);
  }

  void SetRKStage(int stage) override
  {
    rk_stage_ = stage;
    rk_state_generation_ = kInvalidGeneration;
  }

  void SetRKStageState(int stage,
                       std::uint64_t state_generation) override
  {
    rk_stage_ = stage;
    rk_state_generation_ = state_generation;
  }

  void EndRKStep(const Vector&,
                 real_t,
                 std::uint64_t state_generation) override
  {
    accepted_state_generation_ = state_generation;
    rk_stage_ = -1;
    rk_state_generation_ = state_generation;
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
    bool reusable_stage0) override
  {
    MFEM_VERIFY(state.Size() == state_.Size(),
                "Sigma output preparation received an unexpected state size.");
    if (accepted_state_generation_ != kInvalidGeneration)
    {
      MFEM_VERIFY(state_generation == accepted_state_generation_,
                  "Sigma output preparation received stale state provenance.");
    }

    IGRSigmaPreparationResult result;
    result.reusable_stage0 = reusable_stage0;
    result.output_only = !reusable_stage0;

    if (igr_.HasPendingStage0PrefetchAccounting() &&
        (!reusable_stage0 ||
         prefetched_stage0_generation_ != state_generation ||
         prefetched_stage0_step_ != next_rk_step))
    {
      AbandonPendingStage0Prefetch(/*restore_warm_start=*/reusable_stage0);
    }

    if (!sigma_initialized_ || sigma_state_generation_ != state_generation)
    {
      igr_.SetRKStepIndex(next_rk_step);
      // A reusable solve is the next production stage-0 solve and must honor
      // that stage's configured tolerance. A final output-only solve has no
      // production stage to replace and uses the base tolerance.
      igr_.SetRKStageIndex(reusable_stage0 ? 0 : -1);
      igr_.SetCurrentTime(time);
      igr_.SetCurrentTimeStep(std::numeric_limits<double>::quiet_NaN());
      igr_.UpdateFromState(
        state,
        /*warm_start=*/true,
        reusable_stage0
          ? ::dg_igr::IGRSolvePurpose::ProductionStage0Prefetch
                        : ::dg_igr::IGRSolvePurpose::OutputOnly);
      if (mfem::Mpi::WorldSize() > 1)
      {
        igr_.GetSigma().ExchangeFaceNbrData();
      }
      sigma_initialized_ = true;
      sigma_state_generation_ = state_generation;
      result.solved = true;
      if (reusable_stage0)
      {
        ++sigma_prefetch_solve_count_;
      }
      else
      {
        ++sigma_output_only_solve_count_;
      }
    }

    if (reusable_stage0)
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

  void Mult(const Vector& U, Vector& dUdt) const override
  {
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
      igr_.SetRKStageIndex(igr_solve_stage);
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

    // Forward to the DG operator
    base_.Mult(U, dUdt);
  }

  real_t GetMaxCharSpeed() { return base_.GetMaxCharSpeed(); }

private:
  static constexpr std::uint64_t kInvalidGeneration =
    std::numeric_limits<std::uint64_t>::max();

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
};

} // namespace mfem

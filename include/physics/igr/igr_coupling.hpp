#pragma once

#include <algorithm>

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

/// Wrapper operator: recompute sigma each Mult() and forward to DG.
/// @ingroup physics_igr
class EulerWithIGR : public TimeDependentOperator, public RKStageAwareOperator
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

  void SetTime(double t) override { base_.SetTime(t); }

  void BeginRKStep(int step, real_t, real_t) override
  {
    rk_step_ = step;
    rk_stage_ = -1;
  }

  void SetRKStage(int stage) override { rk_stage_ = stage; }

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

    if (update_sigma)
    {
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
      if (rk_step_ >= 0)
      {
        last_sigma_step_ = rk_step_;
      }
    }

    // Forward to the DG operator
    base_.Mult(U, dUdt);
  }

  real_t GetMaxCharSpeed() { return base_.GetMaxCharSpeed(); }

private:
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
};

} // namespace mfem

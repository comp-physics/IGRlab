#pragma once

#include "mfem.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
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
class EulerWithIGR : public TimeDependentOperator
{
public:
  EulerWithIGR(DGHyperbolicConservationLaws& base,
               ::dg_igr::IGRSolver& igr,
               ParGridFunction& state)
    : TimeDependentOperator(base.Height()),
      base_(base), igr_(igr), state_(state) {}

  void SetTime(double t) override { base_.SetTime(t); }

  void Mult(const Vector& U, Vector& dUdt) const override
  {
    // Recompute Sigma with a few warm-started sweeps;
    // FluxWithIGR reads Sigma directly (as a ParGridFunction*) so
    // no explicit plumbing into DG is needed here.
    igr_.UpdateFromState(state_, /*warm_start=*/true);

    // IMPORTANT: make Sigma available on face neighbors across MPI subdomains
    igr_.GetSigma().ExchangeFaceNbrData();

    // Forward to the DG operator
    base_.Mult(U, dUdt);
  }

  real_t GetMaxCharSpeed() { return base_.GetMaxCharSpeed(); }

private:
  DGHyperbolicConservationLaws& base_;
  ::dg_igr::IGRSolver&          igr_;
  ParGridFunction&              state_;  // view of current state (vdim=num_eq)
};

} // namespace mfem

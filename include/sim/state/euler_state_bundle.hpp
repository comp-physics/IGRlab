#pragma once

/**
 * @file sim/state/euler_state_bundle.hpp
 * @brief Bundled Euler state and visualization fields.
 */

#include "mfem.hpp"
#include "physics/igr/IGR.hpp"

#include "sim/context/example_context.hpp"
#include "sim/state/state_views.hpp"
#include "sim/io/vis_helpers.hpp"

namespace dg_igr_examples
{

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

} // namespace dg_igr_examples

//                                Supersonic Cylinder - Parallel Version
//
// Compile with: make supersonic_cylinder
//
// Sample runs:
//
//       mpirun -np 4 ex18p -p 1 -rs 2 -rp 1 -o 1 -s 3
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 3 -s 4
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 5 -s 6
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 1 -s 3 -mf
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 3 -s 3 -mf
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation in parallel.
//
//                (u_t, v)_T - (F(u), ∇ v)_T + <F̂(u,n), [[v]]>_F = 0
//
//               where (⋅,⋅)_T is volume integration, and <⋅,⋅>_F is face
//               integration, F is the Euler flux function, and F̂ is the
//               numerical flux.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates usage of mfem::DGHyperbolicConservationLaws
//               that wraps NonlinearFormIntegrators containing element and face
//               integration schemes. In this case the system also involves an
//               external approximate Riemann solver for the DG interface flux.
//               By default, weak-divergence is pre-assembled in element-wise
//               manner, which corresponds to (I_h(F(u_h)), ∇ v). This yields
//               better performance and similar accuracy for the included test
//               problems. This can be turned off and use nonlinear assembly
//               similar to matrix-free assembly when -mf flag is provided.
//               It also demonstrates how to use GLVis for in-situ visualization
//               of vector grid function and how to set top-view.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include "mfem.hpp"

#include <iostream>
#include <memory>

#include "../common/solver.hpp"

#include "double_mach_ref.hpp"

namespace
{

struct DoubleMachOptions
{
  dg_igr_examples::InflowOptions inflow;
};

} // namespace

int main(int argc, char *argv[])
{
  dg_igr_examples::CaseSpec<DoubleMachOptions, mfem::RusanovFluxWithSides> spec;
  spec.case_name = "double_mach_ref";
  spec.default_mesh_path = mfem::DoubleMachRefDefaultMeshPath();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.need_sigma = true;
  spec.need_h1 = true;
  spec.vis.h1_diagnostics = true;
  spec.vis.precision = 8;

  spec.configure_case = [](dg_igr_examples::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 4.0;
    cfg.limiter_pipeline.venkat_mode = mfem::VenkatLimiterMode::kXY;
  };

  spec.add_options = [](mfem::OptionsParser& args, DoubleMachOptions& opts)
  {
    opts.inflow.vx = 2.0;
    dg_igr_examples::AddInflowOptions(args, opts.inflow);
  };

  spec.make_ic = [](const DoubleMachOptions&,
                    const dg_igr_examples::ExampleContext& ctx,
                    double gamma)
  {
    const int base_cells_x = 32;
    const int base_cells_y = 126;
    const int total_ref = ctx.run_cfg.opts.ser_ref_levels
                          + ctx.run_cfg.opts.par_ref_levels;
    const int mx = base_cells_x << total_ref;
    const int my = base_cells_y << total_ref;
    auto ic_fun = mfem::DoubleMachReflectionInitialCondition(mx, my, gamma);
    return std::make_unique<mfem::VectorFunctionCoefficient>(ic_fun);
  };

  spec.build_markers = [](dg_igr_examples::ExampleContext& ctx)
  {
    return dg_igr_examples::BuildBoundaryMarkers(
        ctx.Mesh(),
        {{dg_igr_examples::BoundaryRole::BottomInlet, {1}},
         {dg_igr_examples::BoundaryRole::BottomWall, {5}},
         {dg_igr_examples::BoundaryRole::Outlet, {2}},
         {dg_igr_examples::BoundaryRole::Top, {3}},
         {dg_igr_examples::BoundaryRole::Inlet, {4}}});
  };

  spec.apply_bc = [](const DoubleMachOptions&,
                     const dg_igr_examples::ExampleContext&,
                     auto& bc,
                     const auto& markers)
  {
    const mfem::real_t gamma = 1.4;
    mfem::Vector Ub = mfem::DoubleMachPostShockState(gamma, /*M=*/2.0);
    bc.AddSupersonicInletFromBuilder(markers, [&]() { return Ub; });
    bc.AddSupersonicOutlet(markers, dg_igr_examples::BoundaryRole::BottomInlet);
    bc.AddSupersonicOutlet(markers);
    bc.AddSupersonicOutlet(markers, dg_igr_examples::BoundaryRole::Top);
    bc.AddSlipWall(markers, dg_igr_examples::BoundaryRole::BottomWall);
  };

  spec.postprocess = [](const DoubleMachOptions&,
                        const dg_igr_examples::ExampleContext&,
                        const dg_igr_examples::EulerStateBundle& state,
                        mfem::VectorCoefficient& ic)
  {
    const mfem::real_t error = state.sol.ComputeLpError(2, ic);
    if (mfem::Mpi::Root())
    {
      std::cout << "Solution error: " << error << std::endl;
    }
    return 0;
  };

  return dg_igr_examples::Solver::solve(spec, argc, argv);
}

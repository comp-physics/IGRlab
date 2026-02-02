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

#include "sim/igr_sim.hpp"

#include "supersonic_cylinder.hpp"

namespace
{

struct SupersonicOptions
{};

} // namespace

int main(int argc, char *argv[])
{
  dg_igr_examples::CaseSpec<SupersonicOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "supersonic_cylinder";
  spec.default_mesh_path = mfem::SupersonicCylinderDefaultMeshPath();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.use_igr = false;
  spec.need_sigma = false;
  spec.need_h1 = false;
  spec.vis.precision = 8;

  spec.configure_case = [](dg_igr_examples::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 2.0;
    cfg.defaults.vtk_save = false;
    cfg.rk3_config.limit_initial = false;
    cfg.limiter_pipeline.venkat_mode = mfem::VenkatLimiterMode::kNone;
    cfg.limiter_pipeline.zhang_shu = false;
  };

  spec.make_ic = [](const SupersonicOptions&,
                    const dg_igr_examples::ExampleContext&,
                    double gamma)
  {
    auto ic_fun = mfem::EulerInitialCondition();
    return std::make_unique<mfem::VectorFunctionCoefficient>(ic_fun);
  };

  spec.build_markers = [](dg_igr_examples::ExampleContext& ctx)
  {
    return dg_igr_examples::BuildBoundaryMarkers(
        ctx.Mesh(),
        {{dg_igr_examples::BoundaryRole::Inlet, {4}},
         {dg_igr_examples::BoundaryRole::Outlet, {2}},
         {dg_igr_examples::BoundaryRole::Slip, {1, 3}},
         {dg_igr_examples::BoundaryRole::Cylinder, {5}}});
  };

  spec.apply_bc = [](const SupersonicOptions&,
                     const dg_igr_examples::ExampleContext&,
                     auto& bc,
                     const auto& markers)
  {
    mfem::Vector Uin(4);
    Uin(0) = 1.4;
    Uin(1) = 1.4 * 1.2;
    Uin(2) = 0.0;
    Uin(3) = 8.8;

    bc.AddSupersonicInletFromBuilder(markers, [&]() { return Uin; });
    bc.AddSupersonicOutlet(markers);
    bc.AddSlipWall(markers);
    bc.AddSlipWall(markers, dg_igr_examples::BoundaryRole::Cylinder);
  };

  spec.postprocess = [](const SupersonicOptions&,
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

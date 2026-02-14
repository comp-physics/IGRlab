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

#include <cmath>
#include <iostream>
#include <memory>

#include "sim/igr_sim.hpp"

#include "shock_vortex_interaction.hpp"

namespace
{

using ShockVortexOptions = dg_sim::EmptyCaseOptions;

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<ShockVortexOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "shock_vortex_interaction";
  spec.default_mesh_path = mfem::ShockVortexDefaultMeshPath();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.need_h1 = true;
  spec.diagnostics.h1_diagnostics = true;
  spec.glvis.precision = 8;
  spec.vtk.print_dataset_name = true;

  spec.configure_case = [](dg_sim::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 6.0;
    cfg.rk3_config.limit_initial = false;
  };

  spec.make_ic = [](const ShockVortexOptions&,
                    const dg_sim::ExampleContext& ctx,
                    double gamma)
  {
    const mfem::real_t Ms = 1.2;
    const mfem::real_t Mv = 0.25;
    const int base_cells = 32;
    const int total_ref =
      ctx.run_cfg.opts.ser_ref_levels + ctx.run_cfg.opts.par_ref_levels;
    const int mx = base_cells << total_ref;
    const int my = base_cells << total_ref;
    auto ic_fun = mfem::GetShockVortexInoueHattoriInit(gamma, mx, my, Ms, Mv);
    return std::make_unique<mfem::VectorFunctionCoefficient>(4, ic_fun);
  };

  spec.build_markers = [](dg_sim::ExampleContext& ctx)
  {
    return dg_sim::BuildBoundaryMarkers(ctx.Mesh(),
                                        {{dg_sim::BoundaryRole::Inlet, {2}},
                                         {dg_sim::BoundaryRole::Outlet, {4}},
                                         {dg_sim::BoundaryRole::Slip, {1, 3}}});
  };

  spec.apply_bc = [](const ShockVortexOptions&,
                     const dg_sim::ExampleContext&,
                     auto& bc,
                     const auto& markers)
  {
    const mfem::real_t gamma = 1.4;
    const mfem::real_t Ms = 1.2;
    const mfem::real_t rho1 = 1.0;
    const mfem::real_t p1 = 1.0;
    const mfem::real_t c1 = std::sqrt(gamma * p1 / rho1);
    const mfem::real_t u1x = -Ms * c1;
    const mfem::real_t u1y = 0.0;
    const mfem::real_t v1sq = u1x * u1x + u1y * u1y;
    mfem::Vector Uin(4);
    Uin(0) = rho1;
    Uin(1) = rho1 * u1x;
    Uin(2) = rho1 * u1y;
    Uin(3) = p1 / (gamma - 1.0) + 0.5 * rho1 * v1sq;

    bc.AddSupersonicInletFromBuilder(markers, [&]() { return Uin; });
    bc.AddSupersonicOutlet(markers);
    bc.AddSlipWall(markers);
  };

  spec.postprocess = [](const ShockVortexOptions&,
                        const dg_sim::ExampleContext&,
                        const dg_sim::EulerStateBundle& state,
                        mfem::VectorCoefficient& ic)
  {
    const mfem::real_t error = state.sol.ComputeLpError(2, ic);
    if (mfem::Mpi::Root())
    {
      std::cout << "Solution error: " << error << std::endl;
    }
    return 0;
  };

  return dg_sim::Solver::solve(spec, argc, argv);
}

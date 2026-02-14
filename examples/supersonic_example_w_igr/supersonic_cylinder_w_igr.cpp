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

#include "supersonic_cylinder_w_igr.hpp"

namespace
{

struct InflowOptions
{
  double vx = 3.0;
  double vy = 0.0;
};

inline void AddInflowOptions(mfem::OptionsParser& args,
                             InflowOptions& inflow_opts)
{
  args.AddOption(&inflow_opts.vx,
                 "-vx",
                 "--velocity-x",
                 "Free-stream / inflow velocity x-component.");
  args.AddOption(&inflow_opts.vy,
                 "-vy",
                 "--velocity-y",
                 "Free-stream / inflow velocity y-component.");
}

struct SupersonicIGROptions
{
  InflowOptions inflow;
};

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<SupersonicIGROptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "supersonic_cylinder_w_igr";
  spec.default_mesh_path = mfem::SupersonicCylinderIGRDefaultMeshPath();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.need_h1 = true;
  spec.diagnostics.h1_diagnostics = true;
  spec.glvis.precision = 8;

  spec.configure_case = [](dg_sim::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 4.0;
    cfg.rk3_config.limit_initial = false;
    //  cfg.igr_config.smoot = dg_igr::IGRSmootherType::L1GS;
    cfg.limiter_pipeline.zhang_shu = true;
    //  cfg.limiter_pipeline.krivodonova = true;
  };

  spec.add_options = [](mfem::OptionsParser& args, SupersonicIGROptions& opts)
  {
    AddInflowOptions(args, opts.inflow);
  };

  spec.make_ic = [](const SupersonicIGROptions& opts,
                    const dg_sim::ExampleContext& ctx,
                    double gamma)
  {
    mfem::Vector Uin(4);
    Uin(0) = 1.4;
    Uin(1) = 1.4 * opts.inflow.vx;
    Uin(2) = 1.4 * opts.inflow.vy;
    Uin(3) = 8.8;

    const double x_s = 0.2;
    const double hmin =
      dg_sim::ComputeAlphaInfo(ctx.Mesh(), ctx.run_cfg.igr_opts.c_alpha_user)
        .hmin;
    const double delta = hmin * 15.0;

    const double rho_in = 1.0;
    const double u_in = opts.inflow.vx;
    const double v_in = opts.inflow.vy;
    const double p_in = 1.0;
    mfem::Vector Uamb;
    mfem::BuildInletAndAmbientStates_FromPrimitives(rho_in,
                                                    u_in,
                                                    v_in,
                                                    p_in,
                                                    gamma,
                                                    Uin,
                                                    Uamb);
    auto ic_fun = mfem::SmoothInflowBufferIC_2D(Uin, Uamb, gamma, x_s, delta);
    return std::make_unique<mfem::VectorFunctionCoefficient>(ic_fun);
  };

  spec.build_markers = [](dg_sim::ExampleContext& ctx)
  {
    return dg_sim::BuildBoundaryMarkers(
      ctx.Mesh(),
      {{dg_sim::BoundaryRole::Inlet, {4}},
       {dg_sim::BoundaryRole::Outlet, {2}},
       {dg_sim::BoundaryRole::Slip, {1, 3}},
       {dg_sim::BoundaryRole::Cylinder, {5}}});
  };

  spec.apply_bc = [](const SupersonicIGROptions& opts,
                     const dg_sim::ExampleContext&,
                     auto& bc,
                     const auto& markers)
  {
    mfem::Vector Uin(4);
    Uin(0) = 1.4;
    Uin(1) = 1.4 * opts.inflow.vx;
    Uin(2) = 1.4 * opts.inflow.vy;
    Uin(3) = 8.8;

    bc.AddSupersonicInletFromBuilder(markers, [&]() { return Uin; });
    bc.AddSupersonicOutlet(markers);
    bc.AddSlipWall(markers);
    bc.AddSlipWall(markers, dg_sim::BoundaryRole::Cylinder);
  };

  spec.postprocess = [](const SupersonicIGROptions&,
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

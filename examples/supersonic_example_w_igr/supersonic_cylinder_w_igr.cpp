#include "igrlab/compat.hpp"
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

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

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
  const char* startup = "normal";
  double startup_width = 0.0;
};

mfem::VectorFunctionCoefficient MakeCylinderStartup(
  const SupersonicIGROptions& opts, const mfem::Vector& Uin,
  double gamma, double hmin)
{
  const std::string mode = opts.startup;
  MFEM_VERIFY(mode == "normal" || mode == "full",
              "Cylinder startup must be normal or full.");
  MFEM_VERIFY(std::isfinite(opts.startup_width) && opts.startup_width >= 0.0,
              "Cylinder startup width must be finite and nonnegative.");
  const double delta = opts.startup_width > 0.0
                         ? opts.startup_width : std::max(8.0 * hmin, 0.08);
  if (mfem::Mpi::Root())
  {
    std::cout << "[cylinder-startup] mode=" << mode << " width=" << delta
              << " width_policy=" << (opts.startup_width > 0.0 ? "physical" : "legacy-hmin")
              << " center_x=0.6 center_y=0 radius=0.25" << std::endl;
  }
  return mode == "full"
           ? mfem::SupersonicCylinderSmoothedVelocityIC(
               Uin, gamma, 0.6, 0.0, 0.25, delta)
           : mfem::SupersonicCylinderSmoothedNormalVelocityIC(
               Uin, gamma, 0.6, 0.0, 0.25, delta);
}

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<SupersonicIGROptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "supersonic_cylinder_w_igr";
  spec.default_mesh_path = mfem::SupersonicCylinderIGRDefaultMeshPath();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.dg_backend_default = dg_sim::DGBackend::DG;
  spec.use_igr = true;
  spec.need_h1 = false;
  spec.diagnostics.h1_diagnostics = false;
  spec.diagnostics.l2_diagnostics = true;
  spec.diagnostics.mpi_rank_field = true;
  spec.diagnostics.zhang_shu_limiter = true;
  spec.glvis.precision = std::numeric_limits<mfem::real_t>::max_digits10;
  spec.vtk.enabled = true;
  spec.vtk.print_dataset_name = true;

  spec.configure_case = [](dg_sim::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 4.0;
    cfg.rk3_config.limit_initial = true;
    cfg.igr_config.alpha_length_scale_policy =
      ::igrlab::IGRAlphaLengthScalePolicy::Max;
    //  cfg.igr_config.smoot = dg_igr::IGRSmootherType::L1GS;
    cfg.limiter_pipeline.zhang_shu = true;
    // cfg.limiter_pipeline.krivodonova = true;
    // cfg.rk3_config.max_limiter_steps = -1;
  };

  spec.add_options = [](mfem::OptionsParser& args, SupersonicIGROptions& opts)
  {
    AddInflowOptions(args, opts.inflow);
    args.AddOption(&opts.startup, "-startup", "--startup",
                   "Cylinder startup velocity damping: normal (default) or full.");
    args.AddOption(&opts.startup_width, "-startup-width", "--startup-width",
                   "Physical radial smoothing width; zero preserves max(8*hmin,0.08).");
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

    const double hmin = dg_sim::ComputeAlphaInfo(
                          ctx.Mesh(),
                          ctx.run_cfg.igr_opts.c_alpha_user,
                          dg_sim::ResolveIGRAlphaLengthScalePolicy(ctx.run_cfg))
                          .hmin;
    auto ic_fun = MakeCylinderStartup(opts, Uin, gamma, hmin);
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

#if defined(DG_IGR_HAS_HDG_MAIN_BACKEND) && DG_IGR_HAS_HDG_MAIN_BACKEND
  spec.make_hdg_case_data = [](const SupersonicIGROptions& opts,
                                const dg_sim::CaseConfig& case_config,
                                const dg_sim::CommonOptions& options,
                                const dg_sim::CaseContext& context)
  {
    dg_sim::hdg::HdgEulerCaseData data;
    data.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(options.order);

    mfem::Vector Uin(4);
    Uin(0) = 1.4;
    Uin(1) = 1.4 * opts.inflow.vx;
    Uin(2) = 1.4 * opts.inflow.vy;
    Uin(3) = 8.8;

    // The legacy case's smoothed startup depends only on the globally reduced
    // hmin; c_alpha and its length-scale policy do not change that quantity.
    const double hmin =
      dg_sim::ComputeAlphaInfo(
        context.Mesh(),
        /*c_alpha_user=*/1.0,
        case_config.igr_config.alpha_length_scale_policy)
        .hmin;
    auto initial_condition = MakeCylinderStartup(
      opts, Uin, case_config.gamma, hmin);
    data.initial_condition =
      std::make_unique<mfem::VectorFunctionCoefficient>(initial_condition);

    const auto markers = dg_sim::BuildBoundaryMarkers(
      context.Mesh(),
      {{dg_sim::BoundaryRole::Inlet, {4}},
       {dg_sim::BoundaryRole::Outlet, {2}},
       {dg_sim::BoundaryRole::Slip, {1, 3}},
       {dg_sim::BoundaryRole::Cylinder, {5}}});
    data.boundary_plan.AddSupersonicInlet(
      markers.At(dg_sim::BoundaryRole::Inlet), [Uin] { return Uin; });
    data.boundary_plan.AddSupersonicOutlet(
      markers.At(dg_sim::BoundaryRole::Outlet));
    data.boundary_plan.AddSlipWall(markers.At(dg_sim::BoundaryRole::Slip));
    data.boundary_plan.AddSlipWall(markers.At(dg_sim::BoundaryRole::Cylinder));
    return data;
  };
#endif

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

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
//               The example demonstrates usage of
//               mfem::DGHyperbolicConservationLaws that wraps
//               NonlinearFormIntegrators containing element and face
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
#include <fstream>
#include <iostream>
#include <sstream>

#include "sim/igr_sim.hpp"

#include "sedov_taylor_blast.hpp"

namespace {

struct SedovOptions {};

} // namespace

int main(int argc, char *argv[]) {
  dg_igr_examples::CaseSpec<SedovOptions, mfem::RusanovFluxWithSides> spec;
  spec.case_name = "sedov_taylor_blast";
  spec.default_mesh_path = mfem::SedovTaylorDefaultMeshPath();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.need_sigma = true;
  spec.need_h1 = true;
  spec.vis.h1_diagnostics = false;
  spec.vis.precision = 8;
  // Callback defaults (explicit to show the available options).
  spec.callback_defaults.enable_alive = true;
  spec.callback_defaults.enable_analysis = true;
  spec.callback_defaults.enable_vtk = true;
  spec.callback_defaults.enable_summary = true;
  spec.callback_defaults.compute_minmax = true;
  // Custom callbacks example (optional):
  // spec.make_callbacks = [&](const auto&, auto&) {
  //   auto cb =
  //       dg_igr_examples::MakeDefaultTimeLoopCallbacks(spec.callback_defaults);
  //   // Example: extend analysis with custom diagnostics.
  //   // cb.on_analysis = [&](const dg_igr_examples::AnalysisInfo& info) { ...
  //   }; return cb;
  // };

  spec.configure_case = [](dg_igr_examples::CaseConfig &cfg) {
    cfg.defaults.t_final = 4.0;
    cfg.rk3_config.limit_initial = false;
    cfg.limiter_pipeline.zhang_shu = true;
    cfg.limiter_pipeline.krivodonova = true;
  };

  spec.make_ic = [](const SedovOptions &,
                    const dg_igr_examples::ExampleContext &ctx, double gamma) {
    const int base_cells = 32;
    const int total_ref =
        ctx.run_cfg.opts.ser_ref_levels + ctx.run_cfg.opts.par_ref_levels;
    (void)total_ref;
    const int mx = base_cells;
    const int my = base_cells;
    auto ic_fun = mfem::GetSedovTaylorBlastInit(gamma, 1.0, mx, my);
    return std::make_unique<mfem::VectorFunctionCoefficient>(4, ic_fun);
  };

  spec.build_markers = [](dg_igr_examples::ExampleContext &ctx) {
    return dg_igr_examples::BuildBoundaryMarkers(
        ctx.Mesh(), {{dg_igr_examples::BoundaryRole::Slip, {1, 2, 3, 4}}});
  };

  spec.apply_bc = [](const SedovOptions &,
                     const dg_igr_examples::ExampleContext &, auto &bc,
                     const auto &markers) { bc.AddSlipWall(markers); };

  spec.postprocess = [](const SedovOptions &,
                        const dg_igr_examples::ExampleContext &,
                        const dg_igr_examples::EulerStateBundle &state,
                        mfem::VectorCoefficient &ic) {
    const mfem::real_t error = state.sol.ComputeLpError(2, ic);
    if (mfem::Mpi::Root()) {
      std::cout << "Solution error: " << error << std::endl;
    }
    return 0;
  };

  return dg_igr_examples::Solver::solve(spec, argc, argv);
}

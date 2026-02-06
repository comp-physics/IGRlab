#include "mfem.hpp"

#include <iostream>
#include <memory>

#include "sim/igr_sim.hpp"

#include "supersonic_cylinder.hpp"

namespace {

struct InflowOptions {
  double vx = 3.0;
  double vy = 0.0;
};

inline void AddInflowOptions(mfem::OptionsParser &args,
                             InflowOptions &inflow_opts) {
  args.AddOption(&inflow_opts.vx, "-vx", "--velocity-x",
                 "Free-stream / inflow velocity x-component.");
  args.AddOption(&inflow_opts.vy, "-vy", "--velocity-y",
                 "Free-stream / inflow velocity y-component.");
}

struct SupersonicOptions {
  InflowOptions inflow;
};

} // namespace

int main(int argc, char *argv[]) {
  dg_igr_examples::CaseSpec<SupersonicOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "dgsem_supersonic_cylinder";
  spec.default_mesh_path = mfem::SupersonicCylinderDefaultMeshPath();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.use_igr = false;
  spec.need_sigma = false;
  spec.need_h1 = false;
  spec.vis.precision = 8;
  spec.dg_backend_default = dg_igr_examples::DGBackend::DGSEM;

  spec.configure_case = [](dg_igr_examples::CaseConfig &cfg) {
    cfg.defaults.t_final = 2.0;
    cfg.defaults.vtk_save = false;
    cfg.defaults.par_ref_levels = 0;
    cfg.defaults.dgsem_volume_flux = "ranocha";
    cfg.defaults.dgsem_surface_flux = "rusanov";
    cfg.rk3_config.limit_initial = false;
    cfg.limiter_pipeline.venkat_mode = mfem::VenkatLimiterMode::kNone;
    cfg.limiter_pipeline.zhang_shu = true;
  };

  spec.add_options = [](mfem::OptionsParser &args, SupersonicOptions &opts) {
    AddInflowOptions(args, opts.inflow);
  };

  spec.make_ic = [](const SupersonicOptions &opts,
                    const dg_igr_examples::ExampleContext &, double) {
    mfem::Vector Uin(4);
    Uin(0) = 1.4;
    Uin(1) = 1.4 * opts.inflow.vx;
    Uin(2) = 1.4 * opts.inflow.vy;
    Uin(3) = 8.8;

    auto ic_fun = mfem::EulerInitialCondition(Uin);
    return std::make_unique<mfem::VectorFunctionCoefficient>(ic_fun);
  };

  spec.build_markers = [](dg_igr_examples::ExampleContext &ctx) {
    return dg_igr_examples::BuildBoundaryMarkers(
        ctx.Mesh(), {{dg_igr_examples::BoundaryRole::Inlet, {4}},
                     {dg_igr_examples::BoundaryRole::Outlet, {2}},
                     {dg_igr_examples::BoundaryRole::Slip, {1, 3}},
                     {dg_igr_examples::BoundaryRole::Cylinder, {5}}});
  };

  spec.apply_bc = [](const SupersonicOptions &opts,
                     const dg_igr_examples::ExampleContext &, auto &bc,
                     const auto &markers) {
    mfem::Vector Uin(4);
    Uin(0) = 1.4;
    Uin(1) = 1.4 * opts.inflow.vx;
    Uin(2) = 1.4 * opts.inflow.vy;
    Uin(3) = 8.8;

    bc.AddSupersonicInletFromBuilder(markers, [&]() { return Uin; });
    bc.AddSupersonicOutlet(markers);
    bc.AddSlipWall(markers);
    bc.AddSlipWall(markers, dg_igr_examples::BoundaryRole::Cylinder);
  };

  spec.postprocess = [](const SupersonicOptions &,
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

#include "mfem.hpp"

#include <iostream>
#include <memory>

#include "sim/igr_sim.hpp"

#include "forward_facing_step.hpp"

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

struct ForwardFacingOptions
{
  InflowOptions inflow;
};

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<ForwardFacingOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "dgsem_forward_facing_step";
  spec.default_mesh_path = mfem::ForwardFacingStepDefaultMeshPath();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.use_igr = false;
  spec.need_sigma = false;
  spec.need_h1 = false;
  spec.diagnostics.h1_diagnostics = false;
  spec.glvis.precision = 8;
  spec.dg_backend_default = dg_sim::DGBackend::DGSEM;

  spec.configure_case = [](dg_sim::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 4.0;
    cfg.defaults.vtk_save = false;
    cfg.defaults.par_ref_levels = 0;
    cfg.defaults.dgsem_volume_flux = "chandrashekar";
    cfg.defaults.dgsem_surface_flux = "es-lf-entropy";
    // cfg.rk3_config.max_limiter_steps = 500;
    cfg.rk3_config.limit_initial = false;
    cfg.limiter_pipeline.zhang_shu = true;
    cfg.limiter_pipeline.krivodonova = false;
  };

  spec.add_options = [](mfem::OptionsParser& args, ForwardFacingOptions& opts)
  {
    AddInflowOptions(args, opts.inflow);
  };

  spec.make_ic =
    [](const ForwardFacingOptions& opts, const dg_sim::ExampleContext&, double)
  {
    mfem::Vector Uin(4);
    Uin(0) = 1.4;
    Uin(1) = 1.4 * opts.inflow.vx;
    Uin(2) = 1.4 * opts.inflow.vy;
    Uin(3) = 8.8;
    auto ic_fun = mfem::EulerInitialCondition(Uin);
    return dg_sim::MakeVectorFunctionIC(ic_fun);
  };

  spec.build_markers = [](dg_sim::ExampleContext& ctx)
  {
    return dg_sim::BuildBoundaryMarkers(
      ctx.Mesh(),
      {{dg_sim::BoundaryRole::Inlet, {1, 2}},
       {dg_sim::BoundaryRole::Outlet, {3}},
       {dg_sim::BoundaryRole::Slip, {4, 5, 6, 7}}});
  };

  spec.apply_bc = [](const ForwardFacingOptions& opts,
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
  };

  spec.postprocess = [](const ForwardFacingOptions&,
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

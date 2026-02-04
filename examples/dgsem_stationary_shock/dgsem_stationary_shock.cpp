#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <memory>

#include "sim/igr_sim.hpp"

#include "stationary_shock.hpp"

namespace
{

using StationaryShockOptions = dg_igr_examples::EmptyCaseOptions;
constexpr double kShockMach = 2.0;
constexpr double kVortexMach = 0.0;
constexpr double kShockSmoothing = 1.0;

} // namespace

int main(int argc, char *argv[])
{
  dg_igr_examples::CaseSpec<StationaryShockOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "dgsem_stationary_shock";
  spec.default_mesh_path = "stationary_shock.msh";
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.use_igr = false;
  spec.need_sigma = false;
  spec.need_h1 = false;
  spec.vis.h1_diagnostics = true;
  spec.vis.precision = 8;
  spec.vtk.print_dataset_name = true;
  spec.dg_backend_default = dg_igr_examples::DGBackend::DGSEM;

  spec.configure_case = [](dg_igr_examples::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 6.0;
    cfg.defaults.vtk_save = false;
    cfg.defaults.par_ref_levels = 0;
    cfg.defaults.dgsem_volume_flux = "ranocha";
    cfg.defaults.dgsem_surface_flux = "es-winters";
    cfg.rk3_config.limit_initial = false;
  };

  spec.make_ic = [](const StationaryShockOptions&,
                    const dg_igr_examples::ExampleContext& ctx,
                    double gamma)
  {
    const mfem::real_t Ms = kShockMach;
    const mfem::real_t Mv = kVortexMach;
    const mfem::real_t shock_smoothing = kShockSmoothing;
    const int base_cells = 32;
    const int total_ref = ctx.run_cfg.opts.ser_ref_levels
                          + ctx.run_cfg.opts.par_ref_levels;
    const int mx = base_cells << total_ref;
    const int my = base_cells << total_ref;
    auto ic_fun = mfem::GetShockVortexInoueHattoriInit(
        gamma, mx, my, Ms, Mv, shock_smoothing);
    return std::make_unique<mfem::VectorFunctionCoefficient>(4, ic_fun);
  };

  spec.build_markers = [](dg_igr_examples::ExampleContext& ctx)
  {
    return dg_igr_examples::BuildBoundaryMarkers(
        ctx.Mesh(),
        {{dg_igr_examples::BoundaryRole::Inlet, {2}},
         {dg_igr_examples::BoundaryRole::Outlet, {4}},
         {dg_igr_examples::BoundaryRole::Slip, {1, 3}}});
  };

  spec.apply_bc = [](const StationaryShockOptions&,
                     const dg_igr_examples::ExampleContext&,
                     auto& bc,
                     const auto& markers)
  {
    const mfem::real_t gamma = 1.4;
    const mfem::real_t Ms = kShockMach;
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

  spec.postprocess = [](const StationaryShockOptions&,
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

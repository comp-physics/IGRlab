#include "igrlab/compat.hpp"
#include "mfem.hpp"

#include <cmath>
#include <iostream>
#include <memory>

#include "sim/igr_sim.hpp"

#include "stationary_shock.hpp"

namespace
{

using StationaryShockOptions = dg_sim::EmptyCaseOptions;
constexpr double kShockMach = 2.0;
constexpr double kVortexMach = 0.0;
constexpr double kShockSmoothing = 1.0;

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<StationaryShockOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "dgsem_stationary_shock";
  spec.default_mesh_path = "stationary_shock.msh";
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.use_igr = true;
  spec.need_h1 = false;
  spec.diagnostics.h1_diagnostics = true;
  spec.diagnostics.l2_diagnostics = true;
  spec.diagnostics.mpi_rank_field = true;
  spec.glvis.precision = 8;
  spec.vtk.print_dataset_name = true;
  spec.dg_backend_default = dg_sim::DGBackend::DGSEM;

  spec.configure_case = [](dg_sim::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 6.0;
    cfg.defaults.vtk_save = true;
    cfg.defaults.par_ref_levels = 0;
    cfg.defaults.dgsem_volume_flux = "central";
    cfg.defaults.dgsem_volume_dissipation = "none";
    cfg.defaults.dgsem_surface_flux = "rusanov";
    cfg.rk3_config.limit_initial = false;

    // IGR linear-solver reference defaults (runtime-overridable via CLI):
    //   -im/--igr-solve-mode
    //   -ils/--igr-linear-solver
    //   -ipc/--igr-preconditioner
    //   -iip/--igr-ip-kind
    //   -irtol/--igr-lin-rtol
    //   -iatol/--igr-lin-atol
    //   -imaxit/--igr-lin-maxit
    //   -iprint/--igr-lin-print
    cfg.igr_config.solve_mode = igrlab::IGRSolveMode::Krylov;
    // Auto policy:
    //   H1 -> CG + BoomerAMG
    //   L2/DG -> GMRES + BoomerAMG
    cfg.igr_config.lin_solver_type = igrlab::IGRLinearSolverType::Auto;
    cfg.igr_config.prec_type = igrlab::IGRPrecondType::Auto;
    cfg.igr_config.ip_kind = igrlab::IGRIPKind::SIPG;
    cfg.igr_config.lin_rtol = 1.0e-6;
    cfg.igr_config.lin_atol = 0.0;
    cfg.igr_config.lin_maxit = 1000;
    cfg.igr_config.lin_print = 0;
  };

  spec.make_ic = [](const StationaryShockOptions&,
                    const dg_sim::ExampleContext& ctx,
                    double gamma)
  {
    const mfem::real_t Ms = kShockMach;
    const mfem::real_t Mv = kVortexMach;
    const mfem::real_t shock_smoothing = kShockSmoothing;
    const int base_cells = 32;
    const int total_ref =
      ctx.run_cfg.opts.ser_ref_levels + ctx.run_cfg.opts.par_ref_levels;
    const int mx = base_cells << total_ref;
    const int my = base_cells << total_ref;
    auto ic_fun = mfem::GetShockVortexInoueHattoriInit(gamma,
                                                       mx,
                                                       my,
                                                       Ms,
                                                       Mv,
                                                       shock_smoothing);
    return dg_sim::MakeVectorFunctionIC(4, ic_fun);
  };

  spec.build_markers = [](dg_sim::ExampleContext& ctx)
  {
    return dg_sim::BuildBoundaryMarkers(ctx.Mesh(),
                                        {{dg_sim::BoundaryRole::Inlet, {2}},
                                         {dg_sim::BoundaryRole::Outlet, {4}},
                                         {dg_sim::BoundaryRole::Slip, {1, 3}}});
  };

  spec.apply_bc = [](const StationaryShockOptions&,
                     const dg_sim::ExampleContext&,
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

#include "mfem.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "sim/igr_sim.hpp"
#include "sim/time/time_defaults.hpp"

#include "shock_vortex_interaction.hpp"

namespace
{

using InoueHattoriStationaryShock1DYOptions = dg_sim::EmptyCaseOptions;

constexpr int kNx = 32;
constexpr int kNy = 1;
constexpr double kGamma = 1.4;
constexpr double kShockMach = 2.0;
constexpr double kVortexMach = 0.0;
constexpr double kShockSmoothing = 1.0;
constexpr double kXLeft = -30.0;
constexpr double kXRight = 10.0;
constexpr double kYBottom = -20.0;
constexpr double kYTop = 20.0;
constexpr double kVortexX = 4.0;
constexpr double kVortexY = 0.0;
constexpr double kShockX = 0.0;

std::string EnsureOneCellYMesh(const char* case_source_file)
{
  namespace fs = std::filesystem;
  const fs::path case_dir = fs::path(case_source_file).parent_path();
  const fs::path mesh_path =
    case_dir / "inoue_hattori_stationary_shock_1dy.mesh";
  if (!fs::exists(mesh_path))
  {
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(kNx,
                                                  kNy,
                                                  mfem::Element::QUADRILATERAL,
                                                  /*gen_edges=*/true,
                                                  /*sx=*/kXRight - kXLeft,
                                                  /*sy=*/kYTop - kYBottom);
    for (int vi = 0; vi < mesh.GetNV(); ++vi)
    {
      double* v = mesh.GetVertex(vi);
      v[0] += kXLeft;
      v[1] += kYBottom;
    }
    std::ofstream out(mesh_path);
    MFEM_VERIFY(out.good(), "Failed to write one-cell-Y mesh at " << mesh_path);
    mesh.Print(out);
  }
  return mesh_path.string();
}

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<InoueHattoriStationaryShock1DYOptions,
                   mfem::DGSEMSurfaceFlux>
    spec;
  spec.case_name = "inoue_hattori_stationary_shock_1dy";
  spec.default_mesh_path = EnsureOneCellYMesh(__FILE__);
  spec.case_file = __FILE__;
  spec.gamma = kGamma;
  spec.dg_backend_default = dg_sim::DGBackend::DG;
  spec.need_h1 = false;
  spec.use_igr = true;
  spec.diagnostics.h1_diagnostics = true;
  spec.diagnostics.l2_diagnostics = true;
  spec.diagnostics.mpi_rank_field = true;
  spec.glvis.precision = 8;
  spec.vtk.enabled = true;
  spec.vtk.print_dataset_name = true;

  spec.configure_case = [](dg_sim::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 6.0;
    cfg.defaults.par_ref_levels = 0;
    cfg.rk3_config.limit_initial = false;

    cfg.igr_config.solve_mode = dg_igr::IGRSolveMode::Krylov;
    cfg.igr_config.lin_solver_type = dg_igr::IGRLinearSolverType::Auto;
    cfg.igr_config.prec_type = dg_igr::IGRPrecondType::Auto;
    cfg.igr_config.ip_kind = dg_igr::IGRIPKind::SIPG;
    cfg.igr_config.lin_rtol = 1.0e-6;
    cfg.igr_config.lin_atol = 0.0;
    cfg.igr_config.lin_maxit = 1000;
    cfg.igr_config.lin_print = 0;

    // Tuned performance preset.
    cfg.igr_config.amg_coarsen_type = 10;
    cfg.igr_config.amg_relax_type = 6;
    cfg.igr_config.amg_max_levels = 10;
    cfg.igr_config.amg_strong_threshold = 0.5;
    cfg.igr_config.sigma_cadence = dg_igr::IGRSigmaUpdateCadence::EveryStep;
    cfg.igr_config.sigma_update_interval = 1;
    cfg.igr_config.stage_rtol_scale = 10.0;
  };

  spec.make_ic = [](const InoueHattoriStationaryShock1DYOptions&,
                    const dg_sim::ExampleContext& ctx,
                    double gamma)
  {
    const int total_ref =
      ctx.run_cfg.opts.ser_ref_levels + ctx.run_cfg.opts.par_ref_levels;
    const int mx = kNx << total_ref;
    const int my = kNy << total_ref;

    auto ic_fun = mfem::GetShockVortexInoueHattoriInitCustom(gamma,
                                                             mx,
                                                             my,
                                                             kShockMach,
                                                             kVortexMach,
                                                             kShockSmoothing,
                                                             kXLeft,
                                                             kXRight,
                                                             kYBottom,
                                                             kYTop,
                                                             kVortexX,
                                                             kVortexY,
                                                             kShockX);
    return dg_sim::MakeVectorFunctionIC(4, ic_fun);
  };

  spec.build_markers = [](dg_sim::ExampleContext& ctx)
  {
    return dg_sim::BuildBoundaryMarkers(ctx.Mesh(),
                                        {{dg_sim::BoundaryRole::Inlet, {2}},
                                         {dg_sim::BoundaryRole::Outlet, {4}},
                                         {dg_sim::BoundaryRole::Slip, {1, 3}}});
  };

  spec.apply_bc = [](const InoueHattoriStationaryShock1DYOptions&,
                     const dg_sim::ExampleContext&,
                     auto& bc,
                     const auto& markers)
  {
    const mfem::real_t rho1 = 1.0;
    const mfem::real_t p1 = 1.0;
    const mfem::real_t c1 = std::sqrt(kGamma * p1 / rho1);
    const mfem::real_t u1x = -kShockMach * c1;
    const mfem::real_t u1y = 0.0;
    const mfem::real_t v1sq = u1x * u1x + u1y * u1y;

    mfem::Vector Uin(4);
    Uin(0) = rho1;
    Uin(1) = rho1 * u1x;
    Uin(2) = rho1 * u1y;
    Uin(3) = p1 / (kGamma - 1.0) + 0.5 * rho1 * v1sq;

    const mfem::real_t m1sq = kShockMach * kShockMach;
    const mfem::real_t rho2 =
      rho1 * (kGamma + 1.0) / (kGamma - 1.0 + 2.0 / m1sq);
    const mfem::real_t u2x = u1x * (rho1 / rho2);
    const mfem::real_t u2y = 0.0;
    const mfem::real_t p2 =
      p1 * (1.0 + 2.0 * kGamma / (kGamma + 1.0) * (m1sq - 1.0));
    const mfem::real_t v2sq = u2x * u2x + u2y * u2y;

    mfem::Vector Uout_ref(4);
    Uout_ref(0) = rho2;
    Uout_ref(1) = rho2 * u2x;
    Uout_ref(2) = rho2 * u2y;
    Uout_ref(3) = p2 / (kGamma - 1.0) + 0.5 * rho2 * v2sq;

    bc.AddSupersonicInletFromBuilder(markers, [&]() { return Uin; });
    bc.AddNonReflectingOutlet(markers, Uout_ref);
    bc.AddSlipWall(markers);
  };

  spec.postprocess = [](const InoueHattoriStationaryShock1DYOptions&,
                        const dg_sim::ExampleContext& ctx,
                        const dg_sim::EulerStateBundle& state,
                        mfem::VectorCoefficient& ic)
  {
    const mfem::real_t error = state.sol.ComputeLpError(2, ic);
    if (mfem::Mpi::Root())
    {
      std::cout << "Solution error: " << error << std::endl;
    }
    if (state.field_list.igr)
    {
      const auto sigma_minmax =
        dg_sim::ComputeGlobalMinMax(state.field_list.igr->GetSigma(),
                                    ctx.Mesh().GetComm());
      if (mfem::Mpi::Root())
      {
        std::cout << "Sigma min/max: " << sigma_minmax.first << " / "
                  << sigma_minmax.second << std::endl;
      }
    }
    return 0;
  };

  return dg_sim::Solver::solve(spec, argc, argv);
}

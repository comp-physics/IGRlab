#include "mfem.hpp"

#include <cmath>

#include "sim/igr_sim.hpp"

#include "../inoue_hattori/inoue_hattori_mesh_utils.hpp"
#include "shock_vortex_interaction.hpp"

namespace
{

using InoueHattoriLargeDomainSlipTBOptions = dg_sim::EmptyCaseOptions;

constexpr int kNx = 32;
constexpr int kNy = 32;
constexpr double kGamma = 1.4;
constexpr double kShockMach = 2.0;
constexpr double kVortexMach = 0.25;
constexpr double kShockSmoothing = 4.0;
constexpr double kXLeft = -30.0;
constexpr double kXRight = 10.0;
constexpr double kYBottom = -20.0;
constexpr double kYTop = 20.0;
constexpr double kVortexX = 5.0;
constexpr double kVortexY = 0.0;
constexpr double kShockX = -10.0;
constexpr dg_sim::inoue_hattori::PeriodicYMeshConfig kMeshCfg = {
  kNx,
  kNy,
  kXLeft,
  kXRight,
  kYBottom,
  kYTop,
  "inoue_hattori_large_domain_slip_tb.mesh",
  "Inoue-Hattori large-domain slip-top-bottom"};

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<InoueHattoriLargeDomainSlipTBOptions, mfem::DGSEMSurfaceFlux>
    spec;
  spec.case_name = "inoue_hattori_large_domain_slip_tb";
  spec.default_mesh_path =
    dg_sim::inoue_hattori::EnsureNonPeriodicMeshFile(__FILE__, kMeshCfg);
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
  };

  spec.make_ic = [](const InoueHattoriLargeDomainSlipTBOptions&,
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
    return dg_sim::inoue_hattori::BuildInletOutletSlipMarkers(ctx.Mesh(), kMeshCfg);
  };

  spec.apply_bc = [](const InoueHattoriLargeDomainSlipTBOptions&,
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

  return dg_sim::Solver::solve(spec, argc, argv);
}

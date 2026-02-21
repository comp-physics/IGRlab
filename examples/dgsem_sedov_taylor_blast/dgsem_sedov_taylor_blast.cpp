#include "mfem.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include "sim/igr_sim.hpp"

#include "sedov_taylor_blast.hpp"

namespace
{

struct SedovOptions
{
};

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<SedovOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "dgsem_sedov_taylor_blast";
  spec.default_mesh_path = mfem::SedovTaylorDefaultMeshPath();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.use_igr = true;
  spec.need_h1 = true;
  spec.diagnostics.h1_diagnostics = false;
  spec.glvis.precision = 8;
  spec.dg_backend_default = dg_sim::DGBackend::DGSEM;
  spec.vtk.enabled = false;

  spec.configure_case = [](dg_sim::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 4.0;
    cfg.defaults.par_ref_levels = 0;
    cfg.defaults.dgsem_volume_flux = "ranocha";
    cfg.defaults.dgsem_surface_flux = "es-winters";
    cfg.rk3_config.limit_initial = false;
    cfg.limiter_pipeline.zhang_shu = true;
    cfg.limiter_pipeline.krivodonova = false;
  };

  spec.make_ic =
    [](const SedovOptions&, const dg_sim::ExampleContext& ctx, double gamma)
  {
    const int base_cells = 32;
    const int total_ref =
      ctx.run_cfg.opts.ser_ref_levels + ctx.run_cfg.opts.par_ref_levels;
    (void)total_ref;
    const int mx = base_cells;
    const int my = base_cells;
    auto ic_fun = mfem::GetSedovTaylorBlastInit(gamma, 1.0, mx, my);
    return dg_sim::MakeVectorFunctionIC(4, ic_fun);
  };

  spec.build_markers = [](dg_sim::ExampleContext& ctx)
  {
    return dg_sim::BuildBoundaryMarkers(
      ctx.Mesh(),
      {{dg_sim::BoundaryRole::Slip, {1, 2, 3, 4}}});
  };

  spec.apply_bc = [](const SedovOptions&,
                     const dg_sim::ExampleContext&,
                     auto& bc,
                     const auto& markers)
  {
    bc.AddSlipWall(markers);
  };

  spec.postprocess = [](const SedovOptions&,
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

#include "mfem.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "physics/euler/conversions.hpp"
#include "sim/igr_sim.hpp"

namespace
{

struct SmoothAdvectionOptions
{};

std::string EnsurePeriodicMeshFile()
{
  namespace fs = std::filesystem;
  const fs::path case_dir = fs::path(__FILE__).parent_path();
  const fs::path mesh_path = case_dir / "smooth_advection_periodic.msh";
  if (!fs::exists(mesh_path))
  {
    constexpr int nx = 32;
    constexpr int ny = 32;
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
        nx, ny, mfem::Element::QUADRILATERAL,
        /*gen_edges=*/true, /*sx=*/1.0, /*sy=*/1.0);
    std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                              mfem::Vector({0.0, 1.0})};
    mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
        mesh, mesh.CreatePeriodicVertexMapping(translations));

    std::ofstream out(mesh_path);
    MFEM_VERIFY(out.good(),
                "Failed to create periodic mesh file at " << mesh_path);
    periodic.Print(out);
  }
  return mesh_path.filename().string();
}

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<SmoothAdvectionOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "dgsem_smooth_advection_periodic";
  spec.default_mesh_path = EnsurePeriodicMeshFile();
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.use_igr = false;
  spec.need_sigma = false;
  spec.need_h1 = false;
  spec.dg_backend_default = dg_sim::DGBackend::DGSEM;
  spec.vis.precision = 8;
  spec.vtk.print_dataset_name = true;

  spec.configure_case = [](dg_sim::CaseConfig& cfg)
  {
    cfg.defaults.t_final = 0.1;
    cfg.defaults.vtk_save = false;
    cfg.defaults.par_ref_levels = 0;
    cfg.defaults.dgsem_volume_flux = "ranocha";
    cfg.defaults.dgsem_surface_flux = "es-winters";
    cfg.rk3_config.limit_initial = false;
    cfg.limiter_pipeline.venkat_mode = mfem::VenkatLimiterMode::kNone;
    cfg.limiter_pipeline.zhang_shu = false;
  };

  spec.make_ic = [](const SmoothAdvectionOptions&,
                    const dg_sim::ExampleContext&,
                    double gamma)
  {
    const double rho0 = 1.0;
    const double u0 = 2.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double amp = 0.1;
    const double sigma = 0.12;

    auto ic_fun = [=](const mfem::Vector& x, mfem::Vector& U)
    {
      const double dx = x(0) - 0.5;
      const double dy = x(1) - 0.5;
      const double r2 = dx * dx + dy * dy;
      const double rho = rho0 + amp * std::exp(-r2 / (sigma * sigma));

      mfem::Vector prim(4);
      prim(0) = rho;
      prim(1) = u0;
      prim(2) = v0;
      prim(3) = p0;
      mfem::EulerPrimToConservative(prim, /*dim=*/2, gamma, U);
    };

    return dg_sim::MakeVectorFunctionIC(4, ic_fun);
  };

  spec.build_markers = [](dg_sim::ExampleContext&)
  {
    return dg_sim::BoundaryMarkerMap{};
  };

  spec.apply_bc = [](const SmoothAdvectionOptions&,
                     const dg_sim::ExampleContext&,
                     auto&,
                     const auto&)
  {
  };

  return dg_sim::Solver::solve(spec, argc, argv);
}

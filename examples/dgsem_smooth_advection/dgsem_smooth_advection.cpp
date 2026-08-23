#include "mfem.hpp"

#include <cmath>
#include <initializer_list>
#include <memory>
#include <string>

#include "physics/euler/conversions.hpp"
#include "sim/igr_sim.hpp"

namespace
{

struct SmoothAdvectionOptions
{
  std::string bc_mode = "inlet-outlet";
};

std::unique_ptr<mfem::VectorCoefficient>
MakeSmoothAdvectionInitialCondition(const double gamma)
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
}

mfem::Vector MakeInletState(const double gamma)
{
  mfem::Vector prim(4);
  prim(0) = 1.0;
  prim(1) = 2.0;
  prim(2) = 0.0;
  prim(3) = 1.0;

  mfem::Vector inlet_state(4);
  mfem::EulerPrimToConservative(prim, /*dim=*/2, gamma, inlet_state);
  return inlet_state;
}

mfem::Array<int>
MarkerForAttributes(const mfem::ParMesh& mesh,
                    const std::initializer_list<int> attributes)
{
  const int maximum_attribute =
    mesh.bdr_attributes.Size() == 0 ? 0 : mesh.bdr_attributes.Max();
  mfem::Array<int> marker(maximum_attribute);
  marker = 0;
  for (const int attribute : attributes)
  {
    MFEM_VERIFY(attribute > 0 && attribute <= maximum_attribute,
                "Smooth-advection mesh is missing boundary attribute "
                  << attribute << '.');
    marker[attribute - 1] = 1;
  }
  return marker;
}

dg_sim::BoundaryMarkerMap
BuildSmoothAdvectionMarkers(const mfem::ParMesh& mesh)
{
  dg_sim::BoundaryMarkerMap markers;
  markers.Add(dg_sim::BoundaryRole::Inlet, MarkerForAttributes(mesh, {4}));
  markers.Add(dg_sim::BoundaryRole::Outlet, MarkerForAttributes(mesh, {2}));
  markers.Add(dg_sim::BoundaryRole::Slip, MarkerForAttributes(mesh, {1, 3}));
  return markers;
}

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<SmoothAdvectionOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "dgsem_smooth_advection";
  spec.default_mesh_path = "smooth_advection.msh";
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.use_igr = false;
  spec.need_h1 = false;
  spec.dg_backend_default = dg_sim::DGBackend::DGSEM;
  spec.glvis.precision = 8;
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

  spec.add_options = [](mfem::OptionsParser& args, SmoothAdvectionOptions& opts)
  {
    args.AddOption(&opts.bc_mode,
                   "-bm",
                   "--bc-mode",
                   "Boundary mode: inlet-outlet | all-outlet | all-slip.");
  };

  spec.make_ic = [](const SmoothAdvectionOptions&,
                    const dg_sim::ExampleContext&,
                    double gamma)
  {
    return MakeSmoothAdvectionInitialCondition(gamma);
  };

  spec.build_markers = [](dg_sim::ExampleContext& ctx)
  {
    return BuildSmoothAdvectionMarkers(ctx.Mesh());
  };

  spec.apply_bc = [](const SmoothAdvectionOptions& opts,
                     const dg_sim::ExampleContext&,
                     auto& bc,
                     const auto& markers)
  {
    const mfem::Vector Uin = MakeInletState(1.4);

    const std::string mode = dg_sim::ToLowerCopy(opts.bc_mode);
    if (mode == "inlet-outlet")
    {
      bc.AddSupersonicInletFromBuilder(markers, [&]() { return Uin; });
      bc.AddSupersonicOutlet(markers);
      bc.AddSlipWall(markers);
      return;
    }
    if (mode == "all-outlet")
    {
      bc.AddSupersonicOutlet(markers, dg_sim::BoundaryRole::Inlet);
      bc.AddSupersonicOutlet(markers, dg_sim::BoundaryRole::Outlet);
      bc.AddSupersonicOutlet(markers, dg_sim::BoundaryRole::Slip);
      return;
    }
    if (mode == "all-slip")
    {
      bc.AddSlipWall(markers, dg_sim::BoundaryRole::Inlet);
      bc.AddSlipWall(markers, dg_sim::BoundaryRole::Outlet);
      bc.AddSlipWall(markers, dg_sim::BoundaryRole::Slip);
      return;
    }
    MFEM_VERIFY(false,
                "Unknown bc-mode '"
                  << opts.bc_mode
                  << "'. Use inlet-outlet | all-outlet | all-slip.");
  };

#if DG_IGR_HAS_HDG_MAIN_BACKEND
  spec.make_hdg_case_data = [](const SmoothAdvectionOptions& opts,
                                const dg_sim::CaseConfig& case_config,
                                const dg_sim::CommonOptions& options,
                                const dg_sim::CaseContext& context)
  {
    dg_sim::hdg::HdgEulerCaseData case_data;
    case_data.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(options.order);
    case_data.initial_condition =
      MakeSmoothAdvectionInitialCondition(case_config.gamma);
    const auto markers = BuildSmoothAdvectionMarkers(context.Mesh());
    const std::string mode = dg_sim::ToLowerCopy(opts.bc_mode);
    if (mode == "inlet-outlet")
    {
      const mfem::Vector inlet_state = MakeInletState(case_config.gamma);
      case_data.boundary_plan.AddSupersonicInlet(
        markers.At(dg_sim::BoundaryRole::Inlet),
        [inlet_state] { return inlet_state; });
      case_data.boundary_plan.AddSupersonicOutlet(
        markers.At(dg_sim::BoundaryRole::Outlet));
      case_data.boundary_plan.AddSlipWall(markers.At(dg_sim::BoundaryRole::Slip));
      return case_data;
    }
    if (mode == "all-outlet")
    {
      case_data.boundary_plan.AddSupersonicOutlet(
        markers.At(dg_sim::BoundaryRole::Inlet));
      case_data.boundary_plan.AddSupersonicOutlet(
        markers.At(dg_sim::BoundaryRole::Outlet));
      case_data.boundary_plan.AddSupersonicOutlet(
        markers.At(dg_sim::BoundaryRole::Slip));
      return case_data;
    }
    if (mode == "all-slip")
    {
      case_data.boundary_plan.AddSlipWall(markers.At(dg_sim::BoundaryRole::Inlet));
      case_data.boundary_plan.AddSlipWall(markers.At(dg_sim::BoundaryRole::Outlet));
      case_data.boundary_plan.AddSlipWall(markers.At(dg_sim::BoundaryRole::Slip));
      return case_data;
    }
    MFEM_VERIFY(false,
                "Unknown bc-mode '"
                  << opts.bc_mode
                  << "'. Use inlet-outlet | all-outlet | all-slip.");
    return case_data;
  };
#endif

  return dg_sim::Solver::solve(spec, argc, argv);
}

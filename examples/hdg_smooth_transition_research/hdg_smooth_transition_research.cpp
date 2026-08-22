#include "mfem.hpp"

#include <cmath>
#include <memory>
#include <utility>

#include "physics/euler/conversions.hpp"
#include "sim/igr_sim.hpp"

namespace
{

struct SmoothTransitionResearchOptions
{
};

std::unique_ptr<mfem::VectorCoefficient>
MakeSmoothTransitionInitialCondition(const double gamma)
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double transition_width = 1.0 / 8.0;
  const double kappa = std::atanh(0.8) / std::sin(pi * transition_width);

  auto initial_condition = [gamma, kappa](const mfem::Vector& x,
                                          mfem::Vector& state)
  {
    const double chi = 0.5 *
      (1.0 - std::tanh(kappa * std::cos(2.0 * pi * x(0))));
    mfem::Vector primitive(4);
    primitive(0) = 0.125 + 0.875 * chi;
    primitive(1) = 0.0;
    primitive(2) = 0.0;
    primitive(3) = 0.1 + 0.9 * chi;
    mfem::EulerPrimToConservative(primitive, /*dim=*/2, gamma, state);
  };
  return dg_sim::MakeVectorFunctionIC(4, initial_condition);
}

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<SmoothTransitionResearchOptions, mfem::DGSEMSurfaceFlux>
    spec;
  spec.case_name = "hdg_smooth_transition_research";
  // This committed mesh is the deterministic periodic N8 research input.
  // Its four seam attributes are intentionally all lowered as periodic.
  spec.default_mesh_path = "hdg_smooth_transition_periodic_n8.mesh";
  spec.case_file = __FILE__;
  spec.gamma = 1.4;
  spec.use_igr = false;
  spec.need_h1 = false;
  spec.dg_backend_default = dg_sim::DGBackend::DG;
  spec.glvis.enabled = false;
  spec.vtk.enabled = false;

  spec.configure_case = [](dg_sim::CaseConfig& cfg)
  {
    // Keep default execution intentionally small.  HDG resolution selects
    // backward Euler; process coverage supplies its own two-step smoke tuple.
    cfg.defaults.t_final = 2.0e-4;
    cfg.defaults.dt = 1.0e-4;
    cfg.defaults.vtk_save = false;
    cfg.defaults.glvis = false;
    cfg.defaults.par_ref_levels = 0;
  };

  // The ordinary path deliberately shares the same analytic case definition,
  // so this executable can also serve as a DG/DGSEM comparison point.  The
  // periodic topology is carried by the mesh and needs no physical DG BCs.
  spec.make_ic = [](const SmoothTransitionResearchOptions&,
                    const dg_sim::ExampleContext&,
                    const double gamma)
  {
    return MakeSmoothTransitionInitialCondition(gamma);
  };
  spec.build_markers = [](dg_sim::ExampleContext&)
  {
    return dg_sim::BoundaryMarkerMap{};
  };
  spec.apply_bc = [](const SmoothTransitionResearchOptions&,
                     const dg_sim::ExampleContext&,
                     auto&,
                     const auto&) {
  };

  spec.make_hdg_case_data = [](const SmoothTransitionResearchOptions&,
                                const dg_sim::CaseConfig& case_config,
                                const dg_sim::CommonOptions& options,
                                const dg_sim::CaseContext& context)
  {
    dg_sim::hdg::HdgEulerCaseData data;
    data.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(options.order);
    // These are declared numerical roles, not a legacy-solver selector.
    data.numerical_policy.volume_projection_quadrature_order = 14;
    data.numerical_policy.true_trace_projection_quadrature_order = 14;
    data.numerical_policy.conventional_assembly_quadrature_order = 11;
    data.numerical_policy.igr_assembly_quadrature_order = 12;
    data.numerical_policy.audit_quadrature_order = 14;
    data.numerical_policy.observable_quadrature_order = 18;
    data.numerical_policy.maximum_nonlinear_updates = 20;
    data.numerical_policy.maximum_linear_iterations = 200;
    data.initial_condition =
      MakeSmoothTransitionInitialCondition(case_config.gamma);

    MFEM_VERIFY(context.Mesh().bdr_attributes.Size() > 0,
                "W8C research mesh must provide periodic seam attributes.");
    mfem::Array<int> periodic_marker(context.Mesh().bdr_attributes.Max());
    periodic_marker = 1;
    data.boundary_plan.AddPeriodic(std::move(periodic_marker));

    // static_source remains null: the generic main adapter treats it as zero.
    // This case shares only analytic initial data with the frozen W8C family.
    // It intentionally does not reproduce checkpoints or any
    // scientific-parity schedule, and never
    // imports or calls the legacy SC25/application solver implementation.
    return data;
  };

  return dg_sim::Solver::solve(spec, argc, argv);
}

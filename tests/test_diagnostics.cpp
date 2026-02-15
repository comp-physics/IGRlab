#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/io/vis_helpers.hpp"
#include "sim/state/euler_state_bundle.hpp"

#include <limits>

#include "test_utils.hpp"

namespace dg_igr_tests
{

namespace
{

dg_sim::RunConfig MakeRunConfig(int order)
{
  dg_sim::RunConfig cfg;
  cfg.case_config.gamma = 1.4;
  cfg.opts.order = order;
  cfg.opts.glvis = false;
  cfg.opts.vtk_save = false;
  cfg.opts.ser_ref_levels = 0;
  cfg.opts.par_ref_levels = 0;
  cfg.dg_backend = dg_sim::DGBackend::DG;
  cfg.mesh_path.clear();
  return cfg;
}

int FindLeftElement(const dg_sim::ExampleContext& ctx)
{
  mfem::IntegrationPoint ip_center;
  ip_center.Set2(0.5, 0.5);
  int best_e = -1;
  double best_x = std::numeric_limits<double>::infinity();
  for (int e = 0; e < ctx.Mesh().GetNE(); ++e)
  {
    mfem::ElementTransformation* Tr = ctx.spaces.fes.GetElementTransformation(e);
    mfem::Vector x;
    Tr->Transform(ip_center, x);
    if (x(0) < best_x)
    {
      best_x = x(0);
      best_e = e;
    }
  }
  return best_e;
}

int FindRightElement(const dg_sim::ExampleContext& ctx)
{
  mfem::IntegrationPoint ip_center;
  ip_center.Set2(0.5, 0.5);
  int best_e = -1;
  double best_x = -std::numeric_limits<double>::infinity();
  for (int e = 0; e < ctx.Mesh().GetNE(); ++e)
  {
    mfem::ElementTransformation* Tr = ctx.spaces.fes.GetElementTransformation(e);
    mfem::Vector x;
    Tr->Transform(ip_center, x);
    if (x(0) > best_x)
    {
      best_x = x(0);
      best_e = e;
    }
  }
  return best_e;
}

} // namespace

int RunDiagnosticsTests(TestContext& ctx)
{
  constexpr double gamma = 1.4;
  constexpr double rho_floor = 1.0e-12;

  // H1 diagnostics scalar outputs must live on scalar spaces (vdim=1).
  {
    mfem::Mesh serial_mesh =
      mfem::Mesh::MakeCartesian2D(1,
                                  1,
                                  mfem::Element::QUADRILATERAL,
                                  /*gen_edges=*/true,
                                  /*sx=*/1.0,
                                  /*sy=*/1.0);
    auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    dg_sim::RunConfig run_cfg = MakeRunConfig(/*order=*/2);
    dg_sim::ExampleContext example_ctx(run_cfg,
                                       std::move(pmesh),
                                       /*sigma_space=*/"H1",
                                       /*need_sigma=*/false,
                                       /*need_h1=*/true);
    dg_sim::EulerStateBundle bundle(example_ctx, gamma, rho_floor);
    FillConstantState(
      EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.25, /*v=*/-0.05, /*p=*/1.0, gamma),
      bundle.sol);

    dg_sim::H1Diagnostics h1_diag(*example_ctx.spaces.h1_vfes,
                                  bundle.sol,
                                  example_ctx.dim,
                                  rho_floor);
    h1_diag.Update();

    ctx.Check(h1_diag.u_h1.ParFESpace()->GetVDim() == example_ctx.dim,
              "H1 diagnostics velocity field uses vector space.");
    ctx.Check(h1_diag.g_div.ParFESpace()->GetVDim() == 1,
              "H1 diagnostics div field uses scalar space.");
    ctx.Check(h1_diag.g_gfrob.ParFESpace()->GetVDim() == 1,
              "H1 diagnostics grad-frobenius field uses scalar space.");
    ctx.Check(h1_diag.g_umag.ParFESpace()->GetVDim() == 1,
              "H1 diagnostics speed field uses scalar space.");
  }

  // DG-to-H1 projection must use arithmetic averaging at shared H1 dofs.
  {
    mfem::Mesh serial_mesh =
      mfem::Mesh::MakeCartesian2D(2,
                                  1,
                                  mfem::Element::QUADRILATERAL,
                                  /*gen_edges=*/true,
                                  /*sx=*/1.0,
                                  /*sy=*/1.0);
    auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    dg_sim::RunConfig run_cfg = MakeRunConfig(/*order=*/0);
    dg_sim::ExampleContext example_ctx(run_cfg,
                                       std::move(pmesh),
                                       /*sigma_space=*/"H1",
                                       /*need_sigma=*/false,
                                       /*need_h1=*/true);
    dg_sim::EulerStateBundle bundle(example_ctx, gamma, rho_floor);

    mfem::VectorFunctionCoefficient ic(
      4,
      [](const mfem::Vector& x, mfem::Vector& y)
      {
        y.SetSize(4);
        const double rho = 1.0;
        const double ux = (x(0) < 0.5) ? 0.0 : 2.0;
        const double p = 1.0;
        y(0) = rho;
        y(1) = rho * ux;
        y(2) = 0.0;
        y(3) = p / 0.4 + 0.5 * rho * ux * ux;
      });
    bundle.sol.ProjectCoefficient(ic);

    dg_sim::H1Diagnostics h1_diag(*example_ctx.spaces.h1_vfes,
                                  bundle.sol,
                                  example_ctx.dim,
                                  rho_floor);
    h1_diag.Update();

    const int left_e = FindLeftElement(example_ctx);
    const int right_e = FindRightElement(example_ctx);
    ctx.Check(left_e >= 0 && right_e >= 0 && left_e != right_e,
              "Diagnostics test mesh has distinct left/right elements.");

    mfem::Vector x_left(2);
    x_left(0) = 0.5 - 1.0e-10;
    x_left(1) = 0.5;
    mfem::Vector x_right(2);
    x_right(0) = 0.5 + 1.0e-10;
    x_right(1) = 0.5;
    mfem::IntegrationPoint ip_left;
    mfem::IntegrationPoint ip_right;
    {
      mfem::ElementTransformation* Tr_left =
        example_ctx.spaces.h1_vfes->GetElementTransformation(left_e);
      Tr_left->TransformBack(x_left, ip_left);
    }
    {
      mfem::ElementTransformation* Tr_right =
        example_ctx.spaces.h1_vfes->GetElementTransformation(right_e);
      Tr_right->TransformBack(x_right, ip_right);
    }

    mfem::Vector ul, ur;
    {
      mfem::ElementTransformation* Tr_left =
        example_ctx.spaces.h1_vfes->GetElementTransformation(left_e);
      h1_diag.u_h1.GetVectorValue(*Tr_left, ip_left, ul);
    }
    {
      mfem::ElementTransformation* Tr_right =
        example_ctx.spaces.h1_vfes->GetElementTransformation(right_e);
      h1_diag.u_h1.GetVectorValue(*Tr_right, ip_right, ur);
    }

    // With arithmetic averaging, shared interface velocity is (0 + 2)/2 = 1.
    ctx.CheckNear(ul(0),
                  1.0,
                  1.0e-8,
                  "H1 velocity projection averages discontinuous DG states at interface (left).");
    ctx.CheckNear(ur(0),
                  1.0,
                  1.0e-8,
                  "H1 velocity projection averages discontinuous DG states at interface (right).");
  }

  // L2 diagnostics remain broken/local (no H1-style shared-dof averaging).
  {
    mfem::Mesh serial_mesh =
      mfem::Mesh::MakeCartesian2D(2,
                                  1,
                                  mfem::Element::QUADRILATERAL,
                                  /*gen_edges=*/true,
                                  /*sx=*/1.0,
                                  /*sy=*/1.0);
    auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
    serial_mesh.Clear();

    dg_sim::RunConfig run_cfg = MakeRunConfig(/*order=*/1);
    dg_sim::ExampleContext example_ctx(run_cfg,
                                       std::move(pmesh),
                                       /*sigma_space=*/"L2",
                                       /*need_sigma=*/false,
                                       /*need_h1=*/false);
    dg_sim::EulerStateBundle bundle(example_ctx, gamma, rho_floor);

    mfem::VectorFunctionCoefficient ic(
      4,
      [](const mfem::Vector& x, mfem::Vector& y)
      {
        y.SetSize(4);
        const double rho = 1.0;
        const double ux = (x(0) < 0.5) ? x(0) : (2.0 * x(0) - 0.5);
        const double p = 1.0;
        y(0) = rho;
        y(1) = rho * ux;
        y(2) = 0.0;
        y(3) = p / 0.4 + 0.5 * rho * ux * ux;
      });
    bundle.sol.ProjectCoefficient(ic);

    dg_sim::L2Diagnostics l2_diag(example_ctx.spaces.fes,
                                  bundle.sol,
                                  example_ctx.dim,
                                  rho_floor);
    l2_diag.Update();

    ctx.Check(l2_diag.g_div_l2.ParFESpace()->GetVDim() == 1,
              "L2 diagnostics div field uses scalar space.");
    ctx.Check(l2_diag.g_gfrob_l2.ParFESpace()->GetVDim() == 1,
              "L2 diagnostics grad-frobenius field uses scalar space.");
    ctx.Check(dynamic_cast<const mfem::DG_FECollection*>(
                l2_diag.g_div_l2.ParFESpace()->FEColl()) != nullptr,
              "L2 diagnostics use DG/L2 finite element collection.");

    const int left_e = FindLeftElement(example_ctx);
    const int right_e = FindRightElement(example_ctx);
    ctx.Check(left_e >= 0 && right_e >= 0 && left_e != right_e,
              "L2 diagnostics test mesh has distinct left/right elements.");

    mfem::IntegrationPoint ip_left;
    mfem::IntegrationPoint ip_right;
    ip_left.Set2(0.5, 0.5);
    ip_right.Set2(0.5, 0.5);

    const double div_left =
      [&]()
      {
        mfem::ElementTransformation* Tr_left =
          example_ctx.spaces.fes.GetElementTransformation(left_e);
        return l2_diag.g_div_l2.GetValue(*Tr_left, ip_left);
      }();
    const double div_right =
      [&]()
      {
        mfem::ElementTransformation* Tr_right =
          example_ctx.spaces.fes.GetElementTransformation(right_e);
        return l2_diag.g_div_l2.GetValue(*Tr_right, ip_right);
      }();

    // For u_x = x (left) and u_x = 2x (right), broken gradients stay local:
    // div_left ~= 1, div_right ~= 2 in the two elements.
    ctx.CheckNear(div_left,
                  1.0,
                  1.0e-6,
                  "L2 diagnostics keep local left-element divergence.");
    ctx.CheckNear(div_right,
                  2.0,
                  1.0e-6,
                  "L2 diagnostics keep local right-element divergence.");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests

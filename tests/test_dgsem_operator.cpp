#include "numerics/dgsem/dgsem_operator.hpp"

#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "numerics/timestepping/RK3SSPWithLimiter.hpp"
#include "physics/bc/boundary_state.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace dg_igr_tests
{

int RunDGSEMOperatorTests(TestContext& ctx)
{
  const double tol = 1.0e-10;

  // 1D constant-state cancellation.
  constexpr int dim = 1;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(2, 1.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

  mfem::DG_FECollection fec(/*order=*/2, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_eq,
                                   mfem::Ordering::byNODES);

  mfem::ParGridFunction U(&vfes);
  const mfem::Vector cons_const =
      EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/0.0, /*p=*/1.0, gamma);
  FillConstantState(cons_const, U);

  mfem::EulerFlux flux(dim, gamma);
  mfem::RusanovFluxWithSides num_flux(flux);

  constexpr int int_order_offset = 2;
  auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                       int_order_offset);
  mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form_integrator),
                                          /*preassembleWeakDivergence=*/true);
  oper.SetNumericalFlux(&num_flux);
  oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

  mfem::Array<int> bdr_marker;
  const int max_attr = pmesh.bdr_attributes.Max();
  if (max_attr > 0)
  {
    bdr_marker.SetSize(max_attr);
    bdr_marker = 1;
    auto outlet = std::make_unique<mfem::SupersonicOutletState>();
    oper.AddBdrFaceIntegrator(
        new mfem::BdrHyperbolicFluxIntegrator(num_flux, std::move(outlet),
                                              int_order_offset),
        bdr_marker);
  }

  mfem::Vector x;
  U.GetTrueDofs(x);

  mfem::dgsem::DGSEMOperator dgsem_op(oper);

  mfem::Vector z_surf(vfes.GetTrueVSize());
  mfem::Vector z_vol(vfes.GetTrueVSize());

  dgsem_op.AssembleSurface(x, z_surf);
  z_vol = 0.0;
  dgsem_op.AssembleVolume(x, z_vol);

  mfem::Vector z_sum(z_surf.Size());
  z_sum = z_surf;
  z_sum += z_vol;

  const double norm_surf = GlobalL2Norm(z_surf);
  const double norm_vol = GlobalL2Norm(z_vol);
  const double norm_sum = GlobalL2Norm(z_sum);

  ctx.Check(norm_surf > 0.0, "DGSEM surface contribution is nonzero");
  ctx.Check(norm_vol > 0.0, "DGSEM volume contribution is nonzero");
  ctx.Check(norm_sum < tol,
            "DGSEM surface + volume cancel for constant state (norm=" +
                std::to_string(norm_sum) + ")");

  mfem::Vector y(vfes.GetTrueVSize());
  dgsem_op.ApplyInverseMass(z_sum, y);
  const double resid_norm = GlobalL2Norm(y);
  ctx.Check(resid_norm < tol,
            "DGSEM operator residual ~0 after M^{-1} (norm=" +
                std::to_string(resid_norm) + ")");

  ctx.AddParity(norm_surf + norm_vol + norm_sum + resid_norm);

  // 2D constant-state cancellation on a single quad.
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;

    mfem::Mesh mesh2 = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                                   /*gen_edges=*/true, 1.0, 1.0);
    mfem::ParMesh pmesh2(MPI_COMM_WORLD, mesh2);

    mfem::DG_FECollection fec2(/*order=*/2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh2, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    mfem::ParGridFunction U2(&vfes2);
    const mfem::Vector cons_const2 =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const2, U2);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);

    constexpr int int_order_offset2 = 2;
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    oper2.SetNumericalFlux(&num_flux2);
    oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

    mfem::Array<int> bdr_marker2;
    const int max_attr2 = pmesh2.bdr_attributes.Max();
    if (max_attr2 > 0)
    {
      bdr_marker2.SetSize(max_attr2);
      bdr_marker2 = 1;
      auto outlet2 = std::make_unique<mfem::SupersonicOutletState>();
      oper2.AddBdrFaceIntegrator(
          new mfem::BdrHyperbolicFluxIntegrator(num_flux2, std::move(outlet2),
                                                int_order_offset2),
          bdr_marker2);
    }

    mfem::Vector x2;
    U2.GetTrueDofs(x2);

    mfem::dgsem::DGSEMOperator dgsem_op2(oper2);

    mfem::Vector z_surf2(vfes2.GetTrueVSize());
    mfem::Vector z_vol2(vfes2.GetTrueVSize());
    dgsem_op2.AssembleSurface(x2, z_surf2);
    z_vol2 = 0.0;
    dgsem_op2.AssembleVolume(x2, z_vol2);

    mfem::Vector z_sum2(z_surf2.Size());
    z_sum2 = z_surf2;
    z_sum2 += z_vol2;

    const double norm_sum2 = GlobalL2Norm(z_sum2);
    ctx.Check(norm_sum2 < tol,
              "DGSEM 2D surface + volume cancel for constant state (norm=" +
                  std::to_string(norm_sum2) + ")");

    mfem::Vector y2(vfes2.GetTrueVSize());
    dgsem_op2.ApplyInverseMass(z_sum2, y2);
    const double resid_norm2 = GlobalL2Norm(y2);
    ctx.Check(resid_norm2 < tol,
              "DGSEM 2D residual ~0 after M^{-1} (norm=" +
                  std::to_string(resid_norm2) + ")");

    ctx.AddParity(norm_sum2 + resid_norm2);
  }

  // 2D periodic boundary cancellation (no BC integrators).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;

    const int n = 3;
    const bool sfc = false;
    mfem::Mesh orig_mesh =
        mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::QUADRILATERAL,
                                    /*gen_edges=*/false, 1.0, 1.0, sfc);
    std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                              mfem::Vector({0.0, 1.0})};
    mfem::Mesh mesh =
        mfem::Mesh::MakePeriodic(orig_mesh,
                                 orig_mesh.CreatePeriodicVertexMapping(translations));

    mfem::DG_FECollection fec2(/*order=*/2, dim2, mfem::BasisType::GaussLobatto);
    mfem::FiniteElementSpace vfes2(&mesh, &fec2, num_eq2,
                                   mfem::Ordering::byNODES);

    mfem::GridFunction U2(&vfes2);
    const mfem::Vector cons_const2 =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const2, U2);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);

    constexpr int int_order_offset2 = 2;
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    oper2.SetNumericalFlux(&num_flux2);
    oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

    mfem::Vector x2;
    U2.GetTrueDofs(x2);

    mfem::dgsem::DGSEMOperator dgsem_op2(oper2);
    mfem::Vector z_surf2(vfes2.GetTrueVSize());
    mfem::Vector z_vol2(vfes2.GetTrueVSize());
    dgsem_op2.AssembleSurface(x2, z_surf2);
    z_vol2 = 0.0;
    dgsem_op2.AssembleVolume(x2, z_vol2);

    mfem::Vector z_sum2(z_surf2.Size());
    z_sum2 = z_surf2;
    z_sum2 += z_vol2;

    const double norm_sum2 = GlobalL2Norm(z_sum2);
    ctx.Check(norm_sum2 < tol,
              "DGSEM periodic surface + volume cancel (norm=" +
                  std::to_string(norm_sum2) + ")");

    mfem::Vector y2(vfes2.GetTrueVSize());
    dgsem_op2.ApplyInverseMass(z_sum2, y2);
    const double resid_norm2 = GlobalL2Norm(y2);
    ctx.Check(resid_norm2 < tol,
              "DGSEM periodic residual ~0 after M^{-1} (norm=" +
                  std::to_string(resid_norm2) + ")");

    ctx.AddParity(norm_sum2 + resid_norm2);
  }

  // Face orientation/permute invariance across a shared face.
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;

    mfem::Mesh mesh =
        mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL,
                                    /*gen_edges=*/true, 1.0, 1.0);
    // Rotate vertex ordering for element 0 (permute, preserves orientation).
    {
      mfem::Element* el0 = mesh.GetElement(0);
      int* v = el0->GetVertices();
      const int tmp = v[0];
      v[0] = v[1];
      v[1] = v[2];
      v[2] = v[3];
      v[3] = tmp;
    }
    // Rotate vertex ordering for element 1 with a different permutation.
    {
      mfem::Element* el1 = mesh.GetElement(1);
      int* v = el1->GetVertices();
      const int v0 = v[0];
      const int v1 = v[1];
      const int v2 = v[2];
      const int v3 = v[3];
      v[0] = v2;
      v[1] = v3;
      v[2] = v0;
      v[3] = v1;
    }
    mesh.FinalizeTopology();
    mesh.Finalize(/*refine=*/false, /*fix_orientation=*/false);

    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

    mfem::DG_FECollection fec2(/*order=*/2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    mfem::ParGridFunction U2(&vfes2);
    const mfem::Vector cons_const2 =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const2, U2);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);

    constexpr int int_order_offset2 = 2;
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    oper2.SetNumericalFlux(&num_flux2);
    oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

    mfem::Array<int> bdr_marker2;
    const int max_attr2 = pmesh.bdr_attributes.Max();
    if (max_attr2 > 0)
    {
      bdr_marker2.SetSize(max_attr2);
      bdr_marker2 = 1;
      auto outlet2 = std::make_unique<mfem::SupersonicOutletState>();
      oper2.AddBdrFaceIntegrator(
          new mfem::BdrHyperbolicFluxIntegrator(num_flux2, std::move(outlet2),
                                                int_order_offset2),
          bdr_marker2);
    }

    mfem::Vector x2;
    U2.GetTrueDofs(x2);

    mfem::dgsem::DGSEMOperator dgsem_op2(oper2);
    mfem::Vector z_sum2(vfes2.GetTrueVSize());
    z_sum2 = 0.0;
    dgsem_op2.AssembleSurface(x2, z_sum2);
    dgsem_op2.AssembleVolume(x2, z_sum2);

    mfem::Vector y2(vfes2.GetTrueVSize());
    dgsem_op2.ApplyInverseMass(z_sum2, y2);
    const double resid_norm2 = GlobalL2Norm(y2);
    ctx.Check(resid_norm2 < tol,
              "DGSEM orientation/permute invariance (norm=" +
                  std::to_string(resid_norm2) + ")");
  }

  // High-order constant-state residuals (p=4/5).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    const double tol_hi = 1.0e-9;
    const int orders[] = {4, 5};

    for (const int order : orders)
    {
      mfem::Mesh mesh2 =
          mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                      /*gen_edges=*/true, 1.0, 1.0);
      mfem::ParMesh pmesh2(MPI_COMM_WORLD, mesh2);

      mfem::DG_FECollection fec2(order, dim2, mfem::BasisType::GaussLobatto);
      mfem::ParFiniteElementSpace vfes2(&pmesh2, &fec2, num_eq2,
                                        mfem::Ordering::byNODES);

      mfem::ParGridFunction U2(&vfes2);
      const mfem::Vector cons_const2 =
          EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
      FillConstantState(cons_const2, U2);

      mfem::EulerFlux flux2(dim2, gamma);
      mfem::RusanovFluxWithSides num_flux2(flux2);

      constexpr int int_order_offset2 = 2;
      auto form_integrator2 =
          std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                           int_order_offset2);
      mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                               /*preassembleWeakDivergence=*/true);
      oper2.SetNumericalFlux(&num_flux2);
      oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);

      mfem::Array<int> bdr_marker2;
      const int max_attr2 = pmesh2.bdr_attributes.Max();
      if (max_attr2 > 0)
      {
        bdr_marker2.SetSize(max_attr2);
        bdr_marker2 = 1;
        auto outlet2 = std::make_unique<mfem::SupersonicOutletState>();
        oper2.AddBdrFaceIntegrator(
            new mfem::BdrHyperbolicFluxIntegrator(num_flux2, std::move(outlet2),
                                                  int_order_offset2),
            bdr_marker2);
      }

      mfem::Vector x2;
      U2.GetTrueDofs(x2);

      mfem::dgsem::DGSEMOperator dgsem_op2(oper2);
      mfem::Vector z_sum2(vfes2.GetTrueVSize());
      z_sum2 = 0.0;
      dgsem_op2.AssembleSurface(x2, z_sum2);
      dgsem_op2.AssembleVolume(x2, z_sum2);

      mfem::Vector y2(vfes2.GetTrueVSize());
      dgsem_op2.ApplyInverseMass(z_sum2, y2);
      const double resid_norm2 = GlobalL2Norm(y2);
      ctx.Check(resid_norm2 < tol_hi,
                "DGSEM high-order residual ~0 (p=" +
                    std::to_string(order) +
                    ", norm=" + std::to_string(resid_norm2) + ")");
    }
  }

  // Conservation over one RK step on a periodic mesh (mass/energy invariants).
  {
    constexpr int dim2 = 2;
    constexpr int num_eq2 = dim2 + 2;
    const double tol_cons = 1.0e-10;

    const int n = 3;
    const bool sfc = false;
    mfem::Mesh orig_mesh =
        mfem::Mesh::MakeCartesian2D(n, n, mfem::Element::QUADRILATERAL,
                                    /*gen_edges=*/false, 1.0, 1.0, sfc);
    std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                              mfem::Vector({0.0, 1.0})};
    mfem::Mesh mesh =
        mfem::Mesh::MakePeriodic(orig_mesh,
                                 orig_mesh.CreatePeriodicVertexMapping(translations));
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

    mfem::DG_FECollection fec2(/*order=*/2, dim2, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes2(&pmesh, &fec2, num_eq2,
                                      mfem::Ordering::byNODES);

    class SmoothConservativeCoeff final : public mfem::VectorCoefficient
    {
    public:
      SmoothConservativeCoeff(double gamma)
        : mfem::VectorCoefficient(kNumEq), gamma_(gamma) {}

      void Eval(mfem::Vector& V,
                mfem::ElementTransformation& Tr,
                const mfem::IntegrationPoint& ip) override
      {
        mfem::Vector x(kDim);
        Tr.Transform(ip, x);

        constexpr double kPi = 3.14159265358979323846;
        const double sx = std::sin(2.0 * kPi * x(0));
        const double sy = std::sin(2.0 * kPi * x(1));
        const double rho = 1.0 + 0.1 * sx * sy;
        const double u = 0.2 + 0.05 * std::cos(2.0 * kPi * x(1));
        const double v = -0.1 + 0.05 * std::sin(2.0 * kPi * x(0));
        const double p = 1.0 + 0.05 * sx;

        V = EulerConsFromPrim2D(rho, u, v, p, gamma_);
      }

    private:
      enum { kDim = 2, kNumEq = kDim + 2 };
      double gamma_;
    };

    mfem::ParGridFunction U2(&vfes2);
    SmoothConservativeCoeff coeff(gamma);
    U2.ProjectCoefficient(coeff);

    const double mass_before = IntegrateComponent(U2, 0);
    const double momx_before = IntegrateComponent(U2, 1);
    const double momy_before = IntegrateComponent(U2, 2);
    const double energy_before = IntegrateComponent(U2, 3);

    mfem::EulerFlux flux2(dim2, gamma);
    mfem::RusanovFluxWithSides num_flux2(flux2);

    constexpr int int_order_offset2 = 2;
    auto form_integrator2 =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux2,
                                                         int_order_offset2);
    mfem::DGHyperbolicConservationLaws oper2(vfes2, std::move(form_integrator2),
                                             /*preassembleWeakDivergence=*/true);
    oper2.SetNumericalFlux(&num_flux2);
    oper2.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper2.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(/*order=*/2));

    mfem::RK3SSPWithLimiter rk(vfes2);
    rk.Init(oper2);

    double t = 0.0;
    double dt = 1.0e-3;
    mfem::Vector x2;
    U2.GetTrueDofs(x2);
    rk.Step(x2, t, dt);
    U2.SetFromTrueDofs(x2);

    const double mass_after = IntegrateComponent(U2, 0);
    const double momx_after = IntegrateComponent(U2, 1);
    const double momy_after = IntegrateComponent(U2, 2);
    const double energy_after = IntegrateComponent(U2, 3);

    ctx.CheckNear(mass_after, mass_before, tol_cons,
                  "DGSEM periodic RK3 mass conserved");
    ctx.CheckNear(momx_after, momx_before, tol_cons,
                  "DGSEM periodic RK3 x-momentum conserved");
    ctx.CheckNear(momy_after, momy_before, tol_cons,
                  "DGSEM periodic RK3 y-momentum conserved");
    ctx.CheckNear(energy_after, energy_before, tol_cons,
                  "DGSEM periodic RK3 energy conserved");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests

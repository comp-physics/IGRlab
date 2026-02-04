#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_entropy_flux.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/igr/igr_flux.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <cstdlib>
#include <string>
#include <vector>

namespace dg_igr_tests
{

namespace
{

double MaxAbsEntry(const mfem::Vector& vec)
{
  double max_val = 0.0;
  for (int i = 0; i < vec.Size(); ++i)
  {
    max_val = std::max(max_val, std::abs(vec(i)));
  }
  return max_val;
}

} // namespace

int RunDGSEMIGRTests(TestContext& ctx)
{
  const double tol = 1.0e-10;
  constexpr int dim = 2;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;

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

  mfem::DG_FECollection fec(/*order=*/2, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_eq,
                                   mfem::Ordering::byNODES);

  mfem::ParGridFunction U(&vfes);
  const mfem::Vector cons_const =
      EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
  FillConstantState(cons_const, U);

  mfem::DG_FECollection sigma_fec(/*order=*/1, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace sigma_fes(&pmesh, &sigma_fec, /*vdim=*/1,
                                        mfem::Ordering::byNODES);
  mfem::ParGridFunction sigma(&sigma_fes);
  const double sigma_value = 0.25;
  mfem::ConstantCoefficient sigma_coeff(sigma_value);
  sigma.ProjectCoefficient(sigma_coeff);
  sigma.ExchangeFaceNbrData();

  mfem::EulerFlux base_flux(dim, gamma);
  mfem::FluxWithIGR igr_flux(base_flux);
  igr_flux.SetSigmaField(&sigma);

  mfem::DGSEMSurfaceFlux num_flux(igr_flux);
  num_flux.SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::Rusanov);
  num_flux.SetGamma(gamma);

  constexpr int int_order_offset = 2;
  auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                       int_order_offset);
  mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form_integrator),
                                          /*preassembleWeakDivergence=*/true);
  oper.SetNumericalFlux(&num_flux);
  oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
  oper.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(/*order=*/2));

  mfem::Vector x;
  mfem::Vector y(vfes.GetTrueVSize());
  U.GetTrueDofs(x);
  oper.Mult(x, y);

  const double resid_norm = GlobalL2Norm(y);
  const double resid_max = GlobalMax(MaxAbsEntry(y));
  ctx.AddParity(resid_norm);
  ctx.Check(resid_norm < tol,
            "DGSEM + IGR constant-sigma residual ~0 (norm=" +
                FormatSci(resid_norm) +
                ", max|entry|=" + FormatSci(resid_max) + ")");

  // Non-periodic mesh + BC wiring: constant sigma should still cancel.
  {
    mfem::Mesh mesh_np = mfem::Mesh::MakeCartesian2D(
        /*nx=*/2, /*ny=*/2, mfem::Element::QUADRILATERAL,
        /*gen_edges=*/true, /*sx=*/1.0, /*sy=*/1.0);
    mfem::ParMesh pmesh_np(MPI_COMM_WORLD, mesh_np);

    mfem::DG_FECollection fec_np(/*order=*/2, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes_np(&pmesh_np, &fec_np, num_eq,
                                        mfem::Ordering::byNODES);

    mfem::ParGridFunction U_np(&vfes_np);
    FillConstantState(cons_const, U_np);

    mfem::DG_FECollection sigma_fec_np(/*order=*/1, dim,
                                       mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace sigma_fes_np(&pmesh_np, &sigma_fec_np, /*vdim=*/1,
                                             mfem::Ordering::byNODES);
    mfem::ParGridFunction sigma_np(&sigma_fes_np);
    sigma_np.ProjectCoefficient(sigma_coeff);
    sigma_np.ExchangeFaceNbrData();

    mfem::FluxWithIGR igr_flux_np(base_flux);
    igr_flux_np.SetSigmaField(&sigma_np);

    mfem::DGSEMSurfaceFlux num_flux_np(igr_flux_np);
    num_flux_np.SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::Rusanov);
    num_flux_np.SetGamma(gamma);

    auto form_integrator_np =
        std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux_np,
                                                         int_order_offset);
    mfem::DGHyperbolicConservationLaws oper_np(vfes_np,
                                               std::move(form_integrator_np),
                                               /*preassembleWeakDivergence=*/true);
    oper_np.SetNumericalFlux(&num_flux_np);
    oper_np.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper_np.SetFaceIntegrationRule(mfem::dgsem::GetLGLFaceRule(/*order=*/2));

    mfem::Array<int> bdr_marker;
    const int max_attr = pmesh_np.bdr_attributes.Max();
    if (max_attr > 0)
    {
      bdr_marker.SetSize(max_attr);
      bdr_marker = 1;
      auto outlet = std::make_unique<mfem::SupersonicOutletState>();
      oper_np.AddBdrFaceIntegrator(
          new mfem::BdrHyperbolicFluxIntegrator(num_flux_np,
                                                std::move(outlet),
                                                int_order_offset),
          bdr_marker);
    }

    mfem::Vector x_np;
    mfem::Vector y_np(vfes_np.GetTrueVSize());
    U_np.GetTrueDofs(x_np);
    oper_np.Mult(x_np, y_np);

    const double resid_norm_np = GlobalL2Norm(y_np);
    const double resid_max_np = GlobalMax(MaxAbsEntry(y_np));
    ctx.Check(resid_norm_np < tol,
              "DGSEM + IGR constant-sigma residual ~0 with BCs (norm=" +
                  FormatSci(resid_norm_np) +
                  ", max|entry|=" + FormatSci(resid_max_np) + ")");
  }

  // DGSEM + IGR entropy production regression (smooth state).
  {
    const mfem::Vector uL =
        EulerConsFromPrim2D(1.0, 0.2, -0.1, 1.0, gamma);
    const mfem::Vector uR =
        EulerConsFromPrim2D(1.01, 0.22, -0.08, 1.02, gamma);
    mfem::Vector normal(2);
    normal(0) = 0.6;
    normal(1) = -0.8;

    mfem::FaceElementTransformations* Tr_face = pmesh.GetFaceElementTransformations(0);
    MFEM_VERIFY(Tr_face != nullptr, "Failed to obtain face transformations.");
    const mfem::IntegrationRule& ir_face =
        mfem::IntRules.Get(mfem::Geometry::SEGMENT, /*order=*/1);
    const mfem::IntegrationPoint& ip_face = ir_face.IntPoint(0);
    Tr_face->SetAllIntPoints(&ip_face);

    auto entropy_prod = [&](mfem::DGSEMSurfaceFluxType surface_type,
                            mfem::dgsem::EntropyJacobianMode mode) {
      mfem::DGSEMSurfaceFlux es_flux(igr_flux);
      es_flux.SetSurfaceFluxType(surface_type);
      es_flux.SetEntropyFluxType(mfem::dgsem::EntropyFluxType::Ranocha);
      es_flux.SetEntropyJacobianMode(mode);
      es_flux.SetGamma(gamma);

      mfem::Vector flux_es(uL.Size());
      es_flux.Eval(uL, uR, normal, *Tr_face, flux_es);

      mfem::Vector flux_ec(uL.Size());
      mfem::dgsem::EntropyConservativeFluxDotN(
          uL, uR, normal, gamma, mfem::dgsem::EntropyFluxType::Ranocha,
          flux_ec, sigma_value, sigma_value);

      mfem::Vector diss(uL.Size());
      diss = flux_es;
      diss -= flux_ec;

      mfem::Vector wL, wR;
      mfem::dgsem::EulerEntropyVariables(uL, /*dim=*/2, gamma, sigma_value, wL);
      mfem::dgsem::EulerEntropyVariables(uR, /*dim=*/2, gamma, sigma_value, wR);

      mfem::Vector wjump(wL.Size());
      wjump = wR;
      wjump -= wL;

      return wjump * diss;
    };

    const double prod_winters =
        entropy_prod(mfem::DGSEMSurfaceFluxType::EntropyStableWinters,
                     mfem::dgsem::EntropyJacobianMode::FiniteDifference);
    const double prod_lf_fd =
        entropy_prod(mfem::DGSEMSurfaceFluxType::EntropyStableLFEntropyVars,
                     mfem::dgsem::EntropyJacobianMode::FiniteDifference);
    const double prod_lf_analytic =
        entropy_prod(mfem::DGSEMSurfaceFluxType::EntropyStableLFEntropyVars,
                     mfem::dgsem::EntropyJacobianMode::Analytic);

    // Baselines recorded from DG_IGR_PRINT_ENTROPY=1 runs of this test.
    const double expected_winters = -1.81662e-4;
    const double expected_lf = -5.74081e-4;
    const double tol_prod = 1.0e-2;

    if (std::getenv("DG_IGR_PRINT_ENTROPY"))
    {
      std::cout << "igr_entropy_prod_winters=" << prod_winters << "\n";
      std::cout << "igr_entropy_prod_lf_fd=" << prod_lf_fd << "\n";
      std::cout << "igr_entropy_prod_lf_analytic=" << prod_lf_analytic << "\n";
    }

    ctx.Check(prod_winters < 0.0,
              "IGR entropy production (Winters) is non-positive");
    ctx.Check(prod_lf_fd < 0.0,
              "IGR entropy production (LF entropy vars, FD) is non-positive");
    ctx.Check(prod_lf_analytic < 0.0,
              "IGR entropy production (LF entropy vars, analytic) is non-positive");

    ctx.CheckNear(prod_winters, expected_winters, tol_prod,
                  "IGR entropy production (Winters) regression");
    ctx.CheckNear(prod_lf_fd, expected_lf, tol_prod,
                  "IGR entropy production (LF entropy vars, FD) regression");
    ctx.CheckNear(prod_lf_analytic, expected_lf, tol_prod,
                  "IGR entropy production (LF entropy vars, analytic) regression");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests

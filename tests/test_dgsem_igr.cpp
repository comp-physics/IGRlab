#include "igrlab/environment.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_entropy_flux.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/igr/igr_flux.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"
#include "test_dgsem_reference_igr.hpp"

#include <cstdlib>
#include <cstring>
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

// A derived wrapper retains the original side-dispatch composition and also
// checks that optimizing the concrete wrapper does not bypass user overrides.
class LegacyIGRSpeedOracle final : public mfem::FluxWithIGR
{
public:
  using mfem::FluxWithIGR::FluxWithIGR;
  mutable int side_calls = 0;
  mfem::real_t speed_offset = 0.0;

  mfem::real_t ComputeFluxDotNAtSide(
    const mfem::Vector& state,
    const mfem::Vector& normal,
    mfem::FaceElementTransformations& tr,
    mfem::FaceSide side,
    mfem::Vector& flux) const override
  {
    ++side_calls;
    const mfem::real_t speed =
      mfem::FluxWithIGR::ComputeFluxDotNAtSide(state, normal, tr, side, flux);
    return speed_offset == 0.0 ? speed : speed + speed_offset;
  }
};

void CheckEntropySurfaceSpeedOracle(TestContext& ctx)
{
  constexpr double gamma = 1.4;
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  mfem::DG_FECollection fec(2, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fes(&mesh, &fec);
  mfem::ParGridFunction sigma(&fes);
  for (int element = 0; element < fes.GetNE(); ++element)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(element, dofs);
    mfem::Vector values(dofs.Size());
    for (int i = 0; i < values.Size(); ++i)
    {
      values(i) = 0.05 + 0.125 * element + 0.01 * i;
    }
    sigma.SetSubVector(dofs, values);
  }

  int interior_face = -1;
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    if (mesh.GetInteriorFaceTransformations(face))
    {
      interior_face = face;
      break;
    }
  }
  MFEM_VERIFY(interior_face >= 0 && mesh.GetNBE() > 0,
              "Entropy surface oracle requires interior and boundary faces.");
  const mfem::IntegrationRule& rule =
    mfem::IntRules.Get(mfem::Geometry::SEGMENT, 3);
  const mfem::Vector left = EulerConsFromPrim2D(1.1, 0.4, -0.3, 1.3, gamma);
  const mfem::Vector right = EulerConsFromPrim2D(0.8, -0.2, 0.15, 0.9, gamma);
  const mfem::Vector normal({1.2, -1.6});
  mfem::EulerFlux base(2, gamma);
  mfem::FluxWithIGR actual_igr(base);
  LegacyIGRSpeedOracle oracle_igr(base);
  mfem::DGSEMSurfaceFlux actual(actual_igr), oracle(oracle_igr);
  actual.SetGamma(gamma);
  oracle.SetGamma(gamma);
  actual.SetEntropyJacobianMode(mfem::dgsem::EntropyJacobianMode::Analytic);
  oracle.SetEntropyJacobianMode(mfem::dgsem::EntropyJacobianMode::Analytic);
  bool exact_values = true;
  bool side_dispatch = true;
  bool exact_telemetry = true;
  bool discontinuous_sigma = false;

  // Change the policies on existing flux objects, including returning to the
  // fast path after all policies were enabled. Null sigma also remains valid.
  for (const auto* field : {&sigma, static_cast<mfem::ParGridFunction*>(nullptr)})
  {
    actual_igr.SetSigmaField(field);
    oracle_igr.SetSigmaField(field);
    for (const int policy : {0, 1, 2, 4, 8, 15, 0})
    {
      for (mfem::FluxWithIGR* igr :
           {&actual_igr, static_cast<mfem::FluxWithIGR*>(&oracle_igr)})
      {
        igr->SetSigmaAwareCFL(policy & 1);
        igr->SetSigmaAwareRusanov(policy & 2);
        igr->SetFrozenSigmaSpectralSpeed(policy & 4);
        igr->SetInvariantDomainFlux(policy & 8);
      }
      for (const auto type :
           {mfem::DGSEMSurfaceFluxType::EntropyStableWinters,
            mfem::DGSEMSurfaceFluxType::EntropyStableLFEntropyVars})
      {
        actual.SetSurfaceFluxType(type);
        oracle.SetSurfaceFluxType(type);
        for (const bool boundary : {false, true})
        {
          auto* tr = boundary ? mesh.GetBdrFaceTransformations(0)
                              : mesh.GetInteriorFaceTransformations(interior_face);
          MFEM_VERIFY(tr, "Missing entropy surface oracle transformation.");
          tr->SetAllIntPoints(&rule.IntPoint(boundary ? 0 : 1));
          if (field && !boundary)
          {
            discontinuous_sigma =
              actual_igr.GetSigmaAtSide(*tr, mfem::FaceSide::Left) !=
              actual_igr.GetSigmaAtSide(*tr, mfem::FaceSide::Right);
          }
          const auto before = actual_igr.GetFrozenSigmaTelemetry();
          mfem::Vector actual_flux(4), oracle_flux(4);
          const mfem::real_t actual_speed =
            actual.Eval(left, right, normal, *tr, actual_flux);
          tr->SetAllIntPoints(&rule.IntPoint(boundary ? 0 : 1));
          oracle_igr.side_calls = 0;
          const mfem::real_t oracle_speed =
            oracle.Eval(left, right, normal, *tr, oracle_flux);
          exact_values = exact_values &&
            std::memcmp(&actual_speed, &oracle_speed, sizeof(mfem::real_t)) == 0 &&
            std::memcmp(actual_flux.HostRead(), oracle_flux.HostRead(),
                        4 * sizeof(mfem::real_t)) == 0;
          side_dispatch = side_dispatch && oracle_igr.side_calls == 2;
          const auto a = actual_igr.GetFrozenSigmaTelemetry();
          const auto b = oracle_igr.GetFrozenSigmaTelemetry();
          const std::uint64_t expected_samples = field && (policy & 12) ? 2 : 0;
          exact_telemetry = exact_telemetry &&
            a.sample_count - before.sample_count == expected_samples &&
            a.enabled == b.enabled && a.sample_count == b.sample_count &&
            a.nonpositive_count == b.nonpositive_count &&
            a.nonfinite_count == b.nonfinite_count &&
            a.minimum_raw_q_sigma == b.minimum_raw_q_sigma &&
            a.minimum_q_sigma_over_gamma_p == b.minimum_q_sigma_over_gamma_p;
        }
      }
    }
  }
  ctx.Check(discontinuous_sigma, "Entropy surface oracle samples unequal sigma traces");
  ctx.Check(exact_values,
            "Entropy surface flux and speed match legacy side dispatch bitwise");
  ctx.Check(side_dispatch, "Entropy surface retains derived IGR side overrides");
  ctx.Check(exact_telemetry,
            "Entropy surface policy transitions retain exact sigma telemetry");

  auto* tr = mesh.GetInteriorFaceTransformations(interior_face);
  tr->SetAllIntPoints(&rule.IntPoint(0));
  oracle.SetSurfaceFluxType(mfem::DGSEMSurfaceFluxType::EntropyStableWinters);
  mfem::Vector flux(4);
  const mfem::real_t original_speed = oracle.Eval(left, right, normal, *tr, flux);
  oracle_igr.speed_offset = 0.125;
  oracle_igr.side_calls = 0;
  const mfem::real_t overridden_speed = oracle.Eval(left, right, normal, *tr, flux);
  ctx.Check(oracle_igr.side_calls == 2 &&
              overridden_speed == original_speed + oracle_igr.speed_offset,
            "Entropy surface preserves a derived wrapper's speed override");
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
    mfem::Mesh::MakeCartesian2D(n,
                                n,
                                mfem::Element::QUADRILATERAL,
                                /*gen_edges=*/false,
                                1.0,
                                1.0,
                                sfc);
  std::vector<mfem::Vector> translations = {mfem::Vector({1.0, 0.0}),
                                            mfem::Vector({0.0, 1.0})};
  mfem::Mesh mesh = mfem::Mesh::MakePeriodic(
    orig_mesh,
    orig_mesh.CreatePeriodicVertexMapping(translations));

  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

  mfem::DG_FECollection fec(/*order=*/2, dim, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace vfes(&pmesh,
                                   &fec,
                                   num_eq,
                                   mfem::Ordering::byNODES);

  mfem::ParGridFunction U(&vfes);
  const mfem::Vector cons_const =
    EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
  FillConstantState(cons_const, U);

  mfem::DG_FECollection sigma_fec(/*order=*/1,
                                  dim,
                                  mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace sigma_fes(&pmesh,
                                        &sigma_fec,
                                        /*vdim=*/1,
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
  mfem::DGHyperbolicConservationLaws oper(vfes,
                                          std::move(form_integrator),
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
  ctx.Check(
    resid_norm < tol,
    "DGSEM + IGR constant-sigma residual ~0 (norm=" + FormatSci(resid_norm) +
      ", max|entry|=" + FormatSci(resid_max) + ")");

  // Non-periodic mesh + BC wiring: constant sigma should still cancel.
  {
    mfem::Mesh mesh_np = mfem::Mesh::MakeCartesian2D(
      /*nx=*/2,
      /*ny=*/2,
      mfem::Element::QUADRILATERAL,
      /*gen_edges=*/true,
      /*sx=*/1.0,
      /*sy=*/1.0);
    mfem::ParMesh pmesh_np(MPI_COMM_WORLD, mesh_np);

    mfem::DG_FECollection fec_np(/*order=*/2,
                                 dim,
                                 mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes_np(&pmesh_np,
                                        &fec_np,
                                        num_eq,
                                        mfem::Ordering::byNODES);

    mfem::ParGridFunction U_np(&vfes_np);
    FillConstantState(cons_const, U_np);

    mfem::DG_FECollection sigma_fec_np(/*order=*/1,
                                       dim,
                                       mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace sigma_fes_np(&pmesh_np,
                                             &sigma_fec_np,
                                             /*vdim=*/1,
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
    mfem::DGHyperbolicConservationLaws oper_np(
      vfes_np,
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
    const mfem::Vector uL = EulerConsFromPrim2D(1.0, 0.2, -0.1, 1.0, gamma);
    const mfem::Vector uR = EulerConsFromPrim2D(1.01, 0.22, -0.08, 1.02, gamma);
    mfem::Vector normal(2);
    normal(0) = 0.6;
    normal(1) = -0.8;

    mfem::FaceElementTransformations* Tr_face =
      pmesh.GetFaceElementTransformations(0);
    MFEM_VERIFY(Tr_face != nullptr, "Failed to obtain face transformations.");
    const mfem::IntegrationRule& ir_face =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, /*order=*/1);
    const mfem::IntegrationPoint& ip_face = ir_face.IntPoint(0);
    Tr_face->SetAllIntPoints(&ip_face);

    auto entropy_prod = [&](mfem::DGSEMSurfaceFluxType surface_type,
                            mfem::dgsem::EntropyJacobianMode mode)
    {
      mfem::DGSEMSurfaceFlux es_flux(igr_flux);
      es_flux.SetSurfaceFluxType(surface_type);
      es_flux.SetEntropyFluxType(mfem::dgsem::EntropyFluxType::Ranocha);
      es_flux.SetEntropyJacobianMode(mode);
      es_flux.SetGamma(gamma);

      mfem::Vector flux_es(uL.Size());
      es_flux.Eval(uL, uR, normal, *Tr_face, flux_es);

      mfem::Vector flux_ec(uL.Size());
      mfem::dgsem::EntropyConservativeFluxDotN(
        uL,
        uR,
        normal,
        gamma,
        mfem::dgsem::EntropyFluxType::Ranocha,
        flux_ec,
        sigma_value,
        sigma_value);

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

    if (::igrlab::GetEnv("DG_IGR_PRINT_ENTROPY"))
    {
      std::cout << "igr_entropy_prod_winters=" << prod_winters << "\n";
      std::cout << "igr_entropy_prod_lf_fd=" << prod_lf_fd << "\n";
      std::cout << "igr_entropy_prod_lf_analytic=" << prod_lf_analytic << "\n";
    }

    ctx.Check(prod_winters < 0.0,
              "IGR entropy production (Winters) is non-positive");
    ctx.Check(prod_lf_fd < 0.0,
              "IGR entropy production (LF entropy vars, FD) is non-positive");
    ctx.Check(
      prod_lf_analytic < 0.0,
      "IGR entropy production (LF entropy vars, analytic) is non-positive");

    ctx.CheckNear(prod_winters,
                  expected_winters,
                  tol_prod,
                  "IGR entropy production (Winters) regression");
    ctx.CheckNear(prod_lf_fd,
                  expected_lf,
                  tol_prod,
                  "IGR entropy production (LF entropy vars, FD) regression");
    ctx.CheckNear(
      prod_lf_analytic,
      expected_lf,
      tol_prod,
      "IGR entropy production (LF entropy vars, analytic) regression");
  }

  RunDGSEMScalarIGRScopeFixture(ctx);
  CheckEntropySurfaceSpeedOracle(ctx);

  return ctx.failures;
}

} // namespace igrlab_tests

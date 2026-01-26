#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/bc/boundary_state.hpp"

#include <memory>
#include <string>

#include "mfem.hpp"
#include "test_utils.hpp"

namespace dg_igr_tests
{

int RunDGOperatorTests(TestContext& ctx)
{
  constexpr int dim = 2;
  constexpr int num_eq = dim + 2;
  constexpr double gamma = 1.4;
  const double tol = 1.0e-10;

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
      /*nx=*/2, /*ny=*/2, mfem::Element::QUADRILATERAL,
      /*generate_edges=*/true, /*sx=*/1.0, /*sy=*/1.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

  mfem::DG_FECollection fec(/*order=*/1, dim);
  mfem::ParFiniteElementSpace vfes(&pmesh, &fec, /*vdim=*/num_eq,
                                   mfem::Ordering::byNODES);

  mfem::ParGridFunction U(&vfes);
  const mfem::Vector cons_const = EulerConsFromPrim2D(
      /*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
  FillConstantState(cons_const, U);

  mfem::EulerFlux flux(dim, gamma);
  mfem::RusanovFluxWithSides num_flux(flux);

  constexpr int int_order_offset = 2;
  auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux, int_order_offset);
  mfem::DGHyperbolicConservationLaws oper(vfes, std::move(form_integrator),
                                          /*preassembleWeakDivergence=*/true);

  // Apply slip-wall boundaries on all attributes to close the domain.
  mfem::Array<int> bdr_marker;
  const int max_attr = pmesh.bdr_attributes.Max();
  if (max_attr > 0)
  {
    bdr_marker.SetSize(max_attr);
    bdr_marker = 1;
    auto slip = std::make_unique<mfem::SlipWallState>(dim);
    oper.AddBdrFaceIntegrator(
        new mfem::BdrHyperbolicFluxIntegrator(num_flux, std::move(slip),
                                              int_order_offset),
        bdr_marker);
  }

  mfem::Vector x;
  mfem::Vector y(vfes.GetTrueVSize());
  U.GetTrueDofs(x);
  oper.Mult(x, y);

  const double resid_norm = GlobalL2Norm(y);
  ctx.AddParity(resid_norm);
  ctx.Check(resid_norm < tol,
            "DG operator residual is near zero for a constant state (norm=" +
                std::to_string(resid_norm) + ")");

  return ctx.failures;
}

} // namespace dg_igr_tests

#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_flux.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"
#include "numerics/dgsem/dgsem_volume_flux.hpp"

#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/bc/boundary_state.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace dg_igr_tests
{

namespace
{

double MaxAbsEntry(const mfem::DenseMatrix& mat)
{
  double max_val = 0.0;
  for (int i = 0; i < mat.Height(); ++i)
  {
    for (int j = 0; j < mat.Width(); ++j)
    {
      max_val = std::max(max_val, std::abs(mat(i, j)));
    }
  }
  return max_val;
}

double MaxAbsEntry(const mfem::Vector& vec)
{
  double max_val = 0.0;
  for (int i = 0; i < vec.Size(); ++i)
  {
    max_val = std::max(max_val, std::abs(vec(i)));
  }
  return max_val;
}

mfem::DenseMatrix BuildConstantStateMatrix(int dof, const mfem::Vector& state)
{
  const int num_eq = state.Size();
  mfem::DenseMatrix u(dof, num_eq);
  for (int i = 0; i < dof; ++i)
  {
    for (int eq = 0; eq < num_eq; ++eq)
    {
      u(i, eq) = state(eq);
    }
  }
  return u;
}

std::vector<double> OffDiagonalRowSum(const mfem::DenseMatrix& mat)
{
  std::vector<double> sums(mat.Height(), 0.0);
  for (int i = 0; i < mat.Height(); ++i)
  {
    double s = 0.0;
    for (int j = 0; j < mat.Width(); ++j)
    {
      if (i == j) { continue; }
      s += mat(i, j);
    }
    sums[i] = s;
  }
  return sums;
}

} // namespace

int RunDGSEMFluxTests(TestContext& ctx)
{
  const double tol = 1.0e-12;

  // SymmetricAverageContravariantFlux helper.
  {
    const int num_eq = 3;
    const int dim = 2;
    const int dof = 4;
    mfem::DenseMatrix flux(num_eq, dim * dof);
    for (int eq = 0; eq < num_eq; ++eq)
    {
      for (int col = 0; col < dim * dof; ++col)
      {
        flux(eq, col) = 0.1 * (eq + 1) + 0.01 * (col + 1);
      }
    }

    mfem::Vector out(num_eq);
    mfem::Vector out_swap(num_eq);
    const int local_i = 1;
    const int local_j = 3;
    const int dir = 1;

    mfem::dgsem::SymmetricAverageContravariantFlux(
        flux, local_i, local_j, dir, dim, out);
    mfem::dgsem::SymmetricAverageContravariantFlux(
        flux, local_j, local_i, dir, dim, out_swap);

    for (int eq = 0; eq < num_eq; ++eq)
    {
      const double expected =
          0.5 * (flux(eq, local_i * dim + dir) + flux(eq, local_j * dim + dir));
      ctx.CheckNear(out(eq), expected, tol,
                    "SymmetricAverageContravariantFlux matches average");
      ctx.CheckNear(out_swap(eq), expected, tol,
                    "SymmetricAverageContravariantFlux is symmetric");
    }
  }

  // LGL face rule sanity.
  {
    const int order = 4;
    const mfem::IntegrationRule& ir = mfem::dgsem::GetLGLFaceRule(order);
    ctx.Check(ir.GetNPoints() == order + 1,
              "LGL face rule uses p+1 points");
    ctx.CheckNear(ir.IntPoint(0).x, 0.0, tol,
                  "LGL face rule starts at x=0");
    ctx.CheckNear(ir.IntPoint(ir.GetNPoints() - 1).x, 1.0, tol,
                  "LGL face rule ends at x=1");
    double weight_sum = 0.0;
    for (int i = 0; i < ir.GetNPoints(); ++i)
    {
      weight_sum += ir.IntPoint(i).weight;
    }
    ctx.CheckNear(weight_sum, 1.0, tol,
                  "LGL face rule weights sum to 1");
    ctx.Check(ir.GetOrder() == 2 * order - 1,
              "LGL face rule order is 2p-1");
  }

  // 2D surface quadrature sanity vs LGL face nodes.
  {
    const int order = 3;
    const auto& basis = mfem::dgsem::GetLGLBasis1D(order);
    const mfem::Vector& nodes1d = basis.nodes;

    mfem::DG_FECollection fec(order, 2, mfem::BasisType::GaussLobatto);
    const mfem::FiniteElement* fe =
        fec.FiniteElementForGeometry(mfem::Geometry::SQUARE);

    const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(*fe);
    const int n1d = order + 1;

    auto check_face = [&](const std::string& label, bool check_i, int value) {
      std::vector<int> found(n1d, 0);
      for (const auto& ijk : dof_to_ijk)
      {
        const int i = ijk[0];
        const int j = ijk[1];
        if ((check_i ? i : j) != value) { continue; }
        const int tangential = check_i ? j : i;
        found[tangential] = 1;
      }
      for (int n = 0; n < n1d; ++n)
      {
        ctx.Check(found[n] == 1,
                  "Face contains all LGL nodes (" + label + ")");
      }
    };

    check_face("x=0", /*check_i=*/true, /*value=*/0);
    check_face("x=1", /*check_i=*/true, /*value=*/n1d - 1);
    check_face("y=0", /*check_i=*/false, /*value=*/0);
    check_face("y=1", /*check_i=*/false, /*value=*/n1d - 1);
  }

  // Flux differencing kernels: constant state matches Dsplit off-diagonal sums.
  {
    const int order = 3;
    const int num_eq = 3;
    const auto& basis = mfem::dgsem::GetLGLBasis1D(order);
    const int n1d = basis.n;
    const mfem::Vector state =
        EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/0.0, /*p=*/1.0, /*gamma=*/1.4);

    auto central_flux = [](const mfem::Vector& ul, const mfem::Vector& ur,
                           int, mfem::Vector& f) {
      const int n = ul.Size();
      f.SetSize(n);
      for (int i = 0; i < n; ++i)
      {
        f(i) = 0.5 * (ul(i) + ur(i));
      }
    };

    // Symmetry check for the two-point flux callback.
    {
      mfem::Vector a(num_eq), b(num_eq), fab(num_eq), fba(num_eq);
      for (int i = 0; i < num_eq; ++i)
      {
        a(i) = 1.0 + 0.2 * i;
        b(i) = 2.0 + 0.3 * i;
      }
      central_flux(a, b, 0, fab);
      central_flux(b, a, 0, fba);
      for (int i = 0; i < num_eq; ++i)
      {
        ctx.CheckNear(fab(i), fba(i), tol,
                      "Central flux is symmetric in its states");
      }
    }

    const std::vector<double> row_sum = OffDiagonalRowSum(basis.Dsplit);

    // 1D kernel
    {
      mfem::DenseMatrix u = BuildConstantStateMatrix(n1d, state);
      mfem::DenseMatrix du;
      mfem::dgsem::FluxDifferencing1D(u, basis.Dsplit, central_flux, du);
      mfem::DenseMatrix expected(n1d, num_eq);
      for (int i = 0; i < n1d; ++i)
      {
        for (int eq = 0; eq < num_eq; ++eq)
        {
          expected(i, eq) = row_sum[i] * state(eq);
        }
      }
      expected -= du;
      const double err = MaxAbsEntry(expected);
      ctx.Check(err < 1.0e-10,
                "FluxDifferencing1D matches Dsplit off-diagonal sums");
    }

    // Manufactured polynomial exactness (p=2, u=x(1-x)).
    {
      const int poly_order = 2;
      const auto& poly_basis = mfem::dgsem::GetLGLBasis1D(poly_order);
      const int n = poly_basis.n;

      mfem::DenseMatrix u(n, 1);
      mfem::Vector ucol(n);
      for (int i = 0; i < n; ++i)
      {
        const double x = poly_basis.nodes(i);
        const double val = x * (1.0 - x);
        u(i, 0) = val;
        ucol(i) = val;
      }

      auto linear_flux = [](const mfem::Vector& ul, const mfem::Vector& ur,
                            int, mfem::Vector& f) {
        f.SetSize(1);
        f(0) = 0.5 * (ul(0) + ur(0));
      };

      mfem::DenseMatrix du;
      mfem::dgsem::FluxDifferencing1D(u, poly_basis.Dsplit, linear_flux, du);

      mfem::Vector dcol(n);
      poly_basis.D.Mult(ucol, dcol);

      mfem::DenseMatrix expected(n, 1);
      for (int i = 0; i < n; ++i)
      {
        expected(i, 0) = dcol(i);
      }
      expected -= du;
      const double err = MaxAbsEntry(expected);
      ctx.Check(err < 1.0e-10,
                "FluxDifferencing1D matches strong derivative for u=x(1-x)");
    }

    // 2D kernel
    {
      const int dim = 2;
      const int dof = n1d * n1d;
      mfem::DenseMatrix u = BuildConstantStateMatrix(dof, state);
      mfem::DenseMatrix du;

      mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
      const mfem::FiniteElement* fe =
          fec.FiniteElementForGeometry(mfem::Geometry::SQUARE);
      const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(*fe);

      mfem::dgsem::FluxDifferencing2D(u, basis.Dsplit, dof_to_ijk,
                                      central_flux, du);
      mfem::DenseMatrix expected(dof, num_eq);
      for (int local = 0; local < dof; ++local)
      {
        const int i = dof_to_ijk[local][0];
        const int j = dof_to_ijk[local][1];
        const double scale = row_sum[i] + row_sum[j];
        for (int eq = 0; eq < num_eq; ++eq)
        {
          expected(local, eq) = scale * state(eq);
        }
      }
      expected -= du;
      const double err = MaxAbsEntry(expected);
      ctx.Check(err < 1.0e-10,
                "FluxDifferencing2D matches Dsplit off-diagonal sums");
    }

    // 2D kernel on a single quad mesh (ordering sanity).
    {
      const int dim = 2;
      mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                                    /*gen_edges=*/true, 1.0, 1.0);
      mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
      mfem::ParFiniteElementSpace fes(&pmesh, &fec, num_eq,
                                      mfem::Ordering::byNODES);

      const mfem::FiniteElement* fe = fes.GetFE(0);
      const int dof = fe->GetDof();
      const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(*fe);

      mfem::DenseMatrix u = BuildConstantStateMatrix(dof, state);
      mfem::DenseMatrix du;
      mfem::dgsem::FluxDifferencing2D(u, basis.Dsplit, dof_to_ijk,
                                      central_flux, du);

      mfem::DenseMatrix expected(dof, num_eq);
      for (int local = 0; local < dof; ++local)
      {
        const int i = dof_to_ijk[local][0];
        const int j = dof_to_ijk[local][1];
        const double scale = row_sum[i] + row_sum[j];
        for (int eq = 0; eq < num_eq; ++eq)
        {
          expected(local, eq) = scale * state(eq);
        }
      }
      expected -= du;
      const double err = MaxAbsEntry(expected);
      ctx.Check(err < 1.0e-10,
                "FluxDifferencing2D matches Dsplit sums on single quad");
    }
  }

  // Integration test: DGSEM volume mode constant-state residual in 1D.
  {
    constexpr int dim = 1;
    constexpr int num_eq = dim + 2;
    constexpr double gamma = 1.4;
    const double tol_resid = 1.0e-10;

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
    mfem::Vector y(vfes.GetTrueVSize());
    U.GetTrueDofs(x);
    oper.Mult(x, y);

    const double resid_norm = GlobalL2Norm(y);
    const double resid_max = GlobalMax(MaxAbsEntry(y));
    ctx.AddParity(resid_norm);
    ctx.Check(resid_norm < tol_resid,
              "DGSEM volume mode residual ~0 for constant 1D state (norm=" +
                  std::to_string(resid_norm) +
                  ", max|entry|=" + std::to_string(resid_max) + ")");
  }

  // Integration test: DGSEM volume mode constant-state residual in 2D.
  {
    constexpr int dim = 2;
    constexpr int num_eq = dim + 2;
    constexpr double gamma = 1.4;
    const double tol_resid = 1.0e-10;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                                  /*gen_edges=*/true, 1.0, 1.0);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

    mfem::DG_FECollection fec(/*order=*/2, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_eq,
                                     mfem::Ordering::byNODES);

    mfem::ParGridFunction U(&vfes);
    const mfem::Vector cons_const =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
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
    mfem::Vector y(vfes.GetTrueVSize());
    U.GetTrueDofs(x);
    oper.Mult(x, y);

    const double resid_norm = GlobalL2Norm(y);
    const double resid_max = GlobalMax(MaxAbsEntry(y));
    ctx.AddParity(resid_norm);
    ctx.Check(resid_norm < tol_resid,
              "DGSEM volume mode residual ~0 for constant 2D state (norm=" +
                  std::to_string(resid_norm) +
                  ", max|entry|=" + std::to_string(resid_max) + ")");
  }

  // Integration test: DGSEM constant-state residual on a curved quad mesh.
  {
    constexpr int dim = 2;
    constexpr int num_eq = dim + 2;
    constexpr double gamma = 1.4;
    const double tol_resid = 1.0e-10;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL,
                                                  /*gen_edges=*/true, 1.0, 1.0);
    mesh.SetCurvature(2);
    mfem::GridFunction& nodes = *mesh.GetNodes();
    mfem::FiniteElementSpace* nodes_fes = nodes.FESpace();
    for (int i = 0; i < nodes_fes->GetNDofs(); ++i)
    {
      const int xdof = nodes_fes->DofToVDof(i, 0);
      const int ydof = nodes_fes->DofToVDof(i, 1);
      const double x = nodes(xdof);
      const double y = nodes(ydof);
      const double bump = 0.15 * x * (1.0 - x) * y * (1.0 - y);
      nodes(ydof) = y + bump;
    }

    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

    mfem::DG_FECollection fec(/*order=*/2, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(&pmesh, &fec, num_eq,
                                     mfem::Ordering::byNODES);

    mfem::ParGridFunction U(&vfes);
    const mfem::Vector cons_const =
        EulerConsFromPrim2D(/*rho=*/1.0, /*u=*/0.0, /*v=*/0.0, /*p=*/1.0, gamma);
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
    mfem::Vector y(vfes.GetTrueVSize());
    U.GetTrueDofs(x);
    oper.Mult(x, y);

    const double resid_norm = GlobalL2Norm(y);
    const double resid_max = GlobalMax(MaxAbsEntry(y));
    ctx.AddParity(resid_norm);
    ctx.Check(resid_norm < tol_resid,
              "DGSEM curved-quad residual ~0 for constant state (norm=" +
                  std::to_string(resid_norm) +
                  ", max|entry|=" + std::to_string(resid_max) + ")");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests

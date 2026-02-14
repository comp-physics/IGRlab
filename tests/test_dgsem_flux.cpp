#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_entropy_flux.hpp"
#include "numerics/dgsem/dgsem_flux.hpp"
#include "numerics/dgsem/dgsem_operator.hpp"
#include "numerics/dgsem/dgsem_quadrature.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"
#include "numerics/dgsem/dgsem_volume_flux.hpp"

#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/dgsem_surface_flux.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/igr/igr_flux.hpp"
#include "physics/bc/boundary_state.hpp"
#include "utils/math/constants.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
      if (i == j)
      {
        continue;
      }
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

    mfem::dgsem::SymmetricAverageContravariantFlux(flux,
                                                   local_i,
                                                   local_j,
                                                   dir,
                                                   dim,
                                                   out);
    mfem::dgsem::SymmetricAverageContravariantFlux(flux,
                                                   local_j,
                                                   local_i,
                                                   dir,
                                                   dim,
                                                   out_swap);

    for (int eq = 0; eq < num_eq; ++eq)
    {
      const double expected =
        0.5 * (flux(eq, local_i * dim + dir) + flux(eq, local_j * dim + dir));
      ctx.CheckNear(out(eq),
                    expected,
                    tol,
                    "SymmetricAverageContravariantFlux matches average");
      ctx.CheckNear(out_swap(eq),
                    expected,
                    tol,
                    "SymmetricAverageContravariantFlux is symmetric");
    }
  }

  // LGL face rule sanity.
  {
    const int order = 4;
    const mfem::IntegrationRule& ir = mfem::dgsem::GetLGLFaceRule(order);
    ctx.Check(ir.GetNPoints() == order + 1, "LGL face rule uses p+1 points");
    ctx.CheckNear(ir.IntPoint(0).x, 0.0, tol, "LGL face rule starts at x=0");
    ctx.CheckNear(ir.IntPoint(ir.GetNPoints() - 1).x,
                  1.0,
                  tol,
                  "LGL face rule ends at x=1");
    double weight_sum = 0.0;
    for (int i = 0; i < ir.GetNPoints(); ++i)
    {
      weight_sum += ir.IntPoint(i).weight;
    }
    ctx.CheckNear(weight_sum, 1.0, tol, "LGL face rule weights sum to 1");
    ctx.Check(ir.GetOrder() == 2 * order - 1, "LGL face rule order is 2p-1");
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

    auto check_face = [&](const std::string& label, bool check_i, int value)
    {
      std::vector<int> found(n1d, 0);
      for (const auto& ijk : dof_to_ijk)
      {
        const int i = ijk[0];
        const int j = ijk[1];
        if ((check_i ? i : j) != value)
        {
          continue;
        }
        const int tangential = check_i ? j : i;
        found[tangential] = 1;
      }
      for (int n = 0; n < n1d; ++n)
      {
        ctx.Check(found[n] == 1, "Face contains all LGL nodes (" + label + ")");
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

    auto central_flux =
      [](const mfem::Vector& ul, const mfem::Vector& ur, int, mfem::Vector& f)
    {
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
        ctx.CheckNear(fab(i),
                      fba(i),
                      tol,
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

      auto linear_flux =
        [](const mfem::Vector& ul, const mfem::Vector& ur, int, mfem::Vector& f)
      {
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

      mfem::dgsem::FluxDifferencing2D(u,
                                      basis.Dsplit,
                                      dof_to_ijk,
                                      central_flux,
                                      du);
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
      mfem::Mesh mesh =
        mfem::Mesh::MakeCartesian2D(1,
                                    1,
                                    mfem::Element::QUADRILATERAL,
                                    /*gen_edges=*/true,
                                    1.0,
                                    1.0);
      mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
      mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLobatto);
      mfem::ParFiniteElementSpace fes(&pmesh,
                                      &fec,
                                      num_eq,
                                      mfem::Ordering::byNODES);

      const mfem::FiniteElement* fe = fes.GetFE(0);
      const int dof = fe->GetDof();
      const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(*fe);

      mfem::DenseMatrix u = BuildConstantStateMatrix(dof, state);
      mfem::DenseMatrix du;
      mfem::dgsem::FluxDifferencing2D(u,
                                      basis.Dsplit,
                                      dof_to_ijk,
                                      central_flux,
                                      du);

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
    mfem::ParFiniteElementSpace vfes(&pmesh,
                                     &fec,
                                     num_eq,
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
    mfem::DGHyperbolicConservationLaws oper(vfes,
                                            std::move(form_integrator),
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
        new mfem::BdrHyperbolicFluxIntegrator(num_flux,
                                              std::move(outlet),
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
                FormatSci(resid_norm) + ", max|entry|=" + FormatSci(resid_max) +
                ")");
  }

  // Volume dissipation modes are runtime-selectable and active in DGSEM volume kernel.
  {
    constexpr int dim = 1;
    constexpr int num_eq = dim + 2;
    constexpr double gamma = 1.4;
    constexpr double kPi = dg_sim::math::kPi;
    const double tol_zero = 1.0e-11;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(1, 1.0);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mfem::DG_FECollection fec(/*order=*/3, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(&pmesh,
                                     &fec,
                                     num_eq,
                                     mfem::Ordering::byNODES);

    mfem::EulerFlux flux(dim, gamma);
    mfem::RusanovFluxWithSides num_flux(flux);
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                       /*int_order_offset=*/2);
    mfem::DGHyperbolicConservationLaws oper(vfes,
                                            std::move(form_integrator),
                                            /*preassembleWeakDivergence=*/true);
    oper.SetNumericalFlux(&num_flux);
    oper.SetVolumeMode(mfem::VolumeMode::DGSEMFluxDiff);
    oper.SetDGSEMVolumeFlux(mfem::DGSEMVolumeFlux::Central);
    oper.SetDGSEMGamma(gamma);
    oper.SetDGSEMEntropyJacobianAnalytic(false);

    mfem::Vector x(vfes.GetTrueVSize());
    x = 0.0;
    {
      const mfem::FiniteElement* fe = vfes.GetFE(0);
      const int dof = fe->GetDof();
      const mfem::IntegrationRule& nodes = fe->GetNodes();
      mfem::Array<int> vdofs;
      vfes.GetElementVDofs(0, vdofs);
      mfem::Vector xelem(dof * num_eq);
      mfem::DenseMatrix Xmat(xelem.GetData(), dof, num_eq);
      for (int j = 0; j < dof; ++j)
      {
        const double xi = nodes.IntPoint(j).x;
        const double rho = 1.0 + 0.20 * std::sin(2.0 * kPi * xi);
        const double u = 0.25 + 0.15 * std::cos(2.0 * kPi * xi);
        const double p = 1.0 + 0.10 * std::sin(kPi * xi);
        const mfem::Vector cons = EulerConsFromPrim1D(rho, u, p, gamma);
        for (int eq = 0; eq < num_eq; ++eq)
        {
          Xmat(j, eq) = cons(eq);
        }
      }
      x.SetSubVector(vdofs, xelem);
    }

    auto assemble_volume =
      [&](mfem::DGSEMVolumeDissipation mode, mfem::Vector& z)
    {
      oper.SetDGSEMVolumeDissipation(mode);
      z.SetSize(vfes.GetTrueVSize());
      z = 0.0;
      mfem::dgsem::DGSEMOperator dgsem_op(oper);
      dgsem_op.AssembleVolume(x, z);
    };

    mfem::Vector z_none, z_llf, z_entropy;
    assemble_volume(mfem::DGSEMVolumeDissipation::None, z_none);
    assemble_volume(mfem::DGSEMVolumeDissipation::LLF, z_llf);
    assemble_volume(mfem::DGSEMVolumeDissipation::EntropyVarsLF, z_entropy);

    const double norm_none = GlobalL2Norm(z_none);
    const double norm_llf = GlobalL2Norm(z_llf);
    const double norm_entropy = GlobalL2Norm(z_entropy);
    ctx.Check(
      norm_none > 0.0,
      "DGSEM volume (no dissipation) is non-trivial for non-constant state");
    ctx.Check(
      norm_llf > 0.0,
      "DGSEM LLF volume dissipation mode produces non-trivial contribution");
    ctx.Check(norm_entropy > 0.0,
              "DGSEM entropy-LF volume dissipation mode produces non-trivial "
              "contribution");

    mfem::Vector diff_llf(z_llf);
    diff_llf -= z_none;
    mfem::Vector diff_entropy(z_entropy);
    diff_entropy -= z_none;
    mfem::Vector diff_between(z_entropy);
    diff_between -= z_llf;

    ctx.Check(GlobalL2Norm(diff_llf) > 1.0e-12,
              "DGSEM LLF volume dissipation changes the volume residual");
    ctx.Check(
      GlobalL2Norm(diff_entropy) > 1.0e-12,
      "DGSEM entropy-LF volume dissipation changes the volume residual");
    ctx.Check(GlobalL2Norm(diff_between) > 1.0e-12,
              "DGSEM LLF and entropy-LF volume dissipation are distinct");

    // Constant-state invariance with dissipation active.
    const mfem::Vector cons_const =
      EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/0.15, /*p=*/1.0, gamma);
    mfem::ParGridFunction Uconst(&vfes);
    FillConstantState(cons_const, Uconst);
    Uconst.GetTrueDofs(x);
    assemble_volume(mfem::DGSEMVolumeDissipation::None, z_none);
    assemble_volume(mfem::DGSEMVolumeDissipation::LLF, z_llf);
    assemble_volume(mfem::DGSEMVolumeDissipation::EntropyVarsLF, z_entropy);
    diff_llf = z_llf;
    diff_llf -= z_none;
    diff_entropy = z_entropy;
    diff_entropy -= z_none;
    ctx.Check(GlobalL2Norm(diff_llf) < tol_zero,
              "DGSEM LLF volume dissipation leaves constant-state volume term "
              "unchanged");
    ctx.Check(GlobalL2Norm(diff_entropy) < tol_zero,
              "DGSEM entropy-LF volume dissipation leaves constant-state "
              "volume term unchanged");
  }

  // Integration test: DGSEM volume mode constant-state residual in 2D.
  {
    constexpr int dim = 2;
    constexpr int num_eq = dim + 2;
    constexpr double gamma = 1.4;
    const double tol_resid = 1.0e-10;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1,
                                                  1,
                                                  mfem::Element::QUADRILATERAL,
                                                  /*gen_edges=*/true,
                                                  1.0,
                                                  1.0);
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

    mfem::EulerFlux flux(dim, gamma);
    mfem::RusanovFluxWithSides num_flux(flux);

    constexpr int int_order_offset = 2;
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                       int_order_offset);
    mfem::DGHyperbolicConservationLaws oper(vfes,
                                            std::move(form_integrator),
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
        new mfem::BdrHyperbolicFluxIntegrator(num_flux,
                                              std::move(outlet),
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
                FormatSci(resid_norm) + ", max|entry|=" + FormatSci(resid_max) +
                ")");
  }

  // Integration test: DGSEM constant-state residual on a curved quad mesh.
  {
    constexpr int dim = 2;
    constexpr int num_eq = dim + 2;
    constexpr double gamma = 1.4;
    const double tol_resid = 1.0e-10;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1,
                                                  1,
                                                  mfem::Element::QUADRILATERAL,
                                                  /*gen_edges=*/true,
                                                  1.0,
                                                  1.0);
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
    mfem::ParFiniteElementSpace vfes(&pmesh,
                                     &fec,
                                     num_eq,
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
    mfem::DGHyperbolicConservationLaws oper(vfes,
                                            std::move(form_integrator),
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
        new mfem::BdrHyperbolicFluxIntegrator(num_flux,
                                              std::move(outlet),
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
                FormatSci(resid_norm) + ", max|entry|=" + FormatSci(resid_max) +
                ")");
  }

  // Discrete metric/GCL sanity: stronger deformation on a quad mesh.
  {
    constexpr int dim = 2;
    constexpr int num_eq = dim + 2;
    constexpr double gamma = 1.4;
    constexpr double kPi = dg_sim::math::kPi;
    const double tol_resid = 1.0e-10;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1,
                                                  1,
                                                  mfem::Element::QUADRILATERAL,
                                                  /*gen_edges=*/true,
                                                  1.0,
                                                  1.0);
    mesh.SetCurvature(4);
    mfem::GridFunction& nodes = *mesh.GetNodes();
    mfem::FiniteElementSpace* nodes_fes = nodes.FESpace();
    for (int i = 0; i < nodes_fes->GetNDofs(); ++i)
    {
      const int xdof = nodes_fes->DofToVDof(i, 0);
      const int ydof = nodes_fes->DofToVDof(i, 1);
      const double x = nodes(xdof);
      const double y = nodes(ydof);
      const double warp = std::sin(kPi * x) * std::sin(kPi * y);
      nodes(xdof) = x + 0.2 * warp;
      nodes(ydof) = y - 0.15 * warp;
    }

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

    mfem::EulerFlux flux(dim, gamma);
    mfem::RusanovFluxWithSides num_flux(flux);

    constexpr int int_order_offset = 2;
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                       int_order_offset);
    mfem::DGHyperbolicConservationLaws oper(vfes,
                                            std::move(form_integrator),
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
        new mfem::BdrHyperbolicFluxIntegrator(num_flux,
                                              std::move(outlet),
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
              "DGSEM strongly-deformed quad residual ~0 (norm=" +
                FormatSci(resid_norm) + ", max|entry|=" + FormatSci(resid_max) +
                ")");
  }

  // IGR-aware DGSEM volume flux responds to varying sigma.
  {
    constexpr int dim = 1;
    constexpr int num_eq = dim + 2;
    constexpr double gamma = 1.4;
    const double tol_zero = 1.0e-10;

    mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(2, 1.0);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, mesh);

    mfem::DG_FECollection fec(/*order=*/2, dim, mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace vfes(&pmesh,
                                     &fec,
                                     num_eq,
                                     mfem::Ordering::byNODES);

    mfem::ParGridFunction U(&vfes);
    const mfem::Vector cons_const =
      EulerConsFromPrim1D(/*rho=*/1.0, /*u=*/0.0, /*p=*/1.0, gamma);
    FillConstantState(cons_const, U);

    mfem::DG_FECollection sigma_fec(/*order=*/1,
                                    dim,
                                    mfem::BasisType::GaussLobatto);
    mfem::ParFiniteElementSpace sigma_fes(&pmesh,
                                          &sigma_fec,
                                          /*vdim=*/1,
                                          mfem::Ordering::byNODES);
    mfem::ParGridFunction sigma(&sigma_fes);

    class SigmaLinearCoefficient final : public mfem::Coefficient
    {
    public:
      double Eval(mfem::ElementTransformation& T,
                  const mfem::IntegrationPoint& ip) override
      {
        mfem::Vector x(1);
        T.Transform(ip, x);
        return 0.2 + 0.3 * x(0);
      }
    };

    SigmaLinearCoefficient sigma_coeff;
    sigma.ProjectCoefficient(sigma_coeff);
    sigma.ExchangeFaceNbrData();

    mfem::EulerFlux base_flux(dim, gamma);
    mfem::FluxWithIGR igr_flux(base_flux);
    igr_flux.SetSigmaField(&sigma);
    mfem::RusanovFluxWithSides num_flux(igr_flux);

    constexpr int int_order_offset = 2;
    auto form_integrator =
      std::make_unique<mfem::HyperbolicFormIntegrator>(num_flux,
                                                       int_order_offset);
    mfem::DGHyperbolicConservationLaws oper(vfes,
                                            std::move(form_integrator),
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
        new mfem::BdrHyperbolicFluxIntegrator(num_flux,
                                              std::move(outlet),
                                              int_order_offset),
        bdr_marker);
    }

    mfem::Vector x;
    mfem::Vector y(vfes.GetTrueVSize());
    U.GetTrueDofs(x);
    oper.Mult(x, y);
    const double resid_sigma = GlobalL2Norm(y);

    mfem::ConstantCoefficient sigma_zero(0.0);
    sigma.ProjectCoefficient(sigma_zero);
    sigma.ExchangeFaceNbrData();
    oper.Mult(x, y);
    const double resid_zero = GlobalL2Norm(y);

    ctx.Check(resid_zero < tol_zero, "DGSEM IGR residual ~0 when sigma = 0");
    ctx.Check(resid_sigma > 1.0e-8,
              "DGSEM IGR residual responds to varying sigma");
  }

  // Entropy-conservative flux helpers (Chandrashekar/Ranocha) symmetry + consistency.
  {
    const double gamma = 1.4;
    const mfem::Vector ul1 = EulerConsFromPrim1D(1.2, 0.3, 1.1, gamma);
    const mfem::Vector ur1 = EulerConsFromPrim1D(0.9, -0.1, 0.7, gamma);
    const mfem::Vector ul2 = EulerConsFromPrim2D(1.0, 0.2, -0.3, 1.0, gamma);
    const mfem::Vector ur2 = EulerConsFromPrim2D(0.8, -0.1, 0.4, 0.9, gamma);

    auto physical_flux_1d = [&](const mfem::Vector& u, mfem::Vector& f)
    {
      const double rho = u(0);
      const double m = u(1);
      const double E = u(2);
      const double v = m / rho;
      const double p = (gamma - 1.0) * (E - 0.5 * rho * v * v);
      f.SetSize(3);
      f(0) = rho * v;
      f(1) = rho * v * v + p;
      f(2) = (E + p) * v;
    };

    auto physical_flux_2d = [&](const mfem::Vector& u, int dir, mfem::Vector& f)
    {
      const double rho = u(0);
      const double m1 = u(1);
      const double m2 = u(2);
      const double E = u(3);
      const double v1 = m1 / rho;
      const double v2 = m2 / rho;
      const double p = (gamma - 1.0) * (E - 0.5 * rho * (v1 * v1 + v2 * v2));
      f.SetSize(4);
      if (dir == 0)
      {
        f(0) = rho * v1;
        f(1) = rho * v1 * v1 + p;
        f(2) = rho * v1 * v2;
        f(3) = (E + p) * v1;
      }
      else
      {
        f(0) = rho * v2;
        f(1) = rho * v1 * v2;
        f(2) = rho * v2 * v2 + p;
        f(3) = (E + p) * v2;
      }
    };

    for (auto type : {mfem::dgsem::EntropyFluxType::Chandrashekar,
                      mfem::dgsem::EntropyFluxType::Ranocha})
    {
      mfem::Vector f_lr, f_rl;
      mfem::dgsem::EntropyConservativeFlux(ul1, ur1, 0, gamma, type, f_lr);
      mfem::dgsem::EntropyConservativeFlux(ur1, ul1, 0, gamma, type, f_rl);
      for (int i = 0; i < f_lr.Size(); ++i)
      {
        ctx.CheckNear(f_lr(i), f_rl(i), tol, "EC flux is symmetric in 1D");
      }

      mfem::Vector f_cons, f_phys;
      mfem::dgsem::EntropyConservativeFlux(ul1, ul1, 0, gamma, type, f_cons);
      physical_flux_1d(ul1, f_phys);
      for (int i = 0; i < f_cons.Size(); ++i)
      {
        ctx.CheckNear(f_cons(i),
                      f_phys(i),
                      1.0e-10,
                      "EC flux matches physical flux in 1D");
      }

      mfem::Vector f2_lr, f2_rl;
      mfem::dgsem::EntropyConservativeFlux(ul2, ur2, 0, gamma, type, f2_lr);
      mfem::dgsem::EntropyConservativeFlux(ur2, ul2, 0, gamma, type, f2_rl);
      for (int i = 0; i < f2_lr.Size(); ++i)
      {
        ctx.CheckNear(f2_lr(i),
                      f2_rl(i),
                      tol,
                      "EC flux is symmetric in 2D (dir=0)");
      }

      mfem::Vector f2_cons, f2_phys;
      mfem::dgsem::EntropyConservativeFlux(ul2, ul2, 1, gamma, type, f2_cons);
      physical_flux_2d(ul2, 1, f2_phys);
      for (int i = 0; i < f2_cons.Size(); ++i)
      {
        ctx.CheckNear(f2_cons(i),
                      f2_phys(i),
                      1.0e-10,
                      "EC flux matches physical flux in 2D (dir=1)");
      }
    }
  }

  // Flux differencing with EC flux: constant state gives row-sum scaling.
  {
    const double gamma = 1.4;
    const int order = 3;
    const auto& basis = mfem::dgsem::GetLGLBasis1D(order);
    const int n1d = basis.n;
    const std::vector<double> row_sum = OffDiagonalRowSum(basis.Dsplit);

    const mfem::Vector state1d = EulerConsFromPrim1D(1.0, 0.2, 1.0, gamma);

    auto ec_flux = [&](const mfem::Vector& ul,
                       const mfem::Vector& ur,
                       int dir,
                       mfem::Vector& f)
    {
      mfem::dgsem::EntropyConservativeFlux(
        ul,
        ur,
        dir,
        gamma,
        mfem::dgsem::EntropyFluxType::Ranocha,
        f);
    };

    mfem::Vector f_u;
    mfem::dgsem::EntropyConservativeFlux(state1d,
                                         state1d,
                                         0,
                                         gamma,
                                         mfem::dgsem::EntropyFluxType::Ranocha,
                                         f_u);
    mfem::DenseMatrix u = BuildConstantStateMatrix(n1d, state1d);
    mfem::DenseMatrix du;
    mfem::dgsem::FluxDifferencing1D(u, basis.Dsplit, ec_flux, du);
    mfem::DenseMatrix expected(n1d, f_u.Size());
    for (int i = 0; i < n1d; ++i)
    {
      for (int eq = 0; eq < f_u.Size(); ++eq)
      {
        expected(i, eq) = row_sum[i] * f_u(eq);
      }
    }
    expected -= du;
    ctx.Check(MaxAbsEntry(expected) < 1.0e-10,
              "FluxDifferencing1D constant state with EC flux");

    const mfem::Vector state2d =
      EulerConsFromPrim2D(1.0, -0.1, 0.2, 1.0, gamma);
    mfem::DenseMatrix u2d(n1d * n1d, state2d.Size());
    for (int i = 0; i < n1d * n1d; ++i)
    {
      for (int eq = 0; eq < state2d.Size(); ++eq)
      {
        u2d(i, eq) = state2d(eq);
      }
    }
    mfem::DG_FECollection fec(order, 2, mfem::BasisType::GaussLobatto);
    const mfem::FiniteElement* fe =
      fec.FiniteElementForGeometry(mfem::Geometry::SQUARE);
    const auto& dof_to_ijk = mfem::dgsem::GetTensorDofToIJK(*fe);
    mfem::DenseMatrix du2d;
    mfem::dgsem::FluxDifferencing2D(u2d,
                                    basis.Dsplit,
                                    dof_to_ijk,
                                    ec_flux,
                                    du2d);

    mfem::Vector fx, fy;
    mfem::dgsem::EntropyConservativeFlux(state2d,
                                         state2d,
                                         0,
                                         gamma,
                                         mfem::dgsem::EntropyFluxType::Ranocha,
                                         fx);
    mfem::dgsem::EntropyConservativeFlux(state2d,
                                         state2d,
                                         1,
                                         gamma,
                                         mfem::dgsem::EntropyFluxType::Ranocha,
                                         fy);
    mfem::DenseMatrix expected2d(n1d * n1d, state2d.Size());
    for (int local = 0; local < static_cast<int>(dof_to_ijk.size()); ++local)
    {
      const int i = dof_to_ijk[local][0];
      const int j = dof_to_ijk[local][1];
      for (int eq = 0; eq < state2d.Size(); ++eq)
      {
        expected2d(local, eq) = row_sum[i] * fx(eq) + row_sum[j] * fy(eq);
      }
    }
    expected2d -= du2d;
    ctx.Check(MaxAbsEntry(expected2d) < 1.0e-10,
              "FluxDifferencing2D constant state with EC flux");
  }

  // Entropy-stable surface flux (Winters): constant state reduces to physical flux.
  {
    const double gamma = 1.4;
    mfem::EulerFlux flux(2, gamma);
    mfem::DGSEMSurfaceFlux es_flux(flux);
    es_flux.SetSurfaceFluxType(
      mfem::DGSEMSurfaceFluxType::EntropyStableWinters);
    es_flux.SetEntropyFluxType(mfem::dgsem::EntropyFluxType::Ranocha);
    es_flux.SetGamma(gamma);

    const mfem::Vector u = EulerConsFromPrim2D(1.0, 0.3, -0.2, 1.0, gamma);
    mfem::Vector normal(2);
    normal(0) = 1.2;
    normal(1) = -0.7;
    mfem::FaceElementTransformations Tr;

    mfem::Vector flux_es(u.Size());
    es_flux.Eval(u, u, normal, Tr, flux_es);

    mfem::Vector fx, fy;
    auto physical_flux_2d =
      [&](const mfem::Vector& state, int dir, mfem::Vector& f)
    {
      const double rho = state(0);
      const double m1 = state(1);
      const double m2 = state(2);
      const double E = state(3);
      const double v1 = m1 / rho;
      const double v2 = m2 / rho;
      const double p = (gamma - 1.0) * (E - 0.5 * rho * (v1 * v1 + v2 * v2));
      f.SetSize(4);
      if (dir == 0)
      {
        f(0) = rho * v1;
        f(1) = rho * v1 * v1 + p;
        f(2) = rho * v1 * v2;
        f(3) = (E + p) * v1;
      }
      else
      {
        f(0) = rho * v2;
        f(1) = rho * v1 * v2;
        f(2) = rho * v2 * v2 + p;
        f(3) = (E + p) * v2;
      }
    };
    physical_flux_2d(u, 0, fx);
    physical_flux_2d(u, 1, fy);

    for (int eq = 0; eq < u.Size(); ++eq)
    {
      const double expected = normal(0) * fx(eq) + normal(1) * fy(eq);
      ctx.CheckNear(flux_es(eq),
                    expected,
                    1.0e-10,
                    "Entropy-stable surface flux matches physical flux");
    }

    mfem::Vector diss;
    mfem::dgsem::WintersDissipation(u, u, normal, gamma, diss);
    ctx.Check(MaxAbsEntry(diss) < 1.0e-12,
              "Winters dissipation is zero for identical states");
  }

  // Entropy-stable LF entropy-variable surface flux: constant state sanity.
  {
    const double gamma = 1.4;
    mfem::EulerFlux flux(2, gamma);
    mfem::DGSEMSurfaceFlux es_flux(flux);
    es_flux.SetSurfaceFluxType(
      mfem::DGSEMSurfaceFluxType::EntropyStableLFEntropyVars);
    es_flux.SetEntropyFluxType(mfem::dgsem::EntropyFluxType::Ranocha);
    es_flux.SetGamma(gamma);

    const mfem::Vector u = EulerConsFromPrim2D(1.1, 0.25, -0.15, 1.2, gamma);
    mfem::Vector normal(2);
    normal(0) = -0.9;
    normal(1) = 0.6;
    mfem::FaceElementTransformations Tr;

    mfem::Vector flux_es(u.Size());
    es_flux.Eval(u, u, normal, Tr, flux_es);

    mfem::Vector fx, fy;
    auto physical_flux_2d =
      [&](const mfem::Vector& state, int dir, mfem::Vector& f)
    {
      const double rho = state(0);
      const double m1 = state(1);
      const double m2 = state(2);
      const double E = state(3);
      const double v1 = m1 / rho;
      const double v2 = m2 / rho;
      const double p = (gamma - 1.0) * (E - 0.5 * rho * (v1 * v1 + v2 * v2));
      f.SetSize(4);
      if (dir == 0)
      {
        f(0) = rho * v1;
        f(1) = rho * v1 * v1 + p;
        f(2) = rho * v1 * v2;
        f(3) = (E + p) * v1;
      }
      else
      {
        f(0) = rho * v2;
        f(1) = rho * v1 * v2;
        f(2) = rho * v2 * v2 + p;
        f(3) = (E + p) * v2;
      }
    };
    physical_flux_2d(u, 0, fx);
    physical_flux_2d(u, 1, fy);

    for (int eq = 0; eq < u.Size(); ++eq)
    {
      const double expected = normal(0) * fx(eq) + normal(1) * fy(eq);
      ctx.CheckNear(flux_es(eq),
                    expected,
                    5.0e-8,
                    "Entropy-variable LF surface flux matches physical flux");
    }

    mfem::Vector diss;
    mfem::dgsem::EntropyVariableLFDissipation(u,
                                              u,
                                              gamma,
                                              /*scale=*/1.0,
                                              diss);
    ctx.Check(MaxAbsEntry(diss) < 1.0e-12,
              "Entropy-variable LF dissipation is zero for identical states");
  }

  // Entropy production regression (smooth state, ES fluxes).
  {
    const double gamma = 1.4;
    const mfem::Vector uL = EulerConsFromPrim2D(1.0, 0.2, -0.1, 1.0, gamma);
    const mfem::Vector uR = EulerConsFromPrim2D(1.01, 0.22, -0.08, 1.02, gamma);
    mfem::Vector normal(2);
    normal(0) = 0.6;
    normal(1) = -0.8;

    auto entropy_prod = [&](mfem::DGSEMSurfaceFluxType surface_type,
                            mfem::dgsem::EntropyJacobianMode mode)
    {
      mfem::EulerFlux flux(2, gamma);
      mfem::DGSEMSurfaceFlux es_flux(flux);
      es_flux.SetSurfaceFluxType(surface_type);
      es_flux.SetEntropyFluxType(mfem::dgsem::EntropyFluxType::Ranocha);
      es_flux.SetEntropyJacobianMode(mode);
      es_flux.SetGamma(gamma);

      mfem::FaceElementTransformations Tr;
      mfem::Vector flux_es(uL.Size());
      es_flux.Eval(uL, uR, normal, Tr, flux_es);

      mfem::Vector flux_ec(uL.Size());
      mfem::dgsem::EntropyConservativeFluxDotN(
        uL,
        uR,
        normal,
        gamma,
        mfem::dgsem::EntropyFluxType::Ranocha,
        flux_ec);

      mfem::Vector diss(uL.Size());
      diss = flux_es;
      diss -= flux_ec;

      mfem::Vector wL, wR;
      mfem::dgsem::EulerEntropyVariables(uL,
                                         /*dim=*/2,
                                         gamma,
                                         /*sigma=*/0.0,
                                         wL);
      mfem::dgsem::EulerEntropyVariables(uR,
                                         /*dim=*/2,
                                         gamma,
                                         /*sigma=*/0.0,
                                         wR);

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

    const double expected_winters = -2.46898e-4;
    const double expected_lf = -7.91901e-4;
    const double tol_prod = 1.0e-2;

    if (std::getenv("DG_IGR_PRINT_ENTROPY"))
    {
      std::cout << "entropy_prod_winters=" << prod_winters << "\n";
      std::cout << "entropy_prod_lf_fd=" << prod_lf_fd << "\n";
      std::cout << "entropy_prod_lf_analytic=" << prod_lf_analytic << "\n";
    }

    ctx.Check(prod_winters < 0.0,
              "Entropy production (Winters) is non-positive");
    ctx.Check(prod_lf_fd < 0.0,
              "Entropy production (LF entropy vars, FD) is non-positive");
    ctx.Check(prod_lf_analytic < 0.0,
              "Entropy production (LF entropy vars, analytic) is non-positive");

    ctx.CheckNear(prod_winters,
                  expected_winters,
                  tol_prod,
                  "Entropy production (Winters) regression");
    ctx.CheckNear(prod_lf_fd,
                  expected_lf,
                  tol_prod,
                  "Entropy production (LF entropy vars, FD) regression");
    ctx.CheckNear(prod_lf_analytic,
                  expected_lf,
                  tol_prod,
                  "Entropy production (LF entropy vars, analytic) regression");

    ctx.CheckNear(prod_lf_fd,
                  prod_lf_analytic,
                  1.0e-2,
                  "LF entropy production FD vs analytic close");
  }

  return ctx.failures;
}

} // namespace dg_igr_tests

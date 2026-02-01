#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"

#include "mfem.hpp"
#include "test_utils.hpp"

#include <cmath>
#include <string>

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

} // namespace

int RunDGSEMBasisTests(TestContext& ctx)
{
  const double tol = 1.0e-12;
  const int orders[] = {1, 2, 3, 4};

  for (int order : orders)
  {
    const auto& basis = mfem::dgsem::GetLGLBasis1D(order);

    double weight_sum = 0.0;
    for (int i = 0; i < basis.n; ++i)
    {
      weight_sum += basis.weights(i);
    }
    ctx.CheckNear(weight_sum, 1.0, tol,
                  "LGL weights sum to 1 on [0,1] (order=" + std::to_string(order) + ")");

    // SBP identity: M D + D^T M = B
    mfem::DenseMatrix M(basis.n);
    M = 0.0;
    for (int i = 0; i < basis.n; ++i)
    {
      M(i, i) = basis.weights(i);
    }

    mfem::DenseMatrix MD(basis.n);
    mfem::DenseMatrix DtM(basis.n);
    mfem::Mult(M, basis.D, MD);
    mfem::DenseMatrix Dt;
    Dt.Transpose(basis.D);
    mfem::Mult(Dt, M, DtM);

    mfem::DenseMatrix sbp(basis.n);
    sbp = 0.0;
    sbp += MD;
    sbp += DtM;

    mfem::DenseMatrix B(basis.n);
    B = 0.0;
    if (basis.n > 0)
    {
      B(0, 0) = -1.0;
      B(basis.n - 1, basis.n - 1) = 1.0;
    }

    sbp -= B;
    const double sbp_err = MaxAbsEntry(sbp);
    ctx.Check(sbp_err < 1.0e-10,
              "SBP identity holds (order=" + std::to_string(order) + ")");

    // Dsplit consistency: 2 D - M^{-1} B
    mfem::DenseMatrix Dsplit_expected(basis.n);
    Dsplit_expected = basis.D;
    for (int i = 0; i < basis.n; ++i)
    {
      for (int j = 0; j < basis.n; ++j)
      {
        Dsplit_expected(i, j) *= 2.0;
      }
    }
    for (int i = 0; i < basis.n; ++i)
    {
      const double b = (i == 0) ? -1.0 : (i == basis.n - 1 ? 1.0 : 0.0);
      Dsplit_expected(i, i) -= b * basis.inv_weights(i);
    }
    Dsplit_expected -= basis.Dsplit;
    const double dsplit_err = MaxAbsEntry(Dsplit_expected);
    ctx.Check(dsplit_err < 1.0e-10,
              "Dsplit matches D - 0.5 M^{-1} B (order=" + std::to_string(order) + ")");
  }

  // Lexicographic mapping sanity (1D + 2D, Gauss-Lobatto).
  {
    const int order = 3;
    const mfem::real_t* nodes = mfem::poly1d.GetPoints(order, mfem::BasisType::GaussLobatto);

    mfem::DG_FECollection fec_1d(order, 1, mfem::BasisType::GaussLobatto);
    const mfem::FiniteElement* fe_1d = fec_1d.FiniteElementForGeometry(mfem::Geometry::SEGMENT);
    const auto& map_1d = mfem::dgsem::GetTensorDofToIJK(*fe_1d);
    for (int ld = 0; ld < fe_1d->GetDof(); ++ld)
    {
      const int i = map_1d[ld][0];
      const double x = fe_1d->GetNodes().IntPoint(ld).x;
      ctx.CheckNear(x, nodes[i], 1.0e-12, "1D lexicographic mapping matches nodes");
    }

    mfem::DG_FECollection fec_2d(order, 2, mfem::BasisType::GaussLobatto);
    const mfem::FiniteElement* fe_2d = fec_2d.FiniteElementForGeometry(mfem::Geometry::SQUARE);
    const auto& map_2d = mfem::dgsem::GetTensorDofToIJK(*fe_2d);
    for (int ld = 0; ld < fe_2d->GetDof(); ++ld)
    {
      const int i = map_2d[ld][0];
      const int j = map_2d[ld][1];
      const mfem::IntegrationPoint& ip = fe_2d->GetNodes().IntPoint(ld);
      ctx.CheckNear(ip.x, nodes[i], 1.0e-12, "2D lexicographic x matches nodes");
      ctx.CheckNear(ip.y, nodes[j], 1.0e-12, "2D lexicographic y matches nodes");
    }
  }

  return ctx.failures;
}

} // namespace dg_igr_tests

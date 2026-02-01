#include "numerics/dgsem/dgsem_basis.hpp"

#include <map>

namespace mfem
{
namespace dgsem
{

LGLBasis1D BuildLGLBasis1D(int order)
{
  MFEM_VERIFY(order >= 0, "LGLBasis1D requires order >= 0.");

  LGLBasis1D basis;
  basis.order = order;
  basis.n = order + 1;

  // Nodes + weights from MFEM Gauss-Lobatto quadrature on [0,1].
  mfem::IntegrationRule ir;
  mfem::QuadratureFunctions1D qf;
  qf.GaussLobatto(basis.n, &ir);

  basis.nodes.SetSize(basis.n);
  basis.weights.SetSize(basis.n);
  basis.inv_weights.SetSize(basis.n);
  for (int i = 0; i < basis.n; ++i)
  {
    const mfem::IntegrationPoint& ip = ir.IntPoint(i);
    basis.nodes(i) = ip.x;
    basis.weights(i) = ip.weight;
    basis.inv_weights(i) = 1.0 / ip.weight;
  }

  // Strong derivative matrix D_ij = l_j'(x_i) using MFEM's nodal basis.
  basis.D.SetSize(basis.n);
  mfem::Vector u(basis.n);
  mfem::Vector d(basis.n);
  mfem::Poly_1D::Basis& poly_basis = mfem::poly1d.GetBasis(order,
                                                          mfem::BasisType::GaussLobatto);
  for (int i = 0; i < basis.n; ++i)
  {
    poly_basis.Eval(basis.nodes(i), u, d);
    for (int j = 0; j < basis.n; ++j)
    {
      basis.D(i, j) = d(j);
    }
  }

  // Build Dhat = -M^{-1} D^T M and Dsplit = D - 0.5 M^{-1} B.
  basis.Dhat.SetSize(basis.n);
  basis.Dsplit.SetSize(basis.n);
  mfem::Vector bvec(basis.n);
  bvec = 0.0;
  if (basis.n > 0)
  {
    bvec(0) = -1.0;
    bvec(basis.n - 1) = 1.0;
  }

  for (int i = 0; i < basis.n; ++i)
  {
    for (int j = 0; j < basis.n; ++j)
    {
      basis.Dhat(i, j) = -(basis.weights(j) / basis.weights(i)) * basis.D(j, i);
      basis.Dsplit(i, j) = basis.D(i, j);
    }
  }
  for (int i = 0; i < basis.n; ++i)
  {
    basis.Dsplit(i, i) -= 0.5 * bvec(i) * basis.inv_weights(i);
  }

  return basis;
}

const LGLBasis1D& GetLGLBasis1D(int order)
{
  static std::map<int, LGLBasis1D> cache;
  auto it = cache.find(order);
  if (it != cache.end())
  {
    return it->second;
  }
  auto inserted = cache.emplace(order, BuildLGLBasis1D(order));
  return inserted.first->second;
}

} // namespace dgsem
} // namespace mfem

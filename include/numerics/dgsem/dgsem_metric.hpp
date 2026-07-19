#pragma once

/**
 * @file numerics/dgsem/dgsem_metric.hpp
 * @brief Discrete metric identities for two-dimensional tensor DGSEM.
 * @ingroup numerics_dgsem
 */

#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_tensor.hpp"

#include "mfem.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace mfem
{
namespace dgsem
{

/** POD storage for one nodal two-dimensional contravariant metric.
 *
 * The first index is the reference direction and the second is the physical
 * direction. Thus `data[2*reference + physical]` is one entry of adj(J).
 */
struct ContravariantMetric2D
{
  std::array<double, 4> data{{0.0, 0.0, 0.0, 0.0}};

  double operator()(int reference, int physical) const
  {
    return data[static_cast<std::size_t>(2 * reference + physical)];
  }
};

/** Build curl-form nodal metrics from the LGL interpolation of the geometry.
 *
 * For x=(x,y), the rows are
 *
 *   Ja^1 = (D_eta y, -D_eta x),
 *   Ja^2 = (-D_xi y, D_xi x).
 *
 * Because both mixed derivatives use the same tensor LGL derivative matrix,
 * their discrete divergence vanishes to roundoff. Endpoint rows also provide
 * the scaled face normals used by the surface operator.
 */
inline void BuildContravariantMetrics2D(
  const FiniteElement& fe,
  ElementTransformation& transformation,
  std::vector<ContravariantMetric2D>& metrics)
{
  MFEM_VERIFY(fe.GetDim() == 2,
              "BuildContravariantMetrics2D requires a 2D element.");
  const int dof = fe.GetDof();
  const int order = fe.GetOrder();
  const LGLBasis1D& basis = GetLGLBasis1D(order);
  const int n1d = basis.n;
  MFEM_VERIFY(dof == n1d * n1d,
              "Discrete DGSEM metrics require a tensor element.");

  const auto& dof_to_ijk = GetTensorDofToIJK(fe);
  std::vector<int> lex_to_local(static_cast<std::size_t>(dof), -1);
  for (int local = 0; local < dof; ++local)
  {
    const int i = dof_to_ijk[static_cast<std::size_t>(local)][0];
    const int j = dof_to_ijk[static_cast<std::size_t>(local)][1];
    lex_to_local[static_cast<std::size_t>(i + n1d * j)] = local;
  }

  std::vector<std::array<double, 2>> coordinates(
    static_cast<std::size_t>(dof));
  Vector physical(2);
  const IntegrationRule& nodes = fe.GetNodes();
  MFEM_VERIFY(nodes.GetNPoints() == dof,
              "Discrete DGSEM metrics require nodal element points.");
  for (int local = 0; local < dof; ++local)
  {
    transformation.Transform(nodes.IntPoint(local), physical);
    coordinates[static_cast<std::size_t>(local)] = {physical(0), physical(1)};
  }

  metrics.assign(static_cast<std::size_t>(dof), ContravariantMetric2D{});
  for (int j = 0; j < n1d; ++j)
  {
    for (int i = 0; i < n1d; ++i)
    {
      double dx_dxi = 0.0;
      double dy_dxi = 0.0;
      double dx_deta = 0.0;
      double dy_deta = 0.0;
      for (int m = 0; m < n1d; ++m)
      {
        const int local_xi =
          lex_to_local[static_cast<std::size_t>(m + n1d * j)];
        const int local_eta =
          lex_to_local[static_cast<std::size_t>(i + n1d * m)];
        dx_dxi += basis.D(i, m) *
                  coordinates[static_cast<std::size_t>(local_xi)][0];
        dy_dxi += basis.D(i, m) *
                  coordinates[static_cast<std::size_t>(local_xi)][1];
        dx_deta += basis.D(j, m) *
                   coordinates[static_cast<std::size_t>(local_eta)][0];
        dy_deta += basis.D(j, m) *
                   coordinates[static_cast<std::size_t>(local_eta)][1];
      }

      const int local =
        lex_to_local[static_cast<std::size_t>(i + n1d * j)];
      ContravariantMetric2D& metric =
        metrics[static_cast<std::size_t>(local)];
      metric.data = {dy_deta, -dx_deta, -dy_dxi, dx_dxi};
    }
  }
}

/** Identify the constant reference coordinate of a 2D element face. */
inline int ReferenceFaceDirection2D(const DenseMatrix& reference_tangent)
{
  MFEM_VERIFY(reference_tangent.Height() == 2 &&
                reference_tangent.Width() == 1,
              "ReferenceFaceDirection2D requires a 2x1 face tangent.");
  return std::abs(reference_tangent(0, 0)) <
             std::abs(reference_tangent(1, 0))
           ? 0
           : 1;
}

/** Return the outward endpoint metric normal used by the DGSEM face flux. */
inline void BuildNodalFaceNormal2D(
  const ContravariantMetric2D& metric,
  const DenseMatrix& reference_tangent,
  const IntegrationPoint& element_point,
  Vector& normal)
{
  MFEM_VERIFY(normal.Size() == 2,
              "BuildNodalFaceNormal2D requires a two-entry normal.");
  const int direction = ReferenceFaceDirection2D(reference_tangent);
  const double coordinate =
    direction == 0 ? element_point.x : element_point.y;
  MFEM_VERIFY(std::abs(coordinate) < 1.0e-10 ||
                std::abs(coordinate - 1.0) < 1.0e-10,
              "DGSEM face point is not on a reference endpoint.");
  const double sign = coordinate < 0.5 ? -1.0 : 1.0;
  for (int physical = 0; physical < 2; ++physical)
  {
    normal(physical) = sign * metric(direction, physical);
  }
}

} // namespace dgsem
} // namespace mfem

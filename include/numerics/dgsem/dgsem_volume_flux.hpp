#pragma once

/**
 * @file numerics/dgsem/dgsem_volume_flux.hpp
 * @brief Symmetric two-point volume flux helpers for DGSEM.
 * @ingroup numerics_dgsem
 */

#include "mfem.hpp"

#include <algorithm>
#include <cmath>

namespace mfem
{
namespace dgsem
{

/**
 * @brief Central symmetric two-point volume flux based on contravariant nodal fluxes.
 *
 * Computes Ĝ(u_i,u_j) = 0.5 * (G_i + G_j), where G_i/G_j are contravariant
 * fluxes at nodal states. This is consistent and symmetric by construction.
 *
 * @param[in]  flux     Contravariant flux matrix G (num_eq × (dim*dof)).
 * @param[in]  local_i  Local DOF index for the first node.
 * @param[in]  local_j  Local DOF index for the second node.
 * @param[in]  dir      Direction index in reference space.
 * @param[in]  dim      Space dimension.
 * @param[out] out      Output vector (size = num_eq).
 *
 * @note The input flux should be computed via FluxFunction::ComputeFlux(), so
 *       IGR (FluxWithIGR) contributions are included automatically.
 */
inline void SymmetricAverageContravariantFlux(const DenseMatrix& flux,
                                              int local_i,
                                              int local_j,
                                              int dir,
                                              int dim,
                                              Vector& out)
{
  const int num_eq = flux.NumRows();
  const int col_i = local_i * dim + dir;
  const int col_j = local_j * dim + dir;
  for (int eq = 0; eq < num_eq; ++eq)
  {
    out(eq) = 0.5 * (flux(eq, col_i) + flux(eq, col_j));
  }
}

/**
 * @brief Euler directional LF speed for flux projected along a contravariant vector.
 *
 * Uses lambda = |u dot a| + c |a| where `a` is the contravariant direction
 * vector associated with the reference-space derivative direction.
 */
inline double
EulerDirectionalLaxFriedrichsScale(const Vector& state,
                                   const Vector& contravariant_dir,
                                   int dim,
                                   double gamma,
                                   double sigma = 0.0)
{
  MFEM_VERIFY(
    state.Size() == dim + 2,
    "EulerDirectionalLaxFriedrichsScale expects Euler conservative state.");
  MFEM_VERIFY(contravariant_dir.Size() == dim,
              "EulerDirectionalLaxFriedrichsScale direction size mismatch.");

  constexpr double kFloor = 1.0e-14;
  const double rho = std::max(state(0), kFloor);
  const double rhoE = state(1 + dim);

  double mom2 = 0.0;
  double u_dot_a = 0.0;
  double a2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const double m = state(1 + d);
    const double u = m / rho;
    mom2 += m * m;
    u_dot_a += u * contravariant_dir(d);
    a2 += contravariant_dir(d) * contravariant_dir(d);
  }

  const double kinetic = 0.5 * mom2 / rho;
  const double p = std::max((gamma - 1.0) * (rhoE - kinetic) + sigma, kFloor);
  const double c = std::sqrt(std::max(gamma * p / rho, kFloor));
  const double a_norm = std::sqrt(std::max(a2, 0.0));
  const double speed = std::abs(u_dot_a) + c * a_norm;
  return std::isfinite(speed) ? speed : 0.0;
}

} // namespace dgsem
} // namespace mfem

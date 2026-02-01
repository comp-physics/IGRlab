#pragma once

/**
 * @file numerics/dgsem/dgsem_volume_flux.hpp
 * @brief Symmetric two-point volume flux helpers for DGSEM.
 * @ingroup numerics_dgsem
 */

#include "mfem.hpp"

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

} // namespace dgsem
} // namespace mfem

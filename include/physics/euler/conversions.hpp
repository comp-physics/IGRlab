#pragma once

#include "mfem.hpp"

/**
 * @file physics/euler/conversions.hpp
 * @brief Euler primitive <-> conservative conversions.
 * @ingroup physics_euler
 */

namespace mfem
{

/**
 * @brief Convert primitive state to conservative state (Euler).
 * @param[in]  prim  Primitive state [rho, u(0..dim-1), p].
 * @param[out] cons  Conservative state [rho, rho*u(0..dim-1), rho*E].
 * @param[in]  dim   Spatial dimension.
 * @param[in]  gamma Ratio of specific heats.
 */
inline void PrimToCons(const mfem::Vector &prim,
                       mfem::Vector &cons,
                       int dim,
                       double gamma)
{
  MFEM_VERIFY(prim.Size() >= dim + 2, "Primitive state must have [rho, u(0..dim-1), p]");
  MFEM_VERIFY(cons.Size() >= dim + 2, "Conservative state must have [rho, rho*u(0..dim-1), rhoE]");

  const double rho = prim(0);
  double kinetic = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const double u = prim(1 + d);
    cons(1 + d) = rho * u;
    kinetic += 0.5 * u * u;
  }
  const double p = prim(1 + dim);
  const double E = p / ((gamma - 1.0) * rho) + kinetic;

  cons(0) = rho;
  cons(1 + dim) = rho * E;
}

/**
 * @brief Convert conservative state to primitive state (Euler).
 * @param[in]  cons  Conservative state [rho, rho*u(0..dim-1), rho*E].
 * @param[out] prim  Primitive state [rho, u(0..dim-1), p].
 * @param[in]  dim   Spatial dimension.
 * @param[in]  gamma Ratio of specific heats.
 */
inline void ConsToPrim(const mfem::Vector &cons,
                       mfem::Vector &prim,
                       int dim,
                       double gamma)
{
  MFEM_VERIFY(cons.Size() >= dim + 2, "Conservative state must have [rho, rho*u(0..dim-1), rhoE]");
  MFEM_VERIFY(prim.Size() >= dim + 2, "Primitive state must have [rho, u(0..dim-1), p]");

  const double rho = cons(0);
  prim(0) = rho;

  double vel2 = 0.0;
  for (int d = 0; d < dim; ++d)
  {
    const double m = cons(1 + d);
    const double u = m / rho;
    prim(1 + d) = u;
    vel2 += u * u;
  }

  const double E = cons(1 + dim) / rho;
  const double p = (gamma - 1.0) * rho * (E - 0.5 * vel2);
  prim(1 + dim) = p;
}

} // namespace mfem

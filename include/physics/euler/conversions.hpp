#pragma once

#include "mfem.hpp"
#include "utils/config/deprecation.hpp"

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
 * @param[in]  dim   Spatial dimension.
 * @param[in]  gamma Ratio of specific heats.
 * @param[out] cons  Conservative state [rho, rho*u(0..dim-1), rho*E].
 */
inline void EulerPrimToConservative(const mfem::Vector& prim,
                                    int dim,
                                    double gamma,
                                    mfem::Vector& cons)
{
  MFEM_VERIFY(prim.Size() >= dim + 2,
              "Primitive state must have [rho, u(0..dim-1), p]");
  MFEM_VERIFY(cons.Size() >= dim + 2,
              "Conservative state must have [rho, rho*u(0..dim-1), rhoE]");

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
 * @param[in]  dim   Spatial dimension.
 * @param[in]  gamma Ratio of specific heats.
 * @param[out] prim  Primitive state [rho, u(0..dim-1), p].
 */
inline void EulerConservativeToPrimitive(const mfem::Vector& cons,
                                         int dim,
                                         double gamma,
                                         mfem::Vector& prim)
{
  MFEM_VERIFY(cons.Size() >= dim + 2,
              "Conservative state must have [rho, rho*u(0..dim-1), rhoE]");
  MFEM_VERIFY(prim.Size() >= dim + 2,
              "Primitive state must have [rho, u(0..dim-1), p]");

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

/**
 * @brief Backwards-compatible primitive-to-conservative conversion.
 * @deprecated Prefer EulerPrimToConservative(prim, dim, gamma, cons).
 */
DG_IGR_DEPRECATED("Use EulerPrimToConservative(prim, dim, gamma, cons)")
inline void
PrimToCons(const mfem::Vector& prim, mfem::Vector& cons, int dim, double gamma)
{
  EulerPrimToConservative(prim, dim, gamma, cons);
}

/**
 * @brief Backwards-compatible conservative-to-primitive conversion.
 * @deprecated Prefer EulerConservativeToPrimitive(cons, dim, gamma, prim).
 */
DG_IGR_DEPRECATED("Use EulerConservativeToPrimitive(cons, dim, gamma, prim)")
inline void
ConsToPrim(const mfem::Vector& cons, mfem::Vector& prim, int dim, double gamma)
{
  EulerConservativeToPrimitive(cons, dim, gamma, prim);
}

} // namespace mfem

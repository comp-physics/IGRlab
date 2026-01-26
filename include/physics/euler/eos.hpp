#pragma once

#include <algorithm>
#include "mfem.hpp"

/**
 * @file physics/euler/eos.hpp
 * @brief Euler equation-of-state helpers.
 * @ingroup physics_euler
 */

namespace mfem
{

/**
 * @brief Compute pressure p from a conservative state.
 * @param[in] cons   Conservative state [rho, rho*u(0..dim-1), rho*E].
 * @param[in] dim    Spatial dimension.
 * @param[in] gamma  Ratio of specific heats.
 * @param[in] rho_floor Density floor for robustness.
 */
inline double PressureFromCons(const mfem::Vector &cons,
                               int dim,
                               double gamma,
                               double rho_floor = 0.0)
{
  const double rho = std::max(cons(0), rho_floor);
  const double E   = cons(1 + dim);
  double mom2 = 0.0;
  for (int d = 0; d < dim; ++d) { mom2 += cons(1 + d) * cons(1 + d); }
  const double k = 0.5 * mom2 / rho;
  return (gamma - 1.0) * (E - k);
}

} // namespace mfem

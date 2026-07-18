#pragma once

#include "mfem.hpp"
#include <vector>

/**
 * @file numerics/limiters/zhang_shu.hpp
 * @brief Zhang-Shu positivity-preserving limiter for Euler.
 * @ingroup numerics_limiters
 */

namespace mfem
{

/// Points at which Zhang-Shu positivity is enforced.
enum class ZhangShuPointSet
{
  /// Legacy volume/face quadrature points.
  kQuadrature,
  /// Quadrature points plus FE nodes consumed by production operators.
  kConsumer
};

struct ZhangShuLimiterElementDiagnostics
{
  mfem::Vector active;
  mfem::Vector theta_rho;
  mfem::Vector theta_p;
  mfem::Vector collapsed_average;
  /// Component-major change applied to an inadmissible average: e + ne*c.
  mfem::Vector average_repair;

  void SetSize(int ne, int num_eq = 0)
  {
    active.SetSize(ne);
    theta_rho.SetSize(ne);
    theta_p.SetSize(ne);
    collapsed_average.SetSize(ne);
    average_repair.SetSize(ne * num_eq);
    active = 0.0;
    theta_rho = 1.0;
    theta_p = 1.0;
    collapsed_average = 0.0;
    average_repair = 0.0;
  }
};

/**
 * @brief Zhang-Shu positivity-preserving limiter for DG Euler.
 *
 * Enforces non-negativity of density and pressure at a set of intra-element
 * positivity points by blending each cell's polynomial solution toward its
 * cell average. The default consumer point set includes FE nodes evaluated by
 * production Euler and IGR operators.
 *
 * @note Assumes vfes uses Ordering::byNODES.
 * @note Operates locally on each element (no communication). In parallel,
 *       call sol.ExchangeFaceNbrData() after limiting to refresh ghosts.
 */
void ApplyZhangShuPositivityLimiter(mfem::ParFiniteElementSpace& vfes,
                                    mfem::ParGridFunction& U,
                                    int dim,
                                    double gamma,
                                    double rho_min,
                                    double p_min,
                                    int quad_order,
                                    const std::vector<int>& face_quad_orders = {},
                                    int p_bisect_iters = 12,
                                    ZhangShuPointSet point_set =
                                      ZhangShuPointSet::kConsumer,
                                    ZhangShuLimiterElementDiagnostics*
                                      diagnostics = nullptr);

} // namespace mfem

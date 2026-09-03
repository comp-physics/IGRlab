#pragma once

#include "mfem.hpp"

#include <optional>
#include <stdexcept>
#include <string>
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

/**
 * @brief Recoverable failure raised when one or more elements fail limiter
 * admission for a complete stage.
 *
 * A time integrator may catch this exception at its outer transaction
 * boundary, restore the pre-attempt state, reduce the timestep, and retry.
 * Other limiter failures remain fatal and must not be converted into retries.
 * Parallel producers must make the failure decision collectively and throw on
 * every rank in the state communicator.
 */
class RecoverableLimiterFailure : public std::runtime_error
{
public:
  explicit RecoverableLimiterFailure(int global_rejected_elements)
    : std::runtime_error(
        "Zhang-Shu limiter rejected a stage because " +
        std::to_string(global_rejected_elements) +
        " element(s) failed limiter admission globally."),
      global_rejected_elements_(global_rejected_elements)
  {
  }

  int GlobalRejectedElementCount() const noexcept
  {
    return global_rejected_elements_;
  }

private:
  int global_rejected_elements_ = 0;
};

struct ZhangShuLimiterElementDiagnostics
{
  /// One for an element whose density or pressure polynomial was blended.
  mfem::Vector active;
  /// Density blend factor in [0,1]; one means no density blending.
  mfem::Vector theta_rho;
  /// Pressure blend factor in [0,1]; one means no pressure blending.
  mfem::Vector theta_p;
  /// Elements whose mean made the limiter reject the complete stage.
  mfem::Vector inadmissible_average;
  /// Retained for compatibility; strict limiting never collapses an average.
  mfem::Vector collapsed_average;
  /// Retained for compatibility; strict limiting never repairs an average.
  mfem::Vector average_repair;

  void SetSize(int ne, int num_eq = 0)
  {
    active.SetSize(ne);
    theta_rho.SetSize(ne);
    theta_p.SetSize(ne);
    inadmissible_average.SetSize(ne);
    collapsed_average.SetSize(ne);
    average_repair.SetSize(ne * num_eq);
    active = 0.0;
    theta_rho = 1.0;
    theta_p = 1.0;
    inadmissible_average = 0.0;
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
 * @note The limiter-admission decision is collective on vfes.GetComm(). If an
 *       element has an inadmissible mean or a non-finite enforced consumer
 *       point, the input state is left unchanged on every rank and
 *       RecoverableLimiterFailure is thrown.
 * @note Otherwise operates locally on each element. In parallel, call
 *       sol.ExchangeFaceNbrData() after limiting to refresh ghosts.
 * @param[in] enforce_vtk_points Explicit refined-VTK-point policy. Omission
 *       preserves the legacy DG_IGR_LIMITER_ENFORCE_VTK_POINTS fallback.
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
                                      diagnostics = nullptr,
                                    std::optional<bool> enforce_vtk_points =
                                      std::nullopt);

} // namespace mfem

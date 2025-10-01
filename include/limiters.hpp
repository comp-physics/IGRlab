#pragma once
#include "mfem.hpp"

/**
 * @file limiters.hpp
 * @brief Positivity-preserving limiters for DG Euler in MFEM.
 */

/**
 * @brief Zhang–Shu (a.k.a. Shu) positivity-preserving limiter for DG Euler.
 *
 * Enforces non-negativity of density and pressure at a set of intra-element
 * “positivity points” by blending each cell’s polynomial solution toward its
 * cell average:
 *
 *   U_lim(x) = Ū + θ (U(x) − Ū),  with θ ∈ [0, 1] chosen per element,
 *
 * so that for all positivity points in the element:
 *   ρ ≥ rho_min,   p ≥ p_min,  with  p = (γ−1)\*(E − |m|²/(2ρ)).
 *
 * The procedure is conservative (preserves the cell average) unless the
 * average itself is inadmissible, in which case the element is set to a
 * constant admissible state with p = p_min.
 *
 * @note Assumes @p vfes uses @c Ordering::byNODES (your setup already does).
 * @note Operates locally on each element (no communication). In parallel,
 *       call @c sol.ExchangeFaceNbrData() after limiting to refresh ghosts.
 *
 * @param[in]  vfes            Vector FE space of @p U (vdim = dim+2).
 * @param[in,out] U            Conservative state (DG) to be limited in-place.
 * @param[in]  dim             Spatial dimension (2 or 3).
 * @param[in]  gamma           Specific heat ratio γ.
 * @param[in]  rho_min         Density floor (e.g., 1e-12 in your units).
 * @param[in]  p_min           Pressure floor (e.g., 1e-12 in your units).
 * @param[in]  quad_order      Quadrature order for positivity checks
 *                             (suggestion: max(2*p, 4)).
 * @param[in]  p_bisect_iters  Bisection iterations when enforcing p ≥ p_min
 *                             (default ~12 is plenty).
 */
void ApplyZhangShuPositivityLimiter(mfem::ParFiniteElementSpace &vfes,
                                    mfem::ParGridFunction &U,
                                    int dim, double gamma,
                                    double rho_min, double p_min,
                                    int quad_order,
                                    int p_bisect_iters = 12);

struct PressureCoefficient : public mfem::Coefficient
{
  const mfem::ParGridFunction *U = nullptr;
  int dim = 2;
  double gamma = 1.4;
  mutable mfem::Vector state; // reuse buffer

  PressureCoefficient() = default;
  PressureCoefficient(const mfem::ParGridFunction &U_, int dim_, double g_)
    : U(&U_), dim(dim_), gamma(g_), state(dim_+2) {}

  void SetState(const mfem::ParGridFunction &U_) { U = &U_; }
  void SetDimGamma(int d, double g) { dim = d; gamma = g; state.SetSize(d+2); }

  double Eval(mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip) override
  {
    MFEM_VERIFY(U, "PressureCoefficient: U pointer not set.");
    T.SetIntPoint(&ip);
    U->GetVectorValue(T, ip, state);   // [rho, m, E]
    const double rho = state(0);
    const double E   = state(1 + dim);
    double mom2 = 0.0; for (int d = 0; d < dim; ++d) { const double m = state(1+d); mom2 += m*m; }
    const double k = 0.5 * mom2 / std::max(rho, 1e-16);
    return (gamma - 1.0) * (E - k);
  }
};

#pragma once

#include "mfem.hpp"

#include <array>
#include <cstddef>

/**
 * @file numerics/timestepping/ssprk_coefficients.hpp
 * @brief Authoritative explicit SSP Runge--Kutta method data.
 * @ingroup numerics_timestepping
 */

namespace mfem
{

/// Explicit SSP Runge--Kutta methods supported by the shared limiter lifecycle.
enum class SSPRKMethod
{
  SSPRK33,
  SSPRK104,
  ResearchRK4,
  ResearchDOPRI5,
  ResearchRadauPicard5
};

/// Optimal three-stage, third-order SSP Runge--Kutta method.
struct SSPRK33Coefficients
{
  static constexpr std::size_t stage_count = 3;
  static constexpr int order = 3;
  static constexpr real_t ssp_coefficient = 1.0;
  using Row = std::array<real_t, stage_count>;
  using Matrix = std::array<Row, stage_count>;

  inline static constexpr Matrix a = {{
    {{0.0, 0.0, 0.0}},
    {{1.0, 0.0, 0.0}},
    {{0.25, 0.25, 0.0}},
  }};
  inline static constexpr Row b = {{1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0}};
  inline static constexpr Row c = {{0.0, 1.0, 0.5}};
};

/**
 * @brief Optimal ten-stage, fourth-order SSP Runge--Kutta method.
 *
 * This is SSPRK(10,4) from D. I. Ketcheson, SIAM J. Sci. Comput. 30(4),
 * 2113--2136 (2008), DOI 10.1137/07070485X. Its SSP coefficient is six.
 */
struct SSPRK104Coefficients
{
  static constexpr std::size_t stage_count = 10;
  static constexpr int order = 4;
  static constexpr real_t ssp_coefficient = 6.0;
  using Row = std::array<real_t, stage_count>;
  using Matrix = std::array<Row, stage_count>;

  inline static constexpr Matrix a = {{
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {{1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {{1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {{1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {{1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 6.0, 0.0, 0.0, 0.0, 0.0}},
    {{1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0, 0.0}},
    {{1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0}},
    {{1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0}},
  }};
  inline static constexpr Row b = {{0.1, 0.1, 0.1, 0.1, 0.1,
                                     0.1, 0.1, 0.1, 0.1, 0.1}};
  inline static constexpr Row c = {{0.0,
                                     1.0 / 6.0,
                                     1.0 / 3.0,
                                     0.5,
                                     2.0 / 3.0,
                                     1.0 / 3.0,
                                     0.5,
                                     2.0 / 3.0,
                                     5.0 / 6.0,
                                     1.0}};
};

/// Classical four-stage RK4; no positive SSP coefficient is claimed.
struct ResearchRK4Coefficients
{
  static constexpr std::size_t stage_count = 4;
  static constexpr int order = 4;
  using Row = std::array<real_t, stage_count>;
  using Matrix = std::array<Row, stage_count>;
  inline static constexpr Matrix a = {{
    {{0.0, 0.0, 0.0, 0.0}},
    {{0.5, 0.0, 0.0, 0.0}},
    {{0.0, 0.5, 0.0, 0.0}},
    {{0.0, 0.0, 1.0, 0.0}}
  }};
  inline static constexpr Row b = {{1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0}};
  inline static constexpr Row c = {{0.0, 0.5, 0.5, 1.0}};
};

/** Dormand--Prince fifth-order principal formula, fixed step, six RHS calls.
 *
 * Dormand and Prince, JCAM 6 (1980), 19--26,
 * DOI 10.1016/0771-050X(80)90013-3. The seventh evaluation is used by the
 * embedded estimator/FSAL implementation, not by this principal update.
 * No embedded error controller or FSAL cache is implemented here.
 */
struct ResearchDOPRI5Coefficients
{
  static constexpr std::size_t stage_count = 6;
  static constexpr int order = 5;
  using Row = std::array<real_t, stage_count>;
  using Matrix = std::array<Row, stage_count>;
  inline static constexpr Matrix a = {{
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {{1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    {{3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0}},
    {{44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0}},
    {{19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0,
      -212.0/729.0, 0.0, 0.0}},
    {{9017.0/3168.0, -355.0/33.0, 46732.0/5247.0,
      49.0/176.0, -5103.0/18656.0, 0.0}}
  }};
  inline static constexpr Row b = {{35.0/384.0, 0.0, 500.0/1113.0,
                                     125.0/192.0, -2187.0/6784.0, 11.0/84.0}};
  inline static constexpr Row c = {{0.0, 1.0/5.0, 3.0/10.0,
                                     4.0/5.0, 8.0/9.0, 1.0}};
};

/** Explicit, fixed-sweep approximation to three-node Radau IIA collocation.
 *
 * One initial RHS creates Euler predictors at all three nodes. Four Picard
 * sweeps then evaluate the previous sweep's three node states independently.
 * The final Radau quadrature update has order five for smooth reduced ODEs;
 * it does not inherit the converged implicit Radau method's stability.
 * Algebraic IGR variables must be solved at each evaluated node state.
 */
struct ResearchRadauPicard5Coefficients
{
  static constexpr std::size_t node_count = 3;
  static constexpr int sweeps = 4;
  static constexpr std::size_t stage_count = 1 + node_count * sweeps;
  static constexpr int order = 5;
  static constexpr real_t sqrt6 = 2.4494897427831780981972840747058914;
  using Row = std::array<real_t, node_count>;
  using Matrix = std::array<Row, node_count>;
  inline static constexpr Row c = {{(4.0-sqrt6)/10.0,
                                     (4.0+sqrt6)/10.0, 1.0}};
  inline static constexpr Matrix q = {{
    {{(88.0-7.0*sqrt6)/360.0, (296.0-169.0*sqrt6)/1800.0,
      (-2.0+3.0*sqrt6)/225.0}},
    {{(296.0+169.0*sqrt6)/1800.0, (88.0+7.0*sqrt6)/360.0,
      (-2.0-3.0*sqrt6)/225.0}},
    {{(16.0-sqrt6)/36.0, (16.0+sqrt6)/36.0, 1.0/9.0}}
  }};
  inline static constexpr Row b = q[2];
};

constexpr int SSPRKStageCount(SSPRKMethod method)
{
  switch (method)
  {
  case SSPRKMethod::SSPRK33:
    return static_cast<int>(SSPRK33Coefficients::stage_count);
  case SSPRKMethod::SSPRK104:
    return static_cast<int>(SSPRK104Coefficients::stage_count);
  case SSPRKMethod::ResearchRK4:
    return static_cast<int>(ResearchRK4Coefficients::stage_count);
  case SSPRKMethod::ResearchDOPRI5:
    return static_cast<int>(ResearchDOPRI5Coefficients::stage_count);
  case SSPRKMethod::ResearchRadauPicard5:
    return static_cast<int>(ResearchRadauPicard5Coefficients::stage_count);
  }
  return 0;
}

constexpr int SSPRKOrder(SSPRKMethod method)
{
  switch (method)
  {
  case SSPRKMethod::SSPRK33:
    return SSPRK33Coefficients::order;
  case SSPRKMethod::SSPRK104:
    return SSPRK104Coefficients::order;
  case SSPRKMethod::ResearchRK4:
    return ResearchRK4Coefficients::order;
  case SSPRKMethod::ResearchDOPRI5:
    return ResearchDOPRI5Coefficients::order;
  case SSPRKMethod::ResearchRadauPicard5:
    return ResearchRadauPicard5Coefficients::order;
  }
  return 0;
}

constexpr real_t SSPRKCoefficient(SSPRKMethod method)
{
  switch (method)
  {
  case SSPRKMethod::SSPRK33:
    return SSPRK33Coefficients::ssp_coefficient;
  case SSPRKMethod::SSPRK104:
    return SSPRK104Coefficients::ssp_coefficient;
  case SSPRKMethod::ResearchRK4:
  case SSPRKMethod::ResearchDOPRI5:
  case SSPRKMethod::ResearchRadauPicard5:
    return 0.0;
  }
  return 0.0;
}

/// Whether a valid RHS stage has a later RHS stage in the same RK step.
constexpr bool IsIntermediateRKStage(int stage, int stage_count)
{
  return stage >= 0 && stage_count > 0 && stage < stage_count &&
         stage + 1 < stage_count;
}

} // namespace mfem

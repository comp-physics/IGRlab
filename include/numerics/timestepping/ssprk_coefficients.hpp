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
  SSPRK104
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

constexpr int SSPRKStageCount(SSPRKMethod method)
{
  switch (method)
  {
  case SSPRKMethod::SSPRK33:
    return static_cast<int>(SSPRK33Coefficients::stage_count);
  case SSPRKMethod::SSPRK104:
    return static_cast<int>(SSPRK104Coefficients::stage_count);
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

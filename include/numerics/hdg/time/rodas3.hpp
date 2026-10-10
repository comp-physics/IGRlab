#pragma once

/**
 * @file numerics/hdg/time/rodas3.hpp
 * @brief Exact coefficient descriptor for the classic RODAS3 ROW 3(2) pair.
 *
 * The coefficient set and the order/stability labels are from A. Sandu,
 * J. G. Verwer, J. G. Blom, E. J. Spee, G. R. Carmichael, and F. A. Potra,
 * "Benchmarking stiff ODE solvers for atmospheric chemistry problems II:
 * Rosenbrock solvers," Atmospheric Environment 31(20) (1997), 3459--3472,
 * section 4, DOI 10.1016/S1352-2310(97)83212-8:
 * https://ir.cwi.nl/pub/125/0125D.pdf
 *
 * This header describes the method only.  It does not own stage state, choose
 * nonlinear or linear solvers, or alter the HDG time-integration lifecycle.
 */

#include <array>
#include <cstddef>
#include <cstdint>

namespace dg::hdg::time
{

/** A small exact rational used to keep published tableau data auditable. */
struct RationalCoefficient
{
  std::int64_t numerator;
  std::int64_t denominator;

  constexpr double value() const noexcept
  {
    return static_cast<double>(numerator) /
        static_cast<double>(denominator);
  }
};

namespace detail
{

constexpr std::int64_t Absolute(const std::int64_t value) noexcept
{
  return value < 0 ? -value : value;
}

constexpr std::int64_t GreatestCommonDivisor(std::int64_t lhs,
                                             std::int64_t rhs) noexcept
{
  lhs = Absolute(lhs);
  rhs = Absolute(rhs);
  while (rhs != 0)
  {
    const std::int64_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs == 0 ? 1 : lhs;
}

constexpr RationalCoefficient Normalize(
    const RationalCoefficient value) noexcept
{
  const std::int64_t divisor =
    GreatestCommonDivisor(value.numerator, value.denominator);
  const std::int64_t sign = value.denominator < 0 ? -1 : 1;
  return {sign * value.numerator / divisor,
          sign * value.denominator / divisor};
}

}  // namespace detail

constexpr bool operator==(const RationalCoefficient lhs,
                          const RationalCoefficient rhs) noexcept
{
  const RationalCoefficient lhs_normalized = detail::Normalize(lhs);
  const RationalCoefficient rhs_normalized = detail::Normalize(rhs);
  return lhs_normalized.numerator == rhs_normalized.numerator &&
      lhs_normalized.denominator == rhs_normalized.denominator;
}

constexpr bool operator!=(const RationalCoefficient lhs,
                          const RationalCoefficient rhs) noexcept
{
  return !(lhs == rhs);
}

namespace detail
{

constexpr RationalCoefficient Add(const RationalCoefficient lhs,
                                  const RationalCoefficient rhs) noexcept
{
  const std::int64_t denominator_gcd =
    GreatestCommonDivisor(lhs.denominator, rhs.denominator);
  return Normalize({
    lhs.numerator * (rhs.denominator / denominator_gcd) +
      rhs.numerator * (lhs.denominator / denominator_gcd),
    (lhs.denominator / denominator_gcd) * rhs.denominator
  });
}

constexpr RationalCoefficient Subtract(const RationalCoefficient lhs,
                                       const RationalCoefficient rhs) noexcept
{
  return Add(lhs, {-rhs.numerator, rhs.denominator});
}

constexpr RationalCoefficient Multiply(const RationalCoefficient lhs,
                                       const RationalCoefficient rhs) noexcept
{
  const std::int64_t lhs_cancellation =
    GreatestCommonDivisor(lhs.numerator, rhs.denominator);
  const std::int64_t rhs_cancellation =
    GreatestCommonDivisor(rhs.numerator, lhs.denominator);
  return Normalize({
    (lhs.numerator / lhs_cancellation) *
      (rhs.numerator / rhs_cancellation),
    (lhs.denominator / rhs_cancellation) *
      (rhs.denominator / lhs_cancellation)
  });
}

template <std::size_t N>
constexpr RationalCoefficient Sum(
    const std::array<RationalCoefficient, N>& values) noexcept
{
  RationalCoefficient sum{0, 1};
  for (const RationalCoefficient value : values)
  {
    sum = Add(sum, value);
  }
  return sum;
}

}  // namespace detail

/**
 * Classic four-stage RODAS3 coefficient and algorithm descriptor.
 *
 * The Gamma table includes its diagonal.  A consumer moves the common
 * diagonal gamma to the left-hand side, giving the shared stage operator
 * M - h*gamma*J.  For a non-autonomous singular-mass system M*y'=F(t,y),
 * stage i is
 *
 * (M-h*gamma*J) k_i = h F(t+c_i*h, y+sum_j A_ij*k_j)
 *   + h J sum_j Gamma_ij*k_j + h^2 d_i F_t,
 *
 * where both sums contain only j<i.  See G. Steinebach, "Construction of
 * Rosenbrock--Wanner method Rodas5P and numerical benchmarks within the Julia
 * Differential Equations package," BIT Numerical Mathematics 63:27 (2023),
 * equations (1.1)--(1.3), DOI 10.1007/s10543-023-00967-x, for this
 * mass-matrix form.
 *
 * The standard index-one order claim is conditional on a regular index-one
 * DAE and an exact frozen Jacobian.  M. Roche, "Rosenbrock Methods for
 * Differential Algebraic Equations," Numerische Mathematik 52 (1988),
 * 45--63, DOI 10.1007/BF01401021, derives the supplementary algebraic order
 * conditions.  Steinebach (2023), Tables 2 and 3, restates the ODE and
 * index-one conditions, while Table 1 reports Rodas3 order 3(2) on the
 * standard index-one DAE test.  Up through order three, the principal formula
 * must satisfy ODE conditions 1--4 and supplementary index-one condition 18.
 * The embedded formula needs ODE conditions 1--2 for order two; Table 3 adds
 * no supplementary index-one condition below order three.  These identities
 * are checked exactly below.  They do not assert parabolic stiff order: the
 * same Table 1 reports boundary-driven order reduction for Rodas3.
 */
struct Rodas3Coefficients
{
  static constexpr std::size_t stage_count = 4;
  static constexpr std::size_t residual_evaluation_count = 3;
  static constexpr int principal_order = 3;
  static constexpr int embedded_order = 2;
  static constexpr int index_one_principal_order = 3;
  static constexpr int index_one_embedded_order = 2;
  static constexpr bool principal_l_stable = true;
  static constexpr bool embedded_l_stable = true;
  static constexpr bool principal_stiffly_accurate = true;
  static constexpr bool embedded_is_final_stage_predictor = true;

  using Row = std::array<RationalCoefficient, stage_count>;
  using Matrix = std::array<Row, stage_count>;

  inline static constexpr RationalCoefficient common_gamma{1, 2};

  inline static constexpr Matrix alpha{{
    Row{{RationalCoefficient{0, 1}, RationalCoefficient{0, 1},
         RationalCoefficient{0, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{0, 1}, RationalCoefficient{0, 1},
         RationalCoefficient{0, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{1, 1}, RationalCoefficient{0, 1},
         RationalCoefficient{0, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{3, 4}, RationalCoefficient{-1, 4},
         RationalCoefficient{1, 2}, RationalCoefficient{0, 1}}}
  }};

  inline static constexpr Matrix gamma{{
    Row{{RationalCoefficient{1, 2}, RationalCoefficient{0, 1},
         RationalCoefficient{0, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{1, 1}, RationalCoefficient{1, 2},
         RationalCoefficient{0, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{-1, 4}, RationalCoefficient{-1, 4},
         RationalCoefficient{1, 2}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{1, 12}, RationalCoefficient{1, 12},
         RationalCoefficient{-2, 3}, RationalCoefficient{1, 2}}}
  }};

  /** B=A+Gamma, useful for transformed implementations and stiff accuracy. */
  inline static constexpr Matrix beta{{
    Row{{RationalCoefficient{1, 2}, RationalCoefficient{0, 1},
         RationalCoefficient{0, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{1, 1}, RationalCoefficient{1, 2},
         RationalCoefficient{0, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{3, 4}, RationalCoefficient{-1, 4},
         RationalCoefficient{1, 2}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{5, 6}, RationalCoefficient{-1, 6},
         RationalCoefficient{-1, 6}, RationalCoefficient{1, 2}}}
  }};

  /** Exact inverse of B, also used by the index-one order conditions. */
  inline static constexpr Matrix beta_inverse{{
    Row{{RationalCoefficient{2, 1}, RationalCoefficient{0, 1},
         RationalCoefficient{0, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{-4, 1}, RationalCoefficient{2, 1},
         RationalCoefficient{0, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{-5, 1}, RationalCoefficient{1, 1},
         RationalCoefficient{2, 1}, RationalCoefficient{0, 1}}},
    Row{{RationalCoefficient{-19, 3}, RationalCoefficient{1, 1},
         RationalCoefficient{2, 3}, RationalCoefficient{2, 1}}}
  }};

  inline static constexpr Row principal_weights{{
    RationalCoefficient{5, 6}, RationalCoefficient{-1, 6},
    RationalCoefficient{-1, 6}, RationalCoefficient{1, 2}
  }};

  inline static constexpr Row embedded_weights{{
    RationalCoefficient{3, 4}, RationalCoefficient{-1, 4},
    RationalCoefficient{1, 2}, RationalCoefficient{0, 1}
  }};

  inline static constexpr Row error_weights{{
    RationalCoefficient{1, 12}, RationalCoefficient{1, 12},
    RationalCoefficient{-2, 3}, RationalCoefficient{1, 2}
  }};

  inline static constexpr Row stage_abscissae{{
    RationalCoefficient{0, 1}, RationalCoefficient{0, 1},
    RationalCoefficient{1, 1}, RationalCoefficient{1, 1}
  }};

  /** Coefficients d_i multiplying h^2*F_t in the mass-matrix formula. */
  inline static constexpr Row time_derivative_weights{{
    RationalCoefficient{1, 2}, RationalCoefficient{3, 2},
    RationalCoefficient{0, 1}, RationalCoefficient{0, 1}
  }};

  /** Stages one and two share exactly one residual evaluation point. */
  inline static constexpr std::array<std::size_t, stage_count>
      residual_evaluation_slot{{0, 0, 1, 2}};

  static constexpr double Alpha(const std::size_t i,
                                const std::size_t j) noexcept
  {
    return alpha[i][j].value();
  }

  static constexpr double Gamma(const std::size_t i,
                                const std::size_t j) noexcept
  {
    return gamma[i][j].value();
  }

  static constexpr double PrincipalWeight(const std::size_t i) noexcept
  {
    return principal_weights[i].value();
  }

  static constexpr double EmbeddedWeight(const std::size_t i) noexcept
  {
    return embedded_weights[i].value();
  }

  static constexpr double ErrorWeight(const std::size_t i) noexcept
  {
    return error_weights[i].value();
  }

  static constexpr double ShiftScale(const double step_size) noexcept
  {
    return step_size * common_gamma.value();
  }
};

namespace detail
{

constexpr bool Rodas3DenominatorsArePositive() noexcept
{
  for (std::size_t i = 0; i < Rodas3Coefficients::stage_count; ++i)
  {
    if (Rodas3Coefficients::principal_weights[i].denominator <= 0 ||
        Rodas3Coefficients::embedded_weights[i].denominator <= 0 ||
        Rodas3Coefficients::error_weights[i].denominator <= 0 ||
        Rodas3Coefficients::stage_abscissae[i].denominator <= 0 ||
        Rodas3Coefficients::time_derivative_weights[i].denominator <= 0)
    {
      return false;
    }
    for (std::size_t j = 0; j < Rodas3Coefficients::stage_count; ++j)
    {
      if (Rodas3Coefficients::alpha[i][j].denominator <= 0 ||
          Rodas3Coefficients::gamma[i][j].denominator <= 0 ||
          Rodas3Coefficients::beta[i][j].denominator <= 0 ||
          Rodas3Coefficients::beta_inverse[i][j].denominator <= 0)
      {
        return false;
      }
    }
  }
  return Rodas3Coefficients::common_gamma.denominator > 0;
}

constexpr bool Rodas3TriangularCommonShiftIsExact() noexcept
{
  const RationalCoefficient zero{0, 1};
  for (std::size_t i = 0; i < Rodas3Coefficients::stage_count; ++i)
  {
    for (std::size_t j = 0; j < Rodas3Coefficients::stage_count; ++j)
    {
      if (j >= i && Rodas3Coefficients::alpha[i][j] != zero)
      {
        return false;
      }
      if (j > i && Rodas3Coefficients::gamma[i][j] != zero)
      {
        return false;
      }
    }
    if (Rodas3Coefficients::gamma[i][i] !=
        Rodas3Coefficients::common_gamma)
    {
      return false;
    }
  }
  return true;
}

constexpr bool Rodas3DerivedTablesAreExact() noexcept
{
  for (std::size_t i = 0; i < Rodas3Coefficients::stage_count; ++i)
  {
    RationalCoefficient alpha_sum{0, 1};
    RationalCoefficient gamma_sum{0, 1};
    for (std::size_t j = 0; j < Rodas3Coefficients::stage_count; ++j)
    {
      if (Rodas3Coefficients::beta[i][j] !=
          Add(Rodas3Coefficients::alpha[i][j],
              Rodas3Coefficients::gamma[i][j]))
      {
        return false;
      }
      alpha_sum = Add(alpha_sum, Rodas3Coefficients::alpha[i][j]);
      gamma_sum = Add(gamma_sum, Rodas3Coefficients::gamma[i][j]);
    }
    if (alpha_sum != Rodas3Coefficients::stage_abscissae[i] ||
        gamma_sum != Rodas3Coefficients::time_derivative_weights[i])
    {
      return false;
    }
  }
  return true;
}

constexpr bool Rodas3WeightsAreExact() noexcept
{
  for (std::size_t j = 0; j < Rodas3Coefficients::stage_count; ++j)
  {
    if (Rodas3Coefficients::principal_weights[j] !=
          Rodas3Coefficients::beta[Rodas3Coefficients::stage_count - 1][j] ||
        Rodas3Coefficients::embedded_weights[j] !=
          Rodas3Coefficients::alpha[Rodas3Coefficients::stage_count - 1][j] ||
        Rodas3Coefficients::error_weights[j] !=
          Subtract(Rodas3Coefficients::principal_weights[j],
                   Rodas3Coefficients::embedded_weights[j]))
    {
      return false;
    }
  }
  return Sum(Rodas3Coefficients::principal_weights) ==
           RationalCoefficient{1, 1} &&
      Sum(Rodas3Coefficients::embedded_weights) ==
           RationalCoefficient{1, 1} &&
      Sum(Rodas3Coefficients::error_weights) == RationalCoefficient{0, 1};
}

constexpr bool Rodas3BetaInverseIsExact() noexcept
{
  for (std::size_t i = 0; i < Rodas3Coefficients::stage_count; ++i)
  {
    for (std::size_t j = 0; j < Rodas3Coefficients::stage_count; ++j)
    {
      RationalCoefficient product{0, 1};
      for (std::size_t k = 0; k < Rodas3Coefficients::stage_count; ++k)
      {
        product = Add(
          product,
          Multiply(Rodas3Coefficients::beta[i][k],
                   Rodas3Coefficients::beta_inverse[k][j]));
      }
      if (product != RationalCoefficient{i == j ? 1 : 0, 1})
      {
        return false;
      }
    }
  }
  return true;
}

constexpr RationalCoefficient Rodas3BetaRowSum(
    const std::size_t row) noexcept
{
  RationalCoefficient sum{0, 1};
  for (std::size_t j = 0; j < Rodas3Coefficients::stage_count; ++j)
  {
    sum = Add(sum, Rodas3Coefficients::beta[row][j]);
  }
  return sum;
}

constexpr RationalCoefficient Rodas3WeightBetaMoment(
    const Rodas3Coefficients::Row& weights) noexcept
{
  RationalCoefficient moment{0, 1};
  for (std::size_t i = 0; i < Rodas3Coefficients::stage_count; ++i)
  {
    moment = Add(moment,
                 Multiply(weights[i], Rodas3BetaRowSum(i)));
  }
  return moment;
}

constexpr RationalCoefficient Rodas3WeightAlphaSquareMoment(
    const Rodas3Coefficients::Row& weights) noexcept
{
  RationalCoefficient moment{0, 1};
  for (std::size_t i = 0; i < Rodas3Coefficients::stage_count; ++i)
  {
    moment = Add(
      moment,
      Multiply(
        weights[i],
        Multiply(Rodas3Coefficients::stage_abscissae[i],
                 Rodas3Coefficients::stage_abscissae[i])));
  }
  return moment;
}

constexpr RationalCoefficient Rodas3WeightBetaChainMoment(
    const Rodas3Coefficients::Row& weights) noexcept
{
  RationalCoefficient moment{0, 1};
  for (std::size_t i = 0; i < Rodas3Coefficients::stage_count; ++i)
  {
    for (std::size_t j = 0; j < Rodas3Coefficients::stage_count; ++j)
    {
      moment = Add(
        moment,
        Multiply(
          weights[i],
          Multiply(Rodas3Coefficients::beta[i][j],
                   Rodas3BetaRowSum(j))));
    }
  }
  return moment;
}

/** Steinebach (2023), Table 3, supplementary index-one condition 18. */
constexpr RationalCoefficient Rodas3IndexOneOrderThreeMoment(
    const Rodas3Coefficients::Row& weights) noexcept
{
  RationalCoefficient moment{0, 1};
  for (std::size_t i = 0; i < Rodas3Coefficients::stage_count; ++i)
  {
    for (std::size_t j = 0; j < Rodas3Coefficients::stage_count; ++j)
    {
      const RationalCoefficient alpha_squared = Multiply(
        Rodas3Coefficients::stage_abscissae[j],
        Rodas3Coefficients::stage_abscissae[j]);
      moment = Add(
        moment,
        Multiply(
          weights[i],
          Multiply(Rodas3Coefficients::beta_inverse[i][j],
                   alpha_squared)));
    }
  }
  return moment;
}

constexpr RationalCoefficient Rodas3StiffLimitMoment(
    const Rodas3Coefficients::Row& weights) noexcept
{
  RationalCoefficient moment{0, 1};
  for (std::size_t i = 0; i < Rodas3Coefficients::stage_count; ++i)
  {
    for (std::size_t j = 0; j < Rodas3Coefficients::stage_count; ++j)
    {
      moment = Add(
        moment,
        Multiply(weights[i], Rodas3Coefficients::beta_inverse[i][j]));
    }
  }
  return moment;
}

/** Steinebach (2023), Table 2, ODE conditions 1--4. */
constexpr bool Rodas3PrincipalOdeOrderThreeConditionsAreExact() noexcept
{
  return Sum(Rodas3Coefficients::principal_weights) ==
          RationalCoefficient{1, 1} &&
      Rodas3WeightBetaMoment(Rodas3Coefficients::principal_weights) ==
          RationalCoefficient{1, 2} &&
      Rodas3WeightAlphaSquareMoment(
        Rodas3Coefficients::principal_weights) ==
          RationalCoefficient{1, 3} &&
      Rodas3WeightBetaChainMoment(
        Rodas3Coefficients::principal_weights) ==
          RationalCoefficient{1, 6};
}

/** Table 2 conditions 1--2; Table 3 adds none for index-one order two. */
constexpr bool Rodas3EmbeddedIndexOneOrderTwoConditionsAreExact() noexcept
{
  return Sum(Rodas3Coefficients::embedded_weights) ==
          RationalCoefficient{1, 1} &&
      Rodas3WeightBetaMoment(Rodas3Coefficients::embedded_weights) ==
          RationalCoefficient{1, 2};
}

constexpr bool Rodas3PrincipalIndexOneOrderThreeConditionsAreExact() noexcept
{
  return Rodas3PrincipalOdeOrderThreeConditionsAreExact() &&
      Rodas3IndexOneOrderThreeMoment(
        Rodas3Coefficients::principal_weights) ==
          RationalCoefficient{1, 1};
}

/** Exact R(infinity)=1-b^T*B^{-1}*1 cancellation for both formulas. */
constexpr bool Rodas3StiffLimitCancellationIsExact() noexcept
{
  return Rodas3StiffLimitMoment(Rodas3Coefficients::principal_weights) ==
          RationalCoefficient{1, 1} &&
      Rodas3StiffLimitMoment(Rodas3Coefficients::embedded_weights) ==
          RationalCoefficient{1, 1};
}

}  // namespace detail

static_assert(detail::Rodas3DenominatorsArePositive(),
              "RODAS3 rational coefficients require positive denominators");
static_assert(detail::Rodas3TriangularCommonShiftIsExact(),
              "RODAS3 must retain one common shifted stage operator");
static_assert(detail::Rodas3DerivedTablesAreExact(),
              "RODAS3 derived B, abscissa, or F_t data drifted");
static_assert(detail::Rodas3WeightsAreExact(),
              "RODAS3 principal, embedded, or error weights drifted");
static_assert(detail::Rodas3BetaInverseIsExact(),
              "RODAS3 B inverse must remain exact");
static_assert(detail::Rodas3PrincipalIndexOneOrderThreeConditionsAreExact(),
              "RODAS3 must retain all ODE and index-one order-three conditions");
static_assert(detail::Rodas3EmbeddedIndexOneOrderTwoConditionsAreExact(),
              "RODAS3 embedding must retain its index-one order-two conditions");
static_assert(detail::Rodas3StiffLimitCancellationIsExact(),
              "both RODAS3 formulas must retain exact stiff-limit damping");
static_assert(Rodas3Coefficients::residual_evaluation_slot[0] ==
                Rodas3Coefficients::residual_evaluation_slot[1] &&
              Rodas3Coefficients::residual_evaluation_slot[2] !=
                Rodas3Coefficients::residual_evaluation_slot[1] &&
              Rodas3Coefficients::residual_evaluation_slot[3] !=
                Rodas3Coefficients::residual_evaluation_slot[2],
              "RODAS3 must retain four stages and three residual evaluations");

}  // namespace dg::hdg::time

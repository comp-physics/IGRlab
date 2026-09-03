#include "numerics/hdg/time/rodas3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <string>

namespace
{

using Coefficients = dg::hdg::time::Rodas3Coefficients;
using Rational = dg::hdg::time::RationalCoefficient;

static_assert(Coefficients::stage_count == 4,
              "classic RODAS3 has four causal stages");
static_assert(Coefficients::residual_evaluation_count == 3,
              "classic RODAS3 reuses the first residual evaluation");
static_assert(Coefficients::common_gamma == Rational{1, 2},
              "classic RODAS3 uses gamma=1/2");
static_assert(Coefficients::principal_weights[0] == Rational{5, 6} &&
              Coefficients::principal_weights[3] == Rational{1, 2},
              "principal weights must match the published tableau");
static_assert(Coefficients::embedded_weights[0] == Rational{3, 4} &&
              Coefficients::embedded_weights[3] == Rational{0, 1},
              "embedded weights must match the final stage predictor");
static_assert(Coefficients::beta_inverse[3][0] == Rational{-19, 3} &&
              Coefficients::beta_inverse[3][2] == Rational{2, 3},
              "the exact transformed-stage inverse must remain available");

class Checks
{
public:
  void Expect(const bool condition, const std::string& message)
  {
    if (!condition)
    {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  int failures() const noexcept { return failures_; }

private:
  int failures_ = 0;
};

bool Near(const double lhs, const double rhs,
          const double tolerance = 2.0e-13)
{
  return std::abs(lhs - rhs) <=
      tolerance * (1.0 + std::abs(lhs) + std::abs(rhs));
}

struct ScalarStep
{
  std::array<double, Coefficients::stage_count> increments{};
  double principal = 0.0;
  double embedded = 0.0;
  double estimate = 0.0;
};

/** Test-only implementation of one autonomous scalar mass-matrix stage set. */
ScalarStep AutonomousScalarLinearStep(const double mass,
                                      const double jacobian,
                                      const double step_size,
                                      const double initial)
{
  ScalarStep result;
  const double shifted_operator =
    mass - Coefficients::ShiftScale(step_size) * jacobian;

  for (std::size_t i = 0; i < Coefficients::stage_count; ++i)
  {
    double stage_state = initial;
    double jacobian_history = 0.0;
    for (std::size_t j = 0; j < i; ++j)
    {
      stage_state += Coefficients::Alpha(i, j) * result.increments[j];
      jacobian_history +=
        Coefficients::Gamma(i, j) * result.increments[j];
    }
    const double right_hand_side =
      step_size * jacobian * stage_state +
      step_size * jacobian * jacobian_history;
    result.increments[i] = right_hand_side / shifted_operator;
  }

  result.principal = initial;
  result.embedded = initial;
  result.estimate = 0.0;
  for (std::size_t i = 0; i < Coefficients::stage_count; ++i)
  {
    result.principal +=
      Coefficients::PrincipalWeight(i) * result.increments[i];
    result.embedded +=
      Coefficients::EmbeddedWeight(i) * result.increments[i];
    result.estimate +=
      Coefficients::ErrorWeight(i) * result.increments[i];
  }
  return result;
}

template <typename Scalar>
Scalar PrincipalStabilityFunction(const Scalar z)
{
  const Scalar denominator = z - Scalar{2.0};
  const Scalar denominator_squared = denominator * denominator;
  return Scalar{8.0} *
      (z * z * z - Scalar{6.0} * z + Scalar{6.0}) /
      (Scalar{3.0} * denominator_squared * denominator_squared);
}

template <typename Scalar>
Scalar EmbeddedStabilityFunction(const Scalar z)
{
  const Scalar denominator = z - Scalar{2.0};
  return Scalar{2.0} *
      (z * z + Scalar{2.0} * z - Scalar{4.0}) /
      (denominator * denominator * denominator);
}

double IntegrateScalar(const int steps, const bool use_embedded)
{
  const double step_size = 1.0 / static_cast<double>(steps);
  double value = 1.0;
  for (int step = 0; step < steps; ++step)
  {
    const ScalarStep pair =
      AutonomousScalarLinearStep(1.0, -1.0, step_size, value);
    value = use_embedded ? pair.embedded : pair.principal;
  }
  return value;
}

struct IndexOneDaeState
{
  double differential = 0.0;
  double algebraic = 0.0;
};

struct IndexOneDaeStep
{
  std::array<IndexOneDaeState, Coefficients::stage_count> increments{};
  IndexOneDaeState principal{};
  IndexOneDaeState embedded{};
  IndexOneDaeState estimate{};
};

IndexOneDaeState AddScaled(const IndexOneDaeState state,
                           const double scale,
                           const IndexOneDaeState increment)
{
  return {state.differential + scale * increment.differential,
          state.algebraic + scale * increment.algebraic};
}

IndexOneDaeState Add(const IndexOneDaeState lhs,
                     const IndexOneDaeState rhs)
{
  return {lhs.differential + rhs.differential,
          lhs.algebraic + rhs.algebraic};
}

IndexOneDaeState Scale(const double scale, const IndexOneDaeState state)
{
  return {scale * state.differential, scale * state.algebraic};
}

/**
 * Semi-explicit index-one DAE
 *
 *   x' = z,
 *    0 = -x-z,
 *
 * with M=diag(1,0), g_z=-1, and consistent initial state (1,-1).
 * Its exact solution is x=exp(-t), z=-exp(-t).  The stage solve below applies
 * the published singular-mass equation directly; it never inverts M.
 */
IndexOneDaeState IndexOneDaeResidual(const IndexOneDaeState state)
{
  return {state.algebraic,
          -state.differential - state.algebraic};
}

IndexOneDaeState SolveIndexOneShift(const double step_size,
                                    const IndexOneDaeState rhs)
{
  const double shifted_scale = Coefficients::ShiftScale(step_size);
  // M-h*gamma*J = [[1,-h*gamma],[h*gamma,h*gamma]].
  const double determinant =
    shifted_scale * (1.0 + shifted_scale);
  return {
    shifted_scale * (rhs.differential + rhs.algebraic) / determinant,
    (rhs.algebraic - shifted_scale * rhs.differential) / determinant
  };
}

IndexOneDaeStep IndexOneDaeLinearStep(const double step_size,
                                      const IndexOneDaeState initial)
{
  IndexOneDaeStep result;
  for (std::size_t i = 0; i < Coefficients::stage_count; ++i)
  {
    IndexOneDaeState stage_state = initial;
    IndexOneDaeState jacobian_history{};
    for (std::size_t j = 0; j < i; ++j)
    {
      stage_state = AddScaled(
        stage_state, Coefficients::Alpha(i, j), result.increments[j]);
      jacobian_history = AddScaled(
        jacobian_history, Coefficients::Gamma(i, j), result.increments[j]);
    }
    const IndexOneDaeState rhs = Add(
      Scale(step_size, IndexOneDaeResidual(stage_state)),
      Scale(step_size, IndexOneDaeResidual(jacobian_history)));
    result.increments[i] = SolveIndexOneShift(step_size, rhs);
  }

  result.principal = initial;
  result.embedded = initial;
  for (std::size_t i = 0; i < Coefficients::stage_count; ++i)
  {
    result.principal = AddScaled(
      result.principal,
      Coefficients::PrincipalWeight(i),
      result.increments[i]);
    result.embedded = AddScaled(
      result.embedded,
      Coefficients::EmbeddedWeight(i),
      result.increments[i]);
    result.estimate = AddScaled(
      result.estimate,
      Coefficients::ErrorWeight(i),
      result.increments[i]);
  }
  return result;
}

IndexOneDaeState IntegrateIndexOneDae(const int steps,
                                      const bool use_embedded)
{
  const double step_size = 1.0 / static_cast<double>(steps);
  IndexOneDaeState state{1.0, -1.0};
  for (int step = 0; step < steps; ++step)
  {
    const IndexOneDaeStep pair = IndexOneDaeLinearStep(step_size, state);
    state = use_embedded ? pair.embedded : pair.principal;
  }
  return state;
}

double IndexOneDaeError(const IndexOneDaeState state)
{
  const double exact = std::exp(-1.0);
  return std::max(std::abs(state.differential - exact),
                  std::abs(state.algebraic + exact));
}

}  // namespace

int main()
{
  Checks checks;

  checks.Expect(Coefficients::principal_order == 3 &&
                  Coefficients::embedded_order == 2 &&
                  Coefficients::index_one_principal_order == 3 &&
                  Coefficients::index_one_embedded_order == 2,
                "the descriptor exposes the published ODE and index-one 3(2) orders");
  checks.Expect(Coefficients::principal_l_stable &&
                  Coefficients::embedded_l_stable &&
                  Coefficients::principal_stiffly_accurate &&
                  Coefficients::embedded_is_final_stage_predictor,
                "the descriptor exposes the published stability structure");

  constexpr std::array<double, 4> stability_points{{
    -0.1, -1.0, -10.0, -100.0
  }};
  for (const double z : stability_points)
  {
    const ScalarStep pair = AutonomousScalarLinearStep(1.0, z, 1.0, 1.0);
    checks.Expect(Near(pair.principal, PrincipalStabilityFunction(z)),
                  "principal stages reproduce the exact scalar stability function");
    checks.Expect(Near(pair.embedded, EmbeddedStabilityFunction(z)),
                  "embedded stages reproduce the exact scalar stability function");
    checks.Expect(Near(pair.principal - pair.embedded, pair.estimate),
                  "embedded difference equals the published error-weight update");
  }

  {
    // On z=i*y the exact stability functions obey
    //   1-|R_3|^2 = y^4(9y^4+80y^2+96)/(9(y^2+4)^4),
    //   1-|R_2|^2 = y^4(y^2+8)/(y^2+4)^3.
    // Both are nonnegative.  Together with the only pole at z=2 and the exact
    // stiff-limit cancellation checked in the coefficient header, this checks
    // the published L-stability labels independently of the stage evaluator.
    constexpr std::array<double, 5> imaginary_axis_points{{
      0.0, 0.1, 1.0, 10.0, 100.0
    }};
    for (const double y : imaginary_axis_points)
    {
      const std::complex<double> z{0.0, y};
      const double y_squared = y * y;
      const double y_fourth = y_squared * y_squared;
      const double principal_defect =
        y_fourth *
          (9.0 * y_fourth + 80.0 * y_squared + 96.0) /
        (9.0 * std::pow(y_squared + 4.0, 4));
      const double embedded_defect =
        y_fourth * (y_squared + 8.0) /
        std::pow(y_squared + 4.0, 3);
      checks.Expect(
        Near(1.0 - std::norm(PrincipalStabilityFunction(z)),
             principal_defect),
        "principal stability function satisfies its imaginary-axis identity");
      checks.Expect(
        Near(1.0 - std::norm(EmbeddedStabilityFunction(z)),
             embedded_defect),
        "embedded stability function satisfies its imaginary-axis identity");
    }
  }

  {
    const ScalarStep stiff =
      AutonomousScalarLinearStep(1.0, -1.0e6, 1.0, 1.0);
    checks.Expect(std::abs(stiff.principal) < 3.0e-5 &&
                    std::abs(stiff.embedded) < 3.0e-5,
                  "both formulas damp the stiff scalar limit toward zero");
  }

  {
    const double exact = std::exp(-1.0);
    const double principal_error_20 =
      std::abs(IntegrateScalar(20, false) - exact);
    const double principal_error_40 =
      std::abs(IntegrateScalar(40, false) - exact);
    const double principal_error_80 =
      std::abs(IntegrateScalar(80, false) - exact);
    const double embedded_error_20 =
      std::abs(IntegrateScalar(20, true) - exact);
    const double embedded_error_40 =
      std::abs(IntegrateScalar(40, true) - exact);
    const double embedded_error_80 =
      std::abs(IntegrateScalar(80, true) - exact);

    const double principal_ratio_1 = principal_error_20 / principal_error_40;
    const double principal_ratio_2 = principal_error_40 / principal_error_80;
    const double embedded_ratio_1 = embedded_error_20 / embedded_error_40;
    const double embedded_ratio_2 = embedded_error_40 / embedded_error_80;
    checks.Expect(principal_ratio_1 > 7.0 && principal_ratio_1 < 9.0 &&
                    principal_ratio_2 > 7.0 && principal_ratio_2 < 9.0,
                  "principal scalar integration converges at global order three");
    checks.Expect(embedded_ratio_1 > 3.5 && embedded_ratio_1 < 4.5 &&
                    embedded_ratio_2 > 3.5 && embedded_ratio_2 < 4.5,
                  "embedded scalar integration converges at global order two");
  }

  {
    constexpr std::array<int, 3> step_counts{{20, 40, 80}};
    std::array<double, step_counts.size()> principal_errors{};
    std::array<double, step_counts.size()> embedded_errors{};
    for (std::size_t i = 0; i < step_counts.size(); ++i)
    {
      const IndexOneDaeState principal =
        IntegrateIndexOneDae(step_counts[i], false);
      const IndexOneDaeState embedded =
        IntegrateIndexOneDae(step_counts[i], true);
      principal_errors[i] = IndexOneDaeError(principal);
      embedded_errors[i] = IndexOneDaeError(embedded);
      checks.Expect(std::abs(principal.differential +
                               principal.algebraic) < 2.0e-13 &&
                      std::abs(embedded.differential +
                               embedded.algebraic) < 2.0e-13,
                    "both formulas preserve the linear index-one constraint");
    }

    const double principal_ratio_1 =
      principal_errors[0] / principal_errors[1];
    const double principal_ratio_2 =
      principal_errors[1] / principal_errors[2];
    const double embedded_ratio_1 =
      embedded_errors[0] / embedded_errors[1];
    const double embedded_ratio_2 =
      embedded_errors[1] / embedded_errors[2];
    checks.Expect(principal_ratio_1 > 7.5 && principal_ratio_1 < 8.5 &&
                    principal_ratio_2 > 7.5 && principal_ratio_2 < 8.5,
                  "principal singular-mass DAE integration has global order three");
    checks.Expect(embedded_ratio_1 > 3.6 && embedded_ratio_1 < 4.4 &&
                    embedded_ratio_2 > 3.6 && embedded_ratio_2 < 4.4,
                  "embedded singular-mass DAE integration has global order two");

    constexpr std::array<double, 3> estimator_steps{{0.05, 0.025, 0.0125}};
    std::array<double, estimator_steps.size()> estimator_magnitudes{};
    for (std::size_t i = 0; i < estimator_steps.size(); ++i)
    {
      const IndexOneDaeStep pair =
        IndexOneDaeLinearStep(estimator_steps[i], {1.0, -1.0});
      checks.Expect(
        Near(pair.principal.differential - pair.embedded.differential,
             pair.estimate.differential) &&
          Near(pair.principal.algebraic - pair.embedded.algebraic,
               pair.estimate.algebraic),
        "the DAE principal/embedded difference equals the error-weight update");
      estimator_magnitudes[i] = std::abs(pair.estimate.differential);
    }
    const double estimator_ratio_1 =
      estimator_magnitudes[0] / estimator_magnitudes[1];
    const double estimator_ratio_2 =
      estimator_magnitudes[1] / estimator_magnitudes[2];
    checks.Expect(estimator_ratio_1 > 7.2 && estimator_ratio_1 < 8.6 &&
                    estimator_ratio_2 > 7.2 && estimator_ratio_2 < 8.6,
                  "the differential DAE error estimate has local order three");
  }

  return checks.failures() == 0 ? 0 : 1;
}

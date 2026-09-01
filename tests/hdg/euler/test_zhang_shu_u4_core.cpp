#include "numerics/hdg/euler/zhang_shu_u4.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace
{

namespace zs = dg::hdg::euler::zhang_shu;

bool Check(const bool condition, const char* const name)
{
  if (!condition) { std::cerr << "FAILED: " << name << '\n'; }
  return condition;
}

bool Near(const double left, const double right, const double tolerance)
{
  return std::abs(left - right) <= tolerance;
}

bool SameBits(const double& left, const double& right)
{
  return std::memcmp(&left, &right, sizeof(double)) == 0;
}

template <std::size_t N>
bool SameBits(const std::array<double, N>& left,
              const std::array<double, N>& right)
{
  return std::memcmp(left.data(), right.data(), N * sizeof(double)) == 0;
}

std::array<double, 4> State(const double density,
                            const double momentum_x,
                            const double momentum_y,
                            const double pressure,
                            const double gamma)
{
  return {density,
          momentum_x,
          momentum_y,
          pressure / (gamma - 1.0) +
              0.5 * (momentum_x * momentum_x +
                     momentum_y * momentum_y) /
                  density};
}

bool CheckStrictMeanAdmission()
{
  const zs::Parameters parameters{1.4, 0.1, 0.1, 48};
  const auto valid = State(1.0, 0.2, -0.1, 1.0, parameters.gamma);
  auto density_floor_mean = valid;
  density_floor_mean[0] = parameters.density_floor;
  const auto pressure_below_floor_mean =
      State(1.0, 0.0, 0.0, 0.5 * parameters.pressure_floor,
            parameters.gamma);
  auto nonfinite_mean = valid;
  nonfinite_mean[3] = std::numeric_limits<double>::quiet_NaN();

  bool passed = true;
  passed = Check(zs::ValidateAdmissibleMean(valid.data(), parameters) ==
                         zs::Status::success,
                     "valid mean is admitted") &&
      passed;
  passed = Check(
               zs::ValidateAdmissibleMean(density_floor_mean.data(),
                                          parameters) ==
                   zs::Status::inadmissible_mean_density,
               "mean on density floor is rejected without repair") &&
      passed;
  passed = Check(
               zs::ValidateAdmissibleMean(pressure_below_floor_mean.data(),
                                          parameters) ==
                   zs::Status::inadmissible_mean_pressure,
               "mean below pressure floor is rejected without repair") &&
      passed;
  passed = Check(zs::ValidateAdmissibleMean(nonfinite_mean.data(),
                                           parameters) ==
                         zs::Status::nonfinite_mean,
                     "nonfinite mean is rejected") &&
      passed;

  const zs::Parameters invalid_gamma{1.0, 0.1, 0.1, 12};
  const zs::Parameters invalid_iterations{1.4, 0.1, 0.1, 0};
  passed = Check(!zs::ValidParameters(invalid_gamma) &&
                         !zs::ValidParameters(invalid_iterations),
                     "invalid limiter parameters fail closed") &&
      passed;
  return passed;
}

bool CheckDensityAndPressureRoots()
{
  const zs::Parameters parameters{1.4, 0.1, 0.1, 52};
  const std::array<double, 4> mean{1.0, 0.0, 0.0, 2.5};
  const std::array<double, 4> density_point{-0.2, 0.0, 0.0, 1.0};
  const zs::ThetaResult density =
      zs::DensityTheta(mean.data(), density_point.data(), parameters);
  double density_limited[4]{};
  const zs::Status density_blend =
      zs::BlendDensityOnlyU4(mean.data(), density_point.data(), density.theta,
                            density_limited);

  bool passed = Check(density.ok() && Near(density.theta, 0.75, 4.0e-15) &&
                              density_blend == zs::Status::success &&
                              density_limited[0] >=
                                  parameters.density_floor &&
                              SameBits(density_limited[1], density_point[1]) &&
                              SameBits(density_limited[2], density_point[2]) &&
                              SameBits(density_limited[3], density_point[3]),
                          "density pass changes density alone at its root");

  const std::array<double, 4> pressure_point{0.4, 1.0, 0.0, 1.0};
  const zs::ThetaResult pressure =
      zs::PressureTheta(mean.data(), pressure_point.data(), parameters);
  double pressure_limited[4]{};
  const zs::Status pressure_blend =
      zs::BlendU4(mean.data(), pressure_point.data(), pressure.theta,
                  pressure_limited);
  double final_pressure = 0.0;
  const bool pressure_finite =
      zs::Pressure(pressure_limited, parameters.gamma, final_pressure);
  const double independent_root =
      (2.85 - std::sqrt(2.85 * 2.85 - 4.0 * 0.4 * 2.25)) /
      (2.0 * 0.4);
  const double bracket_width = std::ldexp(1.0, -parameters.pressure_bisections);
  passed = Check(pressure.ok() && pressure.active() &&
                         pressure_blend == zs::Status::success &&
                         pressure_finite &&
                         final_pressure >= parameters.pressure_floor &&
                         pressure.theta <= independent_root + bracket_width &&
                         std::abs(independent_root - pressure.theta) <=
                             2.0 * bracket_width,
                     "pressure pass returns the admissible bisection bracket") &&
      passed;
  return passed;
}

bool CheckSequentialElementScaling()
{
  constexpr int scalar_dofs = 4;
  constexpr int components = 7;
  const zs::Parameters parameters{1.4, 0.1, 0.1, 52};
  std::array<double, components * scalar_dofs> coefficients{
    -0.2, 1.4, 1.4, 1.4,
     1.0,-1.0, 0.0, 0.0,
     0.0, 0.0, 1.0,-1.0,
     1.0, 4.0, 2.5, 2.5,
    11.0,12.0,13.0,14.0,
    21.0,22.0,23.0,24.0,
    31.0,32.0,33.0,34.0
  };
  const auto before = coefficients;

  std::array<double, 4> mean{};
  for (int component = 0; component < 4; ++component)
  {
    for (int dof = 0; dof < scalar_dofs; ++dof)
    {
      mean[component] += coefficients[component * scalar_dofs + dof] /
                         static_cast<double>(scalar_dofs);
    }
  }

  const zs::ElementFactors factors = zs::ComputeElementFactors(
      mean.data(), coefficients.data(), scalar_dofs,
      /*point_stride=*/1, /*component_stride=*/scalar_dofs, parameters);
  const zs::ScaleResult scaled = zs::ScaleU4CoefficientsAboutMean(
      coefficients.data(), scalar_dofs, /*component_stride=*/scalar_dofs,
      /*dof_stride=*/1, mean.data(), factors, parameters);

  bool passed = Check(factors.ok() && factors.active() &&
                              factors.density_theta < 1.0 &&
                              factors.pressure_theta < 1.0 && scaled.ok() &&
                              scaled.modified &&
                              scaled.maximum_absolute_correction > 0.0,
                          "sequential factors activate and report correction");
  double density_limited_point[4]{};
  const double original_point[4] = {-0.2, 1.0, 0.0, 1.0};
  const zs::Status density_blend = zs::BlendDensityOnlyU4(
      mean.data(), original_point, factors.density_theta,
      density_limited_point);
  const double pressure_energy =
      parameters.pressure_floor / (parameters.gamma - 1.0);
  const double drho = density_limited_point[0] - mean[0];
  const double dmx = density_limited_point[1] - mean[1];
  const double dmy = density_limited_point[2] - mean[2];
  const double denergy = density_limited_point[3] - mean[3];
  const double mean_internal = mean[3] - pressure_energy;
  const double quadratic_a = drho * denergy -
      0.5 * (dmx * dmx + dmy * dmy);
  const double quadratic_b = mean[0] * denergy +
      drho * mean_internal - (mean[1] * dmx + mean[2] * dmy);
  const double quadratic_c = mean[0] * mean_internal -
      0.5 * (mean[1] * mean[1] + mean[2] * mean[2]);
  const double discriminant =
      quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c;
  const double independent_pressure_root =
      (-quadratic_b - std::sqrt(discriminant)) / (2.0 * quadratic_a);
  const double bracket_width =
      std::ldexp(1.0, -parameters.pressure_bisections);
  passed = Check(
               density_blend == zs::Status::success &&
                   factors.pressure_theta > 0.5 &&
                   factors.pressure_theta <=
                       independent_pressure_root + bracket_width &&
                   std::abs(independent_pressure_root -
                            factors.pressure_theta) <=
                       2.0 * bracket_width,
               "sequential pressure factor brackets independent root") &&
      passed;
  for (int component = 0; component < 4; ++component)
  {
    double limited_mean = 0.0;
    for (int dof = 0; dof < scalar_dofs; ++dof)
    {
      limited_mean += coefficients[component * scalar_dofs + dof] /
                      static_cast<double>(scalar_dofs);
    }
    passed = Check(Near(limited_mean, mean[component], 4.0e-15),
                   "coefficient scaling preserves the U4 mean") &&
        passed;
  }
  for (int dof = 0; dof < scalar_dofs; ++dof)
  {
    double point[4]{};
    for (int component = 0; component < 4; ++component)
    {
      point[component] = coefficients[component * scalar_dofs + dof];
    }
    double pressure = 0.0;
    passed = Check(point[0] >= parameters.density_floor &&
                           zs::Pressure(point, parameters.gamma, pressure) &&
                           pressure >= parameters.pressure_floor,
                       "scaled positivity point meets both floors") &&
        passed;
  }
  for (int component = 4; component < components; ++component)
  {
    for (int dof = 0; dof < scalar_dofs; ++dof)
    {
      const int index = component * scalar_dofs + dof;
      passed = Check(SameBits(coefficients[index], before[index]),
                     "IGR auxiliary coefficient remains bitwise untouched") &&
          passed;
    }
  }
  return passed;
}

bool CheckDensityOnlyScalingLeavesOtherFieldsExact()
{
  constexpr int scalar_dofs = 2;
  const zs::Parameters parameters{1.4, 0.1, 0.1, 48};
  std::array<double, 8> coefficients{
    -0.2, 2.2,
     0.0, 0.0,
     0.0, 0.0,
     2.5, 2.5
  };
  const auto before = coefficients;
  const std::array<double, 4> mean{1.0, 0.0, 0.0, 2.5};
  const zs::ElementFactors factors = zs::ComputeElementFactors(
      mean.data(), coefficients.data(), scalar_dofs,
      /*point_stride=*/1, /*component_stride=*/scalar_dofs, parameters);
  const zs::ScaleResult scaled = zs::ScaleU4CoefficientsAboutMean(
      coefficients.data(), scalar_dofs, scalar_dofs, 1, mean.data(), factors,
      parameters);

  bool passed = Check(factors.ok() && factors.density_theta < 1.0 &&
                              factors.pressure_theta == 1.0 && scaled.ok() &&
                              scaled.modified,
                          "density-only element activates only first factor");
  passed = Check(coefficients[0] != before[0] &&
                         coefficients[1] != before[1],
                     "density-only scaling changes density coefficients") &&
      passed;
  for (int component = 1; component < 4; ++component)
  {
    for (int dof = 0; dof < scalar_dofs; ++dof)
    {
      const int index = component * scalar_dofs + dof;
      passed = Check(SameBits(coefficients[index], before[index]),
                     "density-only scaling preserves momentum and energy") &&
          passed;
    }
  }
  return passed;
}

bool CheckFailureIsAtomicAndIdentityIsExact()
{
  const zs::Parameters parameters{1.4, 0.1, 0.1, 24};
  const std::array<double, 4> bad_mean{0.1, 0.0, 0.0, 2.5};
  std::array<double, 8> coefficients{1.0, 2.0, 0.0, 0.0,
                                      0.0, 0.0, 2.5, 2.5};
  const auto before = coefficients;
  const zs::ElementFactors rejected = zs::ComputeElementFactors(
      bad_mean.data(), coefficients.data(), 2,
      /*point_stride=*/1, /*component_stride=*/2, parameters);
  const zs::ScaleResult rejected_scale = zs::ScaleU4CoefficientsAboutMean(
      coefficients.data(), 2, 2, 1, bad_mean.data(), rejected, parameters);

  bool passed = Check(
      rejected.status == zs::Status::inadmissible_mean_density &&
          rejected.density_coefficient_theta() == 1.0 &&
          rejected.other_coefficient_theta() == 1.0 &&
          rejected_scale.status == zs::Status::inadmissible_mean_density &&
          !rejected_scale.modified && SameBits(coefficients, before),
      "inadmissible mean cannot repair or mutate coefficients");

  const zs::ElementFactors active{zs::Status::success, 0.5, 0.5, -1};
  const zs::ScaleResult forced_bad_mean =
      zs::ScaleU4CoefficientsAboutMean(coefficients.data(), 2, 2, 1,
                                      bad_mean.data(), active, parameters);
  passed = Check(
               forced_bad_mean.status ==
                       zs::Status::inadmissible_mean_density &&
                   !forced_bad_mean.modified && SameBits(coefficients, before),
               "scaling independently rejects an inadmissible mean") &&
      passed;

  const auto good_mean = State(1.5, 0.0, 0.0, 1.0, parameters.gamma);
  const zs::ElementFactors identity{zs::Status::success, 1.0, 1.0, -1};
  const zs::ScaleResult identity_scale = zs::ScaleU4CoefficientsAboutMean(
      coefficients.data(), 2, 2, 1, good_mean.data(), identity, parameters);
  passed = Check(identity_scale.ok() && !identity_scale.modified &&
                         identity_scale.maximum_absolute_correction == 0.0 &&
                         SameBits(coefficients, before),
                     "identity factors are a bitwise no-op") &&
      passed;

  auto nonfinite_coefficients = coefficients;
  nonfinite_coefficients[7] = std::numeric_limits<double>::infinity();
  const auto nonfinite_before = nonfinite_coefficients;
  const zs::ScaleResult nonfinite_scale =
      zs::ScaleU4CoefficientsAboutMean(nonfinite_coefficients.data(), 2, 2, 1,
                                      good_mean.data(), active, parameters);
  passed = Check(nonfinite_scale.status == zs::Status::nonfinite_point &&
                         !nonfinite_scale.modified &&
                         SameBits(nonfinite_coefficients, nonfinite_before),
                     "coefficient validation prevents partial mutation") &&
      passed;

  auto overlapping_coefficients = coefficients;
  const auto overlapping_before = overlapping_coefficients;
  const zs::ScaleResult overlapping_scale =
      zs::ScaleU4CoefficientsAboutMean(overlapping_coefficients.data(), 2,
                                      /*component_stride=*/1,
                                      /*dof_stride=*/1, good_mean.data(),
                                      active, parameters);
  passed = Check(overlapping_scale.status == zs::Status::invalid_layout &&
                         !overlapping_scale.modified &&
                         SameBits(overlapping_coefficients,
                                  overlapping_before),
                     "overlapping coefficient strides fail atomically") &&
      passed;

  const zs::ElementFactors malformed{zs::Status::success, 2.0, 0.5, -1};
  const zs::ScaleResult malformed_scale =
      zs::ScaleU4CoefficientsAboutMean(coefficients.data(), 2, 2, 1,
                                      good_mean.data(), malformed, parameters);
  passed = Check(malformed_scale.status == zs::Status::invalid_parameters &&
                         !malformed_scale.modified &&
                         SameBits(coefficients, before),
                     "raw factors outside the unit interval fail atomically") &&
      passed;
  return passed;
}

}  // namespace

int main()
{
  bool passed = true;
  passed = CheckStrictMeanAdmission() && passed;
  passed = CheckDensityAndPressureRoots() && passed;
  passed = CheckSequentialElementScaling() && passed;
  passed = CheckDensityOnlyScalingLeavesOtherFieldsExact() && passed;
  passed = CheckFailureIsAtomicAndIdentityIsExact() && passed;
  if (passed) { std::cout << "Zhang--Shu U4 scalar core checks passed\n"; }
  return passed ? 0 : 1;
}

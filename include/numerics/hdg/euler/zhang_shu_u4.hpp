#pragma once

/**
 * @file numerics/hdg/euler/zhang_shu_u4.hpp
 * @brief Backend-neutral Zhang--Shu scalar algebra for two-dimensional U4.
 *
 * This header owns only the element-local scalar mathematics.  A lifecycle
 * consumer remains responsible for constructing a cell average with its
 * physical quadrature, evaluating every admitted positivity point, updating
 * only the four differential Euler fields, and rebuilding/checking the HDG
 * trace and any IGR algebraic fields after a nontrivial correction.
 *
 * The algorithm performs the Zhang--Shu density pass first, modifying density
 * alone.  It then uses those density-limited point states to locate the
 * pressure-floor root by a conservative bisection.  The pressure factor scales
 * all four conservative fields; consequently density receives the product of
 * the two factors while momentum and total energy receive only the pressure
 * factor.  Scaling coefficient deviations from the same physical cell average
 * preserves that average for a constant-preserving nodal basis.
 *
 * An inadmissible cell average is a hard failure.  No function in this header
 * repairs, clamps, or mutates such an average or its coefficients.
 */

#include <cstddef>
#include <cstdint>

#ifndef DG_IGR_HDG_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define DG_IGR_HDG_HOST_DEVICE __host__ __device__
#else
#define DG_IGR_HDG_HOST_DEVICE
#endif
#endif

namespace dg::hdg::euler::zhang_shu
{

inline constexpr int u4_components = 4;
inline constexpr int maximum_pressure_bisections = 64;
inline constexpr double max_finite_double = 0x1.fffffffffffffp+1023;
inline constexpr double machine_epsilon = 0x1p-52;

enum class Status : std::uint8_t
{
  success,
  invalid_parameters,
  invalid_layout,
  nonfinite_mean,
  inadmissible_mean_density,
  inadmissible_mean_pressure,
  nonfinite_point,
  inadmissible_density_point,
  nonfinite_result
};

struct Parameters final
{
  double gamma = 1.4;
  double density_floor = 1.0e-12;
  double pressure_floor = 1.0e-12;
  int pressure_bisections = 12;
};

struct ThetaResult final
{
  Status status = Status::invalid_parameters;
  double theta = 1.0;

  DG_IGR_HDG_HOST_DEVICE constexpr bool ok() const noexcept
  {
    return status == Status::success;
  }

  DG_IGR_HDG_HOST_DEVICE constexpr bool active() const noexcept
  {
    return ok() && theta < 1.0;
  }
};

/** Two-pass element factors over one caller-supplied positivity-point set. */
struct ElementFactors final
{
  Status status = Status::invalid_parameters;
  double density_theta = 1.0;
  double pressure_theta = 1.0;
  int failed_point = -1;

  DG_IGR_HDG_HOST_DEVICE constexpr bool ok() const noexcept
  {
    return status == Status::success;
  }

  DG_IGR_HDG_HOST_DEVICE constexpr bool active() const noexcept
  {
    return ok() && (density_theta < 1.0 || pressure_theta < 1.0);
  }

  /** Failure deliberately maps to identity; callers must still check ok(). */
  DG_IGR_HDG_HOST_DEVICE constexpr double density_coefficient_theta()
      const noexcept
  {
    return ok() ? density_theta * pressure_theta : 1.0;
  }

  /** Momentum and energy are unchanged by the density-only first pass. */
  DG_IGR_HDG_HOST_DEVICE constexpr double other_coefficient_theta()
      const noexcept
  {
    return ok() ? pressure_theta : 1.0;
  }

  DG_IGR_HDG_HOST_DEVICE constexpr double component_theta(
      const int component) const noexcept
  {
    return component == 0 ? density_coefficient_theta()
                          : other_coefficient_theta();
  }
};

struct ScaleResult final
{
  Status status = Status::invalid_parameters;
  double maximum_absolute_correction = 0.0;
  bool modified = false;

  DG_IGR_HDG_HOST_DEVICE constexpr bool ok() const noexcept
  {
    return status == Status::success;
  }
};

DG_IGR_HDG_HOST_DEVICE constexpr bool Finite(const double value) noexcept
{
  return value >= -max_finite_double && value <= max_finite_double;
}

DG_IGR_HDG_HOST_DEVICE constexpr double Absolute(
    const double value) noexcept
{
  return value < 0.0 ? -value : value;
}

DG_IGR_HDG_HOST_DEVICE constexpr double Minimum(const double left,
                                                const double right) noexcept
{
  return left < right ? left : right;
}

DG_IGR_HDG_HOST_DEVICE constexpr double Maximum(const double left,
                                                const double right) noexcept
{
  return left > right ? left : right;
}

DG_IGR_HDG_HOST_DEVICE constexpr bool ValidParameters(
    const Parameters& parameters) noexcept
{
  return Finite(parameters.gamma) && parameters.gamma > 1.0 &&
      Finite(parameters.density_floor) && parameters.density_floor > 0.0 &&
      Finite(parameters.pressure_floor) && parameters.pressure_floor > 0.0 &&
      parameters.pressure_bisections > 0 &&
      parameters.pressure_bisections <= maximum_pressure_bisections;
}

DG_IGR_HDG_HOST_DEVICE inline bool FiniteU4(const double* const state) noexcept
{
  if (state == nullptr) { return false; }
  for (int component = 0; component < u4_components; ++component)
  {
    if (!Finite(state[component])) { return false; }
  }
  return true;
}

/**
 * Evaluate ideal-gas pressure without an artificial density denominator.
 * The output is unchanged if the input cannot produce a finite pressure.
 */
DG_IGR_HDG_HOST_DEVICE inline bool Pressure(
    const double* const state,
    const double gamma,
    double& pressure) noexcept
{
  if (!FiniteU4(state) || !Finite(gamma) || !(gamma > 1.0) ||
      !(state[0] > 0.0))
  {
    return false;
  }

  const double momentum_scale =
      Maximum(Absolute(state[1]), Absolute(state[2]));
  double kinetic = 0.0;
  if (momentum_scale > 0.0)
  {
    const double scaled_x = state[1] / momentum_scale;
    const double scaled_y = state[2] / momentum_scale;
    const double scale_over_density = momentum_scale / state[0];
    kinetic = 0.5 * momentum_scale * scale_over_density *
        (scaled_x * scaled_x + scaled_y * scaled_y);
  }
  const double candidate = (gamma - 1.0) * (state[3] - kinetic);
  if (!Finite(kinetic) || !Finite(candidate)) { return false; }
  pressure = candidate;
  return true;
}

/** Strictly validate the physical cell mean; this routine never repairs it. */
DG_IGR_HDG_HOST_DEVICE inline Status ValidateAdmissibleMean(
    const double* const mean,
    const Parameters& parameters) noexcept
{
  if (!ValidParameters(parameters)) { return Status::invalid_parameters; }
  if (!FiniteU4(mean)) { return Status::nonfinite_mean; }
  if (!(mean[0] > parameters.density_floor))
  {
    return Status::inadmissible_mean_density;
  }
  double pressure = 0.0;
  if (!Pressure(mean, parameters.gamma, pressure))
  {
    return Status::nonfinite_mean;
  }
  if (!(pressure > parameters.pressure_floor))
  {
    return Status::inadmissible_mean_pressure;
  }
  return Status::success;
}

/**
 * First-pass coefficient for one point.  The scaled-ratio form avoids an
 * overflow in mean_density - point_density for opposite extreme values.
 */
DG_IGR_HDG_HOST_DEVICE inline ThetaResult DensityTheta(
    const double* const mean,
    const double* const point,
    const Parameters& parameters) noexcept
{
  const Status mean_status = ValidateAdmissibleMean(mean, parameters);
  if (mean_status != Status::success) { return {mean_status, 1.0}; }
  if (!FiniteU4(point)) { return {Status::nonfinite_point, 1.0}; }
  if (point[0] >= parameters.density_floor)
  {
    return {Status::success, 1.0};
  }

  const double scale = Maximum(
      Maximum(Absolute(mean[0]), Absolute(point[0])),
      Absolute(parameters.density_floor));
  if (!(scale > 0.0) || !Finite(scale))
  {
    return {Status::nonfinite_result, 1.0};
  }
  const double scaled_mean = mean[0] / scale;
  const double scaled_point = point[0] / scale;
  const double scaled_floor = parameters.density_floor / scale;
  const double denominator = scaled_mean - scaled_point;
  if (!(denominator > 0.0) || !Finite(denominator))
  {
    return {Status::nonfinite_result, 1.0};
  }
  double theta = (scaled_mean - scaled_floor) / denominator;
  if (!Finite(theta)) { return {Status::nonfinite_result, 1.0}; }
  theta = Maximum(0.0, Minimum(1.0, theta));

  // The exact affine root can round to the inadmissible side when evaluated
  // in the blend's operation order.  Retreat by a few ulps and fail closed to
  // the admissible mean if that rare retreat itself cannot be represented.
  double limited_density =
      (1.0 - theta) * mean[0] + theta * point[0];
  if (!Finite(limited_density))
  {
    return {Status::nonfinite_result, 1.0};
  }
  if (limited_density < parameters.density_floor)
  {
    const double retreated = theta * (1.0 - 16.0 * machine_epsilon);
    theta = retreated < theta ? retreated : 0.0;
    limited_density = (1.0 - theta) * mean[0] + theta * point[0];
    if (!Finite(limited_density))
    {
      return {Status::nonfinite_result, 1.0};
    }
    if (limited_density < parameters.density_floor) { theta = 0.0; }
  }
  return {Status::success, theta};
}

/** Atomic affine blend: failed calls leave output untouched. */
DG_IGR_HDG_HOST_DEVICE inline Status BlendU4(
    const double* const mean,
    const double* const point,
    const double theta,
    double* const output) noexcept
{
  if (output == nullptr || !FiniteU4(mean) || !FiniteU4(point) ||
      !Finite(theta) || theta < 0.0 || theta > 1.0)
  {
    return Status::invalid_layout;
  }

  double candidate[u4_components];
  if (theta == 1.0)
  {
    for (int component = 0; component < u4_components; ++component)
    {
      candidate[component] = point[component];
    }
  }
  else if (theta == 0.0)
  {
    for (int component = 0; component < u4_components; ++component)
    {
      candidate[component] = mean[component];
    }
  }
  else
  {
    const double mean_weight = 1.0 - theta;
    for (int component = 0; component < u4_components; ++component)
    {
      candidate[component] =
          mean_weight * mean[component] + theta * point[component];
      if (!Finite(candidate[component])) { return Status::nonfinite_result; }
    }
  }
  for (int component = 0; component < u4_components; ++component)
  {
    output[component] = candidate[component];
  }
  return Status::success;
}

/**
 * Apply the first Zhang--Shu pass to density only.  Momentum and total energy
 * are copied bitwise from the point.  Failed calls leave output untouched.
 */
DG_IGR_HDG_HOST_DEVICE inline Status BlendDensityOnlyU4(
    const double* const mean,
    const double* const point,
    const double density_theta,
    double* const output) noexcept
{
  if (output == nullptr || !FiniteU4(mean) || !FiniteU4(point) ||
      !Finite(density_theta) || density_theta < 0.0 || density_theta > 1.0)
  {
    return Status::invalid_layout;
  }

  double density = point[0];
  if (density_theta == 0.0)
  {
    density = mean[0];
  }
  else if (density_theta != 1.0)
  {
    density = (1.0 - density_theta) * mean[0] +
        density_theta * point[0];
    if (!Finite(density)) { return Status::nonfinite_result; }
  }
  output[0] = density;
  for (int component = 1; component < u4_components; ++component)
  {
    output[component] = point[component];
  }
  return Status::success;
}

/**
 * Locate the pressure-floor root for a point after the density pass.  The
 * returned lower bracket is admissible under the same pressure evaluation.
 */
DG_IGR_HDG_HOST_DEVICE inline ThetaResult PressureTheta(
    const double* const mean,
    const double* const density_limited_point,
    const Parameters& parameters) noexcept
{
  const Status mean_status = ValidateAdmissibleMean(mean, parameters);
  if (mean_status != Status::success) { return {mean_status, 1.0}; }
  if (!FiniteU4(density_limited_point))
  {
    return {Status::nonfinite_point, 1.0};
  }
  if (!(density_limited_point[0] >= parameters.density_floor))
  {
    return {Status::inadmissible_density_point, 1.0};
  }

  double point_pressure = 0.0;
  if (Pressure(density_limited_point, parameters.gamma, point_pressure) &&
      point_pressure >= parameters.pressure_floor)
  {
    return {Status::success, 1.0};
  }

  double lower = 0.0;
  double upper = 1.0;
  for (int iteration = 0;
       iteration < parameters.pressure_bisections;
       ++iteration)
  {
    const double middle = 0.5 * (lower + upper);
    double candidate[u4_components];
    const Status blend_status =
        BlendU4(mean, density_limited_point, middle, candidate);
    double candidate_pressure = 0.0;
    if (blend_status == Status::success &&
        Pressure(candidate, parameters.gamma, candidate_pressure) &&
        candidate_pressure >= parameters.pressure_floor)
    {
      lower = middle;
    }
    else
    {
      upper = middle;
    }
  }
  return {Status::success, lower};
}

/**
 * Compute the exact two-pass element factors from an arbitrary strided point
 * table.  Point q, component c is points[q*point_stride+c*component_stride].
 * The table is read-only, including on every failure path.
 */
DG_IGR_HDG_HOST_DEVICE inline ElementFactors ComputeElementFactors(
    const double* const mean,
    const double* const points,
    const int point_count,
    const int point_stride,
    const int component_stride,
    const Parameters& parameters) noexcept
{
  ElementFactors factors{};
  factors.status = ValidateAdmissibleMean(mean, parameters);
  if (factors.status != Status::success) { return factors; }
  if (points == nullptr || point_count <= 0 || point_stride <= 0 ||
      component_stride <= 0)
  {
    factors.status = Status::invalid_layout;
    return factors;
  }

  for (int point_index = 0; point_index < point_count; ++point_index)
  {
    double point[u4_components];
    for (int component = 0; component < u4_components; ++component)
    {
      const std::int64_t index =
          static_cast<std::int64_t>(point_index) * point_stride +
          static_cast<std::int64_t>(component) * component_stride;
      point[component] = points[index];
    }
    const ThetaResult point_result = DensityTheta(mean, point, parameters);
    if (!point_result.ok())
    {
      factors.status = point_result.status;
      factors.failed_point = point_index;
      return factors;
    }
    factors.density_theta =
        Minimum(factors.density_theta, point_result.theta);
  }

  for (int point_index = 0; point_index < point_count; ++point_index)
  {
    double point[u4_components];
    double density_limited_point[u4_components];
    for (int component = 0; component < u4_components; ++component)
    {
      const std::int64_t index =
          static_cast<std::int64_t>(point_index) * point_stride +
          static_cast<std::int64_t>(component) * component_stride;
      point[component] = points[index];
    }
    const Status blend_status = BlendDensityOnlyU4(
        mean, point, factors.density_theta, density_limited_point);
    if (blend_status != Status::success)
    {
      factors.status = blend_status;
      factors.failed_point = point_index;
      return factors;
    }
    const ThetaResult point_result =
        PressureTheta(mean, density_limited_point, parameters);
    if (!point_result.ok())
    {
      factors.status = point_result.status;
      factors.failed_point = point_index;
      return factors;
    }
    factors.pressure_theta =
        Minimum(factors.pressure_theta, point_result.theta);
  }
  factors.status = Status::success;
  return factors;
}

/**
 * Scale only the first four component slabs of a strided coefficient table.
 * Coefficient (c,i) is coefficients[c*component_stride+i*dof_stride].
 *
 * For an IGR interior table, pass the leading U4 slabs.  Auxiliary components
 * are neither inspected nor modified.  Component slabs must not overlap:
 * component_stride must span at least scalar_dofs*dof_stride entries.  The
 * operation is atomic with respect to detectable failures: it validates every
 * result before its write pass.
 */
DG_IGR_HDG_HOST_DEVICE inline ScaleResult ScaleU4CoefficientsAboutMean(
    double* const coefficients,
    const int scalar_dofs,
    const int component_stride,
    const int dof_stride,
    const double* const mean,
    const ElementFactors& factors,
    const Parameters& parameters) noexcept
{
  ScaleResult result{};
  result.status = factors.status;
  if (!factors.ok()) { return result; }
  const std::int64_t component_span =
      static_cast<std::int64_t>(scalar_dofs) * dof_stride;
  if (coefficients == nullptr || scalar_dofs <= 0 || component_stride <= 0 ||
      dof_stride <= 0 || component_stride < component_span)
  {
    result.status = Status::invalid_layout;
    return result;
  }
  result.status = ValidateAdmissibleMean(mean, parameters);
  if (result.status != Status::success) { return result; }

  double local_mean[u4_components];
  for (int component = 0; component < u4_components; ++component)
  {
    local_mean[component] = mean[component];
  }

  if (!Finite(factors.density_theta) || factors.density_theta < 0.0 ||
      factors.density_theta > 1.0 || !Finite(factors.pressure_theta) ||
      factors.pressure_theta < 0.0 || factors.pressure_theta > 1.0)
  {
    result.status = Status::invalid_parameters;
    return result;
  }
  const double density_theta = factors.density_coefficient_theta();
  const double other_theta = factors.other_coefficient_theta();
  if (density_theta == 1.0 && other_theta == 1.0)
  {
    result.status = Status::success;
    return result;
  }

  double maximum_correction = 0.0;
  for (int component = 0; component < u4_components; ++component)
  {
    for (int dof = 0; dof < scalar_dofs; ++dof)
    {
      const std::int64_t index =
          static_cast<std::int64_t>(component) * component_stride +
          static_cast<std::int64_t>(dof) * dof_stride;
      const double value = coefficients[index];
      if (!Finite(value))
      {
        result.status = Status::nonfinite_point;
        return result;
      }
      double scaled = value;
      if (component == 0 && factors.density_theta != 1.0)
      {
        scaled = (1.0 - factors.density_theta) * local_mean[component] +
            factors.density_theta * scaled;
      }
      if (factors.pressure_theta != 1.0)
      {
        scaled = (1.0 - factors.pressure_theta) * local_mean[component] +
            factors.pressure_theta * scaled;
      }
      const double correction = Absolute(scaled - value);
      if (!Finite(scaled) || !Finite(correction))
      {
        result.status = Status::nonfinite_result;
        return result;
      }
      maximum_correction = Maximum(maximum_correction, correction);
    }
  }

  for (int component = 0; component < u4_components; ++component)
  {
    for (int dof = 0; dof < scalar_dofs; ++dof)
    {
      const std::int64_t index =
          static_cast<std::int64_t>(component) * component_stride +
          static_cast<std::int64_t>(dof) * dof_stride;
      if (component == 0 && factors.density_theta != 1.0)
      {
        coefficients[index] =
            (1.0 - factors.density_theta) * local_mean[component] +
            factors.density_theta * coefficients[index];
      }
      if (factors.pressure_theta != 1.0)
      {
        coefficients[index] =
            (1.0 - factors.pressure_theta) * local_mean[component] +
            factors.pressure_theta * coefficients[index];
      }
    }
  }
  result.status = Status::success;
  result.maximum_absolute_correction = maximum_correction;
  result.modified = maximum_correction > 0.0;
  return result;
}

}  // namespace dg::hdg::euler::zhang_shu

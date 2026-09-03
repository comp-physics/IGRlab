#pragma once

#include "mfem/general/annotation.hpp"

#include <cmath>
#include <limits>

/**
 * @file physics/bc/detail/euler_boundary_state.hpp
 * @brief Internal host/device Euler exterior-state closure shared by DG and HDG.
 */

namespace mfem::detail
{

enum class EulerBoundaryStateKind : int
{
  unsupported = 0,
  supersonic_inlet = 1,
  supersonic_outlet = 2,
  subsonic_outlet_pressure = 3,
  nonreflecting_outlet = 4,
  subsonic_inlet = 5,
  far_field = 6,
  slip_wall = 7
};

template <typename Scalar>
struct EulerPrimitiveNormalState
{
  Scalar rho = Scalar(1.0);
  Scalar pressure = Scalar(1.0);
  Scalar normal_velocity = Scalar(0.0);
  Scalar tangential_velocity = Scalar(0.0);
  Scalar sound_speed = Scalar(1.0);
};

template <typename Scalar>
MFEM_HOST_DEVICE inline bool EulerFinitePositive(const Scalar value)
{
  // Ordered comparisons reject NaN; the upper bound rejects infinity on both
  // host and device without relying on a backend-specific isfinite overload.
  return value > Scalar(0.0) &&
         value <= std::numeric_limits<Scalar>::max();
}

template <typename Scalar>
MFEM_HOST_DEVICE inline bool EulerConservativeStateMeetsContract(
    const Scalar* state, const int dimension, const int equations,
    const Scalar gamma, const Scalar density_floor,
    const Scalar pressure_floor)
{
  if (!state || equations != dimension + 2 || dimension < 1 || dimension > 3 ||
      !EulerFinitePositive(gamma - Scalar(1.0)) ||
      !EulerFinitePositive(density_floor) ||
      !EulerFinitePositive(pressure_floor))
    return false;
  for (int equation = 0; equation < equations; ++equation)
  {
    const Scalar value = state[equation];
    if (!(value >= -std::numeric_limits<Scalar>::max()) ||
        value > std::numeric_limits<Scalar>::max())
      return false;
  }
  const Scalar density = state[0];
  if (!(density >= density_floor))
    return false;
  Scalar momentum_squared = Scalar(0.0);
  for (int d = 0; d < dimension; ++d)
    momentum_squared += state[1 + d] * state[1 + d];
  const Scalar pressure = (gamma - Scalar(1.0)) *
      (state[1 + dimension] -
       Scalar(0.5) * momentum_squared / density);
  return pressure >= pressure_floor &&
         pressure <= std::numeric_limits<Scalar>::max();
}

template <typename Scalar>
MFEM_HOST_DEVICE inline void EulerBuildNormalTangential(
    const Scalar* normal, const int dimension, Scalar* normal_unit,
    Scalar* tangent_unit)
{
  Scalar norm_squared = Scalar(0.0);
  for (int d = 0; d < dimension; ++d)
  {
    normal_unit[d] = normal[d];
    norm_squared += normal[d] * normal[d];
    tangent_unit[d] = Scalar(0.0);
  }
  const Scalar norm = sqrt(norm_squared);
  if (norm > Scalar(0.0))
  {
    const Scalar inverse_norm = Scalar(1.0) / norm;
    for (int d = 0; d < dimension; ++d)
      normal_unit[d] *= inverse_norm;
  }
  else
  {
    for (int d = 0; d < dimension; ++d)
      normal_unit[d] = Scalar(0.0);
    normal_unit[0] = Scalar(1.0);
  }

  if (dimension == 2)
  {
    tangent_unit[0] = -normal_unit[1];
    tangent_unit[1] = normal_unit[0];
  }
}

template <typename Scalar>
MFEM_HOST_DEVICE inline EulerPrimitiveNormalState<Scalar>
EulerConservativeToPrimitiveNormal(const Scalar* conservative,
                                   const int dimension,
                                   const Scalar gamma,
                                   const Scalar* normal_unit,
                                   const Scalar* tangent_unit,
                                   const Scalar density_floor,
                                   const Scalar pressure_floor)
{
  EulerPrimitiveNormalState<Scalar> result;
  const Scalar density = fmax(conservative[0], density_floor);
  Scalar velocity[3] = {Scalar(0.0), Scalar(0.0), Scalar(0.0)};
  Scalar velocity_squared = Scalar(0.0);
  for (int d = 0; d < dimension; ++d)
  {
    velocity[d] = conservative[1 + d] / density;
    velocity_squared += velocity[d] * velocity[d];
  }

  const Scalar specific_energy = conservative[1 + dimension] / density;
  const Scalar kinetic_energy = Scalar(0.5) * velocity_squared;
  const Scalar pressure = fmax(
      (gamma - Scalar(1.0)) * density *
          (specific_energy - kinetic_energy),
      pressure_floor);
  const Scalar sound_speed =
      sqrt(fmax(gamma * pressure / density, Scalar(0.0)));

  result.rho = density;
  result.pressure = pressure;
  result.sound_speed = sound_speed;
  result.normal_velocity = Scalar(0.0);
  result.tangential_velocity = Scalar(0.0);
  for (int d = 0; d < dimension; ++d)
    result.normal_velocity += velocity[d] * normal_unit[d];
  if (dimension == 2)
    result.tangential_velocity =
        velocity[0] * tangent_unit[0] + velocity[1] * tangent_unit[1];
  return result;
}

template <typename Scalar>
MFEM_HOST_DEVICE inline void EulerPrimitiveNormalToConservative(
    const EulerPrimitiveNormalState<Scalar>& primitive,
    const int dimension, const Scalar gamma, const Scalar* normal_unit,
    const Scalar* tangent_unit, Scalar* conservative)
{
  Scalar velocity[3] = {Scalar(0.0), Scalar(0.0), Scalar(0.0)};
  if (dimension == 1)
    velocity[0] = primitive.normal_velocity;
  else
  {
    velocity[0] = primitive.normal_velocity * normal_unit[0] +
        primitive.tangential_velocity * tangent_unit[0];
    velocity[1] = primitive.normal_velocity * normal_unit[1] +
        primitive.tangential_velocity * tangent_unit[1];
  }

  conservative[0] = primitive.rho;
  Scalar velocity_squared = Scalar(0.0);
  for (int d = 0; d < dimension; ++d)
  {
    conservative[1 + d] = primitive.rho * velocity[d];
    velocity_squared += velocity[d] * velocity[d];
  }
  const Scalar specific_energy =
      primitive.pressure /
          ((gamma - Scalar(1.0)) * primitive.rho) +
      Scalar(0.5) * velocity_squared;
  conservative[1 + dimension] = primitive.rho * specific_energy;
}

template <typename Scalar>
MFEM_HOST_DEVICE inline Scalar EulerEntropyLike(
    const Scalar pressure, const Scalar density, const Scalar gamma,
    const Scalar density_floor)
{
  return pressure / pow(fmax(density, density_floor), gamma);
}

template <typename Scalar>
MFEM_HOST_DEVICE inline EulerPrimitiveNormalState<Scalar>
EulerSubsonicOutletPressureClosure(
    const EulerPrimitiveNormalState<Scalar>& inner, const Scalar gamma,
    const Scalar pressure_out, const Scalar density_floor,
    const Scalar pressure_floor)
{
  EulerPrimitiveNormalState<Scalar> result = inner;
  if (inner.normal_velocity >= inner.sound_speed ||
      inner.normal_velocity <= -inner.sound_speed)
    return result;

  const Scalar gamma_minus_one = gamma - Scalar(1.0);
  const Scalar entropy =
      EulerEntropyLike(inner.pressure, inner.rho, gamma, density_floor);
  const Scalar pressure = fmax(pressure_out, pressure_floor);
  const Scalar density = pow(pressure / entropy, Scalar(1.0) / gamma);
  const Scalar sound_speed =
      sqrt(fmax(gamma * pressure / density, Scalar(0.0)));
  const Scalar outgoing_invariant = inner.normal_velocity +
      Scalar(2.0) * inner.sound_speed / gamma_minus_one;

  result.rho = fmax(density, density_floor);
  result.pressure = pressure;
  result.sound_speed = sound_speed;
  result.normal_velocity = outgoing_invariant -
      Scalar(2.0) * sound_speed / gamma_minus_one;
  result.tangential_velocity = inner.tangential_velocity;
  return result;
}

template <typename Scalar>
MFEM_HOST_DEVICE inline EulerPrimitiveNormalState<Scalar>
EulerNonreflectingOutletClosure(
    const EulerPrimitiveNormalState<Scalar>& inner,
    const EulerPrimitiveNormalState<Scalar>& reference,
    const Scalar gamma, const Scalar density_floor,
    const Scalar pressure_floor)
{
  EulerPrimitiveNormalState<Scalar> result = inner;
  if (inner.normal_velocity >= inner.sound_speed ||
      inner.normal_velocity <= -inner.sound_speed)
    return result;

  const Scalar gamma_minus_one = gamma - Scalar(1.0);
  const Scalar outgoing_invariant = inner.normal_velocity +
      Scalar(2.0) * inner.sound_speed / gamma_minus_one;
  const Scalar incoming_invariant = reference.normal_velocity -
      Scalar(2.0) * reference.sound_speed / gamma_minus_one;
  const Scalar normal_velocity =
      Scalar(0.5) * (outgoing_invariant + incoming_invariant);
  const Scalar sound_speed = Scalar(0.25) * gamma_minus_one *
      (outgoing_invariant - incoming_invariant);
  if (!(sound_speed > Scalar(0.0)))
    return result;

  const Scalar entropy =
      EulerEntropyLike(reference.pressure,
                       reference.rho,
                       gamma,
                       density_floor);
  const Scalar entropy_scale = gamma * entropy;
  if (!EulerFinitePositive(entropy) ||
      !EulerFinitePositive(entropy_scale))
    return result;
  const Scalar density = pow(
      (sound_speed * sound_speed) / entropy_scale,
      Scalar(1.0) / gamma_minus_one);
  if (!EulerFinitePositive(density))
    return result;
  const Scalar admitted_density = fmax(density, density_floor);
  const Scalar pressure =
      entropy * pow(admitted_density, gamma);
  if (!(pressure >= Scalar(0.0)) ||
      pressure > std::numeric_limits<Scalar>::max())
    return result;

  result.rho = admitted_density;
  result.pressure = fmax(pressure, pressure_floor);
  result.sound_speed = sound_speed;
  result.normal_velocity = normal_velocity;
  result.tangential_velocity = inner.tangential_velocity;
  return result;
}

template <typename Scalar>
MFEM_HOST_DEVICE inline EulerPrimitiveNormalState<Scalar>
EulerSubsonicInletClosure(
    const EulerPrimitiveNormalState<Scalar>& inner,
    const EulerPrimitiveNormalState<Scalar>& target, const Scalar gamma,
    const Scalar density_floor, const Scalar pressure_floor)
{
  EulerPrimitiveNormalState<Scalar> result = target;
  if (inner.normal_velocity <= -inner.sound_speed)
    return result;
  if (inner.normal_velocity >= inner.sound_speed)
    return inner;

  const Scalar gamma_minus_one = gamma - Scalar(1.0);
  const Scalar outgoing_invariant = inner.normal_velocity +
      Scalar(2.0) * inner.sound_speed / gamma_minus_one;
  const Scalar incoming_invariant = target.normal_velocity -
      Scalar(2.0) * target.sound_speed / gamma_minus_one;
  const Scalar normal_velocity =
      Scalar(0.5) * (outgoing_invariant + incoming_invariant);
  const Scalar sound_speed = Scalar(0.25) * gamma_minus_one *
      (outgoing_invariant - incoming_invariant);
  if (!(sound_speed > Scalar(0.0)))
    return target;

  const Scalar entropy =
      EulerEntropyLike(target.pressure, target.rho, gamma, density_floor);
  const Scalar entropy_scale = gamma * entropy;
  if (!EulerFinitePositive(entropy) ||
      !EulerFinitePositive(entropy_scale))
    return target;
  const Scalar density = pow(
      (sound_speed * sound_speed) / entropy_scale,
      Scalar(1.0) / gamma_minus_one);
  if (!EulerFinitePositive(density))
    return target;
  const Scalar admitted_density = fmax(density, density_floor);
  const Scalar pressure =
      entropy * pow(admitted_density, gamma);
  if (!(pressure >= Scalar(0.0)) ||
      pressure > std::numeric_limits<Scalar>::max())
    return target;

  result.rho = admitted_density;
  result.pressure = fmax(pressure, pressure_floor);
  result.sound_speed = sound_speed;
  result.normal_velocity = normal_velocity;
  result.tangential_velocity = target.tangential_velocity;
  return result;
}

/**
 * Pure Euler closure. Callers may supply the admitted density/pressure floors
 * used by characteristic reconstruction; legacy callers retain 1e-12.
 * Supersonic copy branches still require the supplied state to be preflighted.
 */
template <typename Scalar>
MFEM_HOST_DEVICE inline bool EvaluateEulerBoundaryState(
    const EulerBoundaryStateKind kind, const int dimension,
    const int equations, const Scalar gamma, const Scalar pressure_out,
    const Scalar* reference_state, const Scalar* normal,
    const Scalar* interior, Scalar* exterior,
    const Scalar density_floor = Scalar(1.0e-12),
    const Scalar pressure_floor = Scalar(1.0e-12))
{
  if (equations != dimension + 2)
    return false;

  if (kind == EulerBoundaryStateKind::supersonic_inlet)
  {
    for (int equation = 0; equation < equations; ++equation)
      exterior[equation] = reference_state[equation];
    return true;
  }
  if (kind == EulerBoundaryStateKind::supersonic_outlet)
  {
    for (int equation = 0; equation < equations; ++equation)
      exterior[equation] = interior[equation];
    return true;
  }
  if (kind == EulerBoundaryStateKind::slip_wall)
  {
    for (int equation = 0; equation < equations; ++equation)
      exterior[equation] = interior[equation];
    const Scalar density = fmax(interior[0], density_floor);
    exterior[0] = density;
    Scalar normal_unit[3] = {Scalar(0.0), Scalar(0.0), Scalar(0.0)};
    Scalar tangent_unit[3] = {Scalar(0.0), Scalar(0.0), Scalar(0.0)};
    EulerBuildNormalTangential(
        normal, dimension, normal_unit, tangent_unit);
    Scalar normal_velocity = Scalar(0.0);
    for (int d = 0; d < dimension; ++d)
      normal_velocity += (interior[1 + d] / density) * normal_unit[d];
    for (int d = 0; d < dimension; ++d)
    {
      const Scalar reflected_velocity = interior[1 + d] / density -
          Scalar(2.0) * normal_velocity * normal_unit[d];
      exterior[1 + d] = density * reflected_velocity;
    }
    return true;
  }

  Scalar normal_unit[3] = {Scalar(0.0), Scalar(0.0), Scalar(0.0)};
  Scalar tangent_unit[3] = {Scalar(0.0), Scalar(0.0), Scalar(0.0)};
  EulerBuildNormalTangential(normal, dimension, normal_unit, tangent_unit);
  const EulerPrimitiveNormalState<Scalar> inner =
      EulerConservativeToPrimitiveNormal(
          interior,
          dimension,
          gamma,
          normal_unit,
          tangent_unit,
          density_floor,
          pressure_floor);

  EulerPrimitiveNormalState<Scalar> boundary = inner;
  if (kind == EulerBoundaryStateKind::subsonic_outlet_pressure)
    boundary =
        EulerSubsonicOutletPressureClosure(inner,
                                           gamma,
                                           pressure_out,
                                           density_floor,
                                           pressure_floor);
  else if (kind == EulerBoundaryStateKind::nonreflecting_outlet)
  {
    const EulerPrimitiveNormalState<Scalar> reference =
        EulerConservativeToPrimitiveNormal(
            reference_state,
            dimension,
            gamma,
            normal_unit,
            tangent_unit,
            density_floor,
            pressure_floor);
    boundary = EulerNonreflectingOutletClosure(
      inner, reference, gamma, density_floor, pressure_floor);
  }
  else if (kind == EulerBoundaryStateKind::subsonic_inlet)
  {
    const EulerPrimitiveNormalState<Scalar> target =
        EulerConservativeToPrimitiveNormal(
            reference_state,
            dimension,
            gamma,
            normal_unit,
            tangent_unit,
            density_floor,
            pressure_floor);
    boundary = EulerSubsonicInletClosure(
      inner, target, gamma, density_floor, pressure_floor);
  }
  else if (kind == EulerBoundaryStateKind::far_field)
  {
    const EulerPrimitiveNormalState<Scalar> far_field =
        EulerConservativeToPrimitiveNormal(
            reference_state,
            dimension,
            gamma,
            normal_unit,
            tangent_unit,
            density_floor,
            pressure_floor);
    if (inner.normal_velocity <= -inner.sound_speed)
    {
      for (int equation = 0; equation < equations; ++equation)
        exterior[equation] = reference_state[equation];
      return true;
    }
    if (inner.normal_velocity >= inner.sound_speed)
    {
      for (int equation = 0; equation < equations; ++equation)
        exterior[equation] = interior[equation];
      return true;
    }
    boundary = inner.normal_velocity < Scalar(0.0)
        ? EulerSubsonicInletClosure(
            inner, far_field, gamma, density_floor, pressure_floor)
        : EulerNonreflectingOutletClosure(
            inner, far_field, gamma, density_floor, pressure_floor);
  }
  else
    return false;

  EulerPrimitiveNormalToConservative(
      boundary, dimension, gamma, normal_unit, tangent_unit, exterior);
  return true;
}

} // namespace mfem::detail

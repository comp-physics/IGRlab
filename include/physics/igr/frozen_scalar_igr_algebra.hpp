#pragma once

#include "mfem/config/config.hpp"
#include "mfem/general/annotation.hpp"

#include <cmath>

/**
 * @file physics/igr/frozen_scalar_igr_algebra.hpp
 * @brief Storage-independent point algebra for the frozen scalar IGR model.
 *
 * Every operation is a C++17-compatible, MFEM_HOST_DEVICE free function over
 * mfem::real_t scalars and raw arrays. Array layouts are described explicitly
 * by strides, so this surface does not depend on finite-element, quadrature,
 * grid-function, or regularization-contract types.
 */

namespace dg_igr
{
namespace frozen_scalar_igr_algebra
{

MFEM_HOST_DEVICE inline mfem::real_t
DensityWithFloor(mfem::real_t density, mfem::real_t density_floor)
{
  return density < density_floor ? density_floor : density;
}

MFEM_HOST_DEVICE inline mfem::real_t
SafeInverseDensity(mfem::real_t density, mfem::real_t density_floor)
{
  return static_cast<mfem::real_t>(1.0) /
         DensityWithFloor(density, density_floor);
}

MFEM_HOST_DEVICE inline mfem::real_t
AlphaOverDensity(mfem::real_t alpha,
                 mfem::real_t density,
                 mfem::real_t density_floor)
{
  return alpha / DensityWithFloor(density, density_floor);
}

MFEM_HOST_DEVICE inline mfem::real_t
ConservativeVelocityComponent(mfem::real_t momentum,
                              mfem::real_t density,
                              mfem::real_t density_floor)
{
  return momentum / DensityWithFloor(density, density_floor);
}

/** Convert conservative-state gradients to grad(u) by the quotient rule.
 *
 * state[0] is density and state[1+i] is momentum i. The conservative gradient
 * contains density and momentum rows; any energy row is intentionally unused.
 */
MFEM_HOST_DEVICE inline void ConservativeStateGradientToVelocityGradient(
  int dimension,
  const mfem::real_t* state,
  int state_component_stride,
  const mfem::real_t* conservative_gradient,
  int gradient_component_stride,
  int gradient_direction_stride,
  mfem::real_t density_floor,
  mfem::real_t* velocity_gradient,
  int velocity_component_stride,
  int velocity_direction_stride)
{
  const mfem::real_t density = DensityWithFloor(state[0], density_floor);
  const mfem::real_t density_squared = density * density;
  for (int i = 0; i < dimension; ++i)
  {
    const mfem::real_t momentum = state[(1 + i) * state_component_stride];
    for (int j = 0; j < dimension; ++j)
    {
      const mfem::real_t momentum_gradient =
        conservative_gradient[(1 + i) * gradient_component_stride +
                              j * gradient_direction_stride];
      const mfem::real_t density_gradient =
        conservative_gradient[j * gradient_direction_stride];
      velocity_gradient[i * velocity_component_stride +
                        j * velocity_direction_stride] =
        (density * momentum_gradient - momentum * density_gradient) /
        density_squared;
    }
  }
}

/** IGR source from an already-evaluated velocity gradient (H1Projection semantics). */
MFEM_HOST_DEVICE inline mfem::real_t
SourceFromVelocityGradient(int dimension,
                           const mfem::real_t* velocity_gradient,
                           int velocity_component_stride,
                           int velocity_direction_stride,
                           mfem::real_t alpha)
{
  mfem::real_t divergence = static_cast<mfem::real_t>(0.0);
  mfem::real_t trace_gradient_squared = static_cast<mfem::real_t>(0.0);
  for (int i = 0; i < dimension; ++i)
  {
    divergence += velocity_gradient[i * velocity_component_stride +
                                    i * velocity_direction_stride];
  }
  for (int i = 0; i < dimension; ++i)
  {
    for (int j = 0; j < dimension; ++j)
    {
      trace_gradient_squared +=
        velocity_gradient[i * velocity_component_stride +
                          j * velocity_direction_stride] *
        velocity_gradient[j * velocity_component_stride +
                          i * velocity_direction_stride];
    }
  }
  return alpha * (trace_gradient_squared + divergence * divergence);
}

/** IGR source from a conservative state and its gradient (BrokenDG semantics). */
MFEM_HOST_DEVICE inline mfem::real_t
SourceFromConservativeStateGradient(int dimension,
                                    const mfem::real_t* state,
                                    int state_component_stride,
                                    const mfem::real_t* conservative_gradient,
                                    int gradient_component_stride,
                                    int gradient_direction_stride,
                                    mfem::real_t alpha,
                                    mfem::real_t density_floor)
{
  mfem::real_t velocity_gradient[9] = {static_cast<mfem::real_t>(0.0),
                                       static_cast<mfem::real_t>(0.0),
                                       static_cast<mfem::real_t>(0.0),
                                       static_cast<mfem::real_t>(0.0),
                                       static_cast<mfem::real_t>(0.0),
                                       static_cast<mfem::real_t>(0.0),
                                       static_cast<mfem::real_t>(0.0),
                                       static_cast<mfem::real_t>(0.0),
                                       static_cast<mfem::real_t>(0.0)};
  ConservativeStateGradientToVelocityGradient(dimension,
                                              state,
                                              state_component_stride,
                                              conservative_gradient,
                                              gradient_component_stride,
                                              gradient_direction_stride,
                                              density_floor,
                                              velocity_gradient,
                                              3,
                                              1);
  return SourceFromVelocityGradient(dimension, velocity_gradient, 3, 1, alpha);
}

MFEM_HOST_DEVICE inline mfem::real_t
TensorSigmaMomentumDiagonalTerm(mfem::real_t sigma)
{
  return sigma;
}

MFEM_HOST_DEVICE inline mfem::real_t
TensorSigmaEnergyTerm(mfem::real_t sigma,
                      mfem::real_t momentum,
                      mfem::real_t density,
                      mfem::real_t density_floor)
{
  return sigma * (momentum / DensityWithFloor(density, density_floor));
}

/** Preserve the packed weak-divergence path's precomputed-inverse ordering. */
MFEM_HOST_DEVICE inline mfem::real_t
TensorSigmaEnergyTermFromInverseDensity(mfem::real_t sigma,
                                        mfem::real_t momentum,
                                        mfem::real_t inverse_density)
{
  return sigma * momentum * inverse_density;
}

/** Add the pressure-like scalar IGR correction to an Euler tensor flux. */
MFEM_HOST_DEVICE inline void AddTensorSigmaFlux(int dimension,
                                                const mfem::real_t* state,
                                                int state_component_stride,
                                                mfem::real_t sigma,
                                                mfem::real_t density_floor,
                                                mfem::real_t* tensor_flux,
                                                int flux_equation_stride,
                                                int flux_direction_stride)
{
  for (int d = 0; d < dimension; ++d)
  {
    tensor_flux[(1 + d) * flux_equation_stride + d * flux_direction_stride] +=
      TensorSigmaMomentumDiagonalTerm(sigma);
  }
  for (int d = 0; d < dimension; ++d)
  {
    tensor_flux[(1 + dimension) * flux_equation_stride +
                d * flux_direction_stride] +=
      TensorSigmaEnergyTerm(sigma,
                            state[(1 + d) * state_component_stride],
                            state[0],
                            density_floor);
  }
}

/** Add the single-state pressure-like scalar IGR correction to F(U).n. */
MFEM_HOST_DEVICE inline void AddNormalSigmaFlux(int dimension,
                                                const mfem::real_t* state,
                                                int state_component_stride,
                                                const mfem::real_t* normal,
                                                int normal_stride,
                                                mfem::real_t sigma,
                                                mfem::real_t density_floor,
                                                mfem::real_t* normal_flux,
                                                int flux_component_stride)
{
  for (int d = 0; d < dimension; ++d)
  {
    normal_flux[(1 + d) * flux_component_stride] +=
      sigma * normal[d * normal_stride];
  }
  const mfem::real_t density = DensityWithFloor(state[0], density_floor);
  mfem::real_t normal_momentum = static_cast<mfem::real_t>(0.0);
  for (int d = 0; d < dimension; ++d)
  {
    normal_momentum +=
      state[(1 + d) * state_component_stride] * normal[d * normal_stride];
  }
  normal_flux[(1 + dimension) * flux_component_stride] +=
    sigma * (normal_momentum / density);
}

/** Exact line average of (m.n)/rho between two positive-density states. */
MFEM_HOST_DEVICE inline mfem::real_t
ExactAverageNormalVelocity(int dimension,
                           const mfem::real_t* state1,
                           int state1_component_stride,
                           const mfem::real_t* state2,
                           int state2_component_stride,
                           const mfem::real_t* normal,
                           int normal_stride)
{
  const mfem::real_t density1 = state1[0];
  const mfem::real_t density2 = state2[0];
  mfem::real_t normal_momentum1 = static_cast<mfem::real_t>(0.0);
  mfem::real_t normal_momentum2 = static_cast<mfem::real_t>(0.0);
  for (int d = 0; d < dimension; ++d)
  {
    normal_momentum1 +=
      state1[(1 + d) * state1_component_stride] * normal[d * normal_stride];
    normal_momentum2 +=
      state2[(1 + d) * state2_component_stride] * normal[d * normal_stride];
  }

  const mfem::real_t momentum_difference = normal_momentum2 - normal_momentum1;
  const mfem::real_t density_difference = density2 - density1;
  const mfem::real_t epsilon =
    static_cast<mfem::real_t>(1.0e-14) *
    fmax(static_cast<mfem::real_t>(1.0), fabs(density1));
  if (fabs(density_difference) <= epsilon)
  {
    return (normal_momentum1 +
            static_cast<mfem::real_t>(0.5) * momentum_difference) /
           density1;
  }

  const mfem::real_t inverse_density_difference =
    static_cast<mfem::real_t>(1.0) / density_difference;
  const mfem::real_t inverse_density_difference_squared =
    inverse_density_difference * inverse_density_difference;
  const mfem::real_t log_density_ratio = log1p(density_difference / density1);
  return momentum_difference * inverse_density_difference +
         (normal_momentum1 * density_difference -
          momentum_difference * density1) *
           inverse_density_difference_squared * log_density_ratio;
}

MFEM_HOST_DEVICE inline mfem::real_t
ExactTwoStateAverageNormalSigmaEnergyTerm(int dimension,
                                          const mfem::real_t* state1,
                                          int state1_component_stride,
                                          const mfem::real_t* state2,
                                          int state2_component_stride,
                                          const mfem::real_t* normal,
                                          int normal_stride,
                                          mfem::real_t sigma)
{
  return sigma * ExactAverageNormalVelocity(dimension,
                                            state1,
                                            state1_component_stride,
                                            state2,
                                            state2_component_stride,
                                            normal,
                                            normal_stride);
}

/** Add the exact line-averaged scalar IGR correction to an average flux. */
MFEM_HOST_DEVICE inline void
AddTwoStateAverageNormalSigmaFlux(int dimension,
                                  const mfem::real_t* state1,
                                  int state1_component_stride,
                                  const mfem::real_t* state2,
                                  int state2_component_stride,
                                  const mfem::real_t* normal,
                                  int normal_stride,
                                  mfem::real_t sigma,
                                  mfem::real_t* normal_flux,
                                  int flux_component_stride)
{
  for (int d = 0; d < dimension; ++d)
  {
    normal_flux[(1 + d) * flux_component_stride] +=
      sigma * normal[d * normal_stride];
  }
  normal_flux[(1 + dimension) * flux_component_stride] +=
    ExactTwoStateAverageNormalSigmaEnergyTerm(dimension,
                                              state1,
                                              state1_component_stride,
                                              state2,
                                              state2_component_stride,
                                              normal,
                                              normal_stride,
                                              sigma);
}

MFEM_HOST_DEVICE inline mfem::real_t
SigmaCharacteristicScale(mfem::real_t density,
                         mfem::real_t sigma,
                         mfem::real_t density_floor)
{
  return sqrt(fabs(sigma) / fmax(density, density_floor));
}

MFEM_HOST_DEVICE inline mfem::real_t
FluxCharacteristicSpeed(mfem::real_t base_speed,
                        mfem::real_t density,
                        mfem::real_t sigma,
                        mfem::real_t density_floor)
{
  return base_speed + SigmaCharacteristicScale(density, sigma, density_floor);
}

MFEM_HOST_DEVICE inline mfem::real_t
SelectRusanovDissipationSpeed(mfem::real_t reported_speed,
                              mfem::real_t density,
                              mfem::real_t sigma,
                              mfem::real_t density_floor,
                              bool sigma_aware_cfl,
                              bool sigma_aware_rusanov)
{
  const mfem::real_t sigma_scale =
    SigmaCharacteristicScale(density, sigma, density_floor);
  const mfem::real_t base_speed =
    fmax(reported_speed - (sigma_aware_cfl ? sigma_scale : 0.0), 0.0);
  return base_speed + (sigma_aware_rusanov ? sigma_scale : 0.0);
}

MFEM_HOST_DEVICE inline mfem::real_t
FluxInvariantDomainSpeed(mfem::real_t base_speed,
                         mfem::real_t density,
                         mfem::real_t pressure,
                         mfem::real_t transport_speed,
                         mfem::real_t sigma,
                         mfem::real_t gamma,
                         mfem::real_t density_floor,
                         mfem::real_t pressure_floor)
{
  const mfem::real_t safe_density = fmax(density, density_floor);
  const mfem::real_t safe_pressure = fmax(pressure, pressure_floor);
  const mfem::real_t pressure_scale =
    sqrt(fmax((gamma - 1.0) / (2.0 * safe_density * safe_pressure), 0.0));
  const mfem::real_t split_speed =
    transport_speed + fabs(pressure + sigma) * pressure_scale;
  return fmax(base_speed, split_speed);
}

} // namespace frozen_scalar_igr_algebra
} // namespace dg_igr

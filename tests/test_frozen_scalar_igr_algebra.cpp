#include "igrlab/compat.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

namespace
{

using ::igrlab::frozen_scalar_igr_algebra::AddNormalSigmaFlux;
using ::igrlab::frozen_scalar_igr_algebra::AddNormalSigmaFluxJacobian;
using ::igrlab::frozen_scalar_igr_algebra::AddTensorSigmaFlux;
using ::igrlab::frozen_scalar_igr_algebra::AddTensorSigmaFluxJacobian;
using ::igrlab::frozen_scalar_igr_algebra::AddTwoStateAverageNormalSigmaFlux;
using ::igrlab::frozen_scalar_igr_algebra::AlphaOverDensity;
using ::igrlab::frozen_scalar_igr_algebra::
  ConservativeStateGradientToVelocityGradient;
using ::igrlab::frozen_scalar_igr_algebra::ExactAverageNormalVelocity;
using ::igrlab::frozen_scalar_igr_algebra::
  ExactTwoStateAverageNormalSigmaEnergyTerm;
using ::igrlab::frozen_scalar_igr_algebra::FluxCharacteristicSpeed;
using ::igrlab::frozen_scalar_igr_algebra::FluxInvariantDomainSpeed;
using ::igrlab::frozen_scalar_igr_algebra::
  FrozenSigmaCharacteristicModulusBound;
using ::igrlab::frozen_scalar_igr_algebra::
  FrozenSigmaFluxIsLocallyStrictlyHyperbolic;
using ::igrlab::frozen_scalar_igr_algebra::
  FrozenSigmaLocalHyperbolicityMargin;
using ::igrlab::frozen_scalar_igr_algebra::SafeInverseDensity;
using ::igrlab::frozen_scalar_igr_algebra::SourceFromConservativeStateGradient;
using ::igrlab::frozen_scalar_igr_algebra::SourceFromVelocityGradient;
using ::igrlab::frozen_scalar_igr_algebra::
  TensorSigmaEnergyTermFromInverseDensity;

static_assert(std::is_trivially_copyable<mfem::real_t>::value,
              "IGR algebra scalar arguments must remain trivially copyable");

struct Checks
{
  int failures = 0;

  void True(bool condition, const std::string& label)
  {
    if (!condition)
    {
      ++failures;
      std::cerr << "FAIL: " << label << '\n';
    }
  }

  void Near(mfem::real_t actual,
            mfem::real_t expected,
            mfem::real_t tolerance,
            const std::string& label)
  {
    const mfem::real_t scale =
      std::max(static_cast<mfem::real_t>(1.0), std::abs(expected));
    if (!(std::abs(actual - expected) <= tolerance * scale))
    {
      ++failures;
      std::cerr << "FAIL: " << label << " actual=" << actual
                << " expected=" << expected << '\n';
    }
  }
};

void CheckFrozenSigmaSpeedContract(Checks& checks)
{
  constexpr mfem::real_t density_floor =
    static_cast<mfem::real_t>(1.0e-12);
  constexpr mfem::real_t pressure_floor =
    static_cast<mfem::real_t>(1.0e-12);

  // This is the audit counterexample: the Euler radius and the one-state LF
  // split threshold both underbound the positive-sigma frozen-flux radius.
  {
    constexpr mfem::real_t density = static_cast<mfem::real_t>(1.0);
    constexpr mfem::real_t pressure = static_cast<mfem::real_t>(1.0);
    constexpr mfem::real_t sigma = static_cast<mfem::real_t>(1.0);
    constexpr mfem::real_t gamma = static_cast<mfem::real_t>(1.4);
    constexpr mfem::real_t transport_speed = static_cast<mfem::real_t>(0.0);
    const mfem::real_t base_speed = sqrt(gamma * pressure / density);
    const mfem::real_t split_speed =
      fabs(pressure + sigma) *
      sqrt((gamma - static_cast<mfem::real_t>(1.0)) /
           (static_cast<mfem::real_t>(2.0) * density * pressure));
    const mfem::real_t acoustic_numerator =
      gamma * pressure +
      (gamma - static_cast<mfem::real_t>(1.0)) * sigma;
    const mfem::real_t frozen_sigma_radius =
      sqrt(acoustic_numerator / density);
    const mfem::real_t speed = FluxInvariantDomainSpeed(base_speed,
                                                        density,
                                                        pressure,
                                                        transport_speed,
                                                        sigma,
                                                        gamma,
                                                        density_floor,
                                                        pressure_floor);

    checks.True(base_speed < frozen_sigma_radius &&
                  split_speed < frozen_sigma_radius,
                "positive-sigma counterexample underbounds the local "
                "frozen-flux spectrum before correction");
    checks.True(FrozenSigmaFluxIsLocallyStrictlyHyperbolic(
                  pressure, sigma, gamma),
                "positive-sigma counterexample is locally hyperbolic");
    checks.Near(FrozenSigmaLocalHyperbolicityMargin(pressure, sigma, gamma),
                acoustic_numerator,
                static_cast<mfem::real_t>(2.0e-14),
                "frozen-sigma acoustic numerator");
    checks.Near(FrozenSigmaCharacteristicModulusBound(density,
                                                       pressure,
                                                       transport_speed,
                                                       sigma,
                                                       gamma,
                                                       density_floor),
                frozen_sigma_radius,
                static_cast<mfem::real_t>(2.0e-14),
                "frozen-sigma spectral radius counterexample");
    checks.Near(speed,
                frozen_sigma_radius,
                static_cast<mfem::real_t>(2.0e-14),
                "invariant-domain speed includes frozen-sigma spectrum");
  }

  // Negative sigma can remain locally hyperbolic. Its acoustic radius is
  // then smaller than Euler's, which must still remain in the combined bound.
  {
    constexpr mfem::real_t density = static_cast<mfem::real_t>(0.7);
    constexpr mfem::real_t pressure = static_cast<mfem::real_t>(2.0);
    constexpr mfem::real_t sigma = static_cast<mfem::real_t>(-1.0);
    constexpr mfem::real_t gamma = static_cast<mfem::real_t>(1.67);
    constexpr mfem::real_t transport_speed = static_cast<mfem::real_t>(0.37);
    const mfem::real_t base_speed =
      transport_speed + sqrt(gamma * pressure / density);
    const mfem::real_t acoustic_numerator =
      gamma * pressure +
      (gamma - static_cast<mfem::real_t>(1.0)) * sigma;
    const mfem::real_t local_radius =
      transport_speed + sqrt(acoustic_numerator / density);
    const mfem::real_t speed = FluxInvariantDomainSpeed(base_speed,
                                                        density,
                                                        pressure,
                                                        transport_speed,
                                                        sigma,
                                                        gamma,
                                                        density_floor,
                                                        pressure_floor);
    checks.True(FrozenSigmaFluxIsLocallyStrictlyHyperbolic(
                  pressure, sigma, gamma),
                "moderate negative sigma remains locally hyperbolic");
    checks.Near(FrozenSigmaCharacteristicModulusBound(density,
                                                       pressure,
                                                       transport_speed,
                                                       sigma,
                                                       gamma,
                                                       density_floor),
                local_radius,
                static_cast<mfem::real_t>(2.0e-14),
                "negative-sigma hyperbolic spectral radius");
    checks.Near(speed,
                base_speed,
                static_cast<mfem::real_t>(2.0e-14),
                "combined bound retains larger ordinary Euler radius");
  }

  // At and beyond loss of local hyperbolicity, retain finite arithmetic for
  // Rusanov while exposing the failed q>0 contract separately.
  {
    constexpr mfem::real_t density = static_cast<mfem::real_t>(1.0);
    constexpr mfem::real_t pressure = static_cast<mfem::real_t>(1.0);
    constexpr mfem::real_t gamma = static_cast<mfem::real_t>(1.4);
    constexpr mfem::real_t transport_speed = static_cast<mfem::real_t>(0.25);
    constexpr mfem::real_t negative_sigma = static_cast<mfem::real_t>(-4.0);
    const mfem::real_t base_speed =
      transport_speed + sqrt(gamma * pressure / density);
    const mfem::real_t negative_margin =
      FrozenSigmaLocalHyperbolicityMargin(pressure, negative_sigma, gamma);
    const mfem::real_t modulus_bound =
      FrozenSigmaCharacteristicModulusBound(density,
                                             pressure,
                                             transport_speed,
                                             negative_sigma,
                                             gamma,
                                             density_floor);
    const mfem::real_t nonhyperbolic_speed =
      FluxInvariantDomainSpeed(base_speed,
                               density,
                               pressure,
                               transport_speed,
                               negative_sigma,
                               gamma,
                               density_floor,
                               pressure_floor);
    checks.True(negative_margin < static_cast<mfem::real_t>(0.0) &&
                  !FrozenSigmaFluxIsLocallyStrictlyHyperbolic(
                    pressure, negative_sigma, gamma),
                "large negative sigma exposes loss of local hyperbolicity");
    checks.Near(modulus_bound,
                transport_speed + sqrt(fabs(negative_margin) / density),
                static_cast<mfem::real_t>(2.0e-14),
                "non-hyperbolic acoustic roots receive a finite modulus "
                "bound");
    checks.True(std::isfinite(nonhyperbolic_speed) &&
                  nonhyperbolic_speed >= base_speed &&
                  nonhyperbolic_speed >= modulus_bound,
                "non-hyperbolic invariant-domain arithmetic remains finite "
                "and conservative in modulus");

    constexpr mfem::real_t zero_margin_gamma =
      static_cast<mfem::real_t>(1.5);
    constexpr mfem::real_t zero_margin_sigma =
      static_cast<mfem::real_t>(-3.0);
    const mfem::real_t zero_margin_base_speed =
      transport_speed + sqrt(zero_margin_gamma * pressure / density);
    const mfem::real_t zero_margin_speed =
      FluxInvariantDomainSpeed(zero_margin_base_speed,
                               density,
                               pressure,
                               transport_speed,
                               zero_margin_sigma,
                               zero_margin_gamma,
                               density_floor,
                               pressure_floor);
    checks.True(!FrozenSigmaFluxIsLocallyStrictlyHyperbolic(
                  pressure, zero_margin_sigma, zero_margin_gamma) &&
                  std::isfinite(zero_margin_speed),
                "zero acoustic margin is explicitly non-strict but finite");
  }

  // The caller supplies unit-direction transport speed. Verify a near-floor,
  // nondefault-gamma oblique state gives the same bound for a scaled normal.
  {
    constexpr mfem::real_t gamma = static_cast<mfem::real_t>(1.67);
    constexpr mfem::real_t configured_density_floor =
      static_cast<mfem::real_t>(5.0e-7);
    constexpr mfem::real_t configured_pressure_floor =
      static_cast<mfem::real_t>(1.0e-6);
    const mfem::real_t density =
      std::nextafter(configured_density_floor,
                     std::numeric_limits<mfem::real_t>::infinity());
    const mfem::real_t pressure =
      std::nextafter(configured_pressure_floor,
                     std::numeric_limits<mfem::real_t>::infinity());
    constexpr mfem::real_t sigma = static_cast<mfem::real_t>(2.0e-6);
    constexpr std::array<mfem::real_t, 2> velocity{{
      static_cast<mfem::real_t>(0.4), static_cast<mfem::real_t>(-0.3)}};
    constexpr std::array<mfem::real_t, 2> normal{{
      static_cast<mfem::real_t>(0.6), static_cast<mfem::real_t>(-0.8)}};
    constexpr mfem::real_t normal_scale = static_cast<mfem::real_t>(7.25);
    auto normalized_transport = [&](mfem::real_t scale)
    {
      mfem::real_t dot = static_cast<mfem::real_t>(0.0);
      mfem::real_t norm_squared = static_cast<mfem::real_t>(0.0);
      for (int d = 0; d < 2; ++d)
      {
        const mfem::real_t scaled_normal = scale * normal[d];
        dot += velocity[d] * scaled_normal;
        norm_squared += scaled_normal * scaled_normal;
      }
      return fabs(dot) / sqrt(norm_squared);
    };
    const mfem::real_t unit_transport =
      normalized_transport(static_cast<mfem::real_t>(1.0));
    const mfem::real_t scaled_transport = normalized_transport(normal_scale);
    const mfem::real_t base_speed =
      unit_transport + sqrt(gamma * pressure / density);
    const mfem::real_t unit_bound = FluxInvariantDomainSpeed(
      base_speed,
      density,
      pressure,
      unit_transport,
      sigma,
      gamma,
      configured_density_floor,
      configured_pressure_floor);
    const mfem::real_t scaled_bound = FluxInvariantDomainSpeed(
      base_speed,
      density,
      pressure,
      scaled_transport,
      sigma,
      gamma,
      configured_density_floor,
      configured_pressure_floor);
    checks.True(FrozenSigmaFluxIsLocallyStrictlyHyperbolic(
                  pressure, sigma, gamma) && std::isfinite(unit_bound),
                "near-floor nondefault-gamma state has a finite local "
                "hyperbolic bound");
    checks.Near(scaled_transport,
                unit_transport,
                static_cast<mfem::real_t>(2.0e-14),
                "scaled oblique normal preserves unit-direction transport");
    checks.Near(scaled_bound,
                unit_bound,
                static_cast<mfem::real_t>(2.0e-14),
                "scaled oblique normal preserves the reported speed bound");
  }
}

mfem::real_t ExpectedSource(int dimension)
{
  if (dimension == 1)
  {
    return static_cast<mfem::real_t>(1.0);
  }
  if (dimension == 2)
  {
    return static_cast<mfem::real_t>(7.5);
  }
  return static_cast<mfem::real_t>(243.0);
}

mfem::real_t SourceAlpha(int dimension)
{
  if (dimension == 1)
  {
    return static_cast<mfem::real_t>(2.0);
  }
  if (dimension == 2)
  {
    return static_cast<mfem::real_t>(0.25);
  }
  return static_cast<mfem::real_t>(0.5);
}

void FillVelocityGradient(int dimension, mfem::real_t* gradient)
{
  std::fill(gradient, gradient + 9, static_cast<mfem::real_t>(0.0));
  if (dimension == 1)
  {
    gradient[0] = static_cast<mfem::real_t>(0.5);
    return;
  }
  if (dimension == 2)
  {
    gradient[0] = static_cast<mfem::real_t>(1.0);
    gradient[1] = static_cast<mfem::real_t>(2.0);
    gradient[3] = static_cast<mfem::real_t>(-3.0);
    gradient[4] = static_cast<mfem::real_t>(4.0);
    return;
  }
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      gradient[3 * i + j] = static_cast<mfem::real_t>(3 * i + j + 1);
    }
  }
}

void CheckGradientInputs(int dimension, Checks& checks)
{
  constexpr mfem::real_t density = static_cast<mfem::real_t>(2.0);
  constexpr mfem::real_t density_floor = static_cast<mfem::real_t>(0.5);
  const int equations = dimension + 2;
  std::array<mfem::real_t, 5> state{};
  std::array<mfem::real_t, 15> conservative_gradient{};
  std::array<mfem::real_t, 9> velocity_gradient{};
  std::array<mfem::real_t, 9> converted{};
  FillVelocityGradient(dimension, velocity_gradient.data());

  state[0] = density;
  for (int i = 0; i < dimension; ++i)
  {
    const mfem::real_t velocity = static_cast<mfem::real_t>(0.25 * (i + 1));
    state[1 + i] = density * velocity;
    for (int j = 0; j < dimension; ++j)
    {
      const mfem::real_t density_gradient =
        static_cast<mfem::real_t>(0.1 * (j + 1));
      conservative_gradient[equations * j] = density_gradient;
      conservative_gradient[1 + i + equations * j] =
        density * velocity_gradient[3 * i + j] + velocity * density_gradient;
    }
  }

  ConservativeStateGradientToVelocityGradient(dimension,
                                              state.data(),
                                              1,
                                              conservative_gradient.data(),
                                              1,
                                              equations,
                                              density_floor,
                                              converted.data(),
                                              3,
                                              1);
  for (int i = 0; i < dimension; ++i)
  {
    for (int j = 0; j < dimension; ++j)
    {
      checks.Near(converted[3 * i + j],
                  velocity_gradient[3 * i + j],
                  static_cast<mfem::real_t>(2.0e-14),
                  "conservative-to-velocity gradient dim=" +
                    std::to_string(dimension));
    }
  }

  const mfem::real_t alpha = SourceAlpha(dimension);
  const mfem::real_t h1_source =
    SourceFromVelocityGradient(dimension,
                               velocity_gradient.data(),
                               3,
                               1,
                               alpha);
  const mfem::real_t broken_source =
    SourceFromConservativeStateGradient(dimension,
                                        state.data(),
                                        1,
                                        conservative_gradient.data(),
                                        1,
                                        equations,
                                        alpha,
                                        density_floor);
  checks.Near(h1_source,
              ExpectedSource(dimension),
              static_cast<mfem::real_t>(2.0e-14),
              "H1 source oracle dim=" + std::to_string(dimension));
  checks.Near(broken_source,
              h1_source,
              static_cast<mfem::real_t>(2.0e-14),
              "BrokenDG matches consistent H1 input dim=" +
                std::to_string(dimension));
}

void CheckFluxIdentities(int dimension, Checks& checks)
{
  const int equations = dimension + 2;
  constexpr mfem::real_t density_floor = static_cast<mfem::real_t>(0.5);
  constexpr mfem::real_t sigma = static_cast<mfem::real_t>(1.25);
  std::array<mfem::real_t, 5> state{};
  std::array<mfem::real_t, 3> normal{{static_cast<mfem::real_t>(0.5),
                                      static_cast<mfem::real_t>(-1.0),
                                      static_cast<mfem::real_t>(2.0)}};
  std::array<mfem::real_t, 3> momentum{{static_cast<mfem::real_t>(1.0),
                                        static_cast<mfem::real_t>(-2.0),
                                        static_cast<mfem::real_t>(0.25)}};
  std::array<mfem::real_t, 15> tensor{};
  std::array<mfem::real_t, 5> normal_flux{};

  state[0] = static_cast<mfem::real_t>(0.25);
  for (int d = 0; d < dimension; ++d)
  {
    state[1 + d] = momentum[d];
  }

  AddTensorSigmaFlux(dimension,
                     state.data(),
                     1,
                     sigma,
                     density_floor,
                     tensor.data(),
                     1,
                     equations);
  AddNormalSigmaFlux(dimension,
                     state.data(),
                     1,
                     normal.data(),
                     1,
                     sigma,
                     density_floor,
                     normal_flux.data(),
                     1);

  for (int equation = 0; equation < equations; ++equation)
  {
    mfem::real_t contraction = static_cast<mfem::real_t>(0.0);
    for (int d = 0; d < dimension; ++d)
    {
      contraction += tensor[equation + equations * d] * normal[d];
    }
    checks.Near(contraction,
                normal_flux[equation],
                static_cast<mfem::real_t>(2.0e-14),
                "tensor-normal contraction dim=" + std::to_string(dimension));
  }
  for (int i = 0; i < dimension; ++i)
  {
    for (int j = 0; j < dimension; ++j)
    {
      checks.Near(tensor[(1 + i) + equations * j],
                  i == j ? sigma : static_cast<mfem::real_t>(0.0),
                  static_cast<mfem::real_t>(2.0e-14),
                  "tensor sigma identity dim=" + std::to_string(dimension));
    }
  }

  std::array<mfem::real_t, 3> zero_normal{};
  std::array<mfem::real_t, 5> zero_flux{};
  AddNormalSigmaFlux(dimension,
                     state.data(),
                     1,
                     zero_normal.data(),
                     1,
                     sigma,
                     density_floor,
                     zero_flux.data(),
                     1);
  for (int equation = 0; equation < equations; ++equation)
  {
    checks.Near(zero_flux[equation],
                static_cast<mfem::real_t>(0.0),
                static_cast<mfem::real_t>(0.0),
                "zero-normal degeneracy dim=" + std::to_string(dimension));
  }
}

void CheckFluxJacobians(int dimension, Checks& checks)
{
  const int equations = dimension + 2;
  const int energy_equation = dimension + 1;
  constexpr mfem::real_t density_floor = static_cast<mfem::real_t>(0.5);
  constexpr mfem::real_t sigma = static_cast<mfem::real_t>(1.25);
  constexpr mfem::real_t fd_step = static_cast<mfem::real_t>(1.0e-6);
  const std::array<mfem::real_t, 3> normal{{static_cast<mfem::real_t>(0.5),
                                            static_cast<mfem::real_t>(-1.0),
                                            static_cast<mfem::real_t>(2.0)}};
  const std::array<mfem::real_t, 3> momentum{{static_cast<mfem::real_t>(1.0),
                                              static_cast<mfem::real_t>(-2.0),
                                              static_cast<mfem::real_t>(0.25)}};

  for (const mfem::real_t density : {static_cast<mfem::real_t>(1.75),
                                     static_cast<mfem::real_t>(0.25)})
  {
    const bool above_floor = density > density_floor;
    std::array<mfem::real_t, 5> state{};
    std::array<mfem::real_t, 75> tensor_jacobian{};
    std::array<mfem::real_t, 25> normal_jacobian{};
    state[0] = density;
    for (int d = 0; d < dimension; ++d)
    {
      state[1 + d] = momentum[d];
    }

    AddTensorSigmaFluxJacobian(dimension,
                               state.data(),
                               1,
                               sigma,
                               density_floor,
                               tensor_jacobian.data(),
                               1,
                               equations,
                               equations * equations);
    AddNormalSigmaFluxJacobian(dimension,
                               state.data(),
                               1,
                               normal.data(),
                               1,
                               sigma,
                               density_floor,
                               normal_jacobian.data(),
                               1,
                               equations);

    mfem::real_t normal_momentum = static_cast<mfem::real_t>(0.0);
    for (int d = 0; d < dimension; ++d)
    {
      normal_momentum += momentum[d] * normal[d];
    }
    const mfem::real_t inverse_density =
      static_cast<mfem::real_t>(1.0) /
      (above_floor ? density : density_floor);
    for (int equation = 0; equation < equations; ++equation)
    {
      for (int state_component = 0; state_component < equations;
           ++state_component)
      {
        mfem::real_t normal_expected = static_cast<mfem::real_t>(0.0);
        if (equation == energy_equation && state_component == 0 &&
            above_floor)
        {
          normal_expected = -sigma * normal_momentum / (density * density);
        }
        if (equation == energy_equation && state_component >= 1 &&
            state_component <= dimension)
        {
          normal_expected = sigma * normal[state_component - 1] * inverse_density;
        }
        checks.Near(normal_jacobian[equation + equations * state_component],
                    normal_expected,
                    static_cast<mfem::real_t>(2.0e-14),
                    "normal sigma Jacobian analytic dim=" +
                      std::to_string(dimension));

        mfem::real_t normal_contraction = static_cast<mfem::real_t>(0.0);
        for (int d = 0; d < dimension; ++d)
        {
          mfem::real_t tensor_expected = static_cast<mfem::real_t>(0.0);
          if (equation == energy_equation && state_component == 0 &&
              above_floor)
          {
            tensor_expected = -sigma * momentum[d] / (density * density);
          }
          if (equation == energy_equation && state_component == 1 + d)
          {
            tensor_expected = sigma * inverse_density;
          }
          const mfem::real_t tensor_actual =
            tensor_jacobian[equation + equations * state_component +
                            equations * equations * d];
          checks.Near(tensor_actual,
                      tensor_expected,
                      static_cast<mfem::real_t>(2.0e-14),
                      "tensor sigma Jacobian analytic dim=" +
                        std::to_string(dimension));
          normal_contraction += tensor_actual * normal[d];
        }
        checks.Near(normal_contraction,
                    normal_jacobian[equation + equations * state_component],
                    static_cast<mfem::real_t>(2.0e-14),
                    "Jacobian tensor-normal contraction dim=" +
                      std::to_string(dimension));
      }
    }

    for (int state_component = 0; state_component < equations;
         ++state_component)
    {
      std::array<mfem::real_t, 5> state_minus = state;
      std::array<mfem::real_t, 5> state_plus = state;
      std::array<mfem::real_t, 15> tensor_minus{};
      std::array<mfem::real_t, 15> tensor_plus{};
      std::array<mfem::real_t, 5> normal_minus{};
      std::array<mfem::real_t, 5> normal_plus{};
      state_minus[state_component] -= fd_step;
      state_plus[state_component] += fd_step;
      AddTensorSigmaFlux(dimension,
                         state_minus.data(),
                         1,
                         sigma,
                         density_floor,
                         tensor_minus.data(),
                         1,
                         equations);
      AddTensorSigmaFlux(dimension,
                         state_plus.data(),
                         1,
                         sigma,
                         density_floor,
                         tensor_plus.data(),
                         1,
                         equations);
      AddNormalSigmaFlux(dimension,
                         state_minus.data(),
                         1,
                         normal.data(),
                         1,
                         sigma,
                         density_floor,
                         normal_minus.data(),
                         1);
      AddNormalSigmaFlux(dimension,
                         state_plus.data(),
                         1,
                         normal.data(),
                         1,
                         sigma,
                         density_floor,
                         normal_plus.data(),
                         1);
      for (int equation = 0; equation < equations; ++equation)
      {
        const mfem::real_t normal_fd =
          (normal_plus[equation] - normal_minus[equation]) /
          (static_cast<mfem::real_t>(2.0) * fd_step);
        checks.Near(normal_fd,
                    normal_jacobian[equation + equations * state_component],
                    static_cast<mfem::real_t>(2.0e-8),
                    "normal sigma Jacobian centered FD dim=" +
                      std::to_string(dimension));
        for (int d = 0; d < dimension; ++d)
        {
          const mfem::real_t tensor_fd =
            (tensor_plus[equation + equations * d] -
             tensor_minus[equation + equations * d]) /
            (static_cast<mfem::real_t>(2.0) * fd_step);
          checks.Near(tensor_fd,
                      tensor_jacobian[equation + equations * state_component +
                                      equations * equations * d],
                      static_cast<mfem::real_t>(2.0e-8),
                      "tensor sigma Jacobian centered FD dim=" +
                        std::to_string(dimension));
        }
      }
    }
  }

  std::array<mfem::real_t, 5> zero_momentum_state{};
  std::array<mfem::real_t, 3> zero_normal{};
  std::array<mfem::real_t, 75> tensor_jacobian{};
  std::array<mfem::real_t, 25> normal_jacobian{};
  zero_momentum_state[0] = static_cast<mfem::real_t>(1.75);
  AddTensorSigmaFluxJacobian(dimension,
                             zero_momentum_state.data(),
                             1,
                             sigma,
                             density_floor,
                             tensor_jacobian.data(),
                             1,
                             equations,
                             equations * equations);
  AddNormalSigmaFluxJacobian(dimension,
                             zero_momentum_state.data(),
                             1,
                             zero_normal.data(),
                             1,
                             sigma,
                             density_floor,
                             normal_jacobian.data(),
                             1,
                             equations);
  for (int equation = 0; equation < equations; ++equation)
  {
    for (int state_component = 0; state_component < equations;
         ++state_component)
    {
      checks.Near(normal_jacobian[equation + equations * state_component],
                  static_cast<mfem::real_t>(0.0),
                  static_cast<mfem::real_t>(0.0),
                  "zero-normal sigma Jacobian dim=" +
                    std::to_string(dimension));
    }
  }
  for (int d = 0; d < dimension; ++d)
  {
    checks.Near(tensor_jacobian[energy_equation +
                                equations * equations * d],
                static_cast<mfem::real_t>(0.0),
                static_cast<mfem::real_t>(0.0),
                "zero-momentum tensor density derivative dim=" +
                  std::to_string(dimension));
  }
}

void CheckTwoStateAverage(int dimension, Checks& checks)
{
  constexpr mfem::real_t sigma = static_cast<mfem::real_t>(0.75);
  std::array<mfem::real_t, 5> state1{};
  std::array<mfem::real_t, 5> state2{};
  std::array<mfem::real_t, 3> normal{};
  normal[0] = static_cast<mfem::real_t>(1.0);
  state1[0] = static_cast<mfem::real_t>(0.9);
  state2[0] = static_cast<mfem::real_t>(1.4);
  state1[1] = static_cast<mfem::real_t>(1.0);
  state2[1] = static_cast<mfem::real_t>(1.0);

  const mfem::real_t expected_log_average =
    std::log(state2[0] / state1[0]) / (state2[0] - state1[0]);
  const mfem::real_t average = ExactAverageNormalVelocity(dimension,
                                                          state1.data(),
                                                          1,
                                                          state2.data(),
                                                          1,
                                                          normal.data(),
                                                          1);
  checks.Near(average,
              expected_log_average,
              static_cast<mfem::real_t>(3.0e-14),
              "exact logarithmic two-state average dim=" +
                std::to_string(dimension));
  checks.Near(ExactAverageNormalVelocity(dimension,
                                         state2.data(),
                                         1,
                                         state1.data(),
                                         1,
                                         normal.data(),
                                         1),
              average,
              static_cast<mfem::real_t>(3.0e-14),
              "two-state symmetry dim=" + std::to_string(dimension));
  checks.Near(ExactTwoStateAverageNormalSigmaEnergyTerm(dimension,
                                                        state1.data(),
                                                        1,
                                                        state2.data(),
                                                        1,
                                                        normal.data(),
                                                        1,
                                                        sigma),
              sigma * expected_log_average,
              static_cast<mfem::real_t>(3.0e-14),
              "two-state sigma energy term dim=" + std::to_string(dimension));

  state1[0] = static_cast<mfem::real_t>(2.0);
  state2[0] =
    static_cast<mfem::real_t>(2.0) + static_cast<mfem::real_t>(1.0e-15);
  state1[1] = static_cast<mfem::real_t>(3.0);
  state2[1] = static_cast<mfem::real_t>(-1.0);
  checks.Near(ExactAverageNormalVelocity(dimension,
                                         state1.data(),
                                         1,
                                         state2.data(),
                                         1,
                                         normal.data(),
                                         1),
              static_cast<mfem::real_t>(0.5),
              static_cast<mfem::real_t>(2.0e-14),
              "equal-density degeneracy dim=" + std::to_string(dimension));

  state2 = state1;
  std::array<mfem::real_t, 5> single_flux{};
  std::array<mfem::real_t, 5> average_flux{};
  AddNormalSigmaFlux(dimension,
                     state1.data(),
                     1,
                     normal.data(),
                     1,
                     sigma,
                     static_cast<mfem::real_t>(1.0e-10),
                     single_flux.data(),
                     1);
  AddTwoStateAverageNormalSigmaFlux(dimension,
                                    state1.data(),
                                    1,
                                    state2.data(),
                                    1,
                                    normal.data(),
                                    1,
                                    sigma,
                                    average_flux.data(),
                                    1);
  for (int equation = 0; equation < dimension + 2; ++equation)
  {
    checks.Near(average_flux[equation],
                single_flux[equation],
                static_cast<mfem::real_t>(2.0e-14),
                "identical-state normal/average identity dim=" +
                  std::to_string(dimension));
  }
}

} // namespace

int main()
{
  Checks checks;
  constexpr mfem::real_t floor = static_cast<mfem::real_t>(0.5);
  checks.Near(SafeInverseDensity(static_cast<mfem::real_t>(2.0), floor),
              static_cast<mfem::real_t>(0.5),
              static_cast<mfem::real_t>(0.0),
              "inverse density above floor");
  checks.Near(SafeInverseDensity(static_cast<mfem::real_t>(0.25), floor),
              static_cast<mfem::real_t>(2.0),
              static_cast<mfem::real_t>(0.0),
              "inverse density at floor");
  checks.Near(AlphaOverDensity(static_cast<mfem::real_t>(3.0),
                               static_cast<mfem::real_t>(-1.0),
                               floor),
              static_cast<mfem::real_t>(6.0),
              static_cast<mfem::real_t>(0.0),
              "alpha over floored density");
  checks.Near(
    TensorSigmaEnergyTermFromInverseDensity(static_cast<mfem::real_t>(1.25),
                                            static_cast<mfem::real_t>(-2.0),
                                            static_cast<mfem::real_t>(0.5)),
    static_cast<mfem::real_t>(-1.25),
    static_cast<mfem::real_t>(0.0),
    "precomputed-inverse tensor energy ordering");
  checks.Near(FluxCharacteristicSpeed(static_cast<mfem::real_t>(1.0),
                                      static_cast<mfem::real_t>(0.25),
                                      static_cast<mfem::real_t>(-2.0),
                                      floor),
              static_cast<mfem::real_t>(3.0),
              static_cast<mfem::real_t>(2.0e-14),
              "characteristic scale uses absolute sigma and density floor");
  CheckFrozenSigmaSpeedContract(checks);

  {
    const std::array<mfem::real_t, 3> state{{static_cast<mfem::real_t>(0.25),
                                             static_cast<mfem::real_t>(1.0),
                                             static_cast<mfem::real_t>(3.0)}};
    const std::array<mfem::real_t, 3> gradient{
      {static_cast<mfem::real_t>(0.2),
       static_cast<mfem::real_t>(0.7),
       static_cast<mfem::real_t>(0.0)}};
    const mfem::real_t broken =
      SourceFromConservativeStateGradient(1,
                                          state.data(),
                                          1,
                                          gradient.data(),
                                          1,
                                          3,
                                          2.0,
                                          floor);
    const mfem::real_t h1_gradient = static_cast<mfem::real_t>(0.6);
    const mfem::real_t h1 =
      SourceFromVelocityGradient(1, &h1_gradient, 1, 1, 2.0);
    checks.Near(broken,
                static_cast<mfem::real_t>(1.44),
                static_cast<mfem::real_t>(2.0e-14),
                "BrokenDG quotient rule uses floored density");
    checks.Near(h1,
                broken,
                static_cast<mfem::real_t>(2.0e-14),
                "H1 consumes an already-evaluated gradient");
  }

  for (int dimension = 1; dimension <= 3; ++dimension)
  {
    CheckGradientInputs(dimension, checks);
    CheckFluxIdentities(dimension, checks);
    CheckFluxJacobians(dimension, checks);
    CheckTwoStateAverage(dimension, checks);
  }

  if (checks.failures == 0)
  {
    std::cout << "Frozen scalar IGR algebra point tests passed (C++17, "
                 "dimensions 1/2/3).\n";
  }
  return checks.failures == 0 ? 0 : 1;
}

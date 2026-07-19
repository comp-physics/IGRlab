#include "physics/igr/frozen_scalar_igr_algebra.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

namespace
{

using ::dg_igr::frozen_scalar_igr_algebra::AddNormalSigmaFlux;
using ::dg_igr::frozen_scalar_igr_algebra::AddTensorSigmaFlux;
using ::dg_igr::frozen_scalar_igr_algebra::AddTwoStateAverageNormalSigmaFlux;
using ::dg_igr::frozen_scalar_igr_algebra::AlphaOverDensity;
using ::dg_igr::frozen_scalar_igr_algebra::
  ConservativeStateGradientToVelocityGradient;
using ::dg_igr::frozen_scalar_igr_algebra::ExactAverageNormalVelocity;
using ::dg_igr::frozen_scalar_igr_algebra::
  ExactTwoStateAverageNormalSigmaEnergyTerm;
using ::dg_igr::frozen_scalar_igr_algebra::FluxCharacteristicSpeed;
using ::dg_igr::frozen_scalar_igr_algebra::SafeInverseDensity;
using ::dg_igr::frozen_scalar_igr_algebra::SourceFromConservativeStateGradient;
using ::dg_igr::frozen_scalar_igr_algebra::SourceFromVelocityGradient;
using ::dg_igr::frozen_scalar_igr_algebra::
  TensorSigmaEnergyTermFromInverseDensity;

static_assert(std::is_trivially_copyable<mfem::real_t>::value,
              "IGR algebra scalar arguments must remain trivially copyable");

struct Checks
{
  int failures = 0;

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
    CheckTwoStateAverage(dimension, checks);
  }

  if (checks.failures == 0)
  {
    std::cout << "Frozen scalar IGR algebra point tests passed (C++17, "
                 "dimensions 1/2/3).\n";
  }
  return checks.failures == 0 ? 0 : 1;
}

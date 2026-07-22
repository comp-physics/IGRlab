#include "numerics/hdg/euler/smooth_euler.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{
using dg::hdg::euler::ConservedState;
using dg::hdg::euler::FluxJacobian;
using dg::hdg::euler::FluxVector;
using dg::hdg::euler::ManufacturedConservative;
using dg::hdg::euler::ManufacturedMaximumWaveSpeed;
using dg::hdg::euler::ManufacturedSource;
using dg::hdg::euler::NormalFlux;
using dg::hdg::euler::NormalFluxJacobian;
using dg::hdg::euler::PhysicsStatus;
using dg::hdg::euler::RunSmoothEulerExperiment;
using dg::hdg::euler::SmoothEulerExperiment;
using dg::hdg::euler::ExperimentEvidence;
using dg::hdg::euler::fixed_scalar_tau;
using dg::hdg::euler::conserved_components;

constexpr int ByNodesIndex(const int scalar_dof, const int scalar_count,
                           const int component)
{
  return scalar_dof + scalar_count * component;
}

constexpr int ComponentFromByNodesIndex(const int vector_dof,
                                        const int scalar_count)
{
  return vector_dof / scalar_count;
}

constexpr int ScalarFromByNodesIndex(const int vector_dof,
                                     const int scalar_count)
{
  return vector_dof - scalar_count * ComponentFromByNodesIndex(vector_dof,
                                                                 scalar_count);
}

static_assert(ByNodesIndex(3, 11, 2) == 25,
              "MFEM byNODES is scalar_dof + scalar_count * component");
static_assert(ComponentFromByNodesIndex(25, 11) == 2,
              "byNODES inverse must retain the scalar extent");
static_assert(ScalarFromByNodesIndex(25, 11) == 3,
              "byNODES inverse must recover the scalar dof");

bool Close(const double left, const double right, const double tolerance)
{
  return std::abs(left - right) <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

int TestHandFluxAndFailureAtomicity()
{
  const ConservedState state{{1.0, 2.0, 3.0, 10.0}};
  FluxVector flux{{-7.0, -7.0, -7.0, -7.0}};
  if (!NormalFlux(state, {{1.0, 0.0}}, flux).ok()) { return 1; }
  const FluxVector expected{{2.0, 5.4, 6.0, 22.8}};
  for (int component = 0; component < conserved_components; ++component)
  {
    if (!Close(flux[component], expected[component], 2.0e-14)) { return 2; }
  }
  const FluxVector saved = flux;
  ConservedState bad_density = state;
  bad_density[0] = 0.0;
  if (NormalFlux(bad_density, {{1.0, 0.0}}, flux).status != PhysicsStatus::invalid_density ||
      flux != saved) { return 3; }
  ConservedState bad_pressure = state;
  bad_pressure[3] = 6.5;
  if (NormalFlux(bad_pressure, {{1.0, 0.0}}, flux).status != PhysicsStatus::invalid_pressure ||
      flux != saved) { return 4; }
  ConservedState non_finite = state;
  non_finite[2] = std::numeric_limits<double>::infinity();
  if (NormalFlux(non_finite, {{1.0, 0.0}}, flux).status != PhysicsStatus::non_finite ||
      flux != saved) { return 5; }
  return 0;
}

int TestAnalyticFluxJacobian()
{
  const ConservedState state{{1.2, 0.35, -0.22, 2.8}};
  const std::array<double, 2> normal{{0.6, -0.8}};
  FluxJacobian analytic{};
  if (!NormalFluxJacobian(state, normal, analytic).ok()) { return 1; }
  if (analytic[0][0] != 0.0) { return 2; }
  for (int column = 0; column < conserved_components; ++column)
  {
    ConservedState plus = state;
    ConservedState minus = state;
    const double step = 1.0e-7 * std::max(1.0, std::abs(state[column]));
    plus[column] += step;
    minus[column] -= step;
    FluxVector positive{};
    FluxVector negative{};
    if (!NormalFlux(plus, normal, positive).ok() || !NormalFlux(minus, normal, negative).ok())
    {
      return 3;
    }
    for (int row = 0; row < conserved_components; ++row)
    {
      const double difference = (positive[row] - negative[row]) / (2.0 * step);
      if (!Close(analytic[row][column], difference, 2.0e-6)) { return 4; }
    }
  }
  return 0;
}

int TestManufacturedSource()
{
  constexpr double x = 0.173, y = 0.619, h = 1.0e-5;
  ConservedState source{};
  if (!ManufacturedSource(x, y, source).ok()) { return 1; }
  FluxVector xp{}, xm{}, yp{}, ym{};
  ConservedState state{};
  if (!ManufacturedConservative(x + h, y, state).ok() || !NormalFlux(state, {{1.0, 0.0}}, xp).ok() ||
      !ManufacturedConservative(x - h, y, state).ok() || !NormalFlux(state, {{1.0, 0.0}}, xm).ok() ||
      !ManufacturedConservative(x, y + h, state).ok() || !NormalFlux(state, {{0.0, 1.0}}, yp).ok() ||
      !ManufacturedConservative(x, y - h, state).ok() || !NormalFlux(state, {{0.0, 1.0}}, ym).ok())
  {
    return 2;
  }
  for (int component = 0; component < conserved_components; ++component)
  {
    const double finite_difference = (xp[component] - xm[component] + yp[component] - ym[component]) /
                                     (2.0 * h);
    if (!Close(source[component], finite_difference, 2.0e-6)) { return 3; }
  }
  const double maximum_wave_speed = ManufacturedMaximumWaveSpeed();
  return maximum_wave_speed < 1.78 && fixed_scalar_tau - maximum_wave_speed > 0.72 ? 0 : 4;
}

int TestMfemByNodesAndSignedVDofs()
{
  constexpr int scalar_count = 9;
  for (int component = 0; component < conserved_components; ++component)
  {
    for (int scalar = 0; scalar < scalar_count; ++scalar)
    {
      const int vector_dof = ByNodesIndex(scalar, scalar_count, component);
      if (ComponentFromByNodesIndex(vector_dof, scalar_count) != component ||
          ScalarFromByNodesIndex(vector_dof, scalar_count) != scalar)
      {
        return 1;
      }
    }
  }
  double positive_sign = 0.0;
  if (mfem::FiniteElementSpace::DecodeDof(23, positive_sign) != 23 ||
      positive_sign != 1.0)
  {
    return 2;
  }
  double negative_sign = 0.0;
  if (mfem::FiniteElementSpace::DecodeDof(-24, negative_sign) != 23 ||
      negative_sign != -1.0)
  {
    return 3;
  }
  return 0;
}

int TestFailurePreservingSetup()
{
  SmoothEulerExperiment experiment(1);
  if (!experiment.BuildPeriodicMesh(3) || !experiment.ProjectOldStage())
  {
    return 1;
  }
  if (experiment.BuildPeriodicMesh(2)) { return 2; }
  // A failed fresh build cannot displace the previously complete mesh/space
  // graph, so the already configured projection remains reusable.
  if (!experiment.ProjectOldStage()) { return 3; }
  return experiment.BuildPeriodicMesh(3) && experiment.ProjectOldStage() ? 0 : 4;
}

int TestCompleteExperimentGate()
{
  ExperimentEvidence evidence{};
  evidence.passed = true;
  // This is deliberately a real complete-gate invocation, not a print-only
  // status. Until the source implementation contains all independent oracles
  // and the convergence ladder, its false result fails this executable.
  return RunSmoothEulerExperiment(evidence) && evidence.passed ? 0 : 1;
}

} // namespace

int main()
{
  const int flux = TestHandFluxAndFailureAtomicity();
  if (flux != 0) { return 10 + flux; }
  const int jacobian = TestAnalyticFluxJacobian();
  if (jacobian != 0) { return 20 + jacobian; }
  const int source = TestManufacturedSource();
  if (source != 0) { return 30 + source; }
  const int ordering = TestMfemByNodesAndSignedVDofs();
  if (ordering != 0) { return 40 + ordering; }
  const int setup = TestFailurePreservingSetup();
  if (setup != 0) { return 50 + setup; }
  const int experiment = TestCompleteExperimentGate();
  return experiment == 0 ? 0 : 60 + experiment;
}

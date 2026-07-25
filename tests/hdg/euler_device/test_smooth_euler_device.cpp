#include "numerics/hdg/euler/smooth_euler.hpp"
#include "numerics/hdg/euler/smooth_euler_device.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

namespace dg::hdg::euler::detail
{
bool PrepareSmoothEulerDeviceOracles();
}

namespace
{
using dg::hdg::device::ExecutionPath;
using dg::hdg::euler::ConservedState;
using dg::hdg::euler::DeviceEulerEvidence;
using dg::hdg::euler::DeviceEulerStatus;
using dg::hdg::euler::FluxJacobian;
using dg::hdg::euler::FluxVector;
using dg::hdg::euler::NormalFlux;
using dg::hdg::euler::NormalFluxJacobian;
using dg::hdg::euler::PhysicsStatus;
using dg::hdg::euler::RunSmoothEulerDeviceExperiment;

bool Near(const double actual, const double expected, const double tolerance)
{
  return std::isfinite(actual) &&
         std::abs(actual - expected) <= tolerance *
             std::max(1.0, std::abs(expected));
}

int TestIndependentFluxAndJacobian()
{
  const ConservedState state{{1.2, 0.48, -0.3, 2.3835}};
  const std::array<double, 2> normal{{0.6, 0.8}};
  const FluxVector expected_flux{{0.048, 0.5592, 0.708, 0.13134}};
  const FluxJacobian expected_jacobian{{
      {{0.0, 0.6, 0.8, 0.0}},
      {{0.0107, 0.184, 0.38, 0.24}},
      {{0.0456, -0.278, -0.08, 0.32}},
      {{-0.10767, 1.63535, 2.193, 0.056}}}};
  FluxVector flux{};
  FluxJacobian jacobian{};
  if (!NormalFlux(state, normal, flux).ok() ||
      !NormalFluxJacobian(state, normal, jacobian).ok())
  {
    return 10;
  }
  for (int row = 0; row < dg::hdg::euler::conserved_components; ++row)
  {
    if (!Near(flux[static_cast<std::size_t>(row)],
              expected_flux[static_cast<std::size_t>(row)], 5.0e-13))
    {
      return 11;
    }
    for (int column = 0; column < dg::hdg::euler::conserved_components;
         ++column)
    {
      if (!Near(jacobian[static_cast<std::size_t>(row)]
                         [static_cast<std::size_t>(column)],
                expected_jacobian[static_cast<std::size_t>(row)]
                                   [static_cast<std::size_t>(column)],
                5.0e-13))
      {
        return 12;
      }
    }
  }

  for (int column = 0; column < dg::hdg::euler::conserved_components; ++column)
  {
    ConservedState plus = state;
    ConservedState minus = state;
    const double step = 1.0e-7 *
        std::max(1.0, std::abs(state[static_cast<std::size_t>(column)]));
    plus[static_cast<std::size_t>(column)] += step;
    minus[static_cast<std::size_t>(column)] -= step;
    FluxVector plus_flux{}, minus_flux{};
    if (!NormalFlux(plus, normal, plus_flux).ok() ||
        !NormalFlux(minus, normal, minus_flux).ok())
    {
      return 13;
    }
    for (int row = 0; row < dg::hdg::euler::conserved_components; ++row)
    {
      const double finite_difference =
          (plus_flux[static_cast<std::size_t>(row)] -
           minus_flux[static_cast<std::size_t>(row)]) /
          (2.0 * step);
      if (!Near(finite_difference,
                jacobian[static_cast<std::size_t>(row)]
                          [static_cast<std::size_t>(column)],
                2.0e-6))
      {
        return 14;
      }
    }
  }

  FluxVector preserved_flux{{3.0, 5.0, 7.0, 11.0}};
  const FluxVector saved_flux = preserved_flux;
  ConservedState invalid = state;
  invalid[0] = -1.0;
  if (NormalFlux(invalid, normal, preserved_flux).status !=
          PhysicsStatus::invalid_density ||
      preserved_flux != saved_flux)
  {
    return 15;
  }
  FluxJacobian preserved_jacobian = expected_jacobian;
  const FluxJacobian saved_jacobian = preserved_jacobian;
  const std::array<double, 2> nonfinite_normal{{
      std::numeric_limits<double>::quiet_NaN(), 0.0}};
  if (NormalFluxJacobian(state, nonfinite_normal, preserved_jacobian).status !=
          PhysicsStatus::invalid_normal ||
      preserved_jacobian != saved_jacobian)
  {
    return 16;
  }
  return 0;
}

void PrintFailureEvidence(const DeviceEulerEvidence& evidence,
                          const int invocation)
{
  std::cerr << "HDG_EULER_DEVICE_FAILURE invocation=" << invocation
            << " status="
            << static_cast<int>(evidence.status)
            << " pass=" << evidence.pass
            << " cases=" << evidence.cases.size() << '\n';
  for (const auto& current : evidence.cases)
  {
    std::cerr << "HDG_EULER_DEVICE_FAILURE_CASE p=" << current.p
              << " status=" << static_cast<int>(current.status)
              << " generation=" << current.generation
              << " path=" << static_cast<int>(current.execution_path)
              << " fallback=" << current.accounting.host_reference_fallbacks
              << " persistent_bytes=" << current.accounting.persistent_bytes
              << " allocations=" << current.accounting.allocation_events
              << " h2d=" << current.accounting.host_to_device_bytes
              << " d2h=" << current.accounting.device_to_host_bytes
              << " steady_allocations="
              << current.accounting.steady_state_allocation_events
              << " full_trace_flow=" << current.full_trace_flow
              << " physics_fd="
              << current.max_physics_finite_difference_error
              << " hand_flux=" << current.max_hand_flux_error
              << " constant=" << current.constant_residual
              << " residual_parity=" << current.max_residual_parity_error
              << " jacobian_parity=" << current.max_jacobian_parity_error
              << " condensed_parity=" << current.max_condensed_parity_error
              << " action_parity=" << current.max_schur_action_parity_error
              << " trace_parity=" << current.max_trace_solution_parity_error
              << " recovery_parity=" << current.max_recovery_parity_error
              << " true_residual=" << current.true_residual << '\n';
  }
}

int ValidateDeviceEvidence(const DeviceEulerEvidence& evidence,
                           const int invocation)
{
  if (!evidence.pass || evidence.status != DeviceEulerStatus::success ||
      evidence.cases.size() != 4)
  {
    PrintFailureEvidence(evidence, invocation);
    return 20;
  }
  for (std::size_t index = 0; index < evidence.cases.size(); ++index)
  {
    const auto& current = evidence.cases[index];
    const int p = static_cast<int>(index) + 1;
    const int scalar_dofs = (p + 1) * (p + 1);
    const int trace_dofs = 4 * (p + 1);
    if (!current.pass || current.status != DeviceEulerStatus::success ||
        current.p != p || current.N != 3 || current.elements != 9 ||
        current.interior_unknowns !=
            dg::hdg::euler::conserved_components * scalar_dofs ||
        current.trace_unknowns !=
            dg::hdg::euler::conserved_components * trace_dofs ||
        current.generation == 0 ||
        current.max_physics_finite_difference_error > 2.0e-6 ||
        current.max_hand_flux_error > 5.0e-12 ||
        current.constant_residual > 5.0e-12 ||
        current.max_residual_parity_error > 5.0e-11 ||
        current.max_jacobian_parity_error > 5.0e-11 ||
        current.max_condensed_parity_error > 5.0e-11 ||
        current.max_schur_action_parity_error > 5.0e-9 ||
        current.max_recovery_parity_error > 5.0e-9 ||
        current.accounting.steady_state_allocation_events != 0)
    {
      PrintFailureEvidence(evidence, invocation);
      return 21 + p;
    }
    if (p == 1)
    {
      if (!current.full_trace_flow ||
          current.max_trace_solution_parity_error > 5.0e-9 ||
          current.true_residual > 1.0e-10)
      {
        PrintFailureEvidence(evidence, invocation);
        return 25;
      }
    }
    else if (current.full_trace_flow ||
             !std::isinf(current.max_trace_solution_parity_error) ||
             !std::isinf(current.true_residual))
    {
      PrintFailureEvidence(evidence, invocation);
      return 25 + p;
    }
#if defined(MFEM_USE_CUDA)
    if (current.execution_path != ExecutionPath::mfem_device ||
        current.accounting.host_reference_fallbacks != 0 ||
        current.accounting.persistent_bytes == 0 ||
        current.accounting.allocation_events == 0 ||
        current.accounting.host_to_device_bytes == 0 ||
        current.accounting.device_to_host_bytes == 0)
    {
      PrintFailureEvidence(evidence, invocation);
      return 30 + p;
    }
#else
    if (current.execution_path != ExecutionPath::host_reference ||
        current.accounting.host_reference_fallbacks == 0)
    {
      PrintFailureEvidence(evidence, invocation);
      return 30 + p;
    }
#endif
    std::cout << "HDG_EULER_DEVICE_CASE invocation=" << invocation
              << " p=" << current.p
              << " N=" << current.N
              << " path="
              << (current.execution_path == ExecutionPath::mfem_device ?
                      "mfem_device" : "host_reference")
              << " fallback=" << current.accounting.host_reference_fallbacks
              << " persistent_bytes=" << current.accounting.persistent_bytes
              << " allocations=" << current.accounting.allocation_events
              << " full_trace_flow=" << current.full_trace_flow
              << " residual_parity=" << current.max_residual_parity_error
              << " jacobian_parity=" << current.max_jacobian_parity_error
              << " condensed_parity=" << current.max_condensed_parity_error
              << " action_parity=" << current.max_schur_action_parity_error
              << " trace_parity=" << current.max_trace_solution_parity_error
              << " recovery_parity=" << current.max_recovery_parity_error
              << " true_residual=" << current.true_residual << '\n';
  }
  return 0;
}

int TestDeviceExperiment()
{
  for (int invocation = 1; invocation <= 2; ++invocation)
  {
    DeviceEulerEvidence evidence;
    if (!RunSmoothEulerDeviceExperiment(evidence))
    {
      PrintFailureEvidence(evidence, invocation);
      return 40 + invocation;
    }
    const int validation = ValidateDeviceEvidence(evidence, invocation);
    if (validation != 0) { return validation; }
  }
  return 0;
}
}  // namespace

int main()
{
  static_assert(!std::is_polymorphic<dg::hdg::euler::DeviceEulerEvidence>::value,
                "device experiment must not be polymorphic");
  if (!dg::hdg::euler::detail::PrepareSmoothEulerDeviceOracles())
  {
    std::cerr << "HDG_EULER_DEVICE_FAILURE stage=oracle_preparation\n";
    return 1;
  }
#if defined(MFEM_USE_CUDA)
  mfem::Device device("cuda");
#else
  mfem::Device device("cpu");
#endif
  (void)device;
  if (!mfem::Device::IsConfigured())
  {
    std::cerr << "HDG_EULER_DEVICE_FAILURE stage=device_configuration\n";
    return 2;
  }
  const int physics = TestIndependentFluxAndJacobian();
  if (physics != 0) { return physics; }
  return TestDeviceExperiment();
}

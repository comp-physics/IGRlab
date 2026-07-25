#pragma once

/**
 * @file numerics/hdg/euler/smooth_euler_device.hpp
 * @brief Static MFEM-native FP64 single-device smooth-Euler experiment.
 *
 * This default-off experiment is a correctness composition of the accepted
 * serial smooth-Euler formulation and the accepted fixed-shape device core.
 * It is not a supported solver, an MPI-device path, or a performance claim.
 */

#include "numerics/hdg/device/device_core.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace dg::hdg::euler
{
enum class DeviceEulerStatus : unsigned char
{
  success,
  setup_failure,
  projection_failure,
  assembly_failure,
  condensation_failure,
  trace_assembly_failure,
  trace_solve_failure,
  recovery_failure,
  oracle_failure,
  non_finite
};

struct DeviceEulerCaseEvidence
{
  int p = 0;
  int N = 0;
  DeviceEulerStatus status = DeviceEulerStatus::setup_failure;
  device::ExecutionPath execution_path = device::ExecutionPath::host_reference;
  device::TransferAccounting accounting{};
  std::uint64_t generation = 0;
  int elements = 0;
  int interior_unknowns = 0;
  int trace_unknowns = 0;
  bool full_trace_flow = false;
  double max_physics_finite_difference_error =
      std::numeric_limits<double>::infinity();
  double max_hand_flux_error = std::numeric_limits<double>::infinity();
  double constant_residual = std::numeric_limits<double>::infinity();
  double max_residual_parity_error = std::numeric_limits<double>::infinity();
  double max_jacobian_parity_error = std::numeric_limits<double>::infinity();
  double max_condensed_parity_error = std::numeric_limits<double>::infinity();
  double max_schur_action_parity_error =
      std::numeric_limits<double>::infinity();
  double max_trace_solution_parity_error =
      std::numeric_limits<double>::infinity();
  double max_recovery_parity_error = std::numeric_limits<double>::infinity();
  double true_residual = std::numeric_limits<double>::infinity();
  bool pass = false;
};

struct DeviceEulerEvidence
{
  DeviceEulerStatus status = DeviceEulerStatus::setup_failure;
  std::vector<DeviceEulerCaseEvidence> cases;
  bool pass = false;
};

/**
 * Runs p=1..4 at N=3 under the caller-configured MFEM backend, including the
 * complete trace flow for p=1. The caller retains sole mfem::Device ownership.
 */
bool RunSmoothEulerDeviceExperiment(DeviceEulerEvidence& evidence);

static_assert(!std::is_polymorphic<DeviceEulerCaseEvidence>::value,
              "smooth-Euler device evidence must remain static");
static_assert(!std::is_polymorphic<DeviceEulerEvidence>::value,
              "smooth-Euler device experiment must remain static");
}  // namespace dg::hdg::euler

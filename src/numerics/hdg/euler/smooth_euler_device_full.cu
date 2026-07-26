#include "numerics/hdg/euler/smooth_euler_device_full.hpp"

#include "../device/pivoted_solve.hpp"

#include <HYPRE_config.h>
#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>
#include <_hypre_parcsr_ls.h>
#include <_hypre_utilities.h>
#include <cuda_runtime.h>
#include <mfem.hpp>
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE)
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dg::hdg::euler
{
namespace
{
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double pivot_scale = 64.0 * experiment_epsilon;
constexpr double physics_gate = 2.0e-6;
constexpr double constant_gate = 5.0e-12;
constexpr double local_gate = 5.0e-11;
constexpr double action_gate = 5.0e-9;
constexpr double true_residual_gate = 1.0e-10;
constexpr int faces_per_element = 4;
constexpr std::size_t max_app_forall_records = 4096;
std::uint64_t process_marker_ordinal = 0;

template <typename Callable>
bool ExecuteMarkedZeroAllocationInterval(
    FullDeviceMeshEvidence& evidence, const std::string& label,
    Callable&& callable)
{
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE)
  const std::uint64_t identity = ++process_marker_ordinal;
  ++evidence.telemetry.external_zero_allocation_interval_sequence;
  const std::string identity_suffix =
      "_g" + std::to_string(evidence.committed_generation + 1) + "_i" +
      std::to_string(identity);
  const std::string begin =
      "DG_IGR_ZERO_ALLOC_BEGIN_" + label + identity_suffix;
  const std::string end =
      "DG_IGR_ZERO_ALLOC_END_" + label + identity_suffix;
  umpire::mark_event(begin);
  const bool result = std::forward<Callable>(callable)();
  umpire::mark_event(end);
  ++evidence.telemetry.external_zero_allocation_intervals_marked;
  return result;
#else
  (void)label;
  ++evidence.telemetry.fallback_events;
  return false;
#endif
}

template <typename Callable>
bool ExecuteMarkedAllocationChurnInterval(
    FullDeviceMeshEvidence& evidence, const std::string& label,
    Callable&& callable)
{
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE)
  const std::uint64_t identity = ++process_marker_ordinal;
  ++evidence.telemetry.external_zero_allocation_interval_sequence;
  const std::string identity_suffix =
      "_g" + std::to_string(evidence.committed_generation + 1) + "_i" +
      std::to_string(identity);
  const std::string begin =
      "DG_IGR_ALLOC_CHURN_BEGIN_" + label + identity_suffix;
  const std::string end =
      "DG_IGR_ALLOC_CHURN_END_" + label + identity_suffix;
  umpire::mark_event(begin);
  const bool result = std::forward<Callable>(callable)();
  umpire::mark_event(end);
  ++evidence.telemetry.external_allocation_churn_intervals_marked;
  return result;
#else
  (void)label;
  ++evidence.telemetry.fallback_events;
  return false;
#endif
}

std::size_t PhaseIndex(const FullDevicePhase phase) noexcept
{
  return static_cast<std::size_t>(phase);
}

void InitializePhases(FullDeviceMeshEvidence& evidence)
{
  for (std::size_t index = 0; index < evidence.phases.size(); ++index)
  {
    evidence.phases[index].phase = static_cast<FullDevicePhase>(index);
  }
}

void MarkPhase(FullDeviceMeshEvidence& evidence, const FullDevicePhase phase,
               const FullDeviceStatus status, const std::uint64_t generation,
               const bool resident)
{
  auto& record = evidence.phases[PhaseIndex(phase)];
  record.status = status;
  record.generation = generation;
  record.persistent_bytes = evidence.telemetry.persistent_bytes;
  record.allocation_events = evidence.telemetry.allocation_events;
  record.hypre_refresh_endpoint_growth_events =
      evidence.telemetry.hypre_refresh_endpoint_growth_events;
  record.hypre_refresh_endpoint_growth_bytes =
      evidence.telemetry.hypre_refresh_endpoint_growth_bytes;
  record.hypre_persistent_bytes = evidence.telemetry.hypre_persistent_bytes;
  record.hypre_setup_refresh_events =
      evidence.telemetry.hypre_setup_refresh_events;
  record.host_to_device_bytes = evidence.telemetry.host_to_device_bytes;
  record.device_to_host_bytes = evidence.telemetry.device_to_host_bytes;
  record.synchronization_events = evidence.telemetry.synchronization_events;
  record.fallback_events = evidence.telemetry.fallback_events;
  record.managed_device_objects_checked =
      evidence.telemetry.managed_device_objects_checked;
  record.managed_device_objects_valid =
      evidence.telemetry.managed_device_objects_valid;
  record.app_forall_submissions = evidence.telemetry.app_forall_submissions;
  record.app_forall_device_tickets =
      evidence.telemetry.app_forall_device_tickets;
  record.app_forall_host_fallbacks =
      evidence.telemetry.app_forall_host_fallbacks;
  record.reached = true;
  record.device_resident = resident;
  record.contemporaneous_audit = true;
  record.app_forall_census_complete =
      evidence.telemetry.app_forall_fallback_census_complete;
  record.mfem_internal_backend_observable = false;
  record.hypre_internal_backend_observable = false;
  evidence.phase_audit_log.push_back(record);
}

bool SamePhaseRecord(const FullDevicePhaseTelemetry& first,
                     const FullDevicePhaseTelemetry& second)
{
  return first.phase == second.phase && first.status == second.status &&
      first.generation == second.generation &&
      first.persistent_bytes == second.persistent_bytes &&
      first.allocation_events == second.allocation_events &&
      first.hypre_refresh_endpoint_growth_events ==
          second.hypre_refresh_endpoint_growth_events &&
      first.hypre_refresh_endpoint_growth_bytes ==
          second.hypre_refresh_endpoint_growth_bytes &&
      first.hypre_persistent_bytes == second.hypre_persistent_bytes &&
      first.hypre_setup_refresh_events ==
          second.hypre_setup_refresh_events &&
      first.host_to_device_bytes == second.host_to_device_bytes &&
      first.device_to_host_bytes == second.device_to_host_bytes &&
      first.synchronization_events == second.synchronization_events &&
      first.fallback_events == second.fallback_events &&
      first.managed_device_objects_checked ==
          second.managed_device_objects_checked &&
      first.managed_device_objects_valid ==
          second.managed_device_objects_valid &&
      first.app_forall_submissions == second.app_forall_submissions &&
      first.app_forall_device_tickets ==
          second.app_forall_device_tickets &&
      first.app_forall_host_fallbacks ==
          second.app_forall_host_fallbacks &&
      first.reached == second.reached &&
      first.device_resident == second.device_resident &&
      first.contemporaneous_audit == second.contemporaneous_audit &&
      first.app_forall_census_complete ==
          second.app_forall_census_complete &&
      first.mfem_internal_backend_observable ==
          second.mfem_internal_backend_observable &&
      first.hypre_internal_backend_observable ==
          second.hypre_internal_backend_observable;
}

bool SamePhaseLog(const std::vector<FullDevicePhaseTelemetry>& first,
                  const std::vector<FullDevicePhaseTelemetry>& second)
{
  return first.size() == second.size() &&
      std::equal(first.begin(), first.end(), second.begin(),
                 SamePhaseRecord);
}

void SealFailureBoundary(FullDeviceMeshEvidence& evidence,
                         const FullDevicePhase phase,
                         const FullDeviceStatus status,
                         const std::uint64_t generation)
{
  if (!evidence.failure_phase_audit_valid)
  {
    MarkPhase(evidence, phase, status, generation, false);
    evidence.failure_phase = phase;
    evidence.failure_phase_status = status;
    evidence.failure_phase_generation = generation;
    evidence.failure_phase_audit_valid = true;
  }
}

MFEM_HOST_DEVICE int LocalByNodes(const int scalar, const int scalar_count,
                                  const int component)
{
  return scalar + scalar_count * component;
}

MFEM_HOST_DEVICE bool FiniteFour(const double* state)
{
  for (int component = 0; component < conserved_components; ++component)
  {
    if (!isfinite(state[component])) { return false; }
  }
  return true;
}

MFEM_HOST_DEVICE bool FluxAndJacobian(const double* state, const double nx,
                                      const double ny, double* flux,
                                      double* jacobian)
{
  if (!FiniteFour(state) || !isfinite(nx) || !isfinite(ny) ||
      (nx == 0.0 && ny == 0.0) || !(state[0] > 0.0))
  {
    return false;
  }
  const double rho = state[0];
  const double mx = state[1];
  const double my = state[2];
  const double energy = state[3];
  const double kinetic = 0.5 * (mx * mx + my * my) / rho;
  const double pressure = (gamma - 1.0) * (energy - kinetic);
  if (!(pressure > 0.0) || !isfinite(pressure)) { return false; }
  const double q = mx * nx + my * ny;
  flux[0] = q;
  flux[1] = mx * q / rho + pressure * nx;
  flux[2] = my * q / rho + pressure * ny;
  flux[3] = (energy + pressure) * q / rho;

  const double pressure_rho =
      (gamma - 1.0) * 0.5 * (mx * mx + my * my) / (rho * rho);
  const double pressure_mx = -(gamma - 1.0) * mx / rho;
  const double pressure_my = -(gamma - 1.0) * my / rho;
  const double pressure_energy = gamma - 1.0;
  const double enthalpy = energy + pressure;
  for (int entry = 0; entry < conserved_components * conserved_components;
       ++entry)
  {
    jacobian[entry] = 0.0;
  }
  jacobian[1] = nx;
  jacobian[2] = ny;
  jacobian[4] = -mx * q / (rho * rho) + pressure_rho * nx;
  jacobian[5] = q / rho + mx * nx / rho + pressure_mx * nx;
  jacobian[6] = mx * ny / rho + pressure_my * nx;
  jacobian[7] = pressure_energy * nx;
  jacobian[8] = -my * q / (rho * rho) + pressure_rho * ny;
  jacobian[9] = my * nx / rho + pressure_mx * ny;
  jacobian[10] = q / rho + my * ny / rho + pressure_my * ny;
  jacobian[11] = pressure_energy * ny;
  jacobian[12] = pressure_rho * q / rho - enthalpy * q / (rho * rho);
  jacobian[13] = pressure_mx * q / rho + enthalpy * nx / rho;
  jacobian[14] = pressure_my * q / rho + enthalpy * ny / rho;
  jacobian[15] = (1.0 + pressure_energy) * q / rho;
  return FiniteFour(flux) && FiniteFour(jacobian) &&
         FiniteFour(jacobian + conserved_components) &&
         FiniteFour(jacobian + 2 * conserved_components) &&
         FiniteFour(jacobian + 3 * conserved_components);
}

MFEM_HOST_DEVICE void ManufacturedState(const double x, const double y,
                                        double* state)
{
  const double sx = sin(2.0 * pi * x);
  const double sy = sin(2.0 * pi * y);
  const double cx = cos(2.0 * pi * x);
  const double cy = cos(2.0 * pi * y);
  const double rho = 1.0 + 0.1 * sx * sy;
  const double u = 0.35 + 0.05 * cx * sy;
  const double v = -0.20 + 0.05 * sx * cy;
  const double pressure = 1.0 + 0.1 * cx * cy;
  state[0] = rho;
  state[1] = rho * u;
  state[2] = rho * v;
  state[3] = pressure / (gamma - 1.0) +
             0.5 * rho * (u * u + v * v);
}

MFEM_HOST_DEVICE void PerturbedState(const double x, const double y,
                                     double* state)
{
  const double sx = sin(2.0 * pi * x);
  const double sy = sin(2.0 * pi * y);
  const double cx = cos(2.0 * pi * x);
  const double cy = cos(2.0 * pi * y);
  double exact[conserved_components];
  ManufacturedState(x, y, exact);
  const double rho0 = exact[0];
  const double u0 = exact[1] / rho0;
  const double v0 = exact[2] / rho0;
  const double kinetic0 = 0.5 * (exact[1] * exact[1] + exact[2] * exact[2]) /
                          rho0;
  const double p0 = (gamma - 1.0) * (exact[3] - kinetic0);
  const double psi_one = sx * cy;
  const double psi_two = cx * sy;
  const double psi_three = sin(2.0 * pi * (x + y));
  const double rho = rho0 * (1.0 + 0.04 * psi_one);
  const double u = u0 + 0.03 * psi_two;
  const double v = v0 - 0.03 * psi_one;
  const double pressure = p0 * (1.0 + 0.04 * psi_three);
  state[0] = rho;
  state[1] = rho * u;
  state[2] = rho * v;
  state[3] = pressure / (gamma - 1.0) +
             0.5 * rho * (u * u + v * v);
}

MFEM_HOST_DEVICE bool ManufacturedSourceDevice(const double x, const double y,
                                                double* source)
{
  const double sx = sin(2.0 * pi * x);
  const double sy = sin(2.0 * pi * y);
  const double cx = cos(2.0 * pi * x);
  const double cy = cos(2.0 * pi * y);
  double state[conserved_components];
  ManufacturedState(x, y, state);
  const double rho = state[0];
  const double u = state[1] / rho;
  const double v = state[2] / rho;
  const double rho_x = 0.2 * pi * cx * sy;
  const double rho_y = 0.2 * pi * sx * cy;
  const double u_x = -0.1 * pi * sx * sy;
  const double u_y = 0.1 * pi * cx * cy;
  const double v_x = 0.1 * pi * cx * cy;
  const double v_y = -0.1 * pi * sx * sy;
  const double p_x = -0.2 * pi * sx * cy;
  const double p_y = -0.2 * pi * cx * sy;
  double derivative_x[conserved_components] = {
      rho_x, rho_x * u + rho * u_x, rho_x * v + rho * v_x,
      p_x / (gamma - 1.0) + 0.5 * rho_x * (u * u + v * v) +
          rho * (u * u_x + v * v_x)};
  double derivative_y[conserved_components] = {
      rho_y, rho_y * u + rho * u_y, rho_y * v + rho * v_y,
      p_y / (gamma - 1.0) + 0.5 * rho_y * (u * u + v * v) +
          rho * (u * u_y + v * v_y)};
  double unused[conserved_components];
  double jacobian_x[conserved_components * conserved_components];
  double jacobian_y[conserved_components * conserved_components];
  if (!FluxAndJacobian(state, 1.0, 0.0, unused, jacobian_x) ||
      !FluxAndJacobian(state, 0.0, 1.0, unused, jacobian_y))
  {
    return false;
  }
  for (int row = 0; row < conserved_components; ++row)
  {
    source[row] = 0.0;
    for (int column = 0; column < conserved_components; ++column)
    {
      source[row] += jacobian_x[row * conserved_components + column] *
                         derivative_x[column] +
                     jacobian_y[row * conserved_components + column] *
                         derivative_y[column];
    }
  }
  return FiniteFour(source);
}

MFEM_HOST_DEVICE bool AdmissibleState(const double* state, double& density,
                                      double& pressure, double& wave_speed)
{
  if (!FiniteFour(state) || !(state[0] >= 1.0e-8)) { return false; }
  density = state[0];
  const double kinetic =
      0.5 * (state[1] * state[1] + state[2] * state[2]) / state[0];
  pressure = (gamma - 1.0) * (state[3] - kinetic);
  if (!(pressure >= 1.0e-8) || !isfinite(pressure)) { return false; }
  const double u = state[1] / state[0];
  const double v = state[2] / state[0];
  wave_speed = sqrt(u * u + v * v) + sqrt(gamma * pressure / state[0]);
  return isfinite(wave_speed) && wave_speed < fixed_scalar_tau;
}

struct ErrorQuadratureStorage
{
  int volume_points = 0;
  int face_points = 0;
  mfem::Vector volume_shape;
  mfem::Vector volume_weight;
  mfem::Vector volume_position;
  mfem::Vector face_shape;
  mfem::Vector face_weight;
  mfem::Vector face_position;
  mfem::Array<int> face_signed_vdofs;
};

struct DeviceGraph
{
  std::unique_ptr<mfem::Mesh> serial_mesh;
  std::unique_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> interior_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
  std::unique_ptr<mfem::SparseMatrix> local_trace_matrix;
  std::unique_ptr<mfem::HypreParMatrix> true_trace_matrix;
  std::unique_ptr<mfem::HypreParVector> hypre_rhs;
  std::unique_ptr<mfem::HypreParVector> hypre_solution;
  std::unique_ptr<mfem::HypreILU> ilu;
  std::unique_ptr<mfem::HypreGMRES> gmres;
  mfem::Array<int> signed_trace_dofs;
  mfem::Array<int> element_to_true_csr;
  int p = 0;
  int N = 0;
  int elements = 0;
  int scalar_dofs = 0;
  int face_scalar_dofs = 0;
  int local_interior = 0;
  int local_trace = 0;
  int volume_points = 0;
  int face_points = 0;
  int local_trace_dofs = 0;
  int true_trace_dofs = 0;
  bool true_operator_ready = false;
  std::vector<int> host_signed_trace_dofs;
};

struct DeviceStorage
{
  mfem::Vector old_state, entry_state, current_state, base_state, trial_state;
  mfem::Vector entry_trace, current_trace, base_trace, trial_trace;
  mfem::Vector local_trace, local_trace_increment;
  mfem::Vector volume_shape, volume_derivative, volume_weight, volume_position;
  mfem::Vector face_interior_shape, face_trace_shape, face_weight, face_normal;
  mfem::Vector residual_r, residual_g, a, b, c, d, f, g;
  mfem::Vector lu, pivots, solve_workspace, schur, condensed_rhs;
  mfem::Vector local_sparse_rhs, true_rhs, true_action;
  mfem::Vector distributed_trace_increment;
  mfem::Vector recovered, delta_state;
  mfem::Vector physics_reduction, prescreen_reduction, kernel_status;
  mfem::Vector volume_error_reduction, face_error_reduction;
  mfem::Vector condensation_audit, recovery_audit, state_audit;
  mfem::Vector true_matrix_audit;
  mfem::Vector rollback_reduction;
  mfem::Vector app_forall_device_tickets;
  ErrorQuadratureStorage qe, qe4;
  std::uint64_t generation = 0;
  std::uint64_t residual_generation = 0;
  std::uint64_t condensation_generation = 0;
  std::uint64_t trace_system_generation = 0;
  std::uint64_t trace_solution_generation = 0;
  std::uint64_t recovery_generation = 0;
  bool hypre_solve_warmed = false;
  bool hypre_setup_refresh_required = false;
  std::array<std::uint64_t, max_app_forall_records>
      app_forall_host_fallback_tickets{};
  std::array<FullDevicePhase, max_app_forall_records>
      app_forall_phase{};
  std::array<std::uint64_t, max_app_forall_records>
      app_forall_generation{};
  std::array<std::size_t, max_app_forall_records>
      app_forall_kernel{};
  std::size_t app_forall_submissions = 0;
  std::size_t app_forall_rejected_submissions = 0;
  std::size_t active_phase_begin = 0;
  std::size_t active_kernel_ordinal = 0;
  FullDevicePhase active_phase = FullDevicePhase::setup;
  std::uint64_t active_phase_generation = 0;
  struct AllocatorSnapshot
  {
    std::size_t mfem_current_bytes = 0;
    std::size_t mfem_actual_bytes = 0;
    std::size_t mfem_high_watermark_bytes = 0;
    std::size_t mfem_active_allocations = 0;
    std::size_t hypre_current_bytes = 0;
    std::size_t hypre_actual_bytes = 0;
    std::size_t hypre_high_watermark_bytes = 0;
    std::size_t hypre_active_allocations = 0;
    bool complete = false;
  } allocator_baseline{};
};

bool BuildGraphAndStorage(int p, int N, DeviceGraph& graph,
                          DeviceStorage& storage,
                          FullDeviceMeshEvidence& evidence);
bool Prescreen(const DeviceGraph& graph, DeviceStorage& storage,
               FullDeviceMeshEvidence& evidence);
void GatherCurrentTrace(const DeviceGraph& graph, DeviceStorage& storage);
bool AssembleLinearization(const DeviceGraph& graph, DeviceStorage& storage,
                           FullDeviceMeshEvidence& evidence);
double ResidualNorm(const DeviceGraph& graph, DeviceStorage& storage,
                    FullDeviceTransferTelemetry& telemetry);
bool Condense(const DeviceGraph& graph, DeviceStorage& storage,
              FullDeviceMeshEvidence& evidence);
bool AssembleTraceSystem(DeviceGraph& graph, DeviceStorage& storage,
                         FullDeviceMeshEvidence& evidence);
bool SolveTrace(DeviceGraph& graph, DeviceStorage& storage,
                FullDeviceMeshEvidence& evidence);
bool Recover(DeviceGraph& graph, DeviceStorage& storage,
             FullDeviceMeshEvidence& evidence);
bool TrialState(const DeviceGraph& graph, DeviceStorage& storage,
                double alpha);
double StateUpdateError(const DeviceGraph& graph, DeviceStorage& storage,
                        double alpha, FullDeviceMeshEvidence& evidence);
bool RestoreEntryAndCheck(const DeviceGraph& graph, DeviceStorage& storage,
                          FullDeviceMeshEvidence& evidence,
                          bool& had_uncommitted_change);
bool IntegrateErrors(const DeviceGraph& graph, DeviceStorage& storage,
                     const ErrorQuadratureStorage& quadrature,
                     double& volume_relative_l2, double& trace_relative_l2,
                     FullDeviceTransferTelemetry& telemetry);
void CopyTestVector(const mfem::Vector& source,
                    std::vector<double>& destination,
                    FullDeviceTransferTelemetry& telemetry);
void CopyTestPrefix(const mfem::Vector& source, std::size_t size,
                    std::vector<double>& destination,
                    FullDeviceTransferTelemetry& telemetry);
bool CaptureAllocatorSnapshot(
    DeviceStorage::AllocatorSnapshot& snapshot,
    FullDeviceTransferTelemetry& telemetry);
bool AuditManagedResidency(const DeviceGraph& graph, DeviceStorage& storage,
                           FullDeviceMeshEvidence& evidence);
bool EndPhaseAudit(const DeviceGraph& graph, DeviceStorage& storage,
                   FullDeviceMeshEvidence& evidence, FullDevicePhase phase,
                   FullDeviceStatus status, std::uint64_t generation);
void InvalidateDerivedProducts(DeviceStorage& storage);
bool InvalidateAndProbeConsumers(DeviceGraph& graph, DeviceStorage& storage,
                                 FullDeviceMeshEvidence& evidence);
bool BuildErrorQuadrature(DeviceGraph& graph, int quadrature_order,
                          ErrorQuadratureStorage& storage,
                          FullDeviceTransferTelemetry& telemetry);

bool FitsInt(const std::size_t size) noexcept
{
  return size <= static_cast<std::size_t>(std::numeric_limits<int>::max());
}

bool StageVector(mfem::Vector& destination, const std::vector<double>& source,
                 FullDeviceTransferTelemetry& telemetry)
{
  if (!FitsInt(source.size())) { return false; }
  destination.SetSize(static_cast<int>(source.size()));
  destination.UseDevice(true);
  if (!source.empty())
  {
    std::copy(source.begin(), source.end(), destination.HostWrite());
    (void)destination.Read();
  }
  telemetry.persistent_bytes += source.size() * sizeof(double);
  telemetry.host_to_device_bytes += source.size() * sizeof(double);
  ++telemetry.allocation_events;
  return true;
}

bool StageArray(mfem::Array<int>& destination, const std::vector<int>& source,
                FullDeviceTransferTelemetry& telemetry)
{
  if (!FitsInt(source.size())) { return false; }
  destination.SetSize(static_cast<int>(source.size()));
  destination.GetMemory().UseDevice(true);
  if (!source.empty())
  {
    std::copy(source.begin(), source.end(), destination.HostWrite());
    (void)destination.Read();
  }
  telemetry.persistent_bytes += source.size() * sizeof(int);
  telemetry.host_to_device_bytes += source.size() * sizeof(int);
  ++telemetry.allocation_events;
  return true;
}

bool WarmVector(mfem::Vector& vector, const std::size_t size,
                FullDeviceTransferTelemetry& telemetry)
{
  if (size == 0 || !FitsInt(size)) { return false; }
  vector.SetSize(static_cast<int>(size));
  vector.UseDevice(true);
  vector = 0.0;
  telemetry.persistent_bytes += size * sizeof(double);
  ++telemetry.allocation_events;
  return true;
}

template <typename Body>
bool AppForall(DeviceStorage& storage, const int count, Body body)
{
  if (count <= 0 || !mfem::Device::Allows(mfem::Backend::CUDA) ||
      mfem::Device::Allows(mfem::Backend::DEBUG_DEVICE) ||
      storage.app_forall_device_tickets.Size() !=
          static_cast<int>(max_app_forall_records) ||
      storage.app_forall_submissions >= max_app_forall_records)
  {
    ++storage.app_forall_rejected_submissions;
    return false;
  }
  const std::size_t slot = storage.app_forall_submissions;
  const std::size_t kernel = storage.active_kernel_ordinal++;
  for (std::size_t prior = 0; prior < slot; ++prior)
  {
    if (storage.app_forall_phase[prior] == storage.active_phase &&
        storage.app_forall_generation[prior] ==
            storage.active_phase_generation &&
        storage.app_forall_kernel[prior] == kernel)
    {
      ++storage.app_forall_rejected_submissions;
      return false;
    }
  }
  storage.app_forall_phase[slot] = storage.active_phase;
  storage.app_forall_generation[slot] = storage.active_phase_generation;
  storage.app_forall_kernel[slot] = kernel;
  double* device_tickets = storage.app_forall_device_tickets.ReadWrite();
  std::uint64_t* host_tickets =
      storage.app_forall_host_fallback_tickets.data();
  mfem::ForallWrap<1>(
      true, count,
      [=] MFEM_HOST_DEVICE(int entry)
      {
        if (entry == 0) { device_tickets[slot] = 1.0; }
        body(entry);
      },
      [=](int entry)
      {
        if (entry == 0) { ++host_tickets[slot]; }
      });
  ++storage.app_forall_submissions;
  return true;
}

void BeginAppPhase(DeviceStorage& storage, const FullDevicePhase phase,
                   const std::uint64_t generation)
{
  storage.active_phase = phase;
  storage.active_phase_generation = generation;
  storage.active_phase_begin = storage.app_forall_submissions;
  storage.active_kernel_ordinal = storage.app_forall_submissions;
}

struct AppTicketAudit
{
  std::size_t total_device_tickets = 0;
  std::size_t total_host_fallbacks = 0;
  std::size_t phase_submissions = 0;
  std::size_t phase_device_tickets = 0;
  std::size_t phase_host_fallbacks = 0;
  bool complete = false;
  bool phase_complete = false;
};

AppTicketAudit AuditAppTickets(DeviceStorage& storage,
                               FullDeviceTransferTelemetry& telemetry,
                               const FullDevicePhase phase,
                               const std::uint64_t generation,
                               const bool phase_only)
{
  AppTicketAudit result;
  const std::size_t submissions = storage.app_forall_submissions;
  if (submissions > max_app_forall_records ||
      storage.active_phase_begin > submissions ||
      storage.app_forall_device_tickets.Size() !=
          static_cast<int>(max_app_forall_records))
  {
    return result;
  }
  const bool was_host_valid =
      storage.app_forall_device_tickets.GetMemory().HostIsValid();
  const double* device_tickets = submissions == 0 ? nullptr :
      storage.app_forall_device_tickets.GetMemory().Read(
          mfem::MemoryClass::HOST, static_cast<int>(submissions));
  if (submissions > 0 && !was_host_valid)
  {
    telemetry.device_to_host_bytes += submissions * sizeof(double);
    ++telemetry.synchronization_events;
  }
  bool all_valid = storage.app_forall_rejected_submissions == 0;
  bool phase_valid = all_valid;
  for (std::size_t slot = 0; slot < submissions; ++slot)
  {
    const bool device_valid = std::isfinite(device_tickets[slot]) &&
                              device_tickets[slot] == 1.0;
    const bool host_valid =
        storage.app_forall_host_fallback_tickets[slot] == 0;
    result.total_device_tickets += device_valid ? 1 : 0;
    result.total_host_fallbacks +=
        storage.app_forall_host_fallback_tickets[slot];
    all_valid = all_valid && device_valid && host_valid;
    if (slot >= storage.active_phase_begin)
    {
      ++result.phase_submissions;
      const std::size_t expected_kernel = slot;
      const bool identity_valid = storage.app_forall_phase[slot] == phase &&
          storage.app_forall_generation[slot] == generation &&
          storage.app_forall_kernel[slot] == expected_kernel;
      result.phase_device_tickets += device_valid ? 1 : 0;
      result.phase_host_fallbacks +=
          storage.app_forall_host_fallback_tickets[slot];
      phase_valid = phase_valid && device_valid && host_valid &&
                    identity_valid;
    }
  }
  result.complete = all_valid &&
      result.total_device_tickets == submissions &&
      result.total_host_fallbacks == 0;
  result.phase_complete = phase_valid &&
      (!phase_only || storage.active_phase == phase) &&
      (!phase_only || storage.active_phase_generation == generation) &&
      result.phase_device_tickets == result.phase_submissions &&
      result.phase_host_fallbacks == 0;
  return result;
}

bool CaptureAllocatorSnapshot(
    DeviceStorage::AllocatorSnapshot& snapshot,
    FullDeviceTransferTelemetry& telemetry)
{
  snapshot = {};
  if (cudaDeviceSynchronize() != cudaSuccess) { return false; }
  ++telemetry.synchronization_events;
  ++telemetry.allocator_snapshot_events;
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE)
  try
  {
    auto& manager = umpire::ResourceManager::getInstance();
    const char* mfem_name = mfem::MemoryManager::GetUmpireDeviceAllocatorName();
    const char* hypre_name =
        hypre_HandleUmpireDevicePoolName(hypre_handle());
    if (mfem_name == nullptr || hypre_name == nullptr || mfem_name[0] == '\0' ||
        hypre_name[0] == '\0' || !manager.isAllocator(mfem_name) ||
        !manager.isAllocator(hypre_name))
    {
      return false;
    }
    const umpire::Allocator mfem_allocator = manager.getAllocator(mfem_name);
    const umpire::Allocator hypre_allocator = manager.getAllocator(hypre_name);
    snapshot.mfem_current_bytes = mfem_allocator.getCurrentSize();
    snapshot.mfem_actual_bytes = mfem_allocator.getActualSize();
    snapshot.mfem_high_watermark_bytes = mfem_allocator.getHighWatermark();
    snapshot.mfem_active_allocations = mfem_allocator.getAllocationCount();
    snapshot.hypre_current_bytes = hypre_allocator.getCurrentSize();
    snapshot.hypre_actual_bytes = hypre_allocator.getActualSize();
    snapshot.hypre_high_watermark_bytes = hypre_allocator.getHighWatermark();
    snapshot.hypre_active_allocations = hypre_allocator.getAllocationCount();
    telemetry.mfem_allocator_high_watermark_bytes = std::max(
        telemetry.mfem_allocator_high_watermark_bytes,
        snapshot.mfem_high_watermark_bytes);
    telemetry.hypre_allocator_high_watermark_bytes = std::max(
        telemetry.hypre_allocator_high_watermark_bytes,
        snapshot.hypre_high_watermark_bytes);
    snapshot.complete = true;
    telemetry.allocator_endpoint_census_complete = true;
    return true;
  }
  catch (const std::exception&)
  {
    return false;
  }
#else
  (void)telemetry;
  return false;
#endif
}

bool EndPhaseAudit(const DeviceGraph& graph, DeviceStorage& storage,
                   FullDeviceMeshEvidence& evidence,
                   const FullDevicePhase phase, const FullDeviceStatus status,
                   const std::uint64_t generation)
{
  auto& telemetry = evidence.telemetry;
  if (cudaPeekAtLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
  {
    MarkPhase(evidence, phase, FullDeviceStatus::residency_failure,
              generation, false);
    evidence.failure_phase = phase;
    evidence.failure_phase_status = FullDeviceStatus::residency_failure;
    evidence.failure_phase_generation = generation;
    evidence.failure_phase_audit_valid = true;
    return false;
  }
  ++telemetry.synchronization_events;
  ++telemetry.phase_residency_audit_events;
  std::size_t checked = 0;
  std::size_t valid_count = 0;
  const auto valid = [&](const auto& memory, const int capacity,
                         const bool require_host_invalid)
  {
    ++checked;
    const mfem::MemoryType memory_type = memory.GetDeviceMemoryType();
    const bool result =
        memory.UseDevice() && memory.Capacity() >= capacity &&
        memory.DeviceIsValid() &&
        (memory_type == mfem::MemoryType::DEVICE ||
         memory_type == mfem::MemoryType::DEVICE_UMPIRE) &&
        (!require_host_invalid || !memory.HostIsValid());
    if (result) { ++valid_count; }
    return result;
  };
  const auto check_vector = [&](const mfem::Vector& vector,
                                const bool require_host_invalid)
  {
    return valid(vector.GetMemory(), vector.Size(), require_host_invalid);
  };
  bool resident = mfem::Device::Allows(mfem::Backend::CUDA) &&
                  !mfem::Device::Allows(mfem::Backend::DEBUG_DEVICE);
  switch (phase)
  {
    case FullDevicePhase::setup:
      resident = resident &&
          valid(graph.signed_trace_dofs.GetMemory(),
                graph.signed_trace_dofs.Size(), false) &&
          check_vector(storage.current_state, false) &&
          check_vector(storage.current_trace, false) &&
          check_vector(storage.local_trace, true) &&
          check_vector(storage.volume_shape, false) &&
          check_vector(storage.face_trace_shape, false);
      break;
    case FullDevicePhase::local_assembly:
      resident = resident && check_vector(storage.residual_r, true) &&
          check_vector(storage.residual_g, true) &&
          check_vector(storage.a, true) && check_vector(storage.b, true) &&
          check_vector(storage.c, true) && check_vector(storage.d, true) &&
          check_vector(storage.f, true) && check_vector(storage.g, true) &&
          check_vector(storage.kernel_status, true);
      break;
    case FullDevicePhase::condensation:
      resident = resident && check_vector(storage.lu, true) &&
          check_vector(storage.pivots, true) &&
          check_vector(storage.solve_workspace, true) &&
          check_vector(storage.schur, true) &&
          check_vector(storage.condensed_rhs, true) &&
          check_vector(storage.condensation_audit, true) &&
          check_vector(storage.kernel_status, true);
      break;
    case FullDevicePhase::trace_assembly:
      resident = resident &&
          valid(graph.element_to_true_csr.GetMemory(),
                graph.element_to_true_csr.Size(), false) &&
          check_vector(storage.local_sparse_rhs, true) &&
          check_vector(storage.true_rhs, true);
      break;
    case FullDevicePhase::trace_solve:
      resident = resident && check_vector(storage.true_action, true);
      break;
    case FullDevicePhase::recovery:
      resident = resident &&
          check_vector(storage.local_trace_increment, true) &&
          check_vector(storage.recovered, true) &&
          check_vector(storage.delta_state, true) &&
          check_vector(storage.recovery_audit, true);
      break;
    case FullDevicePhase::state_update:
      resident = resident && check_vector(storage.trial_state, true) &&
          check_vector(storage.trial_trace, true) &&
          check_vector(storage.current_state, true) &&
          check_vector(storage.current_trace, true);
      break;
    case FullDevicePhase::count:
      resident = false;
      break;
  }
  if (phase >= FullDevicePhase::trace_assembly)
  {
    resident = resident && graph.true_trace_matrix && graph.hypre_rhs &&
               graph.hypre_solution;
    if (graph.true_trace_matrix && graph.hypre_rhs && graph.hypre_solution)
    {
      hypre_ParCSRMatrix* raw_matrix = *graph.true_trace_matrix;
      hypre_ParVector* raw_rhs = *graph.hypre_rhs;
      hypre_ParVector* raw_solution = *graph.hypre_solution;
      const HYPRE_MemoryLocation matrix_location =
          hypre_ParCSRMatrixMemoryLocation(raw_matrix);
      const HYPRE_MemoryLocation rhs_location =
          hypre_ParVectorMemoryLocation(raw_rhs);
      const HYPRE_MemoryLocation solution_location =
          hypre_ParVectorMemoryLocation(raw_solution);
      resident = resident && matrix_location == HYPRE_MEMORY_DEVICE &&
          rhs_location == HYPRE_MEMORY_DEVICE &&
          solution_location == HYPRE_MEMORY_DEVICE &&
          hypre_GetExecPolicy1(matrix_location) == HYPRE_EXEC_DEVICE &&
          hypre_GetExecPolicy1(rhs_location) == HYPRE_EXEC_DEVICE &&
          hypre_GetExecPolicy1(solution_location) == HYPRE_EXEC_DEVICE &&
          hypre_GetExecPolicy2(matrix_location, rhs_location) ==
              HYPRE_EXEC_DEVICE &&
          hypre_GetExecPolicy2(matrix_location, solution_location) ==
              HYPRE_EXEC_DEVICE;
    }
  }

  const bool phase_has_submissions =
      storage.app_forall_submissions > storage.active_phase_begin;
  resident = resident && valid(
      storage.app_forall_device_tickets.GetMemory(),
      storage.app_forall_device_tickets.Size(), phase_has_submissions);
  const AppTicketAudit ticket_audit = AuditAppTickets(
      storage, telemetry, phase, generation, true);
  telemetry.app_forall_submissions = storage.app_forall_submissions;
  telemetry.app_forall_device_tickets = ticket_audit.total_device_tickets;
  telemetry.app_forall_host_fallbacks = ticket_audit.total_host_fallbacks;
  telemetry.app_forall_rejected_submissions =
      storage.app_forall_rejected_submissions;
  telemetry.backend_path_audit_events =
      ticket_audit.total_device_tickets + ticket_audit.total_host_fallbacks;
  telemetry.fallback_events = ticket_audit.total_host_fallbacks;
  telemetry.app_forall_fallback_census_complete = ticket_audit.complete;
  telemetry.mfem_internal_backend_observable = false;
  telemetry.hypre_internal_backend_observable = false;
  telemetry.fallback_census_complete = false;
  telemetry.cuda_kernel_execution_verified =
      telemetry.app_forall_fallback_census_complete &&
      ticket_audit.total_device_tickets > 0;
  telemetry.managed_device_objects_checked += checked;
  telemetry.managed_device_objects_valid += valid_count;
  resident = resident && checked == valid_count &&
      ticket_audit.phase_complete;
  MarkPhase(evidence, phase,
            resident ? status : FullDeviceStatus::residency_failure,
            generation, resident);
  auto& record = evidence.phases[PhaseIndex(phase)];
  record.managed_device_objects_checked = checked;
  record.managed_device_objects_valid = valid_count;
  record.app_forall_submissions = ticket_audit.phase_submissions;
  record.app_forall_device_tickets =
      ticket_audit.phase_device_tickets;
  record.app_forall_host_fallbacks =
      ticket_audit.phase_host_fallbacks;
  record.app_forall_census_complete = ticket_audit.phase_complete;
  evidence.phase_audit_log.back() = record;
  if (!resident && !evidence.failure_phase_audit_valid)
  {
    evidence.failure_phase = phase;
    evidence.failure_phase_status = FullDeviceStatus::residency_failure;
    evidence.failure_phase_generation = generation;
    evidence.failure_phase_audit_valid = true;
  }
  return resident;
}

bool AuditManagedResidency(const DeviceGraph& graph, DeviceStorage& storage,
                           FullDeviceMeshEvidence& evidence)
{
  auto& telemetry = evidence.telemetry;
  telemetry.managed_device_objects_checked = 0;
  telemetry.managed_device_objects_valid = 0;
  const auto check_memory = [&](const auto& memory)
  {
    ++telemetry.managed_device_objects_checked;
    const mfem::MemoryType memory_type = memory.GetDeviceMemoryType();
    if (memory.UseDevice() && memory.Capacity() > 0 &&
        memory.DeviceIsValid() &&
        (memory_type == mfem::MemoryType::DEVICE ||
         memory_type == mfem::MemoryType::DEVICE_UMPIRE))
    {
      ++telemetry.managed_device_objects_valid;
    }
  };
  const std::initializer_list<const mfem::Vector*> vectors{
      &storage.old_state,
      &storage.entry_state,
      &storage.current_state,
      &storage.base_state,
      &storage.trial_state,
      &storage.entry_trace,
      &storage.current_trace,
      &storage.base_trace,
      &storage.trial_trace,
      &storage.local_trace,
      &storage.local_trace_increment,
      &storage.volume_shape,
      &storage.volume_derivative,
      &storage.volume_weight,
      &storage.volume_position,
      &storage.face_interior_shape,
      &storage.face_trace_shape,
      &storage.face_weight,
      &storage.face_normal,
      &storage.residual_r,
      &storage.residual_g,
      &storage.a,
      &storage.b,
      &storage.c,
      &storage.d,
      &storage.f,
      &storage.g,
      &storage.lu,
      &storage.pivots,
      &storage.solve_workspace,
      &storage.schur,
      &storage.condensed_rhs,
      &storage.local_sparse_rhs,
      &storage.true_rhs,
      &storage.true_action,
      &storage.distributed_trace_increment,
      &storage.recovered,
      &storage.delta_state,
      &storage.physics_reduction,
      &storage.prescreen_reduction,
      &storage.kernel_status,
      &storage.volume_error_reduction,
      &storage.face_error_reduction,
      &storage.condensation_audit,
      &storage.recovery_audit,
      &storage.state_audit,
      &storage.true_matrix_audit,
      &storage.rollback_reduction,
      &storage.app_forall_device_tickets,
      &storage.qe.volume_shape,
      &storage.qe.volume_weight,
      &storage.qe.volume_position,
      &storage.qe.face_shape,
      &storage.qe.face_weight,
      &storage.qe.face_position,
      &storage.qe4.volume_shape,
      &storage.qe4.volume_weight,
      &storage.qe4.volume_position,
      &storage.qe4.face_shape,
      &storage.qe4.face_weight,
      &storage.qe4.face_position};
  for (const mfem::Vector* vector : vectors)
    check_memory(vector->GetMemory());
  check_memory(graph.signed_trace_dofs.GetMemory());
  check_memory(graph.element_to_true_csr.GetMemory());
  check_memory(storage.qe.face_signed_vdofs.GetMemory());
  check_memory(storage.qe4.face_signed_vdofs.GetMemory());
  bool hypre_resident = graph.true_trace_matrix && graph.hypre_rhs &&
                        graph.hypre_solution;
  if (hypre_resident)
  {
    hypre_ParCSRMatrix* raw_matrix = *graph.true_trace_matrix;
    hypre_ParVector* raw_rhs = *graph.hypre_rhs;
    hypre_ParVector* raw_solution = *graph.hypre_solution;
    hypre_resident =
        hypre_ParCSRMatrixMemoryLocation(raw_matrix) == HYPRE_MEMORY_DEVICE &&
        hypre_ParVectorMemoryLocation(raw_rhs) == HYPRE_MEMORY_DEVICE &&
        hypre_ParVectorMemoryLocation(raw_solution) == HYPRE_MEMORY_DEVICE;
  }
  const AppTicketAudit ticket_audit = AuditAppTickets(
      storage, telemetry, storage.active_phase,
      storage.active_phase_generation, false);
  telemetry.app_forall_submissions = storage.app_forall_submissions;
  telemetry.app_forall_device_tickets = ticket_audit.total_device_tickets;
  telemetry.app_forall_host_fallbacks = ticket_audit.total_host_fallbacks;
  telemetry.app_forall_rejected_submissions =
      storage.app_forall_rejected_submissions;
  telemetry.backend_path_audit_events =
      telemetry.app_forall_device_tickets +
      telemetry.app_forall_host_fallbacks;
  telemetry.fallback_events = telemetry.app_forall_host_fallbacks;
  telemetry.app_forall_fallback_census_complete = ticket_audit.complete;
  telemetry.cuda_kernel_execution_verified =
      telemetry.app_forall_fallback_census_complete &&
      telemetry.app_forall_device_tickets > 0;
  evidence.cuda_kernel_execution_verified =
      telemetry.cuda_kernel_execution_verified;
  evidence.managed_residency_verified =
      telemetry.managed_device_objects_checked > 0 &&
      telemetry.managed_device_objects_checked ==
          telemetry.managed_device_objects_valid &&
      telemetry.cuda_kernel_execution_verified && hypre_resident;
  evidence.allocation_observability_reported =
      telemetry.allocator_endpoint_census_complete &&
      !telemetry.whole_transaction_zero_allocation_claimed &&
      telemetry.hypre_setup_refresh_allocation_churn_reported &&
      telemetry.external_umpire_event_audit_required &&
      !telemetry.external_umpire_event_results_available_in_process &&
      telemetry.marked_zero_allocation_scope_excludes_trace_assembly_solve &&
      telemetry.external_zero_allocation_intervals_marked > 0 &&
      telemetry.external_allocation_churn_intervals_marked > 0;
  evidence.app_forall_fallback_census_complete =
      telemetry.app_forall_fallback_census_complete;
  evidence.mfem_internal_backend_observable = false;
  evidence.hypre_internal_backend_observable = false;
  evidence.fallback_census_complete = false;
  telemetry.mfem_internal_backend_observable = false;
  telemetry.hypre_internal_backend_observable = false;
  telemetry.fallback_census_complete = false;
  return evidence.managed_residency_verified &&
         evidence.allocation_observability_reported &&
         evidence.app_forall_fallback_census_complete;
}

void ConfigureFrozenILU(mfem::HypreILU& ilu)
{
  ilu.SetType(0);
  ilu.SetLevelOfFill(0);
  ilu.SetMaxIter(1);
  ilu.SetTol(0.0);
  ilu.SetPrintLevel(0);
}

bool VerifyFrozenILU(const DeviceGraph& graph,
                     FullDeviceMeshEvidence& evidence)
{
  if (!graph.ilu) { return false; }
  auto* data = reinterpret_cast<hypre_ParILUData*>(
      static_cast<HYPRE_Solver>(*graph.ilu));
  if (data == nullptr) { return false; }
  evidence.effective_hypre_ilu_type = hypre_ParILUDataIluType(data);
  evidence.effective_hypre_ilu_level_of_fill = hypre_ParILUDataLfil(data);
  ++evidence.telemetry.hypre_ilu_configuration_checks;
  return evidence.effective_hypre_ilu_type == 0 &&
         evidence.effective_hypre_ilu_level_of_fill == 0;
}

}  // namespace
}  // namespace dg::hdg::euler

namespace dg::hdg::euler
{
namespace
{
bool RunPhysicsChecks(DeviceStorage& storage,
                      FullDeviceMeshEvidence& evidence)
{
  double* output = storage.physics_reduction.Write();
  AppForall(storage, 1, [=] MFEM_HOST_DEVICE(int)
  {
    const double state[conserved_components] = {1.2, 0.48, -0.3, 2.3835};
    const double nx = 0.6;
    const double ny = 0.8;
    double flux[conserved_components];
    double jacobian[conserved_components * conserved_components];
    double maximum_fd = 0.0;
    double maximum_hand = 0.0;
    if (!FluxAndJacobian(state, nx, ny, flux, jacobian))
    {
      output[0] = output[1] = output[2] = 1.0e300;
      return;
    }
    const double expected_flux[conserved_components] =
        {0.048, 0.5592, 0.708, 0.13134};
    for (int row = 0; row < conserved_components; ++row)
    {
      maximum_hand = fmax(maximum_hand, fabs(flux[row] - expected_flux[row]));
    }
    for (int column = 0; column < conserved_components; ++column)
    {
      double plus[conserved_components];
      double minus[conserved_components];
      for (int component = 0; component < conserved_components; ++component)
      {
        plus[component] = minus[component] = state[component];
      }
      const double step = 1.0e-7 * fmax(1.0, fabs(state[column]));
      plus[column] += step;
      minus[column] -= step;
      double plus_flux[conserved_components], minus_flux[conserved_components];
      double unused[conserved_components * conserved_components];
      if (!FluxAndJacobian(plus, nx, ny, plus_flux, unused) ||
          !FluxAndJacobian(minus, nx, ny, minus_flux, unused))
      {
        maximum_fd = 1.0e300;
        break;
      }
      for (int row = 0; row < conserved_components; ++row)
      {
        const double finite_difference =
            (plus_flux[row] - minus_flux[row]) / (2.0 * step);
        maximum_fd = fmax(maximum_fd,
                          fabs(finite_difference -
                               jacobian[row * conserved_components + column]) /
                              fmax(1.0, fabs(jacobian[
                                  row * conserved_components + column])));
      }
    }
    double constant_residual = 0.0;
    const double normals[8] = {1.0, 0.0, -1.0, 0.0,
                               0.0, 1.0, 0.0, -1.0};
    double paired[conserved_components] = {0.0, 0.0, 0.0, 0.0};
    for (int face = 0; face < 4; ++face)
    {
      double face_flux[conserved_components];
      double unused[conserved_components * conserved_components];
      if (!FluxAndJacobian(state, normals[2 * face], normals[2 * face + 1],
                           face_flux, unused))
      {
        constant_residual = 1.0e300;
        break;
      }
      for (int component = 0; component < conserved_components; ++component)
      {
        paired[component] += face_flux[component];
      }
    }
    for (int component = 0; component < conserved_components; ++component)
    {
      constant_residual = fmax(constant_residual, fabs(paired[component]));
    }
    output[0] = maximum_fd;
    output[1] = maximum_hand;
    output[2] = constant_residual;
  });
  const double* host = storage.physics_reduction.HostRead();
  evidence.telemetry.device_to_host_bytes += 3 * sizeof(double);
  ++evidence.telemetry.synchronization_events;
  evidence.max_physics_finite_difference_error = host[0];
  evidence.constant_state_residual = std::max(host[1], host[2]);
  return std::isfinite(host[0]) && std::isfinite(host[1]) &&
         std::isfinite(host[2]) && host[0] <= physics_gate &&
         host[1] <= constant_gate && host[2] <= constant_gate;
}

void CaptureInitialSnapshot(const DeviceGraph& graph,
                            const DeviceStorage& storage,
                            FullDeviceMeshEvidence& evidence,
                            const bool stage_global_linear_oracle)
{
  auto& snapshot = evidence.test_snapshot;
  snapshot.polynomial_order = graph.p;
  snapshot.elements_per_direction = graph.N;
  snapshot.elements = graph.elements;
  snapshot.interior_dofs = graph.local_interior;
  snapshot.trace_dofs = graph.local_trace;
  snapshot.global_trace_dofs = graph.local_trace_dofs;
  snapshot.assembly_quadrature_order = 5 * graph.p + 16;
  CopyTestVector(storage.old_state, snapshot.old_interior, evidence.telemetry);
  CopyTestVector(storage.entry_state, snapshot.initial_interior,
                 evidence.telemetry);
  CopyTestVector(storage.entry_trace, snapshot.initial_trace,
                 evidence.telemetry);
  const std::size_t ni = static_cast<std::size_t>(graph.local_interior);
  const std::size_t nt = static_cast<std::size_t>(graph.local_trace);
  CopyTestPrefix(storage.a, ni * ni, snapshot.first_a, evidence.telemetry);
  CopyTestPrefix(storage.b, ni * nt, snapshot.first_b, evidence.telemetry);
  CopyTestPrefix(storage.c, nt * ni, snapshot.first_c, evidence.telemetry);
  CopyTestPrefix(storage.d, nt * nt, snapshot.first_d, evidence.telemetry);
  CopyTestPrefix(storage.f, ni, snapshot.first_f, evidence.telemetry);
  CopyTestPrefix(storage.g, nt, snapshot.first_g, evidence.telemetry);
  if (stage_global_linear_oracle)
  {
    CopyTestVector(storage.residual_r, snapshot.global_residual_r,
                   evidence.telemetry);
    CopyTestVector(storage.residual_g, snapshot.global_residual_g,
                   evidence.telemetry);
    CopyTestVector(storage.a, snapshot.global_a, evidence.telemetry);
    CopyTestVector(storage.b, snapshot.global_b, evidence.telemetry);
    CopyTestVector(storage.c, snapshot.global_c, evidence.telemetry);
    CopyTestVector(storage.d, snapshot.global_d, evidence.telemetry);
    CopyTestVector(storage.f, snapshot.global_f, evidence.telemetry);
    CopyTestVector(storage.g, snapshot.global_g, evidence.telemetry);
  }
}

void CaptureFirstLinearSnapshot(const DeviceGraph& graph,
                                const DeviceStorage& storage,
                                FullDeviceMeshEvidence& evidence,
                                const bool stage_global_linear_oracle)
{
  const std::size_t ni = static_cast<std::size_t>(graph.local_interior);
  const std::size_t nt = static_cast<std::size_t>(graph.local_trace);
  auto& snapshot = evidence.test_snapshot;
  CopyTestPrefix(storage.schur, nt * nt, snapshot.first_schur,
                 evidence.telemetry);
  CopyTestPrefix(storage.condensed_rhs, nt,
                 snapshot.first_condensed_rhs, evidence.telemetry);
  CopyTestPrefix(storage.local_trace_increment, nt,
                 snapshot.first_local_trace_increment, evidence.telemetry);
  CopyTestPrefix(storage.recovered, ni, snapshot.first_recovered_increment,
                 evidence.telemetry);
  if (stage_global_linear_oracle)
  {
    CopyTestVector(storage.schur, snapshot.global_schur, evidence.telemetry);
    CopyTestVector(storage.condensed_rhs, snapshot.global_condensed_rhs,
                   evidence.telemetry);
    CopyTestVector(storage.distributed_trace_increment,
                   snapshot.global_trace_increment, evidence.telemetry);
    CopyTestVector(storage.local_trace_increment,
                   snapshot.global_local_trace_increment,
                   evidence.telemetry);
    CopyTestVector(storage.recovered, snapshot.global_recovered_increment,
                   evidence.telemetry);
    const int rows = graph.true_trace_dofs;
    const auto& offset_memory = graph.true_trace_matrix->GetDiagMemoryI();
    const auto& column_memory = graph.true_trace_matrix->GetDiagMemoryJ();
    const auto& value_memory = graph.true_trace_matrix->GetDiagMemoryData();
    const HYPRE_Int* offsets =
        offset_memory.Read(mfem::MemoryClass::HOST, rows + 1);
    const int nonzeros = static_cast<int>(offsets[rows]);
    if (nonzeros >= 0 && nonzeros <= column_memory.Capacity() &&
        nonzeros <= value_memory.Capacity())
    {
      const HYPRE_Int* columns =
          column_memory.Read(mfem::MemoryClass::HOST, nonzeros);
      const HYPRE_Real* values =
          value_memory.Read(mfem::MemoryClass::HOST, nonzeros);
      snapshot.actual_true_csr_offsets.resize(rows + 1);
      snapshot.actual_true_csr_columns.resize(nonzeros);
      snapshot.actual_true_csr_values.resize(nonzeros);
      for (int entry = 0; entry <= rows; ++entry)
        snapshot.actual_true_csr_offsets[entry] =
            static_cast<int>(offsets[entry]);
      for (int entry = 0; entry < nonzeros; ++entry)
      {
        snapshot.actual_true_csr_columns[entry] =
            static_cast<int>(columns[entry]);
        snapshot.actual_true_csr_values[entry] = values[entry];
      }
      const std::size_t bytes =
          static_cast<std::size_t>(rows + 1) * sizeof(HYPRE_Int) +
          static_cast<std::size_t>(nonzeros) *
              (sizeof(HYPRE_Int) + sizeof(HYPRE_Real));
      evidence.telemetry.device_to_host_bytes += bytes;
      evidence.telemetry.test_only_device_to_host_bytes += bytes;
      evidence.telemetry.synchronization_events += 3;
    }
    CopyTestVector(static_cast<const mfem::Vector&>(*graph.hypre_rhs),
                   snapshot.actual_true_rhs, evidence.telemetry);
    snapshot.global_linear_oracle_staged = true;
  }
}

void CaptureFinalSnapshot(const DeviceGraph& graph, const DeviceStorage& storage,
                          FullDeviceMeshEvidence& evidence,
                          const bool stage_global_linear_oracle,
                          const double last_accepted_alpha)
{
  auto& snapshot = evidence.test_snapshot;
  CopyTestVector(storage.current_state, snapshot.final_interior,
                 evidence.telemetry);
  CopyTestVector(storage.current_trace, snapshot.final_trace,
                 evidence.telemetry);
  snapshot.signed_trace_dofs = graph.host_signed_trace_dofs;
  if (stage_global_linear_oracle)
  {
    CopyTestVector(storage.residual_r, snapshot.final_residual_r,
                   evidence.telemetry);
    CopyTestVector(storage.residual_g, snapshot.final_residual_g,
                   evidence.telemetry);
    CopyTestVector(storage.base_state, snapshot.last_base_interior,
                   evidence.telemetry);
    CopyTestVector(storage.base_trace, snapshot.last_base_trace,
                   evidence.telemetry);
    CopyTestVector(storage.delta_state, snapshot.last_delta_interior,
                   evidence.telemetry);
    CopyTestVector(storage.distributed_trace_increment,
                   snapshot.last_delta_trace, evidence.telemetry);
    snapshot.last_accepted_alpha = last_accepted_alpha;
  }
}

FullDevicePhase FailurePhase(const FullDeviceStatus status)
{
  switch (status)
  {
    case FullDeviceStatus::assembly_failure:
      return FullDevicePhase::local_assembly;
    case FullDeviceStatus::condensation_failure:
      return FullDevicePhase::condensation;
    case FullDeviceStatus::trace_assembly_failure:
      return FullDevicePhase::trace_assembly;
    case FullDeviceStatus::trace_solve_failure:
    case FullDeviceStatus::linear_residual_failure:
      return FullDevicePhase::trace_solve;
    case FullDeviceStatus::recovery_failure:
      return FullDevicePhase::recovery;
    case FullDeviceStatus::line_search_exhausted:
    case FullDeviceStatus::iteration_limit:
    case FullDeviceStatus::injected_failure:
      return FullDevicePhase::state_update;
    default:
      return FullDevicePhase::setup;
  }
}

void SanitizeFailedTransaction(FullDeviceMeshEvidence& evidence,
                               const FullDeviceStatus status,
                               const std::uint64_t attempted_generation)
{
  FullDeviceTestSnapshot rollback_only;
  rollback_only.rollback_entry_interior =
      std::move(evidence.test_snapshot.rollback_entry_interior);
  rollback_only.rollback_changed_interior =
      std::move(evidence.test_snapshot.rollback_changed_interior);
  rollback_only.rollback_restored_interior =
      std::move(evidence.test_snapshot.rollback_restored_interior);
  rollback_only.rollback_entry_trace =
      std::move(evidence.test_snapshot.rollback_entry_trace);
  rollback_only.rollback_changed_trace =
      std::move(evidence.test_snapshot.rollback_changed_trace);
  rollback_only.rollback_restored_trace =
      std::move(evidence.test_snapshot.rollback_restored_trace);
  evidence.test_snapshot = std::move(rollback_only);
  evidence.discarded_nonlinear_iterations = evidence.nonlinear_iterations;
  evidence.discarded_linear_iterations = evidence.linear_iterations;
  evidence.discarded_attempt_generation = attempted_generation;
  evidence.discarded_accepted_residual = evidence.final_residual;
  evidence.nonlinear_iterations = 0;
  evidence.linear_iterations = 0;
  evidence.committed_generation = 0;
  evidence.final_residual = evidence.initial_residual;
  evidence.true_relative_residual = std::numeric_limits<double>::infinity();
  evidence.schur_action_error = std::numeric_limits<double>::infinity();
  evidence.trace_solution_error = std::numeric_limits<double>::infinity();
  evidence.recovery_error = std::numeric_limits<double>::infinity();
  evidence.state_update_error = std::numeric_limits<double>::infinity();
  evidence.hypre_matrix_device = false;
  evidence.hypre_rhs_device = false;
  evidence.hypre_solution_device = false;
  evidence.hypre_gmres = false;
  evidence.hypre_ilu_type_zero_fill_zero = false;
  evidence.hypre_policy_before_solve_device = false;
  evidence.hypre_policy_after_solve_device = false;
  evidence.effective_hypre_ilu_type = -1;
  evidence.effective_hypre_ilu_level_of_fill = -1;
  evidence.status = status;
  evidence.pass = false;
}

bool RunMeshCase(const FullDeviceRequest& request, const int p, const int N,
                 FullDeviceMeshEvidence& evidence)
{
  evidence = {};
  evidence.polynomial_order = p;
  evidence.elements_per_direction = N;
  InitializePhases(evidence);
  DeviceGraph graph;
  DeviceStorage storage;
  const auto fail = [&](const FullDeviceStatus status)
  {
    bool had_uncommitted_change = false;
    const std::uint64_t attempted_generation = storage.generation;
    SealFailureBoundary(evidence, FailurePhase(status), status,
                        attempted_generation);
    const std::vector<FullDevicePhaseTelemetry> sealed_phase_log =
        evidence.phase_audit_log;
    const bool restored = RestoreEntryAndCheck(
        graph, storage, evidence, had_uncommitted_change);
    evidence.rollback_had_uncommitted_change = had_uncommitted_change;
    const bool rejected =
        InvalidateAndProbeConsumers(graph, storage, evidence);
    const bool residency = AuditManagedResidency(graph, storage, evidence);
    SanitizeFailedTransaction(evidence, status, attempted_generation);
    evidence.phase_audit_append_only =
        SamePhaseLog(sealed_phase_log, evidence.phase_audit_log);
    if (!restored || !residency || !rejected)
    {
      evidence.status = (!restored || !rejected) ?
          FullDeviceStatus::rollback_failure : FullDeviceStatus::residency_failure;
    }
    return false;
  };

  if (!BuildGraphAndStorage(p, N, graph, storage, evidence))
  {
    SealFailureBoundary(evidence, FullDevicePhase::setup,
                        FullDeviceStatus::setup_failure, 0);
    evidence.status = FullDeviceStatus::setup_failure;
    return false;
  }
  const std::string marker_suffix = "_p" + std::to_string(p) + "_N" +
                                    std::to_string(N);
  if (!RunPhysicsChecks(storage, evidence))
  {
    return fail(FullDeviceStatus::non_finite);
  }
  if (!Prescreen(graph, storage, evidence))
  {
    return fail(FullDeviceStatus::initial_admissibility_failure);
  }
  if (!EndPhaseAudit(graph, storage, evidence, FullDevicePhase::setup,
                     FullDeviceStatus::success, 0))
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  if (!ExecuteMarkedZeroAllocationInterval(
          evidence, "local_assembly" + marker_suffix,
          [&]() { return AssembleLinearization(graph, storage, evidence); }))
  {
    return fail(FullDeviceStatus::assembly_failure);
  }
  double residual = ResidualNorm(graph, storage, evidence.telemetry);
  if (!std::isfinite(residual) || !(residual > 1.0e-8))
  {
    return fail(FullDeviceStatus::assembly_failure);
  }
  evidence.initial_residual = residual;
  evidence.final_residual = residual;
  if (request.enable_test_oracle_staging)
  {
    CaptureInitialSnapshot(graph, storage, evidence,
                           request.enable_global_linear_oracle_staging);
  }
  const double target = 1.0e-10 * (1.0 + residual);
  bool captured_first_linear = false;
  double last_accepted_alpha = 0.0;
  for (int nonlinear_iteration = 0; nonlinear_iteration < 20;
       ++nonlinear_iteration)
  {
    if (evidence.nonlinear_iterations >= 2 && residual <= target) { break; }
    storage.base_state = storage.current_state;
    storage.base_trace = storage.current_trace;
    if (!ExecuteMarkedZeroAllocationInterval(
            evidence, "condensation" + marker_suffix,
            [&]() { return Condense(graph, storage, evidence); }))
    {
      return fail(FullDeviceStatus::condensation_failure);
    }
    if (!AssembleTraceSystem(graph, storage, evidence))
    {
      return fail(FullDeviceStatus::trace_assembly_failure);
    }
    if (!ExecuteMarkedAllocationChurnInterval(
            evidence, "trace_solve" + marker_suffix,
            [&]() { return SolveTrace(graph, storage, evidence); }))
    {
      const FullDeviceStatus status = evidence.status ==
              FullDeviceStatus::linear_residual_failure ?
          FullDeviceStatus::linear_residual_failure :
          FullDeviceStatus::trace_solve_failure;
      return fail(status);
    }
    if (!ExecuteMarkedZeroAllocationInterval(
            evidence, "recovery" + marker_suffix,
            [&]() { return Recover(graph, storage, evidence); }))
    {
      return fail(FullDeviceStatus::recovery_failure);
    }
    if (request.enable_test_oracle_staging && !captured_first_linear)
    {
      CaptureFirstLinearSnapshot(graph, storage, evidence,
                                 request.enable_global_linear_oracle_staging);
      captured_first_linear = true;
    }

    bool accepted = false;
    for (int trial_index = 0; trial_index <= 12; ++trial_index)
    {
      const double alpha = std::ldexp(1.0, -trial_index);
      double update_error = std::numeric_limits<double>::infinity();
      if (!ExecuteMarkedZeroAllocationInterval(
              evidence, "state_update" + marker_suffix,
              [&]()
              {
                if (!TrialState(graph, storage, alpha)) { return false; }
                update_error = StateUpdateError(
                    graph, storage, alpha, evidence);
                return true;
              }))
      {
        return fail(FullDeviceStatus::recovery_failure);
      }
      if (!std::isfinite(update_error) || update_error > action_gate)
      {
        return fail(FullDeviceStatus::non_finite);
      }
      evidence.state_update_error =
          std::isfinite(evidence.state_update_error) ?
              std::max(evidence.state_update_error, update_error) :
              update_error;
      if (!Prescreen(graph, storage, evidence))
      {
        storage.current_state = storage.base_state;
        storage.current_trace = storage.base_trace;
        ++evidence.rejected_trials;
        continue;
      }
      if (!ExecuteMarkedZeroAllocationInterval(
              evidence, "local_assembly" + marker_suffix,
              [&]()
              { return AssembleLinearization(graph, storage, evidence); }))
      {
        return fail(FullDeviceStatus::assembly_failure);
      }
      const double trial_residual =
          ResidualNorm(graph, storage, evidence.telemetry);
      if (!std::isfinite(trial_residual))
      {
        return fail(FullDeviceStatus::assembly_failure);
      }
      if (trial_residual > (1.0 - 1.0e-4 * alpha) * residual)
      {
        storage.current_state = storage.base_state;
        storage.current_trace = storage.base_trace;
        ++evidence.rejected_trials;
        continue;
      }
      residual = trial_residual;
      last_accepted_alpha = alpha;
      evidence.final_residual = residual;
      ++evidence.nonlinear_iterations;
      evidence.committed_generation =
          static_cast<std::uint64_t>(evidence.nonlinear_iterations);
      accepted = true;
      if (request.inject_failure_after_first_update &&
          p == request.injected_failure_polynomial_order &&
          N == request.injected_failure_elements_per_direction &&
          evidence.nonlinear_iterations == 1)
      {
        if (request.inject_nonfinite_before_rollback)
        {
          double* state = storage.current_state.ReadWrite();
          AppForall(storage, 1, [=] MFEM_HOST_DEVICE(int)
          { state[0] = NAN; });
        }
        return fail(FullDeviceStatus::injected_failure);
      }
      break;
    }
    if (!accepted) { return fail(FullDeviceStatus::line_search_exhausted); }
  }
  if (evidence.nonlinear_iterations < 2 || residual > target)
  {
    return fail(FullDeviceStatus::iteration_limit);
  }
  if (!IntegrateErrors(graph, storage, storage.qe,
                       evidence.volume_relative_l2_qe,
                       evidence.trace_relative_l2_qe, evidence.telemetry) ||
      !IntegrateErrors(graph, storage, storage.qe4,
                       evidence.volume_relative_l2_qe4,
                       evidence.trace_relative_l2_qe4, evidence.telemetry))
  {
    return fail(FullDeviceStatus::error_integration_failure);
  }
  const auto quadrature_pair = [](const double first, const double second)
  {
    const double difference = std::abs(first - second);
    return std::isfinite(first) && std::isfinite(second) && first >= 1.0e-10 &&
           second >= 1.0e-10 &&
           difference / std::max(second, 1.0e-300) <= 1.0e-8 &&
           second > 100.0 * difference;
  };
  if (!quadrature_pair(evidence.volume_relative_l2_qe,
                       evidence.volume_relative_l2_qe4) ||
      !quadrature_pair(evidence.trace_relative_l2_qe,
                       evidence.trace_relative_l2_qe4))
  {
    return fail(FullDeviceStatus::error_integration_failure);
  }
  if (request.enable_test_oracle_staging)
  {
    CaptureFinalSnapshot(graph, storage, evidence,
                         request.enable_global_linear_oracle_staging,
                         last_accepted_alpha);
  }
  const std::vector<FullDevicePhaseTelemetry> sealed_phase_log =
      evidence.phase_audit_log;
  if (!AuditManagedResidency(graph, storage, evidence))
  {
    return fail(FullDeviceStatus::residency_failure);
  }
  evidence.phase_audit_append_only =
      SamePhaseLog(sealed_phase_log, evidence.phase_audit_log);
  evidence.status = FullDeviceStatus::success;
  evidence.pass = evidence.max_physics_finite_difference_error <= physics_gate &&
      evidence.constant_state_residual <= constant_gate &&
      evidence.local_equation_error <= local_gate &&
      evidence.condensed_equation_error <= local_gate &&
      evidence.schur_action_error <= action_gate &&
      evidence.trace_solution_error <= action_gate &&
      evidence.recovery_error <= action_gate &&
      evidence.state_update_error <= action_gate &&
      evidence.true_relative_residual <= true_residual_gate &&
      evidence.bynodes_vdim4 && evidence.negative_trace_vdof_count == 0 &&
      evidence.reversed_trace_permutation_observed &&
      evidence.shared_face_orientation_action_verified &&
      evidence.exact_ptslp_assembly &&
      evidence.hypre_matrix_device && evidence.hypre_rhs_device &&
      evidence.hypre_solution_device && evidence.hypre_gmres &&
      evidence.hypre_policy_before_solve_device &&
      evidence.hypre_policy_after_solve_device &&
      evidence.hypre_ilu_type_zero_fill_zero &&
      evidence.allocation_observability_reported &&
      evidence.app_forall_fallback_census_complete &&
      !evidence.mfem_internal_backend_observable &&
      !evidence.hypre_internal_backend_observable &&
      !evidence.fallback_census_complete &&
      evidence.phase_audit_append_only &&
      evidence.managed_residency_verified &&
      evidence.cuda_kernel_execution_verified &&
      evidence.telemetry.fallback_events == 0 &&
      evidence.telemetry.dense_host_trace_inverse_events == 0 &&
      evidence.telemetry.solver_substitution_events == 0;
  return evidence.pass;
}

bool ValidateConvergence(FullDeviceEvidence& evidence)
{
  for (int p = 1; p <= 4; ++p)
  {
    auto& convergence = evidence.convergence[static_cast<std::size_t>(p - 1)];
    convergence.polynomial_order = p;
    for (std::size_t mesh_index = 0; mesh_index < 3; ++mesh_index)
    {
      const auto match = std::find_if(
          evidence.cases.begin(), evidence.cases.end(),
          [p, mesh_index](const FullDeviceMeshEvidence& current)
          {
            return current.polynomial_order == p &&
                   current.elements_per_direction ==
                       std::array<int, 3>{{3, 6, 12}}[mesh_index];
          });
      if (match == evidence.cases.end() || !match->pass) { return false; }
      convergence.volume_errors[mesh_index] = match->volume_relative_l2_qe4;
      convergence.trace_errors[mesh_index] = match->trace_relative_l2_qe4;
      if (!std::isfinite(convergence.volume_errors[mesh_index]) ||
          !std::isfinite(convergence.trace_errors[mesh_index]) ||
          convergence.volume_errors[mesh_index] < 1.0e-10 ||
          convergence.trace_errors[mesh_index] < 1.0e-10)
      {
        return false;
      }
    }
    for (int interval = 0; interval < 2; ++interval)
    {
      if (!(convergence.volume_errors[interval] >
            convergence.volume_errors[interval + 1]) ||
          !(convergence.trace_errors[interval] >
            convergence.trace_errors[interval + 1]))
      {
        return false;
      }
      convergence.volume_rates[interval] = std::log2(
          convergence.volume_errors[interval] /
          convergence.volume_errors[interval + 1]);
      convergence.trace_rates[interval] = std::log2(
          convergence.trace_errors[interval] /
          convergence.trace_errors[interval + 1]);
      if (convergence.volume_rates[interval] < p + 0.25 ||
          convergence.trace_rates[interval] < p + 0.45)
      {
        return false;
      }
    }
    convergence.trace_three_level_slope =
        (std::log2(convergence.trace_errors[0]) -
         std::log2(convergence.trace_errors[2])) /
        2.0;
    if (!std::isfinite(convergence.trace_three_level_slope) ||
        convergence.trace_three_level_slope < p + 0.50)
    {
      return false;
    }
    convergence.pass = true;
  }
  return true;
}

}  // namespace

bool RunSmoothEulerDeviceFullExperiment(const FullDeviceRequest& request,
                                        FullDeviceEvidence& evidence)
{
  FullDeviceEvidence next;
  if (!request.enabled)
  {
    next.status = FullDeviceStatus::default_disabled;
    evidence = std::move(next);
    return false;
  }
  if (!mfem::Device::IsConfigured() ||
      !mfem::Device::Allows(mfem::Backend::CUDA_MASK) ||
      mfem::Device::Allows(mfem::Backend::DEBUG_DEVICE))
  {
    next.status = FullDeviceStatus::setup_failure;
    evidence = std::move(next);
    return false;
  }
  if ((request.enable_global_linear_oracle_staging &&
       !request.enable_test_oracle_staging) ||
      (request.inject_nonfinite_before_rollback &&
       !request.inject_failure_after_first_update) ||
      (request.inject_failure_after_first_update &&
       (request.injected_failure_polynomial_order < 1 ||
        request.injected_failure_polynomial_order > 4 ||
        request.injected_failure_elements_per_direction < 3 ||
        (request.run_complete_ladder &&
         request.injected_failure_elements_per_direction != 3 &&
         request.injected_failure_elements_per_direction != 6 &&
         request.injected_failure_elements_per_direction != 12) ||
        (!request.run_complete_ladder &&
         (request.injected_failure_polynomial_order != 1 ||
          request.injected_failure_elements_per_direction != 3)))))
  {
    next.status = FullDeviceStatus::setup_failure;
    evidence = std::move(next);
    return false;
  }
  mfem::Hypre::Init();
  if (!mfem::HypreUsingGPU())
  {
    next.status = FullDeviceStatus::setup_failure;
    evidence = std::move(next);
    return false;
  }
  HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_UNDEFINED;
  HYPRE_ExecutionPolicy execution_policy = HYPRE_EXEC_UNDEFINED;
  HYPRE_ClearAllErrors();
  if (HYPRE_GetMemoryLocation(&memory_location) != 0 ||
      HYPRE_GetExecutionPolicy(&execution_policy) != 0 ||
      memory_location != HYPRE_MEMORY_DEVICE ||
      execution_policy != HYPRE_EXEC_DEVICE || HYPRE_GetError() != 0)
  {
    next.status = FullDeviceStatus::residency_failure;
    evidence = std::move(next);
    return false;
  }
  next.real_mfem_cuda_path = true;
  next.real_hypre_device_path = true;
  const std::array<int, 3> widths{{3, 6, 12}};
  const auto run_case = [&](const int p, const int N)
  {
    FullDeviceMeshEvidence current;
    FullDeviceRequest case_request = request;
    case_request.enable_global_linear_oracle_staging =
        request.enable_global_linear_oracle_staging && N == 3;
    const bool passed = RunMeshCase(case_request, p, N, current);
    if (!passed)
    {
      next.cases.clear();
      next.convergence = {};
      next.status = current.status;
      next.cases.push_back(std::move(current));
      return false;
    }
    next.cases.push_back(std::move(current));
    return true;
  };
  if (!request.run_complete_ladder)
  {
    if (!run_case(1, widths[0]))
    {
      evidence = std::move(next);
      return false;
    }
  }
  else
  {
    for (int p = 1; p <= 4; ++p)
      if (!run_case(p, widths[0]))
      {
        evidence = std::move(next);
        return false;
      }
    for (std::size_t width_index = 1; width_index < widths.size();
         ++width_index)
      for (int p = 1; p <= 4; ++p)
        if (!run_case(p, widths[width_index]))
        {
          evidence = std::move(next);
          return false;
        }
  }
  if (request.run_complete_ladder && !ValidateConvergence(next))
  {
    next.status = FullDeviceStatus::rate_failure;
    evidence = std::move(next);
    return false;
  }
  next.status = FullDeviceStatus::success;
  next.pass = true;
  evidence = std::move(next);
  return true;
}
}  // namespace dg::hdg::euler

namespace dg::hdg::euler
{
namespace
{
bool Prescreen(const DeviceGraph& graph, DeviceStorage& storage,
               FullDeviceMeshEvidence& evidence)
{
  GatherCurrentTrace(graph, storage);
  storage.prescreen_reduction = 0.0;
  const double* state_coefficients = storage.current_state.Read();
  const double* trace_coefficients = storage.local_trace.Read();
  const double* volume_shape = storage.volume_shape.Read();
  const double* face_interior_shape = storage.face_interior_shape.Read();
  const double* face_trace_shape = storage.face_trace_shape.Read();
  double* telemetry = storage.prescreen_reduction.Write();
  const int elements = graph.elements;
  const int scalar_dofs = graph.scalar_dofs;
  const int face_scalar_dofs = graph.face_scalar_dofs;
  const int local_interior = graph.local_interior;
  const int local_trace = graph.local_trace;
  const int volume_points = graph.volume_points;
  const int face_points = graph.face_points;
  AppForall(storage, elements, [=] MFEM_HOST_DEVICE(int element)
  {
    const std::size_t base = static_cast<std::size_t>(element) * 4;
    telemetry[base] = 1.0;
    telemetry[base + 1] = 1.0e300;
    telemetry[base + 2] = 1.0e300;
    telemetry[base + 3] = 0.0;
    const std::size_t element_ni =
        static_cast<std::size_t>(element) * local_interior;
    const std::size_t element_nt =
        static_cast<std::size_t>(element) * local_trace;
    for (int q = 0; q < volume_points; ++q)
    {
      const std::size_t quadrature =
          static_cast<std::size_t>(element) * volume_points + q;
      double state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
      for (int scalar = 0; scalar < scalar_dofs; ++scalar)
      {
        const double shape = volume_shape[quadrature * scalar_dofs + scalar];
        for (int component = 0; component < conserved_components; ++component)
        {
          state[component] += shape * state_coefficients[
              element_ni + LocalByNodes(scalar, scalar_dofs, component)];
        }
      }
      double rho = 0.0, pressure = 0.0, speed = 0.0;
      if (!AdmissibleState(state, rho, pressure, speed))
      {
        telemetry[base] = -1.0;
        return;
      }
      telemetry[base + 1] = fmin(telemetry[base + 1], rho);
      telemetry[base + 2] = fmin(telemetry[base + 2], pressure);
      telemetry[base + 3] = fmax(telemetry[base + 3], speed);
    }
    for (int local_face = 0; local_face < faces_per_element; ++local_face)
      for (int q = 0; q < face_points; ++q)
      {
        const std::size_t quadrature =
            (static_cast<std::size_t>(element) * faces_per_element +
             local_face) *
                face_points +
            q;
        const int trace_offset =
            local_face * conserved_components * face_scalar_dofs;
        double interior_state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
        double trace_state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
        for (int scalar = 0; scalar < scalar_dofs; ++scalar)
        {
          const double shape =
              face_interior_shape[quadrature * scalar_dofs + scalar];
          for (int component = 0; component < conserved_components; ++component)
          {
            interior_state[component] += shape * state_coefficients[
                element_ni + LocalByNodes(scalar, scalar_dofs, component)];
          }
        }
        for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
        {
          const double shape =
              face_trace_shape[quadrature * face_scalar_dofs + scalar];
          for (int component = 0; component < conserved_components; ++component)
          {
            trace_state[component] += shape * trace_coefficients[
                element_nt + trace_offset +
                LocalByNodes(scalar, face_scalar_dofs, component)];
          }
        }
        for (int which = 0; which < 2; ++which)
        {
          const double* state = which == 0 ? interior_state : trace_state;
          double rho = 0.0, pressure = 0.0, speed = 0.0;
          if (!AdmissibleState(state, rho, pressure, speed))
          {
            telemetry[base] = -2.0;
            return;
          }
          telemetry[base + 1] = fmin(telemetry[base + 1], rho);
          telemetry[base + 2] = fmin(telemetry[base + 2], pressure);
          telemetry[base + 3] = fmax(telemetry[base + 3], speed);
        }
      }
  });
  const double* host = storage.prescreen_reduction.HostRead();
  evidence.telemetry.device_to_host_bytes +=
      static_cast<std::size_t>(elements) * 4 * sizeof(double);
  ++evidence.telemetry.synchronization_events;
  double minimum_density = std::numeric_limits<double>::infinity();
  double minimum_pressure = std::numeric_limits<double>::infinity();
  double maximum_wave_speed = 0.0;
  for (int element = 0; element < elements; ++element)
  {
    const std::size_t base = static_cast<std::size_t>(element) * 4;
    if (host[base] != 1.0) { return false; }
    minimum_density = std::min(minimum_density, host[base + 1]);
    minimum_pressure = std::min(minimum_pressure, host[base + 2]);
    maximum_wave_speed = std::max(maximum_wave_speed, host[base + 3]);
  }
  evidence.minimum_density =
      std::min(evidence.minimum_density, minimum_density);
  evidence.minimum_pressure =
      std::min(evidence.minimum_pressure, minimum_pressure);
  evidence.maximum_wave_speed =
      std::max(evidence.maximum_wave_speed, maximum_wave_speed);
  return std::isfinite(minimum_density) && std::isfinite(minimum_pressure) &&
         std::isfinite(maximum_wave_speed);
}

void ScatterElementVector(const DeviceGraph& graph, DeviceStorage& storage,
                          const mfem::Vector& input, mfem::Vector& local)
{
  local = 0.0;
  const int count = graph.elements * graph.local_trace;
  const int* signed_map = graph.signed_trace_dofs.Read();
  const double* values = input.Read();
  double* assembled = local.Write();
  AppForall(storage, count, [=] MFEM_HOST_DEVICE(int entry)
  {
    const int encoded = signed_map[entry];
    const double sign = encoded >= 0 ? 1.0 : -1.0;
    const int global = encoded >= 0 ? encoded : -1 - encoded;
    AtomicAdd(assembled[global], sign * values[entry]);
  });
}

double ResidualNorm(const DeviceGraph& graph, DeviceStorage& storage,
                    FullDeviceTransferTelemetry& telemetry)
{
  if (storage.residual_generation == 0 ||
      storage.residual_generation != storage.generation)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  ScatterElementVector(graph, storage, storage.residual_g,
                       storage.local_sparse_rhs);
  graph.trace_space->Dof_TrueDof_Matrix()->MultTranspose(
      storage.local_sparse_rhs, storage.true_action);
  const double interior = storage.residual_r.Norml2();
  const double trace = storage.true_action.Norml2();
  telemetry.device_to_host_bytes += 2 * sizeof(double);
  telemetry.synchronization_events += 2;
  return std::hypot(interior, trace);
}

bool TrialState(const DeviceGraph& graph, DeviceStorage& storage,
                const double alpha)
{
  if (storage.recovery_generation == 0 || !std::isfinite(alpha))
  {
    return false;
  }
  BeginAppPhase(storage, FullDevicePhase::state_update,
                storage.generation + 1);
  const int interior_count = graph.elements * graph.local_interior;
  const int trace_count = graph.local_trace_dofs;
  const double* base_state = storage.base_state.Read();
  const double* delta_state = storage.delta_state.Read();
  double* trial_state = storage.trial_state.Write();
  AppForall(storage, interior_count, [=] MFEM_HOST_DEVICE(int entry)
  {
    trial_state[entry] = base_state[entry] + alpha * delta_state[entry];
  });
  const double* base_trace = storage.base_trace.Read();
  const double* delta_trace = storage.distributed_trace_increment.Read();
  double* trial_trace = storage.trial_trace.Write();
  AppForall(storage, trace_count, [=] MFEM_HOST_DEVICE(int entry)
  {
    trial_trace[entry] = base_trace[entry] + alpha * delta_trace[entry];
  });
  storage.current_state = storage.trial_state;
  storage.current_trace = storage.trial_trace;
  return true;
}

void InvalidateDerivedProducts(DeviceStorage& storage)
{
  storage.residual_generation = 0;
  storage.condensation_generation = 0;
  storage.trace_system_generation = 0;
  storage.trace_solution_generation = 0;
  storage.recovery_generation = 0;
}

bool InvalidateAndProbeConsumers(DeviceGraph& graph, DeviceStorage& storage,
                                 FullDeviceMeshEvidence& evidence)
{
  InvalidateDerivedProducts(storage);
  evidence.invalidated_product_count = 5;
  FullDeviceMeshEvidence probe;
  InitializePhases(probe);
  std::size_t rejected = 0;
  FullDeviceTransferTelemetry telemetry;
  if (std::isnan(ResidualNorm(graph, storage, telemetry))) { ++rejected; }
  if (!Condense(graph, storage, probe)) { ++rejected; }
  if (!AssembleTraceSystem(graph, storage, probe)) { ++rejected; }
  if (!SolveTrace(graph, storage, probe)) { ++rejected; }
  if (!Recover(graph, storage, probe)) { ++rejected; }
  if (!TrialState(graph, storage, 1.0)) { ++rejected; }
  evidence.rejected_consumer_count = rejected;
  evidence.rollback_rejected_post_failure_consumption = rejected == 6;
  return evidence.rollback_rejected_post_failure_consumption;
}

double StateUpdateError(const DeviceGraph& graph, DeviceStorage& storage,
                        const double alpha,
                        FullDeviceMeshEvidence& evidence)
{
  const double* base_state = storage.base_state.Read();
  const double* delta_state = storage.delta_state.Read();
  const double* trial_state = storage.trial_state.Read();
  const double* base_trace = storage.base_trace.Read();
  const double* delta_trace = storage.distributed_trace_increment.Read();
  const double* trial_trace = storage.trial_trace.Read();
  double* audit = storage.state_audit.Write();
  const int state_size = graph.elements * graph.local_interior;
  const int trace_size = graph.local_trace_dofs;
  AppForall(storage, 1, [=] MFEM_HOST_DEVICE(int)
  {
    double maximum = 0.0;
    for (int entry = 0; entry < state_size; ++entry)
      maximum = fmax(maximum, fabs(trial_state[entry] -
          (base_state[entry] + alpha * delta_state[entry])));
    for (int entry = 0; entry < trace_size; ++entry)
      maximum = fmax(maximum, fabs(trial_trace[entry] -
          (base_trace[entry] + alpha * delta_trace[entry])));
    audit[0] = maximum;
  });
  if (!EndPhaseAudit(graph, storage, evidence,
                     FullDevicePhase::state_update,
                     FullDeviceStatus::success, storage.generation + 1))
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double value = storage.state_audit.HostRead()[0];
  evidence.telemetry.device_to_host_bytes += sizeof(double);
  ++evidence.telemetry.synchronization_events;
  return value;
}

bool RestoreEntryAndCheck(const DeviceGraph& graph, DeviceStorage& storage,
                          FullDeviceMeshEvidence& evidence,
                          bool& had_uncommitted_change)
{
  double* check = storage.rollback_reduction.Write();
  const double* state = storage.current_state.Read();
  const double* entry_state = storage.entry_state.Read();
  const double* trace = storage.current_trace.Read();
  const double* entry_trace = storage.entry_trace.Read();
  const int state_size = graph.elements * graph.local_interior;
  const int trace_size = graph.local_trace_dofs;
  AppForall(storage, 1, [=] MFEM_HOST_DEVICE(int)
  {
    double code = 0.0;
    for (int entry = 0; entry < state_size; ++entry)
    {
      union Bits { double value; std::uint64_t bits; } current{}, original{};
      current.value = state[entry];
      original.value = entry_state[entry];
      if (!isfinite(current.value) || !isfinite(original.value)) { code = 2.0; }
      else if (current.bits != original.bits && code == 0.0) { code = 1.0; }
    }
    for (int entry = 0; entry < trace_size; ++entry)
    {
      union Bits { double value; std::uint64_t bits; } current{}, original{};
      current.value = trace[entry];
      original.value = entry_trace[entry];
      if (!isfinite(current.value) || !isfinite(original.value)) { code = 2.0; }
      else if (current.bits != original.bits && code == 0.0) { code = 1.0; }
    }
    check[0] = code;
  });
  const double changed_code = storage.rollback_reduction.HostRead()[0];
  evidence.telemetry.device_to_host_bytes += sizeof(double);
  ++evidence.telemetry.synchronization_events;
  had_uncommitted_change = changed_code != 0.0;
  evidence.rollback_nonfinite_change_observed = changed_code == 2.0;
  if (!evidence.test_snapshot.initial_interior.empty())
  {
    CopyTestVector(storage.entry_state,
                   evidence.test_snapshot.rollback_entry_interior,
                   evidence.telemetry);
    CopyTestVector(storage.entry_trace,
                   evidence.test_snapshot.rollback_entry_trace,
                   evidence.telemetry);
    CopyTestVector(storage.current_state,
                   evidence.test_snapshot.rollback_changed_interior,
                   evidence.telemetry);
    CopyTestVector(storage.current_trace,
                   evidence.test_snapshot.rollback_changed_trace,
                   evidence.telemetry);
  }

  storage.current_state = storage.entry_state;
  storage.current_trace = storage.entry_trace;
  check = storage.rollback_reduction.Write();
  state = storage.current_state.Read();
  entry_state = storage.entry_state.Read();
  trace = storage.current_trace.Read();
  entry_trace = storage.entry_trace.Read();
  AppForall(storage, 1, [=] MFEM_HOST_DEVICE(int)
  {
    double code = 0.0;
    for (int entry = 0; entry < state_size; ++entry)
    {
      union Bits { double value; std::uint64_t bits; } current{}, original{};
      current.value = state[entry];
      original.value = entry_state[entry];
      if (!isfinite(current.value) || !isfinite(original.value)) { code = 2.0; }
      else if (current.bits != original.bits && code == 0.0) { code = 1.0; }
    }
    for (int entry = 0; entry < trace_size; ++entry)
    {
      union Bits { double value; std::uint64_t bits; } current{}, original{};
      current.value = trace[entry];
      original.value = entry_trace[entry];
      if (!isfinite(current.value) || !isfinite(original.value)) { code = 2.0; }
      else if (current.bits != original.bits && code == 0.0) { code = 1.0; }
    }
    check[0] = code;
  });
  const double restored_code = storage.rollback_reduction.HostRead()[0];
  evidence.telemetry.device_to_host_bytes += sizeof(double);
  ++evidence.telemetry.synchronization_events;
  if (!evidence.test_snapshot.initial_interior.empty())
  {
    CopyTestVector(storage.current_state,
                   evidence.test_snapshot.rollback_restored_interior,
                   evidence.telemetry);
    CopyTestVector(storage.current_trace,
                   evidence.test_snapshot.rollback_restored_trace,
                   evidence.telemetry);
  }
  evidence.rollback_restored = restored_code == 0.0;
  return evidence.rollback_restored;
}

bool IntegrateErrors(const DeviceGraph& graph, DeviceStorage& storage,
                     const ErrorQuadratureStorage& quadrature,
                     double& volume_relative_l2, double& trace_relative_l2,
                     FullDeviceTransferTelemetry& telemetry)
{
  storage.volume_error_reduction = 0.0;
  const double* coefficients = storage.current_state.Read();
  const double* shape = quadrature.volume_shape.Read();
  const double* weight = quadrature.volume_weight.Read();
  const double* position = quadrature.volume_position.Read();
  double* reduction = storage.volume_error_reduction.Write();
  const int elements = graph.elements;
  const int scalar_dofs = graph.scalar_dofs;
  const int volume_points = quadrature.volume_points;
  const int local_interior = graph.local_interior;
  AppForall(storage, elements, [=] MFEM_HOST_DEVICE(int element)
  {
    double error = 0.0;
    double norm = 0.0;
    double measure = 0.0;
    for (int q = 0; q < volume_points; ++q)
    {
      const std::size_t quadrature_index =
          static_cast<std::size_t>(element) * volume_points + q;
      double numerical[conserved_components] = {0.0, 0.0, 0.0, 0.0};
      double exact[conserved_components];
      ManufacturedState(position[quadrature_index * dimension],
                        position[quadrature_index * dimension + 1], exact);
      for (int scalar = 0; scalar < scalar_dofs; ++scalar)
      {
        const double value = shape[quadrature_index * scalar_dofs + scalar];
        for (int component = 0; component < conserved_components; ++component)
        {
          numerical[component] += value * coefficients[
              static_cast<std::size_t>(element) * local_interior +
              LocalByNodes(scalar, scalar_dofs, component)];
        }
      }
      for (int component = 0; component < conserved_components; ++component)
      {
        const double difference = numerical[component] - exact[component];
        error += weight[quadrature_index] * difference * difference;
        norm += weight[quadrature_index] * exact[component] * exact[component];
      }
      measure += weight[quadrature_index];
    }
    const std::size_t base = static_cast<std::size_t>(element) * 3;
    reduction[base] = static_cast<double>(error);
    reduction[base + 1] = static_cast<double>(norm);
    reduction[base + 2] = static_cast<double>(measure);
  });
  const double* volume_host = storage.volume_error_reduction.HostRead();
  long double volume_error = 0.0, volume_norm = 0.0, volume_measure = 0.0;
  for (int element = 0; element < elements; ++element)
  {
    const std::size_t base = static_cast<std::size_t>(element) * 3;
    volume_error += volume_host[base];
    volume_norm += volume_host[base + 1];
    volume_measure += volume_host[base + 2];
  }
  telemetry.device_to_host_bytes +=
      static_cast<std::size_t>(elements) * 3 * sizeof(double);
  ++telemetry.synchronization_events;

  storage.face_error_reduction = 0.0;
  const int faces = graph.mesh->GetNumFaces();
  const int face_points = quadrature.face_points;
  const int face_scalar_dofs = graph.face_scalar_dofs;
  const int trace_vdofs = conserved_components * face_scalar_dofs;
  const int* signed_vdofs = quadrature.face_signed_vdofs.Read();
  const double* trace_coefficients = storage.current_trace.Read();
  const double* face_shape = quadrature.face_shape.Read();
  const double* face_weight = quadrature.face_weight.Read();
  const double* face_position = quadrature.face_position.Read();
  reduction = storage.face_error_reduction.Write();
  AppForall(storage, faces, [=] MFEM_HOST_DEVICE(int face)
  {
    double error = 0.0;
    double norm = 0.0;
    double measure = 0.0;
    for (int q = 0; q < face_points; ++q)
    {
      const std::size_t quadrature_index =
          static_cast<std::size_t>(face) * face_points + q;
      double numerical[conserved_components] = {0.0, 0.0, 0.0, 0.0};
      double exact[conserved_components];
      ManufacturedState(face_position[quadrature_index * dimension],
                        face_position[quadrature_index * dimension + 1], exact);
      for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
      {
        const double value =
            face_shape[quadrature_index * face_scalar_dofs + scalar];
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = LocalByNodes(scalar, face_scalar_dofs, component);
          const int encoded = signed_vdofs[face * trace_vdofs + local];
          const double sign = encoded >= 0 ? 1.0 : -1.0;
          const int global = encoded >= 0 ? encoded : -1 - encoded;
          numerical[component] += value * sign * trace_coefficients[global];
        }
      }
      for (int component = 0; component < conserved_components; ++component)
      {
        const double difference = numerical[component] - exact[component];
        error += face_weight[quadrature_index] * difference * difference;
        norm += face_weight[quadrature_index] * exact[component] *
                exact[component];
      }
      measure += face_weight[quadrature_index];
    }
    const std::size_t base = static_cast<std::size_t>(face) * 3;
    reduction[base] = static_cast<double>(error);
    reduction[base + 1] = static_cast<double>(norm);
    reduction[base + 2] = static_cast<double>(measure);
  });
  const double* face_host = storage.face_error_reduction.HostRead();
  long double trace_error = 0.0, trace_norm = 0.0, trace_measure = 0.0;
  for (int face = 0; face < faces; ++face)
  {
    const std::size_t base = static_cast<std::size_t>(face) * 3;
    trace_error += face_host[base];
    trace_norm += face_host[base + 1];
    trace_measure += face_host[base + 2];
  }
  telemetry.device_to_host_bytes +=
      static_cast<std::size_t>(faces) * 3 * sizeof(double);
  ++telemetry.synchronization_events;
  volume_relative_l2 = std::sqrt(static_cast<double>(volume_error / volume_norm));
  trace_relative_l2 = std::sqrt(static_cast<double>(trace_error / trace_norm));
  const double expected_trace_measure = 2.0 * graph.N;
  return std::isfinite(volume_relative_l2) &&
         std::isfinite(trace_relative_l2) && volume_relative_l2 >= 1.0e-10 &&
         trace_relative_l2 >= 1.0e-10 &&
         std::abs(static_cast<double>(volume_measure) - 1.0) <=
             256.0 * experiment_epsilon &&
         std::abs(static_cast<double>(trace_measure) - expected_trace_measure) <=
             256.0 * experiment_epsilon *
                 std::max(1.0, expected_trace_measure);
}

void CopyTestVector(const mfem::Vector& source, std::vector<double>& destination,
                    FullDeviceTransferTelemetry& telemetry)
{
  const double* values = source.HostRead();
  destination.assign(values, values + source.Size());
  const std::size_t bytes = static_cast<std::size_t>(source.Size()) *
                            sizeof(double);
  telemetry.device_to_host_bytes += bytes;
  telemetry.test_only_device_to_host_bytes += bytes;
  ++telemetry.synchronization_events;
}

void CopyTestPrefix(const mfem::Vector& source, const std::size_t size,
                    std::vector<double>& destination,
                    FullDeviceTransferTelemetry& telemetry)
{
  const double* values = source.HostRead();
  destination.assign(values, values + size);
  // HostRead synchronizes the full logical MFEM vector even though the
  // bounded oracle retains only a prefix.
  const std::size_t bytes =
      static_cast<std::size_t>(source.Size()) * sizeof(double);
  telemetry.device_to_host_bytes += bytes;
  telemetry.test_only_device_to_host_bytes += bytes;
  ++telemetry.synchronization_events;
}

}  // namespace
}  // namespace dg::hdg::euler

namespace dg::hdg::euler
{
namespace
{
bool BuildTrueCsrMap(DeviceGraph& graph,
                     FullDeviceTransferTelemetry& telemetry)
{
  if (!graph.true_trace_matrix || graph.true_trace_matrix->Height() !=
                                      graph.true_trace_dofs)
  {
    return false;
  }
  const int rows = graph.true_trace_dofs;
  const int nonzeros = static_cast<int>(
      graph.true_trace_matrix->GetDiagMemoryData().Capacity());
  const HYPRE_Int* offsets = graph.true_trace_matrix->GetDiagMemoryI().Read(
      mfem::MemoryClass::HOST, rows + 1);
  const HYPRE_Int* columns = graph.true_trace_matrix->GetDiagMemoryJ().Read(
      mfem::MemoryClass::HOST, nonzeros);
  const int* signed_map = graph.host_signed_trace_dofs.data();
  std::vector<int> mapping(
      static_cast<std::size_t>(graph.elements) * graph.local_trace *
          graph.local_trace,
      -1);
  for (int element = 0; element < graph.elements; ++element)
    for (int row = 0; row < graph.local_trace; ++row)
    {
      const int row_encoded =
          signed_map[static_cast<std::size_t>(element) * graph.local_trace + row];
      const int global_row = row_encoded >= 0 ? row_encoded : -1 - row_encoded;
      for (int column = 0; column < graph.local_trace; ++column)
      {
        const int column_encoded = signed_map[
            static_cast<std::size_t>(element) * graph.local_trace + column];
        const int global_column =
            column_encoded >= 0 ? column_encoded : -1 - column_encoded;
        int found = -1;
        for (int entry = offsets[global_row];
             entry < offsets[global_row + 1]; ++entry)
        {
          if (columns[entry] == global_column)
          {
            found = entry;
            break;
          }
        }
        if (found < 0) { return false; }
        mapping[(static_cast<std::size_t>(element) * graph.local_trace + row) *
                    graph.local_trace +
                column] = found;
      }
    }
  telemetry.device_to_host_bytes +=
      static_cast<std::size_t>(rows + 1 + nonzeros) * sizeof(HYPRE_Int);
  ++telemetry.synchronization_events;
  return StageArray(graph.element_to_true_csr, mapping, telemetry);
}

void AssembleLocalTraceRhs(const DeviceGraph& graph, DeviceStorage& storage)
{
  const int elements = graph.elements;
  const int nt = graph.local_trace;
  double* local_rhs = storage.local_sparse_rhs.Write();
  const double* rhs = storage.condensed_rhs.Read();
  const int* signed_map = graph.signed_trace_dofs.Read();
  AppForall(storage, graph.local_trace_dofs,
            [=] MFEM_HOST_DEVICE(int entry)
  {
    local_rhs[entry] = 0.0;
  });
  AppForall(storage, elements * nt, [=] MFEM_HOST_DEVICE(int entry)
  {
    const int encoded = signed_map[entry];
    const double sign = encoded >= 0 ? 1.0 : -1.0;
    const int global = encoded >= 0 ? encoded : -1 - encoded;
    AtomicAdd(local_rhs[global], sign * rhs[entry]);
  });
}

void LaunchTrueMatrixAudit(const DeviceGraph& graph, DeviceStorage& storage)
{
  const int rows = graph.true_trace_dofs;
  const int nonzeros = static_cast<int>(
      graph.true_trace_matrix->GetDiagMemoryData().Capacity());
  const HYPRE_Int* offsets = graph.true_trace_matrix->GetDiagMemoryI().Read(
      mfem::GetHypreMemoryClass(), rows + 1);
  const HYPRE_Int* columns = graph.true_trace_matrix->GetDiagMemoryJ().Read(
      mfem::GetHypreMemoryClass(), nonzeros);
  const HYPRE_Real* values =
      graph.true_trace_matrix->GetDiagMemoryData().Read(
          mfem::GetHypreMemoryClass(), nonzeros);
  double* audit = storage.true_matrix_audit.Write();
  AppForall(storage, 1, [=] MFEM_HOST_DEVICE(int)
  {
    double valid = 1.0;
    double minimum_diagonal = 1.0e300;
    double maximum_entry = 0.0;
    for (int row = 0; row < rows; ++row)
    {
      bool found_diagonal = false;
      for (int entry = offsets[row]; entry < offsets[row + 1]; ++entry)
      {
        const double value = values[entry];
        if (!isfinite(value)) { valid = -1.0; }
        maximum_entry = fmax(maximum_entry, fabs(value));
        if (columns[entry] == row)
        {
          found_diagonal = true;
          minimum_diagonal = fmin(minimum_diagonal, fabs(value));
        }
      }
      if (!found_diagonal) { valid = -2.0; }
    }
    if (!(minimum_diagonal > 0.0) || !isfinite(minimum_diagonal) ||
        !isfinite(maximum_entry))
    {
      valid = -3.0;
    }
    audit[0] = valid;
    audit[1] = minimum_diagonal;
    audit[2] = maximum_entry;
  });
}

bool ConsumeTrueMatrixAudit(const DeviceGraph& graph, DeviceStorage& storage,
                            FullDeviceMeshEvidence& evidence)
{
  const double* host = storage.true_matrix_audit.HostRead();
  evidence.telemetry.device_to_host_bytes += 3 * sizeof(double);
  ++evidence.telemetry.synchronization_events;
  evidence.minimum_trace_diagonal_magnitude =
      std::min(evidence.minimum_trace_diagonal_magnitude, host[1]);
  evidence.maximum_trace_matrix_entry =
      std::max(evidence.maximum_trace_matrix_entry, host[2]);
  return host[0] == 1.0 && std::isfinite(host[1]) && host[1] > 0.0 &&
         std::isfinite(host[2]);
}

void ScatterCondensedToTrueMatrix(const DeviceGraph& graph,
                                  DeviceStorage& storage)
{
  const int elements = graph.elements;
  const int nt = graph.local_trace;
  const int nonzeros = static_cast<int>(
      graph.true_trace_matrix->GetDiagMemoryData().Capacity());
  graph.true_trace_matrix->HypreReadWrite();
  double* matrix_data = graph.true_trace_matrix->GetDiagMemoryData().ReadWrite(
      mfem::GetHypreMemoryClass(), nonzeros);
  const double* schur = storage.schur.Read();
  const int* signed_map = graph.signed_trace_dofs.Read();
  const int* csr_map = graph.element_to_true_csr.Read();
  AppForall(storage, nonzeros, [=] MFEM_HOST_DEVICE(int entry)
  {
    matrix_data[entry] = 0.0;
  });
  AppForall(storage, elements * nt * nt,
            [=] MFEM_HOST_DEVICE(int entry)
  {
    const int element = entry / (nt * nt);
    const int remainder = entry - element * nt * nt;
    const int row = remainder / nt;
    const int column = remainder - row * nt;
    const int row_encoded = signed_map[element * nt + row];
    const int column_encoded = signed_map[element * nt + column];
    const double sign = (row_encoded >= 0 ? 1.0 : -1.0) *
                        (column_encoded >= 0 ? 1.0 : -1.0);
    AtomicAdd(matrix_data[csr_map[entry]], sign * schur[entry]);
  });
}

bool InitializeHypreTraceObjects(DeviceGraph& graph, DeviceStorage& storage,
                                 FullDeviceMeshEvidence& evidence)
{
  try
  {
    mfem::ParBilinearForm form(graph.trace_space.get());
    graph.true_trace_matrix.reset(
        form.ParallelAssemble(graph.local_trace_matrix.get()));
    if (!graph.true_trace_matrix ||
        graph.true_trace_matrix->Height() != graph.true_trace_dofs ||
        graph.true_trace_matrix->Width() != graph.true_trace_dofs)
    {
      return false;
    }
    graph.true_trace_matrix->HypreRead();
    if (!BuildTrueCsrMap(graph, evidence.telemetry)) { return false; }
    // ParallelAssemble establishes the exact P^T S_L P graph.  Re-scatter the
    // current numerical Schur values through that graph so the first solve,
    // like every later solve, consumes device-authored values directly.
    ScatterCondensedToTrueMatrix(graph, storage);
    graph.hypre_rhs = std::make_unique<mfem::HypreParVector>(
        MPI_COMM_WORLD, graph.trace_space->GlobalTrueVSize(), storage.true_rhs,
        0, graph.trace_space->GetTrueDofOffsets());
    graph.hypre_solution =
        std::make_unique<mfem::HypreParVector>(*graph.true_trace_matrix);
    graph.hypre_rhs->HypreRead();
    *graph.hypre_solution = 0.0;
    graph.hypre_solution->HypreReadWrite();
    HYPRE_ClearAllErrors();
    graph.ilu = std::make_unique<mfem::HypreILU>();
    ConfigureFrozenILU(*graph.ilu);
    graph.gmres =
        std::make_unique<mfem::HypreGMRES>(*graph.true_trace_matrix);
    graph.gmres->SetTol(1.0e-13);
    graph.gmres->SetAbsTol(0.0);
    graph.gmres->SetMaxIter(std::max(200, graph.true_trace_dofs));
    graph.gmres->SetKDim(50);
    graph.gmres->SetLogging(1);
    graph.gmres->SetPrintLevel(0);
    graph.gmres->SetZeroInitialIterate();
    graph.gmres->SetPreconditioner(*graph.ilu);
  }
  catch (const std::exception&)
  {
    return false;
  }
  evidence.telemetry.allocation_events += 5;
  evidence.hypre_gmres = true;
  evidence.hypre_ilu_type_zero_fill_zero = VerifyFrozenILU(graph, evidence);
  const std::size_t nonzeros =
      graph.true_trace_matrix->GetDiagMemoryData().Capacity();
  const std::size_t known_hypre_bytes =
      (static_cast<std::size_t>(graph.true_trace_dofs) + 1 + nonzeros) *
          sizeof(HYPRE_Int) +
      nonzeros * sizeof(HYPRE_Real) +
      static_cast<std::size_t>(graph.true_trace_dofs) * sizeof(HYPRE_Real);
  evidence.telemetry.hypre_persistent_bytes += known_hypre_bytes;
  evidence.telemetry.persistent_bytes += known_hypre_bytes;
  graph.true_operator_ready = true;
  return evidence.hypre_ilu_type_zero_fill_zero && HYPRE_GetError() == 0;
}

bool AssembleTraceSystem(DeviceGraph& graph, DeviceStorage& storage,
                         FullDeviceMeshEvidence& evidence)
{
  if (storage.condensation_generation == 0 ||
      storage.condensation_generation != storage.generation)
  {
    return false;
  }
  BeginAppPhase(storage, FullDevicePhase::trace_assembly,
                storage.generation);
  AssembleLocalTraceRhs(graph, storage);
  graph.trace_space->Dof_TrueDof_Matrix()->MultTranspose(
      storage.local_sparse_rhs, storage.true_rhs);
  if (!graph.true_operator_ready)
  {
    if (!InitializeHypreTraceObjects(graph, storage, evidence))
    {
      evidence.status = FullDeviceStatus::trace_assembly_failure;
      return false;
    }
  }
  else
  {
    ScatterCondensedToTrueMatrix(graph, storage);
    storage.hypre_setup_refresh_required = true;
  }
  LaunchTrueMatrixAudit(graph, storage);
  if (!EndPhaseAudit(graph, storage, evidence,
                     FullDevicePhase::trace_assembly,
                     FullDeviceStatus::success, storage.generation))
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  if (!ConsumeTrueMatrixAudit(graph, storage, evidence))
  {
    evidence.status = FullDeviceStatus::trace_assembly_failure;
    return false;
  }
  evidence.exact_ptslp_assembly = true;
  storage.trace_system_generation = storage.condensation_generation;
  return true;
}

bool SolveTrace(DeviceGraph& graph, DeviceStorage& storage,
                FullDeviceMeshEvidence& evidence)
{
  if (storage.trace_system_generation == 0 ||
      storage.trace_system_generation != storage.generation)
  {
    return false;
  }
  if (!graph.true_operator_ready || !graph.true_trace_matrix ||
      !graph.hypre_rhs || !graph.hypre_solution || !graph.gmres ||
      !graph.ilu)
  {
    return false;
  }
  BeginAppPhase(storage, FullDevicePhase::trace_solve,
                storage.generation);
  HYPRE_ClearAllErrors();
  graph.true_trace_matrix->HypreRead();
  graph.hypre_rhs->HypreRead();
  *graph.hypre_solution = 0.0;
  graph.hypre_solution->HypreReadWrite();
  hypre_ParCSRMatrix* raw_matrix = *graph.true_trace_matrix;
  hypre_ParVector* raw_rhs = *graph.hypre_rhs;
  hypre_ParVector* raw_solution = *graph.hypre_solution;
  const HYPRE_MemoryLocation matrix_location_before =
      hypre_ParCSRMatrixMemoryLocation(raw_matrix);
  const HYPRE_MemoryLocation rhs_location_before =
      hypre_ParVectorMemoryLocation(raw_rhs);
  const HYPRE_MemoryLocation solution_location_before =
      hypre_ParVectorMemoryLocation(raw_solution);
  evidence.hypre_policy_before_solve_device =
      matrix_location_before == HYPRE_MEMORY_DEVICE &&
      rhs_location_before == HYPRE_MEMORY_DEVICE &&
      solution_location_before == HYPRE_MEMORY_DEVICE &&
      hypre_GetExecPolicy1(matrix_location_before) == HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy1(rhs_location_before) == HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy1(solution_location_before) == HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy2(matrix_location_before, rhs_location_before) ==
          HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy2(matrix_location_before, solution_location_before) ==
          HYPRE_EXEC_DEVICE;
  if (!evidence.hypre_policy_before_solve_device)
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  DeviceStorage::AllocatorSnapshot allocator_before;
  if (!CaptureAllocatorSnapshot(allocator_before, evidence.telemetry))
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  if (storage.hypre_setup_refresh_required)
  {
    HYPRE_ClearAllErrors();
    try
    {
      graph.gmres->SetOperator(*graph.true_trace_matrix);
      // HypreILU::SetOperator resets its parameters to MFEM defaults.  Restore
      // the sealed ILU(0) configuration after every operator refresh and
      // before GMRES is allowed to run HYPRE setup.
      ConfigureFrozenILU(*graph.ilu);
    }
    catch (const std::exception&)
    {
      evidence.status = FullDeviceStatus::trace_solve_failure;
      return false;
    }
    if (HYPRE_GetError() != 0 || !VerifyFrozenILU(graph, evidence))
    {
      evidence.status = FullDeviceStatus::trace_solve_failure;
      return false;
    }
    ++evidence.telemetry.hypre_setup_refresh_events;
    storage.hypre_setup_refresh_required = false;
  }
  // HYPRE setup (and every SetOperator refresh above) performs logical
  // allocator churn.  It is reported explicitly and is not mislabeled as a
  // zero-allocation interval; cumulative calls are audited from UMPIRE_EVENTS.
  evidence.telemetry.hypre_setup_refresh_allocation_churn_reported = true;
  graph.gmres->Mult(*graph.hypre_rhs, *graph.hypre_solution);
  if (HYPRE_GetError() != 0)
  {
    evidence.status = FullDeviceStatus::trace_solve_failure;
    return false;
  }
  HYPRE_Int converged = 0;
  if (HYPRE_GMRESGetConverged(*graph.gmres, &converged) != 0 ||
      converged != 1)
  {
    evidence.status = FullDeviceStatus::trace_solve_failure;
    return false;
  }
  int iterations = 0;
  double reported = 0.0;
  graph.gmres->GetNumIterations(iterations);
  graph.gmres->GetFinalResidualNorm(reported);
  if (!VerifyFrozenILU(graph, evidence))
  {
    evidence.status = FullDeviceStatus::trace_solve_failure;
    return false;
  }
  const HYPRE_MemoryLocation matrix_location_after =
      hypre_ParCSRMatrixMemoryLocation(raw_matrix);
  const HYPRE_MemoryLocation rhs_location_after =
      hypre_ParVectorMemoryLocation(raw_rhs);
  const HYPRE_MemoryLocation solution_location_after =
      hypre_ParVectorMemoryLocation(raw_solution);
  evidence.hypre_policy_after_solve_device =
      matrix_location_after == HYPRE_MEMORY_DEVICE &&
      rhs_location_after == HYPRE_MEMORY_DEVICE &&
      solution_location_after == HYPRE_MEMORY_DEVICE &&
      hypre_GetExecPolicy1(matrix_location_after) == HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy1(rhs_location_after) == HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy1(solution_location_after) == HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy2(matrix_location_after, rhs_location_after) ==
          HYPRE_EXEC_DEVICE &&
      hypre_GetExecPolicy2(matrix_location_after, solution_location_after) ==
          HYPRE_EXEC_DEVICE;
  if (!evidence.hypre_policy_after_solve_device)
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  DeviceStorage::AllocatorSnapshot allocator_after;
  if (!CaptureAllocatorSnapshot(allocator_after, evidence.telemetry))
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  if (!storage.hypre_solve_warmed)
  {
    ++evidence.telemetry.hypre_shape_warmup_setup_events;
    const std::size_t retained_bytes =
        (allocator_after.mfem_current_bytes >
             allocator_before.mfem_current_bytes ?
             allocator_after.mfem_current_bytes -
                 allocator_before.mfem_current_bytes :
             0) +
        (allocator_after.hypre_current_bytes >
             allocator_before.hypre_current_bytes ?
             allocator_after.hypre_current_bytes -
                 allocator_before.hypre_current_bytes :
             0);
    evidence.telemetry.hypre_persistent_bytes += retained_bytes;
    evidence.telemetry.persistent_bytes += retained_bytes;
    if (retained_bytes > 0) { ++evidence.telemetry.allocation_events; }
    storage.allocator_baseline = allocator_after;
    storage.hypre_solve_warmed = true;
  }
  else
  {
    const auto growth = [](const std::size_t current,
                           const std::size_t baseline)
    { return current > baseline ? current - baseline : 0; };
    const std::size_t growth_bytes =
        growth(allocator_after.mfem_high_watermark_bytes,
               storage.allocator_baseline.mfem_high_watermark_bytes) +
        growth(allocator_after.hypre_high_watermark_bytes,
               storage.allocator_baseline.hypre_high_watermark_bytes) +
        growth(allocator_after.mfem_actual_bytes,
               storage.allocator_baseline.mfem_actual_bytes) +
        growth(allocator_after.hypre_actual_bytes,
               storage.allocator_baseline.hypre_actual_bytes) +
        growth(allocator_after.mfem_current_bytes,
               storage.allocator_baseline.mfem_current_bytes) +
        growth(allocator_after.hypre_current_bytes,
               storage.allocator_baseline.hypre_current_bytes);
    const bool allocation_count_growth =
        allocator_after.mfem_active_allocations >
            storage.allocator_baseline.mfem_active_allocations ||
        allocator_after.hypre_active_allocations >
            storage.allocator_baseline.hypre_active_allocations;
    if (growth_bytes > 0 || allocation_count_growth)
    {
      ++evidence.telemetry.hypre_refresh_endpoint_growth_events;
      evidence.telemetry.hypre_refresh_endpoint_growth_bytes += growth_bytes;
    }
  }
  graph.true_trace_matrix->Mult(*graph.hypre_solution, storage.true_action);
  storage.true_action -= storage.true_rhs;
  if (!EndPhaseAudit(graph, storage, evidence, FullDevicePhase::trace_solve,
                     FullDeviceStatus::success, storage.generation))
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  const double rhs_norm = storage.true_rhs.Norml2();
  const double true_residual = storage.true_action.Norml2() /
      std::max(rhs_norm, 1.0e-300);
  evidence.telemetry.device_to_host_bytes +=
      sizeof(HYPRE_Int) + sizeof(int) + 3 * sizeof(double);
  evidence.telemetry.synchronization_events += 5;
  evidence.linear_iterations += iterations;
  evidence.true_relative_residual = true_residual;
  if (!std::isfinite(reported) || !std::isfinite(true_residual) ||
      true_residual > true_residual_gate)
  {
    evidence.status = FullDeviceStatus::linear_residual_failure;
    return false;
  }
  evidence.hypre_matrix_device =
      hypre_ParCSRMatrixMemoryLocation(raw_matrix) == HYPRE_MEMORY_DEVICE;
  evidence.hypre_rhs_device =
      hypre_ParVectorMemoryLocation(raw_rhs) == HYPRE_MEMORY_DEVICE;
  evidence.hypre_solution_device =
      hypre_ParVectorMemoryLocation(raw_solution) == HYPRE_MEMORY_DEVICE;
  if (!evidence.hypre_matrix_device || !evidence.hypre_rhs_device ||
      !evidence.hypre_solution_device)
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  evidence.schur_action_error = true_residual;
  storage.trace_solution_generation = storage.trace_system_generation;
  storage.recovery_generation = 0;
  ++evidence.telemetry.solver_configuration_checks;
  if (!evidence.hypre_gmres ||
      !evidence.hypre_ilu_type_zero_fill_zero || !graph.gmres || !graph.ilu)
  {
    ++evidence.telemetry.solver_substitution_events;
    evidence.status = FullDeviceStatus::trace_solve_failure;
    return false;
  }
  return true;
}

bool Recover(DeviceGraph& graph, DeviceStorage& storage,
             FullDeviceMeshEvidence& evidence)
{
  if (storage.trace_solution_generation == 0 ||
      storage.trace_solution_generation != storage.generation)
  {
    return false;
  }
  BeginAppPhase(storage, FullDevicePhase::recovery, storage.generation);
  graph.trace_space->Dof_TrueDof_Matrix()->Mult(
      *graph.hypre_solution, storage.distributed_trace_increment);
  storage.true_action =
      static_cast<const mfem::Vector&>(*graph.hypre_solution);
  storage.true_action -= storage.distributed_trace_increment;
  const double trace_difference = storage.true_action.Norml2();
  const double trace_norm = graph.hypre_solution->Norml2();
  evidence.telemetry.device_to_host_bytes += 2 * sizeof(double);
  evidence.telemetry.synchronization_events += 2;
  evidence.trace_solution_error =
      trace_difference / std::max(trace_norm, 1.0e-300);
  if (!std::isfinite(evidence.trace_solution_error) ||
      evidence.trace_solution_error > action_gate)
  {
    evidence.status = FullDeviceStatus::recovery_failure;
    return false;
  }
  const int elements = graph.elements;
  const int ni = graph.local_interior;
  const int nt = graph.local_trace;
  const int* signed_map = graph.signed_trace_dofs.Read();
  const double* distributed_trace = storage.distributed_trace_increment.Read();
  double* local_trace_increment = storage.local_trace_increment.Write();
  AppForall(storage, elements * nt, [=] MFEM_HOST_DEVICE(int entry)
  {
    const int encoded = signed_map[entry];
    const double sign = encoded >= 0 ? 1.0 : -1.0;
    const int global = encoded >= 0 ? encoded : -1 - encoded;
    local_trace_increment[entry] = sign * distributed_trace[global];
  });

  const double* lu = storage.lu.Read();
  const double* pivots = storage.pivots.Read();
  const double* a = storage.a.Read();
  const double* b = storage.b.Read();
  const double* f = storage.f.Read();
  const double* lambda = storage.local_trace_increment.Read();
  double* recovered = storage.recovered.Write();
  double* audit = storage.recovery_audit.Write();
  AppForall(storage, elements, [=] MFEM_HOST_DEVICE(int element)
  {
    double* local = recovered + static_cast<std::size_t>(element) * ni;
    for (int row = 0; row < ni; ++row)
    {
      double value = f[static_cast<std::size_t>(element) * ni + row];
      for (int column = 0; column < nt; ++column)
      {
        value -= b[(static_cast<std::size_t>(element) * ni + row) * nt +
                   column] *
                 lambda[static_cast<std::size_t>(element) * nt + column];
      }
      local[row] = value;
    }
    device::detail::SolvePivoted(
        lu + static_cast<std::size_t>(element) * ni * ni,
        pivots + static_cast<std::size_t>(element) * ni, ni, local, 1);
    double maximum = 0.0;
    for (int row = 0; row < ni; ++row)
    {
      double residual = -f[static_cast<std::size_t>(element) * ni + row];
      for (int column = 0; column < ni; ++column)
        residual += a[(static_cast<std::size_t>(element) * ni + row) * ni +
                      column] * local[column];
      for (int column = 0; column < nt; ++column)
        residual += b[(static_cast<std::size_t>(element) * ni + row) * nt +
                      column] *
                    lambda[static_cast<std::size_t>(element) * nt + column];
      maximum = fmax(maximum, fabs(residual));
    }
    audit[element] = maximum;
  });
  storage.delta_state = storage.recovered;
  if (!EndPhaseAudit(graph, storage, evidence, FullDevicePhase::recovery,
                     FullDeviceStatus::success, storage.generation))
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  const double* audit_host = storage.recovery_audit.HostRead();
  evidence.telemetry.device_to_host_bytes +=
      static_cast<std::size_t>(elements) * sizeof(double);
  ++evidence.telemetry.synchronization_events;
  evidence.recovery_error = 0.0;
  for (int element = 0; element < elements; ++element)
    evidence.recovery_error =
        std::max(evidence.recovery_error, audit_host[element]);
  if (!std::isfinite(evidence.recovery_error) ||
      evidence.recovery_error > action_gate)
  {
    evidence.status = FullDeviceStatus::recovery_failure;
    return false;
  }
  storage.recovery_generation = storage.trace_solution_generation;
  return true;
}

}  // namespace
}  // namespace dg::hdg::euler

namespace dg::hdg::euler
{
namespace
{
void GatherCurrentTrace(const DeviceGraph& graph, DeviceStorage& storage)
{
  const int count = graph.elements * graph.local_trace;
  const int* signed_map = graph.signed_trace_dofs.Read();
  const double* global = storage.current_trace.Read();
  double* local = storage.local_trace.Write();
  AppForall(storage, count, [=] MFEM_HOST_DEVICE(int entry)
  {
    const int encoded = signed_map[entry];
    const double sign = encoded >= 0 ? 1.0 : -1.0;
    const int dof = encoded >= 0 ? encoded : -1 - encoded;
    local[entry] = sign * global[dof];
  });
}

bool CheckKernelValidity(DeviceStorage& storage, const int elements,
                         FullDeviceTransferTelemetry& telemetry)
{
  const double* host = storage.kernel_status.HostRead();
  telemetry.device_to_host_bytes +=
      static_cast<std::size_t>(elements) * sizeof(double);
  ++telemetry.synchronization_events;
  for (int element = 0; element < elements; ++element)
  {
    if (host[element] != 1.0) { return false; }
  }
  return true;
}

bool AssembleLinearization(const DeviceGraph& graph, DeviceStorage& storage,
                           FullDeviceMeshEvidence& evidence)
{
  storage.residual_generation = 0;
  storage.condensation_generation = 0;
  storage.trace_system_generation = 0;
  BeginAppPhase(storage, FullDevicePhase::local_assembly,
                storage.generation + 1);
  GatherCurrentTrace(graph, storage);
  storage.residual_r = 0.0;
  storage.residual_g = 0.0;
  storage.a = 0.0;
  storage.b = 0.0;
  storage.c = 0.0;
  storage.d = 0.0;
  storage.f = 0.0;
  storage.g = 0.0;
  storage.kernel_status = 0.0;

  const double* old_coefficients = storage.old_state.Read();
  const double* current_coefficients = storage.current_state.Read();
  const double* trace_coefficients = storage.local_trace.Read();
  const double* volume_shape = storage.volume_shape.Read();
  const double* volume_derivative = storage.volume_derivative.Read();
  const double* volume_weight = storage.volume_weight.Read();
  const double* volume_position = storage.volume_position.Read();
  const double* face_interior_shape = storage.face_interior_shape.Read();
  const double* face_trace_shape = storage.face_trace_shape.Read();
  const double* face_weight = storage.face_weight.Read();
  const double* face_normal = storage.face_normal.Read();
  double* residual_r = storage.residual_r.Write();
  double* residual_g = storage.residual_g.Write();
  double* a = storage.a.Write();
  double* b = storage.b.Write();
  double* c = storage.c.Write();
  double* d = storage.d.Write();
  double* f = storage.f.Write();
  double* g = storage.g.Write();
  double* telemetry = storage.kernel_status.Write();
  const int elements = graph.elements;
  const int scalar_dofs = graph.scalar_dofs;
  const int face_scalar_dofs = graph.face_scalar_dofs;
  const int local_interior = graph.local_interior;
  const int local_trace = graph.local_trace;
  const int volume_points = graph.volume_points;
  const int face_points = graph.face_points;
  AppForall(storage, elements, [=] MFEM_HOST_DEVICE(int element)
  {
    telemetry[element] = 1.0;
    const std::size_t element_ni =
        static_cast<std::size_t>(element) * local_interior;
    const std::size_t element_nt =
        static_cast<std::size_t>(element) * local_trace;
    const std::size_t a_offset = element_ni * local_interior;
    const std::size_t b_offset = element_ni * local_trace;
    const std::size_t c_offset = element_nt * local_interior;
    const std::size_t d_offset = element_nt * local_trace;
    for (int q = 0; q < volume_points; ++q)
    {
      const std::size_t quadrature =
          static_cast<std::size_t>(element) * volume_points + q;
      const double weight = volume_weight[quadrature];
      if (!(weight > 0.0) || !isfinite(weight))
      {
        telemetry[element] = -1.0;
        return;
      }
      double state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
      double old_state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
      for (int scalar = 0; scalar < scalar_dofs; ++scalar)
      {
        const double shape = volume_shape[quadrature * scalar_dofs + scalar];
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = LocalByNodes(scalar, scalar_dofs, component);
          state[component] += shape * current_coefficients[element_ni + local];
          old_state[component] += shape * old_coefficients[element_ni + local];
        }
      }
      double source[conserved_components] = {0.0, 0.0, 0.0, 0.0};
      double flux_x[conserved_components];
      double flux_y[conserved_components];
      double jacobian_x[conserved_components * conserved_components];
      double jacobian_y[conserved_components * conserved_components];
      if (!FiniteFour(state) || !FiniteFour(old_state) ||
          !ManufacturedSourceDevice(
              volume_position[quadrature * dimension],
              volume_position[quadrature * dimension + 1], source) ||
          !FluxAndJacobian(state, 1.0, 0.0, flux_x, jacobian_x) ||
          !FluxAndJacobian(state, 0.0, 1.0, flux_y, jacobian_y))
      {
        telemetry[element] = -2.0;
        return;
      }
      for (int row = 0; row < scalar_dofs; ++row)
      {
        const double row_shape = volume_shape[quadrature * scalar_dofs + row];
        const double derivative_x =
            volume_derivative[(quadrature * scalar_dofs + row) * dimension];
        const double derivative_y =
            volume_derivative[(quadrature * scalar_dofs + row) * dimension + 1];
        for (int row_component = 0;
             row_component < conserved_components; ++row_component)
        {
          const int local_row =
              LocalByNodes(row, scalar_dofs, row_component);
          residual_r[element_ni + local_row] += weight *
              (row_shape * (state[row_component] - old_state[row_component]) /
                   backward_euler_dt -
               derivative_x * flux_x[row_component] -
               derivative_y * flux_y[row_component] -
               row_shape * source[row_component]);
          for (int column = 0; column < scalar_dofs; ++column)
          {
            const double column_shape =
                volume_shape[quadrature * scalar_dofs + column];
            for (int column_component = 0;
                 column_component < conserved_components; ++column_component)
            {
              const int local_column =
                  LocalByNodes(column, scalar_dofs, column_component);
              const double mass = row_component == column_component ?
                  row_shape * column_shape / backward_euler_dt : 0.0;
              const double volume =
                  -(derivative_x *
                        jacobian_x[row_component * conserved_components +
                                   column_component] +
                    derivative_y *
                        jacobian_y[row_component * conserved_components +
                                   column_component]) *
                  column_shape;
              a[a_offset + static_cast<std::size_t>(local_row) *
                                 local_interior +
                local_column] += weight * (mass + volume);
            }
          }
        }
      }
    }

    for (int local_face = 0; local_face < faces_per_element; ++local_face)
    {
      const int trace_offset =
          local_face * conserved_components * face_scalar_dofs;
      for (int q = 0; q < face_points; ++q)
      {
        const std::size_t quadrature =
            (static_cast<std::size_t>(element) * faces_per_element +
             local_face) *
                face_points +
            q;
        const double weight = face_weight[quadrature];
        double interior_state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
        double trace_state[conserved_components] = {0.0, 0.0, 0.0, 0.0};
        for (int scalar = 0; scalar < scalar_dofs; ++scalar)
        {
          const double shape =
              face_interior_shape[quadrature * scalar_dofs + scalar];
          for (int component = 0; component < conserved_components; ++component)
          {
            interior_state[component] += shape *
                current_coefficients[element_ni +
                    LocalByNodes(scalar, scalar_dofs, component)];
          }
        }
        for (int scalar = 0; scalar < face_scalar_dofs; ++scalar)
        {
          const double shape =
              face_trace_shape[quadrature * face_scalar_dofs + scalar];
          for (int component = 0; component < conserved_components; ++component)
          {
            trace_state[component] += shape *
                trace_coefficients[element_nt + trace_offset +
                    LocalByNodes(scalar, face_scalar_dofs, component)];
          }
        }
        double trace_flux[conserved_components];
        double trace_jacobian[conserved_components * conserved_components];
        if (!(weight > 0.0) || !isfinite(weight) ||
            !FiniteFour(interior_state) || !FiniteFour(trace_state) ||
            !FluxAndJacobian(trace_state,
                             face_normal[quadrature * dimension],
                             face_normal[quadrature * dimension + 1],
                             trace_flux, trace_jacobian))
        {
          telemetry[element] = -3.0;
          return;
        }
        for (int row = 0; row < scalar_dofs; ++row)
        {
          const double row_shape =
              face_interior_shape[quadrature * scalar_dofs + row];
          for (int component = 0; component < conserved_components; ++component)
          {
            const int local_row = LocalByNodes(row, scalar_dofs, component);
            residual_r[element_ni + local_row] += weight * row_shape *
                (trace_flux[component] + fixed_scalar_tau *
                    (interior_state[component] - trace_state[component]));
          }
          for (int column = 0; column < scalar_dofs; ++column)
          {
            const double column_shape =
                face_interior_shape[quadrature * scalar_dofs + column];
            for (int component = 0; component < conserved_components; ++component)
            {
              const int local_row = LocalByNodes(row, scalar_dofs, component);
              const int local_column =
                  LocalByNodes(column, scalar_dofs, component);
              a[a_offset + static_cast<std::size_t>(local_row) *
                                 local_interior +
                local_column] += weight * row_shape * fixed_scalar_tau *
                                 column_shape;
            }
          }
          for (int column = 0; column < face_scalar_dofs; ++column)
          {
            const double column_shape =
                face_trace_shape[quadrature * face_scalar_dofs + column];
            for (int row_component = 0;
                 row_component < conserved_components; ++row_component)
              for (int column_component = 0;
                   column_component < conserved_components; ++column_component)
              {
                const int local_row =
                    LocalByNodes(row, scalar_dofs, row_component);
                const int local_column = trace_offset +
                    LocalByNodes(column, face_scalar_dofs, column_component);
                b[b_offset + static_cast<std::size_t>(local_row) * local_trace +
                  local_column] += weight * row_shape * column_shape *
                    (trace_jacobian[row_component * conserved_components +
                                    column_component] -
                     (row_component == column_component ? fixed_scalar_tau :
                                                          0.0));
              }
          }
        }
        for (int row = 0; row < face_scalar_dofs; ++row)
        {
          const double row_shape =
              face_trace_shape[quadrature * face_scalar_dofs + row];
          for (int component = 0; component < conserved_components; ++component)
          {
            const int local_row = trace_offset +
                LocalByNodes(row, face_scalar_dofs, component);
            residual_g[element_nt + local_row] += weight * row_shape *
                (trace_flux[component] + fixed_scalar_tau *
                    (interior_state[component] - trace_state[component]));
          }
          for (int column = 0; column < scalar_dofs; ++column)
          {
            const double column_shape =
                face_interior_shape[quadrature * scalar_dofs + column];
            for (int component = 0; component < conserved_components; ++component)
            {
              const int local_row = trace_offset +
                  LocalByNodes(row, face_scalar_dofs, component);
              const int local_column =
                  LocalByNodes(column, scalar_dofs, component);
              c[c_offset + static_cast<std::size_t>(local_row) *
                                 local_interior +
                local_column] += weight * row_shape * fixed_scalar_tau *
                                 column_shape;
            }
          }
          for (int column = 0; column < face_scalar_dofs; ++column)
          {
            const double column_shape =
                face_trace_shape[quadrature * face_scalar_dofs + column];
            for (int row_component = 0;
                 row_component < conserved_components; ++row_component)
              for (int column_component = 0;
                   column_component < conserved_components; ++column_component)
              {
                const int local_row = trace_offset +
                    LocalByNodes(row, face_scalar_dofs, row_component);
                const int local_column = trace_offset +
                    LocalByNodes(column, face_scalar_dofs, column_component);
                d[d_offset + static_cast<std::size_t>(local_row) * local_trace +
                  local_column] += weight * row_shape * column_shape *
                    (trace_jacobian[row_component * conserved_components +
                                    column_component] -
                     (row_component == column_component ? fixed_scalar_tau :
                                                          0.0));
              }
          }
        }
      }
    }
    for (int row = 0; row < local_interior; ++row)
    {
      f[element_ni + row] = -residual_r[element_ni + row];
    }
    for (int row = 0; row < local_trace; ++row)
    {
      g[element_nt + row] = -residual_g[element_nt + row];
    }
  });
  const std::uint64_t assembled_generation = storage.generation + 1;
  if (!EndPhaseAudit(graph, storage, evidence,
                     FullDevicePhase::local_assembly,
                     FullDeviceStatus::success, assembled_generation))
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  if (!CheckKernelValidity(storage, graph.elements, evidence.telemetry))
  {
    evidence.status = FullDeviceStatus::assembly_failure;
    return false;
  }
  ++storage.generation;
  storage.residual_generation = storage.generation;
  storage.condensation_generation = 0;
  storage.trace_system_generation = 0;
  return true;
}

bool Condense(const DeviceGraph& graph, DeviceStorage& storage,
              FullDeviceMeshEvidence& evidence)
{
  if (storage.residual_generation == 0 ||
      storage.residual_generation != storage.generation)
  {
    return false;
  }
  BeginAppPhase(storage, FullDevicePhase::condensation,
                storage.generation);
  storage.trace_solution_generation = 0;
  storage.recovery_generation = 0;
  if (!storage.a.GetMemory().DeviceIsValid() ||
      !storage.lu.GetMemory().DeviceIsValid() ||
      !storage.solve_workspace.GetMemory().DeviceIsValid())
  {
    ++evidence.telemetry.dense_host_trace_inverse_events;
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  storage.lu = storage.a;
  storage.condensed_rhs = storage.g;
  storage.schur = 0.0;
  storage.pivots = -1.0;
  storage.kernel_status = 0.0;
  const double* b = storage.b.Read();
  const double* original_a = storage.a.Read();
  const double* c = storage.c.Read();
  const double* d = storage.d.Read();
  const double* f = storage.f.Read();
  const double* original_g = storage.g.Read();
  double* lu = storage.lu.ReadWrite();
  double* pivots = storage.pivots.Write();
  double* workspace = storage.solve_workspace.Write();
  double* schur = storage.schur.Write();
  double* rhs = storage.condensed_rhs.ReadWrite();
  double* telemetry = storage.kernel_status.Write();
  double* audit = storage.condensation_audit.Write();
  const int elements = graph.elements;
  const int ni = graph.local_interior;
  const int nt = graph.local_trace;
  AppForall(storage, elements, [=] MFEM_HOST_DEVICE(int element)
  {
    telemetry[element] = 1.0;
    double* local_lu = lu + static_cast<std::size_t>(element) * ni * ni;
    double* local_pivots = pivots + static_cast<std::size_t>(element) * ni;
    double scale = 0.0;
    for (int entry = 0; entry < ni * ni; ++entry)
    {
      scale = fmax(scale, fabs(local_lu[entry]));
    }
    for (int column = 0; column < ni; ++column)
    {
      int pivot = column;
      for (int row = column + 1; row < ni; ++row)
      {
        if (fabs(local_lu[row * ni + column]) >
            fabs(local_lu[pivot * ni + column]))
        {
          pivot = row;
        }
      }
      local_pivots[column] = static_cast<double>(pivot);
      if (pivot != column)
        for (int entry = 0; entry < ni; ++entry)
        {
          const double temporary = local_lu[column * ni + entry];
          local_lu[column * ni + entry] = local_lu[pivot * ni + entry];
          local_lu[pivot * ni + entry] = temporary;
        }
      if (!(fabs(local_lu[column * ni + column]) > pivot_scale * scale))
      {
        telemetry[element] = -1.0;
        return;
      }
      for (int row = column + 1; row < ni; ++row)
      {
        local_lu[row * ni + column] /= local_lu[column * ni + column];
        for (int entry = column + 1; entry < ni; ++entry)
        {
          local_lu[row * ni + entry] -=
              local_lu[row * ni + column] * local_lu[column * ni + entry];
        }
      }
    }
    double* local_workspace = workspace +
        static_cast<std::size_t>(element) * ni * (nt + 1);
    for (int row = 0; row < ni; ++row)
    {
      for (int column = 0; column < nt; ++column)
      {
        local_workspace[row * (nt + 1) + column] =
            b[(static_cast<std::size_t>(element) * ni + row) * nt + column];
      }
      local_workspace[row * (nt + 1) + nt] =
          f[static_cast<std::size_t>(element) * ni + row];
    }
    device::detail::SolvePivoted(local_lu, local_pivots, ni,
                                 local_workspace, nt + 1);
    for (int row = 0; row < nt; ++row)
    {
      double condensed = rhs[static_cast<std::size_t>(element) * nt + row];
      for (int interior = 0; interior < ni; ++interior)
      {
        condensed -=
            c[(static_cast<std::size_t>(element) * nt + row) * ni + interior] *
            local_workspace[interior * (nt + 1) + nt];
      }
      rhs[static_cast<std::size_t>(element) * nt + row] = condensed;
      for (int column = 0; column < nt; ++column)
      {
        double value =
            d[(static_cast<std::size_t>(element) * nt + row) * nt + column];
        for (int interior = 0; interior < ni; ++interior)
        {
          value -=
              c[(static_cast<std::size_t>(element) * nt + row) * ni + interior] *
              local_workspace[interior * (nt + 1) + column];
        }
        schur[(static_cast<std::size_t>(element) * nt + row) * nt + column] =
            value;
      }
    }
    double local_error = 0.0;
    for (int row = 0; row < ni; ++row)
      for (int column = 0; column <= nt; ++column)
      {
        double value = 0.0;
        for (int inner = 0; inner < ni; ++inner)
          value += original_a[(static_cast<std::size_t>(element) * ni + row) *
                                  ni + inner] *
                   local_workspace[inner * (nt + 1) + column];
        const double expected = column == nt ?
            f[static_cast<std::size_t>(element) * ni + row] :
            b[(static_cast<std::size_t>(element) * ni + row) * nt + column];
        local_error = fmax(local_error, fabs(value - expected));
      }
    double condensed_error = 0.0;
    for (int row = 0; row < nt; ++row)
    {
      double expected_rhs =
          original_g[static_cast<std::size_t>(element) * nt + row];
      for (int inner = 0; inner < ni; ++inner)
        expected_rhs -=
            c[(static_cast<std::size_t>(element) * nt + row) * ni + inner] *
            local_workspace[inner * (nt + 1) + nt];
      condensed_error = fmax(condensed_error,
          fabs(rhs[static_cast<std::size_t>(element) * nt + row] -
               expected_rhs));
      for (int column = 0; column < nt; ++column)
      {
        double expected_schur =
            d[(static_cast<std::size_t>(element) * nt + row) * nt + column];
        for (int inner = 0; inner < ni; ++inner)
          expected_schur -=
              c[(static_cast<std::size_t>(element) * nt + row) * ni + inner] *
              local_workspace[inner * (nt + 1) + column];
        condensed_error = fmax(condensed_error,
            fabs(schur[(static_cast<std::size_t>(element) * nt + row) * nt +
                       column] - expected_schur));
      }
    }
    audit[2 * element] = local_error;
    audit[2 * element + 1] = condensed_error;
  });
  if (!EndPhaseAudit(graph, storage, evidence,
                     FullDevicePhase::condensation,
                     FullDeviceStatus::success, storage.generation))
  {
    evidence.status = FullDeviceStatus::residency_failure;
    return false;
  }
  if (!CheckKernelValidity(storage, graph.elements, evidence.telemetry))
  {
    evidence.status = FullDeviceStatus::condensation_failure;
    return false;
  }
  const double* audit_host = storage.condensation_audit.HostRead();
  evidence.telemetry.device_to_host_bytes +=
      static_cast<std::size_t>(graph.elements) * 2 * sizeof(double);
  ++evidence.telemetry.synchronization_events;
  evidence.local_equation_error = 0.0;
  evidence.condensed_equation_error = 0.0;
  for (int element = 0; element < graph.elements; ++element)
  {
    evidence.local_equation_error = std::max(
        evidence.local_equation_error, audit_host[2 * element]);
    evidence.condensed_equation_error = std::max(
        evidence.condensed_equation_error, audit_host[2 * element + 1]);
  }
  if (!std::isfinite(evidence.local_equation_error) ||
      !std::isfinite(evidence.condensed_equation_error) ||
      evidence.local_equation_error > local_gate ||
      evidence.condensed_equation_error > local_gate)
  {
    evidence.status = FullDeviceStatus::condensation_failure;
    return false;
  }
  storage.condensation_generation = storage.residual_generation;
  storage.trace_system_generation = 0;
  return true;
}

}  // namespace
}  // namespace dg::hdg::euler

namespace dg::hdg::euler
{
namespace
{
bool BuildErrorQuadrature(DeviceGraph& graph, const int quadrature_order,
                          ErrorQuadratureStorage& storage,
                          FullDeviceTransferTelemetry& telemetry)
{
  const mfem::FiniteElement* first_fe = graph.interior_space->GetFE(0);
  if (first_fe == nullptr) { return false; }
  const mfem::IntegrationRule& first_volume_rule =
      mfem::IntRules.Get(first_fe->GetGeomType(), quadrature_order);
  mfem::Array<int> first_faces, first_orientations;
  graph.mesh->GetElementEdges(0, first_faces, first_orientations);
  if (first_faces.Size() != faces_per_element) { return false; }
  const mfem::FiniteElement* first_face_fe =
      graph.trace_space->GetFaceElement(first_faces[0]);
  if (first_face_fe == nullptr) { return false; }
  const mfem::IntegrationRule& first_face_rule =
      mfem::IntRules.Get(first_face_fe->GetGeomType(), quadrature_order);
  storage.volume_points = first_volume_rule.GetNPoints();
  storage.face_points = first_face_rule.GetNPoints();
  if (storage.volume_points <= 0 || storage.face_points <= 0) { return false; }

  std::vector<double> volume_shape(
      static_cast<std::size_t>(graph.elements) * storage.volume_points *
      graph.scalar_dofs);
  std::vector<double> volume_weight(
      static_cast<std::size_t>(graph.elements) * storage.volume_points);
  std::vector<double> volume_position(volume_weight.size() * dimension);
  for (int element = 0; element < graph.elements; ++element)
  {
    const mfem::FiniteElement* fe = graph.interior_space->GetFE(element);
    mfem::ElementTransformation* transformation =
        graph.mesh->GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr ||
        fe->GetDof() != graph.scalar_dofs)
    {
      return false;
    }
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    if (rule.GetNPoints() != storage.volume_points) { return false; }
    mfem::Vector shape(graph.scalar_dofs), physical(dimension);
    for (int q = 0; q < storage.volume_points; ++q)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      const std::size_t quadrature =
          static_cast<std::size_t>(element) * storage.volume_points + q;
      volume_weight[quadrature] = point.weight * transformation->Weight();
      for (int direction = 0; direction < dimension; ++direction)
      {
        volume_position[quadrature * dimension + direction] =
            physical[direction];
      }
      for (int scalar = 0; scalar < graph.scalar_dofs; ++scalar)
      {
        volume_shape[quadrature * graph.scalar_dofs + scalar] = shape[scalar];
      }
    }
  }

  const int faces = graph.mesh->GetNumFaces();
  std::vector<double> face_shape(
      static_cast<std::size_t>(faces) * storage.face_points *
      graph.face_scalar_dofs);
  std::vector<double> face_weight(
      static_cast<std::size_t>(faces) * storage.face_points);
  std::vector<double> face_position(face_weight.size() * dimension);
  std::vector<int> face_signed_vdofs(
      static_cast<std::size_t>(faces) * conserved_components *
      graph.face_scalar_dofs);
  for (int face = 0; face < faces; ++face)
  {
    const mfem::FiniteElement* fe = graph.trace_space->GetFaceElement(face);
    mfem::ElementTransformation* transformation =
        graph.mesh->GetFaceTransformation(face);
    mfem::Array<int> vdofs;
    graph.trace_space->GetFaceVDofs(face, vdofs);
    if (fe == nullptr || transformation == nullptr ||
        fe->GetDof() != graph.face_scalar_dofs ||
        vdofs.Size() != conserved_components * graph.face_scalar_dofs)
    {
      return false;
    }
    const mfem::IntegrationRule& rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    if (rule.GetNPoints() != storage.face_points) { return false; }
    mfem::Vector shape(graph.face_scalar_dofs), physical(dimension);
    for (int local = 0; local < vdofs.Size(); ++local)
    {
      face_signed_vdofs[static_cast<std::size_t>(face) * vdofs.Size() + local] =
          vdofs[local];
    }
    for (int q = 0; q < storage.face_points; ++q)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      const std::size_t quadrature =
          static_cast<std::size_t>(face) * storage.face_points + q;
      face_weight[quadrature] = point.weight * transformation->Weight();
      for (int direction = 0; direction < dimension; ++direction)
      {
        face_position[quadrature * dimension + direction] = physical[direction];
      }
      for (int scalar = 0; scalar < graph.face_scalar_dofs; ++scalar)
      {
        face_shape[quadrature * graph.face_scalar_dofs + scalar] = shape[scalar];
      }
    }
  }
  return StageVector(storage.volume_shape, volume_shape, telemetry) &&
         StageVector(storage.volume_weight, volume_weight, telemetry) &&
         StageVector(storage.volume_position, volume_position, telemetry) &&
         StageVector(storage.face_shape, face_shape, telemetry) &&
         StageVector(storage.face_weight, face_weight, telemetry) &&
         StageVector(storage.face_position, face_position, telemetry) &&
         StageArray(storage.face_signed_vdofs, face_signed_vdofs, telemetry);
}

bool BuildGraphAndStorage(const int p, const int N, DeviceGraph& graph,
                          DeviceStorage& storage,
                          FullDeviceMeshEvidence& evidence)
{
  ++evidence.telemetry.backend_selection_checks;
  if (!mfem::Device::IsConfigured() ||
      !mfem::Device::Allows(mfem::Backend::CUDA_MASK) ||
      mfem::Device::Allows(mfem::Backend::DEBUG_DEVICE))
  {
    return false;
  }
  int mpi_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  if (mpi_size != 1 || p < 1 || p > 4 || N < 3) { return false; }
  graph.p = p;
  graph.N = N;
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        N, N, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(dimension));
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    graph.serial_mesh = std::make_unique<mfem::Mesh>(mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations)));
    graph.mesh =
        std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, *graph.serial_mesh);
    graph.interior_collection = std::make_unique<mfem::L2_FECollection>(
        p, dimension, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    graph.trace_collection =
        std::make_unique<mfem::DG_Interface_FECollection>(
            p, dimension, mfem::FiniteElement::VALUE,
            mfem::BasisType::GaussLegendre);
    graph.interior_space = std::make_unique<mfem::ParFiniteElementSpace>(
        graph.mesh.get(), graph.interior_collection.get(), conserved_components,
        mfem::Ordering::byNODES);
    graph.trace_space = std::make_unique<mfem::ParFiniteElementSpace>(
        graph.mesh.get(), graph.trace_collection.get(), conserved_components,
        mfem::Ordering::byNODES);
  }
  catch (const std::exception&)
  {
    return false;
  }
  graph.elements = graph.mesh->GetNE();
  const mfem::FiniteElement* first_fe = graph.interior_space->GetFE(0);
  mfem::Array<int> first_faces, first_orientations;
  graph.mesh->GetElementEdges(0, first_faces, first_orientations);
  if (graph.elements != N * N || first_fe == nullptr ||
      first_faces.Size() != faces_per_element ||
      graph.interior_space->GetOrdering() != mfem::Ordering::byNODES ||
      graph.trace_space->GetOrdering() != mfem::Ordering::byNODES)
  {
    return false;
  }
  const mfem::FiniteElement* first_face_fe =
      graph.trace_space->GetFaceElement(first_faces[0]);
  if (first_face_fe == nullptr) { return false; }
  graph.scalar_dofs = first_fe->GetDof();
  graph.face_scalar_dofs = first_face_fe->GetDof();
  graph.local_interior = conserved_components * graph.scalar_dofs;
  graph.local_trace = faces_per_element * conserved_components *
                      graph.face_scalar_dofs;
  graph.local_trace_dofs = graph.trace_space->GetVSize();
  graph.true_trace_dofs = graph.trace_space->GetTrueVSize();
  // The callable residual and the independent final-PDE reconstruction use
  // the same admitted high-order rule, while assembling through independent
  // implementations.  A lower production rule can converge its discrete
  // residual while leaving the admitted reconstructed PDE residual above the
  // frozen tolerance.
  const int quadrature_order = 5 * p + 16;
  graph.volume_points =
      mfem::IntRules.Get(first_fe->GetGeomType(), quadrature_order).GetNPoints();
  graph.face_points = mfem::IntRules.Get(first_face_fe->GetGeomType(),
                                         quadrature_order).GetNPoints();
  if (graph.scalar_dofs <= 0 || graph.face_scalar_dofs <= 0 ||
      graph.local_interior <= 0 || graph.local_trace <= 0 ||
      graph.local_trace_dofs <= 0 ||
      graph.local_trace_dofs != graph.true_trace_dofs ||
      graph.volume_points <= 0 || graph.face_points <= 0)
  {
    return false;
  }

  std::vector<double> old_state(
      static_cast<std::size_t>(graph.elements) * graph.local_interior);
  std::vector<double> current_state(old_state.size());
  std::vector<double> current_trace(graph.local_trace_dofs, 0.0);
  std::vector<bool> trace_assigned(graph.local_trace_dofs, false);
  std::vector<int> signed_trace_dofs(
      static_cast<std::size_t>(graph.elements) * graph.local_trace);
  std::vector<double> volume_shape(
      static_cast<std::size_t>(graph.elements) * graph.volume_points *
      graph.scalar_dofs);
  std::vector<double> volume_derivative(volume_shape.size() * dimension);
  std::vector<double> volume_weight(
      static_cast<std::size_t>(graph.elements) * graph.volume_points);
  std::vector<double> volume_position(volume_weight.size() * dimension);
  std::vector<double> face_interior_shape(
      static_cast<std::size_t>(graph.elements) * faces_per_element *
      graph.face_points * graph.scalar_dofs);
  std::vector<double> face_trace_shape(
      static_cast<std::size_t>(graph.elements) * faces_per_element *
      graph.face_points * graph.face_scalar_dofs);
  std::vector<double> face_weight(
      static_cast<std::size_t>(graph.elements) * faces_per_element *
      graph.face_points);
  std::vector<double> face_normal(face_weight.size() * dimension);
  int negative_trace_vdof_count = 0;
  bool reversed_trace_permutation_observed = false;

  for (int element = 0; element < graph.elements; ++element)
  {
    const mfem::FiniteElement* fe = graph.interior_space->GetFE(element);
    mfem::ElementTransformation* transformation =
        graph.mesh->GetElementTransformation(element);
    if (fe == nullptr || transformation == nullptr ||
        fe->GetDof() != graph.scalar_dofs)
    {
      return false;
    }
    const mfem::IntegrationRule& nodes = fe->GetNodes();
    if (nodes.GetNPoints() != graph.scalar_dofs) { return false; }
    for (int scalar = 0; scalar < graph.scalar_dofs; ++scalar)
    {
      mfem::Vector physical(dimension);
      transformation->Transform(nodes.IntPoint(scalar), physical);
      double exact[conserved_components], perturbed[conserved_components];
      ManufacturedState(physical[0], physical[1], exact);
      PerturbedState(physical[0], physical[1], perturbed);
      for (int component = 0; component < conserved_components; ++component)
      {
        const std::size_t offset = static_cast<std::size_t>(element) *
                                       graph.local_interior +
                                   LocalByNodes(scalar, graph.scalar_dofs,
                                                component);
        old_state[offset] = exact[component];
        current_state[offset] = perturbed[component];
      }
    }
    const mfem::IntegrationRule& volume_rule =
        mfem::IntRules.Get(fe->GetGeomType(), quadrature_order);
    if (volume_rule.GetNPoints() != graph.volume_points) { return false; }
    mfem::Vector shape(graph.scalar_dofs), physical(dimension);
    mfem::DenseMatrix derivative(graph.scalar_dofs, dimension);
    for (int q = 0; q < graph.volume_points; ++q)
    {
      const mfem::IntegrationPoint& point = volume_rule.IntPoint(q);
      transformation->SetIntPoint(&point);
      transformation->Transform(point, physical);
      fe->CalcShape(point, shape);
      fe->CalcPhysDShape(*transformation, derivative);
      const std::size_t quadrature =
          static_cast<std::size_t>(element) * graph.volume_points + q;
      volume_weight[quadrature] = point.weight * transformation->Weight();
      for (int direction = 0; direction < dimension; ++direction)
      {
        volume_position[quadrature * dimension + direction] =
            physical[direction];
      }
      for (int scalar = 0; scalar < graph.scalar_dofs; ++scalar)
      {
        volume_shape[quadrature * graph.scalar_dofs + scalar] = shape[scalar];
        for (int direction = 0; direction < dimension; ++direction)
        {
          volume_derivative[(quadrature * graph.scalar_dofs + scalar) *
                                dimension +
                            direction] = derivative(scalar, direction);
        }
      }
    }

    mfem::Array<int> faces, orientations;
    graph.mesh->GetElementEdges(element, faces, orientations);
    if (faces.Size() != faces_per_element ||
        orientations.Size() != faces_per_element)
    {
      return false;
    }
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces_per_element; ++local_face)
    {
      const int face = faces[local_face];
      mfem::FaceElementTransformations* face_transformation =
          graph.mesh->GetFaceElementTransformations(face);
      const mfem::FiniteElement* trace_fe =
          graph.trace_space->GetFaceElement(face);
      mfem::Array<int> face_vdofs;
      graph.trace_space->GetFaceVDofs(face, face_vdofs);
      if (face_transformation == nullptr ||
          face_transformation->Face == nullptr || trace_fe == nullptr ||
          trace_fe->GetDof() != graph.face_scalar_dofs ||
          face_vdofs.Size() !=
              conserved_components * graph.face_scalar_dofs)
      {
        return false;
      }
      const bool is_element_one = face_transformation->Elem1No == element;
      if (!is_element_one && face_transformation->Elem2No != element)
      {
        return false;
      }
      const mfem::IntegrationRule& face_nodes = trace_fe->GetNodes();
      for (int scalar = 0; scalar < graph.face_scalar_dofs; ++scalar)
      {
        mfem::Vector physical(dimension);
        face_transformation->Face->Transform(face_nodes.IntPoint(scalar),
                                             physical);
        double perturbed[conserved_components];
        PerturbedState(physical[0], physical[1], perturbed);
        for (int component = 0; component < conserved_components; ++component)
        {
          const int local = LocalByNodes(scalar, graph.face_scalar_dofs,
                                         component);
          double sign = 1.0;
          const int global = DecodeSignedVDof(face_vdofs[local], sign);
          if (global < 0 || global >= graph.local_trace_dofs) { return false; }
          negative_trace_vdof_count += face_vdofs[local] < 0 ? 1 : 0;
          signed_trace_dofs[static_cast<std::size_t>(element) *
                                graph.local_trace +
                            trace_offset + local] = face_vdofs[local];
          const double value = sign * perturbed[component];
          if (trace_assigned[global] &&
              std::abs(current_trace[global] - value) > 1.0e-12)
          {
            return false;
          }
          current_trace[global] = value;
          trace_assigned[global] = true;
        }
      }
      reversed_trace_permutation_observed =
          reversed_trace_permutation_observed ||
          orientations[local_face] == -1;
      const mfem::IntegrationRule& face_rule =
          mfem::IntRules.Get(trace_fe->GetGeomType(), quadrature_order);
      if (face_rule.GetNPoints() != graph.face_points) { return false; }
      mfem::Vector interior_shape(graph.scalar_dofs);
      mfem::Vector trace_shape(graph.face_scalar_dofs);
      for (int q = 0; q < graph.face_points; ++q)
      {
        const mfem::IntegrationPoint& face_point = face_rule.IntPoint(q);
        face_transformation->SetAllIntPoints(&face_point);
        const mfem::IntegrationPoint& element_point = is_element_one ?
            face_transformation->GetElement1IntPoint() :
            face_transformation->GetElement2IntPoint();
        mfem::Vector scaled_normal(dimension);
        mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled_normal);
        const double measure = scaled_normal.Norml2();
        if (!(measure > 0.0) || !std::isfinite(measure)) { return false; }
        const double side = is_element_one ? 1.0 : -1.0;
        fe->CalcShape(element_point, interior_shape);
        trace_fe->CalcShape(face_point, trace_shape);
        const std::size_t quadrature =
            (static_cast<std::size_t>(element) * faces_per_element +
             local_face) *
                graph.face_points +
            q;
        face_weight[quadrature] = face_point.weight * measure;
        face_normal[quadrature * dimension] =
            side * scaled_normal[0] / measure;
        face_normal[quadrature * dimension + 1] =
            side * scaled_normal[1] / measure;
        for (int scalar = 0; scalar < graph.scalar_dofs; ++scalar)
        {
          face_interior_shape[quadrature * graph.scalar_dofs + scalar] =
              interior_shape[scalar];
        }
        for (int scalar = 0; scalar < graph.face_scalar_dofs; ++scalar)
        {
          face_trace_shape[quadrature * graph.face_scalar_dofs + scalar] =
              trace_shape[scalar];
        }
      }
      trace_offset += face_vdofs.Size();
    }
    if (trace_offset != graph.local_trace) { return false; }
  }
  if (std::find(trace_assigned.begin(), trace_assigned.end(), false) !=
      trace_assigned.end())
  {
    return false;
  }
  graph.host_signed_trace_dofs = signed_trace_dofs;

  auto& telemetry = evidence.telemetry;
  if (!StageArray(graph.signed_trace_dofs, signed_trace_dofs, telemetry) ||
      !StageVector(storage.old_state, old_state, telemetry) ||
      !StageVector(storage.entry_state, current_state, telemetry) ||
      !StageVector(storage.current_state, current_state, telemetry) ||
      !StageVector(storage.base_state, current_state, telemetry) ||
      !StageVector(storage.trial_state, current_state, telemetry) ||
      !StageVector(storage.entry_trace, current_trace, telemetry) ||
      !StageVector(storage.current_trace, current_trace, telemetry) ||
      !StageVector(storage.base_trace, current_trace, telemetry) ||
      !StageVector(storage.trial_trace, current_trace, telemetry) ||
      !StageVector(storage.volume_shape, volume_shape, telemetry) ||
      !StageVector(storage.volume_derivative, volume_derivative, telemetry) ||
      !StageVector(storage.volume_weight, volume_weight, telemetry) ||
      !StageVector(storage.volume_position, volume_position, telemetry) ||
      !StageVector(storage.face_interior_shape, face_interior_shape, telemetry) ||
      !StageVector(storage.face_trace_shape, face_trace_shape, telemetry) ||
      !StageVector(storage.face_weight, face_weight, telemetry) ||
      !StageVector(storage.face_normal, face_normal, telemetry))
  {
    return false;
  }

  const std::size_t elements = static_cast<std::size_t>(graph.elements);
  const std::size_t ni = static_cast<std::size_t>(graph.local_interior);
  const std::size_t nt = static_cast<std::size_t>(graph.local_trace);
  const std::size_t local_trace_dofs =
      static_cast<std::size_t>(graph.local_trace_dofs);
  const std::size_t true_trace_dofs =
      static_cast<std::size_t>(graph.true_trace_dofs);
  const std::size_t faces = static_cast<std::size_t>(graph.mesh->GetNumFaces());
  const std::array<std::pair<mfem::Vector*, std::size_t>, 32> warm{{
      {&storage.local_trace, elements * nt},
      {&storage.local_trace_increment, elements * nt},
      {&storage.residual_r, elements * ni},
      {&storage.residual_g, elements * nt},
      {&storage.a, elements * ni * ni},
      {&storage.b, elements * ni * nt},
      {&storage.c, elements * nt * ni},
      {&storage.d, elements * nt * nt},
      {&storage.f, elements * ni},
      {&storage.g, elements * nt},
      {&storage.lu, elements * ni * ni},
      {&storage.pivots, elements * ni},
      {&storage.solve_workspace, elements * ni * (nt + 1)},
      {&storage.schur, elements * nt * nt},
      {&storage.condensed_rhs, elements * nt},
      {&storage.local_sparse_rhs, local_trace_dofs},
      {&storage.true_rhs, true_trace_dofs},
      {&storage.true_action, true_trace_dofs},
      {&storage.distributed_trace_increment, local_trace_dofs},
      {&storage.recovered, elements * ni},
      {&storage.delta_state, elements * ni},
      {&storage.physics_reduction, 3},
      {&storage.prescreen_reduction, elements * 4},
      {&storage.kernel_status, elements},
      {&storage.volume_error_reduction, elements * 3},
      {&storage.face_error_reduction, faces * 3},
      {&storage.condensation_audit, elements * 2},
      {&storage.recovery_audit, elements},
      {&storage.state_audit, 1},
      {&storage.true_matrix_audit, 3},
      {&storage.rollback_reduction, 1},
      {&storage.app_forall_device_tickets, max_app_forall_records}}};
  for (const auto& item : warm)
  {
    if (!WarmVector(*item.first, item.second, telemetry)) { return false; }
  }
  // The first true-to-local prolongation lazily creates a HYPRE device work
  // buffer whose shape is fixed by this trace space.  Perform that allocation
  // in setup so every marked recovery interval measures the warmed, steady
  // recovery path rather than misclassifying one-time backend setup as an
  // application steady-state allocation.
  storage.true_action = 0.0;
  storage.distributed_trace_increment = 0.0;
  graph.trace_space->Dof_TrueDof_Matrix()->Mult(
      storage.true_action, storage.distributed_trace_increment);
  if (!storage.distributed_trace_increment.GetMemory().DeviceIsValid())
  {
    return false;
  }
  const int elements_i = graph.elements;
  const int nt_i = graph.local_trace;
  const int* signed_map = graph.signed_trace_dofs.Read();
  const double* global_trace = storage.current_trace.Read();
  double* local_trace = storage.local_trace.Write();
  AppForall(storage, elements_i * nt_i,
            [=] MFEM_HOST_DEVICE(int entry)
  {
    const int encoded = signed_map[entry];
    const double sign = encoded >= 0 ? 1.0 : -1.0;
    const int global = encoded >= 0 ? encoded : -1 - encoded;
    local_trace[entry] = sign * global_trace[global];
  });

  try
  {
    graph.local_trace_matrix =
        std::make_unique<mfem::SparseMatrix>(graph.local_trace_dofs);
    for (int element = 0; element < graph.elements; ++element)
      for (int row = 0; row < graph.local_trace; ++row)
      {
        double row_sign = 1.0;
        const int global_row = DecodeSignedVDof(
            signed_trace_dofs[static_cast<std::size_t>(element) *
                                  graph.local_trace +
                              row],
            row_sign);
        for (int column = 0; column < graph.local_trace; ++column)
        {
          double column_sign = 1.0;
          const int global_column = DecodeSignedVDof(
              signed_trace_dofs[static_cast<std::size_t>(element) *
                                    graph.local_trace +
                                column],
              column_sign);
          graph.local_trace_matrix->Add(global_row, global_column,
                                        row_sign * column_sign);
        }
      }
    graph.local_trace_matrix->Finalize();
    std::fill_n(graph.local_trace_matrix->HostWriteData(),
                graph.local_trace_matrix->NumNonZeroElems(), 1.0);
  }
  catch (const std::exception&)
  {
    return false;
  }

  if (!BuildErrorQuadrature(graph, 4 * p + 12, storage.qe, telemetry) ||
      !BuildErrorQuadrature(graph, 4 * p + 16, storage.qe4, telemetry))
  {
    return false;
  }
  bool orientation_action_verified = false;
  const int* reversed = graph.trace_collection->DofOrderForOrientation(
      mfem::Geometry::SEGMENT, -1);
  if (reversed != nullptr && reversed_trace_permutation_observed)
  {
    bool nontrivial_positive_permutation = graph.face_scalar_dofs > 1;
    for (int scalar = 0; scalar < graph.face_scalar_dofs; ++scalar)
    {
      nontrivial_positive_permutation =
          nontrivial_positive_permutation && reversed[scalar] >= 0 &&
          reversed[scalar] < graph.face_scalar_dofs &&
          reversed[scalar] == graph.face_scalar_dofs - 1 - scalar;
    }
    const mfem::FiniteElement* trace_fe =
        graph.trace_space->GetFaceElement(0);
    if (nontrivial_positive_permutation && trace_fe != nullptr)
    {
      const mfem::IntegrationRule& rule = mfem::IntRules.Get(
          trace_fe->GetGeomType(), 2 * p + 8);
      mfem::Vector forward_shape(graph.face_scalar_dofs);
      mfem::Vector reversed_shape(graph.face_scalar_dofs);
      long double forward_action = 0.0L;
      long double reversed_action = 0.0L;
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const mfem::IntegrationPoint& point = rule.IntPoint(q);
        mfem::IntegrationPoint reverse_point(point);
        reverse_point.x = 1.0 - point.x;
        trace_fe->CalcShape(point, forward_shape);
        trace_fe->CalcShape(reverse_point, reversed_shape);
        double forward_value = 0.0;
        double reversed_value = 0.0;
        for (int scalar = 0; scalar < graph.face_scalar_dofs; ++scalar)
        {
          const double coefficient = 1.0 + 0.25 * scalar;
          forward_value += forward_shape[scalar] * coefficient;
          reversed_value += reversed_shape[scalar] *
              (1.0 + 0.25 * reversed[scalar]);
        }
        forward_action += point.weight * forward_value * forward_value;
        reversed_action += point.weight * reversed_value * reversed_value;
      }
      orientation_action_verified =
          std::abs(static_cast<double>(forward_action - reversed_action)) <=
          256.0 * experiment_epsilon *
              std::max(1.0, std::abs(static_cast<double>(forward_action)));
    }
  }
  evidence.negative_trace_vdof_count = negative_trace_vdof_count;
  evidence.reversed_trace_permutation_observed =
      reversed_trace_permutation_observed;
  evidence.shared_face_orientation_action_verified =
      orientation_action_verified;
  evidence.bynodes_vdim4 = true;
  evidence.elements = graph.elements;
  evidence.interior_unknowns_per_element = graph.local_interior;
  evidence.trace_unknowns_per_element = graph.local_trace;
  evidence.local_trace_unknowns = graph.local_trace_dofs;
  evidence.true_trace_unknowns = graph.true_trace_dofs;
  return true;
}

}  // namespace
}  // namespace dg::hdg::euler

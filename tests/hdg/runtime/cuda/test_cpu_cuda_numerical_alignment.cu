#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cpu::detail
{

// Test-only access to the conservation gate already retained by the generic
// CPU numerics core.  No solver state or diagnostic surface is published.
struct CpuQualificationTestAccess
{
  static bool AuditLastConservation(
      const CpuNumericsCore& core, double& maximum_error,
      double& maximum_relative_error, bool& gate) noexcept
  {
    return core.AuditLastBorrowedEulerConservation(
        maximum_error, maximum_relative_error, gate);
  }
};

} // namespace dg::hdg::runtime::cpu::detail

namespace dg::hdg::runtime::cuda_w8a::detail
{

struct CrossBackendPreparedHealth
{
  double initial_residual = std::numeric_limits<double>::infinity();
  double final_residual = std::numeric_limits<double>::infinity();
  double true_relative_residual = std::numeric_limits<double>::infinity();
  double maximum_conservation_error =
      std::numeric_limits<double>::infinity();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::infinity();
  int nonlinear_updates = -1;
  int trace_solves = -1;
  bool finite = false;
  bool admissible = false;
  bool recovery = false;
  bool true_residual = false;
  bool conservation = false;
  bool success = false;
};

// This accessor exists only in this qualification executable.  It observes
// the already-computed prepared gates and copies the logical committed true
// trace to host storage; it launches no numerical work and publishes no API.
struct CudaQualificationTestAccess
{
  template <typename Engine>
  static bool CapturePreparedHealth(
      Engine& engine, CrossBackendPreparedHealth& health) noexcept
  {
    health = {};
    const auto& pending = engine.pending_;
    const auto& numerical = pending.numerical;
    health.initial_residual = numerical.initial_residual;
    health.final_residual = numerical.final_residual;
    health.true_relative_residual = numerical.true_relative_residual;
    health.maximum_conservation_error =
        numerical.maximum_conservation_error;
    health.maximum_conservation_relative_error =
        numerical.maximum_conservation_relative_error;
    health.nonlinear_updates = numerical.nonlinear_updates;
    health.trace_solves = numerical.trace_solves;
    health.finite = numerical.finite;
    health.admissible = numerical.admissible;
    health.recovery = numerical.recovery;
    health.true_residual = numerical.true_residual;
    health.conservation = numerical.conservation;
    health.success = numerical.success;
    return engine.initialized_ && pending.active && pending.staging &&
        pending.prestage_allowed && numerical.candidate_ready;
  }

  static bool CapturePreparedHealth(
      cuda::GenericEulerHdgCudaSession& session,
      CrossBackendPreparedHealth& health) noexcept
  {
    return session.borrowed_rank1_ &&
        CapturePreparedHealth(*session.borrowed_rank1_, health);
  }

  template <typename Engine>
  static bool ExportCommittedTrace(
      Engine& engine, std::vector<double>& trace,
      int& trace_components) noexcept
  {
    trace.clear();
    trace_components = 0;
    auto& numerics = engine.numerics_;
    const auto committed = numerics.CommittedSlot();
    bool ready = engine.initialized_ && !engine.pending_.active &&
        committed.trace.coefficients != nullptr &&
        committed.trace.count != 0 && numerics.shape_.trace_components > 0;
    try
    {
      if (ready) trace.resize(committed.trace.count);
    }
    catch (...) { ready = false; }
    if (!numerics.CollectiveContinue(ready)) return false;
    const bool copied = CudaTransportPolicy::DeviceToHost(
            trace.data(), committed.trace.coefficients,
            trace.size() * sizeof(double),
            CudaTransferCategory::generic_host_bridge,
            CudaTransferSite::generic_host_export) &&
        CudaTransportPolicy::Synchronize(
            CudaTransferSite::generic_host_export);
    if (!numerics.CollectiveContinue(copied)) return false;
    trace_components = numerics.shape_.trace_components;
    return true;
  }

  static bool ExportCommittedTrace(
      cuda::GenericEulerHdgCudaSession& session,
      std::vector<double>& trace, int& trace_components) noexcept
  {
    return session.borrowed_rank1_ && ExportCommittedTrace(
        *session.borrowed_rank1_, trace, trace_components);
  }
};

} // namespace dg::hdg::runtime::cuda_w8a::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::ContractStatus;
using dg::hdg::generic::EulerBoundaryPolicy;
using dg::hdg::generic::EulerBoundaryPolicyKind;
using dg::hdg::generic::EulerBoundaryPolicyTableView;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::StateCoordinates;
using dg::hdg::runtime::cpu::CpuAttemptStatus;
using dg::hdg::runtime::cpu::detail::CpuNumericsCore;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::CudaMutableHostL2Output;
using dg::hdg::runtime::cuda_w8a::detail::CrossBackendPreparedHealth;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;

constexpr int components = dg::hdg::generic::euler_2d_conserved_components;
constexpr int order = 1;
constexpr double comparison_relative_tolerance = 5.0e-8;
constexpr double comparison_absolute_tolerance = 5.0e-9;

constexpr std::uint32_t common_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

enum class InitialMode
{
  smooth_periodic,
  stationary_contact
};

struct Case
{
  std::string_view name;
  InitialMode initial_mode = InitialMode::smooth_periodic;
  RegularizationKind regularization = RegularizationKind::none;
  bool physical = false;
  int steps = 1;
  double dt = 1.0e-5;
};

struct BoundaryContext
{
  std::array<EulerBoundaryPolicy, 4> policies{};
  std::uint64_t calls = 0;
};

struct BorrowedFields
{
  std::vector<double> initial;
  std::size_t scalar_count = 0;
};

struct ComparisonMetrics
{
  double relative_l2 = std::numeric_limits<double>::infinity();
  double maximum_absolute = std::numeric_limits<double>::infinity();
  double scale = 1.0;
  bool passed = false;
};

struct StepState
{
  std::vector<double> physical_interior;
  std::vector<double> true_trace;
  double time = 0.0;
  std::uint64_t step = 0;
  std::uint64_t generation = 0;
  double maximum_wave_speed = 0.0;
  double backend_stable_dt = 0.0;
  double initial_residual = 0.0;
  double final_residual = 0.0;
  double true_relative_residual = 0.0;
  double maximum_conservation_error = 0.0;
  double maximum_conservation_relative_error = 0.0;
  int nonlinear_updates = 0;
  int trace_solves = 0;
  int trace_components = 0;
};

struct Trajectory
{
  std::vector<StepState> states;
  bool passed = false;
};

bool All(const bool local)
{
  const int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

mfem::ParMesh MakeMesh(const bool physical)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      4, 4, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  if (physical) return mfem::ParMesh(MPI_COMM_WORLD, serial);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0] = 0.0;
  translations[1] = 0.0;
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      serial, serial.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

void EvaluateInitial(const InitialMode mode, const mfem::Vector& point,
                     mfem::Vector& state)
{
  state.SetSize(components);
  if (mode == InitialMode::stationary_contact)
  {
    const double density = point[0] < 0.5 ? 1.0 : 1.2;
    state[0] = density;
    state[1] = 0.0;
    state[2] = 0.0;
    state[3] = 2.5;
    return;
  }
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double sx = std::sin(2.0 * pi * point[0]);
  const double sy = std::sin(2.0 * pi * point[1]);
  const double cx = std::cos(2.0 * pi * point[0]);
  const double cy = std::cos(2.0 * pi * point[1]);
  const double density = 1.0 + 0.04 * sx * sy;
  const double velocity_x = 0.18 + 0.02 * cx * sy;
  const double velocity_y = -0.09 + 0.015 * sx * cy;
  const double pressure = 1.0 + 0.03 * cx * cy;
  state[0] = density;
  state[1] = density * velocity_x;
  state[2] = density * velocity_y;
  state[3] = pressure / 0.4 +
      0.5 * density *
          (velocity_x * velocity_x + velocity_y * velocity_y);
}

BorrowedFields MakeFields(mfem::ParMesh& mesh, const InitialMode mode)
{
  BorrowedFields fields;
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, components, mfem::Ordering::byNODES);
  mfem::ParGridFunction projected(&space);
  const auto evaluator = [mode](const mfem::Vector& point,
                                mfem::Vector& state)
  { EvaluateInitial(mode, point, state); };
  mfem::VectorFunctionCoefficient coefficient(components, evaluator);
  projected.ProjectCoefficient(coefficient);
  fields.scalar_count = static_cast<std::size_t>(space.GetVSize() / components);
  const double* values = projected.HostRead();
  if (values != nullptr && projected.Size() > 0)
    fields.initial.assign(values, values + projected.Size());
  return fields;
}

bool EulerExterior(const void* opaque,
                   const dg::hdg::generic::BoundaryPointView& point,
                   const double* interior, double* exterior) noexcept
{
  auto* context = static_cast<BoundaryContext*>(const_cast<void*>(opaque));
  if (context == nullptr || point.position == nullptr ||
      point.outward_normal == nullptr || point.space_dimension != 2 ||
      point.boundary_attribute < 1 || point.boundary_attribute > 4 ||
      interior == nullptr || exterior == nullptr)
    return false;
  ++context->calls;
  std::copy_n(interior, components, exterior);
  const double normal_momentum =
      interior[1] * point.outward_normal[0] +
      interior[2] * point.outward_normal[1];
  exterior[1] -= 2.0 * normal_momentum * point.outward_normal[0];
  exterior[2] -= 2.0 * normal_momentum * point.outward_normal[1];
  return true;
}

void ConfigureBoundary(BoundaryContext& boundary)
{
  const std::array<double, components> unused{};
  for (EulerBoundaryPolicy& policy : boundary.policies)
    policy = {EulerBoundaryPolicyKind::slip_wall, unused, 0.0};
}

EulerHdgRequest MakeRequest(
    const mfem::ParMesh& mesh, const BorrowedFields& fields,
    BoundaryContext& boundary, const Case& specification,
    const BackendKind backend)
{
  ConfigureBoundary(boundary);
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {backend, backend == BackendKind::cuda ? 0 : -1};
  request.capabilities = common_capabilities | CapabilityBit(
      backend == BackendKind::cuda ? Capability::cuda : Capability::cpu);
  request.initial_state = {
      fields.initial.data(), fields.initial.size(), components};
  // Deliberately exercise the common projected-trace path.  Supplied traces
  // already have independent p1-p3 coverage in the generic extension suite.
  request.initial_true_trace = {nullptr, 0, components};
  request.source = {nullptr, 0, components};
  request.boundary.context = &boundary;
  request.boundary.exterior_state = EulerExterior;
  request.boundary.policies = specification.physical
      ? EulerBoundaryPolicyTableView{
            boundary.policies.data(), boundary.policies.size(), 2, 1.4}
      : EulerBoundaryPolicyTableView{nullptr, 0, 2, 1.4};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
  request.numerical_policy.stabilization =
      specification.regularization == RegularizationKind::igr
      ? dg::hdg::generic::RegularizationSelection{
            RegularizationKind::igr, 0.2, 0.8}
      : dg::hdg::generic::RegularizationSelection{
            RegularizationKind::none, 0.0, 0.0};
  return request;
}

ComparisonMetrics Compare(const std::vector<double>& left,
                          const std::vector<double>& right)
{
  ComparisonMetrics metrics;
  if (left.size() != right.size() || left.empty()) return metrics;
  long double difference_squared = 0.0L;
  long double reference_squared = 0.0L;
  metrics.maximum_absolute = 0.0;
  metrics.scale = 1.0;
  for (std::size_t index = 0; index < left.size(); ++index)
  {
    if (!std::isfinite(left[index]) || !std::isfinite(right[index]))
      return metrics;
    const double difference = left[index] - right[index];
    difference_squared += static_cast<long double>(difference) * difference;
    reference_squared += std::max(
        static_cast<long double>(left[index]) * left[index],
        static_cast<long double>(right[index]) * right[index]);
    metrics.maximum_absolute = std::max(
        metrics.maximum_absolute, std::abs(difference));
    metrics.scale = std::max(
        metrics.scale, std::max(std::abs(left[index]),
                                std::abs(right[index])));
  }
  metrics.relative_l2 = static_cast<double>(
      std::sqrt(difference_squared) /
      std::max(1.0L, std::sqrt(reference_squared)));
  metrics.passed =
      metrics.relative_l2 <= comparison_relative_tolerance &&
      metrics.maximum_absolute <= comparison_absolute_tolerance +
          comparison_relative_tolerance * metrics.scale;
  return metrics;
}

bool PhysicalCoefficients(const std::vector<double>& values,
                          const std::size_t scalar_count)
{
  if (values.size() != scalar_count * components || scalar_count == 0)
    return false;
  for (std::size_t scalar = 0; scalar < scalar_count; ++scalar)
  {
    const double density = values[scalar];
    const double momentum_x = values[scalar_count + scalar];
    const double momentum_y = values[2 * scalar_count + scalar];
    const double energy = values[3 * scalar_count + scalar];
    const double pressure = 0.4 *
        (energy - 0.5 *
            (momentum_x * momentum_x + momentum_y * momentum_y) /
                density);
    if (!(density > 1.0e-8) || !(pressure > 1.0e-8) ||
        !std::isfinite(density) || !std::isfinite(pressure))
      return false;
  }
  return true;
}

bool HasContactJump(const BorrowedFields& fields)
{
  if (fields.scalar_count == 0 ||
      fields.initial.size() != fields.scalar_count * components)
    return false;
  const auto bounds = std::minmax_element(
      fields.initial.begin(),
      fields.initial.begin() + static_cast<std::ptrdiff_t>(
          fields.scalar_count));
  return *bounds.second - *bounds.first >= 0.19;
}

bool CaptureCpuState(CpuNumericsCore& core,
                     const StateCoordinates& coordinates,
                     const std::size_t scalar_count, StepState& state)
{
  state.physical_interior.assign(scalar_count * components, 0.0);
  const auto snapshot = core.Snapshot(coordinates);
  if (!core.ExportCommittedInterior(
          state.physical_interior.data(), state.physical_interior.size()) ||
      snapshot.trace.data == nullptr || snapshot.trace.size == 0)
    return false;
  state.true_trace.assign(
      snapshot.trace.data, snapshot.trace.data + snapshot.trace.size);
  state.time = coordinates.time;
  state.step = coordinates.step;
  state.generation = coordinates.generation;
  return core.CommittedBorrowedEulerDiagnostics(
             state.maximum_wave_speed, state.backend_stable_dt) &&
      PhysicalCoefficients(state.physical_interior, scalar_count);
}

Trajectory RunCpu(const EulerHdgRequest& request,
                  const BorrowedFields& fields, const Case& specification)
{
  Trajectory trajectory;
  CpuNumericsCore core;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
          ContractStatus::accepted ||
      !core.InitializeBorrowedEuler(request))
    return trajectory;
  StateCoordinates coordinates{};
  StepState initial;
  if (!CaptureCpuState(core, coordinates, fields.scalar_count, initial))
    return trajectory;
  initial.trace_components = specification.regularization ==
          RegularizationKind::igr ? 5 : components;
  if (initial.true_trace.size() %
          static_cast<std::size_t>(initial.trace_components) != 0)
    return trajectory;
  trajectory.states.push_back(std::move(initial));

  for (int step = 0; step < specification.steps; ++step)
  {
    if (!core.AdvanceBorrowedEuler(specification.dt, coordinates))
      return trajectory;
    const auto& attempt = core.evidence().last_attempt;
    bool conservation_gate = true;
    double conservation_error = 0.0;
    double conservation_relative_error = 0.0;
    const bool conservation_ready =
        dg::hdg::runtime::cpu::detail::CpuQualificationTestAccess::
            AuditLastConservation(
                core, conservation_error, conservation_relative_error,
                conservation_gate);
    const bool health = conservation_ready && conservation_gate &&
        attempt.status == CpuAttemptStatus::success && attempt.committed &&
        attempt.finite_gate && attempt.admissibility_gate &&
        attempt.recovery_gate && attempt.nonlinear_gate &&
        attempt.true_residual_gate &&
        attempt.nonlinear_updates <=
            request.numerical_policy.maximum_nonlinear_updates &&
        std::isfinite(attempt.initial_nonlinear_residual) &&
        std::isfinite(attempt.final_nonlinear_residual) &&
        attempt.final_nonlinear_residual <=
            request.numerical_policy.nonlinear_relative_tolerance *
                (1.0 + attempt.initial_nonlinear_residual) &&
        attempt.maximum_true_relative_residual <=
            request.numerical_policy.true_residual_relative_tolerance &&
        conservation_relative_error <=
            request.numerical_policy.conservation_relative_tolerance;
    StepState state;
    if (!health ||
        !CaptureCpuState(core, coordinates, fields.scalar_count, state))
      return trajectory;
    state.initial_residual = attempt.initial_nonlinear_residual;
    state.final_residual = attempt.final_nonlinear_residual;
    state.true_relative_residual = attempt.maximum_true_relative_residual;
    state.maximum_conservation_error = conservation_error;
    state.maximum_conservation_relative_error =
        conservation_relative_error;
    state.nonlinear_updates = attempt.nonlinear_updates;
    state.trace_solves = static_cast<int>(
        attempt.work.complete_trace_solves);
    state.trace_components = specification.regularization ==
            RegularizationKind::igr ? 5 : components;
    if (state.true_trace.size() %
            static_cast<std::size_t>(state.trace_components) != 0)
      return trajectory;
    trajectory.states.push_back(std::move(state));
  }
  trajectory.passed = trajectory.states.size() ==
      static_cast<std::size_t>(specification.steps + 1);
  return trajectory;
}

bool CaptureCudaState(GenericEulerHdgCudaSession& session,
                      const std::size_t scalar_count, StepState& state)
{
  state.physical_interior.assign(scalar_count * components, 0.0);
  if (!session.ExportCommittedInterior(CudaMutableHostL2Output{
          state.physical_interior.data(), scalar_count, components}) ||
      !CudaQualificationTestAccess::ExportCommittedTrace(
          session, state.true_trace, state.trace_components) ||
      !PhysicalCoefficients(state.physical_interior, scalar_count))
    return false;
  const auto coordinates = session.coordinates();
  state.time = coordinates.time;
  state.step = coordinates.step;
  state.generation = coordinates.generation;
  const auto diagnostics = session.diagnostics();
  if (!diagnostics.available) return false;
  state.maximum_wave_speed = diagnostics.maximum_wave_speed;
  state.backend_stable_dt = diagnostics.backend_stable_dt;
  return true;
}

Trajectory RunCuda(const EulerHdgRequest& request,
                   const BorrowedFields& fields, const Case& specification)
{
  Trajectory trajectory;
  GenericEulerHdgCudaSession session;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
          ContractStatus::accepted ||
      session.Initialize(request).status !=
          dg::hdg::generic::HdgEulerSessionStatus::initialized)
    return trajectory;
  StepState initial;
  if (!CaptureCudaState(session, fields.scalar_count, initial))
    return trajectory;
  trajectory.states.push_back(std::move(initial));

  for (int step = 0; step < specification.steps; ++step)
  {
    GenericEulerCudaPreparedStep authority;
    GenericEulerHdgCudaSession::prepared_output_type output{};
    if (session.Prepare(specification.dt, authority, output).status !=
        dg::hdg::generic::HdgEulerSessionStatus::prepared)
      return trajectory;
    CrossBackendPreparedHealth health;
    const bool captured =
        CudaQualificationTestAccess::CapturePreparedHealth(session, health);
    const bool healthy = captured && health.success && health.finite &&
        health.admissible && health.recovery && health.true_residual &&
        health.conservation &&
        health.nonlinear_updates >= 0 &&
        health.nonlinear_updates <=
            request.numerical_policy.maximum_nonlinear_updates &&
        health.trace_solves >= 1 &&
        std::isfinite(health.initial_residual) &&
        std::isfinite(health.final_residual) &&
        health.final_residual <=
            request.numerical_policy.nonlinear_relative_tolerance *
                (1.0 + health.initial_residual) &&
        health.true_relative_residual <=
            request.numerical_policy.true_residual_relative_tolerance &&
        health.maximum_conservation_relative_error <=
            request.numerical_policy.conservation_relative_tolerance;
    if (!healthy || !session.AcknowledgePreparedOutput(authority, output) ||
        session.Commit(std::move(authority), output).status !=
            dg::hdg::generic::HdgEulerSessionStatus::committed)
      return trajectory;
    StepState state;
    if (!CaptureCudaState(session, fields.scalar_count, state))
      return trajectory;
    state.initial_residual = health.initial_residual;
    state.final_residual = health.final_residual;
    state.true_relative_residual = health.true_relative_residual;
    state.maximum_conservation_error = health.maximum_conservation_error;
    state.maximum_conservation_relative_error =
        health.maximum_conservation_relative_error;
    state.nonlinear_updates = health.nonlinear_updates;
    state.trace_solves = health.trace_solves;
    trajectory.states.push_back(std::move(state));
  }
  trajectory.passed = trajectory.states.size() ==
      static_cast<std::size_t>(specification.steps + 1);
  return trajectory;
}

bool CompareTrajectory(const Case& specification,
                       const Trajectory& cpu, const Trajectory& cuda)
{
  if (!cpu.passed || !cuda.passed ||
      cpu.states.size() != cuda.states.size())
    return false;
  bool passed = true;
  for (std::size_t index = 0; index < cpu.states.size(); ++index)
  {
    const StepState& reference = cpu.states[index];
    const StepState& device = cuda.states[index];
    const ComparisonMetrics interior = Compare(
        reference.physical_interior, device.physical_interior);
    const ComparisonMetrics trace = Compare(
        reference.true_trace, device.true_trace);
    const auto near_scalar = [](const double left, const double right)
    {
      const double scale = std::max({1.0, std::abs(left), std::abs(right)});
      return std::isfinite(left) && std::isfinite(right) &&
          std::abs(left - right) <=
              comparison_absolute_tolerance +
              comparison_relative_tolerance * scale;
    };
    // Both backends independently enforce the same 1e-10 acceptance policy.
    // Their ordered local accumulation is identical, but MPI and host/device
    // FP64 instruction ordering may differ; 4096 eps isolates that justified
    // arithmetic freedom without weakening either backend's gate.
    const auto near_conservation = [](const double left, const double right)
    {
      constexpr double tolerance = 4096.0 *
          std::numeric_limits<double>::epsilon();
      return std::isfinite(left) && std::isfinite(right) &&
          std::abs(left - right) <= tolerance;
    };
    const bool state_passed = interior.passed && trace.passed &&
        reference.trace_components == device.trace_components &&
        reference.step == device.step &&
        reference.generation == device.generation &&
        reference.time == device.time &&
        near_scalar(reference.maximum_wave_speed,
                    device.maximum_wave_speed) &&
        near_scalar(reference.backend_stable_dt,
                    device.backend_stable_dt) &&
        near_conservation(reference.maximum_conservation_error,
                          device.maximum_conservation_error) &&
        near_conservation(
            reference.maximum_conservation_relative_error,
            device.maximum_conservation_relative_error);
    passed = state_passed && passed;
    if (!state_passed)
    {
      std::cerr << "cross-backend mismatch case=" << specification.name
                << " state=" << index
                << " interior_rel=" << interior.relative_l2
                << " interior_inf=" << interior.maximum_absolute
                << " trace_rel=" << trace.relative_l2
                << " trace_inf=" << trace.maximum_absolute
                << " trace_components=" << reference.trace_components
                << '/' << device.trace_components
                << " wave=" << reference.maximum_wave_speed
                << '/' << device.maximum_wave_speed
                << " stable_dt=" << reference.backend_stable_dt
                << '/' << device.backend_stable_dt
                << " conservation="
                << reference.maximum_conservation_error << '/'
                << device.maximum_conservation_error
                << " conservation_relative="
                << reference.maximum_conservation_relative_error << '/'
                << device.maximum_conservation_relative_error << '\n';
    }
  }
  return passed;
}

Trajectory RunCase(const Case& specification, const BackendKind backend)
{
  mfem::ParMesh mesh = MakeMesh(specification.physical);
  BorrowedFields fields = MakeFields(mesh, specification.initial_mode);
  if (fields.initial.size() != fields.scalar_count * components ||
      !PhysicalCoefficients(fields.initial, fields.scalar_count) ||
      (specification.initial_mode == InitialMode::stationary_contact &&
       !HasContactJump(fields)))
    return {};
  BoundaryContext boundary{};
  const EulerHdgRequest request = MakeRequest(
      mesh, fields, boundary, specification, backend);
  return backend == BackendKind::cpu
      ? RunCpu(request, fields, specification)
      : RunCuda(request, fields, specification);
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks != 1)
  {
    std::cerr << "CPU/CUDA numerical-alignment qualification is rank-one; "
                 "rank-two correctness has separate generic coverage\n";
    return 2;
  }
  const std::array<Case, 4> cases{{
      {"conventional-smooth-one-step", InitialMode::smooth_periodic,
       RegularizationKind::none, false, 1, 1.0e-5},
      {"igr-smooth-short-trajectory", InitialMode::smooth_periodic,
       RegularizationKind::igr, false, 3, 1.0e-5},
      {"igr-periodic-contact", InitialMode::stationary_contact,
       RegularizationKind::igr, false, 2, 1.0e-5},
      {"igr-slip-boundary-contact", InitialMode::stationary_contact,
       RegularizationKind::igr, true, 2, 1.0e-5}}};

  // MFEM and HYPRE device selection is process-global.  Form every CPU
  // reference trajectory under the default host policy first, then select
  // CUDA once and form all device trajectories.  Interleaving the policies
  // in one process would make backend agreement depend on global setup order.
  std::array<Trajectory, 4> cpu_trajectories{};
  for (std::size_t index = 0; index < cases.size(); ++index)
    cpu_trajectories[index] = RunCase(cases[index], BackendKind::cpu);
  bool cpu_phase_ready = true;
  for (const Trajectory& trajectory : cpu_trajectories)
    cpu_phase_ready = trajectory.passed && cpu_phase_ready;
  if (!All(cpu_phase_ready))
  {
    std::cerr << "CPU reference trajectory construction failed\n";
    return 1;
  }

  // A successful host solve may leave HYPRE's process-global convergence bit
  // set even though the independently checked residual gates passed.  Clear
  // that library-global state at this explicit test-only backend boundary;
  // ordinary applications select exactly one backend per process.
  const bool hypre_transition_clean = HYPRE_ClearAllErrors() == 0 &&
      HYPRE_GetError() == 0;
  if (!All(hypre_transition_clean))
  {
    std::cerr << "HYPRE host-to-CUDA test transition failed\n";
    return 1;
  }

  mfem::Device device("cuda");
  bool passed = true;
  for (std::size_t index = 0; index < cases.size(); ++index)
  {
    const Trajectory cuda = RunCase(cases[index], BackendKind::cuda);
    const bool case_passed = CompareTrajectory(
        cases[index], cpu_trajectories[index], cuda);
    if (!case_passed)
    {
      std::cerr << "cross-backend qualification failed case="
                << cases[index].name << " cpu_states="
                << cpu_trajectories[index].states.size()
                << " cuda_states=" << cuda.states.size() << '\n';
    }
    passed = case_passed && passed;
  }
  passed = All(passed);
  if (passed)
    std::cout << "HDG CPU/CUDA numerical alignment and IGR discontinuity "
                 "qualification passed\n";
  return passed ? 0 : 1;
}

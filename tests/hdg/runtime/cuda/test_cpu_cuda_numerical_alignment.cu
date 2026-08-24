#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
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

constexpr bool CudaPreparedIluAccountingValid(
    const CudaTensorCoreEvidence& tensor, const int trace_solves,
    const int linear_iterations, const generic::CudaEngineKind engine,
    const bool igr_active) noexcept
{
  const auto preconditioner =
      CudaTracePreconditionerForEngine(engine, igr_active);
  const bool ilu = CudaTracePreconditionerIsIlu(preconditioner);
  const auto factor_reuse_policy =
      CudaIluFactorReusePolicyForEngine(engine, igr_active);
  const auto triangular_solve_kind =
      CudaIluTriangularSolveKindForEngine(engine, igr_active);
  const int lower_jacobi_iterations =
      CudaIluLowerJacobiIterations(engine, igr_active);
  const int upper_jacobi_iterations =
      CudaIluUpperJacobiIterations(engine, igr_active);
  if (!ilu)
    return preconditioner == CudaTracePreconditionerKind::hypre_diag_scale &&
        tensor.ilu_type == -1 && tensor.ilu_level_of_fill == -1 &&
        tensor.ilu_local_reordering == -1 &&
        tensor.ilu_application_iterations == 0 &&
        triangular_solve_kind ==
            CudaIluTriangularSolveKind::not_applicable &&
        tensor.ilu_triangular_solve_kind == triangular_solve_kind &&
        tensor.ilu_lower_jacobi_iterations == 0 &&
        tensor.ilu_upper_jacobi_iterations == 0 &&
        tensor.ilu_accounting_kind == CudaIluAccountingKind::unavailable &&
        factor_reuse_policy == CudaIluFactorReusePolicy::not_applicable &&
        tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
        tensor.ilu_factor_setups == 0 &&
        tensor.ilu_factor_reuses == 0 &&
        tensor.ilu_preconditioner_callbacks_lower_bound == 0 &&
        tensor.ilu_preconditioner_callbacks_upper_bound == 0 &&
        tensor.ilu_inner_iterations_lower_bound == 0 &&
        tensor.ilu_inner_iterations_upper_bound == 0 &&
        tensor.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
        tensor.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
        tensor.ilu_factor_age_generations == 0 &&
        tensor.ilu_accounting_complete;
  const int expected_fill = 0;
  const int expected_reordering = igr_active ? 1 : 0;
  if (trace_solves <= 0 || linear_iterations < 0 || tensor.ilu_type != 0 ||
      tensor.ilu_level_of_fill != expected_fill ||
      tensor.ilu_local_reordering != expected_reordering ||
      tensor.ilu_application_iterations != 1 ||
      tensor.ilu_triangular_solve_kind != triangular_solve_kind ||
      tensor.ilu_lower_jacobi_iterations != lower_jacobi_iterations ||
      tensor.ilu_upper_jacobi_iterations != upper_jacobi_iterations ||
      (triangular_solve_kind == CudaIluTriangularSolveKind::direct &&
       (lower_jacobi_iterations != 0 || upper_jacobi_iterations != 0)) ||
      (triangular_solve_kind ==
           CudaIluTriangularSolveKind::iterative_jacobi &&
       (lower_jacobi_iterations != 5 || upper_jacobi_iterations != 5)) ||
      tensor.ilu_accounting_kind !=
          CudaIluAccountingKind::derived_bounds_v1 ||
      !tensor.ilu_accounting_complete ||
      tensor.ilu_preconditioner_callbacks_lower_bound >
          tensor.ilu_preconditioner_callbacks_upper_bound)
    return false;
  const std::uint64_t solves = static_cast<std::uint64_t>(trace_solves);
  const bool factor_work = factor_reuse_policy ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve
      ? tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
            tensor.ilu_factor_setups == solves &&
            tensor.ilu_factor_reuses == 0 &&
            tensor.ilu_factor_age_generations == 0
      : factor_reuse_policy ==
              CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
            tensor.ilu_factor_reuse_policy == factor_reuse_policy &&
            tensor.ilu_factor_setups == 1 &&
            tensor.ilu_factor_reuses == solves - 1 &&
            tensor.ilu_factor_age_generations == solves - 1;
  if (!factor_work) return false;
  const std::uint64_t iterations =
      static_cast<std::uint64_t>(linear_iterations);
  const std::uint64_t lower =
      tensor.ilu_preconditioner_callbacks_lower_bound;
  const std::uint64_t upper =
      tensor.ilu_preconditioner_callbacks_upper_bound;
  const bool callback_bounds = iterations == 0
      ? lower == 0 && upper == 0
      : iterations <= std::numeric_limits<std::uint64_t>::max()/2 &&
            iterations < lower && upper == 2*iterations;
  const std::uint64_t multiplier = static_cast<std::uint64_t>(
      tensor.ilu_application_iterations);
  const std::uint64_t factor_spmv_multiplier =
      triangular_solve_kind == CudaIluTriangularSolveKind::iterative_jacobi
      ? 8 : 0;
  return callback_bounds &&
      lower <= std::numeric_limits<std::uint64_t>::max()/multiplier &&
      upper <= std::numeric_limits<std::uint64_t>::max()/multiplier &&
      tensor.ilu_inner_iterations_lower_bound == lower*multiplier &&
      tensor.ilu_inner_iterations_upper_bound == upper*multiplier &&
      (factor_spmv_multiplier == 0 ||
       (tensor.ilu_inner_iterations_lower_bound <=
            std::numeric_limits<std::uint64_t>::max()/
                factor_spmv_multiplier &&
        tensor.ilu_inner_iterations_upper_bound <=
            std::numeric_limits<std::uint64_t>::max()/
                factor_spmv_multiplier)) &&
      tensor.ilu_jacobi_factor_spmv_updates_lower_bound ==
          tensor.ilu_inner_iterations_lower_bound*factor_spmv_multiplier &&
      tensor.ilu_jacobi_factor_spmv_updates_upper_bound ==
          tensor.ilu_inner_iterations_upper_bound*factor_spmv_multiplier;
}

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
  int linear_iterations = -1;
  int line_search_trials = -1;
  int rejected_line_search_trials = -1;
  CudaTensorCoreEvidence tensor_core{};
  bool finite = false;
  bool admissible = false;
  bool recovery = false;
  bool true_residual = false;
  bool conservation = false;
  bool no_fallback = false;
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
    health.linear_iterations = numerical.linear_iterations;
    health.line_search_trials = numerical.line_search_trials;
    health.rejected_line_search_trials =
        numerical.rejected_line_search_trials;
    health.tensor_core = pending.evidence.tensor_core;
    health.finite = numerical.finite;
    health.admissible = numerical.admissible;
    health.recovery = numerical.recovery;
    health.true_residual = numerical.true_residual;
    health.conservation = numerical.conservation;
    health.no_fallback = pending.evidence.no_fallback;
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
using dg::hdg::generic::CudaEngineKind;
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
using dg::hdg::runtime::cuda_w8a::detail::CrossBackendPreparedHealth;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;

constexpr int components = dg::hdg::generic::euler_2d_conserved_components;
constexpr double comparison_relative_tolerance = 5.0e-8;
constexpr double comparison_absolute_tolerance = 5.0e-9;
constexpr double sigma_relative_tolerance = 5.0e-6;
constexpr double sigma_absolute_tolerance = 8192.0 *
    std::numeric_limits<double>::epsilon();
constexpr double igr_transition_x = 0.4;
constexpr double igr_transition_width = 0.06;
constexpr double igr_localization_half_width = 0.25;
constexpr double igr_minimum_near_far_energy_ratio = 1.5;
constexpr double igr_minimum_near_energy_fraction = 0.60;

constexpr std::uint32_t common_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

enum class InitialMode
{
  smooth_periodic,
  localized_igr_contact
};

struct Case
{
  std::string_view name;
  InitialMode initial_mode = InitialMode::smooth_periodic;
  RegularizationKind regularization = RegularizationKind::none;
  bool physical = false;
  int steps = 1;
  double dt = 1.0e-5;
  bool require_localized_sigma = false;
  double static_source_scale = 0.0;
};

struct BoundaryContext
{
  std::array<EulerBoundaryPolicy, 4> policies{};
  std::uint64_t calls = 0;
};

struct BorrowedFields
{
  std::vector<double> initial;
  std::vector<double> source;
  std::size_t scalar_count = 0;
};

struct ComparisonMetrics
{
  double relative_l2 = std::numeric_limits<double>::infinity();
  double maximum_absolute = std::numeric_limits<double>::infinity();
  double scale = 1.0;
  bool passed = false;
};

struct SigmaLocalization
{
  double global_l2_squared = 0.0;
  double global_maximum_absolute = 0.0;
  double near_mean_square = 0.0;
  double far_mean_square = 0.0;
  double near_energy_fraction = 0.0;
  bool available = false;
  bool finite = false;
  bool nonzero = false;
  bool localized = false;
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
  int linear_iterations = 0;
  int line_search_trials = 0;
  int rejected_line_search_trials = 0;
  std::uint64_t tensor_condensation_launches = 0;
  std::uint64_t tensor_mma_tiles = 0;
  std::uint64_t fallback_count = 0;
  int trace_components = 0;
  SigmaLocalization sigma{};
};

struct Trajectory
{
  std::vector<StepState> states;
  bool passed = false;
};

bool All(const bool local, const MPI_Comm communicator = MPI_COMM_WORLD)
{
  const int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && output == 1;
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
  if (mode == InitialMode::localized_igr_contact)
  {
    // Keep the density contact discontinuous, but move it off the 4x4 mesh
    // lines and superpose a small resolved normal-velocity transition.  The
    // old mesh-aligned zero-velocity contact had identically zero IGR q
    // forcing and therefore could not qualify localized regularization.
    const double density = point[0] < igr_transition_x ? 1.0 : 1.2;
    const double velocity_x = 0.02 * std::tanh(
        (point[0] - igr_transition_x) / igr_transition_width);
    constexpr double pressure = 1.0;
    state[0] = density;
    state[1] = density * velocity_x;
    state[2] = 0.0;
    state[3] = pressure / 0.4 +
        0.5 * density * velocity_x * velocity_x;
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

BorrowedFields MakeFields(mfem::ParMesh& mesh, const InitialMode mode,
                          const int order, const double source_scale)
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
  if (source_scale != 0.0 && !fields.initial.empty())
  {
    fields.source.resize(fields.initial.size(), 0.0);
    constexpr std::array<double, components> source_components{
        1.0e-3, 5.0e-4, -2.5e-4, 1.5e-3};
    for (int component = 0; component < components; ++component)
      for (std::size_t scalar = 0; scalar < fields.scalar_count; ++scalar)
        fields.source[static_cast<std::size_t>(component) *
                          fields.scalar_count + scalar] =
            source_scale * source_components[component];
  }
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
    const BackendKind backend, const int order,
    const CudaEngineKind cuda_engine)
{
  ConfigureBoundary(boundary);
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {backend, backend == BackendKind::cuda ? 0 : -1,
                     backend == BackendKind::cuda
                         ? cuda_engine : CudaEngineKind::ordinary};
  request.capabilities = common_capabilities | CapabilityBit(
      backend == BackendKind::cuda ? Capability::cuda : Capability::cpu);
  request.initial_state = {
      fields.initial.data(), fields.initial.size(), components};
  // Deliberately exercise the common projected-trace path.  Supplied traces
  // already have independent p1-p3 coverage in the generic extension suite.
  request.initial_true_trace = {nullptr, 0, components};
  request.source = {
      fields.source.empty() ? nullptr : fields.source.data(),
      fields.source.size(), components};
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

bool ComputeSigmaLocalization(mfem::ParMesh& mesh, const int order,
                              const std::vector<double>& sigma,
                              SigmaLocalization& localization)
{
  localization = {};
  std::array<double, 5> local{};
  double local_maximum = 0.0;
  const bool local_ready = [&]() noexcept
  {
    try
    {
      mfem::L2_FECollection collection(
          order, 2, mfem::BasisType::GaussLegendre,
          mfem::FiniteElement::VALUE);
      mfem::ParFiniteElementSpace space(
          &mesh, &collection, 1,
          mfem::Ordering::byNODES);
      if (space.GetVSize() < 0 ||
          sigma.size() != static_cast<std::size_t>(space.GetVSize()))
        return false;
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
        const mfem::FiniteElement* finite_element = space.GetFE(element);
        mfem::ElementTransformation* transformation =
            mesh.GetElementTransformation(element);
        mfem::Array<int> vdofs;
        space.GetElementVDofs(element, vdofs);
        if (finite_element == nullptr || transformation == nullptr ||
            finite_element->GetDof() <= 0 ||
            vdofs.Size() != finite_element->GetDof())
          return false;
        const mfem::IntegrationRule& rule = mfem::IntRules.Get(
            finite_element->GetGeomType(), 2 * order + 4);
        mfem::Vector shape(finite_element->GetDof());
        mfem::Vector physical(mesh.SpaceDimension());
        for (int quadrature = 0; quadrature < rule.GetNPoints(); ++quadrature)
        {
          const mfem::IntegrationPoint& point = rule.IntPoint(quadrature);
          finite_element->CalcShape(point, shape);
          transformation->SetIntPoint(&point);
          transformation->Transform(point, physical);
          const double weight = point.weight * transformation->Weight();
          if (!(weight > 0.0) || !std::isfinite(weight) ||
              physical.Size() < 1 || !std::isfinite(physical[0]))
            return false;
          double value = 0.0;
          for (int scalar = 0; scalar < finite_element->GetDof(); ++scalar)
          {
            double sign = 1.0;
            int dof = vdofs[scalar];
            if (dof < 0)
            {
              dof = -1 - dof;
              sign = -1.0;
            }
            if (dof < 0 || dof >= space.GetVSize() ||
                !std::isfinite(sigma[static_cast<std::size_t>(dof)]))
              return false;
            value += sign * shape[scalar] *
                sigma[static_cast<std::size_t>(dof)];
          }
          if (!std::isfinite(value)) return false;
          const double energy = weight * value * value;
          const bool near = std::abs(physical[0] - igr_transition_x) <=
              igr_localization_half_width;
          local[0] += energy;
          local[near ? 1 : 2] += energy;
          local[near ? 3 : 4] += weight;
          local_maximum = std::max(
              local_maximum, static_cast<double>(std::abs(value)));
        }
      }
      return std::all_of(local.begin(), local.end(),
                         [](const double value)
                         { return std::isfinite(value) && value >= 0.0; }) &&
          std::isfinite(local_maximum);
    }
    catch (...) { return false; }
  }();
  if (!All(local_ready, mesh.GetComm())) return false;

  std::array<double, 5> global{};
  double global_maximum = 0.0;
  const bool energy_reduced = MPI_Allreduce(
      local.data(), global.data(), static_cast<int>(global.size()),
      MPI_DOUBLE, MPI_SUM, mesh.GetComm()) == MPI_SUCCESS;
  const bool maximum_reduced = MPI_Allreduce(
      &local_maximum, &global_maximum, 1, MPI_DOUBLE, MPI_MAX,
      mesh.GetComm()) == MPI_SUCCESS;
  if (!All(energy_reduced && maximum_reduced, mesh.GetComm())) return false;
  const double total_measure = global[3] + global[4];
  if (!(total_measure > 0.0) || !(global[3] > 0.0) ||
      !(global[4] > 0.0) || !std::isfinite(total_measure) ||
      !std::isfinite(global_maximum))
    return false;

  const double near_mean_square = global[1] / global[3];
  const double far_mean_square = global[2] / global[4];
  const double near_energy_fraction = global[0] > 0.0
      ? global[1] / global[0] : 0.0;
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double energy_floor =
      4096.0 * epsilon * epsilon * total_measure;
  localization.global_l2_squared = global[0];
  localization.global_maximum_absolute = global_maximum;
  localization.near_mean_square = near_mean_square;
  localization.far_mean_square = far_mean_square;
  localization.near_energy_fraction = near_energy_fraction;
  localization.available = true;
  localization.finite = std::isfinite(localization.global_l2_squared) &&
      std::isfinite(localization.global_maximum_absolute) &&
      std::isfinite(localization.near_mean_square) &&
      std::isfinite(localization.far_mean_square) &&
      std::isfinite(localization.near_energy_fraction);
  localization.nonzero = localization.finite && global[0] > energy_floor &&
      global_maximum > 64.0 * std::numeric_limits<double>::epsilon();
  const double near_far_ratio = far_mean_square > 0.0
      ? near_mean_square / far_mean_square
      : std::numeric_limits<double>::infinity();
  // The p1-p3 control observations all had ratios >= 1.82 and near-energy
  // fractions >= 0.645.  Retain diagnosed margin while requiring meaningful
  // concentration instead of a numerical inequality at roundoff scale.
  localization.localized = localization.nonzero &&
      near_far_ratio >= igr_minimum_near_far_energy_ratio &&
      near_energy_fraction >= igr_minimum_near_energy_fraction;
  return localization.finite;
}

bool HasIgrContactAndVelocityTransition(const BorrowedFields& fields)
{
  if (fields.scalar_count == 0 ||
      fields.initial.size() != fields.scalar_count * components)
    return false;
  const auto density_bounds = std::minmax_element(
      fields.initial.begin(),
      fields.initial.begin() + static_cast<std::ptrdiff_t>(
          fields.scalar_count));
  double minimum_velocity = std::numeric_limits<double>::infinity();
  double maximum_velocity = -std::numeric_limits<double>::infinity();
  for (std::size_t scalar = 0; scalar < fields.scalar_count; ++scalar)
  {
    const double density = fields.initial[scalar];
    const double velocity =
        fields.initial[fields.scalar_count + scalar] / density;
    minimum_velocity = std::min(minimum_velocity, velocity);
    maximum_velocity = std::max(maximum_velocity, velocity);
  }
  return *density_bounds.second - *density_bounds.first >= 0.19 &&
      std::isfinite(minimum_velocity) && std::isfinite(maximum_velocity) &&
      maximum_velocity - minimum_velocity >= 0.02;
}

bool CaptureCpuState(CpuNumericsCore& core,
                     const StateCoordinates& coordinates,
                     mfem::ParMesh& mesh, const int order,
                     const std::size_t scalar_count,
                     const RegularizationKind regularization,
                     StepState& state)
{
  state.physical_interior.assign(scalar_count * components, 0.0);
  std::vector<double> sigma(
      regularization == RegularizationKind::igr ? scalar_count : 0, 0.0);
  const auto snapshot = core.Snapshot(coordinates);
  if (!core.ExportCommittedFields({
          state.physical_interior.data(),
          sigma.empty() ? nullptr : sigma.data(), scalar_count}) ||
      snapshot.trace.data == nullptr || snapshot.trace.size == 0)
    return false;
  state.true_trace.assign(
      snapshot.trace.data, snapshot.trace.data + snapshot.trace.size);
  state.time = coordinates.time;
  state.step = coordinates.step;
  state.generation = coordinates.generation;
  return core.CommittedBorrowedEulerDiagnostics(
             state.maximum_wave_speed, state.backend_stable_dt) &&
      PhysicalCoefficients(state.physical_interior, scalar_count) &&
      (regularization != RegularizationKind::igr ||
       ComputeSigmaLocalization(mesh, order, sigma, state.sigma));
}

Trajectory RunCpu(const EulerHdgRequest& request,
                  mfem::ParMesh& mesh, const BorrowedFields& fields,
                  const Case& specification)
{
  Trajectory trajectory;
  CpuNumericsCore core;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
          ContractStatus::accepted ||
      !core.InitializeBorrowedEuler(request))
    return trajectory;
  StateCoordinates coordinates{};
  StepState initial;
  if (!CaptureCpuState(core, coordinates, mesh, request.polynomial_order,
                       fields.scalar_count, specification.regularization,
                       initial))
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
        !CaptureCpuState(core, coordinates, mesh, request.polynomial_order,
                         fields.scalar_count, specification.regularization,
                         state))
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
    state.linear_iterations = attempt.linear_iterations;
    state.line_search_trials = static_cast<int>(
        attempt.work.trial_reassemblies);
    state.rejected_line_search_trials = static_cast<int>(
        attempt.work.trial_reassemblies >=
                attempt.work.accepted_newton_updates
            ? attempt.work.trial_reassemblies -
                  attempt.work.accepted_newton_updates
            : 0);
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
                      mfem::ParMesh& mesh, const int order,
                      const std::size_t scalar_count,
                      const RegularizationKind regularization,
                      StepState& state)
{
  state.physical_interior.assign(scalar_count * components, 0.0);
  std::vector<double> sigma(
      regularization == RegularizationKind::igr ? scalar_count : 0, 0.0);
  if (!session.ExportCommittedFields({
          state.physical_interior.data(),
          sigma.empty() ? nullptr : sigma.data(), scalar_count}) ||
      !CudaQualificationTestAccess::ExportCommittedTrace(
          session, state.true_trace, state.trace_components) ||
      !PhysicalCoefficients(state.physical_interior, scalar_count) ||
      (regularization == RegularizationKind::igr &&
       !ComputeSigmaLocalization(mesh, order, sigma, state.sigma)))
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
                   mfem::ParMesh& mesh, const BorrowedFields& fields,
                   const Case& specification)
{
  Trajectory trajectory;
  GenericEulerHdgCudaSession session;
  if (dg::hdg::generic::ValidateEulerHdgRequest(request) !=
          ContractStatus::accepted ||
      session.Initialize(request).status !=
          dg::hdg::generic::HdgEulerSessionStatus::initialized)
    return trajectory;
  StepState initial;
  if (!CaptureCudaState(session, mesh, request.polynomial_order,
                        fields.scalar_count, specification.regularization,
                        initial))
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
    const bool expected_engine =
        health.tensor_core.selected_engine == request.backend.cuda_engine &&
        health.tensor_core.trace_preconditioner_configured &&
        health.tensor_core.trace_preconditioner ==
            dg::hdg::runtime::cuda_w8a::CudaTracePreconditionerForEngine(
                request.backend.cuda_engine,
                specification.regularization == RegularizationKind::igr) &&
        dg::hdg::runtime::cuda_w8a::detail::
            CudaPreparedIluAccountingValid(
                health.tensor_core, health.trace_solves,
                health.linear_iterations, request.backend.cuda_engine,
                specification.regularization == RegularizationKind::igr) &&
        health.tensor_core.fallback_count == 0 && health.no_fallback &&
        (request.backend.cuda_engine == CudaEngineKind::ordinary
             ? health.tensor_core.condensation_launches == 0 &&
                   health.tensor_core.mma_tiles == 0
             : health.tensor_core.hardware_supported &&
                   health.tensor_core.condensation_launches > 0 &&
                   health.tensor_core.mma_tiles > 0);
    const bool healthy = captured && health.success && health.finite &&
        health.admissible && health.recovery && health.true_residual &&
        health.conservation && expected_engine &&
        health.nonlinear_updates >= 0 &&
        health.nonlinear_updates <=
            request.numerical_policy.maximum_nonlinear_updates &&
        health.trace_solves >= 1 &&
        health.linear_iterations >= 0 && health.line_search_trials >= 0 &&
        health.rejected_line_search_trials >= 0 &&
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
    if (!CaptureCudaState(session, mesh, request.polynomial_order,
                          fields.scalar_count, specification.regularization,
                          state))
      return trajectory;
    state.initial_residual = health.initial_residual;
    state.final_residual = health.final_residual;
    state.true_relative_residual = health.true_relative_residual;
    state.maximum_conservation_error = health.maximum_conservation_error;
    state.maximum_conservation_relative_error =
        health.maximum_conservation_relative_error;
    state.nonlinear_updates = health.nonlinear_updates;
    state.trace_solves = health.trace_solves;
    state.linear_iterations = health.linear_iterations;
    state.line_search_trials = health.line_search_trials;
    state.rejected_line_search_trials =
        health.rejected_line_search_trials;
    state.tensor_condensation_launches =
        health.tensor_core.condensation_launches;
    state.tensor_mma_tiles = health.tensor_core.mma_tiles;
    state.fallback_count = health.tensor_core.fallback_count;
    trajectory.states.push_back(std::move(state));
  }
  trajectory.passed = trajectory.states.size() ==
      static_cast<std::size_t>(specification.steps + 1);
  return trajectory;
}

bool CompareTrajectory(const Case& specification,
                       const std::string_view comparison,
                       const Trajectory& left, const Trajectory& right)
{
  if (!left.passed || !right.passed ||
      left.states.size() != right.states.size())
    return false;
  bool passed = true;
  for (std::size_t index = 0; index < left.states.size(); ++index)
  {
    const StepState& reference = left.states[index];
    const StepState& device = right.states[index];
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
    const auto near_sigma = [](const double left, const double right)
    {
      const double scale = std::max(std::abs(left), std::abs(right));
      return std::isfinite(left) && std::isfinite(right) &&
          std::abs(left - right) <= sigma_absolute_tolerance +
              sigma_relative_tolerance * scale;
    };
    // Both backends independently enforce the same 1e-10 acceptance policy.
    // CPU accumulates in element/quadrature order while CUDA uses a fixed
    // block/site reduction tree; MPI and host/device FP64 ordering may also
    // differ.  The existing 4096-epsilon comparison isolates that justified
    // reduction freedom without weakening either backend's gate.
    const auto near_conservation = [](const double left, const double right)
    {
      constexpr double tolerance = 4096.0 *
          std::numeric_limits<double>::epsilon();
      return std::isfinite(left) && std::isfinite(right) &&
          std::abs(left - right) <= tolerance;
    };
    const bool igr_sigma = specification.regularization ==
        RegularizationKind::igr;
    const bool sigma_metrics = !igr_sigma ||
        (reference.sigma.available && device.sigma.available &&
         reference.sigma.finite && device.sigma.finite &&
         near_sigma(reference.sigma.global_l2_squared,
                    device.sigma.global_l2_squared) &&
         near_sigma(reference.sigma.global_maximum_absolute,
                    device.sigma.global_maximum_absolute) &&
         near_sigma(reference.sigma.near_mean_square,
                    device.sigma.near_mean_square) &&
         near_sigma(reference.sigma.far_mean_square,
                    device.sigma.far_mean_square) &&
         near_sigma(reference.sigma.near_energy_fraction,
                    device.sigma.near_energy_fraction));
    // Sigma is neutral at generation zero.  Every accepted state of the two
    // discontinuous IGR cases must thereafter carry finite nonzero energy
    // concentrated near the declared off-grid velocity transition.  These
    // are global quadrature integrals, so no coefficient or rank ordering is
    // part of the qualification.
    const bool require_localization =
        specification.require_localized_sigma && index != 0;
    const bool sigma_localization = !require_localization ||
        (reference.sigma.nonzero && reference.sigma.localized &&
         device.sigma.nonzero && device.sigma.localized);
    const bool state_passed = interior.passed && trace.passed &&
        sigma_metrics && sigma_localization &&
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
      std::cerr << "cross-engine mismatch comparison=" << comparison
                << " case=" << specification.name
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
                << device.maximum_conservation_relative_error
                << " residuals=" << reference.initial_residual << '/'
                << device.initial_residual << ','
                << reference.final_residual << '/'
                << device.final_residual << ','
                << reference.true_relative_residual << '/'
                << device.true_relative_residual
                << " work=" << reference.nonlinear_updates << '/'
                << device.nonlinear_updates << ','
                << reference.linear_iterations << '/'
                << device.linear_iterations << ','
                << reference.line_search_trials << '/'
                << device.line_search_trials << " tensor="
                << reference.tensor_condensation_launches << '/'
                << device.tensor_condensation_launches << ','
                << reference.tensor_mma_tiles << '/'
                << device.tensor_mma_tiles << " fallback="
                << reference.fallback_count << '/'
                << device.fallback_count << " sigma_l2="
                << reference.sigma.global_l2_squared << '/'
                << device.sigma.global_l2_squared << " sigma_max="
                << reference.sigma.global_maximum_absolute << '/'
                << device.sigma.global_maximum_absolute << " sigma_near="
                << reference.sigma.near_mean_square << '/'
                << device.sigma.near_mean_square << " sigma_far="
                << reference.sigma.far_mean_square << '/'
                << device.sigma.far_mean_square << " sigma_fraction="
                << reference.sigma.near_energy_fraction << '/'
                << device.sigma.near_energy_fraction << " sigma_flags="
                << reference.sigma.available << reference.sigma.finite
                << reference.sigma.nonzero << reference.sigma.localized << '/'
                << device.sigma.available << device.sigma.finite
                << device.sigma.nonzero << device.sigma.localized << '\n';
    }
  }
  return passed;
}

Trajectory RunCase(const Case& specification, const BackendKind backend,
                   const int order,
                   const CudaEngineKind cuda_engine =
                       CudaEngineKind::ordinary)
{
  mfem::ParMesh mesh = MakeMesh(specification.physical);
  BorrowedFields fields = MakeFields(
      mesh, specification.initial_mode, order,
      specification.static_source_scale);
  if (fields.initial.size() != fields.scalar_count * components ||
      !PhysicalCoefficients(fields.initial, fields.scalar_count) ||
      (specification.initial_mode == InitialMode::localized_igr_contact &&
       !HasIgrContactAndVelocityTransition(fields)))
    return {};
  BoundaryContext boundary{};
  const EulerHdgRequest request = MakeRequest(
      mesh, fields, boundary, specification, backend, order, cuda_engine);
  return backend == BackendKind::cpu
      ? RunCpu(request, mesh, fields, specification)
      : RunCuda(request, mesh, fields, specification);
}

bool ParseOrder(const int argc, char* argv[], int& order) noexcept
{
  order = 1;
  if (argc == 1) return true;
  if (argc != 3 || std::string_view(argv[1]) != "--order") return false;
  int parsed = 0;
  const std::string_view value(argv[2]);
  const auto result = std::from_chars(
      value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} ||
      result.ptr != value.data() + value.size() || parsed < 1 || parsed > 3)
    return false;
  order = parsed;
  return true;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int order = 1;
  if (!ParseOrder(argc, argv, order))
  {
    std::cerr << "Usage: " << argv[0] << " [--order 1|2|3]\n";
    return 2;
  }
  if (ranks != 1)
  {
    std::cerr << "CPU/CUDA numerical-alignment qualification is rank-one; "
                 "rank-two correctness has separate generic coverage\n";
    return 2;
  }
  const std::array<Case, 5> cases{{
      {"conventional-smooth-short-trajectory", InitialMode::smooth_periodic,
       RegularizationKind::none, false, 3, 1.0e-5, false, 0.0},
      {"conventional-smooth-slip-source-trajectory",
       InitialMode::smooth_periodic, RegularizationKind::none, true, 2,
       1.0e-5, false, 1.0},
      {"igr-smooth-short-trajectory", InitialMode::smooth_periodic,
       RegularizationKind::igr, false, 3, 1.0e-5, false, 0.0},
      {"igr-periodic-localized-contact", InitialMode::localized_igr_contact,
       RegularizationKind::igr, false, 2, 1.0e-5, true, 0.0},
      {"igr-slip-boundary-localized-contact",
       InitialMode::localized_igr_contact,
       RegularizationKind::igr, true, 2, 1.0e-5, true, 0.0}}};

  // MFEM and HYPRE device selection is process-global.  Form every CPU
  // reference trajectory under the default host policy first, then select
  // CUDA once and form all device trajectories.  Interleaving the policies
  // in one process would make backend agreement depend on global setup order.
  std::array<Trajectory, 5> cpu_trajectories{};
  for (std::size_t index = 0; index < cases.size(); ++index)
    cpu_trajectories[index] = RunCase(
        cases[index], BackendKind::cpu, order);
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
  std::array<Trajectory, 5> ordinary_trajectories{};
  std::array<Trajectory, 5> tensor_trajectories{};
  for (std::size_t index = 0; index < cases.size(); ++index)
    ordinary_trajectories[index] = RunCase(
        cases[index], BackendKind::cuda, order,
        CudaEngineKind::ordinary);
  const bool ordinary_phase_ready = std::all_of(
      ordinary_trajectories.begin(), ordinary_trajectories.end(),
      [](const Trajectory& trajectory) { return trajectory.passed; });
  if (!All(ordinary_phase_ready))
  {
    std::cerr << "ordinary CUDA trajectory construction failed order="
              << order << '\n';
    return 1;
  }
  const bool cuda_engine_transition_clean = HYPRE_ClearAllErrors() == 0 &&
      HYPRE_GetError() == 0;
  if (!All(cuda_engine_transition_clean))
  {
    std::cerr << "ordinary-to-tensor CUDA test transition failed\n";
    return 1;
  }
  for (std::size_t index = 0; index < cases.size(); ++index)
    tensor_trajectories[index] = RunCase(
        cases[index], BackendKind::cuda, order,
        CudaEngineKind::tensor_core);
  const bool tensor_phase_ready = std::all_of(
      tensor_trajectories.begin(), tensor_trajectories.end(),
      [](const Trajectory& trajectory) { return trajectory.passed; });
  if (!All(tensor_phase_ready))
  {
    std::cerr << "tensor CUDA trajectory construction failed order="
              << order << '\n';
    return 1;
  }

  bool passed = true;
  for (std::size_t index = 0; index < cases.size(); ++index)
  {
    const bool cpu_ordinary = CompareTrajectory(
        cases[index], "cpu-vs-ordinary", cpu_trajectories[index],
        ordinary_trajectories[index]);
    const bool cpu_tensor = CompareTrajectory(
        cases[index], "cpu-vs-tensor", cpu_trajectories[index],
        tensor_trajectories[index]);
    const bool ordinary_tensor = CompareTrajectory(
        cases[index], "ordinary-vs-tensor", ordinary_trajectories[index],
        tensor_trajectories[index]);
    const bool case_passed =
        cpu_ordinary && cpu_tensor && ordinary_tensor;
    if (!case_passed)
    {
      std::cerr << "cross-engine qualification failed case="
                << cases[index].name << " order=" << order
                << " comparisons=" << cpu_ordinary << '/'
                << cpu_tensor << '/' << ordinary_tensor
                << " cpu_states="
                << cpu_trajectories[index].states.size()
                << " ordinary_states="
                << ordinary_trajectories[index].states.size()
                << " tensor_states="
                << tensor_trajectories[index].states.size() << '\n';
    }
    passed = case_passed && passed;
  }
  passed = All(passed);
  if (passed)
    std::cout << "HDG CPU/ordinary-CUDA/tensor-CUDA numerical alignment and "
                 "IGR discontinuity qualification passed order="
              << order << '\n';
  return passed ? 0 : 1;
}

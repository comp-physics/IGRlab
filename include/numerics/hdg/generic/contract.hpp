#pragma once

/**
 * @file numerics/hdg/generic/contract.hpp
 * @brief Borrowed-input admission contract for a 2D Euler HDG engine.
 *
 * This header intentionally owns neither an MFEM mesh nor state.  It is an
 * admission boundary only; time integration, output, solver lifetime, and
 * restart are outside this contract.
 */

#include "numerics/hdg/generic/polynomial_order.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

namespace dg::hdg::generic
{

inline constexpr int euler_2d_conserved_components = 4;

enum class BackendKind : std::uint8_t
{
  cpu,
  cuda
};

/** Explicit CUDA numerical leaf selected behind the shared HDG lifecycle. */
enum class CudaEngineKind : std::uint8_t
{
  ordinary = 0,
  tensor_core = 1
};

constexpr bool IsKnownCudaEngineKind(const CudaEngineKind kind) noexcept
{
  return kind == CudaEngineKind::ordinary ||
      kind == CudaEngineKind::tensor_core;
}

/** Optional numerical augmentation selected by a borrowed Euler request. */
enum class RegularizationKind : std::uint8_t
{
  none = 0,
  igr = 1
};

/** Typed temporal architecture selected within the shared Euler lifecycle. */
enum class EulerHdgTimeTreatment : std::uint8_t
{
  // Zero preserves Backward Euler for value-initialized and legacy aggregate
  // requests while the new member remains a trailing POD extension.
  backward_euler = 0,
  partitioned_backward_euler = 1,
  rodas3 = 2
};

constexpr bool IsKnownEulerHdgTimeTreatment(
    const EulerHdgTimeTreatment treatment) noexcept
{
  return treatment == EulerHdgTimeTreatment::backward_euler ||
      treatment == EulerHdgTimeTreatment::partitioned_backward_euler ||
      treatment == EulerHdgTimeTreatment::rodas3;
}

constexpr const char* EulerHdgTimeTreatmentName(
    const EulerHdgTimeTreatment treatment) noexcept
{
  switch (treatment)
  {
  case EulerHdgTimeTreatment::backward_euler:
    return "backward-euler";
  case EulerHdgTimeTreatment::partitioned_backward_euler:
    return "partitioned-backward-euler";
  case EulerHdgTimeTreatment::rodas3:
    return "rodas3";
  }
  return "unknown";
}

/** Numerical role of one condensed true-trace linear solve. */
enum class EulerHdgLinearSolveRole : std::uint8_t
{
  newton_correction = 0,
  final_coherent = 1
};

/**
 * Trailing, zero-compatible regularization payload.  The exact object bits of
 * alpha and beta are collective request identity (not merely their values).
 */
struct RegularizationSelection
{
  RegularizationKind kind;
  double alpha;
  double beta;
};

/** Optional positivity limiter selected within the shared Euler lifecycle. */
enum class EulerHdgPositivityLimiterKind : std::uint8_t
{
  // Zero preserves the unmodified HDG state for legacy aggregate policies.
  none = 0,
  zhang_shu = 1
};

constexpr bool IsKnownEulerHdgPositivityLimiterKind(
    const EulerHdgPositivityLimiterKind kind) noexcept
{
  return kind == EulerHdgPositivityLimiterKind::none ||
      kind == EulerHdgPositivityLimiterKind::zhang_shu;
}

/** Backend-neutral positivity points consumed by the Zhang--Shu limiter. */
enum class EulerHdgZhangShuPointSet : std::uint8_t
{
  // Consumer adds the reference-element vertices and nodal coefficients to
  // the volume/face residual rules. It is zero so the complete inactive
  // payload is value-initialized.
  consumer = 0,
  // Quadrature contains only the volume and face residual rules.
  quadrature = 1
};

constexpr bool IsKnownEulerHdgZhangShuPointSet(
    const EulerHdgZhangShuPointSet point_set) noexcept
{
  return point_set == EulerHdgZhangShuPointSet::consumer ||
      point_set == EulerHdgZhangShuPointSet::quadrature;
}

/**
 * Typed Zhang--Shu admission policy.
 *
 * The inactive spelling is exactly the value-initialized object: `none`,
 * `consumer`, zero floors, and zero bisection iterations.
 * This makes the enclosing trailing policy extension safe for old aggregate
 * initializers while keeping inactive numerical identity unambiguous.
 */
struct EulerHdgPositivityLimiterPolicy
{
  EulerHdgPositivityLimiterKind kind;
  EulerHdgZhangShuPointSet point_set;
  double density_floor;
  double pressure_floor;
  int pressure_bisection_iterations;
};

/** Independent MFEM integration-rule orders for element and face assembly. */
struct EulerHdgAssemblyQuadratureOrders
{
  int volume_order;
  int face_order;
};

/** Rank-identical numerical choices for one Euler HDG invocation. */
struct EulerHdgNumericalPolicy
{
  // All quadrature members are MFEM integration-rule orders.
  int volume_projection_quadrature_order;
  int true_trace_projection_quadrature_order;
  int conventional_assembly_quadrature_order;
  EulerHdgAssemblyQuadratureOrders igr_assembly_quadrature_orders;
  int maximum_nonlinear_updates;
  // Zero selects the legacy adaptive cap: max(100, global true-trace size).
  // A positive value is an exact per-solve iteration cap.
  int maximum_linear_iterations;
  int maximum_line_search_trials;
  double nonlinear_relative_tolerance;
  double newton_correction_linear_relative_tolerance;
  double final_coherent_linear_relative_tolerance;
  // A strictly smaller value enables one same-system tightening after a
  // finite FP64 true-residual miss.  Equality disables tightening.
  double true_residual_tightening_linear_relative_tolerance;
  double true_residual_relative_tolerance;
  double conservation_relative_tolerance;
  double armijo_constant;
  RegularizationSelection stabilization;
  /** Trailing zero-compatible policy; `none` is the production default. */
  EulerHdgPositivityLimiterPolicy positivity_limiter;
};

/** Preserve the pre-policy generic CPU numerical choices for order @p p. */
constexpr EulerHdgNumericalPolicy DefaultEulerHdgNumericalPolicy(
    const int p) noexcept
{
  const int quadrature = 2 * p + 2;
  // Preserve the established low-order IGR rule, and integrate products of
  // the higher-order basis without reducing the local mass matrix's rank.
  const int igr_quadrature = std::max(12, quadrature);
  return {quadrature, quadrature, quadrature,
          {igr_quadrature, igr_quadrature}, 80, 0, 13,
          1.0e-10, 1.0e-12, 1.0e-12, 1.0e-12, 1.0e-10,
          1.0e-10, 1.0e-4, {RegularizationKind::none, 0.0, 0.0}, {}};
}

/** Resolve the active assembly rules without duplicating engine policy. */
constexpr EulerHdgAssemblyQuadratureOrders
ResolveEulerHdgAssemblyQuadratureOrders(
    const EulerHdgNumericalPolicy& policy) noexcept
{
  return policy.stabilization.kind == RegularizationKind::igr
      ? policy.igr_assembly_quadrature_orders
      : EulerHdgAssemblyQuadratureOrders{
            policy.conventional_assembly_quadrature_order,
            policy.conventional_assembly_quadrature_order};
}

constexpr double EulerHdgRequestedLinearRelativeTolerance(
    const EulerHdgNumericalPolicy& policy,
    const EulerHdgLinearSolveRole role) noexcept
{
  return role == EulerHdgLinearSolveRole::newton_correction
      ? policy.newton_correction_linear_relative_tolerance
      : policy.final_coherent_linear_relative_tolerance;
}

/** Resolve the documented adaptive linear cap for a true-trace system. */
constexpr int ResolveEulerHdgMaximumLinearIterations(
    const int requested, const int global_true_trace_size) noexcept
{
  return requested > 0 ? requested : std::max(100, global_true_trace_size);
}

/**
 * Fixed GMRES restart dimension for one true-trace system lifetime.
 *
 * This deliberately does not depend on the mutable per-solve iteration cap:
 * HYPRE allocates its Krylov basis for the configured restart dimension.
 */
constexpr int EulerHdgKrylovRestartDimension(
    const int global_true_trace_size) noexcept
{
  return std::min(100, std::max(20, global_true_trace_size));
}

/**
 * Normalize a global conservation defect by an extensive FP64 reference.
 *
 * The domain measure supplies the absolute part of the mixed scale.  Unlike
 * a literal `1 + balance_scale`, both numerator and denominator therefore
 * grow by the same factor when an otherwise identical problem is replicated
 * over a larger domain.
 */
inline double EulerHdgConservationRelativeError(
    const double absolute_error, const double balance_scale,
    const double domain_measure) noexcept
{
  if (!std::isfinite(absolute_error) || absolute_error < 0.0 ||
      !std::isfinite(balance_scale) || balance_scale < 0.0 ||
      !std::isfinite(domain_measure) || !(domain_measure > 0.0))
    return std::numeric_limits<double>::infinity();
  const double denominator = domain_measure + balance_scale;
  return std::isfinite(denominator) && denominator > 0.0
      ? absolute_error / denominator
      : std::numeric_limits<double>::infinity();
}

enum class ContractStatus : std::uint8_t
{
  accepted,
  missing_capability,
  null_mesh,
  null_communicator,
  incompatible_communicator,
  nonconforming_mesh,
  wrong_dimension,
  empty_mesh,
  nonquadrilateral_mesh,
  unsupported_polynomial_order,
  invalid_backend,
  invalid_initial_state,
  invalid_source,
  invalid_boundary_operator,
  collective_failure,
  invalid_regularization,
  invalid_initial_trace,
  invalid_numerical_policy,
  invalid_time_treatment
};

enum class Capability : std::uint32_t
{
  conforming_quadrilateral_mesh = 1u << 0,
  polynomial_order_one_to_ten = 1u << 1,
  // Retain the old spelling and bit for callers of the earlier contract.
  polynomial_order_one_to_three = polynomial_order_one_to_ten,
  cpu = 1u << 2,
  cuda = 1u << 3,
  external_initial_state = 1u << 4,
  external_source = 1u << 5,
  numerical_boundary_operator = 1u << 6
};

constexpr std::uint32_t CapabilityBit(const Capability capability) noexcept
{
  return static_cast<std::uint32_t>(capability);
}

/** Explicit backend selection; CUDA requires a non-negative device ordinal. */
struct BackendSelection
{
  BackendKind kind;
  int device_ordinal;
  // ordinary is zero so existing two-member aggregate initializers retain the
  // production default while the contract remains a trivial POD.
  CudaEngineKind cuda_engine;
};

/** A non-owning component-major or engine-defined Euler field view. */
struct EulerFieldView
{
  const double* data;
  std::size_t scalar_count;
  int components;
};

/** Point data supplied to a host numerical boundary callback by an engine. */
struct BoundaryPointView
{
  const double* position;
  const double* outward_normal;
  int space_dimension;
  int boundary_attribute;
};

/**
 * A borrowed host numerical boundary operator.  The callback may inspect only
 * the supplied point and interior state and writes one four-component exterior
 * state.  It receives no time coordinate: its numerical value is therefore an
 * autonomous, deterministic map of `(context, point, interior_state)` for one
 * admitted lifecycle.  Observational counters may change, but mutable context
 * must not change the returned state.  RODAS3 may reevaluate this map at
 * bounded FP64 perturbations to form its frozen boundary Jacobian.  The
 * callback receives no mesh, solver, lifecycle, or state-owner handle.  The
 * companion attribute-indexed POD table carries the same resolved policy data
 * for device backends; it is copied during Initialize and never retained.
 */
using NumericalBoundaryState = bool (*) (
    const void* context,
    const BoundaryPointView& point,
    const double* interior_state,
    double* exterior_state) noexcept;

/** Backend-neutral, attribute-indexed Euler boundary policy. */
enum class EulerBoundaryPolicyKind : std::uint8_t
{
  periodic_topology,
  supersonic_inlet,
  subsonic_inlet,
  supersonic_outlet,
  subsonic_outlet_pressure,
  far_field,
  nonreflecting_outlet,
  slip_wall
};

/**
 * POD boundary data shared by host callbacks and device lowering.  Entry zero
 * describes boundary attribute one.  The table is borrowed for Initialize;
 * an engine must copy it before retaining numerical state.
 */
struct EulerBoundaryPolicy
{
  EulerBoundaryPolicyKind kind;
  std::array<double, euler_2d_conserved_components> reference_state;
  double pressure_out;
};

struct EulerBoundaryPolicyTableView
{
  const EulerBoundaryPolicy* policies;
  std::size_t count;
  int space_dimension;
  double gamma;
};

struct BoundaryOperatorView
{
  const void* context;
  NumericalBoundaryState exterior_state;
  EulerBoundaryPolicyTableView policies;
};

/**
 * All pointers and views are borrowed for the caller-defined invocation.
 * `mesh` and `communicator` must describe the same externally owned ParMesh.
 * A collective caller supplies a non-null `mesh` on every participating rank;
 * that mesh communicator is the admission-convergence domain.
 */
struct EulerHdgRequest
{
  const mfem::ParMesh* mesh;
  MPI_Comm communicator;
  int polynomial_order;
  BackendSelection backend;
  std::uint32_t capabilities;
  EulerFieldView initial_state;
  /** Independently projected component-major owned true-trace coordinates. */
  EulerFieldView initial_true_trace;
  EulerFieldView source;
  BoundaryOperatorView boundary;
  EulerHdgNumericalPolicy numerical_policy;
  /** Trailing zero-compatible typed time architecture; Backward Euler is zero. */
  EulerHdgTimeTreatment time_treatment;
};

static_assert(std::is_standard_layout<BackendSelection>::value &&
                  std::is_trivial<BackendSelection>::value,
              "generic HDG backend selection must stay POD");
static_assert(std::is_standard_layout<EulerFieldView>::value &&
                  std::is_trivial<EulerFieldView>::value,
              "generic HDG field views must stay POD");
static_assert(std::is_standard_layout<BoundaryPointView>::value &&
                  std::is_trivial<BoundaryPointView>::value,
              "generic HDG boundary point views must stay POD");
static_assert(std::is_standard_layout<BoundaryOperatorView>::value &&
                  std::is_trivial<BoundaryOperatorView>::value,
              "generic HDG boundary views must stay POD");
static_assert(std::is_standard_layout<EulerBoundaryPolicy>::value &&
                  std::is_trivial<EulerBoundaryPolicy>::value,
              "generic HDG boundary policies must stay POD");
static_assert(std::is_standard_layout<EulerBoundaryPolicyTableView>::value &&
                  std::is_trivial<EulerBoundaryPolicyTableView>::value,
              "generic HDG boundary-policy views must stay POD");
static_assert(std::is_standard_layout<RegularizationSelection>::value &&
                  std::is_trivial<RegularizationSelection>::value,
              "generic HDG regularization must stay POD");
static_assert(
    std::is_standard_layout<EulerHdgPositivityLimiterPolicy>::value &&
        std::is_trivial<EulerHdgPositivityLimiterPolicy>::value,
    "generic HDG positivity-limiter policy must stay POD");
static_assert(
    std::is_standard_layout<EulerHdgAssemblyQuadratureOrders>::value &&
        std::is_trivial<EulerHdgAssemblyQuadratureOrders>::value,
    "generic HDG assembly quadrature orders must stay POD");
static_assert(std::is_standard_layout<EulerHdgNumericalPolicy>::value &&
                  std::is_trivial<EulerHdgNumericalPolicy>::value,
              "generic HDG numerical policy must stay POD");
static_assert(std::is_standard_layout<EulerHdgRequest>::value &&
                  std::is_trivial<EulerHdgRequest>::value &&
                  !std::is_polymorphic<EulerHdgRequest>::value,
              "generic HDG request must stay a nonvirtual POD contract");

constexpr bool HasCapabilities(const std::uint32_t available,
                               const std::uint32_t required) noexcept
{
  return (available & required) == required;
}

inline std::uint64_t DoubleBits(const double value) noexcept
{
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

constexpr bool IsKnownEulerBoundaryPolicyKind(
    const EulerBoundaryPolicyKind kind) noexcept
{
  switch (kind)
  {
  case EulerBoundaryPolicyKind::periodic_topology:
  case EulerBoundaryPolicyKind::supersonic_inlet:
  case EulerBoundaryPolicyKind::subsonic_inlet:
  case EulerBoundaryPolicyKind::supersonic_outlet:
  case EulerBoundaryPolicyKind::subsonic_outlet_pressure:
  case EulerBoundaryPolicyKind::far_field:
  case EulerBoundaryPolicyKind::nonreflecting_outlet:
  case EulerBoundaryPolicyKind::slip_wall:
    return true;
  }
  return false;
}

inline bool IsAbsentEulerBoundaryPolicyTable(
    const EulerBoundaryPolicyTableView& table) noexcept
{
  return table.policies == nullptr && table.count == 0 &&
      table.space_dimension == 0 && DoubleBits(table.gamma) == 0;
}

inline bool IsValidEulerBoundaryPolicyTable(
    const EulerBoundaryPolicyTableView& table,
    const int expected_space_dimension,
    const int expected_attribute_count,
    const int retained_attribute_count,
    const std::vector<unsigned char>& physical_attribute_mask) noexcept
{
  // Preserve the original callback-only generic contract.  CUDA application
  // requests supply the table; a backend that needs device lowering may reject
  // this compatibility spelling during its own admission.
  if (IsAbsentEulerBoundaryPolicyTable(table)) return true;
  const bool physical_boundary_present = std::any_of(
      physical_attribute_mask.begin(), physical_attribute_mask.end(),
      [](const unsigned char value) { return value != 0; });
  const bool empty_periodic_table = !physical_boundary_present &&
      table.count == 0;
  const bool exact_retained_table = table.count ==
      static_cast<std::size_t>(retained_attribute_count);
  if (expected_space_dimension != 2 || table.space_dimension != 2 ||
      expected_attribute_count < 0 || retained_attribute_count < 0 ||
      physical_attribute_mask.size() !=
          static_cast<std::size_t>(retained_attribute_count) ||
      expected_attribute_count != (physical_boundary_present
          ? retained_attribute_count : 0) ||
      (!empty_periodic_table && !exact_retained_table) ||
      !std::isfinite(table.gamma) || table.gamma <= 1.0)
  {
    return false;
  }
  if (table.count == 0) return table.policies == nullptr;
  if (table.policies == nullptr) return false;
  for (std::size_t index = 0; index < table.count; ++index)
  {
    const EulerBoundaryPolicy& policy = table.policies[index];
    const bool physical_attribute = physical_attribute_mask[index] != 0;
    const bool periodic_policy =
        policy.kind == EulerBoundaryPolicyKind::periodic_topology;
    if (!IsKnownEulerBoundaryPolicyKind(policy.kind) ||
        physical_attribute == periodic_policy ||
        !std::isfinite(policy.pressure_out))
    {
      return false;
    }
    for (const double value : policy.reference_state)
    {
      if (!std::isfinite(value)) return false;
    }
  }
  return true;
}

/** Exact collective identity for optional attribute-indexed policy data. */
inline bool ExactEulerBoundaryPolicyConsensus(
    const MPI_Comm communicator,
    const EulerBoundaryPolicyTableView& table) noexcept
{
  if (communicator == MPI_COMM_NULL) return false;
  const std::uint64_t local_count = static_cast<std::uint64_t>(table.count);
  std::uint64_t minimum_count = 0;
  std::uint64_t maximum_count = 0;
  const int local_dimension = table.space_dimension;
  int minimum_dimension = 0;
  int maximum_dimension = 0;
  const std::uint64_t local_gamma = DoubleBits(table.gamma);
  std::uint64_t minimum_gamma = 0;
  std::uint64_t maximum_gamma = 0;
  const int locally_accessible =
      (table.count == 0 ? table.policies == nullptr
                        : table.policies != nullptr) ? 1 : 0;
  int globally_accessible = 0;
  const int minimum_count_status = MPI_Allreduce(
      &local_count, &minimum_count, 1, MPI_UINT64_T, MPI_MIN, communicator);
  const int maximum_count_status = MPI_Allreduce(
      &local_count, &maximum_count, 1, MPI_UINT64_T, MPI_MAX, communicator);
  const int minimum_dimension_status = MPI_Allreduce(
      &local_dimension, &minimum_dimension, 1, MPI_INT, MPI_MIN, communicator);
  const int maximum_dimension_status = MPI_Allreduce(
      &local_dimension, &maximum_dimension, 1, MPI_INT, MPI_MAX, communicator);
  const int minimum_gamma_status = MPI_Allreduce(
      &local_gamma, &minimum_gamma, 1, MPI_UINT64_T, MPI_MIN, communicator);
  const int maximum_gamma_status = MPI_Allreduce(
      &local_gamma, &maximum_gamma, 1, MPI_UINT64_T, MPI_MAX, communicator);
  const int accessible_status = MPI_Allreduce(
      &locally_accessible, &globally_accessible, 1, MPI_INT, MPI_MIN,
      communicator);
  const bool metadata_ready = minimum_count_status == MPI_SUCCESS &&
      maximum_count_status == MPI_SUCCESS &&
      minimum_dimension_status == MPI_SUCCESS &&
      maximum_dimension_status == MPI_SUCCESS &&
      minimum_gamma_status == MPI_SUCCESS &&
      maximum_gamma_status == MPI_SUCCESS && accessible_status == MPI_SUCCESS;
  if (!metadata_ready || minimum_count != maximum_count ||
      minimum_dimension != maximum_dimension || minimum_gamma != maximum_gamma ||
      globally_accessible != 1)
  {
    return false;
  }

  for (std::size_t index = 0; index < table.count; ++index)
  {
    const EulerBoundaryPolicy& policy = table.policies[index];
    const int local_kind = static_cast<int>(policy.kind);
    int minimum_kind = 0;
    int maximum_kind = 0;
    std::uint64_t local_values[euler_2d_conserved_components + 1]{};
    std::uint64_t minimum_values[euler_2d_conserved_components + 1]{};
    std::uint64_t maximum_values[euler_2d_conserved_components + 1]{};
    for (int component = 0; component < euler_2d_conserved_components;
         ++component)
    {
      local_values[component] = DoubleBits(
          policy.reference_state[static_cast<std::size_t>(component)]);
    }
    local_values[euler_2d_conserved_components] =
        DoubleBits(policy.pressure_out);
    const int minimum_kind_status = MPI_Allreduce(
        &local_kind, &minimum_kind, 1, MPI_INT, MPI_MIN, communicator);
    const int maximum_kind_status = MPI_Allreduce(
        &local_kind, &maximum_kind, 1, MPI_INT, MPI_MAX, communicator);
    const int minimum_values_status = MPI_Allreduce(
        local_values, minimum_values, euler_2d_conserved_components + 1,
        MPI_UINT64_T, MPI_MIN, communicator);
    const int maximum_values_status = MPI_Allreduce(
        local_values, maximum_values, euler_2d_conserved_components + 1,
        MPI_UINT64_T, MPI_MAX, communicator);
    if (minimum_kind_status != MPI_SUCCESS ||
        maximum_kind_status != MPI_SUCCESS ||
        minimum_values_status != MPI_SUCCESS ||
        maximum_values_status != MPI_SUCCESS ||
        minimum_kind != maximum_kind ||
        !std::equal(std::begin(minimum_values), std::end(minimum_values),
                    std::begin(maximum_values)))
    {
      return false;
    }
  }
  return true;
}

inline bool ValidRegularizationSelection(
    const RegularizationSelection selection) noexcept
{
  if (selection.kind == RegularizationKind::none)
  {
    return DoubleBits(selection.alpha) == 0 && DoubleBits(selection.beta) == 0;
  }
  return selection.kind == RegularizationKind::igr &&
      std::isfinite(selection.alpha) && selection.alpha > 0.0 &&
      std::isfinite(selection.beta) && selection.beta > 0.0;
}

inline bool ValidEulerHdgPositivityLimiterPolicy(
    const EulerHdgPositivityLimiterPolicy& policy) noexcept
{
  if (policy.kind == EulerHdgPositivityLimiterKind::none)
  {
    return policy.point_set == EulerHdgZhangShuPointSet::consumer &&
        DoubleBits(policy.density_floor) == 0 &&
        DoubleBits(policy.pressure_floor) == 0 &&
        policy.pressure_bisection_iterations == 0;
  }
  return policy.kind == EulerHdgPositivityLimiterKind::zhang_shu &&
      IsKnownEulerHdgZhangShuPointSet(policy.point_set) &&
      std::isfinite(policy.density_floor) && policy.density_floor > 0.0 &&
      std::isfinite(policy.pressure_floor) && policy.pressure_floor > 0.0 &&
      policy.pressure_bisection_iterations > 0 &&
      policy.pressure_bisection_iterations <= 64;
}

inline bool ValidEulerHdgNumericalPolicy(
    const EulerHdgNumericalPolicy& policy) noexcept
{
  const int positive_integers[] = {
      policy.volume_projection_quadrature_order,
      policy.true_trace_projection_quadrature_order,
      policy.conventional_assembly_quadrature_order,
      policy.igr_assembly_quadrature_orders.volume_order,
      policy.igr_assembly_quadrature_orders.face_order,
      policy.maximum_nonlinear_updates,
      policy.maximum_line_search_trials};
  for (const int value : positive_integers)
  {
    if (value <= 0) { return false; }
  }
  if (policy.maximum_linear_iterations < 0) { return false; }
  const double positive_reals[] = {policy.nonlinear_relative_tolerance,
                                   policy.newton_correction_linear_relative_tolerance,
                                   policy.final_coherent_linear_relative_tolerance,
                                   policy.true_residual_tightening_linear_relative_tolerance,
                                   policy.true_residual_relative_tolerance,
                                   policy.conservation_relative_tolerance};
  for (const double value : positive_reals)
  {
    if (!std::isfinite(value) || value <= 0.0) { return false; }
  }
  return policy.true_residual_tightening_linear_relative_tolerance <=
          policy.newton_correction_linear_relative_tolerance &&
      policy.true_residual_tightening_linear_relative_tolerance <=
          policy.final_coherent_linear_relative_tolerance &&
      std::isfinite(policy.armijo_constant) &&
      policy.armijo_constant > 0.0 && policy.armijo_constant < 1.0 &&
      ValidRegularizationSelection(policy.stabilization) &&
      ValidEulerHdgPositivityLimiterPolicy(policy.positivity_limiter);
}

/**
 * Periodic MFEM meshes retain boundary-element annotations for faces that
 * become interior (or rank-shared) after periodic identification.  This
 * helper distinguishes those annotations from a true physical boundary face
 * rather than using GetNBE() directly.
 */
inline bool LocalPhysicalBoundaryAttributeMask(
    const mfem::ParMesh& mesh, int& retained_attribute_count,
    std::vector<int>& physical_attribute_mask) noexcept
{
  retained_attribute_count = 0;
  physical_attribute_mask.clear();
  try
  {
    const int faces = mesh.GetNumFaces();
    const int boundary_elements = mesh.GetNBE();
    const int shared_faces = mesh.GetNSharedFaces();
    if (faces < 0 || boundary_elements < 0 || shared_faces < 0)
      return false;
    retained_attribute_count = mesh.bdr_attributes.Size() == 0
        ? 0 : mesh.bdr_attributes.Max();
    if (retained_attribute_count < 0) return false;

    std::vector<unsigned char> shared(
        static_cast<std::size_t>(faces), 0);
    std::vector<int> face_to_boundary(
        static_cast<std::size_t>(faces), -1);
    physical_attribute_mask.assign(
        static_cast<std::size_t>(retained_attribute_count), 0);
    for (int shared_face = 0; shared_face < shared_faces; ++shared_face)
    {
      const int face = mesh.GetSharedFace(shared_face);
      if (face < 0 || face >= faces) return false;
      shared[static_cast<std::size_t>(face)] = 1;
    }
    for (int boundary = 0; boundary < boundary_elements; ++boundary)
    {
      const int face = mesh.GetBdrElementFaceIndex(boundary);
      const int attribute = mesh.GetBdrAttribute(boundary);
      if (face < 0 || face >= faces || attribute <= 0 ||
          attribute > retained_attribute_count)
        return false;
      face_to_boundary[static_cast<std::size_t>(face)] = boundary;
    }
    for (int face = 0; face < faces; ++face)
    {
      const bool physical = !mesh.FaceIsInterior(face) &&
          shared[static_cast<std::size_t>(face)] == 0;
      if (!physical) continue;
      const int boundary =
          face_to_boundary[static_cast<std::size_t>(face)];
      if (boundary < 0 || boundary >= boundary_elements) return false;
      const int attribute = mesh.GetBdrAttribute(boundary);
      if (attribute <= 0 || attribute > retained_attribute_count)
        return false;
      physical_attribute_mask[
          static_cast<std::size_t>(attribute - 1)] = 1;
    }
    return true;
  }
  catch (...) { return false; }
}

/**
 * Resolve the attribute-table extent for physical faces collectively.
 *
 * MFEM preserves the original boundary attributes after periodic vertex
 * identification, so bdr_attributes.Max() alone cannot distinguish a fully
 * periodic mesh from a physical-boundary mesh.  Conversely, an MPI rank may
 * own no physical face even though another rank does.  Use the request
 * communicator to make both decisions global before validating the shared
 * attribute-indexed policy table.
 */
inline bool GlobalPhysicalBoundaryAttributeCount(
    const mfem::ParMesh& mesh, const MPI_Comm communicator,
    int& attribute_count, int& retained_attribute_count,
    std::vector<unsigned char>& physical_attribute_mask) noexcept
{
  if (communicator == MPI_COMM_NULL) return false;
  attribute_count = 0;
  retained_attribute_count = 0;
  physical_attribute_mask.clear();
  int local_maximum = 0;
  std::vector<int> local_physical_attributes;
  const int local_topology_ready = LocalPhysicalBoundaryAttributeMask(
      mesh, local_maximum, local_physical_attributes) ? 1 : 0;
  int global_topology_ready = 0;
  int global_maximum = 0;
  const bool topology_ready = MPI_Allreduce(
      &local_topology_ready, &global_topology_ready, 1, MPI_INT, MPI_MIN,
      communicator) == MPI_SUCCESS;
  const bool maximum_ready = MPI_Allreduce(
      &local_maximum, &global_maximum, 1, MPI_INT, MPI_MAX,
      communicator) == MPI_SUCCESS;
  if (!topology_ready || !maximum_ready || global_topology_ready != 1 ||
      global_maximum < 0)
  {
    return false;
  }

  int local_storage_ready = 1;
  std::vector<int> global_physical_attributes;
  try
  {
    local_physical_attributes.resize(
        static_cast<std::size_t>(global_maximum), 0);
    global_physical_attributes.assign(
        static_cast<std::size_t>(global_maximum), 0);
    physical_attribute_mask.assign(
        static_cast<std::size_t>(global_maximum), 0);
  }
  catch (...) { local_storage_ready = 0; }
  int global_storage_ready = 0;
  if (MPI_Allreduce(&local_storage_ready, &global_storage_ready, 1, MPI_INT,
                    MPI_MIN, communicator) != MPI_SUCCESS ||
      global_storage_ready != 1)
    return false;

  int dummy_send = 0;
  int dummy_receive = 0;
  const int mask_status = MPI_Allreduce(
      global_maximum == 0 ? &dummy_send : local_physical_attributes.data(),
      global_maximum == 0 ? &dummy_receive
                          : global_physical_attributes.data(),
      global_maximum, MPI_INT, MPI_MAX, communicator);
  if (mask_status != MPI_SUCCESS) return false;
  bool physical_boundary_present = false;
  for (int attribute = 0; attribute < global_maximum; ++attribute)
  {
    const int value = global_physical_attributes[
        static_cast<std::size_t>(attribute)];
    if (value < 0 || value > 1) return false;
    physical_attribute_mask[static_cast<std::size_t>(attribute)] =
        static_cast<unsigned char>(value);
    physical_boundary_present = physical_boundary_present || value != 0;
  }
  if (physical_boundary_present && global_maximum == 0) return false;
  attribute_count = physical_boundary_present ? global_maximum : 0;
  retained_attribute_count = global_maximum;
  return true;
}

constexpr std::uint32_t RequiredCapabilities(
    const BackendSelection backend) noexcept
{
  const std::uint32_t common =
      CapabilityBit(Capability::conforming_quadrilateral_mesh) |
      CapabilityBit(Capability::polynomial_order_one_to_ten) |
      CapabilityBit(Capability::external_initial_state) |
      CapabilityBit(Capability::external_source) |
      CapabilityBit(Capability::numerical_boundary_operator);
  switch (backend.kind)
  {
    case BackendKind::cpu:
      return common | CapabilityBit(Capability::cpu);
    case BackendKind::cuda:
      return common | CapabilityBit(Capability::cuda);
  }
  return 0;
}

constexpr bool IsValidField(const EulerFieldView field,
                            const bool required) noexcept
{
  if (field.components != euler_2d_conserved_components)
  {
    return false;
  }
  if (field.scalar_count == 0)
  {
    return !required && field.data == nullptr;
  }
  return field.data != nullptr &&
      field.scalar_count % static_cast<std::size_t>(field.components) == 0;
}

constexpr int CollectiveStatusCode(const ContractStatus status) noexcept
{
  return status == ContractStatus::accepted
      ? std::numeric_limits<int>::max()
      : static_cast<int>(status);
}

constexpr ContractStatus StatusFromCollectiveCode(const int code) noexcept
{
  return code == std::numeric_limits<int>::max()
      ? ContractStatus::accepted
      : static_cast<ContractStatus>(code);
}

constexpr void RecordFailure(ContractStatus& status,
                             const ContractStatus candidate) noexcept
{
  if (status == ContractStatus::accepted ||
      static_cast<int>(candidate) < static_cast<int>(status))
  {
    status = candidate;
  }
}

inline ContractStatus CollectAdmissionStatus(
    MPI_Comm communicator,
    const ContractStatus local_status,
    const int local_elements) noexcept
{
  int global_status_code = CollectiveStatusCode(local_status);
  long long global_elements = 0;
  const long long local_element_count = local_elements;
  const int status_result = MPI_Allreduce(
      MPI_IN_PLACE, &global_status_code, 1, MPI_INT, MPI_MIN, communicator);
  const int elements_result = MPI_Allreduce(
      &local_element_count, &global_elements, 1, MPI_LONG_LONG_INT, MPI_SUM,
      communicator);
  if (status_result != MPI_SUCCESS || elements_result != MPI_SUCCESS)
  {
    return ContractStatus::collective_failure;
  }
  const ContractStatus status = StatusFromCollectiveCode(global_status_code);
  return status == ContractStatus::accepted && global_elements <= 0
      ? ContractStatus::empty_mesh
      : status;
}

inline bool ExactNumericalPolicyConsensus(
    const MPI_Comm communicator, const EulerHdgNumericalPolicy& policy) noexcept
{
  const int integers[] = {
      policy.volume_projection_quadrature_order,
      policy.true_trace_projection_quadrature_order,
      policy.conventional_assembly_quadrature_order,
      policy.igr_assembly_quadrature_orders.volume_order,
      policy.igr_assembly_quadrature_orders.face_order,
      policy.maximum_nonlinear_updates,
      policy.maximum_linear_iterations,
      policy.maximum_line_search_trials,
      static_cast<int>(policy.stabilization.kind),
      static_cast<int>(policy.positivity_limiter.kind),
      static_cast<int>(policy.positivity_limiter.point_set),
      policy.positivity_limiter.pressure_bisection_iterations};
  std::uint64_t reals[] = {
      DoubleBits(policy.nonlinear_relative_tolerance),
      DoubleBits(policy.newton_correction_linear_relative_tolerance),
      DoubleBits(policy.final_coherent_linear_relative_tolerance),
      DoubleBits(policy.true_residual_tightening_linear_relative_tolerance),
      DoubleBits(policy.true_residual_relative_tolerance),
      DoubleBits(policy.conservation_relative_tolerance),
      DoubleBits(policy.armijo_constant),
      DoubleBits(policy.stabilization.alpha),
      DoubleBits(policy.stabilization.beta),
      DoubleBits(policy.positivity_limiter.density_floor),
      DoubleBits(policy.positivity_limiter.pressure_floor)};
  int minimum_integers[sizeof(integers) / sizeof(*integers)]{};
  int maximum_integers[sizeof(integers) / sizeof(*integers)]{};
  std::uint64_t minimum_reals[sizeof(reals) / sizeof(*reals)]{};
  std::uint64_t maximum_reals[sizeof(reals) / sizeof(*reals)]{};
  const int integer_count =
      static_cast<int>(sizeof(integers) / sizeof(*integers));
  const int real_count = static_cast<int>(sizeof(reals) / sizeof(*reals));
  const bool minimum_integers_ready = MPI_Allreduce(
      integers, minimum_integers, integer_count, MPI_INT, MPI_MIN,
      communicator) == MPI_SUCCESS;
  const bool maximum_integers_ready = MPI_Allreduce(
      integers, maximum_integers, integer_count, MPI_INT, MPI_MAX,
      communicator) == MPI_SUCCESS;
  const bool minimum_reals_ready = MPI_Allreduce(
      reals, minimum_reals, real_count, MPI_UINT64_T, MPI_MIN,
      communicator) == MPI_SUCCESS;
  const bool maximum_reals_ready = MPI_Allreduce(
      reals, maximum_reals, real_count, MPI_UINT64_T, MPI_MAX,
      communicator) == MPI_SUCCESS;
  return minimum_integers_ready && maximum_integers_ready &&
      minimum_reals_ready && maximum_reals_ready &&
      std::equal(std::begin(minimum_integers), std::end(minimum_integers),
                 std::begin(maximum_integers)) &&
      std::equal(std::begin(minimum_reals), std::end(minimum_reals),
                 std::begin(maximum_reals));
}

inline bool ExactBackendSelectionConsensus(
    const MPI_Comm communicator, const BackendSelection backend) noexcept
{
  const int local[] = {
      static_cast<int>(backend.kind), backend.device_ordinal,
      static_cast<int>(backend.cuda_engine)};
  int minimum[sizeof(local) / sizeof(*local)]{};
  int maximum[sizeof(local) / sizeof(*local)]{};
  const int count = static_cast<int>(sizeof(local) / sizeof(*local));
  const bool minimum_ready = MPI_Allreduce(
      local, minimum, count, MPI_INT, MPI_MIN, communicator) == MPI_SUCCESS;
  const bool maximum_ready = MPI_Allreduce(
      local, maximum, count, MPI_INT, MPI_MAX, communicator) == MPI_SUCCESS;
  return minimum_ready && maximum_ready &&
      std::equal(std::begin(minimum), std::end(minimum),
                 std::begin(maximum));
}

inline bool ExactEulerHdgTimeTreatmentConsensus(
    const MPI_Comm communicator,
    const EulerHdgTimeTreatment treatment) noexcept
{
  const int local = static_cast<int>(treatment);
  int minimum = local;
  int maximum = local;
  const bool minimum_ready = MPI_Allreduce(
      &local, &minimum, 1, MPI_INT, MPI_MIN, communicator) == MPI_SUCCESS;
  const bool maximum_ready = MPI_Allreduce(
      &local, &maximum, 1, MPI_INT, MPI_MAX, communicator) == MPI_SUCCESS;
  return minimum_ready && maximum_ready && minimum == maximum;
}

inline ContractStatus ValidateEulerHdgRequest(
    const EulerHdgRequest& request) noexcept
{
  int mpi_initialized = 0;
  int mpi_finalized = 0;
  if (MPI_Initialized(&mpi_initialized) != MPI_SUCCESS || !mpi_initialized ||
      MPI_Finalized(&mpi_finalized) != MPI_SUCCESS || mpi_finalized)
  {
    return ContractStatus::incompatible_communicator;
  }
  if (request.communicator == MPI_COMM_NULL)
  {
    return ContractStatus::null_communicator;
  }
  // The request communicator is the sole collective authority.  Establish
  // mesh presence on it before any rank dereferences a borrowed mesh: one
  // rank-local null must reject the complete request instead of letting the
  // null rank return while its peers enter mesh collectives.
  const int local_mesh_present = request.mesh == nullptr ? 0 : 1;
  int every_mesh_present = 0;
  if (MPI_Allreduce(&local_mesh_present, &every_mesh_present, 1, MPI_INT,
                    MPI_MIN, request.communicator) != MPI_SUCCESS)
  {
    return ContractStatus::collective_failure;
  }
  if (every_mesh_present != 1) { return ContractStatus::null_mesh; }

  const MPI_Comm mesh_communicator = request.mesh->GetComm();
  ContractStatus local_status = ContractStatus::accepted;
  int comparison = MPI_UNEQUAL;
  if (MPI_Comm_compare(request.communicator, mesh_communicator,
                       &comparison) != MPI_SUCCESS ||
      (comparison != MPI_IDENT && comparison != MPI_CONGRUENT))
  {
    RecordFailure(local_status, ContractStatus::incompatible_communicator);
  }
  // Keep this fixed sequence even for malformed local input: validation is
  // collective, and every numerical choice has exact rank agreement.
  if (!ExactNumericalPolicyConsensus(request.communicator,
                                     request.numerical_policy) ||
      !ValidEulerHdgNumericalPolicy(request.numerical_policy))
  {
    RecordFailure(local_status, ContractStatus::invalid_numerical_policy);
  }
  if (!ExactEulerHdgTimeTreatmentConsensus(request.communicator,
                                           request.time_treatment) ||
      !IsKnownEulerHdgTimeTreatment(request.time_treatment))
  {
    RecordFailure(local_status, ContractStatus::invalid_time_treatment);
  }
  else if (request.numerical_policy.positivity_limiter.kind !=
               EulerHdgPositivityLimiterKind::none &&
           request.time_treatment != EulerHdgTimeTreatment::rodas3)
  {
    // The first HDG Zhang--Shu lifecycle is a filtered RODAS3 method.  Never
    // accept an active policy on a Backward-Euler path that does not execute
    // the filter.
    RecordFailure(local_status, ContractStatus::invalid_numerical_policy);
  }
  if (!ExactBackendSelectionConsensus(request.communicator, request.backend) ||
      (request.backend.kind != BackendKind::cpu &&
       request.backend.kind != BackendKind::cuda) ||
      !IsKnownCudaEngineKind(request.backend.cuda_engine) ||
      (request.backend.kind == BackendKind::cpu &&
       request.backend.cuda_engine != CudaEngineKind::ordinary))
  {
    RecordFailure(local_status, ContractStatus::invalid_backend);
  }
  else if (!HasCapabilities(request.capabilities,
                            RequiredCapabilities(request.backend)))
  {
    RecordFailure(local_status, ContractStatus::missing_capability);
  }
  if (request.mesh->Nonconforming())
  {
    RecordFailure(local_status, ContractStatus::nonconforming_mesh);
  }
  else if (request.mesh->Dimension() != 2)
  {
    RecordFailure(local_status, ContractStatus::wrong_dimension);
  }
  const int local_elements = request.mesh->GetNE();
  for (int element = 0; element < local_elements; ++element)
  {
    if (request.mesh->GetElementBaseGeometry(element) != mfem::Geometry::SQUARE)
    {
      RecordFailure(local_status, ContractStatus::nonquadrilateral_mesh);
      break;
    }
  }
  if (request.polynomial_order < minimum_polynomial_order ||
      request.polynomial_order > maximum_polynomial_order)
  {
    RecordFailure(local_status, ContractStatus::unsupported_polynomial_order);
  }
  // IGR is an augmentation selected inside either conventional backend; it
  // does not select another Euler solver or narrow the admitted order and
  // physical-boundary surface.
  if ((request.backend.kind == BackendKind::cpu &&
       request.backend.device_ordinal != -1) ||
      (request.backend.kind == BackendKind::cuda &&
       request.backend.device_ordinal < 0))
  {
    RecordFailure(local_status, ContractStatus::invalid_backend);
  }
  // A globally nonempty borrowed mesh may legitimately leave this rank with
  // no volume dofs.  Such a rank contributes an empty exact local field while
  // still joining every collective admission and trace operation.
  if (!IsValidField(request.initial_state, local_elements > 0))
  {
    RecordFailure(local_status, ContractStatus::invalid_initial_state);
  }
  // Empty is a compatibility request for the pre-policy derived-trace path.
  // New main-adapter requests always supply the independently projected view.
  if (!IsValidField(request.initial_true_trace, false))
  {
    RecordFailure(local_status, ContractStatus::invalid_initial_trace);
  }
  if (!IsValidField(request.source, false))
  {
    RecordFailure(local_status, ContractStatus::invalid_source);
  }
  int boundary_attribute_count = 0;
  int retained_boundary_attribute_count = 0;
  std::vector<unsigned char> physical_boundary_attribute_mask;
  const bool boundary_attribute_count_ready =
      GlobalPhysicalBoundaryAttributeCount(
          *request.mesh, request.communicator, boundary_attribute_count,
          retained_boundary_attribute_count,
          physical_boundary_attribute_mask);
  // Do not short-circuit this collective on a rank-local malformed table.
  // Every mesh rank enters the same metadata/payload sequence and receives the
  // same invalid-boundary disposition for unequal policy bits.
  const bool exact_boundary_policy = ExactEulerBoundaryPolicyConsensus(
      request.communicator, request.boundary.policies);
  const bool valid_boundary_policy = IsValidEulerBoundaryPolicyTable(
      request.boundary.policies, request.mesh->SpaceDimension(),
      boundary_attribute_count, retained_boundary_attribute_count,
      physical_boundary_attribute_mask);
  if (!boundary_attribute_count_ready ||
      request.boundary.exterior_state == nullptr ||
      !exact_boundary_policy || !valid_boundary_policy)
  {
    RecordFailure(local_status, ContractStatus::invalid_boundary_operator);
  }
  return CollectAdmissionStatus(
      request.communicator, local_status, local_elements);
}

}  // namespace dg::hdg::generic

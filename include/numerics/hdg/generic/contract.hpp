#pragma once

/**
 * @file numerics/hdg/generic/contract.hpp
 * @brief Borrowed-input admission contract for a future 2D Euler HDG engine.
 *
 * This header intentionally owns neither an MFEM mesh nor state.  It is an
 * admission boundary only; time integration, output, solver lifetime, and
 * restart are outside this contract.
 */

#include <mfem.hpp>

#include <algorithm>
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
inline constexpr int minimum_polynomial_order = 1;
inline constexpr int maximum_polynomial_order = 3;

enum class BackendKind : std::uint8_t
{
  cpu,
  cuda
};

/** Private numerical regularization selected by a borrowed Euler request. */
enum class RegularizationKind : std::uint8_t
{
  none = 0,
  igr = 1
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

/** Rank-identical numerical choices for one Euler HDG invocation. */
struct EulerHdgNumericalPolicy
{
  // All quadrature members are MFEM integration-rule orders.
  int volume_projection_quadrature_order;
  int true_trace_projection_quadrature_order;
  int conventional_assembly_quadrature_order;
  int igr_assembly_quadrature_order;
  int audit_quadrature_order;
  int observable_quadrature_order;
  int maximum_nonlinear_updates;
  // Zero selects the legacy adaptive cap: max(100, global true-trace size).
  // A positive value is an exact per-solve iteration cap.
  int maximum_linear_iterations;
  int maximum_line_search_trials;
  double nonlinear_relative_tolerance;
  double linear_relative_tolerance;
  double true_residual_relative_tolerance;
  double conservation_relative_tolerance;
  double armijo_constant;
  RegularizationSelection stabilization;
};

/** Preserve the pre-policy generic CPU numerical choices for order @p p. */
constexpr EulerHdgNumericalPolicy DefaultEulerHdgNumericalPolicy(
    const int p) noexcept
{
  const int quadrature = 2 * p + 2;
  // The private p1 IGR helper predated this policy and used order 12.
  return {quadrature, quadrature, quadrature, 12, quadrature,
          quadrature, 80, 0, 13, 1.0e-10, 1.0e-12, 1.0e-10,
          1.0e-10, 1.0e-4, {RegularizationKind::none, 0.0, 0.0}};
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
  invalid_numerical_policy
};

enum class Capability : std::uint32_t
{
  conforming_quadrilateral_mesh = 1u << 0,
  polynomial_order_one_to_three = 1u << 1,
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
};

/** A non-owning component-major or engine-defined Euler field view. */
struct EulerFieldView
{
  const double* data;
  std::size_t scalar_count;
  int components;
};

/** Point data supplied to a numerical boundary callback by a future engine. */
struct BoundaryPointView
{
  const double* position;
  const double* outward_normal;
  int space_dimension;
  int boundary_attribute;
};

/**
 * A borrowed numerical boundary operator.  The callback may inspect only the
 * supplied point and interior state and writes one four-component exterior
 * state.  It receives no mesh, solver, lifecycle, or state-owner handle.
 */
using NumericalBoundaryState = bool (*) (
    const void* context,
    const BoundaryPointView& point,
    const double* interior_state,
    double* exterior_state) noexcept;

struct BoundaryOperatorView
{
  const void* context;
  NumericalBoundaryState exterior_state;
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
static_assert(std::is_standard_layout<RegularizationSelection>::value &&
                  std::is_trivial<RegularizationSelection>::value,
              "generic HDG regularization must stay POD");
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

inline bool ValidEulerHdgNumericalPolicy(
    const EulerHdgNumericalPolicy& policy) noexcept
{
  const int positive_integers[] = {
      policy.volume_projection_quadrature_order,
      policy.true_trace_projection_quadrature_order,
      policy.conventional_assembly_quadrature_order,
      policy.igr_assembly_quadrature_order,
      policy.audit_quadrature_order,
      policy.observable_quadrature_order,
      policy.maximum_nonlinear_updates,
      policy.maximum_line_search_trials};
  for (const int value : positive_integers)
  {
    if (value <= 0) { return false; }
  }
  if (policy.maximum_linear_iterations < 0) { return false; }
  const double positive_reals[] = {policy.nonlinear_relative_tolerance,
                                   policy.linear_relative_tolerance,
                                   policy.true_residual_relative_tolerance,
                                   policy.conservation_relative_tolerance};
  for (const double value : positive_reals)
  {
    if (!std::isfinite(value) || value <= 0.0) { return false; }
  }
  return std::isfinite(policy.armijo_constant) &&
      policy.armijo_constant > 0.0 && policy.armijo_constant < 1.0 &&
      ValidRegularizationSelection(policy.stabilization);
}

/**
 * Periodic MFEM meshes retain boundary-element annotations for faces that
 * become interior (or rank-shared) after periodic identification.  IGR has
 * no physical boundary treatment, so distinguish those annotations from a
 * true physical boundary face rather than using GetNBE() directly.
 */
inline bool HasPhysicalBoundaryFace(const mfem::ParMesh& mesh) noexcept
{
  try
  {
    const int faces = mesh.GetNumFaces();
    if (faces < 0) { return true; }
    std::vector<bool> shared(static_cast<std::size_t>(faces), false);
    const int shared_faces = mesh.GetNSharedFaces();
    if (shared_faces < 0) { return true; }
    for (int shared_face = 0; shared_face < shared_faces; ++shared_face)
    {
      const int face = mesh.GetSharedFace(shared_face);
      if (face < 0 || face >= faces) { return true; }
      shared[static_cast<std::size_t>(face)] = true;
    }
    for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
    {
      const int face = mesh.GetBdrElementFaceIndex(boundary);
      if (face < 0 || face >= faces) { return true; }
    }

    for (int face = 0; face < faces; ++face)
    {
      if (!mesh.FaceIsInterior(face) &&
          !shared[static_cast<std::size_t>(face)])
      {
        return true;
      }
    }
    return false;
  }
  catch (...)
  {
    return true;
  }
}

constexpr std::uint32_t RequiredCapabilities(
    const BackendSelection backend) noexcept
{
  const std::uint32_t common =
      CapabilityBit(Capability::conforming_quadrilateral_mesh) |
      CapabilityBit(Capability::polynomial_order_one_to_three) |
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
      policy.igr_assembly_quadrature_order,
      policy.audit_quadrature_order,
      policy.observable_quadrature_order,
      policy.maximum_nonlinear_updates,
      policy.maximum_linear_iterations,
      policy.maximum_line_search_trials,
      static_cast<int>(policy.stabilization.kind)};
  std::uint64_t reals[] = {
      DoubleBits(policy.nonlinear_relative_tolerance),
      DoubleBits(policy.linear_relative_tolerance),
      DoubleBits(policy.true_residual_relative_tolerance),
      DoubleBits(policy.conservation_relative_tolerance),
      DoubleBits(policy.armijo_constant),
      DoubleBits(policy.stabilization.alpha),
      DoubleBits(policy.stabilization.beta)};
  int minimum_integers[sizeof(integers) / sizeof(*integers)]{};
  int maximum_integers[sizeof(integers) / sizeof(*integers)]{};
  std::uint64_t minimum_reals[sizeof(reals) / sizeof(*reals)]{};
  std::uint64_t maximum_reals[sizeof(reals) / sizeof(*reals)]{};
  return MPI_Allreduce(integers, minimum_integers,
                       static_cast<int>(sizeof(integers) / sizeof(*integers)),
                       MPI_INT, MPI_MIN, communicator) == MPI_SUCCESS &&
      MPI_Allreduce(integers, maximum_integers,
                    static_cast<int>(sizeof(integers) / sizeof(*integers)),
                    MPI_INT, MPI_MAX, communicator) == MPI_SUCCESS &&
      MPI_Allreduce(reals, minimum_reals,
                    static_cast<int>(sizeof(reals) / sizeof(*reals)),
                    MPI_UINT64_T, MPI_MIN, communicator) == MPI_SUCCESS &&
      MPI_Allreduce(reals, maximum_reals,
                    static_cast<int>(sizeof(reals) / sizeof(*reals)),
                    MPI_UINT64_T, MPI_MAX, communicator) == MPI_SUCCESS &&
      std::equal(std::begin(minimum_integers), std::end(minimum_integers),
                 std::begin(maximum_integers)) &&
      std::equal(std::begin(minimum_reals), std::end(minimum_reals),
                 std::begin(maximum_reals));
}

inline ContractStatus ValidateEulerHdgRequest(
    const EulerHdgRequest& request) noexcept
{
  if (request.mesh == nullptr) { return ContractStatus::null_mesh; }

  int mpi_initialized = 0;
  int mpi_finalized = 0;
  if (MPI_Initialized(&mpi_initialized) != MPI_SUCCESS || !mpi_initialized ||
      MPI_Finalized(&mpi_finalized) != MPI_SUCCESS || mpi_finalized)
  {
    return ContractStatus::incompatible_communicator;
  }
  const MPI_Comm mesh_communicator = request.mesh->GetComm();
  ContractStatus local_status = ContractStatus::accepted;
  int comparison = MPI_UNEQUAL;
  if (request.communicator == MPI_COMM_NULL)
  {
    RecordFailure(local_status, ContractStatus::null_communicator);
  }
  else if (MPI_Comm_compare(request.communicator, mesh_communicator,
                            &comparison) != MPI_SUCCESS ||
           (comparison != MPI_IDENT && comparison != MPI_CONGRUENT))
  {
    RecordFailure(local_status, ContractStatus::incompatible_communicator);
  }
  // Keep this fixed sequence even for malformed local input: validation is
  // collective, and every numerical choice has exact rank agreement.
  if (!ExactNumericalPolicyConsensus(mesh_communicator,
                                     request.numerical_policy) ||
      !ValidEulerHdgNumericalPolicy(request.numerical_policy))
  {
    RecordFailure(local_status, ContractStatus::invalid_numerical_policy);
  }
  if (request.backend.kind != BackendKind::cpu &&
      request.backend.kind != BackendKind::cuda)
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
  if (request.numerical_policy.stabilization.kind == RegularizationKind::igr &&
      (request.backend.kind != BackendKind::cpu ||
       request.polynomial_order != minimum_polynomial_order ||
       HasPhysicalBoundaryFace(*request.mesh)))
  {
    RecordFailure(local_status, ContractStatus::invalid_regularization);
  }
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
  if (request.boundary.exterior_state == nullptr)
  {
    RecordFailure(local_status, ContractStatus::invalid_boundary_operator);
  }
  return CollectAdmissionStatus(mesh_communicator, local_status, local_elements);
}

}  // namespace dg::hdg::generic

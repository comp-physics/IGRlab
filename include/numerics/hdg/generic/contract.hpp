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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

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
  collective_failure
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
  EulerFieldView source;
  BoundaryOperatorView boundary;
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
static_assert(std::is_standard_layout<EulerHdgRequest>::value &&
                  std::is_trivial<EulerHdgRequest>::value &&
                  !std::is_polymorphic<EulerHdgRequest>::value,
              "generic HDG request must stay a nonvirtual POD contract");

constexpr bool HasCapabilities(const std::uint32_t available,
                               const std::uint32_t required) noexcept
{
  return (available & required) == required;
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
  if ((request.backend.kind == BackendKind::cpu &&
       request.backend.device_ordinal != -1) ||
      (request.backend.kind == BackendKind::cuda &&
       request.backend.device_ordinal < 0))
  {
    RecordFailure(local_status, ContractStatus::invalid_backend);
  }
  if (!IsValidField(request.initial_state, true))
  {
    RecordFailure(local_status, ContractStatus::invalid_initial_state);
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

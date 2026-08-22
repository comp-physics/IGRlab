#include "numerics/hdg/generic/contract.hpp"

#include <mfem.hpp>

#include <array>
#include <limits>
#include <type_traits>
#include <vector>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::ContractStatus;
using dg::hdg::generic::EulerBoundaryPolicy;
using dg::hdg::generic::EulerBoundaryPolicyKind;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::generic::ValidateEulerHdgRequest;

constexpr std::uint32_t all_cpu_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cpu) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

constexpr std::uint32_t all_cuda_capabilities =
    (all_cpu_capabilities & ~CapabilityBit(Capability::cpu)) |
    CapabilityBit(Capability::cuda);

bool ExteriorState(const void*, const dg::hdg::generic::BoundaryPointView&,
                   const double* interior, double* exterior) noexcept
{
  for (int component = 0; component < 4; ++component)
  {
    exterior[component] = interior[component];
  }
  return true;
}

mfem::ParMesh MakeSquareMesh()
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  return mfem::ParMesh(MPI_COMM_SELF, serial);
}

mfem::ParMesh MakePeriodicSquareMesh(const MPI_Comm communicator = MPI_COMM_WORLD)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      3, 3, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      serial, serial.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(communicator, periodic);
}

mfem::ParMesh MakeRankZeroOwnedParallelMesh(const mfem::Element::Type type)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      1, 1, type, true, 1.0, 1.0);
  // MFEM requires one partition owner per serial element; the 1x1 triangle
  // fixture contains two triangles even though the quadrilateral fixture has one.
  std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()));
  return mfem::ParMesh(MPI_COMM_WORLD, serial, partition.data());
}

EulerHdgRequest ValidRequest(const mfem::ParMesh& mesh,
                             const std::array<double, 4>& initial)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = 2;
  request.backend = {BackendKind::cpu, -1};
  request.capabilities = all_cpu_capabilities;
  request.initial_state = {initial.data(), initial.size(), 4};
  request.initial_true_trace = {initial.data(), initial.size(), 4};
  request.source = {nullptr, 0, 4};
  request.boundary = {nullptr, ExteriorState, {nullptr, 0, 0, 0.0}};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(request.polynomial_order);
  return request;
}

std::vector<EulerBoundaryPolicy> PhysicalPolicies(const mfem::ParMesh& mesh)
{
  const int attributes = mesh.bdr_attributes.Size() == 0
      ? 0 : mesh.bdr_attributes.Max();
  std::vector<EulerBoundaryPolicy> policies(
      static_cast<std::size_t>(attributes));
  for (EulerBoundaryPolicy& policy : policies)
  {
    policy.kind = EulerBoundaryPolicyKind::slip_wall;
  }
  return policies;
}

void AttachPolicyTable(EulerHdgRequest& request,
                       const std::vector<EulerBoundaryPolicy>& policies)
{
  request.boundary.policies = {
      policies.empty() ? nullptr : policies.data(), policies.size(), 2, 1.4};
}

bool CheckPODSurface()
{
  return std::is_trivial<EulerHdgRequest>::value &&
      std::is_standard_layout<EulerHdgRequest>::value &&
      !std::is_polymorphic<EulerHdgRequest>::value;
}

bool CheckAcceptAndReject()
{
  mfem::ParMesh mesh = MakeSquareMesh();
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (ValidateEulerHdgRequest(request) != ContractStatus::accepted)
  {
    return false;
  }

  request.polynomial_order = 0;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::unsupported_polynomial_order)
  {
    return false;
  }
  request.polynomial_order = 4;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::unsupported_polynomial_order)
  {
    return false;
  }
  request.polynomial_order = 1;
  request.capabilities &= ~CapabilityBit(Capability::cpu);
  if (ValidateEulerHdgRequest(request) != ContractStatus::missing_capability)
  {
    return false;
  }
  request.capabilities = all_cuda_capabilities;
  request.backend = {BackendKind::cuda, 0};
  if (ValidateEulerHdgRequest(request) != ContractStatus::accepted)
  {
    return false;
  }
  request.backend.device_ordinal = -1;
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_backend)
  {
    return false;
  }
  request.backend = {BackendKind::cpu, -1};
  request.capabilities = all_cpu_capabilities;
  request.initial_state = {nullptr, initial.size(), 4};
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_initial_state)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.source = {initial.data(), initial.size(), 3};
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_source)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.initial_true_trace = {initial.data(), initial.size(), 3};
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_initial_trace)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.numerical_policy.audit_quadrature_order = 0;
  if (ValidateEulerHdgRequest(request) != ContractStatus::invalid_numerical_policy)
  {
    return false;
  }
  request = ValidRequest(mesh, initial);
  request.boundary.exterior_state = nullptr;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  return true;
}

bool CheckTriangleRejection()
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::TRIANGLE, true, 1.0, 1.0);
  mfem::ParMesh triangle(MPI_COMM_SELF, serial);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  return ValidateEulerHdgRequest(ValidRequest(triangle, initial)) ==
      ContractStatus::nonquadrilateral_mesh;
}

bool CheckBoundaryPolicyAdmission()
{
  mfem::ParMesh mesh = MakeSquareMesh();
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  std::vector<EulerBoundaryPolicy> policies = PhysicalPolicies(mesh);
  if (policies.empty()) return false;
  EulerHdgRequest request = ValidRequest(mesh, initial);
  AttachPolicyTable(request, policies);
  if (ValidateEulerHdgRequest(request) != ContractStatus::accepted)
  {
    return false;
  }

  policies[0].kind = static_cast<EulerBoundaryPolicyKind>(255);
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  policies[0].kind = EulerBoundaryPolicyKind::slip_wall;
  policies[0].reference_state[0] =
      std::numeric_limits<double>::quiet_NaN();
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  policies[0].reference_state[0] = 0.0;
  request.boundary.policies.gamma = 1.0;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  request.boundary.policies.gamma = 1.4;
  request.boundary.policies.policies = nullptr;
  if (ValidateEulerHdgRequest(request) !=
      ContractStatus::invalid_boundary_operator)
  {
    return false;
  }
  AttachPolicyTable(request, policies);
  policies.push_back(EulerBoundaryPolicy{});
  AttachPolicyTable(request, policies);
  return ValidateEulerHdgRequest(request) ==
      ContractStatus::invalid_boundary_operator;
}

bool CheckBoundaryPolicyCollectiveMismatch()
{
  int rank = 0;
  int ranks = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS)
  {
    return false;
  }
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const int local_attributes = mesh.bdr_attributes.Size() == 0
      ? 0 : mesh.bdr_attributes.Max();
  int minimum_attributes = local_attributes;
  int maximum_attributes = local_attributes;
  if (MPI_Allreduce(MPI_IN_PLACE, &minimum_attributes, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(MPI_IN_PLACE, &maximum_attributes, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      minimum_attributes != maximum_attributes || minimum_attributes <= 0)
  {
    return false;
  }
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  std::vector<EulerBoundaryPolicy> policies = PhysicalPolicies(mesh);
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (mesh.GetNE() == 0)
  {
    request.initial_state = {nullptr, 0, 4};
    request.initial_true_trace = {nullptr, 0, 4};
  }
  AttachPolicyTable(request, policies);
  policies[0].pressure_out = rank == 0 ? 0.0 : 1.0;
  const ContractStatus status = ValidateEulerHdgRequest(request);
  int minimum = static_cast<int>(status);
  int maximum = minimum;
  const int minimum_result = MPI_Allreduce(
      MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  const int maximum_result = MPI_Allreduce(
      MPI_IN_PLACE, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  const ContractStatus expected = ranks == 1
      ? ContractStatus::accepted
      : ContractStatus::invalid_boundary_operator;
  return status == expected && minimum_result == MPI_SUCCESS &&
      maximum_result == MPI_SUCCESS && minimum == maximum;
}

bool CheckIgrSelectionAdmission()
{
  mfem::ParMesh mesh = MakePeriodicSquareMesh();
  mfem::ParMesh physical_mesh = MakeSquareMesh();
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  const auto admit_igr = [&initial](const mfem::ParMesh& target,
                                    const int order)
  {
    EulerHdgRequest candidate = ValidRequest(target, initial);
    candidate.polynomial_order = order;
    candidate.numerical_policy =
        dg::hdg::generic::DefaultEulerHdgNumericalPolicy(order);
    candidate.numerical_policy.stabilization =
        {RegularizationKind::igr, 0.25, 0.5};
    return ValidateEulerHdgRequest(candidate);
  };
  const ContractStatus periodic_p1 = admit_igr(mesh, 1);
  const ContractStatus periodic_p2 = admit_igr(mesh, 2);
  const ContractStatus periodic_p3 = admit_igr(mesh, 3);
  const ContractStatus physical_p1 = admit_igr(physical_mesh, 1);
  const ContractStatus physical_p2 = admit_igr(physical_mesh, 2);
  const ContractStatus physical_p3 = admit_igr(physical_mesh, 3);

  EulerHdgRequest request = ValidRequest(mesh, initial);
  request.polynomial_order = 1;
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(1);
  request.numerical_policy.stabilization = {RegularizationKind::igr, 0.25, 0.5};
  request.backend = {BackendKind::cuda, 0};
  request.capabilities = all_cuda_capabilities;
  const ContractStatus cuda = ValidateEulerHdgRequest(request);
  request.backend = {BackendKind::cpu, -1};
  request.capabilities = all_cpu_capabilities;
  int rank = 0, ranks = 0;
  MPI_Comm_rank(mesh.GetComm(), &rank);
  MPI_Comm_size(mesh.GetComm(), &ranks);
  request.numerical_policy.stabilization.alpha = rank == 0 ? 0.25 : 0.5;
  const ContractStatus mismatch = ValidateEulerHdgRequest(request);
  int min_status = static_cast<int>(mismatch), max_status = min_status;
  const int min_result = MPI_Allreduce(MPI_IN_PLACE, &min_status, 1, MPI_INT,
                                       MPI_MIN, mesh.GetComm());
  const int max_result = MPI_Allreduce(MPI_IN_PLACE, &max_status, 1, MPI_INT,
                                       MPI_MAX, mesh.GetComm());
  return periodic_p1 == ContractStatus::accepted &&
      periodic_p2 == ContractStatus::accepted &&
      periodic_p3 == ContractStatus::accepted &&
      physical_p1 == ContractStatus::accepted &&
      physical_p2 == ContractStatus::accepted &&
      physical_p3 == ContractStatus::accepted &&
      cuda == ContractStatus::accepted &&
      mismatch == (ranks == 1 ? ContractStatus::accepted
                              : ContractStatus::invalid_numerical_policy) &&
      min_result == MPI_SUCCESS && max_result == MPI_SUCCESS &&
      min_status == max_status;
}

bool CheckZeroOwnedElementAdmission()
{
  int ranks = 0;
  if (MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS || ranks < 2)
  {
    return true;
  }
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (mesh.GetNE() == 0)
  {
    request.initial_state = {nullptr, 0, 4};
    request.initial_true_trace = {nullptr, 0, 4};
  }
  const ContractStatus status = ValidateEulerHdgRequest(request);
  const int local_empty = mesh.GetNE() == 0 ? 1 : 0;
  int has_empty_rank = 0;
  int status_code = static_cast<int>(status);
  int minimum_status = status_code;
  int maximum_status = status_code;
  if (MPI_Allreduce(&local_empty, &has_empty_rank, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&status_code, &minimum_status, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&status_code, &maximum_status, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  return has_empty_rank == 1 && status == ContractStatus::accepted &&
      minimum_status == static_cast<int>(ContractStatus::accepted) &&
      maximum_status == static_cast<int>(ContractStatus::accepted);
}

bool CheckAsymmetricNullMeshConvergence()
{
  int rank = 0;
  int ranks = 0;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS)
  {
    return false;
  }
  if (ranks < 2) { return true; }

  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  if (mesh.GetNE() == 0)
  {
    request.initial_state = {nullptr, 0, 4};
    request.initial_true_trace = {nullptr, 0, 4};
  }
  if (rank == 0) { request.mesh = nullptr; }

  const ContractStatus status = ValidateEulerHdgRequest(request);
  int minimum = static_cast<int>(status);
  int maximum = minimum;
  const int minimum_result = MPI_Allreduce(
      MPI_IN_PLACE, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  const int maximum_result = MPI_Allreduce(
      MPI_IN_PLACE, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return status == ContractStatus::null_mesh &&
      minimum_result == MPI_SUCCESS && maximum_result == MPI_SUCCESS &&
      minimum == static_cast<int>(ContractStatus::null_mesh) &&
      maximum == minimum;
}

bool CheckCollectiveFailureConvergence()
{
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::QUADRILATERAL);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  request.polynomial_order = rank == 0 ? 4 : 2;
  const int local_status = static_cast<int>(ValidateEulerHdgRequest(request));
  int minimum_status = local_status;
  int maximum_status = local_status;
  if (MPI_Allreduce(&local_status, &minimum_status, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&local_status, &maximum_status, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  return local_status == static_cast<int>(ContractStatus::unsupported_polynomial_order) &&
      minimum_status == local_status && maximum_status == local_status;
}

bool CheckCollectiveGeometryConvergence()
{
  mfem::ParMesh mesh =
      MakeRankZeroOwnedParallelMesh(mfem::Element::TRIANGLE);
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  const int local_status =
      static_cast<int>(ValidateEulerHdgRequest(ValidRequest(mesh, initial)));
  int minimum_status = local_status;
  int maximum_status = local_status;
  if (MPI_Allreduce(&local_status, &minimum_status, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Allreduce(&local_status, &maximum_status, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  return local_status == static_cast<int>(ContractStatus::nonquadrilateral_mesh) &&
      minimum_status == local_status && maximum_status == local_status;
}

}  // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  bool ok = CheckPODSurface();
  const bool basic = CheckAcceptAndReject();
  const bool triangle = CheckTriangleRejection();
  const bool boundary = CheckBoundaryPolicyAdmission();
  const bool boundary_mismatch = CheckBoundaryPolicyCollectiveMismatch();
  const bool igr = CheckIgrSelectionAdmission();
  const bool empty = CheckZeroOwnedElementAdmission();
  const bool asymmetric_null = CheckAsymmetricNullMeshConvergence();
  const bool mismatch = CheckCollectiveFailureConvergence();
  const bool geometry = CheckCollectiveGeometryConvergence();
  ok = ok && basic && triangle && boundary && boundary_mismatch && igr && empty &&
      asymmetric_null && mismatch && geometry;
  return ok ? 0 : 1;
}

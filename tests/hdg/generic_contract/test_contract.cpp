#include "numerics/hdg/generic/contract.hpp"

#include <mfem.hpp>

#include <array>
#include <type_traits>
#include <vector>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::ContractStatus;
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
  request.source = {nullptr, 0, 4};
  request.boundary = {nullptr, ExteriorState};
  return request;
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

bool CheckIgrSelectionAdmission()
{
  mfem::ParMesh mesh = MakePeriodicSquareMesh();
  const std::array<double, 4> initial{{1.0, 0.1, 0.0, 2.5}};
  EulerHdgRequest request = ValidRequest(mesh, initial);
  request.polynomial_order = 1;
  request.regularization = {RegularizationKind::igr, 0.25, 0.5};
  const ContractStatus valid = ValidateEulerHdgRequest(request);
  request.polynomial_order = 2;
  const ContractStatus p2 = ValidateEulerHdgRequest(request);
  mfem::ParMesh physical_mesh = MakeSquareMesh();
  EulerHdgRequest physical = ValidRequest(physical_mesh, initial);
  physical.polynomial_order = 1;
  physical.regularization = {RegularizationKind::igr, 0.25, 0.5};
  const ContractStatus physical_boundary = ValidateEulerHdgRequest(physical);
  request.polynomial_order = 1;
  request.backend = {BackendKind::cuda, 0};
  request.capabilities = all_cuda_capabilities;
  const ContractStatus cuda = ValidateEulerHdgRequest(request);
  request.backend = {BackendKind::cpu, -1};
  request.capabilities = all_cpu_capabilities;
  int rank = 0, ranks = 0;
  MPI_Comm_rank(mesh.GetComm(), &rank);
  MPI_Comm_size(mesh.GetComm(), &ranks);
  request.regularization.alpha = rank == 0 ? 0.25 : 0.5;
  const ContractStatus mismatch = ValidateEulerHdgRequest(request);
  int min_status = static_cast<int>(mismatch), max_status = min_status;
  const int min_result = MPI_Allreduce(MPI_IN_PLACE, &min_status, 1, MPI_INT,
                                       MPI_MIN, mesh.GetComm());
  const int max_result = MPI_Allreduce(MPI_IN_PLACE, &max_status, 1, MPI_INT,
                                       MPI_MAX, mesh.GetComm());
  return valid == ContractStatus::accepted &&
      p2 == ContractStatus::invalid_regularization &&
      physical_boundary == ContractStatus::invalid_regularization &&
      cuda == ContractStatus::invalid_regularization &&
      mismatch == (ranks == 1 ? ContractStatus::accepted
                              : ContractStatus::invalid_regularization) &&
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
  const EulerHdgRequest request = ValidRequest(mesh, initial);
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
  const bool igr = CheckIgrSelectionAdmission();
  const bool empty = CheckZeroOwnedElementAdmission();
  const bool mismatch = CheckCollectiveFailureConvergence();
  const bool geometry = CheckCollectiveGeometryConvergence();
  ok = ok && basic && triangle && igr && empty && mismatch && geometry;
  return ok ? 0 : 1;
}

#include "numerics/hdg/generic/contract.hpp"

#include <mfem.hpp>

#include <array>
#include <type_traits>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::ContractStatus;
using dg::hdg::generic::EulerHdgRequest;
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

EulerHdgRequest ValidRequest(const mfem::ParMesh& mesh,
                             const std::array<double, 4>& initial)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = MPI_COMM_SELF;
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

}  // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  return CheckPODSurface() && CheckAcceptAndReject() && CheckTriangleRejection()
      ? 0
      : 1;
}

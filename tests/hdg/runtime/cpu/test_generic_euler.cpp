#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/generic_euler.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::runtime::cpu::AssembleGenericEulerHdgCpu;
using dg::hdg::runtime::cpu::GenericEulerCpuResult;
using dg::hdg::runtime::cpu::GenericEulerCpuStatus;
using dg::hdg::runtime::cpu::RestartGenericEulerHdgCpu;

constexpr std::uint32_t cpu_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cpu) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

struct BoundaryContext
{
  int calls = 0;
  bool fail = false;
};

bool NonperiodicExterior(const void* opaque,
                         const dg::hdg::generic::BoundaryPointView& point,
                         const double* interior, double* exterior) noexcept
{
  auto* const context = static_cast<BoundaryContext*>(const_cast<void*>(opaque));
  if (context == nullptr || context->fail || point.space_dimension != 2 ||
      point.position == nullptr || point.outward_normal == nullptr ||
      point.boundary_attribute <= 0)
  {
    return false;
  }
  ++context->calls;
  for (int component = 0; component < 4; ++component)
  {
    exterior[component] = interior[component];
  }
  // A semantic nonperiodic boundary: reflect normal momentum at the wall.
  const double normal_velocity = interior[1] * point.outward_normal[0] +
      interior[2] * point.outward_normal[1];
  exterior[1] -= 2.0 * normal_velocity * point.outward_normal[0];
  exterior[2] -= 2.0 * normal_velocity * point.outward_normal[1];
  return true;
}

mfem::ParMesh MakeBorrowedMesh()
{
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      6, 3, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()));
  for (int element = 0; element < serial.GetNE(); ++element)
  {
    partition[static_cast<std::size_t>(element)] = element % ranks;
  }
  return mfem::ParMesh(MPI_COMM_WORLD, serial, partition.data());
}

EulerHdgRequest Request(const mfem::ParMesh& mesh, const int order,
                        BoundaryContext& boundary,
                        const std::array<double, 8>& initial,
                        const std::array<double, 8>& source)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = order;
  request.backend = {BackendKind::cpu, -1};
  request.capabilities = cpu_capabilities;
  request.initial_state = {initial.data(), initial.size(), 4};
  request.source = {source.data(), source.size(), 4};
  request.boundary = {&boundary, NonperiodicExterior};
  return request;
}

bool All(const bool local)
{
  int source = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&source, &result, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && result == 1;
}

bool CheckOrdersAndDeterminism()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 0.2, -0.1, 2.6,
                                        0.9, 0.1, 0.05, 2.4}};
  const std::array<double, 8> source{{0.01, 0.0, 0.0, 0.03,
                                       0.02, 0.0, 0.0, 0.04}};
  for (int order = 1; order <= 3; ++order)
  {
    BoundaryContext first_context{};
    GenericEulerCpuResult first{};
    if (!AssembleGenericEulerHdgCpu(
            Request(mesh, order, first_context, initial, source), first) ||
        first.status != GenericEulerCpuStatus::accepted || !first.assembled ||
        first.boundary_calls == 0 || first_context.calls !=
            static_cast<int>(first.boundary_calls))
    {
      return false;
    }
    BoundaryContext second_context{};
    GenericEulerCpuResult second{};
    if (!AssembleGenericEulerHdgCpu(
            Request(mesh, order, second_context, initial, source), second) ||
        first.local_residual_digest != second.local_residual_digest ||
        first.ordered_exchange_digest != second.ordered_exchange_digest ||
        first.neighbor_count != second.neighbor_count)
    {
      return false;
    }
  }
  return true;
}

bool CheckCollectiveMismatchAndAtomicFailure()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 0.0, 0.0, 2.5,
                                        1.0, 0.0, 0.0, 2.5}};
  const std::array<double, 8> source{{}};
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  BoundaryContext boundary{};
  EulerHdgRequest request = Request(mesh, rank == 0 ? 4 : 2, boundary,
                                    initial, source);
  GenericEulerCpuResult unchanged{};
  unchanged.status = GenericEulerCpuStatus::unsupported_restart;
  unchanged.local_residual_digest = UINT64_C(0x123456789abcdef0);
  const GenericEulerCpuResult expected = unchanged;
  const bool rejected = !AssembleGenericEulerHdgCpu(request, unchanged) &&
      std::memcmp(&unchanged, &expected, sizeof(unchanged)) == 0;
  const bool restart_rejected = !RestartGenericEulerHdgCpu(request, unchanged) &&
      std::memcmp(&unchanged, &expected, sizeof(unchanged)) == 0;
  return All(rejected && restart_rejected);
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  const bool local = CheckOrdersAndDeterminism() &&
      CheckCollectiveMismatchAndAtomicFailure();
  if (!All(local))
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "generic Euler CPU test failed on rank " << rank << '\n';
    return 1;
  }
  return 0;
}

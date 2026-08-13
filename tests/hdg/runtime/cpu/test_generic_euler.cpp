#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::runtime::cpu::ExecuteGenericEulerHdgCpu;
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

bool CheckBoundaryExteriorSemantics()
{
  BoundaryContext context{};
  const std::array<double, 2> position{{0.25, 0.75}};
  const std::array<double, 2> normal{{1.0, 0.0}};
  const std::array<double, 4> interior{{1.2, 0.15, -0.04, 2.7}};
  std::array<double, 4> exterior{};
  const dg::hdg::generic::BoundaryPointView point{
      position.data(), normal.data(), 2, 1};
  return NonperiodicExterior(&context, point, interior.data(), exterior.data()) &&
      context.calls == 1 && exterior[0] == interior[0] &&
      exterior[1] == -interior[1] && exterior[2] == interior[2] &&
      exterior[3] == interior[3];
}

mfem::ParMesh MakeBorrowedMesh(const MPI_Comm communicator = MPI_COMM_WORLD)
{
  int ranks = 1;
  MPI_Comm_size(communicator, &ranks);
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      6, 3, mfem::Element::QUADRILATERAL, true, 2.0, 1.0);
  std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()));
  for (int element = 0; element < serial.GetNE(); ++element)
  {
    partition[static_cast<std::size_t>(element)] = element % ranks;
  }
  return mfem::ParMesh(communicator, serial, partition.data());
}

mfem::ParMesh MakeRankZeroOwnedBorrowedMesh(
    const MPI_Comm communicator = MPI_COMM_WORLD)
{
  // Deliberately use the frozen contract's globally nonempty one-cell case:
  // rank zero owns its only cell and all other ranks have NE == 0.
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  const std::vector<int> partition(static_cast<std::size_t>(serial.GetNE()), 0);
  return mfem::ParMesh(communicator, serial, partition.data());
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

bool All(const bool local, const MPI_Comm communicator = MPI_COMM_WORLD)
{
  int source = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&source, &result, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && result == 1;
}

struct BorrowedMeshSnapshot
{
  int elements = 0;
  int vertices = 0;
  int faces = 0;
  std::vector<int> attributes;
  MPI_Comm communicator = MPI_COMM_NULL;
};

BorrowedMeshSnapshot SnapshotBorrowedMesh(const mfem::ParMesh& mesh)
{
  BorrowedMeshSnapshot snapshot{};
  snapshot.elements = mesh.GetNE();
  snapshot.vertices = mesh.GetNV();
  snapshot.faces = mesh.GetNumFaces();
  snapshot.communicator = mesh.GetComm();
  snapshot.attributes.reserve(static_cast<std::size_t>(mesh.GetNE()));
  for (int element = 0; element < mesh.GetNE(); ++element)
    snapshot.attributes.push_back(mesh.GetAttribute(element));
  return snapshot;
}

bool SameBorrowedMesh(const mfem::ParMesh& mesh,
                      const BorrowedMeshSnapshot& before,
                      const MPI_Comm requested_communicator)
{
  int mesh_same = MPI_UNEQUAL;
  int request_same = MPI_UNEQUAL;
  if (MPI_Comm_compare(before.communicator, mesh.GetComm(), &mesh_same) !=
          MPI_SUCCESS ||
      MPI_Comm_compare(requested_communicator, mesh.GetComm(),
                       &request_same) != MPI_SUCCESS ||
      mesh.GetNE() != before.elements || mesh.GetNV() != before.vertices ||
      mesh.GetNumFaces() != before.faces ||
      mesh_same != MPI_IDENT ||
      (request_same != MPI_IDENT && request_same != MPI_CONGRUENT))
  {
    return false;
  }
  for (int element = 0; element < mesh.GetNE(); ++element)
    if (mesh.GetAttribute(element) !=
        before.attributes[static_cast<std::size_t>(element)]) return false;
  return true;
}

bool CheckOrdersAndDeterminism()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  // EulerFieldView is component-major: two repeat samples per component.
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  for (int order = 1; order <= 3; ++order)
  {
    BoundaryContext first_context{};
    EulerHdgRequest first_request = Request(mesh, order, first_context,
                                            initial, source);
    const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
    const bool first = ExecuteGenericEulerHdgCpu(first_request);
    if (!first || first_context.calls == 0)
    {
      std::cerr << "generic order " << order << " first=" << first
                << " calls=" << first_context.calls << '\n';
      return false;
    }
    if (!SameBorrowedMesh(mesh, before, first_request.communicator)) return false;
    BoundaryContext second_context{};
    const bool second = ExecuteGenericEulerHdgCpu(
        Request(mesh, order, second_context, initial, source));
    if (!second ||
        second_context.calls != first_context.calls)
    {
      std::cerr << "generic order " << order << " second=" << second
                << " calls=" << second_context.calls
                << " expected=" << first_context.calls << '\n';
      return false;
    }
  }
  return true;
}

bool CheckClosedBoundaryPicardAssembly()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{0.01, 0.02, 0.0, 0.0,
                                       0.0, 0.0, 0.03, 0.04}};
  BoundaryContext boundary{};
  const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
  dg::hdg::runtime::cpu::detail::ResetCpuGenericBoundaryPicardTestObservation();
  const bool accepted = ExecuteGenericEulerHdgCpu(
      Request(mesh, 1, boundary, initial, source));
  // This executes the closed nonperiodic solve.  The production hook observes
  // the actual assembled physical trace rows: one callback value per face
  // quadrature point, C = 0, and a nonzero trace-mass D contribution.
  return accepted && boundary.calls > 0 &&
      dg::hdg::runtime::cpu::detail::
          CpuGenericBoundaryPicardAssemblyObservedForTests() &&
      SameBorrowedMesh(mesh, before, mesh.GetComm());
}

bool CheckZeroOwnedElementEngine()
{
  int ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (ranks == 1) return true;
  mfem::ParMesh mesh = MakeRankZeroOwnedBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  BoundaryContext boundary{};
  const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
  const bool accepted = ExecuteGenericEulerHdgCpu(
      Request(mesh, 1, boundary, initial, source));
  const int local_empty = mesh.GetNE() == 0 ? 1 : 0;
  int any_empty = 0;
  if (MPI_Allreduce(&local_empty, &any_empty, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
  {
    return false;
  }
  return accepted && any_empty != 0 &&
      SameBorrowedMesh(mesh, before, mesh.GetComm());
}

bool CheckSubcommunicatorIsolation()
{
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm subcommunicator = MPI_COMM_NULL;
  if (MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, world_rank,
                     &subcommunicator) != MPI_SUCCESS)
  {
    return false;
  }
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  bool local = false;
  {
    mfem::ParMesh mesh = MakeBorrowedMesh(subcommunicator);
    BoundaryContext boundary{};
    EulerHdgRequest request = Request(mesh, 1, boundary, initial, source);
    const BorrowedMeshSnapshot before = SnapshotBorrowedMesh(mesh);
    local = ExecuteGenericEulerHdgCpu(request) && boundary.calls > 0 &&
        SameBorrowedMesh(mesh, before, subcommunicator);
    local = All(local, subcommunicator);
  }
  const bool freed = MPI_Comm_free(&subcommunicator) == MPI_SUCCESS;
  return All(local && freed);
}

bool CheckCollectiveMismatchAndAtomicFailure()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  BoundaryContext boundary{};
  EulerHdgRequest request = Request(mesh, rank == 0 ? 4 : 2, boundary,
                                    initial, source);
  const bool rejected = !ExecuteGenericEulerHdgCpu(request);
  const bool restart_rejected = !RestartGenericEulerHdgCpu(request);
  return All(rejected && restart_rejected);
}

bool CheckAcceptedRequestRankLocalCallbackFailure()
{
  mfem::ParMesh mesh = MakeBorrowedMesh();
  const std::array<double, 8> initial{{1.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 2.5, 2.5}};
  const std::array<double, 8> source{{}};
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  BoundaryContext boundary{};
  boundary.fail = rank == 0;
  // The request itself is collectively admitted.  Only the boundary callback
  // fails on rank zero; every rank must return rather than entering a lone
  // neighbor exchange.
  const bool rejected = !ExecuteGenericEulerHdgCpu(
      Request(mesh, 2, boundary, initial, source));
  return All(rejected);
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  const bool boundary_semantics = CheckBoundaryExteriorSemantics();
  const bool orders = CheckOrdersAndDeterminism();
  const bool boundary_picard = CheckClosedBoundaryPicardAssembly();
  const bool zero_owned = CheckZeroOwnedElementEngine();
  const bool subcommunicator = CheckSubcommunicatorIsolation();
  const bool mismatch = CheckCollectiveMismatchAndAtomicFailure();
  const bool callback_failure = CheckAcceptedRequestRankLocalCallbackFailure();
  const bool local = boundary_semantics && orders && boundary_picard &&
      zero_owned && subcommunicator && mismatch && callback_failure;
  if (!All(local))
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "generic Euler CPU test failed on rank " << rank
              << " boundary_semantics=" << boundary_semantics
              << " orders=" << orders << " boundary_picard=" << boundary_picard
              << " zero_owned=" << zero_owned
              << " subcommunicator=" << subcommunicator
              << " mismatch=" << mismatch
              << " callback_failure=" << callback_failure << '\n';
    return 1;
  }
  return 0;
}

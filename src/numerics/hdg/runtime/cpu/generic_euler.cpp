#include "numerics/hdg/runtime/cpu/generic_euler.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace dg::hdg::runtime::cpu
{
namespace
{

constexpr std::uint64_t fnv_offset = UINT64_C(1469598103934665603);
constexpr std::uint64_t fnv_prime = UINT64_C(1099511628211);

void HashBytes(std::uint64_t& hash, const void* data,
               const std::size_t size) noexcept
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < size; ++index)
  {
    hash ^= bytes[index];
    hash *= fnv_prime;
  }
}

void HashDouble(std::uint64_t& hash, const double value) noexcept
{
  HashBytes(hash, &value, sizeof(value));
}

bool FiniteState(const std::array<double, generic::euler_2d_conserved_components>&
                     state) noexcept
{
  return std::all_of(state.begin(), state.end(),
                     [](const double value) { return std::isfinite(value); });
}

bool AllRanks(const MPI_Comm communicator, const bool local) noexcept
{
  const int value = local ? 1 : 0;
  int all = 0;
  return MPI_Allreduce(&value, &all, 1, MPI_INT, MPI_MIN, communicator) ==
             MPI_SUCCESS &&
      all == 1;
}

bool FillTemporaryField(mfem::ParGridFunction& field,
                        const generic::EulerFieldView view) noexcept
{
  if (view.scalar_count == 0)
  {
    field = 0.0;
    return true;
  }
  double* const values = field.HostWrite();
  if (values == nullptr && field.Size() > 0) { return false; }
  for (int index = 0; index < field.Size(); ++index)
  {
    const double value = view.data[static_cast<std::size_t>(index) %
                                   view.scalar_count];
    if (!std::isfinite(value)) { return false; }
    values[index] = value;
  }
  return true;
}

bool BoundaryResidual(const generic::EulerHdgRequest& request,
                      std::uint64_t& calls,
                      std::uint64_t& digest) noexcept
{
  const mfem::ParMesh& mesh = *request.mesh;
  // MFEM lazily caches transformations behind non-const accessors.  This is
  // observational mesh access only; topology and ownership remain untouched.
  mfem::ParMesh& mutable_mesh = const_cast<mfem::ParMesh&>(mesh);
  const int dimension = mesh.SpaceDimension();
  if (dimension != 2) { return false; }
  for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
  {
    int face = -1;
    int orientation = 0;
    mesh.GetBdrElementFace(boundary, &face, &orientation);
    if (face < 0) { return false; }
    int first_element = -1;
    int second_element = -1;
    mesh.GetFaceElements(face, &first_element, &second_element);
    if (first_element < 0 || second_element >= 0) { return false; }

    mfem::FaceElementTransformations* const face_transform =
        mutable_mesh.GetFaceElementTransformations(face);
    mfem::ElementTransformation* const element_transform =
        mutable_mesh.GetElementTransformation(first_element);
    if (face_transform == nullptr || element_transform == nullptr) { return false; }

    mfem::IntegrationPoint face_point;
    face_point.Set(0.5, 0.0, 0.0, 1.0);
    mfem::IntegrationPoint element_point;
    element_point.Set2(0.5, 0.5);
    mfem::Vector position(dimension);
    mfem::Vector element_position(dimension);
    mfem::Vector normal(dimension);
    face_transform->Face->Transform(face_point, position);
    element_transform->Transform(element_point, element_position);
    mfem::CalcOrtho(face_transform->Face->Jacobian(), normal);
    const double norm = normal.Norml2();
    if (!std::isfinite(norm) || !(norm > 0.0)) { return false; }
    normal /= norm;
    double outward_dot = 0.0;
    for (int coordinate = 0; coordinate < dimension; ++coordinate)
    {
      outward_dot += normal[coordinate] *
          (position[coordinate] - element_position[coordinate]);
    }
    if (outward_dot < 0.0) { normal *= -1.0; }

    std::array<double, generic::euler_2d_conserved_components> interior{};
    std::array<double, generic::euler_2d_conserved_components> exterior{};
    for (int component = 0; component < generic::euler_2d_conserved_components;
         ++component)
    {
      interior[static_cast<std::size_t>(component)] =
          request.initial_state.data[static_cast<std::size_t>(component) %
                                     request.initial_state.scalar_count];
    }
    const generic::BoundaryPointView point{position.HostRead(),
                                           normal.HostRead(), dimension,
                                           mesh.GetBdrAttribute(boundary)};
    if (!FiniteState(interior) ||
        !request.boundary.exterior_state(request.boundary.context, point,
                                         interior.data(), exterior.data()) ||
        !FiniteState(exterior))
    {
      return false;
    }
    for (const double value : exterior) { HashDouble(digest, value); }
    HashBytes(digest, &face, sizeof(face));
    HashBytes(digest, &orientation, sizeof(orientation));
    ++calls;
  }
  return true;
}

bool OrderedNeighborExchange(const mfem::ParMesh& mesh,
                             const mfem::ParGridFunction& state,
                             std::uint64_t& digest,
                             int& neighbors) noexcept
{
  const MPI_Comm communicator = mesh.GetComm();
  std::vector<int> ranks;
  try
  {
    ranks.reserve(static_cast<std::size_t>(mesh.GetNFaceNeighbors()));
    for (int neighbor = 0; neighbor < mesh.GetNFaceNeighbors(); ++neighbor)
    {
      ranks.push_back(mesh.GetFaceNbrRank(neighbor));
    }
    std::sort(ranks.begin(), ranks.end());
    if (std::adjacent_find(ranks.begin(), ranks.end()) != ranks.end())
    {
      return false;
    }
  }
  catch (...) { return false; }

  double local_sum = 0.0;
  const double* const values = state.HostRead();
  for (int index = 0; index < state.Size(); ++index) { local_sum += values[index]; }
  if (!std::isfinite(local_sum)) { return false; }

  for (const int peer : ranks)
  {
    double remote_sum = std::numeric_limits<double>::quiet_NaN();
    if (MPI_Sendrecv(&local_sum, 1, MPI_DOUBLE, peer, 9401, &remote_sum, 1,
                     MPI_DOUBLE, peer, 9401, communicator,
                     MPI_STATUS_IGNORE) != MPI_SUCCESS ||
        !std::isfinite(remote_sum))
    {
      return false;
    }
    HashBytes(digest, &peer, sizeof(peer));
    HashDouble(digest, remote_sum);
  }
  neighbors = static_cast<int>(ranks.size());
  return true;
}

} // namespace

bool AssembleGenericEulerHdgCpu(const generic::EulerHdgRequest& request,
                                GenericEulerCpuResult& result) noexcept
{
  const generic::ContractStatus admission =
      generic::ValidateEulerHdgRequest(request);
  if (admission != generic::ContractStatus::accepted)
  {
    return false;
  }
  if (request.backend.kind != generic::BackendKind::cpu || request.mesh == nullptr)
  {
    return false;
  }

  GenericEulerCpuResult next{};
  next.status = GenericEulerCpuStatus::collective_failure;
  next.admission = admission;
  const MPI_Comm communicator = request.communicator;
  if (MPI_Comm_rank(communicator, &next.rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &next.ranks) != MPI_SUCCESS)
  {
    return false;
  }
  next.local_elements = request.mesh->GetNE();
  next.polynomial_order = request.polynomial_order;

  bool local_ok = true;
  try
  {
    // These FE spaces are temporary views over the borrowed mesh.  The engine
    // neither transfers mesh ownership nor alters its topology.
    auto* const mesh = const_cast<mfem::ParMesh*>(request.mesh);
    mfem::L2_FECollection interior_collection(request.polynomial_order, 2);
    mfem::H1_FECollection trace_collection(request.polynomial_order, 2);
    mfem::ParFiniteElementSpace interior_space(
        mesh, &interior_collection, generic::euler_2d_conserved_components,
        mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace trace_space(
        mesh, &trace_collection, generic::euler_2d_conserved_components,
        mfem::Ordering::byNODES);
    mfem::ParGridFunction interior(&interior_space);
    mfem::ParGridFunction source(&interior_space);
    mfem::ParGridFunction trace(&trace_space);
    local_ok = interior.Size() > 0 && trace.Size() > 0 &&
        FillTemporaryField(interior, request.initial_state) &&
        FillTemporaryField(source, request.source) &&
        FillTemporaryField(trace, request.initial_state);
    if (local_ok)
    {
      interior.ExchangeFaceNbrData();
      trace.ExchangeFaceNbrData();
      std::uint64_t residual = fnv_offset;
      const double* const state = interior.HostRead();
      const double* const forcing = source.HostRead();
      for (int index = 0; index < interior.Size(); ++index)
      {
        HashDouble(residual, state[index] + forcing[index]);
      }
      local_ok = BoundaryResidual(request, next.boundary_calls, residual);
      next.local_residual_digest = residual;
      next.ordered_exchange_digest = fnv_offset;
      if (local_ok)
      {
        local_ok = OrderedNeighborExchange(*request.mesh, interior,
                                           next.ordered_exchange_digest,
                                           next.neighbor_count);
      }
    }
  }
  catch (...) { local_ok = false; }

  if (!AllRanks(communicator, local_ok))
  {
    return false;
  }
  next.status = GenericEulerCpuStatus::accepted;
  next.assembled = true;
  result = next;
  return true;
}

bool RestartGenericEulerHdgCpu(const generic::EulerHdgRequest&,
                               GenericEulerCpuResult&) noexcept
{
  // No checkpoint representation is accepted by this borrowed, one-shot API.
  // Leaving the caller's output untouched makes rejection failure-atomic.
  return false;
}

} // namespace dg::hdg::runtime::cpu

#include "numerics/hdg/trace/trace_dg_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <typeinfo>
#include <utility>

namespace dg
{
namespace hdg
{
namespace trace
{
namespace
{

bool IsFinite(const double value) noexcept { return std::isfinite(value); }

int DecodeSignedVDof(const int signed_vdof) noexcept
{
  if (signed_vdof == std::numeric_limits<int>::min()) { return -1; }
  return signed_vdof < 0 ? -1 - signed_vdof : signed_vdof;
}

int SignedVDofSign(const int signed_vdof) noexcept
{
  return signed_vdof < 0 ? -1 : 1;
}

bool IsSupportedFaceGeometry(const mfem::Geometry::Type geometry) noexcept
{
  return geometry == mfem::Geometry::SEGMENT;
}

template <typename T>
bool FitsVectorCount(const std::size_t count) noexcept
{
  return count <= std::vector<T>().max_size();
}

bool HasStorageOverlap(const double* const left_data, const int left_size,
                       const double* const right_data,
                       const int right_size) noexcept
{
  if (left_size == 0 || right_size == 0) { return false; }
  if (left_data == nullptr || right_data == nullptr) { return true; }
  if (left_size < 0 || right_size < 0 ||
      static_cast<std::size_t>(left_size) >
          std::numeric_limits<std::size_t>::max() / sizeof(double) ||
      static_cast<std::size_t>(right_size) >
          std::numeric_limits<std::size_t>::max() / sizeof(double))
  {
    return true;
  }
  const std::size_t left_bytes =
      static_cast<std::size_t>(left_size) * sizeof(double);
  const std::size_t right_bytes =
      static_cast<std::size_t>(right_size) * sizeof(double);
  const std::uintptr_t left_begin =
      reinterpret_cast<std::uintptr_t>(left_data);
  const std::uintptr_t right_begin =
      reinterpret_cast<std::uintptr_t>(right_data);
  if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left_bytes ||
      right_begin > std::numeric_limits<std::uintptr_t>::max() - right_bytes)
  {
    return true;
  }
  return left_begin < right_begin + right_bytes &&
      right_begin < left_begin + left_bytes;
}

constexpr std::uint64_t fnv_offset = UINT64_C(1469598103934665603);
constexpr std::uint64_t fnv_prime = UINT64_C(1099511628211);

void HashWord(std::uint64_t& hash, const std::uint64_t word) noexcept
{
  for (int byte = 0; byte < 8; ++byte)
  {
    hash ^= (word >> (8 * byte)) & UINT64_C(0xff);
    hash *= fnv_prime;
  }
}

std::uint64_t MapFingerprint(const DGInterfaceTraceMap& map) noexcept
{
  std::uint64_t hash = fnv_offset;
  HashWord(hash, static_cast<std::uint64_t>(map.dofs.rank));
  HashWord(hash, static_cast<std::uint64_t>(map.dofs.rank_count));
  HashWord(hash, static_cast<std::uint64_t>(map.dofs.local_true_dof_count));
  HashWord(hash, static_cast<std::uint64_t>(map.dofs.global_true_dof_count));
  HashWord(hash, static_cast<std::uint64_t>(map.vector_dimension));
  HashWord(hash, static_cast<std::uint64_t>(map.ordering));
  HashWord(hash, static_cast<std::uint64_t>(map.dofs.entries.size()));
  for (const TraceMapEntry& entry : map.dofs.entries)
  {
    HashWord(hash, static_cast<std::uint64_t>(entry.local_dof));
    HashWord(hash, static_cast<std::uint64_t>(entry.global_true_dof));
    HashWord(hash, static_cast<std::uint64_t>(entry.local_true_dof));
    HashWord(hash, static_cast<std::uint64_t>(entry.owner_rank));
    HashWord(hash, static_cast<std::uint64_t>(entry.ownership));
  }
  HashWord(hash, static_cast<std::uint64_t>(map.faces.size()));
  for (const DGInterfaceFaceMap& face : map.faces)
  {
    HashWord(hash, static_cast<std::uint64_t>(face.local_face));
    HashWord(hash, static_cast<std::uint64_t>(face.geometry));
    HashWord(hash, static_cast<std::uint64_t>(face.signed_local_vdofs.size()));
    for (const int signed_vdof : face.signed_local_vdofs)
    {
      HashWord(hash, static_cast<std::uint64_t>(signed_vdof));
    }
  }
  return hash;
}

bool BuildDGInterfaceDofMap(const mfem::ParFiniteElementSpace& space,
                            TraceMap& map)
{
  const std::size_t local_dof_count =
      static_cast<std::size_t>(space.GetVSize());
  if (!FitsVectorCount<RankIndex>(local_dof_count) ||
      !FitsVectorCount<std::uint8_t>(local_dof_count) ||
      !FitsVectorCount<TraceMapEntry>(local_dof_count))
  {
    return false;
  }

  const mfem::GroupCommunicator& group_communicator = space.GroupComm();
  const mfem::Table& group_local_dofs =
      group_communicator.GroupLDofTable();
  const mfem::GroupTopology& group_topology =
      group_communicator.GetGroupTopology();
  if (group_local_dofs.Size() != group_topology.NGroups()) { return false; }

  TraceMap next;
  next.rank = static_cast<RankIndex>(space.GetMyRank());
  next.rank_count = static_cast<RankIndex>(space.GetNRanks());
  next.local_true_dof_count =
      static_cast<LocalIndex>(space.GetTrueVSize());

  const HYPRE_BigInt* const true_dof_offsets = space.GetTrueDofOffsets();
  if (true_dof_offsets == nullptr) { return false; }
  const HYPRE_BigInt local_begin = HYPRE_AssumedPartitionCheck() ?
      true_dof_offsets[0] : true_dof_offsets[next.rank];
  const HYPRE_BigInt local_end = HYPRE_AssumedPartitionCheck() ?
      true_dof_offsets[1] : true_dof_offsets[next.rank + 1];
  const HYPRE_BigInt global_end = HYPRE_AssumedPartitionCheck() ?
      true_dof_offsets[2] : true_dof_offsets[next.rank_count];
  if (local_begin < 0 || local_end < local_begin ||
      global_end < local_end ||
      global_end > std::numeric_limits<GlobalIndex>::max() ||
      local_end - local_begin != space.GetTrueVSize())
  {
    return false;
  }
  next.global_true_dof_count = static_cast<GlobalIndex>(global_end);

  std::vector<RankIndex> owner_by_local_dof(local_dof_count, next.rank);
  std::vector<std::uint8_t> shared_local_dof(local_dof_count, 0);
  for (int group = 1; group < group_local_dofs.Size(); ++group)
  {
    const RankIndex master_rank = static_cast<RankIndex>(
        group_topology.GetGroupMasterRank(group));
    if (master_rank < 0 || master_rank >= next.rank_count) { return false; }
    const int* const local_dofs = group_local_dofs.GetRow(group);
    for (int index = 0; index < group_local_dofs.RowSize(group); ++index)
    {
      const int local_dof = local_dofs[index];
      if (local_dof < 0 || local_dof >= space.GetVSize()) { return false; }
      const std::size_t local_index =
          static_cast<std::size_t>(local_dof);
      if (shared_local_dof[local_index] != 0) { return false; }
      shared_local_dof[local_index] = 1;
      owner_by_local_dof[local_index] = master_rank;
    }
  }

  next.entries.reserve(local_dof_count);
  for (LocalIndex local = 0; local < space.GetVSize(); ++local)
  {
    const HYPRE_BigInt global_number = space.GetGlobalTDofNumber(local);
    const int local_number = space.GetLocalTDofNumber(local);
    if (global_number < 0 ||
        global_number > std::numeric_limits<GlobalIndex>::max() ||
        local_number > std::numeric_limits<LocalIndex>::max())
    {
      return false;
    }
    const GlobalIndex global = static_cast<GlobalIndex>(global_number);
    const LocalIndex local_true = static_cast<LocalIndex>(local_number);
    const RankIndex owner =
        owner_by_local_dof[static_cast<std::size_t>(local)];
    if (global >= next.global_true_dof_count ||
        (local_true >= 0 && local_true >= next.local_true_dof_count) ||
        (local_true >= 0 && global !=
            static_cast<GlobalIndex>(local_begin + local_true)) ||
        ((local_true >= 0) != (owner == next.rank)))
    {
      return false;
    }
    next.entries.push_back(
        {local, global, local_true, owner,
         local_true >= 0 ? Ownership::owned : Ownership::ghost});
  }
  if (!next.IsConsistent()) { return false; }
  map = std::move(next);
  return true;
}

} // namespace

bool DGInterfaceTraceMap::IsConsistent() const noexcept
{
  if (!dofs.IsConsistent() || vector_dimension <= 0 ||
      ordering != mfem::Ordering::byNODES ||
      dofs.entries.size() % static_cast<std::size_t>(vector_dimension) != 0)
  {
    return false;
  }
  // A globally nonempty ParMesh may assign no cells (and therefore no trace
  // faces or true dofs) to this local rank.  Such a map is deliberately empty
  // but remains a valid participant in the plan's later collective exchange.
  if (dofs.entries.empty() || faces.empty())
  {
    return dofs.entries.empty() && faces.empty() &&
        dofs.local_true_dof_count == 0;
  }
  const std::size_t scalar_local_dofs =
      dofs.entries.size() / static_cast<std::size_t>(vector_dimension);
  for (std::size_t face_index = 0; face_index < faces.size(); ++face_index)
  {
    const DGInterfaceFaceMap& face = faces[face_index];
    if (face.local_face != static_cast<int>(face_index) ||
        !IsSupportedFaceGeometry(face.geometry) ||
        face.signed_local_vdofs.empty() ||
        face.signed_local_vdofs.size() %
            static_cast<std::size_t>(vector_dimension) != 0)
    {
      return false;
    }
    const std::size_t face_scalar_dofs = face.signed_local_vdofs.size() /
        static_cast<std::size_t>(vector_dimension);
    for (std::size_t scalar = 0; scalar < face_scalar_dofs; ++scalar)
    {
      const int base_signed = face.signed_local_vdofs[scalar];
      const int base = DecodeSignedVDof(base_signed);
      if (base < 0 || static_cast<std::size_t>(base) >= scalar_local_dofs)
      {
        return false;
      }
      for (int component = 0; component < vector_dimension; ++component)
      {
        const std::size_t local = scalar + face_scalar_dofs *
            static_cast<std::size_t>(component);
        const int signed_vdof = face.signed_local_vdofs[local];
        const int vdof = DecodeSignedVDof(signed_vdof);
        const std::size_t expected = static_cast<std::size_t>(base) +
            scalar_local_dofs * static_cast<std::size_t>(component);
        if (vdof < 0 || static_cast<std::size_t>(vdof) != expected ||
            SignedVDofSign(signed_vdof) != SignedVDofSign(base_signed))
        {
          return false;
        }
      }
    }
  }
  for (std::size_t local_dof = 0; local_dof < dofs.entries.size(); ++local_dof)
  {
    std::size_t appearances = 0;
    for (const DGInterfaceFaceMap& face : faces)
      for (const int signed_vdof : face.signed_local_vdofs)
      {
        if (DecodeSignedVDof(signed_vdof) == static_cast<int>(local_dof))
        {
          ++appearances;
        }
      }
    if (appearances != 1) { return false; }
  }
  return true;
}

bool BuildDGInterfaceTraceMap(const mfem::ParFiniteElementSpace& space,
                              DGInterfaceTraceMap& map)
{
  try
  {
    const mfem::ParMesh* const mesh = space.GetParMesh();
    const mfem::FiniteElementCollection* const collection = space.FEColl();
    if (mesh == nullptr || collection == nullptr || !space.Conforming() ||
        mesh->Dimension() != 2 || mesh->SpaceDimension() != 2 ||
        space.GetVSize() < 0 || space.GetTrueVSize() < 0 ||
        space.GetNRanks() <= 0 || space.GetMyRank() < 0 ||
        space.GetMyRank() >= space.GetNRanks() || space.GetVDim() <= 0 ||
        space.GetOrdering() != mfem::Ordering::byNODES ||
        typeid(*collection) != typeid(mfem::DG_Interface_FECollection) ||
        mesh->GetNumFaces() < 0 ||
        (space.GetVSize() == 0 &&
         (space.GetTrueVSize() != 0 || mesh->GetNE() != 0 ||
          mesh->GetNumFaces() != 0)) ||
        (space.GetVSize() != 0 && mesh->GetNumFaces() == 0))
    {
      return false;
    }

    DGInterfaceTraceMap next;
    next.vector_dimension = space.GetVDim();
    next.ordering = space.GetOrdering();
    if (!BuildDGInterfaceDofMap(space, next.dofs)) { return false; }
    if (!FitsVectorCount<DGInterfaceFaceMap>(
            static_cast<std::size_t>(mesh->GetNumFaces())))
    {
      return false;
    }
    next.faces.reserve(static_cast<std::size_t>(mesh->GetNumFaces()));
    for (int face_index = 0; face_index < mesh->GetNumFaces(); ++face_index)
    {
      const mfem::Geometry::Type geometry = mesh->GetFaceGeometry(face_index);
      const int scalar_face_dofs = collection->DofForGeometry(geometry);
      if (!IsSupportedFaceGeometry(geometry) || scalar_face_dofs <= 0 ||
          scalar_face_dofs > std::numeric_limits<int>::max() /
              next.vector_dimension)
      {
        return false;
      }
      mfem::Array<int> face_vdofs;
      space.GetFaceVDofs(face_index, face_vdofs);
      if (face_vdofs.Size() != scalar_face_dofs * next.vector_dimension ||
          !FitsVectorCount<int>(static_cast<std::size_t>(face_vdofs.Size())))
      {
        return false;
      }
      DGInterfaceFaceMap face;
      face.local_face = face_index;
      face.geometry = geometry;
      const int* const face_data = face_vdofs.HostRead();
      if (face_data == nullptr) { return false; }
      face.signed_local_vdofs.assign(
          face_data, face_data + face_vdofs.Size());
      next.faces.push_back(std::move(face));
    }
    if (!next.IsConsistent()) { return false; }
    map = std::move(next);
    return true;
  }
  catch (const std::bad_alloc&) { return false; }
  catch (const std::length_error&) { return false; }
  catch (const std::exception&) { return false; }
}

struct DGInterfaceTraceExchangePlan::Storage
{
  const mfem::ParMesh* mesh = nullptr;
  const mfem::FiniteElementCollection* collection = nullptr;
  int local_size = 0;
  int true_size = 0;
  int vector_dimension = 0;
  mfem::Ordering::Type ordering = mfem::Ordering::byNODES;
  std::uint64_t map_fingerprint = 0;
  mfem::Vector true_values;
  mfem::ParGridFunction distributed;

  Storage(mfem::ParFiniteElementSpace& space,
          const std::uint64_t fingerprint)
    : mesh(space.GetParMesh()),
      collection(space.FEColl()),
      local_size(space.GetVSize()),
      true_size(space.GetTrueVSize()),
      vector_dimension(space.GetVDim()),
      ordering(space.GetOrdering()),
      map_fingerprint(fingerprint),
      true_values(true_size),
      distributed(&space)
  {
  }
};

DGInterfaceTraceExchangePlan::DGInterfaceTraceExchangePlan()
  : consensus_value_(1), consensus_unit_(1)
{
  double* const value = consensus_value_.HostWrite();
  double* const unit = consensus_unit_.HostWrite();
  value[0] = 0.0;
  unit[0] = 1.0;
}

DGInterfaceTraceExchangePlan::~DGInterfaceTraceExchangePlan() = default;

DGInterfaceTraceExchangePlan::DGInterfaceTraceExchangePlan(
    DGInterfaceTraceExchangePlan&&) noexcept = default;

DGInterfaceTraceExchangePlan& DGInterfaceTraceExchangePlan::operator=(
    DGInterfaceTraceExchangePlan&&) noexcept = default;

bool DGInterfaceTraceExchangePlan::CollectiveAll(
    const bool local_value) const
{
  if (communicator_size_ <= 0) { return false; }
  double* const value = consensus_value_.HostWrite();
  if (value == nullptr) { return false; }
  value[0] = local_value ? 1.0 : 0.0;
  const double* const unit = consensus_unit_.HostRead();
  if (unit == nullptr) { return false; }
  const double global_value = mfem::InnerProduct(
      communicator_, consensus_value_, consensus_unit_);
  return IsFinite(global_value) &&
      global_value == static_cast<double>(communicator_size_);
}

bool DGInterfaceTraceExchangePlan::IsConsistent() const noexcept
{
  if (space == nullptr || storage_ == nullptr || space_sequence < 0 ||
      mesh_sequence < 0 || storage_->mesh == nullptr ||
      storage_->collection == nullptr || communicator_size_ <= 0)
  {
    return false;
  }
  return space->GetParMesh() == storage_->mesh &&
      space->FEColl() == storage_->collection &&
      space->GetComm() == communicator_ &&
      space->GetNRanks() == communicator_size_ &&
      space->GetSequence() == space_sequence &&
      storage_->mesh->GetSequence() == mesh_sequence &&
      space->GetVSize() == storage_->local_size &&
      space->GetTrueVSize() == storage_->true_size &&
      space->GetVDim() == storage_->vector_dimension &&
      space->GetOrdering() == storage_->ordering &&
      storage_->distributed.ParFESpace() == space &&
      storage_->true_values.Size() == storage_->true_size &&
      storage_->distributed.Size() == storage_->local_size &&
      MapFingerprint(map) == storage_->map_fingerprint;
}

bool DGInterfaceTraceExchangePlan::Distribute(
    mfem::ParFiniteElementSpace& target_space,
    const mfem::Vector& owned_true_values,
    mfem::Vector& local_values) const
{
  bool local_valid = true;
  const bool exact_space = &target_space == space;
  local_valid = exact_space && local_valid;
  local_valid = storage_ != nullptr && local_valid;

  const double* input = nullptr;
  const double* output = nullptr;
  if (storage_ != nullptr)
  {
    local_valid = target_space.GetParMesh() == storage_->mesh && local_valid;
    local_valid = target_space.FEColl() == storage_->collection && local_valid;
    local_valid = target_space.GetSequence() == space_sequence && local_valid;
    local_valid = storage_->mesh != nullptr && local_valid;
    if (storage_->mesh != nullptr)
    {
      local_valid = storage_->mesh->GetSequence() == mesh_sequence &&
          local_valid;
    }
    local_valid = target_space.GetVSize() == storage_->local_size &&
        local_valid;
    local_valid = target_space.GetTrueVSize() == storage_->true_size &&
        local_valid;
    local_valid = target_space.GetVDim() == storage_->vector_dimension &&
        local_valid;
    local_valid = target_space.GetOrdering() == storage_->ordering &&
        local_valid;
    local_valid = storage_->distributed.ParFESpace() == space && local_valid;
    local_valid = storage_->true_values.Size() == storage_->true_size &&
        local_valid;
    local_valid = storage_->distributed.Size() == storage_->local_size &&
        local_valid;
    local_valid = MapFingerprint(map) == storage_->map_fingerprint &&
        local_valid;
    local_valid = owned_true_values.Size() == storage_->true_size &&
        local_valid;
    local_valid = local_values.Size() == storage_->local_size && local_valid;

    try
    {
      if (owned_true_values.Size() == storage_->true_size)
      {
        input = owned_true_values.HostRead();
        local_valid =
            (input != nullptr || owned_true_values.Size() == 0) && local_valid;
        if (input != nullptr)
        {
          for (int i = 0; i < owned_true_values.Size(); ++i)
          {
            local_valid = IsFinite(input[i]) && local_valid;
          }
        }
      }
      if (local_values.Size() == storage_->local_size)
      {
        output = local_values.HostRead();
        local_valid =
            (output != nullptr || local_values.Size() == 0) && local_valid;
      }
      local_valid = !HasStorageOverlap(
          input, owned_true_values.Size(), output, local_values.Size()) &&
          local_valid;
    }
    catch (const std::exception&)
    {
      local_valid = false;
    }
  }

  if (!CollectiveAll(local_valid)) { return false; }

  bool local_copy_ready = true;
  try
  {
    input = owned_true_values.HostRead();
    double* const true_values = storage_->true_values.HostWrite();
    local_copy_ready =
        (storage_->true_size == 0 ||
         (input != nullptr && true_values != nullptr));
    if (local_copy_ready)
    {
      for (int i = 0; i < storage_->true_size; ++i)
      {
        true_values[i] = input[i];
      }
    }
  }
  catch (const std::exception&)
  {
    local_copy_ready = false;
  }
  if (!CollectiveAll(local_copy_ready)) { return false; }

  storage_->distributed.Distribute(storage_->true_values);
  const double* distributed = nullptr;
  bool local_result_valid = true;
  try
  {
    distributed = storage_->distributed.HostRead();
    local_result_valid =
        distributed != nullptr || storage_->local_size == 0;
    if (distributed != nullptr)
    {
      for (int i = 0; i < storage_->local_size; ++i)
      {
        local_result_valid = IsFinite(distributed[i]) && local_result_valid;
      }
    }
  }
  catch (const std::exception&)
  {
    local_result_valid = false;
  }
  if (!CollectiveAll(local_result_valid)) { return false; }

  double* committed = nullptr;
  bool local_commit_ready = true;
  try
  {
    committed = local_values.HostWrite();
    local_commit_ready = committed != nullptr || storage_->local_size == 0;
  }
  catch (const std::exception&)
  {
    local_commit_ready = false;
  }
  if (!CollectiveAll(local_commit_ready)) { return false; }
  for (int i = 0; i < storage_->local_size; ++i)
  {
    committed[i] = distributed[i];
  }
  return true;
}

bool BuildDGInterfaceTraceExchangePlan(
    const mfem::ParFiniteElementSpace& space,
    DGInterfaceTraceExchangePlan& plan)
{
  try
  {
    DGInterfaceTraceExchangePlan next;
    if (!BuildDGInterfaceTraceMap(space, next.map) || space.GetMesh() == nullptr)
    {
      return false;
    }
    next.space = &space;
    next.space_sequence = space.GetSequence();
    next.mesh_sequence = space.GetMesh()->GetSequence();
    next.communicator_ = space.GetComm();
    next.communicator_size_ = space.GetNRanks();
    next.storage_ = std::make_unique<DGInterfaceTraceExchangePlan::Storage>(
        const_cast<mfem::ParFiniteElementSpace&>(space),
        MapFingerprint(next.map));
    if (!next.IsConsistent()) { return false; }
    plan = std::move(next);
    return true;
  }
  catch (const std::bad_alloc&) { return false; }
  catch (const std::length_error&) { return false; }
  catch (const std::exception&) { return false; }
}

} // namespace trace
} // namespace hdg
} // namespace dg

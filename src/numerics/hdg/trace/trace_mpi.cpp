#include "numerics/hdg/trace/trace_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
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

bool IsValidIndex(const LocalIndex value, const LocalIndex size) noexcept
{
  return value >= 0 && value < size;
}

template <typename T>
bool FitsVectorCount(const std::size_t count) noexcept
{
  return count <= std::vector<T>().max_size();
}

bool TrySquareCount(const LocalIndex size, std::size_t& count) noexcept
{
  if (size < 0 || size > std::numeric_limits<int>::max()) { return false; }
  const std::size_t extent = static_cast<std::size_t>(size);
  if (extent != 0 && extent > std::numeric_limits<std::size_t>::max() / extent)
  {
    return false;
  }
  count = extent * extent;
  return FitsVectorCount<double>(count);
}

bool HasStorageOverlap(const mfem::Vector& left,
                       const mfem::Vector& right)
{
  if (left.Size() == 0 || right.Size() == 0) { return false; }
  if (left.GetData() == nullptr || right.GetData() == nullptr) { return true; }
  if (static_cast<std::size_t>(left.Size()) >
          std::numeric_limits<std::size_t>::max() / sizeof(double) ||
      static_cast<std::size_t>(right.Size()) >
          std::numeric_limits<std::size_t>::max() / sizeof(double))
  {
    return true;
  }
  const std::size_t left_bytes = static_cast<std::size_t>(left.Size()) * sizeof(double);
  const std::size_t right_bytes = static_cast<std::size_t>(right.Size()) * sizeof(double);
  const std::uintptr_t left_begin = reinterpret_cast<std::uintptr_t>(left.GetData());
  const std::uintptr_t right_begin = reinterpret_cast<std::uintptr_t>(right.GetData());
  if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left_bytes ||
      right_begin > std::numeric_limits<std::uintptr_t>::max() - right_bytes)
  {
    return true;
  }
  return left_begin < right_begin + right_bytes &&
         right_begin < left_begin + left_bytes;
}

TraceStatus CommitStagedVector(mfem::Vector& staged, mfem::Vector& output)
{
  if (output.OwnsData() || output.GetData() == nullptr)
  {
    output.Swap(staged);
    return TraceStatus::success;
  }
  if (output.Size() != staged.Size()) { return TraceStatus::size_mismatch; }
  for (int i = 0; i < staged.Size(); ++i) { output[i] = staged[i]; }
  return TraceStatus::success;
}

TraceStatus ValidateOutputShape(const mfem::Vector& output, const int size)
{
  if (!output.OwnsData() && output.GetData() != nullptr && output.Size() != size)
  {
    return TraceStatus::size_mismatch;
  }
  return TraceStatus::success;
}

bool IsValidGeometry(const mfem::Geometry::Type geometry) noexcept
{
  const int value = static_cast<int>(geometry);
  return value > static_cast<int>(mfem::Geometry::INVALID) &&
         value < static_cast<int>(mfem::Geometry::NUM_GEOMETRIES);
}

int OrientationCount(const mfem::Geometry::Type geometry) noexcept
{
  switch (geometry)
  {
    case mfem::Geometry::POINT: return 1;
    case mfem::Geometry::SEGMENT: return 2;
    case mfem::Geometry::TRIANGLE: return 6;
    case mfem::Geometry::SQUARE: return 8;
    case mfem::Geometry::TETRAHEDRON: return 24;
    case mfem::Geometry::CUBE:
    case mfem::Geometry::PRISM:
    case mfem::Geometry::PYRAMID:
    case mfem::Geometry::INVALID:
    case mfem::Geometry::NUM_GEOMETRIES: return 0;
  }
  return 0;
}

bool SameMap(const TraceMap& left, const TraceMap& right) noexcept
{
  if (left.rank != right.rank || left.rank_count != right.rank_count ||
      left.local_true_dof_count != right.local_true_dof_count ||
      left.global_true_dof_count != right.global_true_dof_count ||
      left.entries.size() != right.entries.size())
  {
    return false;
  }
  for (std::size_t i = 0; i < left.entries.size(); ++i)
  {
    const TraceMapEntry& a = left.entries[i];
    const TraceMapEntry& b = right.entries[i];
    if (a.local_dof != b.local_dof || a.global_true_dof != b.global_true_dof ||
        a.local_true_dof != b.local_true_dof || a.owner_rank != b.owner_rank ||
        a.ownership != b.ownership)
    {
      return false;
    }
  }
  return true;
}

} // namespace

bool TraceMap::IsConsistent() const noexcept
{
  if (rank < 0 || rank_count <= 0 || rank >= rank_count ||
      local_true_dof_count < 0 || global_true_dof_count < 0 ||
      static_cast<std::size_t>(local_true_dof_count) > entries.size() ||
      entries.size() > static_cast<std::size_t>(std::numeric_limits<LocalIndex>::max()))
  {
    return false;
  }
  std::vector<std::uint8_t> seen_owned;
  try
  {
    seen_owned.assign(static_cast<std::size_t>(local_true_dof_count), 0);
  }
  catch (const std::bad_alloc&) { return false; }
  catch (const std::length_error&) { return false; }

  LocalIndex owned = 0;
  for (std::size_t i = 0; i < entries.size(); ++i)
  {
    const TraceMapEntry& entry = entries[i];
    if (entry.local_dof != static_cast<LocalIndex>(i) ||
        entry.global_true_dof < 0 || entry.global_true_dof >= global_true_dof_count ||
        entry.owner_rank < 0 || entry.owner_rank >= rank_count ||
        !IsValidOwnership(entry.ownership) ||
        (entry.ownership == Ownership::owned &&
         (entry.owner_rank != rank || entry.local_true_dof < 0 ||
          entry.local_true_dof >= local_true_dof_count)) ||
        (entry.ownership == Ownership::ghost && entry.local_true_dof != invalid_local_index))
    {
      return false;
    }
    if (entry.ownership == Ownership::owned)
    {
      // The bounds above make local true IDs a compact coverage set.
      const std::size_t true_dof =
          static_cast<std::size_t>(entry.local_true_dof);
      if (seen_owned[true_dof] != 0) { return false; }
      seen_owned[true_dof] = 1;
      ++owned;
    }
  }
  return owned == local_true_dof_count;
}

std::size_t TraceMap::OwnedCount() const noexcept
{
  return static_cast<std::size_t>(std::count_if(
      entries.begin(), entries.end(), [](const TraceMapEntry& entry)
      { return entry.ownership == Ownership::owned; }));
}

std::size_t TraceMap::GhostCount() const noexcept
{
  return entries.size() - OwnedCount();
}

TraceStatus BuildTraceMap(const mfem::ParFiniteElementSpace& space,
                          TraceMap& map)
{
  try
  {
    if (!space.Conforming() || space.GetVSize() < 0 || space.GetTrueVSize() < 0 ||
        space.GetNRanks() <= 0 || space.GetMyRank() < 0 ||
        space.GetMyRank() >= space.GetNRanks() || space.GetVDim() != 1 ||
        space.FEColl() == nullptr ||
        dynamic_cast<const mfem::H1_FECollection*>(space.FEColl()) == nullptr)
    {
      return TraceStatus::invalid_argument;
    }
    const std::size_t local_dof_count = static_cast<std::size_t>(space.GetVSize());
    if (!FitsVectorCount<RankIndex>(local_dof_count) ||
        !FitsVectorCount<TraceMapEntry>(local_dof_count))
    {
      return TraceStatus::size_mismatch;
    }
    const mfem::GroupCommunicator& group_communicator = space.GroupComm();
    const mfem::Table& group_local_dofs = group_communicator.GroupLDofTable();
    const mfem::GroupTopology& group_topology =
        group_communicator.GetGroupTopology();
    if (group_local_dofs.Size() != group_topology.NGroups())
    {
      return TraceStatus::invalid_map;
    }
    TraceMap next;
    next.rank = static_cast<RankIndex>(space.GetMyRank());
    next.rank_count = static_cast<RankIndex>(space.GetNRanks());
    next.local_true_dof_count = static_cast<LocalIndex>(space.GetTrueVSize());
    const HYPRE_BigInt* const true_dof_offsets = space.GetTrueDofOffsets();
    if (true_dof_offsets == nullptr) { return TraceStatus::invalid_map; }
    const HYPRE_BigInt local_begin = HYPRE_AssumedPartitionCheck() ?
        true_dof_offsets[0] : true_dof_offsets[next.rank];
    const HYPRE_BigInt local_end = HYPRE_AssumedPartitionCheck() ?
        true_dof_offsets[1] : true_dof_offsets[next.rank + 1];
    const HYPRE_BigInt global_end = HYPRE_AssumedPartitionCheck() ?
        true_dof_offsets[2] : true_dof_offsets[next.rank_count];
    if (local_begin < 0 || local_end < local_begin || global_end < local_end ||
        global_end > std::numeric_limits<GlobalIndex>::max() ||
        local_end - local_begin != space.GetTrueVSize())
    {
      return TraceStatus::invalid_map;
    }
    next.global_true_dof_count = static_cast<GlobalIndex>(global_end);
    if (!FitsVectorCount<std::uint8_t>(local_dof_count))
    {
      return TraceStatus::size_mismatch;
    }
    std::vector<RankIndex> owner_by_local_dof(local_dof_count, next.rank);
    std::vector<std::uint8_t> shared_local_dof(local_dof_count, 0);
    for (int group = 1; group < group_local_dofs.Size(); ++group)
    {
      const RankIndex master_rank = static_cast<RankIndex>(
          group_topology.GetGroupMasterRank(group));
      if (master_rank < 0 || master_rank >= next.rank_count)
      {
        return TraceStatus::invalid_map;
      }
      const int* const local_dofs = group_local_dofs.GetRow(group);
      for (int index = 0; index < group_local_dofs.RowSize(group); ++index)
      {
        const int local_dof = local_dofs[index];
        if (local_dof < 0 || local_dof >= space.GetVSize())
        {
          return TraceStatus::invalid_map;
        }
        const std::size_t local_index = static_cast<std::size_t>(local_dof);
        if (shared_local_dof[local_index] != 0)
        {
          return TraceStatus::invalid_map;
        }
        shared_local_dof[local_index] = 1;
        owner_by_local_dof[local_index] = master_rank;
      }
    }
    next.entries.reserve(local_dof_count);
    for (LocalIndex local = 0; local < space.GetVSize(); ++local)
    {
      const auto global_true_dof = space.GetGlobalTDofNumber(local);
      if (global_true_dof < 0 ||
          global_true_dof > std::numeric_limits<GlobalIndex>::max())
      {
        return TraceStatus::invalid_map;
      }
      const GlobalIndex global = static_cast<GlobalIndex>(global_true_dof);
      const LocalIndex local_true = static_cast<LocalIndex>(space.GetLocalTDofNumber(local));
      const RankIndex owner = owner_by_local_dof[static_cast<std::size_t>(local)];
      if (global >= next.global_true_dof_count ||
          (local_true >= 0 && local_true >= next.local_true_dof_count) ||
          (local_true >= 0 && global !=
              static_cast<GlobalIndex>(local_begin + local_true)) ||
          (local_true >= 0) != (owner == next.rank))
      {
        return TraceStatus::invalid_map;
      }
      next.entries.push_back({local, global, local_true, owner,
                              local_true >= 0 ? Ownership::owned : Ownership::ghost});
    }
    if (!next.IsConsistent()) { return TraceStatus::invalid_map; }
    map = std::move(next);
    return TraceStatus::success;
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
  catch (const std::exception&) { return TraceStatus::invalid_map; }
}

bool TraceOrientationMap::IsConsistent() const noexcept
{
  if (source.size() != sign.size() ||
      source.size() > static_cast<std::size_t>(std::numeric_limits<LocalIndex>::max()))
  {
    return false;
  }
  for (std::size_t destination = 0; destination < source.size(); ++destination)
  {
    const LocalIndex input = source[destination];
    if (input < 0 || static_cast<std::size_t>(input) >= source.size() ||
        (sign[destination] != -1 && sign[destination] != 1))
    {
      return false;
    }
    for (std::size_t earlier = 0; earlier < destination; ++earlier)
    {
      if (source[earlier] == input) { return false; }
    }
  }
  return true;
}

TraceStatus TraceOrientationMap::Apply(const mfem::Vector& input,
                                       mfem::Vector& output) const
{
  if (!IsConsistent()) { return TraceStatus::invalid_orientation; }
  try
  {
    if (input.Size() != static_cast<int>(source.size()))
    {
      return TraceStatus::size_mismatch;
    }
    if (HasStorageOverlap(input, output)) { return TraceStatus::invalid_argument; }
    const TraceStatus output_status = ValidateOutputShape(output, input.Size());
    if (output_status != TraceStatus::success) { return output_status; }
    mfem::Vector staged(input.Size());
    for (int destination = 0; destination < staged.Size(); ++destination)
    {
      const double value = input[source[static_cast<std::size_t>(destination)]];
      if (!IsFinite(value)) { return TraceStatus::non_finite; }
      staged[destination] = sign[static_cast<std::size_t>(destination)] * value;
    }
    return CommitStagedVector(staged, output);
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
  catch (const std::exception&) { return TraceStatus::invalid_orientation; }
}

TraceStatus BuildOrientationMap(const mfem::FiniteElementCollection& collection,
                                const mfem::Geometry::Type geometry,
                                const int orientation,
                                const LocalIndex dof_count,
                                TraceOrientationMap& map)
{
  if (!IsValidGeometry(geometry) || dof_count < 0)
  {
    return TraceStatus::invalid_argument;
  }
  if (orientation < 0 || orientation >= OrientationCount(geometry))
  {
    return TraceStatus::invalid_orientation;
  }
  try
  {
    if (OrientationCount(geometry) == 0 ||
        collection.DofForGeometry(geometry) != dof_count ||
        !FitsVectorCount<LocalIndex>(static_cast<std::size_t>(dof_count)) ||
        !FitsVectorCount<std::int8_t>(static_cast<std::size_t>(dof_count)))
    {
      return TraceStatus::invalid_argument;
    }
    const int* order = collection.DofOrderForOrientation(geometry, orientation);
    if (dof_count != 0 && order == nullptr) { return TraceStatus::invalid_orientation; }
    TraceOrientationMap next;
    next.source.resize(static_cast<std::size_t>(dof_count));
    next.sign.resize(static_cast<std::size_t>(dof_count));
    for (LocalIndex destination = 0; destination < dof_count; ++destination)
    {
      const int encoded = order[destination];
      const LocalIndex source = encoded < 0 ? -1 - encoded : encoded;
      if (!IsValidIndex(source, dof_count)) { return TraceStatus::invalid_orientation; }
      next.source[static_cast<std::size_t>(destination)] = source;
      next.sign[static_cast<std::size_t>(destination)] = encoded < 0 ? -1 : 1;
    }
    if (!next.IsConsistent()) { return TraceStatus::invalid_orientation; }
    map = std::move(next);
    return TraceStatus::success;
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
  catch (const std::exception&) { return TraceStatus::invalid_orientation; }
}

TraceStatus TraceExchangePlan::Distribute(mfem::ParFiniteElementSpace& space,
                                          const mfem::Vector& owned_true_values,
                                          mfem::Vector& local_values) const
{
  try
  {
    if (!IsConsistent() || &space != this->space ||
        space.GetSequence() != space_sequence || space.GetMesh() == nullptr ||
        space.GetMesh()->GetSequence() != mesh_sequence ||
        owned_true_values.Size() != space.GetTrueVSize() ||
        HasStorageOverlap(owned_true_values, local_values))
    {
      return TraceStatus::invalid_map;
    }
    const TraceStatus output_status = ValidateOutputShape(local_values, space.GetVSize());
    if (output_status != TraceStatus::success) { return output_status; }
    for (int i = 0; i < owned_true_values.Size(); ++i)
    {
      if (!IsFinite(owned_true_values[i])) { return TraceStatus::non_finite; }
    }
    TraceMap current;
    const TraceStatus map_status = BuildTraceMap(space, current);
    if (map_status != TraceStatus::success) { return map_status; }
    if (!SameMap(map, current)) { return TraceStatus::invalid_map; }
    mfem::ParGridFunction local(&space);
    local.Distribute(owned_true_values);
    mfem::Vector staged(space.GetVSize());
    for (int i = 0; i < staged.Size(); ++i)
    {
      if (!IsFinite(local[i])) { return TraceStatus::non_finite; }
      staged[i] = local[i];
    }
    return CommitStagedVector(staged, local_values);
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
  catch (const std::exception&) { return TraceStatus::invalid_map; }
}

TraceStatus BuildTraceExchangePlan(const mfem::ParFiniteElementSpace& space,
                                   TraceExchangePlan& plan)
{
  try
  {
    TraceExchangePlan next;
    const TraceStatus status = BuildTraceMap(space, next.map);
    if (status != TraceStatus::success) { return status; }
    const mfem::Mesh* mesh = space.GetMesh();
    if (mesh == nullptr) { return TraceStatus::invalid_argument; }
    next.space = &space;
    next.space_sequence = space.GetSequence();
    next.mesh_sequence = mesh->GetSequence();
    plan = std::move(next);
    return TraceStatus::success;
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
  catch (const std::exception&) { return TraceStatus::invalid_map; }
}

bool AssembledTraceAction::IsConsistent() const noexcept
{
  std::size_t count = 0;
  return TrySquareCount(size_, count) && values_.size() == count &&
      std::all_of(values_.begin(), values_.end(),
                  [](const double value) { return IsFinite(value); });
}

TraceStatus AssembledTraceAction::Reset(const LocalIndex size)
{
  std::size_t count = 0;
  if (!TrySquareCount(size, count)) { return TraceStatus::invalid_action; }
  try
  {
    std::vector<double> next(count, 0.0);
    values_.swap(next);
    size_ = size;
    return TraceStatus::success;
  }
  catch (const std::bad_alloc&) { return TraceStatus::invalid_action; }
  catch (const std::length_error&) { return TraceStatus::invalid_action; }
}

TraceStatus AssembledTraceAction::Add(const LocalIndex row,
                                      const LocalIndex column,
                                      const double value)
{
  if (!IsConsistent() || !IsValidIndex(row, size_) || !IsValidIndex(column, size_))
  {
    return TraceStatus::invalid_action;
  }
  if (!IsFinite(value)) { return TraceStatus::non_finite; }
  const std::size_t index = static_cast<std::size_t>(row) *
      static_cast<std::size_t>(size_) + static_cast<std::size_t>(column);
  const double next = values_[index] + value;
  if (!IsFinite(next)) { return TraceStatus::non_finite; }
  values_[index] = next;
  return TraceStatus::success;
}

TraceStatus AssembledTraceAction::Apply(const mfem::Vector& input,
                                        mfem::Vector& output) const
{
  if (!IsConsistent()) { return TraceStatus::invalid_action; }
  try
  {
    if (input.Size() != size_) { return TraceStatus::size_mismatch; }
    if (HasStorageOverlap(input, output)) { return TraceStatus::invalid_argument; }
    const TraceStatus output_status = ValidateOutputShape(output, size_);
    if (output_status != TraceStatus::success) { return output_status; }
    mfem::Vector staged(size_);
    for (LocalIndex row = 0; row < size_; ++row)
    {
      double value = 0.0;
      for (LocalIndex column = 0; column < size_; ++column)
      {
        if (!IsFinite(input[column])) { return TraceStatus::non_finite; }
        const double term = values_[static_cast<std::size_t>(row) *
            static_cast<std::size_t>(size_) + static_cast<std::size_t>(column)] *
            input[column];
        if (!IsFinite(term)) { return TraceStatus::non_finite; }
        const double next = value + term;
        if (!IsFinite(next)) { return TraceStatus::non_finite; }
        value = next;
      }
      staged[row] = value;
    }
    return CommitStagedVector(staged, output);
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
  catch (const std::exception&) { return TraceStatus::invalid_action; }
}

bool ReferenceTraceAction::IsConsistent() const noexcept
{
  std::size_t count = 0;
  if (!TrySquareCount(size_, count)) { return false; }
  return std::all_of(contributions_.begin(), contributions_.end(),
                     [this](const TraceContribution contribution)
  {
    return IsValidIndex(contribution.row, size_) &&
           IsValidIndex(contribution.column, size_) && IsFinite(contribution.value);
  });
}

TraceStatus ReferenceTraceAction::Reset(const LocalIndex size)
{
  std::size_t count = 0;
  if (!TrySquareCount(size, count)) { return TraceStatus::invalid_action; }
  size_ = size;
  contributions_.clear();
  return TraceStatus::success;
}

TraceStatus ReferenceTraceAction::Add(const TraceContribution contribution)
{
  if (!IsConsistent() || !IsValidIndex(contribution.row, size_) ||
      !IsValidIndex(contribution.column, size_))
  {
    return TraceStatus::invalid_action;
  }
  if (!IsFinite(contribution.value)) { return TraceStatus::non_finite; }
  if (contributions_.size() == contributions_.max_size())
  {
    return TraceStatus::invalid_action;
  }
  try
  {
    contributions_.push_back(contribution);
    return TraceStatus::success;
  }
  catch (const std::bad_alloc&) { return TraceStatus::invalid_action; }
  catch (const std::length_error&) { return TraceStatus::invalid_action; }
}

TraceStatus ReferenceTraceAction::Apply(const mfem::Vector& input,
                                        mfem::Vector& output) const
{
  if (!IsConsistent()) { return TraceStatus::invalid_action; }
  try
  {
    if (input.Size() != size_) { return TraceStatus::size_mismatch; }
    if (HasStorageOverlap(input, output)) { return TraceStatus::invalid_argument; }
    const TraceStatus output_status = ValidateOutputShape(output, size_);
    if (output_status != TraceStatus::success) { return output_status; }
    for (int i = 0; i < input.Size(); ++i)
    {
      if (!IsFinite(input[i])) { return TraceStatus::non_finite; }
    }
    mfem::Vector staged(size_);
    staged = 0.0;
    for (const TraceContribution contribution : contributions_)
    {
      const double term = contribution.value * input[contribution.column];
      if (!IsFinite(term)) { return TraceStatus::non_finite; }
      const double next = staged[contribution.row] + term;
      if (!IsFinite(next)) { return TraceStatus::non_finite; }
      staged[contribution.row] = next;
    }
    return CommitStagedVector(staged, output);
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
  catch (const std::exception&) { return TraceStatus::invalid_action; }
}

TraceStatus CompareTraceActions(const AssembledTraceAction& assembled,
                                const ReferenceTraceAction& reference,
                                const mfem::Vector& input,
                                const double tolerance,
                                double& maximum_error)
{
  if (tolerance < 0.0 || !IsFinite(tolerance) || assembled.Size() != reference.Size())
  {
    return TraceStatus::invalid_argument;
  }
  mfem::Vector assembled_output;
  mfem::Vector reference_output;
  const TraceStatus assembled_status = assembled.Apply(input, assembled_output);
  if (assembled_status != TraceStatus::success) { return assembled_status; }
  const TraceStatus reference_status = reference.Apply(input, reference_output);
  if (reference_status != TraceStatus::success) { return reference_status; }
  double candidate_error = 0.0;
  for (int i = 0; i < assembled_output.Size(); ++i)
  {
    candidate_error = std::max(candidate_error,
                               std::abs(assembled_output[i] - reference_output[i]));
  }
  if (candidate_error > tolerance) { return TraceStatus::invalid_action; }
  maximum_error = candidate_error;
  return TraceStatus::success;
}

} // namespace trace
} // namespace hdg
} // namespace dg

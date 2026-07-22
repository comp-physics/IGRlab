#include "numerics/hdg/trace/trace_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

RankIndex OwnerFor(const GlobalIndex global,
                   const std::vector<GlobalIndex>& offsets) noexcept
{
  if (global < 0 || offsets.size() < 2 || global >= offsets.back())
  {
    return invalid_rank;
  }
  const auto upper = std::upper_bound(offsets.begin(), offsets.end(), global);
  return static_cast<RankIndex>(std::distance(offsets.begin(), upper) - 1);
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
                       const mfem::Vector& right) noexcept
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
      entries.size() > static_cast<std::size_t>(std::numeric_limits<LocalIndex>::max()))
  {
    return false;
  }
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
      for (std::size_t earlier = 0; earlier < i; ++earlier)
      {
        if (entries[earlier].ownership == Ownership::owned &&
            entries[earlier].local_true_dof == entry.local_true_dof)
        {
          return false;
        }
      }
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
  if (!space.Conforming() || space.GetVSize() < 0 || space.GetTrueVSize() < 0 ||
      space.GetNRanks() <= 0 || space.GetMyRank() < 0 ||
      space.GetMyRank() >= space.GetNRanks())
  {
    return TraceStatus::invalid_argument;
  }
  const std::size_t rank_count = static_cast<std::size_t>(space.GetNRanks());
  if (!FitsVectorCount<long long>(rank_count) ||
      rank_count == std::numeric_limits<std::size_t>::max() ||
      !FitsVectorCount<GlobalIndex>(rank_count + 1))
  {
    return TraceStatus::size_mismatch;
  }
  try
  {
    const long long local_count = space.GetTrueVSize();
    std::vector<long long> counts(rank_count);
    if (MPI_Allgather(&local_count, 1, MPI_LONG_LONG,
                      counts.data(), 1, MPI_LONG_LONG, space.GetComm()) != MPI_SUCCESS)
    {
      return TraceStatus::invalid_map;
    }
    std::vector<GlobalIndex> offsets(rank_count + 1, 0);
    for (RankIndex process = 0; process < space.GetNRanks(); ++process)
    {
      if (counts[static_cast<std::size_t>(process)] < 0 ||
          offsets[static_cast<std::size_t>(process)] >
              std::numeric_limits<GlobalIndex>::max() -
                  counts[static_cast<std::size_t>(process)])
      {
        return TraceStatus::invalid_map;
      }
      offsets[static_cast<std::size_t>(process) + 1] =
          offsets[static_cast<std::size_t>(process)] +
          counts[static_cast<std::size_t>(process)];
    }
    TraceMap next;
    next.rank = static_cast<RankIndex>(space.GetMyRank());
    next.rank_count = static_cast<RankIndex>(space.GetNRanks());
    next.local_true_dof_count = static_cast<LocalIndex>(space.GetTrueVSize());
    next.global_true_dof_count = offsets.back();
    if (next.global_true_dof_count !=
        static_cast<GlobalIndex>(space.GlobalTrueVSize()))
    {
      return TraceStatus::invalid_map;
    }
    if (!FitsVectorCount<TraceMapEntry>(static_cast<std::size_t>(space.GetVSize())))
    {
      return TraceStatus::size_mismatch;
    }
    next.entries.reserve(static_cast<std::size_t>(space.GetVSize()));
    for (LocalIndex local = 0; local < space.GetVSize(); ++local)
    {
      const GlobalIndex global = static_cast<GlobalIndex>(space.GetGlobalTDofNumber(local));
      const LocalIndex local_true = static_cast<LocalIndex>(space.GetLocalTDofNumber(local));
      const RankIndex owner = OwnerFor(global, offsets);
      if (owner == invalid_rank ||
          (local_true >= 0 && local_true >= next.local_true_dof_count) ||
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
  if (input.Size() != static_cast<int>(source.size())) { return TraceStatus::size_mismatch; }
  if (HasStorageOverlap(input, output)) { return TraceStatus::invalid_argument; }
  try
  {
    mfem::Vector staged(input.Size());
    for (int destination = 0; destination < staged.Size(); ++destination)
    {
      const double value = input[source[static_cast<std::size_t>(destination)]];
      if (!IsFinite(value)) { return TraceStatus::non_finite; }
      staged[destination] = sign[static_cast<std::size_t>(destination)] * value;
    }
    output.Swap(staged);
    return TraceStatus::success;
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
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
  if (OrientationCount(geometry) == 0 ||
      collection.DofForGeometry(geometry) != dof_count ||
      !FitsVectorCount<LocalIndex>(static_cast<std::size_t>(dof_count)) ||
      !FitsVectorCount<std::int8_t>(static_cast<std::size_t>(dof_count)))
  {
    return TraceStatus::invalid_argument;
  }
  try
  {
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
}

TraceStatus TraceExchangePlan::Distribute(mfem::ParFiniteElementSpace& space,
                                          const mfem::Vector& owned_true_values,
                                          mfem::Vector& local_values) const
{
  if (!IsConsistent() || &space != this->space ||
      owned_true_values.Size() != space.GetTrueVSize() ||
      HasStorageOverlap(owned_true_values, local_values))
  {
    return TraceStatus::invalid_map;
  }
  for (int i = 0; i < owned_true_values.Size(); ++i)
  {
    if (!IsFinite(owned_true_values[i])) { return TraceStatus::non_finite; }
  }
  TraceMap current;
  const TraceStatus map_status = BuildTraceMap(space, current);
  if (map_status != TraceStatus::success) { return map_status; }
  if (!SameMap(map, current)) { return TraceStatus::invalid_map; }
  try
  {
    mfem::ParGridFunction local(&space);
    local.Distribute(owned_true_values);
    mfem::Vector staged(space.GetVSize());
    for (int i = 0; i < staged.Size(); ++i)
    {
      if (!IsFinite(local[i])) { return TraceStatus::non_finite; }
      staged[i] = local[i];
    }
    local_values.Swap(staged);
    return TraceStatus::success;
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
}

TraceStatus BuildTraceExchangePlan(const mfem::ParFiniteElementSpace& space,
                                   TraceExchangePlan& plan)
{
  TraceExchangePlan next;
  const TraceStatus status = BuildTraceMap(space, next.map);
  if (status != TraceStatus::success) { return status; }
  next.space = &space;
  plan = std::move(next);
  return TraceStatus::success;
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
  if (input.Size() != size_) { return TraceStatus::size_mismatch; }
  if (HasStorageOverlap(input, output)) { return TraceStatus::invalid_argument; }
  try
  {
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
    output.Swap(staged);
    return TraceStatus::success;
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
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
  if (input.Size() != size_) { return TraceStatus::size_mismatch; }
  if (HasStorageOverlap(input, output)) { return TraceStatus::invalid_argument; }
  for (int i = 0; i < input.Size(); ++i)
  {
    if (!IsFinite(input[i])) { return TraceStatus::non_finite; }
  }
  try
  {
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
    output.Swap(staged);
    return TraceStatus::success;
  }
  catch (const std::bad_alloc&) { return TraceStatus::size_mismatch; }
  catch (const std::length_error&) { return TraceStatus::size_mismatch; }
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

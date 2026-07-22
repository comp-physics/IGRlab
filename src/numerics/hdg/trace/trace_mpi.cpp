#include "numerics/hdg/trace/trace_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

bool ValidActionSize(const LocalIndex size) noexcept
{
  return size >= 0 && static_cast<std::size_t>(size) <=
                          std::numeric_limits<std::size_t>::max() /
                              static_cast<std::size_t>(size == 0 ? 1 : size);
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
  const long long local_count = space.GetTrueVSize();
  std::vector<long long> counts(static_cast<std::size_t>(space.GetNRanks()));
  if (MPI_Allgather(&local_count, 1, MPI_LONG_LONG,
                    counts.data(), 1, MPI_LONG_LONG, space.GetComm()) != MPI_SUCCESS)
  {
    return TraceStatus::invalid_map;
  }
  std::vector<GlobalIndex> offsets(static_cast<std::size_t>(space.GetNRanks()) + 1,
                                   0);
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
  if (&input == &output) { return TraceStatus::invalid_argument; }
  output.SetSize(input.Size());
  for (int destination = 0; destination < output.Size(); ++destination)
  {
    const double value = input[source[static_cast<std::size_t>(destination)]];
    if (!IsFinite(value)) { return TraceStatus::non_finite; }
    output[destination] = sign[static_cast<std::size_t>(destination)] * value;
  }
  return TraceStatus::success;
}

TraceStatus BuildOrientationMap(const mfem::FiniteElementCollection& collection,
                                const mfem::Geometry::Type geometry,
                                const int orientation,
                                const LocalIndex dof_count,
                                TraceOrientationMap& map)
{
  if (dof_count < 0 || collection.DofForGeometry(geometry) != dof_count)
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

TraceStatus TraceExchangePlan::Distribute(mfem::ParFiniteElementSpace& space,
                                          const mfem::Vector& owned_true_values,
                                          mfem::Vector& local_values) const
{
  if (!IsConsistent() || map.rank != space.GetMyRank() ||
      map.rank_count != space.GetNRanks() ||
      map.local_true_dof_count != space.GetTrueVSize() ||
      owned_true_values.Size() != space.GetTrueVSize())
  {
    return TraceStatus::size_mismatch;
  }
  for (int i = 0; i < owned_true_values.Size(); ++i)
  {
    if (!IsFinite(owned_true_values[i])) { return TraceStatus::non_finite; }
  }
  mfem::ParGridFunction local(&space);
  local.Distribute(owned_true_values);
  local_values.SetSize(space.GetVSize());
  local_values = local;
  return TraceStatus::success;
}

TraceStatus BuildTraceExchangePlan(const mfem::ParFiniteElementSpace& space,
                                   TraceExchangePlan& plan)
{
  TraceExchangePlan next;
  const TraceStatus status = BuildTraceMap(space, next.map);
  if (status != TraceStatus::success) { return status; }
  plan = std::move(next);
  return TraceStatus::success;
}

bool AssembledTraceAction::IsConsistent() const noexcept
{
  return ValidActionSize(size_) && values_.size() ==
      static_cast<std::size_t>(size_) * static_cast<std::size_t>(size_);
}

TraceStatus AssembledTraceAction::Reset(const LocalIndex size)
{
  if (!ValidActionSize(size)) { return TraceStatus::invalid_action; }
  size_ = size;
  values_.assign(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), 0.0);
  return TraceStatus::success;
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
  double& entry = values_[static_cast<std::size_t>(row) * size_ + column];
  entry += value;
  return IsFinite(entry) ? TraceStatus::success : TraceStatus::non_finite;
}

TraceStatus AssembledTraceAction::Apply(const mfem::Vector& input,
                                        mfem::Vector& output) const
{
  if (!IsConsistent()) { return TraceStatus::invalid_action; }
  if (input.Size() != size_) { return TraceStatus::size_mismatch; }
  if (&input == &output) { return TraceStatus::invalid_argument; }
  output.SetSize(size_);
  for (LocalIndex row = 0; row < size_; ++row)
  {
    double value = 0.0;
    for (LocalIndex column = 0; column < size_; ++column)
    {
      if (!IsFinite(input[column])) { return TraceStatus::non_finite; }
      value += values_[static_cast<std::size_t>(row) * size_ + column] * input[column];
    }
    if (!IsFinite(value)) { return TraceStatus::non_finite; }
    output[row] = value;
  }
  return TraceStatus::success;
}

bool ReferenceTraceAction::IsConsistent() const noexcept
{
  if (!ValidActionSize(size_)) { return false; }
  return std::all_of(contributions_.begin(), contributions_.end(),
                     [this](const TraceContribution contribution)
  {
    return IsValidIndex(contribution.row, size_) &&
           IsValidIndex(contribution.column, size_) && IsFinite(contribution.value);
  });
}

TraceStatus ReferenceTraceAction::Reset(const LocalIndex size)
{
  if (!ValidActionSize(size)) { return TraceStatus::invalid_action; }
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
  contributions_.push_back(contribution);
  return TraceStatus::success;
}

TraceStatus ReferenceTraceAction::Apply(const mfem::Vector& input,
                                        mfem::Vector& output) const
{
  if (!IsConsistent()) { return TraceStatus::invalid_action; }
  if (input.Size() != size_) { return TraceStatus::size_mismatch; }
  if (&input == &output) { return TraceStatus::invalid_argument; }
  output.SetSize(size_);
  output = 0.0;
  for (int i = 0; i < input.Size(); ++i)
  {
    if (!IsFinite(input[i])) { return TraceStatus::non_finite; }
  }
  for (const TraceContribution contribution : contributions_)
  {
    output[contribution.row] += contribution.value * input[contribution.column];
    if (!IsFinite(output[contribution.row])) { return TraceStatus::non_finite; }
  }
  return TraceStatus::success;
}

TraceStatus CompareTraceActions(const AssembledTraceAction& assembled,
                                const ReferenceTraceAction& reference,
                                const mfem::Vector& input,
                                const double tolerance,
                                double& maximum_error)
{
  maximum_error = std::numeric_limits<double>::infinity();
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
  maximum_error = 0.0;
  for (int i = 0; i < assembled_output.Size(); ++i)
  {
    maximum_error = std::max(maximum_error,
                             std::abs(assembled_output[i] - reference_output[i]));
  }
  return maximum_error <= tolerance ? TraceStatus::success : TraceStatus::invalid_action;
}

} // namespace trace
} // namespace hdg
} // namespace dg

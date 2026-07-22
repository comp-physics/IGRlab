#pragma once

/**
 * @file numerics/hdg/trace/trace_mpi.hpp
 * @brief Deterministic, MFEM-owned HDG trace map and exchange primitives.
 *
 * This layer deliberately describes only trace topology and linear actions.
 * It has no model, flux, or condensation dependency. Trace ownership comes
 * from MFEM true dofs; ghost values are refreshed through MFEM's
 * prolongation/distribution operator, rather than a private MPI protocol.
 */

#include "numerics/hdg/hdg_topology.hpp"

#include <mfem.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dg
{
namespace hdg
{
namespace trace
{

enum class TraceStatus : std::uint8_t
{
  success,
  invalid_argument,
  invalid_map,
  invalid_orientation,
  invalid_action,
  size_mismatch,
  non_finite
};

/** One local MFEM vector dof and its deterministic true-dof identity. */
struct TraceMapEntry
{
  LocalIndex local_dof = invalid_local_index;
  GlobalIndex global_true_dof = invalid_global_index;
  LocalIndex local_true_dof = invalid_local_index;
  RankIndex owner_rank = invalid_rank;
  Ownership ownership = Ownership::ghost;
};

/** Local trace indexing in ascending MFEM local-dof order. */
struct TraceMap
{
  RankIndex rank = invalid_rank;
  RankIndex rank_count = 0;
  LocalIndex local_true_dof_count = 0;
  GlobalIndex global_true_dof_count = 0;
  std::vector<TraceMapEntry> entries;

  bool IsConsistent() const noexcept;
  std::size_t OwnedCount() const noexcept;
  std::size_t GhostCount() const noexcept;
};

/** Build ownership from a conforming MFEM true-dof partition. */
TraceStatus BuildTraceMap(const mfem::ParFiniteElementSpace& space,
                          TraceMap& map);

/** A signed local permutation. source[i] is read into destination[i]. */
struct TraceOrientationMap
{
  std::vector<LocalIndex> source;
  std::vector<std::int8_t> sign;

  bool IsConsistent() const noexcept;
  TraceStatus Apply(const mfem::Vector& input, mfem::Vector& output) const;
};

/** Decode MFEM's signed DofOrderForOrientation convention deterministically. */
TraceStatus BuildOrientationMap(const mfem::FiniteElementCollection& collection,
                                mfem::Geometry::Type geometry,
                                int orientation,
                                LocalIndex dof_count,
                                TraceOrientationMap& map);

/**
 * An exchange plan is metadata plus the MFEM prolongation operation required
 * to update ghost/local dofs from rank-owned true dofs. It owns no MPI
 * requests or buffers and therefore follows MFEM's communicator and memory
 * configuration (including RAJA/Umpire-enabled MFEM builds).
 */
struct TraceExchangePlan
{
  /** Exact space instance used to create map; no compatible-size substitution. */
  const mfem::ParFiniteElementSpace* space = nullptr;
  TraceMap map;

  bool IsConsistent() const noexcept
  {
    return space != nullptr && map.IsConsistent();
  }
  TraceStatus Distribute(mfem::ParFiniteElementSpace& space,
                         const mfem::Vector& owned_true_values,
                         mfem::Vector& local_values) const;
};

TraceStatus BuildTraceExchangePlan(const mfem::ParFiniteElementSpace& space,
                                   TraceExchangePlan& plan);

/** Dense, deterministic assembled trace action used only as a small host
 * foundation and test oracle. Production storage/backends can use the same
 * map and exchange contracts without inheriting this container.
 */
class AssembledTraceAction
{
public:
  TraceStatus Reset(LocalIndex size);
  TraceStatus Add(LocalIndex row, LocalIndex column, double value);
  TraceStatus Apply(const mfem::Vector& input, mfem::Vector& output) const;

  LocalIndex Size() const noexcept { return size_; }
  bool IsConsistent() const noexcept;

private:
  LocalIndex size_ = 0;
  std::vector<double> values_;
};

/** Reference action intentionally evaluates local contributions directly. */
struct TraceContribution
{
  LocalIndex row = invalid_local_index;
  LocalIndex column = invalid_local_index;
  double value = 0.0;
};

class ReferenceTraceAction
{
public:
  TraceStatus Reset(LocalIndex size);
  TraceStatus Add(TraceContribution contribution);
  TraceStatus Apply(const mfem::Vector& input, mfem::Vector& output) const;

  LocalIndex Size() const noexcept { return size_; }
  bool IsConsistent() const noexcept;

private:
  LocalIndex size_ = 0;
  std::vector<TraceContribution> contributions_;
};

/** Exact (zero-tolerance) or bounded action comparison. */
TraceStatus CompareTraceActions(const AssembledTraceAction& assembled,
                                const ReferenceTraceAction& reference,
                                const mfem::Vector& input,
                                double tolerance,
                                double& maximum_error);

} // namespace trace
} // namespace hdg
} // namespace dg

#include "numerics/hdg/trace/trace_mpi.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace
{

using dg::hdg::GlobalIndex;
using dg::hdg::LocalIndex;
using dg::hdg::Ownership;
using dg::hdg::trace::AssembledTraceAction;
using dg::hdg::trace::BuildOrientationMap;
using dg::hdg::trace::BuildTraceExchangePlan;
using dg::hdg::trace::CompareTraceActions;
using dg::hdg::trace::ReferenceTraceAction;
using dg::hdg::trace::TraceContribution;
using dg::hdg::trace::TraceExchangePlan;
using dg::hdg::trace::TraceMap;
using dg::hdg::trace::TraceStatus;

bool SameValues(const mfem::Vector& left, const mfem::Vector& right)
{
  if (left.Size() != right.Size()) { return false; }
  for (int i = 0; i < left.Size(); ++i)
  {
    if (left[i] != right[i]) { return false; }
  }
  return true;
}

bool CheckActions()
{
  AssembledTraceAction assembled;
  ReferenceTraceAction reference;
  if (assembled.Reset(3) != TraceStatus::success ||
      reference.Reset(3) != TraceStatus::success) { return false; }
  const TraceContribution contributions[] = {
      {0, 0, 2.0}, {0, 1, -1.0}, {1, 0, 3.0},
      {1, 2, 0.5}, {2, 1, -4.0}, {2, 2, 1.0}};
  for (const TraceContribution contribution : contributions)
  {
    if (assembled.Add(contribution.row, contribution.column, contribution.value) !=
            TraceStatus::success || reference.Add(contribution) != TraceStatus::success)
    {
      return false;
    }
  }
  mfem::Vector input(3);
  input[0] = 0.25;
  input[1] = -2.0;
  input[2] = 1.5;
  double error = 0.0;
  if (CompareTraceActions(assembled, reference, input, 0.0, error) !=
          TraceStatus::success || error != 0.0) { return false; }
  input[1] = std::numeric_limits<double>::quiet_NaN();
  mfem::Vector output(3);
  output = 17.0;
  const mfem::Vector output_before(output);
  if (assembled.Apply(input, output) != TraceStatus::non_finite ||
      reference.Apply(input, output) != TraceStatus::non_finite ||
      !SameValues(output, output_before) ||
      assembled.Apply(input, input) != TraceStatus::invalid_argument ||
      assembled.Add(3, 0, 1.0) != TraceStatus::invalid_action)
  {
    return false;
  }

  mfem::Vector backing(4);
  backing = 3.0;
  mfem::Vector overlap_input;
  mfem::Vector overlap_output;
  overlap_input.MakeRef(backing, 0, 3);
  overlap_output.MakeRef(backing, 1, 3);
  const mfem::Vector backing_before(backing);
  if (assembled.Apply(overlap_input, overlap_output) != TraceStatus::invalid_argument ||
      reference.Apply(overlap_input, overlap_output) != TraceStatus::invalid_argument ||
      !SameValues(backing, backing_before))
  {
    return false;
  }

  AssembledTraceAction bounded;
  if (bounded.Reset(1) != TraceStatus::success ||
      bounded.Add(0, 0, std::numeric_limits<double>::max()) != TraceStatus::success ||
      bounded.Add(0, 0, std::numeric_limits<double>::max()) != TraceStatus::non_finite ||
      !bounded.IsConsistent() ||
      bounded.Reset(std::numeric_limits<LocalIndex>::max()) != TraceStatus::invalid_action ||
      bounded.Size() != 1)
  {
    return false;
  }
  mfem::Vector one(1);
  mfem::Vector bounded_output;
  one[0] = 1.0;
  return bounded.Apply(one, bounded_output) == TraceStatus::success &&
         bounded_output[0] == std::numeric_limits<double>::max();
}

bool CheckOrientation()
{
  mfem::H1_FECollection collection(3, 2);
  const LocalIndex count = collection.DofForGeometry(mfem::Geometry::SEGMENT);
  dg::hdg::trace::TraceOrientationMap forward;
  dg::hdg::trace::TraceOrientationMap reverse;
  if (count <= 0 || BuildOrientationMap(collection, mfem::Geometry::SEGMENT, 1,
      count, forward) != TraceStatus::success ||
      BuildOrientationMap(collection, mfem::Geometry::SEGMENT, 0, count, reverse) !=
          TraceStatus::success) { return false; }
  mfem::Vector source(count);
  for (int i = 0; i < count; ++i) { source[i] = 1.25 + i; }
  mfem::Vector oriented;
  if (!forward.IsConsistent() || !reverse.IsConsistent() ||
      forward.Apply(source, oriented) != TraceStatus::success ||
      oriented.Size() != source.Size())
  {
    return false;
  }
  const mfem::Vector oriented_before(oriented);
  source[0] = std::numeric_limits<double>::quiet_NaN();
  if (forward.Apply(source, oriented) != TraceStatus::non_finite ||
      !SameValues(oriented, oriented_before))
  {
    return false;
  }
  source[0] = 1.25;
  if (BuildOrientationMap(collection, mfem::Geometry::SEGMENT, -1, count, forward) !=
          TraceStatus::invalid_orientation ||
      BuildOrientationMap(collection, mfem::Geometry::SEGMENT, 2, count, forward) !=
          TraceStatus::invalid_orientation ||
      BuildOrientationMap(collection, mfem::Geometry::CUBE, 0, 0, forward) !=
          TraceStatus::invalid_orientation ||
      BuildOrientationMap(collection, static_cast<mfem::Geometry::Type>(
          mfem::Geometry::INVALID), 0, count, forward) != TraceStatus::invalid_argument ||
      BuildOrientationMap(collection, static_cast<mfem::Geometry::Type>(-2), 0, count,
                          forward) != TraceStatus::invalid_argument ||
      BuildOrientationMap(collection, static_cast<mfem::Geometry::Type>(
          mfem::Geometry::NUM_GEOMETRIES), 0, count, forward) != TraceStatus::invalid_argument ||
      BuildOrientationMap(collection, static_cast<mfem::Geometry::Type>(
          static_cast<int>(mfem::Geometry::NUM_GEOMETRIES) + 1), 0, count,
                          forward) != TraceStatus::invalid_argument)
  {
    return false;
  }

  mfem::ND_FECollection signed_collection(2, 2);
  const LocalIndex signed_count = signed_collection.DofForGeometry(
      mfem::Geometry::SEGMENT);
  dg::hdg::trace::TraceOrientationMap signed_map;
  mfem::Vector signed_input(signed_count);
  signed_input = 1.0;
  mfem::Vector signed_output;
  if (signed_count <= 0 || BuildOrientationMap(signed_collection,
      mfem::Geometry::SEGMENT, 0, signed_count, signed_map) != TraceStatus::success ||
      signed_map.Apply(signed_input, signed_output) != TraceStatus::success)
  {
    return false;
  }
  bool mfem_reports_negative = false;
  bool application_preserves_negative = false;
  const int* mfem_order = signed_collection.DofOrderForOrientation(
      mfem::Geometry::SEGMENT, 0);
  for (int i = 0; i < signed_count; ++i)
  {
    mfem_reports_negative = mfem_reports_negative || mfem_order[i] < 0;
    application_preserves_negative = application_preserves_negative || signed_output[i] < 0.0;
  }
  return mfem_reports_negative && application_preserves_negative;
}

bool CheckMapAndExchange(MPI_Comm comm)
{
  int rank_count = 0;
  MPI_Comm_size(comm, &rank_count);
  if (rank_count != 1 && rank_count != 2) { return true; }
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian1D(4, 1.0);
  mfem::ParMesh parallel_mesh(comm, serial_mesh);
  mfem::H1_Trace_FECollection collection(1, 1);
  mfem::ParFiniteElementSpace trace_space(&parallel_mesh, &collection);
  TraceExchangePlan plan;
  if (BuildTraceExchangePlan(trace_space, plan) != TraceStatus::success ||
      !plan.IsConsistent() || plan.map.OwnedCount() !=
          static_cast<std::size_t>(trace_space.GetTrueVSize()) ||
      plan.map.entries.size() != static_cast<std::size_t>(trace_space.GetVSize()))
  {
    return false;
  }
  mfem::Vector true_values(trace_space.GetTrueVSize());
  const GlobalIndex offset = static_cast<GlobalIndex>(trace_space.GetMyTDofOffset());
  for (int i = 0; i < true_values.Size(); ++i)
  {
    true_values[i] = 0.5 + static_cast<double>(offset + i);
  }
  mfem::Vector local_values;
  if (plan.Distribute(trace_space, true_values, local_values) != TraceStatus::success)
  {
    return false;
  }
  for (const auto& entry : plan.map.entries)
  {
    const double expected = 0.5 + static_cast<double>(entry.global_true_dof);
    if (std::abs(local_values[entry.local_dof] - expected) != 0.0 ||
        (entry.ownership == Ownership::owned) !=
            (entry.owner_rank == trace_space.GetMyRank())) { return false; }
  }

  mfem::ParFiniteElementSpace unrelated_space(&parallel_mesh, &collection);
  mfem::Vector unrelated_true(unrelated_space.GetTrueVSize());
  unrelated_true = 1.0;
  mfem::Vector unrelated_local(1);
  unrelated_local[0] = -11.0;
  if (plan.Distribute(unrelated_space, unrelated_true, unrelated_local) !=
          TraceStatus::invalid_map || unrelated_local.Size() != 1 ||
      unrelated_local[0] != -11.0)
  {
    return false;
  }

  if (plan.map.entries.size() > 1)
  {
    TraceExchangePlan stale = plan;
    std::swap(stale.map.entries[0].global_true_dof,
              stale.map.entries[1].global_true_dof);
    mfem::Vector stale_local(1);
    stale_local[0] = -13.0;
    if (!stale.IsConsistent() ||
        stale.Distribute(trace_space, true_values, stale_local) !=
            TraceStatus::invalid_map || stale_local.Size() != 1 ||
        stale_local[0] != -13.0)
    {
      return false;
    }
  }

  mfem::Vector non_finite_true(true_values);
  non_finite_true[0] = std::numeric_limits<double>::quiet_NaN();
  mfem::Vector preserved_local(1);
  preserved_local[0] = -19.0;
  if (plan.Distribute(trace_space, non_finite_true, preserved_local) !=
          TraceStatus::non_finite || preserved_local.Size() != 1 ||
      preserved_local[0] != -19.0)
  {
    return false;
  }

  TraceMap invalid = plan.map;
  invalid.entries[0].owner_rank = invalid.rank_count;
  return !invalid.IsConsistent();
}

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  const bool passed = CheckActions() && CheckOrientation() &&
                      CheckMapAndExchange(MPI_COMM_WORLD);
  int local = passed ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Finalize();
  return global == 1 ? 0 : 1;
}

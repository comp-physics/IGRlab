#include "numerics/hdg/trace/trace_mpi.hpp"

#include <cmath>
#include <limits>

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
  mfem::Vector output;
  return assembled.Apply(input, output) == TraceStatus::non_finite &&
         assembled.Apply(input, input) == TraceStatus::invalid_argument &&
         assembled.Add(3, 0, 1.0) == TraceStatus::invalid_action;
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
  const int* expected = collection.DofOrderForOrientation(
      mfem::Geometry::SEGMENT, 1);
  if (!forward.IsConsistent() || !reverse.IsConsistent() || expected == nullptr ||
      forward.Apply(source, oriented) != TraceStatus::success ||
      oriented.Size() != source.Size())
  {
    return false;
  }
  for (int i = 0; i < count; ++i)
  {
    const int encoded = expected[i];
    const int source_index = encoded < 0 ? -1 - encoded : encoded;
    const double expected_value = (encoded < 0 ? -1.0 : 1.0) * source[source_index];
    if (forward.source[static_cast<std::size_t>(i)] != source_index ||
        forward.sign[static_cast<std::size_t>(i)] != (encoded < 0 ? -1 : 1) ||
        oriented[i] != expected_value)
    {
      return false;
    }
  }
  return true;
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

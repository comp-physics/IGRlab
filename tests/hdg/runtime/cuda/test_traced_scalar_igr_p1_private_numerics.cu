#include "numerics/hdg/runtime/cuda/traced_scalar_igr_p1_private_numerics.cuh"

#include <HYPRE.h>
#include <mfem.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <memory>
#include <limits>
#include <string_view>
#include <vector>

namespace
{
using Topology = dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics::RetainedTopology;

bool All(const bool local) noexcept
{
  int in = local ? 1 : 0, out = 0;
  return MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS && out == 1;
}

int Decode(const int value) noexcept
{ return value == std::numeric_limits<int>::min() ? -1 : (value < 0 ? -1 - value : value); }

struct OracleCsrView final
{
  mfem::SparseMatrix diagonal, offdiagonal;
  HYPRE_BigInt* cmap = nullptr;
  HYPRE_BigInt begin = -1, end = -1;

  bool Bind(mfem::HypreParMatrix& matrix)
  {
    matrix.HostReadWrite(); matrix.GetDiag(diagonal); matrix.GetOffd(offdiagonal, cmap);
    const HYPRE_BigInt* starts = matrix.GetRowStarts();
    int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!starts) return false;
    begin = HYPRE_AssumedPartitionCheck() ? starts[0] : starts[rank];
    end = begin + diagonal.Height();
    return begin >= 0 && end >= begin && offdiagonal.Height() == diagonal.Height();
  }

  bool Find(const HYPRE_BigInt row, const HYPRE_BigInt column,
            int& entry, bool& offd) const
  {
    entry = -1; offd = false;
    const int* di = diagonal.GetI(); const int* dj = diagonal.GetJ();
    const int* oi = offdiagonal.GetI(); const int* oj = offdiagonal.GetJ();
    if (!di || diagonal.Height() < 0 || offdiagonal.Height() != diagonal.Height() ||
        row < begin || row >= end) return false;
    const int local_row = static_cast<int>(row - begin);
    if (column >= begin && column < end)
    {
      if (!dj) return false;
      const int local_column = static_cast<int>(column - begin);
      for (int index = di[local_row]; index < di[local_row + 1]; ++index)
        if (dj[index] == local_column) { entry = index; return true; }
      return false;
    }
    if (!oi || !oj || !cmap) return false;
    for (int index = oi[local_row]; index < oi[local_row + 1]; ++index)
      if (oj[index] >= 0 && oj[index] < offdiagonal.Width() && cmap[oj[index]] == column)
      { entry = index; offd = true; return true; }
    return false;
  }
};

bool SameArena(const Topology::ArenaPlan& left, const Topology::ArenaPlan& right)
{
  return left.element_count == right.element_count && left.interior_values == right.interior_values &&
      left.trace_values == right.trace_values && left.integer_logical_bytes == right.integer_logical_bytes &&
      left.double_logical_bytes == right.double_logical_bytes && left.integer_physical_bytes == right.integer_physical_bytes &&
      left.double_physical_bytes == right.double_physical_bytes && left.interior_map_offset == right.interior_map_offset &&
      left.trace_map_offset == right.trace_map_offset && left.vertices_offset == right.vertices_offset &&
      left.volume_quadrature_offset == right.volume_quadrature_offset && left.segment_quadrature_offset == right.segment_quadrature_offset &&
      left.volume_quadrature_values == right.volume_quadrature_values &&
      left.segment_quadrature_values == right.segment_quadrature_values && left.symbolic_slots == right.symbolic_slots;
}

bool SameCsr(const std::vector<Topology::RetainedCsrSlot>& left,
             const std::vector<Topology::RetainedCsrSlot>& right)
{
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (left[index].source_slot != right[index].source_slot ||
        left[index].signed_row_vdof != right[index].signed_row_vdof ||
        left[index].signed_column_vdof != right[index].signed_column_vdof ||
        left[index].global_row != right[index].global_row || left[index].global_column != right[index].global_column ||
        left[index].owner_rank != right[index].owner_rank || left[index].csr_entry != right[index].csr_entry ||
        left[index].csr_entry_is_offd != right[index].csr_entry_is_offd ||
        left[index].row_sign != right[index].row_sign || left[index].column_sign != right[index].column_sign) return false;
  return true;
}

bool SameExchange(const std::vector<Topology::ValueExchangeSlot>& left,
                  const std::vector<Topology::ValueExchangeSlot>& right)
{
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (left[index].source_slot != right[index].source_slot || left[index].peer_rank != right[index].peer_rank ||
        left[index].csr_entry != right[index].csr_entry || left[index].csr_entry_is_offd != right[index].csr_entry_is_offd) return false;
  return true;
}

std::unique_ptr<mfem::ParMesh> PeriodicThreeByThree(const bool rank_zero_only)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(3, 3, mfem::Element::QUADRILATERAL,
      false, 1.0, 1.0, true);
  std::vector<mfem::Vector> translation(2, mfem::Vector(2));
  translation[0] = 0.0; translation[1] = 0.0;
  translation[0][0] = 1.0; translation[1][1] = 1.0;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(serial,
      serial.CreatePeriodicVertexMapping(translation));
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  std::vector<int> partition(static_cast<std::size_t>(periodic.GetNE()));
  for (int element = 0; element < periodic.GetNE(); ++element)
    partition[static_cast<std::size_t>(element)] = rank_zero_only ? 0 : element % ranks;
  return std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, periodic, partition.data());
}

bool ComponentMajorMaps(const Topology& topology)
{
  const auto* interior = topology.InteriorSpace();
  const auto* trace = topology.TraceSpace();
  bool local = topology.IsReady() && interior && trace && interior->GetVDim() == 7 &&
      trace->GetVDim() == 5 && interior->GetOrdering() == mfem::Ordering::byNODES &&
      trace->GetOrdering() == mfem::Ordering::byNODES &&
      topology.SignedElementInteriorVDofs().size() ==
          static_cast<std::size_t>(topology.LocalElementCount()) * 28 &&
      topology.SignedElementTraceVDofs().size() ==
          static_cast<std::size_t>(topology.LocalElementCount()) * 40;
  for (int element = 0; local && element < topology.LocalElementCount(); ++element)
  {
    mfem::Array<int> actual_interior, actual_trace;
    interior->GetElementVDofs(element, actual_interior);
    trace->GetElementVDofs(element, actual_trace);
    local = actual_interior.Size() == 28 && actual_trace.Size() == 40;
    for (int component = 0; local && component < 7; ++component)
      for (int dof = 0; local && dof < 4; ++dof)
      {
        const int slot = component * 4 + dof;
        const int value = topology.SignedElementInteriorVDofs()[static_cast<std::size_t>(element) * 28 + slot];
        const int base = topology.SignedElementInteriorVDofs()[static_cast<std::size_t>(element) * 28 + dof];
        local = value == actual_interior[slot] && Decode(value) >= 0 &&
            Decode(value) == Decode(base) + component * (interior->GetVSize() / 7) &&
            (value < 0) == (base < 0);
      }
    for (int component = 0; local && component < 5; ++component)
      for (int face_dof = 0; local && face_dof < 8; ++face_dof)
      {
        const int slot = component * 8 + face_dof;
        const int value = topology.SignedElementTraceVDofs()[static_cast<std::size_t>(element) * 40 + slot];
        const int base = topology.SignedElementTraceVDofs()[static_cast<std::size_t>(element) * 40 + face_dof];
        local = value == actual_trace[slot] && Decode(value) >= 0 &&
            Decode(value) == Decode(base) + component * (trace->GetVSize() / 5) &&
            (value < 0) == (base < 0);
      }
  }
  return All(local);
}

bool RetainedCsrAndExchange(Topology& topology)
{
  const auto* trace = topology.TraceSpace();
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  bool local = trace != nullptr;
  std::vector<int> owner(trace ? static_cast<std::size_t>(trace->GetVSize()) : 0, rank);
  if (trace)
  {
    const mfem::GroupCommunicator& groups = trace->GroupComm();
    const mfem::Table& local_dofs = groups.GroupLDofTable();
    const mfem::GroupTopology& group_topology = groups.GetGroupTopology();
    local = local && local_dofs.Size() == group_topology.NGroups();
    for (int group = 1; local && group < local_dofs.Size(); ++group)
    {
      const int master = group_topology.GetGroupMasterRank(group);
      const int* const dofs = local_dofs.GetRow(group);
      local = master >= 0 && master < ranks &&
          (local_dofs.RowSize(group) == 0 || dofs != nullptr);
      for (int index = 0; local && index < local_dofs.RowSize(group); ++index)
      {
        local = dofs[index] >= 0 && dofs[index] < trace->GetVSize();
        if (local) owner[static_cast<std::size_t>(dofs[index])] = master;
      }
    }
  }
  std::unique_ptr<mfem::SparseMatrix> graph;
  std::unique_ptr<mfem::ParBilinearForm> form;
  try
  {
    graph = std::make_unique<mfem::SparseMatrix>(trace ? trace->GetVSize() : 0);
    if (trace)
      for (int element = 0; element < topology.LocalElementCount(); ++element)
      {
        mfem::Array<int> element_vdofs;
        trace->GetElementVDofs(element, element_vdofs);
        mfem::DenseMatrix pattern(40);
        local = local && element_vdofs.Size() == 40;
        if (!local) break;
        for (int row = 0; row < 40; ++row)
          for (int column = 0; column < 40; ++column)
            pattern(row, column) = static_cast<double>((element_vdofs[row] < 0 ? -1 : 1) *
                (element_vdofs[column] < 0 ? -1 : 1));
        graph->AddSubMatrix(element_vdofs, element_vdofs, pattern, 1);
      }
    graph->Finalize(0, false);
    form = trace ? std::make_unique<mfem::ParBilinearForm>(const_cast<mfem::ParFiniteElementSpace*>(trace)) : nullptr;
    local = local && graph != nullptr && form != nullptr;
  }
  catch (...) { local = false; }
  if (!All(local)) return false;
  std::unique_ptr<mfem::HypreParMatrix> oracle;
  try { oracle.reset(form->ParallelAssemble(graph.get())); local = oracle != nullptr; }
  catch (...) { local = false; }
  if (!All(local)) return false;
  OracleCsrView oracle_view;
  local = oracle_view.Bind(*oracle);
  if (!All(local)) return false;
  const auto& slots = topology.RetainedCsrSlots();
  const auto& owned = topology.OwnedCsrSlots();
  const auto& outgoing = topology.OutgoingValueSlots();
  const auto& incoming = topology.IncomingValueSlots();
  const auto& vdofs = topology.SignedElementTraceVDofs();
  const std::size_t expected = static_cast<std::size_t>(topology.LocalElementCount()) * 1600;
  local = local && slots.size() == expected && vdofs.size() ==
      static_cast<std::size_t>(topology.LocalElementCount()) * 40;
  std::vector<std::uint64_t> expected_owned, expected_outgoing;
  std::vector<int> expected_destination;
  for (std::size_t source = 0; local && source < expected; ++source)
  {
    const std::size_t element = source / 1600;
    const int row = static_cast<int>((source % 1600) / 40);
    const int column = static_cast<int>(source % 40);
    const int signed_row = vdofs[element * 40 + row];
    const int signed_column = vdofs[element * 40 + column];
    const int local_row = Decode(signed_row), local_column = Decode(signed_column);
    local = local && slots[source].source_slot == source && local_row >= 0 && local_column >= 0 &&
        local_row < trace->GetVSize() && local_column < trace->GetVSize() &&
        slots[source].signed_row_vdof == signed_row && slots[source].signed_column_vdof == signed_column &&
        slots[source].row_sign == (signed_row < 0 ? -1 : 1) &&
        slots[source].column_sign == (signed_column < 0 ? -1 : 1) &&
        slots[source].global_row == static_cast<std::uint64_t>(trace->GetGlobalTDofNumber(local_row)) &&
        slots[source].global_column == static_cast<std::uint64_t>(trace->GetGlobalTDofNumber(local_column)) &&
        slots[source].owner_rank == owner[static_cast<std::size_t>(local_row)];
    if (owner[static_cast<std::size_t>(local_row)] == rank) expected_owned.push_back(source);
    else { expected_outgoing.push_back(source); expected_destination.push_back(owner[static_cast<std::size_t>(local_row)]); }
  }
  std::vector<std::size_t> order(expected_outgoing.size());
  for (std::size_t index = 0; index < order.size(); ++index) order[index] = index;
  std::stable_sort(order.begin(), order.end(), [&](const std::size_t left, const std::size_t right)
    { return expected_destination[left] < expected_destination[right]; });
  local = local && owned.size() == expected_owned.size() && outgoing.size() == expected_outgoing.size() &&
      owned.size() + outgoing.size() == expected;
  for (std::size_t index = 0; local && index < owned.size(); ++index)
  {
    const auto& slot = owned[index];
    int entry = -1; bool offd = false;
    local = slot.source_slot == expected_owned[index] &&
        oracle_view.Find(static_cast<HYPRE_BigInt>(slot.global_row),
                         static_cast<HYPRE_BigInt>(slot.global_column), entry, offd) &&
        slot.csr_entry == entry && slot.csr_entry_is_offd == offd;
  }
  for (std::size_t index = 0; local && index < outgoing.size(); ++index)
    local = outgoing[index].source_slot == expected_outgoing[order[index]] &&
        outgoing[index].peer_rank == expected_destination[order[index]] &&
        (index == 0 || outgoing[index - 1].peer_rank <= outgoing[index].peer_rank);
  for (const auto& slot : incoming) local = local && slot.csr_entry >= 0;
  if (!All(local)) return false;
  std::vector<double> values(expected);
  for (std::size_t index = 0; index < values.size(); ++index)
    values[index] = static_cast<double>(rank) * 10000000.0 + static_cast<double>(index) + 0.25;
  const bool exchanged = topology.ExchangeOwnedCsrValues(values);
  local = local && exchanged && topology.PublishedIncomingValues().size() == incoming.size();
  for (const double value : topology.PublishedIncomingValues()) local = local && std::isfinite(value);
  if (!All(local)) return false;
  // Independently replay the rank-major caller-value exchange from the
  // reconstructed source ordering, rather than inspecting production buffers.
  std::vector<int> send_counts(static_cast<std::size_t>(ranks), 0),
      receive_counts(static_cast<std::size_t>(ranks), 0), send_offsets(static_cast<std::size_t>(ranks), 0),
      receive_offsets(static_cast<std::size_t>(ranks), 0);
  std::vector<double> expected_send(expected_outgoing.size());
  std::vector<std::uint64_t> expected_key_send(2 * expected_outgoing.size());
  for (std::size_t output = 0; output < order.size(); ++output)
  {
    const std::size_t original = order[output];
    ++send_counts[static_cast<std::size_t>(expected_destination[original])];
    const std::size_t source = static_cast<std::size_t>(expected_outgoing[original]);
    const int row = static_cast<int>((source % 1600) / 40);
    const int column = static_cast<int>(source % 40);
    const std::size_t element = source / 1600;
    expected_send[output] = values[source] * static_cast<double>(
        (vdofs[element * 40 + row] < 0 ? -1 : 1) *
        (vdofs[element * 40 + column] < 0 ? -1 : 1));
    expected_key_send[2 * output] = static_cast<std::uint64_t>(trace->GetGlobalTDofNumber(
        Decode(vdofs[element * 40 + row])));
    expected_key_send[2 * output + 1] = static_cast<std::uint64_t>(trace->GetGlobalTDofNumber(
        Decode(vdofs[element * 40 + column])));
  }
  for (int peer = 1; peer < ranks; ++peer) send_offsets[static_cast<std::size_t>(peer)] =
      send_offsets[static_cast<std::size_t>(peer - 1)] + send_counts[static_cast<std::size_t>(peer - 1)];
  const bool count_exchange = MPI_Alltoall(send_counts.data(), 1, MPI_INT, receive_counts.data(), 1,
      MPI_INT, MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!All(count_exchange)) return false;
  for (int peer = 1; peer < ranks; ++peer) receive_offsets[static_cast<std::size_t>(peer)] =
      receive_offsets[static_cast<std::size_t>(peer - 1)] + receive_counts[static_cast<std::size_t>(peer - 1)];
  local = true;
  for (int peer = 0; peer < ranks; ++peer)
    local = receive_counts[static_cast<std::size_t>(peer)] >= 0 &&
        receive_offsets[static_cast<std::size_t>(peer)] >= 0 && local;
  const std::size_t expected_receive_size = static_cast<std::size_t>(
      receive_offsets.back() + receive_counts.back());
  local = local && expected_receive_size == incoming.size() &&
      expected_receive_size <= static_cast<std::size_t>(std::numeric_limits<int>::max() / 2);
  if (!All(local)) return false;
  std::vector<double> expected_receive(expected_receive_size);
  std::vector<int> send_key_counts(static_cast<std::size_t>(ranks)), receive_key_counts(static_cast<std::size_t>(ranks)),
      send_key_offsets(static_cast<std::size_t>(ranks)), receive_key_offsets(static_cast<std::size_t>(ranks));
  for (int peer = 0; peer < ranks; ++peer)
  {
    send_key_counts[static_cast<std::size_t>(peer)] = 2 * send_counts[static_cast<std::size_t>(peer)];
    receive_key_counts[static_cast<std::size_t>(peer)] = 2 * receive_counts[static_cast<std::size_t>(peer)];
    send_key_offsets[static_cast<std::size_t>(peer)] = 2 * send_offsets[static_cast<std::size_t>(peer)];
    receive_key_offsets[static_cast<std::size_t>(peer)] = 2 * receive_offsets[static_cast<std::size_t>(peer)];
  }
  std::vector<std::uint64_t> expected_key_receive(2 * expected_receive_size);
  const bool key_exchange = MPI_Alltoallv(expected_key_send.data(), send_key_counts.data(), send_key_offsets.data(), MPI_UINT64_T,
      expected_key_receive.data(), receive_key_counts.data(), receive_key_offsets.data(), MPI_UINT64_T,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!All(key_exchange)) return false;
  for (std::size_t index = 0; local && index < incoming.size(); ++index)
  {
    int peer = 0;
    while (peer + 1 < ranks && index >= static_cast<std::size_t>(
        receive_offsets[static_cast<std::size_t>(peer + 1)])) ++peer;
    int entry = -1; bool offd = false;
    local = oracle_view.Find(static_cast<HYPRE_BigInt>(expected_key_receive[2 * index]),
        static_cast<HYPRE_BigInt>(expected_key_receive[2 * index + 1]), entry, offd) &&
        incoming[index].csr_entry == entry && incoming[index].csr_entry_is_offd == offd;
    local = local && incoming[index].peer_rank == peer;
  }
  const bool value_exchange = MPI_Alltoallv(expected_send.data(), send_counts.data(), send_offsets.data(), MPI_DOUBLE,
      expected_receive.data(), receive_counts.data(), receive_offsets.data(), MPI_DOUBLE,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!All(value_exchange)) return false;
  local = local && count_exchange && key_exchange && value_exchange &&
      expected_receive.size() == incoming.size() &&
      expected_receive == topology.PublishedIncomingValues();
  const std::vector<double> published_snapshot = topology.PublishedIncomingValues();
  std::vector<double> invalid(values);
  if (rank == 0 && invalid.empty()) invalid.push_back(0.0);
  else if (rank == 0) invalid[0] = std::numeric_limits<double>::infinity();
  const bool rejected = !topology.ExchangeOwnedCsrValues(invalid);
  local = local && rejected && topology.PublishedIncomingValues() == published_snapshot;
  if (!All(local)) return false;
  const bool retried = topology.ExchangeOwnedCsrValues(values);
  local = local && retried && topology.PublishedIncomingValues() == published_snapshot;
  if (ranks == 1) local = local && outgoing.empty() && incoming.empty() && owned.size() == expected;
  return All(local);
}

bool Fixture(const bool rank_zero_only)
{
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  std::unique_ptr<mfem::ParMesh> mesh;
  bool mesh_ready = false;
  try { mesh = PeriodicThreeByThree(rank_zero_only); mesh_ready = mesh != nullptr; }
  catch (...) {}
  if (!All(mesh_ready)) return false;
  Topology topology;
  const bool setup = topology.Setup(*mesh);
  bool local = setup && ComponentMajorMaps(topology) && RetainedCsrAndExchange(topology) && topology.IdentityPrimed() &&
      topology.GlobalTrueTraceSize() == 180 && std::isfinite(topology.PrimeResidual()) &&
      topology.PrimeResidual() <= 1.0e-12 && topology.PrimeMaximumSolutionError() <= 1.0e-12 &&
      topology.PrimeUsedDeviceIlu0();
  const auto& arena = topology.Arena();
  local = local && arena.integer_physical_bytes > 0 && arena.double_physical_bytes > 0 &&
      arena.interior_map_offset % static_cast<std::int64_t>(alignof(int)) == 0 &&
      arena.trace_map_offset % static_cast<std::int64_t>(alignof(int)) == 0 &&
      arena.vertices_offset % static_cast<std::int64_t>(alignof(double)) == 0;
  long long local_true = topology.LocalTrueTraceSize(), global_true = 0;
  const bool true_sum = MPI_Allreduce(&local_true, &global_true, 1, MPI_LONG_LONG,
                                      MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS;
  int local_elements = topology.LocalElementCount(), global_elements = 0;
  const bool element_sum = MPI_Allreduce(&local_elements, &global_elements, 1, MPI_INT,
                                         MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS;
  local = local && true_sum && element_sum && global_true == 180 && global_elements == 9;
  if (ranks == 1)
    local = local && topology.LocalElementCount() == 9 && topology.LocalTraceVSize() == 180 &&
        topology.LocalTrueTraceSize() == 180 && topology.InteriorSpace()->GetVSize() == 252;
  else if (rank_zero_only)
    local = local && (rank == 0 ? topology.LocalElementCount() == 9 &&
        topology.LocalTraceVSize() == 180 && topology.LocalTrueTraceSize() == 180 &&
        topology.InteriorSpace()->GetVSize() == 252 && topology.OwnedCsrSlots().size() == 9 * 1600 &&
        topology.OutgoingValueSlots().empty() && topology.IncomingValueSlots().empty() : topology.LocalElementCount() == 0 &&
        topology.LocalTraceVSize() == 0 && topology.LocalTrueTraceSize() == 0 &&
        topology.InteriorSpace()->GetVSize() == 0 && arena.integer_logical_bytes == 0 &&
        arena.double_logical_bytes > 0 && topology.SignedElementInteriorVDofs().empty() &&
        topology.SignedElementTraceVDofs().empty() && topology.RetainedCsrSlots().empty() &&
        topology.OwnedCsrSlots().empty() && topology.OutgoingValueSlots().empty() &&
        topology.IncomingValueSlots().empty() && topology.PublishedIncomingValues().empty());
  else
  {
    local = local && (rank == 0 ? topology.LocalElementCount() == 5 &&
        topology.LocalTraceVSize() == 160 && topology.LocalTrueTraceSize() > 0 &&
        topology.InteriorSpace()->GetVSize() == 140 : topology.LocalElementCount() == 4 &&
        topology.LocalTraceVSize() == 140 && topology.LocalTrueTraceSize() > 0 &&
        topology.InteriorSpace()->GetVSize() == 112);
    const unsigned long long local_traffic[2] = {
        static_cast<unsigned long long>(topology.OutgoingValueSlots().size()),
        static_cast<unsigned long long>(topology.IncomingValueSlots().size())};
    unsigned long long global_traffic[2] = {0, 0};
    const bool traffic_reduced = MPI_Allreduce(local_traffic, global_traffic, 2,
        MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool directional = rank == 0 ? topology.OutgoingValueSlots().empty() &&
        !topology.IncomingValueSlots().empty() : !topology.OutgoingValueSlots().empty() &&
        topology.IncomingValueSlots().empty();
    local = local && traffic_reduced && global_traffic[0] == global_traffic[1] &&
        global_traffic[0] > 0 && directional;
    int local_offd = 0, global_offd = 0;
    for (const auto& slot : topology.OwnedCsrSlots()) local_offd = local_offd || slot.csr_entry_is_offd;
    for (const auto& slot : topology.IncomingValueSlots()) local_offd = local_offd || slot.csr_entry_is_offd;
    const bool offd_reduced = MPI_Allreduce(&local_offd, &global_offd, 1, MPI_INT, MPI_MAX,
                                            MPI_COMM_WORLD) == MPI_SUCCESS;
    local = local && offd_reduced && global_offd == 1;
  }
  const auto csr_snapshot = topology.RetainedCsrSlots();
  const auto owned_snapshot = topology.OwnedCsrSlots();
  const auto outgoing_snapshot = topology.OutgoingValueSlots();
  const auto incoming_snapshot = topology.IncomingValueSlots();
  const auto interior_map_snapshot = topology.SignedElementInteriorVDofs();
  const auto trace_map_snapshot = topology.SignedElementTraceVDofs();
  const auto* const trace_space_snapshot = topology.TraceSpace();
  const auto* const interior_space_snapshot = topology.InteriorSpace();
  const auto arena_snapshot = arena;
  const std::vector<double> published_snapshot = topology.PublishedIncomingValues();
  const bool identity_snapshot = topology.IdentityPrimed();
  const double residual_snapshot = topology.PrimeResidual();
  const double solution_error_snapshot = topology.PrimeMaximumSolutionError();
  const bool device_ilu_snapshot = topology.PrimeUsedDeviceIlu0();
  const bool second = topology.Setup(*mesh);
  int empty = topology.LocalElementCount() == 0 ? 1 : 0, empty_total = 0;
  const bool empty_count = MPI_Allreduce(&empty, &empty_total, 1, MPI_INT, MPI_SUM,
                                         MPI_COMM_WORLD) == MPI_SUCCESS;
  local = local && (!rank_zero_only || (empty_count && empty_total == 1));
  return All(local && !second && topology.IsReady() && !topology.IsPoisoned() &&
      SameCsr(topology.RetainedCsrSlots(), csr_snapshot) &&
      SameCsr(topology.OwnedCsrSlots(), owned_snapshot) &&
      SameExchange(topology.OutgoingValueSlots(), outgoing_snapshot) &&
      SameExchange(topology.IncomingValueSlots(), incoming_snapshot) &&
      topology.SignedElementInteriorVDofs() == interior_map_snapshot &&
      topology.SignedElementTraceVDofs() == trace_map_snapshot &&
      topology.TraceSpace() == trace_space_snapshot && topology.InteriorSpace() == interior_space_snapshot &&
      SameArena(topology.Arena(), arena_snapshot) && topology.PublishedIncomingValues() == published_snapshot &&
      topology.IdentityPrimed() == identity_snapshot && topology.PrimeResidual() == residual_snapshot &&
      topology.PrimeMaximumSolutionError() == solution_error_snapshot &&
      topology.PrimeUsedDeviceIlu0() == device_ilu_snapshot);
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string_view mode = argc == 2 ? argv[1] : "";
  int rank = 0, ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  bool pass = (mode.empty() || mode == "--rank0-only") && (ranks == 1 || ranks == 2);
  bool cuda_hypre_ready = false;
  std::unique_ptr<mfem::Device> device;
  try
  {
    device = std::make_unique<mfem::Device>("cuda");
    mfem::Hypre::Init();
    cuda_hypre_ready = mfem::Device::IsConfigured() && mfem::HypreUsingGPU();
  }
  catch (...) { cuda_hypre_ready = false; }
  pass = All(cuda_hypre_ready) && pass;
  if (pass) pass = Fixture(mode == "--rank0-only");
  pass = All(pass);
  if (!pass && rank == 0) std::cerr << "traced-scalar IGR CUDA topology/prime failed\n";
  return pass ? 0 : 1;
}

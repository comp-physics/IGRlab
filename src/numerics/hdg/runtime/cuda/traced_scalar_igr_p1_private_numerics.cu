#include "numerics/hdg/runtime/cuda/traced_scalar_igr_p1_private_numerics.cuh"

#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/trace/trace_dg_mpi.hpp"

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics
{
namespace
{
constexpr std::size_t sentinel_bytes = 1;

bool WorldAll(const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

bool WorldFreshRange(const bool fresh, int& minimum, int& maximum) noexcept
{
  const int value = fresh ? 1 : 0;
  minimum = 0;
  maximum = 0;
  const int minimum_status = MPI_Allreduce(&value, &minimum, 1, MPI_INT, MPI_MIN,
                                           MPI_COMM_WORLD);
  const int maximum_status = MPI_Allreduce(&value, &maximum, 1, MPI_INT, MPI_MAX,
                                           MPI_COMM_WORLD);
  return minimum_status == MPI_SUCCESS && maximum_status == MPI_SUCCESS;
}

bool Add(const std::size_t a, const std::size_t b, std::size_t& out) noexcept
{
  if (a > std::numeric_limits<std::size_t>::max() - b) return false;
  out = a + b;
  return true;
}

bool Multiply(const std::size_t a, const std::size_t b,
              std::size_t& out) noexcept
{
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}

bool Align(const std::size_t value, const std::size_t alignment,
           std::size_t& out) noexcept
{
  const std::size_t remainder = value % alignment;
  return remainder == 0 ? (out = value, true) :
      Add(value, alignment - remainder, out);
}

bool AddSlice(std::size_t& cursor, const std::size_t count,
              const std::size_t width, std::int64_t& offset) noexcept
{
  std::size_t aligned = 0;
  std::size_t bytes = 0;
  if (!Align(cursor, alignof(std::max_align_t), aligned) ||
      !Multiply(count, width, bytes) ||
      aligned > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      !Add(aligned, bytes, cursor)) return false;
  offset = static_cast<std::int64_t>(aligned);
  return true;
}

template <typename T>
bool Resize(std::vector<T>& values, const std::size_t count) noexcept
{
  try { values.resize(count); }
  catch (...) { return false; }
  return values.size() == count;
}

bool Release(void*& token, const cuda_w8a::CudaAllocationSite site) noexcept
{
  if (token == nullptr) return true;
  // A failed policy release leaves the token intact: the poisoned retained
  // object quarantines it rather than claiming the memory was released.
  if (!cuda_w8a::UmpireDeviceAllocatorPolicy::Deallocate(token, site)) return false;
  token = nullptr;
  return true;
}

bool CopyInto(std::vector<unsigned char>& storage, const std::int64_t offset,
              const void* source, const std::size_t bytes) noexcept
{
  if (offset < 0 || static_cast<std::size_t>(offset) > storage.size() ||
      bytes > storage.size() - static_cast<std::size_t>(offset) ||
      (bytes != 0 && source == nullptr)) return false;
  if (bytes != 0) std::memcpy(storage.data() + offset, source, bytes);
  return true;
}

int Sign(const int value) noexcept { return value < 0 ? -1 : 1; }

int Decode(const int value) noexcept
{
  return value == std::numeric_limits<int>::min() ? -1 :
      (value < 0 ? -1 - value : value);
}

bool BigIntToWire(const HYPRE_BigInt value, std::uint64_t& wire) noexcept
{
  if (value < 0) return false;
  wire = static_cast<std::uint64_t>(value);
  return static_cast<HYPRE_BigInt>(wire) == value;
}

bool ValidateCsr(mfem::SparseMatrix& matrix, const int rows) noexcept
{
  const int nnz = matrix.NumNonZeroElems();
  const int* const i = matrix.GetI();
  const int* const j = matrix.GetJ();
  const double* const data = matrix.GetData();
  if (rows < 0 || nnz < 0 || (rows > 0 && i == nullptr) ||
      (nnz > 0 && (j == nullptr || data == nullptr))) return false;
  if (i == nullptr) return rows == 0 && nnz == 0;
  if (i[0] != 0 || i[rows] != nnz) return false;
  for (int row = 0; row < rows; ++row)
  {
    if (i[row] > i[row + 1] || i[row] < 0 || i[row + 1] > nnz) return false;
  }
  return true;
}

bool AffinePositiveElement(mfem::ParMesh& mesh, const int element,
                           std::vector<double>& vertices) noexcept
{
  if (mesh.GetElementBaseGeometry(element) != mfem::Geometry::SQUARE) return false;
  mfem::ElementTransformation* transformation = mesh.GetElementTransformation(element);
  if (transformation == nullptr) return false;
  constexpr std::array<std::array<double, 2>, 5> points{{
      {{0.5, 0.5}}, {{0.0, 0.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}, {{0.0, 1.0}}}};
  std::array<double, 4> center_jacobian{};
  for (int point = 0; point < static_cast<int>(points.size()); ++point)
  {
    mfem::IntegrationPoint ip;
    ip.Set2(points[point][0], points[point][1]);
    transformation->SetIntPoint(&ip);
    const mfem::DenseMatrix& jacobian = transformation->Jacobian();
    if (jacobian.Height() != 2 || jacobian.Width() != 2) return false;
    for (int row = 0; row < 2; ++row)
      for (int column = 0; column < 2; ++column)
      {
        const double value = jacobian(row, column);
        if (!std::isfinite(value)) return false;
        const int index = 2 * row + column;
        if (point == 0) center_jacobian[index] = value;
        else if (std::abs(value - center_jacobian[index]) >
                 64.0 * std::numeric_limits<double>::epsilon() *
                     std::max(1.0, std::abs(center_jacobian[index]))) return false;
      }
    const double determinant = jacobian(0, 0) * jacobian(1, 1) -
        jacobian(0, 1) * jacobian(1, 0);
    if (!std::isfinite(determinant) || determinant <= 0.0) return false;
    if (point != 0)
    {
      mfem::Vector physical;
      transformation->Transform(ip, physical);
      if (physical.Size() != 2 || !std::isfinite(physical[0]) ||
          !std::isfinite(physical[1])) return false;
      const std::size_t offset = static_cast<std::size_t>(element) * 8 +
          static_cast<std::size_t>(point - 1) * 2;
      vertices[offset] = physical[0];
      vertices[offset + 1] = physical[1];
    }
  }
  return true;
}
} // namespace

struct RetainedTopology::Implementation final
{
  mfem::ParMesh* mesh = nullptr;
  long mesh_sequence = -1;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::L2_FECollection> interior_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> interior_space;
  trace::DGInterfaceTraceExchangePlan trace_exchange_plan;
  std::vector<int> signed_interior_vdofs;
  std::vector<int> signed_trace_vdofs;
  std::vector<RetainedCsrSlot> csr_slots;
  std::vector<RetainedCsrSlot> owned_csr_slots;
  std::vector<ValueExchangeSlot> outgoing_value_slots;
  std::vector<ValueExchangeSlot> incoming_value_slots;
  std::vector<std::uint64_t> outgoing_pair_wire;
  std::vector<int> outgoing_counts, outgoing_offsets;
  std::vector<int> incoming_counts, incoming_offsets;
  std::vector<double> outgoing_values, incoming_staging, published_incoming_values;
  std::vector<double> vertices, volume_quadrature, segment_quadrature;
  ArenaPlan arena;
  void* integer_arena = nullptr;
  void* double_arena = nullptr;
  bool integer_release_attempted = false;
  bool double_release_attempted = false;
  std::unique_ptr<mfem::HypreParMatrix> matrix;
  mfem::SparseMatrix csr_diagonal, csr_offdiagonal;
  HYPRE_BigInt* csr_offd_columns = nullptr;
  HYPRE_BigInt csr_row_begin = -1, csr_row_end = -1;
  std::unique_ptr<mfem::HypreParVector> rhs, solution, action;
  std::unique_ptr<mfem::HypreILU> ilu;
  std::unique_ptr<mfem::HypreGMRES> gmres;
  bool identity_primed = false;
  bool device_ilu0 = false;
  double prime_residual = std::numeric_limits<double>::infinity();
  double prime_maximum_solution_error = std::numeric_limits<double>::infinity();
  bool ready = false, poisoned = false;

  ~Implementation()
  {
    // HYPRE/MFEM wrappers must release their views before raw policy storage.
    gmres.reset(); ilu.reset(); action.reset(); solution.reset(); rhs.reset(); matrix.reset();
    if (!integer_release_attempted)
    {
      integer_release_attempted = true;
      (void)Release(integer_arena, cuda_w8a::CudaAllocationSite::persistent_integer_arena);
    }
    if (!double_release_attempted)
    {
      double_release_attempted = true;
      (void)Release(double_arena, cuda_w8a::CudaAllocationSite::persistent_double_arena);
    }
  }

  bool Consistent() const noexcept
  {
    return ready && !poisoned && mesh != nullptr &&
        mesh->GetSequence() == mesh_sequence && trace_collection && interior_collection &&
        trace_space && interior_space && trace_space->GetVDim() == trace_components &&
        interior_space->GetVDim() == interior_components &&
        trace_space->GetOrdering() == mfem::Ordering::byNODES &&
        interior_space->GetOrdering() == mfem::Ordering::byNODES &&
        trace_exchange_plan.IsConsistent() && integer_arena != nullptr &&
        double_arena != nullptr && matrix && rhs && solution && action && ilu && gmres &&
        identity_primed && std::isfinite(prime_residual) &&
        std::isfinite(prime_maximum_solution_error) &&
        arena.interior_values == signed_interior_vdofs.size() &&
        arena.trace_values == signed_trace_vdofs.size() &&
        csr_slots.size() == arena.symbolic_slots &&
        owned_csr_slots.size() + outgoing_value_slots.size() == csr_slots.size() &&
        outgoing_pair_wire.size() == 2 * outgoing_value_slots.size() &&
        outgoing_counts.size() == incoming_counts.size() &&
        outgoing_offsets.size() == outgoing_counts.size() + 1 &&
        incoming_offsets.size() == incoming_counts.size() + 1 &&
        !outgoing_offsets.empty() && !incoming_offsets.empty() &&
        outgoing_offsets.back() == static_cast<int>(outgoing_value_slots.size()) &&
        incoming_offsets.back() == static_cast<int>(incoming_value_slots.size()) &&
        outgoing_values.size() == outgoing_value_slots.size() &&
        incoming_staging.size() == incoming_value_slots.size() &&
        published_incoming_values.size() == incoming_value_slots.size() &&
        arena.integer_physical_bytes >= arena.integer_logical_bytes &&
        arena.double_physical_bytes >= arena.double_logical_bytes &&
        matrix->Height() == trace_space->GetTrueVSize() &&
        matrix->Width() == trace_space->GetTrueVSize() &&
        rhs->Size() == trace_space->GetTrueVSize() &&
        solution->Size() == trace_space->GetTrueVSize() &&
        action->Size() == trace_space->GetTrueVSize();
  }
};

namespace
{
bool CacheCsrViews(RetainedTopology::Implementation& state) noexcept
{
  try
  {
    if (!state.matrix || !state.trace_space) return false;
    state.matrix->HostReadWrite();
    state.csr_offd_columns = nullptr;
    state.matrix->GetDiag(state.csr_diagonal);
    state.matrix->GetOffd(state.csr_offdiagonal, state.csr_offd_columns);
    const int rows = state.csr_diagonal.Height();
    const HYPRE_BigInt* starts = state.matrix->GetRowStarts();
    const int rank = state.trace_exchange_plan.map.dofs.rank;
    if (!starts || rows != state.trace_space->GetTrueVSize() || rows < 0 ||
        state.csr_offdiagonal.Height() != rows ||
        !ValidateCsr(state.csr_diagonal, rows) || !ValidateCsr(state.csr_offdiagonal, rows)) return false;
    state.csr_row_begin = HYPRE_AssumedPartitionCheck() ? starts[0] : starts[rank];
    if (state.csr_row_begin < 0 ||
        (rows > 0 && state.csr_row_begin > std::numeric_limits<HYPRE_BigInt>::max() - rows)) return false;
    state.csr_row_end = state.csr_row_begin + rows;
    return state.csr_row_end >= state.csr_row_begin;
  }
  catch (...) { return false; }
}

bool FindCsrEntry(RetainedTopology::Implementation& state,
                  const HYPRE_BigInt row, const HYPRE_BigInt column,
                  int& entry, bool& is_offd) noexcept
{
  entry = -1; is_offd = false;
  try
  {
    const int rows = state.csr_diagonal.Height();
    const int* di = state.csr_diagonal.GetI(); const int* dj = state.csr_diagonal.GetJ();
    const int* oi = state.csr_offdiagonal.GetI(); const int* oj = state.csr_offdiagonal.GetJ();
    if (rows < 0 || state.csr_offdiagonal.Height() != rows ||
        state.csr_row_begin < 0 || state.csr_row_end < state.csr_row_begin ||
        row < state.csr_row_begin || row >= state.csr_row_end ||
        (rows > 0 && di == nullptr)) return false;
    const int local_row = static_cast<int>(row - state.csr_row_begin);
    if (column >= state.csr_row_begin && column < state.csr_row_end)
    {
      if (dj == nullptr) return false;
      const int local_column = static_cast<int>(column - state.csr_row_begin);
      for (int i = di[local_row]; i < di[local_row + 1]; ++i)
        if (dj[i] == local_column) { entry = i; return true; }
      return false;
    }
    if (oi == nullptr || oj == nullptr || state.csr_offd_columns == nullptr) return false;
    for (int i = oi[local_row]; i < oi[local_row + 1]; ++i)
      if (oj[i] >= 0 && oj[i] < state.csr_offdiagonal.Width() &&
          state.csr_offd_columns[oj[i]] == column)
      { entry = i; is_offd = true; return true; }
  }
  catch (...) { return false; }
  return false;
}

bool AdmitLocalMesh(mfem::ParMesh& mesh) noexcept
{
  try
  {
    const mfem::GridFunction* nodes = mesh.GetNodes();
    if (mesh.Dimension() != RetainedTopology::dimension ||
        mesh.SpaceDimension() != RetainedTopology::dimension || mesh.GetNE() < 0 ||
        generic::HasPhysicalBoundaryFace(mesh)) return false;
    for (int element = 0; element < mesh.GetNE(); ++element)
      if (mesh.GetElementBaseGeometry(element) != mfem::Geometry::SQUARE ||
          (nodes != nullptr && (!nodes->FESpace() ||
              nodes->FESpace()->GetFE(element)->GetOrder() != 1))) return false;
    return true;
  }
  catch (...) { return false; }
}

bool AllocateCollections(RetainedTopology::Implementation& state) noexcept
{
  try
  {
    state.trace_collection = std::make_unique<mfem::DG_Interface_FECollection>(
        1, 2, mfem::FiniteElement::VALUE, mfem::BasisType::GaussLegendre);
    state.interior_collection = std::make_unique<mfem::L2_FECollection>(
        1, 2, mfem::BasisType::GaussLegendre, mfem::FiniteElement::VALUE);
    return state.trace_collection != nullptr && state.interior_collection != nullptr;
  }
  catch (...) { return false; }
}

bool ConstructTraceSpace(RetainedTopology::Implementation& state,
                         mfem::ParMesh& mesh) noexcept
{
  try
  {
    if (!state.trace_collection) return false;
    state.trace_space = std::make_unique<mfem::ParFiniteElementSpace>(
        &mesh, state.trace_collection.get(), 5, mfem::Ordering::byNODES);
    return state.trace_space != nullptr && state.trace_space->GetVSize() >= 0 &&
        state.trace_space->GetTrueVSize() >= 0;
  }
  catch (...) { return false; }
}

bool ConstructInteriorSpace(RetainedTopology::Implementation& state,
                            mfem::ParMesh& mesh) noexcept
{
  try
  {
    if (!state.interior_collection) return false;
    state.interior_space = std::make_unique<mfem::ParFiniteElementSpace>(
        &mesh, state.interior_collection.get(), 7, mfem::Ordering::byNODES);
    return state.interior_space != nullptr && state.interior_space->GetVSize() >= 0;
  }
  catch (...) { return false; }
}

bool BuildLocalMapsGeometryQuadrature(RetainedTopology::Implementation& state,
                                      mfem::ParMesh& mesh) noexcept
{
  try
  {
    if (!state.trace_space || !state.interior_space) return false;
    state.mesh = &mesh;
    state.mesh_sequence = mesh.GetSequence();
    const std::size_t elements = static_cast<std::size_t>(mesh.GetNE());
    std::size_t interior_count = 0, trace_count = 0, vertex_count = 0;
    if (!Multiply(elements, 28, interior_count) || !Multiply(elements, 40, trace_count) ||
        !Multiply(elements, 8, vertex_count) || !Resize(state.signed_interior_vdofs, interior_count) ||
        !Resize(state.signed_trace_vdofs, trace_count) || !Resize(state.vertices, vertex_count)) return false;
    for (int element = 0; element < mesh.GetNE(); ++element)
    {
      mfem::Array<int> interior, trace;
      state.interior_space->GetElementVDofs(element, interior);
      state.trace_space->GetElementVDofs(element, trace);
      if (interior.Size() != 28 || trace.Size() != 40) return false;
      std::copy(interior.HostRead(), interior.HostRead() + interior.Size(), state.signed_interior_vdofs.begin() +
          static_cast<std::size_t>(element) * 28);
      std::copy(trace.HostRead(), trace.HostRead() + trace.Size(), state.signed_trace_vdofs.begin() +
          static_cast<std::size_t>(element) * 40);
      if (!AffinePositiveElement(mesh, element, state.vertices)) return false;
    }
    const mfem::IntegrationRule& volume = mfem::IntRules.Get(mfem::Geometry::SQUARE, 12);
    const mfem::IntegrationRule& segment = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 12);
    if (volume.GetNPoints() <= 0 || segment.GetNPoints() <= 0 ||
        !Resize(state.volume_quadrature, static_cast<std::size_t>(volume.GetNPoints()) * 3) ||
        !Resize(state.segment_quadrature, static_cast<std::size_t>(segment.GetNPoints()) * 2)) return false;
    for (int q = 0; q < volume.GetNPoints(); ++q)
    {
      const auto& ip = volume.IntPoint(q);
      if (!std::isfinite(ip.x) || !std::isfinite(ip.y) || !std::isfinite(ip.weight)) return false;
      state.volume_quadrature[3 * q] = ip.x; state.volume_quadrature[3 * q + 1] = ip.y;
      state.volume_quadrature[3 * q + 2] = ip.weight;
    }
    for (int q = 0; q < segment.GetNPoints(); ++q)
    {
      const auto& ip = segment.IntPoint(q);
      if (!std::isfinite(ip.x) || !std::isfinite(ip.weight)) return false;
      state.segment_quadrature[2 * q] = ip.x; state.segment_quadrature[2 * q + 1] = ip.weight;
    }
    state.arena.element_count = elements;
    state.arena.interior_values = interior_count;
    state.arena.trace_values = trace_count;
    state.arena.volume_quadrature_values = state.volume_quadrature.size();
    state.arena.segment_quadrature_values = state.segment_quadrature.size();
    std::size_t integer_cursor = 0, double_cursor = 0;
    if (!AddSlice(integer_cursor, interior_count, sizeof(int), state.arena.interior_map_offset) ||
        !AddSlice(integer_cursor, trace_count, sizeof(int), state.arena.trace_map_offset) ||
        !AddSlice(double_cursor, state.vertices.size(), sizeof(double), state.arena.vertices_offset) ||
        !AddSlice(double_cursor, state.volume_quadrature.size(), sizeof(double), state.arena.volume_quadrature_offset) ||
        !AddSlice(double_cursor, state.segment_quadrature.size(), sizeof(double), state.arena.segment_quadrature_offset)) return false;
    state.arena.integer_logical_bytes = integer_cursor;
    state.arena.double_logical_bytes = double_cursor;
    state.arena.integer_physical_bytes = std::max(integer_cursor, sentinel_bytes);
    state.arena.double_physical_bytes = std::max(double_cursor, sentinel_bytes);
    return true;
  }
  catch (...) { return false; }
}

bool BuildRetainedCsrMetadata(RetainedTopology::Implementation& state) noexcept
{
  try
  {
    const auto& map = state.trace_exchange_plan.map.dofs;
    const int ranks = map.rank_count;
    const int rank = map.rank;
    std::size_t slot_count = 0;
    if (ranks <= 0 || rank < 0 || rank >= ranks ||
        !Multiply(state.arena.element_count, 1600, slot_count) ||
        state.signed_trace_vdofs.size() != state.arena.trace_values ||
        map.entries.size() != static_cast<std::size_t>(state.trace_space->GetVSize()) ||
        !Resize(state.csr_slots, slot_count)) return false;
    state.owned_csr_slots.clear(); state.incoming_value_slots.clear();
    state.outgoing_value_slots.clear(); state.outgoing_pair_wire.clear();
    for (std::size_t element = 0; element < state.arena.element_count; ++element)
      for (int row = 0; row < 40; ++row)
        for (int column = 0; column < 40; ++column)
        {
          const std::size_t source = element * 1600 +
              static_cast<std::size_t>(row) * 40 + static_cast<std::size_t>(column);
          const int signed_row = state.signed_trace_vdofs[element * 40 + row];
          const int signed_column = state.signed_trace_vdofs[element * 40 + column];
          const int local_row = Decode(signed_row);
          const int local_column = Decode(signed_column);
          if (local_row < 0 || local_column < 0 ||
              static_cast<std::size_t>(local_row) >= map.entries.size() ||
              static_cast<std::size_t>(local_column) >= map.entries.size()) return false;
          const auto& row_entry = map.entries[static_cast<std::size_t>(local_row)];
          const auto& column_entry = map.entries[static_cast<std::size_t>(local_column)];
          std::uint64_t global_row = 0, global_column = 0;
          if (row_entry.global_true_dof < 0 || column_entry.global_true_dof < 0 ||
              row_entry.owner_rank < 0 || row_entry.owner_rank >= ranks ||
              !BigIntToWire(static_cast<HYPRE_BigInt>(row_entry.global_true_dof), global_row) ||
              !BigIntToWire(static_cast<HYPRE_BigInt>(column_entry.global_true_dof), global_column))
            return false;
          RetainedTopology::RetainedCsrSlot slot;
          slot.source_slot = static_cast<std::uint64_t>(source);
          slot.signed_row_vdof = signed_row;
          slot.signed_column_vdof = signed_column;
          slot.global_row = global_row; slot.global_column = global_column;
          slot.owner_rank = row_entry.owner_rank;
          slot.row_sign = Sign(signed_row); slot.column_sign = Sign(signed_column);
          if (slot.owner_rank == rank)
          {
            if (!FindCsrEntry(state, static_cast<HYPRE_BigInt>(global_row),
                              static_cast<HYPRE_BigInt>(global_column),
                              slot.csr_entry, slot.csr_entry_is_offd)) return false;
            state.owned_csr_slots.push_back(slot);
          }
          else
          {
            state.outgoing_value_slots.push_back({slot.source_slot, slot.owner_rank, -1, false});
            state.outgoing_pair_wire.push_back(global_row);
            state.outgoing_pair_wire.push_back(global_column);
          }
          state.csr_slots[source] = slot;
        }
    state.arena.symbolic_slots = slot_count;
    if (state.outgoing_value_slots.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2) ||
        state.outgoing_pair_wire.size() != 2 * state.outgoing_value_slots.size()) return false;
    std::vector<std::size_t> order(state.outgoing_value_slots.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](const std::size_t left, const std::size_t right)
      { return state.outgoing_value_slots[left].peer_rank < state.outgoing_value_slots[right].peer_rank; });
    std::vector<RetainedTopology::ValueExchangeSlot> sorted_slots(
        state.outgoing_value_slots.size());
    std::vector<std::uint64_t> sorted_wire(state.outgoing_pair_wire.size());
    for (std::size_t index = 0; index < order.size(); ++index)
    {
      sorted_slots[index] = state.outgoing_value_slots[order[index]];
      sorted_wire[2 * index] = state.outgoing_pair_wire[2 * order[index]];
      sorted_wire[2 * index + 1] = state.outgoing_pair_wire[2 * order[index] + 1];
    }
    state.outgoing_value_slots.swap(sorted_slots); state.outgoing_pair_wire.swap(sorted_wire);
    if (!Resize(state.incoming_counts, static_cast<std::size_t>(ranks)) ||
        !Resize(state.incoming_offsets, static_cast<std::size_t>(ranks) + 1) ||
        !Resize(state.outgoing_counts, static_cast<std::size_t>(ranks)) ||
        !Resize(state.outgoing_offsets, static_cast<std::size_t>(ranks) + 1)) return false;
    std::fill(state.incoming_counts.begin(), state.incoming_counts.end(), 0);
    std::fill(state.outgoing_counts.begin(), state.outgoing_counts.end(), 0);
    for (const auto& slot : state.outgoing_value_slots)
    {
      if (slot.peer_rank < 0 || slot.peer_rank >= ranks || slot.peer_rank == rank ||
          state.outgoing_counts[static_cast<std::size_t>(slot.peer_rank)] == std::numeric_limits<int>::max()) return false;
      ++state.outgoing_counts[static_cast<std::size_t>(slot.peer_rank)];
    }
    for (int peer = 0; peer < ranks; ++peer)
    {
      if (peer > 0 && state.outgoing_offsets[static_cast<std::size_t>(peer - 1)] >
          std::numeric_limits<int>::max() - state.outgoing_counts[static_cast<std::size_t>(peer - 1)]) return false;
      state.outgoing_offsets[static_cast<std::size_t>(peer)] = peer == 0 ? 0 :
          state.outgoing_offsets[static_cast<std::size_t>(peer - 1)] + state.outgoing_counts[static_cast<std::size_t>(peer - 1)];
    }
    state.outgoing_offsets[static_cast<std::size_t>(ranks)] = static_cast<int>(state.outgoing_value_slots.size());
    return true;
  }
  catch (...) { return false; }
}

bool BuildValueExchangePlan(RetainedTopology::Implementation& state) noexcept
{
  const int ranks = state.trace_exchange_plan.map.dofs.rank_count;
  bool local_ready = ranks > 0 && state.outgoing_pair_wire.size() == 2 * state.outgoing_value_slots.size();
  if (!WorldAll(local_ready)) return false;
  bool collective_ok = MPI_Alltoall(state.outgoing_counts.data(), 1, MPI_INT,
      state.incoming_counts.data(), 1, MPI_INT, MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!WorldAll(collective_ok)) return false;
  std::size_t incoming = 0;
  local_ready = true;
  for (int peer = 0; peer < ranks; ++peer)
  {
    const int count = state.incoming_counts[static_cast<std::size_t>(peer)];
    local_ready = count >= 0 && incoming <= static_cast<std::size_t>(std::numeric_limits<int>::max()) - static_cast<std::size_t>(count) && local_ready;
    if (local_ready) { state.incoming_offsets[static_cast<std::size_t>(peer)] = static_cast<int>(incoming); incoming += static_cast<std::size_t>(count); }
  }
  if (ranks > 0) state.incoming_offsets[static_cast<std::size_t>(ranks)] = static_cast<int>(incoming);
  std::vector<int> send_key_counts, receive_key_counts, send_key_offsets, receive_key_offsets;
  try
  {
    send_key_counts.resize(static_cast<std::size_t>(ranks));
    receive_key_counts.resize(static_cast<std::size_t>(ranks));
    send_key_offsets.resize(static_cast<std::size_t>(ranks));
    receive_key_offsets.resize(static_cast<std::size_t>(ranks));
  }
  catch (...) { local_ready = false; }
  local_ready = local_ready && state.outgoing_offsets[static_cast<std::size_t>(ranks)] <=
      std::numeric_limits<int>::max() / 2 &&
      state.incoming_offsets[static_cast<std::size_t>(ranks)] <= std::numeric_limits<int>::max() / 2;
  if (!WorldAll(local_ready)) return false;
  for (int peer = 0; peer < ranks; ++peer)
  {
    local_ready = state.outgoing_counts[static_cast<std::size_t>(peer)] <= std::numeric_limits<int>::max() / 2 &&
        state.incoming_counts[static_cast<std::size_t>(peer)] <= std::numeric_limits<int>::max() / 2 &&
        state.outgoing_offsets[static_cast<std::size_t>(peer)] <= std::numeric_limits<int>::max() / 2 &&
        state.incoming_offsets[static_cast<std::size_t>(peer)] <= std::numeric_limits<int>::max() / 2 && local_ready;
    if (local_ready)
    { send_key_counts[peer] = 2 * state.outgoing_counts[peer]; receive_key_counts[peer] = 2 * state.incoming_counts[peer];
      send_key_offsets[peer] = 2 * state.outgoing_offsets[peer]; receive_key_offsets[peer] = 2 * state.incoming_offsets[peer]; }
  }
  std::vector<std::uint64_t> incoming_wire;
  try { if (local_ready) incoming_wire.resize(2 * incoming); else incoming_wire.clear(); }
  catch (...) { local_ready = false; }
  if (!WorldAll(local_ready)) return false;
  collective_ok = MPI_Alltoallv(state.outgoing_pair_wire.data(), send_key_counts.data(), send_key_offsets.data(), MPI_UINT64_T,
      incoming_wire.data(), receive_key_counts.data(), receive_key_offsets.data(), MPI_UINT64_T, MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!WorldAll(collective_ok)) return false;
  try
  {
    state.incoming_value_slots.resize(incoming);
    for (std::size_t index = 0; index < incoming; ++index)
    {
      int peer = 0;
      while (peer + 1 < ranks && index >= static_cast<std::size_t>(
          state.incoming_offsets[static_cast<std::size_t>(peer + 1)])) ++peer;
      int entry = -1; bool offd = false;
      local_ready = incoming_wire[2 * index] <= static_cast<std::uint64_t>(std::numeric_limits<HYPRE_BigInt>::max()) &&
          incoming_wire[2 * index + 1] <= static_cast<std::uint64_t>(std::numeric_limits<HYPRE_BigInt>::max()) && local_ready;
      if (local_ready) local_ready = FindCsrEntry(state, static_cast<HYPRE_BigInt>(incoming_wire[2 * index]),
          static_cast<HYPRE_BigInt>(incoming_wire[2 * index + 1]), entry, offd);
      state.incoming_value_slots[index] = {0, peer, entry, offd};
    }
    local_ready = local_ready && Resize(state.outgoing_values, state.outgoing_value_slots.size()) &&
        Resize(state.incoming_staging, incoming) && Resize(state.published_incoming_values, incoming);
  }
  catch (...) { local_ready = false; }
  return WorldAll(local_ready);
}

bool AllocatePackedArenas(RetainedTopology::Implementation& state) noexcept
{
  const bool integer_ok = cuda_w8a::UmpireDeviceAllocatorPolicy::Allocate(
      state.arena.integer_physical_bytes, alignof(std::max_align_t),
      cuda_w8a::CudaAllocationSite::persistent_integer_arena, state.integer_arena);
  const bool double_ok = cuda_w8a::UmpireDeviceAllocatorPolicy::Allocate(
      state.arena.double_physical_bytes, alignof(std::max_align_t),
      cuda_w8a::CudaAllocationSite::persistent_double_arena, state.double_arena);
  if (integer_ok && double_ok) return true;
  // Preserve any successful allocation in the poisoned candidate.  Its
  // destructor makes the single permitted release attempt after all wrappers
  // have been destroyed; a failed release remains quarantined, not retried.
  return false;
}

bool UploadPackedArenas(RetainedTopology::Implementation& state) noexcept
{
  try
  {
    std::vector<unsigned char> integers(state.arena.integer_physical_bytes, 0);
    std::vector<unsigned char> doubles(state.arena.double_physical_bytes, 0);
    const bool packed = CopyInto(integers, state.arena.interior_map_offset,
        state.signed_interior_vdofs.data(), state.signed_interior_vdofs.size() * sizeof(int)) &&
        CopyInto(integers, state.arena.trace_map_offset, state.signed_trace_vdofs.data(),
                 state.signed_trace_vdofs.size() * sizeof(int)) &&
        CopyInto(doubles, state.arena.vertices_offset, state.vertices.data(),
                 state.vertices.size() * sizeof(double)) &&
        CopyInto(doubles, state.arena.volume_quadrature_offset, state.volume_quadrature.data(),
                 state.volume_quadrature.size() * sizeof(double)) &&
        CopyInto(doubles, state.arena.segment_quadrature_offset, state.segment_quadrature.data(),
                 state.segment_quadrature.size() * sizeof(double));
    return packed && cuda_w8a::CudaTransportPolicy::HostToDevice(
        state.integer_arena, integers.data(), integers.size(),
        cuda_w8a::CudaTransferCategory::topology, cuda_w8a::CudaTransferSite::map_upload,
        integers.size(), integers.size()) && cuda_w8a::CudaTransportPolicy::HostToDevice(
        state.double_arena, doubles.data(), doubles.size(),
        cuda_w8a::CudaTransferCategory::topology, cuda_w8a::CudaTransferSite::quadrature_upload,
        doubles.size(), doubles.size()) && cuda_w8a::CudaTransportPolicy::Synchronize(
        cuda_w8a::CudaTransferSite::quadrature_upload);
  }
  catch (...) { return false; }
}

bool BuildGraphAndForm(RetainedTopology::Implementation& state,
                       std::unique_ptr<mfem::SparseMatrix>& graph,
                       std::unique_ptr<mfem::ParBilinearForm>& form) noexcept
{
  try
  {
    const int local = state.trace_space->GetVSize();
    graph = std::make_unique<mfem::SparseMatrix>(local);
    mfem::DenseMatrix signed_pattern(40);
    for (int element = 0; element < state.mesh->GetNE(); ++element)
    {
      mfem::Array<int> dofs(40);
      std::copy_n(state.signed_trace_vdofs.data() + static_cast<std::size_t>(element) * 40,
                  40, dofs.HostReadWrite());
      for (int row = 0; row < 40; ++row)
        for (int column = 0; column < 40; ++column)
          signed_pattern(row, column) = static_cast<double>(Sign(dofs[row]) * Sign(dofs[column]));
      graph->AddSubMatrix(dofs, dofs, signed_pattern, 1);
    }
    graph->Finalize(0, false);
    form = std::make_unique<mfem::ParBilinearForm>(state.trace_space.get());
    return graph && form && state.arena.symbolic_slots == state.csr_slots.size();
  }
  catch (...) { return false; }
}

bool MakeIdentity(RetainedTopology::Implementation& state) noexcept
{
  try
  {
    const int rows = state.trace_space->GetTrueVSize();
    state.matrix->HostReadWrite();
    mfem::SparseMatrix diagonal, offdiagonal;
    HYPRE_BigInt* offd_columns = nullptr;
    state.matrix->GetDiag(diagonal);
    state.matrix->GetOffd(offdiagonal, offd_columns);
    if (!ValidateCsr(diagonal, rows) || !ValidateCsr(offdiagonal, rows)) return false;
    double* data = diagonal.GetData();
    const int* i = diagonal.GetI();
    const int* j = diagonal.GetJ();
    if (diagonal.NumNonZeroElems() > 0) std::fill(data, data + diagonal.NumNonZeroElems(), 0.0);
    for (int row = 0; row < rows; ++row)
    {
      int matches = 0;
      for (int entry = i[row]; entry < i[row + 1]; ++entry)
        if (j[entry] == row) { data[entry] = 1.0; ++matches; }
      if (matches != 1) return false;
    }
    if (offdiagonal.NumNonZeroElems() > 0)
      std::fill(offdiagonal.GetData(), offdiagonal.GetData() + offdiagonal.NumNonZeroElems(), 0.0);
    state.matrix->HypreReadWrite();
    return state.matrix->Height() == rows && state.matrix->Width() == rows &&
        state.matrix->GetGlobalNumRows() == state.trace_space->GlobalTrueVSize() &&
        state.matrix->GetGlobalNumCols() == state.trace_space->GlobalTrueVSize();
  }
  catch (...) { return false; }
}

bool PrimeIdentity(RetainedTopology::Implementation& state, const int ranks) noexcept
{
  bool local = false;
  try
  {
    state.rhs = std::make_unique<mfem::HypreParVector>(state.trace_space.get());
    state.solution = std::make_unique<mfem::HypreParVector>(state.trace_space.get());
    state.action = std::make_unique<mfem::HypreParVector>(state.trace_space.get());
    state.ilu = std::make_unique<mfem::HypreILU>();
    state.gmres = std::make_unique<mfem::HypreGMRES>(MPI_COMM_WORLD);
    state.rhs->UseDevice(true); state.solution->UseDevice(true); state.action->UseDevice(true);
    state.matrix->HypreReadWrite(); state.rhs->HypreReadWrite();
    state.solution->HypreReadWrite(); state.action->HypreReadWrite();
    local = cuda_w8a::detail::CudaHypreDevicePolicyValid(*state.matrix, *state.rhs,
        *state.solution);
  }
  catch (...) { local = false; }
  if (!WorldAll(local)) return false;
  cuda_w8a::detail::CudaHypreSetupOrderEvidence order{};
  try { local = cuda_w8a::detail::CudaConfigurePersistentHypre(*state.matrix, *state.ilu,
      *state.gmres, ranks, 0, order) && order.ConfigurationExactForGeneration(0); }
  catch (...) { local = false; }
  if (!WorldAll(local)) return false;
  try
  {
    *state.rhs = 1.0; *state.solution = 0.0; *state.action = 0.0;
    state.rhs->HypreRead(); state.solution->HypreReadWrite(); state.action->HypreReadWrite();
    const bool policy = cuda_w8a::detail::CudaHypreDevicePolicyValid(*state.matrix, *state.rhs,
        *state.solution);
    const int incoming_error = HYPRE_GetError();
    const int clear_status = HYPRE_ClearAllErrors();
    local = policy && incoming_error == 0 && clear_status == 0;
  }
  catch (...) { local = false; }
  if (!WorldAll(local)) return false;
  int setup_status = -1, setup_error = -1;
  try { setup_status = HYPRE_ParCSRGMRESSetup(static_cast<HYPRE_Solver>(*state.gmres),
      *state.matrix, *state.rhs, *state.solution); setup_error = HYPRE_GetError(); }
  catch (...) {}
  if (!WorldAll(setup_status == 0 && setup_error == 0)) return false;
  int solve_status = -1, solve_error = -1;
  try { solve_status = HYPRE_ParCSRGMRESSolve(static_cast<HYPRE_Solver>(*state.gmres),
      *state.matrix, *state.rhs, *state.solution); solve_error = HYPRE_GetError(); }
  catch (...) {}
  if (!WorldAll(solve_status == 0 && solve_error == 0)) return false;
  HYPRE_Int converged = 0, iterations = -1;
  double reported = std::numeric_limits<double>::infinity();
  int converged_status = -1, iterations_status = -1, residual_status = -1, query_error = -1;
  try
  {
    const HYPRE_Solver solver = static_cast<HYPRE_Solver>(*state.gmres);
    converged_status = HYPRE_GMRESGetConverged(solver, &converged);
    iterations_status = HYPRE_ParCSRGMRESGetNumIterations(solver, &iterations);
    residual_status = HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(solver, &reported);
    query_error = HYPRE_GetError();
  }
  catch (...) {}
  local = converged_status == 0 && iterations_status == 0 && residual_status == 0 &&
      query_error == 0 && converged == 1 && iterations >= 0 && std::isfinite(reported) &&
      reported <= 1.0e-10;
  if (!WorldAll(local)) return false;
  double residual = std::numeric_limits<double>::infinity();
  double maximum_error = std::numeric_limits<double>::infinity();
  try
  {
    state.matrix->Mult(*state.solution, *state.action); *state.action -= *state.rhs;
    local = true;
  }
  catch (...) { local = false; }
  if (!WorldAll(local)) return false;
  try
  {
    const double square = mfem::InnerProduct(MPI_COMM_WORLD, *state.action, *state.action);
    residual = square >= 0.0 ? std::sqrt(square) : std::numeric_limits<double>::infinity();
    local = std::isfinite(residual);
  }
  catch (...) { local = false; }
  if (!WorldAll(local)) return false;
  try
  {
    const double* x = state.solution->HostRead();
    local = state.solution->Size() == 0 || x != nullptr;
    maximum_error = 0.0;
    for (int row = 0; row < state.solution->Size() && local; ++row)
      maximum_error = std::max(maximum_error, std::abs(x[row] - 1.0));
  }
  catch (...) { local = false; }
  if (!WorldAll(local)) return false;
  double global_error = 0.0;
  if (MPI_Allreduce(&maximum_error, &global_error, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return false;
  maximum_error = global_error;
  try
  {
    state.prime_residual = residual; state.prime_maximum_solution_error = maximum_error;
    state.device_ilu0 = cuda_w8a::detail::CudaHypreDevicePolicyValid(*state.matrix,
        *state.rhs, *state.solution) && order.ConfigurationExactForGeneration(0) &&
        setup_status == 0 && solve_status == 0 && converged_status == 0 &&
        iterations_status == 0 && residual_status == 0;
    local = state.device_ilu0 && residual <= 1.0e-12 && maximum_error <= 1.0e-12;
    *state.rhs = 0.0; *state.solution = 0.0; *state.action = 0.0;
  }
  catch (...) { local = false; }
  if (!WorldAll(local)) return false;
  state.identity_primed = true;
  return true;
}
} // namespace

RetainedTopology::RetainedTopology() noexcept = default;
RetainedTopology::~RetainedTopology() = default;
RetainedTopology::RetainedTopology(RetainedTopology&&) noexcept = default;
RetainedTopology& RetainedTopology::operator=(RetainedTopology&&) noexcept = default;

bool RetainedTopology::Setup(mfem::ParMesh& mesh) noexcept
{
  int fresh_minimum = 0;
  int fresh_maximum = 0;
  const bool fresh_range = WorldFreshRange(!setup_attempted_, fresh_minimum,
                                           fresh_maximum);
  if (!fresh_range || fresh_minimum != fresh_maximum)
  {
    // A divergent re-entry consumes the fresh ranks too.  This prevents a
    // later invocation from pairing a fresh rank with a terminal peer.
    setup_attempted_ = true;
    terminal_poisoned_ = true;
    if (implementation_) implementation_->poisoned = true;
    return false;
  }
  // A symmetric second call is an ordinary one-shot rejection: preserve the
  // ready object (or an already poisoned candidate) exactly as it was.
  if (fresh_minimum == 0) return false;
  setup_attempted_ = true;
  int relation = MPI_UNEQUAL, ranks = 0;
  const bool admitted = mesh.GetComm() != MPI_COMM_NULL &&
      MPI_Comm_compare(mesh.GetComm(), MPI_COMM_WORLD, &relation) == MPI_SUCCESS &&
      (relation == MPI_IDENT || relation == MPI_CONGRUENT) &&
      MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS && (ranks == 1 || ranks == 2);
  std::unique_ptr<Implementation> candidate;
  try { candidate = std::make_unique<Implementation>(); }
  catch (...) {}
  const auto poison = [&]() noexcept
  {
    terminal_poisoned_ = true;
    if (candidate) { candidate->poisoned = true; implementation_ = std::move(candidate); }
    return false;
  };
  if (!WorldAll(candidate != nullptr)) return poison();
  if (!WorldAll(admitted)) return poison();
  // No rank may reach a collective-capable MFEM constructor until every peer
  // has passed the preceding local-only phase.
  if (!WorldAll(AdmitLocalMesh(mesh))) return poison();
  if (!WorldAll(AllocateCollections(*candidate))) return poison();
  if (!WorldAll(ConstructTraceSpace(*candidate, mesh))) return poison();
  if (!WorldAll(ConstructInteriorSpace(*candidate, mesh))) return poison();
  if (!WorldAll(BuildLocalMapsGeometryQuadrature(*candidate, mesh))) return poison();
  bool plan_ready = false;
  try { plan_ready = trace::BuildDGInterfaceTraceExchangePlan(*candidate->trace_space,
                                                                candidate->trace_exchange_plan); }
  catch (...) { plan_ready = false; }
  if (!WorldAll(plan_ready)) return poison();
  if (!WorldAll(cuda_w8a::UmpireDeviceAllocatorPolicy::PrepareDeviceArenaAllocator())) return poison();
  if (!WorldAll(AllocatePackedArenas(*candidate))) return poison();
  if (!WorldAll(UploadPackedArenas(*candidate))) return poison();
  std::unique_ptr<mfem::SparseMatrix> graph;
  std::unique_ptr<mfem::ParBilinearForm> form;
  if (!WorldAll(BuildGraphAndForm(*candidate, graph, form))) return poison();
  try { candidate->matrix.reset(form->ParallelAssemble(graph.get())); }
  catch (...) { candidate->matrix.reset(); }
  if (!WorldAll(candidate->matrix != nullptr)) return poison();
  if (!WorldAll(CacheCsrViews(*candidate))) return poison();
  // CSR slots refer to the assembled host CSR matrix, so they intentionally
  // live in retained host vectors rather than the earlier topology arenas.
  if (!WorldAll(BuildRetainedCsrMetadata(*candidate))) return poison();
  if (!BuildValueExchangePlan(*candidate)) return poison();
  if (!WorldAll(candidate->matrix != nullptr && MakeIdentity(*candidate))) return poison();
  if (!WorldAll(PrimeIdentity(*candidate, ranks))) return poison();
  long long census[3] = {mesh.GetNE(), candidate->trace_space->GetTrueVSize(),
                         candidate->identity_primed ? 1 : 0};
  long long global[3] = {0, 0, 0};
  const bool census_ok = MPI_Allreduce(census, global, 3, MPI_LONG_LONG, MPI_SUM,
      MPI_COMM_WORLD) == MPI_SUCCESS && global[0] >= 0 && global[1] > 0 && global[2] == ranks;
  if (!census_ok) return poison();
  candidate->ready = true;
  if (!WorldAll(candidate->Consistent())) return poison();
  implementation_ = std::move(candidate);
  return true;
}

bool RetainedTopology::IsReady() const noexcept
{ return implementation_ && implementation_->Consistent(); }
bool RetainedTopology::IsPoisoned() const noexcept
{ return terminal_poisoned_ || (implementation_ && implementation_->poisoned); }
int RetainedTopology::LocalElementCount() const noexcept
{ return IsReady() ? implementation_->mesh->GetNE() : -1; }
int RetainedTopology::LocalTraceVSize() const noexcept
{ return IsReady() ? implementation_->trace_space->GetVSize() : -1; }
int RetainedTopology::LocalTrueTraceSize() const noexcept
{ return IsReady() ? implementation_->trace_space->GetTrueVSize() : -1; }
long long RetainedTopology::GlobalTrueTraceSize() const noexcept
{ return IsReady() ? implementation_->trace_space->GlobalTrueVSize() : -1; }
const mfem::ParFiniteElementSpace* RetainedTopology::TraceSpace() const noexcept
{ return IsReady() ? implementation_->trace_space.get() : nullptr; }
const mfem::ParFiniteElementSpace* RetainedTopology::InteriorSpace() const noexcept
{ return IsReady() ? implementation_->interior_space.get() : nullptr; }
const std::vector<int>& RetainedTopology::SignedElementInteriorVDofs() const noexcept
{ static const std::vector<int> empty; return IsReady() ? implementation_->signed_interior_vdofs : empty; }
const std::vector<int>& RetainedTopology::SignedElementTraceVDofs() const noexcept
{ static const std::vector<int> empty; return IsReady() ? implementation_->signed_trace_vdofs : empty; }
const RetainedTopology::ArenaPlan& RetainedTopology::Arena() const noexcept
{ static const ArenaPlan empty; return IsReady() ? implementation_->arena : empty; }
const std::vector<RetainedTopology::RetainedCsrSlot>& RetainedTopology::RetainedCsrSlots() const noexcept
{ static const std::vector<RetainedCsrSlot> empty; return IsReady() ? implementation_->csr_slots : empty; }
const std::vector<RetainedTopology::RetainedCsrSlot>& RetainedTopology::OwnedCsrSlots() const noexcept
{ static const std::vector<RetainedCsrSlot> empty; return IsReady() ? implementation_->owned_csr_slots : empty; }
const std::vector<RetainedTopology::ValueExchangeSlot>& RetainedTopology::OutgoingValueSlots() const noexcept
{ static const std::vector<ValueExchangeSlot> empty; return IsReady() ? implementation_->outgoing_value_slots : empty; }
const std::vector<RetainedTopology::ValueExchangeSlot>& RetainedTopology::IncomingValueSlots() const noexcept
{ static const std::vector<ValueExchangeSlot> empty; return IsReady() ? implementation_->incoming_value_slots : empty; }

bool RetainedTopology::ExchangeOwnedCsrValues(const std::vector<double>& source_values) noexcept
{
  bool local_ready = IsReady();
  if (!WorldAll(local_ready)) return false;
  bool input_ready = source_values.size() == implementation_->csr_slots.size() &&
      implementation_->outgoing_values.size() == implementation_->outgoing_value_slots.size() &&
      implementation_->incoming_staging.size() == implementation_->incoming_value_slots.size() &&
      implementation_->published_incoming_values.size() == implementation_->incoming_value_slots.size();
  for (const double value : source_values) input_ready = std::isfinite(value) && input_ready;
  if (!WorldAll(input_ready)) return false;
  local_ready = true;
  try
  {
    for (std::size_t index = 0; index < implementation_->outgoing_value_slots.size(); ++index)
    {
      const std::size_t source = static_cast<std::size_t>(implementation_->outgoing_value_slots[index].source_slot);
      local_ready = source < source_values.size() && source < implementation_->csr_slots.size() && local_ready;
      if (source < source_values.size() && source < implementation_->csr_slots.size())
        implementation_->outgoing_values[index] = source_values[source] *
            static_cast<double>(implementation_->csr_slots[source].row_sign *
                                implementation_->csr_slots[source].column_sign);
    }
  }
  catch (...) { local_ready = false; }
  if (!WorldAll(local_ready)) return false;
  const bool exchanged = MPI_Alltoallv(implementation_->outgoing_values.data(),
      implementation_->outgoing_counts.data(), implementation_->outgoing_offsets.data(), MPI_DOUBLE,
      implementation_->incoming_staging.data(), implementation_->incoming_counts.data(),
      implementation_->incoming_offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!WorldAll(exchanged)) return false;
  local_ready = true;
  for (const double value : implementation_->incoming_staging)
    local_ready = std::isfinite(value) && local_ready;
  if (!WorldAll(local_ready)) return false;
  // The buffers are fixed by Setup.  Publication therefore cannot allocate
  // and occurs only after every rank has observed a successful collective.
  std::copy(implementation_->incoming_staging.begin(), implementation_->incoming_staging.end(),
            implementation_->published_incoming_values.begin());
  return true;
}

const std::vector<double>& RetainedTopology::PublishedIncomingValues() const noexcept
{ static const std::vector<double> empty; return IsReady() ? implementation_->published_incoming_values : empty; }
bool RetainedTopology::IdentityPrimed() const noexcept
{ return IsReady() && implementation_->identity_primed; }
double RetainedTopology::PrimeResidual() const noexcept
{ return IsReady() ? implementation_->prime_residual : std::numeric_limits<double>::infinity(); }
double RetainedTopology::PrimeMaximumSolutionError() const noexcept
{ return IsReady() ? implementation_->prime_maximum_solution_error : std::numeric_limits<double>::infinity(); }
bool RetainedTopology::PrimeUsedDeviceIlu0() const noexcept
{ return IsReady() && implementation_->device_ilu0; }
} // namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics

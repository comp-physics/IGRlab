#include "numerics/hdg/runtime/cpu/detail/traced_scalar_igr_p1_global_step.hpp"

#include "numerics/hdg/core/host_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cpu::detail
{
namespace
{
namespace local = traced_scalar_igr_p1;

bool Finite(const double value) noexcept { return std::isfinite(value); }

bool AllFinite(const mfem::Vector& values) noexcept
{
  if (values.Size() == 0) { return true; }
  const double* const data = values.HostRead();
  return data != nullptr && std::all_of(data, data + values.Size(), Finite);
}

bool AllFinite(const std::vector<double>& values) noexcept
{
  return std::all_of(values.begin(), values.end(), Finite);
}

bool CollectiveAll(const MPI_Comm communicator, const bool local_ok) noexcept
{
  int source = local_ok ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&source, &result, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && result == 1;
}

TracedScalarIgrP1GlobalStepResult CollectiveStepResult(
    const MPI_Comm communicator,
    const TracedScalarIgrP1GlobalStepResult local_result) noexcept
{
  const int local_code = static_cast<int>(local_result);
  int global_code = static_cast<int>(TracedScalarIgrP1GlobalStepResult::failure);
  if (MPI_Allreduce(&local_code, &global_code, 1, MPI_INT, MPI_MAX,
                    communicator) != MPI_SUCCESS ||
      global_code < static_cast<int>(TracedScalarIgrP1GlobalStepResult::success) ||
      global_code > static_cast<int>(TracedScalarIgrP1GlobalStepResult::failure))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  return static_cast<TracedScalarIgrP1GlobalStepResult>(global_code);
}

bool CopyVector(const mfem::Vector& source, mfem::Vector& destination) noexcept
{
  if (source.Size() != destination.Size()) { return false; }
  const double* const from = source.HostRead();
  double* const to = destination.HostWrite();
  if ((from == nullptr || to == nullptr) && source.Size() != 0) { return false; }
  if (source.Size() != 0)
  {
    std::memcpy(to, from, static_cast<std::size_t>(source.Size()) * sizeof(double));
  }
  return true;
}

bool CheckedProduct(const std::size_t left, const std::size_t right,
                    std::size_t& product) noexcept
{
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
  {
    return false;
  }
  product = left * right;
  return true;
}

bool CheckedBatchCapacity(const int elements, const std::size_t entries,
                          std::size_t& capacity) noexcept
{
  return elements >= 0 && CheckedProduct(static_cast<std::size_t>(elements),
                                         entries, capacity);
}

bool CheckedGlobalTraceKey(const HYPRE_BigInt value,
                           std::uint64_t& key) noexcept
{
  if (value < 0) { return false; }
  if (sizeof(HYPRE_BigInt) > sizeof(std::uint64_t) &&
      value > static_cast<HYPRE_BigInt>(std::numeric_limits<std::uint64_t>::max()))
  {
    return false;
  }
  key = static_cast<std::uint64_t>(value);
  return true;
}

bool HasCapacity(const mfem::Vector& values, const int required) noexcept
{
  return required >= 0 && values.Capacity() >= required;
}

bool HasCapacity(const std::vector<double>& values,
                 const std::size_t required) noexcept
{
  return values.capacity() >= required && values.size() <= values.capacity();
}

int DecodeSignedVDof(const int signed_vdof, double& sign) noexcept
{
  if (signed_vdof == std::numeric_limits<int>::min()) { return -1; }
  sign = signed_vdof < 0 ? -1.0 : 1.0;
  return signed_vdof < 0 ? -1 - signed_vdof : signed_vdof;
}

template <class T>
StridedBlockView<const T> ConstView(const std::vector<T>& values,
                                    const int batches, const int rows,
                                    const int columns) noexcept
{
  return {values.data(), batches, rows, columns, rows * columns, columns, 1};
}

template <class T>
StridedBlockView<T> View(std::vector<T>& values, const int batches,
                         const int rows, const int columns) noexcept
{
  return {values.data(), batches, rows, columns, rows * columns, columns, 1};
}

bool SourceIsZero(const TracedScalarIgrP1GlobalStepInput& input) noexcept
{
  return input.static_euler_rhs_coefficients == nullptr ||
      input.static_euler_rhs_coefficients->Size() == 0;
}

bool CaptureElementVDofs(const mfem::ParFiniteElementSpace* const space,
                         const int elements, const int expected_per_element,
                         std::vector<int>& destination) noexcept
{
  if (space == nullptr || expected_per_element <= 0) { return false; }
  std::size_t entries = 0;
  if (!CheckedBatchCapacity(elements,
                            static_cast<std::size_t>(expected_per_element),
                            entries)) { return false; }
  try
  {
    destination.resize(entries);
    for (int element = 0; element < elements; ++element)
    {
      mfem::Array<int> vdofs;
      space->GetElementVDofs(element, vdofs);
      if (vdofs.Size() != expected_per_element || vdofs.HostRead() == nullptr)
        return false;
      for (int index = 0; index < expected_per_element; ++index)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(vdofs[index], sign);
        if (vdof < 0 || vdof >= space->GetVSize()) { return false; }
        destination[static_cast<std::size_t>(element) *
                        static_cast<std::size_t>(expected_per_element) + index] =
            vdofs[index];
      }
    }
    return true;
  }
  catch (const std::exception&) { return false; }
}

bool ValidParameters(const local::Parameters& parameters) noexcept
{
  return Finite(parameters.gamma) && parameters.gamma > 1.0 &&
      Finite(parameters.alpha) && parameters.alpha > 0.0 &&
      Finite(parameters.beta) && parameters.beta > 0.0 &&
      Finite(parameters.dt) && parameters.dt > 0.0 &&
      Finite(parameters.density_floor) && parameters.density_floor > 0.0 &&
      parameters.quadrature_order >= 2 && parameters.quadrature_order <= 32;
}

bool ValidInput(const TracedScalarIgrP1GlobalStepInput& input) noexcept
{
  if (input.communicator == MPI_COMM_NULL || input.layout == nullptr ||
      !input.layout->IsUsable() || input.current_true_trace == nullptr ||
      input.previous_true_trace == nullptr || input.current_interior == nullptr ||
      input.previous_interior == nullptr ||
      input.current_interior == input.previous_interior ||
      input.current_true_trace == input.previous_true_trace ||
      input.current_true_trace->Size() != input.layout->TraceTrueVSize() ||
      input.previous_true_trace->Size() != input.layout->TraceTrueVSize() ||
      input.current_interior->Size() != input.layout->InteriorLocalVSize() ||
      input.previous_interior->Size() != input.layout->InteriorLocalVSize() ||
      !AllFinite(*input.current_true_trace) ||
      !AllFinite(*input.previous_true_trace) || !AllFinite(*input.current_interior) ||
      !AllFinite(*input.previous_interior) ||
      !ValidParameters(input.parameters) ||
      input.affine_quad_count != input.layout->LocalElementCount() ||
      (input.affine_quad_count > 0 && input.affine_quads == nullptr))
  {
    return false;
  }
  if (SourceIsZero(input)) { return true; }
  const mfem::ParFiniteElementSpace* const source_space =
      input.static_euler_rhs_space;
  const mfem::Vector& source = *input.static_euler_rhs_coefficients;
  const mfem::ParFiniteElementSpace* const interior_space =
      input.layout->InteriorSpace();
  return source_space != nullptr && interior_space != nullptr &&
      source_space->GetParMesh() == interior_space->GetParMesh() &&
      source_space->GetVDim() == 4 &&
      source_space->GetOrdering() == mfem::Ordering::byNODES &&
      source_space->GetVSize() % 4 == 0 &&
      source.Size() == source_space->GetVSize() && AllFinite(source);
}

bool FillElementSource(const mfem::ParFiniteElementSpace* source_space,
                       const mfem::Vector* source_coefficients,
                       const int* const bound_signed_vdofs,
                       const int element, const mfem::IntegrationRule& rule,
                       double* samples, mfem::Vector& shape) noexcept
{
  if (source_coefficients == nullptr || source_coefficients->Size() == 0)
  {
    std::fill(samples, samples + static_cast<std::size_t>(rule.GetNPoints()) * 4,
              0.0);
    return true;
  }
  if (source_space == nullptr || samples == nullptr ||
      source_space->GetOrdering() != mfem::Ordering::byNODES ||
      source_space->GetVSize() % 4 != 0)
  {
    return false;
  }
  const mfem::FiniteElement* const finite_element = source_space->GetFE(element);
  if (finite_element == nullptr || finite_element->GetDof() != 4)
  {
    return false;
  }
  const double* const coefficients = source_coefficients->HostRead();
  if (coefficients == nullptr) { return false; }
  if (bound_signed_vdofs == nullptr)
  {
    mfem::Array<int> fallback_vdofs;
    source_space->GetElementVDofs(element, fallback_vdofs);
    if (fallback_vdofs.Size() != 16) { return false; }
    return FillElementSource(source_space, source_coefficients,
                             fallback_vdofs.HostRead(), element, rule, samples,
                             shape);
  }
  const int* const signed_vdofs = bound_signed_vdofs;
  if (shape.Size() != 4) { return false; }
  for (int point = 0; point < rule.GetNPoints(); ++point)
  {
    finite_element->CalcShape(rule.IntPoint(point), shape);
    for (int component = 0; component < 4; ++component)
    {
      double value = 0.0;
      for (int dof = 0; dof < 4; ++dof)
      {
        double sign = 1.0;
        // MFEM byNODES is component-blocked: [component][scalar dof].
        const int vdof = DecodeSignedVDof(signed_vdofs[component * 4 + dof], sign);
        if (vdof < 0 || vdof >= source_coefficients->Size()) { return false; }
        value += shape[dof] * sign * coefficients[vdof];
      }
      if (!Finite(value)) { return false; }
      samples[static_cast<std::size_t>(point) * 4 + component] = value;
    }
  }
  return true;
}

bool MaxAbsAndNorm(const MPI_Comm communicator, const mfem::Vector& values,
                   double& maximum, double& norm) noexcept
{
  const double* const data = values.HostRead();
  bool local_ok = data != nullptr || values.Size() == 0;
  double local_maximum = 0.0;
  for (int index = 0; index < values.Size() && local_ok; ++index)
  {
    local_ok = Finite(data[index]);
    if (!local_ok) { break; }
    local_maximum = std::max(local_maximum, std::abs(data[index]));
  }
  if (!CollectiveAll(communicator, local_ok) ||
      MPI_Allreduce(&local_maximum, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                    communicator) != MPI_SUCCESS || !Finite(maximum) ||
      maximum < 0.0)
  {
    return false;
  }
  if (maximum == 0.0) { norm = 0.0; return true; }
  double local_sum = 0.0;
  for (int index = 0; index < values.Size(); ++index)
  {
    const double scaled = data[index] / maximum;
    local_sum += scaled * scaled;
  }
  double global_sum = 0.0;
  if (!CollectiveAll(communicator, Finite(local_sum)) ||
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                    communicator) != MPI_SUCCESS || !Finite(global_sum) ||
      global_sum <= 0.0)
  {
    return false;
  }
  norm = maximum * std::sqrt(global_sum);
  return Finite(norm);
}

/** Scale-safe norm of all broken interior values and owned assembled trace values. */
bool MaxAbsAndNorm(const MPI_Comm communicator, const std::vector<double>& interior,
                   const mfem::Vector& true_trace, double& maximum,
                   double& norm) noexcept
{
  const double* const trace = true_trace.HostRead();
  bool local_ok = (trace != nullptr || true_trace.Size() == 0) &&
      AllFinite(interior);
  double local_maximum = 0.0;
  for (const double value : interior)
  {
    local_maximum = std::max(local_maximum, std::abs(value));
  }
  for (int index = 0; index < true_trace.Size() && local_ok; ++index)
  {
    local_ok = Finite(trace[index]);
    if (!local_ok) { break; }
    local_maximum = std::max(local_maximum, std::abs(trace[index]));
  }
  if (!CollectiveAll(communicator, local_ok) ||
      MPI_Allreduce(&local_maximum, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                    communicator) != MPI_SUCCESS || !Finite(maximum) ||
      maximum < 0.0)
  {
    return false;
  }
  if (maximum == 0.0) { norm = 0.0; return true; }
  double local_sum = 0.0;
  for (const double value : interior)
  {
    const double scaled = value / maximum;
    local_sum += scaled * scaled;
  }
  for (int index = 0; index < true_trace.Size(); ++index)
  {
    const double scaled = trace[index] / maximum;
    local_sum += scaled * scaled;
  }
  double global_sum = 0.0;
  if (!CollectiveAll(communicator, Finite(local_sum)) ||
      MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                    communicator) != MPI_SUCCESS || !Finite(global_sum) ||
      global_sum <= 0.0)
  {
    return false;
  }
  norm = maximum * std::sqrt(global_sum);
  return Finite(norm);
}

} // namespace

/*
 * A deliberately narrow retained true-trace system.  The public scratch
 * owner below supplies the frozen layout lifetime; this type does not expose
 * a session or attempt protocol.  The graph is assembled exactly once from
 * signed element vdofs, then all numerical assembly writes through the
 * recorded CSR slots.  Rows owned by another rank are exchanged in fixed
 * rank-major order, so this works for an arbitrary number of neighbours.
 */
struct RetainedTraceSystem final
{
  struct LocalDof final { HYPRE_BigInt global = -1; int owner = -1; };
  struct Slot final
  {
    std::size_t schur = 0;
    int entry = -1;
    double sign = 1.0;
    bool offd = false;
  };
  struct Outgoing final
  {
    std::size_t schur = 0;
    double sign = 1.0;
    int destination = -1;
  };
  struct Incoming final { int entry = -1; bool offd = false; };

  MPI_Comm communicator = MPI_COMM_NULL;
  const BorrowedP1DGInterfaceLayout* layout = nullptr;
  const mfem::ParFiniteElementSpace* trace_space = nullptr;
  int rank = -1;
  int ranks = 0;
  int elements = -1;
  int local_size = -1;
  int true_size = -1;
  long trace_sequence = -1;
  long mesh_sequence = -1;
  bool refreshed = false;
  std::vector<int> element_vdofs;
  std::vector<LocalDof> local_dofs;
  std::unique_ptr<mfem::HypreParMatrix> matrix;
  mfem::SparseMatrix diagonal;
  mfem::SparseMatrix offdiagonal;
  HYPRE_BigInt* offd_columns = nullptr;
  std::vector<Slot> owned;
  std::vector<Outgoing> outgoing;
  std::vector<Incoming> incoming;
  std::vector<int> send_counts, receive_counts, send_offsets, receive_offsets;
  std::vector<double> send_values, receive_values;
  mfem::Vector local_rhs;
  mfem::Vector true_rhs;
  std::unique_ptr<mfem::HypreParVector> rhs;
  std::unique_ptr<mfem::HypreParVector> solution;
  std::unique_ptr<mfem::HypreParVector> residual;
  std::unique_ptr<mfem::HypreILU> ilu;
  std::unique_ptr<mfem::HypreGMRES> gmres;

  void MarkUnrefreshed() noexcept { refreshed = false; }

  bool IsFrozen() const noexcept
  {
    if (communicator == MPI_COMM_NULL || !layout || !trace_space || rank < 0 ||
        ranks <= 0 || elements < 0 || local_size < 0 || true_size < 0 ||
        !matrix || !rhs || !solution || !residual || !ilu || !gmres ||
        matrix->Height() != true_size || matrix->Width() != true_size ||
        diagonal.Height() != true_size || offdiagonal.Height() != true_size ||
        element_vdofs.size() != static_cast<std::size_t>(elements) * 40 ||
        local_dofs.size() != static_cast<std::size_t>(local_size) ||
        send_counts.size() != static_cast<std::size_t>(ranks) ||
        receive_counts.size() != static_cast<std::size_t>(ranks) ||
        send_offsets.size() != static_cast<std::size_t>(ranks) ||
        receive_offsets.size() != static_cast<std::size_t>(ranks) ||
        send_values.size() != outgoing.size() || receive_values.size() != incoming.size() ||
        local_rhs.Size() != local_size || true_rhs.Size() != true_size ||
        rhs->Size() != true_size || solution->Size() != true_size ||
        residual->Size() != true_size)
    {
      return false;
    }
    int send_cursor = 0;
    int receive_cursor = 0;
    for (int peer = 0; peer < ranks; ++peer)
    {
      const std::size_t index = static_cast<std::size_t>(peer);
      if (send_counts[index] < 0 || receive_counts[index] < 0 ||
          send_offsets[index] != send_cursor || receive_offsets[index] != receive_cursor ||
          send_cursor > std::numeric_limits<int>::max() - send_counts[index] ||
          receive_cursor > std::numeric_limits<int>::max() - receive_counts[index])
      {
        return false;
      }
      send_cursor += send_counts[index]; receive_cursor += receive_counts[index];
    }
    if (static_cast<std::size_t>(send_cursor) != outgoing.size() ||
        static_cast<std::size_t>(receive_cursor) != incoming.size()) return false;
    const std::size_t schur_entries = static_cast<std::size_t>(elements) * 1600;
    const int diagonal_nnz = diagonal.NumNonZeroElems();
    const int offdiagonal_nnz = offdiagonal.NumNonZeroElems();
    if (diagonal_nnz < 0 || offdiagonal_nnz < 0 ||
        (offdiagonal_nnz != 0 && offd_columns == nullptr) ||
        (true_size == 0 && (!owned.empty() || !incoming.empty()))) return false;
    for (const Slot& term : owned)
      if (term.entry < 0 || term.schur >= schur_entries ||
          term.entry >= (term.offd ? offdiagonal_nnz : diagonal_nnz)) return false;
    for (const Outgoing& term : outgoing)
      if (term.destination < 0 || term.destination >= ranks ||
          term.destination == rank || term.schur >= schur_entries) return false;
    for (const Incoming& term : incoming)
      if (term.entry < 0 || term.entry >=
          (term.offd ? offdiagonal_nnz : diagonal_nnz)) return false;
    return true;
  }

  bool FindSlot(const HYPRE_BigInt row, const HYPRE_BigInt column,
                int& entry, bool& is_offd) const noexcept
  {
    const HYPRE_BigInt* starts = matrix ? matrix->GetRowStarts() : nullptr;
    const int local_rows = diagonal.Height();
    const int* di = diagonal.GetI();
    const int* dj = diagonal.GetJ();
    const int* oi = offdiagonal.GetI();
    const int* oj = offdiagonal.GetJ();
    if (local_rows == 0) { return false; }
    if (!starts || !di) { return false; }
    const HYPRE_BigInt begin = HYPRE_AssumedPartitionCheck() ? starts[0] : starts[rank];
    const HYPRE_BigInt end = begin + local_rows;
    if (row < begin || row >= end) { return false; }
    const int local_row = static_cast<int>(row - begin);
    if (column >= begin && column < end)
    {
      if (!dj) { return false; }
      const int local_column = static_cast<int>(column - begin);
      for (int index = di[local_row]; index < di[local_row + 1]; ++index)
      {
        if (dj[index] == local_column) { entry = index; is_offd = false; return true; }
      }
      return false;
    }
    if (!oi || !oj || !offd_columns) { return false; }
    for (int index = oi[local_row]; index < oi[local_row + 1]; ++index)
    {
      if (offd_columns[oj[index]] == column)
      { entry = index; is_offd = true; return true; }
    }
    return false;
  }

  bool BindSolver() noexcept
  {
    try
    {
      if (!matrix || !ilu || !gmres) { return false; }
      gmres->SetOperator(*matrix);
      // SetOperator may recreate the bound HypreILU with defaults.  Match the
      // accepted retained-core rebind sequence before it is reattached.
      ilu->SetType(0); ilu->SetLevelOfFill(0); ilu->SetMaxIter(1);
      ilu->SetTol(0.0); ilu->SetLocalReordering(0); ilu->SetPrintLevel(0);
      gmres->SetPreconditioner(*ilu);
      return true;
    }
    catch (const std::exception&) { return false; }
  }

  bool InitialBindSolver() noexcept
  {
    try
    {
      if (!matrix || !ilu || !gmres) { return false; }
      ilu->SetType(0); ilu->SetLevelOfFill(0); ilu->SetMaxIter(1);
      ilu->SetTol(0.0); ilu->SetLocalReordering(0); ilu->SetPrintLevel(0);
      gmres->SetPreconditioner(*ilu);
      gmres->SetOperator(*matrix);
      return true;
    }
    catch (const std::exception&) { return false; }
  }

  bool Initialize(MPI_Comm comm, const BorrowedP1DGInterfaceLayout& bound) noexcept;
  bool Prime() noexcept;
  bool Refresh(const std::vector<double>& schur_values,
               const std::vector<double>& reduced_values) noexcept;
  bool Solve(const double scale, mfem::Vector& true_trace,
             double& residual_norm) noexcept;
};

bool RetainedTraceSystem::Initialize(
    const MPI_Comm comm, const BorrowedP1DGInterfaceLayout& bound) noexcept
{
  communicator = comm; layout = &bound; trace_space = bound.TraceSpace();
  int comparison = MPI_UNEQUAL;
  bool local_ok = communicator != MPI_COMM_NULL && bound.IsUsable() &&
      trace_space != nullptr && MPI_Comm_rank(communicator, &rank) == MPI_SUCCESS &&
      MPI_Comm_size(communicator, &ranks) == MPI_SUCCESS && rank >= 0 &&
      ranks > 0 && MPI_Comm_compare(communicator, trace_space->GetComm(),
                                    &comparison) == MPI_SUCCESS &&
      (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  try
  {
    elements = bound.LocalElementCount(); local_size = bound.TraceLocalVSize();
    true_size = bound.TraceTrueVSize();
    trace_sequence = trace_space->GetSequence();
    const mfem::ParMesh* const mesh = trace_space->GetParMesh();
    mesh_sequence = mesh ? mesh->GetSequence() : -1;
    local_ok = elements >= 0 && local_size >= 0 && true_size >= 0 &&
        trace_sequence >= 0 && mesh_sequence >= 0 &&
        trace_space->GetVSize() == local_size && trace_space->GetTrueVSize() == true_size;
    element_vdofs.resize(static_cast<std::size_t>(elements) * 40);
    local_dofs.resize(static_cast<std::size_t>(local_size));
    std::vector<int> owner(static_cast<std::size_t>(local_size), rank);
    const mfem::GroupCommunicator& group = trace_space->GroupComm();
    const mfem::Table& group_dofs = group.GroupLDofTable();
    const mfem::GroupTopology& topology = group.GetGroupTopology();
    local_ok = local_ok && group_dofs.Size() == topology.NGroups();
    for (int group_index = 1; group_index < group_dofs.Size() && local_ok;
         ++group_index)
    {
      const int master = topology.GetGroupMasterRank(group_index);
      const int* const dofs = group_dofs.GetRow(group_index);
      local_ok = master >= 0 && master < ranks &&
          (group_dofs.RowSize(group_index) == 0 || dofs != nullptr);
      for (int index = 0; index < group_dofs.RowSize(group_index) && local_ok;
           ++index)
      {
        local_ok = dofs[index] >= 0 && dofs[index] < local_size;
        if (local_ok) { owner[static_cast<std::size_t>(dofs[index])] = master; }
      }
    }
    for (int dof = 0; dof < local_size && local_ok; ++dof)
    {
      const HYPRE_BigInt global = trace_space->GetGlobalTDofNumber(dof);
      const int local_true = trace_space->GetLocalTDofNumber(dof);
      local_ok = global >= 0 && owner[static_cast<std::size_t>(dof)] >= 0 &&
          owner[static_cast<std::size_t>(dof)] < ranks &&
          ((local_true >= 0) == (owner[static_cast<std::size_t>(dof)] == rank));
      local_dofs[static_cast<std::size_t>(dof)] = {global,
          owner[static_cast<std::size_t>(dof)]};
    }
    for (int element = 0; element < elements && local_ok; ++element)
    {
      mfem::Array<int> vdofs;
      trace_space->GetElementVDofs(element, vdofs);
      local_ok = vdofs.Size() == 40;
      for (int index = 0; index < 40 && local_ok; ++index)
      {
        double sign = 1.0;
        const int dof = DecodeSignedVDof(vdofs[index], sign);
        local_ok = dof >= 0 && dof < local_size;
        element_vdofs[static_cast<std::size_t>(element) * 40 + index] = vdofs[index];
      }
    }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }

  std::unique_ptr<mfem::SparseMatrix> graph;
  std::unique_ptr<mfem::ParBilinearForm> parallel_form;
  try
  {
    graph = std::make_unique<mfem::SparseMatrix>(local_size);
    mfem::DenseMatrix pattern(40);
    for (int element = 0; element < elements; ++element)
    {
      mfem::Array<int> vdofs(40);
      for (int row = 0; row < 40; ++row)
      {
        vdofs[row] = element_vdofs[static_cast<std::size_t>(element) * 40 + row];
        double row_sign = 1.0;
        (void)DecodeSignedVDof(vdofs[row], row_sign);
        for (int column = 0; column < 40; ++column)
        {
          double column_sign = 1.0;
          (void)DecodeSignedVDof(element_vdofs[static_cast<std::size_t>(element) * 40 + column], column_sign);
          pattern(row, column) = row_sign * column_sign;
        }
      }
      graph->AddSubMatrix(vdofs, vdofs, pattern, 1);
    }
    graph->Finalize(0, false);
    parallel_form = std::make_unique<mfem::ParBilinearForm>(
        const_cast<mfem::ParFiniteElementSpace*>(trace_space));
    local_ok = parallel_form != nullptr;
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }

  try
  {
    // Every rank reaches this opaque collective only after its local signed
    // symbolic graph has been admitted.
    matrix.reset(parallel_form->ParallelAssemble(graph.get()));
    local_ok = matrix != nullptr && matrix->Height() == true_size &&
        matrix->Width() == true_size &&
        matrix->GetGlobalNumRows() == trace_space->GlobalTrueVSize() &&
        matrix->GetGlobalNumCols() == trace_space->GlobalTrueVSize() &&
        matrix->GetGlobalNumRows() > 0 &&
        matrix->GetGlobalNumRows() <= std::numeric_limits<int>::max();
    if (local_ok)
    {
      matrix->HostReadWrite(); matrix->GetDiag(diagonal);
      matrix->GetOffd(offdiagonal, offd_columns);
      local_ok = diagonal.Height() == true_size;
    }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }

  std::vector<std::array<std::uint64_t, 2>> outgoing_keys;
  try
  {
    for (int element = 0; element < elements && local_ok; ++element)
      for (int row = 0; row < 40 && local_ok; ++row)
      {
        double row_sign = 1.0;
        const int row_dof = DecodeSignedVDof(element_vdofs[static_cast<std::size_t>(element) * 40 + row], row_sign);
        local_ok = row_dof >= 0 && row_dof < local_size;
        const LocalDof& row_info = local_dofs[static_cast<std::size_t>(row_dof)];
        for (int column = 0; column < 40 && local_ok; ++column)
        {
          double column_sign = 1.0;
          const int column_dof = DecodeSignedVDof(element_vdofs[static_cast<std::size_t>(element) * 40 + column], column_sign);
          local_ok = column_dof >= 0 && column_dof < local_size;
          const LocalDof& column_info = local_dofs[static_cast<std::size_t>(column_dof)];
          const std::size_t index = static_cast<std::size_t>(element) * 1600 +
              static_cast<std::size_t>(row) * 40 + column;
          if (row_info.owner == rank)
          {
            int entry = -1; bool is_offd = false;
            local_ok = FindSlot(row_info.global, column_info.global, entry, is_offd);
            if (local_ok) { owned.push_back({index, entry, row_sign * column_sign, is_offd}); }
          }
          else
          {
            std::uint64_t row_key = 0;
            std::uint64_t column_key = 0;
            local_ok = CheckedGlobalTraceKey(row_info.global, row_key) &&
                CheckedGlobalTraceKey(column_info.global, column_key);
            if (!local_ok) { break; }
            outgoing.push_back({index, row_sign * column_sign, row_info.owner});
            outgoing_keys.push_back({row_key, column_key});
          }
        }
      }
    std::vector<std::size_t> order(outgoing.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b)
      { return outgoing[a].destination < outgoing[b].destination; });
    std::vector<Outgoing> sorted(outgoing.size());
    std::vector<std::array<std::uint64_t, 2>> sorted_keys(outgoing.size());
    for (std::size_t index = 0; index < order.size(); ++index)
    { sorted[index] = outgoing[order[index]]; sorted_keys[index] = outgoing_keys[order[index]]; }
    outgoing.swap(sorted); outgoing_keys.swap(sorted_keys);
    send_counts.assign(static_cast<std::size_t>(ranks), 0);
    send_offsets.assign(static_cast<std::size_t>(ranks), 0);
    for (const Outgoing& term : outgoing)
    {
      local_ok = local_ok && term.destination >= 0 && term.destination < ranks && term.destination != rank;
      if (local_ok)
      {
        int& count = send_counts[static_cast<std::size_t>(term.destination)];
        local_ok = count >= 0 && count < std::numeric_limits<int>::max() / 2;
        if (local_ok) { ++count; }
      }
    }
    for (int peer = 1; peer < ranks && local_ok; ++peer)
    {
      const int previous_offset = send_offsets[static_cast<std::size_t>(peer - 1)];
      const int previous_count = send_counts[static_cast<std::size_t>(peer - 1)];
      local_ok = previous_offset >= 0 && previous_count >= 0 &&
          previous_offset <= std::numeric_limits<int>::max() - previous_count;
      if (local_ok) send_offsets[static_cast<std::size_t>(peer)] =
          previous_offset + previous_count;
    }
    if (local_ok && ranks > 0)
    {
      const int final_offset = send_offsets.back();
      const int final_count = send_counts.back();
      local_ok = final_offset >= 0 && final_count >= 0 &&
          final_offset <= std::numeric_limits<int>::max() - final_count &&
          final_offset + final_count <= std::numeric_limits<int>::max() / 2 &&
          static_cast<std::size_t>(final_offset + final_count) == outgoing.size();
    }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  try
  {
    receive_counts.assign(static_cast<std::size_t>(ranks), 0);
    receive_offsets.assign(static_cast<std::size_t>(ranks), 0);
    local_ok = receive_counts.size() == static_cast<std::size_t>(ranks) &&
        receive_offsets.size() == static_cast<std::size_t>(ranks);
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  const bool count_exchange = MPI_Alltoall(
      send_counts.data(), 1, MPI_INT, receive_counts.data(), 1, MPI_INT,
      communicator) == MPI_SUCCESS;
  if (!CollectiveAll(communicator, count_exchange)) { return false; }
  std::size_t incoming_count = 0;
  for (int peer = 0; peer < ranks; ++peer)
  {
    const int count = receive_counts[static_cast<std::size_t>(peer)];
    local_ok = local_ok && count >= 0 &&
        count <= std::numeric_limits<int>::max() / 2;
    if (peer > 0 && local_ok)
    {
      const int previous_offset = receive_offsets[static_cast<std::size_t>(peer - 1)];
      const int previous_count = receive_counts[static_cast<std::size_t>(peer - 1)];
      local_ok = previous_offset >= 0 && previous_count >= 0 &&
          previous_offset <= std::numeric_limits<int>::max() - previous_count;
      if (local_ok) receive_offsets[static_cast<std::size_t>(peer)] =
          previous_offset + previous_count;
    }
  }
  if (ranks > 0 && local_ok)
  {
    const int final_offset = receive_offsets.back();
    const int final_count = receive_counts.back();
    local_ok = final_offset >= 0 && final_count >= 0 &&
        final_offset <= std::numeric_limits<int>::max() - final_count;
    local_ok = local_ok && final_offset + final_count <=
        std::numeric_limits<int>::max() / 2;
    if (local_ok) incoming_count = static_cast<std::size_t>(final_offset + final_count);
  }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  std::vector<std::array<std::uint64_t, 2>> incoming_keys;
  std::vector<int> send_key_counts, receive_key_counts, send_key_offsets,
      receive_key_offsets;
  try
  {
    incoming_keys.resize(incoming_count);
    send_key_counts.assign(static_cast<std::size_t>(ranks), 0);
    receive_key_counts.assign(static_cast<std::size_t>(ranks), 0);
    send_key_offsets.assign(static_cast<std::size_t>(ranks), 0);
    receive_key_offsets.assign(static_cast<std::size_t>(ranks), 0);
    local_ok = incoming_keys.size() == incoming_count &&
        send_key_counts.size() == static_cast<std::size_t>(ranks) &&
        receive_key_counts.size() == static_cast<std::size_t>(ranks) &&
        send_key_offsets.size() == static_cast<std::size_t>(ranks) &&
        receive_key_offsets.size() == static_cast<std::size_t>(ranks);
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  for (int peer = 0; peer < ranks; ++peer)
  {
    local_ok = local_ok && send_counts[peer] >= 0 && receive_counts[peer] >= 0 &&
        send_offsets[peer] >= 0 && receive_offsets[peer] >= 0 &&
        send_counts[peer] <= std::numeric_limits<int>::max() / 2 &&
        receive_counts[peer] <= std::numeric_limits<int>::max() / 2 &&
        send_offsets[peer] <= std::numeric_limits<int>::max() / 2 &&
        receive_offsets[peer] <= std::numeric_limits<int>::max() / 2;
    if (local_ok)
    {
      send_key_counts[peer] = 2 * send_counts[peer]; receive_key_counts[peer] = 2 * receive_counts[peer];
      send_key_offsets[peer] = 2 * send_offsets[peer]; receive_key_offsets[peer] = 2 * receive_offsets[peer];
    }
  }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  const bool key_exchange = MPI_Alltoallv(
      outgoing_keys.data(), send_key_counts.data(), send_key_offsets.data(),
      MPI_UINT64_T, incoming_keys.data(), receive_key_counts.data(),
      receive_key_offsets.data(), MPI_UINT64_T, communicator) == MPI_SUCCESS;
  if (!CollectiveAll(communicator, key_exchange)) { return false; }
  try
  {
    incoming.resize(incoming_count); send_values.resize(outgoing.size()); receive_values.resize(incoming_count);
    for (std::size_t index = 0; index < incoming_count && local_ok; ++index)
    {
      int entry = -1; bool is_offd = false;
      local_ok = incoming_keys[index][0] <= static_cast<std::uint64_t>(
              std::numeric_limits<HYPRE_BigInt>::max()) &&
          incoming_keys[index][1] <= static_cast<std::uint64_t>(
              std::numeric_limits<HYPRE_BigInt>::max()) &&
          FindSlot(static_cast<HYPRE_BigInt>(incoming_keys[index][0]),
                   static_cast<HYPRE_BigInt>(incoming_keys[index][1]), entry,
                   is_offd);
      incoming[index] = {entry, is_offd};
    }
    local_rhs.SetSize(local_size); true_rhs.SetSize(true_size);
    rhs = std::make_unique<mfem::HypreParVector>(const_cast<mfem::ParFiniteElementSpace*>(trace_space));
    solution = std::make_unique<mfem::HypreParVector>(const_cast<mfem::ParFiniteElementSpace*>(trace_space));
    residual = std::make_unique<mfem::HypreParVector>(const_cast<mfem::ParFiniteElementSpace*>(trace_space));
    ilu = std::make_unique<mfem::HypreILU>(); gmres = std::make_unique<mfem::HypreGMRES>(communicator);
    gmres->SetTol(1.0e-12); gmres->SetAbsTol(0.0);
    const int global = static_cast<int>(matrix->GetGlobalNumRows());
    gmres->SetMaxIter(std::max(100, global)); gmres->SetKDim(std::min(100, std::max(20, global)));
    gmres->SetLogging(0); gmres->SetPrintLevel(0);
    local_ok = InitialBindSolver();
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  refreshed = false;
  return true;
}

bool RetainedTraceSystem::Prime() noexcept
{
  refreshed = false;
  bool local_ok = matrix && rhs && solution && residual && gmres && true_size >= 0;
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  try
  {
    matrix->HostReadWrite(); matrix->GetDiag(diagonal); matrix->GetOffd(offdiagonal, offd_columns);
    double* const diagonal_data = diagonal.GetData();
    double* const offd_data = offdiagonal.GetData();
    const int diagonal_nnz = diagonal.NumNonZeroElems();
    const int offd_nnz = offdiagonal.NumNonZeroElems();
    local_ok = (diagonal_nnz == 0 || diagonal_data != nullptr) &&
        (offd_nnz == 0 || offd_data != nullptr) && diagonal.Height() == true_size;
    if (local_ok)
    {
      if (diagonal_nnz) std::fill(diagonal_data, diagonal_data + diagonal_nnz, 0.0);
      if (offd_nnz) std::fill(offd_data, offd_data + offd_nnz, 0.0);
      const int* const row_offsets = diagonal.GetI();
      const int* const columns = diagonal.GetJ();
      local_ok = (true_size == 0) || (row_offsets != nullptr && columns != nullptr);
      for (int row = 0; row < true_size && local_ok; ++row)
      {
        bool found = false;
        for (int entry = row_offsets[row]; entry < row_offsets[row + 1]; ++entry)
          if (columns[entry] == row) { diagonal_data[entry] = 1.0; found = true; break; }
        local_ok = found;
      }
      if (local_ok) { matrix->HypreRead(); local_ok = BindSolver(); }
      if (local_ok) { *rhs = 1.0; *solution = 0.0; }
    }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  try { gmres->Mult(*rhs, *solution); local_ok = AllFinite(*solution); }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  try { *rhs = 0.0; *solution = 0.0; *residual = 0.0; }
  catch (const std::exception&) { local_ok = false; }
  refreshed = false;
  return CollectiveAll(communicator, local_ok);
}

bool RetainedTraceSystem::Refresh(const std::vector<double>& schur_values,
                                  const std::vector<double>& reduced_values) noexcept
{
  const std::size_t expected_schur = static_cast<std::size_t>(elements) * 1600;
  const std::size_t expected_reduced = static_cast<std::size_t>(elements) * 40;
  bool local_ok = matrix && schur_values.size() == expected_schur &&
      reduced_values.size() == expected_reduced && AllFinite(schur_values) &&
      AllFinite(reduced_values);
  if (!CollectiveAll(communicator, local_ok)) { refreshed = false; return false; }
  refreshed = false;
  try
  {
    matrix->HostReadWrite(); matrix->GetDiag(diagonal); matrix->GetOffd(offdiagonal, offd_columns);
    double* diagonal_data = diagonal.GetData(); double* offd_data = offdiagonal.GetData();
    const int diagonal_nnz = diagonal.NumNonZeroElems(); const int offd_nnz = offdiagonal.NumNonZeroElems();
    local_ok = (diagonal_nnz == 0 || diagonal_data != nullptr) && (offd_nnz == 0 || offd_data != nullptr);
    if (local_ok)
    {
      if (diagonal_nnz) std::fill(diagonal_data, diagonal_data + diagonal_nnz, 0.0);
      if (offd_nnz) std::fill(offd_data, offd_data + offd_nnz, 0.0);
      local_rhs = 0.0;
      for (int element = 0; element < elements && local_ok; ++element)
        for (int row = 0; row < 40; ++row)
        {
          double sign = 1.0; const int dof = DecodeSignedVDof(element_vdofs[static_cast<std::size_t>(element) * 40 + row], sign);
          local_ok = local_ok && dof >= 0 && dof < local_rhs.Size();
          if (local_ok) local_rhs[dof] += sign * reduced_values[static_cast<std::size_t>(element) * 40 + row];
        }
      for (const Slot& term : owned)
      {
        double* data = term.offd ? offd_data : diagonal_data;
        local_ok = local_ok && data != nullptr && term.entry >= 0 && term.schur < schur_values.size();
        if (local_ok) data[term.entry] += term.sign * schur_values[term.schur];
      }
      for (std::size_t index = 0; index < outgoing.size() && local_ok; ++index)
      {
        local_ok = outgoing[index].schur < schur_values.size();
        if (local_ok) send_values[index] = outgoing[index].sign * schur_values[outgoing[index].schur];
      }
      local_ok = local_ok && AllFinite(local_rhs) && AllFinite(send_values);
    }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  const bool value_exchange = MPI_Alltoallv(
      send_values.data(), send_counts.data(), send_offsets.data(), MPI_DOUBLE,
      receive_values.data(), receive_counts.data(), receive_offsets.data(),
      MPI_DOUBLE, communicator) == MPI_SUCCESS;
  if (!CollectiveAll(communicator, value_exchange)) { return false; }
  try
  {
    double* diagonal_data = diagonal.GetData(); double* offd_data = offdiagonal.GetData();
    local_ok = AllFinite(receive_values);
    for (std::size_t index = 0; index < incoming.size() && local_ok; ++index)
    {
      double* data = incoming[index].offd ? offd_data : diagonal_data;
      local_ok = data != nullptr && incoming[index].entry >= 0;
      if (local_ok) data[incoming[index].entry] += receive_values[index];
    }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  try
  {
    trace_space->Dof_TrueDof_Matrix()->MultTranspose(local_rhs, true_rhs);
    local_ok = AllFinite(true_rhs);
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  try
  {
    matrix->HypreRead();
    local_ok = BindSolver();
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { return false; }
  refreshed = true;
  return true;
}

bool RetainedTraceSystem::Solve(const double scale, mfem::Vector& true_trace,
                                double& residual_norm) noexcept
{
  bool local_ok = refreshed && Finite(scale) && scale > 0.0 && rhs && solution && residual;
  if (!CollectiveAll(communicator, local_ok)) { refreshed = false; return false; }
  try
  {
    const double* values = true_rhs.HostRead(); double* scaled = rhs->HostWrite();
    local_ok = (true_size == 0 || (values != nullptr && scaled != nullptr));
    for (int index = 0; index < true_size && local_ok; ++index) { scaled[index] = values[index] / scale; local_ok = Finite(scaled[index]); }
    *solution = 0.0;
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { refreshed = false; return false; }
  try { gmres->Mult(*rhs, *solution); }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { refreshed = false; return false; }
  try { matrix->Mult(*solution, *residual); *residual -= *rhs; }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { refreshed = false; return false; }
  double maximum = 0.0;
  if (!MaxAbsAndNorm(communicator, *residual, maximum, residual_norm))
  { refreshed = false; return false; }
  try
  {
    const double* solved = solution->HostRead(); double* destination = true_trace.HostWrite();
    local_ok = AllFinite(*solution) && residual_norm <= 1.0e-10 &&
        (true_size == 0 || (solved != nullptr && destination != nullptr));
    for (int index = 0; index < true_size && local_ok; ++index) { destination[index] = scale * solved[index]; local_ok = Finite(destination[index]); }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) { refreshed = false; return false; }
  residual_norm *= scale;
  refreshed = false;
  return Finite(residual_norm);
}

/** Retained non-Hypre storage for one solve/recovery call. */
struct SolveAndRecoverTracedScalarIgrP1State final
{
  mfem::Vector true_trace;
  mfem::Vector next_trace;
  mfem::Vector local_trace;
  mfem::Vector next_interior;
  std::vector<double> recovered;
  std::vector<double> local_trace_values;
  mfem::Vector gathered_trace;
  mfem::Vector element_interior;
  TracedScalarIgrP1TraceSolveOutput result;

  bool HasExpectedStorage(const int trace_true_size, const int trace_local_size,
                          const int interior_local_size,
                          const int elements) const noexcept
  {
    std::size_t recovered_capacity = 0;
    std::size_t local_trace_capacity = 0;
    return trace_true_size >= 0 && trace_local_size >= 0 &&
        interior_local_size >= 0 && elements >= 0 &&
        CheckedBatchCapacity(elements, 28, recovered_capacity) &&
        CheckedBatchCapacity(elements, 40, local_trace_capacity) &&
        true_trace.Size() == trace_true_size &&
        next_trace.Size() == trace_true_size &&
        local_trace.Size() == trace_local_size &&
        next_interior.Size() == interior_local_size &&
        gathered_trace.Size() == 40 && element_interior.Size() == 28 &&
        HasCapacity(true_trace, trace_true_size) &&
        HasCapacity(next_trace, trace_true_size) &&
        HasCapacity(local_trace, trace_local_size) &&
        HasCapacity(next_interior, interior_local_size) &&
        HasCapacity(gathered_trace, 40) && HasCapacity(element_interior, 28) &&
        recovered.size() == recovered_capacity &&
        local_trace_values.size() == local_trace_capacity &&
        HasCapacity(recovered, recovered_capacity) &&
        HasCapacity(local_trace_values, local_trace_capacity) &&
        result.trace_increment.Size() == trace_true_size &&
        result.interior_increment.Size() == interior_local_size &&
        HasCapacity(result.trace_increment, trace_true_size) &&
        HasCapacity(result.interior_increment, interior_local_size);
  }
};

static bool SolveAndRecoverTracedScalarIgrP1TraceWithState(
    const TracedScalarIgrP1TraceSolveInput& input,
    SolveAndRecoverTracedScalarIgrP1State& state,
    RetainedTraceSystem* const retained = nullptr) noexcept
{
  if (input.communicator == MPI_COMM_NULL) { return false; }
  bool local_input_ok = input.layout != nullptr && input.trace_space != nullptr &&
      input.true_matrix != nullptr && input.true_rhs_values != nullptr &&
      input.condensation != nullptr;
  if (local_input_ok)
  {
    local_input_ok = input.layout->IsUsable() &&
        input.trace_space == input.layout->TraceSpace() &&
        input.true_rhs_values->Size() == input.layout->TraceTrueVSize() &&
        input.true_matrix->Height() == input.layout->TraceTrueVSize() &&
        input.true_matrix->Width() == input.layout->TraceTrueVSize() &&
        input.true_matrix->GetGlobalNumRows() == input.trace_space->GlobalTrueVSize() &&
        input.true_matrix->GetGlobalNumCols() == input.trace_space->GlobalTrueVSize();
  }
  if (!CollectiveAll(input.communicator, local_input_ok)) { return false; }
  const int elements = input.layout->LocalElementCount();
  if (!CollectiveAll(input.communicator, elements == 0 ||
      (input.condensation->ready() && input.condensation->batch_count() ==
       static_cast<std::size_t>(elements)))) { return false; }

  double rhs_maximum = 0.0;
  double rhs_norm = 0.0;
  if (!MaxAbsAndNorm(input.communicator, *input.true_rhs_values, rhs_maximum,
                     rhs_norm) || !CollectiveAll(input.communicator, Finite(rhs_norm)))
  {
    return false;
  }
  const bool exact_zero_rhs = rhs_maximum == 0.0;
  if (!exact_zero_rhs && rhs_maximum < std::numeric_limits<double>::min())
  {
    return false;
  }
  mfem::Vector& true_trace = state.true_trace;
  mfem::Vector& next_trace = state.next_trace;
  mfem::Vector& local_trace = state.local_trace;
  mfem::Vector& next_interior = state.next_interior;
  std::vector<double>& recovered = state.recovered;
  std::vector<double>& local_trace_values = state.local_trace_values;
  mfem::Vector& gathered_trace = state.gathered_trace;
  mfem::Vector& element_interior = state.element_interior;
  TracedScalarIgrP1TraceSolveOutput& result = state.result;
  bool local_ok = true;
  try
  {
    true_trace.SetSize(input.layout->TraceTrueVSize()); true_trace = 0.0;
    next_trace.SetSize(input.layout->TraceTrueVSize());
    local_trace.SetSize(input.layout->TraceLocalVSize());
    next_interior.SetSize(input.layout->InteriorLocalVSize());
    gathered_trace.SetSize(40); element_interior.SetSize(28);
    recovered.resize(static_cast<std::size_t>(elements) * 28);
    local_trace_values.resize(static_cast<std::size_t>(elements) * 40);
    if (result.trace_increment.Size() != input.layout->TraceTrueVSize())
    {
      result.trace_increment.SetSize(input.layout->TraceTrueVSize());
    }
    if (result.interior_increment.Size() != input.layout->InteriorLocalVSize())
    {
      result.interior_increment.SetSize(input.layout->InteriorLocalVSize());
    }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  double linear_norm = 0.0;

  if (!exact_zero_rhs && retained != nullptr)
  {
    if (!retained->Solve(rhs_maximum, true_trace, linear_norm)) { return false; }
  }
  else if (!exact_zero_rhs)
  {
    mfem::Vector scaled_values(input.layout->TraceTrueVSize());
    const double* const rhs = input.true_rhs_values->HostRead();
    double* const scaled = scaled_values.HostWrite();
    bool local_ok = (rhs != nullptr && scaled != nullptr) || scaled_values.Size() == 0;
    for (int index = 0; index < scaled_values.Size() && local_ok; ++index)
    {
      scaled[index] = rhs[index] / rhs_maximum;
      local_ok = Finite(scaled[index]);
    }
    if (!CollectiveAll(input.communicator, local_ok)) { return false; }
    mfem::HypreParVector scaled_rhs(
        const_cast<mfem::ParFiniteElementSpace*>(input.trace_space));
    mfem::HypreParVector scaled_solution(
        const_cast<mfem::ParFiniteElementSpace*>(input.trace_space));
    local_ok = CopyVector(scaled_values, scaled_rhs);
    if (!CollectiveAll(input.communicator, local_ok)) { return false; }
    scaled_solution = 0.0;
    mfem::HypreILU ilu;
    ilu.SetType(0); ilu.SetLevelOfFill(0); ilu.SetMaxIter(1); ilu.SetTol(0.0);
    ilu.SetLocalReordering(0); ilu.SetPrintLevel(0);
    mfem::HypreGMRES gmres(input.communicator);
    const int global_size = static_cast<int>(input.true_matrix->GetGlobalNumRows());
    gmres.SetTol(1.0e-12); gmres.SetAbsTol(0.0);
    gmres.SetMaxIter(std::max(100, global_size));
    gmres.SetKDim(std::min(100, std::max(20, global_size)));
    gmres.SetLogging(0); gmres.SetPrintLevel(0);
    gmres.SetPreconditioner(ilu); gmres.SetOperator(*input.true_matrix);
    gmres.Mult(scaled_rhs, scaled_solution);
    mfem::HypreParVector scaled_residual(
        const_cast<mfem::ParFiniteElementSpace*>(input.trace_space));
    input.true_matrix->Mult(scaled_solution, scaled_residual);
    scaled_residual -= scaled_rhs;
    double residual_maximum = 0.0;
    double residual_norm = 0.0;
    if (!MaxAbsAndNorm(input.communicator, scaled_residual, residual_maximum,
                       residual_norm)) { return false; }
    local_ok = AllFinite(scaled_solution) && residual_norm <= 1.0e-10;
    if (!CollectiveAll(input.communicator, local_ok)) { return false; }
    const double* const scaled_solution_data = scaled_solution.HostRead();
    double* const solution = true_trace.HostWrite();
    local_ok = (scaled_solution_data != nullptr && solution != nullptr) ||
        true_trace.Size() == 0;
    for (int index = 0; index < true_trace.Size() && local_ok; ++index)
    {
      solution[index] = rhs_maximum * scaled_solution_data[index];
      local_ok = Finite(solution[index]);
    }
    if (!CollectiveAll(input.communicator, local_ok)) { return false; }
    linear_norm = rhs_maximum * residual_norm;
    if (!Finite(linear_norm)) { return false; }
  }

  local_ok = CopyVector(true_trace, next_trace);
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  local_ok = input.layout->DistributeTrace(next_trace, local_trace);
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  try
  {
    next_interior = 0.0;
    if (elements > 0)
    {
      for (int element = 0; element < elements && local_ok; ++element)
      {
        local_ok = input.layout->GatherTraceElement(element, local_trace,
                                                    gathered_trace);
        const double* const gathered = gathered_trace.HostRead();
        if (local_ok && gathered != nullptr)
        {
          std::memcpy(local_trace_values.data() +
                          static_cast<std::size_t>(element) * 40,
                      gathered, 40 * sizeof(double));
        }
        else if (local_ok) { local_ok = false; }
      }
      if (local_ok) { local_ok = input.condensation->Recover(ConstView(local_trace_values, elements, 40, 1), View(recovered, elements, 28, 1)) == core::Status::success; }
      for (int element = 0; element < elements && local_ok; ++element)
      {
        std::memcpy(element_interior.HostWrite(), recovered.data() + static_cast<std::size_t>(element) * 28, 28 * sizeof(double));
        local_ok = input.layout->ScatterInteriorElement(element, element_interior, next_interior);
      }
    }
    local_ok = local_ok && AllFinite(next_trace) && AllFinite(next_interior);
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  // The retained result is written in place only after all solve/recovery
  // gates; it is never swapped into an external caller output.
  local_ok = result.trace_increment.Size() == next_trace.Size() &&
      result.interior_increment.Size() == next_interior.Size() &&
      HasCapacity(result.trace_increment, next_trace.Size()) &&
      HasCapacity(result.interior_increment, next_interior.Size()) &&
      CopyVector(next_trace, result.trace_increment) &&
      CopyVector(next_interior, result.interior_increment);
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  result.reduced_rhs_norm = rhs_norm;
  result.linear_residual_norm = linear_norm;
  return true;
}

bool SolveAndRecoverTracedScalarIgrP1Trace(
    const TracedScalarIgrP1TraceSolveInput& input,
    TracedScalarIgrP1TraceSolveOutput& output) noexcept
{
  SolveAndRecoverTracedScalarIgrP1State state;
  if (!SolveAndRecoverTracedScalarIgrP1TraceWithState(input, state))
  {
    return false;
  }
  TracedScalarIgrP1TraceSolveOutput published;
  bool local_ok = true;
  try
  {
    published.trace_increment.SetSize(state.result.trace_increment.Size());
    published.interior_increment.SetSize(state.result.interior_increment.Size());
    local_ok = CopyVector(state.result.trace_increment, published.trace_increment) &&
        CopyVector(state.result.interior_increment, published.interior_increment);
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(input.communicator, local_ok)) { return false; }
  published.reduced_rhs_norm = state.result.reduced_rhs_norm;
  published.linear_residual_norm = state.result.linear_residual_norm;
  output.trace_increment.Swap(published.trace_increment);
  output.interior_increment.Swap(published.interior_increment);
  output.reduced_rhs_norm = published.reduced_rhs_norm;
  output.linear_residual_norm = published.linear_residual_norm;
  return true;
}

/** Per-call ownership for the one shared convenience compute path. */
struct EphemeralStepState final
{
  mfem::Vector current_local_trace;
  mfem::Vector previous_local_trace;
  std::vector<double> a, b, c, d, f, g, source_samples;
  std::vector<double> schur, reduced;
  mfem::Vector raw_local_trace;
  mfem::Vector raw_true_trace;
  mfem::Vector current_element_interior;
  mfem::Vector previous_element_interior;
  mfem::Vector current_element_trace;
  mfem::Vector previous_element_trace;
  std::vector<local::LocalOperator> operators;
  const std::vector<int>* source_element_vdofs = nullptr;
  const std::vector<int>* trace_element_vdofs = nullptr;
  mfem::Vector source_shape;
  core::HostReferenceEngine condensation;
  mfem::Vector local_rhs;
  mfem::Vector true_rhs_values;
  SolveAndRecoverTracedScalarIgrP1State solve_and_recovery;
};

struct TracedScalarIgrP1Scratch::Implementation final
{
  MPI_Comm communicator = MPI_COMM_NULL;
  const BorrowedP1DGInterfaceLayout* layout = nullptr;
  const mfem::ParFiniteElementSpace* interior_space = nullptr;
  const mfem::ParFiniteElementSpace* trace_space = nullptr;
  const mfem::ParMesh* mesh = nullptr;
  int interior_size = -1;
  int trace_true_size = -1;
  int trace_local_size = -1;
  int elements = -1;
  std::size_t source_sample_capacity = 0;
  const mfem::ParFiniteElementSpace* source_space = nullptr;
  long source_sequence = -1;
  std::vector<int> source_element_vdofs;
  std::vector<int> trace_element_vdofs;
  EphemeralStepState state{};
  RetainedTraceSystem retained{};
  bool retained_initialized = false;
  bool initialized = false;
  bool initialization_attempted = false;
  bool initialization_poisoned = false;

  bool StorageMatches(const TracedScalarIgrP1GlobalStepInput& input) const noexcept
  {
    int comparison = MPI_UNEQUAL;
    const bool communicator_matches = input.communicator != MPI_COMM_NULL &&
        MPI_Comm_compare(communicator, input.communicator, &comparison) == MPI_SUCCESS &&
        (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
    std::size_t a_capacity = 0, b_capacity = 0, c_capacity = 0;
    std::size_t d_capacity = 0, f_capacity = 0, g_capacity = 0;
    std::size_t schur_capacity = 0, reduced_capacity = 0;
    const bool capacities_are_valid =
        CheckedBatchCapacity(elements, 28 * 28, a_capacity) &&
        CheckedBatchCapacity(elements, 28 * 40, b_capacity) &&
        CheckedBatchCapacity(elements, 40 * 28, c_capacity) &&
        CheckedBatchCapacity(elements, 40 * 40, d_capacity) &&
        CheckedBatchCapacity(elements, 28, f_capacity) &&
        CheckedBatchCapacity(elements, 40, g_capacity) &&
        CheckedBatchCapacity(elements, 40 * 40, schur_capacity) &&
        CheckedBatchCapacity(elements, 40, reduced_capacity);
    return !initialization_poisoned && communicator_matches && capacities_are_valid && input.layout == layout && layout &&
        layout->IsUsable() && layout->InteriorSpace() == interior_space &&
        layout->TraceSpace() == trace_space && interior_space && trace_space &&
        interior_space->GetParMesh() == mesh && trace_space->GetParMesh() == mesh &&
        interior_space->GetVSize() == interior_size &&
        trace_space->GetVSize() == trace_local_size &&
        trace_space->GetTrueVSize() == trace_true_size &&
        layout->InteriorLocalVSize() == interior_size &&
        layout->TraceTrueVSize() == trace_true_size &&
        layout->TraceLocalVSize() == trace_local_size &&
        layout->LocalElementCount() == elements &&
        state.current_local_trace.Size() == trace_local_size &&
        state.previous_local_trace.Size() == trace_local_size &&
        state.raw_local_trace.Size() == trace_local_size &&
        state.raw_true_trace.Size() == trace_true_size &&
        state.local_rhs.Size() == trace_local_size &&
        state.true_rhs_values.Size() == trace_true_size &&
        state.current_local_trace.Capacity() >= trace_local_size &&
        state.previous_local_trace.Capacity() >= trace_local_size &&
        state.raw_local_trace.Capacity() >= trace_local_size &&
        state.raw_true_trace.Capacity() >= trace_true_size &&
        state.local_rhs.Capacity() >= trace_local_size &&
        state.true_rhs_values.Capacity() >= trace_true_size &&
        HasCapacity(state.a, a_capacity) && HasCapacity(state.b, b_capacity) &&
        HasCapacity(state.c, c_capacity) && HasCapacity(state.d, d_capacity) &&
        HasCapacity(state.f, f_capacity) && HasCapacity(state.g, g_capacity) &&
        HasCapacity(state.schur, schur_capacity) &&
        HasCapacity(state.reduced, reduced_capacity) &&
        HasCapacity(state.source_samples, source_sample_capacity) &&
        state.a.size() <= state.a.capacity() &&
        state.b.size() <= state.b.capacity() &&
        state.c.size() <= state.c.capacity() &&
        state.d.size() <= state.d.capacity() &&
        state.f.size() <= state.f.capacity() &&
        state.g.size() <= state.g.capacity() &&
        state.schur.size() <= state.schur.capacity() &&
        state.reduced.size() <= state.reduced.capacity() &&
        state.source_samples.size() <= state.source_samples.capacity() &&
        state.operators.size() == static_cast<std::size_t>(elements) &&
        state.current_element_interior.Size() == 28 &&
        state.previous_element_interior.Size() == 28 &&
        state.current_element_trace.Size() == 40 &&
        state.previous_element_trace.Size() == 40 &&
        state.source_shape.Size() == 4 &&
        state.solve_and_recovery.HasExpectedStorage(
            trace_true_size, trace_local_size, interior_size, elements) &&
        (!retained_initialized ||
         (retained.communicator == communicator && retained.layout == layout &&
          retained.trace_space == trace_space && retained.elements == elements &&
          retained.local_size == trace_local_size && retained.true_size == trace_true_size &&
          retained.trace_sequence == trace_space->GetSequence() && mesh &&
          retained.mesh_sequence == mesh->GetSequence() && retained.IsFrozen()));
  }

  bool Matches(const TracedScalarIgrP1GlobalStepInput& input) const noexcept
  {
    const bool source_matches = source_space == input.static_euler_rhs_space &&
        ((source_space == nullptr && source_element_vdofs.empty()) ||
         (source_space != nullptr && source_space->GetSequence() == source_sequence &&
          source_element_vdofs.size() == static_cast<std::size_t>(elements) * 16));
    return initialized && StorageMatches(input) && source_matches &&
        state.source_element_vdofs == &source_element_vdofs &&
        state.trace_element_vdofs == &trace_element_vdofs &&
        trace_element_vdofs.size() == static_cast<std::size_t>(elements) * 40;
  }
};

// Input/trace/source staging, element assembly/condensation, true-system
// assembly, and solve/recovery remain deliberately ordered in this one path.
// The next scratch leaf can replace only this ephemeral owner without cloning
// the numerical or collective implementation.
static TracedScalarIgrP1GlobalStepResult ComputeEphemeralStep(
    const TracedScalarIgrP1GlobalStepInput& input,
    TracedScalarIgrP1GlobalStepOutput& output,
    EphemeralStepState& state,
    RetainedTraceSystem* const retained = nullptr) noexcept
{
  // The communicator has been admitted by the caller.  This initial local
  // check is deliberately before every opaque collective phase.
  if (input.communicator == MPI_COMM_NULL)
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  if (!CollectiveAll(input.communicator, ValidInput(input)))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  const BorrowedP1DGInterfaceLayout& layout = *input.layout;
  const int elements = layout.LocalElementCount();
  const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
  if (!CollectiveAll(input.communicator, trace_space != nullptr))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  mfem::Vector& current_local_trace = state.current_local_trace;
  mfem::Vector& previous_local_trace = state.previous_local_trace;
  std::vector<double>& a = state.a;
  std::vector<double>& b = state.b;
  std::vector<double>& c = state.c;
  std::vector<double>& d = state.d;
  std::vector<double>& f = state.f;
  std::vector<double>& g = state.g;
  std::vector<double>& source_samples = state.source_samples;
  std::vector<double>& schur = state.schur;
  std::vector<double>& reduced = state.reduced;
  // Retained scratch preserves capacity only; every numerical phase starts
  // from the same empty block sequence as the legacy ephemeral call.
  a.clear(); b.clear(); c.clear(); d.clear(); f.clear(); g.clear();
  source_samples.clear(); schur.clear(); reduced.clear();
  bool local_ok = true;
  try
  {
    current_local_trace.SetSize(layout.TraceLocalVSize());
    previous_local_trace.SetSize(layout.TraceLocalVSize());
    if (state.current_element_interior.Size() != 28) state.current_element_interior.SetSize(28);
    if (state.previous_element_interior.Size() != 28) state.previous_element_interior.SetSize(28);
    if (state.current_element_trace.Size() != 40) state.current_element_trace.SetSize(40);
    if (state.previous_element_trace.Size() != 40) state.previous_element_trace.SetSize(40);
    if (state.source_shape.Size() != 4) state.source_shape.SetSize(4);
    if (state.operators.size() != static_cast<std::size_t>(elements))
      state.operators.resize(static_cast<std::size_t>(elements));
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        mfem::Geometry::SQUARE, input.parameters.quadrature_order);
    const std::size_t count = static_cast<std::size_t>(elements);
    const std::size_t points = static_cast<std::size_t>(volume_rule.GetNPoints());
    local_ok = volume_rule.GetNPoints() > 0 &&
        count <= std::numeric_limits<std::size_t>::max() / (28 * 28) &&
        count <= std::numeric_limits<std::size_t>::max() / (40 * 40) &&
        count <= std::numeric_limits<std::size_t>::max() / (points * 4);
    if (local_ok)
    {
      a.reserve(count * 28 * 28); b.reserve(count * 28 * 40);
      c.reserve(count * 40 * 28); d.reserve(count * 40 * 40);
      f.reserve(count * 28); g.reserve(count * 40);
      source_samples.resize(count * points * 4, 0.0);
      schur.resize(count * 40 * 40); reduced.resize(count * 40);
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  // Both trace distributions are collective through the existing exact-space
  // exchange plan, so all local allocation/shape checks have happened above.
  local_ok = layout.DistributeTrace(*input.current_true_trace,
                                    current_local_trace) &&
      layout.DistributeTrace(*input.previous_true_trace, previous_local_trace);
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  TracedScalarIgrP1GlobalStepResult local_result =
      TracedScalarIgrP1GlobalStepResult::success;
  try
  {
    const mfem::IntegrationRule& volume_rule = mfem::IntRules.Get(
        mfem::Geometry::SQUARE, input.parameters.quadrature_order);
    for (int element = 0; element < elements && local_ok; ++element)
    {
      mfem::Vector& current_interior = state.current_element_interior;
      mfem::Vector& previous_interior = state.previous_element_interior;
      mfem::Vector& trace = state.current_element_trace;
      mfem::Vector& previous_trace = state.previous_element_trace;
      local_ok = layout.GatherInteriorElement(element, *input.current_interior,
                                              current_interior) &&
          layout.GatherInteriorElement(element, *input.previous_interior,
                                      previous_interior) &&
          layout.GatherTraceElement(element, current_local_trace, trace) &&
          layout.GatherTraceElement(element, previous_local_trace, previous_trace) &&
          FillElementSource(input.static_euler_rhs_space,
                            input.static_euler_rhs_coefficients,
                            state.source_element_vdofs == nullptr ? nullptr :
                                state.source_element_vdofs->data() +
                                    static_cast<std::size_t>(element) * 16,
                            element,
                            volume_rule,
                            source_samples.data() +
                                static_cast<std::size_t>(element) *
                                    static_cast<std::size_t>(volume_rule.GetNPoints()) * 4,
                            state.source_shape);
      local::LocalOperator& operator_data = state.operators[
          static_cast<std::size_t>(element)];
      if (local_ok)
      {
        const local::ElementState state{
            current_interior.HostRead(), previous_interior.HostRead(),
            trace.HostRead(), previous_trace.HostRead(),
            {source_samples.data() + static_cast<std::size_t>(element) *
                 static_cast<std::size_t>(volume_rule.GetNPoints()) * 4,
             static_cast<std::size_t>(volume_rule.GetNPoints())}};
        const local::Status assembly = local::Assemble(
            input.affine_quads[element], input.parameters, state, operator_data);
        if (assembly == local::Status::inadmissible_state)
        {
          local_result = TracedScalarIgrP1GlobalStepResult::inadmissible_trial;
          local_ok = false;
        }
        else
        {
          local_ok = assembly == local::Status::success;
        }
      }
      if (local_ok)
      {
        a.insert(a.end(), operator_data.A.begin(), operator_data.A.end());
        b.insert(b.end(), operator_data.B.begin(), operator_data.B.end());
        c.insert(c.end(), operator_data.C.begin(), operator_data.C.end());
        d.insert(d.end(), operator_data.D.begin(), operator_data.D.end());
        f.insert(f.end(), operator_data.f.begin(), operator_data.f.end());
        g.insert(g.end(), operator_data.g.begin(), operator_data.g.end());
        local_ok = AllFinite(operator_data.interior_residual) &&
            AllFinite(operator_data.trace_residual);
      }
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (local_result == TracedScalarIgrP1GlobalStepResult::success && !local_ok)
  {
    local_result = TracedScalarIgrP1GlobalStepResult::failure;
  }
  const TracedScalarIgrP1GlobalStepResult assembly_result =
      CollectiveStepResult(input.communicator, local_result);
  if (assembly_result != TracedScalarIgrP1GlobalStepResult::success)
  {
    return assembly_result;
  }

  mfem::Vector& raw_local_trace = state.raw_local_trace;
  mfem::Vector& raw_true_trace = state.raw_true_trace;
  try
  {
    raw_local_trace.SetSize(layout.TraceLocalVSize()); raw_local_trace = 0.0;
    raw_true_trace.SetSize(layout.TraceTrueVSize()); raw_true_trace = 0.0;
    for (int element = 0; element < elements && local_ok; ++element)
    {
      const int* signed_vdofs = nullptr;
      if (state.trace_element_vdofs != nullptr)
      {
        signed_vdofs = state.trace_element_vdofs->data() +
            static_cast<std::size_t>(element) * 40;
      }
      else
      {
        mfem::Array<int> fallback_vdofs;
        trace_space->GetElementVDofs(element, fallback_vdofs);
        local_ok = fallback_vdofs.Size() == 40;
        signed_vdofs = local_ok ? fallback_vdofs.HostRead() : nullptr;
        if (!local_ok || signed_vdofs == nullptr) { break; }
        for (int row = 0; row < 40 && local_ok; ++row)
        {
          double sign = 1.0;
          const int vdof = DecodeSignedVDof(signed_vdofs[row], sign);
          local_ok = vdof >= 0 && vdof < raw_local_trace.Size();
          if (local_ok)
          {
            raw_local_trace[vdof] += sign *
                g[static_cast<std::size_t>(element) * 40 + row];
            local_ok = Finite(raw_local_trace[vdof]);
          }
        }
        continue;
      }
      local_ok = local_ok && signed_vdofs != nullptr;
      for (int row = 0; row < 40 && local_ok; ++row)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(signed_vdofs[row], sign);
        local_ok = vdof >= 0 && vdof < raw_local_trace.Size();
        if (local_ok)
        {
          raw_local_trace[vdof] += sign *
              g[static_cast<std::size_t>(element) * 40 + row];
          local_ok = Finite(raw_local_trace[vdof]);
        }
      }
    }
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  trace_space->Dof_TrueDof_Matrix()->MultTranspose(raw_local_trace,
                                                    raw_true_trace);
  if (!CollectiveAll(input.communicator, AllFinite(raw_true_trace)))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  double initial_maximum = 0.0;
  double initial_norm = 0.0;
  if (!MaxAbsAndNorm(input.communicator, f, raw_true_trace, initial_maximum,
                     initial_norm))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  core::HostReferenceEngine& condensation = state.condensation;
  if (elements > 0)
  {
    const core::LocalSystemBatchView system{
        {4, 8, 7, 5}, ConstView(a, elements, 28, 28),
        ConstView(b, elements, 28, 40), ConstView(c, elements, 40, 28),
        ConstView(d, elements, 40, 40), ConstView(f, elements, 28, 1),
        ConstView(g, elements, 40, 1)};
    // Condense the complete local batch once.  Recover below then uses these
    // exact retained LU factors rather than a second factorization.
    local_ok = condensation.Condense(
        system, {View(schur, elements, 40, 40), View(reduced, elements, 40, 1)})
        == core::Status::success && AllFinite(schur) && AllFinite(reduced);
  }
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  if (retained != nullptr)
  {
    if (!retained->Refresh(schur, reduced))
    {
      return TracedScalarIgrP1GlobalStepResult::failure;
    }
    const TracedScalarIgrP1TraceSolveInput solve_input{
        input.communicator, &layout, trace_space, retained->matrix.get(),
        &retained->true_rhs, &condensation};
    if (!SolveAndRecoverTracedScalarIgrP1TraceWithState(
            solve_input, state.solve_and_recovery, retained))
    {
      retained->MarkUnrefreshed();
      return TracedScalarIgrP1GlobalStepResult::failure;
    }
    // A refresh is single-use, including the exact-zero RHS path that skips
    // GMRES.  A subsequent attempt must complete a new numerical refresh.
    retained->MarkUnrefreshed();
    const TracedScalarIgrP1TraceSolveOutput& solved =
        state.solve_and_recovery.result;
    // Scratch callers bind their output storage before entry.  Publish only
    // after every numerical and collective gate, without swapping its backing.
    local_ok = output.trace_increment.Size() == solved.trace_increment.Size() &&
        output.interior_increment.Size() == solved.interior_increment.Size() &&
        HasCapacity(output.trace_increment, solved.trace_increment.Size()) &&
        HasCapacity(output.interior_increment, solved.interior_increment.Size());
    const double* const solved_trace = solved.trace_increment.HostRead();
    const double* const solved_interior = solved.interior_increment.HostRead();
    double* const output_trace = output.trace_increment.HostWrite();
    double* const output_interior = output.interior_increment.HostWrite();
    local_ok = local_ok &&
        ((solved.trace_increment.Size() == 0) ||
         (solved_trace != nullptr && output_trace != nullptr)) &&
        ((solved.interior_increment.Size() == 0) ||
         (solved_interior != nullptr && output_interior != nullptr));
    if (!CollectiveAll(input.communicator, local_ok))
    {
      retained->MarkUnrefreshed();
      return TracedScalarIgrP1GlobalStepResult::failure;
    }
    if (solved.trace_increment.Size() != 0)
      std::memcpy(output_trace, solved_trace,
                  static_cast<std::size_t>(solved.trace_increment.Size()) * sizeof(double));
    if (solved.interior_increment.Size() != 0)
      std::memcpy(output_interior, solved_interior,
                  static_cast<std::size_t>(solved.interior_increment.Size()) * sizeof(double));
    output.initial_residual_norm = initial_norm;
    output.reduced_rhs_norm = solved.reduced_rhs_norm;
    output.linear_residual_norm = solved.linear_residual_norm;
    return TracedScalarIgrP1GlobalStepResult::success;
  }

  // The convenience overload retains no matrix or Hypre state.  The scratch
  // overload returned above owns the frozen graph and its retained objects.
  std::unique_ptr<mfem::SparseMatrix> local_matrix;
  mfem::Vector& local_rhs = state.local_rhs;
  mfem::Vector& true_rhs_values = state.true_rhs_values;
  try
  {
    local_matrix = std::make_unique<mfem::SparseMatrix>(layout.TraceLocalVSize());
    local_rhs.SetSize(layout.TraceLocalVSize()); local_rhs = 0.0;
    true_rhs_values.SetSize(layout.TraceTrueVSize()); true_rhs_values = 0.0;
    for (int element = 0; element < elements && local_ok; ++element)
    {
      mfem::Array<int> vdofs;
      trace_space->GetElementVDofs(element, vdofs);
      local_ok = vdofs.Size() == 40;
      mfem::DenseMatrix local_schur(40);
      for (int row = 0; row < 40 && local_ok; ++row)
      {
        double sign = 1.0;
        const int vdof = DecodeSignedVDof(vdofs[row], sign);
        local_ok = vdof >= 0 && vdof < local_rhs.Size();
        if (!local_ok) { break; }
        local_rhs[vdof] += sign * reduced[static_cast<std::size_t>(element) * 40 + row];
        // DenseMatrix storage is not a contract for the row-major Schur
        // vector; assign every entry explicitly to preclude a transpose.
        for (int column = 0; column < 40; ++column)
        {
          local_schur(row, column) = schur[
              static_cast<std::size_t>(element) * 40 * 40 +
              static_cast<std::size_t>(row) * 40 + column];
        }
      }
      if (local_ok) { local_matrix->AddSubMatrix(vdofs, vdofs, local_schur); }
    }
    if (local_ok) { local_matrix->Finalize(); }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  // The opaque parallel assembly is entered only after all ranks have a fully
  // formed local matrix.  Do not catch this collective operation.
  mfem::ParBilinearForm parallel_form(const_cast<mfem::ParFiniteElementSpace*>(trace_space));
  std::unique_ptr<mfem::HypreParMatrix> true_matrix(
      parallel_form.ParallelAssemble(local_matrix.get()));
  local_ok = true_matrix != nullptr &&
      true_matrix->Height() == layout.TraceTrueVSize() &&
      true_matrix->Width() == layout.TraceTrueVSize() &&
      true_matrix->GetGlobalNumRows() == trace_space->GlobalTrueVSize() &&
      true_matrix->GetGlobalNumCols() == trace_space->GlobalTrueVSize() &&
      true_matrix->GetGlobalNumRows() > 0 &&
      true_matrix->GetGlobalNumRows() <= std::numeric_limits<int>::max();
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }

  trace_space->Dof_TrueDof_Matrix()->MultTranspose(local_rhs, true_rhs_values);
  if (!CollectiveAll(input.communicator, AllFinite(true_rhs_values)))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  const TracedScalarIgrP1TraceSolveInput solve_input{
      input.communicator, &layout, trace_space, true_matrix.get(),
      &true_rhs_values, &condensation};
  if (!SolveAndRecoverTracedScalarIgrP1TraceWithState(
          solve_input, state.solve_and_recovery))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  const TracedScalarIgrP1TraceSolveOutput& solved =
      state.solve_and_recovery.result;
  TracedScalarIgrP1GlobalStepOutput published;
  try
  {
    published.trace_increment.SetSize(solved.trace_increment.Size());
    published.interior_increment.SetSize(solved.interior_increment.Size());
    local_ok = CopyVector(solved.trace_increment, published.trace_increment) &&
        CopyVector(solved.interior_increment, published.interior_increment);
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  published.initial_residual_norm = initial_norm;
  published.reduced_rhs_norm = solved.reduced_rhs_norm;
  published.linear_residual_norm = solved.linear_residual_norm;
  output.trace_increment.Swap(published.trace_increment);
  output.interior_increment.Swap(published.interior_increment);
  output.initial_residual_norm = published.initial_residual_norm;
  output.reduced_rhs_norm = published.reduced_rhs_norm;
  output.linear_residual_norm = published.linear_residual_norm;
  return TracedScalarIgrP1GlobalStepResult::success;
}

TracedScalarIgrP1GlobalStepResult ComputeTracedScalarIgrP1GlobalStep(
    const TracedScalarIgrP1GlobalStepInput& input,
    TracedScalarIgrP1GlobalStepOutput& output) noexcept
{
  EphemeralStepState state;
  return ComputeEphemeralStep(input, output, state);
}

TracedScalarIgrP1Scratch::TracedScalarIgrP1Scratch(
    const MPI_Comm communicator, const BorrowedP1DGInterfaceLayout& layout) noexcept
{
  try
  {
    if (communicator == MPI_COMM_NULL || !layout.IsUsable()) { return; }
    const mfem::ParFiniteElementSpace* const interior_space = layout.InteriorSpace();
    const mfem::ParFiniteElementSpace* const trace_space = layout.TraceSpace();
    const mfem::ParMesh* const mesh =
        interior_space ? interior_space->GetParMesh() : nullptr;
    const int interior_size = layout.InteriorLocalVSize();
    const int trace_true_size = layout.TraceTrueVSize();
    const int trace_local_size = layout.TraceLocalVSize();
    const int elements = layout.LocalElementCount();
    std::size_t a_capacity = 0, b_capacity = 0, c_capacity = 0;
    std::size_t d_capacity = 0, f_capacity = 0, g_capacity = 0;
    std::size_t schur_capacity = 0, reduced_capacity = 0;
    const bool sizes_are_usable = interior_space != nullptr && trace_space != nullptr &&
        mesh != nullptr && trace_space->GetParMesh() == mesh &&
        interior_size >= 0 && trace_true_size >= 0 && trace_local_size >= 0 &&
        interior_space->GetVSize() == interior_size &&
        trace_space->GetVSize() == trace_local_size &&
        trace_space->GetTrueVSize() == trace_true_size &&
        elements >= 0 &&
        CheckedBatchCapacity(elements, 28 * 28, a_capacity) &&
        CheckedBatchCapacity(elements, 28 * 40, b_capacity) &&
        CheckedBatchCapacity(elements, 40 * 28, c_capacity) &&
        CheckedBatchCapacity(elements, 40 * 40, d_capacity) &&
        CheckedBatchCapacity(elements, 28, f_capacity) &&
        CheckedBatchCapacity(elements, 40, g_capacity) &&
        CheckedBatchCapacity(elements, 40 * 40, schur_capacity) &&
        CheckedBatchCapacity(elements, 40, reduced_capacity);
    if (!sizes_are_usable) { return; }
    const mfem::IntegrationRule& maximum_rule = mfem::IntRules.Get(
        mfem::Geometry::SQUARE, 32);
    std::size_t source_entries = 0;
    std::size_t source_sample_capacity = 0;
    if (maximum_rule.GetNPoints() <= 0 ||
        !CheckedProduct(static_cast<std::size_t>(maximum_rule.GetNPoints()), 4,
                        source_entries) ||
        !CheckedBatchCapacity(elements, source_entries, source_sample_capacity))
    {
      return;
    }
    implementation_ = std::make_unique<Implementation>();
    Implementation& state = *implementation_;
    state.communicator = communicator;
    state.layout = &layout;
    state.interior_space = interior_space;
    state.trace_space = trace_space;
    state.mesh = mesh;
    state.interior_size = interior_size;
    state.trace_true_size = trace_true_size;
    state.trace_local_size = trace_local_size;
    state.elements = elements;
    state.source_sample_capacity = source_sample_capacity;
    state.state.current_local_trace.SetSize(trace_local_size);
    state.state.previous_local_trace.SetSize(trace_local_size);
    state.state.raw_local_trace.SetSize(trace_local_size);
    state.state.raw_true_trace.SetSize(trace_true_size);
    state.state.local_rhs.SetSize(trace_local_size);
    state.state.true_rhs_values.SetSize(trace_true_size);
    state.state.a.reserve(a_capacity); state.state.b.reserve(b_capacity);
    state.state.c.reserve(c_capacity); state.state.d.reserve(d_capacity);
    state.state.f.reserve(f_capacity); state.state.g.reserve(g_capacity);
    state.state.schur.reserve(schur_capacity); state.state.reduced.reserve(reduced_capacity);
    state.state.source_samples.reserve(source_sample_capacity);
    SolveAndRecoverTracedScalarIgrP1State& solve = state.state.solve_and_recovery;
    solve.true_trace.SetSize(trace_true_size);
    solve.next_trace.SetSize(trace_true_size);
    solve.local_trace.SetSize(trace_local_size);
    solve.next_interior.SetSize(interior_size);
    solve.gathered_trace.SetSize(40);
    solve.element_interior.SetSize(28);
    solve.recovered.resize(f_capacity);
    solve.local_trace_values.resize(g_capacity);
    solve.result.trace_increment.SetSize(trace_true_size);
    solve.result.interior_increment.SetSize(interior_size);
    TracedScalarIgrP1GlobalStepInput binding{};
    binding.communicator = communicator;
    binding.layout = &layout;
    state.state.current_element_interior.SetSize(28);
    state.state.previous_element_interior.SetSize(28);
    state.state.current_element_trace.SetSize(40);
    state.state.previous_element_trace.SetSize(40);
    state.state.source_shape.SetSize(4);
    state.state.operators.resize(static_cast<std::size_t>(elements));
    state.state.source_element_vdofs = &state.source_element_vdofs;
    state.state.trace_element_vdofs = &state.trace_element_vdofs;
    if (!state.StorageMatches(binding)) { implementation_.reset(); }
  }
  catch (...) { implementation_.reset(); }
}

TracedScalarIgrP1Scratch::~TracedScalarIgrP1Scratch() = default;
TracedScalarIgrP1Scratch::TracedScalarIgrP1Scratch(
    TracedScalarIgrP1Scratch&&) noexcept = default;
TracedScalarIgrP1Scratch& TracedScalarIgrP1Scratch::operator=(
    TracedScalarIgrP1Scratch&&) noexcept = default;

bool TracedScalarIgrP1Scratch::Initialize(
    const TracedScalarIgrP1GlobalStepInput& input) noexcept
{
  if (input.communicator == MPI_COMM_NULL)
  {
    if (implementation_)
    {
      implementation_->initialization_attempted = true;
      implementation_->initialization_poisoned = true;
    }
    return false;
  }
  const bool local_admission = implementation_ != nullptr &&
      !implementation_->initialization_attempted &&
      implementation_->StorageMatches(input) && ValidInput(input);
  if (!CollectiveAll(input.communicator, local_admission))
  {
    if (implementation_) { implementation_->initialization_attempted = true; implementation_->initialization_poisoned = true; }
    return false;
  }
  TracedScalarIgrP1Scratch::Implementation& implementation = *implementation_;
  implementation.initialization_attempted = true;
  bool local_ok = true;
  try
  {
    implementation.source_space = input.static_euler_rhs_space;
    implementation.source_sequence = implementation.source_space == nullptr ? -1 :
        implementation.source_space->GetSequence();
    for (int order = 2; order <= 32; ++order)
    {
      (void)mfem::IntRules.Get(mfem::Geometry::SQUARE, order);
      (void)mfem::IntRules.Get(mfem::Geometry::SEGMENT, order);
    }
    local_ok = CaptureElementVDofs(implementation.trace_space,
        implementation.elements, 40, implementation.trace_element_vdofs) &&
        (implementation.source_space == nullptr || CaptureElementVDofs(
            implementation.source_space, implementation.elements, 16,
            implementation.source_element_vdofs)) &&
        implementation.state.condensation.Prepare({4, 8, 7, 5},
            static_cast<std::size_t>(implementation.elements)) == core::Status::success;
    if (local_ok)
      for (local::LocalOperator& operator_data : implementation.state.operators)
        operator_data.Resize();
  }
  catch (const std::exception&) { local_ok = false; }
  if (!CollectiveAll(input.communicator, local_ok))
  {
    implementation.initialization_poisoned = true;
    return false;
  }
  if (!implementation.retained.Initialize(input.communicator, *input.layout) ||
      !implementation.retained.Prime())
  {
    implementation.initialization_poisoned = true;
    return false;
  }
  implementation.retained_initialized = true;
  implementation.initialized = true;
  if (!CollectiveAll(input.communicator, implementation.Matches(input)))
  {
    implementation.initialization_poisoned = true;
    implementation.initialized = false;
    return false;
  }
  return true;
}

bool TracedScalarIgrP1Scratch::IsReady() const noexcept
{
  if (!implementation_ || !implementation_->initialized ||
      implementation_->initialization_poisoned) return false;
  TracedScalarIgrP1GlobalStepInput binding{};
  binding.communicator = implementation_->communicator;
  binding.layout = implementation_->layout;
  binding.static_euler_rhs_space = implementation_->source_space;
  return implementation_->Matches(binding);
}

std::size_t TracedScalarIgrP1Scratch::WorkspaceBytes() const noexcept
{
  if (!IsReady()) return 0;
  const Implementation& impl = *implementation_;
  std::size_t total = 0;
  const auto add = [&total](const std::size_t count, const std::size_t width)
  {
    if (width != 0 && count > (std::numeric_limits<std::size_t>::max() - total) / width)
      return false;
    total += count * width;
    return true;
  };
  const auto add_vector = [&add](const mfem::Vector& values)
  { return add(static_cast<std::size_t>(values.Capacity()), sizeof(double)); };
  const auto add_doubles = [&add](const std::vector<double>& values)
  { return add(values.capacity(), sizeof(double)); };
  const auto add_ints = [&add](const std::vector<int>& values)
  { return add(values.capacity(), sizeof(int)); };
  const EphemeralStepState& state = impl.state;
  bool ok = add(sizeof(Implementation), 1) &&
      add_ints(impl.source_element_vdofs) && add_ints(impl.trace_element_vdofs) &&
      add_vector(state.current_local_trace) && add_vector(state.previous_local_trace) &&
      add_vector(state.raw_local_trace) && add_vector(state.raw_true_trace) &&
      add_vector(state.current_element_interior) && add_vector(state.previous_element_interior) &&
      add_vector(state.current_element_trace) && add_vector(state.previous_element_trace) &&
      add_vector(state.source_shape) && add_vector(state.local_rhs) &&
      add_vector(state.true_rhs_values) && add_doubles(state.a) && add_doubles(state.b) &&
      add_doubles(state.c) && add_doubles(state.d) && add_doubles(state.f) &&
      add_doubles(state.g) && add_doubles(state.schur) && add_doubles(state.reduced) &&
      add_doubles(state.source_samples) &&
      add(state.operators.capacity(), sizeof(local::LocalOperator));
  const SolveAndRecoverTracedScalarIgrP1State& solve = state.solve_and_recovery;
  ok = ok && add_vector(solve.true_trace) && add_vector(solve.next_trace) &&
      add_vector(solve.local_trace) && add_vector(solve.next_interior) &&
      add_vector(solve.gathered_trace) && add_vector(solve.element_interior) &&
      add_doubles(solve.recovered) && add_doubles(solve.local_trace_values) &&
      add_vector(solve.result.trace_increment) && add_vector(solve.result.interior_increment);
  const RetainedTraceSystem& retained = impl.retained;
  ok = ok && add(retained.element_vdofs.capacity(), sizeof(int)) &&
      add(retained.local_dofs.capacity(), sizeof(RetainedTraceSystem::LocalDof)) &&
      add(retained.owned.capacity(), sizeof(RetainedTraceSystem::Slot)) &&
      add(retained.outgoing.capacity(), sizeof(RetainedTraceSystem::Outgoing)) &&
      add(retained.incoming.capacity(), sizeof(RetainedTraceSystem::Incoming)) &&
      add_ints(retained.send_counts) && add_ints(retained.receive_counts) &&
      add_ints(retained.send_offsets) && add_ints(retained.receive_offsets) &&
      add_doubles(retained.send_values) && add_doubles(retained.receive_values) &&
      add_vector(retained.local_rhs) && add_vector(retained.true_rhs) &&
      add(static_cast<std::size_t>(std::max(0, retained.diagonal.NumNonZeroElems())),
          sizeof(double) + sizeof(int)) &&
      add(static_cast<std::size_t>(std::max(0, retained.offdiagonal.NumNonZeroElems())),
          sizeof(double) + sizeof(int) + sizeof(HYPRE_BigInt)) &&
      add(static_cast<std::size_t>(std::max(0, retained.true_size) + 1), sizeof(int)) &&
      add(static_cast<std::size_t>(std::max(0, retained.true_size)), 3 * sizeof(double)) &&
      add(sizeof(mfem::HypreParMatrix) + sizeof(mfem::HypreParVector) * 3 +
          sizeof(mfem::HypreILU) + sizeof(mfem::HypreGMRES), 1);
  return ok ? total : 0;
}

bool TracedScalarIgrP1Scratch::IsUsable() const noexcept
{
  if (!implementation_ || implementation_->communicator == MPI_COMM_NULL ||
      implementation_->layout == nullptr) return false;
  TracedScalarIgrP1GlobalStepInput binding{};
  binding.communicator = implementation_->communicator;
  binding.layout = implementation_->layout;
  return implementation_->StorageMatches(binding);
}

TracedScalarIgrP1GlobalStepResult ComputeTracedScalarIgrP1GlobalStep(
    const TracedScalarIgrP1GlobalStepInput& input,
    TracedScalarIgrP1GlobalStepOutput& output,
    TracedScalarIgrP1Scratch& scratch) noexcept
{
  const bool local_ok = scratch.implementation_ && scratch.implementation_->Matches(input) &&
      output.trace_increment.Size() == input.layout->TraceTrueVSize() &&
      output.interior_increment.Size() == input.layout->InteriorLocalVSize() &&
      HasCapacity(output.trace_increment, output.trace_increment.Size()) &&
      HasCapacity(output.interior_increment, output.interior_increment.Size());
  if (input.communicator == MPI_COMM_NULL ||
      !CollectiveAll(input.communicator, local_ok))
  {
    return TracedScalarIgrP1GlobalStepResult::failure;
  }
  TracedScalarIgrP1Scratch::Implementation& implementation =
      *scratch.implementation_;
  return ComputeEphemeralStep(input, output, implementation.state,
                              &implementation.retained);
}

} // namespace dg::hdg::runtime::cpu::detail

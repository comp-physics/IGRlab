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
  std::vector<double> vertices, volume_quadrature, segment_quadrature;
  ArenaPlan arena;
  void* integer_arena = nullptr;
  void* double_arena = nullptr;
  bool integer_release_attempted = false;
  bool double_release_attempted = false;
  std::unique_ptr<mfem::HypreParMatrix> matrix;
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
      if (!Add(state.arena.symbolic_slots, 1600, state.arena.symbolic_slots)) return false;
    }
    graph->Finalize(0, false);
    form = std::make_unique<mfem::ParBilinearForm>(state.trace_space.get());
    return graph && form;
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
  if (!WorldAll(!setup_attempted_))
  {
    // A divergent re-entry consumes the fresh ranks too.  This prevents a
    // later invocation from pairing a fresh rank with a terminal peer.
    setup_attempted_ = true;
    terminal_poisoned_ = true;
    if (implementation_) implementation_->poisoned = true;
    return false;
  }
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
bool RetainedTopology::IdentityPrimed() const noexcept
{ return IsReady() && implementation_->identity_primed; }
double RetainedTopology::PrimeResidual() const noexcept
{ return IsReady() ? implementation_->prime_residual : std::numeric_limits<double>::infinity(); }
double RetainedTopology::PrimeMaximumSolutionError() const noexcept
{ return IsReady() ? implementation_->prime_maximum_solution_error : std::numeric_limits<double>::infinity(); }
bool RetainedTopology::PrimeUsedDeviceIlu0() const noexcept
{ return IsReady() && implementation_->device_ilu0; }
} // namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private_numerics

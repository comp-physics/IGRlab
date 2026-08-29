#include "numerics/hdg/runtime/cuda/native_rank1_trace_solver.cuh"

#include <cooperative_groups.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace dg::hdg::runtime::cuda_w8a::detail
{
namespace
{
namespace cg = cooperative_groups;

inline constexpr std::uint32_t failure_invalid_policy = 1U << 0;
inline constexpr std::uint32_t failure_invalid_system = 1U << 1;
inline constexpr std::uint32_t failure_invalid_preconditioner = 1U << 2;
inline constexpr std::uint32_t failure_invalid_workspace = 1U << 3;
inline constexpr std::uint32_t failure_nonfinite = 1U << 4;
inline constexpr std::uint32_t failure_invalid_permutation = 1U << 5;
inline constexpr std::uint32_t failure_numerical_breakdown = 1U << 6;
inline constexpr int native_trace_projection_tile = 8;
inline constexpr int native_trace_schwarz_maximum_incidence = 2;
inline constexpr int native_trace_schwarz_faces = 4;
inline constexpr int native_trace_schwarz_igr_components = 5;
inline constexpr int native_trace_schwarz_euler_components = 4;
inline constexpr double native_trace_schwarz_minimum_scale = 0x1p-40;
inline constexpr double native_trace_schwarz_maximum_scale = 0x1p40;

__host__ __device__ constexpr bool NativeTracePreconditionerIsElementSchwarz(
    const NativeTracePreconditionerKind kind) noexcept
{
  return kind == NativeTracePreconditionerKind::
          element_overlap_schwarz_fp64 ||
      kind == NativeTracePreconditionerKind::
          element_overlap_schwarz_equilibrated_field_split_fp64 ||
      kind == NativeTracePreconditionerKind::
          element_overlap_schwarz_warp_equilibrated_field_split_fp64;
}

__host__ __device__ constexpr bool
NativeTracePreconditionerIsEquilibratedFieldSplit(
    const NativeTracePreconditionerKind kind) noexcept
{
  return kind == NativeTracePreconditionerKind::
          element_overlap_schwarz_equilibrated_field_split_fp64 ||
      kind == NativeTracePreconditionerKind::
          element_overlap_schwarz_warp_equilibrated_field_split_fp64;
}

__host__ __device__ constexpr bool
NativeTracePreconditionerIsWarpEquilibratedFieldSplit(
    const NativeTracePreconditionerKind kind) noexcept
{
  return kind == NativeTracePreconditionerKind::
      element_overlap_schwarz_warp_equilibrated_field_split_fp64;
}

__host__ __device__ constexpr bool
NativeTracePreconditionerUsesFixedSigmaJacobi(
    const NativeTracePreconditionerKind kind) noexcept
{
  return kind == NativeTracePreconditionerKind::
          element_overlap_schwarz_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == NativeTracePreconditionerKind::
          element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64;
}

constexpr NativeTracePreconditionerKind
NativeTraceFixedSigmaJacobiBasePreconditioner(
    const NativeTracePreconditionerKind kind) noexcept
{
  return kind == NativeTracePreconditionerKind::
              element_overlap_schwarz_equilibrated_field_split_sigma_jacobi3_fp64
      ? NativeTracePreconditionerKind::
            element_overlap_schwarz_equilibrated_field_split_fp64
      : kind == NativeTracePreconditionerKind::
              element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64
          ? NativeTracePreconditionerKind::
                element_overlap_schwarz_warp_equilibrated_field_split_fp64
          : kind;
}

__host__ __device__ constexpr bool NativeTraceOperatorIsElementMatrixFree(
    const NativeTraceOperatorKind kind) noexcept
{
  return kind == NativeTraceOperatorKind::element_matrix_free_fp64 ||
      kind == NativeTraceOperatorKind::
          element_matrix_free_equilibrated_field_split_fp64 ||
      kind == NativeTraceOperatorKind::
          element_matrix_free_warp_equilibrated_field_split_fp64;
}

__host__ __device__ constexpr bool NativeTraceOperatorMatchesPreconditioner(
    const NativeTraceOperatorKind operator_kind,
    const NativeTracePreconditionerKind preconditioner_kind) noexcept
{
  return (operator_kind == NativeTraceOperatorKind::element_matrix_free_fp64 &&
          preconditioner_kind == NativeTracePreconditionerKind::
              element_overlap_schwarz_fp64) ||
      (operator_kind == NativeTraceOperatorKind::
               element_matrix_free_equilibrated_field_split_fp64 &&
       preconditioner_kind == NativeTracePreconditionerKind::
           element_overlap_schwarz_equilibrated_field_split_fp64) ||
      (operator_kind == NativeTraceOperatorKind::
               element_matrix_free_warp_equilibrated_field_split_fp64 &&
       preconditioner_kind == NativeTracePreconditionerKind::
           element_overlap_schwarz_warp_equilibrated_field_split_fp64);
}

bool CheckedAdd(const std::size_t left, const std::size_t right,
                std::size_t& result) noexcept
{
  if (right > std::numeric_limits<std::size_t>::max() - left) return false;
  result = left + right;
  return true;
}

bool CheckedMultiply(const std::size_t left, const std::size_t right,
                     std::size_t& result) noexcept
{
  if (left != 0 &&
      right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  result = left * right;
  return true;
}

bool IsAligned(const void* pointer, const std::size_t alignment) noexcept
{
  return pointer != nullptr && alignment != 0 &&
      reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

struct AddressRange final
{
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
  bool valid = false;
};

AddressRange MakeAddressRange(const void* pointer, const std::size_t count,
                              const std::size_t element_size) noexcept
{
  AddressRange result{};
  std::size_t bytes = 0;
  if (pointer == nullptr || count == 0 ||
      !CheckedMultiply(count, element_size, bytes))
    return result;
  result.begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - result.begin)
    return AddressRange{};
  result.end = result.begin + bytes;
  result.valid = result.end > result.begin;
  return result;
}

bool AddressRangesOverlap(const AddressRange& left,
                          const AddressRange& right) noexcept
{
  return !left.valid || !right.valid ||
      (left.begin < right.end && right.begin < left.end);
}

__device__ int GlobalThreadIndex() noexcept
{
  return static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ int GlobalThreadStride() noexcept
{
  return static_cast<int>(gridDim.x * blockDim.x);
}

__device__ void MarkFailure(NativeTraceSolveControl* control,
                            const std::uint32_t failure) noexcept
{
  atomicOr(reinterpret_cast<unsigned int*>(&control->failure_mask), failure);
}

__device__ NativeTraceSolveStatus FailureStatus(
    const std::uint32_t failure_mask) noexcept
{
  if ((failure_mask & failure_invalid_policy) != 0)
    return NativeTraceSolveStatus::invalid_policy;
  if ((failure_mask & failure_invalid_workspace) != 0)
    return NativeTraceSolveStatus::invalid_workspace;
  if ((failure_mask & failure_invalid_system) != 0)
    return NativeTraceSolveStatus::invalid_system;
  if ((failure_mask & failure_invalid_preconditioner) != 0 ||
      (failure_mask & failure_invalid_permutation) != 0)
    return NativeTraceSolveStatus::invalid_preconditioner;
  if ((failure_mask & failure_nonfinite) != 0)
    return NativeTraceSolveStatus::nonfinite_input;
  return NativeTraceSolveStatus::numerical_breakdown;
}

__device__ bool FinishOnFailure(const cg::grid_group& grid,
                                NativeTraceSolveControl* control) noexcept
{
  if (grid.thread_rank() == 0 && control->failure_mask != 0)
    control->status = FailureStatus(control->failure_mask);
  grid.sync();
  return control->failure_mask != 0;
}

__device__ void ValidateCsr(const NativeTraceCsrView matrix,
                            const bool require_diagonal_first,
                            const bool require_nonzero_diagonal,
                            const std::uint32_t structural_failure,
                            NativeTraceSolveControl* control) noexcept
{
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  if (thread == 0 &&
      (matrix.row_offsets[0] != 0 ||
       matrix.row_offsets[matrix.rows] != matrix.nonzeros))
    MarkFailure(control, structural_failure);

  for (int row = thread; row < matrix.rows; row += stride)
  {
    const int begin = matrix.row_offsets[row];
    const int end = matrix.row_offsets[row + 1];
    if (begin < 0 || end <= begin || end > matrix.nonzeros)
    {
      MarkFailure(control, structural_failure);
      continue;
    }
    if (require_diagonal_first && matrix.columns[begin] != row)
      MarkFailure(control, structural_failure);
    int diagonal_count = 0;
    for (int entry = begin; entry < end; ++entry)
    {
      const int column = matrix.columns[entry];
      const double value = matrix.values[entry];
      if (column < 0 || column >= matrix.column_count)
        MarkFailure(control, structural_failure);
      if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
      if (column == row)
      {
        ++diagonal_count;
        if (require_nonzero_diagonal && value == 0.0)
          MarkFailure(control, structural_failure);
      }
    }
    if (diagonal_count != 1) MarkFailure(control, structural_failure);
  }
}

__device__ void ValidatePermutation(
    const NativeTraceIlu0View ilu0, const NativeTraceWorkspaceView workspace,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int row = thread; row < ilu0.factor.rows; row += stride)
    workspace.permutation_marks[row] = 0;
  grid.sync();

  for (int row = thread; row < ilu0.factor.rows; row += stride)
  {
    const int original = ilu0.permutation[row];
    if (original < 0 || original >= ilu0.factor.rows)
    {
      MarkFailure(control, failure_invalid_permutation);
      continue;
    }
    if (atomicCAS(workspace.permutation_marks + original, 0, 1) != 0)
      MarkFailure(control, failure_invalid_permutation);
  }
  grid.sync();

  for (int row = thread; row < ilu0.factor.rows; row += stride)
    if (workspace.permutation_marks[row] != 1)
      MarkFailure(control, failure_invalid_permutation);
}

__device__ int DecodeSignedDof(const int encoded, double& sign) noexcept
{
  sign = encoded >= 0 ? 1.0 : -1.0;
  return encoded >= 0 ? encoded : -1 - encoded;
}

__host__ __device__ int FieldSplitScalarFaceRows(
    const int patch_rows) noexcept
{
  return patch_rows /
      (native_trace_schwarz_faces * native_trace_schwarz_igr_components);
}

__host__ __device__ int FieldSplitEulerRows(const int patch_rows) noexcept
{
  return native_trace_schwarz_faces *
      native_trace_schwarz_euler_components *
      FieldSplitScalarFaceRows(patch_rows);
}

__host__ __device__ int FieldSplitOriginalRow(
    const int patch_rows, const int packed_row) noexcept
{
  const int scalar_rows = FieldSplitScalarFaceRows(patch_rows);
  const int euler_rows = FieldSplitEulerRows(patch_rows);
  const int face_span = native_trace_schwarz_igr_components * scalar_rows;
  if (packed_row < euler_rows)
  {
    const int euler_face_span =
        native_trace_schwarz_euler_components * scalar_rows;
    const int face = packed_row / euler_face_span;
    return face * face_span + packed_row - face * euler_face_span;
  }
  const int sigma = packed_row - euler_rows;
  const int face = sigma / scalar_rows;
  return face * face_span +
      native_trace_schwarz_euler_components * scalar_rows +
      sigma - face * scalar_rows;
}

__host__ __device__ int FieldSplitPackedRow(
    const int patch_rows, const int original_row) noexcept
{
  const int scalar_rows = FieldSplitScalarFaceRows(patch_rows);
  const int euler_rows = FieldSplitEulerRows(patch_rows);
  const int face_span = native_trace_schwarz_igr_components * scalar_rows;
  const int face = original_row / face_span;
  const int within_face = original_row - face * face_span;
  const int component = within_face / scalar_rows;
  const int scalar = within_face - component * scalar_rows;
  if (component < native_trace_schwarz_euler_components)
    return face * native_trace_schwarz_euler_components * scalar_rows +
        component * scalar_rows + scalar;
  return euler_rows + face * scalar_rows + scalar;
}

__host__ __device__ std::size_t FieldSplitSigmaLuOffset(
    const int patch_rows) noexcept
{
  const int euler_rows = FieldSplitEulerRows(patch_rows);
  return static_cast<std::size_t>(euler_rows) * euler_rows;
}

__host__ __device__ std::size_t FieldSplitLowerCouplingOffset(
    const int patch_rows) noexcept
{
  const int sigma_rows = patch_rows - FieldSplitEulerRows(patch_rows);
  return FieldSplitSigmaLuOffset(patch_rows) +
      static_cast<std::size_t>(sigma_rows) * sigma_rows;
}

__host__ __device__ std::size_t FieldSplitLeftScaleOffset(
    const int patch_rows) noexcept
{
  const int euler_rows = FieldSplitEulerRows(patch_rows);
  const int sigma_rows = patch_rows - euler_rows;
  return FieldSplitLowerCouplingOffset(patch_rows) +
      static_cast<std::size_t>(sigma_rows) * euler_rows;
}

__host__ __device__ std::size_t FieldSplitRightScaleOffset(
    const int patch_rows) noexcept
{
  return FieldSplitLeftScaleOffset(patch_rows) + patch_rows;
}

__host__ __device__ std::size_t FieldSplitWarpInverseDiagonalOffset(
    const int patch_rows) noexcept
{
  return FieldSplitRightScaleOffset(patch_rows) + patch_rows;
}

__host__ __device__ std::size_t FieldSplitWarpRhsPermutationOffset(
    const int patch_rows) noexcept
{
  return FieldSplitWarpInverseDiagonalOffset(patch_rows) + patch_rows;
}

__host__ __device__ std::size_t FieldSplitWarpRequiredDoubles(
    const int patch_rows) noexcept
{
  return FieldSplitWarpRhsPermutationOffset(patch_rows) + patch_rows;
}

__device__ void SolvePivotedDenseLu(
    const double* const lu, const int* const pivots, const int rows,
    double* const solution) noexcept
{
  for (int row = 0; row < rows; ++row)
  {
    const int pivot = pivots[row];
    if (pivot != row)
    {
      const double value = solution[row];
      solution[row] = solution[pivot];
      solution[pivot] = value;
    }
  }
  for (int row = 0; row < rows; ++row)
  {
    double value = solution[row];
    for (int column = 0; column < row; ++column)
      value -= lu[row * rows + column] * solution[column];
    solution[row] = value;
  }
  for (int row = rows - 1; row >= 0; --row)
  {
    double value = solution[row];
    for (int column = row + 1; column < rows; ++column)
      value -= lu[row * rows + column] * solution[column];
    solution[row] = value / lu[row * rows + row];
  }
}

template <int TileWidth>
__device__ double FieldSplitTileSum(double value,
                                    const unsigned int mask) noexcept
{
#pragma unroll
  for (int offset = TileWidth / 2; offset > 0; offset /= 2)
    value += __shfl_down_sync(mask, value, offset, TileWidth);
  return value;
}

template <int TileWidth>
__device__ void SolveFieldSplitLuCooperatively(
    const double* const lu, const double* const inverse_diagonal,
    const int rows, double* const solution, const int lane,
    const unsigned int mask) noexcept
{
  for (int row = 1; row < rows; ++row)
  {
    double partial = 0.0;
    for (int column = lane; column < row; column += TileWidth)
      partial = fma(lu[row * rows + column], solution[column], partial);
    const double sum = FieldSplitTileSum<TileWidth>(partial, mask);
    if (lane == 0) solution[row] -= sum;
    __syncwarp(mask);
  }
  for (int row = rows - 1; row >= 0; --row)
  {
    double partial = 0.0;
    for (int column = row + 1 + lane; column < rows;
         column += TileWidth)
      partial = fma(lu[row * rows + column], solution[column], partial);
    const double sum = FieldSplitTileSum<TileWidth>(partial, mask);
    if (lane == 0)
      solution[row] = (solution[row] - sum) * inverse_diagonal[row];
    __syncwarp(mask);
  }
}

template <int TileWidth>
__device__ void ApplyWarpEquilibratedFieldSplit(
    const NativeTraceElementSchwarzView schwarz,
    NativeTraceSolveControl* const control) noexcept
{
  static_assert(TileWidth == 16 || TileWidth == 32);
  const int lane = static_cast<int>(threadIdx.x) % TileWidth;
  const int tile_in_warp =
      (static_cast<int>(threadIdx.x) & 31) / TileWidth;
  const unsigned int mask = TileWidth == 32
      ? 0xffffffffU : 0xffffU << (16 * tile_in_warp);
  const int tile = GlobalThreadIndex() / TileWidth;
  const int tile_stride = GlobalThreadStride() / TileWidth;
  const int patch_rows = schwarz.patch_rows;
  const int euler_rows = FieldSplitEulerRows(patch_rows);
  const int sigma_rows = patch_rows - euler_rows;

  for (int element = tile; element < schwarz.elements;
       element += tile_stride)
  {
    const std::size_t vector_offset =
        static_cast<std::size_t>(element) * patch_rows;
    const std::size_t matrix_offset = vector_offset * patch_rows;
    const double* const storage = schwarz.patch_lu + matrix_offset;
    const double* const euler_lu = storage;
    const double* const sigma_lu =
        storage + FieldSplitSigmaLuOffset(patch_rows);
    const double* const lower_coupling =
        storage + FieldSplitLowerCouplingOffset(patch_rows);
    const double* const left_scale =
        storage + FieldSplitLeftScaleOffset(patch_rows);
    const double* const right_scale =
        storage + FieldSplitRightScaleOffset(patch_rows);
    const double* const inverse_diagonal =
        storage + FieldSplitWarpInverseDiagonalOffset(patch_rows);
    const double* const rhs_permutation =
        storage + FieldSplitWarpRhsPermutationOffset(patch_rows);
    const double* const rhs = schwarz.element_rhs + vector_offset;
    double* const solution = schwarz.element_solution + vector_offset;

    // Apply the setup-time pivot permutation while packing the Euler RHS.
    for (int packed = lane; packed < euler_rows; packed += TileWidth)
    {
      const int source = static_cast<int>(rhs_permutation[packed]);
      const int original = FieldSplitOriginalRow(patch_rows, source);
      const double value = left_scale[source] * rhs[original];
      solution[packed] = value;
      if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
    }
    __syncwarp(mask);
    SolveFieldSplitLuCooperatively<TileWidth>(
        euler_lu, inverse_diagonal, euler_rows, solution, lane, mask);

    // The sigma pivot acts on the already-adjusted RHS.  Use its final source
    // row directly so no recurring serial swap chain remains.
    for (int sigma = 0; sigma < sigma_rows; ++sigma)
    {
      const int packed = euler_rows + sigma;
      const int source = static_cast<int>(rhs_permutation[packed]);
      const int source_sigma = source - euler_rows;
      double partial = 0.0;
      for (int euler = lane; euler < euler_rows; euler += TileWidth)
        partial = fma(lower_coupling[source_sigma * euler_rows + euler],
                      solution[euler], partial);
      const double coupling = FieldSplitTileSum<TileWidth>(partial, mask);
      if (lane == 0)
      {
        const int original = FieldSplitOriginalRow(patch_rows, source);
        const double value = left_scale[source] * rhs[original] - coupling;
        solution[packed] = value;
        if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
      }
      __syncwarp(mask);
    }

    for (int packed = lane; packed < euler_rows; packed += TileWidth)
      solution[packed] *= right_scale[packed];
    __syncwarp(mask);
    SolveFieldSplitLuCooperatively<TileWidth>(
        sigma_lu, inverse_diagonal + euler_rows, sigma_rows,
        solution + euler_rows, lane, mask);
    for (int sigma = lane; sigma < sigma_rows; sigma += TileWidth)
    {
      const int packed = euler_rows + sigma;
      solution[packed] *= right_scale[packed];
      if (!isfinite(solution[packed]))
        MarkFailure(control, failure_nonfinite);
    }
    __syncwarp(mask);
  }
}

__device__ void ValidateElementSchwarz(
    const NativeTraceElementSchwarzView schwarz,
    const bool equilibrated_field_split,
    const bool warp_equilibrated_field_split,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  const int patch_rows = schwarz.patch_rows;
  const int element_slots = schwarz.elements * patch_rows;
  const int patch_entries = element_slots * patch_rows;
  for (int entry = thread; entry < patch_entries; entry += stride)
  {
    if (!isfinite(schwarz.patch_matrix[entry]) ||
        !isfinite(schwarz.patch_lu[entry]))
      MarkFailure(control, failure_nonfinite);
  }
  for (int slot = thread; slot < element_slots; slot += stride)
  {
    double sign = 1.0;
    const int truth = DecodeSignedDof(
        schwarz.signed_element_true_dofs[slot], sign);
    if (truth < 0 || truth >= schwarz.true_rows ||
        (sign != 1.0 && sign != -1.0))
      MarkFailure(control, failure_invalid_preconditioner);
  }
  for (int item = thread; item < element_slots; item += stride)
  {
    const int local_item = item % patch_rows;
    const int pivot = schwarz.pivots[item];
    const std::size_t matrix_offset =
        static_cast<std::size_t>(item / patch_rows) * patch_rows *
        patch_rows;
    if (equilibrated_field_split)
    {
      const int euler_rows = FieldSplitEulerRows(patch_rows);
      const int block_rows = local_item < euler_rows
          ? euler_rows : patch_rows - euler_rows;
      const int block_column = local_item < euler_rows
          ? local_item : local_item - euler_rows;
      const std::size_t factor_offset = local_item < euler_rows
          ? 0 : FieldSplitSigmaLuOffset(patch_rows);
      const double diagonal = schwarz.patch_lu[
          matrix_offset + factor_offset +
          static_cast<std::size_t>(block_column) * block_rows +
          block_column];
      const double left_scale = schwarz.patch_lu[
          matrix_offset + FieldSplitLeftScaleOffset(patch_rows) +
          local_item];
      const double right_scale = schwarz.patch_lu[
          matrix_offset + FieldSplitRightScaleOffset(patch_rows) +
          local_item];
      if (pivot < block_column || pivot >= block_rows || diagonal == 0.0 ||
          !isfinite(diagonal) || !isfinite(left_scale) ||
          !isfinite(right_scale) ||
          left_scale < native_trace_schwarz_minimum_scale ||
          left_scale > native_trace_schwarz_maximum_scale ||
          right_scale < native_trace_schwarz_minimum_scale ||
          right_scale > native_trace_schwarz_maximum_scale)
        MarkFailure(control, failure_invalid_preconditioner);
      if (warp_equilibrated_field_split)
      {
        const double inverse_diagonal = schwarz.patch_lu[
            matrix_offset +
            FieldSplitWarpInverseDiagonalOffset(patch_rows) + local_item];
        const double permutation_value = schwarz.patch_lu[
            matrix_offset +
            FieldSplitWarpRhsPermutationOffset(patch_rows) + local_item];
        const int block_begin = local_item < euler_rows ? 0 : euler_rows;
        const int source = static_cast<int>(permutation_value);
        const double reciprocal_error =
            fabs(fma(diagonal, inverse_diagonal, -1.0));
        if (!isfinite(inverse_diagonal) || inverse_diagonal == 0.0 ||
            !isfinite(permutation_value) ||
            permutation_value != static_cast<double>(source) ||
            source < block_begin || source >= block_begin + block_rows ||
            !isfinite(reciprocal_error) ||
            reciprocal_error > 8.0 * 2.2204460492503131e-16)
          MarkFailure(control, failure_invalid_preconditioner);
      }
    }
    else
    {
      const double diagonal = schwarz.patch_lu[
          matrix_offset + static_cast<std::size_t>(local_item) * patch_rows +
          local_item];
      if (pivot < local_item || pivot >= patch_rows || diagonal == 0.0 ||
          !isfinite(diagonal))
        MarkFailure(control, failure_invalid_preconditioner);
    }
  }
  for (int row = thread; row < schwarz.true_rows; row += stride)
  {
    int multiplicity = 0;
    for (int incidence = 0;
         incidence < native_trace_schwarz_maximum_incidence;
         ++incidence)
    {
      const int slot = schwarz.true_incidence_slots[
          native_trace_schwarz_maximum_incidence * row + incidence];
      if (slot < 0) continue;
      if (slot >= element_slots)
      {
        MarkFailure(control, failure_invalid_preconditioner);
        continue;
      }
      double sign = 1.0;
      const int truth = DecodeSignedDof(
          schwarz.signed_element_true_dofs[slot], sign);
      if (truth != row)
        MarkFailure(control, failure_invalid_preconditioner);
      ++multiplicity;
    }
    if (multiplicity < schwarz.minimum_multiplicity ||
        multiplicity > schwarz.maximum_multiplicity)
      MarkFailure(control, failure_invalid_preconditioner);
  }
  grid.sync();
}

__device__ void ValidateFixedSigmaJacobiRows(
    const NativeTraceElementSchwarzView schwarz,
    const NativeTraceWorkspaceView workspace,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  const int patch_rows = schwarz.patch_rows;
  for (int row = thread; row < schwarz.true_rows; row += stride)
  {
    int sigma_field = -1;
    for (int incidence = 0;
         incidence < native_trace_schwarz_maximum_incidence;
         ++incidence)
    {
      const int slot = schwarz.true_incidence_slots[
          native_trace_schwarz_maximum_incidence * row + incidence];
      if (slot < 0) continue;
      const int local_row = slot % patch_rows;
      const int packed_row = FieldSplitPackedRow(patch_rows, local_row);
      const int incidence_sigma =
          packed_row >= FieldSplitEulerRows(patch_rows) ? 1 : 0;
      if (sigma_field >= 0 && sigma_field != incidence_sigma)
        MarkFailure(control, failure_invalid_preconditioner);
      sigma_field = incidence_sigma;
    }
    if (sigma_field < 0)
      MarkFailure(control, failure_invalid_preconditioner);
    workspace.permutation_marks[row] = sigma_field;
  }
  grid.sync();
}

__device__ double GridDot(const double* left, const double* right,
                          const int size,
                          const NativeTraceWorkspaceView workspace,
                          NativeTraceSolveControl* control,
                          const cg::grid_group& grid,
                          double* block_scratch) noexcept
{
  double local = 0.0;
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int index = thread; index < size; index += stride)
    local += left[index] * right[index];

  block_scratch[threadIdx.x] = local;
  __syncthreads();
  for (int offset = native_trace_cooperative_threads / 2;
       offset > 0; offset /= 2)
  {
    if (static_cast<int>(threadIdx.x) < offset)
      block_scratch[threadIdx.x] += block_scratch[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    workspace.reduction[blockIdx.x] = block_scratch[0];
  grid.sync();

  if (blockIdx.x == 0)
  {
    double block_total = 0.0;
    for (int block = static_cast<int>(threadIdx.x);
         block < static_cast<int>(gridDim.x);
         block += native_trace_cooperative_threads)
      block_total += workspace.reduction[block];
    block_scratch[threadIdx.x] = block_total;
    __syncthreads();
    for (int offset = native_trace_cooperative_threads / 2;
         offset > 0; offset /= 2)
    {
      if (static_cast<int>(threadIdx.x) < offset)
        block_scratch[threadIdx.x] += block_scratch[threadIdx.x + offset];
      __syncthreads();
    }
    if (threadIdx.x == 0) control->reduction_value = block_scratch[0];
  }
  grid.sync();
  return control->reduction_value;
}

__device__ double GridNorm(const double* values, const int size,
                           const NativeTraceWorkspaceView workspace,
                           NativeTraceSolveControl* control,
                           const cg::grid_group& grid,
                           double* block_scratch) noexcept
{
  const double square = GridDot(values, values, size, workspace, control,
                                grid, block_scratch);
  if (!isfinite(square) || square < 0.0)
    MarkFailure(control, failure_numerical_breakdown);
  grid.sync();
  return square >= 0.0 && isfinite(square) ? sqrt(square) : 0.0;
}

__device__ void ScaleVector(double* values, const double scale,
                            const int size,
                            NativeTraceSolveControl* control,
                            const cg::grid_group& grid) noexcept
{
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int index = thread; index < size; index += stride)
  {
    values[index] *= scale;
    if (!isfinite(values[index])) MarkFailure(control, failure_nonfinite);
  }
  grid.sync();
}

__device__ void AxpyVector(double* destination, const double scale,
                           const double* source, const int size,
                           NativeTraceSolveControl* control,
                           const cg::grid_group& grid) noexcept
{
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int index = thread; index < size; index += stride)
  {
    destination[index] += scale * source[index];
    if (!isfinite(destination[index]))
      MarkFailure(control, failure_nonfinite);
  }
  grid.sync();
}

__device__ void GridBasisDotTile(
    const double* basis, const double* values, const int size,
    const int basis_begin, const int basis_count, double* coefficients,
    const NativeTraceWorkspaceView workspace,
    NativeTraceSolveControl* control, const cg::grid_group& grid,
    double* block_scratch) noexcept
{
  double local[native_trace_projection_tile] = {};
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int index = thread; index < size; index += stride)
  {
    const double value = values[index];
#pragma unroll
    for (int offset = 0; offset < native_trace_projection_tile; ++offset)
      if (offset < basis_count)
        local[offset] += basis[
            static_cast<std::size_t>(basis_begin + offset) * size + index] *
            value;
  }

  for (int offset = 0; offset < basis_count; ++offset)
  {
    block_scratch[threadIdx.x] = local[offset];
    __syncthreads();
    for (int reduction_offset = native_trace_cooperative_threads / 2;
         reduction_offset > 0; reduction_offset /= 2)
    {
      if (static_cast<int>(threadIdx.x) < reduction_offset)
        block_scratch[threadIdx.x] +=
            block_scratch[threadIdx.x + reduction_offset];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      workspace.reduction[
          static_cast<std::size_t>(offset) * gridDim.x + blockIdx.x] =
          block_scratch[0];
    __syncthreads();
  }
  grid.sync();

  if (blockIdx.x == 0)
  {
    for (int offset = 0; offset < basis_count; ++offset)
    {
      double block_total = 0.0;
      for (int block = static_cast<int>(threadIdx.x);
           block < static_cast<int>(gridDim.x);
           block += native_trace_cooperative_threads)
        block_total += workspace.reduction[
            static_cast<std::size_t>(offset) * gridDim.x + block];
      block_scratch[threadIdx.x] = block_total;
      __syncthreads();
      for (int reduction_offset = native_trace_cooperative_threads / 2;
           reduction_offset > 0; reduction_offset /= 2)
      {
        if (static_cast<int>(threadIdx.x) < reduction_offset)
          block_scratch[threadIdx.x] +=
              block_scratch[threadIdx.x + reduction_offset];
        __syncthreads();
      }
      if (threadIdx.x == 0)
      {
        const double coefficient = block_scratch[0];
        coefficients[basis_begin + offset] = coefficient;
        if (!isfinite(coefficient))
          MarkFailure(control, failure_numerical_breakdown);
      }
      __syncthreads();
    }
  }
  grid.sync();
}

__device__ void ProjectBasisSpace(
    double* values, const double* basis, const double* coefficients,
    const int basis_count, const int size, double* hessenberg,
    const int hessenberg_stride, const int column, const bool accumulate,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  if (grid.thread_rank() == 0)
  {
    for (int basis_index = 0; basis_index < basis_count; ++basis_index)
    {
      double& entry = hessenberg[
          static_cast<std::size_t>(basis_index) * hessenberg_stride + column];
      entry = accumulate ? entry + coefficients[basis_index]
                         : coefficients[basis_index];
      if (!isfinite(entry))
        MarkFailure(control, failure_numerical_breakdown);
    }
  }

  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int index = thread; index < size; index += stride)
  {
    double projection = 0.0;
    for (int basis_index = 0; basis_index < basis_count; ++basis_index)
      projection += coefficients[basis_index] * basis[
          static_cast<std::size_t>(basis_index) * size + index];
    values[index] -= projection;
    if (!isfinite(values[index])) MarkFailure(control, failure_nonfinite);
  }
  grid.sync();
}

__device__ void FormBasisLinearCombination(
    double* destination, const double* basis, const double* coefficients,
    const int basis_count, const int size,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int index = thread; index < size; index += stride)
  {
    double value = 0.0;
    for (int basis_index = 0; basis_index < basis_count; ++basis_index)
      value += coefficients[basis_index] * basis[
          static_cast<std::size_t>(basis_index) * size + index];
    destination[index] = value;
    if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
  }
  grid.sync();
}

__device__ void SparseMatrixVector(
    const NativeTraceCsrView matrix, const double* input, double* output,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int row = thread; row < matrix.rows; row += stride)
  {
    double value = 0.0;
    for (int entry = matrix.row_offsets[row];
         entry < matrix.row_offsets[row + 1]; ++entry)
      value += matrix.values[entry] * input[matrix.columns[entry]];
    output[row] = value;
    if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
  }
  grid.sync();
}

__device__ void ApplyTraceOperator(
    const NativeTraceSystemView system,
    const NativeTracePreconditionerView preconditioner,
    const double* input, double* output,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  if (system.operator_kind == NativeTraceOperatorKind::assembled_csr_fp64)
  {
    SparseMatrixVector(system.matrix, input, output, control, grid);
    return;
  }
  if (!NativeTraceOperatorIsElementMatrixFree(system.operator_kind))
  {
    MarkFailure(control, failure_invalid_system);
    grid.sync();
    return;
  }

  const NativeTraceElementSchwarzView schwarz =
      preconditioner.element_schwarz;
  const int patch_rows = schwarz.patch_rows;
  const int element_slots = schwarz.elements * patch_rows;
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  if (grid.thread_rank() == 0) ++control->operator_applications;
  grid.sync();

  // Exact FP64 T gather and row-major S_e action. Each element row is owned
  // by one thread and accumulates columns in increasing local order.
  for (int slot = thread; slot < element_slots; slot += stride)
  {
    const int element = slot / patch_rows;
    const int row = slot - element * patch_rows;
    const std::size_t row_offset =
        (static_cast<std::size_t>(element) * patch_rows + row) * patch_rows;
    const std::size_t map_offset =
        static_cast<std::size_t>(element) * patch_rows;
    double value = 0.0;
    for (int column = 0; column < patch_rows; ++column)
    {
      double sign = 1.0;
      const int truth = DecodeSignedDof(
          schwarz.signed_element_true_dofs[map_offset + column], sign);
      value += system.element_schur[row_offset + column] *
          sign * input[truth];
    }
    schwarz.element_solution[slot] = value;
    if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
  }
  grid.sync();

  // Exact FP64 T^T reduction. Static incidence order makes the one/two-term
  // sum deterministic and avoids global atomics.
  for (int row = thread; row < system.matrix.rows; row += stride)
  {
    double value = 0.0;
    for (int incidence = 0;
         incidence < native_trace_schwarz_maximum_incidence; ++incidence)
    {
      const int slot = schwarz.true_incidence_slots[
          native_trace_schwarz_maximum_incidence * row + incidence];
      if (slot < 0) continue;
      double sign = 1.0;
      (void)DecodeSignedDof(
          schwarz.signed_element_true_dofs[slot], sign);
      value += sign * schwarz.element_solution[slot];
    }
    output[row] = value;
    if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
  }
  grid.sync();
}

__device__ void ComputeActualResidual(
    const NativeTraceSystemView system,
    const NativeTracePreconditionerView preconditioner, double* residual,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  ApplyTraceOperator(system, preconditioner, system.solution, residual,
                     control, grid);
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int row = thread; row < system.matrix.rows; row += stride)
  {
    const double value = system.rhs[row] - residual[row];
    residual[row] = value;
    if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
  }
  grid.sync();
}

__device__ void ApplyPreconditioner(
    const NativeTraceSystemView system,
    const NativeTracePreconditionerView preconditioner,
    const NativeTraceWorkspaceView workspace,
    const double* input, double* output,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  const int rows = system.matrix.rows;
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  if (grid.thread_rank() == 0) ++control->preconditioner_applications;
  grid.sync();

  if (preconditioner.kind == NativeTracePreconditionerKind::diagonal)
  {
    for (int row = thread; row < rows; row += stride)
    {
      const double value = input[row] /
          system.matrix.values[system.matrix.row_offsets[row]];
      output[row] = value;
      if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
    }
    grid.sync();
    return;
  }

  if (NativeTracePreconditionerIsElementSchwarz(preconditioner.kind))
  {
    const NativeTraceElementSchwarzView schwarz =
        preconditioner.element_schwarz;
    const bool equilibrated_field_split =
        NativeTracePreconditionerIsEquilibratedFieldSplit(
            preconditioner.kind);
    const bool warp_equilibrated_field_split =
        NativeTracePreconditionerIsWarpEquilibratedFieldSplit(
            preconditioner.kind);
    const int patch_rows = schwarz.patch_rows;
    const int element_slots = schwarz.elements * patch_rows;

    // Gather the complete signed patch RHS before touching output. The final
    // GMRES update intentionally permits input == output.
    for (int slot = thread; slot < element_slots; slot += stride)
    {
      double sign = 1.0;
      const int truth = DecodeSignedDof(
          schwarz.signed_element_true_dofs[slot], sign);
      schwarz.element_rhs[slot] = sign * input[truth];
      if (!equilibrated_field_split)
        schwarz.element_solution[slot] = schwarz.element_rhs[slot];
    }
    grid.sync();

    if (warp_equilibrated_field_split)
    {
      if (patch_rows == 40)
        ApplyWarpEquilibratedFieldSplit<16>(schwarz, control);
      else
        ApplyWarpEquilibratedFieldSplit<32>(schwarz, control);
      grid.sync();
    }

    if (!warp_equilibrated_field_split ||
        control->preconditioner_applications == 1)
      for (int element = thread; element < schwarz.elements;
           element += stride)
    {
      const std::size_t vector_offset =
          static_cast<std::size_t>(element) * patch_rows;
      const std::size_t matrix_offset = vector_offset * patch_rows;
      const double* const matrix =
          schwarz.patch_matrix + matrix_offset;
      const double* const lu = schwarz.patch_lu + matrix_offset;
      const int* const pivots = schwarz.pivots + vector_offset;
      const double* const rhs = schwarz.element_rhs + vector_offset;
      double* const solution = schwarz.element_solution + vector_offset;

      if (equilibrated_field_split && !warp_equilibrated_field_split)
      {
        const int euler_rows = FieldSplitEulerRows(patch_rows);
        const int sigma_rows = patch_rows - euler_rows;
        const double* const euler_lu = lu;
        const double* const sigma_lu =
            lu + FieldSplitSigmaLuOffset(patch_rows);
        const double* const lower_coupling =
            lu + FieldSplitLowerCouplingOffset(patch_rows);
        const double* const left_scale =
            lu + FieldSplitLeftScaleOffset(patch_rows);
        const double* const right_scale =
            lu + FieldSplitRightScaleOffset(patch_rows);

        // Lower multiplicative block factorization
        // [A_UU 0; A_sigma,U A_sigma,sigma].  The retained global operator
        // and GMRES remain unchanged.  Packing is only a private local solve
        // layout; the lower coupling is retained in the same two-sided
        // scaling, and the externally committed trace stays binary64
        // Uhat4/Uhat5.
        for (int packed = 0; packed < euler_rows; ++packed)
        {
          const int original = FieldSplitOriginalRow(patch_rows, packed);
          solution[packed] = left_scale[packed] * rhs[original];
        }
        SolvePivotedDenseLu(
            euler_lu, pivots, euler_rows, solution);

        for (int sigma = 0; sigma < sigma_rows; ++sigma)
        {
          const int packed = euler_rows + sigma;
          const int original_row =
              FieldSplitOriginalRow(patch_rows, packed);
          double value = left_scale[packed] * rhs[original_row];
          for (int euler = 0; euler < euler_rows; ++euler)
            value -= lower_coupling[sigma * euler_rows + euler] *
                solution[euler];
          solution[packed] = value;
        }
        for (int packed = 0; packed < euler_rows; ++packed)
          solution[packed] *= right_scale[packed];
        SolvePivotedDenseLu(
            sigma_lu, pivots + euler_rows, sigma_rows,
            solution + euler_rows);
        for (int sigma = 0; sigma < sigma_rows; ++sigma)
        {
          const int packed = euler_rows + sigma;
          solution[packed] *= right_scale[packed];
        }
      }
      else if (!warp_equilibrated_field_split)
        SolvePivotedDenseLu(lu, pivots, patch_rows, solution);

      // Validate the first actual application of each native solve against
      // the retained unfactored patch. The synchronized aggregate counter
      // makes this uniform across the cooperative grid; later applications
      // avoid recurring dense validation traffic.
      if (control->preconditioner_applications == 1)
      {
        if (equilibrated_field_split)
        {
          const int euler_rows = FieldSplitEulerRows(patch_rows);
          const int sigma_rows = patch_rows - euler_rows;
          double euler_matrix_norm = 0.0;
          double euler_solution_norm = 0.0;
          double euler_rhs_norm = 0.0;
          double euler_residual_norm = 0.0;
          for (int euler_row = 0; euler_row < euler_rows; ++euler_row)
          {
            const int original_row =
                FieldSplitOriginalRow(patch_rows, euler_row);
            double row_norm = 0.0;
            double residual = -rhs[original_row];
            euler_solution_norm = fmax(
                euler_solution_norm, fabs(solution[euler_row]));
            euler_rhs_norm = fmax(euler_rhs_norm, fabs(rhs[original_row]));
            for (int euler_column = 0; euler_column < euler_rows;
                 ++euler_column)
            {
              const int original_column =
                  FieldSplitOriginalRow(patch_rows, euler_column);
              const double coefficient =
                  matrix[original_row * patch_rows + original_column];
              row_norm += fabs(coefficient);
              residual += coefficient * solution[euler_column];
            }
            euler_matrix_norm = fmax(euler_matrix_norm, row_norm);
            euler_residual_norm = fmax(
                euler_residual_norm, fabs(residual));
          }

          double sigma_matrix_norm = 0.0;
          double sigma_solution_norm = 0.0;
          double sigma_rhs_norm = 0.0;
          double sigma_residual_norm = 0.0;
          for (int sigma_row = 0; sigma_row < sigma_rows; ++sigma_row)
          {
            const int packed_row = euler_rows + sigma_row;
            const int original_row =
                FieldSplitOriginalRow(patch_rows, packed_row);
            double adjusted_rhs = rhs[original_row];
            for (int euler_column = 0; euler_column < euler_rows;
                 ++euler_column)
            {
              const int original_column =
                  FieldSplitOriginalRow(patch_rows, euler_column);
              adjusted_rhs -=
                  matrix[original_row * patch_rows + original_column] *
                  solution[euler_column];
            }
            double row_norm = 0.0;
            double residual = -adjusted_rhs;
            sigma_solution_norm = fmax(
                sigma_solution_norm, fabs(solution[packed_row]));
            sigma_rhs_norm = fmax(sigma_rhs_norm, fabs(adjusted_rhs));
            for (int sigma_column = 0; sigma_column < sigma_rows;
                 ++sigma_column)
            {
              const int packed_column = euler_rows + sigma_column;
              const int original_column =
                  FieldSplitOriginalRow(patch_rows, packed_column);
              const double coefficient =
                  matrix[original_row * patch_rows + original_column];
              row_norm += fabs(coefficient);
              residual += coefficient * solution[packed_column];
            }
            sigma_matrix_norm = fmax(sigma_matrix_norm, row_norm);
            sigma_residual_norm = fmax(
                sigma_residual_norm, fabs(residual));
          }
          const double euler_backward_error = euler_residual_norm /
              fmax(euler_matrix_norm * euler_solution_norm + euler_rhs_norm,
                   0x1p-1022);
          const double sigma_backward_error = sigma_residual_norm /
              fmax(sigma_matrix_norm * sigma_solution_norm + sigma_rhs_norm,
                   0x1p-1022);
          const double backward_error_limit =
              4096.0 * 2.2204460492503131e-16 * patch_rows;
          if (!isfinite(euler_backward_error) ||
              !isfinite(sigma_backward_error) ||
              euler_backward_error > backward_error_limit ||
              sigma_backward_error > backward_error_limit)
            MarkFailure(control, failure_invalid_preconditioner);
        }
        else
        {
          double matrix_norm = 0.0;
          double solution_norm = 0.0;
          double rhs_norm = 0.0;
          double residual_norm = 0.0;
          for (int row = 0; row < patch_rows; ++row)
          {
            double row_norm = 0.0;
            double residual = -rhs[row];
            solution_norm = fmax(solution_norm, fabs(solution[row]));
            rhs_norm = fmax(rhs_norm, fabs(rhs[row]));
            for (int column = 0; column < patch_rows; ++column)
            {
              const double coefficient = matrix[row * patch_rows + column];
              row_norm += fabs(coefficient);
              residual += coefficient * solution[column];
            }
            matrix_norm = fmax(matrix_norm, row_norm);
            residual_norm = fmax(residual_norm, fabs(residual));
          }
          const double denominator =
              fmax(matrix_norm * solution_norm + rhs_norm, 0x1p-1022);
          const double backward_error = residual_norm / denominator;
          const double backward_error_limit =
              4096.0 * 2.2204460492503131e-16 * patch_rows;
          if (!isfinite(backward_error) ||
              backward_error > backward_error_limit)
            MarkFailure(control, failure_invalid_preconditioner);
        }
      }
    }
    grid.sync();
    if (control->failure_mask != 0) return;

    // One thread owns one true row and reduces its static one/two element
    // incidences in fixed order. This is deterministic inverse-multiplicity
    // weighting and uses no global atomics.
    for (int row = thread; row < rows; row += stride)
    {
      double value = 0.0;
      int multiplicity = 0;
      for (int incidence = 0;
           incidence < native_trace_schwarz_maximum_incidence; ++incidence)
      {
        const int slot = schwarz.true_incidence_slots[
            native_trace_schwarz_maximum_incidence * row + incidence];
        if (slot < 0) continue;
        double sign = 1.0;
        (void)DecodeSignedDof(
            schwarz.signed_element_true_dofs[slot], sign);
        const int solution_slot = equilibrated_field_split
            ? (slot / patch_rows) * patch_rows +
                FieldSplitPackedRow(patch_rows, slot % patch_rows)
            : slot;
        value += sign * schwarz.element_solution[solution_slot];
        ++multiplicity;
      }
      output[row] = value / static_cast<double>(multiplicity);
      if (!isfinite(output[row])) MarkFailure(control, failure_nonfinite);
    }
    grid.sync();
    return;
  }

  const NativeTraceIlu0View ilu0 = preconditioner.ilu0;
  double* const work = workspace.vectors;
  double* const permuted = workspace.vectors + rows;
  double* const lower = workspace.vectors +
      2 * static_cast<std::size_t>(rows);

  // HYPRE gather: permuted[new] = original[permutation[new]].
  for (int row = thread; row < rows; row += stride)
    permuted[row] = input[ilu0.permutation[row]];
  grid.sync();

  // Zero-start Jacobi L solve.  The first iteration is the identity transform.
  for (int row = thread; row < rows; row += stride)
    lower[row] = permuted[row];
  grid.sync();
  for (int iteration = 1;
       iteration < ilu0.lower_jacobi_iterations; ++iteration)
  {
    for (int row = thread; row < rows; row += stride)
    {
      double value = 0.0;
      for (int entry = ilu0.factor.row_offsets[row];
           entry < ilu0.factor.row_offsets[row + 1]; ++entry)
      {
        const int column = ilu0.factor.columns[entry];
        if (column < row)
          value += ilu0.factor.values[entry] * lower[column];
      }
      work[row] = value;
    }
    grid.sync();
    for (int row = thread; row < rows; row += stride)
    {
      lower[row] = permuted[row] - work[row];
      if (!isfinite(lower[row])) MarkFailure(control, failure_nonfinite);
    }
    grid.sync();
  }

  // Zero-start Jacobi U solve.  inverse_diagonal is HYPRE extraction type 2.
  for (int row = thread; row < rows; row += stride)
    permuted[row] = ilu0.inverse_diagonal[row] * lower[row];
  grid.sync();
  for (int iteration = 1;
       iteration < ilu0.upper_jacobi_iterations; ++iteration)
  {
    for (int row = thread; row < rows; row += stride)
    {
      double value = 0.0;
      for (int entry = ilu0.factor.row_offsets[row];
           entry < ilu0.factor.row_offsets[row + 1]; ++entry)
      {
        const int column = ilu0.factor.columns[entry];
        if (column > row)
          value += ilu0.factor.values[entry] * permuted[column];
      }
      work[row] = value;
    }
    grid.sync();
    for (int row = thread; row < rows; row += stride)
    {
      permuted[row] = ilu0.inverse_diagonal[row] *
          (lower[row] - work[row]);
      if (!isfinite(permuted[row])) MarkFailure(control, failure_nonfinite);
    }
    grid.sync();
  }

  // HYPRE scatter: original[permutation[new]] = permuted[new].
  for (int row = thread; row < rows; row += stride)
    output[ilu0.permutation[row]] = permuted[row];
  grid.sync();
}

__device__ void ApplyFixedSigmaJacobiPreconditioner(
    const NativeTraceSystemView system,
    const NativeTracePreconditionerView preconditioner,
    const NativeTraceWorkspaceView workspace,
    const double* input, double* output,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  ApplyPreconditioner(system, preconditioner, workspace, input, output,
                      control, grid);
  if (control->failure_mask != 0) return;

  const NativeTraceElementSchwarzView schwarz =
      preconditioner.element_schwarz;
  const int rows = system.matrix.rows;
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();

  // The exact local lower field split above supplies a nonzero stationary
  // predictor. Refine only the scalar global sigma rows with three fixed
  // damped Jacobi corrections. Euler rows remain fixed, so this is a lower
  // block-GS preconditioner refinement rather than a replacement for the
  // coupled U7/Uhat5 system. The authoritative outer GMRES and
  // application-level FP64 true-residual rebuild remain unchanged.
  double* const next_sigma = workspace.vectors + rows;
  for (int sweep = 0; sweep < native_trace_sigma_jacobi_sweeps; ++sweep)
  {
    for (int row = thread; row < rows; row += stride)
    {
      if (workspace.permutation_marks[row] != 1) continue;
      int rhs_slot = -1;
      for (int incidence = 0;
           incidence < native_trace_schwarz_maximum_incidence;
           ++incidence)
      {
        const int slot = schwarz.true_incidence_slots[
            native_trace_schwarz_maximum_incidence * row + incidence];
        if (slot >= 0)
        {
          rhs_slot = slot;
          break;
        }
      }
      if (rhs_slot < 0)
      {
        MarkFailure(control, failure_invalid_preconditioner);
        continue;
      }
      double rhs_sign = 1.0;
      const int rhs_truth = DecodeSignedDof(
          schwarz.signed_element_true_dofs[rhs_slot], rhs_sign);
      if (rhs_truth != row)
      {
        MarkFailure(control, failure_invalid_preconditioner);
        continue;
      }
      const double rhs_value =
          rhs_sign * schwarz.element_rhs[rhs_slot];
      const int begin = system.matrix.row_offsets[row];
      const int end = system.matrix.row_offsets[row + 1];
      double action = 0.0;
      for (int entry = begin; entry < end; ++entry)
        action += system.matrix.values[entry] *
            output[system.matrix.columns[entry]];
      const double diagonal = system.matrix.values[begin];
      const double value = output[row] +
          native_trace_sigma_jacobi_damping *
              (rhs_value - action) / diagonal;
      next_sigma[row] = value;
      if (!isfinite(value)) MarkFailure(control, failure_nonfinite);
    }
    grid.sync();
    if (control->failure_mask != 0) return;
    for (int row = thread; row < rows; row += stride)
      if (workspace.permutation_marks[row] == 1)
        output[row] = next_sigma[row];
    grid.sync();
  }
}

__device__ void SetFinalResidualMetadata(
    NativeTraceSolveControl* control, const double residual) noexcept
{
  control->actual_residual_norm = residual;
  control->relative_residual_norm = control->rhs_norm > 0.0
      ? residual / control->rhs_norm : residual;
}

template <bool UseCgs2, bool FixedSigmaJacobi>
__global__ void NativeRankOneTraceSolveKernel(
    const NativeTraceSystemView system,
    const NativeTracePreconditionerView preconditioner,
    const NativeTraceWorkspaceView workspace,
    const NativeTraceSolvePolicy policy,
    const NativeTraceCooperativeSupport support)
{
  cg::grid_group grid = cg::this_grid();
  __shared__ double block_scratch[native_trace_cooperative_threads];
  NativeTraceSolveControl* const control = workspace.device_control;
  const int rows = system.matrix.rows;
  const int krylov_dimension = policy.krylov_dimension;
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();

  if (grid.thread_rank() == 0)
  {
    control->schema = native_trace_control_schema;
    control->status = NativeTraceSolveStatus::running;
    control->failure_mask = 0;
    control->iterations = 0;
    control->restart_cycles = 0;
    control->preconditioner_applications = 0;
    control->operator_applications = 0;
    control->actual_residual_checks = 0;
    control->rhs_norm = 0.0;
    control->initial_residual_norm = 0.0;
    control->estimated_residual_norm = 0.0;
    control->actual_residual_norm = 0.0;
    control->relative_residual_norm = 0.0;
    control->reduction_value = 0.0;
  }
  grid.sync();

  if (grid.thread_rank() == 0)
  {
    if (policy.rank_count != 1 || krylov_dimension <= 0 ||
        policy.maximum_iterations <= 0 ||
        !(policy.relative_tolerance > 0.0) ||
        !isfinite(policy.relative_tolerance) ||
        policy.absolute_tolerance < 0.0 ||
        !isfinite(policy.absolute_tolerance))
      MarkFailure(control, failure_invalid_policy);
    if (support.threads_per_block != native_trace_cooperative_threads ||
        support.cooperative_blocks != static_cast<int>(gridDim.x) ||
        !support.supported)
      MarkFailure(control, failure_invalid_workspace);
  }

  ValidateCsr(system.matrix, true, true, failure_invalid_system, control);
  for (int row = thread; row < rows; row += stride)
    if (!isfinite(system.rhs[row])) MarkFailure(control, failure_nonfinite);

  if (preconditioner.kind ==
      NativeTracePreconditionerKind::ilu0_rcm_jacobi)
  {
    ValidateCsr(preconditioner.ilu0.factor, false, true,
                failure_invalid_preconditioner, control);
    for (int row = thread; row < rows; row += stride)
    {
      const double inverse = preconditioner.ilu0.inverse_diagonal[row];
      if (!isfinite(inverse)) MarkFailure(control, failure_nonfinite);
      if (inverse == 0.0)
        MarkFailure(control, failure_invalid_preconditioner);
    }
    ValidatePermutation(preconditioner.ilu0, workspace, control, grid);
  }
  else if (NativeTracePreconditionerIsElementSchwarz(
               preconditioner.kind))
  {
    ValidateElementSchwarz(
        preconditioner.element_schwarz,
        NativeTracePreconditionerIsEquilibratedFieldSplit(
            preconditioner.kind),
        NativeTracePreconditionerIsWarpEquilibratedFieldSplit(
            preconditioner.kind),
        control, grid);
    if constexpr (FixedSigmaJacobi)
    {
      if (!NativeTracePreconditionerIsEquilibratedFieldSplit(
              preconditioner.kind))
      {
        if (grid.thread_rank() == 0)
          MarkFailure(control, failure_invalid_preconditioner);
        grid.sync();
      }
      else
        ValidateFixedSigmaJacobiRows(
            preconditioner.element_schwarz, workspace, control, grid);
    }
  }
  if (NativeTraceOperatorIsElementMatrixFree(system.operator_kind))
  {
    const NativeTraceElementSchwarzView schwarz =
        preconditioner.element_schwarz;
    const int entries = schwarz.elements * schwarz.patch_rows *
        schwarz.patch_rows;
    if (!NativeTraceOperatorMatchesPreconditioner(
            system.operator_kind, preconditioner.kind) ||
        system.element_schur == nullptr)
      MarkFailure(control, failure_invalid_system);
    else
      for (int entry = thread; entry < entries; entry += stride)
        if (!isfinite(system.element_schur[entry]))
          MarkFailure(control, failure_nonfinite);
  }
  else if (system.operator_kind !=
           NativeTraceOperatorKind::assembled_csr_fp64)
    MarkFailure(control, failure_invalid_system);
  grid.sync();
  if (FinishOnFailure(grid, control)) return;

  double* const hessenberg = workspace.small;
  double* const cosine = hessenberg +
      static_cast<std::size_t>(krylov_dimension + 1) * krylov_dimension;
  double* const sine = cosine + krylov_dimension;
  double* const residual_scalars = sine + krylov_dimension;
  double* const work = workspace.vectors;
  double* const first_basis = workspace.basis;

  for (int row = thread; row < rows; row += stride)
  {
    system.solution[row] = 0.0;
    first_basis[row] = system.rhs[row];
  }
  grid.sync();

  double rhs_norm = GridNorm(system.rhs, rows, workspace, control,
                             grid, block_scratch);
  double residual_norm = GridNorm(first_basis, rows, workspace, control,
                                  grid, block_scratch);
  if (grid.thread_rank() == 0)
  {
    control->rhs_norm = rhs_norm;
    control->initial_residual_norm = residual_norm;
    control->estimated_residual_norm = residual_norm;
    control->actual_residual_checks = 1;
    SetFinalResidualMetadata(control, residual_norm);
  }
  grid.sync();
  if (FinishOnFailure(grid, control)) return;

  const double denominator = rhs_norm > 0.0 ? rhs_norm : residual_norm;
  const double epsilon = fmax(policy.absolute_tolerance,
                              policy.relative_tolerance * denominator);
  if (!isfinite(epsilon)) MarkFailure(control, failure_invalid_policy);
  grid.sync();
  if (FinishOnFailure(grid, control)) return;

  if (residual_norm == 0.0)
  {
    if (grid.thread_rank() == 0)
      control->status = NativeTraceSolveStatus::converged;
    return;
  }

  int iteration = 0;
  double real_residual_norm_old = rhs_norm;
  while (iteration < policy.maximum_iterations)
  {
    if (grid.thread_rank() == 0)
    {
      ++control->restart_cycles;
      residual_scalars[0] = residual_norm;
    }
    grid.sync();

    // Pinned HYPRE performs a real-residual check before normalizing an
    // already-small restart residual.
    if (residual_norm <= epsilon)
    {
      ComputeActualResidual(system, preconditioner, first_basis, control,
                            grid);
      const double actual = GridNorm(first_basis, rows, workspace, control,
                                     grid, block_scratch);
      if (grid.thread_rank() == 0)
      {
        ++control->actual_residual_checks;
        SetFinalResidualMetadata(control, actual);
      }
      grid.sync();
      if (FinishOnFailure(grid, control)) return;
      if (actual <= epsilon)
      {
        if (grid.thread_rank() == 0)
        {
          control->estimated_residual_norm = residual_norm;
          control->status = NativeTraceSolveStatus::converged;
        }
        return;
      }
      residual_norm = actual;
    }

    ScaleVector(first_basis, 1.0 / residual_norm, rows, control, grid);
    if (FinishOnFailure(grid, control)) return;
    int restart_size = 0;
    while (restart_size < krylov_dimension &&
           iteration < policy.maximum_iterations)
    {
      const int column = restart_size;
      ++restart_size;
      ++iteration;
      double* const current_basis = workspace.basis +
          static_cast<std::size_t>(column) * rows;
      double* const next_basis = workspace.basis +
          static_cast<std::size_t>(column + 1) * rows;

      if constexpr (FixedSigmaJacobi)
        ApplyFixedSigmaJacobiPreconditioner(
            system, preconditioner, workspace, current_basis, work,
            control, grid);
      else
        ApplyPreconditioner(system, preconditioner, workspace,
                            current_basis, work, control, grid);
      if (FinishOnFailure(grid, control)) return;
      ApplyTraceOperator(system, preconditioner, work, next_basis, control,
                         grid);
      if (FinishOnFailure(grid, control)) return;

      if constexpr (UseCgs2)
      {
        const std::size_t projection_partial_doubles =
            static_cast<std::size_t>(
                krylov_dimension < native_trace_projection_tile
                    ? krylov_dimension : native_trace_projection_tile) *
            gridDim.x;
        double* const projection_coefficients =
            workspace.reduction + projection_partial_doubles;
        // Deterministic FP64 CGS2 for the long ILU-preconditioned IGR solve.
        // Each bounded tile shares one pair of grid barriers for all its dot
        // products, then each pass applies the complete basis projection in
        // one vector traversal.
        for (int basis_begin = 0; basis_begin < restart_size;
             basis_begin += native_trace_projection_tile)
        {
          const int basis_count =
              restart_size - basis_begin < native_trace_projection_tile
                  ? restart_size - basis_begin
                  : native_trace_projection_tile;
          GridBasisDotTile(workspace.basis, next_basis, rows, basis_begin,
                           basis_count, projection_coefficients, workspace,
                           control, grid, block_scratch);
        }
        ProjectBasisSpace(next_basis, workspace.basis,
                          projection_coefficients, restart_size, rows,
                          hessenberg, krylov_dimension, column, false,
                          control, grid);
        if (FinishOnFailure(grid, control)) return;
        for (int basis_begin = 0; basis_begin < restart_size;
             basis_begin += native_trace_projection_tile)
        {
          const int basis_count =
              restart_size - basis_begin < native_trace_projection_tile
                  ? restart_size - basis_begin
                  : native_trace_projection_tile;
          GridBasisDotTile(workspace.basis, next_basis, rows, basis_begin,
                           basis_count, projection_coefficients, workspace,
                           control, grid, block_scratch);
        }
        ProjectBasisSpace(next_basis, workspace.basis,
                          projection_coefficients, restart_size, rows,
                          hessenberg, krylov_dimension, column, true,
                          control, grid);
        if (FinishOnFailure(grid, control)) return;
      }
      else
      {
        // The conventional diagonal-preconditioned solve converges in very
        // few iterations.  Preserve the lower-register pinned-HYPRE MGS path;
        // CGS2's extra pass and register footprint regress this short solve.
        for (int basis_index = 0; basis_index < restart_size; ++basis_index)
        {
          const double* const basis = workspace.basis +
              static_cast<std::size_t>(basis_index) * rows;
          const double inner = GridDot(basis, next_basis, rows, workspace,
                                       control, grid, block_scratch);
          if (!isfinite(inner))
            MarkFailure(control, failure_numerical_breakdown);
          if (grid.thread_rank() == 0)
            hessenberg[static_cast<std::size_t>(basis_index) *
                krylov_dimension + column] = inner;
          grid.sync();
          if (FinishOnFailure(grid, control)) return;
          AxpyVector(next_basis, -inner, basis, rows, control, grid);
        }
      }

      const double next_norm = GridNorm(next_basis, rows, workspace, control,
                                        grid, block_scratch);
      if (grid.thread_rank() == 0)
        hessenberg[static_cast<std::size_t>(restart_size) *
            krylov_dimension + column] = next_norm;
      grid.sync();
      if (FinishOnFailure(grid, control)) return;
      if (next_norm != 0.0)
      {
        ScaleVector(next_basis, 1.0 / next_norm, rows, control, grid);
        if (FinishOnFailure(grid, control)) return;
      }

      if (grid.thread_rank() == 0)
      {
        for (int rotation = 1; rotation < restart_size; ++rotation)
        {
          const std::size_t upper_index =
              static_cast<std::size_t>(rotation - 1) *
                  krylov_dimension + column;
          const std::size_t lower_index =
              static_cast<std::size_t>(rotation) *
                  krylov_dimension + column;
          const double upper = hessenberg[upper_index];
          hessenberg[upper_index] = sine[rotation - 1] *
              hessenberg[lower_index] + cosine[rotation - 1] * upper;
          hessenberg[lower_index] = -sine[rotation - 1] * upper +
              cosine[rotation - 1] * hessenberg[lower_index];
        }
        const std::size_t diagonal_index =
            static_cast<std::size_t>(column) * krylov_dimension + column;
        const std::size_t subdiagonal_index =
            static_cast<std::size_t>(column + 1) *
                krylov_dimension + column;
        double gamma = sqrt(hessenberg[subdiagonal_index] *
                                hessenberg[subdiagonal_index] +
                            hessenberg[diagonal_index] *
                                hessenberg[diagonal_index]);
        if (gamma == 0.0) gamma = 1.0e-16;
        if (!isfinite(gamma))
          MarkFailure(control, failure_numerical_breakdown);
        else
        {
          cosine[column] = hessenberg[diagonal_index] / gamma;
          sine[column] = hessenberg[subdiagonal_index] / gamma;
          residual_scalars[column + 1] =
              -hessenberg[subdiagonal_index] *
              residual_scalars[column] / gamma;
          residual_scalars[column] =
              cosine[column] * residual_scalars[column];
          hessenberg[diagonal_index] =
              sine[column] * hessenberg[subdiagonal_index] +
              cosine[column] * hessenberg[diagonal_index];
          residual_norm = fabs(residual_scalars[column + 1]);
          if (!isfinite(cosine[column]) || !isfinite(sine[column]) ||
              !isfinite(residual_scalars[column]) ||
              !isfinite(residual_scalars[column + 1]) ||
              !isfinite(hessenberg[diagonal_index]) ||
              !isfinite(residual_norm))
            MarkFailure(control, failure_numerical_breakdown);
          else
          {
            control->iterations = iteration;
            control->estimated_residual_norm = residual_norm;
          }
        }
      }
      grid.sync();
      if (FinishOnFailure(grid, control)) return;
      residual_norm = control->estimated_residual_norm;
      if (residual_norm <= epsilon) break;
    }

    // Solve the small upper-triangular system in exactly HYPRE's order.
    if (grid.thread_rank() == 0)
    {
      const int last = restart_size - 1;
      const double last_diagonal =
          hessenberg[static_cast<std::size_t>(last) *
              krylov_dimension + last];
      if (last_diagonal == 0.0 || !isfinite(last_diagonal))
        MarkFailure(control, failure_numerical_breakdown);
      else
      {
        residual_scalars[last] /= last_diagonal;
        if (!isfinite(residual_scalars[last]))
          MarkFailure(control, failure_numerical_breakdown);
        for (int row = last - 1;
             row >= 0 && control->failure_mask == 0; --row)
        {
          double value = 0.0;
          for (int column = row + 1; column < restart_size; ++column)
            value -= hessenberg[static_cast<std::size_t>(row) *
                krylov_dimension + column] * residual_scalars[column];
          value += residual_scalars[row];
          const double diagonal = hessenberg[
              static_cast<std::size_t>(row) * krylov_dimension + row];
          if (diagonal == 0.0 || !isfinite(diagonal))
          {
            MarkFailure(control, failure_numerical_breakdown);
            break;
          }
          residual_scalars[row] = value / diagonal;
          if (!isfinite(residual_scalars[row]))
          {
            MarkFailure(control, failure_numerical_breakdown);
            break;
          }
        }
      }
    }
    grid.sync();
    if (FinishOnFailure(grid, control)) return;

    if constexpr (UseCgs2)
      FormBasisLinearCombination(work, workspace.basis, residual_scalars,
                                 restart_size, rows, control, grid);
    else
    {
      const double* const last_basis = workspace.basis +
          static_cast<std::size_t>(restart_size - 1) * rows;
      for (int row = thread; row < rows; row += stride)
        work[row] = residual_scalars[restart_size - 1] * last_basis[row];
      grid.sync();
      for (int basis_index = restart_size - 2;
           basis_index >= 0; --basis_index)
      {
        const double* const basis = workspace.basis +
            static_cast<std::size_t>(basis_index) * rows;
        AxpyVector(work, residual_scalars[basis_index], basis, rows,
                   control, grid);
      }
    }
    if constexpr (FixedSigmaJacobi)
      ApplyFixedSigmaJacobiPreconditioner(
          system, preconditioner, workspace, work, work, control, grid);
    else
      ApplyPreconditioner(system, preconditioner, workspace,
                          work, work, control, grid);
    if (FinishOnFailure(grid, control)) return;
    AxpyVector(system.solution, 1.0, work, rows, control, grid);
    if (FinishOnFailure(grid, control)) return;

    bool false_convergence_restart = false;
    if (residual_norm <= epsilon)
    {
      ComputeActualResidual(system, preconditioner, first_basis, control,
                            grid);
      const double actual = GridNorm(first_basis, rows, workspace, control,
                                     grid, block_scratch);
      if (grid.thread_rank() == 0)
      {
        ++control->actual_residual_checks;
        SetFinalResidualMetadata(control, actual);
      }
      grid.sync();
      if (FinishOnFailure(grid, control)) return;
      if (actual <= epsilon)
      {
        if (grid.thread_rank() == 0)
          control->status = NativeTraceSolveStatus::converged;
        return;
      }
      if (actual >= real_residual_norm_old)
      {
        if (grid.thread_rank() == 0)
          control->status =
              NativeTraceSolveStatus::bounded_real_residual_stagnation;
        return;
      }
      residual_norm = actual;
      real_residual_norm_old = actual;
      false_convergence_restart = true;
    }

    if (!false_convergence_restart)
    {
      // Reconstruct HYPRE's restart residual from the Givens rotations.  This
      // deliberately avoids an extra A*x residual evaluation.
      if (grid.thread_rank() == 0)
      {
        for (int index = restart_size; index > 0; --index)
        {
          residual_scalars[index - 1] =
              -sine[index - 1] * residual_scalars[index];
          residual_scalars[index] =
              cosine[index - 1] * residual_scalars[index];
        }
      }
      grid.sync();
      if constexpr (UseCgs2)
        FormBasisLinearCombination(first_basis, workspace.basis,
                                   residual_scalars, restart_size + 1,
                                   rows, control, grid);
      else
      {
        double* const last_restart_basis = workspace.basis +
            static_cast<std::size_t>(restart_size) * rows;
        AxpyVector(last_restart_basis,
                   residual_scalars[restart_size] - 1.0,
                   last_restart_basis, rows, control, grid);
        for (int basis_index = restart_size - 1;
             basis_index > 0; --basis_index)
        {
          const double* const basis = workspace.basis +
              static_cast<std::size_t>(basis_index) * rows;
          AxpyVector(last_restart_basis, residual_scalars[basis_index],
                     basis, rows, control, grid);
        }
        AxpyVector(first_basis, residual_scalars[0] - 1.0,
                   first_basis, rows, control, grid);
        AxpyVector(first_basis, 1.0, last_restart_basis, rows, control, grid);
      }
      if (FinishOnFailure(grid, control)) return;
    }
  }

  // Match the established bounded HYPRE lifecycle: publish one final actual
  // residual for a maximum-iteration exit, then let the application rebuild
  // the authoritative FP64 true residual from the current matrix.
  ComputeActualResidual(system, preconditioner, first_basis, control, grid);
  const double actual = GridNorm(first_basis, rows, workspace, control,
                                 grid, block_scratch);
  if (grid.thread_rank() == 0)
  {
    ++control->actual_residual_checks;
    SetFinalResidualMetadata(control, actual);
  }
  grid.sync();
  if (FinishOnFailure(grid, control)) return;
  if (grid.thread_rank() == 0)
  {
    control->iterations = iteration;
    control->estimated_residual_norm = residual_norm;
    control->status = NativeTraceSolveStatus::maximum_iterations;
  }
}

} // namespace

extern "C" NativeTraceWorkspaceRequirements
QueryNativeTraceWorkspaceRequirements(const int rows,
                                      const int krylov_dimension,
                                      const int cooperative_blocks,
                                      const bool igr_active) noexcept
{
  NativeTraceWorkspaceRequirements result{};
  if (rows <= 0 || krylov_dimension <= 0 ||
      krylov_dimension == std::numeric_limits<int>::max() ||
      cooperative_blocks <= 0)
    return result;

  const std::size_t row_count = static_cast<std::size_t>(rows);
  const std::size_t dimension = static_cast<std::size_t>(krylov_dimension);
  std::size_t dimension_plus_one = 0;
  std::size_t hessenberg = 0;
  std::size_t scalar_vectors = 0;
  std::size_t small = 0;
  if (!CheckedAdd(dimension, 1, dimension_plus_one) ||
      !CheckedMultiply(row_count, dimension_plus_one,
                       result.basis_doubles) ||
      !CheckedMultiply(row_count, igr_active ? 3U : 1U,
                       result.vector_doubles) ||
      !CheckedMultiply(dimension_plus_one, dimension, hessenberg) ||
      !CheckedMultiply(dimension, 2, scalar_vectors) ||
      !CheckedAdd(scalar_vectors, dimension_plus_one, scalar_vectors) ||
      !CheckedAdd(hessenberg, scalar_vectors, small))
    return NativeTraceWorkspaceRequirements{};

  if (igr_active)
  {
    std::size_t projection_partials = 0;
    const std::size_t projection_tile = std::min(
        dimension, static_cast<std::size_t>(native_trace_projection_tile));
    if (!CheckedMultiply(projection_tile,
                         static_cast<std::size_t>(cooperative_blocks),
                         projection_partials) ||
        !CheckedAdd(projection_partials, dimension,
                    result.reduction_doubles))
      return NativeTraceWorkspaceRequirements{};
  }
  else
    result.reduction_doubles = static_cast<std::size_t>(cooperative_blocks);
  result.small_doubles = small;
  result.permutation_integers = igr_active ? row_count : 0;
  result.control_double_slots = NativeTraceControlDoubleSlots();
  result.valid = true;
  return result;
}

extern "C" NativeTraceCooperativeSupport
QueryNativeRankOneTraceCooperativeSupport(const int rows,
                                          const int krylov_dimension,
                                          const bool igr_active,
                                          const bool fixed_sigma_jacobi) noexcept
{
  NativeTraceCooperativeSupport result{};
  if (rows <= 0 || krylov_dimension <= 0 ||
      krylov_dimension == std::numeric_limits<int>::max() ||
      (fixed_sigma_jacobi && !igr_active))
    return result;

  int device = -1;
  int cooperative = 0;
  int major = 0;
  int minor = 0;
  int multiprocessors = 0;
  int active_blocks = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess)
    return NativeTraceCooperativeSupport{};

  const cudaError_t occupancy = fixed_sigma_jacobi
      ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, NativeRankOneTraceSolveKernel<true, true>,
            native_trace_cooperative_threads, 0)
      : igr_active
          ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks, NativeRankOneTraceSolveKernel<true, false>,
                native_trace_cooperative_threads, 0)
          : cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks, NativeRankOneTraceSolveKernel<false, false>,
                native_trace_cooperative_threads, 0);
  if (occupancy != cudaSuccess) return NativeTraceCooperativeSupport{};

  // This leaf is an SM120 tensor-backend implementation, not a generic CUDA
  // fallback.  A different device is rejected rather than silently rerouted.
  if (cooperative == 0 || major != 12 || minor != 0 ||
      multiprocessors <= 0 || active_blocks <= 0)
    return result;

  const long long resident_capacity =
      static_cast<long long>(multiprocessors) * active_blocks;
  const long long useful_blocks =
      (static_cast<long long>(rows) + native_trace_cooperative_threads - 1) /
      native_trace_cooperative_threads;
  const long long launch_blocks =
      useful_blocks < resident_capacity ? useful_blocks : resident_capacity;
  if (launch_blocks <= 0 ||
      launch_blocks > std::numeric_limits<int>::max())
    return result;

  result.supported = true;
  result.device = device;
  result.compute_major = major;
  result.compute_minor = minor;
  result.multiprocessors = multiprocessors;
  result.threads_per_block = native_trace_cooperative_threads;
  result.maximum_active_blocks_per_multiprocessor = active_blocks;
  result.cooperative_blocks = static_cast<int>(launch_blocks);
  return result;
}

extern "C" bool LaunchNativeRankOneTraceSolve(
    const NativeTraceSystemView& system,
    const NativeTracePreconditionerView& preconditioner,
    const NativeTraceWorkspaceView& workspace,
    const NativeTraceSolvePolicy& policy,
    const NativeTraceCooperativeSupport& support,
    const cudaStream_t stream) noexcept
{
  const bool use_fixed_sigma_jacobi =
      NativeTracePreconditionerUsesFixedSigmaJacobi(preconditioner.kind);
  const NativeTracePreconditionerKind kernel_preconditioner_kind =
      NativeTraceFixedSigmaJacobiBasePreconditioner(preconditioner.kind);
  const bool use_ilu = kernel_preconditioner_kind ==
      NativeTracePreconditionerKind::ilu0_rcm_jacobi;
  const bool use_element_schwarz =
      NativeTracePreconditionerIsElementSchwarz(kernel_preconditioner_kind);
  const bool use_matrix_free =
      NativeTraceOperatorIsElementMatrixFree(system.operator_kind);
  const bool use_cgs2 = use_ilu || use_element_schwarz;
  const NativeTraceWorkspaceRequirements requirements =
      QueryNativeTraceWorkspaceRequirements(
          system.matrix.rows, policy.krylov_dimension,
          support.cooperative_blocks, use_cgs2);
  const AddressRange basis_range = MakeAddressRange(
      workspace.basis, requirements.basis_doubles, sizeof(double));
  const AddressRange vector_range = MakeAddressRange(
      workspace.vectors, requirements.vector_doubles, sizeof(double));
  const AddressRange reduction_range = MakeAddressRange(
      workspace.reduction, requirements.reduction_doubles, sizeof(double));
  const AddressRange small_range = MakeAddressRange(
      workspace.small, requirements.small_doubles, sizeof(double));
  const AddressRange control_range = MakeAddressRange(
      workspace.device_control, 1, sizeof(NativeTraceSolveControl));
  const AddressRange double_ranges[] = {
      basis_range, vector_range, reduction_range, small_range, control_range};
  const std::size_t system_rows = system.matrix.rows > 0
      ? static_cast<std::size_t>(system.matrix.rows) : 0;
  const std::size_t system_nonzeros = system.matrix.nonzeros > 0
      ? static_cast<std::size_t>(system.matrix.nonzeros) : 0;
  const AddressRange system_ranges[] = {
      MakeAddressRange(system.matrix.row_offsets, system_rows + 1,
                       sizeof(int)),
      MakeAddressRange(system.matrix.columns, system_nonzeros, sizeof(int)),
      MakeAddressRange(system.matrix.values, system_nonzeros, sizeof(double)),
      MakeAddressRange(system.rhs, system_rows, sizeof(double)),
      MakeAddressRange(system.solution, system_rows, sizeof(double))};
  bool workspace_overlap = false;
  for (std::size_t left = 0; left < 5; ++left)
  {
    for (std::size_t right = left + 1; right < 5; ++right)
      workspace_overlap = workspace_overlap ||
          AddressRangesOverlap(double_ranges[left], double_ranges[right]);
    for (const AddressRange& borrowed : system_ranges)
      workspace_overlap = workspace_overlap ||
          AddressRangesOverlap(double_ranges[left], borrowed);
  }
  if (use_cgs2)
  {
    const AddressRange permutation_range = MakeAddressRange(
        workspace.permutation_marks, requirements.permutation_integers,
        sizeof(int));
    for (const AddressRange& range : double_ranges)
      workspace_overlap = workspace_overlap ||
          AddressRangesOverlap(permutation_range, range);
    for (const AddressRange& borrowed : system_ranges)
      workspace_overlap = workspace_overlap ||
          AddressRangesOverlap(permutation_range, borrowed);

    if (use_ilu)
    {
      const NativeTraceIlu0View& ilu0 = preconditioner.ilu0;
      const std::size_t factor_rows = ilu0.factor.rows > 0
          ? static_cast<std::size_t>(ilu0.factor.rows) : 0;
      const std::size_t factor_nonzeros = ilu0.factor.nonzeros > 0
          ? static_cast<std::size_t>(ilu0.factor.nonzeros) : 0;
      const AddressRange factor_ranges[] = {
          MakeAddressRange(ilu0.factor.row_offsets, factor_rows + 1,
                           sizeof(int)),
          MakeAddressRange(ilu0.factor.columns, factor_nonzeros, sizeof(int)),
          MakeAddressRange(ilu0.factor.values, factor_nonzeros,
                           sizeof(double)),
          MakeAddressRange(ilu0.permutation, factor_rows, sizeof(int)),
          MakeAddressRange(ilu0.inverse_diagonal, factor_rows,
                           sizeof(double))};
      for (const AddressRange& borrowed : factor_ranges)
      {
        workspace_overlap = workspace_overlap ||
            AddressRangesOverlap(permutation_range, borrowed);
        for (const AddressRange& range : double_ranges)
          workspace_overlap = workspace_overlap ||
              AddressRangesOverlap(range, borrowed);
      }
    }
    else
    {
      const NativeTraceElementSchwarzView& schwarz =
          preconditioner.element_schwarz;
      std::size_t element_slots = 0;
      std::size_t patch_entries = 0;
      std::size_t incidence_slots = 0;
      const bool sizes_valid = schwarz.elements > 0 &&
          schwarz.patch_rows > 0 && schwarz.true_rows > 0 &&
          CheckedMultiply(static_cast<std::size_t>(schwarz.elements),
                          static_cast<std::size_t>(schwarz.patch_rows),
                          element_slots) &&
          CheckedMultiply(element_slots,
                          static_cast<std::size_t>(schwarz.patch_rows),
                          patch_entries) &&
          CheckedMultiply(static_cast<std::size_t>(schwarz.true_rows),
                          static_cast<std::size_t>(
                              native_trace_schwarz_maximum_incidence),
                          incidence_slots);
      const AddressRange schwarz_ranges[] = {
          MakeAddressRange(schwarz.patch_matrix, patch_entries,
                           sizeof(double)),
          MakeAddressRange(schwarz.patch_lu, patch_entries, sizeof(double)),
          MakeAddressRange(schwarz.element_rhs, element_slots,
                           sizeof(double)),
          MakeAddressRange(schwarz.element_solution, element_slots,
                           sizeof(double)),
          MakeAddressRange(schwarz.pivots, element_slots, sizeof(int)),
          MakeAddressRange(schwarz.signed_element_true_dofs, element_slots,
                           sizeof(int)),
          MakeAddressRange(schwarz.true_incidence_slots, incidence_slots,
                           sizeof(int))};
      const AddressRange schur_range = use_matrix_free
          ? MakeAddressRange(system.element_schur, patch_entries,
                             sizeof(double))
          : AddressRange{};
      workspace_overlap = workspace_overlap || !sizes_valid;
      for (std::size_t left = 0; left < 7; ++left)
      {
        for (std::size_t right = left + 1; right < 7; ++right)
          workspace_overlap = workspace_overlap ||
              AddressRangesOverlap(schwarz_ranges[left],
                                   schwarz_ranges[right]);
        workspace_overlap = workspace_overlap ||
            AddressRangesOverlap(permutation_range, schwarz_ranges[left]);
        for (const AddressRange& range : double_ranges)
          workspace_overlap = workspace_overlap ||
              AddressRangesOverlap(range, schwarz_ranges[left]);
        for (const AddressRange& borrowed : system_ranges)
          workspace_overlap = workspace_overlap ||
              AddressRangesOverlap(borrowed, schwarz_ranges[left]);
      }
      if (use_matrix_free)
      {
        for (const AddressRange& range : double_ranges)
          workspace_overlap = workspace_overlap ||
              AddressRangesOverlap(range, schur_range);
        workspace_overlap = workspace_overlap ||
            AddressRangesOverlap(permutation_range, schur_range);
        for (const AddressRange& borrowed : system_ranges)
          workspace_overlap = workspace_overlap ||
              AddressRangesOverlap(borrowed, schur_range);
        for (const AddressRange& borrowed : schwarz_ranges)
          workspace_overlap = workspace_overlap ||
              AddressRangesOverlap(borrowed, schur_range);
      }
    }
  }
  int current_device = -1;
  if (!requirements.valid || workspace_overlap || !support.supported ||
      (use_fixed_sigma_jacobi &&
       (!use_element_schwarz ||
        !NativeTracePreconditionerIsEquilibratedFieldSplit(
            kernel_preconditioner_kind))) ||
      cudaGetDevice(&current_device) != cudaSuccess ||
      current_device != support.device || support.compute_major != 12 ||
      support.compute_minor != 0 || support.multiprocessors <= 0 ||
      support.threads_per_block != native_trace_cooperative_threads ||
      support.maximum_active_blocks_per_multiprocessor <= 0 ||
      support.cooperative_blocks <= 0 ||
      static_cast<long long>(support.cooperative_blocks) >
          static_cast<long long>(support.multiprocessors) *
              support.maximum_active_blocks_per_multiprocessor ||
      policy.rank_count != 1 || policy.krylov_dimension <= 0 ||
      policy.maximum_iterations <= 0 ||
      !(policy.relative_tolerance > 0.0) ||
      !std::isfinite(policy.relative_tolerance) ||
      policy.absolute_tolerance < 0.0 ||
      !std::isfinite(policy.absolute_tolerance) ||
      system.matrix.rows <= 0 ||
      system.matrix.column_count != system.matrix.rows ||
      system.matrix.nonzeros < system.matrix.rows ||
      system.matrix.row_offsets == nullptr ||
      system.matrix.columns == nullptr || system.matrix.values == nullptr ||
      system.rhs == nullptr || system.solution == nullptr ||
      system.rhs == system.solution ||
      !IsAligned(system.matrix.row_offsets, alignof(int)) ||
      !IsAligned(system.matrix.columns, alignof(int)) ||
      !IsAligned(system.matrix.values, alignof(double)) ||
      !IsAligned(system.rhs, alignof(double)) ||
      !IsAligned(system.solution, alignof(double)) ||
      !IsAligned(workspace.basis, alignof(double)) ||
      !IsAligned(workspace.vectors, alignof(double)) ||
      !IsAligned(workspace.reduction, alignof(double)) ||
      !IsAligned(workspace.small, alignof(double)) ||
      !IsAligned(workspace.device_control,
                 alignof(NativeTraceSolveControl)) ||
      workspace.basis_doubles < requirements.basis_doubles ||
      workspace.vector_doubles < requirements.vector_doubles ||
      workspace.reduction_doubles < requirements.reduction_doubles ||
      workspace.small_doubles < requirements.small_doubles ||
      workspace.permutation_integers < requirements.permutation_integers ||
      (use_matrix_free &&
       (!NativeTraceOperatorMatchesPreconditioner(
            system.operator_kind, kernel_preconditioner_kind) ||
        system.element_schur == nullptr ||
        !IsAligned(system.element_schur, alignof(double)))) ||
      (!use_matrix_free &&
       (system.operator_kind !=
            NativeTraceOperatorKind::assembled_csr_fp64 ||
        system.element_schur != nullptr)))
    return false;

  if (kernel_preconditioner_kind ==
      NativeTracePreconditionerKind::diagonal)
  {
    if (use_cgs2) return false;
  }
  else if (kernel_preconditioner_kind ==
           NativeTracePreconditionerKind::ilu0_rcm_jacobi)
  {
    const NativeTraceIlu0View& ilu0 = preconditioner.ilu0;
    if (ilu0.factor.rows != system.matrix.rows ||
        ilu0.factor.column_count != system.matrix.rows ||
        ilu0.factor.nonzeros < system.matrix.rows ||
        ilu0.factor.row_offsets == nullptr || ilu0.factor.columns == nullptr ||
        ilu0.factor.values == nullptr || ilu0.permutation == nullptr ||
        ilu0.inverse_diagonal == nullptr ||
        ilu0.lower_jacobi_iterations != native_trace_ilu_jacobi_iterations ||
        ilu0.upper_jacobi_iterations != native_trace_ilu_jacobi_iterations ||
        !IsAligned(ilu0.factor.row_offsets, alignof(int)) ||
        !IsAligned(ilu0.factor.columns, alignof(int)) ||
        !IsAligned(ilu0.factor.values, alignof(double)) ||
        !IsAligned(ilu0.permutation, alignof(int)) ||
        !IsAligned(ilu0.inverse_diagonal, alignof(double)) ||
        !IsAligned(workspace.permutation_marks, alignof(int)))
      return false;
  }
  else if (NativeTracePreconditionerIsElementSchwarz(
               kernel_preconditioner_kind))
  {
    const NativeTraceElementSchwarzView& schwarz =
        preconditioner.element_schwarz;
    const bool field_split =
        NativeTracePreconditionerIsEquilibratedFieldSplit(
            kernel_preconditioner_kind);
    const bool warp_field_split =
        NativeTracePreconditionerIsWarpEquilibratedFieldSplit(
            kernel_preconditioner_kind);
    const int scalar_face_rows =
        FieldSplitScalarFaceRows(schwarz.patch_rows);
    const int euler_rows = FieldSplitEulerRows(schwarz.patch_rows);
    const std::size_t field_split_required =
        FieldSplitRightScaleOffset(schwarz.patch_rows) +
        (schwarz.patch_rows > 0
             ? static_cast<std::size_t>(schwarz.patch_rows) : 0);
    const std::size_t patch_capacity = schwarz.patch_rows > 0
        ? static_cast<std::size_t>(schwarz.patch_rows) *
            schwarz.patch_rows : 0;
    const std::size_t selected_field_split_required = warp_field_split
        ? FieldSplitWarpRequiredDoubles(schwarz.patch_rows)
        : field_split_required;
    if (schwarz.elements <= 0 ||
        (schwarz.patch_rows != 40 && schwarz.patch_rows != 60 &&
         schwarz.patch_rows != 80) ||
        (field_split &&
         (scalar_face_rows < 2 || scalar_face_rows > 4 ||
          euler_rows != 4 * schwarz.patch_rows / 5 ||
          selected_field_split_required > patch_capacity)) ||
        schwarz.true_rows != system.matrix.rows ||
        schwarz.minimum_multiplicity < 1 ||
        schwarz.maximum_multiplicity < schwarz.minimum_multiplicity ||
        schwarz.maximum_multiplicity >
            native_trace_schwarz_maximum_incidence ||
        schwarz.patch_matrix == nullptr || schwarz.patch_lu == nullptr ||
        schwarz.pivots == nullptr ||
        schwarz.signed_element_true_dofs == nullptr ||
        schwarz.true_incidence_slots == nullptr ||
        schwarz.element_rhs == nullptr ||
        schwarz.element_solution == nullptr ||
        !IsAligned(schwarz.patch_matrix, alignof(double)) ||
        !IsAligned(schwarz.patch_lu, alignof(double)) ||
        !IsAligned(schwarz.pivots, alignof(int)) ||
        !IsAligned(schwarz.signed_element_true_dofs, alignof(int)) ||
        !IsAligned(schwarz.true_incidence_slots, alignof(int)) ||
        !IsAligned(schwarz.element_rhs, alignof(double)) ||
        !IsAligned(schwarz.element_solution, alignof(double)) ||
        !IsAligned(workspace.permutation_marks, alignof(int)))
      return false;
  }
  else
    return false;

  NativeTraceSystemView system_argument = system;
  NativeTracePreconditionerView preconditioner_argument = preconditioner;
  preconditioner_argument.kind = kernel_preconditioner_kind;
  NativeTraceWorkspaceView workspace_argument = workspace;
  NativeTraceSolvePolicy policy_argument = policy;
  NativeTraceCooperativeSupport support_argument = support;
  void* arguments[] = {&system_argument, &preconditioner_argument,
                       &workspace_argument, &policy_argument,
                       &support_argument};
  const void* const kernel = use_fixed_sigma_jacobi
      ? reinterpret_cast<const void*>(
            NativeRankOneTraceSolveKernel<true, true>)
      : use_cgs2
          ? reinterpret_cast<const void*>(
                NativeRankOneTraceSolveKernel<true, false>)
          : reinterpret_cast<const void*>(
                NativeRankOneTraceSolveKernel<false, false>);
  const cudaError_t launch = cudaLaunchCooperativeKernel(
      kernel,
      dim3(static_cast<unsigned int>(support.cooperative_blocks)),
      dim3(static_cast<unsigned int>(native_trace_cooperative_threads)),
      arguments, 0, stream);
  return launch == cudaSuccess;
}

} // namespace dg::hdg::runtime::cuda_w8a::detail

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

__device__ void ComputeActualResidual(
    const NativeTraceSystemView system, double* residual,
    NativeTraceSolveControl* control,
    const cg::grid_group& grid) noexcept
{
  const int thread = GlobalThreadIndex();
  const int stride = GlobalThreadStride();
  for (int row = thread; row < system.matrix.rows; row += stride)
  {
    double value = system.rhs[row];
    for (int entry = system.matrix.row_offsets[row];
         entry < system.matrix.row_offsets[row + 1]; ++entry)
      value -= system.matrix.values[entry] *
          system.solution[system.matrix.columns[entry]];
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

__device__ void SetFinalResidualMetadata(
    NativeTraceSolveControl* control, const double residual) noexcept
{
  control->actual_residual_norm = residual;
  control->relative_residual_norm = control->rhs_norm > 0.0
      ? residual / control->rhs_norm : residual;
}

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
      ComputeActualResidual(system, first_basis, control, grid);
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

      ApplyPreconditioner(system, preconditioner, workspace,
                          current_basis, work, control, grid);
      if (FinishOnFailure(grid, control)) return;
      SparseMatrixVector(system.matrix, work, next_basis, control, grid);
      if (FinishOnFailure(grid, control)) return;

      // Pinned HYPRE modified Gram-Schmidt, one deterministic reduction per
      // basis vector and the same update order.
      for (int basis_index = 0; basis_index < restart_size; ++basis_index)
      {
        const double* const basis = workspace.basis +
            static_cast<std::size_t>(basis_index) * rows;
        const double inner = GridDot(basis, next_basis, rows, workspace,
                                     control, grid, block_scratch);
        if (!isfinite(inner)) MarkFailure(control, failure_numerical_breakdown);
        if (grid.thread_rank() == 0)
          hessenberg[static_cast<std::size_t>(basis_index) *
              krylov_dimension + column] = inner;
        grid.sync();
        if (FinishOnFailure(grid, control)) return;
        AxpyVector(next_basis, -inner, basis, rows, control, grid);
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
    ApplyPreconditioner(system, preconditioner, workspace,
                        work, work, control, grid);
    if (FinishOnFailure(grid, control)) return;
    AxpyVector(system.solution, 1.0, work, rows, control, grid);
    if (FinishOnFailure(grid, control)) return;

    bool false_convergence_restart = false;
    if (residual_norm <= epsilon)
    {
      ComputeActualResidual(system, first_basis, control, grid);
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
      if (FinishOnFailure(grid, control)) return;
    }
  }

  // Match the established bounded HYPRE lifecycle: publish one final actual
  // residual for a maximum-iteration exit, then let the application rebuild
  // the authoritative FP64 true residual from the current matrix.
  ComputeActualResidual(system, first_basis, control, grid);
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
                                          const bool igr_active) noexcept
{
  (void)igr_active;
  NativeTraceCooperativeSupport result{};
  if (rows <= 0 || krylov_dimension <= 0 ||
      krylov_dimension == std::numeric_limits<int>::max())
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
                             device) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, NativeRankOneTraceSolveKernel,
          native_trace_cooperative_threads, 0) != cudaSuccess)
    return NativeTraceCooperativeSupport{};

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
  const bool igr = preconditioner.kind ==
      NativeTracePreconditionerKind::ilu0_rcm_jacobi;
  const NativeTraceWorkspaceRequirements requirements =
      QueryNativeTraceWorkspaceRequirements(
          system.matrix.rows, policy.krylov_dimension,
          support.cooperative_blocks, igr);
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
  if (igr)
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

    const NativeTraceIlu0View& ilu0 = preconditioner.ilu0;
    const std::size_t factor_rows = ilu0.factor.rows > 0
        ? static_cast<std::size_t>(ilu0.factor.rows) : 0;
    const std::size_t factor_nonzeros = ilu0.factor.nonzeros > 0
        ? static_cast<std::size_t>(ilu0.factor.nonzeros) : 0;
    const AddressRange factor_ranges[] = {
        MakeAddressRange(ilu0.factor.row_offsets, factor_rows + 1,
                         sizeof(int)),
        MakeAddressRange(ilu0.factor.columns, factor_nonzeros, sizeof(int)),
        MakeAddressRange(ilu0.factor.values, factor_nonzeros, sizeof(double)),
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
  int current_device = -1;
  if (!requirements.valid || workspace_overlap || !support.supported ||
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
      workspace.permutation_integers < requirements.permutation_integers)
    return false;

  if (preconditioner.kind == NativeTracePreconditionerKind::diagonal)
  {
    if (igr) return false;
  }
  else if (preconditioner.kind ==
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
  else
    return false;

  NativeTraceSystemView system_argument = system;
  NativeTracePreconditionerView preconditioner_argument = preconditioner;
  NativeTraceWorkspaceView workspace_argument = workspace;
  NativeTraceSolvePolicy policy_argument = policy;
  NativeTraceCooperativeSupport support_argument = support;
  void* arguments[] = {&system_argument, &preconditioner_argument,
                       &workspace_argument, &policy_argument,
                       &support_argument};
  const cudaError_t launch = cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(NativeRankOneTraceSolveKernel),
      dim3(static_cast<unsigned int>(support.cooperative_blocks)),
      dim3(static_cast<unsigned int>(native_trace_cooperative_threads)),
      arguments, 0, stream);
  return launch == cudaSuccess;
}

} // namespace dg::hdg::runtime::cuda_w8a::detail

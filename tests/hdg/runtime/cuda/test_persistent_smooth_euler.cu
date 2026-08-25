#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include "numerics/hdg/runtime/cuda/native_rank1_trace_solver.cuh"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::cuda_w8a;

static_assert(std::is_aggregate<CudaConfig>::value,
              "benchmark policy selection must not change CudaConfig ABI");

constexpr bool TrueResidualAuthorityPolicy() noexcept
{
  dg::hdg::runtime::cuda_w8a::detail::CudaSolveDiagnostics diagnostics{};
  diagnostics.device_policy = true;
  diagnostics.trace_solver =
      dg::hdg::runtime::cuda_w8a::detail::CudaTraceSolverKind::
          hypre_parcsr_gmres;
  diagnostics.configured_preconditioner = true;
  diagnostics.configured_ilu = true;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_ilu0;
  diagnostics.finite = true;
  diagnostics.iterations = 0;
  diagnostics.reported_residual = 1.1286157960703033e-12;
  diagnostics.reported_tolerance_met = false;
  const bool reported_tolerance_is_not_a_gate =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.ilu_accounting_kind = CudaIluAccountingKind::unavailable;
  diagnostics.ilu_factor_reuse_policy =
      CudaIluFactorReusePolicy::not_applicable;
  diagnostics.ilu_factor_setups =
      std::numeric_limits<std::uint64_t>::max();
  diagnostics.ilu_factor_reuses =
      std::numeric_limits<std::uint64_t>::max();
  diagnostics.ilu_preconditioner_callbacks_lower_bound = 9;
  diagnostics.ilu_preconditioner_callbacks_upper_bound = 1;
  diagnostics.ilu_inner_iterations_lower_bound = 9;
  diagnostics.ilu_inner_iterations_upper_bound = 1;
  diagnostics.ilu_jacobi_factor_spmv_updates_lower_bound = 72;
  diagnostics.ilu_jacobi_factor_spmv_updates_upper_bound = 8;
  diagnostics.ilu_factor_age_generations =
      std::numeric_limits<std::uint64_t>::max();
  diagnostics.ilu_accounting_complete = false;
  const bool malformed_aggregate_is_not_a_gate =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  auto unconfigured = diagnostics;
  unconfigured.trace_solver =
      dg::hdg::runtime::cuda_w8a::detail::CudaTraceSolverKind::unconfigured;
  const bool unconfigured_solver_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(unconfigured, 8);
  auto native = diagnostics;
  native.trace_solver =
      dg::hdg::runtime::cuda_w8a::detail::CudaTraceSolverKind::
          native_rank1_fp64;
  native.configured_native_solver = true;
  const bool native_solver_is_admitted =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(native, 8);
  diagnostics.configured_ilu = false;
  diagnostics.configured_diagonal = true;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_diag_scale;
  const bool diagonal_is_valid =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.configured_preconditioner = false;
  diagnostics.configured_diagonal = false;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::unconfigured;
  const bool neither_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.configured_preconditioner = true;
  diagnostics.configured_ilu = true;
  diagnostics.configured_diagonal = true;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_ilu0;
  const bool both_are_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.configured_diagonal = false;
  diagnostics.iterations = 8;
  const bool cap_is_inclusive =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.iterations = -1;
  const bool negative_iteration_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  diagnostics.iterations = 9;
  const bool over_cap_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
  return reported_tolerance_is_not_a_gate &&
      malformed_aggregate_is_not_a_gate &&
      unconfigured_solver_is_rejected && native_solver_is_admitted &&
      diagonal_is_valid &&
      neither_is_rejected && both_are_rejected && cap_is_inclusive &&
      negative_iteration_is_rejected && over_cap_is_rejected;
}

static_assert(TrueResidualAuthorityPolicy(),
              "the rebuilt true residual must remain authoritative");

constexpr bool TraceSolverSelectionPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  using dg::hdg::runtime::cuda_w8a::detail::CudaTraceSolverKind;
  using dg::hdg::runtime::cuda_w8a::detail::ResolveCudaTraceSolverKind;
  return ResolveCudaTraceSolverKind(
             CudaEngineKind::ordinary, 1, false, true, true, true) ==
          CudaTraceSolverKind::hypre_parcsr_gmres &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::ordinary, 2, true, true, true, true) ==
          CudaTraceSolverKind::hypre_host_staged_rank2 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, true, true, true) ==
          CudaTraceSolverKind::native_rank1_fp64 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, false, false, false) ==
          CudaTraceSolverKind::unconfigured &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, true, true, true, true) ==
          CudaTraceSolverKind::hypre_host_staged_rank2 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, false, true, true, true) ==
          CudaTraceSolverKind::unconfigured;
}

static_assert(TraceSolverSelectionPolicy(),
              "trace solver selection must be typed and fail closed");

bool NativeBoundedResultPolicy() noexcept
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  NativeTraceSolvePolicy policy{};
  policy.rank_count = 1;
  policy.krylov_dimension = 4;
  policy.maximum_iterations = 8;
  policy.relative_tolerance = 1.0e-6;
  NativeTraceSolveControl result{};
  result.schema = native_trace_control_schema;
  result.status = NativeTraceSolveStatus::maximum_iterations;
  result.iterations = 8;
  result.restart_cycles = 2;
  result.preconditioner_applications = 10;
  result.actual_residual_checks = 2;
  result.rhs_norm = 1.0;
  result.initial_residual_norm = 1.0;
  result.estimated_residual_norm = 1.0e-3;
  result.actual_residual_norm = 1.0e-3;
  result.relative_residual_norm = 1.0e-3;
  const bool bounded_maximum_reaches_true_gate =
      NativeTraceResultIsReadyForTrueResidual(result, policy);
  result.status = NativeTraceSolveStatus::converged;
  const bool false_internal_convergence_is_rejected =
      !NativeTraceResultIsReadyForTrueResidual(result, policy);
  result.status = NativeTraceSolveStatus::maximum_iterations;
  result.iterations = 7;
  const bool premature_maximum_is_rejected =
      !NativeTraceResultIsReadyForTrueResidual(result, policy);
  result.status =
      NativeTraceSolveStatus::bounded_real_residual_stagnation;
  const bool bounded_stagnation_reaches_true_gate =
      NativeTraceResultIsReadyForTrueResidual(result, policy);
  result.iterations = 0;
  const bool fabricated_stagnation_is_rejected =
      !NativeTraceResultIsReadyForTrueResidual(result, policy);
  result.iterations = 7;
  result.failure_mask = 1;
  const bool failed_stagnation_is_rejected =
      !NativeTraceResultIsReadyForTrueResidual(result, policy);
  return bounded_maximum_reaches_true_gate &&
      false_internal_convergence_is_rejected &&
      premature_maximum_is_rejected &&
      bounded_stagnation_reaches_true_gate &&
      fabricated_stagnation_is_rejected &&
      failed_stagnation_is_rejected;
}

bool NativeWorkspaceRequirementsPolicy() noexcept
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  const NativeTraceWorkspaceRequirements conventional =
      QueryNativeTraceWorkspaceRequirements(10, 4, 2, false);
  const NativeTraceWorkspaceRequirements igr =
      QueryNativeTraceWorkspaceRequirements(10, 4, 2, true);
  return conventional.valid && igr.valid &&
      conventional.basis_doubles == 50 && igr.basis_doubles == 50 &&
      conventional.vector_doubles == 10 && igr.vector_doubles == 30 &&
      conventional.reduction_doubles == 2 &&
      igr.reduction_doubles == 12 &&
      conventional.small_doubles == 33 && igr.small_doubles == 33 &&
      conventional.permutation_integers == 0 &&
      igr.permutation_integers == 10 &&
      conventional.control_double_slots == NativeTraceControlDoubleSlots() &&
      igr.control_double_slots == NativeTraceControlDoubleSlots();
}

template <typename T>
class NativeDeviceAllocation
{
 public:
  NativeDeviceAllocation() = default;
  NativeDeviceAllocation(const NativeDeviceAllocation&) = delete;
  NativeDeviceAllocation& operator=(const NativeDeviceAllocation&) = delete;
  ~NativeDeviceAllocation()
  {
    if (pointer_ != nullptr) (void)cudaFree(pointer_);
  }

  bool Allocate(const std::size_t count) noexcept
  {
    if (pointer_ != nullptr) return false;
    count_ = count;
    return count == 0 ||
        cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)) ==
            cudaSuccess;
  }

  bool CopyFrom(const std::vector<T>& values) noexcept
  {
    return Allocate(values.size()) &&
        (values.empty() ||
         cudaMemcpy(pointer_, values.data(), values.size() * sizeof(T),
                    cudaMemcpyHostToDevice) == cudaSuccess);
  }

  bool CopyTo(std::vector<T>& values) const noexcept
  {
    return values.size() == count_ &&
        (values.empty() ||
         cudaMemcpy(values.data(), pointer_, values.size() * sizeof(T),
                    cudaMemcpyDeviceToHost) == cudaSuccess);
  }

  bool CopyOneTo(T& value) const noexcept
  {
    return count_ == 1 && pointer_ != nullptr &&
        cudaMemcpy(&value, pointer_, sizeof(T), cudaMemcpyDeviceToHost) ==
            cudaSuccess;
  }

  T* data() noexcept { return pointer_; }
  const T* data() const noexcept { return pointer_; }

 private:
  T* pointer_ = nullptr;
  std::size_t count_ = 0;
};

struct NativeCsrFixture
{
  int rows = 0;
  std::vector<int> offsets{};
  std::vector<int> columns{};
  std::vector<double> values{};
};

struct NativeIluFixture
{
  NativeCsrFixture factor{};
  std::vector<int> permutation{};
  std::vector<double> inverse_diagonal{};
};

struct NativeSolveFixture
{
  NativeCsrFixture matrix{};
  std::vector<double> right_hand_side{};
  NativeIluFixture ilu{};
  dg::hdg::runtime::cuda_w8a::detail::NativeTraceSolvePolicy policy{};
  bool igr = false;
};

struct NativeSolveFixtureResult
{
  dg::hdg::runtime::cuda_w8a::detail::NativeTraceSolveControl control{};
  std::vector<double> solution{};
};

NativeCsrFixture DenseToDiagonalFirstCsr(const std::vector<double>& dense,
                                         const int rows)
{
  NativeCsrFixture result{};
  if (rows <= 0 || dense.size() != static_cast<std::size_t>(rows) * rows)
    return result;
  result.rows = rows;
  result.offsets.reserve(static_cast<std::size_t>(rows) + 1);
  result.columns.reserve(dense.size());
  result.values.reserve(dense.size());
  result.offsets.push_back(0);
  for (int row = 0; row < rows; ++row)
  {
    result.columns.push_back(row);
    result.values.push_back(dense[static_cast<std::size_t>(row) * rows + row]);
    for (int column = 0; column < rows; ++column)
      if (column != row)
      {
        result.columns.push_back(column);
        result.values.push_back(
            dense[static_cast<std::size_t>(row) * rows + column]);
      }
    result.offsets.push_back(static_cast<int>(result.columns.size()));
  }
  return result;
}

bool NativeCsrFixtureIsValid(const NativeCsrFixture& matrix) noexcept
{
  if (matrix.rows <= 0 ||
      matrix.offsets.size() != static_cast<std::size_t>(matrix.rows) + 1 ||
      matrix.columns.size() != matrix.values.size() ||
      matrix.offsets.front() != 0 ||
      matrix.offsets.back() != static_cast<int>(matrix.values.size()))
    return false;
  for (int row = 0; row < matrix.rows; ++row)
  {
    const int begin = matrix.offsets[static_cast<std::size_t>(row)];
    const int end = matrix.offsets[static_cast<std::size_t>(row) + 1];
    if (begin < 0 || end <= begin ||
        end > static_cast<int>(matrix.values.size()) ||
        matrix.columns[static_cast<std::size_t>(begin)] != row)
      return false;
    int diagonal_count = 0;
    for (int entry = begin; entry < end; ++entry)
    {
      const int column = matrix.columns[static_cast<std::size_t>(entry)];
      const double value = matrix.values[static_cast<std::size_t>(entry)];
      if (column < 0 || column >= matrix.rows || !std::isfinite(value))
        return false;
      if (column == row) ++diagonal_count;
    }
    if (diagonal_count != 1) return false;
  }
  return true;
}

std::vector<double> CsrToDense(const NativeCsrFixture& matrix)
{
  std::vector<double> dense(
      static_cast<std::size_t>(matrix.rows) * matrix.rows, 0.0);
  for (int row = 0; row < matrix.rows; ++row)
    for (int entry = matrix.offsets[static_cast<std::size_t>(row)];
         entry < matrix.offsets[static_cast<std::size_t>(row) + 1]; ++entry)
      dense[static_cast<std::size_t>(row) * matrix.rows +
            matrix.columns[static_cast<std::size_t>(entry)]] =
          matrix.values[static_cast<std::size_t>(entry)];
  return dense;
}

bool SolveDenseReference(std::vector<double> matrix,
                         std::vector<double> right_hand_side,
                         std::vector<double>& solution)
{
  const int rows = static_cast<int>(right_hand_side.size());
  if (rows <= 0 ||
      matrix.size() != static_cast<std::size_t>(rows) * rows)
    return false;
  double scale = 0.0;
  for (const double value : matrix)
  {
    if (!std::isfinite(value)) return false;
    scale = std::max(scale, std::abs(value));
  }
  for (const double value : right_hand_side)
    if (!std::isfinite(value)) return false;
  const double pivot_floor = 128.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, scale);
  for (int column = 0; column < rows; ++column)
  {
    int pivot = column;
    for (int row = column + 1; row < rows; ++row)
      if (std::abs(matrix[static_cast<std::size_t>(row) * rows + column]) >
          std::abs(matrix[static_cast<std::size_t>(pivot) * rows + column]))
        pivot = row;
    if (!(std::abs(matrix[static_cast<std::size_t>(pivot) * rows + column]) >
          pivot_floor))
      return false;
    if (pivot != column)
    {
      for (int entry = 0; entry < rows; ++entry)
        std::swap(matrix[static_cast<std::size_t>(column) * rows + entry],
                  matrix[static_cast<std::size_t>(pivot) * rows + entry]);
      std::swap(right_hand_side[static_cast<std::size_t>(column)],
                right_hand_side[static_cast<std::size_t>(pivot)]);
    }
    const double diagonal =
        matrix[static_cast<std::size_t>(column) * rows + column];
    for (int row = column + 1; row < rows; ++row)
    {
      const double multiplier =
          matrix[static_cast<std::size_t>(row) * rows + column] / diagonal;
      matrix[static_cast<std::size_t>(row) * rows + column] = 0.0;
      for (int entry = column + 1; entry < rows; ++entry)
        matrix[static_cast<std::size_t>(row) * rows + entry] -= multiplier *
            matrix[static_cast<std::size_t>(column) * rows + entry];
      right_hand_side[static_cast<std::size_t>(row)] -=
          multiplier * right_hand_side[static_cast<std::size_t>(column)];
    }
  }
  solution.assign(static_cast<std::size_t>(rows), 0.0);
  for (int row = rows - 1; row >= 0; --row)
  {
    double value = right_hand_side[static_cast<std::size_t>(row)];
    for (int column = row + 1; column < rows; ++column)
      value -= matrix[static_cast<std::size_t>(row) * rows + column] *
          solution[static_cast<std::size_t>(column)];
    const double diagonal = matrix[static_cast<std::size_t>(row) * rows + row];
    solution[static_cast<std::size_t>(row)] = value / diagonal;
    if (!std::isfinite(solution[static_cast<std::size_t>(row)])) return false;
  }
  return true;
}

double VectorNorm(const std::vector<double>& values) noexcept
{
  long double square = 0.0L;
  for (const double value : values)
    square += static_cast<long double>(value) * value;
  return static_cast<double>(std::sqrt(square));
}

bool CsrResidualNorm(const NativeCsrFixture& matrix,
                     const std::vector<double>& right_hand_side,
                     const std::vector<double>& solution,
                     double& norm) noexcept
{
  if (!NativeCsrFixtureIsValid(matrix) ||
      right_hand_side.size() != static_cast<std::size_t>(matrix.rows) ||
      solution.size() != right_hand_side.size())
    return false;
  long double square = 0.0L;
  for (int row = 0; row < matrix.rows; ++row)
  {
    long double residual = right_hand_side[static_cast<std::size_t>(row)];
    for (int entry = matrix.offsets[static_cast<std::size_t>(row)];
         entry < matrix.offsets[static_cast<std::size_t>(row) + 1]; ++entry)
      residual -= static_cast<long double>(
          matrix.values[static_cast<std::size_t>(entry)]) *
          solution[static_cast<std::size_t>(
              matrix.columns[static_cast<std::size_t>(entry)])];
    square += residual * residual;
  }
  norm = static_cast<double>(std::sqrt(square));
  return std::isfinite(norm);
}

double MaximumScaledDifference(const std::vector<double>& first,
                               const std::vector<double>& second) noexcept
{
  if (first.size() != second.size())
    return std::numeric_limits<double>::infinity();
  double maximum = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index)
  {
    if (!std::isfinite(first[index]) || !std::isfinite(second[index]))
      return std::numeric_limits<double>::infinity();
    maximum = std::max(maximum, std::abs(first[index] - second[index]) /
        std::max({1.0, std::abs(first[index]), std::abs(second[index])}));
  }
  return maximum;
}

bool RunNativeSolveFixture(const NativeSolveFixture& fixture,
                           NativeSolveFixtureResult& result)
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  result = {};
  const int rows = fixture.matrix.rows;
  const bool ilu_shape = !fixture.igr ||
      (NativeCsrFixtureIsValid(fixture.ilu.factor) &&
       fixture.ilu.factor.rows == rows &&
       fixture.ilu.permutation.size() == static_cast<std::size_t>(rows) &&
       fixture.ilu.inverse_diagonal.size() == static_cast<std::size_t>(rows));
  if (!NativeCsrFixtureIsValid(fixture.matrix) || !ilu_shape ||
      fixture.right_hand_side.size() != static_cast<std::size_t>(rows) ||
      fixture.policy.rank_count != 1 || fixture.policy.krylov_dimension <= 0 ||
      fixture.policy.maximum_iterations <= 0)
    return false;

  const NativeTraceCooperativeSupport support =
      QueryNativeRankOneTraceCooperativeSupport(
          rows, fixture.policy.krylov_dimension, fixture.igr);
  if (!support.supported || support.cooperative_blocks <= 0) return false;
  const NativeTraceWorkspaceRequirements requirements =
      QueryNativeTraceWorkspaceRequirements(
          rows, fixture.policy.krylov_dimension, support.cooperative_blocks,
          fixture.igr);
  if (!requirements.valid) return false;

  NativeDeviceAllocation<int> matrix_offsets;
  NativeDeviceAllocation<int> matrix_columns;
  NativeDeviceAllocation<double> matrix_values;
  NativeDeviceAllocation<double> right_hand_side;
  NativeDeviceAllocation<double> solution;
  NativeDeviceAllocation<int> factor_offsets;
  NativeDeviceAllocation<int> factor_columns;
  NativeDeviceAllocation<double> factor_values;
  NativeDeviceAllocation<int> permutation;
  NativeDeviceAllocation<double> inverse_diagonal;
  NativeDeviceAllocation<double> basis;
  NativeDeviceAllocation<double> vectors;
  NativeDeviceAllocation<double> reduction;
  NativeDeviceAllocation<double> small;
  NativeDeviceAllocation<int> permutation_marks;
  NativeDeviceAllocation<NativeTraceSolveControl> control;
  if (!matrix_offsets.CopyFrom(fixture.matrix.offsets) ||
      !matrix_columns.CopyFrom(fixture.matrix.columns) ||
      !matrix_values.CopyFrom(fixture.matrix.values) ||
      !right_hand_side.CopyFrom(fixture.right_hand_side) ||
      !solution.Allocate(static_cast<std::size_t>(rows)) ||
      !basis.Allocate(requirements.basis_doubles) ||
      !vectors.Allocate(requirements.vector_doubles) ||
      !reduction.Allocate(requirements.reduction_doubles) ||
      !small.Allocate(requirements.small_doubles) ||
      !permutation_marks.Allocate(requirements.permutation_integers) ||
      !control.Allocate(1))
    return false;
  if (fixture.igr &&
      (!factor_offsets.CopyFrom(fixture.ilu.factor.offsets) ||
       !factor_columns.CopyFrom(fixture.ilu.factor.columns) ||
       !factor_values.CopyFrom(fixture.ilu.factor.values) ||
       !permutation.CopyFrom(fixture.ilu.permutation) ||
       !inverse_diagonal.CopyFrom(fixture.ilu.inverse_diagonal)))
    return false;

  NativeTraceSystemView system{};
  system.matrix = {matrix_offsets.data(), matrix_columns.data(),
                   matrix_values.data(), rows, rows,
                   static_cast<int>(fixture.matrix.values.size())};
  system.rhs = right_hand_side.data();
  system.solution = solution.data();
  NativeTracePreconditionerView preconditioner{};
  if (fixture.igr)
  {
    preconditioner.kind =
        NativeTracePreconditionerKind::ilu0_rcm_jacobi;
    preconditioner.ilu0.factor = {
        factor_offsets.data(), factor_columns.data(), factor_values.data(),
        rows, rows, static_cast<int>(fixture.ilu.factor.values.size())};
    preconditioner.ilu0.permutation = permutation.data();
    preconditioner.ilu0.inverse_diagonal = inverse_diagonal.data();
    preconditioner.ilu0.lower_jacobi_iterations =
        native_trace_ilu_jacobi_iterations;
    preconditioner.ilu0.upper_jacobi_iterations =
        native_trace_ilu_jacobi_iterations;
  }
  NativeTraceWorkspaceView workspace{};
  workspace.basis = basis.data();
  workspace.basis_doubles = requirements.basis_doubles;
  workspace.vectors = vectors.data();
  workspace.vector_doubles = requirements.vector_doubles;
  workspace.reduction = reduction.data();
  workspace.reduction_doubles = requirements.reduction_doubles;
  workspace.small = small.data();
  workspace.small_doubles = requirements.small_doubles;
  workspace.permutation_marks = permutation_marks.data();
  workspace.permutation_integers = requirements.permutation_integers;
  workspace.device_control = control.data();

  const bool launched = LaunchNativeRankOneTraceSolve(
      system, preconditioner, workspace, fixture.policy, support, nullptr);
  if (!launched || cudaDeviceSynchronize() != cudaSuccess) return false;
  result.solution.resize(static_cast<std::size_t>(rows));
  return solution.CopyTo(result.solution) &&
      control.CopyOneTo(result.control);
}

std::vector<double> ConventionalFixtureMatrix()
{
  return {4.0, -1.0, 0.0, 0.0, 0.0,
          -1.0, 4.0, -1.0, 0.0, 0.0,
          0.0, -1.0, 4.0, -1.0, 0.0,
          0.0, 0.0, -1.0, 4.0, -1.0,
          0.0, 0.0, 0.0, -1.0, 3.0};
}

NativeSolveFixture MakeConventionalFixture(const int krylov_dimension,
                                            const int maximum_iterations,
                                            const double tolerance)
{
  NativeSolveFixture fixture{};
  fixture.matrix = DenseToDiagonalFirstCsr(ConventionalFixtureMatrix(), 5);
  fixture.right_hand_side = {1.0, 2.0, -1.0, 0.5, 3.0};
  fixture.policy.rank_count = 1;
  fixture.policy.krylov_dimension = krylov_dimension;
  fixture.policy.maximum_iterations = maximum_iterations;
  fixture.policy.relative_tolerance = tolerance;
  fixture.policy.absolute_tolerance = 0.0;
  return fixture;
}

NativeSolveFixture MakeIgrFixture()
{
  // Form A = P^T L U P with HYPRE's gather permutation.  Five zero-start
  // Jacobi sweeps are exact for these four-row triangular factors, so the
  // native right preconditioner should reduce GMRES to one iteration.
  constexpr int rows = 4;
  const std::array<int, rows> permutation{{2, 0, 3, 1}};
  const std::vector<double> lower{{
      1.0, 0.0, 0.0, 0.0,
      0.10, 1.0, 0.0, 0.0,
      0.0, -0.20, 1.0, 0.0,
      0.0, 0.0, 0.15, 1.0}};
  const std::vector<double> upper{{
      2.0, 0.25, 0.0, 0.0,
      0.0, 3.0, -0.30, 0.0,
      0.0, 0.0, 4.0, 0.40,
      0.0, 0.0, 0.0, 5.0}};
  std::vector<double> permuted_matrix(rows * rows, 0.0);
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < rows; ++column)
      for (int inner = 0; inner < rows; ++inner)
        permuted_matrix[static_cast<std::size_t>(row) * rows + column] +=
            lower[static_cast<std::size_t>(row) * rows + inner] *
            upper[static_cast<std::size_t>(inner) * rows + column];
  std::vector<double> matrix(rows * rows, 0.0);
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < rows; ++column)
      matrix[static_cast<std::size_t>(permutation[row]) * rows +
             permutation[column]] =
          permuted_matrix[static_cast<std::size_t>(row) * rows + column];
  std::vector<double> combined_factor(rows * rows, 0.0);
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < rows; ++column)
      combined_factor[static_cast<std::size_t>(row) * rows + column] =
          column < row
          ? lower[static_cast<std::size_t>(row) * rows + column]
          : upper[static_cast<std::size_t>(row) * rows + column];

  NativeSolveFixture fixture{};
  fixture.matrix = DenseToDiagonalFirstCsr(matrix, rows);
  fixture.right_hand_side = {1.0, -2.0, 0.5, 3.0};
  fixture.ilu.factor = DenseToDiagonalFirstCsr(combined_factor, rows);
  fixture.ilu.permutation.assign(permutation.begin(), permutation.end());
  fixture.ilu.inverse_diagonal = {0.5, 1.0 / 3.0, 0.25, 0.2};
  fixture.policy.rank_count = 1;
  fixture.policy.krylov_dimension = 2;
  fixture.policy.maximum_iterations = 12;
  fixture.policy.relative_tolerance = 1.0e-12;
  fixture.policy.absolute_tolerance = 0.0;
  fixture.igr = true;
  return fixture;
}

NativeSolveFixture MakeIgrCgs2TileFixture(const int krylov_dimension)
{
  // A defective upper-Jordan operator with a dense right-hand side produces a
  // long, nonnormal Krylov sequence.  The identity factor deliberately keeps
  // the right preconditioner neutral while selecting the production IGR/CGS2
  // specialization, including its permutation and five-sweep ILU path.
  constexpr int rows = 32;
  constexpr double superdiagonal = 0.75;
  std::vector<double> matrix(static_cast<std::size_t>(rows) * rows, 0.0);
  std::vector<double> identity(static_cast<std::size_t>(rows) * rows, 0.0);
  for (int row = 0; row < rows; ++row)
  {
    matrix[static_cast<std::size_t>(row) * rows + row] = 1.0;
    identity[static_cast<std::size_t>(row) * rows + row] = 1.0;
    if (row + 1 < rows)
      matrix[static_cast<std::size_t>(row) * rows + row + 1] =
          superdiagonal;
  }

  NativeSolveFixture fixture{};
  fixture.matrix = DenseToDiagonalFirstCsr(matrix, rows);
  fixture.right_hand_side.assign(static_cast<std::size_t>(rows), 1.0);
  fixture.ilu.factor = DenseToDiagonalFirstCsr(identity, rows);
  fixture.ilu.permutation.resize(static_cast<std::size_t>(rows));
  fixture.ilu.inverse_diagonal.assign(static_cast<std::size_t>(rows), 1.0);
  for (int row = 0; row < rows; ++row)
    fixture.ilu.permutation[static_cast<std::size_t>(row)] = rows - 1 - row;
  fixture.policy.rank_count = 1;
  fixture.policy.krylov_dimension = krylov_dimension;
  fixture.policy.maximum_iterations = 256;
  fixture.policy.relative_tolerance = 1.0e-10;
  fixture.policy.absolute_tolerance = 0.0;
  fixture.igr = true;
  return fixture;
}

bool IgrCgs2TileFixtureHostContract(const NativeSolveFixture& fixture)
{
  constexpr double superdiagonal = 0.75;
  if (!fixture.igr || fixture.matrix.rows != 32) return false;
  const std::vector<double> dense = CsrToDense(fixture.matrix);
  if (dense.size() != static_cast<std::size_t>(fixture.matrix.rows) *
          fixture.matrix.rows ||
      dense[1] != superdiagonal || dense[fixture.matrix.rows] != 0.0)
    return false;

  std::vector<double> direct;
  if (!SolveDenseReference(dense, fixture.right_hand_side, direct))
    return false;
  std::vector<double> expected(
      static_cast<std::size_t>(fixture.matrix.rows), 1.0);
  for (int row = fixture.matrix.rows - 2; row >= 0; --row)
    expected[static_cast<std::size_t>(row)] =
        1.0 - superdiagonal * expected[static_cast<std::size_t>(row + 1)];
  return MaximumScaledDifference(direct, expected) == 0.0;
}

bool SameNativeSolveResult(
    const dg::hdg::runtime::cuda_w8a::detail::NativeTraceSolveControl& first,
    const dg::hdg::runtime::cuda_w8a::detail::NativeTraceSolveControl& second)
    noexcept
{
  // reduction_value is transient reduction scratch, not part of the solve
  // result contract.  Every externally meaningful fixed scalar must repeat.
  return first.schema == second.schema && first.status == second.status &&
      first.failure_mask == second.failure_mask &&
      first.iterations == second.iterations &&
      first.restart_cycles == second.restart_cycles &&
      first.preconditioner_applications ==
          second.preconditioner_applications &&
      first.actual_residual_checks == second.actual_residual_checks &&
      first.rhs_norm == second.rhs_norm &&
      first.initial_residual_norm == second.initial_residual_norm &&
      first.estimated_residual_norm == second.estimated_residual_norm &&
      first.actual_residual_norm == second.actual_residual_norm &&
      first.relative_residual_norm == second.relative_residual_norm;
}

bool NativeConvergedFixtureExact(const NativeSolveFixture& fixture,
                                 const NativeSolveFixtureResult& result,
                                 const bool require_restart,
                                 const int exact_iterations)
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  std::vector<double> direct_solution;
  double residual_norm = std::numeric_limits<double>::infinity();
  const bool direct = SolveDenseReference(
      CsrToDense(fixture.matrix), fixture.right_hand_side, direct_solution);
  const bool residual = CsrResidualNorm(
      fixture.matrix, fixture.right_hand_side, result.solution, residual_norm);
  const double rhs_norm = VectorNorm(fixture.right_hand_side);
  const double metadata_scale = std::max(
      {1.0, residual_norm, result.control.actual_residual_norm});
  const double metadata_tolerance = 4096.0 *
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(fixture.matrix.rows) * metadata_scale;
  const bool iteration_exact = exact_iterations < 0
      ? result.control.iterations > fixture.policy.krylov_dimension
      : result.control.iterations == exact_iterations;
  return direct && residual &&
      result.control.schema == native_trace_control_schema &&
      result.control.status == NativeTraceSolveStatus::converged &&
      result.control.failure_mask == 0 && iteration_exact &&
      result.control.iterations <= fixture.policy.maximum_iterations &&
      (!require_restart || result.control.restart_cycles >= 2) &&
      result.control.preconditioner_applications ==
          result.control.iterations + result.control.restart_cycles &&
      result.control.actual_residual_checks >= 2 &&
      NativeTraceResultIsReadyForTrueResidual(result.control, fixture.policy) &&
      MaximumScaledDifference(result.solution, direct_solution) <= 1.0e-9 &&
      residual_norm <= 4.0 * fixture.policy.relative_tolerance * rhs_norm &&
      std::abs(result.control.actual_residual_norm - residual_norm) <=
          metadata_tolerance &&
      std::abs(result.control.relative_residual_norm -
               residual_norm / rhs_norm) <= metadata_tolerance;
}

bool NativeIgrCgs2TileFixturesExact()
{
  constexpr std::array<int, 4> dimensions{{7, 8, 9, 20}};
  for (const int dimension : dimensions)
  {
    const NativeSolveFixture fixture = MakeIgrCgs2TileFixture(dimension);
    NativeSolveFixtureResult first{};
    NativeSolveFixtureResult second{};
    const bool qualified = IgrCgs2TileFixtureHostContract(fixture) &&
        RunNativeSolveFixture(fixture, first) &&
        RunNativeSolveFixture(fixture, second) &&
        NativeConvergedFixtureExact(fixture, first, true, -1) &&
        NativeConvergedFixtureExact(fixture, second, true, -1) &&
        MaximumScaledDifference(first.solution, second.solution) == 0.0 &&
        SameNativeSolveResult(first.control, second.control);
    if (!qualified)
    {
      std::cerr << "native trace IGR CGS2 tile fixture failed kdim="
                << dimension << " first_status="
                << dg::hdg::runtime::cuda_w8a::detail::
                       NativeTraceSolveStatusName(first.control.status)
                << " second_status="
                << dg::hdg::runtime::cuda_w8a::detail::
                       NativeTraceSolveStatusName(second.control.status)
                << " first_iterations=" << first.control.iterations
                << " second_iterations=" << second.control.iterations << '\n';
      return false;
    }
  }
  return true;
}

bool NativeMaximumIterationFixtureExact(
    const NativeSolveFixture& fixture,
    const NativeSolveFixtureResult& result) noexcept
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  double residual_norm = std::numeric_limits<double>::infinity();
  const double rhs_norm = VectorNorm(fixture.right_hand_side);
  if (!CsrResidualNorm(fixture.matrix, fixture.right_hand_side,
                       result.solution, residual_norm))
    return false;
  const double metadata_scale = std::max(
      {1.0, residual_norm, result.control.actual_residual_norm});
  const double metadata_tolerance = 4096.0 *
      std::numeric_limits<double>::epsilon() *
      static_cast<double>(fixture.matrix.rows) * metadata_scale;
  return result.control.schema == native_trace_control_schema &&
      result.control.status == NativeTraceSolveStatus::maximum_iterations &&
      result.control.failure_mask == 0 &&
      result.control.iterations == fixture.policy.maximum_iterations &&
      result.control.restart_cycles == 1 &&
      result.control.preconditioner_applications == 2 &&
      result.control.actual_residual_checks >= 2 &&
      NativeTraceResultIsReadyForTrueResidual(result.control, fixture.policy) &&
      residual_norm > fixture.policy.relative_tolerance * rhs_norm &&
      std::abs(result.control.actual_residual_norm - residual_norm) <=
          metadata_tolerance &&
      std::abs(result.control.relative_residual_norm -
               residual_norm / rhs_norm) <= metadata_tolerance;
}

bool NativeBreakdownFixtureExact()
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  NativeSolveFixture fixture{};
  // D^{-1}b lies in A's nullspace.  Arnoldi therefore produces an exact zero
  // first column and the guarded triangular solve must reject the result.
  fixture.matrix = DenseToDiagonalFirstCsr(
      {1.0, -1.0, 1.0, -1.0}, 2);
  fixture.right_hand_side = {1.0, -1.0};
  fixture.policy.rank_count = 1;
  fixture.policy.krylov_dimension = 1;
  fixture.policy.maximum_iterations = 4;
  fixture.policy.relative_tolerance = 1.0e-12;
  NativeSolveFixtureResult result{};
  return RunNativeSolveFixture(fixture, result) &&
      result.control.schema == native_trace_control_schema &&
      result.control.status == NativeTraceSolveStatus::numerical_breakdown &&
      result.control.failure_mask != 0 && result.control.iterations == 1 &&
      result.control.restart_cycles == 1 &&
      !NativeTraceResultIsReadyForTrueResidual(
          result.control, fixture.policy);
}

bool NativeTraceDeviceFixtures() noexcept
{
  try
  {
    // A generic five-row system cannot finish within one GMRES(2) cycle;
    // compare its restarted native result with an independent pivoted solve.
    const NativeSolveFixture conventional =
        MakeConventionalFixture(2, 64, 1.0e-12);
    NativeSolveFixtureResult conventional_result{};
    const bool conventional_ready = RunNativeSolveFixture(
            conventional, conventional_result) &&
        NativeConvergedFixtureExact(
            conventional, conventional_result, true, -1);
    if (!conventional_ready)
    {
      std::cerr << "native trace conventional fixture failed status="
                << dg::hdg::runtime::cuda_w8a::detail::
                       NativeTraceSolveStatusName(
                           conventional_result.control.status)
                << " iterations=" << conventional_result.control.iterations
                << " restarts=" << conventional_result.control.restart_cycles
                << '\n';
      return false;
    }

    const NativeSolveFixture igr = MakeIgrFixture();
    NativeSolveFixtureResult igr_result{};
    const bool igr_ready = RunNativeSolveFixture(igr, igr_result) &&
        NativeConvergedFixtureExact(igr, igr_result, false, 1);
    if (!igr_ready)
    {
      std::cerr << "native trace IGR fixture failed status="
                << dg::hdg::runtime::cuda_w8a::detail::
                       NativeTraceSolveStatusName(igr_result.control.status)
                << " iterations=" << igr_result.control.iterations << '\n';
      return false;
    }

    if (!NativeIgrCgs2TileFixturesExact()) return false;

    const NativeSolveFixture maximum =
        MakeConventionalFixture(1, 1, 1.0e-30);
    NativeSolveFixtureResult maximum_result{};
    const bool maximum_ready = RunNativeSolveFixture(maximum, maximum_result) &&
        NativeMaximumIterationFixtureExact(maximum, maximum_result);
    if (!maximum_ready)
    {
      std::cerr << "native trace maximum-iteration fixture failed status="
                << dg::hdg::runtime::cuda_w8a::detail::
                       NativeTraceSolveStatusName(maximum_result.control.status)
                << " iterations=" << maximum_result.control.iterations << '\n';
      return false;
    }

    if (!NativeBreakdownFixtureExact())
    {
      std::cerr << "native trace numerical-breakdown fixture failed\n";
      return false;
    }
    return true;
  }
  catch (...) { return false; }
}

constexpr bool IluSetupOrderPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  using dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupTransition;
  dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupState rejected{};
  if (rejected.Advance(CudaHypreSetupTransition::set_operator) ||
      rejected.Advance(CudaHypreSetupTransition::bind_preconditioner))
    return false;
  dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupState duplicate{};
  if (!duplicate.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      duplicate.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      duplicate.Advance(CudaHypreSetupTransition::set_operator))
    return false;
  dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupState skipped{};
  if (!skipped.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      !skipped.Advance(CudaHypreSetupTransition::set_operator) ||
      !skipped.Advance(CudaHypreSetupTransition::ilu_type) ||
      skipped.Advance(CudaHypreSetupTransition::ilu_reordering) ||
      skipped.Advance(CudaHypreSetupTransition::ilu_fill))
    return false;
  dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupState evidence{};
  evidence.matrix_generation = 23;
  evidence.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_ilu0;
  evidence.ilu_type = 0;
  evidence.ilu_level_of_fill = 0;
  evidence.ilu_local_reordering = 1;
  evidence.ilu_application_iterations = 1;
  evidence.ilu_triangular_solve_kind =
      CudaIluTriangularSolveKind::iterative_jacobi;
  evidence.ilu_lower_jacobi_iterations = 5;
  evidence.ilu_upper_jacobi_iterations = 5;
  if (!evidence.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      !evidence.Advance(CudaHypreSetupTransition::set_operator) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_type) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_fill) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_reordering) ||
      !evidence.Advance(
          CudaHypreSetupTransition::ilu_tri_solve_configured) ||
      !evidence.Advance(
          CudaHypreSetupTransition::ilu_iterative_setup_off) ||
      !evidence.Advance(
          CudaHypreSetupTransition::ilu_application_iterations) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_tol0) ||
      !evidence.Advance(CudaHypreSetupTransition::ilu_print0) ||
      !evidence.Advance(CudaHypreSetupTransition::rebind_preconditioner) ||
      !evidence.Advance(CudaHypreSetupTransition::gmres_settings))
    return false;
  auto wrong_fill = evidence;
  wrong_fill.ilu_level_of_fill = 1;
  auto direct = evidence;
  direct.ilu_triangular_solve_kind = CudaIluTriangularSolveKind::direct;
  direct.ilu_lower_jacobi_iterations = 0;
  direct.ilu_upper_jacobi_iterations = 0;
  auto ordinary_conventional = direct;
  ordinary_conventional.ilu_local_reordering = 0;
  const CudaTracePolicyDescriptor common_fast_ordinary_igr =
      ResolveCudaTracePolicy(CudaEngineKind::ordinary, true,
                             CudaTracePolicyKind::common_fast);
  if (!evidence.ConfigurationExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      !evidence.ConfigurationExactForGeneration(
          23, common_fast_ordinary_igr) ||
      wrong_fill.ConfigurationExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      evidence.ConfigurationExactForGeneration(
          23, CudaEngineKind::ordinary,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      !direct.ConfigurationExactForGeneration(
          23, CudaEngineKind::ordinary,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      !ordinary_conventional.ConfigurationExactForGeneration(
          23, CudaEngineKind::ordinary,
          CudaTracePreconditionerKind::hypre_ilu0, false) ||
      direct.ConfigurationExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      evidence.ConfigurationExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, false) ||
      !evidence.Advance(CudaHypreSetupTransition::mult))
    return false;
  auto extra_terminal = evidence;
  if (extra_terminal.Advance(CudaHypreSetupTransition::mult) ||
      extra_terminal.SolveExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true))
    return false;
  return evidence.SolveExactForGeneration(
             23, CudaEngineKind::tensor_core,
             CudaTracePreconditionerKind::hypre_ilu0, true) &&
      !evidence.SolveExactForGeneration(
          24, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, true);
}

static_assert(IluSetupOrderPolicy(),
              "ILU setup and application work must be generation-exact");

constexpr bool DiagScaleSetupOrderPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  using dg::hdg::runtime::cuda_w8a::detail::CudaHypreSetupTransition;
  dg::hdg::runtime::cuda_w8a::detail::CudaDiagScaleSetupState evidence{};
  evidence.matrix_generation = 17;
  evidence.selected_engine = CudaEngineKind::tensor_core;
  evidence.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_diag_scale;
  auto rejected = evidence;
  if (rejected.Advance(CudaHypreSetupTransition::set_operator) ||
      rejected.Advance(CudaHypreSetupTransition::bind_preconditioner))
    return false;
  auto duplicate = evidence;
  if (!duplicate.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      duplicate.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      duplicate.Advance(CudaHypreSetupTransition::set_operator))
    return false;
  auto skipped = evidence;
  if (!skipped.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      skipped.Advance(CudaHypreSetupTransition::rebind_preconditioner) ||
      skipped.Advance(CudaHypreSetupTransition::set_operator))
    return false;
  if (!evidence.Advance(CudaHypreSetupTransition::bind_preconditioner) ||
      !evidence.Advance(CudaHypreSetupTransition::set_operator) ||
      !evidence.Advance(CudaHypreSetupTransition::rebind_preconditioner) ||
      !evidence.Advance(CudaHypreSetupTransition::gmres_settings) ||
      !evidence.ConfigurationExactForGeneration(
          17, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      evidence.ConfigurationExactForGeneration(
          17, CudaEngineKind::ordinary,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      evidence.ConfigurationExactForGeneration(
          17, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, true) ||
      evidence.SolveExactForGeneration(
          17, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      !evidence.Advance(CudaHypreSetupTransition::mult))
    return false;
  auto common_fast_ordinary = evidence;
  common_fast_ordinary.selected_engine = CudaEngineKind::ordinary;
  if (!common_fast_ordinary.SolveExactForGeneration(
          17, ResolveCudaTracePolicy(
                  CudaEngineKind::ordinary, false,
                  CudaTracePolicyKind::common_fast)))
    return false;
  auto extra_terminal = evidence;
  if (extra_terminal.Advance(CudaHypreSetupTransition::mult) ||
      extra_terminal.SolveExactForGeneration(
          17, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, false))
    return false;
  return evidence.SolveExactForGeneration(
             17, CudaEngineKind::tensor_core,
             CudaTracePreconditionerKind::hypre_diag_scale, false) &&
      !evidence.SolveExactForGeneration(
          18, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_diag_scale, false);
}

static_assert(DiagScaleSetupOrderPolicy(),
              "diagonal scaling must be rebound for each matrix generation");

constexpr bool TracePreconditionerPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  return CudaTracePreconditionerForEngine(
             CudaEngineKind::ordinary, false) ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      CudaTracePreconditionerForEngine(CudaEngineKind::ordinary, true) ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      CudaTracePreconditionerForEngine(CudaEngineKind::tensor_core, false) ==
          CudaTracePreconditionerKind::hypre_diag_scale &&
      CudaTracePreconditionerForEngine(CudaEngineKind::tensor_core, true) ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      CudaTracePreconditionerIsIlu(
          CudaTracePreconditionerKind::hypre_ilu0) &&
      !CudaTracePreconditionerIsIlu(
          CudaTracePreconditionerKind::hypre_diag_scale) &&
      CudaIluType(CudaTracePreconditionerKind::hypre_ilu0) == 0 &&
      CudaIluType(CudaTracePreconditionerKind::hypre_diag_scale) == -1 &&
      CudaIluLevelOfFill(CudaTracePreconditionerKind::hypre_ilu0) == 0 &&
      CudaIluLevelOfFill(
          CudaTracePreconditionerKind::hypre_diag_scale) == -1 &&
      CudaIluLocalReordering(
          CudaTracePreconditionerKind::hypre_ilu0, false) == 0 &&
      CudaIluLocalReordering(
          CudaTracePreconditionerKind::hypre_ilu0, true) == 1 &&
      CudaIluApplicationIterations(
          CudaTracePreconditionerKind::hypre_ilu0) == 1 &&
      CudaIluApplicationIterations(
          CudaTracePreconditionerKind::hypre_diag_scale) == 0 &&
      CudaIluApplicationIterations(CudaEngineKind::ordinary, false) == 1 &&
      CudaIluApplicationIterations(CudaEngineKind::ordinary, true) == 1 &&
      CudaIluApplicationIterations(CudaEngineKind::tensor_core, true) == 1 &&
      CudaIluFactorReusePolicyForEngine(CudaEngineKind::ordinary, false) ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve &&
      CudaIluFactorReusePolicyForEngine(CudaEngineKind::ordinary, true) ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve &&
      CudaIluFactorReusePolicyForEngine(CudaEngineKind::tensor_core, false) ==
          CudaIluFactorReusePolicy::not_applicable &&
      CudaIluFactorReusePolicyForEngine(CudaEngineKind::tensor_core, true) ==
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
      CudaIluTriangularSolveKindForEngine(
          CudaEngineKind::ordinary, false) ==
          CudaIluTriangularSolveKind::direct &&
      CudaIluTriangularSolveKindForEngine(
          CudaEngineKind::ordinary, true) ==
          CudaIluTriangularSolveKind::direct &&
      CudaIluTriangularSolveKindForEngine(
          CudaEngineKind::tensor_core, false) ==
          CudaIluTriangularSolveKind::not_applicable &&
      CudaIluTriangularSolveKindForEngine(
          CudaEngineKind::tensor_core, true) ==
          CudaIluTriangularSolveKind::iterative_jacobi &&
      CudaIluTriangularSolve(CudaEngineKind::ordinary, true) == 1 &&
      CudaIluTriangularSolve(CudaEngineKind::tensor_core, false) == -1 &&
      CudaIluTriangularSolve(CudaEngineKind::tensor_core, true) == 0 &&
      CudaIluLowerJacobiIterations(CudaEngineKind::ordinary, true) == 0 &&
      CudaIluUpperJacobiIterations(CudaEngineKind::ordinary, true) == 0 &&
      CudaIluLowerJacobiIterations(CudaEngineKind::tensor_core, true) == 5 &&
      CudaIluUpperJacobiIterations(CudaEngineKind::tensor_core, true) == 5 &&
      std::string_view(CudaTracePreconditionerName(
          CudaTracePreconditionerKind::hypre_ilu0)) == "hypre_ilu0" &&
      std::string_view(CudaIluTriangularSolveKindName(
          CudaIluTriangularSolveKind::iterative_jacobi)) ==
          "iterative_jacobi" &&
      std::string_view(CudaIluFactorReusePolicyName(
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt)) ==
          "first_trace_solve_per_attempt";
}

static_assert(TracePreconditionerPolicy(),
              "trace preconditioner selection must be engine-and-mode exact");

constexpr bool TracePolicyAblationDescriptorPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  const auto ordinary_conventional_default = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, false,
      CudaTracePolicyKind::engine_default);
  const auto ordinary_igr_default = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::engine_default);
  const auto tensor_conventional_default = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::engine_default);
  const auto tensor_igr_default = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::engine_default);
  const auto ordinary_conventional_common = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, false,
      CudaTracePolicyKind::common_fast);
  const auto tensor_conventional_common = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::common_fast);
  const auto ordinary_igr_common = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::common_fast);
  const auto tensor_igr_common = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::common_fast);
  const auto unknown = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, false,
      static_cast<CudaTracePolicyKind>(255));
  return CudaTracePolicyDescriptorIsExact(ordinary_conventional_default) &&
      ordinary_conventional_default.trace_preconditioner ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      ordinary_conventional_default.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::direct &&
      ordinary_conventional_default.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve &&
      CudaTracePolicyDescriptorIsExact(ordinary_igr_default) &&
      ordinary_igr_default.ilu_local_reordering == 1 &&
      ordinary_igr_default.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::direct &&
      ordinary_igr_default.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve &&
      CudaTracePolicyDescriptorIsExact(tensor_conventional_default) &&
      tensor_conventional_default.trace_preconditioner ==
          CudaTracePreconditionerKind::hypre_diag_scale &&
      tensor_conventional_default.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::not_applicable &&
      CudaTracePolicyDescriptorIsExact(tensor_igr_default) &&
      tensor_igr_default.ilu_local_reordering == 1 &&
      tensor_igr_default.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::iterative_jacobi &&
      tensor_igr_default.ilu_lower_jacobi_iterations == 5 &&
      tensor_igr_default.ilu_upper_jacobi_iterations == 5 &&
      tensor_igr_default.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
      CudaTracePolicyDescriptorIsExact(ordinary_conventional_common) &&
      CudaTracePolicyDescriptorIsExact(tensor_conventional_common) &&
      ordinary_conventional_common.trace_preconditioner ==
          CudaTracePreconditionerKind::hypre_diag_scale &&
      tensor_conventional_common.trace_preconditioner ==
          CudaTracePreconditionerKind::hypre_diag_scale &&
      ordinary_conventional_common.ilu_type == -1 &&
      ordinary_conventional_common.ilu_level_of_fill == -1 &&
      ordinary_conventional_common.ilu_local_reordering == -1 &&
      ordinary_conventional_common.ilu_application_iterations == 0 &&
      ordinary_conventional_common.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::not_applicable &&
      ordinary_conventional_common.ilu_lower_jacobi_iterations == 0 &&
      ordinary_conventional_common.ilu_upper_jacobi_iterations == 0 &&
      ordinary_conventional_common.ilu_accounting_kind ==
          CudaIluAccountingKind::unavailable &&
      ordinary_conventional_common.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::not_applicable &&
      CudaTracePolicyDescriptorIsExact(ordinary_igr_common) &&
      CudaTracePolicyDescriptorIsExact(tensor_igr_common) &&
      ordinary_igr_common.trace_preconditioner ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      ordinary_igr_common.ilu_type == 0 &&
      ordinary_igr_common.ilu_level_of_fill == 0 &&
      ordinary_igr_common.ilu_local_reordering == 1 &&
      ordinary_igr_common.ilu_application_iterations == 1 &&
      ordinary_igr_common.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::iterative_jacobi &&
      ordinary_igr_common.ilu_lower_jacobi_iterations == 5 &&
      ordinary_igr_common.ilu_upper_jacobi_iterations == 5 &&
      ordinary_igr_common.ilu_accounting_kind ==
          CudaIluAccountingKind::derived_bounds_v1 &&
      ordinary_igr_common.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt &&
      tensor_igr_common.ilu_local_reordering ==
          ordinary_igr_common.ilu_local_reordering &&
      tensor_igr_common.ilu_triangular_solve_kind ==
          ordinary_igr_common.ilu_triangular_solve_kind &&
      tensor_igr_common.ilu_factor_reuse_policy ==
          ordinary_igr_common.ilu_factor_reuse_policy &&
      !unknown.valid && !CudaTracePolicyDescriptorIsExact(unknown) &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::engine_default)) == "engine-default" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::common_fast)) == "common-fast";
}

static_assert(TracePolicyAblationDescriptorPolicy(),
              "benchmark trace-policy descriptors must be exact");

constexpr bool IluDerivedBoundsPolicy() noexcept
{
  using dg::hdg::runtime::cuda_w8a::detail::
      CudaDeriveIluAccountingBoundsV1;
  const auto zero = CudaDeriveIluAccountingBoundsV1(0, 20, 1);
  const auto restart_boundary =
      CudaDeriveIluAccountingBoundsV1(20, 20, 1);
  const auto restart_boundary_jacobi =
      CudaDeriveIluAccountingBoundsV1(20, 20, 1, 8);
  const auto restart_tail =
      CudaDeriveIluAccountingBoundsV1(21, 20, 2, 8);
  const auto zero_krylov = CudaDeriveIluAccountingBoundsV1(1, 0, 1);
  const auto zero_application = CudaDeriveIluAccountingBoundsV1(1, 20, 0);
  const auto addition_overflow = CudaDeriveIluAccountingBoundsV1(
      std::numeric_limits<std::uint64_t>::max(), 1, 1);
  const auto multiplication_overflow = CudaDeriveIluAccountingBoundsV1(
      std::numeric_limits<std::uint64_t>::max()/4+1,
      std::numeric_limits<std::uint64_t>::max(), 2);
  const auto jacobi_multiplication_overflow =
      CudaDeriveIluAccountingBoundsV1(
          std::numeric_limits<std::uint64_t>::max()/4,
          std::numeric_limits<std::uint64_t>::max(), 1, 8);
  return zero.ilu_accounting_complete &&
      zero.ilu_preconditioner_callbacks_lower_bound == 0 &&
      zero.ilu_preconditioner_callbacks_upper_bound == 0 &&
      zero.ilu_inner_iterations_lower_bound == 0 &&
      zero.ilu_inner_iterations_upper_bound == 0 &&
      zero.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
      zero.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
      restart_boundary.ilu_accounting_complete &&
      restart_boundary.ilu_preconditioner_callbacks_lower_bound == 21 &&
      restart_boundary.ilu_preconditioner_callbacks_upper_bound == 40 &&
      restart_boundary.ilu_inner_iterations_lower_bound == 21 &&
      restart_boundary.ilu_inner_iterations_upper_bound == 40 &&
      restart_boundary.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
      restart_boundary.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
      restart_boundary_jacobi.ilu_accounting_complete &&
      restart_boundary_jacobi.
          ilu_jacobi_factor_spmv_updates_lower_bound == 168 &&
      restart_boundary_jacobi.
          ilu_jacobi_factor_spmv_updates_upper_bound == 320 &&
      restart_tail.ilu_accounting_complete &&
      restart_tail.ilu_preconditioner_callbacks_lower_bound == 23 &&
      restart_tail.ilu_preconditioner_callbacks_upper_bound == 42 &&
      restart_tail.ilu_inner_iterations_lower_bound == 46 &&
      restart_tail.ilu_inner_iterations_upper_bound == 84 &&
      restart_tail.ilu_jacobi_factor_spmv_updates_lower_bound == 368 &&
      restart_tail.ilu_jacobi_factor_spmv_updates_upper_bound == 672 &&
      !zero_krylov.ilu_accounting_complete &&
      !zero_application.ilu_accounting_complete &&
      !addition_overflow.ilu_accounting_complete &&
      !multiplication_overflow.ilu_accounting_complete &&
      !jacobi_multiplication_overflow.ilu_accounting_complete;
}

static_assert(IluDerivedBoundsPolicy(),
              "derived ILU bounds must cover zero, restart, tail, and "
              "overflow cases");

constexpr dg::hdg::runtime::cuda_w8a::detail::CudaSolveDiagnostics
IluDiagnostics(const bool igr_active,
               const dg::hdg::generic::CudaEngineKind engine,
               const CudaIluFactorReusePolicy factor_reuse_policy,
               const std::uint64_t factor_setups,
               const std::uint64_t factor_reuses,
               const std::uint64_t factor_age,
               const int iterations) noexcept
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  CudaSolveDiagnostics diagnostics{};
  diagnostics.device_policy = true;
  diagnostics.trace_solver =
      engine == dg::hdg::generic::CudaEngineKind::tensor_core
      ? CudaTraceSolverKind::native_rank1_fp64
      : CudaTraceSolverKind::hypre_parcsr_gmres;
  diagnostics.configured_native_solver =
      diagnostics.trace_solver == CudaTraceSolverKind::native_rank1_fp64;
  diagnostics.configured_preconditioner = true;
  diagnostics.configured_ilu = true;
  diagnostics.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_ilu0;
  diagnostics.ilu_type = 0;
  diagnostics.ilu_level_of_fill = 0;
  diagnostics.ilu_local_reordering = igr_active ? 1 : 0;
  diagnostics.ilu_application_iterations = 1;
  diagnostics.ilu_triangular_solve_kind =
      CudaIluTriangularSolveKindForEngine(engine, igr_active);
  diagnostics.ilu_lower_jacobi_iterations =
      CudaIluLowerJacobiIterations(engine, igr_active);
  diagnostics.ilu_upper_jacobi_iterations =
      CudaIluUpperJacobiIterations(engine, igr_active);
  diagnostics.krylov_dimension = 4;
  diagnostics.igr_active = igr_active;
  diagnostics.ilu_accounting_kind =
      CudaIluAccountingKind::derived_bounds_v1;
  diagnostics.ilu_factor_reuse_policy = factor_reuse_policy;
  diagnostics.ilu_factor_setups = factor_setups;
  diagnostics.ilu_factor_reuses = factor_reuses;
  diagnostics.ilu_factor_age_generations = factor_age;
  diagnostics.iterations = iterations;
  const std::uint64_t factor_spmv_updates_per_application =
      diagnostics.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::iterative_jacobi
      ? 8 : 0;
  const auto bounds = CudaDeriveIluAccountingBoundsV1(
      static_cast<std::uint64_t>(iterations), 4, 1,
      factor_spmv_updates_per_application);
  diagnostics.ilu_preconditioner_callbacks_lower_bound =
      bounds.ilu_preconditioner_callbacks_lower_bound;
  diagnostics.ilu_preconditioner_callbacks_upper_bound =
      bounds.ilu_preconditioner_callbacks_upper_bound;
  diagnostics.ilu_inner_iterations_lower_bound =
      bounds.ilu_inner_iterations_lower_bound;
  diagnostics.ilu_inner_iterations_upper_bound =
      bounds.ilu_inner_iterations_upper_bound;
  diagnostics.ilu_jacobi_factor_spmv_updates_lower_bound =
      bounds.ilu_jacobi_factor_spmv_updates_lower_bound;
  diagnostics.ilu_jacobi_factor_spmv_updates_upper_bound =
      bounds.ilu_jacobi_factor_spmv_updates_upper_bound;
  diagnostics.ilu_accounting_complete = bounds.ilu_accounting_complete;
  return diagnostics;
}

constexpr bool IluWorkAccumulationPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  using namespace dg::hdg::runtime::cuda_w8a::detail;

  CudaTensorCoreEvidence invalid_evidence{};
  invalid_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (CudaInitializeIluEvidence(
          invalid_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, false) ||
      invalid_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence diagonal_evidence{};
  diagonal_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (!CudaInitializeIluEvidence(
          diagonal_evidence,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      diagonal_evidence.ilu_factor_reuse_policy !=
          CudaIluFactorReusePolicy::not_applicable ||
      diagonal_evidence.ilu_triangular_solve_kind !=
          CudaIluTriangularSolveKind::not_applicable ||
      diagonal_evidence.ilu_lower_jacobi_iterations != 0 ||
      diagonal_evidence.ilu_upper_jacobi_iterations != 0 ||
      diagonal_evidence.ilu_factor_setups != 0 ||
      diagonal_evidence.ilu_factor_reuses != 0 ||
      diagonal_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 0 ||
      diagonal_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 0 ||
      diagonal_evidence.ilu_factor_age_generations != 0 ||
      !diagonal_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence common_diagonal_evidence{};
  common_diagonal_evidence.selected_engine = CudaEngineKind::ordinary;
  common_diagonal_evidence.trace_policy = CudaTracePolicyKind::common_fast;
  if (!CudaInitializeIluEvidence(
          common_diagonal_evidence,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      common_diagonal_evidence.ilu_factor_reuse_policy !=
          CudaIluFactorReusePolicy::not_applicable ||
      !common_diagonal_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence ordinary_evidence{};
  ordinary_evidence.selected_engine = CudaEngineKind::ordinary;
  if (!CudaInitializeIluEvidence(
          ordinary_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, false))
    return false;
  const auto refresh = IluDiagnostics(
      false, CudaEngineKind::ordinary,
      CudaIluFactorReusePolicy::refresh_each_trace_solve,
      1, 0, 0, 9);
  if (!CudaAccumulateIluSolveEvidence(ordinary_evidence, refresh) ||
      !CudaAccumulateIluSolveEvidence(ordinary_evidence, refresh) ||
      ordinary_evidence.ilu_factor_reuse_policy !=
          CudaIluFactorReusePolicy::refresh_each_trace_solve ||
      ordinary_evidence.ilu_factor_setups != 2 ||
      ordinary_evidence.ilu_factor_reuses != 0 ||
      ordinary_evidence.ilu_factor_age_generations != 0 ||
      ordinary_evidence.ilu_triangular_solve_kind !=
          CudaIluTriangularSolveKind::direct ||
      ordinary_evidence.ilu_lower_jacobi_iterations != 0 ||
      ordinary_evidence.ilu_upper_jacobi_iterations != 0 ||
      ordinary_evidence.ilu_preconditioner_callbacks_lower_bound != 24 ||
      ordinary_evidence.ilu_preconditioner_callbacks_upper_bound != 36 ||
      ordinary_evidence.ilu_inner_iterations_lower_bound != 24 ||
      ordinary_evidence.ilu_inner_iterations_upper_bound != 36 ||
      ordinary_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 0 ||
      ordinary_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 0 ||
      !ordinary_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence tensor_igr_evidence{};
  tensor_igr_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (!CudaInitializeIluEvidence(
          tensor_igr_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, true))
    return false;
  const auto first = IluDiagnostics(
      true, CudaEngineKind::tensor_core,
      CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
      1, 0, 0, 9);
  const auto reuse = IluDiagnostics(
      true, CudaEngineKind::tensor_core,
      CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
      0, 1, 1, 0);
  if (!CudaAccumulateIluSolveEvidence(tensor_igr_evidence, first) ||
      !CudaAccumulateIluSolveEvidence(tensor_igr_evidence, reuse) ||
      tensor_igr_evidence.ilu_factor_reuse_policy !=
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt ||
      tensor_igr_evidence.ilu_factor_setups != 1 ||
      tensor_igr_evidence.ilu_factor_reuses != 1 ||
      tensor_igr_evidence.ilu_factor_age_generations != 1 ||
      tensor_igr_evidence.ilu_triangular_solve_kind !=
          CudaIluTriangularSolveKind::iterative_jacobi ||
      tensor_igr_evidence.ilu_lower_jacobi_iterations != 5 ||
      tensor_igr_evidence.ilu_upper_jacobi_iterations != 5 ||
      tensor_igr_evidence.ilu_preconditioner_callbacks_lower_bound != 12 ||
      tensor_igr_evidence.ilu_preconditioner_callbacks_upper_bound != 18 ||
      tensor_igr_evidence.ilu_inner_iterations_lower_bound != 12 ||
      tensor_igr_evidence.ilu_inner_iterations_upper_bound != 18 ||
      tensor_igr_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 96 ||
      tensor_igr_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 144 ||
      !tensor_igr_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence common_ordinary_igr_evidence{};
  common_ordinary_igr_evidence.selected_engine = CudaEngineKind::ordinary;
  common_ordinary_igr_evidence.trace_policy =
      CudaTracePolicyKind::common_fast;
  if (!CudaInitializeIluEvidence(
          common_ordinary_igr_evidence,
      CudaTracePreconditionerKind::hypre_ilu0, true) ||
      !CudaAccumulateIluSolveEvidence(
          common_ordinary_igr_evidence, first) ||
      !CudaAccumulateIluSolveEvidence(
          common_ordinary_igr_evidence, reuse) ||
      common_ordinary_igr_evidence.ilu_factor_reuse_policy !=
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt ||
      common_ordinary_igr_evidence.ilu_triangular_solve_kind !=
          CudaIluTriangularSolveKind::iterative_jacobi ||
      common_ordinary_igr_evidence.ilu_lower_jacobi_iterations != 5 ||
      common_ordinary_igr_evidence.ilu_upper_jacobi_iterations != 5 ||
      common_ordinary_igr_evidence.ilu_factor_setups != 1 ||
      common_ordinary_igr_evidence.ilu_factor_reuses != 1 ||
      !common_ordinary_igr_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence incomplete_evidence{};
  incomplete_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (!CudaInitializeIluEvidence(
          incomplete_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, true))
    return false;
  auto malformed_bounds = first;
  malformed_bounds.ilu_preconditioner_callbacks_lower_bound = 11;
  malformed_bounds.ilu_inner_iterations_lower_bound = 11;
  malformed_bounds.ilu_accounting_complete = false;
  const bool malformed_numerically_ready =
      CudaTraceSolveReadyForTrueResidualVerification(malformed_bounds, 9);
  if (!CudaAccumulateIluSolveEvidence(
          incomplete_evidence, malformed_bounds) ||
      !malformed_numerically_ready ||
      incomplete_evidence.ilu_factor_setups != 1 ||
      incomplete_evidence.ilu_factor_reuses != 0 ||
      incomplete_evidence.ilu_factor_age_generations != 0 ||
      incomplete_evidence.ilu_preconditioner_callbacks_lower_bound != 0 ||
      incomplete_evidence.ilu_preconditioner_callbacks_upper_bound != 0 ||
      incomplete_evidence.ilu_inner_iterations_lower_bound != 0 ||
      incomplete_evidence.ilu_inner_iterations_upper_bound != 0 ||
      incomplete_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 0 ||
      incomplete_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 0 ||
      incomplete_evidence.ilu_accounting_complete)
    return false;
  if (!CudaAccumulateIluSolveEvidence(incomplete_evidence, reuse) ||
      incomplete_evidence.ilu_factor_setups != 1 ||
      incomplete_evidence.ilu_factor_reuses != 1 ||
      incomplete_evidence.ilu_factor_age_generations != 1 ||
      incomplete_evidence.ilu_preconditioner_callbacks_lower_bound != 0 ||
      incomplete_evidence.ilu_preconditioner_callbacks_upper_bound != 0 ||
      incomplete_evidence.ilu_inner_iterations_lower_bound != 0 ||
      incomplete_evidence.ilu_inner_iterations_upper_bound != 0 ||
      incomplete_evidence.ilu_jacobi_factor_spmv_updates_lower_bound != 0 ||
      incomplete_evidence.ilu_jacobi_factor_spmv_updates_upper_bound != 0 ||
      incomplete_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence hard_failure_evidence{};
  hard_failure_evidence.selected_engine = CudaEngineKind::tensor_core;
  if (!CudaInitializeIluEvidence(
          hard_failure_evidence,
          CudaTracePreconditionerKind::hypre_ilu0, true) ||
      CudaAccumulateIluSolveEvidence(hard_failure_evidence, reuse))
    return false;
  return hard_failure_evidence.ilu_factor_setups == 0 &&
      hard_failure_evidence.ilu_factor_reuses == 0 &&
      hard_failure_evidence.ilu_factor_age_generations == 0 &&
      hard_failure_evidence.ilu_preconditioner_callbacks_lower_bound == 0 &&
      hard_failure_evidence.ilu_preconditioner_callbacks_upper_bound == 0 &&
      hard_failure_evidence.ilu_inner_iterations_lower_bound == 0 &&
      hard_failure_evidence.ilu_inner_iterations_upper_bound == 0 &&
      hard_failure_evidence.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
      hard_failure_evidence.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
      !hard_failure_evidence.ilu_accounting_complete;
}

static_assert(IluWorkAccumulationPolicy(),
              "ILU work accumulation must preserve derived bounded work");

constexpr bool IluRefreshConsensusPolicy() noexcept
{
  using dg::hdg::runtime::cuda_w8a::detail::CudaIluRefreshDecisionRequiresConsensus;
  return !CudaIluRefreshDecisionRequiresConsensus(
             CudaIluFactorReusePolicy::not_applicable) &&
      !CudaIluRefreshDecisionRequiresConsensus(
          CudaIluFactorReusePolicy::refresh_each_trace_solve) &&
      CudaIluRefreshDecisionRequiresConsensus(
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt);
}

static_assert(IluRefreshConsensusPolicy(),
              "only retained attempt factors require branch consensus");

bool IluFactorStateLifecyclePolicy() noexcept
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  AttemptIdentity first{};
  first.sequence = 7;
  first.committed_step = 3;
  first.committed_generation = 5;
  first.dt = 1.0e-4;
  AttemptIdentity retry = first;
  retry.sequence = 8;
  retry.retry_index = 1;

  CudaIluAttemptFactorState state{};
  state.active = true;
  state.factor_ready = true;
  if (CudaBeginIluFactorAttempt(
          state, first,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
          101, 201, 0, 8, 3) || state.active || state.factor_ready)
    return false;
  if (!CudaBeginIluFactorAttempt(
          state, first,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
          101, 201, 301, 8, 3) ||
      !CudaIluFactorAttemptMatches(
          state, first,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt) ||
      CudaIluFactorAttemptMatches(
          state, retry,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt) ||
      CudaIluFactorAttemptMatches(
          state, first,
          CudaIluFactorReusePolicy::refresh_each_trace_solve) ||
      !CudaIluFactorRefreshRequired(state))
    return false;
  state.factor_ready = true;
  state.hypre_matrix_token = 401;
  state.ilu_handle_token = 501;
  state.factor_matrix_token = 601;
  state.factor_values_token = 701;
  state.factor_permutation_token = 801;
  state.jacobi_diagonal_vector_token = 901;
  state.jacobi_diagonal_values_token = 1001;
  state.factor_generation = 17;
  state.last_solve_generation = 17;
  if (CudaIluFactorRefreshRequired(state)) return false;
  {
    CudaIluFactorAttemptEndGuard end_factor_attempt(state);
    if (!state.active || !state.factor_ready) return false;
  }
  const AttemptIdentity empty_attempt{};
  if (state.active || state.factor_ready ||
      state.policy != CudaIluFactorReusePolicy::not_applicable ||
      !::dg::hdg::runtime::detail::SameAttempt(
          state.attempt, empty_attempt) ||
      state.matrix_object_token != 0 || state.hypre_matrix_token != 0 ||
      state.diag_offsets_token != 0 || state.diag_columns_token != 0 ||
      state.diag_values_token != 0 || state.offd_values_token != 0 ||
      state.offd_offsets_token != 0 || state.offd_columns_token != 0 ||
      state.ilu_handle_token != 0 || state.factor_matrix_token != 0 ||
      state.factor_offsets_token != 0 ||
      state.factor_columns_token != 0 ||
      state.factor_values_token != 0 ||
      state.factor_permutation_token != 0 ||
      state.native_inverse_diagonal_token != 0 ||
      state.jacobi_diagonal_vector_token != 0 ||
      state.jacobi_diagonal_values_token != 0 ||
      state.matrix_rows != 0 ||
      state.diag_nonzeros != 0 || state.offd_nonzeros != 0 ||
      state.factor_rows != 0 || state.factor_nonzeros != 0 ||
      state.factor_generation != 0 || state.last_solve_generation != 0)
    return false;

  CudaIluAttemptFactorState refresh_each{};
  if (!CudaBeginIluFactorAttempt(
          refresh_each, first,
          CudaIluFactorReusePolicy::refresh_each_trace_solve,
          102, 202, 0, 9, 0))
    return false;
  refresh_each.factor_ready = true;
  if (!CudaIluFactorRefreshRequired(refresh_each)) return false;

  if (!CudaBeginIluFactorAttempt(
          state, retry,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt,
          101, 201, 301, 8, 3))
    return false;
  return CudaIluFactorAttemptMatches(
             state, retry,
             CudaIluFactorReusePolicy::first_trace_solve_per_attempt) &&
      !CudaIluFactorAttemptMatches(
          state, first,
          CudaIluFactorReusePolicy::first_trace_solve_per_attempt) &&
      CudaIluFactorRefreshRequired(state) && !state.factor_ready &&
      state.hypre_matrix_token == 0 && state.ilu_handle_token == 0 &&
      state.diag_offsets_token == 0 && state.diag_columns_token == 0 &&
      state.diag_values_token == 201 &&
      state.offd_offsets_token == 0 && state.offd_columns_token == 0 &&
      state.offd_values_token == 301 &&
      state.factor_matrix_token == 0 && state.factor_values_token == 0 &&
      state.factor_offsets_token == 0 &&
      state.factor_columns_token == 0 &&
      state.factor_permutation_token == 0 &&
      state.native_inverse_diagonal_token == 0 &&
      state.jacobi_diagonal_vector_token == 0 &&
      state.jacobi_diagonal_values_token == 0 &&
      state.matrix_rows == 0 && state.factor_rows == 0 &&
      state.factor_nonzeros == 0 &&
      state.factor_generation == 0 && state.last_solve_generation == 0;
}

struct ModelTag { inline static constexpr std::string_view id = "euler.compressible.2d"; };
struct FormulationTag { inline static constexpr std::string_view id = "hdg.conventional"; };
struct TraceTag { inline static constexpr std::string_view id = "mfem.dg_interface.trace"; };
struct StabilizationTag { inline static constexpr std::string_view id = "tau.fixed"; };
struct TimeTag { inline static constexpr std::string_view id = "backward_euler"; };
struct RankOneTag { inline static constexpr std::string_view id = "cuda.rank1.one_gpu"; };
struct RankTwoTag { inline static constexpr std::string_view id = "cuda.rank2.one_gpu.host_staged"; };
struct PrecisionTag { inline static constexpr std::string_view id = "fp64"; };

template <typename Backend>
using Selection = StaticRunSelection<ModelTag, FormulationTag, TraceTag,
    StabilizationTag, TimeTag, Backend, PrecisionTag>;
template <typename Backend>
using Engine = PersistentCudaSmoothEulerEngine<Selection<Backend>,
    UmpireDeviceAllocatorPolicy, CudaTransportPolicy, CudaMpiCollectivePolicy>;

bool All(const bool local) noexcept
{
  int value = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && result == 1;
}

bool CompactRange(const CudaStepEvidence& evidence,
                  const CudaTransferLedgerRangeV1& range) noexcept
{
  const CudaTransferLedgerV1 ledger = CudaTransportPolicy::Ledger();
  return range.begin_ordinal != 0 && range.end_ordinal >= range.begin_ordinal &&
      range.ordered_digest != 0 && range.complete &&
      evidence.transfer_range.complete && evidence.transfers.complete &&
      evidence.transfers.event_count != 0 && ledger.totals.complete &&
      !ledger.overflow && ledger.observed_count >= range.end_ordinal;
}

bool SetupValid(const CudaRunEvidence& setup, const int ranks,
                const bool igr_active) noexcept
{
  return setup.status == CudaRunStatus::ready && setup.ranks == ranks &&
      setup.tensor_core.trace_preconditioner_configured &&
      setup.tensor_core.trace_preconditioner ==
          CudaTracePreconditionerForEngine(
              setup.tensor_core.selected_engine, igr_active) &&
      setup.mfem_device_valid && setup.hypre_cuda_umpire_device &&
      setup.same_physical_gpu && setup.no_host_solver_fallback &&
      setup.reuse.persistent_graph && setup.reuse.persistent_slots &&
      setup.reuse.persistent_workspace && (ranks == 1 || setup.host_staged_rank2);
}

CudaConfig Config(const int ranks, const bool functional = false) noexcept
{
  CudaConfig config{};
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  if (functional)
  {
    config.order = functional_polynomial_order;
    config.mesh_n = functional_elements_per_direction;
  }
  return config;
}

template <typename Backend>
bool Initialize(Engine<Backend>& engine, CudaRunEvidence& setup,
                const int ranks) noexcept
{
  const bool initialized = engine.Initialize(setup);
  return All(initialized && SetupValid(setup, ranks, false));
}

template <typename Backend>
bool Commit(Engine<Backend>& engine, const bool retry,
            CudaSerializedCheckpointV1& checkpoint) noexcept
{
  CudaPrecommitSnapshot snapshot{};
  CudaStepEvidence evidence{};
  const bool prepared = engine.PrepareStep(retry, snapshot, evidence);
  if (!All(prepared && snapshot.numerical_candidate_ready &&
           snapshot.products.CoupledAndOrdered() && evidence.no_fallback &&
           evidence.device_resident))
    return false;
  const bool staged = engine.StagePreparedCheckpoint(checkpoint, evidence);
  if (!All(staged)) { return false; }
  const bool marked = engine.MarkOutputStaged(checkpoint);
  if (!All(marked)) { return false; }
  const bool commit_prepared = engine.PrepareCommit(checkpoint, evidence);
  if (!All(commit_prepared)) { return false; }
  const bool provisional = engine.CommitPrepared(checkpoint, evidence);
  if (!All(provisional)) { return false; }
  const bool finalized = engine.FinalizeCommit(checkpoint, evidence);
  const bool valid = finalized && checkpoint.successful_commit &&
      evidence.committed && evidence.status == CudaRunStatus::committed &&
      evidence.after.step == snapshot.target.step &&
      evidence.after_interior_hash == checkpoint.descriptor.interior_hash &&
      evidence.after_trace_hash == checkpoint.descriptor.trace_hash &&
      CompactRange(evidence, checkpoint.staging_range);
  return All(valid);
}

template <typename Backend>
bool RestoreChecked(Engine<Backend>& engine,
                    const CudaSerializedCheckpointV1& checkpoint,
                    CudaStepEvidence& evidence) noexcept
{
  const bool restored = engine.Restore(checkpoint, evidence);
  const bool valid = restored && evidence.committed &&
      evidence.status == CudaRunStatus::committed &&
      evidence.contract_reason == FailureReason::none &&
      dg::hdg::runtime::detail::SameCoordinates(
          evidence.after, checkpoint.descriptor.coordinates) &&
      evidence.after_interior_hash == checkpoint.descriptor.interior_hash &&
      evidence.after_trace_hash == checkpoint.descriptor.trace_hash &&
      CompactRange(evidence, evidence.transfer_range);
  return All(valid);
}

bool SameCheckpointIdentity(const CudaSerializedCheckpointV1& left,
                            const CudaSerializedCheckpointV1& right) noexcept
{
  return left.local_mesh_hash == right.local_mesh_hash &&
      left.local_partition_hash == right.local_partition_hash &&
      left.local_topology_hash == right.local_topology_hash &&
      left.descriptor.mesh_hash == right.descriptor.mesh_hash &&
      left.descriptor.partition_hash == right.descriptor.partition_hash &&
      left.descriptor.topology_hash == right.descriptor.topology_hash;
}

bool SameCheckpointState(const CudaSerializedCheckpointV1& left,
                         const CudaSerializedCheckpointV1& right) noexcept
{
  return dg::hdg::runtime::detail::SameCoordinates(
             left.descriptor.coordinates, right.descriptor.coordinates) &&
      left.descriptor.interior_hash == right.descriptor.interior_hash &&
      left.descriptor.trace_hash == right.descriptor.trace_hash &&
      left.local_interior_hash == right.local_interior_hash &&
      left.local_trace_hash == right.local_trace_hash &&
      SameCheckpointIdentity(left, right);
}

template <typename Backend>
bool FreshCleanCommit(const int ranks) noexcept
{
  auto engine = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence setup{};
  CudaSerializedCheckpointV1 checkpoint{};
  const bool initialized = Initialize(*engine, setup, ranks);
  if (!initialized) { return false; }
  return Commit(*engine, false, checkpoint);
}

template <typename Backend>
bool LifecycleAndRecovery(const int ranks) noexcept
{
  auto engine = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence setup{};
  if (!Initialize(*engine, setup, ranks)) { return false; }
  CudaSerializedCheckpointV1 initial{};
  if (!Commit(*engine, false, initial)) { return false; }
  const auto* contract = engine->contract();
  if (!All(contract != nullptr)) { return false; }
  const StateCoordinates after_first = contract->coordinates();

  CudaPrecommitSnapshot prepared_snapshot{};
  CudaStepEvidence prepared_evidence{};
  CudaSerializedCheckpointV1 prepared{};
  const bool prepared_step = engine->PrepareStep(false, prepared_snapshot,
                                                  prepared_evidence);
  if (!All(prepared_step && prepared_snapshot.numerical_candidate_ready))
    return false;
  const bool prepared_stage = engine->StagePreparedCheckpoint(prepared,
                                                               prepared_evidence);
  if (!All(prepared_stage)) { return false; }
  const bool prepared_mark = engine->MarkOutputStaged(prepared);
  if (!All(prepared_mark)) { return false; }
  const bool prepared_commit = engine->PrepareCommit(prepared, prepared_evidence);
  if (!All(prepared_commit)) { return false; }
  const bool aborted = engine->AbortPrepared(prepared,
      FailureReason::backend_failure, prepared_evidence);
  const bool abort_valid = aborted && !prepared.successful_commit &&
      !prepared.candidate_frozen && !prepared.checkpoint_staged &&
      !prepared.output_staged && prepared_evidence.retry_available &&
      prepared_evidence.coordinates_preserved &&
      prepared_evidence.committed_bytes_preserved &&
      dg::hdg::runtime::detail::SameCoordinates(contract->coordinates(),
                                                 after_first);
  if (!All(abort_valid)) { return false; }

  CudaSerializedCheckpointV1 after_abort{};
  if (!Commit(*engine, true, after_abort)) { return false; }
  const StateCoordinates after_abort_commit = contract->coordinates();

  CudaPrecommitSnapshot provisional_snapshot{};
  CudaStepEvidence provisional_evidence{};
  CudaSerializedCheckpointV1 provisional{};
  const bool provisional_step = engine->PrepareStep(false, provisional_snapshot,
                                                     provisional_evidence);
  if (!All(provisional_step && provisional_snapshot.numerical_candidate_ready))
    return false;
  const bool provisional_stage = engine->StagePreparedCheckpoint(
      provisional, provisional_evidence);
  if (!All(provisional_stage)) { return false; }
  const bool provisional_mark = engine->MarkOutputStaged(provisional);
  if (!All(provisional_mark)) { return false; }
  const bool provisional_prepare = engine->PrepareCommit(provisional,
                                                          provisional_evidence);
  if (!All(provisional_prepare)) { return false; }
  const bool provisioned = engine->CommitPrepared(provisional,
                                                   provisional_evidence);
  if (!All(provisioned)) { return false; }
  const bool reverted = engine->RevertCommit(provisional,
      FailureReason::backend_failure, provisional_evidence);
  const bool revert_valid = reverted && !provisional.successful_commit &&
      !provisional.candidate_frozen && !provisional.checkpoint_staged &&
      !provisional.output_staged && provisional_evidence.retry_available &&
      provisional_evidence.coordinates_preserved &&
      provisional_evidence.committed_bytes_preserved &&
      dg::hdg::runtime::detail::SameCoordinates(contract->coordinates(),
                                                 after_abort_commit);
  if (!All(revert_valid)) { return false; }

  CudaSerializedCheckpointV1 final{};
  if (!Commit(*engine, true, final)) { return false; }
  engine.reset();

  // A checksum-rejected restore must leave the physical committed slot usable.
  auto reference = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence reference_setup{};
  CudaSerializedCheckpointV1 clean_reference{};
  if (!Initialize(*reference, reference_setup, ranks)) { return false; }
  if (!Commit(*reference, false, clean_reference)) { return false; }
  reference.reset();

  auto corrupt_reader = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence corrupt_setup{};
  if (!Initialize(*corrupt_reader, corrupt_setup, ranks)) { return false; }
  CudaSerializedCheckpointV1 corrupt = final;
  corrupt.interior[0] += 1.0 + (corrupt.interior[0] < 0.0
      ? -corrupt.interior[0] : corrupt.interior[0]);
  CudaStepEvidence corrupt_evidence{};
  const bool corrupt_return = corrupt_reader->Restore(corrupt, corrupt_evidence);
  const bool corrupt_rejected = !corrupt_return &&
      corrupt_evidence.status == CudaRunStatus::runtime_failure &&
      corrupt_evidence.contract_reason == FailureReason::restart_incompatible;
  if (!All(corrupt_rejected)) { return false; }
  CudaSerializedCheckpointV1 after_corrupt{};
  if (!Commit(*corrupt_reader, false, after_corrupt)) { return false; }
  const bool corrupt_bytes_preserved = SameCheckpointState(after_corrupt,
                                                            clean_reference);
  if (!All(corrupt_bytes_preserved)) { return false; }
  corrupt_reader.reset();

  // A consumed checkpoint is likewise rejected without modifying publication.
  auto subject = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence subject_setup{};
  if (!Initialize(*subject, subject_setup, ranks)) { return false; }
  CudaStepEvidence subject_restore{};
  if (!RestoreChecked(*subject, final, subject_restore)) { return false; }
  CudaStepEvidence consumed{};
  const bool consumed_return = subject->Restore(final, consumed);
  const bool consumed_rejected = !consumed_return &&
      consumed.status == CudaRunStatus::runtime_failure &&
      consumed.contract_reason == FailureReason::restart_incompatible;
  if (!All(consumed_rejected)) { return false; }
  CudaSerializedCheckpointV1 subject_next{};
  if (!Commit(*subject, false, subject_next)) { return false; }
  subject.reset();

  auto control = std::make_unique<Engine<Backend>>(Config(ranks));
  CudaRunEvidence control_setup{};
  if (!Initialize(*control, control_setup, ranks)) { return false; }
  CudaStepEvidence control_restore{};
  if (!RestoreChecked(*control, final, control_restore)) { return false; }
  CudaSerializedCheckpointV1 control_next{};
  if (!Commit(*control, false, control_next)) { return false; }
  return All(SameCheckpointState(subject_next, control_next));
}

template <typename Backend>
bool TokenFaults(const int ranks) noexcept
{
  const auto run = [&](const CudaTokenFault fault,
                       const FailureReason expected) noexcept
  {
    CudaConfig config = Config(ranks);
    config.injection.token_fault = fault;
    config.injection.source_rank = ranks - 1;
    auto engine = std::make_unique<Engine<Backend>>(config);
    CudaRunEvidence setup{};
    if (!Initialize(*engine, setup, ranks)) { return false; }
    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence evidence{};
    CudaSerializedCheckpointV1 checkpoint{};
    const bool prepared = engine->PrepareStep(false, snapshot, evidence);
    if (fault != CudaTokenFault::issue_matrix_before_residual)
    {
      if (!All(prepared && snapshot.numerical_candidate_ready &&
               !snapshot.products.CoupledAndOrdered()))
        return false;
      const bool staged = engine->StagePreparedCheckpoint(checkpoint, evidence);
      if (!All(staged)) { return false; }
      const bool marked = engine->MarkOutputStaged(checkpoint);
      if (!All(marked)) { return false; }
      const bool commit_prepared = engine->PrepareCommit(checkpoint, evidence);
      const auto* contract = engine->contract();
      const bool rejected = commit_prepared && contract != nullptr &&
          evidence.status == CudaRunStatus::retry_available &&
          evidence.contract_reason == expected && evidence.retry_available &&
          !evidence.committed && evidence.coordinates_preserved &&
          evidence.committed_bytes_preserved &&
          !checkpoint.candidate_frozen && !checkpoint.checkpoint_staged &&
          !checkpoint.output_staged && !checkpoint.successful_commit &&
          contract->telemetry().commits == 0;
      if (!All(rejected)) { return false; }
      engine.reset();
      return FreshCleanCommit<Backend>(ranks);
    }
    if (!All(prepared && snapshot.numerical_candidate_ready &&
             snapshot.products.CoupledAndOrdered()))
      return false;
    const bool staged = engine->StagePreparedCheckpoint(checkpoint, evidence);
    if (!All(staged)) { return false; }
    const bool marked = engine->MarkOutputStaged(checkpoint);
    if (!All(marked)) { return false; }
    const bool commit_prepared = engine->PrepareCommit(checkpoint, evidence);
    if (!All(commit_prepared)) { return false; }
    const bool provisioned = engine->CommitPrepared(checkpoint, evidence);
    if (!All(provisioned)) { return false; }
    const bool finalized = engine->FinalizeCommit(checkpoint, evidence);
    const auto* contract = engine->contract();
    const bool recovered = finalized && checkpoint.successful_commit &&
        contract != nullptr && contract->telemetry().commits == 1 &&
        contract->telemetry().out_of_order_product_issues_rejected == 1 &&
        contract->telemetry().last_attempt.validated_product_mask == 0x0fU &&
        CompactRange(evidence, checkpoint.staging_range);
    return All(recovered);
  };
  if (!run(CudaTokenFault::omit_solution, FailureReason::missing_product))
    return false;
  if (!run(CudaTokenFault::stale_attempt, FailureReason::missing_product))
    return false;
  return run(CudaTokenFault::issue_matrix_before_residual, FailureReason::none);
}

template <typename Backend>
bool FunctionalRestart(const int ranks) noexcept
{
  auto reference = std::make_unique<Engine<Backend>>(Config(ranks, true));
  CudaRunEvidence reference_setup{};
  if (!Initialize(*reference, reference_setup, ranks)) { return false; }
  CudaSerializedCheckpointV1 reference_final{};
  CudaSerializedCheckpointV1 bridge{};
  for (int step = 1; step <= functional_accepted_steps; ++step)
  {
    CudaSerializedCheckpointV1 checkpoint{};
    if (!Commit(*reference, false, checkpoint)) { return false; }
    if (step == functional_checkpoint_step) { bridge = checkpoint; }
    reference_final = checkpoint;
  }
  const auto* reference_contract = reference->contract();
  if (!All(reference_contract != nullptr &&
           reference_contract->telemetry().commits == functional_accepted_steps))
    return false;
  reference.reset();

  auto writer = std::make_unique<Engine<Backend>>(Config(ranks, true));
  CudaRunEvidence writer_setup{};
  if (!Initialize(*writer, writer_setup, ranks)) { return false; }
  CudaSerializedCheckpointV1 split{};
  for (int step = 1; step <= functional_checkpoint_step; ++step)
    if (!Commit(*writer, false, split)) { return false; }
  const auto* writer_contract = writer->contract();
  const bool checkpoint_agrees = writer_contract != nullptr &&
      writer_contract->telemetry().commits == functional_checkpoint_step &&
      SameCheckpointState(split, bridge);
  if (!All(checkpoint_agrees)) { return false; }
  writer.reset();

  auto reader = std::make_unique<Engine<Backend>>(Config(ranks, true));
  CudaRunEvidence reader_setup{};
  if (!Initialize(*reader, reader_setup, ranks)) { return false; }
  CudaStepEvidence restored{};
  if (!RestoreChecked(*reader, split, restored)) { return false; }
  CudaSerializedCheckpointV1 resumed_final{};
  for (int step = functional_checkpoint_step + 1;
       step <= functional_accepted_steps; ++step)
    if (!Commit(*reader, false, resumed_final)) { return false; }
  const auto* reader_contract = reader->contract();
  const bool same_final = reader_contract != nullptr &&
      reader_contract->telemetry().commits == functional_checkpoint_step &&
      SameCheckpointState(reference_final, resumed_final) &&
      reference_final.descriptor.coordinates.step == functional_accepted_steps &&
      reference_final.descriptor.coordinates.generation == functional_accepted_steps;
  return All(same_final);
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  int rank = 0;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const std::string_view mode = argc == 2 ? argv[1] : "";
  const bool known_mode = mode.empty() || mode == "--token-negatives" ||
      mode == "--functional";
  const bool policies = NativeBoundedResultPolicy() &&
      NativeWorkspaceRequirementsPolicy() &&
      IluFactorStateLifecyclePolicy();
  const bool lifecycle = known_mode && (ranks == 1
      ? (mode == "--functional" ? FunctionalRestart<RankOneTag>(ranks)
         : mode == "--token-negatives" ? TokenFaults<RankOneTag>(ranks)
         : LifecycleAndRecovery<RankOneTag>(ranks))
      : ranks == 2
      ? (mode == "--functional" ? FunctionalRestart<RankTwoTag>(ranks)
         : mode == "--token-negatives" ? TokenFaults<RankTwoTag>(ranks)
         : LifecycleAndRecovery<RankTwoTag>(ranks))
      : false);
  const bool native_device_fixtures = ranks != 1 || !mode.empty() ||
      (lifecycle && NativeTraceDeviceFixtures());
  const bool pass = policies && lifecycle && native_device_fixtures;
  if (!pass && rank == 0)
    std::cerr << "compact CUDA W8A audit failed\n";
  return pass ? 0 : 1;
}

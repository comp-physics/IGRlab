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

constexpr CudaTracePolicyDescriptor ResolveCudaTracePolicy(
    const dg::hdg::generic::CudaEngineKind engine, const bool igr_active,
    const CudaTracePolicyKind requested) noexcept
{
  return dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy(
      engine, igr_active, 1, dg::hdg::generic::minimum_polynomial_order,
      requested);
}

constexpr bool NativeTraceCooperativeSpecializationPolicy() noexcept
{
  using dg::hdg::runtime::cuda_w8a::detail::
      CudaNativeTraceCooperativeSupport;
  using dg::hdg::runtime::cuda_w8a::detail::
      CudaNativeTraceCooperativeSupportMatchesPreconditioner;
  CudaNativeTraceCooperativeSupport legacy{};
  legacy.fixed_sigma_jacobi = false;
  CudaNativeTraceCooperativeSupport fixed{};
  fixed.fixed_sigma_jacobi = true;
  return
      CudaNativeTraceCooperativeSupportMatchesPreconditioner(
          legacy,
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_fp64) &&
      !CudaNativeTraceCooperativeSupportMatchesPreconditioner(
          legacy,
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64) &&
      CudaNativeTraceCooperativeSupportMatchesPreconditioner(
          fixed,
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64) &&
      !CudaNativeTraceCooperativeSupportMatchesPreconditioner(
          fixed,
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_fp64);
}

static_assert(NativeTraceCooperativeSpecializationPolicy(),
              "native trace launch support must match the exact fixed-sigma "
              "kernel specialization");

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
  auto schwarz = native;
  schwarz.configured_ilu = false;
  schwarz.configured_element_schwarz = true;
  schwarz.trace_preconditioner =
      CudaTracePreconditionerKind::native_element_schwarz_fp64;
  const bool schwarz_zero_iteration_is_admitted =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(schwarz, 8);
  schwarz.iterations = 1;
  const bool schwarz_zero_application_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(schwarz, 8);
  schwarz.element_schwarz_preconditioner_applications = 1;
  const bool schwarz_positive_application_is_admitted =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(schwarz, 8);
  auto mgr = diagnostics;
  mgr.trace_solver =
      dg::hdg::runtime::cuda_w8a::detail::CudaTraceSolverKind::
          hypre_fgmres_mgr_rank1;
  mgr.configured_ilu = false;
  mgr.configured_mgr = true;
  mgr.trace_preconditioner =
      CudaTracePreconditionerKind::hypre_mgr_field_split;
  mgr.mgr_setup_calls = 1;
  mgr.mgr_f_ilu_setups = 1;
  const bool mgr_solver_is_admitted =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(mgr, 8);
  auto mgr_retry = mgr;
  mgr_retry.mgr_setup_calls = 0;
  mgr_retry.mgr_f_ilu_setups = 0;
  mgr_retry.mgr_setup_reused = true;
  const bool mgr_retry_is_admitted =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(mgr_retry, 8);
  mgr_retry.mgr_setup_reused = false;
  const bool mgr_skipped_setup_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(mgr_retry, 8);
  mgr_retry = mgr;
  mgr_retry.mgr_setup_reused = true;
  const bool mgr_false_reuse_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(mgr_retry, 8);
  mgr.iterations = 1;
  const bool mgr_zero_application_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(mgr, 8);
  mgr.mgr_preconditioner_applications = 1;
  mgr.mgr_f_ilu_applications = 1;
  const bool mgr_positive_application_is_admitted =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(mgr, 8);
  mgr.iterations = 0;
  mgr.configured_diagonal = true;
  const bool mgr_mixed_preconditioner_is_rejected =
      !dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(mgr, 8);
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
      schwarz_zero_iteration_is_admitted &&
      schwarz_zero_application_is_rejected &&
      schwarz_positive_application_is_admitted &&
      mgr_solver_is_admitted && mgr_retry_is_admitted &&
      mgr_skipped_setup_is_rejected && mgr_false_reuse_is_rejected &&
      mgr_zero_application_is_rejected &&
      mgr_positive_application_is_admitted &&
      mgr_mixed_preconditioner_is_rejected &&
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
             CudaEngineKind::ordinary, 1, false, true, true, true,
             CudaTracePolicyKind::engine_default) ==
          CudaTraceSolverKind::hypre_parcsr_gmres &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::ordinary, 2, true, true, true, true,
          CudaTracePolicyKind::engine_default) ==
          CudaTraceSolverKind::hypre_host_staged_rank2 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, true, true, true,
          CudaTracePolicyKind::engine_default) ==
          CudaTraceSolverKind::native_rank1_fp64 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, false, false, false,
          CudaTracePolicyKind::engine_default) ==
          CudaTraceSolverKind::unconfigured &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, true, true, true, true,
          CudaTracePolicyKind::engine_default) ==
          CudaTraceSolverKind::hypre_host_staged_rank2 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, false, true, true, true,
          CudaTracePolicyKind::engine_default) ==
          CudaTraceSolverKind::unconfigured &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, true, true, true,
          CudaTracePolicyKind::equilibrated_mgr) ==
          CudaTraceSolverKind::hypre_fgmres_mgr_rank1 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::ordinary, 1, false, true, true, true,
          CudaTracePolicyKind::equilibrated_mgr) ==
          CudaTraceSolverKind::unconfigured &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, true, true, true, true,
          CudaTracePolicyKind::equilibrated_mgr) ==
          CudaTraceSolverKind::unconfigured &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, true, true, true,
          CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64) ==
          CudaTraceSolverKind::native_rank1_fp64 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, true, true, true, true,
          CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64) ==
          CudaTraceSolverKind::unconfigured &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, true, true, true,
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_equilibrated_field_split_fp64) ==
          CudaTraceSolverKind::native_rank1_fp64 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, true, true, true, true,
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_equilibrated_field_split_fp64) ==
          CudaTraceSolverKind::unconfigured &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, true, true, true,
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64) ==
          CudaTraceSolverKind::native_rank1_fp64 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, true, true, true, true,
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64) ==
          CudaTraceSolverKind::unconfigured &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, true, true, true,
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64) ==
          CudaTraceSolverKind::native_rank1_fp64 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, true, true, true, true,
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64) ==
          CudaTraceSolverKind::unconfigured &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 1, false, true, true, true,
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64) ==
          CudaTraceSolverKind::native_rank1_fp64 &&
      ResolveCudaTraceSolverKind(
          CudaEngineKind::tensor_core, 2, true, true, true, true,
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64) ==
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

class RankOneMgrStateOwner
{
 public:
  RankOneMgrStateOwner() = default;
  RankOneMgrStateOwner(const RankOneMgrStateOwner&) = delete;
  RankOneMgrStateOwner& operator=(const RankOneMgrStateOwner&) = delete;
  ~RankOneMgrStateOwner()
  {
    dg::hdg::runtime::cuda_w8a::detail::CudaDestroyRankOneMgrState(state);
  }

  dg::hdg::runtime::cuda_w8a::detail::CudaRankOneMgrState state{};
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

bool ValuesClose(const double first, const double second,
                 const double multiplier = 256.0) noexcept
{
  return std::isfinite(first) && std::isfinite(second) &&
      std::abs(first - second) <= multiplier *
          std::numeric_limits<double>::epsilon() *
          std::max({1.0, std::abs(first), std::abs(second)});
}

NativeCsrFixture MakeMgrFieldSplitMatrix(const double perturbation)
{
  constexpr int scalar_trace_dofs = 2;
  constexpr int components = 5;
  constexpr int rows = scalar_trace_dofs * components;
  std::vector<double> dense(static_cast<std::size_t>(rows) * rows, 0.0);
  for (int row = 0; row < rows; ++row)
  {
    const int component = row / scalar_trace_dofs;
    const int scalar = row % scalar_trace_dofs;
    dense[static_cast<std::size_t>(row) * rows + row] =
        (2.5 + 0.4 * component + 0.2 * scalar) *
        (1.0 + (perturbation - 1.0) * 0.04 * (row + 1));
    for (int column = row + 1; column < rows; ++column)
    {
      const int column_component = column / scalar_trace_dofs;
      const int column_scalar = column % scalar_trace_dofs;
      double coupling = 0.0;
      if (component == column_component)
        coupling = -0.08;
      else if (scalar == column_scalar &&
               (component == components - 1 ||
                column_component == components - 1))
        coupling = 0.04;
      else if (column == row + 1)
        coupling = -0.015;
      coupling *= 1.0 + (perturbation - 1.0) *
          0.02 * (row + column + 1);
      dense[static_cast<std::size_t>(row) * rows + column] = coupling;
      dense[static_cast<std::size_t>(column) * rows + row] = coupling;
    }
  }
  return DenseToDiagonalFirstCsr(dense, rows);
}

std::unique_ptr<mfem::HypreParMatrix> MakeRankOneHypreMatrix(
    const NativeCsrFixture& fixture)
{
  if (!NativeCsrFixtureIsValid(fixture)) return nullptr;
  try
  {
    std::vector<HYPRE_BigInt> global_columns(fixture.columns.begin(),
                                              fixture.columns.end());
    std::array<HYPRE_BigInt, 2> partition{{0, fixture.rows}};
    return std::make_unique<mfem::HypreParMatrix>(
        MPI_COMM_SELF, fixture.rows, fixture.rows, fixture.rows,
        fixture.offsets.data(), global_columns.data(), fixture.values.data(),
        partition.data(), partition.data());
  }
  catch (...) { return nullptr; }
}

bool CopyDeviceValues(const double* const source, const std::size_t count,
                      std::vector<double>& values) noexcept
{
  values.assign(count, 0.0);
  return count == 0 || (source != nullptr &&
      cudaMemcpy(values.data(), source, count * sizeof(double),
                 cudaMemcpyDeviceToHost) == cudaSuccess);
}

bool RankOneEquilibratedMgrFixture()
{
  using dg::hdg::generic::CudaEngineKind;
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  constexpr int scalar_trace_dofs = 2;
  constexpr int rows = scalar_trace_dofs * igr_trace_components;
  constexpr double relative_tolerance = 1.0e-12;
  constexpr int maximum_iterations = 50;
  constexpr int krylov_dimension = 10;

  const NativeCsrFixture first_fixture = MakeMgrFieldSplitMatrix(1.0);
  const NativeCsrFixture second_fixture = MakeMgrFieldSplitMatrix(1.125);
  const std::vector<double> first_rhs{
      1.0, -0.5, 0.25, 1.5, -1.0, 0.75, 2.0, -1.25, 0.5, 1.0};
  const std::vector<double> second_rhs{
      -0.75, 1.25, 0.5, -1.0, 1.75, 0.25, -0.5, 2.0, 1.0, -1.5};
  if (!NativeCsrFixtureIsValid(first_fixture) ||
      !NativeCsrFixtureIsValid(second_fixture) ||
      first_fixture.rows != rows || second_fixture.rows != rows)
    return false;

  auto first_original = MakeRankOneHypreMatrix(first_fixture);
  auto second_original = MakeRankOneHypreMatrix(second_fixture);
  if (!first_original || !second_original) return false;
  auto first_equilibrated =
      std::make_unique<mfem::HypreParMatrix>(*first_original);
  auto second_equilibrated =
      std::make_unique<mfem::HypreParMatrix>(*second_original);
  auto first_original_rhs =
      std::make_unique<mfem::HypreParVector>(*first_original);
  auto first_equilibrated_rhs =
      std::make_unique<mfem::HypreParVector>(*first_equilibrated);
  auto first_solution =
      std::make_unique<mfem::HypreParVector>(*first_equilibrated);
  auto second_original_rhs =
      std::make_unique<mfem::HypreParVector>(*second_original);
  auto second_equilibrated_rhs =
      std::make_unique<mfem::HypreParVector>(*second_equilibrated);
  auto second_solution =
      std::make_unique<mfem::HypreParVector>(*second_equilibrated);

  std::copy(first_rhs.begin(), first_rhs.end(),
            first_original_rhs->HostWrite());
  std::copy(second_rhs.begin(), second_rhs.end(),
            second_original_rhs->HostWrite());
  *first_equilibrated_rhs = 0.0;
  *first_solution = 0.0;
  *second_equilibrated_rhs = 0.0;
  *second_solution = 0.0;

  RankOneMgrStateOwner owner;
  auto& state = owner.state;
  const CudaTracePolicyDescriptor policy = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::equilibrated_mgr);
  if (!CudaTracePolicyDescriptorIsExact(policy) ||
      !CudaConfigureRankOneMgrState(
          state, MPI_COMM_SELF, rows, scalar_trace_dofs,
          relative_tolerance, maximum_iterations, krylov_dimension) ||
      !state.configured || state.fgmres == nullptr || state.mgr == nullptr ||
      state.f_ilu == nullptr || state.coarse_amg == nullptr ||
      state.configured_relative_tolerance != relative_tolerance ||
      state.point_markers.size() != static_cast<std::size_t>(rows))
    return false;
  for (int row = 0; row < rows; ++row)
    if (state.point_markers[static_cast<std::size_t>(row)] !=
            row / scalar_trace_dofs)
      return false;
  if (state.coarse_point_counts[0] != 1 ||
      state.coarse_point_indexes[0] != igr_trace_components - 1)
    return false;
  const HYPRE_Solver f_ilu_identity = state.f_ilu;

  const auto run_generation = [&state, &policy](
      mfem::HypreParMatrix& original,
      mfem::HypreParMatrix& equilibrated,
      mfem::HypreParVector& original_rhs,
      mfem::HypreParVector& equilibrated_rhs,
      mfem::HypreParVector& solution,
      const NativeCsrFixture& fixture,
      const std::vector<double>& expected_rhs,
      const std::uint64_t generation,
      CudaSolveDiagnostics& diagnostics) noexcept
  {
    NativeDeviceAllocation<double> left_scale;
    NativeDeviceAllocation<double> right_scale;
    NativeDeviceAllocation<double> row_norm;
    NativeDeviceAllocation<double> column_norm;
    NativeDeviceAllocation<double> control;
    NativeDeviceAllocation<int> status;
    if (!left_scale.Allocate(static_cast<std::size_t>(fixture.rows)) ||
        !right_scale.Allocate(static_cast<std::size_t>(fixture.rows)) ||
        !row_norm.Allocate(static_cast<std::size_t>(fixture.rows)) ||
        !column_norm.Allocate(static_cast<std::size_t>(fixture.rows)) ||
        !control.Allocate(cuda_rank1_equilibration_control_doubles) ||
        !status.Allocate(1))
      return false;

    original.HypreRead();
    equilibrated.HypreReadWrite();
    original_rhs.HypreRead();
    equilibrated_rhs.HypreReadWrite();
    solution.HypreReadWrite();
    CudaTrueMatrixDeviceView original_view{};
    CudaTrueMatrixDeviceView equilibrated_view{};
    const double* const original_rhs_device =
        static_cast<const mfem::Vector&>(original_rhs).Read();
    double* const equilibrated_rhs_device =
        static_cast<mfem::Vector&>(equilibrated_rhs).ReadWrite();
    double* const solution_device =
        static_cast<mfem::Vector&>(solution).ReadWrite();
    if (!CudaGetTrueMatrixDeviceView(original, original_view) ||
        !CudaGetTrueMatrixDeviceView(equilibrated, equilibrated_view) ||
        original_view.rows != fixture.rows ||
        original_view.offd_nonzeros != 0 ||
        equilibrated_view.offd_nonzeros != 0 ||
        original_rhs_device == nullptr || equilibrated_rhs_device == nullptr ||
        solution_device == nullptr)
      return false;

    std::vector<double> original_values_before;
    std::vector<double> original_rhs_before;
    if (!CopyDeviceValues(
            original_view.diag_values,
            static_cast<std::size_t>(original_view.diag_nonzeros),
            original_values_before) ||
        !CopyDeviceValues(original_rhs_device,
                          static_cast<std::size_t>(fixture.rows),
                          original_rhs_before))
      return false;
    const bool launched = CudaLaunchBuildRankOneEquilibratedSystem(
        original_view, equilibrated_view, original_rhs_device,
        equilibrated_rhs_device, left_scale.data(), right_scale.data(),
        row_norm.data(), column_norm.data(), control.data(), status.data(),
        cuda_rank1_equilibration_passes, nullptr);
    if (!launched || cudaDeviceSynchronize() != cudaSuccess) return false;

    int status_value = -1;
    std::vector<double> left(static_cast<std::size_t>(fixture.rows));
    std::vector<double> right(static_cast<std::size_t>(fixture.rows));
    std::vector<double> observed_control(
        cuda_rank1_equilibration_control_doubles);
    std::vector<double> scaled_rhs(static_cast<std::size_t>(fixture.rows));
    std::vector<double> scaled_values;
    std::vector<double> original_values_after;
    std::vector<double> original_rhs_after;
    if (!status.CopyOneTo(status_value) || status_value != 0 ||
        !left_scale.CopyTo(left) || !right_scale.CopyTo(right) ||
        !control.CopyTo(observed_control) ||
        !CopyDeviceValues(equilibrated_rhs_device, scaled_rhs.size(),
                          scaled_rhs) ||
        !CopyDeviceValues(
            equilibrated_view.diag_values,
            static_cast<std::size_t>(equilibrated_view.diag_nonzeros),
            scaled_values) ||
        !CopyDeviceValues(
            original_view.diag_values,
            static_cast<std::size_t>(original_view.diag_nonzeros),
            original_values_after) ||
        !CopyDeviceValues(original_rhs_device,
                          static_cast<std::size_t>(fixture.rows),
                          original_rhs_after) ||
        original_values_before != original_values_after ||
        original_rhs_before != original_rhs_after ||
        original_rhs_after != expected_rhs)
      return false;

    std::vector<int> offsets(static_cast<std::size_t>(fixture.rows) + 1);
    std::vector<int> columns(original_values_before.size());
    if (cudaMemcpy(offsets.data(), original_view.diag_offsets,
                   offsets.size() * sizeof(int), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(columns.data(), original_view.diag_columns,
                   columns.size() * sizeof(int), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
      return false;
    for (int row = 0; row < fixture.rows; ++row)
    {
      if (!(left[static_cast<std::size_t>(row)] >=
                cuda_rank1_equilibration_minimum_scale &&
            left[static_cast<std::size_t>(row)] <=
                cuda_rank1_equilibration_maximum_scale &&
            right[static_cast<std::size_t>(row)] >=
                cuda_rank1_equilibration_minimum_scale &&
            right[static_cast<std::size_t>(row)] <=
                cuda_rank1_equilibration_maximum_scale) ||
          !ValuesClose(
              scaled_rhs[static_cast<std::size_t>(row)],
              left[static_cast<std::size_t>(row)] *
                  expected_rhs[static_cast<std::size_t>(row)]))
        return false;
      for (int entry = offsets[static_cast<std::size_t>(row)];
           entry < offsets[static_cast<std::size_t>(row) + 1]; ++entry)
      {
        const int column = columns[static_cast<std::size_t>(entry)];
        if (column < 0 || column >= fixture.rows ||
            !ValuesClose(
                scaled_values[static_cast<std::size_t>(entry)],
                left[static_cast<std::size_t>(row)] *
                    original_values_before[static_cast<std::size_t>(entry)] *
                    right[static_cast<std::size_t>(column)]))
          return false;
      }
    }
    const bool nonunit_scale = std::any_of(
        left.begin(), left.end(), [](const double value)
        { return !ValuesClose(value, 1.0); }) ||
        std::any_of(right.begin(), right.end(), [](const double value)
        { return !ValuesClose(value, 1.0); });
    if (!nonunit_scale) return false;
    if (!ValuesClose(observed_control[0],
                     *std::min_element(left.begin(), left.end())) ||
        !ValuesClose(observed_control[1],
                     *std::max_element(left.begin(), left.end())) ||
        !ValuesClose(observed_control[2],
                     *std::min_element(right.begin(), right.end())) ||
        !ValuesClose(observed_control[3],
                     *std::max_element(right.begin(), right.end())))
      return false;

    equilibrated.ResetTranspose();
    equilibrated.HypreRead();
    if (!CudaSolveRankOneMgr(
            state, equilibrated, equilibrated_rhs, solution, policy,
            relative_tolerance, maximum_iterations, krylov_dimension,
            generation, diagnostics) ||
        diagnostics.mgr_setup_calls != 1 ||
        diagnostics.mgr_setup_reused ||
        diagnostics.mgr_f_ilu_setups != 1 ||
        diagnostics.iterations <= 0 ||
        diagnostics.mgr_preconditioner_applications == 0 ||
        diagnostics.mgr_f_ilu_applications !=
            diagnostics.mgr_preconditioner_applications ||
        !CudaTraceSolveReadyForTrueResidualVerification(
            diagnostics, maximum_iterations))
      return false;

    // The one-shot true-residual tightening path changes only the outer
    // FlexGMRES tolerance.  It retains this exact scaled system and the
    // current-generation MGR hierarchy/F-ILU factor.
    const double tightening_tolerance = relative_tolerance * 0.1;
    const std::uint64_t setup_before_retry = state.setup_calls;
    const std::uint64_t applications_before_retry =
        state.preconditioner_applications;
    CudaSolveDiagnostics retry_diagnostics{};
    if (!CudaSolveRankOneMgr(
            state, equilibrated, equilibrated_rhs, solution, policy,
            tightening_tolerance, maximum_iterations, krylov_dimension,
            generation, retry_diagnostics, true) ||
        state.configured_relative_tolerance != tightening_tolerance ||
        state.setup_calls != setup_before_retry ||
        retry_diagnostics.mgr_setup_calls != 0 ||
        !retry_diagnostics.mgr_setup_reused ||
        retry_diagnostics.mgr_f_ilu_setups != 0 ||
        retry_diagnostics.iterations <= 0 ||
        retry_diagnostics.mgr_preconditioner_applications == 0 ||
        retry_diagnostics.mgr_f_ilu_applications !=
            retry_diagnostics.mgr_preconditioner_applications ||
        state.preconditioner_applications <= applications_before_retry ||
        !CudaTraceSolveReadyForTrueResidualVerification(
            retry_diagnostics, maximum_iterations))
      return false;

    std::vector<double> equilibrated_solution(
        static_cast<std::size_t>(fixture.rows));
    if (!CopyDeviceValues(solution_device, equilibrated_solution.size(),
                          equilibrated_solution) ||
        !CudaLaunchUnscaleRankOneSolution(
            solution_device, right_scale.data(), fixture.rows, nullptr) ||
        cudaDeviceSynchronize() != cudaSuccess)
      return false;
    std::vector<double> unscaled_solution;
    if (!CopyDeviceValues(solution_device,
                          static_cast<std::size_t>(fixture.rows),
                          unscaled_solution))
      return false;
    for (int row = 0; row < fixture.rows; ++row)
      if (!ValuesClose(
              unscaled_solution[static_cast<std::size_t>(row)],
              right[static_cast<std::size_t>(row)] *
                  equilibrated_solution[static_cast<std::size_t>(row)]))
        return false;

    double residual_norm = std::numeric_limits<double>::infinity();
    if (!CsrResidualNorm(
            fixture, expected_rhs, unscaled_solution, residual_norm) ||
        residual_norm > 1.0e-8 * std::max(1.0, VectorNorm(expected_rhs)))
      return false;
    auto original_residual =
        std::make_unique<mfem::HypreParVector>(original);
    original.Mult(solution, *original_residual);
    *original_residual -= original_rhs;
    const double rebuilt_original_residual = original_residual->Norml2();
    return std::isfinite(rebuilt_original_residual) &&
        rebuilt_original_residual <=
            1.0e-8 * std::max(1.0, VectorNorm(expected_rhs)) &&
        ValuesClose(rebuilt_original_residual, residual_norm, 8192.0);
  };

  CudaSolveDiagnostics first_diagnostics{};
  if (!run_generation(
          *first_original, *first_equilibrated, *first_original_rhs,
          *first_equilibrated_rhs, *first_solution, first_fixture, first_rhs,
          41, first_diagnostics) ||
      state.setup_calls != 1 || !state.has_setup_generation ||
      state.last_setup_matrix_generation != 41 ||
      state.setup_generation_pending || state.f_ilu != f_ilu_identity)
    return false;
  const std::uint64_t first_applications =
      state.preconditioner_applications;

  // Retry authority is bound to the exact retained matrix/RHS/solution
  // objects as well as the generation; a same-number foreign system fails
  // before changing setup or application aggregates.
  CudaSolveDiagnostics foreign_retry_diagnostics{};
  if (CudaSolveRankOneMgr(
          state, *second_equilibrated, *second_equilibrated_rhs,
          *second_solution, policy, relative_tolerance, maximum_iterations,
          krylov_dimension, 41, foreign_retry_diagnostics, true) ||
      state.setup_calls != 1 ||
      state.preconditioner_applications != first_applications)
    return false;

  // The setup generation is a strict matrix-generation authority. Reusing a
  // generation fails closed before HYPRE setup and cannot change aggregates.
  CudaSolveDiagnostics duplicate_diagnostics{};
  if (CudaSolveRankOneMgr(
          state, *first_equilibrated, *first_equilibrated_rhs,
          *first_solution, policy, relative_tolerance, maximum_iterations,
          krylov_dimension, 41, duplicate_diagnostics, false) ||
      state.setup_calls != 1 ||
      state.preconditioner_applications != first_applications)
    return false;

  CudaSolveDiagnostics second_diagnostics{};
  if (!run_generation(
          *second_original, *second_equilibrated, *second_original_rhs,
          *second_equilibrated_rhs, *second_solution, second_fixture,
          second_rhs, 42, second_diagnostics) ||
      state.setup_calls != 2 || !state.has_setup_generation ||
      state.last_setup_matrix_generation != 42 ||
      state.setup_generation_pending || state.f_ilu != f_ilu_identity ||
      state.preconditioner_applications <= first_applications)
    return false;

  CudaDestroyRankOneMgrState(state);
  return !state.configured && state.fgmres == nullptr &&
      state.mgr == nullptr && state.f_ilu == nullptr &&
      state.coarse_amg == nullptr && state.point_markers.empty() &&
      state.setup_calls == 0 && state.preconditioner_applications == 0 &&
      state.setup_matrix_token == 0 && state.setup_rhs_token == 0 &&
      state.setup_solution_token == 0 &&
      !std::isfinite(state.configured_relative_tolerance) &&
      !state.has_setup_generation && !state.setup_generation_pending;
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
          rows, fixture.policy.krylov_dimension, fixture.igr, false);
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

bool NativeElementSchwarzDeviceFixtureForOrder(const int order)
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  if (order < 1 || order > 3) return false;
  const int rows = 20 * (order + 1);
  const int elements = 1;
  const int patch_entries = rows * rows;
  const int incidence_slots = 2 * rows;
  std::vector<double> identity(static_cast<std::size_t>(patch_entries), 0.0);
  std::vector<int> offsets(static_cast<std::size_t>(rows) + 1);
  std::vector<int> columns(static_cast<std::size_t>(rows));
  std::vector<double> diagonal(static_cast<std::size_t>(rows), 1.0);
  std::vector<double> right_hand_side(static_cast<std::size_t>(rows));
  std::vector<int> signed_dofs(static_cast<std::size_t>(rows));
  std::vector<int> first_sources(static_cast<std::size_t>(patch_entries));
  std::vector<int> second_sources(
      static_cast<std::size_t>(patch_entries), -1);
  std::vector<int> incidence(static_cast<std::size_t>(incidence_slots), -1);
  for (int row = 0; row < rows; ++row)
  {
    identity[static_cast<std::size_t>(row) * rows + row] = 1.0;
    offsets[static_cast<std::size_t>(row)] = row;
    columns[static_cast<std::size_t>(row)] = row;
    right_hand_side[static_cast<std::size_t>(row)] =
        0.25 + static_cast<double>(row) / rows;
    signed_dofs[static_cast<std::size_t>(row)] = row;
    incidence[2 * static_cast<std::size_t>(row)] = row;
  }
  offsets[static_cast<std::size_t>(rows)] = rows;
  for (int entry = 0; entry < patch_entries; ++entry)
    first_sources[static_cast<std::size_t>(entry)] = entry;

  NativeDeviceAllocation<double> schur;
  NativeDeviceAllocation<double> patch_matrix;
  NativeDeviceAllocation<double> patch_lu;
  NativeDeviceAllocation<double> patch_rhs;
  NativeDeviceAllocation<double> patch_solution;
  NativeDeviceAllocation<double> status;
  NativeDeviceAllocation<int> signed_element_trace;
  NativeDeviceAllocation<int> first_source;
  NativeDeviceAllocation<int> second_source;
  NativeDeviceAllocation<int> pivots;
  NativeDeviceAllocation<int> true_incidence;
  if (!schur.CopyFrom(identity) ||
      !patch_matrix.Allocate(patch_entries) ||
      !patch_lu.Allocate(patch_entries) ||
      !patch_rhs.Allocate(rows) || !patch_solution.Allocate(rows) ||
      !status.Allocate(elements) ||
      !signed_element_trace.CopyFrom(signed_dofs) ||
      !first_source.CopyFrom(first_sources) ||
      !second_source.CopyFrom(second_sources) || !pivots.Allocate(rows) ||
      !true_incidence.CopyFrom(incidence))
    return false;

  CudaDeviceView setup_view{};
  setup_view.schur = schur.data();
  setup_view.status = status.data();
  setup_view.signed_element_trace = signed_element_trace.data();
  setup_view.element_schwarz_patch_matrix = patch_matrix.data();
  setup_view.element_schwarz_patch_lu = patch_lu.data();
  setup_view.element_schwarz_rhs = patch_rhs.data();
  setup_view.element_schwarz_solution = patch_solution.data();
  setup_view.element_schwarz_first_source = first_source.data();
  setup_view.element_schwarz_second_source = second_source.data();
  setup_view.element_schwarz_signed_true_dof =
      signed_element_trace.data();
  setup_view.element_schwarz_true_incidence_slots = true_incidence.data();
  setup_view.element_schwarz_pivots = pivots.data();
  CudaMeshShape shape{};
  shape.elements = elements;
  shape.trace_dofs = rows;
  shape.trace_components = 5;
  shape.order = order;
  shape.igr_active = true;
  if (!CudaLaunchSetupNativeElementSchwarz(
          setup_view, shape,
          CudaTracePreconditionerKind::native_element_schwarz_fp64) ||
      cudaDeviceSynchronize() != cudaSuccess)
    return false;
  double setup_status = 0.0;
  if (!status.CopyOneTo(setup_status) || setup_status != 1.0) return false;

  NativeTraceSolvePolicy policy{};
  policy.rank_count = 1;
  policy.krylov_dimension = 4;
  policy.maximum_iterations = 8;
  policy.relative_tolerance = 1.0e-12;
  const NativeTraceCooperativeSupport support =
      QueryNativeRankOneTraceCooperativeSupport(
          rows, policy.krylov_dimension, true, false);
  const NativeTraceWorkspaceRequirements requirements =
      QueryNativeTraceWorkspaceRequirements(
          rows, policy.krylov_dimension, support.cooperative_blocks, true);
  if (!support.supported || !requirements.valid) return false;

  NativeDeviceAllocation<int> matrix_offsets;
  NativeDeviceAllocation<int> matrix_columns;
  NativeDeviceAllocation<double> matrix_values;
  NativeDeviceAllocation<double> device_rhs;
  NativeDeviceAllocation<double> solution;
  NativeDeviceAllocation<double> basis;
  NativeDeviceAllocation<double> vectors;
  NativeDeviceAllocation<double> reduction;
  NativeDeviceAllocation<double> small;
  NativeDeviceAllocation<int> permutation_marks;
  NativeDeviceAllocation<NativeTraceSolveControl> control;
  if (!matrix_offsets.CopyFrom(offsets) ||
      !matrix_columns.CopyFrom(columns) ||
      !matrix_values.CopyFrom(diagonal) ||
      !device_rhs.CopyFrom(right_hand_side) || !solution.Allocate(rows) ||
      !basis.Allocate(requirements.basis_doubles) ||
      !vectors.Allocate(requirements.vector_doubles) ||
      !reduction.Allocate(requirements.reduction_doubles) ||
      !small.Allocate(requirements.small_doubles) ||
      !permutation_marks.Allocate(requirements.permutation_integers) ||
      !control.Allocate(1))
    return false;

  NativeTraceSystemView system{};
  system.matrix = {matrix_offsets.data(), matrix_columns.data(),
                   matrix_values.data(), rows, rows, rows};
  system.rhs = device_rhs.data();
  system.solution = solution.data();
  NativeTracePreconditionerView preconditioner{};
  preconditioner.kind =
      NativeTracePreconditionerKind::element_overlap_schwarz_fp64;
  preconditioner.element_schwarz = {
      patch_matrix.data(), patch_lu.data(), pivots.data(),
      signed_element_trace.data(), true_incidence.data(), patch_rhs.data(),
      patch_solution.data(), elements, rows, rows, 1, 1};
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
  if (!LaunchNativeRankOneTraceSolve(
          system, preconditioner, workspace, policy, support, nullptr) ||
      cudaDeviceSynchronize() != cudaSuccess)
    return false;
  std::vector<double> host_solution(static_cast<std::size_t>(rows));
  NativeTraceSolveControl observation{};
  if (!solution.CopyTo(host_solution) || !control.CopyOneTo(observation) ||
      observation.status != NativeTraceSolveStatus::converged ||
      observation.preconditioner_applications <= 0)
    return false;
  for (int row = 0; row < rows; ++row)
    if (std::abs(host_solution[static_cast<std::size_t>(row)] -
                 right_hand_side[static_cast<std::size_t>(row)]) > 1.0e-11)
      return false;

  // The bounded equilibrated field-split ablation reuses the same exact
  // patch/topology storage.  Identity blocks make its lower multiplicative
  // action exact and independently exercise the packed Euler/sigma factors,
  // two-sided scales, solve dispatch, and metadata kind.
  if (!CudaLaunchSetupNativeElementSchwarz(
          setup_view, shape,
          CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_fp64) ||
      cudaMemset(solution.data(), 0,
                 static_cast<std::size_t>(rows) * sizeof(double)) !=
          cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return false;
  preconditioner.kind = NativeTracePreconditionerKind::
      element_overlap_schwarz_equilibrated_field_split_fp64;
  if (!LaunchNativeRankOneTraceSolve(
          system, preconditioner, workspace, policy, support, nullptr) ||
      cudaDeviceSynchronize() != cudaSuccess ||
      !solution.CopyTo(host_solution) || !control.CopyOneTo(observation) ||
      observation.status != NativeTraceSolveStatus::converged ||
      observation.preconditioner_applications <= 0)
    return false;
  for (int row = 0; row < rows; ++row)
    if (std::abs(host_solution[static_cast<std::size_t>(row)] -
                 right_hand_side[static_cast<std::size_t>(row)]) > 1.0e-11)
      return false;

  // The combined policy is a separately typed operator/preconditioner pair.
  // It reuses the exact element Schur block for the global action and the
  // already-built equilibrated factors for the preconditioner.  Cross-pairing
  // either operator kind must fail before a kernel is queued.
  system.operator_kind = NativeTraceOperatorKind::
      element_matrix_free_equilibrated_field_split_fp64;
  system.element_schur = schur.data();
  if (cudaMemset(solution.data(), 0,
                 static_cast<std::size_t>(rows) * sizeof(double)) !=
          cudaSuccess ||
      !LaunchNativeRankOneTraceSolve(
          system, preconditioner, workspace, policy, support, nullptr) ||
      cudaDeviceSynchronize() != cudaSuccess ||
      !solution.CopyTo(host_solution) || !control.CopyOneTo(observation) ||
      observation.status != NativeTraceSolveStatus::converged ||
      observation.preconditioner_applications <= 0 ||
      observation.operator_applications <= 0)
    return false;
  for (int row = 0; row < rows; ++row)
    if (std::abs(host_solution[static_cast<std::size_t>(row)] -
                 right_hand_side[static_cast<std::size_t>(row)]) > 1.0e-11)
      return false;

  // The cooperative variant has a distinct setup/preconditioner/operator
  // triplet. Exercise a genuinely pivoted, coupled lower field system so its
  // setup-time permutation folding and cooperative Euler-to-sigma action are
  // not admitted by an identity-only test.
  const NativeElementSchwarzFieldSplitLayout warp_layout =
      NativeElementSchwarzIgrFieldSplitLayout(order);
  if (!warp_layout.warp_valid) return false;
  std::vector<double> coupled_patch(
      static_cast<std::size_t>(patch_entries), 0.0);
  const auto set_packed = [&](const int packed_row, const int packed_column,
                              const double value) {
    const int original_row = NativeElementSchwarzIgrFieldSplitPatchRow(
        order, packed_row);
    const int original_column = NativeElementSchwarzIgrFieldSplitPatchRow(
        order, packed_column);
    coupled_patch[static_cast<std::size_t>(original_row) * rows +
                  original_column] = value;
  };
  for (int packed = 0; packed < warp_layout.euler_rows; ++packed)
  {
    set_packed(packed, packed, 2.0 + 0.01 * packed);
    if (packed > 1) set_packed(packed, packed - 1, 0.02);
  }
  set_packed(0, 0, 0.1);
  set_packed(0, 1, 1.0);
  set_packed(1, 0, 10.0);
  set_packed(1, 1, 1.0);
  for (int sigma = 0; sigma < warp_layout.sigma_rows; ++sigma)
  {
    const int packed = warp_layout.euler_rows + sigma;
    set_packed(packed, packed, 1.5 + 0.02 * sigma);
    if (sigma > 1) set_packed(packed, packed - 1, -0.015);
    for (int euler = 0; euler < warp_layout.euler_rows; ++euler)
      set_packed(packed, euler,
                 0.002 * (1 + (sigma + 2 * euler) % 5));
  }
  set_packed(warp_layout.euler_rows, warp_layout.euler_rows, 0.2);
  set_packed(warp_layout.euler_rows, warp_layout.euler_rows + 1, 1.0);
  set_packed(warp_layout.euler_rows + 1, warp_layout.euler_rows, 8.0);
  set_packed(warp_layout.euler_rows + 1,
             warp_layout.euler_rows + 1, 0.5);

  std::vector<double> reference_solution(static_cast<std::size_t>(rows));
  std::vector<double> coupled_rhs(static_cast<std::size_t>(rows), 0.0);
  for (int row = 0; row < rows; ++row)
    reference_solution[static_cast<std::size_t>(row)] =
        0.2 + static_cast<double>((7 * row) % 17) / 19.0;
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < rows; ++column)
      coupled_rhs[static_cast<std::size_t>(row)] = std::fma(
          coupled_patch[static_cast<std::size_t>(row) * rows + column],
          reference_solution[static_cast<std::size_t>(column)],
          coupled_rhs[static_cast<std::size_t>(row)]);
  if (cudaMemcpy(schur.data(), coupled_patch.data(),
                 static_cast<std::size_t>(patch_entries) * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(device_rhs.data(), coupled_rhs.data(),
                 static_cast<std::size_t>(rows) * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      !CudaLaunchSetupNativeElementSchwarz(
          setup_view, shape,
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_fp64) ||
      cudaMemset(solution.data(), 0,
                 static_cast<std::size_t>(rows) * sizeof(double)) !=
          cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess)
    return false;
  std::vector<int> host_pivots(static_cast<std::size_t>(rows));
  if (!pivots.CopyTo(host_pivots)) return false;
  bool euler_pivoted = false;
  bool sigma_pivoted = false;
  for (int packed = 0; packed < warp_layout.euler_rows; ++packed)
    euler_pivoted = euler_pivoted ||
        host_pivots[static_cast<std::size_t>(packed)] != packed;
  for (int sigma = 0; sigma < warp_layout.sigma_rows; ++sigma)
    sigma_pivoted = sigma_pivoted ||
        host_pivots[static_cast<std::size_t>(
            warp_layout.euler_rows + sigma)] != sigma;
  if (!euler_pivoted || !sigma_pivoted) return false;
  preconditioner.kind = NativeTracePreconditionerKind::
      element_overlap_schwarz_warp_equilibrated_field_split_fp64;
  system.operator_kind = NativeTraceOperatorKind::
      element_matrix_free_warp_equilibrated_field_split_fp64;
  if (!LaunchNativeRankOneTraceSolve(
          system, preconditioner, workspace, policy, support, nullptr) ||
      cudaDeviceSynchronize() != cudaSuccess ||
      !solution.CopyTo(host_solution) || !control.CopyOneTo(observation) ||
      observation.status != NativeTraceSolveStatus::converged ||
      observation.preconditioner_applications <= 0 ||
      observation.operator_applications <= 0)
    return false;
  for (int row = 0; row < rows; ++row)
    if (std::abs(host_solution[static_cast<std::size_t>(row)] -
                 reference_solution[static_cast<std::size_t>(row)]) >
        1.0e-10)
      return false;

  // Corrupting the setup-only reciprocal is detected by the bounded upfront
  // preconditioner validation before GMRES can consume it.
  const double invalid_inverse = 0.0;
  if (cudaMemcpy(
          patch_lu.data() + warp_layout.warp_inverse_diagonal_offset,
          &invalid_inverse, sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemset(solution.data(), 0,
                 static_cast<std::size_t>(rows) * sizeof(double)) !=
          cudaSuccess ||
      !LaunchNativeRankOneTraceSolve(
          system, preconditioner, workspace, policy, support, nullptr) ||
      cudaDeviceSynchronize() != cudaSuccess ||
      !control.CopyOneTo(observation) ||
      observation.status != NativeTraceSolveStatus::invalid_preconditioner ||
      observation.failure_mask == 0)
    return false;

  auto mismatched = system;
  mismatched.operator_kind = NativeTraceOperatorKind::element_matrix_free_fp64;
  if (LaunchNativeRankOneTraceSolve(
          mismatched, preconditioner, workspace, policy, support, nullptr))
    return false;
  auto full_schwarz = preconditioner;
  full_schwarz.kind =
      NativeTracePreconditionerKind::element_overlap_schwarz_fp64;
  if (LaunchNativeRankOneTraceSolve(
          system, full_schwarz, workspace, policy, support, nullptr))
    return false;
  auto scalar_field_split = preconditioner;
  scalar_field_split.kind = NativeTracePreconditionerKind::
      element_overlap_schwarz_equilibrated_field_split_fp64;
  if (LaunchNativeRankOneTraceSolve(
          system, scalar_field_split, workspace, policy, support, nullptr))
    return false;
  mismatched.operator_kind = static_cast<NativeTraceOperatorKind>(255);
  return !LaunchNativeRankOneTraceSolve(
      mismatched, preconditioner, workspace, policy, support, nullptr);
}

bool NativeElementSchwarzDeviceFixture()
{
  for (int order = 1; order <= 3; ++order)
    if (!NativeElementSchwarzDeviceFixtureForOrder(order)) return false;
  return true;
}

bool NativeFixedSigmaJacobiOrderFixture(const int order,
                                        const bool cooperative_warp)
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  if (order < 1 || order > 3) return false;
  const CudaTracePreconditionerKind setup_preconditioner = cooperative_warp
      ? CudaTracePreconditionerKind::
            native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64
      : CudaTracePreconditionerKind::
            native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64;
  const NativeTracePreconditionerKind native_preconditioner = cooperative_warp
      ? NativeTracePreconditionerKind::
            element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64
      : NativeTracePreconditionerKind::
            element_overlap_schwarz_equilibrated_field_split_sigma_jacobi3_fp64;
  const NativeTraceOperatorKind native_operator = cooperative_warp
      ? NativeTraceOperatorKind::
            element_matrix_free_warp_equilibrated_field_split_fp64
      : NativeTraceOperatorKind::
            element_matrix_free_equilibrated_field_split_fp64;
  const int scalar_face_rows = order + 1;
  const int patch_rows = 20 * scalar_face_rows;
  const int face_rows = 5 * scalar_face_rows;
  const int scalar_patch_rows = 4 * scalar_face_rows;
  const int scalar_true_rows = 2 * scalar_patch_rows - scalar_face_rows;
  const int true_rows = 5 * scalar_true_rows;
  constexpr int elements = 2;
  const int patch_entries = patch_rows * patch_rows;
  const int element_slots = elements * patch_rows;

  const auto is_sigma_row = [scalar_face_rows](const int row)
  {
    return (row % (5 * scalar_face_rows)) / scalar_face_rows == 4;
  };
  const auto paired_euler_row = [scalar_face_rows](const int sigma_row)
  {
    const int face_rows = 5 * scalar_face_rows;
    const int face = sigma_row / face_rows;
    const int scalar = sigma_row % scalar_face_rows;
    return ((face + 1) % 4) * face_rows + scalar;
  };
  const auto paired_sigma_row = [scalar_face_rows](const int sigma_row)
  {
    const int face_rows = 5 * scalar_face_rows;
    const int face = sigma_row / face_rows;
    const int scalar = sigma_row % scalar_face_rows;
    return ((face + 1) % 4) * face_rows + 4 * scalar_face_rows + scalar;
  };
  const auto diagonal_value = [face_rows](const int row)
  {
    return 1.25 + 0.005 * static_cast<double>(row % face_rows);
  };
  const auto coupling_value = [scalar_face_rows](const int sigma_row)
  {
    return 0.125 + 0.01 *
        static_cast<double>(sigma_row % scalar_face_rows);
  };
  const auto sigma_coupling_value = [scalar_face_rows](const int sigma_row)
  {
    return 0.055 + 0.005 *
        static_cast<double>(sigma_row % scalar_face_rows);
  };

  // Two elements share one complete face. Each scalar sigma row couples to
  // an Euler row and a different sigma row on the next local face. The
  // sigma-sigma face cycle crosses warp boundaries for every p1--p3 fixture
  // and makes every correction depend on the previous sweep's complete sigma
  // vector, so this distinguishes synchronous Jacobi/double buffering from an
  // in-place Gauss-Seidel update or a missing inter-warp barrier. A shared sigma
  // row also sees distinct neighbors from the two patches. No individual
  // principal patch is the global inverse and all three global sigma
  // corrections are nonzero. Element zero uses negative trace orientation, so
  // reconstructing the true RHS from its first incidence also requires the
  // retained sign.
  std::vector<double> element_schur(
      static_cast<std::size_t>(elements) * patch_entries, 0.0);
  for (int element = 0; element < elements; ++element)
  {
    double* const schur = element_schur.data() +
        static_cast<std::size_t>(element) * patch_entries;
    for (int row = 0; row < patch_rows; ++row)
    {
      schur[static_cast<std::size_t>(row) * patch_rows + row] =
          diagonal_value(row);
      if (is_sigma_row(row))
      {
        schur[static_cast<std::size_t>(row) * patch_rows +
              paired_euler_row(row)] = coupling_value(row);
        schur[static_cast<std::size_t>(row) * patch_rows +
              paired_sigma_row(row)] = sigma_coupling_value(row);
      }
    }
  }

  std::vector<int> true_dof(static_cast<std::size_t>(element_slots));
  std::vector<int> signed_dofs(static_cast<std::size_t>(element_slots));
  for (int row = 0; row < patch_rows; ++row)
  {
    const int face = row / face_rows;
    const int face_row = row % face_rows;
    const int component = face_row / scalar_face_rows;
    const int scalar = face_row % scalar_face_rows;
    const int local_scalar = face * scalar_face_rows + scalar;
    const int first_truth =
        component * scalar_true_rows + local_scalar;
    const int second_truth = component * scalar_true_rows +
        3 * scalar_face_rows + local_scalar;
    true_dof[static_cast<std::size_t>(row)] = first_truth;
    signed_dofs[static_cast<std::size_t>(row)] = -1 - first_truth;
    true_dof[static_cast<std::size_t>(patch_rows + row)] = second_truth;
    signed_dofs[static_cast<std::size_t>(patch_rows + row)] = second_truth;
  }

  std::vector<double> right_hand_side(
      static_cast<std::size_t>(true_rows));
  for (int row = 0; row < true_rows; ++row)
    right_hand_side[static_cast<std::size_t>(row)] =
        0.75 + 0.01 * static_cast<double>(row);

  std::vector<int> incidence(
      2 * static_cast<std::size_t>(true_rows), -1);
  std::vector<int> sigma_class(static_cast<std::size_t>(true_rows), -1);
  for (int slot = 0; slot < element_slots; ++slot)
  {
    const int truth = true_dof[static_cast<std::size_t>(slot)];
    int* const row_incidence = incidence.data() + 2 * truth;
    if (row_incidence[0] < 0)
      row_incidence[0] = slot;
    else if (row_incidence[1] < 0)
      row_incidence[1] = slot;
    else
      return false;
    const int local_sigma = is_sigma_row(slot % patch_rows) ? 1 : 0;
    if (sigma_class[static_cast<std::size_t>(truth)] >= 0 &&
        sigma_class[static_cast<std::size_t>(truth)] != local_sigma)
      return false;
    sigma_class[static_cast<std::size_t>(truth)] = local_sigma;
  }
  for (int row = 0; row < true_rows; ++row)
    if (sigma_class[static_cast<std::size_t>(row)] !=
        (row >= 4 * scalar_true_rows ? 1 : 0))
      return false;

  std::vector<int> local_for_truth(
      static_cast<std::size_t>(elements) * true_rows, -1);
  for (int slot = 0; slot < element_slots; ++slot)
    local_for_truth[static_cast<std::size_t>(slot / patch_rows) * true_rows +
                    true_dof[static_cast<std::size_t>(slot)]] =
        slot % patch_rows;

  std::vector<double> global_matrix(
      static_cast<std::size_t>(true_rows) * true_rows, 0.0);
  for (int element = 0; element < elements; ++element)
    for (int row = 0; row < patch_rows; ++row)
      for (int column = 0; column < patch_rows; ++column)
      {
        const int true_row = true_dof[
            static_cast<std::size_t>(element) * patch_rows + row];
        const int true_column = true_dof[
            static_cast<std::size_t>(element) * patch_rows + column];
        // Both orientations are uniform within their element, so the signed
        // row/column product is positive for this deliberately simple map.
        global_matrix[static_cast<std::size_t>(true_row) * true_rows +
                      true_column] += element_schur[
            static_cast<std::size_t>(element) * patch_entries +
            static_cast<std::size_t>(row) * patch_rows + column];
      }
  constexpr int warp_rows = 32;
  bool has_sigma_cross_warp_coupling = false;
  for (int row = 0; row < true_rows; ++row)
    for (int column = 0; column < true_rows; ++column)
      has_sigma_cross_warp_coupling = has_sigma_cross_warp_coupling ||
          (row != column &&
           row / warp_rows != column / warp_rows &&
           sigma_class[static_cast<std::size_t>(row)] == 1 &&
           sigma_class[static_cast<std::size_t>(column)] == 1 &&
           global_matrix[static_cast<std::size_t>(row) * true_rows +
                         column] != 0.0);
  if (!has_sigma_cross_warp_coupling) return false;
  const NativeCsrFixture matrix =
      DenseToDiagonalFirstCsr(global_matrix, true_rows);
  if (!NativeCsrFixtureIsValid(matrix)) return false;

  std::vector<int> first_sources(
      static_cast<std::size_t>(elements) * patch_entries);
  std::vector<int> second_sources(first_sources.size(), -1);
  for (int target_element = 0; target_element < elements; ++target_element)
    for (int row = 0; row < patch_rows; ++row)
      for (int column = 0; column < patch_rows; ++column)
      {
        const int true_row = true_dof[
            static_cast<std::size_t>(target_element) * patch_rows + row];
        const int true_column = true_dof[
            static_cast<std::size_t>(target_element) * patch_rows + column];
        const std::size_t target =
            static_cast<std::size_t>(target_element) * patch_entries +
            static_cast<std::size_t>(row) * patch_rows + column;
        int source_count = 0;
        for (int source_element = 0; source_element < elements;
             ++source_element)
        {
          const int source_row = local_for_truth[
              static_cast<std::size_t>(source_element) * true_rows +
              true_row];
          const int source_column = local_for_truth[
              static_cast<std::size_t>(source_element) * true_rows +
              true_column];
          if (source_row < 0 || source_column < 0) continue;
          const int source = source_element * patch_entries +
              source_row * patch_rows + source_column;
          if (source_count == 0)
            first_sources[target] = source;
          else if (source_count == 1)
            second_sources[target] = source;
          else
            return false;
          ++source_count;
        }
        if (source_count == 0) return false;
      }

  NativeDeviceAllocation<double> schur;
  NativeDeviceAllocation<double> patch_matrix;
  NativeDeviceAllocation<double> patch_lu;
  NativeDeviceAllocation<double> patch_rhs;
  NativeDeviceAllocation<double> patch_solution;
  NativeDeviceAllocation<double> status;
  NativeDeviceAllocation<int> signed_element_trace;
  NativeDeviceAllocation<int> first_source;
  NativeDeviceAllocation<int> second_source;
  NativeDeviceAllocation<int> pivots;
  NativeDeviceAllocation<int> true_incidence;
  if (!schur.CopyFrom(element_schur) ||
      !patch_matrix.Allocate(
          static_cast<std::size_t>(elements) * patch_entries) ||
      !patch_lu.Allocate(
          static_cast<std::size_t>(elements) * patch_entries) ||
      !patch_rhs.Allocate(static_cast<std::size_t>(element_slots)) ||
      !patch_solution.Allocate(static_cast<std::size_t>(element_slots)) ||
      !status.Allocate(elements) ||
      !signed_element_trace.CopyFrom(signed_dofs) ||
      !first_source.CopyFrom(first_sources) ||
      !second_source.CopyFrom(second_sources) ||
      !pivots.Allocate(static_cast<std::size_t>(element_slots)) ||
      !true_incidence.CopyFrom(incidence))
    return false;

  CudaDeviceView setup_view{};
  setup_view.schur = schur.data();
  setup_view.status = status.data();
  setup_view.signed_element_trace = signed_element_trace.data();
  setup_view.element_schwarz_patch_matrix = patch_matrix.data();
  setup_view.element_schwarz_patch_lu = patch_lu.data();
  setup_view.element_schwarz_rhs = patch_rhs.data();
  setup_view.element_schwarz_solution = patch_solution.data();
  setup_view.element_schwarz_first_source = first_source.data();
  setup_view.element_schwarz_second_source = second_source.data();
  setup_view.element_schwarz_signed_true_dof =
      signed_element_trace.data();
  setup_view.element_schwarz_true_incidence_slots = true_incidence.data();
  setup_view.element_schwarz_pivots = pivots.data();
  CudaMeshShape shape{};
  shape.elements = elements;
  shape.trace_dofs = patch_rows;
  shape.trace_components = 5;
  shape.true_trace_count = true_rows;
  shape.order = order;
  shape.igr_active = true;
  if (!CudaLaunchSetupNativeElementSchwarz(
          setup_view, shape, setup_preconditioner) ||
      cudaDeviceSynchronize() != cudaSuccess)
    return false;
  std::vector<double> host_status(static_cast<std::size_t>(elements));
  if (!status.CopyTo(host_status)) return false;
  for (const double value : host_status)
    if (value != 1.0) return false;

  NativeTraceSolvePolicy policy{};
  policy.rank_count = 1;
  policy.krylov_dimension = 1;
  policy.maximum_iterations = 1;
  policy.relative_tolerance = 1.0e-14;
  const NativeTraceCooperativeSupport support =
      QueryNativeRankOneTraceCooperativeSupport(
          true_rows, policy.krylov_dimension, true, true);
  const NativeTraceWorkspaceRequirements requirements =
      QueryNativeTraceWorkspaceRequirements(
          true_rows, policy.krylov_dimension,
          support.cooperative_blocks, true);
  if (!support.supported || !support.fixed_sigma_jacobi ||
      !requirements.valid)
    return false;

  NativeDeviceAllocation<int> device_matrix_offsets;
  NativeDeviceAllocation<int> device_matrix_columns;
  NativeDeviceAllocation<double> device_matrix_values;
  NativeDeviceAllocation<double> device_rhs;
  NativeDeviceAllocation<double> solution;
  NativeDeviceAllocation<double> basis;
  NativeDeviceAllocation<double> vectors;
  NativeDeviceAllocation<double> reduction;
  NativeDeviceAllocation<double> small;
  NativeDeviceAllocation<int> permutation_marks;
  NativeDeviceAllocation<NativeTraceSolveControl> control;
  if (!device_matrix_offsets.CopyFrom(matrix.offsets) ||
      !device_matrix_columns.CopyFrom(matrix.columns) ||
      !device_matrix_values.CopyFrom(matrix.values) ||
      !device_rhs.CopyFrom(right_hand_side) ||
      !solution.Allocate(static_cast<std::size_t>(true_rows)) ||
      !basis.Allocate(requirements.basis_doubles) ||
      !vectors.Allocate(requirements.vector_doubles) ||
      !reduction.Allocate(requirements.reduction_doubles) ||
      !small.Allocate(requirements.small_doubles) ||
      !permutation_marks.Allocate(requirements.permutation_integers) ||
      !control.Allocate(1) ||
      cudaMemset(solution.data(), 0,
                 static_cast<std::size_t>(true_rows) * sizeof(double)) !=
          cudaSuccess)
    return false;

  NativeTraceSystemView system{};
  system.matrix = {
      device_matrix_offsets.data(), device_matrix_columns.data(),
      device_matrix_values.data(), true_rows, true_rows,
      static_cast<int>(matrix.values.size())};
  system.rhs = device_rhs.data();
  system.solution = solution.data();
  system.operator_kind = native_operator;
  system.element_schur = schur.data();
  NativeTracePreconditionerView preconditioner{};
  preconditioner.kind = native_preconditioner;
  preconditioner.element_schwarz = {
      patch_matrix.data(), patch_lu.data(), pivots.data(),
      signed_element_trace.data(), true_incidence.data(), patch_rhs.data(),
      patch_solution.data(), elements, patch_rows, true_rows, 1, 2};
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
  auto mismatched_support = support;
  mismatched_support.fixed_sigma_jacobi = false;
  if (LaunchNativeRankOneTraceSolve(
          system, preconditioner, workspace, policy, mismatched_support,
          nullptr))
    return false;
  auto base_preconditioner = preconditioner;
  base_preconditioner.kind = cooperative_warp
      ? NativeTracePreconditionerKind::
            element_overlap_schwarz_warp_equilibrated_field_split_fp64
      : NativeTracePreconditionerKind::
            element_overlap_schwarz_equilibrated_field_split_fp64;
  if (LaunchNativeRankOneTraceSolve(
          system, base_preconditioner, workspace, policy, support, nullptr))
    return false;
  if (!LaunchNativeRankOneTraceSolve(
          system, preconditioner, workspace, policy, support, nullptr) ||
      cudaDeviceSynchronize() != cudaSuccess)
    return false;

  std::vector<double> host_solution(static_cast<std::size_t>(true_rows));
  NativeTraceSolveControl observation{};
  if (!solution.CopyTo(host_solution) || !control.CopyOneTo(observation) ||
      observation.status != NativeTraceSolveStatus::maximum_iterations ||
      observation.preconditioner_applications != 2)
    return false;

  // Independently construct each exact principal-patch solve on the CPU,
  // scatter with the signed inverse-multiplicity map, and then apply three
  // synchronous damped global sigma-row corrections. Euler entries remain
  // unchanged on every sweep. A one-iteration right-GMRES result is analytic:
  // x = h/(h^2+||w_perp||^2) * M^{-1}b.
  std::vector<double> predictor(static_cast<std::size_t>(true_rows), 0.0);
  std::vector<int> multiplicity(static_cast<std::size_t>(true_rows), 0);
  for (int element = 0; element < elements; ++element)
  {
    std::vector<double> principal(
        static_cast<std::size_t>(patch_entries), 0.0);
    std::vector<double> local_rhs(static_cast<std::size_t>(patch_rows));
    for (int row = 0; row < patch_rows; ++row)
    {
      const int true_row = true_dof[
          static_cast<std::size_t>(element) * patch_rows + row];
      const double sign = element == 0 ? -1.0 : 1.0;
      local_rhs[static_cast<std::size_t>(row)] = sign *
          right_hand_side[static_cast<std::size_t>(true_row)];
      for (int column = 0; column < patch_rows; ++column)
      {
        const int true_column = true_dof[
            static_cast<std::size_t>(element) * patch_rows + column];
        principal[static_cast<std::size_t>(row) * patch_rows + column] =
            sign * sign * global_matrix[
                static_cast<std::size_t>(true_row) * true_rows + true_column];
      }
    }
    std::vector<double> local_solution;
    if (!SolveDenseReference(principal, local_rhs, local_solution))
      return false;
    for (int row = 0; row < patch_rows; ++row)
    {
      const int truth = true_dof[
          static_cast<std::size_t>(element) * patch_rows + row];
      const double sign = element == 0 ? -1.0 : 1.0;
      predictor[static_cast<std::size_t>(truth)] +=
          sign * local_solution[static_cast<std::size_t>(row)];
      ++multiplicity[static_cast<std::size_t>(truth)];
    }
  }
  for (int row = 0; row < true_rows; ++row)
  {
    if (multiplicity[static_cast<std::size_t>(row)] < 1 ||
        multiplicity[static_cast<std::size_t>(row)] > 2)
      return false;
    predictor[static_cast<std::size_t>(row)] /=
        static_cast<double>(multiplicity[static_cast<std::size_t>(row)]);
  }
  std::vector<double> preconditioned = predictor;
  std::vector<double> next = predictor;
  bool nonzero_sigma_correction = false;
  for (int sweep = 0; sweep < native_trace_sigma_jacobi_sweeps; ++sweep)
  {
    next = preconditioned;
    for (int row = 0; row < true_rows; ++row)
    {
      if (sigma_class[static_cast<std::size_t>(row)] != 1) continue;
      double action = 0.0;
      for (int column = 0; column < true_rows; ++column)
        action += global_matrix[
            static_cast<std::size_t>(row) * true_rows + column] *
            preconditioned[static_cast<std::size_t>(column)];
      const double diagonal = global_matrix[
          static_cast<std::size_t>(row) * true_rows + row];
      const double value =
          preconditioned[static_cast<std::size_t>(row)] +
          native_trace_sigma_jacobi_damping *
              (right_hand_side[static_cast<std::size_t>(row)] - action) /
                  diagonal;
      nonzero_sigma_correction = nonzero_sigma_correction ||
          std::abs(value -
                   preconditioned[static_cast<std::size_t>(row)]) > 1.0e-8;
      next[static_cast<std::size_t>(row)] = value;
    }
    preconditioned.swap(next);
    for (int row = 0; row < true_rows; ++row)
      if (sigma_class[static_cast<std::size_t>(row)] == 0 &&
          preconditioned[static_cast<std::size_t>(row)] !=
              predictor[static_cast<std::size_t>(row)])
        return false;
  }
  if (!nonzero_sigma_correction) return false;

  // A lexicographic in-place sigma sweep is deliberately not equivalent for
  // this cross-coupled fixture. Keep this assertion so removing either the
  // sigma-sigma entries or the device double buffer cannot make the CPU
  // reference accidentally accept Gauss-Seidel semantics.
  std::vector<double> in_place = predictor;
  for (int sweep = 0; sweep < native_trace_sigma_jacobi_sweeps; ++sweep)
    for (int row = 0; row < true_rows; ++row)
    {
      if (sigma_class[static_cast<std::size_t>(row)] != 1) continue;
      double action = 0.0;
      for (int column = 0; column < true_rows; ++column)
        action += global_matrix[
            static_cast<std::size_t>(row) * true_rows + column] *
            in_place[static_cast<std::size_t>(column)];
      const double diagonal = global_matrix[
          static_cast<std::size_t>(row) * true_rows + row];
      in_place[static_cast<std::size_t>(row)] +=
          native_trace_sigma_jacobi_damping *
          (right_hand_side[static_cast<std::size_t>(row)] - action) /
              diagonal;
    }
  bool differs_from_in_place = false;
  for (int row = 0; row < true_rows; ++row)
    if (sigma_class[static_cast<std::size_t>(row)] == 1)
      differs_from_in_place = differs_from_in_place ||
          std::abs(in_place[static_cast<std::size_t>(row)] -
                   preconditioned[static_cast<std::size_t>(row)]) > 1.0e-10;
  if (!differs_from_in_place) return false;

  double rhs_norm_squared = 0.0;
  std::vector<double> operator_action(static_cast<std::size_t>(true_rows));
  for (int row = 0; row < true_rows; ++row)
  {
    rhs_norm_squared += right_hand_side[static_cast<std::size_t>(row)] *
        right_hand_side[static_cast<std::size_t>(row)];
    double value = 0.0;
    for (int column = 0; column < true_rows; ++column)
      value += global_matrix[
          static_cast<std::size_t>(row) * true_rows + column] *
          preconditioned[static_cast<std::size_t>(column)];
    operator_action[static_cast<std::size_t>(row)] = value;
  }
  double hessenberg = 0.0;
  for (int row = 0; row < true_rows; ++row)
    hessenberg += right_hand_side[static_cast<std::size_t>(row)] *
        operator_action[static_cast<std::size_t>(row)] / rhs_norm_squared;
  double perpendicular_norm_squared = 0.0;
  for (int row = 0; row < true_rows; ++row)
  {
    const double value = operator_action[static_cast<std::size_t>(row)] -
        hessenberg * right_hand_side[static_cast<std::size_t>(row)];
    perpendicular_norm_squared += value * value / rhs_norm_squared;
  }
  const double denominator = hessenberg * hessenberg +
      perpendicular_norm_squared;
  if (!(denominator > 0.0) || !std::isfinite(denominator)) return false;
  const double coefficient = hessenberg / denominator;
  for (int row = 0; row < true_rows; ++row)
  {
    const double expected = coefficient *
        preconditioned[static_cast<std::size_t>(row)];
    const double tolerance = 5.0e-10 * std::max(1.0, std::abs(expected));
    if (!std::isfinite(host_solution[static_cast<std::size_t>(row)]) ||
        std::abs(host_solution[static_cast<std::size_t>(row)] - expected) >
            tolerance)
      return false;
  }
  auto mismatched_system = system;
  mismatched_system.operator_kind = cooperative_warp
      ? NativeTraceOperatorKind::
            element_matrix_free_equilibrated_field_split_fp64
      : NativeTraceOperatorKind::
            element_matrix_free_warp_equilibrated_field_split_fp64;
  if (LaunchNativeRankOneTraceSolve(
          mismatched_system, preconditioner, workspace, policy, support,
          nullptr))
    return false;
  auto mismatched_preconditioner = preconditioner;
  mismatched_preconditioner.kind = cooperative_warp
      ? NativeTracePreconditionerKind::
            element_overlap_schwarz_equilibrated_field_split_sigma_jacobi3_fp64
      : NativeTracePreconditionerKind::
            element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64;
  return !LaunchNativeRankOneTraceSolve(
      system, mismatched_preconditioner, workspace, policy, support, nullptr);
}

bool NativeFixedSigmaJacobiDeviceFixtures()
{
  for (int order = 1; order <= 3; ++order)
    if (!NativeFixedSigmaJacobiOrderFixture(order, false) ||
        !NativeFixedSigmaJacobiOrderFixture(order, true))
      return false;
  return true;
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
      first.operator_applications == second.operator_applications &&
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
  const CudaTracePolicyDescriptor common_fast_tensor_igr =
      ResolveCudaTracePolicy(CudaEngineKind::tensor_core, true,
                             CudaTracePolicyKind::common_fast);
  if (!evidence.ConfigurationExactForGeneration(
          23, common_fast_tensor_igr) ||
      !evidence.ConfigurationExactForGeneration(
          23, common_fast_ordinary_igr) ||
      wrong_fill.ConfigurationExactForGeneration(
          23, common_fast_tensor_igr) ||
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
          23, common_fast_tensor_igr) ||
      evidence.ConfigurationExactForGeneration(
          23, CudaEngineKind::tensor_core,
          CudaTracePreconditionerKind::hypre_ilu0, false) ||
      !evidence.Advance(CudaHypreSetupTransition::mult))
    return false;
  auto extra_terminal = evidence;
  if (extra_terminal.Advance(CudaHypreSetupTransition::mult) ||
      extra_terminal.SolveExactForGeneration(
          23, common_fast_tensor_igr))
    return false;
  return evidence.SolveExactForGeneration(
             23, common_fast_tensor_igr) &&
      !evidence.SolveExactForGeneration(
          24, common_fast_tensor_igr);
}

static_assert(IluSetupOrderPolicy(),
              "ILU setup and application work must be generation-exact");

constexpr bool OrdinaryRankOneRefreshRoutingPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  return CudaOrdinaryRankOneRefreshPolicyIsExact(
             ResolveCudaTracePolicy(
                 CudaEngineKind::ordinary, false,
                 CudaTracePolicyKind::engine_default)) &&
      CudaOrdinaryRankOneRefreshPolicyIsExact(
          ResolveCudaTracePolicy(
              CudaEngineKind::ordinary, true,
              CudaTracePolicyKind::engine_default)) &&
      !CudaOrdinaryRankOneRefreshPolicyIsExact(
          dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy(
              CudaEngineKind::ordinary, true, 2,
              dg::hdg::generic::minimum_polynomial_order,
              CudaTracePolicyKind::engine_default)) &&
      !CudaOrdinaryRankOneRefreshPolicyIsExact(
          ResolveCudaTracePolicy(
              CudaEngineKind::ordinary, true,
              CudaTracePolicyKind::common_fast)) &&
      !CudaOrdinaryRankOneRefreshPolicyIsExact(
          ResolveCudaTracePolicy(
              CudaEngineKind::tensor_core, false,
              CudaTracePolicyKind::engine_default)) &&
      !CudaOrdinaryRankOneRefreshPolicyIsExact(
          ResolveCudaTracePolicy(
              CudaEngineKind::tensor_core, true,
              CudaTracePolicyKind::engine_default));
}

static_assert(OrdinaryRankOneRefreshRoutingPolicy(),
              "only exact ordinary refresh-each policies use the lean solve");

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

constexpr bool ProductionTracePolicyResolutionPolicy() noexcept
{
  using dg::hdg::generic::CudaEngineKind;
  using dg::hdg::runtime::cuda_w8a::detail::CudaTraceSolverKind;
  using dg::hdg::runtime::cuda_w8a::detail::ResolveCudaTraceSolverKind;
  constexpr CudaTracePolicyKind warp = CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64;
  for (int order = dg::hdg::generic::minimum_polynomial_order;
       order <= dg::hdg::generic::maximum_polynomial_order; ++order)
  {
    const auto tensor_rank1 =
        dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy(
            CudaEngineKind::tensor_core, true, 1, order,
            CudaTracePolicyKind::engine_default);
    if (!CudaTracePolicyDescriptorIsExact(tensor_rank1) ||
        tensor_rank1.requested_selection !=
            CudaTracePolicyKind::engine_default ||
        tensor_rank1.resolved_selection != warp ||
        tensor_rank1.rank_count != 1 ||
        tensor_rank1.polynomial_order != order ||
        tensor_rank1.trace_preconditioner != CudaTracePreconditionerKind::
            native_element_schwarz_warp_equilibrated_field_split_fp64 ||
        ResolveCudaTraceSolverKind(
            tensor_rank1.selected_engine, tensor_rank1.rank_count, false,
            true, true, true, tensor_rank1.resolved_selection) !=
            CudaTraceSolverKind::native_rank1_fp64)
      return false;

    const auto tensor_rank2 =
        dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy(
            CudaEngineKind::tensor_core, true, 2, order,
            CudaTracePolicyKind::engine_default);
    if (!CudaTracePolicyDescriptorIsExact(tensor_rank2) ||
        tensor_rank2.requested_selection !=
            CudaTracePolicyKind::engine_default ||
        tensor_rank2.resolved_selection !=
            CudaTracePolicyKind::engine_default ||
        tensor_rank2.trace_preconditioner !=
            CudaTracePreconditionerKind::hypre_ilu0 ||
        ResolveCudaTraceSolverKind(
            tensor_rank2.selected_engine, tensor_rank2.rank_count, true,
            true, true, true, tensor_rank2.resolved_selection) !=
            CudaTraceSolverKind::hypre_host_staged_rank2)
      return false;

    const auto rejected_rank2_explicit =
        dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy(
            CudaEngineKind::tensor_core, true, 2, order, warp);
    if (rejected_rank2_explicit.valid ||
        CudaTracePolicyDescriptorIsExact(rejected_rank2_explicit))
      return false;

    for (const bool igr_active : {false, true})
    {
      for (const int ranks : {1, 2})
      {
        const auto ordinary =
            dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy(
                CudaEngineKind::ordinary, igr_active, ranks, order,
                CudaTracePolicyKind::engine_default);
        if (!CudaTracePolicyDescriptorIsExact(ordinary) ||
            ordinary.requested_selection !=
                CudaTracePolicyKind::engine_default ||
            ordinary.resolved_selection !=
                CudaTracePolicyKind::engine_default ||
            ordinary.trace_preconditioner !=
                CudaTracePreconditionerKind::hypre_ilu0)
          return false;
      }
    }
  }
  return !dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy(
              CudaEngineKind::tensor_core, true, 0, 1,
              CudaTracePolicyKind::engine_default).valid &&
      !dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy(
              CudaEngineKind::tensor_core, true, 1, 4,
              CudaTracePolicyKind::engine_default).valid;
}

static_assert(ProductionTracePolicyResolutionPolicy(),
              "production trace-policy resolution must be exact and closed");

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
  const auto tensor_igr_mgr = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::equilibrated_mgr);
  const auto tensor_conventional_mgr = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::equilibrated_mgr);
  const auto ordinary_igr_mgr = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::equilibrated_mgr);
  const auto tensor_igr_schwarz = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::native_element_schwarz_fp64);
  const auto tensor_conventional_schwarz = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::native_element_schwarz_fp64);
  const auto ordinary_igr_schwarz = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::native_element_schwarz_fp64);
  const auto tensor_igr_matrix_free = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64);
  const auto tensor_conventional_matrix_free = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64);
  const auto ordinary_igr_matrix_free = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64);
  const auto tensor_igr_field_split = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64);
  const auto tensor_conventional_field_split = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64);
  const auto ordinary_igr_field_split = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64);
  const auto tensor_igr_combined = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64);
  const auto tensor_conventional_combined = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64);
  const auto ordinary_igr_combined = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64);
  const auto tensor_igr_warp = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64);
  const auto tensor_conventional_warp = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64);
  const auto ordinary_igr_warp = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64);
  const auto tensor_igr_sigma_jacobi = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64);
  const auto tensor_conventional_sigma_jacobi = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64);
  const auto ordinary_igr_sigma_jacobi = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64);
  const auto tensor_igr_warp_sigma_jacobi = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, true,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64);
  const auto tensor_conventional_warp_sigma_jacobi = ResolveCudaTracePolicy(
      CudaEngineKind::tensor_core, false,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64);
  const auto ordinary_igr_warp_sigma_jacobi = ResolveCudaTracePolicy(
      CudaEngineKind::ordinary, true,
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64);
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
      tensor_igr_default.requested_selection ==
          CudaTracePolicyKind::engine_default &&
      tensor_igr_default.resolved_selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64 &&
      tensor_igr_default.trace_preconditioner == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_fp64 &&
      tensor_igr_default.ilu_local_reordering == -1 &&
      tensor_igr_default.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::not_applicable &&
      tensor_igr_default.ilu_lower_jacobi_iterations == 0 &&
      tensor_igr_default.ilu_upper_jacobi_iterations == 0 &&
      tensor_igr_default.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::not_applicable &&
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
      CudaTracePolicyDescriptorIsExact(tensor_igr_mgr) &&
      tensor_igr_mgr.trace_preconditioner ==
          CudaTracePreconditionerKind::hypre_mgr_field_split &&
      tensor_igr_mgr.ilu_type == -1 &&
      tensor_igr_mgr.ilu_level_of_fill == -1 &&
      tensor_igr_mgr.ilu_local_reordering == -1 &&
      tensor_igr_mgr.ilu_application_iterations == 0 &&
      tensor_igr_mgr.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::not_applicable &&
      tensor_igr_mgr.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::not_applicable &&
      CudaTracePolicyDescriptorIsExact(tensor_igr_schwarz) &&
      tensor_igr_schwarz.trace_preconditioner ==
          CudaTracePreconditionerKind::native_element_schwarz_fp64 &&
      tensor_igr_schwarz.ilu_type == -1 &&
      tensor_igr_schwarz.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::not_applicable &&
      !tensor_conventional_schwarz.valid &&
      !CudaTracePolicyDescriptorIsExact(tensor_conventional_schwarz) &&
      !ordinary_igr_schwarz.valid &&
      !CudaTracePolicyDescriptorIsExact(ordinary_igr_schwarz) &&
      CudaTracePolicyDescriptorIsExact(tensor_igr_matrix_free) &&
      tensor_igr_matrix_free.trace_preconditioner ==
          CudaTracePreconditionerKind::native_element_schwarz_fp64 &&
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          tensor_igr_matrix_free.resolved_selection) &&
      !tensor_conventional_matrix_free.valid &&
      !CudaTracePolicyDescriptorIsExact(tensor_conventional_matrix_free) &&
      !ordinary_igr_matrix_free.valid &&
      !CudaTracePolicyDescriptorIsExact(ordinary_igr_matrix_free) &&
      CudaTracePolicyDescriptorIsExact(tensor_igr_field_split) &&
      tensor_igr_field_split.trace_preconditioner ==
          CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_fp64 &&
      tensor_igr_field_split.ilu_type == -1 &&
      tensor_igr_field_split.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::not_applicable &&
      !CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          tensor_igr_field_split.resolved_selection) &&
      !tensor_conventional_field_split.valid &&
      !CudaTracePolicyDescriptorIsExact(tensor_conventional_field_split) &&
      !ordinary_igr_field_split.valid &&
      !CudaTracePolicyDescriptorIsExact(ordinary_igr_field_split) &&
      CudaTracePolicyDescriptorIsExact(tensor_igr_combined) &&
      tensor_igr_combined.trace_preconditioner ==
          CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_fp64 &&
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          tensor_igr_combined.resolved_selection) &&
      CudaTracePolicyUsesNativeElementMatrixFreeEquilibratedFieldSplit(
          tensor_igr_combined.resolved_selection) &&
      !CudaTracePolicyUsesNativeElementMatrixFreeEquilibratedFieldSplit(
          tensor_igr_matrix_free.resolved_selection) &&
      !tensor_conventional_combined.valid &&
      !CudaTracePolicyDescriptorIsExact(tensor_conventional_combined) &&
      !ordinary_igr_combined.valid &&
      !CudaTracePolicyDescriptorIsExact(ordinary_igr_combined) &&
      CudaTracePolicyDescriptorIsExact(tensor_igr_warp) &&
      tensor_igr_warp.trace_preconditioner ==
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_fp64 &&
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          tensor_igr_warp.resolved_selection) &&
      CudaTracePolicyUsesNativeElementMatrixFreeEquilibratedFieldSplit(
          tensor_igr_warp.resolved_selection) &&
      CudaTracePolicyUsesNativeElementMatrixFreeWarpEquilibratedFieldSplit(
          tensor_igr_warp.resolved_selection) &&
      !CudaTracePolicyUsesNativeElementMatrixFreeWarpEquilibratedFieldSplit(
          tensor_igr_combined.resolved_selection) &&
      !tensor_conventional_warp.valid &&
      !CudaTracePolicyDescriptorIsExact(tensor_conventional_warp) &&
      !ordinary_igr_warp.valid &&
      !CudaTracePolicyDescriptorIsExact(ordinary_igr_warp) &&
      CudaTracePolicyDescriptorIsExact(tensor_igr_sigma_jacobi) &&
      tensor_igr_sigma_jacobi.trace_preconditioner ==
          CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64 &&
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          tensor_igr_sigma_jacobi.resolved_selection) &&
      CudaTracePolicyUsesNativeElementMatrixFreeEquilibratedFieldSplit(
          tensor_igr_sigma_jacobi.resolved_selection) &&
      CudaTracePreconditionerUsesFixedSigmaJacobi(
          tensor_igr_sigma_jacobi.trace_preconditioner) &&
      CudaTracePreconditionerFixedSigmaJacobiSweeps(
          tensor_igr_sigma_jacobi.trace_preconditioner) == 3 &&
      CudaTracePreconditionerFixedSigmaJacobiDamping(
          tensor_igr_sigma_jacobi.trace_preconditioner) ==
          native_element_schwarz_sigma_jacobi_damping &&
      !tensor_conventional_sigma_jacobi.valid &&
      !CudaTracePolicyDescriptorIsExact(
          tensor_conventional_sigma_jacobi) &&
      !ordinary_igr_sigma_jacobi.valid &&
      !CudaTracePolicyDescriptorIsExact(ordinary_igr_sigma_jacobi) &&
      CudaTracePolicyDescriptorIsExact(tensor_igr_warp_sigma_jacobi) &&
      tensor_igr_warp_sigma_jacobi.trace_preconditioner ==
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64 &&
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(
          tensor_igr_warp_sigma_jacobi.resolved_selection) &&
      CudaTracePolicyUsesNativeElementMatrixFreeEquilibratedFieldSplit(
          tensor_igr_warp_sigma_jacobi.resolved_selection) &&
      CudaTracePolicyUsesNativeElementMatrixFreeWarpEquilibratedFieldSplit(
          tensor_igr_warp_sigma_jacobi.resolved_selection) &&
      CudaTracePreconditionerIsNativeElementSchwarzWarpEquilibratedFieldSplit(
          tensor_igr_warp_sigma_jacobi.trace_preconditioner) &&
      CudaTracePreconditionerUsesFixedSigmaJacobi(
          tensor_igr_warp_sigma_jacobi.trace_preconditioner) &&
      CudaTracePreconditionerFixedSigmaJacobiSweeps(
          tensor_igr_warp_sigma_jacobi.trace_preconditioner) == 3 &&
      CudaTracePreconditionerFixedSigmaJacobiDamping(
          tensor_igr_warp_sigma_jacobi.trace_preconditioner) ==
          native_element_schwarz_sigma_jacobi_damping &&
      !tensor_conventional_warp_sigma_jacobi.valid &&
      !CudaTracePolicyDescriptorIsExact(
          tensor_conventional_warp_sigma_jacobi) &&
      !ordinary_igr_warp_sigma_jacobi.valid &&
      !CudaTracePolicyDescriptorIsExact(ordinary_igr_warp_sigma_jacobi) &&
      !tensor_conventional_mgr.valid &&
      !CudaTracePolicyDescriptorIsExact(tensor_conventional_mgr) &&
      !ordinary_igr_mgr.valid &&
      !CudaTracePolicyDescriptorIsExact(ordinary_igr_mgr) &&
      !unknown.valid && !CudaTracePolicyDescriptorIsExact(unknown) &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::engine_default)) == "engine-default" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::common_fast)) == "common-fast" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::equilibrated_mgr)) ==
          "equilibrated-mgr" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::native_element_schwarz_fp64)) ==
          "native-element-schwarz-fp64" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64)) ==
          "native-element-schwarz-matrix-free-fp64" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::
              native_element_schwarz_equilibrated_field_split_fp64)) ==
          "native-element-schwarz-equilibrated-field-split-fp64" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_equilibrated_field_split_fp64)) ==
          "native-element-schwarz-matrix-free-equilibrated-field-split-fp64" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64)) ==
          "native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64)) ==
          "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi3-fp64" &&
      std::string_view(CudaTracePolicyKindName(
          CudaTracePolicyKind::
              native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64)) ==
          "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi3-fp64" &&
      std::string_view(CudaTracePreconditionerName(
          CudaTracePreconditionerKind::hypre_mgr_field_split)) ==
          "hypre_mgr_field_split" &&
      std::string_view(CudaTracePreconditionerName(
          CudaTracePreconditionerKind::native_element_schwarz_fp64)) ==
          "native_element_schwarz_fp64" &&
      std::string_view(CudaTracePreconditionerName(
          CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_fp64)) ==
          "native_element_schwarz_equilibrated_field_split_fp64" &&
      std::string_view(CudaTracePreconditionerName(
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_fp64)) ==
          "native_element_schwarz_warp_equilibrated_field_split_fp64" &&
      std::string_view(CudaTracePreconditionerName(
          CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64)) ==
          "native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64" &&
      std::string_view(CudaTracePreconditionerName(
          CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64)) ==
          "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64";
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
  diagonal_evidence.requested_trace_policy =
      CudaTracePolicyKind::engine_default;
  diagonal_evidence.resolved_trace_policy =
      CudaTracePolicyKind::engine_default;
  diagonal_evidence.trace_policy_rank_count = 1;
  diagonal_evidence.trace_policy_order = 1;
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
  common_diagonal_evidence.requested_trace_policy =
      CudaTracePolicyKind::common_fast;
  common_diagonal_evidence.resolved_trace_policy =
      CudaTracePolicyKind::common_fast;
  common_diagonal_evidence.trace_policy_rank_count = 1;
  common_diagonal_evidence.trace_policy_order = 1;
  if (!CudaInitializeIluEvidence(
          common_diagonal_evidence,
          CudaTracePreconditionerKind::hypre_diag_scale, false) ||
      common_diagonal_evidence.ilu_factor_reuse_policy !=
          CudaIluFactorReusePolicy::not_applicable ||
      !common_diagonal_evidence.ilu_accounting_complete)
    return false;

  CudaTensorCoreEvidence ordinary_evidence{};
  ordinary_evidence.selected_engine = CudaEngineKind::ordinary;
  ordinary_evidence.requested_trace_policy =
      CudaTracePolicyKind::engine_default;
  ordinary_evidence.resolved_trace_policy =
      CudaTracePolicyKind::engine_default;
  ordinary_evidence.trace_policy_rank_count = 1;
  ordinary_evidence.trace_policy_order = 1;
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
  tensor_igr_evidence.requested_trace_policy =
      CudaTracePolicyKind::common_fast;
  tensor_igr_evidence.resolved_trace_policy =
      CudaTracePolicyKind::common_fast;
  tensor_igr_evidence.trace_policy_rank_count = 1;
  tensor_igr_evidence.trace_policy_order = 1;
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
  common_ordinary_igr_evidence.requested_trace_policy =
      CudaTracePolicyKind::common_fast;
  common_ordinary_igr_evidence.resolved_trace_policy =
      CudaTracePolicyKind::common_fast;
  common_ordinary_igr_evidence.trace_policy_rank_count = 1;
  common_ordinary_igr_evidence.trace_policy_order = 1;
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
  incomplete_evidence.requested_trace_policy =
      CudaTracePolicyKind::common_fast;
  incomplete_evidence.resolved_trace_policy =
      CudaTracePolicyKind::common_fast;
  incomplete_evidence.trace_policy_rank_count = 1;
  incomplete_evidence.trace_policy_order = 1;
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
  hard_failure_evidence.requested_trace_policy =
      CudaTracePolicyKind::common_fast;
  hard_failure_evidence.resolved_trace_policy =
      CudaTracePolicyKind::common_fast;
  hard_failure_evidence.trace_policy_rank_count = 1;
  hard_failure_evidence.trace_policy_order = 1;
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

bool ElementSchwarzFactorStateLifecyclePolicy() noexcept
{
  using namespace dg::hdg::runtime::cuda_w8a::detail;
  AttemptIdentity attempt{};
  attempt.sequence = 17;
  attempt.committed_step = 5;
  attempt.committed_generation = 9;
  attempt.dt = 2.0e-4;
  AttemptIdentity foreign = attempt;
  foreign.sequence = 18;
  CudaElementSchwarzAttemptState state{};
  if (CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 10, false))
    return false;
  state.attempt = attempt;
  state.active = true;
  if (!CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 10, false) ||
      CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 10, true))
    return false;
  state.factor_ready = true;
  state.factor_generation = 10;
  if (CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 11, false) ||
      CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 10, true))
    return false;
  state.last_solve_generation = 10;
  if (!CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 10, true) ||
      !CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 11, false) ||
      CudaElementSchwarzFactorTransitionIsExact(
          state, foreign, 11, false))
    return false;
  state.factor_generation = 11;
  if (CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 12, false) ||
      CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 11, true))
    return false;
  state.last_solve_generation = 11;
  if (!CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 11, true) ||
      !CudaElementSchwarzFactorTransitionIsExact(
          state, attempt, 12, false))
    return false;
  {
    CudaElementSchwarzAttemptEndGuard end_attempt(state);
    if (!state.active || !state.factor_ready) return false;
  }
  return !state.active && !state.factor_ready &&
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
      IluFactorStateLifecyclePolicy() &&
      ElementSchwarzFactorStateLifecyclePolicy();
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
      (lifecycle && NativeTraceDeviceFixtures() &&
       NativeElementSchwarzDeviceFixture() &&
       NativeFixedSigmaJacobiDeviceFixtures());
  const bool mgr_device_fixture = ranks != 1 || !mode.empty() ||
      (native_device_fixtures && RankOneEquilibratedMgrFixture());
  const bool pass = policies && lifecycle && native_device_fixtures &&
      mgr_device_fixture;
  if (!pass && rank == 0)
    std::cerr << "compact CUDA W8A audit failed\n";
  return pass ? 0 : 1;
}

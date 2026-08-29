#pragma once

/**
 * @file numerics/hdg/runtime/cuda/native_rank1_trace_solver.cuh
 * @brief Private rank-one cooperative FP64 trace-solver leaf.
 *
 * This interface borrows already-resident application and HYPRE storage.  It
 * owns no matrix, factor, vector, stream, or workspace allocation and exposes
 * no application selector.  One fixed-size control POD is the complete solve
 * result; the leaf does not retain iteration history or per-event records.
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dg::hdg::runtime::cuda_w8a::detail
{

inline constexpr std::uint32_t native_trace_control_schema = 1;
inline constexpr int native_trace_cooperative_threads = 256;
inline constexpr int native_trace_ilu_jacobi_iterations = 5;
inline constexpr int native_trace_sigma_jacobi3_sweeps = 3;
inline constexpr int native_trace_sigma_jacobi5_sweeps = 5;
inline constexpr int native_trace_sigma_jacobi7_sweeps = 7;
inline constexpr int native_trace_sigma_jacobi_maximum_sweeps = 10;
inline constexpr double native_trace_sigma_jacobi_damping =
    0x1.5555555555555p-1;

enum class NativeTracePreconditionerKind : std::int32_t
{
  diagonal = 1,
  ilu0_rcm_jacobi = 2,
  element_overlap_schwarz_fp64 = 3,
  element_overlap_schwarz_equilibrated_field_split_fp64 = 4,
  element_overlap_schwarz_warp_equilibrated_field_split_fp64 = 5,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi3_fp64 = 6,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64 = 7,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi1_fp64 = 8,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64 = 9,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi2_fp64 = 10,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64 = 11,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi4_fp64 = 12,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64 = 13,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi5_fp64 = 14,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64 = 15,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi6_fp64 = 16,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64 = 17,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi7_fp64 = 18,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64 = 19,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi8_fp64 = 20,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64 = 21,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi9_fp64 = 22,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64 = 23,
  element_overlap_schwarz_equilibrated_field_split_sigma_jacobi10_fp64 = 24,
  element_overlap_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64 = 25
};

enum class NativeTraceSolveStatus : std::int32_t
{
  uninitialized = 0,
  running = 1,
  converged = 2,
  invalid_policy = 3,
  invalid_system = 4,
  invalid_preconditioner = 5,
  invalid_workspace = 6,
  unsupported_cooperative_launch = 7,
  nonfinite_input = 8,
  numerical_breakdown = 9,
  maximum_iterations = 10,
  bounded_real_residual_stagnation = 11,
  launch_failure = 12
};

/** A borrowed, rank-local FP64 CSR matrix in device memory. */
struct NativeTraceCsrView final
{
  const int* row_offsets = nullptr;
  const int* columns = nullptr;
  const double* values = nullptr;
  int rows = 0;
  int column_count = 0;
  int nonzeros = 0;
};

enum class NativeTraceOperatorKind : std::int32_t
{
  assembled_csr_fp64 = 1,
  element_matrix_free_fp64 = 2,
  element_matrix_free_equilibrated_field_split_fp64 = 3,
  element_matrix_free_warp_equilibrated_field_split_fp64 = 4
};

/** Current true-trace system and HYPRE-owned solution storage. */
struct NativeTraceSystemView final
{
  NativeTraceCsrView matrix{};
  const double* rhs = nullptr;
  double* solution = nullptr;
  NativeTraceOperatorKind operator_kind =
      NativeTraceOperatorKind::assembled_csr_fp64;
  // Retained row-major [element][row][column] Schur blocks. Maps and the
  // persistent element-action scratch are shared with the exact Schwarz view.
  const double* element_schur = nullptr;
};

/**
 * Borrowed HYPRE ILU0 device factor.
 *
 * `permutation` has the exact HYPRE gather convention
 * `permuted[new] = original[permutation[new]]`.  The combined factor stores
 * strict L below the diagonal, the U diagonal, and strict U above it.
 * `inverse_diagonal` is HYPRE extraction type 2, namely `1 / U_ii`.
 */
struct NativeTraceIlu0View final
{
  NativeTraceCsrView factor{};
  const int* permutation = nullptr;
  const double* inverse_diagonal = nullptr;
  int lower_jacobi_iterations = 0;
  int upper_jacobi_iterations = 0;
};

/** Persistent exact principal patches and their pivoted FP64 factors. */
struct NativeTraceElementSchwarzView final
{
  const double* patch_matrix = nullptr;
  const double* patch_lu = nullptr;
  const int* pivots = nullptr;
  const int* signed_element_true_dofs = nullptr;
  const int* true_incidence_slots = nullptr;
  double* element_rhs = nullptr;
  double* element_solution = nullptr;
  int elements = 0;
  int patch_rows = 0;
  int true_rows = 0;
  int minimum_multiplicity = 0;
  int maximum_multiplicity = 0;
};

struct NativeTracePreconditionerView final
{
  NativeTracePreconditionerKind kind =
      NativeTracePreconditionerKind::diagonal;
  NativeTraceIlu0View ilu0{};
  NativeTraceElementSchwarzView element_schwarz{};
};

/** Pinned HYPRE GMRES policy; all other convergence switches are fixed off. */
struct NativeTraceSolvePolicy final
{
  int rank_count = 0;
  int krylov_dimension = 0;
  int maximum_iterations = 0;
  double relative_tolerance = 0.0;
  double absolute_tolerance = 0.0;
};

/**
 * Constant-size device result and scalar scratch.
 *
 * `reduction_value` is private transient storage while the cooperative kernel
 * is active.  It has no evidentiary meaning after completion.
 */
struct alignas(8) NativeTraceSolveControl final
{
  std::uint32_t schema = 0;
  NativeTraceSolveStatus status = NativeTraceSolveStatus::uninitialized;
  std::uint32_t failure_mask = 0;
  int iterations = 0;
  int restart_cycles = 0;
  int preconditioner_applications = 0;
  int operator_applications = 0;
  int actual_residual_checks = 0;
  double rhs_norm = 0.0;
  double initial_residual_norm = 0.0;
  double estimated_residual_norm = 0.0;
  double actual_residual_norm = 0.0;
  double relative_residual_norm = 0.0;
  double reduction_value = 0.0;
};

struct NativeTraceWorkspaceRequirements final
{
  bool valid = false;
  std::size_t basis_doubles = 0;
  std::size_t vector_doubles = 0;
  std::size_t reduction_doubles = 0;
  std::size_t small_doubles = 0;
  std::size_t permutation_integers = 0;
  std::size_t control_double_slots = 0;
};

/** App-owned, persistent device slices with explicit scalar capacities. */
struct NativeTraceWorkspaceView final
{
  double* basis = nullptr;
  std::size_t basis_doubles = 0;
  double* vectors = nullptr;
  std::size_t vector_doubles = 0;
  double* reduction = nullptr;
  std::size_t reduction_doubles = 0;
  double* small = nullptr;
  std::size_t small_doubles = 0;
  int* permutation_marks = nullptr;
  std::size_t permutation_integers = 0;
  NativeTraceSolveControl* device_control = nullptr;
};

/** Exact current-device cooperative occupancy admitted at initialization. */
struct NativeTraceCooperativeSupport final
{
  bool supported = false;
  /** Exact kernel specialization whose occupancy produced this record. */
  bool fixed_sigma_jacobi = false;
  int device = -1;
  int compute_major = 0;
  int compute_minor = 0;
  int multiprocessors = 0;
  int threads_per_block = 0;
  int maximum_active_blocks_per_multiprocessor = 0;
  int cooperative_blocks = 0;
};

constexpr std::size_t NativeTraceControlDoubleSlots() noexcept
{
  return (sizeof(NativeTraceSolveControl) + sizeof(double) - 1) /
      sizeof(double);
}

inline const char* NativeTraceSolveStatusName(
    const NativeTraceSolveStatus status) noexcept
{
  switch (status)
  {
  case NativeTraceSolveStatus::uninitialized: return "uninitialized";
  case NativeTraceSolveStatus::running: return "running";
  case NativeTraceSolveStatus::converged: return "converged";
  case NativeTraceSolveStatus::invalid_policy: return "invalid_policy";
  case NativeTraceSolveStatus::invalid_system: return "invalid_system";
  case NativeTraceSolveStatus::invalid_preconditioner:
    return "invalid_preconditioner";
  case NativeTraceSolveStatus::invalid_workspace: return "invalid_workspace";
  case NativeTraceSolveStatus::unsupported_cooperative_launch:
    return "unsupported_cooperative_launch";
  case NativeTraceSolveStatus::nonfinite_input: return "nonfinite_input";
  case NativeTraceSolveStatus::numerical_breakdown:
    return "numerical_breakdown";
  case NativeTraceSolveStatus::maximum_iterations:
    return "maximum_iterations";
  case NativeTraceSolveStatus::bounded_real_residual_stagnation:
    return "bounded_real_residual_stagnation";
  case NativeTraceSolveStatus::launch_failure: return "launch_failure";
  }
  return "unknown";
}

/**
 * Pure host admission of a converged or bounded non-hard terminal result.
 * Maximum-iteration and nondecreasing-real-residual exits retain distinct
 * statuses and do not claim internal convergence.  The application
 * independently rebuilds the authoritative FP64 true residual after this
 * structural/finite admission.
 */
inline bool NativeTraceResultIsReadyForTrueResidual(
    const NativeTraceSolveControl& result,
    const NativeTraceSolvePolicy& policy) noexcept
{
  const bool converged =
      result.status == NativeTraceSolveStatus::converged;
  const bool bounded_maximum =
      result.status == NativeTraceSolveStatus::maximum_iterations &&
      result.iterations == policy.maximum_iterations;
  const bool bounded_real_residual_stagnation =
      result.status ==
          NativeTraceSolveStatus::bounded_real_residual_stagnation &&
      result.iterations > 0 && result.restart_cycles > 0 &&
      result.actual_residual_checks >= 2;
  if (result.schema != native_trace_control_schema ||
      (!converged && !bounded_maximum &&
       !bounded_real_residual_stagnation) ||
      result.failure_mask != 0 || policy.rank_count != 1 ||
      policy.krylov_dimension <= 0 || policy.maximum_iterations <= 0 ||
      result.iterations < 0 ||
      result.iterations > policy.maximum_iterations ||
      result.restart_cycles < 0 || result.preconditioner_applications < 0 ||
      result.operator_applications < 0 ||
      result.actual_residual_checks <= 0 ||
      !std::isfinite(result.rhs_norm) || result.rhs_norm < 0.0 ||
      !std::isfinite(result.initial_residual_norm) ||
      result.initial_residual_norm < 0.0 ||
      !std::isfinite(result.estimated_residual_norm) ||
      result.estimated_residual_norm < 0.0 ||
      !std::isfinite(result.actual_residual_norm) ||
      result.actual_residual_norm < 0.0 ||
      !std::isfinite(result.relative_residual_norm) ||
      result.relative_residual_norm < 0.0 ||
      !std::isfinite(policy.relative_tolerance) ||
      !(policy.relative_tolerance > 0.0) ||
      !std::isfinite(policy.absolute_tolerance) ||
      policy.absolute_tolerance < 0.0)
    return false;
  const double denominator = result.rhs_norm > 0.0
      ? result.rhs_norm : result.initial_residual_norm;
  const double tolerance = std::max(
      policy.absolute_tolerance,
      policy.relative_tolerance * denominator);
  const double expected_relative = result.rhs_norm > 0.0
      ? result.actual_residual_norm / result.rhs_norm
      : result.actual_residual_norm;
  const double consistency_scale = std::max(
      {1.0, std::abs(expected_relative),
       std::abs(result.relative_residual_norm)});
  const double consistency_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * consistency_scale;
  return std::isfinite(tolerance) &&
      (!converged || result.actual_residual_norm <= tolerance) &&
      std::abs(result.relative_residual_norm - expected_relative) <=
          consistency_tolerance;
}

extern "C" NativeTraceWorkspaceRequirements
QueryNativeTraceWorkspaceRequirements(int rows, int krylov_dimension,
                                      int cooperative_blocks,
                                      bool igr_active) noexcept;

extern "C" NativeTraceCooperativeSupport
QueryNativeRankOneTraceCooperativeSupport(int rows, int krylov_dimension,
                                          bool igr_active,
                                          bool fixed_sigma_jacobi) noexcept;

/** Queue exactly one cooperative kernel.  This call never copies or syncs. */
extern "C" bool LaunchNativeRankOneTraceSolve(
    const NativeTraceSystemView& system,
    const NativeTracePreconditionerView& preconditioner,
    const NativeTraceWorkspaceView& workspace,
    const NativeTraceSolvePolicy& policy,
    const NativeTraceCooperativeSupport& support,
    cudaStream_t stream) noexcept;

static_assert(std::is_standard_layout_v<NativeTraceCsrView>);
static_assert(std::is_trivially_copyable_v<NativeTraceCsrView>);
static_assert(std::is_standard_layout_v<NativeTraceSolveControl>);
static_assert(std::is_trivially_copyable_v<NativeTraceSolveControl>);
static_assert(alignof(NativeTraceSolveControl) <= alignof(double));
static_assert(NativeTraceControlDoubleSlots() * sizeof(double) >=
              sizeof(NativeTraceSolveControl));

} // namespace dg::hdg::runtime::cuda_w8a::detail

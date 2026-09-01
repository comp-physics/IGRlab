#pragma once

/**
 * @file numerics/hdg/runtime/cuda/persistent_smooth_euler_types.cuh
 * @brief CUDA-qualified values for bounded W8A smoke/functional paths.
 *
 * This header intentionally does not define a canonical formulation tag,
 * common FailurePhase, or cross-backend DTO.  The caller supplies Selection;
 * every public value type remains qualified by cuda_w8a until a coordinator-
 * owned common ABI is accepted.
 */

#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/run_contract.hpp"

#include <mfem.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

namespace dg::hdg::runtime::cuda
{
class GenericEulerHdgCudaSession;
}

namespace dg::hdg::runtime::cuda_w8a
{

inline constexpr std::string_view experiment_id =
    "exp.hdg.euler.conventional.smooth_persistent.w8a.v1";
inline constexpr std::string_view runtime_layout_id =
    "l2-interior.true-trace-owned.component-major.v1";
inline constexpr int smoke_polynomial_order = 1;
inline constexpr int smoke_elements_per_direction = 3;
inline constexpr int functional_polynomial_order = 2;
inline constexpr int functional_elements_per_direction = 6;
// Retained spellings keep the admitted p1/N3 tests source-compatible.
inline constexpr int polynomial_order = smoke_polynomial_order;
inline constexpr int elements_per_direction = smoke_elements_per_direction;
inline constexpr double fixed_dt = 0.05;
inline constexpr int smoke_accepted_steps = 3;
inline constexpr int functional_accepted_steps = 20;
inline constexpr int functional_checkpoint_step = 10;
inline constexpr int functional_injected_step = 14;
inline constexpr int accepted_steps = smoke_accepted_steps;
inline constexpr int conserved_components = 4;
inline constexpr int igr_published_components = 5;
inline constexpr int igr_interior_components = 7;
inline constexpr int igr_trace_components = 5;
inline constexpr int legacy_scalar_workspace_scalars = 11;
inline constexpr int euler_conservation_result_scalars =
    2 * igr_trace_components + 2;
inline constexpr int dimension = 2;
// p1 values are retained for the Admission-02 oracle fixtures.  Runtime
// kernels use the counts carried by CudaMeshShape; storage uses the maxima.
inline constexpr int scalar_dofs_per_element = 4;
inline constexpr int scalar_dofs_per_face = 2;
inline constexpr int faces_per_element = 4;
inline constexpr int interior_dofs_per_element =
    conserved_components * scalar_dofs_per_element;
inline constexpr int trace_dofs_per_element =
    faces_per_element * conserved_components * scalar_dofs_per_face;
// Frozen W8A DTO/oracle maxima remain p1/p2.  Generic borrowed mode uses its
// checked dynamic arena and the separate p3/IGR per-element bounds below.
inline constexpr int maximum_scalar_dofs_per_element = 9;
inline constexpr int maximum_scalar_dofs_per_face = 3;
inline constexpr int maximum_interior_dofs_per_element =
    conserved_components * maximum_scalar_dofs_per_element;
inline constexpr int maximum_trace_dofs_per_element =
    faces_per_element * conserved_components * maximum_scalar_dofs_per_face;
inline constexpr int generic_maximum_scalar_dofs_per_element = 16;
inline constexpr int generic_maximum_scalar_dofs_per_face = 4;
inline constexpr int generic_maximum_interior_dofs_per_element =
    igr_interior_components * generic_maximum_scalar_dofs_per_element;
inline constexpr int generic_maximum_trace_dofs_per_element =
    faces_per_element * igr_trace_components *
        generic_maximum_scalar_dofs_per_face;
inline constexpr int maximum_rank_elements = 36;
inline constexpr int maximum_local_trace_dofs = 864;
inline constexpr int maximum_true_trace_dofs = 864;
inline constexpr int maximum_element_trace_dofs =
    maximum_rank_elements * maximum_trace_dofs_per_element;
inline constexpr int maximum_volume_points = 64;
inline constexpr int maximum_face_points = 8;
inline constexpr int maximum_local_matrix_entries =
    maximum_rank_elements * maximum_trace_dofs_per_element *
        maximum_trace_dofs_per_element;
inline constexpr int maximum_true_matrix_entries =
    maximum_local_matrix_entries;
inline constexpr int maximum_global_dense_entries =
    maximum_true_trace_dofs * maximum_true_trace_dofs;
inline constexpr int maximum_exchange_entries = maximum_local_matrix_entries;
inline constexpr int maximum_attempt_records = 8;
inline constexpr int maximum_device_tickets = 128;
inline constexpr int maximum_transfer_events = 16384;
inline constexpr int maximum_allocation_events = 64;
inline constexpr int assembly_quadrature_order = 11;
inline constexpr int functional_assembly_quadrature_order = 14;

/**
 * Private rank-one trace work selector.
 *
 * This is not a mathematical-layout or backend-selection ABI: every mode
 * retains the complete Uhat4/sigmahat vectors and true matrix.  The two
 * projected values are admitted only for the corresponding partitioned-IGR
 * tensor solve and otherwise fail closed at the native launch seam.
 */
enum class CudaNativeTraceActiveSubspace : std::uint8_t
{
  full = 0,
  euler_uhat4 = 1,
  scalar_sigmahat = 2
};

constexpr bool IsKnownCudaNativeTraceActiveSubspace(
    const CudaNativeTraceActiveSubspace subspace) noexcept
{
  return subspace == CudaNativeTraceActiveSubspace::full ||
      subspace == CudaNativeTraceActiveSubspace::euler_uhat4 ||
      subspace == CudaNativeTraceActiveSubspace::scalar_sigmahat;
}

constexpr bool CudaNativeTraceActiveSubspaceIsProjected(
    const CudaNativeTraceActiveSubspace subspace) noexcept
{
  return subspace == CudaNativeTraceActiveSubspace::euler_uhat4 ||
      subspace == CudaNativeTraceActiveSubspace::scalar_sigmahat;
}

enum class CudaFailurePhase : unsigned char
{
  setup,
  assembly,
  condensation,
  trace_solve,
  recovery,
  after_recovery_before_commit,
  diagnostics,
  rollback_audit,
  terminal_evidence
};

enum class CudaRunStatus : unsigned char
{
  ready,
  committed,
  retry_available,
  success,
  injected_failure,
  setup_failure,
  runtime_failure,
  scientific_gate_failure,
  terminal_failure
};

enum class CudaTokenFault : unsigned char
{
  none,
  omit_solution,
  issue_matrix_before_residual,
  stale_attempt
};

struct CudaInjection
{
  CudaFailurePhase phase = CudaFailurePhase::after_recovery_before_commit;
  std::uint64_t target_step = 2;
  int source_rank = 0;
  bool enabled = false;
  CudaTokenFault token_fault = CudaTokenFault::none;
};

/**
 * Caller-owned L2 data in MFEM Ordering::byNODES form.  `scalar_count` is
 * the local scalar L2 dof count (not an element count); the fixed four
 * conserved components therefore occupy `4 * scalar_count` doubles.
 *
 * These pointers are an Initialize-only borrowing ABI.  The persistent CUDA
 * engine validates and copies them before it allocates its numerical state;
 * it never retains either input pointer.
 */
struct CudaBorrowedHostL2Input
{
  const double* coefficients = nullptr;
  std::size_t scalar_count = 0;
  int components = conserved_components;
};

/** Mutable caller-owned L2 destination for an explicit host export. */
struct CudaMutableHostL2Output
{
  double* coefficients = nullptr;
  std::size_t scalar_count = 0;
  int components = conserved_components;
};

/** The small, value-only physical-boundary subset admitted by CUDA. */
enum class CudaStaticBoundaryPolicyKind : std::uint8_t
{
  none = 0,
  supersonic_inlet = 1,
  supersonic_outlet = 2,
  slip_wall = 3,
  subsonic_inlet = 4,
  subsonic_outlet_pressure = 5,
  far_field = 6,
  nonreflecting_outlet = 7,
  // Retained reject-only spellings used by legacy negative-admission tests.
  subsonic_outlet = 8,
  nonreflecting = 9
};

/** Caller-borrowed initialization descriptor; values are copied immediately. */
struct CudaStaticBoundaryPolicy
{
  int attribute = 0;
  CudaStaticBoundaryPolicyKind kind = CudaStaticBoundaryPolicyKind::none;
  std::array<double, conserved_components> reference_state{{0.0, 0.0, 0.0,
                                                              0.0}};
  double pressure_out = 0.0;
};

inline constexpr std::size_t maximum_static_boundary_policies = 32;

struct CudaStaticBoundaryPolicyTable
{
  const CudaStaticBoundaryPolicy* policies = nullptr;
  std::size_t count = 0;
  double gamma = 1.4;
};

inline bool IsCudaAdmittedStaticBoundaryPolicy(
    const CudaStaticBoundaryPolicyKind kind) noexcept
{
  return kind == CudaStaticBoundaryPolicyKind::supersonic_inlet ||
      kind == CudaStaticBoundaryPolicyKind::supersonic_outlet ||
      kind == CudaStaticBoundaryPolicyKind::slip_wall ||
      kind == CudaStaticBoundaryPolicyKind::subsonic_inlet ||
      kind == CudaStaticBoundaryPolicyKind::subsonic_outlet_pressure ||
      kind == CudaStaticBoundaryPolicyKind::far_field ||
      kind == CudaStaticBoundaryPolicyKind::nonreflecting_outlet;
}

inline bool IsCudaPhysicalEulerState(
    const std::array<double, conserved_components>& state) noexcept
{
  constexpr double gamma = 1.4;
  for (const double value : state)
    if (!std::isfinite(value)) return false;
  if (!(state[0] > 0.0)) return false;
  const double pressure = (gamma - 1.0) *
      (state[3] - 0.5 * (state[1] * state[1] + state[2] * state[2]) /
                       state[0]);
  return pressure > 0.0 && std::isfinite(pressure);
}

inline bool ValidateCudaStaticBoundaryPolicyTable(
    const CudaStaticBoundaryPolicyTable& table) noexcept
{
  if (!std::isfinite(table.gamma) ||
      generic::DoubleBits(table.gamma) != generic::DoubleBits(1.4))
    return false;
  if (table.count == 0) return table.policies == nullptr;
  if (table.policies == nullptr || table.count > maximum_static_boundary_policies)
    return false;
  for (std::size_t i = 0; i < table.count; ++i)
  {
    const CudaStaticBoundaryPolicy& policy = table.policies[i];
    if (policy.attribute <= 0 ||
        !IsCudaAdmittedStaticBoundaryPolicy(policy.kind))
      return false;
    const bool needs_reference =
        policy.kind == CudaStaticBoundaryPolicyKind::supersonic_inlet ||
        policy.kind == CudaStaticBoundaryPolicyKind::subsonic_inlet ||
        policy.kind == CudaStaticBoundaryPolicyKind::far_field ||
        policy.kind == CudaStaticBoundaryPolicyKind::nonreflecting_outlet;
    if (needs_reference)
    {
      if (!IsCudaPhysicalEulerState(policy.reference_state)) return false;
    }
    else
      for (const double value : policy.reference_state)
        if (value != 0.0 || std::signbit(value)) return false;
    if (policy.kind == CudaStaticBoundaryPolicyKind::
            subsonic_outlet_pressure)
    {
      if (!std::isfinite(policy.pressure_out) ||
          !(policy.pressure_out > 0.0)) return false;
    }
    else if (policy.pressure_out != 0.0 ||
             std::signbit(policy.pressure_out))
      return false;
    if (i != 0 && table.policies[i - 1].attribute >= policy.attribute)
      return false;
  }
  return true;
}

inline bool CudaBorrowedHostModeConsensus(const MPI_Comm communicator,
                                          const bool requested,
                                          bool& agreed_mode) noexcept
{
  if (communicator == MPI_COMM_NULL) return false;
  const int local = requested ? 1 : 0;
  int minimum = 0;
  int maximum = 0;
  const bool minimum_ready = MPI_Allreduce(&local, &minimum, 1, MPI_INT,
      MPI_MIN, communicator) == MPI_SUCCESS;
  const bool maximum_ready = MPI_Allreduce(&local, &maximum, 1, MPI_INT,
      MPI_MAX, communicator) == MPI_SUCCESS;
  const bool consensus_ok = minimum_ready && maximum_ready &&
      minimum == maximum && (minimum == 0 || minimum == 1);
  agreed_mode = consensus_ok && minimum == 1;
  return consensus_ok;
}

inline bool CudaEngineKindConsensus(
    const MPI_Comm communicator,
    const generic::CudaEngineKind requested,
    generic::CudaEngineKind& agreed_engine) noexcept
{
  if (communicator == MPI_COMM_NULL) return false;
  const int local = static_cast<int>(requested);
  int minimum = 0;
  int maximum = 0;
  const bool minimum_ready = MPI_Allreduce(&local, &minimum, 1, MPI_INT,
      MPI_MIN, communicator) == MPI_SUCCESS;
  const bool maximum_ready = MPI_Allreduce(&local, &maximum, 1, MPI_INT,
      MPI_MAX, communicator) == MPI_SUCCESS;
  const bool consensus_ok = minimum_ready && maximum_ready &&
      minimum == maximum &&
      generic::IsKnownCudaEngineKind(
          static_cast<generic::CudaEngineKind>(minimum));
  if (consensus_ok)
    agreed_engine = static_cast<generic::CudaEngineKind>(minimum);
  return consensus_ok;
}

inline bool IsCudaBorrowedGlobalTopologyIdentity(
    const StableIdentity& value) noexcept
{
  return value.valid() && !value.empty();
}

inline bool IsCudaAbsentBorrowedGlobalTopologyIdentity(
    const StableIdentity& value) noexcept
{
  return value.empty();
}

/** Benchmark-only selection of the trace-solver policy under comparison. */
enum class CudaTracePolicyKind : std::uint8_t
{
  engine_default = 0,
  common_fast = 1,
  equilibrated_mgr = 2,
  native_element_schwarz_fp64 = 3,
  native_element_schwarz_matrix_free_fp64 = 4,
  native_element_schwarz_equilibrated_field_split_fp64 = 5,
  native_element_schwarz_matrix_free_equilibrated_field_split_fp64 = 6,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64 = 7,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64 =
      8,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64 =
      9,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi1_fp64 =
      10,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi1_fp64 =
      11,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi2_fp64 =
      12,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi2_fp64 =
      13,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi4_fp64 =
      14,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi4_fp64 =
      15,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi5_fp64 =
      16,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi5_fp64 =
      17,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi6_fp64 =
      18,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi6_fp64 =
      19,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi7_fp64 =
      20,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi7_fp64 =
      21,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi8_fp64 =
      22,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi8_fp64 =
      23,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi9_fp64 =
      24,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi9_fp64 =
      25,
  native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi10_fp64 =
      26,
  native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi10_fp64 =
      27,
  native_element_schwarz_matrix_free_warp_coupled_fp64 = 28,
  native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi2_fp64 = 29,
  native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi10_fp64 = 30,
  native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64 =
      31
};

constexpr bool IsKnownCudaTracePolicyKind(
    const CudaTracePolicyKind kind) noexcept
{
  return kind == CudaTracePolicyKind::engine_default ||
      kind == CudaTracePolicyKind::common_fast ||
      kind == CudaTracePolicyKind::equilibrated_mgr ||
      kind == CudaTracePolicyKind::native_element_schwarz_fp64 ||
      kind ==
          CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi10_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64;
}

constexpr const char* CudaTracePolicyKindName(
    const CudaTracePolicyKind kind) noexcept
{
  switch (kind)
  {
  case CudaTracePolicyKind::engine_default:
    return "engine-default";
  case CudaTracePolicyKind::common_fast:
    return "common-fast";
  case CudaTracePolicyKind::equilibrated_mgr:
    return "equilibrated-mgr";
  case CudaTracePolicyKind::native_element_schwarz_fp64:
    return "native-element-schwarz-fp64";
  case CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64:
    return "native-element-schwarz-matrix-free-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_equilibrated_field_split_fp64:
    return "native-element-schwarz-equilibrated-field-split-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi3-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi3-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi1_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi1-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi1_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi1-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi2_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi2-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi2_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi2-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi4_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi4-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi4_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi4-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi5_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi5-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi5_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi5-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi6_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi6-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi6_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi6-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi7_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi7-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi7_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi7-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi8_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi8-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi8_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi8-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi9_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi9-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi9_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi9-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi10_fp64:
    return "native-element-schwarz-matrix-free-equilibrated-field-split-sigma-jacobi10-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi10_fp64:
    return "native-element-schwarz-matrix-free-warp-equilibrated-field-split-sigma-jacobi10-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_coupled_fp64:
    return "native-element-schwarz-matrix-free-warp-coupled-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi2_fp64:
    return "native-element-schwarz-matrix-free-warp-coupled-sigma-jacobi2-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi10_fp64:
    return "native-element-schwarz-matrix-free-warp-coupled-sigma-jacobi10-fp64";
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64:
    return "native-element-schwarz-matrix-free-warp-robust-coupled-sigma-jacobi10-fp64";
  }
  return "unknown";
}

struct CudaConfig
{
  StableIdentity experiment = StableIdentity::Copy(experiment_id);
  int order = polynomial_order;
  int mesh_n = elements_per_direction;
  double dt = fixed_dt;
  int requested_device = 0;
  generic::CudaEngineKind cuda_engine = generic::CudaEngineKind::ordinary;
  int expected_rank_count = 1;
  bool require_host_staged_rank2 = false;
  // The generic CUDA topology path borrows these objects for the complete
  // engine lifetime.  It never repartitions, periodicizes, or otherwise
  // mutates the supplied ParMesh.  The legacy generated-topology path remains
  // selected unless this flag is set so frozen application callers retain
  // their existing configuration ABI.
  bool use_borrowed_topology = false;
  mfem::ParMesh* borrowed_mesh = nullptr;
  MPI_Comm borrowed_communicator = MPI_COMM_NULL;
  // Generic-session field mode is opt-in.  The default keeps the frozen W8A
  // manufactured interior/source bytes and behavior exactly intact.
  bool use_borrowed_host_inputs = false;
  CudaBorrowedHostL2Input borrowed_initial{};
  // Optional independently projected Euler true-trace coordinates.  Empty
  // retains the established derived-trace compatibility path.  IGR appends a
  // private zero sigma-hat slab after these four public component slabs.
  CudaBorrowedHostL2Input borrowed_initial_true_trace{};
  // The sole empty spelling is {nullptr, 0, 4}; it selects a zero source.
  CudaBorrowedHostL2Input borrowed_source{};
  // Nonempty only with a static physical-boundary policy table: this is the
  // rank-common caller declaration of the complete borrowed global topology.
  // It is an Initialize-only admission identity, not a mesh pointer, hash of
  // local partition bytes, or post-initialization evidence payload.
  StableIdentity borrowed_global_topology{};
  // This descriptor is valid only for borrowed-host sessions.  Initialize
  // validates and copies it; no caller policy pointer survives publication.
  CudaStaticBoundaryPolicyTable borrowed_boundary_policies{};
  // Host callback is used only while deriving a compatibility initial trace;
  // device assembly consumes the copied value-only table above.
  generic::BoundaryOperatorView borrowed_boundary{};
  // Generic requests select the same engine with either 4/4 conventional or
  // 7/5 IGR storage.  The policy is ignored by the frozen generated-topology
  // route, preserving its exact behavior.
  bool use_generic_numerical_policy = false;
  generic::EulerHdgNumericalPolicy numerical_policy =
      generic::DefaultEulerHdgNumericalPolicy(polynomial_order);
  generic::EulerHdgTimeTreatment time_treatment =
      generic::EulerHdgTimeTreatment::backward_euler;
  bool capture_oracle = false;
  bool force_private_scientific_gate_failure = false;
  CudaInjection injection{};
};

/**
 * Select the communicator that owns every collective in one CUDA engine.
 * Borrowed generic sessions retain their request/mesh communicator for the
 * full engine lifetime.  The frozen generated-topology/direct-config route
 * remains MPI_COMM_WORLD-bound for compatibility.
 */
inline MPI_Comm CudaCollectiveCommunicator(const CudaConfig& config) noexcept
{
  return config.borrowed_communicator != MPI_COMM_NULL
      ? config.borrowed_communicator : MPI_COMM_WORLD;
}

enum class CudaAllocationSite : unsigned char
{
  mfem_device_pool_probe,
  persistent_double_arena,
  persistent_integer_arena
};

enum class CudaTransferDirection : unsigned char
{
  host_to_device,
  device_to_host,
  device_to_device,
  mpi_host_staging
};

enum class CudaTransferCategory : unsigned char
{
  topology,
  initial_state,
  attempt_history,
  checkpoint_candidate_staging,
  checkpoint_restore_import,
  checkpoint_restore_materialization,
  scalar_diagnostic,
  host_staged_trace_exchange,
  oracle_export,
  generic_host_bridge,
  generic_output_metadata_staging
};

enum class CudaTransferSite : unsigned char
{
  topology_upload,
  initial_interior_upload,
  initial_trace_upload,
  attempt_seed_interior,
  attempt_seed_trace,
  nonlinear_base_interior,
  nonlinear_base_trace,
  nonlinear_base_local_trace,
  nonlinear_restore_interior,
  nonlinear_restore_trace,
  nonlinear_restore_local_trace,
  nonlinear_accept_interior,
  nonlinear_accept_trace,
  candidate_finalize_interior,
  candidate_finalize_trace,
  checkpoint_stage_interior,
  checkpoint_stage_trace,
  checkpoint_import_interior,
  checkpoint_import_trace,
  checkpoint_materialize_interior,
  checkpoint_materialize_trace,
  quadrature_upload,
  map_upload,
  matrix_remote_gather,
  matrix_remote_scatter,
  rhs_remote_gather,
  rhs_remote_scatter,
  trace_solution_stage_out,
  trace_ghost_stage_in,
  scalar_status,
  state_hash,
  oracle_export,
  generic_host_export,
  output_metadata_stage_interior,
  output_metadata_stage_trace
};

struct CudaPolicyContext
{
  std::uint64_t attempt_sequence = 0;
  std::uint64_t target_step = 0;
  std::uint64_t generation = 0;
  int rank = -1;
  CudaFailurePhase phase = CudaFailurePhase::setup;
};

struct CudaDeviceTicket
{
  CudaFailurePhase phase = CudaFailurePhase::setup;
  std::uint64_t attempt_sequence = 0;
  std::uint64_t generation = 0;
  int compiled_architecture = 0;
  int device_ordinal = -1;
  bool kernel_executed = false;
};

struct CudaProductEvidence
{
  std::uint64_t residual_generation = 0;
  std::uint64_t matrix_generation = 0;
  std::uint64_t factor_generation = 0;
  std::uint64_t solution_generation = 0;
  std::uint8_t issued_mask = 0;

  constexpr bool CoupledAndOrdered() const noexcept
  {
    return issued_mask == 0x0fU && residual_generation != 0 &&
        residual_generation == matrix_generation &&
        matrix_generation == factor_generation &&
        factor_generation == solution_generation;
  }
};

struct CudaReuseEvidence
{
  std::uintptr_t topology_token = 0;
  std::uintptr_t double_arena_token = 0;
  std::uintptr_t integer_arena_token = 0;
  std::uintptr_t matrix_token = 0;
  // Identity of the selected persistent trace preconditioner.  ILU policies
  // store the HYPRE handle; conventional tensor CUDA stores its diagonal-
  // scale object because that preconditioner has no allocated HYPRE handle.
  std::uintptr_t preconditioner_token = 0;
  std::uintptr_t workspace_token = 0;
  std::array<std::uintptr_t, 2> interior_slot_tokens{{0, 0}};
  std::array<std::uintptr_t, 2> trace_slot_tokens{{0, 0}};
  // Composite observed signal: W8-owned calls plus MFEM/HYPRE pool high-water
  // growth. This does not observe opaque churn that reuses pool capacity.
  std::uint64_t baseline_allocation_events = 0;
  std::uint64_t post_baseline_allocation_events = 0;
  // Exact application-owned policy-call boundary.
  std::uint64_t baseline_w8_owned_allocation_calls = 0;
  std::uint64_t post_baseline_w8_owned_allocation_calls = 0;
  bool observes_w8_owned_calls_and_pool_growth_only = true;
  bool opaque_pool_reuse_churn_observed = false;
  bool persistent_graph = false;
  bool persistent_slots = false;
  bool persistent_workspace = false;
};

struct CudaTransferTotals
{
  std::uint64_t host_to_device_bytes = 0;
  std::uint64_t device_to_host_bytes = 0;
  std::uint64_t device_to_device_bytes = 0;
  std::uint64_t mpi_staging_bytes = 0;
  std::uint64_t synchronization_events = 0;
  std::uint64_t event_count = 0;
  bool complete = true;
};

struct CudaTransferLedgerV1
{
  std::uint32_t schema = 1;
  CudaTransferTotals totals{};
  std::uint64_t observed_count = 0;
  std::uint64_t ordered_digest = 1469598103934665603ULL;
  bool overflow = false;
};

struct CudaAllocationLedgerV1
{
  std::uint32_t schema = 1;
  std::uint64_t observed_count = 0;
  std::uint64_t ordered_digest = 1469598103934665603ULL;
  bool overflow = false;
};

struct CudaTransferLedgerRangeV1
{
  std::uint64_t begin_ordinal = 0;
  std::uint64_t end_ordinal = 0;
  std::uint64_t ordered_digest = 0;
  bool complete = false;
};

/** Exact trace-preconditioner policy selected behind the shared FP64 GMRES. */
enum class CudaTracePreconditionerKind : std::uint8_t
{
  unconfigured = 0,
  hypre_ilu0 = 1,
  hypre_diag_scale = 2,
  hypre_mgr_field_split = 3,
  native_element_schwarz_fp64 = 4,
  native_element_schwarz_equilibrated_field_split_fp64 = 5,
  native_element_schwarz_warp_equilibrated_field_split_fp64 = 6,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64 = 7,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64 = 8,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64 = 9,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64 = 10,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64 = 11,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64 = 12,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64 = 13,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64 = 14,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64 = 15,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64 = 16,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64 = 17,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64 = 18,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64 = 19,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64 = 20,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64 = 21,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64 = 22,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64 = 23,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64 = 24,
  native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64 = 25,
  native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64 = 26,
  native_element_schwarz_warp_coupled_fp64 = 27,
  native_element_schwarz_warp_coupled_sigma_jacobi2_fp64 = 28,
  native_element_schwarz_warp_coupled_sigma_jacobi10_fp64 = 29,
  native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64 = 30
};

/** Deterministic lifetime policy for a selected HYPRE ILU(0) factor. */
enum class CudaIluFactorReusePolicy : std::uint8_t
{
  not_applicable = 0,
  refresh_each_trace_solve = 1,
  first_trace_solve_per_attempt = 2
};

/** Fixed triangular-application policy for a selected HYPRE ILU factor. */
enum class CudaIluTriangularSolveKind : std::uint8_t
{
  not_applicable = 0,
  direct = 1,
  iterative_jacobi = 2
};

constexpr const char* CudaIluTriangularSolveKindName(
    const CudaIluTriangularSolveKind kind) noexcept
{
  switch (kind)
  {
  case CudaIluTriangularSolveKind::direct:
    return "direct";
  case CudaIluTriangularSolveKind::iterative_jacobi:
    return "iterative_jacobi";
  case CudaIluTriangularSolveKind::not_applicable:
    break;
  }
  return "not_applicable";
}

constexpr const char* CudaIluFactorReusePolicyName(
    const CudaIluFactorReusePolicy policy) noexcept
{
  switch (policy)
  {
  case CudaIluFactorReusePolicy::refresh_each_trace_solve:
    return "refresh_each_trace_solve";
  case CudaIluFactorReusePolicy::first_trace_solve_per_attempt:
    return "first_trace_solve_per_attempt";
  case CudaIluFactorReusePolicy::not_applicable:
    break;
  }
  return "not_applicable";
}

/** Nonintrusive ILU-work accounting carried by aggregate solve evidence. */
enum class CudaIluAccountingKind : std::uint8_t
{
  unavailable = 0,
  derived_bounds_v1 = 1
};

constexpr const char* CudaIluAccountingKindName(
    const CudaIluAccountingKind kind) noexcept
{
  switch (kind)
  {
  case CudaIluAccountingKind::derived_bounds_v1:
    return "derived_bounds_v1";
  case CudaIluAccountingKind::unavailable:
    break;
  }
  return "unavailable";
}

constexpr CudaTracePreconditionerKind CudaTracePreconditionerForEngine(
    const generic::CudaEngineKind engine, const bool igr_active) noexcept
{
  // Diagonal scaling removes recurring ILU setup from the conventional
  // tensor engine, whose trace systems converge in a handful of iterations.
  // Both ordinary modes and coupled tensor-IGR retain the qualified ILU(0)
  // policy; IGR selects RCM through the kind-typed reordering helper below.
  // This is a typed, deterministic engine-and-mode policy, never a solve-time
  // fallback.
  return engine == generic::CudaEngineKind::tensor_core && !igr_active
      ? CudaTracePreconditionerKind::hypre_diag_scale
      : CudaTracePreconditionerKind::hypre_ilu0;
}

constexpr bool CudaTracePreconditionerIsIlu(
    const CudaTracePreconditionerKind kind) noexcept
{
  return kind == CudaTracePreconditionerKind::hypre_ilu0;
}

constexpr bool CudaTracePreconditionerIsNativeElementSchwarz(
    const CudaTracePreconditionerKind kind) noexcept
{
  return kind ==
          CudaTracePreconditionerKind::native_element_schwarz_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_coupled_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_coupled_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_coupled_sigma_jacobi10_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64;
}

/** Typed rank-one policy whose effective factor is selected per matrix. */
constexpr bool CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(
    const CudaTracePreconditionerKind kind) noexcept
{
  return kind == CudaTracePreconditionerKind::
      native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64;
}

constexpr bool
CudaTracePreconditionerIsNativeElementSchwarzEquilibratedFieldSplit(
    const CudaTracePreconditionerKind kind) noexcept
{
  return kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64;
}

/** Benchmark-only cooperative FP64 field-patch application. */
constexpr bool
CudaTracePreconditionerIsNativeElementSchwarzWarpEquilibratedFieldSplit(
    const CudaTracePreconditionerKind kind) noexcept
{
  return kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64;
}

/** Cooperative FP64 application of one fully coupled principal-patch LU. */
constexpr bool CudaTracePreconditionerIsNativeElementSchwarzWarpCoupled(
    const CudaTracePreconditionerKind kind) noexcept
{
  return kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_coupled_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_coupled_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_coupled_sigma_jacobi10_fp64;
}

inline constexpr int native_element_schwarz_sigma_jacobi3_sweeps = 3;
inline constexpr int native_element_schwarz_sigma_jacobi5_sweeps = 5;
inline constexpr int native_element_schwarz_sigma_jacobi7_sweeps = 7;
inline constexpr int native_element_schwarz_sigma_jacobi_maximum_sweeps = 10;
inline constexpr double native_element_schwarz_sigma_jacobi_damping =
    0x1.5555555555555p-1;

constexpr bool CudaTracePreconditionerUsesFixedSigmaJacobi(
    const CudaTracePreconditionerKind kind) noexcept
{
  return kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_coupled_sigma_jacobi2_fp64 ||
      kind == CudaTracePreconditionerKind::
          native_element_schwarz_warp_coupled_sigma_jacobi10_fp64 ||
      CudaTracePreconditionerIsNativeElementSchwarzRobustCoupled(kind);
}

constexpr int CudaTracePreconditionerFixedSigmaJacobiSweeps(
    const CudaTracePreconditionerKind kind) noexcept
{
  switch (kind)
  {
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64:
    return 1;
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_sigma_jacobi2_fp64:
    return 2;
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64:
    return native_element_schwarz_sigma_jacobi3_sweeps;
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64:
    return 4;
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64:
    return native_element_schwarz_sigma_jacobi5_sweeps;
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64:
    return 6;
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64:
    return native_element_schwarz_sigma_jacobi7_sweeps;
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64:
    return 8;
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64:
    return 9;
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64:
    return native_element_schwarz_sigma_jacobi_maximum_sweeps;
  default:
    return 0;
  }
}

constexpr double CudaTracePreconditionerFixedSigmaJacobiDamping(
    const CudaTracePreconditionerKind kind) noexcept
{
  return CudaTracePreconditionerUsesFixedSigmaJacobi(kind)
      ? native_element_schwarz_sigma_jacobi_damping : 0.0;
}

constexpr CudaTracePreconditionerKind
CudaTracePreconditionerForFixedSigmaJacobiSweeps(
    const int sweeps, const bool warp) noexcept
{
  switch (sweeps)
  {
  case 1:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64;
  case 2:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64;
  case native_element_schwarz_sigma_jacobi3_sweeps:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64;
  case 4:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64;
  case native_element_schwarz_sigma_jacobi5_sweeps:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64;
  case 6:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64;
  case native_element_schwarz_sigma_jacobi7_sweeps:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64;
  case 8:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64;
  case 9:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64;
  case native_element_schwarz_sigma_jacobi_maximum_sweeps:
    return warp
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64
        : CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64;
  default:
    return CudaTracePreconditionerKind::unconfigured;
  }
}

/** Exact fixed-sweep variants of the fully coupled cooperative patch. */
constexpr CudaTracePreconditionerKind
CudaTraceCoupledWarpPreconditionerForFixedSigmaJacobiSweeps(
    const int sweeps) noexcept
{
  switch (sweeps)
  {
  case 2:
    return CudaTracePreconditionerKind::
        native_element_schwarz_warp_coupled_sigma_jacobi2_fp64;
  case native_element_schwarz_sigma_jacobi_maximum_sweeps:
    return CudaTracePreconditionerKind::
        native_element_schwarz_warp_coupled_sigma_jacobi10_fp64;
  default:
    return CudaTracePreconditionerKind::unconfigured;
  }
}

/** Exact element-Schur operator selection for native rank-one policies. */
constexpr bool CudaTracePolicyUsesNativeElementMatrixFreeOperator(
    const CudaTracePolicyKind kind) noexcept
{
  return kind ==
          CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi10_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64;
}

constexpr bool CudaTracePolicyUsesNativeElementMatrixFreeWarpRobustCoupled(
    const CudaTracePolicyKind kind) noexcept
{
  return kind == CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64;
}

/** Exact matrix-free/operator and cooperative full-patch pairing. */
constexpr bool CudaTracePolicyUsesNativeElementMatrixFreeWarpCoupled(
    const CudaTracePolicyKind kind) noexcept
{
  return kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi10_fp64;
}

/** Exact matrix-free/operator and equilibrated-field-split pairing. */
constexpr bool
CudaTracePolicyUsesNativeElementMatrixFreeEquilibratedFieldSplit(
    const CudaTracePolicyKind kind) noexcept
{
  return kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi10_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi10_fp64;
}

/** Exact matrix-free/operator and cooperative field-split pairing. */
constexpr bool
CudaTracePolicyUsesNativeElementMatrixFreeWarpEquilibratedFieldSplit(
    const CudaTracePolicyKind kind) noexcept
{
  return kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi1_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi2_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi4_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi5_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi6_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi7_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi8_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi9_fp64 ||
      kind == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi10_fp64;
}

constexpr int CudaTracePolicyFixedSigmaJacobiSweeps(
    const CudaTracePolicyKind kind) noexcept
{
  switch (kind)
  {
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi1_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi1_fp64:
    return 1;
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi2_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi2_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi2_fp64:
    return 2;
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64:
    return native_element_schwarz_sigma_jacobi3_sweeps;
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi4_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi4_fp64:
    return 4;
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi5_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi5_fp64:
    return native_element_schwarz_sigma_jacobi5_sweeps;
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi6_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi6_fp64:
    return 6;
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi7_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi7_fp64:
    return native_element_schwarz_sigma_jacobi7_sweeps;
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi8_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi8_fp64:
    return 8;
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi9_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi9_fp64:
    return 9;
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi10_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi10_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi10_fp64:
  case CudaTracePolicyKind::
      native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64:
    return native_element_schwarz_sigma_jacobi_maximum_sweeps;
  default:
    return 0;
  }
}

constexpr CudaIluFactorReusePolicy CudaIluFactorReusePolicyForEngine(
    const generic::CudaEngineKind engine, const bool igr_active) noexcept
{
  if (!CudaTracePreconditionerIsIlu(
          CudaTracePreconditionerForEngine(engine, igr_active)))
    return CudaIluFactorReusePolicy::not_applicable;
  return engine == generic::CudaEngineKind::tensor_core && igr_active
      ? CudaIluFactorReusePolicy::first_trace_solve_per_attempt
      : CudaIluFactorReusePolicy::refresh_each_trace_solve;
}

constexpr int CudaIluType(
    const CudaTracePreconditionerKind kind) noexcept
{
  return CudaTracePreconditionerIsIlu(kind) ? 0 : -1;
}

constexpr int CudaIluLevelOfFill(
    const CudaTracePreconditionerKind kind) noexcept
{
  switch (kind)
  {
  case CudaTracePreconditionerKind::hypre_ilu0:
    return 0;
  case CudaTracePreconditionerKind::hypre_diag_scale:
  case CudaTracePreconditionerKind::hypre_mgr_field_split:
  case CudaTracePreconditionerKind::native_element_schwarz_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_sigma_jacobi2_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::unconfigured:
    break;
  }
  return -1;
}

constexpr int CudaIluLocalReordering(
    const CudaTracePreconditionerKind kind,
    const bool igr_active) noexcept
{
  switch (kind)
  {
  case CudaTracePreconditionerKind::hypre_ilu0:
    return igr_active ? 1 : 0;
  case CudaTracePreconditionerKind::hypre_diag_scale:
  case CudaTracePreconditionerKind::hypre_mgr_field_split:
  case CudaTracePreconditionerKind::native_element_schwarz_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_sigma_jacobi2_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64:
  case CudaTracePreconditionerKind::unconfigured:
    break;
  }
  return -1;
}

constexpr int CudaIluApplicationIterations(
    const CudaTracePreconditionerKind kind) noexcept
{
  return CudaTracePreconditionerIsIlu(kind) ? 1 : 0;
}

// Compatibility overload for existing application provenance call sites;
// production solver configuration consumes the kind-typed overload above.
constexpr int CudaIluApplicationIterations(
    const generic::CudaEngineKind engine, const bool igr_active) noexcept
{
  return CudaIluApplicationIterations(
      CudaTracePreconditionerForEngine(engine, igr_active));
}

// HYPRE's direct sparse triangular solve is the qualified ordinary-CUDA
// policy.  The coupled tensor engine uses HYPRE's deterministic iterative
// triangular solve so its recurring ILU applications avoid the dependency-
// limited sparse SpSM path.  This is a fixed engine/mode policy, never an
// adaptive fallback.
constexpr CudaIluTriangularSolveKind CudaIluTriangularSolveKindForEngine(
    const generic::CudaEngineKind engine, const bool igr_active) noexcept
{
  if (!CudaTracePreconditionerIsIlu(
          CudaTracePreconditionerForEngine(engine, igr_active)))
    return CudaIluTriangularSolveKind::not_applicable;
  return engine == generic::CudaEngineKind::tensor_core && igr_active
      ? CudaIluTriangularSolveKind::iterative_jacobi
      : CudaIluTriangularSolveKind::direct;
}

constexpr int CudaIluTriangularSolve(
    const generic::CudaEngineKind engine, const bool igr_active) noexcept
{
  switch (CudaIluTriangularSolveKindForEngine(engine, igr_active))
  {
  case CudaIluTriangularSolveKind::iterative_jacobi:
    return 0;
  case CudaIluTriangularSolveKind::direct:
    return 1;
  case CudaIluTriangularSolveKind::not_applicable:
    break;
  }
  return -1;
}

constexpr int CudaIluLowerJacobiIterations(
    const generic::CudaEngineKind engine, const bool igr_active) noexcept
{
  return CudaIluTriangularSolve(engine, igr_active) == 0 ? 5 : 0;
}

constexpr int CudaIluUpperJacobiIterations(
    const generic::CudaEngineKind engine, const bool igr_active) noexcept
{
  return CudaIluTriangularSolve(engine, igr_active) == 0 ? 5 : 0;
}

constexpr int CudaIluJacobiFactorSpmvUpdatesPerApplication(
    const generic::CudaEngineKind engine, const bool igr_active) noexcept
{
  const int lower = CudaIluLowerJacobiIterations(engine, igr_active);
  const int upper = CudaIluUpperJacobiIterations(engine, igr_active);
  // HYPRE's zero-start first iterate is a pointwise transform; only the
  // remaining iterates launch factor SpMVs.
  return (lower > 0 ? lower - 1 : 0) +
      (upper > 0 ? upper - 1 : 0);
}

constexpr const char* CudaTracePreconditionerName(
    const CudaTracePreconditionerKind kind) noexcept
{
  switch (kind)
  {
  case CudaTracePreconditionerKind::hypre_ilu0:
    return "hypre_ilu0";
  case CudaTracePreconditionerKind::hypre_diag_scale:
    return "hypre_diag_scale";
  case CudaTracePreconditionerKind::hypre_mgr_field_split:
    return "hypre_mgr_field_split";
  case CudaTracePreconditionerKind::native_element_schwarz_fp64:
    return "native_element_schwarz_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_fp64:
    return "native_element_schwarz_equilibrated_field_split_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi3_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi3_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi1_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi1_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi2_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi2_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi4_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi4_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi5_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi5_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi6_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi6_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi7_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi7_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi8_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi8_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi9_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi9_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64:
    return "native_element_schwarz_equilibrated_field_split_sigma_jacobi10_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64:
    return "native_element_schwarz_warp_equilibrated_field_split_sigma_jacobi10_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_fp64:
    return "native_element_schwarz_warp_coupled_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_sigma_jacobi2_fp64:
    return "native_element_schwarz_warp_coupled_sigma_jacobi2_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_coupled_sigma_jacobi10_fp64:
    return "native_element_schwarz_warp_coupled_sigma_jacobi10_fp64";
  case CudaTracePreconditionerKind::
      native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64:
    return "native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64";
  case CudaTracePreconditionerKind::unconfigured:
    break;
  }
  return "unconfigured";
}

/** Exact, scalar trace-solver policy resolved before device initialization. */
struct CudaTracePolicyDescriptor
{
  generic::CudaEngineKind selected_engine =
      generic::CudaEngineKind::ordinary;
  CudaTracePolicyKind requested_selection =
      CudaTracePolicyKind::engine_default;
  CudaTracePolicyKind resolved_selection =
      CudaTracePolicyKind::engine_default;
  bool igr_active = false;
  int rank_count = 0;
  int polynomial_order = 0;
  CudaTracePreconditionerKind trace_preconditioner =
      CudaTracePreconditionerKind::unconfigured;
  int ilu_type = -1;
  int ilu_level_of_fill = -1;
  int ilu_local_reordering = -1;
  int ilu_application_iterations = 0;
  CudaIluTriangularSolveKind ilu_triangular_solve_kind =
      CudaIluTriangularSolveKind::not_applicable;
  int ilu_lower_jacobi_iterations = 0;
  int ilu_upper_jacobi_iterations = 0;
  CudaIluAccountingKind ilu_accounting_kind =
      CudaIluAccountingKind::unavailable;
  CudaIluFactorReusePolicy ilu_factor_reuse_policy =
      CudaIluFactorReusePolicy::not_applicable;
  bool valid = false;
};

/**
 * Deterministic production request selected before policy resolution.
 *
 * RODAS3 reuses one common shifted operator across its causal stages.  On the
 * qualified rank-one conventional tensor surface, request the persistent
 * matrix-free trace action and retained element-Schwarz factors explicitly so
 * consensus, provenance, and the engine all describe the same architecture.
 * Every pre-existing mode keeps the engine-default request.
 */
constexpr CudaTracePolicyKind DefaultCudaTracePolicyRequest(
    const generic::CudaEngineKind engine, const bool igr_active,
    const int rank_count,
    const generic::EulerHdgTimeTreatment time_treatment) noexcept
{
  return engine == generic::CudaEngineKind::tensor_core && !igr_active &&
          rank_count == 1 &&
          time_treatment == generic::EulerHdgTimeTreatment::rodas3
      ? CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64
      : CudaTracePolicyKind::engine_default;
}

constexpr CudaTracePolicyDescriptor ResolveCudaTracePolicy(
    const generic::CudaEngineKind engine, const bool igr_active,
    const int rank_count, const int polynomial_order,
    const CudaTracePolicyKind requested_selection) noexcept
{
  CudaTracePolicyDescriptor result{};
  result.selected_engine = engine;
  result.requested_selection = requested_selection;
  result.resolved_selection = requested_selection;
  result.igr_active = igr_active;
  result.rank_count = rank_count;
  result.polynomial_order = polynomial_order;
  if (!generic::IsKnownCudaEngineKind(engine) ||
      !IsKnownCudaTracePolicyKind(requested_selection) ||
      (rank_count != 1 && rank_count != 2) ||
      polynomial_order < generic::minimum_polynomial_order ||
      polynomial_order > generic::maximum_polynomial_order)
    return result;

  // Production selection is deliberately deterministic and fail closed.  A
  // rank-one tensor-IGR request resolves engine-default before consensus or
  // allocation to the robust cooperative J10 policy for p1--p3.  That policy
  // attempts the fast equilibrated lower field split on every new matrix and
  // selects its fully coupled tensor-native leaf only for an exact structural
  // applicability rejection.  Rank two retains the explicitly reported
  // host-staged HYPRE route.  Non-default selectors remain explicit rank-one
  // ablations.
  if (requested_selection == CudaTracePolicyKind::engine_default &&
      engine == generic::CudaEngineKind::tensor_core && igr_active &&
      rank_count == 1)
  {
    result.resolved_selection = CudaTracePolicyKind::
        native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64;
  }
  else if (requested_selection != CudaTracePolicyKind::engine_default &&
           rank_count != 1)
  {
    return result;
  }
  const CudaTracePolicyKind selection = result.resolved_selection;

  // The field-split baseline is an explicit rank-one tensor-IGR experiment.
  // Rank/topology admission is performed by the private solver router; every
  // other engine/mode tuple fails here before any device or HYPRE work.
  if (selection == CudaTracePolicyKind::equilibrated_mgr)
  {
    if (engine != generic::CudaEngineKind::tensor_core || !igr_active)
      return result;
    result.trace_preconditioner =
        CudaTracePreconditionerKind::hypre_mgr_field_split;
    result.valid = true;
    return result;
  }

  // Benchmark-only native Schwarz admission. Rank, topology, and exact
  // p1--p3 patch-shape checks are completed by the private solver router and
  // initialization plan before any device setup is allowed.
  if (selection == CudaTracePolicyKind::native_element_schwarz_fp64 ||
      selection ==
          CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64)
  {
    // The plain retained patch LU and matrix-free element action are
    // component-count generic. Conventional Uhat4 uses the existing non-IGR
    // cooperative specialization; only the assembled ablation stays on its
    // previously qualified IGR surface.
    if (engine != generic::CudaEngineKind::tensor_core ||
        (!igr_active && selection != CudaTracePolicyKind::
            native_element_schwarz_matrix_free_fp64))
      return result;
    result.trace_preconditioner =
        CudaTracePreconditionerKind::native_element_schwarz_fp64;
    result.valid = true;
    return result;
  }

  // Production-capable robust tensor-native policy.  The descriptor remains
  // this exact typed selector for consensus and provenance; the rank-one
  // runtime first attempts the fast cooperative lower J10 factor for every new
  // matrix and selects the cooperative fully coupled J10 specialization only
  // after either a classified structural block-factor failure or the exact
  // first-application retained-patch rejection.  Neither branch is an
  // ordinary-CUDA or HYPRE fallback; all physical calls remain accounted.
  if (selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_robust_coupled_sigma_jacobi10_fp64)
  {
    if (engine != generic::CudaEngineKind::tensor_core || !igr_active ||
        rank_count != 1)
      return result;
    result.trace_preconditioner = CudaTracePreconditionerKind::
        native_element_schwarz_warp_robust_coupled_sigma_jacobi10_fp64;
    result.valid = true;
    return result;
  }

  // Explicit prototype of the exact fully coupled principal-patch factor.
  // One warp applies the retained pivoted FP64 LU; optional J2/J10 policies
  // append the same bounded sigma-row corrections used by the lower-split
  // ablations.  The matrix-free global action and outer true-residual gate
  // remain unchanged, and no policy has a fallback.
  if (selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi2_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_coupled_sigma_jacobi10_fp64)
  {
    if (engine != generic::CudaEngineKind::tensor_core || !igr_active)
      return result;
    const int fixed_sweeps =
        CudaTracePolicyFixedSigmaJacobiSweeps(selection);
    result.trace_preconditioner = fixed_sweeps == 0
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_coupled_fp64
        : CudaTraceCoupledWarpPreconditionerForFixedSigmaJacobiSweeps(
              fixed_sweeps);
    if (result.trace_preconditioner ==
        CudaTracePreconditionerKind::unconfigured)
      return result;
    result.valid = true;
    return result;
  }

  // Benchmark-only native multiplicative field split.  The assembled policy
  // retains the exact global FP64 CSR action; the separately typed combined
  // policy selects the exact element-Schur action.  The separately typed
  // sigma-Jacobi policies retain that same action and lower factorization,
  // then apply one through ten fixed damped global sigma-row corrections
  // inside the native preconditioner.  All retain the same FP64 outer GMRES
  // and true-residual
  // authority, and none has a fallback.
  if (selection == CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi3_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi1_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi2_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi4_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi5_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi6_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi7_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi8_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi9_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_sigma_jacobi10_fp64)
  {
    if (engine != generic::CudaEngineKind::tensor_core || !igr_active)
      return result;
    const int fixed_sweeps =
        CudaTracePolicyFixedSigmaJacobiSweeps(selection);
    result.trace_preconditioner = fixed_sweeps == 0
        ? CudaTracePreconditionerKind::
              native_element_schwarz_equilibrated_field_split_fp64
        : CudaTracePreconditionerForFixedSigmaJacobiSweeps(
              fixed_sweeps, false);
    if (result.trace_preconditioner ==
        CudaTracePreconditionerKind::unconfigured)
      return result;
    result.valid = true;
    return result;
  }

  // Cooperative-patch policies retain the exact FP64 matrix-free operator and
  // outer GMRES.  Engine-default reaches the order-selected fixed-sigma
  // policy above; explicitly requested base and fixed-sweep policies remain
  // separately typed ablations with the same bounded sigma-row corrections as
  // the scalar variants.
  if (selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi3_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi1_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi2_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi4_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi5_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi6_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi7_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi8_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi9_fp64 ||
      selection == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_sigma_jacobi10_fp64)
  {
    if (engine != generic::CudaEngineKind::tensor_core || !igr_active)
      return result;
    const int fixed_sweeps =
        CudaTracePolicyFixedSigmaJacobiSweeps(selection);
    result.trace_preconditioner = fixed_sweeps == 0
        ? CudaTracePreconditionerKind::
              native_element_schwarz_warp_equilibrated_field_split_fp64
        : CudaTracePreconditionerForFixedSigmaJacobiSweeps(
              fixed_sweeps, true);
    if (result.trace_preconditioner ==
        CudaTracePreconditionerKind::unconfigured)
      return result;
    result.valid = true;
    return result;
  }

  // common-fast deliberately gives both CUDA engines the already-qualified
  // tensor trace policy.  This changes only the benchmark trace leaf; the
  // selected CUDA engine continues to control condensation and tensor work.
  const generic::CudaEngineKind policy_engine =
      selection == CudaTracePolicyKind::common_fast
      ? generic::CudaEngineKind::tensor_core : engine;
  result.trace_preconditioner =
      CudaTracePreconditionerForEngine(policy_engine, igr_active);
  result.ilu_type = CudaIluType(result.trace_preconditioner);
  result.ilu_level_of_fill =
      CudaIluLevelOfFill(result.trace_preconditioner);
  result.ilu_local_reordering =
      CudaIluLocalReordering(result.trace_preconditioner, igr_active);
  result.ilu_application_iterations =
      CudaIluApplicationIterations(result.trace_preconditioner);
  result.ilu_triangular_solve_kind =
      CudaIluTriangularSolveKindForEngine(policy_engine, igr_active);
  result.ilu_lower_jacobi_iterations =
      CudaIluLowerJacobiIterations(policy_engine, igr_active);
  result.ilu_upper_jacobi_iterations =
      CudaIluUpperJacobiIterations(policy_engine, igr_active);
  result.ilu_accounting_kind = CudaTracePreconditionerIsIlu(
      result.trace_preconditioner)
      ? CudaIluAccountingKind::derived_bounds_v1
      : CudaIluAccountingKind::unavailable;
  result.ilu_factor_reuse_policy =
      CudaIluFactorReusePolicyForEngine(policy_engine, igr_active);
  result.valid = true;
  return result;
}

constexpr bool CudaTracePolicyDescriptorExact(
    const CudaTracePolicyDescriptor& actual,
    const CudaTracePolicyDescriptor& expected) noexcept
{
  return actual.selected_engine == expected.selected_engine &&
      actual.requested_selection == expected.requested_selection &&
      actual.resolved_selection == expected.resolved_selection &&
      actual.igr_active == expected.igr_active &&
      actual.rank_count == expected.rank_count &&
      actual.polynomial_order == expected.polynomial_order &&
      actual.trace_preconditioner == expected.trace_preconditioner &&
      actual.ilu_type == expected.ilu_type &&
      actual.ilu_level_of_fill == expected.ilu_level_of_fill &&
      actual.ilu_local_reordering == expected.ilu_local_reordering &&
      actual.ilu_application_iterations ==
          expected.ilu_application_iterations &&
      actual.ilu_triangular_solve_kind ==
          expected.ilu_triangular_solve_kind &&
      actual.ilu_lower_jacobi_iterations ==
          expected.ilu_lower_jacobi_iterations &&
      actual.ilu_upper_jacobi_iterations ==
          expected.ilu_upper_jacobi_iterations &&
      actual.ilu_accounting_kind == expected.ilu_accounting_kind &&
      actual.ilu_factor_reuse_policy ==
          expected.ilu_factor_reuse_policy &&
      actual.valid == expected.valid;
}

constexpr bool CudaTracePolicyDescriptorIsExact(
    const CudaTracePolicyDescriptor& descriptor) noexcept
{
  return descriptor.valid && CudaTracePolicyDescriptorExact(
      descriptor, ResolveCudaTracePolicy(
          descriptor.selected_engine, descriptor.igr_active,
          descriptor.rank_count, descriptor.polynomial_order,
          descriptor.requested_selection));
}

constexpr bool CudaOrdinaryRankOneRefreshPolicyIsExact(
    const CudaTracePolicyDescriptor& descriptor) noexcept
{
  return descriptor.selected_engine == generic::CudaEngineKind::ordinary &&
      CudaTracePolicyDescriptorIsExact(descriptor) &&
      descriptor.requested_selection ==
          CudaTracePolicyKind::engine_default &&
      descriptor.resolved_selection ==
          CudaTracePolicyKind::engine_default &&
      descriptor.rank_count == 1 &&
      descriptor.trace_preconditioner ==
          CudaTracePreconditionerKind::hypre_ilu0 &&
      descriptor.ilu_factor_reuse_policy ==
          CudaIluFactorReusePolicy::refresh_each_trace_solve &&
      descriptor.ilu_triangular_solve_kind ==
          CudaIluTriangularSolveKind::direct &&
      descriptor.ilu_lower_jacobi_iterations == 0 &&
      descriptor.ilu_upper_jacobi_iterations == 0;
}

/** Exact engine-default policy admitted by the two-rank host-staged leaf. */
constexpr bool CudaHostStagedRankTwoPolicyIsExact(
    const CudaTracePolicyDescriptor& descriptor) noexcept
{
  return CudaTracePolicyDescriptorIsExact(descriptor) &&
      descriptor.rank_count == 2 &&
      descriptor.requested_selection ==
          CudaTracePolicyKind::engine_default &&
      descriptor.resolved_selection ==
          CudaTracePolicyKind::engine_default &&
      descriptor.trace_preconditioner == CudaTracePreconditionerForEngine(
          descriptor.selected_engine, descriptor.igr_active) &&
      (CudaTracePreconditionerIsIlu(descriptor.trace_preconditioner) ||
       descriptor.trace_preconditioner ==
           CudaTracePreconditionerKind::hypre_diag_scale);
}

/** Production tensor-leaf selection and work census for one scope. */
struct CudaTensorCoreEvidence
{
  generic::CudaEngineKind selected_engine =
      generic::CudaEngineKind::ordinary;
  CudaTracePolicyKind requested_trace_policy =
      CudaTracePolicyKind::engine_default;
  CudaTracePolicyKind resolved_trace_policy =
      CudaTracePolicyKind::engine_default;
  int trace_policy_rank_count = 0;
  int trace_policy_order = 0;
  CudaTracePreconditionerKind trace_preconditioner =
      CudaTracePreconditionerKind::unconfigured;
  int ilu_type = -1;
  int ilu_level_of_fill = -1;
  int ilu_local_reordering = -1;
  int ilu_application_iterations = 0;
  CudaIluTriangularSolveKind ilu_triangular_solve_kind =
      CudaIluTriangularSolveKind::not_applicable;
  int ilu_lower_jacobi_iterations = 0;
  int ilu_upper_jacobi_iterations = 0;
  CudaIluAccountingKind ilu_accounting_kind =
      CudaIluAccountingKind::unavailable;
  CudaIluFactorReusePolicy ilu_factor_reuse_policy =
      CudaIluFactorReusePolicy::not_applicable;
  std::uint64_t ilu_factor_setups = 0;
  std::uint64_t ilu_factor_reuses = 0;
  std::uint64_t ilu_preconditioner_callbacks_lower_bound = 0;
  std::uint64_t ilu_preconditioner_callbacks_upper_bound = 0;
  std::uint64_t ilu_inner_iterations_lower_bound = 0;
  std::uint64_t ilu_inner_iterations_upper_bound = 0;
  std::uint64_t ilu_jacobi_factor_spmv_updates_lower_bound = 0;
  std::uint64_t ilu_jacobi_factor_spmv_updates_upper_bound = 0;
  // Maximum factor age observed in this aggregate scope, never a history.
  std::uint64_t ilu_factor_age_generations = 0;
  // Volume assembly and condensation retain distinct meanings.  These two
  // fields count successful production volume invocations and their exact
  // FP64 m8n8k4 work; mma_tiles below remains condensation-only.
  std::uint64_t volume_assembly_launches = 0;
  std::uint64_t volume_mma_tiles = 0;
  std::uint64_t condensation_launches = 0;
  std::uint64_t mma_tiles = 0;
  std::uint64_t fallback_count = 0;
  // Constant-size trace-solver provenance.  These are aggregate counters and
  // configuration predicates, never a per-solve history or event stream.
  std::uint64_t native_rank1_trace_solve_calls = 0;
  std::uint64_t hypre_trace_solve_calls = 0;
  std::uint64_t equilibration_builds = 0;
  std::uint64_t equilibration_passes = 0;
  std::uint64_t mgr_setup_calls = 0;
  std::uint64_t mgr_preconditioner_applications = 0;
  std::uint64_t mgr_f_ilu_setups = 0;
  std::uint64_t mgr_f_ilu_applications = 0;
  std::uint64_t element_schwarz_patch_setups = 0;
  std::uint64_t element_schwarz_preconditioner_applications = 0;
  std::uint64_t native_matrix_free_trace_operator_applications = 0;
  // Fixed aggregate branch census for the robust tensor-native policy.  Setup
  // and physical solve-call deltas are committed together only after the
  // effective branch, generic Schwarz evidence, and native call accounting
  // are complete.  A second physical call occurs only after the exact first
  // lower-factor application check selects the coupled leaf; it is not a
  // generic solve fallback.  No matrix, event, or per-solve history is
  // retained.
  std::uint64_t robust_lower_patch_setups = 0;
  std::uint64_t robust_lower_trace_solve_calls = 0;
  std::uint64_t robust_coupled_patch_setups = 0;
  std::uint64_t robust_coupled_trace_solve_calls = 0;
  int element_schwarz_patch_rows = 0;
  int element_schwarz_minimum_multiplicity = 0;
  int element_schwarz_maximum_multiplicity = 0;
  double minimum_left_scale = 0.0;
  double maximum_left_scale = 0.0;
  double minimum_right_scale = 0.0;
  double maximum_right_scale = 0.0;
  bool hardware_supported = false;
  bool trace_preconditioner_configured = false;
  bool native_rank1_trace_solver_configured = false;
  bool hypre_mgr_rank1_trace_solver_configured = false;
  bool native_element_schwarz_configured = false;
  bool native_matrix_free_trace_operator_configured = false;
  bool robust_lower_specialization_supported = false;
  bool robust_coupled_specialization_supported = false;
  bool robust_branch_accounting_complete = false;
  bool hypre_host_staged_rank2_trace_solver_configured = false;
  bool trace_solver_accounting_complete = false;
  bool ilu_accounting_complete = false;
  bool fallback_accounting_complete = false;
};

/** Precision/correction work emitted by the numerical engine for one attempt. */
struct CudaPrecisionWorkEvidence
{
  std::uint64_t iterative_refinement_steps = 0;
  std::uint64_t residual_replacements = 0;
  std::uint64_t precision_corrections = 0;
  bool all_fp64_policy = false;
  bool accounting_complete = false;
};

struct CudaStepEvidence
{
  CudaRunStatus status = CudaRunStatus::ready;
  CudaFailurePhase phase = CudaFailurePhase::setup;
  FailureReason contract_reason = FailureReason::none;
  AttemptIdentity attempt{};
  StateCoordinates before{};
  StateCoordinates after{};
  StateHash before_interior_hash{};
  StateHash before_trace_hash{};
  StateHash after_interior_hash{};
  StateHash after_trace_hash{};
  CudaProductEvidence products{};
  CudaTensorCoreEvidence tensor_core{};
  CudaPrecisionWorkEvidence precision_work{};
  CudaTransferTotals transfers{};
  CudaTransferLedgerRangeV1 transfer_range{};
  double initial_residual = std::numeric_limits<double>::infinity();
  double final_residual = std::numeric_limits<double>::infinity();
  double true_relative_residual = std::numeric_limits<double>::infinity();
  double temporal_error_norm = 0.0;
  double final_algebraic_residual = 0.0;
  double final_algebraic_correction_norm = 0.0;
  std::uint64_t complete_assemblies = 0;
  std::uint64_t complete_condensations = 0;
  std::uint64_t complete_recoveries = 0;
  std::uint64_t common_shift_operator_builds = 0;
  std::uint64_t common_shift_operator_reuses = 0;
  std::uint64_t condensed_rhs_refreshes = 0;
  std::uint64_t common_shift_trace_solves = 0;
  std::uint64_t common_shift_preconditioner_setups = 0;
  std::uint64_t common_shift_preconditioner_reuses = 0;
  std::uint64_t positivity_limiter_calls = 0;
  std::uint64_t positivity_limiter_active_calls = 0;
  // An admitted Zhang--Shu map cannot repair a cell whose conservative
  // physical mean is already below either configured floor.  Retain only the
  // bounded attempt aggregate so a terminal adaptive rejection can be
  // distinguished from a pointwise/stage admissibility failure; no element
  // identity or event history is recorded.
  std::uint64_t positivity_limiter_inadmissible_mean_rejections = 0;
  std::uint64_t positivity_algebraic_projections = 0;
  std::uint64_t common_shift_operator_rebuilds = 0;
  double maximum_positivity_limiter_correction_norm = 0.0;
  int nonlinear_updates = 0;
  int trace_solves = 0;
  // Work census copied from the internal numerical result without changing
  // the public generic-session API.
  int linear_iterations = 0;
  int linear_solve_attempts = 0;
  int newton_correction_linear_iterations = 0;
  int final_coherent_linear_iterations = 0;
  int scalar_closure_solves = 0;
  int scalar_closure_linear_iterations = 0;
  int scalar_closure_linear_solve_attempts = 0;
  int scalar_closure_true_residual_tightenings = 0;
  int true_residual_tightenings = 0;
  int line_search_trials = 0;
  int rejected_line_search_trials = 0;
  int source_rank = -1;
  std::array<CudaDeviceTicket, maximum_device_tickets> device_tickets{};
  std::size_t device_ticket_count = 0;
  bool committed = false;
  bool retry_available = false;
  bool collective_consensus = false;
  bool committed_bytes_preserved = false;
  bool coordinates_preserved = false;
  bool no_fallback = false;
  bool device_resident = false;
};

/** Candidate bytes and coordinates visible before contract publication. */
struct CudaPrecommitSnapshot
{
  AttemptIdentity attempt{};
  StateCoordinates committed{};
  StateCoordinates target{};
  StateHash candidate_interior_hash{};
  StateHash candidate_trace_hash{};
  StateHash candidate_local_interior_hash{};
  StateHash candidate_local_trace_hash{};
  CudaProductEvidence products{};
  std::size_t interior_coefficients = 0;
  std::size_t true_trace_coefficients = 0;
  CudaFailurePhase completed_phase =
      CudaFailurePhase::after_recovery_before_commit;
  bool numerical_candidate_ready = false;
  bool staging_must_precede_finalize = true;
  bool finalize_has_no_fallible_posthook = true;
};

/**
 * Rank-local numerical payload plus rank-common checkpoint identity.
 * The descriptor is produced by the frozen contract token.  Only FinalizeCommit
 * seals successful_commit for durable use; provisional exchange is reversible.
 */
struct CudaSerializedCheckpointV1
{
  std::uint32_t schema = 1;
  int source_rank = -1;
  int rank_count = 0;
  AttemptIdentity attempt{};
  /** Correlates one live facade staging transaction; never contract authority. */
  std::uint64_t facade_incarnation = 0;
  CheckpointDescriptor descriptor{};
  StateHash local_mesh_hash{};
  StateHash local_partition_hash{};
  StateHash local_topology_hash{};
  StateHash local_interior_hash{};
  StateHash local_trace_hash{};
  std::uint64_t interior_count = 0;
  std::uint64_t owned_true_trace_count = 0;
  std::uintptr_t interior_source_token = 0;
  std::uintptr_t trace_source_token = 0;
  std::uint64_t staging_generation = 0;
  std::array<double,
             maximum_rank_elements * maximum_interior_dofs_per_element>
      interior{};
  std::array<double, maximum_true_trace_dofs> owned_true_trace{};
  CudaTransferLedgerRangeV1 staging_range{};
  bool candidate_frozen = false;
  bool checkpoint_staged = false;
  bool output_staged = false;
  bool successful_commit = false;
};

/**
 * Affine generic-output authority.  It deliberately carries metadata only:
 * generic VTK exports the staged L2 state explicitly, so it must not be
 * bounded by the frozen checkpoint payload used by legacy restart routes.
 */
struct CudaPreparedOutputMetadataV1
{
  std::uint32_t schema = 1;
  int source_rank = -1;
  int rank_count = 0;
  AttemptIdentity attempt{};
  std::uint64_t facade_incarnation = 0;
  CheckpointDescriptor descriptor{};
  StateHash local_mesh_hash{};
  StateHash local_partition_hash{};
  StateHash local_topology_hash{};
  StateHash local_interior_hash{};
  StateHash local_trace_hash{};
  std::uint64_t interior_count = 0;
  std::uint64_t owned_true_trace_count = 0;
  std::uintptr_t interior_source_token = 0;
  std::uintptr_t trace_source_token = 0;
  std::uint64_t staging_generation = 0;
  std::uint64_t staging_host_to_device_bytes = 0;
  std::uint64_t staging_device_to_host_bytes = 0;
  std::uint64_t staging_device_to_device_bytes = 0;
  CudaTransferLedgerRangeV1 staging_range{};
  bool candidate_frozen = false;
  bool checkpoint_staged = false;
  bool output_staged = false;
  bool successful_commit = false;
};

struct CudaRunEvidence
{
  CudaRunStatus status = CudaRunStatus::ready;
  FailureReason setup_origin = FailureReason::none;
  std::array<CudaStepEvidence, maximum_attempt_records> attempts{};
  std::size_t attempt_count = 0;
  CudaReuseEvidence reuse{};
  StateCoordinates final_coordinates{};
  int rank = -1;
  int ranks = 0;
  int device = -1;
  std::array<unsigned char, 16> physical_gpu_uuid{};
  // Initialization/admission evidence only. Per-attempt launch and MMA work
  // is carried by CudaStepEvidence and is not accumulated into this record.
  CudaTensorCoreEvidence tensor_core{};
  bool same_physical_gpu = false;
  bool mfem_device_valid = false;
  bool hypre_cuda_umpire_device = false;
  bool signed_owner_ghost_semantics = false;
  bool borrowed_topology = false;
  bool host_staged_rank2 = false;
  bool no_host_solver_fallback = false;
  bool pass = false;
};

/**
 * Test/export seam.  It is deliberately POD and contains no production
 * helper, pointer, callback, allocator, or callable object.  The independent
 * oracle rebuilds physics, face signs, condensation, true residual, and
 * recovery from these bytes.
 */
struct CudaOraclePod
{
  std::uint32_t schema = 1;
  int rank = -1;
  int ranks = 0;
  int elements = 0;
  int interior_count = 0;
  int local_trace_count = 0;
  int true_trace_count = 0;
  int global_true_trace_count = 0;
  int volume_points = 0;
  int face_points = 0;
  int scalar_interior_dofs = 0;
  int scalar_face_dofs = 0;
  int interior_dofs = 0;
  int trace_dofs = 0;
  int true_csr_rows = 0;
  int true_csr_nonzeros = 0;
  std::uint64_t attempt_sequence = 0;
  std::uint64_t product_generation = 0;
  double dt = fixed_dt;
  std::array<int, maximum_rank_elements * maximum_trace_dofs_per_element>
      signed_element_trace{};
  std::array<std::int64_t, maximum_local_trace_dofs> local_to_global_true{};
  std::array<std::int64_t, maximum_true_trace_dofs> owned_global_true{};
  std::array<std::int64_t, maximum_rank_elements * dimension>
      element_coordinate_keys{};
  std::array<int, maximum_true_matrix_entries + 1> true_csr_offsets{};
  std::array<std::int64_t, maximum_true_matrix_entries> true_csr_columns{};
  std::array<double, maximum_true_matrix_entries> true_csr_values{};
  std::array<double, maximum_true_trace_dofs> true_rhs{};
  std::array<double, maximum_true_trace_dofs> true_solution{};
  std::array<double, maximum_true_trace_dofs> current_true_trace{};
  std::array<double, maximum_rank_elements * maximum_volume_points *
                         maximum_scalar_dofs_per_element>
      volume_shape{};
  std::array<double, maximum_rank_elements * maximum_volume_points *
                         maximum_scalar_dofs_per_element * dimension>
      volume_derivative{};
  std::array<double, maximum_rank_elements * maximum_volume_points>
      volume_weight{};
  std::array<double, maximum_rank_elements * maximum_volume_points * dimension>
      volume_position{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points * maximum_scalar_dofs_per_element>
      face_interior_shape{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points * maximum_scalar_dofs_per_face>
      face_trace_shape{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points>
      face_weight{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points * dimension>
      face_normal{};
  std::array<double, maximum_rank_elements * maximum_interior_dofs_per_element>
      old_interior{};
  std::array<double, maximum_rank_elements * maximum_interior_dofs_per_element>
      current_interior{};
  std::array<double, maximum_element_trace_dofs> current_local_trace{};
  std::array<double, maximum_local_trace_dofs> distributed_local_trace{};
  std::array<double, maximum_rank_elements * maximum_interior_dofs_per_element>
      residual_r{};
  std::array<double, maximum_rank_elements * maximum_trace_dofs_per_element>
      residual_g{};
  std::array<double, maximum_rank_elements * maximum_interior_dofs_per_element *
                         maximum_interior_dofs_per_element>
      a{};
  std::array<double, maximum_rank_elements * maximum_interior_dofs_per_element *
                         maximum_trace_dofs_per_element>
      b{};
  std::array<double, maximum_rank_elements * maximum_trace_dofs_per_element *
                         maximum_interior_dofs_per_element>
      c{};
  std::array<double, maximum_rank_elements * maximum_trace_dofs_per_element *
                         maximum_trace_dofs_per_element>
      d{};
  std::array<double, maximum_rank_elements * maximum_interior_dofs_per_element>
      f{};
  std::array<double, maximum_rank_elements * maximum_trace_dofs_per_element>
      g{};
  std::array<double, maximum_rank_elements * maximum_trace_dofs_per_element *
                         maximum_trace_dofs_per_element>
      schur{};
  std::array<double, maximum_rank_elements * maximum_trace_dofs_per_element>
      condensed_rhs{};
  std::array<double, maximum_rank_elements * maximum_trace_dofs_per_element>
      local_trace_increment{};
  std::array<double, maximum_rank_elements * maximum_interior_dofs_per_element>
      recovered_increment{};
};

struct CudaInteriorState
{
  double* coefficients = nullptr;
  std::size_t count = 0;
  StateHash hash{};
  StateHash local_hash{};
  std::uint64_t generation = 0;
  std::uintptr_t storage_token = 0;
  bool mfem_device_valid = false;
};

struct CudaTrueTraceState
{
  double* coefficients = nullptr;
  std::size_t count = 0;
  StateHash hash{};
  StateHash local_hash{};
  std::uint64_t generation = 0;
  std::uintptr_t storage_token = 0;
  std::uintptr_t mfem_owner_token = 0;
  bool mfem_owned = false;
  bool mfem_device_valid = false;
};

struct CudaFutureSharedAbiRequest
{
  // Minimal future coordinator-owned seam: one common phase enum and one
  // immutable backend-neutral smoke state/export schema.  CUDA does not
  // define either name in this slice.
  static constexpr std::string_view phase_enum = "FailurePhase";
  static constexpr std::string_view state_export = "PersistentStateExportV1";
  // Frozen at c87984fe: the shared affine lifecycle now provides the exact
  // reversible state boundary used here.
  static constexpr std::string_view prepare_finish =
      "FreezeCandidate -> PrepareCommit";
  static constexpr std::string_view commit_prepared =
      "CommitPrepared -> FinalizeCommit/RevertCommit";
  static constexpr std::string_view abort_prepared =
      "AbortPrepared";
  // Minimal remaining projection: shared TransferAccounting needs exact D2D
  // bytes, transfer event count, and an immutable ordered-ledger digest.  The
  // CUDA-owned V1 ledger remains authoritative until that ABI is accepted.
  static constexpr std::string_view transfer_accounting =
      "device_to_device_bytes + transfer_event_count + ordered_ledger_digest";
  // Checkpoint descriptors deliberately carry rank-common aggregate hashes;
  // this leaf separately serializes each local piece hash and payload.
  static constexpr std::string_view checkpoint_local_piece =
      "RankLocalCheckpointPieceV1";
  // The application can count every W8-owned allocator-policy call and can
  // observe MFEM/HYPRE pool high-water growth. Umpire does not expose a stable
  // application-owned callback for allocate/free churn that reuses capacity,
  // so this leaf deliberately makes no zero-all-allocator-activity claim.
  static constexpr std::string_view allocation_observation_scope =
      "W8-owned calls + pool high-water growth; opaque reuse churn unobserved";
  // FreezeCandidate is rank-local and returns optional authority.  If a
  // collective census ever observes a mixed frozen/unfrozen result, no current
  // shared operation can roll both phases back without dropping a live token.
  static constexpr std::string_view mixed_freeze_abort =
      "AbortFrozenCollective(optional<StagingToken>&&, FailureReason)";
};

static_assert(std::is_standard_layout<CudaOraclePod>::value &&
                  std::is_trivially_copyable<CudaOraclePod>::value,
              "independent CUDA oracle seam must remain POD");
static_assert(std::is_standard_layout<CudaSerializedCheckpointV1>::value &&
                  std::is_trivially_copyable<CudaSerializedCheckpointV1>::value,
              "CUDA checkpoint schema must remain POD");
static_assert(std::is_standard_layout<CudaPreparedOutputMetadataV1>::value &&
                  std::is_trivially_copyable<CudaPreparedOutputMetadataV1>::value,
              "CUDA prepared-output metadata must remain POD");
static_assert(!std::is_polymorphic<CudaConfig>::value &&
                  !std::is_polymorphic<CudaRunEvidence>::value,
              "CUDA W8A runtime values must use static dispatch");

} // namespace dg::hdg::runtime::cuda_w8a

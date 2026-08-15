#pragma once

#include "numerics/hdg/generic/contract.hpp"

/**
 * @file numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp
 * @brief Bounded smoke and functional conventional Euler HDG CPU adapter.
 *
 * The caller supplies the complete static Selection.  This CPU leaf does not
 * define the campaign's canonical case tag or any cross-backend DTO.  Its
 * active state consists of two disjoint complete slots: rank-local L2
 * coefficients and MFEM-owned true-trace coefficients.  Publication swaps
 * one backing handle and is noexcept.
 */

#include "numerics/hdg/runtime/run_contract.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>

namespace dg::hdg::runtime::cpu
{

namespace detail
{

inline constexpr std::string_view cpu_runtime_layout_identity =
    "l2-interior.true-trace-owned.component-major.v1";

} // namespace detail

inline constexpr int cpu_dimension = 2;
inline constexpr int cpu_conserved_components = 4;
inline constexpr int cpu_smoke_polynomial_order = 1;
inline constexpr int cpu_smoke_elements_per_direction = 3;
inline constexpr int cpu_smoke_assembly_quadrature_order = 11;
inline constexpr int cpu_functional_polynomial_order = 2;
inline constexpr int cpu_functional_minimum_elements_per_direction = 3;
inline constexpr int cpu_functional_principal_elements_per_direction = 6;
inline constexpr int cpu_functional_maximum_elements_per_direction = 12;
inline constexpr int cpu_functional_assembly_quadrature_order = 14;
inline constexpr std::uint64_t cpu_functional_accepted_steps = 20;
inline constexpr std::uint64_t cpu_functional_checkpoint_step = 10;
inline constexpr std::uint64_t cpu_functional_retry_target_step = 14;
// Retained smoke aliases preserve the accepted p1/N3 source contract.
inline constexpr int cpu_polynomial_order = cpu_smoke_polynomial_order;
inline constexpr int cpu_elements_per_direction =
    cpu_smoke_elements_per_direction;
inline constexpr int cpu_assembly_quadrature_order =
    cpu_smoke_assembly_quadrature_order;
inline constexpr double cpu_backward_euler_dt = 0.05;
inline constexpr double cpu_gamma = 1.4;
inline constexpr double cpu_fixed_tau = 2.5;
inline constexpr int cpu_maximum_newton_updates = 20;
inline constexpr int cpu_maximum_line_search_trials = 13;
inline constexpr double cpu_armijo_constant = 1.0e-4;
inline constexpr double cpu_nonlinear_tolerance = 1.0e-10;

enum class CpuScenario : unsigned char
{
  smoke,
  functional
};

/** CPU-local phase vocabulary pending a coordinator-owned shared ABI. */
enum class CpuFailurePhase : unsigned char
{
  none,
  setup,
  assembly,
  condensation,
  trace_solve,
  recovery,
  after_recovery_before_commit,
  diagnostics,
  checkpoint_staging,
  output_staging,
  commit,
  rollback_audit,
  terminal_evidence
};

enum class CpuAttemptStatus : unsigned char
{
  ready,
  success,
  injected_failure,
  runtime_failure,
  invalid_request,
  retry_exhausted
};

/** Test-only product protocol controls; none is the only scientific value. */
enum class CpuProductFault : unsigned char
{
  none,
  missing_solution,
  duplicate_residual,
  out_of_order_matrix
};

struct CpuPersistentConfig
{
  StableIdentity experiment{};
  StableIdentity source{};
  CpuScenario scenario = CpuScenario::smoke;
  int polynomial_order = cpu_polynomial_order;
  int elements_per_direction = cpu_elements_per_direction;
  int assembly_quadrature_order = cpu_assembly_quadrature_order;
  double dt = cpu_backward_euler_dt;
  int maximum_newton_updates = cpu_maximum_newton_updates;
  int maximum_line_search_trials = cpu_maximum_line_search_trials;
  double armijo_constant = cpu_armijo_constant;
  double nonlinear_tolerance = cpu_nonlinear_tolerance;
  double true_residual_tolerance = cpu_nonlinear_tolerance;
  bool allow_one_retry = true;
  bool force_private_scientific_gate_failure = false;
};

CpuPersistentConfig MakeCpuSmokeConfig() noexcept;
CpuPersistentConfig MakeCpuFunctionalConfig(int elements_per_direction)
    noexcept;

struct CpuFailureInjection
{
  CpuFailurePhase phase = CpuFailurePhase::none;
  std::uint64_t target_step = 0;
  int source_rank = -1;
  CpuProductFault product_fault = CpuProductFault::none;
};

struct CpuCoefficientView
{
  const double* data = nullptr;
  std::size_t size = 0;
  std::size_t capacity = 0;
};

struct CpuCanonicalEntry
{
  std::uint64_t global_id = 0;
  double value = 0.0;
};


enum class CpuOwnedAllocationSite : unsigned char
{
  slot_cell,
  slot_backing,
  slot_interior,
  slot_trace,
  topology_vector,
  local_system_vector,
  condensation_vector,
  recovery_vector,
  sparse_exchange_graph,
  sparse_exchange_buffer,
  mfem_wrapper,
  core_wrapper,
  contract_wrapper
};

struct CpuOwnedAllocationEvent
{
  CpuOwnedAllocationSite site = CpuOwnedAllocationSite::slot_cell;
  std::uint64_t ordinal = 0;
  std::size_t bytes = 0;
  std::size_t alignment = 0;
  std::size_t capacity = 0;
  std::uintptr_t backing_token = 0;
};

struct CpuWorkCensus
{
  std::uint64_t complete_assemblies = 0;
  std::uint64_t complete_condensations = 0;
  std::uint64_t complete_factor_setups = 0;
  std::uint64_t complete_trace_solves = 0;
  std::uint64_t complete_recoveries = 0;
  std::uint64_t trial_reassemblies = 0;
  std::uint64_t accepted_newton_updates = 0;
  std::uint64_t phase_entries = 0;
  std::uint64_t phase_completions = 0;
};

struct CpuStorageEvidence
{
  std::uintptr_t committed_storage_token = 0;
  std::uintptr_t candidate_storage_token = 0;
  std::size_t interior_size = 0;
  std::size_t interior_capacity = 0;
  std::size_t trace_size = 0;
  std::size_t trace_capacity = 0;
  std::size_t workspace_bytes = 0;
  std::size_t sparse_owned_term_capacity = 0;
  std::size_t sparse_outgoing_term_capacity = 0;
  std::size_t sparse_incoming_slot_capacity = 0;
  std::size_t true_diagonal_nnz = 0;
  std::size_t true_offdiagonal_nnz = 0;
  std::uintptr_t sparse_owned_term_token = 0;
  std::uintptr_t sparse_outgoing_term_token = 0;
  std::uintptr_t sparse_incoming_slot_token = 0;
  std::uint64_t topology_generation = 0;
  std::uint64_t storage_generation = 0;
  std::uint64_t allocation_baseline = 0;
  std::uint64_t allocation_events_after_baseline = 0;
  bool opaque_mfem_hypre_allocations_unobserved = true;
  bool opaque_standard_library_control_blocks_unobserved = true;
  bool publish_is_complete_slot_exchange = false;
  bool candidate_retains_exact_old_state_after_publish = false;
  bool revert_publish_is_exact_byte_inverse = false;
  bool publish_and_revert_preserve_disjoint_backing = false;
};

struct CpuAttemptEvidence
{
  CpuAttemptStatus status = CpuAttemptStatus::ready;
  CpuFailurePhase failure_phase = CpuFailurePhase::none;
  AttemptIdentity identity{};
  StateCoordinates entry_coordinates{};
  StateCoordinates exit_coordinates{};
  std::uint64_t candidate_cursor = 0;
  int rank = -1;
  int ranks = 0;
  int failure_source_rank = -1;
  int nonlinear_updates = 0;
  int linear_iterations = 0;
  int final_line_search_trials = 0;
  double initial_nonlinear_residual = 0.0;
  double final_nonlinear_residual = 0.0;
  double maximum_true_relative_residual = 0.0;
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
  double maximum_wave_speed = 0.0;
  StateHash committed_interior_before{};
  StateHash committed_trace_before{};
  StateHash committed_interior_after{};
  StateHash committed_trace_after{};
  std::uint64_t candidate_generation = 0;
  bool candidate_generation_invalidated = false;
  CpuWorkCensus work{};
  TransferAccounting transfers{0, 0, 0, 0, 0, true};
  ProhibitedActionAccounting prohibited{};
  bool finite_gate = false;
  bool admissibility_gate = false;
  bool recovery_gate = false;
  bool nonlinear_gate = false;
  bool true_residual_gate = false;
  bool rollback_byte_equal = false;
  bool committed = false;
};

struct CpuAcceptedEvidence
{
  bool valid = false;
  StateCoordinates coordinates{};
  StateHash interior_hash{};
  StateHash trace_hash{};
  StateHash rank_common_interior_hash{};
  StateHash rank_common_trace_hash{};
  CpuAttemptEvidence attempt{};
};

struct CpuRunEvidence
{
  RuntimeTupleDescriptor tuple{};
  StableIdentity experiment{};
  StableIdentity source{};
  int rank = -1;
  int ranks = 0;
  int local_elements = 0;
  int global_elements = 0;
  int polynomial_order = 0;
  int elements_per_direction = 0;
  int assembly_quadrature_order = 0;
  double dt = 0.0;
  std::uint64_t local_owned_interior_coefficients = 0;
  std::uint64_t global_owned_interior_coefficients = 0;
  std::uint64_t local_owned_true_trace_coefficients = 0;
  std::uint64_t global_owned_true_trace_coefficients = 0;
  std::uint64_t unique_periodic_faces = 0;
  std::uint64_t negative_signed_trace_dofs = 0;
  StateHash mesh_hash{};
  StateHash partition_hash{};
  StateHash topology_hash{};
  StateHash initial_interior_hash{};
  StateHash initial_trace_hash{};
  CpuStorageEvidence storage{};
  CpuAttemptEvidence last_attempt{};
  CpuAcceptedEvidence accepted{};
  std::size_t accepted_step_count = 0;
  std::uint64_t topology_builds = 0;
  std::uint64_t attempts = 0;
  std::uint64_t failed_attempts = 0;
  std::uint64_t retries = 0;
  std::uint64_t commits = 0;
};

struct CpuStateSnapshot
{
  StateCoordinates coordinates{};
  StateHash interior_hash{};
  StateHash trace_hash{};
  StateHash state_hash{};
  CpuCoefficientView interior{};
  CpuCoefficientView trace{};
  const std::uint64_t* physical_interior_ids = nullptr;
  std::size_t physical_interior_id_count = 0;
  int scalar_interior_dofs = 0;
};

template <typename Selection>
class PersistentSmoothEulerCpuAdapter;

/**
 * Attempt-qualified numerical candidate exposed only between Prepare* and
 * AbortPrepared/FinalizePrepared.  It is a read-only staging input, never a
 * published state.  A stale sequence/cursor is rejected by the facade.
 */
struct CpuPreparedStep
{
  CpuPreparedStep() noexcept = default;
  CpuPreparedStep(const CpuPreparedStep&) = delete;
  CpuPreparedStep& operator=(const CpuPreparedStep&) = delete;
  CpuPreparedStep(CpuPreparedStep&& other) noexcept { MoveFrom(other); }
  CpuPreparedStep& operator=(CpuPreparedStep&& other) noexcept
  {
    if (this != &other && !valid) { MoveFrom(other); }
    return *this;
  }

  AttemptIdentity identity{};
  std::uint64_t candidate_cursor = 0;
  StateCoordinates entry_coordinates{};
  StateCoordinates target_coordinates{};
  CheckpointDescriptor future_checkpoint{};
  CpuCoefficientView interior{};
  CpuCoefficientView trace{};
  StateHash interior_hash{};
  StateHash trace_hash{};
  StateHash state_hash{};
  std::uint64_t candidate_generation = 0;
  CpuFailurePhase failure_phase = CpuFailurePhase::none;
  bool numerically_ready = false;
  bool valid = false;

  void Invalidate() noexcept
  {
    identity = {};
    candidate_cursor = 0;
    entry_coordinates = {};
    target_coordinates = {};
    future_checkpoint = {};
    interior = {};
    trace = {};
    interior_hash = {};
    trace_hash = {};
    state_hash = {};
    candidate_generation = 0;
    failure_phase = CpuFailurePhase::none;
    numerically_ready = false;
    owner_incarnation = 0;
    authority_serial = 0;
    valid = false;
  }

private:
  std::uint64_t owner_incarnation = 0;
  std::uint64_t authority_serial = 0;
  void MoveFrom(CpuPreparedStep& other) noexcept
  {
    identity = other.identity;
    candidate_cursor = other.candidate_cursor;
    entry_coordinates = other.entry_coordinates;
    target_coordinates = other.target_coordinates;
    future_checkpoint = other.future_checkpoint;
    interior = other.interior;
    trace = other.trace;
    interior_hash = other.interior_hash;
    trace_hash = other.trace_hash;
    state_hash = other.state_hash;
    candidate_generation = other.candidate_generation;
    owner_incarnation = other.owner_incarnation;
    authority_serial = other.authority_serial;
    failure_phase = other.failure_phase;
    numerically_ready = other.numerically_ready;
    valid = other.valid;
    other.Invalidate();
  }
  template <typename Selection>
  friend class PersistentSmoothEulerCpuAdapter;
};

struct CpuPreparedCommit
{
  CpuPreparedCommit() noexcept = default;
  CpuPreparedCommit(const CpuPreparedCommit&) = delete;
  CpuPreparedCommit& operator=(const CpuPreparedCommit&) = delete;
  CpuPreparedCommit(CpuPreparedCommit&& other) noexcept { MoveFrom(other); }
  CpuPreparedCommit& operator=(CpuPreparedCommit&& other) noexcept
  {
    if (this != &other && !valid) { MoveFrom(other); }
    return *this;
  }
  CheckpointDescriptor future_checkpoint{};
  std::uint64_t candidate_generation = 0;
  bool valid = false;
  void Invalidate() noexcept
  {
    future_checkpoint = {};
    candidate_generation = 0;
    owner_incarnation = 0;
    authority_serial = 0;
    valid = false;
  }
private:
  std::uint64_t owner_incarnation = 0;
  std::uint64_t authority_serial = 0;
  void MoveFrom(CpuPreparedCommit& other) noexcept
  {
    future_checkpoint = other.future_checkpoint;
    owner_incarnation = other.owner_incarnation;
    authority_serial = other.authority_serial;
    candidate_generation = other.candidate_generation;
    valid = other.valid;
    other.Invalidate();
  }
  template <typename Selection>
  friend class PersistentSmoothEulerCpuAdapter;
};

struct CpuProvisionalCommit
{
  CpuProvisionalCommit() noexcept = default;
  CpuProvisionalCommit(const CpuProvisionalCommit&) = delete;
  CpuProvisionalCommit& operator=(const CpuProvisionalCommit&) = delete;
  CpuProvisionalCommit(CpuProvisionalCommit&& other) noexcept
  {
    MoveFrom(other);
  }
  CpuProvisionalCommit& operator=(CpuProvisionalCommit&& other) noexcept
  {
    if (this != &other && !valid) { MoveFrom(other); }
    return *this;
  }
  std::uint64_t candidate_generation = 0;
  bool valid = false;
  void Invalidate() noexcept
  {
    candidate_generation = 0;
    owner_incarnation = 0;
    authority_serial = 0;
    valid = false;
  }
private:
  std::uint64_t owner_incarnation = 0;
  std::uint64_t authority_serial = 0;
  void MoveFrom(CpuProvisionalCommit& other) noexcept
  {
    owner_incarnation = other.owner_incarnation;
    authority_serial = other.authority_serial;
    candidate_generation = other.candidate_generation;
    valid = other.valid;
    other.Invalidate();
  }
  template <typename Selection>
  friend class PersistentSmoothEulerCpuAdapter;
};

struct CpuCandidateConsumeCapability
{
  CpuCandidateConsumeCapability() noexcept = default;
  CpuCandidateConsumeCapability(const CpuCandidateConsumeCapability&) = delete;
  CpuCandidateConsumeCapability& operator=(
      const CpuCandidateConsumeCapability&) = delete;
  CpuCandidateConsumeCapability(CpuCandidateConsumeCapability&& other) noexcept
  {
    MoveFrom(other);
  }
  CpuCandidateConsumeCapability& operator=(
      CpuCandidateConsumeCapability&& other) noexcept
  {
    if (this != &other && !valid) { MoveFrom(other); }
    return *this;
  }
  void Invalidate() noexcept
  {
    owner_incarnation = 0;
    candidate_generation = 0;
    authority_serial = 0;
    valid = false;
  }
private:
  std::uint64_t owner_incarnation = 0;
  std::uint64_t candidate_generation = 0;
  std::uint64_t authority_serial = 0;
public:
  bool valid = false;
private:
  CpuCandidateConsumeCapability(const std::uint64_t owner,
                                const std::uint64_t generation,
                                const std::uint64_t serial,
                                const bool is_valid) noexcept
      : owner_incarnation(owner),
        candidate_generation(generation),
        authority_serial(serial),
        valid(is_valid)
  {}
  void MoveFrom(CpuCandidateConsumeCapability& other) noexcept
  {
    owner_incarnation = other.owner_incarnation;
    candidate_generation = other.candidate_generation;
    authority_serial = other.authority_serial;
    valid = other.valid;
    other.Invalidate();
  }
  template <typename Selection>
  friend class PersistentSmoothEulerCpuAdapter;
};

struct CpuSlotCell;

struct CpuInteriorState
{
  std::shared_ptr<CpuSlotCell> cell{};
};

struct CpuTraceState
{
  std::shared_ptr<CpuSlotCell> cell{};
};

using CpuStateSlot = CompleteStateSlot<CpuInteriorState, CpuTraceState>;

/** Single-handle, allocation-free active-state storage policy. */
struct CpuStateStoragePolicy
{
  static constexpr bool preparation_preserves_committed = true;
  static constexpr bool private_copy_is_independent = true;
  static constexpr bool publish_is_atomic = true;
  static constexpr bool publish_preserves_disjoint_backing = true;
  static constexpr bool publish_is_complete_slot_exchange = true;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = true;
  static constexpr bool publish_and_revert_preserve_disjoint_backing = true;

  static bool PrepareCandidate(const CpuStateSlot&, CpuStateSlot&) noexcept;
  static bool PrepareRestore(const CpuStateSlot&, CpuStateSlot&) noexcept;
  static bool StoreCandidate(CpuStateSlot&, const CpuStateSlot&) noexcept;
  static bool CopyOut(const CpuStateSlot&, CpuStateSlot&) noexcept;
  static void DiscardCandidate(CpuStateSlot&) noexcept;
  static bool CanPublish(const CpuStateSlot&, const CpuStateSlot&) noexcept;
  static void Publish(CpuStateSlot&, CpuStateSlot&) noexcept;
  static void RevertPublish(CpuStateSlot&, CpuStateSlot&) noexcept;
  static bool Disjoint(const CpuStateSlot&, const CpuStateSlot&) noexcept;
  static StateHash HashInterior(const CpuInteriorState&) noexcept;
  static StateHash HashTrace(const CpuTraceState&) noexcept;
};

/** MPI_COMM_WORLD, exactly-one-or-two-rank collective policy. */
struct CpuWorldCollectivePolicy
{
  static CollectiveAdmissionReport Admit(const AttemptCollectiveContext&,
                                          bool,
                                          FailureReason,
                                          std::uint32_t) noexcept;
  static CollectiveGateReport Reduce(const AttemptCollectiveContext&,
                                     const GateReport&,
                                     FailureReason,
                                     std::uint32_t) noexcept;
  static CollectiveRestoreReport ReduceRestore(
      const RestoreCollectiveContext&,
      RestartCompatibility,
      FailureReason,
      std::uint32_t) noexcept;
};

struct CpuCheckpoint
{
  CpuCheckpoint() noexcept = default;
  CpuCheckpoint(const CpuCheckpoint&) = delete;
  CpuCheckpoint& operator=(const CpuCheckpoint&) = delete;
  CpuCheckpoint(CpuCheckpoint&&) noexcept = default;
  CpuCheckpoint& operator=(CpuCheckpoint&&) noexcept = default;
  CheckpointDescriptor descriptor{};
  CpuStateSlot state{};
  bool valid = false;

  CpuCoefficientView interior() const noexcept;
  CpuCoefficientView trace() const noexcept;
};

/**
 * Collective fresh-process import.  Each rank supplies its own rank-local
 * payload; descriptor identity is rank-common and is revalidated before the
 * independently allocated checkpoint becomes valid.
 */
bool ImportCpuCheckpoint(const CheckpointDescriptor&,
                         const double* local_interior,
                         std::size_t local_interior_size,
                         const double* local_true_trace,
                         std::size_t local_true_trace_size,
                         CpuCheckpoint&) noexcept;

namespace detail
{

/**
 * Private one-shot seams used only to prove collective failure admission.
 * They are inert unless an owned leaf test explicitly arms one on all ranks.
 * Every seam is at a W8-owned boundary before an opaque communication call or
 * after such a call has returned.  There is deliberately no seam inside
 * MFEM/HYPRE/MPI: an opaque call that does not return is process-terminal and
 * is never represented as a recoverable RuntimeFailure or retry.
 */
enum class CpuCollectiveTestFault : unsigned char
{
  none,
  trace_load_preparation,
  assembly_completion,
  residual_map_preparation,
  condensation_outgoing_preparation,
  initialization_before_sparse_count_exchange,
  initialization_between_sparse_exchanges,
  initialization_before_gmres_prime,
  initialization_facade_readiness,
  initialization_facade_readiness_reset_refusal,
  commit_prepared_rejection
};

struct CpuCollectiveTestConsumptionEvidence
{
  std::uint64_t arm_generation = 0;
  std::uint64_t local_consumption_count = 0;
  std::uint64_t owner_incarnation = 0;
  std::uint64_t consumed_incarnation = 0;
  std::uint64_t last_rejected_incarnation = 0;
  std::uint64_t local_wrong_owner_rejections = 0;
  std::uint64_t local_reset_completions = 0;
  CpuCollectiveTestFault consumed_fault = CpuCollectiveTestFault::none;
  int consumed_rank = -1;
  int target_rank = -1;
  bool armed = false;
  bool reset_collectively_attested = false;
  bool thread_confined = true;
};

// Arm, execute, and observe on the same test thread.  The thread-confined
// logical state and owner incarnation prevent another adapter from consuming
// it, including another adapter running sequentially on that thread.
void ArmCpuCollectiveTestFault(CpuCollectiveTestFault, int target_rank,
                               std::uint64_t owner_incarnation) noexcept;
template <typename Selection>
void ArmCpuCollectiveTestFault(
    CpuCollectiveTestFault, int target_rank,
    const PersistentSmoothEulerCpuAdapter<Selection>& owner) noexcept;
void ResetCpuCollectiveTestFault() noexcept;
CpuCollectiveTestConsumptionEvidence ObserveCpuCollectiveTestFault() noexcept;

/** Test-only assertion of the borrowed physical-boundary Picard row. */
void ResetCpuGenericBoundaryPicardTestObservation() noexcept;
bool CpuGenericBoundaryPicardAssemblyObservedForTests() noexcept;

struct CpuCollectiveTestAccess
{
  template <typename Selection>
  static std::uint64_t CoreIncarnation(
      const PersistentSmoothEulerCpuAdapter<Selection>&) noexcept;
};

struct CpuInitialization
{
  CpuStateSlot committed{};
  CpuStateSlot candidate{};
  PersistentResourceDescriptor resources{};
  StateCoordinates coordinates{};
};

struct CpuNumericalResult
{
  CpuAttemptEvidence evidence{};
  GateReport gates{};
  FailureReason originating_failure = FailureReason::none;
};

enum class CpuBorrowedEulerOperation : unsigned char
{
  prepare = 0,
  commit = 1,
  abort = 2,
  export_prepared_buffer = 3,
  export_prepared_grid_function = 4
};

class CpuNumericsCore
{
public:
  CpuNumericsCore();
  ~CpuNumericsCore();
  CpuNumericsCore(CpuNumericsCore&&) noexcept;
  CpuNumericsCore& operator=(CpuNumericsCore&&) noexcept;
  CpuNumericsCore(const CpuNumericsCore&) = delete;
  CpuNumericsCore& operator=(const CpuNumericsCore&) = delete;

  bool Initialize(const CpuPersistentConfig&,
                  const RuntimeTupleDescriptor&,
                  CpuInitialization&) noexcept;
  /**
   * One borrowed-input invocation through this core's existing Euler HDG
   * assembly, condensation, trace solve, and recovery phases.  The mesh is
   * observed only: ownership remains with the request's caller.
   */
  bool RunBorrowedEuler(const generic::EulerHdgRequest&) noexcept;
  /** Initialize one borrowed generic Euler lifecycle without advancing it. */
  bool InitializeBorrowedEuler(const generic::EulerHdgRequest&) noexcept;
  /** Advance one borrowed generic Euler attempt and publish only on success. */
  bool AdvanceBorrowedEuler(double, StateCoordinates&) noexcept;
  /** Prepare one borrowed candidate without changing the committed slot. */
  bool PrepareBorrowedEuler(double, StateCoordinates&,
                            std::uint64_t&) noexcept;
  /** Publish exactly one active prepared candidate. */
  bool CommitBorrowedEuler(std::uint64_t, StateCoordinates&, double&,
                           double&) noexcept;
  /** Scan the committed borrowed interior and trace for CFL diagnostics. */
  bool CommittedBorrowedEulerDiagnostics(double&, double&) noexcept;
  /** Discard exactly one active prepared candidate. */
  bool AbortBorrowedEuler(std::uint64_t,
                          const StateCoordinates&) noexcept;
  /** Copy only the committed physical interior into validated host output. */
  bool ExportCommittedInterior(double*, std::size_t) const noexcept;
  bool ExportCommittedInterior(mfem::ParGridFunction&) const noexcept;
  bool ExportPreparedInterior(std::uint64_t, double*, std::size_t) const noexcept;
  bool ExportPreparedInterior(std::uint64_t,
                              mfem::ParGridFunction&) const noexcept;
  bool AdmitInitialization(bool facade_empty,
                           const CpuPersistentConfig&,
                           const RuntimeTupleDescriptor&) const noexcept;
  bool CollectiveAll(bool) const noexcept;
  bool AdmitBorrowedEulerOperation(CpuBorrowedEulerOperation) const noexcept;
  bool ResetAfterInitializationFailure() noexcept;
  bool ConsumeCollectiveTestFault(CpuCollectiveTestFault) noexcept;
  std::uint64_t incarnation() const noexcept;
  CpuNumericalResult Execute(const AttemptIdentity&,
                             const StateCoordinates&,
                             const CpuFailureInjection&) noexcept;
  void Complete(const AttemptOutcome&,
                const StateCoordinates&,
                const RunTelemetry&) noexcept;
  bool CaptureCheckpoint(const CheckpointDescriptor&,
                         CpuCheckpoint&) noexcept;
  void CompleteRestore(const RestoreOutcome&,
                       const StateCoordinates&,
                       const RunTelemetry&) noexcept;
  CpuStateSnapshot Snapshot(const StateCoordinates&) const noexcept;
  CpuStateSnapshot CandidateSnapshot(const StateCoordinates&) const noexcept;
  void MarkPreparedFailure(CpuFailurePhase, int) noexcept;
  void MarkPreparedFailure(CpuFailurePhase, int, CpuAttemptStatus) noexcept;
  bool AuthorizePublicationChain(const AttemptIdentity&,
                                 std::uint64_t candidate_generation,
                                 const StateHash& candidate_state_hash) noexcept;
  bool CanAuthorizePublicationChain(const AttemptIdentity&,
                                    std::uint64_t candidate_generation,
                                    const StateHash& candidate_state_hash) const noexcept;
  bool CandidateGenerationActive(std::uint64_t) const noexcept;
  void InvalidateCandidateGeneration() noexcept;
  bool ObserveOwnedAllocation(CpuOwnedAllocationSite,
                              std::size_t bytes,
                              std::size_t alignment,
                              std::size_t capacity,
                              const void* token) noexcept;
  void SealOwnedAllocationBaseline() noexcept;
  const CpuRunEvidence& evidence() const noexcept;
  bool ExportCanonicalCommitted(CpuCanonicalEntry*, std::size_t,
                                std::size_t&) const noexcept;
  bool ExportCanonicalCandidate(CpuCanonicalEntry*, std::size_t,
                                std::size_t&) const noexcept;

private:
  bool InitializeInternal(const CpuPersistentConfig&,
                          const RuntimeTupleDescriptor&,
                          const generic::EulerHdgRequest*,
                          CpuInitialization&) noexcept;
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
  std::uint64_t incarnation_ = 0;
};

} // namespace detail

/**
 * Caller-selected, statically dispatched persistent adapter.
 *
 * Selection must expose Descriptor() and an exact `experiment_id` string
 * view.  Tests and later CLI ownership define their tags outside this leaf.
 */
template <typename Selection>
class PersistentSmoothEulerCpuAdapter
{
public:
  using contract_type = PersistentRunContract<Selection,
                                              CpuInteriorState,
                                              CpuTraceState,
                                              CpuStateStoragePolicy,
                                              CpuWorldCollectivePolicy>;

  static_assert(!std::is_polymorphic<Selection>::value,
                "CPU persistent HDG selection must be static");

  PersistentSmoothEulerCpuAdapter() = default;
  ~PersistentSmoothEulerCpuAdapter() = default;
  PersistentSmoothEulerCpuAdapter(const PersistentSmoothEulerCpuAdapter&) =
      delete;
  PersistentSmoothEulerCpuAdapter& operator=(
      const PersistentSmoothEulerCpuAdapter&) = delete;
  PersistentSmoothEulerCpuAdapter(PersistentSmoothEulerCpuAdapter&&) = delete;
  PersistentSmoothEulerCpuAdapter& operator=(
      PersistentSmoothEulerCpuAdapter&&) = delete;

  bool Initialize(const CpuPersistentConfig& config = MakeCpuSmokeConfig())
      noexcept
  {
    static_assert(::dg::hdg::runtime::detail::SupportsTwoPhaseSlotExchange<
                      CpuStateStoragePolicy, CpuStateSlot>::value,
                  "CPU facade requires the frozen reversible-slot contract");
    const RuntimeTupleDescriptor tuple = Selection::Descriptor();
    const StableIdentity selected_experiment =
        StableIdentity::Copy(std::string_view{Selection::experiment_id});
    const bool local_preconditions = FacadeEmptyForInitialization() &&
        tuple.Valid() && selected_experiment.valid() &&
        selected_experiment == config.experiment;
    if (!core_.AdmitInitialization(local_preconditions, config, tuple))
    {
      return false;
    }

    detail::CpuInitialization initialized;
    const bool core_ready = core_.Initialize(config, tuple, initialized);
    if (!core_.CollectiveAll(core_ready))
    {
      if (!RestoreEmptyAfterInitializationFailure()) { return false; }
      return false;
    }

    std::unique_ptr<contract_type> next_contract;
    const std::uint64_t next_incarnation = core_.incarnation();
    bool local_facade_ready = false;
    try
    {
      next_contract = std::make_unique<contract_type>(
          std::move(initialized.committed), std::move(initialized.candidate),
          initialized.coordinates, initialized.resources);
      local_facade_ready = next_incarnation != 0 && next_contract &&
          core_.ObserveOwnedAllocation(
              CpuOwnedAllocationSite::contract_wrapper,
              sizeof(contract_type), alignof(contract_type), 1,
              next_contract.get()) &&
          next_contract->status() == RunStatus::ready;
      if (local_facade_ready)
      {
        const bool facade_fault = core_.ConsumeCollectiveTestFault(
            detail::CpuCollectiveTestFault::initialization_facade_readiness) ||
            core_.ConsumeCollectiveTestFault(
                detail::CpuCollectiveTestFault::
                    initialization_facade_readiness_reset_refusal);
        local_facade_ready = !facade_fault;
      }
    }
    catch (...)
    {
      local_facade_ready = false;
    }
    if (!core_.CollectiveAll(local_facade_ready))
    {
      next_contract.reset();
      if (!RestoreEmptyAfterInitializationFailure()) { return false; }
      return false;
    }

    contract_ = std::move(next_contract);
    facade_incarnation_ = next_incarnation;
    core_.SealOwnedAllocationBaseline();
    const bool published_ready = contract_ && facade_incarnation_ != 0 &&
        contract_->status() == RunStatus::ready;
    if (!core_.CollectiveAll(published_ready))
    {
      if (!RestoreEmptyAfterInitializationFailure()) { return false; }
      return false;
    }
    return true;
  }

  bool PrepareAdvance(
      CpuPreparedStep& prepared,
      const CpuFailureInjection& injection = CpuFailureInjection{}) noexcept
  {
    if (prepared.valid) { return false; }
    prepared = CpuPreparedStep{};
    if (!contract_ || FacadeBusy()) { return false; }
    const BeginOutcome begin = contract_->BeginStep(
        {InteriorSpace::l2, core_.evidence().dt});
    return begin.started && PrepareStarted(begin, injection, prepared);
  }

  bool PrepareRetry(CpuPreparedStep& prepared) noexcept
  {
    if (prepared.valid) { return false; }
    prepared = CpuPreparedStep{};
    if (!contract_ || FacadeBusy()) { return false; }
    const BeginOutcome begin = contract_->Retry();
    return begin.started && PrepareStarted(begin, {}, prepared);
  }

  std::optional<CpuCandidateConsumeCapability> AcquirePreparedCandidate(
      const CpuPreparedStep& prepared) noexcept
  {
    if (!MatchesStaging(prepared) ||
        !core_.CandidateGenerationActive(prepared.candidate_generation))
    {
      return std::nullopt;
    }
    return CpuCandidateConsumeCapability{facade_incarnation_,
                                         prepared.candidate_generation,
                                         NextAuthoritySerial(), true};
  }

  bool ConsumePreparedCandidate(CpuCandidateConsumeCapability&& capability,
                                CpuStateSnapshot& snapshot) noexcept
  {
    if (!capability.valid ||
        capability.owner_incarnation != facade_incarnation_ ||
        capability.authority_serial == 0 ||
        !core_.CandidateGenerationActive(capability.candidate_generation))
    {
      return false;
    }
    capability.Invalidate();
    snapshot = core_.CandidateSnapshot(coordinates());
    return snapshot.state_hash != StateHash{};
  }

  bool PrepareCommit(CpuPreparedStep& staging,
                     CpuPreparedCommit& prepared) noexcept
  {
    if (prepared.valid) { return false; }
    prepared = CpuPreparedCommit{};
    if (!MatchesStaging(staging) || !staging_token_) { return false; }
    const AttemptReport report = PreparedReport();
    auto next = contract_->PrepareCommit(std::move(*staging_token_), report,
                                         std::move(prepared_products_));
    staging_token_.reset();
    staging.Invalidate();
    staging_active_ = false;
    if (!next)
    {
      last_transition_outcome_ = CurrentFailureOutcome();
      CompleteAttempt(last_transition_outcome_);
      return false;
    }
    prepared_token_.emplace(std::move(*next));
    prepared.future_checkpoint = prepared_token_->future_checkpoint();
    prepared.owner_incarnation = facade_incarnation_;
    prepared.authority_serial = active_authority_serial_;
    prepared.candidate_generation = active_candidate_generation_;
    prepared.valid = true;
    commit_prepared_active_ = true;
    return true;
  }

  bool CommitPrepared(CpuPreparedCommit& prepared,
                      CpuProvisionalCommit& provisional) noexcept
  {
    if (provisional.valid) { return false; }
    provisional = CpuProvisionalCommit{};
    if (!MatchesPreparedCommit(prepared) || !prepared_token_) { return false; }
    auto next = contract_->CommitPrepared(std::move(*prepared_token_));
    prepared_token_.reset();
    prepared.Invalidate();
    commit_prepared_active_ = false;
    if (!next)
    {
      last_transition_outcome_ = CurrentFailureOutcome();
      CompleteAttempt(last_transition_outcome_);
      return false;
    }
    const bool commit_rejection = core_.ConsumeCollectiveTestFault(
        detail::CpuCollectiveTestFault::commit_prepared_rejection);
    if (!core_.CollectiveAll(!commit_rejection))
    {
      MarkFacadeFailure(CpuFailurePhase::commit, -1);
      const AttemptOutcome outcome = contract_->RevertCommit(
          std::move(*next), FailureReason::backend_failure);
      last_transition_outcome_ = outcome;
      CompleteAttempt(outcome);
      return false;
    }
    provisional_token_.emplace(std::move(*next));
    provisional.owner_incarnation = facade_incarnation_;
    provisional.authority_serial = active_authority_serial_;
    provisional.candidate_generation = active_candidate_generation_;
    provisional.valid = true;
    provisional_active_ = true;
    return true;
  }

  AttemptOutcome FinalizeCommit(CpuProvisionalCommit& provisional) noexcept
  {
    if (!MatchesProvisional(provisional) || !provisional_token_)
    {
      return StaleOutcome();
    }
    const AttemptOutcome outcome =
        contract_->FinalizeCommit(std::move(*provisional_token_));
    provisional_token_.reset();
    provisional.Invalidate();
    provisional_active_ = false;
    CompleteAttempt(outcome);
    last_transition_outcome_ = outcome;
    return outcome;
  }

  AttemptOutcome AbortPrepared(CpuPreparedCommit& prepared,
                               const CpuFailurePhase phase,
                               const int source_rank = -1) noexcept
  {
    if (!MatchesPreparedCommit(prepared) || !prepared_token_)
    {
      return StaleOutcome();
    }
    MarkFacadeFailure(phase, source_rank);
    const AttemptOutcome outcome = contract_->AbortPrepared(
        std::move(*prepared_token_), FailureReason::backend_failure);
    prepared_token_.reset();
    prepared.Invalidate();
    commit_prepared_active_ = false;
    CompleteAttempt(outcome);
    last_transition_outcome_ = outcome;
    return outcome;
  }

  AttemptOutcome RevertCommit(CpuProvisionalCommit& provisional,
                              const CpuFailurePhase phase,
                              const int source_rank = -1) noexcept
  {
    if (!MatchesProvisional(provisional) || !provisional_token_)
    {
      return StaleOutcome();
    }
    MarkFacadeFailure(phase, source_rank);
    const AttemptOutcome outcome = contract_->RevertCommit(
        std::move(*provisional_token_), FailureReason::backend_failure);
    provisional_token_.reset();
    provisional.Invalidate();
    provisional_active_ = false;
    CompleteAttempt(outcome);
    last_transition_outcome_ = outcome;
    return outcome;
  }

  AttemptOutcome AbortPrepared(CpuPreparedStep& staging,
                               const CpuFailurePhase phase,
                               const int source_rank = -1) noexcept
  {
    if (!MatchesStaging(staging)) { return StaleOutcome(); }
    MarkFacadeFailure(phase, source_rank);
    CpuPreparedCommit prepared;
    if (!PrepareCommit(staging, prepared)) { return last_transition_outcome_; }
    return AbortPrepared(prepared, phase, source_rank);
  }

  AttemptOutcome FinalizePrepared(CpuPreparedStep& staging) noexcept
  {
    if (!MatchesStaging(staging)) { return StaleOutcome(); }
    CpuPreparedCommit prepared;
    if (!PrepareCommit(staging, prepared)) { return last_transition_outcome_; }
    CpuProvisionalCommit provisional;
    if (!CommitPrepared(prepared, provisional))
    {
      return last_transition_outcome_;
    }
    return FinalizeCommit(provisional);
  }

  AttemptOutcome Advance(
      const CpuFailureInjection& injection = CpuFailureInjection{}) noexcept
  {
    CpuPreparedStep prepared;
    if (!PrepareAdvance(prepared, injection)) { return CurrentFailureOutcome(); }
    return FinalizePrepared(prepared);
  }

  AttemptOutcome Retry() noexcept
  {
    CpuPreparedStep prepared;
    if (!PrepareRetry(prepared)) { return CurrentFailureOutcome(); }
    return FinalizePrepared(prepared);
  }

  bool CaptureCheckpoint(CpuCheckpoint& checkpoint) noexcept
  {
    if (!contract_) { return false; }
    CheckpointDescriptor descriptor;
    return contract_->TryMakeCheckpoint(descriptor) &&
        core_.CaptureCheckpoint(descriptor, checkpoint);
  }

  RestoreOutcome Restore(const CpuCheckpoint& checkpoint) noexcept
  {
    if (!contract_ || !checkpoint.valid)
    {
      return {false, RestartCompatibility::policy_mismatch};
    }
    const RestoreOutcome outcome =
        contract_->Restore(checkpoint.descriptor, checkpoint.state);
    core_.CompleteRestore(outcome, contract_->coordinates(),
                          contract_->telemetry());
    return outcome;
  }

  RunStatus status() const noexcept
  {
    return contract_ ? contract_->status() : RunStatus::invalid_request;
  }
  StateCoordinates coordinates() const noexcept
  {
    return contract_ ? contract_->coordinates() : StateCoordinates{};
  }
  const RunTelemetry* contract_telemetry() const noexcept
  {
    return contract_ ? &contract_->telemetry() : nullptr;
  }
  AttemptOutcome last_transition_outcome() const noexcept
  {
    return last_transition_outcome_;
  }
  const CpuRunEvidence& evidence() const noexcept { return core_.evidence(); }
  CpuStateSnapshot Snapshot() const noexcept
  {
    return core_.Snapshot(coordinates());
  }
  CpuStateSnapshot LogicalPublishedSnapshot() const noexcept
  {
    return provisional_active_ ? core_.CandidateSnapshot(coordinates())
                               : core_.Snapshot(coordinates());
  }
  bool ExportCanonicalCommitted(CpuCanonicalEntry* entries,
                                const std::size_t capacity,
                                std::size_t& written) const noexcept
  {
    return core_.ExportCanonicalCommitted(entries, capacity, written);
  }
  bool ExportCanonicalCandidate(CpuCanonicalEntry* entries,
                                const std::size_t capacity,
                                std::size_t& written) const noexcept
  {
    return core_.ExportCanonicalCandidate(entries, capacity, written);
  }

private:
  bool FacadeEmptyForInitialization() const noexcept
  {
    return !initialization_terminal_ && !contract_ &&
        facade_incarnation_ == 0 && !staging_token_ && !prepared_token_ &&
        !provisional_token_ && !staging_active_ &&
        !commit_prepared_active_ && !provisional_active_;
  }
  void ClearFacadeInitializationState() noexcept
  {
    contract_.reset();
    staging_token_.reset();
    prepared_token_.reset();
    provisional_token_.reset();
    prepared_begin_ = {};
    prepared_numerical_ = {};
    ResetProductBundle();
    last_transition_outcome_ = {};
    facade_incarnation_ = 0;
    next_authority_serial_ = 1;
    active_authority_serial_ = 0;
    active_candidate_generation_ = 0;
    staging_active_ = false;
    commit_prepared_active_ = false;
    provisional_active_ = false;
  }
  bool RestoreEmptyAfterInitializationFailure() noexcept
  {
    ClearFacadeInitializationState();
    const bool reset_complete = core_.ResetAfterInitializationFailure();
    initialization_terminal_ = !reset_complete;
    return reset_complete;
  }
  std::uint64_t NextAuthoritySerial() noexcept
  {
    if (next_authority_serial_ == 0 ||
        next_authority_serial_ == std::numeric_limits<std::uint64_t>::max())
    {
      return 0;
    }
    return next_authority_serial_++;
  }
  bool FacadeBusy() const noexcept
  {
    return staging_active_ || commit_prepared_active_ || provisional_active_;
  }
  AttemptOutcome StaleOutcome() const noexcept
  {
    return {status(), FailureReason::stale_attempt, false,
            status() == RunStatus::retry_available};
  }
  AttemptOutcome CurrentFailureOutcome() const noexcept
  {
    return {status(), contract_ ? contract_->last_failure()
                                : FailureReason::backend_failure,
            false, status() == RunStatus::retry_available};
  }
  void ResetProductBundle() noexcept
  {
    prepared_products_.residual.reset();
    prepared_products_.matrix.reset();
    prepared_products_.factor.reset();
    prepared_products_.solution.reset();
  }
  bool IssuePreparedProducts(const BeginOutcome& begin,
                             const CpuStateSnapshot& candidate,
                             const CpuProductFault fault) noexcept
  {
    ResetProductBundle();
    if (!core_.AuthorizePublicationChain(begin.attempt,
                                          active_candidate_generation_,
                                          candidate.state_hash))
    {
      return prepared_numerical_.originating_failure != FailureReason::none;
    }
    if (fault == CpuProductFault::out_of_order_matrix)
    {
      (void)contract_->template IssueProduct<DerivedProductKind::matrix>(
          begin.attempt);
      prepared_numerical_.originating_failure = FailureReason::invalid_product;
    }
    prepared_products_.residual =
        contract_->template IssueProduct<DerivedProductKind::residual>(
            begin.attempt);
    if (fault == CpuProductFault::duplicate_residual)
    {
      (void)contract_->template IssueProduct<DerivedProductKind::residual>(
          begin.attempt);
      prepared_numerical_.originating_failure = FailureReason::invalid_product;
    }
    prepared_products_.matrix =
        contract_->template IssueProduct<DerivedProductKind::matrix>(
            begin.attempt);
    prepared_products_.factor =
        contract_->template IssueProduct<DerivedProductKind::factor>(
            begin.attempt);
    if (fault != CpuProductFault::missing_solution)
    {
      prepared_products_.solution =
          contract_->template IssueProduct<DerivedProductKind::solution>(
              begin.attempt);
    }
    if (fault != CpuProductFault::none)
    {
      prepared_numerical_.originating_failure = FailureReason::invalid_product;
      MarkFacadeFailure(CpuFailurePhase::diagnostics, -1);
    }
    return true;
  }
  bool PrepareStarted(const BeginOutcome& begin,
                      const CpuFailureInjection& injection,
                      CpuPreparedStep& prepared) noexcept
  {
    const StateCoordinates entry = contract_->coordinates();
    detail::CpuNumericalResult numerical;
    const auto capability = contract_->AcquireCandidate(begin.attempt);
    if (capability) { numerical = core_.Execute(begin.attempt, entry, injection); }
    else
    {
      numerical.originating_failure = FailureReason::stale_attempt;
      numerical.evidence.identity = begin.attempt;
      numerical.evidence.entry_coordinates = entry;
      numerical.evidence.candidate_cursor = begin.attempt.sequence;
      numerical.evidence.transfers.complete = true;
    }
    prepared_begin_ = begin;
    prepared_numerical_ = numerical;
    active_candidate_generation_ = numerical.evidence.candidate_generation;
    active_authority_serial_ = NextAuthoritySerial();
    const CpuStateSnapshot candidate = core_.CandidateSnapshot(entry);
    if (active_authority_serial_ == 0 ||
        !IssuePreparedProducts(begin, candidate, injection.product_fault))
    {
      prepared_numerical_.originating_failure = FailureReason::backend_failure;
      prepared_numerical_.gates.success = false;
    }
    auto frozen = contract_->FreezeCandidate(begin.attempt,
                                              begin.attempt.sequence);
    if (!frozen)
    {
      core_.InvalidateCandidateGeneration();
      return false;
    }
    staging_token_.emplace(std::move(*frozen));
    staging_active_ = true;
    prepared.identity = begin.attempt;
    prepared.candidate_cursor = begin.attempt.sequence;
    prepared.entry_coordinates = staging_token_->entry_coordinates();
    prepared.target_coordinates = staging_token_->target_coordinates();
    prepared.future_checkpoint = staging_token_->future_checkpoint();
    prepared.interior = candidate.interior;
    prepared.trace = candidate.trace;
    prepared.interior_hash = candidate.interior_hash;
    prepared.trace_hash = candidate.trace_hash;
    prepared.state_hash = candidate.state_hash;
    prepared.candidate_generation = active_candidate_generation_;
    prepared.owner_incarnation = facade_incarnation_;
    prepared.authority_serial = active_authority_serial_;
    prepared.failure_phase = numerical.evidence.failure_phase;
    prepared.numerically_ready = numerical.originating_failure ==
            FailureReason::none && numerical.gates.All();
    prepared.valid = true;
    return true;
  }
  bool MatchesStaging(const CpuPreparedStep& prepared) const noexcept
  {
    return staging_active_ && prepared.valid && prepared_begin_.started &&
        prepared.owner_incarnation == facade_incarnation_ &&
        prepared.authority_serial == active_authority_serial_ &&
        prepared.candidate_generation == active_candidate_generation_ &&
        ::dg::hdg::runtime::detail::SameAttempt(prepared.identity,
                                                prepared_begin_.attempt);
  }
  bool MatchesPreparedCommit(const CpuPreparedCommit& prepared) const noexcept
  {
    return commit_prepared_active_ && prepared.valid &&
        prepared.owner_incarnation == facade_incarnation_ &&
        prepared.authority_serial == active_authority_serial_ &&
        prepared.candidate_generation == active_candidate_generation_;
  }
  bool MatchesProvisional(const CpuProvisionalCommit& provisional) const noexcept
  {
    return provisional_active_ && provisional.valid &&
        provisional.owner_incarnation == facade_incarnation_ &&
        provisional.authority_serial == active_authority_serial_ &&
        provisional.candidate_generation == active_candidate_generation_;
  }
  AttemptReport PreparedReport() const noexcept
  {
    AttemptReport report;
    report.gates = prepared_numerical_.gates;
    report.telemetry.identity = prepared_begin_.attempt;
    report.telemetry.transfers = prepared_numerical_.evidence.transfers;
    report.telemetry.prohibited = prepared_numerical_.evidence.prohibited;
    report.telemetry.true_relative_residual =
        prepared_numerical_.evidence.maximum_true_relative_residual;
    report.candidate_cursor = prepared_begin_.attempt.sequence;
    report.originating_failure = prepared_numerical_.originating_failure;
    return report;
  }
  void MarkFacadeFailure(const CpuFailurePhase phase,
                         const int source_rank) noexcept
  {
    prepared_numerical_.originating_failure = FailureReason::backend_failure;
    prepared_numerical_.gates.success = false;
    prepared_numerical_.evidence.failure_phase = phase;
    prepared_numerical_.evidence.failure_source_rank = source_rank;
    const CpuAttemptStatus disposition = source_rank >= 0
        ? CpuAttemptStatus::injected_failure
        : CpuAttemptStatus::runtime_failure;
    core_.MarkPreparedFailure(phase, source_rank, disposition);
  }
  void CompleteAttempt(const AttemptOutcome& outcome) noexcept
  {
    core_.InvalidateCandidateGeneration();
    core_.Complete(outcome, contract_->coordinates(), contract_->telemetry());
    prepared_begin_ = {};
    prepared_numerical_ = {};
    active_candidate_generation_ = 0;
    active_authority_serial_ = 0;
    ResetProductBundle();
  }

  friend struct detail::CpuCollectiveTestAccess;
  detail::CpuNumericsCore core_{};
  std::unique_ptr<contract_type> contract_{};
  BeginOutcome prepared_begin_{};
  detail::CpuNumericalResult prepared_numerical_{};
  typename contract_type::ProductBundle prepared_products_{};
  std::optional<typename contract_type::StagingToken> staging_token_{};
  std::optional<typename contract_type::PreparedCommitToken> prepared_token_{};
  std::optional<typename contract_type::ProvisionalCommitToken>
      provisional_token_{};
  AttemptOutcome last_transition_outcome_{};
  std::uint64_t facade_incarnation_ = 0;
  std::uint64_t next_authority_serial_ = 1;
  std::uint64_t active_authority_serial_ = 0;
  std::uint64_t active_candidate_generation_ = 0;
  bool staging_active_ = false;
  bool commit_prepared_active_ = false;
  bool provisional_active_ = false;
  bool initialization_terminal_ = false;
};

template <typename Selection>
std::uint64_t detail::CpuCollectiveTestAccess::CoreIncarnation(
    const PersistentSmoothEulerCpuAdapter<Selection>& owner) noexcept
{
  return owner.core_.incarnation();
}

template <typename Selection>
void detail::ArmCpuCollectiveTestFault(
    const detail::CpuCollectiveTestFault fault, const int target_rank,
    const PersistentSmoothEulerCpuAdapter<Selection>& owner) noexcept
{
  detail::ArmCpuCollectiveTestFault(
      fault, target_rank,
      detail::CpuCollectiveTestAccess::CoreIncarnation(owner));
}

} // namespace dg::hdg::runtime::cpu

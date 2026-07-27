#pragma once

/**
 * @file numerics/hdg/runtime/cuda/persistent_smooth_euler_types.cuh
 * @brief CUDA-qualified value types for the bounded W8A persistent smoke path.
 *
 * This header intentionally does not define a canonical formulation tag,
 * common FailurePhase, or cross-backend DTO.  The caller supplies Selection;
 * every public value type remains qualified by cuda_w8a until a coordinator-
 * owned common ABI is accepted.
 */

#include "numerics/hdg/runtime/run_contract.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>

namespace dg::hdg::runtime::cuda_w8a
{

inline constexpr std::string_view experiment_id =
    "exp.hdg.euler.conventional.smooth_persistent.w8a.v1";
inline constexpr int polynomial_order = 1;
inline constexpr int elements_per_direction = 3;
inline constexpr double fixed_dt = 0.05;
inline constexpr int accepted_steps = 3;
inline constexpr int conserved_components = 4;
inline constexpr int dimension = 2;
inline constexpr int scalar_dofs_per_element = 4;
inline constexpr int scalar_dofs_per_face = 2;
inline constexpr int faces_per_element = 4;
inline constexpr int interior_dofs_per_element =
    conserved_components * scalar_dofs_per_element;
inline constexpr int trace_dofs_per_element =
    faces_per_element * conserved_components * scalar_dofs_per_face;
inline constexpr int maximum_rank_elements = 9;
inline constexpr int maximum_local_trace_dofs = 144;
inline constexpr int maximum_true_trace_dofs = 144;
inline constexpr int maximum_element_trace_dofs =
    maximum_rank_elements * trace_dofs_per_element;
inline constexpr int maximum_volume_points = 36;
inline constexpr int maximum_face_points = 6;
inline constexpr int maximum_local_matrix_entries =
    maximum_rank_elements * trace_dofs_per_element * trace_dofs_per_element;
inline constexpr int maximum_true_matrix_entries = 20736;
inline constexpr int maximum_exchange_entries = maximum_local_matrix_entries;
inline constexpr int maximum_attempt_records = 8;
inline constexpr int maximum_device_tickets = 128;
inline constexpr int maximum_transfer_events = 16384;
inline constexpr int maximum_allocation_events = 64;
inline constexpr int assembly_quadrature_order = 11;

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

struct CudaConfig
{
  StableIdentity experiment = StableIdentity::Copy(experiment_id);
  int order = polynomial_order;
  int mesh_n = elements_per_direction;
  double dt = fixed_dt;
  int requested_device = 0;
  int expected_rank_count = 1;
  bool require_host_staged_rank2 = false;
  bool capture_oracle = false;
  CudaInjection injection{};
};

enum class CudaAllocationSite : unsigned char
{
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
  oracle_export
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
  oracle_export
};

struct CudaPolicyContext
{
  std::uint64_t attempt_sequence = 0;
  std::uint64_t target_step = 0;
  std::uint64_t generation = 0;
  int rank = -1;
  CudaFailurePhase phase = CudaFailurePhase::setup;
};

/** Fixed-schema production transfer event; no pointer conveys authority. */
struct TransferEventV1
{
  std::uint64_t ordinal = 0;
  CudaTransferDirection direction = CudaTransferDirection::host_to_device;
  CudaTransferCategory category = CudaTransferCategory::topology;
  CudaTransferSite site = CudaTransferSite::topology_upload;
  CudaPolicyContext context{};
  std::uint64_t bytes = 0;
  std::uintptr_t source_token = 0;
  std::uintptr_t destination_token = 0;
  std::uint64_t source_capacity = 0;
  std::uint64_t destination_capacity = 0;
  std::uint64_t endpoint_token = 0;
  bool success = false;
};

/** Fixed-schema production allocation event. */
struct AllocationEventV1
{
  std::uint64_t ordinal = 0;
  CudaAllocationSite site = CudaAllocationSite::persistent_double_arena;
  CudaPolicyContext context{};
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
  std::uint64_t capacity = 0;
  std::uintptr_t storage_token = 0;
  std::uint64_t generation = 0;
  bool allocation = true;
  bool success = false;
};

using CudaTransferEvent = TransferEventV1;
using CudaAllocationEvent = AllocationEventV1;

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
  std::uintptr_t factor_token = 0;
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
  std::uint32_t capacity = maximum_transfer_events;
  CudaTransferTotals totals{};
  std::array<TransferEventV1, maximum_transfer_events> events{};
  /** Number retained in events; never exceeds events.size(). */
  std::size_t count = 0;
  std::uint64_t observed_count = 0;
  std::uint64_t dropped_count = 0;
  std::uint64_t ordered_digest = 1469598103934665603ULL;
  bool overflow = false;
};

struct CudaAllocationLedgerV1
{
  std::uint32_t schema = 1;
  std::uint32_t capacity = maximum_allocation_events;
  std::array<AllocationEventV1, maximum_allocation_events> events{};
  std::size_t count = 0;
  std::uint64_t observed_count = 0;
  std::uint64_t dropped_count = 0;
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
  CudaTransferTotals transfers{};
  CudaTransferLedgerRangeV1 transfer_range{};
  double initial_residual = std::numeric_limits<double>::infinity();
  double final_residual = std::numeric_limits<double>::infinity();
  double true_relative_residual = std::numeric_limits<double>::infinity();
  int nonlinear_updates = 0;
  int trace_solves = 0;
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
             maximum_rank_elements * interior_dofs_per_element> interior{};
  std::array<double, maximum_true_trace_dofs> owned_true_trace{};
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
  bool same_physical_gpu = false;
  bool mfem_device_valid = false;
  bool hypre_cuda_umpire_device = false;
  bool signed_owner_ghost_semantics = false;
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
  int true_csr_rows = 0;
  int true_csr_nonzeros = 0;
  std::uint64_t attempt_sequence = 0;
  std::uint64_t product_generation = 0;
  double dt = fixed_dt;
  std::array<int, maximum_rank_elements * trace_dofs_per_element>
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
                         scalar_dofs_per_element>
      volume_shape{};
  std::array<double, maximum_rank_elements * maximum_volume_points *
                         scalar_dofs_per_element * dimension>
      volume_derivative{};
  std::array<double, maximum_rank_elements * maximum_volume_points>
      volume_weight{};
  std::array<double, maximum_rank_elements * maximum_volume_points * dimension>
      volume_position{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points * scalar_dofs_per_element>
      face_interior_shape{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points * scalar_dofs_per_face>
      face_trace_shape{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points>
      face_weight{};
  std::array<double, maximum_rank_elements * faces_per_element *
                         maximum_face_points * dimension>
      face_normal{};
  std::array<double, maximum_rank_elements * interior_dofs_per_element>
      old_interior{};
  std::array<double, maximum_rank_elements * interior_dofs_per_element>
      current_interior{};
  std::array<double, maximum_element_trace_dofs> current_local_trace{};
  std::array<double, maximum_local_trace_dofs> distributed_local_trace{};
  std::array<double, maximum_rank_elements * interior_dofs_per_element>
      residual_r{};
  std::array<double, maximum_rank_elements * trace_dofs_per_element>
      residual_g{};
  std::array<double, maximum_rank_elements * interior_dofs_per_element *
                         interior_dofs_per_element>
      a{};
  std::array<double, maximum_rank_elements * interior_dofs_per_element *
                         trace_dofs_per_element>
      b{};
  std::array<double, maximum_rank_elements * trace_dofs_per_element *
                         interior_dofs_per_element>
      c{};
  std::array<double, maximum_rank_elements * trace_dofs_per_element *
                         trace_dofs_per_element>
      d{};
  std::array<double, maximum_rank_elements * interior_dofs_per_element> f{};
  std::array<double, maximum_rank_elements * trace_dofs_per_element> g{};
  std::array<double, maximum_rank_elements * trace_dofs_per_element *
                         trace_dofs_per_element>
      schur{};
  std::array<double, maximum_rank_elements * trace_dofs_per_element>
      condensed_rhs{};
  std::array<double, maximum_rank_elements * trace_dofs_per_element>
      local_trace_increment{};
  std::array<double, maximum_rank_elements * interior_dofs_per_element>
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
static_assert(std::is_standard_layout<TransferEventV1>::value &&
                  std::is_trivially_copyable<TransferEventV1>::value &&
                  std::is_standard_layout<AllocationEventV1>::value &&
                  std::is_trivially_copyable<AllocationEventV1>::value &&
                  std::is_standard_layout<CudaSerializedCheckpointV1>::value &&
                  std::is_trivially_copyable<CudaSerializedCheckpointV1>::value,
              "CUDA accounting and checkpoint schemas must remain POD");
static_assert(!std::is_polymorphic<CudaConfig>::value &&
                  !std::is_polymorphic<CudaRunEvidence>::value,
              "CUDA W8A runtime values must use static dispatch");

} // namespace dg::hdg::runtime::cuda_w8a

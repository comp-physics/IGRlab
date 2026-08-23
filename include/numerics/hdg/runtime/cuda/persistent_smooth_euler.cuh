#pragma once

/**
 * @file numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh
 * @brief Persistent conventional smooth-Euler CUDA smoke/functional engine.
 *
 * Numerical provenance: the pointwise Euler flux/source, W7 weak signs,
 * A/B/C/D condensation, HypreGMRES+HypreILU(0), and recovery equations are a
 * minimal port of accepted W7 commit ac271a038c9d26648fce8f2ebf6afb0113a92495.
 * The W7 monolithic RunSmoothEulerDeviceFullExperiment lifecycle is neither
 * included nor called.  Persistence, contract publication, retry, policies,
 * and exported POD are W8A-owned code.
 */

#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_types.cuh"
#include "numerics/hdg/runtime/runtime_selection.hpp"
#include "numerics/hdg/trace/trace_dg_mpi.hpp"
#include "physics/bc/boundary_state.hpp"

#include <HYPRE_config.h>
#include <cuda_runtime.h>
#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a
{

namespace detail
{
struct CudaQualificationTestAccess;
}

/** Production W8-owned allocator. All explicit device allocation is routed here. */
struct UmpireDeviceAllocatorPolicy
{
  static void ResetLedger() noexcept;
  /** Register MFEM's lazy DEVICE_UMPIRE pool through one recorded MFEM probe. */
  static bool PrepareDeviceArenaAllocator() noexcept;
  static bool Allocate(std::size_t bytes, std::size_t alignment,
                       CudaAllocationSite site, void*& pointer) noexcept;
  /** Return true only when the policy attests the exact release completed. */
  static bool Deallocate(void* pointer, CudaAllocationSite site) noexcept;
  static std::uint64_t AllocationEvents() noexcept;
  static std::uint64_t W8OwnedAllocationCalls() noexcept;
  static std::uint64_t DeallocationEvents() noexcept;
  static bool DevicePoolPolicyValid() noexcept;
  static CudaAllocationLedgerV1 Ledger() noexcept;
};

/** Production W8-owned transport. All explicit CUDA copies route here. */
struct CudaTransportPolicy
{
  static void ResetLedger() noexcept;
  static void BeginAttempt(const CudaPolicyContext& context) noexcept;
  static void SetPhase(CudaFailurePhase phase, std::uint64_t generation) noexcept;
  static bool HostToDevice(void* destination, const void* source,
                           std::size_t bytes, CudaTransferCategory category,
                           CudaTransferSite site,
                           std::size_t source_capacity = 0,
                           std::size_t destination_capacity = 0) noexcept;
  static bool DeviceToHost(void* destination, const void* source,
                           std::size_t bytes, CudaTransferCategory category,
                           CudaTransferSite site,
                           std::size_t source_capacity = 0,
                           std::size_t destination_capacity = 0) noexcept;
  static bool DeviceToDevice(void* destination, const void* source,
                             std::size_t bytes, CudaTransferCategory category,
                             CudaTransferSite site,
                             std::size_t source_capacity = 0,
                             std::size_t destination_capacity = 0) noexcept;
  static bool RecordMpiStaging(std::size_t bytes,
                               CudaTransferCategory category,
                               CudaTransferSite site,
                               const void* source,
                               const void* destination,
                               std::size_t source_capacity,
                               std::size_t destination_capacity,
                               bool success) noexcept;
  static bool Synchronize(CudaTransferSite site) noexcept;
  static CudaTransferTotals Accounting() noexcept;
  static CudaTransferLedgerV1 Ledger() noexcept;
  static CudaTransferLedgerRangeV1 AttemptRange() noexcept;
  static void BeginStagingRange() noexcept;
  static CudaTransferLedgerRangeV1 EndStagingRange() noexcept;
};

/** Real rank-1/rank-2 reduction policy consumed by PersistentRunContract. */
struct CudaMpiCollectivePolicy
{
  explicit CudaMpiCollectivePolicy(
      const MPI_Comm communicator = MPI_COMM_WORLD) noexcept
      : communicator_(communicator)
  {}

  CollectiveAdmissionReport Admit(
      const AttemptCollectiveContext& context, bool prepared,
      FailureReason reason, std::uint32_t expected_ranks) const noexcept;
  CollectiveGateReport Reduce(
      const AttemptCollectiveContext& context, const GateReport& gates,
      FailureReason reason, std::uint32_t expected_ranks) const noexcept;
  CollectiveRestoreReport ReduceRestore(
      const RestoreCollectiveContext& context,
      RestartCompatibility compatibility, FailureReason reason,
      std::uint32_t expected_ranks) const noexcept;

private:
  MPI_Comm communicator_ = MPI_COMM_WORLD;
};

namespace detail
{

// Stable private prestage-denial mask (bit set means the named predicate
// failed): finite, admissible, recovery, true residual, telemetry,
// no-fallback, success, not-injected, and the existing collective result.
enum CudaPrestageFailureBit : std::uint64_t
{
  prestage_finite_failure = std::uint64_t{1} << 0U,
  prestage_admissible_failure = std::uint64_t{1} << 1U,
  prestage_recovery_failure = std::uint64_t{1} << 2U,
  prestage_true_residual_failure = std::uint64_t{1} << 3U,
  prestage_telemetry_failure = std::uint64_t{1} << 4U,
  prestage_no_fallback_failure = std::uint64_t{1} << 5U,
  prestage_success_failure = std::uint64_t{1} << 6U,
  prestage_injected_failure = std::uint64_t{1} << 7U,
  prestage_collective_failure = std::uint64_t{1} << 8U
};

inline constexpr std::uint64_t CudaPrestageFailureMask(
    const bool finite, const bool admissible, const bool recovery,
    const bool true_residual, const bool telemetry, const bool no_fallback,
    const bool success, const bool injected,
    const bool collective_result) noexcept
{
  return (!finite ? prestage_finite_failure : 0) |
      (!admissible ? prestage_admissible_failure : 0) |
      (!recovery ? prestage_recovery_failure : 0) |
      (!true_residual ? prestage_true_residual_failure : 0) |
      (!telemetry ? prestage_telemetry_failure : 0) |
      (!no_fallback ? prestage_no_fallback_failure : 0) |
      (!success ? prestage_success_failure : 0) |
      (injected ? prestage_injected_failure : 0) |
      (!collective_result ? prestage_collective_failure : 0);
}

static inline void ReportCudaPrestageDenial(
    const int rank, const std::uint64_t failure_mask,
    const FailureReason reason, const CudaFailurePhase phase,
    const double initial_residual, const double final_residual,
    const double true_relative_residual, const int nonlinear_updates,
    const int trace_solves, const CudaTransferTotals& transfers,
    const std::uint64_t allocation_current,
    const std::uint64_t allocation_baseline,
    const std::uint64_t allocation_calls_current,
    const std::uint64_t allocation_calls_baseline) noexcept
{
  std::fprintf(stderr,
      "CUDA_PRESTAGE_DENY rank=%d mask=0x%llx reason=%u phase=%u "
      "residuals=%.17g/%.17g/%.17g updates=%d solves=%d "
      "transfer=%u/%llu/%llu/%llu/%llu/%llu/%llu alloc=%llu/%llu "
      "calls=%llu/%llu\n",
      rank, static_cast<unsigned long long>(failure_mask),
      static_cast<unsigned>(reason), static_cast<unsigned>(phase),
      initial_residual, final_residual, true_relative_residual,
      nonlinear_updates, trace_solves, transfers.complete ? 1U : 0U,
      static_cast<unsigned long long>(transfers.event_count),
      static_cast<unsigned long long>(transfers.synchronization_events),
      static_cast<unsigned long long>(transfers.host_to_device_bytes),
      static_cast<unsigned long long>(transfers.device_to_host_bytes),
      static_cast<unsigned long long>(transfers.device_to_device_bytes),
      static_cast<unsigned long long>(transfers.mpi_staging_bytes),
      static_cast<unsigned long long>(allocation_current),
      static_cast<unsigned long long>(allocation_baseline),
      static_cast<unsigned long long>(allocation_calls_current),
      static_cast<unsigned long long>(allocation_calls_baseline));
}

enum class CudaCopyOutPurpose : unsigned char
{
  none,
  attempt_seed,
  checkpoint_staging,
  output_metadata_staging
};

inline bool SameRange(const double* first, std::size_t first_count,
                      const double* second, std::size_t second_count) noexcept
{
  if (first == nullptr || second == nullptr) { return false; }
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto first_end = first_begin + first_count * sizeof(double);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  const auto second_end = second_begin + second_count * sizeof(double);
  return first_begin < second_end && second_begin < first_end;
}

template <typename TransportPolicy>
struct CudaContractStoragePolicy
{
  using state_slot = CompleteStateSlot<CudaInteriorState, CudaTrueTraceState>;

  static constexpr bool preparation_preserves_committed = true;
  static constexpr bool private_copy_is_independent = true;
  static constexpr bool publish_is_atomic = true;
  static constexpr bool publish_preserves_disjoint_backing = true;
  static constexpr bool publish_is_complete_slot_exchange = true;
  static constexpr bool candidate_retains_exact_old_state_after_publish = true;
  static constexpr bool revert_publish_is_exact_byte_inverse = true;
  static constexpr bool publish_and_revert_preserve_disjoint_backing = true;

  static bool CopyInterior(CudaInteriorState& destination,
                           const CudaInteriorState& source,
                           CudaTransferCategory category,
                           CudaTransferSite site) noexcept
  {
    if (destination.coefficients == nullptr || source.coefficients == nullptr ||
        destination.count != source.count ||
        !TransportPolicy::DeviceToDevice(
            destination.coefficients, source.coefficients,
            source.count * sizeof(double), category, site))
    {
      return false;
    }
    destination.hash = source.hash;
    destination.local_hash = source.local_hash;
    destination.generation = source.generation;
    destination.mfem_device_valid = source.mfem_device_valid;
    return true;
  }

  static bool CopyTrace(CudaTrueTraceState& destination,
                        const CudaTrueTraceState& source,
                        CudaTransferCategory category,
                        CudaTransferSite site) noexcept
  {
    if (destination.coefficients == nullptr || source.coefficients == nullptr ||
        destination.count != source.count ||
        !TransportPolicy::DeviceToDevice(
            destination.coefficients, source.coefficients,
            source.count * sizeof(double), category, site))
    {
      return false;
    }
    destination.hash = source.hash;
    destination.local_hash = source.local_hash;
    destination.generation = source.generation;
    destination.mfem_device_valid = source.mfem_device_valid;
    return true;
  }

  static bool PrepareCandidate(const state_slot& committed,
                               state_slot& candidate) noexcept
  {
    const bool interior = CopyInterior(
        candidate.interior, committed.interior,
        CudaTransferCategory::attempt_history,
        CudaTransferSite::attempt_seed_interior);
    const bool trace = CopyTrace(candidate.trace, committed.trace,
        CudaTransferCategory::attempt_history,
        CudaTransferSite::attempt_seed_trace);
    const bool synchronized =
        TransportPolicy::Synchronize(CudaTransferSite::attempt_seed_trace);
    return interior && trace && synchronized;
  }

  static bool PrepareRestore(const state_slot& loaded,
                             state_slot& candidate) noexcept
  {
    const bool interior = CopyInterior(
        candidate.interior, loaded.interior,
        CudaTransferCategory::checkpoint_restore_materialization,
        CudaTransferSite::checkpoint_materialize_interior);
    const bool trace = CopyTrace(
        candidate.trace, loaded.trace,
        CudaTransferCategory::checkpoint_restore_materialization,
        CudaTransferSite::checkpoint_materialize_trace);
    const bool synchronized = TransportPolicy::Synchronize(
        CudaTransferSite::checkpoint_materialize_trace);
    return interior && trace && synchronized;
  }

  static bool StoreCandidate(state_slot& candidate,
                             const state_slot& value) noexcept
  {
    const bool interior = CopyInterior(
        candidate.interior, value.interior,
        CudaTransferCategory::attempt_history,
        CudaTransferSite::candidate_finalize_interior);
    const bool trace = CopyTrace(
        candidate.trace, value.trace,
        CudaTransferCategory::attempt_history,
        CudaTransferSite::candidate_finalize_trace);
    const bool synchronized = TransportPolicy::Synchronize(
        CudaTransferSite::candidate_finalize_trace);
    return interior && trace && synchronized;
  }

  static bool CopyOut(const state_slot& source, state_slot& copy) noexcept
  {
    const CudaCopyOutPurpose purpose = copy_out_purpose_;
    copy_out_purpose_ = CudaCopyOutPurpose::none;
    if (purpose == CudaCopyOutPurpose::attempt_seed)
    {
      const bool interior = CopyInterior(
          copy.interior, source.interior,
          CudaTransferCategory::attempt_history,
          CudaTransferSite::attempt_seed_interior);
      const bool trace = CopyTrace(
          copy.trace, source.trace, CudaTransferCategory::attempt_history,
          CudaTransferSite::attempt_seed_trace);
      const bool synchronized =
          TransportPolicy::Synchronize(CudaTransferSite::attempt_seed_trace);
      return interior && trace && synchronized;
    }
    if (purpose == CudaCopyOutPurpose::checkpoint_staging)
    {
      const bool interior = CopyInterior(
          copy.interior, source.interior,
          CudaTransferCategory::checkpoint_candidate_staging,
          CudaTransferSite::checkpoint_stage_interior);
      const bool trace = CopyTrace(
          copy.trace, source.trace,
          CudaTransferCategory::checkpoint_candidate_staging,
          CudaTransferSite::checkpoint_stage_trace);
      const bool synchronized = TransportPolicy::Synchronize(
          CudaTransferSite::checkpoint_stage_trace);
      return interior && trace && synchronized;
    }
    if (purpose == CudaCopyOutPurpose::output_metadata_staging)
    {
      const bool interior = CopyInterior(
          copy.interior, source.interior,
          CudaTransferCategory::generic_output_metadata_staging,
          CudaTransferSite::output_metadata_stage_interior);
      const bool trace = CopyTrace(
          copy.trace, source.trace,
          CudaTransferCategory::generic_output_metadata_staging,
          CudaTransferSite::output_metadata_stage_trace);
      const bool synchronized = TransportPolicy::Synchronize(
          CudaTransferSite::output_metadata_stage_trace);
      return interior && trace && synchronized;
    }
    return false;
  }

  static void SetCopyOutPurpose(const CudaCopyOutPurpose purpose) noexcept
  {
    copy_out_purpose_ = purpose;
  }

  static void DiscardCandidate(state_slot& candidate) noexcept
  {
    candidate.interior.generation = 0;
    candidate.trace.generation = 0;
    candidate.interior.hash = {};
    candidate.trace.hash = {};
    candidate.interior.local_hash = {};
    candidate.trace.local_hash = {};
  }

  static bool CanPublish(const state_slot& committed,
                         const state_slot& candidate) noexcept
  {
    return Disjoint(committed, candidate) &&
        committed.interior.count == candidate.interior.count &&
        committed.trace.count == candidate.trace.count &&
        committed.interior.mfem_device_valid &&
        candidate.interior.mfem_device_valid && committed.trace.mfem_owned &&
        candidate.trace.mfem_owned && committed.trace.mfem_device_valid &&
        candidate.trace.mfem_device_valid;
  }

  static void Publish(state_slot& committed, state_slot& candidate) noexcept
  {
    using std::swap;
    swap(committed, candidate); // noexcept handle publication; payloads stay put
  }

  static void RevertPublish(state_slot& committed,
                            state_slot& candidate) noexcept
  {
    using std::swap;
    swap(committed, candidate);
  }

  static bool Disjoint(const state_slot& first,
                       const state_slot& second) noexcept
  {
    return !SameRange(first.interior.coefficients, first.interior.count,
                      second.interior.coefficients, second.interior.count) &&
        !SameRange(first.trace.coefficients, first.trace.count,
                   second.trace.coefficients, second.trace.count) &&
        first.interior.storage_token != second.interior.storage_token &&
        first.trace.storage_token != second.trace.storage_token;
  }

  static StateHash HashInterior(const CudaInteriorState& state) noexcept
  {
    return state.hash;
  }
  static StateHash HashTrace(const CudaTrueTraceState& state) noexcept
  {
    return state.hash;
  }

private:
  inline static thread_local CudaCopyOutPurpose copy_out_purpose_ =
      CudaCopyOutPurpose::none;

};

struct CudaNumericalResult
{
  bool candidate_ready = false;
  bool finite = false;
  bool admissible = false;
  bool recovery = false;
  bool true_residual = false;
  bool conservation = true;
  bool telemetry = false;
  bool success = false;
  bool injected = false;
  int injection_source_rank = -1;
  CudaFailurePhase phase = CudaFailurePhase::setup;
  FailureReason reason = FailureReason::backend_failure;
  CudaProductEvidence products{};
  double initial_residual = std::numeric_limits<double>::infinity();
  double final_residual = std::numeric_limits<double>::infinity();
  double true_relative_residual = std::numeric_limits<double>::infinity();
  double maximum_conservation_error = 0.0;
  double maximum_conservation_relative_error = 0.0;
  int nonlinear_updates = 0;
  int trace_solves = 0;
  // Sum HYPRE GMRES iterations from every trace solve, including the final
  // coherent reconstruction solve.  Line-search trials count every attempted
  // alpha; rejected trials are the attempts restored before the next alpha.
  int linear_iterations = 0;
  int line_search_trials = 0;
  int rejected_line_search_trials = 0;
};

template <typename AllocatorPolicy, typename TransportPolicy>
class CudaPersistentNumerics;

inline bool ExactId(const StableIdentity& actual,
                    const std::string_view expected) noexcept
{
  return actual == StableIdentity::Copy(expected);
}

inline std::uint64_t AcquireCudaFacadeIncarnation() noexcept
{
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1, std::memory_order_relaxed);
}

template <typename Selection>
inline bool ExactSelection(const CudaConfig& config) noexcept
{
  const RuntimeTupleDescriptor tuple = Selection::Descriptor();
  const bool rank_one = config.expected_rank_count == 1 &&
      ExactId(tuple.backend, "cuda.rank1.one_gpu");
  const bool rank_two = config.expected_rank_count == 2 &&
      ExactId(tuple.backend, "cuda.rank2.one_gpu.host_staged");
  const bool legacy_fixed = !config.use_borrowed_host_inputs &&
      ExactId(tuple.stabilization, "tau.fixed");
  const bool borrowed_current_llf = config.use_borrowed_host_inputs &&
      ExactId(tuple.stabilization,
              borrowed_euler_llf_stabilization_id);
  return Selection::interior_space == InteriorSpace::l2 && tuple.Valid() &&
      ExactId(tuple.model, "euler.compressible.2d") &&
      ExactId(tuple.formulation, "hdg.conventional") &&
      ExactId(tuple.trace, "mfem.dg_interface.trace") &&
      (legacy_fixed || borrowed_current_llf) &&
      ExactId(tuple.time_treatment, "backward_euler") &&
      ExactId(tuple.precision, "fp64") && (rank_one || rank_two);
}

} // namespace detail

#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_impl.cuh"

template <typename Selection,
          typename AllocatorPolicy = UmpireDeviceAllocatorPolicy,
          typename TransportPolicy = CudaTransportPolicy,
          typename CollectivePolicy = CudaMpiCollectivePolicy>
class PersistentCudaSmoothEulerEngine
{
public:
  using storage_policy = detail::CudaContractStoragePolicy<TransportPolicy>;
  using contract_type = PersistentRunContract<Selection,
                                              CudaInteriorState,
                                              CudaTrueTraceState,
                                              storage_policy,
                                              CollectivePolicy>;
  using state_slot = typename contract_type::state_slot;

  explicit PersistentCudaSmoothEulerEngine(const CudaConfig& config) noexcept
      : config_(config),
        communicator_(CudaCollectiveCommunicator(config)),
        facade_incarnation_(
            detail::AcquireCudaFacadeIncarnation())
  {
    static_assert(!std::is_polymorphic<Selection>::value,
                  "CUDA selection must remain static");
    static_assert(!std::is_polymorphic<AllocatorPolicy>::value,
                  "CUDA allocator policy must remain static");
    static_assert(!std::is_polymorphic<TransportPolicy>::value,
                  "CUDA transport policy must remain static");
    static_assert(!std::is_copy_constructible<
                      typename contract_type::StagingToken>::value &&
                      std::is_move_constructible<
                          typename contract_type::StagingToken>::value &&
                      !std::is_move_assignable<
                          typename contract_type::StagingToken>::value,
                  "staging authority must remain affine");
    static_assert(!std::is_copy_constructible<
                      typename contract_type::PreparedCommitToken>::value &&
                      std::is_move_constructible<
                          typename contract_type::PreparedCommitToken>::value &&
                      !std::is_move_assignable<
                          typename contract_type::PreparedCommitToken>::value,
                  "prepared-commit authority must remain affine");
    static_assert(!std::is_copy_constructible<
                      typename contract_type::ProvisionalCommitToken>::value &&
                      std::is_move_constructible<
                          typename contract_type::ProvisionalCommitToken>::value &&
                      !std::is_move_assignable<
                          typename contract_type::ProvisionalCommitToken>::value,
                  "provisional-commit authority must remain affine");
  }

  PersistentCudaSmoothEulerEngine(const PersistentCudaSmoothEulerEngine&) =
      delete;
  PersistentCudaSmoothEulerEngine& operator=(
      const PersistentCudaSmoothEulerEngine&) = delete;

  bool Initialize(CudaRunEvidence& evidence) noexcept
  {
    evidence = {};
    MPI_Comm_rank(communicator_, &evidence.rank);
    MPI_Comm_size(communicator_, &evidence.ranks);
    const bool descriptor_consensus = ConfigurationConsensus();
    const int local_config =
        ValidateConfig() && facade_incarnation_ != 0 ? 1 : 0;
    int global_config = 0;
    const bool validity_consensus = MPI_Allreduce(
        &local_config, &global_config, 1, MPI_INT, MPI_MIN,
        communicator_) == MPI_SUCCESS && global_config == 1;
    if (!descriptor_consensus || !validity_consensus)
    {
      ForgetBorrowedHostInputPointers();
      evidence.status = CudaRunStatus::setup_failure;
      evidence.setup_origin = descriptor_consensus
          ? FailureReason::backend_failure
          : FailureReason::collective_protocol;
      detail::ReportCudaSetupFailure(
          evidence.rank, descriptor_consensus
              ? "config-validity" : "descriptor-consensus");
      return false;
    }
    AllocatorPolicy::ResetLedger();
    const bool numerics_initialized = numerics_.Initialize(
        config_, evidence, Selection::Descriptor());
    // The numerical setup has copied every admitted host coefficient into
    // engine-owned arrays.  Retain only the borrowed topology thereafter.
    ForgetBorrowedHostInputPointers();
    if (!numerics_initialized)
    {
      evidence.status = CudaRunStatus::setup_failure;
      evidence.setup_origin = FailureReason::backend_failure;
      detail::ReportCudaSetupFailure(evidence.rank, "numerics-initialize");
      return false;
    }
    StateCoordinates initial{};
    initial.dt = 0.0;
    contract_.emplace(numerics_.CommittedSlot(), numerics_.CandidateSlot(),
                      initial, numerics_.Resources(), 0,
                      CollectivePolicy(communicator_));
    if (contract_->status() != RunStatus::ready)
    {
      evidence.status = CudaRunStatus::setup_failure;
      evidence.setup_origin = contract_->last_failure();
      detail::ReportCudaSetupFailure(evidence.rank, "contract-ready");
      return false;
    }
    if (config_.use_borrowed_host_inputs &&
        !numerics_.CommittedBorrowedEulerDiagnostics(
            numerics_.CommittedSlot(), config_.order,
            committed_maximum_wave_speed_, committed_backend_stable_dt_))
    {
      evidence.status = CudaRunStatus::setup_failure;
      evidence.setup_origin = FailureReason::backend_failure;
      detail::ReportCudaSetupFailure(evidence.rank, "initial-cfl-diagnostics");
      return false;
    }
    committed_diagnostics_valid_ = config_.use_borrowed_host_inputs;
    initialized_ = true;
    evidence.reuse = numerics_.ReuseEvidence();
    evidence.status = CudaRunStatus::ready;
    return true;
  }

  /** Complete numerical work, freeze the contract candidate, and retain the
   * affine staging token internally.  No rank may stage before the final
   * all-rank prestaging agreement in this call. */
  bool PrepareStep(const bool retry, const double dt,
                   CudaPrecommitSnapshot& snapshot,
                   CudaStepEvidence& evidence) noexcept
  {
    snapshot = {};
    if (pending_.active || !initialized_) { return false; }
    if (!PreparePending(retry, dt, evidence)) { return false; }
    if (!pending_.staging) { return false; }
    snapshot.attempt = pending_.attempt;
    snapshot.committed = pending_.staging->entry_coordinates();
    snapshot.target = pending_.staging->target_coordinates();
    snapshot.candidate_interior_hash = pending_.candidate_interior_hash;
    snapshot.candidate_trace_hash = pending_.candidate_trace_hash;
    snapshot.candidate_local_interior_hash =
        pending_.candidate_local_interior_hash;
    snapshot.candidate_local_trace_hash =
        pending_.candidate_local_trace_hash;
    snapshot.products = pending_.numerical.products;
    snapshot.interior_coefficients = numerics_.WorkspaceSlot().interior.count;
    snapshot.true_trace_coefficients = numerics_.WorkspaceSlot().trace.count;
    snapshot.completed_phase = pending_.numerical.phase;
    snapshot.numerical_candidate_ready = pending_.numerical.candidate_ready;
    return true;
  }

  /** Retained W8A entry point: false mode still advances at exactly .05. */
  bool PrepareStep(const bool retry, CudaPrecommitSnapshot& snapshot,
                   CudaStepEvidence& evidence) noexcept
  { return PrepareStep(retry, fixed_dt, snapshot, evidence); }

  /** Token-validated candidate copy followed by real D2H checkpoint staging. */
  bool StagePreparedCheckpoint(CudaSerializedCheckpointV1& checkpoint,
                               CudaStepEvidence& evidence) noexcept
  {
    checkpoint = {};
    if (!pending_.active || !pending_.staging ||
        !pending_.prestage_allowed)
      return false;
    checkpoint.attempt = pending_.attempt;
    checkpoint.facade_incarnation = facade_incarnation_;
    checkpoint.descriptor = pending_.staging->future_checkpoint();
    checkpoint.candidate_frozen = true;
    TransportPolicy::BeginStagingRange();
    storage_policy::SetCopyOutPurpose(
        detail::CudaCopyOutPurpose::checkpoint_staging);
    state_slot& staging_slot = numerics_.WorkspaceSlot();
    const bool token_copy = contract_->CopyPrepared(
        *pending_.staging, staging_slot);
    storage_policy::SetCopyOutPurpose(detail::CudaCopyOutPurpose::none);
    const bool local_staged = token_copy &&
        numerics_.StageCheckpoint(staging_slot, checkpoint);
    const bool all_staged = numerics_.CollectiveContinue(local_staged);
    pending_.checkpoint_staged = all_staged;
    if (!all_staged)
    {
      pending_.prestage_allowed = false;
      pending_.report.originating_failure = FailureReason::backend_failure;
      pending_.report.gates.success = false;
    }
    UpdateAttemptEvidence(pending_.evidence);
    checkpoint.staging_range = TransportPolicy::EndStagingRange();
    checkpoint.staging_range.complete =
        checkpoint.staging_range.complete && all_staged;
    evidence = pending_.evidence;
    return all_staged;
  }

  /**
   * Generic borrowed-mode staging copies the frozen slot device-to-device but
   * never materializes the legacy fixed checkpoint payload.  The caller can
   * export that slot through the explicit L2 bridge while this authority is
   * live.  Restart/checkpoint routes continue to use StagePreparedCheckpoint.
   */
  bool StagePreparedOutputMetadata(CudaPreparedOutputMetadataV1& metadata,
                                   CudaStepEvidence& evidence) noexcept
  {
    metadata = {};
    if (!pending_.active || !pending_.staging || !pending_.prestage_allowed)
      return false;
    metadata.attempt = pending_.attempt;
    metadata.facade_incarnation = facade_incarnation_;
    metadata.descriptor = pending_.staging->future_checkpoint();
    metadata.candidate_frozen = true;
    const CudaTransferTotals transfers_before = TransportPolicy::Accounting();
    TransportPolicy::BeginStagingRange();
    storage_policy::SetCopyOutPurpose(
        detail::CudaCopyOutPurpose::output_metadata_staging);
    state_slot& staging_slot = numerics_.WorkspaceSlot();
    const bool token_copy = contract_->CopyPrepared(*pending_.staging,
                                                     staging_slot);
    storage_policy::SetCopyOutPurpose(detail::CudaCopyOutPurpose::none);
    metadata.source_rank = numerics_.rank();
    metadata.rank_count = config_.expected_rank_count;
    metadata.local_mesh_hash = numerics_.LocalMeshHash();
    metadata.local_partition_hash = numerics_.LocalPartitionHash();
    metadata.local_topology_hash = numerics_.LocalTopologyHash();
    metadata.local_interior_hash = staging_slot.interior.local_hash;
    metadata.local_trace_hash = staging_slot.trace.local_hash;
    metadata.interior_count = staging_slot.interior.count;
    metadata.owned_true_trace_count = staging_slot.trace.count;
    metadata.interior_source_token = staging_slot.interior.storage_token;
    metadata.trace_source_token = staging_slot.trace.storage_token;
    metadata.staging_generation = staging_slot.interior.generation;
    const bool all_staged = numerics_.CollectiveContinue(token_copy);
    metadata.checkpoint_staged = all_staged;
    pending_.checkpoint_staged = all_staged;
    if (!all_staged)
    {
      pending_.prestage_allowed = false;
      pending_.report.originating_failure = FailureReason::backend_failure;
      pending_.report.gates.success = false;
    }
    UpdateAttemptEvidence(pending_.evidence);
    metadata.staging_range = TransportPolicy::EndStagingRange();
    const CudaTransferTotals transfers_after = TransportPolicy::Accounting();
    metadata.staging_host_to_device_bytes =
        transfers_after.host_to_device_bytes - transfers_before.host_to_device_bytes;
    metadata.staging_device_to_host_bytes =
        transfers_after.device_to_host_bytes - transfers_before.device_to_host_bytes;
    metadata.staging_device_to_device_bytes =
        transfers_after.device_to_device_bytes - transfers_before.device_to_device_bytes;
    metadata.staging_range.complete = metadata.staging_range.complete &&
        all_staged;
    evidence = pending_.evidence;
    return all_staged;
  }

  /** Freeze the independent oracle bytes during the same candidate epoch. */
  bool StagePreparedOracle(CudaOraclePod& pod,
                           CudaStepEvidence& evidence) noexcept
  {
    pod = {};
    if (!pending_.active || !pending_.staging ||
        !pending_.prestage_allowed || !pending_.checkpoint_staged)
      return false;
    const bool local_export = numerics_.ExportOracle(pod);
    const bool all_exported = numerics_.CollectiveContinue(local_export);
    pending_.oracle_staged = all_exported;
    if (all_exported)
    {
      pending_.oracle = pod;
      pending_.has_oracle = true;
    }
    else
    {
      pending_.prestage_allowed = false;
      pending_.report.originating_failure = FailureReason::backend_failure;
      pending_.report.gates.success = false;
    }
    UpdateAttemptEvidence(pending_.evidence);
    evidence = pending_.evidence;
    return all_exported;
  }

  /** A caller invokes this only after its output artifact is staged. */
  template <typename StagedOutput>
  bool MarkOutputStaged(StagedOutput& checkpoint) noexcept
  {
    const bool local_valid = pending_.active && pending_.staging &&
        pending_.prestage_allowed && pending_.checkpoint_staged &&
        checkpoint.candidate_frozen && checkpoint.checkpoint_staged &&
        !checkpoint.output_staged && !checkpoint.successful_commit &&
        checkpoint.facade_incarnation == facade_incarnation_ &&
        checkpoint.attempt.sequence == pending_.attempt.sequence &&
        checkpoint.attempt.committed_step == pending_.attempt.committed_step &&
        checkpoint.attempt.committed_generation ==
            pending_.attempt.committed_generation &&
        checkpoint.attempt.retry_index == pending_.attempt.retry_index &&
        std::memcmp(&checkpoint.attempt.dt, &pending_.attempt.dt,
                    sizeof(double)) == 0 &&
        dg::hdg::runtime::detail::SameCheckpoint(
            checkpoint.descriptor, pending_.staging->future_checkpoint());
    if (!numerics_.CollectiveContinue(local_valid))
      return false;
    checkpoint.output_staged = true;
    pending_.output_staged = true;
    return true;
  }

  /** Consume StagingToken and retain PreparedCommitToken internally. */
  template <typename StagedOutput>
  bool PrepareCommit(StagedOutput& checkpoint,
                     CudaStepEvidence& evidence) noexcept
  {
    const bool local_staging = pending_.active && pending_.staging.has_value();
    if (!numerics_.CollectiveContinue(local_staging))
      return RetainAffineProtocolFailure(evidence);
    const bool success_path = pending_.prestage_allowed &&
        pending_.checkpoint_staged && pending_.output_staged &&
        (!config_.capture_oracle || pending_.oracle_staged);
    if (!success_path &&
        pending_.report.originating_failure == FailureReason::none)
    {
      pending_.report.originating_failure = FailureReason::backend_failure;
      pending_.report.gates.success = false;
    }
    const CudaTransferSite staging_site = std::is_same<
        StagedOutput, CudaPreparedOutputMetadataV1>::value
        ? CudaTransferSite::output_metadata_stage_trace
        : CudaTransferSite::checkpoint_stage_trace;
    const bool copies_complete = TransportPolicy::Synchronize(staging_site);
    if (!numerics_.CollectiveContinue(copies_complete))
    {
      pending_.report.originating_failure = FailureReason::backend_failure;
      pending_.report.gates.success = false;
    }
    UpdateAttemptEvidence(pending_.evidence);
    UpdateSharedAccounting();
    auto prepared = contract_->PrepareCommit(
        std::move(*pending_.staging), pending_.report,
        std::move(pending_.bundle));
    pending_.staging.reset();
    if (!prepared)
    {
      const AttemptOutcome outcome{
          contract_->status(), contract_->last_failure(), false,
          contract_->status() == RunStatus::retry_available};
      FinishOutcome(outcome, pending_.evidence);
      evidence = pending_.evidence;
      AbortStagedArtifacts(checkpoint);
      ClearConsumedPending();
      return outcome.retry_available;
    }
    pending_.prepared_checkpoint = prepared->checkpoint_descriptor();
    pending_.prepared.emplace(std::move(*prepared));
    evidence = pending_.evidence;
    return true;
  }

  /** Reversible whole-slot exchange; old logical publication remains visible. */
  template <typename StagedOutput>
  bool CommitPrepared(StagedOutput& checkpoint,
                      CudaStepEvidence& evidence) noexcept
  {
    const bool local_prepared = pending_.active && pending_.prepared.has_value();
    if (!numerics_.CollectiveContinue(local_prepared))
      return RetainAffineProtocolFailure(evidence);
    const bool branch_agreed = numerics_.CollectiveContinue(
        pending_.checkpoint_staged && pending_.output_staged);
    if (!branch_agreed)
      return AbortPrepared(checkpoint, FailureReason::collective_protocol,
                           evidence);
    auto provisional = contract_->CommitPrepared(std::move(*pending_.prepared));
    pending_.prepared.reset();
    if (!provisional) { return false; }
    pending_.provisional.emplace(std::move(*provisional));
    evidence = pending_.evidence;
    return true;
  }

  /** Sole logical publication boundary; validation failure remains reversible. */
  template <typename StagedOutput>
  bool FinalizeCommit(StagedOutput& checkpoint,
                      CudaStepEvidence& evidence) noexcept
  {
    const bool local_provisional =
        pending_.active && pending_.provisional.has_value();
    if (!numerics_.CollectiveContinue(local_provisional))
      return RetainAffineProtocolFailure(evidence);
    const bool local_artifacts_publishable = checkpoint.checkpoint_staged &&
        checkpoint.output_staged &&
        checkpoint.facade_incarnation == facade_incarnation_ &&
        checkpoint.attempt.sequence == pending_.attempt.sequence &&
        dg::hdg::runtime::detail::SameCheckpoint(
            checkpoint.descriptor, pending_.prepared_checkpoint);
    if (!numerics_.CollectiveContinue(local_artifacts_publishable))
    {
      return RevertCommit(checkpoint, FailureReason::collective_protocol,
                          evidence);
    }
    if (!local_artifacts_publishable) return false;
    // Compute trace-inclusive diagnostics from candidate slot 1 while the
    // provisional authority remains reversible.  A failed scan reverts the
    // provisional publication instead of rejecting after bytes are committed.
    double candidate_maximum_wave_speed = 0.0;
    double candidate_backend_stable_dt = std::numeric_limits<double>::quiet_NaN();
    if (config_.use_borrowed_host_inputs &&
        !numerics_.CommittedBorrowedEulerDiagnostics(
            numerics_.CandidateSlot(), config_.order,
            candidate_maximum_wave_speed, candidate_backend_stable_dt))
      return RevertCommit(checkpoint, FailureReason::backend_failure, evidence);
    const AttemptOutcome outcome =
        contract_->FinalizeCommit(std::move(*pending_.provisional));
    pending_.provisional.reset();
    if (outcome.committed)
    {
      numerics_.SwapPublishedSlotHandles();
      if (config_.use_borrowed_host_inputs)
      {
        committed_maximum_wave_speed_ = candidate_maximum_wave_speed;
        committed_backend_stable_dt_ = candidate_backend_stable_dt;
        committed_diagnostics_valid_ = true;
      }
    }
    checkpoint.successful_commit = outcome.committed;
    FinishOutcome(outcome, pending_.evidence);
    if (pending_.has_oracle)
    {
      last_oracle_ = pending_.oracle;
      last_oracle_valid_ = true;
    }
    evidence = pending_.evidence;
    ClearConsumedPending();
    return outcome.committed;
  }

  /** FailureReason::none is normalized to backend_failure and cannot commit. */
  template <typename StagedOutput>
  bool AbortPrepared(StagedOutput& checkpoint,
                     FailureReason reason,
                     CudaStepEvidence& evidence) noexcept
  {
    if (reason == FailureReason::none) { reason = FailureReason::backend_failure; }
    const int local_phase = !pending_.active ? 0 :
        (pending_.staging ? 1 : (pending_.prepared ? 2 : 0));
    int agreed_phase = 0;
    FailureReason agreed_reason = FailureReason::collective_protocol;
    if (!CollectiveAffineDecision(local_phase, reason, agreed_phase,
                                  agreed_reason))
      return RetainAffineProtocolFailure(evidence);
    reason = agreed_reason;
    pending_.prestage_allowed = false;
    pending_.report.originating_failure = reason;
    pending_.report.gates.success = false;
    if (agreed_phase == 1) { return PrepareCommit(checkpoint, evidence); }
    if (agreed_phase != 2) { return RetainAffineProtocolFailure(evidence); }
    const AttemptOutcome outcome =
        contract_->AbortPrepared(std::move(*pending_.prepared), reason);
    pending_.prepared.reset();
    FinishOutcome(outcome, pending_.evidence);
    evidence = pending_.evidence;
    AbortStagedArtifacts(checkpoint);
    ClearConsumedPending();
    return outcome.retry_available;
  }

  template <typename StagedOutput>
  bool RevertCommit(StagedOutput& checkpoint,
                    FailureReason reason,
                    CudaStepEvidence& evidence) noexcept
  {
    if (reason == FailureReason::none) { reason = FailureReason::backend_failure; }
    const int local_phase = pending_.active && pending_.provisional ? 3 : 0;
    int agreed_phase = 0;
    FailureReason agreed_reason = FailureReason::collective_protocol;
    if (!CollectiveAffineDecision(local_phase, reason, agreed_phase,
                                  agreed_reason) || agreed_phase != 3)
      return RetainAffineProtocolFailure(evidence);
    reason = agreed_reason;
    const AttemptOutcome outcome =
        contract_->RevertCommit(std::move(*pending_.provisional), reason);
    pending_.provisional.reset();
    AbortStagedArtifacts(checkpoint);
    FinishOutcome(outcome, pending_.evidence);
    evidence = pending_.evidence;
    ClearConsumedPending();
    return outcome.retry_available;
  }

  /** Deserialize rank-local bytes, recompute device hashes, then restore. */
  bool Restore(const CudaSerializedCheckpointV1& checkpoint,
               CudaStepEvidence& evidence) noexcept
  {
    evidence = {};
    if (!initialized_ || pending_.active) { return false; }
    evidence.before = contract_->coordinates();
    const StateCoordinates restored_coordinates =
        checkpoint.descriptor.coordinates;
    const std::uint64_t restored_frontier = restored_coordinates.cursor;
    const bool local_frontier_valid =
        contract_->status() == RunStatus::ready &&
        evidence.before.step == 0 && evidence.before.generation == 0 &&
        evidence.before.cursor == 0 && evidence.before.time == 0.0 &&
        checkpoint.attempt.sequence == restored_frontier &&
        restored_frontier != 0 &&
        restored_frontier != std::numeric_limits<std::uint64_t>::max() &&
        restored_coordinates.step != 0 &&
        restored_coordinates.generation != 0 &&
        checkpoint.attempt.committed_step == restored_coordinates.step - 1 &&
        checkpoint.attempt.committed_generation ==
            restored_coordinates.generation - 1;
    if (!numerics_.CollectiveContinue(local_frontier_valid))
    {
      evidence.status = CudaRunStatus::runtime_failure;
      evidence.contract_reason = FailureReason::restart_incompatible;
      return false;
    }
    CudaPolicyContext context{};
    context.attempt_sequence = restored_frontier;
    context.target_step = checkpoint.descriptor.coordinates.step;
    context.generation = checkpoint.descriptor.coordinates.generation;
    context.rank = numerics_.rank();
    context.phase = CudaFailurePhase::setup;
    evidence.transfer_range.begin_ordinal =
        TransportPolicy::Ledger().observed_count + 1;
    TransportPolicy::BeginAttempt(context);
    state_slot& loaded = numerics_.WorkspaceSlot();
    const bool imported = numerics_.ImportCheckpoint(checkpoint, loaded);
    if (!imported)
    {
      evidence.status = CudaRunStatus::runtime_failure;
      evidence.contract_reason = FailureReason::restart_incompatible;
      UpdateAttemptEvidence(evidence);
      return false;
    }
    // The accepted public constructor carries the attempt frontier while the
    // c879 Restore seam restores coordinates only. Recreate the still-pristine
    // contract before publication so the next sequence/candidate cursor is
    // strictly newer and every capability from the pre-restore incarnation is
    // stale. A later shared Restore frontier implementation is compatible with
    // this identical seed and can replace this owned bridge.
    contract_.reset();
    StateCoordinates initial{};
    contract_.emplace(numerics_.CommittedSlot(), numerics_.CandidateSlot(),
                      initial, numerics_.Resources(), restored_frontier,
                      CollectivePolicy(communicator_));
    const RestoreOutcome restored =
        contract_->Restore(checkpoint.descriptor, loaded);
    if (!restored.restored)
    {
      contract_.reset();
      contract_.emplace(numerics_.CommittedSlot(), numerics_.CandidateSlot(),
                        initial, numerics_.Resources(), 0,
                        CollectivePolicy(communicator_));
      next_attempt_sequence_ = 0;
    }
    else
    {
      numerics_.SwapPublishedSlotHandles();
      next_attempt_sequence_ = restored_frontier;
    }
    evidence.after = contract_->coordinates();
    evidence.after_interior_hash = loaded.interior.hash;
    evidence.after_trace_hash = loaded.trace.hash;
    evidence.committed = restored.restored;
    evidence.collective_consensus = restored.compatibility !=
        RestartCompatibility::collective_protocol;
    evidence.contract_reason = restored.restored ? FailureReason::none :
        FailureReason::restart_incompatible;
    evidence.status = restored.restored ? CudaRunStatus::committed :
        CudaRunStatus::runtime_failure;
    UpdateAttemptEvidence(evidence);
    return restored.restored;
  }

  bool ExportOracle(CudaOraclePod& pod) const noexcept
  {
    if (!last_oracle_valid_) { return false; }
    pod = last_oracle_;
    return true;
  }

  /** Explicitly map the currently committed interior into caller L2 storage. */
  bool ExportCommittedInterior(CudaMutableHostL2Output destination) noexcept
  {
    if (!initialized_ || pending_.active) { return false; }
    return numerics_.ExportInterior(numerics_.CommittedSlot().interior,
                                    destination);
  }

  bool CachedBorrowedEulerDiagnostics(double& maximum_wave_speed,
                                      double& backend_stable_dt) const noexcept
  {
    maximum_wave_speed = committed_maximum_wave_speed_;
    backend_stable_dt = committed_backend_stable_dt_;
    return initialized_ && committed_diagnostics_valid_;
  }

  /** Export only the currently staged candidate; no logical publication occurs. */
  bool ExportPreparedInterior(CudaMutableHostL2Output destination) noexcept
  {
    if (!initialized_ || !pending_.active || !pending_.staging ||
        !pending_.prestage_allowed || !pending_.checkpoint_staged)
      return false;
    return numerics_.ExportInterior(numerics_.WorkspaceSlot().interior,
                                    destination);
  }

  CudaReuseEvidence ReuseEvidence() const noexcept
  { return numerics_.ReuseEvidence(); }

  bool LocalCheckpointIdentity(CudaSerializedCheckpointV1& identity) const
      noexcept
  {
    identity = {};
    if (!initialized_ || pending_.active) { return false; }
    identity.source_rank = numerics_.rank();
    identity.rank_count = config_.expected_rank_count;
    identity.local_mesh_hash = numerics_.LocalMeshHash();
    identity.local_partition_hash = numerics_.LocalPartitionHash();
    identity.local_topology_hash = numerics_.LocalTopologyHash();
    return true;
  }

  const contract_type* contract() const noexcept
  {
    return contract_ ? &*contract_ : nullptr;
  }

private:
  friend struct detail::CudaQualificationTestAccess;

  void ForgetBorrowedHostInputPointers() noexcept
  {
    config_.borrowed_initial.coefficients = nullptr;
    config_.borrowed_initial_true_trace.coefficients = nullptr;
    config_.borrowed_source.coefficients = nullptr;
    config_.borrowed_boundary_policies.policies = nullptr;
    config_.borrowed_boundary_policies.count = 0;
    config_.borrowed_boundary = {};
    config_.borrowed_global_topology = {};
  }

  template <typename StagedOutput>
  static void AbortStagedArtifacts(StagedOutput& checkpoint) noexcept
  {
    checkpoint.candidate_frozen = false;
    checkpoint.checkpoint_staged = false;
    checkpoint.output_staged = false;
    checkpoint.successful_commit = false;
  }

  bool CollectiveAffineDecision(const int local_phase,
                                FailureReason local_reason,
                                int& agreed_phase,
                                FailureReason& agreed_reason) const noexcept
  {
    const auto reason_value = static_cast<unsigned>(local_reason);
    if (reason_value == static_cast<unsigned>(FailureReason::none) ||
        reason_value >= static_cast<unsigned>(FailureReason::count))
      local_reason = FailureReason::collective_protocol;
    const int local[2] = {local_phase, static_cast<int>(local_reason)};
    int minimum[2] = {local[0], local[1]};
    int maximum[2] = {local[0], local[1]};
    const bool minimum_ready = MPI_Allreduce(
        local, minimum, 2, MPI_INT, MPI_MIN, communicator_) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local, maximum, 2, MPI_INT, MPI_MAX, communicator_) == MPI_SUCCESS;
    int ranks = 0;
    const bool size_ready =
        MPI_Comm_size(communicator_, &ranks) == MPI_SUCCESS;
    const bool agreed = minimum_ready && maximum_ready && size_ready &&
        ranks == config_.expected_rank_count && minimum[0] == maximum[0] &&
        minimum[1] == maximum[1] && minimum[0] != 0;
    agreed_phase = agreed ? minimum[0] : 0;
    agreed_reason = agreed ? static_cast<FailureReason>(minimum[1]) :
        FailureReason::collective_protocol;
    return agreed;
  }

  bool RetainAffineProtocolFailure(CudaStepEvidence& evidence) noexcept
  {
    if (pending_.active)
    {
      pending_.evidence.status = CudaRunStatus::terminal_failure;
      pending_.evidence.contract_reason = FailureReason::collective_protocol;
      pending_.evidence.collective_consensus = false;
      UpdateAttemptEvidence(pending_.evidence);
      evidence = pending_.evidence;
    }
    else
    {
      evidence.status = CudaRunStatus::terminal_failure;
      evidence.contract_reason = FailureReason::collective_protocol;
      evidence.collective_consensus = false;
    }
    // No frozen/prepared/provisional token is dropped or consumed here.  The
    // facade becomes unusable until destroyed because c879 has no mixed-phase
    // collective abort authority.
    initialized_ = false;
    return false;
  }

  static TransferAccounting ProjectSharedAccounting(
      const CudaTransferTotals& totals,
      const std::uint64_t allocation_events = 0) noexcept
  {
    // The frozen shared ABI has no D2D/event-count fields.  This is the exact
    // representable projection only; the CUDA ledger/totals remain
    // authoritative for D2D bytes and every event identity.
    return {totals.host_to_device_bytes,
            totals.device_to_host_bytes,
            totals.mpi_staging_bytes,
            allocation_events,
            totals.synchronization_events,
            totals.complete};
  }

  void UpdateSharedAccounting() noexcept
  {
    const std::uint64_t allocations =
        AllocatorPolicy::W8OwnedAllocationCalls() -
        pending_.w8_owned_allocation_begin;
    pending_.report.telemetry.transfers = ProjectSharedAccounting(
        TransportPolicy::Accounting(), allocations);
  }

  void UpdateAttemptEvidence(CudaStepEvidence& evidence) const noexcept
  {
    evidence.transfers = TransportPolicy::Accounting();
    evidence.transfer_range = TransportPolicy::AttemptRange();
    evidence.transfer_range.complete =
        evidence.transfer_range.complete && evidence.transfers.complete;
  }

  void FinishOutcome(const AttemptOutcome& outcome,
                     CudaStepEvidence& evidence) noexcept
  {
    evidence.contract_reason = outcome.reason;
    evidence.after = contract_->coordinates();
    evidence.committed = outcome.committed;
    evidence.retry_available = outcome.retry_available;
    evidence.collective_consensus =
        outcome.reason != FailureReason::collective_protocol;
    evidence.after_interior_hash = outcome.committed
        ? pending_.candidate_interior_hash : evidence.before_interior_hash;
    evidence.after_trace_hash = outcome.committed
        ? pending_.candidate_trace_hash : evidence.before_trace_hash;
    evidence.committed_bytes_preserved = outcome.committed ||
        (evidence.before_interior_hash == evidence.after_interior_hash &&
         evidence.before_trace_hash == evidence.after_trace_hash);
    evidence.coordinates_preserved = outcome.committed ||
        (evidence.before.time == evidence.after.time &&
         evidence.before.step == evidence.after.step &&
         evidence.before.generation == evidence.after.generation &&
         evidence.before.cursor == evidence.after.cursor);
    evidence.status = outcome.committed ? CudaRunStatus::committed :
        (outcome.retry_available ? CudaRunStatus::retry_available :
         (pending_.numerical.injected ? CudaRunStatus::injected_failure :
                                       CudaRunStatus::terminal_failure));
    UpdateAttemptEvidence(evidence);
  }

  void ClearConsumedPending() noexcept
  {
    // Call only after the live affine token has been consumed.
    pending_.staging.reset();
    pending_.prepared.reset();
    pending_.provisional.reset();
    pending_.active = false;
    pending_.prestage_allowed = false;
    pending_.checkpoint_staged = false;
    pending_.oracle_staged = false;
    pending_.output_staged = false;
    pending_.has_oracle = false;
  }

  bool ValidateConfig() const noexcept
  {
    const bool borrowed_generic_shape = config_.use_borrowed_topology &&
        config_.use_borrowed_host_inputs && config_.mesh_n >= 0 &&
        config_.order >= generic::minimum_polynomial_order &&
        config_.order <= generic::maximum_polynomial_order;
    const bool smoke_shape = config_.order == smoke_polynomial_order &&
        config_.mesh_n == smoke_elements_per_direction;
    const bool functional_shape =
        config_.order == functional_polynomial_order &&
        config_.mesh_n == functional_elements_per_direction;
    const bool numerical_injection_phase =
        config_.injection.phase == CudaFailurePhase::assembly ||
        config_.injection.phase == CudaFailurePhase::condensation ||
        config_.injection.phase == CudaFailurePhase::trace_solve ||
        config_.injection.phase == CudaFailurePhase::recovery ||
        config_.injection.phase == CudaFailurePhase::diagnostics ||
        config_.injection.phase ==
            CudaFailurePhase::after_recovery_before_commit;
    const bool topology_configuration = config_.use_borrowed_topology
        ? config_.borrowed_mesh != nullptr &&
            config_.borrowed_communicator != MPI_COMM_NULL
        : config_.borrowed_mesh == nullptr &&
            config_.borrowed_communicator == MPI_COMM_NULL;
    const auto empty_l2 = [](const CudaBorrowedHostL2Input& field)
    {
      return field.coefficients == nullptr && field.scalar_count == 0 &&
          field.components == conserved_components;
    };
    const bool generic_l2_configuration = config_.use_borrowed_host_inputs
        ? config_.borrowed_initial.coefficients != nullptr &&
            config_.borrowed_initial.scalar_count != 0 &&
            config_.borrowed_initial.components == conserved_components &&
            ((config_.borrowed_source.coefficients == nullptr &&
              config_.borrowed_source.scalar_count == 0 &&
              config_.borrowed_source.components == conserved_components) ||
             (config_.borrowed_source.coefficients != nullptr &&
              config_.borrowed_source.scalar_count != 0 &&
              config_.borrowed_source.components == conserved_components))
        : empty_l2(config_.borrowed_initial) &&
            empty_l2(config_.borrowed_initial_true_trace) &&
            empty_l2(config_.borrowed_source);
    const bool initial_trace_configuration = config_.use_borrowed_host_inputs
        ? empty_l2(config_.borrowed_initial_true_trace) ||
            (config_.borrowed_initial_true_trace.coefficients != nullptr &&
             config_.borrowed_initial_true_trace.scalar_count != 0 &&
             config_.borrowed_initial_true_trace.components ==
                 conserved_components)
        : empty_l2(config_.borrowed_initial_true_trace);
    const bool generic_numerical_configuration =
        config_.use_generic_numerical_policy
        ? config_.use_borrowed_host_inputs &&
            generic::ValidEulerHdgNumericalPolicy(config_.numerical_policy)
        : !config_.use_borrowed_host_inputs ||
            config_.numerical_policy.stabilization.kind ==
                generic::RegularizationKind::none;
    const bool boundary_policy_configuration =
        config_.use_borrowed_host_inputs
        ? ValidateCudaStaticBoundaryPolicyTable(
              config_.borrowed_boundary_policies)
        : config_.borrowed_boundary_policies.policies == nullptr &&
            config_.borrowed_boundary_policies.count == 0;
    const bool borrowed_topology_identity = config_.use_borrowed_host_inputs
        ? (config_.borrowed_boundary_policies.count == 0
            ? IsCudaAbsentBorrowedGlobalTopologyIdentity(
                config_.borrowed_global_topology)
            : IsCudaBorrowedGlobalTopologyIdentity(
                config_.borrowed_global_topology))
        : IsCudaAbsentBorrowedGlobalTopologyIdentity(
            config_.borrowed_global_topology);
    const bool fixed_oracle_requested_for_generic_storage =
        config_.use_borrowed_topology && config_.use_borrowed_host_inputs &&
        config_.capture_oracle;
    return config_.experiment == StableIdentity::Copy(experiment_id) &&
        (borrowed_generic_shape || smoke_shape || functional_shape) &&
        topology_configuration &&
        generic_l2_configuration &&
        initial_trace_configuration &&
        generic_numerical_configuration &&
        (!config_.use_generic_numerical_policy || borrowed_generic_shape) &&
        boundary_policy_configuration &&
        borrowed_topology_identity &&
        !fixed_oracle_requested_for_generic_storage &&
        std::isfinite(config_.dt) && config_.dt > 0.0 &&
        (config_.use_borrowed_host_inputs ||
         std::memcmp(&config_.dt, &fixed_dt, sizeof(double)) == 0) &&
        config_.requested_device == 0 &&
        (config_.expected_rank_count == 1 ||
         config_.expected_rank_count == 2) &&
        (config_.expected_rank_count == 1 ||
         config_.require_host_staged_rank2) &&
        (!config_.injection.enabled ||
         (numerical_injection_phase &&
          config_.injection.target_step != 0 &&
          config_.injection.source_rank >= 0 &&
          config_.injection.source_rank < config_.expected_rank_count)) &&
        detail::ExactSelection<Selection>(config_);
  }

  using ConfigDescriptor = std::array<unsigned char, 1024>;
  using PolicyDescriptor = std::array<unsigned char, 2048>;
  using TopologyDescriptor = std::array<unsigned char, 128>;

  template <typename Value>
  static void AppendConfigValue(ConfigDescriptor& descriptor,
                                std::size_t& cursor, bool& complete,
                                const Value& value) noexcept
  {
    static_assert(std::is_trivially_copyable<Value>::value,
                  "configuration descriptor values must be bitwise stable");
    if (!complete || sizeof(Value) > descriptor.size() - cursor)
    {
      complete = false;
      return;
    }
    std::memcpy(descriptor.data() + cursor, &value, sizeof(Value));
    cursor += sizeof(Value);
  }

  static void AppendConfigId(ConfigDescriptor& descriptor,
                             std::size_t& cursor, bool& complete,
                             const StableIdentity& id) noexcept
  {
    const std::uint64_t size = static_cast<std::uint64_t>(id.size());
    const std::uint8_t valid = id.valid() ? 1U : 0U;
    AppendConfigValue(descriptor, cursor, complete, size);
    AppendConfigValue(descriptor, cursor, complete, valid);
    if (!complete || id.size() > descriptor.size() - cursor)
    {
      complete = false;
      return;
    }
    std::memcpy(descriptor.data() + cursor, id.data(), id.size());
    cursor += id.size();
  }

  bool MakeConfigDescriptor(ConfigDescriptor& descriptor) const noexcept
  {
    // CudaOraclePod is a frozen fixed-capacity ABI.  Borrowed host sessions
    // may use dynamic storage, so their output route is metadata-only and may
    // not request fixed oracle materialization.  False mode writes the same
    // descriptor bytes as before.
    if (config_.use_borrowed_topology && config_.use_borrowed_host_inputs &&
        config_.capture_oracle)
      return false;
    descriptor = {};
    std::size_t cursor = 0;
    bool complete = true;
    const auto append_bool = [&](const bool value)
    {
      const std::uint8_t bit = value ? 1U : 0U;
      AppendConfigValue(descriptor, cursor, complete, bit);
    };
    AppendConfigId(descriptor, cursor, complete, config_.experiment);
    AppendConfigValue(descriptor, cursor, complete, config_.order);
    AppendConfigValue(descriptor, cursor, complete, config_.mesh_n);
    AppendConfigValue(descriptor, cursor, complete, config_.dt);
    AppendConfigValue(descriptor, cursor, complete, config_.requested_device);
    AppendConfigValue(descriptor, cursor, complete,
                      config_.expected_rank_count);
    append_bool(config_.require_host_staged_rank2);
    // Pointer and communicator identities are rank-local borrowed handles;
    // their compatibility is proved by the pre-device topology admission.
    append_bool(config_.use_borrowed_topology);
    // Keep all legacy false-mode descriptor bytes exactly as they were.  L2
    // scalar counts are partition-local, so true mode records only common ABI
    // facts and lets the collective topology/input admission check local size.
    if (config_.use_borrowed_host_inputs)
    {
      append_bool(true);
      AppendConfigValue(descriptor, cursor, complete,
                        config_.borrowed_initial.components);
      AppendConfigValue(descriptor, cursor, complete,
                        config_.borrowed_initial_true_trace.components);
      append_bool(config_.borrowed_initial_true_trace.coefficients != nullptr);
      AppendConfigValue(descriptor, cursor, complete,
                        config_.borrowed_source.components);
      append_bool(config_.borrowed_source.coefficients != nullptr);
      append_bool(config_.use_generic_numerical_policy);
      if (config_.use_generic_numerical_policy)
      {
        const auto& policy = config_.numerical_policy;
        AppendConfigValue(descriptor, cursor, complete,
                          policy.volume_projection_quadrature_order);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.true_trace_projection_quadrature_order);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.conventional_assembly_quadrature_order);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.igr_assembly_quadrature_order);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.maximum_nonlinear_updates);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.maximum_linear_iterations);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.maximum_line_search_trials);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.nonlinear_relative_tolerance);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.linear_relative_tolerance);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.true_residual_relative_tolerance);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.conservation_relative_tolerance);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.armijo_constant);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.stabilization.kind);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.stabilization.alpha);
        AppendConfigValue(descriptor, cursor, complete,
                          policy.stabilization.beta);
      }
    }
    append_bool(config_.capture_oracle);
    AppendConfigValue(descriptor, cursor, complete, config_.injection.phase);
    AppendConfigValue(descriptor, cursor, complete,
                      config_.injection.target_step);
    AppendConfigValue(descriptor, cursor, complete,
                      config_.injection.source_rank);
    append_bool(config_.injection.enabled);
    AppendConfigValue(descriptor, cursor, complete,
                      config_.injection.token_fault);
    const RuntimeTupleDescriptor tuple = Selection::Descriptor();
    AppendConfigId(descriptor, cursor, complete, tuple.model);
    AppendConfigId(descriptor, cursor, complete, tuple.formulation);
    AppendConfigId(descriptor, cursor, complete, tuple.trace);
    AppendConfigId(descriptor, cursor, complete, tuple.stabilization);
    AppendConfigId(descriptor, cursor, complete, tuple.time_treatment);
    AppendConfigId(descriptor, cursor, complete, tuple.backend);
    AppendConfigId(descriptor, cursor, complete, tuple.precision);
    AppendConfigValue(descriptor, cursor, complete,
                      Selection::interior_space);
    return complete;
  }

  bool ConfigurationConsensus() const noexcept
  {
    bool borrowed_host_mode = false;
    // This collective is intentionally before MakeConfigDescriptor: that
    // descriptor has a borrowed-host-only branch and must never be entered
    // asymmetrically.
    if (!CudaBorrowedHostModeConsensus(communicator_,
                                       config_.use_borrowed_host_inputs,
                                       borrowed_host_mode))
      return false;
    ConfigDescriptor local{};
    ConfigDescriptor minimum{};
    ConfigDescriptor maximum{};
    const bool descriptor_complete = MakeConfigDescriptor(local);
    minimum = local;
    maximum = local;
    int ranks = 0;
    const bool size_ready =
        MPI_Comm_size(communicator_, &ranks) == MPI_SUCCESS;
    const bool minimum_ready = MPI_Allreduce(
        local.data(), minimum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MIN, communicator_) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local.data(), maximum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MAX, communicator_) == MPI_SUCCESS;
    const bool base_consensus = descriptor_complete && size_ready && minimum_ready &&
        maximum_ready && ranks == config_.expected_rank_count &&
        minimum == maximum;
    if (!base_consensus || !borrowed_host_mode) return base_consensus;
    // True borrowed mode always proves the exact policy table.  The topology
    // identity is a physical-boundary-only contract so legacy periodic
    // borrowed callers retain their empty config spelling.
    if (!PolicyTableConsensus()) return false;
    return config_.borrowed_boundary_policies.count == 0 ||
        TopologyIdentityConsensus();
  }

  bool TopologyIdentityConsensus() const noexcept
  {
    TopologyDescriptor local{}, minimum{}, maximum{};
    const StableIdentity& identity = config_.borrowed_global_topology;
    const std::uint64_t length = identity.size();
    const std::uint8_t valid = identity.valid() ? 1U : 0U;
    bool complete = IsCudaBorrowedGlobalTopologyIdentity(identity) &&
        length <= local.size() - sizeof(length) - sizeof(valid);
    std::size_t cursor = 0;
    if (complete)
    {
      std::memcpy(local.data() + cursor, &length, sizeof(length));
      cursor += sizeof(length);
      std::memcpy(local.data() + cursor, &valid, sizeof(valid));
      cursor += sizeof(valid);
      std::memcpy(local.data() + cursor, identity.data(),
                  static_cast<std::size_t>(length));
    }
    minimum = local;
    maximum = local;
    const int complete_bit = complete ? 1 : 0;
    int all_complete = 0;
    const bool complete_ready = MPI_Allreduce(
        &complete_bit, &all_complete, 1, MPI_INT, MPI_MIN,
        communicator_) == MPI_SUCCESS;
    const bool minimum_ready = MPI_Allreduce(
        local.data(), minimum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MIN, communicator_) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local.data(), maximum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MAX, communicator_) == MPI_SUCCESS;
    return complete_ready && minimum_ready && maximum_ready &&
        all_complete == 1 && minimum == maximum;
  }

  bool PolicyTableConsensus() const noexcept
  {
    PolicyDescriptor local{}, minimum{}, maximum{};
    std::size_t cursor = 0;
    bool complete = ValidateCudaStaticBoundaryPolicyTable(
        config_.borrowed_boundary_policies);
    const auto append = [&](const auto& value)
    {
      using value_type = std::decay_t<decltype(value)>;
      static_assert(std::is_trivially_copyable<value_type>::value,
                    "policy descriptor values must be bitwise stable");
      if (!complete || sizeof(value_type) > local.size() - cursor)
      { complete = false; return; }
      std::memcpy(local.data() + cursor, &value, sizeof(value_type));
      cursor += sizeof(value_type);
    };
    append(config_.borrowed_boundary_policies.count);
    append(config_.borrowed_boundary_policies.gamma);
    const bool readable = config_.borrowed_boundary_policies.policies != nullptr &&
        config_.borrowed_boundary_policies.count <= maximum_static_boundary_policies;
    for (std::size_t index = 0; index < maximum_static_boundary_policies; ++index)
    {
      const CudaStaticBoundaryPolicy empty{};
      const CudaStaticBoundaryPolicy& policy = readable && index <
          config_.borrowed_boundary_policies.count
          ? config_.borrowed_boundary_policies.policies[index] : empty;
      append(policy.attribute);
      const std::uint8_t kind = static_cast<std::uint8_t>(policy.kind);
      append(kind);
      for (const double reference : policy.reference_state) append(reference);
      append(policy.pressure_out);
    }
    minimum = local;
    maximum = local;
    const int complete_bit = complete ? 1 : 0;
    int all_complete = 0;
    const bool complete_ready = MPI_Allreduce(
        &complete_bit, &all_complete, 1, MPI_INT, MPI_MIN,
        communicator_) == MPI_SUCCESS;
    const bool minimum_ready = MPI_Allreduce(
        local.data(), minimum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MIN, communicator_) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local.data(), maximum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MAX, communicator_) == MPI_SUCCESS;
    return complete_ready && minimum_ready && maximum_ready &&
        all_complete == 1 && minimum == maximum;
  }

  bool PreparePending(const bool retry, const double dt,
                      CudaStepEvidence& evidence) noexcept
  {
    evidence = {};
    evidence.before = contract_->coordinates();
    CudaPolicyContext context{};
    context.attempt_sequence = ++next_attempt_sequence_;
    context.target_step = evidence.before.step + 1;
    context.generation = evidence.before.generation + 1;
    context.rank = numerics_.rank();
    TransportPolicy::BeginAttempt(context);
    pending_.ledger_begin = TransportPolicy::Ledger().observed_count + 1;
    pending_.w8_owned_allocation_begin =
        AllocatorPolicy::W8OwnedAllocationCalls();
    const bool dt_valid = std::isfinite(dt) && dt > 0.0 &&
        (config_.use_borrowed_host_inputs ||
         std::memcmp(&dt, &fixed_dt, sizeof(double)) == 0) &&
        (!retry || std::memcmp(&dt, &retained_retry_dt_, sizeof(double)) == 0);
    if (!numerics_.CollectiveContinue(dt_valid)) return false;
    const BeginOutcome begin = retry ? contract_->Retry() :
        contract_->BeginStep({InteriorSpace::l2, dt});
    evidence.attempt = begin.attempt;
    if (!begin.started)
    {
      evidence.status = CudaRunStatus::terminal_failure;
      evidence.contract_reason = begin.reason;
      evidence.after = contract_->coordinates();
      return false;
    }
    if (begin.attempt.sequence != context.attempt_sequence)
    {
      return FinishBackendFailure(begin.attempt, evidence,
                                  CudaFailurePhase::setup);
    }

    state_slot& workspace = numerics_.WorkspaceSlot();
    storage_policy::SetCopyOutPurpose(detail::CudaCopyOutPurpose::attempt_seed);
    const bool seed_ready = contract_->CopyPublished(
        contract_->Published(), workspace);
    storage_policy::SetCopyOutPurpose(detail::CudaCopyOutPurpose::none);
    if (!numerics_.CollectiveContinue(seed_ready) ||
        !numerics_.PrepareAttempt(begin.attempt, workspace,
                                  evidence.before.step == 0 && !retry))
    {
      return FinishBackendFailure(begin.attempt, evidence,
                                  CudaFailurePhase::setup);
    }
    evidence.before_interior_hash = workspace.interior.hash;
    evidence.before_trace_hash = workspace.trace.hash;

    const detail::CudaNumericalResult numerical = numerics_.Execute(
        config_, begin.attempt, workspace);
    typename contract_type::ProductBundle bundle{};
    detail::CudaNumericalResult adjusted = numerical;
    IssueProducts(begin.attempt, adjusted, bundle);
    if (!numerics_.CollectiveContinue(adjusted.candidate_ready))
    {
      return FinishNumericalFailure(begin.attempt, adjusted, evidence);
    }
    bool stored = false;
    const auto capability = contract_->AcquireCandidate(begin.attempt);
    stored = capability && contract_->StoreCandidate(*capability, workspace);
    if (!stored)
    {
      adjusted.success = false;
      adjusted.telemetry = false;
      adjusted.reason = FailureReason::candidate_preparation;
    }
    if (!numerics_.CollectiveContinue(stored))
    {
      return FinishBackendFailure(begin.attempt, evidence,
                                  CudaFailurePhase::after_recovery_before_commit);
    }

    const CudaTransferTotals totals = TransportPolicy::Accounting();
    AttemptReport report{};
    report.gates = {adjusted.finite,
                    adjusted.admissible,
                    adjusted.recovery,
                    adjusted.true_residual,
                    adjusted.telemetry && numerics_.NoFallback(),
                    adjusted.success};
    report.telemetry.identity = begin.attempt;
    report.telemetry.transfers = ProjectSharedAccounting(totals);
    report.telemetry.prohibited.fallback_events =
        numerics_.NoFallback() ? 0 : 1;
    report.telemetry.true_relative_residual = adjusted.true_relative_residual;
    report.candidate_cursor = begin.attempt.sequence;
    report.originating_failure = adjusted.success ? FailureReason::none :
        adjusted.reason;
    pending_.active = true;
    pending_.prestage_allowed = false;
    pending_.attempt = begin.attempt;
    retained_retry_dt_ = begin.attempt.dt;
    pending_.report = report;
    pending_.bundle = std::move(bundle);
    pending_.numerical = adjusted;
    pending_.candidate_interior_hash = workspace.interior.hash;
    pending_.candidate_trace_hash = workspace.trace.hash;
    pending_.candidate_local_interior_hash = workspace.interior.local_hash;
    pending_.candidate_local_trace_hash = workspace.trace.local_hash;
    evidence.phase = adjusted.phase;
    evidence.contract_reason = adjusted.reason;
    evidence.after = evidence.before;
    evidence.products = adjusted.products;
    evidence.transfers = totals;
    evidence.initial_residual = adjusted.initial_residual;
    evidence.final_residual = adjusted.final_residual;
    evidence.true_relative_residual = adjusted.true_relative_residual;
    evidence.nonlinear_updates = adjusted.nonlinear_updates;
    evidence.trace_solves = adjusted.trace_solves;
    evidence.linear_iterations = adjusted.linear_iterations;
    evidence.line_search_trials = adjusted.line_search_trials;
    evidence.rejected_line_search_trials =
        adjusted.rejected_line_search_trials;
    evidence.source_rank = adjusted.injection_source_rank;
    evidence.device_ticket_count = numerics_.TicketCount();
    std::copy_n(numerics_.Tickets().begin(), evidence.device_ticket_count,
                evidence.device_tickets.begin());
    evidence.committed = false;
    evidence.retry_available = false;
    evidence.collective_consensus = false;
    evidence.no_fallback = numerics_.NoFallback();
    evidence.device_resident = numerics_.DeviceResident();
    evidence.status = CudaRunStatus::ready;
    pending_.evidence = evidence;
    auto staging = contract_->FreezeCandidate(begin.attempt,
                                               report.candidate_cursor);
    const bool frozen = static_cast<bool>(staging);
    int frozen_ranks = 0;
    const bool frozen_census = numerics_.CollectiveCount(frozen, frozen_ranks);
    if (!frozen_census ||
        (frozen_ranks != 0 && frozen_ranks != config_.expected_rank_count))
    {
      // The shared c879 API has no phase-polymorphic collective abort.  Retain
      // any live affine authority; dropping it or entering different rollback
      // APIs would be incorrect.  CudaFutureSharedAbiRequest records the exact
      // minimal extension required for this terminal protocol condition.
      if (staging) { pending_.staging.emplace(std::move(*staging)); }
      pending_.report.originating_failure = FailureReason::collective_protocol;
      pending_.report.gates.success = false;
      pending_.evidence.status = CudaRunStatus::terminal_failure;
      pending_.evidence.contract_reason = FailureReason::collective_protocol;
      pending_.evidence.collective_consensus = false;
      UpdateAttemptEvidence(pending_.evidence);
      evidence = pending_.evidence;
      initialized_ = false;
      return false;
    }
    if (!staging) { return FinishFrozenFailure(evidence); }
    pending_.staging.emplace(std::move(*staging));
    const bool local_prestage_allowed =
        adjusted.success && !adjusted.injected;
    const bool collective_prestage_allowed =
        numerics_.CollectiveContinue(local_prestage_allowed);
    pending_.prestage_allowed = collective_prestage_allowed;
    if (!pending_.prestage_allowed)
    {
      const CudaReuseEvidence reuse = numerics_.ReuseEvidence();
      const std::uint64_t failure_mask = detail::CudaPrestageFailureMask(
          adjusted.finite, adjusted.admissible, adjusted.recovery,
          adjusted.true_residual, adjusted.telemetry,
          numerics_.NoFallback(), adjusted.success, adjusted.injected,
          collective_prestage_allowed);
      // A requested one-shot injection is an expected rollback stimulus, not
      // a terminal backend diagnostic.  Its durable observability is the
      // retry/status evidence returned by the engine and application.
      if (!adjusted.injected)
      {
        detail::ReportCudaPrestageDenial(
            numerics_.rank(), failure_mask, adjusted.reason, adjusted.phase,
            adjusted.initial_residual, adjusted.final_residual,
            adjusted.true_relative_residual, adjusted.nonlinear_updates,
            adjusted.trace_solves, totals,
            reuse.baseline_allocation_events +
                reuse.post_baseline_allocation_events,
            reuse.baseline_allocation_events,
            reuse.baseline_w8_owned_allocation_calls +
                reuse.post_baseline_w8_owned_allocation_calls,
            reuse.baseline_w8_owned_allocation_calls);
      }
      pending_.report.gates.success = false;
      if (pending_.report.originating_failure == FailureReason::none)
        pending_.report.originating_failure = FailureReason::backend_failure;
    }
    UpdateAttemptEvidence(pending_.evidence);
    evidence = pending_.evidence;
    return true;
  }

  void IssueProducts(const AttemptIdentity& attempt,
                     detail::CudaNumericalResult& numerical,
                     typename contract_type::ProductBundle& bundle) noexcept
  {
    if (!numerical.candidate_ready) { return; }
    const CudaTokenFault fault = config_.injection.token_fault;
    if (fault == CudaTokenFault::issue_matrix_before_residual)
    {
      (void)contract_->template IssueProduct<DerivedProductKind::matrix>(
          attempt);
    }
    AttemptIdentity issue_attempt = attempt;
    if (fault == CudaTokenFault::stale_attempt) { ++issue_attempt.sequence; }
    bundle.residual = contract_->template IssueProduct<
        DerivedProductKind::residual>(issue_attempt);
    bundle.matrix = contract_->template IssueProduct<
        DerivedProductKind::matrix>(attempt);
    bundle.factor = contract_->template IssueProduct<
        DerivedProductKind::factor>(attempt);
    if (fault != CudaTokenFault::omit_solution)
    {
      bundle.solution = contract_->template IssueProduct<
          DerivedProductKind::solution>(attempt);
    }
    numerical.products.issued_mask =
        static_cast<std::uint8_t>((bundle.residual ? 1U : 0U) |
                                  (bundle.matrix ? 2U : 0U) |
                                  (bundle.factor ? 4U : 0U) |
                                  (bundle.solution ? 8U : 0U));
  }

  bool FinishBackendFailure(const AttemptIdentity& attempt,
                            CudaStepEvidence& evidence,
                            const CudaFailurePhase phase) noexcept
  {
    AttemptReport report{};
    report.gates = {};
    report.telemetry.identity = attempt;
    report.telemetry.transfers = ProjectSharedAccounting(
        TransportPolicy::Accounting(),
        AllocatorPolicy::W8OwnedAllocationCalls() -
            pending_.w8_owned_allocation_begin);
    report.telemetry.true_relative_residual =
        std::numeric_limits<double>::infinity();
    report.candidate_cursor = attempt.sequence;
    report.originating_failure = FailureReason::backend_failure;
    pending_.active = true;
    pending_.prestage_allowed = false;
    pending_.attempt = attempt;
    pending_.report = report;
    pending_.bundle = {};
    pending_.numerical = {};
    pending_.numerical.phase = phase;
    evidence.phase = phase;
    evidence.status = CudaRunStatus::ready;
    evidence.contract_reason = FailureReason::backend_failure;
    evidence.after = evidence.before;
    pending_.evidence = evidence;
    const AttemptOutcome outcome = contract_->FinishAttempt(
        attempt, report, std::move(pending_.bundle));
    FinishOutcome(outcome, pending_.evidence);
    evidence = pending_.evidence;
    ClearConsumedPending();
    return false;
  }

  bool FinishNumericalFailure(
      const AttemptIdentity& attempt,
      const detail::CudaNumericalResult& numerical,
      CudaStepEvidence& evidence) noexcept
  {
    const CudaTransferTotals totals = TransportPolicy::Accounting();
    const FailureReason reason = numerical.reason == FailureReason::none
        ? FailureReason::backend_failure : numerical.reason;
    AttemptReport report{};
    report.gates = {numerical.finite, numerical.admissible,
                    numerical.recovery, numerical.true_residual,
                    numerical.telemetry && numerics_.NoFallback(), false};
    report.telemetry.identity = attempt;
    report.telemetry.transfers = ProjectSharedAccounting(
        totals, AllocatorPolicy::W8OwnedAllocationCalls() -
                    pending_.w8_owned_allocation_begin);
    report.telemetry.prohibited.fallback_events =
        numerics_.NoFallback() ? 0 : 1;
    report.telemetry.true_relative_residual =
        numerical.true_relative_residual;
    report.candidate_cursor = attempt.sequence;
    report.originating_failure = reason;
    pending_.active = true;
    pending_.prestage_allowed = false;
    pending_.attempt = attempt;
    pending_.report = report;
    pending_.bundle = {};
    pending_.numerical = numerical;
    evidence.phase = numerical.phase;
    evidence.contract_reason = reason;
    evidence.after = evidence.before;
    evidence.products = numerical.products;
    evidence.transfers = totals;
    evidence.initial_residual = numerical.initial_residual;
    evidence.final_residual = numerical.final_residual;
    evidence.true_relative_residual = numerical.true_relative_residual;
    evidence.nonlinear_updates = numerical.nonlinear_updates;
    evidence.trace_solves = numerical.trace_solves;
    evidence.linear_iterations = numerical.linear_iterations;
    evidence.line_search_trials = numerical.line_search_trials;
    evidence.rejected_line_search_trials =
        numerical.rejected_line_search_trials;
    evidence.source_rank = numerical.injection_source_rank;
    evidence.device_ticket_count = numerics_.TicketCount();
    std::copy_n(numerics_.Tickets().begin(), evidence.device_ticket_count,
                evidence.device_tickets.begin());
    evidence.no_fallback = numerics_.NoFallback();
    evidence.device_resident = numerics_.DeviceResident();
    evidence.status = CudaRunStatus::ready;
    pending_.evidence = evidence;
    const AttemptOutcome outcome = contract_->FinishAttempt(
        attempt, report, std::move(pending_.bundle));
    FinishOutcome(outcome, pending_.evidence);
    evidence = pending_.evidence;
    ClearConsumedPending();
    return false;
  }

  bool FinishFrozenFailure(CudaStepEvidence& evidence) noexcept
  {
    pending_.report.originating_failure = FailureReason::collective_protocol;
    pending_.report.gates.success = false;
    UpdateSharedAccounting();
    const AttemptOutcome outcome = contract_->FinishAttempt(
        pending_.attempt, pending_.report, std::move(pending_.bundle));
    FinishOutcome(outcome, pending_.evidence);
    evidence = pending_.evidence;
    ClearConsumedPending();
    return false;
  }

  struct PendingAttempt
  {
    bool active = false;
    bool prestage_allowed = false;
    bool checkpoint_staged = false;
    bool oracle_staged = false;
    bool output_staged = false;
    bool has_oracle = false;
    AttemptIdentity attempt{};
    AttemptReport report{};
    typename contract_type::ProductBundle bundle{};
    std::optional<typename contract_type::StagingToken> staging{};
    std::optional<typename contract_type::PreparedCommitToken> prepared{};
    std::optional<typename contract_type::ProvisionalCommitToken>
        provisional{};
    detail::CudaNumericalResult numerical{};
    CudaStepEvidence evidence{};
    StateHash candidate_interior_hash{};
    StateHash candidate_trace_hash{};
    StateHash candidate_local_interior_hash{};
    StateHash candidate_local_trace_hash{};
    CheckpointDescriptor prepared_checkpoint{};
    std::uint64_t ledger_begin = 0;
    std::uint64_t w8_owned_allocation_begin = 0;
    CudaOraclePod oracle{};
  };

  CudaConfig config_{};
  MPI_Comm communicator_ = MPI_COMM_WORLD;
  detail::CudaPersistentNumerics<AllocatorPolicy, TransportPolicy> numerics_{};
  std::optional<contract_type> contract_{};
  PendingAttempt pending_{};
  CudaOraclePod last_oracle_{};
  std::uint64_t next_attempt_sequence_ = 0;
  double retained_retry_dt_ = fixed_dt;
  double committed_maximum_wave_speed_ = 0.0;
  double committed_backend_stable_dt_ = std::numeric_limits<double>::quiet_NaN();
  std::uint64_t facade_incarnation_ = 0;
  bool initialized_ = false;
  bool committed_diagnostics_valid_ = false;
  bool last_oracle_valid_ = false;
};

} // namespace dg::hdg::runtime::cuda_w8a

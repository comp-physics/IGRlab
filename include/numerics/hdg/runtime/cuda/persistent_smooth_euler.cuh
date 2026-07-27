#pragma once

/**
 * @file numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh
 * @brief Persistent conventional smooth-Euler CUDA smoke engine.
 *
 * Numerical provenance: the pointwise Euler flux/source, W7 weak signs,
 * A/B/C/D condensation, HypreGMRES+HypreILU(0), and recovery equations are a
 * minimal port of accepted W7 commit ac271a038c9d26648fce8f2ebf6afb0113a92495.
 * The W7 monolithic RunSmoothEulerDeviceFullExperiment lifecycle is neither
 * included nor called.  Persistence, contract publication, retry, policies,
 * and exported POD are W8A-owned code.
 */

#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_types.cuh"
#include "numerics/hdg/trace/trace_dg_mpi.hpp"

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

/** Production W8-owned allocator. All explicit device allocation is routed here. */
struct UmpireDeviceAllocatorPolicy
{
  static void ResetLedger() noexcept;
  /** Register MFEM's lazy DEVICE_UMPIRE pool through one recorded MFEM probe. */
  static bool PrepareDeviceArenaAllocator() noexcept;
  static bool Allocate(std::size_t bytes, std::size_t alignment,
                       CudaAllocationSite site, void*& pointer) noexcept;
  static void Deallocate(void* pointer, CudaAllocationSite site) noexcept;
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
};

/** Real rank-1/rank-2 reduction policy consumed by PersistentRunContract. */
struct CudaMpiCollectivePolicy
{
  static CollectiveAdmissionReport Admit(
      const AttemptCollectiveContext& context, bool prepared,
      FailureReason reason, std::uint32_t expected_ranks) noexcept;
  static CollectiveGateReport Reduce(
      const AttemptCollectiveContext& context, const GateReport& gates,
      FailureReason reason, std::uint32_t expected_ranks) noexcept;
  static CollectiveRestoreReport ReduceRestore(
      const RestoreCollectiveContext& context,
      RestartCompatibility compatibility, FailureReason reason,
      std::uint32_t expected_ranks) noexcept;
};

namespace detail
{

enum class CudaCopyOutPurpose : unsigned char
{
  none,
  attempt_seed,
  checkpoint_staging
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
  int nonlinear_updates = 0;
  int trace_solves = 0;
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
  return Selection::interior_space == InteriorSpace::l2 && tuple.Valid() &&
      ExactId(tuple.model, "euler.compressible.2d") &&
      ExactId(tuple.formulation, "hdg.conventional") &&
      ExactId(tuple.trace, "mfem.dg_interface.trace") &&
      ExactId(tuple.stabilization, "tau.fixed") &&
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
      : config_(config), facade_incarnation_(
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
    MPI_Comm_rank(MPI_COMM_WORLD, &evidence.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &evidence.ranks);
    const bool descriptor_consensus = ConfigurationConsensus();
    const int local_config =
        ValidateConfig() && facade_incarnation_ != 0 ? 1 : 0;
    int global_config = 0;
    const bool validity_consensus = MPI_Allreduce(
        &local_config, &global_config, 1, MPI_INT, MPI_MIN,
        MPI_COMM_WORLD) == MPI_SUCCESS && global_config == 1;
    if (!descriptor_consensus || !validity_consensus)
    {
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
    if (!numerics_.Initialize(config_, evidence, Selection::Descriptor()))
    {
      evidence.status = CudaRunStatus::setup_failure;
      evidence.setup_origin = FailureReason::backend_failure;
      detail::ReportCudaSetupFailure(evidence.rank, "numerics-initialize");
      return false;
    }
    StateCoordinates initial{};
    initial.dt = 0.0;
    contract_.emplace(numerics_.CommittedSlot(), numerics_.CandidateSlot(),
                      initial, numerics_.Resources());
    if (contract_->status() != RunStatus::ready)
    {
      evidence.status = CudaRunStatus::setup_failure;
      evidence.setup_origin = contract_->last_failure();
      detail::ReportCudaSetupFailure(evidence.rank, "contract-ready");
      return false;
    }
    initialized_ = true;
    evidence.reuse = numerics_.ReuseEvidence();
    evidence.status = CudaRunStatus::ready;
    return true;
  }

  /** Complete numerical work, freeze the contract candidate, and retain the
   * affine staging token internally.  No rank may stage before the final
   * all-rank prestaging agreement in this call. */
  bool PrepareStep(const bool retry, CudaPrecommitSnapshot& snapshot,
                   CudaStepEvidence& evidence) noexcept
  {
    snapshot = {};
    if (pending_.active || !initialized_) { return false; }
    if (!PreparePending(retry, evidence)) { return false; }
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
    const std::uint64_t staging_begin =
        TransportPolicy::Ledger().observed_count + 1;
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
    const CudaTransferLedgerV1 ledger = TransportPolicy::Ledger();
    checkpoint.staging_range.begin_ordinal = staging_begin;
    checkpoint.staging_range.end_ordinal = ledger.observed_count;
    checkpoint.staging_range.ordered_digest = RangeDigest(
        ledger, staging_begin, ledger.observed_count);
    checkpoint.staging_range.complete = all_staged && !ledger.overflow &&
        ledger.observed_count + 1 >= staging_begin;
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
  bool MarkOutputStaged(CudaSerializedCheckpointV1& checkpoint) noexcept
  {
    if (!pending_.active || !pending_.staging ||
        !pending_.prestage_allowed || !pending_.checkpoint_staged ||
        checkpoint.facade_incarnation != facade_incarnation_ ||
        checkpoint.attempt.sequence != pending_.attempt.sequence ||
        !dg::hdg::runtime::detail::SameCheckpoint(
            checkpoint.descriptor, pending_.staging->future_checkpoint()))
      return false;
    checkpoint.output_staged = true;
    pending_.output_staged = true;
    return true;
  }

  /** Consume StagingToken and retain PreparedCommitToken internally. */
  bool PrepareCommit(CudaSerializedCheckpointV1& checkpoint,
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
    const bool copies_complete = TransportPolicy::Synchronize(
        CudaTransferSite::checkpoint_stage_trace);
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
  bool CommitPrepared(CudaSerializedCheckpointV1& checkpoint,
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

  /** Sole logical publication boundary; no fallible CUDA or staging hook follows. */
  bool FinalizeCommit(CudaSerializedCheckpointV1& checkpoint,
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
    if (!local_artifacts_publishable)
      return false;
    const AttemptOutcome outcome =
        contract_->FinalizeCommit(std::move(*pending_.provisional));
    pending_.provisional.reset();
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
  bool AbortPrepared(CudaSerializedCheckpointV1& checkpoint,
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

  bool RevertCommit(CudaSerializedCheckpointV1& checkpoint,
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
                      initial, numerics_.Resources(), restored_frontier);
    const RestoreOutcome restored =
        contract_->Restore(checkpoint.descriptor, loaded);
    if (!restored.restored)
    {
      contract_.reset();
      contract_.emplace(numerics_.CommittedSlot(), numerics_.CandidateSlot(),
                        initial, numerics_.Resources(), 0);
      next_attempt_sequence_ = 0;
    }
    else
    {
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

  CudaTransferLedgerV1 TransferLedger() const noexcept
  { return TransportPolicy::Ledger(); }
  CudaAllocationLedgerV1 AllocationLedger() const noexcept
  { return AllocatorPolicy::Ledger(); }

  const contract_type* contract() const noexcept
  {
    return contract_ ? &*contract_ : nullptr;
  }

private:
  static void AbortStagedArtifacts(
      CudaSerializedCheckpointV1& checkpoint) noexcept
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
        local, minimum, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local, maximum, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS;
    int ranks = 0;
    const bool size_ready =
        MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS;
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

  static std::uint64_t RangeDigest(const CudaTransferLedgerV1& ledger,
                                   const std::uint64_t begin,
                                   const std::uint64_t end) noexcept
  {
    std::uint64_t digest = 1469598103934665603ULL;
    for (std::size_t index = 0; index < ledger.count; ++index)
    {
      const TransferEventV1& event = ledger.events[index];
      if (event.ordinal < begin || event.ordinal > end) { continue; }
      digest ^= event.endpoint_token;
      digest *= 1099511628211ULL;
    }
    return digest;
  }

  void UpdateAttemptEvidence(CudaStepEvidence& evidence) const noexcept
  {
    const CudaTransferLedgerV1 ledger = TransportPolicy::Ledger();
    evidence.transfers = TransportPolicy::Accounting();
    if (evidence.transfer_range.begin_ordinal == 0)
      evidence.transfer_range.begin_ordinal = pending_.ledger_begin;
    evidence.transfer_range.end_ordinal = ledger.observed_count;
    evidence.transfer_range.ordered_digest = RangeDigest(
        ledger, evidence.transfer_range.begin_ordinal,
        evidence.transfer_range.end_ordinal);
    evidence.transfer_range.complete = !ledger.overflow &&
        evidence.transfers.complete &&
        evidence.transfer_range.end_ordinal + 1 >=
            evidence.transfer_range.begin_ordinal;
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
    return config_.experiment == StableIdentity::Copy(experiment_id) &&
        config_.order == polynomial_order &&
        config_.mesh_n == elements_per_direction &&
        std::isfinite(config_.dt) &&
        std::memcmp(&config_.dt, &fixed_dt, sizeof(double)) == 0 &&
        config_.requested_device == 0 &&
        (config_.expected_rank_count == 1 ||
         config_.expected_rank_count == 2) &&
        (config_.expected_rank_count == 1 ||
         config_.require_host_staged_rank2) &&
        (!config_.injection.enabled ||
         (config_.injection.phase ==
              CudaFailurePhase::after_recovery_before_commit &&
          config_.injection.target_step == 2 &&
          config_.injection.source_rank >= 0 &&
          config_.injection.source_rank < config_.expected_rank_count)) &&
        detail::ExactSelection<Selection>(config_);
  }

  using ConfigDescriptor = std::array<unsigned char, 1024>;

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
    ConfigDescriptor local{};
    ConfigDescriptor minimum{};
    ConfigDescriptor maximum{};
    const bool descriptor_complete = MakeConfigDescriptor(local);
    minimum = local;
    maximum = local;
    int ranks = 0;
    const bool size_ready =
        MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS;
    const bool minimum_ready = MPI_Allreduce(
        local.data(), minimum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool maximum_ready = MPI_Allreduce(
        local.data(), maximum.data(), static_cast<int>(local.size()),
        MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS;
    return descriptor_complete && size_ready && minimum_ready &&
        maximum_ready && ranks == config_.expected_rank_count &&
        minimum == maximum;
  }

  bool PreparePending(const bool retry, CudaStepEvidence& evidence) noexcept
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
    const BeginOutcome begin = retry ? contract_->Retry() :
        contract_->BeginStep({InteriorSpace::l2, config_.dt});
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
    bool stored = false;
    if (adjusted.candidate_ready)
    {
      const auto capability = contract_->AcquireCandidate(begin.attempt);
      stored = capability && contract_->StoreCandidate(*capability, workspace);
      if (!stored)
      {
        adjusted.success = false;
        adjusted.telemetry = false;
        adjusted.reason = FailureReason::candidate_preparation;
      }
    }
    if (!numerics_.CollectiveContinue(adjusted.candidate_ready && stored))
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
    pending_.prestage_allowed = numerics_.CollectiveContinue(
        adjusted.success && !adjusted.injected);
    if (!pending_.prestage_allowed)
    {
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
  detail::CudaPersistentNumerics<AllocatorPolicy, TransportPolicy> numerics_{};
  std::optional<contract_type> contract_{};
  PendingAttempt pending_{};
  CudaOraclePod last_oracle_{};
  std::uint64_t next_attempt_sequence_ = 0;
  std::uint64_t facade_incarnation_ = 0;
  bool initialized_ = false;
  bool last_oracle_valid_ = false;
};

} // namespace dg::hdg::runtime::cuda_w8a

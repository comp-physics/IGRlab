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
#include <cmath>
#include <cstddef>
#include <cstdint>
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
  static bool Allocate(std::size_t bytes, std::size_t alignment,
                       CudaAllocationSite site, void*& pointer) noexcept;
  static void Deallocate(void* pointer, CudaAllocationSite site) noexcept;
  static std::uint64_t AllocationEvents() noexcept;
  static std::uint64_t DeallocationEvents() noexcept;
  static bool DevicePoolPolicyValid() noexcept;
};

/** Production W8-owned transport. All explicit CUDA copies route here. */
struct CudaTransportPolicy
{
  static void BeginAttempt(const CudaPolicyContext& context) noexcept;
  static void SetPhase(CudaFailurePhase phase, std::uint64_t generation) noexcept;
  static bool HostToDevice(void* destination, const void* source,
                           std::size_t bytes, CudaTransferCategory category,
                           CudaTransferSite site) noexcept;
  static bool DeviceToHost(void* destination, const void* source,
                           std::size_t bytes, CudaTransferCategory category,
                           CudaTransferSite site) noexcept;
  static bool DeviceToDevice(void* destination, const void* source,
                             std::size_t bytes, CudaTransferCategory category,
                             CudaTransferSite site) noexcept;
  static void RecordMpiStaging(std::size_t bytes,
                               CudaTransferSite site) noexcept;
  static bool Synchronize(CudaTransferSite site) noexcept;
  static CudaTransferTotals Accounting() noexcept;
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
    destination.generation = source.generation;
    destination.mfem_device_valid = source.mfem_device_valid;
    return true;
  }

  static bool PrepareCandidate(const state_slot& committed,
                               state_slot& candidate) noexcept
  {
    return CopyInterior(candidate.interior, committed.interior,
                        CudaTransferCategory::state_prepare,
                        CudaTransferSite::candidate_prepare_interior) &&
        CopyTrace(candidate.trace, committed.trace,
                  CudaTransferCategory::state_prepare,
                  CudaTransferSite::candidate_prepare_trace);
  }

  static bool PrepareRestore(const state_slot& loaded,
                             state_slot& candidate) noexcept
  {
    return CopyInterior(candidate.interior, loaded.interior,
                        CudaTransferCategory::state_store,
                        CudaTransferSite::candidate_store_interior) &&
        CopyTrace(candidate.trace, loaded.trace,
                  CudaTransferCategory::state_store,
                  CudaTransferSite::candidate_store_trace);
  }

  static bool StoreCandidate(state_slot& candidate,
                             const state_slot& value) noexcept
  {
    return CopyInterior(candidate.interior, value.interior,
                        CudaTransferCategory::state_store,
                        CudaTransferSite::candidate_store_interior) &&
        CopyTrace(candidate.trace, value.trace,
                  CudaTransferCategory::state_store,
                  CudaTransferSite::candidate_store_trace);
  }

  static bool CopyOut(const state_slot& source, state_slot& copy) noexcept
  {
    return CopyInterior(copy.interior, source.interior,
                        CudaTransferCategory::state_copy_out,
                        CudaTransferSite::published_copy_interior) &&
        CopyTrace(copy.trace, source.trace,
                  CudaTransferCategory::state_copy_out,
                  CudaTransferSite::published_copy_trace);
  }

  static void DiscardCandidate(state_slot& candidate) noexcept
  {
    candidate.interior.generation = 0;
    candidate.trace.generation = 0;
    candidate.interior.hash = {};
    candidate.trace.hash = {};
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
      : config_(config)
  {
    static_assert(!std::is_polymorphic<Selection>::value,
                  "CUDA selection must remain static");
    static_assert(!std::is_polymorphic<AllocatorPolicy>::value,
                  "CUDA allocator policy must remain static");
    static_assert(!std::is_polymorphic<TransportPolicy>::value,
                  "CUDA transport policy must remain static");
  }

  PersistentCudaSmoothEulerEngine(const PersistentCudaSmoothEulerEngine&) =
      delete;
  PersistentCudaSmoothEulerEngine& operator=(
      const PersistentCudaSmoothEulerEngine&) = delete;

  bool Initialize(CudaRunEvidence& evidence) noexcept
  {
    evidence = {};
    if (!ValidateConfig() ||
        !numerics_.Initialize(config_, evidence, Selection::Descriptor()))
    {
      evidence.status = CudaRunStatus::setup_failure;
      return false;
    }
    StateCoordinates initial{};
    initial.dt = 0.0;
    contract_.emplace(numerics_.CommittedSlot(), numerics_.CandidateSlot(),
                      initial, numerics_.Resources());
    if (contract_->status() != RunStatus::ready)
    {
      evidence.status = CudaRunStatus::setup_failure;
      return false;
    }
    initialized_ = true;
    evidence.reuse = numerics_.ReuseEvidence();
    evidence.status = CudaRunStatus::ready;
    return true;
  }

  /**
   * Complete all fallible numerical work and expose immutable candidate
   * coordinates/hashes without publishing contract state.  Shared output and
   * checkpoint staging must occur after this call and before FinalizePrepared.
   */
  bool PrepareStep(const bool retry, CudaPrecommitSnapshot& snapshot,
                   CudaStepEvidence& evidence) noexcept
  {
    snapshot = {};
    if (pending_.active || !initialized_) { return false; }
    if (!PreparePending(retry, evidence)) { return false; }
    snapshot.attempt = pending_.attempt;
    snapshot.committed = evidence.before;
    snapshot.target = {evidence.before.time + config_.dt,
                       evidence.before.step + 1,
                       evidence.before.generation + 1,
                       pending_.attempt.sequence,
                       config_.dt};
    snapshot.candidate_interior_hash =
        pending_.candidate_interior_hash;
    snapshot.candidate_trace_hash = pending_.candidate_trace_hash;
    snapshot.products = pending_.numerical.products;
    snapshot.interior_coefficients =
        numerics_.WorkspaceSlot().interior.count;
    snapshot.true_trace_coefficients = numerics_.WorkspaceSlot().trace.count;
    snapshot.completed_phase = pending_.numerical.phase;
    snapshot.numerical_candidate_ready = pending_.numerical.candidate_ready;
    return true;
  }

  /** Record that checkpoint and output artifacts are staged and immutable. */
  bool MarkPreparedStagingReady(const AttemptIdentity& attempt) noexcept
  {
    if (!pending_.active || !contract_->IsCurrentAttempt(attempt))
      return false;
    pending_.staging_ready = true;
    return true;
  }

  /** Abort already-staged outer artifacts, then consume the live attempt by rollback. */
  bool AbortPrepared(const FailureReason reason,
                     CudaStepEvidence& evidence) noexcept
  {
    if (!pending_.active) { return false; }
    // Non-source ranks pass none while participating in the same collective
    // abort. A source rank supplies the concrete staging failure.
    if (reason != FailureReason::none)
    {
      pending_.report.originating_failure = reason;
      pending_.report.gates.success = false;
    }
    return CompletePrepared(evidence);
  }

  /**
   * Publish contract state only after every fallible outer staging operation
   * has succeeded.  There is intentionally no fallible post-finalize hook.
   * A true atomic state+artifact Commit failure remains impossible until the
   * CudaFutureSharedAbiRequest contract extension is coordinator-owned.
   */
  bool FinalizePrepared(CudaStepEvidence& evidence) noexcept
  {
    return pending_.active && pending_.staging_ready &&
        CompletePrepared(evidence);
  }

  bool ExportOracle(CudaOraclePod& pod) noexcept
  {
    return initialized_ && numerics_.ExportOracle(pod);
  }

  const contract_type* contract() const noexcept
  {
    return contract_ ? &*contract_ : nullptr;
  }

private:
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
    if (!contract_->CopyPublished(contract_->Published(), workspace) ||
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
    if (adjusted.candidate_ready)
    {
      const auto capability = contract_->AcquireCandidate(begin.attempt);
      if (!capability || !contract_->StoreCandidate(*capability, workspace))
      {
        adjusted.success = false;
        adjusted.telemetry = false;
        adjusted.reason = FailureReason::candidate_preparation;
      }
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
    report.telemetry.transfers = {
        totals.host_to_device_bytes,
        totals.device_to_host_bytes,
        totals.mpi_staging_bytes,
        0,
        totals.synchronization_events,
        totals.complete};
    report.telemetry.prohibited.fallback_events =
        numerics_.NoFallback() ? 0 : 1;
    report.telemetry.true_relative_residual = adjusted.true_relative_residual;
    report.candidate_cursor = begin.attempt.sequence;
    report.originating_failure = adjusted.injected
        ? FailureReason::backend_failure
        : adjusted.reason;
    pending_.active = true;
    pending_.staging_ready = false;
    pending_.attempt = begin.attempt;
    pending_.report = report;
    pending_.bundle = std::move(bundle);
    pending_.numerical = adjusted;
    pending_.candidate_interior_hash = workspace.interior.hash;
    pending_.candidate_trace_hash = workspace.trace.hash;
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
    evidence.source_rank =
        config_.injection.enabled &&
                config_.injection.target_step == evidence.before.step + 1 &&
                begin.attempt.retry_index == 0
            ? config_.injection.source_rank
            : -1;
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
    return true;
  }

  bool CompletePrepared(CudaStepEvidence& evidence) noexcept
  {
    if (!pending_.active) { return false; }
    const detail::CudaNumericalResult numerical = pending_.numerical;
    state_slot& workspace = numerics_.WorkspaceSlot();
    // The numerical candidate has already been copied into the contract's
    // private candidate slot. Restore this facade workspace to the published
    // state before the irreversible FinishAttempt boundary. That makes the
    // rollback audit copy part of the attempt telemetry and leaves no
    // fallible post-publication transfer.
    if (!contract_->CopyPublished(contract_->Published(), workspace))
    {
      pending_.report.originating_failure = FailureReason::backend_failure;
      pending_.report.gates.success = false;
    }
    const CudaTransferTotals totals = TransportPolicy::Accounting();
    pending_.report.telemetry.transfers = {
        totals.host_to_device_bytes,
        totals.device_to_host_bytes,
        totals.mpi_staging_bytes,
        0,
        totals.synchronization_events,
        totals.complete};
    pending_.evidence.transfers = totals;
    const AttemptOutcome outcome = contract_->FinishAttempt(
        pending_.attempt, pending_.report, std::move(pending_.bundle));
    evidence = pending_.evidence;
    pending_.active = false;
    evidence.contract_reason = outcome.reason;
    evidence.after = contract_->coordinates();
    evidence.committed = outcome.committed;
    evidence.retry_available = outcome.retry_available;
    evidence.collective_consensus = outcome.reason !=
        FailureReason::collective_protocol;
    evidence.after_interior_hash = outcome.committed
        ? pending_.candidate_interior_hash : workspace.interior.hash;
    evidence.after_trace_hash = outcome.committed
        ? pending_.candidate_trace_hash : workspace.trace.hash;
    evidence.committed_bytes_preserved = outcome.committed ||
        (evidence.before_interior_hash == evidence.after_interior_hash &&
         evidence.before_trace_hash == evidence.after_trace_hash);
    evidence.coordinates_preserved = outcome.committed ||
        (evidence.before.time == evidence.after.time &&
         evidence.before.step == evidence.after.step &&
         evidence.before.generation == evidence.after.generation);
    evidence.status = outcome.committed ? CudaRunStatus::committed :
        (outcome.retry_available ? CudaRunStatus::retry_available :
         (numerical.injected ? CudaRunStatus::injected_failure :
                               CudaRunStatus::terminal_failure));
    return outcome.committed || outcome.retry_available;
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
    report.telemetry.transfers.complete = true;
    report.telemetry.true_relative_residual =
        std::numeric_limits<double>::infinity();
    report.candidate_cursor = attempt.sequence;
    report.originating_failure = FailureReason::backend_failure;
    pending_.active = true;
    pending_.staging_ready = false;
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
    return true;
  }

  struct PendingAttempt
  {
    bool active = false;
    bool staging_ready = false;
    AttemptIdentity attempt{};
    AttemptReport report{};
    typename contract_type::ProductBundle bundle{};
    detail::CudaNumericalResult numerical{};
    CudaStepEvidence evidence{};
    StateHash candidate_interior_hash{};
    StateHash candidate_trace_hash{};
  };

  CudaConfig config_{};
  detail::CudaPersistentNumerics<AllocatorPolicy, TransportPolicy> numerics_{};
  std::optional<contract_type> contract_{};
  PendingAttempt pending_{};
  std::uint64_t next_attempt_sequence_ = 0;
  bool initialized_ = false;
};

} // namespace dg::hdg::runtime::cuda_w8a

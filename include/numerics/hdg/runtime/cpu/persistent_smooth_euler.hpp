#pragma once

/**
 * @file numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp
 * @brief Bounded p=1/N=3 persistent conventional Euler HDG CPU adapter.
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
#include <optional>
#include <string_view>
#include <type_traits>

namespace dg::hdg::runtime::cpu
{

inline constexpr int cpu_dimension = 2;
inline constexpr int cpu_conserved_components = 4;
inline constexpr int cpu_polynomial_order = 1;
inline constexpr int cpu_elements_per_direction = 3;
inline constexpr int cpu_assembly_quadrature_order = 11;
inline constexpr double cpu_backward_euler_dt = 0.05;
inline constexpr double cpu_gamma = 1.4;
inline constexpr double cpu_fixed_tau = 2.5;
inline constexpr int cpu_maximum_newton_updates = 20;
inline constexpr int cpu_maximum_line_search_trials = 13;
inline constexpr double cpu_armijo_constant = 1.0e-4;
inline constexpr double cpu_nonlinear_tolerance = 1.0e-10;

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
};

CpuPersistentConfig MakeCpuSmokeConfig() noexcept;

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

struct CpuTrialEvidence
{
  int newton_iteration = 0;
  int trial_index = 0;
  double alpha = 0.0;
  double base_residual = 0.0;
  double armijo_boundary = 0.0;
  double trial_residual = 0.0;
  double minimum_density = 0.0;
  double minimum_pressure = 0.0;
  double maximum_wave_speed = 0.0;
  bool prescreen_passed = false;
  bool reassembled = false;
  bool accepted = false;
};

struct CpuProductRecord
{
  DerivedProductKind kind = DerivedProductKind::residual;
  std::uint64_t attempt_sequence = 0;
  std::uint64_t candidate_cursor = 0;
  StateHash payload_hash{};
  bool produced = false;
};

struct CpuWorkCensus
{
  std::uint64_t complete_assemblies = 0;
  std::uint64_t complete_condensations = 0;
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
  std::uint64_t topology_generation = 0;
  std::uint64_t storage_generation = 0;
  std::uint64_t allocation_baseline = 0;
  std::uint64_t allocation_events_after_baseline = 0;
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
  std::array<CpuProductRecord, 4> products{};
  std::array<CpuTrialEvidence,
             cpu_maximum_newton_updates * cpu_maximum_line_search_trials>
      trials{};
  std::size_t trial_count = 0;
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
  std::uint64_t local_owned_interior_coefficients = 0;
  std::uint64_t global_owned_interior_coefficients = 0;
  std::uint64_t local_owned_true_trace_coefficients = 0;
  std::uint64_t global_owned_true_trace_coefficients = 0;
  std::uint64_t unique_periodic_faces = 0;
  std::uint64_t negative_signed_trace_dofs = 0;
  StateHash mesh_hash{};
  StateHash partition_hash{};
  StateHash topology_hash{};
  CpuStorageEvidence storage{};
  CpuAttemptEvidence last_attempt{};
  CpuAcceptedEvidence accepted{};
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
  CpuCoefficientView interior{};
  CpuCoefficientView trace{};
};

/**
 * Attempt-qualified numerical candidate exposed only between Prepare* and
 * AbortPrepared/FinalizePrepared.  It is a read-only staging input, never a
 * published state.  A stale sequence/cursor is rejected by the facade.
 */
struct CpuPreparedStep
{
  AttemptIdentity identity{};
  std::uint64_t candidate_cursor = 0;
  StateCoordinates entry_coordinates{};
  StateCoordinates target_coordinates{};
  CpuCoefficientView interior{};
  CpuCoefficientView trace{};
  StateHash interior_hash{};
  StateHash trace_hash{};
  CpuFailurePhase failure_phase = CpuFailurePhase::none;
  bool numerically_ready = false;
  bool valid = false;
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

  static bool PrepareCandidate(const CpuStateSlot&, CpuStateSlot&) noexcept;
  static bool PrepareRestore(const CpuStateSlot&, CpuStateSlot&) noexcept;
  static bool StoreCandidate(CpuStateSlot&, const CpuStateSlot&) noexcept;
  static bool CopyOut(const CpuStateSlot&, CpuStateSlot&) noexcept;
  static void DiscardCandidate(CpuStateSlot&) noexcept;
  static bool CanPublish(const CpuStateSlot&, const CpuStateSlot&) noexcept;
  static void Publish(CpuStateSlot&, CpuStateSlot&) noexcept;
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
  CheckpointDescriptor descriptor{};
  CpuStateSlot state{};
  bool valid = false;

  CpuCoefficientView interior() const noexcept;
  CpuCoefficientView trace() const noexcept;
};

namespace detail
{

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
  const CpuRunEvidence& evidence() const noexcept;
  bool ExportCanonicalCommitted(CpuCanonicalEntry*, std::size_t,
                                std::size_t&) const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
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
    if (contract_) { return false; }
    const RuntimeTupleDescriptor tuple = Selection::Descriptor();
    const StableIdentity selected_experiment =
        StableIdentity::Copy(std::string_view{Selection::experiment_id});
    if (!tuple.Valid() || !selected_experiment.valid() ||
        selected_experiment != config.experiment)
    {
      return false;
    }
    detail::CpuInitialization initialized;
    if (!core_.Initialize(config, tuple, initialized)) { return false; }
    try
    {
      contract_ = std::make_unique<contract_type>(
          std::move(initialized.committed), std::move(initialized.candidate),
          initialized.coordinates, initialized.resources);
    }
    catch (...)
    {
      contract_.reset();
      return false;
    }
    return contract_->status() == RunStatus::ready;
  }

  bool PrepareAdvance(
      CpuPreparedStep& prepared,
      const CpuFailureInjection& injection = CpuFailureInjection{}) noexcept
  {
    prepared = {};
    if (!contract_ || prepared_active_) { return false; }
    const BeginOutcome begin = contract_->BeginStep(
        {InteriorSpace::l2, cpu_backward_euler_dt});
    return begin.started && PrepareStarted(begin, injection, prepared);
  }

  bool PrepareRetry(CpuPreparedStep& prepared) noexcept
  {
    prepared = {};
    if (!contract_ || prepared_active_) { return false; }
    const BeginOutcome begin = contract_->Retry();
    if (!begin.started) { return false; }
    CpuFailureInjection disabled{};
    return PrepareStarted(begin, disabled, prepared);
  }

  AttemptOutcome AbortPrepared(CpuPreparedStep& prepared,
                               const CpuFailurePhase phase,
                               const int source_rank = -1) noexcept
  {
    if (!MatchesPrepared(prepared))
    {
      return {status(), FailureReason::stale_attempt, false,
              status() == RunStatus::retry_available};
    }
    prepared_numerical_.originating_failure = FailureReason::backend_failure;
    prepared_numerical_.gates.success = false;
    prepared_numerical_.evidence.failure_phase = phase;
    prepared_numerical_.evidence.failure_source_rank = source_rank;
    core_.MarkPreparedFailure(phase, source_rank);
    return FinishPrepared(prepared);
  }

  AttemptOutcome FinalizePrepared(CpuPreparedStep& prepared) noexcept
  {
    if (!MatchesPrepared(prepared))
    {
      return {status(), FailureReason::stale_attempt, false,
              status() == RunStatus::retry_available};
    }
    return FinishPrepared(prepared);
  }

  AttemptOutcome Advance(
      const CpuFailureInjection& injection = CpuFailureInjection{}) noexcept
  {
    CpuPreparedStep prepared;
    if (!PrepareAdvance(prepared, injection))
    {
      return {status(), contract_ ? contract_->last_failure()
                                  : FailureReason::backend_failure,
              false, status() == RunStatus::retry_available};
    }
    return FinalizePrepared(prepared);
  }

  AttemptOutcome Retry() noexcept
  {
    CpuPreparedStep prepared;
    if (!PrepareRetry(prepared))
    {
      return {status(), contract_ ? contract_->last_failure()
                                  : FailureReason::backend_failure,
              false, false};
    }
    return FinalizePrepared(prepared);
  }

  bool CaptureCheckpoint(CpuCheckpoint& checkpoint) noexcept
  {
    if (!contract_) { return false; }
    CheckpointDescriptor descriptor;
    if (!contract_->TryMakeCheckpoint(descriptor)) { return false; }
    return core_.CaptureCheckpoint(descriptor, checkpoint);
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

  const CpuRunEvidence& evidence() const noexcept { return core_.evidence(); }

  CpuStateSnapshot Snapshot() const noexcept
  {
    return core_.Snapshot(coordinates());
  }

  bool ExportCanonicalCommitted(CpuCanonicalEntry* entries,
                                const std::size_t capacity,
                                std::size_t& written) const noexcept
  {
    return core_.ExportCanonicalCommitted(entries, capacity, written);
  }

private:
  bool PrepareStarted(const BeginOutcome& begin,
                      const CpuFailureInjection& injection,
                      CpuPreparedStep& prepared) noexcept
  {
    const StateCoordinates entry = contract_->coordinates();
    const auto capability = contract_->AcquireCandidate(begin.attempt);
    detail::CpuNumericalResult numerical;
    if (!capability)
    {
      numerical.originating_failure = FailureReason::stale_attempt;
      numerical.evidence.identity = begin.attempt;
      numerical.evidence.entry_coordinates = entry;
      numerical.evidence.candidate_cursor = begin.attempt.sequence;
      numerical.evidence.transfers.complete = true;
    }
    else
    {
      numerical = core_.Execute(begin.attempt, entry, injection);
    }

    prepared_begin_ = begin;
    prepared_numerical_ = numerical;
    prepared_product_fault_ = injection.product_fault;
    prepared_active_ = true;
    const CpuStateSnapshot candidate = core_.CandidateSnapshot(entry);
    prepared.identity = begin.attempt;
    prepared.candidate_cursor = begin.attempt.sequence;
    prepared.entry_coordinates = entry;
    prepared.target_coordinates = {
        entry.time + cpu_backward_euler_dt,
        entry.step + 1,
        entry.generation + 1,
        begin.attempt.sequence,
        cpu_backward_euler_dt};
    prepared.interior = candidate.interior;
    prepared.trace = candidate.trace;
    prepared.interior_hash = candidate.interior_hash;
    prepared.trace_hash = candidate.trace_hash;
    prepared.failure_phase = numerical.evidence.failure_phase;
    prepared.numerically_ready = numerical.originating_failure ==
            FailureReason::none && numerical.gates.All();
    prepared.valid = true;
    return true;
  }

  bool MatchesPrepared(const CpuPreparedStep& prepared) const noexcept
  {
    return prepared_active_ && prepared.valid && prepared_begin_.started &&
        ::dg::hdg::runtime::detail::SameAttempt(prepared.identity,
                                                prepared_begin_.attempt) &&
        prepared.candidate_cursor == prepared_begin_.attempt.sequence;
  }

  AttemptOutcome FinishPrepared(CpuPreparedStep& prepared) noexcept
  {
    const BeginOutcome begin = prepared_begin_;
    detail::CpuNumericalResult numerical = prepared_numerical_;

    typename contract_type::ProductBundle bundle;
    const CpuProductFault fault = prepared_product_fault_;
    if (fault != CpuProductFault::none)
    {
      numerical.evidence.status = CpuAttemptStatus::runtime_failure;
      numerical.evidence.failure_phase = CpuFailurePhase::diagnostics;
      core_.MarkPreparedFailure(CpuFailurePhase::diagnostics, -1);
    }
    if (fault == CpuProductFault::out_of_order_matrix)
    {
      (void)contract_->template IssueProduct<DerivedProductKind::matrix>(
          begin.attempt);
      numerical.originating_failure = FailureReason::invalid_product;
    }

    if (numerical.evidence.products[0].produced)
    {
      bundle.residual =
          contract_->template IssueProduct<DerivedProductKind::residual>(
              begin.attempt);
      if (fault == CpuProductFault::duplicate_residual)
      {
        (void)contract_->template IssueProduct<DerivedProductKind::residual>(
            begin.attempt);
        numerical.originating_failure = FailureReason::invalid_product;
      }
    }
    if (numerical.evidence.products[1].produced)
    {
      bundle.matrix =
          contract_->template IssueProduct<DerivedProductKind::matrix>(
              begin.attempt);
    }
    if (numerical.evidence.products[2].produced)
    {
      bundle.factor =
          contract_->template IssueProduct<DerivedProductKind::factor>(
              begin.attempt);
    }
    if (numerical.evidence.products[3].produced &&
        fault != CpuProductFault::missing_solution)
    {
      bundle.solution =
          contract_->template IssueProduct<DerivedProductKind::solution>(
              begin.attempt);
    }

    AttemptReport report;
    report.gates = numerical.gates;
    report.telemetry.identity = begin.attempt;
    report.telemetry.transfers = numerical.evidence.transfers;
    report.telemetry.prohibited = numerical.evidence.prohibited;
    report.telemetry.true_relative_residual =
        numerical.evidence.maximum_true_relative_residual;
    report.candidate_cursor = begin.attempt.sequence;
    report.originating_failure = numerical.originating_failure;
    const AttemptOutcome outcome = contract_->FinishAttempt(
        begin.attempt, report, std::move(bundle));
    core_.Complete(outcome, contract_->coordinates(), contract_->telemetry());
    prepared = {};
    prepared_begin_ = {};
    prepared_numerical_ = {};
    prepared_product_fault_ = CpuProductFault::none;
    prepared_active_ = false;
    return outcome;
  }

  detail::CpuNumericsCore core_{};
  std::unique_ptr<contract_type> contract_{};
  BeginOutcome prepared_begin_{};
  detail::CpuNumericalResult prepared_numerical_{};
  CpuProductFault prepared_product_fault_ = CpuProductFault::none;
  bool prepared_active_ = false;
};

} // namespace dg::hdg::runtime::cpu

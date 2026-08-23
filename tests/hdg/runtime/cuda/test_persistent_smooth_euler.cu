#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"

#include <mfem.hpp>

#include <iostream>
#include <memory>
#include <string_view>

namespace
{
using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::cuda_w8a;

constexpr bool TrueResidualAuthorityPolicy() noexcept
{
  dg::hdg::runtime::cuda_w8a::detail::CudaSolveDiagnostics diagnostics{};
  diagnostics.device_policy = true;
  diagnostics.configured_ilu0 = true;
  diagnostics.finite = true;
  diagnostics.iterations = 0;
  diagnostics.reported_residual = 1.1286157960703033e-12;
  diagnostics.reported_tolerance_met = false;
  const bool reported_tolerance_is_not_a_gate =
      dg::hdg::runtime::cuda_w8a::detail::
          CudaTraceSolveReadyForTrueResidualVerification(diagnostics, 8);
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
  return reported_tolerance_is_not_a_gate && cap_is_inclusive &&
      negative_iteration_is_rejected && over_cap_is_rejected;
}

static_assert(TrueResidualAuthorityPolicy(),
              "the rebuilt true residual must remain authoritative");
static_assert(
    dg::hdg::runtime::cuda_w8a::detail::CudaIluLocalReordering(false) == 0 &&
    dg::hdg::runtime::cuda_w8a::detail::CudaIluLocalReordering(true) == 1,
    "only coupled IGR uses local RCM ordering");

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

bool SetupValid(const CudaRunEvidence& setup, const int ranks) noexcept
{
  return setup.status == CudaRunStatus::ready && setup.ranks == ranks &&
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
  return All(initialized && SetupValid(setup, ranks));
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
  const bool pass = known_mode && (ranks == 1
      ? (mode == "--functional" ? FunctionalRestart<RankOneTag>(ranks)
         : mode == "--token-negatives" ? TokenFaults<RankOneTag>(ranks)
         : LifecycleAndRecovery<RankOneTag>(ranks))
      : ranks == 2
      ? (mode == "--functional" ? FunctionalRestart<RankTwoTag>(ranks)
         : mode == "--token-negatives" ? TokenFaults<RankTwoTag>(ranks)
         : LifecycleAndRecovery<RankTwoTag>(ranks))
      : false);
  if (!pass && rank == 0)
    std::cerr << "compact CUDA W8A audit failed\n";
  return pass ? 0 : 1;
}

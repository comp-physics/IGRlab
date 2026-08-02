#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr_rank2.cuh"
#include "sc25_scalar_igr_rank2_independent_oracle.cuh"
#include "sc25_scalar_igr_rank2_manifest.hpp"

#include <mfem.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200
#error "HDG-EXP-SCALAR-DEVICE-01 rank-two requires native SM120"
#endif

namespace
{
namespace scalar = ::dg::hdg::runtime::cuda_w8b::sc25;
namespace cpu = ::dg::hdg::runtime::cpu::sc25;
namespace oracle = ::dg_igr_hdg_sc25_cuda_rank2_test;
namespace manifest = ::dg_igr_hdg_sc25_rank2_manifest;

bool CollectiveAll(const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

bool SameAcrossRanks(const manifest::Digest& local) noexcept
{
  std::array<manifest::Digest, 2> gathered{};
  return MPI_Allgather(local.data(), static_cast<int>(local.size()), MPI_BYTE,
                       gathered.data(), static_cast<int>(local.size()),
                       MPI_BYTE, MPI_COMM_WORLD) == MPI_SUCCESS &&
      manifest::DigestEqual(gathered[0], gathered[1]);
}

bool ReadEnvironment(const char* const name, const std::size_t maximum,
                     std::string& value)
{
  const char* const raw = std::getenv(name);
  if (raw == nullptr) return false;
  const std::string candidate(raw);
  if (candidate.empty() || candidate.size() > maximum) return false;
  value = candidate;
  return true;
}

bool CudaAccepted(const scalar::ScalarCudaStatus status) noexcept
{
  return status == scalar::ScalarCudaStatus::accepted ||
      status == scalar::ScalarCudaStatus::accepted_after_one_retry;
}

std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank2> CreateCuda()
{
  scalar::ScalarCudaConfig config{};
  config.polynomial_degree = 1;
  config.elements_per_direction =
      cpu::persistent_elements_per_direction;
  config.requested_device = 0;
  config.require_host_staged_rank2 = true;
  std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank2> result;
  const scalar::ScalarCudaStatus status =
      scalar::PersistentSc25ScalarIgrCudaRank2::Create(
          MPI_COMM_WORLD, config, result);
  if (status != scalar::ScalarCudaStatus::ready) return {};
  return result;
}

struct HypreAdmission final
{
  HYPRE_Int initialized = 0;
  HYPRE_Int error = 0;
  HYPRE_Int memory_query = -1;
  HYPRE_Int execution_query = -1;
  HYPRE_MemoryLocation memory = HYPRE_MEMORY_UNDEFINED;
  HYPRE_ExecutionPolicy execution = HYPRE_EXEC_UNDEFINED;
  bool using_gpu = false;

  bool valid_cuda() const noexcept
  {
    return initialized != 0 && error == 0 && memory_query == 0 &&
        execution_query == 0 && memory == HYPRE_MEMORY_DEVICE &&
        execution == HYPRE_EXEC_DEVICE && using_gpu;
  }

};

HypreAdmission CaptureHypreAdmission()
{
  HypreAdmission result{};
  result.initialized = HYPRE_Initialized();
  result.memory_query = HYPRE_GetMemoryLocation(&result.memory);
  result.execution_query = HYPRE_GetExecutionPolicy(&result.execution);
  result.using_gpu = mfem::HypreUsingGPU();
  result.error = HYPRE_GetError();
  return result;
}

HypreAdmission InitializeHypre()
{
  mfem::Hypre::Init();
  return CaptureHypreAdmission();
}

void EmitHypreAdmission(const int rank, const HypreAdmission& value)
{
  std::cerr << "SC25_CUDA_RANK2_HYPRE rank=" << rank
            << " initialized=" << value.initialized
            << " error=" << value.error
            << " memory_query=" << value.memory_query
            << " memory=" << static_cast<int>(value.memory)
            << " execution_query=" << value.execution_query
            << " execution=" << static_cast<int>(value.execution)
            << " using_gpu=" << (value.using_gpu ? 1 : 0) << '\n';
}

bool LoadCpuReferenceManifest(const manifest::DirectoryHandle& directory,
                              const std::string& run_id,
                              const std::string& complete_sha256,
                              manifest::CpuReferenceManifest& reference,
                              std::string& producer_executable_sha256)
{
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  manifest::Digest expected_complete{};
  bool local = manifest::SafeRunId(run_id) &&
      manifest::ParseHexDigest(complete_sha256, expected_complete);

  std::vector<std::uint8_t> complete_bytes;
  local = local && manifest::ReadFileAt(
      directory, manifest::complete_filename, complete_bytes);
  const manifest::Digest observed_complete = local
      ? ::dg::hdg::runtime::ComputeSha256(complete_bytes.data(),
                                          complete_bytes.size())
      : manifest::Digest{};
  local = local &&
      manifest::DigestEqual(observed_complete, expected_complete);

  manifest::CompleteManifest complete{};
  local = local && manifest::DecodeComplete(
      complete_bytes, run_id, directory.identity(), complete);
  const std::size_t index = static_cast<std::size_t>(rank);
  std::vector<std::uint8_t> shard_bytes;
  if (local)
  {
    local = manifest::ReadFileAt(
                directory, complete.shards[index].filename, shard_bytes) &&
        shard_bytes.size() == complete.shards[index].bytes;
  }
  const manifest::Digest shard_digest = local
      ? ::dg::hdg::runtime::ComputeSha256(shard_bytes.data(), shard_bytes.size())
      : manifest::Digest{};
  local = local &&
      manifest::DigestEqual(shard_digest, complete.shards[index].sha256) &&
      manifest::DecodeCpuManifest(
          shard_bytes, run_id, complete.executable_sha256, rank,
          directory.identity(), reference);
  if (!CollectiveAll(local))
  {
    std::cerr << "SC25_CUDA_RANK2_MANIFEST_REJECT rank=" << rank
              << " run_id=" << run_id << '\n';
    return false;
  }
  producer_executable_sha256 = complete.executable_sha256;
  return true;
}

int RunC3(const manifest::CpuReferenceManifest& cpu_reference)
{
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int failures = 0;
  const auto require = [&](const bool local, const std::string_view message)
  {
    if (!CollectiveAll(local))
    {
      if (rank == 0) { std::cerr << "C3: " << message << '\n'; }
      ++failures;
      return false;
    }
    return true;
  };

  // The host-policy producer is a profile-local parity reference bound to the
  // exact accepted CPU receipt.  It is deliberately not a repeat of CPU-
  // profile admission.  This CUDA process consumes immutable state bytes and
  // never constructs a CPU owner under the process-global CUDA HYPRE policy.
  constexpr std::string_view expected_cpu_receipt =
      "composition_commit=8e0c8cf56300d27b2d393aa3cd4eecb0507b97e0;"
      "composition_tree=d1419c84b41e870b8b0f30cc4c054ed4bbb8d588;"
      "cpu_header_blob=e37a439dfdd8ece59d226ce2b1a448c4b60d85bb;"
      "cpu_source_blob=9f337faf90bb0913212bbe0fad2d7aeee6c619b7;"
      "oracle_hpp_blob=1ddce570deee15902679463a2a75d3ccdd1a248b;"
      "oracle_cpp_blob=9420e66504a9a521fcac4a2770a4afbbebed1710;"
      "cpu_test_blob=c83865c2cdd01346b0a3a9bfd72ccbc2eb4cb54e;"
      "evidence=/tmp/HDG-W8B-DEV2-CPU3-01/attempt-002/SHA256SUMS;"
      "evidence_sha256=53a7cfaad21db64afe6854c157323478d7e757b38b2736baba1e9d20e85b8554";
  if (!require(manifest::accepted_cpu_oracle_receipt == expected_cpu_receipt,
               "exact accepted CPU cross-backend reference receipt"))
    return 1;
  std::array<cpu::PersistentRank2TrajectoryState,
             cpu::persistent_required_steps + 1> cpu_states{};
  std::array<cpu::PersistentRank2TrajectoryState,
             cpu::persistent_required_steps + 1> cuda_baseline_states{};
  for (std::size_t step = 0; step < cpu_states.size(); ++step)
    cpu_states[step] = cpu_reference.steps[step].state;
  if (!require(manifest::ValidCpuManifest(cpu_reference),
               "validated CPU reference manifest") ||
      !require(cpu_reference.checkpoint_coordinates.step == 10 &&
                   cpu_reference.steps[10].summary.coordinates.step == 10,
               "CPU reference checkpoint coordinates/hashes"))
    return 1;
  // Test-only immutable-before storage is allocated before any owner executes
  // its initialization/warm plateau and is reused by every negative campaign.
  const auto negative_evidence_before =
      std::make_unique<scalar::ScalarCudaEvidence>();

  auto cuda_owner = CreateCuda();
  if (!require(cuda_owner != nullptr, "CUDA rank-two creation")) return 1;
  const scalar::ScalarCudaStatus cuda_initialize =
      cuda_owner->InitializePersistentTrajectory();
  if (!require(cuda_initialize == scalar::ScalarCudaStatus::ready,
               "CUDA rank-two initialization"))
    return 1;

  cpu::PersistentRank2TrajectoryState cuda_state{};
  bool cuda_state_copied =
      cuda_owner->CopyPublishedTrajectoryState(cuda_state);
  if (!require(cuda_state_copied, "CUDA step-zero state copy")) return 1;
  cuda_baseline_states[0] = cuda_state;
  // CollectiveAll makes both rank-local shard comparisons one global
  // reconstruction gate.
  if (!require(
          oracle::CrossBackendParityAgainstAcceptedCpuReceipt(
              cpu_states[0], cuda_state, 1.0e-10),
          "CUDA step-zero accepted CPU parity"))
    return 1;
  const auto& initialization_evidence = cuda_owner->trajectory_evidence();
  if (!require(initialization_evidence.snapshot_count == 1 &&
                   oracle::OperatorSnapshotParity(
                       cpu_reference.steps[0],
                       initialization_evidence.snapshots[0], 0),
               "CUDA/accepted-CPU initialization operator evidence"))
    return 1;

  scalar::PersistentSc25ScalarIgrCudaRank2::Transaction invalid{};
  const bool invalid_installed =
      cuda_owner->InstallPersistentTrajectoryTransaction(invalid);
  if (!require(!invalid_installed, "invalid affine transaction rejected"))
    return 1;

  cpu::PersistentRank2TrajectoryState before_prepare{};
  cpu::PersistentRank2TrajectoryState after_prepare{};
  const bool before_prepare_copied =
      cuda_owner->CopyPublishedTrajectoryState(before_prepare);
  if (!require(before_prepare_copied, "pre-prepare committed state")) return 1;
  scalar::PersistentSc25ScalarIgrCudaRank2::Transaction prepared{};
  const scalar::ScalarCudaStatus prepared_status =
      cuda_owner->PreparePersistentTrajectoryTransaction(prepared);
  if (!require(CudaAccepted(prepared_status), "affine prepare")) return 1;
  if (!require(prepared.valid() && !prepared.provisionally_installed(),
               "prepared affine token state"))
    return 1;
  const bool after_prepare_copied =
      cuda_owner->CopyPublishedTrajectoryState(after_prepare);
  if (!require(after_prepare_copied, "post-prepare committed state")) return 1;
  if (!require(oracle::SameStateBytes(before_prepare, after_prepare),
               "prepare did not publish"))
    return 1;
  const bool prepared_installed =
      cuda_owner->InstallPersistentTrajectoryTransaction(prepared);
  if (!require(prepared_installed, "affine provisional install")) return 1;
  if (!require(prepared.provisionally_installed(),
               "installed affine token state"))
    return 1;
  const bool prepared_reverted =
      cuda_owner->RevertPersistentTrajectoryTransaction(std::move(prepared));
  if (!require(prepared_reverted, "affine revert")) return 1;
  if (!require(!prepared.valid(), "reverted affine token consumed")) return 1;
  const bool after_revert_copied =
      cuda_owner->CopyPublishedTrajectoryState(after_prepare);
  if (!require(after_revert_copied, "post-revert committed state")) return 1;
  if (!require(oracle::SameStateBytes(before_prepare, after_prepare),
               "affine prepare/install/revert is atomic"))
    return 1;

  scalar::ScalarCudaCheckpoint cuda_checkpoint10{};
  for (int step = 1; step <= cpu::persistent_required_steps; ++step)
  {
    const scalar::ScalarCudaStatus advanced =
        cuda_owner->AdvancePersistentTrajectory();
    if (!require(CudaAccepted(advanced), "CUDA baseline trajectory")) return 1;
    cuda_state_copied = cuda_owner->CopyPublishedTrajectoryState(cuda_state);
    if (!require(cuda_state_copied, "CUDA baseline state copy")) return 1;
    cuda_baseline_states[static_cast<std::size_t>(step)] = cuda_state;
    if (!require(
            oracle::CrossBackendParityAgainstAcceptedCpuReceipt(
                cpu_states[step], cuda_state, 1.0e-10),
            "CUDA/accepted-CPU per-step parity"))
      return 1;
    const auto& stage_evidence = cuda_owner->trajectory_evidence();
    if (!require(stage_evidence.snapshot_count == step + 1 &&
                     oracle::OperatorSnapshotParity(
                         cpu_reference.steps[static_cast<std::size_t>(step)],
                         stage_evidence.snapshots[static_cast<std::size_t>(step)],
                         step),
                 "CUDA/accepted-CPU per-stage operator evidence"))
      return 1;
    if (step == 10)
    {
      const bool captured =
          cuda_owner->CaptureTrajectoryCheckpoint(cuda_checkpoint10);
      if (!require(captured, "CUDA two-shard checkpoint")) return 1;
    }
  }
  const scalar::ScalarCudaEvidence& baseline =
      cuda_owner->trajectory_evidence();
  if (!require(baseline.accepted_steps == 20 &&
                   baseline.attempted_steps == 20 &&
                   baseline.retry_attempts == 0 &&
                   baseline.step_evidence_count == 20 &&
                   baseline.snapshot_count == 21,
               "CUDA baseline trajectory census"))
    return 1;
  if (!require(oracle::RuntimeEvidence(baseline, rank),
               "CUDA residency/reuse/ledger evidence"))
    return 1;
  if (!require(oracle::Conservation(baseline),
               "CUDA accepted conservation evidence"))
    return 1;
  if (!require(oracle::AcceptedTrajectoryResidualBounds(baseline),
               "CUDA nonlinear, true-GMRES, and local residual bounds"))
    return 1;
  const cpu::PersistentRank2TrajectoryState cuda_final = cuda_state;
  cuda_owner.reset();

  // Fresh and used-owner checkpoint restore/replay.
  auto restart = CreateCuda();
  if (!require(restart != nullptr, "fresh restart owner creation")) return 1;
  const scalar::ScalarCudaStatus restart_initialized =
      restart->InitializePersistentTrajectory();
  if (!require(restart_initialized == scalar::ScalarCudaStatus::ready,
               "fresh restart owner initialization"))
    return 1;
  const scalar::ScalarCudaStatus restored =
      restart->RestoreTrajectoryCheckpoint(cuda_checkpoint10);
  if (!require(restored == scalar::ScalarCudaStatus::ready,
               "fresh two-shard restore"))
    return 1;
  if (!require(oracle::CoherentReplayEvidence(
                   restart->trajectory_evidence(), 10),
               "fresh restore invalidates all future snapshot evidence"))
    return 1;
  const bool restored_state_copied =
      restart->CopyPublishedTrajectoryState(cuda_state);
  if (!require(restored_state_copied, "fresh restored state copy")) return 1;
  if (!require(
          oracle::CrossBackendParityAgainstAcceptedCpuReceipt(
              cpu_states[10], cuda_state, 1.0e-10),
          "fresh restored accepted-CPU parity"))
    return 1;
  for (int step = 11; step <= cpu::persistent_required_steps; ++step)
  {
    const scalar::ScalarCudaStatus replayed =
        restart->AdvancePersistentTrajectory();
    if (!require(CudaAccepted(replayed), "fresh restore replay")) return 1;
    const auto& replay_evidence = restart->trajectory_evidence();
    const std::size_t replay_index = static_cast<std::size_t>(step - 11);
    if (!require(oracle::CoherentReplayEvidence(replay_evidence, 10) &&
                     oracle::OperatorSnapshotParity(
                         cpu_reference.steps[static_cast<std::size_t>(step)],
                         replay_evidence.snapshots[replay_index], step),
                 "fresh replay appends dense coherent snapshot evidence"))
      return 1;
  }
  const bool replay_state_copied =
      restart->CopyPublishedTrajectoryState(cuda_state);
  if (!require(replay_state_copied, "replay final state copy")) return 1;
  if (!require(oracle::SameStateBytes(cuda_final, cuda_state),
               "bitwise restart continuation"))
    return 1;
  const scalar::ScalarCudaStatus used_restore =
      restart->RestoreTrajectoryCheckpoint(cuda_checkpoint10);
  if (!require(used_restore == scalar::ScalarCudaStatus::ready,
               "used-owner restore"))
    return 1;
  if (!require(oracle::CoherentReplayEvidence(
                   restart->trajectory_evidence(), 10),
               "used-owner restore invalidates prior/future snapshots"))
    return 1;
  if (!require(restart->CopyPublishedTrajectoryState(cuda_state) &&
                   oracle::SameStateBytes(cuda_baseline_states[10],
                                          cuda_state),
               "used-owner restore installs exact step-ten bytes"))
    return 1;
  const scalar::ScalarCudaStatus used_replay =
      restart->AdvancePersistentTrajectory();
  if (!require(CudaAccepted(used_replay), "used-owner replay")) return 1;
  const auto& used_replay_evidence = restart->trajectory_evidence();
  if (!require(oracle::CoherentReplayEvidence(used_replay_evidence, 10) &&
                   used_replay_evidence.snapshot_count == 1 &&
                   oracle::OperatorSnapshotParity(
                       cpu_reference.steps[11],
                       used_replay_evidence.snapshots[0], 11) &&
                   restart->CopyPublishedTrajectoryState(cuda_state) &&
                   oracle::SameStateBytes(cuda_baseline_states[11],
                                          cuda_state),
               "used-owner replay appends one coherent snapshot"))
    return 1;
  restart.reset();

  // Each rank-one-origin campaign must roll back exactly once and publish the
  // same step-seven bytes as the independent baseline continuation.
  struct FaultCase final
  {
    scalar::FailureInjection injection;
    scalar::ScalarCudaStatus failure_status;
    scalar::ScalarCudaFailureCause failure_cause;
  };
  const std::array<FaultCase, 4> fault_cases{{
      {scalar::FailureInjection::local_factor,
       scalar::ScalarCudaStatus::injected_local_factor_failure,
       scalar::ScalarCudaFailureCause::injected_local_factor},
      {scalar::FailureInjection::trace_linear,
       scalar::ScalarCudaStatus::injected_trace_linear_failure,
       scalar::ScalarCudaFailureCause::injected_trace_linear},
      {scalar::FailureInjection::admissibility,
       scalar::ScalarCudaStatus::injected_admissibility_failure,
       scalar::ScalarCudaFailureCause::injected_admissibility},
      {scalar::FailureInjection::newton_iteration_limit,
       scalar::ScalarCudaStatus::injected_newton_iteration_limit,
       scalar::ScalarCudaFailureCause::injected_newton_limit}}};
  for (const FaultCase& campaign : fault_cases)
  {
    auto fault = CreateCuda();
    if (!require(fault != nullptr, "fault owner creation")) return 1;
    if (!require(fault->InitializePersistentTrajectory() ==
                     scalar::ScalarCudaStatus::ready,
                 "fault campaign initialization"))
      return 1;
    for (int step = 1; step <= 6; ++step)
    {
      if (!require(CudaAccepted(fault->AdvancePersistentTrajectory()),
                   "fault campaign prefix"))
        return 1;
    }
    cpu::PersistentRank2TrajectoryState fault_before{};
    if (!require(fault->CopyPublishedTrajectoryState(fault_before),
                 "fault pre-state capture"))
      return 1;
    const int fault_before_accepted_steps =
        fault->trajectory_evidence().accepted_steps;
    const scalar::ScalarCudaStatus fault_result =
        fault->AdvancePersistentTrajectory(campaign.injection);
    if (!require(fault_result ==
                     scalar::ScalarCudaStatus::accepted_after_one_retry,
                 "rank1-origin collective rollback and same-dt retry"))
      return 1;
    cpu::PersistentRank2TrajectoryState fault_after{};
    if (!require(fault->CopyPublishedTrajectoryState(fault_after),
                 "fault post-retry state capture"))
      return 1;
    const scalar::ScalarCudaEvidence& fault_evidence =
        fault->trajectory_evidence();
    const auto& failed = fault_evidence.attempts[6];
    const auto& retry = fault_evidence.attempts[7];
    const auto& accepted_step = fault_evidence.steps[6];
    const bool local_origin_identity = rank == 1
        ? failed.injection == campaign.injection
        : failed.injection == scalar::FailureInjection::none;
    const int local_origin = failed.injection == campaign.injection ? 1 : 0;
    std::array<int, 2> origin_by_rank{{-1, -1}};
    const bool origin_gathered = MPI_Allgather(
        &local_origin, 1, MPI_INT, origin_by_rank.data(), 1, MPI_INT,
        MPI_COMM_WORLD) == MPI_SUCCESS;
    const bool limit_counts =
        campaign.injection != scalar::FailureInjection::newton_iteration_limit
        ? failed.attempted_newton_iterations == 1
        : failed.attempted_newton_iterations ==
              scalar::newton_maximum_iterations &&
              failed.newton_iterations == 0;
    const bool retry_provenance =
        local_origin_identity && origin_gathered &&
        origin_by_rank[0] == 0 && origin_by_rank[1] == 1 && limit_counts &&
        fault_before_accepted_steps == 6 &&
        fault_evidence.accepted_steps == 7 &&
        fault_evidence.attempted_steps == 8 &&
        fault_evidence.retry_attempts == 1 &&
        fault_evidence.first_failure == campaign.failure_status &&
        fault_evidence.attempt_count == 8 &&
        failed.status == campaign.failure_status &&
        failed.failure_cause == campaign.failure_cause &&
        failed.target_step == 7 && failed.retry_index == 0 &&
        retry.status == scalar::ScalarCudaStatus::accepted_after_one_retry &&
        retry.injection == scalar::FailureInjection::none &&
        retry.target_step == 7 && retry.retry_index == 1 &&
        failed.dt == scalar::fixed_dt && retry.dt == scalar::fixed_dt &&
        accepted_step.accepted && accepted_step.step == 7 &&
        accepted_step.attempt_index == 7 &&
        oracle::SameHashes(failed.entry_hashes, retry.entry_hashes) &&
        oracle::SameHashes(retry.candidate_hashes,
                           accepted_step.accepted_hashes) &&
        !oracle::SameHashes(failed.candidate_hashes,
                            accepted_step.accepted_hashes) &&
        fault_evidence.coordinates.step == 7 &&
        fault_evidence.coordinates.generation == 7 &&
        fault_evidence.coordinates.cursor == 7 &&
        fault_evidence.coordinates.dt == scalar::fixed_dt;
    if (!require(oracle::SameStateBytes(fault_before,
                                        cuda_baseline_states[6]) &&
                     oracle::SameStateBytes(fault_after,
                                            cuda_baseline_states[7]),
                 "failure rollback and retry preserve baseline bytes") ||
        !require(retry_provenance,
                 "exact four-campaign failure/retry provenance") ||
        !require(oracle::AcceptedAttemptResidualBounds(retry),
                 "retry admissibility, conservation, Newton and true-GMRES"))
      return 1;
  }

  const auto reject = [&](scalar::ScalarCudaCheckpoint candidate,
                          const scalar::ScalarCudaStatus expected,
                          const std::string_view message)
  {
    auto owner = CreateCuda();
    if (!require(owner != nullptr, "negative owner creation")) return;
    const scalar::ScalarCudaStatus initialized =
        owner->InitializePersistentTrajectory();
    if (!require(initialized == scalar::ScalarCudaStatus::ready,
                 "negative owner initialization"))
      return;
    cpu::PersistentRank2TrajectoryState state_before{};
    const bool before_copied =
        owner->CopyPublishedTrajectoryState(state_before);
    if (!require(before_copied, "negative pre-state capture")) return;
    *negative_evidence_before = owner->trajectory_evidence();
    const scalar::ScalarCudaStatus rejected =
        owner->RestoreTrajectoryCheckpoint(candidate);
    if (!require(rejected == expected, message)) return;
    cpu::PersistentRank2TrajectoryState state_after{};
    const bool after_copied =
        owner->CopyPublishedTrajectoryState(state_after);
    if (!require(after_copied, "negative post-state capture")) return;
    const scalar::ScalarCudaEvidence& evidence_after =
        owner->trajectory_evidence();
    require(oracle::SameStateBytes(state_before, state_after) &&
                oracle::SameLogicalEvidence(*negative_evidence_before,
                                            evidence_after),
            "corrupt checkpoint preserves state and logical evidence");
  };

  scalar::ScalarCudaCheckpoint identity = cuda_checkpoint10;
  if (rank == 1) { identity.shard_rank = 0; }
  reject(identity, scalar::ScalarCudaStatus::restart_identity_failure,
         "rank-local identity negative");
  scalar::ScalarCudaCheckpoint bytes = cuda_checkpoint10;
  bytes.interior_component_slabs[0] += 1.0;
  reject(bytes, scalar::ScalarCudaStatus::restart_hash_failure,
         "raw-byte hash negative");
  scalar::ScalarCudaCheckpoint manifest = cuda_checkpoint10;
  manifest.canonical_manifest_hash.bytes[0] ^= 0x1U;
  reject(manifest, scalar::ScalarCudaStatus::restart_hash_failure,
         "combined-manifest negative");
  scalar::ScalarCudaCheckpoint rho_zero = cuda_checkpoint10;
  rho_zero.interior_component_slabs[0] = 0.0;
  if (!require(oracle::ResealCheckpoint(rho_zero),
               "rho-zero adversarial checkpoint reseal"))
    return 1;
  reject(rho_zero, scalar::ScalarCudaStatus::restart_admissibility_failure,
         "authenticated rho-zero semantic negative");
  scalar::ScalarCudaCheckpoint sigma_shift = cuda_checkpoint10;
  constexpr std::size_t sigma_component_slab =
      scalar::state_components * scalar::persistent_elements *
      scalar::volume_nodes;
  sigma_shift.interior_component_slabs[sigma_component_slab] += 1.0;
  if (!require(oracle::ResealCheckpoint(sigma_shift),
               "Sigma-shift adversarial checkpoint reseal"))
    return 1;
  reject(sigma_shift,
         scalar::ScalarCudaStatus::restart_auxiliary_residual_failure,
         "authenticated Sigma+1 auxiliary-residual negative");

  int global_failures = 0;
  const int terminal_collective = MPI_Allreduce(
      &failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (terminal_collective != MPI_SUCCESS)
  {
    std::cerr << "C3: terminal failure reduction failed rank=" << rank
              << '\n';
    return 1;
  }
  return global_failures == 0 ? 0 : 1;
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  std::string directory_path;
  std::string run_id;
  std::string complete_sha256;
  bool local = argc == 1 && ranks == 2 &&
      ReadEnvironment("DG_IGR_SC25_R2_HANDOFF_DIR", 4096, directory_path) &&
      ReadEnvironment("DG_IGR_SC25_R2_RUN_ID", 128, run_id) &&
      ReadEnvironment("DG_IGR_SC25_R2_COMPLETE_SHA256", 64,
                      complete_sha256) &&
      manifest::SafeRunId(run_id);
  manifest::Digest complete_digest{};
  local = local &&
      manifest::ParseHexDigest(complete_sha256, complete_digest);
  if (!CollectiveAll(local))
  {
    if (rank == 0)
      std::cerr << "CUDA C3 requires mpiexec -n 2 and exact handoff "
                   "environment\n";
    return 1;
  }

  std::string environment_identity = directory_path;
  environment_identity.push_back('\0');
  environment_identity.append(run_id);
  environment_identity.push_back('\0');
  environment_identity.append(complete_sha256);
  const manifest::Digest environment_digest =
      ::dg::hdg::runtime::ComputeSha256(
          environment_identity.data(), environment_identity.size());
  local = SameAcrossRanks(environment_digest);

  manifest::DirectoryHandle directory;
  local = directory.Open(directory_path) && local;
  const std::array<std::uint64_t, 2> local_directory{{
      directory.identity().device, directory.identity().inode}};
  std::array<std::uint64_t, 4> gathered_directories{{0, 0, 0, 0}};
  const bool directory_gathered = MPI_Allgather(
      local_directory.data(), 2, MPI_UINT64_T, gathered_directories.data(),
      2, MPI_UINT64_T, MPI_COMM_WORLD) == MPI_SUCCESS;
  local = local && directory_gathered &&
      gathered_directories[0] == gathered_directories[2] &&
      gathered_directories[1] == gathered_directories[3] &&
      gathered_directories[1] != 0;
  manifest::ManifestSetIdentity manifest_before{};
  local = local &&
      manifest::CaptureManifestSetIdentity(directory, manifest_before) &&
      manifest::DigestEqual(manifest_before.files[2].sha256,
                            complete_digest);
  if (!CollectiveAll(local))
  {
    std::cerr << "CUDA C3 rejected environment or exact immutable manifest "
                 "set rank=" << rank << '\n';
    return 1;
  }

  // COMPLETE and this rank's shard are authenticated before CUDA or HYPRE
  // policy is selected in this fresh consumer process.
  manifest::CpuReferenceManifest cpu_reference{};
  std::string producer_executable_sha256;
  if (!LoadCpuReferenceManifest(directory, run_id, complete_sha256,
                                cpu_reference,
                                producer_executable_sha256))
    return 1;

  std::unique_ptr<mfem::Device> device;
  HypreAdmission cuda_hypre{};
  bool local_admission = false;
  try
  {
    device = std::make_unique<mfem::Device>("cuda");
    cuda_hypre = InitializeHypre();
    local_admission = cuda_hypre.valid_cuda();
  }
  catch (...)
  {
    local_admission = false;
  }
  if (!CollectiveAll(local_admission))
  {
    EmitHypreAdmission(rank, cuda_hypre);
    return 1;
  }

  const int result = RunC3(cpu_reference);
  const HypreAdmission post_cuda = CaptureHypreAdmission();
  const bool cuda_policy_stable = CollectiveAll(post_cuda.valid_cuda());
  if (!cuda_policy_stable) EmitHypreAdmission(rank, post_cuda);

  manifest::ManifestSetIdentity manifest_after{};
  const bool immutable =
      manifest::CaptureManifestSetIdentity(directory, manifest_after) &&
      manifest_after == manifest_before;
  const bool immutable_collective = CollectiveAll(immutable);
  if (!immutable_collective)
  {
    if (rank == 0)
      std::cerr << "CUDA C3 manifest set changed during consumption\n";
    return 1;
  }
  if (!cuda_policy_stable || result != 0) return 1;

  if (rank == 0)
  {
    std::cout << "SC25_CUDA_RANK2_C3_PASS run_id=" << run_id
              << " complete_sha256=" << complete_sha256
              << " producer_executable_sha256="
              << producer_executable_sha256
              << " ranks=2 elements=64 true_trace=1280 states=21 steps=20 "
                 "host_staged=1 sm=120 manifest_unchanged=1\n";
    std::cout << "SC25_CUDA_RANK2_MANIFEST_CONSUMED run_id=" << run_id
              << " complete_sha256=" << complete_sha256
              << " producer_executable_sha256="
              << producer_executable_sha256 << '\n';
  }
  return 0;
}

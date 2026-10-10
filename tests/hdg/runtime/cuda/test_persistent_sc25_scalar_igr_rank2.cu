#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr_rank2.cuh"

#include <HYPRE_utilities.h>
#include <mfem.hpp>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

#if defined(__CUDA_ARCH__)
static_assert(__CUDA_ARCH__ ==
              dg::hdg::runtime::cuda_w8b::sc25::rank2::required_native_architecture,
              "SC25 rank-two requires the configured native device image");
#endif

namespace
{
namespace scalar = ::dg::hdg::runtime::cuda_w8b::sc25::rank2;

bool CollectiveAll(const bool local) noexcept
{
  int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

bool CudaHypreAdmissionActive() noexcept
{
  HYPRE_MemoryLocation memory = HYPRE_MEMORY_UNDEFINED;
  HYPRE_ExecutionPolicy execution = HYPRE_EXEC_UNDEFINED;
  return HYPRE_Initialized() != 0 && HYPRE_GetMemoryLocation(&memory) == 0 &&
      HYPRE_GetExecutionPolicy(&execution) == 0 &&
      memory == HYPRE_MEMORY_DEVICE && execution == HYPRE_EXEC_DEVICE &&
      mfem::HypreUsingGPU() && HYPRE_GetError() == 0;
}

bool Accepted(const scalar::ScalarCudaStatus status) noexcept
{
  return status == scalar::ScalarCudaStatus::accepted ||
      status == scalar::ScalarCudaStatus::accepted_after_one_retry;
}

bool SameState(const scalar::PublishedState& left,
               const scalar::PublishedState& right) noexcept
{
  return left.u == right.u && left.sigma == right.sigma && left.z == right.z &&
      left.uhat == right.uhat && left.sigmahat == right.sigmahat;
}

bool Require(const bool local, const int rank, const char* const stage,
             const scalar::ScalarCudaStatus status) noexcept
{
  if (!local)
    std::cerr << "SC25_CUDA_RANK2_FAIL rank=" << rank << " stage=" << stage
              << " status=" << static_cast<int>(status) << '\n';
  return CollectiveAll(local);
}

bool CompactEvidenceValid(const scalar::ScalarCudaEvidence& evidence,
                          const bool require_converged_nonlinear = true) noexcept
{
  const auto& residency = evidence.residency;
  const auto& allocations = evidence.allocations;
  return evidence.transfer_totals.complete && evidence.accepted_steps > 0 &&
      evidence.maximum_attempted_newton_iterations > 0 &&
      evidence.maximum_trace_iterations >= 0 &&
      std::isfinite(evidence.minimum_density) &&
      std::isfinite(evidence.minimum_pressure) &&
      std::isfinite(evidence.minimum_speed_margin) &&
      evidence.minimum_density > 0.0 && evidence.minimum_pressure > 0.0 &&
      evidence.minimum_speed_margin > 0.0 &&
      evidence.maximum_conservation_error <= 2.0e-10 &&
      std::isfinite(evidence.maximum_nonlinear_residual_ratio) &&
      (!require_converged_nonlinear ||
       evidence.maximum_nonlinear_residual_ratio <= 1.0) &&
      evidence.maximum_trace_residual_ratio <= 1.0 &&
      evidence.maximum_local_application_residual_ratio <= 1.0 &&
      residency.mfem_cuda_device && residency.hypre_cuda &&
      residency.umpire_device_pool && residency.state_device_resident &&
      residency.topology_device_resident && residency.workspace_device_resident &&
      residency.trace_solve_device_resident && residency.host_staged_rank2 &&
      residency.same_device_uuid && !residency.host_numerical_fallback &&
      allocations.warm_plateau &&
      allocations.zero_post_baseline_observed_growth &&
      allocations.zero_post_baseline_owned_allocation &&
      allocations.zero_post_baseline_owned_deallocation &&
      evidence.launches.fallback_calls == 0;
}

bool CreateInitialized(std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank2>&
                           owner,
                       const scalar::ScalarCudaConfig& config,
                       const int rank, const char* const stage) noexcept
{
  scalar::ScalarCudaStatus status =
      scalar::PersistentSc25ScalarIgrCudaRank2::Create(MPI_COMM_WORLD, config,
                                                        owner);
  if (!Require(status == scalar::ScalarCudaStatus::ready && owner, rank, stage,
               status))
    return false;
  status = owner->InitializePersistentTrajectory();
  return Require(status == scalar::ScalarCudaStatus::ready, rank, stage,
                 status);
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (!Require(ranks == 2, rank, "ranks",
               scalar::ScalarCudaStatus::uninitialized))
    return 1;

  std::unique_ptr<mfem::Device> device;
  bool cuda_hypre_admission = false;
  try
  {
    device = std::make_unique<mfem::Device>("cuda");
    mfem::Hypre::Init();
    cuda_hypre_admission = CudaHypreAdmissionActive();
  }
  catch (...) {}
  if (!Require(cuda_hypre_admission, rank, "cuda-hypre-admission",
               scalar::ScalarCudaStatus::uninitialized))
    return 1;

  scalar::ScalarCudaConfig config{};
  std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank2> baseline;
  if (!CreateInitialized(baseline, config, rank, "baseline-initialize"))
    return 1;

  auto status = baseline->AdvancePersistentTrajectory();
  if (!Require(Accepted(status), rank, "baseline-step-1", status)) return 1;
  scalar::ScalarCudaCheckpoint checkpoint{};
  const bool captured = baseline->CaptureTrajectoryCheckpoint(checkpoint) &&
      checkpoint.descriptor.successful_commit &&
      checkpoint.descriptor.coordinates.step == 1 &&
      checkpoint.descriptor.coordinates.generation == 1 &&
      checkpoint.shard_manifest_hash != ::dg::hdg::runtime::StateHash{} &&
      checkpoint.canonical_manifest_hash != ::dg::hdg::runtime::StateHash{};
  if (!Require(captured, rank, "checkpoint-capture", baseline->status()))
    return 1;

  status = baseline->AdvancePersistentTrajectory();
  if (!Require(Accepted(status), rank, "baseline-step-2", status)) return 1;
  scalar::PublishedState uninterrupted{};
  const auto& baseline_evidence = baseline->trajectory_evidence();
  if (!Require(baseline->CopyPublishedTrajectoryState(uninterrupted) &&
                   baseline_evidence.accepted_steps == 2 &&
                   baseline_evidence.attempted_steps == 2 &&
                   baseline_evidence.rejected_steps == 0 &&
                   baseline_evidence.retry_attempts == 0 &&
                   CompactEvidenceValid(baseline_evidence),
               rank, "baseline-evidence", baseline->status()))
    return 1;

  scalar::ScalarCudaCheckpoint corrupt = checkpoint;
  corrupt.canonical_manifest_hash.bytes[0] ^= std::uint8_t{1};
  std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank2> rejected_restore;
  if (!CreateInitialized(rejected_restore, config, rank, "corrupt-initialize"))
    return 1;
  status = rejected_restore->RestoreTrajectoryCheckpoint(corrupt);
  if (!Require(status == scalar::ScalarCudaStatus::restart_hash_failure,
               rank, "corrupt-rejected", status))
    return 1;
  rejected_restore.reset();

  std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank2> restarted;
  if (!CreateInitialized(restarted, config, rank, "restart-initialize"))
    return 1;
  status = restarted->RestoreTrajectoryCheckpoint(checkpoint);
  if (!Require(status == scalar::ScalarCudaStatus::ready, rank,
               "checkpoint-restore", status))
    return 1;
  status = restarted->AdvancePersistentTrajectory();
  if (!Require(Accepted(status), rank, "restart-step-2", status)) return 1;
  scalar::PublishedState continued{};
  const auto& restart_evidence = restarted->trajectory_evidence();
  if (!Require(restarted->CopyPublishedTrajectoryState(continued) &&
                   SameState(uninterrupted, continued) &&
                   restart_evidence.accepted_steps == 2 &&
                   restart_evidence.attempted_steps == 1 &&
                   restart_evidence.rejected_steps == 0 &&
                   CompactEvidenceValid(restart_evidence),
               rank, "restart-continuation", restarted->status()))
    return 1;
  restarted.reset();

  std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank2> retry;
  if (!CreateInitialized(retry, config, rank, "retry-initialize")) return 1;
  status = retry->AdvancePersistentTrajectory(scalar::FailureInjection::local_factor);
  const auto& retry_evidence = retry->trajectory_evidence();
  if (!Require(status == scalar::ScalarCudaStatus::accepted_after_one_retry &&
                   retry_evidence.accepted_steps == 1 &&
                   retry_evidence.attempted_steps == 2 &&
                   retry_evidence.rejected_steps == 1 &&
                   retry_evidence.retry_attempts == 1 &&
                   retry_evidence.first_failure ==
                       scalar::ScalarCudaStatus::injected_local_factor_failure &&
                   retry_evidence.maximum_nonlinear_residual_ratio > 1.0 &&
                   CompactEvidenceValid(retry_evidence, false),
               rank, "retry-aggregate", status))
    return 1;

  if (rank == 0)
    std::cout << "SC25_CUDA_RANK2_PASS ranks=2 compact-evidence=1 "
                 "checkpoint-restart=1 retry=1\n";
  return 0;
}

#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr_rank2.cuh"

#include <mfem.hpp>
#include <mpi.h>

#include <iostream>
#include <memory>

namespace
{

namespace scalar = ::dg::hdg::runtime::cuda_w8b::sc25::rank2;

bool Accepted(const scalar::ScalarCudaStatus status) noexcept
{
  return status == scalar::ScalarCudaStatus::accepted ||
      status == scalar::ScalarCudaStatus::accepted_after_one_retry;
}

bool CollectiveAll(const bool local) noexcept
{
  const int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

bool SameState(const scalar::PublishedState& left,
               const scalar::PublishedState& right) noexcept
{
  return left.u == right.u && left.sigma == right.sigma && left.z == right.z &&
      left.uhat == right.uhat && left.sigmahat == right.sigmahat;
}

bool RequireCollective(const bool local, const int rank, const char* stage,
                       const scalar::ScalarCudaStatus status)
{
  if (!local)
    std::cerr << "SC25_CUDA_RANK2_W8C_SMOOTH_FAIL rank=" << rank
              << " stage=" << stage
              << " status=" << static_cast<int>(status) << '\n';
  return CollectiveAll(local);
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (!RequireCollective(ranks == 2, rank, "ranks",
                         scalar::ScalarCudaStatus::uninitialized))
    return 1;

  mfem::Device device("cuda");
  scalar::ScalarCudaConfig config{};
  config.lifecycle = scalar::ScalarCudaLifecycle::w8c_smooth_transition_v2;

  std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank2> owner;
  auto status = scalar::PersistentSc25ScalarIgrCudaRank2::Create(
      MPI_COMM_WORLD, config, owner);
  bool ready = status == scalar::ScalarCudaStatus::ready && owner;
  if (!RequireCollective(ready, rank, "create", status)) return 1;
  status = owner->InitializePersistentTrajectory();
  ready = status == scalar::ScalarCudaStatus::ready;
  if (!ready)
  {
    const auto& evidence = owner->trajectory_evidence();
    std::cerr << "SC25_CUDA_RANK2_W8C_SMOOTH_INIT_EVIDENCE rank=" << rank
              << " trace_iterations=" << evidence.maximum_trace_iterations
              << " trace_ratio=" << evidence.maximum_trace_residual_ratio
              << '\n';
  }
  if (!RequireCollective(ready, rank, "initialize", status)) return 1;

  scalar::ScalarCudaCheckpoint checkpoint{};
  for (int step = 1; step <= scalar::smooth_transition_required_steps; ++step)
  {
    status = owner->AdvancePersistentTrajectory();
    ready = Accepted(status);
    if (!RequireCollective(ready, rank, "advance", status)) return 1;
    if (step == 32)
    {
      ready = owner->CaptureTrajectoryCheckpoint(checkpoint) &&
          checkpoint.scenario == ::dg::hdg::runtime::StableIdentity::Copy(
              scalar::cuda_rank2_smooth_transition_fixture_id) &&
          checkpoint.descriptor.coordinates.step == 32 &&
          checkpoint.descriptor.coordinates.dt == scalar::smooth_transition_dt;
      if (!RequireCollective(ready, rank, "checkpoint", owner->status()))
        return 1;
    }
  }
  scalar::PublishedState uninterrupted{};
  ready = owner->CopyPublishedTrajectoryState(uninterrupted);
  const auto& evidence = owner->trajectory_evidence();
  ready = ready && evidence.coordinates.step ==
          static_cast<std::uint64_t>(scalar::smooth_transition_required_steps) &&
      evidence.coordinates.dt == scalar::smooth_transition_dt &&
      evidence.accepted_steps == scalar::smooth_transition_required_steps &&
      evidence.residency.host_staged_rank2 && evidence.residency.same_device_uuid &&
      evidence.launches.fallback_calls == 0;
  if (!RequireCollective(ready, rank, "uninterrupted", owner->status()))
    return 1;
  owner.reset();

  std::unique_ptr<scalar::PersistentSc25ScalarIgrCudaRank2> restarted;
  status = scalar::PersistentSc25ScalarIgrCudaRank2::Create(
      MPI_COMM_WORLD, config, restarted);
  ready = status == scalar::ScalarCudaStatus::ready && restarted;
  if (!RequireCollective(ready, rank, "restart-create", status)) return 1;
  status = restarted->InitializePersistentTrajectory();
  ready = status == scalar::ScalarCudaStatus::ready;
  if (!RequireCollective(ready, rank, "restart-initialize", status))
    return 1;
  status = restarted->RestoreTrajectoryCheckpoint(checkpoint);
  ready = status == scalar::ScalarCudaStatus::ready;
  if (!RequireCollective(ready, rank, "restore", status)) return 1;
  for (int step = 33; step <= scalar::smooth_transition_required_steps; ++step)
  {
    status = restarted->AdvancePersistentTrajectory();
    ready = Accepted(status);
    if (!RequireCollective(ready, rank, "restart-advance", status))
      return 1;
  }
  scalar::PublishedState restored{};
  ready = restarted->CopyPublishedTrajectoryState(restored) &&
      SameState(uninterrupted, restored);
  if (!RequireCollective(ready, rank, "state-compare", restarted->status()))
    return 1;

  if (rank == 0)
    std::cout << "SC25_CUDA_RANK2_W8C_SMOOTH_PASS ranks=2 steps=64 "
                 "host_staged=1 checkpoint_restart=1\n";
  return 0;
}

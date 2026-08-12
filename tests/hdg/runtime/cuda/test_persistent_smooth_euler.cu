#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include "persistent_smooth_euler_oracle.cuh"

#include <HYPRE_utilities.h>
#include <_hypre_utilities.h>
#include <mfem.hpp>
#include <umpire/ResourceManager.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>


namespace
{
using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::cuda_w8a;

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
    UmpireDeviceAllocatorPolicy, CudaTransportPolicy,
    CudaMpiCollectivePolicy>;

bool All(const bool local) noexcept
{
  int value = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && result == 1;
}

template <typename Backend>
bool RunCompactStep(const int ranks, const bool token_negative,
                    const int steps) noexcept
{
  CudaConfig config{};
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  auto engine = std::make_unique<Engine<Backend>>(config);
  CudaRunEvidence setup{};
  if (!engine->Initialize(setup)) { return false; }
  const bool setup_valid = setup.status == CudaRunStatus::ready &&
      setup.ranks == ranks && setup.mfem_device_valid &&
      setup.hypre_cuda_umpire_device && setup.same_physical_gpu &&
      setup.no_host_solver_fallback && setup.reuse.persistent_graph &&
      setup.reuse.persistent_slots && setup.reuse.persistent_workspace &&
      (ranks == 1 || setup.host_staged_rank2);
  if (!All(setup_valid)) { return false; }

  for (int step = 0; step < steps; ++step)
  {
    CudaPrecommitSnapshot snapshot{};
    CudaStepEvidence evidence{};
    CudaSerializedCheckpointV1 checkpoint{};
    if (!engine->PrepareStep(false, snapshot, evidence) ||
        !snapshot.numerical_candidate_ready ||
        !engine->StagePreparedCheckpoint(checkpoint, evidence))
      return false;
    const CudaTransferLedgerRangeV1 staged = checkpoint.staging_range;
    const bool sealed = staged.begin_ordinal != 0 &&
        staged.end_ordinal >= staged.begin_ordinal && staged.complete &&
        staged.ordered_digest != 0 && evidence.transfer_range.complete &&
        evidence.transfers.complete && evidence.transfers.event_count != 0;
    if (!All(sealed)) { return false; }
    if (token_negative)
    {
      CudaSerializedCheckpointV1 forged = checkpoint;
      ++forged.attempt.sequence;
      const bool rejected = !engine->MarkOutputStaged(forged);
      const bool retriable = engine->AbortPrepared(
          checkpoint, FailureReason::backend_failure, evidence);
      if (!All(rejected && retriable && !checkpoint.successful_commit))
        return false;
      continue;
    }
    if (!engine->MarkOutputStaged(checkpoint) ||
        !engine->PrepareCommit(checkpoint, evidence) ||
        !engine->CommitPrepared(checkpoint, evidence) ||
        !engine->FinalizeCommit(checkpoint, evidence))
      return false;
    const CudaTransferLedgerV1 ledger = CudaTransportPolicy::Ledger();
    const bool committed = checkpoint.successful_commit &&
        evidence.committed && evidence.after.step ==
            static_cast<std::uint64_t>(step + 1) &&
        ledger.observed_count >= evidence.transfer_range.end_ordinal &&
        ledger.totals.complete && !ledger.overflow &&
        sizeof(CudaTransferLedgerV1) < 128 &&
        sizeof(CudaAllocationLedgerV1) < 64;
    if (!All(committed)) { return false; }
  }
  return true;
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  (void)device;
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  const std::string_view mode = argc == 2 ? argv[1] : "";
  const bool tokens = mode == "--token-negatives";
  const int steps = mode == "--functional" ? 2 : 1;
  const bool known_mode = mode.empty() || tokens || mode == "--functional";
  const bool pass = known_mode && (ranks == 1
      ? RunCompactStep<RankOneTag>(ranks, tokens, steps)
      : ranks == 2 ? RunCompactStep<RankTwoTag>(ranks, tokens, steps)
                   : false);
  if (!pass && ranks == 0)
    std::cerr << "compact CUDA W8A audit failed\n";
  return pass ? 0 : 1;
}

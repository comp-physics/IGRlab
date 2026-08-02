#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
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

namespace
{
namespace cpu = ::dg::hdg::runtime::cpu::sc25;
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

bool CpuAccepted(const cpu::TrajectoryStatus status) noexcept
{
  return status == cpu::TrajectoryStatus::HdgSc25Accepted ||
      status == cpu::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry;
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

  bool valid_host() const noexcept
  {
    return initialized != 0 && error == 0 && memory_query == 0 &&
        execution_query == 0 && memory == HYPRE_MEMORY_HOST &&
        execution == HYPRE_EXEC_HOST && !using_gpu;
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

void EmitHypreAdmission(const int rank, const HypreAdmission& value)
{
  std::cerr << "SC25_CPU_RANK2_HYPRE rank=" << rank
            << " initialized=" << value.initialized
            << " error=" << value.error
            << " memory_query=" << value.memory_query
            << " memory=" << static_cast<int>(value.memory)
            << " execution_query=" << value.execution_query
            << " execution=" << static_cast<int>(value.execution)
            << " using_gpu=" << (value.using_gpu ? 1 : 0) << '\n';
}

std::unique_ptr<cpu::PersistentSc25ScalarIgrCpuFoundation> CreateCpu()
{
  cpu::FoundationConfig config{};
  config.polynomial_degree = 1;
  config.elements_per_direction = cpu::persistent_elements_per_direction;
  std::unique_ptr<cpu::PersistentSc25ScalarIgrCpuFoundation> owner;
  if (cpu::PersistentSc25ScalarIgrCpuFoundation::Create(
          MPI_COMM_WORLD, cpu::CpuMpiRank2Tag{}, config, owner) !=
      cpu::FoundationStatus::ready)
    return {};
  return owner;
}

int EmitCpuReferenceManifest(const manifest::DirectoryHandle& directory,
                             const std::string& run_id,
                             const std::string& executable_sha256)
{
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto require = [&](const bool local, const std::string_view message)
  {
    if (!CollectiveAll(local))
    {
      if (rank == 0) std::cerr << "CPU manifest: " << message << '\n';
      return false;
    }
    return true;
  };

  if (!require(manifest::SafeRunId(run_id), "safe run identity") ||
      !require(executable_sha256.size() == 64,
               "live executable SHA-256 identity") ||
      !require(manifest::accepted_cpu_oracle_receipt.find(
                   "evidence_sha256=") != std::string_view::npos,
               "accepted CPU receipt binding"))
    return 1;

  manifest::CpuReferenceManifest reference{};
  reference.run_id = run_id;
  reference.executable_sha256 = executable_sha256;
  reference.directory = directory.identity();
  reference.rank = rank;

  auto cpu_owner = CreateCpu();
  if (!require(cpu_owner != nullptr, "CPU rank-two creation")) return 1;
  const cpu::TrajectoryStatus initialized =
      cpu_owner->InitializePersistentRank2Trajectory();
  if (!require(initialized == cpu::TrajectoryStatus::ready,
               "CPU rank-two initialization"))
    return 1;
  const HypreAdmission initialized_hypre = CaptureHypreAdmission();
  if (!require(initialized_hypre.valid_host(),
               "host HYPRE after initialization"))
  {
    EmitHypreAdmission(rank, initialized_hypre);
    return 1;
  }

  reference.census = cpu_owner->rank2_trajectory_evidence().census;
  if (!require(manifest::ExpectedCensus(reference.census, rank),
               "exact CPU rank-two ownership census"))
    return 1;
  bool copied = cpu_owner->CopyPublishedRank2TrajectoryState(
      reference.steps[0].state);
  reference.steps[0].summary = manifest::CaptureCpuStepSummary(
      cpu_owner->rank2_trajectory_evidence(), 0);
  if (!require(copied, "CPU step-zero state copy")) return 1;

  cpu::PersistentRank2TrajectoryCheckpoint checkpoint10{};
  for (int step = 1; step <= cpu::persistent_required_steps; ++step)
  {
    const cpu::TrajectoryStatus advanced =
        cpu_owner->AdvancePersistentRank2Trajectory();
    if (!require(CpuAccepted(advanced), "CPU reference trajectory")) return 1;
    copied = cpu_owner->CopyPublishedRank2TrajectoryState(
        reference.steps[static_cast<std::size_t>(step)].state);
    reference.steps[static_cast<std::size_t>(step)].summary =
        manifest::CaptureCpuStepSummary(
            cpu_owner->rank2_trajectory_evidence(), step);
    if (!require(copied, "CPU reference state copy")) return 1;
    if (step == 10)
    {
      const bool captured =
          cpu_owner->CaptureRank2TrajectoryCheckpoint(checkpoint10);
      if (!require(captured, "CPU step-10 checkpoint capture")) return 1;
    }
  }

  const auto& evidence = cpu_owner->rank2_trajectory_evidence();
  const bool policy = evidence.accepted_steps == 20 &&
      evidence.attempted_steps == 20 && evidence.retry_attempts == 0 &&
      evidence.attempt_count == 20 && evidence.step_evidence_count == 20 &&
      evidence.trace_restart_dimension == 50 &&
      evidence.trace_maximum_iterations == 200 && evidence.ilu_type == 0 &&
      evidence.ilu_level_of_fill == 0 && evidence.ilu_local_reordering == 1;
  if (!require(policy, "accepted CPU trajectory and solver policy")) return 1;
  bool snapshots_captured = true;
  for (int step = 0; step <= cpu::persistent_required_steps; ++step)
  {
    snapshots_captured = manifest::CaptureCpuOperatorSnapshot(
        evidence, step, reference.steps[static_cast<std::size_t>(step)]) &&
        snapshots_captured;
  }
  if (!require(snapshots_captured,
               "initialization and twenty deterministic CPU operator snapshots"))
    return 1;
  if (!require(manifest::SameStateBytes(
                   checkpoint10.state, reference.steps[10].state),
               "CPU step-10 checkpoint state bytes"))
    return 1;
  reference.checkpoint_coordinates = checkpoint10.descriptor.coordinates;
  reference.checkpoint_hashes = checkpoint10.raw_field_hashes;
  reference.checkpoint_manifest_hash = checkpoint10.canonical_manifest_hash;
  if (!require(manifest::ValidCpuManifest(reference),
               "complete CPU manifest scientific/configuration contract"))
    return 1;

  cpu_owner.reset();
  const HypreAdmission post_cpu = CaptureHypreAdmission();
  if (!require(post_cpu.valid_host(), "host HYPRE after CPU run"))
  {
    EmitHypreAdmission(rank, post_cpu);
    return 1;
  }

  const std::vector<std::uint8_t> encoded =
      manifest::EncodeCpuManifest(reference);
  const std::string filename = manifest::RankFilename(rank);
  const std::string temporary = manifest::RankTemporaryFilename(rank);
  const bool fresh_paths = manifest::EntryAbsent(directory, filename) &&
      manifest::EntryAbsent(directory, temporary);
  if (!require(fresh_paths, "fresh per-rank manifest paths")) return 1;
  const bool protocol_negatives = !encoded.empty() &&
      manifest::CpuManifestProtocolNegatives(encoded, reference);
  if (!require(protocol_negatives,
               "checksum/truncation/trailing/wrong-token shard negatives"))
    return 1;
  const bool local_write =
      manifest::WriteFileAt(directory, temporary, encoded);
  const bool stale_temp_rejected =
      !manifest::WriteFileAt(directory, temporary, encoded);
  if (!require(local_write && stale_temp_rejected,
               "exclusive temp write/readback and stale rejection"))
    return 1;
  if (!require(manifest::AtomicRenameAt(directory, temporary, filename),
               "atomic per-rank manifest publication"))
    return 1;

  const manifest::Digest local_digest =
      ::dg::hdg::runtime::ComputeSha256(encoded.data(), encoded.size());
  const std::uint64_t local_size = static_cast<std::uint64_t>(encoded.size());
  std::array<manifest::Digest, 2> digests{};
  std::array<std::uint64_t, 2> sizes{{0, 0}};
  const bool digest_gathered = MPI_Allgather(
      local_digest.data(), static_cast<int>(local_digest.size()), MPI_BYTE,
      digests.data(), static_cast<int>(local_digest.size()), MPI_BYTE,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool size_gathered = MPI_Allgather(
      &local_size, 1, MPI_UINT64_T, sizes.data(), 1, MPI_UINT64_T,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!require(digest_gathered && size_gathered,
               "fixed shard digest/size all-gather"))
    return 1;

  std::vector<std::uint8_t> complete_bytes;
  bool complete_written = true;
  if (rank == 0)
  {
    manifest::CompleteManifest complete{};
    complete.run_id = run_id;
    complete.executable_sha256 = executable_sha256;
    complete.directory = directory.identity();
    for (int item = 0; item < 2; ++item)
    {
      complete.shards[static_cast<std::size_t>(item)].filename =
          manifest::RankFilename(item);
      complete.shards[static_cast<std::size_t>(item)].bytes =
          sizes[static_cast<std::size_t>(item)];
      complete.shards[static_cast<std::size_t>(item)].sha256 =
          digests[static_cast<std::size_t>(item)];
    }
    complete_bytes = manifest::EncodeComplete(complete);
    complete_written =
        manifest::EntryAbsent(directory, manifest::complete_filename) &&
        manifest::EntryAbsent(
            directory, manifest::complete_temporary_filename) &&
        !complete_bytes.empty() &&
        manifest::CompleteProtocolNegatives(complete_bytes, complete) &&
        manifest::WriteFileAt(
            directory, manifest::complete_temporary_filename,
            complete_bytes) &&
        manifest::AtomicRenameAt(
            directory, manifest::complete_temporary_filename,
            manifest::complete_filename);
  }
  if (!require(rank != 0 || complete_written,
               "atomic COMPLETE manifest publication and negatives"))
    return 1;

  manifest::Digest complete_digest{};
  if (rank == 0)
    complete_digest = ::dg::hdg::runtime::ComputeSha256(
        complete_bytes.data(), complete_bytes.size());
  const bool broadcast = MPI_Bcast(
      complete_digest.data(), static_cast<int>(complete_digest.size()),
      MPI_BYTE, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!require(broadcast, "COMPLETE digest broadcast")) return 1;

  manifest::ManifestSetIdentity published{};
  std::vector<std::uint8_t> observed_complete_bytes;
  manifest::CompleteManifest observed_complete{};
  manifest::CpuReferenceManifest observed_shard{};
  bool validated = manifest::CaptureManifestSetIdentity(directory, published) &&
      manifest::ReadFileAt(directory, manifest::complete_filename,
                           observed_complete_bytes);
  const manifest::Digest observed_complete_digest = validated
      ? ::dg::hdg::runtime::ComputeSha256(observed_complete_bytes.data(),
                                          observed_complete_bytes.size())
      : manifest::Digest{};
  validated = validated &&
      manifest::DigestEqual(observed_complete_digest, complete_digest) &&
      manifest::DecodeComplete(observed_complete_bytes, run_id,
                               executable_sha256, directory.identity(),
                               observed_complete);
  std::vector<std::uint8_t> observed_shard_bytes;
  const std::size_t rank_index = static_cast<std::size_t>(rank);
  if (validated)
    validated = manifest::ReadFileAt(
                    directory, observed_complete.shards[rank_index].filename,
                    observed_shard_bytes) &&
        observed_shard_bytes.size() ==
            observed_complete.shards[rank_index].bytes;
  const manifest::Digest observed_shard_digest = validated
      ? ::dg::hdg::runtime::ComputeSha256(observed_shard_bytes.data(),
                                          observed_shard_bytes.size())
      : manifest::Digest{};
  validated = validated &&
      manifest::DigestEqual(observed_shard_digest,
                            observed_complete.shards[rank_index].sha256) &&
      manifest::DecodeCpuManifest(
          observed_shard_bytes, run_id, executable_sha256, rank,
          directory.identity(), observed_shard) &&
      manifest::ValidCpuManifest(observed_shard);
  if (!require(validated, "collective final manifest-set authentication"))
    return 1;

  manifest::Digest final_executable_digest{};
  const bool stable_executable =
      manifest::ReadLiveExecutableSha256(final_executable_digest) &&
      manifest::Hex(final_executable_digest) == executable_sha256;
  if (!require(stable_executable, "stable live producer executable bytes"))
    return 1;

  if (rank == 0)
  {
    std::cout << "SC25_CPU_RANK2_MANIFEST_PASS run_id=" << run_id
              << " complete_sha256=" << manifest::Hex(complete_digest)
              << " producer_executable_sha256=" << executable_sha256
              << " rank0_sha256=" << manifest::Hex(digests[0])
              << " rank1_sha256=" << manifest::Hex(digests[1])
              << " ranks=2 states=21 steps=20\n";
  }
  return 0;
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
  bool local = argc == 1 && ranks == 2 &&
      ReadEnvironment("DG_IGR_SC25_R2_HANDOFF_DIR", 4096, directory_path) &&
      ReadEnvironment("DG_IGR_SC25_R2_RUN_ID", 128, run_id) &&
      manifest::SafeRunId(run_id);
  if (!CollectiveAll(local))
  {
    if (rank == 0)
      std::cerr << "CPU manifest requires mpiexec -n 2 and exact handoff "
                   "environment\n";
    return 1;
  }

  std::string environment_identity = directory_path;
  environment_identity.push_back('\0');
  environment_identity.append(run_id);
  const manifest::Digest environment_digest =
      ::dg::hdg::runtime::ComputeSha256(
          environment_identity.data(), environment_identity.size());
  local = SameAcrossRanks(environment_digest);

  manifest::Digest executable_digest{};
  const bool executable_read =
      manifest::ReadLiveExecutableSha256(executable_digest);
  const bool executable_shared = SameAcrossRanks(executable_digest);
  local = local && executable_read && executable_shared;
  const std::string executable_sha256 = manifest::Hex(executable_digest);

  manifest::DirectoryHandle directory;
  local = local && directory.Open(directory_path);
  const std::array<std::uint64_t, 2> local_directory{{
      directory.identity().device, directory.identity().inode}};
  std::array<std::uint64_t, 4> gathered_directories{{0, 0, 0, 0}};
  const bool directory_gathered = MPI_Allgather(
      local_directory.data(), 2, MPI_UINT64_T, gathered_directories.data(),
      2, MPI_UINT64_T, MPI_COMM_WORLD) == MPI_SUCCESS;
  local = local && directory_gathered &&
      gathered_directories[0] == gathered_directories[2] &&
      gathered_directories[1] == gathered_directories[3] &&
      gathered_directories[1] != 0 && manifest::EmptyDirectory(directory);
  if (!CollectiveAll(local))
  {
    std::cerr << "CPU manifest rejected environment, executable, or empty "
                 "pinned owner-only directory rank=" << rank << '\n';
    return 1;
  }

  std::unique_ptr<mfem::Device> device;
  HypreAdmission hypre{};
  local = false;
  try
  {
    device = std::make_unique<mfem::Device>("cpu");
    mfem::Hypre::Init();
    hypre = CaptureHypreAdmission();
    local = hypre.valid_host();
  }
  catch (...)
  {
    local = false;
  }
  if (!CollectiveAll(local))
  {
    EmitHypreAdmission(rank, hypre);
    return 1;
  }
  return EmitCpuReferenceManifest(directory, run_id, executable_sha256);
}

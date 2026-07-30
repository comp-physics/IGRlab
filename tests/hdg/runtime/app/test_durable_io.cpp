#include "numerics/hdg/runtime/app/durable_io.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace
{
using namespace dg::hdg::runtime::app;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "durable I/O test must compile as strict C++17");
static_assert(!std::is_copy_constructible<AtomicOutputDirectory>::value,
              "an output transaction is affine");
static_assert(!std::is_copy_constructible<DurableStagedFileV1>::value,
              "a reversible staged file is affine");
static_assert(!std::is_copy_constructible<
                  DurableGenerationTransactionV1>::value,
              "a generation transaction is affine");

bool Check(const bool condition, const char* name)
{
  if (!condition) std::cerr << "FAILED: " << name << '\n';
  return condition;
}

std::string ReadText(const std::filesystem::path& path)
{
  std::vector<std::uint8_t> bytes;
  std::string detail;
  if (ReadDurableFile(path, 4096, bytes, detail) != DurableIoStatus::ok)
    return {};
  return {bytes.begin(), bytes.end()};
}

std::string ReadPlainText(const std::filesystem::path& path)
{
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>{input},
          std::istreambuf_iterator<char>{}};
}

bool StageText(DurableGenerationTransactionV1& transaction,
               const std::string_view name,
               const std::string_view text,
               std::string& detail)
{
  return transaction.Stage(
      name, reinterpret_cast<const std::uint8_t*>(text.data()), text.size(),
      detail) == DurableIoStatus::ok;
}

bool CommitGeneration(const std::filesystem::path& output,
                      const std::filesystem::path& checkpoint,
                      const std::uint64_t generation,
                      const std::string_view suffix,
                      std::string& detail)
{
  DurableGenerationTransactionV1 transaction;
  return transaction.Begin(output, generation, detail) == DurableIoStatus::ok &&
      StageText(transaction, "state.vtk", std::string{"state-"} +
                    std::string{suffix}, detail) &&
      StageText(transaction, "checkpoint.v2", std::string{"checkpoint-"} +
                    std::string{suffix}, detail) &&
      StageText(transaction, "run-metadata.json", std::string{"metadata-"} +
                    std::string{suffix}, detail) &&
      transaction.AddAlias(checkpoint, "checkpoint.v2", detail) ==
          DurableIoStatus::ok &&
      transaction.Publish(detail) == DurableIoStatus::ok;
}

bool ProcessDeathCase(const std::filesystem::path& root,
                      const DurableGenerationCrashBoundaryV1 boundary,
                      const int ordinal,
                      std::string& detail)
{
  const auto case_root = root / ("death-" + std::to_string(ordinal));
  const auto output = case_root / "output";
  const auto checkpoint = case_root / "restart.checkpoint";
  std::error_code error;
  std::filesystem::create_directories(output, error);
  if (error || !CommitGeneration(output, checkpoint, 0, "old", detail))
    return false;

  const pid_t child = ::fork();
  if (child == 0)
  {
    DurableGenerationTransactionV1 transaction;
    bool ready = transaction.Begin(output, 1, detail) == DurableIoStatus::ok &&
        StageText(transaction, "state.vtk", "state-new", detail) &&
        StageText(transaction, "checkpoint.v2", "checkpoint-new", detail) &&
        StageText(transaction, "run-metadata.json", "metadata-new", detail) &&
        transaction.AddAlias(checkpoint, "checkpoint.v2", detail) ==
            DurableIoStatus::ok;
    if (!ready) ::_exit(91);
    if (boundary == DurableGenerationCrashBoundaryV1::none) ::_exit(86);
    transaction.SetCrashBoundaryForTest(boundary);
    (void)transaction.Publish(detail);
    ::_exit(92);
  }
  if (child < 0) return false;
  int status = 0;
  if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 86)
    return false;
  const auto complete = [&](const std::string_view suffix)
  {
    const std::string owned_suffix{suffix};
    return ReadText(output / "state.vtk") == "state-" + owned_suffix &&
        ReadText(checkpoint) == "checkpoint-" + owned_suffix &&
        ReadText(output / "run-metadata.json") ==
            "metadata-" + owned_suffix;
  };
  if (boundary == DurableGenerationCrashBoundaryV1::after_marker_rename)
  {
    // The atomic rename may be visible even though the containing directory
    // was not durably synced. Either complete generation is valid recovery;
    // a partial mix is not.
    return complete("old") || complete("new");
  }
  return complete("old");
}

std::filesystem::path CurrentGeneration(
    const std::filesystem::path& output)
{
  std::error_code error;
  const auto target = std::filesystem::read_symlink(
      output / ".dg-igr-current", error);
  return error ? std::filesystem::path{} :
                 (output / target).lexically_normal();
}

bool WriteText(const std::filesystem::path& path,
               const std::string_view text)
{
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  return output.good();
}

bool ManifestCorruptionCase(const std::filesystem::path& root,
                            const int kind,
                            std::string& detail)
{
  const auto case_root = root / ("manifest-" + std::to_string(kind));
  const auto output = case_root / "output";
  const auto checkpoint = case_root / "restart.checkpoint";
  std::error_code error;
  std::filesystem::create_directories(output, error);
  if (error || !CommitGeneration(output, checkpoint, 0, "old", detail))
    return false;
  const auto generation = CurrentGeneration(output);
  if (generation.empty()) return false;
  const auto manifest = generation / "manifest.v1";
  bool corrupted = false;
  switch (kind)
  {
    case 0:
      // Preserve the recorded length while changing the digest.
      corrupted = WriteText(generation / "state.vtk", "state-bad");
      break;
    case 1:
    {
      std::string text = ReadPlainText(manifest);
      const std::string record = "artifact\tstate.vtk\t9\t";
      const std::size_t position = text.find(record);
      if (position != std::string::npos)
      {
        text.replace(position, record.size(), "artifact\tstate.vtk\t8\t");
        corrupted = WriteText(manifest, text);
      }
      break;
    }
    case 2:
      corrupted = WriteText(manifest, "not-a-generation-manifest\n");
      break;
    case 3:
      corrupted = WriteText(generation / "extra.bin", "extra");
      break;
    case 4:
      corrupted = std::filesystem::remove(
          generation / "state.vtk", error) && !error;
      break;
    case 5:
      corrupted = std::filesystem::remove(manifest, error) && !error;
      break;
    default: return false;
  }
  if (!corrupted) return false;
  std::vector<std::uint8_t> restart;
  const bool restart_rejected = detail::ReadCurrentGenerationArtifact(
      checkpoint, 4096, restart, detail) == DurableIoStatus::verify_failed;
  DurableGenerationTransactionV1 probe;
  return restart_rejected &&
      probe.Begin(output, 1, detail) == DurableIoStatus::verify_failed &&
      !probe.active() && !probe.published();
}

bool RestartAuthorityCase(const std::filesystem::path& root,
                          std::string& detail)
{
  const auto case_root = root / "restart-authority";
  const auto output = case_root / "output";
  const auto checkpoint = case_root / "restart.checkpoint";
  const auto ordinary = case_root / "ordinary.checkpoint";
  const auto stale = case_root / "stale.checkpoint";
  std::error_code error;
  std::filesystem::create_directories(output, error);
  if (error || !CommitGeneration(output, checkpoint, 0, "old", detail))
    return false;
  const auto old_generation = CurrentGeneration(output);
  std::vector<std::uint8_t> bytes;
  const bool old_authorized = detail::ReadCurrentGenerationArtifact(
      checkpoint, 4096, bytes, detail) == DurableIoStatus::ok &&
      std::string{bytes.begin(), bytes.end()} == "checkpoint-old";
  const bool direct_rejected = detail::ReadCurrentGenerationArtifact(
      old_generation / "checkpoint.v2", 4096, bytes, detail) ==
          DurableIoStatus::verify_failed;
  const bool ordinary_written = WriteText(ordinary, "checkpoint-old");
  const bool ordinary_rejected = ordinary_written &&
      detail::ReadCurrentGenerationArtifact(
          ordinary, 4096, bytes, detail) == DurableIoStatus::verify_failed;
  if (!CommitGeneration(output, checkpoint, 1, "new", detail)) return false;
  std::filesystem::create_symlink(
      old_generation / "checkpoint.v2", stale, error);
  const bool stale_rejected = !error &&
      detail::ReadCurrentGenerationArtifact(
          stale, 4096, bytes, detail) == DurableIoStatus::verify_failed;
  const bool current_authorized = detail::ReadCurrentGenerationArtifact(
      checkpoint, 4096, bytes, detail) == DurableIoStatus::ok &&
      std::string{bytes.begin(), bytes.end()} == "checkpoint-new";
  return old_authorized && direct_rejected && ordinary_rejected &&
      stale_rejected && current_authorized;
}

bool PublishedGenerationRevertCase(const std::filesystem::path& root,
                                   std::string& detail)
{
  const auto case_root = root / "published-revert";
  const auto output = case_root / "output";
  const auto checkpoint = case_root / "restart.checkpoint";
  std::error_code error;
  std::filesystem::create_directories(output, error);
  if (error || !CommitGeneration(output, checkpoint, 0, "old", detail))
    return false;
  DurableGenerationTransactionV1 transaction;
  const bool staged =
      transaction.Begin(output, 1, detail) == DurableIoStatus::ok &&
      StageText(transaction, "state.vtk", "state-new", detail) &&
      StageText(transaction, "checkpoint.v2", "checkpoint-new", detail) &&
      StageText(transaction, "run-metadata.json", "metadata-new", detail) &&
      transaction.AddAlias(checkpoint, "checkpoint.v2", detail) ==
          DurableIoStatus::ok;
  if (!staged || transaction.Publish(detail) != DurableIoStatus::ok ||
      !transaction.published() || ReadText(checkpoint) != "checkpoint-new")
    return false;
  if (transaction.Revert(detail) != DurableIoStatus::ok ||
      transaction.published() || transaction.active())
    return false;
  std::vector<std::uint8_t> bytes;
  return ReadText(output / "state.vtk") == "state-old" &&
      ReadText(checkpoint) == "checkpoint-old" &&
      ReadText(output / "run-metadata.json") == "metadata-old" &&
      detail::ReadCurrentGenerationArtifact(
          checkpoint, 4096, bytes, detail) == DurableIoStatus::ok &&
      std::string{bytes.begin(), bytes.end()} == "checkpoint-old";
}

bool FinalSyncFailureCase(const std::filesystem::path& root,
                          std::string& detail)
{
  const auto case_root = root / "final-sync-failure";
  const auto output = case_root / "output";
  const auto checkpoint = case_root / "restart.checkpoint";
  std::error_code error;
  std::filesystem::create_directories(output, error);
  if (error || !CommitGeneration(output, checkpoint, 0, "old", detail))
    return false;

  DurableGenerationTransactionV1 transaction;
  const bool staged =
      transaction.Begin(output, 1, detail) == DurableIoStatus::ok &&
      StageText(transaction, "state.vtk", "state-new", detail) &&
      StageText(transaction, "checkpoint.v2", "checkpoint-new", detail) &&
      StageText(transaction, "run-metadata.json", "metadata-new", detail) &&
      transaction.AddAlias(checkpoint, "checkpoint.v2", detail) ==
          DurableIoStatus::ok;
  if (!staged || ::setenv(
          "DG_IGR_HDG_TEST_FINAL_DIRECTORY_SYNC_FAILURE", "1", 1) != 0)
    return false;
  const DurableIoStatus status = transaction.Publish(detail);
  const int unset_status = ::unsetenv(
      "DG_IGR_HDG_TEST_FINAL_DIRECTORY_SYNC_FAILURE");
  const bool publication_rejected = status == DurableIoStatus::sync_failed &&
      !transaction.published() && transaction.active();
  const bool previous_authoritative =
      ReadText(output / "state.vtk") == "state-old" &&
      ReadText(checkpoint) == "checkpoint-old" &&
      ReadText(output / "run-metadata.json") == "metadata-old";
  const bool reverted = transaction.Revert(detail) == DurableIoStatus::ok;
  DurableGenerationTransactionV1 probe;
  const bool reopen = probe.Begin(output, 2, detail) == DurableIoStatus::ok &&
      probe.Revert(detail) == DurableIoStatus::ok;
  return unset_status == 0 && publication_rejected &&
      previous_authoritative && reverted && reopen;
}
} // namespace

int main()
{
  const std::filesystem::path root = std::filesystem::path{"/tmp"} /
      ("dg-igr-hdg-io-" + std::to_string(static_cast<long long>(::getpid())));
  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::filesystem::create_directory(root, error);
  bool passed = Check(!error, "create_test_parent");
  std::string detail;
  const std::vector<std::uint8_t> payload{0, 1, 2, 3, 0xff};
  const auto artifact = root / "checkpoint.v2";
  passed = Check(PublishDurableFile(artifact, payload.data(), payload.size(),
                                    detail) == DurableIoStatus::ok,
                 "publish_fsync_verify") && passed;
  std::vector<std::uint8_t> read;
  passed = Check(ReadDurableFile(artifact, payload.size(), read, detail) ==
                         DurableIoStatus::ok && read == payload,
                 "read_exact_bytes") && passed;
  passed = Check(PublishDurableFile(artifact, payload.data(), payload.size(),
                                    detail) == DurableIoStatus::already_exists,
                 "never_overwrite_checkpoint") && passed;

  AtomicOutputDirectory output;
  const auto final = root / "output";
  passed = Check(output.Begin(final, detail) == DurableIoStatus::ok &&
                     output.Write("state-step-000000.vtk", "vtk\n", detail) ==
                         DurableIoStatus::ok &&
                     output.Write("run-metadata.json", "{}\n", detail) ==
                         DurableIoStatus::ok &&
                     output.Publish(detail) == DurableIoStatus::ok &&
                     std::filesystem::is_regular_file(
                         final / "state-step-000000.vtk"),
                 "atomic_output_directory") && passed;
  AtomicOutputDirectory duplicate;
  passed = Check(duplicate.Begin(final, detail) ==
                         DurableIoStatus::already_exists,
                 "never_overwrite_output") && passed;
  AtomicOutputDirectory missing_parent;
  passed = Check(missing_parent.Begin(root / "missing" / "output", detail) ==
                          DurableIoStatus::invalid_path,
                  "parent_must_exist") && passed;

  const auto durable = root / "durable-output";
  passed = Check(CreateDurableDirectory(durable, detail) ==
                         DurableIoStatus::ok,
                 "create_durable_output") && passed;
  DurableStagedFileV1 reversible;
  const auto reversible_path = durable / "candidate.vtk";
  passed = Check(reversible.Stage(reversible_path, payload.data(),
                                  payload.size(), detail) ==
                         DurableIoStatus::ok &&
                     !std::filesystem::exists(reversible_path) &&
                     reversible.Publish(detail) == DurableIoStatus::ok &&
                     std::filesystem::is_regular_file(reversible_path) &&
                     reversible.Revert(detail) == DurableIoStatus::ok &&
                     !std::filesystem::exists(reversible_path),
                 "published_candidate_is_reversible") && passed;
  const auto prior_path = durable / "prior.vtk";
  passed = Check(PublishDurableFile(prior_path, payload.data(), payload.size(),
                                    detail) == DurableIoStatus::ok,
                 "publish_prior_output") && passed;
  DurableStagedFileV1 failed_later;
  const auto later_path = durable / "later.vtk";
  passed = Check(failed_later.Stage(later_path, payload.data(), payload.size(),
                                    detail) == DurableIoStatus::ok &&
                     failed_later.Revert(detail) == DurableIoStatus::ok &&
                     std::filesystem::is_regular_file(prior_path) &&
                     !std::filesystem::exists(later_path),
                 "later_rollback_preserves_prior_publication") && passed;
  const auto empty = root / "empty-transaction";
  passed = Check(CreateDurableDirectory(empty, detail) == DurableIoStatus::ok &&
                     DiscardDurableDirectoryIfEmpty(empty, detail) ==
                         DurableIoStatus::ok &&
                     !std::filesystem::exists(empty) &&
                     DiscardDurableDirectoryIfEmpty(empty, detail) ==
                         DurableIoStatus::ok &&
                     DiscardDurableDirectoryIfEmpty(durable, detail) ==
                         DurableIoStatus::ok &&
                     std::filesystem::is_regular_file(prior_path),
                 "empty_directory_rollback_preserves_nonempty_prior") && passed;

  const std::array<DurableGenerationCrashBoundaryV1, 5> death_boundaries{
      DurableGenerationCrashBoundaryV1::none,
      DurableGenerationCrashBoundaryV1::after_manifest,
      DurableGenerationCrashBoundaryV1::after_generation_rename,
      DurableGenerationCrashBoundaryV1::after_aliases,
      DurableGenerationCrashBoundaryV1::after_marker_rename};
  for (std::size_t index = 0; index < death_boundaries.size(); ++index)
  {
    passed = Check(ProcessDeathCase(root, death_boundaries[index],
                                    static_cast<int>(index), detail),
                   "generation_process_death_is_atomic") && passed;
  }
  for (int kind = 0; kind < 6; ++kind)
  {
    passed = Check(ManifestCorruptionCase(root, kind, detail),
                   "generation_manifest_integrity_is_fail_closed") && passed;
  }
  passed = Check(FinalSyncFailureCase(root, detail),
                 "final_directory_sync_failure_preserves_previous_generation") &&
      passed;
  passed = Check(RestartAuthorityCase(root, detail),
                 "restart_requires_current_manifest_authority") && passed;
  passed = Check(PublishedGenerationRevertCase(root, detail),
                 "published_generation_remains_reversible_until_commit") &&
      passed;
  std::filesystem::remove_all(root, error);
  if (passed) std::cout << "HDG runtime durable I/O PASS\n";
  return passed ? 0 : 1;
}

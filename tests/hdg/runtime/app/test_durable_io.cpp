#include "numerics/hdg/runtime/app/durable_io.hpp"

#include <filesystem>
#include <iostream>
#include <string>
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

bool Check(const bool condition, const char* name)
{
  if (!condition) std::cerr << "FAILED: " << name << '\n';
  return condition;
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
  std::filesystem::remove_all(root, error);
  if (passed) std::cout << "HDG runtime durable I/O PASS\n";
  return passed ? 0 : 1;
}

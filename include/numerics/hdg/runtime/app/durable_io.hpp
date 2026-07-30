#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace dg::hdg::runtime::app
{

enum class DurableIoStatus : unsigned char
{
  ok,
  invalid_path,
  already_exists,
  create_failed,
  open_failed,
  write_failed,
  sync_failed,
  verify_failed,
  rename_failed,
  read_failed,
  too_large,
  cleanup_failed
};

std::string_view ToString(DurableIoStatus status) noexcept;

DurableIoStatus ReadDurableFile(const std::filesystem::path& path,
                                std::size_t maximum_bytes,
                                std::vector<std::uint8_t>& bytes,
                                std::string& detail) noexcept;

namespace detail
{
/** Internal restart authority; ordinary or stale checkpoint paths fail. */
DurableIoStatus ReadCurrentGenerationArtifact(
    const std::filesystem::path& path,
    std::size_t maximum_bytes,
    std::vector<std::uint8_t>& bytes,
    std::string& detail) noexcept;
}

DurableIoStatus PublishDurableFile(const std::filesystem::path& path,
                                   const std::uint8_t* bytes,
                                   std::size_t byte_count,
                                   std::string& detail) noexcept;

DurableIoStatus CreateDurableDirectory(const std::filesystem::path& path,
                                       std::string& detail) noexcept;

/** Remove a task-created directory only while it is still empty. */
DurableIoStatus DiscardDurableDirectoryIfEmpty(
    const std::filesystem::path& path, std::string& detail) noexcept;

/** One reversible, verified file publication used by the solver transaction. */
class DurableStagedFileV1 final
{
public:
  DurableStagedFileV1() = default;
  ~DurableStagedFileV1();
  DurableStagedFileV1(const DurableStagedFileV1&) = delete;
  DurableStagedFileV1& operator=(const DurableStagedFileV1&) = delete;

  DurableIoStatus Stage(const std::filesystem::path& final_path,
                        const std::uint8_t* bytes,
                        std::size_t byte_count,
                        std::string& detail) noexcept;
  DurableIoStatus Publish(std::string& detail) noexcept;
  DurableIoStatus Revert(std::string& detail) noexcept;
  bool staged() const noexcept { return staged_; }
  bool published() const noexcept { return published_; }
  const std::filesystem::path& final_path() const noexcept
  { return final_path_; }

private:
  void BestEffortDiscardTemporary() noexcept;

  std::filesystem::path final_path_{};
  std::filesystem::path temporary_path_{};
  bool staged_ = false;
  bool published_ = false;
};

enum class DurableGenerationCrashBoundaryV1 : unsigned char
{
  none,
  after_manifest,
  after_generation_rename,
  after_aliases,
  after_marker_rename
};

/**
 * A private application artifact transaction.
 *
 * Every generation is immutable.  Logical artifact paths are stable symbolic
 * links through one atomic `.dg-igr-current` marker, so readers observe the
 * previous complete generation until the marker changes to the next complete
 * generation.
 */
class DurableGenerationTransactionV1 final
{
public:
  DurableGenerationTransactionV1() = default;
  ~DurableGenerationTransactionV1();
  DurableGenerationTransactionV1(
      const DurableGenerationTransactionV1&) = delete;
  DurableGenerationTransactionV1& operator=(
      const DurableGenerationTransactionV1&) = delete;

  DurableIoStatus Begin(const std::filesystem::path& output_root,
                        std::uint64_t generation,
                        std::string& detail) noexcept;
  DurableIoStatus Stage(std::string_view logical_name,
                        const std::uint8_t* bytes,
                        std::size_t byte_count,
                        std::string& detail) noexcept;
  DurableIoStatus AddAlias(const std::filesystem::path& alias_path,
                           std::string_view logical_name,
                           std::string& detail) noexcept;
  DurableIoStatus Publish(std::string& detail) noexcept;
  DurableIoStatus Revert(std::string& detail) noexcept;

  bool active() const noexcept { return active_; }
  bool staged() const noexcept { return staged_; }
  bool published() const noexcept { return published_; }

  // Used only by the durable-I/O subprocess tests.
  void SetCrashBoundaryForTest(
      DurableGenerationCrashBoundaryV1 boundary) noexcept
  { test_crash_boundary_ = boundary; }

private:
  struct Alias
  {
    std::filesystem::path path{};
    std::filesystem::path target{};
  };

  void BestEffortCleanup() noexcept;

  std::filesystem::path output_root_{};
  std::filesystem::path generations_root_{};
  std::filesystem::path staging_path_{};
  std::filesystem::path final_generation_path_{};
  std::filesystem::path previous_marker_target_{};
  std::vector<Alias> aliases_{};
  std::vector<std::filesystem::path> created_aliases_{};
  std::uint64_t generation_ = 0;
  bool active_ = false;
  bool staged_ = false;
  bool generation_renamed_ = false;
  bool published_ = false;
  DurableGenerationCrashBoundaryV1 test_crash_boundary_ =
      DurableGenerationCrashBoundaryV1::none;
};

class AtomicOutputDirectory final
{
public:
  AtomicOutputDirectory() = default;
  ~AtomicOutputDirectory();
  AtomicOutputDirectory(const AtomicOutputDirectory&) = delete;
  AtomicOutputDirectory& operator=(const AtomicOutputDirectory&) = delete;

  DurableIoStatus Begin(const std::filesystem::path& final_path,
                        std::string& detail) noexcept;
  DurableIoStatus Write(std::string_view relative_name,
                        std::string_view contents,
                        std::string& detail) noexcept;
  DurableIoStatus Publish(std::string& detail) noexcept;
  const std::filesystem::path& staging_path() const noexcept
  { return staging_path_; }
  const std::filesystem::path& final_path() const noexcept
  { return final_path_; }

private:
  void BestEffortCleanup() noexcept;

  std::filesystem::path final_path_{};
  std::filesystem::path staging_path_{};
  bool active_ = false;
};

} // namespace dg::hdg::runtime::app

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

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

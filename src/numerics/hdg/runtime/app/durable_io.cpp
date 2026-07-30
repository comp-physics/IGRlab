#include "numerics/hdg/runtime/app/durable_io.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <system_error>
#include <unistd.h>

namespace dg::hdg::runtime::app
{
namespace
{

std::string ErrorText(const char* operation, const int error)
{
  std::string text{operation};
  text += ": ";
  text += std::strerror(error);
  return text;
}

bool SafeRelativeName(const std::string_view name) noexcept
{
  return !name.empty() && name.front() != '/' && name.find("..") ==
      std::string_view::npos && name.find('/') == std::string_view::npos;
}

DurableIoStatus SyncDirectory(const std::filesystem::path& directory,
                              std::string& detail) noexcept
{
  const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
  if (descriptor < 0)
  {
    detail = ErrorText("open parent directory", errno);
    return DurableIoStatus::open_failed;
  }
  const int result = ::fsync(descriptor);
  const int saved = errno;
  (void)::close(descriptor);
  if (result != 0)
  {
    detail = ErrorText("fsync parent directory", saved);
    return DurableIoStatus::sync_failed;
  }
  return DurableIoStatus::ok;
}

DurableIoStatus WriteAndVerify(const std::filesystem::path& path,
                               const std::uint8_t* bytes,
                               const std::size_t byte_count,
                               std::string& detail) noexcept
{
  if ((byte_count != 0 && bytes == nullptr) ||
      byte_count > static_cast<std::size_t>(
          std::numeric_limits<ssize_t>::max()))
  {
    detail = "invalid byte buffer";
    return DurableIoStatus::invalid_path;
  }
  const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL,
                                S_IRUSR | S_IWUSR | S_IRGRP);
  if (descriptor < 0)
  {
    detail = ErrorText("open temporary", errno);
    return DurableIoStatus::open_failed;
  }
  std::size_t written = 0;
  while (written < byte_count)
  {
    const ssize_t count = ::write(descriptor, bytes + written,
                                  byte_count - written);
    if (count < 0 && errno == EINTR) { continue; }
    if (count <= 0)
    {
      const int saved = errno;
      (void)::close(descriptor);
      detail = ErrorText("write temporary", saved);
      return DurableIoStatus::write_failed;
    }
    written += static_cast<std::size_t>(count);
  }
  if (::fsync(descriptor) != 0)
  {
    const int saved = errno;
    (void)::close(descriptor);
    detail = ErrorText("fsync temporary", saved);
    return DurableIoStatus::sync_failed;
  }
  if (::close(descriptor) != 0)
  {
    detail = ErrorText("close temporary", errno);
    return DurableIoStatus::sync_failed;
  }
  std::vector<std::uint8_t> verified;
  DurableIoStatus status = ReadDurableFile(path, byte_count, verified, detail);
  if (status != DurableIoStatus::ok || verified.size() != byte_count ||
      !std::equal(verified.begin(), verified.end(), bytes))
  {
    if (status == DurableIoStatus::ok) { detail = "temporary byte verification mismatch"; }
    return DurableIoStatus::verify_failed;
  }
  return DurableIoStatus::ok;
}

std::filesystem::path TemporarySibling(const std::filesystem::path& path)
{
  std::filesystem::path temporary = path;
  temporary += ".tmp.";
  temporary += std::to_string(static_cast<long long>(::getpid()));
  return temporary;
}

} // namespace

std::string_view ToString(const DurableIoStatus value) noexcept
{
  switch (value)
  {
    case DurableIoStatus::ok: return "ok";
    case DurableIoStatus::invalid_path: return "invalid_path";
    case DurableIoStatus::already_exists: return "already_exists";
    case DurableIoStatus::create_failed: return "create_failed";
    case DurableIoStatus::open_failed: return "open_failed";
    case DurableIoStatus::write_failed: return "write_failed";
    case DurableIoStatus::sync_failed: return "sync_failed";
    case DurableIoStatus::verify_failed: return "verify_failed";
    case DurableIoStatus::rename_failed: return "rename_failed";
    case DurableIoStatus::read_failed: return "read_failed";
    case DurableIoStatus::too_large: return "too_large";
    case DurableIoStatus::cleanup_failed: return "cleanup_failed";
  }
  return "unknown";
}

DurableIoStatus ReadDurableFile(const std::filesystem::path& path,
                                const std::size_t maximum_bytes,
                                std::vector<std::uint8_t>& bytes,
                                std::string& detail) noexcept
{
  bytes.clear();
  try
  {
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error)
    {
      detail = "file_size: " + error.message();
      return DurableIoStatus::read_failed;
    }
    if (size > maximum_bytes || size > bytes.max_size())
    {
      detail = "file exceeds admitted size";
      return DurableIoStatus::too_large;
    }
    bytes.resize(static_cast<std::size_t>(size));
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0)
    {
      detail = ErrorText("open", errno);
      bytes.clear();
      return DurableIoStatus::open_failed;
    }
    std::size_t read = 0;
    while (read < bytes.size())
    {
      const ssize_t count = ::read(descriptor, bytes.data() + read,
                                   bytes.size() - read);
      if (count < 0 && errno == EINTR) { continue; }
      if (count <= 0)
      {
        const int saved = errno;
        (void)::close(descriptor);
        detail = count == 0 ? "unexpected end of file"
                            : ErrorText("read", saved);
        bytes.clear();
        return DurableIoStatus::read_failed;
      }
      read += static_cast<std::size_t>(count);
    }
    if (::close(descriptor) != 0)
    {
      detail = ErrorText("close", errno);
      bytes.clear();
      return DurableIoStatus::read_failed;
    }
    return DurableIoStatus::ok;
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    bytes.clear();
    return DurableIoStatus::read_failed;
  }
}

DurableIoStatus PublishDurableFile(const std::filesystem::path& path,
                                   const std::uint8_t* bytes,
                                   const std::size_t byte_count,
                                   std::string& detail) noexcept
{
  if (path.empty() || path.filename().empty())
  {
    detail = "empty publication path";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    const std::filesystem::path parent = path.has_parent_path()
        ? path.parent_path() : std::filesystem::path{"."};
    std::error_code error;
    if (!std::filesystem::is_directory(parent, error) || error)
    {
      detail = "publication parent must already exist";
      return DurableIoStatus::invalid_path;
    }
    if (std::filesystem::exists(path, error) || error)
    {
      detail = "publication path already exists";
      return DurableIoStatus::already_exists;
    }
    const std::filesystem::path temporary = TemporarySibling(path);
    if (std::filesystem::exists(temporary, error))
    {
      detail = "temporary path already exists";
      return DurableIoStatus::already_exists;
    }
    DurableIoStatus status = WriteAndVerify(temporary, bytes, byte_count,
                                            detail);
    if (status != DurableIoStatus::ok)
    {
      std::filesystem::remove(temporary, error);
      return status;
    }
    if (::link(temporary.c_str(), path.c_str()) != 0)
    {
      const int saved = errno;
      std::filesystem::remove(temporary, error);
      detail = ErrorText("link publication", saved);
      return saved == EEXIST ? DurableIoStatus::already_exists
                             : DurableIoStatus::rename_failed;
    }
    if (::unlink(temporary.c_str()) != 0)
    {
      detail = ErrorText("unlink temporary publication", errno);
      return DurableIoStatus::rename_failed;
    }
    return SyncDirectory(parent, detail);
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::write_failed;
  }
}

DurableIoStatus CreateDurableDirectory(const std::filesystem::path& path,
                                       std::string& detail) noexcept
{
  if (path.empty() || path.filename().empty())
  {
    detail = "empty directory publication path";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    const std::filesystem::path parent = path.has_parent_path()
        ? path.parent_path() : std::filesystem::path{"."};
    std::error_code error;
    if (!std::filesystem::is_directory(parent, error) || error)
    {
      detail = "directory parent must already exist";
      return DurableIoStatus::invalid_path;
    }
    if (std::filesystem::exists(path, error) || error)
    {
      detail = "directory publication path already exists";
      return DurableIoStatus::already_exists;
    }
    if (!std::filesystem::create_directory(path, error) || error)
    {
      detail = "create durable directory: " + error.message();
      return DurableIoStatus::create_failed;
    }
    return SyncDirectory(parent, detail);
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::create_failed;
  }
}

DurableIoStatus DiscardDurableDirectoryIfEmpty(
    const std::filesystem::path& path, std::string& detail) noexcept
{
  if (path.empty() || path.filename().empty())
  {
    detail = "empty directory cleanup path";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    std::error_code error;
    if (!std::filesystem::exists(path, error))
    {
      return error ? DurableIoStatus::cleanup_failed : DurableIoStatus::ok;
    }
    if (!std::filesystem::is_directory(path, error) || error)
    {
      detail = "directory cleanup target is not a directory";
      return DurableIoStatus::invalid_path;
    }
    if (!std::filesystem::is_empty(path, error) || error)
    {
      if (error)
      {
        detail = "inspect directory cleanup target: " + error.message();
        return DurableIoStatus::cleanup_failed;
      }
      return DurableIoStatus::ok;
    }
    if (!std::filesystem::remove(path, error) || error)
    {
      detail = "remove empty durable directory: " + error.message();
      return DurableIoStatus::cleanup_failed;
    }
    const std::filesystem::path parent = path.has_parent_path()
        ? path.parent_path() : std::filesystem::path{"."};
    return SyncDirectory(parent, detail);
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::cleanup_failed;
  }
}

DurableStagedFileV1::~DurableStagedFileV1()
{
  BestEffortDiscardTemporary();
}

DurableIoStatus DurableStagedFileV1::Stage(
    const std::filesystem::path& final_path, const std::uint8_t* bytes,
    const std::size_t byte_count, std::string& detail) noexcept
{
  if (staged_ || published_ || final_path.empty() ||
      final_path.filename().empty())
  {
    detail = "invalid staged-file state/path";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    const std::filesystem::path parent = final_path.has_parent_path()
        ? final_path.parent_path() : std::filesystem::path{"."};
    std::error_code error;
    if (!std::filesystem::is_directory(parent, error) || error)
    {
      detail = "staged-file parent must already exist";
      return DurableIoStatus::invalid_path;
    }
    if (std::filesystem::exists(final_path, error) || error)
    {
      detail = "staged-file final path already exists";
      return DurableIoStatus::already_exists;
    }
    final_path_ = final_path;
    temporary_path_ = TemporarySibling(final_path);
    if (std::filesystem::exists(temporary_path_, error) || error)
    {
      detail = "staged-file temporary path already exists";
      final_path_.clear();
      temporary_path_.clear();
      return DurableIoStatus::already_exists;
    }
    const DurableIoStatus status =
        WriteAndVerify(temporary_path_, bytes, byte_count, detail);
    if (status != DurableIoStatus::ok)
    {
      std::filesystem::remove(temporary_path_, error);
      final_path_.clear();
      temporary_path_.clear();
      return status;
    }
    staged_ = true;
    return DurableIoStatus::ok;
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    std::error_code ignored;
    if (!temporary_path_.empty())
      std::filesystem::remove(temporary_path_, ignored);
    final_path_.clear();
    temporary_path_.clear();
    staged_ = false;
    return DurableIoStatus::write_failed;
  }
}

DurableIoStatus DurableStagedFileV1::Publish(std::string& detail) noexcept
{
  if (!staged_ || published_)
  {
    detail = "staged file is not publishable";
    return DurableIoStatus::invalid_path;
  }
  if (::link(temporary_path_.c_str(), final_path_.c_str()) != 0)
  {
    const int saved = errno;
    detail = ErrorText("link staged publication", saved);
    return saved == EEXIST ? DurableIoStatus::already_exists
                           : DurableIoStatus::rename_failed;
  }
  published_ = true;
  if (::unlink(temporary_path_.c_str()) != 0)
  {
    detail = ErrorText("unlink staged temporary", errno);
    return DurableIoStatus::rename_failed;
  }
  staged_ = false;
  temporary_path_.clear();
  const std::filesystem::path parent = final_path_.has_parent_path()
      ? final_path_.parent_path() : std::filesystem::path{"."};
  return SyncDirectory(parent, detail);
}

DurableIoStatus DurableStagedFileV1::Revert(std::string& detail) noexcept
{
  if (!staged_ && !published_) { return DurableIoStatus::ok; }
  try
  {
    std::error_code error;
    if (staged_ && !temporary_path_.empty() &&
        !std::filesystem::remove(temporary_path_, error) && error)
    {
      detail = "remove staged temporary: " + error.message();
      return DurableIoStatus::cleanup_failed;
    }
    staged_ = false;
    temporary_path_.clear();
    if (published_ &&
        (!std::filesystem::remove(final_path_, error) || error))
    {
      detail = "remove reverted publication: " + error.message();
      return DurableIoStatus::cleanup_failed;
    }
    const std::filesystem::path parent = final_path_.has_parent_path()
        ? final_path_.parent_path() : std::filesystem::path{"."};
    published_ = false;
    final_path_.clear();
    return SyncDirectory(parent, detail);
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::cleanup_failed;
  }
}

void DurableStagedFileV1::BestEffortDiscardTemporary() noexcept
{
  if (!staged_ || temporary_path_.empty()) { return; }
  std::error_code ignored;
  std::filesystem::remove(temporary_path_, ignored);
  staged_ = false;
  temporary_path_.clear();
}

AtomicOutputDirectory::~AtomicOutputDirectory() { BestEffortCleanup(); }

DurableIoStatus AtomicOutputDirectory::Begin(
    const std::filesystem::path& final_path, std::string& detail) noexcept
{
  if (active_ || final_path.empty() || final_path.filename().empty())
  {
    detail = "invalid output directory state/path";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    std::error_code error;
    if (std::filesystem::exists(final_path, error))
    {
      detail = "output directory already exists";
      return DurableIoStatus::already_exists;
    }
    final_path_ = final_path;
    staging_path_ = TemporarySibling(final_path);
    if (std::filesystem::exists(staging_path_, error))
    {
      detail = "output staging directory already exists";
      return DurableIoStatus::already_exists;
    }
    const auto parent = final_path_.has_parent_path()
        ? final_path_.parent_path() : std::filesystem::path{"."};
    if (!std::filesystem::is_directory(parent, error) || error)
    {
      detail = "output parent must already exist";
      return DurableIoStatus::invalid_path;
    }
    if (!std::filesystem::create_directory(staging_path_, error) || error)
    {
      detail = "create output staging: " + error.message();
      return DurableIoStatus::create_failed;
    }
    active_ = true;
    return DurableIoStatus::ok;
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::create_failed;
  }
}

DurableIoStatus AtomicOutputDirectory::Write(
    const std::string_view relative_name, const std::string_view contents,
    std::string& detail) noexcept
{
  if (!active_ || !SafeRelativeName(relative_name))
  {
    detail = "invalid output member name";
    return DurableIoStatus::invalid_path;
  }
  const std::filesystem::path path = staging_path_ /
      std::string{relative_name};
  return PublishDurableFile(
      path, reinterpret_cast<const std::uint8_t*>(contents.data()),
      contents.size(), detail);
}

DurableIoStatus AtomicOutputDirectory::Publish(std::string& detail) noexcept
{
  if (!active_)
  {
    detail = "output session is not active";
    return DurableIoStatus::invalid_path;
  }
  const auto parent = final_path_.has_parent_path()
      ? final_path_.parent_path() : std::filesystem::path{"."};
  DurableIoStatus status = SyncDirectory(staging_path_, detail);
  if (status != DurableIoStatus::ok) { return status; }
  if (::rename(staging_path_.c_str(), final_path_.c_str()) != 0)
  {
    detail = ErrorText("rename output directory", errno);
    return DurableIoStatus::rename_failed;
  }
  active_ = false;
  staging_path_.clear();
  return SyncDirectory(parent, detail);
}

void AtomicOutputDirectory::BestEffortCleanup() noexcept
{
  if (!active_ || staging_path_.empty()) { return; }
  std::error_code ignored;
  std::filesystem::remove_all(staging_path_, ignored);
  active_ = false;
}

} // namespace dg::hdg::runtime::app

#include "numerics/hdg/runtime/app/durable_io.hpp"

#include "numerics/hdg/runtime/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <sstream>
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

std::string Hex(const Sha256Digest& hash)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string text(64, '0');
  for (std::size_t index = 0; index < hash.size(); ++index)
  {
    text[2 * index] = digits[hash[index] >> 4U];
    text[2 * index + 1] = digits[hash[index] & 0xfU];
  }
  return text;
}

bool SameSymlink(const std::filesystem::path& path,
                 const std::filesystem::path& expected) noexcept
{
  try
  {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    return !error && std::filesystem::is_symlink(status) &&
        std::filesystem::read_symlink(path, error) == expected && !error;
  }
  catch (...) { return false; }
}

DurableIoStatus EnsureSymlink(const std::filesystem::path& path,
                              const std::filesystem::path& target,
                              bool& created,
                              std::string& detail) noexcept
{
  created = false;
  try
  {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (!error && status.type() != std::filesystem::file_type::not_found)
    {
      if (SameSymlink(path, target)) return DurableIoStatus::ok;
      detail = "artifact alias already exists with a different target";
      return DurableIoStatus::already_exists;
    }
    error.clear();
    const auto parent = path.has_parent_path()
        ? path.parent_path() : std::filesystem::path{"."};
    if (!std::filesystem::is_directory(parent, error) || error)
    {
      detail = "artifact alias parent is unavailable";
      return DurableIoStatus::invalid_path;
    }
    std::filesystem::create_symlink(target, path, error);
    if (error)
    {
      detail = "create artifact alias: " + error.message();
      return DurableIoStatus::create_failed;
    }
    created = true;
    return SyncDirectory(parent, detail);
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::create_failed;
  }
}

void CrashAt(const DurableGenerationCrashBoundaryV1 selected,
             const DurableGenerationCrashBoundaryV1 reached) noexcept
{
  if (selected == reached) ::_exit(86);
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

DurableGenerationTransactionV1::~DurableGenerationTransactionV1()
{
  BestEffortCleanup();
}

DurableIoStatus DurableGenerationTransactionV1::Begin(
    const std::filesystem::path& output_root,
    const std::uint64_t generation, std::string& detail) noexcept
{
  if (active_ || published_ || output_root.empty())
  {
    detail = "invalid generation transaction state/root";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    std::error_code error;
    if (!std::filesystem::is_directory(output_root, error) || error)
    {
      detail = "generation output root must already exist";
      return DurableIoStatus::invalid_path;
    }
    output_root_ = std::filesystem::absolute(output_root).lexically_normal();
    generations_root_ = output_root_ / ".dg-igr-generations";
    if (!std::filesystem::exists(generations_root_, error))
    {
      const DurableIoStatus made =
          CreateDurableDirectory(generations_root_, detail);
      if (made != DurableIoStatus::ok) return made;
    }
    else if (!std::filesystem::is_directory(generations_root_, error) || error)
    {
      detail = "generation root is not a directory";
      return DurableIoStatus::invalid_path;
    }

    static std::atomic<unsigned long long> sequence{0};
    const auto ordinal = sequence.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream name;
    name << "generation-" << generation << '-' <<
        static_cast<long long>(::getpid()) << '-' << ordinal;
    final_generation_path_ = generations_root_ / name.str();
    staging_path_ = generations_root_ / (".tmp-" + name.str());
    if (std::filesystem::exists(final_generation_path_, error) ||
        std::filesystem::exists(staging_path_, error))
    {
      detail = "generation path already exists";
      return DurableIoStatus::already_exists;
    }
    if (!std::filesystem::create_directory(staging_path_, error) || error)
    {
      detail = "create generation staging directory: " + error.message();
      return DurableIoStatus::create_failed;
    }

    const std::filesystem::path marker = output_root_ / ".dg-igr-current";
    const auto marker_status = std::filesystem::symlink_status(marker, error);
    if (!error && marker_status.type() != std::filesystem::file_type::not_found)
    {
      if (!std::filesystem::is_symlink(marker_status))
      {
        detail = "generation marker is not a symbolic link";
        BestEffortCleanup();
        return DurableIoStatus::verify_failed;
      }
      const auto target = std::filesystem::read_symlink(marker, error);
      const auto current = (output_root_ / target).lexically_normal();
      if (error || current.parent_path() != generations_root_ ||
          !std::filesystem::is_directory(current, error) || error ||
          !std::filesystem::is_regular_file(current / "manifest.v1", error) ||
          error)
      {
        detail = "generation marker does not name a complete generation";
        BestEffortCleanup();
        return DurableIoStatus::verify_failed;
      }
      for (const auto& member : std::filesystem::directory_iterator(current))
      {
        if (member.path().filename() == "manifest.v1") continue;
        if (!member.is_regular_file())
        {
          detail = "current generation contains a non-file artifact";
          BestEffortCleanup();
          return DurableIoStatus::verify_failed;
        }
        std::filesystem::create_hard_link(
            member.path(), staging_path_ / member.path().filename(), error);
        if (error)
        {
          detail = "clone current generation artifact: " + error.message();
          BestEffortCleanup();
          return DurableIoStatus::create_failed;
        }
      }
    }
    else if (error && error != std::errc::no_such_file_or_directory)
    {
      detail = "inspect generation marker: " + error.message();
      BestEffortCleanup();
      return DurableIoStatus::read_failed;
    }
    generation_ = generation;
    active_ = true;
    return DurableIoStatus::ok;
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    BestEffortCleanup();
    return DurableIoStatus::create_failed;
  }
}

DurableIoStatus DurableGenerationTransactionV1::Stage(
    const std::string_view logical_name, const std::uint8_t* bytes,
    const std::size_t byte_count, std::string& detail) noexcept
{
  if (!active_ || published_ || !SafeRelativeName(logical_name) ||
      logical_name == "manifest.v1" || logical_name == ".dg-igr-current")
  {
    detail = "invalid generation artifact name/state";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    const std::filesystem::path artifact =
        staging_path_ / std::string{logical_name};
    std::error_code error;
    const auto status = std::filesystem::symlink_status(artifact, error);
    if (!error && status.type() != std::filesystem::file_type::not_found &&
        (!std::filesystem::remove(artifact, error) || error))
    {
      detail = "replace staged generation artifact: " + error.message();
      return DurableIoStatus::cleanup_failed;
    }
    const DurableIoStatus written =
        WriteAndVerify(artifact, bytes, byte_count, detail);
    if (written == DurableIoStatus::ok) staged_ = true;
    return written;
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::write_failed;
  }
}

DurableIoStatus DurableGenerationTransactionV1::AddAlias(
    const std::filesystem::path& alias_path,
    const std::string_view logical_name, std::string& detail) noexcept
{
  if (!active_ || alias_path.empty() || !SafeRelativeName(logical_name) ||
      !std::filesystem::is_regular_file(
          staging_path_ / std::string{logical_name}))
  {
    detail = "invalid generation alias";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    const auto absolute_alias =
        std::filesystem::absolute(alias_path).lexically_normal();
    const auto target = output_root_ / ".dg-igr-current" /
        std::string{logical_name};
    for (const Alias& alias : aliases_)
      if (alias.path == absolute_alias)
        return alias.target == target ? DurableIoStatus::ok
                                      : DurableIoStatus::already_exists;
    aliases_.push_back({absolute_alias, target});
    return DurableIoStatus::ok;
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::invalid_path;
  }
}

DurableIoStatus DurableGenerationTransactionV1::Publish(
    std::string& detail) noexcept
{
  if (!active_ || published_ || !staged_)
  {
    detail = "generation transaction is not publishable";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    std::vector<std::filesystem::path> artifacts;
    for (const auto& member : std::filesystem::directory_iterator(staging_path_))
      if (member.is_regular_file() && member.path().filename() != "manifest.v1")
        artifacts.push_back(member.path());
    std::sort(artifacts.begin(), artifacts.end());
    std::ostringstream manifest;
    manifest << "DG_IGR_DURABLE_GENERATION_V1\n"
             << "generation\t" << generation_ << '\n';
    for (const auto& artifact : artifacts)
    {
      std::vector<std::uint8_t> bytes;
      DurableIoStatus read = ReadDurableFile(
          artifact, 256U * 1024U * 1024U, bytes, detail);
      if (read != DurableIoStatus::ok) return read;
      manifest << "artifact\t" << artifact.filename().string() << '\t'
               << bytes.size() << '\t'
               << Hex(ComputeSha256(bytes.data(), bytes.size())) << '\n';
    }
    const std::string manifest_text = manifest.str();
    DurableIoStatus status = WriteAndVerify(
        staging_path_ / "manifest.v1",
        reinterpret_cast<const std::uint8_t*>(manifest_text.data()),
        manifest_text.size(), detail);
    if (status != DurableIoStatus::ok) return status;
    status = SyncDirectory(staging_path_, detail);
    if (status != DurableIoStatus::ok) return status;
    CrashAt(test_crash_boundary_,
            DurableGenerationCrashBoundaryV1::after_manifest);

    if (::rename(staging_path_.c_str(), final_generation_path_.c_str()) != 0)
    {
      detail = ErrorText("rename immutable generation", errno);
      return DurableIoStatus::rename_failed;
    }
    generation_renamed_ = true;
    staging_path_.clear();
    status = SyncDirectory(generations_root_, detail);
    if (status != DurableIoStatus::ok) return status;
    CrashAt(test_crash_boundary_,
            DurableGenerationCrashBoundaryV1::after_generation_rename);

    for (const auto& artifact : artifacts)
    {
      const auto logical = artifact.filename();
      bool created = false;
      status = EnsureSymlink(output_root_ / logical,
                             std::filesystem::path{".dg-igr-current"} /
                                 logical,
                             created, detail);
      if (status != DurableIoStatus::ok) return status;
      if (created) created_aliases_.push_back(output_root_ / logical);
    }
    for (const Alias& alias : aliases_)
    {
      bool created = false;
      status = EnsureSymlink(alias.path, alias.target, created, detail);
      if (status != DurableIoStatus::ok) return status;
      if (created) created_aliases_.push_back(alias.path);
    }
    CrashAt(test_crash_boundary_,
            DurableGenerationCrashBoundaryV1::after_aliases);

    const std::filesystem::path marker = output_root_ / ".dg-igr-current";
    const std::filesystem::path temporary = TemporarySibling(marker);
    std::error_code error;
    std::filesystem::remove(temporary, error);
    error.clear();
    const auto relative_target = std::filesystem::path{
        ".dg-igr-generations"} / final_generation_path_.filename();
    std::filesystem::create_symlink(relative_target, temporary, error);
    if (error)
    {
      detail = "create generation marker: " + error.message();
      return DurableIoStatus::create_failed;
    }
    if (::rename(temporary.c_str(), marker.c_str()) != 0)
    {
      const int saved = errno;
      std::filesystem::remove(temporary, error);
      detail = ErrorText("replace generation marker", saved);
      return DurableIoStatus::rename_failed;
    }
    published_ = true;
    active_ = false;
    CrashAt(test_crash_boundary_,
            DurableGenerationCrashBoundaryV1::after_marker_rename);
    return SyncDirectory(output_root_, detail);
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::write_failed;
  }
}

DurableIoStatus DurableGenerationTransactionV1::Revert(
    std::string& detail) noexcept
{
  if (published_)
  {
    detail = "a committed generation cannot be reverted";
    return DurableIoStatus::invalid_path;
  }
  try
  {
    std::error_code error;
    for (auto path = created_aliases_.rbegin();
         path != created_aliases_.rend(); ++path)
    {
      if (!std::filesystem::remove(*path, error) && error)
      {
        detail = "remove uncommitted artifact alias: " + error.message();
        return DurableIoStatus::cleanup_failed;
      }
      error.clear();
    }
    created_aliases_.clear();
    const auto remove = generation_renamed_ ? final_generation_path_
                                            : staging_path_;
    if (!remove.empty() && std::filesystem::exists(remove, error))
    {
      error.clear();
      std::filesystem::remove_all(remove, error);
      if (error)
      {
        detail = "remove uncommitted generation: " + error.message();
        return DurableIoStatus::cleanup_failed;
      }
      const DurableIoStatus sync = SyncDirectory(generations_root_, detail);
      if (sync != DurableIoStatus::ok) return sync;
    }
    active_ = false;
    staged_ = false;
    generation_renamed_ = false;
    staging_path_.clear();
    final_generation_path_.clear();
    aliases_.clear();
    return DurableIoStatus::ok;
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return DurableIoStatus::cleanup_failed;
  }
}

void DurableGenerationTransactionV1::BestEffortCleanup() noexcept
{
  if (published_ || (!active_ && !generation_renamed_)) return;
  std::string ignored;
  (void)Revert(ignored);
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

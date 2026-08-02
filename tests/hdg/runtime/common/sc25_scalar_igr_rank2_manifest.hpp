#pragma once

#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#include "numerics/hdg/runtime/sha256.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace dg_igr_hdg_sc25_rank2_manifest
{
namespace cpu = ::dg::hdg::runtime::cpu::sc25;

inline constexpr std::string_view accepted_cpu_oracle_receipt =
    "composition_commit=8e0c8cf56300d27b2d393aa3cd4eecb0507b97e0;"
    "composition_tree=d1419c84b41e870b8b0f30cc4c054ed4bbb8d588;"
    "cpu_header_blob=e37a439dfdd8ece59d226ce2b1a448c4b60d85bb;"
    "cpu_source_blob=9f337faf90bb0913212bbe0fad2d7aeee6c619b7;"
    "oracle_hpp_blob=1ddce570deee15902679463a2a75d3ccdd1a248b;"
    "oracle_cpp_blob=9420e66504a9a521fcac4a2770a4afbbebed1710;"
    "cpu_test_blob=c83865c2cdd01346b0a3a9bfd72ccbc2eb4cb54e;"
    "evidence=/tmp/HDG-W8B-DEV2-CPU3-01/attempt-002/SHA256SUMS;"
    "evidence_sha256=53a7cfaad21db64afe6854c157323478d7e757b38b2736baba1e9d20e85b8554";

inline constexpr std::string_view mfem_pin =
    "d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec";
inline constexpr std::uint32_t manifest_schema_version = 2;
inline constexpr std::uint32_t manifest_endian_marker = 0x01020304U;
inline constexpr std::uint64_t manifest_double_marker =
    0x3ff0000000000000ULL;
inline constexpr std::size_t maximum_manifest_bytes = 8U * 1024U * 1024U;
inline constexpr std::uint64_t maximum_executable_bytes =
    1024ULL * 1024ULL * 1024ULL;
inline constexpr double trace_relative_tolerance = 1.0e-12;
inline constexpr double trace_absolute_tolerance = 1.0e-14;

static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                  std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53,
              "C3 manifest requires IEEE-754 binary64");

using Digest = ::dg::hdg::runtime::Sha256Digest;

inline bool DigestEqual(const Digest& left, const Digest& right) noexcept
{
  return std::equal(left.begin(), left.end(), right.begin());
}

inline std::string Hex(const Digest& digest)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t i = 0; i < digest.size(); ++i)
  {
    result[2 * i] = digits[digest[i] >> 4U];
    result[2 * i + 1] = digits[digest[i] & 0x0fU];
  }
  return result;
}

inline bool ParseHexDigest(const std::string_view text, Digest& digest)
    noexcept
{
  if (text.size() != 64) return false;
  const auto nibble = [](const char value) -> int
  {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return 10 + value - 'a';
    return -1;
  };
  for (std::size_t i = 0; i < digest.size(); ++i)
  {
    const int high = nibble(text[2 * i]);
    const int low = nibble(text[2 * i + 1]);
    if (high < 0 || low < 0) return false;
    digest[i] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return true;
}

inline bool SafeRunId(const std::string_view value) noexcept
{
  if (value.empty() || value.size() > 128) return false;
  for (const char character : value)
  {
    const bool accepted =
        (character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9') || character == '-' ||
        character == '_';
    if (!accepted) return false;
  }
  return true;
}

inline std::uint64_t DoubleBits(const double value) noexcept
{
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "FP64 manifest contract");
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline double BitsDouble(const std::uint64_t bits) noexcept
{
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

struct DirectoryIdentity final
{
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
};

inline bool operator==(const DirectoryIdentity& left,
                       const DirectoryIdentity& right) noexcept
{
  return left.device == right.device && left.inode == right.inode;
}

inline bool operator!=(const DirectoryIdentity& left,
                       const DirectoryIdentity& right) noexcept
{
  return !(left == right);
}

/**
 * Pinned local handoff directory. This excludes symlink/FIFO and cross-user
 * substitution; it is not a cryptographic claim against a hostile process
 * running as the same uid.
 */
class DirectoryHandle final
{
public:
  DirectoryHandle() noexcept = default;
  ~DirectoryHandle() { Reset(); }
  DirectoryHandle(const DirectoryHandle&) = delete;
  DirectoryHandle& operator=(const DirectoryHandle&) = delete;

  DirectoryHandle(DirectoryHandle&& other) noexcept
      : descriptor_(other.descriptor_), identity_(other.identity_)
  {
    other.descriptor_ = -1;
    other.identity_ = {};
  }

  DirectoryHandle& operator=(DirectoryHandle&& other) noexcept
  {
    if (this != &other)
    {
      Reset();
      descriptor_ = other.descriptor_;
      identity_ = other.identity_;
      other.descriptor_ = -1;
      other.identity_ = {};
    }
    return *this;
  }

  bool Open(const std::string_view path) noexcept
  {
    Reset();
    if (path.empty() || path.size() > 4096 ||
        path.find('\0') != std::string_view::npos)
      return false;
    std::array<char, 4097> owned{};
    std::copy(path.begin(), path.end(), owned.begin());
    const int candidate = ::open(
        owned.data(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (candidate < 0) return false;
    struct stat status{};
    const bool valid = ::fstat(candidate, &status) == 0 &&
        S_ISDIR(status.st_mode) && status.st_uid == ::geteuid() &&
        (status.st_mode & (S_IRWXG | S_IRWXO)) == 0 &&
        (status.st_mode & S_IRWXU) == S_IRWXU;
    if (!valid)
    {
      ::close(candidate);
      return false;
    }
    descriptor_ = candidate;
    identity_.device = static_cast<std::uint64_t>(status.st_dev);
    identity_.inode = static_cast<std::uint64_t>(status.st_ino);
    return identity_.inode != 0;
  }

  bool Stable() const noexcept
  {
    struct stat status{};
    return descriptor_ >= 0 && ::fstat(descriptor_, &status) == 0 &&
        S_ISDIR(status.st_mode) && status.st_uid == ::geteuid() &&
        (status.st_mode & (S_IRWXG | S_IRWXO)) == 0 &&
        (status.st_mode & S_IRWXU) == S_IRWXU &&
        static_cast<std::uint64_t>(status.st_dev) == identity_.device &&
        static_cast<std::uint64_t>(status.st_ino) == identity_.inode;
  }

  bool valid() const noexcept { return descriptor_ >= 0; }
  int descriptor() const noexcept { return descriptor_; }
  const DirectoryIdentity& identity() const noexcept { return identity_; }

private:
  void Reset() noexcept
  {
    if (descriptor_ >= 0) ::close(descriptor_);
    descriptor_ = -1;
    identity_ = {};
  }

  int descriptor_ = -1;
  DirectoryIdentity identity_{};
};

class ByteWriter final
{
public:
  void U8(const std::uint8_t value) { bytes_.push_back(value); }

  void U32(const std::uint32_t value)
  {
    for (unsigned shift = 0; shift < 32; shift += 8)
      U8(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }

  void I32(const std::int32_t value)
  {
    U32(static_cast<std::uint32_t>(value));
  }

  void U64(const std::uint64_t value)
  {
    for (unsigned shift = 0; shift < 64; shift += 8)
      U8(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }

  void F64(const double value) { U64(DoubleBits(value)); }

  void Raw(const void* data, const std::size_t length)
  {
    const auto* first = static_cast<const std::uint8_t*>(data);
    bytes_.insert(bytes_.end(), first, first + length);
  }

  void String(const std::string_view value)
  {
    U64(static_cast<std::uint64_t>(value.size()));
    if (!value.empty()) Raw(value.data(), value.size());
  }

  void DigestValue(const Digest& value) { Raw(value.data(), value.size()); }

  const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }
  std::vector<std::uint8_t> Take() noexcept { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class ByteReader final
{
public:
  ByteReader(const std::uint8_t* data, const std::size_t length) noexcept
      : data_(data), length_(length)
  {}

  bool U8(std::uint8_t& value) noexcept
  {
    if (remaining() < 1) return false;
    value = data_[cursor_++];
    return true;
  }

  bool U32(std::uint32_t& value) noexcept
  {
    if (remaining() < 4) return false;
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
      value |= static_cast<std::uint32_t>(data_[cursor_++]) << shift;
    return true;
  }

  bool I32(std::int32_t& value) noexcept
  {
    std::uint32_t bits = 0;
    if (!U32(bits)) return false;
    value = static_cast<std::int32_t>(bits);
    return true;
  }

  bool U64(std::uint64_t& value) noexcept
  {
    if (remaining() < 8) return false;
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
      value |= static_cast<std::uint64_t>(data_[cursor_++]) << shift;
    return true;
  }

  bool F64(double& value) noexcept
  {
    std::uint64_t bits = 0;
    if (!U64(bits)) return false;
    value = BitsDouble(bits);
    return true;
  }

  bool Raw(void* destination, const std::size_t length) noexcept
  {
    if (remaining() < length) return false;
    if (length != 0) std::memcpy(destination, data_ + cursor_, length);
    cursor_ += length;
    return true;
  }

  bool String(std::string& value, const std::size_t maximum = 4096)
  {
    std::uint64_t length = 0;
    if (!U64(length) || length > maximum || remaining() < length) return false;
    value.assign(reinterpret_cast<const char*>(data_ + cursor_),
                 static_cast<std::size_t>(length));
    cursor_ += static_cast<std::size_t>(length);
    return true;
  }

  bool DigestValue(Digest& value) noexcept
  {
    return Raw(value.data(), value.size());
  }

  std::size_t remaining() const noexcept { return length_ - cursor_; }
  bool done() const noexcept { return cursor_ == length_; }

private:
  const std::uint8_t* data_ = nullptr;
  std::size_t length_ = 0;
  std::size_t cursor_ = 0;
};

inline void AppendHash(ByteWriter& writer,
                       const ::dg::hdg::runtime::StateHash& hash)
{
  writer.Raw(hash.bytes.data(), hash.bytes.size());
}

inline bool ReadHash(ByteReader& reader,
                     ::dg::hdg::runtime::StateHash& hash) noexcept
{
  return reader.Raw(hash.bytes.data(), hash.bytes.size());
}

inline void AppendCoordinates(
    ByteWriter& writer, const ::dg::hdg::runtime::StateCoordinates& value)
{
  writer.F64(value.time);
  writer.U64(value.step);
  writer.U64(value.generation);
  writer.U64(value.cursor);
  writer.F64(value.dt);
}

inline bool ReadCoordinates(
    ByteReader& reader, ::dg::hdg::runtime::StateCoordinates& value) noexcept
{
  return reader.F64(value.time) && reader.U64(value.step) &&
      reader.U64(value.generation) && reader.U64(value.cursor) &&
      reader.F64(value.dt);
}

inline void AppendCensus(ByteWriter& writer,
                         const cpu::Rank2OwnershipCensus& value)
{
  writer.I32(value.rank);
  writer.I32(value.ranks);
  writer.I32(value.local_elements);
  writer.I32(value.local_interior_unknowns);
  writer.I32(value.local_faces);
  writer.I32(value.owned_unique_faces);
  writer.I32(value.scalar_owned_true_dofs);
  writer.I32(value.combined_owned_trace_dofs);
  writer.I32(value.combined_ghost_trace_dofs);
  writer.I32(value.global_unique_faces);
  writer.I32(value.global_combined_true_dofs);
}

inline bool ReadCensus(ByteReader& reader,
                       cpu::Rank2OwnershipCensus& value) noexcept
{
  return reader.I32(value.rank) && reader.I32(value.ranks) &&
      reader.I32(value.local_elements) &&
      reader.I32(value.local_interior_unknowns) &&
      reader.I32(value.local_faces) &&
      reader.I32(value.owned_unique_faces) &&
      reader.I32(value.scalar_owned_true_dofs) &&
      reader.I32(value.combined_owned_trace_dofs) &&
      reader.I32(value.combined_ghost_trace_dofs) &&
      reader.I32(value.global_unique_faces) &&
      reader.I32(value.global_combined_true_dofs);
}

inline void AppendDirectoryIdentity(ByteWriter& writer,
                                    const DirectoryIdentity& value)
{
  writer.U64(value.device);
  writer.U64(value.inode);
}

inline bool ReadDirectoryIdentity(ByteReader& reader,
                                  DirectoryIdentity& value) noexcept
{
  return reader.U64(value.device) && reader.U64(value.inode);
}

inline void AppendVector(ByteWriter& writer,
                         const std::vector<double>& values)
{
  writer.U64(static_cast<std::uint64_t>(values.size()));
  for (const double value : values) writer.F64(value);
}

inline bool ReadVector(ByteReader& reader, std::vector<double>& values,
                       const std::size_t exact_length)
{
  std::uint64_t length = 0;
  if (!reader.U64(length) || length != exact_length) return false;
  std::vector<double> candidate(exact_length);
  for (double& value : candidate)
    if (!reader.F64(value)) return false;
  values = std::move(candidate);
  return true;
}

inline void AppendState(ByteWriter& writer,
                        const cpu::PersistentRank2TrajectoryState& value)
{
  AppendVector(writer, value.u);
  AppendVector(writer, value.sigma);
  AppendVector(writer, value.z);
  AppendVector(writer, value.uhat);
  AppendVector(writer, value.sigmahat);
}

inline bool ReadState(ByteReader& reader,
                      cpu::PersistentRank2TrajectoryState& value,
                      const int rank)
{
  const std::size_t uhat = rank == 0 ? 576U : 448U;
  const std::size_t sigmahat = rank == 0 ? 144U : 112U;
  return ReadVector(reader, value.u, 512) &&
      ReadVector(reader, value.sigma, 128) &&
      ReadVector(reader, value.z, 256) &&
      ReadVector(reader, value.uhat, uhat) &&
      ReadVector(reader, value.sigmahat, sigmahat);
}

struct CpuStepSummary final
{
  std::int32_t status = 0;
  ::dg::hdg::runtime::StateCoordinates coordinates{};
  std::int32_t accepted_steps = 0;
  std::int32_t attempted_steps = 0;
  std::int32_t retry_attempts = 0;
  std::int32_t attempt_count = 0;
  std::int32_t stage_count = 0;
  std::int32_t newton_iterations = 0;
  std::int32_t gmres_solves = 0;
  std::int32_t gmres_iterations = 0;
  std::int32_t line_search_trials = 0;
  double entry_residual_norm = 0.0;
  double final_residual_norm = 0.0;
  double trace_true_residual_norm = 0.0;
  double trace_right_hand_side_norm = 0.0;
  double maximum_local_application_residual = 0.0;
  double nonlinear_residual_bound = 0.0;
  double stage_residual_norm = 0.0;
  double condensed_rhs_norm = 0.0;
  double trace_solution_norm = 0.0;
  double recovery_norm = 0.0;
  std::array<double, cpu::state_components> committed_integrals{};
  std::array<double, cpu::state_components> conserved_integrals{};
};

struct CpuManifestStep final
{
  CpuStepSummary summary{};
  cpu::PersistentRank2TrajectoryState state{};
  bool operator_valid = false;
  bool initialization_operator = false;
  std::int32_t operator_iteration = -1;
  double operator_entry_residual_norm = 0.0;
  std::int32_t operator_trace_iterations = -1;
  double operator_trace_true_residual_norm = 0.0;
  double operator_trace_right_hand_side_norm = 0.0;
  double operator_maximum_local_application_residual = 0.0;
  cpu::PersistentRank2TrajectoryState operator_entry{};
  cpu::PersistentRank2TrajectoryState recovered_increment{};
  std::vector<double> nonlinear_residual;
  std::vector<double> jacobian_action_input;
  std::vector<double> jacobian_action_result;
  std::vector<double> condensed_right_hand_side;
  std::vector<double> true_trace_solution;
  bool operator_has_integrals = false;
  std::array<double, cpu::state_components> operator_committed_integrals{};
  std::array<double, cpu::state_components> operator_conserved_integrals{};
};

struct CpuReferenceManifest final
{
  std::string run_id;
  std::string executable_sha256;
  DirectoryIdentity directory{};
  std::int32_t rank = -1;
  cpu::Rank2OwnershipCensus census{};
  std::array<CpuManifestStep, cpu::persistent_required_steps + 1> steps{};
  ::dg::hdg::runtime::StateCoordinates checkpoint_coordinates{};
  cpu::Rank2TrajectoryStateHashes checkpoint_hashes{};
  ::dg::hdg::runtime::StateHash checkpoint_manifest_hash{};
};

inline CpuStepSummary CaptureCpuStepSummary(
    const cpu::PersistentRank2TrajectoryEvidence& evidence,
    const int step) noexcept
{
  CpuStepSummary result{};
  result.status = static_cast<std::int32_t>(evidence.status);
  result.coordinates = evidence.coordinates;
  result.accepted_steps = evidence.accepted_steps;
  result.attempted_steps = evidence.attempted_steps;
  result.retry_attempts = evidence.retry_attempts;
  result.attempt_count = evidence.attempt_count;
  result.stage_count = evidence.step_evidence_count;
  if (step > 0 && step <= evidence.attempt_count &&
      step <= evidence.step_evidence_count)
  {
    const auto& attempt = evidence.attempts[static_cast<std::size_t>(step - 1)];
    const auto& stage = evidence.stages[static_cast<std::size_t>(step - 1)];
    result.newton_iterations = attempt.newton_iterations;
    result.gmres_solves = attempt.gmres_solves;
    result.gmres_iterations = attempt.gmres_iterations;
    result.line_search_trials = attempt.line_search_trials;
    result.entry_residual_norm = attempt.entry_residual_norm;
    result.final_residual_norm = attempt.final_residual_norm;
    result.trace_true_residual_norm = attempt.trace_true_residual_norm;
    result.trace_right_hand_side_norm = attempt.trace_right_hand_side_norm;
    result.maximum_local_application_residual =
        attempt.maximum_local_application_residual;
    result.nonlinear_residual_bound = attempt.nonlinear_residual_bound;
    result.stage_residual_norm = stage.residual_norm;
    result.condensed_rhs_norm = stage.condensed_rhs_norm;
    result.trace_solution_norm = stage.trace_solution_norm;
    result.recovery_norm = stage.recovery_norm;
    result.committed_integrals = stage.committed_integrals;
    result.conserved_integrals = stage.conserved_integrals;
  }
  return result;
}

inline bool CaptureCpuOperatorSnapshot(
    const cpu::PersistentRank2TrajectoryEvidence& evidence,
    const int step, CpuManifestStep& destination) noexcept
{
  if (step < 0 || step > cpu::persistent_required_steps) return false;
  const cpu::Rank2TrajectoryNewtonEvidence* source = nullptr;
  if (step == 0)
  {
    source = &evidence.initialization_operator;
    destination.initialization_operator = true;
  }
  else
  {
    const auto& attempt =
        evidence.attempts[static_cast<std::size_t>(step - 1)];
    if (attempt.target_step != step || attempt.newton_iterations <= 0 ||
        attempt.newton_iterations > cpu::persistent_newton_maximum_iterations)
      return false;
    source = &attempt.newton[static_cast<std::size_t>(
        attempt.newton_iterations - 1)];
    const auto& stage = evidence.stages[static_cast<std::size_t>(step - 1)];
    destination.operator_has_integrals = true;
    destination.operator_committed_integrals = stage.committed_integrals;
    destination.operator_conserved_integrals = stage.conserved_integrals;
  }
  if (source == nullptr || !source->attempted || !source->completed)
    return false;
  destination.operator_valid = true;
  destination.operator_iteration = source->iteration;
  destination.operator_entry_residual_norm = source->entry_residual_norm;
  destination.operator_trace_iterations = source->trace_iterations;
  destination.operator_trace_true_residual_norm =
      source->trace_true_residual_norm;
  destination.operator_trace_right_hand_side_norm =
      source->trace_right_hand_side_norm;
  destination.operator_maximum_local_application_residual =
      source->maximum_local_application_residual;
  destination.operator_entry = source->entry;
  destination.recovered_increment = source->recovered_increment;
  destination.nonlinear_residual = source->nonlinear_residual;
  destination.jacobian_action_input = source->jacobian_action_input;
  destination.jacobian_action_result = source->jacobian_action_result;
  destination.condensed_right_hand_side =
      source->condensed_right_hand_side;
  destination.true_trace_solution = source->true_trace_solution;
  return true;
}

inline void AppendOperatorSnapshot(ByteWriter& writer,
                                   const CpuManifestStep& value)
{
  writer.U8(value.operator_valid ? 1U : 0U);
  writer.U8(value.initialization_operator ? 1U : 0U);
  writer.I32(value.operator_iteration);
  writer.F64(value.operator_entry_residual_norm);
  writer.I32(value.operator_trace_iterations);
  writer.F64(value.operator_trace_true_residual_norm);
  writer.F64(value.operator_trace_right_hand_side_norm);
  writer.F64(value.operator_maximum_local_application_residual);
  AppendState(writer, value.operator_entry);
  AppendState(writer, value.recovered_increment);
  AppendVector(writer, value.nonlinear_residual);
  AppendVector(writer, value.jacobian_action_input);
  AppendVector(writer, value.jacobian_action_result);
  AppendVector(writer, value.condensed_right_hand_side);
  AppendVector(writer, value.true_trace_solution);
  writer.U8(value.operator_has_integrals ? 1U : 0U);
  for (const double item : value.operator_committed_integrals)
    writer.F64(item);
  for (const double item : value.operator_conserved_integrals)
    writer.F64(item);
}

inline bool ReadOperatorSnapshot(ByteReader& reader, CpuManifestStep& value,
                                 const int rank)
{
  std::uint8_t valid = 0;
  std::uint8_t initialization = 0;
  std::uint8_t has_integrals = 0;
  const std::size_t owned = rank == 0 ? 720U : 560U;
  if (!reader.U8(valid) || !reader.U8(initialization) || valid > 1U ||
      initialization > 1U || !reader.I32(value.operator_iteration) ||
      !reader.F64(value.operator_entry_residual_norm) ||
      !reader.I32(value.operator_trace_iterations) ||
      !reader.F64(value.operator_trace_true_residual_norm) ||
      !reader.F64(value.operator_trace_right_hand_side_norm) ||
      !reader.F64(value.operator_maximum_local_application_residual) ||
      !ReadState(reader, value.operator_entry, rank) ||
      !ReadState(reader, value.recovered_increment, rank) ||
      !ReadVector(reader, value.nonlinear_residual, 896U + owned) ||
      !ReadVector(reader, value.jacobian_action_input, 896U + owned) ||
      !ReadVector(reader, value.jacobian_action_result, 896U + owned) ||
      !ReadVector(reader, value.condensed_right_hand_side, owned) ||
      !ReadVector(reader, value.true_trace_solution, owned) ||
      !reader.U8(has_integrals) || has_integrals > 1U)
    return false;
  value.operator_valid = valid != 0;
  value.initialization_operator = initialization != 0;
  value.operator_has_integrals = has_integrals != 0;
  for (double& item : value.operator_committed_integrals)
    if (!reader.F64(item)) return false;
  for (double& item : value.operator_conserved_integrals)
    if (!reader.F64(item)) return false;
  return true;
}

inline void AppendStepSummary(ByteWriter& writer,
                              const CpuStepSummary& value)
{
  writer.I32(value.status);
  AppendCoordinates(writer, value.coordinates);
  writer.I32(value.accepted_steps);
  writer.I32(value.attempted_steps);
  writer.I32(value.retry_attempts);
  writer.I32(value.attempt_count);
  writer.I32(value.stage_count);
  writer.I32(value.newton_iterations);
  writer.I32(value.gmres_solves);
  writer.I32(value.gmres_iterations);
  writer.I32(value.line_search_trials);
  writer.F64(value.entry_residual_norm);
  writer.F64(value.final_residual_norm);
  writer.F64(value.trace_true_residual_norm);
  writer.F64(value.trace_right_hand_side_norm);
  writer.F64(value.maximum_local_application_residual);
  writer.F64(value.nonlinear_residual_bound);
  writer.F64(value.stage_residual_norm);
  writer.F64(value.condensed_rhs_norm);
  writer.F64(value.trace_solution_norm);
  writer.F64(value.recovery_norm);
  for (const double item : value.committed_integrals) writer.F64(item);
  for (const double item : value.conserved_integrals) writer.F64(item);
}

inline bool ReadStepSummary(ByteReader& reader, CpuStepSummary& value) noexcept
{
  if (!reader.I32(value.status) ||
      !ReadCoordinates(reader, value.coordinates) ||
      !reader.I32(value.accepted_steps) ||
      !reader.I32(value.attempted_steps) ||
      !reader.I32(value.retry_attempts) ||
      !reader.I32(value.attempt_count) || !reader.I32(value.stage_count) ||
      !reader.I32(value.newton_iterations) ||
      !reader.I32(value.gmres_solves) ||
      !reader.I32(value.gmres_iterations) ||
      !reader.I32(value.line_search_trials) ||
      !reader.F64(value.entry_residual_norm) ||
      !reader.F64(value.final_residual_norm) ||
      !reader.F64(value.trace_true_residual_norm) ||
      !reader.F64(value.trace_right_hand_side_norm) ||
      !reader.F64(value.maximum_local_application_residual) ||
      !reader.F64(value.nonlinear_residual_bound) ||
      !reader.F64(value.stage_residual_norm) ||
      !reader.F64(value.condensed_rhs_norm) ||
      !reader.F64(value.trace_solution_norm) ||
      !reader.F64(value.recovery_norm))
    return false;
  for (double& item : value.committed_integrals)
    if (!reader.F64(item)) return false;
  for (double& item : value.conserved_integrals)
    if (!reader.F64(item)) return false;
  return true;
}

inline void AppendRank2Hashes(ByteWriter& writer,
                              const cpu::Rank2TrajectoryStateHashes& value)
{
  AppendHash(writer, value.u);
  AppendHash(writer, value.sigma);
  AppendHash(writer, value.z);
  AppendHash(writer, value.uhat);
  AppendHash(writer, value.sigmahat);
  AppendHash(writer, value.interior);
  AppendHash(writer, value.trace);
  AppendHash(writer, value.shard);
}

inline bool ReadRank2Hashes(ByteReader& reader,
                            cpu::Rank2TrajectoryStateHashes& value) noexcept
{
  return ReadHash(reader, value.u) && ReadHash(reader, value.sigma) &&
      ReadHash(reader, value.z) && ReadHash(reader, value.uhat) &&
      ReadHash(reader, value.sigmahat) && ReadHash(reader, value.interior) &&
      ReadHash(reader, value.trace) && ReadHash(reader, value.shard);
}

inline bool NonzeroHash(const ::dg::hdg::runtime::StateHash& value) noexcept
{
  return std::any_of(value.bytes.begin(), value.bytes.end(),
                     [](const std::uint8_t byte) { return byte != 0; });
}

inline bool ExpectedCensus(const cpu::Rank2OwnershipCensus& census,
                           const int rank) noexcept
{
  const int scalar_owned = rank == 0 ? 144 : 112;
  const int trace_owned = rank == 0 ? 720 : 560;
  const int ghosts = rank == 0 ? 0 : 160;
  return census.rank == rank && census.ranks == 2 &&
      census.local_elements == 32 && census.local_interior_unknowns == 896 &&
      census.local_faces == 72 &&
      census.owned_unique_faces == scalar_owned / 2 &&
      census.scalar_owned_true_dofs == scalar_owned &&
      census.combined_owned_trace_dofs == trace_owned &&
      census.combined_ghost_trace_dofs == ghosts &&
      census.global_unique_faces == 128 &&
      census.global_combined_true_dofs == 1280;
}

inline bool ValidCpuManifest(const CpuReferenceManifest& value) noexcept
{
  if (!SafeRunId(value.run_id) || value.executable_sha256.size() != 64 ||
      value.directory.inode == 0 || value.rank < 0 || value.rank > 1 ||
      DoubleBits(1.0) != manifest_double_marker ||
      !ExpectedCensus(value.census, value.rank))
    return false;
  Digest ignored{};
  if (!ParseHexDigest(value.executable_sha256, ignored)) return false;
  for (int step = 0; step <= cpu::persistent_required_steps; ++step)
  {
    const auto& record = value.steps[static_cast<std::size_t>(step)];
    const auto& summary = record.summary;
    const bool status_ok = step == 0
        ? summary.status == static_cast<std::int32_t>(cpu::TrajectoryStatus::ready)
        : (summary.status == static_cast<std::int32_t>(
               cpu::TrajectoryStatus::HdgSc25Accepted) ||
           summary.status == static_cast<std::int32_t>(
               cpu::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry));
    if (!status_ok || summary.coordinates.step !=
                          static_cast<std::uint64_t>(step) ||
        summary.coordinates.generation != static_cast<std::uint64_t>(step) ||
        summary.coordinates.cursor != static_cast<std::uint64_t>(step) ||
        summary.coordinates.dt != cpu::persistent_dt ||
        summary.coordinates.time != step * cpu::persistent_dt ||
        summary.accepted_steps != step || summary.attempted_steps != step ||
        summary.retry_attempts != 0 || summary.attempt_count != step ||
        summary.stage_count != step)
      return false;
    if (record.state.u.size() != 512 || record.state.sigma.size() != 128 ||
        record.state.z.size() != 256 ||
        record.state.uhat.size() != (value.rank == 0 ? 576U : 448U) ||
        record.state.sigmahat.size() != (value.rank == 0 ? 144U : 112U))
      return false;
    const auto finite_vector = [](const std::vector<double>& values)
    {
      return std::all_of(values.begin(), values.end(),
                         [](const double item) { return std::isfinite(item); });
    };
    if (!finite_vector(record.state.u) || !finite_vector(record.state.sigma) ||
        !finite_vector(record.state.z) || !finite_vector(record.state.uhat) ||
        !finite_vector(record.state.sigmahat))
      return false;
    const std::size_t owned = value.rank == 0 ? 720U : 560U;
    const auto valid_state = [&](
        const cpu::PersistentRank2TrajectoryState& state)
    {
      return state.u.size() == 512U && state.sigma.size() == 128U &&
          state.z.size() == 256U &&
          state.uhat.size() == (value.rank == 0 ? 576U : 448U) &&
          state.sigmahat.size() == (value.rank == 0 ? 144U : 112U) &&
          finite_vector(state.u) && finite_vector(state.sigma) &&
          finite_vector(state.z) && finite_vector(state.uhat) &&
          finite_vector(state.sigmahat);
    };
    const std::array<double, 4> operator_scalars{{
        record.operator_entry_residual_norm,
        record.operator_trace_true_residual_norm,
        record.operator_trace_right_hand_side_norm,
        record.operator_maximum_local_application_residual}};
    if (!record.operator_valid ||
        record.initialization_operator != (step == 0) ||
        record.operator_iteration < 0 ||
        record.operator_trace_iterations < 0 ||
        !std::all_of(operator_scalars.begin(), operator_scalars.end(),
                     [](const double item) { return std::isfinite(item); }) ||
        !valid_state(record.operator_entry) ||
        !valid_state(record.recovered_increment) ||
        record.nonlinear_residual.size() != 896U + owned ||
        record.jacobian_action_input.size() != 896U + owned ||
        record.jacobian_action_result.size() != 896U + owned ||
        record.condensed_right_hand_side.size() != owned ||
        record.true_trace_solution.size() != owned ||
        !finite_vector(record.nonlinear_residual) ||
        !finite_vector(record.jacobian_action_input) ||
        !finite_vector(record.jacobian_action_result) ||
        !finite_vector(record.condensed_right_hand_side) ||
        !finite_vector(record.true_trace_solution) ||
        record.operator_has_integrals != (step > 0) ||
        !std::all_of(record.operator_committed_integrals.begin(),
                     record.operator_committed_integrals.end(),
                     [](const double item) { return std::isfinite(item); }) ||
        !std::all_of(record.operator_conserved_integrals.begin(),
                     record.operator_conserved_integrals.end(),
                     [](const double item) { return std::isfinite(item); }))
      return false;
    if (step > 0)
    {
      const std::array<double, 10> scalars{{
          summary.entry_residual_norm, summary.final_residual_norm,
          summary.trace_true_residual_norm,
          summary.trace_right_hand_side_norm,
          summary.maximum_local_application_residual,
          summary.nonlinear_residual_bound, summary.stage_residual_norm,
          summary.condensed_rhs_norm, summary.trace_solution_norm,
          summary.recovery_norm}};
      if (summary.newton_iterations < 0 || summary.gmres_solves < 0 ||
          summary.gmres_iterations < 0 || summary.line_search_trials < 0 ||
          !std::all_of(scalars.begin(), scalars.end(),
                       [](const double item) { return std::isfinite(item); }) ||
          !std::all_of(summary.committed_integrals.begin(),
                       summary.committed_integrals.end(),
                       [](const double item) { return std::isfinite(item); }) ||
          !std::all_of(summary.conserved_integrals.begin(),
                       summary.conserved_integrals.end(),
                       [](const double item) { return std::isfinite(item); }))
        return false;
    }
  }
  return value.checkpoint_coordinates.step == 10 &&
      value.checkpoint_coordinates.generation == 10 &&
      value.checkpoint_coordinates.cursor == 10 &&
      value.checkpoint_coordinates.dt == cpu::persistent_dt &&
      value.checkpoint_coordinates.time == 10 * cpu::persistent_dt &&
      NonzeroHash(value.checkpoint_hashes.u) &&
      NonzeroHash(value.checkpoint_hashes.sigma) &&
      NonzeroHash(value.checkpoint_hashes.z) &&
      NonzeroHash(value.checkpoint_hashes.uhat) &&
      NonzeroHash(value.checkpoint_hashes.sigmahat) &&
      NonzeroHash(value.checkpoint_hashes.interior) &&
      NonzeroHash(value.checkpoint_hashes.trace) &&
      NonzeroHash(value.checkpoint_hashes.shard) &&
      NonzeroHash(value.checkpoint_manifest_hash);
}

inline std::vector<std::uint8_t> Envelope(
    const std::array<std::uint8_t, 8>& magic,
    const std::array<std::uint8_t, 8>& terminal,
    const std::vector<std::uint8_t>& payload)
{
  if (DoubleBits(1.0) != manifest_double_marker ||
      payload.size() > maximum_manifest_bytes)
    return {};
  ByteWriter writer;
  writer.Raw(magic.data(), magic.size());
  writer.U32(manifest_schema_version);
  writer.U32(manifest_endian_marker);
  writer.U64(manifest_double_marker);
  writer.U64(static_cast<std::uint64_t>(payload.size()));
  if (!payload.empty()) writer.Raw(payload.data(), payload.size());
  const Digest checksum = ::dg::hdg::runtime::ComputeSha256(
      writer.bytes().data(), writer.bytes().size());
  writer.DigestValue(checksum);
  writer.Raw(terminal.data(), terminal.size());
  return writer.Take();
}

inline bool OpenEnvelope(const std::vector<std::uint8_t>& bytes,
                         const std::array<std::uint8_t, 8>& magic,
                         const std::array<std::uint8_t, 8>& terminal,
                         const std::uint8_t*& payload,
                         std::size_t& payload_size) noexcept
{
  constexpr std::size_t header_size = 8 + 4 + 4 + 8 + 8;
  constexpr std::size_t suffix_size = 32 + 8;
  if (bytes.size() < header_size + suffix_size ||
      bytes.size() > maximum_manifest_bytes ||
      DoubleBits(1.0) != manifest_double_marker ||
      !std::equal(magic.begin(), magic.end(), bytes.begin()))
    return false;
  ByteReader header(bytes.data() + 8, header_size - 8);
  std::uint32_t schema = 0;
  std::uint32_t endian = 0;
  std::uint64_t double_marker = 0;
  std::uint64_t encoded_payload_size = 0;
  if (!header.U32(schema) || !header.U32(endian) ||
      !header.U64(double_marker) || !header.U64(encoded_payload_size) ||
      !header.done() || schema != manifest_schema_version ||
      endian != manifest_endian_marker ||
      double_marker != manifest_double_marker ||
      encoded_payload_size > maximum_manifest_bytes ||
      encoded_payload_size != bytes.size() - header_size - suffix_size)
    return false;
  const std::size_t checksum_offset =
      header_size + static_cast<std::size_t>(encoded_payload_size);
  const Digest observed = ::dg::hdg::runtime::ComputeSha256(
      bytes.data(), checksum_offset);
  if (!std::equal(observed.begin(), observed.end(),
                  bytes.begin() + static_cast<std::ptrdiff_t>(checksum_offset)) ||
      !std::equal(terminal.begin(), terminal.end(),
                  bytes.begin() +
                      static_cast<std::ptrdiff_t>(checksum_offset + 32)))
    return false;
  payload = bytes.data() + header_size;
  payload_size = static_cast<std::size_t>(encoded_payload_size);
  return true;
}

inline std::vector<std::uint8_t> EncodeCpuManifest(
    const CpuReferenceManifest& value)
{
  if (!ValidCpuManifest(value)) return {};
  ByteWriter payload;
  payload.String(value.run_id);
  payload.String(value.executable_sha256);
  payload.String(accepted_cpu_oracle_receipt);
  payload.String(cpu::projection_id);
  payload.String(cpu::cpu_rank2_tuple_id);
  payload.String(cpu::persistent_rank2_fixture_id);
  payload.String(mfem_pin);
  AppendDirectoryIdentity(payload, value.directory);
  payload.I32(value.rank);
  payload.I32(2);
  payload.I32(1);
  payload.I32(cpu::persistent_elements_per_direction);
  payload.F64(cpu::persistent_dt);
  payload.I32(cpu::persistent_trace_restart_dimension);
  payload.I32(cpu::persistent_trace_maximum_iterations);
  payload.F64(trace_relative_tolerance);
  payload.F64(trace_absolute_tolerance);
  payload.I32(0);
  payload.I32(0);
  payload.I32(1);
  AppendCensus(payload, value.census);
  payload.U32(cpu::persistent_required_steps + 1);
  for (const auto& step : value.steps)
  {
    AppendStepSummary(payload, step.summary);
    AppendState(payload, step.state);
    AppendOperatorSnapshot(payload, step);
  }
  AppendCoordinates(payload, value.checkpoint_coordinates);
  AppendRank2Hashes(payload, value.checkpoint_hashes);
  AppendHash(payload, value.checkpoint_manifest_hash);
  static constexpr std::array<std::uint8_t, 8> magic{{
      'D', 'G', 'C', '3', 'M', '0', '0', '1'}};
  static constexpr std::array<std::uint8_t, 8> terminal{{
      'C', '3', 'M', 'E', 'N', 'D', '0', '1'}};
  return Envelope(magic, terminal, payload.bytes());
}

inline bool DecodeCpuManifest(const std::vector<std::uint8_t>& bytes,
                              const std::string_view expected_run_id,
                              const std::string_view expected_executable,
                              const int expected_rank,
                              const DirectoryIdentity& expected_directory,
                              CpuReferenceManifest& value)
{
  static constexpr std::array<std::uint8_t, 8> magic{{
      'D', 'G', 'C', '3', 'M', '0', '0', '1'}};
  static constexpr std::array<std::uint8_t, 8> terminal{{
      'C', '3', 'M', 'E', 'N', 'D', '0', '1'}};
  const std::uint8_t* payload = nullptr;
  std::size_t payload_size = 0;
  if (!OpenEnvelope(bytes, magic, terminal, payload, payload_size)) return false;
  ByteReader reader(payload, payload_size);
  CpuReferenceManifest candidate{};
  std::string receipt;
  std::string projection;
  std::string tuple;
  std::string scenario;
  std::string mfem;
  std::int32_t ranks = 0;
  std::int32_t polynomial_degree = 0;
  std::int32_t elements_per_direction = 0;
  double dt = 0.0;
  std::int32_t restart = 0;
  std::int32_t maximum_iterations = 0;
  double relative_tolerance = 0.0;
  double absolute_tolerance = 0.0;
  std::int32_t ilu_type = -1;
  std::int32_t ilu_fill = -1;
  std::int32_t ilu_reordering = -1;
  std::uint32_t states = 0;
  if (!reader.String(candidate.run_id, 128) ||
      !reader.String(candidate.executable_sha256, 64) ||
      !reader.String(receipt) || !reader.String(projection, 128) ||
      !reader.String(tuple, 128) || !reader.String(scenario, 128) ||
      !reader.String(mfem, 64) ||
      !ReadDirectoryIdentity(reader, candidate.directory) ||
      !reader.I32(candidate.rank) || !reader.I32(ranks) ||
      !reader.I32(polynomial_degree) ||
      !reader.I32(elements_per_direction) || !reader.F64(dt) ||
      !reader.I32(restart) || !reader.I32(maximum_iterations) ||
      !reader.F64(relative_tolerance) ||
      !reader.F64(absolute_tolerance) || !reader.I32(ilu_type) ||
      !reader.I32(ilu_fill) || !reader.I32(ilu_reordering) ||
      !ReadCensus(reader, candidate.census) || !reader.U32(states))
    return false;
  if (candidate.run_id != expected_run_id ||
      candidate.executable_sha256 != expected_executable ||
      receipt != accepted_cpu_oracle_receipt ||
      projection != cpu::projection_id || tuple != cpu::cpu_rank2_tuple_id ||
      scenario != cpu::persistent_rank2_fixture_id || mfem != mfem_pin ||
      candidate.directory != expected_directory ||
      candidate.rank != expected_rank || ranks != 2 || polynomial_degree != 1 ||
      elements_per_direction != cpu::persistent_elements_per_direction ||
      dt != cpu::persistent_dt ||
      restart != cpu::persistent_trace_restart_dimension ||
      maximum_iterations != cpu::persistent_trace_maximum_iterations ||
      relative_tolerance != trace_relative_tolerance ||
      absolute_tolerance != trace_absolute_tolerance ||
      ilu_type != 0 || ilu_fill != 0 || ilu_reordering != 1 ||
      states != cpu::persistent_required_steps + 1)
    return false;
  for (auto& step : candidate.steps)
  {
    if (!ReadStepSummary(reader, step.summary) ||
        !ReadState(reader, step.state, expected_rank) ||
        !ReadOperatorSnapshot(reader, step, expected_rank))
      return false;
  }
  if (!ReadCoordinates(reader, candidate.checkpoint_coordinates) ||
      !ReadRank2Hashes(reader, candidate.checkpoint_hashes) ||
      !ReadHash(reader, candidate.checkpoint_manifest_hash) || !reader.done() ||
      !ValidCpuManifest(candidate))
    return false;
  value = std::move(candidate);
  return true;
}

inline std::string RankFilename(const int rank)
{
  return std::string("rank-") + std::to_string(rank) + ".c3m";
}

inline std::string RankTemporaryFilename(const int rank)
{
  return std::string("rank-") + std::to_string(rank) + ".c3m.tmp";
}

inline constexpr std::string_view complete_filename = "COMPLETE.c3m";
inline constexpr std::string_view complete_temporary_filename =
    "COMPLETE.c3m.tmp";

inline bool SafeManifestBasename(const std::string_view value) noexcept
{
  return value == "rank-0.c3m" || value == "rank-1.c3m" ||
      value == "rank-0.c3m.tmp" || value == "rank-1.c3m.tmp" ||
      value == complete_filename || value == complete_temporary_filename;
}

inline bool BasenameBuffer(const std::string_view value,
                           std::array<char, 32>& output) noexcept
{
  if (!SafeManifestBasename(value) || value.size() >= output.size())
    return false;
  output.fill('\0');
  std::copy(value.begin(), value.end(), output.begin());
  return true;
}

inline bool EntryAbsent(const DirectoryHandle& directory,
                        const std::string_view basename) noexcept
{
  if (!directory.Stable()) return false;
  std::array<char, 32> name{};
  if (!BasenameBuffer(basename, name)) return false;
  struct stat status{};
  if (::fstatat(directory.descriptor(), name.data(), &status,
                AT_SYMLINK_NOFOLLOW) == 0)
    return false;
  return errno == ENOENT;
}

inline bool ValidManifestFileStatus(const DirectoryHandle& directory,
                                    const struct stat& status) noexcept
{
  return directory.Stable() && S_ISREG(status.st_mode) &&
      status.st_uid == ::geteuid() && status.st_nlink == 1 &&
      status.st_size > 0 &&
      static_cast<std::uint64_t>(status.st_size) <= maximum_manifest_bytes &&
      static_cast<std::uint64_t>(status.st_dev) ==
          directory.identity().device &&
      (status.st_mode & (S_IRWXG | S_IRWXO)) == 0;
}

inline bool ReadFileAt(const DirectoryHandle& directory,
                       const std::string_view basename,
                       std::vector<std::uint8_t>& bytes)
{
  if (!directory.Stable()) return false;
  std::array<char, 32> name{};
  if (!BasenameBuffer(basename, name)) return false;
  const int descriptor = ::openat(
      directory.descriptor(), name.data(),
      O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (descriptor < 0) return false;
  struct stat before{};
  bool valid = ::fstat(descriptor, &before) == 0 &&
      ValidManifestFileStatus(directory, before);
  std::vector<std::uint8_t> candidate;
  if (valid) candidate.resize(static_cast<std::size_t>(before.st_size));
  std::size_t received = 0;
  while (valid && received < candidate.size())
  {
    const ssize_t count = ::read(descriptor, candidate.data() + received,
                                 candidate.size() - received);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0)
    {
      valid = false;
      break;
    }
    received += static_cast<std::size_t>(count);
  }
  std::uint8_t trailing = 0;
  ssize_t trailing_count = -1;
  while (valid)
  {
    trailing_count = ::read(descriptor, &trailing, 1);
    if (trailing_count < 0 && errno == EINTR) continue;
    break;
  }
  struct stat after{};
  valid = valid && trailing_count == 0 && ::fstat(descriptor, &after) == 0 &&
      ValidManifestFileStatus(directory, after) &&
      before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
      before.st_size == after.st_size && before.st_mode == after.st_mode &&
      received == candidate.size();
  if (::close(descriptor) != 0) valid = false;
  if (!valid) return false;
  bytes = std::move(candidate);
  return true;
}

inline bool WriteFileAt(const DirectoryHandle& directory,
                        const std::string_view basename,
                        const std::vector<std::uint8_t>& bytes)
{
  if (!directory.Stable() || bytes.empty() ||
      bytes.size() > maximum_manifest_bytes)
    return false;
  std::array<char, 32> name{};
  if (!BasenameBuffer(basename, name)) return false;
  const int descriptor = ::openat(
      directory.descriptor(), name.data(),
      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC,
      0600);
  if (descriptor < 0) return false;
  struct stat before{};
  bool valid = ::fstat(descriptor, &before) == 0 &&
      S_ISREG(before.st_mode) && before.st_uid == ::geteuid() &&
      before.st_nlink == 1 && before.st_size == 0 &&
      static_cast<std::uint64_t>(before.st_dev) == directory.identity().device &&
      (before.st_mode & (S_IRWXG | S_IRWXO)) == 0;
  std::size_t written = 0;
  while (valid && written < bytes.size())
  {
    const ssize_t count = ::write(descriptor, bytes.data() + written,
                                  bytes.size() - written);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0)
    {
      valid = false;
      break;
    }
    written += static_cast<std::size_t>(count);
  }
  if (valid && ::fsync(descriptor) != 0) valid = false;
  struct stat after{};
  valid = valid && ::fstat(descriptor, &after) == 0 &&
      ValidManifestFileStatus(directory, after) &&
      before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
      static_cast<std::uint64_t>(after.st_size) == bytes.size() &&
      written == bytes.size();
  if (::close(descriptor) != 0) valid = false;
  std::vector<std::uint8_t> readback;
  return valid && ReadFileAt(directory, basename, readback) &&
      readback == bytes;
}

inline bool AtomicRenameAt(const DirectoryHandle& directory,
                           const std::string_view temporary,
                           const std::string_view final) noexcept
{
  if (!directory.Stable()) return false;
  const bool valid_pair =
      (temporary == "rank-0.c3m.tmp" && final == "rank-0.c3m") ||
      (temporary == "rank-1.c3m.tmp" && final == "rank-1.c3m") ||
      (temporary == complete_temporary_filename &&
       final == complete_filename);
  if (!valid_pair || !EntryAbsent(directory, final)) return false;
  std::array<char, 32> temporary_name{};
  std::array<char, 32> final_name{};
  if (!BasenameBuffer(temporary, temporary_name) ||
      !BasenameBuffer(final, final_name))
    return false;
  struct stat before{};
  if (::fstatat(directory.descriptor(), temporary_name.data(), &before,
                AT_SYMLINK_NOFOLLOW) != 0 ||
      !ValidManifestFileStatus(directory, before))
    return false;
#if defined(SYS_renameat2)
  constexpr unsigned rename_noreplace = 1U;
  if (::syscall(SYS_renameat2, directory.descriptor(), temporary_name.data(),
                directory.descriptor(), final_name.data(),
                rename_noreplace) != 0 ||
      ::fsync(directory.descriptor()) != 0)
    return false;
#else
  return false;
#endif
  struct stat after{};
  return ::fstatat(directory.descriptor(), final_name.data(), &after,
                   AT_SYMLINK_NOFOLLOW) == 0 &&
      ValidManifestFileStatus(directory, after) &&
      before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
      before.st_size == after.st_size && EntryAbsent(directory, temporary);
}

struct CompleteEntry final
{
  std::string filename;
  std::uint64_t bytes = 0;
  Digest sha256{};
};

struct CompleteManifest final
{
  std::string run_id;
  std::string executable_sha256;
  DirectoryIdentity directory{};
  std::array<CompleteEntry, 2> shards{};
};

inline std::vector<std::uint8_t> EncodeComplete(
    const CompleteManifest& value)
{
  Digest executable_digest{};
  if (!SafeRunId(value.run_id) || value.executable_sha256.size() != 64 ||
      !ParseHexDigest(value.executable_sha256, executable_digest) ||
      value.directory.inode == 0 ||
      value.shards[0].filename != "rank-0.c3m" ||
      value.shards[1].filename != "rank-1.c3m" ||
      value.shards[0].bytes == 0 || value.shards[1].bytes == 0 ||
      value.shards[0].bytes > maximum_manifest_bytes ||
      value.shards[1].bytes > maximum_manifest_bytes)
    return {};
  ByteWriter payload;
  payload.String(value.run_id);
  payload.String(value.executable_sha256);
  payload.String(accepted_cpu_oracle_receipt);
  payload.String(cpu::projection_id);
  payload.String(cpu::cpu_rank2_tuple_id);
  payload.String(cpu::persistent_rank2_fixture_id);
  payload.String(mfem_pin);
  AppendDirectoryIdentity(payload, value.directory);
  payload.U32(2);
  for (const auto& shard : value.shards)
  {
    payload.String(shard.filename);
    payload.U64(shard.bytes);
    payload.DigestValue(shard.sha256);
  }
  static constexpr std::array<std::uint8_t, 8> magic{{
      'D', 'G', 'C', '3', 'C', '0', '0', '1'}};
  static constexpr std::array<std::uint8_t, 8> terminal{{
      'C', '3', 'C', 'E', 'N', 'D', '0', '1'}};
  return Envelope(magic, terminal, payload.bytes());
}

inline bool DecodeComplete(const std::vector<std::uint8_t>& bytes,
                           const std::string_view expected_run_id,
                           const DirectoryIdentity& expected_directory,
                           CompleteManifest& value)
{
  static constexpr std::array<std::uint8_t, 8> magic{{
      'D', 'G', 'C', '3', 'C', '0', '0', '1'}};
  static constexpr std::array<std::uint8_t, 8> terminal{{
      'C', '3', 'C', 'E', 'N', 'D', '0', '1'}};
  const std::uint8_t* payload = nullptr;
  std::size_t payload_size = 0;
  if (!OpenEnvelope(bytes, magic, terminal, payload, payload_size)) return false;
  ByteReader reader(payload, payload_size);
  CompleteManifest candidate{};
  std::string receipt;
  std::string projection;
  std::string tuple;
  std::string scenario;
  std::string mfem;
  std::uint32_t shard_count = 0;
  if (!reader.String(candidate.run_id, 128) ||
      !reader.String(candidate.executable_sha256, 64) ||
      !reader.String(receipt) || !reader.String(projection, 128) ||
      !reader.String(tuple, 128) || !reader.String(scenario, 128) ||
      !reader.String(mfem, 64) ||
      !ReadDirectoryIdentity(reader, candidate.directory) ||
      !reader.U32(shard_count) || shard_count != 2)
    return false;
  for (auto& shard : candidate.shards)
  {
    if (!reader.String(shard.filename, 32) || !reader.U64(shard.bytes) ||
        !reader.DigestValue(shard.sha256))
      return false;
  }
  Digest executable_digest{};
  if (!reader.done() || candidate.run_id != expected_run_id ||
      !ParseHexDigest(candidate.executable_sha256, executable_digest) ||
      receipt != accepted_cpu_oracle_receipt ||
      projection != cpu::projection_id || tuple != cpu::cpu_rank2_tuple_id ||
      scenario != cpu::persistent_rank2_fixture_id || mfem != mfem_pin ||
      candidate.directory != expected_directory ||
      candidate.shards[0].filename != "rank-0.c3m" ||
      candidate.shards[1].filename != "rank-1.c3m" ||
      candidate.shards[0].bytes == 0 || candidate.shards[1].bytes == 0 ||
      candidate.shards[0].bytes > maximum_manifest_bytes ||
      candidate.shards[1].bytes > maximum_manifest_bytes)
    return false;
  value = std::move(candidate);
  return true;
}

inline bool DecodeComplete(const std::vector<std::uint8_t>& bytes,
                           const std::string_view expected_run_id,
                           const std::string_view expected_executable,
                           const DirectoryIdentity& expected_directory,
                           CompleteManifest& value)
{
  CompleteManifest candidate{};
  if (!DecodeComplete(bytes, expected_run_id, expected_directory, candidate) ||
      candidate.executable_sha256 != expected_executable)
    return false;
  value = std::move(candidate);
  return true;
}

inline std::string DifferentDigestText(const std::string& value)
{
  std::string different = value;
  if (!different.empty()) different[0] = different[0] == '0' ? '1' : '0';
  return different;
}

inline bool CpuManifestProtocolNegatives(
    const std::vector<std::uint8_t>& bytes,
    const CpuReferenceManifest& expected)
{
  CpuReferenceManifest decoded{};
  if (!DecodeCpuManifest(bytes, expected.run_id, expected.executable_sha256,
                         expected.rank, expected.directory, decoded))
    return false;
  std::vector<std::uint8_t> truncated = bytes;
  truncated.pop_back();
  std::vector<std::uint8_t> trailing = bytes;
  trailing.push_back(0);
  std::vector<std::uint8_t> checksum_mutation = bytes;
  checksum_mutation[32] ^= 0x1U;
  DirectoryIdentity wrong_directory = expected.directory;
  ++wrong_directory.inode;
  CpuReferenceManifest rejected{};
  return !DecodeCpuManifest(truncated, expected.run_id,
                            expected.executable_sha256, expected.rank,
                            expected.directory, rejected) &&
      !DecodeCpuManifest(trailing, expected.run_id,
                         expected.executable_sha256, expected.rank,
                         expected.directory, rejected) &&
      !DecodeCpuManifest(checksum_mutation, expected.run_id,
                         expected.executable_sha256, expected.rank,
                         expected.directory, rejected) &&
      !DecodeCpuManifest(bytes, expected.run_id + "x",
                         expected.executable_sha256, expected.rank,
                         expected.directory, rejected) &&
      !DecodeCpuManifest(bytes, expected.run_id,
                         DifferentDigestText(expected.executable_sha256),
                         expected.rank, expected.directory, rejected) &&
      !DecodeCpuManifest(bytes, expected.run_id,
                         expected.executable_sha256, 1 - expected.rank,
                         expected.directory, rejected) &&
      !DecodeCpuManifest(bytes, expected.run_id,
                         expected.executable_sha256, expected.rank,
                         wrong_directory, rejected);
}

inline bool CompleteProtocolNegatives(const std::vector<std::uint8_t>& bytes,
                                      const CompleteManifest& expected)
{
  CompleteManifest decoded{};
  if (!DecodeComplete(bytes, expected.run_id, expected.executable_sha256,
                      expected.directory, decoded))
    return false;
  std::vector<std::uint8_t> truncated = bytes;
  truncated.pop_back();
  std::vector<std::uint8_t> trailing = bytes;
  trailing.push_back(0);
  std::vector<std::uint8_t> checksum_mutation = bytes;
  checksum_mutation[32] ^= 0x1U;
  DirectoryIdentity wrong_directory = expected.directory;
  ++wrong_directory.inode;
  CompleteManifest rejected{};
  return !DecodeComplete(truncated, expected.run_id,
                         expected.executable_sha256, expected.directory,
                         rejected) &&
      !DecodeComplete(trailing, expected.run_id,
                      expected.executable_sha256, expected.directory,
                      rejected) &&
      !DecodeComplete(checksum_mutation, expected.run_id,
                      expected.executable_sha256, expected.directory,
                      rejected) &&
      !DecodeComplete(bytes, expected.run_id + "x",
                      expected.executable_sha256, expected.directory,
                      rejected) &&
      !DecodeComplete(bytes, expected.run_id,
                      DifferentDigestText(expected.executable_sha256),
                      expected.directory, rejected) &&
      !DecodeComplete(bytes, expected.run_id,
                      expected.executable_sha256, wrong_directory,
                      rejected);
}

inline bool DirectoryCensus(const DirectoryHandle& directory,
                            const bool require_final_set) noexcept
{
  if (!directory.Stable()) return false;
  const int descriptor = ::openat(directory.descriptor(), ".",
                                  O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (descriptor < 0) return false;
  DIR* const stream = ::fdopendir(descriptor);
  if (stream == nullptr)
  {
    ::close(descriptor);
    return false;
  }
  std::array<bool, 3> seen{{false, false, false}};
  bool valid = true;
  errno = 0;
  while (valid)
  {
    struct dirent* const entry = ::readdir(stream);
    if (entry == nullptr)
    {
      if (errno != 0) valid = false;
      break;
    }
    const std::string_view name(entry->d_name);
    if (name == "." || name == "..") continue;
    if (!require_final_set)
    {
      valid = false;
      break;
    }
    std::size_t index = seen.size();
    if (name == "rank-0.c3m") index = 0;
    if (name == "rank-1.c3m") index = 1;
    if (name == complete_filename) index = 2;
    if (index == seen.size() || seen[index])
    {
      valid = false;
      break;
    }
    seen[index] = true;
  }
  if (::closedir(stream) != 0) valid = false;
  return valid && directory.Stable() &&
      (!require_final_set || std::all_of(seen.begin(), seen.end(),
                                         [](const bool value)
                                         { return value; }));
}

inline bool EmptyDirectory(const DirectoryHandle& directory) noexcept
{
  return DirectoryCensus(directory, false);
}

struct ManifestFileIdentity final
{
  std::uint64_t bytes = 0;
  Digest sha256{};
};

inline bool operator==(const ManifestFileIdentity& left,
                       const ManifestFileIdentity& right) noexcept
{
  return left.bytes == right.bytes && DigestEqual(left.sha256, right.sha256);
}

struct ManifestSetIdentity final
{
  DirectoryIdentity directory{};
  std::array<ManifestFileIdentity, 3> files{};
};

inline bool operator==(const ManifestSetIdentity& left,
                       const ManifestSetIdentity& right) noexcept
{
  return left.directory == right.directory && left.files == right.files;
}

inline bool CaptureManifestSetIdentity(const DirectoryHandle& directory,
                                       ManifestSetIdentity& identity)
{
  if (!DirectoryCensus(directory, true)) return false;
  static constexpr std::array<std::string_view, 3> filenames{{
      "rank-0.c3m", "rank-1.c3m", complete_filename}};
  ManifestSetIdentity candidate{};
  candidate.directory = directory.identity();
  for (std::size_t index = 0; index < filenames.size(); ++index)
  {
    std::vector<std::uint8_t> bytes;
    if (!ReadFileAt(directory, filenames[index], bytes)) return false;
    candidate.files[index].bytes = static_cast<std::uint64_t>(bytes.size());
    candidate.files[index].sha256 = ::dg::hdg::runtime::ComputeSha256(
        bytes.data(), bytes.size());
  }
  if (!DirectoryCensus(directory, true)) return false;
  identity = candidate;
  return true;
}

/** Hash the bytes of the kernel-bound live executable, never argv[0]. */
inline bool ReadLiveExecutableSha256(Digest& digest) noexcept
{
  const int descriptor = ::open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) return false;
  struct stat before{};
  bool valid = ::fstat(descriptor, &before) == 0 &&
      S_ISREG(before.st_mode) && before.st_uid == ::geteuid() &&
      before.st_nlink >= 1 && before.st_size > 0 &&
      static_cast<std::uint64_t>(before.st_size) <= maximum_executable_bytes;
  ::dg::hdg::runtime::Sha256 sha256;
  std::array<std::uint8_t, 64U * 1024U> buffer{};
  std::uint64_t received = 0;
  while (valid)
  {
    const ssize_t count = ::read(descriptor, buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR) continue;
    if (count < 0)
    {
      valid = false;
      break;
    }
    if (count == 0) break;
    received += static_cast<std::uint64_t>(count);
    valid = received <= maximum_executable_bytes &&
        sha256.Update(buffer.data(), static_cast<std::size_t>(count));
  }
  struct stat after{};
  valid = valid && ::fstat(descriptor, &after) == 0 &&
      S_ISREG(after.st_mode) && after.st_uid == ::geteuid() &&
      before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
      before.st_size == after.st_size && before.st_mode == after.st_mode &&
      received == static_cast<std::uint64_t>(before.st_size);
  if (::close(descriptor) != 0) valid = false;
  Digest candidate{};
  valid = valid && sha256.Finalize(candidate);
  if (!valid) return false;
  digest = candidate;
  return true;
}

inline bool SameBytes(const std::vector<double>& left,
                      const std::vector<double>& right) noexcept
{
  return left.size() == right.size() &&
      (left.empty() || std::memcmp(left.data(), right.data(),
                                  left.size() * sizeof(double)) == 0);
}

inline bool SameStateBytes(const cpu::PersistentRank2TrajectoryState& left,
                           const cpu::PersistentRank2TrajectoryState& right)
    noexcept
{
  return SameBytes(left.u, right.u) && SameBytes(left.sigma, right.sigma) &&
      SameBytes(left.z, right.z) && SameBytes(left.uhat, right.uhat) &&
      SameBytes(left.sigmahat, right.sigmahat);
}

} // namespace dg_igr_hdg_sc25_rank2_manifest

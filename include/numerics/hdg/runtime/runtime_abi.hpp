#pragma once

/**
 * @file numerics/hdg/runtime/runtime_abi.hpp
 * @brief Pointer-free shared W8A runtime records and canonical byte codecs.
 */

#include "numerics/hdg/runtime/run_contract.hpp"
#include "numerics/hdg/runtime/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dg::hdg::runtime
{

enum class FailurePhase : std::uint8_t
{
  Setup = 0,
  Assembly = 1,
  Condensation = 2,
  TraceSolve = 3,
  Recovery = 4,
  Diagnostics = 5,
  CheckpointStaging = 6,
  OutputStaging = 7,
  AfterRecoveryBeforeCommit = 8,
  Commit = 9,
  RollbackAudit = 10,
  TerminalEvidence = 11
};

enum class TransferDirection : std::uint8_t
{
  HostToDevice = 0,
  DeviceToHost = 1,
  DeviceToDevice = 2,
  MpiHostStaging = 3
};

enum class TransferCategory : std::uint8_t
{
  InitialState = 0,
  OutputSnapshot = 1,
  CheckpointWrite = 2,
  CheckpointRead = 3,
  OutputCheckpointShared = 4,
  ValidationHistoryAppend = 5,
  ValidationHistory = 6,
  ScalarDiagnostic = 7,
  HostStagedTraceExchange = 8
};

enum class AllocationOperation : std::uint8_t
{
  Allocate = 0,
  Deallocate = 1
};

constexpr bool IsKnownFailurePhase(const FailurePhase value) noexcept
{
  return static_cast<std::uint8_t>(value) <=
      static_cast<std::uint8_t>(FailurePhase::TerminalEvidence);
}

constexpr bool IsKnownTransferDirection(
    const TransferDirection value) noexcept
{
  return static_cast<std::uint8_t>(value) <=
      static_cast<std::uint8_t>(TransferDirection::MpiHostStaging);
}

constexpr bool IsKnownTransferCategory(
    const TransferCategory value) noexcept
{
  return static_cast<std::uint8_t>(value) <=
      static_cast<std::uint8_t>(TransferCategory::HostStagedTraceExchange);
}

constexpr bool IsKnownAllocationOperation(
    const AllocationOperation value) noexcept
{
  return static_cast<std::uint8_t>(value) <=
      static_cast<std::uint8_t>(AllocationOperation::Deallocate);
}

/** Declaration order is the serialized field order, never the object bytes. */
struct TransferEventV1
{
  std::uint64_t ordinal = 0;
  std::uint32_t call_site_id = 0;
  TransferDirection direction = TransferDirection::HostToDevice;
  TransferCategory category = TransferCategory::InitialState;
  FailurePhase phase = FailurePhase::Setup;
  std::uint8_t success = 0;
  std::uint64_t bytes = 0;
  std::uint64_t source_capacity_bytes = 0;
  std::uint64_t destination_capacity_bytes = 0;
  std::uint64_t source_storage_token = 0;
  std::uint64_t destination_storage_token = 0;
  std::uint64_t endpoint_token = 0;
  std::uint64_t attempt_cursor = 0;
  std::uint64_t target_step = 0;
  std::uint64_t generation = 0;
  std::int32_t rank = -1;
};

struct AllocationEventV1
{
  std::uint64_t ordinal = 0;
  std::uint32_t call_site_id = 0;
  AllocationOperation operation = AllocationOperation::Allocate;
  FailurePhase phase = FailurePhase::Setup;
  std::uint8_t success = 0;
  std::uint64_t requested_bytes = 0;
  std::uint64_t requested_alignment = 0;
  std::uint64_t capacity_bytes = 0;
  std::uint64_t storage_token = 0;
  std::uint64_t attempt_cursor = 0;
  std::uint64_t target_step = 0;
  std::uint64_t generation = 0;
  std::int32_t rank = -1;
};

inline constexpr std::size_t transfer_event_v1_serialized_size = 92;
inline constexpr std::size_t allocation_event_v1_serialized_size = 75;
using SerializedTransferEventV1 =
    std::array<std::uint8_t, transfer_event_v1_serialized_size>;
using SerializedAllocationEventV1 =
    std::array<std::uint8_t, allocation_event_v1_serialized_size>;

namespace abi_detail
{

template <typename Unsigned>
void WriteLittleEndian(const Unsigned value,
                       std::uint8_t* output,
                       std::size_t& offset) noexcept
{
  static_assert(std::is_unsigned<Unsigned>::value,
                "little-endian writer requires an unsigned integer");
  for (std::size_t index = 0; index < sizeof(Unsigned); ++index)
  {
    output[offset++] =
        static_cast<std::uint8_t>(value >> (8U * index));
  }
}

template <typename Unsigned>
Unsigned ReadLittleEndian(const std::uint8_t* input,
                          std::size_t& offset) noexcept
{
  static_assert(std::is_unsigned<Unsigned>::value,
                "little-endian reader requires an unsigned integer");
  Unsigned value = 0;
  for (std::size_t index = 0; index < sizeof(Unsigned); ++index)
  {
    value |= static_cast<Unsigned>(input[offset++]) << (8U * index);
  }
  return value;
}

inline std::int32_t DecodeInt32(const std::uint32_t value) noexcept
{
  constexpr std::uint32_t positive_maximum =
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
  if (value <= positive_maximum) { return static_cast<std::int32_t>(value); }
  return static_cast<std::int32_t>(
      -1 - static_cast<std::int32_t>(
               std::numeric_limits<std::uint32_t>::max() - value));
}

inline bool Valid(const TransferEventV1& event) noexcept
{
  return IsKnownTransferDirection(event.direction) &&
      IsKnownTransferCategory(event.category) &&
      IsKnownFailurePhase(event.phase) && event.success <= 1;
}

inline bool Valid(const AllocationEventV1& event) noexcept
{
  return IsKnownAllocationOperation(event.operation) &&
      IsKnownFailurePhase(event.phase) && event.success <= 1;
}

} // namespace abi_detail

inline bool SerializeTransferEventV1(
    const TransferEventV1& event,
    SerializedTransferEventV1& serialized) noexcept
{
  if (!abi_detail::Valid(event)) { return false; }
  SerializedTransferEventV1 candidate{};
  std::size_t offset = 0;
  abi_detail::WriteLittleEndian(event.ordinal, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.call_site_id, candidate.data(), offset);
  candidate[offset++] = static_cast<std::uint8_t>(event.direction);
  candidate[offset++] = static_cast<std::uint8_t>(event.category);
  candidate[offset++] = static_cast<std::uint8_t>(event.phase);
  candidate[offset++] = event.success;
  abi_detail::WriteLittleEndian(event.bytes, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.source_capacity_bytes,
                                candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.destination_capacity_bytes,
                                candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.source_storage_token,
                                candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.destination_storage_token,
                                candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.endpoint_token, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.attempt_cursor, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.target_step, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.generation, candidate.data(), offset);
  abi_detail::WriteLittleEndian(static_cast<std::uint32_t>(event.rank),
                                candidate.data(), offset);
  if (offset != candidate.size()) { return false; }
  serialized = candidate;
  return true;
}

inline bool DecodeTransferEventV1(const std::uint8_t* serialized,
                                  const std::size_t length,
                                  TransferEventV1& event) noexcept
{
  if (serialized == nullptr || length != transfer_event_v1_serialized_size)
  {
    return false;
  }
  TransferEventV1 candidate{};
  std::size_t offset = 0;
  candidate.ordinal =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.call_site_id =
      abi_detail::ReadLittleEndian<std::uint32_t>(serialized, offset);
  candidate.direction = static_cast<TransferDirection>(serialized[offset++]);
  candidate.category = static_cast<TransferCategory>(serialized[offset++]);
  candidate.phase = static_cast<FailurePhase>(serialized[offset++]);
  candidate.success = serialized[offset++];
  candidate.bytes =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.source_capacity_bytes =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.destination_capacity_bytes =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.source_storage_token =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.destination_storage_token =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.endpoint_token =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.attempt_cursor =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.target_step =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.generation =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.rank = abi_detail::DecodeInt32(
      abi_detail::ReadLittleEndian<std::uint32_t>(serialized, offset));
  if (offset != length || !abi_detail::Valid(candidate)) { return false; }
  event = candidate;
  return true;
}

inline bool DeserializeTransferEventV1(const std::uint8_t* serialized,
                                       const std::size_t length,
                                       TransferEventV1& event) noexcept
{
  return DecodeTransferEventV1(serialized, length, event);
}

inline bool SerializeAllocationEventV1(
    const AllocationEventV1& event,
    SerializedAllocationEventV1& serialized) noexcept
{
  if (!abi_detail::Valid(event)) { return false; }
  SerializedAllocationEventV1 candidate{};
  std::size_t offset = 0;
  abi_detail::WriteLittleEndian(event.ordinal, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.call_site_id, candidate.data(), offset);
  candidate[offset++] = static_cast<std::uint8_t>(event.operation);
  candidate[offset++] = static_cast<std::uint8_t>(event.phase);
  candidate[offset++] = event.success;
  abi_detail::WriteLittleEndian(event.requested_bytes,
                                candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.requested_alignment,
                                candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.capacity_bytes, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.storage_token, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.attempt_cursor, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.target_step, candidate.data(), offset);
  abi_detail::WriteLittleEndian(event.generation, candidate.data(), offset);
  abi_detail::WriteLittleEndian(static_cast<std::uint32_t>(event.rank),
                                candidate.data(), offset);
  if (offset != candidate.size()) { return false; }
  serialized = candidate;
  return true;
}

inline bool DecodeAllocationEventV1(const std::uint8_t* serialized,
                                    const std::size_t length,
                                    AllocationEventV1& event) noexcept
{
  if (serialized == nullptr || length != allocation_event_v1_serialized_size)
  {
    return false;
  }
  AllocationEventV1 candidate{};
  std::size_t offset = 0;
  candidate.ordinal =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.call_site_id =
      abi_detail::ReadLittleEndian<std::uint32_t>(serialized, offset);
  candidate.operation =
      static_cast<AllocationOperation>(serialized[offset++]);
  candidate.phase = static_cast<FailurePhase>(serialized[offset++]);
  candidate.success = serialized[offset++];
  candidate.requested_bytes =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.requested_alignment =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.capacity_bytes =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.storage_token =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.attempt_cursor =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.target_step =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.generation =
      abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset);
  candidate.rank = abi_detail::DecodeInt32(
      abi_detail::ReadLittleEndian<std::uint32_t>(serialized, offset));
  if (offset != length || !abi_detail::Valid(candidate)) { return false; }
  event = candidate;
  return true;
}

inline bool DeserializeAllocationEventV1(const std::uint8_t* serialized,
                                         const std::size_t length,
                                         AllocationEventV1& event) noexcept
{
  return DecodeAllocationEventV1(serialized, length, event);
}

enum class RankPayloadKind : std::uint8_t
{
  Interior = 0x01,
  TrueTrace = 0x02
};

constexpr bool IsKnownRankPayloadKind(const RankPayloadKind kind) noexcept
{
  return kind == RankPayloadKind::Interior ||
      kind == RankPayloadKind::TrueTrace;
}

struct RankPayloadTupleV1
{
  std::uint32_t rank = 0;
  std::uint64_t byte_count = 0;
  StateHash local_hash{};
};

/** A typed digest prevents swapping interior and true-trace manifests. */
struct RankPayloadHashV1
{
  RankPayloadKind kind = RankPayloadKind::Interior;
  StateHash hash{};
};

inline constexpr std::array<std::uint8_t, 25>
    rank_payload_manifest_domain_v1{{
        'D', 'G', 'I', 'G', 'R', '-', 'H', 'D', 'G', '-', 'R', 'A', 'N',
        'K', '-', 'P', 'A', 'Y', 'L', 'O', 'A', 'D', '-', 'V', '1'}};
inline constexpr std::size_t rank_payload_manifest_header_size_v1 = 30;
inline constexpr std::size_t rank_payload_manifest_tuple_size_v1 = 44;

inline bool RankPayloadManifestSizeV1(const std::size_t rank_count,
                                      std::size_t& size) noexcept
{
  if (rank_count == 0 ||
      rank_count > std::numeric_limits<std::uint32_t>::max() ||
      rank_count >
          (std::numeric_limits<std::size_t>::max() -
           rank_payload_manifest_header_size_v1) /
              rank_payload_manifest_tuple_size_v1)
  {
    return false;
  }
  size = rank_payload_manifest_header_size_v1 +
      rank_count * rank_payload_manifest_tuple_size_v1;
  return true;
}

inline bool RankPayloadTuplesValidV1(const RankPayloadKind kind,
                                     const RankPayloadTupleV1* tuples,
                                     const std::size_t rank_count) noexcept
{
  if (!IsKnownRankPayloadKind(kind) || tuples == nullptr || rank_count == 0 ||
      rank_count > std::numeric_limits<std::uint32_t>::max())
  {
    return false;
  }
  for (std::size_t index = 0; index < rank_count; ++index)
  {
    if (tuples[index].rank != index) { return false; }
  }
  return true;
}

inline bool SerializeRankPayloadManifestV1(
    const RankPayloadKind kind,
    const RankPayloadTupleV1* tuples,
    const std::size_t rank_count,
    std::uint8_t* serialized,
    const std::size_t length) noexcept
{
  std::size_t expected = 0;
  if (!RankPayloadManifestSizeV1(rank_count, expected) ||
      expected != length || serialized == nullptr ||
      !RankPayloadTuplesValidV1(kind, tuples, rank_count))
  {
    return false;
  }

  std::copy(rank_payload_manifest_domain_v1.begin(),
            rank_payload_manifest_domain_v1.end(), serialized);
  std::size_t offset = rank_payload_manifest_domain_v1.size();
  serialized[offset++] = static_cast<std::uint8_t>(kind);
  abi_detail::WriteLittleEndian(static_cast<std::uint32_t>(rank_count),
                                serialized, offset);
  for (std::size_t index = 0; index < rank_count; ++index)
  {
    abi_detail::WriteLittleEndian(tuples[index].rank, serialized, offset);
    abi_detail::WriteLittleEndian(tuples[index].byte_count,
                                  serialized, offset);
    std::copy(tuples[index].local_hash.bytes.begin(),
              tuples[index].local_hash.bytes.end(), serialized + offset);
    offset += tuples[index].local_hash.bytes.size();
  }
  return offset == length;
}

inline bool HashRankPayloadManifestV1(const RankPayloadKind kind,
                                      const RankPayloadTupleV1* tuples,
                                      const std::size_t rank_count,
                                      RankPayloadHashV1& result) noexcept
{
  std::size_t ignored_size = 0;
  if (!RankPayloadManifestSizeV1(rank_count, ignored_size) ||
      !RankPayloadTuplesValidV1(kind, tuples, rank_count))
  {
    return false;
  }

  Sha256 hash;
  const std::uint8_t kind_byte = static_cast<std::uint8_t>(kind);
  std::array<std::uint8_t, 4> count_bytes{};
  std::size_t offset = 0;
  abi_detail::WriteLittleEndian(static_cast<std::uint32_t>(rank_count),
                                count_bytes.data(), offset);
  if (!hash.Update(rank_payload_manifest_domain_v1.data(),
                   rank_payload_manifest_domain_v1.size()) ||
      !hash.Update(&kind_byte, 1) ||
      !hash.Update(count_bytes.data(), count_bytes.size()))
  {
    return false;
  }
  for (std::size_t index = 0; index < rank_count; ++index)
  {
    std::array<std::uint8_t, 12> prefix{};
    offset = 0;
    abi_detail::WriteLittleEndian(tuples[index].rank, prefix.data(), offset);
    abi_detail::WriteLittleEndian(tuples[index].byte_count,
                                  prefix.data(), offset);
    if (!hash.Update(prefix.data(), prefix.size()) ||
        !hash.Update(tuples[index].local_hash.bytes.data(),
                     tuples[index].local_hash.bytes.size()))
    {
      return false;
    }
  }
  Sha256Digest digest{};
  if (!hash.Finalize(digest)) { return false; }
  RankPayloadHashV1 candidate{};
  candidate.kind = kind;
  candidate.hash.bytes = digest;
  result = candidate;
  return true;
}

/** Validate the complete encoded shape before hashing any byte. */
inline bool DecodeAndHashRankPayloadManifestV1(
    const std::uint8_t* serialized,
    const std::size_t length,
    RankPayloadHashV1& result) noexcept
{
  if (serialized == nullptr ||
      length < rank_payload_manifest_header_size_v1 ||
      !std::equal(rank_payload_manifest_domain_v1.begin(),
                  rank_payload_manifest_domain_v1.end(), serialized))
  {
    return false;
  }

  std::size_t offset = rank_payload_manifest_domain_v1.size();
  const RankPayloadKind kind =
      static_cast<RankPayloadKind>(serialized[offset++]);
  const std::uint32_t rank_count =
      abi_detail::ReadLittleEndian<std::uint32_t>(serialized, offset);
  std::size_t expected = 0;
  if (!IsKnownRankPayloadKind(kind) ||
      !RankPayloadManifestSizeV1(rank_count, expected) || expected != length)
  {
    return false;
  }

  for (std::uint32_t rank = 0; rank < rank_count; ++rank)
  {
    const std::uint32_t encoded_rank =
        abi_detail::ReadLittleEndian<std::uint32_t>(serialized, offset);
    if (encoded_rank != rank) { return false; }
    static_cast<void>(
        abi_detail::ReadLittleEndian<std::uint64_t>(serialized, offset));
    offset += StateHash{}.bytes.size();
  }
  if (offset != length) { return false; }

  const Sha256Digest digest = ComputeSha256(serialized, length);
  RankPayloadHashV1 candidate{};
  candidate.kind = kind;
  candidate.hash.bytes = digest;
  result = candidate;
  return true;
}

struct CheckpointMetadataV1
{
  CheckpointDescriptor checkpoint{};
  StateHash local_interior_hash{};
  StateHash local_true_trace_hash{};
};

inline bool MakeCheckpointMetadataV1(
    const CheckpointDescriptor& checkpoint,
    const RankPayloadHashV1& rank_common_interior,
    const RankPayloadHashV1& rank_common_true_trace,
    const StateHash& local_interior_hash,
    const StateHash& local_true_trace_hash,
    CheckpointMetadataV1& metadata) noexcept
{
  if (rank_common_interior.kind != RankPayloadKind::Interior ||
      rank_common_true_trace.kind != RankPayloadKind::TrueTrace)
  {
    return false;
  }
  CheckpointMetadataV1 candidate{};
  candidate.checkpoint = checkpoint;
  candidate.checkpoint.interior_hash = rank_common_interior.hash;
  candidate.checkpoint.trace_hash = rank_common_true_trace.hash;
  candidate.local_interior_hash = local_interior_hash;
  candidate.local_true_trace_hash = local_true_trace_hash;
  metadata = candidate;
  return true;
}

static_assert(sizeof(FailurePhase) == sizeof(std::uint8_t) &&
                  sizeof(TransferDirection) == sizeof(std::uint8_t) &&
                  sizeof(TransferCategory) == sizeof(std::uint8_t) &&
                  sizeof(AllocationOperation) == sizeof(std::uint8_t),
              "shared runtime enum ordinals require exact uint8 storage");
static_assert(static_cast<std::uint8_t>(FailurePhase::Setup) == 0 &&
                  static_cast<std::uint8_t>(FailurePhase::Assembly) == 1 &&
                  static_cast<std::uint8_t>(FailurePhase::Condensation) == 2 &&
                  static_cast<std::uint8_t>(FailurePhase::TraceSolve) == 3 &&
                  static_cast<std::uint8_t>(FailurePhase::Recovery) == 4 &&
                  static_cast<std::uint8_t>(FailurePhase::Diagnostics) == 5 &&
                  static_cast<std::uint8_t>(
                      FailurePhase::CheckpointStaging) == 6 &&
                  static_cast<std::uint8_t>(FailurePhase::OutputStaging) ==
                      7 &&
                  static_cast<std::uint8_t>(
                      FailurePhase::AfterRecoveryBeforeCommit) == 8 &&
                  static_cast<std::uint8_t>(FailurePhase::Commit) == 9 &&
                  static_cast<std::uint8_t>(FailurePhase::RollbackAudit) ==
                      10 &&
                  static_cast<std::uint8_t>(FailurePhase::TerminalEvidence) ==
                      11,
              "FailurePhase ordinals are part of the shared ABI");
static_assert(
    static_cast<std::uint8_t>(TransferDirection::HostToDevice) == 0 &&
        static_cast<std::uint8_t>(TransferDirection::DeviceToHost) == 1 &&
        static_cast<std::uint8_t>(TransferDirection::DeviceToDevice) == 2 &&
        static_cast<std::uint8_t>(TransferDirection::MpiHostStaging) == 3,
              "TransferDirection ordinals are part of the shared ABI");
static_assert(
    static_cast<std::uint8_t>(TransferCategory::InitialState) == 0 &&
        static_cast<std::uint8_t>(TransferCategory::OutputSnapshot) == 1 &&
        static_cast<std::uint8_t>(TransferCategory::CheckpointWrite) == 2 &&
        static_cast<std::uint8_t>(TransferCategory::CheckpointRead) == 3 &&
        static_cast<std::uint8_t>(
            TransferCategory::OutputCheckpointShared) == 4 &&
        static_cast<std::uint8_t>(
            TransferCategory::ValidationHistoryAppend) == 5 &&
        static_cast<std::uint8_t>(TransferCategory::ValidationHistory) == 6 &&
        static_cast<std::uint8_t>(TransferCategory::ScalarDiagnostic) == 7 &&
        static_cast<std::uint8_t>(
            TransferCategory::HostStagedTraceExchange) == 8,
              "TransferCategory ordinals are part of the shared ABI");
static_assert(
    static_cast<std::uint8_t>(AllocationOperation::Allocate) == 0 &&
        static_cast<std::uint8_t>(AllocationOperation::Deallocate) == 1,
              "AllocationOperation ordinals are part of the shared ABI");
static_assert(std::is_standard_layout<TransferEventV1>::value &&
                  std::is_trivially_copyable<TransferEventV1>::value &&
                  std::is_standard_layout<AllocationEventV1>::value &&
                  std::is_trivially_copyable<AllocationEventV1>::value &&
                  std::is_standard_layout<RankPayloadTupleV1>::value &&
                  std::is_trivially_copyable<RankPayloadTupleV1>::value &&
                  std::is_standard_layout<RankPayloadHashV1>::value &&
                  std::is_trivially_copyable<RankPayloadHashV1>::value &&
                  std::is_standard_layout<CheckpointMetadataV1>::value &&
                  std::is_trivially_copyable<CheckpointMetadataV1>::value,
              "shared runtime ABI values must remain standard-layout POD");

} // namespace dg::hdg::runtime

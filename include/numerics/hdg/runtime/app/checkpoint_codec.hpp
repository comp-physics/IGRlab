#pragma once

/**
 * @file numerics/hdg/runtime/app/checkpoint_codec.hpp
 * @brief Explicit little-endian W8A checkpoint and manifest codec.
 *
 * The durable form is field-wise and versioned.  Object representations and
 * view pointers are never serialized.  Decoded payload views borrow the input
 * byte buffer and must not outlive it.
 */

#include "numerics/hdg/runtime/runtime_abi.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace dg::hdg::runtime::app
{

inline constexpr std::uint32_t application_checkpoint_codec_version_v1 = 1;

struct ConstByteView
{
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;

  constexpr bool Valid() const noexcept
  {
    return size == 0 || data != nullptr;
  }
};

/** Non-owning view staged by an existing backend/run-contract adapter. */
struct StagedStateViewV1
{
  std::uint32_t rank = 0;
  ConstByteView interior{};
  ConstByteView owned_true_trace{};
};

enum class PublicationPhaseV1 : std::uint8_t
{
  staged = 0,
  validated = 1,
  published = 2
};

struct AtomicPublicationStateV1
{
  PublicationPhaseV1 phase = PublicationPhaseV1::staged;
  std::uint64_t transaction_id = 0;
  StateHash previous_manifest_hash{};
};

enum class CheckpointCodecStatus : std::uint8_t
{
  ok = 0,
  invalid_argument,
  invalid_descriptor,
  invalid_publication_state,
  invalid_publication_transition,
  unpublished_artifact,
  unsupported_version,
  truncated,
  length_mismatch,
  invalid_magic,
  checksum_mismatch,
  invalid_rank_manifest,
  rank_tuple_mismatch,
  local_payload_hash_mismatch,
  aggregate_payload_hash_mismatch,
  restart_tuple_mismatch,
  trailing_bytes,
  size_overflow
};

/** Only staged -> validated -> published is legal; no skip or reversal. */
CheckpointCodecStatus AdvancePublicationStateV1(
    AtomicPublicationStateV1& state,
    PublicationPhaseV1 next) noexcept;

struct CheckpointWriteViewV1
{
  CheckpointDescriptor descriptor{};
  AtomicPublicationStateV1 publication{};
  const StagedStateViewV1* ranks = nullptr;
  std::size_t rank_count = 0;
};

struct DecodedRankStateViewV1
{
  std::uint32_t rank = 0;
  ConstByteView interior{};
  StateHash interior_hash{};
  ConstByteView owned_true_trace{};
  StateHash owned_true_trace_hash{};
};

/** Metadata is owned; rank payload pointers borrow the encoded input. */
struct DecodedCheckpointV1
{
  CheckpointDescriptor descriptor{};
  AtomicPublicationStateV1 publication{};
  std::vector<DecodedRankStateViewV1> ranks{};
  StateHash artifact_hash{};
};

CheckpointCodecStatus EncodeCheckpointV1(
    const CheckpointWriteViewV1& checkpoint,
    std::vector<std::uint8_t>& encoded);

CheckpointCodecStatus DecodeCheckpointV1(
    ConstByteView encoded,
    DecodedCheckpointV1& checkpoint,
    bool require_published = true);

/**
 * Preflight only the immutable same-tuple/resource identity.  The existing
 * PersistentRunContract::Restore remains the authoritative lifecycle and
 * coordinate/hash admission gate.
 */
CheckpointCodecStatus ValidateSameTupleRestartV1(
    const CheckpointDescriptor& checkpoint,
    const CheckpointDescriptor& expected) noexcept;

static_assert(!std::is_polymorphic<StagedStateViewV1>::value,
              "staged state must remain a non-polymorphic view");
static_assert(!std::is_polymorphic<DecodedCheckpointV1>::value,
              "checkpoint metadata must not introduce virtual dispatch");

} // namespace dg::hdg::runtime::app

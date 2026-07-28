#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/runtime_selection.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <vector>

namespace
{

using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::app;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "checkpoint codec test must compile in strict C++17");
static_assert(std::is_trivially_copyable<StagedStateViewV1>::value &&
                  !std::is_polymorphic<StagedStateViewV1>::value,
              "staged state is a non-owning static view");

bool Check(const bool condition, const char* name)
{
  if (!condition) { std::cerr << "FAILED: " << name << '\n'; }
  return condition;
}

StateHash FilledHash(const std::uint8_t first) noexcept
{
  StateHash hash{};
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    hash.bytes[index] = static_cast<std::uint8_t>(first + index);
  }
  return hash;
}

StateHash PayloadHash(const ConstByteView payload) noexcept
{
  StateHash hash{};
  hash.bytes = ComputeSha256(payload.data, payload.size);
  return hash;
}

bool SetAggregateHashes(const StagedStateViewV1* ranks,
                        const std::size_t rank_count,
                        CheckpointDescriptor& descriptor)
{
  std::vector<RankPayloadTupleV1> interior(rank_count);
  std::vector<RankPayloadTupleV1> trace(rank_count);
  for (std::size_t index = 0; index < rank_count; ++index)
  {
    interior[index] = {ranks[index].rank,
                       ranks[index].interior.size,
                       PayloadHash(ranks[index].interior)};
    trace[index] = {ranks[index].rank,
                    ranks[index].owned_true_trace.size,
                    PayloadHash(ranks[index].owned_true_trace)};
  }
  RankPayloadHashV1 interior_hash{};
  RankPayloadHashV1 trace_hash{};
  if (!HashRankPayloadManifestV1(RankPayloadKind::Interior, interior.data(),
                                 interior.size(), interior_hash) ||
      !HashRankPayloadManifestV1(RankPayloadKind::TrueTrace, trace.data(),
                                 trace.size(), trace_hash))
  {
    return false;
  }
  descriptor.interior_hash = interior_hash.hash;
  descriptor.trace_hash = trace_hash.hash;
  return true;
}

CheckpointDescriptor MakeDescriptor(const StagedStateViewV1* ranks,
                                    const std::size_t rank_count)
{
  CheckpointDescriptor descriptor{};
  descriptor.schema_version = checkpoint_schema_version;
  descriptor.successful_commit = true;
  descriptor.restart_mode = RestartMode::same_backend_rank_partition;
  descriptor.tuple = CpuRank2RuntimeSelection::Descriptor();
  descriptor.mfem_pin = StableIdentity::Copy("d9d6526cc1749980");
  descriptor.mesh_hash = FilledHash(0x10);
  descriptor.partition_hash = FilledHash(0x30);
  descriptor.topology_hash = FilledHash(0x50);
  descriptor.rank_count = static_cast<std::uint32_t>(rank_count);
  descriptor.polynomial_order = 1;
  descriptor.layout = StableIdentity::Copy("l2-interior.true-trace-owned.v1");
  descriptor.precision = descriptor.tuple.precision;
  descriptor.cursor = 10;
  descriptor.dt = 0.05;
  descriptor.coordinates = {0.5, 10, 10, 10, 0.05};
  descriptor.coordinate_fingerprint =
      detail::CoordinateFingerprint(descriptor.coordinates);
  static_cast<void>(SetAggregateHashes(ranks, rank_count, descriptor));
  return descriptor;
}

bool SealChecksum(std::vector<std::uint8_t>& bytes)
{
  if (bytes.size() < Sha256Digest{}.size()) { return false; }
  const std::size_t content_size = bytes.size() - Sha256Digest{}.size();
  const Sha256Digest checksum = ComputeSha256(bytes.data(), content_size);
  std::copy(checksum.begin(), checksum.end(), bytes.begin() + content_size);
  return true;
}

std::uint64_t ReadU64(const std::uint8_t* bytes) noexcept
{
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < sizeof(value); ++index)
  {
    value |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
  }
  return value;
}

struct Fixture
{
  std::array<std::uint8_t, 5> interior0{{1, 2, 3, 4, 5}};
  std::array<std::uint8_t, 3> trace0{{6, 7, 8}};
  std::array<std::uint8_t, 4> interior1{{9, 10, 11, 12}};
  std::array<std::uint8_t, 6> trace1{{13, 14, 15, 16, 17, 18}};

  std::array<StagedStateViewV1, 2> Views() const noexcept
  {
    return {{{0,
              {interior0.data(), interior0.size()},
              {trace0.data(), trace0.size()}},
             {1,
              {interior1.data(), interior1.size()},
              {trace1.data(), trace1.size()}}}};
  }
};

AtomicPublicationStateV1 PublishedState() noexcept
{
  return {PublicationPhaseV1::published, 42, FilledHash(0x70)};
}

bool CheckPublicationProtocol()
{
  AtomicPublicationStateV1 state{
      PublicationPhaseV1::staged, 42, FilledHash(0x70)};
  bool passed = Check(
      AdvancePublicationStateV1(state, PublicationPhaseV1::published) ==
              CheckpointCodecStatus::invalid_publication_transition &&
          state.phase == PublicationPhaseV1::staged,
      "publication_cannot_skip_validation");
  passed = Check(
               AdvancePublicationStateV1(
                   state, PublicationPhaseV1::validated) ==
                       CheckpointCodecStatus::ok &&
                   state.phase == PublicationPhaseV1::validated,
               "publication_staged_to_validated") &&
      passed;
  passed = Check(
               AdvancePublicationStateV1(
                   state, PublicationPhaseV1::staged) ==
                       CheckpointCodecStatus::invalid_publication_transition &&
                   state.phase == PublicationPhaseV1::validated,
               "publication_cannot_reverse") &&
      passed;
  passed = Check(
               AdvancePublicationStateV1(
                   state, PublicationPhaseV1::published) ==
                       CheckpointCodecStatus::ok &&
                   state.phase == PublicationPhaseV1::published,
               "publication_validated_to_published") &&
      passed;

  AtomicPublicationStateV1 invalid{};
  passed = Check(
               AdvancePublicationStateV1(
                   invalid, PublicationPhaseV1::validated) ==
                   CheckpointCodecStatus::invalid_publication_state,
               "publication_requires_transaction_identity") &&
      passed;
  return passed;
}

bool CheckRoundTripAndFieldwiseEncoding()
{
  const Fixture fixture;
  const auto ranks = fixture.Views();
  const CheckpointDescriptor descriptor =
      MakeDescriptor(ranks.data(), ranks.size());
  const CheckpointWriteViewV1 write{
      descriptor, PublishedState(), ranks.data(), ranks.size()};
  std::vector<std::uint8_t> encoded;
  bool passed = Check(EncodeCheckpointV1(write, encoded) ==
                          CheckpointCodecStatus::ok,
                      "checkpoint_encode_published") &&
      Check(encoded.size() > 64 && encoded[0] == 'D' && encoded[1] == 'G' &&
                encoded[16] == 1 && encoded[17] == 0 && encoded[18] == 0 &&
                encoded[19] == 0 && ReadU64(encoded.data() + 20) ==
                    encoded.size() &&
                encoded[28] ==
                    static_cast<std::uint8_t>(PublicationPhaseV1::published) &&
                ReadU64(encoded.data() + 29) == 42,
            "checkpoint_header_is_explicit_little_endian") &&
      passed;

  DecodedCheckpointV1 decoded;
  passed = Check(DecodeCheckpointV1({encoded.data(), encoded.size()}, decoded) ==
                         CheckpointCodecStatus::ok &&
                     detail::SameCheckpoint(decoded.descriptor, descriptor) &&
                     decoded.publication.phase ==
                         PublicationPhaseV1::published &&
                     decoded.publication.transaction_id == 42 &&
                     decoded.ranks.size() == 2 &&
                     decoded.ranks[0].rank == 0 &&
                     decoded.ranks[0].interior.size == fixture.interior0.size() &&
                     std::equal(fixture.interior0.begin(),
                                fixture.interior0.end(),
                                decoded.ranks[0].interior.data) &&
                     decoded.ranks[0].owned_true_trace.size ==
                         fixture.trace0.size() &&
                     decoded.ranks[1].rank == 1 &&
                     decoded.ranks[1].owned_true_trace.size ==
                         fixture.trace1.size() &&
                     decoded.ranks[0].interior_hash ==
                         PayloadHash(ranks[0].interior) &&
                     decoded.ranks[1].owned_true_trace_hash ==
                         PayloadHash(ranks[1].owned_true_trace),
                 "checkpoint_round_trip_exact_descriptor_rank_payloads") &&
      passed;

  const Fixture second_fixture;
  const auto second_ranks = second_fixture.Views();
  std::vector<std::uint8_t> independently_staged;
  passed = Check(
               ranks[0].interior.data != second_ranks[0].interior.data &&
                   EncodeCheckpointV1({descriptor,
                                       PublishedState(),
                                       second_ranks.data(),
                                       second_ranks.size()},
                                      independently_staged) ==
                       CheckpointCodecStatus::ok &&
                   independently_staged == encoded,
               "nonowning_view_addresses_are_not_serialized") &&
      passed;
  return passed;
}

bool CheckPublicationAdmission()
{
  const Fixture fixture;
  const auto ranks = fixture.Views();
  const CheckpointDescriptor descriptor =
      MakeDescriptor(ranks.data(), ranks.size());
  AtomicPublicationStateV1 staged{
      PublicationPhaseV1::staged, 43, FilledHash(0x71)};
  std::vector<std::uint8_t> encoded;
  DecodedCheckpointV1 decoded;
  return Check(
      EncodeCheckpointV1(
          {descriptor, staged, ranks.data(), ranks.size()}, encoded) ==
              CheckpointCodecStatus::ok &&
          DecodeCheckpointV1({encoded.data(), encoded.size()}, decoded) ==
              CheckpointCodecStatus::unpublished_artifact &&
          DecodeCheckpointV1({encoded.data(), encoded.size()}, decoded,
                             false) == CheckpointCodecStatus::ok &&
          decoded.publication.phase == PublicationPhaseV1::staged,
      "restore_rejects_unpublished_artifact_but_staging_audit_decodes");
}

bool CheckCorruptionAndTruncation()
{
  const Fixture fixture;
  const auto ranks = fixture.Views();
  const CheckpointDescriptor descriptor =
      MakeDescriptor(ranks.data(), ranks.size());
  std::vector<std::uint8_t> encoded;
  if (EncodeCheckpointV1(
          {descriptor, PublishedState(), ranks.data(), ranks.size()},
          encoded) != CheckpointCodecStatus::ok)
  {
    return Check(false, "corruption_fixture_encode");
  }
  DecodedCheckpointV1 decoded;
  if (DecodeCheckpointV1({encoded.data(), encoded.size()}, decoded) !=
      CheckpointCodecStatus::ok)
  {
    return Check(false, "corruption_fixture_decode");
  }
  const std::size_t payload_offset = static_cast<std::size_t>(
      decoded.ranks[0].interior.data - encoded.data());
  constexpr std::size_t rank_record_size = 4 + 8 + 32 + 8 + 32;
  const std::size_t manifest_count_offset =
      payload_offset - ranks.size() * rank_record_size - 4;

  bool passed = true;
  std::vector<std::uint8_t> damaged = encoded;
  damaged[payload_offset] ^= 0x80U;
  passed = Check(DecodeCheckpointV1({damaged.data(), damaged.size()}, decoded) ==
                         CheckpointCodecStatus::checksum_mismatch,
                     "payload_corruption_rejected_by_artifact_checksum") &&
      passed;

  damaged = encoded;
  damaged.pop_back();
  passed = Check(DecodeCheckpointV1({damaged.data(), damaged.size()}, decoded) ==
                         CheckpointCodecStatus::truncated,
                     "truncated_checkpoint_rejected") &&
      passed;

  damaged = encoded;
  damaged.push_back(0);
  passed = Check(DecodeCheckpointV1({damaged.data(), damaged.size()}, decoded) ==
                         CheckpointCodecStatus::length_mismatch,
                     "trailing_checkpoint_bytes_rejected") &&
      passed;

  damaged = encoded;
  damaged[0] ^= 1U;
  passed = Check(DecodeCheckpointV1({damaged.data(), damaged.size()}, decoded) ==
                         CheckpointCodecStatus::invalid_magic,
                     "bad_magic_rejected") &&
      passed;

  damaged = encoded;
  damaged[16] = 2;
  passed = Check(DecodeCheckpointV1({damaged.data(), damaged.size()}, decoded) ==
                         CheckpointCodecStatus::unsupported_version,
                     "unknown_codec_version_rejected") &&
      passed;

  damaged = encoded;
  damaged[payload_offset] ^= 0x40U;
  static_cast<void>(SealChecksum(damaged));
  passed = Check(DecodeCheckpointV1({damaged.data(), damaged.size()}, decoded) ==
                         CheckpointCodecStatus::local_payload_hash_mismatch,
                     "payload_hash_mismatch_rejected_after_valid_checksum") &&
      passed;

  damaged = encoded;
  damaged[manifest_count_offset + 4] = 1;
  static_cast<void>(SealChecksum(damaged));
  passed = Check(DecodeCheckpointV1({damaged.data(), damaged.size()}, decoded) ==
                         CheckpointCodecStatus::invalid_rank_manifest,
                     "rank_tuple_order_corruption_rejected") &&
      passed;

  damaged = encoded;
  const auto aggregate = std::search(
      damaged.begin(), damaged.end(), descriptor.interior_hash.bytes.begin(),
      descriptor.interior_hash.bytes.end());
  if (aggregate == damaged.end())
  {
    passed = Check(false, "aggregate_hash_field_located") && passed;
  }
  else
  {
    *aggregate ^= 1U;
    static_cast<void>(SealChecksum(damaged));
    passed = Check(
                 DecodeCheckpointV1({damaged.data(), damaged.size()}, decoded) ==
                     CheckpointCodecStatus::aggregate_payload_hash_mismatch,
                 "aggregate_manifest_hash_corruption_rejected") &&
        passed;
  }
  return passed;
}

bool CheckRankAndRestartTupleValidation()
{
  const Fixture fixture;
  auto ranks = fixture.Views();
  const CheckpointDescriptor descriptor =
      MakeDescriptor(ranks.data(), ranks.size());
  std::vector<std::uint8_t> sentinel{{0xaa, 0xbb}};
  ranks[0].rank = 1;
  bool passed = Check(
      EncodeCheckpointV1(
          {descriptor, PublishedState(), ranks.data(), ranks.size()},
          sentinel) == CheckpointCodecStatus::invalid_rank_manifest &&
          sentinel == std::vector<std::uint8_t>({0xaa, 0xbb}),
      "encoder_rejects_rank_tuple_without_mutating_output");

  passed = Check(ValidateSameTupleRestartV1(descriptor, descriptor) ==
                         CheckpointCodecStatus::ok,
                     "same_tuple_restart_identity_admitted") &&
      passed;
  CheckpointDescriptor other = descriptor;
  other.tuple = CpuSerialRuntimeSelection::Descriptor();
  other.precision = other.tuple.precision;
  passed = Check(ValidateSameTupleRestartV1(descriptor, other) ==
                         CheckpointCodecStatus::restart_tuple_mismatch,
                     "cross_backend_restart_tuple_rejected") &&
      passed;
  other = descriptor;
  other.partition_hash.bytes[0] ^= 1U;
  passed = Check(ValidateSameTupleRestartV1(descriptor, other) ==
                         CheckpointCodecStatus::restart_tuple_mismatch,
                     "cross_partition_restart_tuple_rejected") &&
      passed;
  return passed;
}

} // namespace

int main()
{
  const bool passed = CheckPublicationProtocol() &&
      CheckRoundTripAndFieldwiseEncoding() && CheckPublicationAdmission() &&
      CheckCorruptionAndTruncation() &&
      CheckRankAndRestartTupleValidation();
  if (passed)
  {
    std::cout << "HDG W8A checkpoint codec checks passed\n";
    return 0;
  }
  return 1;
}

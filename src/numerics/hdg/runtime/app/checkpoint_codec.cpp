#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace dg::hdg::runtime::app
{
namespace codec_detail
{

constexpr std::array<std::uint8_t, 16> magic{{
    'D', 'G', 'I', 'G', 'R', '-', 'H', 'D', 'G', '-', 'C', 'K', 'P', 'T',
    '1', 0}};
constexpr std::size_t checksum_size = StateHash{}.bytes.size();
constexpr std::size_t total_size_offset = magic.size() + sizeof(std::uint32_t);

bool KnownPhase(const PublicationPhaseV1 phase) noexcept
{
  return phase == PublicationPhaseV1::staged ||
      phase == PublicationPhaseV1::validated ||
      phase == PublicationPhaseV1::published;
}

bool AddWouldOverflow(const std::size_t left,
                      const std::size_t right) noexcept
{
  return right > std::numeric_limits<std::size_t>::max() - left;
}

class Writer
{
public:
  void U8(const std::uint8_t value) { bytes_.push_back(value); }

  template <typename Unsigned>
  void UnsignedLittleEndian(const Unsigned value)
  {
    static_assert(std::is_unsigned<Unsigned>::value,
                  "codec writer requires unsigned integer fields");
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index)
    {
      bytes_.push_back(
          static_cast<std::uint8_t>(value >> (8U * index)));
    }
  }

  void Bytes(const std::uint8_t* data, const std::size_t size)
  {
    if (size != 0) { bytes_.insert(bytes_.end(), data, data + size); }
  }

  void Hash(const StateHash& hash)
  {
    Bytes(hash.bytes.data(), hash.bytes.size());
  }

  bool Identity(const StableIdentity& identity)
  {
    if (!identity.valid() || identity.empty() ||
        identity.size() > std::numeric_limits<std::uint16_t>::max())
    {
      return false;
    }
    UnsignedLittleEndian(static_cast<std::uint16_t>(identity.size()));
    Bytes(reinterpret_cast<const std::uint8_t*>(identity.data()),
          identity.size());
    return true;
  }

  bool PatchU64(const std::size_t offset, const std::uint64_t value)
  {
    if (offset > bytes_.size() || bytes_.size() - offset < sizeof(value))
    {
      return false;
    }
    for (std::size_t index = 0; index < sizeof(value); ++index)
    {
      bytes_[offset + index] =
          static_cast<std::uint8_t>(value >> (8U * index));
    }
    return true;
  }

  std::vector<std::uint8_t>& bytes() noexcept { return bytes_; }

private:
  std::vector<std::uint8_t> bytes_{};
};

class Reader
{
public:
  Reader(const std::uint8_t* data, const std::size_t size) noexcept
      : data_(data), size_(size)
  {}

  bool U8(std::uint8_t& value) noexcept
  {
    if (!Available(1)) { return false; }
    value = data_[offset_++];
    return true;
  }

  template <typename Unsigned>
  bool UnsignedLittleEndian(Unsigned& value) noexcept
  {
    static_assert(std::is_unsigned<Unsigned>::value,
                  "codec reader requires unsigned integer fields");
    if (!Available(sizeof(Unsigned))) { return false; }
    value = 0;
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index)
    {
      value |= static_cast<Unsigned>(data_[offset_++]) << (8U * index);
    }
    return true;
  }

  bool Bytes(std::uint8_t* output, const std::size_t size) noexcept
  {
    if (!Available(size)) { return false; }
    if (size != 0) { std::copy_n(data_ + offset_, size, output); }
    offset_ += size;
    return true;
  }

  bool View(const std::size_t size, ConstByteView& view) noexcept
  {
    if (!Available(size)) { return false; }
    view = {data_ + offset_, size};
    offset_ += size;
    return true;
  }

  bool Hash(StateHash& hash) noexcept
  {
    return Bytes(hash.bytes.data(), hash.bytes.size());
  }

  bool Identity(StableIdentity& identity) noexcept
  {
    std::uint16_t length = 0;
    if (!UnsignedLittleEndian(length) || length == 0 ||
        !Available(length))
    {
      return false;
    }
    const std::string_view view{
        reinterpret_cast<const char*>(data_ + offset_), length};
    const StableIdentity candidate = StableIdentity::Copy(view);
    if (!candidate.valid() || candidate.empty()) { return false; }
    offset_ += length;
    identity = candidate;
    return true;
  }

  bool Skip(const std::size_t size) noexcept
  {
    if (!Available(size)) { return false; }
    offset_ += size;
    return true;
  }

  bool AtEnd() const noexcept { return offset_ == size_; }

private:
  bool Available(const std::size_t size) const noexcept
  {
    return offset_ <= size_ && size <= size_ - offset_;
  }

  const std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t offset_ = 0;
};

std::uint64_t DoubleBits(const double value) noexcept
{
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

double BitsDouble(const std::uint64_t bits) noexcept
{
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool DescriptorStructurallyValid(const CheckpointDescriptor& value) noexcept
{
  return value.schema_version == checkpoint_schema_version &&
      value.successful_commit &&
      value.restart_mode == RestartMode::same_backend_rank_partition &&
      value.tuple.Valid() && value.mfem_pin.valid() &&
      !value.mfem_pin.empty() && value.rank_count != 0 &&
      value.polynomial_order >= 0 && value.layout.valid() &&
      !value.layout.empty() && value.precision.valid() &&
      !value.precision.empty() && value.precision == value.tuple.precision &&
      value.cursor == value.coordinates.cursor &&
      std::isfinite(value.coordinates.time) && std::isfinite(value.dt) &&
      value.dt > 0.0 && std::isfinite(value.coordinates.dt) &&
      value.coordinates.dt > 0.0 &&
      DoubleBits(value.dt) == DoubleBits(value.coordinates.dt) &&
      value.coordinate_fingerprint ==
          ::dg::hdg::runtime::detail::CoordinateFingerprint(
              value.coordinates);
}

bool WriteDescriptor(Writer& writer, const CheckpointDescriptor& value)
{
  writer.UnsignedLittleEndian(value.schema_version);
  writer.U8(value.successful_commit ? 1U : 0U);
  writer.U8(static_cast<std::uint8_t>(value.restart_mode));
  if (!writer.Identity(value.tuple.model) ||
      !writer.Identity(value.tuple.formulation) ||
      !writer.Identity(value.tuple.trace) ||
      !writer.Identity(value.tuple.stabilization) ||
      !writer.Identity(value.tuple.time_treatment) ||
      !writer.Identity(value.tuple.backend) ||
      !writer.Identity(value.tuple.precision) ||
      !writer.Identity(value.mfem_pin))
  {
    return false;
  }
  writer.Hash(value.mesh_hash);
  writer.Hash(value.partition_hash);
  writer.Hash(value.topology_hash);
  writer.UnsignedLittleEndian(value.rank_count);
  writer.UnsignedLittleEndian(static_cast<std::uint32_t>(
      value.polynomial_order));
  if (!writer.Identity(value.layout) || !writer.Identity(value.precision))
  {
    return false;
  }
  writer.UnsignedLittleEndian(value.cursor);
  writer.UnsignedLittleEndian(DoubleBits(value.dt));
  writer.UnsignedLittleEndian(DoubleBits(value.coordinates.time));
  writer.UnsignedLittleEndian(value.coordinates.step);
  writer.UnsignedLittleEndian(value.coordinates.generation);
  writer.UnsignedLittleEndian(value.coordinates.cursor);
  writer.UnsignedLittleEndian(DoubleBits(value.coordinates.dt));
  writer.UnsignedLittleEndian(value.coordinate_fingerprint);
  writer.Hash(value.interior_hash);
  writer.Hash(value.trace_hash);
  return true;
}

bool ReadDescriptor(Reader& reader, CheckpointDescriptor& value) noexcept
{
  CheckpointDescriptor candidate{};
  std::uint8_t successful = 0;
  std::uint8_t restart = 0;
  std::uint32_t polynomial_order = 0;
  std::uint64_t dt = 0;
  std::uint64_t time = 0;
  std::uint64_t coordinate_dt = 0;
  if (!reader.UnsignedLittleEndian(candidate.schema_version) ||
      !reader.U8(successful) || !reader.U8(restart) ||
      !reader.Identity(candidate.tuple.model) ||
      !reader.Identity(candidate.tuple.formulation) ||
      !reader.Identity(candidate.tuple.trace) ||
      !reader.Identity(candidate.tuple.stabilization) ||
      !reader.Identity(candidate.tuple.time_treatment) ||
      !reader.Identity(candidate.tuple.backend) ||
      !reader.Identity(candidate.tuple.precision) ||
      !reader.Identity(candidate.mfem_pin) ||
      !reader.Hash(candidate.mesh_hash) ||
      !reader.Hash(candidate.partition_hash) ||
      !reader.Hash(candidate.topology_hash) ||
      !reader.UnsignedLittleEndian(candidate.rank_count) ||
      !reader.UnsignedLittleEndian(polynomial_order) ||
      !reader.Identity(candidate.layout) ||
      !reader.Identity(candidate.precision) ||
      !reader.UnsignedLittleEndian(candidate.cursor) ||
      !reader.UnsignedLittleEndian(dt) ||
      !reader.UnsignedLittleEndian(time) ||
      !reader.UnsignedLittleEndian(candidate.coordinates.step) ||
      !reader.UnsignedLittleEndian(candidate.coordinates.generation) ||
      !reader.UnsignedLittleEndian(candidate.coordinates.cursor) ||
      !reader.UnsignedLittleEndian(coordinate_dt) ||
      !reader.UnsignedLittleEndian(candidate.coordinate_fingerprint) ||
      !reader.Hash(candidate.interior_hash) ||
      !reader.Hash(candidate.trace_hash))
  {
    return false;
  }
  if (successful > 1 ||
      restart != static_cast<std::uint8_t>(
                     RestartMode::same_backend_rank_partition))
  {
    return false;
  }
  candidate.successful_commit = successful != 0;
  candidate.restart_mode = static_cast<RestartMode>(restart);
  candidate.polynomial_order = abi_detail::DecodeInt32(polynomial_order);
  candidate.dt = BitsDouble(dt);
  candidate.coordinates.time = BitsDouble(time);
  candidate.coordinates.dt = BitsDouble(coordinate_dt);
  value = candidate;
  return true;
}

struct RankMetadata
{
  std::uint32_t rank = 0;
  std::uint64_t interior_size = 0;
  StateHash interior_hash{};
  std::uint64_t trace_size = 0;
  StateHash trace_hash{};
};

bool SameHash(const Sha256Digest& digest, const StateHash& hash) noexcept
{
  return std::equal(digest.begin(), digest.end(), hash.bytes.begin());
}

StateHash AsStateHash(const Sha256Digest& digest) noexcept
{
  StateHash result{};
  result.bytes = digest;
  return result;
}

bool AggregateHashes(const std::vector<RankMetadata>& metadata,
                     RankPayloadHashV1& interior,
                     RankPayloadHashV1& trace)
{
  std::vector<RankPayloadTupleV1> interior_tuples(metadata.size());
  std::vector<RankPayloadTupleV1> trace_tuples(metadata.size());
  for (std::size_t index = 0; index < metadata.size(); ++index)
  {
    interior_tuples[index] = {metadata[index].rank,
                              metadata[index].interior_size,
                              metadata[index].interior_hash};
    trace_tuples[index] = {metadata[index].rank,
                           metadata[index].trace_size,
                           metadata[index].trace_hash};
  }
  return HashRankPayloadManifestV1(RankPayloadKind::Interior,
                                   interior_tuples.data(),
                                   interior_tuples.size(), interior) &&
      HashRankPayloadManifestV1(RankPayloadKind::TrueTrace,
                                trace_tuples.data(), trace_tuples.size(),
                                trace);
}

} // namespace codec_detail

CheckpointCodecStatus AdvancePublicationStateV1(
    AtomicPublicationStateV1& state,
    const PublicationPhaseV1 next) noexcept
{
  if (state.transaction_id == 0 || !codec_detail::KnownPhase(state.phase) ||
      !codec_detail::KnownPhase(next))
  {
    return CheckpointCodecStatus::invalid_publication_state;
  }
  const bool valid =
      (state.phase == PublicationPhaseV1::staged &&
       next == PublicationPhaseV1::validated) ||
      (state.phase == PublicationPhaseV1::validated &&
       next == PublicationPhaseV1::published);
  if (!valid)
  {
    return CheckpointCodecStatus::invalid_publication_transition;
  }
  state.phase = next;
  return CheckpointCodecStatus::ok;
}

CheckpointCodecStatus EncodeCheckpointV1(
    const CheckpointWriteViewV1& checkpoint,
    std::vector<std::uint8_t>& encoded)
{
  using namespace codec_detail;
  if (!DescriptorStructurallyValid(checkpoint.descriptor))
  {
    return CheckpointCodecStatus::invalid_descriptor;
  }
  if (!KnownPhase(checkpoint.publication.phase) ||
      checkpoint.publication.transaction_id == 0)
  {
    return CheckpointCodecStatus::invalid_publication_state;
  }
  if (checkpoint.ranks == nullptr || checkpoint.rank_count == 0 ||
      checkpoint.rank_count != checkpoint.descriptor.rank_count ||
      checkpoint.rank_count > std::numeric_limits<std::uint32_t>::max())
  {
    return CheckpointCodecStatus::invalid_rank_manifest;
  }

  std::vector<RankMetadata> metadata(checkpoint.rank_count);
  std::size_t payload_size = 0;
  for (std::size_t index = 0; index < checkpoint.rank_count; ++index)
  {
    const StagedStateViewV1& rank = checkpoint.ranks[index];
    if (rank.rank != index || !rank.interior.Valid() ||
        !rank.owned_true_trace.Valid() ||
        rank.interior.size > std::numeric_limits<std::uint64_t>::max() ||
        rank.owned_true_trace.size >
            std::numeric_limits<std::uint64_t>::max() ||
        AddWouldOverflow(payload_size, rank.interior.size) ||
        AddWouldOverflow(payload_size + rank.interior.size,
                         rank.owned_true_trace.size))
    {
      return CheckpointCodecStatus::invalid_rank_manifest;
    }
    metadata[index] = {
        rank.rank,
        static_cast<std::uint64_t>(rank.interior.size),
        AsStateHash(ComputeSha256(rank.interior.data, rank.interior.size)),
        static_cast<std::uint64_t>(rank.owned_true_trace.size),
        AsStateHash(ComputeSha256(rank.owned_true_trace.data,
                                  rank.owned_true_trace.size))};
    payload_size += rank.interior.size + rank.owned_true_trace.size;
  }

  RankPayloadHashV1 aggregate_interior{};
  RankPayloadHashV1 aggregate_trace{};
  if (!AggregateHashes(metadata, aggregate_interior, aggregate_trace) ||
      aggregate_interior.hash != checkpoint.descriptor.interior_hash ||
      aggregate_trace.hash != checkpoint.descriptor.trace_hash)
  {
    return CheckpointCodecStatus::aggregate_payload_hash_mismatch;
  }

  Writer writer;
  writer.Bytes(magic.data(), magic.size());
  writer.UnsignedLittleEndian(application_checkpoint_codec_version_v1);
  writer.UnsignedLittleEndian(std::uint64_t{0});
  writer.U8(static_cast<std::uint8_t>(checkpoint.publication.phase));
  writer.UnsignedLittleEndian(checkpoint.publication.transaction_id);
  writer.Hash(checkpoint.publication.previous_manifest_hash);
  if (!WriteDescriptor(writer, checkpoint.descriptor))
  {
    return CheckpointCodecStatus::invalid_descriptor;
  }
  writer.UnsignedLittleEndian(
      static_cast<std::uint32_t>(checkpoint.rank_count));
  for (const RankMetadata& rank : metadata)
  {
    writer.UnsignedLittleEndian(rank.rank);
    writer.UnsignedLittleEndian(rank.interior_size);
    writer.Hash(rank.interior_hash);
    writer.UnsignedLittleEndian(rank.trace_size);
    writer.Hash(rank.trace_hash);
  }
  for (std::size_t index = 0; index < checkpoint.rank_count; ++index)
  {
    writer.Bytes(checkpoint.ranks[index].interior.data,
                 checkpoint.ranks[index].interior.size);
    writer.Bytes(checkpoint.ranks[index].owned_true_trace.data,
                 checkpoint.ranks[index].owned_true_trace.size);
  }
  if (AddWouldOverflow(writer.bytes().size(), checksum_size) ||
      writer.bytes().size() + checksum_size >
          std::numeric_limits<std::uint64_t>::max())
  {
    return CheckpointCodecStatus::size_overflow;
  }
  const std::uint64_t final_size =
      static_cast<std::uint64_t>(writer.bytes().size() + checksum_size);
  if (!writer.PatchU64(total_size_offset, final_size))
  {
    return CheckpointCodecStatus::size_overflow;
  }
  const Sha256Digest checksum =
      ComputeSha256(writer.bytes().data(), writer.bytes().size());
  writer.Bytes(checksum.data(), checksum.size());
  encoded = std::move(writer.bytes());
  return CheckpointCodecStatus::ok;
}

CheckpointCodecStatus DecodeCheckpointV1(
    const ConstByteView encoded,
    DecodedCheckpointV1& checkpoint,
    const bool require_published)
{
  using namespace codec_detail;
  constexpr std::size_t prefix_size =
      magic.size() + sizeof(std::uint32_t) + sizeof(std::uint64_t);
  if (!encoded.Valid()) { return CheckpointCodecStatus::invalid_argument; }
  if (encoded.size < prefix_size + checksum_size)
  {
    return CheckpointCodecStatus::truncated;
  }
  if (!std::equal(magic.begin(), magic.end(), encoded.data))
  {
    return CheckpointCodecStatus::invalid_magic;
  }

  Reader prefix(encoded.data + magic.size(),
                encoded.size - magic.size());
  std::uint32_t version = 0;
  std::uint64_t declared_size = 0;
  if (!prefix.UnsignedLittleEndian(version) ||
      !prefix.UnsignedLittleEndian(declared_size))
  {
    return CheckpointCodecStatus::truncated;
  }
  if (version != application_checkpoint_codec_version_v1)
  {
    return CheckpointCodecStatus::unsupported_version;
  }
  if (declared_size > encoded.size)
  {
    return CheckpointCodecStatus::truncated;
  }
  if (declared_size < encoded.size)
  {
    return CheckpointCodecStatus::length_mismatch;
  }

  const std::size_t content_size = encoded.size - checksum_size;
  const Sha256Digest checksum = ComputeSha256(encoded.data, content_size);
  if (!std::equal(checksum.begin(), checksum.end(),
                  encoded.data + content_size))
  {
    return CheckpointCodecStatus::checksum_mismatch;
  }

  Reader reader(encoded.data, content_size);
  if (!reader.Skip(prefix_size))
  {
    return CheckpointCodecStatus::truncated;
  }
  DecodedCheckpointV1 candidate{};
  std::uint8_t phase = 0;
  if (!reader.U8(phase) ||
      !reader.UnsignedLittleEndian(candidate.publication.transaction_id) ||
      !reader.Hash(candidate.publication.previous_manifest_hash))
  {
    return CheckpointCodecStatus::truncated;
  }
  candidate.publication.phase = static_cast<PublicationPhaseV1>(phase);
  if (!KnownPhase(candidate.publication.phase) ||
      candidate.publication.transaction_id == 0)
  {
    return CheckpointCodecStatus::invalid_publication_state;
  }
  if (require_published &&
      candidate.publication.phase != PublicationPhaseV1::published)
  {
    return CheckpointCodecStatus::unpublished_artifact;
  }
  if (!ReadDescriptor(reader, candidate.descriptor))
  {
    return CheckpointCodecStatus::truncated;
  }
  if (!DescriptorStructurallyValid(candidate.descriptor))
  {
    return CheckpointCodecStatus::invalid_descriptor;
  }

  std::uint32_t rank_count = 0;
  if (!reader.UnsignedLittleEndian(rank_count))
  {
    return CheckpointCodecStatus::truncated;
  }
  if (rank_count == 0 || rank_count != candidate.descriptor.rank_count)
  {
    return CheckpointCodecStatus::rank_tuple_mismatch;
  }
  std::vector<RankMetadata> metadata(rank_count);
  for (std::uint32_t index = 0; index < rank_count; ++index)
  {
    RankMetadata& rank = metadata[index];
    if (!reader.UnsignedLittleEndian(rank.rank) ||
        !reader.UnsignedLittleEndian(rank.interior_size) ||
        !reader.Hash(rank.interior_hash) ||
        !reader.UnsignedLittleEndian(rank.trace_size) ||
        !reader.Hash(rank.trace_hash))
    {
      return CheckpointCodecStatus::truncated;
    }
    if (rank.rank != index ||
        rank.interior_size > std::numeric_limits<std::size_t>::max() ||
        rank.trace_size > std::numeric_limits<std::size_t>::max())
    {
      return CheckpointCodecStatus::invalid_rank_manifest;
    }
  }

  candidate.ranks.resize(rank_count);
  for (std::uint32_t index = 0; index < rank_count; ++index)
  {
    const RankMetadata& metadata_rank = metadata[index];
    DecodedRankStateViewV1& decoded_rank = candidate.ranks[index];
    decoded_rank.rank = metadata_rank.rank;
    decoded_rank.interior_hash = metadata_rank.interior_hash;
    decoded_rank.owned_true_trace_hash = metadata_rank.trace_hash;
    if (!reader.View(static_cast<std::size_t>(metadata_rank.interior_size),
                     decoded_rank.interior) ||
        !reader.View(static_cast<std::size_t>(metadata_rank.trace_size),
                     decoded_rank.owned_true_trace))
    {
      return CheckpointCodecStatus::truncated;
    }
    if (!SameHash(ComputeSha256(decoded_rank.interior.data,
                                decoded_rank.interior.size),
                  metadata_rank.interior_hash) ||
        !SameHash(ComputeSha256(decoded_rank.owned_true_trace.data,
                                decoded_rank.owned_true_trace.size),
                  metadata_rank.trace_hash))
    {
      return CheckpointCodecStatus::local_payload_hash_mismatch;
    }
  }
  if (!reader.AtEnd()) { return CheckpointCodecStatus::trailing_bytes; }

  RankPayloadHashV1 aggregate_interior{};
  RankPayloadHashV1 aggregate_trace{};
  if (!AggregateHashes(metadata, aggregate_interior, aggregate_trace) ||
      aggregate_interior.hash != candidate.descriptor.interior_hash ||
      aggregate_trace.hash != candidate.descriptor.trace_hash)
  {
    return CheckpointCodecStatus::aggregate_payload_hash_mismatch;
  }
  candidate.artifact_hash.bytes = checksum;
  checkpoint = std::move(candidate);
  return CheckpointCodecStatus::ok;
}

CheckpointCodecStatus ValidateSameTupleRestartV1(
    const CheckpointDescriptor& checkpoint,
    const CheckpointDescriptor& expected) noexcept
{
  if (!codec_detail::DescriptorStructurallyValid(checkpoint) ||
      !codec_detail::DescriptorStructurallyValid(expected))
  {
    return CheckpointCodecStatus::invalid_descriptor;
  }
  if (checkpoint.schema_version != expected.schema_version ||
      checkpoint.restart_mode != expected.restart_mode ||
      checkpoint.tuple != expected.tuple ||
      checkpoint.mfem_pin != expected.mfem_pin ||
      checkpoint.mesh_hash != expected.mesh_hash ||
      checkpoint.partition_hash != expected.partition_hash ||
      checkpoint.topology_hash != expected.topology_hash ||
      checkpoint.rank_count != expected.rank_count ||
      checkpoint.polynomial_order != expected.polynomial_order ||
      checkpoint.layout != expected.layout ||
      checkpoint.precision != expected.precision)
  {
    return CheckpointCodecStatus::restart_tuple_mismatch;
  }
  return CheckpointCodecStatus::ok;
}

} // namespace dg::hdg::runtime::app

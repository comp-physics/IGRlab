#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>

namespace dg::hdg::runtime::app
{
namespace codec_detail
{

constexpr std::array<std::uint8_t, 16> magic{{
    'D', 'G', 'I', 'G', 'R', '-', 'H', 'D', 'G', '-', 'C', 'K', 'P', 'T',
    '1', 0}};
constexpr std::size_t checksum_size = StateHash{}.bytes.size();
constexpr std::size_t total_size_offset = magic.size() + sizeof(std::uint32_t);
constexpr std::size_t rank_record_size =
    sizeof(std::uint32_t) +
    2 * (sizeof(std::uint32_t) + sizeof(std::uint64_t) +
         sizeof(std::uint64_t) + sizeof(std::uint8_t) +
         sizeof(std::uint64_t) + StateHash{}.bytes.size());

bool Add(const std::uint64_t left,
         const std::uint64_t right,
         std::uint64_t& result) noexcept
{
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
  {
    return false;
  }
  result = left + right;
  return true;
}

bool Multiply(const std::uint64_t left,
              const std::uint64_t right,
              std::uint64_t& result) noexcept
{
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
  {
    return false;
  }
  result = left * right;
  return true;
}

bool NonzeroHash(const StateHash& hash) noexcept
{
  return std::any_of(hash.bytes.begin(), hash.bytes.end(),
                     [](const std::uint8_t byte) { return byte != 0; });
}

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

void LittleEndianBytes(const std::uint64_t value,
                       std::array<std::uint8_t, 8>& bytes) noexcept
{
  for (std::size_t index = 0; index < bytes.size(); ++index)
  {
    bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
  }
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

  bool Hash(StateHash& hash) noexcept
  {
    return Bytes(hash.bytes.data(), hash.bytes.size());
  }

  bool Identity(StableIdentity& identity) noexcept
  {
    std::uint16_t length = 0;
    if (!UnsignedLittleEndian(length) || length == 0 || !Available(length))
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

  std::size_t Remaining() const noexcept
  {
    return offset_ <= size_ ? size_ - offset_ : 0;
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

bool DescriptorStructurallyValid(const CheckpointDescriptor& value) noexcept
{
  return value.schema_version == checkpoint_schema_version &&
      value.successful_commit &&
      value.restart_mode == RestartMode::same_backend_rank_partition &&
      value.tuple.Valid() && value.mfem_pin.valid() &&
      !value.mfem_pin.empty() && value.rank_count != 0 &&
      value.rank_count <= application_checkpoint_max_rank_count_v1 &&
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
  writer.UnsignedLittleEndian(
      static_cast<std::uint32_t>(value.polynomial_order));
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

struct BlockMetadata
{
  std::uint32_t component_count = 0;
  std::uint64_t coefficient_count = 0;
  std::uint64_t dofs_per_component = 0;
  CoefficientOrderingV1 ordering = CoefficientOrderingV1::component_major;
  std::uint64_t byte_count = 0;
  StateHash hash{};
};

struct RankMetadata
{
  std::uint32_t rank = 0;
  BlockMetadata interior{};
  BlockMetadata trace{};
};

CheckpointCodecStatus ValidateBlockView(
    const Fp64CoefficientViewV1& block,
    BlockMetadata& metadata) noexcept
{
  std::uint64_t expected_count = 0;
  std::uint64_t byte_count = 0;
  if ((block.coefficient_count != 0 && block.data == nullptr) ||
      block.component_count != euler_state_component_count_v1 ||
      block.dofs_per_component == 0 ||
      block.ordering != CoefficientOrderingV1::component_major)
  {
    return CheckpointCodecStatus::invalid_coefficient_schema;
  }
  if (!Multiply(block.component_count, block.dofs_per_component,
                expected_count) ||
      expected_count != block.coefficient_count ||
      !Multiply(block.coefficient_count, fp64_coefficient_bytes_v1,
                byte_count))
  {
    return CheckpointCodecStatus::coefficient_count_mismatch;
  }
  Sha256 hash;
  std::array<std::uint8_t, 8> bytes{};
  for (std::uint64_t index = 0; index < block.coefficient_count; ++index)
  {
    if (!std::isfinite(block.data[index]))
    {
      return CheckpointCodecStatus::noncanonical_fp64;
    }
    LittleEndianBytes(DoubleBits(block.data[index]), bytes);
    if (!hash.Update(bytes.data(), bytes.size()))
    {
      return CheckpointCodecStatus::invalid_argument;
    }
  }
  Sha256Digest digest{};
  if (!hash.Finalize(digest))
  {
    return CheckpointCodecStatus::invalid_argument;
  }
  StateHash state_hash{};
  state_hash.bytes = digest;
  metadata = {block.component_count,
              block.coefficient_count,
              block.dofs_per_component,
              block.ordering,
              byte_count,
              state_hash};
  return CheckpointCodecStatus::ok;
}

CheckpointCodecStatus InspectRanks(
    const StagedStateViewV1* ranks,
    const std::size_t rank_count,
    const std::uint32_t expected_rank_count,
    std::array<RankMetadata, application_checkpoint_max_rank_count_v1>&
        metadata,
    std::uint64_t& total_interior,
    std::uint64_t& total_trace) noexcept
{
  if (ranks == nullptr || rank_count == 0 ||
      rank_count > application_checkpoint_max_rank_count_v1 ||
      rank_count != expected_rank_count)
  {
    return CheckpointCodecStatus::invalid_rank_manifest;
  }
  total_interior = 0;
  total_trace = 0;
  for (std::size_t index = 0; index < rank_count; ++index)
  {
    if (ranks[index].rank != index)
    {
      return CheckpointCodecStatus::invalid_rank_manifest;
    }
    RankMetadata candidate{};
    candidate.rank = ranks[index].rank;
    CheckpointCodecStatus status =
        ValidateBlockView(ranks[index].interior, candidate.interior);
    if (status != CheckpointCodecStatus::ok) { return status; }
    status = ValidateBlockView(ranks[index].owned_true_trace, candidate.trace);
    if (status != CheckpointCodecStatus::ok) { return status; }
    if (!Add(total_interior, candidate.interior.coefficient_count,
             total_interior) ||
        !Add(total_trace, candidate.trace.coefficient_count, total_trace))
    {
      return CheckpointCodecStatus::size_overflow;
    }
    metadata[index] = candidate;
  }
  return CheckpointCodecStatus::ok;
}

bool AggregateHashes(
    const std::array<RankMetadata, application_checkpoint_max_rank_count_v1>&
        metadata,
    const std::size_t rank_count,
    RankPayloadHashV1& interior,
    RankPayloadHashV1& trace) noexcept
{
  std::array<RankPayloadTupleV1, application_checkpoint_max_rank_count_v1>
      interior_tuples{};
  std::array<RankPayloadTupleV1, application_checkpoint_max_rank_count_v1>
      trace_tuples{};
  for (std::size_t index = 0; index < rank_count; ++index)
  {
    interior_tuples[index] = {metadata[index].rank,
                              metadata[index].interior.byte_count,
                              metadata[index].interior.hash};
    trace_tuples[index] = {metadata[index].rank,
                           metadata[index].trace.byte_count,
                           metadata[index].trace.hash};
  }
  return HashRankPayloadManifestV1(RankPayloadKind::Interior,
                                   interior_tuples.data(), rank_count,
                                   interior) &&
      HashRankPayloadManifestV1(RankPayloadKind::TrueTrace,
                                trace_tuples.data(), rank_count, trace);
}

void WriteBlockMetadata(Writer& writer, const BlockMetadata& block)
{
  writer.UnsignedLittleEndian(block.component_count);
  writer.UnsignedLittleEndian(block.coefficient_count);
  writer.UnsignedLittleEndian(block.dofs_per_component);
  writer.U8(static_cast<std::uint8_t>(block.ordering));
  writer.UnsignedLittleEndian(block.byte_count);
  writer.Hash(block.hash);
}

bool ReadBlockMetadata(Reader& reader, BlockMetadata& block) noexcept
{
  std::uint8_t ordering = 0;
  if (!reader.UnsignedLittleEndian(block.component_count) ||
      !reader.UnsignedLittleEndian(block.coefficient_count) ||
      !reader.UnsignedLittleEndian(block.dofs_per_component) ||
      !reader.U8(ordering) ||
      !reader.UnsignedLittleEndian(block.byte_count) ||
      !reader.Hash(block.hash))
  {
    return false;
  }
  block.ordering = static_cast<CoefficientOrderingV1>(ordering);
  return true;
}

CheckpointCodecStatus ValidateBlockMetadata(
    const BlockMetadata& block) noexcept
{
  std::uint64_t expected_count = 0;
  std::uint64_t expected_bytes = 0;
  if (block.component_count != euler_state_component_count_v1 ||
      block.dofs_per_component == 0 ||
      block.ordering != CoefficientOrderingV1::component_major)
  {
    return CheckpointCodecStatus::invalid_coefficient_schema;
  }
  if (!Multiply(block.component_count, block.dofs_per_component,
                expected_count) ||
      expected_count != block.coefficient_count ||
      !Multiply(block.coefficient_count, fp64_coefficient_bytes_v1,
                expected_bytes) ||
      expected_bytes != block.byte_count)
  {
    return CheckpointCodecStatus::coefficient_count_mismatch;
  }
  return CheckpointCodecStatus::ok;
}

void WriteCoefficientBlock(Writer& writer,
                           const Fp64CoefficientViewV1& block)
{
  std::array<std::uint8_t, 8> bytes{};
  for (std::uint64_t index = 0; index < block.coefficient_count; ++index)
  {
    LittleEndianBytes(DoubleBits(block.data[index]), bytes);
    writer.Bytes(bytes.data(), bytes.size());
  }
}

CheckpointCodecStatus DecodeCoefficientBlock(
    Reader& reader,
    const BlockMetadata& metadata,
    DecodedCoefficientBlockV1& block) noexcept
{
  Sha256 hash;
  std::array<std::uint8_t, 8> bytes{};
  for (std::uint64_t index = 0; index < metadata.coefficient_count; ++index)
  {
    if (!reader.Bytes(bytes.data(), bytes.size()) ||
        !hash.Update(bytes.data(), bytes.size()))
    {
      return CheckpointCodecStatus::truncated;
    }
    std::uint64_t bits = 0;
    for (std::size_t byte = 0; byte < bytes.size(); ++byte)
    {
      bits |= static_cast<std::uint64_t>(bytes[byte]) << (8U * byte);
    }
    const double value = BitsDouble(bits);
    if (!std::isfinite(value))
    {
      return CheckpointCodecStatus::noncanonical_fp64;
    }
    block.coefficients[static_cast<std::size_t>(index)] = value;
  }
  Sha256Digest digest{};
  if (!hash.Finalize(digest) || digest != metadata.hash.bytes)
  {
    return CheckpointCodecStatus::local_payload_hash_mismatch;
  }
  block.component_count = metadata.component_count;
  block.dofs_per_component = metadata.dofs_per_component;
  block.ordering = metadata.ordering;
  block.payload_hash = metadata.hash;
  return CheckpointCodecStatus::ok;
}

} // namespace codec_detail

struct PublicationAccessV1
{
  static const CheckpointDescriptor& Descriptor(
      const CheckpointContractEvidenceV1& value) noexcept
  {
    return value.descriptor_;
  }
  static const CheckpointDescriptor& Descriptor(
      const DurableRankTemporariesV1& value) noexcept
  {
    return value.descriptor_;
  }
  static const CheckpointDescriptor& Descriptor(
      const PublishedCheckpointManifestV1& value) noexcept
  {
    return value.validated_.temporaries_.descriptor_;
  }
  static const CheckpointContractEvidenceV1& Evidence(
      const PublishedCheckpointManifestV1& value) noexcept
  {
    return value.validated_.temporaries_.evidence_;
  }
  static const RankTemporaryPublicationFactsV1& TemporaryFacts(
      const PublishedCheckpointManifestV1& value) noexcept
  {
    return value.validated_.temporaries_.facts_;
  }
  static const CollectiveManifestValidationFactsV1& ValidationFacts(
      const PublishedCheckpointManifestV1& value) noexcept
  {
    return value.validated_.facts_;
  }
  static const AtomicManifestPublicationFactsV1& PublicationFacts(
      const PublishedCheckpointManifestV1& value) noexcept
  {
    return value.facts_;
  }
  static std::uint64_t InteriorCount(
      const PublishedCheckpointManifestV1& value) noexcept
  {
    return value.validated_.temporaries_.interior_coefficients_;
  }
  static std::uint64_t TraceCount(
      const PublishedCheckpointManifestV1& value) noexcept
  {
    return value.validated_.temporaries_.trace_coefficients_;
  }
  static DurableRankTemporariesV1 Durable(
      CheckpointContractEvidenceV1 evidence,
      const CheckpointDescriptor descriptor,
      const RankTemporaryPublicationFactsV1 facts,
      const std::uint64_t interior,
      const std::uint64_t trace) noexcept
  {
    return DurableRankTemporariesV1(std::move(evidence), descriptor, facts,
                                    interior, trace);
  }
  static ValidatedCheckpointManifestV1 Validated(
      DurableRankTemporariesV1 temporaries,
      const CollectiveManifestValidationFactsV1 facts) noexcept
  {
    return ValidatedCheckpointManifestV1(std::move(temporaries), facts);
  }
  static PublishedCheckpointManifestV1 Published(
      ValidatedCheckpointManifestV1 validated,
      const AtomicManifestPublicationFactsV1 facts) noexcept
  {
    return PublishedCheckpointManifestV1(std::move(validated), facts);
  }
  static const StableIdentity& Card(
      const CheckpointContractEvidenceV1& evidence) noexcept
  {
    return evidence.card_id_;
  }
  static const StableIdentity& Case(
      const CheckpointContractEvidenceV1& evidence) noexcept
  {
    return evidence.case_id_;
  }
  static const StableIdentity& Checkpoint(
      const CheckpointContractEvidenceV1& evidence) noexcept
  {
    return evidence.checkpoint_id_;
  }
};

PublicationTransitionV1<DurableRankTemporariesV1>
RecordDurableRankTemporariesV1(
    CheckpointContractEvidenceV1&& evidence,
    const RankTemporaryPublicationFactsV1& facts,
    const StagedStateViewV1* ranks,
    const std::size_t rank_count)
{
  using namespace codec_detail;
  PublicationTransitionV1<DurableRankTemporariesV1> result{};
  const CheckpointDescriptor& contract_descriptor =
      PublicationAccessV1::Descriptor(evidence);
  if (!DescriptorStructurallyValid(contract_descriptor) ||
      contract_descriptor.coordinates.step !=
          smooth_periodic_euler_v1_checkpoint_step ||
      contract_descriptor.coordinate_fingerprint == 0)
  {
    result.status = CheckpointCodecStatus::invalid_descriptor;
    return result;
  }
  if (facts.participating_ranks != contract_descriptor.rank_count ||
      facts.source_rank != smooth_periodic_euler_v1_source_rank ||
      !NonzeroHash(facts.temporary_file_set_hash) ||
      !facts.all_rank_temporaries_created_exclusive ||
      !facts.all_rank_temporaries_synced ||
      !facts.collective_temporary_consensus)
  {
    result.status = CheckpointCodecStatus::invalid_publication_state;
    return result;
  }
  std::array<RankMetadata, application_checkpoint_max_rank_count_v1> metadata{};
  std::uint64_t interior_count = 0;
  std::uint64_t trace_count = 0;
  result.status = InspectRanks(ranks, rank_count, contract_descriptor.rank_count,
                               metadata, interior_count, trace_count);
  if (result.status != CheckpointCodecStatus::ok) { return result; }
  if (metadata[0].interior.hash != contract_descriptor.interior_hash ||
      metadata[0].trace.hash != contract_descriptor.trace_hash)
  {
    result.status = CheckpointCodecStatus::local_payload_hash_mismatch;
    return result;
  }
  RankPayloadHashV1 aggregate_interior{};
  RankPayloadHashV1 aggregate_trace{};
  if (!AggregateHashes(metadata, rank_count, aggregate_interior,
                       aggregate_trace))
  {
    result.status = CheckpointCodecStatus::aggregate_payload_hash_mismatch;
    return result;
  }
  CheckpointDescriptor manifest_descriptor = contract_descriptor;
  manifest_descriptor.interior_hash = aggregate_interior.hash;
  manifest_descriptor.trace_hash = aggregate_trace.hash;
  result.status = CheckpointCodecStatus::ok;
  result.authority.emplace(PublicationAccessV1::Durable(
      std::move(evidence), manifest_descriptor, facts, interior_count,
      trace_count));
  return result;
}

PublicationTransitionV1<ValidatedCheckpointManifestV1>
ValidateCollectiveCheckpointManifestV1(
    DurableRankTemporariesV1&& temporaries,
    const CollectiveManifestValidationFactsV1& facts) noexcept
{
  PublicationTransitionV1<ValidatedCheckpointManifestV1> result{};
  const CheckpointDescriptor& descriptor =
      PublicationAccessV1::Descriptor(temporaries);
  if (facts.participating_ranks != descriptor.rank_count ||
      !facts.descriptor_consensus || !facts.rank_manifest_consensus ||
      !facts.payload_hash_consensus)
  {
    result.status = CheckpointCodecStatus::invalid_publication_state;
    return result;
  }
  result.status = CheckpointCodecStatus::ok;
  result.authority.emplace(
      PublicationAccessV1::Validated(std::move(temporaries), facts));
  return result;
}

PublicationTransitionV1<PublishedCheckpointManifestV1>
PublishCheckpointManifestV1(
    ValidatedCheckpointManifestV1&& validated,
    const AtomicManifestPublicationFactsV1& facts) noexcept
{
  PublicationTransitionV1<PublishedCheckpointManifestV1> result{};
  if (facts.source_rank != smooth_periodic_euler_v1_source_rank ||
      !facts.manifest_created_exclusive || !facts.manifest_synced ||
      !facts.atomic_rename_complete ||
      !facts.publication_directory_synced ||
      !facts.collective_visibility_consensus)
  {
    result.status = CheckpointCodecStatus::invalid_publication_state;
    return result;
  }
  result.status = CheckpointCodecStatus::ok;
  result.authority.emplace(
      PublicationAccessV1::Published(std::move(validated), facts));
  return result;
}

CheckpointCodecStatus EncodeCheckpointV1(
    const PublishedCheckpointManifestV1& publication,
    const StagedStateViewV1* ranks,
    const std::size_t rank_count,
    std::vector<std::uint8_t>& encoded)
{
  using namespace codec_detail;
  try
  {
    const CheckpointDescriptor& descriptor =
        PublicationAccessV1::Descriptor(publication);
    const CheckpointContractEvidenceV1& evidence =
        PublicationAccessV1::Evidence(publication);
    const RankTemporaryPublicationFactsV1& temporary =
        PublicationAccessV1::TemporaryFacts(publication);
    const CollectiveManifestValidationFactsV1& validation =
        PublicationAccessV1::ValidationFacts(publication);
    const AtomicManifestPublicationFactsV1& published =
        PublicationAccessV1::PublicationFacts(publication);
    if (!DescriptorStructurallyValid(descriptor) ||
        temporary.participating_ranks != descriptor.rank_count ||
        validation.participating_ranks != descriptor.rank_count ||
        temporary.source_rank != smooth_periodic_euler_v1_source_rank ||
        published.source_rank != smooth_periodic_euler_v1_source_rank)
    {
      return CheckpointCodecStatus::invalid_publication_state;
    }
    std::array<RankMetadata, application_checkpoint_max_rank_count_v1>
        metadata{};
    std::uint64_t interior_count = 0;
    std::uint64_t trace_count = 0;
    CheckpointCodecStatus status = InspectRanks(
        ranks, rank_count, descriptor.rank_count, metadata, interior_count,
        trace_count);
    if (status != CheckpointCodecStatus::ok) { return status; }
    if (interior_count != PublicationAccessV1::InteriorCount(publication) ||
        trace_count != PublicationAccessV1::TraceCount(publication))
    {
      return CheckpointCodecStatus::coefficient_count_mismatch;
    }
    RankPayloadHashV1 aggregate_interior{};
    RankPayloadHashV1 aggregate_trace{};
    if (!AggregateHashes(metadata, rank_count, aggregate_interior,
                         aggregate_trace) ||
        aggregate_interior.hash != descriptor.interior_hash ||
        aggregate_trace.hash != descriptor.trace_hash)
    {
      return CheckpointCodecStatus::aggregate_payload_hash_mismatch;
    }

    Writer writer;
    writer.Bytes(magic.data(), magic.size());
    writer.UnsignedLittleEndian(application_checkpoint_codec_version_v1);
    writer.UnsignedLittleEndian(std::uint64_t{0});
    writer.U8(static_cast<std::uint8_t>(
        PublicationPhaseV1::manifest_atomically_published));
    writer.UnsignedLittleEndian(descriptor.coordinate_fingerprint);
    writer.Hash(temporary.temporary_file_set_hash);
    writer.Hash(temporary.previous_manifest_hash);
    writer.UnsignedLittleEndian(temporary.participating_ranks);
    writer.UnsignedLittleEndian(temporary.source_rank);
    for (const bool flag :
         {temporary.all_rank_temporaries_created_exclusive,
          temporary.all_rank_temporaries_synced,
          temporary.collective_temporary_consensus,
          validation.descriptor_consensus,
          validation.rank_manifest_consensus,
          validation.payload_hash_consensus,
          published.manifest_created_exclusive,
          published.manifest_synced,
          published.atomic_rename_complete,
          published.publication_directory_synced,
          published.collective_visibility_consensus})
    {
      writer.U8(flag ? 1U : 0U);
    }
    if (!writer.Identity(PublicationAccessV1::Card(evidence)) ||
        !writer.Identity(PublicationAccessV1::Case(evidence)) ||
        !writer.Identity(PublicationAccessV1::Checkpoint(evidence)) ||
        !WriteDescriptor(writer, descriptor))
    {
      return CheckpointCodecStatus::invalid_checkpoint_identity;
    }
    writer.UnsignedLittleEndian(static_cast<std::uint32_t>(rank_count));
    writer.UnsignedLittleEndian(interior_count);
    writer.UnsignedLittleEndian(trace_count);
    for (std::size_t index = 0; index < rank_count; ++index)
    {
      writer.UnsignedLittleEndian(metadata[index].rank);
      WriteBlockMetadata(writer, metadata[index].interior);
      WriteBlockMetadata(writer, metadata[index].trace);
    }
    for (std::size_t index = 0; index < rank_count; ++index)
    {
      WriteCoefficientBlock(writer, ranks[index].interior);
      WriteCoefficientBlock(writer, ranks[index].owned_true_trace);
    }
    if (writer.bytes().size() >
        std::numeric_limits<std::uint64_t>::max() - checksum_size)
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
  catch (const std::bad_alloc&)
  {
    return CheckpointCodecStatus::allocation_failure;
  }
  catch (const std::length_error&)
  {
    return CheckpointCodecStatus::allocation_failure;
  }
}

CheckpointCodecStatus ValidateSameTupleRestartV1(
    const CheckpointDescriptor& checkpoint,
    const ValidatedApplicationConfig& expected_config,
    const PersistentResourceDescriptor& expected_resources) noexcept
{
  if (!codec_detail::DescriptorStructurallyValid(checkpoint) ||
      !expected_resources.Valid())
  {
    return CheckpointCodecStatus::invalid_descriptor;
  }
  if (checkpoint.tuple != expected_config.tuple())
  {
    return CheckpointCodecStatus::restart_tuple_mismatch;
  }
  if (checkpoint.mfem_pin != expected_resources.mfem_pin ||
      checkpoint.mesh_hash != expected_resources.mesh_hash ||
      checkpoint.partition_hash != expected_resources.partition_hash ||
      checkpoint.topology_hash != expected_resources.topology_hash ||
      checkpoint.rank_count != expected_resources.rank_count ||
      checkpoint.rank_count != expected_config.mpi_rank_count() ||
      checkpoint.polynomial_order != expected_resources.polynomial_order ||
      checkpoint.polynomial_order != expected_config.polynomial_order() ||
      checkpoint.layout != expected_resources.layout ||
      checkpoint.precision != expected_config.tuple().precision)
  {
    return CheckpointCodecStatus::resource_mismatch;
  }
  return CheckpointCodecStatus::ok;
}

CheckpointCodecStatus DecodeCheckpointV1(
    const ConstByteView encoded,
    const ValidatedApplicationConfig& expected_config,
    const PersistentResourceDescriptor& expected_resources,
    DecodedCheckpointV1& checkpoint)
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
  Reader prefix(encoded.data + magic.size(), encoded.size - magic.size());
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
  if (!reader.Skip(prefix_size)) { return CheckpointCodecStatus::truncated; }
  std::uint8_t phase = 0;
  PublicationAuditV1 publication{};
  std::array<std::uint8_t, 11> flags{};
  if (!reader.U8(phase) ||
      !reader.UnsignedLittleEndian(publication.transaction_id) ||
      !reader.Hash(publication.temporary_file_set_hash) ||
      !reader.Hash(publication.previous_manifest_hash) ||
      !reader.UnsignedLittleEndian(publication.participating_ranks) ||
      !reader.UnsignedLittleEndian(publication.source_rank))
  {
    return CheckpointCodecStatus::truncated;
  }
  for (std::uint8_t& flag : flags)
  {
    if (!reader.U8(flag)) { return CheckpointCodecStatus::truncated; }
  }
  publication.phase = static_cast<PublicationPhaseV1>(phase);
  if (publication.phase !=
          PublicationPhaseV1::manifest_atomically_published ||
      publication.transaction_id == 0 ||
      publication.source_rank != smooth_periodic_euler_v1_source_rank ||
      !NonzeroHash(publication.temporary_file_set_hash) ||
      std::any_of(flags.begin(), flags.end(),
                  [](const std::uint8_t flag) { return flag != 1; }))
  {
    return CheckpointCodecStatus::unpublished_artifact;
  }

  StableIdentity card_id{};
  StableIdentity case_id{};
  StableIdentity checkpoint_id{};
  CheckpointDescriptor descriptor{};
  if (!reader.Identity(card_id) || !reader.Identity(case_id) ||
      !reader.Identity(checkpoint_id) ||
      !ReadDescriptor(reader, descriptor))
  {
    return CheckpointCodecStatus::truncated;
  }
  if (!DescriptorStructurallyValid(descriptor))
  {
    return CheckpointCodecStatus::invalid_descriptor;
  }
  if (card_id != expected_config.card_id() ||
      case_id != expected_config.case_id() ||
      checkpoint_id != expected_config.checkpoint_id() ||
      descriptor.coordinates.step != expected_config.checkpoint_step() ||
      descriptor.coordinates.cursor != expected_config.checkpoint_step())
  {
    return CheckpointCodecStatus::invalid_checkpoint_identity;
  }
  CheckpointCodecStatus status = ValidateSameTupleRestartV1(
      descriptor, expected_config, expected_resources);
  if (status != CheckpointCodecStatus::ok) { return status; }

  std::uint32_t rank_count = 0;
  std::uint64_t aggregate_interior_count = 0;
  std::uint64_t aggregate_trace_count = 0;
  if (!reader.UnsignedLittleEndian(rank_count) ||
      !reader.UnsignedLittleEndian(aggregate_interior_count) ||
      !reader.UnsignedLittleEndian(aggregate_trace_count))
  {
    return CheckpointCodecStatus::truncated;
  }
  if (rank_count == 0 ||
      rank_count > application_checkpoint_max_rank_count_v1 ||
      rank_count != descriptor.rank_count ||
      rank_count != publication.participating_ranks)
  {
    return CheckpointCodecStatus::rank_tuple_mismatch;
  }
  const std::uint64_t metadata_bytes =
      static_cast<std::uint64_t>(rank_count) * rank_record_size;
  if (metadata_bytes > reader.Remaining())
  {
    return CheckpointCodecStatus::truncated;
  }

  std::array<RankMetadata, application_checkpoint_max_rank_count_v1> metadata{};
  std::uint64_t expected_payload_bytes = 0;
  std::uint64_t summed_interior = 0;
  std::uint64_t summed_trace = 0;
  for (std::uint32_t index = 0; index < rank_count; ++index)
  {
    RankMetadata& rank = metadata[index];
    if (!reader.UnsignedLittleEndian(rank.rank) ||
        !ReadBlockMetadata(reader, rank.interior) ||
        !ReadBlockMetadata(reader, rank.trace))
    {
      return CheckpointCodecStatus::truncated;
    }
    if (rank.rank != index)
    {
      return CheckpointCodecStatus::invalid_rank_manifest;
    }
    status = ValidateBlockMetadata(rank.interior);
    if (status != CheckpointCodecStatus::ok) { return status; }
    status = ValidateBlockMetadata(rank.trace);
    if (status != CheckpointCodecStatus::ok) { return status; }
    if (!Add(expected_payload_bytes, rank.interior.byte_count,
             expected_payload_bytes) ||
        !Add(expected_payload_bytes, rank.trace.byte_count,
             expected_payload_bytes) ||
        !Add(summed_interior, rank.interior.coefficient_count,
             summed_interior) ||
        !Add(summed_trace, rank.trace.coefficient_count, summed_trace))
    {
      return CheckpointCodecStatus::size_overflow;
    }
  }
  if (summed_interior != aggregate_interior_count ||
      summed_trace != aggregate_trace_count)
  {
    return CheckpointCodecStatus::coefficient_count_mismatch;
  }
  if (expected_payload_bytes != reader.Remaining())
  {
    return expected_payload_bytes > reader.Remaining()
        ? CheckpointCodecStatus::truncated
        : CheckpointCodecStatus::trailing_bytes;
  }

  RankPayloadHashV1 aggregate_interior{};
  RankPayloadHashV1 aggregate_trace{};
  if (!AggregateHashes(metadata, rank_count, aggregate_interior,
                       aggregate_trace) ||
      aggregate_interior.hash != descriptor.interior_hash ||
      aggregate_trace.hash != descriptor.trace_hash)
  {
    return CheckpointCodecStatus::aggregate_payload_hash_mismatch;
  }

  DecodedCheckpointV1 candidate{};
  try
  {
    candidate.ranks.resize(rank_count);
    for (std::uint32_t index = 0; index < rank_count; ++index)
    {
      candidate.ranks[index].interior.coefficients.resize(
          static_cast<std::size_t>(metadata[index].interior.coefficient_count));
      candidate.ranks[index].owned_true_trace.coefficients.resize(
          static_cast<std::size_t>(metadata[index].trace.coefficient_count));
    }
  }
  catch (const std::bad_alloc&)
  {
    return CheckpointCodecStatus::allocation_failure;
  }
  catch (const std::length_error&)
  {
    return CheckpointCodecStatus::allocation_failure;
  }
  for (std::uint32_t index = 0; index < rank_count; ++index)
  {
    candidate.ranks[index].rank = metadata[index].rank;
    status = DecodeCoefficientBlock(reader, metadata[index].interior,
                                    candidate.ranks[index].interior);
    if (status != CheckpointCodecStatus::ok)
    {
      return status;
    }
    status = DecodeCoefficientBlock(reader, metadata[index].trace,
                                    candidate.ranks[index].owned_true_trace);
    if (status != CheckpointCodecStatus::ok)
    {
      return status;
    }
  }
  if (!reader.AtEnd()) { return CheckpointCodecStatus::trailing_bytes; }
  candidate.descriptor = descriptor;
  candidate.card_id = card_id;
  candidate.case_id = case_id;
  candidate.checkpoint_id = checkpoint_id;
  candidate.publication = publication;
  candidate.aggregate_interior_coefficients = aggregate_interior_count;
  candidate.aggregate_true_trace_coefficients = aggregate_trace_count;
  candidate.artifact_hash.bytes = checksum;
  checkpoint = std::move(candidate);
  return CheckpointCodecStatus::ok;
}

} // namespace dg::hdg::runtime::app

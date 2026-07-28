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
    'D', 'G', 'I', 'G', 'R', '-', 'H', 'D', 'G', '-', 'M', 'E', 'M', 'V',
    '1', 0}};
constexpr std::size_t checksum_size = StateHash{}.bytes.size();
constexpr std::size_t total_size_offset = magic.size() + sizeof(std::uint32_t);
constexpr std::size_t prefix_size =
    magic.size() + sizeof(std::uint32_t) + sizeof(std::uint64_t);
constexpr std::size_t block_metadata_size =
    sizeof(std::uint32_t) + sizeof(std::uint64_t) + sizeof(std::uint64_t) +
    sizeof(std::uint8_t) + sizeof(std::uint64_t) + StateHash{}.bytes.size();
constexpr std::size_t rank_record_size =
    sizeof(std::uint32_t) + 2 * block_metadata_size;

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
                     [](const std::uint8_t value) { return value != 0; });
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
    const StableIdentity candidate = StableIdentity::Copy(std::string_view{
        reinterpret_cast<const char*>(data_ + offset_), length});
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

  std::size_t offset() const noexcept { return offset_; }
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
      value.coordinates.generation == value.coordinates.step &&
      std::isfinite(value.coordinates.time) && std::isfinite(value.dt) &&
      value.dt > 0.0 && std::isfinite(value.coordinates.dt) &&
      value.coordinates.dt > 0.0 &&
      DoubleBits(value.dt) == DoubleBits(value.coordinates.dt) &&
      value.coordinate_fingerprint ==
          ::dg::hdg::runtime::detail::CoordinateFingerprint(
              value.coordinates) &&
      NonzeroHash(value.interior_hash) && NonzeroHash(value.trace_hash);
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

struct ArtifactConfig
{
  StableIdentity card_id{};
  StableIdentity case_id{};
  StableIdentity scenario_id{};
  StableIdentity checkpoint_id{};
  StableIdentity mesh_family_id{};
  StableIdentity source_id{};
  StableIdentity flux_id{};
  StableIdentity time_rule_id{};
  StableIdentity layout_id{};
  StateHash scientific_config_sha256{};
  StateHash run_config_sha256{};
  ApplicationScenarioV1 scenario = ApplicationScenarioV1::unknown;
  InteriorSpace interior_space = InteriorSpace::l2;
  std::uint32_t mesh_n = 0;
  int polynomial_order = -1;
  int q_a = 0;
  int q_e = 0;
  int q_e4 = 0;
  int q_o = 0;
  double gamma = 0.0;
  double tau = 0.0;
  double dt = 0.0;
  std::uint64_t accepted_steps = 0;
  double final_time = 0.0;
  std::array<std::uint64_t, 3> output_steps{};
  std::uint64_t checkpoint_step = 0;
  RestartMode restart_mode = RestartMode::same_backend_rank_partition;
  RestartTrajectoryModeV1 restart_trajectory =
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple;
  std::uint64_t failure_committed_step = 0;
  std::uint64_t failure_target_step = 0;
  FailurePhase failure_phase = FailurePhase::Setup;
  RollbackObligationV1 rollback_obligation =
      RollbackObligationV1::collective_exact_complete_state;
  DeferredSourceRankRuleV1 source_rank_rule =
      DeferredSourceRankRuleV1::rank_zero_for_rank1_rank_one_for_rank2;
  std::uint32_t retry_limit = 0;
  std::uint64_t seed = 0;
};

bool WriteConfig(Writer& writer, const ValidatedApplicationConfig& config)
{
  if (!writer.Identity(config.card_id()) ||
      !writer.Identity(config.case_id()) ||
      !writer.Identity(config.scenario_id()) ||
      !writer.Identity(config.checkpoint_id()) ||
      !writer.Identity(config.mesh_family_id()) ||
      !writer.Identity(config.source_id()) ||
      !writer.Identity(config.flux_id()) ||
      !writer.Identity(config.time_rule_id()) ||
      !writer.Identity(config.layout_id()))
  {
    return false;
  }
  writer.Hash(config.scientific_config_sha256());
  writer.Hash(config.run_config_sha256());
  writer.U8(static_cast<std::uint8_t>(config.scenario()));
  writer.U8(static_cast<std::uint8_t>(config.interior_space()));
  writer.UnsignedLittleEndian(config.mesh_n());
  writer.UnsignedLittleEndian(
      static_cast<std::uint32_t>(config.polynomial_order()));
  writer.UnsignedLittleEndian(static_cast<std::uint32_t>(config.q_a()));
  writer.UnsignedLittleEndian(static_cast<std::uint32_t>(config.q_e()));
  writer.UnsignedLittleEndian(static_cast<std::uint32_t>(config.q_e4()));
  writer.UnsignedLittleEndian(static_cast<std::uint32_t>(config.q_o()));
  writer.UnsignedLittleEndian(DoubleBits(config.gamma()));
  writer.UnsignedLittleEndian(DoubleBits(config.tau()));
  writer.UnsignedLittleEndian(DoubleBits(config.dt()));
  writer.UnsignedLittleEndian(config.accepted_steps());
  writer.UnsignedLittleEndian(DoubleBits(config.final_time()));
  for (const std::uint64_t step : config.output_steps())
  {
    writer.UnsignedLittleEndian(step);
  }
  writer.UnsignedLittleEndian(config.checkpoint_step());
  writer.U8(static_cast<std::uint8_t>(config.restart_mode()));
  writer.U8(static_cast<std::uint8_t>(config.restart_trajectory()));
  writer.UnsignedLittleEndian(config.injected_failure_committed_step());
  writer.UnsignedLittleEndian(config.injected_failure_target_step());
  writer.U8(static_cast<std::uint8_t>(config.injected_failure_phase()));
  writer.U8(static_cast<std::uint8_t>(config.rollback_obligation()));
  writer.U8(static_cast<std::uint8_t>(config.source_rank_rule()));
  writer.UnsignedLittleEndian(config.retry_limit());
  writer.UnsignedLittleEndian(config.seed());
  return true;
}

bool ReadConfig(Reader& reader, ArtifactConfig& value) noexcept
{
  ArtifactConfig candidate{};
  std::uint8_t scenario = 0;
  std::uint8_t interior_space = 0;
  std::uint32_t polynomial_order = 0;
  std::uint32_t q_a = 0;
  std::uint32_t q_e = 0;
  std::uint32_t q_e4 = 0;
  std::uint32_t q_o = 0;
  std::uint64_t gamma = 0;
  std::uint64_t tau = 0;
  std::uint64_t dt = 0;
  std::uint64_t final_time = 0;
  std::uint8_t restart = 0;
  std::uint8_t trajectory = 0;
  std::uint8_t failure_phase = 0;
  std::uint8_t rollback = 0;
  std::uint8_t source_rule = 0;
  if (!reader.Identity(candidate.card_id) ||
      !reader.Identity(candidate.case_id) ||
      !reader.Identity(candidate.scenario_id) ||
      !reader.Identity(candidate.checkpoint_id) ||
      !reader.Identity(candidate.mesh_family_id) ||
      !reader.Identity(candidate.source_id) ||
      !reader.Identity(candidate.flux_id) ||
      !reader.Identity(candidate.time_rule_id) ||
      !reader.Identity(candidate.layout_id) ||
      !reader.Hash(candidate.scientific_config_sha256) ||
      !reader.Hash(candidate.run_config_sha256) || !reader.U8(scenario) ||
      !reader.U8(interior_space) ||
      !reader.UnsignedLittleEndian(candidate.mesh_n) ||
      !reader.UnsignedLittleEndian(polynomial_order) ||
      !reader.UnsignedLittleEndian(q_a) ||
      !reader.UnsignedLittleEndian(q_e) ||
      !reader.UnsignedLittleEndian(q_e4) ||
      !reader.UnsignedLittleEndian(q_o) ||
      !reader.UnsignedLittleEndian(gamma) ||
      !reader.UnsignedLittleEndian(tau) ||
      !reader.UnsignedLittleEndian(dt) ||
      !reader.UnsignedLittleEndian(candidate.accepted_steps) ||
      !reader.UnsignedLittleEndian(final_time))
  {
    return false;
  }
  for (std::uint64_t& step : candidate.output_steps)
  {
    if (!reader.UnsignedLittleEndian(step)) { return false; }
  }
  if (!reader.UnsignedLittleEndian(candidate.checkpoint_step) ||
      !reader.U8(restart) || !reader.U8(trajectory) ||
      !reader.UnsignedLittleEndian(candidate.failure_committed_step) ||
      !reader.UnsignedLittleEndian(candidate.failure_target_step) ||
      !reader.U8(failure_phase) || !reader.U8(rollback) ||
      !reader.U8(source_rule) ||
      !reader.UnsignedLittleEndian(candidate.retry_limit) ||
      !reader.UnsignedLittleEndian(candidate.seed))
  {
    return false;
  }
  candidate.scenario = static_cast<ApplicationScenarioV1>(scenario);
  candidate.interior_space = static_cast<InteriorSpace>(interior_space);
  candidate.polynomial_order = abi_detail::DecodeInt32(polynomial_order);
  candidate.q_a = abi_detail::DecodeInt32(q_a);
  candidate.q_e = abi_detail::DecodeInt32(q_e);
  candidate.q_e4 = abi_detail::DecodeInt32(q_e4);
  candidate.q_o = abi_detail::DecodeInt32(q_o);
  candidate.gamma = BitsDouble(gamma);
  candidate.tau = BitsDouble(tau);
  candidate.dt = BitsDouble(dt);
  candidate.final_time = BitsDouble(final_time);
  candidate.restart_mode = static_cast<RestartMode>(restart);
  candidate.restart_trajectory =
      static_cast<RestartTrajectoryModeV1>(trajectory);
  candidate.failure_phase = static_cast<FailurePhase>(failure_phase);
  candidate.rollback_obligation =
      static_cast<RollbackObligationV1>(rollback);
  candidate.source_rank_rule =
      static_cast<DeferredSourceRankRuleV1>(source_rule);
  value = candidate;
  return true;
}

bool ConfigMatches(const ArtifactConfig& value,
                   const ValidatedApplicationConfig& expected) noexcept
{
  return value.card_id == expected.card_id() &&
      value.case_id == expected.case_id() &&
      value.scenario_id == expected.scenario_id() &&
      value.checkpoint_id == expected.checkpoint_id() &&
      value.mesh_family_id == expected.mesh_family_id() &&
      value.source_id == expected.source_id() &&
      value.flux_id == expected.flux_id() &&
      value.time_rule_id == expected.time_rule_id() &&
      value.layout_id == expected.layout_id() &&
      value.scientific_config_sha256 == expected.scientific_config_sha256() &&
      value.run_config_sha256 == expected.run_config_sha256() &&
      value.scenario == expected.scenario() &&
      value.interior_space == expected.interior_space() &&
      value.mesh_n == expected.mesh_n() &&
      value.polynomial_order == expected.polynomial_order() &&
      value.q_a == expected.q_a() && value.q_e == expected.q_e() &&
      value.q_e4 == expected.q_e4() && value.q_o == expected.q_o() &&
      DoubleBits(value.gamma) == DoubleBits(expected.gamma()) &&
      DoubleBits(value.tau) == DoubleBits(expected.tau()) &&
      DoubleBits(value.dt) == DoubleBits(expected.dt()) &&
      value.accepted_steps == expected.accepted_steps() &&
      DoubleBits(value.final_time) == DoubleBits(expected.final_time()) &&
      value.output_steps == expected.output_steps() &&
      value.checkpoint_step == expected.checkpoint_step() &&
      value.restart_mode == expected.restart_mode() &&
      value.restart_trajectory == expected.restart_trajectory() &&
      value.failure_committed_step ==
          expected.injected_failure_committed_step() &&
      value.failure_target_step == expected.injected_failure_target_step() &&
      value.failure_phase == expected.injected_failure_phase() &&
      value.rollback_obligation == expected.rollback_obligation() &&
      value.source_rank_rule == expected.source_rank_rule() &&
      value.retry_limit == expected.retry_limit() &&
      value.seed == expected.seed();
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
  if (block.data == nullptr || block.coefficient_count == 0 ||
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

CheckpointCodecStatus PreflightCoefficientBlock(
    Reader& reader,
    const BlockMetadata& metadata) noexcept
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
    if (!std::isfinite(BitsDouble(bits)))
    {
      return CheckpointCodecStatus::noncanonical_fp64;
    }
  }
  Sha256Digest digest{};
  if (!hash.Finalize(digest) || digest != metadata.hash.bytes)
  {
    return CheckpointCodecStatus::local_payload_hash_mismatch;
  }
  return CheckpointCodecStatus::ok;
}

void CopyCoefficientBlock(Reader& reader,
                          const BlockMetadata& metadata,
                          double* destination) noexcept
{
  std::array<std::uint8_t, 8> bytes{};
  for (std::uint64_t index = 0; index < metadata.coefficient_count; ++index)
  {
    static_cast<void>(reader.Bytes(bytes.data(), bytes.size()));
    std::uint64_t bits = 0;
    for (std::size_t byte = 0; byte < bytes.size(); ++byte)
    {
      bits |= static_cast<std::uint64_t>(bytes[byte]) << (8U * byte);
    }
    destination[index] = BitsDouble(bits);
  }
}

bool BufferMatches(const Fp64CoefficientBufferV1& buffer,
                   const BlockMetadata& metadata) noexcept
{
  return buffer.data != nullptr &&
      buffer.capacity >= metadata.coefficient_count &&
      buffer.coefficient_count == metadata.coefficient_count &&
      buffer.component_count == metadata.component_count &&
      buffer.dofs_per_component == metadata.dofs_per_component &&
      buffer.ordering == metadata.ordering;
}

} // namespace codec_detail

bool ValidCheckpointRuntimeIdentityV1(
    const ValidatedApplicationConfig& config,
    const CheckpointRuntimeIdentityV1& identity) noexcept
{
  if (!identity.physical_gpu_uuid.valid() ||
      identity.physical_gpu_uuid.empty() ||
      !codec_detail::NonzeroHash(identity.ownership_hash))
  {
    return false;
  }
  switch (config.deferred_runtime().gpu_rule)
  {
    case DeferredGpuRuleV1::no_cuda_device:
      return identity.physical_gpu_uuid.view() == no_physical_gpu_uuid_v1;
    case DeferredGpuRuleV1::one_physical_gpu:
    case DeferredGpuRuleV1::one_shared_physical_gpu:
      return identity.physical_gpu_uuid.view() != no_physical_gpu_uuid_v1;
  }
  return false;
}

CheckpointCodecStatus ValidateSameTupleRestartV1(
    const CheckpointDescriptor& checkpoint,
    const ValidatedApplicationConfig& expected_config,
    const PersistentResourceDescriptor& expected_resources,
    const CheckpointRuntimeIdentityV1& expected_runtime_identity,
    const CheckpointRuntimeIdentityV1& artifact_runtime_identity) noexcept
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
  if (!ValidCheckpointRuntimeIdentityV1(expected_config,
                                        expected_runtime_identity) ||
      expected_runtime_identity.physical_gpu_uuid !=
          artifact_runtime_identity.physical_gpu_uuid ||
      expected_runtime_identity.ownership_hash !=
          artifact_runtime_identity.ownership_hash)
  {
    return CheckpointCodecStatus::invalid_runtime_identity;
  }
  if (checkpoint.mfem_pin != expected_resources.mfem_pin ||
      checkpoint.mesh_hash != expected_resources.mesh_hash ||
      checkpoint.partition_hash != expected_resources.partition_hash ||
      checkpoint.topology_hash != expected_resources.topology_hash ||
      checkpoint.rank_count != expected_resources.rank_count ||
      checkpoint.rank_count !=
          expected_config.deferred_runtime().required_mpi_rank_count ||
      checkpoint.polynomial_order != expected_resources.polynomial_order ||
      checkpoint.polynomial_order != expected_config.polynomial_order() ||
      checkpoint.layout != expected_resources.layout ||
      checkpoint.layout != expected_config.layout_id() ||
      checkpoint.precision != expected_config.tuple().precision ||
      checkpoint.coordinates.step != expected_config.checkpoint_step() ||
      checkpoint.coordinates.cursor != expected_config.checkpoint_step() ||
      DoubleBits(checkpoint.dt) != DoubleBits(expected_config.dt()))
  {
    return CheckpointCodecStatus::resource_mismatch;
  }
  return CheckpointCodecStatus::ok;
}

CheckpointCodecStatus EncodeCheckpointV1(
    const CapturedCheckpointV1& checkpoint,
    const StagedStateViewV1* ranks,
    const std::size_t rank_count,
    std::vector<std::uint8_t>& encoded)
{
  using namespace codec_detail;
  try
  {
    const ValidatedApplicationConfig& config = checkpoint.config();
    const CheckpointRuntimeIdentityV1& runtime =
        checkpoint.runtime_identity();
    const CheckpointDescriptor& captured_descriptor = checkpoint.descriptor();
    if (!DescriptorStructurallyValid(captured_descriptor) ||
        !ValidCheckpointRuntimeIdentityV1(config, runtime) ||
        captured_descriptor.tuple != config.tuple() ||
        captured_descriptor.rank_count !=
            config.deferred_runtime().required_mpi_rank_count ||
        captured_descriptor.polynomial_order != config.polynomial_order() ||
        captured_descriptor.coordinates.step != config.checkpoint_step())
    {
      return CheckpointCodecStatus::invalid_descriptor;
    }

    std::array<RankMetadata, application_checkpoint_max_rank_count_v1>
        metadata{};
    std::uint64_t interior_count = 0;
    std::uint64_t trace_count = 0;
    CheckpointCodecStatus status = InspectRanks(
        ranks, rank_count, captured_descriptor.rank_count, metadata,
        interior_count, trace_count);
    if (status != CheckpointCodecStatus::ok) { return status; }
    const std::uint32_t source_rank =
        config.deferred_runtime().required_injected_source_rank;
    if (source_rank >= rank_count ||
        metadata[source_rank].interior.hash !=
            captured_descriptor.interior_hash ||
        metadata[source_rank].trace.hash != captured_descriptor.trace_hash)
    {
      return CheckpointCodecStatus::local_payload_hash_mismatch;
    }
    RankPayloadHashV1 aggregate_interior{};
    RankPayloadHashV1 aggregate_trace{};
    if (!AggregateHashes(metadata, rank_count, aggregate_interior,
                         aggregate_trace))
    {
      return CheckpointCodecStatus::aggregate_payload_hash_mismatch;
    }
    CheckpointDescriptor descriptor = captured_descriptor;
    descriptor.interior_hash = aggregate_interior.hash;
    descriptor.trace_hash = aggregate_trace.hash;

    Writer writer;
    writer.Bytes(magic.data(), magic.size());
    writer.UnsignedLittleEndian(application_checkpoint_codec_version_v1);
    writer.UnsignedLittleEndian(std::uint64_t{0});
    if (!WriteConfig(writer, config) ||
        !writer.Identity(runtime.physical_gpu_uuid))
    {
      return CheckpointCodecStatus::invalid_config_identity;
    }
    writer.Hash(runtime.ownership_hash);
    if (!WriteDescriptor(writer, descriptor))
    {
      return CheckpointCodecStatus::invalid_descriptor;
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

CheckpointCodecStatus DecodeCheckpointV1(
    const ConstByteView encoded,
    const ValidatedApplicationConfig& expected_config,
    const PersistentResourceDescriptor& expected_resources,
    const CheckpointRuntimeIdentityV1& expected_runtime_identity,
    RankStateBufferV1* rank_buffers,
    const std::size_t rank_buffer_count,
    DecodedCheckpointMetadataV1& output_metadata)
{
  using namespace codec_detail;
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
  ArtifactConfig artifact_config{};
  CheckpointRuntimeIdentityV1 artifact_runtime{};
  CheckpointDescriptor descriptor{};
  if (!ReadConfig(reader, artifact_config) ||
      !reader.Identity(artifact_runtime.physical_gpu_uuid) ||
      !reader.Hash(artifact_runtime.ownership_hash) ||
      !ReadDescriptor(reader, descriptor))
  {
    return CheckpointCodecStatus::truncated;
  }
  if (!ConfigMatches(artifact_config, expected_config))
  {
    return CheckpointCodecStatus::invalid_config_identity;
  }
  if (!DescriptorStructurallyValid(descriptor))
  {
    return CheckpointCodecStatus::invalid_descriptor;
  }
  CheckpointCodecStatus status = ValidateSameTupleRestartV1(
      descriptor, expected_config, expected_resources,
      expected_runtime_identity, artifact_runtime);
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
      rank_count != expected_config.deferred_runtime().required_mpi_rank_count)
  {
    return CheckpointCodecStatus::rank_tuple_mismatch;
  }
  if (rank_buffers == nullptr || rank_buffer_count != rank_count)
  {
    return CheckpointCodecStatus::output_buffer_mismatch;
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
    if (rank_buffers[index].rank != index ||
        !BufferMatches(rank_buffers[index].interior, rank.interior) ||
        !BufferMatches(rank_buffers[index].owned_true_trace, rank.trace))
    {
      return CheckpointCodecStatus::output_buffer_mismatch;
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

  const std::size_t payload_offset = reader.offset();
  for (std::uint32_t index = 0; index < rank_count; ++index)
  {
    status = PreflightCoefficientBlock(reader, metadata[index].interior);
    if (status != CheckpointCodecStatus::ok) { return status; }
    status = PreflightCoefficientBlock(reader, metadata[index].trace);
    if (status != CheckpointCodecStatus::ok) { return status; }
  }
  if (!reader.AtEnd()) { return CheckpointCodecStatus::trailing_bytes; }

  DecodedCheckpointMetadataV1 candidate_metadata{};
  candidate_metadata.descriptor = descriptor;
  candidate_metadata.card_id = artifact_config.card_id;
  candidate_metadata.case_id = artifact_config.case_id;
  candidate_metadata.scenario_id = artifact_config.scenario_id;
  candidate_metadata.checkpoint_id = artifact_config.checkpoint_id;
  candidate_metadata.physical_gpu_uuid = artifact_runtime.physical_gpu_uuid;
  candidate_metadata.scientific_config_sha256 =
      artifact_config.scientific_config_sha256;
  candidate_metadata.run_config_sha256 = artifact_config.run_config_sha256;
  candidate_metadata.ownership_hash = artifact_runtime.ownership_hash;
  candidate_metadata.mesh_n = artifact_config.mesh_n;
  candidate_metadata.rank_count = rank_count;
  candidate_metadata.aggregate_interior_coefficients =
      aggregate_interior_count;
  candidate_metadata.aggregate_true_trace_coefficients =
      aggregate_trace_count;
  candidate_metadata.artifact_hash.bytes = checksum;

  Reader payload(encoded.data + payload_offset,
                 static_cast<std::size_t>(expected_payload_bytes));
  for (std::uint32_t index = 0; index < rank_count; ++index)
  {
    CopyCoefficientBlock(payload, metadata[index].interior,
                         rank_buffers[index].interior.data);
    CopyCoefficientBlock(payload, metadata[index].trace,
                         rank_buffers[index].owned_true_trace.data);
  }
  output_metadata = candidate_metadata;
  return CheckpointCodecStatus::ok;
}

} // namespace dg::hdg::runtime::app

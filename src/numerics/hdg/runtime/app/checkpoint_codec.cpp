#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"

#include "../cpu/hdg_cpu_serial_lifecycle.hpp"

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
    '2', 0}};
constexpr std::uint8_t restart_delta_schema_v2 = 1;
constexpr std::uint8_t restart_delta_field_count_v2 = 6;
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

struct AddressRange
{
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

bool MakeAddressRange(const void* data,
                      const std::size_t size,
                      AddressRange& range) noexcept
{
  if (data == nullptr || size == 0) { return false; }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
  if (size > std::numeric_limits<std::uintptr_t>::max() - begin)
  {
    return false;
  }
  range = {begin, begin + size};
  return true;
}

bool Overlaps(const AddressRange left, const AddressRange right) noexcept
{
  return left.begin < right.end && right.begin < left.end;
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

  template <std::size_t Capacity>
  bool Identity(const StableId<Capacity>& identity)
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

  template <std::size_t Capacity>
  bool Identity(StableId<Capacity>& identity) noexcept
  {
    std::uint16_t length = 0;
    if (!UnsignedLittleEndian(length) || length == 0 || !Available(length))
    {
      return false;
    }
    const StableId<Capacity> candidate = StableId<Capacity>::Copy(
        std::string_view{reinterpret_cast<const char*>(data_ + offset_),
                         length});
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

bool DescriptorStructurallyValid(
    const CheckpointDescriptor& value,
    const ValidatedApplicationConfig& config) noexcept
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
      ScenarioCheckpointTimeMatchesV1(value.coordinates.time, config) &&
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
  StateHash writer_run_config_sha256{};
  ApplicationScenarioV1 scenario = ApplicationScenarioV1::unknown;
  ApplicationBackend backend = ApplicationBackend::unknown;
  InteriorSpace interior_space = InteriorSpace::l2;
  DeferredRuntimePredicatesV1 deferred{};
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
  std::uint64_t scenario_checkpoint_step = 0;
  ApplicationRunRoleV1 run_role =
      ApplicationRunRoleV1::uninterrupted_reference;
  struct Path
  {
    bool present = false;
    StableId<512> identity{};
  };
  Path output_directory{};
  Path evidence_directory{};
  ValidationCaptureModeV1 validation_capture = ValidationCaptureModeV1::none;
  Path checkpoint_out{};
  OptionalCheckpointStepV1 checkpoint_step{};
  Path restart_from{};
  StopAfterCheckpointV1 stop_after_checkpoint = StopAfterCheckpointV1::no;
  FailureInjectionControlV1 injection{};
  RestartMode restart_mode = RestartMode::same_backend_rank_partition;
  RestartTrajectoryModeV1 restart_trajectory =
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple;
  RollbackObligationV1 rollback_obligation =
      RollbackObligationV1::collective_exact_complete_state;
  DeferredSourceRankRuleV1 source_rank_rule =
      DeferredSourceRankRuleV1::rank_zero_for_rank1_rank_one_for_rank2;
  std::uint32_t retry_limit = 0;
  std::uint64_t seed = 0;
};

template <typename Path>
bool WritePath(Writer& writer, const Path& path)
{
  writer.U8(path.present() ? 1U : 0U);
  return !path.present() || writer.Identity(path.identity());
}

bool ReadPath(Reader& reader, ArtifactConfig::Path& path) noexcept
{
  std::uint8_t present = 0;
  if (!reader.U8(present) || present > 1) { return false; }
  path.present = present != 0;
  return !path.present || reader.Identity(path.identity);
}

template <typename Path>
bool SamePath(const ArtifactConfig::Path& artifact,
              const Path& expected) noexcept
{
  return artifact.present == expected.present() &&
      (!artifact.present || artifact.identity == expected.identity());
}

struct RestartDeltaManifestV2
{
  std::uint8_t schema = 0;
  std::uint8_t count = 0;
  std::uint8_t role_field = 0;
  ApplicationRunRoleV1 writer_role =
      ApplicationRunRoleV1::uninterrupted_reference;
  ApplicationRunRoleV1 reader_role =
      ApplicationRunRoleV1::uninterrupted_reference;
  std::uint8_t binding_field = 0;
  std::uint8_t checkpoint_binding = 0;
  std::uint8_t checkpoint_step_field = 0;
  OptionalCheckpointStepV1 writer_checkpoint_step{};
  OptionalCheckpointStepV1 reader_checkpoint_step{};
  std::uint8_t stop_field = 0;
  StopAfterCheckpointV1 writer_stop = StopAfterCheckpointV1::no;
  StopAfterCheckpointV1 reader_stop = StopAfterCheckpointV1::no;
  std::uint8_t injection_field = 0;
  FailureInjectionControlV1 writer_injection{};
  FailureInjectionControlV1 reader_injection{};
  std::uint8_t retry_field = 0;
  std::uint32_t writer_retry = 0;
  std::uint32_t reader_retry = 0;
};

void WriteInjection(Writer& writer,
                    const FailureInjectionControlV1 value)
{
  writer.U8(static_cast<std::uint8_t>(value.mode));
  writer.U8(static_cast<std::uint8_t>(value.phase));
  writer.UnsignedLittleEndian(value.target_step);
  writer.UnsignedLittleEndian(value.source_rank);
}

bool ReadInjection(Reader& reader,
                   FailureInjectionControlV1& value) noexcept
{
  std::uint8_t mode = 0;
  std::uint8_t phase = 0;
  if (!reader.U8(mode) || !reader.U8(phase) ||
      !reader.UnsignedLittleEndian(value.target_step) ||
      !reader.UnsignedLittleEndian(value.source_rank))
  {
    return false;
  }
  value.mode = static_cast<FailureInjectionModeV1>(mode);
  value.phase = static_cast<FailurePhase>(phase);
  return true;
}

void WriteOptionalStep(Writer& writer,
                       const OptionalCheckpointStepV1 value)
{
  writer.U8(value.present ? 1U : 0U);
  writer.UnsignedLittleEndian(value.value);
}

bool ReadOptionalStep(Reader& reader,
                      OptionalCheckpointStepV1& value) noexcept
{
  std::uint8_t present = 0;
  if (!reader.U8(present) || present > 1 ||
      !reader.UnsignedLittleEndian(value.value))
  {
    return false;
  }
  value.present = present != 0;
  return value.present || value.value == 0;
}

void WriteRestartDeltaManifestV2(
    Writer& writer,
    const ValidatedApplicationConfig& writer_config)
{
  const bool scalar =
      writer_config.scenario() == ApplicationScenarioV1::scalar_igr_sc25_v3;
  const bool smooth_transition = writer_config.case_id().view() ==
      cpu::w8c::SmoothTransitionV2Case::case_id;
  const FailureInjectionControlV1 scalar_writer_injection{
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      scalar_igr_sc25_v3_failure_target_step,
      writer_config.deferred_runtime().required_injected_source_rank};
  const FailureInjectionControlV1 conventional_reader_injection{
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      writer_config.scenario() == ApplicationScenarioV1::smoke
          ? smooth_periodic_euler_v1_smoke_failure_target_step
          : smooth_periodic_euler_v1_functional_failure_target_step,
      writer_config.deferred_runtime().required_injected_source_rank};
  const FailureInjectionControlV1 writer_injection = smooth_transition
      ? FailureInjectionControlV1{}
      : (scalar ? scalar_writer_injection : FailureInjectionControlV1{});
  const FailureInjectionControlV1 reader_injection = smooth_transition
      ? FailureInjectionControlV1{}
      : (scalar ? FailureInjectionControlV1{}
                : conventional_reader_injection);
  writer.U8(restart_delta_schema_v2);
  writer.U8(restart_delta_field_count_v2);
  writer.U8(1);
  writer.U8(static_cast<std::uint8_t>(ApplicationRunRoleV1::split_writer));
  writer.U8(static_cast<std::uint8_t>(ApplicationRunRoleV1::split_reader));
  writer.U8(2);
  writer.U8(1); // checkpoint-out identity becomes restart-from identity
  writer.U8(3);
  WriteOptionalStep(writer, OptionalCheckpointStepV1::Present(
                                writer_config.scenario_checkpoint_step()));
  WriteOptionalStep(writer, {});
  writer.U8(4);
  writer.U8(static_cast<std::uint8_t>(StopAfterCheckpointV1::yes));
  writer.U8(static_cast<std::uint8_t>(StopAfterCheckpointV1::no));
  writer.U8(5);
  WriteInjection(writer, writer_injection);
  WriteInjection(writer, reader_injection);
  writer.U8(6);
  writer.UnsignedLittleEndian(smooth_transition
      ? std::uint32_t{0}
      : (scalar ? scalar_igr_sc25_v3_retry_limit : std::uint32_t{0}));
  writer.UnsignedLittleEndian(smooth_transition
      ? std::uint32_t{0}
      : (scalar ? std::uint32_t{0}
                : smooth_periodic_euler_v1_retry_limit));
}

bool ReadRestartDeltaManifestV2(
    Reader& reader,
    RestartDeltaManifestV2& value) noexcept
{
  std::uint8_t writer_role = 0;
  std::uint8_t reader_role = 0;
  std::uint8_t writer_stop = 0;
  std::uint8_t reader_stop = 0;
  if (!reader.U8(value.schema) || !reader.U8(value.count) ||
      !reader.U8(value.role_field) || !reader.U8(writer_role) ||
      !reader.U8(reader_role) || !reader.U8(value.binding_field) ||
      !reader.U8(value.checkpoint_binding) ||
      !reader.U8(value.checkpoint_step_field) ||
      !ReadOptionalStep(reader, value.writer_checkpoint_step) ||
      !ReadOptionalStep(reader, value.reader_checkpoint_step) ||
      !reader.U8(value.stop_field) || !reader.U8(writer_stop) ||
      !reader.U8(reader_stop) || !reader.U8(value.injection_field) ||
      !ReadInjection(reader, value.writer_injection) ||
      !ReadInjection(reader, value.reader_injection) ||
      !reader.U8(value.retry_field) ||
      !reader.UnsignedLittleEndian(value.writer_retry) ||
      !reader.UnsignedLittleEndian(value.reader_retry))
  {
    return false;
  }
  value.writer_role = static_cast<ApplicationRunRoleV1>(writer_role);
  value.reader_role = static_cast<ApplicationRunRoleV1>(reader_role);
  value.writer_stop = static_cast<StopAfterCheckpointV1>(writer_stop);
  value.reader_stop = static_cast<StopAfterCheckpointV1>(reader_stop);
  return true;
}

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
  writer.U8(static_cast<std::uint8_t>(config.backend()));
  writer.U8(static_cast<std::uint8_t>(config.interior_space()));
  writer.UnsignedLittleEndian(
      config.deferred_runtime().required_mpi_rank_count);
  writer.U8(static_cast<std::uint8_t>(config.deferred_runtime().gpu_rule));
  writer.U8(static_cast<std::uint8_t>(
      config.deferred_runtime().source_rank_rule));
  writer.UnsignedLittleEndian(
      config.deferred_runtime().required_injected_source_rank);
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
  writer.UnsignedLittleEndian(config.scenario_checkpoint_step());
  writer.U8(static_cast<std::uint8_t>(config.run_role()));
  if (!WritePath(writer, config.output_directory()) ||
      !WritePath(writer, config.evidence_directory()))
  {
    return false;
  }
  writer.U8(static_cast<std::uint8_t>(config.validation_capture()));
  if (!WritePath(writer, config.checkpoint_out())) { return false; }
  WriteOptionalStep(writer, config.checkpoint_step());
  if (!WritePath(writer, config.restart_from())) { return false; }
  writer.U8(static_cast<std::uint8_t>(config.stop_after_checkpoint()));
  WriteInjection(writer, config.injection());
  writer.U8(static_cast<std::uint8_t>(config.restart_mode()));
  writer.U8(static_cast<std::uint8_t>(config.restart_trajectory()));
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
  std::uint8_t backend = 0;
  std::uint8_t interior_space = 0;
  std::uint8_t gpu_rule = 0;
  std::uint8_t deferred_source_rule = 0;
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
  std::uint8_t run_role = 0;
  std::uint8_t capture = 0;
  std::uint8_t stop = 0;
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
      !reader.Hash(candidate.writer_run_config_sha256) ||
      !reader.U8(scenario) || !reader.U8(backend) ||
      !reader.U8(interior_space) ||
      !reader.UnsignedLittleEndian(
          candidate.deferred.required_mpi_rank_count) ||
      !reader.U8(gpu_rule) || !reader.U8(deferred_source_rule) ||
      !reader.UnsignedLittleEndian(
          candidate.deferred.required_injected_source_rank) ||
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
  if (!reader.UnsignedLittleEndian(candidate.scenario_checkpoint_step) ||
      !reader.U8(run_role) || !ReadPath(reader, candidate.output_directory) ||
      !ReadPath(reader, candidate.evidence_directory) ||
      !reader.U8(capture) || !ReadPath(reader, candidate.checkpoint_out) ||
      !ReadOptionalStep(reader, candidate.checkpoint_step) ||
      !ReadPath(reader, candidate.restart_from) || !reader.U8(stop) ||
      !ReadInjection(reader, candidate.injection) ||
      !reader.U8(restart) || !reader.U8(trajectory) ||
      !reader.U8(rollback) ||
      !reader.U8(source_rule) ||
      !reader.UnsignedLittleEndian(candidate.retry_limit) ||
      !reader.UnsignedLittleEndian(candidate.seed))
  {
    return false;
  }
  candidate.scenario = static_cast<ApplicationScenarioV1>(scenario);
  candidate.backend = static_cast<ApplicationBackend>(backend);
  candidate.interior_space = static_cast<InteriorSpace>(interior_space);
  candidate.deferred.gpu_rule = static_cast<DeferredGpuRuleV1>(gpu_rule);
  candidate.deferred.source_rank_rule =
      static_cast<DeferredSourceRankRuleV1>(deferred_source_rule);
  candidate.polynomial_order = abi_detail::DecodeInt32(polynomial_order);
  candidate.q_a = abi_detail::DecodeInt32(q_a);
  candidate.q_e = abi_detail::DecodeInt32(q_e);
  candidate.q_e4 = abi_detail::DecodeInt32(q_e4);
  candidate.q_o = abi_detail::DecodeInt32(q_o);
  candidate.gamma = BitsDouble(gamma);
  candidate.tau = BitsDouble(tau);
  candidate.dt = BitsDouble(dt);
  candidate.final_time = BitsDouble(final_time);
  candidate.run_role = static_cast<ApplicationRunRoleV1>(run_role);
  candidate.validation_capture =
      static_cast<ValidationCaptureModeV1>(capture);
  candidate.stop_after_checkpoint =
      static_cast<StopAfterCheckpointV1>(stop);
  candidate.restart_mode = static_cast<RestartMode>(restart);
  candidate.restart_trajectory =
      static_cast<RestartTrajectoryModeV1>(trajectory);
  candidate.rollback_obligation =
      static_cast<RollbackObligationV1>(rollback);
  candidate.source_rank_rule =
      static_cast<DeferredSourceRankRuleV1>(source_rule);
  value = candidate;
  return true;
}

bool ConfigMatchesSplitRestart(
    const ArtifactConfig& value,
    const RestartDeltaManifestV2& delta,
    const ValidatedApplicationConfig& expected) noexcept
{
  StateHash derived_writer_run_hash{};
  const FailureInjectionControlV1 canonical_off{};
  const bool scalar = expected.scalar_igr();
  const bool smooth_transition = expected.case_id().view() ==
      cpu::w8c::SmoothTransitionV2Case::case_id;
  const FailureInjectionControlV1 scalar_writer_injection{
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      scalar_igr_sc25_v3_failure_target_step,
      expected.deferred_runtime().required_injected_source_rank};
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
      NonzeroHash(value.writer_run_config_sha256) &&
      NonzeroHash(expected.run_config_sha256()) &&
      value.writer_run_config_sha256 != expected.run_config_sha256() &&
      DeriveSplitWriterRunConfigHashV1(expected, derived_writer_run_hash) &&
      value.writer_run_config_sha256 == derived_writer_run_hash &&
      value.scenario == expected.scenario() &&
      value.backend == expected.backend() &&
      value.interior_space == expected.interior_space() &&
      value.deferred.required_mpi_rank_count ==
          expected.deferred_runtime().required_mpi_rank_count &&
      value.deferred.gpu_rule == expected.deferred_runtime().gpu_rule &&
      value.deferred.source_rank_rule ==
          expected.deferred_runtime().source_rank_rule &&
      value.deferred.required_injected_source_rank ==
          expected.deferred_runtime().required_injected_source_rank &&
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
      value.scenario_checkpoint_step == expected.scenario_checkpoint_step() &&
      value.run_role == ApplicationRunRoleV1::split_writer &&
      expected.run_role() == ApplicationRunRoleV1::split_reader &&
      SamePath(value.output_directory, expected.output_directory()) &&
      SamePath(value.evidence_directory, expected.evidence_directory()) &&
      value.validation_capture == expected.validation_capture() &&
      value.checkpoint_out.present && !value.restart_from.present &&
      !expected.checkpoint_out().present() && expected.restart_from().present() &&
      value.checkpoint_out.identity == expected.restart_from().identity() &&
      value.checkpoint_step.present &&
      value.checkpoint_step.value == expected.scenario_checkpoint_step() &&
      !expected.checkpoint_step().present &&
      value.stop_after_checkpoint == StopAfterCheckpointV1::yes &&
      expected.stop_after_checkpoint() == StopAfterCheckpointV1::no &&
      value.injection == (smooth_transition
          ? canonical_off
          : (scalar ? scalar_writer_injection : canonical_off)) &&
      expected.injection() == (smooth_transition
          ? canonical_off
          : (scalar ? canonical_off : delta.reader_injection)) &&
      value.restart_mode == expected.restart_mode() &&
      value.restart_trajectory == expected.restart_trajectory() &&
      value.rollback_obligation == expected.rollback_obligation() &&
      value.source_rank_rule == expected.source_rank_rule() &&
      value.retry_limit == (smooth_transition
          ? 0U : (scalar ? scalar_igr_sc25_v3_retry_limit : 0U)) &&
      expected.retry_limit() == (smooth_transition
          ? 0U : (scalar ? 0U : smooth_periodic_euler_v1_retry_limit)) &&
      value.seed == expected.seed() &&
      delta.schema == restart_delta_schema_v2 &&
      delta.count == restart_delta_field_count_v2 &&
      delta.role_field == 1 &&
      delta.writer_role == value.run_role &&
      delta.reader_role == expected.run_role() &&
      delta.binding_field == 2 && delta.checkpoint_binding == 1 &&
      delta.checkpoint_step_field == 3 &&
      delta.writer_checkpoint_step.present &&
      delta.writer_checkpoint_step.value == value.checkpoint_step.value &&
      !delta.reader_checkpoint_step.present &&
      delta.reader_checkpoint_step.value == 0 &&
      delta.stop_field == 4 &&
      delta.writer_stop == value.stop_after_checkpoint &&
      delta.reader_stop == expected.stop_after_checkpoint() &&
      delta.injection_field == 5 &&
      delta.writer_injection == value.injection &&
      delta.reader_injection == expected.injection() &&
      delta.retry_field == 6 && delta.writer_retry == value.retry_limit &&
      delta.reader_retry == expected.retry_limit();
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
    const std::uint32_t expected_component_count,
    const std::uint64_t expected_dofs_per_component,
    const std::uint64_t exact_coefficient_count,
    BlockMetadata& metadata) noexcept
{
  std::uint64_t expected_count = 0;
  std::uint64_t byte_count = 0;
  if (block.data == nullptr || block.coefficient_count == 0 ||
      block.component_count != expected_component_count ||
      block.ordering != CoefficientOrderingV1::component_major)
  {
    return CheckpointCodecStatus::invalid_coefficient_schema;
  }
  if (block.dofs_per_component != expected_dofs_per_component ||
      block.coefficient_count != exact_coefficient_count ||
      !Multiply(block.component_count, block.dofs_per_component,
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
    const ScenarioCoefficientExtentsV1& extents,
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
    CheckpointCodecStatus status = ValidateBlockView(
        ranks[index].interior,
        extents.interior_component_count,
        extents.ranks[index].interior_dofs_per_component,
        extents.ranks[index].interior_coefficient_count,
        candidate.interior);
    if (status != CheckpointCodecStatus::ok) { return status; }
    status = ValidateBlockView(
        ranks[index].owned_true_trace,
        extents.trace_component_count,
        extents.ranks[index].trace_dofs_per_component,
        extents.ranks[index].trace_coefficient_count,
        candidate.trace);
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
    const BlockMetadata& block,
    const std::uint32_t expected_component_count,
    const std::uint64_t expected_dofs_per_component,
    const std::uint64_t exact_coefficient_count) noexcept
{
  std::uint64_t expected_count = 0;
  std::uint64_t expected_bytes = 0;
  if (block.component_count != expected_component_count ||
      block.ordering != CoefficientOrderingV1::component_major)
  {
    return CheckpointCodecStatus::invalid_coefficient_schema;
  }
  if (block.dofs_per_component != expected_dofs_per_component ||
      block.coefficient_count != exact_coefficient_count ||
      !Multiply(block.component_count, block.dofs_per_component,
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

bool DecodeDestinationsAreDisjoint(
    const ConstByteView encoded,
    const RankStateBufferV1* rank_buffers,
    const std::uint32_t rank_count,
    const std::array<RankMetadata, application_checkpoint_max_rank_count_v1>&
        metadata,
    const DecodedCheckpointMetadataV1& output_metadata) noexcept
{
  AddressRange encoded_range{};
  AddressRange metadata_range{};
  if (!MakeAddressRange(encoded.data, encoded.size, encoded_range) ||
      !MakeAddressRange(&output_metadata, sizeof(output_metadata),
                        metadata_range) ||
      Overlaps(encoded_range, metadata_range))
  {
    return false;
  }

  std::array<AddressRange, 2 * application_checkpoint_max_rank_count_v1>
      outputs{};
  std::size_t output_count = 0;
  for (std::uint32_t rank = 0; rank < rank_count; ++rank)
  {
    const std::array<const Fp64CoefficientBufferV1*, 2> blocks{{
        &rank_buffers[rank].interior,
        &rank_buffers[rank].owned_true_trace}};
    const std::array<const BlockMetadata*, 2> block_metadata{{
        &metadata[rank].interior, &metadata[rank].trace}};
    for (std::size_t block = 0; block < blocks.size(); ++block)
    {
      std::uint64_t byte_count = 0;
      if (!Multiply(block_metadata[block]->coefficient_count,
                    fp64_coefficient_bytes_v1, byte_count) ||
          byte_count > std::numeric_limits<std::size_t>::max() ||
          !MakeAddressRange(
              blocks[block]->data, static_cast<std::size_t>(byte_count),
              outputs[output_count]) ||
          Overlaps(outputs[output_count], encoded_range) ||
          Overlaps(outputs[output_count], metadata_range))
      {
        return false;
      }
      for (std::size_t prior = 0; prior < output_count; ++prior)
      {
        if (Overlaps(outputs[output_count], outputs[prior])) { return false; }
      }
      ++output_count;
    }
  }
  return true;
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

std::optional<CapturedCheckpointV1> CaptureCheckpointPiecesV1(
    const CheckpointDescriptor* rank_descriptors,
    const std::size_t rank_count,
    const ValidatedApplicationConfig& config,
    const CheckpointRuntimeIdentityV1& runtime_identity) noexcept
{
  if (rank_descriptors == nullptr || rank_count == 0 ||
      rank_count > application_checkpoint_max_rank_count_v1 ||
      rank_count != config.deferred_runtime().required_mpi_rank_count ||
      config.run_role() != ApplicationRunRoleV1::split_writer ||
      !ValidCheckpointRuntimeIdentityV1(config, runtime_identity))
  {
    return std::nullopt;
  }
  std::array<CheckpointDescriptor, application_checkpoint_max_rank_count_v1>
      descriptors{};
  for (std::size_t rank = 0; rank < rank_count; ++rank)
  {
    const CheckpointDescriptor& descriptor = rank_descriptors[rank];
    if (!codec_detail::DescriptorStructurallyValid(descriptor, config) ||
        descriptor.tuple != config.tuple() ||
        descriptor.rank_count != rank_count ||
        descriptor.polynomial_order != config.polynomial_order() ||
        descriptor.coordinates.step != config.scenario_checkpoint_step() ||
        descriptor.coordinates.generation !=
            config.scenario_checkpoint_step() ||
        !ScenarioCheckpointTimeMatchesV1(descriptor.coordinates.time, config))
    {
      return std::nullopt;
    }
    if (rank != 0)
    {
      CheckpointDescriptor common = descriptor;
      CheckpointDescriptor first = descriptors[0];
      common.interior_hash = {};
      common.trace_hash = {};
      first.interior_hash = {};
      first.trace_hash = {};
      if (!::dg::hdg::runtime::detail::SameCheckpoint(first, common))
      {
        return std::nullopt;
      }
    }
    descriptors[rank] = descriptor;
  }
  return CapturedCheckpointV1(
      descriptors, static_cast<std::uint32_t>(rank_count), config,
      runtime_identity);
}

bool ScenarioCheckpointTimeMatchesV1(
    const double time,
    const ValidatedApplicationConfig& config) noexcept
{
  if (!std::isfinite(time) || !std::isfinite(config.dt())) { return false; }
  const double expected =
      static_cast<double>(config.scenario_checkpoint_step()) * config.dt();
  if (!std::isfinite(expected)) { return false; }
  const double scale = std::max(1.0, std::abs(expected));
  return std::abs(time - expected) <=
      8.0 * std::numeric_limits<double>::epsilon() * scale;
}

namespace detail
{
namespace
{
template <bool ToComponentMajor>
bool TransposeScalarIgrInteriorV1(
    const std::vector<double>& source,
    const std::uint64_t scalar_dofs_per_element,
    std::vector<double>& destination) noexcept
{
  constexpr std::uint64_t u_components = 4;
  constexpr std::uint64_t sigma_components = 1;
  constexpr std::uint64_t z_components = 2;
  constexpr std::uint64_t components =
      u_components + sigma_components + z_components;
  if (source.empty() || scalar_dofs_per_element == 0 ||
      source.size() % components != 0)
    return false;
  const std::uint64_t dofs_per_component = source.size() / components;
  if (dofs_per_component % scalar_dofs_per_element != 0)
    return false;
  const std::uint64_t elements =
      dofs_per_component / scalar_dofs_per_element;
  try
  {
    std::vector<double> candidate(source.size(), 0.0);
    const std::uint64_t u_count = u_components * dofs_per_component;
    const std::uint64_t sigma_count = dofs_per_component;
    for (std::uint64_t element = 0; element < elements; ++element)
    {
      for (std::uint64_t node = 0; node < scalar_dofs_per_element; ++node)
      {
        for (std::uint64_t component = 0; component < u_components;
             ++component)
        {
          const std::uint64_t blocked =
              element * u_components * scalar_dofs_per_element +
              component * scalar_dofs_per_element + node;
          const std::uint64_t component_major =
              component * dofs_per_component +
              element * scalar_dofs_per_element + node;
          if constexpr (ToComponentMajor)
            candidate[component_major] = source[blocked];
          else
            candidate[blocked] = source[component_major];
        }
        const std::uint64_t sigma_blocked =
            u_count + element * scalar_dofs_per_element + node;
        const std::uint64_t sigma_major =
            u_components * dofs_per_component +
            element * scalar_dofs_per_element + node;
        if constexpr (ToComponentMajor)
          candidate[sigma_major] = source[sigma_blocked];
        else
          candidate[sigma_blocked] = source[sigma_major];
        for (std::uint64_t component = 0; component < z_components;
             ++component)
        {
          const std::uint64_t blocked =
              u_count + sigma_count +
              element * z_components * scalar_dofs_per_element +
              component * scalar_dofs_per_element + node;
          const std::uint64_t component_major =
              (u_components + sigma_components + component) *
                  dofs_per_component +
              element * scalar_dofs_per_element + node;
          if constexpr (ToComponentMajor)
            candidate[component_major] = source[blocked];
          else
            candidate[blocked] = source[component_major];
        }
      }
    }
    destination = std::move(candidate);
    return true;
  }
  catch (...) { return false; }
}
} // namespace

bool ScalarIgrFieldBlocksToComponentMajorV1(
    const std::vector<double>& field_blocks,
    const std::uint64_t scalar_dofs_per_element,
    std::vector<double>& component_major) noexcept
{
  return TransposeScalarIgrInteriorV1<true>(
      field_blocks, scalar_dofs_per_element, component_major);
}

bool ScalarIgrComponentMajorToFieldBlocksV1(
    const std::vector<double>& component_major,
    const std::uint64_t scalar_dofs_per_element,
    std::vector<double>& field_blocks) noexcept
{
  return TransposeScalarIgrInteriorV1<false>(
      component_major, scalar_dofs_per_element, field_blocks);
}
} // namespace detail

bool DeriveScenarioCoefficientExtentsV1(
    const ValidatedApplicationConfig& config,
    const std::uint32_t rank_count,
    ScenarioCoefficientExtentsV1& extents) noexcept
{
  if (rank_count == 0 ||
      rank_count > application_checkpoint_max_rank_count_v1 ||
      rank_count != config.deferred_runtime().required_mpi_rank_count ||
      config.mesh_n() == 0 || config.polynomial_order() < 0)
  {
    return false;
  }
  std::uint64_t element_count = 0;
  if (!codec_detail::Multiply(config.mesh_n(), config.mesh_n(),
                              element_count) ||
      element_count == 0)
  {
    return false;
  }
  std::array<std::uint64_t, application_checkpoint_max_rank_count_v1>
      owned_elements{};
  std::array<std::uint64_t, application_checkpoint_max_rank_count_v1>
      owned_faces{};
  const auto element_rank = [rank_count, element_count](
                                const std::uint64_t element) noexcept {
    return static_cast<std::uint32_t>((element * rank_count) / element_count);
  };
  for (std::uint64_t element = 0; element < element_count; ++element)
  {
    ++owned_elements[element_rank(element)];
  }
  const std::uint64_t n = config.mesh_n();
  for (std::uint64_t y = 0; y < n; ++y)
  {
    for (std::uint64_t x = 0; x < n; ++x)
    {
      const std::uint64_t element = y * n + x;
      const std::uint64_t left = y * n + (x + n - 1) % n;
      const std::uint64_t below = ((y + n - 1) % n) * n + x;
      ++owned_faces[std::min(element_rank(left), element_rank(element))];
      ++owned_faces[std::min(element_rank(below), element_rank(element))];
    }
  }

  ScenarioCoefficientExtentsV1 candidate{};
  candidate.rank_count = rank_count;
  const CheckpointCoefficientSchemaV1 schema =
      CheckpointCoefficientSchema(config);
  candidate.interior_component_count = schema.interior_component_count;
  candidate.trace_component_count = schema.trace_component_count;
  if (schema.interior_component_count == 0 ||
      schema.trace_component_count == 0)
    return false;
  const std::uint64_t p1 =
      static_cast<std::uint64_t>(config.polynomial_order() + 1);
  std::uint64_t interior_basis = 0;
  if (!codec_detail::Multiply(p1, p1, interior_basis)) { return false; }
  for (std::uint32_t rank = 0; rank < rank_count; ++rank)
  {
    RankCoefficientExtentsV1& value = candidate.ranks[rank];
    if (!codec_detail::Multiply(owned_elements[rank], interior_basis,
                                value.interior_dofs_per_component) ||
        !codec_detail::Multiply(value.interior_dofs_per_component,
                                schema.interior_component_count,
                                value.interior_coefficient_count) ||
        !codec_detail::Multiply(owned_faces[rank], p1,
                                value.trace_dofs_per_component) ||
        !codec_detail::Multiply(value.trace_dofs_per_component,
                                schema.trace_component_count,
                                value.trace_coefficient_count) ||
        !codec_detail::Add(candidate.aggregate_interior_coefficient_count,
                           value.interior_coefficient_count,
                           candidate.aggregate_interior_coefficient_count) ||
        !codec_detail::Add(candidate.aggregate_trace_coefficient_count,
                           value.trace_coefficient_count,
                           candidate.aggregate_trace_coefficient_count))
    {
      return false;
    }
  }
  extents = candidate;
  return true;
}

CheckpointCoefficientSchemaV1 CheckpointCoefficientSchema(
    const ValidatedApplicationConfig& config) noexcept
{
  return config.scalar_igr()
      ? CheckpointCoefficientSchemaV1{
            scalar_igr_interior_component_count_v1,
            scalar_igr_trace_component_count_v1}
      : CheckpointCoefficientSchemaV1{
            euler_state_component_count_v1,
            euler_state_component_count_v1};
}

CheckpointCodecStatus ValidateSameTupleRestartV1(
    const CheckpointDescriptor& checkpoint,
    const ValidatedApplicationConfig& expected_config,
    const PersistentResourceDescriptor& expected_resources,
    const CheckpointRuntimeIdentityV1& expected_runtime_identity,
    const CheckpointRuntimeIdentityV1& artifact_runtime_identity) noexcept
{
  if (!codec_detail::DescriptorStructurallyValid(checkpoint,
                                                 expected_config) ||
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
      checkpoint.coordinates.step !=
          expected_config.scenario_checkpoint_step() ||
      checkpoint.coordinates.cursor !=
          expected_config.scenario_checkpoint_step() ||
      !ScenarioCheckpointTimeMatchesV1(checkpoint.coordinates.time,
                                       expected_config) ||
      codec_detail::DoubleBits(checkpoint.dt) !=
          codec_detail::DoubleBits(expected_config.dt()))
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
    const CheckpointDescriptor& captured_descriptor =
        checkpoint.rank_descriptor(0);
    if (checkpoint.rank_count() != rank_count ||
        !DescriptorStructurallyValid(captured_descriptor, config) ||
        !ValidCheckpointRuntimeIdentityV1(config, runtime) ||
        captured_descriptor.tuple != config.tuple() ||
        captured_descriptor.rank_count !=
            config.deferred_runtime().required_mpi_rank_count ||
        captured_descriptor.polynomial_order != config.polynomial_order() ||
        captured_descriptor.coordinates.step !=
            config.scenario_checkpoint_step() ||
        config.run_role() != ApplicationRunRoleV1::split_writer)
    {
      return CheckpointCodecStatus::invalid_descriptor;
    }
    ScenarioCoefficientExtentsV1 extents{};
    if (!DeriveScenarioCoefficientExtentsV1(
            config, captured_descriptor.rank_count, extents))
    {
      return CheckpointCodecStatus::invalid_descriptor;
    }

    std::array<RankMetadata, application_checkpoint_max_rank_count_v1>
        metadata{};
    std::uint64_t interior_count = 0;
    std::uint64_t trace_count = 0;
    CheckpointCodecStatus status = InspectRanks(
        ranks, rank_count, captured_descriptor.rank_count, extents, metadata,
        interior_count, trace_count);
    if (status != CheckpointCodecStatus::ok) { return status; }
    if (interior_count != extents.aggregate_interior_coefficient_count ||
        trace_count != extents.aggregate_trace_coefficient_count)
    {
      return CheckpointCodecStatus::coefficient_count_mismatch;
    }
    for (std::size_t rank = 0; rank < rank_count; ++rank)
    {
      const CheckpointDescriptor& live = checkpoint.rank_descriptor(rank);
      if (!DescriptorStructurallyValid(live, config) ||
          metadata[rank].interior.hash != live.interior_hash ||
          metadata[rank].trace.hash != live.trace_hash)
      {
        return CheckpointCodecStatus::local_payload_hash_mismatch;
      }
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
    writer.UnsignedLittleEndian(application_checkpoint_codec_version_v2);
    writer.UnsignedLittleEndian(std::uint64_t{0});
    const StableIdentity partition_layout_rule =
        StableIdentity::Copy(checkpoint_partition_layout_rule_id_v1);
    if (!WriteConfig(writer, config))
    {
      return CheckpointCodecStatus::invalid_config_identity;
    }
    WriteRestartDeltaManifestV2(writer, config);
    if (!writer.Identity(runtime.physical_gpu_uuid) ||
        !writer.Identity(partition_layout_rule))
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
  AddressRange encoded_range{};
  AddressRange output_metadata_range{};
  if (encoded.size != 0 &&
      (!MakeAddressRange(encoded.data, encoded.size, encoded_range) ||
       !MakeAddressRange(&output_metadata, sizeof(output_metadata),
                         output_metadata_range) ||
       Overlaps(encoded_range, output_metadata_range)))
  {
    return CheckpointCodecStatus::output_buffer_overlap;
  }
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
  if (version != application_checkpoint_codec_version_v2)
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
  RestartDeltaManifestV2 restart_delta{};
  CheckpointRuntimeIdentityV1 artifact_runtime{};
  StableIdentity artifact_partition_layout_rule{};
  CheckpointDescriptor descriptor{};
  if (!ReadConfig(reader, artifact_config) ||
      !ReadRestartDeltaManifestV2(reader, restart_delta) ||
      !reader.Identity(artifact_runtime.physical_gpu_uuid) ||
      !reader.Identity(artifact_partition_layout_rule) ||
      !reader.Hash(artifact_runtime.ownership_hash) ||
      !ReadDescriptor(reader, descriptor))
  {
    return CheckpointCodecStatus::truncated;
  }
  if (!ConfigMatchesSplitRestart(artifact_config, restart_delta,
                                 expected_config))
  {
    return CheckpointCodecStatus::invalid_config_identity;
  }
  if (artifact_partition_layout_rule.view() !=
          checkpoint_partition_layout_rule_id_v1 ||
      !DescriptorStructurallyValid(descriptor, expected_config))
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
  ScenarioCoefficientExtentsV1 extents{};
  if (!DeriveScenarioCoefficientExtentsV1(expected_config, rank_count,
                                          extents))
  {
    return CheckpointCodecStatus::invalid_descriptor;
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
    status = ValidateBlockMetadata(
        rank.interior, extents.interior_component_count,
        extents.ranks[index].interior_dofs_per_component,
        extents.ranks[index].interior_coefficient_count);
    if (status != CheckpointCodecStatus::ok) { return status; }
    status = ValidateBlockMetadata(
        rank.trace, extents.trace_component_count,
        extents.ranks[index].trace_dofs_per_component,
        extents.ranks[index].trace_coefficient_count);
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
  if (!DecodeDestinationsAreDisjoint(encoded, rank_buffers, rank_count,
                                     metadata, output_metadata))
  {
    return CheckpointCodecStatus::output_buffer_overlap;
  }
  if (summed_interior != aggregate_interior_count ||
      summed_trace != aggregate_trace_count ||
      aggregate_interior_count !=
          extents.aggregate_interior_coefficient_count ||
      aggregate_trace_count != extents.aggregate_trace_coefficient_count)
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
  candidate_metadata.partition_layout_rule_id =
      artifact_partition_layout_rule;
  candidate_metadata.writer_run_role = artifact_config.run_role;
  candidate_metadata.scientific_config_sha256 =
      artifact_config.scientific_config_sha256;
  candidate_metadata.writer_run_config_sha256 =
      artifact_config.writer_run_config_sha256;
  candidate_metadata.reader_run_config_sha256 =
      expected_config.run_config_sha256();
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

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::app;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "checkpoint codec test must compile in strict C++17");
static_assert(application_checkpoint_max_rank_count_v1 == 2 &&
                  euler_state_component_count_v1 == 4 &&
                  fp64_coefficient_bytes_v1 == 8,
              "v1 rank, component, and binary64 bounds are exact");
static_assert(!std::is_default_constructible<CapturedCheckpointV1>::value &&
                  std::is_copy_constructible<CapturedCheckpointV1>::value,
              "capture is live-contract-originated but repeatably encodable");

bool Check(const bool condition, const char* name)
{
  if (!condition) { std::cerr << "FAILED: " << name << '\n'; }
  return condition;
}

constexpr std::array<std::uint32_t, 64> oracle_sha256_round{{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U}};

std::uint32_t RotateRight(const std::uint32_t value,
                          const unsigned int bits) noexcept
{
  return (value >> bits) | (value << (32U - bits));
}

class OracleSha256
{
public:
  OracleSha256() noexcept
      : state_{{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U}}
  {}

  void Update(const std::uint8_t* data, const std::size_t size) noexcept
  {
    for (std::size_t index = 0; index < size; ++index)
    {
      buffer_[buffered_++] = data[index];
      ++bytes_;
      if (buffered_ == buffer_.size())
      {
        Transform(buffer_);
        buffered_ = 0;
      }
    }
  }

  StateHash Finalize() noexcept
  {
    const std::uint64_t bit_count = bytes_ * 8U;
    buffer_[buffered_++] = 0x80U;
    if (buffered_ > 56)
    {
      while (buffered_ < buffer_.size()) { buffer_[buffered_++] = 0; }
      Transform(buffer_);
      buffered_ = 0;
    }
    while (buffered_ < 56) { buffer_[buffered_++] = 0; }
    for (int shift = 56; shift >= 0; shift -= 8)
    {
      buffer_[buffered_++] =
          static_cast<std::uint8_t>(bit_count >> shift);
    }
    Transform(buffer_);
    StateHash result{};
    for (std::size_t index = 0; index < state_.size(); ++index)
    {
      result.bytes[4 * index] =
          static_cast<std::uint8_t>(state_[index] >> 24U);
      result.bytes[4 * index + 1] =
          static_cast<std::uint8_t>(state_[index] >> 16U);
      result.bytes[4 * index + 2] =
          static_cast<std::uint8_t>(state_[index] >> 8U);
      result.bytes[4 * index + 3] =
          static_cast<std::uint8_t>(state_[index]);
    }
    return result;
  }

private:
  void Transform(const std::array<std::uint8_t, 64>& block) noexcept
  {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index)
    {
      words[index] =
          (static_cast<std::uint32_t>(block[4 * index]) << 24U) |
          (static_cast<std::uint32_t>(block[4 * index + 1]) << 16U) |
          (static_cast<std::uint32_t>(block[4 * index + 2]) << 8U) |
          static_cast<std::uint32_t>(block[4 * index + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index)
    {
      const std::uint32_t s0 = RotateRight(words[index - 15], 7) ^
          RotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3U);
      const std::uint32_t s1 = RotateRight(words[index - 2], 17) ^
          RotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10U);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index)
    {
      const std::uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^
          RotateRight(e, 25);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
          h + sum1 + choice + oracle_sha256_round[index] + words[index];
      const std::uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^
          RotateRight(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffered_ = 0;
  std::uint64_t bytes_ = 0;
};

StateHash OracleHash(const std::uint8_t* data,
                     const std::size_t size) noexcept
{
  OracleSha256 hash;
  hash.Update(data, size);
  return hash.Finalize();
}

std::uint64_t OracleBits(const double value) noexcept
{
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void AppendU32(std::vector<std::uint8_t>& bytes, const std::uint32_t value)
{
  for (unsigned int shift = 0; shift < 32; shift += 8)
  {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void AppendU64(std::vector<std::uint8_t>& bytes, const std::uint64_t value)
{
  for (unsigned int shift = 0; shift < 64; shift += 8)
  {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

template <std::size_t Size>
StateHash OracleCoefficientHash(
    const std::array<double, Size>& values) noexcept
{
  std::array<std::uint8_t, Size * 8> bytes{};
  std::size_t offset = 0;
  for (const double value : values)
  {
    const std::uint64_t bits = OracleBits(value);
    for (unsigned int shift = 0; shift < 64; shift += 8)
    {
      bytes[offset++] = static_cast<std::uint8_t>(bits >> shift);
    }
  }
  return OracleHash(bytes.data(), bytes.size());
}

StateHash OracleManifestHash(const std::uint8_t kind,
                             const std::uint64_t byte_count,
                             const StateHash& local_hash)
{
  const std::array<std::uint8_t, 25> domain{{
      'D', 'G', 'I', 'G', 'R', '-', 'H', 'D', 'G', '-', 'R', 'A', 'N',
      'K', '-', 'P', 'A', 'Y', 'L', 'O', 'A', 'D', '-', 'V', '1'}};
  std::vector<std::uint8_t> bytes(domain.begin(), domain.end());
  bytes.push_back(kind);
  AppendU32(bytes, 1);
  AppendU32(bytes, 0);
  AppendU64(bytes, byte_count);
  bytes.insert(bytes.end(), local_hash.bytes.begin(), local_hash.bytes.end());
  return OracleHash(bytes.data(), bytes.size());
}

std::string Hex(const StateHash& hash)
{
  constexpr char digits[] = "0123456789abcdef";
  std::string text(hash.bytes.size() * 2, '0');
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    text[2 * index] = digits[hash.bytes[index] >> 4U];
    text[2 * index + 1] = digits[hash.bytes[index] & 0x0fU];
  }
  return text;
}

std::vector<std::uint8_t> HexBytes(const std::string_view text)
{
  const auto nibble = [](const char value) -> std::uint8_t {
    if (value >= '0' && value <= '9')
    {
      return static_cast<std::uint8_t>(value - '0');
    }
    return static_cast<std::uint8_t>(value - 'a' + 10);
  };
  std::vector<std::uint8_t> bytes(text.size() / 2);
  for (std::size_t index = 0; index < bytes.size(); ++index)
  {
    bytes[index] = static_cast<std::uint8_t>(
        (nibble(text[2 * index]) << 4U) | nibble(text[2 * index + 1]));
  }
  return bytes;
}

using State = std::array<double, 4>;
using Slot = CompleteStateSlot<State, State>;

struct TestStoragePolicy
{
  static constexpr bool preparation_preserves_committed = true;
  static constexpr bool private_copy_is_independent = true;
  static constexpr bool publish_is_atomic = true;
  static constexpr bool publish_preserves_disjoint_backing = true;

  static bool Disjoint(const Slot& left, const Slot& right) noexcept
  {
    return &left != &right;
  }
  static bool PrepareCandidate(const Slot& committed, Slot& candidate) noexcept
  {
    candidate = committed;
    return true;
  }
  static bool PrepareRestore(const Slot& loaded, Slot& candidate) noexcept
  {
    candidate = loaded;
    return true;
  }
  static bool StoreCandidate(Slot& candidate, const Slot& value) noexcept
  {
    candidate = value;
    return true;
  }
  static bool CopyOut(const Slot& source, Slot& copy) noexcept
  {
    copy = source;
    return true;
  }
  static void DiscardCandidate(Slot&) noexcept {}
  static bool CanPublish(const Slot& committed,
                         const Slot& candidate) noexcept
  {
    return Disjoint(committed, candidate);
  }
  static void Publish(Slot& committed, Slot& candidate) noexcept
  {
    using std::swap;
    swap(committed, candidate);
  }
  static StateHash HashInterior(const State& values) noexcept
  {
    return OracleCoefficientHash(values);
  }
  static StateHash HashTrace(const State& values) noexcept
  {
    return OracleCoefficientHash(values);
  }
};

struct TestCollectivePolicy
{
  static CollectiveAdmissionReport Admit(
      const AttemptCollectiveContext& context,
      const bool prepared,
      const FailureReason reason,
      const std::uint32_t ranks) noexcept
  {
    return {context, ranks, true, true, prepared,
            reason != FailureReason::none, 0, reason};
  }
  static CollectiveGateReport Reduce(
      const AttemptCollectiveContext& context,
      const GateReport& gates,
      const FailureReason reason,
      const std::uint32_t ranks) noexcept
  {
    return {context, gates, ranks, true, true,
            reason != FailureReason::none, 0, reason};
  }
  static CollectiveRestoreReport ReduceRestore(
      const RestoreCollectiveContext& context,
      const RestartCompatibility compatibility,
      const FailureReason reason,
      const std::uint32_t ranks) noexcept
  {
    return {context, ranks, true, true,
            compatibility != RestartCompatibility::compatible,
            0, 0, reason, compatibility};
  }
};

using Contract = PersistentRunContract<CpuSerialRuntimeSelection,
                                       State,
                                       State,
                                       TestStoragePolicy,
                                       TestCollectivePolicy>;

StateHash FilledHash(const std::uint8_t first) noexcept
{
  StateHash hash{};
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    hash.bytes[index] = static_cast<std::uint8_t>(first + index);
  }
  return hash;
}

PersistentResourceDescriptor TestResources()
{
  PersistentResourceDescriptor resources{};
  resources.mfem_pin = StableIdentity::Copy("d9d6526cc1749980");
  resources.mesh_hash = FilledHash(0x10);
  resources.partition_hash = FilledHash(0x30);
  resources.topology_hash = FilledHash(0x50);
  resources.rank_count = 1;
  resources.polynomial_order = 1;
  resources.layout = StableIdentity::Copy(
      smooth_periodic_euler_v1_layout_id);
  resources.reuse = {true, true, true, false, 1, 1, 128, 64};
  return resources;
}

CheckpointRuntimeIdentityV1 TestRuntimeIdentity()
{
  CheckpointRuntimeIdentityV1 identity{};
  identity.ownership_hash = FilledHash(0x70);
  return identity;
}

ApplicationValidation TestValidation()
{
  ApplicationOptionsView options;
  options.scientific_config_sha256 = FilledHash(0x90);
  options.run_config_sha256 = FilledHash(0xb0);
  return ValidateApplicationOptions(options);
}

Contract::ProductBundle IssueAll(Contract& contract,
                                 const AttemptIdentity& attempt)
{
  Contract::ProductBundle products{};
  products.residual =
      contract.IssueProduct<DerivedProductKind::residual>(attempt);
  products.matrix = contract.IssueProduct<DerivedProductKind::matrix>(attempt);
  products.factor = contract.IssueProduct<DerivedProductKind::factor>(attempt);
  products.solution =
      contract.IssueProduct<DerivedProductKind::solution>(attempt);
  return products;
}

bool AdvanceOneCommit(Contract& contract)
{
  const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.05});
  if (!begin.started) { return false; }
  AttemptReport report{};
  report.gates = {true, true, true, true, true, true};
  report.telemetry.transfers = {0, 0, 0, 0, 0, true};
  report.telemetry.true_relative_residual = 0.0;
  report.candidate_cursor = 1;
  return contract.FinishAttempt(begin.attempt, report,
                                IssueAll(contract, begin.attempt)).committed;
}

struct CoefficientFixture
{
  State interior{{1.0, -2.0, 3.5, -0.0}};
  State trace{{5.0, 6.25, -7.5, 8.0}};

  std::array<StagedStateViewV1, 1> Views() const noexcept
  {
    return {{{0,
              {interior.data(), 4, 4, 1,
               CoefficientOrderingV1::component_major},
              {trace.data(), 4, 4, 1,
               CoefficientOrderingV1::component_major}}}};
  }
};

std::optional<CapturedCheckpointV1> MakeCapture(
    const ValidatedApplicationConfig& config,
    const PersistentResourceDescriptor& resources,
    const CheckpointRuntimeIdentityV1& runtime,
    const CoefficientFixture& fixture)
{
  const Slot committed{fixture.interior, fixture.trace};
  Slot candidate{};
  Contract contract(committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  if (!AdvanceOneCommit(contract)) { return std::nullopt; }
  return CaptureCheckpointV1(contract, config, runtime);
}

bool EncodeFixture(const CoefficientFixture& fixture,
                   const ValidatedApplicationConfig& config,
                   const PersistentResourceDescriptor& resources,
                   const CheckpointRuntimeIdentityV1& runtime,
                   std::vector<std::uint8_t>& encoded)
{
  const auto views = fixture.Views();
  const auto capture = MakeCapture(config, resources, runtime, fixture);
  return capture &&
      EncodeCheckpointV1(*capture, views.data(), views.size(), encoded) ==
      CheckpointCodecStatus::ok;
}

std::uint64_t ReadU64(const std::vector<std::uint8_t>& bytes,
                      const std::size_t offset)
{
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 8; ++index)
  {
    value |= static_cast<std::uint64_t>(bytes[offset + index]) <<
        (8U * index);
  }
  return value;
}

void WriteU32(std::vector<std::uint8_t>& bytes,
              const std::size_t offset,
              const std::uint32_t value)
{
  for (std::size_t index = 0; index < 4; ++index)
  {
    bytes[offset + index] =
        static_cast<std::uint8_t>(value >> (8U * index));
  }
}

void WriteU64(std::vector<std::uint8_t>& bytes,
              const std::size_t offset,
              const std::uint64_t value)
{
  for (std::size_t index = 0; index < 8; ++index)
  {
    bytes[offset + index] =
        static_cast<std::uint8_t>(value >> (8U * index));
  }
}

bool Seal(std::vector<std::uint8_t>& bytes)
{
  if (bytes.size() < 32) { return false; }
  const StateHash hash = OracleHash(bytes.data(), bytes.size() - 32);
  std::copy(hash.bytes.begin(), hash.bytes.end(), bytes.end() - 32);
  return true;
}

struct NamedOffset
{
  const char* name = nullptr;
  std::size_t offset = 0;
};

struct LayoutOffsets
{
  std::vector<NamedOffset> exact_fields{};
  std::size_t descriptor_interior_hash = 0;
  std::size_t manifest_rank_count = 0;
  std::size_t aggregate_interior_count = 0;
  std::size_t rank_record = 0;
  std::size_t interior_coefficient_count = 0;
  std::size_t interior_hash = 0;
  std::size_t payload = 0;
};

std::size_t IdentityField(const std::vector<std::uint8_t>& bytes,
                          std::size_t offset,
                          const char* name,
                          std::vector<NamedOffset>& fields)
{
  const std::uint16_t size = static_cast<std::uint16_t>(bytes[offset]) |
      static_cast<std::uint16_t>(bytes[offset + 1] << 8U);
  fields.push_back({name, offset + 2});
  return offset + 2 + size;
}

LayoutOffsets Locate(const std::vector<std::uint8_t>& bytes)
{
  LayoutOffsets layout{};
  std::size_t offset = 28;
  for (const char* name :
       {"config.card", "config.case", "config.scenario",
        "config.checkpoint", "config.mesh-family", "config.source",
        "config.flux", "config.time-rule", "config.layout"})
  {
    offset = IdentityField(bytes, offset, name, layout.exact_fields);
  }
  for (const char* name : {"config.scientific-sha", "config.run-sha"})
  {
    layout.exact_fields.push_back({name, offset});
    offset += 32;
  }
  const std::array<std::pair<const char*, std::size_t>, 24> config_scalars{{
      {"config.scenario-enum", 1}, {"config.interior-space", 1},
      {"config.mesh-n", 4},
      {"config.p", 4},             {"config.qA", 4},
      {"config.qE", 4},            {"config.qE4", 4},
      {"config.qO", 4},            {"config.gamma", 8},
      {"config.tau", 8},           {"config.dt", 8},
      {"config.steps", 8},         {"config.final-time", 8},
      {"config.output-0", 8},      {"config.output-1", 8},
      {"config.output-2", 8},      {"config.checkpoint-step", 8},
      {"config.restart", 1},       {"config.trajectory", 1},
      {"config.failure-commit", 8},{"config.failure-target", 8},
      {"config.failure-phase", 1}, {"config.rollback", 1},
      {"config.source-rule", 1}}};
  for (const auto& field : config_scalars)
  {
    layout.exact_fields.push_back({field.first, offset});
    offset += field.second;
  }
  layout.exact_fields.push_back({"config.retry", offset});
  offset += 4;
  layout.exact_fields.push_back({"config.seed", offset});
  offset += 8;
  offset = IdentityField(bytes, offset, "runtime.gpu-uuid",
                         layout.exact_fields);
  layout.exact_fields.push_back({"runtime.ownership-hash", offset});
  offset += 32;

  layout.exact_fields.push_back({"descriptor.schema", offset});
  offset += 4;
  layout.exact_fields.push_back({"descriptor.success", offset++});
  layout.exact_fields.push_back({"descriptor.restart", offset++});
  for (const char* name :
       {"descriptor.model", "descriptor.formulation", "descriptor.trace",
        "descriptor.stabilization", "descriptor.time", "descriptor.backend",
        "descriptor.precision", "descriptor.mfem-pin"})
  {
    offset = IdentityField(bytes, offset, name, layout.exact_fields);
  }
  for (const char* name : {"descriptor.mesh-hash",
                           "descriptor.partition-hash",
                           "descriptor.topology-hash"})
  {
    layout.exact_fields.push_back({name, offset});
    offset += 32;
  }
  layout.exact_fields.push_back({"descriptor.rank-count", offset});
  offset += 4;
  layout.exact_fields.push_back({"descriptor.p", offset});
  offset += 4;
  offset = IdentityField(bytes, offset, "descriptor.layout",
                         layout.exact_fields);
  offset = IdentityField(bytes, offset, "descriptor.precision-repeat",
                         layout.exact_fields);
  for (const char* name : {"descriptor.cursor", "descriptor.dt",
                           "descriptor.time-coordinate", "descriptor.step",
                           "descriptor.generation", "descriptor.coord-cursor",
                           "descriptor.coord-dt", "descriptor.fingerprint"})
  {
    layout.exact_fields.push_back({name, offset});
    offset += 8;
  }
  layout.descriptor_interior_hash = offset;
  layout.exact_fields.push_back({"descriptor.interior-hash", offset});
  offset += 32;
  layout.exact_fields.push_back({"descriptor.trace-hash", offset});
  offset += 32;

  layout.manifest_rank_count = offset;
  layout.exact_fields.push_back({"manifest.rank-count", offset});
  offset += 4;
  layout.aggregate_interior_count = offset;
  layout.exact_fields.push_back({"manifest.interior-count", offset});
  offset += 8;
  layout.exact_fields.push_back({"manifest.trace-count", offset});
  offset += 8;
  layout.rank_record = offset;
  layout.exact_fields.push_back({"rank.rank", offset});
  offset += 4;
  const auto block = [&](const char* prefix) {
    layout.exact_fields.push_back({prefix, offset});
    offset += 4;
    layout.exact_fields.push_back({"rank.coefficient-count", offset});
    if (layout.interior_coefficient_count == 0)
    {
      layout.interior_coefficient_count = offset;
    }
    offset += 8;
    layout.exact_fields.push_back({"rank.dofs-per-component", offset});
    offset += 8;
    layout.exact_fields.push_back({"rank.ordering", offset});
    offset += 1;
    layout.exact_fields.push_back({"rank.byte-count", offset});
    offset += 8;
    layout.exact_fields.push_back({"rank.local-hash", offset});
    if (layout.interior_hash == 0) { layout.interior_hash = offset; }
    offset += 32;
  };
  block("rank.interior-components");
  block("rank.trace-components");
  layout.payload = offset;
  return layout;
}

bool SameDescriptorIndependent(const CheckpointDescriptor& left,
                               const CheckpointDescriptor& right)
{
  return left.schema_version == right.schema_version &&
      left.successful_commit == right.successful_commit &&
      left.restart_mode == right.restart_mode && left.tuple == right.tuple &&
      left.mfem_pin == right.mfem_pin && left.mesh_hash == right.mesh_hash &&
      left.partition_hash == right.partition_hash &&
      left.topology_hash == right.topology_hash &&
      left.rank_count == right.rank_count &&
      left.polynomial_order == right.polynomial_order &&
      left.layout == right.layout && left.precision == right.precision &&
      left.cursor == right.cursor &&
      OracleBits(left.dt) == OracleBits(right.dt) &&
      OracleBits(left.coordinates.time) ==
          OracleBits(right.coordinates.time) &&
      left.coordinates.step == right.coordinates.step &&
      left.coordinates.generation == right.coordinates.generation &&
      left.coordinates.cursor == right.coordinates.cursor &&
      OracleBits(left.coordinates.dt) ==
          OracleBits(right.coordinates.dt) &&
      left.coordinate_fingerprint == right.coordinate_fingerprint &&
      left.interior_hash == right.interior_hash &&
      left.trace_hash == right.trace_hash;
}

struct DecodeSentinel
{
  State interior{{101.0, 102.0, 103.0, 104.0}};
  State trace{{201.0, 202.0, 203.0, 204.0}};
  DecodedCheckpointMetadataV1 metadata{};

  DecodeSentinel()
  {
    metadata.descriptor.schema_version = 77;
    metadata.descriptor.mfem_pin = StableIdentity::Copy("sentinel-mfem");
    metadata.card_id = StableIdentity::Copy("sentinel-card");
    metadata.case_id = StableIdentity::Copy("sentinel-case");
    metadata.scenario_id = StableIdentity::Copy("sentinel-scenario");
    metadata.checkpoint_id = StableIdentity::Copy("sentinel-checkpoint");
    metadata.physical_gpu_uuid = StableIdentity::Copy("sentinel-gpu");
    metadata.scientific_config_sha256 = FilledHash(0xd0);
    metadata.run_config_sha256 = FilledHash(0xe0);
    metadata.ownership_hash = FilledHash(0xf0);
    metadata.mesh_n = 99;
    metadata.rank_count = 88;
    metadata.aggregate_interior_coefficients = 77;
    metadata.aggregate_true_trace_coefficients = 66;
    metadata.artifact_hash = FilledHash(0xa0);
  }

  std::array<RankStateBufferV1, 1> Buffers() noexcept
  {
    return {{{0,
              {interior.data(), interior.size(), interior.size(), 4, 1,
               CoefficientOrderingV1::component_major},
              {trace.data(), trace.size(), trace.size(), 4, 1,
               CoefficientOrderingV1::component_major}}}};
  }
};

bool SameMetadata(const DecodedCheckpointMetadataV1& left,
                  const DecodedCheckpointMetadataV1& right)
{
  return SameDescriptorIndependent(left.descriptor, right.descriptor) &&
      left.card_id == right.card_id && left.case_id == right.case_id &&
      left.scenario_id == right.scenario_id &&
      left.checkpoint_id == right.checkpoint_id &&
      left.physical_gpu_uuid == right.physical_gpu_uuid &&
      left.scientific_config_sha256 == right.scientific_config_sha256 &&
      left.run_config_sha256 == right.run_config_sha256 &&
      left.ownership_hash == right.ownership_hash &&
      left.mesh_n == right.mesh_n && left.rank_count == right.rank_count &&
      left.aggregate_interior_coefficients ==
          right.aggregate_interior_coefficients &&
      left.aggregate_true_trace_coefficients ==
          right.aggregate_true_trace_coefficients &&
      left.artifact_hash == right.artifact_hash;
}

bool SentinelUnchanged(const DecodeSentinel& value,
                       const DecodeSentinel& original)
{
  return value.interior == original.interior && value.trace == original.trace &&
      SameMetadata(value.metadata, original.metadata);
}

// Filled from an independent fieldwise Python oracle after the format is
// frozen.  The literal includes every artifact byte and the final SHA-256.
constexpr std::string_view golden_artifact_hex =
    "44474947522d4844472d4d454d56310001000000dc0400000000000033006578702e6864672e65756c65722e"
    "636f6e76656e74696f6e616c2e736d6f6f74685f70657273697374656e742e7738612e76312100636f6e7665"
    "6e74696f6e616c2d736d6f6f74682d70657273697374656e742d763109005738412d534d4f4b453400636865"
    "636b706f696e742e6864672e65756c65722e736d6f6f74682d706572696f6469632e736d6f6b652e73746570"
    "2d312e76311700706572696f6469632d63617274657369616e2d717561641f00736f757263652e65756c6572"
    "2e736d6f6f74682e616e616c797469632e76312900666c75782e6864672e636f6e76656e74696f6e616c2e66"
    "697865642d7363616c61722d7461752e76310e006261636b776172645f65756c65722f006c322d696e746572"
    "696f722e747275652d74726163652d6f776e65642e636f6d706f6e656e742d6d616a6f722e76319091929394"
    "95969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0"
    "c1c2c3c4c5c6c7c8c9cacbcccdcecf000003000000010000000b000000100000001400000015000000666666"
    "666666f63f00000000000004409a9999999999a93f0300000000000000333333333333c33f00000000000000"
    "0001000000000000000300000000000000010000000000000000000100000000000000020000000000000008"
    "000001000000314b4f4d5341385704006e6f6e65707172737475767778797a7b7c7d7e7f8081828384858687"
    "88898a8b8c8d8e8f010000000100150065756c65722e636f6d707265737369626c652e326410006864672e63"
    "6f6e76656e74696f6e616c17006d66656d2e64675f696e746572666163652e747261636509007461752e6669"
    "7865640e006261636b776172645f65756c65720a006370752e73657269616c04006670363410006439643635"
    "3236636331373439393830101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f30"
    "3132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c"
    "5d5e5f606162636465666768696a6b6c6d6e6f01000000010000002f006c322d696e746572696f722e747275"
    "652d74726163652d6f776e65642e636f6d706f6e656e742d6d616a6f722e7631040066703634010000000000"
    "00009a9999999999a93f9a9999999999a93f0100000000000000010000000000000001000000000000009a99"
    "99999999a93f12e4ea20d43b982f66d70231bb8654d88e509ef55a64a36059cd1bbf1f33b6197de3a4acacc1"
    "5098bfb3882f02ddf0634171bd039abb25ef0c9e9d9c287ca402234a394184a57d4301000000040000000000"
    "0000040000000000000000000000040000000400000000000000010000000000000000200000000000000063"
    "b84f617d28be16889d09da2bf88a144d908cdc462a176953c8c67191d6835904000000040000000000000001"
    "00000000000000002000000000000000b0c0959fdf6d33e5435ebb3bb41ea582d274b360e1597fdd886ac1eb"
    "eb361f36000000000000f03f00000000000000c00000000000000c4000000000000000800000000000001440"
    "00000000000019400000000000001ec0000000000000204066f62405b9fe6f9de15874ce90826cabfa8b5d0b"
    "3d7676b5e434dbcec0cae25c";

bool CheckIndependentOracle()
{
  const std::array<std::uint8_t, 3> abc{{'a', 'b', 'c'}};
  return Check(
      Hex(OracleHash(abc.data(), abc.size())) ==
          "ba7816bf8f01cfea414140de5dae2223"
          "b00361a396177a9cb410ff61f20015ad",
      "independent_sha256_matches_hard_coded_FIPS_vector");
}

bool CheckLiveCaptureAndRepeatableEncode()
{
  const CoefficientFixture fixture;
  const PersistentResourceDescriptor resources = TestResources();
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation validation = TestValidation();
  const ValidatedApplicationConfig* config = validation.config();
  bool passed = Check(config != nullptr, "checkpoint_static_config_valid");
  if (config == nullptr) { return false; }

  const Slot committed{fixture.interior, fixture.trace};
  Slot candidate{};
  Contract uncommitted(
      committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  passed = Check(!CaptureCheckpointV1(uncommitted, *config, runtime),
                 "uncommitted_contract_cannot_be_captured") &&
      passed;
  CheckpointRuntimeIdentityV1 invalid_runtime{};
  passed = Check(!ValidCheckpointRuntimeIdentityV1(*config, invalid_runtime),
                 "runtime_identity_requires_ownership_hash") &&
      passed;

  const auto capture = MakeCapture(*config, resources, runtime, fixture);
  passed = Check(capture.has_value(),
                 "successful_live_step1_commit_is_captured") &&
      passed;
  if (!capture) { return false; }
  const CapturedCheckpointV1 copy = *capture;
  const auto views = fixture.Views();
  std::vector<std::uint8_t> first;
  std::vector<std::uint8_t> second;
  passed = Check(EncodeCheckpointV1(*capture, views.data(), views.size(),
                                    first) == CheckpointCodecStatus::ok &&
                     EncodeCheckpointV1(copy, views.data(), views.size(),
                                        second) == CheckpointCodecStatus::ok &&
                     first == second,
                 "committed_capture_can_be_reencoded_without_affine_authority") &&
      passed;
  return passed;
}

bool CheckGoldenRoundTrip()
{
  const CoefficientFixture fixture;
  const PersistentResourceDescriptor resources = TestResources();
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation validation = TestValidation();
  const ValidatedApplicationConfig* config = validation.config();
  if (config == nullptr) { return Check(false, "golden_static_config"); }
  std::vector<std::uint8_t> encoded;
  bool passed = Check(EncodeFixture(fixture, *config, resources, runtime,
                                    encoded),
                      "golden_fixture_encodes");
  const std::vector<std::uint8_t> golden = HexBytes(golden_artifact_hex);
  passed = Check(!golden.empty() && encoded == golden,
                 "entire_artifact_matches_hard_coded_golden_stream") &&
      passed;
  passed = Check(encoded.size() > 64 && encoded[0] == 'D' &&
                     encoded[16] == 1 && ReadU64(encoded, 20) == encoded.size(),
                 "independent_header_field_decode") &&
      passed;

  const StateHash expected_interior = OracleCoefficientHash(fixture.interior);
  const StateHash expected_trace = OracleCoefficientHash(fixture.trace);
  passed = Check(
               Hex(expected_interior) ==
                       "63b84f617d28be16889d09da2bf88a144d908cdc462a176953c8c67191d68359" &&
                   Hex(expected_trace) ==
                       "b0c0959fdf6d33e5435ebb3bb41ea582d274b360e1597fdd886ac1ebeb361f36",
               "hard_coded_local_payload_digests") &&
      passed;
  const LayoutOffsets layout = Locate(encoded);
  const std::vector<std::uint8_t> expected_payload = HexBytes(
      "000000000000f03f00000000000000c00000000000000c40"
      "000000000000008000000000000014400000000000001940"
      "0000000000001ec00000000000002040");
  passed = Check(std::equal(expected_payload.begin(), expected_payload.end(),
                           encoded.begin() + layout.payload),
                 "all_FP64_coefficients_are_hard_coded_little_endian") &&
      passed;

  DecodeSentinel output;
  auto buffers = output.Buffers();
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *config, resources,
                         runtime, buffers.data(), buffers.size(),
                         output.metadata) == CheckpointCodecStatus::ok,
                 "golden_artifact_decodes_into_preallocated_buffers") &&
      passed;
  CheckpointDescriptor expected_descriptor{};
  const Slot committed{fixture.interior, fixture.trace};
  Slot candidate{};
  Contract contract(committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  passed = Check(AdvanceOneCommit(contract) &&
                     contract.TryMakeCheckpoint(expected_descriptor),
                 "independent_expected_descriptor_available") &&
      passed;
  expected_descriptor.interior_hash =
      OracleManifestHash(1, 32, expected_interior);
  expected_descriptor.trace_hash = OracleManifestHash(2, 32, expected_trace);
  passed = Check(
               output.interior == fixture.interior &&
                   output.trace == fixture.trace &&
                   SameDescriptorIndependent(output.metadata.descriptor,
                                             expected_descriptor) &&
                   output.metadata.card_id == config->card_id() &&
                   output.metadata.case_id == config->case_id() &&
                   output.metadata.scenario_id == config->scenario_id() &&
                   output.metadata.checkpoint_id == config->checkpoint_id() &&
                   output.metadata.physical_gpu_uuid ==
                       runtime.physical_gpu_uuid &&
                   output.metadata.scientific_config_sha256 ==
                       config->scientific_config_sha256() &&
                   output.metadata.run_config_sha256 ==
                       config->run_config_sha256() &&
                   output.metadata.ownership_hash == runtime.ownership_hash &&
                   output.metadata.mesh_n == 3 &&
                   output.metadata.rank_count == 1 &&
                   output.metadata.aggregate_interior_coefficients == 4 &&
                   output.metadata.aggregate_true_trace_coefficients == 4 &&
                   Hex(output.metadata.artifact_hash) ==
                       "66f62405b9fe6f9de15874ce90826cab"
                       "fa8b5d0b3d7676b5e434dbcec0cae25c",
               "all_identity_descriptor_count_and_buffer_fields_decode") &&
      passed;
  return passed;
}

bool CheckEncoderControls()
{
  const CoefficientFixture fixture;
  auto views = fixture.Views();
  const PersistentResourceDescriptor resources = TestResources();
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation validation = TestValidation();
  const ValidatedApplicationConfig* config = validation.config();
  if (config == nullptr) { return Check(false, "encoder_static_config"); }
  const auto capture = MakeCapture(*config, resources, runtime, fixture);
  if (!capture) { return Check(false, "encoder_live_capture"); }
  bool passed = true;
  const auto expect_failure = [&](const std::array<StagedStateViewV1, 1>& value,
                                  const CheckpointCodecStatus expected,
                                  const char* name) {
    std::vector<std::uint8_t> output{{0xaa, 0xbb}};
    const CheckpointCodecStatus status = EncodeCheckpointV1(
        *capture, value.data(), value.size(), output);
    passed = Check(status == expected &&
                       output == std::vector<std::uint8_t>({0xaa, 0xbb}),
                   name) &&
        passed;
  };
  auto invalid = views;
  invalid[0].interior.component_count = 3;
  expect_failure(invalid, CheckpointCodecStatus::invalid_coefficient_schema,
                 "encoder_component_schema_failure_atomic");
  invalid = views;
  invalid[0].interior.coefficient_count = 5;
  expect_failure(invalid, CheckpointCodecStatus::coefficient_count_mismatch,
                 "encoder_exact_count_failure_atomic");
  invalid = views;
  invalid[0].interior.ordering = static_cast<CoefficientOrderingV1>(255);
  expect_failure(invalid, CheckpointCodecStatus::invalid_coefficient_schema,
                 "encoder_order_failure_atomic");
  invalid = views;
  invalid[0].rank = 1;
  expect_failure(invalid, CheckpointCodecStatus::invalid_rank_manifest,
                 "encoder_rank_order_failure_atomic");
  State nonfinite = fixture.interior;
  nonfinite[0] = std::numeric_limits<double>::infinity();
  invalid = views;
  invalid[0].interior.data = nonfinite.data();
  expect_failure(invalid, CheckpointCodecStatus::noncanonical_fp64,
                 "encoder_nonfinite_failure_atomic");
  return passed;
}

bool CheckDecodeMutationCensusAndAtomicity()
{
  const CoefficientFixture fixture;
  const PersistentResourceDescriptor resources = TestResources();
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation validation = TestValidation();
  const ValidatedApplicationConfig* config = validation.config();
  if (config == nullptr) { return Check(false, "mutation_static_config"); }
  std::vector<std::uint8_t> encoded;
  if (!EncodeFixture(fixture, *config, resources, runtime, encoded))
  {
    return Check(false, "mutation_control_fixture");
  }
  const LayoutOffsets layout = Locate(encoded);
  bool passed = true;
  const auto expect_failure = [&](std::vector<std::uint8_t> bytes,
                                  const CheckpointCodecStatus expected,
                                  const char* name) {
    DecodeSentinel output;
    const DecodeSentinel original = output;
    auto buffers = output.Buffers();
    const CheckpointCodecStatus status = DecodeCheckpointV1(
        {bytes.data(), bytes.size()}, *config, resources, runtime,
        buffers.data(), buffers.size(), output.metadata);
    passed = Check(status == expected && SentinelUnchanged(output, original),
                   name) &&
        passed;
  };
  const auto expect_any_failure = [&](std::vector<std::uint8_t> bytes,
                                      const char* name) {
    DecodeSentinel output;
    const DecodeSentinel original = output;
    auto buffers = output.Buffers();
    const CheckpointCodecStatus status = DecodeCheckpointV1(
        {bytes.data(), bytes.size()}, *config, resources, runtime,
        buffers.data(), buffers.size(), output.metadata);
    passed = Check(status != CheckpointCodecStatus::ok &&
                       SentinelUnchanged(output, original),
                   name) &&
        passed;
  };

  std::vector<std::uint8_t> damaged = encoded;
  damaged[0] ^= 1U;
  expect_failure(damaged, CheckpointCodecStatus::invalid_magic,
                 "magic_mutation_is_failure_atomic");
  damaged = encoded;
  damaged[16] = 2;
  expect_failure(damaged, CheckpointCodecStatus::unsupported_version,
                 "version_mutation_is_failure_atomic");
  damaged = encoded;
  WriteU64(damaged, 20, encoded.size() + 1);
  expect_failure(damaged, CheckpointCodecStatus::truncated,
                 "declared_large_length_is_failure_atomic");
  damaged = encoded;
  WriteU64(damaged, 20, encoded.size() - 1);
  expect_failure(damaged, CheckpointCodecStatus::length_mismatch,
                 "declared_short_length_is_failure_atomic");
  damaged = encoded;
  damaged.back() ^= 1U;
  expect_failure(damaged, CheckpointCodecStatus::checksum_mismatch,
                 "checksum_mutation_is_failure_atomic");
  damaged = encoded;
  damaged.pop_back();
  expect_failure(damaged, CheckpointCodecStatus::truncated,
                 "truncation_is_failure_atomic");
  damaged = encoded;
  damaged.push_back(0);
  expect_failure(damaged, CheckpointCodecStatus::length_mismatch,
                 "trailing_artifact_byte_is_failure_atomic");

  for (const NamedOffset& field : layout.exact_fields)
  {
    damaged = encoded;
    damaged[field.offset] ^= 1U;
    static_cast<void>(Seal(damaged));
    expect_any_failure(damaged, field.name);
  }
  passed = Check(layout.exact_fields.size() == 83,
                 "frozen_exact_field_mutation_census_is_83") &&
      passed;

  damaged = encoded;
  WriteU32(damaged, layout.manifest_rank_count,
           std::numeric_limits<std::uint32_t>::max());
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::rank_tuple_mismatch,
                 "oversized_rank_count_rejected_before_output_write");
  damaged = encoded;
  WriteU64(damaged, layout.interior_coefficient_count,
           std::numeric_limits<std::uint64_t>::max());
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::coefficient_count_mismatch,
                 "oversized_block_count_rejected_before_output_write");
  damaged = encoded;
  WriteU64(damaged, layout.aggregate_interior_count,
           std::numeric_limits<std::uint64_t>::max());
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::coefficient_count_mismatch,
                 "oversized_aggregate_count_rejected_before_output_write");

  damaged = encoded;
  WriteU64(damaged, layout.payload, UINT64_C(0x7ff8000000000000));
  const StateHash nan_local =
      OracleHash(damaged.data() + layout.payload, 32);
  std::copy(nan_local.bytes.begin(), nan_local.bytes.end(),
            damaged.begin() + layout.interior_hash);
  const StateHash nan_aggregate = OracleManifestHash(1, 32, nan_local);
  std::copy(nan_aggregate.bytes.begin(), nan_aggregate.bytes.end(),
            damaged.begin() + layout.descriptor_interior_hash);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::noncanonical_fp64,
                 "nonfinite_FP64_rejected_in_first_pass");

  DecodeSentinel capacity_output;
  const DecodeSentinel original = capacity_output;
  auto short_buffers = capacity_output.Buffers();
  short_buffers[0].interior.capacity = 3;
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *config, resources,
                         runtime, short_buffers.data(), short_buffers.size(),
                         capacity_output.metadata) ==
                         CheckpointCodecStatus::output_buffer_mismatch &&
                     SentinelUnchanged(capacity_output, original),
                 "preallocated_capacity_checked_before_any_output_write") &&
      passed;

  CheckpointRuntimeIdentityV1 wrong_runtime = runtime;
  wrong_runtime.ownership_hash.bytes[0] ^= 1U;
  DecodeSentinel runtime_output;
  const DecodeSentinel runtime_original = runtime_output;
  auto runtime_buffers = runtime_output.Buffers();
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *config, resources,
                         wrong_runtime, runtime_buffers.data(),
                         runtime_buffers.size(), runtime_output.metadata) ==
                         CheckpointCodecStatus::invalid_runtime_identity &&
                     SentinelUnchanged(runtime_output, runtime_original),
                 "ownership_identity_checked_before_any_output_write") &&
      passed;
  return passed;
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckIndependentOracle() && passed;
  passed = CheckLiveCaptureAndRepeatableEncode() && passed;
  passed = CheckGoldenRoundTrip() && passed;
  passed = CheckEncoderControls() && passed;
  passed = CheckDecodeMutationCensusAndAtomicity() && passed;
  if (passed)
  {
    std::cout << "HDG W8A in-memory checkpoint codec checks passed\n";
    return 0;
  }
  return 1;
}

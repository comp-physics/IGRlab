#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
static_assert(!std::is_default_constructible<CheckpointContractEvidenceV1>::value &&
                  !std::is_default_constructible<
                      PublishedCheckpointManifestV1>::value &&
                  !std::is_copy_constructible<
                      PublishedCheckpointManifestV1>::value,
              "checkpoint and publication authority cannot be forged");

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
StateHash OracleCoefficientHash(const std::array<double, Size>& values)
{
  std::vector<std::uint8_t> bytes;
  bytes.reserve(Size * 8);
  for (const double value : values) { AppendU64(bytes, OracleBits(value)); }
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
  resources.layout = StableIdentity::Copy("l2-interior.true-trace-owned.v1");
  resources.reuse = {true, true, true, false, 1, 1, 128, 64};
  return resources;
}

ApplicationValidation TestValidation()
{
  ApplicationOptionsView options;
  options.backend_id = CpuSerialBackendTag::id;
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

bool AdvanceTenCommits(Contract& contract)
{
  for (std::uint64_t step = 1; step <= 10; ++step)
  {
    const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.05});
    if (!begin.started) { return false; }
    AttemptReport report{};
    report.gates = {true, true, true, true, true, true};
    report.telemetry.transfers = {0, 0, 0, 0, 0, true};
    report.telemetry.true_relative_residual = 0.0;
    report.candidate_cursor = step;
    const AttemptOutcome outcome = contract.FinishAttempt(
        begin.attempt, report, IssueAll(contract, begin.attempt));
    if (!outcome.committed) { return false; }
  }
  return true;
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

std::optional<PublishedCheckpointManifestV1> MakePublished(
    const ValidatedApplicationConfig& config,
    const std::array<StagedStateViewV1, 1>& ranks,
    const PersistentResourceDescriptor& resources)
{
  const Slot committed{{{1.0, -2.0, 3.5, -0.0}},
                       {{5.0, 6.25, -7.5, 8.0}}};
  Slot candidate{};
  Contract contract(committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  if (!AdvanceTenCommits(contract)) { return std::nullopt; }
  auto evidence = CaptureCheckpointContractEvidenceV1(contract, config);
  if (!evidence) { return std::nullopt; }
  RankTemporaryPublicationFactsV1 temporary{};
  temporary.participating_ranks = 1;
  temporary.source_rank = 0;
  temporary.temporary_file_set_hash = FilledHash(0x70);
  temporary.previous_manifest_hash = FilledHash(0x90);
  temporary.all_rank_temporaries_created_exclusive = true;
  temporary.all_rank_temporaries_synced = true;
  temporary.collective_temporary_consensus = true;
  auto durable = RecordDurableRankTemporariesV1(
      std::move(*evidence), temporary, ranks.data(), ranks.size());
  if (durable.status != CheckpointCodecStatus::ok || !durable.authority)
  {
    return std::nullopt;
  }
  CollectiveManifestValidationFactsV1 collective{1, true, true, true};
  auto validated = ValidateCollectiveCheckpointManifestV1(
      std::move(*durable.authority), collective);
  if (validated.status != CheckpointCodecStatus::ok || !validated.authority)
  {
    return std::nullopt;
  }
  AtomicManifestPublicationFactsV1 atomic{0, true, true, true, true, true};
  auto published =
      PublishCheckpointManifestV1(std::move(*validated.authority), atomic);
  if (published.status != CheckpointCodecStatus::ok || !published.authority)
  {
    return std::nullopt;
  }
  return std::move(*published.authority);
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

std::size_t SkipIdentity(const std::vector<std::uint8_t>& bytes,
                         std::size_t offset)
{
  const std::uint16_t size = static_cast<std::uint16_t>(bytes[offset]) |
      static_cast<std::uint16_t>(bytes[offset + 1] << 8U);
  return offset + 2 + size;
}

struct LayoutOffsets
{
  std::size_t descriptor_start = 0;
  std::size_t descriptor_interior_hash = 0;
  std::size_t descriptor_trace_hash = 0;
  std::size_t manifest_rank_count = 0;
  std::size_t rank_record = 0;
  std::size_t interior_coefficient_count = 0;
  std::size_t interior_hash = 0;
  std::size_t trace_hash = 0;
  std::size_t payload = 0;
};

LayoutOffsets Locate(const std::vector<std::uint8_t>& bytes)
{
  std::size_t offset = 120;
  offset = SkipIdentity(bytes, offset);
  offset = SkipIdentity(bytes, offset);
  offset = SkipIdentity(bytes, offset);
  LayoutOffsets layout{};
  layout.descriptor_start = offset;
  offset += 6;
  for (int identity = 0; identity < 8; ++identity)
  {
    offset = SkipIdentity(bytes, offset);
  }
  offset += 3 * 32 + 4 + 4;
  offset = SkipIdentity(bytes, offset);
  offset = SkipIdentity(bytes, offset);
  offset += 8 + 8 + 5 * 8 + 8;
  layout.descriptor_interior_hash = offset;
  layout.descriptor_trace_hash = offset + 32;
  offset += 64;
  layout.manifest_rank_count = offset;
  offset += 4 + 8 + 8;
  layout.rank_record = offset;
  layout.interior_coefficient_count = offset + 4 + 4;
  layout.interior_hash = offset + 4 + 4 + 8 + 8 + 1 + 8;
  layout.trace_hash = offset + 4 + 61 + 4 + 8 + 8 + 1 + 8;
  layout.payload = offset + 126;
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

DecodedCheckpointV1 SentinelOutput()
{
  DecodedCheckpointV1 sentinel{};
  sentinel.card_id = StableIdentity::Copy("sentinel-card");
  sentinel.aggregate_interior_coefficients = 999;
  sentinel.ranks.resize(1);
  sentinel.ranks[0].rank = 77;
  sentinel.ranks[0].interior.coefficients = {123.0};
  return sentinel;
}

bool SentinelUnchanged(const DecodedCheckpointV1& value)
{
  return value.card_id.view() == "sentinel-card" &&
      value.aggregate_interior_coefficients == 999 &&
      value.ranks.size() == 1 && value.ranks[0].rank == 77 &&
      value.ranks[0].interior.coefficients == std::vector<double>{123.0};
}

// Filled after independent format derivation; this literal is the entire v1
// artifact, including its final SHA-256 checksum.
constexpr std::string_view golden_artifact_hex =
    "44474947522d4844472d434b5054310001000000890300000000000002480f00ba88f27202707172"
    "737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f909192939495969798999a"
    "9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeaf01000000000000000101010101010101010101"
    "33006578702e6864672e65756c65722e636f6e76656e74696f6e616c2e736d6f6f74685f70657273"
    "697374656e742e7738612e76312100636f6e76656e74696f6e616c2d736d6f6f74682d7065727369"
    "7374656e742d76312f00636865636b706f696e742e6864672e65756c65722e736d6f6f74682d7065"
    "72696f6469632e737465702d31302e7631010000000100150065756c65722e636f6d707265737369"
    "626c652e326410006864672e636f6e76656e74696f6e616c17006d66656d2e64675f696e74657266"
    "6163652e747261636509007461752e66697865640e006261636b776172645f65756c65720a006370"
    "752e73657269616c0400667036341000643964363532366363313734393938301011121314151617"
    "18191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
    "404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f6061626364656667"
    "68696a6b6c6d6e6f01000000010000001f006c322d696e746572696f722e747275652d7472616365"
    "2d6f776e65642e76310400667036340a000000000000009a9999999999a93fffffffffffffdf3f0a"
    "000000000000000a000000000000000a000000000000009a9999999999a93f480f00ba88f2720266"
    "d70231bb8654d88e509ef55a64a36059cd1bbf1f33b6197de3a4acacc15098bfb3882f02ddf06341"
    "71bd039abb25ef0c9e9d9c287ca402234a394184a57d430100000004000000000000000400000000"
    "00000000000000040000000400000000000000010000000000000000200000000000000063b84f61"
    "7d28be16889d09da2bf88a144d908cdc462a176953c8c67191d68359040000000400000000000000"
    "0100000000000000002000000000000000b0c0959fdf6d33e5435ebb3bb41ea582d274b360e1597f"
    "dd886ac1ebeb361f36000000000000f03f00000000000000c00000000000000c4000000000000000"
    "80000000000000144000000000000019400000000000001ec0000000000000204021007369ba0487"
    "fcf180ef5aa01463e0de33e7a13bee4e7fcf98266ee35adad8";

bool CheckIndependentOracle()
{
  const std::array<std::uint8_t, 3> abc{{'a', 'b', 'c'}};
  return Check(
      Hex(OracleHash(abc.data(), abc.size())) ==
          "ba7816bf8f01cfea414140de5dae2223"
          "b00361a396177a9cb410ff61f20015ad",
      "independent_sha256_matches_hard_coded_FIPS_vector");
}

bool CheckPublicationAuthority()
{
  const CoefficientFixture fixture;
  const auto ranks = fixture.Views();
  const PersistentResourceDescriptor resources = TestResources();
  const ApplicationValidation validation = TestValidation();
  bool passed = Check(validation.accepted(), "publication_config_valid");

  const Slot committed{fixture.interior, fixture.trace};
  Slot candidate{};
  Contract uncommitted(
      committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  passed = Check(!CaptureCheckpointContractEvidenceV1(
                         uncommitted, validation.config()),
                 "uncommitted_contract_cannot_issue_checkpoint_evidence") &&
      passed;

  Contract contract(committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  passed = Check(AdvanceTenCommits(contract),
                 "contract_reaches_successful_step10_commit") &&
      passed;
  auto evidence =
      CaptureCheckpointContractEvidenceV1(contract, validation.config());
  passed = Check(evidence.has_value(),
                 "successful_live_contract_issues_affine_evidence") &&
      passed;
  if (!evidence) { return false; }

  RankTemporaryPublicationFactsV1 temporary{};
  temporary.participating_ranks = 1;
  temporary.source_rank = 0;
  temporary.temporary_file_set_hash = FilledHash(0x70);
  temporary.previous_manifest_hash = FilledHash(0x90);
  auto rejected = RecordDurableRankTemporariesV1(
      std::move(*evidence), temporary, ranks.data(), ranks.size());
  passed = Check(rejected.status ==
                         CheckpointCodecStatus::invalid_publication_state &&
                     !rejected.authority,
                 "temporary_files_must_be_exclusive_durable_collective") &&
      passed;
  temporary.all_rank_temporaries_created_exclusive = true;
  temporary.all_rank_temporaries_synced = true;
  temporary.collective_temporary_consensus = true;
  auto durable = RecordDurableRankTemporariesV1(
      std::move(*evidence), temporary, ranks.data(), ranks.size());
  passed = Check(durable.status == CheckpointCodecStatus::ok &&
                     durable.authority.has_value(),
                 "durable_rank_temporaries_recorded") &&
      passed;
  if (!durable.authority) { return false; }

  CollectiveManifestValidationFactsV1 collective{1, true, true, false};
  auto invalid_collective = ValidateCollectiveCheckpointManifestV1(
      std::move(*durable.authority), collective);
  passed = Check(invalid_collective.status ==
                         CheckpointCodecStatus::invalid_publication_state &&
                     !invalid_collective.authority,
                 "manifest_requires_payload_hash_consensus") &&
      passed;
  collective.payload_hash_consensus = true;
  auto validated = ValidateCollectiveCheckpointManifestV1(
      std::move(*durable.authority), collective);
  passed = Check(validated.status == CheckpointCodecStatus::ok &&
                     validated.authority.has_value(),
                 "collective_manifest_validation_recorded") &&
      passed;
  if (!validated.authority) { return false; }

  AtomicManifestPublicationFactsV1 atomic{0, true, true, false, true, true};
  auto invalid_publish = PublishCheckpointManifestV1(
      std::move(*validated.authority), atomic);
  passed = Check(invalid_publish.status ==
                         CheckpointCodecStatus::invalid_publication_state &&
                     !invalid_publish.authority,
                 "manifest_requires_atomic_rename") &&
      passed;
  atomic.atomic_rename_complete = true;
  auto published = PublishCheckpointManifestV1(
      std::move(*validated.authority), atomic);
  passed = Check(published.status == CheckpointCodecStatus::ok &&
                     published.authority.has_value(),
                 "source_rank_zero_atomic_publication_recorded") &&
      passed;
  return passed;
}

bool EncodeFixture(const CoefficientFixture& fixture,
                   const ValidatedApplicationConfig& config,
                   const PersistentResourceDescriptor& resources,
                   std::vector<std::uint8_t>& encoded)
{
  const auto ranks = fixture.Views();
  auto published = MakePublished(config, ranks, resources);
  return published &&
      EncodeCheckpointV1(*published, ranks.data(), ranks.size(), encoded) ==
      CheckpointCodecStatus::ok;
}

bool CheckGoldenRoundTrip()
{
  const CoefficientFixture fixture;
  const auto ranks = fixture.Views();
  const PersistentResourceDescriptor resources = TestResources();
  const ApplicationValidation validation = TestValidation();
  std::vector<std::uint8_t> encoded;
  bool passed = Check(validation.accepted() &&
                          EncodeFixture(fixture, validation.config(),
                                        resources, encoded),
                      "golden_fixture_encodes");
  const std::vector<std::uint8_t> golden = HexBytes(golden_artifact_hex);
  passed = Check(!golden.empty() && encoded == golden,
                 "entire_artifact_matches_hard_coded_golden_stream") &&
      passed;
  passed = Check(encoded.size() > 64 && encoded[0] == 'D' &&
                     encoded[16] == 1 && ReadU64(encoded, 20) == encoded.size() &&
                     encoded[28] == static_cast<std::uint8_t>(
                         PublicationPhaseV1::manifest_atomically_published),
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

  DecodedCheckpointV1 decoded;
  passed = Check(DecodeCheckpointV1({encoded.data(), encoded.size()},
                                    validation.config(), resources,
                                    decoded) == CheckpointCodecStatus::ok,
                 "golden_artifact_decodes") &&
      passed;
  CheckpointDescriptor contract_descriptor{};
  const Slot committed{fixture.interior, fixture.trace};
  Slot candidate{};
  Contract contract(committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  passed = Check(AdvanceTenCommits(contract) &&
                     contract.TryMakeCheckpoint(contract_descriptor),
                 "independent_expected_descriptor_available") &&
      passed;
  contract_descriptor.interior_hash =
      OracleManifestHash(1, 32, expected_interior);
  contract_descriptor.trace_hash = OracleManifestHash(2, 32, expected_trace);
  passed = Check(
               SameDescriptorIndependent(decoded.descriptor,
                                         contract_descriptor) &&
                   decoded.card_id.view() ==
                       smooth_periodic_euler_v1_card_id &&
                   decoded.case_id.view() ==
                       smooth_periodic_euler_v1_case_id &&
                   decoded.checkpoint_id.view() ==
                       smooth_periodic_euler_v1_checkpoint_id &&
                   decoded.publication.phase ==
                       PublicationPhaseV1::manifest_atomically_published &&
                   decoded.publication.source_rank == 0 &&
                   decoded.publication.participating_ranks == 1 &&
                   decoded.aggregate_interior_coefficients == 4 &&
                   decoded.aggregate_true_trace_coefficients == 4 &&
                   Hex(decoded.artifact_hash) ==
                       "21007369ba0487fcf180ef5aa01463e0"
                       "de33e7a13bee4e7fcf98266ee35adad8" &&
                   decoded.ranks.size() == 1 &&
                   decoded.ranks[0].interior.component_count == 4 &&
                   decoded.ranks[0].interior.dofs_per_component == 1 &&
                   decoded.ranks[0].interior.coefficients ==
                       std::vector<double>(fixture.interior.begin(),
                                           fixture.interior.end()) &&
                   decoded.ranks[0].owned_true_trace.coefficients ==
                       std::vector<double>(fixture.trace.begin(),
                                           fixture.trace.end()),
               "every_descriptor_publication_schema_and_payload_field_decodes") &&
      passed;
  return passed;
}

bool CheckEncoderMutationControls()
{
  const CoefficientFixture fixture;
  auto ranks = fixture.Views();
  const PersistentResourceDescriptor resources = TestResources();
  const ApplicationValidation validation = TestValidation();
  bool passed = Check(validation.accepted(), "encoder_control_config");
  auto published = MakePublished(validation.config(), ranks, resources);
  passed = Check(published.has_value(), "encoder_control_publication") && passed;
  if (!published) { return false; }

  const auto expect_failure = [&](const std::array<StagedStateViewV1, 1>& value,
                                  const CheckpointCodecStatus expected,
                                  const char* name) {
    std::vector<std::uint8_t> output{{0xaa, 0xbb}};
    const CheckpointCodecStatus status = EncodeCheckpointV1(
        *published, value.data(), value.size(), output);
    passed = Check(status == expected &&
                       output == std::vector<std::uint8_t>({0xaa, 0xbb}),
                   name) &&
        passed;
  };
  auto invalid = ranks;
  invalid[0].interior.component_count = 3;
  expect_failure(invalid, CheckpointCodecStatus::invalid_coefficient_schema,
                 "component_count_rejected_atomically");
  invalid = ranks;
  invalid[0].interior.coefficient_count = 5;
  expect_failure(invalid, CheckpointCodecStatus::coefficient_count_mismatch,
                 "coefficient_count_rejected_atomically");
  invalid = ranks;
  invalid[0].interior.ordering = static_cast<CoefficientOrderingV1>(255);
  expect_failure(invalid, CheckpointCodecStatus::invalid_coefficient_schema,
                 "coefficient_order_rejected_atomically");
  invalid = ranks;
  invalid[0].rank = 1;
  expect_failure(invalid, CheckpointCodecStatus::invalid_rank_manifest,
                 "rank_order_rejected_atomically");
  State nonfinite = fixture.interior;
  nonfinite[0] = std::numeric_limits<double>::infinity();
  invalid = ranks;
  invalid[0].interior.data = nonfinite.data();
  expect_failure(invalid, CheckpointCodecStatus::noncanonical_fp64,
                 "nonfinite_FP64_rejected_atomically");
  return passed;
}

bool CheckDecodeFailureAtomicityAndBounds()
{
  const CoefficientFixture fixture;
  const PersistentResourceDescriptor resources = TestResources();
  const ApplicationValidation validation = TestValidation();
  std::vector<std::uint8_t> encoded;
  if (!validation.accepted() ||
      !EncodeFixture(fixture, validation.config(), resources, encoded))
  {
    return Check(false, "decode_control_fixture");
  }
  const LayoutOffsets layout = Locate(encoded);
  bool passed = true;
  const auto expect_failure = [&](std::vector<std::uint8_t> bytes,
                                  const CheckpointCodecStatus expected,
                                  const char* name) {
    DecodedCheckpointV1 output = SentinelOutput();
    const CheckpointCodecStatus status = DecodeCheckpointV1(
        {bytes.data(), bytes.size()}, validation.config(), resources, output);
    passed = Check(status == expected && SentinelUnchanged(output), name) &&
        passed;
  };

  std::vector<std::uint8_t> damaged = encoded;
  damaged[layout.payload] ^= 1U;
  expect_failure(damaged, CheckpointCodecStatus::checksum_mismatch,
                 "checksum_corruption_is_failure_atomic");
  damaged = encoded;
  damaged.pop_back();
  expect_failure(damaged, CheckpointCodecStatus::truncated,
                 "truncation_is_failure_atomic");
  damaged = encoded;
  damaged.push_back(0);
  expect_failure(damaged, CheckpointCodecStatus::length_mismatch,
                 "trailing_bytes_are_failure_atomic");
  damaged = encoded;
  WriteU32(damaged, layout.manifest_rank_count,
           std::numeric_limits<std::uint32_t>::max());
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::rank_tuple_mismatch,
                 "huge_rank_count_rejected_before_allocation");
  damaged = encoded;
  WriteU64(damaged, layout.interior_coefficient_count,
           std::numeric_limits<std::uint64_t>::max());
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::coefficient_count_mismatch,
                 "huge_coefficient_count_rejected_before_allocation");
  damaged = encoded;
  WriteU64(damaged, layout.manifest_rank_count + 4,
           std::numeric_limits<std::uint64_t>::max());
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::coefficient_count_mismatch,
                 "huge_aggregate_count_rejected_before_allocation");
  damaged = encoded;
  damaged[layout.payload] ^= 1U;
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::local_payload_hash_mismatch,
                 "payload_hash_mismatch_is_failure_atomic");

  damaged = encoded;
  WriteU64(damaged, layout.payload, 0x7ff8000000000000ULL);
  const StateHash nan_local =
      OracleHash(damaged.data() + layout.payload, 32);
  std::copy(nan_local.bytes.begin(), nan_local.bytes.end(),
            damaged.begin() + layout.interior_hash);
  const StateHash nan_aggregate = OracleManifestHash(1, 32, nan_local);
  std::copy(nan_aggregate.bytes.begin(), nan_aggregate.bytes.end(),
            damaged.begin() + layout.descriptor_interior_hash);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::noncanonical_fp64,
                 "noncanonical_FP64_decode_is_failure_atomic");

  for (std::size_t flag = 0; flag < 11; ++flag)
  {
    damaged = encoded;
    damaged[109 + flag] = 0;
    static_cast<void>(Seal(damaged));
    expect_failure(damaged, CheckpointCodecStatus::unpublished_artifact,
                   "every_publication_protocol_flag_is_required");
  }
  damaged = encoded;
  damaged[28] = 0;
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::unpublished_artifact,
                 "published_phase_is_required");
  damaged = encoded;
  WriteU64(damaged, 29, 0);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::unpublished_artifact,
                 "transaction_identity_is_required");
  damaged = encoded;
  std::fill_n(damaged.begin() + 37, 32, 0);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::unpublished_artifact,
                 "temporary_file_set_identity_is_required");
  damaged = encoded;
  WriteU32(damaged, 101, 2);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::rank_tuple_mismatch,
                 "publication_rank_count_must_match_descriptor");
  damaged = encoded;
  WriteU32(damaged, 105, 1);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::unpublished_artifact,
                 "source_rank_zero_is_required");

  damaged = encoded;
  damaged[122] ^= 1U;
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::invalid_checkpoint_identity,
                 "card_identity_corruption_is_failure_atomic");
  damaged = encoded;
  WriteU32(damaged, layout.descriptor_start, 2);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::invalid_descriptor,
                 "descriptor_schema_corruption_is_failure_atomic");
  damaged = encoded;
  damaged[layout.descriptor_interior_hash - 1] ^= 1U;
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::invalid_descriptor,
                 "coordinate_fingerprint_corruption_is_failure_atomic");

  PersistentResourceDescriptor wrong_resources = resources;
  wrong_resources.partition_hash.bytes[0] ^= 1U;
  DecodedCheckpointV1 output = SentinelOutput();
  passed = Check(DecodeCheckpointV1({encoded.data(), encoded.size()},
                                    validation.config(), wrong_resources,
                                    output) ==
                         CheckpointCodecStatus::resource_mismatch &&
                     SentinelUnchanged(output),
                 "same_tuple_resource_mismatch_precedes_allocation") &&
      passed;
  return passed;
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckIndependentOracle() && passed;
  passed = CheckPublicationAuthority() && passed;
  passed = CheckGoldenRoundTrip() && passed;
  passed = CheckEncoderMutationControls() && passed;
  passed = CheckDecodeFailureAtomicityAndBounds() && passed;
  if (passed)
  {
    std::cout << "HDG W8A checkpoint codec checks passed\n";
    return 0;
  }
  return 1;
}

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
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
static_assert(application_checkpoint_codec_version_v2 == 2 &&
                  application_checkpoint_max_rank_count_v1 == 2 &&
                  euler_state_component_count_v1 == 4 &&
                  fp64_coefficient_bytes_v1 == 8,
              "v2 version and rank, component, binary64 bounds are exact");
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

StateHash OracleCoefficientHash(
    const std::vector<double>& values) noexcept
{
  OracleSha256 hash;
  std::array<std::uint8_t, 8> bytes{};
  for (const double value : values)
  {
    const std::uint64_t bits = OracleBits(value);
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
      bytes[index] = static_cast<std::uint8_t>(bits >> (8U * index));
    }
    hash.Update(bytes.data(), bytes.size());
  }
  return hash.Finalize();
}

struct OracleRankTuple
{
  std::uint32_t rank = 0;
  std::uint64_t byte_count = 0;
  StateHash local_hash{};
};

StateHash OracleManifestHash(const std::uint8_t kind,
                             const OracleRankTuple* ranks,
                             const std::size_t rank_count)
{
  const std::array<std::uint8_t, 25> domain{{
      'D', 'G', 'I', 'G', 'R', '-', 'H', 'D', 'G', '-', 'R', 'A', 'N',
      'K', '-', 'P', 'A', 'Y', 'L', 'O', 'A', 'D', '-', 'V', '1'}};
  std::vector<std::uint8_t> bytes(domain.begin(), domain.end());
  bytes.push_back(kind);
  AppendU32(bytes, static_cast<std::uint32_t>(rank_count));
  for (std::size_t index = 0; index < rank_count; ++index)
  {
    AppendU32(bytes, ranks[index].rank);
    AppendU64(bytes, ranks[index].byte_count);
    bytes.insert(bytes.end(), ranks[index].local_hash.bytes.begin(),
                 ranks[index].local_hash.bytes.end());
  }
  return OracleHash(bytes.data(), bytes.size());
}

StateHash OracleManifestHash(const std::uint8_t kind,
                             const std::uint64_t byte_count,
                             const StateHash& local_hash)
{
  const OracleRankTuple rank{0, byte_count, local_hash};
  return OracleManifestHash(kind, &rank, 1);
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

using State = std::vector<double>;
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

using SerialContract = PersistentRunContract<CpuSerialRuntimeSelection,
                                             State,
                                             State,
                                             TestStoragePolicy,
                                             TestCollectivePolicy>;
using Rank2Contract = PersistentRunContract<CpuRank2RuntimeSelection,
                                            State,
                                            State,
                                            TestStoragePolicy,
                                            TestCollectivePolicy>;
using SerialContractView = RankCommittedContractViewV1<
    CpuSerialRuntimeSelection,
    State,
    State,
    TestStoragePolicy,
    TestCollectivePolicy>;
using Rank2ContractView = RankCommittedContractViewV1<
    CpuRank2RuntimeSelection,
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

PersistentResourceDescriptor TestResources(
    const std::uint32_t rank_count,
    const int polynomial_order = 1)
{
  PersistentResourceDescriptor resources{};
  resources.mfem_pin = StableIdentity::Copy("d9d6526cc1749980");
  resources.mesh_hash = FilledHash(0x10);
  resources.partition_hash = FilledHash(0x30);
  resources.topology_hash = FilledHash(0x50);
  resources.rank_count = rank_count;
  resources.polynomial_order = polynomial_order;
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

ApplicationOptionsView TestOptions(const std::string_view backend,
                                   const bool persistent = true)
{
  ApplicationOptionsView options;
  options.backend_id = backend;
  if (!persistent)
  {
    options.evidence_directory = {};
    options.validation_capture = ValidationCaptureModeV1::none;
  }
  return options;
}

ApplicationOptionsView FunctionalTestOptions(
    const std::string_view backend,
    const bool persistent = true)
{
  ApplicationOptionsView options = TestOptions(backend, persistent);
  options.scenario_id = smooth_periodic_euler_v1_functional_id;
  options.polynomial_order =
      smooth_periodic_euler_v1_functional_polynomial_order;
  options.mesh_n = smooth_periodic_euler_v1_functional_mesh_n;
  options.q_a = smooth_periodic_euler_v1_functional_q_a;
  options.q_e = smooth_periodic_euler_v1_functional_q_e;
  options.q_e4 = smooth_periodic_euler_v1_functional_q_e4;
  options.q_o = smooth_periodic_euler_v1_functional_q_o;
  options.accepted_steps = smooth_periodic_euler_v1_functional_steps;
  options.final_time = smooth_periodic_euler_v1_functional_final_time;
  options.output_steps = smooth_periodic_euler_v1_functional_output_steps;
  options.checkpoint_id =
      smooth_periodic_euler_v1_functional_checkpoint_id;
  options.checkpoint_step = OptionalCheckpointStepV1::Present(
      smooth_periodic_euler_v1_functional_checkpoint_step);
  options.seed = smooth_periodic_euler_v1_functional_seed;
  return options;
}

ApplicationValidation TestValidation(const std::string_view backend,
                                     const bool persistent = true)
{
  ApplicationOptionsView options = TestOptions(backend, persistent);
  options.run_role = ApplicationRunRoleV1::split_writer;
  options.stop_after_checkpoint = StopAfterCheckpointV1::yes;
  return ValidateApplicationOptions(options);
}

ApplicationValidation TestReaderValidation(
    const std::string_view backend,
    const bool persistent = true)
{
  ApplicationOptionsView options = TestOptions(backend, persistent);
  const std::string_view artifact = options.checkpoint_out.canonical;
  options.run_role = ApplicationRunRoleV1::split_reader;
  options.checkpoint_out = {};
  options.checkpoint_step = {};
  options.restart_from = CheckpointArtifactPathV1::Present(artifact);
  options.injection = {
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      smooth_periodic_euler_v1_smoke_failure_target_step,
      backend == CpuRank2BackendTag::id ? 1U : 0U};
  options.retry_limit = smooth_periodic_euler_v1_retry_limit;
  return ValidateApplicationOptions(options);
}

ApplicationValidation FunctionalTestValidation(
    const bool reader)
{
  ApplicationOptionsView options =
      FunctionalTestOptions(CpuSerialBackendTag::id);
  const std::string_view artifact = options.checkpoint_out.canonical;
  if (reader)
  {
    options.run_role = ApplicationRunRoleV1::split_reader;
    options.checkpoint_out = {};
    options.checkpoint_step = {};
    options.restart_from = CheckpointArtifactPathV1::Present(artifact);
    options.injection = {
        FailureInjectionModeV1::prescribed,
        smooth_periodic_euler_v1_failure_phase,
        smooth_periodic_euler_v1_functional_failure_target_step,
        0};
    options.retry_limit = smooth_periodic_euler_v1_retry_limit;
  }
  else
  {
    options.run_role = ApplicationRunRoleV1::split_writer;
    options.stop_after_checkpoint = StopAfterCheckpointV1::yes;
  }
  return ValidateApplicationOptions(options);
}

template <typename ContractType>
typename ContractType::ProductBundle IssueAll(
    ContractType& contract,
    const AttemptIdentity& attempt)
{
  typename ContractType::ProductBundle products{};
  products.residual =
      contract.template IssueProduct<DerivedProductKind::residual>(attempt);
  products.matrix =
      contract.template IssueProduct<DerivedProductKind::matrix>(attempt);
  products.factor =
      contract.template IssueProduct<DerivedProductKind::factor>(attempt);
  products.solution =
      contract.template IssueProduct<DerivedProductKind::solution>(attempt);
  return products;
}

template <typename ContractType>
bool AdvanceOneCommit(ContractType& contract)
{
  const BeginOutcome begin = contract.BeginStep({InteriorSpace::l2, 0.05});
  if (!begin.started) { return false; }
  AttemptReport report{};
  report.gates = {true, true, true, true, true, true};
  report.telemetry.transfers = {0, 0, 0, 0, 0, true};
  report.telemetry.true_relative_residual = 0.0;
  report.candidate_cursor = begin.attempt.sequence;
  return contract.FinishAttempt(begin.attempt, report,
                                IssueAll(contract, begin.attempt)).committed;
}

template <typename ContractType>
bool AdvanceCommits(ContractType& contract, const std::uint64_t count)
{
  for (std::uint64_t commit = 0; commit < count; ++commit)
  {
    if (!AdvanceOneCommit(contract)) { return false; }
  }
  return true;
}

struct CoefficientFixture
{
  State interior{};
  State trace{};

  CoefficientFixture() : interior(144), trace(144)
  {
    for (std::size_t index = 0; index < interior.size(); ++index)
    {
      interior[index] = static_cast<double>(index + 1);
      trace[index] = static_cast<double>(1001 + index);
    }
  }

  std::array<StagedStateViewV1, 1> Views() const noexcept
  {
    return {{{0,
              {interior.data(), interior.size(), 4, 36,
               CoefficientOrderingV1::component_major},
              {trace.data(), trace.size(), 4, 36,
               CoefficientOrderingV1::component_major}}}};
  }
};

struct Rank2CoefficientFixture
{
  std::array<State, 2> interior{{State(80), State(64)}};
  std::array<State, 2> trace{{State(112), State(32)}};

  Rank2CoefficientFixture()
  {
    for (std::size_t index = 0; index < interior[0].size(); ++index)
    {
      interior[0][index] = static_cast<double>(index + 1);
    }
    for (std::size_t index = 0; index < trace[0].size(); ++index)
    {
      trace[0][index] = static_cast<double>(1001 + index);
    }
    for (std::size_t index = 0; index < interior[1].size(); ++index)
    {
      interior[1][index] = static_cast<double>(2001 + index);
    }
    for (std::size_t index = 0; index < trace[1].size(); ++index)
    {
      trace[1][index] = static_cast<double>(3001 + index);
    }
  }

  std::array<StagedStateViewV1, 2> Views() const noexcept
  {
    return {{{0,
              {interior[0].data(), interior[0].size(), 4, 20,
               CoefficientOrderingV1::component_major},
              {trace[0].data(), trace[0].size(), 4, 28,
               CoefficientOrderingV1::component_major}},
             {1,
              {interior[1].data(), interior[1].size(), 4, 16,
               CoefficientOrderingV1::component_major},
              {trace[1].data(), trace[1].size(), 4, 8,
               CoefficientOrderingV1::component_major}}}};
  }
};

struct FunctionalCoefficientFixture
{
  State interior = State(1296);
  State trace = State(864);

  FunctionalCoefficientFixture()
  {
    for (std::size_t index = 0; index < interior.size(); ++index)
    {
      interior[index] = static_cast<double>(4001 + index);
    }
    for (std::size_t index = 0; index < trace.size(); ++index)
    {
      trace[index] = static_cast<double>(8001 + index);
    }
  }

  std::array<StagedStateViewV1, 1> Views() const noexcept
  {
    return {{{0,
              {interior.data(), interior.size(), 4, 324,
               CoefficientOrderingV1::component_major},
              {trace.data(), trace.size(), 4, 216,
               CoefficientOrderingV1::component_major}}}};
  }
};

template <typename Fixture>
std::optional<CapturedCheckpointV1> MakeSerialCapture(
    const ValidatedApplicationConfig& config,
    const PersistentResourceDescriptor& resources,
    const CheckpointRuntimeIdentityV1& runtime,
    const Fixture& fixture,
    const std::uint64_t commits)
{
  const Slot committed{fixture.interior, fixture.trace};
  Slot candidate{};
  SerialContract contract(
      committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  if (!AdvanceCommits(contract, commits)) { return std::nullopt; }
  const std::array<SerialContractView, 1> contracts{{{0, &contract}}};
  return CaptureCheckpointV1(contracts.data(), contracts.size(), config,
                             runtime);
}

std::optional<CapturedCheckpointV1> MakeCapture(
    const ValidatedApplicationConfig& config,
    const PersistentResourceDescriptor& resources,
    const CheckpointRuntimeIdentityV1& runtime,
    const CoefficientFixture& fixture)
{
  return MakeSerialCapture(config, resources, runtime, fixture, 1);
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

void AppendU16(std::vector<std::uint8_t>& bytes, const std::uint16_t value)
{
  bytes.push_back(static_cast<std::uint8_t>(value));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void AppendHash(std::vector<std::uint8_t>& bytes, const StateHash& hash)
{
  bytes.insert(bytes.end(), hash.bytes.begin(), hash.bytes.end());
}

void AppendIdentity(std::vector<std::uint8_t>& bytes,
                    const StableIdentity& identity)
{
  AppendU16(bytes, static_cast<std::uint16_t>(identity.size()));
  const auto* first = reinterpret_cast<const std::uint8_t*>(identity.data());
  bytes.insert(bytes.end(), first, first + identity.size());
}

template <typename Path>
void AppendPath(std::vector<std::uint8_t>& bytes, const Path& path)
{
  bytes.push_back(path.present() ? 1U : 0U);
  if (path.present()) { AppendIdentity(bytes, path.identity()); }
}

void AppendOptionalStep(std::vector<std::uint8_t>& bytes,
                        const OptionalCheckpointStepV1 step)
{
  bytes.push_back(step.present ? 1U : 0U);
  AppendU64(bytes, step.value);
}

void AppendInjection(std::vector<std::uint8_t>& bytes,
                     const FailureInjectionControlV1 injection)
{
  bytes.push_back(static_cast<std::uint8_t>(injection.mode));
  bytes.push_back(static_cast<std::uint8_t>(injection.phase));
  AppendU64(bytes, injection.target_step);
  AppendU32(bytes, injection.source_rank);
}

void AppendDescriptor(std::vector<std::uint8_t>& bytes,
                      const CheckpointDescriptor& descriptor)
{
  AppendU32(bytes, descriptor.schema_version);
  bytes.push_back(descriptor.successful_commit ? 1U : 0U);
  bytes.push_back(static_cast<std::uint8_t>(descriptor.restart_mode));
  for (const StableIdentity* identity :
       {&descriptor.tuple.model, &descriptor.tuple.formulation,
        &descriptor.tuple.trace, &descriptor.tuple.stabilization,
        &descriptor.tuple.time_treatment, &descriptor.tuple.backend,
        &descriptor.tuple.precision, &descriptor.mfem_pin})
  {
    AppendIdentity(bytes, *identity);
  }
  AppendHash(bytes, descriptor.mesh_hash);
  AppendHash(bytes, descriptor.partition_hash);
  AppendHash(bytes, descriptor.topology_hash);
  AppendU32(bytes, descriptor.rank_count);
  AppendU32(bytes, static_cast<std::uint32_t>(descriptor.polynomial_order));
  AppendIdentity(bytes, descriptor.layout);
  AppendIdentity(bytes, descriptor.precision);
  AppendU64(bytes, descriptor.cursor);
  AppendU64(bytes, OracleBits(descriptor.dt));
  AppendU64(bytes, OracleBits(descriptor.coordinates.time));
  AppendU64(bytes, descriptor.coordinates.step);
  AppendU64(bytes, descriptor.coordinates.generation);
  AppendU64(bytes, descriptor.coordinates.cursor);
  AppendU64(bytes, OracleBits(descriptor.coordinates.dt));
  AppendU64(bytes, descriptor.coordinate_fingerprint);
  AppendHash(bytes, descriptor.interior_hash);
  AppendHash(bytes, descriptor.trace_hash);
}

std::vector<std::uint8_t> IndependentGoldenV2(
    const ValidatedApplicationConfig& config,
    const CheckpointRuntimeIdentityV1& runtime,
    const CheckpointDescriptor& descriptor,
    const CoefficientFixture& fixture)
{
  const std::array<std::uint8_t, 16> magic{{
      'D', 'G', 'I', 'G', 'R', '-', 'H', 'D', 'G', '-', 'M', 'E', 'M', 'V',
      '2', 0}};
  std::vector<std::uint8_t> bytes(magic.begin(), magic.end());
  AppendU32(bytes, application_checkpoint_codec_version_v2);
  AppendU64(bytes, 0);
  for (const StableIdentity* identity :
       {&config.card_id(), &config.case_id(), &config.scenario_id(),
        &config.checkpoint_id(), &config.mesh_family_id(), &config.source_id(),
        &config.flux_id(), &config.time_rule_id(), &config.layout_id()})
  {
    AppendIdentity(bytes, *identity);
  }
  AppendHash(bytes, config.scientific_config_sha256());
  AppendHash(bytes, config.run_config_sha256());
  bytes.push_back(static_cast<std::uint8_t>(config.scenario()));
  bytes.push_back(static_cast<std::uint8_t>(config.backend()));
  bytes.push_back(static_cast<std::uint8_t>(config.interior_space()));
  AppendU32(bytes, config.deferred_runtime().required_mpi_rank_count);
  bytes.push_back(static_cast<std::uint8_t>(
      config.deferred_runtime().gpu_rule));
  bytes.push_back(static_cast<std::uint8_t>(
      config.deferred_runtime().source_rank_rule));
  AppendU32(bytes,
            config.deferred_runtime().required_injected_source_rank);
  AppendU32(bytes, config.mesh_n());
  AppendU32(bytes, static_cast<std::uint32_t>(config.polynomial_order()));
  AppendU32(bytes, static_cast<std::uint32_t>(config.q_a()));
  AppendU32(bytes, static_cast<std::uint32_t>(config.q_e()));
  AppendU32(bytes, static_cast<std::uint32_t>(config.q_e4()));
  AppendU32(bytes, static_cast<std::uint32_t>(config.q_o()));
  AppendU64(bytes, OracleBits(config.gamma()));
  AppendU64(bytes, OracleBits(config.tau()));
  AppendU64(bytes, OracleBits(config.dt()));
  AppendU64(bytes, config.accepted_steps());
  AppendU64(bytes, OracleBits(config.final_time()));
  for (const std::uint64_t step : config.output_steps())
  {
    AppendU64(bytes, step);
  }
  AppendU64(bytes, config.scenario_checkpoint_step());
  bytes.push_back(static_cast<std::uint8_t>(config.run_role()));
  AppendPath(bytes, config.output_directory());
  AppendPath(bytes, config.evidence_directory());
  bytes.push_back(static_cast<std::uint8_t>(config.validation_capture()));
  AppendPath(bytes, config.checkpoint_out());
  AppendOptionalStep(bytes, config.checkpoint_step());
  AppendPath(bytes, config.restart_from());
  bytes.push_back(static_cast<std::uint8_t>(config.stop_after_checkpoint()));
  AppendInjection(bytes, config.injection());
  bytes.push_back(static_cast<std::uint8_t>(config.restart_mode()));
  bytes.push_back(static_cast<std::uint8_t>(config.restart_trajectory()));
  bytes.push_back(static_cast<std::uint8_t>(config.rollback_obligation()));
  bytes.push_back(static_cast<std::uint8_t>(config.source_rank_rule()));
  AppendU32(bytes, config.retry_limit());
  AppendU64(bytes, config.seed());

  bytes.push_back(1); // v2 restart-delta schema
  bytes.push_back(6); // exact field count
  bytes.push_back(1);
  bytes.push_back(static_cast<std::uint8_t>(ApplicationRunRoleV1::split_writer));
  bytes.push_back(static_cast<std::uint8_t>(ApplicationRunRoleV1::split_reader));
  bytes.push_back(2);
  bytes.push_back(1);
  bytes.push_back(3);
  AppendOptionalStep(bytes, OptionalCheckpointStepV1::Present(
                                config.scenario_checkpoint_step()));
  AppendOptionalStep(bytes, {});
  bytes.push_back(4);
  bytes.push_back(static_cast<std::uint8_t>(StopAfterCheckpointV1::yes));
  bytes.push_back(static_cast<std::uint8_t>(StopAfterCheckpointV1::no));
  bytes.push_back(5);
  AppendInjection(bytes, {});
  AppendInjection(bytes,
                  {FailureInjectionModeV1::prescribed,
                   smooth_periodic_euler_v1_failure_phase,
                   smooth_periodic_euler_v1_smoke_failure_target_step,
                   config.deferred_runtime().required_injected_source_rank});
  bytes.push_back(6);
  AppendU32(bytes, 0);
  AppendU32(bytes, smooth_periodic_euler_v1_retry_limit);

  AppendIdentity(bytes, runtime.physical_gpu_uuid);
  AppendIdentity(bytes,
                 StableIdentity::Copy(checkpoint_partition_layout_rule_id_v1));
  AppendHash(bytes, runtime.ownership_hash);
  AppendDescriptor(bytes, descriptor);
  AppendU32(bytes, 1);
  AppendU64(bytes, fixture.interior.size());
  AppendU64(bytes, fixture.trace.size());
  const StateHash interior_hash = OracleCoefficientHash(fixture.interior);
  const StateHash trace_hash = OracleCoefficientHash(fixture.trace);
  AppendU32(bytes, 0);
  for (const auto& block :
       std::array<std::pair<const State*, StateHash>, 2>{{
           {&fixture.interior, interior_hash}, {&fixture.trace, trace_hash}}})
  {
    AppendU32(bytes, 4);
    AppendU64(bytes, block.first->size());
    AppendU64(bytes, block.first->size() / 4);
    bytes.push_back(
        static_cast<std::uint8_t>(CoefficientOrderingV1::component_major));
    AppendU64(bytes, block.first->size() * sizeof(double));
    AppendHash(bytes, block.second);
  }
  for (const State* state : {&fixture.interior, &fixture.trace})
  {
    for (const double value : *state) { AppendU64(bytes, OracleBits(value)); }
  }
  WriteU64(bytes, 20, bytes.size() + StateHash{}.bytes.size());
  AppendHash(bytes, OracleHash(bytes.data(), bytes.size()));
  return bytes;
}

std::uint64_t OracleCoordinateFingerprint(
    const StateCoordinates& coordinates) noexcept
{
  const std::array<std::uint64_t, 5> fields{{
      OracleBits(coordinates.time), coordinates.step,
      coordinates.generation, coordinates.cursor, OracleBits(coordinates.dt)}};
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (const std::uint64_t field : fields)
  {
    for (unsigned int shift = 0; shift < 64; shift += 8)
    {
      hash ^= static_cast<std::uint8_t>(field >> shift);
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
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
  std::size_t config_run_hash = 0;
  std::size_t output_directory_identity = 0;
  std::size_t descriptor_interior_hash = 0;
  std::size_t descriptor_coordinate_time = 0;
  std::size_t descriptor_coordinate_fingerprint = 0;
  std::size_t manifest_rank_count = 0;
  std::size_t aggregate_interior_count = 0;
  std::size_t rank_record = 0;
  std::size_t interior_component_count = 0;
  std::size_t interior_coefficient_count = 0;
  std::size_t interior_dofs_per_component = 0;
  std::size_t interior_byte_count = 0;
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

std::size_t PathField(const std::vector<std::uint8_t>& bytes,
                      std::size_t offset,
                      const char* name,
                      std::vector<NamedOffset>& fields)
{
  fields.push_back({name, offset});
  const bool present = bytes[offset++] != 0;
  return present ? IdentityField(bytes, offset, name, fields) : offset;
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
    if (std::string_view{name} == "config.run-sha")
    {
      layout.config_run_hash = offset;
    }
    offset += 32;
  }
  const std::array<std::pair<const char*, std::size_t>, 23> config_scalars{{
      {"config.scenario-enum", 1}, {"config.backend", 1},
      {"config.interior-space", 1}, {"config.deferred-ranks", 4},
      {"config.deferred-gpu", 1}, {"config.deferred-source-rule", 1},
      {"config.deferred-source-rank", 4}, {"config.mesh-n", 4},
      {"config.p", 4},             {"config.qA", 4},
      {"config.qE", 4},            {"config.qE4", 4},
      {"config.qO", 4},            {"config.gamma", 8},
      {"config.tau", 8},           {"config.dt", 8},
      {"config.steps", 8},         {"config.final-time", 8},
      {"config.output-0", 8},      {"config.output-1", 8},
      {"config.output-2", 8},      {"config.scenario-checkpoint-step", 8},
      {"config.writer-role", 1}}};
  for (const auto& field : config_scalars)
  {
    layout.exact_fields.push_back({field.first, offset});
    offset += field.second;
  }
  layout.output_directory_identity = offset + 3;
  offset = PathField(bytes, offset, "config.output-directory",
                     layout.exact_fields);
  offset = PathField(bytes, offset, "config.evidence-directory",
                     layout.exact_fields);
  layout.exact_fields.push_back({"config.capture", offset++});
  offset = PathField(bytes, offset, "config.checkpoint-out",
                     layout.exact_fields);
  layout.exact_fields.push_back({"config.checkpoint-step-present", offset});
  ++offset;
  layout.exact_fields.push_back({"config.checkpoint-step-value", offset});
  offset += 8;
  offset = PathField(bytes, offset, "config.restart-from",
                     layout.exact_fields);
  layout.exact_fields.push_back({"config.stop", offset++});
  for (const auto& field :
       std::array<std::pair<const char*, std::size_t>, 8>{{
           {"config.injection-mode", 1}, {"config.injection-phase", 1},
           {"config.injection-target", 8}, {"config.injection-source", 4},
           {"config.restart", 1}, {"config.trajectory", 1},
           {"config.rollback", 1}, {"config.source-rule", 1}}})
  {
    layout.exact_fields.push_back({field.first, offset});
    offset += field.second;
  }
  layout.exact_fields.push_back({"config.retry", offset});
  offset += 4;
  layout.exact_fields.push_back({"config.seed", offset});
  offset += 8;

  for (const auto& field :
       std::array<std::pair<const char*, std::size_t>, 26>{{
           {"delta.schema", 1}, {"delta.count", 1},
           {"delta.role-field", 1}, {"delta.writer-role", 1},
           {"delta.reader-role", 1}, {"delta.binding-field", 1},
           {"delta.binding", 1}, {"delta.checkpoint-field", 1},
           {"delta.writer-step-present", 1}, {"delta.writer-step", 8},
           {"delta.reader-step-present", 1}, {"delta.reader-step", 8},
           {"delta.stop-field", 1}, {"delta.writer-stop", 1},
           {"delta.reader-stop", 1}, {"delta.injection-field", 1},
           {"delta.writer-injection-mode", 1},
           {"delta.writer-injection-phase", 1},
           {"delta.writer-injection-target", 8},
           {"delta.writer-injection-source", 4},
           {"delta.reader-injection-mode", 1},
           {"delta.reader-injection-phase", 1},
           {"delta.reader-injection-target", 8},
           {"delta.reader-injection-source", 4},
           {"delta.retry-field", 1}, {"delta.writer-retry", 4}}})
  {
    layout.exact_fields.push_back({field.first, offset});
    offset += field.second;
  }
  layout.exact_fields.push_back({"delta.reader-retry", offset});
  offset += 4;
  offset = IdentityField(bytes, offset, "runtime.gpu-uuid",
                         layout.exact_fields);
  offset = IdentityField(bytes, offset, "runtime.partition-layout-rule",
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
    if (std::string_view{name} == "descriptor.time-coordinate")
    {
      layout.descriptor_coordinate_time = offset;
    }
    if (std::string_view{name} == "descriptor.fingerprint")
    {
      layout.descriptor_coordinate_fingerprint = offset;
    }
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
  const auto block = [&](const char* prefix, const bool interior) {
    layout.exact_fields.push_back({prefix, offset});
    if (interior) { layout.interior_component_count = offset; }
    offset += 4;
    layout.exact_fields.push_back({"rank.coefficient-count", offset});
    if (interior) { layout.interior_coefficient_count = offset; }
    offset += 8;
    layout.exact_fields.push_back({"rank.dofs-per-component", offset});
    if (interior) { layout.interior_dofs_per_component = offset; }
    offset += 8;
    layout.exact_fields.push_back({"rank.ordering", offset});
    offset += 1;
    layout.exact_fields.push_back({"rank.byte-count", offset});
    if (interior) { layout.interior_byte_count = offset; }
    offset += 8;
    layout.exact_fields.push_back({"rank.local-hash", offset});
    if (interior) { layout.interior_hash = offset; }
    offset += 32;
  };
  block("rank.interior-components", true);
  block("rank.trace-components", false);
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
  State interior = State(144, 101.0);
  State trace = State(144, 201.0);
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
    metadata.partition_layout_rule_id =
        StableIdentity::Copy("sentinel-partition-rule");
    metadata.scientific_config_sha256 = FilledHash(0xd0);
    metadata.writer_run_config_sha256 = FilledHash(0xe0);
    metadata.reader_run_config_sha256 = FilledHash(0xe8);
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
              {interior.data(), interior.size(), interior.size(), 4, 36,
               CoefficientOrderingV1::component_major},
              {trace.data(), trace.size(), trace.size(), 4, 36,
               CoefficientOrderingV1::component_major}}}};
  }
};

struct Rank2DecodeOutput
{
  std::array<State, 2> interior{{State(80, -1.0), State(64, -1.0)}};
  std::array<State, 2> trace{{State(112, -1.0), State(32, -1.0)}};
  DecodedCheckpointMetadataV1 metadata{};

  std::array<RankStateBufferV1, 2> Buffers() noexcept
  {
    return {{{0,
              {interior[0].data(), interior[0].size(), interior[0].size(), 4,
               20, CoefficientOrderingV1::component_major},
              {trace[0].data(), trace[0].size(), trace[0].size(), 4, 28,
               CoefficientOrderingV1::component_major}},
             {1,
              {interior[1].data(), interior[1].size(), interior[1].size(), 4,
               16, CoefficientOrderingV1::component_major},
              {trace[1].data(), trace[1].size(), trace[1].size(), 4, 8,
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
      left.partition_layout_rule_id == right.partition_layout_rule_id &&
      left.writer_run_role == right.writer_run_role &&
      left.scientific_config_sha256 == right.scientific_config_sha256 &&
      left.writer_run_config_sha256 == right.writer_run_config_sha256 &&
      left.reader_run_config_sha256 == right.reader_run_config_sha256 &&
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

// Historical full-stream draft fixture retained to prove that v2 does not
// silently accept the superseded v1 wire identity.
constexpr std::string_view rejected_v1_artifact_hex =
    "44474947522d4844472d4d454d56310001000000de0d00000000000033006578702e6864672e65756c65722e636f6e76"
    "656e74696f6e616c2e736d6f6f74685f70657273697374656e742e7738612e76312100636f6e76656e74696f6e616c2d"
    "736d6f6f74682d70657273697374656e742d763109005738412d534d4f4b453400636865636b706f696e742e6864672e"
    "65756c65722e736d6f6f74682d706572696f6469632e736d6f6b652e737465702d312e76311700706572696f6469632d"
    "63617274657369616e2d717561641f00736f757263652e65756c65722e736d6f6f74682e616e616c797469632e763129"
    "00666c75782e6864672e636f6e76656e74696f6e616c2e66697865642d7363616c61722d7461752e76310e006261636b"
    "776172645f65756c65722f006c322d696e746572696f722e747275652d74726163652d6f776e65642e636f6d706f6e65"
    "6e742d6d616a6f722e7631909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4"
    "b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecf000003000000010000000b00000010000000140000"
    "0015000000666666666666f63f00000000000004409a9999999999a93f0300000000000000333333333333c33f000000"
    "000000000001000000000000000300000000000000010000000000000000000100000000000000020000000000000008"
    "000001000000314b4f4d5341385704006e6f6e654000706572696f6469632d63617274657369616e2e636f6e74696775"
    "6f75732d656c656d656e742e6d696e2d61646a6163656e742d666163652d6f776e65722e763170717273747576777879"
    "7a7b7c7d7e7f808182838485868788898a8b8c8d8e8f010000000100150065756c65722e636f6d707265737369626c65"
    "2e326410006864672e636f6e76656e74696f6e616c17006d66656d2e64675f696e746572666163652e74726163650900"
    "7461752e66697865640e006261636b776172645f65756c65720a006370752e73657269616c0400667036341000643964"
    "36353236636331373439393830101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132"
    "333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162"
    "636465666768696a6b6c6d6e6f01000000010000002f006c322d696e746572696f722e747275652d74726163652d6f77"
    "6e65642e636f6d706f6e656e742d6d616a6f722e763104006670363401000000000000009a9999999999a93f9a999999"
    "9999a93f0100000000000000010000000000000001000000000000009a9999999999a93f12e4ea20d43b982fe0dae187"
    "1c4c4e987465c19e3e4a5accf48be2b532aeb1d1de1621432dc7d0edf963c248575b26fe92bc5ae29a5ac6d027732c1a"
    "5c68f5deb8130641ecb1ae28010000009000000000000000900000000000000000000000040000009000000000000000"
    "240000000000000000800400000000000005bccc0bd8f36d56256ec2b51f43394c5657b3d503feb5c83f60b42ffb384b"
    "e00400000090000000000000002400000000000000008004000000000000cfd19833fb0bbcdacb37ece36935eabdc4c7"
    "510b69246bd5866599a659882126000000000000f03f0000000000000040000000000000084000000000000010400000"
    "00000000144000000000000018400000000000001c400000000000002040000000000000224000000000000024400000"
    "00000000264000000000000028400000000000002a400000000000002c400000000000002e4000000000000030400000"
    "000000003140000000000000324000000000000033400000000000003440000000000000354000000000000036400000"
    "000000003740000000000000384000000000000039400000000000003a400000000000003b400000000000003c400000"
    "000000003d400000000000003e400000000000003f400000000000004040000000000080404000000000000041400000"
    "000000804140000000000000424000000000008042400000000000004340000000000080434000000000000044400000"
    "000000804440000000000000454000000000008045400000000000004640000000000080464000000000000047400000"
    "00000080474000000000000048400000000000804840000000000000494000000000008049400000000000004a400000"
    "000000804a400000000000004b400000000000804b400000000000004c400000000000804c400000000000004d400000"
    "000000804d400000000000004e400000000000804e400000000000004f400000000000804f4000000000000050400000"
    "00000040504000000000008050400000000000c050400000000000005140000000000040514000000000008051400000"
    "000000c051400000000000005240000000000040524000000000008052400000000000c0524000000000000053400000"
    "00000040534000000000008053400000000000c053400000000000005440000000000040544000000000008054400000"
    "000000c054400000000000005540000000000040554000000000008055400000000000c0554000000000000056400000"
    "00000040564000000000008056400000000000c056400000000000005740000000000040574000000000008057400000"
    "000000c057400000000000005840000000000040584000000000008058400000000000c0584000000000000059400000"
    "00000040594000000000008059400000000000c059400000000000005a400000000000405a400000000000805a400000"
    "000000c05a400000000000005b400000000000405b400000000000805b400000000000c05b400000000000005c400000"
    "000000405c400000000000805c400000000000c05c400000000000005d400000000000405d400000000000805d400000"
    "000000c05d400000000000005e400000000000405e400000000000805e400000000000c05e400000000000005f400000"
    "000000405f400000000000805f400000000000c05f400000000000006040000000000020604000000000004060400000"
    "00000060604000000000008060400000000000a060400000000000c060400000000000e0604000000000000061400000"
    "0000002061400000000000406140000000000060614000000000008061400000000000a061400000000000c061400000"
    "000000e0614000000000000062400000000000488f400000000000508f400000000000588f400000000000608f400000"
    "000000688f400000000000708f400000000000788f400000000000808f400000000000888f400000000000908f400000"
    "000000988f400000000000a08f400000000000a88f400000000000b08f400000000000b88f400000000000c08f400000"
    "000000c88f400000000000d08f400000000000d88f400000000000e08f400000000000e88f400000000000f08f400000"
    "000000f88f4000000000000090400000000000049040000000000008904000000000000c904000000000001090400000"
    "000000149040000000000018904000000000001c90400000000000209040000000000024904000000000002890400000"
    "0000002c904000000000003090400000000000349040000000000038904000000000003c904000000000004090400000"
    "000000449040000000000048904000000000004c90400000000000509040000000000054904000000000005890400000"
    "0000005c904000000000006090400000000000649040000000000068904000000000006c904000000000007090400000"
    "000000749040000000000078904000000000007c90400000000000809040000000000084904000000000008890400000"
    "0000008c904000000000009090400000000000949040000000000098904000000000009c90400000000000a090400000"
    "000000a490400000000000a890400000000000ac90400000000000b090400000000000b490400000000000b890400000"
    "000000bc90400000000000c090400000000000c490400000000000c890400000000000cc90400000000000d090400000"
    "000000d490400000000000d890400000000000dc90400000000000e090400000000000e490400000000000e890400000"
    "000000ec90400000000000f090400000000000f490400000000000f890400000000000fc904000000000000091400000"
    "000000049140000000000008914000000000000c91400000000000109140000000000014914000000000001891400000"
    "0000001c914000000000002091400000000000249140000000000028914000000000002c914000000000003091400000"
    "000000349140000000000038914000000000003c91400000000000409140000000000044914000000000004891400000"
    "0000004c914000000000005091400000000000549140000000000058914000000000005c914000000000006091400000"
    "000000649140000000000068914000000000006c91400000000000709140000000000074914000000000007891400000"
    "0000007c914000000000008091400000000000849140000000000088914000000000008c914000000000009091400000"
    "000000949140000000000098914000000000009c91400000000000a091400000000000a491400000000000a891400000"
    "000000ac91400000000000b091400000000000b491400000000000b891400000000000bc91400000000000c091400000"
    "000000c491400000000000c891400000000000cc91400000000000d091400000000000d491400000000000d891400000"
    "000000dc91400000000000e09140b13e2e869287ea62be89252fb3e7c91e87c01291bea201f462e9c8780cfc120a";

ScenarioCoefficientExtentsV1 IndependentExtents(
    const std::uint32_t n,
    const std::uint32_t polynomial_order,
    const std::uint32_t rank_count)
{
  ScenarioCoefficientExtentsV1 result{};
  result.rank_count = rank_count;
  const std::uint64_t elements =
      static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n);
  const auto owner = [elements, rank_count](
                         const std::uint64_t element) noexcept {
    return static_cast<std::uint32_t>(element * rank_count / elements);
  };
  std::array<std::uint64_t, application_checkpoint_max_rank_count_v1>
      element_counts{};
  std::array<std::uint64_t, application_checkpoint_max_rank_count_v1>
      face_counts{};
  for (std::uint64_t element = 0; element < elements; ++element)
  {
    ++element_counts[owner(element)];
  }
  for (std::uint64_t face = 0; face < 2 * elements; ++face)
  {
    const bool vertical = face < elements;
    const std::uint64_t local = vertical ? face : face - elements;
    const std::uint64_t row = local / n;
    const std::uint64_t column = local % n;
    const std::uint64_t first = row * n + column;
    const std::uint64_t second = vertical
        ? row * n + (column + n - 1) % n
        : ((row + n - 1) % n) * n + column;
    ++face_counts[std::min(owner(first), owner(second))];
  }
  const std::uint64_t p1 = static_cast<std::uint64_t>(polynomial_order) + 1;
  for (std::uint32_t rank = 0; rank < rank_count; ++rank)
  {
    RankCoefficientExtentsV1& extents = result.ranks[rank];
    extents.interior_dofs_per_component = element_counts[rank] * p1 * p1;
    extents.interior_coefficient_count =
        extents.interior_dofs_per_component * 4;
    extents.trace_dofs_per_component = face_counts[rank] * p1;
    extents.trace_coefficient_count = extents.trace_dofs_per_component * 4;
    result.aggregate_interior_coefficient_count +=
        extents.interior_coefficient_count;
    result.aggregate_trace_coefficient_count += extents.trace_coefficient_count;
  }
  return result;
}

bool SameExtents(const ScenarioCoefficientExtentsV1& left,
                 const ScenarioCoefficientExtentsV1& right) noexcept
{
  if (left.rank_count != right.rank_count ||
      left.aggregate_interior_coefficient_count !=
          right.aggregate_interior_coefficient_count ||
      left.aggregate_trace_coefficient_count !=
          right.aggregate_trace_coefficient_count)
  {
    return false;
  }
  for (std::uint32_t rank = 0; rank < left.rank_count; ++rank)
  {
    const RankCoefficientExtentsV1& a = left.ranks[rank];
    const RankCoefficientExtentsV1& b = right.ranks[rank];
    if (a.interior_dofs_per_component != b.interior_dofs_per_component ||
        a.interior_coefficient_count != b.interior_coefficient_count ||
        a.trace_dofs_per_component != b.trace_dofs_per_component ||
        a.trace_coefficient_count != b.trace_coefficient_count)
    {
      return false;
    }
  }
  return true;
}

bool CheckIndependentOracle()
{
  const std::array<std::uint8_t, 3> abc{{'a', 'b', 'c'}};
  return Check(
      Hex(OracleHash(abc.data(), abc.size())) ==
          "ba7816bf8f01cfea414140de5dae2223"
          "b00361a396177a9cb410ff61f20015ad",
      "independent_sha256_matches_hard_coded_FIPS_vector");
}

bool CheckExactScenarioExtents()
{
  const ApplicationValidation serial_validation =
      TestValidation(CpuSerialBackendTag::id);
  const ApplicationValidation rank2_validation =
      TestValidation(CpuRank2BackendTag::id);
  const ValidatedApplicationConfig* serial = serial_validation.config();
  const ValidatedApplicationConfig* rank2 = rank2_validation.config();
  bool passed = Check(serial != nullptr && rank2 != nullptr,
                      "rank1_and_rank2_static_configs_validate");
  if (serial == nullptr || rank2 == nullptr) { return false; }

  const ScenarioCoefficientExtentsV1 serial_oracle =
      IndependentExtents(3, 1, 1);
  const ScenarioCoefficientExtentsV1 rank2_oracle =
      IndependentExtents(3, 1, 2);
  passed = Check(
               serial_oracle.ranks[0].interior_coefficient_count == 144 &&
                   serial_oracle.ranks[0].trace_coefficient_count == 144 &&
                   rank2_oracle.ranks[0].interior_coefficient_count == 80 &&
                   rank2_oracle.ranks[1].interior_coefficient_count == 64 &&
                   rank2_oracle.ranks[0].trace_coefficient_count == 112 &&
                   rank2_oracle.ranks[1].trace_coefficient_count == 32,
               "independent_periodic_partition_extents_are_exact") &&
      passed;
  ScenarioCoefficientExtentsV1 serial_actual{};
  ScenarioCoefficientExtentsV1 rank2_actual{};
  passed = Check(
               DeriveScenarioCoefficientExtentsV1(*serial, 1,
                                                  serial_actual) &&
                   DeriveScenarioCoefficientExtentsV1(*rank2, 2,
                                                      rank2_actual) &&
                   SameExtents(serial_actual, serial_oracle) &&
                   SameExtents(rank2_actual, rank2_oracle),
               "codec_extents_match_independent_face_enumeration") &&
      passed;
  return passed;
}

bool CheckLiveCaptureAndRepeatableEncode()
{
  const CoefficientFixture fixture;
  const PersistentResourceDescriptor resources = TestResources(1);
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation validation =
      TestValidation(CpuSerialBackendTag::id);
  const ValidatedApplicationConfig* config = validation.config();
  bool passed = Check(config != nullptr, "checkpoint_static_config_valid");
  if (config == nullptr) { return false; }

  const Slot committed{fixture.interior, fixture.trace};
  Slot candidate{};
  SerialContract uncommitted(
      committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  const std::array<SerialContractView, 1> uncommitted_view{{
      {0, &uncommitted}}};
  passed = Check(!CaptureCheckpointV1(uncommitted_view.data(),
                                      uncommitted_view.size(), *config,
                                      runtime),
                 "uncommitted_contract_cannot_be_captured") &&
      passed;
  const std::array<SerialContractView, 1> null_view{{{0, nullptr}}};
  passed = Check(!CaptureCheckpointV1(null_view.data(), null_view.size(),
                                      *config, runtime),
                 "missing_live_contract_cannot_be_captured") &&
      passed;
  Slot wrong_time_candidate{};
  SerialContract wrong_time(
      committed, wrong_time_candidate, {0.1, 0, 0, 0, 0.05}, resources);
  const std::array<SerialContractView, 1> wrong_time_view{{
      {0, &wrong_time}}};
  passed = Check(AdvanceOneCommit(wrong_time) &&
                     !CaptureCheckpointV1(wrong_time_view.data(),
                                          wrong_time_view.size(), *config,
                                          runtime),
                 "live_contract_with_wrong_finite_checkpoint_time_rejected") &&
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

bool CheckRank2CaptureAndAllRankEvidence()
{
  Rank2CoefficientFixture fixture;
  const PersistentResourceDescriptor resources = TestResources(2);
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation validation =
      TestValidation(CpuRank2BackendTag::id);
  const ApplicationValidation reader_validation =
      TestReaderValidation(CpuRank2BackendTag::id);
  const ValidatedApplicationConfig* config = validation.config();
  const ValidatedApplicationConfig* reader_config =
      reader_validation.config();
  if (config == nullptr || reader_config == nullptr)
  {
    return Check(false, "rank2_static_config_valid");
  }

  const Slot committed0{fixture.interior[0], fixture.trace[0]};
  const Slot committed1{fixture.interior[1], fixture.trace[1]};
  Slot candidate0{};
  Slot candidate1{};
  Rank2Contract contract0(
      committed0, candidate0, {0.0, 0, 0, 0, 0.05}, resources);
  Rank2Contract contract1(
      committed1, candidate1, {0.0, 0, 0, 0, 0.05}, resources);
  bool passed = Check(AdvanceOneCommit(contract0) &&
                          AdvanceOneCommit(contract1),
                      "rank2_live_contracts_commit");
  const std::array<Rank2ContractView, 2> valid{{
      {0, &contract0}, {1, &contract1}}};
  const std::array<Rank2ContractView, 2> duplicate{{
      {0, &contract0}, {1, &contract0}}};
  const std::array<Rank2ContractView, 2> out_of_order{{
      {1, &contract0}, {0, &contract1}}};
  const std::array<Rank2ContractView, 2> missing{{
      {0, &contract0}, {1, nullptr}}};
  passed = Check(!CaptureCheckpointV1(valid.data(), 1, *config, runtime),
                 "rank2_capture_rejects_missing_rank_record") &&
      passed;
  passed = Check(!CaptureCheckpointV1(duplicate.data(), duplicate.size(),
                                      *config, runtime),
                 "rank2_capture_rejects_duplicate_live_contract") &&
      passed;
  passed = Check(!CaptureCheckpointV1(out_of_order.data(),
                                      out_of_order.size(), *config, runtime),
                 "rank2_capture_rejects_out_of_order_rank_records") &&
      passed;
  passed = Check(!CaptureCheckpointV1(missing.data(), missing.size(), *config,
                                      runtime),
                 "rank2_capture_rejects_non_live_rank_record") &&
      passed;
  PersistentResourceDescriptor incompatible_resources = resources;
  incompatible_resources.mesh_hash.bytes[0] ^= 1U;
  Slot incompatible_candidate{};
  Rank2Contract incompatible_contract(
      committed1, incompatible_candidate, {0.0, 0, 0, 0, 0.05},
      incompatible_resources);
  const std::array<Rank2ContractView, 2> incompatible{{
      {0, &contract0}, {1, &incompatible_contract}}};
  passed = Check(AdvanceOneCommit(incompatible_contract) &&
                     !CaptureCheckpointV1(incompatible.data(),
                                          incompatible.size(), *config,
                                          runtime),
                 "rank2_capture_rejects_non_common_descriptor") &&
      passed;

  const auto capture = CaptureCheckpointV1(valid.data(), valid.size(),
                                            *config, runtime);
  passed = Check(capture.has_value() && capture->rank_count() == 2,
                 "rank2_capture_retains_one_descriptor_per_rank") &&
      passed;
  if (!capture) { return false; }
  auto views = fixture.Views();
  std::vector<std::uint8_t> encoded;
  passed = Check(EncodeCheckpointV1(*capture, views.data(), views.size(),
                                    encoded) == CheckpointCodecStatus::ok,
                 "rank2_exact_extent_fixture_encodes") &&
      passed;

  Rank2DecodeOutput output;
  auto buffers = output.Buffers();
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *reader_config,
                         resources,
                         runtime, buffers.data(), buffers.size(),
                         output.metadata) == CheckpointCodecStatus::ok &&
                     output.interior == fixture.interior &&
                     output.trace == fixture.trace &&
                     output.metadata.rank_count == 2 &&
                     output.metadata.aggregate_interior_coefficients == 144 &&
                     output.metadata.aggregate_true_trace_coefficients == 144,
                 "rank2_exact_extents_round_trip") &&
      passed;

  fixture.interior[0][0] += 0.25;
  views = fixture.Views();
  std::vector<std::uint8_t> unchanged{{0xaa, 0xbb}};
  passed = Check(
               EncodeCheckpointV1(*capture, views.data(), views.size(),
                                  unchanged) ==
                       CheckpointCodecStatus::local_payload_hash_mismatch &&
                   unchanged == std::vector<std::uint8_t>({0xaa, 0xbb}),
               "non_injection_source_rank_payload_requires_own_live_hash") &&
      passed;
  return passed;
}

bool CheckNonEvidenceSplitRoundTrip()
{
  const CoefficientFixture fixture;
  const PersistentResourceDescriptor resources = TestResources(1);
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation writer_validation =
      TestValidation(CpuSerialBackendTag::id, false);
  const ApplicationValidation reader_validation =
      TestReaderValidation(CpuSerialBackendTag::id, false);
  const ValidatedApplicationConfig* writer = writer_validation.config();
  const ValidatedApplicationConfig* reader = reader_validation.config();
  bool passed = Check(writer != nullptr && reader != nullptr,
                      "non_evidence_split_configs_validate");
  if (writer == nullptr || reader == nullptr) { return false; }

  StateHash derived_writer_hash{};
  passed = Check(
               writer->validation_capture() == ValidationCaptureModeV1::none &&
                   reader->validation_capture() ==
                       ValidationCaptureModeV1::none &&
                   !writer->evidence_directory().present() &&
                   !reader->evidence_directory().present() &&
                   DeriveSplitWriterRunConfigHashV1(*reader,
                                                    derived_writer_hash) &&
                   derived_writer_hash == writer->run_config_sha256() &&
                   writer->run_config_sha256() !=
                       reader->run_config_sha256() &&
                   writer->scientific_config_sha256() ==
                       reader->scientific_config_sha256(),
               "absent_evidence_presence_preserved_in_writer_hash") &&
      passed;

  std::vector<std::uint8_t> encoded;
  passed = Check(EncodeFixture(fixture, *writer, resources, runtime, encoded),
                 "non_evidence_split_writer_encodes") &&
      passed;
  DecodeSentinel output;
  auto buffers = output.Buffers();
  passed = Check(
               DecodeCheckpointV1(
                   {encoded.data(), encoded.size()}, *reader, resources,
                   runtime, buffers.data(), buffers.size(), output.metadata) ==
                       CheckpointCodecStatus::ok &&
                   output.interior == fixture.interior &&
                   output.trace == fixture.trace &&
                   output.metadata.scientific_config_sha256 ==
                       reader->scientific_config_sha256() &&
                   output.metadata.writer_run_config_sha256 ==
                       writer->run_config_sha256() &&
                   output.metadata.reader_run_config_sha256 ==
                       reader->run_config_sha256() &&
                   output.metadata.descriptor.coordinates.step == 1 &&
                   output.metadata.descriptor.coordinates.time == 0.05,
               "non_evidence_writer_reader_codec_round_trip") &&
      passed;
  return passed;
}

bool CheckFunctionalSplitRoundTrip()
{
  const FunctionalCoefficientFixture fixture;
  const PersistentResourceDescriptor resources = TestResources(1, 2);
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation writer_validation =
      FunctionalTestValidation(false);
  const ApplicationValidation reader_validation =
      FunctionalTestValidation(true);
  const ValidatedApplicationConfig* writer = writer_validation.config();
  const ValidatedApplicationConfig* reader = reader_validation.config();
  bool passed = Check(writer != nullptr && reader != nullptr,
                      "functional_split_configs_validate");
  if (writer == nullptr || reader == nullptr) { return false; }

  StateHash derived_writer_hash{};
  ScenarioCoefficientExtentsV1 extents{};
  passed = Check(
               writer->scenario() == ApplicationScenarioV1::functional &&
                   writer->polynomial_order() == 2 && writer->mesh_n() == 6 &&
                   writer->run_role() == ApplicationRunRoleV1::split_writer &&
                   reader->run_role() == ApplicationRunRoleV1::split_reader &&
                   writer->checkpoint_out().present() &&
                   reader->restart_from().present() &&
                   writer->checkpoint_out().identity() ==
                       reader->restart_from().identity() &&
                   writer->checkpoint_step().present &&
                   writer->checkpoint_step().value == 10 &&
                   !reader->checkpoint_step().present &&
                   writer->stop_after_checkpoint() ==
                       StopAfterCheckpointV1::yes &&
                   reader->stop_after_checkpoint() ==
                       StopAfterCheckpointV1::no &&
                   writer->injection() == FailureInjectionControlV1{} &&
                   reader->injection() ==
                       FailureInjectionControlV1{
                           FailureInjectionModeV1::prescribed,
                           smooth_periodic_euler_v1_failure_phase,
                           smooth_periodic_euler_v1_functional_failure_target_step,
                           0} &&
                   writer->retry_limit() == 0 && reader->retry_limit() == 1 &&
                   DeriveSplitWriterRunConfigHashV1(*reader,
                                                    derived_writer_hash) &&
                   derived_writer_hash == writer->run_config_sha256() &&
                   writer->run_config_sha256() !=
                       reader->run_config_sha256() &&
                   writer->scientific_config_sha256() ==
                       reader->scientific_config_sha256() &&
                   DeriveScenarioCoefficientExtentsV1(*reader, 1, extents) &&
                   extents.ranks[0].interior_dofs_per_component == 324 &&
                   extents.ranks[0].interior_coefficient_count == 1296 &&
                   extents.ranks[0].trace_dofs_per_component == 216 &&
                   extents.ranks[0].trace_coefficient_count == 864,
               "functional_exact_six_delta_hash_and_extent_contract") &&
      passed;

  const auto capture = MakeSerialCapture(
      *writer, resources, runtime, fixture,
      smooth_periodic_euler_v1_functional_checkpoint_step);
  passed = Check(capture.has_value(),
                 "functional_step10_live_capture") &&
      passed;
  if (!capture) { return false; }
  const auto views = fixture.Views();
  std::vector<std::uint8_t> encoded;
  passed = Check(EncodeCheckpointV1(*capture, views.data(), views.size(),
                                    encoded) == CheckpointCodecStatus::ok,
                 "functional_step10_checkpoint_encodes") &&
      passed;

  State decoded_interior(fixture.interior.size(), -1.0);
  State decoded_trace(fixture.trace.size(), -1.0);
  RankStateBufferV1 buffer{
      0,
      {decoded_interior.data(), decoded_interior.size(),
       decoded_interior.size(), 4, 324,
       CoefficientOrderingV1::component_major},
      {decoded_trace.data(), decoded_trace.size(), decoded_trace.size(), 4,
       216, CoefficientOrderingV1::component_major}};
  DecodedCheckpointMetadataV1 metadata{};
  passed = Check(
               DecodeCheckpointV1(
                   {encoded.data(), encoded.size()}, *reader, resources,
                   runtime, &buffer, 1, metadata) ==
                       CheckpointCodecStatus::ok &&
                   decoded_interior == fixture.interior &&
                   decoded_trace == fixture.trace &&
                   metadata.mesh_n == 6 &&
                   metadata.descriptor.polynomial_order == 2 &&
                   metadata.descriptor.coordinates.step == 10 &&
                   metadata.descriptor.coordinates.generation == 10 &&
                   metadata.descriptor.coordinates.cursor == 10 &&
                   ScenarioCheckpointTimeMatchesV1(
                       metadata.descriptor.coordinates.time, *reader) &&
                   metadata.aggregate_interior_coefficients == 1296 &&
                   metadata.aggregate_true_trace_coefficients == 864 &&
                   metadata.scientific_config_sha256 ==
                       reader->scientific_config_sha256() &&
                   metadata.writer_run_config_sha256 ==
                       writer->run_config_sha256() &&
                   metadata.reader_run_config_sha256 ==
                       reader->run_config_sha256(),
               "functional_step10_target14_delta_round_trip") &&
      passed;
  return passed;
}

bool CheckGoldenRoundTrip()
{
  const CoefficientFixture fixture;
  const PersistentResourceDescriptor resources = TestResources(1);
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation validation =
      TestValidation(CpuSerialBackendTag::id);
  const ApplicationValidation reader_validation =
      TestReaderValidation(CpuSerialBackendTag::id);
  const ValidatedApplicationConfig* config = validation.config();
  const ValidatedApplicationConfig* reader_config =
      reader_validation.config();
  if (config == nullptr || reader_config == nullptr)
  {
    return Check(false, "golden_static_config");
  }
  std::vector<std::uint8_t> encoded;
  bool passed = Check(EncodeFixture(fixture, *config, resources, runtime,
                                    encoded),
                      "golden_fixture_encodes");
  const StateHash expected_interior = OracleCoefficientHash(fixture.interior);
  const StateHash expected_trace = OracleCoefficientHash(fixture.trace);
  CheckpointDescriptor expected_descriptor{};
  const Slot committed{fixture.interior, fixture.trace};
  Slot candidate{};
  SerialContract contract(
      committed, candidate, {0.0, 0, 0, 0, 0.05}, resources);
  passed = Check(AdvanceOneCommit(contract) &&
                     contract.TryMakeCheckpoint(expected_descriptor),
                 "independent_expected_descriptor_available") &&
      passed;
  expected_descriptor.interior_hash =
      OracleManifestHash(1, 1152, expected_interior);
  expected_descriptor.trace_hash =
      OracleManifestHash(2, 1152, expected_trace);
  const std::vector<std::uint8_t> golden = IndependentGoldenV2(
      *config, runtime, expected_descriptor, fixture);
  passed = Check(!golden.empty() && encoded == golden,
                 "entire_v2_artifact_matches_independent_fieldwise_golden") &&
      passed;
  passed = Check(encoded.size() > 64 && encoded[0] == 'D' &&
                     encoded[14] == '2' && encoded[16] == 2 &&
                     ReadU64(encoded, 20) == encoded.size(),
                 "independent_header_field_decode") &&
      passed;

  passed = Check(
               Hex(expected_interior) ==
                       "05bccc0bd8f36d56256ec2b51f43394c"
                       "5657b3d503feb5c83f60b42ffb384be0" &&
                   Hex(expected_trace) ==
                       "cfd19833fb0bbcdacb37ece36935eabd"
                       "c4c7510b69246bd5866599a659882126",
               "hard_coded_local_payload_digests") &&
      passed;
  const LayoutOffsets layout = Locate(encoded);
  std::vector<std::uint8_t> expected_payload;
  expected_payload.reserve((fixture.interior.size() + fixture.trace.size()) *
                           sizeof(double));
  for (const State* state : {&fixture.interior, &fixture.trace})
  {
    for (const double value : *state)
    {
      AppendU64(expected_payload, OracleBits(value));
    }
  }
  passed = Check(layout.payload + expected_payload.size() + 32 ==
                         encoded.size() &&
                     std::equal(expected_payload.begin(),
                                expected_payload.end(),
                                encoded.begin() + layout.payload),
                 "all_FP64_coefficients_are_hard_coded_little_endian") &&
      passed;

  DecodeSentinel output;
  auto buffers = output.Buffers();
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *reader_config,
                         resources,
                         runtime, buffers.data(), buffers.size(),
                         output.metadata) == CheckpointCodecStatus::ok,
                 "golden_artifact_decodes_into_preallocated_buffers") &&
      passed;
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
                   output.metadata.partition_layout_rule_id.view() ==
                       checkpoint_partition_layout_rule_id_v1 &&
                   output.metadata.scientific_config_sha256 ==
                       config->scientific_config_sha256() &&
                   output.metadata.writer_run_role ==
                       ApplicationRunRoleV1::split_writer &&
                   output.metadata.writer_run_config_sha256 ==
                       config->run_config_sha256() &&
                   output.metadata.reader_run_config_sha256 ==
                       reader_config->run_config_sha256() &&
                   output.metadata.writer_run_config_sha256 !=
                       output.metadata.reader_run_config_sha256 &&
                   output.metadata.ownership_hash == runtime.ownership_hash &&
                   output.metadata.mesh_n == 3 &&
                   output.metadata.rank_count == 1 &&
                   output.metadata.aggregate_interior_coefficients == 144 &&
                   output.metadata.aggregate_true_trace_coefficients == 144 &&
                   output.metadata.artifact_hash ==
                       OracleHash(golden.data(), golden.size() - 32),
               "all_identity_descriptor_count_and_buffer_fields_decode") &&
      passed;

  CheckpointDescriptor wrong_time = expected_descriptor;
  wrong_time.coordinates.time = 0.1;
  wrong_time.coordinate_fingerprint =
      OracleCoordinateFingerprint(wrong_time.coordinates);
  passed = Check(ValidateSameTupleRestartV1(
                         wrong_time, *reader_config, resources, runtime,
                         runtime) ==
                         CheckpointCodecStatus::invalid_descriptor,
                 "restart_validation_rejects_correlated_wrong_finite_time") &&
      passed;
  return passed;
}

bool CheckEncoderControls()
{
  const CoefficientFixture fixture;
  auto views = fixture.Views();
  const PersistentResourceDescriptor resources = TestResources(1);
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation validation =
      TestValidation(CpuSerialBackendTag::id);
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
  invalid[0].interior.coefficient_count = 140;
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
  const PersistentResourceDescriptor resources = TestResources(1);
  const CheckpointRuntimeIdentityV1 runtime = TestRuntimeIdentity();
  const ApplicationValidation validation =
      TestValidation(CpuSerialBackendTag::id);
  const ApplicationValidation reader_validation =
      TestReaderValidation(CpuSerialBackendTag::id);
  const ValidatedApplicationConfig* config = validation.config();
  const ValidatedApplicationConfig* reader_config =
      reader_validation.config();
  if (config == nullptr || reader_config == nullptr)
  {
    return Check(false, "mutation_static_config");
  }
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
        {bytes.data(), bytes.size()}, *reader_config, resources, runtime,
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
        {bytes.data(), bytes.size()}, *reader_config, resources, runtime,
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
  expect_failure(HexBytes(rejected_v1_artifact_hex),
                 CheckpointCodecStatus::invalid_magic,
                 "draft_v1_magic_is_rejected");
  damaged = encoded;
  damaged[16] = 1;
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
  passed = Check(layout.exact_fields.size() == 129,
                 "frozen_v2_exact_field_mutation_census_is_129") &&
      passed;

  ApplicationOptionsView correlated_options =
      TestOptions(CpuSerialBackendTag::id);
  correlated_options.run_role = ApplicationRunRoleV1::split_writer;
  correlated_options.stop_after_checkpoint = StopAfterCheckpointV1::yes;
  correlated_options.output_directory =
      OutputDirectoryPathV1::Present("w8a/result");
  const ApplicationValidation correlated_validation =
      ValidateApplicationOptions(correlated_options);
  const ValidatedApplicationConfig* correlated_config =
      correlated_validation.config();
  if (correlated_config == nullptr)
  {
    return Check(false, "correlated_writer_config_valid");
  }
  damaged = encoded;
  const std::string_view correlated_path = "w8a/result";
  std::copy(correlated_path.begin(), correlated_path.end(),
            damaged.begin() + layout.output_directory_identity);
  std::copy(correlated_config->run_config_sha256().bytes.begin(),
            correlated_config->run_config_sha256().bytes.end(),
            damaged.begin() + layout.config_run_hash);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::invalid_config_identity,
                 "correlated_config_rehash_and_reseal_rejected");

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
      OracleHash(damaged.data() + layout.payload, 1152);
  std::copy(nan_local.bytes.begin(), nan_local.bytes.end(),
            damaged.begin() + layout.interior_hash);
  const StateHash nan_aggregate = OracleManifestHash(1, 1152, nan_local);
  std::copy(nan_aggregate.bytes.begin(), nan_aggregate.bytes.end(),
            damaged.begin() + layout.descriptor_interior_hash);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::noncanonical_fp64,
                 "nonfinite_FP64_rejected_in_first_pass");

  damaged = encoded;
  const StateCoordinates wrong_coordinates{0.1, 1, 1, 1, 0.05};
  WriteU64(damaged, layout.descriptor_coordinate_time,
           OracleBits(wrong_coordinates.time));
  WriteU64(damaged, layout.descriptor_coordinate_fingerprint,
           OracleCoordinateFingerprint(wrong_coordinates));
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::invalid_descriptor,
                 "correlated_wrong_finite_checkpoint_time_rejected");

  damaged = encoded;
  constexpr std::uint64_t shortened_count = 140;
  constexpr std::uint64_t shortened_dofs = 35;
  constexpr std::uint64_t shortened_bytes = shortened_count * sizeof(double);
  const std::size_t removed_begin =
      layout.payload + static_cast<std::size_t>(shortened_bytes);
  const std::size_t removed_end = layout.payload + 144 * sizeof(double);
  damaged.erase(damaged.begin() + removed_begin,
                damaged.begin() + removed_end);
  WriteU64(damaged, 20, damaged.size());
  WriteU64(damaged, layout.aggregate_interior_count, shortened_count);
  WriteU64(damaged, layout.interior_coefficient_count, shortened_count);
  WriteU64(damaged, layout.interior_dofs_per_component, shortened_dofs);
  WriteU64(damaged, layout.interior_byte_count, shortened_bytes);
  const StateHash shortened_local =
      OracleHash(damaged.data() + layout.payload, shortened_bytes);
  std::copy(shortened_local.bytes.begin(), shortened_local.bytes.end(),
            damaged.begin() + layout.interior_hash);
  const StateHash shortened_aggregate =
      OracleManifestHash(1, shortened_bytes, shortened_local);
  std::copy(shortened_aggregate.bytes.begin(),
            shortened_aggregate.bytes.end(),
            damaged.begin() + layout.descriptor_interior_hash);
  static_cast<void>(Seal(damaged));
  expect_failure(damaged, CheckpointCodecStatus::coefficient_count_mismatch,
                 "correlated_count_payload_and_hash_mutation_rejected");

  DecodeSentinel capacity_output;
  const DecodeSentinel original = capacity_output;
  auto short_buffers = capacity_output.Buffers();
  short_buffers[0].interior.capacity = 143;
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *reader_config,
                         resources,
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
                         {encoded.data(), encoded.size()}, *reader_config,
                         resources,
                         wrong_runtime, runtime_buffers.data(),
                         runtime_buffers.size(), runtime_output.metadata) ==
                         CheckpointCodecStatus::invalid_runtime_identity &&
                     SentinelUnchanged(runtime_output, runtime_original),
                 "ownership_identity_checked_before_any_output_write") &&
      passed;

  DecodeSentinel input_alias_output;
  const DecodeSentinel input_alias_original = input_alias_output;
  const std::vector<std::uint8_t> encoded_before_alias = encoded;
  auto input_alias_buffers = input_alias_output.Buffers();
  input_alias_buffers[0].interior.data =
      reinterpret_cast<double*>(encoded.data());
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *reader_config,
                         resources, runtime, input_alias_buffers.data(),
                         input_alias_buffers.size(),
                         input_alias_output.metadata) ==
                         CheckpointCodecStatus::output_buffer_overlap &&
                     encoded == encoded_before_alias &&
                     SentinelUnchanged(input_alias_output,
                                       input_alias_original),
                 "encoded_input_output_alias_rejected") &&
      passed;

  DecodeSentinel metadata_output_alias;
  const DecodeSentinel metadata_output_original = metadata_output_alias;
  auto metadata_output_buffers = metadata_output_alias.Buffers();
  metadata_output_buffers[0].interior.data =
      reinterpret_cast<double*>(&metadata_output_alias.metadata);
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *reader_config,
                         resources, runtime, metadata_output_buffers.data(),
                         metadata_output_buffers.size(),
                         metadata_output_alias.metadata) ==
                         CheckpointCodecStatus::output_buffer_overlap &&
                     SentinelUnchanged(metadata_output_alias,
                                       metadata_output_original),
                 "metadata_output_alias_rejected") &&
      passed;

  DecodeSentinel output_alias;
  const DecodeSentinel output_alias_original = output_alias;
  State overlapping_storage(288, -7.0);
  const State overlapping_original = overlapping_storage;
  auto output_alias_buffers = output_alias.Buffers();
  output_alias_buffers[0].interior.data = overlapping_storage.data();
  output_alias_buffers[0].owned_true_trace.data =
      overlapping_storage.data() + 72;
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *reader_config,
                         resources, runtime, output_alias_buffers.data(),
                         output_alias_buffers.size(), output_alias.metadata) ==
                         CheckpointCodecStatus::output_buffer_overlap &&
                     overlapping_storage == overlapping_original &&
                     SentinelUnchanged(output_alias, output_alias_original),
                 "partially_overlapping_outputs_rejected") &&
      passed;

  DecodeSentinel overflow_output;
  const DecodeSentinel overflow_original = overflow_output;
  auto overflow_buffers = overflow_output.Buffers();
  overflow_buffers[0].interior.data = reinterpret_cast<double*>(
      std::numeric_limits<std::uintptr_t>::max() - 7U);
  passed = Check(DecodeCheckpointV1(
                         {encoded.data(), encoded.size()}, *reader_config,
                         resources, runtime, overflow_buffers.data(),
                         overflow_buffers.size(), overflow_output.metadata) ==
                         CheckpointCodecStatus::output_buffer_overlap &&
                     SentinelUnchanged(overflow_output, overflow_original),
                 "synthetic_output_address_overflow_rejected") &&
      passed;

  const std::size_t alias_words =
      (encoded.size() + sizeof(std::max_align_t) - 1) /
          sizeof(std::max_align_t) +
      1;
  std::vector<std::max_align_t> metadata_input_storage(alias_words);
  auto* metadata_input_bytes = reinterpret_cast<std::uint8_t*>(
      metadata_input_storage.data());
  std::memcpy(metadata_input_bytes, encoded.data(), encoded.size());
  auto* overlapping_metadata =
      ::new (static_cast<void*>(metadata_input_bytes))
          DecodedCheckpointMetadataV1{};
  overlapping_metadata->card_id = StableIdentity::Copy("alias-sentinel");
  const DecodedCheckpointMetadataV1 overlapping_metadata_original =
      *overlapping_metadata;
  DecodeSentinel metadata_input_output;
  auto metadata_input_buffers = metadata_input_output.Buffers();
  passed = Check(DecodeCheckpointV1(
                         {metadata_input_bytes, encoded.size()},
                         *reader_config, resources, runtime,
                         metadata_input_buffers.data(),
                         metadata_input_buffers.size(),
                         *overlapping_metadata) ==
                         CheckpointCodecStatus::output_buffer_overlap &&
                     SameMetadata(*overlapping_metadata,
                                  overlapping_metadata_original),
                 "metadata_encoded_input_alias_rejected_before_parse") &&
      passed;
  overlapping_metadata->~DecodedCheckpointMetadataV1();
  return passed;
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckIndependentOracle() && passed;
  passed = CheckExactScenarioExtents() && passed;
  passed = CheckLiveCaptureAndRepeatableEncode() && passed;
  passed = CheckRank2CaptureAndAllRankEvidence() && passed;
  passed = CheckNonEvidenceSplitRoundTrip() && passed;
  passed = CheckFunctionalSplitRoundTrip() && passed;
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

#pragma once

/**
 * @file numerics/hdg/runtime/app/checkpoint_codec.hpp
 * @brief Canonical in-memory W8A FP64 checkpoint artifact codec.
 *
 * This slice performs no filesystem I/O and makes no durability/publication
 * claim.  The later single-owner checkpoint I/O implementation must create
 * rank temporaries exclusively, fsync every temporary, reach collective
 * manifest consensus, publish with a final atomic rename, fsync the parent
 * directory, and establish collective visibility.  No caller boolean in this
 * API stands in for any of those operations.
 */

#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace dg::hdg::runtime::app
{

inline constexpr std::uint32_t application_checkpoint_codec_version_v1 = 1;
inline constexpr std::uint32_t application_checkpoint_max_rank_count_v1 = 2;
inline constexpr std::uint32_t euler_state_component_count_v1 = 4;
inline constexpr std::uint64_t fp64_coefficient_bytes_v1 = 8;
inline constexpr std::string_view no_physical_gpu_uuid_v1 = "none";
inline constexpr std::string_view checkpoint_partition_layout_rule_id_v1 =
    "periodic-cartesian.contiguous-element.min-adjacent-face-owner.v1";

static_assert(sizeof(double) == fp64_coefficient_bytes_v1 &&
                  std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2 &&
                  std::numeric_limits<double>::digits == 53 &&
                  std::numeric_limits<double>::max_exponent == 1024,
              "checkpoint v1 requires IEEE-754 binary64 double");

struct ConstByteView
{
  const std::uint8_t* data = nullptr;
  std::size_t size = 0;
  constexpr bool Valid() const noexcept { return size == 0 || data != nullptr; }
};

enum class CoefficientOrderingV1 : std::uint8_t
{
  component_major = 0
};

struct Fp64CoefficientViewV1
{
  const double* data = nullptr;
  std::uint64_t coefficient_count = 0;
  std::uint32_t component_count = 0;
  std::uint64_t dofs_per_component = 0;
  CoefficientOrderingV1 ordering = CoefficientOrderingV1::component_major;
};

struct StagedStateViewV1
{
  std::uint32_t rank = 0;
  Fp64CoefficientViewV1 interior{};
  Fp64CoefficientViewV1 owned_true_trace{};
};

/** Exact caller-owned decode destination; the codec never allocates it. */
struct Fp64CoefficientBufferV1
{
  double* data = nullptr;
  std::uint64_t capacity = 0;
  std::uint64_t coefficient_count = 0;
  std::uint32_t component_count = 0;
  std::uint64_t dofs_per_component = 0;
  CoefficientOrderingV1 ordering = CoefficientOrderingV1::component_major;
};

struct RankStateBufferV1
{
  std::uint32_t rank = 0;
  Fp64CoefficientBufferV1 interior{};
  Fp64CoefficientBufferV1 owned_true_trace{};
};

/**
 * Runtime identity supplied by the future backend after resource discovery.
 * It is checkpoint metadata, not application admission or execution authority.
 */
struct CheckpointRuntimeIdentityV1
{
  StableIdentity physical_gpu_uuid =
      StableIdentity::Copy(no_physical_gpu_uuid_v1);
  StateHash ownership_hash{};
};

struct RankCoefficientExtentsV1
{
  std::uint64_t interior_dofs_per_component = 0;
  std::uint64_t interior_coefficient_count = 0;
  std::uint64_t trace_dofs_per_component = 0;
  std::uint64_t trace_coefficient_count = 0;
};

struct ScenarioCoefficientExtentsV1
{
  std::uint32_t rank_count = 0;
  std::array<RankCoefficientExtentsV1,
             application_checkpoint_max_rank_count_v1> ranks{};
  std::uint64_t aggregate_interior_coefficient_count = 0;
  std::uint64_t aggregate_trace_coefficient_count = 0;
};

enum class CheckpointCodecStatus : std::uint8_t
{
  ok = 0,
  invalid_argument,
  invalid_descriptor,
  invalid_checkpoint_identity,
  invalid_config_identity,
  invalid_runtime_identity,
  unsupported_version,
  truncated,
  length_mismatch,
  invalid_magic,
  checksum_mismatch,
  invalid_rank_manifest,
  rank_tuple_mismatch,
  invalid_coefficient_schema,
  coefficient_count_mismatch,
  noncanonical_fp64,
  local_payload_hash_mismatch,
  aggregate_payload_hash_mismatch,
  restart_tuple_mismatch,
  resource_mismatch,
  output_buffer_mismatch,
  allocation_failure,
  trailing_bytes,
  size_overflow
};

template <typename Selection,
          typename InteriorState,
          typename TraceState,
          typename StoragePolicy,
          typename CollectivePolicy>
struct RankCommittedContractViewV1;

class CapturedCheckpointV1 final
{
public:
  CapturedCheckpointV1(const CapturedCheckpointV1&) = default;
  CapturedCheckpointV1(CapturedCheckpointV1&&) = default;
  CapturedCheckpointV1& operator=(const CapturedCheckpointV1&) = default;
  CapturedCheckpointV1& operator=(CapturedCheckpointV1&&) = default;

  std::uint32_t rank_count() const noexcept { return rank_count_; }
  const CheckpointDescriptor& rank_descriptor(
      const std::size_t rank) const noexcept
  {
    return rank_descriptors_[rank];
  }
  const ValidatedApplicationConfig& config() const noexcept { return config_; }
  const CheckpointRuntimeIdentityV1& runtime_identity() const noexcept
  {
    return runtime_identity_;
  }

private:
  CapturedCheckpointV1(
      std::array<CheckpointDescriptor,
                 application_checkpoint_max_rank_count_v1> rank_descriptors,
      std::uint32_t rank_count,
      ValidatedApplicationConfig config,
      CheckpointRuntimeIdentityV1 runtime_identity) noexcept
      : rank_descriptors_(rank_descriptors),
        rank_count_(rank_count),
        config_(config),
        runtime_identity_(runtime_identity)
  {}

  std::array<CheckpointDescriptor, application_checkpoint_max_rank_count_v1>
      rank_descriptors_{};
  std::uint32_t rank_count_ = 0;
  ValidatedApplicationConfig config_;
  CheckpointRuntimeIdentityV1 runtime_identity_{};

  template <typename Selection,
            typename InteriorState,
            typename TraceState,
            typename StoragePolicy,
            typename CollectivePolicy>
  friend std::optional<CapturedCheckpointV1> CaptureCheckpointV1(
      const RankCommittedContractViewV1<Selection,
                                        InteriorState,
                                        TraceState,
                                        StoragePolicy,
                                        CollectivePolicy>*,
      std::size_t,
      const ValidatedApplicationConfig&,
      const CheckpointRuntimeIdentityV1&) noexcept;
};

bool ValidCheckpointRuntimeIdentityV1(
    const ValidatedApplicationConfig& config,
    const CheckpointRuntimeIdentityV1& identity) noexcept;
bool ScenarioCheckpointTimeMatchesV1(
    double time,
    const ValidatedApplicationConfig& config) noexcept;
bool DeriveScenarioCoefficientExtentsV1(
    const ValidatedApplicationConfig& config,
    std::uint32_t rank_count,
    ScenarioCoefficientExtentsV1& extents) noexcept;

template <typename Selection,
          typename InteriorState,
          typename TraceState,
          typename StoragePolicy,
          typename CollectivePolicy>
struct RankCommittedContractViewV1
{
  std::uint32_t rank = 0;
  const PersistentRunContract<Selection,
                              InteriorState,
                              TraceState,
                              StoragePolicy,
                              CollectivePolicy>* contract = nullptr;
};

template <typename Selection,
          typename InteriorState,
          typename TraceState,
          typename StoragePolicy,
          typename CollectivePolicy>
std::optional<CapturedCheckpointV1> CaptureCheckpointV1(
    const RankCommittedContractViewV1<Selection,
                                      InteriorState,
                                      TraceState,
                                      StoragePolicy,
                                      CollectivePolicy>* rank_contracts,
    const std::size_t rank_count,
    const ValidatedApplicationConfig& config,
    const CheckpointRuntimeIdentityV1& runtime_identity) noexcept
{
  using Contract = PersistentRunContract<Selection,
                                         InteriorState,
                                         TraceState,
                                         StoragePolicy,
                                         CollectivePolicy>;
  if (!ValidCheckpointRuntimeIdentityV1(config, runtime_identity) ||
      rank_contracts == nullptr || rank_count == 0 ||
      rank_count > application_checkpoint_max_rank_count_v1 ||
      rank_count != config.deferred_runtime().required_mpi_rank_count)
  {
    return std::nullopt;
  }
  std::array<CheckpointDescriptor, application_checkpoint_max_rank_count_v1>
      descriptors{};
  for (std::size_t rank = 0; rank < rank_count; ++rank)
  {
    const Contract* contract = rank_contracts[rank].contract;
    if (rank_contracts[rank].rank != rank || contract == nullptr)
    {
      return std::nullopt;
    }
    for (std::size_t prior = 0; prior < rank; ++prior)
    {
      if (rank_contracts[prior].contract == contract) { return std::nullopt; }
    }
    const StateCoordinates coordinates = contract->coordinates();
    const RunTelemetry& telemetry = contract->telemetry();
    const auto published = contract->Published();
    StateCoordinates published_coordinates{};
    CheckpointDescriptor descriptor{};
    if (contract->status() != RunStatus::committed ||
        !telemetry.accepted.valid ||
        telemetry.commits < config.checkpoint_step() ||
        !contract->PublishedCoordinates(published, published_coordinates) ||
        !::dg::hdg::runtime::detail::SameCoordinates(
            coordinates, published_coordinates) ||
        coordinates.step != config.checkpoint_step() ||
        coordinates.generation != config.checkpoint_step() ||
        coordinates.cursor != config.checkpoint_step() ||
        coordinates.dt != config.dt() ||
        !ScenarioCheckpointTimeMatchesV1(coordinates.time, config) ||
        contract->resources().rank_count != rank_count ||
        contract->resources().polynomial_order != config.polynomial_order() ||
        contract->resources().layout != config.layout_id() ||
        !contract->TryMakeCheckpoint(descriptor) ||
        descriptor.tuple != config.tuple() ||
        descriptor.rank_count != rank_count ||
        descriptor.polynomial_order != config.polynomial_order() ||
        descriptor.coordinates.step != config.checkpoint_step() ||
        descriptor.coordinates.cursor != config.checkpoint_step() ||
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

struct DecodedCheckpointMetadataV1
{
  CheckpointDescriptor descriptor{};
  StableIdentity card_id{};
  StableIdentity case_id{};
  StableIdentity scenario_id{};
  StableIdentity checkpoint_id{};
  StableIdentity physical_gpu_uuid{};
  StableIdentity partition_layout_rule_id{};
  StateHash scientific_config_sha256{};
  StateHash run_config_sha256{};
  StateHash ownership_hash{};
  std::uint32_t mesh_n = 0;
  std::uint32_t rank_count = 0;
  std::uint64_t aggregate_interior_coefficients = 0;
  std::uint64_t aggregate_true_trace_coefficients = 0;
  StateHash artifact_hash{};
};

CheckpointCodecStatus EncodeCheckpointV1(
    const CapturedCheckpointV1& checkpoint,
    const StagedStateViewV1* ranks,
    std::size_t rank_count,
    std::vector<std::uint8_t>& encoded);

CheckpointCodecStatus DecodeCheckpointV1(
    ConstByteView encoded,
    const ValidatedApplicationConfig& expected_config,
    const PersistentResourceDescriptor& expected_resources,
    const CheckpointRuntimeIdentityV1& expected_runtime_identity,
    RankStateBufferV1* rank_buffers,
    std::size_t rank_buffer_count,
    DecodedCheckpointMetadataV1& metadata);

CheckpointCodecStatus ValidateSameTupleRestartV1(
    const CheckpointDescriptor& checkpoint,
    const ValidatedApplicationConfig& expected_config,
    const PersistentResourceDescriptor& expected_resources,
    const CheckpointRuntimeIdentityV1& expected_runtime_identity,
    const CheckpointRuntimeIdentityV1& artifact_runtime_identity) noexcept;

static_assert(!std::is_default_constructible<CapturedCheckpointV1>::value,
              "checkpoint capture must originate from a live contract");
static_assert(std::is_copy_constructible<CapturedCheckpointV1>::value,
              "committed in-memory checkpoints may be re-encoded");
static_assert(!std::is_polymorphic<CapturedCheckpointV1>::value,
              "checkpoint capture uses static dispatch");

} // namespace dg::hdg::runtime::app

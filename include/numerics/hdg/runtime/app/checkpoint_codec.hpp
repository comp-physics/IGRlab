#pragma once

/**
 * @file numerics/hdg/runtime/app/checkpoint_codec.hpp
 * @brief Explicit little-endian W8A FP64 checkpoint/manifest contract.
 *
 * Durable state is encoded coefficient-by-coefficient; no object bytes or
 * pointers are serialized.  Publication authority is affine and can begin
 * only from a live PersistentRunContract that reports a successful committed
 * checkpoint at the frozen checkpoint coordinate.
 */

#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::app
{

inline constexpr std::uint32_t application_checkpoint_codec_version_v1 = 1;
inline constexpr std::uint32_t application_checkpoint_max_rank_count_v1 = 2;
inline constexpr std::uint32_t euler_state_component_count_v1 = 4;
inline constexpr std::uint64_t fp64_coefficient_bytes_v1 = 8;

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

/** Non-owning typed view; count must equal components*dofs-per-component. */
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

enum class PublicationPhaseV1 : std::uint8_t
{
  rank_temporaries_durable = 0,
  collective_manifest_validated = 1,
  manifest_atomically_published = 2
};

enum class CheckpointCodecStatus : std::uint8_t
{
  ok = 0,
  invalid_argument,
  invalid_descriptor,
  invalid_checkpoint_identity,
  invalid_publication_state,
  unpublished_artifact,
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
  allocation_failure,
  trailing_bytes,
  size_overflow
};

struct RankTemporaryPublicationFactsV1
{
  std::uint32_t participating_ranks = 0;
  std::uint32_t source_rank = 0;
  StateHash temporary_file_set_hash{};
  StateHash previous_manifest_hash{};
  bool all_rank_temporaries_created_exclusive = false;
  bool all_rank_temporaries_synced = false;
  bool collective_temporary_consensus = false;
};

struct CollectiveManifestValidationFactsV1
{
  std::uint32_t participating_ranks = 0;
  bool descriptor_consensus = false;
  bool rank_manifest_consensus = false;
  bool payload_hash_consensus = false;
};

struct AtomicManifestPublicationFactsV1
{
  std::uint32_t source_rank = 0;
  bool manifest_created_exclusive = false;
  bool manifest_synced = false;
  bool atomic_rename_complete = false;
  bool publication_directory_synced = false;
  bool collective_visibility_consensus = false;
};

class CheckpointContractEvidenceV1 final
{
public:
  CheckpointContractEvidenceV1(const CheckpointContractEvidenceV1&) = delete;
  CheckpointContractEvidenceV1& operator=(
      const CheckpointContractEvidenceV1&) = delete;
  CheckpointContractEvidenceV1(CheckpointContractEvidenceV1&&) = default;
  CheckpointContractEvidenceV1& operator=(CheckpointContractEvidenceV1&&) =
      default;

private:
  CheckpointContractEvidenceV1(CheckpointDescriptor descriptor,
                               StableIdentity card_id,
                               StableIdentity case_id,
                               StableIdentity checkpoint_id) noexcept
      : descriptor_(descriptor),
        card_id_(card_id),
        case_id_(case_id),
        checkpoint_id_(checkpoint_id)
  {}

  CheckpointDescriptor descriptor_{};
  StableIdentity card_id_{};
  StableIdentity case_id_{};
  StableIdentity checkpoint_id_{};

  template <typename Selection,
            typename InteriorState,
            typename TraceState,
            typename StoragePolicy,
            typename CollectivePolicy>
  friend std::optional<CheckpointContractEvidenceV1>
  CaptureCheckpointContractEvidenceV1(
      const PersistentRunContract<Selection,
                                  InteriorState,
                                  TraceState,
                                  StoragePolicy,
                                  CollectivePolicy>&,
      const ValidatedApplicationConfig&) noexcept;
  friend class DurableRankTemporariesV1;
  friend struct PublicationAccessV1;
};

template <typename Selection,
          typename InteriorState,
          typename TraceState,
          typename StoragePolicy,
          typename CollectivePolicy>
std::optional<CheckpointContractEvidenceV1>
CaptureCheckpointContractEvidenceV1(
    const PersistentRunContract<Selection,
                                InteriorState,
                                TraceState,
                                StoragePolicy,
                                CollectivePolicy>& contract,
    const ValidatedApplicationConfig& config) noexcept
{
  CheckpointDescriptor descriptor{};
  const StateCoordinates coordinates = contract.coordinates();
  const RunTelemetry& telemetry = contract.telemetry();
  const typename PersistentRunContract<Selection,
                                       InteriorState,
                                       TraceState,
                                       StoragePolicy,
                                       CollectivePolicy>::PublishedCapability
      published = contract.Published();
  StateCoordinates published_coordinates{};
  if (contract.status() != RunStatus::committed ||
      !telemetry.accepted.valid ||
      telemetry.commits < smooth_periodic_euler_v1_checkpoint_step ||
      !contract.PublishedCoordinates(published, published_coordinates) ||
      !::dg::hdg::runtime::detail::SameCoordinates(
          coordinates, published_coordinates) ||
      coordinates.step != config.checkpoint_step() ||
      coordinates.generation != config.checkpoint_step() ||
      coordinates.step != smooth_periodic_euler_v1_checkpoint_step ||
      coordinates.cursor != config.checkpoint_step() ||
      coordinates.dt != config.dt() ||
      contract.resources().rank_count != config.mpi_rank_count() ||
      !contract.TryMakeCheckpoint(descriptor) ||
      descriptor.tuple != config.tuple() ||
      descriptor.polynomial_order != config.polynomial_order() ||
      descriptor.rank_count != config.mpi_rank_count() ||
      descriptor.coordinates.step != config.checkpoint_step() ||
      descriptor.coordinates.cursor != config.checkpoint_step())
  {
    return std::nullopt;
  }
  return CheckpointContractEvidenceV1(
      descriptor, config.card_id(), config.case_id(), config.checkpoint_id());
}

class DurableRankTemporariesV1 final
{
public:
  DurableRankTemporariesV1(const DurableRankTemporariesV1&) = delete;
  DurableRankTemporariesV1& operator=(const DurableRankTemporariesV1&) =
      delete;
  DurableRankTemporariesV1(DurableRankTemporariesV1&&) = default;
  DurableRankTemporariesV1& operator=(DurableRankTemporariesV1&&) = default;

private:
  DurableRankTemporariesV1(CheckpointContractEvidenceV1 evidence,
                           CheckpointDescriptor descriptor,
                           RankTemporaryPublicationFactsV1 facts,
                           std::uint64_t interior_coefficients,
                           std::uint64_t trace_coefficients) noexcept
      : evidence_(std::move(evidence)),
        descriptor_(descriptor),
        facts_(facts),
        interior_coefficients_(interior_coefficients),
        trace_coefficients_(trace_coefficients)
  {}

  CheckpointContractEvidenceV1 evidence_;
  CheckpointDescriptor descriptor_{};
  RankTemporaryPublicationFactsV1 facts_{};
  std::uint64_t interior_coefficients_ = 0;
  std::uint64_t trace_coefficients_ = 0;

  friend struct PublicationAccessV1;
};

class ValidatedCheckpointManifestV1 final
{
public:
  ValidatedCheckpointManifestV1(const ValidatedCheckpointManifestV1&) =
      delete;
  ValidatedCheckpointManifestV1& operator=(
      const ValidatedCheckpointManifestV1&) = delete;
  ValidatedCheckpointManifestV1(ValidatedCheckpointManifestV1&&) = default;
  ValidatedCheckpointManifestV1& operator=(ValidatedCheckpointManifestV1&&) =
      default;

private:
  ValidatedCheckpointManifestV1(
      DurableRankTemporariesV1 temporaries,
      CollectiveManifestValidationFactsV1 facts) noexcept
      : temporaries_(std::move(temporaries)), facts_(facts)
  {}

  DurableRankTemporariesV1 temporaries_;
  CollectiveManifestValidationFactsV1 facts_{};

  friend struct PublicationAccessV1;
};

class PublishedCheckpointManifestV1 final
{
public:
  PublishedCheckpointManifestV1(const PublishedCheckpointManifestV1&) =
      delete;
  PublishedCheckpointManifestV1& operator=(
      const PublishedCheckpointManifestV1&) = delete;
  PublishedCheckpointManifestV1(PublishedCheckpointManifestV1&&) = default;
  PublishedCheckpointManifestV1& operator=(PublishedCheckpointManifestV1&&) =
      default;

private:
  PublishedCheckpointManifestV1(
      ValidatedCheckpointManifestV1 validated,
      AtomicManifestPublicationFactsV1 facts) noexcept
      : validated_(std::move(validated)), facts_(facts)
  {}

  ValidatedCheckpointManifestV1 validated_;
  AtomicManifestPublicationFactsV1 facts_{};

  friend struct PublicationAccessV1;
};

template <typename Authority>
struct PublicationTransitionV1
{
  CheckpointCodecStatus status = CheckpointCodecStatus::invalid_argument;
  std::optional<Authority> authority{};
};

PublicationTransitionV1<DurableRankTemporariesV1>
RecordDurableRankTemporariesV1(
    CheckpointContractEvidenceV1&& evidence,
    const RankTemporaryPublicationFactsV1& facts,
    const StagedStateViewV1* ranks,
    std::size_t rank_count);

PublicationTransitionV1<ValidatedCheckpointManifestV1>
ValidateCollectiveCheckpointManifestV1(
    DurableRankTemporariesV1&& temporaries,
    const CollectiveManifestValidationFactsV1& facts) noexcept;

PublicationTransitionV1<PublishedCheckpointManifestV1>
PublishCheckpointManifestV1(
    ValidatedCheckpointManifestV1&& validated,
    const AtomicManifestPublicationFactsV1& facts) noexcept;

struct PublicationAuditV1
{
  PublicationPhaseV1 phase = PublicationPhaseV1::rank_temporaries_durable;
  std::uint64_t transaction_id = 0;
  StateHash temporary_file_set_hash{};
  StateHash previous_manifest_hash{};
  std::uint32_t participating_ranks = 0;
  std::uint32_t source_rank = 0;
};

struct DecodedCoefficientBlockV1
{
  std::uint32_t component_count = 0;
  std::uint64_t dofs_per_component = 0;
  CoefficientOrderingV1 ordering = CoefficientOrderingV1::component_major;
  StateHash payload_hash{};
  std::vector<double> coefficients{};
};

struct DecodedRankStateV1
{
  std::uint32_t rank = 0;
  DecodedCoefficientBlockV1 interior{};
  DecodedCoefficientBlockV1 owned_true_trace{};
};

struct DecodedCheckpointV1
{
  CheckpointDescriptor descriptor{};
  StableIdentity card_id{};
  StableIdentity case_id{};
  StableIdentity checkpoint_id{};
  PublicationAuditV1 publication{};
  std::vector<DecodedRankStateV1> ranks{};
  std::uint64_t aggregate_interior_coefficients = 0;
  std::uint64_t aggregate_true_trace_coefficients = 0;
  StateHash artifact_hash{};
};

CheckpointCodecStatus EncodeCheckpointV1(
    const PublishedCheckpointManifestV1& publication,
    const StagedStateViewV1* ranks,
    std::size_t rank_count,
    std::vector<std::uint8_t>& encoded);

CheckpointCodecStatus DecodeCheckpointV1(
    ConstByteView encoded,
    const ValidatedApplicationConfig& expected_config,
    const PersistentResourceDescriptor& expected_resources,
    DecodedCheckpointV1& checkpoint);

CheckpointCodecStatus ValidateSameTupleRestartV1(
    const CheckpointDescriptor& checkpoint,
    const ValidatedApplicationConfig& expected_config,
    const PersistentResourceDescriptor& expected_resources) noexcept;

static_assert(!std::is_default_constructible<CheckpointContractEvidenceV1>::value,
              "contract checkpoint evidence cannot be fabricated");
static_assert(!std::is_default_constructible<PublishedCheckpointManifestV1>::value,
              "published authority cannot be fabricated");
static_assert(!std::is_copy_constructible<PublishedCheckpointManifestV1>::value,
              "published authority is affine");
static_assert(!std::is_polymorphic<PublishedCheckpointManifestV1>::value,
              "publication protocol uses static dispatch");

} // namespace dg::hdg::runtime::app

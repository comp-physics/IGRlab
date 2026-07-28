#pragma once

/**
 * @file numerics/hdg/runtime/app/application_contract.hpp
 * @brief Frozen host-side W8A application selection contract.
 *
 * Validation is completed before backend dispatch.  Future entry definitions
 * must populate ObservedRuntimeFactsView from the live MPI/CUDA runtime and
 * drive the existing PersistentRunContract; this contract defines no second
 * backend lifecycle.
 */

#include "numerics/hdg/runtime/runtime_abi.hpp"
#include "numerics/hdg/runtime/runtime_selection.hpp"

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace dg::hdg::runtime::app
{

class ApplicationValidation;

inline constexpr std::string_view smooth_periodic_euler_v1_case_id =
    conventional_persistent_case_id;
inline constexpr std::string_view smooth_periodic_euler_v1_card_id =
    conventional_persistent_card_id;
inline constexpr std::string_view smooth_periodic_euler_v1_checkpoint_id =
    "checkpoint.hdg.euler.smooth-periodic.step-10.v1";

inline constexpr int smooth_periodic_euler_v1_polynomial_order = 1;
inline constexpr double smooth_periodic_euler_v1_dt = 0.05;
inline constexpr std::uint64_t smooth_periodic_euler_v1_minimum_steps = 20;
inline constexpr std::array<std::uint64_t, 3>
    smooth_periodic_euler_v1_output_steps{{0, 10, 20}};
inline constexpr std::uint64_t smooth_periodic_euler_v1_checkpoint_step = 10;
inline constexpr std::uint64_t smooth_periodic_euler_v1_failure_step = 14;
inline constexpr FailurePhase smooth_periodic_euler_v1_failure_phase =
    FailurePhase::AfterRecoveryBeforeCommit;
inline constexpr std::uint32_t smooth_periodic_euler_v1_retry_limit = 1;
inline constexpr std::uint32_t smooth_periodic_euler_v1_source_rank = 0;

enum class ApplicationBackend : std::uint8_t
{
  cpu_serial = 0,
  cpu_rank2 = 1,
  cuda_rank1_one_gpu = 2,
  cuda_rank2_one_gpu_host_staged = 3,
  unknown = 255
};

enum class RollbackObligationV1 : std::uint8_t
{
  collective_exact_complete_state = 0
};

enum class RestartTrajectoryModeV1 : std::uint8_t
{
  uninterrupted_and_split_same_tuple = 0
};

enum class CheckpointSourceRankRuleV1 : std::uint8_t
{
  rank_zero_publishes_manifest = 0
};

enum class ApplicationRejection : std::uint8_t
{
  none = 0,
  unknown_card,
  unknown_case,
  unknown_tuple,
  unsupported_h1,
  invalid_interior_space,
  unknown_backend,
  unbuilt_backend,
  invalid_mpi_facts,
  invalid_gpu_facts,
  invalid_polynomial_order,
  invalid_mesh,
  insufficient_step_bound,
  invalid_dt,
  invalid_output_schedule,
  invalid_checkpoint_schedule,
  invalid_checkpoint_identity,
  invalid_restart_mode,
  invalid_restart_trajectory,
  invalid_failure_injection,
  invalid_failure_phase,
  invalid_rollback_obligation,
  invalid_source_rank_rule,
  invalid_retry_limit
};

/** Facts observed from MPI/CUDA before dispatch; configuration cannot invent them. */
struct ObservedRuntimeFactsView
{
  bool mpi_initialized = true;
  std::uint32_t mpi_rank_count = 1;
  std::uint32_t local_mpi_rank = 0;
  std::uint32_t participating_gpu_count = 0;
  std::int32_t selected_gpu_ordinal = -1;
  std::string_view selected_gpu_identity{};
  bool gpu_identity_consistent_across_ranks = true;
  bool all_ranks_share_one_gpu = false;
};

/** Non-owning command/configuration view consumed only during validation. */
struct ApplicationOptionsView
{
  std::string_view card_id = smooth_periodic_euler_v1_card_id;
  std::string_view case_id = smooth_periodic_euler_v1_case_id;
  std::string_view model_id = EulerCompressible2dTag::id;
  std::string_view formulation_id = ConventionalHdgTag::id;
  std::string_view trace_id = MfemDgInterfaceTraceTag::id;
  std::string_view stabilization_id = FixedScalarTauTag::id;
  std::string_view time_treatment_id = BackwardEulerRuntimeTag::id;
  std::string_view backend_id = CpuSerialBackendTag::id;
  std::string_view precision_id = Fp64RuntimeTag::id;
  InteriorSpace interior_space = InteriorSpace::l2;
  int polynomial_order = smooth_periodic_euler_v1_polynomial_order;
  std::uint32_t mesh_n = 6;
  std::uint64_t step_bound = smooth_periodic_euler_v1_minimum_steps;
  double dt = smooth_periodic_euler_v1_dt;
  std::array<std::uint64_t, 3> output_steps =
      smooth_periodic_euler_v1_output_steps;
  std::uint64_t checkpoint_step =
      smooth_periodic_euler_v1_checkpoint_step;
  std::string_view checkpoint_id =
      smooth_periodic_euler_v1_checkpoint_id;
  RestartMode restart_mode = RestartMode::same_backend_rank_partition;
  RestartTrajectoryModeV1 restart_trajectory =
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple;
  std::uint64_t injected_failure_step =
      smooth_periodic_euler_v1_failure_step;
  FailurePhase injected_failure_phase =
      smooth_periodic_euler_v1_failure_phase;
  RollbackObligationV1 rollback_obligation =
      RollbackObligationV1::collective_exact_complete_state;
  CheckpointSourceRankRuleV1 source_rank_rule =
      CheckpointSourceRankRuleV1::rank_zero_publishes_manifest;
  std::uint32_t checkpoint_source_rank =
      smooth_periodic_euler_v1_source_rank;
  std::uint32_t retry_limit = smooth_periodic_euler_v1_retry_limit;
  ObservedRuntimeFactsView observed{};
};

class ValidatedApplicationConfig final
{
public:
  ValidatedApplicationConfig(const ValidatedApplicationConfig&) = default;
  ValidatedApplicationConfig(ValidatedApplicationConfig&&) = default;
  ValidatedApplicationConfig& operator=(const ValidatedApplicationConfig&) =
      default;
  ValidatedApplicationConfig& operator=(ValidatedApplicationConfig&&) =
      default;

  ApplicationBackend backend() const noexcept { return backend_; }
  const RuntimeTupleDescriptor& tuple() const noexcept { return tuple_; }
  InteriorSpace interior_space() const noexcept { return interior_space_; }
  int polynomial_order() const noexcept { return polynomial_order_; }
  std::uint32_t mesh_n() const noexcept { return mesh_n_; }
  std::uint64_t step_bound() const noexcept { return step_bound_; }
  double dt() const noexcept { return dt_; }
  const std::array<std::uint64_t, 3>& output_steps() const noexcept
  {
    return output_steps_;
  }
  std::uint64_t checkpoint_step() const noexcept { return checkpoint_step_; }
  const StableIdentity& card_id() const noexcept { return card_id_; }
  const StableIdentity& case_id() const noexcept { return case_id_; }
  const StableIdentity& checkpoint_id() const noexcept
  {
    return checkpoint_id_;
  }
  RestartMode restart_mode() const noexcept { return restart_mode_; }
  RestartTrajectoryModeV1 restart_trajectory() const noexcept
  {
    return restart_trajectory_;
  }
  std::uint64_t injected_failure_step() const noexcept
  {
    return injected_failure_step_;
  }
  FailurePhase injected_failure_phase() const noexcept
  {
    return injected_failure_phase_;
  }
  RollbackObligationV1 rollback_obligation() const noexcept
  {
    return rollback_obligation_;
  }
  CheckpointSourceRankRuleV1 source_rank_rule() const noexcept
  {
    return source_rank_rule_;
  }
  std::uint32_t checkpoint_source_rank() const noexcept
  {
    return checkpoint_source_rank_;
  }
  std::uint32_t retry_limit() const noexcept { return retry_limit_; }
  std::uint32_t mpi_rank_count() const noexcept { return mpi_rank_count_; }
  std::uint32_t local_mpi_rank() const noexcept { return local_mpi_rank_; }
  std::uint32_t participating_gpu_count() const noexcept
  {
    return participating_gpu_count_;
  }
  std::int32_t selected_gpu_ordinal() const noexcept
  {
    return selected_gpu_ordinal_;
  }
  const StableIdentity& selected_gpu_identity() const noexcept
  {
    return selected_gpu_identity_;
  }
  bool gpu_identity_consistent_across_ranks() const noexcept
  {
    return gpu_identity_consistent_across_ranks_;
  }
  bool all_ranks_share_one_gpu() const noexcept
  {
    return all_ranks_share_one_gpu_;
  }

private:
  ValidatedApplicationConfig() noexcept = default;
  ValidatedApplicationConfig(const ApplicationOptionsView& options,
                             ApplicationBackend backend,
                             RuntimeTupleDescriptor tuple) noexcept;

  ApplicationBackend backend_ = ApplicationBackend::unknown;
  RuntimeTupleDescriptor tuple_{};
  InteriorSpace interior_space_ = InteriorSpace::l2;
  int polynomial_order_ = -1;
  std::uint32_t mesh_n_ = 0;
  std::uint64_t step_bound_ = 0;
  double dt_ = 0.0;
  std::array<std::uint64_t, 3> output_steps_{};
  std::uint64_t checkpoint_step_ = 0;
  StableIdentity card_id_{};
  StableIdentity case_id_{};
  StableIdentity checkpoint_id_{};
  RestartMode restart_mode_ = RestartMode::same_backend_rank_partition;
  RestartTrajectoryModeV1 restart_trajectory_ =
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple;
  std::uint64_t injected_failure_step_ = 0;
  FailurePhase injected_failure_phase_ = FailurePhase::Setup;
  RollbackObligationV1 rollback_obligation_ =
      RollbackObligationV1::collective_exact_complete_state;
  CheckpointSourceRankRuleV1 source_rank_rule_ =
      CheckpointSourceRankRuleV1::rank_zero_publishes_manifest;
  std::uint32_t checkpoint_source_rank_ = 0;
  std::uint32_t retry_limit_ = 0;
  std::uint32_t mpi_rank_count_ = 0;
  std::uint32_t local_mpi_rank_ = 0;
  std::uint32_t participating_gpu_count_ = 0;
  std::int32_t selected_gpu_ordinal_ = -1;
  StableIdentity selected_gpu_identity_{};
  bool gpu_identity_consistent_across_ranks_ = false;
  bool all_ranks_share_one_gpu_ = false;

  friend class ApplicationValidation;
  friend ApplicationValidation ValidateApplicationOptions(
      const ApplicationOptionsView&) noexcept;
};

class ApplicationValidation final
{
public:
  bool accepted() const noexcept { return accepted_; }
  ApplicationRejection rejection() const noexcept { return rejection_; }
  const ValidatedApplicationConfig& config() const noexcept { return config_; }

private:
  ApplicationValidation(bool accepted,
                        ApplicationRejection rejection,
                        ValidatedApplicationConfig config) noexcept
      : accepted_(accepted), rejection_(rejection), config_(config)
  {}

  bool accepted_ = false;
  ApplicationRejection rejection_ = ApplicationRejection::unknown_tuple;
  ValidatedApplicationConfig config_{};

  friend ApplicationValidation ValidateApplicationOptions(
      const ApplicationOptionsView&) noexcept;
};

ApplicationBackend ParseApplicationBackend(std::string_view id) noexcept;
bool CompiledBackendAvailable(ApplicationBackend backend) noexcept;
bool IsAdmittedMesh(ApplicationBackend backend,
                    std::uint32_t mesh_n) noexcept;

constexpr bool IsSharedFourShapeMesh(const std::uint32_t mesh_n) noexcept
{
  return mesh_n == 6;
}

ApplicationValidation ValidateApplicationOptions(
    const ApplicationOptionsView& options) noexcept;

struct ApplicationRunResult
{
  RunStatus status = RunStatus::invalid_request;
  FailureReason reason = FailureReason::invalid_resource_descriptor;
  StateCoordinates coordinates{};
  std::uint32_t retries = 0;
};

template <typename BackendTag>
struct IsFrozenApplicationBackend : std::false_type
{};
template <>
struct IsFrozenApplicationBackend<CpuSerialBackendTag> : std::true_type
{};
template <>
struct IsFrozenApplicationBackend<CpuRank2BackendTag> : std::true_type
{};
template <>
struct IsFrozenApplicationBackend<CudaRank1OneGpuBackendTag>
    : std::true_type
{};
template <>
struct IsFrozenApplicationBackend<CudaRank2OneGpuHostStagedBackendTag>
    : std::true_type
{};

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuSerialBackendTag,
    const ValidatedApplicationConfig& config);
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuRank2BackendTag,
    const ValidatedApplicationConfig& config);
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank1OneGpuBackendTag,
    const ValidatedApplicationConfig& config);
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank2OneGpuHostStagedBackendTag,
    const ValidatedApplicationConfig& config);

static_assert(!std::is_default_constructible<ValidatedApplicationConfig>::value,
              "validated configuration cannot bypass validation");
static_assert(!std::is_aggregate<ValidatedApplicationConfig>::value,
              "validated configuration is opaque");
static_assert(!std::is_polymorphic<ValidatedApplicationConfig>::value,
              "application configuration remains statically dispatched");

} // namespace dg::hdg::runtime::app

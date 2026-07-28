#pragma once

/**
 * @file numerics/hdg/runtime/app/application_contract.hpp
 * @brief Static request admission for the two frozen W8A scenarios.
 *
 * This boundary validates syntax, exact card/scenario values, and compiled
 * backend selection only.  It does not accept caller-authored MPI or device
 * observations and grants no execution authority.  The later backend entry
 * owns MPI_Comm_size and CUDA UUID/shared-device observation before execution.
 */

#include "numerics/hdg/runtime/runtime_abi.hpp"
#include "numerics/hdg/runtime/runtime_selection.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>

namespace dg::hdg::runtime::app
{

class ApplicationValidation;

inline constexpr std::string_view smooth_periodic_euler_v1_card_id =
    conventional_persistent_card_id;
inline constexpr std::string_view smooth_periodic_euler_v1_case_id =
    conventional_persistent_case_id;
inline constexpr std::string_view smooth_periodic_euler_v1_mesh_family_id =
    "periodic-cartesian-quad";
inline constexpr std::string_view smooth_periodic_euler_v1_source_id =
    "source.euler.smooth.analytic.v1";
inline constexpr std::string_view smooth_periodic_euler_v1_flux_id =
    "flux.hdg.conventional.fixed-scalar-tau.v1";
inline constexpr std::string_view smooth_periodic_euler_v1_time_rule_id =
    BackwardEulerRuntimeTag::id;
inline constexpr std::string_view smooth_periodic_euler_v1_layout_id =
    "l2-interior.true-trace-owned.component-major.v1";

inline constexpr std::string_view smooth_periodic_euler_v1_smoke_id =
    "W8A-SMOKE";
inline constexpr std::string_view smooth_periodic_euler_v1_functional_id =
    "W8A-FUNCTIONAL";
inline constexpr std::string_view smooth_periodic_euler_v1_smoke_checkpoint_id =
    "checkpoint.hdg.euler.smooth-periodic.smoke.step-1.v1";
inline constexpr std::string_view
    smooth_periodic_euler_v1_functional_checkpoint_id =
        "checkpoint.hdg.euler.smooth-periodic.functional.step-10.v1";

inline constexpr double smooth_periodic_euler_v1_gamma = 1.4;
inline constexpr double smooth_periodic_euler_v1_tau = 2.5;
inline constexpr double smooth_periodic_euler_v1_dt = 0.05;

inline constexpr int smooth_periodic_euler_v1_smoke_polynomial_order = 1;
inline constexpr std::uint32_t smooth_periodic_euler_v1_smoke_mesh_n = 3;
inline constexpr int smooth_periodic_euler_v1_smoke_q_a = 11;
inline constexpr int smooth_periodic_euler_v1_smoke_q_e = 16;
inline constexpr int smooth_periodic_euler_v1_smoke_q_e4 = 20;
inline constexpr int smooth_periodic_euler_v1_smoke_q_o = 21;
inline constexpr std::uint64_t smooth_periodic_euler_v1_smoke_steps = 3;
inline constexpr double smooth_periodic_euler_v1_smoke_final_time = 0.15;
inline constexpr std::array<std::uint64_t, 3>
    smooth_periodic_euler_v1_smoke_output_steps{{0, 1, 3}};
inline constexpr std::uint64_t
    smooth_periodic_euler_v1_smoke_checkpoint_step = 1;
inline constexpr std::uint64_t
    smooth_periodic_euler_v1_smoke_failure_committed_step = 1;
inline constexpr std::uint64_t
    smooth_periodic_euler_v1_smoke_failure_target_step = 2;
inline constexpr std::uint64_t smooth_periodic_euler_v1_smoke_seed =
    UINT64_C(0x573841534d4f4b31);

inline constexpr int smooth_periodic_euler_v1_functional_polynomial_order = 2;
inline constexpr std::uint32_t smooth_periodic_euler_v1_functional_mesh_n = 6;
inline constexpr int smooth_periodic_euler_v1_functional_q_a = 14;
inline constexpr int smooth_periodic_euler_v1_functional_q_e = 20;
inline constexpr int smooth_periodic_euler_v1_functional_q_e4 = 24;
inline constexpr int smooth_periodic_euler_v1_functional_q_o = 26;
inline constexpr std::uint64_t smooth_periodic_euler_v1_functional_steps = 20;
inline constexpr double smooth_periodic_euler_v1_functional_final_time = 1.0;
inline constexpr std::array<std::uint64_t, 3>
    smooth_periodic_euler_v1_functional_output_steps{{0, 10, 20}};
inline constexpr std::uint64_t
    smooth_periodic_euler_v1_functional_checkpoint_step = 10;
inline constexpr std::uint64_t
    smooth_periodic_euler_v1_functional_failure_committed_step = 13;
inline constexpr std::uint64_t
    smooth_periodic_euler_v1_functional_failure_target_step = 14;
inline constexpr std::uint64_t smooth_periodic_euler_v1_functional_seed =
    UINT64_C(0x57384146554e4331);

inline constexpr FailurePhase smooth_periodic_euler_v1_failure_phase =
    FailurePhase::AfterRecoveryBeforeCommit;
inline constexpr std::uint32_t smooth_periodic_euler_v1_retry_limit = 1;

enum class ApplicationScenarioV1 : std::uint8_t
{
  smoke = 0,
  functional = 1,
  unknown = 255
};

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

enum class DeferredSourceRankRuleV1 : std::uint8_t
{
  rank_zero_for_rank1_rank_one_for_rank2 = 0
};

enum class DeferredGpuRuleV1 : std::uint8_t
{
  no_cuda_device = 0,
  one_physical_gpu = 1,
  one_shared_physical_gpu = 2
};

struct DeferredRuntimePredicatesV1
{
  std::uint32_t required_mpi_rank_count = 0;
  DeferredGpuRuleV1 gpu_rule = DeferredGpuRuleV1::no_cuda_device;
  DeferredSourceRankRuleV1 source_rank_rule =
      DeferredSourceRankRuleV1::rank_zero_for_rank1_rank_one_for_rank2;
  std::uint32_t required_injected_source_rank = 0;
};

enum class ApplicationRejection : std::uint8_t
{
  none = 0,
  unknown_card,
  unknown_case,
  unknown_scenario,
  unknown_tuple,
  unsupported_h1,
  invalid_interior_space,
  unknown_backend,
  unbuilt_backend,
  invalid_scientific_identity,
  invalid_config_hash,
  invalid_polynomial_mesh_pair,
  invalid_quadrature,
  invalid_dt,
  invalid_step_count,
  invalid_final_time,
  invalid_output_schedule,
  invalid_checkpoint_schedule,
  invalid_checkpoint_identity,
  invalid_restart_mode,
  invalid_restart_trajectory,
  invalid_failure_injection,
  invalid_failure_phase,
  invalid_rollback_obligation,
  invalid_source_rank_rule,
  invalid_retry_limit,
  invalid_seed
};

/** Non-owning CLI/config view; runtime resource observations are absent. */
struct ApplicationOptionsView
{
  std::string_view card_id = smooth_periodic_euler_v1_card_id;
  std::string_view case_id = smooth_periodic_euler_v1_case_id;
  std::string_view scenario_id = smooth_periodic_euler_v1_smoke_id;
  std::string_view model_id = EulerCompressible2dTag::id;
  std::string_view formulation_id = ConventionalHdgTag::id;
  std::string_view trace_id = MfemDgInterfaceTraceTag::id;
  std::string_view stabilization_id = FixedScalarTauTag::id;
  std::string_view time_treatment_id = BackwardEulerRuntimeTag::id;
  std::string_view backend_id = CpuSerialBackendTag::id;
  std::string_view precision_id = Fp64RuntimeTag::id;
  std::string_view mesh_family_id = smooth_periodic_euler_v1_mesh_family_id;
  std::string_view source_id = smooth_periodic_euler_v1_source_id;
  std::string_view flux_id = smooth_periodic_euler_v1_flux_id;
  std::string_view time_rule_id = smooth_periodic_euler_v1_time_rule_id;
  std::string_view layout_id = smooth_periodic_euler_v1_layout_id;
  InteriorSpace interior_space = InteriorSpace::l2;
  int polynomial_order = smooth_periodic_euler_v1_smoke_polynomial_order;
  std::uint32_t mesh_n = smooth_periodic_euler_v1_smoke_mesh_n;
  int q_a = smooth_periodic_euler_v1_smoke_q_a;
  int q_e = smooth_periodic_euler_v1_smoke_q_e;
  int q_e4 = smooth_periodic_euler_v1_smoke_q_e4;
  int q_o = smooth_periodic_euler_v1_smoke_q_o;
  double gamma = smooth_periodic_euler_v1_gamma;
  double tau = smooth_periodic_euler_v1_tau;
  double dt = smooth_periodic_euler_v1_dt;
  std::uint64_t accepted_steps = smooth_periodic_euler_v1_smoke_steps;
  double final_time = smooth_periodic_euler_v1_smoke_final_time;
  std::array<std::uint64_t, 3> output_steps =
      smooth_periodic_euler_v1_smoke_output_steps;
  std::uint64_t checkpoint_step =
      smooth_periodic_euler_v1_smoke_checkpoint_step;
  std::string_view checkpoint_id =
      smooth_periodic_euler_v1_smoke_checkpoint_id;
  RestartMode restart_mode = RestartMode::same_backend_rank_partition;
  RestartTrajectoryModeV1 restart_trajectory =
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple;
  std::uint64_t injected_failure_committed_step =
      smooth_periodic_euler_v1_smoke_failure_committed_step;
  std::uint64_t injected_failure_target_step =
      smooth_periodic_euler_v1_smoke_failure_target_step;
  FailurePhase injected_failure_phase = smooth_periodic_euler_v1_failure_phase;
  RollbackObligationV1 rollback_obligation =
      RollbackObligationV1::collective_exact_complete_state;
  DeferredSourceRankRuleV1 source_rank_rule =
      DeferredSourceRankRuleV1::rank_zero_for_rank1_rank_one_for_rank2;
  std::uint32_t retry_limit = smooth_periodic_euler_v1_retry_limit;
  std::uint64_t seed = smooth_periodic_euler_v1_smoke_seed;
  StateHash scientific_config_sha256{};
  StateHash run_config_sha256{};
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

  ApplicationScenarioV1 scenario() const noexcept { return scenario_; }
  ApplicationBackend backend() const noexcept { return backend_; }
  const RuntimeTupleDescriptor& tuple() const noexcept { return tuple_; }
  const StableIdentity& card_id() const noexcept { return card_id_; }
  const StableIdentity& case_id() const noexcept { return case_id_; }
  const StableIdentity& scenario_id() const noexcept { return scenario_id_; }
  const StableIdentity& checkpoint_id() const noexcept
  {
    return checkpoint_id_;
  }
  const StableIdentity& mesh_family_id() const noexcept
  {
    return mesh_family_id_;
  }
  const StableIdentity& source_id() const noexcept { return source_id_; }
  const StableIdentity& flux_id() const noexcept { return flux_id_; }
  const StableIdentity& time_rule_id() const noexcept { return time_rule_id_; }
  const StableIdentity& layout_id() const noexcept { return layout_id_; }
  InteriorSpace interior_space() const noexcept { return interior_space_; }
  int polynomial_order() const noexcept { return polynomial_order_; }
  std::uint32_t mesh_n() const noexcept { return mesh_n_; }
  int q_a() const noexcept { return q_a_; }
  int q_e() const noexcept { return q_e_; }
  int q_e4() const noexcept { return q_e4_; }
  int q_o() const noexcept { return q_o_; }
  double gamma() const noexcept { return gamma_; }
  double tau() const noexcept { return tau_; }
  double dt() const noexcept { return dt_; }
  std::uint64_t accepted_steps() const noexcept { return accepted_steps_; }
  double final_time() const noexcept { return final_time_; }
  const std::array<std::uint64_t, 3>& output_steps() const noexcept
  {
    return output_steps_;
  }
  std::uint64_t checkpoint_step() const noexcept { return checkpoint_step_; }
  RestartMode restart_mode() const noexcept { return restart_mode_; }
  RestartTrajectoryModeV1 restart_trajectory() const noexcept
  {
    return restart_trajectory_;
  }
  std::uint64_t injected_failure_committed_step() const noexcept
  {
    return injected_failure_committed_step_;
  }
  std::uint64_t injected_failure_target_step() const noexcept
  {
    return injected_failure_target_step_;
  }
  FailurePhase injected_failure_phase() const noexcept
  {
    return injected_failure_phase_;
  }
  RollbackObligationV1 rollback_obligation() const noexcept
  {
    return rollback_obligation_;
  }
  DeferredSourceRankRuleV1 source_rank_rule() const noexcept
  {
    return source_rank_rule_;
  }
  std::uint32_t retry_limit() const noexcept { return retry_limit_; }
  std::uint64_t seed() const noexcept { return seed_; }
  const StateHash& scientific_config_sha256() const noexcept
  {
    return scientific_config_sha256_;
  }
  const StateHash& run_config_sha256() const noexcept
  {
    return run_config_sha256_;
  }
  const DeferredRuntimePredicatesV1& deferred_runtime() const noexcept
  {
    return deferred_runtime_;
  }

private:
  ValidatedApplicationConfig(const ApplicationOptionsView& options,
                             ApplicationScenarioV1 scenario,
                             ApplicationBackend backend,
                             RuntimeTupleDescriptor tuple,
                             DeferredRuntimePredicatesV1 deferred) noexcept;

  ApplicationScenarioV1 scenario_ = ApplicationScenarioV1::unknown;
  ApplicationBackend backend_ = ApplicationBackend::unknown;
  RuntimeTupleDescriptor tuple_{};
  StableIdentity card_id_{};
  StableIdentity case_id_{};
  StableIdentity scenario_id_{};
  StableIdentity checkpoint_id_{};
  StableIdentity mesh_family_id_{};
  StableIdentity source_id_{};
  StableIdentity flux_id_{};
  StableIdentity time_rule_id_{};
  StableIdentity layout_id_{};
  InteriorSpace interior_space_ = InteriorSpace::l2;
  int polynomial_order_ = -1;
  std::uint32_t mesh_n_ = 0;
  int q_a_ = 0;
  int q_e_ = 0;
  int q_e4_ = 0;
  int q_o_ = 0;
  double gamma_ = 0.0;
  double tau_ = 0.0;
  double dt_ = 0.0;
  std::uint64_t accepted_steps_ = 0;
  double final_time_ = 0.0;
  std::array<std::uint64_t, 3> output_steps_{};
  std::uint64_t checkpoint_step_ = 0;
  RestartMode restart_mode_ = RestartMode::same_backend_rank_partition;
  RestartTrajectoryModeV1 restart_trajectory_ =
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple;
  std::uint64_t injected_failure_committed_step_ = 0;
  std::uint64_t injected_failure_target_step_ = 0;
  FailurePhase injected_failure_phase_ = FailurePhase::Setup;
  RollbackObligationV1 rollback_obligation_ =
      RollbackObligationV1::collective_exact_complete_state;
  DeferredSourceRankRuleV1 source_rank_rule_ =
      DeferredSourceRankRuleV1::rank_zero_for_rank1_rank_one_for_rank2;
  std::uint32_t retry_limit_ = 0;
  std::uint64_t seed_ = 0;
  StateHash scientific_config_sha256_{};
  StateHash run_config_sha256_{};
  DeferredRuntimePredicatesV1 deferred_runtime_{};

  friend class ApplicationValidation;
  friend ApplicationValidation ValidateApplicationOptions(
      const ApplicationOptionsView&) noexcept;
};

class ApplicationValidation final
{
public:
  bool accepted() const noexcept { return config_.has_value(); }
  ApplicationRejection rejection() const noexcept { return rejection_; }
  const ValidatedApplicationConfig* config() const noexcept
  {
    return config_ ? &*config_ : nullptr;
  }

private:
  explicit ApplicationValidation(ApplicationRejection rejection) noexcept
      : rejection_(rejection)
  {}
  explicit ApplicationValidation(ValidatedApplicationConfig config) noexcept
      : rejection_(ApplicationRejection::none), config_(config)
  {}

  ApplicationRejection rejection_ = ApplicationRejection::unknown_scenario;
  std::optional<ValidatedApplicationConfig> config_{};

  friend ApplicationValidation ValidateApplicationOptions(
      const ApplicationOptionsView&) noexcept;
};

ApplicationScenarioV1 ParseApplicationScenario(std::string_view id) noexcept;
ApplicationBackend ParseApplicationBackend(std::string_view id) noexcept;
bool CompiledBackendAvailable(ApplicationBackend backend) noexcept;
ApplicationValidation ValidateApplicationOptions(
    const ApplicationOptionsView& options) noexcept;

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

struct ApplicationRunResult
{
  RunStatus status = RunStatus::invalid_request;
  FailureReason reason = FailureReason::invalid_resource_descriptor;
  StateCoordinates coordinates{};
  std::uint32_t retries = 0;
};

// Opaque contexts are constructed only by the later single-owner executable.
// Entry implementations must derive MPI_Comm_size themselves.  CUDA entries
// must additionally query the CUDA UUID and shared-one-GPU fact in .cu code.
class MpiApplicationContextV1;
class CudaApplicationContextV1;

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuSerialBackendTag,
    const ApplicationOptionsView& request,
    MpiApplicationContextV1& context);
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuRank2BackendTag,
    const ApplicationOptionsView& request,
    MpiApplicationContextV1& context);
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank1OneGpuBackendTag,
    const ApplicationOptionsView& request,
    CudaApplicationContextV1& context);
ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank2OneGpuHostStagedBackendTag,
    const ApplicationOptionsView& request,
    CudaApplicationContextV1& context);

static_assert(!std::is_default_constructible<ValidatedApplicationConfig>::value,
              "accepted configuration cannot bypass static validation");
static_assert(!std::is_aggregate<ValidatedApplicationConfig>::value,
              "accepted configuration is opaque");
static_assert(!std::is_polymorphic<ValidatedApplicationConfig>::value,
              "static admission remains non-polymorphic");

} // namespace dg::hdg::runtime::app

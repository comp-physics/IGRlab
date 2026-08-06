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

inline constexpr std::string_view scalar_igr_sc25_v3_card_id =
    scalar_igr_persistent_card_id;
inline constexpr std::string_view scalar_igr_sc25_v3_case_id =
    scalar_igr_persistent_case_id;
inline constexpr std::string_view scalar_igr_sc25_v3_scenario_id =
    "W8B-SC25-N8-K1-20STEP-v1";
inline constexpr std::string_view scalar_igr_sc25_v3_source_id =
    "source.sc25.periodic.smooth.v3";
inline constexpr std::string_view scalar_igr_sc25_v3_flux_id =
    "flux.hdg.igr.traced-scalar.sc25.v3";
inline constexpr std::string_view scalar_igr_sc25_v3_layout_id =
    "mfem.l2.byNODES.U4.Sigma1.z2.separate.v3";
inline constexpr std::string_view scalar_igr_sc25_v3_checkpoint_id =
    "checkpoint.hdg.igr.sc25.periodic.step-10.v3";
inline constexpr int scalar_igr_sc25_v3_polynomial_order = 1;
inline constexpr std::uint32_t scalar_igr_sc25_v3_mesh_n = 8;
inline constexpr int scalar_igr_sc25_v3_q_a = 12;
inline constexpr int scalar_igr_sc25_v3_q_e = 14;
inline constexpr int scalar_igr_sc25_v3_q_e4 = 16;
inline constexpr int scalar_igr_sc25_v3_q_o = 18;
inline constexpr double scalar_igr_sc25_v3_dt = 1.0 / 4096.0;
inline constexpr std::uint64_t scalar_igr_sc25_v3_steps = 20;
inline constexpr double scalar_igr_sc25_v3_final_time = 20.0 / 4096.0;
inline constexpr std::array<std::uint64_t, 3>
    scalar_igr_sc25_v3_output_steps{{0, 10, 20}};
inline constexpr std::uint64_t scalar_igr_sc25_v3_checkpoint_step = 10;
inline constexpr std::uint64_t scalar_igr_sc25_v3_failure_target_step = 7;
inline constexpr std::uint32_t scalar_igr_sc25_v3_retry_limit = 1;
inline constexpr std::uint64_t scalar_igr_sc25_v3_seed =
    UINT64_C(0x5738425343323553);

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
inline constexpr std::uint64_t
    smooth_periodic_euler_v1_alternate_functional_seed =
        UINT64_C(0xc90f38ff2a043f24);

inline constexpr FailurePhase smooth_periodic_euler_v1_failure_phase =
    FailurePhase::AfterRecoveryBeforeCommit;
inline constexpr std::uint32_t smooth_periodic_euler_v1_retry_limit = 1;

enum class ApplicationScenarioV1 : std::uint8_t
{
  smoke = 0,
  functional = 1,
  scalar_igr_sc25_v3 = 2,
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

enum class ApplicationRunRoleV1 : std::uint8_t
{
  uninterrupted_reference = 0,
  split_writer = 1,
  split_reader = 2,
  focused_terminal_injection = 3,
  alternate_seed_reference = 4
};

enum class ValidationCaptureModeV1 : std::uint8_t
{
  none = 0,
  persistent_v1 = 1
};

enum class StopAfterCheckpointV1 : std::uint8_t
{
  no = 0,
  yes = 1
};

enum class FailureInjectionModeV1 : std::uint8_t
{
  off = 0,
  prescribed = 1
};

struct FailureInjectionControlV1
{
  FailureInjectionModeV1 mode = FailureInjectionModeV1::off;
  FailurePhase phase = FailurePhase::Setup;
  std::uint64_t target_step = 0;
  std::uint32_t source_rank = 0;
};

constexpr bool operator==(const FailureInjectionControlV1& left,
                          const FailureInjectionControlV1& right) noexcept
{
  return left.mode == right.mode && left.phase == right.phase &&
      left.target_step == right.target_step &&
      left.source_rank == right.source_rank;
}

constexpr bool operator!=(const FailureInjectionControlV1& left,
                          const FailureInjectionControlV1& right) noexcept
{
  return !(left == right);
}

struct OutputDirectoryPathV1
{
  bool present = false;
  std::string_view canonical{};

  static constexpr OutputDirectoryPathV1 Present(
      const std::string_view value) noexcept
  {
    return {true, value};
  }
};

struct EvidenceDirectoryPathV1
{
  bool present = false;
  std::string_view canonical{};

  static constexpr EvidenceDirectoryPathV1 Present(
      const std::string_view value) noexcept
  {
    return {true, value};
  }
};

struct CheckpointArtifactPathV1
{
  bool present = false;
  std::string_view canonical{};

  static constexpr CheckpointArtifactPathV1 Present(
      const std::string_view value) noexcept
  {
    return {true, value};
  }
};

struct OptionalCheckpointStepV1
{
  bool present = false;
  std::uint64_t value = 0;

  static constexpr OptionalCheckpointStepV1 Present(
      const std::uint64_t step) noexcept
  {
    return {true, step};
  }
};

template <typename Domain>
class CanonicalPathIdentityV1 final
{
public:
  constexpr CanonicalPathIdentityV1() noexcept = default;
  constexpr bool present() const noexcept { return present_; }
  constexpr const StableId<512>& identity() const noexcept
  {
    return identity_;
  }

private:
  constexpr CanonicalPathIdentityV1(const bool present,
                                    const std::string_view canonical) noexcept
      : present_(present), identity_(StableId<512>::Copy(canonical))
  {}

  bool present_ = false;
  StableId<512> identity_{};

  friend class ValidatedApplicationConfig;
};

struct OutputDirectoryPathDomainV1;
struct EvidenceDirectoryPathDomainV1;
struct CheckpointArtifactPathDomainV1;
using OutputDirectoryIdentityV1 =
    CanonicalPathIdentityV1<OutputDirectoryPathDomainV1>;
using EvidenceDirectoryIdentityV1 =
    CanonicalPathIdentityV1<EvidenceDirectoryPathDomainV1>;
using CheckpointArtifactIdentityV1 =
    CanonicalPathIdentityV1<CheckpointArtifactPathDomainV1>;

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
  invalid_device,
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
  invalid_run_role,
  invalid_path_identity,
  invalid_validation_capture,
  invalid_checkpoint_control,
  invalid_stop_control,
  invalid_injection_control,
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
  int device_id = -1;
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
  std::string_view checkpoint_id =
      smooth_periodic_euler_v1_smoke_checkpoint_id;
  ApplicationRunRoleV1 run_role =
      ApplicationRunRoleV1::uninterrupted_reference;
  OutputDirectoryPathV1 output_directory =
      OutputDirectoryPathV1::Present("w8a/output");
  // Reserved checkpoint-ABI fields. W8A external evidence is owned by the
  // development runner, so the application admits only the disabled values.
  EvidenceDirectoryPathV1 evidence_directory{};
  ValidationCaptureModeV1 validation_capture = ValidationCaptureModeV1::none;
  CheckpointArtifactPathV1 checkpoint_out =
      CheckpointArtifactPathV1::Present("w8a/checkpoint");
  OptionalCheckpointStepV1 checkpoint_step =
      OptionalCheckpointStepV1::Present(
          smooth_periodic_euler_v1_smoke_checkpoint_step);
  CheckpointArtifactPathV1 restart_from{};
  StopAfterCheckpointV1 stop_after_checkpoint =
      StopAfterCheckpointV1::no;
  FailureInjectionControlV1 injection{};
  RestartMode restart_mode = RestartMode::same_backend_rank_partition;
  RestartTrajectoryModeV1 restart_trajectory =
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple;
  RollbackObligationV1 rollback_obligation =
      RollbackObligationV1::collective_exact_complete_state;
  DeferredSourceRankRuleV1 source_rank_rule =
      DeferredSourceRankRuleV1::rank_zero_for_rank1_rank_one_for_rank2;
  std::uint32_t retry_limit = 0;
  std::uint64_t seed = smooth_periodic_euler_v1_smoke_seed;
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
  bool scalar_igr() const noexcept
  {
    return scenario_ == ApplicationScenarioV1::scalar_igr_sc25_v3;
  }
  int device_id() const noexcept { return device_id_; }
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
  std::uint64_t scenario_checkpoint_step() const noexcept
  {
    return scenario_checkpoint_step_;
  }
  ApplicationRunRoleV1 run_role() const noexcept { return run_role_; }
  const OutputDirectoryIdentityV1& output_directory() const noexcept
  {
    return output_directory_;
  }
  const EvidenceDirectoryIdentityV1& evidence_directory() const noexcept
  {
    return evidence_directory_;
  }
  ValidationCaptureModeV1 validation_capture() const noexcept
  {
    return validation_capture_;
  }
  const CheckpointArtifactIdentityV1& checkpoint_out() const noexcept
  {
    return checkpoint_out_;
  }
  OptionalCheckpointStepV1 checkpoint_step() const noexcept
  {
    return checkpoint_step_;
  }
  const CheckpointArtifactIdentityV1& restart_from() const noexcept
  {
    return restart_from_;
  }
  StopAfterCheckpointV1 stop_after_checkpoint() const noexcept
  {
    return stop_after_checkpoint_;
  }
  FailureInjectionControlV1 injection() const noexcept { return injection_; }
  RestartMode restart_mode() const noexcept { return restart_mode_; }
  RestartTrajectoryModeV1 restart_trajectory() const noexcept
  {
    return restart_trajectory_;
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
  int device_id_ = -1;
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
  std::uint64_t scenario_checkpoint_step_ = 0;
  ApplicationRunRoleV1 run_role_ =
      ApplicationRunRoleV1::uninterrupted_reference;
  OutputDirectoryIdentityV1 output_directory_{};
  EvidenceDirectoryIdentityV1 evidence_directory_{};
  ValidationCaptureModeV1 validation_capture_ =
      ValidationCaptureModeV1::none;
  CheckpointArtifactIdentityV1 checkpoint_out_{};
  OptionalCheckpointStepV1 checkpoint_step_{};
  CheckpointArtifactIdentityV1 restart_from_{};
  StopAfterCheckpointV1 stop_after_checkpoint_ =
      StopAfterCheckpointV1::no;
  FailureInjectionControlV1 injection_{};
  RestartMode restart_mode_ = RestartMode::same_backend_rank_partition;
  RestartTrajectoryModeV1 restart_trajectory_ =
      RestartTrajectoryModeV1::uninterrupted_and_split_same_tuple;
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
bool DeriveSplitWriterRunConfigHashV1(
    const ValidatedApplicationConfig& split_reader,
    StateHash& writer_run_config_sha256) noexcept;

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
  enum class FailureClass : std::uint8_t
  {
    none = 0,
    configuration,
    checkpoint_data,
    filesystem,
    backend,
    scientific,
    injected
  };

  RunStatus status = RunStatus::invalid_request;
  FailureReason reason = FailureReason::invalid_resource_descriptor;
  StateCoordinates coordinates{};
  std::uint32_t retries = 0;
  FailureClass failure_class = FailureClass::configuration;
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

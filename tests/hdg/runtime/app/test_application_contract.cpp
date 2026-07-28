#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <type_traits>

namespace
{

using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::app;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "application contract test must compile in strict C++17");
static_assert(!std::is_default_constructible<ValidatedApplicationConfig>::value &&
                  !std::is_aggregate<ValidatedApplicationConfig>::value &&
                  !std::is_polymorphic<ValidatedApplicationConfig>::value &&
                  !std::is_polymorphic<FailureInjectionControlV1>::value,
              "validated controls are closed non-polymorphic values");
static_assert(!std::is_same<OutputDirectoryPathV1,
                            EvidenceDirectoryPathV1>::value &&
                  !std::is_same<OutputDirectoryIdentityV1,
                                EvidenceDirectoryIdentityV1>::value &&
                  !std::is_same<OutputDirectoryIdentityV1,
                                CheckpointArtifactIdentityV1>::value,
              "path domains are not interchangeable");
static_assert(IsFrozenApplicationBackend<CpuSerialBackendTag>::value &&
                  IsFrozenApplicationBackend<CpuRank2BackendTag>::value &&
                  IsFrozenApplicationBackend<
                      CudaRank1OneGpuBackendTag>::value &&
                  IsFrozenApplicationBackend<
                      CudaRank2OneGpuHostStagedBackendTag>::value &&
                  !IsFrozenApplicationBackend<int>::value,
              "the static boundary recognizes exactly four backends");

bool Check(const bool condition, const char* name)
{
  if (!condition) { std::cerr << "FAILED: " << name << '\n'; }
  return condition;
}

bool Nonzero(const StateHash& hash)
{
  for (const std::uint8_t byte : hash.bytes)
  {
    if (byte != 0) { return true; }
  }
  return false;
}

ApplicationOptionsView Smoke(const std::string_view backend)
{
  ApplicationOptionsView options;
  options.backend_id = backend;
  return options;
}

ApplicationOptionsView Functional(const std::string_view backend,
                                  const std::uint32_t mesh_n = 6)
{
  ApplicationOptionsView options = Smoke(backend);
  options.scenario_id = smooth_periodic_euler_v1_functional_id;
  options.polynomial_order =
      smooth_periodic_euler_v1_functional_polynomial_order;
  options.mesh_n = mesh_n;
  options.q_a = smooth_periodic_euler_v1_functional_q_a;
  options.q_e = smooth_periodic_euler_v1_functional_q_e;
  options.q_e4 = smooth_periodic_euler_v1_functional_q_e4;
  options.q_o = smooth_periodic_euler_v1_functional_q_o;
  options.accepted_steps = smooth_periodic_euler_v1_functional_steps;
  options.final_time = smooth_periodic_euler_v1_functional_final_time;
  options.output_steps = smooth_periodic_euler_v1_functional_output_steps;
  options.checkpoint_step = OptionalCheckpointStepV1::Present(
      smooth_periodic_euler_v1_functional_checkpoint_step);
  options.checkpoint_id =
      smooth_periodic_euler_v1_functional_checkpoint_id;
  options.seed = smooth_periodic_euler_v1_functional_seed;
  return options;
}

std::uint32_t SourceRank(const std::string_view backend)
{
  return backend == CpuRank2BackendTag::id ||
          backend == CudaRank2OneGpuHostStagedBackendTag::id
      ? 1U
      : 0U;
}

ApplicationOptionsView SplitWriter(ApplicationOptionsView options)
{
  options.run_role = ApplicationRunRoleV1::split_writer;
  options.stop_after_checkpoint = StopAfterCheckpointV1::yes;
  return options;
}

ApplicationOptionsView SplitReader(ApplicationOptionsView options)
{
  const std::string_view checkpoint = options.checkpoint_out.canonical;
  const bool smoke =
      options.scenario_id == smooth_periodic_euler_v1_smoke_id;
  options.run_role = ApplicationRunRoleV1::split_reader;
  options.checkpoint_out = {};
  options.checkpoint_step = {};
  options.restart_from = CheckpointArtifactPathV1::Present(checkpoint);
  options.stop_after_checkpoint = StopAfterCheckpointV1::no;
  options.injection = {
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      smoke ? smooth_periodic_euler_v1_smoke_failure_target_step
            : smooth_periodic_euler_v1_functional_failure_target_step,
      SourceRank(options.backend_id)};
  options.retry_limit = smooth_periodic_euler_v1_retry_limit;
  return options;
}

ApplicationOptionsView Focused(ApplicationOptionsView options,
                               const FailurePhase phase,
                               const std::uint32_t source_rank)
{
  options.run_role = ApplicationRunRoleV1::focused_terminal_injection;
  const bool staging = phase == FailurePhase::CheckpointStaging ||
      phase == FailurePhase::OutputStaging;
  options.injection = {FailureInjectionModeV1::prescribed, phase,
                       staging
                           ? smooth_periodic_euler_v1_smoke_checkpoint_step
                           : UINT64_C(2),
                       source_rank};
  return options;
}

bool CheckExactAcceptedFieldsAndHashes()
{
  const ApplicationValidation first =
      ValidateApplicationOptions(Smoke(CpuSerialBackendTag::id));
  const ApplicationValidation repeat =
      ValidateApplicationOptions(Smoke(CpuSerialBackendTag::id));
  const ValidatedApplicationConfig* config = first.config();
  const ValidatedApplicationConfig* repeated = repeat.config();
  bool passed = Check(first.accepted() && config != nullptr &&
                          repeated != nullptr,
                      "default_uninterrupted_smoke_accepted");
  if (config == nullptr || repeated == nullptr) { return false; }
  passed = Check(
               config->scenario() == ApplicationScenarioV1::smoke &&
                   config->backend() == ApplicationBackend::cpu_serial &&
                   config->tuple() == CpuSerialRuntimeSelection::Descriptor() &&
                   config->scenario_checkpoint_step() == 1 &&
                   config->run_role() ==
                       ApplicationRunRoleV1::uninterrupted_reference &&
                   config->output_directory().present() &&
                   config->output_directory().identity().view() ==
                       "w8a/output" &&
                   config->evidence_directory().present() &&
                   config->validation_capture() ==
                       ValidationCaptureModeV1::persistent_v1 &&
                   config->checkpoint_out().present() &&
                   config->checkpoint_step().present &&
                   config->checkpoint_step().value == 1 &&
                   !config->restart_from().present() &&
                   config->stop_after_checkpoint() ==
                       StopAfterCheckpointV1::no &&
                   config->injection() == FailureInjectionControlV1{} &&
                   config->retry_limit() == 0 &&
                   config->seed() == smooth_periodic_euler_v1_smoke_seed &&
                   Nonzero(config->scientific_config_sha256()) &&
                   Nonzero(config->run_config_sha256()) &&
                   config->scientific_config_sha256() ==
                       repeated->scientific_config_sha256() &&
                   config->run_config_sha256() ==
                       repeated->run_config_sha256(),
               "validated_config_owns_typed_controls_and_deterministic_hashes") &&
      passed;

  ApplicationOptionsView changed = Smoke(CpuSerialBackendTag::id);
  changed.output_directory = OutputDirectoryPathV1::Present("other/output");
  const ApplicationValidation changed_validation =
      ValidateApplicationOptions(changed);
  const ValidatedApplicationConfig* changed_config =
      changed_validation.config();
  passed = Check(changed_config != nullptr &&
                     changed_config->scientific_config_sha256() ==
                         config->scientific_config_sha256() &&
                     changed_config->run_config_sha256() !=
                         config->run_config_sha256(),
                 "run_only_path_changes_only_run_hash") &&
      passed;
  return passed;
}

bool CheckRoleBackendMatrix()
{
  struct BackendCase
  {
    std::string_view id;
    ApplicationBackend backend;
  };
  const std::array<BackendCase, 4> backends{{
      {CpuSerialBackendTag::id, ApplicationBackend::cpu_serial},
      {CpuRank2BackendTag::id, ApplicationBackend::cpu_rank2},
      {CudaRank1OneGpuBackendTag::id,
       ApplicationBackend::cuda_rank1_one_gpu},
      {CudaRank2OneGpuHostStagedBackendTag::id,
       ApplicationBackend::cuda_rank2_one_gpu_host_staged}}};
  bool passed = true;
  for (const BackendCase& backend : backends)
  {
    for (const ApplicationOptionsView& scenario :
         {Smoke(backend.id), Functional(backend.id)})
    {
      for (const ApplicationOptionsView& role :
           {scenario, SplitWriter(scenario), SplitReader(scenario)})
      {
        const ApplicationValidation validation =
            ValidateApplicationOptions(role);
        passed = Check(CompiledBackendAvailable(backend.backend)
                           ? validation.accepted()
                           : validation.rejection() ==
                               ApplicationRejection::unbuilt_backend,
                       "role_scenario_backend_matrix") &&
            passed;
      }
    }
  }
  return passed;
}

bool CheckFocusedAndAlternateRoles()
{
  const std::array<FailurePhase, 8> phases{{
      FailurePhase::Assembly, FailurePhase::Condensation,
      FailurePhase::TraceSolve, FailurePhase::Recovery,
      FailurePhase::Diagnostics, FailurePhase::CheckpointStaging,
      FailurePhase::OutputStaging,
      FailurePhase::AfterRecoveryBeforeCommit}};
  bool passed = true;
  struct FocusedBackend
  {
    std::string_view id;
    ApplicationBackend backend;
    std::uint32_t ranks;
  };
  const std::array<FocusedBackend, 4> backends{{
      {CpuSerialBackendTag::id, ApplicationBackend::cpu_serial, 1},
      {CpuRank2BackendTag::id, ApplicationBackend::cpu_rank2, 2},
      {CudaRank1OneGpuBackendTag::id,
       ApplicationBackend::cuda_rank1_one_gpu, 1},
      {CudaRank2OneGpuHostStagedBackendTag::id,
       ApplicationBackend::cuda_rank2_one_gpu_host_staged, 2}}};
  for (const FailurePhase phase : phases)
  {
    for (const FocusedBackend& backend : backends)
    {
      if (!CompiledBackendAvailable(backend.backend)) { continue; }
      for (std::uint32_t source = 0; source < backend.ranks; ++source)
      {
        passed = Check(ValidateApplicationOptions(
                           Focused(Smoke(backend.id), phase, source))
                           .accepted(),
                       "focused_phase_backend_source_matrix") &&
            passed;
      }
    }
  }
  ApplicationOptionsView alternate = Functional(CpuSerialBackendTag::id);
  alternate.run_role = ApplicationRunRoleV1::alternate_seed_reference;
  alternate.seed = smooth_periodic_euler_v1_alternate_functional_seed;
  passed = Check(ValidateApplicationOptions(alternate).accepted(),
                 "alternate_functional_seed_reference_accepted") &&
      passed;
  return passed;
}

bool CheckNegativeRoleControls()
{
  bool passed = true;
  const auto reject = [&](const ApplicationOptionsView& request,
                          const char* name) {
    passed = Check(!ValidateApplicationOptions(request).accepted(), name) &&
        passed;
  };
  ApplicationOptionsView request = Smoke(CpuSerialBackendTag::id);
  request.run_role = static_cast<ApplicationRunRoleV1>(255);
  reject(request, "unknown_role_rejected");
  request = Smoke(CpuSerialBackendTag::id);
  request.output_directory = {};
  reject(request, "missing_output_directory_rejected");
  request = Smoke(CpuSerialBackendTag::id);
  request.output_directory = OutputDirectoryPathV1::Present({});
  reject(request, "empty_present_output_directory_rejected");
  request = Smoke(CpuSerialBackendTag::id);
  request.evidence_directory = {};
  reject(request, "missing_evidence_directory_rejected");
  request = Smoke(CpuSerialBackendTag::id);
  request.validation_capture = ValidationCaptureModeV1::none;
  reject(request, "capture_evidence_mismatch_rejected");
  request = Smoke(CpuSerialBackendTag::id);
  request.validation_capture = static_cast<ValidationCaptureModeV1>(255);
  reject(request, "unknown_capture_mode_rejected");
  request = Smoke(CpuSerialBackendTag::id);
  request.evidence_directory = {};
  request.validation_capture = ValidationCaptureModeV1::none;
  passed = Check(ValidateApplicationOptions(request).accepted(),
                 "ordinary_non_evidence_pair_accepted") &&
      passed;
  request = SplitWriter(Smoke(CpuSerialBackendTag::id));
  request.checkpoint_out = {};
  reject(request, "writer_missing_checkpoint_out_rejected");
  request = SplitWriter(Smoke(CpuSerialBackendTag::id));
  request.checkpoint_step = OptionalCheckpointStepV1::Present(2);
  reject(request, "wrong_checkpoint_step_rejected");
  request = SplitWriter(Smoke(CpuSerialBackendTag::id));
  request.checkpoint_step = {false, 1};
  reject(request, "absent_nonzero_checkpoint_step_rejected");
  request = SplitWriter(Smoke(CpuSerialBackendTag::id));
  request.restart_from =
      CheckpointArtifactPathV1::Present("unexpected/restart");
  reject(request, "writer_restart_path_rejected");
  request = SplitWriter(Smoke(CpuSerialBackendTag::id));
  request.stop_after_checkpoint = StopAfterCheckpointV1::no;
  reject(request, "writer_stop_control_rejected");
  request = SplitWriter(Smoke(CpuSerialBackendTag::id));
  request.stop_after_checkpoint = static_cast<StopAfterCheckpointV1>(255);
  reject(request, "unknown_stop_control_rejected");
  request = SplitReader(Smoke(CpuSerialBackendTag::id));
  request.restart_from = {};
  reject(request, "reader_missing_restart_path_rejected");
  request = SplitReader(Smoke(CpuSerialBackendTag::id));
  request.checkpoint_step = OptionalCheckpointStepV1::Present(1);
  reject(request, "reader_checkpoint_step_rejected");
  request = SplitReader(Smoke(CpuSerialBackendTag::id));
  request.injection.phase = FailurePhase::Recovery;
  reject(request, "reader_injection_phase_rejected");
  request = SplitReader(Smoke(CpuSerialBackendTag::id));
  request.injection.mode = static_cast<FailureInjectionModeV1>(255);
  reject(request, "unknown_injection_mode_rejected");
  request = SplitReader(Smoke(CpuSerialBackendTag::id));
  request.injection.target_step = 3;
  reject(request, "reader_injection_target_rejected");
  request = SplitReader(Smoke(CpuSerialBackendTag::id));
  request.injection.source_rank = 1;
  reject(request, "reader_injection_source_rejected");
  request = SplitReader(Smoke(CpuSerialBackendTag::id));
  request.retry_limit = 0;
  reject(request, "reader_retry_rejected");
  request = Smoke(CpuSerialBackendTag::id);
  request.seed = 0;
  reject(request, "scenario_seed_rejected");
  request = Focused(Smoke(CpuSerialBackendTag::id), FailurePhase::Recovery, 0);
  request.injection.phase = FailurePhase::Commit;
  reject(request, "unlisted_focused_phase_rejected");
  request = Focused(Smoke(CpuSerialBackendTag::id), FailurePhase::Recovery, 0);
  request.injection.target_step = 1;
  reject(request, "focused_nonstaging_target_rejected");
  request = Focused(Smoke(CpuSerialBackendTag::id),
                    FailurePhase::CheckpointStaging, 0);
  request.injection.target_step = 2;
  reject(request, "focused_staging_target_rejected");
  request = Functional(CpuSerialBackendTag::id);
  request.run_role = ApplicationRunRoleV1::alternate_seed_reference;
  reject(request, "alternate_role_requires_alternate_seed");
  return passed;
}

bool CheckScientificCrossProducts()
{
  bool passed = true;
  for (const std::uint32_t mesh : {3U, 6U, 12U})
  {
    passed = Check(ValidateApplicationOptions(
                       Functional(CpuSerialBackendTag::id, mesh)).accepted(),
                   "serial_functional_ladder") &&
        passed;
  }
  ApplicationOptionsView request = Smoke(CpuSerialBackendTag::id);
  request.mesh_n = 6;
  passed = Check(ValidateApplicationOptions(request).rejection() ==
                     ApplicationRejection::invalid_polynomial_mesh_pair,
                 "smoke_cross_product_rejected") &&
      passed;
  request = Functional(CpuRank2BackendTag::id, 3);
  passed = Check(ValidateApplicationOptions(request).rejection() ==
                     (CompiledBackendAvailable(ApplicationBackend::cpu_rank2)
                          ? ApplicationRejection::invalid_polynomial_mesh_pair
                          : ApplicationRejection::unbuilt_backend),
                 "rank2_serial_ladder_cross_product_rejected") &&
      passed;
  return passed;
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckExactAcceptedFieldsAndHashes() && passed;
  passed = CheckRoleBackendMatrix() && passed;
  passed = CheckFocusedAndAlternateRoles() && passed;
  passed = CheckNegativeRoleControls() && passed;
  passed = CheckScientificCrossProducts() && passed;
  if (passed)
  {
    std::cout << "HDG W8A typed application admission checks passed\n";
    return 0;
  }
  return 1;
}

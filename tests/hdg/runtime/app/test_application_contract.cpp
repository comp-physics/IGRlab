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
static_assert(DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU == 0 ||
                  DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU == 1,
              "scalar-IGR CPU availability is a build-owned boolean");
static_assert(DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK1 == 0 ||
                  DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK1 == 1,
              "scalar-IGR CUDA rank-1 availability is build-owned");
static_assert(DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK2 == 0 ||
                  DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK2 == 1,
              "scalar-IGR CUDA rank-2 availability is build-owned");

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
  options.device_id = backend == CudaRank1OneGpuBackendTag::id ||
          backend == CudaRank2OneGpuHostStagedBackendTag::id
      ? 0 : -1;
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

ApplicationOptionsView ScalarIgr(const std::string_view backend)
{
  ApplicationOptionsView options = Smoke(backend);
  options.card_id = scalar_igr_sc25_v3_card_id;
  options.case_id = scalar_igr_sc25_v3_case_id;
  options.scenario_id = scalar_igr_sc25_v3_scenario_id;
  options.model_id = TracedScalarSc25ProjectionV3Tag::id;
  options.formulation_id = FullyHybridizedScalarSc25V3Tag::id;
  options.trace_id = DualConservativeScalarTraceV1Tag::id;
  options.stabilization_id = W7Tau25DensityHeightP2V3Tag::id;
  options.time_treatment_id = BackwardEulerMonolithicV1Tag::id;
  options.source_id = scalar_igr_sc25_v3_source_id;
  options.flux_id = scalar_igr_sc25_v3_flux_id;
  options.time_rule_id = BackwardEulerMonolithicV1Tag::id;
  options.layout_id = scalar_igr_sc25_v3_layout_id;
  options.polynomial_order = scalar_igr_sc25_v3_polynomial_order;
  options.mesh_n = scalar_igr_sc25_v3_mesh_n;
  options.q_a = scalar_igr_sc25_v3_q_a;
  options.q_e = scalar_igr_sc25_v3_q_e;
  options.q_e4 = scalar_igr_sc25_v3_q_e4;
  options.q_o = scalar_igr_sc25_v3_q_o;
  options.dt = scalar_igr_sc25_v3_dt;
  options.accepted_steps = scalar_igr_sc25_v3_steps;
  options.final_time = scalar_igr_sc25_v3_final_time;
  options.output_steps = scalar_igr_sc25_v3_output_steps;
  options.checkpoint_id = scalar_igr_sc25_v3_checkpoint_id;
  options.checkpoint_step =
      OptionalCheckpointStepV1::Present(scalar_igr_sc25_v3_checkpoint_step);
  options.injection = {
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      scalar_igr_sc25_v3_failure_target_step,
      backend == CpuRank2BackendTag::id ||
              backend == CudaRank2OneGpuHostStagedBackendTag::id
          ? 1U : 0U};
  options.retry_limit = scalar_igr_sc25_v3_retry_limit;
  options.seed = scalar_igr_sc25_v3_seed;
  return options;
}

ApplicationOptionsView W8cCase(const bool scalar, const bool smooth,
                               const std::string_view backend =
                                   CpuSerialBackendTag::id)
{
  ApplicationOptionsView options = scalar
      ? ScalarIgr(backend)
      : Smoke(backend);
  options.case_id = smooth ? "periodic-smooth-transition-v2"
                           : "periodic-double-riemann-v1";
  options.scenario_id = smooth ? "W8C-V2-N8-DT10" : "W8C-N8-DT10";
  options.source_id = scalar
      ? (smooth ? "source.sc25.periodic-smooth-transition.v3"
                : "source.sc25.periodic-double-riemann.v3")
      : (smooth ? "source.euler.periodic-smooth-transition.v2"
                : "source.euler.periodic-double-riemann.v1");
  options.polynomial_order = 1;
  options.mesh_n = 8;
  options.q_a = scalar ? 12 : 11;
  options.q_e = 0;
  options.q_e4 = 0;
  options.q_o = 18;
  options.dt = 1.0 / 1024.0;
  options.accepted_steps = 64;
  options.final_time = 1.0 / 16.0;
  options.output_steps = {{0, 32, 64}};
  options.checkpoint_id = scalar
      ? (smooth
             ? "checkpoint.hdg.igr.sc25.periodic-smooth-transition.n8-dt10.step-32.v2"
             : "checkpoint.hdg.igr.sc25.periodic-double-riemann.n8-dt10.step-32.v1")
      : (smooth
             ? "checkpoint.hdg.euler.periodic-smooth-transition.n8-dt10.step-32.v2"
             : "checkpoint.hdg.euler.periodic-double-riemann.n8-dt10.step-32.v1");
  options.checkpoint_step = OptionalCheckpointStepV1::Present(32);
  options.injection = {};
  options.retry_limit = 0;
  options.seed = smooth ? UINT64_C(0x573843534d4f5632)
                        : UINT64_C(0x5738434452495631);
  return options;
}

ApplicationOptionsView W8cSplitReader(ApplicationOptionsView options)
{
  const std::string_view checkpoint = options.checkpoint_out.canonical;
  options.run_role = ApplicationRunRoleV1::split_reader;
  options.checkpoint_out = {};
  options.checkpoint_step = {};
  options.restart_from = CheckpointArtifactPathV1::Present(checkpoint);
  options.stop_after_checkpoint = StopAfterCheckpointV1::no;
  options.injection = {};
  options.retry_limit = 0;
  return options;
}

void SelectTuple(ApplicationOptionsView& options, const bool scalar)
{
  if (scalar)
  {
    options.model_id = TracedScalarSc25ProjectionV3Tag::id;
    options.formulation_id = FullyHybridizedScalarSc25V3Tag::id;
    options.trace_id = DualConservativeScalarTraceV1Tag::id;
    options.stabilization_id = W7Tau25DensityHeightP2V3Tag::id;
    options.time_treatment_id = BackwardEulerMonolithicV1Tag::id;
  }
  else
  {
    options.model_id = EulerCompressible2dTag::id;
    options.formulation_id = ConventionalHdgTag::id;
    options.trace_id = MfemDgInterfaceTraceTag::id;
    options.stabilization_id = FixedScalarTauTag::id;
    options.time_treatment_id = BackwardEulerRuntimeTag::id;
  }
}

ApplicationOptionsView RoleForScenario(
    ApplicationOptionsView options, const ApplicationScenarioV1 scenario,
    const ApplicationRunRoleV1 role)
{
  const std::string_view checkpoint = options.checkpoint_out.canonical;
  options.run_role = role;
  if (role == ApplicationRunRoleV1::split_writer)
  {
    options.stop_after_checkpoint = StopAfterCheckpointV1::yes;
    return options;
  }
  if (role != ApplicationRunRoleV1::split_reader) return options;

  options.checkpoint_out = {};
  options.checkpoint_step = {};
  options.restart_from = CheckpointArtifactPathV1::Present(checkpoint);
  options.stop_after_checkpoint = StopAfterCheckpointV1::no;
  if (scenario == ApplicationScenarioV1::scalar_igr_sc25_v3)
  {
    options.injection = {};
    options.retry_limit = 0;
  }
  else
  {
    options.injection = {
        FailureInjectionModeV1::prescribed,
        smooth_periodic_euler_v1_failure_phase,
        scenario == ApplicationScenarioV1::smoke
            ? smooth_periodic_euler_v1_smoke_failure_target_step
            : smooth_periodic_euler_v1_functional_failure_target_step,
        options.backend_id == CpuRank2BackendTag::id ||
                options.backend_id ==
                    CudaRank2OneGpuHostStagedBackendTag::id
            ? 1U : 0U};
    options.retry_limit = smooth_periodic_euler_v1_retry_limit;
  }
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

ApplicationOptionsView CanonicalScalarCheckpointWriter(
    ApplicationOptionsView options)
{
  options = SplitWriter(options);
  options.injection = {
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      scalar_igr_sc25_v3_failure_target_step,
      SourceRank(options.backend_id)};
  options.retry_limit = scalar_igr_sc25_v3_retry_limit;
  return options;
}

ApplicationOptionsView SplitReader(ApplicationOptionsView options)
{
  const std::string_view checkpoint = options.checkpoint_out.canonical;
  const bool smoke =
      options.scenario_id == smooth_periodic_euler_v1_smoke_id;
  const bool scalar = options.card_id == scalar_igr_sc25_v3_card_id;
  options.run_role = ApplicationRunRoleV1::split_reader;
  options.checkpoint_out = {};
  options.checkpoint_step = {};
  options.restart_from = CheckpointArtifactPathV1::Present(checkpoint);
  options.stop_after_checkpoint = StopAfterCheckpointV1::no;
  options.injection = scalar
      ? FailureInjectionControlV1{}
      : FailureInjectionControlV1{
            FailureInjectionModeV1::prescribed,
            smooth_periodic_euler_v1_failure_phase,
            smoke ? smooth_periodic_euler_v1_smoke_failure_target_step
                  : smooth_periodic_euler_v1_functional_failure_target_step,
            SourceRank(options.backend_id)};
  options.retry_limit = scalar ? 0 : smooth_periodic_euler_v1_retry_limit;
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
  options.retry_limit = smooth_periodic_euler_v1_retry_limit;
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
                   !config->evidence_directory().present() &&
                   config->validation_capture() ==
                       ValidationCaptureModeV1::none &&
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

bool CheckW8cSmoothTransitionClosure()
{
  bool passed = true;
  for (const bool scalar : {false, true})
  {
    const ApplicationValidation smooth_validation =
        ValidateApplicationOptions(W8cCase(scalar, true));
    const ApplicationValidation exact_validation =
        ValidateApplicationOptions(W8cCase(scalar, false));
    const ValidatedApplicationConfig* smooth = smooth_validation.config();
    const ValidatedApplicationConfig* exact = exact_validation.config();
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU == 0
    if (scalar)
    {
      passed = Check(smooth == nullptr && exact == nullptr &&
                         smooth_validation.rejection() ==
                             ApplicationRejection::unbuilt_backend &&
                         exact_validation.rejection() ==
                             ApplicationRejection::unbuilt_backend,
                     "smooth_and_exact_scalar_cases_unbuilt_together") &&
          passed;
      continue;
    }
#endif
    passed = Check(smooth != nullptr && exact != nullptr &&
                       smooth->backend() == ApplicationBackend::cpu_serial &&
                       smooth->scenario() ==
                           (scalar
                                ? ApplicationScenarioV1::scalar_igr_sc25_v3
                                : ApplicationScenarioV1::smoke) &&
                       smooth->case_id().view() ==
                           "periodic-smooth-transition-v2" &&
                       smooth->scenario_checkpoint_step() == 32 &&
                       Nonzero(smooth->scientific_config_sha256()) &&
                       Nonzero(smooth->run_config_sha256()) &&
                       smooth->scientific_config_sha256() !=
                           exact->scientific_config_sha256() &&
                       smooth->run_config_sha256() !=
                           exact->run_config_sha256(),
                   "smooth_case_private_scenario_and_hash_separation") &&
        passed;

    ApplicationOptionsView writer = W8cCase(scalar, true);
    writer.run_role = ApplicationRunRoleV1::split_writer;
    writer.stop_after_checkpoint = StopAfterCheckpointV1::yes;
    const ApplicationValidation writer_validation =
        ValidateApplicationOptions(writer);
    const ApplicationValidation reader_validation =
        ValidateApplicationOptions(W8cSplitReader(W8cCase(scalar, true)));
    passed = Check(writer_validation.accepted() &&
                       reader_validation.accepted(),
                   "smooth_split_writer_reader_use_off_zero_controls") &&
        passed;

    ApplicationOptionsView crossed = W8cCase(scalar, true);
    crossed.case_id = "periodic-double-riemann-v1";
    passed = Check(!ValidateApplicationOptions(crossed).accepted(),
                   "crossed_smooth_descriptor_exact_case_rejected") &&
        passed;
    crossed = W8cCase(scalar, true);
    crossed.source_id = scalar
        ? "source.sc25.periodic-double-riemann.v3"
        : "source.euler.periodic-double-riemann.v1";
    passed = Check(ValidateApplicationOptions(crossed).rejection() ==
                       ApplicationRejection::invalid_scientific_identity,
                   "crossed_smooth_source_identity_rejected") &&
        passed;

    ApplicationOptionsView rank2 = W8cCase(scalar, true);
    rank2.backend_id = CpuRank2BackendTag::id;
    const ApplicationValidation rank2_validation =
        ValidateApplicationOptions(rank2);
    passed = Check(scalar ? rank2_validation.accepted()
                          : rank2_validation.rejection() ==
                                ApplicationRejection::unknown_backend,
                   "smooth_scalar_cpu_rank2_is_frozen") &&
        passed;
    if (scalar)
    {
      rank2.scenario_id = "W8C-V2-N16-DT11";
      passed = Check(ValidateApplicationOptions(rank2).rejection() ==
                         ApplicationRejection::unknown_backend,
                     "smooth_scalar_cpu_rank2_rejects_n16") &&
          passed;
    }
  }

  ApplicationOptionsView cuda_rank2 = W8cCase(
      true, true, CudaRank2OneGpuHostStagedBackendTag::id);
  const ApplicationValidation cuda_rank2_validation =
      ValidateApplicationOptions(cuda_rank2);
#if DG_IGR_HDG_APP_HAS_CUDA_RANK2 && DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK2
  passed = Check(cuda_rank2_validation.accepted() &&
                     cuda_rank2_validation.config() != nullptr &&
                     cuda_rank2_validation.config()->backend() ==
                         ApplicationBackend::cuda_rank2_one_gpu_host_staged,
                 "smooth_scalar_cuda_rank2_is_frozen") &&
      passed;
  ApplicationOptionsView cuda_writer = SplitWriter(cuda_rank2);
  cuda_writer.injection = {
      FailureInjectionModeV1::prescribed,
      FailurePhase::AfterRecoveryBeforeCommit,
      scalar_igr_sc25_v3_failure_target_step, 1};
  cuda_writer.retry_limit = scalar_igr_sc25_v3_retry_limit;
  const ApplicationValidation cuda_reader_validation =
      ValidateApplicationOptions(SplitReader(cuda_rank2));
  StateHash cuda_writer_hash{};
  passed = Check(ValidateApplicationOptions(cuda_writer).accepted() &&
                     cuda_reader_validation.accepted() &&
                     DeriveSplitWriterRunConfigHashV1(
                         *cuda_reader_validation.config(), cuda_writer_hash),
                 "smooth_scalar_cuda_rank2_retry_writer_reader_controls") &&
      passed;
  cuda_rank2.scenario_id = "W8C-V2-N16-DT11";
  passed = Check(ValidateApplicationOptions(cuda_rank2).rejection() ==
                     ApplicationRejection::unknown_backend,
                 "smooth_scalar_cuda_rank2_rejects_n16") &&
      passed;
#else
  passed = Check(cuda_rank2_validation.rejection() ==
                     ApplicationRejection::unbuilt_backend,
                 "smooth_scalar_cuda_rank2_unbuilt") &&
      passed;
#endif

  const ApplicationValidation cuda_rank1 = ValidateApplicationOptions(
      W8cCase(true, true, CudaRank1OneGpuBackendTag::id));
  passed = Check(!cuda_rank1.accepted(),
                 "smooth_scalar_cuda_rank1_rejected") &&
      passed;
  const ApplicationValidation conventional_cuda_rank2 =
      ValidateApplicationOptions(W8cCase(
          false, true, CudaRank2OneGpuHostStagedBackendTag::id));
  passed = Check(!conventional_cuda_rank2.accepted(),
                 "smooth_conventional_cuda_rank2_rejected") &&
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
  request.evidence_directory =
      EvidenceDirectoryPathV1::Present("unaccepted/evidence");
  request.validation_capture = ValidationCaptureModeV1::persistent_v1;
  reject(request, "application_capture_pair_rejected");
  request = Smoke(CpuSerialBackendTag::id);
  request.validation_capture = static_cast<ValidationCaptureModeV1>(255);
  reject(request, "unknown_capture_mode_rejected");
  request = Smoke(CpuSerialBackendTag::id);
  passed = Check(ValidateApplicationOptions(request).accepted(),
                 "external_evidence_default_accepted") &&
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
  request.retry_limit = 0;
  reject(request, "focused_retry_required");
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

bool CheckCardScenarioTupleClosure()
{
  struct BackendRow
  {
    std::string_view id;
    ApplicationBackend backend;
    bool scalar_available;
  };
  const std::array<BackendRow, 4> backends{{
      {CpuSerialBackendTag::id, ApplicationBackend::cpu_serial,
       DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU == 1},
      {CpuRank2BackendTag::id, ApplicationBackend::cpu_rank2,
       DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU == 1},
      {CudaRank1OneGpuBackendTag::id,
       ApplicationBackend::cuda_rank1_one_gpu,
       DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK1 == 1},
      {CudaRank2OneGpuHostStagedBackendTag::id,
       ApplicationBackend::cuda_rank2_one_gpu_host_staged,
       DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK2 == 1}}};
  const std::array<std::string_view, 2> cards{{
      smooth_periodic_euler_v1_card_id, scalar_igr_sc25_v3_card_id}};
  const std::array<std::string_view, 2> cases{{
      smooth_periodic_euler_v1_case_id, scalar_igr_sc25_v3_case_id}};
  const std::array<std::string_view, 3> scenarios{{
      smooth_periodic_euler_v1_smoke_id,
      smooth_periodic_euler_v1_functional_id,
      scalar_igr_sc25_v3_scenario_id}};

  bool passed = true;
  std::size_t total_rows = 0;
  std::size_t coherent = 0;
  std::size_t incoherent_rejections = 0;
  std::size_t expected_acceptances = 0;
  std::size_t observed_acceptances = 0;
  for (std::size_t card = 0; card < cards.size(); ++card)
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index)
      for (std::size_t tuple = 0; tuple < 2; ++tuple)
        for (std::size_t scenario = 0; scenario < scenarios.size(); ++scenario)
          for (const BackendRow& backend : backends)
          {
            const bool scalar_card = card == 1;
            ApplicationOptionsView options = scalar_card
                ? ScalarIgr(backend.id)
                : (scenario == 1 ? Functional(backend.id)
                                 : Smoke(backend.id));
            options.card_id = cards[card];
            options.case_id = cases[case_index];
            options.scenario_id = scenarios[scenario];
            SelectTuple(options, tuple == 1);

            const bool coherent_row =
                (!scalar_card && case_index == 0 && tuple == 0 &&
                 scenario < 2) ||
                (scalar_card && case_index == 1 && tuple == 1 &&
                 scenario == 2);
            const bool available = CompiledBackendAvailable(backend.backend) &&
                (!scalar_card || backend.scalar_available);
            const bool expected = coherent_row && available;
            const ApplicationValidation validation =
                ValidateApplicationOptions(options);
            ++total_rows;
            coherent += coherent_row ? 1U : 0U;
            expected_acceptances += expected ? 1U : 0U;
            observed_acceptances += validation.accepted() ? 1U : 0U;
            incoherent_rejections +=
                !coherent_row && !validation.accepted() ? 1U : 0U;
            passed = Check(validation.accepted() == expected,
                           "card_case_tuple_scenario_backend_row") && passed;
          }
  passed = Check(total_rows == 96 && coherent == 12 &&
                     incoherent_rejections == total_rows - coherent &&
                     observed_acceptances == expected_acceptances,
                 "exact_96_row_admission_census") && passed;

  ApplicationOptionsView crossed = Smoke(CpuSerialBackendTag::id);
  crossed.case_id = scalar_igr_sc25_v3_case_id;
  passed = Check(ValidateApplicationOptions(crossed).rejection() ==
                     ApplicationRejection::unknown_case,
                 "conventional_card_scalar_case_rejected") && passed;
  crossed = ScalarIgr(CpuSerialBackendTag::id);
  crossed.case_id = smooth_periodic_euler_v1_case_id;
  passed = Check(ValidateApplicationOptions(crossed).rejection() ==
                     ApplicationRejection::unknown_case,
                 "scalar_card_conventional_case_rejected") && passed;
  crossed = Smoke(CpuSerialBackendTag::id);
  SelectTuple(crossed, true);
  passed = Check(ValidateApplicationOptions(crossed).rejection() ==
                     ApplicationRejection::unknown_tuple,
                 "conventional_card_scalar_tuple_rejected") && passed;
  crossed = ScalarIgr(CpuSerialBackendTag::id);
  SelectTuple(crossed, false);
  passed = Check(ValidateApplicationOptions(crossed).rejection() ==
                     ApplicationRejection::unknown_tuple,
                 "scalar_card_conventional_tuple_rejected") && passed;
  return passed;
}

bool CheckMismatchedScenarioRoleFamilies()
{
  ApplicationOptionsView conventional_scalar =
      Functional(CpuSerialBackendTag::id);
  conventional_scalar.scenario_id = scalar_igr_sc25_v3_scenario_id;
  conventional_scalar.checkpoint_step = OptionalCheckpointStepV1::Present(
      scalar_igr_sc25_v3_checkpoint_step);
  conventional_scalar.injection = {
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      scalar_igr_sc25_v3_failure_target_step, 0};
  conventional_scalar.retry_limit = scalar_igr_sc25_v3_retry_limit;
  conventional_scalar.seed = scalar_igr_sc25_v3_seed;

  ApplicationOptionsView scalar_smoke =
      ScalarIgr(CpuSerialBackendTag::id);
  scalar_smoke.scenario_id = smooth_periodic_euler_v1_smoke_id;
  scalar_smoke.checkpoint_step = OptionalCheckpointStepV1::Present(
      smooth_periodic_euler_v1_smoke_checkpoint_step);
  scalar_smoke.injection = {};
  scalar_smoke.retry_limit = 0;
  scalar_smoke.seed = smooth_periodic_euler_v1_smoke_seed;

  ApplicationOptionsView scalar_functional =
      ScalarIgr(CpuSerialBackendTag::id);
  scalar_functional.scenario_id = smooth_periodic_euler_v1_functional_id;
  scalar_functional.injection = {};
  scalar_functional.retry_limit = 0;
  scalar_functional.seed = smooth_periodic_euler_v1_functional_seed;

  bool passed = true;
  const std::array<std::pair<ApplicationOptionsView, ApplicationScenarioV1>, 3>
      families{{
          {conventional_scalar,
           ApplicationScenarioV1::scalar_igr_sc25_v3},
          {scalar_smoke, ApplicationScenarioV1::smoke},
          {scalar_functional, ApplicationScenarioV1::functional}}};
  for (const auto& family : families)
    for (const ApplicationRunRoleV1 role :
         {ApplicationRunRoleV1::uninterrupted_reference,
          ApplicationRunRoleV1::split_writer,
          ApplicationRunRoleV1::split_reader})
    {
      const ApplicationValidation validation = ValidateApplicationOptions(
          RoleForScenario(family.first, family.second, role));
      passed = Check(validation.rejection() ==
                         ApplicationRejection::unknown_scenario,
                     "mismatched_card_scenario_role_rejected_early") &&
          passed;
    }
  return passed;
}

bool CheckStaticEntryOffDiagonalMatrix()
{
  const auto matches = [](const ApplicationOptionsView& request,
                          const ApplicationBackend selected_backend,
                          const RuntimeTupleDescriptor& selected_tuple) {
    const ApplicationValidation validation =
        ValidateApplicationOptions(request);
    return validation.accepted() &&
        validation.config()->backend() == selected_backend &&
        validation.config()->tuple() == selected_tuple;
  };
  struct Row
  {
    ApplicationOptionsView request;
    ApplicationBackend selected_backend;
    RuntimeTupleDescriptor selected_tuple;
  };
  const std::array<Row, 8> rows{{
      {Smoke(CudaRank1OneGpuBackendTag::id),
       ApplicationBackend::cpu_serial,
       CpuSerialRuntimeSelection::Descriptor()},
      {Smoke(CpuSerialBackendTag::id),
       ApplicationBackend::cuda_rank1_one_gpu,
       CudaRank1OneGpuRuntimeSelection::Descriptor()},
      {Smoke(CudaRank2OneGpuHostStagedBackendTag::id),
       ApplicationBackend::cpu_rank2,
       CpuRank2RuntimeSelection::Descriptor()},
      {Smoke(CpuRank2BackendTag::id),
       ApplicationBackend::cuda_rank2_one_gpu_host_staged,
       CudaRank2OneGpuHostStagedRuntimeSelection::Descriptor()},
      {ScalarIgr(CudaRank1OneGpuBackendTag::id),
       ApplicationBackend::cpu_serial,
       ScalarIgrCpuSerialRuntimeSelection::Descriptor()},
      {ScalarIgr(CpuSerialBackendTag::id),
       ApplicationBackend::cuda_rank1_one_gpu,
       ScalarIgrCudaRank1RuntimeSelection::Descriptor()},
      {ScalarIgr(CudaRank2OneGpuHostStagedBackendTag::id),
       ApplicationBackend::cpu_rank2,
       ScalarIgrCpuRank2RuntimeSelection::Descriptor()},
      {ScalarIgr(CpuRank2BackendTag::id),
       ApplicationBackend::cuda_rank2_one_gpu_host_staged,
       ScalarIgrCudaRank2RuntimeSelection::Descriptor()}}};
  bool passed = true;
  for (const Row& row : rows)
    passed = Check(!matches(row.request, row.selected_backend,
                            row.selected_tuple),
                   "same_rank_static_entry_off_diagonal_rejected") && passed;
  return passed;
}

bool CheckCoherentScalarIdentityEquivalence()
{
  struct BackendRow
  {
    std::string_view id;
    ApplicationBackend backend;
    bool scalar_available;
    RuntimeTupleDescriptor tuple;
  };
  const std::array<BackendRow, 4> rows{{
      {CpuSerialBackendTag::id, ApplicationBackend::cpu_serial,
       DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU == 1,
       ScalarIgrCpuSerialRuntimeSelection::Descriptor()},
      {CpuRank2BackendTag::id, ApplicationBackend::cpu_rank2,
       DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU == 1,
       ScalarIgrCpuRank2RuntimeSelection::Descriptor()},
      {CudaRank1OneGpuBackendTag::id,
       ApplicationBackend::cuda_rank1_one_gpu,
       DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK1 == 1,
       ScalarIgrCudaRank1RuntimeSelection::Descriptor()},
      {CudaRank2OneGpuHostStagedBackendTag::id,
       ApplicationBackend::cuda_rank2_one_gpu_host_staged,
       DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK2 == 1,
       ScalarIgrCudaRank2RuntimeSelection::Descriptor()}}};
  bool passed = true;
  for (const BackendRow& row : rows)
  {
    const ApplicationOptionsView base = ScalarIgr(row.id);
    const ApplicationValidation first = ValidateApplicationOptions(base);
    const ApplicationValidation repeat = ValidateApplicationOptions(base);
    if (!CompiledBackendAvailable(row.backend) || !row.scalar_available)
    {
      passed = Check(first.rejection() ==
                         ApplicationRejection::unbuilt_backend,
                     "coherent_scalar_unbuilt_rejection") && passed;
      continue;
    }
    const ApplicationValidation writer = ValidateApplicationOptions(
        CanonicalScalarCheckpointWriter(base));
    const ApplicationValidation reader =
        ValidateApplicationOptions(SplitReader(base));
    StateHash derived_writer{};
    const bool derived = reader.accepted() &&
        DeriveSplitWriterRunConfigHashV1(*reader.config(), derived_writer);
    CheckpointDescriptor checkpoint{};
    if (first.accepted())
    {
      checkpoint.schema_version = checkpoint_schema_version;
      checkpoint.tuple = first.config()->tuple();
    }
    passed = Check(
                 first.accepted() && repeat.accepted() && writer.accepted() &&
                     first.config()->scenario() ==
                         ApplicationScenarioV1::scalar_igr_sc25_v3 &&
                     first.config()->backend() == row.backend &&
                     first.config()->tuple() == row.tuple &&
                     first.config()->scientific_config_sha256() ==
                         repeat.config()->scientific_config_sha256() &&
                     first.config()->run_config_sha256() ==
                         repeat.config()->run_config_sha256() &&
                     writer.config()->scientific_config_sha256() ==
                         first.config()->scientific_config_sha256() &&
                     writer.config()->run_config_sha256() !=
                         first.config()->run_config_sha256() &&
                     derived &&
                     derived_writer == writer.config()->run_config_sha256() &&
                     checkpoint.schema_version == checkpoint_schema_version &&
                     checkpoint.tuple == row.tuple,
                 "coherent_scalar_hash_and_checkpoint_schema_equivalence") &&
        passed;
  }
  return passed;
}

bool CheckScalarIgrMatrix()
{
  bool passed = true;
  const auto expected = [](const ApplicationBackend backend) {
    return CompiledBackendAvailable(backend) &&
        DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU == 1;
  };
  for (const auto& item : std::array<std::pair<std::string_view,
                                               ApplicationBackend>, 2>{{
           {CpuSerialBackendTag::id, ApplicationBackend::cpu_serial},
           {CpuRank2BackendTag::id, ApplicationBackend::cpu_rank2}}})
  {
    const ApplicationValidation validation =
        ValidateApplicationOptions(ScalarIgr(item.first));
    passed = Check(expected(item.second) ? validation.accepted()
                                        : validation.rejection() ==
                                              ApplicationRejection::unbuilt_backend,
                   "scalar_igr_cpu_static_matrix") && passed;
    if (validation.accepted())
      passed = Check(validation.config()->scalar_igr() &&
                         validation.config()->scenario() ==
                             ApplicationScenarioV1::scalar_igr_sc25_v3,
                     "scalar_igr_exact_scenario") && passed;
  }
  for (const auto& item : std::array<std::pair<std::string_view, bool>, 2>{{
           {CudaRank1OneGpuBackendTag::id,
            DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK1 == 1},
           {CudaRank2OneGpuHostStagedBackendTag::id,
            DG_IGR_HDG_APP_HAS_SCALAR_IGR_CUDA_RANK2 == 1}}})
  {
    const ApplicationValidation validation =
        ValidateApplicationOptions(ScalarIgr(item.first));
    passed = Check(item.second ? validation.accepted()
                               : validation.rejection() ==
                                     ApplicationRejection::unbuilt_backend,
                   "scalar_igr_cuda_static_matrix") && passed;
    if (!item.second) continue;

    const std::uint32_t rank_count =
        item.first == CudaRank2OneGpuHostStagedBackendTag::id ? 2U : 1U;
    ApplicationOptionsView fault_free = ScalarIgr(item.first);
    fault_free.injection = {};
    fault_free.retry_limit = 0;
    const ApplicationValidation fault_free_validation =
        ValidateApplicationOptions(fault_free);
    const ApplicationValidation reader_validation =
        ValidateApplicationOptions(SplitReader(fault_free));
    StateHash expected_writer_hash{};
    const bool expected_writer_ready = reader_validation.accepted() &&
        DeriveSplitWriterRunConfigHashV1(
            *reader_validation.config(), expected_writer_hash);
    const auto canonical_writer_hash_matches =
        [&](const ApplicationOptionsView& live)
        {
          const ApplicationValidation normalized =
              ValidateApplicationOptions(CanonicalScalarCheckpointWriter(live));
          return expected_writer_ready && normalized.accepted() &&
              normalized.config()->run_config_sha256() ==
                  expected_writer_hash;
        };
    const ApplicationValidation fault_free_public_writer =
        ValidateApplicationOptions(SplitWriter(fault_free));
    passed = Check(
                 fault_free_validation.accepted() &&
                     fault_free_public_writer.accepted() &&
                     fault_free_public_writer.config()->run_config_sha256() !=
                         expected_writer_hash &&
                     canonical_writer_hash_matches(fault_free),
                 "scalar_cuda_fault_free_checkpoint_writer_hash_normalizes") &&
        passed;

    for (std::uint32_t source = 0; source < rank_count; ++source)
    {
      for (const FailurePhase phase :
           {FailurePhase::CheckpointStaging, FailurePhase::OutputStaging,
            FailurePhase::AfterRecoveryBeforeCommit})
      {
        ApplicationOptionsView scheduled = ScalarIgr(item.first);
        scheduled.injection = {FailureInjectionModeV1::prescribed, phase,
                               scalar_igr_sc25_v3_checkpoint_step, source};
        const ApplicationValidation scheduled_validation =
            ValidateApplicationOptions(scheduled);
        passed = Check(
                     scheduled_validation.accepted() &&
                         scheduled_validation.config()->injection() ==
                             scheduled.injection &&
                         scheduled_validation.config()->retry_limit() == 1 &&
                         ValidateApplicationOptions(
                             SplitWriter(scheduled)).accepted() &&
                         canonical_writer_hash_matches(scheduled),
                     "scalar_cuda_step10_scheduled_boundary_contract") &&
            passed;
      }

      ApplicationOptionsView post_publication = ScalarIgr(item.first);
      post_publication.injection = {
          FailureInjectionModeV1::prescribed,
          FailurePhase::AfterRecoveryBeforeCommit,
          scalar_igr_sc25_v3_steps, source};
      post_publication.retry_limit = 0;
      const ApplicationValidation post_publication_validation =
          ValidateApplicationOptions(post_publication);
      const ApplicationValidation post_publication_writer_validation =
          ValidateApplicationOptions(SplitWriter(post_publication));
      passed = Check(
                   post_publication_validation.accepted() &&
                       post_publication_validation.config()->injection() ==
                           post_publication.injection &&
                       post_publication_validation.config()->retry_limit() ==
                           0 &&
                       post_publication_writer_validation.rejection() ==
                           ApplicationRejection::invalid_run_role &&
                       canonical_writer_hash_matches(post_publication),
                   "scalar_cuda_step20_postpublication_boundary_contract") &&
          passed;
    }

    ApplicationOptionsView malformed = ScalarIgr(item.first);
    malformed.injection = {
        FailureInjectionModeV1::prescribed,
        FailurePhase::AfterRecoveryBeforeCommit,
        scalar_igr_sc25_v3_checkpoint_step, rank_count};
    passed = Check(ValidateApplicationOptions(malformed).rejection() ==
                       ApplicationRejection::invalid_run_role,
                   "scalar_cuda_rejects_out_of_range_scheduled_source") &&
        passed;
    malformed.injection.source_rank = 0;
    malformed.retry_limit = 0;
    passed = Check(ValidateApplicationOptions(malformed).rejection() ==
                       ApplicationRejection::invalid_run_role,
                   "scalar_cuda_rejects_terminal_step10_controls") && passed;
    malformed.injection.target_step = scalar_igr_sc25_v3_steps;
    malformed.retry_limit = scalar_igr_sc25_v3_retry_limit;
    passed = Check(ValidateApplicationOptions(malformed).rejection() ==
                       ApplicationRejection::invalid_run_role,
                   "scalar_cuda_rejects_retrying_step20_controls") && passed;
    malformed.injection.phase = FailurePhase::CheckpointStaging;
    malformed.retry_limit = 0;
    passed = Check(ValidateApplicationOptions(malformed).rejection() ==
                       ApplicationRejection::invalid_run_role,
                   "scalar_cuda_rejects_nonpublication_step20_phase") &&
        passed;
  }

  ApplicationOptionsView h1 = ScalarIgr(CpuSerialBackendTag::id);
  h1.interior_space = InteriorSpace::h1;
  passed = Check(ValidateApplicationOptions(h1).rejection() ==
                     ApplicationRejection::unsupported_h1,
                 "scalar_igr_h1_honest_rejection") && passed;

#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU == 0
  for (const std::string_view backend :
       {CpuSerialBackendTag::id, CpuRank2BackendTag::id})
  {
    const ApplicationOptionsView scalar_reference = ScalarIgr(backend);
    ApplicationOptionsView malformed_writer = SplitWriter(scalar_reference);
    malformed_writer.injection = {};
    malformed_writer.retry_limit = 0;
    ApplicationOptionsView malformed_reader = SplitReader(scalar_reference);
    malformed_reader.injection = {
        FailureInjectionModeV1::prescribed,
        smooth_periodic_euler_v1_failure_phase,
        smooth_periodic_euler_v1_functional_failure_target_step,
        SourceRank(backend)};
    malformed_reader.retry_limit = smooth_periodic_euler_v1_retry_limit;
    for (const ApplicationOptionsView& request :
         {SplitWriter(scalar_reference), SplitReader(scalar_reference),
          malformed_writer, malformed_reader})
    {
      passed = Check(
                   ValidateApplicationOptions(request).rejection() ==
                       ApplicationRejection::unbuilt_backend,
                   "scalar_igr_unbuilt_cpu_roles_reject_explicitly") &&
          passed;
    }
  }
  return passed;
#else
  const ApplicationOptionsView scalar_reference =
      ScalarIgr(CpuSerialBackendTag::id);
  passed = Check(
               ValidateApplicationOptions(
                   SplitWriter(scalar_reference)).accepted() &&
                   ValidateApplicationOptions(
                       SplitReader(scalar_reference)).accepted(),
               "scalar_igr_writer_reader_role_matrix") && passed;

  ApplicationOptionsView scalar_with_conventional_reference_controls =
      scalar_reference;
  scalar_with_conventional_reference_controls.injection = {};
  scalar_with_conventional_reference_controls.retry_limit = 0;
  passed = Check(
               ValidateApplicationOptions(
                   scalar_with_conventional_reference_controls).rejection() ==
                   ApplicationRejection::invalid_run_role,
               "scalar_igr_rejects_conventional_reference_controls") &&
      passed;

  ApplicationOptionsView scalar_with_conventional_reader_controls =
      SplitReader(scalar_reference);
  scalar_with_conventional_reader_controls.injection = {
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      smooth_periodic_euler_v1_functional_failure_target_step,
      0};
  scalar_with_conventional_reader_controls.retry_limit =
      smooth_periodic_euler_v1_retry_limit;
  passed = Check(
               ValidateApplicationOptions(
                   scalar_with_conventional_reader_controls).rejection() ==
                   ApplicationRejection::invalid_run_role,
               "scalar_igr_rejects_conventional_reader_controls") && passed;

  ApplicationOptionsView conventional_with_scalar_writer_controls =
      SplitWriter(Functional(CpuSerialBackendTag::id));
  conventional_with_scalar_writer_controls.injection =
      scalar_reference.injection;
  conventional_with_scalar_writer_controls.retry_limit =
      scalar_igr_sc25_v3_retry_limit;
  passed = Check(
               ValidateApplicationOptions(
                   conventional_with_scalar_writer_controls).rejection() ==
                   ApplicationRejection::invalid_run_role,
               "conventional_rejects_scalar_writer_controls") && passed;

  ApplicationOptionsView conventional_with_scalar_reader_controls =
      SplitReader(Functional(CpuSerialBackendTag::id));
  conventional_with_scalar_reader_controls.injection = {};
  conventional_with_scalar_reader_controls.retry_limit = 0;
  passed = Check(
               ValidateApplicationOptions(
                   conventional_with_scalar_reader_controls).rejection() ==
                   ApplicationRejection::invalid_run_role,
               "conventional_rejects_scalar_reader_controls") && passed;
  return passed;
#endif
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckExactAcceptedFieldsAndHashes() && passed;
  passed = CheckW8cSmoothTransitionClosure() && passed;
  passed = CheckRoleBackendMatrix() && passed;
  passed = CheckFocusedAndAlternateRoles() && passed;
  passed = CheckNegativeRoleControls() && passed;
  passed = CheckScientificCrossProducts() && passed;
  passed = CheckCardScenarioTupleClosure() && passed;
  passed = CheckMismatchedScenarioRoleFamilies() && passed;
  passed = CheckStaticEntryOffDiagonalMatrix() && passed;
  passed = CheckCoherentScalarIdentityEquivalence() && passed;
  passed = CheckScalarIgrMatrix() && passed;
  if (passed)
  {
    std::cout << "HDG W8A typed application admission checks passed\n";
    return 0;
  }
  return 1;
}

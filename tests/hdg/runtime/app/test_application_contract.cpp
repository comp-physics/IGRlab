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
                  !std::is_aggregate<ValidatedApplicationConfig>::value,
              "accepted configuration has no invalid construction path");
static_assert(IsFrozenApplicationBackend<CpuSerialBackendTag>::value &&
                  IsFrozenApplicationBackend<CpuRank2BackendTag>::value &&
                  IsFrozenApplicationBackend<
                      CudaRank1OneGpuBackendTag>::value &&
                  IsFrozenApplicationBackend<
                      CudaRank2OneGpuHostStagedBackendTag>::value &&
                  !IsFrozenApplicationBackend<int>::value,
              "the static request boundary recognizes exactly four backends");
static_assert(smooth_periodic_euler_v1_smoke_polynomial_order == 1 &&
                  smooth_periodic_euler_v1_smoke_mesh_n == 3 &&
                  smooth_periodic_euler_v1_smoke_steps == 3 &&
                  smooth_periodic_euler_v1_smoke_checkpoint_step == 1 &&
                  smooth_periodic_euler_v1_smoke_failure_target_step == 2 &&
                  smooth_periodic_euler_v1_functional_polynomial_order == 2 &&
                  smooth_periodic_euler_v1_functional_mesh_n == 6 &&
                  smooth_periodic_euler_v1_functional_steps == 20 &&
                  smooth_periodic_euler_v1_functional_checkpoint_step == 10 &&
                  smooth_periodic_euler_v1_functional_failure_target_step == 14,
              "the two card scenarios remain distinct and exact");

bool Check(const bool condition, const char* name)
{
  if (!condition) { std::cerr << "FAILED: " << name << '\n'; }
  return condition;
}

StateHash FilledHash(const std::uint8_t first) noexcept
{
  StateHash value{};
  for (std::size_t index = 0; index < value.bytes.size(); ++index)
  {
    value.bytes[index] = static_cast<std::uint8_t>(first + index);
  }
  return value;
}

ApplicationOptionsView Smoke(const std::string_view backend)
{
  ApplicationOptionsView options;
  options.backend_id = backend;
  options.scientific_config_sha256 = FilledHash(0x10);
  options.run_config_sha256 = FilledHash(0x40);
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
  options.checkpoint_step =
      smooth_periodic_euler_v1_functional_checkpoint_step;
  options.checkpoint_id =
      smooth_periodic_euler_v1_functional_checkpoint_id;
  options.injected_failure_committed_step =
      smooth_periodic_euler_v1_functional_failure_committed_step;
  options.injected_failure_target_step =
      smooth_periodic_euler_v1_functional_failure_target_step;
  options.seed = smooth_periodic_euler_v1_functional_seed;
  return options;
}

bool CheckAcceptedExactFields()
{
  const ApplicationValidation validation =
      ValidateApplicationOptions(Smoke(CpuSerialBackendTag::id));
  const ValidatedApplicationConfig* config = validation.config();
  bool passed = Check(validation.accepted() && config != nullptr &&
                          validation.rejection() == ApplicationRejection::none,
                      "accepted_result_exposes_config_only_on_success");
  if (config == nullptr) { return false; }
  passed = Check(
               config->scenario() == ApplicationScenarioV1::smoke &&
                   config->backend() == ApplicationBackend::cpu_serial &&
                   config->tuple() == CpuSerialRuntimeSelection::Descriptor() &&
                   config->card_id().view() ==
                       smooth_periodic_euler_v1_card_id &&
                   config->case_id().view() ==
                       smooth_periodic_euler_v1_case_id &&
                   config->scenario_id().view() ==
                       smooth_periodic_euler_v1_smoke_id &&
                   config->checkpoint_id().view() ==
                       smooth_periodic_euler_v1_smoke_checkpoint_id &&
                   config->mesh_family_id().view() ==
                       smooth_periodic_euler_v1_mesh_family_id &&
                   config->source_id().view() ==
                       smooth_periodic_euler_v1_source_id &&
                   config->flux_id().view() ==
                       smooth_periodic_euler_v1_flux_id &&
                   config->time_rule_id().view() ==
                       smooth_periodic_euler_v1_time_rule_id &&
                   config->layout_id().view() ==
                       smooth_periodic_euler_v1_layout_id &&
                   config->interior_space() == InteriorSpace::l2 &&
                   config->polynomial_order() == 1 && config->mesh_n() == 3 &&
                   config->q_a() == 11 && config->q_e() == 16 &&
                   config->q_e4() == 20 && config->q_o() == 21 &&
                   config->gamma() == 1.4 && config->tau() == 2.5 &&
                   config->dt() == 0.05 && config->accepted_steps() == 3 &&
                   config->final_time() == 0.15 &&
                   config->output_steps() ==
                       std::array<std::uint64_t, 3>{{0, 1, 3}} &&
                   config->checkpoint_step() == 1 &&
                   config->restart_mode() ==
                       RestartMode::same_backend_rank_partition &&
                   config->restart_trajectory() ==
                       RestartTrajectoryModeV1::
                           uninterrupted_and_split_same_tuple &&
                   config->injected_failure_committed_step() == 1 &&
                   config->injected_failure_target_step() == 2 &&
                   config->injected_failure_phase() ==
                       FailurePhase::AfterRecoveryBeforeCommit &&
                   config->rollback_obligation() ==
                       RollbackObligationV1::
                           collective_exact_complete_state &&
                   config->source_rank_rule() ==
                       DeferredSourceRankRuleV1::
                           rank_zero_for_rank1_rank_one_for_rank2 &&
                   config->retry_limit() == 1 &&
                   config->seed() == UINT64_C(0x573841534d4f4b31) &&
                   config->scientific_config_sha256() == FilledHash(0x10) &&
                   config->run_config_sha256() == FilledHash(0x40) &&
                   config->deferred_runtime().required_mpi_rank_count == 1 &&
                   config->deferred_runtime().gpu_rule ==
                       DeferredGpuRuleV1::no_cuda_device &&
                   config->deferred_runtime().required_injected_source_rank ==
                       0,
               "accepted_smoke_copies_every_exact_identity_and_input") &&
      passed;

  ApplicationOptionsView rejected = Smoke(CpuSerialBackendTag::id);
  rejected.mesh_n = 6;
  const ApplicationValidation rejection =
      ValidateApplicationOptions(rejected);
  passed = Check(!rejection.accepted() && rejection.config() == nullptr &&
                     rejection.rejection() ==
                         ApplicationRejection::invalid_polynomial_mesh_pair,
                 "rejection_exposes_no_invalid_default_config") &&
      passed;
  return passed;
}

bool CheckTwoScenarioBackendMatrix()
{
  struct BackendCase
  {
    std::string_view id;
    ApplicationBackend backend;
    std::uint32_t ranks;
    DeferredGpuRuleV1 gpu_rule;
    std::uint32_t source_rank;
  };
  const std::array<BackendCase, 4> cases{{
      {CpuSerialBackendTag::id, ApplicationBackend::cpu_serial, 1,
       DeferredGpuRuleV1::no_cuda_device, 0},
      {CpuRank2BackendTag::id, ApplicationBackend::cpu_rank2, 2,
       DeferredGpuRuleV1::no_cuda_device, 1},
      {CudaRank1OneGpuBackendTag::id,
       ApplicationBackend::cuda_rank1_one_gpu, 1,
       DeferredGpuRuleV1::one_physical_gpu, 0},
      {CudaRank2OneGpuHostStagedBackendTag::id,
       ApplicationBackend::cuda_rank2_one_gpu_host_staged, 2,
       DeferredGpuRuleV1::one_shared_physical_gpu, 1}}};
  bool passed = true;
  for (const BackendCase& item : cases)
  {
    for (const ApplicationOptionsView request :
         {Smoke(item.id), Functional(item.id)})
    {
      const ApplicationValidation validation =
          ValidateApplicationOptions(request);
      if (CompiledBackendAvailable(item.backend))
      {
        const ValidatedApplicationConfig* config = validation.config();
        passed = Check(
                     config != nullptr && config->backend() == item.backend &&
                         config->deferred_runtime().required_mpi_rank_count ==
                             item.ranks &&
                         config->deferred_runtime().gpu_rule == item.gpu_rule &&
                         config->deferred_runtime().
                                 required_injected_source_rank ==
                             item.source_rank,
                     "smoke_and_principal_functional_each_admit_four_tuples") &&
            passed;
      }
      else
      {
        passed = Check(!validation.accepted() &&
                           validation.rejection() ==
                               ApplicationRejection::unbuilt_backend,
                       "known_unbuilt_backend_rejected_from_build_definition") &&
            passed;
      }
    }
  }
  return passed;
}

bool CheckNoScenarioCrossProducts()
{
  bool passed = true;
  for (const std::uint32_t mesh : {3U, 6U, 12U})
  {
    passed = Check(ValidateApplicationOptions(
                       Functional(CpuSerialBackendTag::id, mesh)).accepted(),
                   "serial_functional_convergence_ladder_admitted") &&
        passed;
  }
  ApplicationOptionsView request = Smoke(CpuSerialBackendTag::id);
  request.mesh_n = 6;
  passed = Check(ValidateApplicationOptions(request).rejection() ==
                     ApplicationRejection::invalid_polynomial_mesh_pair,
                 "p1_N6_cross_product_rejected") &&
      passed;
  request = Smoke(CpuSerialBackendTag::id);
  request.accepted_steps = 4;
  passed = Check(ValidateApplicationOptions(request).rejection() ==
                     ApplicationRejection::invalid_step_count,
                 "smoke_steps_above_exact_bound_rejected") &&
      passed;
  request = Functional(CpuSerialBackendTag::id);
  request.accepted_steps = 21;
  passed = Check(ValidateApplicationOptions(request).rejection() ==
                     ApplicationRejection::invalid_step_count,
                 "functional_steps_above_exact_bound_rejected") &&
      passed;
  request = Functional(CpuRank2BackendTag::id, 3);
  passed = Check(ValidateApplicationOptions(request).rejection() ==
                     ApplicationRejection::invalid_polynomial_mesh_pair,
                 "rank2_cannot_bind_serial_N3_ladder_point") &&
      passed;
  if (CompiledBackendAvailable(ApplicationBackend::cuda_rank1_one_gpu))
  {
    request = Functional(CudaRank1OneGpuBackendTag::id, 12);
    passed = Check(ValidateApplicationOptions(request).rejection() ==
                       ApplicationRejection::invalid_polynomial_mesh_pair,
                   "CUDA_cannot_bind_serial_N12_ladder_point") &&
        passed;
  }
  return passed;
}

bool CheckExactMutationRejections()
{
  bool passed = true;
  const auto expect = [&](ApplicationOptionsView request,
                          const ApplicationRejection expected,
                          const char* name) {
    const ApplicationValidation result = ValidateApplicationOptions(request);
    passed = Check(!result.accepted() && result.config() == nullptr &&
                       result.rejection() == expected,
                   name) &&
        passed;
  };
  ApplicationOptionsView request = Smoke(CpuSerialBackendTag::id);
  request.card_id = "wrong-card";
  expect(request, ApplicationRejection::unknown_card, "card_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.case_id = "wrong-case";
  expect(request, ApplicationRejection::unknown_case, "case_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.scenario_id = "wrong-scenario";
  expect(request, ApplicationRejection::unknown_scenario, "scenario_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.interior_space = InteriorSpace::h1;
  expect(request, ApplicationRejection::unsupported_h1, "H1_explicit");
  request = Smoke(CpuSerialBackendTag::id);
  request.formulation_id = "wrong-tuple";
  expect(request, ApplicationRejection::unknown_tuple, "tuple_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.backend_id = "wrong-backend";
  expect(request, ApplicationRejection::unknown_backend, "backend_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.source_id = "wrong-source";
  expect(request, ApplicationRejection::invalid_scientific_identity,
         "source_flux_time_layout_identities_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.scientific_config_sha256 = {};
  expect(request, ApplicationRejection::invalid_config_hash,
         "scientific_and_run_hash_inputs_required");
  request = Smoke(CpuSerialBackendTag::id);
  request.q_o = 22;
  expect(request, ApplicationRejection::invalid_quadrature,
         "scenario_quadratures_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.dt = 0.1;
  expect(request, ApplicationRejection::invalid_dt, "dt_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.final_time = 0.2;
  expect(request, ApplicationRejection::invalid_final_time,
         "final_time_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.output_steps = {{0, 2, 3}};
  expect(request, ApplicationRejection::invalid_output_schedule,
         "output_schedule_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.checkpoint_step = 2;
  expect(request, ApplicationRejection::invalid_checkpoint_schedule,
         "checkpoint_step_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.checkpoint_id = "wrong-checkpoint";
  expect(request, ApplicationRejection::invalid_checkpoint_identity,
         "checkpoint_identity_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.restart_mode = static_cast<RestartMode>(255);
  expect(request, ApplicationRejection::invalid_restart_mode,
         "restart_mode_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.injected_failure_committed_step = 0;
  expect(request, ApplicationRejection::invalid_failure_injection,
         "failure_commit_target_pair_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.injected_failure_phase = FailurePhase::Recovery;
  expect(request, ApplicationRejection::invalid_failure_phase,
         "failure_phase_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.source_rank_rule = static_cast<DeferredSourceRankRuleV1>(255);
  expect(request, ApplicationRejection::invalid_source_rank_rule,
         "deferred_source_rule_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.retry_limit = 2;
  expect(request, ApplicationRejection::invalid_retry_limit,
         "one_retry_bound");
  request = Smoke(CpuSerialBackendTag::id);
  request.seed = 0;
  expect(request, ApplicationRejection::invalid_seed, "scenario_seed_bound");
  return passed;
}

} // namespace

int main()
{
  bool passed = true;
  passed = CheckAcceptedExactFields() && passed;
  passed = CheckTwoScenarioBackendMatrix() && passed;
  passed = CheckNoScenarioCrossProducts() && passed;
  passed = CheckExactMutationRejections() && passed;
  if (passed)
  {
    std::cout << "HDG W8A static application admission checks passed\n";
    return 0;
  }
  return 1;
}

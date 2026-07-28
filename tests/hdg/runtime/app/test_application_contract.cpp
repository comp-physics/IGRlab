#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace
{

using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::app;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "application contract test must compile in strict C++17");
static_assert(smooth_periodic_euler_v1_polynomial_order == 1 &&
                  smooth_periodic_euler_v1_dt == 0.05 &&
                  smooth_periodic_euler_v1_minimum_steps == 20 &&
                  smooth_periodic_euler_v1_checkpoint_step == 10 &&
                  smooth_periodic_euler_v1_failure_step == 14 &&
                  smooth_periodic_euler_v1_retry_limit == 1,
              "smooth periodic Euler v1 constants are frozen");
static_assert(IsFrozenApplicationBackend<CpuSerialBackendTag>::value &&
                  IsFrozenApplicationBackend<CpuRank2BackendTag>::value &&
                  IsFrozenApplicationBackend<
                      CudaRank1OneGpuBackendTag>::value &&
                  IsFrozenApplicationBackend<
                      CudaRank2OneGpuHostStagedBackendTag>::value,
              "the application boundary is closed over four backends");
static_assert(!IsFrozenApplicationBackend<int>::value,
              "unknown static backend is not an entry");

bool Check(const bool condition, const char* name)
{
  if (!condition) { std::cerr << "FAILED: " << name << '\n'; }
  return condition;
}

BuiltBackendSet AllBuilt() noexcept
{
  return {true, true, true, true};
}

bool CheckFrozenFourShapeTrajectory()
{
  struct Case
  {
    std::string_view backend;
    ApplicationBackend kind;
  };
  const std::array<Case, 4> cases{{
      {CpuSerialBackendTag::id, ApplicationBackend::cpu_serial},
      {CpuRank2BackendTag::id, ApplicationBackend::cpu_rank2},
      {CudaRank1OneGpuBackendTag::id,
       ApplicationBackend::cuda_rank1_one_gpu},
      {CudaRank2OneGpuHostStagedBackendTag::id,
       ApplicationBackend::cuda_rank2_one_gpu_host_staged}}};

  bool passed = true;
  for (const Case& item : cases)
  {
    ApplicationOptionsView options;
    options.backend_id = item.backend;
    options.mesh_n = 6;
    const ApplicationValidation validation =
        ValidateApplicationOptions(options, AllBuilt());
    passed = Check(validation.accepted &&
                       validation.rejection == ApplicationRejection::none &&
                       validation.config.backend == item.kind &&
                       validation.config.tuple.backend.view() == item.backend &&
                       validation.config.interior_space == InteriorSpace::l2 &&
                       validation.config.polynomial_order == 1 &&
                       validation.config.mesh_n == 6 &&
                       validation.config.step_bound >= 20 &&
                       validation.config.dt == 0.05 &&
                       validation.config.output_steps ==
                           std::array<std::uint64_t, 3>{{0, 10, 20}} &&
                       validation.config.checkpoint_step == 10 &&
                       validation.config.restart_mode ==
                           RestartMode::same_backend_rank_partition &&
                       validation.config.injected_failure_step == 14 &&
                       validation.config.retry_limit == 1 &&
                       IsSharedFourShapeMesh(validation.config.mesh_n),
                   "N6_four_shape_trajectory_contract") &&
        passed;
  }
  return passed;
}

bool CheckMeshMatrix()
{
  bool passed = true;
  for (const std::string_view backend :
       {CpuSerialBackendTag::id, CpuRank2BackendTag::id})
  {
    for (const std::uint32_t mesh : {3U, 6U, 12U})
    {
      ApplicationOptionsView options;
      options.backend_id = backend;
      options.mesh_n = mesh;
      passed = Check(ValidateApplicationOptions(options, AllBuilt()).accepted,
                     "CPU_mesh_3_6_12_admitted") &&
          passed;
    }
    ApplicationOptionsView rejected;
    rejected.backend_id = backend;
    rejected.mesh_n = 24;
    passed = Check(ValidateApplicationOptions(rejected, AllBuilt()).rejection ==
                       ApplicationRejection::invalid_mesh,
                   "CPU_other_mesh_rejected") &&
        passed;
  }
  for (const std::string_view backend :
       {CudaRank1OneGpuBackendTag::id,
        CudaRank2OneGpuHostStagedBackendTag::id})
  {
    for (const std::uint32_t mesh : {3U, 6U})
    {
      ApplicationOptionsView options;
      options.backend_id = backend;
      options.mesh_n = mesh;
      passed = Check(ValidateApplicationOptions(options, AllBuilt()).accepted,
                     "CUDA_mesh_3_6_admitted") &&
          passed;
    }
    ApplicationOptionsView rejected;
    rejected.backend_id = backend;
    rejected.mesh_n = 12;
    passed = Check(ValidateApplicationOptions(rejected, AllBuilt()).rejection ==
                       ApplicationRejection::invalid_mesh,
                   "CUDA_mesh_12_rejected") &&
        passed;
  }
  return passed;
}

bool CheckPreExecutionRejections()
{
  bool passed = true;
  ApplicationOptionsView options;
  options.interior_space = InteriorSpace::h1;
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::unsupported_h1,
                 "H1_rejected_pre_execution") &&
      passed;

  options = {};
  options.interior_space = static_cast<InteriorSpace>(255);
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::invalid_interior_space,
                 "unknown_interior_rejected_pre_execution") &&
      passed;

  options = {};
  options.formulation_id = "hdg.unknown";
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::unknown_tuple,
                 "unknown_tuple_rejected_pre_execution") &&
      passed;

  options = {};
  options.backend_id = "backend.unknown";
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::unknown_backend,
                 "unknown_backend_rejected_pre_execution") &&
      passed;

  options = {};
  passed = Check(ValidateApplicationOptions(options, {}).rejection ==
                     ApplicationRejection::unbuilt_backend,
                 "known_unbuilt_backend_rejected_pre_execution") &&
      passed;
  return passed;
}

bool CheckStrictScenarioValidation()
{
  bool passed = true;
  ApplicationOptionsView options;
  options.case_id = "case.unknown";
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::unknown_case,
                 "unknown_case_rejected") &&
      passed;

  options = {};
  options.polynomial_order = 2;
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::invalid_polynomial_order,
                 "non_p1_rejected") &&
      passed;

  options = {};
  options.step_bound = 19;
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::insufficient_step_bound,
                 "short_trajectory_rejected") &&
      passed;

  options = {};
  options.step_bound = 21;
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).accepted,
                 "trajectory_longer_than_twenty_is_admitted") &&
      passed;

  options = {};
  options.dt = 0.1;
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::invalid_dt,
                 "non_frozen_dt_rejected") &&
      passed;

  options = {};
  options.output_steps = {{0, 5, 20}};
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::invalid_output_schedule,
                 "non_frozen_output_schedule_rejected") &&
      passed;

  options = {};
  options.checkpoint_step = 9;
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::invalid_checkpoint_schedule,
                 "non_step10_checkpoint_rejected") &&
      passed;

  options = {};
  options.restart_mode = static_cast<RestartMode>(255);
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::invalid_restart_mode,
                 "non_same_tuple_restart_rejected") &&
      passed;

  options = {};
  options.injected_failure_step = 13;
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::invalid_failure_injection,
                 "non_step14_injection_rejected") &&
      passed;

  options = {};
  options.retry_limit = 2;
  passed = Check(ValidateApplicationOptions(options, AllBuilt()).rejection ==
                     ApplicationRejection::invalid_retry_limit,
                 "more_than_one_retry_rejected") &&
      passed;
  return passed;
}

} // namespace

int main()
{
  const bool passed = CheckFrozenFourShapeTrajectory() && CheckMeshMatrix() &&
      CheckPreExecutionRejections() && CheckStrictScenarioValidation();
  if (passed)
  {
    std::cout << "HDG W8A application contract checks passed\n";
    return 0;
  }
  return 1;
}

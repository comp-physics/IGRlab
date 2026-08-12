#include "numerics/hdg/runtime/app/solver.hpp"

#include "solver_entry_detail.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>

namespace test_detail
{

struct Calls final
{
  std::array<unsigned, 4> downstream{};
  unsigned invalid_context = 0;
  std::string_view card{};
};

Calls calls{};

dg::hdg::runtime::app::ApplicationRunResult Accepted(
    const std::uint64_t identity) noexcept
{
  using namespace dg::hdg::runtime;
  using namespace dg::hdg::runtime::app;
  ApplicationRunResult result{};
  result.status = RunStatus::committed;
  result.reason = FailureReason::none;
  result.coordinates.step = identity;
  result.failure_class = ApplicationRunResult::FailureClass::none;
  return result;
}

template <typename Context>
void ObserveContext(const Context& context) noexcept
{
  if (context.communicator() != MPI_COMM_NULL || context.rank() != -1 ||
      context.ranks() != 0)
    ++calls.invalid_context;
}

} // namespace test_detail

namespace dg::hdg::runtime::app::detail
{

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CpuSerialBackendTag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  ++test_detail::calls.downstream[0];
  test_detail::calls.card = request.card_id;
  test_detail::ObserveContext(context);
  return test_detail::Accepted(101);
}

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CpuRank2BackendTag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  ++test_detail::calls.downstream[1];
  test_detail::calls.card = request.card_id;
  test_detail::ObserveContext(context);
  return test_detail::Accepted(102);
}

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CudaRank1OneGpuBackendTag, const ApplicationOptionsView& request,
    CudaApplicationContextV1& context)
{
  ++test_detail::calls.downstream[2];
  test_detail::calls.card = request.card_id;
  test_detail::ObserveContext(context);
  return test_detail::Accepted(103);
}

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CudaRank2OneGpuHostStagedBackendTag,
    const ApplicationOptionsView& request,
    CudaApplicationContextV1& context)
{
  ++test_detail::calls.downstream[3];
  test_detail::calls.card = request.card_id;
  test_detail::ObserveContext(context);
  return test_detail::Accepted(104);
}

} // namespace dg::hdg::runtime::app::detail

namespace
{

using namespace dg::hdg::runtime;
using namespace dg::hdg::runtime::app;

static_assert(__cplusplus == DG_IGR_HDG_RUNTIME_EXPECTED_CPLUSPLUS,
              "static entry selector test must compile in strict C++17");
static_assert(DG_IGR_HDG_APP_HAS_CUDA_RANK1 == 1 &&
                  DG_IGR_HDG_APP_HAS_CUDA_RANK2 == 1,
              "the complete selector oracle is registered only when all four entries exist");

enum class Entry : std::uint8_t
{
  cpu_rank1,
  cpu_rank2,
  cuda_rank1,
  cuda_rank2
};

struct Row final
{
  Entry entry;
  std::string_view request_backend;
  bool scalar;
  bool diagonal;
  std::size_t expected_downstream;
  std::uint64_t expected_identity;
};

struct Invocation final
{
  ApplicationRunResult result{};
  std::string diagnostic;
};

bool Check(const bool condition, const char* name)
{
  if (!condition) std::cerr << "FAILED: " << name << '\n';
  return condition;
}

bool Absent(const std::string& path) noexcept
{
  std::error_code error;
  const bool present = std::filesystem::exists(path, error);
  return !error && !present;
}

ApplicationOptionsView Request(const Row& row, const std::string& output,
                               const std::string& checkpoint)
{
  ApplicationOptionsView request;
  if (row.scalar)
  {
    request.card_id = scalar_igr_sc25_v3_card_id;
    request.case_id = scalar_igr_sc25_v3_case_id;
    request.scenario_id = scalar_igr_sc25_v3_scenario_id;
    request.model_id = TracedScalarSc25ProjectionV3Tag::id;
    request.formulation_id = FullyHybridizedScalarSc25V3Tag::id;
    request.trace_id = DualConservativeScalarTraceV1Tag::id;
    request.stabilization_id = W7Tau25DensityHeightP2V3Tag::id;
    request.time_treatment_id = BackwardEulerMonolithicV1Tag::id;
    request.source_id = scalar_igr_sc25_v3_source_id;
    request.flux_id = scalar_igr_sc25_v3_flux_id;
    request.time_rule_id = BackwardEulerMonolithicV1Tag::id;
    request.layout_id = scalar_igr_sc25_v3_layout_id;
    request.polynomial_order = scalar_igr_sc25_v3_polynomial_order;
    request.mesh_n = scalar_igr_sc25_v3_mesh_n;
    request.q_a = scalar_igr_sc25_v3_q_a;
    request.q_e = scalar_igr_sc25_v3_q_e;
    request.q_e4 = scalar_igr_sc25_v3_q_e4;
    request.q_o = scalar_igr_sc25_v3_q_o;
    request.dt = scalar_igr_sc25_v3_dt;
    request.accepted_steps = scalar_igr_sc25_v3_steps;
    request.final_time = scalar_igr_sc25_v3_final_time;
    request.output_steps = scalar_igr_sc25_v3_output_steps;
    request.checkpoint_id = scalar_igr_sc25_v3_checkpoint_id;
    request.checkpoint_step = OptionalCheckpointStepV1::Present(
        scalar_igr_sc25_v3_checkpoint_step);
    request.injection = {
        FailureInjectionModeV1::prescribed,
        smooth_periodic_euler_v1_failure_phase,
        scalar_igr_sc25_v3_failure_target_step,
        row.request_backend == CpuRank2BackendTag::id ||
                row.request_backend ==
                    CudaRank2OneGpuHostStagedBackendTag::id
            ? 1U : 0U};
    request.retry_limit = scalar_igr_sc25_v3_retry_limit;
    request.seed = scalar_igr_sc25_v3_seed;
  }
  request.backend_id = row.request_backend;
  request.device_id = row.request_backend == CpuSerialBackendTag::id ||
          row.request_backend == CpuRank2BackendTag::id
      ? -1 : 0;
  request.output_directory = OutputDirectoryPathV1::Present(output);
  request.checkpoint_out = CheckpointArtifactPathV1::Present(checkpoint);
  return request;
}

Invocation Invoke(const Entry entry, const ApplicationOptionsView& request)
{
  constexpr const char* diagnostic = "static-entry-sentinel";
  switch (entry)
  {
    case Entry::cpu_rank1:
    {
      MpiApplicationContextV1 context(MPI_COMM_NULL);
      context.SetDiagnostic(diagnostic);
      return {RunSmoothPeriodicEulerV1(CpuSerialBackendTag{}, request,
                                       context),
              context.diagnostic()};
    }
    case Entry::cpu_rank2:
    {
      MpiApplicationContextV1 context(MPI_COMM_NULL);
      context.SetDiagnostic(diagnostic);
      return {RunSmoothPeriodicEulerV1(CpuRank2BackendTag{}, request,
                                       context),
              context.diagnostic()};
    }
    case Entry::cuda_rank1:
    {
      CudaApplicationContextV1 context(MPI_COMM_NULL);
      context.SetDiagnostic(diagnostic);
      return {RunSmoothPeriodicEulerV1(CudaRank1OneGpuBackendTag{}, request,
                                       context),
              context.diagnostic()};
    }
    case Entry::cuda_rank2:
    {
      CudaApplicationContextV1 context(MPI_COMM_NULL);
      context.SetDiagnostic(diagnostic);
      return {RunSmoothPeriodicEulerV1(
                  CudaRank2OneGpuHostStagedBackendTag{}, request, context),
              context.diagnostic()};
    }
  }
  return {};
}

bool ExactRejection(const ApplicationRunResult& result) noexcept
{
  return result.status == RunStatus::invalid_request &&
      result.reason == FailureReason::invalid_resource_descriptor &&
      result.coordinates.time == 0.0 && result.coordinates.step == 0 &&
      result.coordinates.generation == 0 && result.coordinates.cursor == 0 &&
      result.coordinates.dt == 0.0 && result.retries == 0 &&
      result.failure_class == ApplicationRunResult::FailureClass::configuration;
}

} // namespace

int main()
{
  const std::array<Row, 16> rows{{
      {Entry::cpu_rank1, CpuSerialBackendTag::id, false, true, 0, 101},
      {Entry::cpu_rank1, CpuSerialBackendTag::id, true, true, 0, 101},
      {Entry::cpu_rank2, CpuRank2BackendTag::id, false, true, 1, 102},
      {Entry::cpu_rank2, CpuRank2BackendTag::id, true, true, 1, 102},
      {Entry::cuda_rank1, CudaRank1OneGpuBackendTag::id, false, true, 2, 103},
      {Entry::cuda_rank1, CudaRank1OneGpuBackendTag::id, true, true, 2, 103},
      {Entry::cuda_rank2, CudaRank2OneGpuHostStagedBackendTag::id, false,
       true, 3, 104},
      {Entry::cuda_rank2, CudaRank2OneGpuHostStagedBackendTag::id, true,
       true, 3, 104},
      {Entry::cpu_rank1, CudaRank1OneGpuBackendTag::id, false, false, 0, 0},
      {Entry::cpu_rank1, CudaRank1OneGpuBackendTag::id, true, false, 0, 0},
      {Entry::cuda_rank1, CpuSerialBackendTag::id, false, false, 0, 0},
      {Entry::cuda_rank1, CpuSerialBackendTag::id, true, false, 0, 0},
      {Entry::cpu_rank2, CudaRank2OneGpuHostStagedBackendTag::id, false,
       false, 0, 0},
      {Entry::cpu_rank2, CudaRank2OneGpuHostStagedBackendTag::id, true,
       false, 0, 0},
      {Entry::cuda_rank2, CpuRank2BackendTag::id, false, false, 0, 0},
      {Entry::cuda_rank2, CpuRank2BackendTag::id, true, false, 0, 0}}};

  bool passed = true;
  std::size_t diagonal_count = 0;
  std::size_t off_diagonal_count = 0;
  for (std::size_t index = 0; index < rows.size(); ++index)
  {
    const Row& row = rows[index];
    const std::string stem = "/tmp/dg-igr-hdg-static-entry-" +
        std::to_string(static_cast<long long>(::getpid())) + "-" +
        std::to_string(index);
    const std::string checkpoint = stem + ".checkpoint";
    const bool initially_absent = Absent(stem) && Absent(checkpoint);
    test_detail::calls = {};
    const ApplicationOptionsView request = Request(row, stem, checkpoint);
    const Invocation invocation = Invoke(row.entry, request);
    const bool no_artifacts = Absent(stem) && Absent(checkpoint);
    const unsigned total_calls = test_detail::calls.downstream[0] +
        test_detail::calls.downstream[1] +
        test_detail::calls.downstream[2] +
        test_detail::calls.downstream[3];
    bool row_passed = initially_absent && no_artifacts &&
        invocation.diagnostic == "static-entry-sentinel" &&
        test_detail::calls.invalid_context == 0;
    if (row.diagonal)
    {
      ++diagonal_count;
      row_passed = row_passed && total_calls == 1 &&
          test_detail::calls.downstream[row.expected_downstream] == 1 &&
          test_detail::calls.card ==
              (row.scalar ? scalar_igr_sc25_v3_card_id
                          : smooth_periodic_euler_v1_card_id) &&
          invocation.result.status == RunStatus::committed &&
          invocation.result.reason == FailureReason::none &&
          invocation.result.coordinates.step == row.expected_identity &&
          invocation.result.failure_class ==
              ApplicationRunResult::FailureClass::none;
    }
    else
    {
      ++off_diagonal_count;
      row_passed = row_passed && total_calls == 0 &&
          test_detail::calls.card.empty() &&
          ExactRejection(invocation.result);
    }
    passed = Check(row_passed, row.diagonal
                                  ? "static_entry_diagonal_control"
                                  : "static_entry_off_diagonal_rejection") &&
        passed;
  }
  passed = Check(diagonal_count == 8 && off_diagonal_count == 8,
                 "static_entry_exact_16_row_census") && passed;

  const Row& seam_row = rows[0];
  const std::string seam_stem = "/tmp/dg-igr-hdg-method-seam-" +
      std::to_string(static_cast<long long>(::getpid()));
  const ApplicationOptionsView seam_request =
      Request(seam_row, seam_stem, seam_stem + ".checkpoint");
  const ApplicationValidation seam_validation =
      ValidateApplicationOptions(seam_request);
  MpiApplicationContextV1 seam_context(MPI_COMM_NULL);
  test_detail::calls = {};
  const HdgMethodResultV1 seam_result = RunHdgMethodV1(
      {&seam_request, seam_validation.config(), &seam_context, nullptr});
  passed = Check(seam_validation.accepted() && seam_result.capabilities.supported &&
                     seam_result.checkpoint_identity.mesh_n == seam_request.mesh_n &&
                     seam_result.run.status == RunStatus::committed &&
                     test_detail::calls.downstream[0] == 1 &&
                     test_detail::calls.invalid_context == 0 &&
                     Absent(seam_stem) && Absent(seam_stem + ".checkpoint"),
                 "typed_hdg_method_bridge_uses_existing_static_entry") && passed;
  if (passed)
    std::cout << "HDG static entry selector 16-row oracle PASS\n";
  return passed ? 0 : 1;
}

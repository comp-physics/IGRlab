#include "igrlab/compat.hpp"
#include "sim/hdg/main_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler_app.cuh"
#include "sim/context/case_context.hpp"
#include "sim/hdg/euler_output_fields.hpp"
#include "sim/hdg/parallel_vtk.hpp"
#include "sim/hdg/regularization_scaling.hpp"
#include "sim/hdg/resolved_euler_case.hpp"
#include "sim/hdg/state_io.hpp"
#include "sim/hdg/transactional_time_adapter.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/wiring/hdg_boundary_wiring.hpp"

#if defined(__GNUC__) || defined(__clang__)
// Headerless test seam: ordinary executables provide no definition, while the
// split-process fixture may inject one deterministic prepare-boundary abort.
// This is neither a production option nor persisted numerical identity.
extern "C" bool dg_igr_hdg_cuda_prepare_boundary_failure_for_test(
    std::uint64_t committed_steps, double committed_time) noexcept
    __attribute__((weak));
#endif

namespace dg_sim::hdg
{
namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::BackendSelection;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerHdgPositivityLimiterKind;
using dg::hdg::generic::EulerHdgPositivityLimiterPolicy;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::EulerHdgTimeTreatment;
using dg::hdg::generic::EulerHdgZhangShuPointSet;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::generic::RegularizationSelection;
using dg::hdg::runtime::cuda::GenericEulerCudaCheckpointPayloadViewV1;
using dg::hdg::runtime::cuda::GenericEulerCudaMutableCheckpointPayloadV1;
using dg::hdg::runtime::cuda::GenericEulerCudaPrepareFailureClass;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using GenericEulerHdgCudaSession =
    dg::hdg::runtime::cuda::GenericEulerHdgCudaAppSession;
using dg::hdg::runtime::cuda_w8a::CudaIluAccountingKind;
using dg::hdg::runtime::cuda_w8a::CudaIluAccountingKindName;
using dg::hdg::runtime::cuda_w8a::CudaIluApplicationIterations;
using dg::hdg::runtime::cuda_w8a::CudaIluFactorReusePolicyName;
using dg::hdg::runtime::cuda_w8a::CudaIluLevelOfFill;
using dg::hdg::runtime::cuda_w8a::CudaIluLocalReordering;
using dg::hdg::runtime::cuda_w8a::CudaIluTriangularSolveKindName;
using dg::hdg::runtime::cuda_w8a::CudaIluType;
using dg::hdg::runtime::cuda_w8a::CudaPreparedOutputMetadataV1;
using dg::hdg::runtime::cuda_w8a::CudaStepEvidence;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyDescriptor;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyDescriptorIsExact;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKind;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKindName;
using dg::hdg::runtime::cuda_w8a::CudaTracePreconditionerIsIlu;
using dg::hdg::runtime::cuda_w8a::CudaTracePreconditionerName;
using dg::hdg::runtime::cuda_w8a::DefaultCudaTracePolicyRequest;
using dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy;

using PreparedCudaOutput = CudaPreparedOutputMetadataV1;

struct PreparedCudaCheckpointPayload
{
  std::vector<double> interior;
  std::vector<double> owned_true_trace;
  bool staged = false;
};

constexpr double kRodas3AlgebraicCorrectionFraction = 0.25;
constexpr std::uint32_t kMaximumConsecutiveAdaptiveRejections = 16;

bool HasTestOnlyPrepareBoundaryFailureHook() noexcept
{
#if defined(__GNUC__) || defined(__clang__)
  return dg_igr_hdg_cuda_prepare_boundary_failure_for_test != nullptr;
#else
  return false;
#endif
}

bool TestOnlyPrepareBoundaryFailure(
    const std::uint64_t committed_steps,
    const double committed_time) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
  return dg_igr_hdg_cuda_prepare_boundary_failure_for_test(
      committed_steps, committed_time);
#else
  static_cast<void>(committed_steps);
  static_cast<void>(committed_time);
  return false;
#endif
}

bool SameCoordinates(
    const dg::hdg::generic::HdgEulerSessionCoordinates& lhs,
    const dg::hdg::generic::HdgEulerSessionCoordinates& rhs) noexcept
{
  return lhs.time == rhs.time && lhs.step == rhs.step &&
      lhs.generation == rhs.generation && lhs.dt == rhs.dt;
}

bool IsRoot(const MPI_Comm communicator) noexcept
{
  int initialized = 0;
  int finalized = 0;
  int rank = -1;
  return communicator != MPI_COMM_NULL &&
      MPI_Initialized(&initialized) == MPI_SUCCESS && initialized &&
      MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized &&
      MPI_Comm_rank(communicator, &rank) == MPI_SUCCESS && rank == 0;
}

bool LiveSupportedCommunicator(const MPI_Comm communicator) noexcept
{
  int initialized = 0;
  int finalized = 0;
  int rank = -1;
  int ranks = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Initialized(&initialized) == MPI_SUCCESS && initialized &&
      MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized &&
      MPI_Comm_rank(communicator, &rank) == MPI_SUCCESS &&
      MPI_Comm_size(communicator, &ranks) == MPI_SUCCESS && rank >= 0 &&
      (ranks == 1 || ranks == 2);
}

bool CollectiveAll(const MPI_Comm communicator, const bool local) noexcept
{
  const int input = local ? 1 : 0;
  int output = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                    communicator) == MPI_SUCCESS && output == 1;
}

bool ExactBoolConsensus(const MPI_Comm communicator,
                        const bool value) noexcept
{
  int ranks = 0;
  const int input = value ? 1 : 0;
  int count = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Comm_size(communicator, &ranks) == MPI_SUCCESS && ranks > 0 &&
      MPI_Allreduce(&input, &count, 1, MPI_INT, MPI_SUM,
                    communicator) == MPI_SUCCESS &&
      (count == 0 || count == ranks);
}

bool ExactDoubleConsensus(const MPI_Comm communicator,
                          const double value) noexcept
{
  if (communicator == MPI_COMM_NULL) return false;
  std::uint64_t bits = 0, minimum = 0, maximum = 0;
  static_assert(sizeof(bits) == sizeof(value), "CUDA CFL requires binary64");
  std::memcpy(&bits, &value, sizeof(bits));
  const int minimum_status = MPI_Allreduce(&bits, &minimum, 1,
      MPI_UINT64_T, MPI_MIN, communicator);
  const int maximum_status = MPI_Allreduce(&bits, &maximum, 1,
      MPI_UINT64_T, MPI_MAX, communicator);
  return minimum_status == MPI_SUCCESS && maximum_status == MPI_SUCCESS &&
      minimum == maximum;
}

std::string FinalizeAdaptiveRejectionOrReason(
    const MPI_Comm communicator,
    HdgTransactionalTimeAdapter& time_driver,
    const HdgTransactionalTimeAdapter::StepPlan& plan,
    const HdgTransactionalTimeAdapter::AdaptiveAttempt& attempt,
    const char* const failure_classification)
{
  std::optional<HdgTransactionalTimeAdapter::AdaptiveDecision> decision;
  bool local_ok = false;
  try
  {
    decision.emplace(time_driver.PreviewAdaptiveDecision(attempt));
    local_ok = !decision->Accepted();
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!CollectiveAll(communicator, local_ok) ||
      !ExactDoubleConsensus(communicator, decision->next_dt()))
  {
    return "adaptive CUDA RODAS3 rejection decision is not collective";
  }

  try
  {
    time_driver.FinalizeAdaptiveRejection(*decision);
    local_ok = true;
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!CollectiveAll(communicator, local_ok))
    return "adaptive CUDA RODAS3 rejection transaction failed";

  HdgTransactionalTimeAdapter::AdaptiveControllerState state{};
  try
  {
    state = time_driver.AdaptiveCheckpointState();
    local_ok = true;
  }
  catch (...)
  {
    local_ok = false;
  }
  if (!CollectiveAll(communicator, local_ok) ||
      !ExactDoubleConsensus(communicator, state.next_dt))
  {
    return "adaptive CUDA RODAS3 controller state diverged after rejection";
  }
  const auto terminal_reason = [&](const char* const reason)
  {
    std::ostringstream stream;
    stream << reason << " committed_step=" << time_driver.solver_step()
           << " committed_t=" << std::scientific
           << std::setprecision(17) << time_driver.time()
           << " attempted_dt=" << plan.dt
           << " next_dt=" << state.next_dt
           << " consecutive_rejections=" << state.consecutive_rejections;
    if (failure_classification != nullptr &&
        failure_classification[0] != '\0')
      stream << " failure_class=" << failure_classification;
    return stream.str();
  };
  if (state.consecutive_rejections >=
      kMaximumConsecutiveAdaptiveRejections)
  {
    return terminal_reason(
        "adaptive CUDA RODAS3 exceeded the consecutive-rejection limit");
  }
  if (decision->minimum_dt_reached() &&
      plan.dt <= time_driver.adaptive_controller_config().minimum_dt)
  {
    return terminal_reason(
        "adaptive CUDA RODAS3 rejected an attempt at the minimum timestep");
  }
  return {};
}

const char* PrepareFailureClassName(
    const GenericEulerCudaPrepareFailureClass failure) noexcept
{
  switch (failure)
  {
  case GenericEulerCudaPrepareFailureClass::not_failed:
    return "prepare_not_failed";
  case GenericEulerCudaPrepareFailureClass::unavailable:
    return "prepare_failure_unavailable";
  case GenericEulerCudaPrepareFailureClass::
      zhang_shu_inadmissible_conservative_mean:
    return "zhang_shu_inadmissible_conservative_mean";
  case GenericEulerCudaPrepareFailureClass::finite_gate:
    return "prepare_finite_gate";
  case GenericEulerCudaPrepareFailureClass::admissibility_gate:
    return "prepare_admissibility_gate";
  case GenericEulerCudaPrepareFailureClass::trace_solve:
    return "prepare_trace_solve";
  case GenericEulerCudaPrepareFailureClass::recovery_gate:
    return "prepare_recovery_gate";
  case GenericEulerCudaPrepareFailureClass::true_residual_gate:
    return "prepare_true_residual_gate";
  case GenericEulerCudaPrepareFailureClass::telemetry_gate:
    return "prepare_telemetry_gate";
  case GenericEulerCudaPrepareFailureClass::success_gate:
    return "prepare_success_gate";
  case GenericEulerCudaPrepareFailureClass::collective_protocol:
    return "prepare_collective_protocol";
  case GenericEulerCudaPrepareFailureClass::backend_or_contract:
    return "prepare_backend_or_contract";
  case GenericEulerCudaPrepareFailureClass::mixed_numerical_failure:
    return "prepare_mixed_numerical_failure";
  }
  return "prepare_failure_unavailable";
}

const char* PreparedAssessmentFailureClass(
    const bool assessment_values_valid,
    const bool assessment_values_collective,
    const bool finite_solver_gate,
    const bool residual_gate,
    const bool admissibility_gate,
    const bool algebraic_closure_gate,
    const bool algebraic_correction_gate) noexcept
{
  if (!assessment_values_valid || !assessment_values_collective)
    return "prepared_assessment_invalid";
  const unsigned failed_gates = (!finite_solver_gate ? 1U : 0U) +
      (!residual_gate ? 1U : 0U) + (!admissibility_gate ? 1U : 0U) +
      (!algebraic_closure_gate ? 1U : 0U) +
      (!algebraic_correction_gate ? 1U : 0U);
  if (failed_gates > 1U) return "prepared_multiple_gates";
  if (!finite_solver_gate) return "prepared_solver_finite_gate";
  if (!residual_gate) return "prepared_residual_gate";
  if (!admissibility_gate) return "prepared_admissibility_gate";
  if (!algebraic_closure_gate) return "prepared_algebraic_closure_gate";
  if (!algebraic_correction_gate)
    return "prepared_algebraic_correction_gate";
  return "embedded_temporal_error";
}

int Reject(const HdgEulerMainRequest& request,
           const std::string& reason) noexcept
{
  if (IsRoot(request.communicator))
    std::cerr << "[hdg-main] status=Rejected reason=" << reason
              << " cuda_engine="
              << request.options.hdg_cuda_engine << '\n';
  return 1;
}

void AppendText(std::ostringstream& stream, const std::string& value)
{
  stream << value.size() << ':' << value << ';';
}

void AppendDouble(std::ostringstream& stream, const double value)
{
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value),
                "HDG main consensus requires IEEE binary64 doubles");
  std::memcpy(&bits, &value, sizeof(bits));
  stream << bits << ';';
}

void AppendStateHash(std::ostringstream& stream,
                     const dg::hdg::runtime::StateHash& hash)
{
  for (const std::uint8_t byte : hash.bytes)
    stream << static_cast<unsigned int>(byte) << ',';
  stream << ';';
}

bool NonzeroStateHash(const dg::hdg::runtime::StateHash& hash) noexcept
{
  return std::any_of(hash.bytes.begin(), hash.bytes.end(),
                     [](const std::uint8_t byte) { return byte != 0; });
}

bool ProjectedFieldHash(const double* values,
                        const std::size_t count,
                        dg::hdg::runtime::StateHash& hash) noexcept
{
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "projected input identity requires IEEE binary64 storage");
  if ((count != 0 && values == nullptr) ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(double))
    return false;
  for (std::size_t index = 0; index < count; ++index)
    if (!std::isfinite(values[index])) return false;
  const std::uint8_t empty = 0;
  const auto* bytes = count == 0
      ? &empty : reinterpret_cast<const std::uint8_t*>(values);
  const std::size_t byte_count = count == 0 ? 1 : count * sizeof(double);
  hash = dg::hdg::runtime::app::CheckpointSha256V1(
      {bytes, byte_count});
  return NonzeroStateHash(hash);
}

struct ProjectedProblemRankIdentityV1
{
  std::uint32_t schema = 1;
  int source_rank = -1;
  int rank_count = 0;
  std::uint64_t initial_volume_count = 0;
  std::uint64_t initial_true_trace_count = 0;
  std::uint64_t source_volume_count = 0;
  dg::hdg::runtime::StateHash initial_volume_hash{};
  dg::hdg::runtime::StateHash initial_true_trace_hash{};
  dg::hdg::runtime::StateHash source_volume_hash{};
  bool source_present = false;
};

static_assert(std::is_trivially_copyable<ProjectedProblemRankIdentityV1>::value,
              "projected-input consensus transfer must be a value DTO");

bool CollectiveProjectedProblemIdentity(
    const MPI_Comm communicator,
    const double* initial_volume,
    const std::size_t initial_volume_count,
    const double* initial_true_trace,
    const std::size_t initial_true_trace_count,
    const double* source_volume,
    const std::size_t source_volume_count,
    const bool source_present,
    dg::hdg::runtime::StateHash& identity) noexcept
{
  int rank = -1;
  int ranks = 0;
  bool local_ok = communicator != MPI_COMM_NULL &&
      MPI_Comm_rank(communicator, &rank) == MPI_SUCCESS &&
      MPI_Comm_size(communicator, &ranks) == MPI_SUCCESS && rank >= 0 &&
      ranks > 0 &&
      sizeof(ProjectedProblemRankIdentityV1) <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      initial_volume_count <= std::numeric_limits<std::uint64_t>::max() &&
      initial_true_trace_count <= std::numeric_limits<std::uint64_t>::max() &&
      source_volume_count <= std::numeric_limits<std::uint64_t>::max() &&
      (source_present ||
       (source_volume == nullptr && source_volume_count == 0));
  ProjectedProblemRankIdentityV1 local{};
  if (local_ok)
  {
    local.source_rank = rank;
    local.rank_count = ranks;
    local.initial_volume_count = initial_volume_count;
    local.initial_true_trace_count = initial_true_trace_count;
    local.source_volume_count = source_volume_count;
    local.source_present = source_present;
    local_ok = ProjectedFieldHash(
            initial_volume, initial_volume_count,
            local.initial_volume_hash) &&
        ProjectedFieldHash(
            initial_true_trace, initial_true_trace_count,
            local.initial_true_trace_hash) &&
        ProjectedFieldHash(
            source_volume, source_volume_count,
            local.source_volume_hash);
  }
  if (!CollectiveAll(communicator, local_ok)) return false;

  std::vector<ProjectedProblemRankIdentityV1> gathered;
  try { gathered.resize(static_cast<std::size_t>(ranks)); }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(communicator, local_ok)) return false;
  const bool gathered_ok = MPI_Allgather(
      &local, static_cast<int>(sizeof(local)), MPI_BYTE,
      gathered.data(), static_cast<int>(sizeof(local)), MPI_BYTE,
      communicator) == MPI_SUCCESS;
  if (!CollectiveAll(communicator, gathered_ok)) return false;

  try
  {
    std::ostringstream stream;
    AppendText(stream, "generic-cuda-hdg-projected-problem.v1");
    stream << ranks << ';';
    for (int index = 0; index < ranks; ++index)
    {
      const auto& source = gathered[static_cast<std::size_t>(index)];
      if (source.schema != 1 || source.source_rank != index ||
          source.rank_count != ranks ||
          source.source_present != gathered.front().source_present ||
          !NonzeroStateHash(source.initial_volume_hash) ||
          !NonzeroStateHash(source.initial_true_trace_hash) ||
          !NonzeroStateHash(source.source_volume_hash))
        return false;
      stream << source.schema << ';' << source.source_rank << ';'
             << source.rank_count << ';'
             << source.initial_volume_count << ';'
             << source.initial_true_trace_count << ';'
             << source.source_volume_count << ';'
             << source.source_present << ';';
      AppendStateHash(stream, source.initial_volume_hash);
      AppendStateHash(stream, source.initial_true_trace_hash);
      AppendStateHash(stream, source.source_volume_hash);
    }
    const std::string text = stream.str();
    identity = dg::hdg::runtime::app::CheckpointSha256V1(
        {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
    local_ok = NonzeroStateHash(identity);
  }
  catch (...) { local_ok = false; }
  return CollectiveAll(communicator, local_ok);
}

struct FrozenBoundaryPlan
{
  EulerBoundaryPlan plan;
  std::string consensus_key;
};

// Resolve every analytic boundary value once before consensus and lowering.
// This is the same semantic freezing used by the CPU adapter: CUDA receives
// the generic numerical boundary callback, not a second CUDA-only policy list.
FrozenBoundaryPlan FreezeBoundaryPlan(const EulerBoundaryPlan& source)
{
  FrozenBoundaryPlan frozen;
  std::ostringstream key;
  key << source.Entries().size() << ';';
  for (const EulerBoundaryPlanEntry& entry : source.Entries())
  {
    key << static_cast<int>(entry.kind) << ';'
        << static_cast<int>(entry.inlet_model) << ';'
        << static_cast<int>(entry.outlet_model) << ';'
        << entry.marker.Size() << ';';
    for (int index = 0; index < entry.marker.Size(); ++index)
    {
      key << entry.marker[index] << ';';
    }
    mfem::Array<int> marker(entry.marker);
    switch (entry.kind)
    {
    case EulerBoundaryKind::Periodic:
      frozen.plan.AddPeriodic(std::move(marker));
      break;
    case EulerBoundaryKind::Slip:
      frozen.plan.AddSlipWall(std::move(marker));
      break;
    case EulerBoundaryKind::Inlet:
    {
      if (!entry.state)
      {
        throw std::invalid_argument(
            "HDG boundary plan is missing an inlet state callback");
      }
      const mfem::Vector state = entry.state();
      key << state.Size() << ';';
      for (int component = 0; component < state.Size(); ++component)
      {
        AppendDouble(key, state(component));
      }
      switch (entry.inlet_model)
      {
      case EulerInletModel::Supersonic:
        frozen.plan.AddSupersonicInlet(
            std::move(marker), [state] { return state; });
        break;
      case EulerInletModel::Subsonic:
        frozen.plan.AddSubsonicInlet(
            std::move(marker), [state] { return state; });
        break;
      default:
        throw std::invalid_argument(
            "HDG boundary plan has an unsupported inlet model");
      }
      break;
    }
    case EulerBoundaryKind::Outlet:
      switch (entry.outlet_model)
      {
      case EulerOutletModel::Supersonic:
        frozen.plan.AddSupersonicOutlet(std::move(marker));
        break;
      case EulerOutletModel::SubsonicPressure:
      {
        if (!entry.pressure)
        {
          throw std::invalid_argument(
              "HDG boundary plan is missing an outlet pressure callback");
        }
        const double pressure = entry.pressure();
        AppendDouble(key, pressure);
        frozen.plan.AddSubsonicOutletPressure(
            std::move(marker), [pressure] { return pressure; });
        break;
      }
      case EulerOutletModel::FarField:
      {
        if (!entry.state)
        {
          throw std::invalid_argument(
              "HDG boundary plan is missing a far-field state callback");
        }
        const mfem::Vector state = entry.state();
        key << state.Size() << ';';
        for (int component = 0; component < state.Size(); ++component)
        {
          AppendDouble(key, state(component));
        }
        frozen.plan.AddFarField(
            std::move(marker), [state] { return state; });
        break;
      }
      }
      break;
    case EulerBoundaryKind::NonReflecting:
    {
      if (!entry.state)
      {
        throw std::invalid_argument(
            "HDG boundary plan is missing a non-reflecting state callback");
      }
      const mfem::Vector state = entry.state();
      key << state.Size() << ';';
      for (int component = 0; component < state.Size(); ++component)
      {
        AppendDouble(key, state(component));
      }
      frozen.plan.AddNonReflectingOutlet(
          std::move(marker), [state] { return state; });
      break;
    }
    }
  }
  frozen.consensus_key = key.str();
  return frozen;
}

igrlab::IGRAlphaLengthScalePolicy CanonicalAlphaLengthScalePolicy(
    const HdgEulerMainRequest& request) noexcept
{
  return request.alpha_length_scale_policy_override.value_or(
      request.case_config.igr_config.alpha_length_scale_policy);
}

HdgRegularizationRequest CanonicalRegularization(
    const HdgEulerMainRequest& request)
{
  const std::string selector = ToLowerCopy(request.options.regularization);
  if (selector == "none")
  {
    return {RegularizationKind::none, 0.0, 0.0};
  }
  if (selector == "igr")
  {
    return {RegularizationKind::igr,
            request.options.hdg_igr_alpha,
            request.options.hdg_igr_beta};
  }
  throw std::invalid_argument(
      "HDG regularization must resolve to 'none' or 'igr'");
}

EulerHdgTimeTreatment CanonicalTimeTreatment(
    const HdgEulerMainRequest& request)
{
  const std::string selector = ToLowerCopy(request.options.time_integrator);
  if (selector.empty() || selector == "backward-euler")
  {
    return EulerHdgTimeTreatment::backward_euler;
  }
  if (selector == "partitioned-backward-euler")
  {
    return EulerHdgTimeTreatment::partitioned_backward_euler;
  }
  if (selector == "rodas3")
  {
    return EulerHdgTimeTreatment::rodas3;
  }
  throw std::invalid_argument(
      "HDG time integrator must resolve to 'backward-euler', "
      "'partitioned-backward-euler', or 'rodas3'");
}

CudaEngineKind CanonicalCudaEngine(const HdgEulerMainRequest& request)
{
  const std::optional<HdgCudaEngine> engine =
      TryParseHdgCudaEngine(request.options.hdg_cuda_engine);
  if (!engine)
  {
    throw std::invalid_argument(
        "HDG CUDA engine must resolve to 'ordinary' or 'tensor'");
  }
  return *engine == HdgCudaEngine::TensorCore
      ? CudaEngineKind::tensor_core
      : CudaEngineKind::ordinary;
}

const char* CudaEngineName(const CudaEngineKind engine) noexcept
{
  switch (engine)
  {
  case CudaEngineKind::ordinary:
    return "ordinary";
  case CudaEngineKind::tensor_core:
    return "tensor";
  }
  return "unknown";
}

const char* TraceSolverName(
    const CudaTracePolicyDescriptor& policy) noexcept
{
  if (!CudaTracePolicyDescriptorIsExact(policy)) return "unconfigured";
  if (policy.rank_count == 2) return "hypre-host-staged-rank2";
  if (policy.resolved_selection == CudaTracePolicyKind::equilibrated_mgr)
    return "hypre-fgmres-mgr-rank1";
  return policy.selected_engine == CudaEngineKind::tensor_core
      ? "native-rank1-fp64" : "hypre-parcsr-gmres";
}

std::string CanonicalNumericalPolicy(
    const dg::hdg::generic::EulerHdgNumericalPolicy& policy)
{
  std::ostringstream stream;
  stream << policy.volume_projection_quadrature_order << ';'
         << policy.true_trace_projection_quadrature_order << ';'
         << policy.conventional_assembly_quadrature_order << ';'
         << policy.igr_assembly_quadrature_orders.volume_order << ';'
         << policy.igr_assembly_quadrature_orders.face_order << ';'
         << policy.maximum_nonlinear_updates << ';'
         << policy.maximum_linear_iterations << ';'
         << policy.maximum_line_search_trials << ';'
         << static_cast<int>(policy.stabilization.kind) << ';'
         << static_cast<int>(policy.positivity_limiter.kind) << ';'
         << static_cast<int>(policy.positivity_limiter.point_set) << ';'
         << policy.positivity_limiter.pressure_bisection_iterations << ';';
  AppendDouble(stream, policy.nonlinear_relative_tolerance);
  AppendDouble(stream,
               policy.newton_correction_linear_relative_tolerance);
  AppendDouble(stream, policy.final_coherent_linear_relative_tolerance);
  AppendDouble(
      stream, policy.true_residual_tightening_linear_relative_tolerance);
  AppendDouble(stream, policy.true_residual_relative_tolerance);
  AppendDouble(stream, policy.conservation_relative_tolerance);
  AppendDouble(stream, policy.armijo_constant);
  AppendDouble(stream, policy.stabilization.alpha);
  AppendDouble(stream, policy.stabilization.beta);
  AppendDouble(stream, policy.positivity_limiter.density_floor);
  AppendDouble(stream, policy.positivity_limiter.pressure_floor);
  return stream.str();
}

EulerHdgZhangShuPointSet CanonicalZhangShuPointSet(
    const std::string& point_set)
{
  const std::string selector = ToLowerCopy(point_set);
  if (selector == "consumer")
  {
    return EulerHdgZhangShuPointSet::consumer;
  }
  if (selector == "quadrature")
  {
    return EulerHdgZhangShuPointSet::quadrature;
  }
  throw std::invalid_argument(
      "CUDA HDG Zhang-Shu point set must resolve to 'consumer' or "
      "'quadrature'");
}

EulerHdgPositivityLimiterPolicy CanonicalPositivityLimiterPolicy(
    const HdgEulerMainRequest& request,
    const EulerHdgTimeTreatment time_treatment)
{
  if (time_treatment != EulerHdgTimeTreatment::rodas3 ||
      !ResolveZhangShuPositivityLimiter(
          request.options.positivity_limiter,
          request.case_config.limiter_pipeline.zhang_shu))
    return {};
  return {EulerHdgPositivityLimiterKind::zhang_shu,
          CanonicalZhangShuPointSet(request.options.zhang_shu_points),
          request.case_config.rk3_config.rho_min,
          request.case_config.rk3_config.p_min,
          request.case_config.rk3_config.p_bisect_iters};
}

std::string ConsensusText(const HdgEulerMainRequest& request,
                          const std::string& mesh_path,
                          const HdgRegularizationRequest& regularization,
                          const CudaTracePolicyDescriptor& trace_policy,
                          const EulerHdgTimeTreatment time_treatment)
{
  std::ostringstream stream;
  const CudaEngineKind cuda_engine = trace_policy.selected_engine;
  const mfem::io::VTKOutputConfig vtk = MakeCaseVTKConfig(request.case_config);
  AppendText(stream, request.case_config.case_name);
  AppendText(stream, mesh_path);
  AppendText(stream, ToLowerCopy(request.options.device));
  AppendText(stream, CudaEngineName(cuda_engine));
  const auto trace_preconditioner = trace_policy.trace_preconditioner;
  const bool trace_uses_ilu =
      CudaTracePreconditionerIsIlu(trace_preconditioner);
  const auto ilu_accounting_kind = trace_uses_ilu
      ? CudaIluAccountingKind::derived_bounds_v1
      : CudaIluAccountingKind::unavailable;
  const auto ilu_factor_reuse_policy = trace_policy.ilu_factor_reuse_policy;
  const auto ilu_triangular_solve_kind =
      trace_policy.ilu_triangular_solve_kind;
  const int ilu_lower_jacobi_iterations =
      trace_policy.ilu_lower_jacobi_iterations;
  const int ilu_upper_jacobi_iterations =
      trace_policy.ilu_upper_jacobi_iterations;
  AppendText(stream,
             CudaTracePolicyKindName(trace_policy.requested_selection));
  AppendText(stream,
             CudaTracePolicyKindName(trace_policy.resolved_selection));
  AppendText(stream, TraceSolverName(trace_policy));
  AppendText(stream, CudaTracePreconditionerName(trace_preconditioner));
  stream << CudaIluType(trace_preconditioner) << ';'
         << CudaIluLevelOfFill(trace_preconditioner) << ';'
         << trace_policy.ilu_local_reordering << ';'
         << CudaIluApplicationIterations(trace_preconditioner) << ';';
  AppendText(stream, CudaIluAccountingKindName(ilu_accounting_kind));
  AppendText(stream, CudaIluFactorReusePolicyName(ilu_factor_reuse_policy));
  AppendText(stream,
             CudaIluTriangularSolveKindName(ilu_triangular_solve_kind));
  stream << ilu_lower_jacobi_iterations << ';'
         << ilu_upper_jacobi_iterations << ';';
  AppendText(stream, request.options.restart_vtk);
  AppendText(stream, request.options.hdg_checkpoint);
  AppendText(stream, request.options.hdg_restart_checkpoint);
  AppendText(stream, vtk.prefix_path);
  AppendText(
      stream,
      PositivityLimiterSelectionToString(ParsePositivityLimiterSelection(
          request.options.positivity_limiter)));
  AppendText(stream, ToLowerCopy(request.options.zhang_shu_points));
  stream << request.options.order << ';'
         << request.options.ser_ref_levels << ';'
         << request.options.par_ref_levels << ';'
         << request.options.restart_cycle << ';'
         << request.options.restart_step << ';'
         << request.options.vtk_save << ';'
         << request.options.vtk_zhang_shu_diagnostics << ';'
         << request.options.glvis << ';'
         << request.options.save_state_files << ';'
         << static_cast<int>(vtk.format) << ';'
         << vtk.high_order_output << ';'
         << vtk.levels_of_detail << ';';
  AppendDouble(stream, request.options.dt);
  AppendDouble(stream, request.options.dt_min);
  AppendDouble(stream, request.options.time_atol);
  AppendDouble(stream, request.options.time_rtol);
  AppendDouble(stream, request.options.cfl);
  AppendDouble(stream, request.options.t_final);
  AppendDouble(stream, request.options.dt_save);
  AppendDouble(stream, request.case_config.gamma);
  stream << static_cast<int>(regularization.kind) << ';'
         << static_cast<int>(CanonicalAlphaLengthScalePolicy(request)) << ';'
         << static_cast<int>(time_treatment) << ';';
  AppendDouble(stream, regularization.c_alpha);
  AppendDouble(stream, regularization.euler_trace_penalty);
  return stream.str();
}

dg::hdg::runtime::StateHash CheckpointNumericalIdentity(
    const HdgEulerMainRequest& request,
    const std::string& serialized_mesh,
    const std::string& case_data_consensus,
    const dg::hdg::runtime::StateHash& projected_problem_identity,
    const RegularizationSelection regularization,
    const CudaTracePolicyDescriptor& trace_policy,
    const EulerHdgTimeTreatment time_treatment)
{
  // Artifact locations and VTK/GLVis publication toggles are deliberately
  // absent.  The time cursor persisted in the artifact independently binds
  // its complete control and output schedule.
  std::ostringstream stream;
  AppendText(stream, "generic-cuda-hdg-checkpoint-identity.v1");
  AppendText(stream, request.case_config.case_name);
  AppendText(stream, serialized_mesh);
  AppendText(stream, case_data_consensus);
  AppendStateHash(stream, projected_problem_identity);
  AppendText(stream, CudaEngineName(trace_policy.selected_engine));
  AppendText(stream,
             CudaTracePolicyKindName(trace_policy.requested_selection));
  AppendText(stream,
             CudaTracePolicyKindName(trace_policy.resolved_selection));
  AppendText(stream, TraceSolverName(trace_policy));
  stream << request.options.order << ';'
         << request.options.ser_ref_levels << ';'
         << request.options.par_ref_levels << ';'
         << static_cast<int>(regularization.kind) << ';'
         << static_cast<int>(time_treatment) << ';';
  AppendDouble(stream, request.options.dt);
  AppendDouble(stream, request.options.dt_min);
  AppendDouble(stream, request.options.time_atol);
  AppendDouble(stream, request.options.time_rtol);
  AppendDouble(stream, request.options.cfl);
  AppendDouble(stream, request.options.t_final);
  AppendDouble(stream, request.options.dt_save);
  AppendDouble(stream, request.case_config.gamma);
  AppendDouble(stream, regularization.alpha);
  AppendDouble(stream, regularization.beta);
  const std::string identity = stream.str();
  return dg::hdg::runtime::app::CheckpointSha256V1(
      {reinterpret_cast<const std::uint8_t*>(identity.data()),
       identity.size()});
}

std::string CaseDataConsensusText(
    const HdgEulerCaseData& case_data,
    const FrozenBoundaryPlan& boundary,
    const dg::hdg::generic::EulerHdgNumericalPolicy& numerical_policy)
{
  std::ostringstream stream;
  stream << case_data.initial_condition->GetVDim() << ';'
         << static_cast<bool>(case_data.static_source) << ';';
  if (case_data.static_source)
  {
    stream << case_data.static_source->GetVDim() << ';';
  }
  AppendText(stream, boundary.consensus_key);
  AppendText(stream, CanonicalNumericalPolicy(numerical_policy));
  return stream.str();
}

bool ExactTextConsensus(const MPI_Comm communicator,
                        const std::string& local) noexcept
{
  int rank = -1;
  if (communicator == MPI_COMM_NULL ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) return false;
  std::uint64_t root_size = rank == 0 ? local.size() : 0;
  const bool size_broadcast = MPI_Bcast(&root_size, 1, MPI_UINT64_T, 0,
                                        communicator) == MPI_SUCCESS;
  if (!CollectiveAll(communicator, size_broadcast &&
      root_size <= static_cast<std::uint64_t>(std::numeric_limits<int>::max())))
    return false;
  std::string root;
  bool allocated = true;
  try { root.resize(static_cast<std::size_t>(root_size)); }
  catch (...) { allocated = false; }
  if (!CollectiveAll(communicator, allocated)) return false;
  if (rank == 0 && root_size != 0)
    std::memcpy(root.data(), local.data(), static_cast<std::size_t>(root_size));
  char empty = 0;
  char* data = root.empty() ? &empty : root.data();
  const bool text_broadcast = MPI_Bcast(data, static_cast<int>(root_size),
                                        MPI_CHAR, 0, communicator) == MPI_SUCCESS;
  if (!CollectiveAll(communicator, text_broadcast)) return false;
  return CollectiveAll(
      communicator, local.size() == static_cast<std::size_t>(root_size) &&
          std::equal(local.begin(), local.end(), root.begin()));
}

struct LoadedCudaCheckpoint
{
  HdgTransactionalTimeAdapter::RestartState time_state{};
  CudaPreparedOutputMetadataV1 local_metadata{};
  std::vector<double> local_interior;
  std::vector<double> local_owned_true_trace;
};

bool BroadcastCheckpointBytes(const MPI_Comm communicator,
                              std::vector<std::uint8_t>& bytes) noexcept
{
  int rank = -1;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) return false;
  std::uint64_t byte_count = rank == 0 ? bytes.size() : 0;
  if (MPI_Bcast(&byte_count, 1, MPI_UINT64_T, 0, communicator) != MPI_SUCCESS ||
      byte_count == 0 ||
      byte_count >
          dg::hdg::runtime::app::
              generic_hdg_cuda_checkpoint_maximum_bytes_v1 ||
      byte_count > static_cast<std::uint64_t>(
          std::numeric_limits<int>::max()))
    return false;
  try
  {
    if (rank != 0) bytes.resize(static_cast<std::size_t>(byte_count));
  }
  catch (...) { return false; }
  return MPI_Bcast(bytes.data(), static_cast<int>(byte_count), MPI_BYTE, 0,
                   communicator) == MPI_SUCCESS;
}

bool MaterializeCudaCheckpointMetadata(
    const dg::hdg::runtime::app::GenericHdgCudaOwnedRankCheckpointV1& source,
    CudaPreparedOutputMetadataV1& destination) noexcept
{
  destination = {};
  if (source.source_rank >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      source.rank_count >
          static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
      source.interior.empty() || source.owned_true_trace.empty())
    return false;
  destination.schema = 1;
  destination.source_rank = static_cast<int>(source.source_rank);
  destination.rank_count = static_cast<int>(source.rank_count);
  destination.attempt = source.attempt;
  destination.descriptor = source.descriptor;
  destination.local_mesh_hash = source.local_mesh_hash;
  destination.local_partition_hash = source.local_partition_hash;
  destination.local_topology_hash = source.local_topology_hash;
  destination.local_interior_hash = source.local_interior_hash;
  destination.local_trace_hash = source.local_trace_hash;
  destination.interior_count = source.interior.size();
  destination.owned_true_trace_count = source.owned_true_trace.size();
  destination.staging_generation = source.descriptor.coordinates.generation;
  destination.successful_commit = true;
  return true;
}

bool LoadCudaCheckpoint(
    const MPI_Comm communicator,
    const std::string& path,
    const dg::hdg::runtime::StateHash& numerical_identity,
    const std::uint32_t rank_count,
    LoadedCudaCheckpoint& loaded) noexcept
{
  int rank = -1;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) return false;
  std::vector<std::uint8_t> bytes;
  std::string detail;
  bool local_ok = rank != 0 ||
      dg::hdg::runtime::app::detail::ReadCurrentGenerationArtifact(
          std::filesystem::path{path},
          dg::hdg::runtime::app::
              generic_hdg_cuda_checkpoint_maximum_bytes_v1,
          bytes, detail) == dg::hdg::runtime::app::DurableIoStatus::ok;
  if (!CollectiveAll(communicator, local_ok) ||
      !BroadcastCheckpointBytes(communicator, bytes))
    return false;
  dg::hdg::runtime::app::GenericHdgCudaCheckpointArtifactV1 artifact{};
  local_ok =
      dg::hdg::runtime::app::DecodeGenericHdgCudaCheckpointV1(
          {bytes.data(), bytes.size()}, numerical_identity, rank_count,
          artifact) == dg::hdg::runtime::app::CheckpointCodecStatus::ok &&
      rank >= 0 && static_cast<std::size_t>(rank) < artifact.ranks.size();
  LoadedCudaCheckpoint candidate{};
  if (local_ok)
  {
    auto& source = artifact.ranks[rank];
    local_ok = MaterializeCudaCheckpointMetadata(
        source, candidate.local_metadata);
    if (local_ok)
    {
      candidate.time_state = std::move(artifact.time_state);
      candidate.local_interior = std::move(source.interior);
      candidate.local_owned_true_trace =
          std::move(source.owned_true_trace);
    }
  }
  if (!CollectiveAll(communicator, local_ok)) return false;
  loaded = std::move(candidate);
  return true;
}

struct CudaCheckpointRankIdentityV1
{
  std::uint32_t schema = 0;
  int source_rank = -1;
  int rank_count = 0;
  dg::hdg::runtime::AttemptIdentity attempt{};
  dg::hdg::runtime::CheckpointDescriptor descriptor{};
  dg::hdg::runtime::StateHash local_mesh_hash{};
  dg::hdg::runtime::StateHash local_partition_hash{};
  dg::hdg::runtime::StateHash local_topology_hash{};
  dg::hdg::runtime::StateHash local_interior_hash{};
  dg::hdg::runtime::StateHash local_trace_hash{};
  std::uint64_t interior_count = 0;
  std::uint64_t owned_true_trace_count = 0;
  bool successful_commit = false;
};

static_assert(std::is_trivially_copyable<CudaCheckpointRankIdentityV1>::value,
              "rank-local checkpoint identity gather must be a value DTO");

CudaCheckpointRankIdentityV1 CheckpointRankIdentity(
    const CudaPreparedOutputMetadataV1& source) noexcept
{
  return {source.schema,
          source.source_rank,
          source.rank_count,
          source.attempt,
          source.descriptor,
          source.local_mesh_hash,
          source.local_partition_hash,
          source.local_topology_hash,
          source.local_interior_hash,
          source.local_trace_hash,
          source.interior_count,
          source.owned_true_trace_count,
          source.successful_commit};
}

bool PublishCudaCheckpoint(
    const MPI_Comm communicator,
    const std::string& path,
    const dg::hdg::runtime::StateHash& numerical_identity,
    const HdgTransactionalTimeAdapter::RestartState& time_state,
    const CudaPreparedOutputMetadataV1& local_metadata,
    const PreparedCudaCheckpointPayload& local_payload) noexcept
{
  int rank = -1;
  int ranks = 0;
  if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(communicator, &ranks) != MPI_SUCCESS || ranks < 1 ||
      ranks > static_cast<int>(dg::hdg::runtime::app::
          application_checkpoint_max_rank_count_v1) ||
      sizeof(CudaCheckpointRankIdentityV1) >
          static_cast<std::size_t>(std::numeric_limits<int>::max()))
    return false;
  const bool local_payload_valid = local_payload.staged &&
      local_metadata.schema == 1 && local_metadata.successful_commit &&
      !local_payload.interior.empty() &&
      !local_payload.owned_true_trace.empty() &&
      local_metadata.interior_count == local_payload.interior.size() &&
      local_metadata.owned_true_trace_count ==
          local_payload.owned_true_trace.size() &&
      local_payload.interior.size() <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      local_payload.owned_true_trace.size() <=
          static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (!CollectiveAll(communicator, local_payload_valid)) return false;

  const CudaCheckpointRankIdentityV1 transfer =
      CheckpointRankIdentity(local_metadata);
  std::array<CudaCheckpointRankIdentityV1,
             dg::hdg::runtime::app::application_checkpoint_max_rank_count_v1>
      gathered{};
  const bool identity_gathered = MPI_Gather(
      &transfer, static_cast<int>(sizeof(transfer)), MPI_BYTE,
      gathered.data(), static_cast<int>(sizeof(transfer)), MPI_BYTE,
      0, communicator) == MPI_SUCCESS;
  if (!CollectiveAll(communicator, identity_gathered))
    return false;

  std::array<int,
             dg::hdg::runtime::app::application_checkpoint_max_rank_count_v1>
      interior_counts{}, interior_offsets{}, trace_counts{}, trace_offsets{};
  int aggregate_interior_count = 0;
  int aggregate_trace_count = 0;
  bool local_ok = true;
  if (rank == 0)
  {
    for (int index = 0; local_ok && index < ranks; ++index)
    {
      const auto& source = gathered[index];
      local_ok = source.interior_count > 0 &&
          source.owned_true_trace_count > 0 &&
          source.interior_count <= static_cast<std::uint64_t>(
              std::numeric_limits<int>::max() - aggregate_interior_count) &&
          source.owned_true_trace_count <= static_cast<std::uint64_t>(
              std::numeric_limits<int>::max() - aggregate_trace_count);
      if (!local_ok) break;
      interior_counts[index] = static_cast<int>(source.interior_count);
      trace_counts[index] =
          static_cast<int>(source.owned_true_trace_count);
      interior_offsets[index] = aggregate_interior_count;
      trace_offsets[index] = aggregate_trace_count;
      aggregate_interior_count += interior_counts[index];
      aggregate_trace_count += trace_counts[index];
    }
    constexpr std::uint64_t maximum_coefficients =
        dg::hdg::runtime::app::
            generic_hdg_cuda_checkpoint_maximum_bytes_v1 /
        sizeof(double);
    local_ok = local_ok &&
        static_cast<std::uint64_t>(aggregate_interior_count) <=
            maximum_coefficients &&
        static_cast<std::uint64_t>(aggregate_trace_count) <=
            maximum_coefficients - static_cast<std::uint64_t>(
                aggregate_interior_count);
  }
  if (!CollectiveAll(communicator, local_ok)) return false;

  std::vector<double> gathered_interior;
  std::vector<double> gathered_trace;
  if (rank == 0)
  {
    try
    {
      gathered_interior.resize(
          static_cast<std::size_t>(aggregate_interior_count));
      gathered_trace.resize(
          static_cast<std::size_t>(aggregate_trace_count));
    }
    catch (...) { local_ok = false; }
  }
  if (!CollectiveAll(communicator, local_ok)) return false;
  const bool interior_gathered = MPI_Gatherv(
      local_payload.interior.data(),
      static_cast<int>(local_payload.interior.size()), MPI_DOUBLE,
      gathered_interior.data(), interior_counts.data(),
      interior_offsets.data(), MPI_DOUBLE, 0, communicator) == MPI_SUCCESS;
  const bool trace_gathered = MPI_Gatherv(
      local_payload.owned_true_trace.data(),
      static_cast<int>(local_payload.owned_true_trace.size()), MPI_DOUBLE,
      gathered_trace.data(), trace_counts.data(), trace_offsets.data(),
      MPI_DOUBLE, 0, communicator) == MPI_SUCCESS;
  if (!CollectiveAll(
          communicator, interior_gathered && trace_gathered))
    return false;

  if (rank == 0)
  {
    std::string detail;
    std::array<dg::hdg::runtime::app::GenericHdgCudaCheckpointRankViewV1,
               dg::hdg::runtime::app::
                   application_checkpoint_max_rank_count_v1> views{};
    for (int index = 0; index < ranks; ++index)
    {
      const CudaCheckpointRankIdentityV1& source = gathered[index];
      views[index] = {
          static_cast<std::uint32_t>(source.source_rank),
          static_cast<std::uint32_t>(source.rank_count),
          source.attempt,
          source.descriptor,
          source.local_mesh_hash,
          source.local_partition_hash,
          source.local_topology_hash,
          source.local_interior_hash,
          source.local_trace_hash,
          gathered_interior.data() + interior_offsets[index],
          source.interior_count,
          gathered_trace.data() + trace_offsets[index],
          source.owned_true_trace_count,
          source.successful_commit};
    }
    std::vector<std::uint8_t> encoded;
    local_ok = dg::hdg::runtime::app::EncodeGenericHdgCudaCheckpointV1(
        numerical_identity, time_state, views.data(),
        static_cast<std::size_t>(ranks), encoded) ==
        dg::hdg::runtime::app::CheckpointCodecStatus::ok;
    if (local_ok)
    {
      try
      {
        const std::filesystem::path alias =
            std::filesystem::absolute(path).lexically_normal();
        const std::filesystem::path parent = alias.has_parent_path()
            ? alias.parent_path() : std::filesystem::path{"."};
        const std::filesystem::path store = parent /
            ("." + alias.filename().string() +
             ".dg-igr-hdg-checkpoint-store");
        std::error_code error;
        if (!std::filesystem::exists(store, error))
        {
          local_ok = !error &&
              dg::hdg::runtime::app::CreateDurableDirectory(
                  store, detail) ==
                  dg::hdg::runtime::app::DurableIoStatus::ok;
        }
        else
        {
          local_ok = !error && std::filesystem::is_directory(store, error) &&
              !error;
        }
        dg::hdg::runtime::app::DurableGenerationTransactionV1 transaction;
        constexpr std::string_view logical_name =
            "generic-hdg-cuda-checkpoint.v1";
        if (local_ok)
          local_ok = transaction.Begin(
              store, local_metadata.descriptor.coordinates.cursor,
              detail) == dg::hdg::runtime::app::DurableIoStatus::ok;
        if (local_ok)
          local_ok = transaction.Stage(logical_name, encoded.data(),
                                       encoded.size(), detail) ==
              dg::hdg::runtime::app::DurableIoStatus::ok;
        if (local_ok)
          local_ok = transaction.AddAlias(alias, logical_name, detail) ==
              dg::hdg::runtime::app::DurableIoStatus::ok;
        if (local_ok)
          local_ok = transaction.Publish(detail) ==
              dg::hdg::runtime::app::DurableIoStatus::ok;
      }
      catch (...) { local_ok = false; }
    }
  }
  return CollectiveAll(communicator, local_ok);
}

std::string Validate(const HdgEulerMainRequest& request,
                     const HdgRegularizationRequest& regularization,
                     const CudaEngineKind cuda_engine,
                     const EulerHdgTimeTreatment time_treatment)
{
  if (!LiveSupportedCommunicator(request.communicator))
    return "CUDA HDG requires a live one- or two-rank MPI communicator";
  if (!request.make_case_data)
    return "CUDA HDG case physics factory is empty";
  if (ToLowerCopy(request.options.device) != "cuda")
    return "generic CUDA HDG requires --device cuda";
  if (!dg::hdg::generic::IsKnownCudaEngineKind(cuda_engine))
    return "generic CUDA HDG rejects an invalid CUDA engine";
  if (!ValidHdgRegularizationRequest(regularization))
    return "HDG IGR c_alpha and Euler trace penalty must be finite and positive";
  if (regularization.kind == RegularizationKind::igr &&
      CanonicalAlphaLengthScalePolicy(request) ==
        igrlab::IGRAlphaLengthScalePolicy::Local)
    return "HDG IGR does not support the element-local alpha length policy";
  if (!dg::hdg::generic::IsKnownEulerHdgTimeTreatment(time_treatment))
    return "CUDA HDG time treatment is invalid";
  const bool rodas3 = time_treatment == EulerHdgTimeTreatment::rodas3;
  const PositivityLimiterSelection positivity_limiter =
      ParsePositivityLimiterSelection(request.options.positivity_limiter);
  static_cast<void>(CanonicalZhangShuPointSet(
      request.options.zhang_shu_points));
  if (!rodas3 &&
      positivity_limiter == PositivityLimiterSelection::ZhangShu)
    return "CUDA HDG Zhang-Shu positivity limiting is a RODAS3-only typed "
           "opt-in";
  if (!request.options.restart_vtk.empty() || request.options.restart_cycle >= 0 ||
      request.options.restart_step >= 0 || request.options.save_state_files)
    return "generic CUDA HDG restart/state-file routes are unsupported";
  const auto valid_checkpoint_path = [](const std::string& text) noexcept
  {
    if (text.empty()) return true;
    try
    {
      const std::filesystem::path path{text};
      return !path.empty() && !path.filename().empty() &&
          text.find('\0') == std::string::npos;
    }
    catch (...) { return false; }
  };
  if (!valid_checkpoint_path(request.options.hdg_checkpoint) ||
      !valid_checkpoint_path(request.options.hdg_restart_checkpoint))
    return "generic CUDA HDG checkpoint paths are invalid";
  if (request.options.glvis)
    return "generic CUDA HDG GLVis is unsupported";
  if (!std::isfinite(request.options.dt))
    return "CUDA HDG --time-step must be finite";
  if (rodas3 && request.options.dt == 0.0)
    return "CUDA HDG RODAS3 requires a nonzero --time-step";
  if (rodas3 &&
      (!std::isfinite(request.options.time_atol) ||
       request.options.time_atol < 0.0 ||
       !std::isfinite(request.options.time_rtol) ||
       request.options.time_rtol < 0.0 ||
       (request.options.time_atol == 0.0 &&
        request.options.time_rtol == 0.0)))
    return "CUDA RODAS3 requires finite nonnegative --time-atol and "
           "--time-rtol with at least one positive tolerance";
  const bool adaptive_rodas3 = rodas3 && request.options.dt < 0.0;
  const bool checkpoint_requested =
      !request.options.hdg_checkpoint.empty() ||
      !request.options.hdg_restart_checkpoint.empty();
  if (checkpoint_requested && !rodas3)
    return "generic CUDA HDG checkpoint continuity is a RODAS3-only typed "
           "opt-in";
  if (adaptive_rodas3 &&
      (!std::isfinite(request.options.dt_min) ||
       request.options.dt_min <= 0.0))
    return "adaptive CUDA RODAS3 requires a finite positive --dt-min";
  if (!rodas3 && request.options.dt <= 0.0 &&
      (!std::isfinite(request.options.cfl) || request.options.cfl <= 0.0))
    return "CUDA CFL HDG time integration requires a finite positive --cfl-number";
  if (!std::isfinite(request.options.dt_min) || request.options.dt_min < 0.0)
    return "CUDA HDG --dt-min must be finite and non-negative";
  if (!std::isfinite(request.options.t_final) || request.options.t_final <= 0.0)
    return "CUDA HDG final time must be finite and positive";
  if (adaptive_rodas3 && request.options.dt_min > request.options.t_final)
    return "adaptive CUDA RODAS3 --dt-min must not exceed the final-time horizon";
  if (request.options.vtk_save &&
      (!std::isfinite(request.options.dt_save) || request.options.dt_save <= 0.0))
    return "CUDA HDG ParaView output requires a positive --time-step-save";
  if (request.options.order < dg::hdg::generic::minimum_polynomial_order ||
      request.options.order > dg::hdg::generic::maximum_polynomial_order)
    return "generic CUDA HDG supports polynomial orders one through three";
  if (request.options.ser_ref_levels < 0 || request.options.par_ref_levels < 0)
    return "generic CUDA HDG refinement levels must be nonnegative";
  if (!std::isfinite(request.case_config.gamma) ||
      request.case_config.gamma != mfem::igrlab::EulerDefaults::kGamma)
    return "generic CUDA HDG requires frozen Euler gamma 1.4";
  return {};
}

std::string ValidateCaseData(const HdgEulerCaseData& case_data,
    const dg::hdg::generic::EulerHdgNumericalPolicy& numerical_policy)
{
  if (!case_data.initial_condition)
    return "generic CUDA HDG requires an initial-condition coefficient";
  if (case_data.initial_condition->GetVDim() !=
      dg::hdg::generic::euler_2d_conserved_components)
    return "generic CUDA HDG requires a four-component Euler IC";
  if (case_data.static_source &&
      case_data.static_source->GetVDim() !=
          dg::hdg::generic::euler_2d_conserved_components)
    return "generic CUDA HDG requires a four-component Euler source";
  if (!dg::hdg::generic::ValidEulerHdgNumericalPolicy(numerical_policy))
    return "generic CUDA HDG rejects an invalid numerical policy";
  return {};
}

std::string CollectionName(const HdgEulerMainRequest& request,
                           const CudaEngineKind cuda_engine)
{
  return request.case_config.case_name + "_hdg_cuda_" +
      CudaEngineName(cuda_engine) + "_p_" +
      std::to_string(request.options.order) + "_ref_" +
      std::to_string(request.options.ser_ref_levels +
                     request.options.par_ref_levels);
}

HDGParaViewMetadata Metadata(
    const HdgTransactionalTimeAdapter::OutputEvent& event,
    const std::uint64_t step)
{
  if (event.cycle > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      step > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("CUDA HDG output counter exceeds int");
  return {static_cast<int>(event.cycle), static_cast<int>(step), event.time};
}

bool AbortPrepared(GenericEulerHdgCudaSession& session,
                   GenericEulerCudaPreparedStep& authority,
                   PreparedCudaOutput& native_export) noexcept
{
  return session.Abort(std::move(authority), native_export).status ==
      HdgEulerSessionStatus::aborted;
}

bool PrepareOutput(GenericEulerHdgCudaSession& session,
                   const double dt,
                   GenericEulerCudaPreparedStep& authority,
                   PreparedCudaOutput& native_export,
                   GenericEulerCudaPrepareFailureClass&
                       prepare_failure_class) noexcept
{
  return session.Prepare(
             dt, authority, native_export, &prepare_failure_class).status ==
      HdgEulerSessionStatus::prepared;
}

bool AssessPreparedOutput(
    GenericEulerHdgCudaSession& session,
    const GenericEulerCudaPreparedStep& authority,
    const PreparedCudaOutput& native_export,
    const dg::hdg::generic::HdgEulerPreparedAssessmentPolicy policy,
    dg::hdg::generic::HdgEulerPreparedAssessment& assessment) noexcept
{
  return session.AssessPrepared(
      authority, native_export, policy, assessment);
}

bool ExportPreparedOutputFields(
    GenericEulerHdgCudaSession& session,
    const GenericEulerCudaPreparedStep& authority,
    const PreparedCudaOutput& native_export,
    dg::hdg::generic::HdgEulerHostInteriorFields destination) noexcept
{
  return session.ExportPreparedFields(
      authority, native_export, destination);
}

bool AcknowledgePreparedOutput(
    GenericEulerHdgCudaSession& session,
    const GenericEulerCudaPreparedStep& authority,
    PreparedCudaOutput& native_export) noexcept
{
  return session.AcknowledgePreparedOutput(authority, native_export);
}

bool CommitPreparedOutput(GenericEulerHdgCudaSession& session,
                          GenericEulerCudaPreparedStep& authority,
                          PreparedCudaOutput& native_export) noexcept
{
  return session.Commit(std::move(authority), native_export).status ==
      HdgEulerSessionStatus::committed;
}

bool RejectPreparedOutput(GenericEulerHdgCudaSession& session,
                          GenericEulerCudaPreparedStep& authority,
                          PreparedCudaOutput& native_export) noexcept
{
  return session.RejectPrepared(
             std::move(authority), native_export).status ==
      HdgEulerSessionStatus::aborted;
}

} // namespace

int RunEulerMainBackend(const HdgEulerMainRequest& request)
{
  // Fail locally before any rank/root/collective call when MPI is absent or
  // finalized; a live request communicator owns every operation thereafter.
  if (!LiveSupportedCommunicator(request.communicator))
    return Reject(request, "invalid MPI communicator or lifecycle");

  std::string reason;
  HdgRegularizationRequest regularization_request{};
  ResolvedHdgRegularization resolved_regularization{};
  RegularizationSelection regularization{
      RegularizationKind::none, 0.0, 0.0};
  EulerHdgTimeTreatment time_treatment =
      EulerHdgTimeTreatment::backward_euler;
  CudaEngineKind cuda_engine = CudaEngineKind::ordinary;
  CudaTracePolicyDescriptor trace_policy{};
  int ranks = 0;
  try
  {
    regularization_request = CanonicalRegularization(request);
    time_treatment = CanonicalTimeTreatment(request);
    cuda_engine = CanonicalCudaEngine(request);
    reason = Validate(
        request, regularization_request, cuda_engine, time_treatment);
    if (reason.empty())
    {
      if (MPI_Comm_size(request.communicator, &ranks) != MPI_SUCCESS)
      {
        reason = "CUDA HDG rank-count resolution failed";
      }
      else
      {
        trace_policy = ResolveCudaTracePolicy(
            cuda_engine,
            regularization_request.kind == RegularizationKind::igr,
            ranks, request.options.order,
            DefaultCudaTracePolicyRequest(
                cuda_engine,
                regularization_request.kind == RegularizationKind::igr,
                ranks, time_treatment));
        if (!CudaTracePolicyDescriptorIsExact(trace_policy))
          reason = "CUDA HDG trace policy resolution failed";
      }
    }
  }
  catch (...) { reason = "rank-local CUDA HDG request validation failed"; }
  if (!CollectiveAll(request.communicator, reason.empty()))
    return Reject(request, reason.empty()
        ? "CUDA HDG request validation failed on another rank" : reason);

  std::string mesh_path;
  std::string consensus;
  try
  {
    mesh_path = ResolveCaseMesh(request.case_config, request.options.mesh_file);
    consensus = ConsensusText(request, mesh_path, regularization_request,
                              trace_policy, time_treatment);
  }
  catch (...) { reason = "CUDA HDG configuration resolution failed"; }
  if (!CollectiveAll(request.communicator, !consensus.empty()))
    return Reject(request, reason.empty()
        ? "CUDA HDG configuration resolution failed on another rank" : reason);
  if (!ExactTextConsensus(request.communicator, consensus))
    return Reject(request, "CUDA HDG configuration differs between MPI ranks");

  std::optional<mfem::Mesh> serial_mesh;
  std::string serialized_mesh;
  bool local_ok = false;
  try
  {
    serial_mesh.emplace(LoadSerialMesh(mesh_path, request.options.ser_ref_levels));
    std::ostringstream stream;
    stream.precision(std::numeric_limits<double>::max_digits10);
    serial_mesh->Print(stream);
    serialized_mesh = stream.str();
    local_ok = serial_mesh->Dimension() == 2 &&
        serial_mesh->SpaceDimension() == 2 && !serialized_mesh.empty();
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(request.communicator, local_ok))
    return Reject(request,
                  "CUDA HDG serial mesh loading failed on at least one rank");
  if (!ExactTextConsensus(request.communicator, serialized_mesh))
    return Reject(request,
                  "CUDA HDG serial mesh content differs between MPI ranks");

  std::unique_ptr<mfem::ParMesh> refined_mesh;
  MeshBuildMetadata refined_metadata;
  long long expected_global_elements = 0;
  long long local_refined_elements = 0;
  try
  {
    refined_metadata.mesh_path = mesh_path;
    refined_metadata.mesh_path_precedence = request.options.mesh_file.empty()
        ? MeshPathPrecedence::CaseDefault : MeshPathPrecedence::CommandLineOverride;
    refined_metadata.serial_refinement_levels = request.options.ser_ref_levels;
    refined_metadata.parallel_refinement_levels = request.options.par_ref_levels;
    refined_metadata.serial_element_count = serial_mesh->GetNE();
    refined_mesh = std::make_unique<mfem::ParMesh>(request.communicator,
                                                   *serial_mesh);
    serial_mesh->Clear();
    refined_metadata.local_elements_after_partitioning = refined_mesh->GetNE();
    RefineParMesh(*refined_mesh, request.options.par_ref_levels);
    refined_metadata.local_elements_after_parallel_refinement = refined_mesh->GetNE();
    expected_global_elements = refined_metadata.serial_element_count;
    for (int level = 0; level < request.options.par_ref_levels; ++level)
    {
      if (expected_global_elements >
          std::numeric_limits<long long>::max() / 4)
        throw std::overflow_error("CUDA HDG refined element census overflow");
      expected_global_elements *= 4;
    }
    local_refined_elements = refined_mesh->GetNE();
    local_ok = true;
  }
  catch (...)
  {
    reason = "CUDA HDG mesh construction failed";
    local_ok = false;
  }
  if (!CollectiveAll(request.communicator, local_ok))
    return Reject(request, reason.empty()
        ? "CUDA HDG mesh construction failed on another rank" : reason);

  // No rank can enter this census until every rank has completed local mesh
  // construction, refinement, and expected-count arithmetic.  The count is
  // therefore both collective-order safe and an observable refinement gate.
  long long global_refined_elements = 0;
  const bool census_complete = MPI_Allreduce(
      &local_refined_elements, &global_refined_elements, 1,
      MPI_LONG_LONG_INT, MPI_SUM, request.communicator) == MPI_SUCCESS;
  const bool census_matches = census_complete &&
      global_refined_elements == expected_global_elements;
  if (!CollectiveAll(request.communicator, census_matches))
    return Reject(request, "CUDA HDG parallel refinement census mismatch");

  std::optional<CaseContext> context;
  try
  {
    context.emplace(std::move(refined_mesh), std::move(refined_metadata));
    local_ok = true;
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(request.communicator, context.has_value() && local_ok))
    return Reject(request,
                  "CUDA HDG mesh context publication failed on another rank");

  try
  {
    resolved_regularization = ResolveHdgRegularization(
        context->Mesh(),
        CanonicalAlphaLengthScalePolicy(request),
        regularization_request);
    regularization = resolved_regularization.selection;
    local_ok = dg::hdg::generic::ValidRegularizationSelection(regularization);
  }
  catch (const std::exception& error)
  {
    reason = error.what();
    local_ok = false;
  }
  catch (...)
  {
    reason = "unknown CUDA HDG IGR mesh-scaling failure";
    local_ok = false;
  }
  if (!CollectiveAll(request.communicator, local_ok))
    return Reject(request, reason.empty()
        ? "CUDA HDG IGR mesh scaling failed on another rank" : reason);

  std::optional<HdgEulerCaseData> case_data;
  std::optional<FrozenBoundaryPlan> frozen_boundary;
  dg::hdg::generic::EulerHdgNumericalPolicy numerical_policy{};
  std::string case_data_consensus;
  try
  {
    case_data.emplace(request.make_case_data(*context));
    numerical_policy = case_data->numerical_policy;
    // Case factories own every numerical role except command-line
    // regularization and the shared case limiter pipeline.
    numerical_policy.stabilization = regularization;
    numerical_policy.positivity_limiter =
        CanonicalPositivityLimiterPolicy(request, time_treatment);
    reason = ValidateCaseData(*case_data, numerical_policy);
    if (!reason.empty()) throw std::invalid_argument(reason);
    frozen_boundary.emplace(FreezeBoundaryPlan(case_data->boundary_plan));
    case_data_consensus = CaseDataConsensusText(
        *case_data, *frozen_boundary, numerical_policy);
    local_ok = !case_data_consensus.empty();
  }
  catch (const std::exception& error)
  {
    reason = error.what();
    local_ok = false;
  }
  catch (...)
  {
    reason = "rank-local CUDA HDG case-data resolution failed";
    local_ok = false;
  }
  if (!CollectiveAll(request.communicator,
                     local_ok && case_data.has_value()))
    return Reject(request, reason.empty()
        ? "CUDA HDG case-data factory or validation failed on another rank"
        : reason);
  if (!ExactTextConsensus(request.communicator, case_data_consensus))
    return Reject(request, "CUDA HDG case data differs between MPI ranks");

  local_ok = false;
  std::unique_ptr<mfem::L2_FECollection> collection;
  std::unique_ptr<mfem::DG_Interface_FECollection> trace_collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> volume_space;
  std::unique_ptr<mfem::ParFiniteElementSpace> trace_space;
  std::unique_ptr<mfem::ParGridFunction> initial_state;
  std::unique_ptr<mfem::ParGridFunction> local_initial_trace;
  mfem::Vector initial_true_trace;
  std::unique_ptr<mfem::ParGridFunction> source_state;
  std::unique_ptr<mfem::ParGridFunction> published_state;
  std::optional<dg::hdg::generic::EulerBoundaryContext> boundary;
  try
  {
    collection = std::make_unique<mfem::L2_FECollection>(
        request.options.order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    volume_space = std::make_unique<mfem::ParFiniteElementSpace>(
        &context->Mesh(), collection.get(),
        dg::hdg::generic::euler_2d_conserved_components,
        mfem::Ordering::byNODES);
    initial_state = std::make_unique<mfem::ParGridFunction>(volume_space.get());
    trace_collection = std::make_unique<mfem::DG_Interface_FECollection>(
        request.options.order, 2, mfem::FiniteElement::VALUE,
        mfem::BasisType::GaussLegendre);
    trace_space = std::make_unique<mfem::ParFiniteElementSpace>(
        &context->Mesh(), trace_collection.get(),
        dg::hdg::generic::euler_2d_conserved_components,
        mfem::Ordering::byNODES);
    local_initial_trace =
        std::make_unique<mfem::ParGridFunction>(trace_space.get());
    published_state = std::make_unique<mfem::ParGridFunction>(volume_space.get());
    if (!ProjectEulerCoefficientL2(
            *volume_space, *initial_state, *case_data->initial_condition,
            numerical_policy.volume_projection_quadrature_order) ||
        initial_state->CheckFinite() != 0)
    {
      throw std::runtime_error("HDG L2 volume initial-state projection failed");
    }
    local_ok = volume_space->GetVSize() %
        dg::hdg::generic::euler_2d_conserved_components == 0;
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(request.communicator, local_ok))
    return Reject(request, "CUDA HDG volume initial-state projection failed");

  try
  {
    if (!ProjectEulerTrueTraceCoefficientL2(
            *trace_space, *local_initial_trace, initial_true_trace,
            *case_data->initial_condition,
            numerical_policy.true_trace_projection_quadrature_order))
    {
      throw std::runtime_error("HDG independent L2 true-trace projection failed");
    }
    if (case_data->static_source)
    {
      source_state = std::make_unique<mfem::ParGridFunction>(volume_space.get());
      source_state->ProjectCoefficient(*case_data->static_source);
      MFEM_VERIFY(source_state->CheckFinite() == 0,
                  "HDG refuses non-finite static-source coordinates.");
    }
    const int boundary_attributes = context->Mesh().bdr_attributes.Size() == 0
        ? 0 : context->Mesh().bdr_attributes.Max();
    boundary.emplace(dg::hdg::generic::LowerHdgEulerBoundaryPlan(
        frozen_boundary->plan, 2, request.case_config.gamma,
        boundary_attributes));
    local_ok = boundary->AttributeCount() ==
        static_cast<std::size_t>(boundary_attributes);
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(request.communicator, local_ok))
    return Reject(request,
                  "CUDA HDG state projection or boundary lowering failed");

  const BackendSelection backend{BackendKind::cuda, 0, cuda_engine};
  ResolvedHdgEulerCasePlan resolved_plan{};
  resolved_plan.request = EulerHdgRequest{
      &context->Mesh(), context->Comm(), request.options.order, backend,
      dg::hdg::generic::RequiredCapabilities(backend),
      {initial_state->HostRead(),
       static_cast<std::size_t>(initial_state->Size()),
       dg::hdg::generic::euler_2d_conserved_components},
      {initial_true_trace.HostRead(),
       static_cast<std::size_t>(initial_true_trace.Size()),
       dg::hdg::generic::euler_2d_conserved_components},
      source_state
          ? dg::hdg::generic::EulerFieldView{
                source_state->HostRead(),
                static_cast<std::size_t>(source_state->Size()),
                dg::hdg::generic::euler_2d_conserved_components}
          : dg::hdg::generic::EulerFieldView{
                nullptr, 0, dg::hdg::generic::euler_2d_conserved_components},
      boundary->View(), numerical_policy, time_treatment};

  const bool publish_checkpoints = !request.options.hdg_checkpoint.empty();
  const bool checkpoint_state_requested = publish_checkpoints ||
      !request.options.hdg_restart_checkpoint.empty();
  dg::hdg::runtime::StateHash projected_problem_identity{};
  if (checkpoint_state_requested &&
      !CollectiveProjectedProblemIdentity(
          request.communicator,
          resolved_plan.request.initial_state.data,
          resolved_plan.request.initial_state.scalar_count,
          resolved_plan.request.initial_true_trace.data,
          resolved_plan.request.initial_true_trace.scalar_count,
          resolved_plan.request.source.data,
          resolved_plan.request.source.scalar_count,
          source_state != nullptr, projected_problem_identity))
    return Reject(
        request,
        "CUDA HDG projected problem identity construction failed");
  const dg::hdg::runtime::StateHash checkpoint_numerical_identity =
      checkpoint_state_requested
      ? CheckpointNumericalIdentity(
            request, serialized_mesh, case_data_consensus,
            projected_problem_identity, regularization, trace_policy,
            time_treatment)
      : dg::hdg::runtime::StateHash{};

  // The opaque CUDA facade owns its engine storage. Converge allocation
  // failure before any rank enters the existing collective Initialize path.
  std::optional<GenericEulerHdgCudaSession> allocated_session;
  local_ok = true;
  try { allocated_session.emplace(); }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(request.communicator, local_ok))
    return Reject(request, "generic CUDA session allocation failed");
  auto& session = *allocated_session;
  if (session.Initialize(resolved_plan.request).status !=
      HdgEulerSessionStatus::initialized)
    return Reject(request, "generic CUDA session initialization failed");
  const bool rodas3 = time_treatment == EulerHdgTimeTreatment::rodas3;
  const bool adaptive_rodas3 = rodas3 && request.options.dt < 0.0;
  local_ok = !rodas3 ||
      dg::hdg::generic::HasHdgEulerSessionCapabilities(
          session.capabilities(),
          dg::hdg::generic::HdgEulerSessionCapabilityBit(
              dg::hdg::generic::HdgEulerSessionCapability::
                  prepared_embedded_assessment));
  if (!CollectiveAll(request.communicator, local_ok))
    return Reject(
        request,
        "generic CUDA RODAS3 session lacks prepared embedded assessment");
  std::optional<LoadedCudaCheckpoint> restored_checkpoint;
  if (!request.options.hdg_restart_checkpoint.empty())
  {
    try
    {
      restored_checkpoint.emplace();
      local_ok = LoadCudaCheckpoint(
          request.communicator, request.options.hdg_restart_checkpoint,
          checkpoint_numerical_identity, static_cast<std::uint32_t>(ranks),
          *restored_checkpoint);
    }
    catch (...) { local_ok = false; }
    if (!CollectiveAll(request.communicator, local_ok))
      return Reject(request,
                    "generic CUDA HDG checkpoint decode/identity validation failed");
    local_ok = session.Restore(
        restored_checkpoint->local_metadata,
        GenericEulerCudaCheckpointPayloadViewV1{
            restored_checkpoint->local_interior.data(),
            restored_checkpoint->local_interior.size(),
            restored_checkpoint->local_owned_true_trace.data(),
            restored_checkpoint->local_owned_true_trace.size()}).status ==
        HdgEulerSessionStatus::committed;
    const auto restored_coordinates = session.coordinates();
    local_ok = local_ok &&
        restored_coordinates.time ==
            restored_checkpoint->time_state.cursor.current_time &&
        restored_coordinates.step ==
            restored_checkpoint->time_state.cursor.committed_steps &&
        restored_coordinates.generation ==
            restored_checkpoint->local_metadata.descriptor.coordinates.generation;
    if (!CollectiveAll(request.communicator, local_ok))
      return Reject(request,
                    "generic CUDA HDG checkpoint restore failed closed");
  }
  std::unique_ptr<HdgTransactionalTimeAdapter> time_driver;
  std::unique_ptr<HdgEulerParaViewFields> output_fields;
  std::unique_ptr<HDGParaViewSeries> output;
  try
  {
    const bool cfl_control = !rodas3 && request.options.dt <= 0.0;
    const bool partitioned_backward_euler = time_treatment ==
        EulerHdgTimeTreatment::partitioned_backward_euler;
    resolved_plan.time_control = adaptive_rodas3
        ? HdgTransactionalTimeAdapter::TimeControl::Adaptive(
              0.0, request.options.t_final, std::abs(request.options.dt))
        : cfl_control
        ? HdgTransactionalTimeAdapter::TimeControl::CFL(
              0.0, request.options.t_final, request.options.cfl)
        : HdgTransactionalTimeAdapter::TimeControl::Fixed(
              0.0, request.options.t_final, request.options.dt);
    resolved_plan.output_schedule = request.options.vtk_save
        ? HdgTransactionalTimeAdapter::OutputSchedule::FixedInterval(
              request.options.dt_save,
              time::backend::InitialOutputPolicy::Emit,
              time::backend::FinalOutputPolicy::Emit)
        : HdgTransactionalTimeAdapter::OutputSchedule::Disabled(
              time::backend::InitialOutputPolicy::Suppress,
              time::backend::FinalOutputPolicy::Suppress);
    resolved_plan.time_mode = adaptive_rodas3
        ? HdgEulerTimeMode::AdaptiveRodas3
        : rodas3 ? HdgEulerTimeMode::FixedRodas3
        : partitioned_backward_euler
        ? (cfl_control
              ? HdgEulerTimeMode::CFLPartitionedBackwardEuler
              : HdgEulerTimeMode::FixedPartitionedBackwardEuler)
        : (cfl_control ? HdgEulerTimeMode::CFLBackwardEuler
                       : HdgEulerTimeMode::FixedBackwardEuler);
    const mfem::io::VTKOutputConfig vtk = MakeCaseVTKConfig(request.case_config);
    resolved_plan.output = {request.options.vtk_save, vtk.prefix_path,
                            CollectionName(request, cuda_engine)};
    if (adaptive_rodas3)
    {
      HdgTransactionalTimeAdapter::AdaptiveControllerConfig adaptive_config;
      adaptive_config.minimum_dt = request.options.dt_min;
      adaptive_config.maximum_dt = request.options.t_final;
      time_driver = restored_checkpoint
          ? std::make_unique<HdgTransactionalTimeAdapter>(
                resolved_plan.time_control, resolved_plan.output_schedule,
                adaptive_config, restored_checkpoint->time_state,
                HdgTimeControlMode::AdaptiveRodas3)
          : std::make_unique<HdgTransactionalTimeAdapter>(
                resolved_plan.time_control, resolved_plan.output_schedule,
                adaptive_config, HdgTimeControlMode::AdaptiveRodas3);
    }
    else
    {
      const HdgTimeControlMode mode = rodas3
          ? HdgTimeControlMode::FixedRodas3
          : partitioned_backward_euler
          ? (cfl_control
                ? HdgTimeControlMode::CFLPartitionedBackwardEuler
                : HdgTimeControlMode::FixedPartitionedBackwardEuler)
          : (cfl_control ? HdgTimeControlMode::CFLBackwardEuler
                         : HdgTimeControlMode::FixedBackwardEuler);
      time_driver = restored_checkpoint
          ? std::make_unique<HdgTransactionalTimeAdapter>(
                resolved_plan.time_control, resolved_plan.output_schedule,
                restored_checkpoint->time_state, mode)
          : std::make_unique<HdgTransactionalTimeAdapter>(
                resolved_plan.time_control, resolved_plan.output_schedule,
                mode);
    }
    if (request.options.vtk_save)
    {
      const int levels = vtk.levels_of_detail >= 0 ? vtk.levels_of_detail
                                                     : request.options.order + 1;
      const int output_size = initial_state->Size();
      if (output_size < 0 || published_state->Size() != output_size)
        throw std::runtime_error("CUDA HDG output-state extent mismatch");
      if (output_size != 0)
      {
        const double* const source = initial_state->HostRead();
        double* const destination = published_state->HostWrite();
        if (source == nullptr || destination == nullptr)
          throw std::runtime_error("CUDA HDG output-state storage is null");
        std::copy_n(source, output_size, destination);
      }
      output_fields = std::make_unique<HdgEulerParaViewFields>(
          context->Mesh(), *collection, *published_state,
          regularization.kind == RegularizationKind::igr,
          request.options.vtk_zhang_shu_diagnostics &&
              numerical_policy.positivity_limiter.kind ==
                  EulerHdgPositivityLimiterKind::zhang_shu,
          request.case_config.gamma,
          request.case_config.rk3_config.rho_min);
      HDGParaViewOptions output_options{
          resolved_plan.output.prefix_path,
          resolved_plan.output.collection_name, vtk.format, levels,
          vtk.high_order_output};
      if (restored_checkpoint)
      {
        if (restored_checkpoint->time_state.output_cycle >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
          throw std::runtime_error(
              "CUDA HDG restart output cycle exceeds ParaView range");
        output_options.restart = HDGParaViewOptions::RestartSeed{
            static_cast<int>(restored_checkpoint->time_state.output_cycle),
            restored_checkpoint->time_state.cursor.current_time};
      }
      output = std::make_unique<HDGParaViewSeries>(HDGParaViewRequest{
          context->Mesh(), output_fields->Fields(),
          std::move(output_options)});
    }
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(request.communicator, time_driver != nullptr &&
      (!request.options.vtk_save || output != nullptr)))
    return Reject(request, "CUDA HDG time/output setup failed");

  std::optional<HdgTransactionalTimeAdapter::OutputEvent> initial;
  try
  {
    // A restored cursor has already accounted for its initial publication.
    // Re-emitting cycle zero would both overwrite committed output and make a
    // split process diverge from the persisted output-cycle authority.
    initial = restored_checkpoint
        ? std::nullopt
        : time_driver->InitialOutput();
    local_ok = !initial || output != nullptr;
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(request.communicator, local_ok))
    return Reject(request, "CUDA HDG initial output preview failed");
  if (!ExactBoolConsensus(request.communicator, initial.has_value()))
    return Reject(request, "CUDA HDG initial output differs between ranks");
  if (initial)
  {
    dg::hdg::generic::HdgEulerHostInteriorFields destination{};
    local_ok = output_fields && output_fields->FreshHostOutput(destination);
    if (!CollectiveAll(request.communicator, local_ok))
      return Reject(request, "initial CUDA committed-state output view failed");
    local_ok = session.ExportCommittedFields(destination);
    if (!CollectiveAll(request.communicator, local_ok))
      return Reject(request, "initial CUDA committed-state export failed");
    local_ok = output_fields->UpdateDerived();
    if (!CollectiveAll(request.communicator, local_ok))
      return Reject(request, "initial CUDA derived Euler field update failed");
    try { output->Write(Metadata(*initial, 0)); local_ok = true; }
    catch (...) { local_ok = false; }
    if (!CollectiveAll(request.communicator, local_ok))
      return Reject(request, "CUDA HDG initial ParaView output failed");
  }

  double minimum_accepted_dt = std::numeric_limits<double>::infinity();
  double maximum_accepted_dt = 0.0;
  while (!time_driver->Finished())
  {
    double proposed_dt = 0.0;
    if (time_driver->UsesCflControl())
    {
      const auto diagnostics = session.diagnostics();
      const bool diagnostics_ready =
          dg::hdg::generic::HasHdgEulerSessionCapabilities(
              session.capabilities(),
              dg::hdg::generic::HdgEulerSessionCapabilityBit(
                  dg::hdg::generic::HdgEulerSessionCapability::
                      committed_cfl_diagnostics)) &&
          dg::hdg::generic::IsValidHdgEulerCommittedDiagnostics(
              diagnostics, session.capabilities()) &&
          diagnostics.coordinates.time == time_driver->time() &&
          diagnostics.coordinates.step == time_driver->solver_step() &&
          diagnostics.coordinates.generation == diagnostics.wave_generation;
      proposed_dt = diagnostics.backend_stable_dt * request.options.cfl;
      const bool proposal_valid = diagnostics_ready &&
          std::isfinite(proposed_dt) && proposed_dt > 0.0 &&
          (request.options.dt_min <= 0.0 || proposed_dt >= request.options.dt_min);
      const bool proposal_consensus = ExactDoubleConsensus(
          request.communicator, proposed_dt);
      if (!CollectiveAll(request.communicator,
                         proposal_valid && proposal_consensus))
        return Reject(request,
            "committed CUDA CFL proposal is invalid or below --dt-min");
    }
    const HdgTransactionalTimeAdapter::StepPlan plan = adaptive_rodas3
        ? time_driver->BeginAdaptiveStep()
        : time_driver->BeginStep(proposed_dt, request.options.dt_min);
    const auto coordinates_before_prepare = session.coordinates();
    GenericEulerCudaPreparedStep authority;
    PreparedCudaOutput native_export{};
    PreparedCudaCheckpointPayload checkpoint_payload;
    GenericEulerCudaPrepareFailureClass prepare_failure_class =
        GenericEulerCudaPrepareFailureClass::not_failed;
    const bool prepared_locally =
        PrepareOutput(session, plan.dt, authority, native_export,
                      prepare_failure_class);
    if (!CollectiveAll(request.communicator, prepared_locally))
    {
      const bool uniformly_unprepared =
          CollectiveAll(request.communicator, !prepared_locally);
      const bool coordinates_unchanged = SameCoordinates(
          coordinates_before_prepare, session.coordinates());
      if (!uniformly_unprepared ||
          !CollectiveAll(request.communicator, coordinates_unchanged))
      {
        time_driver->RecordFailure(HdgStepFailurePhase::NonlinearSolve);
        return Reject(
            request,
            "generic CUDA prepare failure did not preserve collective coordinates");
      }
      if (!adaptive_rodas3)
      {
        time_driver->RecordFailure(HdgStepFailurePhase::NonlinearSolve);
        return Reject(request, "generic CUDA prepared step failed");
      }
      local_ok = session.RejectFailedPrepare().status ==
          HdgEulerSessionStatus::aborted;
      const bool abandoned_coordinates_unchanged = SameCoordinates(
          coordinates_before_prepare, session.coordinates());
      if (!CollectiveAll(
              request.communicator,
              local_ok && abandoned_coordinates_unchanged))
      {
        return Reject(
            request,
            "CUDA adaptive failed-prepare rejection did not preserve state");
      }
      const HdgTransactionalTimeAdapter::AdaptiveAttempt failed_attempt{};
      const std::string rejection = FinalizeAdaptiveRejectionOrReason(
          request.communicator, *time_driver, plan, failed_attempt,
          PrepareFailureClassName(prepare_failure_class));
      if (!rejection.empty()) return Reject(request, rejection);
      continue;
    }

    // A test executable linked to this exact production backend may supply a
    // private weak definition to exercise process-boundary rollback after a
    // prescribed committed generation. Production executables have no
    // definition and no CLI/environment stop control. The prepared candidate
    // is aborted before assessment/output and committed coordinates must stay
    // unchanged.
    if (HasTestOnlyPrepareBoundaryFailureHook())
    {
      const bool inject_prepare_failure = TestOnlyPrepareBoundaryFailure(
          time_driver->solver_step(), time_driver->time());
      const bool inject_on_all_ranks =
          CollectiveAll(request.communicator, inject_prepare_failure);
      const bool inject_on_no_ranks =
          CollectiveAll(request.communicator, !inject_prepare_failure);
      if (!inject_on_all_ranks && !inject_on_no_ranks)
      {
        local_ok = AbortPrepared(session, authority, native_export);
        return Reject(
            request,
            local_ok
                ? "test-only CUDA prepare-boundary failure differs between ranks"
                : "CUDA prepared abort failed after test-only gate disagreement");
      }
      if (inject_on_all_ranks)
      {
        local_ok = AbortPrepared(session, authority, native_export);
        const bool coordinates_unchanged = SameCoordinates(
            coordinates_before_prepare, session.coordinates());
        time_driver->RecordFailure(HdgStepFailurePhase::NonlinearSolve);
        if (!CollectiveAll(
                request.communicator,
                local_ok && coordinates_unchanged))
        {
          return Reject(
              request,
              "CUDA prepared abort failed after test-only prepare-boundary failure");
        }
        return Reject(
            request,
            "test-only CUDA prepare-boundary failure injected");
      }
    }

    std::optional<HdgTransactionalTimeAdapter::AdaptiveDecision>
        adaptive_decision;
    if (rodas3)
    {
      const dg::hdg::generic::HdgEulerPreparedAssessmentPolicy
          assessment_policy{request.options.time_atol,
                            request.options.time_rtol,
                            kRodas3AlgebraicCorrectionFraction};
      dg::hdg::generic::HdgEulerPreparedAssessment assessment{};
      const bool assessment_succeeded = AssessPreparedOutput(session,
          authority, native_export, assessment_policy, assessment);
      const bool assessment_values_valid = CollectiveAll(
          request.communicator,
          dg::hdg::generic::IsValidHdgEulerPreparedAssessment(assessment));
      const bool assessment_values_collective = ExactDoubleConsensus(
              request.communicator,
              assessment.normalized_differential_error) &&
          ExactDoubleConsensus(
              request.communicator,
              assessment.normalized_algebraic_correction);
      const bool finite_solver_gate = CollectiveAll(
          request.communicator,
          assessment_succeeded && assessment.finite_gate &&
              assessment.solver_gate);
      const bool residual_gate = CollectiveAll(
          request.communicator, assessment.residual_gate);
      const bool admissibility_gate = CollectiveAll(
          request.communicator, assessment.admissibility_gate);
      const bool algebraic_closure_gate = CollectiveAll(
          request.communicator,
          assessment.algebraic_closure_gate);
      const bool algebraic_correction_gate = CollectiveAll(
          request.communicator,
          assessment.algebraic_correction_gate);
      const bool closure_gate = algebraic_closure_gate &&
          algebraic_correction_gate;

      if (adaptive_rodas3)
      {
        HdgTransactionalTimeAdapter::AdaptiveAttempt attempt{};
        if (assessment_values_valid && assessment_values_collective)
          attempt.collective_normalized_error =
              assessment.normalized_differential_error;
        attempt.collective_solver_success = finite_solver_gate;
        attempt.collective_residuals_satisfied = residual_gate;
        attempt.collective_admissible = admissibility_gate;
        attempt.collective_algebraic_closure_satisfied = closure_gate;
        try
        {
          adaptive_decision.emplace(
              time_driver->PreviewAdaptiveDecision(attempt));
          local_ok = true;
        }
        catch (...)
        {
          local_ok = false;
        }
        if (!CollectiveAll(request.communicator, local_ok) ||
            !ExactDoubleConsensus(
                request.communicator, adaptive_decision->next_dt()))
        {
          local_ok = AbortPrepared(session, authority, native_export);
          if (!CollectiveAll(request.communicator, local_ok))
            return Reject(
                request,
                "CUDA abort failed after adaptive decision failure");
          return Reject(
              request, "adaptive CUDA RODAS3 decision construction failed");
        }
        const bool collectively_accepted = CollectiveAll(
            request.communicator, adaptive_decision->Accepted());
        const bool collectively_rejected = CollectiveAll(
            request.communicator, !adaptive_decision->Accepted());
        if (!collectively_accepted && !collectively_rejected)
        {
          local_ok = AbortPrepared(session, authority, native_export);
          if (!CollectiveAll(request.communicator, local_ok))
            return Reject(
                request,
                "CUDA abort failed after adaptive decision divergence");
          return Reject(
              request, "adaptive CUDA RODAS3 acceptance differs between ranks");
        }
        if (collectively_rejected)
        {
          local_ok = RejectPreparedOutput(session, authority, native_export);
          const bool coordinates_unchanged = SameCoordinates(
              coordinates_before_prepare, session.coordinates());
          if (!CollectiveAll(
                  request.communicator,
                  local_ok && coordinates_unchanged))
          {
            return Reject(
                request, "CUDA adaptive prepared-step rejection failed");
          }
          const HdgTransactionalTimeAdapter::AdaptiveAttempt rejected_attempt{
              assessment_values_valid && assessment_values_collective
                  ? assessment.normalized_differential_error
                  : std::numeric_limits<double>::infinity(),
              finite_solver_gate,
              residual_gate,
              admissibility_gate,
              closure_gate};
          const std::string rejection = FinalizeAdaptiveRejectionOrReason(
              request.communicator, *time_driver, plan, rejected_attempt,
              PreparedAssessmentFailureClass(
                  assessment_values_valid, assessment_values_collective,
                  finite_solver_gate, residual_gate, admissibility_gate,
                  algebraic_closure_gate, algebraic_correction_gate));
          if (!rejection.empty()) return Reject(request, rejection);
          continue;
        }
      }
      else
      {
        const bool fixed_assessment_accepted = CollectiveAll(
            request.communicator,
            assessment_values_valid && assessment_values_collective &&
                finite_solver_gate && residual_gate && admissibility_gate &&
                closure_gate);
        if (!fixed_assessment_accepted)
        {
          local_ok = AbortPrepared(session, authority, native_export);
          if (!CollectiveAll(request.communicator, local_ok))
            return Reject(
                request,
                "CUDA abort failed after fixed RODAS3 assessment rejection");
          return Reject(
              request,
              "fixed CUDA RODAS3 assessment rejected its algebraic correction or solver gates");
        }
      }
    }

    std::optional<HdgTransactionalTimeAdapter::OutputEvent> preview;
    try
    {
      preview = time_driver->PreviewFinalizedOutput();
      local_ok = !preview || output != nullptr;
    }
    catch (...) { local_ok = false; }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      local_ok = AbortPrepared(session, authority, native_export);
      time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
      return Reject(request,
          local_ok ? "CUDA output preview failed"
                   : "CUDA prepared abort failed after preview failure");
    }
    if (!ExactBoolConsensus(request.communicator, preview.has_value()))
    {
      local_ok = AbortPrepared(session, authority, native_export);
      time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
      return Reject(request,
          local_ok ? "CUDA output preview differs between ranks"
                   : "CUDA prepared abort failed after preview disagreement");
    }
    if (preview)
    {
      dg::hdg::generic::HdgEulerHostInteriorFields destination{};
      local_ok = output_fields && output_fields->FreshHostOutput(destination);
      if (!CollectiveAll(request.communicator, local_ok))
      {
        local_ok = AbortPrepared(session, authority, native_export);
        time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
        return Reject(request,
            local_ok ? "CUDA prepared-state output view failed"
                     : "CUDA prepared abort failed after output-view failure");
      }
      local_ok = ExportPreparedOutputFields(
          session, authority, native_export, destination);
      if (!CollectiveAll(request.communicator, local_ok))
      {
        local_ok = AbortPrepared(session, authority, native_export);
        time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
        return Reject(request,
            local_ok ? "CUDA prepared-state export failed"
                     : "CUDA prepared abort failed after export failure");
      }
      local_ok = output_fields->UpdateDerived();
      if (!CollectiveAll(request.communicator, local_ok))
      {
        local_ok = AbortPrepared(session, authority, native_export);
        time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
        return Reject(request,
            local_ok ? "CUDA prepared derived-field update failed"
                     : "CUDA prepared abort failed after derived-field failure");
      }
    }
    if (publish_checkpoints)
    {
      constexpr std::uint64_t maximum_coefficients =
          dg::hdg::runtime::app::
              generic_hdg_cuda_checkpoint_maximum_bytes_v1 /
          sizeof(double);
      local_ok = native_export.interior_count > 0 &&
          native_export.owned_true_trace_count > 0 &&
          native_export.interior_count <= maximum_coefficients &&
          native_export.owned_true_trace_count <=
              maximum_coefficients - native_export.interior_count &&
          native_export.interior_count <=
              std::numeric_limits<std::size_t>::max() &&
          native_export.owned_true_trace_count <=
              std::numeric_limits<std::size_t>::max();
      if (local_ok)
      {
        try
        {
          checkpoint_payload.interior.resize(
              static_cast<std::size_t>(native_export.interior_count));
          checkpoint_payload.owned_true_trace.resize(
              static_cast<std::size_t>(
                  native_export.owned_true_trace_count));
        }
        catch (...) { local_ok = false; }
      }
      if (!CollectiveAll(request.communicator, local_ok))
      {
        local_ok = AbortPrepared(session, authority, native_export);
        time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
        return Reject(
            request,
            local_ok ? "CUDA checkpoint payload allocation failed"
                     : "CUDA prepared abort failed after checkpoint allocation failure");
      }
      local_ok = session.ExportPreparedCheckpointPayload(
          authority, native_export,
          GenericEulerCudaMutableCheckpointPayloadV1{
              checkpoint_payload.interior.data(),
              checkpoint_payload.interior.size(),
              checkpoint_payload.owned_true_trace.data(),
              checkpoint_payload.owned_true_trace.size()});
      if (!CollectiveAll(request.communicator, local_ok))
      {
        local_ok = AbortPrepared(session, authority, native_export);
        time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
        return Reject(
            request,
            local_ok ? "CUDA prepared checkpoint payload export failed"
                     : "CUDA prepared abort failed after checkpoint export failure");
      }
      checkpoint_payload.staged = true;
    }
    if (!AcknowledgePreparedOutput(session, authority, native_export))
    {
      local_ok = AbortPrepared(session, authority, native_export);
      time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
      return Reject(request,
          local_ok ? "CUDA prepared output acknowledgement failed"
                   : "CUDA prepared abort failed after acknowledgement failure");
    }
    std::optional<HdgTransactionalTimeAdapter::OutputEvent> preflight;
    try
    {
      preflight = adaptive_rodas3
          ? time_driver->PreflightAdaptiveSuccess(*adaptive_decision)
          : time_driver->PreflightSuccess();
      local_ok = preflight.has_value() == preview.has_value();
      if (local_ok && preview)
        local_ok = preflight->cycle == preview->cycle &&
            preflight->time == preview->time &&
            preflight->is_initial == preview->is_initial &&
            preflight->is_final == preview->is_final;
    }
    catch (...) { local_ok = false; }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      local_ok = AbortPrepared(session, authority, native_export);
      return Reject(
          request,
          local_ok ? "CUDA time transaction preflight failed"
                   : "CUDA prepared abort failed after time preflight failure");
    }
    if (!CommitPreparedOutput(session, authority, native_export))
      return Reject(request, "generic CUDA prepared commit failed");
    // Publication remains post-commit.  On an output step the explicit export
    // copied the frozen candidate into this private host view, acknowledgement
    // bound that export to the affine authority, and Commit published that
    // exact candidate without mutating the host bytes.  A no-output step needs
    // only the already-staged native metadata acknowledgement.  The explicit
    // checkpoint opt-in separately stages exact dynamic interior and trace
    // payloads while the same affine candidate remains abortable.

    std::optional<HdgTransactionalTimeAdapter::OutputEvent> finalized;
    try
    {
      finalized = adaptive_rodas3
          ? time_driver->FinalizeAdaptiveSuccess(*adaptive_decision)
          : time_driver->FinalizeSuccess();
      local_ok = true;
    }
    catch (...) { local_ok = false; }
    if (!CollectiveAll(request.communicator, local_ok))
      return Reject(request,
                    "CUDA time transaction finalization failed after commit");
    local_ok = preview.has_value() == finalized.has_value();
    if (local_ok && preview)
      local_ok = preview->cycle == finalized->cycle &&
          preview->time == finalized->time &&
          preview->is_initial == finalized->is_initial &&
          preview->is_final == finalized->is_final;
    if (!CollectiveAll(request.communicator, local_ok))
      return Reject(request,
                    "CUDA finalized output differs from its preview");
    if (output && finalized)
    {
      try { output->Write(Metadata(*finalized, time_driver->solver_step())); local_ok = true; }
      catch (...) { local_ok = false; }
      if (!CollectiveAll(request.communicator, local_ok))
        return Reject(request, "CUDA ParaView output failed after commit");
    }
    if (publish_checkpoints)
    {
      HdgTransactionalTimeAdapter::RestartState time_state{};
      try
      {
        time_state = time_driver->CheckpointState();
        local_ok = checkpoint_payload.staged &&
            native_export.successful_commit &&
            native_export.descriptor.coordinates.time ==
                time_state.cursor.current_time &&
            native_export.descriptor.coordinates.step ==
                time_state.cursor.committed_steps;
      }
      catch (...) { local_ok = false; }
      if (!CollectiveAll(request.communicator, local_ok) ||
          !PublishCudaCheckpoint(
              request.communicator, request.options.hdg_checkpoint,
              checkpoint_numerical_identity, time_state, native_export,
              checkpoint_payload))
        return Reject(request,
                      "generic CUDA HDG checkpoint publication failed");
    }
    minimum_accepted_dt = std::min(minimum_accepted_dt, plan.dt);
    maximum_accepted_dt = std::max(maximum_accepted_dt, plan.dt);
    const std::uint64_t committed_step = time_driver->solver_step();
    const bool report_progress = request.options.vis_steps > 0 &&
        (committed_step %
             static_cast<std::uint64_t>(request.options.vis_steps) == 0 ||
         time_driver->Finished());
    if (report_progress && IsRoot(request.communicator))
    {
      std::cout << "[hdg-progress] event=accepted"
                << " step=" << committed_step
                << " time=" << time_driver->time()
                << " accepted_dt=" << plan.dt;
      if (adaptive_rodas3)
      {
        try
        {
          const auto progress_state = time_driver->AdaptiveCheckpointState();
          std::cout << " next_dt=" << progress_state.next_dt
                    << " rejected_attempts="
                    << progress_state.rejected_attempts;
        }
        catch (...) { std::cout << " controller_state=unavailable"; }
      }
      std::cout << '\n';
    }
  }

  HdgTransactionalTimeAdapter::AdaptiveControllerState adaptive_state{};
  if (adaptive_rodas3)
  {
    try
    {
      adaptive_state = time_driver->AdaptiveCheckpointState();
      local_ok = adaptive_state.accepted_steps ==
          time_driver->solver_step();
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
      return Reject(
          request, "adaptive CUDA RODAS3 final controller state is invalid");
  }

  if (IsRoot(request.communicator))
  {
    const auto trace_preconditioner = trace_policy.trace_preconditioner;
    const bool trace_uses_ilu =
        CudaTracePreconditionerIsIlu(trace_preconditioner);
    const auto ilu_accounting_kind = trace_uses_ilu
        ? CudaIluAccountingKind::derived_bounds_v1
        : CudaIluAccountingKind::unavailable;
    const auto ilu_factor_reuse_policy = trace_policy.ilu_factor_reuse_policy;
    const auto ilu_triangular_solve_kind =
        trace_policy.ilu_triangular_solve_kind;
    const int ilu_lower_jacobi_iterations =
        trace_policy.ilu_lower_jacobi_iterations;
    const int ilu_upper_jacobi_iterations =
        trace_policy.ilu_upper_jacobi_iterations;
    std::cout << "[hdg-main] status=Success step=" << time_driver->solver_step()
              << " time=" << time_driver->time() << " output_cycle="
              << time_driver->output_cycle() << " cuda_engine="
              << CudaEngineName(cuda_engine)
              << " time_treatment="
              << dg::hdg::generic::EulerHdgTimeTreatmentName(time_treatment)
              << " requested_trace_policy="
              << CudaTracePolicyKindName(trace_policy.requested_selection)
              << " resolved_trace_policy="
              << CudaTracePolicyKindName(trace_policy.resolved_selection)
              << " trace_solver=" << TraceSolverName(trace_policy)
              << " fallback_count=0 trace_preconditioner="
              << CudaTracePreconditionerName(trace_preconditioner)
              << " trace_ilu_type="
              << CudaIluType(trace_preconditioner)
              << " trace_ilu_level_of_fill="
              << CudaIluLevelOfFill(trace_preconditioner)
              << " trace_ilu_local_reordering="
              << trace_policy.ilu_local_reordering
              << " trace_ilu_application_iterations="
              << CudaIluApplicationIterations(trace_preconditioner)
              << " trace_ilu_accounting_kind="
              << CudaIluAccountingKindName(ilu_accounting_kind)
              << " trace_ilu_factor_reuse_policy="
              << CudaIluFactorReusePolicyName(ilu_factor_reuse_policy)
              << " trace_ilu_triangular_solve_kind="
              << CudaIluTriangularSolveKindName(ilu_triangular_solve_kind)
              << " trace_ilu_lower_jacobi_iterations="
              << ilu_lower_jacobi_iterations
              << " trace_ilu_upper_jacobi_iterations="
              << ilu_upper_jacobi_iterations
              << " regularization="
              << (regularization.kind == RegularizationKind::igr ? "igr" : "none");
    std::cout << " positivity_limiter="
              << (numerical_policy.positivity_limiter.kind ==
                      EulerHdgPositivityLimiterKind::zhang_shu
                  ? "zhang-shu" : "none")
              << " zhang_shu_points="
              << (numerical_policy.positivity_limiter.point_set ==
                      EulerHdgZhangShuPointSet::quadrature
                  ? "quadrature" : "consumer");
    if (regularization.kind == RegularizationKind::igr)
    {
      std::cout << " c_alpha=" << regularization_request.c_alpha
                << " alpha_effective=" << regularization.alpha
                << " alpha_length_policy="
                << IGRAlphaLengthScalePolicyToString(
                     resolved_regularization.alpha_info.policy)
                << " h_eff=" << resolved_regularization.alpha_info.dx_eff
                << " euler_trace_penalty=" << regularization.beta;
    }
    if (adaptive_rodas3)
      std::cout << " rejected_attempts=" << adaptive_state.rejected_attempts;
    std::cout << " minimum_accepted_dt=" << minimum_accepted_dt
              << " maximum_accepted_dt=" << maximum_accepted_dt;
    std::cout << '\n';
  }
  return 0;
}

} // namespace dg_sim::hdg

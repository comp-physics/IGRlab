#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "utils/profiling/nvtx.hpp"
#include "test_cuda_trace_solver_routing.hpp"

#include <_hypre_utilities.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <mfem.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/Umpire.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{

// Test-only POD copied while the prepared authority is live.  Keeping this
// observation in the benchmark translation unit avoids a production
// diagnostic API; its small host copy remains inside the reported full
// Prepare-through-Commit wall interval so overlapping device work is counted.
struct CudaAcceptedStepBenchmarkObservation
{
  CudaStepEvidence evidence{};
  StateHash candidate_interior_hash{};
  StateHash candidate_trace_hash{};
  double maximum_conservation_error =
      std::numeric_limits<double>::infinity();
  double maximum_conservation_relative_error =
      std::numeric_limits<double>::infinity();
  TensorVolumeMmaWorkPlan tensor_volume_work{};
  bool candidate_ready = false;
  bool finite = false;
  bool admissible = false;
  bool recovery = false;
  bool true_residual = false;
  bool conservation = false;
  bool telemetry = false;
  bool success = false;
};

struct CudaAcceptedStepCommittedHashes
{
  StateHash interior{};
  StateHash trace{};
  StateHash local_interior{};
  StateHash local_trace{};
  std::uint64_t generation = 0;
  bool valid = false;
};

struct CudaAcceptedStepProblemMetadata
{
  int elements = 0;
  int interior_components = 0;
  int trace_components = 0;
  int scalar_interior_dofs = 0;
  int scalar_face_dofs = 0;
  int volume_assembly_order = 0;
  int volume_points = 0;
  int interior_unknowns = 0;
  int true_trace_unknowns = 0;
  int global_true_trace_unknowns = 0;
  bool igr_active = false;
};

// The production facade and engine already grant this exact qualification
// friend.  This executable defines only the narrow initialization and
// observation seams needed to bind the ablation policy, work counts, and
// numerical identity to every sample.
struct CudaQualificationTestAccess
{
  static cuda::GenericEulerHdgCudaSession::result_type
  InitializeBenchmarkTracePolicy(
      cuda::GenericEulerHdgCudaSession& session,
      const ::dg::hdg::generic::EulerHdgRequest& request,
      const CudaTracePolicyKind trace_policy) noexcept
  {
    return session.InitializeRequest(request, trace_policy);
  }

  template <typename Engine>
  static bool CapturePrepared(
      Engine& engine,
      CudaAcceptedStepBenchmarkObservation& observation) noexcept
  {
    observation = {};
    if (!engine.initialized_ || !engine.pending_.active ||
        !engine.pending_.staging || !engine.pending_.prestage_allowed ||
        !engine.pending_.checkpoint_staged)
      return false;

    const auto& pending = engine.pending_;
    const auto& numerical = pending.numerical;
    observation.evidence = pending.evidence;
    observation.candidate_interior_hash = pending.candidate_interior_hash;
    observation.candidate_trace_hash = pending.candidate_trace_hash;
    observation.maximum_conservation_error =
        numerical.maximum_conservation_error;
    observation.maximum_conservation_relative_error =
        numerical.maximum_conservation_relative_error;
    const auto& shape = engine.numerics_.shape_;
    observation.tensor_volume_work = PlanTensorVolumeMmaWork(
        shape.elements, shape.interior_components,
        shape.scalar_interior_dofs, shape.interior_dofs,
        shape.trace_dofs, shape.volume_points, shape.igr_active);
    observation.candidate_ready = numerical.candidate_ready;
    observation.finite = numerical.finite;
    observation.admissible = numerical.admissible;
    observation.recovery = numerical.recovery;
    observation.true_residual = numerical.true_residual;
    observation.conservation = numerical.conservation;
    observation.telemetry = numerical.telemetry;
    observation.success = numerical.success;
    return true;
  }

  static bool CapturePrepared(
      cuda::GenericEulerHdgCudaSession& session,
      CudaAcceptedStepBenchmarkObservation& observation) noexcept
  {
    return session.borrowed_rank1_ && !session.borrowed_rank2_ &&
        CapturePrepared(*session.borrowed_rank1_, observation);
  }

  template <typename Engine>
  static bool CaptureCommitted(
      Engine& engine,
      CudaAcceptedStepCommittedHashes& hashes) noexcept
  {
    hashes = {};
    CheckpointDescriptor checkpoint{};
    if (!engine.initialized_ || engine.pending_.active || !engine.contract_ ||
        !engine.contract_->TryMakeCheckpoint(checkpoint))
      return false;
    hashes.interior = checkpoint.interior_hash;
    hashes.trace = checkpoint.trace_hash;
    hashes.generation = checkpoint.coordinates.generation;
    hashes.valid = checkpoint.successful_commit &&
        checkpoint.schema_version == checkpoint_schema_version;
    return hashes.valid;
  }

  static bool CaptureCommitted(
      cuda::GenericEulerHdgCudaSession& session,
      CudaAcceptedStepCommittedHashes& hashes) noexcept
  {
    return session.borrowed_rank1_ && !session.borrowed_rank2_ &&
        CaptureCommitted(*session.borrowed_rank1_, hashes);
  }

  template <typename Engine>
  static bool CaptureProblemMetadata(
      Engine& engine, CudaAcceptedStepProblemMetadata& metadata) noexcept
  {
    metadata = {};
    if (!engine.initialized_) return false;
    const auto& shape = engine.numerics_.shape_;
    metadata.elements = shape.elements;
    metadata.interior_components = shape.interior_components;
    metadata.trace_components = shape.trace_components;
    metadata.scalar_interior_dofs = shape.scalar_interior_dofs;
    metadata.scalar_face_dofs = shape.scalar_face_dofs;
    metadata.volume_assembly_order = shape.volume_assembly_order;
    metadata.volume_points = shape.volume_points;
    metadata.interior_unknowns = shape.interior_count;
    metadata.true_trace_unknowns = shape.true_trace_count;
    metadata.global_true_trace_unknowns = shape.global_true_trace_count;
    metadata.igr_active = shape.igr_active;
    return true;
  }

  static bool CaptureProblemMetadata(
      cuda::GenericEulerHdgCudaSession& session,
      CudaAcceptedStepProblemMetadata& metadata) noexcept
  {
    return session.borrowed_rank1_ && !session.borrowed_rank2_ &&
        CaptureProblemMetadata(*session.borrowed_rank1_, metadata);
  }
};

} // namespace dg::hdg::runtime::cuda_w8a::detail

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::BoundaryPointView;
using dg::hdg::generic::Capability;
using dg::hdg::generic::CapabilityBit;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerBoundaryPolicyTableView;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::runtime::StateHash;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::CudaMutableHostL2Output;
using dg::hdg::runtime::cuda_w8a::CudaPreparedOutputMetadataV1;
using dg::hdg::runtime::cuda_w8a::CudaTensorCoreEvidence;
using dg::hdg::runtime::cuda_w8a::CudaIluAccountingKind;
using dg::hdg::runtime::cuda_w8a::CudaIluAccountingKindName;
using dg::hdg::runtime::cuda_w8a::CudaIluFactorReusePolicy;
using dg::hdg::runtime::cuda_w8a::CudaIluFactorReusePolicyName;
using dg::hdg::runtime::cuda_w8a::CudaIluTriangularSolveKind;
using dg::hdg::runtime::cuda_w8a::CudaIluTriangularSolveKindName;
using dg::hdg::runtime::cuda_w8a::CudaTracePreconditionerIsIlu;
using dg::hdg::runtime::cuda_w8a::CudaTracePreconditionerName;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyDescriptor;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyDescriptorIsExact;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKind;
using dg::hdg::runtime::cuda_w8a::CudaTracePolicyKindName;
using dg::hdg::runtime::cuda_w8a::
    CudaTracePolicyUsesNativeElementMatrixFreeOperator;
using dg::hdg::runtime::cuda_w8a::ResolveCudaTracePolicy;
using dg::hdg::runtime::cuda_w8a::CudaTransferTotals;
using dg::hdg::runtime::cuda_w8a::CudaTransportPolicy;
using dg::hdg::runtime::cuda_w8a::conserved_components;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaAcceptedStepBenchmarkObservation;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaAcceptedStepCommittedHashes;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaAcceptedStepProblemMetadata;
using dg::hdg::runtime::cuda_w8a::detail::CudaQualificationTestAccess;
using dg::hdg::runtime::cuda_w8a::detail::
    CudaPreparedTensorVolumeEvidenceValuesValid;
using dg::hdg::runtime::cuda_w8a::detail::
    NativeElementSchwarzFieldSplitLayout;
using dg::hdg::runtime::cuda_w8a::detail::
    NativeElementSchwarzIgrFieldSplitLayout;
using dg::hdg::runtime::cuda_w8a::detail::
    native_element_schwarz_equilibration_passes;

constexpr int principal_threads = 128;
constexpr int principal_element_threads = 4;
constexpr int principal_elements_per_block =
    principal_threads / principal_element_threads;
constexpr int default_mesh_n = 104;
constexpr int warmup_steps = 1;
constexpr int measured_steps = 5;
constexpr int total_steps = warmup_steps + measured_steps;
constexpr double default_timestep = 1.0e-5;
constexpr double benchmark_gamma = 1.4;

#if !defined(DG_IGR_BENCHMARK_SOURCE_REVISION) || \
    !defined(DG_IGR_BENCHMARK_SOURCE_TREE) || \
    !defined(DG_IGR_BENCHMARK_SOURCE_STATUS) || \
    !defined(DG_IGR_BENCHMARK_BUILD_DIRECTORY) || \
    !defined(DG_IGR_BENCHMARK_CMAKE_CACHE)
#error "accepted-step benchmark source/build identity is not configured"
#endif
constexpr std::uint32_t cuda_capabilities =
    CapabilityBit(Capability::conforming_quadrilateral_mesh) |
    CapabilityBit(Capability::polynomial_order_one_to_three) |
    CapabilityBit(Capability::cuda) |
    CapabilityBit(Capability::external_initial_state) |
    CapabilityBit(Capability::external_source) |
    CapabilityBit(Capability::numerical_boundary_operator);

static_assert(principal_elements_per_block == 32);
static_assert(default_mesh_n * default_mesh_n >=
                  84 * principal_elements_per_block,
              "the principal assembly grid must saturate all 84 SMs");

constexpr bool ValidDomainPeriods(const int mesh_n,
                                  const int domain_periods) noexcept
{
  return mesh_n >= 2 && domain_periods > 0 && domain_periods <= mesh_n &&
      mesh_n % domain_periods == 0;
}

static_assert(ValidDomainPeriods(default_mesh_n, 1),
              "the default unit domain must remain valid");
static_assert(ValidDomainPeriods(104, 26),
              "the fixed-h saturated IGR workload must remain valid");
static_assert(!ValidDomainPeriods(104, 0) &&
              !ValidDomainPeriods(104, 105) &&
              !ValidDomainPeriods(104, 25),
              "invalid domain replication must be rejected");

struct Options
{
  RegularizationKind regularization = RegularizationKind::none;
  CudaEngineKind engine = CudaEngineKind::ordinary;
  CudaTracePolicyKind trace_policy = CudaTracePolicyKind::engine_default;
  int order = 1;
  int mesh_n = default_mesh_n;
  int domain_periods = 1;
  int profile_measured_step = -1;
  double timestep = default_timestep;
  // Backward-compatible common override plus role-specific policy controls.
  double linear_relative_tolerance = 0.0;
  double newton_correction_linear_relative_tolerance = 0.0;
  double final_coherent_linear_relative_tolerance = 0.0;
  double true_residual_tightening_linear_relative_tolerance = 0.0;
  int igr_volume_assembly_quadrature_order = 0;
  int igr_face_assembly_quadrature_order = 0;
  std::string expected_source_revision{};
  std::string expected_source_tree{};
  std::string expected_source_status{};
  std::string expected_build_directory{};
  std::string expected_cmake_cache{};
  bool export_interior = false;
  bool help = false;
};

struct BorrowedFields
{
  std::vector<double> initial{};
  std::size_t volume_scalar_count = 0;
};

struct StepSample
{
  bool warmup = false;
  int measured_index = -1;
  double seconds = std::numeric_limits<double>::infinity();
  std::uint64_t exported_interior_checksum = 0;
  CudaTransferTotals transfers{};
  CudaAcceptedStepBenchmarkObservation prepared{};
  CudaAcceptedStepCommittedHashes committed{};
  dg::hdg::generic::HdgEulerSessionCoordinates coordinates{};
  dg::hdg::generic::HdgEulerCommittedDiagnostics diagnostics{};
};

struct DeviceMemorySnapshot
{
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  bool available = false;
};

struct UmpireHighWatermarkSnapshot
{
  std::size_t mfem_bytes = 0;
  std::size_t hypre_bytes = 0;
  std::size_t sum_bytes = 0;
  bool mfem_available = false;
  bool hypre_available = false;
  bool sum_available = false;
  bool available = false;
};

bool ParsePositiveInt(const std::string_view text, const int minimum,
                      int& value) noexcept
{
  int parsed = 0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      parsed < minimum || parsed > 46340)
    return false;
  value = parsed;
  return true;
}

bool ParsePositiveDouble(const std::string_view text, double& value) noexcept
{
  double parsed = 0.0;
  const auto result = std::from_chars(
      text.data(), text.data() + text.size(), parsed,
      std::chars_format::general);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      !std::isfinite(parsed) || parsed <= 0.0)
    return false;
  value = parsed;
  return true;
}

bool SelectMode(const std::string_view text, Options& options) noexcept
{
  if (text == "conventional")
  {
    options.regularization = RegularizationKind::none;
    return true;
  }
  if (text == "igr")
  {
    options.regularization = RegularizationKind::igr;
    return true;
  }
  return false;
}

bool SelectEngine(const std::string_view text, Options& options) noexcept
{
  if (text == "ordinary")
  {
    options.engine = CudaEngineKind::ordinary;
    return true;
  }
  if (text == "tensor")
  {
    options.engine = CudaEngineKind::tensor_core;
    return true;
  }
  return false;
}

bool SelectTracePolicy(
    const std::string_view text, Options& options) noexcept
{
  if (text == "engine-default")
  {
    options.trace_policy = CudaTracePolicyKind::engine_default;
    return true;
  }
  if (text == "common-fast")
  {
    options.trace_policy = CudaTracePolicyKind::common_fast;
    return true;
  }
  if (text == "equilibrated-mgr")
  {
    options.trace_policy = CudaTracePolicyKind::equilibrated_mgr;
    return true;
  }
  if (text == "native-element-schwarz-fp64")
  {
    options.trace_policy =
        CudaTracePolicyKind::native_element_schwarz_fp64;
    return true;
  }
  if (text == "native-element-schwarz-matrix-free-fp64")
  {
    options.trace_policy =
        CudaTracePolicyKind::native_element_schwarz_matrix_free_fp64;
    return true;
  }
  if (text ==
      "native-element-schwarz-equilibrated-field-split-fp64")
  {
    options.trace_policy = CudaTracePolicyKind::
        native_element_schwarz_equilibrated_field_split_fp64;
    return true;
  }
  if (text ==
      "native-element-schwarz-matrix-free-equilibrated-field-split-fp64")
  {
    options.trace_policy = CudaTracePolicyKind::
        native_element_schwarz_matrix_free_equilibrated_field_split_fp64;
    return true;
  }
  if (text ==
      "native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64")
  {
    options.trace_policy = CudaTracePolicyKind::
        native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64;
    return true;
  }
  return false;
}

const char* EngineName(const CudaEngineKind engine) noexcept
{
  return engine == CudaEngineKind::tensor_core ? "tensor" : "ordinary";
}

const char* TraceSolverName(const CudaTensorCoreEvidence& evidence) noexcept
{
  if (evidence.native_rank1_trace_solver_configured &&
      !evidence.hypre_mgr_rank1_trace_solver_configured &&
      !evidence.hypre_host_staged_rank2_trace_solver_configured)
    return "native-rank1-fp64";
  if (!evidence.native_rank1_trace_solver_configured &&
      !evidence.hypre_mgr_rank1_trace_solver_configured &&
      evidence.hypre_host_staged_rank2_trace_solver_configured)
    return "hypre-host-staged-rank2";
  if (!evidence.native_rank1_trace_solver_configured &&
      evidence.hypre_mgr_rank1_trace_solver_configured &&
      !evidence.hypre_host_staged_rank2_trace_solver_configured)
    return "hypre-fgmres-mgr-rank1";
  if (!evidence.native_rank1_trace_solver_configured &&
      !evidence.hypre_mgr_rank1_trace_solver_configured &&
      !evidence.hypre_host_staged_rank2_trace_solver_configured)
    return "hypre-parcsr-gmres";
  return "invalid";
}

bool ParseOptions(const int argc, char* argv[], Options& options,
                  std::string& error)
{
  for (int index = 1; index < argc; ++index)
  {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h")
    {
      options.help = true;
      continue;
    }
    if (argument == "--igr")
    {
      options.regularization = RegularizationKind::igr;
      continue;
    }
    if (argument == "--conventional")
    {
      options.regularization = RegularizationKind::none;
      continue;
    }
    if (argument == "--export")
    {
      options.export_interior = true;
      continue;
    }
    if (argument == "--no-export")
    {
      options.export_interior = false;
      continue;
    }
    if (argument == "--mode")
    {
      if (++index >= argc || !SelectMode(argv[index], options))
      {
        error = "--mode requires conventional or igr";
        return false;
      }
      continue;
    }
    if (argument == "--engine")
    {
      if (++index >= argc || !SelectEngine(argv[index], options))
      {
        error = "--engine requires ordinary or tensor";
        return false;
      }
      continue;
    }
    if (argument == "--trace-policy")
    {
      if (++index >= argc || !SelectTracePolicy(argv[index], options))
      {
        error = "--trace-policy requires engine-default, common-fast, "
                "equilibrated-mgr, native-element-schwarz-fp64, or "
                "native-element-schwarz-matrix-free-fp64, or "
                "native-element-schwarz-equilibrated-field-split-fp64, or "
                "native-element-schwarz-matrix-free-equilibrated-field-"
                "split-fp64, or native-element-schwarz-matrix-free-warp-"
                "equilibrated-field-split-fp64";
        return false;
      }
      continue;
    }
    if (argument == "--order")
    {
      if (++index >= argc ||
          !ParsePositiveInt(argv[index], 1, options.order) ||
          options.order > 3)
      {
        error = "--order requires 1, 2, or 3";
        return false;
      }
      continue;
    }
    if (argument == "--mesh-n")
    {
      if (++index >= argc ||
          !ParsePositiveInt(argv[index], 2, options.mesh_n))
      {
        error = "--mesh-n requires an integer in [2,46340]";
        return false;
      }
      continue;
    }
    if (argument == "--domain-periods")
    {
      if (++index >= argc ||
          !ParsePositiveInt(argv[index], 1, options.domain_periods))
      {
        error = "--domain-periods requires an integer in [1,46340]";
        return false;
      }
      continue;
    }
    if (argument == "--profile-measured-step")
    {
      if (++index >= argc ||
          !ParsePositiveInt(argv[index], 0,
                            options.profile_measured_step) ||
          options.profile_measured_step >= measured_steps)
      {
        error = "--profile-measured-step requires an integer in [0,4]";
        return false;
      }
      continue;
    }
    if (argument == "--dt")
    {
      if (++index >= argc ||
          !ParsePositiveDouble(argv[index], options.timestep))
      {
        error = "--dt requires a finite positive FP64 value";
        return false;
      }
      continue;
    }
    if (argument == "--linear-rtol")
    {
      if (++index >= argc ||
          !ParsePositiveDouble(argv[index],
                               options.linear_relative_tolerance))
      {
        error = "--linear-rtol requires a finite positive FP64 value";
        return false;
      }
      continue;
    }
    if (argument == "--newton-linear-rtol")
    {
      if (++index >= argc ||
          !ParsePositiveDouble(
              argv[index],
              options.newton_correction_linear_relative_tolerance))
      {
        error = "--newton-linear-rtol requires a finite positive FP64 value";
        return false;
      }
      continue;
    }
    if (argument == "--final-linear-rtol")
    {
      if (++index >= argc ||
          !ParsePositiveDouble(
              argv[index], options.final_coherent_linear_relative_tolerance))
      {
        error = "--final-linear-rtol requires a finite positive FP64 value";
        return false;
      }
      continue;
    }
    if (argument == "--true-residual-tightening-linear-rtol")
    {
      if (++index >= argc ||
          !ParsePositiveDouble(
              argv[index],
              options.true_residual_tightening_linear_relative_tolerance))
      {
        error = "--true-residual-tightening-linear-rtol requires a finite "
                "positive FP64 value";
        return false;
      }
      continue;
    }
    if (argument == "--igr-volume-quadrature-order")
    {
      if (++index >= argc ||
          !ParsePositiveInt(argv[index], 1,
                            options.igr_volume_assembly_quadrature_order) ||
          options.igr_volume_assembly_quadrature_order > 14)
      {
        error =
            "--igr-volume-quadrature-order requires an integer in [1,14]";
        return false;
      }
      continue;
    }
    if (argument == "--igr-face-quadrature-order")
    {
      if (++index >= argc ||
          !ParsePositiveInt(argv[index], 1,
                            options.igr_face_assembly_quadrature_order) ||
          options.igr_face_assembly_quadrature_order > 14)
      {
        error =
            "--igr-face-quadrature-order requires an integer in [1,14]";
        return false;
      }
      continue;
    }
    const auto parse_expected = [&](std::string& value) -> bool
    {
      if (++index >= argc || std::string_view(argv[index]).empty())
        return false;
      value = argv[index];
      return true;
    };
    if (argument == "--expected-source-revision")
    {
      if (!parse_expected(options.expected_source_revision))
      {
        error = "--expected-source-revision requires a value";
        return false;
      }
      continue;
    }
    if (argument == "--expected-source-tree")
    {
      if (!parse_expected(options.expected_source_tree))
      {
        error = "--expected-source-tree requires a value";
        return false;
      }
      continue;
    }
    if (argument == "--expected-source-status")
    {
      if (!parse_expected(options.expected_source_status))
      {
        error = "--expected-source-status requires a value";
        return false;
      }
      continue;
    }
    if (argument == "--expected-build-directory")
    {
      if (!parse_expected(options.expected_build_directory))
      {
        error = "--expected-build-directory requires a value";
        return false;
      }
      continue;
    }
    if (argument == "--expected-cmake-cache")
    {
      if (!parse_expected(options.expected_cmake_cache))
      {
        error = "--expected-cmake-cache requires a value";
        return false;
      }
      continue;
    }
    error = "unknown argument: " + std::string(argument);
    return false;
  }
  if (!ValidDomainPeriods(options.mesh_n, options.domain_periods))
  {
    error = "--domain-periods must be positive, no greater than --mesh-n, "
            "and divide --mesh-n exactly";
    return false;
  }
  const bool have_igr_volume_order =
      options.igr_volume_assembly_quadrature_order != 0;
  const bool have_igr_face_order =
      options.igr_face_assembly_quadrature_order != 0;
  if (have_igr_volume_order != have_igr_face_order)
  {
    error = "the IGR volume and face quadrature options must be supplied "
            "together";
    return false;
  }
  if (options.regularization != RegularizationKind::igr &&
      have_igr_volume_order)
  {
    error = "the IGR quadrature options require --mode igr";
    return false;
  }
  if ((options.trace_policy ==
           CudaTracePolicyKind::native_element_schwarz_fp64 ||
       CudaTracePolicyUsesNativeElementMatrixFreeOperator(
           options.trace_policy) ||
       options.trace_policy == CudaTracePolicyKind::
           native_element_schwarz_equilibrated_field_split_fp64) &&
      (options.engine != CudaEngineKind::tensor_core ||
       options.regularization != RegularizationKind::igr))
  {
    error = "native element-Schwarz policies require --engine tensor and "
            "--mode igr";
    return false;
  }
  const std::array<bool, 5> expected_supplied{{
      !options.expected_source_revision.empty(),
      !options.expected_source_tree.empty(),
      !options.expected_source_status.empty(),
      !options.expected_build_directory.empty(),
      !options.expected_cmake_cache.empty()}};
  if (std::any_of(expected_supplied.begin(), expected_supplied.end(),
                  [](const bool supplied) { return supplied; }) &&
      !std::all_of(expected_supplied.begin(), expected_supplied.end(),
                   [](const bool supplied) { return supplied; }))
  {
    error = "all expected source/build identity options are required together";
    return false;
  }
  return true;
}

bool ExactExpectedBuildIdentity(const Options& options) noexcept
{
  if (options.expected_source_revision.empty()) return true;
  return options.expected_source_revision ==
          DG_IGR_BENCHMARK_SOURCE_REVISION &&
      options.expected_source_tree == DG_IGR_BENCHMARK_SOURCE_TREE &&
      options.expected_source_status == DG_IGR_BENCHMARK_SOURCE_STATUS &&
      options.expected_build_directory == DG_IGR_BENCHMARK_BUILD_DIRECTORY &&
      options.expected_cmake_cache == DG_IGR_BENCHMARK_CMAKE_CACHE;
}

void PrintUsage(const char* executable)
{
  std::cout << "Usage: " << executable
            << " [--mode conventional|igr] [--engine ordinary|tensor]"
               " [--trace-policy engine-default|common-fast|equilibrated-mgr|native-element-schwarz-fp64|native-element-schwarz-matrix-free-fp64|native-element-schwarz-equilibrated-field-split-fp64|native-element-schwarz-matrix-free-equilibrated-field-split-fp64|native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64]"
               " [--order 1|2|3] [--mesh-n N]"
               " [--domain-periods K] [--dt DT] [--linear-rtol RTOL]"
               " [--newton-linear-rtol RTOL] [--final-linear-rtol RTOL]"
               " [--true-residual-tightening-linear-rtol RTOL]"
               " [--igr-volume-quadrature-order Q]"
               " [--igr-face-quadrature-order Q] [--export|--no-export]"
               " [--profile-measured-step 0|1|2|3|4]\n"
               "       " << executable
            << " [--conventional|--igr] [--engine ordinary|tensor]"
               " [--trace-policy engine-default|common-fast|equilibrated-mgr|native-element-schwarz-fp64|native-element-schwarz-matrix-free-fp64|native-element-schwarz-equilibrated-field-split-fp64|native-element-schwarz-matrix-free-equilibrated-field-split-fp64|native-element-schwarz-matrix-free-warp-equilibrated-field-split-fp64]"
               " [--order 1|2|3] [--mesh-n N]"
               " [--domain-periods K] [--dt DT] [--linear-rtol RTOL]"
               " [--newton-linear-rtol RTOL] [--final-linear-rtol RTOL]"
               " [--true-residual-tightening-linear-rtol RTOL]"
               " [--igr-volume-quadrature-order Q]"
               " [--igr-face-quadrature-order Q] [--export|--no-export]"
               " [--profile-measured-step 0|1|2|3|4]\n";
}

mfem::ParMesh MakePeriodicMesh(const int n, const int domain_periods)
{
  const double domain_length = static_cast<double>(domain_periods);
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      n, n, mfem::Element::QUADRILATERAL, true,
      domain_length, domain_length);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0] = 0.0;
  translations[1] = 0.0;
  translations[0][0] = domain_length;
  translations[1][1] = domain_length;
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
      serial, serial.CreatePeriodicVertexMapping(translations));
  return mfem::ParMesh(MPI_COMM_WORLD, periodic);
}

void SmoothPeriodicState(const mfem::Vector& point, const bool wrap_period,
                         mfem::Vector& state)
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double x = wrap_period ? point[0] - std::floor(point[0]) : point[0];
  const double y = wrap_period ? point[1] - std::floor(point[1]) : point[1];
  const double sx = std::sin(2.0 * pi * x);
  const double sy = std::sin(2.0 * pi * y);
  const double cx = std::cos(2.0 * pi * x);
  const double cy = std::cos(2.0 * pi * y);
  const double density = 1.0 + 0.04 * sx * sy;
  const double velocity_x = 0.18 + 0.02 * cx * sy;
  const double velocity_y = -0.09 + 0.015 * sx * cy;
  const double pressure = 1.0 + 0.03 * cx * cy;
  state.SetSize(conserved_components);
  state[0] = density;
  state[1] = density * velocity_x;
  state[2] = density * velocity_y;
  state[3] = pressure / 0.4 + 0.5 * density *
      (velocity_x * velocity_x + velocity_y * velocity_y);
}

BorrowedFields MakeFields(mfem::ParMesh& mesh, const int domain_periods,
                          const int order)
{
  BorrowedFields fields;
  mfem::L2_FECollection collection(
      order, 2, mfem::BasisType::GaussLegendre,
      mfem::FiniteElement::VALUE);
  mfem::ParFiniteElementSpace space(
      &mesh, &collection, conserved_components, mfem::Ordering::byNODES);
  mfem::ParGridFunction projected(&space);
  const auto evaluator = [wrap_period = domain_periods > 1](
      const mfem::Vector& point, mfem::Vector& state)
  { SmoothPeriodicState(point, wrap_period, state); };
  mfem::VectorFunctionCoefficient coefficient(
      conserved_components, evaluator);
  projected.ProjectCoefficient(coefficient);
  fields.volume_scalar_count = static_cast<std::size_t>(
      space.GetVSize() / conserved_components);
  const double* values = projected.HostRead();
  if (values != nullptr && projected.Size() > 0)
    fields.initial.assign(values, values + projected.Size());
  return fields;
}

bool PeriodicExterior(const void*, const BoundaryPointView&,
                      const double* interior, double* exterior) noexcept
{
  if (interior == nullptr || exterior == nullptr) return false;
  std::copy_n(interior, conserved_components, exterior);
  return true;
}

EulerHdgRequest MakeRequest(const mfem::ParMesh& mesh,
                            const BorrowedFields& fields,
                            const Options& options)
{
  EulerHdgRequest request{};
  request.mesh = &mesh;
  request.communicator = mesh.GetComm();
  request.polynomial_order = options.order;
  request.backend = {BackendKind::cuda, 0, options.engine};
  request.capabilities = cuda_capabilities;
  request.initial_state = {
      fields.initial.data(), fields.initial.size(), conserved_components};
  // Exercise the established projected-trace path.  Projection and all other
  // setup happen once before the warmup and are outside every timed sample.
  request.initial_true_trace = {nullptr, 0, conserved_components};
  request.source = {nullptr, 0, conserved_components};
  request.boundary.context = nullptr;
  request.boundary.exterior_state = PeriodicExterior;
  request.boundary.policies = EulerBoundaryPolicyTableView{
      nullptr, 0, 2, benchmark_gamma};
  request.numerical_policy =
      dg::hdg::generic::DefaultEulerHdgNumericalPolicy(options.order);
  if (options.linear_relative_tolerance > 0.0)
  {
    request.numerical_policy.newton_correction_linear_relative_tolerance =
        options.linear_relative_tolerance;
    request.numerical_policy.final_coherent_linear_relative_tolerance =
        options.linear_relative_tolerance;
    request.numerical_policy.
        true_residual_tightening_linear_relative_tolerance =
            options.linear_relative_tolerance;
  }
  if (options.newton_correction_linear_relative_tolerance > 0.0)
    request.numerical_policy.newton_correction_linear_relative_tolerance =
        options.newton_correction_linear_relative_tolerance;
  if (options.final_coherent_linear_relative_tolerance > 0.0)
    request.numerical_policy.final_coherent_linear_relative_tolerance =
        options.final_coherent_linear_relative_tolerance;
  if (options.true_residual_tightening_linear_relative_tolerance > 0.0)
    request.numerical_policy.
        true_residual_tightening_linear_relative_tolerance =
            options.true_residual_tightening_linear_relative_tolerance;
  if (options.igr_volume_assembly_quadrature_order > 0)
    request.numerical_policy.igr_assembly_quadrature_orders = {
        options.igr_volume_assembly_quadrature_order,
        options.igr_face_assembly_quadrature_order};
  if (options.regularization == RegularizationKind::igr)
    request.numerical_policy.stabilization = {
        RegularizationKind::igr, 0.2, 0.8};
  return request;
}

std::string HashHex(const StateHash& hash)
{
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.resize(2 * hash.bytes.size());
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    const std::uint8_t value = hash.bytes[index];
    result[2 * index] = digits[value >> 4U];
    result[2 * index + 1] = digits[value & 0x0fU];
  }
  return result;
}

std::uint64_t StableFp64Checksum(const std::vector<double>& values) noexcept
{
  std::uint64_t checksum = 1469598103934665603ULL;
  for (const double value : values)
  {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned int shift = 0; shift < 64; shift += 8)
    {
      checksum ^= (bits >> shift) & 0xffU;
      checksum *= 1099511628211ULL;
    }
  }
  return checksum;
}

static constexpr bool IluDerivedBoundsAreConsistent(
    const std::uint64_t callbacks_lower_bound,
    const std::uint64_t callbacks_upper_bound,
    const std::uint64_t krylov_restart_dimension,
    const int configured_iterations,
    const std::uint64_t inner_iterations_lower_bound,
    const std::uint64_t inner_iterations_upper_bound,
    const std::uint64_t factor_spmv_updates_per_application,
    const std::uint64_t factor_spmv_updates_lower_bound,
    const std::uint64_t factor_spmv_updates_upper_bound,
    const int observed_linear_iterations,
    const int observed_linear_solve_attempts) noexcept
{
  if (krylov_restart_dimension == 0 || configured_iterations <= 0 ||
      observed_linear_iterations < 0 ||
      observed_linear_solve_attempts < 0 ||
      callbacks_lower_bound > callbacks_upper_bound)
    return false;
  const std::uint64_t iterations =
      static_cast<std::uint64_t>(observed_linear_iterations);
  if (iterations == 0)
  {
    if (callbacks_lower_bound != 0 || callbacks_upper_bound != 0 ||
        inner_iterations_lower_bound != 0 ||
        inner_iterations_upper_bound != 0 ||
        factor_spmv_updates_lower_bound != 0 ||
        factor_spmv_updates_upper_bound != 0)
      return false;
  }
  else
  {
    if (observed_linear_solve_attempts == 0 ||
        iterations > std::numeric_limits<std::uint64_t>::max()/2 ||
        callbacks_lower_bound <= iterations ||
        callbacks_upper_bound != 2*iterations)
      return false;
    // The producer derives I_s + ceil(I_s/K) for every successful trace
    // solve and then accumulates those bounds.  The ceiling is nonadditive,
    // so validate that the aggregate cycle count is feasible for some
    // partition of I across at most `observed_linear_solve_attempts` positive
    // solver invocations;
    // retaining that partition would be per-solve history.
    const std::uint64_t restart_cycles = callbacks_lower_bound-iterations;
    const std::uint64_t solve_attempts =
        static_cast<std::uint64_t>(observed_linear_solve_attempts);
    const std::uint64_t positive_solves =
        std::min(solve_attempts, std::min(iterations, restart_cycles));
    if (positive_solves == 0 ||
        restart_cycles >
            std::numeric_limits<std::uint64_t>::max()/
                krylov_restart_dimension)
      return false;
    const std::uint64_t maximum_partition_iterations =
        restart_cycles*krylov_restart_dimension;
    const std::uint64_t minimum_partition_iterations =
        (restart_cycles-positive_solves)*krylov_restart_dimension+
        positive_solves;
    if (iterations < minimum_partition_iterations ||
        iterations > maximum_partition_iterations)
      return false;
  }
  const std::uint64_t multiplier =
      static_cast<std::uint64_t>(configured_iterations);
  return callbacks_lower_bound <=
          std::numeric_limits<std::uint64_t>::max()/multiplier &&
      callbacks_upper_bound <=
          std::numeric_limits<std::uint64_t>::max()/multiplier &&
      inner_iterations_lower_bound == callbacks_lower_bound*multiplier &&
      inner_iterations_upper_bound == callbacks_upper_bound*multiplier &&
      (factor_spmv_updates_per_application == 0 ||
       (inner_iterations_lower_bound <=
            std::numeric_limits<std::uint64_t>::max()/
                factor_spmv_updates_per_application &&
        inner_iterations_upper_bound <=
            std::numeric_limits<std::uint64_t>::max()/
                factor_spmv_updates_per_application)) &&
      factor_spmv_updates_lower_bound ==
          inner_iterations_lower_bound*factor_spmv_updates_per_application &&
      factor_spmv_updates_upper_bound ==
          inner_iterations_upper_bound*factor_spmv_updates_per_application;
}

static_assert(IluDerivedBoundsAreConsistent(
    0, 0, 20, 1, 0, 0, 0, 0, 0, 0, 1));
static_assert(IluDerivedBoundsAreConsistent(
    21, 40, 20, 1, 21, 40, 0, 0, 0, 20, 1));
static_assert(IluDerivedBoundsAreConsistent(
    21, 40, 20, 1, 21, 40, 8, 168, 320, 20, 1));
static_assert(IluDerivedBoundsAreConsistent(
    23, 42, 20, 2, 46, 84, 8, 368, 672, 21, 1));
static_assert(IluDerivedBoundsAreConsistent(
    22, 40, 20, 1, 22, 40, 0, 0, 0, 20, 2));
static_assert(!IluDerivedBoundsAreConsistent(
    19, 40, 20, 1, 19, 40, 0, 0, 0, 20, 1));
static_assert(!IluDerivedBoundsAreConsistent(
    20, 40, 20, 1, 20, 40, 0, 0, 0, 20, 1));
static_assert(!IluDerivedBoundsAreConsistent(
    22, 40, 20, 1, 22, 40, 0, 0, 0, 20, 1));
static_assert(!IluDerivedBoundsAreConsistent(
    21, 39, 20, 1, 21, 39, 0, 0, 0, 20, 1));
static_assert(!IluDerivedBoundsAreConsistent(
    21, 41, 20, 1, 21, 41, 0, 0, 0, 20, 1));
static_assert(!IluDerivedBoundsAreConsistent(
    21, 40, 20, 1, 21, 40, 8, 167, 320, 20, 1));
static_assert(!IluDerivedBoundsAreConsistent(
    std::numeric_limits<std::uint64_t>::max()/2+1,
    std::numeric_limits<std::uint64_t>::max(), 20, 2,
    0, 0, 0, 0, 0, 1, 1));
static_assert(!IluDerivedBoundsAreConsistent(
    1, 2, 20, 0, 0, 0, 0, 0, 0, 1, 1));
static_assert(!IluDerivedBoundsAreConsistent(
    1, 2, 0, 1, 1, 2, 0, 0, 0, 1, 1));

bool Healthy(const CudaAcceptedStepBenchmarkObservation& observation,
             const CudaEngineKind engine, const bool igr_active,
             const CudaTracePolicyKind trace_policy,
             const std::uint64_t krylov_restart_dimension) noexcept
{
  const auto& evidence = observation.evidence;
  const auto& tensor = evidence.tensor_core;
  const CudaTracePolicyDescriptor expected_policy =
      ResolveCudaTracePolicy(engine, igr_active, trace_policy);
  const auto expected_preconditioner =
      expected_policy.trace_preconditioner;
  const bool ilu_policy = CudaTracePreconditionerIsIlu(
      expected_preconditioner);
  const auto expected_factor_reuse_policy =
      expected_policy.ilu_factor_reuse_policy;
  const int expected_ilu_iterations =
      expected_policy.ilu_application_iterations;
  const int expected_ilu_fill = expected_policy.ilu_level_of_fill;
  const int expected_ilu_reordering =
      expected_policy.ilu_local_reordering;
  const auto expected_triangular_solve_kind =
      expected_policy.ilu_triangular_solve_kind;
  const int expected_lower_jacobi_iterations =
      expected_policy.ilu_lower_jacobi_iterations;
  const int expected_upper_jacobi_iterations =
      expected_policy.ilu_upper_jacobi_iterations;
  const std::uint64_t expected_factor_spmv_updates_per_application =
      expected_triangular_solve_kind ==
          CudaIluTriangularSolveKind::iterative_jacobi
      ? 8 : 0;
  const std::uint64_t trace_solves = evidence.trace_solves >= 0
      ? static_cast<std::uint64_t>(evidence.trace_solves) : 0;
  const std::uint64_t linear_solve_attempts =
      evidence.linear_solve_attempts >= 0
      ? static_cast<std::uint64_t>(evidence.linear_solve_attempts) : 0;
  const bool refresh_each_solve = expected_factor_reuse_policy ==
      CudaIluFactorReusePolicy::refresh_each_trace_solve;
  const bool first_solve_per_attempt = expected_factor_reuse_policy ==
      CudaIluFactorReusePolicy::first_trace_solve_per_attempt;
  const std::uint64_t expected_setups = refresh_each_solve
      ? linear_solve_attempts
      : first_solve_per_attempt && linear_solve_attempts > 0 ? 1 : 0;
  const std::uint64_t expected_reuses = first_solve_per_attempt &&
      linear_solve_attempts > 0 ? linear_solve_attempts - 1 : 0;
  const bool ilu_work_consistent = ilu_policy
      ? evidence.trace_solves >= 0 && evidence.linear_iterations >= 0 &&
            tensor.ilu_type == 0 &&
            tensor.ilu_level_of_fill == expected_ilu_fill &&
            tensor.ilu_local_reordering == expected_ilu_reordering &&
            tensor.ilu_application_iterations == expected_ilu_iterations &&
            tensor.ilu_triangular_solve_kind ==
                expected_triangular_solve_kind &&
            tensor.ilu_lower_jacobi_iterations ==
                expected_lower_jacobi_iterations &&
            tensor.ilu_upper_jacobi_iterations ==
                expected_upper_jacobi_iterations &&
            ((expected_triangular_solve_kind ==
                  CudaIluTriangularSolveKind::direct &&
              expected_lower_jacobi_iterations == 0 &&
              expected_upper_jacobi_iterations == 0) ||
             (expected_triangular_solve_kind ==
                  CudaIluTriangularSolveKind::iterative_jacobi &&
              expected_lower_jacobi_iterations == 5 &&
              expected_upper_jacobi_iterations == 5)) &&
            tensor.ilu_accounting_kind ==
                expected_policy.ilu_accounting_kind &&
            tensor.ilu_factor_reuse_policy ==
                expected_factor_reuse_policy &&
            tensor.ilu_accounting_complete &&
            tensor.ilu_factor_setups == expected_setups &&
            tensor.ilu_factor_reuses == expected_reuses &&
            tensor.ilu_factor_setups + tensor.ilu_factor_reuses ==
                linear_solve_attempts &&
            IluDerivedBoundsAreConsistent(
                tensor.ilu_preconditioner_callbacks_lower_bound,
                tensor.ilu_preconditioner_callbacks_upper_bound,
                krylov_restart_dimension,
                expected_ilu_iterations,
                tensor.ilu_inner_iterations_lower_bound,
                tensor.ilu_inner_iterations_upper_bound,
                expected_factor_spmv_updates_per_application,
                tensor.ilu_jacobi_factor_spmv_updates_lower_bound,
                tensor.ilu_jacobi_factor_spmv_updates_upper_bound,
                evidence.linear_iterations,
                evidence.linear_solve_attempts) &&
            tensor.ilu_factor_age_generations ==
                (trace_solves > 0 ? trace_solves - 1 : 0)
      : tensor.ilu_type == -1 && tensor.ilu_level_of_fill == -1 &&
            tensor.ilu_local_reordering == -1 &&
            tensor.ilu_application_iterations == 0 &&
            expected_triangular_solve_kind ==
                CudaIluTriangularSolveKind::not_applicable &&
            tensor.ilu_triangular_solve_kind ==
                expected_triangular_solve_kind &&
            tensor.ilu_lower_jacobi_iterations == 0 &&
            tensor.ilu_upper_jacobi_iterations == 0 &&
            tensor.ilu_accounting_kind ==
                expected_policy.ilu_accounting_kind &&
            tensor.ilu_factor_reuse_policy ==
                CudaIluFactorReusePolicy::not_applicable &&
            tensor.ilu_factor_setups == 0 &&
            tensor.ilu_factor_reuses == 0 &&
            tensor.ilu_preconditioner_callbacks_lower_bound == 0 &&
            tensor.ilu_preconditioner_callbacks_upper_bound == 0 &&
            tensor.ilu_inner_iterations_lower_bound == 0 &&
            tensor.ilu_inner_iterations_upper_bound == 0 &&
            tensor.ilu_jacobi_factor_spmv_updates_lower_bound == 0 &&
            tensor.ilu_jacobi_factor_spmv_updates_upper_bound == 0 &&
            tensor.ilu_factor_age_generations == 0 &&
            tensor.ilu_accounting_complete;
  const bool expected_mgr_solver =
      trace_policy == CudaTracePolicyKind::equilibrated_mgr;
  const bool expected_matrix_free =
      CudaTracePolicyUsesNativeElementMatrixFreeOperator(trace_policy);
  const bool expected_element_schwarz = trace_policy ==
          CudaTracePolicyKind::native_element_schwarz_fp64 ||
      expected_matrix_free ||
      trace_policy == CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64;
  const bool expected_equilibrated_field_split = trace_policy ==
      CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64 ||
      trace_policy == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64 ||
      trace_policy == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64;
  const bool expected_native_solver =
      engine == CudaEngineKind::tensor_core && !expected_mgr_solver;
  const bool mgr_work_consistent = expected_mgr_solver
      ? tensor.equilibration_builds == trace_solves &&
            tensor.equilibration_passes == 2 * trace_solves &&
            tensor.mgr_setup_calls == trace_solves &&
            tensor.mgr_f_ilu_setups == tensor.mgr_setup_calls &&
            tensor.mgr_f_ilu_applications ==
                tensor.mgr_preconditioner_applications &&
            (evidence.linear_iterations == 0 ||
             tensor.mgr_preconditioner_applications > 0) &&
            std::isfinite(tensor.minimum_left_scale) &&
            std::isfinite(tensor.maximum_left_scale) &&
            std::isfinite(tensor.minimum_right_scale) &&
            std::isfinite(tensor.maximum_right_scale) &&
            tensor.minimum_left_scale > 0.0 &&
            tensor.minimum_left_scale <= tensor.maximum_left_scale &&
            tensor.minimum_right_scale > 0.0 &&
            tensor.minimum_right_scale <= tensor.maximum_right_scale
      : expected_equilibrated_field_split
          ? tensor.equilibration_builds == trace_solves &&
                tensor.equilibration_passes ==
                    native_element_schwarz_equilibration_passes *
                        trace_solves &&
                tensor.mgr_setup_calls == 0 &&
                tensor.mgr_preconditioner_applications == 0 &&
                tensor.mgr_f_ilu_setups == 0 &&
                tensor.mgr_f_ilu_applications == 0
          : tensor.equilibration_builds == 0 &&
                tensor.equilibration_passes == 0 &&
                tensor.mgr_setup_calls == 0 &&
                tensor.mgr_preconditioner_applications == 0 &&
                tensor.mgr_f_ilu_setups == 0 &&
                tensor.mgr_f_ilu_applications == 0;
  const bool element_schwarz_work_consistent = expected_element_schwarz
      ? tensor.native_element_schwarz_configured &&
            tensor.element_schwarz_patch_setups == trace_solves &&
            (evidence.linear_iterations == 0 ||
             tensor.element_schwarz_preconditioner_applications > 0) &&
            (tensor.element_schwarz_patch_rows == 40 ||
             tensor.element_schwarz_patch_rows == 60 ||
             tensor.element_schwarz_patch_rows == 80) &&
            tensor.element_schwarz_minimum_multiplicity >= 1 &&
            tensor.element_schwarz_maximum_multiplicity >=
                tensor.element_schwarz_minimum_multiplicity &&
            tensor.element_schwarz_maximum_multiplicity <= 2
      : !tensor.native_element_schwarz_configured &&
            tensor.element_schwarz_patch_setups == 0 &&
            tensor.element_schwarz_preconditioner_applications == 0 &&
            tensor.element_schwarz_patch_rows == 0 &&
            tensor.element_schwarz_minimum_multiplicity == 0 &&
            tensor.element_schwarz_maximum_multiplicity == 0;
  const bool matrix_free_work_consistent = expected_matrix_free
      ? tensor.native_matrix_free_trace_operator_configured &&
            tensor.native_matrix_free_trace_operator_applications >=
                static_cast<std::uint64_t>(evidence.linear_iterations) &&
            ((evidence.linear_iterations == 0) ==
             (tensor.native_matrix_free_trace_operator_applications == 0))
      : !tensor.native_matrix_free_trace_operator_configured &&
            tensor.native_matrix_free_trace_operator_applications == 0;
  const bool trace_solver_consistent =
      tensor.trace_solver_accounting_complete &&
      matrix_free_work_consistent &&
      tensor.native_rank1_trace_solver_configured ==
          expected_native_solver &&
      tensor.hypre_mgr_rank1_trace_solver_configured ==
          expected_mgr_solver &&
      !tensor.hypre_host_staged_rank2_trace_solver_configured &&
      tensor.native_rank1_trace_solve_calls ==
          (expected_native_solver ? linear_solve_attempts : 0) &&
      tensor.hypre_trace_solve_calls ==
          (expected_native_solver ? 0 : linear_solve_attempts);
  const bool expected_engine =
      CudaTracePolicyDescriptorIsExact(expected_policy) &&
      tensor.selected_engine == engine &&
      tensor.trace_policy == trace_policy &&
      tensor.trace_preconditioner_configured &&
      tensor.trace_preconditioner == expected_preconditioner &&
      ilu_work_consistent && mgr_work_consistent &&
      element_schwarz_work_consistent &&
      trace_solver_consistent &&
      observation.tensor_volume_work.valid &&
      CudaPreparedTensorVolumeEvidenceValuesValid(
          engine, tensor.volume_assembly_launches,
          tensor.volume_mma_tiles,
          engine == CudaEngineKind::tensor_core
              ? observation.tensor_volume_work.mma_tiles : 0) &&
      tensor.fallback_count == 0 && tensor.fallback_accounting_complete &&
      evidence.no_fallback &&
      (engine == CudaEngineKind::ordinary
           ? tensor.condensation_launches == 0 && tensor.mma_tiles == 0
           : tensor.hardware_supported && tensor.condensation_launches > 0 &&
                 tensor.mma_tiles > 0);
  return observation.candidate_ready && observation.finite &&
      observation.admissible && observation.recovery &&
      observation.true_residual && observation.conservation &&
      observation.telemetry && observation.success &&
      evidence.products.CoupledAndOrdered() &&
      std::isfinite(evidence.initial_residual) &&
      std::isfinite(evidence.final_residual) &&
      std::isfinite(evidence.true_relative_residual) &&
      std::isfinite(observation.maximum_conservation_error) &&
      std::isfinite(observation.maximum_conservation_relative_error) &&
      evidence.nonlinear_updates >= 0 && evidence.trace_solves > 0 &&
      evidence.linear_iterations >= 0 &&
      evidence.linear_solve_attempts >= evidence.trace_solves &&
      evidence.newton_correction_linear_iterations >= 0 &&
      evidence.final_coherent_linear_iterations >= 0 &&
      evidence.true_residual_tightenings >= 0 &&
      evidence.linear_iterations ==
          evidence.newton_correction_linear_iterations +
              evidence.final_coherent_linear_iterations &&
      evidence.linear_solve_attempts ==
          evidence.trace_solves + evidence.true_residual_tightenings &&
      evidence.line_search_trials >= 0 &&
      evidence.rejected_line_search_trials >= 0 && expected_engine;
}

bool RunStep(GenericEulerHdgCudaSession& session,
             const BorrowedFields& fields,
             std::vector<double>& exported_interior,
             const double timestep,
             const bool export_interior,
             const CudaEngineKind engine,
             const bool igr_active,
             const CudaTracePolicyKind trace_policy,
             const std::uint64_t krylov_restart_dimension,
             const bool warmup, const int measured_index,
             StepSample& sample, std::string& error)
{
  using clock = std::chrono::steady_clock;
  sample = {};
  sample.warmup = warmup;
  sample.measured_index = measured_index;
  GenericEulerCudaPreparedStep authority{};
  CudaPreparedOutputMetadataV1 output{};

  const auto started = clock::now();
  const auto prepared = session.Prepare(timestep, authority, output);
  const bool observed = prepared.status == HdgEulerSessionStatus::prepared &&
      CudaQualificationTestAccess::CapturePrepared(session, sample.prepared);
  const bool exported = observed && (!export_interior ||
      session.ExportPreparedInterior(
          authority, output, CudaMutableHostL2Output{
              exported_interior.data(), fields.volume_scalar_count,
              conserved_components}));
  const bool acknowledged = exported &&
      session.AcknowledgePreparedOutput(authority, output);
  const auto committed = acknowledged
      ? session.Commit(std::move(authority), output)
      : GenericEulerHdgCudaSession::result_type{};
  const auto finished = clock::now();
  // Prepare begins a new transactional attempt and resets the attempt-local
  // transport ledger.  The ledger after Commit is therefore already the
  // exact accepted-step total; subtracting the previous attempt would make
  // this evidence non-monotone and incorrectly mark it incomplete.
  sample.transfers = CudaTransportPolicy::Accounting();

  // Keep the prepared observation inside the wall-clock interval.  Prepare
  // may deliberately return with same-stream metadata staging still in
  // flight, so subtracting host observation time could also subtract real
  // device work that overlaps it.
  sample.seconds =
      std::chrono::duration<double>(finished - started).count();
  if (!observed || !exported || !acknowledged ||
      committed.status != HdgEulerSessionStatus::committed ||
      !output.successful_commit)
  {
    error = "accepted-step transaction failed: prepare_status=" +
        std::to_string(static_cast<unsigned int>(prepared.status)) +
        ", commit_status=" +
        std::to_string(static_cast<unsigned int>(committed.status));
    return false;
  }
  // The host checksum and committed-state observation below remain outside
  // the timed transaction region.
  sample.exported_interior_checksum = export_interior
      ? StableFp64Checksum(exported_interior) : 0;
  if (!CudaQualificationTestAccess::CaptureCommitted(
          session, sample.committed))
  {
    error = "committed hash capture failed";
    return false;
  }
  sample.coordinates = session.coordinates();
  sample.diagnostics = session.diagnostics();
  if (!Healthy(sample.prepared, engine, igr_active, trace_policy,
               krylov_restart_dimension))
  {
    error = "accepted-step prepared numerical gates failed";
    return false;
  }
  if (sample.prepared.candidate_interior_hash != sample.committed.interior ||
      sample.prepared.candidate_trace_hash != sample.committed.trace)
  {
    error = "prepared and committed hashes differ: interior=" +
        HashHex(sample.prepared.candidate_interior_hash) + "/" +
        HashHex(sample.committed.interior) + ", trace=" +
        HashHex(sample.prepared.candidate_trace_hash) + "/" +
        HashHex(sample.committed.trace);
    return false;
  }
  if (sample.coordinates.generation != sample.committed.generation)
  {
    error = "committed generation differs from coordinates";
    return false;
  }
  if (!dg::hdg::generic::IsValidHdgEulerCommittedDiagnostics(
          sample.diagnostics, session.capabilities()))
  {
    error = "committed diagnostics are invalid";
    return false;
  }
  return true;
}

double Median(std::array<double, measured_steps> values) noexcept
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double PopulationStandardDeviation(
    const std::array<double, measured_steps>& values) noexcept
{
  double mean = 0.0;
  for (const double value : values) mean += value;
  mean /= static_cast<double>(values.size());
  double squared = 0.0;
  for (const double value : values)
  {
    const double difference = value - mean;
    squared += difference * difference;
  }
  return std::sqrt(squared / static_cast<double>(values.size()));
}

void PrintJsonNumber(const double value)
{
  if (std::isfinite(value)) std::cout << value;
  else std::cout << "null";
}

std::string CudaUuidString(const cudaUUID_t& uuid)
{
  static constexpr int group_ends[] = {4, 6, 8, 10};
  std::ostringstream stream;
  stream << "GPU-" << std::hex << std::setfill('0');
  for (int index = 0; index < 16; ++index)
  {
    if (std::find(std::begin(group_ends), std::end(group_ends), index) !=
        std::end(group_ends))
      stream << '-';
    stream << std::setw(2)
           << static_cast<unsigned int>(
                  static_cast<unsigned char>(uuid.bytes[index]));
  }
  return stream.str();
}

DeviceMemorySnapshot CaptureDeviceMemory() noexcept
{
  DeviceMemorySnapshot snapshot;
  snapshot.available = cudaMemGetInfo(
      &snapshot.free_bytes, &snapshot.total_bytes) == cudaSuccess;
  if (!snapshot.available) snapshot = {};
  return snapshot;
}

UmpireHighWatermarkSnapshot CaptureUmpireHighWatermarks() noexcept
{
  UmpireHighWatermarkSnapshot snapshot{};
#if defined(MFEM_USE_UMPIRE) && defined(HYPRE_USING_UMPIRE)
  try
  {
    auto& manager = umpire::ResourceManager::getInstance();
    const char* const mfem_name =
        mfem::MemoryManager::GetUmpireDeviceAllocatorName();
    const char* const hypre_name =
        hypre_HandleUmpireDevicePoolName(hypre_handle());
    snapshot.mfem_available = mfem_name != nullptr && mfem_name[0] != '\0' &&
        manager.isAllocator(mfem_name);
    snapshot.hypre_available =
        hypre_name != nullptr && hypre_name[0] != '\0' &&
        manager.isAllocator(hypre_name);
    if (snapshot.mfem_available)
      snapshot.mfem_bytes =
          manager.getAllocator(mfem_name).getHighWatermark();
    if (snapshot.hypre_available)
      snapshot.hypre_bytes =
          manager.getAllocator(hypre_name).getHighWatermark();
    snapshot.sum_available = snapshot.mfem_available &&
        snapshot.hypre_available &&
        snapshot.mfem_bytes <= std::numeric_limits<std::size_t>::max() -
            snapshot.hypre_bytes;
    if (snapshot.sum_available)
      snapshot.sum_bytes = snapshot.mfem_bytes + snapshot.hypre_bytes;
    snapshot.available = snapshot.sum_available;
  }
  catch (...) { snapshot = {}; }
#endif
  return snapshot;
}

void PrintMemorySnapshot(const DeviceMemorySnapshot& snapshot)
{
  if (!snapshot.available)
  {
    std::cout << "null";
    return;
  }
  std::cout << "{\"free_bytes\":" << snapshot.free_bytes
            << ",\"total_bytes\":" << snapshot.total_bytes
            << ",\"used_bytes\":"
            << snapshot.total_bytes - snapshot.free_bytes << '}';
}

void PrintUmpireHighWatermarks(
    const UmpireHighWatermarkSnapshot& snapshot)
{
  std::cout << std::boolalpha
            << "{\"available\":" << snapshot.available
            << ",\"mfem_available\":" << snapshot.mfem_available
            << ",\"hypre_available\":" << snapshot.hypre_available
            << ",\"sum_available\":" << snapshot.sum_available
            << std::noboolalpha << ",\"mfem_bytes\":";
  if (snapshot.mfem_available) std::cout << snapshot.mfem_bytes;
  else std::cout << "null";
  std::cout << ",\"hypre_bytes\":";
  if (snapshot.hypre_available) std::cout << snapshot.hypre_bytes;
  else std::cout << "null";
  std::cout << ",\"sum_bytes\":";
  if (snapshot.sum_available) std::cout << snapshot.sum_bytes;
  else std::cout << "null";
  std::cout << '}';
}

void PrintStep(const StepSample& sample)
{
  const auto& evidence = sample.prepared.evidence;
  const auto& precision_work = evidence.precision_work;
  std::cout << "{\"phase\":\""
            << (sample.warmup ? "warmup" : "measured") << "\","
            << "\"measured_index\":" << sample.measured_index << ','
            << "\"duration_seconds\":";
  PrintJsonNumber(sample.seconds);
  std::cout << ",\"step\":" << sample.coordinates.step
            << ",\"generation\":" << sample.coordinates.generation
            << ",\"time\":";
  PrintJsonNumber(sample.coordinates.time);
  std::cout << ",\"committed_interior_hash\":\""
            << HashHex(sample.committed.interior) << "\","
            << "\"committed_trace_hash\":\""
            << HashHex(sample.committed.trace) << "\","
            << "\"exported_interior_checksum_fnv1a64\":\""
            << std::hex << std::setw(16) << std::setfill('0')
            << sample.exported_interior_checksum << std::dec
            << std::setfill(' ') << "\","
            << "\"initial_residual\":";
  PrintJsonNumber(evidence.initial_residual);
  std::cout << ",\"final_residual\":";
  PrintJsonNumber(evidence.final_residual);
  std::cout << ",\"true_relative_residual\":";
  PrintJsonNumber(evidence.true_relative_residual);
  std::cout << ",\"maximum_conservation_error\":";
  PrintJsonNumber(sample.prepared.maximum_conservation_error);
  std::cout << ",\"maximum_conservation_relative_error\":";
  PrintJsonNumber(sample.prepared.maximum_conservation_relative_error);
  std::cout << ",\"maximum_wave_speed\":";
  PrintJsonNumber(sample.diagnostics.maximum_wave_speed);
  std::cout << ",\"backend_stable_dt\":";
  PrintJsonNumber(sample.diagnostics.backend_stable_dt);
  std::cout << ",\"attempt_sequence\":" << evidence.attempt.sequence
            << ",\"retry_index\":"
            << static_cast<unsigned int>(evidence.attempt.retry_index)
            << ",\"device_ticket_count\":"
            << evidence.device_ticket_count
            << ",\"nonlinear_updates\":" << evidence.nonlinear_updates
            << ",\"trace_solves\":" << evidence.trace_solves
            << ",\"linear_iterations\":" << evidence.linear_iterations
            << ",\"linear_solve_attempts\":"
            << evidence.linear_solve_attempts
            << ",\"newton_correction_linear_iterations\":"
            << evidence.newton_correction_linear_iterations
            << ",\"final_coherent_linear_iterations\":"
            << evidence.final_coherent_linear_iterations
            << ",\"true_residual_tightenings\":"
            << evidence.true_residual_tightenings
            << ",\"line_search_trials\":" << evidence.line_search_trials
            << ",\"rejected_line_search_trials\":"
            << evidence.rejected_line_search_trials
            << ",\"iterative_refinement_steps\":"
            << precision_work.iterative_refinement_steps
            << ",\"residual_replacements\":"
            << precision_work.residual_replacements
            << ",\"precision_corrections\":"
            << precision_work.precision_corrections
            << ",\"all_fp64_policy\":" << std::boolalpha
            << precision_work.all_fp64_policy
            << ",\"correction_accounting_complete\":"
            << precision_work.accounting_complete << std::noboolalpha
            << ",\"product_generations\":{"
            << "\"residual\":" << evidence.products.residual_generation
            << ",\"matrix\":" << evidence.products.matrix_generation
            << ",\"factor\":" << evidence.products.factor_generation
            << ",\"solution\":" << evidence.products.solution_generation
            << ",\"issued_mask\":"
            << static_cast<unsigned int>(evidence.products.issued_mask) << '}'
            << ",\"tensor_core\":{"
            << "\"selected_engine\":\""
            << EngineName(evidence.tensor_core.selected_engine) << "\","
            << "\"trace_policy\":\""
            << CudaTracePolicyKindName(evidence.tensor_core.trace_policy)
            << "\","
            << "\"trace_preconditioner\":\""
            << CudaTracePreconditionerName(
                   evidence.tensor_core.trace_preconditioner) << "\","
            << "\"trace_solver\":\""
            << TraceSolverName(evidence.tensor_core) << "\","
            << "\"ilu_type\":" << evidence.tensor_core.ilu_type << ','
            << "\"ilu_level_of_fill\":"
            << evidence.tensor_core.ilu_level_of_fill << ','
            << "\"ilu_local_reordering\":"
            << evidence.tensor_core.ilu_local_reordering << ','
            << "\"ilu_application_iterations\":"
            << evidence.tensor_core.ilu_application_iterations << ','
            << "\"ilu_triangular_solve_kind\":\""
            << CudaIluTriangularSolveKindName(
                   evidence.tensor_core.ilu_triangular_solve_kind) << "\","
            << "\"ilu_lower_jacobi_iterations\":"
            << evidence.tensor_core.ilu_lower_jacobi_iterations << ','
            << "\"ilu_upper_jacobi_iterations\":"
            << evidence.tensor_core.ilu_upper_jacobi_iterations << ','
            << "\"ilu_accounting_kind\":\""
            << CudaIluAccountingKindName(
                   evidence.tensor_core.ilu_accounting_kind) << "\","
            << "\"ilu_factor_reuse_policy\":\""
            << CudaIluFactorReusePolicyName(
                   evidence.tensor_core.ilu_factor_reuse_policy) << "\","
            << "\"ilu_factor_setups\":"
            << evidence.tensor_core.ilu_factor_setups << ','
            << "\"ilu_factor_reuses\":"
            << evidence.tensor_core.ilu_factor_reuses << ','
            << "\"ilu_preconditioner_callbacks_lower_bound\":"
            << evidence.tensor_core.
                   ilu_preconditioner_callbacks_lower_bound << ','
            << "\"ilu_preconditioner_callbacks_upper_bound\":"
            << evidence.tensor_core.
                   ilu_preconditioner_callbacks_upper_bound << ','
            << "\"ilu_inner_iterations_lower_bound\":"
            << evidence.tensor_core.ilu_inner_iterations_lower_bound << ','
            << "\"ilu_inner_iterations_upper_bound\":"
            << evidence.tensor_core.ilu_inner_iterations_upper_bound << ','
            << "\"ilu_jacobi_factor_spmv_updates_lower_bound\":"
            << evidence.tensor_core.
                   ilu_jacobi_factor_spmv_updates_lower_bound << ','
            << "\"ilu_jacobi_factor_spmv_updates_upper_bound\":"
            << evidence.tensor_core.
                   ilu_jacobi_factor_spmv_updates_upper_bound << ','
            << "\"ilu_factor_age_generations\":"
            << evidence.tensor_core.ilu_factor_age_generations << ','
            << "\"ilu_accounting_complete\":" << std::boolalpha
            << evidence.tensor_core.ilu_accounting_complete << ','
            << "\"native_rank1_trace_solve_calls\":"
            << evidence.tensor_core.native_rank1_trace_solve_calls << ','
            << "\"hypre_trace_solve_calls\":"
            << evidence.tensor_core.hypre_trace_solve_calls << ','
            << "\"equilibration_builds\":"
            << evidence.tensor_core.equilibration_builds << ','
            << "\"equilibration_passes\":"
            << evidence.tensor_core.equilibration_passes << ','
            << "\"mgr_setup_calls\":"
            << evidence.tensor_core.mgr_setup_calls << ','
            << "\"mgr_preconditioner_applications\":"
            << evidence.tensor_core.mgr_preconditioner_applications << ','
            << "\"mgr_f_ilu_setups\":"
            << evidence.tensor_core.mgr_f_ilu_setups << ','
            << "\"mgr_f_ilu_applications\":"
            << evidence.tensor_core.mgr_f_ilu_applications << ','
            << "\"element_schwarz_patch_setups\":"
            << evidence.tensor_core.element_schwarz_patch_setups << ','
            << "\"element_schwarz_preconditioner_applications\":"
            << evidence.tensor_core.
                   element_schwarz_preconditioner_applications << ','
            << "\"element_schwarz_patch_rows\":"
            << evidence.tensor_core.element_schwarz_patch_rows << ','
            << "\"element_schwarz_minimum_multiplicity\":"
            << evidence.tensor_core.
                   element_schwarz_minimum_multiplicity << ','
            << "\"element_schwarz_maximum_multiplicity\":"
            << evidence.tensor_core.
                   element_schwarz_maximum_multiplicity << ','
            << "\"minimum_left_scale\":"
            << evidence.tensor_core.minimum_left_scale << ','
            << "\"maximum_left_scale\":"
            << evidence.tensor_core.maximum_left_scale << ','
            << "\"minimum_right_scale\":"
            << evidence.tensor_core.minimum_right_scale << ','
            << "\"maximum_right_scale\":"
            << evidence.tensor_core.maximum_right_scale << ','
            << "\"native_rank1_trace_solver_configured\":"
            << std::boolalpha
            << evidence.tensor_core.native_rank1_trace_solver_configured
            << ','
            << "\"native_element_schwarz_configured\":"
            << evidence.tensor_core.native_element_schwarz_configured << ','
            << "\"native_matrix_free_trace_operator_configured\":"
            << evidence.tensor_core.
                   native_matrix_free_trace_operator_configured << ','
            << "\"native_matrix_free_trace_operator_applications\":"
            << evidence.tensor_core.
                   native_matrix_free_trace_operator_applications << ','
            << "\"hypre_mgr_rank1_trace_solver_configured\":"
            << evidence.tensor_core.
                   hypre_mgr_rank1_trace_solver_configured << ','
            << "\"hypre_host_staged_rank2_trace_solver_configured\":"
            << evidence.tensor_core.
                   hypre_host_staged_rank2_trace_solver_configured << ','
            << "\"trace_solver_accounting_complete\":"
            << evidence.tensor_core.trace_solver_accounting_complete << ','
            << "\"trace_preconditioner_configured\":" << std::boolalpha
            << evidence.tensor_core.trace_preconditioner_configured << ','
            << "\"hardware_supported\":" << std::boolalpha
            << evidence.tensor_core.hardware_supported
            << ",\"volume_assembly_launches\":"
            << evidence.tensor_core.volume_assembly_launches
            << ",\"volume_mma_tiles\":"
            << evidence.tensor_core.volume_mma_tiles
            << ",\"condensation_launches\":"
            << evidence.tensor_core.condensation_launches
            << ",\"mma_tiles\":" << evidence.tensor_core.mma_tiles
            << ",\"fallback_count\":"
            << evidence.tensor_core.fallback_count
            << ",\"fallback_accounting_complete\":"
            << evidence.tensor_core.fallback_accounting_complete
            << ",\"no_fallback\":" << evidence.no_fallback
            << std::noboolalpha << '}'
            << ",\"transfers\":{"
            << "\"host_to_device_bytes\":"
            << sample.transfers.host_to_device_bytes
            << ",\"device_to_host_bytes\":"
            << sample.transfers.device_to_host_bytes
            << ",\"device_to_device_bytes\":"
            << sample.transfers.device_to_device_bytes
            << ",\"mpi_staging_bytes\":"
            << sample.transfers.mpi_staging_bytes
            << ",\"synchronization_events\":"
            << sample.transfers.synchronization_events
            << ",\"event_count\":" << sample.transfers.event_count
            << ",\"complete\":" << std::boolalpha
            << sample.transfers.complete << std::noboolalpha << '}'
            << ",\"gates\":{"
            << "\"candidate_ready\":" << std::boolalpha
            << sample.prepared.candidate_ready
            << ",\"finite\":" << sample.prepared.finite
            << ",\"admissible\":" << sample.prepared.admissible
            << ",\"recovery\":" << sample.prepared.recovery
            << ",\"true_residual\":" << sample.prepared.true_residual
            << ",\"conservation\":" << sample.prepared.conservation
            << ",\"telemetry\":" << sample.prepared.telemetry
            << ",\"products_coupled_and_ordered\":"
            << evidence.products.CoupledAndOrdered()
            << ",\"success\":" << sample.prepared.success
            << std::noboolalpha << "}}";
}

int RunBenchmark(const Options& options)
{
  using clock = std::chrono::steady_clock;
  const auto initialization_started = clock::now();
  mfem::Device device("cuda");
  int active_device = -1;
  int driver_version = 0;
  int runtime_version = 0;
  cudaDeviceProp device_properties{};
  if (cudaGetDevice(&active_device) != cudaSuccess || active_device < 0 ||
      cudaGetDeviceProperties(&device_properties, active_device) !=
          cudaSuccess || device_properties.multiProcessorCount <= 0 ||
      cudaDriverGetVersion(&driver_version) != cudaSuccess ||
      cudaRuntimeGetVersion(&runtime_version) != cudaSuccess)
  {
    std::cerr << "{\"schema\":2,\"success\":false,"
                 "\"error\":\"CUDA device metadata query failed\"}\n";
    return 1;
  }
  const DeviceMemorySnapshot memory_before_problem = CaptureDeviceMemory();

  mfem::ParMesh mesh = MakePeriodicMesh(
      options.mesh_n, options.domain_periods);
  const int elements = mesh.GetNE();
  const int principal_blocks =
      (elements + principal_elements_per_block - 1) /
      principal_elements_per_block;
  BorrowedFields fields = MakeFields(
      mesh, options.domain_periods, options.order);
  if (elements != options.mesh_n * options.mesh_n ||
      fields.volume_scalar_count == 0 ||
      fields.initial.size() !=
          fields.volume_scalar_count * conserved_components)
  {
    std::cerr << "{\"schema\":2,\"success\":false,"
                 "\"error\":\"benchmark field initialization failed\"}\n";
    return 1;
  }

  GenericEulerHdgCudaSession session;
  const EulerHdgRequest request = MakeRequest(
      mesh, fields, options);
  if (CudaQualificationTestAccess::InitializeBenchmarkTracePolicy(
          session, request, options.trace_policy).status !=
          HdgEulerSessionStatus::initialized ||
      cudaDeviceSynchronize() != cudaSuccess)
  {
    std::cerr << "{\"schema\":2,\"success\":false,"
                 "\"error\":\"CUDA session initialization failed\"}\n";
    return 1;
  }
  const auto initialization_finished = clock::now();
  const double initialization_seconds =
      std::chrono::duration<double>(
          initialization_finished - initialization_started).count();
  const DeviceMemorySnapshot memory_after_initialization =
      CaptureDeviceMemory();

  CudaAcceptedStepProblemMetadata problem{};
  const bool problem_captured =
      CudaQualificationTestAccess::CaptureProblemMetadata(session, problem);
  const int expected_interior_components =
      options.regularization == RegularizationKind::igr
      ? dg::hdg::runtime::cuda_w8a::igr_interior_components
      : conserved_components;
  const int expected_trace_components =
      options.regularization == RegularizationKind::igr
      ? dg::hdg::runtime::cuda_w8a::igr_trace_components
      : conserved_components;
  const std::int64_t expected_interior_unknowns =
      static_cast<std::int64_t>(elements) *
      problem.scalar_interior_dofs * expected_interior_components;
  const std::int64_t expected_true_trace_unknowns =
      2LL * options.mesh_n * options.mesh_n *
      problem.scalar_face_dofs * expected_trace_components;
  const int expected_volume_assembly_order =
      options.regularization == RegularizationKind::igr
      ? request.numerical_policy.igr_assembly_quadrature_orders.volume_order
      : request.numerical_policy.conventional_assembly_quadrature_order;
  const int expected_volume_points = mfem::IntRules.Get(
      mfem::Geometry::SQUARE,
      expected_volume_assembly_order).GetNPoints();
  if (!problem_captured ||
      problem.elements != elements || problem.scalar_interior_dofs <= 0 ||
      problem.scalar_face_dofs <= 0 ||
      problem.volume_assembly_order != expected_volume_assembly_order ||
      problem.volume_points != expected_volume_points ||
      problem.interior_components != expected_interior_components ||
      problem.trace_components != expected_trace_components ||
      problem.igr_active !=
          (options.regularization == RegularizationKind::igr) ||
      problem.interior_unknowns != expected_interior_unknowns ||
      problem.true_trace_unknowns != expected_true_trace_unknowns ||
      problem.true_trace_unknowns != problem.global_true_trace_unknowns)
  {
    std::cerr << "{\"schema\":2,\"success\":false,"
                 "\"error\":\"CUDA problem metadata validation failed\"}\n";
    return 1;
  }
  const int krylov_restart_dimension =
      dg::hdg::generic::EulerHdgKrylovRestartDimension(
          problem.global_true_trace_unknowns);
  if (krylov_restart_dimension <= 0)
  {
    std::cerr << "{\"schema\":2,\"success\":false,"
                 "\"error\":\"invalid Krylov restart dimension\"}\n";
    return 1;
  }

  std::vector<double> exported_interior(
      options.export_interior ? fields.initial.size() : 0, 0.0);
  std::array<StepSample, total_steps> samples{};
  std::string error;
  for (int index = 0; index < total_steps; ++index)
  {
    const bool warmup = index < warmup_steps;
    const int measured_index = warmup ? -1 : index - warmup_steps;
    const bool profile_this_step = options.profile_measured_step >= 0 &&
        measured_index == options.profile_measured_step;
    const auto run_step = [&]()
    {
      return RunStep(session, fields, exported_interior, options.timestep,
                     options.export_interior, options.engine,
                     options.regularization == RegularizationKind::igr,
                     options.trace_policy,
                     static_cast<std::uint64_t>(krylov_restart_dimension), warmup,
                     measured_index,
                     samples[static_cast<std::size_t>(index)], error);
    };
    bool step_succeeded = false;
    if (profile_this_step)
    {
      dg_igr::observability::set_runtime_enabled(true);
      if (cudaProfilerStart() != cudaSuccess)
      {
        dg_igr::observability::set_runtime_enabled(false);
        std::cerr << "{\"schema\":2,\"success\":false,"
                     "\"error\":\"CUDA profiler start failed\"}\n";
        return 1;
      }
      {
        [[maybe_unused]] dg_igr::NVTXRange accepted_step_range(
            "HDG/profiled-accepted-step");
        step_succeeded = run_step();
      }
      const cudaError_t profile_sync_status = cudaDeviceSynchronize();
      const cudaError_t profile_stop_status = cudaProfilerStop();
      dg_igr::observability::set_runtime_enabled(false);
      if (profile_sync_status != cudaSuccess ||
          profile_stop_status != cudaSuccess)
      {
        std::cerr << "{\"schema\":2,\"success\":false,"
                     "\"error\":\"CUDA profiler completion failed\"}\n";
        return 1;
      }
    }
    else
    {
      step_succeeded = run_step();
    }
    if (!step_succeeded)
    {
      std::cerr << "{\"schema\":2,\"success\":false,\"step_index\":"
                << index << ",\"error\":\"" << error << "\"}\n";
      return 1;
    }
  }

  std::array<double, measured_steps> durations{};
  for (int index = 0; index < measured_steps; ++index)
    durations[static_cast<std::size_t>(index)] =
        samples[static_cast<std::size_t>(index + warmup_steps)].seconds;
  const DeviceMemorySnapshot memory_after_samples = CaptureDeviceMemory();
  const UmpireHighWatermarkSnapshot umpire_high_watermarks =
      CaptureUmpireHighWatermarks();
  const auto& reported_trace_policy =
      samples[static_cast<std::size_t>(warmup_steps)]
          .prepared.evidence.tensor_core;
  const bool production_selectable =
      options.trace_policy == CudaTracePolicyKind::engine_default;
  const bool equilibrated_field_split = options.trace_policy ==
      CudaTracePolicyKind::
          native_element_schwarz_equilibrated_field_split_fp64 ||
      options.trace_policy == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_equilibrated_field_split_fp64 ||
      options.trace_policy == CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64;
  const bool warp_equilibrated_field_split = options.trace_policy ==
      CudaTracePolicyKind::
          native_element_schwarz_matrix_free_warp_equilibrated_field_split_fp64;
  const NativeElementSchwarzFieldSplitLayout field_split_layout =
      equilibrated_field_split
      ? NativeElementSchwarzIgrFieldSplitLayout(options.order)
      : NativeElementSchwarzFieldSplitLayout{};

  std::cout << std::setprecision(17)
            << "{\"schema\":2,"
            << "\"benchmark\":\"dg_igr_hdg_cuda_accepted_step\","
            << "\"success\":true,"
            << "\"variant\":\""
            << (production_selectable ? "production" : "ablation")
            << "\","
            << "\"production_selectable\":"
            << (production_selectable ? "true" : "false") << ','
            << "\"backend\":\"cuda\","
            << "\"cuda_engine\":\"" << EngineName(options.engine)
            << "\","
            << "\"trace_policy\":\""
            << CudaTracePolicyKindName(options.trace_policy) << "\","
            << "\"trace_solver\":\""
            << TraceSolverName(reported_trace_policy) << "\","
            << "\"trace_preconditioner\":\""
            << CudaTracePreconditionerName(
                   reported_trace_policy.trace_preconditioner)
            << "\","
            << "\"trace_ilu_type\":" << reported_trace_policy.ilu_type
            << ",\"trace_ilu_level_of_fill\":"
            << reported_trace_policy.ilu_level_of_fill
            << ",\"trace_ilu_local_reordering\":"
            << reported_trace_policy.ilu_local_reordering
            << ",\"trace_ilu_application_iterations\":"
            << reported_trace_policy.ilu_application_iterations
            << ",\"trace_ilu_triangular_solve_kind\":\""
            << CudaIluTriangularSolveKindName(
                   reported_trace_policy.ilu_triangular_solve_kind) << "\","
            << "\"trace_ilu_lower_jacobi_iterations\":"
            << reported_trace_policy.ilu_lower_jacobi_iterations
            << ",\"trace_ilu_upper_jacobi_iterations\":"
            << reported_trace_policy.ilu_upper_jacobi_iterations
            << ",\"trace_ilu_accounting_kind\":\""
            << CudaIluAccountingKindName(
                   reported_trace_policy.ilu_accounting_kind) << "\","
            << "\"trace_ilu_factor_reuse_policy\":\""
            << CudaIluFactorReusePolicyName(
                   reported_trace_policy.ilu_factor_reuse_policy) << "\","
            << "\"trace_native_element_schwarz_configured\":"
            << std::boolalpha
            << reported_trace_policy.native_element_schwarz_configured << ','
            << "\"trace_native_matrix_free_operator_configured\":"
            << reported_trace_policy.
                   native_matrix_free_trace_operator_configured << ','
            << "\"trace_element_schwarz_patch_rows\":"
            << reported_trace_policy.element_schwarz_patch_rows << ','
            << "\"trace_element_schwarz_minimum_multiplicity\":"
            << reported_trace_policy.element_schwarz_minimum_multiplicity
            << ','
            << "\"trace_element_schwarz_maximum_multiplicity\":"
            << reported_trace_policy.element_schwarz_maximum_multiplicity
            << std::noboolalpha << ','
            << "\"trace_element_schwarz_factorization\":\""
            << (equilibrated_field_split
                    ? "lower-multiplicative-uhat4-sigmahat"
                    : reported_trace_policy.
                            native_element_schwarz_configured
                        ? "full-patch-lu" : "not-applicable")
            << "\","
            << "\"trace_element_schwarz_equilibration_passes\":"
            << (equilibrated_field_split
                    ? native_element_schwarz_equilibration_passes : 0)
            << ','
            << "\"trace_element_schwarz_euler_rows\":"
            << field_split_layout.euler_rows << ','
            << "\"trace_element_schwarz_sigma_rows\":"
            << field_split_layout.sigma_rows << ','
            << "\"trace_element_schwarz_apply\":\""
            << (warp_equilibrated_field_split
                    ? "cooperative-pivoted-lu-fp64"
                    : equilibrated_field_split
                        ? "scalar-pivoted-lu-fp64"
                        : reported_trace_policy.
                                native_element_schwarz_configured
                            ? "scalar-full-patch-pivoted-lu-fp64"
                            : "not-applicable")
            << "\","
            << "\"trace_element_schwarz_apply_tile_width\":"
            << (warp_equilibrated_field_split
                    ? (options.order == 1 ? 16 : 32) : 0) << ','
            << "\"mode\":\""
            << (options.regularization == RegularizationKind::igr
                    ? "igr" : "conventional") << "\","
            << "\"precision\":\"fp64\","
            << "\"time_integrator\":\"backward_euler\","
            << "\"time_step_policy\":\"fixed\","
            << "\"gamma\":" << benchmark_gamma << ','
            << "\"rank_count\":1,"
            << "\"polynomial_order\":" << options.order << ','
            << "\"mesh_n\":" << options.mesh_n << ','
            << "\"domain_periods\":" << options.domain_periods << ','
            << "\"domain_length\":" << options.domain_periods << ','
            << "\"cells_per_period\":"
            << options.mesh_n / options.domain_periods << ','
            << "\"element_h\":"
            << static_cast<double>(options.domain_periods) /
                static_cast<double>(options.mesh_n) << ','
            << "\"elements\":" << elements << ','
            << "\"volume_assembly_order\":"
            << problem.volume_assembly_order << ','
            << "\"volume_points\":" << problem.volume_points << ','
            << "\"interior_components\":"
            << problem.interior_components << ','
            << "\"trace_components\":" << problem.trace_components << ','
            << "\"interior_unknowns\":" << problem.interior_unknowns << ','
            << "\"true_trace_unknowns\":"
            << problem.true_trace_unknowns << ','
            << "\"gpu_sm_count\":"
            << device_properties.multiProcessorCount << ','
            << "\"gpu_name\":\"" << device_properties.name << "\","
            << "\"gpu_device_ordinal\":" << active_device << ','
            << "\"gpu_uuid\":\""
            << CudaUuidString(device_properties.uuid) << "\","
            << "\"gpu_compute_capability\":"
            << 10 * device_properties.major + device_properties.minor << ','
            << "\"cuda_driver_version\":" << driver_version << ','
            << "\"cuda_runtime_version\":" << runtime_version << ','
            << "\"principal_threads\":" << principal_threads << ','
            << "\"principal_element_threads\":"
            << principal_element_threads << ','
            << "\"principal_elements_per_block\":"
            << principal_elements_per_block << ','
            << "\"principal_blocks\":" << principal_blocks << ','
            << "\"principal_execution\":"
               "\"four_lane_subwarp_per_element\","
            << "\"initial_state_kind\":"
               "\"projected_smooth_periodic\","
            << "\"initial_trace_kind\":"
               "\"projected_from_committed_interior\","
            << "\"source_kind\":\"zero\","
            << "\"boundary_kind\":\"periodic_topology\","
            << "\"precision_policy\":{"
               "\"external_state\":\"fp64\","
               "\"local_lu\":\"fp64\","
               "\"local_triangular_solve\":\"fp64\","
               "\"tensor_operands\":\"fp64\","
               "\"tensor_accumulation\":\"fp64\","
               "\"acceptance_gates\":\"fp64\","
               "\"mixed_precision_enabled\":false,"
               "\"iterative_refinement_enabled\":false,"
               "\"residual_replacement_enabled\":false,"
               "\"precision_correction_enabled\":false},"
            << "\"numerical_policy\":{"
            << "\"volume_projection_quadrature_order\":"
            << request.numerical_policy.volume_projection_quadrature_order
            << ",\"true_trace_projection_quadrature_order\":"
            << request.numerical_policy.true_trace_projection_quadrature_order
            << ",\"conventional_assembly_quadrature_order\":"
            << request.numerical_policy.conventional_assembly_quadrature_order
            << ",\"igr_volume_assembly_quadrature_order\":"
            << request.numerical_policy.igr_assembly_quadrature_orders
                   .volume_order
            << ",\"igr_face_assembly_quadrature_order\":"
            << request.numerical_policy.igr_assembly_quadrature_orders
                   .face_order
            << ",\"maximum_nonlinear_updates\":"
            << request.numerical_policy.maximum_nonlinear_updates
            << ",\"requested_maximum_linear_iterations\":"
            << request.numerical_policy.maximum_linear_iterations
            << ",\"resolved_maximum_linear_iterations\":"
            << dg::hdg::generic::ResolveEulerHdgMaximumLinearIterations(
                   request.numerical_policy.maximum_linear_iterations,
                   problem.global_true_trace_unknowns)
            << ",\"krylov_restart_dimension\":"
            << krylov_restart_dimension
            << ",\"maximum_line_search_trials\":"
            << request.numerical_policy.maximum_line_search_trials
            << ",\"nonlinear_relative_tolerance\":"
            << request.numerical_policy.nonlinear_relative_tolerance
            << ",\"newton_correction_linear_relative_tolerance\":"
            << request.numerical_policy.
                   newton_correction_linear_relative_tolerance
            << ",\"final_coherent_linear_relative_tolerance\":"
            << request.numerical_policy.
                   final_coherent_linear_relative_tolerance
            << ",\"true_residual_tightening_linear_relative_tolerance\":"
            << request.numerical_policy.
                   true_residual_tightening_linear_relative_tolerance
            << ",\"true_residual_relative_tolerance\":"
            << request.numerical_policy.true_residual_relative_tolerance
            << ",\"conservation_relative_tolerance\":"
            << request.numerical_policy.conservation_relative_tolerance
            << ",\"armijo_constant\":"
            << request.numerical_policy.armijo_constant
            << ",\"regularization\":\""
            << (request.numerical_policy.stabilization.kind ==
                        RegularizationKind::igr ? "igr" : "none")
            << "\",\"igr_alpha\":"
            << request.numerical_policy.stabilization.alpha
            << ",\"igr_beta\":"
            << request.numerical_policy.stabilization.beta << "},"
            << "\"dt\":" << options.timestep << ','
            << "\"host_export\":"
            << (options.export_interior ? "true" : "false") << ','
            << "\"build_identity\":{"
            << "\"source_revision\":\""
            << DG_IGR_BENCHMARK_SOURCE_REVISION << "\","
            << "\"source_tree\":\""
            << DG_IGR_BENCHMARK_SOURCE_TREE << "\","
            << "\"source_status\":\""
            << DG_IGR_BENCHMARK_SOURCE_STATUS << "\","
            << "\"build_directory\":\""
            << DG_IGR_BENCHMARK_BUILD_DIRECTORY << "\","
            << "\"cmake_cache\":\""
            << DG_IGR_BENCHMARK_CMAKE_CACHE << "\"},"
            << "\"profiling_observability\":"
            << (options.profile_measured_step >= 0 ? "true" : "false")
            << ",\"profile_measured_step\":"
            << options.profile_measured_step << ','
            << "\"initialization_in_primary_timing\":false,"
            << "\"initialization_duration_seconds\":";
  PrintJsonNumber(initialization_seconds);
  std::cout << ",\"device_memory_snapshots\":{"
            << "\"after_device_selection_before_problem\":";
  PrintMemorySnapshot(memory_before_problem);
  std::cout << ",\"after_initialization\":";
  PrintMemorySnapshot(memory_after_initialization);
  std::cout << ",\"after_samples\":";
  PrintMemorySnapshot(memory_after_samples);
  std::cout << "},\"umpire_allocator_high_watermarks_after_samples\":";
  PrintUmpireHighWatermarks(umpire_high_watermarks);
  std::cout << ','
            << "\"timing_scope\":\"Prepare+CapturePreparedObservation+"
            << (options.export_interior ? "ExportPreparedInteriorD2H+" : "")
            << "AcknowledgePreparedOutput+Commit\","
            << "\"warmup_steps\":" << warmup_steps << ','
            << "\"measured_steps\":" << measured_steps << ','
            << "\"warmup_duration_seconds\":";
  PrintJsonNumber(samples[0].seconds);
  std::cout << ",\"raw_durations_seconds\":[";
  for (std::size_t index = 0; index < durations.size(); ++index)
  {
    if (index != 0) std::cout << ',';
    PrintJsonNumber(durations[index]);
  }
  std::cout << "],\"median_seconds\":";
  PrintJsonNumber(Median(durations));
  std::cout << ",\"population_stddev_seconds\":";
  PrintJsonNumber(PopulationStandardDeviation(durations));
  std::cout << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index)
  {
    if (index != 0) std::cout << ',';
    PrintStep(samples[index]);
  }
  std::cout << "]}\n";
  return 0;
}

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  Options options;
  std::string error;
  if (!ParseOptions(argc, argv, options, error))
  {
    std::cerr << error << '\n';
    PrintUsage(argv[0]);
    return 2;
  }
  if (options.help)
  {
    PrintUsage(argv[0]);
    return 0;
  }
  if (ranks != 1)
  {
    std::cerr << "{\"schema\":1,\"success\":false,"
                 "\"error\":\"benchmark requires exactly one MPI rank\"}\n";
    return 2;
  }
  if (!ExactExpectedBuildIdentity(options))
  {
    std::cerr << "{\"schema\":2,\"success\":false,"
                 "\"error\":\"benchmark source/build identity mismatch\"}\n";
    return 2;
  }
  if (options.profile_measured_step >= 0 &&
      !dg_igr::observability::nvtx_backend_enabled)
  {
    std::cerr << "{\"schema\":2,\"success\":false,"
                 "\"error\":\"NVTX profiler build required\"}\n";
    return 2;
  }
  const int result = RunBenchmark(options);
  // Flush CUDA activity records after every benchmark route.  This runs only
  // after the session, mesh, and MFEM device have been destroyed and remains
  // outside initialization and accepted-step timing.  In particular, it lets
  // profiler captures retain short-lived kernel and transfer activity rather
  // than depending on process-exit cleanup.
  const cudaError_t reset_status = cudaDeviceReset();
  if (result == 0 && reset_status != cudaSuccess)
  {
    std::cerr << "{\"schema\":2,\"success\":false,"
                 "\"error\":\"CUDA device reset failed\"}\n";
    return 1;
  }
  return result;
}

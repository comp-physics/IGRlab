#include "numerics/hdg/runtime/app/solver.hpp"

#include "solver_entry_detail.hpp"

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr_rank2.cuh"
#else
#include "numerics/hdg/runtime/cuda/persistent_sc25_scalar_igr.cuh"
#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#endif
#include "numerics/hdg/runtime/sha256.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dg::hdg::runtime::app
{
namespace
{

namespace cuda = ::dg::hdg::runtime::cuda_w8a;
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
namespace scalar = ::dg::hdg::runtime::cuda_w8b::sc25::rank2;
#else
namespace scalar = ::dg::hdg::runtime::cuda_w8b::sc25;
#endif

#if !defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)

template <typename Selection>
using AppCudaEngine = cuda::PersistentCudaSmoothEulerEngine<
    Selection, cuda::UmpireDeviceAllocatorPolicy, cuda::CudaTransportPolicy,
    cuda::CudaMpiCollectivePolicy>;

static_assert(!std::is_polymorphic<CudaRank1OneGpuRuntimeSelection>::value &&
                  !std::is_polymorphic<
                      CudaRank2OneGpuHostStagedRuntimeSelection>::value &&
                  !std::is_polymorphic<cuda::UmpireDeviceAllocatorPolicy>::value &&
                  !std::is_polymorphic<cuda::CudaTransportPolicy>::value,
              "the production CUDA application uses static policies");

bool CollectiveAll(const bool local, const MPI_Comm communicator) noexcept
{
  const int value = local ? 1 : 0;
  int reduced = 0;
  return MPI_Allreduce(&value, &reduced, 1, MPI_INT, MPI_MIN, communicator) ==
          MPI_SUCCESS && reduced == 1;
}

enum class PostPublicationFault : unsigned char
{
  after_marker,
  before_finalize
};

bool ConsumePostPublicationFault(const PostPublicationFault expected,
                                 const std::uint64_t step,
                                 const int rank,
                                 int& source_rank) noexcept
{
  source_rank = -1;
  static bool marker_consumed = false;
  static bool finalize_consumed = false;
  bool& consumed = expected == PostPublicationFault::after_marker
      ? marker_consumed : finalize_consumed;
  if (consumed) return false;
  const char* const raw =
      std::getenv("DG_IGR_HDG_TEST_POST_PUBLICATION_FAILURE");
  if (raw == nullptr) return false;
  const std::string_view value{raw};
  const std::string_view phase = expected == PostPublicationFault::after_marker
      ? std::string_view{"after-marker"}
      : std::string_view{"before-finalize"};
  if (value.size() <= phase.size() + 2 ||
      value.substr(0, phase.size()) != phase ||
      value[phase.size()] != ':')
    return false;
  const std::size_t separator = value.find(':', phase.size() + 1);
  if (separator == std::string_view::npos) return false;
  std::uint64_t selected_step = 0;
  int selected_rank = -1;
  const auto step_result = std::from_chars(
      value.data() + phase.size() + 1, value.data() + separator,
      selected_step);
  const auto rank_result = std::from_chars(
      value.data() + separator + 1, value.data() + value.size(),
      selected_rank);
  if (step_result.ec != std::errc{} ||
      step_result.ptr != value.data() + separator ||
      rank_result.ec != std::errc{} ||
      rank_result.ptr != value.data() + value.size() ||
      selected_step != step || selected_rank != rank)
  {
    if (step_result.ec == std::errc{} &&
        step_result.ptr == value.data() + separator &&
        rank_result.ec == std::errc{} &&
        rank_result.ptr == value.data() + value.size() &&
        selected_step == step)
      source_rank = selected_rank;
    return false;
  }
  source_rank = selected_rank;
  consumed = true;
  return true;
}

bool PostPublicationFaultAllowsRetry() noexcept
{
  const char* const raw =
      std::getenv("DG_IGR_HDG_TEST_POST_PUBLICATION_RETRY");
  return raw == nullptr || std::string_view{raw} == "1";
}

std::string StepName(const std::uint64_t step)
{
  std::ostringstream name;
  name << "state-step-" << std::setw(6) << std::setfill('0') << step
       << ".vtk";
  return name.str();
}

bool OutputStep(const ValidatedApplicationConfig& config,
                const std::uint64_t step) noexcept
{
  return std::find(config.output_steps().begin(), config.output_steps().end(),
                   step) != config.output_steps().end();
}

std::string GpuUuidText(const cuda::CudaRunEvidence& evidence)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string text = "GPU-";
  text.resize(4 + 2 * evidence.physical_gpu_uuid.size());
  for (std::size_t index = 0; index < evidence.physical_gpu_uuid.size(); ++index)
  {
    const unsigned char byte = evidence.physical_gpu_uuid[index];
    text[4 + 2 * index] = digits[byte >> 4U];
    text[5 + 2 * index] = digits[byte & 0xfU];
  }
  return text;
}

std::string Hex(const StateHash& hash)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(2 * hash.bytes.size(), '0');
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    result[2 * index] = digits[hash.bytes[index] >> 4U];
    result[2 * index + 1] = digits[hash.bytes[index] & 0xfU];
  }
  return result;
}

CheckpointRuntimeIdentityV1 RuntimeIdentity(
    const cuda::CudaRunEvidence& evidence,
    const PersistentResourceDescriptor& resources) noexcept
{
  CheckpointRuntimeIdentityV1 identity{};
  const std::string uuid = GpuUuidText(evidence);
  identity.physical_gpu_uuid = StableIdentity::Copy(uuid);
  identity.ownership_hash = resources.partition_hash;
  return identity;
}

cuda::CudaFailurePhase ToCudaPhase(const FailurePhase phase) noexcept
{
  switch (phase)
  {
    case FailurePhase::Assembly: return cuda::CudaFailurePhase::assembly;
    case FailurePhase::Condensation: return cuda::CudaFailurePhase::condensation;
    case FailurePhase::TraceSolve: return cuda::CudaFailurePhase::trace_solve;
    case FailurePhase::Recovery: return cuda::CudaFailurePhase::recovery;
    case FailurePhase::Diagnostics: return cuda::CudaFailurePhase::diagnostics;
    case FailurePhase::AfterRecoveryBeforeCommit:
      return cuda::CudaFailurePhase::after_recovery_before_commit;
    case FailurePhase::Setup: return cuda::CudaFailurePhase::setup;
    case FailurePhase::CheckpointStaging:
    case FailurePhase::OutputStaging:
    case FailurePhase::Commit:
    case FailurePhase::RollbackAudit:
    case FailurePhase::TerminalEvidence:
      return cuda::CudaFailurePhase::terminal_evidence;
  }
  return cuda::CudaFailurePhase::terminal_evidence;
}

const char* CudaPhaseName(const cuda::CudaFailurePhase phase) noexcept
{
  switch (phase)
  {
    case cuda::CudaFailurePhase::setup: return "Setup";
    case cuda::CudaFailurePhase::assembly: return "Assembly";
    case cuda::CudaFailurePhase::condensation: return "Condensation";
    case cuda::CudaFailurePhase::trace_solve: return "TraceSolve";
    case cuda::CudaFailurePhase::recovery: return "Recovery";
    case cuda::CudaFailurePhase::after_recovery_before_commit:
      return "AfterRecoveryBeforeCommit";
    case cuda::CudaFailurePhase::diagnostics: return "Diagnostics";
    case cuda::CudaFailurePhase::rollback_audit: return "RollbackAudit";
    case cuda::CudaFailurePhase::terminal_evidence: return "TerminalEvidence";
  }
  return "Unknown";
}

const char* ApplicationPhaseName(const FailurePhase phase) noexcept
{
  switch (phase)
  {
    case FailurePhase::Setup: return "Setup";
    case FailurePhase::Assembly: return "Assembly";
    case FailurePhase::Condensation: return "Condensation";
    case FailurePhase::TraceSolve: return "TraceSolve";
    case FailurePhase::Recovery: return "Recovery";
    case FailurePhase::Diagnostics: return "Diagnostics";
    case FailurePhase::CheckpointStaging: return "CheckpointStaging";
    case FailurePhase::OutputStaging: return "OutputStaging";
    case FailurePhase::AfterRecoveryBeforeCommit:
      return "AfterRecoveryBeforeCommit";
    case FailurePhase::Commit: return "Commit";
    case FailurePhase::RollbackAudit: return "RollbackAudit";
    case FailurePhase::TerminalEvidence: return "TerminalEvidence";
  }
  return "Unknown";
}

void AppendCudaFailureProvenance(const cuda::CudaStepEvidence& evidence,
                                 std::string& detail)
{
  if (!detail.empty()) detail += "; ";
  detail += "phase=";
  detail += CudaPhaseName(evidence.phase);
  detail += " source_rank=" + std::to_string(evidence.source_rank);
}

void AppendApplicationFailureProvenance(const FailurePhase phase,
                                        const int source_rank,
                                        std::string& detail)
{
  if (!detail.empty()) detail += "; ";
  detail += "phase=";
  detail += ApplicationPhaseName(phase);
  detail += " source_rank=" + std::to_string(source_rank);
}

bool ScientificFailureReason(const FailureReason reason) noexcept
{
  return reason == FailureReason::finite_gate ||
      reason == FailureReason::admissibility_gate ||
      reason == FailureReason::recovery_gate ||
      reason == FailureReason::true_residual_gate ||
      reason == FailureReason::success_gate;
}

cuda::CudaConfig MakeCudaConfig(const ValidatedApplicationConfig& config)
{
  cuda::CudaConfig numerical{};
  numerical.experiment = StableIdentity::Copy(cuda::experiment_id);
  numerical.order = config.polynomial_order();
  numerical.mesh_n = static_cast<int>(config.mesh_n());
  numerical.dt = config.dt();
  numerical.requested_device = config.device_id();
  numerical.expected_rank_count =
      static_cast<int>(config.deferred_runtime().required_mpi_rank_count);
  numerical.require_host_staged_rank2 = numerical.expected_rank_count == 2;
  numerical.capture_oracle = false;
  const char* const scientific_test =
      std::getenv("DG_IGR_HDG_TEST_SCIENTIFIC_GATE_FAILURE");
  numerical.force_private_scientific_gate_failure =
      scientific_test != nullptr && std::string_view{scientific_test} == "1";
  if (config.injection().mode == FailureInjectionModeV1::prescribed &&
      (config.injection().phase == FailurePhase::Assembly ||
       config.injection().phase == FailurePhase::Condensation ||
       config.injection().phase == FailurePhase::TraceSolve ||
       config.injection().phase == FailurePhase::Recovery ||
       config.injection().phase == FailurePhase::Diagnostics ||
       config.injection().phase == FailurePhase::AfterRecoveryBeforeCommit))
  {
    numerical.injection = {
        ToCudaPhase(config.injection().phase),
        config.injection().target_step,
        static_cast<int>(config.injection().source_rank), true,
        cuda::CudaTokenFault::none};
  }
  return numerical;
}

bool GatherCheckpoints(const cuda::CudaSerializedCheckpointV1& local,
                       const CudaApplicationContextV1& context,
                       std::vector<cuda::CudaSerializedCheckpointV1>& gathered,
                       std::string& detail)
{
  static_assert(std::is_trivially_copyable<
                    cuda::CudaSerializedCheckpointV1>::value,
                "CUDA checkpoint transport is POD");
  if (sizeof(cuda::CudaSerializedCheckpointV1) >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    detail = "CUDA checkpoint exceeds MPI byte-count range";
    return false;
  }
  bool resized = true;
  if (context.rank() == 0)
  {
    try { gathered.resize(static_cast<std::size_t>(context.ranks())); }
    catch (...) { resized = false; }
  }
  if (!CollectiveAll(resized, context.communicator()))
  {
    detail = "CUDA checkpoint gather allocation failed";
    return false;
  }
  const int status = MPI_Gather(
      &local, static_cast<int>(sizeof(local)), MPI_BYTE,
      context.rank() == 0 ? gathered.data() : nullptr,
      static_cast<int>(sizeof(local)), MPI_BYTE, 0, context.communicator());
  if (!CollectiveAll(status == MPI_SUCCESS, context.communicator()))
  {
    detail = "MPI_Gather CUDA checkpoint failed";
    return false;
  }
  return true;
}

struct RankStateIdentityBlock
{
  std::uint64_t interior_count = 0;
  std::uint64_t trace_count = 0;
  StateHash interior{};
  StateHash trace{};
};

bool GlobalCudaStateIdentity(
    const cuda::CudaSerializedCheckpointV1& local,
    const CudaApplicationContextV1& context,
    StateHash& identity, std::string& detail) noexcept
{
  static_assert(std::is_trivially_copyable<RankStateIdentityBlock>::value,
                "rank state identity is MPI-byte transportable");
  const RankStateIdentityBlock block{
      local.interior_count, local.owned_true_trace_count,
      local.local_interior_hash, local.local_trace_hash};
  std::array<RankStateIdentityBlock, 2> rank_ordered{};
  const bool gathered = MPI_Allgather(
      &block, static_cast<int>(sizeof(block)), MPI_BYTE,
      rank_ordered.data(), static_cast<int>(sizeof(block)), MPI_BYTE,
      context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(gathered, context.communicator()))
  {
    detail = "rank-ordered CUDA state identity gather failed";
    return false;
  }
  identity.bytes = ComputeSha256(
      rank_ordered.data(), static_cast<std::size_t>(context.ranks()) *
          sizeof(RankStateIdentityBlock));
  return identity != StateHash{};
}

bool MakePhysicalVtk(
    const ValidatedApplicationConfig& config,
    const std::vector<cuda::CudaSerializedCheckpointV1>* rank_states,
    std::string& vtk, std::string& detail)
{
  try
  {
    const int n = static_cast<int>(config.mesh_n());
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        n, n, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(2));
    translations[0][0] = 1.0;
    translations[1][1] = 1.0;
    mfem::Mesh mesh = mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations));
    mfem::L2_FECollection collection(
        config.polynomial_order(), 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::FiniteElementSpace space(
        &mesh, &collection, 4, mfem::Ordering::byNODES);
    mfem::GridFunction field(&space);
    field = 0.0;
    if (rank_states == nullptr)
    {
      auto exact = [](const mfem::Vector& point, mfem::Vector& state)
      {
        state.SetSize(4);
        double values[4]{};
        cuda::detail::CudaManufacturedState(point[0], point[1], values);
        for (int component = 0; component < 4; ++component)
          state[component] = values[component];
      };
      mfem::VectorFunctionCoefficient coefficient(4, exact);
      field.ProjectCoefficient(coefficient);
    }
    else
    {
      double* const field_values = field.HostWrite();
      if (field_values == nullptr && field.Size() != 0)
      {
        detail = "CUDA VTK host field storage is unavailable";
        return false;
      }
      std::fill_n(field_values, field.Size(), 0.0);
      const int ranks = static_cast<int>(rank_states->size());
      if (ranks != 1 && ranks != 2)
      {
        detail = "CUDA VTK rank count is invalid";
        return false;
      }
      const int global_elements = n * n;
      std::array<int, 2> local_elements{};
      for (int element = 0; element < global_elements; ++element)
      {
        const int owner = element * ranks / global_elements;
        const auto& source = (*rank_states)[static_cast<std::size_t>(owner)];
        const mfem::FiniteElement* finite_element = space.GetFE(element);
        if (finite_element == nullptr)
        {
          detail = "CUDA VTK finite element is unavailable";
          return false;
        }
        const int scalar_dofs = finite_element->GetDof();
        const int element_dofs = 4 * scalar_dofs;
        const std::size_t source_offset = static_cast<std::size_t>(
            local_elements[static_cast<std::size_t>(owner)]++) *
            static_cast<std::size_t>(element_dofs);
        if (source_offset + static_cast<std::size_t>(element_dofs) >
            source.interior_count)
        {
          detail = "CUDA VTK rank-local state extent is incomplete";
          return false;
        }
        mfem::Array<int> vector_dofs;
        space.GetElementVDofs(element, vector_dofs);
        if (vector_dofs.Size() != element_dofs)
        {
          detail = "CUDA VTK element map extent is invalid";
          return false;
        }
        for (int component = 0; component < 4; ++component)
          for (int scalar = 0; scalar < scalar_dofs; ++scalar)
          {
            const int local = scalar + component * scalar_dofs;
            const int encoded = vector_dofs[local];
            const int destination = encoded >= 0 ? encoded : -1 - encoded;
            const double sign = encoded >= 0 ? 1.0 : -1.0;
            field_values[destination] = sign * source.interior[
                source_offset + static_cast<std::size_t>(local)];
          }
      }
      for (int rank = 0; rank < ranks; ++rank)
      {
        const auto& source = (*rank_states)[static_cast<std::size_t>(rank)];
        const int expected_elements =
            (global_elements * (rank + 1) + ranks - 1) / ranks -
            (global_elements * rank + ranks - 1) / ranks;
        const std::size_t expected = static_cast<std::size_t>(
            expected_elements * 4 *
            (config.polynomial_order() + 1) *
            (config.polynomial_order() + 1));
        if (source.source_rank != rank || source.rank_count != ranks ||
            source.interior_count != expected)
        {
          detail = "CUDA VTK rank-ordered state census is invalid";
          return false;
        }
      }
    }
    std::ostringstream output;
    output << std::setprecision(17);
    mesh.PrintVTK(output, 1, 1);
    field.SaveVTK(output, "conservative", 1);
    vtk = output.str();
    return output.good();
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return false;
  }
}

bool CandidateVtk(const ValidatedApplicationConfig& config,
                  const cuda::CudaSerializedCheckpointV1& local,
                  const CudaApplicationContextV1& context,
                  std::string& vtk, std::string& detail)
{
  std::vector<cuda::CudaSerializedCheckpointV1> ranks;
  if (!GatherCheckpoints(local, context, ranks, detail)) return false;
  bool built = true;
  if (context.rank() == 0)
    built = MakePhysicalVtk(config, &ranks, vtk, detail);
  return CollectiveAll(built, context.communicator());
}

bool EncodeCheckpoint(const ValidatedApplicationConfig& config,
                      const cuda::CudaSerializedCheckpointV1& local,
                      const cuda::CudaRunEvidence& setup,
                      const PersistentResourceDescriptor& resources,
                      const CudaApplicationContextV1& context,
                      std::vector<std::uint8_t>& encoded,
                      std::string& detail)
{
  std::vector<cuda::CudaSerializedCheckpointV1> gathered;
  if (!GatherCheckpoints(local, context, gathered, detail)) return false;
  bool encoded_ok = true;
  if (context.rank() == 0)
  {
    try
    {
      std::array<CheckpointDescriptor, 2> descriptors{};
      std::array<StagedStateViewV1, 2> states{};
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const auto& source = gathered[static_cast<std::size_t>(rank)];
        descriptors[static_cast<std::size_t>(rank)] = source.descriptor;
        descriptors[static_cast<std::size_t>(rank)].interior_hash.bytes =
            ComputeSha256(source.interior.data(),
                          source.interior_count * sizeof(double));
        descriptors[static_cast<std::size_t>(rank)].trace_hash.bytes =
            ComputeSha256(source.owned_true_trace.data(),
                          source.owned_true_trace_count * sizeof(double));
        states[static_cast<std::size_t>(rank)] = {
            static_cast<std::uint32_t>(rank),
            {source.interior.data(), source.interior_count, 4,
             source.interior_count / 4,
             CoefficientOrderingV1::component_major},
            {source.owned_true_trace.data(), source.owned_true_trace_count, 4,
             source.owned_true_trace_count / 4,
             CoefficientOrderingV1::component_major}};
      }
      const auto captured = CaptureCheckpointPiecesV1(
          descriptors.data(), static_cast<std::size_t>(context.ranks()), config,
          RuntimeIdentity(setup, resources));
      const CheckpointCodecStatus status = captured
          ? EncodeCheckpointV1(*captured, states.data(),
                               static_cast<std::size_t>(context.ranks()), encoded)
          : CheckpointCodecStatus::invalid_descriptor;
      encoded_ok = status == CheckpointCodecStatus::ok;
      if (!encoded_ok && detail.empty())
        detail = "CUDA checkpoint encode failed with status " +
            std::to_string(static_cast<unsigned>(status));
    }
    catch (...) { encoded_ok = false; }
    if (!encoded_ok && detail.empty()) detail = "CUDA checkpoint encode failed";
  }
  return CollectiveAll(encoded_ok, context.communicator());
}

bool StageRootArtifact(DurableGenerationTransactionV1& transaction,
                   const std::string_view logical_name,
                   const std::vector<std::uint8_t>& bytes,
                   const CudaApplicationContextV1& context,
                   std::string& detail)
{
  bool local = true;
  if (context.rank() == 0)
    local = transaction.Stage(logical_name, bytes.data(), bytes.size(), detail) ==
        DurableIoStatus::ok;
  return CollectiveAll(local, context.communicator());
}

bool BeginRootTransaction(DurableGenerationTransactionV1& transaction,
                     const std::filesystem::path& output_directory,
                     const std::uint64_t generation,
                     const CudaApplicationContextV1& context,
                     std::string& detail)
{
  bool local = true;
  if (context.rank() == 0)
    local = transaction.Begin(output_directory, generation, detail) ==
        DurableIoStatus::ok;
  return CollectiveAll(local, context.communicator());
}

bool AddRootAlias(DurableGenerationTransactionV1& transaction,
                  const std::filesystem::path& alias_path,
                  const std::string_view logical_name,
                  const CudaApplicationContextV1& context,
                  std::string& detail)
{
  bool local = true;
  if (context.rank() == 0)
    local = transaction.AddAlias(alias_path, logical_name, detail) ==
        DurableIoStatus::ok;
  return CollectiveAll(local, context.communicator());
}

bool PublishRootTransaction(DurableGenerationTransactionV1& transaction,
                            const CudaApplicationContextV1& context,
                            std::string& detail)
{
  bool local = true;
  if (context.rank() == 0)
  {
    if (transaction.staged())
    {
      const DurableIoStatus status = transaction.Publish(detail);
      local = status == DurableIoStatus::ok;
    }
    else
      local = transaction.Revert(detail) == DurableIoStatus::ok;
  }
  return CollectiveAll(local, context.communicator());
}

bool RevertTransaction(DurableGenerationTransactionV1& transaction,
                       const CudaApplicationContextV1& context,
                       std::string& detail)
{
  bool local = true;
  if (context.rank() == 0)
    local = transaction.Revert(detail) == DurableIoStatus::ok;
  return CollectiveAll(local, context.communicator());
}

template <typename Selection>
bool RestoreCheckpoint(const ValidatedApplicationConfig& config,
                       AppCudaEngine<Selection>& engine,
                       const cuda::CudaRunEvidence& setup,
                       CudaApplicationContextV1& context,
                       cuda::CudaStepEvidence& restored,
                       std::string& detail)
{
  cuda::CudaSerializedCheckpointV1 identity{};
  if (!engine.LocalCheckpointIdentity(identity)) return false;
  const auto* contract = engine.contract();
  if (contract == nullptr) return false;
  ScenarioCoefficientExtentsV1 extents{};
  if (!DeriveScenarioCoefficientExtentsV1(
          config, static_cast<std::uint32_t>(context.ranks()), extents))
    return false;
  const auto& local_extents = extents.ranks[static_cast<std::size_t>(context.rank())];
  std::vector<double> local_interior;
  std::vector<double> local_trace;
  std::array<std::vector<double>, 2> interiors, traces;
  std::array<RankStateBufferV1, 2> buffers{};
  DecodedCheckpointMetadataV1 metadata{};
  bool buffers_ready = true;
  try
  {
    local_interior.resize(static_cast<std::size_t>(
        local_extents.interior_coefficient_count));
    local_trace.resize(static_cast<std::size_t>(
        local_extents.trace_coefficient_count));
  }
  catch (...) { buffers_ready = false; }
  if (!CollectiveAll(buffers_ready, context.communicator()))
  {
    detail = "CUDA restart receive allocation failed";
    return false;
  }
  bool decoded = true;
  if (context.rank() == 0)
  {
    try
    {
      std::vector<std::uint8_t> encoded;
      decoded = detail::ReadCurrentGenerationArtifact(
          std::string{config.restart_from().identity().view()},
          64U * 1024U * 1024U, encoded, detail) == DurableIoStatus::ok;
      for (int rank = 0; decoded && rank < context.ranks(); ++rank)
      {
        const auto& rank_extents =
            extents.ranks[static_cast<std::size_t>(rank)];
        auto& interior = interiors[static_cast<std::size_t>(rank)];
        auto& trace = traces[static_cast<std::size_t>(rank)];
        interior.resize(static_cast<std::size_t>(
            rank_extents.interior_coefficient_count));
        trace.resize(static_cast<std::size_t>(
            rank_extents.trace_coefficient_count));
        buffers[static_cast<std::size_t>(rank)] = {
            static_cast<std::uint32_t>(rank),
            {interior.data(), interior.size(), interior.size(), 4,
             rank_extents.interior_dofs_per_component,
             CoefficientOrderingV1::component_major},
            {trace.data(), trace.size(), trace.size(), 4,
             rank_extents.trace_dofs_per_component,
             CoefficientOrderingV1::component_major}};
      }
      if (decoded)
        decoded = DecodeCheckpointV1(
            {encoded.data(), encoded.size()}, config, contract->resources(),
            RuntimeIdentity(setup, contract->resources()), buffers.data(),
            static_cast<std::size_t>(context.ranks()), metadata) ==
            CheckpointCodecStatus::ok;
    }
    catch (...) { decoded = false; }
    if (!decoded && detail.empty()) detail = "CUDA checkpoint decode rejected";
  }
  if (!CollectiveAll(decoded, context.communicator())) return false;
  const bool descriptor_broadcast = MPI_Bcast(
      &metadata.descriptor, static_cast<int>(sizeof(metadata.descriptor)),
      MPI_BYTE, 0, context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(descriptor_broadcast, context.communicator()))
    return false;
  std::array<int, 2> interior_counts{}, interior_offsets{};
  std::array<int, 2> trace_counts{}, trace_offsets{};
  std::vector<double> flat_interior, flat_trace;
  bool flattened = true;
  if (context.rank() == 0)
  {
    try
    {
      int io = 0;
      int to = 0;
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        if (interiors[index].size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max() - io) ||
            traces[index].size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max() - to))
        {
          flattened = false;
          break;
        }
        interior_counts[index] = static_cast<int>(interiors[index].size());
        trace_counts[index] = static_cast<int>(traces[index].size());
        interior_offsets[index] = io;
        trace_offsets[index] = to;
        io += interior_counts[index];
        to += trace_counts[index];
        flat_interior.insert(flat_interior.end(), interiors[index].begin(),
                             interiors[index].end());
        flat_trace.insert(flat_trace.end(), traces[index].begin(),
                          traces[index].end());
      }
    }
    catch (...) { flattened = false; }
  }
  if (!CollectiveAll(flattened, context.communicator()))
  {
    detail = "CUDA restart scatter staging failed";
    return false;
  }
  const int interior_scatter = MPI_Scatterv(
      flat_interior.data(), interior_counts.data(), interior_offsets.data(),
      MPI_DOUBLE, local_interior.data(), static_cast<int>(local_interior.size()),
      MPI_DOUBLE, 0, context.communicator());
  const int trace_scatter = MPI_Scatterv(
      flat_trace.data(), trace_counts.data(), trace_offsets.data(), MPI_DOUBLE,
      local_trace.data(), static_cast<int>(local_trace.size()), MPI_DOUBLE, 0,
      context.communicator());
  if (!CollectiveAll(interior_scatter == MPI_SUCCESS &&
                         trace_scatter == MPI_SUCCESS,
                     context.communicator()))
  {
    detail = "CUDA restart scatter failed";
    return false;
  }
  cuda::CudaSerializedCheckpointV1 checkpoint{};
  checkpoint.source_rank = context.rank();
  checkpoint.rank_count = context.ranks();
  checkpoint.attempt.sequence = metadata.descriptor.coordinates.cursor;
  checkpoint.attempt.committed_step = metadata.descriptor.coordinates.step - 1;
  checkpoint.attempt.committed_generation =
      metadata.descriptor.coordinates.generation - 1;
  checkpoint.descriptor = metadata.descriptor;
  checkpoint.local_mesh_hash = identity.local_mesh_hash;
  checkpoint.local_partition_hash = identity.local_partition_hash;
  checkpoint.local_topology_hash = identity.local_topology_hash;
  checkpoint.interior_count = local_interior.size();
  checkpoint.owned_true_trace_count = local_trace.size();
  std::copy(local_interior.begin(), local_interior.end(),
            checkpoint.interior.begin());
  std::copy(local_trace.begin(), local_trace.end(),
            checkpoint.owned_true_trace.begin());
  checkpoint.local_interior_hash = cuda::detail::HashHostBytes(
      checkpoint.interior.data(), local_interior.size() * sizeof(double));
  checkpoint.local_trace_hash = cuda::detail::HashHostBytes(
      checkpoint.owned_true_trace.data(), local_trace.size() * sizeof(double));
  std::array<unsigned char, 2 * 32> rank_hashes{};
  bool aggregate_ready = MPI_Allgather(
      checkpoint.local_interior_hash.bytes.data(), 32, MPI_UNSIGNED_CHAR,
      rank_hashes.data(), 32, MPI_UNSIGNED_CHAR,
      context.communicator()) == MPI_SUCCESS;
  if (aggregate_ready)
    checkpoint.descriptor.interior_hash = cuda::detail::HashHostBytes(
        rank_hashes.data(), static_cast<std::size_t>(context.ranks()) * 32);
  const bool trace_aggregate_ready = MPI_Allgather(
      checkpoint.local_trace_hash.bytes.data(), 32, MPI_UNSIGNED_CHAR,
      rank_hashes.data(), 32, MPI_UNSIGNED_CHAR,
      context.communicator()) == MPI_SUCCESS;
  aggregate_ready = aggregate_ready && trace_aggregate_ready;
  if (trace_aggregate_ready)
    checkpoint.descriptor.trace_hash = cuda::detail::HashHostBytes(
        rank_hashes.data(), static_cast<std::size_t>(context.ranks()) * 32);
  if (!CollectiveAll(aggregate_ready, context.communicator()))
  {
    detail = "CUDA contract checkpoint hashes could not be reconstructed";
    return false;
  }
  checkpoint.candidate_frozen = true;
  checkpoint.checkpoint_staged = true;
  checkpoint.output_staged = true;
  checkpoint.successful_commit = true;
  return engine.Restore(checkpoint, restored);
}

std::optional<ValidatedApplicationConfig> CheckpointWriterConfig(
    const ApplicationOptionsView& request,
    const ValidatedApplicationConfig& config)
{
  if (!config.checkpoint_out().present()) return std::nullopt;
  if (config.run_role() == ApplicationRunRoleV1::split_writer)
    return config;
  ApplicationOptionsView writer = request;
  writer.run_role = ApplicationRunRoleV1::split_writer;
  writer.checkpoint_step = OptionalCheckpointStepV1::Present(
      config.scenario_checkpoint_step());
  writer.stop_after_checkpoint = StopAfterCheckpointV1::yes;
  writer.restart_from = {};
  writer.injection = {};
  writer.retry_limit = 0;
  const ApplicationValidation validation = ValidateApplicationOptions(writer);
  return validation.accepted()
      ? std::optional<ValidatedApplicationConfig>{*validation.config()}
      : std::nullopt;
}

std::string Metadata(const ValidatedApplicationConfig& config,
                     const ApplicationRunResult& result,
                     const cuda::CudaRunEvidence& setup,
                     const StateHash& final_interior,
                     const StateHash& final_trace,
                     const StateHash& global_state,
                     const std::string& detail)
{
  std::ostringstream output;
  output << std::setprecision(17)
      << "{\n  \"schema\": \"dg-igr-hdg-run-metadata.v1\",\n"
      << "  \"card\": \"" << config.card_id().view() << "\",\n"
      << "  \"case\": \"" << config.case_id().view() << "\",\n"
      << "  \"backend\": \"" << config.tuple().backend.view() << "\",\n"
      << "  \"gpu_uuid\": \"" << GpuUuidText(setup) << "\",\n"
      << "  \"committed_step\": " << result.coordinates.step << ",\n"
      << "  \"committed_time\": " << result.coordinates.time << ",\n"
      << "  \"status\": \"" << ToString(result.status) << "\",\n"
      << "  \"reason\": \"" << ToString(result.reason) << "\",\n"
      << "  \"retries\": " << result.retries << ",\n"
      << "  \"final_interior_sha256\": \"" << Hex(final_interior)
      << "\",\n"
      << "  \"final_trace_sha256\": \"" << Hex(final_trace) << "\",\n"
      << "  \"global_state_sha256\": \"" << Hex(global_state) << "\",\n"
      << "  \"diagnostic\": \"" << detail << "\"\n}\n";
  return output.str();
}

template <typename Selection>
ApplicationRunResult RunCuda(const ApplicationOptionsView& request,
                             CudaApplicationContextV1& context)
{
  ApplicationRunResult result{};
  const ApplicationValidation validation = ValidateApplicationOptions(request);
  if (!validation.accepted()) return result;
  const ValidatedApplicationConfig& config = *validation.config();
  if (context.ranks() !=
      static_cast<int>(config.deferred_runtime().required_mpi_rank_count))
  {
    context.SetDiagnostic("MPI rank count does not match CUDA backend");
    return result;
  }
  std::string detail;
  const std::filesystem::path output_directory{
      std::string{config.output_directory().identity().view()}};
  bool output_ready = true;
  bool output_directory_created = false;
  if (context.rank() == 0)
  {
    if (config.run_role() == ApplicationRunRoleV1::split_reader)
    {
      std::error_code error;
      const std::filesystem::file_status status =
          std::filesystem::symlink_status(output_directory, error);
      output_ready = !error && std::filesystem::is_directory(status);
      if (!output_ready) detail = "split reader output directory is unavailable";
    }
    else
    {
      output_ready = CreateDurableDirectory(output_directory, detail) ==
          DurableIoStatus::ok;
      output_directory_created = output_ready;
    }
  }
  if (!CollectiveAll(output_ready, context.communicator()))
  {
    if (context.rank() == 0 && output_directory_created)
      (void)DiscardDurableDirectoryIfEmpty(output_directory, detail);
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    context.SetDiagnostic(detail);
    return result;
  }

  cuda::CudaRunEvidence setup{};
  try
  {
    mfem::Device device("cuda");
    const cuda::CudaConfig numerical = MakeCudaConfig(config);
    auto engine = std::make_unique<AppCudaEngine<Selection>>(numerical);
    if (!engine->Initialize(setup))
    {
      result.status = RunStatus::terminal_failure;
      result.reason = setup.setup_origin;
      result.failure_class = ApplicationRunResult::FailureClass::backend;
      detail = "CUDA backend initialization failed";
    }
    if (result.failure_class == ApplicationRunResult::FailureClass::configuration)
    {
      result.failure_class = ApplicationRunResult::FailureClass::none;
      result.reason = FailureReason::none;
    }
    const std::optional<ValidatedApplicationConfig> checkpoint_writer =
        CheckpointWriterConfig(request, config);
    if (config.checkpoint_out().present() && !checkpoint_writer)
    {
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = ApplicationRunResult::FailureClass::configuration;
      detail = "checkpoint writer role normalization failed";
    }
    if (result.failure_class == ApplicationRunResult::FailureClass::none &&
        config.run_role() == ApplicationRunRoleV1::split_reader)
    {
      cuda::CudaStepEvidence restored{};
      if (!RestoreCheckpoint<Selection>(config, *engine, setup,
                                        context, restored, detail))
      {
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::restart_incompatible;
        result.failure_class =
            ApplicationRunResult::FailureClass::checkpoint_data;
      }
    }
    if (result.failure_class == ApplicationRunResult::FailureClass::none &&
        engine->contract() != nullptr &&
        engine->contract()->coordinates().step == 0 && OutputStep(config, 0))
    {
      DurableGenerationTransactionV1 initial_transaction;
      std::vector<std::uint8_t> initial_bytes;
      bool initial_ready = true;
      if (context.rank() == 0)
      {
        try
        {
          std::string initial;
          initial_ready = MakePhysicalVtk(config, nullptr, initial, detail);
          if (initial_ready)
            initial_bytes.assign(initial.begin(), initial.end());
        }
        catch (...) { initial_ready = false; }
      }
      initial_ready = CollectiveAll(initial_ready, context.communicator()) &&
          BeginRootTransaction(initial_transaction, output_directory, 0,
                               context, detail) &&
          StageRootArtifact(initial_transaction, StepName(0), initial_bytes,
                            context, detail) &&
          PublishRootTransaction(initial_transaction, context, detail);
      if (!initial_ready)
      {
        (void)RevertTransaction(initial_transaction, context, detail);
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = ApplicationRunResult::FailureClass::filesystem;
      }
    }

    const std::uint64_t stop_step =
        config.run_role() == ApplicationRunRoleV1::split_writer
        ? config.scenario_checkpoint_step() : config.accepted_steps();
    bool retry = false;
    while (result.failure_class == ApplicationRunResult::FailureClass::none &&
           engine->contract() != nullptr &&
           engine->contract()->coordinates().step < stop_step)
    {
      const std::uint64_t target = engine->contract()->coordinates().step + 1;
      cuda::CudaPrecommitSnapshot snapshot{};
      cuda::CudaStepEvidence evidence{};
      cuda::CudaSerializedCheckpointV1 checkpoint{};
      const bool was_retry = retry;
      const bool prepared = engine->PrepareStep(retry, snapshot, evidence);
      retry = false;
      if (!prepared)
      {
        const bool numerical_injection_fired =
            config.injection().mode == FailureInjectionModeV1::prescribed &&
            config.injection().target_step == target &&
            evidence.phase == ToCudaPhase(config.injection().phase) &&
            evidence.source_rank ==
                static_cast<int>(config.injection().source_rank);
        if (evidence.retry_available &&
            result.retries < config.retry_limit())
        {
          if (numerical_injection_fired)
            AppendCudaFailureProvenance(evidence, detail);
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        result.failure_class = numerical_injection_fired
            ? ApplicationRunResult::FailureClass::injected
            : (ScientificFailureReason(result.reason)
                   ? ApplicationRunResult::FailureClass::scientific
                   : ApplicationRunResult::FailureClass::backend);
        detail = "CUDA candidate preparation failed";
        break;
      }
      const bool inject_now = !was_retry &&
          config.injection().mode == FailureInjectionModeV1::prescribed &&
          config.injection().target_step == target;
      const FailurePhase injection_phase = inject_now
          ? config.injection().phase : FailurePhase::Setup;
      DurableGenerationTransactionV1 transaction;
      std::vector<std::uint8_t> vtk_bytes;
      std::vector<std::uint8_t> checkpoint_bytes;
      std::vector<std::uint8_t> metadata_bytes;
      bool staged = BeginRootTransaction(
          transaction, output_directory, target, context, detail) &&
          injection_phase != FailurePhase::CheckpointStaging &&
          engine->StagePreparedCheckpoint(checkpoint, evidence);
      std::string vtk;
      if (staged && OutputStep(config, target))
        staged = CandidateVtk(config, checkpoint, context, vtk, detail);
      if (staged && context.rank() == 0 && !vtk.empty())
      {
        try { vtk_bytes.assign(vtk.begin(), vtk.end()); }
        catch (...) { staged = false; }
      }
      staged = CollectiveAll(staged, context.communicator());
      if (staged && OutputStep(config, target))
        staged = StageRootArtifact(transaction, StepName(target), vtk_bytes,
                                   context, detail);
      if (staged && config.checkpoint_out().present() &&
          target == config.scenario_checkpoint_step())
      {
        staged = checkpoint_writer && EncodeCheckpoint(
            *checkpoint_writer, checkpoint, setup,
            engine->contract()->resources(), context, checkpoint_bytes,
            detail) && StageRootArtifact(
                transaction, "restart-checkpoint.v2", checkpoint_bytes,
                context, detail) && AddRootAlias(
                    transaction,
                    std::string{config.checkpoint_out().identity().view()},
                    "restart-checkpoint.v2", context, detail);
      }
      if (injection_phase == FailurePhase::OutputStaging) staged = false;
      if (staged) staged = engine->MarkOutputStaged(checkpoint);
      if (!staged)
      {
        const bool cleaned = RevertTransaction(transaction, context, detail);
        const FailureReason abort_reason =
            evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        const bool retry_available = cleaned && engine->AbortPrepared(
            checkpoint, abort_reason, evidence);
        const bool numerical_injection_fired = inject_now &&
            evidence.phase == ToCudaPhase(config.injection().phase) &&
            evidence.source_rank ==
                static_cast<int>(config.injection().source_rank);
        const bool outer_stage_injection_fired = inject_now &&
            (injection_phase == FailurePhase::CheckpointStaging ||
             injection_phase == FailurePhase::OutputStaging);
        if (retry_available && result.retries < config.retry_limit())
        {
          if (numerical_injection_fired)
            AppendCudaFailureProvenance(evidence, detail);
          else if (outer_stage_injection_fired)
            AppendApplicationFailureProvenance(
                injection_phase,
                static_cast<int>(config.injection().source_rank), detail);
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        result.failure_class = numerical_injection_fired ||
                outer_stage_injection_fired
            ? ApplicationRunResult::FailureClass::injected
            : (cleaned ? (ScientificFailureReason(result.reason)
                              ? ApplicationRunResult::FailureClass::scientific
                              : ApplicationRunResult::FailureClass::backend)
                       : ApplicationRunResult::FailureClass::filesystem);
        if (detail.empty()) detail = "CUDA attempt rolled back before commit";
        break;
      }
      if (!engine->PrepareCommit(checkpoint, evidence))
      {
        const bool cleaned = RevertTransaction(transaction, context, detail);
        if (cleaned && evidence.retry_available &&
            result.retries < config.retry_limit())
        {
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        result.failure_class = cleaned
            ? (ScientificFailureReason(result.reason)
                   ? ApplicationRunResult::FailureClass::scientific
                   : ApplicationRunResult::FailureClass::backend)
            : ApplicationRunResult::FailureClass::filesystem;
        detail = "CUDA prepare-commit failed";
        break;
      }

      const bool terminal_target = target == stop_step;
      if (terminal_target)
      {
        ApplicationRunResult anticipated = result;
        anticipated.status = RunStatus::committed;
        anticipated.reason = FailureReason::none;
        anticipated.failure_class = ApplicationRunResult::FailureClass::none;
        anticipated.coordinates = snapshot.target;
        StateHash global_state{};
        bool metadata_ready = GlobalCudaStateIdentity(
            checkpoint, context, global_state, detail);
        if (context.rank() == 0)
        {
          try
          {
            const std::string metadata = Metadata(
                config, anticipated, setup, snapshot.candidate_interior_hash,
                snapshot.candidate_trace_hash, global_state, detail);
            metadata_bytes.assign(metadata.begin(), metadata.end());
          }
          catch (...) { metadata_ready = false; }
        }
        metadata_ready = CollectiveAll(metadata_ready,
                                       context.communicator()) &&
            StageRootArtifact(transaction,
                          config.run_role() ==
                                  ApplicationRunRoleV1::split_writer
                              ? "run-metadata-checkpoint.json"
                              : "run-metadata.json",
                          metadata_bytes, context, detail);
        if (!metadata_ready)
        {
          const bool cleaned = RevertTransaction(transaction, context, detail);
          const bool retry_available = cleaned && engine->AbortPrepared(
              checkpoint, FailureReason::backend_failure, evidence);
          if (retry_available && result.retries < config.retry_limit())
          {
            ++result.retries;
            retry = true;
            continue;
          }
          result.status = RunStatus::terminal_failure;
          result.reason = FailureReason::backend_failure;
          result.failure_class = cleaned
              ? ApplicationRunResult::FailureClass::backend
              : ApplicationRunResult::FailureClass::filesystem;
          detail = "CUDA terminal metadata staging failed";
          break;
        }
      }

      if (injection_phase == FailurePhase::Commit ||
          injection_phase == FailurePhase::TerminalEvidence)
      {
        const bool cleaned = RevertTransaction(transaction, context, detail);
        const bool retry_available = cleaned && engine->AbortPrepared(
            checkpoint, FailureReason::backend_failure, evidence);
        if (retry_available && result.retries < config.retry_limit())
        {
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = ApplicationRunResult::FailureClass::injected;
        detail = "CUDA commit-stage injection did not retry";
        break;
      }

      if (!engine->CommitPrepared(checkpoint, evidence))
      {
        const bool cleaned = RevertTransaction(transaction, context, detail);
        if (cleaned && evidence.retry_available &&
            result.retries < config.retry_limit())
        {
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        result.failure_class = cleaned
            ? (ScientificFailureReason(result.reason)
                   ? ApplicationRunResult::FailureClass::scientific
                   : ApplicationRunResult::FailureClass::backend)
            : ApplicationRunResult::FailureClass::filesystem;
        detail = "CUDA provisional commit failed";
        break;
      }

      if (injection_phase == FailurePhase::RollbackAudit)
      {
        const bool cleaned = RevertTransaction(transaction, context, detail);
        const bool retry_available = engine->RevertCommit(
            checkpoint, FailureReason::backend_failure, evidence);
        if (cleaned && retry_available &&
            result.retries < config.retry_limit())
        {
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = ApplicationRunResult::FailureClass::injected;
        detail = "CUDA rollback-audit injection did not retry";
        break;
      }

      const bool published = PublishRootTransaction(
          transaction, context, detail);
      if (!published)
      {
        const bool cleaned = RevertTransaction(
            transaction, context, detail);
        const bool retry_available = engine->RevertCommit(
            checkpoint, FailureReason::backend_failure, evidence);
        if (cleaned && retry_available &&
            result.retries < config.retry_limit())
        {
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = ApplicationRunResult::FailureClass::filesystem;
        detail = "CUDA artifact publication failed";
        break;
      }
      int marker_source_rank = -1;
      int finalize_source_rank = -1;
      const bool marker_acknowledged = CollectiveAll(
          !ConsumePostPublicationFault(
              PostPublicationFault::after_marker, target, context.rank(),
              marker_source_rank), context.communicator());
      const bool finalize_fault = marker_acknowledged &&
          ConsumePostPublicationFault(
              PostPublicationFault::before_finalize, target, context.rank(),
              finalize_source_rank);
      if (finalize_fault) checkpoint.descriptor.trace_hash.bytes[0] ^= 1U;
      const bool finalize_ready = marker_acknowledged;
      if (!finalize_ready)
      {
        const bool cleaned = RevertTransaction(
            transaction, context, detail);
        const bool retry_available = engine->RevertCommit(
            checkpoint, FailureReason::backend_failure, evidence);
        const bool private_retry = cleaned &&
            PostPublicationFaultAllowsRetry() && result.retries == 0;
        if (cleaned && retry_available &&
            (result.retries < config.retry_limit() || private_retry))
        {
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = cleaned
            ? ApplicationRunResult::FailureClass::backend
            : ApplicationRunResult::FailureClass::filesystem;
        const int source_rank = marker_acknowledged
            ? finalize_source_rank : marker_source_rank;
        detail = (marker_acknowledged
            ? "CUDA finalization readiness failure did not retry"
            : "CUDA post-publication acknowledgement failure did not retry") +
            std::string{" source_rank="} + std::to_string(source_rank);
        break;
      }
      const bool finalize_resolved =
          engine->FinalizeCommit(checkpoint, evidence);
      if (!finalize_resolved || !evidence.committed)
      {
        const bool cleaned = RevertTransaction(
            transaction, context, detail);
        const bool retry_available = evidence.retry_available ||
            engine->RevertCommit(
                checkpoint, FailureReason::backend_failure, evidence);
        const bool private_retry = cleaned && finalize_source_rank >= 0 &&
            PostPublicationFaultAllowsRetry() && result.retries == 0;
        if (cleaned && retry_available &&
            (result.retries < config.retry_limit() || private_retry))
        {
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        result.failure_class = cleaned
            ? (ScientificFailureReason(result.reason)
                   ? ApplicationRunResult::FailureClass::scientific
                   : ApplicationRunResult::FailureClass::backend)
            : ApplicationRunResult::FailureClass::filesystem;
        detail = "CUDA logical finalization failed";
        break;
      }
    }
    if (engine->contract() != nullptr)
      result.coordinates = engine->contract()->coordinates();
    if (result.failure_class == ApplicationRunResult::FailureClass::none)
    {
      result.status = RunStatus::committed;
      result.reason = FailureReason::none;
    }
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
  }
  bool directories_clean = true;
  if (context.rank() == 0 && output_directory_created)
    directories_clean = DiscardDurableDirectoryIfEmpty(
                            output_directory, detail) == DurableIoStatus::ok;
  if (!CollectiveAll(directories_clean, context.communicator()))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::filesystem;
  }
  context.SetDiagnostic(detail);
  return result;
}

#endif

#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
using ScalarOwner = scalar::PersistentSc25ScalarIgrCudaRank2;
inline constexpr bool scalar_rank2 = true;
#else
using ScalarOwner = scalar::PersistentSc25ScalarIgrCudaRank1;
inline constexpr bool scalar_rank2 = false;
#endif
using ScalarCheckpoint = scalar::ScalarCudaCheckpoint;
using ScalarPublishedState = scalar::PublishedState;

static_assert(!std::is_polymorphic<ScalarOwner>::value,
              "the scalar-IGR CUDA application adapter is static");

struct ScalarPackedState final
{
  StateCoordinates coordinates{};
  std::vector<double> interior;
  std::vector<double> trace;
};

struct ScalarResourceRecord final
{
  StateHash mesh{};
  StateHash partition{};
  StateHash topology{};
  std::array<std::uint8_t, 16> gpu_uuid{};
  CheckpointDescriptor private_descriptor{};
  scalar::StateHashes raw_field_hashes{};
  StateHash shard_manifest_hash{};
  StateHash canonical_manifest_hash{};
};

struct ScalarResourceSidecar final
{
  std::array<char, 16> magic{{'D','G','I','G','R','-','S','C','2','5','-','R','E','S','1',0}};
  std::uint32_t version = 1;
  std::uint32_t rank_count = 0;
  std::array<ScalarResourceRecord, 2> ranks{};
  StateHash checksum{};
};

static_assert(std::is_trivially_copyable<ScalarResourceSidecar>::value,
              "the scalar resource sidecar is fixed-layout data");

bool ScalarCollectiveAll(const bool local,
                         const CudaApplicationContextV1& context) noexcept
{
  const int value = local ? 1 : 0;
  int result = 0;
  return MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN,
                       context.communicator()) == MPI_SUCCESS && result == 1;
}

bool ScalarAccepted(const scalar::ScalarCudaStatus status) noexcept
{
  return status == scalar::ScalarCudaStatus::accepted ||
      status == scalar::ScalarCudaStatus::accepted_after_one_retry;
}

bool ScalarRestored(const scalar::ScalarCudaStatus status) noexcept
{
  return status == scalar::ScalarCudaStatus::ready;
}

std::string ScalarStepName(const std::uint64_t step)
{
  std::ostringstream output;
  output << "state-step-" << std::setw(6) << std::setfill('0') << step
         << ".vtk";
  return output.str();
}

bool ScalarOutputStep(const ValidatedApplicationConfig& config,
                      const std::uint64_t step) noexcept
{
  return std::find(config.output_steps().begin(), config.output_steps().end(),
                   step) != config.output_steps().end();
}

template <typename Range>
void ScalarAppend(const Range& source, std::vector<double>& destination)
{
  destination.insert(destination.end(), source.begin(), source.end());
}

bool PackScalarState(const ScalarPublishedState& state,
                     const StateCoordinates& coordinates,
                     ScalarPackedState& output) noexcept
{
  try
  {
    ScalarPackedState candidate;
    candidate.coordinates = coordinates;
    candidate.interior.reserve(state.u.size() + state.sigma.size() +
                               state.z.size());
    candidate.trace.reserve(state.uhat.size() + state.sigmahat.size());
    ScalarAppend(state.u, candidate.interior);
    ScalarAppend(state.sigma, candidate.interior);
    ScalarAppend(state.z, candidate.interior);
    ScalarAppend(state.uhat, candidate.trace);
    ScalarAppend(state.sigmahat, candidate.trace);
    const std::size_t expected_interior = scalar_rank2
        ? static_cast<std::size_t>(scalar::persistent_interior_unknowns)
        : static_cast<std::size_t>(scalar::persistent_interior_unknowns);
    if (candidate.interior.size() != expected_interior ||
        candidate.trace.empty() || candidate.trace.size() % 5 != 0)
      return false;
    output = std::move(candidate);
    return true;
  }
  catch (...) { return false; }
}

bool PrepareScalarCheckpoint(const ScalarPublishedState& state,
                             ScalarCheckpoint& checkpoint) noexcept
{
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  try
  {
    checkpoint.interior_component_slabs.resize(
        scalar::persistent_interior_unknowns);
    checkpoint.owned_trace_component_slabs.resize(
        state.uhat.size() + state.sigmahat.size());
  }
  catch (...) { return false; }
  return checkpoint.interior_component_slabs.size() ==
          scalar::persistent_interior_unknowns &&
      checkpoint.owned_trace_component_slabs.size() ==
          state.uhat.size() + state.sigmahat.size();
#else
  (void)state;
  (void)checkpoint;
  return true;
#endif
}

bool PackedFromCheckpoint(const ScalarCheckpoint& checkpoint,
                          ScalarPackedState& output) noexcept
{
  try
  {
    std::vector<double> component_major(
        checkpoint.interior_component_slabs.begin(),
        checkpoint.interior_component_slabs.end());
    ScalarPackedState candidate;
    candidate.coordinates = checkpoint.descriptor.coordinates;
    if (!detail::ScalarIgrComponentMajorToFieldBlocksV1(
            component_major, scalar::volume_nodes, candidate.interior))
      return false;
    candidate.trace.assign(
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
        checkpoint.owned_trace_component_slabs.begin(),
        checkpoint.owned_trace_component_slabs.end());
#else
        checkpoint.trace_component_slabs.begin(),
        checkpoint.trace_component_slabs.end());
#endif
    output = std::move(candidate);
    return true;
  }
  catch (...) { return false; }
}

StateHash ScalarHashParts(
    const std::initializer_list<std::pair<const double*, std::size_t>>& parts)
    noexcept
{
  Sha256 hash;
  Sha256Digest digest{};
  for (const auto& part : parts)
    if ((part.second != 0 && part.first == nullptr) ||
        !hash.Update(part.first, part.second * sizeof(double)))
      return {};
  return hash.Finalize(digest) ? StateHash{digest} : StateHash{};
}

scalar::StateHashes ScalarStateHashes(const ScalarPackedState& state) noexcept
{
  scalar::StateHashes hashes{};
  const std::size_t scalar_interior = state.interior.size() / 7;
  const std::size_t scalar_trace = state.trace.size() / 5;
  if (scalar_interior == 0 || scalar_trace == 0 ||
      state.interior.size() != 7 * scalar_interior ||
      state.trace.size() != 5 * scalar_trace)
    return {};
  const double* const u = state.interior.data();
  const double* const sigma = u + 4 * scalar_interior;
  const double* const z = sigma + scalar_interior;
  const double* const uhat = state.trace.data();
  const double* const sigmahat = uhat + 4 * scalar_trace;
  hashes.u = ScalarHashParts({{u, 4 * scalar_interior}});
  hashes.sigma = ScalarHashParts({{sigma, scalar_interior}});
  hashes.z = ScalarHashParts({{z, 2 * scalar_interior}});
  hashes.uhat = ScalarHashParts({{uhat, 4 * scalar_trace}});
  hashes.sigmahat = ScalarHashParts({{sigmahat, scalar_trace}});
  hashes.interior = ScalarHashParts({{u, 4 * scalar_interior},
                                     {sigma, scalar_interior},
                                     {z, 2 * scalar_interior}});
  hashes.trace = ScalarHashParts({{uhat, 4 * scalar_trace},
                                  {sigmahat, scalar_trace}});
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  hashes.shard = ScalarHashParts({{u, 4 * scalar_interior},
                                  {sigma, scalar_interior},
                                  {z, 2 * scalar_interior},
                                  {uhat, 4 * scalar_trace},
                                  {sigmahat, scalar_trace}});
#endif
  return hashes;
}

StateHash ScalarCheckpointManifest(const ScalarCheckpoint& checkpoint) noexcept
{
  Sha256 hash;
  Sha256Digest digest{};
  const auto text = [&hash](const std::string_view value) {
    const std::uint64_t size = value.size();
    return hash.Update(&size, sizeof(size)) &&
        hash.Update(value.data(), value.size());
  };
  const CheckpointDescriptor& descriptor = checkpoint.descriptor;
  bool valid = text(checkpoint.projection.view()) &&
      text(checkpoint.tuple.view()) && text(checkpoint.scenario.view()) &&
      text(checkpoint.layout.view());
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  valid = valid && hash.Update(&checkpoint.shard_rank,
                               sizeof(checkpoint.shard_rank)) &&
      hash.Update(&checkpoint.owned_scalar_trace_nodes,
                  sizeof(checkpoint.owned_scalar_trace_nodes));
#endif
  valid = valid &&
      hash.Update(&descriptor.schema_version, sizeof(descriptor.schema_version)) &&
      hash.Update(&descriptor.successful_commit, sizeof(descriptor.successful_commit)) &&
      hash.Update(&descriptor.restart_mode, sizeof(descriptor.restart_mode)) &&
      hash.Update(&descriptor.tuple, sizeof(descriptor.tuple)) &&
      hash.Update(&descriptor.mfem_pin, sizeof(descriptor.mfem_pin)) &&
      hash.Update(descriptor.mesh_hash.bytes.data(), descriptor.mesh_hash.bytes.size()) &&
      hash.Update(descriptor.partition_hash.bytes.data(), descriptor.partition_hash.bytes.size()) &&
      hash.Update(descriptor.topology_hash.bytes.data(), descriptor.topology_hash.bytes.size()) &&
      hash.Update(&descriptor.rank_count, sizeof(descriptor.rank_count)) &&
      hash.Update(&descriptor.polynomial_order, sizeof(descriptor.polynomial_order)) &&
      hash.Update(&descriptor.layout, sizeof(descriptor.layout)) &&
      hash.Update(&descriptor.precision, sizeof(descriptor.precision)) &&
      hash.Update(&descriptor.cursor, sizeof(descriptor.cursor)) &&
      hash.Update(&descriptor.dt, sizeof(descriptor.dt)) &&
      hash.Update(&descriptor.coordinates.time, sizeof(descriptor.coordinates.time)) &&
      hash.Update(&descriptor.coordinates.step, sizeof(descriptor.coordinates.step)) &&
      hash.Update(&descriptor.coordinates.generation, sizeof(descriptor.coordinates.generation)) &&
      hash.Update(&descriptor.coordinates.cursor, sizeof(descriptor.coordinates.cursor)) &&
      hash.Update(&descriptor.coordinates.dt, sizeof(descriptor.coordinates.dt)) &&
      hash.Update(&descriptor.coordinate_fingerprint, sizeof(descriptor.coordinate_fingerprint)) &&
      hash.Update(descriptor.interior_hash.bytes.data(), descriptor.interior_hash.bytes.size()) &&
      hash.Update(descriptor.trace_hash.bytes.data(), descriptor.trace_hash.bytes.size()) &&
      hash.Update(checkpoint.raw_field_hashes.u.bytes.data(), 32) &&
      hash.Update(checkpoint.raw_field_hashes.sigma.bytes.data(), 32) &&
      hash.Update(checkpoint.raw_field_hashes.z.bytes.data(), 32) &&
      hash.Update(checkpoint.raw_field_hashes.uhat.bytes.data(), 32) &&
      hash.Update(checkpoint.raw_field_hashes.sigmahat.bytes.data(), 32);
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  valid = valid && hash.Update(checkpoint.raw_field_hashes.shard.bytes.data(), 32);
#endif
  valid = valid && hash.Update(
      checkpoint.interior_component_slabs.data(),
      checkpoint.interior_component_slabs.size() * sizeof(double));
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  valid = valid && hash.Update(
      checkpoint.owned_trace_component_slabs.data(),
      checkpoint.owned_trace_component_slabs.size() * sizeof(double));
#else
  valid = valid && hash.Update(
      checkpoint.trace_component_slabs.data(),
      checkpoint.trace_component_slabs.size() * sizeof(double));
#endif
  return valid && hash.Finalize(digest) ? StateHash{digest} : StateHash{};
}

bool SealScalarCheckpoint(ScalarCheckpoint& checkpoint,
                          const ScalarPackedState& state,
                          const ScalarResourceRecord& resources,
                          const CudaApplicationContextV1& context) noexcept
{
  bool local_ready = true;
  try
  {
    std::vector<double> component_major;
    local_ready = detail::ScalarIgrFieldBlocksToComponentMajorV1(
        state.interior, scalar::volume_nodes, component_major);
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
    if (local_ready)
      checkpoint.interior_component_slabs.resize(component_major.size());
#endif
    local_ready = local_ready &&
        component_major.size() == checkpoint.interior_component_slabs.size();
    if (local_ready)
      std::copy(component_major.begin(), component_major.end(),
                checkpoint.interior_component_slabs.begin());
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
    if (local_ready)
    {
      checkpoint.owned_trace_component_slabs = state.trace;
      checkpoint.shard_rank = context.rank();
      checkpoint.owned_scalar_trace_nodes =
          static_cast<int>(state.trace.size() / scalar::trace_components);
    }
#else
    local_ready = local_ready &&
        state.trace.size() == checkpoint.trace_component_slabs.size();
    if (local_ready)
      std::copy(state.trace.begin(), state.trace.end(),
                checkpoint.trace_component_slabs.begin());
#endif
    if (local_ready)
      checkpoint.raw_field_hashes = ScalarStateHashes(state);
    checkpoint.descriptor.schema_version = checkpoint_schema_version;
    checkpoint.descriptor.successful_commit = true;
    checkpoint.descriptor.restart_mode =
        RestartMode::same_backend_rank_partition;
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
    checkpoint.descriptor.tuple = scalar::CudaRank2Selection::Descriptor();
#else
    checkpoint.descriptor.tuple = scalar::CudaRank1Selection::Descriptor();
#endif
    checkpoint.descriptor.mfem_pin = StableIdentity::Copy(scalar::mfem_pin);
    checkpoint.descriptor.mesh_hash = resources.mesh;
    checkpoint.descriptor.partition_hash = resources.partition;
    checkpoint.descriptor.topology_hash = resources.topology;
    checkpoint.descriptor.rank_count = scalar_rank2 ? 2U : 1U;
    checkpoint.descriptor.polynomial_order = 1;
    checkpoint.descriptor.layout = StableIdentity::Copy(scalar::interior_layout_id);
    checkpoint.descriptor.precision = StableIdentity::Copy("Fp64");
    checkpoint.descriptor.cursor = state.coordinates.cursor;
    checkpoint.descriptor.dt = state.coordinates.dt;
    checkpoint.descriptor.coordinates = state.coordinates;
    checkpoint.descriptor.coordinate_fingerprint =
        ::dg::hdg::runtime::detail::CoordinateFingerprint(state.coordinates);
    checkpoint.descriptor.interior_hash =
        checkpoint.raw_field_hashes.interior;
    checkpoint.descriptor.trace_hash = checkpoint.raw_field_hashes.trace;
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
    if (local_ready)
      checkpoint.shard_manifest_hash = ScalarCheckpointManifest(checkpoint);
    local_ready = local_ready && checkpoint.shard_manifest_hash != StateHash{};
#else
    if (local_ready)
      checkpoint.canonical_manifest_hash = ScalarCheckpointManifest(checkpoint);
    local_ready = local_ready &&
        checkpoint.canonical_manifest_hash != StateHash{};
#endif
  }
  catch (...) { local_ready = false; }
  if (!ScalarCollectiveAll(local_ready, context)) return false;
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  std::array<StateHash, 2> rank_ordered{};
  const bool gathered = MPI_Allgather(
      checkpoint.shard_manifest_hash.bytes.data(), 32, MPI_UINT8_T,
      rank_ordered.data(), 32, MPI_UINT8_T,
      context.communicator()) == MPI_SUCCESS;
  if (!ScalarCollectiveAll(gathered, context)) return false;
  Sha256 combined;
  Sha256Digest combined_digest{};
  const std::string_view identity =
      "HDG-SC25-CUDA-R2-COMBINED-CHECKPOINT-v1";
  const std::uint64_t rank_count = 2;
  const bool combined_valid =
      combined.Update(identity.data(), identity.size()) &&
      combined.Update(&rank_count, sizeof(rank_count)) &&
      combined.Update(rank_ordered.data(), sizeof(rank_ordered)) &&
      combined.Finalize(combined_digest);
  checkpoint.canonical_manifest_hash = combined_valid
      ? StateHash{combined_digest} : StateHash{};
  return ScalarCollectiveAll(combined_valid, context);
#else
  return true;
#endif
}

bool ScalarBeginTransaction(DurableGenerationTransactionV1& transaction,
                            const std::filesystem::path& output_directory,
                            const std::uint64_t generation,
                            const CudaApplicationContextV1& context,
                            std::string& detail)
{
  bool ready = true;
  if (context.rank() == 0)
    ready = transaction.Begin(output_directory, generation, detail) ==
        DurableIoStatus::ok;
  return ScalarCollectiveAll(ready, context);
}

bool ScalarStageArtifact(DurableGenerationTransactionV1& transaction,
                         const std::string_view name,
                         const std::vector<std::uint8_t>& bytes,
                         const CudaApplicationContextV1& context,
                         std::string& detail)
{
  bool staged = true;
  if (context.rank() == 0)
    staged = transaction.Stage(name, bytes.data(), bytes.size(), detail) ==
        DurableIoStatus::ok;
  return ScalarCollectiveAll(staged, context);
}

bool ScalarAddAlias(DurableGenerationTransactionV1& transaction,
                    const std::filesystem::path& alias,
                    const std::string_view name,
                    const CudaApplicationContextV1& context,
                    std::string& detail)
{
  bool added = true;
  if (context.rank() == 0)
    added = transaction.AddAlias(alias, name, detail) == DurableIoStatus::ok;
  return ScalarCollectiveAll(added, context);
}

bool ScalarPublish(DurableGenerationTransactionV1& transaction,
                   const CudaApplicationContextV1& context,
                   std::string& detail)
{
  bool published = true;
  if (context.rank() == 0)
    published = (transaction.staged() ? transaction.Publish(detail)
                                      : transaction.Revert(detail)) ==
        DurableIoStatus::ok;
  return ScalarCollectiveAll(published, context);
}

bool ScalarRevert(DurableGenerationTransactionV1& transaction,
                  const CudaApplicationContextV1& context,
                  std::string& detail)
{
  bool reverted = true;
  if (context.rank() == 0)
    reverted = transaction.Revert(detail) == DurableIoStatus::ok;
  return ScalarCollectiveAll(reverted, context);
}

bool GatherScalarVector(const std::vector<double>& local,
                        const CudaApplicationContextV1& context,
                        std::array<int, 2>& counts,
                        std::array<int, 2>& displacements,
                        std::vector<double>& gathered,
                        std::string& detail)
{
  const bool representable =
      local.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max());
  const int local_count = representable ? static_cast<int>(local.size()) : 0;
  if (!ScalarCollectiveAll(representable, context))
  {
    detail = "scalar-IGR gather count is not representable";
    return false;
  }
  const bool count_ready = MPI_Gather(
      &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
      context.communicator()) == MPI_SUCCESS;
  if (!ScalarCollectiveAll(count_ready, context)) return false;
  bool allocated = true;
  if (context.rank() == 0)
  {
    try
    {
      int total = 0;
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        displacements[static_cast<std::size_t>(rank)] = total;
        if (counts[static_cast<std::size_t>(rank)] < 0 ||
            total > std::numeric_limits<int>::max() -
                counts[static_cast<std::size_t>(rank)])
          throw std::length_error{"scalar gather count"};
        total += counts[static_cast<std::size_t>(rank)];
      }
      gathered.resize(static_cast<std::size_t>(total));
    }
    catch (...) { allocated = false; }
  }
  if (!ScalarCollectiveAll(allocated, context))
  {
    detail = "scalar-IGR gather allocation failed";
    return false;
  }
  const bool gathered_ok = MPI_Gatherv(
      local.data(), local_count, MPI_DOUBLE,
      context.rank() == 0 ? gathered.data() : nullptr, counts.data(),
      displacements.data(), MPI_DOUBLE, 0, context.communicator()) ==
      MPI_SUCCESS;
  const bool collectively_gathered = ScalarCollectiveAll(gathered_ok, context);
  if (!collectively_gathered)
    detail = "scalar-IGR payload gather failed";
  return collectively_gathered;
}

mfem::Mesh MakeScalarCudaMesh(const ValidatedApplicationConfig& config)
{
  mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
      static_cast<int>(config.mesh_n()), static_cast<int>(config.mesh_n()),
      mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<mfem::Vector> translations(2, mfem::Vector(2));
  translations[0][0] = 1.0;
  translations[1][1] = 1.0;
  return mfem::Mesh::MakePeriodic(
      original, original.CreatePeriodicVertexMapping(translations));
}

struct ScalarVtkWorkspace final
{
  explicit ScalarVtkWorkspace(const ValidatedApplicationConfig& config)
      : mesh(MakeScalarCudaMesh(config)),
        collection(config.polynomial_order(), 2,
                   mfem::BasisType::GaussLegendre,
                   mfem::FiniteElement::VALUE),
        u_space(&mesh, &collection, 4, mfem::Ordering::byNODES),
        sigma_space(&mesh, &collection, 1, mfem::Ordering::byNODES),
        z_space(&mesh, &collection, 2, mfem::Ordering::byNODES),
        u(&u_space), sigma(&sigma_space), z(&z_space)
  {}

  mfem::Mesh mesh;
  mfem::L2_FECollection collection;
  mfem::FiniteElementSpace u_space;
  mfem::FiniteElementSpace sigma_space;
  mfem::FiniteElementSpace z_space;
  mfem::GridFunction u;
  mfem::GridFunction sigma;
  mfem::GridFunction z;
};

bool MakeScalarCudaVtk(const ValidatedApplicationConfig& config,
                       ScalarVtkWorkspace& workspace,
                       const int ranks, const std::array<int, 2>& counts,
                       const std::array<int, 2>& displacements,
                       const std::vector<double>& gathered,
                       std::string& vtk, std::string& detail)
{
  try
  {
    mfem::Mesh& mesh = workspace.mesh;
    mfem::FiniteElementSpace& u_space = workspace.u_space;
    mfem::FiniteElementSpace& sigma_space = workspace.sigma_space;
    mfem::FiniteElementSpace& z_space = workspace.z_space;
    mfem::GridFunction& u = workspace.u;
    mfem::GridFunction& sigma = workspace.sigma;
    mfem::GridFunction& z = workspace.z;
    double* const u_values = u.HostWrite();
    double* const sigma_values = sigma.HostWrite();
    double* const z_values = z.HostWrite();
    std::fill_n(u_values, u.Size(), 0.0);
    std::fill_n(sigma_values, sigma.Size(), 0.0);
    std::fill_n(z_values, z.Size(), 0.0);
    const int n = static_cast<int>(config.mesh_n());
    const int elements = n * n;
    const int scalar_dofs = (config.polynomial_order() + 1) *
        (config.polynomial_order() + 1);
    if (elements != 64 || (ranks != 1 && ranks != 2)) return false;
    for (int rank = 0; rank < ranks; ++rank)
    {
      const int owned = ranks == 1 ? elements : elements / 2;
      if (counts[static_cast<std::size_t>(rank)] !=
          owned * scalar_dofs * scalar::local_components)
        return false;
    }
    const int slab_width = n / 2;
    for (int ey = 0; ey < n; ++ey)
      for (int ex = 0; ex < n; ++ex)
      {
        const int element = ey * n + ex;
        const int owner = ranks == 1 ? 0 : (ex < slab_width ? 0 : 1);
        const int local_ex = ranks == 1 ? ex : ex - owner * slab_width;
        const int local_element = ey * (ranks == 1 ? n : slab_width) +
            local_ex;
        const int base = displacements[static_cast<std::size_t>(owner)];
        const int owned = counts[static_cast<std::size_t>(owner)] /
            (scalar_dofs * scalar::local_components);
        const auto load = [&](mfem::FiniteElementSpace& space,
                              double* const field, const int field_size,
                              const int components,
                              const int field_offset,
                              const int element_stride) {
          mfem::Array<int> dofs;
          space.GetElementVDofs(element, dofs);
          if (dofs.Size() != scalar_dofs * components) return false;
          for (int component = 0; component < components; ++component)
            for (int node = 0; node < scalar_dofs; ++node)
            {
              const int source = base + field_offset +
                  local_element * element_stride + component * scalar_dofs +
                  node;
              const int encoded = dofs[component * scalar_dofs + node];
              const int destination = encoded >= 0 ? encoded : -1 - encoded;
              if (source < base ||
                  source >= base + counts[static_cast<std::size_t>(owner)] ||
                  destination < 0 || destination >= field_size)
                return false;
              field[destination] = (encoded >= 0 ? 1.0 : -1.0) *
                  gathered[static_cast<std::size_t>(source)];
            }
          return true;
        };
        const int local_u = owned * scalar_dofs * 4;
        const int local_sigma = owned * scalar_dofs;
        if (!load(u_space, u_values, u.Size(), 4, 0, scalar_dofs * 4) ||
            !load(sigma_space, sigma_values, sigma.Size(), 1, local_u,
                  scalar_dofs) ||
            !load(z_space, z_values, z.Size(), 2, local_u + local_sigma,
                  scalar_dofs * 2))
          return false;
      }
    std::ostringstream output;
    output << std::setprecision(17);
    mesh.PrintVTK(output, 1, 1);
    u.SaveVTK(output, "U", 1);
    sigma.SaveVTK(output, "Sigma", 1);
    z.SaveVTK(output, "z", 1);
    vtk = output.str();
    return output.good();
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    return false;
  }
}

bool StageScalarCudaVtk(const ValidatedApplicationConfig& config,
                        const ScalarPackedState& state,
                        ScalarVtkWorkspace* const workspace,
                        const CudaApplicationContextV1& context,
                        DurableGenerationTransactionV1& transaction,
                        std::string& detail)
{
  std::array<int, 2> counts{}, displacements{};
  std::vector<double> gathered;
  if (!GatherScalarVector(state.interior, context, counts, displacements,
                          gathered, detail))
    return false;
  std::vector<std::uint8_t> bytes;
  std::string step_name;
  bool ready = true;
  try { step_name = ScalarStepName(state.coordinates.step); }
  catch (...) { ready = false; }
  if (context.rank() == 0 && ready)
  {
    try
    {
      std::string vtk;
      ready = workspace != nullptr &&
          MakeScalarCudaVtk(config, *workspace, context.ranks(), counts,
                                displacements, gathered, vtk, detail);
      if (ready) bytes.assign(vtk.begin(), vtk.end());
    }
    catch (...) { ready = false; }
  }
  ready = ScalarCollectiveAll(ready, context);
  bool staged = false;
  if (ready)
    staged = ScalarStageArtifact(transaction, step_name, bytes, context,
                                 detail);
  return ready && staged;
}

StateHash ScalarAggregateHash(const StateHash& local,
                              const CudaApplicationContextV1& context) noexcept
{
  std::array<StateHash, 2> rank_ordered{};
  const bool gathered = MPI_Allgather(
      local.bytes.data(), 32, MPI_UINT8_T, rank_ordered.data(), 32,
      MPI_UINT8_T, context.communicator()) == MPI_SUCCESS;
  if (!ScalarCollectiveAll(gathered, context)) return {};
  bool local_hashes_nonzero = true;
  for (int rank = 0; rank < context.ranks(); ++rank)
    local_hashes_nonzero = local_hashes_nonzero &&
        rank_ordered[static_cast<std::size_t>(rank)] != StateHash{};
  const StateHash aggregate{ComputeSha256(
      rank_ordered.data(), static_cast<std::size_t>(context.ranks()) *
          sizeof(StateHash))};
  const bool valid = local_hashes_nonzero && aggregate != StateHash{};
  return ScalarCollectiveAll(valid, context) ? aggregate : StateHash{};
}

bool ScalarGpuRecord(const int device,
                     ScalarResourceRecord& record) noexcept
{
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    return false;
  std::copy_n(reinterpret_cast<const std::uint8_t*>(properties.uuid.bytes),
              record.gpu_uuid.size(), record.gpu_uuid.begin());
  return std::any_of(record.gpu_uuid.begin(), record.gpu_uuid.end(),
                     [](const std::uint8_t byte) { return byte != 0; });
}

std::string ScalarGpuUuidText(const std::array<std::uint8_t, 16>& uuid)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string text = "GPU-";
  text.resize(36);
  for (std::size_t index = 0; index < uuid.size(); ++index)
  {
    text[4 + 2 * index] = digits[uuid[index] >> 4U];
    text[5 + 2 * index] = digits[uuid[index] & 0xfU];
  }
  return text;
}

PersistentResourceDescriptor ScalarResources(
    const ScalarResourceSidecar& sidecar,
    const ValidatedApplicationConfig& config) noexcept
{
  PersistentResourceDescriptor resources{};
  resources.mfem_pin = StableIdentity::Copy(scalar::mfem_pin);
  if (sidecar.rank_count == 1)
  {
    resources.mesh_hash = sidecar.ranks[0].mesh;
    resources.partition_hash = sidecar.ranks[0].partition;
    resources.topology_hash = sidecar.ranks[0].topology;
  }
  else
  {
    std::array<StateHash, 2> rank_ordered{};
    for (std::uint32_t rank = 0; rank < sidecar.rank_count; ++rank)
      rank_ordered[rank] = sidecar.ranks[rank].mesh;
    resources.mesh_hash.bytes = ComputeSha256(
        rank_ordered.data(), sidecar.rank_count * sizeof(StateHash));
    for (std::uint32_t rank = 0; rank < sidecar.rank_count; ++rank)
      rank_ordered[rank] = sidecar.ranks[rank].partition;
    resources.partition_hash.bytes = ComputeSha256(
        rank_ordered.data(), sidecar.rank_count * sizeof(StateHash));
    for (std::uint32_t rank = 0; rank < sidecar.rank_count; ++rank)
      rank_ordered[rank] = sidecar.ranks[rank].topology;
    resources.topology_hash.bytes = ComputeSha256(
        rank_ordered.data(), sidecar.rank_count * sizeof(StateHash));
  }
  resources.rank_count = sidecar.rank_count;
  resources.polynomial_order = config.polynomial_order();
  resources.layout = StableIdentity::Copy(config.layout_id().view());
  resources.reuse = {true, true, true, false, 1, 1, 1, 1};
  return resources;
}

CheckpointRuntimeIdentityV1 ScalarRuntimeIdentity(
    const ScalarResourceSidecar& sidecar,
    const PersistentResourceDescriptor& resources) noexcept
{
  CheckpointRuntimeIdentityV1 identity{};
  identity.physical_gpu_uuid = StableIdentity::Copy(
      ScalarGpuUuidText(sidecar.ranks[0].gpu_uuid));
  identity.ownership_hash = resources.partition_hash;
  return identity;
}

bool BuildScalarSidecar(const ScalarCheckpoint& checkpoint,
                        const int device,
                        const CudaApplicationContextV1& context,
                        ScalarResourceSidecar& sidecar,
                        std::string& detail) noexcept
{
  ScalarResourceRecord local{};
  local.mesh = checkpoint.descriptor.mesh_hash;
  local.partition = checkpoint.descriptor.partition_hash;
  local.topology = checkpoint.descriptor.topology_hash;
  local.private_descriptor = checkpoint.descriptor;
  local.raw_field_hashes = checkpoint.raw_field_hashes;
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  local.shard_manifest_hash = checkpoint.shard_manifest_hash;
#endif
  local.canonical_manifest_hash = checkpoint.canonical_manifest_hash;
  bool ready = ScalarGpuRecord(device, local);
  ScalarResourceSidecar candidate{};
  const bool gathered = MPI_Gather(
      &local, static_cast<int>(sizeof(local)), MPI_BYTE,
      candidate.ranks.data(), static_cast<int>(sizeof(local)), MPI_BYTE, 0,
      context.communicator()) == MPI_SUCCESS;
  ready = ready && gathered;
  if (context.rank() == 0 && ready)
  {
    candidate.rank_count = static_cast<std::uint32_t>(context.ranks());
    for (int rank = 1; rank < context.ranks(); ++rank)
      ready = ready && candidate.ranks[static_cast<std::size_t>(rank)].gpu_uuid ==
          candidate.ranks[0].gpu_uuid;
    candidate.checksum.bytes = ComputeSha256(
        &candidate, offsetof(ScalarResourceSidecar, checksum));
    ready = ready && candidate.checksum != StateHash{};
  }
  if (!ScalarCollectiveAll(ready, context))
  {
    detail = "scalar-IGR CUDA resource identity capture failed";
    return false;
  }
  const bool broadcast = MPI_Bcast(
      &candidate, static_cast<int>(sizeof(candidate)), MPI_BYTE, 0,
      context.communicator()) == MPI_SUCCESS;
  if (!ScalarCollectiveAll(broadcast, context)) return false;
  sidecar = candidate;
  return true;
}

bool ValidScalarSidecar(const ScalarResourceSidecar& sidecar,
                        const ValidatedApplicationConfig& config,
                        const CudaApplicationContextV1& context) noexcept
{
  const ScalarResourceSidecar expected{};
  const std::uint32_t required =
      config.deferred_runtime().required_mpi_rank_count;
  const bool bounded_header = sidecar.magic == expected.magic &&
      sidecar.version == 1 && sidecar.rank_count >= 1 &&
      sidecar.rank_count <= sidecar.ranks.size() &&
      sidecar.rank_count == required &&
      sidecar.rank_count == static_cast<std::uint32_t>(context.ranks()) &&
      sidecar.checksum != StateHash{} &&
      sidecar.checksum.bytes == ComputeSha256(
          &sidecar, offsetof(ScalarResourceSidecar, checksum));
  if (!bounded_header) return false;

  ScalarResourceRecord actual{};
  const bool uuid_ready = ScalarGpuRecord(config.device_id(), actual);
  bool valid = uuid_ready;
  for (std::uint32_t rank = 0; rank < sidecar.rank_count; ++rank)
  {
    const ScalarResourceRecord& record =
        sidecar.ranks[static_cast<std::size_t>(rank)];
    const CheckpointDescriptor& descriptor = record.private_descriptor;
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
    const RuntimeTupleDescriptor expected_tuple =
        scalar::CudaRank2Selection::Descriptor();
#else
    const RuntimeTupleDescriptor expected_tuple =
        scalar::CudaRank1Selection::Descriptor();
#endif
    const bool raw_hashes = record.raw_field_hashes.u != StateHash{} &&
        record.raw_field_hashes.sigma != StateHash{} &&
        record.raw_field_hashes.z != StateHash{} &&
        record.raw_field_hashes.uhat != StateHash{} &&
        record.raw_field_hashes.sigmahat != StateHash{} &&
        record.raw_field_hashes.interior != StateHash{} &&
        record.raw_field_hashes.trace != StateHash{};
    const StateCoordinates& coordinates = descriptor.coordinates;
    bool record_valid = record.mesh != StateHash{} &&
        record.partition != StateHash{} && record.topology != StateHash{} &&
        record.gpu_uuid == actual.gpu_uuid && raw_hashes &&
        record.canonical_manifest_hash != StateHash{} &&
        descriptor.schema_version == checkpoint_schema_version &&
        descriptor.successful_commit &&
        descriptor.restart_mode == RestartMode::same_backend_rank_partition &&
        descriptor.tuple == expected_tuple &&
        descriptor.mfem_pin == StableIdentity::Copy(scalar::mfem_pin) &&
        descriptor.mesh_hash == record.mesh &&
        descriptor.partition_hash == record.partition &&
        descriptor.topology_hash == record.topology &&
        descriptor.rank_count == sidecar.rank_count &&
        descriptor.polynomial_order == config.polynomial_order() &&
        descriptor.layout == StableIdentity::Copy(scalar::interior_layout_id) &&
        descriptor.precision == StableIdentity::Copy("Fp64") &&
        descriptor.cursor == coordinates.cursor &&
        descriptor.dt == coordinates.dt &&
        descriptor.coordinate_fingerprint ==
            ::dg::hdg::runtime::detail::CoordinateFingerprint(coordinates) &&
        coordinates.step == config.scenario_checkpoint_step() &&
        coordinates.step == coordinates.generation &&
        coordinates.step == coordinates.cursor &&
        coordinates.time == static_cast<double>(coordinates.step) * config.dt() &&
        coordinates.dt == config.dt() &&
        descriptor.interior_hash == record.raw_field_hashes.interior &&
        descriptor.trace_hash == record.raw_field_hashes.trace;
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
    record_valid = record_valid &&
        record.raw_field_hashes.shard != StateHash{} &&
        record.shard_manifest_hash != StateHash{};
#else
    record_valid = record_valid && record.shard_manifest_hash == StateHash{};
#endif
    valid = valid && record_valid;
  }
  if (sidecar.rank_count == 2)
    valid = valid && sidecar.ranks[0].gpu_uuid == sidecar.ranks[1].gpu_uuid &&
        sidecar.ranks[0].canonical_manifest_hash ==
            sidecar.ranks[1].canonical_manifest_hash;
  return valid;
}

bool StageScalarCudaCheckpoint(
    const ValidatedApplicationConfig& writer,
    const ScalarCheckpoint& checkpoint,
    const ScalarResourceSidecar& sidecar,
    const CudaApplicationContextV1& context,
    DurableGenerationTransactionV1& transaction,
    std::string& detail)
{
  std::vector<double> local_interior;
  std::vector<double> local_trace;
  std::string checkpoint_alias;
  bool locally_prepared = true;
  try
  {
    local_interior.assign(checkpoint.interior_component_slabs.begin(),
                          checkpoint.interior_component_slabs.end());
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
    local_trace = checkpoint.owned_trace_component_slabs;
#else
    local_trace.assign(checkpoint.trace_component_slabs.begin(),
                       checkpoint.trace_component_slabs.end());
#endif
    checkpoint_alias = writer.checkpoint_out().identity().view();
  }
  catch (...) { locally_prepared = false; }
  if (!ScalarCollectiveAll(locally_prepared, context))
  {
    detail = "scalar-IGR CUDA checkpoint local preparation failed";
    return false;
  }
  const PersistentResourceDescriptor resources =
      ScalarResources(sidecar, writer);
  CheckpointDescriptor local_descriptor = checkpoint.descriptor;
  local_descriptor.tuple = writer.tuple();
  local_descriptor.mfem_pin = resources.mfem_pin;
  local_descriptor.mesh_hash = resources.mesh_hash;
  local_descriptor.partition_hash = resources.partition_hash;
  local_descriptor.topology_hash = resources.topology_hash;
  local_descriptor.rank_count = resources.rank_count;
  local_descriptor.polynomial_order = resources.polynomial_order;
  local_descriptor.layout = resources.layout;
  local_descriptor.precision = writer.tuple().precision;
  local_descriptor.interior_hash.bytes = ComputeSha256(
      local_interior.data(), local_interior.size() * sizeof(double));
  local_descriptor.trace_hash.bytes = ComputeSha256(
      local_trace.data(), local_trace.size() * sizeof(double));
  std::array<CheckpointDescriptor, 2> descriptors{};
  const bool descriptors_ready = MPI_Gather(
      &local_descriptor, static_cast<int>(sizeof(local_descriptor)), MPI_BYTE,
      descriptors.data(), static_cast<int>(sizeof(local_descriptor)), MPI_BYTE,
      0, context.communicator()) == MPI_SUCCESS;
  if (!ScalarCollectiveAll(descriptors_ready, context)) return false;
  std::array<int, 2> interior_counts{}, interior_displacements{};
  std::array<int, 2> trace_counts{}, trace_displacements{};
  std::vector<double> interiors, traces;
  const bool interiors_ready = GatherScalarVector(
      local_interior, context, interior_counts, interior_displacements,
      interiors, detail);
  const bool traces_ready = GatherScalarVector(
      local_trace, context, trace_counts, trace_displacements, traces, detail);
  if (!interiors_ready || !traces_ready)
    return false;
  std::vector<std::uint8_t> encoded;
  bool encoded_ok = true;
  if (context.rank() == 0)
  {
    try
    {
      const CheckpointCoefficientSchemaV1 schema =
          CheckpointCoefficientSchema(writer);
      std::array<StagedStateViewV1, 2> states{};
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        states[index] = {
            static_cast<std::uint32_t>(rank),
            {interiors.data() + interior_displacements[index],
             static_cast<std::uint64_t>(interior_counts[index]),
             schema.interior_component_count,
             static_cast<std::uint64_t>(interior_counts[index]) /
                 schema.interior_component_count,
             CoefficientOrderingV1::component_major},
            {traces.data() + trace_displacements[index],
             static_cast<std::uint64_t>(trace_counts[index]),
             schema.trace_component_count,
             static_cast<std::uint64_t>(trace_counts[index]) /
                 schema.trace_component_count,
             CoefficientOrderingV1::component_major}};
      }
      const auto captured = CaptureCheckpointPiecesV1(
          descriptors.data(), static_cast<std::size_t>(context.ranks()),
          writer, ScalarRuntimeIdentity(sidecar, resources));
      encoded_ok = captured && EncodeCheckpointV1(
          *captured, states.data(), static_cast<std::size_t>(context.ranks()),
          encoded) == CheckpointCodecStatus::ok;
    }
    catch (...) { encoded_ok = false; }
  }
  if (!ScalarCollectiveAll(encoded_ok, context))
  {
    detail = "scalar-IGR CUDA checkpoint encode failed";
    return false;
  }
  std::vector<std::uint8_t> sidecar_bytes;
  bool sidecar_ready = true;
  if (context.rank() == 0)
  {
    try
    {
      const auto* begin = reinterpret_cast<const std::uint8_t*>(&sidecar);
      sidecar_bytes.assign(begin, begin + sizeof(sidecar));
    }
    catch (...) { sidecar_ready = false; }
  }
  if (!ScalarCollectiveAll(sidecar_ready, context))
  {
    detail = "scalar-IGR CUDA checkpoint sidecar serialization failed";
    return false;
  }
  const bool checkpoint_staged = ScalarStageArtifact(
      transaction, "restart-checkpoint.v2", encoded, context, detail);
  const bool alias_added = ScalarAddAlias(
      transaction, checkpoint_alias, "restart-checkpoint.v2", context,
      detail);
  const bool sidecar_staged = ScalarStageArtifact(
      transaction, "checkpoint-runtime-identity.v1", sidecar_bytes, context,
      detail);
  return checkpoint_staged && alias_added && sidecar_staged;
}

bool RestoreScalarCudaCheckpoint(
    const ValidatedApplicationConfig& config, ScalarOwner& owner,
    const CudaApplicationContextV1& context, std::uint32_t& retries,
    std::string& detail)
{
  ScalarResourceSidecar sidecar{};
  std::array<std::vector<double>, 2> interiors, traces;
  std::array<RankStateBufferV1, 2> buffers{};
  DecodedCheckpointMetadataV1 metadata{};
  bool decoded = true;
  if (context.rank() == 0)
  {
    try
    {
      std::vector<std::uint8_t> identity_bytes;
      const std::filesystem::path identity_path =
          std::filesystem::path{
              std::string{config.output_directory().identity().view()}} /
          "checkpoint-runtime-identity.v1";
      decoded = detail::ReadCurrentGenerationArtifact(
                    identity_path, sizeof(sidecar), identity_bytes, detail) ==
              DurableIoStatus::ok &&
          identity_bytes.size() == sizeof(sidecar);
      if (decoded)
        std::memcpy(&sidecar, identity_bytes.data(), sizeof(sidecar));
    }
    catch (...) { decoded = false; }
  }
  const bool identity_read = ScalarCollectiveAll(decoded, context);
  const bool identity_broadcast = MPI_Bcast(
      &sidecar, static_cast<int>(sizeof(sidecar)), MPI_BYTE, 0,
      context.communicator()) == MPI_SUCCESS;
  const bool identity_valid = ValidScalarSidecar(sidecar, config, context);
  decoded = identity_read && identity_broadcast && identity_valid;
  if (!ScalarCollectiveAll(decoded, context))
  {
    detail = "scalar-IGR CUDA restart resource identity rejected";
    return false;
  }
  ScenarioCoefficientExtentsV1 extents{};
  const bool extents_ready = DeriveScenarioCoefficientExtentsV1(
      config, static_cast<std::uint32_t>(context.ranks()), extents);
  if (!ScalarCollectiveAll(extents_ready, context))
    return false;
  if (context.rank() == 0)
  {
    try
    {
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const auto& shape = extents.ranks[static_cast<std::size_t>(rank)];
        interiors[rank].resize(shape.interior_coefficient_count);
        traces[rank].resize(shape.trace_coefficient_count);
        buffers[rank] = {
            static_cast<std::uint32_t>(rank),
            {interiors[rank].data(), interiors[rank].size(),
             interiors[rank].size(), extents.interior_component_count,
             shape.interior_dofs_per_component,
             CoefficientOrderingV1::component_major},
            {traces[rank].data(), traces[rank].size(), traces[rank].size(),
             extents.trace_component_count,
             shape.trace_dofs_per_component,
             CoefficientOrderingV1::component_major}};
      }
      std::vector<std::uint8_t> encoded;
      decoded = detail::ReadCurrentGenerationArtifact(
                    std::string{config.restart_from().identity().view()},
                    64U * 1024U * 1024U, encoded, detail) ==
          DurableIoStatus::ok;
      const PersistentResourceDescriptor resources =
          ScalarResources(sidecar, config);
      if (decoded)
        decoded = DecodeCheckpointV1(
            {encoded.data(), encoded.size()}, config, resources,
            ScalarRuntimeIdentity(sidecar, resources), buffers.data(),
            static_cast<std::size_t>(context.ranks()), metadata) ==
            CheckpointCodecStatus::ok;
    }
    catch (...) { decoded = false; }
  }
  if (!ScalarCollectiveAll(decoded, context))
  {
    detail = "scalar-IGR CUDA generic checkpoint decode rejected";
    return false;
  }
  const bool descriptor_ready = MPI_Bcast(
      &metadata.descriptor, static_cast<int>(sizeof(metadata.descriptor)),
      MPI_BYTE, 0, context.communicator()) == MPI_SUCCESS;
  if (!ScalarCollectiveAll(descriptor_ready, context)) return false;
  std::array<int, 2> interior_counts{}, interior_displacements{};
  std::array<int, 2> trace_counts{}, trace_displacements{};
  std::vector<double> flat_interior, flat_trace;
  std::vector<double> local_interior, local_trace;
  bool prepared = true;
  try
  {
    if (context.rank() == 0)
    {
      int interior_offset = 0;
      int trace_offset = 0;
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        if (interiors[index].size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max()) ||
            traces[index].size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max()))
          throw std::length_error{"scalar-IGR restart MPI count"};
        interior_counts[index] = static_cast<int>(interiors[index].size());
        trace_counts[index] = static_cast<int>(traces[index].size());
        if (interior_offset > std::numeric_limits<int>::max() -
                                  interior_counts[index] ||
            trace_offset > std::numeric_limits<int>::max() -
                               trace_counts[index])
          throw std::length_error{"scalar-IGR restart MPI displacement"};
        interior_displacements[index] = interior_offset;
        trace_displacements[index] = trace_offset;
        interior_offset += interior_counts[index];
        trace_offset += trace_counts[index];
      }
      flat_interior.resize(static_cast<std::size_t>(interior_offset));
      flat_trace.resize(static_cast<std::size_t>(trace_offset));
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        std::copy(interiors[index].begin(), interiors[index].end(),
                  flat_interior.begin() + interior_displacements[index]);
        std::copy(traces[index].begin(), traces[index].end(),
                  flat_trace.begin() + trace_displacements[index]);
      }
    }
    const auto& local_shape =
        extents.ranks[static_cast<std::size_t>(context.rank())];
    local_interior.resize(local_shape.interior_coefficient_count);
    local_trace.resize(local_shape.trace_coefficient_count);
  }
  catch (...) { prepared = false; }
  if (!ScalarCollectiveAll(prepared, context))
  {
    detail = "scalar-IGR CUDA restart flattening preparation failed";
    return false;
  }
  const bool interior_scattered = MPI_Scatterv(
      flat_interior.data(), interior_counts.data(), interior_displacements.data(),
      MPI_DOUBLE, local_interior.data(), static_cast<int>(local_interior.size()),
      MPI_DOUBLE, 0, context.communicator()) == MPI_SUCCESS;
  const bool trace_scattered = MPI_Scatterv(
      flat_trace.data(), trace_counts.data(), trace_displacements.data(),
      MPI_DOUBLE, local_trace.data(), static_cast<int>(local_trace.size()),
      MPI_DOUBLE, 0, context.communicator()) == MPI_SUCCESS;
  if (!ScalarCollectiveAll(interior_scattered && trace_scattered, context))
    return false;
  ScalarPackedState state;
  state.coordinates = metadata.descriptor.coordinates;
  bool converted = true;
  try
  {
    converted = detail::ScalarIgrComponentMajorToFieldBlocksV1(
        local_interior, scalar::volume_nodes, state.interior);
    if (converted) state.trace = std::move(local_trace);
  }
  catch (...) { converted = false; }
  if (!ScalarCollectiveAll(converted, context))
  {
    detail = "scalar-IGR CUDA restart coefficient conversion failed";
    return false;
  }
  ScalarCheckpoint checkpoint{};
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  checkpoint.scenario = StableIdentity::Copy(
      config.case_id().view() == "periodic-smooth-transition-v2"
          ? scalar::cuda_rank2_smooth_transition_fixture_id
          : scalar::cuda_rank2_fixture_id);
#endif
  const bool sealed = SealScalarCheckpoint(
      checkpoint, state, sidecar.ranks[static_cast<std::size_t>(context.rank())],
      context);
  if (!ScalarCollectiveAll(sealed, context)) return false;
  const ScalarResourceRecord& private_identity =
      sidecar.ranks[static_cast<std::size_t>(context.rank())];
  checkpoint.descriptor = private_identity.private_descriptor;
  checkpoint.raw_field_hashes = private_identity.raw_field_hashes;
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  checkpoint.shard_manifest_hash = private_identity.shard_manifest_hash;
#endif
  checkpoint.canonical_manifest_hash =
      private_identity.canonical_manifest_hash;
  const scalar::ScalarCudaStatus restore_status =
      owner.RestoreTrajectoryCheckpoint(checkpoint);
  const bool restored = ScalarRestored(restore_status);
  if (!ScalarCollectiveAll(restored, context))
  {
    detail = "scalar-IGR CUDA leaf restart validation failed status=" +
        std::to_string(static_cast<unsigned>(restore_status));
    return false;
  }
  retries = scalar_igr_sc25_v3_retry_limit;
  return true;
}

std::optional<ValidatedApplicationConfig> ScalarCheckpointWriterConfig(
    const ApplicationOptionsView& request,
    const ValidatedApplicationConfig& config)
{
  if (!config.checkpoint_out().present()) return std::nullopt;
  ApplicationOptionsView writer = request;
  writer.run_role = ApplicationRunRoleV1::split_writer;
  writer.checkpoint_step = OptionalCheckpointStepV1::Present(
      config.scenario_checkpoint_step());
  writer.stop_after_checkpoint = StopAfterCheckpointV1::yes;
  writer.restart_from = {};
  writer.injection = {
      FailureInjectionModeV1::prescribed,
      smooth_periodic_euler_v1_failure_phase,
      scalar_igr_sc25_v3_failure_target_step,
      config.deferred_runtime().required_injected_source_rank};
  writer.retry_limit = scalar_igr_sc25_v3_retry_limit;
  const ApplicationValidation validation = ValidateApplicationOptions(writer);
  return validation.accepted()
      ? std::optional<ValidatedApplicationConfig>{*validation.config()}
      : std::nullopt;
}

scalar::FailureInjection ScalarNumericalInjection(
    const FailurePhase phase) noexcept
{
  switch (phase)
  {
    case FailurePhase::Assembly:
    case FailurePhase::Condensation:
      return scalar::FailureInjection::local_factor;
    case FailurePhase::TraceSolve:
      return scalar::FailureInjection::trace_linear;
    case FailurePhase::Recovery:
      return scalar::FailureInjection::admissibility;
    case FailurePhase::Diagnostics:
      return scalar::FailureInjection::newton_iteration_limit;
    default: return scalar::FailureInjection::none;
  }
}

bool ScalarTransactionReady(const ScalarOwner::Transaction& token,
                            const StateCoordinates& expected) noexcept
{
  const StateCoordinates& target = token.target_coordinates();
  return token.valid() && token.provisionally_installed() &&
      target.step == expected.step && target.time == expected.time &&
      target.generation == expected.generation &&
      target.cursor == expected.cursor && target.dt == expected.dt;
}

std::string ScalarHex(const StateHash& hash)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    result[2 * index] = digits[hash.bytes[index] >> 4U];
    result[2 * index + 1] = digits[hash.bytes[index] & 0xfU];
  }
  return result;
}

struct ScalarInjectionAudit final
{
  int configured_source_rank = -1;
  int observed_origin_rank = -1;
  int local_injection_count = 0;
  int non_origin_local_injection_count = 0;
  int collective_decisions = 0;
  int consumed_boundary_count = 0;
};

std::string ScalarCudaMetadata(
    const ValidatedApplicationConfig& config,
    const ApplicationRunResult& result,
    const scalar::ScalarCudaEvidence& evidence,
    const StateHash& initial_hash, const StateHash& final_interior,
    const StateHash& final_trace, const ScalarInjectionAudit& injection,
    const ScalarResourceRecord& resource, const std::string& diagnostic)
{
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  const double maximum_conservation = evidence.maximum_conservation_error;
  const double minimum_density = evidence.minimum_density;
  const double minimum_pressure = evidence.minimum_pressure;
  const double minimum_speed_margin = evidence.minimum_speed_margin;
  const bool trace_residual_gate =
      evidence.maximum_trace_residual_ratio <= 1.0;
  const std::uint64_t physical_cuda_transfers =
      evidence.transfer_totals.event_count;
#else
  double maximum_conservation = 0.0;
  double minimum_density = std::numeric_limits<double>::infinity();
  double minimum_pressure = std::numeric_limits<double>::infinity();
  double minimum_speed_margin = std::numeric_limits<double>::infinity();
  bool trace_residual_gate = true;
  for (int attempt_index = 0; attempt_index < evidence.attempt_count;
       ++attempt_index)
  {
    const auto& attempt = evidence.attempts[
        static_cast<std::size_t>(attempt_index)];
    maximum_conservation = std::max(
        maximum_conservation, attempt.maximum_conservation_error);
    minimum_density = std::min(minimum_density, attempt.minimum_density);
    minimum_pressure = std::min(minimum_pressure, attempt.minimum_pressure);
    minimum_speed_margin = std::min(
        minimum_speed_margin, attempt.minimum_speed_margin);
    for (int iteration = 0; iteration < attempt.newton_iterations; ++iteration)
    {
      const auto& newton = attempt.newton[static_cast<std::size_t>(iteration)];
      const double bound = std::max(
          evidence.trace_absolute_tolerance,
          evidence.trace_relative_tolerance *
              newton.trace_right_hand_side_norm);
      trace_residual_gate = trace_residual_gate &&
          newton.trace_true_residual_norm <= bound;
    }
  }
  const std::uint64_t physical_cuda_transfers = evidence.transfers.count;
#endif
  const bool residency = evidence.residency.mfem_cuda_device &&
      evidence.residency.hypre_cuda && evidence.residency.umpire_device_pool &&
      evidence.residency.state_device_resident &&
      evidence.residency.topology_device_resident &&
      evidence.residency.workspace_device_resident &&
      evidence.residency.trace_solve_device_resident &&
      !evidence.residency.host_numerical_fallback;
  bool rank2_residency = true;
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  rank2_residency = evidence.residency.host_staged_rank2 &&
      evidence.residency.same_device_uuid;
#endif
  const bool allocations = evidence.allocations.warm_plateau &&
      evidence.allocations.zero_post_baseline_observed_growth &&
      evidence.allocations.zero_post_baseline_owned_allocation &&
      evidence.allocations.zero_post_baseline_owned_deallocation;
  const bool solver_contract = evidence.trace_restart == 50 &&
      evidence.trace_maximum_iterations == 200 &&
      evidence.trace_relative_tolerance == 1.0e-12 &&
      evidence.trace_absolute_tolerance == 1.0e-14 &&
      evidence.ilu_type == 0 && evidence.ilu_level_of_fill == 0 &&
      evidence.ilu_local_reordering == 1;
  std::ostringstream output;
  output << std::setprecision(17)
      << "{\n  \"schema\": \"dg-igr-hdg-run-metadata.v1\",\n"
      << "  \"card\": \"" << config.card_id().view() << "\",\n"
      << "  \"case\": \"" << config.case_id().view() << "\",\n"
      << "  \"scenario\": \"" << config.scenario_id().view() << "\",\n"
      << "  \"backend\": \"" << config.tuple().backend.view() << "\",\n"
      << "  \"layout\": \"" << config.layout_id().view() << "\",\n"
      << "  \"fields\": [\"U\", \"Sigma\", \"z\"],\n"
      << "  \"gpu_uuid\": \"" << ScalarGpuUuidText(resource.gpu_uuid)
      << "\",\n  \"dt\": " << config.dt() << ",\n"
      << "  \"requested_steps\": " << config.accepted_steps() << ",\n"
      << "  \"committed_step\": " << result.coordinates.step << ",\n"
      << "  \"committed_time\": " << result.coordinates.time << ",\n"
      << "  \"status\": \"" << ToString(result.status) << "\",\n"
      << "  \"reason\": \"" << ToString(result.reason) << "\",\n"
      << "  \"retries\": " << result.retries << ",\n"
      << "  \"injection_configured_source_rank\": "
      << injection.configured_source_rank << ",\n"
      << "  \"injection_observed_origin_rank\": "
      << injection.observed_origin_rank << ",\n"
      << "  \"injection_local_count\": "
      << injection.local_injection_count << ",\n"
      << "  \"injection_non_origin_local_count\": "
      << injection.non_origin_local_injection_count << ",\n"
      << "  \"injection_collective_decisions\": "
      << injection.collective_decisions << ",\n"
      << "  \"injection_consumed_boundary_count\": "
      << injection.consumed_boundary_count << ",\n"
      << "  \"accepted_steps\": " << result.coordinates.step << ",\n"
      << "  \"attempted_steps\": " << evidence.attempted_steps << ",\n"
      << "  \"physical_cuda_launches\": "
      << evidence.launches.assemble_launches +
             evidence.launches.condense_launches +
             evidence.launches.recover_launches
      << ",\n  \"physical_cuda_transfers\": " << physical_cuda_transfers
      << ",\n  \"physical_cuda_owned_allocations\": "
      << evidence.allocations.current_owned_calls
      << ",\n  \"physical_cuda_observed_allocations\": "
      << evidence.allocations.current_observed_events << ",\n"
      << "  \"maximum_conservation_error\": " << maximum_conservation
      << ",\n  \"minimum_density\": " << minimum_density
      << ",\n  \"minimum_pressure\": " << minimum_pressure
      << ",\n  \"minimum_speed_margin\": " << minimum_speed_margin << ",\n"
      << "  \"trace_residual_gate\": "
      << (trace_residual_gate ? "true" : "false") << ",\n"
      << "  \"solver_contract\": "
      << (solver_contract ? "true" : "false") << ",\n"
      << "  \"persistent_reuse\": "
      << (evidence.reuse.unchanged ? "true" : "false") << ",\n"
      << "  \"zero_fallback\": "
      << (evidence.launches.fallback_calls == 0 ? "true" : "false")
      << ",\n  \"zero_post_baseline_cuda_umpire_allocation\": "
      << (allocations ? "true" : "false") << ",\n"
      << "  \"device_residency\": "
      << (residency ? "true" : "false") << ",\n"
      << "  \"host_staged_same_gpu\": "
      << (rank2_residency ? "true" : "false") << ",\n"
      << "  \"initial_state_sha256\": \"" << ScalarHex(initial_hash)
      << "\",\n  \"final_interior_sha256\": \""
      << ScalarHex(final_interior) << "\",\n"
      << "  \"final_trace_sha256\": \"" << ScalarHex(final_trace)
      << "\",\n  \"diagnostic\": \"" << diagnostic << "\"\n}\n";
  return output.str();
}

ApplicationRunResult RunScalarIgrCuda(
    const ApplicationOptionsView& request,
    CudaApplicationContextV1& context)
{
  ApplicationRunResult result{};
  const ApplicationValidation validation = ValidateApplicationOptions(request);
  if (!validation.accepted())
  {
    result.status = request.interior_space == InteriorSpace::h1
        ? RunStatus::unsupported_h1 : RunStatus::invalid_request;
    result.reason = request.interior_space == InteriorSpace::h1
        ? FailureReason::unsupported_h1
        : FailureReason::invalid_resource_descriptor;
    result.failure_class = ApplicationRunResult::FailureClass::configuration;
    return result;
  }
  const ValidatedApplicationConfig& config = *validation.config();
  if (!config.scalar_igr())
  {
    result.status = RunStatus::invalid_request;
    result.reason = FailureReason::invalid_resource_descriptor;
    result.failure_class = ApplicationRunResult::FailureClass::configuration;
    return result;
  }
  if (context.ranks() != (scalar_rank2 ? 2 : 1))
  {
    result.status = RunStatus::invalid_request;
    result.reason = FailureReason::invalid_resource_descriptor;
    result.failure_class = ApplicationRunResult::FailureClass::configuration;
    return result;
  }
  std::string diagnostic;
  std::filesystem::path output_directory;
  bool request_prepared = true;
  try
  {
    output_directory =
        std::string{config.output_directory().identity().view()};
  }
  catch (...) { request_prepared = false; }
  if (!ScalarCollectiveAll(request_prepared, context))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    context.SetDiagnostic("scalar-IGR CUDA request preparation failed");
    return result;
  }
  bool output_ready = true;
  bool output_created = false;
  if (context.rank() == 0)
  {
    if (config.run_role() == ApplicationRunRoleV1::split_reader)
    {
      std::error_code error;
      output_ready = std::filesystem::is_directory(
          std::filesystem::symlink_status(output_directory, error));
      if (error || !output_ready)
        diagnostic = "scalar-IGR CUDA split-reader output is unavailable";
    }
    else
    {
      output_ready = CreateDurableDirectory(output_directory, diagnostic) ==
          DurableIoStatus::ok;
      output_created = output_ready;
    }
  }
  if (!ScalarCollectiveAll(output_ready, context))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    context.SetDiagnostic(diagnostic);
    return result;
  }

  std::unique_ptr<mfem::Device> device;
  bool device_ready = true;
  try { device = std::make_unique<mfem::Device>("cuda"); }
  catch (...) { device_ready = false; }
  if (!ScalarCollectiveAll(device_ready, context))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    context.SetDiagnostic("scalar-IGR CUDA MFEM device initialization failed");
    return result;
  }
  std::unique_ptr<ScalarVtkWorkspace> vtk_workspace;
  bool vtk_workspace_ready = true;
  if (context.rank() == 0)
  {
    try
    {
      vtk_workspace = std::make_unique<ScalarVtkWorkspace>(config);
      const int local_count = (scalar_rank2 ? 32 : 64) *
          scalar::volume_nodes * scalar::local_components;
      std::array<int, 2> warm_counts{{local_count,
                                      scalar_rank2 ? local_count : 0}};
      std::array<int, 2> warm_displacements{{0,
                                             scalar_rank2 ? local_count : 0}};
      std::vector<double> warm_values(
          static_cast<std::size_t>(local_count * context.ranks()), 0.0);
      std::string warm_vtk;
      vtk_workspace_ready = MakeScalarCudaVtk(
          config, *vtk_workspace, context.ranks(), warm_counts,
          warm_displacements, warm_values, warm_vtk, diagnostic);
    }
    catch (...) { vtk_workspace_ready = false; }
  }
  if (!ScalarCollectiveAll(vtk_workspace_ready, context))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    context.SetDiagnostic("scalar-IGR CUDA VTK workspace initialization failed");
    return result;
  }
  scalar::ScalarCudaConfig numerical{};
  numerical.polynomial_degree = config.polynomial_order();
  numerical.elements_per_direction = static_cast<int>(config.mesh_n());
  numerical.requested_device = config.device_id();
#if defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
  numerical.lifecycle = config.case_id().view() == "periodic-smooth-transition-v2"
      ? scalar::ScalarCudaLifecycle::w8c_smooth_transition_v2
      : scalar::ScalarCudaLifecycle::sc25_periodic_v3;
#endif
  std::unique_ptr<ScalarOwner> owner;
  const bool initialized =
      ScalarOwner::Create(context.communicator(), numerical, owner) ==
          scalar::ScalarCudaStatus::ready && owner &&
      owner->InitializePersistentTrajectory() == scalar::ScalarCudaStatus::ready;
  if (!ScalarCollectiveAll(initialized, context))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    context.SetDiagnostic("scalar-IGR CUDA leaf initialization failed");
    return result;
  }
  result.status = RunStatus::ready;
  result.reason = FailureReason::none;
  result.failure_class = ApplicationRunResult::FailureClass::none;
  if (config.run_role() == ApplicationRunRoleV1::split_reader &&
      !RestoreScalarCudaCheckpoint(config, *owner, context, result.retries,
                                   diagnostic))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::restart_incompatible;
    result.failure_class = ApplicationRunResult::FailureClass::checkpoint_data;
    context.SetDiagnostic(diagnostic);
    return result;
  }

  ScalarPublishedState initial_state;
  ScalarPackedState initial;
  bool state_ready = owner->CopyPublishedTrajectoryState(initial_state) &&
      PackScalarState(initial_state, owner->trajectory_evidence().coordinates,
                      initial);
  if (!ScalarCollectiveAll(state_ready, context))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    context.SetDiagnostic("scalar-IGR CUDA initial state capture failed");
    return result;
  }
  const scalar::StateHashes initial_local = ScalarStateHashes(initial);
  const StateHash initial_interior = ScalarAggregateHash(
      initial_local.interior, context);
  const StateHash initial_trace = ScalarAggregateHash(initial_local.trace,
                                                      context);
  const std::array<StateHash, 2> initial_parts{{initial_interior,
                                                initial_trace}};
  const StateHash initial_hash{ComputeSha256(initial_parts.data(),
                                              sizeof(initial_parts))};
  if (!ScalarCollectiveAll(initial_interior != StateHash{} &&
                               initial_trace != StateHash{} &&
                               initial_hash != StateHash{},
                           context))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    context.SetDiagnostic("scalar-IGR CUDA initial state hashing failed");
    return result;
  }
  if (initial.coordinates.step == 0 && ScalarOutputStep(config, 0))
  {
    DurableGenerationTransactionV1 transaction;
    const bool staged = ScalarBeginTransaction(
                            transaction, output_directory, 0, context,
                            diagnostic) &&
        StageScalarCudaVtk(config, initial, vtk_workspace.get(), context,
                           transaction, diagnostic);
    if (!staged || !ScalarPublish(transaction, context, diagnostic))
    {
      (void)ScalarRevert(transaction, context, diagnostic);
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    }
  }
  std::optional<ValidatedApplicationConfig> checkpoint_writer;
  bool checkpoint_writer_ready = true;
  try { checkpoint_writer = ScalarCheckpointWriterConfig(request, config); }
  catch (...) { checkpoint_writer_ready = false; }
  checkpoint_writer_ready = ScalarCollectiveAll(checkpoint_writer_ready &&
      (!config.checkpoint_out().present() || checkpoint_writer.has_value()),
      context);
  if (!checkpoint_writer_ready)
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::configuration;
    context.SetDiagnostic("scalar-IGR CUDA checkpoint writer preparation failed");
    return result;
  }
  ScalarCheckpoint staged_checkpoint{};
  std::unique_ptr<scalar::ScalarCudaEvidence> staged_evidence;
  bool staging_workspace_ready = true;
  try { staged_evidence = std::make_unique<scalar::ScalarCudaEvidence>(); }
  catch (...) { staging_workspace_ready = false; }
  if (!ScalarCollectiveAll(
          staging_workspace_ready && staged_evidence &&
              PrepareScalarCheckpoint(initial_state, staged_checkpoint),
          context))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    context.SetDiagnostic(
        "scalar-IGR CUDA checkpoint staging workspace allocation failed");
    return result;
  }
  const std::uint64_t stop_step =
      config.run_role() == ApplicationRunRoleV1::split_writer
      ? config.scenario_checkpoint_step() : config.accepted_steps();
  ScalarInjectionAudit injection;
  if (config.injection().mode == FailureInjectionModeV1::prescribed)
    injection.configured_source_rank =
        static_cast<int>(config.injection().source_rank);
  std::string consumed_diagnostic;
  bool injected = false;
  while (result.failure_class == ApplicationRunResult::FailureClass::none &&
         owner->trajectory_evidence().coordinates.step < stop_step)
  {
    const StateCoordinates current = owner->trajectory_evidence().coordinates;
    const std::uint64_t target = current.step + 1;
    const StateCoordinates expected_target{
        static_cast<double>(target) * config.dt(), target,
        current.generation + 1, current.cursor + 1, config.dt()};
    const bool prescribed = !injected &&
        config.injection().mode == FailureInjectionModeV1::prescribed &&
        config.injection().target_step == target;
    const bool local_origin = prescribed &&
        context.rank() == static_cast<int>(config.injection().source_rank);
    const scalar::FailureInjection selected_injection = prescribed
        ? ScalarNumericalInjection(config.injection().phase)
        : scalar::FailureInjection::none;
    const scalar::FailureInjection numerical_injection =
        local_origin ? selected_injection : scalar::FailureInjection::none;
    if (prescribed)
    {
      const bool local_injected = selected_injection ==
              scalar::FailureInjection::none
          ? local_origin
          : numerical_injection != scalar::FailureInjection::none;
      const bool non_origin_injected = local_injected &&
          context.rank() != static_cast<int>(config.injection().source_rank);
      const int local[3] = {local_injected ? 1 : 0,
                            local_injected ? context.rank() + 1 : 0,
                            non_origin_injected ? 1 : 0};
      int global[3] = {};
      const bool reduced = MPI_Allreduce(local, global, 3, MPI_INT, MPI_SUM,
                                         context.communicator()) == MPI_SUCCESS;
      if (!ScalarCollectiveAll(
              reduced && global[0] == 1 && global[2] == 0, context))
      {
        result.failure_class = ApplicationRunResult::FailureClass::backend;
        diagnostic = "scalar-IGR CUDA injection origin consensus failed";
        break;
      }
      injection.observed_origin_rank = global[1] - 1;
      injection.local_injection_count += global[0];
      injection.non_origin_local_injection_count += global[2];
      ++injection.collective_decisions;
    }
    const bool application_injection = prescribed &&
        selected_injection == scalar::FailureInjection::none;
    const bool checkpoint_step = config.checkpoint_out().present() &&
        target == config.scenario_checkpoint_step();
    const bool scheduled_stage = ScalarOutputStep(config, target) ||
        checkpoint_step || target == stop_step;
    typename ScalarOwner::Transaction candidate;
    const scalar::ScalarCudaStatus prepared =
        owner->PreparePersistentTrajectoryTransaction(candidate,
                                                      numerical_injection);
    if (!ScalarCollectiveAll(ScalarAccepted(prepared), context))
    {
      injected = injected || prescribed;
      if (prescribed && result.retries < config.retry_limit())
      {
        ++result.retries;
        continue;
      }
      result.failure_class = prescribed
          ? ApplicationRunResult::FailureClass::injected
          : ApplicationRunResult::FailureClass::scientific;
      const auto& evidence = owner->trajectory_evidence();
      diagnostic = "scalar-IGR CUDA candidate preparation failed status=" +
          std::to_string(static_cast<unsigned>(prepared)) +
          " leaf_status=" +
          std::to_string(static_cast<unsigned>(evidence.status)) +
          " first_failure=" +
          std::to_string(static_cast<unsigned>(evidence.first_failure)) +
          " allocation_events=" +
          std::to_string(evidence.allocations.current_observed_events) +
          " baseline_allocations=" +
          std::to_string(evidence.allocations.baseline_observed_events);
      break;
    }
    DurableGenerationTransactionV1 transaction;
    const bool durable_started = scheduled_stage;
    bool staged = true;
    bool installed = owner->InstallPersistentTrajectoryTransaction(candidate);
    staged = installed;
    if (installed && durable_started)
      staged = ScalarBeginTransaction(
          transaction, output_directory, target, context, diagnostic);
    if (staged && application_injection &&
        config.injection().phase == FailurePhase::AfterRecoveryBeforeCommit &&
        target == scalar_igr_sc25_v3_failure_target_step)
    {
      staged = false;
      consumed_diagnostic = "scalar-IGR CUDA injected pre-stage failure"
          " boundary=post-install-pre-stage"
          " phase=AfterRecoveryBeforeCommit target_step=7"
          " consumed_boundary_count=1 source_rank=" +
          std::to_string(config.injection().source_rank);
      diagnostic = consumed_diagnostic;
      ++injection.consumed_boundary_count;
    }
    ScalarPackedState packed;
    ScalarResourceSidecar sidecar{};
    if (staged && scheduled_stage)
    {
      const bool candidate_staged = owner->StageInstalledTrajectoryTransaction(
          candidate, staged_checkpoint, *staged_evidence);
      staged = ScalarCollectiveAll(
          candidate_staged && PackedFromCheckpoint(staged_checkpoint, packed),
          context);
      if (staged && application_injection &&
          config.injection().phase == FailurePhase::CheckpointStaging)
      {
        staged = false;
        consumed_diagnostic =
            "scalar-IGR CUDA injected checkpoint-staging failure"
            " boundary=checkpoint-staging phase=CheckpointStaging"
            " target_step=10 consumed_boundary_count=1 source_rank=" +
            std::to_string(config.injection().source_rank);
        diagnostic = consumed_diagnostic;
        ++injection.consumed_boundary_count;
      }
      if (staged && checkpoint_step)
        staged = BuildScalarSidecar(
            staged_checkpoint, config.device_id(), context, sidecar,
            diagnostic);
    }
    if (staged && ScalarOutputStep(config, target))
      staged = StageScalarCudaVtk(config, packed, vtk_workspace.get(), context,
                                  transaction, diagnostic);
    if (staged && application_injection &&
        config.injection().phase == FailurePhase::OutputStaging)
    {
      staged = false;
      consumed_diagnostic = "scalar-IGR CUDA injected output-staging failure"
          " boundary=output-staging phase=OutputStaging"
          " target_step=10 consumed_boundary_count=1 source_rank=" +
          std::to_string(config.injection().source_rank);
      diagnostic = consumed_diagnostic;
      ++injection.consumed_boundary_count;
    }
    if (staged && checkpoint_step)
      staged = checkpoint_writer && StageScalarCudaCheckpoint(
          *checkpoint_writer, staged_checkpoint, sidecar, context,
          transaction, diagnostic);
    if (staged && target == stop_step)
    {
      const scalar::StateHashes final_local = ScalarStateHashes(packed);
      const StateHash final_interior = ScalarAggregateHash(
          final_local.interior, context);
      const StateHash final_trace = ScalarAggregateHash(final_local.trace,
                                                        context);
      ScalarResourceRecord resource{};
      bool resource_ready = final_interior != StateHash{} &&
          final_trace != StateHash{} &&
          ScalarGpuRecord(config.device_id(), resource);
      if (target == config.scenario_checkpoint_step())
        resource = sidecar.ranks[static_cast<std::size_t>(context.rank())];
      ApplicationRunResult anticipated = result;
      anticipated.status = RunStatus::committed;
      anticipated.reason = FailureReason::none;
      anticipated.failure_class = ApplicationRunResult::FailureClass::none;
      anticipated.coordinates = packed.coordinates;
      std::vector<std::uint8_t> metadata;
      if (context.rank() == 0 && resource_ready)
      {
        try
        {
          const std::string text = ScalarCudaMetadata(
              config, anticipated, *staged_evidence, initial_hash,
              final_interior, final_trace, injection, resource,
              consumed_diagnostic.empty() ? diagnostic
                                          : consumed_diagnostic);
          metadata.assign(text.begin(), text.end());
        }
        catch (...) { resource_ready = false; }
      }
      resource_ready = ScalarCollectiveAll(resource_ready, context);
      bool metadata_staged = false;
      if (resource_ready)
        metadata_staged = ScalarStageArtifact(
            transaction,
            config.run_role() == ApplicationRunRoleV1::split_writer
                ? "run-metadata-checkpoint.json" : "run-metadata.json",
            metadata, context, diagnostic);
      staged = resource_ready && metadata_staged;
    }
    const bool finalize_ready = ScalarCollectiveAll(
        staged && ScalarTransactionReady(candidate, expected_target), context);
    bool published = finalize_ready;
    if (durable_started && finalize_ready)
      published = ScalarPublish(transaction, context, diagnostic);
    if (published && durable_started && application_injection &&
        config.injection().phase == FailurePhase::AfterRecoveryBeforeCommit &&
        (target == config.scenario_checkpoint_step() ||
         target == config.accepted_steps()))
    {
      published = false;
      consumed_diagnostic =
          "scalar-IGR CUDA injected reversible publication failure"
          " boundary=post-publication-pre-finalize"
          " phase=AfterRecoveryBeforeCommit target_step=" +
          std::to_string(target) + " consumed_boundary_count=1 source_rank=" +
          std::to_string(config.injection().source_rank);
      diagnostic = consumed_diagnostic;
      ++injection.consumed_boundary_count;
    }
    bool committed = false;
    if (published)
    {
      committed = owner->FinalizePersistentTrajectoryTransaction(
          std::move(candidate));
    }
    if (!committed)
    {
      bool durable_reverted = true;
      if (durable_started)
        durable_reverted = ScalarRevert(transaction, context, diagnostic);
      const bool leaf_reverted = candidate.valid() &&
          owner->RevertPersistentTrajectoryTransaction(std::move(candidate));
      if (!ScalarCollectiveAll(durable_reverted && leaf_reverted, context))
      {
        result.failure_class = ApplicationRunResult::FailureClass::backend;
        diagnostic = "scalar-IGR CUDA transaction rollback failed";
        break;
      }
      injected = injected || prescribed;
      if (prescribed && result.retries < config.retry_limit())
      {
        ++result.retries;
        continue;
      }
      result.failure_class = prescribed
          ? ApplicationRunResult::FailureClass::injected
          : ApplicationRunResult::FailureClass::filesystem;
      if (diagnostic.empty())
        diagnostic = "scalar-IGR CUDA durable publication failed";
      break;
    }
    injected = injected || prescribed;
  }
  result.coordinates = owner->trajectory_evidence().coordinates;
  if (result.failure_class == ApplicationRunResult::FailureClass::none)
  {
    result.status = RunStatus::committed;
    result.reason = FailureReason::none;
  }
  else
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
  }
  bool clean = true;
  if (context.rank() == 0 && output_created)
    clean = DiscardDurableDirectoryIfEmpty(output_directory, diagnostic) ==
        DurableIoStatus::ok;
  if (!ScalarCollectiveAll(clean, context))
    result.failure_class = ApplicationRunResult::FailureClass::filesystem;
  context.SetDiagnostic(diagnostic);
  return result;
}

} // namespace

#if !defined(DG_IGR_HDG_SCALAR_IGR_CUDA_RANK2_TRANSLATION_UNIT)
ApplicationRunResult RunScalarIgrCudaRank2Adapter(
    const ApplicationOptionsView&, CudaApplicationContextV1&);

namespace detail
{

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CudaRank1OneGpuBackendTag, const ApplicationOptionsView& request,
    CudaApplicationContextV1& context)
{
  if (request.card_id == scalar_igr_sc25_v3_card_id)
    return RunScalarIgrCuda(request, context);
  return RunCuda<CudaRank1OneGpuRuntimeSelection>(request, context);
}

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CudaRank2OneGpuHostStagedBackendTag,
    const ApplicationOptionsView& request, CudaApplicationContextV1& context)
{
  if (request.card_id == scalar_igr_sc25_v3_card_id)
    return RunScalarIgrCudaRank2Adapter(request, context);
  return RunCuda<CudaRank2OneGpuHostStagedRuntimeSelection>(request, context);
}

} // namespace detail
#else
ApplicationRunResult RunScalarIgrCudaRank2Adapter(
    const ApplicationOptionsView& request, CudaApplicationContextV1& context)
{
  return RunScalarIgrCuda(request, context);
}
#endif

} // namespace dg::hdg::runtime::app

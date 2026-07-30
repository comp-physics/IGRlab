#include "numerics/hdg/runtime/app/solver.hpp"

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include "numerics/hdg/runtime/sha256.hpp"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace dg::hdg::runtime::app
{
namespace
{

namespace cuda = ::dg::hdg::runtime::cuda_w8a;

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
  if (config.injection().mode == FailureInjectionModeV1::prescribed &&
      config.injection().phase == FailurePhase::AfterRecoveryBeforeCommit)
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

std::string InitialVtk(const ValidatedApplicationConfig& config)
{
  const int n = static_cast<int>(config.mesh_n());
  const int points = n * n;
  std::ostringstream output;
  output << std::setprecision(17)
      << "# vtk DataFile Version 3.0\n"
      << "DG-IGR HDG exact initial visualization\nASCII\n"
      << "DATASET POLYDATA\nPOINTS " << points << " double\n";
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      output << (i + 0.5) / n << ' ' << (j + 0.5) / n << " 0\n";
  output << "VERTICES " << points << ' ' << 2 * points << "\n";
  for (int point = 0; point < points; ++point) output << "1 " << point << '\n';
  output << "POINT_DATA " << points
         << "\nSCALARS density double 1\nLOOKUP_TABLE default\n";
  constexpr double pi = 3.141592653589793238462643383279502884;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
    {
      const double x = (i + 0.5) / n;
      const double y = (j + 0.5) / n;
      output << 1.0 + 0.1 * std::sin(2.0 * pi * x) *
                              std::sin(2.0 * pi * y) << '\n';
    }
  return output.str();
}

bool CandidateVtk(const cuda::CudaSerializedCheckpointV1& local,
                  const CudaApplicationContextV1& context,
                  std::string& vtk, std::string& detail)
{
  std::vector<cuda::CudaSerializedCheckpointV1> ranks;
  if (!GatherCheckpoints(local, context, ranks, detail)) return false;
  bool built = true;
  if (context.rank() == 0)
  {
    try
    {
      std::uint64_t dofs = 0;
      for (const auto& rank : ranks)
      {
        if (rank.interior_count % 4 != 0)
        {
          detail = "CUDA component-major checkpoint extent is invalid";
          built = false;
          break;
        }
        dofs += rank.interior_count / 4;
      }
      if (built)
      {
        std::ostringstream output;
        output << std::setprecision(17)
            << "# vtk DataFile Version 3.0\n"
            << "DG-IGR HDG committed coefficient visualization\nASCII\n"
            << "DATASET POLYDATA\nPOINTS " << dofs << " double\n";
        for (std::uint64_t point = 0; point < dofs; ++point)
          output << point << " 0 0\n";
        output << "VERTICES " << dofs << ' ' << 2 * dofs << "\n";
        for (std::uint64_t point = 0; point < dofs; ++point)
          output << "1 " << point << '\n';
        output << "POINT_DATA " << dofs
               << "\nFIELD conservative 1\nU 4 " << dofs << " double\n";
        for (const auto& rank : ranks)
        {
          const std::uint64_t local_dofs = rank.interior_count / 4;
          for (std::uint64_t dof = 0; dof < local_dofs; ++dof)
            output << rank.interior[dof] << ' '
                   << rank.interior[local_dofs + dof] << ' '
                   << rank.interior[2 * local_dofs + dof] << ' '
                   << rank.interior[3 * local_dofs + dof] << '\n';
        }
        vtk = output.str();
        built = output.good();
      }
    }
    catch (...) { built = false; }
  }
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

bool StageRootFile(DurableStagedFileV1& file,
                   const std::filesystem::path& path,
                   const std::vector<std::uint8_t>& bytes,
                   const CudaApplicationContextV1& context,
                   std::string& detail)
{
  bool local = true;
  if (context.rank() == 0)
    local = file.Stage(path, bytes.data(), bytes.size(), detail) ==
        DurableIoStatus::ok;
  return CollectiveAll(local, context.communicator());
}

bool PublishRootFile(DurableStagedFileV1& file,
                     const CudaApplicationContextV1& context,
                     std::string& detail)
{
  bool local = true;
  if (context.rank() == 0 && file.staged())
    local = file.Publish(detail) == DurableIoStatus::ok;
  return CollectiveAll(local, context.communicator());
}

bool RevertTransaction(DurableStagedFileV1& vtk,
                       DurableStagedFileV1& checkpoint,
                       DurableStagedFileV1& metadata,
                       const CudaApplicationContextV1& context,
                       std::string& detail)
{
  bool local = true;
  if (context.rank() == 0)
  {
    local = vtk.Revert(detail) == DurableIoStatus::ok;
    local = checkpoint.Revert(detail) == DurableIoStatus::ok && local;
    local = metadata.Revert(detail) == DurableIoStatus::ok && local;
  }
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
      decoded = ReadDurableFile(
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
  if (config.run_role() != ApplicationRunRoleV1::uninterrupted_reference)
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
      DurableStagedFileV1 initial_file;
      std::vector<std::uint8_t> initial_bytes;
      bool initial_ready = true;
      if (context.rank() == 0)
      {
        try
        {
          const std::string initial = InitialVtk(config);
          initial_bytes.assign(initial.begin(), initial.end());
        }
        catch (...) { initial_ready = false; }
      }
      initial_ready = CollectiveAll(initial_ready, context.communicator()) &&
          StageRootFile(initial_file, output_directory / StepName(0),
                        initial_bytes, context, detail) &&
          PublishRootFile(initial_file, context, detail);
      if (!initial_ready)
      {
        bool cleaned = true;
        if (context.rank() == 0)
          cleaned = initial_file.Revert(detail) == DurableIoStatus::ok;
        (void)CollectiveAll(cleaned, context.communicator());
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
        result.status = RunStatus::terminal_failure;
        result.reason = evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        result.failure_class = ApplicationRunResult::FailureClass::backend;
        detail = "CUDA candidate preparation failed";
        break;
      }
      const bool inject_now = !was_retry &&
          config.injection().mode == FailureInjectionModeV1::prescribed &&
          config.injection().target_step == target;
      const FailurePhase injection_phase = inject_now
          ? config.injection().phase : FailurePhase::Setup;
      DurableStagedFileV1 vtk_file;
      DurableStagedFileV1 checkpoint_file;
      DurableStagedFileV1 metadata_file;
      std::vector<std::uint8_t> vtk_bytes;
      std::vector<std::uint8_t> checkpoint_bytes;
      std::vector<std::uint8_t> metadata_bytes;
      bool staged = injection_phase != FailurePhase::CheckpointStaging &&
          engine->StagePreparedCheckpoint(checkpoint, evidence);
      std::string vtk;
      if (staged && OutputStep(config, target))
        staged = CandidateVtk(checkpoint, context, vtk, detail);
      if (staged && context.rank() == 0 && !vtk.empty())
      {
        try { vtk_bytes.assign(vtk.begin(), vtk.end()); }
        catch (...) { staged = false; }
      }
      staged = CollectiveAll(staged, context.communicator());
      if (staged && OutputStep(config, target))
        staged = StageRootFile(vtk_file, output_directory / StepName(target),
                               vtk_bytes, context, detail);
      if (staged && config.checkpoint_out().present() &&
          target == config.scenario_checkpoint_step())
      {
        staged = checkpoint_writer && EncodeCheckpoint(
            *checkpoint_writer, checkpoint, setup,
            engine->contract()->resources(), context, checkpoint_bytes,
            detail) && StageRootFile(
                checkpoint_file,
                std::string{config.checkpoint_out().identity().view()},
                checkpoint_bytes, context, detail);
      }
      if (injection_phase == FailurePhase::OutputStaging) staged = false;
      if (staged) staged = engine->MarkOutputStaged(checkpoint);
      if (!staged)
      {
        const bool cleaned = RevertTransaction(
            vtk_file, checkpoint_file, metadata_file, context, detail);
        const bool retry_available = cleaned && engine->AbortPrepared(
            checkpoint, FailureReason::backend_failure, evidence);
        if (retry_available && result.retries < config.retry_limit())
        {
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        result.failure_class = inject_now
            ? ApplicationRunResult::FailureClass::injected
            : (cleaned ? ApplicationRunResult::FailureClass::backend
                       : ApplicationRunResult::FailureClass::filesystem);
        if (detail.empty()) detail = "CUDA attempt rolled back before commit";
        break;
      }
      if (!engine->PrepareCommit(checkpoint, evidence))
      {
        const bool cleaned = RevertTransaction(
            vtk_file, checkpoint_file, metadata_file, context, detail);
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
            ? ApplicationRunResult::FailureClass::backend
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
        bool metadata_ready = true;
        if (context.rank() == 0)
        {
          try
          {
            const std::string metadata = Metadata(
                config, anticipated, setup, snapshot.candidate_interior_hash,
                snapshot.candidate_trace_hash, detail);
            metadata_bytes.assign(metadata.begin(), metadata.end());
          }
          catch (...) { metadata_ready = false; }
        }
        metadata_ready = CollectiveAll(metadata_ready,
                                       context.communicator()) &&
            StageRootFile(metadata_file,
                          output_directory /
                              (config.run_role() ==
                                       ApplicationRunRoleV1::split_writer
                                   ? "run-metadata-checkpoint.json"
                                   : "run-metadata.json"),
                          metadata_bytes, context, detail);
        if (!metadata_ready)
        {
          const bool cleaned = RevertTransaction(
              vtk_file, checkpoint_file, metadata_file, context, detail);
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
        const bool cleaned = RevertTransaction(
            vtk_file, checkpoint_file, metadata_file, context, detail);
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
        const bool cleaned = RevertTransaction(
            vtk_file, checkpoint_file, metadata_file, context, detail);
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
            ? ApplicationRunResult::FailureClass::backend
            : ApplicationRunResult::FailureClass::filesystem;
        detail = "CUDA provisional commit failed";
        break;
      }

      if (injection_phase == FailurePhase::RollbackAudit)
      {
        (void)PublishRootFile(vtk_file, context, detail);
        const bool cleaned = RevertTransaction(
            vtk_file, checkpoint_file, metadata_file, context, detail);
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

      const bool published = PublishRootFile(vtk_file, context, detail) &&
          PublishRootFile(checkpoint_file, context, detail) &&
          PublishRootFile(metadata_file, context, detail);
      if (!published)
      {
        const bool cleaned = RevertTransaction(
            vtk_file, checkpoint_file, metadata_file, context, detail);
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
      if (!engine->FinalizeCommit(checkpoint, evidence))
      {
        const bool cleaned = RevertTransaction(
            vtk_file, checkpoint_file, metadata_file, context, detail);
        result.status = RunStatus::terminal_failure;
        result.reason = evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        result.failure_class = cleaned
            ? ApplicationRunResult::FailureClass::backend
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

} // namespace

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank1OneGpuBackendTag, const ApplicationOptionsView& request,
    CudaApplicationContextV1& context)
{
  return RunCuda<CudaRank1OneGpuRuntimeSelection>(request, context);
}

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CudaRank2OneGpuHostStagedBackendTag,
    const ApplicationOptionsView& request, CudaApplicationContextV1& context)
{
  return RunCuda<CudaRank2OneGpuHostStagedRuntimeSelection>(request, context);
}

} // namespace dg::hdg::runtime::app

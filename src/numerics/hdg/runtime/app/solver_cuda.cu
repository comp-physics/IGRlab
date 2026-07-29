#include "numerics/hdg/runtime/app/solver.hpp"

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"

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
  if (context.rank() == 0)
    gathered.resize(static_cast<std::size_t>(context.ranks()));
  const int status = MPI_Gather(
      &local, static_cast<int>(sizeof(local)), MPI_BYTE, gathered.data(),
      static_cast<int>(sizeof(local)), MPI_BYTE, 0, context.communicator());
  if (status != MPI_SUCCESS)
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
  if (context.rank() != 0) return true;
  std::uint64_t dofs = 0;
  for (const auto& rank : ranks)
  {
    if (rank.interior_count % 4 != 0)
    {
      detail = "CUDA component-major checkpoint extent is invalid";
      return false;
    }
    dofs += rank.interior_count / 4;
  }
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
  return output.good();
}

bool PublishCheckpoint(const ValidatedApplicationConfig& config,
                       const cuda::CudaSerializedCheckpointV1& local,
                       const cuda::CudaRunEvidence& setup,
                       const PersistentResourceDescriptor& resources,
                       const CudaApplicationContextV1& context,
                       std::string& detail)
{
  std::vector<cuda::CudaSerializedCheckpointV1> gathered;
  if (!GatherCheckpoints(local, context, gathered, detail)) return false;
  bool published = true;
  if (context.rank() == 0)
  {
    std::array<CheckpointDescriptor, 2> descriptors{};
    std::array<StagedStateViewV1, 2> states{};
    for (int rank = 0; rank < context.ranks(); ++rank)
    {
      const auto& source = gathered[static_cast<std::size_t>(rank)];
      descriptors[static_cast<std::size_t>(rank)] = source.descriptor;
      states[static_cast<std::size_t>(rank)] = {
          static_cast<std::uint32_t>(rank),
          {source.interior.data(), source.interior_count, 4,
           source.interior_count / 4, CoefficientOrderingV1::component_major},
          {source.owned_true_trace.data(), source.owned_true_trace_count, 4,
           source.owned_true_trace_count / 4,
           CoefficientOrderingV1::component_major}};
    }
    const auto captured = CaptureCheckpointPiecesV1(
        descriptors.data(), static_cast<std::size_t>(context.ranks()), config,
        RuntimeIdentity(setup, resources));
    std::vector<std::uint8_t> encoded;
    const CheckpointCodecStatus status = captured
        ? EncodeCheckpointV1(*captured, states.data(),
                             static_cast<std::size_t>(context.ranks()), encoded)
        : CheckpointCodecStatus::invalid_descriptor;
    published = status == CheckpointCodecStatus::ok &&
        PublishDurableFile(
            std::string{config.checkpoint_out().identity().view()},
            encoded.data(), encoded.size(), detail) == DurableIoStatus::ok;
    if (!published && detail.empty()) detail = "CUDA checkpoint encode failed";
  }
  return CollectiveAll(published, context.communicator());
}

template <typename Selection>
bool ProbeLocalIdentity(const cuda::CudaConfig& numerical,
                        cuda::CudaSerializedCheckpointV1& identity,
                        CudaApplicationContextV1& context,
                        std::string& detail)
{
  auto engine = std::make_unique<AppCudaEngine<Selection>>(numerical);
  cuda::CudaRunEvidence setup{};
  if (!engine->Initialize(setup))
  {
    detail = "CUDA restart identity probe initialization failed";
    return false;
  }
  cuda::CudaPrecommitSnapshot snapshot{};
  cuda::CudaStepEvidence evidence{};
  bool ready = engine->PrepareStep(false, snapshot, evidence) &&
      engine->StagePreparedCheckpoint(identity, evidence);
  if (ready)
    (void)engine->AbortPrepared(identity, FailureReason::backend_failure,
                                evidence);
  ready = CollectiveAll(ready, context.communicator());
  if (!ready) detail = "CUDA restart identity probe staging failed";
  return ready;
}

template <typename Selection>
bool RestoreCheckpoint(const ValidatedApplicationConfig& config,
                       const cuda::CudaConfig& numerical,
                       AppCudaEngine<Selection>& engine,
                       const cuda::CudaRunEvidence& setup,
                       CudaApplicationContextV1& context,
                       cuda::CudaStepEvidence& restored,
                       std::string& detail)
{
  cuda::CudaSerializedCheckpointV1 identity{};
  if (!ProbeLocalIdentity<Selection>(numerical, identity, context, detail))
    return false;
  const auto* contract = engine.contract();
  if (contract == nullptr) return false;
  ScenarioCoefficientExtentsV1 extents{};
  if (!DeriveScenarioCoefficientExtentsV1(
          config, static_cast<std::uint32_t>(context.ranks()), extents))
    return false;
  const auto& local_extents = extents.ranks[static_cast<std::size_t>(context.rank())];
  std::vector<double> local_interior(
      static_cast<std::size_t>(local_extents.interior_coefficient_count));
  std::vector<double> local_trace(
      static_cast<std::size_t>(local_extents.trace_coefficient_count));
  std::array<std::vector<double>, 2> interiors, traces;
  std::array<RankStateBufferV1, 2> buffers{};
  DecodedCheckpointMetadataV1 metadata{};
  bool decoded = true;
  if (context.rank() == 0)
  {
    std::vector<std::uint8_t> encoded;
    decoded = ReadDurableFile(
        std::string{config.restart_from().identity().view()},
        64U * 1024U * 1024U, encoded, detail) == DurableIoStatus::ok;
    for (int rank = 0; decoded && rank < context.ranks(); ++rank)
    {
      const auto& rank_extents = extents.ranks[static_cast<std::size_t>(rank)];
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
    if (!decoded && detail.empty()) detail = "CUDA checkpoint decode rejected";
  }
  if (!CollectiveAll(decoded, context.communicator())) return false;
  if (MPI_Bcast(&metadata.descriptor, static_cast<int>(sizeof(metadata.descriptor)),
                MPI_BYTE, 0, context.communicator()) != MPI_SUCCESS)
    return false;
  std::array<int, 2> interior_counts{}, interior_offsets{};
  std::array<int, 2> trace_counts{}, trace_offsets{};
  std::vector<double> flat_interior, flat_trace;
  if (context.rank() == 0)
  {
    int io = 0, to = 0;
    for (int rank = 0; rank < context.ranks(); ++rank)
    {
      const std::size_t index = static_cast<std::size_t>(rank);
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
  if (MPI_Scatterv(flat_interior.data(), interior_counts.data(),
                   interior_offsets.data(), MPI_DOUBLE, local_interior.data(),
                   static_cast<int>(local_interior.size()), MPI_DOUBLE, 0,
                   context.communicator()) != MPI_SUCCESS ||
      MPI_Scatterv(flat_trace.data(), trace_counts.data(), trace_offsets.data(),
                   MPI_DOUBLE, local_trace.data(),
                   static_cast<int>(local_trace.size()), MPI_DOUBLE, 0,
                   context.communicator()) != MPI_SUCCESS)
    return false;
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
  checkpoint.candidate_frozen = true;
  checkpoint.checkpoint_staged = true;
  checkpoint.output_staged = true;
  checkpoint.successful_commit = true;
  return engine.Restore(checkpoint, restored);
}

std::string Metadata(const ValidatedApplicationConfig& config,
                     const ApplicationRunResult& result,
                     const cuda::CudaRunEvidence& setup,
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
  AtomicOutputDirectory output;
  bool output_ready = context.rank() != 0 ||
      output.Begin(std::string{config.output_directory().identity().view()},
                   detail) == DurableIoStatus::ok;
  if (!CollectiveAll(output_ready, context.communicator()))
  {
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
    if (result.failure_class == ApplicationRunResult::FailureClass::none &&
        config.run_role() == ApplicationRunRoleV1::split_reader)
    {
      cuda::CudaStepEvidence restored{};
      if (!RestoreCheckpoint<Selection>(config, numerical, *engine, setup,
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
        engine->contract()->coordinates().step == 0 && context.rank() == 0)
    {
      if (output.Write(StepName(0), InitialVtk(config), detail) !=
          DurableIoStatus::ok)
      {
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = ApplicationRunResult::FailureClass::filesystem;
      }
    }
    const bool initial_output = CollectiveAll(
        result.failure_class == ApplicationRunResult::FailureClass::none,
        context.communicator());
    if (!initial_output && result.failure_class ==
            ApplicationRunResult::FailureClass::none)
    {
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = ApplicationRunResult::FailureClass::filesystem;
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
      const bool app_injection = config.injection().mode ==
              FailureInjectionModeV1::prescribed &&
          config.injection().target_step == target &&
          config.injection().phase != FailurePhase::AfterRecoveryBeforeCommit;
      bool staged = !app_injection &&
          engine->StagePreparedCheckpoint(checkpoint, evidence);
      std::string vtk;
      if (staged && OutputStep(config, target))
        staged = CandidateVtk(checkpoint, context, vtk, detail);
      if (staged) staged = engine->MarkOutputStaged(checkpoint);
      if (!staged)
      {
        const bool retry_available = engine->AbortPrepared(
            checkpoint, FailureReason::backend_failure, evidence);
        if (retry_available && result.retries < config.retry_limit())
        {
          ++result.retries;
          retry = true;
          continue;
        }
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = config.injection().mode ==
                FailureInjectionModeV1::prescribed
            ? ApplicationRunResult::FailureClass::injected
            : ApplicationRunResult::FailureClass::backend;
        detail = "CUDA attempt rolled back before commit";
        break;
      }
      if (!engine->PrepareCommit(checkpoint, evidence) ||
          !engine->CommitPrepared(checkpoint, evidence) ||
          !engine->FinalizeCommit(checkpoint, evidence))
      {
        result.status = RunStatus::terminal_failure;
        result.reason = evidence.contract_reason == FailureReason::none
            ? FailureReason::backend_failure : evidence.contract_reason;
        result.failure_class = ApplicationRunResult::FailureClass::backend;
        detail = "CUDA commit pipeline failed";
        break;
      }
      if (OutputStep(config, target) && context.rank() == 0 &&
          output.Write(StepName(target), vtk, detail) != DurableIoStatus::ok)
      {
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = ApplicationRunResult::FailureClass::filesystem;
      }
      if (!CollectiveAll(result.failure_class ==
                             ApplicationRunResult::FailureClass::none,
                         context.communicator()))
      {
        if (result.failure_class == ApplicationRunResult::FailureClass::none)
        {
          result.status = RunStatus::terminal_failure;
          result.reason = FailureReason::backend_failure;
          result.failure_class = ApplicationRunResult::FailureClass::filesystem;
        }
        break;
      }
      if (config.checkpoint_out().present() &&
          target == config.scenario_checkpoint_step())
      {
        if (!PublishCheckpoint(config, checkpoint, setup,
                               engine->contract()->resources(), context,
                               detail))
        {
          result.status = RunStatus::terminal_failure;
          result.reason = FailureReason::backend_failure;
          result.failure_class = ApplicationRunResult::FailureClass::filesystem;
          break;
        }
      }
    }
    if (engine->contract() != nullptr)
      result.coordinates = engine->contract()->coordinates();
    if (result.failure_class == ApplicationRunResult::FailureClass::none)
    {
      result.status = RunStatus::committed;
      result.reason = FailureReason::none;
    }
    bool metadata_ready = true;
    if (context.rank() == 0)
      metadata_ready = output.Write(
          "run-metadata.json", Metadata(config, result, setup, detail), detail) ==
              DurableIoStatus::ok &&
          output.Publish(detail) == DurableIoStatus::ok;
    if (!CollectiveAll(metadata_ready, context.communicator()))
    {
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    }
  }
  catch (const std::exception& error)
  {
    detail = error.what();
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
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

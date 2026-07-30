#include "numerics/hdg/runtime/app/solver.hpp"

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "numerics/hdg/runtime/sha256.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace dg::hdg::runtime::app
{
namespace
{

using cpu::CpuCanonicalEntry;
using cpu::CpuCheckpoint;
using cpu::CpuCoefficientView;
using cpu::CpuFailureInjection;
using cpu::CpuFailurePhase;
using cpu::CpuPersistentConfig;
using cpu::CpuRunEvidence;
using cpu::CpuStateSnapshot;

CpuFailurePhase ToCpuPhase(const FailurePhase phase) noexcept
{
  switch (phase)
  {
    case FailurePhase::Assembly: return CpuFailurePhase::assembly;
    case FailurePhase::Condensation: return CpuFailurePhase::condensation;
    case FailurePhase::TraceSolve: return CpuFailurePhase::trace_solve;
    case FailurePhase::Recovery: return CpuFailurePhase::recovery;
    case FailurePhase::Diagnostics: return CpuFailurePhase::diagnostics;
    case FailurePhase::CheckpointStaging:
      return CpuFailurePhase::checkpoint_staging;
    case FailurePhase::OutputStaging: return CpuFailurePhase::output_staging;
    case FailurePhase::AfterRecoveryBeforeCommit:
      return CpuFailurePhase::after_recovery_before_commit;
    case FailurePhase::Setup: return CpuFailurePhase::setup;
    case FailurePhase::Commit: return CpuFailurePhase::commit;
    case FailurePhase::RollbackAudit: return CpuFailurePhase::rollback_audit;
    case FailurePhase::TerminalEvidence:
      return CpuFailurePhase::terminal_evidence;
  }
  return CpuFailurePhase::terminal_evidence;
}

constexpr std::string_view mfem_pin =
    "d9d6526cc1749980a2ba1da16e2c1ca1e07d82ec";

struct AppCpuSerialSelection : CpuSerialRuntimeSelection
{
  static constexpr std::string_view experiment_id =
      conventional_persistent_card_id;
};

struct AppCpuRank2Selection : CpuRank2RuntimeSelection
{
  static constexpr std::string_view experiment_id =
      conventional_persistent_card_id;
};

bool CollectiveAll(const bool local, const MPI_Comm communicator) noexcept
{
  const int value = local ? 1 : 0;
  int reduced = 0;
  return MPI_Allreduce(&value, &reduced, 1, MPI_INT, MPI_MIN,
                       communicator) == MPI_SUCCESS && reduced == 1;
}

std::string Hex(const StateHash& hash)
{
  static constexpr char digits[] = "0123456789abcdef";
  std::string text(64, '0');
  for (std::size_t index = 0; index < hash.bytes.size(); ++index)
  {
    text[2 * index] = digits[hash.bytes[index] >> 4U];
    text[2 * index + 1] = digits[hash.bytes[index] & 0xfU];
  }
  return text;
}

bool OutputStep(const ValidatedApplicationConfig& config,
                const std::uint64_t step) noexcept
{
  return std::find(config.output_steps().begin(), config.output_steps().end(),
                   step) != config.output_steps().end();
}

std::string StepName(const std::uint64_t step)
{
  std::ostringstream name;
  name << "state-step-" << std::setw(6) << std::setfill('0') << step
       << ".vtk";
  return name.str();
}

template <typename Adapter>
bool GatherCanonical(const CpuStateSnapshot& snapshot,
                     const Adapter& adapter,
                     const bool candidate,
                     const MpiApplicationContextV1& context,
                     std::vector<CpuCanonicalEntry>& global,
                     std::string& detail)
{
  const std::size_t local_capacity = snapshot.interior.size + snapshot.trace.size;
  std::vector<CpuCanonicalEntry> local;
  std::size_t local_written = 0;
  bool exported = false;
  try
  {
    local.resize(local_capacity);
    exported = (candidate
        ? adapter.ExportCanonicalCandidate(local.data(), local.size(),
                                           local_written)
        : adapter.ExportCanonicalCommitted(local.data(), local.size(),
                                           local_written)) &&
        local_written == local.size();
  }
  catch (...) { exported = false; }
  if (!CollectiveAll(exported, context.communicator()))
  {
    detail = "canonical committed-state export failed";
    return false;
  }
  const bool local_range = local_written <= static_cast<std::size_t>(
      std::numeric_limits<int>::max() / sizeof(CpuCanonicalEntry));
  if (!CollectiveAll(local_range, context.communicator()))
  {
    detail = "canonical export exceeds MPI count range";
    return false;
  }
  const int local_bytes = static_cast<int>(local_written *
                                           sizeof(CpuCanonicalEntry));
  std::array<int, 2> byte_counts{};
  const bool count_gathered = MPI_Gather(
      &local_bytes, 1, MPI_INT, byte_counts.data(), 1, MPI_INT, 0,
      context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(count_gathered, context.communicator()))
  {
    detail = "MPI_Gather canonical counts failed";
    return false;
  }
  std::array<int, 2> displacements{};
  int total_bytes = 0;
  bool root_valid = true;
  if (context.rank() == 0)
  {
    for (int rank = 0; rank < context.ranks(); ++rank)
    {
      displacements[static_cast<std::size_t>(rank)] = total_bytes;
      if (byte_counts[static_cast<std::size_t>(rank)] < 0 ||
          total_bytes > std::numeric_limits<int>::max() -
              byte_counts[static_cast<std::size_t>(rank)])
      {
        detail = "canonical aggregate exceeds MPI count range";
        root_valid = false;
        break;
      }
      total_bytes += byte_counts[static_cast<std::size_t>(rank)];
    }
    if (root_valid)
    {
      try
      {
        global.resize(static_cast<std::size_t>(total_bytes) /
                      sizeof(CpuCanonicalEntry));
      }
      catch (...) { root_valid = false; }
    }
  }
  const bool range_ok = CollectiveAll(
      context.rank() != 0 ||
          (root_valid && total_bytes %
              static_cast<int>(sizeof(CpuCanonicalEntry)) == 0),
      context.communicator());
  if (!range_ok) { return false; }
  const int gather = MPI_Gatherv(
      local.data(), local_bytes, MPI_BYTE, global.data(), byte_counts.data(),
      displacements.data(), MPI_BYTE, 0, context.communicator());
  if (!CollectiveAll(gather == MPI_SUCCESS, context.communicator()))
  {
    detail = "MPI_Gatherv canonical state failed";
    return false;
  }
  bool sorted = true;
  if (context.rank() == 0)
  {
    try
    {
      std::sort(global.begin(), global.end(),
                [](const auto& left, const auto& right)
                { return left.global_id < right.global_id; });
      for (std::size_t index = 1; index < global.size(); ++index)
      {
        if (global[index - 1].global_id >= global[index].global_id)
        {
          detail = "canonical global IDs are not unique";
          sorted = false;
          break;
        }
      }
    }
    catch (...) { sorted = false; }
  }
  return CollectiveAll(sorted, context.communicator());
}

bool MakeVtk(const ValidatedApplicationConfig& config,
             const std::vector<CpuCanonicalEntry>& entries,
             std::string& vtk,
             std::string& detail)
{
  try
  {
    mfem::Mesh original = mfem::Mesh::MakeCartesian2D(
        static_cast<int>(config.mesh_n()), static_cast<int>(config.mesh_n()),
        mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    std::vector<mfem::Vector> translations(2, mfem::Vector(2));
    translations[0][0] = 1.0;
    translations[0][1] = 0.0;
    translations[1][0] = 0.0;
    translations[1][1] = 1.0;
    mfem::Mesh mesh = mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations));
    mfem::L2_FECollection collection(
        config.polynomial_order(), 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::FiniteElementSpace space(&mesh, &collection, 4,
                                   mfem::Ordering::byNODES);
    mfem::GridFunction field(&space);
    field = 0.0;
    constexpr std::uint64_t trace_prefix = std::uint64_t{1} << 63U;
    std::size_t interior = 0;
    for (const CpuCanonicalEntry& entry : entries)
    {
      if ((entry.global_id & trace_prefix) != 0) { continue; }
      if (entry.global_id >= static_cast<std::uint64_t>(field.Size()))
      {
        detail = "canonical interior ID exceeds reconstructed L2 space";
        return false;
      }
      field[static_cast<int>(entry.global_id)] = entry.value;
      ++interior;
    }
    if (interior != static_cast<std::size_t>(field.Size()))
    {
      detail = "canonical interior coverage is incomplete";
      return false;
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

template <typename Adapter>
bool StageStateVtk(const ValidatedApplicationConfig& config,
                   const CpuStateSnapshot& snapshot,
                   const bool candidate,
                   const Adapter& adapter,
                   const std::filesystem::path& output_directory,
                   const MpiApplicationContextV1& context,
                   DurableStagedFileV1& staged,
                   std::vector<std::uint8_t>& payload,
                   std::string& detail)
{
  std::vector<CpuCanonicalEntry> global;
  if (!GatherCanonical(snapshot, adapter, candidate, context, global, detail))
  {
    return false;
  }
  bool staged_ok = true;
  if (context.rank() == 0)
  {
    try
    {
      std::string vtk;
      staged_ok = MakeVtk(config, global, vtk, detail);
      if (staged_ok)
      {
        payload.assign(vtk.begin(), vtk.end());
        staged_ok = staged.Stage(
            output_directory / StepName(snapshot.coordinates.step),
            payload.data(), payload.size(), detail) == DurableIoStatus::ok;
      }
    }
    catch (...)
    {
      staged_ok = false;
      detail = "VTK payload staging allocation failed";
    }
  }
  return CollectiveAll(staged_ok, context.communicator());
}

PersistentResourceDescriptor ResourcesFromEvidence(
    const ValidatedApplicationConfig& config,
    const CpuRunEvidence& evidence) noexcept
{
  PersistentResourceDescriptor resources{};
  resources.mfem_pin = StableIdentity::Copy(mfem_pin);
  resources.mesh_hash = evidence.mesh_hash;
  resources.partition_hash = evidence.partition_hash;
  resources.topology_hash = evidence.topology_hash;
  resources.rank_count = static_cast<std::uint32_t>(evidence.ranks);
  resources.polynomial_order = evidence.polynomial_order;
  resources.layout = StableIdentity::Copy(config.layout_id().view());
  resources.reuse = {true, true, true, false, 1, 1, 1, 1};
  return resources;
}

CheckpointRuntimeIdentityV1 RuntimeIdentity(
    const CpuRunEvidence& evidence) noexcept
{
  CheckpointRuntimeIdentityV1 identity{};
  identity.physical_gpu_uuid = StableIdentity::Copy(no_physical_gpu_uuid_v1);
  identity.ownership_hash = evidence.partition_hash;
  return identity;
}

bool CpuContractPayloadHash(const double* values, const std::size_t count,
                            const MpiApplicationContextV1& context,
                            StateHash& aggregate) noexcept
{
  if ((count != 0 && values == nullptr) ||
      (context.ranks() != 1 && context.ranks() != 2))
    return false;
  std::array<std::uint8_t, 40> local{};
  const std::uint64_t size = static_cast<std::uint64_t>(count);
  for (unsigned int byte = 0; byte < 8; ++byte)
    local[byte] = static_cast<std::uint8_t>(size >> (8U * byte));
  const Sha256Digest local_hash = ComputeSha256(values, count * sizeof(double));
  std::copy(local_hash.begin(), local_hash.end(), local.begin() + 8);
  std::array<std::array<std::uint8_t, 40>, 2> rank_ordered{};
  if (MPI_Allgather(local.data(), static_cast<int>(local.size()), MPI_BYTE,
                    rank_ordered.data(), static_cast<int>(local.size()),
                    MPI_BYTE, context.communicator()) != MPI_SUCCESS)
    return false;
  aggregate.bytes = ComputeSha256(
      rank_ordered.data(),
      static_cast<std::size_t>(context.ranks()) * local.size());
  return aggregate != StateHash{};
}

bool GatherDoubles(const double* local, const std::size_t local_count,
                   const MpiApplicationContextV1& context,
                   std::array<int, 2>& counts,
                   std::array<int, 2>& displacements,
                   std::vector<double>& gathered,
                   std::string& detail)
{
  const bool local_valid =
      local_count <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
      (local_count == 0 || local != nullptr);
  if (!CollectiveAll(local_valid, context.communicator()))
  {
    detail = "rank-local checkpoint count exceeds MPI range";
    return false;
  }
  const int count = static_cast<int>(local_count);
  const bool counts_gathered = MPI_Gather(
      &count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
      context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(counts_gathered, context.communicator()))
  {
    detail = "MPI_Gather checkpoint counts failed";
    return false;
  }
  bool root_valid = true;
  if (context.rank() == 0)
  {
    int total = 0;
    for (int rank = 0; rank < context.ranks(); ++rank)
    {
      displacements[static_cast<std::size_t>(rank)] = total;
      const int next = counts[static_cast<std::size_t>(rank)];
      if (next < 0 || total > std::numeric_limits<int>::max() - next)
      {
        root_valid = false;
        break;
      }
      total += next;
    }
    if (root_valid)
    {
      try { gathered.resize(static_cast<std::size_t>(total)); }
      catch (...) { root_valid = false; }
    }
  }
  if (!CollectiveAll(root_valid, context.communicator()))
  {
    detail = "aggregate checkpoint count exceeds MPI range";
    return false;
  }
  const bool gathered_ok = MPI_Gatherv(
      local, count, MPI_DOUBLE, gathered.data(), counts.data(),
      displacements.data(), MPI_DOUBLE, 0,
      context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(gathered_ok, context.communicator()))
  {
    detail = "MPI_Gatherv checkpoint payload failed";
    return false;
  }
  return true;
}

bool StageCheckpoint(const ValidatedApplicationConfig& config,
                     const CheckpointDescriptor& local_descriptor,
                     const CpuCoefficientView& local_interior,
                     const CpuCoefficientView& local_trace,
                     const CpuRunEvidence& evidence,
                     const MpiApplicationContextV1& context,
                     DurableStagedFileV1& staged,
                     std::vector<std::uint8_t>& encoded,
                     std::string& detail)
{
  static_assert(std::is_trivially_copyable<CheckpointDescriptor>::value,
                "rank checkpoint descriptor must be MPI-byte transportable");
  std::array<CheckpointDescriptor, 2> descriptors{};
  const int descriptor_status = MPI_Gather(
      &local_descriptor, static_cast<int>(sizeof(CheckpointDescriptor)),
      MPI_BYTE, descriptors.data(), static_cast<int>(sizeof(CheckpointDescriptor)),
      MPI_BYTE, 0, context.communicator());
  if (!CollectiveAll(descriptor_status == MPI_SUCCESS,
                     context.communicator()))
  {
    detail = "MPI_Gather checkpoint descriptors failed";
    return false;
  }
  std::array<int, 2> interior_counts{}, interior_displacements{};
  std::array<int, 2> trace_counts{}, trace_displacements{};
  std::vector<double> interior, trace;
  if (!GatherDoubles(local_interior.data, local_interior.size, context,
                     interior_counts, interior_displacements, interior,
                     detail) ||
      !GatherDoubles(local_trace.data, local_trace.size, context, trace_counts,
                     trace_displacements, trace, detail))
  {
    return false;
  }
  bool staged_ok = true;
  if (context.rank() == 0)
  {
    try
    {
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        descriptors[index].interior_hash.bytes = ComputeSha256(
            interior.data() + interior_displacements[index],
            static_cast<std::size_t>(interior_counts[index]) *
                sizeof(double));
        descriptors[index].trace_hash.bytes = ComputeSha256(
            trace.data() + trace_displacements[index],
            static_cast<std::size_t>(trace_counts[index]) * sizeof(double));
      }
      const auto captured = CaptureCheckpointPiecesV1(
          descriptors.data(), static_cast<std::size_t>(context.ranks()), config,
          RuntimeIdentity(evidence));
      std::array<StagedStateViewV1, 2> ranks{};
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        ranks[index].rank = static_cast<std::uint32_t>(rank);
        ranks[index].interior = {
            interior.data() + interior_displacements[index],
            static_cast<std::uint64_t>(interior_counts[index]), 4,
            static_cast<std::uint64_t>(interior_counts[index] / 4),
            CoefficientOrderingV1::component_major};
        ranks[index].owned_true_trace = {
            trace.data() + trace_displacements[index],
            static_cast<std::uint64_t>(trace_counts[index]), 4,
            static_cast<std::uint64_t>(trace_counts[index] / 4),
            CoefficientOrderingV1::component_major};
      }
      const CheckpointCodecStatus codec = captured
          ? EncodeCheckpointV1(*captured, ranks.data(),
                               static_cast<std::size_t>(context.ranks()), encoded)
          : CheckpointCodecStatus::invalid_descriptor;
      staged_ok = codec == CheckpointCodecStatus::ok &&
          staged.Stage(std::string{config.checkpoint_out().identity().view()},
                       encoded.data(), encoded.size(), detail) ==
              DurableIoStatus::ok;
      if (!staged_ok && detail.empty())
      {
        detail = "checkpoint encode failed with status " +
            std::to_string(static_cast<unsigned>(codec));
      }
    }
    catch (...)
    {
      staged_ok = false;
      detail = "checkpoint encode allocation failed";
    }
  }
  return CollectiveAll(staged_ok, context.communicator());
}

bool RestoreCheckpoint(const ValidatedApplicationConfig& config,
                       const CpuRunEvidence& evidence,
                       const MpiApplicationContextV1& context,
                       CpuCheckpoint& checkpoint,
                       std::string& detail)
{
  ScenarioCoefficientExtentsV1 extents{};
  if (!DeriveScenarioCoefficientExtentsV1(
          config, static_cast<std::uint32_t>(context.ranks()), extents))
  {
    detail = "restart coefficient extents are invalid";
    return false;
  }
  std::array<std::vector<double>, 2> interiors, traces;
  std::array<RankStateBufferV1, 2> buffers{};
  DecodedCheckpointMetadataV1 metadata{};
  bool decoded = true;
  if (context.rank() == 0)
  {
    try
    {
      std::vector<std::uint8_t> encoded;
      decoded = ReadDurableFile(
                    std::string{config.restart_from().identity().view()},
                    64U * 1024U * 1024U, encoded, detail) ==
          DurableIoStatus::ok;
      for (int rank = 0; decoded && rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        const auto& rank_extents = extents.ranks[index];
        interiors[index].resize(
            static_cast<std::size_t>(rank_extents.interior_coefficient_count));
        traces[index].resize(
            static_cast<std::size_t>(rank_extents.trace_coefficient_count));
        buffers[index].rank = static_cast<std::uint32_t>(rank);
        buffers[index].interior = {
            interiors[index].data(), interiors[index].size(),
            interiors[index].size(), 4,
            rank_extents.interior_dofs_per_component,
            CoefficientOrderingV1::component_major};
        buffers[index].owned_true_trace = {
            traces[index].data(), traces[index].size(), traces[index].size(), 4,
            rank_extents.trace_dofs_per_component,
            CoefficientOrderingV1::component_major};
      }
      if (decoded)
      {
        const CheckpointCodecStatus status = DecodeCheckpointV1(
            {encoded.data(), encoded.size()}, config,
            ResourcesFromEvidence(config, evidence), RuntimeIdentity(evidence),
            buffers.data(), static_cast<std::size_t>(context.ranks()), metadata);
        decoded = status == CheckpointCodecStatus::ok;
        if (!decoded)
        {
          detail = "checkpoint decode/compatibility rejected with status " +
              std::to_string(static_cast<unsigned>(status));
        }
      }
    }
    catch (...)
    {
      decoded = false;
      detail = "checkpoint decode allocation failed";
    }
  }
  if (!CollectiveAll(decoded, context.communicator())) { return false; }
  const bool descriptor_broadcast = MPI_Bcast(
      &metadata.descriptor, static_cast<int>(sizeof(CheckpointDescriptor)),
      MPI_BYTE, 0, context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(descriptor_broadcast, context.communicator()))
  {
    detail = "MPI_Bcast restart descriptor failed";
    return false;
  }
  const auto& local_extents = extents.ranks[static_cast<std::size_t>(context.rank())];
  std::vector<double> local_interior;
  std::vector<double> local_trace;
  bool local_allocated = true;
  try
  {
    local_interior.resize(
        static_cast<std::size_t>(local_extents.interior_coefficient_count));
    local_trace.resize(
        static_cast<std::size_t>(local_extents.trace_coefficient_count));
  }
  catch (...) { local_allocated = false; }
  if (!CollectiveAll(local_allocated, context.communicator()))
  {
    detail = "restart rank-local allocation failed";
    return false;
  }
  std::array<int, 2> interior_counts{}, interior_displacements{};
  std::array<int, 2> trace_counts{}, trace_displacements{};
  bool payload_prepared = true;
  if (context.rank() == 0)
  {
    try
    {
      int interior_offset = 0;
      int trace_offset = 0;
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        if (interiors[index].size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max() - interior_offset) ||
            traces[index].size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max() - trace_offset))
        {
          payload_prepared = false;
          break;
        }
        interior_counts[index] = static_cast<int>(interiors[index].size());
        interior_displacements[index] = interior_offset;
        trace_counts[index] = static_cast<int>(traces[index].size());
        trace_displacements[index] = trace_offset;
        interior_offset += interior_counts[index];
        trace_offset += trace_counts[index];
      }
      std::vector<double> flat_interior, flat_trace;
      if (payload_prepared)
      {
        flat_interior.reserve(static_cast<std::size_t>(interior_offset));
        flat_trace.reserve(static_cast<std::size_t>(trace_offset));
        for (int rank = 0; rank < context.ranks(); ++rank)
        {
          flat_interior.insert(flat_interior.end(), interiors[rank].begin(),
                               interiors[rank].end());
          flat_trace.insert(flat_trace.end(), traces[rank].begin(),
                            traces[rank].end());
        }
        interiors[0] = std::move(flat_interior);
        traces[0] = std::move(flat_trace);
      }
    }
    catch (...) { payload_prepared = false; }
  }
  if (!CollectiveAll(payload_prepared, context.communicator()))
  {
    detail = "restart aggregate payload preparation failed";
    return false;
  }
  const int interior_scatter = MPI_Scatterv(
      interiors[0].data(), interior_counts.data(), interior_displacements.data(),
      MPI_DOUBLE, local_interior.data(), static_cast<int>(local_interior.size()),
      MPI_DOUBLE, 0, context.communicator());
  const int trace_scatter = MPI_Scatterv(
      traces[0].data(), trace_counts.data(), trace_displacements.data(),
      MPI_DOUBLE, local_trace.data(), static_cast<int>(local_trace.size()),
      MPI_DOUBLE, 0, context.communicator());
  if (!CollectiveAll(interior_scatter == MPI_SUCCESS &&
                         trace_scatter == MPI_SUCCESS,
                     context.communicator()))
  {
    detail = "MPI_Scatterv restart payload failed";
    return false;
  }
  CheckpointDescriptor runtime_descriptor = metadata.descriptor;
  const bool interior_hash_ready = CpuContractPayloadHash(
      local_interior.data(), local_interior.size(), context,
      runtime_descriptor.interior_hash);
  const bool trace_hash_ready = CpuContractPayloadHash(
      local_trace.data(), local_trace.size(), context,
      runtime_descriptor.trace_hash);
  const bool runtime_hashes_ready = interior_hash_ready && trace_hash_ready;
  if (!CollectiveAll(runtime_hashes_ready, context.communicator()))
  {
    detail = "rank-common checkpoint hashes could not be reconstructed";
    return false;
  }
  const bool imported = ImportCpuCheckpoint(
      runtime_descriptor, local_interior.data(), local_interior.size(),
      local_trace.data(), local_trace.size(), checkpoint);
  if (!CollectiveAll(imported, context.communicator()))
  {
    detail = "rank-local checkpoint import failed";
    return false;
  }
  return true;
}

std::optional<ValidatedApplicationConfig> CheckpointWriterConfig(
    const ApplicationOptionsView& request,
    const ValidatedApplicationConfig& config)
{
  if (!config.checkpoint_out().present()) { return std::nullopt; }
  if (config.run_role() != ApplicationRunRoleV1::uninterrupted_reference)
  {
    return config;
  }
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

bool PublishRootFile(DurableStagedFileV1& file,
                     const MpiApplicationContextV1& context,
                     std::string& detail)
{
  bool local = true;
  if (context.rank() == 0 && file.staged())
  {
    local = file.Publish(detail) == DurableIoStatus::ok;
  }
  return CollectiveAll(local, context.communicator());
}

bool RevertTransaction(DurableStagedFileV1& vtk,
                       DurableStagedFileV1& checkpoint,
                       DurableStagedFileV1& metadata,
                       const MpiApplicationContextV1& context,
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

std::string Metadata(const ValidatedApplicationConfig& config,
                     const ApplicationRunResult& result,
                     const CpuRunEvidence& evidence,
                     const StateHash& initial_hash,
                     const StateHash& final_hash,
                     const std::string& detail)
{
  std::ostringstream output;
  output << std::setprecision(17)
      << "{\n"
      << "  \"schema\": \"dg-igr-hdg-run-metadata.v1\",\n"
      << "  \"card\": \"" << config.card_id().view() << "\",\n"
      << "  \"case\": \"" << config.case_id().view() << "\",\n"
      << "  \"scenario\": \"" << config.scenario_id().view() << "\",\n"
      << "  \"backend\": \"" << config.tuple().backend.view() << "\",\n"
      << "  \"layout\": \"" << config.layout_id().view() << "\",\n"
      << "  \"gamma\": " << config.gamma() << ",\n"
      << "  \"tau\": " << config.tau() << ",\n"
      << "  \"dt\": " << config.dt() << ",\n"
      << "  \"requested_steps\": " << config.accepted_steps() << ",\n"
      << "  \"committed_step\": " << result.coordinates.step << ",\n"
      << "  \"committed_time\": " << result.coordinates.time << ",\n"
      << "  \"status\": \"" << ToString(result.status) << "\",\n"
      << "  \"reason\": \"" << ToString(result.reason) << "\",\n"
      << "  \"retries\": " << result.retries << ",\n"
      << "  \"topology_builds\": " << evidence.topology_builds << ",\n"
      << "  \"post_baseline_allocations\": "
      << evidence.storage.allocation_events_after_baseline << ",\n"
      << "  \"initial_state_sha256\": \"" << Hex(initial_hash) << "\",\n"
      << "  \"final_state_sha256\": \"" << Hex(final_hash) << "\",\n"
      << "  \"diagnostic\": \"" << detail << "\"\n"
      << "}\n";
  return output.str();
}

template <typename Selection>
ApplicationRunResult RunCpu(const ApplicationOptionsView& request,
                            MpiApplicationContextV1& context)
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
    return result;
  }
  const ValidatedApplicationConfig& config = *validation.config();
  if (context.ranks() !=
      static_cast<int>(config.deferred_runtime().required_mpi_rank_count))
  {
    context.SetDiagnostic("MPI rank count does not match selected backend");
    result.failure_class = ApplicationRunResult::FailureClass::configuration;
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
    context.SetDiagnostic(detail.empty() ? "output staging failed" : detail);
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    return result;
  }

  cpu::PersistentSmoothEulerCpuAdapter<Selection> adapter;
  CpuPersistentConfig numerical = config.scenario() == ApplicationScenarioV1::smoke
      ? cpu::MakeCpuSmokeConfig()
      : cpu::MakeCpuFunctionalConfig(static_cast<int>(config.mesh_n()));
  // The persistent numerical core must retain its one-retry rollback seam;
  // the application contract independently decides whether a run may use it.
  numerical.allow_one_retry = true;
  const bool initialized = adapter.Initialize(numerical);
  if (!CollectiveAll(initialized, context.communicator()))
  {
    detail = "CPU backend initialization failed";
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
  }
  else
  {
    result.status = RunStatus::ready;
    result.reason = FailureReason::none;
    result.failure_class = ApplicationRunResult::FailureClass::none;
  }

  StateHash initial_hash{};
  const std::optional<ValidatedApplicationConfig> checkpoint_writer =
      CheckpointWriterConfig(request, config);
  if (config.checkpoint_out().present() && !checkpoint_writer)
  {
    detail = "checkpoint writer role normalization failed";
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::configuration;
  }
  if (initialized)
  {
    if (result.failure_class == ApplicationRunResult::FailureClass::none &&
        config.run_role() == ApplicationRunRoleV1::split_reader)
    {
      CpuCheckpoint checkpoint;
      const bool loaded = RestoreCheckpoint(config, adapter.evidence(), context,
                                            checkpoint, detail);
      const RestoreOutcome restored = loaded ? adapter.Restore(checkpoint)
                                             : RestoreOutcome{};
      if (!loaded || !restored.restored)
      {
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::restart_incompatible;
        result.failure_class = ApplicationRunResult::FailureClass::checkpoint_data;
      }
    }
    if (result.failure_class == ApplicationRunResult::FailureClass::none)
    {
      initial_hash = adapter.Snapshot().state_hash;
    }
  }

  if (result.failure_class == ApplicationRunResult::FailureClass::none &&
      adapter.coordinates().step == 0 &&
      OutputStep(config, adapter.coordinates().step))
  {
    DurableStagedFileV1 initial_vtk;
    std::vector<std::uint8_t> initial_vtk_bytes;
    const bool staged = StageStateVtk(
        config, adapter.Snapshot(), false, adapter, output_directory, context,
        initial_vtk, initial_vtk_bytes, detail);
    const bool published = staged &&
        PublishRootFile(initial_vtk, context, detail);
    if (!published)
    {
      bool cleaned = true;
      if (context.rank() == 0)
        cleaned = initial_vtk.Revert(detail) == DurableIoStatus::ok;
      (void)CollectiveAll(cleaned, context.communicator());
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    }
  }

  const std::uint64_t stop_step =
      config.run_role() == ApplicationRunRoleV1::split_writer
      ? config.scenario_checkpoint_step() : config.accepted_steps();
  bool retrying = false;
  while (result.failure_class == ApplicationRunResult::FailureClass::none &&
         adapter.coordinates().step < stop_step)
  {
    const std::uint64_t target = adapter.coordinates().step + 1;
    CpuFailureInjection injection{};
    if (!retrying &&
        config.injection().mode == FailureInjectionModeV1::prescribed &&
        config.injection().target_step == target)
    {
      injection.phase = ToCpuPhase(config.injection().phase);
      injection.target_step = target;
      injection.source_rank = static_cast<int>(config.injection().source_rank);
    }
    AttemptOutcome outcome{};
    const bool app_managed_injection =
        injection.phase == CpuFailurePhase::checkpoint_staging ||
        injection.phase == CpuFailurePhase::output_staging ||
        injection.phase == CpuFailurePhase::commit ||
        injection.phase == CpuFailurePhase::rollback_audit ||
        injection.phase == CpuFailurePhase::terminal_evidence;
    cpu::CpuPreparedStep staging;
    const bool prepared = retrying ? adapter.PrepareRetry(staging)
        : adapter.PrepareAdvance(staging, app_managed_injection
                                      ? CpuFailureInjection{} : injection);
    if (!prepared)
    {
      outcome = adapter.last_transition_outcome();
      if (outcome.reason == FailureReason::none)
        outcome = {adapter.status(), FailureReason::backend_failure, false,
                   adapter.status() == RunStatus::retry_available};
    }
    else if (!staging.numerically_ready)
      outcome = adapter.FinalizePrepared(staging);
    else
    {
      DurableStagedFileV1 vtk_file;
      DurableStagedFileV1 checkpoint_file;
      DurableStagedFileV1 metadata_file;
      std::vector<std::uint8_t> vtk_bytes;
      std::vector<std::uint8_t> checkpoint_bytes;
      std::vector<std::uint8_t> metadata_bytes;
      bool staged = true;
      CpuFailurePhase staging_failure_phase = CpuFailurePhase::output_staging;
      const CpuStateSnapshot candidate{
          staging.target_coordinates, staging.interior_hash,
          staging.trace_hash, staging.state_hash, staging.interior,
          staging.trace};
      if (OutputStep(config, target))
        staged = StageStateVtk(config, candidate, true, adapter,
                               output_directory, context, vtk_file,
                               vtk_bytes, detail);
      if (staged && config.checkpoint_out().present() &&
          target == config.scenario_checkpoint_step())
      {
        staging_failure_phase = CpuFailurePhase::checkpoint_staging;
        staged = checkpoint_writer && StageCheckpoint(
            *checkpoint_writer, staging.future_checkpoint, staging.interior,
            staging.trace, adapter.evidence(), context, checkpoint_file,
            checkpoint_bytes, detail);
      }
      if (!staged)
      {
        const bool cleaned = RevertTransaction(
            vtk_file, checkpoint_file, metadata_file, context, detail);
        outcome = adapter.AbortPrepared(
            staging, cleaned ? staging_failure_phase
                             : CpuFailurePhase::rollback_audit, -1);
      }
      else if (injection.phase == CpuFailurePhase::checkpoint_staging ||
               injection.phase == CpuFailurePhase::output_staging)
      {
        const bool cleaned = RevertTransaction(
            vtk_file, checkpoint_file, metadata_file, context, detail);
        outcome = adapter.AbortPrepared(
            staging, cleaned ? injection.phase
                             : CpuFailurePhase::rollback_audit,
            injection.source_rank);
      }
      else
      {
        cpu::CpuPreparedCommit commit;
        if (!adapter.PrepareCommit(staging, commit))
        {
          const bool cleaned = RevertTransaction(
              vtk_file, checkpoint_file, metadata_file, context, detail);
          outcome = adapter.last_transition_outcome();
          if (!cleaned)
            outcome = {RunStatus::terminal_failure,
                       FailureReason::backend_failure, false, false};
        }
        else
        {
          const bool terminal_target = target == stop_step;
          if (terminal_target)
          {
            ApplicationRunResult anticipated = result;
            anticipated.status = RunStatus::committed;
            anticipated.reason = FailureReason::none;
            anticipated.failure_class =
                ApplicationRunResult::FailureClass::none;
            anticipated.coordinates = candidate.coordinates;
            bool metadata_staged = true;
            if (context.rank() == 0)
            {
              try
              {
                const std::string metadata = Metadata(
                    config, anticipated, adapter.evidence(), initial_hash,
                    candidate.state_hash, detail);
                metadata_bytes.assign(metadata.begin(), metadata.end());
                metadata_staged = metadata_file.Stage(
                    output_directory /
                        (config.run_role() == ApplicationRunRoleV1::split_writer
                             ? "run-metadata-checkpoint.json"
                             : "run-metadata.json"),
                    metadata_bytes.data(), metadata_bytes.size(), detail) ==
                    DurableIoStatus::ok;
              }
              catch (...) { metadata_staged = false; }
            }
            staged = CollectiveAll(metadata_staged, context.communicator());
          }
          if (!staged ||
              injection.phase == CpuFailurePhase::terminal_evidence ||
              injection.phase == CpuFailurePhase::commit)
          {
            const bool cleaned = RevertTransaction(
                vtk_file, checkpoint_file, metadata_file, context, detail);
            const CpuFailurePhase phase = !staged
                ? CpuFailurePhase::terminal_evidence : injection.phase;
            outcome = adapter.AbortPrepared(
                commit, cleaned ? phase : CpuFailurePhase::rollback_audit,
                staged ? injection.source_rank : -1);
          }
          else
          {
            cpu::CpuProvisionalCommit provisional;
            if (!adapter.CommitPrepared(commit, provisional))
            {
              const bool cleaned = RevertTransaction(
                  vtk_file, checkpoint_file, metadata_file, context, detail);
              outcome = adapter.last_transition_outcome();
              if (!cleaned)
                outcome = {RunStatus::terminal_failure,
                           FailureReason::backend_failure, false, false};
            }
            else if (injection.phase == CpuFailurePhase::rollback_audit)
            {
              const bool partially_published =
                  PublishRootFile(vtk_file, context, detail) &&
                  PublishRootFile(checkpoint_file, context, detail) &&
                  PublishRootFile(metadata_file, context, detail);
              const bool cleaned = RevertTransaction(
                  vtk_file, checkpoint_file, metadata_file, context, detail);
              outcome = adapter.RevertCommit(
                  provisional,
                  !cleaned ? CpuFailurePhase::rollback_audit
                           : (partially_published
                                  ? injection.phase
                                  : CpuFailurePhase::output_staging),
                  injection.source_rank);
            }
            else
            {
              const bool published =
                  PublishRootFile(vtk_file, context, detail) &&
                  PublishRootFile(checkpoint_file, context, detail) &&
                  PublishRootFile(metadata_file, context, detail);
              if (!published)
              {
                const bool cleaned = RevertTransaction(
                    vtk_file, checkpoint_file, metadata_file, context, detail);
                outcome = adapter.RevertCommit(
                    provisional,
                    cleaned ? CpuFailurePhase::output_staging
                            : CpuFailurePhase::rollback_audit, -1);
              }
              else
              {
                outcome = adapter.FinalizeCommit(provisional);
                if (!outcome.committed)
                {
                  const bool cleaned = RevertTransaction(
                      vtk_file, checkpoint_file, metadata_file, context,
                      detail);
                  if (!cleaned)
                    outcome = {RunStatus::terminal_failure,
                               FailureReason::backend_failure, false, false};
                }
              }
            }
          }
        }
      }
    }
    if (!outcome.committed && outcome.retry_available &&
        result.retries < config.retry_limit())
    {
      ++result.retries;
      retrying = true;
      continue;
    }
    if (!outcome.committed)
    {
      result.status = outcome.status;
      result.reason = outcome.reason;
      result.failure_class = config.injection().mode ==
              FailureInjectionModeV1::prescribed
          ? ApplicationRunResult::FailureClass::injected
          : ApplicationRunResult::FailureClass::backend;
      if (detail.empty()) detail = "persistent advance did not commit";
      break;
    }
    retrying = false;
  }

  result.coordinates = initialized ? adapter.coordinates() : StateCoordinates{};
  if (result.failure_class == ApplicationRunResult::FailureClass::none)
  {
    result.status = RunStatus::committed;
    result.reason = FailureReason::none;
  }
  bool directories_clean = true;
  if (context.rank() == 0)
  {
    if (output_directory_created)
      directories_clean = DiscardDurableDirectoryIfEmpty(
                              output_directory, detail) == DurableIoStatus::ok;
  }
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

MpiApplicationContextV1::MpiApplicationContextV1(
    const MPI_Comm communicator) noexcept : communicator_(communicator)
{
  if (communicator_ != MPI_COMM_NULL)
  {
    (void)MPI_Comm_rank(communicator_, &rank_);
    (void)MPI_Comm_size(communicator_, &ranks_);
  }
}

CudaApplicationContextV1::CudaApplicationContextV1(
    const MPI_Comm communicator) noexcept : communicator_(communicator)
{
  if (communicator_ != MPI_COMM_NULL)
  {
    (void)MPI_Comm_rank(communicator_, &rank_);
    (void)MPI_Comm_size(communicator_, &ranks_);
  }
}

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuSerialBackendTag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  return RunCpu<AppCpuSerialSelection>(request, context);
}

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuRank2BackendTag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  return RunCpu<AppCpuRank2Selection>(request, context);
}

ApplicationRunResult DispatchApplication(
    const ApplicationOptionsView& request,
    MpiApplicationContextV1& cpu_context,
    CudaApplicationContextV1* cuda_context)
{
  (void)cuda_context;
  ApplicationOptionsView derived = request;
  ApplicationBackend backend = ParseApplicationBackend(request.backend_id);
  if (backend == ApplicationBackend::cpu_serial ||
      backend == ApplicationBackend::cpu_rank2)
  {
    if (cpu_context.ranks() == 1)
      derived.backend_id = CpuSerialBackendTag::id;
    else if (cpu_context.ranks() == 2)
      derived.backend_id = CpuRank2BackendTag::id;
    else
      backend = ApplicationBackend::unknown;
  }
  else if (backend == ApplicationBackend::cuda_rank1_one_gpu ||
           backend == ApplicationBackend::cuda_rank2_one_gpu_host_staged)
  {
    if (cpu_context.ranks() == 1)
      derived.backend_id = CudaRank1OneGpuBackendTag::id;
    else if (cpu_context.ranks() == 2)
      derived.backend_id = CudaRank2OneGpuHostStagedBackendTag::id;
    else
      backend = ApplicationBackend::unknown;
  }
  if (backend != ApplicationBackend::unknown)
    backend = ParseApplicationBackend(derived.backend_id);
  switch (backend)
  {
    case ApplicationBackend::cpu_serial:
      return RunSmoothPeriodicEulerV1(CpuSerialBackendTag{}, derived,
                                      cpu_context);
    case ApplicationBackend::cpu_rank2:
      return RunSmoothPeriodicEulerV1(CpuRank2BackendTag{}, derived,
                                      cpu_context);
    case ApplicationBackend::cuda_rank1_one_gpu:
#if DG_IGR_HDG_APP_HAS_CUDA_RANK1
      if (cuda_context != nullptr)
        return RunSmoothPeriodicEulerV1(CudaRank1OneGpuBackendTag{}, derived,
                                        *cuda_context);
#endif
      break;
    case ApplicationBackend::cuda_rank2_one_gpu_host_staged:
#if DG_IGR_HDG_APP_HAS_CUDA_RANK2
      if (cuda_context != nullptr)
        return RunSmoothPeriodicEulerV1(
            CudaRank2OneGpuHostStagedBackendTag{}, derived, *cuda_context);
#endif
      break;
    case ApplicationBackend::unknown: break;
  }
  return {RunStatus::invalid_request,
          FailureReason::invalid_resource_descriptor, {}, 0,
          ApplicationRunResult::FailureClass::configuration};
}

} // namespace dg::hdg::runtime::app

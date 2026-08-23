#include "numerics/hdg/runtime/app/solver.hpp"

#include "solver_entry_detail.hpp"

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
#include "../cpu/hdg_cpu_serial_lifecycle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#include "numerics/hdg/runtime/sha256.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
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

using cpu::CpuCanonicalEntry;
using cpu::CpuCheckpoint;
using cpu::CpuCoefficientView;
using cpu::CpuFailureInjection;
using cpu::CpuFailurePhase;
using cpu::CpuAttemptEvidence;
using cpu::CpuAttemptStatus;
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

const char* CpuPhaseName(const CpuFailurePhase phase) noexcept
{
  switch (phase)
  {
    case CpuFailurePhase::none: return "None";
    case CpuFailurePhase::setup: return "Setup";
    case CpuFailurePhase::assembly: return "Assembly";
    case CpuFailurePhase::condensation: return "Condensation";
    case CpuFailurePhase::trace_solve: return "TraceSolve";
    case CpuFailurePhase::recovery: return "Recovery";
    case CpuFailurePhase::diagnostics: return "Diagnostics";
    case CpuFailurePhase::checkpoint_staging: return "CheckpointStaging";
    case CpuFailurePhase::output_staging: return "OutputStaging";
    case CpuFailurePhase::after_recovery_before_commit:
      return "AfterRecoveryBeforeCommit";
    case CpuFailurePhase::commit: return "Commit";
    case CpuFailurePhase::rollback_audit: return "RollbackAudit";
    case CpuFailurePhase::terminal_evidence: return "TerminalEvidence";
  }
  return "Unknown";
}

void AppendCpuFailureProvenance(const CpuAttemptEvidence& attempt,
                                std::string& detail)
{
  if (!detail.empty()) detail += "; ";
  detail += "phase=";
  detail += CpuPhaseName(attempt.failure_phase);
  detail += " source_rank=" + std::to_string(attempt.failure_source_rank);
}

bool ScientificFailureReason(const FailureReason reason) noexcept
{
  return reason == FailureReason::finite_gate ||
      reason == FailureReason::admissibility_gate ||
      reason == FailureReason::recovery_gate ||
      reason == FailureReason::true_residual_gate ||
      reason == FailureReason::success_gate;
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

bool GatherDoubles(const double* local, std::size_t local_count,
                   const MpiApplicationContextV1& context,
                   std::array<int, 2>& counts,
                   std::array<int, 2>& displacements,
                   std::vector<double>& gathered, std::string& detail);

bool GatherPhysicalInterior(const CpuStateSnapshot& snapshot,
                            const MpiApplicationContextV1& context,
                            std::array<int, 2>& counts,
                            std::array<int, 2>& displacements,
                            std::vector<CpuCanonicalEntry>& global,
                            std::string& detail)
{
  static_assert(std::is_trivially_copyable<CpuCanonicalEntry>::value,
                "physical VTK records require MPI-byte transport");
  const CpuCoefficientView& interior = snapshot.interior;
  const bool local_valid = interior.size <= static_cast<std::size_t>(
      std::numeric_limits<int>::max()) && interior.data != nullptr &&
      snapshot.physical_interior_ids != nullptr &&
      snapshot.physical_interior_id_count == interior.size &&
      snapshot.scalar_interior_dofs > 0;
  if (!CollectiveAll(local_valid, context.communicator()))
  {
    detail = "VTK rank-local interior extent is invalid";
    return false;
  }
  std::vector<CpuCanonicalEntry> local;
  bool allocated = true;
  try
  {
    local.resize(interior.size);
    for (std::size_t index = 0; index < interior.size; ++index)
      local[index] = {snapshot.physical_interior_ids[index], interior.data[index]};
  }
  catch (...) { allocated = false; }
  if (!CollectiveAll(allocated, context.communicator()))
  {
    detail = "VTK physical record allocation failed";
    return false;
  }
  const int local_count = static_cast<int>(local.size());
  const bool count_gathered = MPI_Gather(
      &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
      context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(count_gathered, context.communicator()))
  {
    detail = "MPI_Gather VTK interior counts failed";
    return false;
  }
  int total = 0;
  bool root_valid = true;
  if (context.rank() == 0)
  {
    for (int rank = 0; rank < context.ranks(); ++rank)
    {
      displacements[static_cast<std::size_t>(rank)] = total;
      if (counts[static_cast<std::size_t>(rank)] < 0 ||
          total > std::numeric_limits<int>::max() -
              counts[static_cast<std::size_t>(rank)])
      {
        detail = "VTK aggregate interior extent exceeds MPI range";
        root_valid = false;
        break;
      }
      total += counts[static_cast<std::size_t>(rank)];
    }
    if (root_valid)
    {
      try { global.resize(static_cast<std::size_t>(total)); }
      catch (...) { root_valid = false; }
    }
  }
  if (!CollectiveAll(root_valid, context.communicator())) return false;
  std::array<int, 2> byte_counts{}, byte_displacements{};
  bool byte_valid = local_count <= std::numeric_limits<int>::max() /
      static_cast<int>(sizeof(CpuCanonicalEntry));
  if (context.rank() == 0)
  {
    for (int rank = 0; rank < context.ranks(); ++rank)
    {
      if (counts[rank] > std::numeric_limits<int>::max() /
              static_cast<int>(sizeof(CpuCanonicalEntry)) ||
          displacements[rank] > std::numeric_limits<int>::max() /
              static_cast<int>(sizeof(CpuCanonicalEntry)))
      {
        byte_valid = false;
        break;
      }
      byte_counts[rank] =
          counts[rank] * static_cast<int>(sizeof(CpuCanonicalEntry));
      byte_displacements[rank] =
          displacements[rank] * static_cast<int>(sizeof(CpuCanonicalEntry));
    }
  }
  if (!CollectiveAll(byte_valid, context.communicator()))
  {
    detail = "VTK physical record bytes exceed MPI range";
    return false;
  }
  const int gather = MPI_Gatherv(local.data(),
      local_count * static_cast<int>(sizeof(CpuCanonicalEntry)), MPI_BYTE,
      global.data(), byte_counts.data(), byte_displacements.data(), MPI_BYTE,
      0, context.communicator());
  if (!CollectiveAll(gather == MPI_SUCCESS, context.communicator()))
  {
    detail = "MPI_Gatherv VTK interior state failed";
    return false;
  }
  return true;
}

bool MakeVtk(const ValidatedApplicationConfig& config,
             const std::vector<CpuCanonicalEntry>& gathered,
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
    const int global_elements = static_cast<int>(config.mesh_n() *
                                                  config.mesh_n());
    const int scalar_dofs = (config.polynomial_order() + 1) *
                            (config.polynomial_order() + 1);
    if (static_cast<int>(gathered.size()) != global_elements * scalar_dofs * 4)
    { detail = "VTK physical record census is invalid"; return false; }
    std::vector<CpuCanonicalEntry> records = gathered;
    std::sort(records.begin(), records.end(),
              [](const CpuCanonicalEntry& a, const CpuCanonicalEntry& b)
              { return a.global_id < b.global_id; });
    for (std::size_t index = 0; index < records.size(); ++index)
      if (records[index].global_id != index)
      { detail = "VTK physical records are not complete and unique"; return false; }
    for (int element = 0; element < global_elements; ++element)
    {
      mfem::Array<int> vector_dofs;
      space.GetElementVDofs(element, vector_dofs);
      if (vector_dofs.Size() != scalar_dofs * 4)
      {
        detail = "rank-ordered VTK element map extent is invalid";
        return false;
      }
      for (int scalar = 0; scalar < scalar_dofs; ++scalar)
        for (int component = 0; component < 4; ++component)
        {
          const std::size_t source = (static_cast<std::size_t>(element) *
              scalar_dofs + scalar) * 4 + component;
          const int encoded = vector_dofs[
              scalar + component * scalar_dofs];
          const int destination = encoded >= 0 ? encoded : -1 - encoded;
          const double sign = encoded >= 0 ? 1.0 : -1.0;
          if (source >= records.size() || destination < 0 || destination >= field.Size())
          {
            detail = "rank-ordered VTK element map is out of range";
            return false;
          }
          field[destination] = sign * records[source].value;
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

bool StageStateVtk(const ValidatedApplicationConfig& config,
                   const CpuStateSnapshot& snapshot,
                   const MpiApplicationContextV1& context,
                   DurableGenerationTransactionV1& transaction,
                   std::vector<std::uint8_t>& payload,
                   std::string& detail)
{
  std::array<int, 2> counts{};
  std::array<int, 2> displacements{};
  std::vector<CpuCanonicalEntry> global;
  if (!GatherPhysicalInterior(snapshot, context, counts,
                              displacements, global, detail))
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
        staged_ok = transaction.Stage(
            StepName(snapshot.coordinates.step),
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

bool GlobalCpuStateIdentity(const CpuStateSnapshot& snapshot,
                            const MpiApplicationContextV1& context,
                            StateHash& identity) noexcept
{
  StateHash interior{};
  StateHash trace{};
  if (!CpuContractPayloadHash(snapshot.interior.data, snapshot.interior.size,
                              context, interior) ||
      !CpuContractPayloadHash(snapshot.trace.data, snapshot.trace.size,
                              context, trace))
    return false;
  std::array<std::uint8_t, 64> ordered{};
  std::copy(interior.bytes.begin(), interior.bytes.end(), ordered.begin());
  std::copy(trace.bytes.begin(), trace.bytes.end(), ordered.begin() + 32);
  identity.bytes = ComputeSha256(ordered.data(), ordered.size());
  return identity != StateHash{};
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
                     DurableGenerationTransactionV1& transaction,
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
        if (interior_counts[index] < 0 || trace_counts[index] < 0 ||
            interior_counts[index] % 4 != 0 || trace_counts[index] % 4 != 0)
        {
          staged_ok = false;
          break;
        }
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
      const CheckpointCodecStatus codec = captured && staged_ok
          ? EncodeCheckpointV1(*captured, ranks.data(),
                               static_cast<std::size_t>(context.ranks()), encoded)
          : CheckpointCodecStatus::invalid_descriptor;
      staged_ok = staged_ok && codec == CheckpointCodecStatus::ok &&
          transaction.Stage("restart-checkpoint.v2", encoded.data(),
                            encoded.size(), detail) == DurableIoStatus::ok &&
          transaction.AddAlias(
              std::string{config.checkpoint_out().identity().view()},
              "restart-checkpoint.v2", detail) == DurableIoStatus::ok;
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
  std::array<RankCoefficientExtentsV1, 2> observed{};
  const std::array<std::uint64_t, 2> local_shape{{
      evidence.local_owned_interior_coefficients,
      evidence.local_owned_true_trace_coefficients}};
  std::array<std::array<std::uint64_t, 2>, 2> all_shapes{};
  const bool shapes_gathered = MPI_Allgather(
      local_shape.data(), 2, MPI_UINT64_T, all_shapes.data(), 2,
      MPI_UINT64_T, context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(shapes_gathered, context.communicator()))
  {
    detail = "MPI_Allgather restart runtime extents failed";
    return false;
  }
  for (int rank = 0; rank < context.ranks(); ++rank)
  {
    const auto& shape = all_shapes[static_cast<std::size_t>(rank)];
    if (shape[0] % 4 != 0 || shape[1] % 4 != 0)
    {
      detail = "restart observed extent is not component-major";
      return false;
    }
    observed[static_cast<std::size_t>(rank)] = {
        shape[0] / 4, shape[0], shape[1] / 4, shape[1]};
  }
  ScenarioCoefficientExtentsV1 extents{};
  const bool observed_rank2 = context.ranks() == 2;
  if (!(observed_rank2
            ? DeriveObservedCoefficientExtentsV1(
                  config, static_cast<std::uint32_t>(context.ranks()),
                  observed.data(), extents)
            : DeriveScenarioCoefficientExtentsV1(
                  config, static_cast<std::uint32_t>(context.ranks()), extents)))
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
      decoded = detail::ReadCurrentGenerationArtifact(
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
  if (config.run_role() == ApplicationRunRoleV1::split_writer)
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

bool BeginRootTransaction(DurableGenerationTransactionV1& transaction,
                          const std::filesystem::path& output_directory,
                          const std::uint64_t generation,
                          const MpiApplicationContextV1& context,
                          std::string& detail)
{
  bool local = true;
  if (context.rank() == 0)
    local = transaction.Begin(output_directory, generation, detail) ==
        DurableIoStatus::ok;
  return CollectiveAll(local, context.communicator());
}

bool PublishRootTransaction(DurableGenerationTransactionV1& transaction,
                            const MpiApplicationContextV1& context,
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
                       const MpiApplicationContextV1& context,
                       std::string& detail)
{
  bool local = true;
  if (context.rank() == 0)
    local = transaction.Revert(detail) == DurableIoStatus::ok;
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

class ApplicationCpuSerialSchedule final
{
public:
  explicit ApplicationCpuSerialSchedule(
      const ValidatedApplicationConfig& config) noexcept : config_(config) {}

  std::uint64_t AcceptedSteps() const noexcept
  { return config_.accepted_steps(); }
  std::uint64_t CheckpointStep() const noexcept
  { return config_.scenario_checkpoint_step(); }
  bool OutputStep(const std::uint64_t step) const noexcept
  { return ::dg::hdg::runtime::app::OutputStep(config_, step); }
  bool CheckpointStep(const std::uint64_t step) const noexcept
  {
    return config_.checkpoint_out().present() &&
        step == config_.scenario_checkpoint_step();
  }
  std::uint64_t FailureTargetStep() const noexcept
  { return config_.injection().target_step; }

private:
  const ValidatedApplicationConfig& config_;
};


template <typename Case>
struct W8cCpuSerialSelection : CpuSerialRuntimeSelection
{
  static constexpr std::string_view experiment_id = Case::experiment_id;
};

template <typename Case>
class W8cConventionalMethod final
{
public:
  struct Transaction final
  {
    cpu::CpuPreparedStep staging;
    cpu::CpuPreparedCommit prepared;
    cpu::CpuProvisionalCommit provisional;
    CpuStateSnapshot candidate{};
    CheckpointDescriptor future_checkpoint{};
  };

  bool Initialize() noexcept
  {
    CpuPersistentConfig numerical = cpu::MakeCpuSmokeConfig();
    numerical.experiment = StableIdentity::Copy(Case::experiment_id);
    numerical.source = StableIdentity::Copy(
        Case::conventional_source_id);
    numerical.polynomial_order = Case::polynomial_order;
    numerical.elements_per_direction = static_cast<int>(Case::mesh_n);
    numerical.assembly_quadrature_order = Case::conventional_q_a;
    numerical.dt = Case::dt;
    numerical.allow_one_retry = true;
    return adapter_.Initialize(numerical);
  }

  bool PrimeTransaction(Transaction& transaction) noexcept
  {
    transaction = {};
    return true;
  }

  bool Prepare(Transaction& transaction, const bool retry,
               const std::uint64_t) noexcept
  {
    transaction = {};
    const bool prepared = retry ? adapter_.PrepareRetry(transaction.staging)
                                : adapter_.PrepareAdvance(transaction.staging);
    if (!prepared) return false;
    auto candidate_capability =
        adapter_.AcquirePreparedCandidate(transaction.staging);
    if (!candidate_capability || !adapter_.ConsumePreparedCandidate(
                                   std::move(*candidate_capability),
                                   transaction.candidate))
    {
      (void)adapter_.AbortPrepared(
          transaction.staging, CpuFailurePhase::diagnostics, 0);
      return false;
    }
    transaction.candidate.coordinates = transaction.staging.target_coordinates;
    transaction.future_checkpoint = transaction.staging.future_checkpoint;
    if (!transaction.staging.numerically_ready)
    {
      (void)adapter_.FinalizePrepared(transaction.staging);
      return false;
    }
    if (!adapter_.PrepareCommit(transaction.staging, transaction.prepared))
      return false;
    transaction.future_checkpoint = transaction.prepared.future_checkpoint;
    return true;
  }

  bool Install(Transaction& transaction) noexcept
  {
    return adapter_.CommitPrepared(transaction.prepared,
                                   transaction.provisional);
  }

  cpu::w8c::LifecycleAttemptOutcome PrepareFailure(
      const Transaction&) const noexcept
  {
    return {true, RetryAvailable(), false};
  }

  cpu::w8c::LifecycleAttemptOutcome AbortPrepared(
      Transaction&& transaction, const cpu::w8c::LifecycleFailure failure,
      const bool durable_reverted) noexcept
  {
    AttemptOutcome outcome = adapter_.last_transition_outcome();
    bool state_reverted = !outcome.committed;
    if (transaction.prepared.valid)
    {
      outcome = adapter_.AbortPrepared(
          transaction.prepared,
          CpuFailurePhase::after_recovery_before_commit, 0);
      state_reverted = !outcome.committed && outcome.retry_available;
    }
    const bool retryable = durable_reverted && state_reverted &&
        (failure == cpu::w8c::LifecycleFailure::staging ||
         failure == cpu::w8c::LifecycleFailure::injected_after_recovery);
    return {durable_reverted && state_reverted, retryable, false};
  }

  cpu::w8c::LifecycleAttemptOutcome RevertInstalled(
      Transaction&& transaction, const cpu::w8c::LifecycleFailure failure,
      const bool durable_reverted) noexcept
  {
    const AttemptOutcome outcome = adapter_.RevertCommit(
        transaction.provisional, CpuFailurePhase::after_recovery_before_commit,
        0);
    const bool state_reverted = !outcome.committed && outcome.retry_available;
    const bool retryable = durable_reverted && state_reverted &&
        failure == cpu::w8c::LifecycleFailure::injected_after_recovery;
    return {durable_reverted && state_reverted, retryable, false};
  }

  void Finalize(Transaction&& transaction) noexcept
  {
    // Install supplied the valid provisional token consumed here.  The
    // run-contract FinalizeCommit implementation has no rejecting transition
    // for such a token, so a contrary outcome is an internal invariant breach,
    // not a recoverable post-publication branch.
    const AttemptOutcome outcome = adapter_.FinalizeCommit(
        transaction.provisional);
    if (!outcome.committed || outcome.reason != FailureReason::none)
      std::abort();
  }

  bool Restore(const CpuCheckpoint& checkpoint) noexcept
  {
    return adapter_.Restore(checkpoint).restored;
  }

  RunStatus Status() const noexcept { return adapter_.status(); }
  FailureReason Reason() const noexcept
  {
    return adapter_.last_transition_outcome().reason;
  }
  bool RetryAvailable() const noexcept
  {
    return adapter_.status() == RunStatus::retry_available;
  }
  constexpr bool ForceTerminalFailure() const noexcept { return true; }
  void OnRetry(cpu::w8c::LifecycleFailure, std::uint32_t) const noexcept {}
  StateCoordinates Coordinates() const noexcept
  {
    return adapter_.coordinates();
  }
  CpuStateSnapshot PublishedState() const noexcept
  {
    return adapter_.Snapshot();
  }
  const CpuStateSnapshot& CandidateState(const Transaction& transaction)
      const noexcept
  {
    return transaction.candidate;
  }
  const CheckpointDescriptor& FutureCheckpoint(
      const Transaction& transaction) const noexcept
  {
    return transaction.future_checkpoint;
  }
  const CpuRunEvidence& Evidence() const noexcept
  {
    return adapter_.evidence();
  }

private:
  cpu::PersistentSmoothEulerCpuAdapter<W8cCpuSerialSelection<Case>> adapter_;
};

template <typename Case>
class W8cConventionalPublication final
{
public:
  W8cConventionalPublication(
      const ApplicationOptionsView& request,
      const ValidatedApplicationConfig& config,
      MpiApplicationContextV1& context, std::string& detail) noexcept
      : config_(config), context_(context), detail_(detail),
        output_directory_(std::string{
            config.output_directory().identity().view()}),
        checkpoint_writer_(CheckpointWriterConfig(request, config))
  {}

  bool Restore(W8cConventionalMethod<Case>& method) noexcept
  {
    if (!EnsureDirectory()) return false;
    CpuCheckpoint checkpoint;
    if (!RestoreCheckpoint(config_, method.Evidence(), context_, checkpoint,
                           detail_) ||
        !method.Restore(checkpoint))
      return false;
    return GlobalCpuStateIdentity(method.PublishedState(), context_,
                                  initial_hash_);
  }

  bool PublishInitial(W8cConventionalMethod<Case>& method) noexcept
  {
    if (!EnsureDirectory() ||
        !GlobalCpuStateIdentity(method.PublishedState(), context_,
                                initial_hash_))
      return false;
    if (Case::output_steps[0] != 0) return true;
    DurableGenerationTransactionV1 initial;
    std::vector<std::uint8_t> vtk;
    const bool staged = BeginRootTransaction(
        initial, output_directory_, 0, context_, detail_) &&
        StageStateVtk(config_, method.PublishedState(), context_, initial,
                      vtk, detail_);
    const bool published = staged &&
        PublishRootTransaction(initial, context_, detail_);
    if (!published) (void)RevertTransaction(initial, context_, detail_);
    return published;
  }

  bool Stage(W8cConventionalMethod<Case>& method,
             const typename W8cConventionalMethod<Case>::Transaction&
                 method_transaction,
             const std::uint64_t target, const bool output,
             const bool checkpoint, const bool terminal,
             const std::uint32_t) noexcept
  {
    try
    {
      if (!output && !checkpoint && !terminal)
      {
        active_ = false;
        return true;
      }
      transaction_.emplace();
      active_ = true;
      bool staged = BeginRootTransaction(
          *transaction_, output_directory_, target, context_, detail_);
      const CpuStateSnapshot& state =
          method.CandidateState(method_transaction);
      if (staged && output)
        staged = StageStateVtk(config_, state, context_, *transaction_, vtk_,
                               detail_);
      if (staged && checkpoint)
        staged = checkpoint_writer_ && StageCheckpoint(
            *checkpoint_writer_, method.FutureCheckpoint(method_transaction),
            state.interior, state.trace, method.Evidence(), context_,
            *transaction_, checkpoint_, detail_);
      if (staged && terminal)
      {
        StateHash final_hash{};
        staged = GlobalCpuStateIdentity(state, context_, final_hash);
        bool root_staged = staged;
        if (staged && context_.rank() == 0)
        {
          ApplicationRunResult anticipated;
          anticipated.status = RunStatus::committed;
          anticipated.reason = FailureReason::none;
          anticipated.coordinates = state.coordinates;
          anticipated.retries = ExpectedRetries();
          anticipated.failure_class = ApplicationRunResult::FailureClass::none;
          const std::string metadata = Metadata(
              config_, anticipated, method.Evidence(), initial_hash_,
              final_hash, detail_);
          metadata_.assign(metadata.begin(), metadata.end());
          root_staged = transaction_->Stage(
              config_.run_role() == ApplicationRunRoleV1::split_writer
                  ? "run-metadata-checkpoint.json" : "run-metadata.json",
              metadata_.data(), metadata_.size(), detail_) ==
              DurableIoStatus::ok;
        }
        staged = CollectiveAll(root_staged, context_.communicator());
      }
      return staged;
    }
    catch (...)
    {
      detail_ = "W8C conventional durable staging failed";
      return false;
    }
  }

  bool ReadyToPublish(
      W8cConventionalMethod<Case>&,
      const typename W8cConventionalMethod<Case>::Transaction&,
      const std::uint32_t) const noexcept
  {
    return true;
  }

  bool Publish(
      W8cConventionalMethod<Case>&,
      const typename W8cConventionalMethod<Case>::Transaction&) noexcept
  {
    if (!active_) return true;
    return PublishRootTransaction(*transaction_, context_, detail_);
  }

  bool Acknowledge(
      W8cConventionalMethod<Case>&,
      const typename W8cConventionalMethod<Case>::Transaction&,
      const std::uint32_t) const noexcept
  {
    return true;
  }

  bool Revert() noexcept
  {
    if (!active_) return true;
    const bool reverted = RevertTransaction(*transaction_, context_, detail_);
    transaction_.reset();
    active_ = false;
    return reverted;
  }

  void Finalize() noexcept
  {
    transaction_.reset();
    active_ = false;
  }

  bool Cleanup() noexcept
  {
    bool clean = true;
    if (context_.rank() == 0 && output_directory_created_)
      clean = DiscardDurableDirectoryIfEmpty(output_directory_, detail_) ==
          DurableIoStatus::ok;
    return CollectiveAll(clean, context_.communicator());
  }

private:
  bool EnsureDirectory() noexcept
  {
    if (directory_ready_) return true;
    bool ready = true;
    if (context_.rank() == 0)
    {
      if (config_.run_role() == ApplicationRunRoleV1::split_reader)
      {
        std::error_code error;
        const std::filesystem::file_status status =
            std::filesystem::symlink_status(output_directory_, error);
        ready = !error && std::filesystem::is_directory(status);
        if (!ready) detail_ = "W8C split-reader output directory is unavailable";
      }
      else
      {
        ready = CreateDurableDirectory(output_directory_, detail_) ==
            DurableIoStatus::ok;
        output_directory_created_ = ready;
      }
    }
    directory_ready_ = CollectiveAll(ready, context_.communicator());
    return directory_ready_;
  }

  std::uint32_t ExpectedRetries() const noexcept
  {
    return config_.injection().mode == FailureInjectionModeV1::prescribed
        ? 1U : 0U;
  }

  const ValidatedApplicationConfig& config_;
  MpiApplicationContextV1& context_;
  std::string& detail_;
  std::filesystem::path output_directory_;
  std::optional<ValidatedApplicationConfig> checkpoint_writer_;
  std::optional<DurableGenerationTransactionV1> transaction_;
  std::vector<std::uint8_t> vtk_;
  std::vector<std::uint8_t> checkpoint_;
  std::vector<std::uint8_t> metadata_;
  StateHash initial_hash_{};
  bool directory_ready_ = false;
  bool output_directory_created_ = false;
  bool active_ = false;
};


ApplicationRunResult W8cResult(
    const cpu::w8c::LifecycleReport& report) noexcept
{
  ApplicationRunResult result;
  result.status = report.status;
  result.reason = report.reason;
  result.coordinates = report.coordinates;
  result.retries = report.retries;
  if (report.failure == cpu::w8c::LifecycleFailure::none)
    result.failure_class = ApplicationRunResult::FailureClass::none;
  else if (report.failure ==
           cpu::w8c::LifecycleFailure::injected_after_recovery)
    result.failure_class = ApplicationRunResult::FailureClass::injected;
  else if (report.failure == cpu::w8c::LifecycleFailure::restart)
    result.failure_class = ApplicationRunResult::FailureClass::checkpoint_data;
  else if (report.failure == cpu::w8c::LifecycleFailure::initial_output ||
           report.failure == cpu::w8c::LifecycleFailure::staging ||
           report.failure == cpu::w8c::LifecycleFailure::publication)
    result.failure_class = ApplicationRunResult::FailureClass::filesystem;
  else if (ScientificFailureReason(report.reason))
    result.failure_class = ApplicationRunResult::FailureClass::scientific;
  else
    result.failure_class = ApplicationRunResult::FailureClass::backend;
  return result;
}

template <typename Case>
ApplicationRunResult RunW8cCaseCpuSerial(
    const ApplicationOptionsView& request, MpiApplicationContextV1& context)
{
  const ApplicationValidation validation = ValidateApplicationOptions(request);
  if (!validation.accepted() || context.ranks() != 1)
  {
    ApplicationRunResult invalid;
    invalid.status = request.interior_space == InteriorSpace::h1
        ? RunStatus::unsupported_h1 : RunStatus::invalid_request;
    invalid.reason = request.interior_space == InteriorSpace::h1
        ? FailureReason::unsupported_h1
        : FailureReason::invalid_resource_descriptor;
    invalid.failure_class = ApplicationRunResult::FailureClass::configuration;
    return invalid;
  }
  const ValidatedApplicationConfig& config = *validation.config();
  cpu::w8c::LifecycleControl control;
  control.restart = config.run_role() == ApplicationRunRoleV1::split_reader;
  control.split_writer =
      config.run_role() == ApplicationRunRoleV1::split_writer;
  control.inject_after_recovery =
      config.injection().mode == FailureInjectionModeV1::prescribed;
  control.maximum_retries = config.retry_limit();
  std::string detail;
  cpu::w8c::LifecycleReport report;
  bool clean = true;
  W8cConventionalMethod<Case> method;
  W8cConventionalPublication<Case> publication(
      request, config, context, detail);
  report = cpu::w8c::RunCpuSerialLifecycle(
      method, publication, control, Case{});
  if (report.failure != cpu::w8c::LifecycleFailure::none)
  {
    const CpuAttemptEvidence& attempt = method.Evidence().last_attempt;
    AppendCpuFailureProvenance(attempt, detail);
  }
  clean = publication.Cleanup();
  ApplicationRunResult result = W8cResult(report);
  if (!clean)
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::filesystem;
  }
  if (detail.empty() && result.failure_class !=
                            ApplicationRunResult::FailureClass::none)
    detail = "W8C shared CPU-serial lifecycle failed";
  context.SetDiagnostic(detail);
  return result;
}

ApplicationRunResult RunW8cCpuSerial(
    const ApplicationOptionsView& request, MpiApplicationContextV1& context)
{
  if (request.case_id == cpu::w8c::SmoothTransitionV2Case::case_id)
  {
    if (request.scenario_id == cpu::w8c::SmoothTransitionN16Dt11Case::scenario_id)
      return RunW8cCaseCpuSerial<cpu::w8c::SmoothTransitionN16Dt11Case>(request, context);
    if (request.scenario_id == cpu::w8c::SmoothTransitionN16Dt12Case::scenario_id)
      return RunW8cCaseCpuSerial<cpu::w8c::SmoothTransitionN16Dt12Case>(request, context);
    if (request.scenario_id == cpu::w8c::SmoothTransitionN32Dt12Case::scenario_id)
      return RunW8cCaseCpuSerial<cpu::w8c::SmoothTransitionN32Dt12Case>(request, context);
    return RunW8cCaseCpuSerial<cpu::w8c::SmoothTransitionV2Case>(request, context);
  }
  return RunW8cCaseCpuSerial<cpu::w8c::ExactJumpV1Case>(request, context);
}

template <typename Selection>
class NoRegularizationCpuSerialMethod final
{
public:
  struct Transaction final
  {
    cpu::CpuPreparedStep staging;
    cpu::CpuPreparedCommit prepared;
    cpu::CpuProvisionalCommit provisional;
    CpuStateSnapshot candidate{};
    CheckpointDescriptor future_checkpoint{};
    CpuFailureInjection injection{};
    AttemptOutcome outcome{};
    CpuFailurePhase rollback_phase = CpuFailurePhase::none;
    int rollback_source_rank = -1;
    bool installed = false;
    bool private_post_publication_retry = false;
  };

  NoRegularizationCpuSerialMethod(
      cpu::PersistentSmoothEulerCpuAdapter<Selection>& adapter,
      const ValidatedApplicationConfig& config,
      MpiApplicationContextV1& context, std::string& detail) noexcept
      : adapter_(adapter), config_(config), context_(context), detail_(detail)
  {}

  bool Initialize() const noexcept { return true; }
  bool PrimeTransaction(Transaction& transaction) noexcept
  {
    transaction = {};
    return true;
  }

  bool Prepare(Transaction& transaction, const bool retrying,
               const std::uint64_t target) noexcept
  {
    transaction = {};
    if (!retrying &&
        config_.injection().mode == FailureInjectionModeV1::prescribed &&
        config_.injection().target_step == target)
    {
      transaction.injection.phase = ToCpuPhase(config_.injection().phase);
      transaction.injection.target_step = target;
      transaction.injection.source_rank =
          static_cast<int>(config_.injection().source_rank);
    }
    const bool app_managed_injection =
        transaction.injection.phase == CpuFailurePhase::checkpoint_staging ||
        transaction.injection.phase == CpuFailurePhase::output_staging ||
        transaction.injection.phase == CpuFailurePhase::commit ||
        transaction.injection.phase == CpuFailurePhase::rollback_audit ||
        transaction.injection.phase == CpuFailurePhase::terminal_evidence;
    const bool locally_prepared = retrying
        ? adapter_.PrepareRetry(transaction.staging)
        : adapter_.PrepareAdvance(
              transaction.staging,
              app_managed_injection ? CpuFailureInjection{}
                                    : transaction.injection);
    const bool prepared = RankCommon(locally_prepared);
    if (!prepared)
    {
      transaction.outcome = adapter_.last_transition_outcome();
      if (transaction.outcome.reason == FailureReason::none)
        transaction.outcome = {
            adapter_.status(), FailureReason::backend_failure, false,
            adapter_.status() == RunStatus::retry_available};
      return false;
    }
    if (!RankCommon(transaction.staging.numerically_ready))
    {
      transaction.outcome = adapter_.FinalizePrepared(transaction.staging);
      return false;
    }
    auto candidate_capability =
        adapter_.AcquirePreparedCandidate(transaction.staging);
    const bool locally_consumed = candidate_capability &&
        adapter_.ConsumePreparedCandidate(std::move(*candidate_capability),
                                          transaction.candidate);
    if (!RankCommon(locally_consumed))
    {
      transaction.outcome = adapter_.AbortPrepared(
          transaction.staging, CpuFailurePhase::diagnostics);
      return false;
    }
    transaction.candidate.coordinates = transaction.staging.target_coordinates;
    transaction.future_checkpoint = transaction.staging.future_checkpoint;
    const bool locally_prepared_commit = adapter_.PrepareCommit(
        transaction.staging, transaction.prepared);
    if (!RankCommon(locally_prepared_commit))
    {
      transaction.outcome = adapter_.last_transition_outcome();
      return false;
    }
    return true;
  }

  cpu::w8c::LifecycleAttemptOutcome PrepareFailure(
      const Transaction& transaction) const noexcept
  {
    return {true, RankCommon(transaction.outcome.retry_available), false};
  }

  bool Install(Transaction& transaction) noexcept
  {
    const bool locally_installed = adapter_.CommitPrepared(
        transaction.prepared, transaction.provisional);
    transaction.installed = RankCommon(locally_installed);
    if (!transaction.installed)
      transaction.outcome = adapter_.last_transition_outcome();
    return transaction.installed;
  }

  cpu::w8c::LifecycleAttemptOutcome AbortPrepared(
      Transaction&& transaction, const cpu::w8c::LifecycleFailure,
      const bool durable_reverted) noexcept
  {
    AttemptOutcome outcome = transaction.outcome;
    if (transaction.prepared.valid)
      outcome = adapter_.AbortPrepared(
          transaction.prepared,
          durable_reverted ? transaction.rollback_phase
                           : CpuFailurePhase::rollback_audit,
          transaction.rollback_source_rank);
    const bool locally_reverted = !outcome.committed &&
        outcome.reason != FailureReason::stale_attempt;
    const bool state_reverted = RankCommon(locally_reverted);
    const bool retry_available = RankCommon(outcome.retry_available);
    return {durable_reverted && state_reverted,
            retry_available, false};
  }

  cpu::w8c::LifecycleAttemptOutcome RevertInstalled(
      Transaction&& transaction, const cpu::w8c::LifecycleFailure failure,
      const bool durable_reverted) noexcept
  {
    const AttemptOutcome outcome = adapter_.RevertCommit(
        transaction.provisional,
        durable_reverted ? transaction.rollback_phase
                         : CpuFailurePhase::rollback_audit,
        transaction.rollback_source_rank);
    const bool locally_reverted = !outcome.committed &&
        outcome.reason != FailureReason::stale_attempt;
    const bool state_reverted = RankCommon(locally_reverted);
    const bool retry_available = RankCommon(outcome.retry_available);
    return {durable_reverted && state_reverted,
            retry_available,
            failure ==
                    cpu::w8c::LifecycleFailure::
                        post_publication_acknowledgement &&
                durable_reverted &&
                transaction.private_post_publication_retry};
  }

  void Finalize(Transaction&& transaction) noexcept
  {
    const AttemptOutcome outcome =
        adapter_.FinalizeCommit(transaction.provisional);
    const bool finalized = RankCommon(
        outcome.committed && outcome.reason == FailureReason::none);
    if (!finalized)
      std::abort();
  }

  RunStatus Status() const noexcept { return adapter_.status(); }
  FailureReason Reason() const noexcept
  { return adapter_.last_transition_outcome().reason; }
  StateCoordinates Coordinates() const noexcept
  { return adapter_.coordinates(); }
  constexpr bool ForceTerminalFailure() const noexcept { return false; }
  void OnRetry(cpu::w8c::LifecycleFailure,
               const std::uint32_t) const noexcept
  {
    try
    {
      AppendCpuFailureProvenance(adapter_.evidence().last_attempt, detail_);
    }
    catch (...) {}
  }

  ApplicationRunResult::FailureClass failure_class() const noexcept
  {
    const CpuAttemptEvidence& failed = adapter_.evidence().last_attempt;
    const bool prescribed_failure_fired =
        config_.injection().mode == FailureInjectionModeV1::prescribed &&
        failed.status == CpuAttemptStatus::injected_failure &&
        failed.failure_phase == ToCpuPhase(config_.injection().phase) &&
        failed.failure_source_rank ==
            static_cast<int>(config_.injection().source_rank);
    const AttemptOutcome outcome = adapter_.last_transition_outcome();
    const bool scientific_failure = ScientificFailureReason(outcome.reason) ||
        (failed.status == CpuAttemptStatus::success &&
         failed.failure_phase == CpuFailurePhase::none &&
         (!failed.finite_gate || !failed.admissibility_gate ||
          !failed.recovery_gate || !failed.nonlinear_gate ||
          !failed.true_residual_gate));
    return prescribed_failure_fired
        ? ApplicationRunResult::FailureClass::injected
        : (scientific_failure
               ? ApplicationRunResult::FailureClass::scientific
               : ApplicationRunResult::FailureClass::backend);
  }

  const CpuRunEvidence& Evidence() const noexcept
  { return adapter_.evidence(); }

private:
  bool RankCommon(const bool local) const noexcept
  {
    if constexpr (std::is_same<Selection, AppCpuRank2Selection>::value)
      return CollectiveAll(local, context_.communicator());
    return local;
  }

  cpu::PersistentSmoothEulerCpuAdapter<Selection>& adapter_;
  const ValidatedApplicationConfig& config_;
  MpiApplicationContextV1& context_;
  std::string& detail_;
};

template <typename Selection>
class NoRegularizationCpuSerialPublication final
{
public:
  using Method = NoRegularizationCpuSerialMethod<Selection>;
  using Transaction = typename Method::Transaction;

  NoRegularizationCpuSerialPublication(
      const ValidatedApplicationConfig& config,
      MpiApplicationContextV1& context,
      const std::filesystem::path& output_directory,
      const std::optional<ValidatedApplicationConfig>& checkpoint_writer,
      const StateHash& initial_hash, ApplicationRunResult& result,
      Method& method, std::string& detail) noexcept
      : config_(config), context_(context),
        output_directory_(output_directory),
        checkpoint_writer_(checkpoint_writer), initial_hash_(initial_hash),
        result_(result), method_(method), detail_(detail)
  {}

  bool Restore(Method&) const noexcept { return true; }
  bool PublishInitial(Method&) const noexcept { return true; }

  bool Stage(Method&, Transaction& method_transaction,
             const std::uint64_t target, const bool output,
             const bool checkpoint, const bool terminal,
             const std::uint32_t retries) noexcept
  {
    const auto stage_impl = [&]() -> bool
    {
      transaction_.emplace();
      active_ = true;
      vtk_.clear();
      checkpoint_.clear();
      metadata_.clear();
      bool staged = BeginRootTransaction(
          *transaction_, output_directory_, target, context_, detail_);
      CpuFailurePhase staging_failure_phase =
          CpuFailurePhase::output_staging;
      const CpuFailureInjection& injection = method_transaction.injection;
      const bool injected_outer_stage =
          injection.phase == CpuFailurePhase::checkpoint_staging ||
          injection.phase == CpuFailurePhase::output_staging;
      if (!injected_outer_stage && output)
        staged = StageStateVtk(
            config_, method_transaction.candidate, context_, *transaction_,
            vtk_, detail_);
      if (!injected_outer_stage && staged && checkpoint)
      {
        staging_failure_phase = CpuFailurePhase::checkpoint_staging;
        staged = checkpoint_writer_ && StageCheckpoint(
            *checkpoint_writer_, method_transaction.future_checkpoint,
            method_transaction.candidate.interior,
            method_transaction.candidate.trace, method_.Evidence(), context_,
            *transaction_, checkpoint_, detail_);
      }
      if (!staged)
      {
        method_transaction.rollback_phase = staging_failure_phase;
        method_transaction.rollback_source_rank = -1;
        return false;
      }
      if (injected_outer_stage)
      {
        method_transaction.rollback_phase = injection.phase;
        method_transaction.rollback_source_rank = injection.source_rank;
        return false;
      }
      if (terminal)
      {
        ApplicationRunResult anticipated = result_;
        anticipated.status = RunStatus::committed;
        anticipated.reason = FailureReason::none;
        anticipated.failure_class =
            ApplicationRunResult::FailureClass::none;
        anticipated.coordinates = method_transaction.candidate.coordinates;
        anticipated.retries = retries;
        StateHash terminal_identity{};
        bool metadata_staged = GlobalCpuStateIdentity(
            method_transaction.candidate, context_, terminal_identity);
        if (context_.rank() == 0)
        {
          try
          {
            const std::string metadata = Metadata(
                config_, anticipated, method_.Evidence(), initial_hash_,
                terminal_identity, detail_);
            metadata_.assign(metadata.begin(), metadata.end());
            metadata_staged = transaction_->Stage(
                config_.run_role() == ApplicationRunRoleV1::split_writer
                    ? "run-metadata-checkpoint.json" : "run-metadata.json",
                metadata_.data(), metadata_.size(), detail_) ==
                DurableIoStatus::ok;
          }
          catch (...) { metadata_staged = false; }
        }
        staged = CollectiveAll(metadata_staged, context_.communicator());
      }
      if (!staged ||
          injection.phase == CpuFailurePhase::terminal_evidence ||
          injection.phase == CpuFailurePhase::commit)
      {
        method_transaction.rollback_phase = !staged
            ? CpuFailurePhase::terminal_evidence : injection.phase;
        method_transaction.rollback_source_rank =
            staged ? injection.source_rank : -1;
        return false;
      }
      return true;
    };
    if constexpr (std::is_same<Selection, AppCpuRank2Selection>::value)
      return stage_impl();
    try { return stage_impl(); }
    catch (...)
    {
      method_transaction.rollback_phase = CpuFailurePhase::terminal_evidence;
      method_transaction.rollback_source_rank = -1;
      return false;
    }
  }

  bool ReadyToPublish(Method&, Transaction& method_transaction,
                      const std::uint32_t) const noexcept
  {
    if (method_transaction.injection.phase !=
        CpuFailurePhase::rollback_audit)
      return true;
    method_transaction.rollback_phase = CpuFailurePhase::rollback_audit;
    method_transaction.rollback_source_rank =
        method_transaction.injection.source_rank;
    return false;
  }

  bool Publish(Method&, Transaction& method_transaction) noexcept
  {
    const bool published = active_ && PublishRootTransaction(
        *transaction_, context_, detail_);
    if (!published)
    {
      method_transaction.rollback_phase = CpuFailurePhase::output_staging;
      method_transaction.rollback_source_rank = -1;
    }
    return published;
  }

  bool Acknowledge(Method&, Transaction& method_transaction,
                   const std::uint32_t retries) noexcept
  {
    int marker_source_rank = -1;
    int finalize_source_rank = -1;
    const std::uint64_t target =
        method_transaction.candidate.coordinates.step;
    const bool marker_acknowledged = CollectiveAll(
        !ConsumePostPublicationFault(
            PostPublicationFault::after_marker, target, context_.rank(),
            marker_source_rank),
        context_.communicator());
    const bool finalize_ready = marker_acknowledged && CollectiveAll(
        !ConsumePostPublicationFault(
            PostPublicationFault::before_finalize, target, context_.rank(),
            finalize_source_rank),
        context_.communicator());
    if (!finalize_ready)
    {
      method_transaction.rollback_phase = marker_acknowledged
          ? CpuFailurePhase::commit : CpuFailurePhase::output_staging;
      method_transaction.rollback_source_rank = marker_acknowledged
          ? finalize_source_rank : marker_source_rank;
      method_transaction.private_post_publication_retry =
          PostPublicationFaultAllowsRetry() && retries == 0;
    }
    return finalize_ready;
  }

  bool Revert() noexcept
  {
    if (!active_) return true;
    const bool reverted = RevertTransaction(
        *transaction_, context_, detail_);
    transaction_.reset();
    active_ = false;
    return reverted;
  }

  void Finalize() noexcept
  {
    transaction_.reset();
    active_ = false;
  }

private:
  const ValidatedApplicationConfig& config_;
  MpiApplicationContextV1& context_;
  const std::filesystem::path& output_directory_;
  const std::optional<ValidatedApplicationConfig>& checkpoint_writer_;
  const StateHash& initial_hash_;
  ApplicationRunResult& result_;
  Method& method_;
  std::string& detail_;
  std::optional<DurableGenerationTransactionV1> transaction_;
  std::vector<std::uint8_t> vtk_;
  std::vector<std::uint8_t> checkpoint_;
  std::vector<std::uint8_t> metadata_;
  bool active_ = false;
};

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
  const char* const scientific_test =
      std::getenv("DG_IGR_HDG_TEST_SCIENTIFIC_GATE_FAILURE");
  if (scientific_test != nullptr && std::string_view{scientific_test} == "1")
    numerical.force_private_scientific_gate_failure = true;
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
    if (result.failure_class == ApplicationRunResult::FailureClass::none &&
        !GlobalCpuStateIdentity(adapter.Snapshot(), context, initial_hash))
    {
      detail = "rank-ordered initial state identity failed";
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = ApplicationRunResult::FailureClass::backend;
    }
  }

  if (result.failure_class == ApplicationRunResult::FailureClass::none &&
      adapter.coordinates().step == 0 &&
      OutputStep(config, adapter.coordinates().step))
  {
    DurableGenerationTransactionV1 initial_transaction;
    std::vector<std::uint8_t> initial_vtk_bytes;
    const bool staged = BeginRootTransaction(
        initial_transaction, output_directory, 0, context, detail) &&
        StageStateVtk(config, adapter.Snapshot(), context, initial_transaction,
                      initial_vtk_bytes, detail);
    const bool published = staged && PublishRootTransaction(
        initial_transaction, context, detail);
    if (!published)
    {
      (void)RevertTransaction(initial_transaction, context, detail);
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    }
  }

  if (result.failure_class == ApplicationRunResult::FailureClass::none)
  {
    NoRegularizationCpuSerialMethod<Selection> method(
        adapter, config, context, detail);
    NoRegularizationCpuSerialPublication<Selection> publication(
        config, context, output_directory, checkpoint_writer, initial_hash,
        result, method, detail);
    cpu::w8c::LifecycleControl control;
    control.split_writer =
        config.run_role() == ApplicationRunRoleV1::split_writer;
    control.maximum_retries = config.retry_limit();
    const cpu::w8c::LifecycleReport report =
        cpu::w8c::RunCpuSerialLifecycleWithSchedule(
            method, publication, control,
            ApplicationCpuSerialSchedule{config});
    result.status = report.status;
    result.reason = report.reason;
    result.coordinates = report.coordinates;
    result.retries = report.retries;
    result.failure_class =
        report.failure == cpu::w8c::LifecycleFailure::none
        ? ApplicationRunResult::FailureClass::none
        : method.failure_class();
    if (report.failure != cpu::w8c::LifecycleFailure::none)
    {
      if (detail.empty()) detail = "persistent advance did not commit";
      AppendCpuFailureProvenance(
          adapter.evidence().last_attempt, detail);
    }
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

namespace detail
{

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CpuSerialBackendTag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  if (cpu::w8c::IsCaseId(request.case_id))
    return RunW8cCpuSerial(request, context);
  return RunCpu<AppCpuSerialSelection>(request, context);
}

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CpuRank2BackendTag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
  return RunCpu<AppCpuRank2Selection>(request, context);
}

} // namespace detail

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
  const bool typed_cpu = backend == ApplicationBackend::cpu_serial ||
      backend == ApplicationBackend::cpu_rank2;
  bool typed_cuda = false;
#if DG_IGR_HDG_APP_HAS_CUDA_RANK1
  typed_cuda = typed_cuda ||
      (backend == ApplicationBackend::cuda_rank1_one_gpu &&
       cuda_context != nullptr);
#endif
#if DG_IGR_HDG_APP_HAS_CUDA_RANK2
  typed_cuda = typed_cuda ||
      (backend == ApplicationBackend::cuda_rank2_one_gpu_host_staged &&
       cuda_context != nullptr);
#endif
  if (typed_cpu || typed_cuda)
  {
    const ApplicationValidation validation =
        ValidateApplicationOptions(derived);
    if (!validation.accepted())
    {
      return {RunStatus::invalid_request,
              FailureReason::invalid_resource_descriptor, {}, 0,
              ApplicationRunResult::FailureClass::configuration};
    }
    const HdgMethodRequestV1 typed_request{
        &derived, validation.config(), typed_cpu ? &cpu_context : nullptr,
        typed_cuda ? cuda_context : nullptr};
    return RunHdgMethodV1(typed_request).run;
  }
  switch (backend)
  {
    case ApplicationBackend::cpu_serial:
    case ApplicationBackend::cpu_rank2: break;
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

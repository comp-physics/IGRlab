#include "numerics/hdg/runtime/app/solver.hpp"

#include "solver_entry_detail.hpp"

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
#include "../cpu/hdg_cpu_serial_lifecycle.hpp"
#include "numerics/hdg/runtime/cpu/persistent_smooth_euler.hpp"
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
#include "numerics/hdg/runtime/cpu/persistent_sc25_scalar_igr.hpp"
#endif
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

#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
namespace dg::hdg::runtime::cpu::sc25::w8c_internal
{
[[gnu::visibility("hidden")]] bool PrimePersistentTrajectoryTransaction(
    PersistentSc25ScalarIgrCpuFoundation& owner,
    PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction& token) noexcept;
[[gnu::visibility("hidden")]] void RequestW8cInitialization(
    const PersistentSc25ScalarIgrCpuFoundation& owner) noexcept;
[[gnu::visibility("hidden")]] void RequestSmoothTransitionInitialization(
    const PersistentSc25ScalarIgrCpuFoundation& owner) noexcept;
[[gnu::visibility("hidden")]] void RequestSmoothTransitionInitialization(
    const PersistentSc25ScalarIgrCpuFoundation& owner,
    int required_steps, double time_step) noexcept;
}
#endif

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

bool ConsumeScalarRankStepFault(const char* const environment,
                                const std::uint64_t step, const int rank,
                                bool& consumed, int& source_rank) noexcept
{
  source_rank = -1;
  if (consumed) return false;
  const char* const raw = std::getenv(environment);
  if (raw == nullptr) return false;
  const std::string_view value{raw};
  const std::size_t separator = value.find(':');
  if (separator == std::string_view::npos) return false;
  std::uint64_t selected_step = 0;
  int selected_rank = -1;
  const auto step_result = std::from_chars(
      value.data(), value.data() + separator, selected_step);
  const auto rank_result = std::from_chars(
      value.data() + separator + 1, value.data() + value.size(),
      selected_rank);
  if (step_result.ec != std::errc{} ||
      step_result.ptr != value.data() + separator ||
      rank_result.ec != std::errc{} ||
      rank_result.ptr != value.data() + value.size())
    return false;
  source_rank = selected_rank;
  if (selected_step != step || selected_rank != rank) return false;
  consumed = true;
  return true;
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

bool GatherPhysicalInterior(const CpuCoefficientView& interior,
                            const MpiApplicationContextV1& context,
                            std::array<int, 2>& counts,
                            std::array<int, 2>& displacements,
                            std::vector<double>& global,
                            std::string& detail)
{
  const bool local_valid = interior.size <= static_cast<std::size_t>(
      std::numeric_limits<int>::max()) &&
      (interior.size == 0 || interior.data != nullptr);
  if (!CollectiveAll(local_valid, context.communicator()))
  {
    detail = "VTK rank-local interior extent is invalid";
    return false;
  }
  const int local_count = static_cast<int>(interior.size);
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
  const int gather = MPI_Gatherv(
      interior.data, local_count, MPI_DOUBLE, global.data(), counts.data(),
      displacements.data(), MPI_DOUBLE, 0, context.communicator());
  if (!CollectiveAll(gather == MPI_SUCCESS, context.communicator()))
  {
    detail = "MPI_Gatherv VTK interior state failed";
    return false;
  }
  return true;
}

bool MakeVtk(const ValidatedApplicationConfig& config,
             const int ranks,
             const std::array<int, 2>& counts,
             const std::array<int, 2>& displacements,
             const std::vector<double>& gathered,
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
    for (int rank = 0; rank < ranks; ++rank)
    {
      const int expected_elements =
          (global_elements * (rank + 1) + ranks - 1) / ranks -
          (global_elements * rank + ranks - 1) / ranks;
      if (counts[static_cast<std::size_t>(rank)] !=
          expected_elements * scalar_dofs * 4)
      {
        detail = "rank-ordered VTK state census is invalid";
        return false;
      }
    }
    std::array<int, 2> local_elements{};
    for (int element = 0; element < global_elements; ++element)
    {
      const int owner = element * ranks / global_elements;
      const int local_element =
          local_elements[static_cast<std::size_t>(owner)]++;
      const int local_scalar_dofs =
          counts[static_cast<std::size_t>(owner)] / 4;
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
          const int source = displacements[static_cast<std::size_t>(owner)] +
              component * local_scalar_dofs +
              local_element * scalar_dofs + scalar;
          const int encoded = vector_dofs[
              scalar + component * scalar_dofs];
          const int destination = encoded >= 0 ? encoded : -1 - encoded;
          const double sign = encoded >= 0 ? 1.0 : -1.0;
          if (source < 0 ||
              source >= displacements[static_cast<std::size_t>(owner)] +
                  counts[static_cast<std::size_t>(owner)] ||
              destination < 0 || destination >= field.Size())
          {
            detail = "rank-ordered VTK element map is out of range";
            return false;
          }
          field[destination] = sign *
              gathered[static_cast<std::size_t>(source)];
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
  std::vector<double> global;
  if (!GatherPhysicalInterior(snapshot.interior, context, counts,
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
      staged_ok = MakeVtk(config, context.ranks(), counts, displacements,
                          global, vtk, detail);
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

#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
namespace sc25 = cpu::sc25;

struct ScalarIgrPackedState final
{
  StateCoordinates coordinates{};
  std::vector<double> interior;
  std::vector<double> trace;
};

template <typename Range>
void AppendScalarRange(const Range& source, std::vector<double>& destination)
{
  destination.insert(destination.end(), source.begin(), source.end());
}

[[maybe_unused]] bool PackScalarIgrState(
    const sc25::PersistentTrajectoryState& state,
    const StateCoordinates& coordinates, ScalarIgrPackedState& output)
{
  try
  {
    ScalarIgrPackedState candidate;
    candidate.coordinates = coordinates;
    candidate.interior.reserve(state.u.size() + state.sigma.size() +
                               state.z.size());
    candidate.trace.reserve(state.uhat.size() + state.sigmahat.size());
    AppendScalarRange(state.u, candidate.interior);
    AppendScalarRange(state.sigma, candidate.interior);
    AppendScalarRange(state.z, candidate.interior);
    AppendScalarRange(state.uhat, candidate.trace);
    AppendScalarRange(state.sigmahat, candidate.trace);
    if (state.u.empty() || state.sigma.size() * 4 != state.u.size() ||
        state.z.size() != state.sigma.size() * 2 ||
        state.uhat.size() != state.sigmahat.size() * 4 ||
        candidate.interior.size() != state.u.size() + state.sigma.size() +
            state.z.size() || candidate.trace.size() !=
            state.uhat.size() + state.sigmahat.size())
      return false;
    output = std::move(candidate);
    return true;
  }
  catch (...) { return false; }
}

[[maybe_unused]] bool PackScalarIgrState(
    const sc25::PersistentRank2TrajectoryState& state,
    const StateCoordinates& coordinates, ScalarIgrPackedState& output)
{
  try
  {
    ScalarIgrPackedState candidate;
    candidate.coordinates = coordinates;
    candidate.interior.reserve(state.u.size() + state.sigma.size() +
                               state.z.size());
    candidate.trace.reserve(state.uhat.size() + state.sigmahat.size());
    AppendScalarRange(state.u, candidate.interior);
    AppendScalarRange(state.sigma, candidate.interior);
    AppendScalarRange(state.z, candidate.interior);
    AppendScalarRange(state.uhat, candidate.trace);
    AppendScalarRange(state.sigmahat, candidate.trace);
    if (candidate.interior.size() !=
            static_cast<std::size_t>(
                sc25::persistent_rank2_local_interior_unknowns) ||
        (candidate.trace.size() !=
             static_cast<std::size_t>(
                 sc25::persistent_rank2_rank0_owned_trace_unknowns) &&
         candidate.trace.size() !=
             static_cast<std::size_t>(
                 sc25::persistent_rank2_rank1_owned_trace_unknowns)))
      return false;
    output = std::move(candidate);
    return true;
  }
  catch (...) { return false; }
}

bool MakeScalarIgrVtk(const ValidatedApplicationConfig& config,
                      const int ranks,
                      const std::array<int, 2>& counts,
                      const std::array<int, 2>& displacements,
                      const std::vector<double>& gathered,
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
    translations[1][1] = 1.0;
    mfem::Mesh mesh = mfem::Mesh::MakePeriodic(
        original, original.CreatePeriodicVertexMapping(translations));
    mfem::L2_FECollection collection(
        config.polynomial_order(), 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    mfem::FiniteElementSpace u_space(&mesh, &collection, 4,
                                     mfem::Ordering::byNODES);
    mfem::FiniteElementSpace sigma_space(&mesh, &collection, 1,
                                         mfem::Ordering::byNODES);
    mfem::FiniteElementSpace z_space(&mesh, &collection, 2,
                                     mfem::Ordering::byNODES);
    mfem::GridFunction u(&u_space), sigma(&sigma_space), z(&z_space);
    u = 0.0;
    sigma = 0.0;
    z = 0.0;
    const int elements = static_cast<int>(config.mesh_n() * config.mesh_n());
    const int scalar_dofs = (config.polynomial_order() + 1) *
                            (config.polynomial_order() + 1);
    if ((ranks != 1 && ranks != 2) ||
        (ranks == 2 && elements != sc25::persistent_elements))
    {
      detail = "scalar-IGR VTK topology is outside the admitted rank layout";
      return false;
    }
    for (int rank = 0; rank < ranks; ++rank)
    {
      const int owned = ranks == 1 ? elements :
          sc25::persistent_rank2_local_elements;
      if (counts[static_cast<std::size_t>(rank)] !=
          owned * scalar_dofs * sc25::local_components)
      {
        detail = "scalar-IGR rank-ordered VTK census is invalid";
        return false;
      }
    }
    std::vector<std::uint8_t> global_seen(
        static_cast<std::size_t>(elements), 0);
    std::array<std::vector<std::uint8_t>, 2> local_seen{};
    for (int rank = 0; rank < ranks; ++rank)
    {
      const int owned = ranks == 1 ? elements :
          sc25::persistent_rank2_local_elements;
      local_seen[static_cast<std::size_t>(rank)].assign(owned, 0);
    }
    std::array<int, 2> observed_local_elements{};
    const int n = static_cast<int>(config.mesh_n());
    const int slab_width = n / 2;
    for (int ey = 0; ey < n; ++ey)
      for (int ex = 0; ex < n; ++ex)
    {
      const int element = ey * n + ex;
      const int owner = ranks == 1 ? 0 : (ex < slab_width ? 0 : 1);
      const int local_ex = ranks == 1 ? ex : ex - owner * slab_width;
      const int local_element = ey * (ranks == 1 ? n : slab_width) + local_ex;
      if (element < 0 || element >= elements || local_element < 0 ||
          local_element >= static_cast<int>(
              local_seen[static_cast<std::size_t>(owner)].size()) ||
          global_seen[static_cast<std::size_t>(element)] ||
          local_seen[static_cast<std::size_t>(owner)]
                    [static_cast<std::size_t>(local_element)])
      {
        detail = "scalar-IGR VTK x-slab element map is duplicate or invalid";
        return false;
      }
      global_seen[static_cast<std::size_t>(element)] = true;
      local_seen[static_cast<std::size_t>(owner)]
                [static_cast<std::size_t>(local_element)] = true;
      ++observed_local_elements[static_cast<std::size_t>(owner)];
      const int base = displacements[static_cast<std::size_t>(owner)];
      const int owned = counts[static_cast<std::size_t>(owner)] /
          (scalar_dofs * sc25::local_components);
      // The accepted packed layout is three field blocks, never seven global
      // component slabs: U[element][component][node], Sigma[element][node],
      // then z[element][component][node]. The serial physical-value oracle
      // locks this ordering for future CPU and CUDA gather implementations.
      const auto load = [&](mfem::FiniteElementSpace& space,
                            mfem::GridFunction& field,
                            const int components,
                            const int field_offset,
                            const int element_stride)
      {
        mfem::Array<int> dofs;
        space.GetElementVDofs(element, dofs);
        if (dofs.Size() != scalar_dofs * components) return false;
        for (int component = 0; component < components; ++component)
          for (int scalar = 0; scalar < scalar_dofs; ++scalar)
          {
            const int source = base +
                field_offset + local_element * element_stride +
                component * scalar_dofs + scalar;
            const int encoded = dofs[component * scalar_dofs + scalar];
            const int destination = encoded >= 0 ? encoded : -1 - encoded;
            if (source < base ||
                source >= base + counts[static_cast<std::size_t>(owner)] ||
                destination < 0 || destination >= field.Size())
              return false;
            field[destination] = (encoded >= 0 ? 1.0 : -1.0) *
                gathered[static_cast<std::size_t>(source)];
          }
        return true;
      };
      const int local_u = owned * scalar_dofs * 4;
      const int local_sigma = owned * scalar_dofs;
      if (!load(u_space, u, 4, 0, scalar_dofs * 4) ||
          !load(sigma_space, sigma, 1, local_u, scalar_dofs) ||
          !load(z_space, z, 2, local_u + local_sigma, scalar_dofs * 2))
      {
        detail = "scalar-IGR VTK element map is invalid";
        return false;
      }
    }
    if (!std::all_of(global_seen.begin(), global_seen.end(),
                     [](const std::uint8_t seen) { return seen == 1; }))
    {
      detail = "scalar-IGR VTK x-slab element map is incomplete";
      return false;
    }
    for (int rank = 0; rank < ranks; ++rank)
    {
      const int expected = ranks == 1 ? elements :
          sc25::persistent_rank2_local_elements;
      if (observed_local_elements[static_cast<std::size_t>(rank)] != expected)
      {
        detail = "scalar-IGR VTK x-slab local census is incomplete";
        return false;
      }
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

[[maybe_unused]] bool StageScalarIgrVtk(
    const ValidatedApplicationConfig& config,
    const ScalarIgrPackedState& state,
    const MpiApplicationContextV1& context,
    DurableGenerationTransactionV1& transaction,
    std::vector<std::uint8_t>& payload, std::string& detail)
{
  std::array<int, 2> counts{}, displacements{};
  std::vector<double> gathered;
  const CpuCoefficientView interior{state.interior.data(),
                                    state.interior.size()};
  if (!GatherPhysicalInterior(interior, context, counts, displacements,
                              gathered, detail))
    return false;
  bool staged = true;
  if (context.rank() == 0)
  {
    try
    {
      std::string vtk;
      staged = MakeScalarIgrVtk(config, context.ranks(), counts,
                                displacements, gathered, vtk, detail);
      if (staged)
      {
        payload.assign(vtk.begin(), vtk.end());
        staged = transaction.Stage(StepName(state.coordinates.step),
                                   payload.data(), payload.size(), detail) ==
            DurableIoStatus::ok;
      }
    }
    catch (...) { staged = false; }
  }
  return CollectiveAll(staged, context.communicator());
}
#endif

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

#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
template <bool Rank2>
struct ScalarIgrCpuTraits;

template <>
struct ScalarIgrCpuTraits<false> final
{
  using State = sc25::PersistentTrajectoryState;
  using Checkpoint = sc25::PersistentTrajectoryCheckpoint;
  using Transaction =
      sc25::PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction;

  static sc25::FoundationStatus Create(
      const sc25::FoundationConfig& config,
      std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>& output)
      noexcept
  {
    return sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
        MPI_COMM_WORLD, sc25::CpuSerialTag{}, config, output);
  }
  static sc25::TrajectoryStatus Initialize(
      sc25::PersistentSc25ScalarIgrCpuFoundation& owner) noexcept
  {
    return owner.InitializePersistentTrajectory();
  }
  static StateCoordinates Coordinates(
      const sc25::PersistentSc25ScalarIgrCpuFoundation& owner) noexcept
  {
    return owner.trajectory_evidence().coordinates;
  }
  static sc25::TrajectoryStatus Prepare(
      sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
      Transaction& transaction,
      const sc25::TrajectoryFailureInjection injection) noexcept
  {
    return owner.PreparePersistentTrajectoryTransaction(transaction,
                                                        injection);
  }
  static bool Prime(sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                    Transaction& transaction) noexcept
  {
    return sc25::w8c_internal::PrimePersistentTrajectoryTransaction(
        owner, transaction);
  }
  static bool Install(sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                      Transaction& transaction) noexcept
  {
    return owner.InstallPersistentTrajectoryTransaction(transaction);
  }
  static bool Revert(sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                     Transaction&& transaction) noexcept
  {
    return owner.RevertPersistentTrajectoryTransaction(
        std::move(transaction));
  }
  static bool Finalize(sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                       Transaction&& transaction) noexcept
  {
    return owner.FinalizePersistentTrajectoryTransaction(
        std::move(transaction));
  }
  static bool FinalizeReady(const Transaction& transaction) noexcept
  {
    return transaction.valid() && transaction.provisionally_installed();
  }
  static bool CopyState(const sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                        State& state) noexcept
  {
    return owner.CopyPublishedTrajectoryState(state);
  }
  static bool Import(sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                     const StateCoordinates& coordinates,
                     const State& state, Checkpoint& checkpoint) noexcept
  {
    return owner.ImportTrajectoryCheckpoint(coordinates, state, checkpoint);
  }
  static sc25::TrajectoryStatus Restore(
      sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
      const Checkpoint& checkpoint) noexcept
  {
    return owner.RestoreTrajectoryCheckpoint(checkpoint);
  }
};

template <>
struct ScalarIgrCpuTraits<true> final
{
  using State = sc25::PersistentRank2TrajectoryState;
  using Checkpoint = sc25::PersistentRank2TrajectoryCheckpoint;
  using Transaction =
      sc25::PersistentSc25ScalarIgrCpuFoundation::Rank2Transaction;

  static sc25::FoundationStatus Create(
      const sc25::FoundationConfig& config,
      std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation>& output)
      noexcept
  {
    return sc25::PersistentSc25ScalarIgrCpuFoundation::Create(
        MPI_COMM_WORLD, sc25::CpuMpiRank2Tag{}, config, output);
  }
  static sc25::TrajectoryStatus Initialize(
      sc25::PersistentSc25ScalarIgrCpuFoundation& owner) noexcept
  {
    return owner.InitializePersistentRank2Trajectory();
  }
  static StateCoordinates Coordinates(
      const sc25::PersistentSc25ScalarIgrCpuFoundation& owner) noexcept
  {
    return owner.rank2_trajectory_evidence().coordinates;
  }
  static sc25::TrajectoryStatus Prepare(
      sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
      Transaction& transaction,
      const sc25::TrajectoryFailureInjection injection) noexcept
  {
    return owner.PreparePersistentRank2TrajectoryTransaction(transaction,
                                                             injection);
  }
  static bool Prime(sc25::PersistentSc25ScalarIgrCpuFoundation&,
                    Transaction&) noexcept
  { return true; }
  static bool Install(sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                      Transaction& transaction) noexcept
  {
    return owner.InstallPersistentRank2TrajectoryTransaction(transaction);
  }
  static bool Revert(sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                     Transaction&& transaction) noexcept
  {
    return owner.RevertPersistentRank2TrajectoryTransaction(
        std::move(transaction));
  }
  static bool Finalize(sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                       Transaction&& transaction) noexcept
  {
    return owner.FinalizePersistentRank2TrajectoryTransaction(
        std::move(transaction));
  }
  static bool FinalizeReady(const Transaction& transaction) noexcept
  {
    return transaction.valid() && transaction.provisionally_installed();
  }
  static bool CopyState(const sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                        State& state) noexcept
  {
    return owner.CopyPublishedRank2TrajectoryState(state);
  }
  static bool Import(sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
                     const StateCoordinates& coordinates,
                     const State& state, Checkpoint& checkpoint) noexcept
  {
    return owner.ImportRank2TrajectoryCheckpoint(coordinates, state,
                                                 checkpoint);
  }
  static sc25::TrajectoryStatus Restore(
      sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
      const Checkpoint& checkpoint) noexcept
  {
    return owner.RestoreRank2TrajectoryCheckpoint(checkpoint);
  }
};

bool ScalarIgrAccepted(const sc25::TrajectoryStatus status) noexcept
{
  return status == sc25::TrajectoryStatus::HdgSc25Accepted ||
      status == sc25::TrajectoryStatus::HdgSc25AcceptedAfterOneRetry;
}

PersistentResourceDescriptor ScalarIgrResources(
    const ValidatedApplicationConfig& config,
    const sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
    const MpiApplicationContextV1& context) noexcept
{
  PersistentResourceDescriptor resources{};
  if (!owner.CopyPersistentResourceDescriptor(resources)) return {};
  resources.layout = StableIdentity::Copy(config.layout_id().view());
  if (context.ranks() == 2)
  {
    const auto aggregate = [&context](const StateHash& local,
                                      StateHash& global) noexcept
    {
      std::array<StateHash, 2> rank_ordered{};
      if (MPI_Allgather(local.bytes.data(),
                        static_cast<int>(local.bytes.size()), MPI_UINT8_T,
                        rank_ordered.data(),
                        static_cast<int>(local.bytes.size()), MPI_UINT8_T,
                        context.communicator()) != MPI_SUCCESS)
        return false;
      global.bytes = ComputeSha256(rank_ordered.data(),
                                   sizeof(rank_ordered));
      return global != StateHash{};
    };
    StateHash mesh{}, partition{}, topology{};
    if (!aggregate(resources.mesh_hash, mesh) ||
        !aggregate(resources.partition_hash, partition) ||
        !aggregate(resources.topology_hash, topology))
      return {};
    resources.mesh_hash = mesh;
    resources.partition_hash = partition;
    resources.topology_hash = topology;
  }
  return resources;
}

CheckpointRuntimeIdentityV1 ScalarIgrRuntimeIdentity(
    const PersistentResourceDescriptor& resources) noexcept
{
  CheckpointRuntimeIdentityV1 identity{};
  identity.physical_gpu_uuid = StableIdentity::Copy(no_physical_gpu_uuid_v1);
  identity.ownership_hash = resources.partition_hash;
  return identity;
}

bool GlobalScalarIgrStateIdentity(const ScalarIgrPackedState& state,
                                  const MpiApplicationContextV1& context,
                                  StateHash& identity) noexcept
{
  StateHash interior{}, trace{};
  if (!CpuContractPayloadHash(state.interior.data(), state.interior.size(),
                              context, interior) ||
      !CpuContractPayloadHash(state.trace.data(), state.trace.size(), context,
                              trace))
    return false;
  std::array<std::uint8_t, 64> ordered{};
  std::copy(interior.bytes.begin(), interior.bytes.end(), ordered.begin());
  std::copy(trace.bytes.begin(), trace.bytes.end(), ordered.begin() + 32);
  identity.bytes = ComputeSha256(ordered.data(), ordered.size());
  return identity != StateHash{};
}

bool GlobalScalarIgrCoordinatesIdentity(
    const StateCoordinates& coordinates,
    const MpiApplicationContextV1& context) noexcept
{
  std::array<StateCoordinates, 2> gathered{};
  const bool collected = MPI_Allgather(
      &coordinates, static_cast<int>(sizeof(coordinates)), MPI_BYTE,
      gathered.data(), static_cast<int>(sizeof(coordinates)), MPI_BYTE,
      context.communicator()) == MPI_SUCCESS;
  if (!collected || context.ranks() < 1 || context.ranks() > 2) return false;
  for (int rank = 1; rank < context.ranks(); ++rank)
    if (!::dg::hdg::runtime::detail::SameCoordinates(
            gathered[0], gathered[static_cast<std::size_t>(rank)]))
      return false;
  return true;
}

template <typename Checkpoint>
bool StageScalarIgrCheckpoint(
    const ValidatedApplicationConfig& config,
    const Checkpoint& private_checkpoint,
    const ScalarIgrPackedState& state,
    const PersistentResourceDescriptor& resources,
    const MpiApplicationContextV1& context,
    DurableGenerationTransactionV1& transaction,
    std::vector<std::uint8_t>& encoded, std::string& detail)
{
  std::vector<double> checkpoint_interior;
  const std::uint64_t scalar_dofs_per_element =
      static_cast<std::uint64_t>(config.polynomial_order() + 1) *
      static_cast<std::uint64_t>(config.polynomial_order() + 1);
  const bool layout_ready = detail::ScalarIgrFieldBlocksToComponentMajorV1(
      state.interior, scalar_dofs_per_element, checkpoint_interior);
  if (!CollectiveAll(layout_ready, context.communicator()))
  {
    detail = "scalar-IGR checkpoint interior layout conversion failed";
    return false;
  }
  CheckpointDescriptor local = private_checkpoint.descriptor;
  local.tuple = config.tuple();
  local.mfem_pin = resources.mfem_pin;
  local.mesh_hash = resources.mesh_hash;
  local.partition_hash = resources.partition_hash;
  local.topology_hash = resources.topology_hash;
  local.rank_count = resources.rank_count;
  local.polynomial_order = resources.polynomial_order;
  local.layout = StableIdentity::Copy(config.layout_id().view());
  local.precision = config.tuple().precision;
  std::array<CheckpointDescriptor, 2> descriptors{};
  const int descriptor_status = MPI_Gather(
      &local, static_cast<int>(sizeof(local)), MPI_BYTE, descriptors.data(),
      static_cast<int>(sizeof(local)), MPI_BYTE, 0, context.communicator());
  if (!CollectiveAll(descriptor_status == MPI_SUCCESS,
                     context.communicator()))
    return false;
  std::array<int, 2> interior_counts{}, interior_displacements{};
  std::array<int, 2> trace_counts{}, trace_displacements{};
  std::vector<double> interior, trace;
  if (!GatherDoubles(checkpoint_interior.data(), checkpoint_interior.size(), context,
                     interior_counts, interior_displacements, interior,
                     detail) ||
      !GatherDoubles(state.trace.data(), state.trace.size(), context,
                     trace_counts, trace_displacements, trace, detail))
    return false;
  bool staged = true;
  if (context.rank() == 0)
  {
    try
    {
      const CheckpointCoefficientSchemaV1 schema =
          CheckpointCoefficientSchema(config);
      std::array<StagedStateViewV1, 2> ranks{};
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        descriptors[index].interior_hash.bytes = ComputeSha256(
            interior.data() + interior_displacements[index],
            static_cast<std::size_t>(interior_counts[index]) * sizeof(double));
        descriptors[index].trace_hash.bytes = ComputeSha256(
            trace.data() + trace_displacements[index],
            static_cast<std::size_t>(trace_counts[index]) * sizeof(double));
        ranks[index].rank = static_cast<std::uint32_t>(rank);
        ranks[index].interior = {
            interior.data() + interior_displacements[index],
            static_cast<std::uint64_t>(interior_counts[index]),
            schema.interior_component_count,
            static_cast<std::uint64_t>(interior_counts[index]) /
                schema.interior_component_count,
            CoefficientOrderingV1::component_major};
        ranks[index].owned_true_trace = {
            trace.data() + trace_displacements[index],
            static_cast<std::uint64_t>(trace_counts[index]),
            schema.trace_component_count,
            static_cast<std::uint64_t>(trace_counts[index]) /
                schema.trace_component_count,
            CoefficientOrderingV1::component_major};
      }
      const auto captured = CaptureCheckpointPiecesV1(
          descriptors.data(), static_cast<std::size_t>(context.ranks()),
          config, ScalarIgrRuntimeIdentity(resources));
      const CheckpointCodecStatus codec = captured
          ? EncodeCheckpointV1(*captured, ranks.data(),
                               static_cast<std::size_t>(context.ranks()),
                               encoded)
          : CheckpointCodecStatus::invalid_descriptor;
      staged = codec == CheckpointCodecStatus::ok &&
          transaction.Stage("restart-checkpoint.v2", encoded.data(),
                            encoded.size(), detail) == DurableIoStatus::ok &&
          transaction.AddAlias(
              std::string{config.checkpoint_out().identity().view()},
              "restart-checkpoint.v2", detail) == DurableIoStatus::ok;
      if (!staged && detail.empty())
        detail = "scalar-IGR checkpoint encode failed";
    }
    catch (...) { staged = false; }
  }
  return CollectiveAll(staged, context.communicator());
}

template <bool Rank2>
bool RestoreScalarIgrCheckpoint(
    const ValidatedApplicationConfig& config,
    const PersistentResourceDescriptor& resources,
    const MpiApplicationContextV1& context,
    sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
    std::uint32_t& prior_retries, std::string& detail)
{
  using Traits = ScalarIgrCpuTraits<Rank2>;
  ScenarioCoefficientExtentsV1 extents{};
  if (!DeriveScenarioCoefficientExtentsV1(
          config, static_cast<std::uint32_t>(context.ranks()), extents))
    return false;
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
        const auto& shape = extents.ranks[index];
        interiors[index].resize(shape.interior_coefficient_count);
        traces[index].resize(shape.trace_coefficient_count);
        buffers[index].rank = static_cast<std::uint32_t>(rank);
        buffers[index].interior = {
            interiors[index].data(), interiors[index].size(),
            interiors[index].size(), extents.interior_component_count,
            shape.interior_dofs_per_component,
            CoefficientOrderingV1::component_major};
        buffers[index].owned_true_trace = {
            traces[index].data(), traces[index].size(), traces[index].size(),
            extents.trace_component_count,
            shape.trace_dofs_per_component,
            CoefficientOrderingV1::component_major};
      }
      if (decoded)
      {
        const CheckpointCodecStatus status = DecodeCheckpointV1(
            {encoded.data(), encoded.size()}, config, resources,
            ScalarIgrRuntimeIdentity(resources), buffers.data(),
            static_cast<std::size_t>(context.ranks()), metadata);
        decoded = status == CheckpointCodecStatus::ok;
      }
    }
    catch (...) { decoded = false; }
  }
  if (!CollectiveAll(decoded, context.communicator())) return false;
  const bool descriptor_broadcast = MPI_Bcast(
      &metadata.descriptor, static_cast<int>(sizeof(metadata.descriptor)),
      MPI_BYTE, 0, context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(descriptor_broadcast, context.communicator()))
  {
    detail = "scalar-IGR restart descriptor broadcast failed";
    return false;
  }

  const auto& shape = extents.ranks[static_cast<std::size_t>(context.rank())];
  std::vector<double> local_interior, local_trace;
  std::array<int, 2> interior_counts{}, interior_displacements{};
  std::array<int, 2> trace_counts{}, trace_displacements{};
  std::vector<double> flat_interior, flat_trace;
  typename Traits::State state{};
  bool prepared = true;
  int restart_fault_source = -1;
  static bool restart_fault_consumed = false;
  try
  {
    local_interior.resize(shape.interior_coefficient_count);
    local_trace.resize(shape.trace_coefficient_count);
    if constexpr (Rank2)
    {
      const std::size_t scalar_interior = local_interior.size() /
          sc25::local_components;
      const std::size_t scalar_trace = local_trace.size() /
          sc25::trace_components;
      state.u.resize(4 * scalar_interior);
      state.sigma.resize(scalar_interior);
      state.z.resize(2 * scalar_interior);
      state.uhat.resize(4 * scalar_trace);
      state.sigmahat.resize(scalar_trace);
    }
    if (context.rank() == 0)
    {
      int ioffset = 0, toffset = 0;
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
        if (ioffset > std::numeric_limits<int>::max() -
                          interior_counts[index] ||
            toffset > std::numeric_limits<int>::max() - trace_counts[index])
          throw std::length_error{"scalar-IGR restart MPI displacement"};
        interior_displacements[index] = ioffset;
        trace_displacements[index] = toffset;
        ioffset += interior_counts[index];
        toffset += trace_counts[index];
      }
      flat_interior.resize(static_cast<std::size_t>(ioffset));
      flat_trace.resize(static_cast<std::size_t>(toffset));
      for (int rank = 0; rank < context.ranks(); ++rank)
      {
        const std::size_t index = static_cast<std::size_t>(rank);
        std::copy(interiors[index].begin(), interiors[index].end(),
                  flat_interior.begin() + interior_displacements[index]);
        std::copy(traces[index].begin(), traces[index].end(),
                  flat_trace.begin() + trace_displacements[index]);
      }
    }
  }
  catch (...) { prepared = false; }
  if constexpr (Rank2)
  {
    prepared = prepared && !ConsumeScalarRankStepFault(
        "DG_IGR_HDG_TEST_SCALAR_RESTART_PREPARATION_FAILURE",
        metadata.descriptor.coordinates.step, context.rank(),
        restart_fault_consumed, restart_fault_source);
  }
  if (!CollectiveAll(prepared, context.communicator()))
  {
    detail = restart_fault_source >= 0
        ? "scalar-IGR restart preparation failure source_rank=" +
              std::to_string(restart_fault_source)
        : "scalar-IGR restart preparation allocation failed";
    return false;
  }
  const bool interior_scattered = MPI_Scatterv(
      flat_interior.data(), interior_counts.data(),
      interior_displacements.data(), MPI_DOUBLE, local_interior.data(),
      static_cast<int>(local_interior.size()), MPI_DOUBLE, 0,
      context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(interior_scattered, context.communicator()))
  {
    detail = "scalar-IGR restart interior scatter failed";
    return false;
  }
  const bool trace_scattered = MPI_Scatterv(
      flat_trace.data(), trace_counts.data(), trace_displacements.data(),
      MPI_DOUBLE, local_trace.data(), static_cast<int>(local_trace.size()),
      MPI_DOUBLE, 0, context.communicator()) == MPI_SUCCESS;
  if (!CollectiveAll(trace_scattered, context.communicator()))
  {
    detail = "scalar-IGR restart trace scatter failed";
    return false;
  }

  std::vector<double> internal_interior;
  const std::uint64_t scalar_dofs_per_element =
      static_cast<std::uint64_t>(config.polynomial_order() + 1) *
      static_cast<std::uint64_t>(config.polynomial_order() + 1);
  const bool layout_ready = detail::ScalarIgrComponentMajorToFieldBlocksV1(
      local_interior, scalar_dofs_per_element, internal_interior);
  if (!CollectiveAll(layout_ready, context.communicator()))
  {
    detail = "scalar-IGR restart interior layout conversion failed";
    return false;
  }

  const std::size_t scalar_interior = internal_interior.size() /
      sc25::local_components;
  const std::size_t scalar_trace = local_trace.size() /
      sc25::trace_components;
  const auto assign = [](auto& destination, const double* source,
                         const std::size_t count)
  {
    std::copy_n(source, count, destination.begin());
  };
  if constexpr (Rank2)
  {
    assign(state.u, internal_interior.data(), 4 * scalar_interior);
    assign(state.sigma, internal_interior.data() + 4 * scalar_interior,
           scalar_interior);
    assign(state.z, internal_interior.data() + 5 * scalar_interior,
           2 * scalar_interior);
    assign(state.uhat, local_trace.data(), 4 * scalar_trace);
    assign(state.sigmahat, local_trace.data() + 4 * scalar_trace,
           scalar_trace);
  }
  else
  {
    std::copy_n(internal_interior.data(), state.u.size(), state.u.begin());
    std::copy_n(internal_interior.data() + state.u.size(), state.sigma.size(),
                state.sigma.begin());
    std::copy_n(internal_interior.data() + state.u.size() + state.sigma.size(),
                state.z.size(), state.z.begin());
    std::copy_n(local_trace.data(), state.uhat.size(), state.uhat.begin());
    std::copy_n(local_trace.data() + state.uhat.size(), state.sigmahat.size(),
                state.sigmahat.begin());
  }
  typename Traits::Checkpoint checkpoint;
  const bool imported = Traits::Import(
      owner, metadata.descriptor.coordinates, state, checkpoint);
  if (!CollectiveAll(imported, context.communicator()))
  {
    detail = "scalar-IGR restart import failed";
    return false;
  }
  const bool restored = ScalarIgrAccepted(Traits::Restore(owner, checkpoint));
  if (!CollectiveAll(restored, context.communicator()))
  {
    detail = "scalar-IGR restart restore failed";
    return false;
  }
  if (restored) prior_retries = scalar_igr_sc25_v3_retry_limit;
  return restored;
}
#endif

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
  const bool w8c_case = cpu::w8c::IsCaseId(config.case_id().view());
  writer.injection = config.scalar_igr() && !w8c_case
      ? config.injection() : FailureInjectionControlV1{};
  writer.retry_limit = config.scalar_igr() && !w8c_case
      ? config.retry_limit() : 0;
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

#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
struct ScalarIgrInjectionAudit final
{
  int configured_source_rank = -1;
  int observed_origin_rank = -1;
  int local_injection_count = 0;
  int non_origin_local_injection_count = 0;
  int collective_decisions = 0;
};

std::string ScalarIgrMetadata(const ValidatedApplicationConfig& config,
                              const ApplicationRunResult& result,
                              const StateHash& initial_hash,
                              const StateHash& final_hash,
                              const ScalarIgrInjectionAudit& injection,
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
      << "  \"fields\": [\"U\", \"Sigma\", \"z\"],\n"
      << "  \"dt\": " << config.dt() << ",\n"
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
      << "  \"initial_state_sha256\": \"" << Hex(initial_hash) << "\",\n"
      << "  \"final_state_sha256\": \"" << Hex(final_hash) << "\",\n"
      << "  \"diagnostic\": \"" << detail << "\"\n"
      << "}\n";
  return output.str();
}

sc25::TrajectoryFailureInjection ScalarIgrNumericalInjection(
    const FailurePhase phase) noexcept
{
  switch (phase)
  {
    case FailurePhase::Assembly:
    case FailurePhase::Condensation:
      return sc25::TrajectoryFailureInjection::local_factor;
    case FailurePhase::TraceSolve:
      return sc25::TrajectoryFailureInjection::trace_linear;
    case FailurePhase::Recovery:
      return sc25::TrajectoryFailureInjection::admissibility;
    case FailurePhase::Diagnostics:
      return sc25::TrajectoryFailureInjection::newton_iteration_limit;
    default: return sc25::TrajectoryFailureInjection::none;
  }
}

template <bool Rank2>
class TracedScalarIgrCpuSerialMethod final
{
public:
  using Traits = ScalarIgrCpuTraits<Rank2>;
  struct Transaction final
  {
    typename Traits::Transaction numerical;
    bool prescribed = false;
    bool app_injected = false;
    bool private_preparation_injected = false;
    bool installed = false;
  };

  TracedScalarIgrCpuSerialMethod(
      sc25::PersistentSc25ScalarIgrCpuFoundation& owner,
      const ValidatedApplicationConfig& config,
      MpiApplicationContextV1& context,
      ScalarIgrInjectionAudit& injection_audit,
      std::string& detail) noexcept
      : owner_(owner), config_(config), context_(context),
        injection_audit_(injection_audit), detail_(detail)
  {}

  bool Initialize() const noexcept { return true; }
  bool PrimeTransaction(Transaction& transaction) noexcept
  {
    transaction.prescribed = false;
    transaction.app_injected = false;
    transaction.private_preparation_injected = false;
    transaction.installed = false;
    return Traits::Prime(owner_, transaction.numerical);
  }

  bool Prepare(Transaction& transaction, const bool,
               const std::uint64_t target) noexcept
  {
    transaction.prescribed = false;
    transaction.app_injected = false;
    transaction.private_preparation_injected = false;
    transaction.installed = false;
    const bool prescribed = !injected_ &&
        config_.injection().mode == FailureInjectionModeV1::prescribed &&
        config_.injection().target_step == target;
    transaction.prescribed = prescribed;
    const int prescribed_source_rank =
        static_cast<int>(config_.injection().source_rank);
    const bool local_prescribed = prescribed &&
        context_.rank() == prescribed_source_rank;
    const FailurePhase phase = prescribed ? config_.injection().phase
                                          : FailurePhase::Setup;
    const sc25::TrajectoryFailureInjection prescribed_numerical = prescribed
        ? ScalarIgrNumericalInjection(phase)
        : sc25::TrajectoryFailureInjection::none;
    const sc25::TrajectoryFailureInjection numerical = local_prescribed
        ? prescribed_numerical : sc25::TrajectoryFailureInjection::none;
    if (prescribed)
    {
      const std::array<int, 3> local_decision{{
          local_prescribed ? 1 : 0,
          local_prescribed ? context_.rank() + 1 : 0,
          local_prescribed && context_.rank() != prescribed_source_rank
              ? 1 : 0}};
      std::array<int, 3> collective_decision{};
      const bool reduced = MPI_Allreduce(
          local_decision.data(), collective_decision.data(),
          static_cast<int>(local_decision.size()), MPI_INT, MPI_SUM,
          context_.communicator()) == MPI_SUCCESS;
      const bool decision_valid = reduced && collective_decision[0] == 1 &&
          collective_decision[1] == prescribed_source_rank + 1 &&
          collective_decision[2] == 0;
      if (!CollectiveAll(decision_valid, context_.communicator()))
      {
        detail_ = "scalar-IGR injection origin consensus failed";
        failure_class_ = ApplicationRunResult::FailureClass::backend;
        return false;
      }
      injection_audit_.observed_origin_rank = collective_decision[1] - 1;
      injection_audit_.local_injection_count += collective_decision[0];
      injection_audit_.non_origin_local_injection_count +=
          collective_decision[2];
      ++injection_audit_.collective_decisions;
    }
    const sc25::TrajectoryStatus status = Traits::Prepare(
        owner_, transaction.numerical, numerical);
    const bool locally_accepted = ScalarIgrAccepted(status);
    const bool accepted = Rank2
        ? CollectiveAll(locally_accepted, context_.communicator())
        : locally_accepted;
    if (!accepted)
    {
      injected_ = injected_ || prescribed;
      prepare_retryable_ = prescribed;
      failure_class_ = prescribed
          ? ApplicationRunResult::FailureClass::injected
          : ApplicationRunResult::FailureClass::scientific;
      detail_ = "scalar-IGR numerical prepare failed";
      return false;
    }
    transaction.app_injected = prescribed &&
        prescribed_numerical == sc25::TrajectoryFailureInjection::none;
    prepare_retryable_ = false;
    return true;
  }

  cpu::w8c::LifecycleAttemptOutcome PrepareFailure(
      const Transaction&) const noexcept
  {
    return {true, prepare_retryable_, false};
  }

  bool Install(Transaction& transaction) noexcept
  {
    transaction.installed = Traits::Install(owner_, transaction.numerical);
    if constexpr (Rank2)
      return CollectiveAll(transaction.installed, context_.communicator());
    return transaction.installed;
  }

  cpu::w8c::LifecycleAttemptOutcome AbortPrepared(
      Transaction&& transaction, const cpu::w8c::LifecycleFailure failure,
      const bool durable_reverted) noexcept
  {
    if (!transaction.installed)
      transaction.installed = Traits::Install(owner_, transaction.numerical);
    const bool local_reverted = transaction.installed &&
        Traits::Revert(owner_, std::move(transaction.numerical));
    const bool reverted = Rank2
        ? CollectiveAll(local_reverted, context_.communicator())
        : local_reverted;
    injected_ = injected_ || transaction.prescribed;
    const bool transaction_injected = transaction.app_injected ||
        transaction.private_preparation_injected;
    failure_class_ = !durable_reverted || !reverted
        ? ApplicationRunResult::FailureClass::backend
        : (transaction_injected
               ? ApplicationRunResult::FailureClass::injected
               : (failure == cpu::w8c::LifecycleFailure::staging
                      ? ApplicationRunResult::FailureClass::filesystem
                      : ApplicationRunResult::FailureClass::backend));
    return {durable_reverted && reverted,
            reverted && transaction_injected, false};
  }

  cpu::w8c::LifecycleAttemptOutcome RevertInstalled(
      Transaction&& transaction, const cpu::w8c::LifecycleFailure failure,
      const bool durable_reverted) noexcept
  {
    const bool local_reverted = transaction.installed &&
        Traits::Revert(owner_, std::move(transaction.numerical));
    const bool reverted = Rank2
        ? CollectiveAll(local_reverted, context_.communicator())
        : local_reverted;
    injected_ = injected_ || transaction.prescribed;
    const bool transaction_injected = transaction.app_injected ||
        transaction.private_preparation_injected;
    if (!durable_reverted || !reverted ||
        failure == cpu::w8c::LifecycleFailure::rollback)
      failure_class_ = ApplicationRunResult::FailureClass::backend;
    else if (transaction_injected)
      failure_class_ = ApplicationRunResult::FailureClass::injected;
    else if (failure == cpu::w8c::LifecycleFailure::publication)
      failure_class_ = ApplicationRunResult::FailureClass::filesystem;
    else
      failure_class_ = ApplicationRunResult::FailureClass::backend;
    return {durable_reverted && reverted,
            reverted && transaction_injected, false};
  }

  void Finalize(Transaction&& transaction) noexcept
  {
    const bool local_finalized = transaction.installed &&
        Traits::Finalize(owner_, std::move(transaction.numerical));
    const bool finalized = CollectiveAll(
        local_finalized, context_.communicator());
    if (!finalized) std::abort();
    injected_ = injected_ || transaction.prescribed;
  }

  RunStatus Status() const noexcept
  {
    if constexpr (Rank2)
      return RunStatus::terminal_failure;
    return owner_.trajectory_evidence().contract_status;
  }
  FailureReason Reason() const noexcept
  {
    return failure_class_ == ApplicationRunResult::FailureClass::none
        ? FailureReason::none : FailureReason::backend_failure;
  }
  StateCoordinates Coordinates() const noexcept
  { return Traits::Coordinates(owner_); }
  constexpr bool ForceTerminalFailure() const noexcept { return true; }
  void OnRetry(cpu::w8c::LifecycleFailure, std::uint32_t) const noexcept {}

  ApplicationRunResult::FailureClass failure_class() const noexcept
  { return failure_class_; }

private:
  sc25::PersistentSc25ScalarIgrCpuFoundation& owner_;
  const ValidatedApplicationConfig& config_;
  MpiApplicationContextV1& context_;
  ScalarIgrInjectionAudit& injection_audit_;
  std::string& detail_;
  ApplicationRunResult::FailureClass failure_class_ =
      ApplicationRunResult::FailureClass::none;
  bool injected_ = false;
  bool prepare_retryable_ = false;
};

template <bool Rank2>
class TracedScalarIgrCpuSerialPublication final
{
public:
  using Method = TracedScalarIgrCpuSerialMethod<Rank2>;
  using Traits = typename Method::Traits;

  TracedScalarIgrCpuSerialPublication(
      const ValidatedApplicationConfig& config,
      const PersistentResourceDescriptor& resources,
      MpiApplicationContextV1& context,
      const std::filesystem::path& output_directory,
      const std::optional<ValidatedApplicationConfig>& checkpoint_writer,
      const StateHash& initial_hash, ApplicationRunResult& result,
      const ScalarIgrInjectionAudit& injection_audit,
      std::string& detail) noexcept
      : config_(config), resources_(resources), context_(context),
        output_directory_(output_directory),
        checkpoint_writer_(checkpoint_writer), initial_hash_(initial_hash),
        result_(result), injection_audit_(injection_audit), detail_(detail)
  {}

  bool Restore(Method&) const noexcept { return true; }
  bool PublishInitial(Method&) const noexcept
  { return true; }

  bool Stage(
      Method&, typename Method::Transaction& method_transaction,
      const std::uint64_t target, const bool output, const bool checkpoint,
      const bool terminal, const std::uint32_t retries) noexcept
  {
    const auto stage_impl = [&]() -> bool
    {
      transaction_.emplace();
      active_ = true;
      vtk_.clear();
      checkpoint_.clear();
      metadata_.clear();
      bool staged = PackScalarIgrState(
          method_transaction.numerical.candidate_state(),
          method_transaction.numerical.future_checkpoint()
              .descriptor.coordinates,
          candidate_);
      if constexpr (Rank2)
      {
        static bool candidate_pack_fault_consumed = false;
        int candidate_pack_fault_source = -1;
        const bool local_candidate_pack_fault = ConsumeScalarRankStepFault(
            "DG_IGR_HDG_TEST_SCALAR_CANDIDATE_PACK_FAILURE", target,
            context_.rank(), candidate_pack_fault_consumed,
            candidate_pack_fault_source);
        const int local_count = local_candidate_pack_fault ? 1 : 0;
        const int local_source = local_candidate_pack_fault
            ? context_.rank() + 1 : 0;
        int collective_count = 0;
        int collective_source = 0;
        const bool reduced = MPI_Allreduce(
            &local_count, &collective_count, 1, MPI_INT, MPI_SUM,
            context_.communicator()) == MPI_SUCCESS &&
            MPI_Allreduce(&local_source, &collective_source, 1, MPI_INT,
                          MPI_SUM, context_.communicator()) == MPI_SUCCESS;
        staged = staged && reduced && collective_count <= 1 &&
            !local_candidate_pack_fault;
        method_transaction.private_preparation_injected =
            reduced && collective_count == 1;
        if (method_transaction.private_preparation_injected)
          detail_ = "scalar-IGR candidate pack failure source_rank=" +
              std::to_string(collective_source - 1);
      }
      staged = CollectiveAll(staged, context_.communicator());
      if (staged)
        staged = CollectiveAll(
            GlobalScalarIgrCoordinatesIdentity(
                candidate_.coordinates, context_),
            context_.communicator());
      if (!staged && !method_transaction.private_preparation_injected)
        detail_ = "scalar-IGR candidate state pack failed";
      if (staged)
        staged = BeginRootTransaction(
            *transaction_, output_directory_, target, context_, detail_);
      if (staged && output)
        staged = StageScalarIgrVtk(
            config_, candidate_, context_, *transaction_, vtk_, detail_);
      if (staged && checkpoint)
        staged = checkpoint_writer_ && StageScalarIgrCheckpoint(
            *checkpoint_writer_,
            method_transaction.numerical.future_checkpoint(), candidate_,
            resources_, context_, *transaction_, checkpoint_, detail_);
      if (staged && terminal)
      {
        StateHash final_hash{};
        bool metadata_ready = GlobalScalarIgrStateIdentity(
            candidate_, context_, final_hash);
        if constexpr (Rank2)
        {
          static bool metadata_fault_consumed = false;
          int metadata_fault_source = -1;
          const bool local_metadata_fault = ConsumeScalarRankStepFault(
              "DG_IGR_HDG_TEST_SCALAR_METADATA_PREPARATION_FAILURE", target,
              context_.rank(), metadata_fault_consumed,
              metadata_fault_source);
          const int local_count = local_metadata_fault ? 1 : 0;
          const int local_source = local_metadata_fault
              ? context_.rank() + 1 : 0;
          int collective_count = 0;
          int collective_source = 0;
          const bool reduced = MPI_Allreduce(
              &local_count, &collective_count, 1, MPI_INT, MPI_SUM,
              context_.communicator()) == MPI_SUCCESS &&
              MPI_Allreduce(&local_source, &collective_source, 1, MPI_INT,
                            MPI_SUM, context_.communicator()) == MPI_SUCCESS;
          metadata_ready = metadata_ready && reduced &&
              collective_count <= 1 && !local_metadata_fault;
          method_transaction.private_preparation_injected =
              method_transaction.private_preparation_injected ||
              (reduced && collective_count == 1);
          if (reduced && collective_count == 1)
            detail_ = "scalar-IGR metadata preparation failure source_rank=" +
                std::to_string(collective_source - 1);
        }
        if (context_.rank() == 0 && metadata_ready)
        {
          try
          {
            ApplicationRunResult anticipated = result_;
            anticipated.status = RunStatus::committed;
            anticipated.reason = FailureReason::none;
            anticipated.coordinates = candidate_.coordinates;
            anticipated.retries = retries;
            const std::string text = ScalarIgrMetadata(
                config_, anticipated, initial_hash_, final_hash,
                injection_audit_, detail_);
            metadata_.assign(text.begin(), text.end());
            metadata_ready = transaction_->Stage(
                config_.run_role() == ApplicationRunRoleV1::split_writer
                    ? "run-metadata-checkpoint.json" : "run-metadata.json",
                metadata_.data(), metadata_.size(), detail_) ==
                DurableIoStatus::ok;
          }
          catch (...)
          {
            metadata_ready = false;
            detail_ = Rank2
                ? "scalar-IGR root metadata preparation failed"
                : "scalar-IGR durable staging failed";
          }
        }
        staged = CollectiveAll(metadata_ready, context_.communicator());
      }
      return staged;
    };
    if constexpr (Rank2) return stage_impl();
    try { return stage_impl(); }
    catch (...)
    {
      detail_ = "scalar-IGR durable staging failed";
      return false;
    }
  }

  bool ReadyToPublish(
      Method&, typename Method::Transaction& transaction,
      const std::uint32_t) noexcept
  {
    if (transaction.app_injected) return false;
    if constexpr (Rank2)
    {
      bool local_ready = Traits::FinalizeReady(transaction.numerical);
      static bool finalize_fault_consumed = false;
      int finalize_fault_source = -1;
      const bool finalize_fault = ConsumeScalarRankStepFault(
          "DG_IGR_HDG_TEST_SCALAR_FINALIZE_PREPARATION_FAILURE",
          candidate_.coordinates.step, context_.rank(),
          finalize_fault_consumed, finalize_fault_source);
      local_ready = local_ready && !finalize_fault;
      const int local_count = finalize_fault ? 1 : 0;
      int collective_count = 0;
      const bool reduced = MPI_Allreduce(
          &local_count, &collective_count, 1, MPI_INT, MPI_SUM,
          context_.communicator()) == MPI_SUCCESS;
      transaction.private_preparation_injected = reduced &&
          collective_count == 1;
      const bool collectively_ready = CollectiveAll(
          local_ready && reduced && collective_count <= 1,
          context_.communicator());
      if (!collectively_ready)
        detail_ = transaction.private_preparation_injected &&
                finalize_fault_source >= 0
            ? "scalar-IGR finalize preparation failure source_rank=" +
                  std::to_string(finalize_fault_source)
            : "scalar-IGR finalize readiness consensus failed";
      return collectively_ready;
    }
    return true;
  }

  bool Publish(
      Method&, const typename Method::Transaction&) noexcept
  {
    return active_ &&
        PublishRootTransaction(*transaction_, context_, detail_);
  }

  bool Acknowledge(
      Method&, const typename Method::Transaction& transaction,
      const std::uint32_t) noexcept
  {
    const bool ready =
        Traits::FinalizeReady(transaction.numerical);
    const bool acknowledged = CollectiveAll(ready, context_.communicator());
    if (!acknowledged)
      detail_ = "scalar-IGR finalize readiness consensus failed";
    return acknowledged;
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
  const PersistentResourceDescriptor& resources_;
  MpiApplicationContextV1& context_;
  const std::filesystem::path& output_directory_;
  const std::optional<ValidatedApplicationConfig>& checkpoint_writer_;
  const StateHash& initial_hash_;
  ApplicationRunResult& result_;
  const ScalarIgrInjectionAudit& injection_audit_;
  std::string& detail_;
  std::optional<DurableGenerationTransactionV1> transaction_;
  ScalarIgrPackedState candidate_;
  std::vector<std::uint8_t> vtk_;
  std::vector<std::uint8_t> checkpoint_;
  std::vector<std::uint8_t> metadata_;
  bool active_ = false;
};

template <bool Rank2>
ApplicationRunResult RunScalarIgrCpu(const ApplicationOptionsView& request,
                                     MpiApplicationContextV1& context)
{
  using Traits = ScalarIgrCpuTraits<Rank2>;
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
  if (context.ranks() != (Rank2 ? 2 : 1))
  {
    result.status = RunStatus::invalid_request;
    result.reason = FailureReason::invalid_resource_descriptor;
    result.failure_class = ApplicationRunResult::FailureClass::configuration;
    return result;
  }
  ScalarIgrInjectionAudit injection_audit;
  if (config.injection().mode == FailureInjectionModeV1::prescribed)
    injection_audit.configured_source_rank = config.injection().source_rank;

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
      output_ready = std::filesystem::is_directory(
          std::filesystem::symlink_status(output_directory, error));
      if (error || !output_ready)
        detail = "scalar-IGR split-reader output directory is unavailable";
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
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    context.SetDiagnostic(detail);
    return result;
  }

  sc25::FoundationConfig foundation_config;
  foundation_config.polynomial_degree = config.polynomial_order();
  foundation_config.elements_per_direction = static_cast<int>(config.mesh_n());
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> owner;
  const bool initialized = Traits::Create(foundation_config, owner) ==
          sc25::FoundationStatus::ready && owner &&
      Traits::Initialize(*owner) == sc25::TrajectoryStatus::ready;
  if (!CollectiveAll(initialized, context.communicator()))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    context.SetDiagnostic("scalar-IGR CPU foundation initialization failed");
    return result;
  }
  const PersistentResourceDescriptor resources =
      ScalarIgrResources(config, *owner, context);
  if (!CollectiveAll(resources.Valid(), context.communicator()))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::invalid_resource_descriptor;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    return result;
  }
  result.status = RunStatus::ready;
  result.reason = FailureReason::none;
  result.failure_class = ApplicationRunResult::FailureClass::none;

  if (config.run_role() == ApplicationRunRoleV1::split_reader &&
      !RestoreScalarIgrCheckpoint<Rank2>(config, resources, context, *owner,
                                        result.retries, detail))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::restart_incompatible;
    result.failure_class = ApplicationRunResult::FailureClass::checkpoint_data;
    context.SetDiagnostic(detail);
    return result;
  }

  typename Traits::State published_state;
  ScalarIgrPackedState published;
  bool state_ready = Traits::CopyState(*owner, published_state) &&
      PackScalarIgrState(published_state, Traits::Coordinates(*owner),
                         published);
  int initial_pack_fault_source = -1;
  int initial_pack_fault_count = 0;
  if constexpr (Rank2)
  {
    static bool initial_pack_fault_consumed = false;
    const bool local_initial_pack_fault = ConsumeScalarRankStepFault(
        "DG_IGR_HDG_TEST_SCALAR_INITIAL_PACK_FAILURE",
        Traits::Coordinates(*owner).step, context.rank(),
        initial_pack_fault_consumed, initial_pack_fault_source);
    const int local_count = local_initial_pack_fault ? 1 : 0;
    const int local_source = local_initial_pack_fault ? context.rank() + 1 : 0;
    int collective_source = 0;
    const bool reduced = MPI_Allreduce(
        &local_count, &initial_pack_fault_count, 1, MPI_INT, MPI_SUM,
        context.communicator()) == MPI_SUCCESS &&
        MPI_Allreduce(&local_source, &collective_source, 1, MPI_INT, MPI_SUM,
                      context.communicator()) == MPI_SUCCESS;
    state_ready = state_ready && reduced && initial_pack_fault_count <= 1 &&
        !local_initial_pack_fault;
    if (initial_pack_fault_count == 1)
      initial_pack_fault_source = collective_source - 1;
  }
  // No rank may enter the payload-hash collectives until every rank has
  // completed the local copy/pack phase.
  if (!CollectiveAll(state_ready, context.communicator()))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = initial_pack_fault_count == 1
        ? ApplicationRunResult::FailureClass::injected
        : ApplicationRunResult::FailureClass::backend;
    detail = initial_pack_fault_count == 1
        ? "scalar-IGR initial pack failure source_rank=" +
              std::to_string(initial_pack_fault_source)
        : "scalar-IGR initial state pack failed";
    bool directory_reverted = true;
    if (context.rank() == 0 && output_directory_created)
      directory_reverted = DiscardDurableDirectoryIfEmpty(
                               output_directory, detail) ==
          DurableIoStatus::ok;
    if (!CollectiveAll(directory_reverted, context.communicator()))
      result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    context.SetDiagnostic(detail);
    return result;
  }
  state_ready = GlobalScalarIgrCoordinatesIdentity(published.coordinates,
                                                   context);
  if (!CollectiveAll(state_ready, context.communicator()))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    detail = "scalar-IGR initial coordinate consensus failed";
    context.SetDiagnostic(detail);
    return result;
  }
  StateHash initial_hash{};
  state_ready = GlobalScalarIgrStateIdentity(
      published, context, initial_hash);
  if (!CollectiveAll(state_ready, context.communicator()))
  {
    result.status = RunStatus::terminal_failure;
    result.reason = FailureReason::backend_failure;
    result.failure_class = ApplicationRunResult::FailureClass::backend;
    return result;
  }

  if (published.coordinates.step == 0 && OutputStep(config, 0))
  {
    DurableGenerationTransactionV1 initial_transaction;
    std::vector<std::uint8_t> vtk;
    const bool staged = BeginRootTransaction(
        initial_transaction, output_directory, 0, context, detail) &&
        StageScalarIgrVtk(config, published, context, initial_transaction,
                          vtk, detail);
    if (!staged || !PublishRootTransaction(initial_transaction, context,
                                           detail))
    {
      (void)RevertTransaction(initial_transaction, context, detail);
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = ApplicationRunResult::FailureClass::filesystem;
    }
  }

  const std::optional<ValidatedApplicationConfig> checkpoint_writer =
      CheckpointWriterConfig(request, config);
  if (result.failure_class == ApplicationRunResult::FailureClass::none)
  {
    TracedScalarIgrCpuSerialMethod<Rank2> method(
        *owner, config, context, injection_audit, detail);
    TracedScalarIgrCpuSerialPublication<Rank2> publication(
        config, resources, context, output_directory, checkpoint_writer,
        initial_hash, result, injection_audit, detail);
    cpu::w8c::LifecycleControl control;
    control.split_writer =
        config.run_role() == ApplicationRunRoleV1::split_writer;
    control.maximum_retries = config.retry_limit();
    control.initial_retries = result.retries;
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
    if (report.failure != cpu::w8c::LifecycleFailure::none &&
        detail.empty())
      detail = report.failure == cpu::w8c::LifecycleFailure::staging ||
              report.failure == cpu::w8c::LifecycleFailure::publication
          ? "scalar-IGR durable publication failed"
          : "scalar-IGR transaction preparation failed";
  }
  result.coordinates = Traits::Coordinates(*owner);
  if (result.failure_class == ApplicationRunResult::FailureClass::none)
  {
    result.status = RunStatus::committed;
    result.reason = FailureReason::none;
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
    transaction.candidate = {
        transaction.staging.target_coordinates,
        transaction.staging.interior_hash,
        transaction.staging.trace_hash,
        transaction.staging.state_hash,
        transaction.staging.interior,
        transaction.staging.trace};
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

#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
template <typename Case>
class W8cScalarMethod final
{
public:
  using Transaction =
      sc25::PersistentSc25ScalarIgrCpuFoundation::Rank1Transaction;

  bool Initialize() noexcept
  {
    sc25::FoundationConfig foundation;
    foundation.polynomial_degree = Case::polynomial_order;
    foundation.elements_per_direction = static_cast<int>(Case::mesh_n);
    if (ScalarIgrCpuTraits<false>::Create(foundation, owner_) !=
            sc25::FoundationStatus::ready || !owner_)
      return false;
    if constexpr (Case::kind == cpu::w8c::CaseKind::smooth_transition_v2)
      cpu::sc25::w8c_internal::RequestSmoothTransitionInitialization(
          *owner_, static_cast<int>(Case::accepted_steps), Case::dt);
    else
      cpu::sc25::w8c_internal::RequestW8cInitialization(*owner_);
    return ScalarIgrCpuTraits<false>::Initialize(*owner_) ==
        sc25::TrajectoryStatus::ready;
  }

  bool PrimeTransaction(Transaction& transaction) noexcept
  {
    return owner_ && ScalarIgrCpuTraits<false>::Prime(*owner_, transaction);
  }

  bool Prepare(Transaction& transaction, const bool retry,
               const std::uint64_t) noexcept
  {
    return owner_ && retry == RetryAvailable() && ScalarIgrAccepted(
        ScalarIgrCpuTraits<false>::Prepare(
            *owner_, transaction, sc25::TrajectoryFailureInjection::none));
  }
  bool Install(Transaction& transaction) noexcept
  {
    return owner_ && ScalarIgrCpuTraits<false>::Install(*owner_, transaction);
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
    const bool installed = owner_ &&
        ScalarIgrCpuTraits<false>::Install(*owner_, transaction);
    retry_available_ = installed && ScalarIgrCpuTraits<false>::Revert(
        *owner_, std::move(transaction));
    const bool retryable = durable_reverted && retry_available_ &&
        failure == cpu::w8c::LifecycleFailure::staging;
    return {durable_reverted && retry_available_, retryable, false};
  }
  cpu::w8c::LifecycleAttemptOutcome RevertInstalled(
      Transaction&& transaction, const cpu::w8c::LifecycleFailure failure,
      const bool durable_reverted) noexcept
  {
    retry_available_ = owner_ && ScalarIgrCpuTraits<false>::Revert(
        *owner_, std::move(transaction));
    const bool retryable = durable_reverted && retry_available_ &&
        failure == cpu::w8c::LifecycleFailure::injected_after_recovery;
    return {durable_reverted && retry_available_, retryable, false};
  }
  void Finalize(Transaction&& transaction) noexcept
  {
    retry_available_ = false;
    // Successful Install leaves the exact valid scalar provisional token.
    // Its run-contract finalizer cannot reject; publication has already
    // released rollback authority when this non-fallible boundary is entered.
    if (!owner_ || !ScalarIgrCpuTraits<false>::Finalize(
                       *owner_, std::move(transaction)))
      std::abort();
  }
  RunStatus Status() const noexcept
  {
    if (retry_available_) return RunStatus::retry_available;
    return owner_ ? owner_->trajectory_evidence().contract_status
                  : RunStatus::invalid_request;
  }
  FailureReason Reason() const noexcept
  {
    return owner_ ? owner_->trajectory_evidence().contract_last_failure
                  : FailureReason::backend_failure;
  }
  bool RetryAvailable() const noexcept
  {
    return retry_available_ ||
        (owner_ && owner_->trajectory_evidence().contract_status ==
                       RunStatus::retry_available);
  }
  constexpr bool ForceTerminalFailure() const noexcept { return true; }
  void OnRetry(cpu::w8c::LifecycleFailure, std::uint32_t) const noexcept {}
  StateCoordinates Coordinates() const noexcept
  {
    return owner_ ? ScalarIgrCpuTraits<false>::Coordinates(*owner_)
                  : StateCoordinates{};
  }
  bool PublishedState(ScalarIgrPackedState& state) const noexcept
  {
    ScalarIgrCpuTraits<false>::State current;
    return owner_ && ScalarIgrCpuTraits<false>::CopyState(*owner_, current) &&
        PackScalarIgrState(current, Coordinates(), state);
  }
  bool CandidateState(const Transaction& transaction,
                      ScalarIgrPackedState& state) const noexcept
  {
    return PackScalarIgrState(
        transaction.candidate_state(),
        transaction.future_checkpoint().descriptor.coordinates, state);
  }
  const sc25::PersistentTrajectoryCheckpoint& FutureCheckpoint(
      const Transaction& transaction) const noexcept
  {
    return transaction.future_checkpoint();
  }
  bool Restore(const ValidatedApplicationConfig& config,
               const PersistentResourceDescriptor& resources,
               const MpiApplicationContextV1& context,
               std::string& detail) noexcept
  {
    std::uint32_t prior_retries = 0;
    return owner_ && RestoreScalarIgrCheckpoint<false>(
        config, resources, context, *owner_, prior_retries, detail);
  }
  const sc25::PersistentSc25ScalarIgrCpuFoundation& Owner() const noexcept
  {
    return *owner_;
  }

private:
  std::unique_ptr<sc25::PersistentSc25ScalarIgrCpuFoundation> owner_;
  bool retry_available_ = false;
};

template <typename Case>
class W8cScalarPublication final
{
public:
  W8cScalarPublication(
      const ApplicationOptionsView& request,
      const ValidatedApplicationConfig& config,
      MpiApplicationContextV1& context, std::string& detail) noexcept
      : config_(config), context_(context), detail_(detail),
        output_directory_(std::string{
            config.output_directory().identity().view()}),
        checkpoint_writer_(CheckpointWriterConfig(request, config))
  {}

  bool Restore(W8cScalarMethod<Case>& method) noexcept
  {
    if (!EnsureDirectory()) return false;
    resources_ = ScalarIgrResources(config_, method.Owner(), context_);
    if (!resources_.Valid() ||
        !method.Restore(config_, resources_, context_, detail_) ||
        !method.PublishedState(published_))
      return false;
    return GlobalScalarIgrStateIdentity(published_, context_, initial_hash_);
  }

  bool PublishInitial(W8cScalarMethod<Case>& method) noexcept
  {
    if (!EnsureDirectory()) return false;
    resources_ = ScalarIgrResources(config_, method.Owner(), context_);
    if (!resources_.Valid() || !method.PublishedState(published_) ||
        !GlobalScalarIgrCoordinatesIdentity(published_.coordinates, context_) ||
        !GlobalScalarIgrStateIdentity(published_, context_, initial_hash_))
      return false;
    DurableGenerationTransactionV1 initial;
    std::vector<std::uint8_t> vtk;
    const bool staged = BeginRootTransaction(
        initial, output_directory_, 0, context_, detail_) &&
        StageScalarIgrVtk(config_, published_, context_, initial, vtk,
                          detail_);
    const bool published = staged &&
        PublishRootTransaction(initial, context_, detail_);
    if (!published) (void)RevertTransaction(initial, context_, detail_);
    return published;
  }

  bool Stage(W8cScalarMethod<Case>& method,
             const typename W8cScalarMethod<Case>::Transaction&
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
      bool staged = method.CandidateState(method_transaction, candidate_) &&
          GlobalScalarIgrCoordinatesIdentity(candidate_.coordinates, context_) &&
          BeginRootTransaction(*transaction_, output_directory_, target,
                               context_, detail_);
      if (staged && output)
        staged = StageScalarIgrVtk(config_, candidate_, context_, *transaction_,
                                  vtk_, detail_);
      if (staged && checkpoint)
        staged = checkpoint_writer_ && StageScalarIgrCheckpoint(
            *checkpoint_writer_, method.FutureCheckpoint(method_transaction),
            candidate_, resources_, context_, *transaction_, checkpoint_,
            detail_);
      if (staged && terminal)
      {
        StateHash final_hash{};
        staged = GlobalScalarIgrStateIdentity(candidate_, context_, final_hash);
        bool root_staged = staged;
        if (staged && context_.rank() == 0)
        {
          ApplicationRunResult anticipated;
          anticipated.status = RunStatus::committed;
          anticipated.reason = FailureReason::none;
          anticipated.coordinates = candidate_.coordinates;
          anticipated.retries = ExpectedRetries();
          anticipated.failure_class = ApplicationRunResult::FailureClass::none;
          ScalarIgrInjectionAudit audit;
          if (anticipated.retries == 1)
          {
            audit.configured_source_rank = 0;
            audit.observed_origin_rank = 0;
            audit.local_injection_count = 1;
            audit.collective_decisions = 1;
          }
          const std::string metadata = ScalarIgrMetadata(
              config_, anticipated, initial_hash_, final_hash, audit, detail_);
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
      detail_ = "W8C scalar-IGR durable staging failed";
      return false;
    }
  }

  bool ReadyToPublish(
      W8cScalarMethod<Case>&,
      const typename W8cScalarMethod<Case>::Transaction&,
      const std::uint32_t) const noexcept
  {
    return true;
  }

  bool Publish(
      W8cScalarMethod<Case>&,
      const typename W8cScalarMethod<Case>::Transaction&) noexcept
  {
    if (!active_) return true;
    return PublishRootTransaction(*transaction_, context_, detail_);
  }
  bool Acknowledge(
      W8cScalarMethod<Case>&,
      const typename W8cScalarMethod<Case>::Transaction&,
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
  PersistentResourceDescriptor resources_{};
  std::optional<DurableGenerationTransactionV1> transaction_;
  ScalarIgrPackedState published_;
  ScalarIgrPackedState candidate_;
  std::vector<std::uint8_t> vtk_;
  std::vector<std::uint8_t> checkpoint_;
  std::vector<std::uint8_t> metadata_;
  StateHash initial_hash_{};
  bool directory_ready_ = false;
  bool output_directory_created_ = false;
  bool active_ = false;
};
#endif

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
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
  if (config.scalar_igr())
  {
    W8cScalarMethod<Case> method;
    W8cScalarPublication<Case> publication(request, config, context, detail);
    report = cpu::w8c::RunCpuSerialLifecycle(
        method, publication, control, Case{});
    clean = publication.Cleanup();
  }
  else
#endif
  {
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
  }
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
    transaction.candidate = {
        transaction.staging.target_coordinates,
        transaction.staging.interior_hash,
        transaction.staging.trace_hash,
        transaction.staging.state_hash,
        transaction.staging.interior,
        transaction.staging.trace};
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
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
  if (request.card_id == scalar_igr_sc25_v3_card_id)
    return RunScalarIgrCpu<false>(request, context);
#endif
  return RunCpu<AppCpuSerialSelection>(request, context);
}

ApplicationRunResult RunSmoothPeriodicEulerV1Downstream(
    CpuRank2BackendTag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
  if (request.card_id == scalar_igr_sc25_v3_card_id)
    return RunScalarIgrCpu<true>(request, context);
#endif
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

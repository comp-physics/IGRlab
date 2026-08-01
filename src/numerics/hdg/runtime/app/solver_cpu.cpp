#include "numerics/hdg/runtime/app/solver.hpp"

#include "numerics/hdg/runtime/app/checkpoint_codec.hpp"
#include "numerics/hdg/runtime/app/command_line.hpp"
#include "numerics/hdg/runtime/app/durable_io.hpp"
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
    candidate.interior.reserve(sc25::persistent_interior_unknowns);
    candidate.trace.reserve(sc25::persistent_true_trace_unknowns);
    AppendScalarRange(state.u, candidate.interior);
    AppendScalarRange(state.sigma, candidate.interior);
    AppendScalarRange(state.z, candidate.interior);
    AppendScalarRange(state.uhat, candidate.trace);
    AppendScalarRange(state.sigmahat, candidate.trace);
    if (candidate.interior.size() != sc25::persistent_interior_unknowns ||
        candidate.trace.size() != sc25::persistent_true_trace_unknowns)
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
    if (elements != sc25::persistent_elements ||
        (ranks != 1 && ranks != 2))
    {
      detail = "scalar-IGR VTK topology is outside the frozen N8 tuple";
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
    std::array<bool, sc25::persistent_elements> global_seen{};
    std::array<std::array<bool, sc25::persistent_elements>, 2> local_seen{};
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
          local_element >= sc25::persistent_elements ||
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
                     [](const bool seen) { return seen; }))
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
  writer.injection = config.scalar_igr()
      ? config.injection() : FailureInjectionControlV1{};
  writer.retry_limit = config.scalar_igr()
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
  if (!config.scalar_igr() || context.ranks() != (Rank2 ? 2 : 1))
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
  const std::uint64_t stop_step =
      config.run_role() == ApplicationRunRoleV1::split_writer
      ? config.scenario_checkpoint_step() : config.accepted_steps();
  bool injected = false;
  while (result.failure_class == ApplicationRunResult::FailureClass::none &&
         Traits::Coordinates(*owner).step < stop_step)
  {
    const std::uint64_t target = Traits::Coordinates(*owner).step + 1;
    const bool prescribed = !injected &&
        config.injection().mode == FailureInjectionModeV1::prescribed &&
        config.injection().target_step == target;
    const int prescribed_source_rank =
        static_cast<int>(config.injection().source_rank);
    const bool local_prescribed = prescribed &&
        context.rank() == prescribed_source_rank;
    const FailurePhase phase = prescribed ? config.injection().phase
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
          local_prescribed ? context.rank() + 1 : 0,
          local_prescribed &&
                  context.rank() != prescribed_source_rank ? 1 : 0}};
      std::array<int, 3> collective_decision{};
      const bool reduced = MPI_Allreduce(
          local_decision.data(), collective_decision.data(),
          static_cast<int>(local_decision.size()), MPI_INT, MPI_SUM,
          context.communicator()) == MPI_SUCCESS;
      const bool decision_valid = reduced && collective_decision[0] == 1 &&
          collective_decision[1] == prescribed_source_rank + 1 &&
          collective_decision[2] == 0;
      if (!CollectiveAll(decision_valid, context.communicator()))
      {
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = ApplicationRunResult::FailureClass::backend;
        detail = "scalar-IGR injection origin consensus failed";
        break;
      }
      injection_audit.observed_origin_rank = collective_decision[1] - 1;
      injection_audit.local_injection_count += collective_decision[0];
      injection_audit.non_origin_local_injection_count +=
          collective_decision[2];
      ++injection_audit.collective_decisions;
    }
    typename Traits::Transaction candidate;
    const sc25::TrajectoryStatus status =
        Traits::Prepare(*owner, candidate, numerical);
    if (!ScalarIgrAccepted(status))
    {
      injected = injected || prescribed;
      if (prescribed && result.retries < config.retry_limit())
      {
        ++result.retries;
        continue;
      }
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = prescribed
          ? ApplicationRunResult::FailureClass::injected
          : ApplicationRunResult::FailureClass::scientific;
      detail = "scalar-IGR numerical prepare failed";
      break;
    }

    ScalarIgrPackedState packed;
    bool packed_ready = PackScalarIgrState(
        candidate.candidate_state(),
        candidate.future_checkpoint().descriptor.coordinates, packed);
    bool private_preparation_injected = false;
    int candidate_pack_fault_source = -1;
    if constexpr (Rank2)
    {
      static bool candidate_pack_fault_consumed = false;
      const bool local_candidate_pack_fault = ConsumeScalarRankStepFault(
          "DG_IGR_HDG_TEST_SCALAR_CANDIDATE_PACK_FAILURE", target,
          context.rank(), candidate_pack_fault_consumed,
          candidate_pack_fault_source);
      const int local_count = local_candidate_pack_fault ? 1 : 0;
      const int local_source = local_candidate_pack_fault
          ? context.rank() + 1 : 0;
      int collective_count = 0, collective_source = 0;
      const bool reduced = MPI_Allreduce(
          &local_count, &collective_count, 1, MPI_INT, MPI_SUM,
          context.communicator()) == MPI_SUCCESS &&
          MPI_Allreduce(&local_source, &collective_source, 1, MPI_INT,
                        MPI_SUM, context.communicator()) == MPI_SUCCESS;
      packed_ready = packed_ready && reduced && collective_count <= 1 &&
          !local_candidate_pack_fault;
      private_preparation_injected = reduced && collective_count == 1;
      if (private_preparation_injected)
        candidate_pack_fault_source = collective_source - 1;
    }
    // BeginRootTransaction is collective.  Converge every local pack result
    // before any rank is allowed to enter it.
    packed_ready = CollectiveAll(packed_ready, context.communicator());
    if (packed_ready)
      packed_ready = CollectiveAll(
          GlobalScalarIgrCoordinatesIdentity(packed.coordinates, context),
          context.communicator());
    if (!packed_ready)
      detail = private_preparation_injected
          ? "scalar-IGR candidate pack failure source_rank=" +
                std::to_string(candidate_pack_fault_source)
          : "scalar-IGR candidate state pack failed";
    bool staged = packed_ready;
    DurableGenerationTransactionV1 transaction;
    std::vector<std::uint8_t> vtk, checkpoint, metadata;
    if (staged)
      staged = BeginRootTransaction(
          transaction, output_directory, target, context, detail);
    if (staged && OutputStep(config, target))
      staged = StageScalarIgrVtk(config, packed, context, transaction, vtk,
                                 detail);
    if (staged && config.checkpoint_out().present() &&
        target == config.scenario_checkpoint_step())
    {
      staged = checkpoint_writer && StageScalarIgrCheckpoint(
          *checkpoint_writer, candidate.future_checkpoint(), packed,
          resources, context, transaction, checkpoint, detail);
    }
    if (staged && target == stop_step)
    {
      StateHash final_hash{};
      bool metadata_ready =
          GlobalScalarIgrStateIdentity(packed, context, final_hash);
      int metadata_fault_source = -1;
      int metadata_fault_count = 0;
      if constexpr (Rank2)
      {
        static bool metadata_fault_consumed = false;
        const bool local_metadata_fault = ConsumeScalarRankStepFault(
            "DG_IGR_HDG_TEST_SCALAR_METADATA_PREPARATION_FAILURE", target,
            context.rank(), metadata_fault_consumed, metadata_fault_source);
        const int local_count = local_metadata_fault ? 1 : 0;
        const int local_source = local_metadata_fault ? context.rank() + 1 : 0;
        int collective_source = 0;
        const bool reduced = MPI_Allreduce(
            &local_count, &metadata_fault_count, 1, MPI_INT, MPI_SUM,
            context.communicator()) == MPI_SUCCESS &&
            MPI_Allreduce(&local_source, &collective_source, 1, MPI_INT,
                          MPI_SUM, context.communicator()) == MPI_SUCCESS;
        metadata_ready = metadata_ready && reduced &&
            metadata_fault_count <= 1 && !local_metadata_fault;
        if (metadata_fault_count == 1)
          metadata_fault_source = collective_source - 1;
        private_preparation_injected = private_preparation_injected ||
            (reduced && metadata_fault_count == 1);
      }
      if (context.rank() == 0 && metadata_ready)
      {
        try
        {
          ApplicationRunResult anticipated = result;
          anticipated.status = RunStatus::committed;
          anticipated.reason = FailureReason::none;
          anticipated.coordinates = packed.coordinates;
          const std::string text = ScalarIgrMetadata(
              config, anticipated, initial_hash, final_hash, injection_audit,
              detail);
          metadata.assign(text.begin(), text.end());
          metadata_ready = transaction.Stage(
              config.run_role() == ApplicationRunRoleV1::split_writer
                  ? "run-metadata-checkpoint.json" : "run-metadata.json",
              metadata.data(), metadata.size(), detail) == DurableIoStatus::ok;
        }
        catch (...)
        {
          metadata_ready = false;
          detail = "scalar-IGR root metadata preparation failed";
        }
      }
      if (metadata_fault_count == 1)
        detail = "scalar-IGR metadata preparation failure source_rank=" +
            std::to_string(metadata_fault_source);
      staged = CollectiveAll(metadata_ready, context.communicator());
    }

    const bool app_injected = prescribed &&
        prescribed_numerical == sc25::TrajectoryFailureInjection::none;
    bool transaction_injected =
        app_injected || private_preparation_injected;
    bool installed = false;
    bool committed = false;
    bool publish_failed = false;
    if (staged)
    {
      installed = Traits::Install(*owner, candidate);
      const bool collectively_installed =
          CollectiveAll(installed, context.communicator());
      if (collectively_installed && !app_injected)
      {
        bool local_finalize_ready = Traits::FinalizeReady(candidate);
        int finalize_fault_source = -1;
        static bool finalize_fault_consumed = false;
        if constexpr (Rank2)
        {
          const bool finalize_fault = ConsumeScalarRankStepFault(
              "DG_IGR_HDG_TEST_SCALAR_FINALIZE_PREPARATION_FAILURE",
              target, context.rank(), finalize_fault_consumed,
              finalize_fault_source);
          local_finalize_ready = local_finalize_ready && !finalize_fault;
          transaction_injected = finalize_fault;
        }
        const bool collectively_finalize_ready =
            CollectiveAll(local_finalize_ready, context.communicator());
        int injected_finalize_count = transaction_injected ? 1 : 0;
        int collective_finalize_count = 0;
        const bool finalize_fault_reduced = MPI_Allreduce(
            &injected_finalize_count, &collective_finalize_count, 1, MPI_INT,
            MPI_SUM, context.communicator()) == MPI_SUCCESS;
        transaction_injected = finalize_fault_reduced &&
            collective_finalize_count == 1;
        if (!collectively_finalize_ready)
        {
          detail = transaction_injected && finalize_fault_source >= 0
              ? "scalar-IGR finalize preparation failure source_rank=" +
                    std::to_string(finalize_fault_source)
              : "scalar-IGR finalize readiness consensus failed";
        }
        else
        {
          const bool published_ok = PublishRootTransaction(
              transaction, context, detail);
          publish_failed = !published_ok;
          if (published_ok)
          {
            // Install plus the rank-common readiness gate establish every
            // documented precondition of this noexcept token consumption.
            const bool finalized =
                Traits::Finalize(*owner, std::move(candidate));
            if (!CollectiveAll(finalized, context.communicator()))
            {
              result.status = RunStatus::terminal_failure;
              result.reason = FailureReason::backend_failure;
              result.failure_class =
                  ApplicationRunResult::FailureClass::backend;
              detail = "scalar-IGR finalize invariant failed";
              break;
            }
            committed = true;
          }
        }
      }
    }
    if (!committed)
    {
      (void)RevertTransaction(transaction, context, detail);
      if (!installed) installed = Traits::Install(*owner, candidate);
      const bool reverted = installed &&
          Traits::Revert(*owner, std::move(candidate));
      if (!CollectiveAll(reverted, context.communicator()))
      {
        result.status = RunStatus::terminal_failure;
        result.reason = FailureReason::backend_failure;
        result.failure_class = ApplicationRunResult::FailureClass::backend;
        detail = "scalar-IGR transaction rollback failed";
        break;
      }
      injected = injected || prescribed;
      if (transaction_injected && result.retries < config.retry_limit())
      {
        ++result.retries;
        continue;
      }
      result.status = RunStatus::terminal_failure;
      result.reason = FailureReason::backend_failure;
      result.failure_class = transaction_injected
          ? ApplicationRunResult::FailureClass::injected
          : (!staged || publish_failed
                 ? ApplicationRunResult::FailureClass::filesystem
                 : ApplicationRunResult::FailureClass::backend);
      if (detail.empty()) detail = !staged || publish_failed
          ? "scalar-IGR durable publication failed"
          : "scalar-IGR transaction preparation failed";
      break;
    }
    injected = injected || prescribed;
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

  const std::uint64_t stop_step =
      config.run_role() == ApplicationRunRoleV1::split_writer
      ? config.scenario_checkpoint_step() : config.accepted_steps();
  bool retrying = false;
  while (result.failure_class == ApplicationRunResult::FailureClass::none &&
         adapter.coordinates().step < stop_step)
  {
    bool private_post_publication_retry = false;
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
      DurableGenerationTransactionV1 transaction;
      std::vector<std::uint8_t> vtk_bytes;
      std::vector<std::uint8_t> checkpoint_bytes;
      std::vector<std::uint8_t> metadata_bytes;
      bool staged = BeginRootTransaction(
          transaction, output_directory, target, context, detail);
      CpuFailurePhase staging_failure_phase = CpuFailurePhase::output_staging;
      const bool injected_outer_stage =
          injection.phase == CpuFailurePhase::checkpoint_staging ||
          injection.phase == CpuFailurePhase::output_staging;
      const CpuStateSnapshot candidate{
          staging.target_coordinates, staging.interior_hash,
          staging.trace_hash, staging.state_hash, staging.interior,
          staging.trace};
      if (!injected_outer_stage && OutputStep(config, target))
        staged = StageStateVtk(config, candidate, context, transaction,
                               vtk_bytes, detail);
      if (!injected_outer_stage && staged &&
          config.checkpoint_out().present() &&
          target == config.scenario_checkpoint_step())
      {
        staging_failure_phase = CpuFailurePhase::checkpoint_staging;
        staged = checkpoint_writer && StageCheckpoint(
            *checkpoint_writer, staging.future_checkpoint, staging.interior,
            staging.trace, adapter.evidence(), context, transaction,
            checkpoint_bytes, detail);
      }
      if (!staged)
      {
        const bool cleaned = RevertTransaction(transaction, context, detail);
        outcome = adapter.AbortPrepared(
            staging, cleaned ? staging_failure_phase
                             : CpuFailurePhase::rollback_audit, -1);
      }
      else if (injection.phase == CpuFailurePhase::checkpoint_staging ||
               injection.phase == CpuFailurePhase::output_staging)
      {
        const bool cleaned = RevertTransaction(transaction, context, detail);
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
          const bool cleaned = RevertTransaction(transaction, context, detail);
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
            StateHash terminal_identity{};
            bool metadata_staged = GlobalCpuStateIdentity(
                candidate, context, terminal_identity);
            if (context.rank() == 0)
            {
              try
              {
                const std::string metadata = Metadata(
                    config, anticipated, adapter.evidence(), initial_hash,
                    terminal_identity, detail);
                metadata_bytes.assign(metadata.begin(), metadata.end());
                metadata_staged = transaction.Stage(
                    config.run_role() == ApplicationRunRoleV1::split_writer
                        ? "run-metadata-checkpoint.json"
                        : "run-metadata.json",
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
            const bool cleaned = RevertTransaction(transaction, context,
                                                   detail);
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
              const bool cleaned = RevertTransaction(transaction, context,
                                                     detail);
              outcome = adapter.last_transition_outcome();
              if (!cleaned)
                outcome = {RunStatus::terminal_failure,
                           FailureReason::backend_failure, false, false};
            }
            else if (injection.phase == CpuFailurePhase::rollback_audit)
            {
              const bool cleaned = RevertTransaction(transaction, context,
                                                     detail);
              outcome = adapter.RevertCommit(
                  provisional,
                  cleaned ? injection.phase
                          : CpuFailurePhase::rollback_audit,
                  injection.source_rank);
            }
            else
            {
              const bool published = PublishRootTransaction(
                  transaction, context, detail);
              if (!published)
              {
                const bool cleaned =
                    RevertTransaction(transaction, context, detail);
                outcome = adapter.RevertCommit(
                    provisional,
                    cleaned ? CpuFailurePhase::output_staging
                            : CpuFailurePhase::rollback_audit, -1);
              }
              else
              {
                int marker_source_rank = -1;
                int finalize_source_rank = -1;
                const bool marker_acknowledged = CollectiveAll(
                    !ConsumePostPublicationFault(
                        PostPublicationFault::after_marker, target,
                        context.rank(), marker_source_rank),
                    context.communicator());
                const bool finalize_ready = marker_acknowledged &&
                    CollectiveAll(
                        !ConsumePostPublicationFault(
                            PostPublicationFault::before_finalize, target,
                            context.rank(), finalize_source_rank),
                        context.communicator());
                if (!finalize_ready)
                {
                  const bool cleaned =
                      RevertTransaction(transaction, context, detail);
                  outcome = adapter.RevertCommit(
                      provisional,
                      cleaned ? (marker_acknowledged
                                     ? CpuFailurePhase::commit
                                     : CpuFailurePhase::output_staging)
                              : CpuFailurePhase::rollback_audit,
                      marker_acknowledged ? finalize_source_rank
                                          : marker_source_rank);
                  private_post_publication_retry = cleaned &&
                      PostPublicationFaultAllowsRetry() &&
                      result.retries == 0;
                }
                else
                {
                  outcome = adapter.FinalizeCommit(provisional);
                  if (!outcome.committed)
                  {
                    const bool cleaned =
                        RevertTransaction(transaction, context, detail);
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
    }
    if (!outcome.committed && outcome.retry_available &&
        (result.retries < config.retry_limit() ||
         private_post_publication_retry))
    {
      AppendCpuFailureProvenance(adapter.evidence().last_attempt, detail);
      ++result.retries;
      retrying = true;
      continue;
    }
    if (!outcome.committed)
    {
      result.status = outcome.status;
      result.reason = outcome.reason;
      const CpuAttemptEvidence& failed = adapter.evidence().last_attempt;
      const bool prescribed_failure_fired =
          config.injection().mode == FailureInjectionModeV1::prescribed &&
          failed.status == CpuAttemptStatus::injected_failure &&
          failed.failure_phase == ToCpuPhase(config.injection().phase) &&
          failed.failure_source_rank ==
              static_cast<int>(config.injection().source_rank);
      const bool scientific_failure = ScientificFailureReason(outcome.reason) ||
          (failed.status == CpuAttemptStatus::success &&
           failed.failure_phase == CpuFailurePhase::none &&
           (!failed.finite_gate || !failed.admissibility_gate ||
            !failed.recovery_gate || !failed.nonlinear_gate ||
            !failed.true_residual_gate));
      result.failure_class = prescribed_failure_fired
          ? ApplicationRunResult::FailureClass::injected
          : (scientific_failure
                 ? ApplicationRunResult::FailureClass::scientific
                 : ApplicationRunResult::FailureClass::backend);
      if (detail.empty()) detail = "persistent advance did not commit";
      AppendCpuFailureProvenance(failed, detail);
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
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
  if (request.card_id == scalar_igr_sc25_v3_card_id)
    return RunScalarIgrCpu<false>(request, context);
#endif
  return RunCpu<AppCpuSerialSelection>(request, context);
}

ApplicationRunResult RunSmoothPeriodicEulerV1(
    CpuRank2BackendTag, const ApplicationOptionsView& request,
    MpiApplicationContextV1& context)
{
#if DG_IGR_HDG_APP_HAS_SCALAR_IGR_CPU
  if (request.card_id == scalar_igr_sc25_v3_card_id)
    return RunScalarIgrCpu<true>(request, context);
#endif
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

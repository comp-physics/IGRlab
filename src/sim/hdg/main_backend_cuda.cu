#include "sim/hdg/main_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "sim/context/case_context.hpp"
#include "sim/hdg/parallel_vtk.hpp"
#include "sim/hdg/resolved_euler_case.hpp"
#include "sim/hdg/state_io.hpp"
#include "sim/hdg/transactional_time_adapter.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/wiring/hdg_boundary_wiring.hpp"

namespace dg_sim::hdg
{
namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::BackendSelection;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::generic::RegularizationKind;
using dg::hdg::generic::RegularizationSelection;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::CudaMutableHostL2Output;
using dg::hdg::runtime::cuda_w8a::CudaPreparedOutputMetadataV1;

bool IsRoot() noexcept
{
  int rank = -1;
  return MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS && rank == 0;
}

bool CollectiveAll(const bool local) noexcept
{
  const int input = local ? 1 : 0;
  int output = 0;
  return MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && output == 1;
}

bool ExactDoubleConsensus(const double value) noexcept
{
  std::uint64_t bits = 0, minimum = 0, maximum = 0;
  static_assert(sizeof(bits) == sizeof(value), "CUDA CFL requires binary64");
  std::memcpy(&bits, &value, sizeof(bits));
  const int minimum_status = MPI_Allreduce(&bits, &minimum, 1,
      MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  const int maximum_status = MPI_Allreduce(&bits, &maximum, 1,
      MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  return minimum_status == MPI_SUCCESS && maximum_status == MPI_SUCCESS &&
      minimum == maximum;
}

int Reject(const std::string& reason) noexcept
{
  if (IsRoot())
    std::cerr << "[hdg-main] status=Rejected reason=" << reason << '\n';
  return 1;
}

bool WorldCongruent(const MPI_Comm communicator) noexcept
{
  int initialized = 0;
  int finalized = 0;
  int comparison = MPI_UNEQUAL;
  int ranks = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Initialized(&initialized) == MPI_SUCCESS && initialized &&
      MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized &&
      MPI_Comm_size(MPI_COMM_WORLD, &ranks) == MPI_SUCCESS &&
      (ranks == 1 || ranks == 2) &&
      MPI_Comm_compare(communicator, MPI_COMM_WORLD, &comparison) == MPI_SUCCESS &&
      (comparison == MPI_IDENT || comparison == MPI_CONGRUENT);
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

RegularizationSelection CanonicalRegularization(
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

std::string CanonicalNumericalPolicy(
    const dg::hdg::generic::EulerHdgNumericalPolicy& policy)
{
  std::ostringstream stream;
  stream << policy.volume_projection_quadrature_order << ';'
         << policy.true_trace_projection_quadrature_order << ';'
         << policy.conventional_assembly_quadrature_order << ';'
         << policy.igr_assembly_quadrature_order << ';'
         << policy.audit_quadrature_order << ';'
         << policy.observable_quadrature_order << ';'
         << policy.maximum_nonlinear_updates << ';'
         << policy.maximum_linear_iterations << ';'
         << policy.maximum_line_search_trials << ';'
         << static_cast<int>(policy.stabilization.kind) << ';';
  AppendDouble(stream, policy.nonlinear_relative_tolerance);
  AppendDouble(stream, policy.linear_relative_tolerance);
  AppendDouble(stream, policy.true_residual_relative_tolerance);
  AppendDouble(stream, policy.conservation_relative_tolerance);
  AppendDouble(stream, policy.armijo_constant);
  AppendDouble(stream, policy.stabilization.alpha);
  AppendDouble(stream, policy.stabilization.beta);
  return stream.str();
}

std::string ConsensusText(const HdgEulerMainRequest& request,
                          const std::string& mesh_path,
                          const RegularizationSelection regularization)
{
  std::ostringstream stream;
  const mfem::io::VTKOutputConfig vtk = MakeCaseVTKConfig(request.case_config);
  AppendText(stream, request.case_config.case_name);
  AppendText(stream, mesh_path);
  AppendText(stream, ToLowerCopy(request.options.device));
  AppendText(stream, request.options.restart_vtk);
  AppendText(stream, vtk.prefix_path);
  stream << request.options.order << ';'
         << request.options.ser_ref_levels << ';'
         << request.options.par_ref_levels << ';'
         << request.options.restart_cycle << ';'
         << request.options.restart_step << ';'
         << request.options.vtk_save << ';'
         << request.options.glvis << ';'
         << request.options.save_state_files << ';'
         << static_cast<int>(vtk.format) << ';'
         << vtk.high_order_output << ';'
         << vtk.levels_of_detail << ';';
  AppendDouble(stream, request.options.dt);
  AppendDouble(stream, request.options.dt_min);
  AppendDouble(stream, request.options.cfl);
  AppendDouble(stream, request.options.t_final);
  AppendDouble(stream, request.options.dt_save);
  AppendDouble(stream, request.case_config.gamma);
  stream << static_cast<int>(regularization.kind) << ';';
  AppendDouble(stream, regularization.alpha);
  AppendDouble(stream, regularization.beta);
  return stream.str();
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

bool ExactTextConsensus(const std::string& local) noexcept
{
  int rank = -1;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) return false;
  std::uint64_t root_size = rank == 0 ? local.size() : 0;
  const bool size_broadcast = MPI_Bcast(&root_size, 1, MPI_UINT64_T, 0,
                                        MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!CollectiveAll(size_broadcast &&
      root_size <= static_cast<std::uint64_t>(std::numeric_limits<int>::max())))
    return false;
  std::string root;
  bool allocated = true;
  try { root.resize(static_cast<std::size_t>(root_size)); }
  catch (...) { allocated = false; }
  if (!CollectiveAll(allocated)) return false;
  if (rank == 0 && root_size != 0)
    std::memcpy(root.data(), local.data(), static_cast<std::size_t>(root_size));
  char empty = 0;
  char* data = root.empty() ? &empty : root.data();
  const bool text_broadcast = MPI_Bcast(data, static_cast<int>(root_size),
                                        MPI_CHAR, 0, MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!CollectiveAll(text_broadcast)) return false;
  return CollectiveAll(local.size() == static_cast<std::size_t>(root_size) &&
                       std::equal(local.begin(), local.end(), root.begin()));
}

std::string Validate(const HdgEulerMainRequest& request,
                     const RegularizationSelection regularization)
{
  if (!WorldCongruent(request.communicator))
    return "CUDA HDG requires MPI_COMM_WORLD-congruent ranks one or two";
  if (!request.make_case_data)
    return "CUDA HDG case physics factory is empty";
  if (ToLowerCopy(request.options.device) != "cuda")
    return "generic CUDA HDG requires --device cuda";
  if (!dg::hdg::generic::ValidRegularizationSelection(regularization))
    return "HDG IGR alpha and beta must be finite and positive";
  if (!request.options.restart_vtk.empty() || request.options.restart_cycle >= 0 ||
      request.options.restart_step >= 0 || request.options.save_state_files)
    return "generic CUDA HDG restart/state-file routes are unsupported";
  if (request.options.glvis)
    return "generic CUDA HDG GLVis is unsupported";
  if (!std::isfinite(request.options.dt))
    return "CUDA HDG --time-step must be finite";
  if (request.options.dt <= 0.0 &&
      (!std::isfinite(request.options.cfl) || request.options.cfl <= 0.0))
    return "CUDA CFL Backward Euler requires a finite positive --cfl-number";
  if (!std::isfinite(request.options.dt_min) || request.options.dt_min < 0.0)
    return "CUDA HDG --dt-min must be finite and non-negative";
  if (!std::isfinite(request.options.t_final) || request.options.t_final <= 0.0)
    return "CUDA HDG final time must be finite and positive";
  if (request.options.vtk_save &&
      (!std::isfinite(request.options.dt_save) || request.options.dt_save <= 0.0))
    return "CUDA HDG ParaView output requires a positive --time-step-save";
  if (request.options.order < dg::hdg::generic::minimum_polynomial_order ||
      request.options.order > dg::hdg::generic::maximum_polynomial_order)
    return "generic CUDA HDG supports polynomial orders one through three";
  if (request.options.ser_ref_levels < 0 || request.options.par_ref_levels < 0)
    return "generic CUDA HDG refinement levels must be nonnegative";
  if (!std::isfinite(request.case_config.gamma) ||
      request.case_config.gamma != mfem::dg_igr::EulerDefaults::kGamma)
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

std::string CollectionName(const HdgEulerMainRequest& request)
{
  return request.case_config.case_name + "_hdg_cuda_p_" +
      std::to_string(request.options.order) + "_n_" +
      std::to_string(request.options.order == 1 ? 3 : 6);
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
                   CudaPreparedOutputMetadataV1& native_export) noexcept
{
  return session.Abort(std::move(authority), native_export).status ==
      HdgEulerSessionStatus::aborted;
}

bool FreshOutputView(mfem::ParGridFunction& published_state,
                     const std::size_t scalar_count,
                     CudaMutableHostL2Output& destination) noexcept
{
  destination = {};
  try
  {
    constexpr std::size_t components =
        dg::hdg::generic::euler_2d_conserved_components;
    if (scalar_count > std::numeric_limits<std::size_t>::max() / components ||
        published_state.Size() < 0 ||
        static_cast<std::size_t>(published_state.Size()) !=
            scalar_count * components)
    {
      return false;
    }
    if (scalar_count == 0)
    {
      destination = {nullptr, 0,
                     dg::hdg::generic::euler_2d_conserved_components};
      return true;
    }
    double* const coefficients = published_state.HostWrite();
    destination = {coefficients, scalar_count,
                   dg::hdg::generic::euler_2d_conserved_components};
    return coefficients != nullptr;
  }
  catch (...) { return false; }
}

} // namespace

int RunEulerMainBackend(const HdgEulerMainRequest& request)
{
  // This adapter is MPI_COMM_WORLD-bound.  Converge a rank-local bad request
  // communicator before any later WORLD collective so a divergent caller
  // cannot return on one rank while another advances into admission.
  if (!CollectiveAll(WorldCongruent(request.communicator)))
    return Reject("invalid MPI_COMM_WORLD-congruent CUDA HDG invocation");

  std::string reason;
  RegularizationSelection regularization{};
  try
  {
    regularization = CanonicalRegularization(request);
    reason = Validate(request, regularization);
  }
  catch (...) { reason = "rank-local CUDA HDG request validation failed"; }
  if (!CollectiveAll(reason.empty()))
    return Reject(reason.empty() ? "CUDA HDG request validation failed on another rank"
                                 : reason);

  std::string mesh_path;
  std::string consensus;
  try
  {
    mesh_path = ResolveCaseMesh(request.case_config, request.options.mesh_file);
    consensus = ConsensusText(request, mesh_path, regularization);
  }
  catch (...) { reason = "CUDA HDG configuration resolution failed"; }
  if (!CollectiveAll(!consensus.empty()))
    return Reject(reason.empty() ? "CUDA HDG configuration resolution failed on another rank"
                                 : reason);
  if (!ExactTextConsensus(consensus))
    return Reject("CUDA HDG configuration differs between MPI ranks");

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
  if (!CollectiveAll(local_ok))
    return Reject("CUDA HDG serial mesh loading failed on at least one rank");
  if (!ExactTextConsensus(serialized_mesh))
    return Reject("CUDA HDG serial mesh content differs between MPI ranks");

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
    refined_mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD,
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
  if (!CollectiveAll(local_ok))
    return Reject(reason.empty() ? "CUDA HDG mesh construction failed on another rank"
                                 : reason);

  // No rank can enter this census until every rank has completed local mesh
  // construction, refinement, and expected-count arithmetic.  The count is
  // therefore both collective-order safe and an observable refinement gate.
  long long global_refined_elements = 0;
  const bool census_complete = MPI_Allreduce(
      &local_refined_elements, &global_refined_elements, 1,
      MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool census_matches = census_complete &&
      global_refined_elements == expected_global_elements;
  if (!CollectiveAll(census_matches))
    return Reject("CUDA HDG parallel refinement census mismatch");

  std::optional<CaseContext> context;
  try
  {
    context.emplace(std::move(refined_mesh), std::move(refined_metadata));
    local_ok = true;
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(context.has_value() && local_ok))
    return Reject("CUDA HDG mesh context publication failed on another rank");

  std::optional<HdgEulerCaseData> case_data;
  std::optional<FrozenBoundaryPlan> frozen_boundary;
  dg::hdg::generic::EulerHdgNumericalPolicy numerical_policy{};
  std::string case_data_consensus;
  try
  {
    case_data.emplace(request.make_case_data(*context));
    numerical_policy = case_data->numerical_policy;
    // Case factories own every numerical role except the command-line
    // regularization toggle and its two IGR coefficients.
    numerical_policy.stabilization = regularization;
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
  if (!CollectiveAll(local_ok && case_data.has_value()))
    return Reject(reason.empty()
        ? "CUDA HDG case-data factory or validation failed on another rank"
        : reason);
  if (!ExactTextConsensus(case_data_consensus))
    return Reject("CUDA HDG case data differs between MPI ranks");

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
  if (!CollectiveAll(local_ok))
    return Reject("CUDA HDG volume initial-state projection failed");

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
  if (!CollectiveAll(local_ok))
    return Reject("CUDA HDG state projection or boundary lowering failed");

  const std::size_t scalar_count = static_cast<std::size_t>(
      volume_space->GetVSize() /
      dg::hdg::generic::euler_2d_conserved_components);
  const BackendSelection backend{BackendKind::cuda, 0};
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
      boundary->View(), numerical_policy};

  GenericEulerHdgCudaSession session;
  if (session.Initialize(resolved_plan.request).status !=
      HdgEulerSessionStatus::initialized)
    return Reject("generic CUDA session initialization failed");
  CudaMutableHostL2Output destination{};
  if (!CollectiveAll(FreshOutputView(*published_state, scalar_count,
                                     destination)))
    return Reject("initial CUDA committed-state output view failed");
  local_ok = session.ExportCommittedInterior(destination);
  if (!CollectiveAll(local_ok))
    return Reject("initial CUDA committed-state export failed");

  std::unique_ptr<HdgTransactionalTimeAdapter> time_driver;
  std::unique_ptr<HDGParaViewSeries> output;
  try
  {
    const bool cfl_control = request.options.dt <= 0.0;
    resolved_plan.time_control = cfl_control
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
    resolved_plan.time_mode = cfl_control
        ? HdgEulerTimeMode::CFLBackwardEuler
        : HdgEulerTimeMode::FixedBackwardEuler;
    const mfem::io::VTKOutputConfig vtk = MakeCaseVTKConfig(request.case_config);
    resolved_plan.output = {request.options.vtk_save, vtk.prefix_path,
                            CollectionName(request)};
    time_driver = std::make_unique<HdgTransactionalTimeAdapter>(
        resolved_plan.time_control, resolved_plan.output_schedule,
        cfl_control ? HdgTimeControlMode::CFLBackwardEuler
                    : HdgTimeControlMode::FixedBackwardEuler);
    if (request.options.vtk_save)
    {
      const int levels = vtk.levels_of_detail >= 0 ? vtk.levels_of_detail
                                                     : request.options.order + 1;
      output = std::make_unique<HDGParaViewSeries>(HDGParaViewRequest{
          context->Mesh(), {{"conservative", *published_state}},
          {resolved_plan.output.prefix_path,
           resolved_plan.output.collection_name, vtk.format, levels,
           vtk.high_order_output}});
    }
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(time_driver != nullptr &&
      (!request.options.vtk_save || output != nullptr)))
    return Reject("CUDA HDG time/output setup failed");
  try
  {
    if (output)
      if (const auto initial = time_driver->InitialOutput())
        output->Write(Metadata(*initial, 0));
    local_ok = true;
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(local_ok))
    return Reject("CUDA HDG initial ParaView output failed");

  while (!time_driver->Finished())
  {
    double proposed_dt = 0.0;
    if (time_driver->mode() == HdgTimeControlMode::CFLBackwardEuler)
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
      const bool proposal_consensus = ExactDoubleConsensus(proposed_dt);
      if (!CollectiveAll(proposal_valid && proposal_consensus))
        return Reject("committed CUDA CFL proposal is invalid or below --dt-min");
    }
    const auto& plan = time_driver->BeginStep(proposed_dt, request.options.dt_min);
    GenericEulerCudaPreparedStep authority;
    CudaPreparedOutputMetadataV1 native_export{};
    if (session.Prepare(plan.dt, authority, native_export).status !=
        HdgEulerSessionStatus::prepared)
    {
      time_driver->RecordFailure(HdgStepFailurePhase::NonlinearSolve);
      return Reject("generic CUDA prepared step failed");
    }
    if (!CollectiveAll(FreshOutputView(*published_state, scalar_count,
                                       destination)))
    {
      local_ok = AbortPrepared(session, authority, native_export);
      time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
      return Reject(local_ok ? "CUDA prepared-state output view failed"
                             : "CUDA prepared abort failed after output-view failure");
    }
    local_ok = session.ExportPreparedInterior(authority, native_export,
                                              destination);
    if (!CollectiveAll(local_ok))
    {
      local_ok = AbortPrepared(session, authority, native_export);
      time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
      return Reject(local_ok ? "CUDA prepared-state export failed"
                             : "CUDA prepared abort failed after export failure");
    }
    if (!session.AcknowledgePreparedOutput(authority, native_export))
    {
      local_ok = AbortPrepared(session, authority, native_export);
      time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
      return Reject(local_ok ? "CUDA prepared output acknowledgement failed"
                             : "CUDA prepared abort failed after acknowledgement failure");
    }

    std::optional<HdgTransactionalTimeAdapter::OutputEvent> preview;
    try { preview = time_driver->PreviewFinalizedOutput(); local_ok = true; }
    catch (...) { local_ok = false; }
    if (!CollectiveAll(local_ok))
    {
      local_ok = AbortPrepared(session, authority, native_export);
      time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
      return Reject(local_ok ? "CUDA output preview failed"
                             : "CUDA prepared abort failed after preview failure");
    }
    if (session.Commit(std::move(authority), native_export).status !=
        HdgEulerSessionStatus::committed)
      return Reject("generic CUDA prepared commit failed");
    // Publication remains post-commit: replace the staged physical candidate
    // with the engine's committed bytes before finalizing time/output.
    if (!CollectiveAll(FreshOutputView(*published_state, scalar_count,
                                       destination)))
      return Reject("CUDA committed-state output view failed after commit");
    local_ok = session.ExportCommittedInterior(destination);
    if (!CollectiveAll(local_ok))
      return Reject("CUDA committed-state export failed after commit");

    std::optional<HdgTransactionalTimeAdapter::OutputEvent> finalized;
    try { finalized = time_driver->FinalizeSuccess(); local_ok = true; }
    catch (...) { local_ok = false; }
    if (!CollectiveAll(local_ok))
      return Reject("CUDA time transaction finalization failed after commit");
    local_ok = preview.has_value() == finalized.has_value();
    if (local_ok && preview)
      local_ok = preview->cycle == finalized->cycle && preview->time == finalized->time;
    if (!CollectiveAll(local_ok))
      return Reject("CUDA finalized output differs from its preview");
    if (output && finalized)
    {
      try { output->Write(Metadata(*finalized, time_driver->solver_step())); local_ok = true; }
      catch (...) { local_ok = false; }
      if (!CollectiveAll(local_ok))
        return Reject("CUDA ParaView output failed after commit");
    }
  }

  if (IsRoot())
    std::cout << "[hdg-main] status=Success step=" << time_driver->solver_step()
              << " time=" << time_driver->time() << " output_cycle="
              << time_driver->output_cycle() << " regularization="
              << (regularization.kind == RegularizationKind::igr ? "igr" : "none")
              << " alpha=" << regularization.alpha
              << " beta=" << regularization.beta << '\n';
  return 0;
}

} // namespace dg_sim::hdg

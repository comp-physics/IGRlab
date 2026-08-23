#include "sim/hdg/main_backend.hpp"

#include <algorithm>
#include <cmath>
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
#include <vector>

#include "numerics/hdg/generic/contract.hpp"
#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "sim/context/case_context.hpp"
#include "sim/hdg/euler_output_fields.hpp"
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
using dg::hdg::generic::RegularizationKind;
using dg::hdg::generic::RegularizationSelection;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerHdgCpuSession;
using dg::hdg::runtime::cpu::GenericEulerCpuPreparedStep;

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

bool LiveCommunicator(const MPI_Comm communicator) noexcept
{
  int initialized = 0;
  int finalized = 0;
  int rank = -1;
  int ranks = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Initialized(&initialized) == MPI_SUCCESS && initialized &&
      MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized &&
      MPI_Comm_rank(communicator, &rank) == MPI_SUCCESS &&
      MPI_Comm_size(communicator, &ranks) == MPI_SUCCESS &&
      rank >= 0 && ranks > 0;
}

bool CollectiveAll(const MPI_Comm communicator, const bool local) noexcept
{
  const int input = local ? 1 : 0;
  int output = 0;
  return communicator != MPI_COMM_NULL &&
      MPI_Allreduce(&input, &output, 1, MPI_INT, MPI_MIN, communicator) ==
          MPI_SUCCESS &&
      output == 1;
}

bool ExactDoubleConsensus(const MPI_Comm communicator, const double value) noexcept
{
  if (communicator == MPI_COMM_NULL) return false;
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value),
                "HDG main consensus requires IEEE binary64 doubles");
  std::memcpy(&bits, &value, sizeof(bits));
  std::uint64_t minimum = 0;
  std::uint64_t maximum = 0;
  const int minimum_status = MPI_Allreduce(
      &bits, &minimum, 1, MPI_UINT64_T, MPI_MIN, communicator);
  const int maximum_status = MPI_Allreduce(
      &bits, &maximum, 1, MPI_UINT64_T, MPI_MAX, communicator);
  return minimum_status == MPI_SUCCESS && maximum_status == MPI_SUCCESS &&
      minimum == maximum;
}

bool ExactTextConsensus(const MPI_Comm communicator,
                        const std::string& local) noexcept
{
  int rank = -1;
  if (communicator == MPI_COMM_NULL ||
      MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS)
  {
    return false;
  }
  std::uint64_t root_size = rank == 0
      ? static_cast<std::uint64_t>(local.size())
      : 0;
  const bool size_broadcast =
      MPI_Bcast(&root_size, 1, MPI_UINT64_T, 0, communicator) == MPI_SUCCESS;
  if (!CollectiveAll(
          communicator,
          size_broadcast &&
              root_size <= static_cast<std::uint64_t>(
                  std::numeric_limits<int>::max())))
  {
    return false;
  }

  std::vector<char> root;
  bool allocated = true;
  try
  {
    root.resize(static_cast<std::size_t>(root_size));
  }
  catch (...)
  {
    allocated = false;
  }
  if (!CollectiveAll(communicator, allocated)) return false;
  if (rank == 0 && root_size != 0)
  {
    std::memcpy(root.data(), local.data(), static_cast<std::size_t>(root_size));
  }
  char empty = 0;
  char* const data = root_size == 0 ? &empty : root.data();
  const bool text_broadcast =
      MPI_Bcast(data, static_cast<int>(root_size), MPI_CHAR, 0,
                communicator) == MPI_SUCCESS;
  if (!CollectiveAll(communicator, text_broadcast))
  {
    return false;
  }
  return CollectiveAll(
      communicator,
      local.size() == static_cast<std::size_t>(root_size) &&
          std::equal(local.begin(), local.end(), root.begin()));
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

std::string CanonicalMainRequest(
    const HdgEulerMainRequest& request,
    const std::string& mesh_path,
    const mfem::io::VTKOutputConfig& vtk,
    const RegularizationSelection regularization)
{
  std::ostringstream stream;
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

std::string CanonicalNumericalPolicy(
    const dg::hdg::generic::EulerHdgNumericalPolicy& policy)
{
  std::ostringstream stream;
  stream << policy.volume_projection_quadrature_order << ';'
         << policy.true_trace_projection_quadrature_order << ';'
         << policy.conventional_assembly_quadrature_order << ';'
         << policy.igr_assembly_quadrature_order << ';'
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
  throw std::invalid_argument("HDG regularization must resolve to 'none' or 'igr'");
}

int Reject(const HdgEulerMainRequest& request,
           const std::string& reason) noexcept
{
  if (IsRoot(request.communicator))
  {
    std::cerr << "[hdg-main] status=Rejected reason=" << reason << '\n';
  }
  return 1;
}

std::string Validate(const HdgEulerMainRequest& request,
                     const RegularizationSelection regularization)
{
  int initialized = 0;
  int finalized = 0;
  if (request.communicator == MPI_COMM_NULL ||
      MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized ||
      MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
  {
    return "invalid MPI communicator or lifecycle";
  }
  if (!request.make_case_data)
  {
    return "HDG case physics factory is empty";
  }
  if (ToLowerCopy(request.options.device) != "cpu")
  {
    return "generic main-backend V1 supports only --device cpu";
  }
  if (!dg::hdg::generic::ValidRegularizationSelection(regularization))
  {
    return "HDG IGR alpha and beta must be finite and positive";
  }
  if (!request.options.restart_vtk.empty() ||
      request.options.restart_cycle >= 0 || request.options.restart_step >= 0)
  {
    return "generic HDG restart is deferred and unsupported";
  }
  if (request.options.glvis)
  {
    return "generic HDG GLVis output is unsupported";
  }
  if (!std::isfinite(request.options.dt))
  {
    return "HDG --time-step must be finite";
  }
  if (request.options.dt <= 0.0 &&
      (!std::isfinite(request.options.cfl) || request.options.cfl <= 0.0))
  {
    return "CFL Backward Euler requires a finite positive --cfl-number";
  }
  if (!std::isfinite(request.options.dt_min) || request.options.dt_min < 0.0)
  {
    return "HDG --dt-min must be finite and non-negative";
  }
  if (!std::isfinite(request.options.t_final) ||
      request.options.t_final <= 0.0)
  {
    return "HDG final time must be finite and positive";
  }
  if (request.options.order < dg::hdg::generic::minimum_polynomial_order ||
      request.options.order > dg::hdg::generic::maximum_polynomial_order)
  {
    return "generic HDG supports polynomial orders one through three";
  }
  if (request.options.ser_ref_levels < 0 ||
      request.options.par_ref_levels < 0)
  {
    return "HDG mesh refinement levels must be non-negative";
  }
  if (request.options.vtk_save &&
      (!std::isfinite(request.options.dt_save) ||
       request.options.dt_save <= 0.0))
  {
    return "HDG ParaView output requires a positive --time-step-save";
  }
  if (!std::isfinite(request.case_config.gamma) ||
      request.case_config.gamma != mfem::dg_igr::EulerDefaults::kGamma)
  {
    return "generic HDG V1 requires the frozen Euler gamma of 1.4";
  }
  return {};
}

std::string CollectionName(const HdgEulerMainRequest& request)
{
  std::ostringstream name;
  name << request.case_config.case_name << "_hdg_p_" << request.options.order
       << "_ref_"
       << request.options.ser_ref_levels + request.options.par_ref_levels;
  return name.str();
}

HDGParaViewMetadata Metadata(
    const HdgTransactionalTimeAdapter::OutputEvent& event,
    const std::uint64_t solver_step)
{
  if (event.cycle > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      solver_step > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
  {
    throw std::overflow_error("HDG output cycle or solver step exceeds int");
  }
  return {static_cast<int>(event.cycle),
          static_cast<int>(solver_step),
          event.time};
}

} // namespace

int RunEulerMainBackend(const HdgEulerMainRequest& request)
{
  try
  {
    if (!LiveCommunicator(request.communicator))
    {
      return Reject(request, "invalid MPI communicator or lifecycle");
    }
    std::string rejection;
    RegularizationSelection regularization{};
    bool local_ok = true;
    try
    {
      regularization = CanonicalRegularization(request);
      rejection = Validate(request, regularization);
      local_ok = rejection.empty();
    }
    catch (const std::exception& error)
    {
      rejection = error.what();
      local_ok = false;
    }
    catch (...)
    {
      rejection = "unknown rank-local request validation failure";
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request, rejection.empty()
          ? "request validation failed on another rank"
          : rejection);
    }

    MeshBuildRequest mesh_request;
    mfem::io::VTKOutputConfig vtk;
    std::string canonical_request;
    try
    {
      mesh_request.mesh_path =
        ResolveCaseMesh(request.case_config, request.options.mesh_file);
      mesh_request.serial_refinement_levels = request.options.ser_ref_levels;
      mesh_request.parallel_refinement_levels = request.options.par_ref_levels;
      mesh_request.mesh_path_precedence = request.options.mesh_file.empty()
        ? MeshPathPrecedence::CaseDefault
        : MeshPathPrecedence::CommandLineOverride;
      vtk = MakeCaseVTKConfig(request.case_config);
      canonical_request = CanonicalMainRequest(
          request, mesh_request.mesh_path, vtk, regularization);
      local_ok = true;
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request,
                    "main-backend configuration resolution failed on at least one rank");
    }
    if (!ExactTextConsensus(request.communicator, canonical_request))
    {
      return Reject(request,
                    "main-backend configuration differs between MPI ranks");
    }

    std::optional<mfem::Mesh> serial_mesh;
    std::string serialized_mesh;
    try
    {
      serial_mesh.emplace(LoadSerialMesh(
          mesh_request.mesh_path,
          mesh_request.serial_refinement_levels));
      std::ostringstream mesh_stream;
      mesh_stream.precision(std::numeric_limits<double>::max_digits10);
      serial_mesh->Print(mesh_stream);
      serialized_mesh = mesh_stream.str();
      local_ok = serial_mesh->Dimension() == 2 &&
          serial_mesh->SpaceDimension() == 2 && !serialized_mesh.empty();
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request, "HDG serial mesh loading failed on at least one rank");
    }
    if (!ExactTextConsensus(request.communicator, serialized_mesh))
    {
      return Reject(request,
                    "HDG serial mesh content differs between MPI ranks");
    }

    std::unique_ptr<CaseContext> context;
    try
    {
      MeshBuildMetadata metadata;
      metadata.mesh_path = mesh_request.mesh_path;
      metadata.serial_refinement_levels =
          mesh_request.serial_refinement_levels;
      metadata.parallel_refinement_levels =
          mesh_request.parallel_refinement_levels;
      metadata.mesh_path_precedence = mesh_request.mesh_path_precedence;
      metadata.serial_element_count = serial_mesh->GetNE();
      auto parallel_mesh = std::make_unique<mfem::ParMesh>(
          request.communicator, *serial_mesh);
      serial_mesh->Clear();
      metadata.local_elements_after_partitioning = parallel_mesh->GetNE();
      RefineParMesh(*parallel_mesh,
                    mesh_request.parallel_refinement_levels);
      metadata.local_elements_after_parallel_refinement =
          parallel_mesh->GetNE();
      context = std::make_unique<CaseContext>(std::move(parallel_mesh),
                                              std::move(metadata));
      local_ok = true;
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request,
                    "HDG parallel mesh construction failed on at least one rank");
    }

    // Case physics is deliberately resolved only after the generic adapter
    // has constructed the backend-neutral mesh.  The factory owns analytic
    // coefficients; this adapter borrows them solely for the projections
    // below, while the semantic boundary plan is frozen before lowering.
    std::optional<HdgEulerCaseData> case_data;
    std::optional<FrozenBoundaryPlan> frozen_boundary;
    dg::hdg::generic::EulerHdgNumericalPolicy numerical_policy{};
    std::string canonical_case_data;
    try
    {
      case_data.emplace(request.make_case_data(*context));
      if (!case_data->initial_condition)
      {
        throw std::invalid_argument("HDG case data is missing an initial condition");
      }
      if (case_data->initial_condition->GetVDim() !=
          dg::hdg::generic::euler_2d_conserved_components)
      {
        throw std::invalid_argument(
            "HDG Euler initial condition must have four components");
      }
      if (case_data->static_source &&
          case_data->static_source->GetVDim() !=
            dg::hdg::generic::euler_2d_conserved_components)
      {
        throw std::invalid_argument(
            "HDG Euler static source must have four components");
      }
      numerical_policy = case_data->numerical_policy;
      // Cases own all numerical roles except command-line regularization.
      numerical_policy.stabilization = regularization;
      if (!dg::hdg::generic::ValidEulerHdgNumericalPolicy(numerical_policy))
      {
        throw std::invalid_argument("HDG case data has an invalid numerical policy");
      }
      frozen_boundary.emplace(FreezeBoundaryPlan(case_data->boundary_plan));
      std::ostringstream case_data_key;
      case_data_key << case_data->initial_condition->GetVDim() << ';'
                    << (case_data->static_source ? 1 : 0) << ';';
      if (case_data->static_source)
      {
        case_data_key << case_data->static_source->GetVDim() << ';';
      }
      AppendText(case_data_key, frozen_boundary->consensus_key);
      AppendText(case_data_key, CanonicalNumericalPolicy(numerical_policy));
      canonical_case_data = case_data_key.str();
      local_ok = true;
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request,
                    "HDG case-data factory failed on at least one rank");
    }
    if (!ExactTextConsensus(request.communicator, canonical_case_data))
    {
      return Reject(request,
                    "HDG case data differs between MPI ranks");
    }

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
          request.options.order, /*dim=*/2,
          mfem::BasisType::GaussLegendre,
          mfem::FiniteElement::VALUE);
      volume_space = std::make_unique<mfem::ParFiniteElementSpace>(
          &context->Mesh(), collection.get(),
          dg::hdg::generic::euler_2d_conserved_components,
          mfem::Ordering::byNODES);
      initial_state =
          std::make_unique<mfem::ParGridFunction>(volume_space.get());
      trace_collection = std::make_unique<mfem::DG_Interface_FECollection>(
          request.options.order, /*dim=*/2, mfem::FiniteElement::VALUE,
          mfem::BasisType::GaussLegendre);
      trace_space = std::make_unique<mfem::ParFiniteElementSpace>(
          &context->Mesh(), trace_collection.get(),
          dg::hdg::generic::euler_2d_conserved_components,
          mfem::Ordering::byNODES);
      local_initial_trace =
          std::make_unique<mfem::ParGridFunction>(trace_space.get());
      published_state =
          std::make_unique<mfem::ParGridFunction>(volume_space.get());
      if (!ProjectEulerCoefficientL2(
              *volume_space, *initial_state, *case_data->initial_condition,
              numerical_policy.volume_projection_quadrature_order) ||
          initial_state->CheckFinite() != 0)
      {
        throw std::runtime_error("HDG L2 volume initial-state projection failed");
      }
      local_ok = true;
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request,
                    "HDG volume initial-state projection failed on at least one rank");
    }

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
        source_state =
          std::make_unique<mfem::ParGridFunction>(volume_space.get());
        source_state->ProjectCoefficient(*case_data->static_source);
        MFEM_VERIFY(source_state->CheckFinite() == 0,
                    "HDG state I/O refuses non-finite static-source coordinates.");
      }
      const int boundary_attributes =
          context->Mesh().bdr_attributes.Size() == 0
          ? 0
          : context->Mesh().bdr_attributes.Max();
      boundary.emplace(dg::hdg::generic::LowerHdgEulerBoundaryPlan(
          frozen_boundary->plan, /*space_dimension=*/2,
          request.case_config.gamma, boundary_attributes));
      local_ok = true;
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request,
                    "HDG state projection or boundary lowering failed on at least one rank");
    }

    const BackendSelection backend{BackendKind::cpu, -1};
    ResolvedHdgEulerCasePlan resolved_plan{};
    resolved_plan.request = EulerHdgRequest{
      &context->Mesh(),
      context->Comm(),
      request.options.order,
      backend,
      dg::hdg::generic::RequiredCapabilities(backend),
      {initial_state->HostRead(), static_cast<std::size_t>(initial_state->Size()),
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
      boundary->View(),
      numerical_policy};

    std::unique_ptr<GenericEulerHdgCpuSession> session;
    try
    {
      session = std::make_unique<GenericEulerHdgCpuSession>();
      local_ok = true;
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request,
                    "generic CPU session construction failed on at least one rank");
    }
    local_ok = session->Initialize(resolved_plan.request).status ==
        GenericEulerCpuSessionStatus::initialized;
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request, "generic CPU session initialization failed");
    }
    std::unique_ptr<HdgTransactionalTimeAdapter> time_driver;
    try
    {
      resolved_plan.output_schedule = request.options.vtk_save
        ? HdgTransactionalTimeAdapter::OutputSchedule::FixedInterval(
            request.options.dt_save,
            time::backend::InitialOutputPolicy::Emit,
            time::backend::FinalOutputPolicy::Emit)
        : HdgTransactionalTimeAdapter::OutputSchedule::Disabled(
            time::backend::InitialOutputPolicy::Suppress,
            time::backend::FinalOutputPolicy::Suppress);
      const bool cfl_control = request.options.dt <= 0.0;
      resolved_plan.time_control = cfl_control
          ? HdgTransactionalTimeAdapter::TimeControl::CFL(
                0.0, request.options.t_final, request.options.cfl)
          : HdgTransactionalTimeAdapter::TimeControl::Fixed(
                0.0, request.options.t_final, request.options.dt);
      resolved_plan.time_mode = cfl_control
          ? HdgEulerTimeMode::CFLBackwardEuler
          : HdgEulerTimeMode::FixedBackwardEuler;
      resolved_plan.output = {request.options.vtk_save,
                              vtk.prefix_path,
                              CollectionName(request)};
      time_driver = std::make_unique<HdgTransactionalTimeAdapter>(
          resolved_plan.time_control, resolved_plan.output_schedule,
          cfl_control ? HdgTimeControlMode::CFLBackwardEuler
                      : HdgTimeControlMode::FixedBackwardEuler);
      local_ok = true;
    }
    catch (...)
    {
      local_ok = false;
    }
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request,
                    "HDG time/output planner construction failed on at least one rank");
    }

    std::unique_ptr<HdgEulerParaViewFields> output_fields;
    std::unique_ptr<HDGParaViewSeries> output;
    if (request.options.vtk_save)
    {
      const int levels = vtk.levels_of_detail >= 0
        ? vtk.levels_of_detail
        : request.options.order + 1;
      try
      {
        const int output_size = initial_state->Size();
        if (output_size < 0 || published_state->Size() != output_size)
        {
          throw std::runtime_error("HDG Euler output-state extent mismatch");
        }
        if (output_size != 0)
        {
          const double* const source = initial_state->HostRead();
          double* const destination = published_state->HostWrite();
          if (source == nullptr || destination == nullptr)
          {
            throw std::runtime_error("HDG Euler output-state storage is null");
          }
          std::copy_n(source, output_size, destination);
        }
        output_fields = std::make_unique<HdgEulerParaViewFields>(
            context->Mesh(), *collection, *published_state,
            regularization.kind == RegularizationKind::igr,
            request.case_config.gamma,
            request.case_config.rk3_config.rho_min);
        const HDGParaViewRequest output_request{
          context->Mesh(),
          output_fields->Fields(),
          {resolved_plan.output.prefix_path,
           resolved_plan.output.collection_name, vtk.format, levels,
           vtk.high_order_output}};
        output = std::make_unique<HDGParaViewSeries>(output_request);
        local_ok = true;
      }
      catch (...)
      {
        local_ok = false;
      }
      if (!CollectiveAll(request.communicator, local_ok))
      {
        return Reject(request,
                      "HDG ParaView setup failed on at least one rank");
      }
      if (const auto initial = time_driver->InitialOutput())
      {
        dg::hdg::generic::HdgEulerHostInteriorFields destination{};
        local_ok = output_fields->FreshHostOutput(destination);
        if (!CollectiveAll(request.communicator, local_ok))
        {
          return Reject(request,
                        "initial Euler/IGR output view construction failed");
        }
        local_ok = session->ExportCommittedFields(destination);
        if (!CollectiveAll(request.communicator, local_ok))
        {
          return Reject(request,
                        "initial committed Euler/IGR field export failed");
        }
        local_ok = output_fields->UpdateDerived();
        if (!CollectiveAll(request.communicator, local_ok))
        {
          return Reject(request,
                        "initial derived Euler field construction failed");
        }
        try
        {
          output->Write(Metadata(*initial, /*solver_step=*/0));
          local_ok = true;
        }
        catch (...) { local_ok = false; }
        if (!CollectiveAll(request.communicator, local_ok))
        {
          return Reject(request, "initial HDG ParaView output failed");
        }
      }
    }

    while (!time_driver->Finished())
    {
      double proposed_dt = 0.0;
      if (time_driver->mode() == HdgTimeControlMode::CFLBackwardEuler)
      {
        const auto diagnostics = session->diagnostics();
        const bool diagnostics_ready =
            dg::hdg::generic::HasHdgEulerSessionCapabilities(
                session->capabilities(),
                dg::hdg::generic::HdgEulerSessionCapabilityBit(
                    dg::hdg::generic::HdgEulerSessionCapability::
                        committed_cfl_diagnostics)) &&
            dg::hdg::generic::IsValidHdgEulerCommittedDiagnostics(
                diagnostics, session->capabilities()) &&
            diagnostics.coordinates.time == time_driver->time() &&
            diagnostics.coordinates.step == time_driver->solver_step() &&
            diagnostics.coordinates.generation == diagnostics.wave_generation;
        proposed_dt = diagnostics.backend_stable_dt * request.options.cfl;
        const bool proposal_valid = diagnostics_ready &&
            std::isfinite(proposed_dt) && proposed_dt > 0.0 &&
            (request.options.dt_min <= 0.0 ||
             proposed_dt >= request.options.dt_min);
        const bool proposal_consensus =
            ExactDoubleConsensus(request.communicator, proposed_dt);
        if (!CollectiveAll(request.communicator,
                           proposal_valid && proposal_consensus))
        {
          return Reject(request,
                        "committed CFL proposal is invalid or below --dt-min");
        }
      }
      const auto& plan = time_driver->BeginStep(proposed_dt,
                                                 request.options.dt_min);
      GenericEulerCpuPreparedStep prepared;
      local_ok = session->Prepare(plan.dt, prepared).status ==
          GenericEulerCpuSessionStatus::prepared;
      if (!CollectiveAll(request.communicator, local_ok))
      {
        // Prepare only mints a token after its collective admission succeeds.
        time_driver->RecordFailure(HdgStepFailurePhase::NonlinearSolve);
        return Reject(request, "generic CPU prepared step failed");
      }

      std::optional<HdgTransactionalTimeAdapter::OutputEvent> preview;
      try
      {
        preview = time_driver->PreviewFinalizedOutput();
        local_ok = true;
      }
      catch (...)
      {
        local_ok = false;
      }
      if (!CollectiveAll(request.communicator, local_ok))
      {
        local_ok = session->Abort(std::move(prepared)).status ==
            GenericEulerCpuSessionStatus::aborted;
        if (!CollectiveAll(request.communicator, local_ok))
        {
          time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
          return Reject(
              request,
              "generic CPU prepared-step abort failed after output preview failure");
        }
        time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
        return Reject(request, "HDG output preview failed");
      }
      if (preview)
      {
        dg::hdg::generic::HdgEulerHostInteriorFields destination{};
        local_ok = output_fields &&
            output_fields->FreshHostOutput(destination);
        if (!CollectiveAll(request.communicator, local_ok))
        {
          local_ok = session->Abort(std::move(prepared)).status ==
              GenericEulerCpuSessionStatus::aborted;
          if (!CollectiveAll(request.communicator, local_ok))
          {
            time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
            return Reject(
                request,
                "generic CPU prepared-step abort failed after field-export failure");
          }
          time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
          return Reject(request, "prepared Euler/IGR output view failed");
        }
        local_ok = session->ExportPreparedFields(prepared, destination);
        if (!CollectiveAll(request.communicator, local_ok))
        {
          local_ok = session->Abort(std::move(prepared)).status ==
              GenericEulerCpuSessionStatus::aborted;
          if (!CollectiveAll(request.communicator, local_ok))
          {
            time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
            return Reject(
                request,
                "generic CPU prepared-step abort failed after field-export failure");
          }
          time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
          return Reject(request, "prepared Euler/IGR field export failed");
        }
        local_ok = output_fields->UpdateDerived();
        if (!CollectiveAll(request.communicator, local_ok))
        {
          local_ok = session->Abort(std::move(prepared)).status ==
              GenericEulerCpuSessionStatus::aborted;
          if (!CollectiveAll(request.communicator, local_ok))
          {
            time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
            return Reject(
                request,
                "generic CPU prepared-step abort failed after derived-field failure");
          }
          time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
          return Reject(request, "prepared derived Euler field update failed");
        }
      }
      local_ok = session->Commit(std::move(prepared)).status ==
          GenericEulerCpuSessionStatus::committed;
      if (!CollectiveAll(request.communicator, local_ok))
      {
        local_ok = session->Abort(std::move(prepared)).status ==
            GenericEulerCpuSessionStatus::aborted;
        if (!CollectiveAll(request.communicator, local_ok))
        {
          return Reject(
              request,
              "generic CPU prepared-step abort failed after commit failure");
        }
        return Reject(request, "generic CPU prepared commit failed");
      }

      std::optional<HdgTransactionalTimeAdapter::OutputEvent> finalized;
      try
      {
        finalized = time_driver->FinalizeSuccess();
        local_ok = true;
      }
      catch (...)
      {
        local_ok = false;
      }
      if (!CollectiveAll(request.communicator, local_ok))
      {
        return Reject(request, "HDG time transaction finalization failed after commit");
      }

      local_ok = preview.has_value() == finalized.has_value();
      if (local_ok && preview)
      {
        local_ok = preview->cycle == finalized->cycle &&
            preview->time == finalized->time &&
            preview->is_initial == finalized->is_initial &&
            preview->is_final == finalized->is_final;
      }
      if (!CollectiveAll(request.communicator, local_ok))
      {
        return Reject(request, "HDG finalized output differs from its preview");
      }

      if (output && finalized)
      {
        try
        {
          output->Write(Metadata(*finalized, time_driver->solver_step()));
          local_ok = true;
        }
        catch (...)
        {
          local_ok = false;
        }
        if (!CollectiveAll(request.communicator, local_ok))
        {
          // The numerical and time transactions are already committed.  The
          // writer cannot be aborted or rolled back; partial output is only a
          // failed-run artifact and never names an uncommitted state.
          return Reject(request, "HDG ParaView output failed after commit");
        }
      }
    }

    if (IsRoot(request.communicator))
    {
      std::cout << "[hdg-main] status=Success step="
                << time_driver->solver_step() << " time="
                << time_driver->time() << " output_cycle="
                << time_driver->output_cycle() << " regularization="
                << (regularization.kind == RegularizationKind::igr ? "igr" : "none")
                << " alpha=" << regularization.alpha
                << " beta=" << regularization.beta << '\n';
    }
    return 0;
  }
  catch (const std::exception& error)
  {
    return Reject(request, error.what());
  }
  catch (...)
  {
    return Reject(request, "unknown main-backend failure");
  }
}

} // namespace dg_sim::hdg

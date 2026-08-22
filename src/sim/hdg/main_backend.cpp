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
#include "sim/hdg/parallel_vtk.hpp"
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
      if (entry.inlet_model == EulerInletModel::Supersonic)
      {
        frozen.plan.AddSupersonicInlet(
            std::move(marker), [state] { return state; });
      }
      else
      {
        frozen.plan.AddSubsonicInlet(
            std::move(marker), [state] { return state; });
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
    const std::string& boundary_key,
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
  stream << request.initial_condition.GetVDim() << ';';
  AppendText(stream, boundary_key);
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
  if (ToLowerCopy(request.options.device) != "cpu")
  {
    return "generic main-backend V1 supports only --device cpu";
  }
  if (!dg::hdg::generic::ValidRegularizationSelection(regularization))
  {
    return "HDG IGR alpha and beta must be finite and positive";
  }
  if (regularization.kind == RegularizationKind::igr &&
      request.options.order != dg::hdg::generic::minimum_polynomial_order)
  {
    return "HDG IGR supports only polynomial order one";
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
  if (request.initial_condition.GetVDim() !=
      dg::hdg::generic::euler_2d_conserved_components)
  {
    return "HDG Euler initial condition must have four components";
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
    std::optional<FrozenBoundaryPlan> frozen_boundary;
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
      frozen_boundary.emplace(FreezeBoundaryPlan(request.boundary_plan));
      canonical_request = CanonicalMainRequest(
          request, mesh_request.mesh_path, vtk,
          frozen_boundary->consensus_key, regularization);
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

    std::unique_ptr<mfem::L2_FECollection> collection;
    std::unique_ptr<mfem::ParFiniteElementSpace> volume_space;
    std::unique_ptr<mfem::ParGridFunction> initial_state;
    std::unique_ptr<mfem::ParGridFunction> published_state;
    std::unique_ptr<HdgStateIo> state_io;
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
      published_state =
          std::make_unique<mfem::ParGridFunction>(volume_space.get());
      state_io = std::make_unique<HdgStateIo>(
          context->Mesh(), *volume_space, *initial_state);
      state_io->ProjectBorrowedInitialConditionOnce(
          request.initial_condition);
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
    const EulerHdgRequest engine_request{
      &context->Mesh(),
      context->Comm(),
      request.options.order,
      backend,
      dg::hdg::generic::RequiredCapabilities(backend),
      state_io->ImportPublishedInitialState(),
      {nullptr, 0, dg::hdg::generic::euler_2d_conserved_components},
      boundary->View(),
      regularization};

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
    local_ok = session->Initialize(engine_request).status ==
        GenericEulerCpuSessionStatus::initialized;
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request, "generic CPU session initialization failed");
    }
    local_ok = session->ExportCommittedInterior(*published_state);
    if (!CollectiveAll(request.communicator, local_ok))
    {
      return Reject(request, "initial committed-state export failed");
    }

    std::unique_ptr<HdgTransactionalTimeAdapter> time_driver;
    try
    {
      const auto output_schedule = request.options.vtk_save
        ? HdgTransactionalTimeAdapter::OutputSchedule::FixedInterval(
            request.options.dt_save,
            time::backend::InitialOutputPolicy::Emit,
            time::backend::FinalOutputPolicy::Emit)
        : HdgTransactionalTimeAdapter::OutputSchedule::Disabled(
            time::backend::InitialOutputPolicy::Suppress,
            time::backend::FinalOutputPolicy::Suppress);
      const bool cfl_control = request.options.dt <= 0.0;
      const auto control = cfl_control
          ? HdgTransactionalTimeAdapter::TimeControl::CFL(
                0.0, request.options.t_final, request.options.cfl)
          : HdgTransactionalTimeAdapter::TimeControl::Fixed(
                0.0, request.options.t_final, request.options.dt);
      time_driver = std::make_unique<HdgTransactionalTimeAdapter>(
          control, output_schedule,
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

    std::unique_ptr<HDGParaViewSeries> output;
    if (request.options.vtk_save)
    {
      const int levels = vtk.levels_of_detail >= 0
        ? vtk.levels_of_detail
        : request.options.order + 1;
      try
      {
        const HDGParaViewRequest output_request{
          context->Mesh(),
          {{"conservative", *published_state}},
          {vtk.prefix_path, CollectionName(request), vtk.format, levels,
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
        output->Write(Metadata(*initial, /*solver_step=*/0));
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

      // Keep the pure time cursor pending until the candidate state has been
      // staged in memory.  All ranks own a valid prepared token from here
      // until the same collective Commit or Abort consumes it.
      local_ok = session->ExportPreparedInterior(prepared, *published_state);
      if (!CollectiveAll(request.communicator, local_ok))
      {
        local_ok = session->Abort(std::move(prepared)).status ==
            GenericEulerCpuSessionStatus::aborted;
        if (!CollectiveAll(request.communicator, local_ok))
        {
          time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
          return Reject(
              request,
              "generic CPU prepared-step abort failed after export failure");
        }
        time_driver->RecordFailure(HdgStepFailurePhase::OutputStaging);
        return Reject(request, "prepared-state export failed");
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

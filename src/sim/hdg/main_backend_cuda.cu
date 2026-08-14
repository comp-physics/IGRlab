#include "sim/hdg/main_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "numerics/hdg/generic/session_contract.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"
#include "sim/context/case_context.hpp"
#include "sim/hdg/parallel_vtk.hpp"
#include "sim/hdg/transactional_time_adapter.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/wiring/hdg_boundary_wiring.hpp"

namespace dg_sim::hdg
{
namespace
{

using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::runtime::cuda_w8a::CudaConfig;
using dg::hdg::runtime::cuda_w8a::CudaMutableHostL2Output;
using dg::hdg::runtime::cuda_w8a::CudaSerializedCheckpointV1;

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

bool ExactFixedDt(const double value) noexcept
{
  const double fixed = dg::hdg::runtime::cuda_w8a::fixed_dt;
  return std::isfinite(value) && std::memcmp(&value, &fixed, sizeof(value)) == 0;
}

bool IntegralMultiple(const double value, const double increment) noexcept
{
  if (!std::isfinite(value) || value <= 0.0 || !ExactFixedDt(increment))
    return false;
  const double count = value / increment;
  const double rounded = std::round(count);
  return rounded >= 1.0 &&
      rounded <= static_cast<double>(std::numeric_limits<std::uint64_t>::max()) &&
      value == rounded * increment;
}

bool OnlyPeriodic(const EulerBoundaryPlan& plan) noexcept
{
  for (const EulerBoundaryPlanEntry& entry : plan.Entries())
    if (entry.kind != EulerBoundaryKind::Periodic) return false;
  return !plan.Entries().empty();
}

std::string ConsensusText(const HdgEulerMainRequest& request,
                          const std::string& mesh_path)
{
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::max_digits10)
         << request.case_config.case_name << '\n'
         << mesh_path << '\n'
         << ToLowerCopy(request.options.device) << '\n'
         << request.options.order << '\n'
         << request.options.ser_ref_levels << '\n'
         << request.options.par_ref_levels << '\n'
         << request.options.dt << '\n'
         << request.options.t_final << '\n'
         << request.options.dt_save << '\n'
         << request.options.vtk_save << '\n'
         << request.options.glvis << '\n'
         << request.options.restart_vtk << '\n'
         << request.options.restart_cycle << '\n'
         << request.options.restart_step << '\n'
         << request.options.cfl << '\n'
         << request.case_config.gamma << '\n'
         << request.initial_condition.GetVDim() << '\n'
         << request.boundary_plan.Entries().size() << '\n';
  for (const EulerBoundaryPlanEntry& entry : request.boundary_plan.Entries())
  {
    stream << static_cast<int>(entry.kind) << ':' << entry.marker.Size() << ':';
    for (int marker = 0; marker < entry.marker.Size(); ++marker)
      stream << entry.marker[marker] << ',';
    stream << '\n';
  }
  const mfem::io::VTKOutputConfig vtk = MakeCaseVTKConfig(request.case_config);
  stream << vtk.prefix_path << '\n' << static_cast<int>(vtk.format) << '\n'
         << vtk.high_order_output << '\n' << vtk.levels_of_detail << '\n';
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

std::string Validate(const HdgEulerMainRequest& request)
{
  if (!WorldCongruent(request.communicator))
    return "CUDA HDG requires MPI_COMM_WORLD-congruent ranks one or two";
  if (ToLowerCopy(request.options.device) != "cuda")
    return "generic CUDA HDG requires --device cuda";
  if (!request.options.restart_vtk.empty() || request.options.restart_cycle >= 0 ||
      request.options.restart_step >= 0 || request.options.save_state_files)
    return "generic CUDA HDG restart/state-file routes are unsupported";
  if (request.options.glvis)
    return "generic CUDA HDG GLVis is unsupported";
  if (!ExactFixedDt(request.options.dt))
    return "generic CUDA HDG requires --time-step 0.05";
  if (!IntegralMultiple(request.options.t_final, request.options.dt))
    return "generic CUDA HDG requires integral 0.05 final-time steps";
  if (request.options.vtk_save &&
      !IntegralMultiple(request.options.dt_save, request.options.dt))
    return "generic CUDA HDG requires a positive integral 0.05 ParaView cadence";
  if (request.options.order != dg::hdg::runtime::cuda_w8a::smoke_polynomial_order &&
      request.options.order != dg::hdg::runtime::cuda_w8a::functional_polynomial_order)
    return "generic CUDA HDG admits only p1/N3 or p2/N6";
  if (request.options.ser_ref_levels != 0 || request.options.par_ref_levels != 0)
    return "generic CUDA HDG does not admit mesh refinement";
  if (request.options.cfl != CommonOptions{}.cfl)
    return "generic CUDA HDG does not admit a caller-selected CFL control";
  if (request.initial_condition.GetVDim() !=
      dg::hdg::runtime::cuda_w8a::conserved_components)
    return "generic CUDA HDG requires a four-component Euler IC";
  if (!OnlyPeriodic(request.boundary_plan))
    return "generic CUDA HDG admits periodic boundary plans only";
  if (!std::isfinite(request.case_config.gamma) ||
      request.case_config.gamma != mfem::dg_igr::EulerDefaults::kGamma)
    return "generic CUDA HDG requires frozen Euler gamma 1.4";
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
                   CudaSerializedCheckpointV1& native_export) noexcept
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
    destination = {published_state.HostWrite(), scalar_count,
                   dg::hdg::runtime::cuda_w8a::conserved_components};
    return destination.coefficients != nullptr;
  }
  catch (...) { return false; }
}

bool ScheduleIsExactlyAdmitted(const CommonOptions& options) noexcept
{
  try
  {
    const auto schedule = options.vtk_save
        ? HdgTransactionalTimeAdapter::OutputSchedule::FixedInterval(
            options.dt_save, time::backend::InitialOutputPolicy::Emit,
            time::backend::FinalOutputPolicy::Emit)
        : HdgTransactionalTimeAdapter::OutputSchedule::Disabled(
            time::backend::InitialOutputPolicy::Suppress,
            time::backend::FinalOutputPolicy::Suppress);
    HdgTransactionalTimeAdapter cursor(
        HdgTransactionalTimeAdapter::TimeControl::Fixed(
            0.0, options.t_final, options.dt), schedule,
        HdgTimeControlMode::FixedBackwardEuler);
    while (!cursor.Finished())
    {
      if (!ExactFixedDt(cursor.BeginStep().dt)) return false;
      static_cast<void>(cursor.FinalizeSuccess());
    }
    return cursor.time() == options.t_final;
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
  try { reason = Validate(request); }
  catch (...) { reason = "rank-local CUDA HDG request validation failed"; }
  if (!CollectiveAll(reason.empty()))
    return Reject(reason.empty() ? "CUDA HDG request validation failed on another rank"
                                 : reason);

  std::string mesh_path;
  std::string consensus;
  try
  {
    mesh_path = ResolveCaseMesh(request.case_config, request.options.mesh_file);
    consensus = ConsensusText(request, mesh_path);
  }
  catch (...) { reason = "CUDA HDG configuration resolution failed"; }
  if (!CollectiveAll(!consensus.empty()))
    return Reject(reason.empty() ? "CUDA HDG configuration resolution failed on another rank"
                                 : reason);
  if (!ExactTextConsensus(consensus))
    return Reject("CUDA HDG configuration differs between MPI ranks");
  if (!CollectiveAll(ScheduleIsExactlyAdmitted(request.options)))
    return Reject("CUDA HDG time/output schedule is not exactly fixed-grid");

  std::optional<mfem::Mesh> serial_mesh;
  std::string serialized_mesh;
  bool local_ok = false;
  try
  {
    serial_mesh.emplace(LoadSerialMesh(mesh_path, 0));
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

  std::optional<CaseContext> context;
  try
  {
    MeshBuildMetadata metadata;
    metadata.mesh_path = mesh_path;
    metadata.mesh_path_precedence = request.options.mesh_file.empty()
        ? MeshPathPrecedence::CaseDefault : MeshPathPrecedence::CommandLineOverride;
    metadata.serial_element_count = serial_mesh->GetNE();
    auto parallel_mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD,
                                                         *serial_mesh);
    serial_mesh->Clear();
    metadata.local_elements_after_partitioning = parallel_mesh->GetNE();
    metadata.local_elements_after_parallel_refinement = parallel_mesh->GetNE();
    context.emplace(std::move(parallel_mesh), std::move(metadata));
    local_ok = true;
  }
  catch (...) { reason = "CUDA HDG periodic mesh construction failed"; }
  if (!CollectiveAll(context.has_value()))
    return Reject(reason.empty() ? "CUDA HDG mesh construction failed on another rank"
                                 : reason);

  local_ok = false;
  try
  {
    const int boundary_attributes = context->Mesh().bdr_attributes.Size() == 0
        ? 0 : context->Mesh().bdr_attributes.Max();
    const auto lowered = dg::hdg::generic::LowerHdgEulerBoundaryPlan(
        request.boundary_plan, 2, request.case_config.gamma,
        boundary_attributes);
    local_ok = lowered.AttributeCount() ==
        static_cast<std::size_t>(boundary_attributes);
    for (int attribute = 1; local_ok && attribute <= boundary_attributes;
         ++attribute)
      local_ok = lowered.IsPeriodicTopologyAttribute(attribute);
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(local_ok))
    return Reject("CUDA HDG periodic boundary lowering failed");

  const int mesh_n = request.options.order == 1
      ? dg::hdg::runtime::cuda_w8a::smoke_elements_per_direction
      : dg::hdg::runtime::cuda_w8a::functional_elements_per_direction;
  local_ok = false;
  std::unique_ptr<mfem::L2_FECollection> collection;
  std::unique_ptr<mfem::ParFiniteElementSpace> volume_space;
  std::unique_ptr<mfem::ParGridFunction> projected_initial;
  std::unique_ptr<mfem::ParGridFunction> published_state;
  try
  {
    collection = std::make_unique<mfem::L2_FECollection>(
        request.options.order, 2, mfem::BasisType::GaussLegendre,
        mfem::FiniteElement::VALUE);
    volume_space = std::make_unique<mfem::ParFiniteElementSpace>(
        &context->Mesh(), collection.get(),
        dg::hdg::runtime::cuda_w8a::conserved_components,
        mfem::Ordering::byNODES);
    projected_initial = std::make_unique<mfem::ParGridFunction>(volume_space.get());
    published_state = std::make_unique<mfem::ParGridFunction>(volume_space.get());
    projected_initial->ProjectCoefficient(request.initial_condition);
    local_ok = projected_initial->CheckFinite() == 0 &&
        volume_space->GetVSize() > 0 &&
        volume_space->GetVSize() % dg::hdg::runtime::cuda_w8a::conserved_components == 0;
  }
  catch (...) { local_ok = false; }
  if (!CollectiveAll(local_ok))
    return Reject("CUDA HDG initial-condition projection failed");

  const std::size_t scalar_count = static_cast<std::size_t>(
      volume_space->GetVSize() / dg::hdg::runtime::cuda_w8a::conserved_components);
  CudaConfig config;
  config.order = request.options.order;
  config.mesh_n = mesh_n;
  config.dt = dg::hdg::runtime::cuda_w8a::fixed_dt;
  config.requested_device = 0;
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  config.expected_rank_count = ranks;
  config.require_host_staged_rank2 = ranks == 2;
  config.use_borrowed_topology = true;
  config.borrowed_mesh = &context->Mesh();
  config.borrowed_communicator = MPI_COMM_WORLD;
  config.use_borrowed_host_inputs = true;
  config.borrowed_initial = {projected_initial->HostRead(), scalar_count,
                             dg::hdg::runtime::cuda_w8a::conserved_components};
  config.borrowed_source = {nullptr, 0,
                            dg::hdg::runtime::cuda_w8a::conserved_components};

  GenericEulerHdgCudaSession session;
  if (session.Initialize(config).status != HdgEulerSessionStatus::initialized)
    return Reject("generic CUDA session initialization failed");
  CudaMutableHostL2Output destination{};
  if (!CollectiveAll(FreshOutputView(*published_state, scalar_count,
                                     destination)) ||
      !session.ExportCommittedInterior(destination))
    return Reject("initial CUDA committed-state export failed");

  std::unique_ptr<HdgTransactionalTimeAdapter> time_driver;
  std::unique_ptr<HDGParaViewSeries> output;
  try
  {
    time_driver = std::make_unique<HdgTransactionalTimeAdapter>(
        HdgTransactionalTimeAdapter::TimeControl::Fixed(
            0.0, request.options.t_final, request.options.dt),
        request.options.vtk_save
            ? HdgTransactionalTimeAdapter::OutputSchedule::FixedInterval(
                request.options.dt_save,
                time::backend::InitialOutputPolicy::Emit,
                time::backend::FinalOutputPolicy::Emit)
            : HdgTransactionalTimeAdapter::OutputSchedule::Disabled(
                time::backend::InitialOutputPolicy::Suppress,
                time::backend::FinalOutputPolicy::Suppress),
        HdgTimeControlMode::FixedBackwardEuler);
    if (request.options.vtk_save)
    {
      const mfem::io::VTKOutputConfig vtk = MakeCaseVTKConfig(request.case_config);
      const int levels = vtk.levels_of_detail >= 0 ? vtk.levels_of_detail
                                                     : request.options.order + 1;
      output = std::make_unique<HDGParaViewSeries>(HDGParaViewRequest{
          context->Mesh(), {{"conservative", *published_state}},
          {vtk.prefix_path, CollectionName(request), vtk.format, levels,
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
    const auto& plan = time_driver->BeginStep();
    if (!CollectiveAll(ExactFixedDt(plan.dt)))
    {
      time_driver->RecordFailure(HdgStepFailurePhase::NonlinearSolve);
      return Reject("CUDA HDG time planner produced a non-fixed step");
    }
    GenericEulerCudaPreparedStep authority;
    CudaSerializedCheckpointV1 native_export{};
    if (session.Prepare(authority, native_export).status !=
        HdgEulerSessionStatus::prepared)
    {
      time_driver->RecordFailure(HdgStepFailurePhase::NonlinearSolve);
      return Reject("generic CUDA prepared step failed");
    }
    if (!CollectiveAll(FreshOutputView(*published_state, scalar_count,
                                       destination)) ||
        !session.ExportPreparedInterior(authority, native_export, destination))
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
                                       destination)) ||
        !session.ExportCommittedInterior(destination))
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
              << time_driver->output_cycle() << '\n';
  return 0;
}

} // namespace dg_sim::hdg

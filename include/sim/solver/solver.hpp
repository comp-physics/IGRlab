#pragma once

/**
 * @file sim/solver/solver.hpp
 * @brief Euler-specific solver orchestration for sim-layer cases.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "mfem.hpp"

#include "physics/igr/igr_coupling.hpp"

#include "sim/case/case_spec.hpp"
#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/context/mesh_build.hpp"
#include "sim/io/save_state.hpp"
#include "sim/io/run_metadata.hpp"
#include "sim/io/vis_helpers.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/io/vtk_restart.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/solver/solver_helpers.hpp"
#include "sim/time/time_defaults.hpp"
#include "sim/wiring/bc_wiring.hpp"
#include "sim/wiring/dg_operator.hpp"
#include "sim/wiring/dgsem_flux_config.hpp"
#include "sim/wiring/igr_physics.hpp"

namespace dg_sim
{

namespace detail
{

struct EulerConservedIntegral
{
  mfem::Vector components;
  double domain_measure = 0.0;
};

/**
 * Integrate every conservative component and the domain measure in one
 * traversal.  This diagnostic is intentionally used only by the opt-in
 * conservation qualification path; device states may synchronize here.
 */
inline EulerConservedIntegral IntegrateEulerConservedState(
  const mfem::ParGridFunction& state,
  DGBackend backend = DGBackend::DG)
{
  const mfem::ParFiniteElementSpace* fes = state.ParFESpace();
  MFEM_VERIFY(fes != nullptr,
              "Euler conservation diagnostics require a parallel FE space.");
  const int components = state.VectorDim();
  MFEM_VERIFY(components > 0,
              "Euler conservation diagnostics require state components.");

  mfem::Vector local(components);
  local = 0.0;
  double local_measure = 0.0;
  mfem::Vector value(components);
  for (int element = 0; element < fes->GetNE(); ++element)
  {
    const mfem::FiniteElement* fe = fes->GetFE(element);
    mfem::ElementTransformation* transformation =
      fes->GetElementTransformation(element);
    MFEM_VERIFY(fe != nullptr && transformation != nullptr,
                "Euler conservation integration is missing element data.");
    const mfem::IntegrationRule* rule = nullptr;
    if (backend == DGBackend::DGSEM)
    {
      // Match the GLL quadrature used to assemble the DGSEM mass exactly.
      // In particular, using a higher-order diagnostic rule on a curved mesh
      // would measure a different discrete conserved quantity.
      static mfem::IntegrationRules lgl_rules(
        0, mfem::Quadrature1D::GaussLobatto);
      const int integration_order = std::max(1, 2 * fe->GetOrder() - 1);
      rule = &lgl_rules.Get(fe->GetGeomType(), integration_order);
    }
    else
    {
      // Include the transformation-weight order so curved external meshes do
      // not turn geometry under-integration into an apparent balance defect.
      const int integration_order =
        2 * fe->GetOrder() + transformation->OrderW() + 2;
      rule = &mfem::IntRules.Get(fe->GetGeomType(), integration_order);
    }
    for (int q = 0; q < rule->GetNPoints(); ++q)
    {
      const mfem::IntegrationPoint& point = rule->IntPoint(q);
      transformation->SetIntPoint(&point);
      state.GetVectorValue(*transformation, point, value);
      const double weight = point.weight * transformation->Weight();
      MFEM_VERIFY(std::isfinite(weight) && weight >= 0.0,
                  "Euler conservation integration produced an invalid weight.");
      local_measure += weight;
      for (int component = 0; component < components; ++component)
      {
        local(component) += weight * value(component);
      }
    }
  }

  EulerConservedIntegral result;
  result.components.SetSize(components);
  MPI_Allreduce(local.GetData(),
                result.components.GetData(),
                components,
                mfem::MPITypeMap<mfem::real_t>::mpi_type,
                MPI_SUM,
                fes->GetComm());
  MPI_Allreduce(&local_measure,
                &result.domain_measure,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                fes->GetComm());
  return result;
}

inline std::string CompactVector(const mfem::Vector& values)
{
  std::ostringstream out;
  out << std::setprecision(17);
  for (int i = 0; i < values.Size(); ++i)
  {
    if (i > 0)
    {
      out << ',';
    }
    out << values(i);
  }
  return out.str();
}

inline double EulerConservationRelativeError(double absolute_error,
                                             double balance_scale,
                                             double domain_measure)
{
  if (!std::isfinite(absolute_error) || absolute_error < 0.0 ||
      !std::isfinite(balance_scale) || balance_scale < 0.0 ||
      !std::isfinite(domain_measure) || domain_measure <= 0.0)
  {
    return std::numeric_limits<double>::infinity();
  }
  const double denominator = domain_measure + balance_scale;
  return std::isfinite(denominator) && denominator > 0.0
           ? absolute_error / denominator
           : std::numeric_limits<double>::infinity();
}

inline std::string MakeEulerOutputDatasetName(
  const CaseConfig& case_cfg,
  const RunConfig& run_cfg,
  double c_alpha_value,
  const VTKRestartInfo* restart_info,
  const ExampleContext& ctx,
  const mfem::ParGridFunction* static_source)
{
  std::string name =
    MakeCaseVTKDatasetName(case_cfg, run_cfg, c_alpha_value);
  if (static_source)
  {
    const std::uint64_t fingerprint =
      ComputeVTKRestartFieldFingerprint(*static_source, ctx);
    std::ostringstream tag;
    tag << std::hex << std::setfill('0') << std::setw(16) << fingerprint;
    name += "_src_" + tag.str();
  }
  if (restart_info)
  {
    name += "_restart_c" + std::to_string(restart_info->cycle) + "_s" +
            std::to_string(restart_info->step);
  }
  int communicator_size = 1;
  int communicator_rank = 0;
  int world_size = 1;
  int world_rank = 0;
  const MPI_Comm communicator = ctx.Mesh().GetComm();
  MPI_Comm_size(communicator, &communicator_size);
  MPI_Comm_rank(communicator, &communicator_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  if (communicator_size != world_size)
  {
    int communicator_root_world_rank = communicator_rank == 0 ? world_rank : -1;
    MPI_Bcast(&communicator_root_world_rank,
              1,
              MPI_INT,
              0,
              communicator);
    name += "_commroot" + std::to_string(communicator_root_world_rank);
  }
  return name;
}

inline void VerifyDGSEMElementGeometry(const mfem::Mesh& mesh)
{
  for (int i = 0; i < mesh.GetNE(); ++i)
  {
    const mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(i);
    if (geom == mfem::Geometry::SEGMENT || geom == mfem::Geometry::SQUARE)
    {
      continue;
    }
    MFEM_VERIFY(false,
                "DGSEM v1 supports tensor-product meshes only (segment/quad). "
                "Found geometry: "
                  << mfem::Geometry::Name[geom] << ".");
  }
}

/**
 * Apply the resolved stagewise Zhang--Shu policy once to the fully initialized
 * Euler state.  This is deliberately separate from RK's legacy LimitInitial
 * option: the latter retains its existing per-step semantics, while this
 * transaction admits the projected or restarted state before output fields,
 * an instantaneous IGR closure, or a CFL/RHS evaluation can consume it.
 *
 * The limiter performs its collective admission preflight before committing
 * any local element update.  Consequently, a
 * RecoverableLimiterFailure leaves the conservative state unchanged on every
 * rank in the state's communicator.
 */
inline bool ApplyInitialEulerZhangShuTransaction(
  const RunConfig& run_cfg,
  EulerStateBundle& bundle,
  mfem::ZhangShuLimiterElementDiagnostics* diagnostics = nullptr)
{
  const mfem::LimiterPipelineConfig& limiter =
    run_cfg.case_config.limiter_pipeline;
  if (!limiter.zhang_shu)
  {
    return false;
  }

  mfem::ParFiniteElementSpace* vfes = bundle.sol.ParFESpace();
  MFEM_VERIFY(vfes != nullptr,
              "Initial Euler admissibility requires a parallel FE space.");
  MFEM_VERIFY(vfes->GetOrdering() == mfem::Ordering::byNODES,
              "Initial Euler admissibility requires byNODES ordering.");
  const int dim = bundle.sol.VectorDim() - 2;
  MFEM_VERIFY(dim > 0 && dim == vfes->GetParMesh()->Dimension(),
              "Initial Euler admissibility received an invalid state layout.");

  std::vector<int> face_quadrature_orders;
  if (run_cfg.dg_backend == DGBackend::DG)
  {
    const int order = vfes->GetFE(0)->GetOrder();
    face_quadrature_orders.push_back(
      2 * order + run_cfg.case_config.int_order_offset);
  }

  const mfem::RK3Config& rk = run_cfg.case_config.rk3_config;
  mfem::ApplyZhangShuPositivityLimiter(*vfes,
                                       bundle.sol,
                                       dim,
                                       rk.gamma,
                                       rk.rho_min,
                                       rk.p_min,
                                       rk.quad_order,
                                       face_quadrature_orders,
                                       rk.p_bisect_iters,
                                       limiter.zhang_shu_points,
                                       diagnostics,
                                       limiter.zhang_shu_vtk_points);

  int communicator_size = 1;
  MPI_Comm_size(vfes->GetComm(), &communicator_size);
  if (communicator_size > 1)
  {
    bundle.sol.ExchangeFaceNbrData();
  }
  return true;
}

inline void PublishInitialZhangShuDiagnostics(
  const mfem::ZhangShuLimiterElementDiagnostics& diagnostics,
  ZhangShuLimiterField& fields)
{
  MFEM_VERIFY(fields.active.Size() == diagnostics.active.Size() &&
                fields.theta_rho.Size() == diagnostics.theta_rho.Size() &&
                fields.theta_p.Size() == diagnostics.theta_p.Size(),
              "Initial Zhang-Shu diagnostics use an incompatible element "
              "space.");
  fields.active = diagnostics.active;
  fields.theta_rho = diagnostics.theta_rho;
  fields.theta_p = diagnostics.theta_p;
}

template <typename NumericalFlux, typename BcSetup, typename Func>
bool WithEulerOperator(ExampleContext& ctx,
                       double gamma,
                       dg_igr::IGRSolver* igr,
                       mfem::ParGridFunction& sol,
                       const mfem::Vector* static_source,
                       dg_igr::igr_internal::CompactSubcellRecovery* subcell_recovery,
                       const BoundaryMarkerMap& markers,
                       BcSetup&& bc_setup,
                       Func&& func)
{
  if (igr)
  {
    auto physics =
      MakeEulerIGRPhysics<NumericalFlux>(ctx,
                                         gamma,
                                         *igr,
                                         markers,
                                         std::forward<BcSetup>(bc_setup));
    if (static_source)
    {
      physics.op.SetStaticSource(*static_source);
    }
    if (subcell_recovery)
      physics.op.SetFaceIntegrationRule(mfem::IntRules.Get(mfem::Geometry::SEGMENT, 8));
    mfem::EulerWithIGR wrapped(physics.op, *igr, sol, subcell_recovery);
    return func(wrapped);
  }

  mfem::EulerFlux base_flux(ctx.dim, gamma);
  NumericalFlux numerical_flux(base_flux);
  ConfigureDGSEMSurfaceFlux(numerical_flux, ctx.run_cfg, gamma);
  mfem::DGHyperbolicConservationLaws op = MakeDGOperator(ctx, numerical_flux);
  if (static_source)
  {
    op.SetStaticSource(*static_source);
  }
  EulerBoundaryWiring<NumericalFlux> bc(
    op,
    numerical_flux,
    ctx.run_cfg.case_config.int_order_offset,
    ctx.dim,
    gamma);
  bc_setup(bc, markers);
  return func(op);
}

enum class SolverRequestAdmission
{
  Accepted,
  ResolutionFailure,
  Mismatch,
  CollectiveFailure
};

/// @brief Admit one identical, resolved request on every MPI rank.
inline SolverRequestAdmission
AdmitSolverRequest(const SolverRequestResolution& request)
{
  constexpr int tuple_size = 8;
  const int local[tuple_size] = {
    static_cast<int>(request.status),
    static_cast<int>(request.discretization),
    static_cast<int>(request.discretization_provenance),
    request.dg_backend ? static_cast<int>(*request.dg_backend) : -1,
    static_cast<int>(request.device_backend),
    static_cast<int>(request.hdg_cuda_engine),
    static_cast<int>(request.regularization),
    static_cast<int>(request.time_integrator)};
  int minimum[tuple_size] = {};
  int maximum[tuple_size] = {};
  const int minimum_status =
    MPI_Allreduce(local, minimum, tuple_size, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  const int maximum_status =
    MPI_Allreduce(local, maximum, tuple_size, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (minimum_status != MPI_SUCCESS || maximum_status != MPI_SUCCESS)
  {
    return SolverRequestAdmission::CollectiveFailure;
  }
  for (int i = 0; i < tuple_size; ++i)
  {
    if (minimum[i] != maximum[i])
    {
      return SolverRequestAdmission::Mismatch;
    }
  }
  return minimum[0] ==
           static_cast<int>(SolverRequestResolutionStatus::Resolved)
           ? SolverRequestAdmission::Accepted
           : SolverRequestAdmission::ResolutionFailure;
}

} // namespace detail

/// @brief Run Euler (with or without IGR) using the shared IO/time-loop wiring.
template <typename OperatorType, typename OnOpenCallback>
inline bool
RunEulerCase(ExampleContext& ctx,
             const CaseConfig& case_cfg,
             const RunConfig& run_cfg,
             EulerStateBundle& bundle,
             mfem::RK3SSPWithLimiter& rk3,
             OperatorType& op,
             bool& glvis_enabled,
             const TimeLoopCallbacks& callbacks,
             int precision,
             OnOpenCallback&& on_open_callback,
             RunSummary* summary_out = nullptr,
             double c_alpha_value = std::numeric_limits<double>::quiet_NaN(),
             bool print_dataset_name = false,
             const VTKRestartInfo* restart_info = nullptr,
             const mfem::ParGridFunction* static_source = nullptr)
{
  int num_procs = 1;
  int my_rank = 0;
  MPI_Comm_size(ctx.Mesh().GetComm(), &num_procs);
  MPI_Comm_rank(ctx.Mesh().GetComm(), &my_rank);
  const bool communicator_root = my_rank == 0;
  const mfem::real_t initial_time = restart_info ? restart_info->time : 0.0;
  MFEM_VERIFY(!restart_info ||
                restart_info->step <= std::numeric_limits<int>::max(),
              "Restart step exceeds the supported time-loop counter range.");
  const int initial_step = restart_info ? static_cast<int>(restart_info->step) : 0;
  const int initial_cycle = restart_info ? restart_info->cycle : 0;
  rk3.EnablePhysicalBoundaryFluxTracking(
    run_cfg.opts.conservation_balance);
  rk3.Init(op);
  MFEM_VERIFY(!run_cfg.opts.conservation_balance ||
                rk3.SupportsPhysicalBoundaryFluxTracking(),
              "Accepted-trajectory conservation balance requires the selected "
              "DG/DGSEM operator to provide exact physical-boundary flux "
              "capture.");
  rk3.SetStepCount(initial_step);
  std::optional<detail::EulerConservedIntegral> initial_conservation;
  std::optional<detail::EulerConservedIntegral> static_source_conservation;
  if (run_cfg.opts.conservation_balance)
  {
    initial_conservation =
      detail::IntegrateEulerConservedState(bundle.sol, run_cfg.dg_backend);
    if (static_source)
    {
      MFEM_VERIFY(static_source->ParFESpace() == bundle.sol.ParFESpace(),
                  "Conservation source must use the active Euler space.");
      static_source_conservation =
        detail::IntegrateEulerConservedState(*static_source,
                                             run_cfg.dg_backend);
      MFEM_VERIFY(static_source_conservation->components.Size() ==
                    initial_conservation->components.Size(),
                  "Static-source conservation component count mismatch.");
    }
  }
  auto* sigma_provider = dynamic_cast<mfem::IGRSigmaStateProvider*>(&op);
  const bool sigma_certification_supported =
    sigma_provider &&
    sigma_provider->SupportsIndependentResidualCertification();

  EulerOutputFieldTiming initial_field_timing;
  double initial_sigma_prepare_s = 0.0;
  mfem::IGRSigmaPreparationResult initial_sigma_preparation;
  if (sigma_provider &&
      (restart_info || run_cfg.opts.vtk_save || glvis_enabled))
  {
    const auto prepare_start = std::chrono::steady_clock::now();
    initial_sigma_preparation = sigma_provider->PrepareSigmaForOutput(
      bundle.sol,
      rk3.GetStateGeneration(),
      initial_step,
      initial_time,
      /*reusable_stage0=*/true,
      /*cold_start=*/restart_info != nullptr,
      /*certify_residual=*/restart_info != nullptr &&
        sigma_certification_supported);
    initial_sigma_prepare_s = SecondsSince(prepare_start);
  }
  if (restart_info && sigma_provider && communicator_root)
  {
    std::cout << "[vtk-restart] event=sigma-initialization"
              << " solved=" << (initial_sigma_preparation.solved ? 1 : 0)
              << " cold_start=1"
              << " reusable_stage0="
              << (initial_sigma_preparation.reusable_stage0 ? 1 : 0)
              << " certification_supported="
              << (sigma_certification_supported ? 1 : 0)
              << " independent_residual_certified="
              << (initial_sigma_preparation.independent_residual_certified ? 1
                                                                          : 0)
              << " independent_residual_rel_rhs="
              << initial_sigma_preparation.independent_residual_rel_rhs
              << "\n";
  }
  if (restart_info && sigma_provider && sigma_certification_supported &&
      !initial_sigma_preparation.independent_residual_certified)
  {
    TimeLoopResult failure;
    failure.steps = initial_step;
    failure.initial_time = initial_time;
    failure.final_time = initial_time;
    failure.success = false;
    failure.termination_reason =
      "igr-restart-closure-certificate-failed";
    PopulateExplicitCFLDiagnostics(
      failure, MakeTimeLoopConfig(run_cfg), rk3);
    if (communicator_root)
    {
      EmitExplicitTimeRunSummary(failure);
      std::cout << "[explicit-time] event=final-status"
                << " final_time=" << initial_time
                << " success=0"
                << " termination_reason=" << failure.termination_reason
                << "\n";
      if (summary_out)
      {
        int communicator_size = 1;
        MPI_Comm_size(ctx.Mesh().GetComm(), &communicator_size);
        *summary_out = BuildRunSummary(ctx,
                                       failure,
                                       /*elapsed_seconds=*/0.0,
                                       communicator_size,
                                       bundle.field_list.igr);
      }
    }
    return false;
  }

  // Output is a consumer of the accepted DAE state, not a prerequisite for
  // admitting it.  In particular, a restart certificate or an initial-output
  // prefetch may reconstruct sigma before RunTimeLoop performs its usual
  // initial q_sigma check.  Fail closed here so no VTK cycle or restart
  // manifest can publish a non-hyperbolic algebraic/conservative pair.
  std::uint64_t initial_q_sigma_nonpositive = 0;
  std::uint64_t initial_q_sigma_nonfinite = 0;
  const bool initial_q_sigma_rejected =
    time_loop_detail::AcceptedStateQSigmaAdmission<OperatorType>::Inspect(
      op,
      bundle.sol,
      initial_time,
      initial_q_sigma_nonpositive,
      initial_q_sigma_nonfinite);
  if (initial_q_sigma_rejected)
  {
    TimeLoopResult failure;
    failure.steps = initial_step;
    failure.initial_time = initial_time;
    failure.final_time = initial_time;
    failure.rejected_attempts = 1;
    failure.rejected_q_sigma_attempts = 1;
    failure.rejected_q_sigma_nonpositive_samples =
      initial_q_sigma_nonpositive;
    failure.rejected_q_sigma_nonfinite_samples = initial_q_sigma_nonfinite;
    const mfem::RKStageAdmissionFailureReason reason =
      initial_q_sigma_nonpositive > 0 && initial_q_sigma_nonfinite > 0
        ? mfem::RKStageAdmissionFailureReason::
            QSigmaNonpositiveAndNonfinite
        : (initial_q_sigma_nonfinite > 0
             ? mfem::RKStageAdmissionFailureReason::QSigmaNonfinite
             : mfem::RKStageAdmissionFailureReason::QSigmaNonpositive);
    failure.last_q_sigma_rejection_reason =
      mfem::RKStageAdmissionFailureReasonName(reason);
    failure.last_q_sigma_rejection_stage = -1;
    failure.success = false;
    failure.termination_reason = "initial-q-sigma-admission-failed";
    PopulateExplicitCFLDiagnostics(
      failure, MakeTimeLoopConfig(run_cfg), rk3);
    if (communicator_root)
    {
      std::cout << "[q-sigma-admission] event=initial-rejection"
                << " time=" << initial_time
                << " step=" << initial_step
                << " reason=" << failure.last_q_sigma_rejection_reason
                << " nonpositive_samples="
                << initial_q_sigma_nonpositive
                << " nonfinite_samples=" << initial_q_sigma_nonfinite
                << " rhs_evaluations=0 output_published=0\n";
      EmitExplicitTimeRunSummary(failure);
      std::cout << "[explicit-time] event=final-status"
                << " final_time=" << initial_time
                << " success=0 termination_reason="
                << failure.termination_reason << "\n";
    }
    if (summary_out)
    {
      *summary_out = BuildRunSummary(ctx,
                                     failure,
                                     /*elapsed_seconds=*/0.0,
                                     num_procs,
                                     bundle.field_list.igr);
    }
    return false;
  }
  UpdateEulerOutputFields(bundle.field_list, &initial_field_timing);
  initial_field_timing.igr_update_s = initial_sigma_prepare_s;
  initial_field_timing.total_s += initial_sigma_prepare_s;
  const double initial_update_fields_s = initial_field_timing.total_s;
  PrintVTKOutputTiming("initial-fields",
                       initial_cycle,
                       initial_step,
                       initial_time,
                       /*is_initial=*/restart_info == nullptr,
                       /*is_final=*/false,
                       initial_update_fields_s,
                       /*paraview_save_s=*/0.0,
                       ctx.Mesh().GetComm());
  PrintVTKOutputFieldTiming("initial-fields",
                            initial_cycle,
                            initial_step,
                            initial_time,
                            /*is_initial=*/restart_info == nullptr,
                            /*is_final=*/false,
                            initial_field_timing,
                            ctx.Mesh().GetComm());

  EulerVisWindows glvis_windows;
  if (glvis_enabled)
  {
    const bool local_ok = bundle.field_list.igr
                            ? OpenEulerIGRGLVisWindows(glvis_windows,
                                                       bundle.field_list,
                                                       ctx.Mesh(),
                                                       num_procs,
                                                       my_rank,
                                                       precision)
                            : OpenEulerGLVisWindows(glvis_windows,
                                                    bundle.field_list,
                                                    ctx.Mesh(),
                                                    num_procs,
                                                    my_rank,
                                                    precision);
    int local_ok_value = local_ok ? 1 : 0;
    int global_ok_value = 0;
    MPI_Allreduce(&local_ok_value,
                  &global_ok_value,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  ctx.Mesh().GetComm());
    if (global_ok_value != 0)
    {
      on_open_callback();
    }
    else
    {
      glvis_enabled = false;
      if (communicator_root)
      {
        std::cout << "Unable to connect to GLVis. Visualization disabled.\n";
      }
    }
  }

  const std::string dc_name = detail::MakeEulerOutputDatasetName(
    case_cfg, run_cfg, c_alpha_value, restart_info, ctx, static_source);
  auto paraview_dc =
    std::make_unique<mfem::ParaViewDataCollection>(dc_name, &ctx.Mesh());
  const mfem::io::VTKOutputConfig vtk_cfg = MakeCaseVTKConfig(case_cfg);
  const std::string dataset_dir =
    (std::filesystem::path(vtk_cfg.prefix_path) / dc_name).string();
  if (run_cfg.opts.vtk_save)
  {
    RegisterEulerVTKFields(*paraview_dc, bundle.field_list);
    int output_order = run_cfg.opts.order;
    if (::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_SUBCELL_RECOVERY", false))
    {
      MFEM_VERIFY(bundle.field_list.sigma && bundle.field_list.sigma->ParFESpace(),
                  "Subcell output requires its separately represented sigma field.");
      output_order = std::max({1,output_order,bundle.field_list.sigma->ParFESpace()->GetMaxElementOrder()});
    }
    ApplyCaseVTKConfig(*paraview_dc, vtk_cfg, output_order);
    if (!restart_info)
    {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      const auto initial_save_start = std::chrono::steady_clock::now();
      paraview_dc->Save();
      const double initial_save_s = SecondsSince(initial_save_start);
      WriteVTKRestartManifest(dataset_dir,
                              dc_name,
                              0,
                              0.0,
                              0,
                              run_cfg,
                              ctx,
                              bundle.field_list.sigma != nullptr,
                              paraview_dc->GetCompressionLevel(),
                              static_source);
      PrintVTKOutputTiming("initial-save",
                           /*cycle=*/0,
                           /*step=*/0,
                           /*t=*/0.0,
                           /*is_initial=*/true,
                           /*is_final=*/false,
                           /*update_fields_s=*/0.0,
                           initial_save_s,
                           ctx.Mesh().GetComm());
      if (callbacks.on_vtk_save)
      {
        VTKSaveInfo info;
        info.cycle = 0;
        info.step = 0;
        info.t = 0.0;
        info.is_initial = true;
        info.is_final = false;
        callbacks.on_vtk_save(info);
      }
    }
    if (communicator_root)
    {
      if (print_dataset_name)
      {
        std::cout << "VTK dataset: " << dc_name << "\n";
      }
    }
  }

  mfem::tic_toc.Clear();
  mfem::tic_toc.Start();

  TimeLoopIO io;
  io.glvis_windows = &glvis_windows;
  io.glvis_enabled = &glvis_enabled;
  io.paraview = paraview_dc.get();
  bool final_sigma_prepared = false;
  bool final_sigma_certification_skipped_nonfinite = false;
  bool final_sigma_transport_admission_failed = false;
  std::uint64_t final_q_sigma_nonpositive = 0;
  std::uint64_t final_q_sigma_nonfinite = 0;
  mfem::IGRSigmaPreparationResult final_sigma_preparation;
  if (run_cfg.opts.vtk_save)
  {
    io.after_vtk_save = [&run_cfg,
                         &ctx,
                         dataset_dir,
                         dc_name,
                         static_source,
                         sigma_saved = bundle.field_list.sigma != nullptr,
                         vtk_compression_level =
                           paraview_dc->GetCompressionLevel()](
                          int step, mfem::real_t t, int cycle)
    {
      WriteVTKRestartManifest(dataset_dir,
                              dc_name,
                              cycle,
                              t,
                              step,
                              run_cfg,
                              ctx,
                              sigma_saved,
                              vtk_compression_level,
                              static_source);
    };
  }
  if (sigma_provider)
  {
    io.prepare_sigma = [sigma_provider,
                        &bundle,
                        &rk3,
                        &op,
                        sigma_certification_supported,
                        &final_sigma_prepared,
                        &final_sigma_certification_skipped_nonfinite,
                        &final_sigma_transport_admission_failed,
                        &final_q_sigma_nonpositive,
                        &final_q_sigma_nonfinite,
                        &final_sigma_preparation](int,
                                                  mfem::real_t t,
                                                  bool is_final)
    {
      if (is_final && final_sigma_prepared)
      {
        return SigmaOutputPreparation{
          0.0,
          !final_sigma_transport_admission_failed &&
            (!sigma_certification_supported ||
             final_sigma_preparation.independent_residual_certified)};
      }
      if (is_final && final_sigma_certification_skipped_nonfinite)
      {
        return SigmaOutputPreparation{0.0, false};
      }
      if (is_final)
      {
        const mfem::dg_igr::NonFiniteSummary final_nonfinite =
          mfem::dg_igr::ScanNonFinite(bundle.sol);
        if (final_nonfinite.global_non_finite != 0)
        {
          final_sigma_certification_skipped_nonfinite = true;
          return SigmaOutputPreparation{0.0, false};
        }
      }
      const auto prepare_start = std::chrono::steady_clock::now();
      const mfem::IGRSigmaPreparationResult preparation =
        sigma_provider->PrepareSigmaForOutput(
          bundle.sol,
          rk3.GetStateGeneration(),
          static_cast<int>(rk3.GetStepCount()),
          t,
          /*reusable_stage0=*/!is_final,
          /*cold_start=*/false,
          /*certify_residual=*/is_final && sigma_certification_supported);
      if (is_final)
      {
        final_sigma_preparation = preparation;
        final_sigma_prepared = true;
        final_sigma_transport_admission_failed =
          time_loop_detail::
            AcceptedStateQSigmaAdmission<OperatorType>::Inspect(
              op,
              bundle.sol,
              t,
              final_q_sigma_nonpositive,
              final_q_sigma_nonfinite);
      }
      return SigmaOutputPreparation{
        SecondsSince(prepare_start),
        !is_final ||
          (!final_sigma_transport_admission_failed &&
           (!sigma_certification_supported ||
            preparation.independent_residual_certified))};
    };
  }

  TimeLoopConfig loop_cfg = MakeTimeLoopConfig(run_cfg);
  if (run_cfg.opts.vtk_save)
  {
    loop_cfg.save_index_offset = restart_info ? restart_info->cycle + 1 : 1;
  }
  loop_cfg.initial_time = initial_time;
  loop_cfg.initial_step = initial_step;
  loop_cfg.initial_state_saved = run_cfg.opts.vtk_save && !restart_info;
  loop_cfg.initial_q_sigma_admitted =
    sigma_provider &&
    ResolveEffectiveIGRSigmaUpdateCadence(run_cfg) ==
      dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep;
  loop_cfg.next_save_time =
    (run_cfg.opts.vtk_save && run_cfg.opts.dt_save > 0.0)
      ? initial_time + run_cfg.opts.dt_save
      : -1.0;
  const TimeLoopHooks hooks =
    MakeEulerTimeLoopHooks(ctx,
                           bundle,
                           num_procs,
                           my_rank,
                           io,
                           callbacks,
                           initial_step);
  TimeLoopResult result =
    RunTimeLoop(op, ctx.Mesh(), bundle.sol, rk3, loop_cfg, hooks);
  std::optional<detail::EulerConservedIntegral> final_conservation;
  mfem::Vector accepted_boundary_flux;
  mfem::Vector accepted_boundary_flux_scale;
  mfem::Vector integrated_static_source;
  mfem::Vector time_integrated_static_source;
  mfem::Vector conservation_defect;
  mfem::Vector conservation_relative_defect;
  double maximum_conservation_absolute_error = 0.0;
  double maximum_conservation_relative_error = 0.0;
  if (run_cfg.opts.conservation_balance)
  {
    MFEM_VERIFY(initial_conservation.has_value(),
                "Conservation balance is missing its initial integral.");
    final_conservation =
      detail::IntegrateEulerConservedState(bundle.sol, run_cfg.dg_backend);
    const int components = initial_conservation->components.Size();
    MFEM_VERIFY(final_conservation->components.Size() == components,
                "Euler conservation component count changed during the run.");
    accepted_boundary_flux.SetSize(components);
    accepted_boundary_flux_scale.SetSize(components);
    integrated_static_source.SetSize(components);
    time_integrated_static_source.SetSize(components);
    accepted_boundary_flux = 0.0;
    accepted_boundary_flux_scale = 0.0;
    integrated_static_source = 0.0;
    time_integrated_static_source = 0.0;
    if (rk3.HasCumulativePhysicalBoundaryFluxIntegral())
    {
      accepted_boundary_flux =
        rk3.GetCumulativePhysicalBoundaryFluxIntegral();
      accepted_boundary_flux_scale =
        rk3.GetCumulativePhysicalBoundaryFluxAbsoluteIntegral();
    }
    if (static_source_conservation)
    {
      integrated_static_source = static_source_conservation->components;
    }
    const double elapsed_physical_time = result.final_time - initial_time;
    MFEM_VERIFY(std::isfinite(elapsed_physical_time) &&
                  elapsed_physical_time >= 0.0,
                "Conservation balance has invalid elapsed physical time.");
    time_integrated_static_source = integrated_static_source;
    time_integrated_static_source *= elapsed_physical_time;
    conservation_defect.SetSize(components);
    conservation_relative_defect.SetSize(components);
    for (int component = 0; component < components; ++component)
    {
      const double state_change =
        final_conservation->components(component) -
        initial_conservation->components(component);
      conservation_defect(component) =
        state_change + accepted_boundary_flux(component) -
        time_integrated_static_source(component);
      const double balance_scale =
        std::abs(state_change) + accepted_boundary_flux_scale(component) +
        std::abs(time_integrated_static_source(component));
      conservation_relative_defect(component) =
        detail::EulerConservationRelativeError(
          std::abs(conservation_defect(component)),
          balance_scale,
          initial_conservation->domain_measure);
      maximum_conservation_absolute_error =
        std::max(maximum_conservation_absolute_error,
                 std::abs(conservation_defect(component)));
      maximum_conservation_relative_error =
        std::max(maximum_conservation_relative_error,
                 conservation_relative_defect(component));
    }
  }
  // Reconstruct and, when supported, independently certify every finite final
  // committed algebraic state. A terminal rejection has already rolled back
  // the conservative state and IGR cadence provenance; certification must not
  // depend on whether a final VTK hook ran. Never replace a typed nonfinite
  // transport failure with an algebraic-solver abort.
  bool final_state_finite = true;
  if (sigma_provider && !final_sigma_prepared &&
      !final_sigma_certification_skipped_nonfinite)
  {
    const mfem::dg_igr::NonFiniteSummary final_nonfinite =
      mfem::dg_igr::ScanNonFinite(bundle.sol);
    final_state_finite = final_nonfinite.global_non_finite == 0;
  }
  if (sigma_provider && !final_sigma_prepared && !final_state_finite)
  {
    final_sigma_certification_skipped_nonfinite = true;
  }
  if (sigma_provider && !final_sigma_prepared && final_state_finite &&
      !final_sigma_certification_skipped_nonfinite)
  {
    final_sigma_preparation = sigma_provider->PrepareSigmaForOutput(
      bundle.sol,
      rk3.GetStateGeneration(),
      static_cast<int>(rk3.GetStepCount()),
      result.final_time,
      /*reusable_stage0=*/false,
      /*cold_start=*/false,
      /*certify_residual=*/sigma_certification_supported);
    final_sigma_prepared = true;
    final_sigma_transport_admission_failed =
      time_loop_detail::AcceptedStateQSigmaAdmission<OperatorType>::Inspect(
        op,
        bundle.sol,
        result.final_time,
        final_q_sigma_nonpositive,
        final_q_sigma_nonfinite);
  }
  if (final_sigma_certification_skipped_nonfinite && result.success)
  {
    result.success = false;
    result.termination_reason = "nonfinite-final-state";
  }
  if (final_sigma_transport_admission_failed && result.success)
  {
    result.success = false;
    result.termination_reason = "igr-final-q-sigma-admission-failed";
  }
  if (sigma_provider &&
      (final_sigma_prepared || final_sigma_certification_skipped_nonfinite))
  {
    const bool certificate_passed =
      final_sigma_prepared &&
      !final_sigma_transport_admission_failed &&
      (!sigma_certification_supported ||
       final_sigma_preparation.independent_residual_certified);
    if (result.success && !certificate_passed)
    {
      result.success = false;
      result.termination_reason = "igr-closure-certificate-failed";
    }
    if (communicator_root)
    {
      std::cout << std::setprecision(17)
                << "[igr-closure-certificate] event=accepted-state"
                << " scope=final-committed-accepted-state"
                << " precision=fp64"
                << " time=" << result.final_time
                << " state_generation=" << rk3.GetStateGeneration()
                << " solved=" << (final_sigma_preparation.solved ? 1 : 0)
                << " supported="
                << (sigma_certification_supported ? 1 : 0)
                << " certified="
                << (final_sigma_preparation.independent_residual_certified ? 1
                                                                          : 0)
                << " residual_rel_rhs="
                << final_sigma_preparation.independent_residual_rel_rhs
                << " run_success=" << (result.success ? 1 : 0)
                << " residual_abs="
                << final_sigma_preparation.independent_residual_abs
                << " relative_limit="
                << final_sigma_preparation.independent_residual_relative_limit
                << " absolute_limit="
                << final_sigma_preparation.independent_residual_absolute_limit
                << " skipped_nonfinite="
                << (final_sigma_certification_skipped_nonfinite ? 1 : 0)
                << " q_sigma_admitted="
                << (final_sigma_transport_admission_failed ? 0 : 1)
                << " q_sigma_nonpositive_samples="
                << final_q_sigma_nonpositive
                << " q_sigma_nonfinite_samples="
                << final_q_sigma_nonfinite
                << "\n";
    }
  }
  if (final_sigma_transport_admission_failed && communicator_root)
  {
    std::cout << "[q-sigma-admission] event=final-certificate-rejection"
              << " time=" << result.final_time
              << " step=" << result.steps
              << " nonpositive_samples=" << final_q_sigma_nonpositive
              << " nonfinite_samples=" << final_q_sigma_nonfinite
              << " output_published=0\n";
  }
  if (run_cfg.opts.conservation_balance && communicator_root)
  {
    MFEM_VERIFY(initial_conservation.has_value() &&
                  final_conservation.has_value(),
                "Conservation balance is missing endpoint integrals.");
    std::cout << std::setprecision(17)
              << "[explicit-conservation] event=accepted-trajectory"
              << " supported=1"
              << " source_supported=1"
              << " components=" << initial_conservation->components.Size()
              << " domain_measure=" << initial_conservation->domain_measure
              << " initial="
              << detail::CompactVector(initial_conservation->components)
              << " final="
              << detail::CompactVector(final_conservation->components)
              << " boundary_flux="
              << detail::CompactVector(accepted_boundary_flux)
              << " boundary_flux_scale="
              << detail::CompactVector(accepted_boundary_flux_scale)
              << " elapsed_physical_time="
              << result.final_time - initial_time
              << " source_integral="
              << detail::CompactVector(integrated_static_source)
              << " source_time_integral="
              << detail::CompactVector(time_integrated_static_source)
              << " defect=" << detail::CompactVector(conservation_defect)
              << " relative_defect="
              << detail::CompactVector(conservation_relative_defect)
              << " maximum_absolute_error="
              << maximum_conservation_absolute_error
              << " maximum_relative_error="
              << maximum_conservation_relative_error
              << " accepted_attempts=" << result.accepted_attempts
              << " rejected_attempts=" << result.rejected_attempts
              << " run_success=" << (result.success ? 1 : 0) << "\n";
  }
  if (communicator_root)
  {
    std::cout << "[explicit-time] event=final-status"
              << " final_time=" << result.final_time
              << " success=" << (result.success ? 1 : 0)
              << " termination_reason=" << result.termination_reason << "\n";
  }
  double maximum_sigma_endpoint_prediction_defect = 0.0;
  double maximum_relative_sigma_endpoint_prediction_defect = 0.0;
  double maximum_sigma_exact_certificate_residual_rel_rhs = 0.0;
  double maximum_sigma_exact_certificate_relative_limit = 0.0;
  double maximum_sigma_exact_certificate_residual_abs = 0.0;
  double maximum_sigma_exact_certificate_absolute_limit = 0.0;
  double maximum_sigma_exact_certificate_acceptance_ratio = 0.0;
  double maximum_sigma_exact_certificate_relative_pass_residual_rel_rhs = 0.0;
  double maximum_sigma_exact_certificate_absolute_only_residual_abs = 0.0;
  if (sigma_provider)
  {
    maximum_sigma_endpoint_prediction_defect =
      sigma_provider->GetMaximumLocalSigmaEndpointPredictionDefect();
    maximum_relative_sigma_endpoint_prediction_defect =
      sigma_provider->GetMaximumLocalRelativeSigmaEndpointPredictionDefect();
    maximum_sigma_exact_certificate_residual_rel_rhs =
      sigma_provider->GetMaximumLocalSigmaExactCertificateResidualRelRHS();
    maximum_sigma_exact_certificate_relative_limit =
      sigma_provider->GetMaximumLocalSigmaExactCertificateRelativeLimit();
    maximum_sigma_exact_certificate_residual_abs =
      sigma_provider->GetMaximumLocalSigmaExactCertificateResidualAbs();
    maximum_sigma_exact_certificate_absolute_limit =
      sigma_provider->GetMaximumLocalSigmaExactCertificateAbsoluteLimit();
    maximum_sigma_exact_certificate_acceptance_ratio =
      sigma_provider->GetMaximumLocalSigmaExactCertificateAcceptanceRatio();
    maximum_sigma_exact_certificate_relative_pass_residual_rel_rhs =
      sigma_provider
        ->GetMaximumLocalSigmaExactCertificateRelativePassResidualRelRHS();
    maximum_sigma_exact_certificate_absolute_only_residual_abs =
      sigma_provider
        ->GetMaximumLocalSigmaExactCertificateAbsoluteOnlyResidualAbs();
    MPI_Allreduce(MPI_IN_PLACE,
                  &maximum_sigma_endpoint_prediction_defect,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  ctx.Mesh().GetComm());
    MPI_Allreduce(MPI_IN_PLACE,
                  &maximum_relative_sigma_endpoint_prediction_defect,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  ctx.Mesh().GetComm());
    MPI_Allreduce(MPI_IN_PLACE,
                  &maximum_sigma_exact_certificate_residual_rel_rhs,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  ctx.Mesh().GetComm());
    MPI_Allreduce(MPI_IN_PLACE,
                  &maximum_sigma_exact_certificate_relative_limit,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  ctx.Mesh().GetComm());
    MPI_Allreduce(MPI_IN_PLACE,
                  &maximum_sigma_exact_certificate_residual_abs,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  ctx.Mesh().GetComm());
    MPI_Allreduce(MPI_IN_PLACE,
                  &maximum_sigma_exact_certificate_absolute_limit,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  ctx.Mesh().GetComm());
    MPI_Allreduce(MPI_IN_PLACE,
                  &maximum_sigma_exact_certificate_acceptance_ratio,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  ctx.Mesh().GetComm());
    MPI_Allreduce(
      MPI_IN_PLACE,
      &maximum_sigma_exact_certificate_relative_pass_residual_rel_rhs,
      1,
      MPI_DOUBLE,
      MPI_MAX,
      ctx.Mesh().GetComm());
    MPI_Allreduce(MPI_IN_PLACE,
                  &maximum_sigma_exact_certificate_absolute_only_residual_abs,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  ctx.Mesh().GetComm());
  }
  if (sigma_provider && communicator_root)
  {
    std::cout << std::setprecision(17)
              << "[igr-sigma-lifecycle] event=run-summary"
              << " prefetch_solves="
              << sigma_provider->GetSigmaPrefetchSolveCount()
              << " stage0_reuses="
              << sigma_provider->GetSigmaStage0ReuseCount()
              << " output_only_solves="
              << sigma_provider->GetSigmaOutputOnlySolveCount()
              << " prefetch_invalidations="
              << sigma_provider->GetSigmaPrefetchInvalidationCount()
              << " endpoint_solves="
              << sigma_provider->GetSigmaEndpointSolveCount()
              << " extrapolated_stages="
              << sigma_provider->GetSigmaExtrapolatedStageCount()
              << " exact_anchor_certificates="
              << sigma_provider->GetSigmaExactCertificateCount()
              << " exact_anchor_absolute_only_passes="
              << sigma_provider->GetSigmaExactCertificateAbsoluteOnlyPassCount()
              << " max_exact_anchor_certificate_residual_rel_rhs="
              << maximum_sigma_exact_certificate_residual_rel_rhs
              << " max_exact_anchor_certificate_relative_limit="
              << maximum_sigma_exact_certificate_relative_limit
              << " max_exact_anchor_certificate_residual_abs="
              << maximum_sigma_exact_certificate_residual_abs
              << " max_exact_anchor_certificate_absolute_limit="
              << maximum_sigma_exact_certificate_absolute_limit
              << " max_exact_anchor_certificate_acceptance_ratio="
              << maximum_sigma_exact_certificate_acceptance_ratio
              << " max_exact_anchor_relative_pass_residual_rel_rhs="
              << maximum_sigma_exact_certificate_relative_pass_residual_rel_rhs
              << " max_exact_anchor_absolute_only_residual_abs="
              << maximum_sigma_exact_certificate_absolute_only_residual_abs
              << " max_endpoint_prediction_defect="
              << maximum_sigma_endpoint_prediction_defect
              << " max_relative_endpoint_prediction_defect="
              << maximum_relative_sigma_endpoint_prediction_defect
              << "\n";
  }
  const bool report_extrapolated_q_sigma =
    sigma_provider &&
    ResolveEffectiveIGRSigmaUpdateCadence(run_cfg) ==
      dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep;
  const bool report_q_sigma =
    run_cfg.igr_opts.igr_invariant_domain_flux ||
    report_extrapolated_q_sigma;
  if (report_q_sigma)
  {
    const auto* q_sigma_provider =
      dynamic_cast<const mfem::IGRFrozenSigmaTelemetryProvider*>(&op);
    MFEM_VERIFY(q_sigma_provider,
                "Selected IGR transport is missing q_sigma "
                "telemetry.");
    const mfem::IGRFrozenSigmaTelemetry q_sigma =
      q_sigma_provider->GetGlobalFrozenSigmaTelemetry();
    MFEM_VERIFY(q_sigma.enabled,
                "Selected IGR q_sigma telemetry was not enabled.");
    if (communicator_root)
    {
      std::cout << std::setprecision(17)
                << "[igr-q-sigma] event=run-summary"
                << " scope="
                << (report_extrapolated_q_sigma
                      ? "application-total-attempted-frozen-sigma-work"
                      : "application-total-attempted-speed-work")
                << " rejected_attempts_included=1"
                << " precision=fp64"
                << " sample_count=" << q_sigma.sample_count
                << " nonpositive_count=" << q_sigma.nonpositive_count
                << " nonfinite_count=" << q_sigma.nonfinite_count
                << " minimum_raw_q_sigma="
                << q_sigma.minimum_raw_q_sigma
                << " minimum_q_sigma_over_gamma_p="
                << q_sigma.minimum_q_sigma_over_gamma_p << "\n";
    }
  }
  if (!result.success)
  {
    mfem::tic_toc.Stop();
    if (communicator_root && summary_out)
    {
      *summary_out = BuildRunSummary(ctx,
                                     result,
                                     mfem::tic_toc.RealTime(),
                                     num_procs,
                                     bundle.field_list.igr);
    }
    return false;
  }

  mfem::tic_toc.Stop();
  if (communicator_root)
  {
    const double elapsed = mfem::tic_toc.RealTime();
    std::cout << " done, " << elapsed << "s." << std::endl;
    if (callbacks.on_summary)
    {
      const RunSummary summary =
        BuildRunSummary(ctx, result, elapsed, num_procs, bundle.field_list.igr);
      if (summary_out)
      {
        *summary_out = summary;
      }
      callbacks.on_summary(summary);
    }
    else if (summary_out)
    {
      *summary_out =
        BuildRunSummary(ctx, result, elapsed, num_procs, bundle.field_list.igr);
    }
  }
  return true;
}

template <typename OperatorType>
inline bool
RunEulerCase(ExampleContext& ctx,
             const CaseConfig& case_cfg,
             const RunConfig& run_cfg,
             EulerStateBundle& bundle,
             mfem::RK3SSPWithLimiter& rk3,
             OperatorType& op,
             bool& glvis_enabled,
             const TimeLoopCallbacks& callbacks,
             int precision,
             RunSummary* summary_out = nullptr,
             double c_alpha_value = std::numeric_limits<double>::quiet_NaN(),
             bool print_dataset_name = false,
             const VTKRestartInfo* restart_info = nullptr,
             const mfem::ParGridFunction* static_source = nullptr)
{
  return RunEulerCase(
    ctx,
    case_cfg,
    run_cfg,
    bundle,
    rk3,
    op,
    glvis_enabled,
    callbacks,
    precision,
    []() {},
    summary_out,
    c_alpha_value,
    print_dataset_name,
    restart_info,
    static_source);
}

struct Solver
{
  template <typename CaseOptions, typename NumericalFlux>
  static int
  solve(const CaseSpec<CaseOptions, NumericalFlux>& spec, int argc, char** argv)
  {
    mfem::Mpi::Init(argc, argv);
    mfem::Hypre::Init();

    CaseConfig case_cfg = MakeCaseConfig(spec.case_name,
                                         spec.default_mesh_path,
                                         spec.case_file ? spec.case_file : "",
                                         spec.gamma);
    // Apply CaseSpec defaults first; configure_case can then override them.
    case_cfg.defaults.glvis = spec.glvis.enabled;
    case_cfg.defaults.vtk_save = spec.vtk.enabled;
    case_cfg.defaults.dg_backend = DGBackendToString(spec.dg_backend_default);
    if (spec.configure_case)
    {
      spec.configure_case(case_cfg);
    }
    case_cfg.rk3_config.gamma = case_cfg.gamma;

    CaseOptions case_opts = spec.options;
    CommonOptions cli_opts = case_cfg.defaults;
    IGROptions cli_igr_opts = spec.igr_defaults;

    mfem::OptionsParser args(argc, argv);
    const std::string mesh_help =
      std::string("MFEM mesh file to use. Defaults to ") +
      DefaultCaseMeshPath(case_cfg) + ".";
    AddCommonOptions(args, cli_opts, mesh_help);
    AddIGROptions(args, cli_igr_opts);
    if (spec.add_options)
    {
      spec.add_options(args, case_opts);
    }
    args.ParseCheck();
    const SolverRequestResolution request_resolution = ResolveSolverRequest(
      case_cfg.defaults.dg_backend,
      spec.use_igr,
      cli_opts.discretization,
      cli_opts.dg_backend,
      cli_opts.regularization,
      cli_opts.time_integrator,
      HasDiscretizationOptionOverride(argc, argv),
      HasDGBackendOptionOverride(argc, argv),
      HasRegularizationOptionOverride(argc, argv),
      HasTimeIntegratorOptionOverride(argc, argv),
      cli_opts.device,
      cli_opts.hdg_cuda_engine);
    const detail::SolverRequestAdmission request_admission =
      detail::AdmitSolverRequest(request_resolution);
    if (request_admission != detail::SolverRequestAdmission::Accepted)
    {
      if (mfem::Mpi::Root())
      {
        if (request_admission ==
            detail::SolverRequestAdmission::ResolutionFailure)
        {
          std::cerr << request_resolution.message << "\n";
        }
        else if (request_admission == detail::SolverRequestAdmission::Mismatch)
        {
          std::cerr << "Solver request differs across MPI ranks; refusing "
                       "runtime construction.\n";
        }
        else
        {
          std::cerr << "Unable to collectively admit the solver request.\n";
        }
      }
      return 1;
    }
    cli_opts.discretization =
      DiscretizationToString(request_resolution.discretization);
    cli_opts.device = DeviceBackendToString(request_resolution.device_backend);
    cli_opts.hdg_cuda_engine =
      HdgCudaEngineToString(request_resolution.hdg_cuda_engine);
    cli_opts.regularization =
      RegularizationToString(request_resolution.regularization);
    cli_opts.time_integrator =
      TimeIntegratorToString(request_resolution.time_integrator);
    MFEM_VERIFY(!cli_opts.restart_vtk.empty() ||
                  (cli_opts.restart_cycle < 0 && cli_opts.restart_step < 0),
                "--restart-cycle/--restart-step require --restart-vtk.");
    if (cli_igr_opts.igr_invariant_domain_flux)
    {
      std::string rejection;
      if (request_resolution.regularization != Regularization::IGR)
      {
        rejection = "--igr-invariant-domain-flux requires active IGR.";
      }
      else if (request_resolution.discretization != Discretization::DG)
      {
        rejection = "--igr-invariant-domain-flux currently requires the "
                    "ordinary DG backend.";
      }
      else if (ParseDGSEMSurfaceFlux(cli_opts.dgsem_surface_flux) !=
               DGSEMSurfaceFlux::Rusanov)
      {
        rejection = "--igr-invariant-domain-flux requires the standard "
                    "Rusanov surface flux.";
      }
      else if (!ResolveZhangShuPositivityLimiter(
                 cli_opts.positivity_limiter,
                 case_cfg.limiter_pipeline.zhang_shu))
      {
        rejection = "--igr-invariant-domain-flux requires the stagewise "
                    "Zhang-Shu limiter.";
      }
      else if (ParseZhangShuPointSet(cli_opts.zhang_shu_points) !=
               mfem::ZhangShuPointSet::kConsumer)
      {
        rejection = "--igr-invariant-domain-flux requires "
                    "--zhang-shu-points consumer so every ordinary-DG "
                    "volume node and face quadrature point is admitted.";
      }
      if (!rejection.empty())
      {
        if (mfem::Mpi::Root())
        {
          std::cerr << rejection << '\n';
        }
        return 1;
      }
    }
    const std::string modal_damping_rejection =
      ValidateModalDampingSelection(request_resolution.discretization,
                                    request_resolution.regularization,
                                    case_cfg,
                                    cli_opts,
                                    cli_igr_opts);
    if (!modal_damping_rejection.empty())
    {
      if (mfem::Mpi::Root())
      {
        std::cerr << modal_damping_rejection << '\n';
      }
      return 1;
    }
    if (request_resolution.discretization == Discretization::HDG)
    {
      const std::string hdg_cadence_rejection =
        ValidateHdgIGRSigmaCadence(request_resolution.discretization,
                                   request_resolution.regularization,
                                   case_cfg,
                                   cli_igr_opts);
      if (!hdg_cadence_rejection.empty())
      {
        if (mfem::Mpi::Root())
        {
          std::cerr << hdg_cadence_rejection << '\n';
        }
        return 1;
      }
#if defined(DG_IGR_HAS_HDG_MAIN_BACKEND) && DG_IGR_HAS_HDG_MAIN_BACKEND
      const bool c_alpha_overridden = HasCAlphaOptionOverride(argc, argv);
      const bool hdg_igr_alpha_overridden =
        HasHdgIgrAlphaOptionOverride(argc, argv);
      if (c_alpha_overridden && hdg_igr_alpha_overridden &&
          cli_igr_opts.c_alpha_user != cli_opts.hdg_igr_alpha)
      {
        if (mfem::Mpi::Root())
        {
          std::cerr << "Conflicting HDG IGR C_alpha values: --c-alpha="
                    << cli_igr_opts.c_alpha_user
                    << " and --hdg-igr-alpha=" << cli_opts.hdg_igr_alpha
                    << ". Use one spelling or supply the same value to both.\n";
        }
        return 1;
      }
      if (c_alpha_overridden)
      {
        cli_opts.hdg_igr_alpha = cli_igr_opts.c_alpha_user;
      }
      MFEM_VERIFY(static_cast<bool>(spec.make_hdg_case_data),
                  "Case '" << case_cfg.case_name
                            << "' does not provide an HDG backend adapter.");
      if (!spec.make_hdg_case_data)
      {
        return 1;
      }
      const hdg::HdgEulerCaseDataFactory make_hdg_case_data =
        [&spec, &case_opts, &case_cfg, &cli_opts](const CaseContext& context)
      {
        return spec.make_hdg_case_data(case_opts, case_cfg, cli_opts, context);
      };
      const std::optional<dg_igr::IGRAlphaLengthScalePolicy>
        hdg_alpha_length_policy_override =
          cli_igr_opts.alpha_length_policy.empty()
            ? std::nullopt
            : std::optional<dg_igr::IGRAlphaLengthScalePolicy>{
                ParseIGRAlphaLengthScalePolicy(
                  cli_igr_opts.alpha_length_policy)};
      const hdg::HdgEulerMainRequest hdg_request{
        case_cfg,
        cli_opts,
        make_hdg_case_data,
        MPI_COMM_WORLD,
        hdg_alpha_length_policy_override};
#if defined(MFEM_USE_CUDA)
      // The CUDA HDG adapter returns before the legacy device construction
      // below.  Keep this device alive for its entire generic backend call so
      // it sees the same selected MFEM device lifetime as every other CUDA
      // execution route, without changing CPU/DG/DGSEM.
      if (request_resolution.device_backend == DeviceBackend::CUDA)
      {
        mfem::Device hdg_cuda_device("cuda");
        if (mfem::Mpi::Root())
        {
          std::cout << "[device] requested=cuda hdg_cuda_engine="
                    << cli_opts.hdg_cuda_engine << '\n';
          hdg_cuda_device.Print();
        }
        return hdg::RunEulerMainBackend(hdg_request);
      }
#endif
      return hdg::RunEulerMainBackend(hdg_request);
#else
      if (mfem::Mpi::Root())
      {
        std::cerr << "Case '" << case_cfg.case_name
                  << "' does not provide an HDG backend adapter.\n";
      }
      return 1;
#endif
    }

    MFEM_VERIFY(request_resolution.dg_backend,
                "DG/DGSEM selection requires a legacy DG backend mapping.");
    cli_opts.dg_backend =
      DGBackendToString(*request_resolution.dg_backend);
    ApplyIGRDerivedRuntimeDefaults(cli_opts,
                                   cli_igr_opts,
                                   HasExactDGFaceOptionOverride(argc, argv));

    const RunConfig run_cfg = MakeRunConfig(
      case_cfg,
      cli_opts,
      cli_igr_opts,
      request_resolution.discretization_provenance,
      request_resolution.regularization,
      request_resolution.time_integrator);
    const std::string cfl_selection_rejection =
      ValidateExplicitSSPRKCFLSelection(run_cfg);
    if (!cfl_selection_rejection.empty())
    {
      if (mfem::Mpi::Root())
      {
        std::cerr << cfl_selection_rejection << '\n';
      }
      return 1;
    }
    const std::string extrapolated_selection_rejection =
      ValidateExtrapolatedEveryStepIGRSelection(run_cfg);
    if (!extrapolated_selection_rejection.empty())
    {
      if (mfem::Mpi::Root())
      {
        std::cerr << extrapolated_selection_rejection << '\n';
      }
      return 1;
    }
    const bool use_igr = run_cfg.regularization == Regularization::IGR;
    if (use_igr && mfem::Mpi::Root())
    {
      std::cout << "[igr-options] event=effective"
                << " exact_dg_face="
                << (run_cfg.igr_opts.exact_dg_face_operator ? "on" : "off")
                << " solver=" << run_cfg.igr_opts.linear_solver
                << " preconditioner=" << run_cfg.igr_opts.preconditioner
                << " nonconvergence_policy="
                << run_cfg.igr_opts.nonconvergence_policy
                << " sigma_cfl="
                << (run_cfg.igr_opts.sigma_aware_cfl ? "on" : "off")
                << " sigma_rusanov="
                << (run_cfg.igr_opts.sigma_aware_rusanov ? "on" : "off")
                << " frozen_sigma_spectral_speed="
                << (UsesFrozenSigmaSpectralSpeed(run_cfg) ? "on" : "off")
                << " invariant_domain_flux="
                << (run_cfg.igr_opts.igr_invariant_domain_flux ? "on" : "off")
                << " stage_rtol_scale="
                << (run_cfg.igr_opts.stage_rtol_scale >= 0.0
                      ? run_cfg.igr_opts.stage_rtol_scale
                      : run_cfg.case_config.igr_config.stage_rtol_scale)
                << " lin_rtol="
                << (run_cfg.igr_opts.lin_rtol >= 0.0
                      ? run_cfg.igr_opts.lin_rtol
                      : run_cfg.case_config.igr_config.lin_rtol)
                << " lin_atol="
                << (run_cfg.igr_opts.lin_atol >= 0.0
                      ? run_cfg.igr_opts.lin_atol
                      : run_cfg.case_config.igr_config.lin_atol)
                << " lin_maxit="
                << (run_cfg.igr_opts.lin_maxit >= 0
                      ? run_cfg.igr_opts.lin_maxit
                      : run_cfg.case_config.igr_config.lin_maxit)
                << "\n";
    }
    if (run_cfg.device_backend == DeviceBackend::CUDA)
    {
#ifndef MFEM_USE_CUDA
      MFEM_VERIFY(false,
                  "Requested '--device cuda' but MFEM was built without CUDA "
                  "support.");
#endif
    }
    mfem::Device device(DeviceBackendToString(run_cfg.device_backend));
    if (mfem::Mpi::Root())
    {
      std::cout << "[device] requested="
                << DeviceBackendToString(run_cfg.device_backend) << "\n";
      device.Print();
    }
    bool glvis_enabled = run_cfg.opts.glvis;
    std::cout.precision(spec.glvis.precision);

    const bool need_sigma_space = use_igr || spec.need_sigma;
    const bool enable_h1_vis_diag =
      spec.diagnostics.h1_diagnostics &&
      (run_cfg.opts.glvis ||
       (run_cfg.opts.vtk_save && run_cfg.opts.vtk_h1_diagnostics));
    const bool enable_l2_diag =
      spec.diagnostics.l2_diagnostics && run_cfg.opts.vtk_save &&
      run_cfg.opts.vtk_l2_diagnostics;
    const bool enable_mpi_rank_diag =
      spec.diagnostics.mpi_rank_field && run_cfg.opts.vtk_save &&
      run_cfg.opts.vtk_mpi_rank_field;
    const bool enable_zhang_shu_diag =
      spec.diagnostics.zhang_shu_limiter &&
      run_cfg.opts.vtk_save &&
      run_cfg.opts.vtk_zhang_shu_diagnostics &&
      run_cfg.case_config.limiter_pipeline.zhang_shu;
    const bool need_h1 = spec.need_h1 || enable_h1_vis_diag;
    ExampleContext ctx = BuildExampleContext(run_cfg,
                                             run_cfg.igr_opts.space,
                                             need_sigma_space,
                                             need_h1);
    const MPI_Comm mesh_communicator = ctx.Mesh().GetComm();
    int mesh_communicator_rank = 0;
    int mesh_communicator_size = 1;
    MPI_Comm_rank(mesh_communicator, &mesh_communicator_rank);
    MPI_Comm_size(mesh_communicator, &mesh_communicator_size);
    const bool mesh_communicator_root = mesh_communicator_rank == 0;
    if (run_cfg.dg_backend == DGBackend::DGSEM)
    {
      detail::VerifyDGSEMElementGeometry(ctx.Mesh());
    }
    if (need_sigma_space)
    {
      MFEM_VERIFY(ctx.spaces.sigma_fes != nullptr,
                  "Sigma space was not created.");
    }
    if (need_h1)
    {
      MFEM_VERIFY(ctx.spaces.h1_vfes != nullptr,
                  "H1 velocity space was not created.");
    }

    const bool use_subcell_recovery =
      ::dg_igr::EnvFlagEnabledDefault("DG_IGR_IGR_SUBCELL_RECOVERY", false);
    const char* quadrature_text = std::getenv("DG_IGR_IGR_SUBCELL_QUADRATURE_MIN_ORDER");
    const std::string subcell_quadrature_text = quadrature_text ? quadrature_text : "0";
    MFEM_VERIFY(subcell_quadrature_text == "0" || subcell_quadrature_text == "4",
                "DG_IGR_IGR_SUBCELL_QUADRATURE_MIN_ORDER must be 0 (native) or 4.");
    const int subcell_quadrature_min_order = subcell_quadrature_text == "4" ? 4 : 0;
    MFEM_VERIFY(subcell_quadrature_min_order == 0 ||
                  (use_subcell_recovery && mfem::Device::Allows(mfem::Backend::CUDA_MASK)),
                "A subcell quadrature floor requires the subcell recovery bridge on CUDA.");
    const char* transport_text = std::getenv("DG_IGR_IGR_SUBCELL_TRANSPORT");
    const std::string subcell_transport = transport_text ? transport_text : "first-order";
    MFEM_VERIFY(subcell_transport == "first-order" || subcell_transport == "muscl",
                "DG_IGR_IGR_SUBCELL_TRANSPORT must be first-order or muscl.");
    const bool subcell_muscl = subcell_transport == "muscl";
    MFEM_VERIFY(!subcell_muscl || (use_subcell_recovery && mfem::Device::Allows(mfem::Backend::CUDA_MASK)),
                "Subcell MUSCL transport requires the subcell recovery bridge and CUDA packed DG faces.");
    std::unique_ptr<dg_igr::igr_internal::CompactSubcellRecovery> subcell_recovery;
    if (use_subcell_recovery)
    {
      MFEM_VERIFY(use_igr && run_cfg.dg_backend == DGBackend::DG && run_cfg.opts.order == 0 &&
                    mesh_communicator_size == 1 && !spec.make_static_source && !spec.after_init &&
                    run_cfg.opts.restart_vtk.empty(),
                  "Subcell recovery requires ordinary DG p0 Euler+IGR, rank one, no static source, "
                  "no post-initialization edit and no restart.");
      MFEM_VERIFY(ToLowerCopy(run_cfg.igr_opts.space) == "l2" || ToLowerCopy(run_cfg.igr_opts.space) == "dg",
                  "Subcell recovery currently requires an L2 sigma space.");
      const char* recovery_text = std::getenv("DG_IGR_IGR_SUBCELL_RECOVERY_KIND");
      const std::string recovery_kind = recovery_text ? recovery_text : "compact";
      MFEM_VERIFY(recovery_kind == "compact" || recovery_kind == "linear",
                  "DG_IGR_IGR_SUBCELL_RECOVERY_KIND must be compact or linear.");
      const char* closure_text = std::getenv("DG_IGR_IGR_SUBCELL_CLOSURE_ORDER");
      const std::string closure_order_text = closure_text ? closure_text : "2";
      MFEM_VERIFY(closure_order_text == "1" || closure_order_text == "2",
                  "DG_IGR_IGR_SUBCELL_CLOSURE_ORDER must be 1 or 2.");
      const int closure_order = closure_order_text == "1" ? 1 : 2;
      subcell_recovery = std::make_unique<dg_igr::igr_internal::CompactSubcellRecovery>(
        ctx.spaces.vfes, run_cfg.case_config.gamma, run_cfg.case_config.rk3_config.rho_min,
        run_cfg.case_config.rk3_config.p_min, recovery_kind == "linear", subcell_muscl);
      // The context owns sigma for output/flux consumers; its state-space order
      // remains p0. Recovery stays p2 for either scalar closure order. The
      // optional integration floor is bound to IGR after its construction.
      ctx.spaces.sigma_fes.reset();
      ctx.spaces.sigma_fec = std::make_unique<mfem::L2_FECollection>(closure_order, 2, mfem::BasisType::GaussLegendre);
      ctx.spaces.sigma_fes = std::make_unique<mfem::ParFiniteElementSpace>(&ctx.Mesh(),ctx.spaces.sigma_fec.get());
      mfem::out << "[igr-subcell] event=configuration transport=p0-independent-means"
                << " transport_reconstruction=" << subcell_transport
                << " recovery=" << recovery_kind << "-conservative recovery_order=2 closure=L2-p" << closure_order
                << " closure_quadrature=" << (subcell_quadrature_min_order ? "minimum-order-4" : "native-scalar-order")
                << " boundary-recovery=constant-nearest-mean"
                << " implementation=host unlimited=1 face_quadrature_order=8\n";
    }

    const mfem::RK3Config rk3_config = ResolveRK3StepperConfig(run_cfg);
    mfem::RK3SSPWithLimiter rk3 =
      MakeRK3Stepper(ctx.spaces.vfes,
                     ctx.dim,
                     rk3_config,
                     run_cfg.time_integrator);
    if (run_cfg.dg_backend == DGBackend::DG)
    {
      const int dg_face_qord = use_subcell_recovery ? 8 :
        2 * ctx.spaces.vfes.GetFE(0)->GetOrder() + run_cfg.case_config.int_order_offset;
      rk3.SetLimiterFaceQuadratureOrders({dg_face_qord});
    }
    ConfigureLimiterPipeline(rk3,
                             run_cfg.case_config.limiter_pipeline,
                             ctx.spaces.vfes,
                             ctx.dim,
                             run_cfg.case_config.gamma);
    rk3.EnableStageNanScan(run_cfg.opts.nan_scan);
    rk3.EnableStageAdmissibilityScan(run_cfg.opts.admissibility_scan);

    const double rho_floor = run_cfg.case_config.rk3_config.rho_min;
    std::unique_ptr<dg_igr::IGRSolver> igr;
    double vtk_c_alpha = spec.vtk.c_alpha_value;
    if (use_igr)
    {
      const auto alpha_policy = ResolveIGRAlphaLengthScalePolicy(run_cfg);
      AlphaInfo alpha_info =
        ComputeAlphaInfo(ctx.Mesh(),
                         run_cfg.igr_opts.c_alpha_user,
                         alpha_policy);
      if (subcell_recovery)
      {
        MFEM_VERIFY(alpha_policy != dg_igr::IGRAlphaLengthScalePolicy::Local,
                    "Subcell bridge requires a fixed physical alpha, not per-subcell h scaling.");
        const char* alpha_text = std::getenv("DG_IGR_IGR_SUBCELL_ALPHA");
        char* end = nullptr;
        const double fixed_alpha = alpha_text ? std::strtod(alpha_text,&end) : 0.0;
        MFEM_VERIFY(alpha_text && end && *end == '\0' && std::isfinite(fixed_alpha) && fixed_alpha > 0.0,
                    "Subcell bridge requires positive physical DG_IGR_IGR_SUBCELL_ALPHA.");
        alpha_info.alpha = fixed_alpha;
        alpha_info.c_alpha = fixed_alpha/(alpha_info.dx_eff*alpha_info.dx_eff);
      }
      const bool element_local_alpha =
        alpha_info.policy == dg_igr::IGRAlphaLengthScalePolicy::Local;
      const double alpha_effective_min =
        element_local_alpha
          ? alpha_info.c_alpha * alpha_info.hmin * alpha_info.hmin
          : alpha_info.alpha;
      const double alpha_effective_max =
        element_local_alpha
          ? alpha_info.c_alpha * alpha_info.hmax * alpha_info.hmax
          : alpha_info.alpha;
      if (mesh_communicator_root)
      {
        std::cout << "[igr-regularization] event=resolved"
                  << " c_alpha=" << alpha_info.c_alpha
                  << " h_eff_policy="
                  << IGRAlphaLengthScalePolicyToString(alpha_info.policy)
                  << " h_min=" << alpha_info.hmin
                  << " h_average=" << alpha_info.havg
                  << " h_max=" << alpha_info.hmax
                  << " h_eff_reference=" << alpha_info.dx_eff
                  << " alpha_effective_min=" << alpha_effective_min
                  << " alpha_effective_max=" << alpha_effective_max << "\n";
      }
      vtk_c_alpha = std::isnan(vtk_c_alpha) ? alpha_info.c_alpha : vtk_c_alpha;
      igr = std::make_unique<dg_igr::IGRSolver>(MakeIGRSolver(ctx.spaces,
                                                              ctx.dim,
                                                              ctx.num_equations,
                                                              run_cfg,
                                                              rho_floor,
                                                              alpha_info,
                                                              subcell_recovery ? &subcell_recovery->StateSpace() : nullptr));
      if (subcell_recovery)
      {
        MFEM_VERIFY(igr->GetSigmaUpdateCadence() == dg_igr::IGRSigmaUpdateCadence::EveryStage,
                    "Subcell bridge requires --igr-sigma-cadence every-stage.");
        igr->SetGradBackend(dg_igr::IGRGradBackend::BrokenDG);
        igr->SetSubcellQuadratureMinimumOrder(subcell_quadrature_min_order);
      }
    }

    MFEM_ASSERT(ctx.spaces.fes.GetOrdering() == mfem::Ordering::byNODES, "");

    HYPRE_BigInt glob_size = ctx.spaces.vfes.GlobalTrueVSize();
    if (mesh_communicator_root)
    {
      std::cout << "Number of unknowns: " << glob_size << std::endl;
    }

    MFEM_VERIFY(spec.make_ic,
                "CaseSpec is missing an initial condition builder.");
    std::unique_ptr<mfem::VectorCoefficient> ic =
      spec.make_ic(case_opts, ctx, case_cfg.gamma);
    MFEM_VERIFY(ic, "CaseSpec initial condition builder returned null.");

    std::unique_ptr<mfem::VectorCoefficient> static_source_coefficient;
    std::unique_ptr<mfem::ParGridFunction> static_source;
    if (spec.make_static_source)
    {
      static_source_coefficient =
        spec.make_static_source(case_opts, ctx, case_cfg.gamma);
      if (static_source_coefficient)
      {
        MFEM_VERIFY(ctx.num_equations == 4 &&
                      static_source_coefficient->GetVDim() == 4,
                    "CaseSpec static source must have four Euler components.");
        MFEM_VERIFY(ctx.spaces.vfes.GetOrdering() == mfem::Ordering::byNODES,
                    "Static-source projection requires byNODES Euler ordering.");
        static_source =
          std::make_unique<mfem::ParGridFunction>(&ctx.spaces.vfes);
        static_source->UseDevice(
          run_cfg.device_backend == DeviceBackend::CUDA);
        static_source->ProjectCoefficient(*static_source_coefficient);
        const mfem::dg_igr::NonFiniteSummary source_scan =
          mfem::dg_igr::ScanNonFinite(*static_source);
        MFEM_VERIFY(source_scan.global_non_finite == 0,
                    "Projected CaseSpec static source contains non-finite "
                    "values.");
        const double source_max_abs =
          std::max(std::abs(source_scan.global_min),
                   std::abs(source_scan.global_max));
        if (source_max_abs == 0.0)
        {
          // Preserve the source-free operator path exactly: no retained field,
          // RHS branch body, device vector add, or memory traffic.
          static_source.reset();
        }
        else
        {
          MFEM_VERIFY(run_cfg.dg_backend == DGBackend::DG ||
                        run_cfg.dg_backend == DGBackend::DGSEM,
                      "A nonzero CaseSpec static source currently requires the "
                      "DG or DGSEM backend (or the existing HDG adapter).");
        }
      }
    }

    EulerStateBundle state_bundle(ctx, case_cfg.gamma, rho_floor);
    if (igr)
    {
      state_bundle.AttachIGR(*igr);
    }
    std::optional<VTKRestartInfo> restart_info;
    if (run_cfg.opts.restart_vtk.empty())
    {
      if (subcell_recovery)
      {
        subcell_recovery->InitializeMeans(*ic,state_bundle.sol);
        state_bundle.pressure_view.Update();
      }
      else state_bundle.ProjectIC(*ic);
    }
    else
    {
      VTKRestartRequest request;
      request.path = run_cfg.opts.restart_vtk;
      request.cycle = run_cfg.opts.restart_cycle;
      request.legacy_step = run_cfg.opts.restart_step;
      restart_info = LoadEulerVTKRestart(
        request,
        run_cfg,
        ctx,
        state_bundle.sol,
        igr != nullptr,
        static_source.get());
      MFEM_VERIFY(run_cfg.opts.t_final > restart_info->time,
                  "-tf/--t-final must be greater than the restart time "
                    << restart_info->time << ".");
      if (mesh_communicator_root)
      {
        std::cout << "[vtk-restart] source=" << restart_info->pvd_path
                  << " cycle=" << restart_info->cycle
                  << " time=" << restart_info->time
                  << " step=" << restart_info->step
                  << " legacy=" << (restart_info->legacy ? 1 : 0)
                  << " ranks=" << mesh_communicator_size
                  << " reconstruction_max_abs="
                  << restart_info->reconstruction_max_abs
                  << " reconstruction_rel_l2="
                  << restart_info->reconstruction_rel_l2
                  << " mesh_point_max_abs="
                  << restart_info->mesh_point_max_abs
                  << " condition_inf="
                  << restart_info->interpolation_condition_inf
                  << " rho_min=" << restart_info->density_min
                  << " p_min=" << restart_info->pressure_min
                  << " sigma_policy=cold-recompute"
                  << " output_mode=new"
                  << " next_cycle=" << restart_info->cycle + 1
                  << " next_time=" << restart_info->time + run_cfg.opts.dt_save
                  << " load_seconds=" << restart_info->load_seconds << "\n";
      }
    }

    TimeLoopCallbacks callbacks =
      spec.make_callbacks
        ? spec.make_callbacks(ctx, state_bundle)
        : MakeDefaultTimeLoopCallbacks(spec.callback_defaults);
    RunSummary captured_summary{};
    bool captured_summary_valid = false;

    std::unique_ptr<H1Diagnostics> h1_diag;
    std::unique_ptr<L2Diagnostics> l2_diag;
    std::unique_ptr<MPIRankField> mpi_rank_field;
    std::unique_ptr<ZhangShuLimiterField> zhang_shu_field;
    if (enable_h1_vis_diag)
    {
      h1_diag = std::make_unique<H1Diagnostics>(*ctx.spaces.h1_vfes,
                                                state_bundle.sol,
                                                ctx.dim,
                                                rho_floor);
      state_bundle.field_list.h1_diag = h1_diag.get();
    }
    if (enable_l2_diag)
    {
      l2_diag = std::make_unique<L2Diagnostics>(ctx.spaces.fes,
                                                state_bundle.sol,
                                                ctx.dim,
                                                rho_floor);
      state_bundle.field_list.l2_diag = l2_diag.get();
    }
    if (enable_mpi_rank_diag)
    {
      mpi_rank_field = std::make_unique<MPIRankField>(ctx.Mesh());
      state_bundle.field_list.mpi_rank = mpi_rank_field.get();
    }
    if (enable_zhang_shu_diag)
    {
      zhang_shu_field = std::make_unique<ZhangShuLimiterField>(ctx.Mesh());
      state_bundle.field_list.zhang_shu = zhang_shu_field.get();
      rk3.SetZhangShuDiagnosticsFields(&zhang_shu_field->active,
                                       &zhang_shu_field->theta_rho,
                                       &zhang_shu_field->theta_p);
    }

    if (spec.after_init)
    {
      spec.after_init(case_opts, ctx, state_bundle);
    }

    const std::string output_dataset_name = detail::MakeEulerOutputDatasetName(
      case_cfg,
      run_cfg,
      vtk_c_alpha,
      restart_info ? &*restart_info : nullptr,
      ctx,
      static_source.get());

    // Admit the actual state produced by projection/restart and after_init
    // before any output or operator construction can trigger an instantaneous
    // closure, CFL probe, or Euler RHS evaluation.
    try
    {
      mfem::ZhangShuLimiterElementDiagnostics initial_limiter_diagnostics;
      mfem::ZhangShuLimiterElementDiagnostics* diagnostics =
        zhang_shu_field ? &initial_limiter_diagnostics : nullptr;
      const bool applied = detail::ApplyInitialEulerZhangShuTransaction(
        run_cfg, state_bundle, diagnostics);
      if (applied && zhang_shu_field)
      {
        detail::PublishInitialZhangShuDiagnostics(initial_limiter_diagnostics,
                                                   *zhang_shu_field);
      }
    }
    catch (const mfem::RecoverableLimiterFailure& failure)
    {
      TimeLoopResult initial_failure;
      initial_failure.steps = restart_info
                                ? static_cast<int>(restart_info->step)
                                : 0;
      initial_failure.initial_time = restart_info ? restart_info->time : 0.0;
      initial_failure.final_time = initial_failure.initial_time;
      initial_failure.rejected_attempts = 1;
      initial_failure.rejected_limiter_elements =
        failure.GlobalRejectedElementCount();
      initial_failure.success = false;
      initial_failure.termination_reason = "initial-limiter-admission-failed";
      PopulateExplicitCFLDiagnostics(
        initial_failure, MakeTimeLoopConfig(run_cfg), rk3);

      int communicator_size = 1;
      MPI_Comm_size(ctx.Mesh().GetComm(), &communicator_size);
      captured_summary = BuildRunSummary(ctx,
                                         initial_failure,
                                         /*elapsed_seconds=*/0.0,
                                         communicator_size,
                                         igr.get());
      captured_summary_valid = true;
      if (mesh_communicator_root)
      {
        std::cout << "[initial-admissibility] event=limiter-rejection"
                  << " rejected_limiter_elements="
                  << failure.GlobalRejectedElementCount()
                  << " state_unchanged=1"
                  << " rhs_evaluations=0"
                  << " closure_evaluations=0\n";
        EmitExplicitTimeRunSummary(initial_failure);
        std::cout << "[explicit-time] event=final-status"
                  << " final_time=" << initial_failure.final_time
                  << " success=0"
                  << " termination_reason="
                  << initial_failure.termination_reason << "\n";
      }
      WriteRunMetadataSidecar(case_cfg,
                              run_cfg,
                              &captured_summary,
                              igr.get(),
                              vtk_c_alpha,
                              argc,
                              argv,
                              /*success=*/false,
                              output_dataset_name,
                              restart_info ? &*restart_info : nullptr,
                              mesh_communicator,
                              static_source.get(),
                              &ctx);
      return 2;
    }

    std::string state_snapshot_dir;
    if (run_cfg.opts.save_state_files)
    {
      const mfem::io::VTKOutputConfig vtk_cfg = MakeCaseVTKConfig(case_cfg);
      state_snapshot_dir = (std::filesystem::path(vtk_cfg.prefix_path) /
                            output_dataset_name / "glvis_state")
                             .string();
      SaveInitialEulerState(ctx,
                            state_bundle.sol,
                            spec.glvis.precision,
                            "euler-mesh",
                            "euler",
                            state_snapshot_dir);
    }

    MFEM_VERIFY(spec.build_markers,
                "CaseSpec is missing a boundary marker builder.");
    const BoundaryMarkerMap bdr_markers = spec.build_markers(ctx);

    const auto on_open = [&h1_diag]()
    {
      if (h1_diag)
      {
        h1_diag->Update();
      }
    };

    MFEM_VERIFY(spec.apply_bc, "CaseSpec is missing boundary wiring.");
    auto bc_setup = [&spec, &case_opts, &ctx](auto& bc, const auto& markers)
    {
      spec.apply_bc(case_opts, ctx, bc, markers);
    };

    const bool ok = detail::WithEulerOperator<NumericalFlux>(
      ctx,
      case_cfg.gamma,
      igr.get(),
      state_bundle.sol,
      static_source.get(),
      subcell_recovery.get(),
      bdr_markers,
      bc_setup,
      [&ctx,
       &case_cfg,
       &run_cfg,
       &state_bundle,
       &rk3,
       &glvis_enabled,
       &callbacks,
       &spec,
       &on_open,
       &captured_summary,
       &restart_info,
       &static_source,
       vtk_c_alpha](auto& op)
      {
        return RunEulerCase(ctx,
                            case_cfg,
                            run_cfg,
                            state_bundle,
                            rk3,
                            op,
                            glvis_enabled,
                            callbacks,
                            spec.glvis.precision,
                            on_open,
                            &captured_summary,
                            vtk_c_alpha,
                            spec.vtk.print_dataset_name,
                            restart_info ? &*restart_info : nullptr,
                            static_source.get());
      });
    // RunEulerCase populates the bounded summary for both success and a clean
    // time-loop failure such as an admissibility rejection at dt_min.
    captured_summary_valid = true;

    if (igr && igr->GetLinearSolveCalls() > 0 && mesh_communicator_root)
    {
      const double avg_linear_time =
        igr->GetTotalLinearSolveSeconds() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_assemble_time =
        igr->GetTotalAssembleSeconds() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_form_time =
        igr->GetTotalFormSystemSeconds() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_pc_setup_time =
        igr->GetTotalPCSetupSeconds() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const int total_linear_iters = igr->GetTotalLinearIterations();
      const double avg_krylov_operator_apply_count =
        igr->GetTotalKrylovOperatorApplyCount() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_krylov_preconditioner_apply_count =
        igr->GetTotalKrylovPreconditionerApplyCount() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double avg_krylov_preconditioner_operator_apply_count =
        igr->GetTotalKrylovPreconditionerOperatorApplyCount() /
        static_cast<double>(igr->GetLinearSolveCalls());
      const double krylov_operator_apply_count_per_iter =
        (total_linear_iters > 0)
          ? (igr->GetTotalKrylovOperatorApplyCount() /
             static_cast<double>(total_linear_iters))
          : 0.0;
      const double krylov_preconditioner_apply_count_per_iter =
        (total_linear_iters > 0)
          ? (igr->GetTotalKrylovPreconditionerApplyCount() /
             static_cast<double>(total_linear_iters))
          : 0.0;
      const double krylov_preconditioner_operator_apply_count_per_iter =
        (total_linear_iters > 0)
          ? (igr->GetTotalKrylovPreconditionerOperatorApplyCount() /
             static_cast<double>(total_linear_iters))
          : 0.0;
      const double igr_total_pct =
        (captured_summary_valid && captured_summary.elapsed > 0.0)
          ? (100.0 * igr->GetTotalIGRWorkSeconds() / captured_summary.elapsed)
          : 0.0;
      const double igr_solve_pct =
        (captured_summary_valid && captured_summary.elapsed > 0.0)
          ? (100.0 * igr->GetTotalLinearSolveSeconds() /
             captured_summary.elapsed)
          : 0.0;
      const double igr_stage_total_pct =
        (igr->GetTotalUpdateCallSeconds() > 0.0)
          ? (100.0 * igr->GetTotalIGRWorkSeconds() /
             igr->GetTotalUpdateCallSeconds())
          : 0.0;
      const double igr_stage_solve_pct =
        (igr->GetTotalUpdateCallSeconds() > 0.0)
          ? (100.0 * igr->GetTotalLinearSolveSeconds() /
             igr->GetTotalUpdateCallSeconds())
          : 0.0;
      std::cout << "[igr-linear] event=run-summary"
                << " calls=" << igr->GetLinearSolveCalls()
                << " converged_calls=" << igr->GetTotalLinearConvergedCalls()
                << " production_calls="
                << igr->GetProductionLinearSolveCalls()
                << " production_converged_calls="
                << igr->GetProductionLinearConvergedCalls()
                << " output_only_calls="
                << igr->GetOutputOnlyLinearSolveCalls()
                << " output_only_converged_calls="
                << igr->GetOutputOnlyLinearConvergedCalls()
                << " output_only_iters="
                << igr->GetOutputOnlyLinearIterations()
                << " output_only_solve_s="
                << igr->GetOutputOnlyLinearSolveSeconds()
                << " output_only_igr_s="
                << igr->GetOutputOnlyIGRWorkSeconds()
                << " total_iters=" << total_linear_iters
                << " avg_iters=" << igr->GetAverageLinearIterations()
                << " max_iters=" << igr->GetMaxLinearIterations()
                << " avg_solve_iter_s="
                << ((total_linear_iters > 0)
                      ? (igr->GetTotalLinearSolveSeconds() /
                         static_cast<double>(total_linear_iters))
                      : 0.0)
                << " avg_krylov_operator_apply_count="
                << avg_krylov_operator_apply_count
                << " avg_krylov_preconditioner_apply_count="
                << avg_krylov_preconditioner_apply_count
                << " avg_krylov_preconditioner_operator_apply_count="
                << avg_krylov_preconditioner_operator_apply_count
                << " krylov_operator_apply_count_per_iter="
                << krylov_operator_apply_count_per_iter
                << " krylov_preconditioner_apply_count_per_iter="
                << krylov_preconditioner_apply_count_per_iter
                << " krylov_preconditioner_operator_apply_count_per_iter="
                << krylov_preconditioner_operator_apply_count_per_iter
                << " avg_assemble_s=" << avg_assemble_time
                << " avg_form_s=" << avg_form_time
                << " avg_pc_setup_s=" << avg_pc_setup_time
                << " avg_solve_s=" << avg_linear_time
                << " total_assemble_s=" << igr->GetTotalAssembleSeconds()
                << " total_form_s=" << igr->GetTotalFormSystemSeconds()
                << " total_pc_setup_s=" << igr->GetTotalPCSetupSeconds()
                << " total_cheb_diag_s="
                << igr->GetTotalChebyshevDiagonalSeconds()
                << " total_cheb_setup_s="
                << igr->GetTotalChebyshevSetupSeconds()
                << " total_cheb_eigen_reuses="
                << igr->GetTotalChebyshevEigenvalueReuses()
                << " total_cheb_smoother_reuses="
                << igr->GetTotalChebyshevSmootherReuses()
                << " total_cheb_smoother_builds="
                << igr->GetTotalChebyshevSmootherBuilds()
                << " total_krylov_operator_apply_s="
                << igr->GetTotalKrylovOperatorApplySeconds()
                << " total_krylov_preconditioner_apply_s="
                << igr->GetTotalKrylovPreconditionerApplySeconds()
                << " total_krylov_preconditioner_operator_apply_s="
                << igr->GetTotalKrylovPreconditionerOperatorApplySeconds()
                << " total_krylov_operator_apply_count="
                << igr->GetTotalKrylovOperatorApplyCount()
                << " total_krylov_preconditioner_apply_count="
                << igr->GetTotalKrylovPreconditionerApplyCount()
                << " total_krylov_preconditioner_operator_apply_count="
                << igr->GetTotalKrylovPreconditionerOperatorApplyCount()
                << " total_solve_s=" << igr->GetTotalLinearSolveSeconds()
                << " total_igr_s=" << igr->GetTotalIGRWorkSeconds()
                << " total_update_s=" << igr->GetTotalUpdateCallSeconds()
                << " solve_pct=" << igr_solve_pct
                << " total_igr_pct=" << igr_total_pct
                << " solve_pct_stage=" << igr_stage_solve_pct
                << " total_igr_pct_stage=" << igr_stage_total_pct
                << " last_solver=" << igr->GetLastLinearSolverName()
                << " last_precond=" << igr->GetLastPrecondName()
                << " last_cheb_diag_s="
                << igr->GetLastChebyshevDiagonalSeconds()
                << " last_cheb_setup_s="
                << igr->GetLastChebyshevSetupSeconds()
                << " last_cheb_eigen_reused="
                << (igr->GetLastChebyshevEigenvalueReused() ? 1 : 0)
                << " last_cheb_smoother_reused="
                << (igr->GetLastChebyshevSmootherReused() ? 1 : 0)
                << " last_cheb_eigen_estimate="
                << igr->GetLastChebyshevEigenvalueEstimate()
                << " last_krylov_operator_apply_s="
                << igr->GetLastKrylovOperatorApplySeconds()
                << " last_krylov_preconditioner_apply_s="
                << igr->GetLastKrylovPreconditionerApplySeconds()
                << " last_krylov_preconditioner_operator_apply_s="
                << igr->GetLastKrylovPreconditionerOperatorApplySeconds()
                << " last_krylov_operator_apply_count="
                << igr->GetLastKrylovOperatorApplyCount()
                << " last_krylov_preconditioner_apply_count="
                << igr->GetLastKrylovPreconditionerApplyCount()
                << " last_krylov_preconditioner_operator_apply_count="
                << igr->GetLastKrylovPreconditionerOperatorApplyCount()
                << " direct_identity_system="
                << (igr->GetLastDirectIdentitySystem() ? 1 : 0)
                << " direct_identity_reason="
                << igr->GetLastDirectIdentityReason()
                << " direct_identity_uses="
                << igr->GetDirectIdentitySystemUses()
                << " exact_face_requested="
                << (igr->GetLastExactFaceRequested() ? 1 : 0)
                << " exact_face_effective="
                << (igr->GetLastExactFaceEffective() ? 1 : 0)
                << " exact_face_reason=" << igr->GetLastExactFaceReason()
                << " exact_face_local_nf="
                << igr->GetLastExactFaceLocalFaceCount()
                << " exact_face_shared_nf="
                << igr->GetLastExactFaceSharedFaceCount()
                << " exact_face_operator_uses="
                << igr->GetExactFaceOperatorUseCount()
                << " exact_face_last_mult_calls="
                << igr->GetLastExactFaceOperatorMultCalls() << "\n";
    }

    if (!ok)
    {
      WriteRunMetadataSidecar(case_cfg,
                              run_cfg,
                              captured_summary_valid ? &captured_summary
                                                     : nullptr,
                              igr.get(),
                              vtk_c_alpha,
                              argc,
                              argv,
                              /*success=*/false,
                              output_dataset_name,
                              restart_info ? &*restart_info : nullptr,
                              mesh_communicator,
                              static_source.get(),
                              &ctx);
      return 2;
    }

    if (run_cfg.opts.save_state_files)
    {
      SaveFinalEulerState(ctx,
                          state_bundle.sol,
                          spec.glvis.precision,
                          "euler-mesh-final",
                          "euler",
                          state_snapshot_dir);
    }

    WriteRunMetadataSidecar(case_cfg,
                            run_cfg,
                            captured_summary_valid ? &captured_summary
                                                   : nullptr,
                            igr.get(),
                            vtk_c_alpha,
                            argc,
                            argv,
                            /*success=*/true,
                            output_dataset_name,
                            restart_info ? &*restart_info : nullptr,
                            mesh_communicator,
                            static_source.get(),
                            &ctx);

    if (spec.postprocess)
    {
      const int code = spec.postprocess(case_opts, ctx, state_bundle, *ic);
      if (code != 0)
      {
        return code;
      }
    }

    return 0;
  }
};

} // namespace dg_sim

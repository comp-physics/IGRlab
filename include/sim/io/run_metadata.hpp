#pragma once

/**
 * @file sim/io/run_metadata.hpp
 * @brief Runtime metadata sidecar writer for VTK datasets.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "mfem.hpp"

#include "physics/igr/IGR.hpp"
#include "sim/config/case_config.hpp"
#include "sim/io/vtk_restart.hpp"
#include "sim/io/vtk_helpers.hpp"
#include "sim/time/time_callbacks.hpp"

namespace dg_sim
{

namespace detail
{

inline std::string JsonEscape(const std::string& input)
{
  std::string out;
  out.reserve(input.size() + 8);
  for (unsigned char c : input)
  {
    switch (c)
    {
    case '\"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20)
      {
        std::ostringstream hex;
        hex << "\\u" << std::hex << std::uppercase << std::setw(4)
            << std::setfill('0') << static_cast<int>(c);
        out += hex.str();
      }
      else
      {
        out.push_back(static_cast<char>(c));
      }
    }
  }
  return out;
}

inline std::string ToJsonString(const std::string& value)
{
  return std::string("\"") + JsonEscape(value) + "\"";
}

inline std::string ToJsonNumber(double value)
{
  if (!std::isfinite(value))
  {
    return "null";
  }

  std::ostringstream os;
  os << std::setprecision(17) << value;
  return os.str();
}

inline std::string ISO8601NowUTC()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &tt);
#else
  gmtime_r(&tt, &tm);
#endif
  std::ostringstream os;
  os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return os.str();
}

inline std::string JoinCLI(int argc, char** argv)
{
  std::ostringstream os;
  for (int i = 0; i < argc; ++i)
  {
    if (i > 0)
    {
      os << ' ';
    }
    os << argv[i];
  }
  return os.str();
}

} // namespace detail

/// @brief Persist a sidecar JSON metadata file next to the VTK dataset.
inline void WriteRunMetadataSidecar(const CaseConfig& case_cfg,
                                    const RunConfig& run_cfg,
                                    const RunSummary* summary,
                                    const dg_igr::IGRSolver* igr,
                                    double c_alpha_value,
                                    int argc,
                                    char** argv,
                                    bool success,
                                    const std::string& output_dataset_name = {},
                                    const VTKRestartInfo* restart_info = nullptr,
                                    MPI_Comm communicator = MPI_COMM_WORLD,
                                    const mfem::ParGridFunction* static_source = nullptr,
                                    const ExampleContext* context = nullptr)
{
  int communicator_rank = 0;
  int communicator_size = 1;
  MPI_Comm_rank(communicator, &communicator_rank);
  MPI_Comm_size(communicator, &communicator_size);
  if (!run_cfg.opts.vtk_save)
  {
    return;
  }
  std::string static_source_fingerprint = "none";
  if (static_source)
  {
    MFEM_VERIFY(context != nullptr,
                "Run metadata needs the active context to identify a static "
                "source.");
    const std::uint64_t fingerprint =
      ComputeVTKRestartFieldFingerprint(*static_source, *context);
    std::ostringstream fingerprint_text;
    fingerprint_text << std::hex << std::setfill('0') << std::setw(16)
                     << fingerprint;
    static_source_fingerprint = fingerprint_text.str();
  }
  if (communicator_rank != 0)
  {
    return;
  }

  const mfem::io::VTKOutputConfig vtk_cfg = MakeCaseVTKConfig(case_cfg);
  std::string dataset_name = output_dataset_name.empty()
                               ? MakeCaseVTKDatasetName(
                                   case_cfg, run_cfg, c_alpha_value)
                               : output_dataset_name;
  if (output_dataset_name.empty() && static_source)
  {
    dataset_name += "_src_" + static_source_fingerprint;
  }
  const std::filesystem::path output_dir =
    std::filesystem::path(vtk_cfg.prefix_path) / dataset_name;
  std::filesystem::create_directories(output_dir);

  const std::filesystem::path metadata_path = output_dir / "run_metadata.json";
  std::ofstream out(metadata_path);
  if (!out)
  {
    std::cout << "[run-metadata] warning=\"failed to open metadata file\""
              << " path=" << metadata_path.string() << "\n";
    return;
  }

  out << std::setprecision(17);
  out << "{\n";
  out << "  \"schema_version\": 1,\n";
  out << "  \"timestamp_utc\": "
      << detail::ToJsonString(detail::ISO8601NowUTC()) << ",\n";
  out << "  \"status\": {\n";
  out << "    \"success\": " << (success ? "true" : "false") << "\n";
  out << "  },\n";
  out << "  \"case\": {\n";
  out << "    \"name\": " << detail::ToJsonString(case_cfg.case_name) << ",\n";
  out << "    \"mesh_path\": " << detail::ToJsonString(run_cfg.mesh_path)
      << ",\n";
  out << "    \"default_mesh\": "
      << detail::ToJsonString(case_cfg.default_mesh_file) << ",\n";
  out << "    \"gamma\": " << case_cfg.gamma << "\n";
  out << "  },\n";
  out << "  \"runtime\": {\n";
  out << "    \"mpi_ranks\": " << communicator_size << ",\n";
  out << "    \"order\": " << run_cfg.opts.order << ",\n";
  out << "    \"serial_refinement\": " << run_cfg.opts.ser_ref_levels << ",\n";
  out << "    \"parallel_refinement\": " << run_cfg.opts.par_ref_levels
      << ",\n";
  out << "    \"device_backend\": "
      << detail::ToJsonString(
           DeviceBackendToString(run_cfg.device_backend))
      << ",\n";
  out << "    \"discretization\": "
      << detail::ToJsonString(
           DiscretizationToString(run_cfg.discretization))
      << ",\n";
  out << "    \"discretization_provenance\": "
      << detail::ToJsonString(DiscretizationProvenanceToString(
           run_cfg.discretization_provenance))
      << ",\n";
  out << "    \"dg_backend\": "
      << detail::ToJsonString(DGBackendToString(run_cfg.dg_backend)) << ",\n";
  out << "    \"time_integrator\": "
      << detail::ToJsonString(TimeIntegratorToString(run_cfg.time_integrator))
      << ",\n";
  out << "    \"dgsem_volume_flux\": "
      << detail::ToJsonString(
           DGSEMVolumeFluxToString(run_cfg.dgsem_volume_flux))
      << ",\n";
  out << "    \"dgsem_volume_dissipation\": "
      << detail::ToJsonString(
           DGSEMVolumeDissipationToString(run_cfg.dgsem_volume_dissipation))
      << ",\n";
  out << "    \"dgsem_surface_flux\": "
      << detail::ToJsonString(
           DGSEMSurfaceFluxToString(run_cfg.dgsem_surface_flux))
      << ",\n";
  out << "    \"dgsem_entropy_jacobian\": "
      << detail::ToJsonString(
           DGSEMEntropyJacobianToString(run_cfg.dgsem_entropy_jacobian))
      << ",\n";
  out << "    \"c_alpha\": " << detail::ToJsonNumber(c_alpha_value)
      << ",\n";
  out << "    \"igr_space\": " << detail::ToJsonString(run_cfg.igr_opts.space)
      << ",\n";
  out << "    \"igr_grad\": " << detail::ToJsonString(run_cfg.igr_opts.grad)
      << ",\n";
  out << "    \"igr_solve_mode\": "
      << detail::ToJsonString(run_cfg.igr_opts.solve_mode) << ",\n";
  out << "    \"igr_linear_solver\": "
      << detail::ToJsonString(run_cfg.igr_opts.linear_solver) << ",\n";
  out << "    \"igr_preconditioner\": "
      << detail::ToJsonString(run_cfg.igr_opts.preconditioner) << ",\n";
  out << "    \"igr_nonconvergence_policy\": "
      << detail::ToJsonString(run_cfg.igr_opts.nonconvergence_policy) << ",\n";
  out << "    \"igr_sigma_cfl\": "
      << (run_cfg.igr_opts.sigma_aware_cfl ? "true" : "false") << ",\n";
  out << "    \"igr_sigma_rusanov\": "
      << (run_cfg.igr_opts.sigma_aware_rusanov ? "true" : "false") << ",\n";
  out << "    \"igr_invariant_domain_flux\": "
      << (run_cfg.igr_opts.igr_invariant_domain_flux ? "true" : "false")
      << ",\n";
  out << "    \"igr_sigma_cadence\": "
      << detail::ToJsonString(run_cfg.igr_opts.sigma_cadence) << ",\n";
  out << "    \"igr_sigma_update_interval\": "
      << run_cfg.igr_opts.sigma_update_interval << ",\n";
  out << "    \"igr_stage_rtol_scale\": " << run_cfg.igr_opts.stage_rtol_scale
      << ",\n";
  out << "    \"igr_lin_rtol\": "
      << detail::ToJsonNumber(
           run_cfg.igr_opts.lin_rtol >= 0.0
             ? run_cfg.igr_opts.lin_rtol
             : run_cfg.case_config.igr_config.lin_rtol)
      << ",\n";
  out << "    \"igr_lin_atol\": "
      << detail::ToJsonNumber(
           run_cfg.igr_opts.lin_atol >= 0.0
             ? run_cfg.igr_opts.lin_atol
             : run_cfg.case_config.igr_config.lin_atol)
      << ",\n";
  out << "    \"igr_lin_maxit\": "
      << (run_cfg.igr_opts.lin_maxit >= 0
            ? run_cfg.igr_opts.lin_maxit
            : run_cfg.case_config.igr_config.lin_maxit)
      << ",\n";
  out << "    \"igr_ip_kind\": "
      << detail::ToJsonString(run_cfg.igr_opts.ip_kind) << ",\n";
  out << "    \"igr_amg_coarsen_type\": " << run_cfg.igr_opts.amg_coarsen_type
      << ",\n";
  out << "    \"igr_amg_relax_type\": " << run_cfg.igr_opts.amg_relax_type
      << ",\n";
  out << "    \"igr_amg_max_levels\": " << run_cfg.igr_opts.amg_max_levels
      << ",\n";
  out << "    \"igr_amg_strong_threshold\": "
      << run_cfg.igr_opts.amg_strong_threshold << ",\n";
  out << "    \"igr_exact_dg_face_operator\": "
      << (run_cfg.igr_opts.exact_dg_face_operator ? "true" : "false")
      << ",\n";
  out << "    \"zhang_shu_points\": "
      << detail::ToJsonString(run_cfg.opts.zhang_shu_points) << ",\n";
  out << "    \"zhang_shu_vtk_points\": "
      << (run_cfg.opts.zhang_shu_vtk_points ? "true" : "false") << ",\n";
  out << "    \"cfl\": " << run_cfg.opts.cfl << ",\n";
  out << "    \"dt\": " << run_cfg.opts.dt << ",\n";
  out << "    \"dt_min\": " << run_cfg.opts.dt_min << ",\n";
  out << "    \"t_final\": " << run_cfg.opts.t_final << ",\n";
  out << "    \"nan_scan\": " << (run_cfg.opts.nan_scan ? "true" : "false")
      << ",\n";
  out << "    \"conservation_balance\": "
      << (run_cfg.opts.conservation_balance ? "true" : "false") << ",\n";
  out << "    \"static_source_present\": "
      << (static_source ? "true" : "false") << ",\n";
  out << "    \"static_source_fingerprint\": "
      << detail::ToJsonString(static_source_fingerprint) << ",\n";
  out << "    \"save_state_files\": "
      << (run_cfg.opts.save_state_files ? "true" : "false") << ",\n";
  out << "    \"vtk_h1_diagnostics\": "
      << (run_cfg.opts.vtk_h1_diagnostics ? "true" : "false") << ",\n";
  out << "    \"vtk_l2_diagnostics\": "
      << (run_cfg.opts.vtk_l2_diagnostics ? "true" : "false") << ",\n";
  out << "    \"vtk_mpi_rank_field\": "
      << (run_cfg.opts.vtk_mpi_rank_field ? "true" : "false") << ",\n";
  out << "    \"vtk_zhang_shu_diagnostics\": "
      << (run_cfg.opts.vtk_zhang_shu_diagnostics ? "true" : "false")
      << "\n";
  out << "  },\n";
  out << "  \"restart\": {\n";
  out << "    \"enabled\": " << (restart_info ? "true" : "false");
  if (restart_info)
  {
    out << ",\n";
    out << "    \"source\": "
        << detail::ToJsonString(restart_info->pvd_path) << ",\n";
    out << "    \"cycle\": " << restart_info->cycle << ",\n";
    out << "    \"time\": " << restart_info->time << ",\n";
    out << "    \"step\": " << restart_info->step << ",\n";
    out << "    \"legacy\": "
        << (restart_info->legacy ? "true" : "false") << ",\n";
    out << "    \"output_mode\": \"new\"\n";
  }
  else
  {
    out << "\n";
  }
  out << "  },\n";

  out << "  \"performance\": {\n";
  if (summary)
  {
    out << "    \"steps\": " << summary->steps << ",\n";
    out << "    \"segment_steps\": " << summary->segment_steps << ",\n";
    out << "    \"accepted_attempts\": " << summary->accepted_attempts
        << ",\n";
    out << "    \"rejected_attempts\": " << summary->rejected_attempts
        << ",\n";
    out << "    \"rejected_limiter_elements\": "
        << summary->rejected_limiter_elements << ",\n";
    out << "    \"rhs_evaluations\": " << summary->rhs_evaluations
        << ",\n";
    out << "    \"initial_time\": " << summary->initial_time << ",\n";
    out << "    \"final_time\": " << summary->final_time << ",\n";
    out << "    \"minimum_attempted_dt\": "
        << summary->minimum_attempted_dt << ",\n";
    out << "    \"minimum_accepted_dt\": " << summary->minimum_accepted_dt
        << ",\n";
    out << "    \"maximum_attempted_dt\": "
        << summary->maximum_attempted_dt << ",\n";
    out << "    \"maximum_accepted_dt\": "
        << summary->maximum_accepted_dt << ",\n";
    out << "    \"dt_min_reached\": "
        << (summary->dt_min_reached ? "true" : "false") << ",\n";
    out << "    \"success\": " << (summary->success ? "true" : "false")
        << ",\n";
    out << "    \"termination_reason\": "
        << detail::ToJsonString(summary->termination_reason) << ",\n";
    out << "    \"elapsed\": " << summary->elapsed << ",\n";
    out << "    \"avg_step\": " << summary->avg_step << ",\n";
    out << "    \"avg_step_per_dof_local\": " << summary->avg_step_per_dof_local
        << ",\n";
    out << "    \"avg_step_per_dof_rank\": " << summary->avg_step_per_dof_rank
        << ",\n";
    out << "    \"local_dofs\": " << summary->local_dofs << ",\n";
    out << "    \"global_dofs\": " << summary->global_dofs << ",\n";
    out << "    \"ranks\": " << summary->ranks << "\n";
  }
  else
  {
    out << "    \"steps\": 0,\n";
    out << "    \"segment_steps\": 0,\n";
    out << "    \"accepted_attempts\": 0,\n";
    out << "    \"rejected_attempts\": 0,\n";
    out << "    \"rejected_limiter_elements\": 0,\n";
    out << "    \"rhs_evaluations\": 0,\n";
    out << "    \"initial_time\": 0.0,\n";
    out << "    \"final_time\": 0.0,\n";
    out << "    \"minimum_attempted_dt\": 0.0,\n";
    out << "    \"minimum_accepted_dt\": 0.0,\n";
    out << "    \"maximum_attempted_dt\": 0.0,\n";
    out << "    \"maximum_accepted_dt\": 0.0,\n";
    out << "    \"dt_min_reached\": false,\n";
    out << "    \"success\": false,\n";
    out << "    \"termination_reason\": \"not-started\",\n";
    out << "    \"elapsed\": 0.0,\n";
    out << "    \"avg_step\": 0.0,\n";
    out << "    \"avg_step_per_dof_local\": 0.0,\n";
    out << "    \"avg_step_per_dof_rank\": 0.0,\n";
    out << "    \"local_dofs\": 0,\n";
    out << "    \"global_dofs\": 0,\n";
    out << "    \"ranks\": " << communicator_size << "\n";
  }
  out << "  },\n";

  out << "  \"igr_linear\": {\n";
  out << "    \"enabled\": " << ((igr != nullptr) ? "true" : "false");
  if (igr)
  {
    const double elapsed_seconds = summary ? summary->elapsed : 0.0;
    const int total_linear_iters = igr->GetTotalLinearIterations();
    out << ",\n";
    out << "    \"calls\": " << igr->GetLinearSolveCalls() << ",\n";
    out << "    \"converged_calls\": " << igr->GetTotalLinearConvergedCalls()
        << ",\n";
    out << "    \"total_iters\": " << total_linear_iters << ",\n";
    out << "    \"avg_iters\": " << igr->GetAverageLinearIterations() << ",\n";
    out << "    \"max_iters\": " << igr->GetMaxLinearIterations() << ",\n";
    out << "    \"avg_solve_iter_s\": "
        << ((total_linear_iters > 0)
              ? (igr->GetTotalLinearSolveSeconds() /
                 static_cast<double>(total_linear_iters))
              : 0.0)
        << ",\n";
    out << "    \"avg_krylov_operator_apply_count\": "
        << ((igr->GetLinearSolveCalls() > 0)
              ? (static_cast<double>(igr->GetTotalKrylovOperatorApplyCount()) /
                 static_cast<double>(igr->GetLinearSolveCalls()))
              : 0.0)
        << ",\n";
    out << "    \"avg_krylov_preconditioner_apply_count\": "
        << ((igr->GetLinearSolveCalls() > 0)
              ? (static_cast<double>(
                   igr->GetTotalKrylovPreconditionerApplyCount()) /
                 static_cast<double>(igr->GetLinearSolveCalls()))
              : 0.0)
        << ",\n";
    out << "    \"avg_krylov_preconditioner_operator_apply_count\": "
        << ((igr->GetLinearSolveCalls() > 0)
              ? (static_cast<double>(
                   igr->GetTotalKrylovPreconditionerOperatorApplyCount()) /
                 static_cast<double>(igr->GetLinearSolveCalls()))
              : 0.0)
        << ",\n";
    out << "    \"krylov_operator_apply_count_per_iter\": "
        << ((total_linear_iters > 0)
              ? (static_cast<double>(igr->GetTotalKrylovOperatorApplyCount()) /
                 static_cast<double>(total_linear_iters))
              : 0.0)
        << ",\n";
    out << "    \"krylov_preconditioner_apply_count_per_iter\": "
        << ((total_linear_iters > 0)
              ? (static_cast<double>(
                   igr->GetTotalKrylovPreconditionerApplyCount()) /
                 static_cast<double>(total_linear_iters))
              : 0.0)
        << ",\n";
    out << "    \"krylov_preconditioner_operator_apply_count_per_iter\": "
        << ((total_linear_iters > 0)
              ? (static_cast<double>(
                   igr->GetTotalKrylovPreconditionerOperatorApplyCount()) /
                 static_cast<double>(total_linear_iters))
              : 0.0)
        << ",\n";
    out << "    \"total_assemble_s\": " << igr->GetTotalAssembleSeconds()
        << ",\n";
    out << "    \"total_form_system_s\": " << igr->GetTotalFormSystemSeconds()
        << ",\n";
    out << "    \"total_pc_setup_s\": " << igr->GetTotalPCSetupSeconds()
        << ",\n";
    out << "    \"total_cheb_diag_s\": "
        << igr->GetTotalChebyshevDiagonalSeconds() << ",\n";
    out << "    \"total_cheb_setup_s\": "
        << igr->GetTotalChebyshevSetupSeconds() << ",\n";
    out << "    \"total_cheb_eigen_reuses\": "
        << igr->GetTotalChebyshevEigenvalueReuses() << ",\n";
    out << "    \"total_cheb_smoother_reuses\": "
        << igr->GetTotalChebyshevSmootherReuses() << ",\n";
    out << "    \"total_cheb_smoother_builds\": "
        << igr->GetTotalChebyshevSmootherBuilds() << ",\n";
    out << "    \"total_krylov_operator_apply_s\": "
        << igr->GetTotalKrylovOperatorApplySeconds() << ",\n";
    out << "    \"total_krylov_preconditioner_apply_s\": "
        << igr->GetTotalKrylovPreconditionerApplySeconds() << ",\n";
    out << "    \"total_krylov_preconditioner_operator_apply_s\": "
        << igr->GetTotalKrylovPreconditionerOperatorApplySeconds() << ",\n";
    out << "    \"total_krylov_operator_apply_count\": "
        << igr->GetTotalKrylovOperatorApplyCount() << ",\n";
    out << "    \"total_krylov_preconditioner_apply_count\": "
        << igr->GetTotalKrylovPreconditionerApplyCount() << ",\n";
    out << "    \"total_krylov_preconditioner_operator_apply_count\": "
        << igr->GetTotalKrylovPreconditionerOperatorApplyCount() << ",\n";
    out << "    \"total_solve_s\": " << igr->GetTotalLinearSolveSeconds()
        << ",\n";
    out << "    \"total_igr_work_s\": " << igr->GetTotalIGRWorkSeconds()
        << ",\n";
    out << "    \"total_update_s\": " << igr->GetTotalUpdateCallSeconds()
        << ",\n";
    out << "    \"linear_solve_fraction\": "
        << ((elapsed_seconds > 0.0)
              ? (igr->GetTotalLinearSolveSeconds() / elapsed_seconds)
              : 0.0)
        << ",\n";
    out << "    \"total_igr_work_fraction\": "
        << ((elapsed_seconds > 0.0)
              ? (igr->GetTotalIGRWorkSeconds() / elapsed_seconds)
              : 0.0)
        << ",\n";
    out << "    \"linear_solve_fraction_stage\": "
        << ((igr->GetTotalUpdateCallSeconds() > 0.0)
              ? (igr->GetTotalLinearSolveSeconds() /
                 igr->GetTotalUpdateCallSeconds())
              : 0.0)
        << ",\n";
    out << "    \"total_igr_work_fraction_stage\": "
        << ((igr->GetTotalUpdateCallSeconds() > 0.0)
              ? (igr->GetTotalIGRWorkSeconds() /
                 igr->GetTotalUpdateCallSeconds())
              : 0.0)
        << ",\n";
    out << "    \"last_solver\": "
        << detail::ToJsonString(igr->GetLastLinearSolverName()) << ",\n";
    out << "    \"last_precond\": "
        << detail::ToJsonString(igr->GetLastPrecondName()) << ",\n";
    out << "    \"last_assemble_s\": " << igr->GetLastAssembleSeconds()
        << ",\n";
    out << "    \"last_form_system_s\": " << igr->GetLastFormSystemSeconds()
        << ",\n";
    out << "    \"last_pc_setup_s\": " << igr->GetLastPCSetupSeconds()
        << ",\n";
    out << "    \"last_cheb_diag_s\": "
        << igr->GetLastChebyshevDiagonalSeconds() << ",\n";
    out << "    \"last_cheb_setup_s\": "
        << igr->GetLastChebyshevSetupSeconds() << ",\n";
    out << "    \"last_cheb_eigen_reused\": "
        << (igr->GetLastChebyshevEigenvalueReused() ? "true" : "false")
        << ",\n";
    out << "    \"last_cheb_smoother_reused\": "
        << (igr->GetLastChebyshevSmootherReused() ? "true" : "false")
        << ",\n";
    out << "    \"last_cheb_eigen_estimate\": "
        << igr->GetLastChebyshevEigenvalueEstimate() << ",\n";
    out << "    \"last_krylov_operator_apply_s\": "
        << igr->GetLastKrylovOperatorApplySeconds() << ",\n";
    out << "    \"last_krylov_preconditioner_apply_s\": "
        << igr->GetLastKrylovPreconditionerApplySeconds() << ",\n";
    out << "    \"last_krylov_preconditioner_operator_apply_s\": "
        << igr->GetLastKrylovPreconditionerOperatorApplySeconds() << ",\n";
    out << "    \"last_krylov_operator_apply_count\": "
        << igr->GetLastKrylovOperatorApplyCount() << ",\n";
    out << "    \"last_krylov_preconditioner_apply_count\": "
        << igr->GetLastKrylovPreconditionerApplyCount() << ",\n";
    out << "    \"last_krylov_preconditioner_operator_apply_count\": "
        << igr->GetLastKrylovPreconditionerOperatorApplyCount() << ",\n";
    out << "    \"last_iters\": " << igr->GetLastLinearIters() << ",\n";
    out << "    \"last_converged\": "
        << (igr->GetLastLinearConverged() ? "true" : "false") << ",\n";
    out << "    \"last_initial_norm\": " << igr->GetLastLinearInitialNorm()
        << ",\n";
    out << "    \"last_final_norm\": " << igr->GetLastLinearFinalNorm()
        << ",\n";
    out << "    \"last_solve_s\": " << igr->GetLastLinearSolveSeconds()
        << ",\n";
    out << "    \"direct_identity_system\": {\n";
    out << "      \"effective\": "
        << (igr->GetLastDirectIdentitySystem() ? "true" : "false") << ",\n";
    out << "      \"reason\": "
        << detail::ToJsonString(igr->GetLastDirectIdentityReason()) << ",\n";
    out << "      \"uses\": " << igr->GetDirectIdentitySystemUses() << "\n";
    out << "    },\n";
    out << "    \"exact_face\": {\n";
    out << "      \"requested\": "
        << (igr->GetLastExactFaceRequested() ? "true" : "false") << ",\n";
    out << "      \"effective\": "
        << (igr->GetLastExactFaceEffective() ? "true" : "false") << ",\n";
    out << "      \"reason\": "
        << detail::ToJsonString(igr->GetLastExactFaceReason()) << ",\n";
    out << "      \"operator_uses\": "
        << igr->GetExactFaceOperatorUseCount() << ",\n";
    out << "      \"local_faces\": "
        << igr->GetLastExactFaceLocalFaceCount() << ",\n";
    out << "      \"shared_faces\": "
        << igr->GetLastExactFaceSharedFaceCount() << ",\n";
    out << "      \"last_operator_mult_calls\": "
        << igr->GetLastExactFaceOperatorMultCalls() << "\n";
    out << "    }\n";
  }
  else
  {
    out << "\n";
  }
  out << "  },\n";

  out << "  \"command\": {\n";
  out << "    \"argv\": " << detail::ToJsonString(detail::JoinCLI(argc, argv))
      << "\n";
  out << "  }\n";
  out << "}\n";
}

} // namespace dg_sim

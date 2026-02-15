#pragma once

/**
 * @file sim/io/run_metadata.hpp
 * @brief Runtime metadata sidecar writer for VTK datasets.
 */

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "mfem.hpp"

#include "physics/igr/IGR.hpp"
#include "sim/config/case_config.hpp"
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
                                    bool success)
{
  if (!run_cfg.opts.vtk_save || !mfem::Mpi::Root())
  {
    return;
  }

  const mfem::io::VTKOutputConfig vtk_cfg = MakeCaseVTKConfig(case_cfg);
  const std::string dataset_name =
    MakeCaseVTKDatasetName(case_cfg, run_cfg, c_alpha_value);
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
  out << "    \"mpi_ranks\": " << mfem::Mpi::WorldSize() << ",\n";
  out << "    \"order\": " << run_cfg.opts.order << ",\n";
  out << "    \"serial_refinement\": " << run_cfg.opts.ser_ref_levels << ",\n";
  out << "    \"parallel_refinement\": " << run_cfg.opts.par_ref_levels
      << ",\n";
  out << "    \"dg_backend\": "
      << detail::ToJsonString(DGBackendToString(run_cfg.dg_backend)) << ",\n";
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
  out << "    \"c_alpha\": " << c_alpha_value << ",\n";
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
  out << "    \"igr_ip_kind\": "
      << detail::ToJsonString(run_cfg.igr_opts.ip_kind) << ",\n";
  out << "    \"cfl\": " << run_cfg.opts.cfl << ",\n";
  out << "    \"dt\": " << run_cfg.opts.dt << ",\n";
  out << "    \"dt_min\": " << run_cfg.opts.dt_min << ",\n";
  out << "    \"t_final\": " << run_cfg.opts.t_final << ",\n";
  out << "    \"nan_scan\": " << (run_cfg.opts.nan_scan ? "true" : "false")
      << ",\n";
  out << "    \"save_state_files\": "
      << (run_cfg.opts.save_state_files ? "true" : "false") << "\n";
  out << "  },\n";

  out << "  \"performance\": {\n";
  if (summary)
  {
    out << "    \"steps\": " << summary->steps << ",\n";
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
    out << "    \"elapsed\": 0.0,\n";
    out << "    \"avg_step\": 0.0,\n";
    out << "    \"avg_step_per_dof_local\": 0.0,\n";
    out << "    \"avg_step_per_dof_rank\": 0.0,\n";
    out << "    \"local_dofs\": 0,\n";
    out << "    \"global_dofs\": 0,\n";
    out << "    \"ranks\": " << mfem::Mpi::WorldSize() << "\n";
  }
  out << "  },\n";

  out << "  \"igr_linear\": {\n";
  out << "    \"enabled\": " << ((igr != nullptr) ? "true" : "false");
  if (igr)
  {
    out << ",\n";
    out << "    \"calls\": " << igr->GetLinearSolveCalls() << ",\n";
    out << "    \"converged_calls\": " << igr->GetTotalLinearConvergedCalls()
        << ",\n";
    out << "    \"avg_iters\": " << igr->GetAverageLinearIterations() << ",\n";
    out << "    \"total_solve_s\": " << igr->GetTotalLinearSolveSeconds()
        << ",\n";
    out << "    \"last_solver\": "
        << detail::ToJsonString(igr->GetLastLinearSolverName()) << ",\n";
    out << "    \"last_precond\": "
        << detail::ToJsonString(igr->GetLastPrecondName()) << ",\n";
    out << "    \"last_iters\": " << igr->GetLastLinearIters() << ",\n";
    out << "    \"last_converged\": "
        << (igr->GetLastLinearConverged() ? "true" : "false") << ",\n";
    out << "    \"last_initial_norm\": " << igr->GetLastLinearInitialNorm()
        << ",\n";
    out << "    \"last_final_norm\": " << igr->GetLastLinearFinalNorm()
        << ",\n";
    out << "    \"last_solve_s\": " << igr->GetLastLinearSolveSeconds() << "\n";
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

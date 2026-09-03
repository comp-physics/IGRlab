#pragma once

/**
 * @file sim/time/time_callbacks.hpp
 * @brief Callback interfaces and payloads for time-loop diagnostics.
 */

#include <functional>
#include <string>

namespace dg_sim
{

/// @brief Per-step timing/metric payload for alive callbacks.
struct TimeLoopInfo
{
  int step = 0;
  double t = 0.0;
  double dt = 0.0;
  double elapsed = 0.0;
  double avg_step = 0.0;
  double avg_step_per_dof_local = 0.0;
  double avg_step_per_dof_rank = 0.0;
  int local_dofs = 0;
  long long global_dofs = 0;
  int ranks = 1;
  bool is_final = false;
  bool has_igr = false;
  int igr_linear_calls = 0;
  int igr_linear_converged_calls = 0;
  int igr_output_only_linear_calls = 0;
  int igr_output_only_linear_converged_calls = 0;
  double igr_linear_avg_iters = 0.0;
  double igr_linear_solve_seconds = 0.0;
  double igr_linear_solve_fraction = 0.0;
  double igr_assemble_seconds = 0.0;
  double igr_form_system_seconds = 0.0;
  double igr_pc_setup_seconds = 0.0;
  double igr_total_work_seconds = 0.0;
  double igr_total_work_fraction = 0.0;
  double igr_output_only_work_seconds = 0.0;
};

/// @brief Analysis payload with optional min/max diagnostics.
struct AnalysisInfo
{
  TimeLoopInfo base;
  bool has_rho = false;
  double rho_min = 0.0;
  double rho_max = 0.0;
  bool has_p = false;
  double p_min = 0.0;
  double p_max = 0.0;
};

/// @brief VTK save event payload.
struct VTKSaveInfo
{
  int cycle = 0;
  int step = 0;
  double t = 0.0;
  bool is_initial = false;
  bool is_final = false;
};

/// @brief End-of-run summary payload.
struct RunSummary
{
  int steps = 0;
  int segment_steps = 0;
  int accepted_attempts = 0;
  int rejected_attempts = 0;
  /// Sum of failed element admissions over rejected limiter attempts.
  long long rejected_limiter_elements = 0;
  long long rhs_evaluations = 0;
  double initial_time = 0.0;
  double final_time = 0.0;
  double minimum_attempted_dt = 0.0;
  double minimum_accepted_dt = 0.0;
  double maximum_attempted_dt = 0.0;
  double maximum_accepted_dt = 0.0;
  bool dt_min_reached = false;
  bool success = false;
  std::string termination_reason = "not-started";
  double elapsed = 0.0;
  double avg_step = 0.0;
  double avg_step_per_dof_local = 0.0;
  double avg_step_per_dof_rank = 0.0;
  int local_dofs = 0;
  long long global_dofs = 0;
  int ranks = 1;
  bool has_igr = false;
  int igr_linear_calls = 0;
  int igr_linear_converged_calls = 0;
  int igr_output_only_linear_calls = 0;
  int igr_output_only_linear_converged_calls = 0;
  double igr_linear_avg_iters = 0.0;
  double igr_linear_solve_seconds = 0.0;
  double igr_linear_solve_fraction = 0.0;
  double igr_assemble_seconds = 0.0;
  double igr_form_system_seconds = 0.0;
  double igr_pc_setup_seconds = 0.0;
  double igr_total_work_seconds = 0.0;
  double igr_total_work_fraction = 0.0;
  double igr_output_only_work_seconds = 0.0;
};

/// @brief Callback bundle for time-loop diagnostics and logging.
struct TimeLoopCallbacks
{
  std::function<void(const TimeLoopInfo&)> on_alive;
  std::function<void(const AnalysisInfo&)> on_analysis;
  std::function<void(const VTKSaveInfo&)> on_vtk_save;
  std::function<void(const RunSummary&)> on_summary;
  bool wants_minmax = false;
};

/// @brief Configuration for the default callback builder.
struct DefaultCallbackConfig
{
  bool enable_alive = true;
  bool enable_analysis = true;
  bool enable_vtk = true;
  bool enable_summary = true;
  bool compute_minmax = true;
};

} // namespace dg_sim

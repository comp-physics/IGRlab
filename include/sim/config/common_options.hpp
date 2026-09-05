#pragma once

/**
 * @file sim/config/common_options.hpp
 * @brief Shared CLI options for sim-layer examples.
 */

#include <string>

#include "mfem.hpp"
#include "utils/config/env_flags.hpp"

namespace dg_sim
{

/**
 * @brief Common runtime options shared by the example drivers.
 */
struct CommonOptions
{
  std::string
    mesh_file; ///< Optional mesh override; empty means "use the case default".
  int ser_ref_levels = 0;        ///< Serial uniform refinement levels.
  int par_ref_levels = 1;        ///< Parallel uniform refinement levels.
  int order = 3;                 ///< DG polynomial order.
  std::string device = "cpu";    ///< Runtime device backend: "cpu" or "cuda".
  std::string dg_backend = "dg"; ///< DG backend: "dg" or "dgsem".
  std::string discretization;
  ///< General discretization selector; empty retains the per-case default.
  std::string regularization;
  ///< Regularization selector: none | igr; empty retains the case default.
  std::string hdg_cuda_engine = "ordinary";
  ///< CUDA HDG engine: ordinary | tensor; ordinary remains the default.
  double hdg_igr_alpha = 1.0;
  ///< Dimensionless HDG IGR C_alpha, mesh-scaled by the case length policy.
  double hdg_igr_beta = 1.0;
  ///< Numerical Euler U4 trace penalty used by HDG IGR; not a physical scalar.
  std::string time_integrator;
  ///< Time-integrator selector; empty selects the discretization default.
  mfem::real_t time_atol = 1.0e-8;
  ///< Differential-state absolute tolerance for embedded adaptive stepping.
  mfem::real_t time_rtol = 1.0e-5;
  ///< Differential-state relative tolerance for embedded adaptive stepping.
  std::string positivity_limiter = "case";
  ///< Positivity limiter selector: case | none | zhang-shu.
  std::string zhang_shu_points =
    "consumer"; ///< Zhang-Shu point set: "consumer" or "quadrature".
  bool zhang_shu_vtk_points = false;
  ///< Also enforce positivity at the refined points used for VTK output.
  std::string dgsem_stab =
    "none"; ///< DGSEM stabilization: "none", "split", or "filter".
  std::string dgsem_volume_flux =
    "central"; ///< DGSEM volume flux: "central", "chandrashekar", or "ranocha".
  std::string dgsem_volume_dissipation =
    "none"; ///< DGSEM volume dissipation: "none", "llf", or "entropy-lf".
  std::string dgsem_surface_flux =
    "rusanov"; ///< DGSEM surface flux: "rusanov", "es-winters", or "es-lf-entropy".
  std::string dgsem_entropy_jacobian =
    "fd"; ///< DGSEM ES-LF entropy Jacobian: "fd" or "analytic".
  mfem::real_t t_final = 1.0; ///< Final time.
  /// Fixed explicit step when cfl is zero; HDG also uses its sign to select
  /// fixed, CFL-controlled, or adaptive stepping.
  mfem::real_t dt = -0.01;
  mfem::real_t dt_min = 1.0e-12; ///< Minimum allowed dt before aborting.
  /// Explicit SSPRK safety factor, or the legacy HDG stable-step multiplier.
  mfem::real_t cfl = 0.3;
  bool glvis = false;            ///< Enable GLVis visualization.
  bool vtk_save = true;          ///< Enable ParaView/VTK output.
  bool vtk_h1_diagnostics =
    true; ///< Enable optional H1 diagnostic fields when the case provides them.
  bool vtk_l2_diagnostics =
    true; ///< Enable optional L2 diagnostic fields when the case provides them.
  bool vtk_mpi_rank_field =
    true; ///< Enable optional per-element MPI rank VTK field.
  bool vtk_zhang_shu_diagnostics =
    true; ///< Enable optional Zhang-Shu limiter diagnostic VTK fields.
  bool save_state_files =
    false; ///< Save GLVis-compatible init/final snapshots to disk.
  mfem::real_t dt_save = 0.01; ///< Save cadence for ParaView output.
  bool preassemble_weak_div =
    true;             ///< Element-assembly vs matrix-free divergence.
  int vis_steps = 50; ///< Visualization cadence in steps.
  int analysis_steps =
    -1; ///< Analysis cadence in steps (-1 = 10x vis_steps, 0 disables).
  bool nan_scan = false; ///< Enable NaN/inf scanning during the time loop.
  bool admissibility_scan =
    false; ///< Enable Euler density/pressure admissibility scans during the time loop.
  bool conservation_balance =
    false; ///< Enable bounded accepted-step Euler conservation diagnostics.
  std::string restart_vtk; ///< PVD file or dataset directory to restart from.
  int restart_cycle = -1;  ///< Selected PVD cycle; -1 selects the latest cycle.
  int restart_step = -1;   ///< Completed global step for legacy VTK cycles.
  std::string hdg_checkpoint;
  ///< Optional rolling committed-boundary HDG checkpoint artifact.
  std::string hdg_restart_checkpoint;
  ///< Optional HDG checkpoint artifact to restore before time integration.
};

/**
 * @brief Register the common example options with MFEM's OptionsParser.
 *
 * @param[in,out] args         Options parser.
 * @param[in,out] opts         Options struct to populate.
 * @param[in]     mesh_help    Help text describing the default mesh.
 */
inline void AddCommonOptions(mfem::OptionsParser& args,
                             CommonOptions& opts,
                             const std::string& mesh_help)
{
  // Preserve the historical environment-only control as a default seed.
  // Either typed CLI spelling registered below remains authoritative.
  opts.zhang_shu_vtk_points = dg_igr::EnvFlagEnabledDefault(
    "DG_IGR_LIMITER_ENFORCE_VTK_POINTS", opts.zhang_shu_vtk_points);

  args.AddOption(&opts.mesh_file, "-m", "--mesh", mesh_help.c_str());
  args.AddOption(&opts.ser_ref_levels,
                 "-rs",
                 "--serial-refine",
                 "Number of times to refine the serial mesh uniformly.");
  args.AddOption(&opts.par_ref_levels,
                 "-rp",
                 "--parallel-refine",
                 "Number of times to refine the parallel mesh uniformly.");
  args.AddOption(&opts.order,
                 "-o",
                 "--order",
                 "Order (degree) of the finite elements.");
  args.AddOption(&opts.device,
                 "-dev",
                 "--device",
                 "Runtime device backend: cpu | cuda.");
  args.AddOption(&opts.dg_backend,
                 "-dg",
                 "--dg-backend",
                 "DG backend: dg | dgsem (default set per case).");
  args.AddOption(&opts.discretization,
                 "-disc",
                 "--discretization",
                 "Discretization: dg | dgsem | hdg (case support required).");
  args.AddOption(&opts.regularization,
                 "-reg",
                 "--regularization",
                 "Regularization: none | igr (case support required).");
  args.AddOption(&opts.hdg_cuda_engine,
                 "-hce",
                 "--hdg-cuda-engine",
                 "CUDA HDG engine: ordinary | tensor (ordinary is the default; "
                 "tensor requires --discretization hdg --device cuda).");
  args.AddOption(&opts.hdg_igr_alpha,
                 "-hia",
                 "--hdg-igr-alpha",
                 "Dimensionless HDG IGR C_alpha; the case mesh-length policy "
                 "sets the physical alpha=C_alpha*h_eff^2 (default 1.0).");
  args.AddOption(&opts.hdg_igr_beta,
                 "-hib",
                 "--hdg-igr-beta",
                 "HDG numerical Euler U4 trace penalty (finite and positive; "
                 "not a second physical IGR scalar; default 1.0).");
  args.AddOption(&opts.time_integrator,
                 "-ti",
                 "--time-integrator",
                 "Time integrator: rk3ssp | ssprk104 | rk4 | dopri5 | radau-picard5 | "
                 "backward-euler | partitioned-backward-euler | rodas3 "
                 "(discretization support required).");
  args.AddOption(&opts.time_atol,
                 "-tatol",
                 "--time-atol",
                 "Absolute differential-state tolerance for adaptive RODAS3.");
  args.AddOption(&opts.time_rtol,
                 "-trtol",
                 "--time-rtol",
                 "Relative differential-state tolerance for adaptive RODAS3.");
  args.AddOption(&opts.positivity_limiter,
                 "-pl",
                 "--positivity-limiter",
                 "Positivity limiter: case | none | zhang-shu. 'case' "
                 "preserves the case default when the selected integrator "
                 "supports limiting; HDG limiting is RODAS3-only.");
  args.AddOption(&opts.zhang_shu_points,
                 "-zsp",
                 "--zhang-shu-points",
                 "Zhang-Shu positivity points: consumer | quadrature.");
  args.AddOption(&opts.zhang_shu_vtk_points,
                 "-zsvtk",
                 "--zhang-shu-vtk-points",
                 "-no-zsvtk",
                 "--no-zhang-shu-vtk-points",
                 "Enable or disable Zhang-Shu enforcement at the refined "
                 "points used for VTK output (default off).");
  args.AddOption(
    &opts.dgsem_stab,
    "-ds",
    "--dgsem-stab",
    "DGSEM stabilization: none | split | filter (not yet implemented).");
  args.AddOption(&opts.dgsem_volume_flux,
                 "-dvf",
                 "--dgsem-volume-flux",
                 "DGSEM volume flux: central | chandrashekar | ranocha.");
  args.AddOption(&opts.dgsem_volume_dissipation,
                 "-dvd",
                 "--dgsem-volume-dissipation",
                 "DGSEM volume dissipation: none | llf | entropy-lf.");
  args.AddOption(&opts.dgsem_surface_flux,
                 "-dsf",
                 "--dgsem-surface-flux",
                 "DGSEM surface flux: rusanov | diffusive-rusanov | es-winters "
                 "| es-lf-entropy.");
  args.AddOption(
    &opts.dgsem_entropy_jacobian,
    "-dej",
    "--dgsem-entropy-jacobian",
    "DGSEM entropy Jacobian: fd | analytic (ES LF entropy variables).");
  args.AddOption(&opts.t_final,
                 "-tf",
                 "--t-final",
                 "Final time; start time is 0.");
  args.AddOption(&opts.dt,
                 "-dt",
                 "--time-step",
                 "Time-step input. For explicit SSPRK, it is used only when "
                 "--cfl-number is zero; a positive CFL safety factor takes "
                 "precedence. For HDG, positive selects fixed stepping, "
                 "nonpositive selects CFL for Backward Euler, and negative "
                 "supplies the initial adaptive RODAS3 step magnitude.");
  args.AddOption(&opts.dt_min,
                 "-dtmin",
                 "--dt-min",
                 "Minimum allowed dt; run aborts if dt falls below this "
                 "threshold.");
  args.AddOption(&opts.cfl,
                 "-c",
                 "--cfl-number",
                 "Explicit SSPRK CFL safety factor theta in (0,1], applied as "
                 "dt=theta*C_SSP*h_min/((2p+1)*lambda_max); zero selects the "
                 "fixed --time-step. This normalization does not certify a "
                 "case's spatial stability limit. For CFL-controlled HDG "
                 "Backward Euler, this remains the backend stable-step "
                 "multiplier.");
  args.AddOption(&opts.glvis,
                 "-glvis",
                 "--glvis",
                 "-no-glvis",
                 "--no-glvis",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&opts.vtk_save,
                 "-vtk",
                 "--vtk_save",
                 "-no-vtk",
                 "--no-vtk_save",
                 "Enable or disable VTK visualization.");
  args.AddOption(&opts.vtk_h1_diagnostics,
                 "-vtk-h1",
                 "--vtk-h1-diagnostics",
                 "-no-vtk-h1",
                 "--no-vtk-h1-diagnostics",
                 "Enable or disable optional H1 diagnostic VTK fields.");
  args.AddOption(&opts.vtk_l2_diagnostics,
                 "-vtk-l2",
                 "--vtk-l2-diagnostics",
                 "-no-vtk-l2",
                 "--no-vtk-l2-diagnostics",
                 "Enable or disable optional L2 diagnostic VTK fields.");
  args.AddOption(&opts.vtk_mpi_rank_field,
                 "-vtk-rank",
                 "--vtk-mpi-rank-field",
                 "-no-vtk-rank",
                 "--no-vtk-mpi-rank-field",
                 "Enable or disable optional per-element MPI rank VTK field.");
  args.AddOption(
    &opts.vtk_zhang_shu_diagnostics,
    "-vtk-zs",
    "--vtk-zhang-shu-diagnostics",
    "-no-vtk-zs",
    "--no-vtk-zhang-shu-diagnostics",
    "Enable or disable optional Zhang-Shu limiter diagnostic VTK fields.");
  args.AddOption(&opts.save_state_files,
                 "-save-state",
                 "--save-state",
                 "-no-save-state",
                 "--no-save-state",
                 "Enable or disable GLVis-compatible init/final state file "
                 "dumps.");
  args.AddOption(&opts.dt_save,
                 "-dt_save",
                 "--time-step-save",
                 "Saving time step for VTK output.");
  args.AddOption(&opts.preassemble_weak_div,
                 "-ea",
                 "--element-assembly-divergence",
                 "-mf",
                 "--matrix-free-divergence",
                 "Weak divergence assembly level\n"
                 "    ea - Element assembly with interpolated F\n"
                 "    mf - Nonlinear assembly in matrix-free manner");
  args.AddOption(&opts.vis_steps,
                 "-vs",
                 "--visualization-steps",
                 "Visualize every n-th timestep.");
  args.AddOption(
    &opts.analysis_steps,
    "-as",
    "--analysis-steps",
    "Print analysis every n-th timestep (0 disables, -1 uses 10x -vs).");
  args.AddOption(&opts.nan_scan,
                 "-nan-scan",
                 "--nan-scan",
                 "-no-nan-scan",
                 "--no-nan-scan",
                 "Enable or disable NaN/inf scanning during time integration "
                 "(per-stage + per-step).");
  args.AddOption(&opts.admissibility_scan,
                 "-adm-scan",
                 "--admissibility-scan",
                 "-no-adm-scan",
                 "--no-admissibility-scan",
                 "Enable or disable Euler density/pressure admissibility "
                 "scans during time integration.");
  args.AddOption(&opts.conservation_balance,
                 "-conservation-balance",
                 "--conservation-balance",
                 "-no-conservation-balance",
                 "--no-conservation-balance",
                 "Enable or disable bounded accepted-step Euler physical-"
                 "boundary conservation diagnostics. This qualification "
                 "mode adds reductions and is excluded from performance "
                 "timings.");
  args.AddOption(&opts.restart_vtk,
                 "-restart-vtk",
                 "--restart-vtk",
                 "Restart from an MFEM ParaView PVD file or dataset directory.");
  args.AddOption(&opts.restart_cycle,
                 "-restart-cycle",
                 "--restart-cycle",
                 "Restart cycle (-1 selects the latest complete cycle).");
  args.AddOption(&opts.restart_step,
                 "-restart-step",
                 "--restart-step",
                 "Completed global RK step for legacy cycles without a manifest.");
  args.AddOption(&opts.hdg_checkpoint,
                 "-hdg-checkpoint",
                 "--hdg-checkpoint",
                 "Rolling committed-boundary HDG checkpoint artifact path; "
                 "empty disables checkpoint publication.");
  args.AddOption(&opts.hdg_restart_checkpoint,
                 "-hdg-restart-checkpoint",
                 "--hdg-restart-checkpoint",
                 "Restore a committed HDG checkpoint artifact before time "
                 "integration; empty disables checkpoint restore.");
}

/**
 * @brief Resolve the mesh path by falling back to the case default.
 */
inline std::string ResolveMeshPath(const std::string& default_mesh,
                                   const std::string& override_mesh)
{
  return override_mesh.empty() ? default_mesh : override_mesh;
}

} // namespace dg_sim

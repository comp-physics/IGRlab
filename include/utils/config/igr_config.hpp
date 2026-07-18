#pragma once

#include "utils/config/dg_igr_defaults.hpp"

/**
 * @file utils/config/igr_config.hpp
 * @brief Configuration enums and defaults for the IGR solver.
 * @ingroup utils_config
 */

namespace dg_igr
{

/// @brief Smoother types used by the IGR solver.
/// @ingroup utils_config
enum class IGRSmootherType
{
  Jacobi,
  L1Jacobi,
  L1GS,
  L1GStr,
  GS
};

/// @brief Gradient construction backend for the IGR RHS.
/// @ingroup utils_config
enum class IGRGradBackend
{
  H1Projection,
  BrokenDG /*, WeakRT, BR2 */
};

/// @brief Interior-penalty DG flavor for diffusion on DG spaces.
/// @ingroup utils_config
enum class IGRIPKind
{
  SIPG = -1,
  IIPG = 0,
  NIPG = +1
};

/// @brief Linear system solve mode for IGR updates.
/// @ingroup utils_config
enum class IGRSolveMode
{
  SmootherSweeps,
  Krylov
};

/// @brief Krylov solver selection for IGR solves.
/// @ingroup utils_config
enum class IGRLinearSolverType
{
  Auto,
  CG,
  GMRES,
  FGMRES,
  BiCGSTAB,
  MINRES
};

/// @brief Preconditioner selection for IGR solves.
/// @ingroup utils_config
enum class IGRPrecondType
{
  Auto,
  BoomerAMG,
  HypreILU,
  HypreEuclid,
  ParaSails,
  HypreSmoother,
  OperatorChebyshev,
  None
};

/// @brief Policy applied when an IGR Krylov solve reaches its iteration cap.
enum class IGRNonconvergencePolicy
{
  Error,
  Accept
};

/// @brief Cadence policy for sigma updates in IGR-coupled Euler solves.
/// @ingroup utils_config
enum class IGRSigmaUpdateCadence
{
  EveryStage,
  EveryStep,
  EveryNSteps
};

/// @brief Policy for converting mesh size into the IGR alpha length scale.
/// @ingroup utils_config
enum class IGRAlphaLengthScalePolicy
{
  Min,
  Average,
  Max,
  Local
};

/// @brief Configuration bundle for IGR defaults and runtime tuning.
/// @ingroup utils_config
struct IGRConfig
{
  double alpha = mfem::dg_igr::IGRDefaults::kAlpha;
  double rho_floor = mfem::dg_igr::IGRDefaults::kRhoFloor;
  IGRAlphaLengthScalePolicy alpha_length_scale_policy =
    IGRAlphaLengthScalePolicy::Average;

  IGRSmootherType smoot = IGRSmootherType::L1Jacobi;
  IGRGradBackend backend = IGRGradBackend::H1Projection;
  IGRIPKind ip_kind = IGRIPKind::SIPG;

  int regular_sweeps = mfem::dg_igr::IGRDefaults::kRegularSweeps;
  int warm_start_sweeps = mfem::dg_igr::IGRDefaults::kWarmStartSweeps;

  IGRSolveMode solve_mode = IGRSolveMode::Krylov;
  IGRLinearSolverType lin_solver_type = IGRLinearSolverType::Auto;
  IGRPrecondType prec_type = IGRPrecondType::Auto;
  IGRNonconvergencePolicy nonconvergence_policy =
    IGRNonconvergencePolicy::Accept;

  double lin_rtol = mfem::dg_igr::IGRDefaults::kLinRtol;
  double lin_atol = mfem::dg_igr::IGRDefaults::kLinAtol;
  int lin_maxit = mfem::dg_igr::IGRDefaults::kLinMaxIters;
  int lin_print = mfem::dg_igr::IGRDefaults::kLinPrint;

  // AMG tuning knobs. Negative values keep MFEM/HYPRE defaults.
  int amg_coarsen_type = -1;
  int amg_relax_type = -1;
  int amg_max_levels = -1;
  double amg_strong_threshold = -1.0;

  // Sigma update cadence for IGR coupling.
  IGRSigmaUpdateCadence sigma_cadence = IGRSigmaUpdateCadence::EveryStage;
  int sigma_update_interval = 1; // used only for EveryNSteps

  // Stage-wise tolerance policy (RK intermediate stages can be looser).
  double stage_rtol_scale = 1.0; // > 1.0 loosens stage-0/1 relative tolerance

  // Use the side-aware exact DG face wrapper for L2/DG IGR.
  bool exact_dg_face_operator = false;

};

} // namespace dg_igr

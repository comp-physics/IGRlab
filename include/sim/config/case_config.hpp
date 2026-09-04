#pragma once

/**
 * @file sim/config/case_config.hpp
 * @brief Case/Run configuration structures for the sim-layer.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "mfem.hpp"
#include "numerics/limiters/limiter_pipeline.hpp"
#include "utils/config/igr_config.hpp"
#include "utils/config/io_config.hpp"
#include "utils/config/timestepping_config.hpp"

#include "sim/config/common_options.hpp"
#include "sim/time/time_loop.hpp"

namespace dg_sim
{

/// @brief Options controlling the IGR discretization wiring in examples.
struct IGROptions
{
  double c_alpha_user = -1.0;
  std::string alpha_length_policy = "";
  std::string space = "H1";
  std::string grad = "H1";
  std::string solve_mode = "krylov";
  std::string linear_solver = "auto";
  std::string preconditioner = "auto";
  std::string nonconvergence_policy = "";
  std::string ip_kind = "sipg";
  std::string sigma_cadence = "";
  int sigma_update_interval = -1;
  double stage_rtol_scale = -1.0;
  double lin_rtol = -1.0;
  double lin_atol = -1.0;
  int lin_maxit = -1;
  int lin_print = std::numeric_limits<int>::min();
  int amg_coarsen_type = -1;
  int amg_relax_type = -1;
  int amg_max_levels = -1;
  double amg_strong_threshold = -1.0;
  bool exact_dg_face_operator = false;
  bool igr_invariant_domain_flux = false;
  bool sigma_aware_cfl = false;
  bool sigma_aware_rusanov = false;
};

/// @brief DG backend selection.
enum class DGBackend
{
  DG,
  DGSEM
};

/// @brief General discretization selection, including unavailable seams.
enum class Discretization
{
  DG,
  DGSEM,
  HDG
};

/// @brief Runtime compute device selection.
enum class DeviceBackend
{
  CPU,
  CUDA
};

/// @brief Explicit CUDA execution engine for the generic HDG lifecycle.
enum class HdgCudaEngine
{
  Ordinary,
  TensorCore
};

/// @brief Regularization selection for the Euler solver request.
enum class Regularization
{
  None,
  IGR
};

/// @brief Time-integrator selection for the Euler solver request.
enum class TimeIntegrator
{
  RK3SSP = 0,
  BackwardEuler = 1,
  PartitionedBackwardEuler = 2,
  Rodas3 = 3,
  SSPRK104 = 4
};

/// @brief Source of the effective general discretization selection.
enum class DiscretizationProvenance
{
  CaseDefault,
  DiscretizationOption,
  DGBackendOption,
  BothOptions
};

/// @brief Outcome of reconciling the general and legacy CLI selectors.
enum class DiscretizationResolutionStatus
{
  Resolved,
  InvalidDiscretization,
  InvalidDGBackend,
  Conflict,
  Unavailable
};

struct DiscretizationResolution
{
  Discretization discretization = Discretization::DG;
  DiscretizationProvenance provenance =
    DiscretizationProvenance::CaseDefault;
  std::optional<DGBackend> dg_backend = DGBackend::DG;
  DiscretizationResolutionStatus status =
    DiscretizationResolutionStatus::Resolved;
  std::string message;

  bool IsResolved() const
  {
    return status == DiscretizationResolutionStatus::Resolved;
  }
};

/// @brief Resolution status for the complete CLI/case solver request.
enum class SolverRequestResolutionStatus
{
  Resolved,
  InvalidDiscretization,
  InvalidDGBackend,
  InvalidDeviceBackend,
  InvalidHdgCudaEngine,
  Conflict,
  InvalidRegularization,
  InvalidTimeIntegrator,
  UnsupportedHdgCudaEngine,
  UnsupportedRegularization,
  UnsupportedTimeIntegrator
};

/// @brief Pure resolved request consumed before runtime construction begins.
struct SolverRequestResolution
{
  Discretization discretization = Discretization::DG;
  DiscretizationProvenance discretization_provenance =
    DiscretizationProvenance::CaseDefault;
  std::optional<DGBackend> dg_backend = DGBackend::DG;
  DeviceBackend device_backend = DeviceBackend::CPU;
  HdgCudaEngine hdg_cuda_engine = HdgCudaEngine::Ordinary;
  Regularization regularization = Regularization::None;
  TimeIntegrator time_integrator = TimeIntegrator::RK3SSP;
  SolverRequestResolutionStatus status =
    SolverRequestResolutionStatus::Resolved;
  std::string message;

  bool IsResolved() const
  {
    return status == SolverRequestResolutionStatus::Resolved;
  }
};

/// @brief Optional DGSEM stabilization hook selection.
enum class DGSEMStabilization
{
  None,
  SplitForm,
  Filter
};

/// @brief DGSEM volume flux selection for entropy-conservative options.
enum class DGSEMVolumeFlux
{
  Central,
  Chandrashekar,
  Ranocha
};

/// @brief DGSEM volume dissipation selection.
enum class DGSEMVolumeDissipation
{
  None,
  LLF,
  EntropyVarsLF
};

/// @brief DGSEM surface flux selection.
enum class DGSEMSurfaceFlux
{
  Rusanov,
  DiffusiveRusanov,
  EntropyStableWinters,
  EntropyStableLFEntropyVars
};

/// @brief Entropy Jacobian evaluation mode for ES LF surface flux.
enum class DGSEMEntropyJacobian
{
  FiniteDifference,
  Analytic
};

/// @brief Per-case configuration defaults plus case identity metadata.
struct CaseConfig
{
  std::string case_name;
  std::string default_mesh_file;
  std::string case_dir;
  int int_order_offset = 1;
  mfem::real_t gamma = 1.4;
  CommonOptions defaults;
  mfem::RK3Config rk3_config;
  dg_igr::IGRConfig igr_config;
  mfem::LimiterPipelineConfig limiter_pipeline;
  mfem::io::VTKOutputConfig vtk_config;
};

/// @brief Resolved runtime configuration after CLI parsing.
struct RunConfig
{
  CaseConfig case_config;
  CommonOptions opts;
  IGROptions igr_opts;
  DeviceBackend device_backend = DeviceBackend::CPU;
  Discretization discretization = Discretization::DG;
  DiscretizationProvenance discretization_provenance =
    DiscretizationProvenance::CaseDefault;
  DGBackend dg_backend = DGBackend::DG;
  HdgCudaEngine hdg_cuda_engine = HdgCudaEngine::Ordinary;
  Regularization regularization = Regularization::IGR;
  TimeIntegrator time_integrator = TimeIntegrator::RK3SSP;
  DGSEMStabilization dgsem_stab = DGSEMStabilization::None;
  DGSEMVolumeFlux dgsem_volume_flux = DGSEMVolumeFlux::Central;
  DGSEMVolumeDissipation dgsem_volume_dissipation =
    DGSEMVolumeDissipation::None;
  DGSEMSurfaceFlux dgsem_surface_flux = DGSEMSurfaceFlux::Rusanov;
  DGSEMEntropyJacobian dgsem_entropy_jacobian =
    DGSEMEntropyJacobian::FiniteDifference;
  std::string mesh_path;
};

inline std::string ToLowerCopy(std::string value)
{
  std::transform(value.begin(),
                 value.end(),
                 value.begin(),
                 [](unsigned char c)
                 { return static_cast<char>(std::tolower(c)); });
  return value;
}

/// @brief Typed command-line override for the case Zhang--Shu policy.
enum class PositivityLimiterSelection
{
  Case,
  None,
  ZhangShu
};

inline PositivityLimiterSelection
ParsePositivityLimiterSelection(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower.empty() || lower == "case")
  {
    return PositivityLimiterSelection::Case;
  }
  if (lower == "none")
  {
    return PositivityLimiterSelection::None;
  }
  if (lower == "zhang-shu" || lower == "zhang_shu" || lower == "zhangshu")
  {
    return PositivityLimiterSelection::ZhangShu;
  }
  MFEM_VERIFY(false,
              "Invalid positivity limiter '"
                << value << "'. Use 'case', 'none', or 'zhang-shu'.");
  return PositivityLimiterSelection::Case;
}

inline const char* PositivityLimiterSelectionToString(
  const PositivityLimiterSelection selection)
{
  switch (selection)
  {
  case PositivityLimiterSelection::Case:
    return "case";
  case PositivityLimiterSelection::None:
    return "none";
  case PositivityLimiterSelection::ZhangShu:
    return "zhang-shu";
  }
  return "case";
}

inline bool ResolveZhangShuPositivityLimiter(
  const PositivityLimiterSelection selection,
  const bool case_default) noexcept
{
  switch (selection)
  {
  case PositivityLimiterSelection::Case:
    return case_default;
  case PositivityLimiterSelection::None:
    return false;
  case PositivityLimiterSelection::ZhangShu:
    return true;
  }
  return case_default;
}

inline bool ResolveZhangShuPositivityLimiter(
  const std::string& selection, const bool case_default)
{
  return ResolveZhangShuPositivityLimiter(
    ParsePositivityLimiterSelection(selection), case_default);
}

inline mfem::ZhangShuPointSet
ParseZhangShuPointSet(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "consumer")
  {
    return mfem::ZhangShuPointSet::kConsumer;
  }
  if (lower == "quadrature")
  {
    return mfem::ZhangShuPointSet::kQuadrature;
  }
  MFEM_VERIFY(false,
              "Invalid Zhang-Shu point set '"
                << value << "'. Use 'consumer' or 'quadrature'.");
  return mfem::ZhangShuPointSet::kConsumer;
}

inline const char*
ZhangShuPointSetToString(mfem::ZhangShuPointSet point_set)
{
  switch (point_set)
  {
  case mfem::ZhangShuPointSet::kQuadrature:
    return "quadrature";
  case mfem::ZhangShuPointSet::kConsumer:
    return "consumer";
  }
  return "consumer";
}

inline dg_igr::IGRIPKind ParseIGRIPKind(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "sipg")
  {
    return dg_igr::IGRIPKind::SIPG;
  }
  if (lower == "iipg")
  {
    return dg_igr::IGRIPKind::IIPG;
  }
  if (lower == "nipg")
  {
    return dg_igr::IGRIPKind::NIPG;
  }
  MFEM_VERIFY(false,
              "Invalid IGR IP kind '" << value
                                      << "'. Use 'sipg', 'iipg', or 'nipg'.");
  return dg_igr::IGRIPKind::SIPG;
}

inline dg_igr::IGRSolveMode ParseIGRSolveMode(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "smoother" || lower == "smoother-sweeps" || lower == "sweeps")
  {
    return dg_igr::IGRSolveMode::SmootherSweeps;
  }
  if (lower == "krylov")
  {
    return dg_igr::IGRSolveMode::Krylov;
  }
  MFEM_VERIFY(false,
              "Invalid IGR solve mode '" << value
                                         << "'. Use 'smoother' or 'krylov'.");
  return dg_igr::IGRSolveMode::Krylov;
}

inline dg_igr::IGRLinearSolverType
ParseIGRLinearSolverType(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "auto")
  {
    return dg_igr::IGRLinearSolverType::Auto;
  }
  if (lower == "cg")
  {
    return dg_igr::IGRLinearSolverType::CG;
  }
  if (lower == "gmres")
  {
    return dg_igr::IGRLinearSolverType::GMRES;
  }
  if (lower == "fgmres")
  {
    return dg_igr::IGRLinearSolverType::FGMRES;
  }
  if (lower == "bicgstab")
  {
    return dg_igr::IGRLinearSolverType::BiCGSTAB;
  }
  if (lower == "minres")
  {
    return dg_igr::IGRLinearSolverType::MINRES;
  }
  MFEM_VERIFY(false,
              "Invalid IGR linear solver '"
                << value << "'. Use 'auto', 'cg', 'gmres', 'fgmres', "
                << "'bicgstab', or 'minres'.");
  return dg_igr::IGRLinearSolverType::Auto;
}

inline dg_igr::IGRPrecondType ParseIGRPrecondType(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "auto")
  {
    return dg_igr::IGRPrecondType::Auto;
  }
  if (lower == "boomeramg" || lower == "amg")
  {
    return dg_igr::IGRPrecondType::BoomerAMG;
  }
  if (lower == "hypreilu" || lower == "ilu")
  {
    return dg_igr::IGRPrecondType::HypreILU;
  }
  if (lower == "hypreeuclid" || lower == "euclid")
  {
    return dg_igr::IGRPrecondType::HypreEuclid;
  }
  if (lower == "parasails" || lower == "hypreparasails")
  {
    return dg_igr::IGRPrecondType::ParaSails;
  }
  if (lower == "hypresmoother" || lower == "smoother")
  {
    return dg_igr::IGRPrecondType::HypreSmoother;
  }
  if (lower == "chebyshev" || lower == "operatorchebyshev" ||
      lower == "operator-chebyshev")
  {
    return dg_igr::IGRPrecondType::OperatorChebyshev;
  }
  if (lower == "none")
  {
    return dg_igr::IGRPrecondType::None;
  }
  MFEM_VERIFY(false,
              "Invalid IGR preconditioner '"
                << value << "'. Use 'auto', 'boomeramg', 'hypreilu', "
                << "'hypreeuclid', 'parasails', "
                << "'hypresmoother', 'chebyshev', or 'none'.");
  return dg_igr::IGRPrecondType::Auto;
}

inline dg_igr::IGRNonconvergencePolicy
ParseIGRNonconvergencePolicy(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "error" || lower == "fail")
  {
    return dg_igr::IGRNonconvergencePolicy::Error;
  }
  if (lower == "accept" || lower == "legacy")
  {
    return dg_igr::IGRNonconvergencePolicy::Accept;
  }
  MFEM_VERIFY(false,
              "Invalid IGR nonconvergence policy '"
                << value << "'. Use 'error' or 'accept'.");
  return dg_igr::IGRNonconvergencePolicy::Error;
}

inline dg_igr::IGRSigmaUpdateCadence
ParseIGRSigmaUpdateCadence(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "every-stage" || lower == "every_stage" || lower == "stage")
  {
    return dg_igr::IGRSigmaUpdateCadence::EveryStage;
  }
  if (lower == "every-step" || lower == "every_step" || lower == "step")
  {
    return dg_igr::IGRSigmaUpdateCadence::EveryStep;
  }
  if (lower == "extrapolated-every-step" ||
      lower == "extrapolated_every_step" || lower == "extrapolated-step")
  {
    return dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep;
  }
  if (lower == "every-n-steps" || lower == "every_n_steps" ||
      lower == "every-n" || lower == "nsteps")
  {
    return dg_igr::IGRSigmaUpdateCadence::EveryNSteps;
  }
  MFEM_VERIFY(false,
              "Invalid IGR sigma cadence '"
                << value
                << "'. Use 'every-stage', 'every-step', "
                   "'extrapolated-every-step', or "
                   "'every-n-steps'.");
  return dg_igr::IGRSigmaUpdateCadence::EveryStage;
}

inline dg_igr::IGRAlphaLengthScalePolicy
ParseIGRAlphaLengthScalePolicy(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "min" || lower == "minimum")
  {
    return dg_igr::IGRAlphaLengthScalePolicy::Min;
  }
  if (lower == "avg" || lower == "average" || lower == "mean")
  {
    return dg_igr::IGRAlphaLengthScalePolicy::Average;
  }
  if (lower == "max" || lower == "maximum")
  {
    return dg_igr::IGRAlphaLengthScalePolicy::Max;
  }
  if (lower == "local" || lower == "local-dx" || lower == "element-local")
  {
    return dg_igr::IGRAlphaLengthScalePolicy::Local;
  }
  MFEM_VERIFY(false,
              "Invalid IGR alpha length-scale policy '"
                << value
                << "'. Use 'min', 'average', 'max', or 'local'.");
  return dg_igr::IGRAlphaLengthScalePolicy::Average;
}

inline const char* IGRIPKindToString(dg_igr::IGRIPKind value)
{
  switch (value)
  {
  case dg_igr::IGRIPKind::SIPG:
    return "sipg";
  case dg_igr::IGRIPKind::IIPG:
    return "iipg";
  case dg_igr::IGRIPKind::NIPG:
    return "nipg";
  }
  return "sipg";
}

inline const char* IGRSolveModeToString(dg_igr::IGRSolveMode value)
{
  switch (value)
  {
  case dg_igr::IGRSolveMode::SmootherSweeps:
    return "smoother";
  case dg_igr::IGRSolveMode::Krylov:
    return "krylov";
  }
  return "krylov";
}

inline const char*
IGRLinearSolverTypeToString(dg_igr::IGRLinearSolverType value)
{
  switch (value)
  {
  case dg_igr::IGRLinearSolverType::Auto:
    return "auto";
  case dg_igr::IGRLinearSolverType::CG:
    return "cg";
  case dg_igr::IGRLinearSolverType::GMRES:
    return "gmres";
  case dg_igr::IGRLinearSolverType::FGMRES:
    return "fgmres";
  case dg_igr::IGRLinearSolverType::BiCGSTAB:
    return "bicgstab";
  case dg_igr::IGRLinearSolverType::MINRES:
    return "minres";
  }
  return "auto";
}

inline const char* IGRPrecondTypeToString(dg_igr::IGRPrecondType value)
{
  switch (value)
  {
  case dg_igr::IGRPrecondType::Auto:
    return "auto";
  case dg_igr::IGRPrecondType::BoomerAMG:
    return "boomeramg";
  case dg_igr::IGRPrecondType::HypreILU:
    return "hypreilu";
  case dg_igr::IGRPrecondType::HypreEuclid:
    return "hypreeuclid";
  case dg_igr::IGRPrecondType::ParaSails:
    return "parasails";
  case dg_igr::IGRPrecondType::HypreSmoother:
    return "hypresmoother";
  case dg_igr::IGRPrecondType::OperatorChebyshev:
    return "chebyshev";
  case dg_igr::IGRPrecondType::None:
    return "none";
  }
  return "auto";
}

inline const char*
IGRSigmaUpdateCadenceToString(dg_igr::IGRSigmaUpdateCadence value)
{
  switch (value)
  {
  case dg_igr::IGRSigmaUpdateCadence::EveryStage:
    return "every-stage";
  case dg_igr::IGRSigmaUpdateCadence::EveryStep:
    return "every-step";
  case dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep:
    return "extrapolated-every-step";
  case dg_igr::IGRSigmaUpdateCadence::EveryNSteps:
    return "every-n-steps";
  }
  return "every-stage";
}

inline const char*
IGRAlphaLengthScalePolicyToString(dg_igr::IGRAlphaLengthScalePolicy value)
{
  switch (value)
  {
  case dg_igr::IGRAlphaLengthScalePolicy::Min:
    return "min";
  case dg_igr::IGRAlphaLengthScalePolicy::Average:
    return "average";
  case dg_igr::IGRAlphaLengthScalePolicy::Max:
    return "max";
  case dg_igr::IGRAlphaLengthScalePolicy::Local:
    return "local";
  }
  return "average";
}

inline std::optional<DGBackend> TryParseDGBackend(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "dg")
  {
    return DGBackend::DG;
  }
  if (lower == "dgsem")
  {
    return DGBackend::DGSEM;
  }
  return std::nullopt;
}

inline DGBackend ParseDGBackend(const std::string& value)
{
  if (const std::optional<DGBackend> backend = TryParseDGBackend(value))
  {
    return *backend;
  }
  MFEM_VERIFY(false,
              "Invalid DG backend '" << value << "'. Use 'dg' or 'dgsem'.");
  return DGBackend::DG;
}

inline std::optional<Discretization>
TryParseDiscretization(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "dg")
  {
    return Discretization::DG;
  }
  if (lower == "dgsem")
  {
    return Discretization::DGSEM;
  }
  if (lower == "hdg")
  {
    return Discretization::HDG;
  }
  return std::nullopt;
}

inline Discretization ParseDiscretization(const std::string& value)
{
  if (const std::optional<Discretization> discretization =
        TryParseDiscretization(value))
  {
    return *discretization;
  }
  MFEM_VERIFY(false,
              "Invalid discretization '" << value
                                          << "'. Use 'dg', 'dgsem', or 'hdg'.");
  return Discretization::DG;
}

inline std::optional<DGBackend>
DGBackendForDiscretization(Discretization discretization)
{
  switch (discretization)
  {
  case Discretization::DG:
    return DGBackend::DG;
  case Discretization::DGSEM:
    return DGBackend::DGSEM;
  case Discretization::HDG:
    return std::nullopt;
  }
  return std::nullopt;
}

inline Discretization DiscretizationForDGBackend(DGBackend backend)
{
  switch (backend)
  {
  case DGBackend::DG:
    return Discretization::DG;
  case DGBackend::DGSEM:
    return Discretization::DGSEM;
  }
  return Discretization::DG;
}

inline const char* DiscretizationToString(Discretization discretization)
{
  switch (discretization)
  {
  case Discretization::DG:
    return "dg";
  case Discretization::DGSEM:
    return "dgsem";
  case Discretization::HDG:
    return "hdg";
  }
  return "dg";
}

inline std::optional<Regularization>
TryParseRegularization(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "none")
  {
    return Regularization::None;
  }
  if (lower == "igr")
  {
    return Regularization::IGR;
  }
  return std::nullopt;
}

inline const char* RegularizationToString(Regularization regularization)
{
  switch (regularization)
  {
  case Regularization::None:
    return "none";
  case Regularization::IGR:
    return "igr";
  }
  return "none";
}

inline std::optional<TimeIntegrator>
TryParseTimeIntegrator(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "rk3ssp")
  {
    return TimeIntegrator::RK3SSP;
  }
  if (lower == "ssprk104")
  {
    return TimeIntegrator::SSPRK104;
  }
  if (lower == "backward-euler")
  {
    return TimeIntegrator::BackwardEuler;
  }
  if (lower == "partitioned-backward-euler")
  {
    return TimeIntegrator::PartitionedBackwardEuler;
  }
  if (lower == "rodas3")
  {
    return TimeIntegrator::Rodas3;
  }
  return std::nullopt;
}

inline const char* TimeIntegratorToString(TimeIntegrator time_integrator)
{
  switch (time_integrator)
  {
  case TimeIntegrator::RK3SSP:
    return "rk3ssp";
  case TimeIntegrator::SSPRK104:
    return "ssprk104";
  case TimeIntegrator::BackwardEuler:
    return "backward-euler";
  case TimeIntegrator::PartitionedBackwardEuler:
    return "partitioned-backward-euler";
  case TimeIntegrator::Rodas3:
    return "rodas3";
  }
  return "rk3ssp";
}

inline bool IsExplicitSSPRKTimeIntegrator(TimeIntegrator time_integrator)
{
  return time_integrator == TimeIntegrator::RK3SSP ||
         time_integrator == TimeIntegrator::SSPRK104;
}

inline const char*
SolverRequestResolutionStatusToString(SolverRequestResolutionStatus status)
{
  switch (status)
  {
  case SolverRequestResolutionStatus::Resolved:
    return "resolved";
  case SolverRequestResolutionStatus::InvalidDiscretization:
    return "invalid-discretization";
  case SolverRequestResolutionStatus::InvalidDGBackend:
    return "invalid-dg-backend";
  case SolverRequestResolutionStatus::InvalidDeviceBackend:
    return "invalid-device-backend";
  case SolverRequestResolutionStatus::InvalidHdgCudaEngine:
    return "invalid-hdg-cuda-engine";
  case SolverRequestResolutionStatus::Conflict:
    return "conflict";
  case SolverRequestResolutionStatus::InvalidRegularization:
    return "invalid-regularization";
  case SolverRequestResolutionStatus::InvalidTimeIntegrator:
    return "invalid-time-integrator";
  case SolverRequestResolutionStatus::UnsupportedHdgCudaEngine:
    return "unsupported-hdg-cuda-engine";
  case SolverRequestResolutionStatus::UnsupportedRegularization:
    return "unsupported-regularization";
  case SolverRequestResolutionStatus::UnsupportedTimeIntegrator:
    return "unsupported-time-integrator";
  }
  return "invalid-request";
}

inline const char*
DiscretizationProvenanceToString(DiscretizationProvenance provenance)
{
  switch (provenance)
  {
  case DiscretizationProvenance::CaseDefault:
    return "case-default";
  case DiscretizationProvenance::DiscretizationOption:
    return "discretization-option";
  case DiscretizationProvenance::DGBackendOption:
    return "dg-backend-option";
  case DiscretizationProvenance::BothOptions:
    return "both-options";
  }
  return "case-default";
}

inline DiscretizationResolution
ResolveDiscretization(const std::string& case_default_dg_backend,
                      const std::string& discretization_value,
                      const std::string& dg_backend_value,
                      bool discretization_overridden,
                      bool dg_backend_overridden)
{
  DiscretizationResolution result;
  if (!discretization_overridden && !dg_backend_overridden)
  {
    const std::optional<DGBackend> backend =
      TryParseDGBackend(case_default_dg_backend);
    if (!backend)
    {
      result.status = DiscretizationResolutionStatus::InvalidDGBackend;
      result.dg_backend.reset();
      result.message = "Invalid per-case DG backend '" +
                       case_default_dg_backend +
                       "'. Use 'dg' or 'dgsem'.";
      return result;
    }
    result.discretization = DiscretizationForDGBackend(*backend);
    result.dg_backend = backend;
    return result;
  }

  std::optional<Discretization> selected;
  if (discretization_overridden)
  {
    selected = TryParseDiscretization(discretization_value);
    if (!selected)
    {
      result.status = DiscretizationResolutionStatus::InvalidDiscretization;
      result.dg_backend.reset();
      result.message = "Invalid discretization '" + discretization_value +
                       "'. Use 'dg', 'dgsem', or 'hdg'.";
      return result;
    }
    result.provenance = dg_backend_overridden
                          ? DiscretizationProvenance::BothOptions
                          : DiscretizationProvenance::DiscretizationOption;
  }

  if (dg_backend_overridden)
  {
    const std::optional<DGBackend> legacy_backend =
      TryParseDGBackend(dg_backend_value);
    if (!legacy_backend)
    {
      result.status = DiscretizationResolutionStatus::InvalidDGBackend;
      result.dg_backend.reset();
      result.message = "Invalid DG backend '" + dg_backend_value +
                       "'. Use 'dg' or 'dgsem'.";
      return result;
    }
    const Discretization legacy_discretization =
      DiscretizationForDGBackend(*legacy_backend);
    if (selected && *selected != legacy_discretization)
    {
      result.status = DiscretizationResolutionStatus::Conflict;
      result.dg_backend.reset();
      result.message = "Conflicting '--discretization " +
                       discretization_value + "' and '--dg-backend " +
                       dg_backend_value + "' selections.";
      return result;
    }
    selected = legacy_discretization;
    if (!discretization_overridden)
    {
      result.provenance = DiscretizationProvenance::DGBackendOption;
    }
  }

  result.discretization = *selected;
  result.dg_backend = DGBackendForDiscretization(result.discretization);
  return result;
}

inline std::optional<DeviceBackend>
TryParseDeviceBackend(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "cpu")
  {
    return DeviceBackend::CPU;
  }
  if (lower == "cuda")
  {
    return DeviceBackend::CUDA;
  }
  return std::nullopt;
}

inline std::optional<HdgCudaEngine>
TryParseHdgCudaEngine(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "ordinary")
  {
    return HdgCudaEngine::Ordinary;
  }
  if (lower == "tensor")
  {
    return HdgCudaEngine::TensorCore;
  }
  return std::nullopt;
}

/// @brief Resolve every solver selector without constructing runtime state.
inline SolverRequestResolution
ResolveSolverRequest(const std::string& case_default_dg_backend,
                     bool case_default_use_igr,
                     const std::string& discretization_value,
                     const std::string& dg_backend_value,
                     const std::string& regularization_value,
                     const std::string& time_integrator_value,
                     bool discretization_overridden,
                     bool dg_backend_overridden,
                     bool regularization_overridden,
                     bool time_integrator_overridden,
                     const std::string& device_value = "cpu",
                     const std::string& hdg_cuda_engine_value = "ordinary")
{
  SolverRequestResolution result;
  const DiscretizationResolution discretization =
    ResolveDiscretization(case_default_dg_backend,
                          discretization_value,
                          dg_backend_value,
                          discretization_overridden,
                          dg_backend_overridden);
  result.discretization = discretization.discretization;
  result.discretization_provenance = discretization.provenance;
  result.dg_backend = discretization.dg_backend;
  if (!discretization.IsResolved())
  {
    result.status =
      discretization.status == DiscretizationResolutionStatus::InvalidDiscretization
        ? SolverRequestResolutionStatus::InvalidDiscretization
        : discretization.status == DiscretizationResolutionStatus::InvalidDGBackend
            ? SolverRequestResolutionStatus::InvalidDGBackend
            : SolverRequestResolutionStatus::Conflict;
    result.message = discretization.message;
    return result;
  }

  const std::optional<DeviceBackend> device =
    TryParseDeviceBackend(device_value);
  if (!device)
  {
    result.status = SolverRequestResolutionStatus::InvalidDeviceBackend;
    result.message = "Invalid device backend '" + device_value +
                     "'. Use 'cpu' or 'cuda'.";
    return result;
  }
  result.device_backend = *device;

  const std::optional<HdgCudaEngine> hdg_cuda_engine =
    TryParseHdgCudaEngine(hdg_cuda_engine_value);
  if (!hdg_cuda_engine)
  {
    result.status = SolverRequestResolutionStatus::InvalidHdgCudaEngine;
    result.message = "Invalid HDG CUDA engine '" + hdg_cuda_engine_value +
                     "'. Use 'ordinary' or 'tensor'.";
    return result;
  }
  result.hdg_cuda_engine = *hdg_cuda_engine;
  if (result.hdg_cuda_engine == HdgCudaEngine::TensorCore &&
      (result.discretization != Discretization::HDG ||
       result.device_backend != DeviceBackend::CUDA))
  {
    result.status = SolverRequestResolutionStatus::UnsupportedHdgCudaEngine;
    result.message = "HDG CUDA engine 'tensor' requires '--discretization hdg "
                     "--device cuda'.";
    return result;
  }

  result.regularization = case_default_use_igr ? Regularization::IGR
                                                : Regularization::None;
  if (regularization_overridden)
  {
    const std::optional<Regularization> regularization =
      TryParseRegularization(regularization_value);
    if (!regularization)
    {
      result.status = SolverRequestResolutionStatus::InvalidRegularization;
      result.message = "Invalid regularization '" + regularization_value +
                       "'. Use 'none' or 'igr'.";
      return result;
    }
    result.regularization = *regularization;
  }

  result.time_integrator =
    result.discretization == Discretization::HDG
      ? TimeIntegrator::BackwardEuler
      : TimeIntegrator::RK3SSP;
  if (time_integrator_overridden)
  {
    const std::optional<TimeIntegrator> time_integrator =
      TryParseTimeIntegrator(time_integrator_value);
    if (!time_integrator)
    {
      result.status = SolverRequestResolutionStatus::InvalidTimeIntegrator;
      result.message = "Invalid time integrator '" + time_integrator_value +
                       "'. Use 'rk3ssp', 'ssprk104', "
                       "'backward-euler', 'partitioned-backward-euler', or "
                       "'rodas3'.";
      return result;
    }
    result.time_integrator = *time_integrator;
  }

  if (result.discretization != Discretization::HDG &&
      !IsExplicitSSPRKTimeIntegrator(result.time_integrator))
  {
    result.status = SolverRequestResolutionStatus::UnsupportedTimeIntegrator;
    result.message = "Time integrator '" +
                     std::string(TimeIntegratorToString(
                       result.time_integrator)) +
                     "' is unavailable for DG/DGSEM; use 'rk3ssp' or "
                     "'ssprk104'.";
    return result;
  }
  if (result.discretization == Discretization::HDG &&
      IsExplicitSSPRKTimeIntegrator(result.time_integrator))
  {
    result.status = SolverRequestResolutionStatus::UnsupportedTimeIntegrator;
    result.message = "Time integrator '" +
                     std::string(TimeIntegratorToString(
                       result.time_integrator)) +
                     "' is unavailable for HDG; use 'backward-euler', "
                     "'partitioned-backward-euler', or 'rodas3'.";
    return result;
  }
  return result;
}

inline DeviceBackend ParseDeviceBackend(const std::string& value)
{
  if (const std::optional<DeviceBackend> backend =
        TryParseDeviceBackend(value))
  {
    return *backend;
  }
  MFEM_VERIFY(false,
              "Invalid device backend '" << value
                                         << "'. Use 'cpu' or 'cuda'.");
  return DeviceBackend::CPU;
}

inline HdgCudaEngine ParseHdgCudaEngine(const std::string& value)
{
  if (const std::optional<HdgCudaEngine> engine =
        TryParseHdgCudaEngine(value))
  {
    return *engine;
  }
  MFEM_VERIFY(false,
              "Invalid HDG CUDA engine '"
                << value << "'. Use 'ordinary' or 'tensor'.");
  return HdgCudaEngine::Ordinary;
}

inline bool IsL2IGRSpaceName(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  return lower == "l2";
}

inline bool IsBrokenDGIGRGradName(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  return lower == "l2" || lower == "brokenl2" || lower == "brokendg" ||
         lower == "broken-dg";
}

inline bool IsExactDGFaceOptionName(const std::string& value)
{
  return value == "-iexface" || value == "--igr-exact-dg-face" ||
         value == "-ino-exface" || value == "--no-igr-exact-dg-face";
}

inline bool IsCAlphaOptionName(const std::string& value)
{
  return value == "-ca" || value == "--c-alpha";
}

inline bool IsHdgIgrAlphaOptionName(const std::string& value)
{
  return value == "-hia" || value == "--hdg-igr-alpha";
}

inline bool HasExactDGFaceOptionOverride(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    if (IsExactDGFaceOptionName(argv[i]))
    {
      return true;
    }
  }
  return false;
}

inline bool HasCAlphaOptionOverride(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    if (IsCAlphaOptionName(argv[i]))
    {
      return true;
    }
  }
  return false;
}

inline bool HasHdgIgrAlphaOptionOverride(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    if (IsHdgIgrAlphaOptionName(argv[i]))
    {
      return true;
    }
  }
  return false;
}

inline bool IsDiscretizationOptionName(const std::string& value)
{
  return value == "-disc" || value == "--discretization";
}

inline bool IsDGBackendOptionName(const std::string& value)
{
  return value == "-dg" || value == "--dg-backend";
}

inline bool IsRegularizationOptionName(const std::string& value)
{
  return value == "-reg" || value == "--regularization";
}

inline bool IsTimeIntegratorOptionName(const std::string& value)
{
  return value == "-ti" || value == "--time-integrator";
}

inline bool HasDiscretizationOptionOverride(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    if (IsDiscretizationOptionName(argv[i]))
    {
      return true;
    }
  }
  return false;
}

inline bool HasDGBackendOptionOverride(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    if (IsDGBackendOptionName(argv[i]))
    {
      return true;
    }
  }
  return false;
}

inline bool HasRegularizationOptionOverride(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    if (IsRegularizationOptionName(argv[i]))
    {
      return true;
    }
  }
  return false;
}

inline bool HasTimeIntegratorOptionOverride(int argc, char** argv)
{
  for (int i = 1; i < argc; ++i)
  {
    if (IsTimeIntegratorOptionName(argv[i]))
    {
      return true;
    }
  }
  return false;
}

inline bool ShouldDefaultExactDGFaceOperator(const CommonOptions& opts,
                                             const IGROptions& igr_opts)
{
  return ParseDeviceBackend(opts.device) == DeviceBackend::CUDA &&
         ParseDGBackend(opts.dg_backend) == DGBackend::DG &&
         IsL2IGRSpaceName(igr_opts.space) &&
         IsBrokenDGIGRGradName(igr_opts.grad);
}

inline void
ApplyIGRDerivedRuntimeDefaults(const CommonOptions& opts,
                               IGROptions& igr_opts,
                               bool exact_face_operator_overridden)
{
  if (igr_opts.nonconvergence_policy.empty())
  {
    igr_opts.nonconvergence_policy = "accept";
  }
  const bool cuda_l2_broken_dg =
    ParseDeviceBackend(opts.device) == DeviceBackend::CUDA &&
    ParseDGBackend(opts.dg_backend) == DGBackend::DG &&
    IsL2IGRSpaceName(igr_opts.space) && IsBrokenDGIGRGradName(igr_opts.grad);
  const bool cpu_l2_dg =
    ParseDeviceBackend(opts.device) == DeviceBackend::CPU &&
    ParseDGBackend(opts.dg_backend) == DGBackend::DG &&
    IsL2IGRSpaceName(igr_opts.space);

  if (!exact_face_operator_overridden &&
      ShouldDefaultExactDGFaceOperator(opts, igr_opts))
  {
    igr_opts.exact_dg_face_operator = true;
  }

  if (cuda_l2_broken_dg && igr_opts.exact_dg_face_operator)
  {
    if (ToLowerCopy(igr_opts.linear_solver) == "auto")
    {
      igr_opts.linear_solver = "minres";
    }
    if (ToLowerCopy(igr_opts.preconditioner) == "auto")
    {
      igr_opts.preconditioner = "chebyshev";
    }
    if (igr_opts.stage_rtol_scale < 0.0)
    {
      igr_opts.stage_rtol_scale = 1.0;
    }
    if (igr_opts.lin_rtol < 0.0)
    {
      igr_opts.lin_rtol = 1.0e-2;
    }
    if (igr_opts.lin_atol < 0.0)
    {
      igr_opts.lin_atol = 0.0;
    }
    if (igr_opts.lin_maxit < 0)
    {
      igr_opts.lin_maxit = 500;
    }
  }
  else if (cpu_l2_dg)
  {
    if (ToLowerCopy(igr_opts.linear_solver) == "auto")
    {
      igr_opts.linear_solver = "gmres";
    }
    if (ToLowerCopy(igr_opts.preconditioner) == "auto")
    {
      igr_opts.preconditioner = "boomeramg";
    }
    if (igr_opts.stage_rtol_scale < 0.0)
    {
      igr_opts.stage_rtol_scale = 1.0;
    }
    if (igr_opts.lin_rtol < 0.0)
    {
      igr_opts.lin_rtol = 1.0e-2;
    }
    if (igr_opts.lin_atol < 0.0)
    {
      igr_opts.lin_atol = 0.0;
    }
    if (igr_opts.lin_maxit < 0)
    {
      igr_opts.lin_maxit = 500;
    }
  }
}

inline DGSEMStabilization ParseDGSEMStabilization(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "none")
  {
    return DGSEMStabilization::None;
  }
  if (lower == "split" || lower == "split-form" || lower == "splitform")
  {
    return DGSEMStabilization::SplitForm;
  }
  if (lower == "filter" || lower == "filtering")
  {
    return DGSEMStabilization::Filter;
  }
  MFEM_VERIFY(false,
              "Invalid DGSEM stabilization '"
                << value << "'. Use 'none', 'split', or 'filter'.");
  return DGSEMStabilization::None;
}

inline const char* DGSEMStabilizationToString(DGSEMStabilization stab)
{
  switch (stab)
  {
  case DGSEMStabilization::None:
    return "none";
  case DGSEMStabilization::SplitForm:
    return "split";
  case DGSEMStabilization::Filter:
    return "filter";
  }
  return "none";
}

inline DGSEMVolumeFlux ParseDGSEMVolumeFlux(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "central")
  {
    return DGSEMVolumeFlux::Central;
  }
  if (lower == "chandrashekar" || lower == "chandra")
  {
    return DGSEMVolumeFlux::Chandrashekar;
  }
  if (lower == "ranocha")
  {
    return DGSEMVolumeFlux::Ranocha;
  }
  MFEM_VERIFY(false,
              "Invalid DGSEM volume flux '"
                << value << "'. Use 'central', 'chandrashekar', or 'ranocha'.");
  return DGSEMVolumeFlux::Central;
}

inline const char* DGSEMVolumeFluxToString(DGSEMVolumeFlux flux)
{
  switch (flux)
  {
  case DGSEMVolumeFlux::Central:
    return "central";
  case DGSEMVolumeFlux::Chandrashekar:
    return "chandrashekar";
  case DGSEMVolumeFlux::Ranocha:
    return "ranocha";
  }
  return "central";
}

inline DGSEMVolumeDissipation
ParseDGSEMVolumeDissipation(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "none")
  {
    return DGSEMVolumeDissipation::None;
  }
  if (lower == "llf" || lower == "lax-friedrichs" || lower == "rusanov")
  {
    return DGSEMVolumeDissipation::LLF;
  }
  if (lower == "entropy-lf" || lower == "entropy-vars-lf" ||
      lower == "entropy-variable-lf" || lower == "es-lf")
  {
    return DGSEMVolumeDissipation::EntropyVarsLF;
  }
  MFEM_VERIFY(false,
              "Invalid DGSEM volume dissipation '"
                << value << "'. Use 'none', 'llf', or 'entropy-lf'.");
  return DGSEMVolumeDissipation::None;
}

inline const char* DGSEMVolumeDissipationToString(DGSEMVolumeDissipation diss)
{
  switch (diss)
  {
  case DGSEMVolumeDissipation::None:
    return "none";
  case DGSEMVolumeDissipation::LLF:
    return "llf";
  case DGSEMVolumeDissipation::EntropyVarsLF:
    return "entropy-lf";
  }
  return "none";
}

inline DGSEMSurfaceFlux ParseDGSEMSurfaceFlux(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "rusanov" || lower == "lf" || lower == "lax-friedrichs")
  {
    return DGSEMSurfaceFlux::Rusanov;
  }
  if (lower == "diffusive-rusanov" || lower == "rusanov-diffusive" ||
      lower == "llf-diffusive" || lower == "diffusive-llf")
  {
    return DGSEMSurfaceFlux::DiffusiveRusanov;
  }
  if (lower == "es-winters" || lower == "entropy-winters" || lower == "winters")
  {
    return DGSEMSurfaceFlux::EntropyStableWinters;
  }
  if (lower == "es-lf-entropy" || lower == "entropy-lf" || lower == "es-lf")
  {
    return DGSEMSurfaceFlux::EntropyStableLFEntropyVars;
  }
  MFEM_VERIFY(false,
              "Invalid DGSEM surface flux '"
                << value
                << "'. Use 'rusanov', 'diffusive-rusanov', 'es-winters', or "
                   "'es-lf-entropy'.");
  return DGSEMSurfaceFlux::Rusanov;
}

inline DGSEMEntropyJacobian ParseDGSEMEntropyJacobian(const std::string& value)
{
  const std::string lower = ToLowerCopy(value);
  if (lower == "fd" || lower == "finite" || lower == "finite-difference")
  {
    return DGSEMEntropyJacobian::FiniteDifference;
  }
  if (lower == "analytic" || lower == "exact")
  {
    return DGSEMEntropyJacobian::Analytic;
  }
  MFEM_VERIFY(false,
              "Invalid DGSEM entropy Jacobian '"
                << value << "'. Use 'fd' or 'analytic'.");
  return DGSEMEntropyJacobian::FiniteDifference;
}

inline const char* DGSEMEntropyJacobianToString(DGSEMEntropyJacobian mode)
{
  switch (mode)
  {
  case DGSEMEntropyJacobian::FiniteDifference:
    return "fd";
  case DGSEMEntropyJacobian::Analytic:
    return "analytic";
  }
  return "fd";
}

inline const char* DGSEMSurfaceFluxToString(DGSEMSurfaceFlux flux)
{
  switch (flux)
  {
  case DGSEMSurfaceFlux::Rusanov:
    return "rusanov";
  case DGSEMSurfaceFlux::DiffusiveRusanov:
    return "diffusive-rusanov";
  case DGSEMSurfaceFlux::EntropyStableWinters:
    return "es-winters";
  case DGSEMSurfaceFlux::EntropyStableLFEntropyVars:
    return "es-lf-entropy";
  }
  MFEM_ABORT(
    "Unhandled DGSEMSurfaceFlux enum value in DGSEMSurfaceFluxToString.");
  return "rusanov";
}

inline const char* DGBackendToString(DGBackend backend)
{
  switch (backend)
  {
  case DGBackend::DG:
    return "dg";
  case DGBackend::DGSEM:
    return "dgsem";
  }
  return "dg";
}

inline const char* DeviceBackendToString(DeviceBackend backend)
{
  switch (backend)
  {
  case DeviceBackend::CPU:
    return "cpu";
  case DeviceBackend::CUDA:
    return "cuda";
  }
  return "cpu";
}

inline const char* HdgCudaEngineToString(HdgCudaEngine engine)
{
  switch (engine)
  {
  case HdgCudaEngine::Ordinary:
    return "ordinary";
  case HdgCudaEngine::TensorCore:
    return "tensor";
  }
  return "unknown";
}

inline std::string CaseDirFromFile(const char* file)
{
  return std::filesystem::path(file).parent_path().string();
}

inline std::string ResolveCaseMesh(const CaseConfig& cfg,
                                   const std::string& override_mesh)
{
  if (!override_mesh.empty())
  {
    return override_mesh;
  }
  return (std::filesystem::path(cfg.case_dir) / cfg.default_mesh_file).string();
}

inline std::string DefaultCaseMeshPath(const CaseConfig& cfg)
{
  return ResolveCaseMesh(cfg, "");
}

inline std::string DefaultVTKPrefixFromCaseDir(const std::string& case_dir)
{
  const auto default_prefix = mfem::io::VTKOutputConfig{}.prefix_path;
  const std::filesystem::path case_path(case_dir);
  if (case_path.empty() || !case_path.has_parent_path() ||
      !case_path.parent_path().has_parent_path())
  {
    return default_prefix;
  }
  return (case_path.parent_path().parent_path() / "out").string();
}

/// @brief Apply global example defaults, overriding any struct defaults.
///
/// This function reinitializes config members with default-constructed values
/// and then applies overrides (e.g., RK3/IGR/limiter settings). If you are
/// looking for where defaults are set, check here first.
inline void ApplyExampleDefaults(CaseConfig& cfg)
{
  cfg.defaults = CommonOptions{};
  cfg.rk3_config = mfem::RK3Config{};
  cfg.igr_config = dg_igr::IGRConfig{};
  cfg.limiter_pipeline = mfem::LimiterPipelineConfig{};
  cfg.vtk_config = mfem::io::VTKOutputConfig{};

  cfg.limiter_pipeline.venkat_opts.eps_kappa = 100.0;

  cfg.rk3_config.gamma = cfg.gamma;
  cfg.rk3_config.rho_min = 5.0e-7;
  cfg.rk3_config.p_min = 1.0e-6;
  cfg.rk3_config.quad_order = 0;
  cfg.rk3_config.p_bisect_iters = 12;
  cfg.rk3_config.limit_initial = false;
  cfg.rk3_config.max_limiter_steps = 0;

  cfg.igr_config.lin_rtol = 1.0e-6;
  cfg.igr_config.lin_atol = 0.0;
  cfg.igr_config.lin_maxit = 1000;
  cfg.igr_config.lin_print = 0;
  cfg.igr_config.alpha_length_scale_policy =
    dg_igr::IGRAlphaLengthScalePolicy::Average;
  cfg.igr_config.sigma_cadence = dg_igr::IGRSigmaUpdateCadence::EveryStage;
  cfg.igr_config.sigma_update_interval = 1;
  cfg.igr_config.stage_rtol_scale = 1.0;
  cfg.igr_config.amg_coarsen_type = -1;
  cfg.igr_config.amg_relax_type = -1;
  cfg.igr_config.amg_max_levels = -1;
  cfg.igr_config.amg_strong_threshold = -1.0;
  cfg.vtk_config.prefix_path = DefaultVTKPrefixFromCaseDir(cfg.case_dir);
}

inline CaseConfig
MakeCaseConfig(const std::string& case_name,
               const std::string& default_mesh_file,
               const char* case_file,
               mfem::real_t gamma = mfem::dg_igr::EulerDefaults::kGamma)
{
  CaseConfig cfg;
  cfg.case_name = case_name;
  cfg.default_mesh_file = default_mesh_file;
  cfg.case_dir = CaseDirFromFile(case_file);
  cfg.gamma = gamma;
  ApplyExampleDefaults(cfg);
  return cfg;
}

inline void AddIGROptions(mfem::OptionsParser& args, IGROptions& igr_opts)
{
  args.AddOption(&igr_opts.c_alpha_user,
                 "-ca",
                 "--c-alpha",
                 "Forward scaling constant C_alpha (default 1.0); for HDG "
                 "this aliases --hdg-igr-alpha when that option is absent.");
  args.AddOption(&igr_opts.alpha_length_policy,
                 "-ial",
                 "--igr-alpha-length",
                 "IGR alpha length-scale policy: min | average | max | local.");
  args.AddOption(&igr_opts.space,
                 "-is",
                 "--igr-space",
                 "Space to solve for sigma: 'H1' or 'L2'.");
  args.AddOption(&igr_opts.grad,
                 "-ig",
                 "--igr-grad",
                 "Gradient reconstruction type: 'H1' or 'BrokenL2'.");
  args.AddOption(&igr_opts.solve_mode,
                 "-im",
                 "--igr-solve-mode",
                 "IGR solve mode: krylov | smoother.");
  args.AddOption(
    &igr_opts.linear_solver,
    "-ils",
    "--igr-linear-solver",
    "IGR linear solver: auto | cg | gmres | fgmres | bicgstab | minres.");
  args.AddOption(&igr_opts.preconditioner,
                 "-ipc",
                 "--igr-preconditioner",
                 "IGR preconditioner: auto | boomeramg | hypreilu | "
                 "hypreeuclid | parasails | "
                 "hypresmoother | chebyshev | none.");
  args.AddOption(&igr_opts.nonconvergence_policy,
                 "-incp",
                 "--igr-nonconvergence-policy",
                 "IGR nonconvergence policy: accept | error (default accept "
                 "retains the capped iterate).");
  args.AddOption(
    &igr_opts.sigma_cadence,
    "-iscad",
    "--igr-sigma-cadence",
    "IGR sigma update cadence: every-stage | every-step | "
    "extrapolated-every-step | every-n-steps.");
  args.AddOption(&igr_opts.sigma_update_interval,
                 "-isn",
                 "--igr-sigma-update-interval",
                 "IGR sigma update interval N for every-n-steps cadence (<0 "
                 "keeps case default).");
  args.AddOption(&igr_opts.stage_rtol_scale,
                 "-isrtol",
                 "--igr-stage-rtol-scale",
                 "IGR stage-rtol multiplier for intermediate RK stages (>1 "
                 "loosens; <0 keeps case default).");
  args.AddOption(&igr_opts.ip_kind,
                 "-iip",
                 "--igr-ip-kind",
                 "IGR DG interior penalty kind: sipg | iipg | nipg.");
  args.AddOption(
    &igr_opts.lin_rtol,
    "-irtol",
    "--igr-lin-rtol",
    "IGR linear solver relative tolerance (<0 keeps case default).");
  args.AddOption(
    &igr_opts.lin_atol,
    "-iatol",
    "--igr-lin-atol",
    "IGR linear solver absolute tolerance (<0 keeps case default).");
  args.AddOption(&igr_opts.lin_maxit,
                 "-imaxit",
                 "--igr-lin-maxit",
                 "IGR linear solver max iterations (<0 keeps case default).");
  args.AddOption(&igr_opts.lin_print,
                 "-iprint",
                 "--igr-lin-print",
                 "IGR linear solver print level (INT_MIN keeps case default).");
  args.AddOption(&igr_opts.amg_coarsen_type,
                 "-iamgc",
                 "--igr-amg-coarsen-type",
                 "IGR BoomerAMG coarsen type (<0 keeps HYPRE default).");
  args.AddOption(&igr_opts.amg_relax_type,
                 "-iamgr",
                 "--igr-amg-relax-type",
                 "IGR BoomerAMG relax type (<0 keeps HYPRE default).");
  args.AddOption(&igr_opts.amg_max_levels,
                 "-iamgl",
                 "--igr-amg-max-levels",
                 "IGR BoomerAMG max levels (<0 keeps HYPRE default).");
  args.AddOption(&igr_opts.amg_strong_threshold,
                 "-iamgst",
                 "--igr-amg-strong-threshold",
                 "IGR BoomerAMG strong-threshold (<0 keeps HYPRE default).");
  args.AddOption(&igr_opts.exact_dg_face_operator,
                 "-iexface",
                 "--igr-exact-dg-face",
                 "-ino-exface",
                 "--no-igr-exact-dg-face",
                 "Enable or disable the side-aware exact DG face operator "
                 "for L2/DG IGR.");
  args.AddOption(&igr_opts.sigma_aware_cfl,
                 "-isigcfl",
                 "--igr-sigma-cfl",
                 "-ino-sigcfl",
                 "--no-igr-sigma-cfl",
                 "Include or exclude the IGR sigma speed in CFL accounting "
                 "(default off).");
  args.AddOption(
    &igr_opts.sigma_aware_rusanov,
    "-isigrus",
    "--igr-sigma-rusanov",
    "-ino-sigrus",
    "--no-igr-sigma-rusanov",
    "Include or exclude the IGR sigma speed in Rusanov dissipation "
    "(default off).");
  args.AddOption(
    &igr_opts.igr_invariant_domain_flux,
    "-iidflux",
    "--igr-invariant-domain-flux",
    "-ino-iidflux",
    "--no-igr-invariant-domain-flux",
    "Use or disable the exact frozen-stage LF split threshold together with "
    "the frozen-sigma characteristic-modulus bound for both CFL accounting "
    "and Rusanov dissipation (default off).");
}

inline RunConfig MakeRunConfig(
  const CaseConfig& case_cfg,
  const CommonOptions& opts,
  const IGROptions& igr_opts,
  DiscretizationProvenance discretization_provenance =
    DiscretizationProvenance::CaseDefault,
  Regularization regularization = Regularization::IGR,
  TimeIntegrator time_integrator = TimeIntegrator::RK3SSP)
{
  RunConfig run_cfg;
  run_cfg.case_config = case_cfg;
  const PositivityLimiterSelection positivity_limiter =
    ParsePositivityLimiterSelection(opts.positivity_limiter);
  const mfem::ZhangShuPointSet zhang_shu_points =
    ParseZhangShuPointSet(opts.zhang_shu_points);
  run_cfg.case_config.limiter_pipeline.zhang_shu =
    ResolveZhangShuPositivityLimiter(
      positivity_limiter, case_cfg.limiter_pipeline.zhang_shu);
  run_cfg.case_config.limiter_pipeline.zhang_shu_points = zhang_shu_points;
  run_cfg.case_config.limiter_pipeline.zhang_shu_vtk_points =
    opts.zhang_shu_vtk_points;
  run_cfg.opts = opts;
  run_cfg.opts.positivity_limiter =
    PositivityLimiterSelectionToString(positivity_limiter);
  run_cfg.opts.zhang_shu_points = ZhangShuPointSetToString(zhang_shu_points);
  run_cfg.igr_opts = igr_opts;
  run_cfg.device_backend = ParseDeviceBackend(opts.device);
  run_cfg.hdg_cuda_engine = ParseHdgCudaEngine(opts.hdg_cuda_engine);
  run_cfg.discretization = opts.discretization.empty()
                              ? DiscretizationForDGBackend(
                                  ParseDGBackend(opts.dg_backend))
                              : ParseDiscretization(opts.discretization);
  run_cfg.discretization_provenance = discretization_provenance;
  run_cfg.regularization = regularization;
  run_cfg.time_integrator = time_integrator;
  const std::optional<DGBackend> dg_backend =
    DGBackendForDiscretization(run_cfg.discretization);
  MFEM_VERIFY(dg_backend,
              "Discretization '"
                << DiscretizationToString(run_cfg.discretization)
                << "' is recognized but unavailable in this executable.");
  run_cfg.dg_backend = dg_backend.value_or(DGBackend::DG);
  run_cfg.dgsem_stab = ParseDGSEMStabilization(opts.dgsem_stab);
  run_cfg.dgsem_volume_flux = ParseDGSEMVolumeFlux(opts.dgsem_volume_flux);
  run_cfg.dgsem_volume_dissipation =
    ParseDGSEMVolumeDissipation(opts.dgsem_volume_dissipation);
  run_cfg.dgsem_surface_flux = ParseDGSEMSurfaceFlux(opts.dgsem_surface_flux);
  run_cfg.dgsem_entropy_jacobian =
    ParseDGSEMEntropyJacobian(opts.dgsem_entropy_jacobian);
  run_cfg.mesh_path = ResolveCaseMesh(case_cfg, opts.mesh_file);
  return run_cfg;
}

inline dg_igr::IGRAlphaLengthScalePolicy
ResolveIGRAlphaLengthScalePolicy(const RunConfig& run_cfg)
{
  if (!run_cfg.igr_opts.alpha_length_policy.empty())
  {
    return ParseIGRAlphaLengthScalePolicy(run_cfg.igr_opts.alpha_length_policy);
  }
  return run_cfg.case_config.igr_config.alpha_length_scale_policy;
}

/// Exact frozen-sigma acoustic speeds belong to the typed extrapolated route.
inline bool UsesFrozenSigmaSpectralSpeed(const RunConfig& run_cfg)
{
  const dg_igr::IGRSigmaUpdateCadence cadence =
    run_cfg.igr_opts.sigma_cadence.empty()
      ? run_cfg.case_config.igr_config.sigma_cadence
      : ParseIGRSigmaUpdateCadence(run_cfg.igr_opts.sigma_cadence);
  return run_cfg.regularization == Regularization::IGR &&
         run_cfg.discretization == Discretization::DGSEM &&
         cadence == dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep;
}

/** Reject cadence requests that the existing HDG DAE backend cannot honor.
 *
 * The HDG adapter reconstructs its algebraic variables inside the implicit
 * stage solve and has no implementation of frozen/extrapolated sigma stages.
 * Validate this before the adapter's early return so a DG-only cadence is
 * never accepted and then silently ignored.
 */
inline std::string ValidateHdgIGRSigmaCadence(
  Discretization discretization,
  Regularization regularization,
  const CaseConfig& case_cfg,
  const IGROptions& igr_opts)
{
  if (discretization != Discretization::HDG)
  {
    return {};
  }
  // A conventional HDG run has no sigma state, but an explicit cadence flag
  // is still a user request and must not disappear at the adapter boundary.
  // Only active IGR inherits the case default when no flag was supplied.
  if (igr_opts.sigma_cadence.empty() &&
      regularization != Regularization::IGR)
  {
    return {};
  }
  const dg_igr::IGRSigmaUpdateCadence cadence =
    igr_opts.sigma_cadence.empty()
      ? case_cfg.igr_config.sigma_cadence
      : ParseIGRSigmaUpdateCadence(igr_opts.sigma_cadence);
  if (cadence == dg_igr::IGRSigmaUpdateCadence::EveryStage)
  {
    return {};
  }
  return "The HDG instantaneous-IGR backend supports only "
         "--igr-sigma-cadence every-stage; split and extrapolated cadences "
         "are qualified only on their explicitly supported DG/DGSEM routes.";
}

/** Validate the production-qualified extrapolated instantaneous-IGR tuple.
 *
 * Extrapolating the algebraic closure is a distinct temporal discretization,
 * not a generic cadence optimization.  Keep it an explicit typed selection:
 * every other DG/IGR configuration retains its established every-stage or
 * every-step behavior instead of silently inheriting an unqualified split.
 */
inline std::string
ValidateExtrapolatedEveryStepIGRSelection(const RunConfig& run_cfg)
{
  const dg_igr::IGRSigmaUpdateCadence cadence =
    run_cfg.igr_opts.sigma_cadence.empty()
      ? run_cfg.case_config.igr_config.sigma_cadence
      : ParseIGRSigmaUpdateCadence(run_cfg.igr_opts.sigma_cadence);
  if (cadence != dg_igr::IGRSigmaUpdateCadence::ExtrapolatedEveryStep)
  {
    return {};
  }

  const char* prefix =
    "--igr-sigma-cadence extrapolated-every-step requires ";
  if (run_cfg.regularization != Regularization::IGR)
  {
    return std::string(prefix) + "active IGR.";
  }
  if (run_cfg.discretization != Discretization::DGSEM ||
      run_cfg.dg_backend != DGBackend::DGSEM)
  {
    return std::string(prefix) + "the DGSEM discretization.";
  }
  if (run_cfg.time_integrator != TimeIntegrator::SSPRK104)
  {
    return std::string(prefix) + "the SSPRK(10,4) time integrator.";
  }
  if (!run_cfg.case_config.limiter_pipeline.zhang_shu ||
      run_cfg.case_config.limiter_pipeline.zhang_shu_points !=
        mfem::ZhangShuPointSet::kConsumer)
  {
    return std::string(prefix) +
           "the stagewise Zhang-Shu limiter on consumer points.";
  }
  const mfem::LimiterPipelineConfig& limiter =
    run_cfg.case_config.limiter_pipeline;
  if (limiter.bj_mode != mfem::BJLimiterMode::kNone ||
      limiter.tvb_mode != mfem::TVBLimiterMode::kNone ||
      limiter.venkat_mode != mfem::VenkatLimiterMode::kNone ||
      limiter.krivodonova)
  {
    return std::string(prefix) +
           "the qualified Zhang-Shu-only limiter pipeline.";
  }
  if (run_cfg.dgsem_stab != DGSEMStabilization::None ||
      run_cfg.dgsem_volume_flux != DGSEMVolumeFlux::Central ||
      run_cfg.dgsem_volume_dissipation != DGSEMVolumeDissipation::LLF ||
      run_cfg.dgsem_surface_flux != DGSEMSurfaceFlux::Rusanov)
  {
    return std::string(prefix) +
           "DGSEM central volume flux, LLF volume dissipation, Rusanov "
           "surface flux, and no additional DGSEM stabilization.";
  }
  if (!IsL2IGRSpaceName(run_cfg.igr_opts.space) ||
      !IsBrokenDGIGRGradName(run_cfg.igr_opts.grad) ||
      ParseIGRIPKind(run_cfg.igr_opts.ip_kind) != dg_igr::IGRIPKind::SIPG ||
      !run_cfg.igr_opts.exact_dg_face_operator)
  {
    return std::string(prefix) +
           "the L2/BrokenL2 SIPG closure with exact DG faces.";
  }
  if (ParseIGRSolveMode(run_cfg.igr_opts.solve_mode) !=
        dg_igr::IGRSolveMode::Krylov ||
      ParseIGRLinearSolverType(run_cfg.igr_opts.linear_solver) !=
        dg_igr::IGRLinearSolverType::CG ||
      ParseIGRPrecondType(run_cfg.igr_opts.preconditioner) !=
        dg_igr::IGRPrecondType::OperatorChebyshev)
  {
    return std::string(prefix) +
           "Krylov CG with the full-system operator-Chebyshev preconditioner "
           "so every exact algebraic anchor can receive an independent FP64 "
           "residual certificate.";
  }
  if (ResolveIGRAlphaLengthScalePolicy(run_cfg) !=
      dg_igr::IGRAlphaLengthScalePolicy::Local)
  {
    return std::string(prefix) +
           "the local alpha=C_alpha*h_eff^2 policy.";
  }
  if (run_cfg.igr_opts.igr_invariant_domain_flux ||
      run_cfg.igr_opts.sigma_aware_cfl ||
      run_cfg.igr_opts.sigma_aware_rusanov)
  {
    return std::string(prefix) +
           "its intrinsic frozen-sigma spectral speed policy (legacy "
           "invariant-domain, sigma-CFL, and sigma-Rusanov toggles off).";
  }
  if (run_cfg.igr_opts.nonconvergence_policy.empty() ||
      ParseIGRNonconvergencePolicy(
        run_cfg.igr_opts.nonconvergence_policy) !=
      dg_igr::IGRNonconvergencePolicy::Error)
  {
    return std::string(prefix) +
           "--igr-nonconvergence-policy error.";
  }
  return {};
}

inline TimeLoopConfig MakeTimeLoopConfig(const RunConfig& run_cfg)
{
  const CommonOptions& opts = run_cfg.opts;
  TimeLoopConfig cfg;
  cfg.t_final = opts.t_final;
  cfg.dt = opts.dt;
  cfg.dt_min = opts.dt_min;
  cfg.cfl = opts.cfl;
  cfg.order = opts.order;
  cfg.vis_steps = opts.vis_steps;
  cfg.vtk_save = opts.vtk_save;
  cfg.dt_save = opts.dt_save;
  cfg.nan_scan = opts.nan_scan;
  if (opts.analysis_steps < 0)
  {
    cfg.analysis_steps = (opts.vis_steps > 0) ? opts.vis_steps * 10 : 0;
  }
  else
  {
    cfg.analysis_steps = opts.analysis_steps;
  }
  cfg.save_index_offset = 0;
  cfg.initial_time = 0.0;
  cfg.initial_step = 0;
  cfg.next_save_time = (opts.vtk_save && opts.dt_save > 0.0) ? opts.dt_save
                                                              : -1.0;
  return cfg;
}

} // namespace dg_sim

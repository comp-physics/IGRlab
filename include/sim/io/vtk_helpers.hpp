#pragma once

#include "igrlab/environment.hpp"

/**
 * @file sim/io/vtk_helpers.hpp
 * @brief VTK naming/config helpers for sim-layer output.
 */

#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "mfem.hpp"

#include "sim/config/case_config.hpp"

namespace dg_sim
{

/// @brief Resolve the sigma cadence after applying the case default.
inline igrlab::IGRSigmaUpdateCadence
ResolveEffectiveIGRSigmaUpdateCadence(const RunConfig& run_cfg)
{
  return run_cfg.igr_opts.sigma_cadence.empty()
           ? run_cfg.case_config.igr_config.sigma_cadence
           : ParseIGRSigmaUpdateCadence(run_cfg.igr_opts.sigma_cadence);
}

/// @brief Resolve the sigma update interval after applying the case default.
inline int ResolveEffectiveIGRSigmaUpdateInterval(const RunConfig& run_cfg)
{
  return run_cfg.igr_opts.sigma_update_interval > 0
           ? run_cfg.igr_opts.sigma_update_interval
           : run_cfg.case_config.igr_config.sigma_update_interval;
}

/// @brief Default VTK output directory for a case (prefix_path/case_name).
inline std::string DefaultCaseVTKOutputDir(const CaseConfig& case_cfg)
{
  return (std::filesystem::path(case_cfg.vtk_config.prefix_path) /
          case_cfg.case_name)
    .string();
}

/// @brief Build a consistent dataset name for ParaView output.
inline std::string MakeCaseVTKDatasetName(
  const CaseConfig& case_cfg,
  const RunConfig& run_cfg,
  double c_alpha_value = std::numeric_limits<double>::quiet_NaN())
{
  const int total_ref =
    run_cfg.opts.ser_ref_levels + run_cfg.opts.par_ref_levels;
  const double c_alpha =
    std::isnan(c_alpha_value) ? run_cfg.igr_opts.c_alpha_user : c_alpha_value;
  std::ostringstream name;
  name << case_cfg.case_name << "_p_" << run_cfg.opts.order << "_ref_"
       << total_ref << "_is_" << run_cfg.igr_opts.space << "_ig_"
       << run_cfg.igr_opts.grad << "_ca_" << c_alpha;
  // Preserve the established RK3SSP dataset name while keeping the typed
  // alternative integrator outputs from colliding with it.
  if (run_cfg.time_integrator != TimeIntegrator::RK3SSP)
  {
    name << "_ti_" << TimeIntegratorToString(run_cfg.time_integrator);
  }
  // Preserve the historical DGSEM tuple's name while distinguishing every
  // non-default transport selector that changes the discrete operator.
  if (run_cfg.dg_backend == DGBackend::DGSEM)
  {
    if (run_cfg.dgsem_stab != DGSEMStabilization::None)
    {
      name << "_dstab_" << DGSEMStabilizationToString(run_cfg.dgsem_stab);
    }
    if (run_cfg.dgsem_volume_flux != DGSEMVolumeFlux::Central)
    {
      name << "_dvf_" << DGSEMVolumeFluxToString(run_cfg.dgsem_volume_flux);
    }
    if (run_cfg.dgsem_volume_dissipation != DGSEMVolumeDissipation::None)
    {
      name << "_dvd_"
           << DGSEMVolumeDissipationToString(
                run_cfg.dgsem_volume_dissipation);
    }
    if (run_cfg.dgsem_surface_flux != DGSEMSurfaceFlux::Rusanov)
    {
      name << "_dsf_" << DGSEMSurfaceFluxToString(run_cfg.dgsem_surface_flux);
    }
    if (run_cfg.dgsem_entropy_jacobian !=
        DGSEMEntropyJacobian::FiniteDifference)
    {
      name << "_dej_"
           << DGSEMEntropyJacobianToString(run_cfg.dgsem_entropy_jacobian);
    }
  }
  if (run_cfg.regularization == Regularization::IGR)
  {
    const auto cadence = ResolveEffectiveIGRSigmaUpdateCadence(run_cfg);
    if (cadence != igrlab::IGRSigmaUpdateCadence::EveryStage)
    {
      name << "_iscad_" << IGRSigmaUpdateCadenceToString(cadence);
      if (cadence == igrlab::IGRSigmaUpdateCadence::EveryNSteps)
      {
        name << "_isn_" << ResolveEffectiveIGRSigmaUpdateInterval(run_cfg);
      }
    }
    const auto alpha_policy = ResolveIGRAlphaLengthScalePolicy(run_cfg);
    if (alpha_policy == igrlab::IGRAlphaLengthScalePolicy::Local)
    {
      name << "_ial_" << IGRAlphaLengthScalePolicyToString(alpha_policy);
    }
    if (run_cfg.igr_opts.exact_dg_face_operator)
    {
      name << "_iexface";
    }
  }
  // The exact IGR-aware invariant-domain face speed changes the numerical
  // flux and therefore must never share an output directory with the legacy
  // Rusanov selection.
  if (run_cfg.igr_opts.igr_invariant_domain_flux)
  {
    name << "_iidflux";
  }
  // Refined VTK points enlarge the limiter's enforced point set and therefore
  // define a distinct numerical policy even when all other selectors match.
  if (run_cfg.opts.zhang_shu_vtk_points)
  {
    name << "_zsvtk";
  }
  if (run_cfg.opts.modal_damping_rate > 0.0)
  {
    name << "_mdr_" << std::setprecision(std::numeric_limits<mfem::real_t>::max_digits10)
         << run_cfg.opts.modal_damping_rate;
  }
  return name.str();
}

/// @brief Apply case defaults to the VTK output config.
inline mfem::io::VTKOutputConfig MakeCaseVTKConfig(const CaseConfig& case_cfg)
{
  mfem::io::VTKOutputConfig cfg = case_cfg.vtk_config;
  cfg.prefix_path = DefaultCaseVTKOutputDir(case_cfg);
  if (const char* prefix = ::igrlab::GetEnv("DG_IGR_VTK_PREFIX_PATH"))
  {
    if (prefix[0] != '\0') { cfg.prefix_path = prefix; }
  }
  return cfg;
}

} // namespace dg_sim

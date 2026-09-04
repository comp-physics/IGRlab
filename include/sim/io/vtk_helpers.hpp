#pragma once

/**
 * @file sim/io/vtk_helpers.hpp
 * @brief VTK naming/config helpers for sim-layer output.
 */

#include <filesystem>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>

#include "mfem.hpp"

#include "sim/config/case_config.hpp"

namespace dg_sim
{

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
       << run_cfg.igr_opts.grad;
  if (run_cfg.igr_opts.sigma_order >= 0)
  {
    name << "_iso_" << ResolveIGRSigmaOrder(run_cfg);
  }
  name << "_ca_" << c_alpha;
  // Preserve the established RK3SSP dataset name while keeping the typed
  // SSPRK(10,4) output from colliding with it.
  if (run_cfg.time_integrator == TimeIntegrator::SSPRK104)
  {
    name << "_ti_" << TimeIntegratorToString(run_cfg.time_integrator);
  }
  if (run_cfg.dg_backend == DGBackend::DG &&
      run_cfg.dg_surface_flux != DGSurfaceFlux::Rusanov)
  {
    name << "_dgsf_" << DGSurfaceFluxToString(run_cfg.dg_surface_flux);
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
  return name.str();
}

/// @brief Apply case defaults to the VTK output config.
inline mfem::io::VTKOutputConfig MakeCaseVTKConfig(const CaseConfig& case_cfg)
{
  mfem::io::VTKOutputConfig cfg = case_cfg.vtk_config;
  cfg.prefix_path = DefaultCaseVTKOutputDir(case_cfg);
  if (const char* prefix = std::getenv("DG_IGR_VTK_PREFIX_PATH"))
  {
    if (prefix[0] != '\0') { cfg.prefix_path = prefix; }
  }
  return cfg;
}

} // namespace dg_sim

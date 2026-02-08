#pragma once

/**
 * @file sim/io/vtk_helpers.hpp
 * @brief VTK naming/config helpers for sim-layer output.
 */

#include <filesystem>
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
  return (std::filesystem::path(case_cfg.vtk_config.prefix_path)
          / case_cfg.case_name).string();
}

/// @brief Build a consistent dataset name for ParaView output.
inline std::string MakeCaseVTKDatasetName(const CaseConfig& case_cfg,
                                          const RunConfig& run_cfg,
                                          double c_alpha_value =
                                             std::numeric_limits<double>::quiet_NaN())
{
  const int total_ref = run_cfg.opts.ser_ref_levels + run_cfg.opts.par_ref_levels;
  const double c_alpha = std::isnan(c_alpha_value)
                             ? run_cfg.igr_opts.c_alpha_user
                             : c_alpha_value;
  std::ostringstream name;
  name << case_cfg.case_name
       << "_p_" << run_cfg.opts.order
       << "_ref_" << total_ref
       << "_is_" << run_cfg.igr_opts.space
       << "_ig_" << run_cfg.igr_opts.grad
       << "_ca_" << c_alpha;
  return name.str();
}

/// @brief Apply case defaults to the VTK output config.
inline mfem::io::VTKOutputConfig MakeCaseVTKConfig(const CaseConfig& case_cfg)
{
  mfem::io::VTKOutputConfig cfg = case_cfg.vtk_config;
  cfg.prefix_path = DefaultCaseVTKOutputDir(case_cfg);
  return cfg;
}

} // namespace dg_sim

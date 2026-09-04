#pragma once

/**
 * @file sim/io/vtk_restart.hpp
 * @brief Strict restart support for MFEM ParaView PVD/PVTU/VTU output.
 */

#include <cstdint>
#include <string>

#include "mfem.hpp"

#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"

namespace dg_sim
{

struct VTKRestartRequest
{
  std::string path;
  int cycle = -1;
  long legacy_step = -1;
};

struct VTKRestartInfo
{
  std::string pvd_path;
  std::string dataset_dir;
  std::string cycle_dir;
  int cycle = -1;
  mfem::real_t time = 0.0;
  long step = -1;
  bool legacy = false;
  double reconstruction_max_abs = 0.0;
  double reconstruction_rel_l2 = 0.0;
  double mesh_point_max_abs = 0.0;
  double interpolation_condition_inf = 0.0;
  double density_min = mfem::infinity();
  double pressure_min = mfem::infinity();
  double load_seconds = 0.0;
};

/// Load one same-partition Float64 VTK cycle into a byNODES Euler L2 state.
VTKRestartInfo LoadEulerVTKRestart(const VTKRestartRequest& request,
                                   const RunConfig& run_cfg,
                                   const ExampleContext& ctx,
                                   mfem::ParGridFunction& state,
                                   bool igr_enabled,
                                   const mfem::ParGridFunction* static_source);

/// Write the versioned completion marker for a successfully saved VTK cycle.
void WriteVTKRestartManifest(const std::string& dataset_dir,
                             const std::string& dataset_name,
                             int cycle,
                             mfem::real_t time,
                             long step,
                             const RunConfig& run_cfg,
                             const ExampleContext& ctx,
                             bool sigma_saved,
                             int vtk_compression_level,
                             const mfem::ParGridFunction* static_source);

/// Stable same-partition mesh/config fingerprint used by restart manifests.
std::uint64_t ComputeVTKRestartMeshFingerprint(const RunConfig& run_cfg,
                                               const ExampleContext& ctx);

/// Stable same-partition fingerprint of a projected distributed field.
std::uint64_t ComputeVTKRestartFieldFingerprint(
  const mfem::ParGridFunction& field,
  const ExampleContext& ctx);

} // namespace dg_sim

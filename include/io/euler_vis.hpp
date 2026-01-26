#pragma once

#include "io/glvis.hpp"
#include "physics/euler/diagnostics.hpp"
#include "physics/igr/IGR.hpp"

/**
 * @file io/euler_vis.hpp
 * @brief Helper utilities for Euler + IGR visualization updates.
 * @ingroup io_visualization
 */

namespace mfem
{
namespace io
{

/// @brief GLVis windows for common Euler fields (density, sigma, energy, pressure, momentum_x).
/// @ingroup io_visualization
struct EulerIGRVisWindows
{
  vis::GLVisWindow* density = nullptr;
  vis::GLVisWindow* sigma = nullptr;
  vis::GLVisWindow* energy = nullptr;
  vis::GLVisWindow* pressure = nullptr;
  vis::GLVisWindow* momentum_x = nullptr;
};

/// @brief Field handles for common Euler quantities used in visualization.
/// @ingroup io_visualization
struct EulerIGRVisFields
{
  const mfem::ParGridFunction* density = nullptr;
  const mfem::ParGridFunction* energy = nullptr;
  const mfem::ParGridFunction* momentum_x = nullptr;
  mfem::ParGridFunction* pressure = nullptr;
};

/// @brief Update IGR (sigma), pressure field, and refresh GLVis windows.
/// @ingroup io_visualization
void UpdateEulerIGRViews(const EulerIGRVisWindows& windows,
                         const EulerIGRVisFields& fields,
                         ::dg_igr::IGRSolver& igr,
                         const mfem::ParGridFunction& state,
                         mfem::PressureCoefficient& pcoef,
                         int num_procs, int my_rank,
                         const mfem::ParMesh& pmesh,
                         double t,
                         bool warm_start);

} // namespace io
} // namespace mfem

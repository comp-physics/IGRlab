#pragma once

#include "mfem.hpp"
#include "utils/config/io_config.hpp"

/**
 * @file io/vtk_output.hpp
 * @brief ParaView/VTK output helpers.
 * @ingroup io_visualization
 */

namespace mfem
{
namespace io
{

/// @brief Common Euler field pointers for VTK export (null entries are skipped).
/// @ingroup io_visualization
struct EulerVTKFields
{
  mfem::ParGridFunction* density = nullptr;
  mfem::ParGridFunction* momentum = nullptr;
  mfem::ParGridFunction* energy = nullptr;
  mfem::ParGridFunction* velocity = nullptr;
  mfem::ParGridFunction* sigma = nullptr;
  mfem::ParGridFunction* pressure = nullptr;
  // Optional H1 diagnostics (typically projected from conservative state).
  mfem::ParGridFunction* velocity_h1 = nullptr;
  mfem::ParGridFunction* speed_h1 = nullptr;
  mfem::ParGridFunction* div_h1 = nullptr;
  mfem::ParGridFunction* grad_frob_h1 = nullptr;
  // Optional broken/L2 diagnostics from conservative state gradients.
  mfem::ParGridFunction* div_l2 = nullptr;
  mfem::ParGridFunction* grad_frob_l2 = nullptr;
  // Optional per-element MPI rank field.
  mfem::ParGridFunction* mpi_rank = nullptr;
  // Optional per-element Zhang-Shu limiter diagnostics.
  mfem::ParGridFunction* zhang_shu_active = nullptr;
  mfem::ParGridFunction* zhang_shu_theta_rho = nullptr;
  mfem::ParGridFunction* zhang_shu_theta_p = nullptr;
};

/// @brief Apply a standardized VTK output configuration to a data collection.
/// @ingroup io_visualization
void ApplyVTKConfig(mfem::ParaViewDataCollection& dc,
                    const VTKOutputConfig& cfg);

/// @brief Register common Euler fields (density, momentum, energy, sigma, pressure).
/// @ingroup io_visualization
void RegisterEulerFields(mfem::ParaViewDataCollection& dc,
                         const EulerVTKFields& fields);

} // namespace io
} // namespace mfem

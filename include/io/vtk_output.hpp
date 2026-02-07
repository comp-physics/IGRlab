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

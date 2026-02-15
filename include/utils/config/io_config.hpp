#pragma once

#include <string>
#include "mfem.hpp"

/**
 * @file utils/config/io_config.hpp
 * @brief Configuration defaults for IO helpers.
 * @ingroup utils_config
 */

namespace vis
{

/// @brief Configuration for GLVis socketstream windows.
/// @ingroup io_visualization
struct GLVisConfig
{
  std::string host = "localhost";
  int port = 19916;
  int precision = 8;
  std::string view = "view 0 0"; ///< Full GLVis view command; empty disables.
  std::string keys = "keys jlm"; ///< Full GLVis keys command; empty disables.
  bool pause = true;             ///< Send "pause" after initial draw.
};

} // namespace vis

namespace mfem
{
namespace io
{

/// @brief Configuration for ParaView/VTK output via ParaViewDataCollection.
/// @ingroup io_visualization
struct VTKOutputConfig
{
  std::string prefix_path = "vtk_output";           ///< Output directory prefix
  mfem::VTKFormat format = mfem::VTKFormat::BINARY; ///< Output format
  bool high_order_output = true; ///< Preserve high-order data
  int levels_of_detail = -1;     ///< <0 keeps current DC setting
};

} // namespace io
} // namespace mfem

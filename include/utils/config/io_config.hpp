#pragma once

#include <string>

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
  std::string view = "view 0 0";  ///< Full GLVis view command; empty disables.
  std::string keys = "keys jlm"; ///< Full GLVis keys command; empty disables.
  bool pause = true;             ///< Send "pause" after initial draw.
};

} // namespace vis

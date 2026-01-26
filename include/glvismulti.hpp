#pragma once
#include <string>
#include "mfem.hpp"

/**
 * @file glvismulti.hpp
 * @brief GLVis helper utilities for multi-window visualization.
 * @ingroup io_visualization
 */

namespace vis {

/// @brief GLVis window wrapper for a socketstream connection.
/// @ingroup io_visualization
struct GLVisWindow {
  mfem::socketstream ss;
  std::string title;
};

bool Open(GLVisWindow& w,
          const char* host, int port, int precision,
          int num_procs, int my_rank,
          const mfem::ParMesh& pmesh,
          const mfem::GridFunction& gf,
          const std::string& window_title);

void Update(GLVisWindow& w,
            int num_procs, int my_rank,
            const mfem::ParMesh& pmesh,
            const mfem::GridFunction& gf,
            double t, const std::string& name);

void Close(GLVisWindow& w);

} // namespace vis

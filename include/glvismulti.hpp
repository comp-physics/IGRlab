#pragma once
#include <string>
#include "mfem.hpp"

namespace vis {

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

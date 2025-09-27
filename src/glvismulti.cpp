#include "glvismulti.hpp"
#include <iostream>   // std::cout
#include <ostream>    // std::flush

namespace vis {

bool Open(GLVisWindow& w,
          const char* host, int port, int precision,
          int num_procs, int my_rank,
          const mfem::ParMesh& pmesh,
          const mfem::GridFunction& gf,
          const std::string& window_title)
{
  w.ss.open(host, port);
  if (!w.ss) {
    if (my_rank == 0) {
      std::cout << "Unable to connect to GLVis at " << host << ":" << port
                << " for window '" << window_title << "'.\n";
    }
    return false;
  }

  w.title = window_title;
  w.ss.precision(precision);

  w.ss << "parallel " << num_procs << " " << my_rank << "\n";
  w.ss << "solution\n" << pmesh << gf;
  w.ss << "window_title '" << window_title << "'\n";
  w.ss << "view 0 0\n";   // top view
  w.ss << "keys jlm\n";   // no perspective/light, show mesh
  w.ss << "pause\n";
  w.ss << std::flush;
  return true;
}

void Update(GLVisWindow& w,
            int num_procs, int my_rank,
            const mfem::ParMesh& pmesh,
            const mfem::GridFunction& gf,
            double t, const std::string& name)
{
  if (!w.ss) { return; }

  w.ss << "parallel " << num_procs << " " << my_rank << "\n";
  w.ss << "solution\n" << pmesh << gf;
  w.ss << "window_title '" << name << ", t = " << t << "'\n";
  w.ss << std::flush;
}

void Close(GLVisWindow& w)
{
  if (w.ss) { w.ss.close(); }
}

} // namespace vis

#include "io/glvis.hpp"
#include <iostream>   // std::cout
#include <ostream>    // std::flush

namespace vis {

namespace {

bool OpenImpl(GLVisWindow& w,
              const char* host, int port, int precision,
              int num_procs, int my_rank,
              const mfem::ParMesh& pmesh,
              const mfem::GridFunction& gf,
              const std::string& window_title,
              const std::string& view_cmd,
              const std::string& keys_cmd,
              bool pause)
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
  if (!view_cmd.empty()) { w.ss << view_cmd << "\n"; }
  if (!keys_cmd.empty()) { w.ss << keys_cmd << "\n"; }
  if (pause) { w.ss << "pause\n"; }
  w.ss << std::flush;
  return true;
}

} // namespace

bool Open(GLVisWindow& w,
          const char* host, int port, int precision,
          int num_procs, int my_rank,
          const mfem::ParMesh& pmesh,
          const mfem::GridFunction& gf,
          const std::string& window_title)
{
  return OpenImpl(w, host, port, precision, num_procs, my_rank, pmesh, gf,
                  window_title, "view 0 0", "keys jlm", true);
}

bool Open(GLVisWindow& w,
          const GLVisConfig& cfg,
          int num_procs, int my_rank,
          const mfem::ParMesh& pmesh,
          const mfem::GridFunction& gf,
          const std::string& window_title)
{
  return OpenImpl(w, cfg.host.c_str(), cfg.port, cfg.precision,
                  num_procs, my_rank, pmesh, gf, window_title,
                  cfg.view, cfg.keys, cfg.pause);
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

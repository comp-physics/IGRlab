#pragma once

/**
 * @file sim/io/save_state.hpp
 * @brief GLVis-compatible initial/final state output helpers.
 */

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "mfem.hpp"

#include "sim/context/example_context.hpp"

namespace dg_sim
{

/// @brief Save the initial mesh + conservative state in GLVis-compatible files.
inline void SaveInitialEulerState(ExampleContext& ctx,
                                  mfem::ParGridFunction& sol,
                                  int precision,
                                  const std::string& mesh_prefix = "euler-mesh",
                                  const std::string& field_prefix = "euler")
{
  std::ostringstream mesh_name;
  mesh_name << mesh_prefix << "." << std::setfill('0') << std::setw(6)
            << mfem::Mpi::WorldRank();
  std::ofstream mesh_ofs(mesh_name.str().c_str());
  mesh_ofs.precision(precision);
  mesh_ofs << ctx.Mesh();

  for (int k = 0; k < ctx.num_equations; k++)
  {
    mfem::ParGridFunction uk(&ctx.spaces.fes,
                             sol.GetData() + k * ctx.spaces.fes.GetNDofs());
    std::ostringstream sol_name;
    sol_name << field_prefix << "-" << k << "-init." << std::setfill('0')
             << std::setw(6) << mfem::Mpi::WorldRank();
    std::ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(precision);
    sol_ofs << uk;
  }
}

/// @brief Save the final mesh + conservative state in GLVis-compatible files.
inline void SaveFinalEulerState(ExampleContext& ctx,
                                mfem::ParGridFunction& sol,
                                int precision,
                                const std::string& mesh_prefix = "euler-mesh-final",
                                const std::string& field_prefix = "euler")
{
  std::ostringstream mesh_name;
  mesh_name << mesh_prefix << "." << std::setfill('0') << std::setw(6)
            << mfem::Mpi::WorldRank();
  std::ofstream mesh_ofs(mesh_name.str().c_str());
  mesh_ofs.precision(precision);
  mesh_ofs << ctx.Mesh();

  for (int k = 0; k < ctx.num_equations; k++)
  {
    mfem::ParGridFunction uk(&ctx.spaces.fes,
                             sol.GetData() + k * ctx.spaces.fes.GetNDofs());
    std::ostringstream sol_name;
    sol_name << field_prefix << "-" << k << "-final." << std::setfill('0')
             << std::setw(6) << mfem::Mpi::WorldRank();
    std::ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(precision);
    sol_ofs << uk;
  }
}

} // namespace dg_sim

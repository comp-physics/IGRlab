#pragma once

/**
 * @file sim/io/save_state.hpp
 * @brief GLVis-compatible initial/final state output helpers.
 */

#include <fstream>
#include <filesystem>
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
                                  const std::string& field_prefix = "euler",
                                  const std::string& output_dir = ".")
{
  std::filesystem::create_directories(output_dir);
  int communicator_rank = 0;
  MPI_Comm_rank(ctx.Mesh().GetComm(), &communicator_rank);

  std::ostringstream mesh_name;
  mesh_name << mesh_prefix << "." << std::setfill('0') << std::setw(6)
            << communicator_rank;
  const std::filesystem::path mesh_path =
    std::filesystem::path(output_dir) / mesh_name.str();
  std::ofstream mesh_ofs(mesh_path.string().c_str());
  mesh_ofs.precision(precision);
  mesh_ofs << ctx.Mesh();

  // The component aliases do not inherit the parent's device-validity flags.
  // Synchronize the parent before constructing host-only output views.
  const mfem::real_t* saved_state = sol.HostRead();
  for (int k = 0; k < ctx.num_equations; k++)
  {
    mfem::ParGridFunction uk(&ctx.spaces.fes,
      const_cast<mfem::real_t*>(saved_state + k * ctx.spaces.fes.GetNDofs()));
    std::ostringstream sol_name;
    sol_name << field_prefix << "-" << k << "-init." << std::setfill('0')
             << std::setw(6) << communicator_rank;
    const std::filesystem::path sol_path =
      std::filesystem::path(output_dir) / sol_name.str();
    std::ofstream sol_ofs(sol_path.string().c_str());
    sol_ofs.precision(precision);
    sol_ofs << uk;
  }
}

/// @brief Save the final mesh + conservative state in GLVis-compatible files.
inline void
SaveFinalEulerState(ExampleContext& ctx,
                    mfem::ParGridFunction& sol,
                    int precision,
                    const std::string& mesh_prefix = "euler-mesh-final",
                    const std::string& field_prefix = "euler",
                    const std::string& output_dir = ".")
{
  std::filesystem::create_directories(output_dir);
  int communicator_rank = 0;
  MPI_Comm_rank(ctx.Mesh().GetComm(), &communicator_rank);

  std::ostringstream mesh_name;
  mesh_name << mesh_prefix << "." << std::setfill('0') << std::setw(6)
            << communicator_rank;
  const std::filesystem::path mesh_path =
    std::filesystem::path(output_dir) / mesh_name.str();
  std::ofstream mesh_ofs(mesh_path.string().c_str());
  mesh_ofs.precision(precision);
  mesh_ofs << ctx.Mesh();

  const mfem::real_t* saved_state = sol.HostRead();
  for (int k = 0; k < ctx.num_equations; k++)
  {
    mfem::ParGridFunction uk(&ctx.spaces.fes,
      const_cast<mfem::real_t*>(saved_state + k * ctx.spaces.fes.GetNDofs()));
    std::ostringstream sol_name;
    sol_name << field_prefix << "-" << k << "-final." << std::setfill('0')
             << std::setw(6) << communicator_rank;
    const std::filesystem::path sol_path =
      std::filesystem::path(output_dir) / sol_name.str();
    std::ofstream sol_ofs(sol_path.string().c_str());
    sol_ofs.precision(precision);
    sol_ofs << uk;
  }
}

} // namespace dg_sim

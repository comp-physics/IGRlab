#pragma once

/**
 * @file sim/context/mesh_build.hpp
 * @brief Mesh loading/refinement helpers and alpha diagnostics.
 */

#include <algorithm>

#include "mfem.hpp"

namespace dg_sim
{

/// @brief Alpha diagnostics used in case naming and solver setup.
struct AlphaInfo
{
  mfem::real_t hmin = mfem::infinity();
  double c_alpha = 1.0;
  mfem::real_t alpha = 1.0;
};

inline mfem::Mesh LoadSerialMesh(const std::string& mesh_path, int ser_ref_levels)
{
  mfem::Mesh mesh(mesh_path);
  for (int lev = 0; lev < ser_ref_levels; ++lev)
  {
    mesh.UniformRefinement();
  }
  return mesh;
}

inline void RefineParMesh(mfem::ParMesh& pmesh, int par_ref_levels)
{
  for (int lev = 0; lev < par_ref_levels; ++lev)
  {
    pmesh.UniformRefinement();
  }
}

inline AlphaInfo ComputeAlphaInfo(const mfem::ParMesh& pmesh, double c_alpha_user)
{
  AlphaInfo info;
  for (int i = 0; i < pmesh.GetNE(); ++i)
  {
    auto& mesh = const_cast<mfem::ParMesh&>(pmesh);
    info.hmin = std::min(info.hmin, mesh.GetElementSize(i, 1));
  }
  MPI_Allreduce(MPI_IN_PLACE, &info.hmin, 1,
                mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_MIN, pmesh.GetComm());

  info.c_alpha = (c_alpha_user > 0.0) ? c_alpha_user : 1.0;
  const mfem::real_t dx_eff = info.hmin;
  info.alpha = static_cast<mfem::real_t>(info.c_alpha * dx_eff * dx_eff);
  return info;
}

} // namespace dg_sim

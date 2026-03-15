#pragma once

/**
 * @file sim/context/mesh_build.hpp
 * @brief Mesh loading/refinement helpers and alpha diagnostics.
 */

#include <algorithm>
#include <limits>

#include "mfem.hpp"
#include "sim/config/case_config.hpp"

namespace dg_sim
{

/// @brief Alpha diagnostics used in case naming and solver setup.
struct AlphaInfo
{
  mfem::real_t hmin = mfem::infinity();
  mfem::real_t havg = mfem::infinity();
  mfem::real_t hmax = 0.0;
  mfem::real_t dx_eff = mfem::infinity();
  double c_alpha = 1.0;
  mfem::real_t alpha = 1.0;
  dg_igr::IGRAlphaLengthScalePolicy policy =
    dg_igr::IGRAlphaLengthScalePolicy::Average;
};

inline mfem::Mesh LoadSerialMesh(const std::string& mesh_path,
                                 int ser_ref_levels)
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

inline AlphaInfo ComputeAlphaInfo(const mfem::ParMesh& pmesh,
                                  double c_alpha_user,
                                  dg_igr::IGRAlphaLengthScalePolicy policy)
{
  AlphaInfo info;
  info.policy = policy;
  mfem::real_t hsum = 0.0;
  long long local_ne = 0;
  for (int i = 0; i < pmesh.GetNE(); ++i)
  {
    auto& mesh = const_cast<mfem::ParMesh&>(pmesh);
    const mfem::real_t h = mesh.GetElementSize(i, 1);
    info.hmin = std::min(info.hmin, h);
    info.hmax = std::max(info.hmax, h);
    hsum += h;
    ++local_ne;
  }
  MPI_Allreduce(MPI_IN_PLACE,
                &info.hmin,
                1,
                mfem::MPITypeMap<mfem::real_t>::mpi_type,
                MPI_MIN,
                pmesh.GetComm());
  MPI_Allreduce(MPI_IN_PLACE,
                &hsum,
                1,
                mfem::MPITypeMap<mfem::real_t>::mpi_type,
                MPI_SUM,
                pmesh.GetComm());
  MPI_Allreduce(MPI_IN_PLACE,
                &info.hmax,
                1,
                mfem::MPITypeMap<mfem::real_t>::mpi_type,
                MPI_MAX,
                pmesh.GetComm());
  long long global_ne = local_ne;
  MPI_Allreduce(MPI_IN_PLACE, &global_ne, 1, MPI_LONG_LONG, MPI_SUM, pmesh.GetComm());

  info.c_alpha = (c_alpha_user > 0.0) ? c_alpha_user : 1.0;
  info.havg = (global_ne > 0)
                ? (hsum / static_cast<mfem::real_t>(global_ne))
                : info.hmin;
  switch (policy)
  {
  case dg_igr::IGRAlphaLengthScalePolicy::Min:
    info.dx_eff = info.hmin;
    info.alpha = static_cast<mfem::real_t>(info.c_alpha * info.dx_eff *
                                           info.dx_eff);
    break;
  case dg_igr::IGRAlphaLengthScalePolicy::Average:
    info.dx_eff = info.havg;
    info.alpha = static_cast<mfem::real_t>(info.c_alpha * info.dx_eff *
                                           info.dx_eff);
    break;
  case dg_igr::IGRAlphaLengthScalePolicy::Max:
    info.dx_eff = info.hmax;
    info.alpha = static_cast<mfem::real_t>(info.c_alpha * info.dx_eff *
                                           info.dx_eff);
    break;
  case dg_igr::IGRAlphaLengthScalePolicy::Local:
    info.dx_eff = info.havg;
    info.alpha = static_cast<mfem::real_t>(info.c_alpha);
    break;
  }
  return info;
}

} // namespace dg_sim

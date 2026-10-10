#pragma once

#include "igrlab/compat.hpp"

/**
 * @file sim/context/mesh_build.hpp
 * @brief Neutral mesh loading/refinement helpers and alpha diagnostics.
 */

#include <algorithm>
#include <limits>
#include <string>

#include "mfem.hpp"
#include "sim/config/case_config.hpp"

namespace dg_sim
{

/// @brief Where the resolved mesh path came from in the case configuration.
enum class MeshPathPrecedence
{
  CaseDefault,
  CommandLineOverride,
  AlreadyResolved
};

/// @brief Input to the neutral mesh construction layer.
struct MeshBuildRequest
{
  std::string mesh_path;
  int serial_refinement_levels = 0;
  int parallel_refinement_levels = 0;
  MeshPathPrecedence mesh_path_precedence = MeshPathPrecedence::AlreadyResolved;
};

/// @brief Record of the resolved mesh input and refinement sequence.
///
/// The element counts deliberately describe only the local rank.  They make no
/// claim about a partitioning policy: MFEM chooses its normal default when the
/// serial mesh becomes a ParMesh.
struct MeshBuildMetadata
{
  std::string mesh_path;
  int serial_refinement_levels = 0;
  int parallel_refinement_levels = 0;
  MeshPathPrecedence mesh_path_precedence = MeshPathPrecedence::AlreadyResolved;
  int serial_element_count = 0;
  int local_elements_after_partitioning = 0;
  int local_elements_after_parallel_refinement = 0;
};

/// @brief Resolve the mesh request after case options have been parsed.
inline MeshBuildRequest ResolveMeshBuildRequest(const RunConfig& run_cfg)
{
  MeshBuildRequest request;
  request.serial_refinement_levels = run_cfg.opts.ser_ref_levels;
  request.parallel_refinement_levels = run_cfg.opts.par_ref_levels;
  request.mesh_path_precedence = run_cfg.opts.mesh_file.empty()
                                  ? MeshPathPrecedence::CaseDefault
                                  : MeshPathPrecedence::CommandLineOverride;
  request.mesh_path = run_cfg.mesh_path.empty()
                        ? ResolveCaseMesh(run_cfg.case_config,
                                          run_cfg.opts.mesh_file)
                        : run_cfg.mesh_path;
  return request;
}

/// @brief Alpha diagnostics used in case naming and solver setup.
struct AlphaInfo
{
  mfem::real_t hmin = mfem::infinity();
  mfem::real_t havg = mfem::infinity();
  mfem::real_t hmax = 0.0;
  mfem::real_t dx_eff = mfem::infinity();
  double c_alpha = 1.0;
  mfem::real_t alpha = 1.0;
  igrlab::IGRAlphaLengthScalePolicy policy =
    igrlab::IGRAlphaLengthScalePolicy::Average;
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
                                  igrlab::IGRAlphaLengthScalePolicy policy)
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
  case igrlab::IGRAlphaLengthScalePolicy::Min:
    info.dx_eff = info.hmin;
    info.alpha = static_cast<mfem::real_t>(info.c_alpha * info.dx_eff *
                                           info.dx_eff);
    break;
  case igrlab::IGRAlphaLengthScalePolicy::Average:
    info.dx_eff = info.havg;
    info.alpha = static_cast<mfem::real_t>(info.c_alpha * info.dx_eff *
                                           info.dx_eff);
    break;
  case igrlab::IGRAlphaLengthScalePolicy::Max:
    info.dx_eff = info.hmax;
    info.alpha = static_cast<mfem::real_t>(info.c_alpha * info.dx_eff *
                                           info.dx_eff);
    break;
  case igrlab::IGRAlphaLengthScalePolicy::Local:
    info.dx_eff = info.havg;
    info.alpha = static_cast<mfem::real_t>(info.c_alpha);
    break;
  }
  return info;
}

} // namespace dg_sim

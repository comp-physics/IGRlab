#pragma once

/**
 * @file sim/context/case_context.hpp
 * @brief Backend-neutral case context for resolved mesh ownership.
 */

#include <memory>
#include <utility>

#include "mfem.hpp"

#include "sim/context/mesh_build.hpp"

namespace dg_sim
{

/// @brief Owns a resolved parallel mesh and exposes its communicator.
///
/// This context deliberately owns no solver, field, time-loop, or numerical
/// backend state.  A backend may layer its spaces over Mesh() after this
/// neutral construction has completed.
class CaseContext
{
public:
  CaseContext(std::unique_ptr<mfem::ParMesh> mesh,
              MeshBuildMetadata mesh_metadata)
    : mesh_(std::move(mesh)),
      mesh_metadata_(std::move(mesh_metadata))
  {
    MFEM_VERIFY(mesh_, "CaseContext requires a resolved parallel mesh.");
  }

  CaseContext(const CaseContext&) = delete;
  CaseContext& operator=(const CaseContext&) = delete;
  CaseContext(CaseContext&&) noexcept = default;
  CaseContext& operator=(CaseContext&&) noexcept = default;

  mfem::ParMesh& Mesh() { return *mesh_; }
  const mfem::ParMesh& Mesh() const { return *mesh_; }
  MPI_Comm Comm() const { return Mesh().GetComm(); }
  const MeshBuildMetadata& MeshMetadata() const { return mesh_metadata_; }

private:
  std::unique_ptr<mfem::ParMesh> mesh_;
  MeshBuildMetadata mesh_metadata_;
};

/// @brief Build a neutral case context using MFEM's default ParMesh partitioning.
inline CaseContext BuildCaseContext(const MeshBuildRequest& request,
                                    MPI_Comm communicator = MPI_COMM_WORLD)
{
  MFEM_VERIFY(!request.mesh_path.empty(), "A case mesh path is required.");
  MFEM_VERIFY(request.serial_refinement_levels >= 0,
              "Serial refinement levels must be non-negative.");
  MFEM_VERIFY(request.parallel_refinement_levels >= 0,
              "Parallel refinement levels must be non-negative.");

  mfem::Mesh serial_mesh =
    LoadSerialMesh(request.mesh_path, request.serial_refinement_levels);
  MeshBuildMetadata metadata;
  metadata.mesh_path = request.mesh_path;
  metadata.serial_refinement_levels = request.serial_refinement_levels;
  metadata.parallel_refinement_levels = request.parallel_refinement_levels;
  metadata.mesh_path_precedence = request.mesh_path_precedence;
  metadata.serial_element_count = serial_mesh.GetNE();

  // Do not supply a partitioner: this is MFEM's default ParMesh construction.
  auto parallel_mesh =
    std::make_unique<mfem::ParMesh>(communicator, serial_mesh);
  serial_mesh.Clear();
  metadata.local_elements_after_partitioning = parallel_mesh->GetNE();

  // Refinement happens only after MFEM has constructed the parallel mesh.
  RefineParMesh(*parallel_mesh, request.parallel_refinement_levels);
  metadata.local_elements_after_parallel_refinement = parallel_mesh->GetNE();

  return CaseContext(std::move(parallel_mesh), std::move(metadata));
}

/// @brief Convenience overload for an already resolved sim-layer run config.
inline CaseContext BuildCaseContext(const RunConfig& run_cfg,
                                    MPI_Comm communicator = MPI_COMM_WORLD)
{
  return BuildCaseContext(ResolveMeshBuildRequest(run_cfg), communicator);
}

} // namespace dg_sim

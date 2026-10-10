#include "sim/context/case_context.hpp"
#include "sim/wiring/euler_boundary_plan.hpp"

#include <memory>
#include <type_traits>

static_assert(std::is_constructible_v<dg_sim::CaseContext,
                                      std::unique_ptr<mfem::ParMesh>,
                                      dg_sim::MeshBuildMetadata>);
static_assert(!std::is_constructible_v<dg_sim::CaseContext,
                                       std::unique_ptr<mfem::ParMesh>,
                                       MPI_Comm,
                                       dg_sim::MeshBuildMetadata>);

int main()
{
  dg_sim::MeshBuildRequest request;
  request.mesh_path = "mesh.mesh";
  request.serial_refinement_levels = 1;
  request.parallel_refinement_levels = 2;

  mfem::Array<int> marker(1);
  marker = 1;
  dg_sim::EulerBoundaryPlan plan;
  plan.AddPeriodic(marker);
  plan.AddSlipWall(marker);
  return plan.Entries().size() == 2 ? 0 : 1;
}

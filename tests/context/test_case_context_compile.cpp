#include "sim/context/case_context.hpp"
#include "sim/wiring/euler_boundary_plan.hpp"

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

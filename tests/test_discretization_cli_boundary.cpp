#include "igrlab/environment.hpp"
#include "sim/solver/solver.hpp"

#include <cstdlib>
#include <fstream>
#include <memory>

namespace
{

void MarkPostSetupCallbackReached()
{
  const char* const path = igrlab::GetEnv("DG_IGR_CONFIG_BOUNDARY_SENTINEL");
  if (path)
  {
    std::ofstream(path) << "make_ic reached\n";
  }
}

} // namespace

int main(int argc, char* argv[])
{
  dg_sim::CaseSpec<dg_sim::EmptyCaseOptions, mfem::DGSEMSurfaceFlux> spec;
  spec.case_name = "discretization-cli-boundary";
  spec.default_mesh_path = "must-not-open.mesh";
  spec.use_igr = false;
  spec.glvis.enabled = false;
  spec.vtk.enabled = false;
  spec.make_ic = [](const dg_sim::EmptyCaseOptions&,
                    const dg_sim::ExampleContext&,
                    double)
  {
    MarkPostSetupCallbackReached();
    return std::make_unique<mfem::VectorFunctionCoefficient>(
      4,
      [](const mfem::Vector&, mfem::Vector&) {});
  };

  return dg_sim::Solver::solve(spec, argc, argv);
}

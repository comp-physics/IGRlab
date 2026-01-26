#include "io/euler_vis.hpp"

namespace mfem
{
namespace io
{

void UpdateEulerIGRViews(const EulerIGRVisWindows& windows,
                         const EulerIGRVisFields& fields,
                         ::dg_igr::IGRSolver& igr,
                         const mfem::ParGridFunction& state,
                         mfem::PressureCoefficient& pcoef,
                         int num_procs, int my_rank,
                         const mfem::ParMesh& pmesh,
                         double t,
                         bool warm_start)
{
  igr.UpdateFromState(state, warm_start);
  igr.GetSigma().ExchangeFaceNbrData();

  if (fields.pressure)
  {
    pcoef.SetState(state);
    fields.pressure->ProjectCoefficient(pcoef);
  }

  if (windows.density && fields.density)
  {
    vis::Update(*windows.density, num_procs, my_rank, pmesh, *fields.density, t, "density");
  }
  if (windows.sigma)
  {
    vis::Update(*windows.sigma, num_procs, my_rank, pmesh, igr.GetSigma(), t, "sigma");
  }
  if (windows.energy && fields.energy)
  {
    vis::Update(*windows.energy, num_procs, my_rank, pmesh, *fields.energy, t, "energy");
  }
  if (windows.pressure && fields.pressure)
  {
    vis::Update(*windows.pressure, num_procs, my_rank, pmesh, *fields.pressure, t, "pressure");
  }
  if (windows.momentum_x && fields.momentum_x)
  {
    vis::Update(*windows.momentum_x, num_procs, my_rank, pmesh, *fields.momentum_x, t, "momentum_x");
  }
}

} // namespace io
} // namespace mfem

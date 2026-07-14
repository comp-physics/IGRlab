#include "io/euler_vis.hpp"

namespace mfem
{
namespace io
{

void UpdateEulerIGRViews(const EulerIGRVisWindows& windows,
                         const EulerIGRVisFields& fields,
                         int num_procs,
                         int my_rank,
                         const mfem::ParMesh& pmesh,
                         double t)
{
  if (windows.density && fields.density)
  {
    vis::Update(*windows.density,
                num_procs,
                my_rank,
                pmesh,
                *fields.density,
                t,
                "density");
  }
  if (windows.sigma && fields.sigma)
  {
    vis::Update(*windows.sigma,
                num_procs,
                my_rank,
                pmesh,
                *fields.sigma,
                t,
                "sigma");
  }
  if (windows.energy && fields.energy)
  {
    vis::Update(*windows.energy,
                num_procs,
                my_rank,
                pmesh,
                *fields.energy,
                t,
                "energy");
  }
  if (windows.pressure && fields.pressure)
  {
    vis::Update(*windows.pressure,
                num_procs,
                my_rank,
                pmesh,
                *fields.pressure,
                t,
                "pressure");
  }
  if (windows.momentum_x && fields.momentum_x)
  {
    vis::Update(*windows.momentum_x,
                num_procs,
                my_rank,
                pmesh,
                *fields.momentum_x,
                t,
                "momentum_x");
  }
}

} // namespace io
} // namespace mfem

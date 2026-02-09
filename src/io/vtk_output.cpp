#include "io/vtk_output.hpp"

namespace mfem
{
namespace io
{

void ApplyVTKConfig(mfem::ParaViewDataCollection& dc,
                    const VTKOutputConfig& cfg)
{
  dc.SetPrefixPath(cfg.prefix_path);
  if (cfg.levels_of_detail >= 0) { dc.SetLevelsOfDetail(cfg.levels_of_detail); }
  dc.SetDataFormat(cfg.format);
  dc.SetHighOrderOutput(cfg.high_order_output);
}

void RegisterEulerFields(mfem::ParaViewDataCollection& dc,
                         const EulerVTKFields& fields)
{
  if (fields.density) { dc.RegisterField("density", fields.density); }
  if (fields.momentum) { dc.RegisterField("momentum", fields.momentum); }
  if (fields.energy) { dc.RegisterField("energy", fields.energy); }
  if (fields.velocity) { dc.RegisterField("velocity", fields.velocity); }
  if (fields.sigma) { dc.RegisterField("sigma", fields.sigma); }
  if (fields.pressure) { dc.RegisterField("pressure", fields.pressure); }
  if (fields.velocity_h1) { dc.RegisterField("velocity_h1", fields.velocity_h1); }
  if (fields.speed_h1) { dc.RegisterField("speed_h1", fields.speed_h1); }
  if (fields.div_h1) { dc.RegisterField("div_h1", fields.div_h1); }
  if (fields.grad_frob_h1) { dc.RegisterField("grad_frob_h1", fields.grad_frob_h1); }
}

} // namespace io
} // namespace mfem

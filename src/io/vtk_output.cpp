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
  if (fields.momentum_x) { dc.RegisterField("momentum_x", fields.momentum_x); }
  if (fields.momentum_y) { dc.RegisterField("momentum_y", fields.momentum_y); }
  if (fields.energy) { dc.RegisterField("energy", fields.energy); }
  if (fields.sigma) { dc.RegisterField("sigma", fields.sigma); }
  if (fields.pressure) { dc.RegisterField("pressure", fields.pressure); }
}

} // namespace io
} // namespace mfem

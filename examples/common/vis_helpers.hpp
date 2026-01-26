#pragma once

#include <utility>

#include "mfem.hpp"

#include "io/euler_vis.hpp"
#include "io/vtk_output.hpp"

#include "state_helpers.hpp"

namespace dg_igr_examples
{

/// @brief Field handles for common Euler visualization/export helpers.
struct EulerFieldList
{
  EulerStateViews* state = nullptr;
  EulerPressureView* pressure = nullptr;
  dg_igr::IGRSolver* igr = nullptr;
};

/// @brief GLVis windows for common Euler fields.
struct EulerVisWindows
{
  vis::GLVisWindow density;
  vis::GLVisWindow sigma;
  vis::GLVisWindow energy;
  vis::GLVisWindow pressure;
  vis::GLVisWindow momentum_x;
};

/// @brief H1 velocity diagnostics for Euler runs.
struct H1Diagnostics
{
  mfem::ParGridFunction u_h1;
  mfem::ParGridFunction g_umag;
  mfem::ParGridFunction g_div;
  mfem::ParGridFunction g_gfrob;
  mfem::VelocityFromConsCoefficient ucoef;
  int dim = 0;

  H1Diagnostics(mfem::ParFiniteElementSpace& h1_vfes,
                mfem::ParGridFunction& sol,
                int dim_in,
                double rho_floor)
    : u_h1(&h1_vfes),
      g_umag(&h1_vfes),
      g_div(&h1_vfes),
      g_gfrob(&h1_vfes),
      ucoef(dim_in, &sol, rho_floor),
      dim(dim_in)
  {
    u_h1 = 0.0;
    g_umag = 0.0;
    g_div = 0.0;
    g_gfrob = 0.0;
  }

  void Update()
  {
    u_h1.ProjectCoefficient(ucoef);
    mfem::GradFrobNormCoeff cF(&u_h1, dim);
    mfem::DivCoeff cD(&u_h1, dim);
    mfem::VelMagCoeff cM(&u_h1, dim);
    g_gfrob.ProjectCoefficient(cF);
    g_div.ProjectCoefficient(cD);
    g_umag.ProjectCoefficient(cM);
  }
};

inline bool OpenEulerIGRGLVisWindows(EulerVisWindows& windows,
                                     const EulerFieldList& fields,
                                     const mfem::ParMesh& pmesh,
                                     int num_procs,
                                     int my_rank,
                                     int precision,
                                     const char* host = "localhost",
                                     int port = 19916)
{
  if (!fields.state || !fields.pressure || !fields.igr)
  {
    return false;
  }

  fields.igr->GetSigma().ExchangeFaceNbrData();

  const bool ok1 = vis::Open(windows.density, host, port, precision,
                             num_procs, my_rank, pmesh, fields.state->density,
                             "density, t = 0");
  const bool ok2 = vis::Open(windows.sigma, host, port, precision,
                             num_procs, my_rank, pmesh, fields.igr->GetSigma(),
                             "sigma, t = 0");
  const bool ok3 = vis::Open(windows.energy, host, port, precision,
                             num_procs, my_rank, pmesh, fields.state->energy,
                             "energy, t = 0");
  const bool ok4 = vis::Open(windows.pressure, host, port, precision,
                             num_procs, my_rank, pmesh, fields.pressure->pressure,
                             "pressure, t = 0");
  const bool ok5 = vis::Open(windows.momentum_x, host, port, precision,
                             num_procs, my_rank, pmesh, fields.state->momentum_x,
                             "momentum_x, t = 0");
  (void)ok4;
  (void)ok5;
  return ok1 && ok2 && ok3;
}

inline bool OpenEulerGLVisWindows(EulerVisWindows& windows,
                                  const EulerFieldList& fields,
                                  const mfem::ParMesh& pmesh,
                                  int num_procs,
                                  int my_rank,
                                  int precision,
                                  const char* host = "localhost",
                                  int port = 19916)
{
  if (!fields.state || !fields.pressure)
  {
    return false;
  }

  const bool ok1 = vis::Open(windows.density, host, port, precision,
                             num_procs, my_rank, pmesh, fields.state->density,
                             "density, t = 0");
  const bool ok2 = vis::Open(windows.energy, host, port, precision,
                             num_procs, my_rank, pmesh, fields.state->energy,
                             "energy, t = 0");
  const bool ok3 = vis::Open(windows.pressure, host, port, precision,
                             num_procs, my_rank, pmesh, fields.pressure->pressure,
                             "pressure, t = 0");
  const bool ok4 = vis::Open(windows.momentum_x, host, port, precision,
                             num_procs, my_rank, pmesh, fields.state->momentum_x,
                             "momentum_x, t = 0");
  (void)ok3;
  (void)ok4;
  return ok1 && ok2;
}

inline mfem::io::EulerVTKFields MakeEulerVTKFields(const EulerFieldList& fields)
{
  mfem::io::EulerVTKFields out;
  if (fields.state)
  {
    out.density = &fields.state->density;
    out.momentum_x = &fields.state->momentum_x;
    out.momentum_y = &fields.state->momentum_y;
    out.energy = &fields.state->energy;
  }
  if (fields.igr)
  {
    out.sigma = &fields.igr->GetSigma();
  }
  if (fields.pressure)
  {
    out.pressure = &fields.pressure->pressure;
  }
  return out;
}

inline void RegisterEulerVTKFields(mfem::ParaViewDataCollection& dc,
                                   const EulerFieldList& fields)
{
  mfem::io::RegisterEulerFields(dc, MakeEulerVTKFields(fields));
}

inline void ApplyCaseVTKConfig(mfem::ParaViewDataCollection& dc,
                               const mfem::io::VTKOutputConfig& cfg,
                               int order)
{
  mfem::io::VTKOutputConfig local_cfg = cfg;
  local_cfg.levels_of_detail = order;
  mfem::io::ApplyVTKConfig(dc, local_cfg);
}

inline mfem::io::EulerIGRVisFields MakeEulerIGRVisFields(const EulerFieldList& fields)
{
  mfem::io::EulerIGRVisFields out;
  if (fields.state)
  {
    out.density = &fields.state->density;
    out.energy = &fields.state->energy;
    out.momentum_x = &fields.state->momentum_x;
  }
  if (fields.pressure)
  {
    out.pressure = &fields.pressure->pressure;
  }
  return out;
}

inline mfem::io::EulerIGRVisWindows MakeEulerIGRVisWindows(EulerVisWindows& windows)
{
  mfem::io::EulerIGRVisWindows out;
  out.density = &windows.density;
  out.sigma = &windows.sigma;
  out.energy = &windows.energy;
  out.pressure = &windows.pressure;
  out.momentum_x = &windows.momentum_x;
  return out;
}

inline void UpdateEulerOutputFields(const EulerFieldList& fields,
                                    mfem::ParGridFunction& state,
                                    bool warm_start)
{
  if (fields.igr)
  {
    fields.igr->UpdateFromState(state, warm_start);
    fields.igr->GetSigma().ExchangeFaceNbrData();
  }
  if (fields.pressure)
  {
    fields.pressure->Update();
  }
}

inline void UpdateEulerGLVis(EulerVisWindows& windows,
                             const EulerFieldList& fields,
                             mfem::ParGridFunction& state,
                             int num_procs, int my_rank,
                             const mfem::ParMesh& pmesh,
                             double t,
                             bool warm_start)
{
  if (fields.igr)
  {
    MFEM_VERIFY(fields.pressure != nullptr,
                "Euler GLVis update with IGR expects a pressure view.");
    const mfem::io::EulerIGRVisFields vis_fields = MakeEulerIGRVisFields(fields);
    const mfem::io::EulerIGRVisWindows vis_windows = MakeEulerIGRVisWindows(windows);
    mfem::io::UpdateEulerIGRViews(vis_windows, vis_fields, *fields.igr, state,
                                  fields.pressure->coeff, num_procs, my_rank, pmesh,
                                  t, warm_start);
    return;
  }

  if (fields.pressure)
  {
    fields.pressure->Update();
  }
  if (!fields.state)
  {
    return;
  }

  vis::Update(windows.density, num_procs, my_rank, pmesh,
              fields.state->density, t, "density");
  vis::Update(windows.energy, num_procs, my_rank, pmesh,
              fields.state->energy, t, "energy");
  if (fields.pressure)
  {
    vis::Update(windows.pressure, num_procs, my_rank, pmesh,
                fields.pressure->pressure, t, "pressure");
  }
  vis::Update(windows.momentum_x, num_procs, my_rank, pmesh,
              fields.state->momentum_x, t, "momentum_x");
}

} // namespace dg_igr_examples

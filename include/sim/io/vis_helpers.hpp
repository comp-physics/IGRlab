#pragma once

#include "igrlab/compat.hpp"

/**
 * @file sim/io/vis_helpers.hpp
 * @brief GLVis/VTK field helpers for Euler visualization.
 */

#include <chrono>
#include <memory>
#include <utility>

#include "mfem.hpp"

#include "io/euler_vis.hpp"
#include "io/vtk_output.hpp"

#include "sim/state/state_views.hpp"

namespace igrlab
{
class IGRSolver;
}

namespace dg_sim
{

struct H1Diagnostics;
struct L2Diagnostics;
struct MPIRankField;
struct ZhangShuLimiterField;

/// @brief Field handles for common Euler visualization/export helpers.
struct EulerFieldList
{
  EulerStateViews* state = nullptr;
  EulerPressureView* pressure = nullptr;
  EulerVelocityView* velocity = nullptr;
  igrlab::IGRSolver* igr = nullptr;
  const mfem::ParGridFunction* sigma = nullptr;
  H1Diagnostics* h1_diag = nullptr;
  L2Diagnostics* l2_diag = nullptr;
  MPIRankField* mpi_rank = nullptr;
  ZhangShuLimiterField* zhang_shu = nullptr;
};

struct EulerOutputFieldTiming
{
  double igr_update_s = 0.0;
  double sigma_exchange_s = 0.0;
  double pressure_s = 0.0;
  double velocity_s = 0.0;
  double h1_diag_s = 0.0;
  double l2_diag_s = 0.0;
  double total_s = 0.0;
};

inline double VisSecondsSince(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
    .count();
}

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
  std::unique_ptr<mfem::ParFiniteElementSpace> h1_scalar_fes;
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
      h1_scalar_fes(
        std::make_unique<mfem::ParFiniteElementSpace>(h1_vfes.GetParMesh(),
                                                      h1_vfes.FEColl(),
                                                      /*vdim=*/1,
                                                      mfem::Ordering::byNODES)),
      g_umag(h1_scalar_fes.get()),
      g_div(h1_scalar_fes.get()),
      g_gfrob(h1_scalar_fes.get()),
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
    // u = m/rho is discontinuous across DG faces; average shared H1 dofs
    // arithmetically so the projected field is partition-independent.
    u_h1.ProjectDiscCoefficient(ucoef, mfem::GridFunction::ARITHMETIC);
    mfem::GradFrobNormCoeff cF(&u_h1, dim);
    mfem::DivCoeff cD(&u_h1, dim);
    mfem::VelMagCoeff cM(&u_h1, dim);
    // grad-based diagnostics are also discontinuous; keep H1 exports
    // partition-consistent by using discontinuous projection with averaging.
    g_gfrob.ProjectDiscCoefficient(cF, mfem::GridFunction::ARITHMETIC);
    g_div.ProjectDiscCoefficient(cD, mfem::GridFunction::ARITHMETIC);
    g_umag.ProjectDiscCoefficient(cM, mfem::GridFunction::ARITHMETIC);
  }
};

/// @brief Broken/L2 diagnostics computed directly from conservative state gradients.
struct L2Diagnostics
{
  mfem::ParGridFunction g_div_l2;
  mfem::ParGridFunction g_gfrob_l2;
  mfem::DivFromConservativeCoeff c_div;
  mfem::GradFrobFromConservativeCoeff c_gfrob;

  L2Diagnostics(mfem::ParFiniteElementSpace& scalar_fes,
                mfem::ParGridFunction& sol,
                int dim_in,
                double rho_floor)
    : g_div_l2(&scalar_fes),
      g_gfrob_l2(&scalar_fes),
      c_div(&sol, dim_in, rho_floor),
      c_gfrob(&sol, dim_in, rho_floor)
  {
    g_div_l2 = 0.0;
    g_gfrob_l2 = 0.0;
  }

  void Update()
  {
    g_div_l2.ProjectCoefficient(c_div);
    g_gfrob_l2.ProjectCoefficient(c_gfrob);
  }
};

/// @brief Per-element MPI rank as an order-0 discontinuous field.
struct MPIRankField
{
  std::unique_ptr<mfem::L2_FECollection> fec0;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes0;
  mfem::ParGridFunction rank;

  explicit MPIRankField(mfem::ParMesh& mesh)
    : fec0(std::make_unique<mfem::L2_FECollection>(0, mesh.Dimension())),
      fes0(std::make_unique<mfem::ParFiniteElementSpace>(&mesh, fec0.get())),
      rank(fes0.get())
  {
    int communicator_rank = 0;
    MPI_Comm_rank(mesh.GetComm(), &communicator_rank);
    rank = static_cast<double>(communicator_rank);
  }
};

/// @brief Per-element Zhang-Shu limiter activity exported as order-0 DG fields.
struct ZhangShuLimiterField
{
  std::unique_ptr<mfem::L2_FECollection> fec0;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes0;
  mfem::ParGridFunction active;
  mfem::ParGridFunction theta_rho;
  mfem::ParGridFunction theta_p;

  explicit ZhangShuLimiterField(mfem::ParMesh& mesh)
    : fec0(std::make_unique<mfem::L2_FECollection>(0, mesh.Dimension())),
      fes0(std::make_unique<mfem::ParFiniteElementSpace>(&mesh, fec0.get())),
      active(fes0.get()),
      theta_rho(fes0.get()),
      theta_p(fes0.get())
  {
    Reset();
  }

  void Reset()
  {
    active = 0.0;
    theta_rho = 1.0;
    theta_p = 1.0;
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
  if (!fields.state || !fields.pressure || !fields.sigma)
  {
    return false;
  }

  const bool ok1 = vis::Open(windows.density,
                             host,
                             port,
                             precision,
                             num_procs,
                             my_rank,
                             pmesh,
                             fields.state->density,
                             "density, t = 0");
  const bool ok2 = vis::Open(windows.sigma,
                             host,
                             port,
                             precision,
                             num_procs,
                             my_rank,
                             pmesh,
                             *fields.sigma,
                             "sigma, t = 0");
  const bool ok3 = vis::Open(windows.energy,
                             host,
                             port,
                             precision,
                             num_procs,
                             my_rank,
                             pmesh,
                             fields.state->energy,
                             "energy, t = 0");
  const bool ok4 = vis::Open(windows.pressure,
                             host,
                             port,
                             precision,
                             num_procs,
                             my_rank,
                             pmesh,
                             fields.pressure->pressure,
                             "pressure, t = 0");
  const bool ok5 = vis::Open(windows.momentum_x,
                             host,
                             port,
                             precision,
                             num_procs,
                             my_rank,
                             pmesh,
                             fields.state->momentum_x,
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

  const bool ok1 = vis::Open(windows.density,
                             host,
                             port,
                             precision,
                             num_procs,
                             my_rank,
                             pmesh,
                             fields.state->density,
                             "density, t = 0");
  const bool ok2 = vis::Open(windows.energy,
                             host,
                             port,
                             precision,
                             num_procs,
                             my_rank,
                             pmesh,
                             fields.state->energy,
                             "energy, t = 0");
  const bool ok3 = vis::Open(windows.pressure,
                             host,
                             port,
                             precision,
                             num_procs,
                             my_rank,
                             pmesh,
                             fields.pressure->pressure,
                             "pressure, t = 0");
  const bool ok4 = vis::Open(windows.momentum_x,
                             host,
                             port,
                             precision,
                             num_procs,
                             my_rank,
                             pmesh,
                             fields.state->momentum_x,
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
    out.momentum = &fields.state->momentum;
    out.energy = &fields.state->energy;
  }
  if (fields.velocity)
  {
    out.velocity = &fields.velocity->velocity;
  }
  if (fields.sigma)
  {
    out.sigma = fields.sigma;
  }
  if (fields.pressure)
  {
    out.pressure = &fields.pressure->pressure;
    out.pressure_coefficient = &fields.pressure->coeff;
  }
  if (fields.h1_diag)
  {
    out.velocity_h1 = &fields.h1_diag->u_h1;
    out.speed_h1 = &fields.h1_diag->g_umag;
    out.div_h1 = &fields.h1_diag->g_div;
    out.grad_frob_h1 = &fields.h1_diag->g_gfrob;
  }
  if (fields.l2_diag)
  {
    out.div_l2 = &fields.l2_diag->g_div_l2;
    out.grad_frob_l2 = &fields.l2_diag->g_gfrob_l2;
  }
  if (fields.mpi_rank)
  {
    out.mpi_rank = &fields.mpi_rank->rank;
  }
  if (fields.zhang_shu)
  {
    out.zhang_shu_active = &fields.zhang_shu->active;
    out.zhang_shu_theta_rho = &fields.zhang_shu->theta_rho;
    out.zhang_shu_theta_p = &fields.zhang_shu->theta_p;
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

inline mfem::io::EulerIGRVisFields
MakeEulerIGRVisFields(const EulerFieldList& fields)
{
  mfem::io::EulerIGRVisFields out;
  if (fields.state)
  {
    out.density = &fields.state->density;
    out.energy = &fields.state->energy;
    out.momentum_x = &fields.state->momentum_x;
  }
  out.sigma = fields.sigma;
  if (fields.pressure)
  {
    out.pressure = &fields.pressure->pressure;
  }
  return out;
}

inline mfem::io::EulerIGRVisWindows
MakeEulerIGRVisWindows(EulerVisWindows& windows)
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
                                    EulerOutputFieldTiming* timing = nullptr)
{
  const auto total_start = std::chrono::steady_clock::now();
  EulerOutputFieldTiming local_timing;
  if (fields.pressure)
  {
    const auto pressure_start = std::chrono::steady_clock::now();
    fields.pressure->Update();
    local_timing.pressure_s = VisSecondsSince(pressure_start);
  }
  if (fields.velocity)
  {
    const auto velocity_start = std::chrono::steady_clock::now();
    fields.velocity->Update();
    local_timing.velocity_s = VisSecondsSince(velocity_start);
  }
  if (fields.h1_diag)
  {
    const auto h1_start = std::chrono::steady_clock::now();
    fields.h1_diag->Update();
    local_timing.h1_diag_s = VisSecondsSince(h1_start);
  }
  if (fields.l2_diag)
  {
    const auto l2_start = std::chrono::steady_clock::now();
    fields.l2_diag->Update();
    local_timing.l2_diag_s = VisSecondsSince(l2_start);
  }
  local_timing.total_s = VisSecondsSince(total_start);
  if (timing)
  {
    *timing = local_timing;
  }
}

inline void UpdateEulerGLVis(EulerVisWindows& windows,
                             const EulerFieldList& fields,
                             int num_procs,
                             int my_rank,
                             const mfem::ParMesh& pmesh,
                             double t)
{
  if (fields.sigma)
  {
    MFEM_VERIFY(fields.pressure != nullptr,
                "Euler GLVis update with IGR expects a pressure view.");
    fields.pressure->Update();
    const mfem::io::EulerIGRVisFields vis_fields =
      MakeEulerIGRVisFields(fields);
    const mfem::io::EulerIGRVisWindows vis_windows =
      MakeEulerIGRVisWindows(windows);
    mfem::io::UpdateEulerIGRViews(vis_windows,
                                  vis_fields,
                                  num_procs,
                                  my_rank,
                                  pmesh,
                                  t);
    return;
  }

  if (fields.pressure)
  {
    fields.pressure->Update();
  }
  if (!fields.state || !fields.pressure)
  {
    return;
  }
  vis::Update(windows.density,
              num_procs,
              my_rank,
              pmesh,
              fields.state->density,
              t,
              "density");
  vis::Update(windows.energy,
              num_procs,
              my_rank,
              pmesh,
              fields.state->energy,
              t,
              "energy");
  vis::Update(windows.pressure,
              num_procs,
              my_rank,
              pmesh,
              fields.pressure->pressure,
              t,
              "pressure");
  vis::Update(windows.momentum_x,
              num_procs,
              my_rank,
              pmesh,
              fields.state->momentum_x,
              t,
              "momentum_x");
}

} // namespace dg_sim

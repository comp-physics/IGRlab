#include "sim/config/case_config.hpp"
#include "sim/context/example_context.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/time/time_defaults.hpp"

#include "test_utils.hpp"

namespace dg_igr_tests
{

int RunTimeDefaultsTests(TestContext& ctx)
{
  constexpr double gamma = 1.4;

  // --- Callback config wiring ---
  {
    dg_sim::DefaultCallbackConfig cfg;
    cfg.enable_alive = true;
    cfg.enable_analysis = true;
    cfg.enable_vtk = false;
    cfg.enable_summary = true;
    cfg.compute_minmax = true;
    const dg_sim::TimeLoopCallbacks callbacks =
      dg_sim::MakeDefaultTimeLoopCallbacks(cfg);
    ctx.Check(static_cast<bool>(callbacks.on_alive),
              "default callbacks: on_alive wired");
    ctx.Check(static_cast<bool>(callbacks.on_analysis),
              "default callbacks: on_analysis wired");
    ctx.Check(!callbacks.on_vtk_save,
              "default callbacks: on_vtk_save disabled by config");
    ctx.Check(static_cast<bool>(callbacks.on_summary),
              "default callbacks: on_summary wired");
    ctx.Check(callbacks.wants_minmax,
              "default callbacks: wants_minmax follows analysis+compute flag");
  }

  {
    dg_sim::DefaultCallbackConfig cfg;
    cfg.enable_alive = false;
    cfg.enable_analysis = false;
    cfg.enable_vtk = true;
    cfg.enable_summary = false;
    cfg.compute_minmax = true;
    const dg_sim::TimeLoopCallbacks callbacks =
      dg_sim::MakeDefaultTimeLoopCallbacks(cfg);
    ctx.Check(!callbacks.on_alive,
              "default callbacks: on_alive disabled by config");
    ctx.Check(!callbacks.on_analysis,
              "default callbacks: on_analysis disabled by config");
    ctx.Check(static_cast<bool>(callbacks.on_vtk_save),
              "default callbacks: on_vtk_save wired");
    ctx.Check(!callbacks.on_summary,
              "default callbacks: on_summary disabled by config");
    ctx.Check(!callbacks.wants_minmax,
              "default callbacks: wants_minmax false when analysis disabled");
  }

  // --- Hook dispatch and payload fields ---
  mfem::Mesh serial_mesh =
    mfem::Mesh::MakeCartesian2D(1,
                                1,
                                mfem::Element::QUADRILATERAL,
                                /*gen_edges=*/true,
                                /*sx=*/1.0,
                                /*sy=*/1.0);
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();

  dg_sim::RunConfig run_cfg;
  run_cfg.opts.order = 1;
  run_cfg.opts.glvis = false;
  run_cfg.opts.vtk_save = false;
  run_cfg.opts.dt_min = 5.0e-13;
  run_cfg.dg_backend = dg_sim::DGBackend::DG;

  {
    const dg_sim::TimeLoopConfig loop_cfg = dg_sim::MakeTimeLoopConfig(run_cfg);
    ctx.CheckNear(loop_cfg.dt_min,
                  run_cfg.opts.dt_min,
                  1.0e-16,
                  "time loop config: dt_min propagated from runtime options");
  }

  dg_sim::ExampleContext example_ctx(run_cfg,
                                     std::move(pmesh),
                                     /*sigma_space=*/"H1",
                                     /*need_sigma=*/false,
                                     /*need_h1=*/false);
  dg_sim::EulerStateBundle bundle(example_ctx, gamma, /*rho_floor=*/1.0e-12);

  const mfem::Vector state =
    EulerConsFromPrim2D(/*rho=*/1.2, /*u=*/0.3, /*v=*/-0.1, /*p=*/0.9, gamma);
  FillConstantState(state, bundle.sol);

  int alive_count = 0;
  int save_count = 0;
  int analysis_count = 0;
  dg_sim::TimeLoopInfo last_alive;
  dg_sim::VTKSaveInfo last_save;
  dg_sim::AnalysisInfo last_analysis;

  dg_sim::TimeLoopCallbacks callbacks;
  callbacks.wants_minmax = true;
  callbacks.on_alive = [&](const dg_sim::TimeLoopInfo& info)
  {
    ++alive_count;
    last_alive = info;
  };
  callbacks.on_vtk_save = [&](const dg_sim::VTKSaveInfo& info)
  {
    ++save_count;
    last_save = info;
  };
  callbacks.on_analysis = [&](const dg_sim::AnalysisInfo& info)
  {
    ++analysis_count;
    last_analysis = info;
  };

  dg_sim::TimeLoopIO io;
  bool glvis_enabled = false;
  io.glvis_enabled = &glvis_enabled;

  dg_sim::TimeLoopHooks hooks =
    dg_sim::MakeEulerTimeLoopHooks(example_ctx,
                                   bundle,
                                   mfem::Mpi::WorldSize(),
                                   mfem::Mpi::WorldRank(),
                                   io,
                                   callbacks);

  ctx.Check(static_cast<bool>(hooks.on_visualize),
            "time hooks: on_visualize is wired");
  ctx.Check(static_cast<bool>(hooks.on_save), "time hooks: on_save is wired");
  ctx.Check(static_cast<bool>(hooks.on_analysis),
            "time hooks: on_analysis is wired");

  hooks.on_visualize(/*step=*/5, /*t=*/0.2, /*dt=*/0.01, /*done=*/false);
  hooks.on_save(/*step=*/5, /*t=*/0.2, /*save_idx=*/3, /*is_final=*/false);
  hooks.on_analysis(/*step=*/5, /*t=*/0.2, /*dt=*/0.01, /*done=*/true);

  ctx.Check(alive_count == 1, "time hooks: on_alive dispatched once");
  ctx.Check(save_count == 1, "time hooks: on_vtk_save dispatched once");
  ctx.Check(analysis_count == 1, "time hooks: on_analysis dispatched once");

  ctx.Check(last_alive.step == 5 && !last_alive.is_final,
            "time hooks: alive payload carries step/final flag");
  ctx.Check(last_save.cycle == 3 && !last_save.is_initial &&
              !last_save.is_final,
            "time hooks: save payload carries cycle/flags");
  ctx.Check(last_analysis.base.step == 5 && last_analysis.base.is_final,
            "time hooks: analysis payload carries step/final flag");
  ctx.Check(last_analysis.has_rho && last_analysis.has_p,
            "time hooks: analysis min/max payload enabled");
  ctx.CheckNear(last_analysis.rho_min,
                1.2,
                1.0e-10,
                "time hooks: rho_min for constant state");
  ctx.CheckNear(last_analysis.rho_max,
                1.2,
                1.0e-10,
                "time hooks: rho_max for constant state");
  ctx.CheckNear(last_analysis.p_min,
                0.9,
                1.0e-10,
                "time hooks: p_min for constant state");
  ctx.CheckNear(last_analysis.p_max,
                0.9,
                1.0e-10,
                "time hooks: p_max for constant state");

  return ctx.failures;
}

} // namespace dg_igr_tests

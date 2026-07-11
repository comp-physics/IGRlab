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

  // --- IGR derived runtime defaults ---
  {
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.device = "cuda";
    opts.dg_backend = "dg";
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    dg_sim::ApplyIGRDerivedRuntimeDefaults(opts,
                                           igr_opts,
                                           /*exact_face_operator_overridden=*/false);
    ctx.Check(igr_opts.exact_dg_face_operator,
              "IGR defaults: CUDA L2/BrokenL2 DG enables exact face");
    ctx.Check(igr_opts.linear_solver == "minres",
              "IGR defaults: CUDA L2/BrokenL2 DG selects MINRES");
    ctx.Check(igr_opts.preconditioner == "chebyshev",
              "IGR defaults: CUDA L2/BrokenL2 DG selects Chebyshev");
    ctx.CheckNear(igr_opts.stage_rtol_scale,
                  1.0,
                  0.0,
                  "IGR defaults: CUDA L2/BrokenL2 DG pins stage rtol scale");
    ctx.CheckNear(igr_opts.lin_rtol,
                  1.0e-5,
                  0.0,
                  "IGR defaults: CUDA L2/BrokenL2 DG uses production rtol");
    ctx.CheckNear(igr_opts.lin_atol,
                  0.0,
                  0.0,
                  "IGR defaults: CUDA L2/BrokenL2 DG uses zero atol");
  }

  {
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.device = "cuda";
    opts.dg_backend = "dg";
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    dg_sim::ApplyIGRDerivedRuntimeDefaults(opts,
                                           igr_opts,
                                           /*exact_face_operator_overridden=*/true);
    ctx.Check(!igr_opts.exact_dg_face_operator,
              "IGR defaults: explicit exact-face override is preserved");
  }

  {
    dg_sim::CommonOptions opts;
    dg_sim::IGROptions igr_opts;
    opts.device = "cpu";
    opts.dg_backend = "dg";
    igr_opts.space = "L2";
    igr_opts.grad = "BrokenL2";
    dg_sim::ApplyIGRDerivedRuntimeDefaults(opts,
                                           igr_opts,
                                           /*exact_face_operator_overridden=*/false);
    ctx.Check(!igr_opts.exact_dg_face_operator,
              "IGR defaults: CPU L2/BrokenL2 keeps exact face opt-in");
  }

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

  struct TimeLoopDummyOperator
  {
    void Mult(const mfem::Vector&, mfem::Vector& y) const { y = 0.0; }
    mfem::real_t GetMaxCharSpeed() const { return 1.0; }
  };
  struct TimeLoopCountingStepper
  {
    int calls = 0;
    void Step(mfem::Vector&, mfem::real_t& t, mfem::real_t& dt)
    {
      ++calls;
      t += dt;
    }
  };
  struct TimeLoopVariableSpeedOperator
  {
    mutable int speed_calls = 0;
    void Mult(const mfem::Vector&, mfem::Vector& y) const { y = 0.0; }
    mfem::real_t GetMaxCharSpeed() const
    {
      ++speed_calls;
      return (speed_calls == 1) ? 1.0 : 200.0;
    }
  };

  // --- dt_min guard uses nominal dt (before stepping) ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.2;
    cfg.dt = 0.04;
    cfg.dt_min = 0.05;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(result.dt_min_reached,
              "time loop: dt_min guard uses nominal dt before stepping");
    ctx.Check(stepper.calls == 0,
              "time loop: dt_min guard aborts before first step");
    ctx.Check(result.steps == 0,
              "time loop: dt_min abort leaves completed step count at zero");
  }

  // --- clipped/snap dt_real below dt_min is allowed (bookkeeping policy) ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.41;
    cfg.dt = 0.2;
    cfg.dt_min = 0.05;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(!result.dt_min_reached,
              "time loop: clipped final dt_real does not trigger dt_min guard");
    ctx.Check(stepper.calls == 3,
              "time loop: clipped final dt_real still advances final step");
    ctx.Check(result.steps == 3,
              "time loop: step counter includes clipped final step");
    ctx.CheckNear(result.final_time,
                  0.41,
                  1.0e-12,
                  "time loop: clipped final dt_real reaches target final time");
  }

  // --- dt_min after CFL update records current step and emits final save ---
  {
    TimeLoopVariableSpeedOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 1.0;
    cfg.dt = 0.0;
    cfg.dt_min = 0.05;
    cfg.cfl = 0.3;
    cfg.order = 1;
    cfg.vtk_save = true;
    cfg.dt_save = 100.0; // avoid save-boundary snapping in this test

    int final_save_calls = 0;
    int final_save_step = -1;
    mfem::real_t final_save_time = -1.0;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int step, mfem::real_t t, int /*save_idx*/, bool is_final)
    {
      if (is_final)
      {
        ++final_save_calls;
        final_save_step = step;
        final_save_time = t;
      }
    };

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(result.dt_min_reached,
              "time loop: dt_min set when CFL update drops nominal dt below threshold");
    ctx.Check(stepper.calls == 1,
              "time loop: dt_min after CFL update occurs after one completed step");
    ctx.Check(result.steps == 1,
              "time loop: dt_min after CFL update preserves completed step count");
    ctx.CheckNear(result.final_time,
                  0.1,
                  1.0e-12,
                  "time loop: dt_min after CFL update preserves advanced time");
    ctx.Check(final_save_calls == 1,
              "time loop: dt_min after CFL update emits one final save callback");
    ctx.Check(final_save_step == result.steps,
              "time loop: final save step matches reported completed steps");
    ctx.CheckNear(final_save_time,
                  result.final_time,
                  1.0e-12,
                  "time loop: final save time matches reported final time");
  }

  // --- zero-step guard: do not call Step when t_final is already reached ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.0;
    cfg.dt = 0.1;
    cfg.dt_min = 0.0;
    cfg.cfl = 0.0;
    cfg.vtk_save = false;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(stepper.calls == 0,
              "time loop: zero dt_real does not invoke stepper");
    ctx.Check(result.steps == 0,
              "time loop: zero dt_real exits before any step is counted");
    ctx.Check(!result.dt_min_reached,
              "time loop: zero-step exit is distinct from dt_min guard");
    ctx.CheckNear(result.final_time,
                  0.0,
                  1.0e-14,
                  "time loop: zero-step exit keeps initial final_time");
  }

  // --- save-boundary snapping advances even when on_save hook is missing ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.5;
    cfg.dt = 0.2;
    cfg.dt_min = 0.0;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.1;

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg);
    ctx.Check(stepper.calls == 5,
              "time loop: save snapping without hook still advances all steps");
    ctx.Check(result.steps == 5,
              "time loop: step counter reaches final time without save hook");
    ctx.CheckNear(result.final_time,
                  0.5,
                  1.0e-12,
                  "time loop: save snapping without hook reaches t_final");
  }

  // --- final time on save boundary emits one final save, not a duplicate ---
  {
    TimeLoopDummyOperator op;
    TimeLoopCountingStepper stepper;
    dg_sim::TimeLoopConfig cfg;
    cfg.t_final = 0.2;
    cfg.dt = 0.1;
    cfg.dt_min = 1.0e-12;
    cfg.cfl = 0.0;
    cfg.vtk_save = true;
    cfg.dt_save = 0.1;

    int save_count = 0;
    int final_save_count = 0;
    int last_save_step = -1;
    int last_save_idx = -1;
    mfem::real_t last_save_time = -1.0;
    bool last_save_final = false;
    dg_sim::TimeLoopHooks hooks;
    hooks.on_save = [&](int step,
                        mfem::real_t t,
                        int save_idx,
                        bool is_final)
    {
      ++save_count;
      if (is_final)
      {
        ++final_save_count;
      }
      last_save_step = step;
      last_save_idx = save_idx;
      last_save_time = t;
      last_save_final = is_final;
    };

    const dg_sim::TimeLoopResult result =
      dg_sim::RunTimeLoop(op, example_ctx.Mesh(), bundle.sol, stepper, cfg, hooks);
    ctx.Check(result.steps == 2,
              "time loop: final save-boundary case completes two steps");
    ctx.Check(save_count == 2,
              "time loop: final save boundary avoids duplicate final save");
    ctx.Check(final_save_count == 1,
              "time loop: final save boundary marks exactly one save final");
    ctx.Check(last_save_step == result.steps && last_save_idx == 1 &&
                last_save_final,
              "time loop: final boundary save carries final step/index flag");
    ctx.CheckNear(last_save_time,
                  result.final_time,
                  1.0e-12,
                  "time loop: final boundary save time matches final time");
  }

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

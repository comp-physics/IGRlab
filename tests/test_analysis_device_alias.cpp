#include "mfem.hpp"

#include "sim/context/example_context.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/time/time_defaults.hpp"

#include <cmath>
#include <iostream>
#include <memory>

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const char* device_config = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(device_config);

  dg_sim::RunConfig cfg;
  cfg.opts.order = 2;
  cfg.dg_backend = dg_sim::DGBackend::DG;

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  dg_sim::ExampleContext ctx(cfg,
                             std::move(pmesh),
                             "L2",
                             /*need_sigma=*/false,
                             /*need_h1=*/false);
  dg_sim::EulerStateBundle bundle(ctx, /*gamma=*/1.4, /*rho_floor=*/1.0e-12);

  bundle.sol = 0.0;
  const int ndofs = ctx.spaces.fes.GetNDofs();
  double* host = bundle.sol.HostReadWrite();
  for (int i = 0; i < ndofs; ++i)
  {
    host[i] = 1.0;
    host[i + 3 * ndofs] = 2.5;
  }

  bundle.sol.UseDevice(true);
  double* state = bundle.sol.ReadWrite();
  mfem::forall(ndofs,
               [=] MFEM_HOST_DEVICE(int i)
               { state[i] = 2.0; });

  dg_sim::AnalysisInfo analysis;
  int calls = 0;
  dg_sim::TimeLoopCallbacks callbacks;
  callbacks.wants_minmax = true;
  callbacks.on_analysis = [&](const dg_sim::AnalysisInfo& info)
  {
    analysis = info;
    ++calls;
  };

  dg_sim::TimeLoopIO io;
  const dg_sim::TimeLoopHooks hooks = dg_sim::MakeEulerTimeLoopHooks(
    ctx,
    bundle,
    mfem::Mpi::WorldSize(),
    mfem::Mpi::WorldRank(),
    io,
    callbacks);
  hooks.on_analysis(/*step=*/1, /*t=*/0.1, /*dt=*/0.01, /*done=*/false);

  const bool pass = calls == 1 && analysis.has_rho &&
                    std::abs(analysis.rho_min - 2.0) <= 1.0e-12 &&
                    std::abs(analysis.rho_max - 2.0) <= 1.0e-12;
  if (!pass)
  {
    std::cerr << "[FAIL] analysis read stale Euler density view"
              << " calls=" << calls << " has_rho=" << analysis.has_rho
              << " rho_min=" << analysis.rho_min
              << " rho_max=" << analysis.rho_max << "\n";
    return 1;
  }

  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] analysis synchronized Euler density after device update\n";
  }
  return 0;
}

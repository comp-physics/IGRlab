#include "igrlab/compat.hpp"
#include "mfem.hpp"

#include "sim/context/example_context.hpp"
#include "sim/state/euler_state_bundle.hpp"
#include "sim/time/time_defaults.hpp"
#include "../src/physics/igr/IGR_device_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const char* device_config = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(device_config);

  dg_sim::RunConfig cfg;
  cfg.opts.order = 2;
  cfg.dg_backend = dg_sim::DGBackend::DG;

  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    4, 2, mfem::Element::QUADRILATERAL, true, 1.0, 0.6);
  for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
  {
    double* coordinate = mesh.GetVertex(vertex);
    const double x = coordinate[0];
    const double y = coordinate[1] / 0.6;
    coordinate[0] = 0.18 * x + 0.82 * x * x;
    coordinate[1] = 0.6 * (0.18 * y + 0.82 * y * y);
  }
  auto pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, mesh);
  mfem::ParMesh* pmesh_view = pmesh.get();
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

  // A first output read leaves the component views host-valid. Later RK
  // updates make only the owning state device-current; a view's HostRead alone
  // cannot discover that change. Exercise the actual common output refresh
  // twice, without an intervening read of the owning host state.
  const double* const original_host_allocation = bundle.sol.GetData();
  (void)bundle.state.density.HostRead();
  (void)bundle.state.momentum.HostRead();
  (void)bundle.state.momentum_x.HostRead();
  (void)bundle.state.momentum_y.HostRead();
  (void)bundle.state.energy.HostRead();
  dg_sim::EulerFieldList conserved_output;
  conserved_output.state = &bundle.state;
  for (int update = 1; update <= 2; ++update)
  {
    double* current = bundle.sol.ReadWrite();
    mfem::forall(ndofs, [=] MFEM_HOST_DEVICE(int i)
    {
      current[i] = 2.0 + update + i / 1024.0;
      current[ndofs + i] = update * 0.25 + i / 2048.0;
      current[2 * ndofs + i] = -update * 0.125 - i / 4096.0;
      current[3 * ndofs + i] = 8.0 + update + i / 512.0;
    });
    if (mfem::Device::Allows(mfem::Backend::CUDA) &&
        bundle.sol.GetMemory().HostIsValid())
    {
      std::cerr << "[FAIL] output regression did not leave a device-current parent\n";
      return 3;
    }
    // Exclude derived projections here: a CPU fallback must not accidentally
    // synchronize the parent and hide a missing conserved-output refresh.
    dg_sim::UpdateEulerOutputFields(conserved_output);
    const double* density = bundle.state.density.HostRead();
    const double* momentum = bundle.state.momentum.HostRead();
    const double* momentum_x = bundle.state.momentum_x.HostRead();
    const double* momentum_y = bundle.state.momentum_y.HostRead();
    const double* energy = bundle.state.energy.HostRead();
    bool current_output = bundle.sol.GetData() == original_host_allocation;
    for (int i = 0; i < ndofs; ++i)
    {
      current_output = current_output &&
        density[i] == 2.0 + update + i / 1024.0 &&
        momentum[i] == update * 0.25 + i / 2048.0 &&
        momentum[ndofs + i] == -update * 0.125 - i / 4096.0 &&
        momentum_x[i] == momentum[i] && momentum_y[i] == momentum[ndofs + i] &&
        energy[i] == 8.0 + update + i / 512.0;
    }
    if (!current_output)
    {
      std::cerr << "[FAIL] output read stale conserved views after device update "
                << update << "\n";
      return 3;
    }
  }

  // Device-local IGR coefficients must consume alpha_e=C_alpha*h_e^2, not a
  // single alpha=C_alpha*h_max^2.  This source is compiled as CUDA and the
  // active backend is selected above, so these kernels execute on the device.
  constexpr int igr_dim = 2;
  constexpr int igr_num_eq = igr_dim + 2;
  constexpr int igr_nq = 3;
  constexpr double c_alpha = 0.41;
  constexpr double rho_floor = 0.2;
  const int igr_ne = pmesh_view->GetNE();
  std::vector<double> expected_alpha(static_cast<std::size_t>(igr_ne));
  double h2_min = std::numeric_limits<double>::infinity();
  double h2_max = 0.0;
  mfem::Vector element_alpha(igr_ne);
  for (int element = 0; element < igr_ne; ++element)
  {
    const double h = pmesh_view->GetElementSize(element, 1);
    const double h2 = h * h;
    h2_min = std::min(h2_min, h2);
    h2_max = std::max(h2_max, h2);
    expected_alpha[static_cast<std::size_t>(element)] = c_alpha * h2;
    element_alpha(element) = c_alpha * h2;
  }

  mfem::Vector igr_state(igr_num_eq * igr_nq * igr_ne);
  mfem::Vector igr_gradient(igr_num_eq * igr_dim * igr_nq * igr_ne);
  mfem::Vector alpha_over_rho(igr_nq * igr_ne);
  mfem::Vector igr_source(igr_nq * igr_ne);
  igr_state = 0.0;
  igr_gradient = 0.0;
  double* igr_state_host = igr_state.HostReadWrite();
  double* igr_gradient_host = igr_gradient.HostReadWrite();
  for (int element = 0; element < igr_ne; ++element)
  {
    for (int q = 0; q < igr_nq; ++q)
    {
      const int q_index = q + igr_nq * element;
      igr_state_host[igr_num_eq * q_index] =
        0.73 + 0.04 * static_cast<double>(element) +
        0.01 * static_cast<double>(q);
      igr_gradient_host[1 +
                        igr_num_eq * (0 + igr_dim * q_index)] = 0.8;
      igr_gradient_host[2 +
                        igr_num_eq * (1 + igr_dim * q_index)] = -0.35;
    }
  }
  element_alpha.UseDevice(true);
  igr_state.UseDevice(true);
  igr_gradient.UseDevice(true);
  alpha_over_rho.UseDevice(true);
  igr_source.UseDevice(true);

  ::igrlab::igr_device::IGRDeviceLocalAlphaOverRhoKernel coefficient_kernel;
  coefficient_kernel.nq = igr_nq;
  coefficient_kernel.ne = igr_ne;
  coefficient_kernel.num_eq = igr_num_eq;
  coefficient_kernel.rho_floor = rho_floor;
  coefficient_kernel.element_alpha = element_alpha.Read();
  coefficient_kernel.u = igr_state.Read();
  coefficient_kernel.ck = alpha_over_rho.Write();
  mfem::forall(igr_ne * igr_nq, coefficient_kernel);

  ::igrlab::igr_device::IGRDeviceLocalSourceKernel source_kernel;
  source_kernel.nq = igr_nq;
  source_kernel.ne = igr_ne;
  source_kernel.dim = igr_dim;
  source_kernel.num_eq = igr_num_eq;
  source_kernel.rho_floor = rho_floor;
  source_kernel.element_alpha = element_alpha.Read();
  source_kernel.u = igr_state.Read();
  source_kernel.grad = igr_gradient.Read();
  source_kernel.src = igr_source.Write();
  mfem::forall(igr_ne * igr_nq, source_kernel);

  const double* coefficient_values = alpha_over_rho.HostRead();
  const double* source_values = igr_source.HostRead();
  const double* state_values = igr_state.HostRead();
  const double* gradient_values = igr_gradient.HostRead();
  const double max_physical_alpha = c_alpha * h2_max;
  double max_coefficient_error = 0.0;
  double max_source_error = 0.0;
  double max_local_vs_max = 0.0;
  for (int element = 0; element < igr_ne; ++element)
  {
    const double alpha_e = expected_alpha[static_cast<std::size_t>(element)];
    for (int q = 0; q < igr_nq; ++q)
    {
      const int q_index = q + igr_nq * element;
      const int state_base = igr_num_eq * q_index;
      const int gradient_base = igr_num_eq * igr_dim * q_index;
      const double expected_coefficient =
        ::igrlab::frozen_scalar_igr_algebra::AlphaOverDensity(
          alpha_e, state_values[state_base], rho_floor);
      const double max_policy_coefficient =
        ::igrlab::frozen_scalar_igr_algebra::AlphaOverDensity(
          max_physical_alpha, state_values[state_base], rho_floor);
      const double expected_source =
        ::igrlab::frozen_scalar_igr_algebra::
          SourceFromConservativeStateGradient(
            igr_dim,
            state_values + state_base,
            1,
            gradient_values + gradient_base,
            1,
            igr_num_eq,
            alpha_e,
            rho_floor);
      const double max_policy_source =
        ::igrlab::frozen_scalar_igr_algebra::
          SourceFromConservativeStateGradient(
            igr_dim,
            state_values + state_base,
            1,
            gradient_values + gradient_base,
            1,
            igr_num_eq,
            max_physical_alpha,
            rho_floor);
      max_coefficient_error =
        std::max(max_coefficient_error,
                 std::abs(coefficient_values[q_index] -
                          expected_coefficient));
      max_source_error =
        std::max(max_source_error,
                 std::abs(source_values[q_index] - expected_source));
      max_local_vs_max =
        std::max({max_local_vs_max,
                  std::abs(coefficient_values[q_index] -
                           max_policy_coefficient),
                  std::abs(source_values[q_index] - max_policy_source)});
    }
  }
  if (!(h2_max > 2.0 * h2_min && max_coefficient_error < 5.0e-13 &&
        max_source_error < 5.0e-13 && max_local_vs_max > 1.0e-3))
  {
    std::cerr << "[FAIL] device local-alpha IGR coefficients"
              << " h2_ratio=" << h2_max / h2_min
              << " coefficient_error=" << max_coefficient_error
              << " source_error=" << max_source_error
              << " local_vs_max=" << max_local_vs_max << "\n";
    return 2;
  }

  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] analysis synchronized Euler density after device update\n";
    std::cout << "[PASS] output synchronized all conserved views after two device updates\n";
    std::cout << "[PASS] device IGR coefficients use per-element "
                 "C_alpha*h_e^2 on a graded mesh\n";
  }
  return 0;
}

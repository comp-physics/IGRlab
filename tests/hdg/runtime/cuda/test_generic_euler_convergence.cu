#include "generic_euler_convergence_fixture.hpp"

#include "numerics/hdg/runtime/cpu/generic_euler.hpp"
#include "numerics/hdg/runtime/cuda/generic_euler.cuh"

#include <mfem.hpp>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>

namespace
{

using dg::hdg::generic::BackendKind;
using dg::hdg::generic::CudaEngineKind;
using dg::hdg::generic::EulerHdgRequest;
using dg::hdg::generic::HdgEulerSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerCpuSessionStatus;
using dg::hdg::runtime::cpu::GenericEulerHdgCpuSession;
using dg::hdg::runtime::cuda::GenericEulerCudaPreparedStep;
using dg::hdg::runtime::cuda::GenericEulerHdgCudaSession;
using dg::hdg::test::generic_euler_convergence::CollectiveAll;
using dg::hdg::test::generic_euler_convergence::ExecutionSelection;

template <typename Coordinates, typename Diagnostics>
bool ValidCommittedMetadata(const Coordinates coordinates,
                            const Diagnostics diagnostics,
                            const int steps, const double dt) noexcept
{
  const double expected_time = static_cast<double>(steps) * dt;
  const double time_scale = std::max(1.0, std::abs(expected_time));
  return coordinates.step == static_cast<std::uint64_t>(steps) &&
      coordinates.generation == static_cast<std::uint64_t>(steps) &&
      std::abs(coordinates.time - expected_time) <=
          64.0 * std::numeric_limits<double>::epsilon() * time_scale &&
      diagnostics.available &&
      diagnostics.wave_generation == coordinates.generation &&
      diagnostics.coordinates.step == coordinates.step &&
      std::isfinite(diagnostics.maximum_wave_speed) &&
      diagnostics.maximum_wave_speed > 0.0;
}

bool AdvanceCpu(const EulerHdgRequest& request, const int steps,
                const double dt, mfem::ParGridFunction& committed)
{
  GenericEulerHdgCpuSession session;
  if (session.Initialize(request).status !=
      GenericEulerCpuSessionStatus::initialized)
  {
    return false;
  }
  for (int step = 0; step < steps; ++step)
  {
    if (session.Advance(dt).status != GenericEulerCpuSessionStatus::advanced)
      return false;
  }
  const bool local = session.ExportCommittedInterior(committed) &&
      ValidCommittedMetadata(
          session.coordinates(), session.diagnostics(), steps, dt);
  return CollectiveAll(local, request.communicator);
}

bool AdvanceCuda(const EulerHdgRequest& request, const int steps,
                 const double dt, mfem::ParGridFunction& committed)
{
  GenericEulerHdgCudaSession session;
  if (session.Initialize(request).status !=
      HdgEulerSessionStatus::initialized)
  {
    return false;
  }
  for (int step = 0; step < steps; ++step)
  {
    GenericEulerCudaPreparedStep authority;
    GenericEulerHdgCudaSession::prepared_output_type output{};
    if (session.Prepare(dt, authority, output).status !=
            HdgEulerSessionStatus::prepared ||
        !session.AcknowledgePreparedOutput(authority, output) ||
        session.Commit(std::move(authority), output).status !=
            HdgEulerSessionStatus::committed)
    {
      return false;
    }
  }
  const int size = committed.Size();
  constexpr int components =
      dg::hdg::generic::euler_2d_conserved_components;
  const bool extent_valid = size > 0 && size % components == 0;
  const bool local = extent_valid && session.ExportCommittedInterior({
      committed.HostWrite(),
      static_cast<std::size_t>(size / components), components}) &&
      ValidCommittedMetadata(
          session.coordinates(), session.diagnostics(), steps, dt);
  return CollectiveAll(local, request.communicator);
}

class ThreeEngineRunner
{
public:
  bool Begin(const ExecutionSelection execution)
  {
    if (execution.backend == BackendKind::cpu)
    {
      if (phase_ != 0 || device_) return false;
      phase_ = 1;
      return true;
    }
    if (execution.backend != BackendKind::cuda) return false;
    if (execution.cuda_engine == CudaEngineKind::ordinary)
    {
      if (phase_ != 1 || device_) return false;
      if (HYPRE_ClearAllErrors() != 0 || HYPRE_GetError() != 0)
        return false;
      device_ = std::make_unique<mfem::Device>("cuda");
      phase_ = 2;
      return true;
    }
    if (execution.cuda_engine == CudaEngineKind::tensor_core)
    {
      if (phase_ != 2 || !device_) return false;
      if (HYPRE_ClearAllErrors() != 0 || HYPRE_GetError() != 0)
        return false;
      phase_ = 3;
      return true;
    }
    return false;
  }

  bool Advance(const EulerHdgRequest& request, const int steps,
               const double dt, mfem::ParGridFunction& committed) const
  {
    if (request.backend.kind == BackendKind::cpu)
      return phase_ == 1 && AdvanceCpu(request, steps, dt, committed);
    return request.backend.kind == BackendKind::cuda && phase_ >= 2 &&
        AdvanceCuda(request, steps, dt, committed);
  }

private:
  std::unique_ptr<mfem::Device> device_;
  int phase_ = 0;
};

} // namespace

int main(int argc, char* argv[])
{
  mfem::MPI_Session mpi(argc, argv);
  int ranks = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  dg::hdg::test::generic_euler_convergence::CaseSelection test_case{};
  const bool parsed =
      dg::hdg::test::generic_euler_convergence::ParseCaseSelection(
          argc, argv, test_case);
  if (!parsed || ranks != 1)
  {
    if (ranks == 1)
      std::cerr << "Usage: " << argv[0]
                << " --order 1|2|3 --regularization conventional|igr"
                   " [--igr-volume-quadrature-order Q"
                   " --igr-face-quadrature-order Q]"
                   " [--newton-linear-rtol R]"
                   " [--final-linear-rtol R]"
                   " [--true-residual-tightening-linear-rtol R]\n";
    else
      std::cerr << "Three-engine convergence qualification is rank-one; "
                   "rank-two CUDA correctness has separate coverage\n";
    return 2;
  }

  const std::array<ExecutionSelection, 3> executions{{
      {BackendKind::cpu, CudaEngineKind::ordinary, "CPU"},
      {BackendKind::cuda, CudaEngineKind::ordinary, "ordinary CUDA"},
      {BackendKind::cuda, CudaEngineKind::tensor_core, "tensor CUDA"}}};
  ThreeEngineRunner runner;
  const bool passed =
      dg::hdg::test::generic_euler_convergence::RunComparisonCase(
          MPI_COMM_WORLD, test_case, executions, runner);
  return passed ? 0 : 1;
}

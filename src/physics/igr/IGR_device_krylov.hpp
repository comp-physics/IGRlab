#pragma once

#include "igrlab/environment.hpp"
#include "mfem.hpp"
#include <memory>
#include <string>

namespace igrlab
{
namespace igr_internal
{
// Rank-one, ordinary CUDA prototype. Recurrences, ordered MGS, reductions and
// cached work vectors stay on device. The ordinary path still reads host
// convergence per iteration; the opt-in cooperative coarse path reads once
// per solve. Outer FGMRES always certifies its final FP64 true residual.
// Keep this private to IGR: no change to MFEM's global arithmetic precision.
inline bool DeviceKrylovRequested(const char* variable,
                                  bool default_device = false)
{
  const char* raw = ::igrlab::GetEnv(variable);
  const std::string value = raw && raw[0] ? raw :
    (default_device ? "cuda-fp64" : "mfem");
  MFEM_VERIFY(value == "mfem" || value == "cuda-fp64",
              variable << " must be mfem|cuda-fp64.");
  return value == "cuda-fp64";
}

// Cooperative execution is independently opt-in; its hardware admission is
// separate from the ordinary device-scalar backend selector.
inline bool CooperativeCoarseCGRequested()
{
  const char* raw = ::igrlab::GetEnv("DG_IGR_IGR_COARSE_CG_EXECUTION");
  const std::string value = raw && raw[0] ? raw : "host-polled";
  MFEM_VERIFY(value == "host-polled" || value == "cooperative",
              "IGRLAB_IGR_COARSE_CG_EXECUTION must be host-polled|cooperative.");
  return value == "cooperative";
}

// Returns null if rank/backend/hardware cannot admit a resident cooperative
// grid. Caller retains the existing device CG in that case. This specialized
// solver requires a finalized CSR operator, fixed OperatorJacobiSmoother,
// zero initialization and an iteration budget no larger than 100.
std::unique_ptr<mfem::IterativeSolver> MakeCooperativeCGSolver(MPI_Comm comm);

// Changes recurrence scaling only; both modes retain the FP64 RHS-relative true gate.
enum class KrylovResidualScale { Initial, RHS };

std::unique_ptr<mfem::IterativeSolver> MakeDeviceFGMRESSolver(
  MPI_Comm comm, int restart_dimension = 50,
  KrylovResidualScale residual_scale = KrylovResidualScale::Initial);
std::unique_ptr<mfem::IterativeSolver> MakeDeviceCGSolver(MPI_Comm comm);
}
}

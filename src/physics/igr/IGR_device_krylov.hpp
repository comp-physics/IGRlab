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
// cached work vectors stay on device; one convergence read still occurs per
// iteration. The outer FGMRES always certifies its final FP64 true residual.
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

std::unique_ptr<mfem::IterativeSolver> MakeDeviceFGMRESSolver(
  MPI_Comm comm, int restart_dimension = 50);
std::unique_ptr<mfem::IterativeSolver> MakeDeviceCGSolver(MPI_Comm comm);
}
}

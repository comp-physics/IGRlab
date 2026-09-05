#pragma once
#include "mfem.hpp"
#include <memory>

namespace dg_igr
{
namespace igr_internal
{
// Fixed current-equation Richardson sweeps, with one final combined norm
// reduction. No inner dot products. Rank-one CUDA research scope.
// Configure current operator/preconditioner before each Mult; on a failed
// tolerance gate, Mult restores its starting guess for a charged Krylov fallback.
// GetFinalNorm then describes the rejected trial, not the restored vector.
std::unique_ptr<mfem::IterativeSolver> MakeStationaryCorrectionSolver(
  MPI_Comm comm, int sweeps, double damping = 1.0);
}
}

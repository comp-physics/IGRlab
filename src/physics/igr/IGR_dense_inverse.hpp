#pragma once

#include "mfem.hpp"
#include <memory>
#include <string>

namespace dg_igr
{
namespace igr_internal
{
// Bounded CUDA research preconditioner. The factory samples the current full
// operator once; SetOperator only checks dimensions and never changes its inverse.
// DG_IGR_IGR_DENSE_SETUP=host (default, n<=2048) or cusolver (n<=4096)
// selects pivoted FP64 setup only; the requested application precision is separate.
std::unique_ptr<mfem::Solver> MakeDenseInversePreconditioner(
  const mfem::Operator& op, const std::string& precision, MPI_Comm comm);
}
}

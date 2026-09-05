#pragma once
#include "mfem.hpp"
#include <memory>
#include <string>

namespace dg_igr
{
namespace igr_internal
{
// Reusable additive element-block + conforming-Q1 Galerkin preconditioner.
// Research scope: rank-one CUDA, scalar conforming-mesh L2 quadrilaterals.
// Dense coarse solve is bounded to 512 unknowns; "cg" is an explicit alternative.
// SetOperator checks dimensions only: outer FGMRES must solve the current A.
std::unique_ptr<mfem::Solver> MakeTwoLevelInversePreconditioner(
  const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes,
  const std::string& coarse_mode = "dense");
}
}

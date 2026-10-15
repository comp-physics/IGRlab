#pragma once

#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include <memory>
#include <string>

namespace igrlab
{
namespace igr_internal
{
// Reusable additive element-block + conforming-Qk Galerkin preconditioner.
// Research scope: rank-one CUDA, scalar conforming-mesh L2 quadrilaterals.
// Coarse order k defaults to 1 and must satisfy 1 <= k <= sigma FE order.
// Dense coarse solve is bounded to 512 unknowns; "cg" is an explicit alternative.
// SetOperator checks dimensions only: outer FGMRES must solve the current A.
// IGRLAB_IGR_TWO_LEVEL_BLOCK_PRECISION=fp32 selects IEEE FP32 inverse-block
// storage/application with FP64 row recovery; defaults remain FP64. The backup
// inverse remains allocated. Application use requires the owned CUDA FGMRES
// true-residual gate; coarse precision is selected separately and remains FP64.
std::unique_ptr<mfem::Solver> MakeTwoLevelInversePreconditioner(
  const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes,
  const std::string& coarse_mode = "dense", int coarse_order = 1);
}
}

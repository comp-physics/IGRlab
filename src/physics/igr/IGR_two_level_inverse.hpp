#pragma once

#include "igrlab/compat.hpp"
#include "igrlab/environment.hpp"
#include "mfem.hpp"
#include <memory>
#include <string>

namespace igrlab
{
namespace igr_internal
{
inline std::string TwoLevelLocalMode()
{
  const char* raw = ::igrlab::GetEnv("DG_IGR_IGR_TWO_LEVEL_LOCAL");
  const std::string mode = raw && raw[0] ? raw : "element";
  MFEM_VERIFY(mode == "element" || mode == "shared-patch-fp64" || mode == "shared-patch-tf32",
              "IGRLAB_IGR_TWO_LEVEL_LOCAL must be element|shared-patch-fp64|shared-patch-tf32.");
  return mode;
}

inline std::string TwoLevelPatchComposition()
{
  const char* raw = ::igrlab::GetEnv("DG_IGR_IGR_TWO_LEVEL_PATCH_COMPOSITION");
  const std::string mode = raw && raw[0] ? raw : "additive";
  MFEM_VERIFY(mode == "additive" || mode == "multiplicative",
              "IGRLAB_IGR_TWO_LEVEL_PATCH_COMPOSITION must be additive|multiplicative.");
  return mode;
}

// Reusable element-block + conforming-Qk Galerkin preconditioner.
// Research scope: rank-one CUDA, scalar conforming-mesh L2 quadrilaterals.
// Coarse order k defaults to 1 and must satisfy 1 <= k <= sigma FE order.
// Dense coarse solve is bounded to 512 unknowns; "cg" is an explicit alternative.
// SetOperator checks dimensions only: outer FGMRES must solve the current A.
// IGRLAB_IGR_TWO_LEVEL_BLOCK_PRECISION=fp32 selects IEEE FP32 inverse-block
// storage/application with FP64 row recovery; defaults remain FP64. The backup
// inverse remains allocated. Application use requires the owned CUDA FGMRES
// true-residual gate; coarse precision is selected separately and remains FP64.
// IGRLAB_IGR_TWO_LEVEL_LOCAL=shared-patch-fp64|shared-patch-tf32 replaces
// element blocks with grouped shared-reference restricted patch corrections.
// This explicit research mode admits uniform affine periodic p5 quads only;
// the original current operator and FP64 outer residual remain authoritative.
// PATCH_COMPOSITION defaults to additive. Explicit multiplicative shared patches
// apply zc + Rpatch(r - A0*zc), retaining the certified first-operator FP64
// face stencil A0. This changes only the preconditioner, never the current A.
std::unique_ptr<mfem::Solver> MakeTwoLevelInversePreconditioner(
  const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes,
  const std::string& coarse_mode = "dense", int coarse_order = 1);
}
}

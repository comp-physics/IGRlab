#pragma once

#include "mfem.hpp"
#include <memory>

namespace igrlab { namespace igr_internal {

enum class TensorMultigridPrecision { FP64, FP32, TF32 };
enum class TensorMultigridSmoother { Separable, Center };

// Experimental rank-one, uniform periodic affine-quadrilateral p5 scalar L2
// preconditioner. Owns a verified snapshot of frozen_A; retains no reference to
// it or sfes. Its 5->3->1 Galerkin V-cycle has pre/post overlapping patch
// smoothing. Separable (the default) applies shared transforms across distinct
// real patch lines. Center composes their center rows once in FP64 and maps
// distinct real residual patches directly to their owning element centers.
// Mixed modes store all noncoarse vectors/operators in FP32; the sparse p1
// Galerkin CG/Jacobi solve is explicitly FP64. TF32 affects only the shared
// patch transforms or center map. Center layouts allow 1 or 4 warps per CTA
// and split-K of 1, 2 or 4, with a final reduction in the center add. The outer
// solver must certify the current original A in
// FP64. SetOperator checks dimensions only; rebuilding is the owner's policy.
std::unique_ptr<mfem::Solver> MakeTensorMultigridPreconditioner(
  const mfem::Operator& frozen_A, mfem::ParFiniteElementSpace& sfes,
  TensorMultigridPrecision precision,
  TensorMultigridSmoother smoother=TensorMultigridSmoother::Separable,
  int center_warps=1,int center_split_k=1);

} }

#pragma once

#include "mfem.hpp"
#include <memory>

namespace igrlab { namespace igr_internal {

enum class TensorMultigridPrecision { FP64, FP32, TF32, TF32x3 };
enum class TensorMultigridSmoother { Separable, Center };
enum class TensorMultigridOperator { Sampled, Shared, Spectral };
enum class TensorMultigridResidual { Auto, FP32 };

// Experimental rank-one, uniform axis-aligned affine-quadrilateral p5 scalar
// L2 preconditioners. Sampled and Shared require periodic topology. Sampled
// owns a verified snapshot of frozen_A. Shared owns its
// translation-averaged five-block reference, an explicit approximation to
// variable frozen coefficients with O(1) stencil storage. Neither retains a
// reference to frozen_A or sfes. Its 5->3->1 Galerkin V-cycle has pre/post overlapping patch
// smoothing. Separable (the default) applies shared transforms across distinct
// real patch lines. Center composes their center rows once in FP64 and maps
// distinct real residual patches directly to their owning element centers.
// Mixed modes store all noncoarse vectors/operators in FP32; the sparse p1
// Galerkin CG/Jacobi solve is explicitly FP64. TF32 affects the patch
// transforms/center map and, in Shared mode, reference residual products.
// Center layouts allow 1 or 4 warps per CTA
// and split-K of 1, 2 or 4, with a final reduction in the center add.
// Shared+Center additionally admits owned_tile=4: one 4x4 owned tile receives
// correction from a 6x6 input patch, using a fixed 64x64 output CTA and fused
// scatter. This option requires the legacy center layout defaults (1,1).
// owned_tile=1 preserves the existing element-center smoother.
// Shared TF32 also admits residual_mode=FP32, keeping WMMA smoothing while
// applying reference residuals with FP32 SIMT. Auto preserves prior placement.
// The outer solver must certify the current original A in FP64. SetOperator
// checks dimensions only; rebuilding is the owner's policy.
// Spectral selects a separate global reference inverse with four dense basis
// transforms and no MG/coarse solve. TF32x3 is restricted to Spectral and uses
// HH+HL+LH (omitting LL) in three TF32 products for each transform. Spectral
// requires default smoother/layout/residual controls and a checked separable
// reference on a periodic Cartesian product or a fully open Cartesian rectangle
// with natural scalar boundaries. The open path checks edge/corner corrections;
// all current physical coefficients remain in the outer A.
std::unique_ptr<mfem::Solver> MakeTensorMultigridPreconditioner(
  const mfem::Operator& frozen_A, mfem::ParFiniteElementSpace& sfes,
  TensorMultigridPrecision precision,
  TensorMultigridSmoother smoother=TensorMultigridSmoother::Separable,
  int center_warps=1,int center_split_k=1,
  TensorMultigridOperator operator_mode=TensorMultigridOperator::Sampled,
  int owned_tile=1,
  TensorMultigridResidual residual_mode=TensorMultigridResidual::Auto);

} }

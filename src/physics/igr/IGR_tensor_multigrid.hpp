#pragma once

#include "mfem.hpp"
#include <memory>

namespace igrlab { namespace igr_internal {

enum class TensorMultigridPrecision { FP64, FP32, TF32 };

// Experimental rank-one, uniform periodic affine-quadrilateral p5 scalar L2
// preconditioner. Owns a verified snapshot of frozen_A; retains no reference to
// it or sfes. Its 5->3->1 Galerkin V-cycle has pre/post overlapping separable
// patch smoothing, with shared transforms across distinct real patch lines.
// Mixed modes store all noncoarse vectors/operators in FP32; the sparse p1
// Galerkin CG/Jacobi solve is explicitly FP64. TF32 affects only the shared
// patch transforms. The outer solver must certify the current original A in
// FP64. SetOperator checks dimensions only; rebuilding is the owner's policy.
std::unique_ptr<mfem::Solver> MakeTensorMultigridPreconditioner(
  const mfem::Operator& frozen_A, mfem::ParFiniteElementSpace& sfes,
  TensorMultigridPrecision precision);

} }

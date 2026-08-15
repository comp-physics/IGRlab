#pragma once

/**
 * @file numerics/hdg/runtime/cuda/traced_scalar_igr_p1_private.cuh
 * @brief Private packed CUDA launch seam for affine-Q1 traced scalar IGR.
 *
 * This is deliberately a detail-only borrowing ABI.  Each element occupies a
 * contiguous, component-major slice: [element][component][local dof].  It
 * creates neither a runtime owner nor any solver, trace, allocation, or MPI
 * policy.  The caller owns all device storage and the raw quadrature tables.
 */

#include "numerics/hdg/runtime/common/traced_scalar_igr_p1_local_algebra.hpp"

#include <cuda_runtime_api.h>

namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private
{
namespace algebra = common::traced_scalar_igr_p1_local_algebra;

inline constexpr int vertices_per_element = 8;

/** Borrowed raw inputs, packed by element. */
struct BatchInput final
{
  int elements = 0;
  const double* vertices = nullptr;             // [E][8]
  const algebra::Parameters* parameters = nullptr; // [E]
  const double* interior = nullptr;             // [E][28]
  const double* previous_interior = nullptr;    // [E][28]
  const double* trace = nullptr;                // [E][40]
  const double* previous_trace = nullptr;       // [E][40]
  // [E][volume_points][4], or null when every source count is zero.
  const double* static_euler_rhs = nullptr;
  // [E]; entries equal volume_points with a source table, or zero when the
  // batch source pointer is null.
  const int* static_euler_rhs_points = nullptr;
  const double* volume_xyw = nullptr;           // [volume_points][x,y,weight]
  int volume_points = 0;
  const double* segment_xw = nullptr;           // [segment_points][x,weight]
  int segment_points = 0;
};

/** Borrowed raw outputs, packed by element. */
struct BatchOutput final
{
  algebra::Status* statuses = nullptr;          // [E]
  double* interior_residual = nullptr;          // [E][28]
  double* trace_residual = nullptr;             // [E][40]
  double* A = nullptr;                          // [E][28][28]
  double* B = nullptr;                          // [E][28][40]
  double* C = nullptr;                          // [E][40][28]
  double* D = nullptr;                          // [E][40][40]
  double* f = nullptr;                          // [E][28]
  double* g = nullptr;                          // [E][40]
};

static_assert(sizeof(algebra::Status) == sizeof(unsigned char),
              "packed status storage is one byte per element");

/**
 * Launch one CUDA thread per packed element.  For each in-range thread, the
 * common local algebra is invoked exactly once and owns that element's status
 * and output-preservation semantics.  A malformed batch envelope returns a
 * CUDA launch error; element-local geometry, state, and source-layout errors
 * are reported in BatchOutput::statuses without affecting other elements.
 */
cudaError_t Launch(const BatchInput& input, const BatchOutput& output,
                   cudaStream_t stream = nullptr) noexcept;

} // namespace dg::hdg::runtime::cuda::traced_scalar_igr_p1_private

#pragma once

#include "IGR_element_block_jacobi.hpp"

namespace igrlab
{
namespace igr_internal
{

/** Bounded tensor inverse with current FP64 physical mass scaling.
 *
 * Assemble initial exact domain/face diagonal blocks through the usual
 * PrepareVolumeBlocks/AddExactFaceBlocks/FactorBlocks sequence exactly once.
 * Establish initial PA/face geometry caches before constructing this backend:
 * MFEM can create mesh Nodes lazily, which changes the geometry sequence.
 * IGRLAB_IGR_LOCAL_TENSOR_REFERENCE=exact (default) preserves the original
 * collocated GL, fixed affine Cartesian Q1 admission: D1D <= 8 and an SPD
 * Kronecker sum H0 = S0 D0 S0. Shared exact axis factors then give the current
 * local inverse for globally uniform changes of initial density/coefficient.
 * Heterogeneous current density remains an approximate SPD surrogate.
 *
 * The opt-in projected reference supports p5 GL scalar quads with fixed
 * conforming Q1/Q2 maps. It checks finite positive signed Jacobians at setup
 * samples; this does not certify validity between samples. Its actual positive
 * mass diagonal is m_i = sum_q w_q det(J_q) invrho_q phi_i(q)^2, with two
 * squared-basis contractions on the operator's mass rule (GL7x7 for Q2).
 * For each physical-boundary mask in local edge order, averaged initial H_e
 * selects shared orthogonal axes Q = Qy tensor Qx via partial traces. Each
 * element caches only positive d_e,k = (Q^T H_e Q)_kk and Mult applies
 * S_current Q diag(1/d_e) Q^T S_current. Axis eigensums are not inverted.
 * This projected operator is SPD in exact arithmetic; it adds an initial-block
 * approximation unless H_e is diagonal in Q. Uniform density changes scale
 * this projected operator exactly, rather than making it an exact inverse of
 * a general mapped current block. Heterogeneous density adds the existing
 * current-mass approximation. Dense EA storage is released after setup.
 * Legacy DG_IGR_IGR_LOCAL_TENSOR_REFERENCE and DG_IGR_LOCAL_TENSOR_REFERENCE
 * aliases are accepted. Reference and execution selections are immutable.
 *
 * The default simt-fp64 execution retains four FP64 contractions.
 * IGRLAB_IGR_LOCAL_TENSOR_EXECUTION (legacy DG_IGR_LOCAL_TENSOR_EXECUTION)
 * additionally selects simt-fp32, simt-dense-fp32, wmma-factor-tf32,
 * wmma-factor-fp64, wmma-dense-tf32 or wmma-dense-fp64. Grouped/dense modes
 * require p5; WMMA requires SM80+. Projected reference rejects dense modes,
 * whose one shared inverse cannot represent element-specific diagonals.
 * Exact dense modes retain one 36x36 inverse per verified class. Every mode
 * fuses native gather, both current FP64 mass factors and FP64 scatter,
 * without a global packed residual buffer or per-element dense inverse.
 * FP32/TF32 modes directly round the scaled residual/factors into their
 * arithmetic range; rounding can perturb symmetry/SPD and is not certified
 * by the FP64 reference admission checks.
 * No assembly, LU, triangular solve, iteration, fallback, or finite scan is
 * performed by the current-coefficient update or inverse application.
 */
std::unique_ptr<ElementBlockJacobi> MakeTensorElementBlockJacobi(
  mfem::ParFiniteElementSpace& space,
  const mfem::QuadratureFunction& initial_invrho);

}
}

#pragma once

#include "IGR_element_block_jacobi.hpp"

namespace igrlab
{
namespace igr_internal
{

/** Bounded tensor inverse with current FP64 collocated mass scaling.
 *
 * Assemble initial exact domain/face diagonal blocks through the usual
 * PrepareVolumeBlocks/AddExactFaceBlocks/FactorBlocks sequence exactly once.
 * Establish initial PA/face geometry caches before constructing this backend:
 * MFEM can create mesh Nodes lazily, which changes the geometry sequence.
 * For a collocated GL scalar space on fixed affine Cartesian quads, setup
 * admits only an SPD Kronecker sum after physical-mass normalization. It then
 * discards dense element build storage and retains shared small tensor factors.
 * RefreshMassScaling updates M from the actual invrho QF and fixed quadrature
 * geometry. Mult applies M^(-1/2) H0^(-1) M^(-1/2). This is the exact current
 * local inverse for globally uniform changes of the initial density/coefficient;
 * heterogeneous density defines an explicitly approximate SPD surrogate in
 * exact arithmetic. The default simt-fp64 execution preserves the existing
 * four FP64 contractions. IGRLAB_IGR_LOCAL_TENSOR_EXECUTION (legacy alias
 * DG_IGR_LOCAL_TENSOR_EXECUTION) is read once at construction; opt-in choices
 * are simt-fp32, simt-dense-fp32, wmma-factor-tf32, wmma-dense-tf32 and
 * wmma-dense-fp64. Grouped/dense modes require p5; WMMA requires SM80+.
 * Dense modes cache one 36x36 reference inverse per verified factor class,
 * never per element. Every mode fuses native gather, both current FP64 mass
 * factors and FP64 scatter, without a global packed residual buffer.
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

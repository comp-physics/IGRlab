#pragma once

#include "IGR_element_block_jacobi.hpp"

namespace igrlab
{
namespace igr_internal
{

/** Bounded FP64 tensor inverse with current collocated mass scaling.
 *
 * Assemble initial exact domain/face diagonal blocks through the usual
 * PrepareVolumeBlocks/AddExactFaceBlocks/FactorBlocks sequence exactly once.
 * For a collocated GL scalar space on fixed affine Cartesian quads, setup
 * admits only an SPD Kronecker sum after physical-mass normalization. It then
 * discards dense build storage and retains shared small tensor factors.
 * RefreshMassScaling updates M from the actual invrho QF and fixed quadrature
 * geometry. Mult applies M^(-1/2) H0^(-1) M^(-1/2). This is the exact current
 * local inverse for globally uniform changes of the initial density/coefficient;
 * heterogeneous density defines an explicitly approximate SPD surrogate.
 * No assembly, LU, triangular solve, iteration, fallback, or finite scan is
 * performed by the current-coefficient update or inverse application.
 */
std::unique_ptr<ElementBlockJacobi> MakeTensorElementBlockJacobi(
  mfem::ParFiniteElementSpace& space,
  const mfem::QuadratureFunction& initial_invrho);

}
}

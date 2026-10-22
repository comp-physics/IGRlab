#pragma once

#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include <memory>

namespace igrlab
{
namespace dg_face
{
struct InteriorFaceDGDiffusionApplyInfo;
struct InteriorFaceDGDiffusionApplyScratch;
}
namespace igr_internal
{

/** Element diagonal, applied by pivoted FP64 LU or an explicit FP64 inverse.
 *
 * PrepareVolumeBlocks refreshes the actual domain integrators in native local
 * ordering. Add exact face blocks before FactorBlocks. FactorBlocks overwrites
 * those matrices with LU factors. The optional explicit-inverse backend also
 * constructs a separate inverse once per FactorBlocks call. Mult never refreshes
 * either representation; the caller controls the block generation/reuse policy.
 * The explicit-inverse backend releases build-only matrices after inversion;
 * another PrepareVolumeBlocks call restores them. Mult input/output storage
 * must be disjoint.
 * No coarse correction, damping, iteration or convergence policy is hidden here.
 * The finite element space/topology must outlive this object and remain fixed.
 */
class ElementBlockJacobi : public mfem::Solver
{
public:
  explicit ElementBlockJacobi(int size) : mfem::Solver(size) { }
  virtual mfem::DenseTensor& PrepareVolumeBlocks(mfem::BilinearForm& form) = 0;
  virtual void FactorBlocks() = 0;
  // The local tensor surrogate changes only its collocated physical mass.
  // Current-LU and frozen explicit-inverse modes retain their existing policy.
  virtual void RefreshMassScaling(const mfem::QuadratureFunction&) { }
};

std::unique_ptr<ElementBlockJacobi> MakeElementBlockJacobi(
  mfem::ParFiniteElementSpace& space, bool explicit_inverse = false);

/** Add exact interior-face A_ee terms to native column-major element blocks.
 *
 * info/work and blocks must describe the same scalar native FE space. Both
 * column sides contribute when a periodic face joins an element to itself.
 * The caller refreshes current face coefficients before invoking this function.
 */
void AddExactFaceBlocks(
  const dg_face::InteriorFaceDGDiffusionApplyInfo& info,
  const dg_face::InteriorFaceDGDiffusionApplyScratch& work,
  mfem::real_t ip_sigma, mfem::DenseTensor& blocks);

}
}

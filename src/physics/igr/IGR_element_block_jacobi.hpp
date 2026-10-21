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

/** Current-coefficient element diagonal, applied by batched pivoted FP64 LU.
 *
 * PrepareVolumeBlocks refreshes the actual domain integrators in native local
 * ordering. Add exact face blocks before FactorBlocks. FactorBlocks overwrites
 * those matrices with LU factors; Mult then solves independent element systems.
 * No coarse correction, damping, iteration or convergence policy is hidden here.
 * The finite element space/topology must outlive this object and remain fixed.
 */
class ElementBlockJacobi : public mfem::Solver
{
public:
  explicit ElementBlockJacobi(int size) : mfem::Solver(size) { }
  virtual mfem::DenseTensor& PrepareVolumeBlocks(mfem::BilinearForm& form) = 0;
  virtual void FactorBlocks() = 0;
};

std::unique_ptr<ElementBlockJacobi> MakeElementBlockJacobi(
  mfem::ParFiniteElementSpace& space);

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

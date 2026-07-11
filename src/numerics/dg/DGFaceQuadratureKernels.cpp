#include "mfem.hpp"
#include "DGFaceQuadratureKernels.hpp"

namespace dg_igr
{
namespace dg_face
{

void LaunchInteriorFaceDGDiffusionApply2D(
  const InteriorFaceDGDiffusionApplyInfo& info,
  const InteriorFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  mfem::real_t ip_sigma,
  mfem::Vector& y)
{
  AddInteriorFaceDGDiffusionApply2D(info, work, scalar, ip_sigma, y);
}

void LaunchSharedFaceDGDiffusionApply2D(
  const SharedFaceDGDiffusionApplyInfo& info,
  const SharedFaceDGDiffusionApplyScratch& work,
  const mfem::Vector& scalar,
  const mfem::Vector& scalar_face_nbr,
  mfem::real_t ip_sigma,
  mfem::Vector& y)
{
  AddSharedFaceDGDiffusionApply2D(info,
                                  work,
                                  scalar,
                                  scalar_face_nbr,
                                  ip_sigma,
                                  y);
}

} // namespace dg_face
} // namespace dg_igr

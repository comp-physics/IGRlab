#include "igrlab/compat.hpp"
#include "IGR_face_density_cuda.hpp"
#include <cuda_runtime.h>

namespace igrlab
{
namespace igr_internal
{
namespace
{
__global__ void GatherSideAlpha(int count, const int* face_elements,
                               const double* element_alpha, double* side_alpha)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) side_alpha[i] = element_alpha[face_elements[i]];
}
}

dg_face::InteriorFaceSideDensityProjectionInfo
ProjectInteriorFaceDensityCoefficientCUDA(
  const mfem::FiniteElementSpace& vfes, const mfem::FaceQuadratureSpace& fqs,
  const mfem::ParGridFunction& state, mfem::real_t scalar_alpha,
  const mfem::Vector* element_alpha, mfem::real_t rho_floor,
  mfem::Vector& side_q, dg_face::InteriorFaceSideDensityProjectionScratch& work,
  const dg_face::InteriorFaceSideDensityProjectionInfo* prepared_geometry)
{
  MFEM_VERIFY(mfem::Device::Allows(mfem::Backend::CUDA) && state.ParFESpace() == &vfes,
              "CUDA side-density projection requires CUDA and matching state space.");
  auto info = prepared_geometry ? *prepared_geometry :
    dg_face::BuildInteriorFaceSideDensityProjectionData2D(
      vfes, fqs, work.shape, work.face_elements, work.element_vdof_map);
  side_q.SetSize(2 * info.nq * info.nf); side_q.UseDevice(true);
  work.side_alpha.SetSize(element_alpha ? 2 * info.nf : 0);
  work.side_alpha.UseDevice(true);
  if (element_alpha)
  {
    MFEM_VERIFY(element_alpha->Size() == info.ne,
                "CUDA side-density projection received mismatched element alpha.");
    if (info.nf)
      GatherSideAlpha<<<(2 * info.nf + 255) / 256, 256>>>(2 * info.nf,
        work.face_elements.Read(), element_alpha->Read(), work.side_alpha.Write());
  }
  dg_face::InteriorFaceSideDensityProjection2D kernel;
  kernel.dof = info.dof; kernel.vdim = vfes.GetVDim();
  kernel.nq = info.nq; kernel.nf = info.nf; kernel.ne = info.ne;
  kernel.alpha = scalar_alpha; kernel.rho_floor = rho_floor;
  kernel.shape = work.shape.Read();
  kernel.side_alpha = element_alpha ? work.side_alpha.Read() : nullptr;
  kernel.face_elements = work.face_elements.Read();
  kernel.element_vdof_map = work.element_vdof_map.Read();
  kernel.state = state.Read(); kernel.side_q = side_q.Write();
  if (info.nf) mfem::forall_2D(info.nf, 2 * info.nq, 1, kernel);
  MFEM_VERIFY(cudaGetLastError() == cudaSuccess, "CUDA side-density launch failed.");
  info.qf_size = side_q.Size();
  return info;
}
}
}

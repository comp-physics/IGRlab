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

// Four warps per block, one element per warp. Gather its 36 density values
// once, then contract the four endpoint traces along six normal tensor lines.
__global__ void TensorDensityTraces(int ne, const int* density_map,
                                    const double* endpoints,
                                    const double* state, double* traces)
{
  constexpr int n = face_density_detail::TensorData::n, dof = n * n;
  constexpr int elements_per_block = 4;
  __shared__ double density[elements_per_block][dof];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int e = blockIdx.x * elements_per_block + warp;
  if (e < ne)
    for (int lex = lane; lex < dof; lex += 32)
      density[warp][lex] = state[density_map[e * dof + lex]];
  __syncthreads();
  if (e < ne && lane < 4 * n)
  {
    const int side = lane / n, tangent = lane % n;
    double sum = 0.0;
#pragma unroll
    for (int k = 0; k < n; ++k)
    {
      const int lex = side < 2 ? k + n * tangent : tangent + n * k;
      sum += endpoints[(side % 2) * n + k] * density[warp][lex];
    }
    traces[(4 * e + side) * n + tangent] = sum;
  }
}

__global__ void PackTensorDensity(int count, const int* face_incidence,
                                  const int* side_maps, const int* tangent_sparse,
                                  const int* tangent_permutation,
                                  const double* tangents, const double* traces,
                                  double scalar_alpha, const double* element_alpha,
                                  double rho_floor, double* side_q,
                                  double* side_alpha)
{
  constexpr int n = face_density_detail::TensorData::n;
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= count) { return; }
  const int side = entry % 2, q = (entry / 2) % n, face = entry / (2 * n);
  const int incidence = face_incidence[side + 2 * face];
  const int map = side_maps[incidence];
  const double* tangent = tangents + (map * n + q) * n;
  const double* trace = traces + incidence * n;
  double rho = 0.0;
  if (tangent_sparse[map])
  {
    const int pivot = tangent_permutation[map * n + q];
    rho = tangent[pivot] * trace[pivot];
  }
  else
  {
    // Retain all actual tangential coefficients, including off-diagonal
    // roundoff from reversed reference coordinates.
#pragma unroll
    for (int k = 0; k < n; ++k) { rho += tangent[k] * trace[k]; }
  }
  const double alpha = element_alpha ? element_alpha[incidence / 4] : scalar_alpha;
  side_q[entry] = frozen_scalar_igr_algebra::AlphaOverDensity(alpha, rho, rho_floor);
  if (side_alpha && q == 0) { side_alpha[side + 2 * face] = alpha; }
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
  const auto* prepared = dynamic_cast<const face_density_detail::QuadratureSpace*>(&fqs);
  auto* tensor = prepared ? prepared->tensor.get() : nullptr;
  auto info = tensor ? tensor->info : prepared_geometry ? *prepared_geometry :
    dg_face::BuildInteriorFaceSideDensityProjectionData2D(
      vfes, fqs, work.shape, work.face_elements, work.element_vdof_map);
  side_q.SetSize(2 * info.nq * info.nf); side_q.UseDevice(true);
  work.side_alpha.SetSize(element_alpha ? 2 * info.nf : 0);
  work.side_alpha.UseDevice(true);
  if (element_alpha)
    MFEM_VERIFY(element_alpha->Size() == info.ne,
                "CUDA side-density projection received mismatched element alpha.");
  if (tensor)
  {
    if (info.nf)
    {
      TensorDensityTraces<<<(info.ne + 3) / 4, 128>>>(info.ne,
        tensor->density_map.Read(), tensor->endpoints.Read(), state.Read(),
        tensor->density_traces.Write());
      const int count = 2 * info.nq * info.nf;
      PackTensorDensity<<<(count + 255) / 256, 256>>>(count,
        tensor->face_incidence.Read(), tensor->side_maps.Read(),
        tensor->tangent_sparse.Read(), tensor->tangent_permutation.Read(),
        tensor->tangents.Read(), tensor->density_traces.Read(), scalar_alpha,
        element_alpha ? element_alpha->Read() : nullptr, rho_floor, side_q.Write(),
        element_alpha ? work.side_alpha.Write() : nullptr);
    }
  }
  else
  {
    if (!prepared && face_density_detail::RequestedExecution() == face_density_detail::Execution::Tensor)
    {
      static bool logged_uncached_fallback = false;
      if (!logged_uncached_fallback && mfem::Mpi::Root())
        std::cout << "[igr-face-density] event=execution requested=tensor selected=dense "
                     "reason=requires-geometry-cache" << std::endl;
      logged_uncached_fallback = true;
    }
    if (element_alpha && info.nf)
      GatherSideAlpha<<<(2 * info.nf + 255) / 256, 256>>>(2 * info.nf,
        work.face_elements.Read(), element_alpha->Read(), work.side_alpha.Write());
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
  }
  MFEM_VERIFY(cudaGetLastError() == cudaSuccess, "CUDA side-density launch failed.");
  info.qf_size = side_q.Size();
  return info;
}
}
}

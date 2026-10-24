#include "IGR_tensor_volume.hpp"
#include "utils/config/env_flags.hpp"
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace igrlab
{
namespace igr_internal
{
namespace
{
constexpr int side = 8;
constexpr int tile_size = side * side;
constexpr int warps = 4;

// All operands are padded column-major 8x8 matrices, aligned to 32 bytes.
// Distinct output is required. SIMT follows the same separable contractions.
template <bool tensor>
__device__ __forceinline__ void WarpProduct(
  const double* a, const double* b, double* c, int inner, int lane)
{
  if constexpr (tensor)
  {
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> accum;
    wmma::fill_fragment(accum, 0.0);
    for (int k = 0; k < inner; k += 4)
    {
      wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::col_major> left;
      wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> right;
      wmma::load_matrix_sync(left, a + side * k, side);
      wmma::load_matrix_sync(right, b + k, side);
      wmma::mma_sync(accum, left, right, accum);
    }
    wmma::store_matrix_sync(c, accum, side, wmma::mem_col_major);
#else
    asm("trap;");
#endif
  }
  else
  {
    for (int i = lane; i < tile_size; i += 32)
    {
      const int row = i % side, col = i / side;
      double value = 0.0;
      for (int k = 0; k < inner; ++k) { value += a[row + side*k] * b[k + side*col]; }
      c[i] = value;
    }
  }
  __syncwarp();
}

// One real element per warp. Both quadrature grids, pointwise geometry/current
// coefficient products, and forward/transpose intermediates stay in shared memory.
// The columns are different nodal/quadrature lines, never replicated RHSs.
template <bool tensor>
__global__ void IGRFusedTensorVolumeKernel(
  int ne, int d, int qm, int qd, const int* dof_map, const double* maps,
  const double* mass, const double* diffusion, const double* x, double* y)
{
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int e = blockIdx.x * warps + warp;
  if (e >= ne) { return; } // warp-uniform; no CTA synchronization below
  __shared__ __align__(32) double u[warps][tile_size];
  __shared__ __align__(32) double t0[warps][tile_size], t1[warps][tile_size];
  __shared__ __align__(32) double v0[warps][tile_size], v1[warps][tile_size];
  __shared__ __align__(32) double result[warps][tile_size];
  const double* bm = maps;
  const double* bmt = maps + tile_size;
  const double* bd = maps + 2*tile_size;
  const double* bdt = maps + 3*tile_size;
  const double* gd = maps + 4*tile_size;
  const double* gdt = maps + 5*tile_size;
  for (int i = lane; i < tile_size; i += 32)
  {
    const int r = i % side, c = i / side;
    double value = 0.0;
    if (r < d && c < d)
    {
      const int index = dof_map[e*d*d + r + d*c];
      value = index >= 0 ? x[index] : -x[-1-index];
    }
    u[warp][i] = value;
  }
  __syncwarp();
  WarpProduct<tensor>(bm, u[warp], t0[warp], d, lane);
  WarpProduct<tensor>(t0[warp], bmt, v0[warp], d, lane);
  for (int i = lane; i < tile_size; i += 32)
  {
    v0[warp][i] = i%side < qm && i/side < qm ?
      v0[warp][i] * mass[e*tile_size+i] : 0.0;
  }
  __syncwarp();
  WarpProduct<tensor>(bmt, v0[warp], t0[warp], qm, lane);
  WarpProduct<tensor>(t0[warp], bm, result[warp], qm, lane);

  WarpProduct<tensor>(gd, u[warp], t0[warp], d, lane);
  WarpProduct<tensor>(bd, u[warp], t1[warp], d, lane);
  WarpProduct<tensor>(t0[warp], bdt, v0[warp], d, lane);
  WarpProduct<tensor>(t1[warp], gdt, v1[warp], d, lane);
  for (int i = lane; i < tile_size; i += 32)
  {
    const double gx = v0[warp][i], gy = v1[warp][i];
    const double* metric = diffusion + e*3*tile_size + i;
    v0[warp][i] = i%side < qd && i/side < qd ? metric[0]*gx + metric[tile_size]*gy : 0.0;
    v1[warp][i] = i%side < qd && i/side < qd ? metric[tile_size]*gx + metric[2*tile_size]*gy : 0.0;
  }
  __syncwarp();
  WarpProduct<tensor>(gdt, v0[warp], t0[warp], qd, lane);
  WarpProduct<tensor>(bdt, v1[warp], t1[warp], qd, lane);
  WarpProduct<tensor>(t0[warp], bd, v0[warp], qd, lane);
  WarpProduct<tensor>(t1[warp], gd, v1[warp], qd, lane);
  for (int i = lane; i < tile_size; i += 32)
  {
    const int r = i%side, c = i/side;
    if (r < d && c < d)
    {
      const int index = dof_map[e*d*d + r + d*c];
      const double value = result[warp][i] + v0[warp][i] + v1[warp][i];
      if (index >= 0) { y[index] = value; }
      else { y[-1-index] = -value; }
    }
  }
}

// Fixed D=6, Qmass=7, Qdiff=6 contractions keep only real rows/columns.
// The actual (possibly nonidentity) maps and full mapped diffusion metric are
// unchanged. Reference maps are shared by the four element warps in a CTA.
__global__ void IGRCompactTensorVolumeKernel(
  int ne, const int* dof_map, const double* maps, const double* mass,
  const double* diffusion, const double* x, double* y)
{
  constexpr int d = 6, qm = 7, points = d*d;
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int e = blockIdx.x*warps+warp;
  __shared__ double bm[qm*d], b[points], g[points];
  __shared__ double u[warps][points], result[warps][points];
  __shared__ double t0[warps][qm*d], t1[warps][qm*d];
  __shared__ double v0[warps][qm*qm], v1[warps][qm*qm];
  if (threadIdx.x < qm*d)
  {
    const int i = threadIdx.x;
    bm[i] = maps[i%qm+side*(i/qm)];
  }
  if (threadIdx.x < points)
  {
    const int i = threadIdx.x, padded = i%d+side*(i/d);
    b[i] = maps[2*tile_size+padded];
    g[i] = maps[4*tile_size+padded];
  }
  __syncthreads(); // partial final CTAs must participate before returning
  if (e >= ne) { return; }
  for (int i = lane; i < points; i += 32)
  {
    const int index = dof_map[e*points+i];
    u[warp][i] = index >= 0 ? x[index] : -x[-1-index];
  }
  __syncwarp();
  for (int i = lane; i < qm*d; i += 32)
  {
    const int qx = i%qm, dy = i/qm;
    double value = 0.0;
#pragma unroll
    for (int k = 0; k < d; ++k) { value += bm[qx+qm*k]*u[warp][k+d*dy]; }
    t0[warp][i] = value;
  }
  __syncwarp();
  for (int i = lane; i < qm*qm; i += 32)
  {
    const int qx = i%qm, qy = i/qm;
    double value = 0.0;
#pragma unroll
    for (int k = 0; k < d; ++k) { value += t0[warp][qx+qm*k]*bm[qy+qm*k]; }
    v0[warp][i] = value*mass[e*tile_size+qx+side*qy];
  }
  __syncwarp();
  for (int i = lane; i < d*qm; i += 32)
  {
    const int dx = i%d, qy = i/d;
    double value = 0.0;
#pragma unroll
    for (int k = 0; k < qm; ++k) { value += bm[k+qm*dx]*v0[warp][k+qm*qy]; }
    t0[warp][i] = value;
  }
  __syncwarp();
  for (int i = lane; i < points; i += 32)
  {
    const int dx = i%d, dy = i/d;
    double value = 0.0;
#pragma unroll
    for (int k = 0; k < qm; ++k) { value += t0[warp][dx+d*k]*bm[k+qm*dy]; }
    result[warp][i] = value;
  }
  __syncwarp();
  for (int i = lane; i < points; i += 32)
  {
    const int qx = i%d, dy = i/d;
    double derivative = 0.0, value = 0.0;
#pragma unroll
    for (int k = 0; k < d; ++k)
    {
      derivative += g[qx+d*k]*u[warp][k+d*dy];
      value += b[qx+d*k]*u[warp][k+d*dy];
    }
    t0[warp][i] = derivative;
    t1[warp][i] = value;
  }
  __syncwarp();
  for (int i = lane; i < points; i += 32)
  {
    const int qx = i%d, qy = i/d;
    double gx = 0.0, gy = 0.0;
#pragma unroll
    for (int k = 0; k < d; ++k)
    {
      gx += t0[warp][qx+d*k]*b[qy+d*k];
      gy += t1[warp][qx+d*k]*g[qy+d*k];
    }
    const double* metric = diffusion+e*3*tile_size+qx+side*qy;
    v0[warp][i] = metric[0]*gx+metric[tile_size]*gy;
    v1[warp][i] = metric[tile_size]*gx+metric[2*tile_size]*gy;
  }
  __syncwarp();
  for (int i = lane; i < points; i += 32)
  {
    const int dx = i%d, qy = i/d;
    double value0 = 0.0, value1 = 0.0;
#pragma unroll
    for (int k = 0; k < d; ++k)
    {
      value0 += g[k+d*dx]*v0[warp][k+d*qy];
      value1 += b[k+d*dx]*v1[warp][k+d*qy];
    }
    t0[warp][i] = value0;
    t1[warp][i] = value1;
  }
  __syncwarp();
  for (int i = lane; i < points; i += 32)
  {
    const int dx = i%d, dy = i/d;
    double value0 = 0.0, value1 = 0.0;
#pragma unroll
    for (int k = 0; k < d; ++k)
    {
      value0 += t0[warp][dx+d*k]*b[k+d*dy];
      value1 += t1[warp][dx+d*k]*g[k+d*dy];
    }
    const double value = result[warp][i]+value0+value1;
    const int index = dof_map[e*points+i];
    if (index >= 0) { y[index] = value; }
    else { y[-1-index] = -value; }
  }
}

// Admission proves that both interpolation maps are diagonal, not that they
// equal the identity. Retain each actual diagonal multiply at its original
// forward/transpose stage. Only the four derivative contractions remain dense.
template <int D, bool direct>
__global__ void IGRFusedCollocatedTensorVolumeKernel(
  int ne, const int* dof_map, const double* maps, const double* mass,
  const double* diffusion, const double* x, double* y,
  int mass_geometry_stride, int diffusion_geometry_stride,
  const double* invrho, const double* ck)
{
  constexpr int points = D*D;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int e = blockIdx.x * warps + warp;
  __shared__ double g[points], bm[D], bd[D];
  __shared__ double u[warps][points], fx[warps][points], fy[warps][points];
  for (int i = threadIdx.x; i < points; i += blockDim.x)
  {
    g[i] = maps[4*tile_size + i%D + side*(i/D)];
  }
  if (threadIdx.x < D)
  {
    const int i = threadIdx.x;
    bm[i] = maps[i + side*i];
    bd[i] = maps[2*tile_size + i + side*i];
  }
  // Every warp must participate before a partial final CTA can return.
  __syncthreads();
  if (e >= ne) { return; }
  for (int i = lane; i < points; i += 32)
  {
    const int index = dof_map[e*points+i];
    u[warp][i] = index >= 0 ? x[index] : -x[-1-index];
  }
  __syncwarp();
  for (int i = lane; i < points; i += 32)
  {
    const int r = i%D, c = i/D;
    double gx = 0.0, gy = 0.0;
#pragma unroll
    for (int k = 0; k < D; ++k)
    {
      gx += g[r+D*k] * u[warp][k+D*c];
      const double by = bd[r] * u[warp][r+D*k];
      gy += by * g[c+D*k];
    }
    gx *= bd[c];
    if constexpr (direct)
    {
      const double* metric = diffusion + e*diffusion_geometry_stride + i;
      const double coefficient = ck[e*points+i];
      // The cached path rounds geometry*coefficient before using the metric.
      // Preserve that intermediate rather than contracting it into a flux.
      const double d00 = __dmul_rn(metric[0], coefficient);
      const double d01 = __dmul_rn(metric[points], coefficient);
      const double d11 = __dmul_rn(metric[2*points], coefficient);
      fx[warp][i] = d00*gx + d01*gy;
      fy[warp][i] = d01*gx + d11*gy;
    }
    else
    {
      const double* metric = diffusion + e*3*tile_size + r + side*c;
      fx[warp][i] = metric[0]*gx + metric[tile_size]*gy;
      fy[warp][i] = metric[tile_size]*gx + metric[2*tile_size]*gy;
    }
  }
  __syncwarp();
  for (int i = lane; i < points; i += 32)
  {
    const int r = i%D, c = i/D;
    double dx = 0.0, dy = 0.0;
#pragma unroll
    for (int k = 0; k < D; ++k)
    {
      dx += g[k+D*r] * fx[warp][k+D*c];
      const double by = bd[r] * fy[warp][r+D*k];
      dy += by * g[k+D*c];
    }
    dx *= bd[c];
    double value = bm[r] * u[warp][i];
    value *= bm[c];
    if constexpr (direct)
    {
      value *= __dmul_rn(mass[e*mass_geometry_stride+i], invrho[e*points+i]);
    }
    else
    {
      value *= mass[e*tile_size+r+side*c];
    }
    value = bm[r] * value;
    value *= bm[c];
    value += dx + dy;
    const int index = dof_map[e*points+i];
    if (index >= 0) { y[index] = value; }
    else { y[-1-index] = -value; }
  }
}

// Instantiate compact shared-memory fields for every admitted D1D.
template <bool direct, int D = 1>
void LaunchCollocatedTensorVolumeKernel(
  int d, int ne, const int* dof_map, const double* maps, const double* mass,
  const double* diffusion, const double* x, double* y,
  int mass_geometry_stride = 0, int diffusion_geometry_stride = 0,
  const double* invrho = nullptr, const double* ck = nullptr)
{
  if (d == D)
  {
    IGRFusedCollocatedTensorVolumeKernel<D, direct><<<(ne+warps-1)/warps,32*warps>>>(
      ne,dof_map,maps,mass,diffusion,x,y,
      mass_geometry_stride,diffusion_geometry_stride,invrho,ck);
  }
  else if constexpr (D < side)
  {
    LaunchCollocatedTensorVolumeKernel<direct, D+1>(
      d,ne,dof_map,maps,mass,diffusion,x,y,
      mass_geometry_stride,diffusion_geometry_stride,invrho,ck);
  }
  else
  {
    MFEM_ABORT("Invalid collocated IGR tensor volume D1D.");
  }
}

const char* CollocatedMapRejection(const mfem::DofToQuad& map, int d)
{
  const int entries = d*d;
  if (map.B.Size() != entries || map.Bt.Size() != entries ||
      map.G.Size() != entries || map.Gt.Size() != entries)
  {
    return "map-size";
  }
  const double* b = map.B.HostRead(); const double* bt = map.Bt.HostRead();
  const double* g = map.G.HostRead(); const double* gt = map.Gt.HostRead();
  for (int j = 0; j < d; ++j) for (int i = 0; i < d; ++i)
  {
    const int forward = i+d*j, transpose = j+d*i;
    if (!std::isfinite(b[forward]) || !std::isfinite(bt[transpose]) ||
        !std::isfinite(g[forward]) || !std::isfinite(gt[transpose]))
    {
      return "nonfinite-map";
    }
    if (i != j && b[forward] != 0.0) { return "nonzero-b-offdiagonal"; }
    if (b[forward] != bt[transpose] || g[forward] != gt[transpose])
    {
      return "transpose-mismatch";
    }
  }
  return nullptr;
}

__global__ void IGRTensorVolumeRefreshKernel(
  int ne, int qm, int qd, const double* mass_geometry,
  const double* diffusion_geometry, const double* invrho, const double* ck,
  double* mass, double* diffusion)
{
  const int entry = blockIdx.x*blockDim.x + threadIdx.x;
  if (entry >= ne*tile_size) { return; }
  const int e = entry/tile_size, q = entry%tile_size;
  const int qx = q%side, qy = q/side;
  mass[entry] = qx < qm && qy < qm ?
    mass_geometry[entry]*invrho[e*qm*qm + qx + qm*qy] : 0.0;
  const double value = qx < qd && qy < qd ? ck[e*qd*qd + qx + qd*qy] : 0.0;
  for (int c = 0; c < 3; ++c)
  {
    const int index = e*3*tile_size + c*tile_size + q;
    diffusion[index] = diffusion_geometry[index]*value;
  }
}

int CheckRule(const mfem::IntegrationRule& ir)
{
  const int q = static_cast<int>(std::lround(std::sqrt(double(ir.GetNPoints()))));
  MFEM_VERIFY(q >= 1 && q <= side && q*q == ir.GetNPoints(),
              "IGR tensor volume requires separate tensor mass/diffusion Q1D <= 8.");
  for (int j = 0; j < q; ++j)
  {
    for (int i = 0; i < q; ++i)
    {
      const auto& ip = ir.IntPoint(i+q*j);
      MFEM_VERIFY(ip.x == ir.IntPoint(i).x && ip.y == ir.IntPoint(j).x &&
                    std::isfinite(ip.x) && std::isfinite(ip.y) &&
                    std::isfinite(ip.weight) && ip.weight > 0.0,
                  "IGR tensor volume requires lexicographic square tensor quadrature.");
    }
  }
  return q;
}

const mfem::QuadratureSpace& CheckQF(const mfem::QuadratureFunction& qf,
                                   mfem::ParFiniteElementSpace& fes)
{
  const auto* space = dynamic_cast<const mfem::QuadratureSpace*>(qf.GetSpace());
  MFEM_VERIFY(space && space->GetMesh() == fes.GetMesh() &&
                space->GetNE() == fes.GetNE() && qf.GetVDim() == 1 &&
                qf.Size() == space->GetSize(),
              "IGR tensor volume requires scalar volume QFs on the active mesh.");
  return *space;
}

std::string TensorVolumeStorageSelection()
{
  const char* value = ::igrlab::GetEnv("DG_IGR_IGR_TENSOR_VOLUME_STORAGE");
  const std::string storage = value ? value : "cached";
  MFEM_VERIFY(storage == "cached" || storage == "direct",
              "IGR tensor volume storage must be cached|direct.");
  return storage;
}

class TensorVolumeOperator final : public mfem::Operator
{
public:
  TensorVolumeOperator(mfem::ParFiniteElementSpace& fes,
                       const mfem::QuadratureFunction& invrho,
                       const mfem::QuadratureFunction& ck,
                       const std::string& execution)
    : mfem::Operator(fes.GetVSize()), fes_(fes), execution_(execution),
      collocated_requested_(::igrlab::EnvFlagEnabledDefault(
        "DG_IGR_IGR_TENSOR_VOLUME_COLLOCATED", false)),
      storage_requested_(TensorVolumeStorageSelection()),
      mass_space_(&CheckQF(invrho, fes)), diffusion_space_(&CheckQF(ck, fes))
  {
    static_assert(sizeof(mfem::real_t) == sizeof(double), "FP64 MFEM required");
    MFEM_VERIFY(execution == "simt" || execution == "wmma", "IGR tensor volume execution must be simt|wmma.");
    MFEM_VERIFY(mfem::Device::Allows(mfem::Backend::CUDA_MASK) &&
                  fes.GetNRanks() == 1 && fes.Conforming() && fes.GetVDim() == 1 &&
                  fes.GetTrueVSize() == fes.GetVSize() &&
                  mfem::IsIdentityProlongation(fes.GetProlongationMatrix()) &&
                  mfem::IsIdentityProlongation(fes.GetRestrictionTransposeOperator()) &&
                  dynamic_cast<const mfem::L2_FECollection*>(fes.FEColl()) &&
                  fes.GetMesh()->Dimension() == 2 && fes.GetMesh()->SpaceDimension() == 2,
                "IGR tensor volume requires rank-one scalar VALUE L2 quads and identity local/true maps.");
    int device = 0; cudaDeviceProp properties{};
    MFEM_VERIFY(cudaGetDevice(&device) == cudaSuccess &&
                  cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
                  (execution != "wmma" || properties.major >= 8),
                "IGR FP64 WMMA requires native SM80 or newer.");
    ne_ = fes.GetNE();
    MFEM_VERIFY(ne_ > 0 && ne_ <= (std::numeric_limits<int>::max()-255)/(3*tile_size),
                "IGR tensor volume element/index bound exceeded.");
    const auto& fe = *fes.GetFE(0);
    d_ = fe.GetOrder()+1;
    MFEM_VERIFY(d_ >= 1 && d_ <= side, "IGR tensor volume supports D1D <= 8.");
    qm_ = CheckRule(mass_space_->GetIntRule(0));
    qd_ = CheckRule(diffusion_space_->GetIntRule(0));
    BuildMaps(fe);
    direct_storage_ = storage_requested_ == "direct" && collocated_enabled_;
    compact_requested_ = ::igrlab::EnvFlagEnabledDefault("DG_IGR_IGR_TENSOR_VOLUME_COMPACT", false);
    compact_enabled_ = compact_requested_ && execution_ == "simt" &&
      d_ == 6 && qm_ == 7 && qd_ == 6 && !direct_storage_;
    compact_reason_ = !compact_requested_ ? "not-requested" :
      execution_ != "simt" ? "execution-not-simt" :
      (d_ != 6 || qm_ != 7 || qd_ != 6) ? "requires-d6-qmass7-qdiff6" :
      direct_storage_ ? "requires-cached-storage" : "admitted";
    storage_reason_ = storage_requested_ == "cached" ? "cached-requested" :
      (direct_storage_ ? "admitted" : "requires-collocated-simt");
    dof_map_.SetSize(ne_*d_*d_);
    int* map = dof_map_.HostWrite();
    std::vector<bool> visited(fes.GetVSize(), false);
    const auto* reference_tensor = dynamic_cast<const mfem::TensorBasisElement*>(&fe);
    MFEM_VERIFY(reference_tensor, "IGR tensor volume requires tensor elements.");
    for (int e = 0; e < ne_; ++e)
    {
      const auto& current = *fes.GetFE(e);
      const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(&current);
      MFEM_VERIFY(tensor && current.GetGeomType() == mfem::Geometry::SQUARE &&
                    current.GetMapType() == mfem::FiniteElement::VALUE &&
                    current.GetOrder() == d_-1 && current.GetDof() == d_*d_ &&
                    tensor->GetBasisType() == reference_tensor->GetBasisType(),
                  "IGR tensor volume requires a uniform tensor VALUE basis.");
      const auto& lex = tensor->GetDofMap();
      MFEM_VERIFY(lex.Size() == 0 || lex.Size() == d_*d_, "Invalid tensor dof permutation.");
      mfem::Array<int> dofs; fes.GetElementDofs(e, dofs);
      MFEM_VERIFY(dofs.Size() == d_*d_, "Invalid scalar L2 element layout.");
      for (int i = 0; i < d_*d_; ++i)
      {
        const int local = lex.Size() ? lex[i] : i;
        MFEM_VERIFY(local >= 0 && local < d_*d_, "Invalid lexicographic dof index.");
        const int encoded = dofs[local], index = encoded >= 0 ? encoded : -1-encoded;
        MFEM_VERIFY(index >= 0 && index < fes.GetVSize() && !visited[index],
                    "IGR tensor volume requires disjoint complete scalar L2 elements.");
        visited[index] = true; map[e*d_*d_+i] = encoded;
      }
      MFEM_VERIFY(mass_space_->Offset(e) == e*qm_*qm_ &&
                    diffusion_space_->Offset(e) == e*qd_*qd_, "Unsupported QF offsets.");
    }
    MFEM_VERIFY(std::all_of(visited.begin(), visited.end(), [](bool value){return value;}) &&
                  invrho.Size() == ne_*qm_*qm_ && ck.Size() == ne_*qd_*qd_,
                "Incomplete IGR tensor volume layout.");
    BuildGeometry();
    space_sequence_ = fes.GetSequence();
    mesh_sequence_ = fes.GetMesh()->GetSequence();
    nodes_sequence_ = fes.GetMesh()->GetNodesSequence();
    maps_.UseDevice(true); mass_geometry_.UseDevice(true); diffusion_geometry_.UseDevice(true);
    dof_map_.Read(); maps_.Read(); mass_geometry_.Read(); diffusion_geometry_.Read();
    if (!direct_storage_)
    {
      mass_.SetSize(ne_*tile_size); mass_.UseDevice(true);
      diffusion_.SetSize(ne_*3*tile_size); diffusion_.UseDevice(true);
      mass_.Write(); diffusion_.Write();
    }
    Refresh(invrho, ck);
    const std::size_t geometry_bytes =
      (std::size_t(mass_geometry_.Size()) + diffusion_geometry_.Size())*sizeof(double);
    const std::size_t coefficient_bytes =
      (std::size_t(mass_.Size()) + diffusion_.Size())*sizeof(double);
    const std::size_t owned_bytes = geometry_bytes + coefficient_bytes +
      std::size_t(dof_map_.Size())*sizeof(int) + std::size_t(maps_.Size())*sizeof(double);
    std::cout << "[igr-tensor-volume] event=setup execution=" << execution_
              << " precision=fp64 d1d=" << d_ << " mass_q1d=" << qm_
              << " diffusion_q1d=" << qd_ << " elements=" << ne_
              << " geometry=per-quadrature coefficient_policy=current-stage"
              << " quadrature=original-separate local_true=identity faces=external"
              << " compact_requested=" << int(compact_requested_)
              << " compact_enabled=" << int(compact_enabled_)
              << " compact_reason=" << compact_reason_
              << " collocated_requested=" << int(collocated_requested_)
              << " collocated_enabled=" << int(collocated_enabled_)
              << " collocated_reason=" << collocated_reason_
              << " storage_requested=" << storage_requested_
              << " storage=" << (direct_storage_ ? "direct" : "cached")
              << " storage_reason=" << storage_reason_
              << " geometry_layout=" << (direct_storage_ ? "compact" : "padded")
              << " mass_geometry_shared=" << int(mass_geometry_stride_ == 0)
              << " diffusion_geometry_shared=" << int(diffusion_geometry_stride_ == 0)
              << " owned_geometry_bytes=" << geometry_bytes
              << " owned_coefficient_bytes=" << coefficient_bytes
              << " owned_bytes=" << owned_bytes
              << " kernel=" << (compact_enabled_ ? "compact-simt-d6-qmass7-qdiff6" :
                  collocated_enabled_ ? "collocated-simt-one-warp-per-element" :
                                       "fused-one-warp-per-element")
              << " fallback=none" << std::endl;
  }

  void Refresh(const mfem::QuadratureFunction& invrho, const mfem::QuadratureFunction& ck)
  {
    CheckUnchanged();
    MFEM_VERIFY(&CheckQF(invrho, fes_) == mass_space_ && &CheckQF(ck, fes_) == diffusion_space_ &&
                  invrho.Size() == ne_*qm_*qm_ && ck.Size() == ne_*qd_*qd_,
                "IGR tensor volume quadrature changed; reconstruct the operator.");
    if (direct_storage_)
    {
      // Keep object identities, not data pointers: MFEM may reallocate or change
      // the authoritative memory location of a current coefficient buffer.
      invrho_ = &invrho;
      ck_ = &ck;
    }
    else
    {
      IGRTensorVolumeRefreshKernel<<<(ne_*tile_size+255)/256,256>>>(
        ne_, qm_, qd_, mass_geometry_.Read(), diffusion_geometry_.Read(),
        invrho.Read(), ck.Read(), mass_.Write(), diffusion_.Write());
      MFEM_VERIFY(cudaGetLastError() == cudaSuccess, "IGR tensor coefficient refresh launch failed.");
    }
    ++refreshes_;
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    CheckUnchanged();
    MFEM_VERIFY(x.Size() == Width() && &x != &y, "IGR tensor volume size/alias mismatch.");
    y.SetSize(Height()); y.UseDevice(true);
    const double* input = x.Read(); double* output = y.Write();
    MFEM_VERIFY(input != output, "IGR tensor volume does not support aliased input/output.");
    if (execution_ == "wmma")
    {
      IGRFusedTensorVolumeKernel<true><<<(ne_+warps-1)/warps,32*warps>>>(
        ne_,d_,qm_,qd_,dof_map_.Read(),maps_.Read(),mass_.Read(),diffusion_.Read(),input,output);
    }
    else if (compact_enabled_)
    {
      IGRCompactTensorVolumeKernel<<<(ne_+warps-1)/warps,32*warps>>>(
        ne_,dof_map_.Read(),maps_.Read(),mass_.Read(),diffusion_.Read(),input,output);
    }
    else if (direct_storage_)
    {
      MFEM_VERIFY(invrho_ && ck_, "Direct IGR tensor coefficients are not bound.");
      LaunchCollocatedTensorVolumeKernel<true>(
        d_,ne_,dof_map_.Read(),maps_.Read(),mass_geometry_.Read(),diffusion_geometry_.Read(),
        input,output,mass_geometry_stride_,diffusion_geometry_stride_,invrho_->Read(),ck_->Read());
    }
    else if (collocated_enabled_)
    {
      LaunchCollocatedTensorVolumeKernel<false>(
        d_,ne_,dof_map_.Read(),maps_.Read(),mass_.Read(),diffusion_.Read(),input,output);
    }
    else
    {
      IGRFusedTensorVolumeKernel<false><<<(ne_+warps-1)/warps,32*warps>>>(
        ne_,d_,qm_,qd_,dof_map_.Read(),maps_.Read(),mass_.Read(),diffusion_.Read(),input,output);
    }
    MFEM_VERIFY(cudaGetLastError() == cudaSuccess, "IGR fused tensor volume launch failed.");
    ++applications_;
  }
  mfem::MemoryClass GetMemoryClass() const override { return mfem::MemoryClass::DEVICE; }
  void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override { Mult(x,y); }
  ~TensorVolumeOperator() override
  {
    std::cout << "[igr-tensor-volume] event=summary execution=" << execution_
              << " coefficient_refreshes=" << refreshes_ << " submitted_applications=" << applications_
              << " compact_requested=" << int(compact_requested_)
              << " compact_enabled=" << int(compact_enabled_)
              << " compact_reason=" << compact_reason_
              << " collocated_requested=" << int(collocated_requested_)
              << " collocated_enabled=" << int(collocated_enabled_)
              << " collocated_reason=" << collocated_reason_
              << " storage=" << (direct_storage_ ? "direct" : "cached")
              << " precision=fp64 fallback=none" << std::endl;
  }

private:
  bool compact_requested_ = false, compact_enabled_ = false;
  std::string compact_reason_ = "not-requested";
  void CheckUnchanged() const
  {
    MFEM_VERIFY(fes_.GetSequence() == space_sequence_ &&
                  fes_.GetMesh()->GetSequence() == mesh_sequence_ &&
                  fes_.GetMesh()->GetNodesSequence() == nodes_sequence_,
                "IGR tensor volume space/geometry changed; reconstruct the operator.");
  }
  void BuildMaps(const mfem::FiniteElement& fe)
  {
    MFEM_VERIFY(fe.GetGeomType() == mfem::Geometry::SQUARE &&
                  dynamic_cast<const mfem::TensorBasisElement*>(&fe), "Tensor quadrilateral required.");
    const auto& mass = fe.GetDofToQuad(mass_space_->GetIntRule(0), mfem::DofToQuad::TENSOR);
    const auto& diff = fe.GetDofToQuad(diffusion_space_->GetIntRule(0), mfem::DofToQuad::TENSOR);
    MFEM_VERIFY(mass.ndof == d_ && diff.ndof == d_ && mass.nqpt == qm_ && diff.nqpt == qd_,
                "IGR tensor basis/quadrature mismatch.");
    if (collocated_requested_)
    {
      if (execution_ != "simt") { collocated_reason_ = "execution-not-simt"; }
      else if (d_ != qm_ || d_ != qd_) { collocated_reason_ = "unequal-d1d-q1d"; }
      else
      {
        const char* reason = CollocatedMapRejection(mass, d_);
        if (!reason) { reason = CollocatedMapRejection(diff, d_); }
        collocated_enabled_ = !reason;
        collocated_reason_ = reason ? reason : "admitted";
      }
    }
    maps_.SetSize(6*tile_size); maps_ = 0.0; double* out = maps_.HostReadWrite();
    auto store = [&](const mfem::Array<mfem::real_t>& values, int q, int block)
    {
      const double* v = values.HostRead();
      for (int j = 0; j < d_; ++j) for (int i = 0; i < q; ++i)
      {
        MFEM_VERIFY(std::isfinite(v[i+q*j]), "Nonfinite IGR tensor basis.");
        out[block*tile_size+i+side*j] = v[i+q*j];
        out[(block+1)*tile_size+j+side*i] = v[i+q*j];
      }
    };
    store(mass.B,qm_,0); store(diff.B,qd_,2); store(diff.G,qd_,4);
  }
  void BuildGeometry()
  {
    const auto& mi = mass_space_->GetIntRule(0);
    const auto& di = diffusion_space_->GetIntRule(0);
    const auto* mg = fes_.GetMesh()->GetGeometricFactors(mi, mfem::GeometricFactors::DETERMINANTS);
    const auto* dg = fes_.GetMesh()->GetGeometricFactors(di, mfem::GeometricFactors::JACOBIANS);
    const double* det = mg->detJ.HostRead(); const double* jac = dg->J.HostRead();
    const int mass_points = direct_storage_ ? qm_*qm_ : tile_size;
    const int diffusion_points = direct_storage_ ? qd_*qd_ : tile_size;
    const int mass_side = direct_storage_ ? qm_ : side;
    const int diffusion_side = direct_storage_ ? qd_ : side;
    mass_geometry_stride_ = mass_points;
    diffusion_geometry_stride_ = 3*diffusion_points;
    mass_geometry_.SetSize(ne_*mass_points); mass_geometry_ = 0.0;
    diffusion_geometry_.SetSize(ne_*3*diffusion_points); diffusion_geometry_ = 0.0;
    double* m = mass_geometry_.HostReadWrite(); double* g = diffusion_geometry_.HostReadWrite();
    for (int e = 0; e < ne_; ++e)
    {
      for (int j = 0; j < qm_; ++j) for (int i = 0; i < qm_; ++i)
      {
        const int q = i+qm_*j;
        MFEM_VERIFY(std::isfinite(det[e*qm_*qm_+q]) && det[e*qm_*qm_+q] > 0.0,
                    "IGR tensor volume requires positive finite Jacobians.");
        m[e*mass_points+i+mass_side*j] = mi.IntPoint(q).weight*det[e*qm_*qm_+q];
      }
      const int nq = qd_*qd_;
      for (int j = 0; j < qd_; ++j) for (int i = 0; i < qd_; ++i)
      {
        const int q = i+qd_*j, offset = e*4*nq+q;
        const double a=jac[offset], b=jac[offset+nq], c=jac[offset+2*nq], d=jac[offset+3*nq];
        const double determinant = a*d-b*c;
        MFEM_VERIFY(std::isfinite(determinant) && determinant > 0.0,
                    "IGR tensor diffusion requires positive finite Jacobians.");
        const double factor = di.IntPoint(q).weight/determinant;
        const int target = e*3*diffusion_points+i+diffusion_side*j;
        g[target] = factor*(c*c+d*d);
        g[target+diffusion_points] = -factor*(c*a+d*b);
        g[target+2*diffusion_points] = factor*(a*a+b*b);
        MFEM_VERIFY(std::isfinite(g[target]) && std::isfinite(g[target+diffusion_points]) &&
                      std::isfinite(g[target+2*diffusion_points]), "Nonfinite IGR tensor diffusion metric.");
      }
    }
    if (direct_storage_)
    {
      mass_geometry_stride_ = ShareIdenticalGeometry(mass_geometry_, mass_points);
      diffusion_geometry_stride_ = ShareIdenticalGeometry(diffusion_geometry_, 3*diffusion_points);
    }
  }
  int ShareIdenticalGeometry(mfem::Vector& geometry, int entries) const
  {
    const double* values = geometry.HostRead();
    const std::size_t bytes = std::size_t(entries)*sizeof(double);
    for (int e = 1; e < ne_; ++e)
    {
      // Bitwise comparison preserves every quadrature value and signed zero.
      if (std::memcmp(values, values+e*entries, bytes) != 0) { return entries; }
    }
    // SetSize alone can retain the full allocation. Swap in a new one-tile
    // owner before either vector has acquired device storage.
    mfem::Vector shared(entries);
    std::copy_n(values, entries, shared.HostWrite());
    geometry.Swap(shared);
    return 0;
  }
  mfem::ParFiniteElementSpace& fes_;
  const std::string execution_;
  const bool collocated_requested_;
  const std::string storage_requested_;
  bool collocated_enabled_=false;
  const char* collocated_reason_="not-requested";
  bool direct_storage_=false;
  const char* storage_reason_="cached-requested";
  const mfem::QuadratureSpace *mass_space_, *diffusion_space_;
  const mfem::QuadratureFunction *invrho_=nullptr, *ck_=nullptr;
  int ne_=0,d_=0,qm_=0,qd_=0;
  int mass_geometry_stride_=tile_size, diffusion_geometry_stride_=3*tile_size;
  long space_sequence_=0,mesh_sequence_=0,nodes_sequence_=0;
  mfem::Array<int> dof_map_;
  mfem::Vector maps_,mass_geometry_,diffusion_geometry_,mass_,diffusion_;
  unsigned long long refreshes_=0;
  mutable unsigned long long applications_=0;
};
}

std::unique_ptr<mfem::Operator> MakeTensorVolumeOperator(
  mfem::ParFiniteElementSpace& sfes, const mfem::QuadratureFunction& invrho,
  const mfem::QuadratureFunction& ck, const std::string& execution)
{
  return std::make_unique<TensorVolumeOperator>(sfes,invrho,ck,execution);
}
void RefreshTensorVolumeCoefficients(mfem::Operator& op,
                                    const mfem::QuadratureFunction& invrho,
                                    const mfem::QuadratureFunction& ck)
{
  auto* tensor = dynamic_cast<TensorVolumeOperator*>(&op);
  MFEM_VERIFY(tensor, "Wrong operator passed to IGR tensor coefficient refresh.");
  tensor->Refresh(invrho,ck);
}
}
}

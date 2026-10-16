#include "igrlab/compat.hpp"
#include "numerics/shared_reference_map.hpp"
#include "utils/config/env_flags.hpp"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
#include <cuda_runtime.h>
#include <mma.h>
#endif

namespace igrlab
{
namespace shared_reference
{
Precision RequestedPrecision()
{
  const char* value = ::igrlab::GetEnv("DG_IGR_SHARED_OPERATOR_PRECISION");
  if (!value || std::strcmp(value, "off") == 0) { return Precision::Off; }
  if (std::strcmp(value, "fp64") == 0) { return Precision::FP64; }
  if (std::strcmp(value, "fp32") == 0) { return Precision::FP32; }
  if (std::strcmp(value, "tf32") == 0) { return Precision::TF32; }
  MFEM_ABORT("SHARED_OPERATOR_PRECISION must be off, fp64, fp32, or tf32.");
  return Precision::Off;
}

const char* PrecisionName(Precision precision)
{
  switch (precision)
  {
    case Precision::Off: return "off";
    case Precision::FP64: return "fp64";
    case Precision::FP32: return "fp32";
    case Precision::TF32: return "tf32";
  }
  return "invalid";
}

#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
namespace
{
template <typename Accumulator>
__global__ void SharedReferenceSIMTKernel(int m, int k, int n,
                                         const mfem::real_t* a, const mfem::real_t* b,
                                         mfem::real_t* c)
{
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= m * n) { return; }
  const int row = entry % m, column = entry / m;
  Accumulator sum = 0;
  for (int j = 0; j < k; ++j)
  {
    sum += static_cast<Accumulator>(a[row + m * j]) *
           static_cast<Accumulator>(b[j + k * column]);
  }
  c[entry] = static_cast<double>(sum);
}

// Four independent warps/CTA, one 16x16 output tile per warp. Both operands
// are column-major. Padding is transient shared memory, not artificial RHSs.
__global__ void SharedReferenceTF32Kernel(int m, int k, int n,
                                         const mfem::real_t* a, const mfem::real_t* b,
                                         mfem::real_t* c)
{
#if __CUDA_ARCH__ >= 800
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int tiles_m = (m - 1) / 16 + 1, tiles_n = (n - 1) / 16 + 1;
  const int tile = blockIdx.x * 4 + warp;
  if (tile >= tiles_m * tiles_n) { return; } // warp-uniform; no CTA barriers
  const int row0 = 16 * (tile % tiles_m), col0 = 16 * (tile / tiles_m);
  __shared__ __align__(32) float sa[4][128], sb[4][128], sc[4][256];
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                 wmma::col_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                 wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> fc;
  wmma::fill_fragment(fc, 0.0f);
  for (int k0 = 0; k0 < k; k0 += 8)
  {
    for (int i = lane; i < 128; i += 32)
    {
      const int ar = row0 + i % 16, ak = k0 + i / 16;
      const int bk = k0 + i % 8, bc = col0 + i / 8;
      sa[warp][i] = wmma::__float_to_tf32(
        ar < m && ak < k ? static_cast<float>(a[ar + m * ak]) : 0.0f);
      sb[warp][i] = wmma::__float_to_tf32(
        bk < k && bc < n ? static_cast<float>(b[bk + k * bc]) : 0.0f);
    }
    __syncwarp();
    wmma::load_matrix_sync(fa, sa[warp], 16);
    wmma::load_matrix_sync(fb, sb[warp], 8);
    wmma::mma_sync(fc, fa, fb, fc);
    __syncwarp(); // all lanes have consumed this panel before reuse
  }
  wmma::store_matrix_sync(sc[warp], fc, 16, wmma::mem_col_major);
  __syncwarp();
  for (int i = lane; i < 256; i += 32)
  {
    const int row = row0 + i % 16, column = col0 + i / 16;
    if (row < m && column < n) { c[row + m * column] = sc[warp][i]; }
  }
#else
  asm("trap;"); // Configure rejects hardware below SM80; never silent no-op.
#endif
}
} // namespace
#endif

void Map::Configure(const mfem::DenseMatrix& matrix, int columns,
                    Precision precision, const char* method)
{
  MFEM_VERIFY(!Ready() && precision != Precision::Off &&
                sizeof(mfem::real_t) == sizeof(double),
              "Shared reference map requires one FP64-state configuration.");
  const int m = matrix.Height(), k = matrix.Width();
  MFEM_VERIFY(m > 0 && k > 0 && columns > 0 && m <= 256 && k <= 512 &&
                columns <= (std::numeric_limits<int>::max() - 255) / std::max(m, k),
              "Shared reference map exceeds its bounded matrix/index layout.");
  MFEM_VERIFY(mfem::Device::Allows(mfem::Backend::CUDA_MASK),
              "Explicit shared reference maps require CUDA.");
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  int device = 0;
  cudaDeviceProp properties{};
  MFEM_VERIFY(cudaGetDevice(&device) == cudaSuccess &&
                cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
                (precision != Precision::TF32 || properties.major >= 8),
              "TF32 shared reference maps require native SM80 or newer.");
  const mfem::real_t* values = matrix.HostRead();
  for (int i = 0; i < m * k; ++i)
  {
    MFEM_VERIFY(std::isfinite(values[i]) &&
                  (precision == Precision::FP64 ||
                   std::isfinite(static_cast<float>(values[i]))),
                "Shared reference matrix is outside the selected finite range.");
  }
  matrix_ = matrix;
  matrix_.GetMemory().UseDevice(true);
  input_.SetSize(k, columns); input_.GetMemory().UseDevice(true);
  output_.SetSize(m, columns); output_.GetMemory().UseDevice(true);
  // Materialize all device buffers during setup. Subsequent Apply does not resize.
  matrix_.Read(); input_.Write(); output_.Write();
  precision_ = precision; method_ = method;
  std::cout << "[shared-reference-map] event=setup method=" << method_
            << " precision=" << PrecisionName(precision_)
            << " m=" << m << " k=" << k << " real_columns=" << columns
            << " backend=" << (precision_ == Precision::TF32 ? "wmma-m16n16k8" : "simt")
            << " state=fp64 nonlinear_flux=fp64 faces=fp64 centering=fp64-anchor"
            << " matrix_shared=1 fallback=none workspace_bytes="
            << (std::size_t(m) * k + std::size_t(m + k) * columns) * sizeof(double)
            << " tensor_proof=requires-profile" << std::endl;
#else
  MFEM_ABORT("Shared reference maps require a CUDA-compiled application.");
#endif
}

void Map::Apply()
{
  MFEM_VERIFY(Ready(), "Shared reference map was not configured.");
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  const int m = matrix_.Height(), k = matrix_.Width(), n = input_.Width();
  const mfem::real_t* a = matrix_.Read(); const mfem::real_t* b = input_.Read();
  mfem::real_t* c = output_.Write();
  if (precision_ == Precision::TF32)
  {
    const int tiles = ((m - 1) / 16 + 1) * ((n - 1) / 16 + 1);
    SharedReferenceTF32Kernel<<<(tiles + 3) / 4, 128>>>(m, k, n, a, b, c);
  }
  else if (precision_ == Precision::FP32)
  {
    SharedReferenceSIMTKernel<float><<<(m * n - 1) / 256 + 1, 256>>>(m, k, n, a, b, c);
  }
  else
  {
    SharedReferenceSIMTKernel<double><<<(m * n - 1) / 256 + 1, 256>>>(m, k, n, a, b, c);
  }
  MFEM_VERIFY(cudaGetLastError() == cudaSuccess, "Shared reference map launch failed.");
  ++applications_;
#else
  MFEM_ABORT("Shared reference map apply requires CUDA.");
#endif
}

Map::~Map()
{
  if (Ready())
  {
    std::cout << "[shared-reference-map] event=summary method=" << method_
              << " precision=" << PrecisionName(precision_)
              << " submitted_applications=" << applications_
              << " fallback=none" << std::endl;
  }
}
} // namespace shared_reference
} // namespace igrlab

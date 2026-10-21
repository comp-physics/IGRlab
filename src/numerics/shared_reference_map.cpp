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
namespace
{
// Frozen source variants qualify launch width and shared-field padding without
// adding runtime policy or changing the conservative map's numerical contract.
constexpr int kDGConservativeFusionThreads = 128;
constexpr int kDGConservativeFusionSharedPitch = 72;
static_assert(kDGConservativeFusionThreads == 64 || kDGConservativeFusionThreads == 128,
              "Conservative fusion tuning supports 64 or 128 threads.");
static_assert(kDGConservativeFusionSharedPitch == 72 || kDGConservativeFusionSharedPitch == 73,
              "Conservative fusion tuning supports shared pitch 72 or 73.");
}

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

template <typename Accumulator>
__global__ void DGSharedTensorLinesSIMTKernel(
  int n, int columns, const int* local, const double* derivative,
  const double* weights, const double* adjugate, const double* input,
  double* output)
{
  const int dof = n*n, entry = blockIdx.x*blockDim.x+threadIdx.x;
  if (entry >= dof*columns) { return; }
  const int node = entry%dof, column = entry/dof, i = node%n, j = node/n;
  const int offset = 2*dof*column;
  Accumulator x = 0, y = 0;
  for (int k = 0; k < n; ++k)
  {
    const int xn = 2*local[k+n*j]+offset, yn = 2*local[i+n*k]+offset;
    // Transform the original centered physical flux in FP64 before the
    // chosen contraction precision. Nonlinear flux/anchoring stay upstream.
    const double bx = adjugate[0]*input[xn]+adjugate[2]*input[xn+1];
    const double by = adjugate[1]*input[yn]+adjugate[3]*input[yn+1];
    x += static_cast<Accumulator>(derivative[i+n*k])*static_cast<Accumulator>(bx);
    y += static_cast<Accumulator>(derivative[j+n*k])*static_cast<Accumulator>(by);
  }
  output[local[node]+dof*column] = weights[j]*static_cast<double>(x)+
                                weights[i]*static_cast<double>(y);
}

// One CTA owns one cell and all four physical equation columns. The native
// row order in the serial FP64 moment sums is part of the existing method.
__global__ void DGSharedConservativeFusionP5FP32Kernel(
  const int* local, const double* derivative, const double* weights,
  const double* adjugate, const double* flux, const double* column_sums,
  const double* row_sums, const double* mass_fractions, const int* state_map,
  double* residual)
{
  constexpr int n = 6, dof = n*n, equations = 4, rows = 2*dof;
  constexpr int pitch = kDGConservativeFusionSharedPitch;
  const int e = blockIdx.x, lane = threadIdx.x;
  __shared__ double centered[equations*pitch], output[equations*dof];
  __shared__ double anchors[2*equations], correction[equations];
  for (int entry = lane; entry < equations*rows; entry += blockDim.x)
    centered[(entry/rows)*pitch+entry%rows] = flux[e*equations*rows+entry];
  if (lane < 2*equations)
    anchors[lane] = flux[e*equations*rows+rows*(lane/2)+lane%2];
  __syncthreads();
  for (int entry = lane; entry < equations*rows; entry += blockDim.x)
  {
    const int shared_entry = (entry/rows)*pitch+entry%rows;
    centered[shared_entry] = centered[shared_entry]-anchors[2*(entry/rows)+entry%2];
  }
  __syncthreads();
  for (int entry = lane; entry < equations*dof; entry += blockDim.x)
  {
    const int node = entry%dof, equation = entry/dof, i = node%n, j = node/n;
    const int offset = pitch*equation;
    float x = 0, y = 0;
    // Preserve the buffered SIMT expressions, cast points, k order and FMA
    // opportunities. Do not drop zero metric products on nonfinite inputs.
    for (int k = 0; k < n; ++k)
    {
      const int xn = 2*local[k+n*j]+offset, yn = 2*local[i+n*k]+offset;
      const double bx = adjugate[0]*centered[xn]+adjugate[2]*centered[xn+1];
      const double by = adjugate[1]*centered[yn]+adjugate[3]*centered[yn+1];
      x += static_cast<float>(derivative[i+n*k])*static_cast<float>(bx);
      y += static_cast<float>(derivative[j+n*k])*static_cast<float>(by);
    }
    output[local[node]+dof*equation] = weights[j]*static_cast<double>(x)+
                                     weights[i]*static_cast<double>(y);
  }
  __syncthreads();
  if (lane < equations)
  {
    double observed = 0.0, expected = 0.0;
    for (int i = 0; i < dof; ++i) { observed += output[i+dof*lane]; }
    for (int j = 0; j < rows; ++j)
      expected += column_sums[j]*centered[j+pitch*lane];
    correction[lane] = expected-observed;
  }
  __syncthreads();
  for (int entry = lane; entry < equations*dof; entry += blockDim.x)
  {
    const int i = entry%dof, equation = entry/dof;
    double value = output[entry]+mass_fractions[e*dof+i]*correction[equation];
    for (int d = 0; d < 2; ++d)
      value += row_sums[i+dof*d]*anchors[2*equation+d];
    residual[state_map[(e*dof+i)*equations+equation]] += value;
  }
}

// A warp owns 16 distinct element/field columns. Each transverse line is a
// genuine RHS of the same 1D weak derivative; both directions accumulate on
// chip. Padding supplies only missing rows/columns, never replicated states.
template <int N>
__global__ void DGSharedTensorLinesTF32Kernel(
  int columns, const int* local, const double* derivative,
  const double* weights, const double* adjugate, const double* input,
  double* output)
{
#if __CUDA_ARCH__ >= 800
  constexpr int dof = N*N;
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  const int first_column = 16*(4*blockIdx.x+warp);
  if (first_column >= columns) { return; } // no block-wide synchronization
  __shared__ __align__(32) float sa[4][8*24], sb[4][16*16], sc[4][16*24];
  __shared__ double result[4][16*dof];
  for (int entry = lane; entry < 16*dof; entry += 32) { result[warp][entry] = 0.0; }
  for (int entry = lane; entry < 8*24; entry += 32)
  {
    if (entry%24 < 16)
    {
      const int row = entry%24, inner = entry/24;
      sa[warp][entry] = nvcuda::wmma::__float_to_tf32(
        row < N && inner < N ? static_cast<float>(derivative[row+N*inner]) : 0.0f);
    }
  }
  __syncwarp();
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> a;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> b;
  wmma::fragment<wmma::accumulator,16,16,8,float> c;
  wmma::load_matrix_sync(a, sa[warp], 24);
  for (int direction = 0; direction < 2; ++direction)
  {
    for (int fixed = 0; fixed < N; ++fixed)
    {
      for (int entry = lane; entry < 16*16; entry += 32)
      {
        if (entry%16 < 8)
        {
          const int inner = entry%16, column = first_column+entry/16;
          double value = 0.0;
          if (inner < N && column < columns)
          {
            const int lex = direction == 0 ? inner+N*fixed : fixed+N*inner;
            const int offset = 2*(local[lex]+dof*column);
            value = adjugate[direction]*input[offset]+
                    adjugate[direction+2]*input[offset+1];
          }
          sb[warp][entry] = wmma::__float_to_tf32(static_cast<float>(value));
        }
      }
      __syncwarp();
      wmma::load_matrix_sync(b, sb[warp], 16);
      wmma::fill_fragment(c, 0.0f);
      wmma::mma_sync(c, a, b, c);
      wmma::store_matrix_sync(sc[warp], c, 24, wmma::mem_col_major);
      __syncwarp();
      for (int entry = lane; entry < 16*24; entry += 32)
      {
        const int coordinate = entry%24, column = entry/24;
        if (coordinate < N)
        {
          const int node = direction == 0 ? coordinate+N*fixed : fixed+N*coordinate;
          result[warp][node+dof*column] += weights[fixed]*static_cast<double>(sc[warp][entry]);
        }
      }
      __syncwarp();
    }
  }
  for (int entry = lane; entry < 16*dof; entry += 32)
  {
    const int column = first_column+entry/dof;
    if (column < columns) { output[local[entry%dof]+dof*column] = result[warp][entry]; }
  }
#else
  asm("trap;");
#endif
}

void ApplyDGSharedTensorLinesTF32(
  int n, int columns, const int* local, const double* derivative,
  const double* weights, const double* adjugate, const double* input,
  double* output)
{
  const int blocks = (columns+63)/64;
#define DG_SHARED_LINES_CASE(N) case N: DGSharedTensorLinesTF32Kernel<N><<<blocks,128>>>(columns,local,derivative,weights,adjugate,input,output); break
  switch (n)
  {
    DG_SHARED_LINES_CASE(2);
    DG_SHARED_LINES_CASE(3);
    DG_SHARED_LINES_CASE(4);
    DG_SHARED_LINES_CASE(5);
    DG_SHARED_LINES_CASE(6);
    DG_SHARED_LINES_CASE(7);
    DG_SHARED_LINES_CASE(8);
    default: MFEM_ABORT("Unsupported DG tensor-line extent.");
  }
#undef DG_SHARED_LINES_CASE
}
} // namespace
#endif

void Map::Configure(const mfem::DenseMatrix& matrix, int columns,
                    Precision precision, const char* method)
{
  ConfigureStorage(matrix, columns, precision, method);
  LogConfiguration();
}

void Map::ConfigureStorage(const mfem::DenseMatrix& matrix, int columns,
                          Precision precision, const char* method, bool buffered)
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
  columns_ = columns;
  if (buffered)
  {
    input_.SetSize(k, columns); input_.GetMemory().UseDevice(true);
    output_.SetSize(m, columns); output_.GetMemory().UseDevice(true);
    input_.Write(); output_.Write();
  }
  // Materialize all device buffers during setup. Subsequent Apply does not resize.
  matrix_.Read();
  conservative_fusion_ = !buffered;
  precision_ = precision; method_ = method;
#else
  MFEM_ABORT("Shared reference maps require a CUDA-compiled application.");
#endif
}

void Map::ConfigureDGTensorLines(const mfem::DenseMatrix& matrix, int columns,
                               Precision precision, const mfem::FiniteElement& fe,
                               const mfem::DenseMatrix& adjugate,
                               bool conservative_fusion)
{
  const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(&fe);
  const int n = fe.GetOrder()+1, dof = n*n;
  MFEM_VERIFY(tensor && tensor->GetBasisType() == mfem::BasisType::GaussLegendre &&
                fe.GetGeomType() == mfem::Geometry::SQUARE &&
                fe.GetMapType() == mfem::FiniteElement::VALUE &&
                n >= 2 && n <= 8 && fe.GetDof() == dof &&
                matrix.Height() == dof && matrix.Width() == 2*dof &&
                adjugate.Height() == 2 && adjugate.Width() == 2,
              "DG tensor lines require affine Gauss-Legendre VALUE quads with p1 through p7.");
  tensor_lex_to_local_.SetSize(dof);
  int* local = tensor_lex_to_local_.HostWrite();
  const auto& lex = tensor->GetDofMap();
  MFEM_VERIFY(lex.Size() == 0 || lex.Size() == dof, "Invalid DG tensor-line dof map.");
  for (int i = 0; i < dof; ++i)
  {
    local[i] = lex.Size() ? lex[i] : i;
    MFEM_VERIFY(local[i] >= 0 && local[i] < dof, "Invalid DG tensor-line dof index.");
    for (int j = 0; j < i; ++j)
      MFEM_VERIFY(local[i] != local[j], "DG tensor-line dof map is not a permutation.");
  }
  mfem::L2_FECollection line_collection(fe.GetOrder(), 1, mfem::BasisType::GaussLegendre);
  const auto& line = *line_collection.FiniteElementForGeometry(mfem::Geometry::SEGMENT);
  const auto& rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2*fe.GetOrder()+1);
  MFEM_VERIFY(rule.GetNPoints() == n && fe.GetNodes().GetNPoints() == dof,
              "DG tensor lines require the original Gauss-Legendre nodal grid.");
  tensor_derivative_.SetSize(dof); tensor_weights_.SetSize(n); tensor_adjugate_.SetSize(4);
  double* derivative = tensor_derivative_.HostWrite();
  double* weights = tensor_weights_.HostWrite(); double* metric = tensor_adjugate_.HostWrite();
  mfem::DenseMatrix dshape(n,1);
  for (int j = 0; j < n; ++j)
  {
    weights[j] = rule.IntPoint(j).weight;
    MFEM_VERIFY(std::isfinite(weights[j]) && weights[j] > 0.0,
                "Invalid DG tensor-line quadrature weight.");
    line.CalcDShape(rule.IntPoint(j), dshape);
    for (int i = 0; i < n; ++i) { derivative[i+n*j] = weights[j]*dshape(i,0); }
  }
  for (int d = 0; d < 2; ++d) for (int direction = 0; direction < 2; ++direction)
  {
    metric[direction+2*d] = adjugate(direction,d);
    MFEM_VERIFY(std::isfinite(metric[direction+2*d]), "Nonfinite DG tensor-line metric.");
  }
  const double node_tolerance = 64*std::numeric_limits<double>::epsilon();
  for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
  {
    const auto& point = fe.GetNodes().IntPoint(local[i+n*j]);
    MFEM_VERIFY(std::abs(point.x-rule.IntPoint(i).x) <= node_tolerance &&
                  std::abs(point.y-rule.IntPoint(j).x) <= node_tolerance,
                "DG tensor-line local ordering does not match its nodal grid.");
  }
  double norm = 0.0, difference = 0.0;
  for (int row = 0; row < dof; ++row)
  {
    const int i = row%n, j = row/n;
    double row_norm = 0.0, row_difference = 0.0;
    for (int column = 0; column < dof; ++column) for (int d = 0; d < 2; ++d)
    {
      const int k = column%n, l = column/n;
      const double reconstructed = (j == l ? weights[j]*derivative[i+n*k]*metric[2*d] : 0.0)+
                                   (i == k ? weights[i]*derivative[j+n*l]*metric[1+2*d] : 0.0);
      const double original = matrix(local[row],2*local[column]+d);
      MFEM_VERIFY(std::isfinite(original) && std::isfinite(reconstructed),
                  "Nonfinite DG tensor-line operator reconstruction.");
      row_norm += std::abs(original); row_difference += std::abs(original-reconstructed);
    }
    norm = std::max(norm,row_norm); difference = std::max(difference,row_difference);
  }
  const double limit = 64*dof*std::numeric_limits<double>::epsilon();
  MFEM_VERIFY(norm > 0.0 && difference <= limit*norm,
              "DG tensor lines disagree with the original assembled weak divergence: relative="
              << difference/norm << " limit=" << limit);
  const bool fusion_admitted = conservative_fusion && n == 6 &&
    precision == Precision::FP32 && columns % 4 == 0;
  ConfigureStorage(matrix, columns, precision, "dg-weakdiv", !fusion_admitted);
  tensor_n1d_ = n;
  tensor_lex_to_local_.GetMemory().UseDevice(true); tensor_derivative_.UseDevice(true);
  tensor_weights_.UseDevice(true); tensor_adjugate_.UseDevice(true);
  tensor_lex_to_local_.Read(); tensor_derivative_.Read(); tensor_weights_.Read(); tensor_adjugate_.Read();
  LogConfiguration();
  std::cout << "[shared-reference-map] event=tensor-factorization method=dg-weakdiv"
            << " n1d=" << n << " matrix_relative_inf_error=" << difference/norm
            << " limit=" << limit << " transverse_mass=gauss-diagonal"
            << " geometry=fp64-adjugate real_columns=" << columns << std::endl;
}

void Map::LogConfiguration() const
{
  const int m = matrix_.Height(), k = matrix_.Width(), columns = columns_;
  std::cout << "[shared-reference-map] event=setup method=" << method_
            << " precision=" << PrecisionName(precision_)
            << " m=" << m << " k=" << k << " real_columns=" << columns
            << " layout=" << (UsesDGTensorLines() ? "tensor-lines" : "dense")
            << " backend=" << (precision_ == Precision::TF32 ? "wmma-m16n16k8" : "simt")
            << " state=fp64 nonlinear_flux=fp64 faces=fp64 centering=fp64-anchor"
            << " matrix_shared=1 fallback=none workspace_bytes="
            << (std::size_t(m) * k + std::size_t(input_.Height()) * input_.Width()+
                std::size_t(output_.Height()) * output_.Width()+
                tensor_derivative_.Size()+tensor_weights_.Size()+tensor_adjugate_.Size()) * sizeof(double)+
               std::size_t(tensor_lex_to_local_.Size())*sizeof(int)
            << " conservative_fusion=" << int(conservative_fusion_)
            << " conservative_fusion_threads=" << kDGConservativeFusionThreads
            << " conservative_fusion_shared_pitch=" << kDGConservativeFusionSharedPitch
            << " tensor_proof=requires-profile" << std::endl;
}

void Map::Apply()
{
  MFEM_VERIFY(Ready(), "Shared reference map was not configured.");
  MFEM_VERIFY(!conservative_fusion_,
              "Fused DG map requires its conservative flux/residual apply.");
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  const int m = matrix_.Height(), k = matrix_.Width(), n = columns_;
  const mfem::real_t* a = matrix_.Read(); const mfem::real_t* b = input_.Read();
  mfem::real_t* c = output_.Write();
  if (UsesDGTensorLines())
  {
    const int* local = tensor_lex_to_local_.Read();
    const double* derivative = tensor_derivative_.Read();
    const double* weights = tensor_weights_.Read();
    const double* metric = tensor_adjugate_.Read();
    if (precision_ == Precision::TF32)
      ApplyDGSharedTensorLinesTF32(tensor_n1d_,n,local,derivative,weights,metric,b,c);
    else if (precision_ == Precision::FP32)
      DGSharedTensorLinesSIMTKernel<float><<<(m*n+255)/256,256>>>(tensor_n1d_,n,local,derivative,weights,metric,b,c);
    else
      DGSharedTensorLinesSIMTKernel<double><<<(m*n+255)/256,256>>>(tensor_n1d_,n,local,derivative,weights,metric,b,c);
  }
  else if (precision_ == Precision::TF32)
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

void Map::ApplyDGConservativeFusion(const mfem::Vector& flux,
                                  const mfem::Vector& column_sums,
                                  const mfem::DenseMatrix& row_sums,
                                  const mfem::Vector& mass_fractions,
                                  const mfem::Array<int>& state_map,
                                  mfem::Vector& residual)
{
  MFEM_VERIFY(Ready() && conservative_fusion_ && tensor_n1d_ == 6 &&
                precision_ == Precision::FP32 && columns_ % 4 == 0,
              "Conservative DG fusion requires its configured p5 FP32 tensor map.");
  const int elements = columns_/4, dof = 36;
  MFEM_VERIFY(flux.Size() == 2*dof*columns_ && column_sums.Size() == 2*dof &&
                row_sums.Height() == dof && row_sums.Width() == 2 &&
                mass_fractions.Size() == elements*dof &&
                state_map.Size() == elements*dof*4 &&
                residual.Size() == elements*dof*4,
              "Conservative DG fusion received incompatible packed arrays.");
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  DGSharedConservativeFusionP5FP32Kernel<<<elements,kDGConservativeFusionThreads>>>(
    tensor_lex_to_local_.Read(), tensor_derivative_.Read(), tensor_weights_.Read(),
    tensor_adjugate_.Read(), flux.Read(), column_sums.Read(), row_sums.Read(),
    mass_fractions.Read(), state_map.Read(), residual.ReadWrite());
  MFEM_VERIFY(cudaGetLastError() == cudaSuccess, "Conservative DG fusion launch failed.");
  ++applications_;
#else
  MFEM_ABORT("Conservative DG fusion requires a CUDA-compiled application.");
#endif
}

Map::~Map()
{
  if (Ready())
  {
    std::cout << "[shared-reference-map] event=summary method=" << method_
              << " precision=" << PrecisionName(precision_)
              << " layout=" << (UsesDGTensorLines() ? "tensor-lines" : "dense")
              << " submitted_applications=" << applications_
              << " fallback=none" << std::endl;
  }
}
} // namespace shared_reference
} // namespace igrlab

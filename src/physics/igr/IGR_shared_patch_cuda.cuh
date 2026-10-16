#pragma once

#include "IGR_two_level_inverse.hpp"
#include "mfem.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace igrlab { namespace igr_internal {
namespace shared_patch_internal {
inline void CUDA(cudaError_t code)
{ MFEM_VERIFY(code == cudaSuccess, "Shared patch CUDA: " << cudaGetErrorString(code)); }
inline void BLAS(cublasStatus_t code)
{ MFEM_VERIFY(code == CUBLAS_STATUS_SUCCESS, "Shared patch cuBLAS status=" << code); }

template <typename Scalar>
static __global__ void PackColumns(int k, int padded_k, const int* inputs,
    const double* x, Scalar* packed, int* exponents, int* invalid)
{
  const int col = blockIdx.x;
  __shared__ double largest[128];
  double maximum = 0.0;
  bool bad = false;
  for (int j = threadIdx.x; j < k; j += blockDim.x)
  {
    const double value = x[inputs[col * k + j]];
    bad = bad || !isfinite(value);
    maximum = fmax(maximum, fabs(value));
  }
  const bool any_bad = __syncthreads_or(bad);
  largest[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = 64; stride; stride /= 2)
  {
    if (threadIdx.x < stride) largest[threadIdx.x] = fmax(largest[threadIdx.x], largest[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    int exponent = 0;
    if (!any_bad) frexp(largest[0], &exponent);
    exponents[col] = exponent;
    invalid[col] = any_bad;
  }
  __syncthreads();
  for (int j = threadIdx.x; j < padded_k; j += blockDim.x)
    packed[col * padded_k + j] = j < k && !any_bad
        ? static_cast<Scalar>(scalbn(x[inputs[col * k + j]], -exponents[col])) : Scalar(0);
}

template <typename Scalar>
static __global__ void UnpackColumns(int m, int padded_m, int columns,
    const int* outputs, const double* coarse, const Scalar* result,
    const int* rhs_exponents, const int* matrix_exponents, const int* invalid,
    const double* column_scales, double* y)
{
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= m * columns) return;
  const int col = entry / m, row = entry % m, target = outputs[entry];
  const double value = scalbn(static_cast<double>(result[col * padded_m + row]),
                              rhs_exponents[col] + matrix_exponents[col]);
  y[target] = invalid[col] ? nan("") : coarse[target] + column_scales[col] * value;
}
// Every output row is owned by exactly one L2 element. A0 has the same
// column-major five-block layout as the independently certified setup sample.
static __global__ void FrozenPatchResidualKernel(int dof, int height,
    const int* inputs, const int* outputs, const double* stencil,
    const double* rhs, const double* coarse, double* residual)
{
  const int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= height) return;
  const int element = entry / dof, row = entry % dof, k = 5*dof;
  double action = 0.0;
  for (int col = 0; col < k; ++col)
    action += stencil[(static_cast<std::size_t>(element)*k+col)*dof+row] *
              coarse[inputs[element*k+col]];
  const int target = outputs[entry];
  residual[target] = rhs[target] - action;
}
} // namespace shared_patch_internal

// Retained only by explicit multiplicative composition. These are the actual
// sampled FP64 blocks, before representative grouping or inverse scaling.
class FrozenPatchResidual
{
public:
  void Configure(int dof, const std::vector<std::array<int,5>>& patch,
      const mfem::Array<int>& element_map, const std::vector<double>& stencil)
  {
    const int ne = static_cast<int>(patch.size()), k = 5*dof;
    MFEM_VERIFY(dof > 0 && ne > 0 && height_ == 0 && element_map.Size() == ne*dof &&
        stencil.size() == std::size_t(ne)*k*dof && stencil.size() <= std::numeric_limits<int>::max(),
        "Invalid frozen shared-patch operator dimensions.");
    dof_ = dof; height_ = ne*dof;
    inputs_.SetSize(ne*k); outputs_.SetSize(height_);
    coefficients_.SetSize(static_cast<int>(stencil.size())); coefficients_.UseDevice(true);
    int* input = inputs_.HostWrite(); int* output = outputs_.HostWrite();
    const int* map = element_map.HostRead();
    std::vector<char> covered(height_,0);
    for (int e = 0; e < ne; ++e)
    {
      for (int row = 0; row < dof; ++row)
      {
        const int target = map[e*dof+row];
        MFEM_VERIFY(target >= 0 && target < height_ && !covered[target],
                    "Frozen shared-patch map must cover disjoint L2 outputs.");
        covered[target] = 1; output[e*dof+row] = target;
      }
      for (int slot = 0; slot < 5; ++slot)
      {
        MFEM_VERIFY(patch[e][slot] >= 0 && patch[e][slot] < ne,
                    "Invalid frozen shared-patch neighbor.");
        for (int col = 0; col < dof; ++col)
          input[e*k+slot*dof+col] = map[patch[e][slot]*dof+col];
      }
    }
    for (double value : stencil) MFEM_VERIFY(std::isfinite(value), "Nonfinite frozen shared-patch coefficient.");
    std::copy(stencil.begin(),stencil.end(),coefficients_.HostWrite());
    (void)inputs_.Read(); (void)outputs_.Read(); (void)coefficients_.Read();
  }
  void Mult(const mfem::Vector& rhs, const mfem::Vector& coarse, mfem::Vector& residual) const
  {
    MFEM_VERIFY(height_ > 0 && rhs.Size() == height_ && coarse.Size() == height_ &&
        &rhs != &residual && &coarse != &residual, "Invalid frozen shared-patch residual application.");
    residual.SetSize(height_); residual.UseDevice(true);
    shared_patch_internal::FrozenPatchResidualKernel<<<(height_+255)/256,256>>>(
        dof_,height_,inputs_.Read(),outputs_.Read(),coefficients_.Read(),
        rhs.Read(),coarse.Read(),residual.Write());
    shared_patch_internal::CUDA(cudaGetLastError());
  }
  std::size_t StorageBytes() const
  { return std::size_t(coefficients_.Size())*sizeof(double) +
           std::size_t(inputs_.Size()+outputs_.Size())*sizeof(int); }
private:
  int dof_=0,height_=0;
  mfem::Array<int> inputs_,outputs_;
  mfem::Vector coefficients_;
};

// One shared matrix per group, one genuinely different patch residual per
// column. This private algebra object is also exercised directly by the fixture.
class SharedPatchBatch
{
public:
  ~SharedPatchBatch()
  {
    if (handle_)
    {
      shared_patch_internal::CUDA(cudaStreamSynchronize(nullptr));
      std::cout << "[igr-shared-patch] event=summary arithmetic=" << (tf32_ ? "tf32" : "fp64")
                << " applications=" << applications_ << " gemm_calls=" << applications_ * groups_
                << " real_rhs_columns=" << applications_ * columns_
                << " composition=" << composition_
                << " fp64_stencil_actions=" << (composition_ == "multiplicative" ? applications_ : 0)
                << " completed=true fallback=none\n";
      cublasDestroy(handle_);
    }
  }
  void Configure(int m, int k, int height, const std::vector<double>& matrices,
      const std::vector<int>& offsets, const std::vector<int>& inputs,
      const std::vector<int>& outputs, const std::vector<double>& scales, bool tf32,
      const std::string& composition = "additive")
  {
    MFEM_VERIFY(composition == "additive" || composition == "multiplicative", "Invalid shared-patch composition.");
    composition_ = composition;
    MFEM_VERIFY(!handle_ && m > 0 && k >= m && offsets.size() >= 2 && offsets.front() == 0,
                "Invalid shared patch batch dimensions.");
    m_ = m; k_ = k; height_ = height; tf32_ = tf32;
    groups_ = static_cast<int>(offsets.size()) - 1; columns_ = offsets.back();
    pm_ = ((m + 15) / 16) * 16; pk_ = ((k + 7) / 8) * 8;
    MFEM_VERIFY(columns_ > 0 && std::size_t(groups_) * m * k == matrices.size() &&
        inputs.size() == std::size_t(k) * columns_ && outputs.size() == std::size_t(m) * columns_ &&
        scales.size() == std::size_t(columns_) && std::size_t(pk_) * columns_ <= (1u << 27),
        "Invalid/bounded shared patch batch storage.");
    std::vector<char> covered(height, 0);
    for (int row : inputs) MFEM_VERIFY(row >= 0 && row < height, "Invalid shared patch input map.");
    for (int row : outputs)
    {
      MFEM_VERIFY(row >= 0 && row < height && !covered[row], "Shared patch outputs must be disjoint.");
      covered[row] = 1;
    }
    MFEM_VERIFY(std::all_of(covered.begin(), covered.end(), [](char x) { return x != 0; }),
                "Shared patch must cover every output exactly once.");
    offsets_ = offsets;
    input_map_.SetSize(static_cast<int>(inputs.size())); output_map_.SetSize(static_cast<int>(outputs.size()));
    std::copy(inputs.begin(), inputs.end(), input_map_.HostWrite());
    std::copy(outputs.begin(), outputs.end(), output_map_.HostWrite());
    column_scales_.SetSize(columns_); matrix_exponents_.SetSize(columns_);
    rhs_exponents_.SetSize(columns_); invalid_.SetSize(columns_);
    std::copy(scales.begin(), scales.end(), column_scales_.HostWrite());
    double* scale = column_scales_.HostReadWrite();
    for (int col = 0; col < columns_; ++col)
      MFEM_VERIFY(scale[col] > 0.0 && std::isfinite(scale[col]), "Invalid shared patch scaling.");
    if (tf32_)
    {
      int device; cudaDeviceProp properties{};
      shared_patch_internal::CUDA(cudaGetDevice(&device));
      shared_patch_internal::CUDA(cudaGetDeviceProperties(&properties, device));
      MFEM_VERIFY(properties.major >= 8, "TF32 shared patches require SM80 or newer.");
      const char* override_value = std::getenv("NVIDIA_TF32_OVERRIDE");
      MFEM_VERIFY(!override_value || std::string(override_value) != "0",
                  "NVIDIA_TF32_OVERRIDE=0 disables requested shared-patch TF32.");
      matrix32_.SetSize(groups_ * pm_ * pk_); packed32_.SetSize(columns_ * pk_); result32_.SetSize(columns_ * pm_);
    }
    else
    {
      matrix64_.SetSize(groups_ * pm_ * pk_); packed64_.SetSize(columns_ * pk_); result64_.SetSize(columns_ * pm_);
      matrix64_.UseDevice(true); packed64_.UseDevice(true); result64_.UseDevice(true);
    }
    float* f = tf32_ ? matrix32_.HostWrite() : nullptr;
    double* d = tf32_ ? nullptr : matrix64_.HostWrite();
    int* exponents = matrix_exponents_.HostWrite();
    for (int group = 0; group < groups_; ++group)
    {
      MFEM_VERIFY(offsets[group + 1] > offsets[group], "Shared patch groups must be nonempty.");
      double maximum = 0.0;
      for (int i = 0; i < m * k; ++i)
      {
        const double value = matrices[group * m * k + i];
        MFEM_VERIFY(std::isfinite(value), "Nonfinite shared patch inverse.");
        maximum = std::max(maximum, std::abs(value));
      }
      MFEM_VERIFY(maximum > 0.0, "Zero shared patch inverse.");
      int exponent = 0; std::frexp(maximum, &exponent);
      for (int col = offsets[group]; col < offsets[group + 1]; ++col) exponents[col] = exponent;
      for (int col = 0; col < pk_; ++col) for (int row = 0; row < pm_; ++row)
      {
        const double value = row < m && col < k ? std::scalbn(matrices[group * m * k + col * m + row], -exponent) : 0.0;
        const int index = (group * pk_ + col) * pm_ + row;
        if (tf32_) f[index] = static_cast<float>(value); else d[index] = value;
      }
    }
    shared_patch_internal::BLAS(cublasCreate(&handle_));
    shared_patch_internal::BLAS(cublasSetStream(handle_, nullptr));
    shared_patch_internal::BLAS(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
    shared_patch_internal::BLAS(cublasSetMathMode(handle_, tf32_ ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
    (void)input_map_.Read(); (void)output_map_.Read(); (void)column_scales_.Read(); (void)matrix_exponents_.Read();
    if (tf32_) (void)matrix32_.Read(); else (void)matrix64_.Read();
  }
  void ConfigureFrozenOperator(int dof, const std::vector<std::array<int,5>>& patch,
      const mfem::Array<int>& element_map, const std::vector<double>& stencil)
  {
    MFEM_VERIFY(handle_ && composition_ == "multiplicative" && dof == m_ &&
        5*dof == k_ && element_map.Size() == height_, "Frozen operator requires multiplicative shared patches.");
    frozen_.Configure(dof,patch,element_map,stencil);
  }
  std::size_t FrozenOperatorBytes() const { return frozen_.StorageBytes(); }
  void Mult(const mfem::Vector& x, const mfem::Vector& coarse, mfem::Vector& y) const
  {
    MFEM_VERIFY(handle_ && x.Size() == height_ && coarse.Size() == height_ && &x != &y && &coarse != &y,
                "Invalid shared patch application.");
    y.SetSize(height_); y.UseDevice(true);
    const mfem::Vector* rhs = &x;
    if (composition_ == "multiplicative")
    {
      frozen_.Mult(x,coarse,residual_);
      rhs = &residual_;
    }
    int* exponents = rhs_exponents_.Write(); int* invalid = invalid_.Write();
    if (tf32_)
    {
      float* packed = packed32_.Write(); float* result = result32_.Write();
      shared_patch_internal::PackColumns<<<columns_,128>>>(k_, pk_, input_map_.Read(), rhs->Read(), packed, exponents, invalid);
      const float one = 1.0f, zero = 0.0f;
      for (int group = 0; group < groups_; ++group)
        shared_patch_internal::BLAS(cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            pm_, offsets_[group+1]-offsets_[group], pk_, &one,
            matrix32_.Read()+group*pm_*pk_, CUDA_R_32F, pm_, packed+offsets_[group]*pk_, CUDA_R_32F, pk_,
            &zero, result+offsets_[group]*pm_, CUDA_R_32F, pm_, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      shared_patch_internal::UnpackColumns<<<(height_+255)/256,256>>>(m_, pm_, columns_, output_map_.Read(),
          coarse.Read(), result, exponents, matrix_exponents_.Read(), invalid, column_scales_.Read(), y.Write());
    }
    else
    {
      double* packed = packed64_.Write(); double* result = result64_.Write();
      shared_patch_internal::PackColumns<<<columns_,128>>>(k_, pk_, input_map_.Read(), rhs->Read(), packed, exponents, invalid);
      const double one = 1.0, zero = 0.0;
      for (int group = 0; group < groups_; ++group)
        shared_patch_internal::BLAS(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
            pm_, offsets_[group+1]-offsets_[group], pk_, &one, matrix64_.Read()+group*pm_*pk_, pm_,
            packed+offsets_[group]*pk_, pk_, &zero, result+offsets_[group]*pm_, pm_));
      shared_patch_internal::UnpackColumns<<<(height_+255)/256,256>>>(m_, pm_, columns_, output_map_.Read(),
          coarse.Read(), result, exponents, matrix_exponents_.Read(), invalid, column_scales_.Read(), y.Write());
    }
    shared_patch_internal::CUDA(cudaGetLastError());
    ++applications_;
  }
private:
  int m_=0, k_=0, pm_=0, pk_=0, height_=0, groups_=0, columns_=0;
  bool tf32_=false;
  std::string composition_="additive";
  FrozenPatchResidual frozen_;
  mutable mfem::Vector residual_;
  cublasHandle_t handle_=nullptr;
  std::vector<int> offsets_;
  mfem::Array<int> input_map_, output_map_, matrix_exponents_;
  mutable mfem::Array<int> rhs_exponents_, invalid_;
  mfem::Vector column_scales_, matrix64_;
  mutable mfem::Vector packed64_, result64_;
  mfem::Array<float> matrix32_;
  mutable mfem::Array<float> packed32_, result32_;
  mutable unsigned long long applications_=0;
};

// Geometry is deliberately narrow: canonical slots are determined in each
// element's reference coordinates, including periodic seams, not by centroids.
inline std::vector<std::array<int,5>> SharedPatchTopology(mfem::ParFiniteElementSpace& sfes)
{
  auto& mesh = *sfes.GetMesh(); const int ne = sfes.GetNE();
  MFEM_VERIFY(ne >= 16 && sfes.GetFE(0)->GetOrder() == 5 && sfes.GetFE(0)->GetDof() == 36,
              "Shared patches currently require periodic uniform p5 quads with at least16 elements.");
  std::vector<std::array<int,5>> patch(ne);
  double hx=0.0, hy=0.0;
  for (int e=0; e<ne; ++e)
  {
    patch[e].fill(-1); patch[e][0]=e;
    auto* transform=mesh.GetElementTransformation(e);
    MFEM_VERIFY(transform->Order() == 1,
                "Shared patches require linear geometry; higher-order geometry is not admitted.");
    for (const auto point : std::array<std::array<double,2>,5>{{{0,0},{1,0},{0,1},{1,1},{.5,.5}}})
    {
      mfem::IntegrationPoint ip; ip.Set2(point[0],point[1]); transform->SetIntPoint(&ip);
      const auto& j=transform->Jacobian();
      if(e==0 && point[0]==0 && point[1]==0) { hx=j(0,0); hy=j(1,1); }
      const double scale=std::max(hx,hy);
      MFEM_VERIFY(hx>0 && hy>0 && std::abs(j(0,0)-hx)<=1e-11*scale && std::abs(j(1,1)-hy)<=1e-11*scale &&
          std::abs(j(0,1))<=1e-11*scale && std::abs(j(1,0))<=1e-11*scale,
          "Shared patches require identical positive axis-aligned affine Jacobians.");
    }
  }
  mfem::IntegrationPoint midpoint; midpoint.Set1w(.5, 1.0);
  for (int face=0; face<mesh.GetNumFaces(); ++face)
  {
    int a,b; mesh.GetFaceElements(face,&a,&b);
    MFEM_VERIFY(a>=0 && b>=0 && a!=b, "Shared patches require periodic interior faces only.");
    auto* transform=mesh.GetFaceElementTransformations(face);
    for (int side=0; side<2; ++side)
    {
      mfem::IntegrationPoint ip;
      (side ? transform->Loc2 : transform->Loc1).Transform(midpoint,ip);
      const int slot=std::abs(ip.x)<1e-12?1:std::abs(ip.x-1)<1e-12?2:
                     std::abs(ip.y)<1e-12?3:std::abs(ip.y-1)<1e-12?4:-1;
      const int e=side?b:a, neighbor=side?a:b;
      MFEM_VERIFY(slot>0 && patch[e][slot]<0, "Invalid/duplicate canonical shared-patch face slot.");
      patch[e][slot]=neighbor;
    }
  }
  for (int e=0; e<ne; ++e)
  {
    for (int i=0;i<5;++i) for (int j=0;j<i;++j)
      MFEM_VERIFY(patch[e][i]>=0 && patch[e][i]!=patch[e][j], "Shared patch contains duplicate/missing neighbors.");
    for (int row=0;row<5;++row) for (int col=0;col<5;++col)
    {
      const auto& first=patch[patch[0][row]]; const auto& current=patch[patch[e][row]];
      const bool expected=std::find(first.begin(),first.end(),patch[0][col])!=first.end();
      const bool actual=std::find(current.begin(),current.end(),patch[e][col])!=current.end();
      MFEM_VERIFY(expected==actual, "Shared patch principal-submatrix topology differs between elements.");
    }
  }
  return patch;
}

inline void ConfigureSharedPatch(SharedPatchBatch& batch, int dof,
    const std::vector<std::array<int,5>>& patch, const mfem::Array<int>& element_map,
    const std::vector<double>& stencil, const std::string& mode,
    const std::string& composition)
{
  const int ne=static_cast<int>(patch.size()), k=5*dof;
  constexpr int groups=4;
  std::vector<double> diagonal(ne); std::vector<int> sorted(ne);
  std::iota(sorted.begin(),sorted.end(),0);
  for(int e=0;e<ne;++e)
  {
    double mean=0.0;
    for(int i=0;i<dof;++i) mean+=stencil[(std::size_t(e)*5*dof+i)*dof+i];
    diagonal[e]=mean/dof;
    MFEM_VERIFY(diagonal[e]>0 && std::isfinite(diagonal[e]), "Shared patch requires positive finite mean diagonals.");
  }
  std::stable_sort(sorted.begin(),sorted.end(),[&](int a,int b){return diagonal[a]<diagonal[b];});
  std::vector<double> rows(groups*dof*k), scales(ne);
  std::vector<int> offsets(groups+1), inputs(ne*k), outputs(ne*dof), representatives(groups);
  double inverse_error=0.0, maximum_scale=0.0, minimum_scale=std::numeric_limits<double>::infinity();
  for(int group=0;group<groups;++group)
  {
    const int begin=ne*group/groups, end=ne*(group+1)/groups, representative=sorted[(begin+end)/2];
    offsets[group]=begin; offsets[group+1]=end; representatives[group]=representative;
    mfem::DenseMatrix a(k); a=0.0;
    for(int blockrow=0;blockrow<5;++blockrow)
    {
      const int e=patch[representative][blockrow];
      for(int slot=0;slot<5;++slot)
      {
        auto found=std::find(patch[representative].begin(),patch[representative].end(),patch[e][slot]);
        if(found==patch[representative].end()) continue;
        const int blockcol=static_cast<int>(found-patch[representative].begin());
        for(int j=0;j<dof;++j) for(int i=0;i<dof;++i)
          a(blockrow*dof+i,blockcol*dof+j)=stencil[((std::size_t(e)*5+slot)*dof+j)*dof+i];
      }
    }
    // Invert the actual principal submatrix: diagonal exterior-face penalties
    // remain present. Its central rows define a restricted additive correction.
    mfem::DenseMatrix transpose(k);
    for(int j=0;j<k;++j) for(int i=0;i<k;++i) transpose(i,j)=a(j,i);
    mfem::DenseMatrixInverse inverse(transpose);
    mfem::Vector unit(k), row(k), residual(k);
    for(int i=0;i<dof;++i)
    {
      unit=0.0; unit(i)=1.0; inverse.Mult(unit,row);
      transpose.Mult(row,residual); residual-=unit;
      inverse_error=std::max(inverse_error,residual.Norml2());
      for(int j=0;j<k;++j) rows[(group*k+j)*dof+i]=row(j);
    }
    for(int col=begin;col<end;++col)
    {
      const int e=sorted[col]; scales[col]=diagonal[representative]/diagonal[e];
      minimum_scale=std::min(minimum_scale,scales[col]); maximum_scale=std::max(maximum_scale,scales[col]);
      for(int i=0;i<dof;++i) outputs[col*dof+i]=element_map[e*dof+i];
      for(int slot=0;slot<5;++slot) for(int j=0;j<dof;++j)
        inputs[col*k+slot*dof+j]=element_map[patch[e][slot]*dof+j];
    }
  }
  MFEM_VERIFY(inverse_error<1e-8 && minimum_scale>=1.0/16 && maximum_scale<=16,
              "Shared patch inverse/scaling exceeds prototype admission: " << inverse_error);
  batch.Configure(dof,k,ne*dof,rows,offsets,inputs,outputs,scales,mode=="shared-patch-tf32",composition);
  if (composition == "multiplicative") batch.ConfigureFrozenOperator(dof,patch,element_map,stencil);
  std::cout << "[igr-shared-patch] event=setup mode=" << mode << " groups=" << groups
            << " matrix_m=" << dof << " matrix_k=" << k << " padded_m=" << ((dof+15)/16)*16
            << " padded_k=" << ((k+7)/8)*8 << " real_columns=" << ne << " min_group_columns=" << ne/groups
            << " representative_elements=" << representatives[0] << ',' << representatives[1] << ',' << representatives[2] << ',' << representatives[3]
            << " scale_min=" << minimum_scale << " scale_max=" << maximum_scale << " inverse_error=" << inverse_error
            << " coefficient_policy=frozen-mean-diagonal-scalar groups_policy=sorted-quartiles"
            << " topology=periodic-affine-p5-five-element rhs=distinct-patches backend=cublas"
            << " composition=" << composition
            << " residual_operator=" << (composition == "multiplicative" ? "frozen-original-fp64" : "none")
            << " frozen_operator_bytes=" << batch.FrozenOperatorBytes()
            << " tensor_execution=requires-profile-proof fallback=none\n";
}
} } // namespace igrlab::igr_internal

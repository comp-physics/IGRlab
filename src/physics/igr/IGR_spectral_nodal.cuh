#pragma once

// Private implementation shared by the spectral inverse and its CUDA tests.
// The host admission proof has no MFEM/CUDA dependency. The kernels below are
// the production implementations, not a separate diagnostic/test action API.
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <math_constants.h>
#endif

namespace igrlab { namespace igr_internal { namespace spectral_nodal_detail {

constexpr int d=6,n=d*d,tensor_threads=64;

struct Factors
{
  // Column-major (actual 1D node, Legendre degree) factors. Native nodes may
  // have any order; tensor positions are x-fast on the sorted actual axes.
  std::array<double,n> x{},y{};
  std::array<int,n> native_to_tensor{},tensor_to_native{};
  double matrix_relative_error=0;
};

struct FactorPacket
{
  double x[n],y[n];
  int native_to_tensor[n],tensor_to_native[n];
};

inline double LegendreValue(int degree,double x)
{
  const double z=2*x-1;
  if(degree==0)return 1;
  double p0=1,p1=z;
  for(int k=2;k<=degree;++k)
  {
    const double p=((2*k-1)*z*p1-(k-1)*p0)/k;p0=p1;p1=p;
  }
  return std::sqrt(2*degree+1.0)*p1;
}

// Admission is deliberately exact for the node Cartesian product and its
// permutation. Every stored 36x36 matrix entry is then checked against the
// factors; an unsupported/malformed representation retains the dense path.
inline bool BuildFactors(int order,int dimension,
    const std::vector<std::array<double,2>>& native_nodes,
    const std::vector<double>& nodal,Factors& factors,std::string& rejection)
{
  rejection.clear();
  auto reject=[&](const char* reason){rejection=reason;return false;};
  if(order!=5 || dimension!=2 || native_nodes.size()!=n)
    return reject("degree-dimension-or-node-count");
  if(nodal.size()!=n*n)return reject("nodal-matrix-size");
  std::vector<double> axes[2];
  for(const auto& point:native_nodes)for(int axis=0;axis<2;++axis)
  {
    if(!std::isfinite(point[axis]) || point[axis]<0 || point[axis]>1)
      return reject("nonfinite-or-out-of-reference-node");
    axes[axis].push_back(point[axis]);
  }
  for(auto& axis:axes)
  {
    std::sort(axis.begin(),axis.end());
    axis.erase(std::unique(axis.begin(),axis.end()),axis.end());
    if(axis.size()!=d)return reject("axis-node-count");
  }
  Factors candidate;
  candidate.native_to_tensor.fill(-1);candidate.tensor_to_native.fill(-1);
  for(int native=0;native<n;++native)
  {
    const auto& point=native_nodes[native];
    const int ix=static_cast<int>(std::lower_bound(axes[0].begin(),axes[0].end(),point[0])-axes[0].begin());
    const int iy=static_cast<int>(std::lower_bound(axes[1].begin(),axes[1].end(),point[1])-axes[1].begin());
    const int tensor=ix+d*iy;
    if(ix>=d || iy>=d || candidate.tensor_to_native[tensor]>=0)
      return reject("native-tensor-map-not-bijective");
    candidate.native_to_tensor[native]=tensor;candidate.tensor_to_native[tensor]=native;
  }
  for(int tensor=0;tensor<n;++tensor)
    if(candidate.tensor_to_native[tensor]<0 ||
        candidate.native_to_tensor[candidate.tensor_to_native[tensor]]!=tensor)
      return reject("native-tensor-map-not-bijective");
  for(int degree=0;degree<d;++degree)for(int node=0;node<d;++node)
  {
    candidate.x[node+d*degree]=LegendreValue(degree,axes[0][node]);
    candidate.y[node+d*degree]=LegendreValue(degree,axes[1][node]);
  }
  double error=0,magnitude=0;
  constexpr double tolerance=64*std::numeric_limits<double>::epsilon();
  for(int modal=0;modal<n;++modal)for(int native=0;native<n;++native)
  {
    const int tensor=candidate.native_to_tensor[native];
    const double expected=candidate.x[tensor%d+d*(modal%d)]*
        candidate.y[tensor/d+d*(modal/d)],value=nodal[native+n*modal];
    if(!std::isfinite(value) || !std::isfinite(expected))
      return reject("nonfinite-nodal-matrix");
    const double delta=expected-value;
    if(std::abs(delta)>tolerance*std::max({1.0,std::abs(expected),std::abs(value)}))
      return reject("nodal-matrix-mismatch");
    error+=delta*delta;magnitude+=value*value;
  }
  candidate.matrix_relative_error=std::sqrt(error/std::max(magnitude,std::numeric_limits<double>::min()));
  if(!std::isfinite(candidate.matrix_relative_error))return reject("nodal-matrix-mismatch");
  factors=candidate;return true;
}

inline FactorPacket PackFactors(const Factors& factors)
{
  FactorPacket packet{};
  for(int i=0;i<n;++i)
  {
    packet.x[i]=factors.x[i];packet.y[i]=factors.y[i];
    packet.native_to_tensor[i]=factors.native_to_tensor[i];
    packet.tensor_to_native[i]=factors.tensor_to_native[i];
  }
  return packet;
}

#if defined(__CUDACC__)
// Unchanged dense fallback. Field storage is column-major nx-by-ny, with six
// modal entries per Cartesian cell direction; map stores native element DOFs.
template<typename T> __global__ void ToSpectralKernel(int nx,int ny,
    const int* elements,const int* map,const double* nodal,const double* x,
    const int* exponent,T* result)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=nx*ny)return;
  if(*exponent==INT_MAX){result[row]=static_cast<T>(CUDART_NAN);return;}
  const int gx=row%nx,gy=row/nx,e=elements[gx/6+(nx/6)*(gy/6)],i=gx%6+6*(gy%6);
  double sum=0;
  for(int j=0;j<36;++j)sum+=nodal[j+36*i]*scalbn(x[map[36*e+j]],-*exponent);
  result[row]=static_cast<T>(sum);
}

template<typename T> __global__ void FromSpectralKernel(int nx,int ny,
    const int* elements,const int* map,const double* nodal,const T* x,
    const int* exponent,double scale,double* result)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=nx*ny)return;
  const int gx=row%nx,gy=row/nx,cx=gx/6,cy=gy/6,e=elements[cx+(nx/6)*cy],i=gx%6+6*(gy%6);
  if(*exponent==INT_MAX){result[map[36*e+i]]=CUDART_NAN;return;}
  double sum=0;
  for(int j=0;j<36;++j)sum+=nodal[i+36*j]*static_cast<double>(x[6*cx+j%6+nx*(6*cy+j/6)]);
  result[map[36*e+i]]=scalbn(sum/scale,*exponent);
}

// Launch one 64-thread block per Cartesian cell. Each mapped FP64 input is
// loaded and scaled once. Both contractions/intermediates are FP64, followed
// by one storage cast. Factoring groups the sums differently from dense36;
// rounding can differ, and the original-current-A certificate is unchanged.
template<typename T> __global__ void ToSpectralTensorKernel(int nx,int ny,
    const int* elements,const int* map,const FactorPacket* factors,const double* x,
    const int* exponent,T* result)
{
  __shared__ double source[n],temporary[n];
  const int cell=blockIdx.x,lane=threadIdx.x,cells_x=nx/d;
  if(cell>=cells_x*(ny/d))return;
  const int cx=cell%cells_x,cy=cell/cells_x,shift=*exponent;
  if(shift==INT_MAX)
  {
    if(lane<n)result[d*cx+lane%d+nx*(d*cy+lane/d)]=static_cast<T>(CUDART_NAN);
    return; // The sentinel branch is uniform across the entire block.
  }
  const int element=elements[cell];
  if(lane<n)
    source[factors->native_to_tensor[lane]]=scalbn(x[map[n*element+lane]],-shift);
  __syncthreads();
  if(lane<n)
  {
    const int ix=lane%d,iy=lane/d;double sum=0;
    #pragma unroll
    for(int j=0;j<d;++j)sum+=factors->x[j+d*ix]*source[j+d*iy];
    temporary[lane]=sum;
  }
  __syncthreads();
  if(lane<n)
  {
    const int ix=lane%d,iy=lane/d;double sum=0;
    #pragma unroll
    for(int j=0;j<d;++j)sum+=factors->y[j+d*iy]*temporary[ix+d*j];
    result[d*cx+ix+nx*(d*cy+iy)]=static_cast<T>(sum);
  }
}

template<typename T> __global__ void FromSpectralTensorKernel(int nx,int ny,
    const int* elements,const int* map,const FactorPacket* factors,const T* x,
    const int* exponent,double scale,double* result)
{
  __shared__ double source[n],temporary[n];
  const int cell=blockIdx.x,lane=threadIdx.x,cells_x=nx/d;
  if(cell>=cells_x*(ny/d))return;
  const int cx=cell%cells_x,cy=cell/cells_x,element=elements[cell],shift=*exponent;
  if(shift==INT_MAX)
  {
    if(lane<n)result[map[n*element+factors->tensor_to_native[lane]]]=CUDART_NAN;
    return;
  }
  if(lane<n)source[lane]=static_cast<double>(x[d*cx+lane%d+nx*(d*cy+lane/d)]);
  __syncthreads();
  if(lane<n)
  {
    const int ix=lane%d,iy=lane/d;double sum=0;
    #pragma unroll
    for(int j=0;j<d;++j)sum+=factors->x[ix+d*j]*source[j+d*iy];
    temporary[lane]=sum;
  }
  __syncthreads();
  if(lane<n)
  {
    const int ix=lane%d,iy=lane/d;double sum=0;
    #pragma unroll
    for(int j=0;j<d;++j)sum+=factors->y[iy+d*j]*temporary[ix+d*j];
    result[map[n*element+factors->tensor_to_native[lane]]]=scalbn(sum/scale,shift);
  }
}
#endif

} } }

// Focused numerical/throughput probe, deliberately outside the production API.
// All variants apply the same central DGSEM operator to Euler+sigma fluxes.
// This is a kernel experiment: sigma is prescribed, not solved in this executable.
#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_device.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <mfem.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void Check(cudaError_t e)
{ if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
void Blas(cublasStatus_t e)
{ if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS status "+std::to_string(e)); }

template<class T> struct Buffer
{
  T* p=nullptr;
  std::size_t count=0;
  explicit Buffer(std::size_t n):count(n)
  { Check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); }
  ~Buffer(){cudaFree(p);}
  Buffer(const Buffer&)=delete;
  Buffer& operator=(const Buffer&)=delete;
  void Upload(const std::vector<T>& v)
  { if(v.size()!=count)throw std::runtime_error("upload size");
    Check(cudaMemcpy(p,v.data(),count*sizeof(T),cudaMemcpyHostToDevice)); }
};

double Seconds(const std::function<void()>& f,int iterations,
    double* warmup_wall_seconds=nullptr)
{
  const auto warmup_start=std::chrono::steady_clock::now();
  for(int i=0;i<3;++i)f();
  Check(cudaDeviceSynchronize());
  if(warmup_wall_seconds)*warmup_wall_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-warmup_start).count();
  cudaEvent_t a,b;Check(cudaEventCreate(&a));Check(cudaEventCreate(&b));
  Check(cudaEventRecord(a));
  for(int i=0;i<iterations;++i)f();
  Check(cudaEventRecord(b));Check(cudaEventSynchronize(b));
  float ms=0;Check(cudaEventElapsedTime(&ms,a,b));
  Check(cudaEventDestroy(a));Check(cudaEventDestroy(b));
  Check(cudaGetLastError());
  return double(ms)*1.e-3/iterations;
}

// One coalesced cooperative block per element and component. Input/output
// remain FP64; FP32 conversion and arithmetic are inside the timed kernel.
template<int N,class T> __global__ void Ordinary(
    const double* f,const double* split,const double* weights,
    double* output,int ne)
{
  constexpr int dof=N*N;
  const int e=blockIdx.x%ne,eq=blockIdx.x/ne,rows=ne*dof;
  const int r=threadIdx.x;
  __shared__ T flux[2*dof];
  if(r<dof)
  {
    flux[r]=T(f[e*dof+r+rows*eq]);
    flux[dof+r]=T(f[e*dof+r+rows*(4+eq)]);
  }
  __syncthreads();
  if(r>=dof)return;
  const int i=r%N,j=r/N;
  T sum=0;
  #pragma unroll
  for(int m=0;m<N;++m)
  {
    if(m!=i)sum-=T(split[i+N*m])*T(.5)*(flux[r]+flux[m+N*j]);
    if(m!=j)sum-=T(split[j+N*m])*T(.5)*(flux[dof+r]+flux[dof+i+N*m]);
  }
  output[e*dof+r+rows*eq]=double(T(weights[i])*T(weights[j])*sum);
}

template<class T> void LaunchOrdinary(int n,int ne,const double* f,
    const double* s,const double* w,double* o)
{
  const int threads=std::max(32,n*n);
  switch(n)
  {
    case 2:Ordinary<2,T><<<ne*4,threads>>>(f,s,w,o,ne);break;
    case 4:Ordinary<4,T><<<ne*4,threads>>>(f,s,w,o,ne);break;
    case 5:Ordinary<5,T><<<ne*4,threads>>>(f,s,w,o,ne);break;
    case 8:Ordinary<8,T><<<ne*4,threads>>>(f,s,w,o,ne);break;
    case 16:Ordinary<16,T><<<ne*4,threads>>>(f,s,w,o,ne);break;
    default:throw std::runtime_error("n must be 2,4,5,8,16");
  }
}

// Same precombined matrix as the tensor path, applied with ordinary FMAs.
// Coefficients are converted once during setup. Separate directional sums and
// the FP64 final weighting match FusedLines' algebra and precision boundaries.
template<int N,class T> __global__ void OrdinaryMatrix(
    const double* f,const T* matrix,const double* weights,double* output,int ne)
{
  constexpr int dof=N*N;
  const int e=blockIdx.x%ne,eq=blockIdx.x/ne,rows=ne*dof,r=threadIdx.x;
  __shared__ T flux[2*dof],a[dof];
  if(r<dof)
  {
    flux[r]=T(f[e*dof+r+rows*eq]);
    flux[dof+r]=T(f[e*dof+r+rows*(4+eq)]);
    a[r]=matrix[r%N+16*(r/N)];
  }
  __syncthreads();
  if(r>=dof)return;
  const int i=r%N,j=r/N;
  T x=0,y=0;
  #pragma unroll
  for(int m=0;m<N;++m)
  {
    x+=a[i+N*m]*flux[m+N*j];
    y+=a[j+N*m]*flux[dof+i+N*m];
  }
  output[e*dof+r+rows*eq]=weights[i]*weights[j]*(double(x)+double(y));
}

template<class T> void LaunchOrdinaryMatrix(int n,int ne,const double* f,
    const T* a,const double* w,double* o)
{
  const int threads=std::max(32,n*n);
  switch(n)
  {
    case 2:OrdinaryMatrix<2,T><<<ne*4,threads>>>(f,a,w,o,ne);break;
    case 4:OrdinaryMatrix<4,T><<<ne*4,threads>>>(f,a,w,o,ne);break;
    case 5:OrdinaryMatrix<5,T><<<ne*4,threads>>>(f,a,w,o,ne);break;
    case 8:OrdinaryMatrix<8,T><<<ne*4,threads>>>(f,a,w,o,ne);break;
    case 16:OrdinaryMatrix<16,T><<<ne*4,threads>>>(f,a,w,o,ne);break;
  }
}

// Native FP64 WMMA: one warp per 8x8 directional output tile. The shared
// coefficient and flux panels are loaded once; padding, tile staging, and the
// final weighted directional sum are all charged inside this single kernel.
// N=8 and N=16 have no padded arithmetic. Small N packs multiple elements.
template<int N> __global__ void FusedDoubleLines(
    const double* flux,const double* matrix,const double* weights,
    double* output,int ne)
{
#if __CUDA_ARCH__ >= 800
  constexpr int group=N<=8?8/N:1,dof=N*N;
  constexpr int row_tiles=(N+7)/8,column_tiles=(group*N+7)/8;
  constexpr int warps=2*row_tiles*column_tiles;
  const int batches=(ne+group-1)/group;
  const int eq=blockIdx.x/batches,first=(blockIdx.x%batches)*group;
  const int thread=threadIdx.x,lane=thread%32,warp=thread/32,rows=ne*dof;
  const int direction=warp/(row_tiles*column_tiles);
  const int row_start=8*(warp%row_tiles);
  const int column_start=8*((warp/row_tiles)%column_tiles);
  __shared__ double a[dof],f[2*group*dof],c[2*group*dof];
  __shared__ __align__(32) double scratch[warps*128];
  for(int k=thread;k<dof;k+=blockDim.x)
    a[k]=matrix[k%N+16*(k/N)];
  for(int k=thread;k<2*group*dof;k+=blockDim.x)
  {
    const int dir=k/(group*dof),element=(k/dof)%group,node=k%dof;
    f[k]=first+element<ne?
        flux[(first+element)*dof+node+rows*(eq+4*dir)]:0.0;
  }
  __syncthreads();
  double* at=scratch+warp*128;
  double* bt=at+32;
  double* product=at+64;
  namespace wmma=nvcuda::wmma;
  wmma::fragment<wmma::accumulator,8,8,4,double> accumulator;
  wmma::fill_fragment(accumulator,0.0);
  #pragma unroll
  for(int chunk=0;chunk<N;chunk+=4)
  {
    const int row=row_start+lane/4,inner_a=chunk+lane%4;
    at[lane]=row<N&&inner_a<N?a[row+N*inner_a]:0.0;
    const int inner_b=chunk+lane/8,column=column_start+lane%8;
    const int element=column/N,line=column%N;
    const int node=direction==0?inner_b+N*line:line+N*inner_b;
    bt[lane]=inner_b<N&&element<group?
        f[direction*group*dof+element*dof+node]:0.0;
    __syncwarp();
    wmma::fragment<wmma::matrix_a,8,8,4,double,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,8,8,4,double,wmma::row_major> bf;
    wmma::load_matrix_sync(af,at,4);
    wmma::load_matrix_sync(bf,bt,8);
    wmma::mma_sync(accumulator,af,bf,accumulator);
    __syncwarp();
  }
  wmma::store_matrix_sync(product,accumulator,8,wmma::mem_row_major);
  __syncwarp();
  for(int entry=lane;entry<64;entry+=32)
  {
    const int coordinate=row_start+entry/8,column=column_start+entry%8;
    const int element=column/N,line=column%N;
    if(coordinate<N&&element<group)
    {
      const int node=direction==0?coordinate+N*line:line+N*coordinate;
      c[direction*group*dof+element*dof+node]=product[entry];
    }
  }
  __syncthreads();
  for(int k=thread;k<group*dof;k+=blockDim.x)
  {
    const int element=k/dof,node=k%dof,i=node%N,j=node/N;
    if(first+element<ne)
      output[(first+element)*dof+node+rows*eq]=
          weights[i]*weights[j]*(c[k]+c[group*dof+k]);
  }
#endif
}

void LaunchFusedDoubleLines(int n,int ne,const double* f,const double* a,
    const double* w,double* o)
{
  const int group=std::max(1,8/n),row_tiles=(n+7)/8;
  const int blocks=4*((ne+group-1)/group);
  const int threads=64*row_tiles*((group*n+7)/8);
  switch(n)
  {
    case 2:FusedDoubleLines<2><<<blocks,threads>>>(f,a,w,o,ne);break;
    case 4:FusedDoubleLines<4><<<blocks,threads>>>(f,a,w,o,ne);break;
    case 5:FusedDoubleLines<5><<<blocks,threads>>>(f,a,w,o,ne);break;
    case 8:FusedDoubleLines<8><<<blocks,threads>>>(f,a,w,o,ne);break;
    case 16:FusedDoubleLines<16><<<blocks,threads>>>(f,a,w,o,ne);break;
  }
}

__host__ __device__ constexpr unsigned Tf32RoundBits(unsigned bits)
{
  // Drop 13 FP32 fraction bits, rounding ties to an even retained bit.
  // Keep infinities and quiet NaNs before the rounding addition can carry.
  const unsigned magnitude=bits&0x7fffffffu;
  if(magnitude>=0x7f800000u)
    return magnitude==0x7f800000u?bits:(bits&0xffffe000u)|0x00400000u;
  return (bits+0xfffu+((bits>>13)&1u))&0xffffe000u;
}

static_assert(Tf32RoundBits(0x3f801000u)==0x3f800000u &&
    Tf32RoundBits(0x3f803000u)==0x3f804000u &&
    Tf32RoundBits(0xbf801000u)==0xbf800000u &&
    Tf32RoundBits(0xbf803000u)==0xbf804000u,"TF32 ties must round to even");
static_assert(Tf32RoundBits(0x3f800fffu)==0x3f800000u &&
    Tf32RoundBits(0x3f801001u)==0x3f802000u &&
    Tf32RoundBits(0x3ffff000u)==0x40000000u,"TF32 rounding and exponent carry");
static_assert(Tf32RoundBits(0x80000000u)==0x80000000u &&
    Tf32RoundBits(0x00001000u)==0x00000000u &&
    Tf32RoundBits(0x00003000u)==0x00004000u,"TF32 zero and subnormal rounding");
static_assert(Tf32RoundBits(0x7f7fffffu)==0x7f800000u &&
    Tf32RoundBits(0xff800000u)==0xff800000u &&
    (Tf32RoundBits(0x7f800001u)&0x7fffffffu)>0x7f800000u,
    "TF32 overflow, infinity and NaN");

__device__ float Tf32(float value)
{
#if __CUDA_ARCH__ >= 900
  float result;
  asm("cvt.rn.tf32.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
#elif __CUDA_ARCH__ >= 800
  // PTX cvt.rn.tf32 requires SM90; SM80's .rna uses ties away from zero.
  return __uint_as_float(Tf32RoundBits(__float_as_uint(value)));
#else
  return value;
#endif
}

__global__ void SplitCoefficients(const double* a,float* high,float* low,int count)
{
  const int k=blockIdx.x*blockDim.x+threadIdx.x;
  if(k<count)
  {
    const float h=Tf32(float(a[k]));
    high[k]=h;low[k]=Tf32(float(a[k]-double(h)));
  }
}

// Two warps compute the two directional contractions. Each tensor tile holds
// complete lines from floor(16/N) elements; conversion, padding and the final
// weighted sum stay on chip. The final partial group is explicitly zeroed.
template<int N,bool Compensate> __global__ void FusedLines(
    const double* flux,const float* coefficient_high,
    const float* coefficient_low,const double* weights,double* output,int ne)
{
#if __CUDA_ARCH__ >= 800
  constexpr int group=16/N,dof=N*N;
  const int batches=(ne+group-1)/group;
  const int eq=blockIdx.x/batches,first=(blockIdx.x%batches)*group;
  const int lane=threadIdx.x%32,direction=threadIdx.x/32,rows=ne*dof;
  __shared__ __align__(32) float tile[Compensate?2048:1280];
  float* ah=tile;float* al=Compensate?tile+256:nullptr;
  float* bh=tile+(Compensate?512:256);
  float* bl=Compensate?tile+1024:nullptr;
  float* c=tile+(Compensate?1536:768);
  for(int k=threadIdx.x;k<256;k+=64)
  {
    ah[k]=coefficient_high[k];
    if constexpr(Compensate)al[k]=coefficient_low[k];
  }
  for(int k=lane;k<256;k+=32)
  {
    const int coordinate=k%16,column=k/16,element=column/N,line=column%N;
    const int node=direction==0?coordinate+N*line:line+N*coordinate;
    const bool valid=coordinate<N&&element<group&&first+element<ne;
    const double value=valid?
        flux[(first+element)*dof+node+rows*(eq+4*direction)]:0.0;
    const float high=Tf32(float(value));
    bh[direction*256+k]=high;
    if constexpr(Compensate)
      bl[direction*256+k]=Tf32(float(value-double(high)));
  }
  __syncthreads();
  namespace wmma=nvcuda::wmma;
  wmma::fragment<wmma::accumulator,16,16,8,float> accumulator;
  wmma::fill_fragment(accumulator,0.0f);
  #pragma unroll
  for(int k=0;k<16;k+=8)
  {
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,
        wmma::col_major> a;
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,
        wmma::col_major> b;
    wmma::load_matrix_sync(a,ah+16*k,16);
    wmma::load_matrix_sync(b,bh+direction*256+k,16);
    wmma::mma_sync(accumulator,a,b,accumulator);
    if constexpr(Compensate)
    {
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,
          wmma::col_major> alo;
      wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,
          wmma::col_major> blo;
      wmma::load_matrix_sync(alo,al+16*k,16);
      wmma::load_matrix_sync(blo,bl+direction*256+k,16);
      wmma::mma_sync(accumulator,a,blo,accumulator);
      wmma::mma_sync(accumulator,alo,b,accumulator);
    }
  }
  wmma::store_matrix_sync(c+direction*256,accumulator,16,wmma::mem_col_major);
  __syncthreads();
  for(int k=threadIdx.x;k<group*dof;k+=64)
  {
    const int element=k/dof,node=k%dof,i=node%N,j=node/N;
    if(first+element<ne)
      output[(first+element)*dof+node+rows*eq]=weights[i]*weights[j]*
          (double(c[i+16*(element*N+j)])+
           double(c[256+j+16*(element*N+i)]));
  }
#endif
}

template<bool Compensate> void LaunchFusedLines(int n,int ne,
    const double* f,const float* ah,const float* al,const double* w,double* o)
{
  const int blocks=4*((ne+16/n-1)/(16/n));
  switch(n)
  {
    case 2:FusedLines<2,Compensate><<<blocks,64>>>(f,ah,al,w,o,ne);break;
    case 4:FusedLines<4,Compensate><<<blocks,64>>>(f,ah,al,w,o,ne);break;
    case 5:FusedLines<5,Compensate><<<blocks,64>>>(f,ah,al,w,o,ne);break;
    case 8:FusedLines<8,Compensate><<<blocks,64>>>(f,ah,al,w,o,ne);break;
    case 16:FusedLines<16,Compensate><<<blocks,64>>>(f,ah,al,w,o,ne);break;
  }
}

// Final bounded arithmetic screen: FP16 inputs with FP32 accumulation. This
// probe rejects values outside FP16's finite range; it does not silently scale
// coefficients or fluxes. Low products restore input bits lost by one FP16 cast.
template<int N,bool Compensate> __global__ void FusedHalfLines(
    const double* flux,const __half* coefficient_high,
    const __half* coefficient_low,const double* weights,double* output,
    int* overflow,int ne)
{
#if __CUDA_ARCH__ >= 800
  constexpr int group=16/N,dof=N*N;
  const int batches=(ne+group-1)/group;
  const int eq=blockIdx.x/batches,first=(blockIdx.x%batches)*group;
  const int lane=threadIdx.x%32,direction=threadIdx.x/32,rows=ne*dof;
  __shared__ __align__(32) __half tile[Compensate?1536:768];
  __shared__ __align__(32) float c[512];
  __half* ah=tile;__half* al=Compensate?tile+256:nullptr;
  __half* bh=tile+(Compensate?512:256);
  __half* bl=Compensate?tile+1024:nullptr;
  for(int k=threadIdx.x;k<256;k+=64)
  {
    ah[k]=coefficient_high[k];
    if constexpr(Compensate)al[k]=coefficient_low[k];
  }
  for(int k=lane;k<256;k+=32)
  {
    const int coordinate=k%16,column=k/16,element=column/N,line=column%N;
    const int node=direction==0?coordinate+N*line:line+N*coordinate;
    const bool valid=coordinate<N&&element<group&&first+element<ne;
    double value=valid?
        flux[(first+element)*dof+node+rows*(eq+4*direction)]:0.0;
    if(!isfinite(value)||fabs(value)>65504.0)
    {atomicExch(overflow,1);value=0.0;}
    const __half high=__float2half_rn(float(value));
    bh[direction*256+k]=high;
    if constexpr(Compensate)
      bl[direction*256+k]=__float2half_rn(float(value-double(__half2float(high))));
  }
  __syncthreads();
  namespace wmma=nvcuda::wmma;
  wmma::fragment<wmma::accumulator,16,16,16,float> accumulator;
  wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
  wmma::fill_fragment(accumulator,0.0f);
  wmma::load_matrix_sync(a,ah,16);
  wmma::load_matrix_sync(b,bh+direction*256,16);
  wmma::mma_sync(accumulator,a,b,accumulator);
  if constexpr(Compensate)
  {
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> alo;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> blo;
    wmma::load_matrix_sync(alo,al,16);
    wmma::load_matrix_sync(blo,bl+direction*256,16);
    wmma::mma_sync(accumulator,a,blo,accumulator);
    wmma::mma_sync(accumulator,alo,b,accumulator);
  }
  wmma::store_matrix_sync(c+direction*256,accumulator,16,wmma::mem_col_major);
  __syncthreads();
  for(int k=threadIdx.x;k<group*dof;k+=64)
  {
    const int element=k/dof,node=k%dof,i=node%N,j=node/N;
    if(first+element<ne)
      output[(first+element)*dof+node+rows*eq]=weights[i]*weights[j]*
          (double(c[i+16*(element*N+j)])+
           double(c[256+j+16*(element*N+i)]));
  }
#endif
}

template<bool Compensate> void LaunchFusedHalfLines(int n,int ne,
    const double* f,const __half* ah,const __half* al,const double* w,
    double* o,int* overflow)
{
  const int blocks=4*((ne+16/n-1)/(16/n));
  switch(n)
  {
    case 2:FusedHalfLines<2,Compensate><<<blocks,64>>>(f,ah,al,w,o,overflow,ne);break;
    case 4:FusedHalfLines<4,Compensate><<<blocks,64>>>(f,ah,al,w,o,overflow,ne);break;
    case 5:FusedHalfLines<5,Compensate><<<blocks,64>>>(f,ah,al,w,o,overflow,ne);break;
    case 8:FusedHalfLines<8,Compensate><<<blocks,64>>>(f,ah,al,w,o,overflow,ne);break;
    case 16:FusedHalfLines<16,Compensate><<<blocks,64>>>(f,ah,al,w,o,overflow,ne);break;
  }
}

void RejectHalfOverflow(const int* overflow)
{
  int value=0;Check(cudaMemcpy(&value,overflow,sizeof(int),cudaMemcpyDeviceToHost));
  if(value)throw std::runtime_error("FP16 input flux exceeds finite range");
}

// Column-major batches concatenate elements and conserved components. The line
// representation packs directions independently. The dense representation
// packs both directions into one longer vector. Padding is written every call.
__global__ void Pack(const double* flux,float* high,float* low,int n,int ne,
    int padded,bool dense,bool compensate,int count)
{
  const int k=blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=count)return;
  const int dof=n*n,rows=ne*dof,inner=dense?2*padded:padded;
  const int col=k/inner,coordinate=k%inner;
  int e=0,eq=0,dir=0,node=0;
  bool valid=true;
  if(dense)
  {
    e=col%ne;eq=col/ne;dir=coordinate/padded;node=coordinate%padded;
    valid=node<dof;
  }
  else
  {
    const int line=col%n,rest=col/n;
    e=rest%ne;eq=(rest/ne)%4;dir=rest/(ne*4);
    node=dir==0?coordinate+n*line:line+n*coordinate;
    valid=coordinate<n;
  }
  const double value=valid?flux[e*dof+node+rows*(eq+4*dir)]:0.0;
  const float h=compensate?Tf32(float(value)):float(value);
  high[k]=h;
  if(compensate)low[k]=Tf32(float(value-double(h)));
}

__global__ void Unpack(const float* c,const double* weights,double* output,
    int n,int ne,int padded,bool dense)
{
  const int k=blockIdx.x*blockDim.x+threadIdx.x,dof=n*n,rows=ne*dof;
  if(k>=4*rows)return;
  const int row=k%rows,eq=k/rows,e=row/dof,node=row%dof;
  if(dense)output[k]=double(c[node+padded*(e+ne*eq)]);
  else
  {
    const int i=node%n,j=node/n;
    const int cx=j+n*(e+ne*eq),cy=i+n*(e+ne*(eq+4));
    output[k]=weights[i]*weights[j]*
        (double(c[i+padded*cx])+double(c[j+padded*cy]));
  }
}

struct Error
{
  double relative_l2=0,maximum_absolute=0,maximum_relative_component_sum=0;
};
Error Compare(const double* device,const std::vector<long double>& reference)
{
  std::vector<double> y(reference.size());
  Check(cudaMemcpy(y.data(),device,y.size()*sizeof(double),cudaMemcpyDeviceToHost));
  long double error2=0,ref2=0;
  long double sum_error[4]={},sum_abs[4]={};
  Error result;
  for(std::size_t k=0;k<y.size();++k)
  {
    if(!std::isfinite(y[k]))throw std::runtime_error("nonfinite output");
    const long double d=static_cast<long double>(y[k])-reference[k];
    error2+=d*d;ref2+=reference[k]*reference[k];
    result.maximum_absolute=std::max(result.maximum_absolute,double(std::abs(d)));
    const int eq=int(k/(y.size()/4));
    sum_error[eq]+=d;sum_abs[eq]+=std::abs(reference[k]);
  }
  result.relative_l2=double(std::sqrt(error2/std::max(ref2,1.e-300L)));
  for(int eq=0;eq<4;++eq)
    result.maximum_relative_component_sum=std::max(
        result.maximum_relative_component_sum,
        double(std::abs(sum_error[eq])/std::max(sum_abs[eq],1.e-300L)));
  return result;
}

void Print(const std::string& method,int n,int ne,int iterations,
    double seconds,const Error& error,double setup_seconds,double pack=0,
    double multiply=0,double unpack=0,long long executed_flops=0,
    std::size_t scratch_bytes=0,std::size_t shared_bytes_per_block=0,
    double warmup_wall_seconds=0)
{
  const long long useful_multiply_adds=2LL*ne*4*n*n*n;
  std::cout<<std::setprecision(17)
      <<"{\"probe\":\"dgsem-central-volume\",\"method\":\""<<method
      <<"\",\"n1d\":"<<n<<",\"elements\":"<<ne
      <<",\"iterations\":"<<iterations<<",\"gpu_seconds\":"<<seconds
      <<",\"setup_seconds\":"<<setup_seconds
      <<",\"warmup_wall_seconds\":"<<warmup_wall_seconds
      <<",\"pack_seconds\":"<<pack<<",\"multiply_seconds\":"<<multiply
      <<",\"unpack_seconds\":"<<unpack
      <<",\"useful_linear_operator_flops\":"<<2*useful_multiply_adds
      <<",\"explicit_padded_gemm_flops\":"<<executed_flops
      <<",\"scratch_bytes\":"<<scratch_bytes
      <<",\"shared_bytes_per_block\":"<<shared_bytes_per_block
      <<",\"relative_l2_error\":"<<error.relative_l2
      <<",\"maximum_absolute_error\":"<<error.maximum_absolute
      <<",\"relative_component_sum_error\":"<<error.maximum_relative_component_sum
      <<",\"reference\":\"long-double-same-operator-on-FP64-flux\""
      <<",\"setup_excludes_reference_validation\":true"
      <<",\"warmup_scope\":\"three-calls-plus-sync-including-first-use\""
      <<",\"phase_timings\":\"independent-not-additive\""
      <<",\"scope\":\"central-volume-only-prescribed-sigma\"}\n";
}

void Run(int n,int ne,int iterations,cublasHandle_t handle)
{
  const auto setup_start=std::chrono::steady_clock::now();
  const int dof=n*n,rows=ne*dof;
  const auto& basis=mfem::dgsem::GetLGLBasis1D(n-1);
  mfem::Array<int> mapping(dof);for(int i=0;i<dof;++i)mapping[i]=i;
  mfem::DenseMatrix states(rows,4),metrics(rows,4),flux,existing_output;
  mfem::Vector sigma(rows),speeds;
  for(int e=0;e<ne;++e)for(int j=0;j<n;++j)for(int i=0;i<n;++i)
  {
    const int r=e*dof+i+n*j;
    const double x=(e%32+basis.nodes[i])/32.;
    const double y=(e/32+basis.nodes[j])/32.;
    const double rho=1+.04*std::sin(6.283185307179586*x)*std::cos(6.283185307179586*y);
    const double u=.18+.03*std::cos(6.283185307179586*y);
    const double v=-.09+.025*std::sin(6.283185307179586*x);
    const double p=1+.03*std::cos(6.283185307179586*(x+y));
    states(r,0)=rho;states(r,1)=rho*u;states(r,2)=rho*v;
    states(r,3)=p/.4+.5*rho*(u*u+v*v);
    // Nonconstant metrics exercise both matrix inputs, including cross terms.
    metrics(r,0)=1+.02*std::sin(x+y);metrics(r,1)=.02*std::cos(x);
    metrics(r,2)=.015*std::sin(y);metrics(r,3)=1+.025*std::cos(x-y);
    sigma[r]=.005*(1+.2*std::cos(6.283185307179586*(x-y)));
  }
  mfem::dgsem::BuildEulerContravariantFluxDevice(2,1.4,1.e-12,
      states,sigma,metrics,flux,speeds);
  Buffer<double> output(4*rows);
  (void)basis.Dsplit.Read();(void)basis.weights.Read();(void)mapping.Read();
  Check(cudaDeviceSynchronize());
  const double shared_setup=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-setup_start).count();
  std::vector<double> host_flux(flux.HostRead(),flux.HostRead()+flux.TotalSize());
  std::vector<long double> reference(4*rows);
  for(int eq=0;eq<4;++eq)for(int e=0;e<ne;++e)
    for(int j=0;j<n;++j)for(int i=0;i<n;++i)
    {
      const int row=e*dof+i+n*j;
      long double sum=0;
      for(int m=0;m<n;++m)
      {
        if(m!=i)sum-=static_cast<long double>(basis.Dsplit(i,m))*.5L*
            (static_cast<long double>(host_flux[row+rows*eq])+
             host_flux[e*dof+m+n*j+rows*eq]);
        if(m!=j)sum-=static_cast<long double>(basis.Dsplit(j,m))*.5L*
            (static_cast<long double>(host_flux[row+rows*(eq+4)])+
             host_flux[e*dof+i+n*m+rows*(eq+4)]);
      }
      reference[row+rows*eq]=static_cast<long double>(basis.weights[i])*
          basis.weights[j]*sum;
    }
  const double* f=flux.Read();const double* s=basis.Dsplit.Read();
  const double* w=basis.weights.Read();
  auto existing=[&]{mfem::dgsem::ApplyCentralVolumeDevice(2,n,mapping,
      basis.Dsplit,basis.weights,flux,existing_output);};
  double warmup=0;
  const double existing_seconds=Seconds(existing,iterations,&warmup);
  Print("existing-fp64",n,ne,iterations,existing_seconds,
      Compare(existing_output.Read(),reference),shared_setup,0,0,0,0,0,0,warmup);
  auto ordinary64=[&]{LaunchOrdinary<double>(n,ne,f,s,w,output.p);};
  const double ordinary64_seconds=Seconds(ordinary64,iterations,&warmup);
  Print("cooperative-fp64",n,ne,iterations,ordinary64_seconds,
      Compare(output.p,reference),shared_setup,0,0,0,0,0,
      2*n*n*sizeof(double),warmup);
  auto ordinary32=[&]{LaunchOrdinary<float>(n,ne,f,s,w,output.p);};
  const double ordinary32_seconds=Seconds(ordinary32,iterations,&warmup);
  Print("cooperative-fp32",n,ne,iterations,ordinary32_seconds,
      Compare(output.p,reference),shared_setup,0,0,0,0,0,
      2*n*n*sizeof(float),warmup);

  {
    const auto fused_setup_start=std::chrono::steady_clock::now();
    std::vector<double> a(256,0.0);
    for(int i=0;i<n;++i)for(int m=0;m<n;++m)if(i!=m)
    {const double value=-.5*basis.Dsplit(i,m);
      a[i+16*m]+=value;a[i+16*i]+=value;}
    const double host_matrix_setup=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-fused_setup_start).count();
    const auto upload64_start=std::chrono::steady_clock::now();
    Buffer<double> ad(256);ad.Upload(a);
    Check(cudaDeviceSynchronize());
    const double matrix64_setup=shared_setup+host_matrix_setup+
        std::chrono::duration<double>(
            std::chrono::steady_clock::now()-upload64_start).count();
    const auto upload32_start=std::chrono::steady_clock::now();
    Buffer<float> af(256);af.Upload(std::vector<float>(a.begin(),a.end()));
    Check(cudaDeviceSynchronize());
    const double matrix32_setup=shared_setup+host_matrix_setup+
        std::chrono::duration<double>(
            std::chrono::steady_clock::now()-upload32_start).count();
    auto matrix64=[&]{LaunchOrdinaryMatrix(n,ne,f,ad.p,w,output.p);};
    const double matrix64_seconds=Seconds(matrix64,iterations,&warmup);
    Print("cooperative-matrix-fp64",n,ne,iterations,matrix64_seconds,
        Compare(output.p,reference),matrix64_setup,0,0,0,0,
        256*sizeof(double),3*n*n*sizeof(double),warmup);
    auto fused64=[&]{LaunchFusedDoubleLines(n,ne,f,ad.p,w,output.p);};
    const double fused64_seconds=Seconds(fused64,iterations,&warmup);
    const int double_group=std::max(1,8/n),double_row_tiles=(n+7)/8;
    const int double_column_tiles=(double_group*n+7)/8;
    const int double_warps=2*double_row_tiles*double_column_tiles;
    const long long double_blocks=4LL*((ne+double_group-1)/double_group);
    Print("fused-lines-fp64-wmma",n,ne,iterations,fused64_seconds,
        Compare(output.p,reference),matrix64_setup,0,0,0,
        double_blocks*double_warps*((n+3)/4)*512,
        256*sizeof(double),sizeof(double)*(n*n+4*double_group*n*n+128*double_warps),
        warmup);
    auto matrix32=[&]{LaunchOrdinaryMatrix(n,ne,f,af.p,w,output.p);};
    const double matrix32_seconds=Seconds(matrix32,iterations,&warmup);
    Print("cooperative-matrix-fp32",n,ne,iterations,matrix32_seconds,
        Compare(output.p,reference),matrix32_setup,0,0,0,0,
        256*sizeof(float),3*n*n*sizeof(float),warmup);
    const auto split_setup_start=std::chrono::steady_clock::now();
    Buffer<float> ah(256),al(256);
    SplitCoefficients<<<1,256>>>(ad.p,ah.p,al.p,256);
    Check(cudaDeviceSynchronize());
    const double fused_setup=matrix64_setup+std::chrono::duration<double>(
        std::chrono::steady_clock::now()-split_setup_start).count();
    const long long blocks=4LL*((ne+16/n-1)/(16/n));
    auto single=[&]{LaunchFusedLines<false>(n,ne,f,ah.p,al.p,w,output.p);};
    const double single_seconds=Seconds(single,iterations,&warmup);
    Print("fused-lines-tf32",n,ne,iterations,single_seconds,
        Compare(output.p,reference),fused_setup,0,0,0,
        blocks*2*2*16*16*16,4096,4*1280,warmup);
    auto corrected=[&]{LaunchFusedLines<true>(n,ne,f,ah.p,al.p,w,output.p);};
    const double corrected_seconds=Seconds(corrected,iterations,&warmup);
    Print("fused-lines-tf32x3",n,ne,iterations,corrected_seconds,
        Compare(output.p,reference),fused_setup,0,0,0,
        blocks*2*2*16*16*16*3,4096,4*(256+256+512+512+512),warmup);

    const auto half_setup_start=std::chrono::steady_clock::now();
    std::vector<__half> host_ah(256),host_al(256);
    for(int k=0;k<256;++k)
    {
      if(!std::isfinite(a[k])||std::abs(a[k])>65504.0)
        throw std::runtime_error("FP16 coefficient exceeds finite range");
      host_ah[k]=__float2half_rn(float(a[k]));
      host_al[k]=__float2half_rn(float(a[k]-double(__half2float(host_ah[k]))));
    }
    Buffer<__half> half_ah(256),half_al(256);
    half_ah.Upload(host_ah);half_al.Upload(host_al);
    Buffer<int> half_overflow(1);half_overflow.Upload(std::vector<int>{0});
    Check(cudaDeviceSynchronize());
    const double half_setup=shared_setup+host_matrix_setup+
        std::chrono::duration<double>(
            std::chrono::steady_clock::now()-half_setup_start).count();
    auto half_single=[&]{LaunchFusedHalfLines<false>(n,ne,f,half_ah.p,
        half_al.p,w,output.p,half_overflow.p);};
    const double half_single_seconds=Seconds(half_single,iterations,&warmup);
    RejectHalfOverflow(half_overflow.p);
    Print("fused-lines-fp16",n,ne,iterations,half_single_seconds,
        Compare(output.p,reference),half_setup,0,0,0,
        blocks*2*2*16*16*16,1028,2*768+4*512,warmup);
    auto half_corrected=[&]{LaunchFusedHalfLines<true>(n,ne,f,half_ah.p,
        half_al.p,w,output.p,half_overflow.p);};
    const double half_corrected_seconds=Seconds(half_corrected,iterations,&warmup);
    RejectHalfOverflow(half_overflow.p);
    Print("fused-lines-fp16x3",n,ne,iterations,half_corrected_seconds,
        Compare(output.p,reference),half_setup,0,0,0,
        blocks*2*2*16*16*16*3,1028,2*1536+4*512,warmup);
  }

  for(bool dense:{false,true})
  {
    const auto matrix_setup_start=std::chrono::steady_clock::now();
    const int padded=dense?((dof+15)/16)*16:16;
    const int inner=dense?2*padded:padded;
    const int columns=dense?4*ne:8*ne*n;
    const int asize=padded*inner,bsize=inner*columns,csize=padded*columns;
    std::vector<double> a(asize,0);
    if(dense)
    {
      for(int j=0;j<n;++j)for(int i=0;i<n;++i)
      {
        const int r=i+n*j;
        const double weight=basis.weights[i]*basis.weights[j];
        for(int m=0;m<n;++m)
        {
          if(m!=i){const double value=-.5*weight*basis.Dsplit(i,m);
            a[r+padded*(m+n*j)]+=value;a[r+padded*r]+=value;}
          if(m!=j){const double value=-.5*weight*basis.Dsplit(j,m);
            a[r+padded*(padded+i+n*m)]+=value;
            a[r+padded*(padded+r)]+=value;}
        }
      }
    }
    else for(int i=0;i<n;++i)for(int m=0;m<n;++m)if(i!=m)
    { const double value=-.5*basis.Dsplit(i,m);
      a[i+padded*m]+=value;a[i+padded*i]+=value; }
    Buffer<double> ad(asize);ad.Upload(a);
    Buffer<float> ah(asize),al(asize),bh(bsize),bl(bsize),c(csize);
    const std::size_t scratch=sizeof(double)*asize+
        sizeof(float)*(2LL*asize+2LL*bsize+csize);
    Check(cudaDeviceSynchronize());
    const double matrix_setup=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-matrix_setup_start).count();
    for(bool compensate:{false,true})
    {
      const auto split_start=std::chrono::steady_clock::now();
      if(compensate)
        SplitCoefficients<<<(asize+255)/256,256>>>(ad.p,ah.p,al.p,asize);
      else
      {std::vector<float> af(a.begin(),a.end());ah.Upload(af);}
      Check(cudaDeviceSynchronize());
      const double coefficient_setup=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-split_start).count();
      auto pack=[&]{Pack<<<(bsize+255)/256,256>>>(f,bh.p,bl.p,n,ne,
          padded,dense,compensate,bsize);};
      auto multiply=[&]
      {
        const float alpha=1,beta0=0,beta1=1;
        const auto gemm=[&](const float* aa,const float* bb,const float* beta)
        {Blas(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,padded,columns,
            inner,&alpha,aa,CUDA_R_32F,padded,bb,CUDA_R_32F,inner,beta,c.p,
            CUDA_R_32F,padded,CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));};
        gemm(ah.p,bh.p,&beta0);
        if(compensate){gemm(ah.p,bl.p,&beta1);gemm(al.p,bh.p,&beta1);}
      };
      auto unpack=[&]{Unpack<<<(4*rows+255)/256,256>>>(c.p,w,output.p,n,ne,
          padded,dense);};
      auto full=[&]{pack();multiply();unpack();};
      const double full_seconds=Seconds(full,iterations,&warmup);
      const Error error=Compare(output.p,reference);
      const double pack_seconds=Seconds(pack,iterations);
      const double multiply_seconds=Seconds(multiply,iterations);
      const double unpack_seconds=Seconds(unpack,iterations);
      const std::string name=std::string(dense?"dense2d":"factorized-lines")+
          (compensate?"-tf32x3":"-tf32");
      Print(name,n,ne,iterations,full_seconds,error,
          shared_setup+matrix_setup+coefficient_setup,pack_seconds,
          multiply_seconds,unpack_seconds,
          2LL*padded*inner*columns*(compensate?3:1),scratch,0,warmup);
    }
  }
}
} // namespace

int main(int argc,char** argv)
{
  const auto process_start=std::chrono::steady_clock::now();
  mfem::MPI_Session mpi(argc,argv);
  try
  {
    int ne=1024,iterations=30,selected_n=0;
    for(int i=1;i<argc;++i)
    {
      const std::string option=argv[i];
      if(option=="--help")
      {std::cout<<"--elements N --iterations N [--n 2|4|5|8|16]\n";return 0;}
      if(++i>=argc)throw std::runtime_error("missing option value");
      const int value=std::stoi(argv[i]);
      if(value<=0)throw std::runtime_error("options must be positive");
      if(option=="--elements")ne=value;
      else if(option=="--iterations")iterations=value;
      else if(option=="--n")selected_n=value;
      else throw std::runtime_error("unknown option "+option);
    }
    if(ne>32768||iterations>10000)throw std::runtime_error("bounded probe limit");
    if(selected_n!=0&&selected_n!=2&&selected_n!=4&&selected_n!=5&&selected_n!=8&&selected_n!=16)
      throw std::runtime_error("n must be 2,4,5,8,16");
    int ranks=0;MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if(ranks!=1)throw std::runtime_error("requires one MPI rank");
    mfem::Device device("cuda");
    cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));
    cublasHandle_t handle;Blas(cublasCreate(&handle));
    int blas_version=0;Blas(cublasGetVersion(handle,&blas_version));
    const double process_initialization=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-process_start).count();
    std::cout<<std::setprecision(17)
        <<"{\"device\":\""<<prop.name<<"\",\"compute_major\":"
        <<prop.major<<",\"compute_minor\":"<<prop.minor
        <<",\"cublas_version\":"<<blas_version
        <<",\"process_initialization_seconds\":"<<process_initialization
        <<",\"process_initialization_scope\":\"MPI-device-context-cublas-handle\"}\n";
    for(int n:{2,4,5,8,16})if(selected_n==0||selected_n==n)
      Run(n,ne,iterations,handle);
    Blas(cublasDestroy(handle));
    return 0;
  }
  catch(const std::exception& e)
  {std::cerr<<"DGSEM tensor probe failed: "<<e.what()<<'\n';return 1;}
}

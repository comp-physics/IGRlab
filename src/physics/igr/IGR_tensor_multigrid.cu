#include "igrlab/compat.hpp"
#include "IGR_tensor_multigrid.hpp"
#include "IGR_tensor_center.hpp"
#include "IGR_device_krylov.hpp"
#include "IGR_shared_patch_cuda.cuh" // Reuse checked periodic face-slot topology.
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <climits>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>

namespace igrlab { namespace igr_internal { namespace {
using Stencil = std::vector<double>;
using Topology = std::vector<std::array<int, 5>>;

void CUDA(cudaError_t status)
{ MFEM_VERIFY(status == cudaSuccess, "Tensor multigrid CUDA: " << cudaGetErrorString(status)); }

const char* Name(TensorMultigridPrecision p)
{ return p == TensorMultigridPrecision::FP64 ? "fp64" : p == TensorMultigridPrecision::FP32 ? "fp32" : "tf32"; }

const char* SmootherName(TensorMultigridSmoother smoother)
{ return smoother==TensorMultigridSmoother::Center ? "restricted-overlap-3x3-center" : "restricted-overlap-3x3-separable"; }

std::size_t Entry(int e, int slot, int i, int j, int n)
{ return ((std::size_t(e) * 5 + slot) * n + j) * n + i; }

// Small deterministic symmetric Jacobi eigensystem. Columns of q are
// orthonormal eigenvectors; no dependency on a platform LAPACK ABI.
void Eigen(std::vector<double> a, int n, std::vector<double>& q,
           std::vector<double>& eigenvalues)
{
  q.assign(n*n, 0.0);
  for (int i=0; i<n; ++i) q[i+n*i]=1.0;
  double scale=0.0;
  for (double v:a) { MFEM_VERIFY(std::isfinite(v), "Nonfinite patch eigensystem."); scale=std::max(scale,std::abs(v)); }
  MFEM_VERIFY(scale>0.0, "Empty patch eigensystem.");
  bool converged=false;
  for (int sweep=0; sweep<100; ++sweep)
  {
    double largest=0.0;
    for (int j=1; j<n; ++j) for (int i=0; i<j; ++i) largest=std::max(largest,std::abs(a[i+n*j]));
    if (largest<=8*std::numeric_limits<double>::epsilon()*scale) { converged=true; break; }
    for (int j=1; j<n; ++j) for (int i=0; i<j; ++i)
    {
      const double off=a[i+n*j];
      if (std::abs(off)<=std::numeric_limits<double>::epsilon()*scale) continue;
      const double tau=(a[j+n*j]-a[i+n*i])/(2*off);
      const double t=std::copysign(1.0,tau)/(std::abs(tau)+std::hypot(1.0,tau));
      const double c=1/std::sqrt(1+t*t), s=t*c;
      const double ai=a[i+n*i], aj=a[j+n*j];
      a[i+n*i]=ai-t*off; a[j+n*j]=aj+t*off;
      a[i+n*j]=a[j+n*i]=0.0;
      for (int k=0;k<n;++k) if(k!=i && k!=j)
      {
        const double x=a[k+n*i],y=a[k+n*j];
        a[k+n*i]=a[i+n*k]=c*x-s*y;
        a[k+n*j]=a[j+n*k]=s*x+c*y;
      }
      for (int k=0;k<n;++k)
      {
        const double x=q[k+n*i],y=q[k+n*j];
        q[k+n*i]=c*x-s*y; q[k+n*j]=s*x+c*y;
      }
    }
  }
  MFEM_VERIFY(converged,"Patch symmetric eigensystem did not converge.");
  eigenvalues.resize(n);
  for(int i=0;i<n;++i) eigenvalues[i]=a[i+n*i];
}

double Legendre(int degree, double x)
{
  const double z=2*x-1;
  if(degree==0) return 1;
  double p0=1,p1=z;
  for(int k=2;k<=degree;++k) { const double p=((2*k-1)*z*p1-(k-1)*p0)/k; p0=p1;p1=p; }
  return std::sqrt(2*degree+1.0)*p1;
}

std::vector<int> Colors(const Topology& topology)
{
  const int ne=static_cast<int>(topology.size());
  std::vector<std::set<int>> conflict(ne);
  for(const auto& support:topology) for(int i:support) for(int j:support)
    if(i!=j) conflict[i].insert(j);
  std::vector<int> colors(ne,-1),used(ne,-1);
  for(int e=0;e<ne;++e)
  {
    for(int j:conflict[e]) if(colors[j]>=0) used[colors[j]]=e;
    int c=0;while(used[c]==e)++c;colors[e]=c;
  }
  // The coloring is a reconstruction condition, not a heuristic assumption.
  for(const auto& support:topology) for(int i=0;i<5;++i) for(int j=0;j<i;++j)
    MFEM_VERIFY(colors[support[i]]!=colors[support[j]],"Tensor MG stencil coloring collision.");
  return colors;
}

std::vector<int> PatchMap(const Topology& topology)
{
  const int ne=static_cast<int>(topology.size());
  std::vector<int> patches(ne*9);
  for(int e=0;e<ne;++e)
  {
    for(int y=0;y<3;++y) for(int x=0;x<3;++x)
    {
      const int ys=y==0?3:y==2?4:0, xs=x==0?1:x==2?2:0;
      const int a=topology[topology[e][ys]][xs], b=topology[topology[e][xs]][ys];
      MFEM_VERIFY(a==b,"Tensor patches require commuting Cartesian face neighbors.");
      patches[9*e+3*y+x]=a;
    }
    for(int i=0;i<9;++i) for(int j=0;j<i;++j)
      MFEM_VERIFY(patches[9*e+i]!=patches[9*e+j],"Tensor patches require nine distinct elements.");
    // Check the complete induced stencil, including periodic seams.
    for(int cell=0;cell<9;++cell) for(int slot=1;slot<5;++slot)
    {
      const int x=cell%3,y=cell/3;
      const int xx=x+(slot==1?-1:slot==2?1:0), yy=y+(slot==3?-1:slot==4?1:0);
      const int neighbor=topology[patches[9*e+cell]][slot];
      if(xx>=0 && xx<3 && yy>=0 && yy<3)
      { MFEM_VERIFY(neighbor==patches[9*e+3*yy+xx],"Tensor patch induced face orientation mismatch."); }
      else
      { MFEM_VERIFY(std::find(patches.begin()+9*e,patches.begin()+9*e+9,neighbor)==patches.begin()+9*e+9,
                    "Tensor patch wraps into itself; use at least four elements in each direction."); }
    }
  }
  return patches;
}

// Exact modal Galerkin snapshot, with every exterior-face diagonal retained.
Stencil Snapshot(const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes,
                 const Topology& topology, std::vector<int>& map,
                 std::vector<double>& nodal, int& probes, int& color_count)
{
  constexpr int d=6,n=36;
  const int ne=sfes.GetNE(),height=sfes.GetVSize();
  MFEM_VERIFY(std::size_t(ne)*5*n*n <= (std::size_t(1)<<26),"Tensor MG frozen stencil exceeds512MiB setup bound.");
  map.resize(height);nodal.resize(n*n);
  std::vector<char> seen(height,0);mfem::Array<int>dofs;
  for(int e=0;e<ne;++e)
  {
    const auto& fe=*sfes.GetFE(e);
    MFEM_VERIFY(fe.GetDof()==n && fe.GetOrder()==5 && fe.GetGeomType()==mfem::Geometry::SQUARE &&
      fe.GetMapType()==mfem::FiniteElement::VALUE && fe.GetNodes().GetNPoints()==n,
      "Tensor MG requires uniform p5 nodal VALUE L2 quads.");
    sfes.GetElementDofs(e,dofs);
    for(int i=0;i<n;++i)
    {
      MFEM_VERIFY(dofs[i]>=0 && dofs[i]<height && !seen[dofs[i]],"Tensor MG requires unsigned disjoint L2 maps.");
      map[e*n+i]=dofs[i];seen[dofs[i]]=1;
      const auto& ip=fe.GetNodes().IntPoint(i);
      for(int j=0;j<n;++j)
      {
        const double value=Legendre(j%d,ip.x)*Legendre(j/d,ip.y);
        if(e==0) nodal[i+n*j]=value;
        else MFEM_VERIFY(value==nodal[i+n*j],"Tensor MG requires identical nodal ordering.");
      }
    }
  }
  const auto colors=Colors(topology);
  color_count=1+*std::max_element(colors.begin(),colors.end());
  Stencil sampled(std::size_t(ne)*5*n*n,0.0);
  mfem::Vector input(height),output(height);input.UseDevice(true);output.UseDevice(true);
  for(int color=0;color<color_count;++color) for(int j=0;j<n;++j)
  {
    double* x=input.HostWrite();std::fill(x,x+height,0.0);
    // Probe modal columns directly, avoiding one of two dense basis changes.
    for(int e=0;e<ne;++e) if(colors[e]==color)
      for(int i=0;i<n;++i) x[map[e*n+i]]=nodal[i+n*j];
    op.Mult(input,output);++probes;
    const double* y=output.HostRead();
    for(int e=0;e<ne;++e) for(int slot=0;slot<5;++slot) if(colors[topology[e][slot]]==color)
      for(int i=0;i<n;++i)
      {
        double value=0;
        for(int k=0;k<n;++k) value+=static_cast<double>(nodal[k+n*i])*y[map[e*n+k]];
        MFEM_VERIFY(std::isfinite(static_cast<double>(value)),"Nonfinite frozen tensor MG stencil.");
        sampled[Entry(e,slot,i,j,n)]=static_cast<double>(value);
      }
  }
  return sampled;
}

Stencil Galerkin(const Stencil& fine,int ne,int d)
{
  const int n=d*d;Stencil result(std::size_t(ne)*5*n*n);
  for(int e=0;e<ne;++e)for(int s=0;s<5;++s)for(int j=0;j<n;++j)for(int i=0;i<n;++i)
    result[Entry(e,s,i,j,n)]=fine[Entry(e,s,i%d+6*(i/d),j%d+6*(j/d),36)];
  return result;
}

// Independent whole-domain probes detect missing graph edges and wrong
// restriction/order. Constant and nonconstant coarse polynomials are checked.
double CheckGalerkin(const mfem::Operator& op,const Topology& topology,
    const std::vector<int>& map,const std::vector<double>& nodal,
    const Stencil& a,int d,int& probes)
{
  const int ne=static_cast<int>(topology.size()),n=d*d;
  mfem::Vector x(ne*36),y(ne*36);x.UseDevice(true);y.UseDevice(true);
  double worst=0;
  for(int sample=0;sample<3;++sample)
  {
    std::vector<double> v(ne*n);
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
      v[e*n+i]=sample==0?(i==0?1.0:0.0):std::sin((sample==1?.17:.31)*(1+e*n+i));
    double* xx=x.HostWrite();
    for(int e=0;e<ne;++e)for(int k=0;k<36;++k)
    {
      double sum=0;for(int j=0;j<n;++j)sum+=static_cast<double>(nodal[k+36*(j%d+6*(j/d))])*v[e*n+j];
      xx[map[e*36+k]]=static_cast<double>(sum);
    }
    op.Mult(x,y);++probes;const double* yy=y.HostRead();
    double error=0,scale=0;
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
    {
      double measured=0,expected=0;
      for(int k=0;k<36;++k)measured+=static_cast<double>(nodal[k+36*(i%d+6*(i/d))])*yy[map[e*36+k]];
      for(int slot=0;slot<5;++slot)for(int j=0;j<n;++j)
        expected+=static_cast<double>(a[Entry(e,slot,i,j,n)])*v[topology[e][slot]*n+j];
      error+=(measured-expected)*(measured-expected);scale+=measured*measured;
    }
    const double relative=static_cast<double>(std::sqrt(error/std::max(scale,1e-300)));
    worst=std::max(worst,relative);
  }
  MFEM_VERIFY(worst<1e-8,"Tensor MG original/Galerkin stencil probe failed: "<<worst);
  return worst;
}

__global__ void InputMaximum(int n,const double* x,double* partial)
{
  __shared__ double values[256];double maximum=0;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
  { const double a=x[i];maximum=!isfinite(a)?CUDART_INF:fmax(maximum,fabs(a)); }
  values[threadIdx.x]=maximum;__syncthreads();
  for(int step=128;step;step/=2){if(threadIdx.x<step)values[threadIdx.x]=fmax(values[threadIdx.x],values[threadIdx.x+step]);__syncthreads();}
  if(threadIdx.x==0)partial[blockIdx.x]=values[0];
}
__global__ void InputExponent(int count,const double* partial,int* exponent)
{
  double maximum=0;for(int i=0;i<count;++i)maximum=fmax(maximum,partial[i]);
  int e=0;if(!isfinite(maximum))e=INT_MAX;else if(maximum>0)frexp(maximum,&e);*exponent=e;
}
template<typename T> __global__ void ToModal(int ne,const int* map,const double* t,
    const double* x,const int* exponent,T* b)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*36)return;
  if(*exponent==INT_MAX){b[row]=static_cast<T>(CUDART_NAN);return;}
  const int e=row/36,i=row%36;double sum=0;
  for(int k=0;k<36;++k)sum+=t[k+36*i]*scalbn(x[map[e*36+k]],-*exponent);
  b[row]=static_cast<T>(sum);
}
template<typename T> __global__ void FromModal(int ne,const int* map,const double* t,
    const T* x,const int* exponent,double scale,double* y)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*36)return;
  const int e=row/36,i=row%36;double sum=0;
  for(int j=0;j<36;++j)sum+=t[i+36*j]*static_cast<double>(x[e*36+j]);
  y[map[row]]=*exponent==INT_MAX?CUDART_NAN:scalbn(sum/scale,*exponent);
}
template<typename T> __global__ void Residual(int n,int ne,const int* neighbors,
    const T* a,const T* b,const T* x,T* r)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*n)return;
  const int e=row/n,i=row%n;T sum=0;
  for(int s=0;s<5;++s)for(int j=0;j<n;++j)
    sum+=a[((e*5+s)*n+j)*n+i]*x[neighbors[e*5+s]*n+j];
  r[row]=b[row]-sum;
}
template<typename T> __global__ void Restrict(int ne,int df,int dc,const T* r,T* b)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x,n=dc*dc;if(row>=ne*n)return;
  const int e=row/n,i=row%n;b[row]=r[e*df*df+i%dc+df*(i/dc)];
}
template<typename T> __global__ void ProlongAdd(int ne,int df,int dc,const T* c,T* x)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x,n=dc*dc;if(row>=ne*n)return;
  const int e=row/n,i=row%n;x[e*df*df+i%dc+df*(i/dc)]+=c[row];
}
template<typename T> __global__ void PackPatch(int ne,int d,const int* patches,const T* r,T* p)
{
  const int s=3*d,row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*s*s)return;
  const int e=row/(s*s),xy=row%(s*s),x=xy%s,y=xy/s;
  const int element=patches[9*e+3*(y/d)+x/d];p[row]=r[element*d*d+(x%d)+d*(y%d)];
}
template<typename T> __global__ void TransposePatch(int ne,int s,const T* a,T* b)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*s*s)return;
  const int e=row/(s*s),i=row%(s*s);b[e*s*s+(i%s)*s+i/s]=a[row];
}
template<typename T> __global__ void DividePatch(int ne,int s,const T* inverse,T* a)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<ne*s*s)a[row]*=inverse[row%(s*s)];
}
template<typename T> __global__ void AddCenter(int ne,int d,const T* factor,const T* patch,T* x)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x,n=d*d,s=3*d;if(row>=ne*n)return;
  const int e=row/n,i=row%n;x[row]+=T(.8)*factor[e]*patch[e*s*s+(d+i%d)+s*(d+i/d)];
}
template<typename T> __global__ void SharedTransformSIMT(int s,int columns,const T* a,const T* b,T* c)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=s*columns)return;
  const int i=row%s,j=row/s;T sum=0;for(int k=0;k<s;++k)sum+=a[i+s*k]*b[k+s*j];c[row]=sum;
}

// Each warp computes16 distinct real patch-line RHSs, except zero-padded tails.
// No RHS is copied into another column. Four independent warp tiles per CTA.
__global__ void IGRTensorMultigridTF32TransformKernel(int s,int columns,
    const float* a,const float* b,float* c)
{
#if __CUDA_ARCH__ >= 800
  const int warp=threadIdx.x/32,lane=threadIdx.x%32,tiles_m=(s+15)/16;
  const int tile=4*blockIdx.x+warp;if(tile>=tiles_m*((columns+15)/16))return;
  const int row0=16*(tile%tiles_m),col0=16*(tile/tiles_m);
  __shared__ __align__(32) float sa[4][128],sb[4][128],sc[4][256];
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> fa;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> fb;
  wmma::fragment<wmma::accumulator,16,16,8,float> fc;wmma::fill_fragment(fc,0.0f);
  for(int k0=0;k0<s;k0+=8)
  {
    for(int i=lane;i<128;i+=32)
    {
      const int ar=row0+i%16,ak=k0+i/16,bk=k0+i%8,bc=col0+i/8;
      sa[warp][i]=wmma::__float_to_tf32(ar<s && ak<s?a[ar+s*ak]:0.0f);
      sb[warp][i]=wmma::__float_to_tf32(bk<s && bc<columns?b[bk+s*bc]:0.0f);
    }
    __syncwarp();wmma::load_matrix_sync(fa,sa[warp],16);wmma::load_matrix_sync(fb,sb[warp],8);
    wmma::mma_sync(fc,fa,fb,fc);__syncwarp();
  }
  wmma::store_matrix_sync(sc[warp],fc,16,wmma::mem_col_major);__syncwarp();
  for(int i=lane;i<256;i+=32)
  { const int r=row0+i%16,col=col0+i/16;if(r<s && col<columns)c[r+s*col]=sc[warp][i]; }
#else
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<s*columns)c[row]=CUDART_NAN_F;
#endif
}

// Each SIMT thread owns one genuine element-center output, for one K slice.
// Split boundaries coincide with the TF32 kernel's eight-entry K panels.
template<typename T> __global__ void IGRTensorMultigridCenterSIMTKernel(
    int ne,int d,const T* response,const T* packed,T* partial,int split_k)
{
  const int n=d*d,k=9*n,row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=ne*n)return;
  const int part=blockIdx.y,panels=(k+7)/8;
  const int begin=8*((panels*part)/split_k);
  const int end=min(k,8*((panels*(part+1))/split_k));
  const int element=row/n,center=row%n;T sum=0;
  for(int j=begin;j<end;++j)sum+=response[center+n*j]*packed[j+k*element];
  partial[part*ne*n+row]=isfinite(sum)?sum:static_cast<T>(CUDART_NAN);
}

// Warps share one converted response panel and each own 16 distinct real
// patch columns. The skewed leading dimensions (24,16,24) replace the tightly
// packed, independently restaged panels of the separable control. All warps
// participate in CTA barriers, including M/N tails; tails are only zero fill.
// Scalar shared transfers traverse physical padded offsets, skipping padding,
// so active lanes in a warp access distinct banks in each transfer.
template<int Warps> __global__ void IGRTensorMultigridCenterTF32Kernel(
    int ne,int d,const float* response,const float* packed,float* partial,int split_k)
{
#if __CUDA_ARCH__ >= 800
  const int n=d*d,k=9*n,tiles_m=(n+15)/16;
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
  const int row0=16*(blockIdx.x%tiles_m);
  const int col0=16*(Warps*(blockIdx.x/tiles_m)+warp);
  const int part=blockIdx.y,panels=(k+7)/8;
  const int first=(panels*part)/split_k,last=(panels*(part+1))/split_k;
  __shared__ __align__(32) float sa[8*24],sb[Warps][16*16],sc[Warps][16*24];
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> fa;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> fb;
  wmma::fragment<wmma::accumulator,16,16,8,float> fc;wmma::fill_fragment(fc,0.0f);
  for(int panel=first;panel<last;++panel)
  {
    const int k0=8*panel;
    for(int i=threadIdx.x;i<8*24;i+=32*Warps) if(i%24<16)
    {
      const int row=row0+i%24,column=k0+i/24;
      const float value=wmma::__float_to_tf32(row<n && column<k?response[row+n*column]:0.0f);
      sa[i]=isfinite(value)?value:CUDART_NAN_F;
    }
    for(int i=lane;i<16*16;i+=32) if(i%16<8)
    {
      const int row=k0+i%16,column=col0+i/16;
      const float value=wmma::__float_to_tf32(row<k && column<ne?packed[row+k*column]:0.0f);
      sb[warp][i]=isfinite(value)?value:CUDART_NAN_F;
    }
    __syncthreads();
    wmma::load_matrix_sync(fa,sa,24);wmma::load_matrix_sync(fb,sb[warp],16);
    wmma::mma_sync(fc,fa,fb,fc);
    __syncthreads();
  }
  wmma::store_matrix_sync(sc[warp],fc,24,wmma::mem_col_major);__syncwarp();
  for(int i=lane;i<16*24;i+=32) if(i%24<16)
  {
    const int row=row0+i%24,column=col0+i/24;
    if(row<n && column<ne)
    {
      const float value=sc[warp][i];
      partial[(part*ne+column)*n+row]=isfinite(value)?value:CUDART_NAN_F;
    }
  }
#else
  // A TF32 request is never silently replaced by a SIMT implementation.
  const int n=d*d;
  for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<ne*n;row+=gridDim.x*blockDim.x)
    partial[blockIdx.y*ne*n+row]=CUDART_NAN_F;
#endif
}

template<typename T> __global__ void IGRTensorMultigridAddCenterKernel(
    int ne,int d,const T* factor,const T* partial,T* x,int split_k)
{
  const int n=d*d,row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*n)return;
  T sum=0;for(int part=0;part<split_k;++part)sum+=partial[part*ne*n+row];
  const T scale=factor[row/n],updated=x[row]+T(.8)*scale*sum;
  x[row]=isfinite(sum)&&isfinite(scale)&&isfinite(updated)?updated:static_cast<T>(CUDART_NAN);
}

void CheckCenterLayout(int ne,int d,int warps,int split_k)
{
  MFEM_VERIFY(ne>0 && (d==4||d==6) && (warps==1||warps==4) &&
    (split_k==1||split_k==2||split_k==4),"Invalid tensor center response layout.");
  MFEM_VERIFY(std::size_t(ne)*9*d*d<=INT_MAX && std::size_t(ne)*d*d*split_k<=INT_MAX,
              "Tensor center response layout exceeds integer indexing.");
}

template<typename From,typename To> __global__ void Convert(int n,const From* x,To* y)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=static_cast<To>(x[i]);}

template<typename T> void Store(mfem::Array<T>& dst,const std::vector<double>& src)
{
  dst.SetSize(static_cast<int>(src.size()));T* out=dst.HostWrite();
  for(std::size_t i=0;i<src.size();++i)
  {out[i]=static_cast<T>(src[i]);MFEM_VERIFY(std::isfinite(static_cast<double>(out[i])),"Tensor MG storage overflow.");}
  (void)dst.Read();
}

struct PatchReference
{
  std::vector<double> forward_x,forward_y,back_x,back_y,inverse,scales;
  int clipped=0;
  double relative_error=0, eigen_error=0;
};

PatchReference Reference(const Stencil& a,int ne,int d)
{
  const int n=d*d,s=3*d,np=s*s;PatchReference result;
  std::vector<double> average(5*n*n,0.0);
  for(int slot=0;slot<5;++slot)for(int j=0;j<n;++j)for(int i=0;i<n;++i)
  {
    double sum=0;for(int e=0;e<ne;++e)sum+=a[Entry(e,slot,i,j,n)];
    average[(slot*n+j)*n+i]=static_cast<double>(sum/ne);
  }
  std::vector<double> patch(np*np,0.0);
  for(int cell=0;cell<9;++cell)for(int slot=0;slot<5;++slot)
  {
    const int x=cell%3,y=cell/3,xx=x+(slot==1?-1:slot==2?1:0),yy=y+(slot==3?-1:slot==4?1:0);
    if(xx<0||xx>=3||yy<0||yy>=3)continue;
    for(int j=0;j<n;++j)for(int i=0;i<n;++i)
    {
      const int row=x*d+i%d+s*(y*d+i/d),col=xx*d+j%d+s*(yy*d+j/d);
      patch[row+np*col]=average[(slot*n+j)*n+i];
    }
  }
  double asymmetry=0,magnitude=0;
  for(int j=0;j<np;++j)for(int i=0;i<np;++i)
  {const double x=patch[i+np*j],y=patch[j+np*i];asymmetry+=(x-y)*(x-y);magnitude+=x*x;}
  MFEM_VERIFY(std::sqrt(asymmetry/magnitude)<1e-10,"Tensor MG patch reference is not symmetric.");
  for(int j=0;j<np;++j)for(int i=0;i<j;++i)
    patch[i+np*j]=patch[j+np*i]=.5*(patch[i+np*j]+patch[j+np*i]);
  std::vector<double> ax(s*s,0),ay(s*s,0);
  for(int j=0;j<s;++j)for(int i=0;i<s;++i)for(int k=0;k<s;++k)
  {ax[i+s*j]+=patch[(i+s*k)+np*(j+s*k)]/s;ay[i+s*j]+=patch[(k+s*i)+np*(k+s*j)]/s;}
  double mean=0;for(int i=0;i<s;++i)mean+=ax[i+s*i]/s;
  double mass=0;for(int slot=0;slot<5;++slot)mass+=average[slot*n*n];
  MFEM_VERIFY(mass>0 && std::isfinite(mass),"Tensor MG requires positive screened constant mode.");
  std::vector<double> lx,ly;Eigen(ax,s,result.back_x,lx);Eigen(ay,s,result.back_y,ly);
  result.forward_x.resize(s*s);result.forward_y.resize(s*s);result.inverse.resize(s*s);
  for(int j=0;j<s;++j)for(int i=0;i<s;++i)
  {
    result.forward_x[i+s*j]=result.back_x[j+s*i];result.forward_y[i+s*j]=result.back_y[j+s*i];
    const double value=lx[i]+ly[j]-mean;
    if(value<mass)++result.clipped;
    // The exact constant-coefficient Kronecker sum has no clipping. Variable
    // coefficients are approximated only inside this positive patch smoother.
    result.inverse[j+s*i]=1/std::max(mass,value); // transposed (y,x) layout
  }
  for(int dim=0;dim<2;++dim)
  {
    const auto& mat=dim?ay:ax;const auto& q=dim?result.back_y:result.back_x;const auto& eig=dim?ly:lx;
    double error=0,scale=0;
    for(int j=0;j<s;++j)for(int i=0;i<s;++i)
    {double v=0;for(int k=0;k<s;++k)v+=static_cast<double>(mat[i+s*k])*q[k+s*j];
     const double delta=v-q[i+s*j]*eig[j];error+=delta*delta;scale+=v*v;}
    result.eigen_error=std::max(result.eigen_error,static_cast<double>(std::sqrt(error/scale)));
  }
  MFEM_VERIFY(result.eigen_error<1e-12,"Tensor patch eigensystem residual failed.");
  double error=0,scale=0;
  for(int j=0;j<np;++j)for(int i=0;i<np;++i)
  {
    const int ix=i%s,iy=i/s,jx=j%s,jy=j/s;
    const double approx=(iy==jy?ax[ix+s*jx]:0)+(ix==jx?ay[iy+s*jy]:0)-(i==j?mean:0);
    const double delta=approx-patch[i+np*j];error+=delta*delta;scale+=patch[i+np*j]*patch[i+np*j];
  }
  result.relative_error=static_cast<double>(std::sqrt(error/scale));
  double reference_diag=0;for(int i=0;i<n;++i)reference_diag+=average[i+n*i];
  result.scales.resize(ne);
  for(int e=0;e<ne;++e)
  {
    double diagonal=0;for(int i=0;i<n;++i)diagonal+=a[Entry(e,0,i,i,n)];
    const double ratio=reference_diag/diagonal;
    MFEM_VERIFY(std::isfinite(ratio)&&ratio>=1.0/16&&ratio<=16,"Tensor MG patch coefficient contrast exceeds16.");
    result.scales[e]=ratio;
  }
  return result;
}

template<typename T> struct Level
{
  int d=0;
  mfem::Array<T> a,forward_x,forward_y,back_x,back_y,inverse,center_response,scale;
  mutable mfem::Array<T> rhs,x,residual,patch_a,patch_b;
};

template<typename T> class TensorMultigrid final:public mfem::Solver
{
public:
  TensorMultigrid(const mfem::Operator& op,mfem::ParFiniteElementSpace& sfes,
      TensorMultigridPrecision precision,TensorMultigridSmoother smoother,
      int center_warps,int center_split_k)
    :mfem::Solver(op.Height()),ne_(sfes.GetNE()),precision_(precision),smoother_(smoother),
     center_warps_(center_warps),center_split_k_(center_split_k)
  {
    const auto start=std::chrono::steady_clock::now();
    int ranks=0;MPI_Comm_size(sfes.GetComm(),&ranks);
    MFEM_VERIFY(ranks==1 && mfem::Device::Allows(mfem::Backend::CUDA) &&
      dynamic_cast<const mfem::L2_FECollection*>(sfes.FEColl()) && sfes.GetVDim()==1 &&
      sfes.GetMesh()->Dimension()==2 && sfes.GetVSize()==height && sfes.GetTrueVSize()==height &&
      op.Width()==height && height==ne_*36,"Tensor MG requires rank-one CUDA scalar p5 L2.");
    int device=0;cudaDeviceProp prop{};CUDA(cudaGetDevice(&device));CUDA(cudaGetDeviceProperties(&prop,device));
    if(precision_==TensorMultigridPrecision::TF32)
    {MFEM_VERIFY(prop.major>=8,"Tensor MG TF32 requires nativeSM80+.");}
    const auto topology=SharedPatchTopology(sfes);const auto patch=PatchMap(topology);
    neighbors_.SetSize(ne_*5);patches_.SetSize(ne_*9);
    for(int e=0;e<ne_;++e)for(int s=0;s<5;++s)neighbors_[5*e+s]=topology[e][s];
    std::copy(patch.begin(),patch.end(),patches_.HostWrite());
    std::vector<int> map;std::vector<double> nodal;int probes=0,colors=0;
    auto fine=Snapshot(op,sfes,topology,map,nodal,probes,colors);
    double probe_error=0;std::array<Stencil,3> matrices;
    for(int l=0;l<3;++l)
    {const int d=l==0?6:l==1?4:2;matrices[l]=l==0?fine:Galerkin(fine,ne_,d);
     probe_error=std::max(probe_error,CheckGalerkin(op,topology,map,nodal,matrices[l],d,probes));}
    operator_scale_=0;
    for(int e=0;e<ne_;++e)for(int i=0;i<36;++i)operator_scale_=std::max(operator_scale_,std::abs(fine[Entry(e,0,i,i,36)]));
    MFEM_VERIFY(operator_scale_>0 && std::isfinite(operator_scale_),"Invalid tensor MG operator scale.");
    element_map_.SetSize(height);std::copy(map.begin(),map.end(),element_map_.HostWrite());
    nodal_.SetSize(36*36);std::copy(nodal.begin(),nodal.end(),nodal_.HostWrite());nodal_.UseDevice(true);
    std::size_t bytes=0,center_bytes=0,patch_bytes=0;int clipped=0;
    double separable_error=0,eigen_error=0,center_setup_seconds=0;
    for(int l=0;l<3;++l)
    {
      auto& level=levels_[l];level.d=l==0?6:l==1?4:2;const int n=level.d*level.d;
      for(double& value:matrices[l])value/=operator_scale_;
      Store(level.a,matrices[l]);bytes+=matrices[l].size()*sizeof(T);
      level.rhs.SetSize(ne_*n);level.x.SetSize(ne_*n);level.residual.SetSize(ne_*n);
      if(l<2)
      {
        const auto ref=Reference(matrices[l],ne_,level.d);
        Store(level.scale,ref.scales);
        const int size=ne_*9*n;level.patch_a.SetSize(size);
        if(smoother_==TensorMultigridSmoother::Center)
        {
          const auto center_start=std::chrono::steady_clock::now();
          const auto response=ComposeTensorCenterResponse(level.d,ref.forward_x,ref.forward_y,
              ref.back_x,ref.back_y,ref.inverse);
          Store(level.center_response,response);
          center_setup_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-center_start).count();
          center_bytes+=response.size()*sizeof(T);
          level.patch_b.SetSize(center_split_k_*ne_*n);
        }
        else
        {
          Store(level.forward_x,ref.forward_x);Store(level.forward_y,ref.forward_y);
          Store(level.back_x,ref.back_x);Store(level.back_y,ref.back_y);
          Store(level.inverse,ref.inverse);level.patch_b.SetSize(size);
        }
        patch_bytes+=std::size_t(level.patch_a.Size()+level.patch_b.Size())*sizeof(T);
        clipped+=ref.clipped;separable_error=std::max(separable_error,ref.relative_error);eigen_error=std::max(eigen_error,ref.eigen_error);
      }
    }
    coarse_=std::make_unique<mfem::SparseMatrix>(4*ne_);
    for(int e=0;e<ne_;++e)for(int s=0;s<5;++s)for(int j=0;j<4;++j)for(int i=0;i<4;++i)
      coarse_->Add(4*e+i,4*topology[e][s]+j,matrices[2][Entry(e,s,i,j,4)]);
    coarse_->Finalize();
    mfem::Vector diagonal(4*ne_);coarse_->GetDiag(diagonal);
    // SparseMatrix::GetDiag writes through the device backend; Vector::operator()
    // does not synchronize host validity. Validate the actual computed entries.
    const double* diagonal_values=diagonal.HostRead();
    for(int i=0;i<diagonal.Size();++i)
      MFEM_VERIFY(std::isfinite(diagonal_values[i])&&diagonal_values[i]>0,
                  "Nonpositive tensor p1 diagonal at row "<<i<<": "<<diagonal_values[i]);
    jacobi_=std::make_unique<mfem::OperatorJacobiSmoother>(diagonal,essential_,1.0);
    coarse_solver_=MakeCooperativeCGSolver(sfes.GetComm());cooperative_=bool(coarse_solver_);
    if(!coarse_solver_)coarse_solver_=MakeDeviceCGSolver(sfes.GetComm());
    coarse_solver_->SetOperator(*coarse_);coarse_solver_->SetPreconditioner(*jacobi_);
    coarse_solver_->SetRelTol(1e-2);coarse_solver_->SetAbsTol(0);coarse_solver_->SetMaxIter(100);
    coarse_solver_->SetPrintLevel(0);coarse_solver_->iterative_mode=false;
    coarse_rhs_.SetSize(4*ne_);coarse_x_.SetSize(4*ne_);coarse_rhs_.UseDevice(true);coarse_x_.UseDevice(true);
    norm_blocks_=std::min(4096,(height+255)/256);norm_partial_.SetSize(norm_blocks_);norm_partial_.UseDevice(true);exponent_.SetSize(1);
    (void)neighbors_.Read();(void)patches_.Read();(void)element_map_.Read();(void)nodal_.Read();
    CUDA(cudaDeviceSynchronize());
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"[igr-tensor-mg] event=setup precision="<<Name(precision_)<<" hierarchy=5,3,1 cycle=multiplicative-v"
      <<" pre_sweeps=1 post_sweeps=1 smoother="<<SmootherName(smoother_)<<" damping=0.8"
      <<" center_warps="<<center_warps_<<" center_split_k="<<center_split_k_
      <<" elements="<<ne_<<" n="<<height<<" coarse_n="<<4*ne_<<" coarse_precision=fp64 coarse_rtol=0.01 coarse_maxiter=100"
      <<" coarse_backend=cuda-fp64 coarse_execution="<<(cooperative_?"cooperative":"host-polled")
      <<" vectors="<<(std::is_same<T,double>::value?"fp64":"fp32")
      <<" real_columns_p5="<<(smoother_==TensorMultigridSmoother::Center?ne_:18*ne_)
      <<" real_columns_p3="<<(smoother_==TensorMultigridSmoother::Center?ne_:12*ne_)
      <<" operator_probes="<<probes<<" colors="<<colors
      <<" center_map_bytes="<<center_bytes<<" patch_workspace_bytes="<<patch_bytes
      <<" center_setup_seconds="<<center_setup_seconds
      <<" stencil_bytes="<<bytes<<" coarse_nnz="<<coarse_->NumNonZeroElems()<<" probe_rel="<<probe_error
      <<" separable_reference_rel="<<separable_error<<" eigen_residual="<<eigen_error<<" clipped_eigenvalues="<<clipped
      <<" setup_seconds="<<seconds<<" frozen_operator=sampled-original-galerkin alpha=inherited-fine"
      <<" tensor_execution="<<(precision_==TensorMultigridPrecision::TF32?"requires-profile-proof":"none")<<" fallback=none\n";
  }
  ~TensorMultigrid() override
  {
    std::cout<<"[igr-tensor-mg] event=summary precision="<<Name(precision_)<<" applications="<<applications_
      <<" smoother="<<SmootherName(smoother_)<<" center_warps="<<center_warps_<<" center_split_k="<<center_split_k_
      <<" coarse_iterations="<<coarse_iterations_<<" coarse_failures="<<coarse_failures_<<" coarse_cap_hits="<<coarse_cap_hits_
      <<" transform_submissions="<<transforms_<<" center_submissions="<<centers_
      <<" center_pack_submissions="<<center_packs_<<" center_add_submissions="<<center_adds_
      <<" cycle=multiplicative-v fallback=none\n";
  }
  void SetOperator(const mfem::Operator& op) override
  {MFEM_VERIFY(op.Height()==height&&op.Width()==width,"Tensor MG frozen hierarchy dimension changed; rebuild required.");}
  mfem::MemoryClass GetMemoryClass() const override{return mfem::MemoryClass::DEVICE;}
  void Mult(const mfem::Vector& b,mfem::Vector& x) const override
  {
    MFEM_VERIFY(b.Size()==height&&&b!=&x,"Tensor MG invalid/aliased input.");x.SetSize(height);x.UseDevice(true);
    InputMaximum<<<norm_blocks_,256>>>(height,b.Read(),norm_partial_.Write());
    InputExponent<<<1,1>>>(norm_blocks_,norm_partial_.Read(),exponent_.Write());
    ToModal<<<(height+255)/256,256>>>(ne_,element_map_.Read(),nodal_.Read(),b.Read(),exponent_.Read(),levels_[0].rhs.Write());
    Cycle(0);
    FromModal<<<(height+255)/256,256>>>(ne_,element_map_.Read(),nodal_.Read(),levels_[0].x.Read(),exponent_.Read(),operator_scale_,x.Write());
    CUDA(cudaGetLastError());++applications_;
  }
private:
  void Transform(int s,const mfem::Array<T>& a,const mfem::Array<T>& b,mfem::Array<T>& c) const
  {
    const int columns=ne_*s;
    if constexpr(std::is_same<T,float>::value)
    {
      if(precision_==TensorMultigridPrecision::TF32)
      {const int tiles=((s+15)/16)*((columns+15)/16);IGRTensorMultigridTF32TransformKernel<<<(tiles+3)/4,128>>>(s,columns,a.Read(),b.Read(),c.Write());}
      else SharedTransformSIMT<<<(s*columns+255)/256,256>>>(s,columns,a.Read(),b.Read(),c.Write());
    }
    else SharedTransformSIMT<<<(s*columns+255)/256,256>>>(s,columns,a.Read(),b.Read(),c.Write());
    ++transforms_;
  }
  void Smooth(int l,const mfem::Array<T>& r) const
  {
    const auto& v=levels_[l];const int d=v.d,s=3*d,size=ne_*s*s;
    PackPatch<<<(size+255)/256,256>>>(ne_,d,patches_.Read(),r.Read(),v.patch_a.Write());
    if(smoother_==TensorMultigridSmoother::Center)
    {
      if constexpr(std::is_same<T,float>::value)
        ApplyTensorCenterResponse(ne_,d,v.center_response.Read(),v.patch_a.Read(),v.patch_b.Write(),
            precision_==TensorMultigridPrecision::TF32,center_warps_,center_split_k_);
      else
        ApplyTensorCenterResponse(ne_,d,v.center_response.Read(),v.patch_a.Read(),v.patch_b.Write(),
            center_warps_,center_split_k_);
      AddTensorCenterResponse(ne_,d,v.scale.Read(),v.patch_b.Read(),v.x.ReadWrite(),center_split_k_);
      ++centers_;++center_packs_;++center_adds_;return;
    }
    Transform(s,v.forward_x,v.patch_a,v.patch_b);
    TransposePatch<<<(size+255)/256,256>>>(ne_,s,v.patch_b.Read(),v.patch_a.Write());
    Transform(s,v.forward_y,v.patch_a,v.patch_b);
    DividePatch<<<(size+255)/256,256>>>(ne_,s,v.inverse.Read(),v.patch_b.ReadWrite());
    Transform(s,v.back_y,v.patch_b,v.patch_a);
    TransposePatch<<<(size+255)/256,256>>>(ne_,s,v.patch_a.Read(),v.patch_b.Write());
    Transform(s,v.back_x,v.patch_b,v.patch_a);
    AddCenter<<<(ne_*d*d+255)/256,256>>>(ne_,d,v.scale.Read(),v.patch_a.Read(),v.x.ReadWrite());
  }
  void Cycle(int l) const
  {
    const auto& v=levels_[l];const int n=ne_*v.d*v.d;
    if(l==2)
    {
      Convert<<<(n+255)/256,256>>>(n,v.rhs.Read(),coarse_rhs_.Write());
      coarse_solver_->Mult(coarse_rhs_,coarse_x_);coarse_iterations_+=coarse_solver_->GetNumIterations();
      if(!coarse_solver_->GetConverged())
      {
        // A finite bounded coarse approximation remains valid for flexible
        // outer GMRES. Only a reported convergence has the p1 true-rBr gate;
        // a cap is explicitly counted, never relabeled as certified.
        bool finite=std::isfinite(coarse_solver_->GetFinalNorm()) && coarse_solver_->GetFinalNorm()>=0;
        if(coarse_solver_->GetNumIterations()==100 && finite)
        {
          const double* values=coarse_x_.HostRead();
          for(int i=0;i<n;++i)finite=finite&&std::isfinite(values[i]);
          if(finite)++coarse_cap_hits_;
        }
        else finite=false;
        if(!finite)++coarse_failures_;
        MFEM_VERIFY(finite,"Tensor MG p1 CG breakdown or nonfinite coarse correction.");
      }
      Convert<<<(n+255)/256,256>>>(n,coarse_x_.Read(),v.x.Write());return;
    }
    CUDA(cudaMemsetAsync(v.x.Write(),0,std::size_t(n)*sizeof(T)));
    Smooth(l,v.rhs);
    Residual<<<(n+255)/256,256>>>(v.d*v.d,ne_,neighbors_.Read(),v.a.Read(),v.rhs.Read(),v.x.Read(),v.residual.Write());
    const auto& c=levels_[l+1];const int nc=ne_*c.d*c.d;
    Restrict<<<(nc+255)/256,256>>>(ne_,v.d,c.d,v.residual.Read(),c.rhs.Write());Cycle(l+1);
    ProlongAdd<<<(nc+255)/256,256>>>(ne_,v.d,c.d,c.x.Read(),v.x.ReadWrite());
    Residual<<<(n+255)/256,256>>>(v.d*v.d,ne_,neighbors_.Read(),v.a.Read(),v.rhs.Read(),v.x.Read(),v.residual.Write());
    Smooth(l,v.residual);
  }
  int ne_,norm_blocks_=0;TensorMultigridPrecision precision_;TensorMultigridSmoother smoother_;
  int center_warps_,center_split_k_;double operator_scale_=1;
  std::array<Level<T>,3> levels_;
  mfem::Array<int> neighbors_,patches_,element_map_,essential_;
  mfem::Vector nodal_;
  mutable mfem::Array<int> exponent_;
  mutable mfem::Vector norm_partial_,coarse_rhs_,coarse_x_;
  std::unique_ptr<mfem::SparseMatrix> coarse_;
  std::unique_ptr<mfem::OperatorJacobiSmoother> jacobi_;
  std::unique_ptr<mfem::IterativeSolver> coarse_solver_;
  bool cooperative_=false;
  mutable unsigned long long applications_=0,coarse_iterations_=0,coarse_failures_=0,coarse_cap_hits_=0,transforms_=0;
  mutable unsigned long long centers_=0,center_packs_=0,center_adds_=0;
};
} // namespace

void ApplyTensorCenterResponse(int ne,int d,const double* response,
    const double* packed,double* partial,int warps,int split_k)
{
  CheckCenterLayout(ne,d,warps,split_k);
  MFEM_VERIFY(response&&packed&&partial,"Null tensor center response storage.");
  const int threads=32*warps;
  IGRTensorMultigridCenterSIMTKernel<<<dim3((ne*d*d+threads-1)/threads,split_k),threads>>>(
      ne,d,response,packed,partial,split_k);
  CUDA(cudaGetLastError());
}

void ApplyTensorCenterResponse(int ne,int d,const float* response,
    const float* packed,float* partial,bool tf32,int warps,int split_k)
{
  CheckCenterLayout(ne,d,warps,split_k);
  MFEM_VERIFY(response&&packed&&partial,"Null tensor center response storage.");
  if(tf32)
  {
    const int tiles_m=(d*d+15)/16,groups=((ne+15)/16+warps-1)/warps;
    const dim3 grid(tiles_m*groups,split_k);
    if(warps==1)IGRTensorMultigridCenterTF32Kernel<1><<<grid,32>>>(ne,d,response,packed,partial,split_k);
    else IGRTensorMultigridCenterTF32Kernel<4><<<grid,128>>>(ne,d,response,packed,partial,split_k);
  }
  else
  {
    const int threads=32*warps;
    IGRTensorMultigridCenterSIMTKernel<<<dim3((ne*d*d+threads-1)/threads,split_k),threads>>>(
        ne,d,response,packed,partial,split_k);
  }
  CUDA(cudaGetLastError());
}

void AddTensorCenterResponse(int ne,int d,const double* factor,
    const double* partial,double* x,int split_k)
{
  CheckCenterLayout(ne,d,1,split_k);
  MFEM_VERIFY(factor&&partial&&x,"Null tensor center add storage.");
  IGRTensorMultigridAddCenterKernel<<<(ne*d*d+255)/256,256>>>(ne,d,factor,partial,x,split_k);
  CUDA(cudaGetLastError());
}

void AddTensorCenterResponse(int ne,int d,const float* factor,
    const float* partial,float* x,int split_k)
{
  CheckCenterLayout(ne,d,1,split_k);
  MFEM_VERIFY(factor&&partial&&x,"Null tensor center add storage.");
  IGRTensorMultigridAddCenterKernel<<<(ne*d*d+255)/256,256>>>(ne,d,factor,partial,x,split_k);
  CUDA(cudaGetLastError());
}

std::unique_ptr<mfem::Solver> MakeTensorMultigridPreconditioner(
    const mfem::Operator& op,mfem::ParFiniteElementSpace& sfes,TensorMultigridPrecision precision,
    TensorMultigridSmoother smoother,int center_warps,int center_split_k)
{
  MFEM_VERIFY(smoother==TensorMultigridSmoother::Separable||smoother==TensorMultigridSmoother::Center,
              "Invalid tensor MG smoother.");
  CheckCenterLayout(sfes.GetNE(),6,center_warps,center_split_k);
  MFEM_VERIFY(smoother==TensorMultigridSmoother::Center||(center_warps==1&&center_split_k==1),
              "Tensor center layout options require the center smoother.");
  if(precision==TensorMultigridPrecision::FP64)
    return std::make_unique<TensorMultigrid<double>>(op,sfes,precision,smoother,center_warps,center_split_k);
  MFEM_VERIFY(precision==TensorMultigridPrecision::FP32||precision==TensorMultigridPrecision::TF32,"Invalid tensor MG precision.");
  return std::make_unique<TensorMultigrid<float>>(op,sfes,precision,smoother,center_warps,center_split_k);
}
} }

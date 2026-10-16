#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
using namespace dg::hdg::runtime::cuda_w8a;
using namespace dg::hdg::runtime::cuda_w8a::detail;
struct Arena
{
  std::vector<void*> buffers;
  void* vendor_handle=nullptr;
  ~Arena()
  {
    CudaDestroyRefinedCublasHandle(vendor_handle);
    for(void* p:buffers)cudaFree(p);
  }
  template <typename T=double> T* Get(std::size_t n)
  {
    T* p=nullptr;
    if(cudaMallocManaged(reinterpret_cast<void**>(&p),n*sizeof(T))!=cudaSuccess)
      throw "allocation failed";
    buffers.push_back(p);std::fill(p,p+n,T{});return p;
  }
};

bool BindBackend(Arena& arena,CudaDeviceView& v,const CudaMeshShape& s)
{
  if(s.local_solve_backend==CudaLocalSolveBackend::simt)return true;
  v.local_solve_vendor_workspace=arena.Get(4*s.elements+1);
  if(!CudaCreateRefinedCublasHandle(arena.vendor_handle))return false;
  v.local_solve_vendor_handle=arena.vendor_handle;
  return true;
}

bool Run(const int n,const CudaLocalSolveKind kind,
         const CudaLocalSolveBackend backend,const int nt=3)
{
  constexpr int ne=3;
  const int cols=nt+1;
  Arena arena;
  CudaDeviceView v{};
  CudaMeshShape s{};
  s.elements=ne;s.interior_dofs=n;s.trace_dofs=nt;
  s.interior_count=ne*n;s.borrowed_host_inputs=true;s.local_solve=kind;
  s.local_solve_backend=backend;
  if(!BindBackend(arena,v,s))return false;
  s.true_trace_count=ne*nt;s.global_true_trace_count=ne*nt;
  s.vector_owned_count=ne*nt;
  v.signed_element_trace=arena.Get<int>(ne*nt);
  v.vector_owned_source=arena.Get<int>(ne*nt);
  v.vector_owned_target=arena.Get<int>(ne*nt);
  for(int i=0;i<ne*nt;++i)
    v.signed_element_trace[i]=v.vector_owned_source[i]=v.vector_owned_target[i]=i;
  v.a=arena.Get(ne*n*n);v.lu=arena.Get(ne*n*n);v.pivots=arena.Get(ne*n);
  v.b=arena.Get(ne*n*nt);v.c=arena.Get(ne*nt*n);v.d=arena.Get(ne*nt*nt);
  v.f=arena.Get(ne*n);v.g=arena.Get(ne*nt);v.schur=arena.Get(ne*nt*nt);
  v.condensed_rhs=arena.Get(ne*nt);v.solve_workspace=arena.Get(ne*n*cols);
  v.local_solve_workspace=arena.Get(ne*n*cols);v.local_solve_work=arena.Get(ne*3);
  v.status=arena.Get(ne);v.scalar=arena.Get(3);
  v.recovered_interior=arena.Get(ne*n);v.trace_increment_local=arena.Get(ne*nt);
  v.residual_r=arena.Get(ne*n);v.residual_g=arena.Get(ne*nt);
  std::fill(v.residual_r,v.residual_r+ne*n,17.0);
  std::fill(v.residual_g,v.residual_g+ne*nt,19.0);
  std::vector<double> exact(ne*n*cols);
  for(int e=0;e<ne;++e)
  {
    for(int row=0;row<n;++row)
      for(int c=0;c<cols;++c)exact[(e*n+row)*cols+c]=0.125*((row+c)%7-3);
    for(int row=0;row<n;++row)
      for(int j=0;j<n;++j)
        v.a[(e*n+row)*n+j]=e==0?(row==j?2.0:0.03125*((row+j)%3-1)):
          (row==j?(e==2?1.e40:1.0):0.0);
    if(e==0)
      for(int j=0;j<n;++j)std::swap(v.a[j],v.a[(n-1)*n+j]);
    if(e==1)
    {
      // Invertible in FP64, exactly singular after rounding to FP32.
      v.a[e*n*n+1]=1.0;v.a[e*n*n+n]=1.0;
      v.a[e*n*n+n+1]=1.0+std::ldexp(1.0,-25);
    }
    for(int row=0;row<n;++row)
      for(int c=0;c<cols;++c)
      {
        double value=0.0;
        for(int j=0;j<n;++j)value+=v.a[(e*n+row)*n+j]*exact[(e*n+j)*cols+c];
        if(c<nt)v.b[(e*n+row)*nt+c]=value;else v.f[e*n+row]=value;
      }
    for(int row=0;row<nt;++row)
    {
      v.g[e*nt+row]=0.25*(row+1);
      v.trace_increment_local[e*nt+row]=0.0625*(row+1);
      for(int j=0;j<n;++j)v.c[(e*nt+row)*n+j]=0.0625*((row+j)%3-1);
      for(int j=0;j<nt;++j)v.d[(e*nt+row)*nt+j]=row==j?3.0:0.125;
    }
  }
  std::uint64_t tiles=999;
  if(!CudaLaunchCondensation(v,s,dg::hdg::generic::CudaEngineKind::ordinary,tiles)||
     cudaDeviceSynchronize()!=cudaSuccess||tiles!=0)return false;
  double maximum_backward_error=0.0;
  for(int e=0;e<ne;++e)
  {
    if(v.status[e]!=1.0||v.local_solve_work[3*e+2]!=1.0)return false;
    if(kind==CudaLocalSolveKind::fp32_refine&&
       v.local_solve_work[3*e+1]!=(e==0?0.0:1.0))return false;
    for(int row=0;row<n;++row)
      for(int c=0;c<cols;++c)
      {
        double residual=c<nt?v.b[(e*n+row)*nt+c]:v.f[e*n+row];
        double scale=std::abs(residual);
        for(int j=0;j<n;++j)
        {
          const double term=v.a[(e*n+row)*n+j]*v.solve_workspace[(e*n+j)*cols+c];
          residual-=term;scale+=std::abs(term);
        }
        maximum_backward_error=std::max(maximum_backward_error,
            scale>0.0?std::abs(residual)/scale:std::abs(residual));
      }
    for(int row=0;row<nt;++row)
      for(int c=0;c<cols;++c)
      {
        double expected=c<nt?v.d[(e*nt+row)*nt+c]:v.g[e*nt+row];
        for(int j=0;j<n;++j)expected-=v.c[(e*nt+row)*n+j]*exact[(e*n+j)*cols+c];
        const double actual=c<nt?v.schur[(e*nt+row)*nt+c]:v.condensed_rhs[e*nt+row];
        if(std::abs(actual-expected)>1.e-11*(1.0+std::abs(expected)))return false;
      }
  }
  if(maximum_backward_error>1.e-13 ||
     (kind==CudaLocalSolveKind::fp32_refine&&v.local_solve_work[0]<1.0))return false;
  if(!CudaLaunchRecover(v,s,dg::hdg::generic::CudaEngineKind::ordinary)||
     !CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)||
     cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int e=0;e<ne;++e)
    for(int row=0;row<n;++row)
    {
      double expected=exact[(e*n+row)*cols+nt];
      for(int c=0;c<nt;++c)expected-=exact[(e*n+row)*cols+c]*v.trace_increment_local[e*nt+c];
      if(std::abs(v.recovered_interior[e*n+row]-expected)>1.e-11)return false;
    }
  for(int i=0;i<ne*n;++i)if(v.residual_r[i]!=17.0)return false;
  for(int i=0;i<ne*nt;++i)if(v.residual_g[i]!=19.0)return false;
  // Residual checking must use A/B/C/D, not reconstruct the rounded Schur.
  // This fixture has independent trace rows, so true assembly is identity.
  for(int e=0;e<ne;++e)
    for(int row=0;row<nt;++row)
    {
      double expected=v.g[e*nt+row];
      for(int j=0;j<n;++j)expected-=v.c[(e*nt+row)*n+j]*v.recovered_interior[e*n+j];
      for(int j=0;j<nt;++j)expected-=v.d[(e*nt+row)*nt+j]*v.trace_increment_local[e*nt+j];
      if(std::abs(v.local_solve_workspace[ne*n+ne*nt+e*nt+row]-expected)>1.e-12)return false;
    }
  // Nonfinite original A must fail even after the bounded FP64 fallback.
  v.a[0]=std::numeric_limits<double>::quiet_NaN();
  if(!CudaLaunchCondensation(v,s,dg::hdg::generic::CudaEngineKind::ordinary,tiles)||
     cudaDeviceSynchronize()!=cudaSuccess||v.status[0]!=-1.0)return false;
  std::cout<<"refined-local n="<<n<<" nrhs="<<cols
           <<" cublas="<<(backend==CudaLocalSolveBackend::cublas_batched)
           <<" mixed="<<(kind==CudaLocalSolveKind::fp32_refine)
           <<" max_backward_error="<<maximum_backward_error<<" passed\n";
  return true;
}

// Exercise the production rank-one RHS assembly and recovered coupled gate.
// Shared trace rows have large cancelling g contributions near convergence;
// local subtraction loses ~1e-16 although the assembled RHS is only 2^-40.
bool SharedTraceCancellation(const CudaLocalSolveKind kind,
                             const dg::hdg::generic::CudaEngineKind engine,
                             const CudaLocalSolveBackend backend)
{
  constexpr int ne=2,n=3,nt=1,cols=2;
  const double delta=std::ldexp(1.0,-40),third=delta/3.0;
  Arena arena;
  CudaDeviceView v{};CudaMeshShape s{};
  s.elements=ne;s.interior_dofs=n;s.trace_dofs=nt;s.interior_count=ne*n;
  s.borrowed_host_inputs=true;s.local_solve=kind;
  s.local_solve_backend=backend;
  if(!BindBackend(arena,v,s))return false;
  s.true_trace_count=1;s.global_true_trace_count=1;s.vector_owned_count=ne;
  v.a=arena.Get(ne*n*n);v.lu=arena.Get(ne*n*n);v.pivots=arena.Get(ne*n);
  v.b=arena.Get(ne*n*nt);v.c=arena.Get(ne*nt*n);v.d=arena.Get(ne*nt*nt);
  v.f=arena.Get(ne*n);v.g=arena.Get(ne*nt);v.schur=arena.Get(ne*nt*nt);
  v.condensed_rhs=arena.Get(ne*nt);v.solve_workspace=arena.Get(ne*n*cols);
  v.local_solve_workspace=arena.Get(ne*n*cols);v.local_solve_work=arena.Get(ne*3);
  v.status=arena.Get(ne);v.scalar=arena.Get(3);v.true_rhs_scratch=arena.Get(1);
  v.recovered_interior=arena.Get(ne*n);v.trace_increment_local=arena.Get(ne*nt);
  v.signed_element_trace=arena.Get<int>(ne);
  v.vector_owned_source=arena.Get<int>(ne);v.vector_owned_target=arena.Get<int>(ne);
  v.signed_element_trace[0]=0;v.signed_element_trace[1]=-1;
  v.vector_owned_source[0]=0;v.vector_owned_source[1]=1;
  for(int e=0;e<ne;++e)
  {
    for(int row=0;row<n;++row)v.a[(e*n+row)*n+row]=1.0;
    v.d[e]=1.0;
  }
  v.f[0]=third;v.f[1]=third;v.f[n]=third;
  v.c[0]=1.0;v.c[1]=1.0;v.c[n]=-1.0;
  v.g[0]=1.0;v.g[1]=1.0-delta;
  std::uint64_t tiles=0;
  if(!CudaLaunchCondensation(v,s,engine,tiles)||
     !CudaLaunchRefinedTrueRhs(v,s)||cudaDeviceSynchronize()!=cudaSuccess)
    return false;
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
  const double legacy_rhs=v.condensed_rhs[0]-v.condensed_rhs[1];
  const double corrected_rhs=v.true_rhs_scratch[0];
  const double true_schur=v.schur[0]+v.schur[1];
  if(!(std::abs(legacy_rhs)>1.e-6*delta)||
     std::abs(corrected_rhs)>1.e-10*delta||true_schur!=2.0)return false;
  const auto recover_and_measure=[&](const double rhs,double& relative)
  {
    v.trace_increment_local[0]=rhs/true_schur;
    v.trace_increment_local[1]=-rhs/true_schur;
    if(!CudaLaunchRecover(v,s,engine)||
       !CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)||
       cudaDeviceSynchronize()!=cudaSuccess||!(v.scalar[1]>0.0))return false;
    relative=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
    return std::isfinite(relative)&&v.true_rhs_scratch[0]==corrected_rhs;
  };
  double correct_relative=0.0,legacy_relative=0.0;
  if(!recover_and_measure(corrected_rhs,correct_relative)||correct_relative>1.e-10||
     !recover_and_measure(legacy_rhs,legacy_relative)||legacy_relative<=1.e-10)
    return false;
  // Reject an erroneous interior recovery as well as the inaccurate trace RHS.
  if(!recover_and_measure(corrected_rhs,correct_relative))return false;
  v.recovered_interior[0]+=delta*1.e-3;
  if(!CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)||
     cudaDeviceSynchronize()!=cudaSuccess)return false;
  const double perturbed_relative=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
  if(!std::isfinite(perturbed_relative)||perturbed_relative<=1.e-10)return false;
  std::cout<<"shared-trace-cancellation cublas="
           <<(backend==CudaLocalSolveBackend::cublas_batched)<<" mixed="<<(kind==CudaLocalSolveKind::fp32_refine)
           <<" tensor="<<(engine==dg::hdg::generic::CudaEngineKind::tensor_core)
           <<" corrected_relative="<<correct_relative
           <<" legacy_relative="<<legacy_relative<<" passed\n";
  return true;
}
}
int main()
{
  try
  {
    for(const auto backend:{CudaLocalSolveBackend::simt,CudaLocalSolveBackend::cublas_batched})
      for(const auto kind:{CudaLocalSolveKind::fp64_refine,CudaLocalSolveKind::fp32_refine})
      {
        if(!Run(5,kind,backend)||!Run(17,kind,backend))return 1;
        // Production IGR p1/p7 shapes include nrhs>n and nrhs>blockDim.
        if(backend==CudaLocalSolveBackend::cublas_batched&&
           (!Run(28,kind,backend,40)||!Run(448,kind,backend,160)))return 1;
        for(const auto engine:{dg::hdg::generic::CudaEngineKind::ordinary,
                               dg::hdg::generic::CudaEngineKind::tensor_core})
          if(!SharedTraceCancellation(kind,engine,backend))return 1;
      }
    return 0;
  }
  catch(const char* error){std::cerr<<error<<'\n';return 2;}
}

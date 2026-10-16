#include "numerics/hdg/runtime/cuda/persistent_smooth_euler.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace dg::hdg::runtime::cuda_w8a::detail
{
// Test-only linked entry point; no production public API is exposed.
bool CudaTestRefinedTriangularSolve(const double*,double*,double*,const double*,
    double*,double*,int,int,bool,bool) noexcept;
}

namespace
{
using namespace dg::hdg::runtime::cuda_w8a;
using namespace dg::hdg::runtime::cuda_w8a::detail;
constexpr auto coupled_trace_policy = ResolveCudaTracePolicy(
    dg::hdg::generic::CudaEngineKind::ordinary,true,1,5,
    CudaTracePolicyKind::ordinary_ilu_jacobi5_refresh);
static_assert(CudaCoupledTracePolicyIsExact(coupled_trace_policy));
static_assert(!CudaOrdinaryRankOneRefreshPolicyIsExact(coupled_trace_policy));
static_assert(!CudaCoupledTracePolicyIsExact(ResolveCudaTracePolicy(
    dg::hdg::generic::CudaEngineKind::ordinary,true,1,5,
    CudaTracePolicyKind::engine_default)));
static_assert(!CudaCoupledTracePolicyIsExact(ResolveCudaTracePolicy(
    dg::hdg::generic::CudaEngineKind::ordinary,true,2,5,
    CudaTracePolicyKind::ordinary_ilu_jacobi5_refresh)));

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

bool TolerancePolicies()
{
  const auto fp64=CudaLocalSolveKind::coupled_krylov_fp64;
  const auto once=CudaLocalSolveKind::coupled_krylov_fp32_once;
  return CudaCoupledAcceptanceTolerance(CudaCoupledAccuracyKind::strict,1.e-10)==1.e-10&&
      CudaCoupledAcceptanceTolerance(CudaCoupledAccuracyKind::physical_1e6,1.e-10)==1.e-6&&
      CudaCoupledAcceptanceTolerance(CudaCoupledAccuracyKind::physical_1e4,1.e-10)==1.e-4&&
      std::isnan(CudaCoupledAcceptanceTolerance(CudaCoupledAccuracyKind::invalid,1.e-10))&&
      CudaCoupledInnerTolerance(fp64,CudaCoupledAccuracyKind::strict,1.e-10)==1.e-12&&
      std::abs(CudaCoupledInnerTolerance(fp64,CudaCoupledAccuracyKind::physical_1e6,1.e-6)-1.e-8)<1.e-22&&
      std::abs(CudaCoupledInnerTolerance(fp64,CudaCoupledAccuracyKind::physical_1e4,1.e-4)-1.e-6)<1.e-20&&
      CudaCoupledInnerTolerance(once,CudaCoupledAccuracyKind::strict,1.e-10)==1.e-6&&
      CudaCoupledInnerTolerance(once,CudaCoupledAccuracyKind::physical_1e4,1.e-4)==1.e-6;
}

bool FewRhsTriangularSolve()
{
  for(const int n:{17,65})for(const int columns:{1,2,3,4,7})
  {
    Arena arena;double* a=arena.Get(n*n);double* lu=arena.Get(n*n);
    double* pivots=arena.Get(n);double* rhs=arena.Get(n*columns);
    double* workspace=arena.Get(n*columns);double* result=arena.Get(n*columns);
    std::vector<double> exact(n*columns);
    for(int i=0;i<n*columns;++i)exact[i]=0.1*std::sin(0.17*(i+1));
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)
      a[i*n+j]=i==j?3.0:0.015*std::cos(0.23*(i+1)*(j+2));
    // Force a long pivot permutation, and exercise non-warp-sized rows.
    for(int j=0;j<n;++j)std::swap(a[j],a[(n-1)*n+j]);
    for(int i=0;i<n;++i)for(int c=0;c<columns;++c)
      for(int j=0;j<n;++j)rhs[i*columns+c]+=a[i*n+j]*exact[j*columns+c];
    const std::vector<double> original_a(a,a+n*n),original_rhs(rhs,rhs+n*columns);
    for(const bool fp32:{false,true})for(const bool cooperative:{false,true})
    {
      if(!CudaTestRefinedTriangularSolve(a,lu,pivots,rhs,workspace,result,n,columns,fp32,cooperative)||
          cudaDeviceSynchronize()!=cudaSuccess)return false;
      double error=0,backward=0;
      for(int i=0;i<n;++i)for(int c=0;c<columns;++c)
      {
        double residual=rhs[i*columns+c],scale=std::abs(residual);
        for(int j=0;j<n;++j){const double term=a[i*n+j]*result[j*columns+c];residual-=term;scale+=std::abs(term);}
        if(!std::isfinite(result[i*columns+c])||!std::isfinite(residual))return false;
        backward=std::max(backward,std::abs(residual)/std::max(scale,1.e-300));
        error=std::max(error,std::abs(result[i*columns+c]-exact[i*columns+c]));
      }
      if(backward>(fp32?2.e-6:2.e-13)||error>(fp32?2.e-6:2.e-13)||
          !std::equal(original_a.begin(),original_a.end(),a)||
          !std::equal(original_rhs.begin(),original_rhs.end(),rhs))
      {
        std::cerr<<"triangular fixture failed n="<<n<<" rhs="<<columns
            <<" fp32="<<fp32<<" cooperative="<<cooperative
            <<" backward="<<backward<<" solution_maxabs="<<error<<'\n';
        return false;
      }
    }
  }
  std::cout<<"refined-triangular single_few_many_rhs=passed fp64_fp32=passed pivoted_tail=passed original_immutable=passed\n";
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
// Small signed shared-face coupled system exercises the production preconditioner,
// original residual refresh, accumulation, and two-RHS final certificate. The
// tiny host trace solve replaces only HYPRE; the application screen covers it.
std::vector<double> DenseSolve(std::vector<double> a,std::vector<double> rhs,int n)
{
  for(int k=0;k<n;++k)
  {
    int pivot=k;for(int i=k+1;i<n;++i)if(std::abs(a[i*n+k])>std::abs(a[pivot*n+k]))pivot=i;
    if(!(std::abs(a[pivot*n+k])>1.e-14))throw "singular fixture";
    for(int j=0;j<n;++j)std::swap(a[k*n+j],a[pivot*n+j]);std::swap(rhs[k],rhs[pivot]);
    for(int i=k+1;i<n;++i)
    {const double t=a[i*n+k]/a[k*n+k];for(int j=k+1;j<n;++j)a[i*n+j]-=t*a[k*n+j];rhs[i]-=t*rhs[k];}
  }
  for(int i=n-1;i>=0;--i){for(int j=i+1;j<n;++j)rhs[i]-=a[i*n+j]*rhs[j];rhs[i]/=a[i*n+i];}
  return rhs;
}
// A periodic ring with one anchor has a nonsingular assembled trace operator,
// although every unanchored local Schur block has a constant null mode. Each
// two-trace patch therefore needs its neighbors' diagonal contributions. The
// old own-element shifted patch stalls with the production restart/budget.
bool CoupledPartialOverlapRing(const CudaCoupledApplyKind apply=CudaCoupledApplyKind::serial)
{
  constexpr int ne=32,n=3,m=2,nt=32,np=16,mp=16,q=16,ni=ne*n,N=ni+nt,l=n+m;
  Arena arena;CudaDeviceView v{};CudaMeshShape s{};
  s.elements=ne;s.interior_dofs=n;s.trace_dofs=m;s.interior_count=ni;
  s.borrowed_host_inputs=true;s.local_solve=CudaLocalSolveKind::coupled_krylov_fp64;
  s.coupled_apply=apply;
  s.local_solve_backend=CudaLocalSolveBackend::cublas_batched;
  s.true_trace_count=nt;s.global_true_trace_count=nt;s.vector_owned_count=ne*m;
  const auto fail=[](const char* stage)
  {std::cerr<<"coupled partial-overlap ring failed stage="<<stage<<'\n';return false;};
  if(!BindBackend(arena,v,s))return fail("backend");
  v.a=arena.Get(ne*n*n);v.lu=arena.Get(ne*n*n);v.pivots=arena.Get(ne*n);
  v.b=arena.Get(ne*n*m);v.c=arena.Get(ne*m*n);v.d=arena.Get(ne*m*m);
  v.f=arena.Get(ni);v.g=arena.Get(ne*m);v.schur=arena.Get(ne*m*m);
  v.condensed_rhs=arena.Get(ne*m);v.solve_workspace=arena.Get(ni*(m+1));
  v.local_solve_workspace=arena.Get(ni*(m+1));v.local_solve_work=arena.Get(ne*3);
  v.status=arena.Get(ne);v.scalar=arena.Get(3);v.true_rhs_scratch=arena.Get(nt);
  v.true_solution=arena.Get(nt);v.recovered_interior=arena.Get(ni);
  v.trace_increment_local=arena.Get(ne*m);
  v.signed_element_trace=arena.Get<int>(ne*m);v.vector_owned_source=arena.Get<int>(ne*m);
  v.vector_owned_target=arena.Get<int>(ne*m);
  v.coupled_lu=arena.Get(ne*np*np);v.coupled_rhs=arena.Get(ne*np*q);
  v.coupled_c=arena.Get(ne*mp*np);v.coupled_product=arena.Get(ne*mp*mp);
  v.coupled_pivots=arena.Get(ne*np);v.coupled_row_exponents=arena.Get(ne*np);
  v.coupled_interior_total=arena.Get(ni);v.coupled_trace_total=arena.Get(nt);
  v.coupled_base_rhs=arena.Get(nt);v.coupled_trace_residual=arena.Get(nt);
  v.coupled_diagnostics=arena.Get(8);
  v.coupled_krylov_workspace=arena.Get(ne*(l*l+m*m+2*m+np)+53*N+nt+128*25+3*26+1);

  const std::vector<double> a={2.0,0.25,0.0,0.0,4.0,0.5,0.0,0.0,8.0};
  const double b[n*m]={1./16,-1./16,1./16,1./16,0.0,1./16};
  const double c[m*n]={1./32,0.0,-1./32,-1./32,1./32,0.0};
  double wb[n*m]{};
  for(int j=0;j<m;++j)
  {
    std::vector<double> rhs(n);for(int i=0;i<n;++i)rhs[i]=b[i*m+j];
    const auto column=DenseSolve(a,rhs,n);
    for(int i=0;i<n;++i)wb[i*m+j]=column[i];
  }
  std::vector<double> exact(N),signs(ne*m);
  std::vector<int> truth(ne*m);
  for(int i=0;i<N;++i)exact[i]=((17*i)%31-15)/16.0;
  for(int e=0;e<ne;++e)
  {
    int permutation[m]={e%2,1-e%2};
    double d[m*m]{};
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)
    {
      d[i*m+j]=(1.0+(e%7)/8.0)*(i==j?0.5:-0.5);
      if(e==0&&i==0&&j==0)d[i*m+j]+=1.0;
      for(int k=0;k<n;++k)d[i*m+j]+=c[i*n+k]*wb[k*m+j];
    }
    for(int j=0;j<m;++j)
    {
      const int slot=e*m+j,t=(e+permutation[j])%nt;
      const double sign=(e+j)%3==0?-1.0:1.0;
      truth[slot]=t;signs[slot]=sign;
      v.signed_element_trace[slot]=sign<0?-1-t:t;
      v.vector_owned_source[slot]=slot;v.vector_owned_target[slot]=t;
    }
    for(int i=0;i<n;++i)
    {
      for(int j=0;j<n;++j)
      {v.a[(e*n+i)*n+j]=a[i*n+j];v.f[e*n+i]+=a[i*n+j]*exact[e*n+j];}
      for(int j=0;j<m;++j)
      {
        const double value=b[i*m+permutation[j]]*signs[e*m+j];
        v.b[(e*n+i)*m+j]=value;
        v.f[e*n+i]+=value*signs[e*m+j]*exact[ni+truth[e*m+j]];
      }
    }
    for(int i=0;i<m;++i)
    {
      for(int j=0;j<n;++j)
      {
        const double value=signs[e*m+i]*c[permutation[i]*n+j];
        v.c[(e*m+i)*n+j]=value;v.g[e*m+i]+=value*exact[e*n+j];
      }
      for(int j=0;j<m;++j)
      {
        const double value=signs[e*m+i]*d[permutation[i]*m+permutation[j]]*signs[e*m+j];
        v.d[(e*m+i)*m+j]=value;
        v.g[e*m+i]+=value*signs[e*m+j]*exact[ni+truth[e*m+j]];
      }
    }
  }
  const std::vector<double> original_a(v.a,v.a+ne*n*n),original_b(v.b,v.b+ne*n*m),
      original_c(v.c,v.c+ne*m*n),original_d(v.d,v.d+ne*m*m),
      original_f(v.f,v.f+ni),original_g(v.g,v.g+ne*m);
  // Assemble the complete original system independently of GPU Schur products,
  // patch assembly, row equilibration, factors, and Krylov vectors.
  std::vector<double> matrix(N*N),rhs(N);
  for(int e=0;e<ne;++e)
  {
    for(int i=0;i<n;++i)
    {
      const int row=e*n+i;rhs[row]=original_f[row];
      for(int j=0;j<n;++j)matrix[row*N+e*n+j]=original_a[(e*n+i)*n+j];
      for(int j=0;j<m;++j)
        matrix[row*N+ni+truth[e*m+j]]+=original_b[(e*n+i)*m+j]*signs[e*m+j];
    }
    for(int i=0;i<m;++i)
    {
      const int row=ni+truth[e*m+i];const double sign=signs[e*m+i];
      rhs[row]+=sign*original_g[e*m+i];
      for(int j=0;j<n;++j)matrix[row*N+e*n+j]+=sign*original_c[(e*m+i)*n+j];
      for(int j=0;j<m;++j)
        matrix[row*N+ni+truth[e*m+j]]+=sign*original_d[(e*m+i)*m+j]*signs[e*m+j];
    }
  }
  const auto oracle=DenseSolve(matrix,rhs,N);
  for(int i=0;i<N;++i)if(std::abs(oracle[i]-exact[i])>1.e-11)return fail("CPU manufactured oracle");
  std::uint64_t tiles=0;
  if(!CudaLaunchCondensation(v,s,dg::hdg::generic::CudaEngineKind::ordinary,tiles)||
     !CudaLaunchRefinedTrueRhs(v,s)||!CudaLaunchCoupledKrylovSetup(v,s)||
     cudaDeviceSynchronize()!=cudaSuccess)return fail("setup");
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return fail("factor status");
  if(!CudaLaunchCoupledKrylovInitialize(v,s))return fail("initialize");
  const auto defects=[&]()
  {
    return CudaLaunchGatherLocalTrace(v.signed_element_trace,ne*m,v.true_solution,v.trace_increment_local)&&
        CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)&&cudaDeviceSynchronize()==cudaSuccess;
  };
  if(!defects())return fail("initial original residual");
  CudaCoupledKrylovResult result{};
  if(!CudaSolveCoupledNativeKrylov(v,s,1.e-12,144,result)||!defects())return fail("native solve");
  const double coupled=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
  if(!CudaLaunchCoupledCertificateSolve(v,s,false)||cudaDeviceSynchronize()!=cudaSuccess)
    return fail("original certificate solve");
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return fail("certificate factor status");
  if(!CudaLaunchCoupledCertificateNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess)
    return fail("original certificate norm");
  const double condensed=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
  std::vector<double> actual(N);
  std::copy(v.coupled_interior_total,v.coupled_interior_total+ni,actual.begin());
  std::copy(v.true_solution,v.true_solution+nt,actual.begin()+ni);
  long double error2=0,rhs2=0;double solution_error=0;
  for(int i=0;i<N;++i)
  {
    long double residual=rhs[i];
    for(int j=0;j<N;++j)residual-=static_cast<long double>(matrix[i*N+j])*actual[j];
    error2+=residual*residual;rhs2+=static_cast<long double>(rhs[i])*rhs[i];
    solution_error=std::max(solution_error,std::abs(actual[i]-oracle[i]));
  }
  const double independent=std::sqrt(static_cast<double>(error2/rhs2));
  std::cout<<"coupled-native-partial-overlap-ring apply="<<CudaCoupledApplyName(apply)<<" iterations="<<result.iterations
      <<" converged="<<result.converged<<" inner_relative="<<result.relative_residual
      <<" original_coupled="<<coupled<<" original_condensed="<<condensed
      <<" CPU_original_relative="<<independent<<" CPU_solution_maxabs="<<solution_error<<'\n';
  if(!result.converged||result.iterations<=0||result.iterations>144||
     result.preconditioner_applications!=result.iterations||
     !std::isfinite(result.relative_residual)||result.relative_residual>1.e-12||
     !std::isfinite(coupled)||coupled>1.e-10||!std::isfinite(condensed)||condensed>1.e-10||
     !std::isfinite(independent)||independent>1.e-10||!std::isfinite(solution_error)||solution_error>2.e-9)
    return fail("strict original acceptance without recovery");
  if(!std::equal(original_a.begin(),original_a.end(),v.a)||!std::equal(original_b.begin(),original_b.end(),v.b)||
     !std::equal(original_c.begin(),original_c.end(),v.c)||!std::equal(original_d.begin(),original_d.end(),v.d)||
     !std::equal(original_f.begin(),original_f.end(),v.f)||!std::equal(original_g.begin(),original_g.end(),v.g))
    return fail("immutable originals");

  // Invalid topology must fail before any patch can be used. Exercise missing
  // source coverage, duplicate local slots, and an unsupported third owner.
  --s.vector_owned_count;
  if(CudaLaunchCoupledKrylovSetup(v,s))return fail("missing coverage admission");
  ++s.vector_owned_count;
  const auto rejected_map=[&]()
  {
    if(!CudaLaunchCoupledKrylovSetup(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
    for(int e=0;e<ne;++e)if(!std::isfinite(v.status[e])||v.status[e]>=0.0)return false;
    return true;
  };
  const int saved_source=v.vector_owned_source[0];
  v.vector_owned_source[0]=v.vector_owned_source[1];
  if(!rejected_map())return fail("duplicate source rejection");
  v.vector_owned_source[0]=saved_source;
  constexpr int moved_slot=2*m;
  const int saved_target=v.vector_owned_target[moved_slot],saved_signed=v.signed_element_trace[moved_slot];
  v.vector_owned_target[moved_slot]=0;
  v.signed_element_trace[moved_slot]=signs[moved_slot]<0?-1:0;
  if(!rejected_map())return fail("third owner rejection");
  v.vector_owned_target[moved_slot]=saved_target;v.signed_element_trace[moved_slot]=saved_signed;
  // Inverse setup consumes the fresh factor. Rebuild it before a new setup;
  // normal Krylov applications and certificates never rebuild that inverse.
  if(!CudaLaunchCondensation(v,s,dg::hdg::generic::CudaEngineKind::ordinary,tiles)||
      !CudaLaunchCoupledKrylovSetup(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return fail("map reset");
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return fail("map reset status");
  std::cout<<"coupled-native-partial-overlap-ring signed_permuted_trace=passed CPU_dense_oracle=passed"
      <<" original_immutable=passed invalid_maps=passed fp64_recovery=0 passed\n";
  return true;
}

bool CoupledSystem(const CudaLocalSolveKind kind,
    const CudaCoupledApplyKind apply=CudaCoupledApplyKind::serial)
{
  constexpr int ne=2,n=65,m=3,np=80,mp=16,q=16;
  Arena arena;CudaDeviceView v{};CudaMeshShape s{};
  s.elements=ne;s.interior_dofs=n;s.trace_dofs=m;s.interior_count=ne*n;
  s.borrowed_host_inputs=true;s.local_solve=kind;s.local_solve_backend=CudaLocalSolveBackend::cublas_batched;
  s.coupled_apply=apply;
  s.true_trace_count=m;s.global_true_trace_count=m;s.vector_owned_count=ne*m;
  if(!BindBackend(arena,v,s))return false;
  v.a=arena.Get(ne*n*n);v.lu=arena.Get(ne*n*n);v.pivots=arena.Get(ne*n);
  v.b=arena.Get(ne*n*m);v.c=arena.Get(ne*m*n);v.d=arena.Get(ne*m*m);
  v.f=arena.Get(ne*n);v.g=arena.Get(ne*m);v.schur=arena.Get(ne*m*m);
  v.condensed_rhs=arena.Get(ne*m);v.solve_workspace=arena.Get(ne*n*(m+1));
  v.local_solve_workspace=arena.Get(ne*n*(m+1));v.local_solve_work=arena.Get(ne*3);
  v.status=arena.Get(ne);v.scalar=arena.Get(3);v.true_rhs_scratch=arena.Get(m);
  v.true_solution=arena.Get(m);v.recovered_interior=arena.Get(ne*n);
  v.trace_increment_local=arena.Get(ne*m);
  v.signed_element_trace=arena.Get<int>(ne*m);v.vector_owned_source=arena.Get<int>(ne*m);
  v.vector_owned_target=arena.Get<int>(ne*m);
  v.coupled_lu=arena.Get(ne*np*np);v.coupled_rhs=arena.Get(ne*np*q);
  v.coupled_c=arena.Get(ne*mp*np);v.coupled_product=arena.Get(ne*mp*mp);
  v.coupled_pivots=arena.Get(ne*np);v.coupled_row_exponents=arena.Get(ne*np);
  v.coupled_interior_total=arena.Get(ne*n);v.coupled_trace_total=arena.Get(m);
  v.coupled_base_rhs=arena.Get(m);v.coupled_trace_residual=arena.Get(m);v.coupled_diagnostics=arena.Get(8);
  std::vector<double> exact_x(ne*n);const double lambda[m]={0.13,-0.19,0.07};
  for(int e=0;e<ne;++e)
  {
    const double sign=e==0?1.0:-1.0;
    for(int j=0;j<m;++j)
    {v.signed_element_trace[e*m+j]=e==0?j:-1-j;v.vector_owned_source[e*m+j]=e*m+j;v.vector_owned_target[e*m+j]=j;}
    for(int i=0;i<n;++i)
    {
      exact_x[e*n+i]=0.2*std::sin(0.2*(i+2*e+1));
      for(int j=0;j<n;++j)v.a[(e*n+i)*n+j]=(i==j?4.0:0.01*std::cos(0.11*(i+1)*(j+1)));
      for(int j=0;j<m;++j)v.b[(e*n+i)*m+j]=0.02*std::sin(0.13*(i+1)*(j+1+e));
    }
    // Nontrivial pivoting across panel boundaries, with power-of-two row scaling.
    for(int j=0;j<n;++j)std::swap(v.a[(e*n)*n+j],v.a[(e*n+n-1)*n+j]);
    for(int i=0;i<n;++i)
    {
      const double scale=std::ldexp(1.0,(i%7)-3);
      for(int j=0;j<n;++j){v.a[(e*n+i)*n+j]*=scale;v.f[e*n+i]+=v.a[(e*n+i)*n+j]*exact_x[e*n+j];}
      for(int j=0;j<m;++j){v.b[(e*n+i)*m+j]*=scale;v.f[e*n+i]+=v.b[(e*n+i)*m+j]*sign*lambda[j];}
    }
    for(int i=0;i<m;++i)
    {
      for(int j=0;j<n;++j){const double c=0.015*std::cos(0.17*(i+1)*(j+1));v.c[(e*m+i)*n+j]=c;v.g[e*m+i]+=c*exact_x[e*n+j];}
      for(int j=0;j<m;++j){const double d=i==j?2.0:0.03;v.d[(e*m+i)*m+j]=d;v.g[e*m+i]+=d*sign*lambda[j];}
    }
  }
  std::vector<double> original_a(v.a,v.a+ne*n*n),original_b(v.b,v.b+ne*n*m),
      original_c(v.c,v.c+ne*m*n),original_d(v.d,v.d+ne*m*m),
      original_f(v.f,v.f+ne*n),original_g(v.g,v.g+ne*m);
  std::uint64_t tiles=0;
  if(!CudaLaunchCondensation(v,s,dg::hdg::generic::CudaEngineKind::ordinary,tiles)||
     !CudaLaunchRefinedTrueRhs(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
  if(!(v.coupled_diagnostics[0]>1.0)||
     (CudaLocalSolveUsesTf32(kind)&&v.coupled_diagnostics[1]!=v.coupled_diagnostics[0]))return false;
  const double gemm_calls=v.coupled_diagnostics[0],tf32_gemm_calls=v.coupled_diagnostics[1];
  if(CudaLocalSolveIsKrylov(kind))
  {
    const int N=ne*n+m,l=n+m;
    v.coupled_krylov_workspace=arena.Get(ne*(l*l+m*m+2*m+np)+53*N+m+128*25+3*26+1);
    if(!CudaLaunchCoupledKrylovSetup(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
    for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
    if(!CudaLaunchCoupledKrylovInitialize(v,s))return false;
    const auto defects=[&]()
    {
      return CudaLaunchGatherLocalTrace(v.signed_element_trace,ne*m,v.true_solution,v.trace_increment_local)&&
          CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)&&cudaDeviceSynchronize()==cudaSuccess;
    };
    const bool once=kind==CudaLocalSolveKind::coupled_krylov_fp32_once;
    const bool fp32=CudaLocalSolveUsesFp32Krylov(kind);
    double coupled=1,condensed=1,raw_coupled=0,raw_condensed=0;
    bool cached=false,accepted=false;int total_iterations=0,corrections=0,checks=0;
    for(int outer=0;outer<(once?1:4);++outer)
    {
      if(!defects())return false;
      CudaCoupledKrylovResult result{};
      if(!CudaSolveCoupledNativeKrylov(v,s,fp32?(once?1.e-6:1.e-3):1.e-12,144,result)||
         !defects()||result.iterations<=0||result.preconditioner_applications!=result.iterations)return false;
      total_iterations+=result.iterations;corrections=outer;
      coupled=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
      if(!std::isfinite(coupled))return false;
      raw_coupled=std::max(raw_coupled,coupled);
      if(!CudaLaunchCoupledCertificateSolve(v,s,cached)||cudaDeviceSynchronize()!=cudaSuccess)return false;
      for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
      cached=true;++checks;
      if(!CudaLaunchCoupledCertificateNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
      condensed=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
      if(!std::isfinite(condensed))return false;raw_condensed=std::max(raw_condensed,condensed);
      if(coupled<=1.e-10&&condensed<=1.e-10){accepted=true;break;}
    }
    if(once)
    {
      // Plain FP32 may be physically useful at a looser tolerance, but cannot
      // be mislabeled as strict success. Exercise original-FP64 recovery.
      if(accepted||!(raw_coupled>1.e-10||raw_condensed>1.e-10))return false;
      if(!CudaLaunchCoupledFallback(v,s)||!CudaLaunchRefinedTrueRhs(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
      std::vector<double> assembled(m*m);
      for(int e=0;e<ne;++e)for(int i=0;i<m*m;++i)assembled[i]+=v.schur[e*m*m+i];
      const auto trace=DenseSolve(assembled,std::vector<double>(v.true_rhs_scratch,v.true_rhs_scratch+m),m);
      std::copy(trace.begin(),trace.end(),v.true_solution);
      if(!CudaLaunchGatherLocalTrace(v.signed_element_trace,ne*m,v.true_solution,v.trace_increment_local)||
         !CudaLaunchCoupledAccumulate(v,s,true)||!defects())return false;
      coupled=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
      if(!CudaLaunchCoupledCertificateSolve(v,s,true)||!CudaLaunchCoupledCertificateNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
      condensed=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
      accepted=coupled<=1.e-10&&condensed<=1.e-10;
    }
    if(!accepted||(fp32&&!once&&corrections==0))return false;
    for(int i=0;i<ne*n;++i)if(std::abs(v.coupled_interior_total[i]-exact_x[i])>2.e-9)return false;
    for(int j=0;j<m;++j)if(std::abs(v.true_solution[j]-lambda[j])>2.e-9)return false;
    if(!std::equal(original_a.begin(),original_a.end(),v.a)||!std::equal(original_b.begin(),original_b.end(),v.b)||
       !std::equal(original_c.begin(),original_c.end(),v.c)||!std::equal(original_d.begin(),original_d.end(),v.d)||
       !std::equal(original_f.begin(),original_f.end(),v.f)||!std::equal(original_g.begin(),original_g.end(),v.g))return false;
    // Rebuild a changed operator/RHS generation and ensure that the original
    // factor cache is explicitly reset. Both A and f change together.
    for(int i=0;i<ne*n*n;++i)v.a[i]*=2.0;
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
    {double value=0;for(int j=0;j<n;++j)value+=v.a[(e*n+i)*n+j]*exact_x[e*n+j];for(int j=0;j<m;++j)value+=v.b[(e*n+i)*m+j]*(e==0?1.0:-1.0)*lambda[j];v.f[e*n+i]=value;}
    if(!CudaLaunchCondensation(v,s,dg::hdg::generic::CudaEngineKind::ordinary,tiles)||
       !CudaLaunchCoupledKrylovSetup(v,s)||!CudaLaunchCoupledKrylovInitialize(v,s)||!defects())return false;
    // Probe the rebuilt numerical kernels with bounded refinement, separately
    // from the once-only admission experiment above. Finite factor status
    // alone would not detect a stale packed operator or stale preconditioner.
    bool rebuilt_accepted=false;
    for(int outer=0;outer<4;++outer)
    {
      if(!defects())return false;
      CudaCoupledKrylovResult rebuilt{};
      if(!CudaSolveCoupledNativeKrylov(v,s,fp32?1.e-3:1.e-12,144,rebuilt)||!defects())return false;
      const double rc=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
      if(!CudaLaunchCoupledCertificateSolve(v,s,outer!=0)||cudaDeviceSynchronize()!=cudaSuccess)return false;
      for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
      if(!CudaLaunchCoupledCertificateNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
      const double rs=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
      if(std::isfinite(rc)&&std::isfinite(rs)&&rc<=1.e-10&&rs<=1.e-10){rebuilt_accepted=true;break;}
    }
    if(!rebuilt_accepted)return false;
    for(int i=0;i<ne*n;++i)if(std::abs(v.coupled_interior_total[i]-exact_x[i])>2.e-9)return false;
    for(int j=0;j<m;++j)if(std::abs(v.true_solution[j]-lambda[j])>2.e-9)return false;
    // Correction RHS normalization must preserve directions that would
    // underflow or overflow a direct FP32 cast. Check the physical FP64
    // accumulated correction against the manufactured solution after undoing
    // only this test's RHS scale; no relaxed production acceptance is used.
    for(const double rhs_scale:{1.e-200,1.e200})
    {
      if(!CudaLaunchCoupledKrylovInitialize(v,s)||!defects())return false;
      for(int i=0;i<ne*n;++i)v.local_solve_workspace[i]*=rhs_scale;
      for(int j=0;j<m;++j)v.local_solve_workspace[ne*n+ne*m+j]*=rhs_scale;
      CudaCoupledKrylovResult scaled{};
      if(!CudaSolveCoupledNativeKrylov(v,s,fp32?1.e-6:1.e-12,144,scaled)||
         cudaDeviceSynchronize()!=cudaSuccess||scaled.iterations<=0)return false;
      for(int i=0;i<ne*n;++i)
        if(!std::isfinite(v.coupled_interior_total[i])||
           std::abs(v.coupled_interior_total[i]/rhs_scale-exact_x[i])>2.e-5)return false;
      for(int j=0;j<m;++j)
        if(!std::isfinite(v.true_solution[j])||std::abs(v.true_solution[j]/rhs_scale-lambda[j])>2.e-5)return false;
    }
    // Nonfinite original coefficients cannot become an accepted FP32 matrix.
    v.a[0]=std::numeric_limits<double>::quiet_NaN();
    if(!CudaLaunchCoupledKrylovSetup(v,s)||cudaDeviceSynchronize()!=cudaSuccess||v.status[0]>=0)return false;
    std::cout<<"coupled-native-krylov mode="<<CudaLocalSolveName(kind)<<" apply="<<CudaCoupledApplyName(apply)
        <<" iterations="<<total_iterations<<" refinement_corrections="<<corrections
        <<" raw_original_coupled="<<raw_coupled<<" raw_original_condensed="<<raw_condensed
        <<" original_coupled="<<coupled<<" original_condensed="<<condensed
        <<" fp64_recovery="<<(once?1:0)<<" certificate_checks="<<checks
        <<" signed_trace=passed original_immutable=passed rebuilt_generation=passed rhs_range=passed nonfinite=passed\n";
    return true;
  }
  std::vector<double> schur(m*m);
  for(int e=0;e<ne;++e)for(int i=0;i<m*m;++i)schur[i]+=v.schur[e*m*m+i];
  const auto gather=[&]()
  {for(int e=0;e<ne;++e)for(int j=0;j<m;++j)v.trace_increment_local[e*m+j]=(e==0?1.0:-1.0)*v.true_solution[j];};
  double relative=1.0,previous=std::numeric_limits<double>::infinity();int corrections=0;
  for(int outer=0;outer<=4;++outer)
  {
    const auto delta=DenseSolve(schur,std::vector<double>(v.true_rhs_scratch,v.true_rhs_scratch+m),m);
    std::copy(delta.begin(),delta.end(),v.true_solution);gather();
    if(!CudaLaunchCoupledAccumulate(v,s,outer==0)||cudaDeviceSynchronize()!=cudaSuccess)return false;
    gather();
    if(!CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
    relative=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
    if(!std::isfinite(relative))return false;
    if(relative<=1.e-10)break;
    if(outer==4||!(relative<previous))return false;previous=relative;
    if(!CudaLaunchCoupledResidualRhs(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
    ++corrections;
  }
  if(kind==CudaLocalSolveKind::coupled_tf32&&corrections==0)return false;
  if(!CudaLaunchCoupledCertificateSolve(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
  if(!CudaLaunchCoupledCertificateNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  const double condensed=std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1]);
  if(!std::isfinite(condensed)||condensed>1.e-10)return false;
  // Independent dense original-Schur residual at the actual trace solution.
  std::vector<double> oracle_rhs(m),oracle_residual(m);
  for(int e=0;e<ne;++e)
  {
    const double sign=e==0?1.0:-1.0;
    std::vector<double> a(v.a+e*n*n,v.a+(e+1)*n*n),f(v.f+e*n,v.f+(e+1)*n),b_lambda(n);
    for(int i=0;i<n;++i)for(int j=0;j<m;++j)b_lambda[i]+=v.b[(e*n+i)*m+j]*sign*v.true_solution[j];
    const auto yf=DenseSolve(a,f,n),yb=DenseSolve(a,b_lambda,n);
    for(int i=0;i<m;++i)
    {
      double rhs=v.g[e*m+i],action=0.0;
      for(int j=0;j<n;++j){rhs-=v.c[(e*m+i)*n+j]*yf[j];action-=v.c[(e*m+i)*n+j]*yb[j];}
      for(int j=0;j<m;++j)action+=v.d[(e*m+i)*m+j]*sign*v.true_solution[j];
      oracle_rhs[i]+=sign*rhs;oracle_residual[i]+=sign*(rhs-action);
    }
  }
  for(int i=0;i<m;++i)
    if(std::abs(oracle_rhs[i]-v.true_rhs_scratch[i])>1.e-12||
       std::abs(oracle_residual[i]-v.coupled_base_rhs[i])>1.e-12)return false;
  if(!std::equal(original_a.begin(),original_a.end(),v.a)||
     !std::equal(original_b.begin(),original_b.end(),v.b)||!std::equal(original_c.begin(),original_c.end(),v.c)||
     !std::equal(original_d.begin(),original_d.end(),v.d)||!std::equal(original_f.begin(),original_f.end(),v.f)||
     !std::equal(original_g.begin(),original_g.end(),v.g))return false;
  // The original gates must reject a perturbed trace even with the retained
  // approximate preconditioner and its successful construction untouched.
  v.true_solution[0]+=1.e-3;gather();
  if(!CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess||
     !(std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1])>1.e-10))return false;
  if(!CudaLaunchCoupledCertificateSolve(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
  if(!CudaLaunchCoupledCertificateNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess||
     !(std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1])>1.e-10))return false;
  // Bounded whole-FP64 recovery uses the same unchanged original blocks.
  if(!CudaLaunchCoupledFallback(v,s)||!CudaLaunchRefinedTrueRhs(v,s)||
     cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
  std::fill(schur.begin(),schur.end(),0.0);
  for(int e=0;e<ne;++e)for(int i=0;i<m*m;++i)schur[i]+=v.schur[e*m*m+i];
  const auto fallback_lambda=DenseSolve(schur,std::vector<double>(v.true_rhs_scratch,v.true_rhs_scratch+m),m);
  std::copy(fallback_lambda.begin(),fallback_lambda.end(),v.true_solution);gather();
  if(!CudaLaunchCoupledAccumulate(v,s,true)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  gather();
  if(!CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess||
     !(std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1])<=1.e-10))return false;
  if(!CudaLaunchCoupledCertificateSolve(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
  if(!CudaLaunchCoupledCertificateNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess||
     !(std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1])<=1.e-10))return false;
  // Rebuild with a different original operator/RHS. A large decoupled interior
  // RHS makes the coupled norm pass while a trace error still fails the
  // original condensed norm. Reusing certificate scratch as r_i would break
  // the correction below; rebuild the original defects exactly as production.
  std::fill(v.a,v.a+ne*n*n,0.0);std::fill(v.b,v.b+ne*n*m,0.0);
  std::fill(v.c,v.c+ne*m*n,0.0);std::fill(v.d,v.d+ne*m*m,0.0);
  std::fill(v.f,v.f+ne*n,0.0);
  for(int e=0;e<ne;++e)
  {
    for(int i=0;i<n;++i)v.a[(e*n+i)*n+i]=1.0;
    v.f[e*n]=(e+1)*1.e4;v.f[e*n+1]=1.0;
    for(int j=0;j<m;++j)
    {
      v.d[(e*m+j)*m+j]=1.0;v.c[(e*m+j)*n+1]=0.125;
      v.g[e*m+j]=0.125+(e==0?1.0:-1.0)*(j+1)*1.e-6;
    }
  }
  if(!CudaLaunchCondensation(v,s,dg::hdg::generic::CudaEngineKind::ordinary,tiles)||
     cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
  for(int j=0;j<m;++j)v.true_solution[j]=(j+1)*1.e-6+1.e-8;
  gather();
  if(!CudaLaunchCoupledAccumulate(v,s,true)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  gather();
  if(!CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess||
     !(std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1])<=1.e-10))return false;
  if(!CudaLaunchCoupledCertificateSolve(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
  if(!CudaLaunchCoupledCertificateNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess||
     !(std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1])>1.e-10))return false;
  if(!CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)||
     !CudaLaunchCoupledResidualRhs(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int j=0;j<m;++j)v.true_solution[j]=v.true_rhs_scratch[j]/2.0;
  gather();
  if(!CudaLaunchCoupledAccumulate(v,s,false)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  gather();
  if(!CudaLaunchRefinedOriginalCoupledSquaredNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess||
     !(std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1])<=1.e-10))return false;
  if(!CudaLaunchCoupledCertificateSolve(v,s)||cudaDeviceSynchronize()!=cudaSuccess)return false;
  for(int e=0;e<ne;++e)if(v.status[e]!=1.0)return false;
  if(!CudaLaunchCoupledCertificateNorms(v,s)||cudaDeviceSynchronize()!=cudaSuccess||
     !(std::sqrt(v.scalar[0])/std::sqrt(v.scalar[1])<=1.e-10))return false;
  std::cout<<"coupled-certificate-continuation mode="<<CudaLocalSolveName(kind)<<" passed\n";
  // Neither construction nor original recovery may accept a nonfinite A.
  v.a[0]=std::numeric_limits<double>::quiet_NaN();
  std::uint64_t ignored=0;
  if(!CudaLaunchCondensation(v,s,dg::hdg::generic::CudaEngineKind::ordinary,ignored)||
     cudaDeviceSynchronize()!=cudaSuccess||v.status[0]!=-1.0||
     !CudaLaunchCoupledFallback(v,s)||cudaDeviceSynchronize()!=cudaSuccess||v.status[0]!=-1.0)return false;
  std::cout<<"coupled-system mode="<<CudaLocalSolveName(kind)<<" outer_corrections="<<corrections
      <<" original_coupled="<<relative<<" original_condensed="<<condensed
      <<" gemm_calls="<<gemm_calls<<" tf32_gemm_calls="<<tf32_gemm_calls<<" passed\n";
  return true;
}
}
int main()
{
  try
  {
    if(!TolerancePolicies()||!FewRhsTriangularSolve())return 1;
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
    for(const auto kind:{CudaLocalSolveKind::coupled_fp64,CudaLocalSolveKind::coupled_tf32})
      if(!CoupledSystem(kind))return 1;
    for(const auto apply:{CudaCoupledApplyKind::serial,CudaCoupledApplyKind::cooperative,CudaCoupledApplyKind::inverse})
      if(!CoupledPartialOverlapRing(apply))return 1;
    for(const auto apply:{CudaCoupledApplyKind::serial,CudaCoupledApplyKind::cooperative,
        CudaCoupledApplyKind::inverse,CudaCoupledApplyKind::inverse_tf32})
      for(const auto kind:{CudaLocalSolveKind::coupled_krylov_fp64,
          CudaLocalSolveKind::coupled_krylov_tf32,CudaLocalSolveKind::coupled_krylov_fp32,
          CudaLocalSolveKind::coupled_krylov_fp32_once})
      {
        if(apply==CudaCoupledApplyKind::inverse_tf32&&!CudaLocalSolveUsesTf32(kind))continue;
        if(!CoupledSystem(kind,apply)){std::cerr<<"native coupled fixture failed mode="<<CudaLocalSolveName(kind)
            <<" apply="<<CudaCoupledApplyName(apply)<<'\n';return 1;}
      }
    return 0;
  }
  catch(const char* error){std::cerr<<error<<'\n';return 2;}
}

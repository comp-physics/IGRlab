#include "coupled_wmma_gemm.cuh"

// Private approximate block preconditioner. Original FP64 blocks are immutable.
// Row-equilibrated, padded row-major batches share one launch layout for the
// FP64 control and FP32/explicit-TF32 construction. No column residual gate is
// imposed here: original coupled/condensed residuals certify the outer solve.
inline int CoupledPad(const int n) noexcept {return ((n+15)/16)*16;}
template <typename T>
__global__ void CoupledPackKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs;
  const int np=((n+15)/16)*16,mp=((m+15)/16)*16,q=((m+16)/16)*16;
  T* a=reinterpret_cast<T*>(v.coupled_lu+static_cast<std::size_t>(e)*np*np);
  T* b=reinterpret_cast<T*>(v.coupled_rhs+static_cast<std::size_t>(e)*np*q);
  T* c=reinterpret_cast<T*>(v.coupled_c+static_cast<std::size_t>(e)*mp*np);
  __shared__ int failed;
  if(threadIdx.x==0)failed=0;
  __syncthreads();
  for(int row=threadIdx.x;row<np;row+=blockDim.x)
  {
    double maximum=0.0;
    if(row<n)for(int j=0;j<n;++j)
    {
      const double x=v.a[(static_cast<std::size_t>(e)*n+row)*n+j];
      if(!isfinite(x))atomicExch(&failed,1);
      maximum=fmax(maximum,fabs(x));
    }
    int exponent=0;if(isfinite(maximum))frexp(maximum,&exponent);
    v.coupled_row_exponents[e*np+row]=exponent;
    for(int j=0;j<np;++j)
    {
      const double x=row<n&&j<n?scalbn(v.a[(static_cast<std::size_t>(e)*n+row)*n+j],-exponent)
          :row==j?1.0:0.0;
      a[row*np+j]=static_cast<T>(x);
      if(!isfinite(static_cast<double>(a[row*np+j])))atomicExch(&failed,1);
    }
    for(int col=0;col<q;++col)
    {
      const double x=row<n&&col<=m?scalbn(col<m?
          v.b[(static_cast<std::size_t>(e)*n+row)*m+col]:v.f[e*n+row],-exponent):0.0;
      b[row*q+col]=static_cast<T>(x);
      if(!isfinite(static_cast<double>(b[row*q+col])))atomicExch(&failed,1);
    }
  }
  for(int i=threadIdx.x;i<mp*np;i+=blockDim.x)
  {
    const int row=i/np,col=i%np;
    c[i]=static_cast<T>(row<m&&col<n?v.c[(static_cast<std::size_t>(e)*m+row)*n+col]:0.0);
    if(!isfinite(static_cast<double>(c[i])))atomicExch(&failed,1);
  }
  __syncthreads();
  if(threadIdx.x==0)v.status[e]=failed?-1.0:1.0;
}
template <typename T>
__global__ void CoupledScrubKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x;if(v.status[e]>0.0)return;
  const int np=((s.interior_dofs+15)/16)*16,q=((s.trace_dofs+16)/16)*16;
  T* a=reinterpret_cast<T*>(v.coupled_lu+static_cast<std::size_t>(e)*np*np);
  T* b=reinterpret_cast<T*>(v.coupled_rhs+static_cast<std::size_t>(e)*np*q);
  for(int i=threadIdx.x;i<np*np;i+=blockDim.x)a[i]=i/np==i%np?T(1):T(0);
  for(int i=threadIdx.x;i<np*q;i+=blockDim.x)b[i]=T(0);
}
template <typename T>
__global__ void CoupledPanelKernel(CudaDeviceView v,CudaMeshShape s,int first,int end)
{
  const int e=blockIdx.x;if(v.status[e]<0.0)return;
  const int np=((s.interior_dofs+15)/16)*16,q=((s.trace_dofs+16)/16)*16;
  T* a=reinterpret_cast<T*>(v.coupled_lu+static_cast<std::size_t>(e)*np*np);
  T* b=reinterpret_cast<T*>(v.coupled_rhs+static_cast<std::size_t>(e)*np*q);
  __shared__ int pivot,failed;
  if(threadIdx.x==0)failed=0;
  __syncthreads();
  for(int k=first;k<end;++k)
  {
    if(threadIdx.x==0)
    {
      pivot=k;
      for(int row=k+1;row<np;++row)
        if(fabs(static_cast<double>(a[row*np+k]))>fabs(static_cast<double>(a[pivot*np+k])))pivot=row;
      v.coupled_pivots[e*np+k]=pivot;
      if(!isfinite(static_cast<double>(a[pivot*np+k]))||
         !(fabs(static_cast<double>(a[pivot*np+k]))>64.0*DBL_EPSILON))failed=1;
    }
    __syncthreads();
    if(failed){if(threadIdx.x==0)v.status[e]=-1.0;return;}
    if(pivot!=k)
    {
      for(int j=threadIdx.x;j<np;j+=blockDim.x)
      {const T x=a[k*np+j];a[k*np+j]=a[pivot*np+j];a[pivot*np+j]=x;}
      for(int j=threadIdx.x;j<q;j+=blockDim.x)
      {const T x=b[k*q+j];b[k*q+j]=b[pivot*q+j];b[pivot*q+j]=x;}
    }
    __syncthreads();
    for(int row=k+1+threadIdx.x;row<np;row+=blockDim.x)a[row*np+k]/=a[k*np+k];
    __syncthreads();
    for(int i=threadIdx.x;i<(np-k-1)*(end-k-1);i+=blockDim.x)
    {
      const int row=k+1+i/(end-k-1),j=k+1+i%(end-k-1);
      a[row*np+j]-=a[row*np+k]*a[k*np+j];
    }
    __syncthreads();
  }
  // L11^{-1} times the trailing matrix columns and all RHS columns. The
  // large L21*U12 / L21*Y1 products are separate explicit TF32 GEMMs below.
  for(int j=end+threadIdx.x;j<np;j+=blockDim.x)
    for(int row=first;row<end;++row)
    {T x=a[row*np+j];for(int k=first;k<row;++k)x-=a[row*np+k]*a[k*np+j];a[row*np+j]=x;}
  for(int col=threadIdx.x;col<q;col+=blockDim.x)
    for(int row=first;row<end;++row)
    {T x=b[row*q+col];for(int k=first;k<row;++k)x-=a[row*np+k]*b[k*q+col];b[row*q+col]=x;}
}
template <typename T>
__global__ void CoupledBackPanelKernel(CudaDeviceView v,CudaMeshShape s,int first,int end)
{
  const int e=blockIdx.x;if(v.status[e]<0.0)return;
  const int np=((s.interior_dofs+15)/16)*16,q=((s.trace_dofs+16)/16)*16;
  const T* a=reinterpret_cast<const T*>(v.coupled_lu+static_cast<std::size_t>(e)*np*np);
  T* b=reinterpret_cast<T*>(v.coupled_rhs+static_cast<std::size_t>(e)*np*q);
  for(int col=threadIdx.x;col<q;col+=blockDim.x)
    for(int row=end-1;row>=first;--row)
    {T x=b[row*q+col];for(int k=row+1;k<end;++k)x-=a[row*np+k]*b[k*q+col];b[row*q+col]=x/a[row*np+row];}
}
template <typename T>
__global__ void CoupledUnpackKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs,q=((m+16)/16)*16;
  const int np=((n+15)/16)*16,mp=((m+15)/16)*16;
  const T* b=reinterpret_cast<const T*>(v.coupled_rhs+static_cast<std::size_t>(e)*np*q);
  const T* product=reinterpret_cast<const T*>(v.coupled_product+static_cast<std::size_t>(e)*mp*mp);
  __shared__ int failed;if(threadIdx.x==0)failed=v.status[e]<0.0;__syncthreads();
  for(int i=threadIdx.x;i<n*(m+1);i+=blockDim.x)
  {
    const double x=static_cast<double>(b[(i/(m+1))*q+i%(m+1)]);
    v.solve_workspace[static_cast<std::size_t>(e)*n*(m+1)+i]=x;
    if(!isfinite(x))atomicExch(&failed,1);
  }
  for(int i=threadIdx.x;i<m*m;i+=blockDim.x)
  {
    const double x=v.d[static_cast<std::size_t>(e)*m*m+i]-static_cast<double>(product[(i/m)*mp+i%m]);
    v.schur[static_cast<std::size_t>(e)*m*m+i]=x;
    if(!isfinite(x))atomicExch(&failed,1);
  }
  __syncthreads();
  if(threadIdx.x==0)
  {v.status[e]=failed?-1.0:1.0;v.local_solve_work[3*e]=0.0;v.local_solve_work[3*e+1]=0.0;v.local_solve_work[3*e+2]=failed?0.0:1.0;}
}
__global__ void CoupledGemmCountKernel(CudaDeviceView v,double calls,bool tf32)
{if(threadIdx.x==0){v.coupled_diagnostics[0]=calls;v.coupled_diagnostics[1]=tf32?calls:0.0;}}
template <typename T>
bool CoupledBuild(const CudaDeviceView& v,const CudaMeshShape& s)
{
  const int np=CoupledPad(s.interior_dofs),mp=CoupledPad(s.trace_dofs),q=CoupledPad(s.trace_dofs+1);
  const auto handle=static_cast<cublasHandle_t>(v.local_solve_vendor_handle);
  if(handle==nullptr||s.elements<=0||s.interior_dofs<=0||s.trace_dofs<=0)return false;
  // The float construction uses explicit WMMA, independent of cuBLAS math
  // mode or NVIDIA_TF32_OVERRIDE. The double control retains its cuBLAS policy.
  constexpr bool use_tf32 = std::is_same<T, float>::value;
  if constexpr(use_tf32)
    if(!CoupledWmmaTf32Supported())return false;
  cublasMath_t original_math;
  if(cublasGetMathMode(handle,&original_math)!=CUBLAS_STATUS_SUCCESS)return false;
  const cublasMath_t requested_math = use_tf32 ? original_math : CUBLAS_PEDANTIC_MATH;
  if(cublasSetMathMode(handle,requested_math)!=CUBLAS_STATUS_SUCCESS)return false;
  struct RestoreMath
  {
    cublasHandle_t handle;cublasMath_t math;
    ~RestoreMath(){(void)cublasSetMathMode(handle,math);}
  } restore_math{handle,original_math};
  const T minus=-1,plus=1,zero=0;
  T* lu=reinterpret_cast<T*>(v.coupled_lu);T* rhs=reinterpret_cast<T*>(v.coupled_rhs);
  T* c=reinterpret_cast<T*>(v.coupled_c);T* p=reinterpret_cast<T*>(v.coupled_product);
  constexpr long long stride_scale=sizeof(double)/sizeof(T);
  const long long ls=stride_scale*np*np,rs=stride_scale*np*q,cs=stride_scale*mp*np,ps=stride_scale*mp*mp;
  unsigned calls=0;
  const auto gemm=[&](int rows,int columns,int inner,const T* a,int lda,long long sa,
                       const T* b,int ldb,long long sb,T* out,int ldc,long long so,bool subtract)
  {
    if(rows==0||columns==0||inner==0)return true;
    if(cudaPeekAtLastError()!=cudaSuccess)return false;
    if constexpr(use_tf32)
    {
      if(!CoupledWmmaTf32Gemm(rows,columns,inner,a,lda,sa,b,ldb,sb,
          out,ldc,so,s.elements,subtract))return false;
    }
    else
    {
      const auto status=cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,
          rows,columns,inner,subtract?&minus:&plus,a,CUDA_R_64F,lda,sa,b,CUDA_R_64F,ldb,sb,
          subtract?&plus:&zero,out,CUDA_R_64F,ldc,so,s.elements,
          CUBLAS_COMPUTE_64F_PEDANTIC,CUBLAS_GEMM_DEFAULT);
      if(status!=CUBLAS_STATUS_SUCCESS)return false;
    }
    ++calls;return true;
  };
  CoupledPackKernel<T><<<s.elements,128>>>(v,s);
  CoupledScrubKernel<T><<<s.elements,128>>>(v,s);
  constexpr int block=32;
  for(int first=0;first<np;first+=block)
  {
    const int end=std::min(first+block,np),width=end-first,remaining=np-end;
    CoupledPanelKernel<T><<<s.elements,128>>>(v,s,first,end);
    CoupledScrubKernel<T><<<s.elements,128>>>(v,s);
    if(!gemm(remaining,remaining,width,lu+first*np+end,np,ls,lu+end*np+first,np,ls,
             lu+end*np+end,np,ls,true)||
       !gemm(q,remaining,width,rhs+first*q,q,rs,lu+end*np+first,np,ls,
             rhs+end*q,q,rs,true))return false;
  }
  for(int end=np;end>0;)
  {
    const int first=((end-1)/block)*block,width=end-first;
    CoupledBackPanelKernel<T><<<s.elements,128>>>(v,s,first,end);
    if(!gemm(q,first,width,rhs+first*q,q,rs,lu+first,np,ls,rhs,q,rs,true))return false;
    end=first;
  }
  // Row-major C*W_B is the column-major product W_B^T*C^T. RHS padding
  // includes f; extra output columns/rows are discarded, never physical DOFs.
  if(!gemm(mp,mp,np,rhs,q,rs,c,np,cs,p,mp,ps,false))return false;
  CoupledUnpackKernel<T><<<s.elements,128>>>(v,s);
  CoupledGemmCountKernel<<<1,1>>>(v,calls,use_tf32);
  return cudaPeekAtLastError()==cudaSuccess;
}

// The retained factor applies to one changing coupled residual RHS. This
// deliberately has no all-column residual/refinement pass.
template <typename T>
__global__ void CoupledResidualSolveKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs;
  const int np=((n+15)/16)*16,q=((m+16)/16)*16;
  const T* lu=reinterpret_cast<const T*>(v.coupled_lu+static_cast<std::size_t>(e)*np*np);
  T* rhs=reinterpret_cast<T*>(v.coupled_rhs+static_cast<std::size_t>(e)*np*q);
  for(int row=threadIdx.x;row<np;row+=blockDim.x)
    rhs[row]=static_cast<T>(row<n?scalbn(v.local_solve_workspace[e*n+row],
        -static_cast<int>(v.coupled_row_exponents[e*np+row])):0.0);
  __syncthreads();
  RefinedLocalTriangularSolve(lu,v.coupled_pivots+e*np,rhs,np,1);
  for(int row=threadIdx.x;row<n;row+=blockDim.x)
    v.solve_workspace[(static_cast<std::size_t>(e)*n+row)*(m+1)+m]=static_cast<double>(rhs[row]);
}
__global__ void CoupledAccumulateKernel(CudaDeviceView v,CudaMeshShape s,bool first)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs;
  for(int row=threadIdx.x;row<n;row+=blockDim.x)
  {
    const double* w=v.solve_workspace+(static_cast<std::size_t>(e)*n+row)*(m+1);
    double x=w[m];for(int j=0;j<m;++j)x-=w[j]*v.trace_increment_local[e*m+j];
    const int i=e*n+row;
    v.coupled_interior_total[i]=first?x:v.coupled_interior_total[i]+x;
    v.recovered_interior[i]=v.coupled_interior_total[i];
  }
}
__global__ void CoupledTraceAccumulateKernel(CudaDeviceView v,int count,bool first)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  v.coupled_trace_total[i]=first?v.true_solution[i]:v.coupled_trace_total[i]+v.true_solution[i];
  v.true_solution[i]=v.coupled_trace_total[i];
}
__global__ void CoupledRestoreKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x;__shared__ int bad;
  if(threadIdx.x==0)bad=0;__syncthreads();
  for(int row=threadIdx.x;row<s.interior_dofs;row+=blockDim.x)
  {const int i=e*s.interior_dofs+row;const double x=v.coupled_interior_total[i];
   v.recovered_interior[i]=x;if(!isfinite(x))atomicExch(&bad,1);}
  __syncthreads();if(threadIdx.x==0)v.status[e]=bad?-1.0:1.0;
}
__global__ void CoupledCopyTraceResidualKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<s.true_trace_count)v.coupled_trace_residual[i]=
      v.local_solve_workspace[s.interior_count+s.elements*s.trace_dofs+i];
}

// Final original condensed certificate uses ONLY f and the actual interior
// defect as RHS. It does not rebuild or certify the n-by-m approximate W_B.
__global__ void CoupledCertificateSolveKernel(CudaDeviceView v,CudaMeshShape s,bool cached_factor=false)
{
  const int e=blockIdx.x,n=s.interior_dofs,np=((n+15)/16)*16,q=((s.trace_dofs+16)/16)*16;
  const double* a=v.a+static_cast<std::size_t>(e)*n*n;
  double* lu=v.lu+static_cast<std::size_t>(e)*n*n;
  double* rhs=v.coupled_rhs+static_cast<std::size_t>(e)*np*q;
  double* x=rhs+2*n;
  double* pivots=v.pivots+e*n;
  // The owner may reuse this factor only within the same original-A generation.
  if(!cached_factor&&!RefinedLocalFactor(a,lu,pivots,n))
  {if(threadIdx.x==0)v.status[e]=-1.0;return;}
  for(int i=threadIdx.x;i<2*n;i+=blockDim.x)
    rhs[i]=i%2==0?v.f[e*n+i/2]:v.local_solve_workspace[e*n+i/2];
  __syncthreads();RefinedLocalTriangularSolve(lu,pivots,rhs,n,2,0,
      s.coupled_apply!=CudaCoupledApplyKind::serial);
  for(int i=threadIdx.x;i<2*n;i+=blockDim.x)x[i]=rhs[i];
  __syncthreads();
  __shared__ int bad,needs;
  for(int step=0;step<=4;++step)
  {
    if(threadIdx.x==0){bad=0;needs=0;}__syncthreads();
    for(int i=threadIdx.x;i<2*n;i+=blockDim.x)
    {
      const int row=i/2,col=i%2;
      const double original=col==0?v.f[e*n+row]:v.local_solve_workspace[e*n+row];
      double r=original,scale=fabs(original);
      for(int j=0;j<n;++j){const double t=a[row*n+j]*x[2*j+col];r-=t;scale+=fabs(t);}
      rhs[i]=r;
      if(!isfinite(r)||!isfinite(scale)||!isfinite(x[i]))atomicExch(&bad,1);
      if(!(fabs(r)<=1.e-13*scale))atomicExch(&needs,1);
    }
    __syncthreads();
    if(bad||!needs||step==4)
    {if(threadIdx.x==0)v.status[e]=!bad&&!needs?1.0:-1.0;return;}
    RefinedLocalTriangularSolve(lu,pivots,rhs,n,2,0,
        s.coupled_apply!=CudaCoupledApplyKind::serial);
    for(int i=threadIdx.x;i<2*n;i+=blockDim.x)x[i]+=rhs[i];
    __syncthreads();
  }
}
__global__ void CoupledCertificateTraceKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs;
  const int np=((n+15)/16)*16,q=((m+16)/16)*16;
  const double* x=v.coupled_rhs+static_cast<std::size_t>(e)*np*q+2*n;
  for(int row=threadIdx.x;row<m;row+=blockDim.x)
  {
    double f=0.0,r=0.0;
    for(int j=0;j<n;++j)
    {const double c=v.c[(static_cast<std::size_t>(e)*m+row)*n+j];f+=c*x[2*j];r+=c*x[2*j+1];}
    v.local_solve_workspace[e*m+row]=f;
    v.local_solve_workspace[s.elements*m+e*m+row]=r;
  }
}

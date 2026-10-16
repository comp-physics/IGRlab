// Private launch-driven full-coupled FGMRES. Original blocks stay immutable.
// The float operator is row-equilibrated before conversion. All Arnoldi
// reductions and the small least-squares problem are FP64; vectors are T.
__host__ __device__ inline int CoupledKrylovPad(int n){return ((n+15)/16)*16;}
constexpr int coupled_krylov_restart=24;
constexpr int coupled_krylov_partial_blocks=128;
constexpr double coupled_patch_shift=1.e-4;

template<class T,class P> struct CoupledKrylovWorkspace
{
  T *op,*basis,*directions,*x,*rhs,*work;
  P *patch,*patch_rhs,*local_rhs;
  double *pivots,*exponents,*multiplicity,*partial,*dots,*second,*small;
};
template<class T,class P>
CoupledKrylovWorkspace<T,P> CoupledKrylovBind(CudaDeviceView v,CudaMeshShape s)
{
  const std::size_t e=s.elements,n=s.interior_dofs,m=s.trace_dofs;
  const std::size_t N=s.interior_count+s.true_trace_count,l=n+m,np=CoupledKrylovPad(n);
  double* p=v.coupled_krylov_workspace;
  const auto take=[&](std::size_t bytes){double* out=p;p+=(bytes+7)/8;return out;};
  CoupledKrylovWorkspace<T,P> w{};
  w.op=reinterpret_cast<T*>(take(e*l*l*sizeof(T)));
  w.patch=reinterpret_cast<P*>(take(e*m*m*sizeof(P)));
  w.patch_rhs=reinterpret_cast<P*>(take(e*m*sizeof(P)));
  w.local_rhs=reinterpret_cast<P*>(take(e*np*sizeof(P)));
  w.basis=reinterpret_cast<T*>(take((coupled_krylov_restart+1)*N*sizeof(T)));
  w.directions=reinterpret_cast<T*>(take(coupled_krylov_restart*N*sizeof(T)));
  w.x=reinterpret_cast<T*>(take(N*sizeof(T)));
  w.rhs=reinterpret_cast<T*>(take(N*sizeof(T)));
  w.work=reinterpret_cast<T*>(take(N*sizeof(T)));
  w.pivots=take(e*m*sizeof(double));w.exponents=take((N+1)*sizeof(double));
  w.multiplicity=take(s.true_trace_count*sizeof(double));
  w.partial=take(coupled_krylov_partial_blocks*(coupled_krylov_restart+1)*sizeof(double));
  w.dots=take((coupled_krylov_restart+2)*sizeof(double));
  w.second=take((coupled_krylov_restart+2)*sizeof(double));
  w.small=take((coupled_krylov_restart+2)*sizeof(double));return w;
}
// CoupledBuild has finished using coupled_c. Reuse that dead product input
// for immutable slot-to-true and two-sided incidence maps, retained until the
// next operator build. Local Schur contributions remain untouched; each patch
// gathers its principal block from their signed global assembly.
struct CoupledKrylovTracePatchMap
{
  int* truth;
  int* incidence;
  int* failed;
};
__host__ __device__ inline CoupledKrylovTracePatchMap CoupledKrylovPatchMap(
    CudaDeviceView v,CudaMeshShape s)
{
  int* p=reinterpret_cast<int*>(v.coupled_c);
  return {p,p+static_cast<std::size_t>(s.elements)*s.trace_dofs,
      p+static_cast<std::size_t>(s.elements)*s.trace_dofs+
          2*static_cast<std::size_t>(s.true_trace_count)};
}
__global__ void CoupledKrylovPatchMapKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=s.vector_owned_count)return;
  const auto map=CoupledKrylovPatchMap(v,s);
  const int source=v.vector_owned_source[i],target=v.vector_owned_target[i];
  if(source<0||source>=s.elements*s.trace_dofs||target<0||target>=s.true_trace_count)
  {atomicExch(map.failed,1);return;}
  if(atomicCAS(map.truth+source,-1,target)!=-1)
  {atomicExch(map.failed,1);return;}
  double sign;const int local=Decode(v.signed_element_trace[source],sign);
  // The existing rank-one operator uses local indices directly. Do not
  // silently build a differently numbered preconditioner for that operator.
  if(local!=target){atomicExch(map.failed,1);return;}
  const int first=atomicCAS(map.incidence+2*target,-1,source);
  if(first==-1)return;
  if(first/s.trace_dofs==source/s.trace_dofs)
  {atomicExch(map.failed,1);return;}
  if(atomicCAS(map.incidence+2*target+1,-1,source)!=-1)
    atomicExch(map.failed,1);
}
__global__ void CoupledKrylovPatchMapValidateKernel(CudaDeviceView v,CudaMeshShape s)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const auto map=CoupledKrylovPatchMap(v,s);
  if(i<s.elements*s.trace_dofs&&map.truth[i]<0)atomicExch(map.failed,1);
  if(i<s.true_trace_count&&map.incidence[2*i]<0)atomicExch(map.failed,1);
}
__device__ inline double CoupledKrylovPrincipalEntry(
    CudaDeviceView v,CudaMeshShape s,int element,int row,int column)
{
  const int m=s.trace_dofs,row_slot=element*m+row,column_slot=element*m+column;
  const auto map=CoupledKrylovPatchMap(v,s);
  const int true_row=map.truth[row_slot],true_column=map.truth[column_slot];
  double value=0.0;
  for(int r=0;r<2;++r)
  {
    const int rs=map.incidence[2*true_row+r];if(rs<0)continue;
    const int re=rs/m;
    for(int c=0;c<2;++c)
    {
      const int cs=map.incidence[2*true_column+c];if(cs<0||cs/m!=re)continue;
      double rsign,csign;Decode(v.signed_element_trace[rs],rsign);Decode(v.signed_element_trace[cs],csign);
      value+=rsign*csign*v.schur[(static_cast<std::size_t>(re)*m+rs%m)*m+cs%m];
    }
  }
  double rsign,csign;Decode(v.signed_element_trace[row_slot],rsign);Decode(v.signed_element_trace[column_slot],csign);
  return rsign*csign*value;
}

__global__ void CoupledKrylovScaleKernel(CudaDeviceView v,CudaMeshShape s,double* scale,double* mult)
{
  if(*CoupledKrylovPatchMap(v,s).failed)return;
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs;
  for(int row=threadIdx.x;row<n;row+=blockDim.x)
  {
    double maximum=0.0;
    for(int j=0;j<n;++j)maximum=fmax(maximum,fabs(v.a[(static_cast<std::size_t>(e)*n+row)*n+j]));
    for(int j=0;j<m;++j)maximum=fmax(maximum,fabs(v.b[(static_cast<std::size_t>(e)*n+row)*m+j]));
    int exponent=0;frexp(maximum,&exponent);scale[e*n+row]=exponent;
  }
  for(int row=threadIdx.x;row<m;row+=blockDim.x)
  {
    double sign;const int t=Decode(v.signed_element_trace[e*m+row],sign);
    double maximum=0.0;
    for(int j=0;j<n;++j)maximum=fmax(maximum,fabs(v.c[(static_cast<std::size_t>(e)*m+row)*n+j]));
    for(int j=0;j<m;++j)maximum=fmax(maximum,fabs(v.d[(static_cast<std::size_t>(e)*m+row)*m+j]));
    // A sum of row maxima bounds each assembled entry. Only positive values
    // participate, independently of orientation or cancellation.
    atomicAdd(scale+s.interior_count+t,maximum);atomicAdd(mult+t,1.0);
  }
}
__global__ void CoupledKrylovTraceExponentKernel(double* scale,int offset,int count)
{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){int e=0;frexp(scale[offset+i],&e);scale[offset+i]=e;}}

template<class T,class P>
__global__ void CoupledKrylovPackKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs,l=n+m;
  __shared__ int invalid;
  if(threadIdx.x==0)invalid=*CoupledKrylovPatchMap(v,s).failed;__syncthreads();
  if(invalid){if(threadIdx.x==0)v.status[e]=-1.0;return;}
  for(int k=threadIdx.x;k<l*l;k+=blockDim.x)
  {
    int row=k/l,col=k%l,global=e*n+row;
    if(row>=n){double sign;global=s.interior_count+Decode(v.signed_element_trace[e*m+row-n],sign);}
    const double value=row<n?(col<n?v.a[(static_cast<std::size_t>(e)*n+row)*n+col]:v.b[(static_cast<std::size_t>(e)*n+row)*m+col-n]):
      (col<n?v.c[(static_cast<std::size_t>(e)*m+row-n)*n+col]:v.d[(static_cast<std::size_t>(e)*m+row-n)*m+col-n]);
    const T converted=static_cast<T>(scalbn(value,-static_cast<int>(w.exponents[global])));
    w.op[static_cast<std::size_t>(e)*l*l+k]=converted;
    if(!isfinite(value)||!isfinite(static_cast<double>(converted)))atomicExch(&invalid,1);
  }
  P* patch=w.patch+static_cast<std::size_t>(e)*m*m;
  // The GEMM product is dead once CoupledBuild unpacked S and W_B. Keep
  // a distinct FP64 source for conversion/factorization of the patch.
  double* patch_source=v.coupled_product+static_cast<std::size_t>(e)*CoupledKrylovPad(m)*CoupledKrylovPad(m);
  for(int row=threadIdx.x;row<m;row+=blockDim.x)
  {
    double sign;const int global=s.interior_count+Decode(v.signed_element_trace[e*m+row],sign);
    double row_norm=0.0;
    for(int j=0;j<m;++j)
    {
      const double value=scalbn(CoupledKrylovPrincipalEntry(v,s,e,row,j),-static_cast<int>(w.exponents[global]));
      patch_source[row*m+j]=value;row_norm+=fabs(value);
      if(!isfinite(value)||!isfinite(static_cast<double>(static_cast<P>(value))))atomicExch(&invalid,1);
    }
    patch_source[row*m+row]+=coupled_patch_shift*fmax(row_norm,1.e-30);
    if(!isfinite(static_cast<double>(static_cast<P>(patch_source[row*m+row]))))atomicExch(&invalid,1);
  }
  __syncthreads();
  if(invalid){if(threadIdx.x==0)v.status[e]=-1.0;return;}
  const bool valid=RefinedLocalFactor(patch_source,patch,w.pivots+e*m,m);
  if(threadIdx.x==0)v.status[e]=valid?1.0:-1.0;
}

template<class T,class P>
__global__ void CoupledKrylovOperatorKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w,const T* x,T* out)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs,l=n+m;
  for(int row=threadIdx.x;row<l;row+=blockDim.x)
  {
    T sum=0;const T* a=w.op+(static_cast<std::size_t>(e)*l+row)*l;
    for(int j=0;j<n;++j)sum+=a[j]*x[e*n+j];
    for(int j=0;j<m;++j){double sign;const int t=Decode(v.signed_element_trace[e*m+j],sign);sum+=a[n+j]*static_cast<T>(sign)*x[s.interior_count+t];}
    if(row<n)out[e*n+row]=sum;
    else {double sign;const int t=Decode(v.signed_element_trace[e*m+row-n],sign);atomicAdd(out+s.interior_count+t,static_cast<T>(sign)*sum);}
  }
}
// Block LDU preconditioning: local A inverse, an overlapping shifted
// assembled-principal trace patch inverse, then retained W_B back substitution.
// No HYPRE calls.
template<class T,class P>
__global__ void CoupledKrylovInteriorKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w,const T* input,T* z,T* trace_rhs)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs,np=CoupledKrylovPad(n),l=n+m;
  P* rhs=w.local_rhs+e*np;
  for(int row=threadIdx.x;row<np;row+=blockDim.x)
    rhs[row]=row<n?static_cast<P>(scalbn(static_cast<double>(input[e*n+row]),
      static_cast<int>(w.exponents[e*n+row]-v.coupled_row_exponents[e*np+row]))):P(0);
  __syncthreads();
  RefinedLocalTriangularSolve(reinterpret_cast<const P*>(v.coupled_lu+static_cast<std::size_t>(e)*np*np),v.coupled_pivots+e*np,rhs,np,1,0,
      s.coupled_apply==CudaCoupledApplyKind::cooperative);
  for(int row=threadIdx.x;row<n;row+=blockDim.x)z[e*n+row]=static_cast<T>(rhs[row]);
  for(int row=threadIdx.x;row<m;row+=blockDim.x)
  {
    T sum=0;for(int j=0;j<n;++j)sum+=w.op[(static_cast<std::size_t>(e)*l+n+row)*l+j]*static_cast<T>(rhs[j]);
    double sign;const int t=Decode(v.signed_element_trace[e*m+row],sign);
    atomicAdd(trace_rhs+s.interior_count+t,-static_cast<T>(sign)*sum);
  }
}
template<class T,class P>
__global__ void CoupledKrylovPatchKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w,const T* rhs,T* z)
{
  const int e=blockIdx.x,m=s.trace_dofs;
  P* r=w.patch_rhs+e*m;
  for(int row=threadIdx.x;row<m;row+=blockDim.x)
  {double sign;const int t=Decode(v.signed_element_trace[e*m+row],sign);r[row]=static_cast<P>(sign*static_cast<double>(rhs[s.interior_count+t])/w.multiplicity[t]);}
  __syncthreads();RefinedLocalTriangularSolve(w.patch+static_cast<std::size_t>(e)*m*m,w.pivots+e*m,r,m,1,0,
      s.coupled_apply==CudaCoupledApplyKind::cooperative);
  for(int row=threadIdx.x;row<m;row+=blockDim.x)
  {double sign;const int t=Decode(v.signed_element_trace[e*m+row],sign);atomicAdd(z+s.interior_count+t,static_cast<T>(sign*static_cast<double>(r[row])/w.multiplicity[t]));}
}
template<class T,class P>
__global__ void CoupledKrylovBackKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w,T* z)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs;
  for(int row=threadIdx.x;row<n;row+=blockDim.x)
  {T value=z[e*n+row];for(int j=0;j<m;++j){double sign;const int t=Decode(v.signed_element_trace[e*m+j],sign);value-=static_cast<T>(v.solve_workspace[(static_cast<std::size_t>(e)*n+row)*(m+1)+j])*static_cast<T>(sign)*z[s.interior_count+t];}z[e*n+row]=value;}
}

// Form each inverse once, using the many-RHS triangular route. The local A
// inverse is compact n-by-n inside the retained padded LU allocation. Its
// temporary is v.lu, which has no certificate factor yet; later original-A
// certificates overwrite only that temporary, never the retained inverse.
// Patch formation uses the dead coupled_product and retains its inverse in
// w.patch. A failed conversion/inversion rejects this setup before any apply.
template<class T,class P,bool Patch>
__global__ void CoupledKrylovBuildInverseKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w)
{
  const int e=blockIdx.x;if(v.status[e]<0.0)return;
  const int n=s.interior_dofs,m=s.trace_dofs,np=CoupledKrylovPad(n),mp=CoupledKrylovPad(m);
  const int count=Patch?m:n,ld=Patch?m:np;
  P* factor=Patch?w.patch+static_cast<std::size_t>(e)*m*m:
      reinterpret_cast<P*>(v.coupled_lu+static_cast<std::size_t>(e)*np*np);
  P* temporary=reinterpret_cast<P*>(Patch?v.coupled_product+static_cast<std::size_t>(e)*mp*mp:
      v.lu+static_cast<std::size_t>(e)*n*n);
  const double* pivots=Patch?w.pivots+e*m:v.coupled_pivots+e*np;
  __shared__ int failed;if(threadIdx.x==0)failed=0;
  __syncthreads();
  for(int i=threadIdx.x;i<count;i+=blockDim.x)
    if(!isfinite(pivots[i])||pivots[i]<0.0||pivots[i]>=count)atomicExch(&failed,1);
  for(int i=threadIdx.x;i<count*count;i+=blockDim.x)
    temporary[i]=i/count==i%count?P(1):P(0);
  __syncthreads();
  if(failed){if(threadIdx.x==0)v.status[e]=-1.0;return;}
  RefinedLocalTriangularSolve(factor,pivots,temporary,count,count,ld);
  for(int i=threadIdx.x;i<count*count;i+=blockDim.x)
    if(!isfinite(static_cast<double>(temporary[i])))atomicExch(&failed,1);
  __syncthreads();
  if(failed){if(threadIdx.x==0)v.status[e]=-1.0;return;}
  for(int i=threadIdx.x;i<count*count;i+=blockDim.x)factor[i]=temporary[i];
}

// Explicit TF32 inverse application. One physical RHS occupies column zero
// of each 16-column MMA tile; the other columns are zero padding, not extra
// useful RHS. Measure this against the SIMT GEMV and cooperative controls.
template<class Out>
__device__ void CoupledKrylovInverseWmmaApply(const float* inverse,
    const float* rhs,Out* output,int n)
{
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr unsigned mask=0xffffffffu;
  __shared__ __align__(32) float tile_a[4][16*8];
  __shared__ __align__(32) float tile_b[4][8*16];
  __shared__ __align__(32) float tile_c[4][16*16];
  for(int row0=warp*16;row0<n;row0+=64)
  {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,8,
        nvcuda::wmma::precision::tf32,nvcuda::wmma::col_major> af;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,8,
        nvcuda::wmma::precision::tf32,nvcuda::wmma::col_major> bf;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,8,float> cf;
    nvcuda::wmma::fill_fragment(cf,0.0f);bool failed=false;
    for(int first=0;first<n;first+=8)
    {
      bool bad=false;
      for(int i=lane;i<16*8;i+=32)
      {
        const int row=row0+i%16,k=first+i/16;
        float value=row<n&&k<n?inverse[row*n+k]:0.0f;
        value=nvcuda::wmma::__float_to_tf32(value);
        if(!isfinite(value)){bad=true;value=0.0f;}tile_a[warp][i]=value;
      }
      for(int i=lane;i<8*16;i+=32)
      {
        const int k=first+i%8,col=i/8;
        float value=col==0&&k<n?rhs[k]:0.0f;
        value=nvcuda::wmma::__float_to_tf32(value);
        if(!isfinite(value)){bad=true;value=0.0f;}tile_b[warp][i]=value;
      }
      failed=__any_sync(mask,bad)||failed;__syncwarp(mask);
      nvcuda::wmma::load_matrix_sync(af,tile_a[warp],16);
      nvcuda::wmma::load_matrix_sync(bf,tile_b[warp],8);
      nvcuda::wmma::mma_sync(cf,af,bf,cf);__syncwarp(mask);
    }
    nvcuda::wmma::store_matrix_sync(tile_c[warp],cf,16,nvcuda::wmma::mem_col_major);
    __syncwarp(mask);
    if(lane<16&&row0+lane<n)
    {
      const float value=tile_c[warp][lane];
      output[row0+lane]=static_cast<Out>(failed||!isfinite(value)?CUDART_NAN_F:value);
    }
    __syncwarp(mask);
  }
#else
  for(int row=threadIdx.x;row<n;row+=blockDim.x)output[row]=static_cast<Out>(CUDART_NAN);
#endif
}
template<class Out,class P,bool Tensor>
__device__ void CoupledKrylovInverseApply(const P* inverse,const P* rhs,Out* output,int n)
{
  if constexpr(Tensor)
  {
    static_assert(std::is_same<P,float>::value);
    CoupledKrylovInverseWmmaApply(inverse,rhs,output,n);
  }
  else
    for(int row=threadIdx.x;row<n;row+=blockDim.x)
    {
      P value=0;for(int j=0;j<n;++j)value+=inverse[row*n+j]*rhs[j];
      output[row]=static_cast<Out>(value);
    }
}
template<class T,class P,bool Tensor>
__global__ void CoupledKrylovInverseInteriorKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w,const T* input,T* z,T* trace_rhs)
{
  const int e=blockIdx.x,n=s.interior_dofs,m=s.trace_dofs,np=CoupledKrylovPad(n),l=n+m;
  P* rhs=w.local_rhs+e*np;
  for(int row=threadIdx.x;row<n;row+=blockDim.x)
    rhs[row]=static_cast<P>(scalbn(static_cast<double>(input[e*n+row]),
        static_cast<int>(w.exponents[e*n+row]-v.coupled_row_exponents[e*np+row])));
  __syncthreads();
  CoupledKrylovInverseApply<T,P,Tensor>(reinterpret_cast<const P*>(v.coupled_lu+static_cast<std::size_t>(e)*np*np),rhs,z+e*n,n);
  __syncthreads();
  for(int row=threadIdx.x;row<m;row+=blockDim.x)
  {
    T value=0;for(int j=0;j<n;++j)value+=w.op[(static_cast<std::size_t>(e)*l+n+row)*l+j]*z[e*n+j];
    double sign;const int t=Decode(v.signed_element_trace[e*m+row],sign);
    atomicAdd(trace_rhs+s.interior_count+t,-static_cast<T>(sign)*value);
  }
}
template<class T,class P,bool Tensor>
__global__ void CoupledKrylovInversePatchKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w,const T* rhs,T* z)
{
  const int e=blockIdx.x,m=s.trace_dofs,mp=CoupledKrylovPad(m);
  P* input=w.patch_rhs+e*m;
  P* solution=reinterpret_cast<P*>(v.coupled_product+static_cast<std::size_t>(e)*mp*mp);
  for(int row=threadIdx.x;row<m;row+=blockDim.x)
  {double sign;const int t=Decode(v.signed_element_trace[e*m+row],sign);input[row]=static_cast<P>(sign*static_cast<double>(rhs[s.interior_count+t])/w.multiplicity[t]);}
  __syncthreads();
  CoupledKrylovInverseApply<P,P,Tensor>(w.patch+static_cast<std::size_t>(e)*m*m,input,solution,m);
  __syncthreads();
  for(int row=threadIdx.x;row<m;row+=blockDim.x)
  {double sign;const int t=Decode(v.signed_element_trace[e*m+row],sign);atomicAdd(z+s.interior_count+t,static_cast<T>(sign*static_cast<double>(solution[row])/w.multiplicity[t]));}
}
template<class T>
__global__ void CoupledKrylovDotKernel(const T* x,const T* basis,int N,int count,double* partial)
{
  const int b=blockIdx.x,j=blockIdx.y;double sum=0.0;
  for(int i=b*blockDim.x+threadIdx.x;i<N;i+=gridDim.x*blockDim.x)
    sum+=static_cast<double>(x[i])*static_cast<double>(basis?basis[static_cast<std::size_t>(j)*N+i]:x[i]);
  __shared__ double sums[128];sums[threadIdx.x]=sum;__syncthreads();
  for(int stride=64;stride;stride/=2){if(threadIdx.x<stride)sums[threadIdx.x]+=sums[threadIdx.x+stride];__syncthreads();}
  if(threadIdx.x==0)partial[j*coupled_krylov_partial_blocks+b]=sums[0];
}
__global__ void CoupledKrylovReduceKernel(const double* partial,double* out,int count)
{
  const int j=blockIdx.x;__shared__ double sums[128];
  sums[threadIdx.x]=partial[j*coupled_krylov_partial_blocks+threadIdx.x];__syncthreads();
  for(int stride=64;stride;stride/=2){if(threadIdx.x<stride)sums[threadIdx.x]+=sums[threadIdx.x+stride];__syncthreads();}
  if(threadIdx.x==0&&j<count)out[j]=sums[0];
}
template<class T>
__global__ void CoupledKrylovVectorKernel(T* out,const T* input,const T* basis,const double* coeff,int N,int count,double scale,int operation)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
  if(operation==0){out[i]=static_cast<T>(scale*static_cast<double>(input[i]));return;}
  if(operation==1){T v=out[i];for(int j=0;j<count;++j)v-=static_cast<T>(coeff[j])*basis[static_cast<std::size_t>(j)*N+i];out[i]=v;return;}
  if(operation==2){T v=out[i];for(int j=0;j<count;++j)v+=static_cast<T>(coeff[j])*basis[static_cast<std::size_t>(j)*N+i];out[i]=v;return;}
  out[i]=input[i]-out[i];
}
__global__ void CoupledKrylovCombineDotsKernel(double* a,const double* b,int count)
{const int i=threadIdx.x;if(i<count)a[i]+=b[i];}
// Normalize each changing residual before FP32 conversion. Compare exponents
// without first forming potentially overflowing/underflowing scaled doubles.
template<class T,class P>
__global__ void CoupledKrylovRhsExponentKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w)
{
  const int N=s.interior_count+s.true_trace_count;int maximum=-4096;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=gridDim.x*blockDim.x)
  {
    const double value=i<s.interior_count?v.local_solve_workspace[i]:
        v.local_solve_workspace[s.elements*s.trace_dofs+i];
    if(!isfinite(value)){maximum=2147483647;continue;}
    if(value!=0.0){int exponent=0;frexp(fabs(value),&exponent);maximum=max(maximum,exponent-static_cast<int>(w.exponents[i]));}
  }
  __shared__ int values[128];values[threadIdx.x]=maximum;__syncthreads();
  for(int stride=64;stride;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]=max(values[threadIdx.x],values[threadIdx.x+stride]);__syncthreads();}
  if(threadIdx.x==0)w.partial[blockIdx.x]=values[0];
}
__global__ void CoupledKrylovRhsExponentReduceKernel(const double* partial,double* exponent)
{
  __shared__ int values[128];values[threadIdx.x]=static_cast<int>(partial[threadIdx.x]);__syncthreads();
  for(int stride=64;stride;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]=max(values[threadIdx.x],values[threadIdx.x+stride]);__syncthreads();}
  if(threadIdx.x==0)*exponent=values[0]==2147483647?CUDART_NAN:values[0]==-4096?0.0:static_cast<double>(values[0]);
}
template<class T,class P>
__global__ void CoupledKrylovRhsKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x,N=s.interior_count+s.true_trace_count;if(i>=N)return;
  const double value=i<s.interior_count?v.local_solve_workspace[i]:v.local_solve_workspace[s.interior_count+s.elements*s.trace_dofs+i-s.interior_count];
  w.rhs[i]=isfinite(w.exponents[N])?static_cast<T>(scalbn(value,
      -static_cast<int>(w.exponents[i])-static_cast<int>(w.exponents[N]))):static_cast<T>(CUDART_NAN);w.x[i]=0;
}
template<class T,class P>
__global__ void CoupledKrylovCommitKernel(CudaDeviceView v,CudaMeshShape s,CoupledKrylovWorkspace<T,P> w)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<s.interior_count){v.coupled_interior_total[i]+=scalbn(static_cast<double>(w.x[i]),static_cast<int>(w.exponents[s.interior_count+s.true_trace_count]));v.recovered_interior[i]=v.coupled_interior_total[i];}
  if(i<s.true_trace_count){v.coupled_trace_total[i]+=scalbn(static_cast<double>(w.x[s.interior_count+i]),static_cast<int>(w.exponents[s.interior_count+s.true_trace_count]));v.true_solution[i]=v.coupled_trace_total[i];}
}

inline bool CoupledKrylovRead(double* host,const double* device,std::size_t count)
{return cudaPeekAtLastError()==cudaSuccess&&CudaTransportPolicy::DeviceToHost(host,device,count*sizeof(double),CudaTransferCategory::scalar_diagnostic,CudaTransferSite::scalar_status)&&CudaTransportPolicy::Synchronize(CudaTransferSite::scalar_status);}
template<class T,class P>
bool CoupledKrylovSetup(CudaDeviceView v,CudaMeshShape s)
{
  const bool inverse=s.coupled_apply==CudaCoupledApplyKind::inverse||
      s.coupled_apply==CudaCoupledApplyKind::inverse_tf32;
  if(s.coupled_apply==CudaCoupledApplyKind::invalid||
      (s.coupled_apply==CudaCoupledApplyKind::inverse_tf32&&
       (!std::is_same<P,float>::value||!CoupledWmmaTf32Supported()))||
      (inverse&&!v.lu))return false;
  auto w=CoupledKrylovBind<T,P>(v,s);const int N=s.interior_count+s.true_trace_count;
  const std::size_t slots=static_cast<std::size_t>(s.elements)*s.trace_dofs;
  const std::size_t map_ints=slots+2*static_cast<std::size_t>(s.true_trace_count);
  const std::size_t capacity=static_cast<std::size_t>(s.elements)*CoupledKrylovPad(s.trace_dofs)*CoupledKrylovPad(s.interior_dofs)*sizeof(double);
  if(s.elements<=0||s.trace_dofs<=0||s.true_trace_count<=0||
      slots>static_cast<std::size_t>(std::numeric_limits<int>::max()-127)||
      s.vector_owned_count!=static_cast<int>(slots)||s.vector_send_count!=0||s.vector_receive_count!=0||
      s.global_true_trace_count!=s.true_trace_count||!v.coupled_c||!v.vector_owned_source||!v.vector_owned_target||
      (map_ints+1)*sizeof(int)>capacity)return false;
  const auto map=CoupledKrylovPatchMap(v,s);
  if(cudaMemsetAsync(map.truth,0xff,map_ints*sizeof(int))!=cudaSuccess||
      cudaMemsetAsync(map.failed,0,sizeof(int))!=cudaSuccess)return false;
  CoupledKrylovPatchMapKernel<<<(s.vector_owned_count+127)/128,128>>>(v,s);
  const int map_count=std::max(static_cast<int>(slots),s.true_trace_count);
  CoupledKrylovPatchMapValidateKernel<<<(map_count+127)/128,128>>>(v,s);
  if(cudaMemsetAsync(w.exponents,0,N*sizeof(double))!=cudaSuccess||cudaMemsetAsync(w.multiplicity,0,s.true_trace_count*sizeof(double))!=cudaSuccess)return false;
  CoupledKrylovScaleKernel<<<s.elements,128>>>(v,s,w.exponents,w.multiplicity);
  CoupledKrylovTraceExponentKernel<<<(s.true_trace_count+127)/128,128>>>(w.exponents,s.interior_count,s.true_trace_count);
  CoupledKrylovPackKernel<T,P><<<s.elements,128>>>(v,s,w);
  if(inverse)
  {
    CoupledKrylovBuildInverseKernel<T,P,false><<<s.elements,128>>>(v,s,w);
    CoupledKrylovBuildInverseKernel<T,P,true><<<s.elements,128>>>(v,s,w);
  }
  return cudaPeekAtLastError()==cudaSuccess;
}
template<class T,class P>
bool CoupledKrylovSolve(CudaDeviceView v,CudaMeshShape s,double tolerance,int maximum,CudaCoupledKrylovResult& result)
{
  if(s.coupled_apply==CudaCoupledApplyKind::invalid||
      (s.coupled_apply==CudaCoupledApplyKind::inverse_tf32&&!std::is_same<P,float>::value))return false;
  result={};auto w=CoupledKrylovBind<T,P>(v,s);const int N=s.interior_count+s.true_trace_count;
  const auto dot=[&](const T* x,const T* basis,int count,double* out){CoupledKrylovDotKernel<T><<<dim3(coupled_krylov_partial_blocks,count),128>>>(x,basis,N,count,w.partial);CoupledKrylovReduceKernel<<<count,128>>>(w.partial,out,count);};
  const auto norm=[&](const T* x,double& norm){dot(x,nullptr,1,w.small);double value=0;if(!CoupledKrylovRead(&value,w.small,1)||!std::isfinite(value)||value<0)return false;norm=std::sqrt(value);return std::isfinite(norm);};
  const auto apply=[&](const T* x,T* out){cudaMemsetAsync(out+s.interior_count,0,s.true_trace_count*sizeof(T));CoupledKrylovOperatorKernel<T,P><<<s.elements,128>>>(v,s,w,x,out);};
  const auto precondition=[&](const T* x,T* z)
  {
    cudaMemcpyAsync(w.work+s.interior_count,x+s.interior_count,s.true_trace_count*sizeof(T),cudaMemcpyDeviceToDevice);
    cudaMemsetAsync(z+s.interior_count,0,s.true_trace_count*sizeof(T));
    if(s.coupled_apply==CudaCoupledApplyKind::inverse)
    {
      CoupledKrylovInverseInteriorKernel<T,P,false><<<s.elements,128>>>(v,s,w,x,z,w.work);
      CoupledKrylovInversePatchKernel<T,P,false><<<s.elements,128>>>(v,s,w,w.work,z);
    }
    else if(s.coupled_apply==CudaCoupledApplyKind::inverse_tf32)
    {
      if constexpr(std::is_same<P,float>::value)
      {
        CoupledKrylovInverseInteriorKernel<T,P,true><<<s.elements,128>>>(v,s,w,x,z,w.work);
        CoupledKrylovInversePatchKernel<T,P,true><<<s.elements,128>>>(v,s,w,w.work,z);
      }
    }
    else
    {
      CoupledKrylovInteriorKernel<T,P><<<s.elements,128>>>(v,s,w,x,z,w.work);
      CoupledKrylovPatchKernel<T,P><<<s.elements,128>>>(v,s,w,w.work,z);
    }
    CoupledKrylovBackKernel<T,P><<<s.elements,128>>>(v,s,w,z);
    ++result.preconditioner_applications;
  };
  CoupledKrylovRhsExponentKernel<T,P><<<coupled_krylov_partial_blocks,128>>>(v,s,w);
  CoupledKrylovRhsExponentReduceKernel<<<1,128>>>(w.partial,w.exponents+N);
  CoupledKrylovRhsKernel<T,P><<<(N+127)/128,128>>>(v,s,w);
  double rhs_norm=0;if(!norm(w.rhs,rhs_norm))return false;
  if(rhs_norm==0){result.relative_residual=0;result.converged=true;return true;}
  double h[(coupled_krylov_restart+1)*coupled_krylov_restart]{},g[coupled_krylov_restart+1]{},cs[coupled_krylov_restart]{},sn[coupled_krylov_restart]{};
  // Keep the asynchronous H2D source alive through the final norm synchronization.
  double y[coupled_krylov_restart]{};
  while(result.iterations<maximum)
  {
    apply(w.x,w.work);CoupledKrylovVectorKernel<T><<<(N+127)/128,128>>>(w.work,w.rhs,nullptr,nullptr,N,0,0,3);
    double beta=0;if(!norm(w.work,beta))return false;
    result.relative_residual=beta/rhs_norm;
    if(result.relative_residual<=tolerance){result.converged=true;break;}
    std::fill(std::begin(h),std::end(h),0.0);std::fill(std::begin(g),std::end(g),0.0);g[0]=beta;
    CoupledKrylovVectorKernel<T><<<(N+127)/128,128>>>(w.basis,w.work,nullptr,nullptr,N,0,1.0/beta,0);
    int columns=0;
    for(int j=0;j<coupled_krylov_restart&&result.iterations<maximum;++j)
    {
      ++result.iterations; // attempted Arnoldi steps, including failed work
      T* z=w.directions+static_cast<std::size_t>(j)*N;
      precondition(w.basis+static_cast<std::size_t>(j)*N,z);apply(z,w.work);
      dot(w.work,w.basis,j+1,w.dots);
      CoupledKrylovVectorKernel<T><<<(N+127)/128,128>>>(w.work,nullptr,w.basis,w.dots,N,j+1,0,1);
      dot(w.work,w.basis,j+1,w.second);
      CoupledKrylovVectorKernel<T><<<(N+127)/128,128>>>(w.work,nullptr,w.basis,w.second,N,j+1,0,1);
      CoupledKrylovCombineDotsKernel<<<1,32>>>(w.dots,w.second,j+1);
      dot(w.work,nullptr,1,w.dots+j+1);
      double values[coupled_krylov_restart+2]{};
      if(!CoupledKrylovRead(values,w.dots,j+2))return false;
      for(int i=0;i<j+2;++i)if(!std::isfinite(values[i]))return false;
      if(values[j+1]<0)return false;
      const double next=std::sqrt(values[j+1]);
      for(int i=0;i<=j;++i)h[j*(coupled_krylov_restart+1)+i]=values[i];
      h[j*(coupled_krylov_restart+1)+j+1]=next;
      if(next>0)CoupledKrylovVectorKernel<T><<<(N+127)/128,128>>>(w.basis+static_cast<std::size_t>(j+1)*N,w.work,nullptr,nullptr,N,0,1.0/next,0);
      for(int i=0;i<j;++i){double& a=h[j*(coupled_krylov_restart+1)+i];double& b=h[j*(coupled_krylov_restart+1)+i+1];const double old=a;a=cs[i]*old+sn[i]*b;b=-sn[i]*old+cs[i]*b;}
      double& diagonal=h[j*(coupled_krylov_restart+1)+j];double& below=h[j*(coupled_krylov_restart+1)+j+1];
      const double length=std::hypot(diagonal,below);if(!std::isfinite(length)||length==0)return false;
      cs[j]=diagonal/length;sn[j]=below/length;diagonal=length;below=0;
      g[j+1]=-sn[j]*g[j];g[j]*=cs[j];columns=j+1;
      if(std::abs(g[j+1])<=tolerance*rhs_norm||next==0)break;
    }
    for(int i=columns-1;i>=0;--i){double value=g[i];for(int j=i+1;j<columns;++j)value-=h[j*(coupled_krylov_restart+1)+i]*y[j];y[i]=value/h[i*(coupled_krylov_restart+1)+i];if(!std::isfinite(y[i]))return false;}
    if(!CudaTransportPolicy::HostToDevice(w.small,y,columns*sizeof(double),CudaTransferCategory::scalar_diagnostic,CudaTransferSite::scalar_status))return false;
    CoupledKrylovVectorKernel<T><<<(N+127)/128,128>>>(w.x,nullptr,w.directions,w.small,N,columns,0,2);
    ++result.restarts;
  }
  apply(w.x,w.work);CoupledKrylovVectorKernel<T><<<(N+127)/128,128>>>(w.work,w.rhs,nullptr,nullptr,N,0,0,3);
  double final_norm=0;if(!norm(w.work,final_norm))return false;
  result.relative_residual=final_norm/rhs_norm;result.converged=result.relative_residual<=tolerance;
  // A bounded inexact correction may still help outer refinement. Its original
  // FP64 residual is always checked by the caller before acceptance.
  CoupledKrylovCommitKernel<T,P><<<(N+127)/128,128>>>(v,s,w);
  return cudaPeekAtLastError()==cudaSuccess;
}

// Test-only batched interior response panels from actual MFEM SIPG operators.
// D^{-1}[b,E] has one physical load and distinct coupled-face response columns.
// Explicit --rhs uses independent manufactured stage defects instead. No
// complete-application gain is inferred from this mechanism experiment.
#include "mfem.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace
{
using Clock=std::chrono::steady_clock;
double Seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void CUDA(cudaError_t e) { MFEM_VERIFY(e==cudaSuccess,cudaGetErrorString(e)); }
void BLAS(cublasStatus_t e) { MFEM_VERIFY(e==CUBLAS_STATUS_SUCCESS,"cuBLAS failure "<<e); }

template<class T> struct Buffer
{
  T* data=nullptr; std::size_t size=0;
  explicit Buffer(std::size_t n):size(n) { CUDA(cudaMalloc(&data,n*sizeof(T))); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&)=delete;
};
__global__ void Convert(int n,const double* x,float* y)
{ int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=static_cast<float>(x[i]); }
__global__ void ConvertBack(int n,const float* x,double* y)
{ int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=static_cast<double>(x[i]); }

template<class Real>
__global__ void Ordinary(int n,int m,const Real* a,const Real* b,Real* c)
{
  const int e=blockIdx.y, entry=blockIdx.x*blockDim.x+threadIdx.x;
  if(entry>=n*m)return;
  const int i=entry/m,j=entry%m;
  Real value=0;
  for(int k=0;k<n;++k)value+=a[(std::size_t(e)*n+i)*n+k]*b[(std::size_t(e)*n+k)*m+j];
  c[std::size_t(e)*n*m+entry]=value;
}

template<bool ThreeProduct>
__global__ void Tensor(int n,int m,const float* a,const float* b,float* c)
{
  using namespace nvcuda;
  const int e=blockIdx.x,i0=16*blockIdx.y,j0=16*blockIdx.z,lane=threadIdx.x;
  __shared__ __align__(32) float ah[128],al[128],bh[128],bl[128],out[256];
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> a_hi,a_lo;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> b_hi,b_lo;
  wmma::fragment<wmma::accumulator,16,16,8,float> sum;
  wmma::fill_fragment(sum,0.0f);
  for(int k0=0;k0<n;k0+=8)
  {
    for(int t=lane;t<128;t+=32)
    {
      const int i=i0+t/8,k=k0+t%8;
      const float av=i<n&&k<n?a[(std::size_t(e)*n+i)*n+k]:0.0f;
      ah[t]=wmma::__float_to_tf32(av);
      if(ThreeProduct)al[t]=wmma::__float_to_tf32(av-ah[t]);
      const int bk=k0+t/16,j=j0+t%16;
      const float bv=bk<n&&j<m?b[(std::size_t(e)*n+bk)*m+j]:0.0f;
      bh[t]=wmma::__float_to_tf32(bv);
      if(ThreeProduct)bl[t]=wmma::__float_to_tf32(bv-bh[t]);
    }
    __syncwarp();
    wmma::load_matrix_sync(a_hi,ah,8);wmma::load_matrix_sync(b_hi,bh,16);
    wmma::mma_sync(sum,a_hi,b_hi,sum);
    if(ThreeProduct)
    {
      wmma::load_matrix_sync(a_lo,al,8);wmma::load_matrix_sync(b_lo,bl,16);
      wmma::mma_sync(sum,a_hi,b_lo,sum);wmma::mma_sync(sum,a_lo,b_hi,sum);
    }
    __syncwarp();
  }
  wmma::store_matrix_sync(out,sum,16,wmma::mem_row_major);__syncwarp();
  for(int t=lane;t<256;t+=32)
    if(i0+t/16<n&&j0+t%16<m)
      c[(std::size_t(e)*n+i0+t/16)*m+j0+t%16]=out[t];
}

struct Panels
{
  int ne=0,n=0,m=0;long long useful_columns=0;double setup_s=0,reference_s=0,omitted_face_relative=0;
  std::vector<double> d,inverse,b,reference;
};
Panels Build(int dim,int nx,int p,int nrhs,double alpha)
{
  auto start=Clock::now();
  mfem::Mesh serial=dim==2?mfem::Mesh::MakeCartesian2D(nx,nx,mfem::Element::QUADRILATERAL,true,1.,1.):
    mfem::Mesh::MakeCartesian3D(nx,nx,nx,mfem::Element::HEXAHEDRON,1.,1.,1.);
  mfem::ParMesh mesh(MPI_COMM_WORLD,serial);
  mfem::L2_FECollection fec(p,dim,mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fes(&mesh,&fec);
  mfem::FunctionCoefficient reaction([](const mfem::Vector& x)
    {return 1./(1.+2.2*(1.+std::tanh((x(0)-.45)/.08)));});
  mfem::FunctionCoefficient diffusion([=](const mfem::Vector& x)
    {return alpha/(1.+2.2*(1.+std::tanh((x(0)-.45)/.08)));});
  mfem::BilinearForm form(&fes);
  form.AddDomainIntegrator(new mfem::MassIntegrator(reaction));
  form.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion));
  form.AddInteriorFaceIntegrator(new mfem::DGDiffusionIntegrator(diffusion,-1.,4.*(p+1)*(p+1)));
  form.Assemble();form.Finalize();
  const mfem::SparseMatrix& a=form.SpMat();
  const auto& nodes=fes.GetFE(0)->GetNodes();
  std::vector<int> bubble,trace;
  for(int j=0;j<nodes.GetNPoints();++j)
  {
    const auto& ip=nodes.IntPoint(j);
    bool face=ip.x<1e-12||ip.x>1.-1e-12||ip.y<1e-12||ip.y>1.-1e-12;
    if(dim==3)face=face||ip.z<1e-12||ip.z>1.-1e-12;
    (face?trace:bubble).push_back(j);
  }
  Panels out;out.ne=mesh.GetNE();out.n=bubble.size();
  MFEM_VERIFY(out.n>0,"A panel requires interior basis functions.");
  std::vector<std::vector<int>> ids(out.ne),columns(out.ne);
  mfem::Array<int> dofs;
  std::vector<bool> is_trace(fes.GetVSize(),false);
  for(int e=0;e<out.ne;++e)
  {fes.GetElementDofs(e,dofs);ids[e].assign(dofs.begin(),dofs.end());for(int j:trace)is_trace[dofs[j]]=true;}
  const int* offsets=a.GetI();const int* indices=a.GetJ();const double* values=a.GetData();
  long double omitted=0,total_face=0;
  for(int e=0;e<out.ne;++e)
  {
    std::set<int> connected;
    double scale=0;
    for(int i:bubble)for(int k=offsets[ids[e][i]];k<offsets[ids[e][i]+1];++k)scale=std::max(scale,std::abs(values[k]));
    for(int i:bubble)for(int k=offsets[ids[e][i]];k<offsets[ids[e][i]+1];++k)
      if(is_trace[indices[k]]&&std::abs(values[k])>1e-13*scale)connected.insert(indices[k]);
    columns[e].assign(connected.begin(),connected.end());
    for(int i:bubble)for(int k=offsets[ids[e][i]];k<offsets[ids[e][i]+1];++k)
      if(is_trace[indices[k]])
      {const long double square=static_cast<long double>(values[k])*values[k];total_face+=square;
       if(!connected.count(indices[k]))omitted+=square;}
    const int count=nrhs?nrhs:1+columns[e].size();
    out.m=std::max(out.m,count);out.useful_columns+=count;
  }
  out.omitted_face_relative=total_face?std::sqrt(double(omitted/total_face)):0;
  MFEM_VERIFY(out.omitted_face_relative<1e-12,"Pruned face couplings exceed the reference threshold.");
  out.d.resize(std::size_t(out.ne)*out.n*out.n);out.inverse.resize(out.d.size());
  out.b.assign(std::size_t(out.ne)*out.n*out.m,0.);out.reference=out.b;
  for(int e=0;e<out.ne;++e)
  {
    mfem::DenseMatrix d(out.n),inverse(out.n),rhs(out.n,out.m),solution(out.n,out.m);
    rhs=0.;
    for(int i=0;i<out.n;++i)for(int j=0;j<out.n;++j)
      d(i,j)=a(ids[e][bubble[i]],ids[e][bubble[j]]);
    for(int i=0;i<out.n;++i)
    {
      if(nrhs)for(int j=0;j<nrhs;++j)rhs(i,j)=std::sin(.23*(1+i)+.31*j+.17*e)+.1*std::cos(.61*j*i);
      else
      {
        rhs(i,0)=std::sin(.23*(1+i)+.17*e);
        for(int j=0;j<int(columns[e].size());++j)rhs(i,j+1)=a(ids[e][bubble[i]],columns[e][j]);
      }
    }
    mfem::DenseMatrix identity(out.n);identity=0.;for(int i=0;i<out.n;++i)identity(i,i)=1.;
    mfem::DenseMatrixInverse solve(d,true);solve.Mult(identity,inverse);
    const auto reference_start=Clock::now();solve.Mult(rhs,solution);out.reference_s+=Seconds(reference_start);
    for(int i=0;i<out.n;++i)for(int j=0;j<out.n;++j)
    {out.d[(std::size_t(e)*out.n+i)*out.n+j]=d(i,j);out.inverse[(std::size_t(e)*out.n+i)*out.n+j]=inverse(i,j);}
    for(int i=0;i<out.n;++i)for(int j=0;j<out.m;++j)
    {out.b[(std::size_t(e)*out.n+i)*out.m+j]=rhs(i,j);out.reference[(std::size_t(e)*out.n+i)*out.m+j]=solution(i,j);}
  }
  out.setup_s=Seconds(start);return out;
}
struct Errors { double solution,residual,maximum_column_solution,maximum_column_residual; };
Errors Error(const Panels& p,const std::vector<double>& x)
{
  long double err=0,ref=0,res=0,bnorm=0;
  double worst_error=0,worst_residual=0;
  for(int e=0;e<p.ne;++e)for(int j=0;j<p.m;++j)
  {
    long double ce=0,cr=0,cs=0,cb=0;
    for(int i=0;i<p.n;++i)
    {
      const auto at=(std::size_t(e)*p.n+i)*p.m+j;
      MFEM_VERIFY(std::isfinite(x[at]),"Nonfinite panel output.");
      const long double delta=x[at]-p.reference[at];ce+=delta*delta;cr+=static_cast<long double>(p.reference[at])*p.reference[at];
      long double r=-p.b[at];
      for(int k=0;k<p.n;++k)r+=static_cast<long double>(p.d[(std::size_t(e)*p.n+i)*p.n+k])*x[(std::size_t(e)*p.n+k)*p.m+j];
      cs+=r*r;cb+=static_cast<long double>(p.b[at])*p.b[at];
    }
    if(cr>0)worst_error=std::max(worst_error,std::sqrt(double(ce/cr)));
    else MFEM_VERIFY(ce==0,"A padded zero column acquired an output.");
    if(cb>0)worst_residual=std::max(worst_residual,std::sqrt(double(cs/cb)));
    err+=ce;ref+=cr;res+=cs;bnorm+=cb;
  }
  return {std::sqrt(double(err/ref)),std::sqrt(double(res/bnorm)),worst_error,worst_residual};
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cuda");
  int ranks=0;MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  MFEM_VERIFY(ranks==1,"This local SIPG fixture is a rank-one probe.");
  int dim=2,nx=3,p=5,nrhs=0,repeats=100;
  for(int i=1;i<argc;++i)
  {
    const std::string key=argv[i];MFEM_VERIFY(i+1<argc,"Missing option value.");const int v=std::stoi(argv[++i]);
    if(key=="--dim")dim=v;else if(key=="--nx")nx=v;else if(key=="--order")p=v;
    else if(key=="--rhs")nrhs=v;else if(key=="--repeats")repeats=v;else MFEM_ABORT("Unknown panel option.");
  }
  MFEM_VERIFY((dim==2||dim==3)&&nx>=1&&nx<=16&&p>=2&&p<=11&&nrhs>=0&&nrhs<=64&&repeats>0,"Invalid bounded panel configuration.");
  MFEM_VERIFY(std::pow(double(nx),dim)*std::pow(double(p-1),2*dim)<=(1<<23),
              "Requested panel factors exceed this bounded probe's storage limit.");
  auto panels=Build(dim,nx,p,nrhs,1./(nx*nx*(p+1.)));
  const int n=panels.n,m=panels.m,ne=panels.ne;
  const auto rhs_nonzero=std::count_if(panels.b.begin(),panels.b.end(),[](double x){return x!=0.;});
  const auto upload_start=Clock::now();
  Buffer<double> a64(panels.inverse.size()),b64(panels.b.size()),c64(panels.b.size());
  Buffer<float> a32(panels.inverse.size()),b32(panels.b.size()),c32(panels.b.size());
  CUDA(cudaMemcpy(a64.data,panels.inverse.data(),a64.size*sizeof(double),cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(b64.data,panels.b.data(),b64.size*sizeof(double),cudaMemcpyHostToDevice));
  const double upload_s=Seconds(upload_start);
  const auto conversion_start=Clock::now();
  Convert<<<(a64.size+255)/256,256>>>(a64.size,a64.data,a32.data);
  Convert<<<(b64.size+255)/256,256>>>(b64.size,b64.data,b32.data);CUDA(cudaDeviceSynchronize());
  const double conversion_s=Seconds(conversion_start);
  cublasHandle_t handle;BLAS(cublasCreate(&handle));BLAS(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
  cudaEvent_t start,stop;CUDA(cudaEventCreate(&start));CUDA(cudaEventCreate(&stop));
  for(const std::string mode:{"simt64","cublas64","simt32","cublas32","cublas_tf32","tf32","tf32x3"})
  {
    BLAS(cublasSetMathMode(handle,mode=="cublas_tf32"?CUBLAS_DEFAULT_MATH:CUBLAS_PEDANTIC_MATH));
    auto launch=[&]()
    {
      if(mode=="simt64")Ordinary<<<dim3((n*m+255)/256,ne),256>>>(n,m,a64.data,b64.data,c64.data);
      else if(mode=="simt32")Ordinary<<<dim3((n*m+255)/256,ne),256>>>(n,m,a32.data,b32.data,c32.data);
      else if(mode=="cublas64")
      {const double one=1,zero=0;BLAS(cublasDgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,n,&one,b64.data,m,n*m,a64.data,n,n*n,&zero,c64.data,m,n*m,ne));}
      else if(mode=="cublas32")
      {const float one=1,zero=0;BLAS(cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,n,&one,b32.data,m,n*m,a32.data,n,n*n,&zero,c32.data,m,n*m,ne));}
      else if(mode=="cublas_tf32")
      {const float one=1,zero=0;BLAS(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,
        m,n,n,&one,b32.data,CUDA_R_32F,m,static_cast<long long>(n)*m,
        a32.data,CUDA_R_32F,n,static_cast<long long>(n)*n,&zero,c32.data,CUDA_R_32F,m,
        static_cast<long long>(n)*m,ne,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT));}
      else if(mode=="tf32")Tensor<false><<<dim3(ne,(n+15)/16,(m+15)/16),32>>>(n,m,a32.data,b32.data,c32.data);
      else Tensor<true><<<dim3(ne,(n+15)/16,(m+15)/16),32>>>(n,m,a32.data,b32.data,c32.data);
    };
    launch();CUDA(cudaDeviceSynchronize());CUDA(cudaEventRecord(start));
    for(int i=0;i<repeats;++i)launch();
    CUDA(cudaEventRecord(stop));CUDA(cudaEventSynchronize(stop));float ms=0;CUDA(cudaEventElapsedTime(&ms,start,stop));
    const bool lower_precision=mode!="simt64"&&mode!="cublas64";
    CUDA(cudaEventRecord(start));
    for(int i=0;i<repeats;++i)
    {
      if(lower_precision)Convert<<<(b64.size+255)/256,256>>>(b64.size,b64.data,b32.data);
      launch();
      if(lower_precision)ConvertBack<<<(c64.size+255)/256,256>>>(c64.size,c32.data,c64.data);
    }
    CUDA(cudaEventRecord(stop));CUDA(cudaEventSynchronize(stop));float pipeline_ms=0;CUDA(cudaEventElapsedTime(&pipeline_ms,start,stop));
    std::vector<double> output(panels.b.size());
    CUDA(cudaMemcpy(output.data(),c64.data,c64.size*sizeof(double),cudaMemcpyDeviceToHost));
    auto error=Error(panels,output);
    const long long useful=static_cast<long long>(n)*n*panels.useful_columns;
    const long long executed=mode.find("tf32")==0?static_cast<long long>(ne)*((n+15)/16*16)*((m+15)/16*16)*((n+7)/8*8)*(mode=="tf32x3"?3:1):static_cast<long long>(ne)*n*n*m;
    std::cout<<std::setprecision(12)<<"condensation_panel mode="<<mode<<" dim="<<dim<<" p="<<p<<" elements="<<ne<<" interior="<<n<<" rhs_max="<<m
      <<" useful_columns="<<panels.useful_columns<<" rhs_kind="<<(nrhs?"manufactured_stage_defects":"manufactured_load_and_face_responses")
      <<" rhs_exact_nonzero_entries="<<rhs_nonzero<<" rhs_stored_entries="<<panels.b.size()
      <<" shared_fixture_preparation_s="<<panels.setup_s-panels.reference_s<<" reference_s="<<panels.reference_s
      <<" omitted_face_relative="<<panels.omitted_face_relative
      <<" upload_all_buffers_s="<<upload_s<<" initial_conversion_s="<<conversion_s<<" apply_us="<<1000.*ms/repeats
      <<" with_rhs_conversion_us="<<1000.*pipeline_ms/repeats
      <<" harness_device_bytes="<<(sizeof(double)+sizeof(float))*(a64.size+2*b64.size)
      <<" useful_products="<<useful<<" logical_dense_products="<<static_cast<long long>(ne)*n*n*m
      <<" custom_executed_products="<<(mode.find("cublas")==0?-1:executed)<<" solution_relative="<<error.solution<<" current_residual="<<error.residual
      <<" maximum_column_solution_relative="<<error.maximum_column_solution<<" maximum_column_residual="<<error.maximum_column_residual<<'\n';
    if(!lower_precision)MFEM_VERIFY(error.maximum_column_solution<1e-10&&error.maximum_column_residual<1e-10,"FP64 panel oracle failed.");
  }
  CUDA(cudaEventDestroy(start));CUDA(cudaEventDestroy(stop));BLAS(cublasDestroy(handle));return 0;
}

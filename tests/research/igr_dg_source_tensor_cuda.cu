// Test-only fixed DG derivative/halo map plus nonlinear IGR source load.
// Device timings include centering, halo gather, conversion and source assembly.
// Euler reconstruction, the elliptic solve and time advancement are NOT timed.
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
  T* p=nullptr;std::size_t n;
  explicit Device(std::size_t size):n(size){Check(cudaMalloc(&p,n*sizeof(T)));}
  ~Device(){cudaFree(p);}
  void Set(const std::vector<T>& a){if(a.size()!=n)throw std::runtime_error("device size");Check(cudaMemcpy(p,a.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> Get(){std::vector<T>a(n);Check(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return a;}
};
template<class T> std::vector<T> Read(std::ifstream& f,std::size_t n){std::vector<T>a(n);f.read(reinterpret_cast<char*>(a.data()),n*sizeof(T));if(!f)throw std::runtime_error("short input");return a;}
__device__ double Velocity(const double* u,int node){return u[2*node+1]/u[2*node];}

template<class Real,bool FloatSource=false> __global__ void Ordinary(int n,int q,int ne,double alpha,const Real* G,const typename std::conditional<FloatSource,float,double>::type* L,const double* mean,const double* h,const double* u,const int* halo,double* gradient,double* load){
  const int e=blockIdx.x,t=threadIdx.x,k=n+2;
  using SourceReal=typename std::conditional<FloatSource,float,double>::type;
  __shared__ double center,values[66];__shared__ Real operands[66];__shared__ SourceReal source[128];
  if(t<32){double sum=0.;for(int j=t;j<n;j+=32){const double v=Velocity(u,e*n+j);values[j]=v;sum+=mean[j]*v;}
    for(int offset=16;offset;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);if(t==0)center=sum;}
  __syncthreads();
  if(t<k)operands[t]=Real((t<n?values[t]:Velocity(u,halo[2*e+t-n]))-center);
  __syncthreads();
  if(t<q){Real sum=0;for(int j=0;j<k;++j)sum+=G[t*k+j]*operands[j];const double g=double(sum)/h[e];gradient[e*q+t]=g;source[t]=SourceReal(2.*alpha*g*g);}
  __syncthreads();
  if(t<n){SourceReal sum=0;for(int j=0;j<q;++j)sum+=L[t*q+j]*source[j];load[e*n+t]=h[e]*double(sum);}
}
__global__ void Pack(int n,int ne,const double* mean,const double* u,const int* halo,float* b){
  const int e=blockIdx.x,t=threadIdx.x;__shared__ double center;
  if(t==0){center=0.;for(int j=0;j<n;++j)center+=mean[j]*Velocity(u,e*n+j);}
  __syncthreads();if(t<n+2){int node=t<n?e*n+t:halo[2*e+t-n];b[t*ne+e]=float(Velocity(u,node)-center);}
}
template<bool Three> __global__ void Tensor(int q,int k,int ne,const float* a,const float* b,float* c){
  using namespace nvcuda;
  int i0=16*blockIdx.x,j0=16*blockIdx.y,lane=threadIdx.x;
  __shared__ __align__(32) float ah[128],al[128],bh[128],bl[128],out[256];
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> A,Ar;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> B,Br;
  wmma::fragment<wmma::accumulator,16,16,8,float> C;wmma::fill_fragment(C,0.f);
  for(int k0=0;k0<k;k0+=8){
    for(int t=lane;t<128;t+=32){int i=i0+t/8,j=k0+t%8;float av=i<q&&j<k?a[i*k+j]:0.f;ah[t]=wmma::__float_to_tf32(av);if(Three)al[t]=wmma::__float_to_tf32(av-ah[t]);
      i=k0+t/16;j=j0+t%16;float bv=i<k&&j<ne?b[i*ne+j]:0.f;bh[t]=wmma::__float_to_tf32(bv);if(Three)bl[t]=wmma::__float_to_tf32(bv-bh[t]);}
    __syncwarp();wmma::load_matrix_sync(A,ah,8);wmma::load_matrix_sync(B,bh,16);wmma::mma_sync(C,A,B,C);
    if(Three){wmma::load_matrix_sync(Ar,al,8);wmma::load_matrix_sync(Br,bl,16);wmma::mma_sync(C,A,Br,C);wmma::mma_sync(C,Ar,B,C);}__syncwarp();
  }
  wmma::store_matrix_sync(out,C,16,wmma::mem_row_major);__syncwarp();
  for(int t=lane;t<256;t+=32){int i=i0+t/16,j=j0+t%16;if(i<q&&j<ne)c[i*ne+j]=out[t];}
}
template<bool FloatSource=false> __global__ void Source(int n,int q,int ne,double alpha,const double* L,const double* h,const float* c,double* gradient,double* load){
  int e=blockIdx.x,t=threadIdx.x;__shared__ double g[128];
  if(t<q){g[t]=double(c[t*ne+e])/h[e];gradient[e*q+t]=g[t];}__syncthreads();
  if(t<n){if(FloatSource){float sum=0.;for(int j=0;j<q;++j)sum+=float(L[t*q+j])*float(2.*alpha*g[j]*g[j]);load[e*n+t]=h[e]*double(sum);}
    else{double sum=0.;for(int j=0;j<q;++j)sum+=L[t*q+j]*2.*alpha*g[j]*g[j];load[e*n+t]=h[e]*sum;}}
}
// One warp owns sixteen elements and retains both tensor operands on chip.
// Both derivative and load projection use three compensated TF32 products.
__global__ void FusedTensor(int n,int q,int ne,double alpha,const float* G,const float* L,const double* mean,const double* h,const double* u,const int* halo,double* gradient,double* load){
  using namespace nvcuda;const int e0=16*blockIdx.x,lane=threadIdx.x,k=n+2;
  __shared__ double center[16];
  __shared__ __align__(32) float panel[66*16],source[128*16],ah[128],al[128],bh[128],bl[128],out[256];
  if(lane<16){int e=e0+lane;double v=0.;if(e<ne)for(int j=0;j<n;++j)v+=mean[j]*Velocity(u,e*n+j);center[lane]=v;}
  __syncwarp();
  for(int t=lane;t<k*16;t+=32){int j=t%16,i=t/16,e=e0+j;double v=0.;if(e<ne){int node=i<n?e*n+i:halo[2*e+i-n];v=Velocity(u,node)-center[j];}panel[t]=float(v);}
  __syncwarp();
  for(int phase=0;phase<2;++phase){int rows=phase?n:q,inner=phase?q:k;const float* matrix=phase?L:G;
    for(int i0=0;i0<rows;i0+=16){
      wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major>A,Ar;
      wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major>B,Br;
      wmma::fragment<wmma::accumulator,16,16,8,float>C;wmma::fill_fragment(C,0.f);
      for(int k0=0;k0<inner;k0+=8){
        for(int t=lane;t<128;t+=32){int i=i0+t/8,j=k0+t%8;float v=i<rows&&j<inner?matrix[i*inner+j]:0.f;ah[t]=wmma::__float_to_tf32(v);al[t]=wmma::__float_to_tf32(v-ah[t]);
          i=k0+t/16;j=t%16;v=i<inner?(phase?source[i*16+j]:panel[i*16+j]):0.f;bh[t]=wmma::__float_to_tf32(v);bl[t]=wmma::__float_to_tf32(v-bh[t]);}
        __syncwarp();wmma::load_matrix_sync(A,ah,8);wmma::load_matrix_sync(Ar,al,8);wmma::load_matrix_sync(B,bh,16);wmma::load_matrix_sync(Br,bl,16);
        wmma::mma_sync(C,A,B,C);wmma::mma_sync(C,A,Br,C);wmma::mma_sync(C,Ar,B,C);__syncwarp();
      }
      wmma::store_matrix_sync(out,C,16,wmma::mem_row_major);__syncwarp();
      for(int t=lane;t<256;t+=32){int i=i0+t/16,j=t%16,e=e0+j;if(i<rows){if(phase){if(e<ne)load[e*n+i]=h[e]*double(out[t]);}
        else{double g=e<ne?double(out[t])/h[e]:0.;if(e<ne)gradient[e*q+i]=g;source[i*16+j]=float(2.*alpha*g*g);}}}__syncwarp();
    }
  }
}
}
int main(int argc,char** argv){try{
  std::string input,output;int copies=1,repeats=100;
  for(int i=1;i<argc;++i){std::string flag=argv[i];if(++i==argc)throw std::runtime_error("missing option");if(flag=="--input")input=argv[i];else if(flag=="--output")output=argv[i];else if(flag=="--copies")copies=std::stoi(argv[i]);else if(flag=="--repeats")repeats=std::stoi(argv[i]);else throw std::runtime_error("unknown option");}
  if(input.empty()||copies<1||repeats<1||copies>4096)throw std::runtime_error("invalid source experiment options");
  std::ifstream f(input,std::ios::binary);auto header=Read<int>(f,3);const int n=header[0],q=header[1],base=header[2];
  if(n<2||n>64||q<n||q>128||base<2||base>65536)throw std::runtime_error("invalid dimensions");
  double alpha=Read<double>(f,1)[0];const int k=n+2,ne=base*copies;
  auto G=Read<double>(f,q*k),L=Read<double>(f,n*q),mean=Read<double>(f,n),h0=Read<double>(f,base),u0=Read<double>(f,2*n*base);
  if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("extra input");
  std::vector<double>h(ne),u(2*n*ne);std::vector<int>halo(2*ne);
  for(int copy=0;copy<copies;++copy){std::copy(h0.begin(),h0.end(),h.begin()+copy*base);std::copy(u0.begin(),u0.end(),u.begin()+copy*2*n*base);
    for(int e=0;e<base;++e){int ge=copy*base+e;halo[2*ge]=e?ge*n-1:ge*n;halo[2*ge+1]=e+1<base?(ge+1)*n:(ge+1)*n-1;}}
  cudaDeviceProp prop;Check(cudaGetDeviceProperties(&prop,0));
  auto setup_start=std::chrono::steady_clock::now();
  Device<double>dG(G.size()),dL(L.size()),dm(n),dh(ne),du(u.size()),dg(ne*q),dl(ne*n);Device<int>di(halo.size());
  Device<float>af(G.size()),lf(L.size()),b(k*ne),c(q*ne);std::vector<float>gf(G.begin(),G.end()),lfloat(L.begin(),L.end());
  dG.Set(G);dL.Set(L);dm.Set(mean);dh.Set(h);du.Set(u);di.Set(halo);af.Set(gf);lf.Set(lfloat);Check(cudaDeviceSynchronize());
  double setup=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_start).count();
  cudaEvent_t begin,end;Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));
  std::vector<double>reference_g,reference_l;std::ofstream fields;if(!output.empty())fields.open(output,std::ios::binary);
  std::cout<<std::setprecision(17)<<"{\"gpu\":\""<<prop.name<<"\",\"n\":"<<n<<",\"q\":"<<q<<",\"base_elements\":"<<base<<",\"copies\":"<<copies<<",\"elements\":"<<ne<<",\"repeats\":"<<repeats<<",\"setup_copy_seconds\":"<<setup<<",\"rows\":[";
  const int threads=std::max(q,k)<=32?32:(std::max(q,k)<=64?64:128);
  for(int method=0;method<7;++method){
    auto launch=[&](){if(method==0)Ordinary<double><<<ne,threads>>>(n,q,ne,alpha,dG.p,dL.p,dm.p,dh.p,du.p,di.p,dg.p,dl.p);
      else if(method==1)Ordinary<float><<<ne,threads>>>(n,q,ne,alpha,af.p,dL.p,dm.p,dh.p,du.p,di.p,dg.p,dl.p);
      else if(method==4)Ordinary<float,true><<<ne,threads>>>(n,q,ne,alpha,af.p,lf.p,dm.p,dh.p,du.p,di.p,dg.p,dl.p);
      else if(method==6)FusedTensor<<<(ne+15)/16,32>>>(n,q,ne,alpha,af.p,lf.p,dm.p,dh.p,du.p,di.p,dg.p,dl.p);
      else{Pack<<<ne,threads>>>(n,ne,dm.p,du.p,di.p,b.p);dim3 grid((q+15)/16,(ne+15)/16);
        if(method==2)Tensor<false><<<grid,32>>>(q,k,ne,af.p,b.p,c.p);else Tensor<true><<<grid,32>>>(q,k,ne,af.p,b.p,c.p);
        if(method==5)Source<true><<<ne,threads>>>(n,q,ne,alpha,dL.p,dh.p,c.p,dg.p,dl.p);else Source<false><<<ne,threads>>>(n,q,ne,alpha,dL.p,dh.p,c.p,dg.p,dl.p);}};
    for(int j=0;j<5;++j)launch();Check(cudaGetLastError());Check(cudaDeviceSynchronize());
    auto wall_start=std::chrono::steady_clock::now();Check(cudaEventRecord(begin));for(int j=0;j<repeats;++j)launch();Check(cudaEventRecord(end));Check(cudaEventSynchronize(end));Check(cudaGetLastError());
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()/repeats;float ms;Check(cudaEventElapsedTime(&ms,begin,end));
    auto gv=dg.Get(),lv=dl.Get();if(method==0){reference_g=gv;reference_l=lv;}
    long double ge=0,gn=0,le=0,ln=0;for(std::size_t j=0;j<gv.size();++j){ge+=(gv[j]-reference_g[j])*(gv[j]-reference_g[j]);gn+=reference_g[j]*reference_g[j];}for(std::size_t j=0;j<lv.size();++j){le+=(lv[j]-reference_l[j])*(lv[j]-reference_l[j]);ln+=reference_l[j]*reference_l[j];}
    if(!std::isfinite(double(ge+gn+le+ln)))throw std::runtime_error("nonfinite output");
    if(fields){fields.write(reinterpret_cast<const char*>(gv.data()),base*q*sizeof(double));fields.write(reinterpret_cast<const char*>(lv.data()),base*n*sizeof(double));}
    long long products=1LL*q*k*ne,padded=1LL*((q+15)/16)*16*((ne+15)/16)*16*((k+7)/8)*8;
    const char* names[]={"ordinary-fp64","ordinary-fp32","tensor-tf32","tensor-tf32x3","ordinary-fp32-source-fp32","tensor-tf32x3-source-fp32","fused-tf32x3"};
    long long executed=(method<2||method==4)?0:padded*(method==2?1:3);
    if(method==6)executed+=3LL*((n+15)/16)*16*((ne+15)/16)*16*((q+7)/8)*8;
    if(method)std::cout<<',';std::cout<<"{\"method\":\""<<names[method]<<"\",\"device_seconds\":"<<ms*.001/repeats<<",\"host_submit_and_device_seconds\":"<<wall<<",\"gradient_relative_euclidean\":"<<std::sqrt(double(ge/std::max(gn,1.e-100L)))<<",\"load_relative_euclidean\":"<<std::sqrt(double(le/std::max(ln,1.e-100L)))<<",\"useful_derivative_products\":"<<products<<",\"tensor_executed_products\":"<<executed<<",\"source_projection_products\":"<<1LL*n*q*ne<<'}';
  }
  std::cout<<"]}\n";cudaEventDestroy(begin);cudaEventDestroy(end);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

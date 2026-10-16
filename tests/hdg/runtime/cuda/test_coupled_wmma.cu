#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{
#include "coupled_wmma_gemm.cuh"

struct View
{
  int buffer,ld;
  long long stride,offset;
  std::size_t At(int e,int row,int col) const
  {return static_cast<std::size_t>(offset+e*stride+row+static_cast<long long>(col)*ld);}
};
struct DeviceBuffers
{
  std::array<float*,3> data{};
  ~DeviceBuffers(){for(float* p:data)if(p)cudaFree(p);}
};
enum class Fault {none,a_nan,b_inf,c_nan,overflow,tf32_rounding_overflow,ignored_c_nan};
float Value(int e,int row,int col,int tag)
{
  // Deliberately not TF32-exact, asymmetric, and different in each real batch.
  return static_cast<float>(0.031*std::sin(0.79*(row+1)+0.43*(col+2)+tag+e)
      +0.047*std::cos(0.31*(row+3)-0.67*(col+1)+2*tag+e));
}
bool Run(const std::string& name,int rows,int columns,int inner,
         View a,View b,View c,bool subtract,Fault fault=Fault::none)
{
  constexpr int elements=2;
  std::array<std::vector<float>,3> original;
  const auto reserve=[&](View v,int r,int cols)
  {
    original[v.buffer].resize(std::max(original[v.buffer].size(),
        v.At(elements-1,r-1,cols-1)+33),std::numeric_limits<float>::quiet_NaN());
  };
  reserve(a,rows,inner);reserve(b,inner,columns);reserve(c,rows,columns);
  for(auto& buffer:original)if(buffer.empty())buffer.resize(33,7.0f);
  // Padding, inter-element gaps, and guards remain NaN. Reading a padded tail
  // instead of supplying zero therefore cannot silently pass this fixture.
  for(int e=0;e<elements;++e)
  {
    for(int j=0;j<inner;++j)for(int i=0;i<rows;++i)
      original[a.buffer][a.At(e,i,j)]=Value(e,i,j,1);
    for(int j=0;j<columns;++j)for(int k=0;k<inner;++k)
      original[b.buffer][b.At(e,k,j)]=Value(e,k,j,4);
    for(int j=0;j<columns;++j)for(int i=0;i<rows;++i)
      original[c.buffer][c.At(e,i,j)]=Value(e,i,j,7);
  }
  if(fault==Fault::a_nan)original[a.buffer][a.At(0,rows-1,inner-1)]=std::numeric_limits<float>::quiet_NaN();
  if(fault==Fault::b_inf)original[b.buffer][b.At(0,inner-1,columns-1)]=std::numeric_limits<float>::infinity();
  if(fault==Fault::c_nan||fault==Fault::ignored_c_nan)
    original[c.buffer][c.At(0,0,0)]=std::numeric_limits<float>::quiet_NaN();
  // The original float is finite, and all B entries have magnitude < 0.1,
  // so its original products fit FP32; conversion to TF32 rounds to infinity.
  if(fault==Fault::tf32_rounding_overflow)
    original[a.buffer][a.At(0,0,0)]=std::numeric_limits<float>::max();
  if(fault==Fault::overflow)
  {
    original[a.buffer][a.At(0,0,0)]=1.e30f;
    original[b.buffer][b.At(0,0,0)]=1.e30f;
  }
  std::array<std::vector<unsigned char>,3> output_mask;
  for(int i=0;i<3;++i)output_mask[i].resize(original[i].size());
  for(int e=0;e<elements;++e)for(int col=0;col<columns;++col)for(int row=0;row<rows;++row)
    output_mask[c.buffer][c.At(e,row,col)]=1;
  // The production updates alias allocations, but never alias live entries.
  for(int e=0;e<elements;++e)
  {
    for(int col=0;col<inner;++col)for(int row=0;row<rows;++row)
      if(output_mask[a.buffer][a.At(e,row,col)])throw "invalid fixture A/output overlap";
    for(int col=0;col<columns;++col)for(int row=0;row<inner;++row)
      if(output_mask[b.buffer][b.At(e,row,col)])throw "invalid fixture B/output overlap";
  }
  DeviceBuffers device;
  for(int i=0;i<3;++i)
  {
    if(cudaMalloc(reinterpret_cast<void**>(&device.data[i]),original[i].size()*sizeof(float))!=cudaSuccess||
       cudaMemcpy(device.data[i],original[i].data(),original[i].size()*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
      throw "fixture allocation/copy failed";
  }
  if(!CoupledWmmaTf32Gemm(rows,columns,inner,device.data[a.buffer]+a.offset,a.ld,a.stride,
      device.data[b.buffer]+b.offset,b.ld,b.stride,device.data[c.buffer]+c.offset,c.ld,c.stride,
      elements,subtract)||cudaDeviceSynchronize()!=cudaSuccess)
  {std::cerr<<name<<": WMMA launch failed\n";return false;}
  auto result=original;
  for(int i=0;i<3;++i)
    if(cudaMemcpy(result[i].data(),device.data[i],result[i].size()*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw "fixture readback failed";
  for(int i=0;i<3;++i)for(std::size_t j=0;j<result[i].size();++j)
    if(!output_mask[i][j]&&std::memcmp(&result[i][j],&original[i][j],sizeof(float))!=0)
    {std::cerr<<name<<": modified input/padding at buffer="<<i<<" index="<<j<<'\n';return false;}
  bool saw_nonfinite=false;
  for(int e=0;e<elements;++e)for(int col=0;col<columns;++col)for(int row=0;row<rows;++row)
  {
    const double got=result[c.buffer][c.At(e,row,col)];
    const bool injected=e==0&&fault!=Fault::none&&fault!=Fault::ignored_c_nan;
    if(injected)
    {saw_nonfinite=saw_nonfinite||!std::isfinite(got);continue;}
    // Independent original-operand CPU oracle; no WMMA layout or TF32 packing
    // is reused. The error bound accounts for two TF32-rounded operands and
    // FP32 accumulation, scaled by sum |a_k b_k| rather than cancellation in C.
    double product=0.0,scale=0.0;
    for(int k=0;k<inner;++k)
    {
      const double term=static_cast<double>(original[a.buffer][a.At(e,row,k)])*
          original[b.buffer][b.At(e,k,col)];
      product+=term;scale+=std::abs(term);
    }
    const double prior=subtract?original[c.buffer][c.At(e,row,col)]:0.0;
    const double expected=subtract?prior-product:product;
    const double bound=0.002*scale+8.0*std::numeric_limits<float>::epsilon()*std::abs(prior)+1.e-8;
    if(!std::isfinite(got)||std::abs(got-expected)>bound)
    {
      std::cerr<<name<<": mismatch e="<<e<<" row="<<row<<" col="<<col
          <<" got="<<got<<" expected="<<expected<<" bound="<<bound<<'\n';return false;
    }
  }
  if(fault!=Fault::none&&fault!=Fault::ignored_c_nan&&!saw_nonfinite)
  {std::cerr<<name<<": nonfinite construction was accepted\n";return false;}
  return true;
}

bool RealShapes(int n,int m)
{
  const int np=((n+15)/16)*16,mp=((m+15)/16)*16,q=((m+16)/16)*16;
  // Production stores each typed matrix in a double-sized allocation. Float
  // batch strides are twice the padded matrix size, not tightly packed.
  const long long ls=2LL*np*np,rs=2LL*np*q,cs=2LL*mp*np,ps=2LL*mp*mp;
  for(int first=0;first<np;first+=32)
  {
    const int end=std::min(first+32,np),width=end-first,remaining=np-end;
    if(remaining&&
       (!Run("LU-same-allocation",remaining,remaining,width,
          {0,np,ls,first*np+end},{0,np,ls,end*np+first},{0,np,ls,end*np+end},true)||
        !Run("RHS-forward-same-allocation",q,remaining,width,
          {0,q,rs,first*q},{1,np,ls,end*np+first},{0,q,rs,end*q},true)))return false;
  }
  for(int end=np;end>0;)
  {
    const int first=((end-1)/32)*32,width=end-first;
    if(first&&!Run("RHS-back-same-allocation",q,first,width,
        {0,q,rs,first*q},{1,np,ls,first},{0,q,rs,0},true))return false;
    end=first;
  }
  return Run("Schur-production-strides",mp,mp,np,
      {0,q,rs,0},{1,np,cs,0},{2,mp,ps,0},false);
}

bool RaggedAndFailure()
{
  const View a{0,27,27*17+19,3},b{1,21,21*23+17,5},c{2,31,31*23+23,7};
  for(bool subtract:{false,true})
    if(!Run("ragged-17x19x13",17,19,13,a,b,c,subtract))return false;
  for(const auto fault:{Fault::a_nan,Fault::b_inf,Fault::c_nan,Fault::overflow,Fault::tf32_rounding_overflow})
    if(!Run("nonfinite",17,19,13,a,b,c,true,fault))return false;
  if(!Run("overwrite-ignores-C",17,19,13,a,b,c,false,Fault::ignored_c_nan))return false;
  for(const int k:{1,7,8,9,31,33})
    if(!Run("tiny-K-tail",1,3,k,{0,5,5LL*k+11,1},
        {1,k+3,3LL*(k+3)+13,2},{2,7,31,3},false))return false;
  // Invalid shape/leading dimension must reject before issuing a kernel.
  if(CoupledWmmaTf32Gemm(1,1,1,nullptr,1,1,nullptr,1,1,nullptr,1,1,1,false))return false;
  return true;
}
} // namespace

int main()
{
  try
  {
    if(!CoupledWmmaTf32Supported())
    {std::cerr<<"coupled-wmma requires SM80 or newer\n";return 1;}
    for(const auto shape:{std::array<int,2>{28,40},{112,80},{252,120},{448,160}})
      if(!RealShapes(shape[0],shape[1]))return 1;
    if(!RaggedAndFailure())return 1;
    std::cout<<"coupled-wmma passed real_panel_shapes=true aliased_subblocks=true "
        "ragged_tails=true subtract=true nonfinite_rejected=true backend=direct-wmma-tf32\n";
    return 0;
  }
  catch(const char* error){std::cerr<<error<<'\n';return 2;}
}

#include "../src/physics/igr/IGR_tensor_center.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using igrlab::igr_internal::AddTensorCenterResponse;
using igrlab::igr_internal::ApplyTensorCenterResponse;
using igrlab::igr_internal::ComposeTensorCenterResponse;

void Require(bool condition,const std::string& message)
{if(!condition)throw std::runtime_error(message);}
void CUDA(cudaError_t status)
{if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}

template<typename T> class DeviceBuffer
{
public:
  explicit DeviceBuffer(std::size_t count):count_(count)
  {CUDA(cudaMalloc(reinterpret_cast<void**>(&data_),count_*sizeof(T)));}
  ~DeviceBuffer(){if(data_)cudaFree(data_);}
  DeviceBuffer(const DeviceBuffer&)=delete;
  DeviceBuffer& operator=(const DeviceBuffer&)=delete;
  T* Data(){return data_;}
  void Upload(const std::vector<T>& values)
  {
    Require(values.size()==count_,"Center-map fixture upload size mismatch.");
    CUDA(cudaMemcpy(data_,values.data(),count_*sizeof(T),cudaMemcpyHostToDevice));
  }
  std::vector<T> Download()const
  {
    std::vector<T> values(count_);
    CUDA(cudaMemcpy(values.data(),data_,count_*sizeof(T),cudaMemcpyDeviceToHost));
    return values;
  }
private:
  std::size_t count_;T* data_=nullptr;
};

struct Factors
{
  int d=0,clipped=0;
  std::vector<double> forward_x,forward_y,back_x,back_y,inverse,response;
  std::vector<long double> row_l1,basis_reference;
  long double response_norm=0;
};

Factors MakeFactors(int d,bool clipped)
{
  Factors f;f.d=d;const int s=3*d,k=s*s,m=d*d;
  f.forward_x.resize(k);f.forward_y.resize(k);f.back_x.resize(k);f.back_y.resize(k);f.inverse.resize(k);
  const long double pi=std::acos(-1.0L);
  // Two different dense orthogonal transforms. Permuting/signing the y basis
  // makes an accidental x/y or inverse-layout interchange observable.
  for(int j=0;j<s;++j)for(int i=0;i<s;++i)
  {
    f.back_x[i+s*j]=static_cast<double>(std::sqrt(2.0L/s)*std::cos(pi*(i+.5L)*(j+.5L)/s));
    const int row=(5*i+1)%s,col=(7*j+3)%s;
    f.back_y[i+s*j]=static_cast<double>((j%3==1?-1.0L:1.0L)*std::sqrt((col==0?1.0L:2.0L)/s)*
      std::cos(pi*(row+.5L)*col/s));
  }
  for(int j=0;j<s;++j)for(int i=0;i<s;++i)
  {
    f.forward_x[i+s*j]=f.back_x[j+s*i];f.forward_y[i+s*j]=f.back_y[j+s*i];
    const double eigenvalue=.12+.009*i*i+.015*i+.21+.017*j*j+.027*j-(clipped?.72:.05);
    const double mass=clipped?.37:.19;if(eigenvalue<mass)++f.clipped;
    f.inverse[j+s*i]=1.0/std::max(mass,eigenvalue); // Existing (y-eigen,x-eigen) storage.
  }
  Require(clipped?f.clipped>0:f.clipped==0,"Center-map fixture did not exercise the requested clipping case.");
  f.response=ComposeTensorCenterResponse(d,f.forward_x,f.forward_y,f.back_x,f.back_y,f.inverse);
  Require(f.response.size()==std::size_t(m)*k,"Center-map composition shape mismatch.");
  f.row_l1.assign(m,0);
  for(int col=0;col<k;++col)for(int row=0;row<m;++row)
  {
    const long double v=f.response[row+m*col];
    Require(std::isfinite(v),"Center-map composition produced a nonfinite entry.");
    f.row_l1[row]+=std::abs(v);f.response_norm+=v*v;
  }
  f.response_norm=std::sqrt(f.response_norm);return f;
}

// Independent oracle: retain the full patch and apply the four 1-D maps in
// long double. This never evaluates the double eigen-index sum used to build H
// or uses the dense H*r loop used by its native CUDA application.
std::vector<long double> SeparablePatch(const Factors& f,const std::vector<long double>& input,bool inverse=false)
{
  const int s=3*f.d,k=s*s;
  Require(input.size()==std::size_t(k),"Separable oracle patch size mismatch.");
  std::vector<long double> a(k),b(k),c(k),out(k);
  for(int y=0;y<s;++y)for(int i=0;i<s;++i)
    for(int x=0;x<s;++x)a[i+s*y]+=static_cast<long double>(f.forward_x[i+s*x])*input[x+s*y];
  for(int j=0;j<s;++j)for(int i=0;i<s;++i)
  {
    for(int y=0;y<s;++y)b[i+s*j]+=static_cast<long double>(f.forward_y[j+s*y])*a[i+s*y];
    const long double diagonal=f.inverse[j+s*i];b[i+s*j]*=inverse?1.0L/diagonal:diagonal;
  }
  for(int y=0;y<s;++y)for(int i=0;i<s;++i)
    for(int j=0;j<s;++j)c[i+s*y]+=static_cast<long double>(f.back_y[y+s*j])*b[i+s*j];
  for(int y=0;y<s;++y)for(int x=0;x<s;++x)
    for(int i=0;i<s;++i)out[x+s*y]+=static_cast<long double>(f.back_x[x+s*i])*c[i+s*y];
  return out;
}

void CheckComposition(Factors& f)
{
  const int d=f.d,s=3*d,k=s*s,m=d*d;long double error=0,scale=0,worst=0;
  f.basis_reference.resize(std::size_t(m)*k);
  std::vector<long double> basis(k,0);
  for(int col=0;col<k;++col)
  {
    basis[col]=1;const auto expected=SeparablePatch(f,basis);basis[col]=0;
    for(int row=0;row<m;++row)
    {
      const long double reference=expected[d+row%d+s*(d+row/d)];
      f.basis_reference[row+m*col]=reference;
      const long double delta=static_cast<long double>(f.response[row+m*col])-reference;
      error+=delta*delta;scale+=reference*reference;worst=std::max(worst,std::abs(delta));
    }
  }
  Require(scale>0&&std::sqrt(error/scale)<2e-12L&&worst<2e-12L*f.response_norm,
    "Center-map host composition disagrees with the independent separable basis oracle.");
}

std::vector<long double> CancellationInput(const Factors& f)
{
  const int d=f.d,s=3*d;std::vector<long double> desired(s*s);
  for(int y=0;y<s;++y)for(int x=0;x<s;++x)
    if(x<d||x>=2*d||y<d||y>=2*d)desired[x+s*y]=std::sin(.37L*(1+x+5*y))+.2L*std::cos(.11L*(3*x-y));
  // A full-patch solution supported outside the center gives a near-null
  // center response, while every packed input entry remains nontrivial.
  return SeparablePatch(f,desired,true);
}

bool CancellationColumn(int e,int k,bool basis)
{return basis?e==k+2:e%9==5;}

template<typename T> std::vector<T> PackedInputs(const Factors& f,int ne,bool basis)
{
  const int s=3*f.d,k=s*s;std::vector<T> packed(std::size_t(ne)*k,0);
  const auto cancellation=CancellationInput(f);
  for(int e=0;e<ne;++e)
  {
    if(basis&&e<k){packed[e*k+(5*e)%k]=T(1);continue;}
    if((basis&&e==k)||(!basis&&e%9==4))continue;
    const int kind=basis?1:e%9;std::uint32_t state=0x9e3779b9u+std::uint32_t(17*e+f.d);
    const int exponent=std::is_same<T,double>::value?660:60;
    const long double magnitude=kind==6?std::scalbn(1.0L,-exponent):kind==7?std::scalbn(1.0L,exponent):1.0L;
    for(int j=0;j<k;++j)
    {
      state^=state<<13;state^=state>>17;state^=state<<5;
      long double value=static_cast<long double>(state&0xffffu)/32768.0L-1.0L;
      if(CancellationColumn(e,k,basis))value=cancellation[j];
      else if(kind==0||kind==8)value=.3L+std::sin(.17L*(1+j+7*e))+.2L*std::cos(.43L*(j%s-2*(j/s)));
      else if(kind==2)value=(j%2?-1.0L:1.0L)*(1+.003L*j);
      else if(kind==3)value=1;
      packed[e*k+j]=static_cast<T>(magnitude*value);
      Require(std::isfinite(packed[e*k+j]),"Center-map native input fixture overflow.");
    }
  }
  return packed;
}

struct Budget
{
  long double relative,normwise,component;
};
struct Errors
{
  long double relative=0,normwise=0,add_normwise=0;
  void Include(const Errors& other)
  {relative=std::max(relative,other.relative);normwise=std::max(normwise,other.normwise);add_normwise=std::max(add_normwise,other.add_normwise);}
};

template<typename T> Errors CheckBatch(const Factors& f,int ne,bool basis,bool tf32,const char* precision)
{
  const int d=f.d,s=3*d,k=s*s,m=d*d,n=ne*m,guard=17;
  const T sentinel=T(-1.234567e25);
  const Budget budget=std::is_same<T,double>::value?Budget{2e-11L,4e-13L,1e-12L}:
    tf32?Budget{5e-3L,1.5e-3L,3e-3L}:Budget{1e-5L,3e-7L,4e-6L};
  const auto packed=PackedInputs<T>(f,ne,basis);
  std::vector<T> response(f.response.begin(),f.response.end()),factor(ne),initial(n+guard,sentinel);
  std::vector<std::vector<long double>> expected(ne);std::vector<long double> input_norm(ne),input_max(ne);
  for(int e=0;e<ne;++e)
  {
    std::vector<long double> patch(k);
    for(int j=0;j<k;++j){patch[j]=packed[e*k+j];input_norm[e]+=patch[j]*patch[j];input_max[e]=std::max(input_max[e],std::abs(patch[j]));}
    input_norm[e]=std::sqrt(input_norm[e]);expected[e].resize(m);
    // Unit inputs are identical in FP64/FP32/TF32, so reuse their independently
    // staged oracle rather than repeating expensive long-double patch work.
    if(basis&&e<k)
      for(int row=0;row<m;++row)expected[e][row]=f.basis_reference[row+m*((5*e)%k)];
    else
    {
      const auto whole=SeparablePatch(f,patch);
      for(int row=0;row<m;++row)expected[e][row]=whole[d+row%d+s*(d+row/d)];
    }
    factor[e]=static_cast<T>(std::scalbn(1.0,(7*e)%9-4));
    long double response_norm=0;
    for(int row=0;row<m;++row)
    {
      response_norm+=expected[e][row]*expected[e][row];
      const long double initial_scale=input_max[e]>0?input_max[e]:1;
      initial[e*m+row]=CancellationColumn(e,k,basis)?T(0):
        static_cast<T>(.04L*initial_scale*std::sin(.31L*(1+row+3*e)));
    }
    if(CancellationColumn(e,k,basis))Require(std::sqrt(response_norm)<1e-5L*f.response_norm*input_norm[e],
      "Center-map cancellation fixture must have a small full-center response.");
  }
  DeviceBuffer<T> device_response(response.size()),device_packed(packed.size()),device_factor(factor.size());
  DeviceBuffer<T> device_partial(4*n+guard),device_result(initial.size());
  device_response.Upload(response);device_packed.Upload(packed);device_factor.Upload(factor);
  Errors errors;
  for(int warps:{1,4})for(int split:{1,2,4})
  {
    device_partial.Upload(std::vector<T>(4*n+guard,sentinel));device_result.Upload(initial);
    if constexpr(std::is_same<T,float>::value)
      ApplyTensorCenterResponse(ne,d,device_response.Data(),device_packed.Data(),device_partial.Data(),tf32,warps,split);
    else ApplyTensorCenterResponse(ne,d,device_response.Data(),device_packed.Data(),device_partial.Data(),warps,split);
    CUDA(cudaGetLastError());
    AddTensorCenterResponse(ne,d,device_factor.Data(),device_partial.Data(),device_result.Data(),split);
    CUDA(cudaGetLastError());
    const auto partial=device_partial.Download(),result=device_result.Download();
    for(int i=split*n;i<4*n+guard;++i)Require(partial[i]==sentinel,"Center-map partial write exceeded its active split layout.");
    for(int i=n;i<n+guard;++i)Require(result[i]==sentinel,"Center-map epilogue wrote beyond the real output tail.");
    for(int e=0;e<ne;++e)
    {
      std::ostringstream context;context<<"d="<<d<<" clipped="<<f.clipped<<" precision="<<precision<<" ne="<<ne
        <<" basis="<<basis<<" warps="<<warps<<" split_k="<<split<<" column="<<e;
      long double error=0,reference_norm=0,add_error=0,initial_norm=0;
      for(int row=0;row<m;++row)
      {
        long double actual=0;
        for(int part=0;part<split;++part)
        {
          const T value=partial[(part*ne+e)*m+row];
          Require(std::isfinite(value),"Nonfinite center-map partial: "+context.str());actual+=value;
          if(input_norm[e]==0)Require(value==T(0),"Center-map zero input produced a nonzero split partial.");
        }
        Require(std::isfinite(result[e*m+row]),"Nonfinite center-map epilogue: "+context.str());
        const long double reference=expected[e][row],delta=actual-reference;
        const long double component_bound=f.row_l1[row]*input_max[e];
        Require(std::abs(delta)<=budget.component*component_bound,"Center-map component absolute bound failed: "+context.str());
        const long double added=static_cast<long double>(initial[e*m+row])+.8L*factor[e]*reference;
        const long double added_delta=static_cast<long double>(result[e*m+row])-added;
        const long double add_bound=.8L*factor[e]*budget.component*component_bound+
          4*std::numeric_limits<T>::epsilon()*std::abs(static_cast<long double>(initial[e*m+row]));
        Require(std::abs(added_delta)<=add_bound,"Center-map scaled/additive epilogue bound failed: "+context.str());
        error+=delta*delta;reference_norm+=reference*reference;add_error+=added_delta*added_delta;
        initial_norm+=static_cast<long double>(initial[e*m+row])*initial[e*m+row];
        if(input_norm[e]==0)Require(result[e*m+row]==initial[e*m+row],"Center-map zero correction changed the existing iterate.");
      }
      error=std::sqrt(error);reference_norm=std::sqrt(reference_norm);add_error=std::sqrt(add_error);
      const long double scale=f.response_norm*input_norm[e];
      Require(error<=budget.normwise*scale+budget.relative*reference_norm,"Center-map normwise action bound failed: "+context.str());
      Require(add_error<=.8L*factor[e]*(budget.normwise*scale+budget.relative*reference_norm)+
        4*std::numeric_limits<T>::epsilon()*std::sqrt(initial_norm),"Center-map normwise epilogue bound failed: "+context.str());
      if(scale>0)
      {
        errors.normwise=std::max(errors.normwise,error/scale);
        errors.add_normwise=std::max(errors.add_normwise,add_error/(.8L*factor[e]*scale+std::sqrt(initial_norm)));
      }
      if(reference_norm>0&&!CancellationColumn(e,k,basis))
      {
        errors.relative=std::max(errors.relative,error/reference_norm);
        Require(error<=budget.relative*reference_norm,"Center-map noncancellation relative bound failed: "+context.str());
      }
    }
  }
  return errors;
}

template<typename T> void CheckPrecision(const Factors& f,bool tf32,const char* name)
{
  const int s=3*f.d;Errors errors;
  // Every input basis, three extra columns, and short/partial CTA and WMMA
  // batches. These columns represent distinct patches; none is a padded copy.
  errors.Include(CheckBatch<T>(f,s*s+3,true,tf32,name));
  for(int ne:{1,15,17,19})errors.Include(CheckBatch<T>(f,ne,false,tf32,name));
  std::cout<<"tensor_center_map_test d="<<f.d<<" clipped="<<f.clipped<<" precision="<<name
    <<" layouts=1,4-warps/1,2,4-splits basis_columns="<<s*s<<" tail_batches=1,15,17,19"
    <<" max_relative="<<static_cast<double>(errors.relative)<<" max_normwise="<<static_cast<double>(errors.normwise)
    <<" max_add_normwise="<<static_cast<double>(errors.add_normwise)<<" passed\n";
}
}

int main()
{
  try
  {
    int ordinal=0;cudaDeviceProp properties{};CUDA(cudaGetDevice(&ordinal));CUDA(cudaGetDeviceProperties(&properties,ordinal));
    if(properties.major<8){std::cout<<"tensor_center_map_test skipped: nativeSM80+ required\n";return 77;}
    for(int d:{6,4})for(bool clipped:{false,true})
    {
      auto factors=MakeFactors(d,clipped);CheckComposition(factors);
      CheckPrecision<double>(factors,false,"fp64");CheckPrecision<float>(factors,false,"fp32");CheckPrecision<float>(factors,true,"tf32");
    }
    std::cout<<"tensor_center_map_test passed independent_separable_oracle=true every_input_basis=true native_tf32=true"
      <<" cancellation=true zero=true scale=true clipped_inverse=true additive_epilogue=true\n";
    return 0;
  }
  catch(const std::exception& e){std::cerr<<"tensor_center_map_test failed: "<<e.what()<<'\n';return 1;}
}

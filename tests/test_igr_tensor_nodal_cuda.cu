#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "../src/physics/igr/IGR_spectral_nodal.cuh"
#include "../src/physics/igr/IGR_tensor_spectral.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
namespace nodal=igrlab::igr_internal::spectral_nodal_detail;
constexpr int d=6,nlocal=d*d,guard=23;
constexpr int warmups=4,repetitions=64;

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
  const T* Data()const{return data_;}
  void Upload(const std::vector<T>& values)
  {
    Require(values.size()==count_,"Nodal fixture upload size mismatch.");
    CUDA(cudaMemcpy(data_,values.data(),count_*sizeof(T),cudaMemcpyHostToDevice));
  }
  std::vector<T> Download()const
  {
    std::vector<T> values(count_);
    CUDA(cudaMemcpy(values.data(),data_,count_*sizeof(T),cudaMemcpyDeviceToHost));
    return values;
  }
private:
  std::size_t count_;
  T* data_=nullptr;
};

// This recurrence and the host oracle use long double throughout. Neither
// reads the admitted 1-D factors or reproduces the two-contraction algorithm.
long double Legendre(int order,long double coordinate)
{
  const long double x=2*coordinate-1;
  if(order==0)return 1;
  long double previous=1,current=x;
  for(int k=2;k<=order;++k)
  {
    const long double next=((2*k-1)*x*current-(k-1)*previous)/k;
    previous=current;current=next;
  }
  return std::sqrt(static_cast<long double>(2*order+1))*current;
}

using Nodes=std::vector<std::array<double,2>>;

std::vector<double> DenseNodal(const Nodes& nodes)
{
  std::vector<double> matrix(nlocal*nlocal);
  for(int column=0;column<nlocal;++column)for(int row=0;row<nlocal;++row)
    matrix[row+nlocal*column]=static_cast<double>(
      Legendre(column%d,nodes[row][0])*Legendre(column/d,nodes[row][1]));
  return matrix;
}

struct Fixture
{
  int cells_x=0,cells_y=0,nx=0,ny=0,ne=0,size=0,basis=0;
  bool reordered=false;
  Nodes nodes;
  std::vector<double> matrix;
  std::vector<int> elements,map;
  nodal::Factors factors;
};

void Admit(Fixture& f)
{
  std::string rejection;
  const bool admitted=nodal::BuildFactors(5,2,f.nodes,f.matrix,f.factors,rejection);
  Require(admitted,"Actual p5 nodal packet rejected: "+rejection);
}

Fixture MakeFixture(int basis,bool reordered)
{
  Fixture f;f.cells_x=5;f.cells_y=7;f.nx=d*f.cells_x;f.ny=d*f.cells_y;
  f.ne=f.cells_x*f.cells_y;f.size=nlocal*f.ne;f.basis=basis;f.reordered=reordered;
  mfem::L2_FECollection collection(5,2,basis);
  const auto& points=collection.FiniteElementForGeometry(mfem::Geometry::SQUARE)->GetNodes();
  Require(points.GetNPoints()==nlocal,"Expected actual p5 GL/GLL nodes.");
  f.nodes.resize(nlocal);
  for(int i=0;i<nlocal;++i)
  {
    const auto& point=points.IntPoint(reordered?(13*i+7)%nlocal:i);
    f.nodes[i]={point.x,point.y};
  }
  f.matrix=DenseNodal(f.nodes);f.elements.resize(f.ne);f.map.resize(f.size);
  int element_step=7;while(std::gcd(element_step,f.ne)!=1)++element_step;
  int dof_step=17;while(std::gcd(dof_step,f.size)!=1)++dof_step;
  for(int cell=0;cell<f.ne;++cell)f.elements[cell]=(element_step*cell+3)%f.ne;
  for(int i=0;i<f.size;++i)f.map[i]=(dof_step*i+11)%f.size;
  Admit(f);return f;
}

void CheckAdmission()
{
  auto f=MakeFixture(mfem::BasisType::GaussLegendre,false);
  auto reject=[&](int order,int dimension,const Nodes& nodes,
                  const std::vector<double>& matrix,const char* kind)
  {
    nodal::Factors factors=f.factors;std::string reason;
    Require(!nodal::BuildFactors(order,dimension,nodes,matrix,factors,reason)&&!reason.empty(),
            std::string("Malformed nodal packet must retain dense fallback: ")+kind);
    Require(factors.x==f.factors.x&&factors.y==f.factors.y&&
            factors.native_to_tensor==f.factors.native_to_tensor&&
            factors.tensor_to_native==f.factors.tensor_to_native&&
            factors.matrix_relative_error==f.factors.matrix_relative_error,
            std::string("Rejected packet changed admitted factors: ")+kind);
    std::cout<<"tensor_nodal_admission rejected="<<kind<<" reason="<<reason<<'\n';
  };
  reject(4,2,f.nodes,f.matrix,"order");reject(5,3,f.nodes,f.matrix,"dimension");
  auto nodes=f.nodes;nodes.pop_back();reject(5,2,nodes,f.matrix,"node_count");
  nodes=f.nodes;nodes[0]=nodes[1];reject(5,2,nodes,f.matrix,"duplicate_tensor_pair");
  nodes=f.nodes;nodes[0][0]+=.002;reject(5,2,nodes,f.matrix,"non_tensor_nodes");
  nodes=f.nodes;nodes[0][1]=std::numeric_limits<double>::quiet_NaN();
  reject(5,2,nodes,f.matrix,"nonfinite_node");
  nodes=f.nodes;nodes[0][0]=-1;reject(5,2,nodes,f.matrix,"outside_reference_node");
  auto matrix=f.matrix;matrix.pop_back();reject(5,2,f.nodes,matrix,"matrix_size");
  matrix=f.matrix;matrix[13+nlocal*17]+=.0001;
  reject(5,2,f.nodes,matrix,"matrix_mismatch");
  matrix=f.matrix;matrix[11]=std::numeric_limits<double>::infinity();
  reject(5,2,f.nodes,matrix,"nonfinite_matrix");
  using igrlab::igr_internal::TensorSpectralLayoutValid;
  Require(TensorSpectralLayoutValid(512,512)&&!TensorSpectralLayoutValid(3,7)&&
          !TensorSpectralLayoutValid(INT_MAX,512),
          "Existing spectral dimension admission changed.");
}

struct DeviceFixture
{
  DeviceBuffer<int> elements,map;
  DeviceBuffer<double> matrix;
  DeviceBuffer<nodal::FactorPacket> factors;
  explicit DeviceFixture(const Fixture& f):elements(f.elements.size()),map(f.map.size()),
    matrix(f.matrix.size()),factors(1)
  {
    elements.Upload(f.elements);map.Upload(f.map);matrix.Upload(f.matrix);
    factors.Upload({nodal::PackFactors(f.factors)});
  }
};

template<typename T> void Forward(bool tensor,const Fixture& f,const DeviceFixture& device,
    const double* input,const int* exponent,T* output)
{
  if(tensor)
    nodal::ToSpectralTensorKernel<T><<<f.ne,64>>>(f.nx,f.ny,device.elements.Data(),device.map.Data(),
      device.factors.Data(),input,exponent,output);
  else
    nodal::ToSpectralKernel<T><<<(f.size+255)/256,256>>>(f.nx,f.ny,device.elements.Data(),device.map.Data(),
      device.matrix.Data(),input,exponent,output);
}

template<typename T> void Reverse(bool tensor,const Fixture& f,const DeviceFixture& device,
    const T* input,const int* exponent,double scale,double* output)
{
  if(tensor)
    nodal::FromSpectralTensorKernel<T><<<f.ne,64>>>(f.nx,f.ny,device.elements.Data(),device.map.Data(),
      device.factors.Data(),input,exponent,scale,output);
  else
    nodal::FromSpectralKernel<T><<<(f.size+255)/256,256>>>(f.nx,f.ny,device.elements.Data(),device.map.Data(),
      device.matrix.Data(),input,exponent,scale,output);
}

int FieldIndex(const Fixture& f,int cell,int mode)
{return d*(cell%f.cells_x)+mode%d+f.nx*(d*(cell/f.cells_x)+mode/d);}

struct Oracle
{
  std::vector<long double> value,bound;
};

Oracle ForwardOracle(const Fixture& f,const std::vector<double>& input,int exponent)
{
  Oracle expected;expected.value.assign(f.size,0);expected.bound.assign(f.size,0);
  for(int cell=0;cell<f.ne;++cell)for(int mode=0;mode<nlocal;++mode)
  {
    const int out=FieldIndex(f,cell,mode),element=f.elements[cell];
    for(int native=0;native<nlocal;++native)
    {
      const long double term=static_cast<long double>(f.matrix[native+nlocal*mode])*
        std::scalbn(static_cast<long double>(input[f.map[nlocal*element+native]]),-exponent);
      expected.value[out]+=term;expected.bound[out]+=std::abs(term);
    }
  }
  return expected;
}

template<typename T> Oracle ReverseOracle(const Fixture& f,const std::vector<T>& input,
                                          int exponent,double scale)
{
  Oracle expected;expected.value.assign(f.size,0);expected.bound.assign(f.size,0);
  for(int cell=0;cell<f.ne;++cell)for(int native=0;native<nlocal;++native)
  {
    const int out=f.map[nlocal*f.elements[cell]+native];
    for(int mode=0;mode<nlocal;++mode)
    {
      const long double term=static_cast<long double>(f.matrix[native+nlocal*mode])*
        static_cast<long double>(input[FieldIndex(f,cell,mode)]);
      expected.value[out]+=term;expected.bound[out]+=std::abs(term);
    }
    expected.value[out]=std::scalbn(expected.value[out]/scale,exponent);
    expected.bound[out]=std::scalbn(expected.bound[out]/scale,exponent);
  }
  return expected;
}

struct Errors
{
  long double relative_l2=0,relative_linf=0,forward_error=0;
};

template<typename T> Errors CheckOracle(const std::vector<T>& actual,const Oracle& expected,
                                       T sentinel,const std::string& context)
{
  long double square=0,norm=0,maximum=0,reference_max=0;Errors error;
  for(std::size_t i=0;i<expected.value.size();++i)
  {
    Require(std::isfinite(actual[i]),context+": unexpected nonfinite result.");
    const long double delta=std::abs(static_cast<long double>(actual[i])-expected.value[i]);
    // The float allowance is one final storage rounding. Intermediate
    // contractions must still meet the FP64 cancellation/forward-error bound.
    const long double rounding=std::is_same<T,float>::value?
      .51L*std::numeric_limits<float>::epsilon()*std::abs(expected.value[i]):0;
    const long double bound=2e-13L*expected.bound[i]+rounding+
      .51L*std::numeric_limits<T>::denorm_min();
    Require(delta<=bound,context+": exceeded dense long-double forward-error budget at "+std::to_string(i));
    if(expected.bound[i]==0)Require(actual[i]==T(0),context+": zero input produced a nonzero output.");
    else error.forward_error=std::max(error.forward_error,delta/expected.bound[i]);
    square+=delta*delta;norm+=expected.value[i]*expected.value[i];
    maximum=std::max(maximum,delta);reference_max=std::max(reference_max,std::abs(expected.value[i]));
  }
  for(std::size_t i=expected.value.size();i<actual.size();++i)
    Require(actual[i]==sentinel,context+": kernel wrote beyond its real output tail.");
  error.relative_l2=norm>0?std::sqrt(square/norm):std::sqrt(square);
  error.relative_linf=reference_max>0?maximum/reference_max:maximum;
  return error;
}

template<typename T> void CheckDirections(const Fixture& f)
{
  const char* storage=std::is_same<T,double>::value?"fp64":"float";
  const T typed_sentinel=static_cast<T>(-1234567.25);
  constexpr double sentinel=-7654321.125;
  DeviceFixture device(f);
  DeviceBuffer<double> input(f.size),back(f.size+guard);
  DeviceBuffer<T> modal(f.size),forward(f.size+guard);
  DeviceBuffer<int> exponent(1);
  long double worst_forward=0,worst_reverse=0;
  for(int power:{-1060,-900,-300,0,300,900,1020})for(int kind=0;kind<8;++kind)
  {
    // The near-maximum case would overflow if scalbn preceded the division.
    const double scale=power==1020?1024.0:1.375;
    std::vector<double> native(f.size);std::vector<T> modes(f.size);
    for(int cell=0;cell<f.ne;++cell)for(int j=0;j<nlocal;++j)
    {
      long double a=.3L+std::sin(.17L*(1+j+7*cell))+.2L*std::cos(.43L*(j%d-2*(j/d)));
      long double b=std::sin(.31L*(1+3*j+cell))-.17L*std::cos(.13L*(j+5*cell));
      // Two batches collectively cover every column of each independent
      // direction. The other cases cover parity cancellation and exact zero.
      if(kind<2){a=j==(cell+kind)%nlocal?1.0L:0.0L;b=j==(cell+kind)%nlocal?-1.0L:0.0L;}
      if(kind==3){a=1;b=j==0?1.0L:0.0L;}
      if(kind==4){a=Legendre(3,f.nodes[j][0])*Legendre(2,f.nodes[j][1]);b=(j%2?-1.0L:1.0L)*(1+.003L*j);}
      if(kind==5){a=0;b=0;}
      if(kind==6){a=(j%2?-1.0L:1.0L)*(1+.003L*j);b=1;}
      if(kind==7)
      {
        a=j%3==0?std::scalbn(1.0L,-900):j%3==1?.8L:-1.2L;
        b=j%3==0?std::scalbn(1.0L,std::is_same<T,double>::value?-900:-100):j%3==1?-.7L:1.1L;
      }
      native[f.map[nlocal*f.elements[cell]+j]]=static_cast<double>(std::scalbn(a,power));
      modes[FieldIndex(f,cell,j)]=static_cast<T>(b);
    }
    input.Upload(native);modal.Upload(modes);exponent.Upload({power});
    const auto expected_forward=ForwardOracle(f,native,power);
    const auto expected_reverse=ReverseOracle(f,modes,power,scale);
    for(bool tensor:{false,true})
    {
      forward.Upload(std::vector<T>(f.size+guard,typed_sentinel));
      back.Upload(std::vector<double>(f.size+guard,sentinel));
      Forward(tensor,f,device,input.Data(),exponent.Data(),forward.Data());
      Reverse(tensor,f,device,modal.Data(),exponent.Data(),scale,back.Data());
      CUDA(cudaGetLastError());
      const std::string context=std::string(tensor?"tensor":"dense")+" "+storage+
        " power="+std::to_string(power)+" kind="+std::to_string(kind);
      const auto a=CheckOracle(forward.Download(),expected_forward,typed_sentinel,context+" forward");
      const auto b=CheckOracle(back.Download(),expected_reverse,sentinel,context+" reverse");
      worst_forward=std::max(worst_forward,a.relative_l2);worst_reverse=std::max(worst_reverse,b.relative_l2);
    }
  }
  // The global exponent sentinel is the production nonfinite-input contract.
  // Reusing every device allocation then checks recovery from the poisoned call.
  for(double bad:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()})
  {
    constexpr double scale=1.375;
    std::vector<double> native(f.size,1);native[f.size/3]=bad;
    input.Upload(native);modal.Upload(std::vector<T>(f.size,T(1)));exponent.Upload({INT_MAX});
    for(bool tensor:{false,true})
    {
      forward.Upload(std::vector<T>(f.size+guard,typed_sentinel));
      back.Upload(std::vector<double>(f.size+guard,sentinel));
      Forward(tensor,f,device,input.Data(),exponent.Data(),forward.Data());
      Reverse(tensor,f,device,modal.Data(),exponent.Data(),scale,back.Data());
      CUDA(cudaGetLastError());const auto a=forward.Download();const auto b=back.Download();
      for(int i=0;i<f.size;++i)Require(std::isnan(a[i])&&std::isnan(b[i]),"INT_MAX must poison every real output.");
      for(int i=f.size;i<f.size+guard;++i)
        Require(a[i]==typed_sentinel&&b[i]==sentinel,"Nonfinite path overwrote the output tail.");
      native.assign(f.size,1);input.Upload(native);exponent.Upload({0});
      Forward(tensor,f,device,input.Data(),exponent.Data(),forward.Data());
      Reverse(tensor,f,device,modal.Data(),exponent.Data(),scale,back.Data());CUDA(cudaGetLastError());
      CheckOracle(forward.Download(),ForwardOracle(f,native,0),typed_sentinel,"finite recovery forward");
      CheckOracle(back.Download(),ReverseOracle(f,std::vector<T>(f.size,T(1)),0,scale),sentinel,"finite recovery reverse");
      native.assign(f.size,0);input.Upload(native);modal.Upload(std::vector<T>(f.size,T(0)));
      Forward(tensor,f,device,input.Data(),exponent.Data(),forward.Data());
      Reverse(tensor,f,device,modal.Data(),exponent.Data(),scale,back.Data());CUDA(cudaGetLastError());
      CheckOracle(forward.Download(),ForwardOracle(f,native,0),typed_sentinel,"zero recovery forward");
      CheckOracle(back.Download(),ReverseOracle(f,std::vector<T>(f.size,T(0)),0,scale),sentinel,"zero recovery reverse");
      native.assign(f.size,1);modal.Upload(std::vector<T>(f.size,T(1)));
      native[f.size/3]=bad;input.Upload(native);exponent.Upload({INT_MAX});
    }
  }
  std::cout<<"tensor_nodal_case storage="<<storage<<" tf32_storage="<<std::is_same<T,float>::value
    <<" basis="<<(f.basis==mfem::BasisType::GaussLegendre?"GL":"GLL")<<" reordered_native="<<f.reordered
    <<" cells_x="<<f.cells_x<<" cells_y="<<f.cells_y<<" scalar_dofs="<<f.size
    <<" dense_tail_threads="<<(256-f.size%256)%256<<" tensor_inactive_threads=28"
    <<" matrix_relative_error="<<f.factors.matrix_relative_error
    <<" max_forward_relative_l2="<<static_cast<double>(worst_forward)
    <<" max_reverse_relative_l2="<<static_cast<double>(worst_reverse)<<" passed\n";
}

Fixture ActualSedovFixture(const std::string& path)
{
  mfem::Mesh serial(path.c_str(),1,1);
  Require(serial.Dimension()==2&&serial.GetNE()==1024&&serial.GetNV()==1089,
          "--benchmark512 requires the original stored 32x32 Sedov mesh.");
  mfem::ParMesh mesh(MPI_COMM_WORLD,serial);
  for(int level=0;level<4;++level)mesh.UniformRefinement();
  mfem::L2_FECollection collection(5,2,mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh,&collection);
  Fixture f;f.cells_x=512;f.cells_y=512;f.nx=3072;f.ny=3072;
  f.ne=mesh.GetNE();f.size=fes.GetVSize();f.basis=mfem::BasisType::GaussLegendre;
  Require(f.ne==512*512&&f.size==9437184&&fes.GetTrueVSize()==f.size,
          "Actual rs=0/rp=4 p5 scalar workload must have 9,437,184 DOFs.");
  double xmin=std::numeric_limits<double>::infinity(),xmax=-xmin,ymin=xmin,ymax=-xmin;
  for(int vertex=0;vertex<mesh.GetNV();++vertex)
  {
    const double* point=mesh.GetVertex(vertex);
    xmin=std::min(xmin,point[0]);xmax=std::max(xmax,point[0]);
    ymin=std::min(ymin,point[1]);ymax=std::max(ymax,point[1]);
  }
  Require(xmin==0&&xmax==1&&ymin==0&&ymax==1,"Original Sedov bounds must be [0,1]^2.");
  const double hx=(xmax-xmin)/512,hy=(ymax-ymin)/512;
  f.elements.assign(f.ne,-1);f.map.resize(f.size);std::vector<char> seen(f.size,0);
  const auto& points=fes.GetFE(0)->GetNodes();f.nodes.resize(nlocal);
  Require(points.GetNPoints()==nlocal,"Actual Sedov scalar nodes must be p5 GL.");
  for(int j=0;j<nlocal;++j)f.nodes[j]={points.IntPoint(j).x,points.IntPoint(j).y};
  const auto& center=mfem::Geometries.GetCenter(mfem::Geometry::SQUARE);
  mfem::Vector physical(2);mfem::Array<int> dofs;
  for(int e=0;e<f.ne;++e)
  {
    const auto& fe=*fes.GetFE(e);
    Require(fe.GetDof()==nlocal&&fe.GetOrder()==5&&fe.GetGeomType()==mfem::Geometry::SQUARE&&
            fe.GetMapType()==mfem::FiniteElement::VALUE,"Actual Sedov FE packet is not scalar p5 VALUE L2.");
    auto* transformation=fes.GetElementTransformation(e);
    Require(transformation->Order()==1,"Actual Sedov geometry is not affine.");
    transformation->SetIntPoint(&center);const auto& jacobian=transformation->Jacobian();
    Require(std::abs(jacobian(0,0)-hx)<1e-13*hx&&std::abs(jacobian(1,1)-hy)<1e-13*hy&&
            std::abs(jacobian(0,1))<1e-13*hx&&std::abs(jacobian(1,0))<1e-13*hy,
            "Actual Sedov geometry is not uniformly positive axis-aligned Cartesian.");
    transformation->Transform(center,physical);
    const int cx=static_cast<int>(std::llround((physical[0]-xmin)/hx-.5));
    const int cy=static_cast<int>(std::llround((physical[1]-ymin)/hy-.5));
    Require(cx>=0&&cx<512&&cy>=0&&cy<512&&
            std::abs((physical[0]-xmin)/hx-(cx+.5))<1e-10&&
            std::abs((physical[1]-ymin)/hy-(cy+.5))<1e-10,
            "Actual Sedov center does not identify a Cartesian cell.");
    const int cell=cx+512*cy;Require(f.elements[cell]<0,"Actual Sedov Cartesian cell repeated.");
    f.elements[cell]=e;fes.GetElementDofs(e,dofs);
    for(int j=0;j<nlocal;++j)
    {
      const auto& point=fe.GetNodes().IntPoint(j);
      Require(point.x==f.nodes[j][0]&&point.y==f.nodes[j][1]&&dofs[j]>=0&&dofs[j]<f.size&&!seen[dofs[j]],
              "Actual Sedov nodal ordering/unsigned disjoint scalar map mismatch.");
      f.map[nlocal*e+j]=dofs[j];seen[dofs[j]]=1;
    }
  }
  Require(std::find(f.elements.begin(),f.elements.end(),-1)==f.elements.end()&&
          std::find(seen.begin(),seen.end(),0)==seen.end(),"Actual Sedov maps do not cover the scalar field.");
  // Match the exact stored production matrix, including its double recurrence
  // rounding. Small correctness fixtures above retain the independent LD map.
  f.matrix.resize(nlocal*nlocal);
  for(int column=0;column<nlocal;++column)for(int row=0;row<nlocal;++row)
    f.matrix[row+nlocal*column]=nodal::LegendreValue(column%d,f.nodes[row][0])*
      nodal::LegendreValue(column/d,f.nodes[row][1]);
  Admit(f);
  std::cout<<"tensor_nodal_benchmark admission=true actual_mesh="<<path<<" serial_refinements=0 parallel_refinements=4"
    <<" order=5 basis=GL elements="<<f.ne<<" scalar_dofs="<<f.size<<" cells_x=512 cells_y=512 modal_nx=3072 modal_ny=3072"
    <<" actual_element_map=true actual_scalar_dof_map=true uniform_affine_geometry=true"
    <<" matrix_relative_error="<<f.factors.matrix_relative_error<<'\n';
  return f;
}

template<typename A,typename B> Errors Compare(const std::vector<A>& actual,const std::vector<B>& expected,int size)
{
  long double square=0,norm=0,maximum=0,reference_max=0;
  for(int i=0;i<size;++i)
  {
    Require(std::isfinite(actual[i])&&std::isfinite(expected[i]),"Nonfinite n512 benchmark output.");
    const long double delta=static_cast<long double>(actual[i])-expected[i];
    square+=delta*delta;norm+=static_cast<long double>(expected[i])*expected[i];
    maximum=std::max(maximum,std::abs(delta));reference_max=std::max(reference_max,std::abs(static_cast<long double>(expected[i])));
  }
  Errors error;error.relative_l2=norm>0?std::sqrt(square/norm):std::sqrt(square);
  error.relative_linf=reference_max>0?maximum/reference_max:maximum;return error;
}

template<typename Launch> double Time(Launch launch)
{
  for(int i=0;i<warmups;++i)launch();
  CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
  cudaEvent_t begin,end;CUDA(cudaEventCreate(&begin));CUDA(cudaEventCreate(&end));
  CUDA(cudaEventRecord(begin));
  for(int i=0;i<repetitions;++i)launch();
  CUDA(cudaEventRecord(end));CUDA(cudaEventSynchronize(end));CUDA(cudaGetLastError());
  float milliseconds=0;CUDA(cudaEventElapsedTime(&milliseconds,begin,end));
  CUDA(cudaEventDestroy(begin));CUDA(cudaEventDestroy(end));return milliseconds/repetitions;
}

template<typename T> void Benchmark(const Fixture& f,bool profile=false,
    const std::string& variant="",const std::string& direction="")
{
  const char* storage=std::is_same<T,double>::value?"fp64":"float";
  constexpr double scale=1.375;DeviceFixture device(f);DeviceBuffer<int> exponent(1);exponent.Upload({3});
  std::vector<double> native(f.size);std::vector<T> modes(f.size);
  for(int i=0;i<f.size;++i)
  {
    native[i]=.35+std::sin(.017*(1+i))+.2*std::cos(.031*i);
    modes[i]=static_cast<T>(std::sin(.023*(1+i))-.3*std::cos(.019*i));
  }
  DeviceBuffer<double> input(f.size),dense_back(f.size),tensor_back(f.size);
  DeviceBuffer<T> modal(f.size),dense_forward(f.size),tensor_forward(f.size);
  input.Upload(native);modal.Upload(modes);
  auto launch=[&](bool tensor,const std::string& which)
  {
    T* forward=tensor?tensor_forward.Data():dense_forward.Data();
    double* back=tensor?tensor_back.Data():dense_back.Data();
    if(which=="forward"||which=="combined")Forward(tensor,f,device,input.Data(),exponent.Data(),forward);
    if(which=="reverse"||which=="combined")
      Reverse(tensor,f,device,which=="combined"?forward:modal.Data(),exponent.Data(),scale,back);
  };
  if(profile)
  {
    launch(variant=="tensor",direction);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    std::cout<<"tensor_nodal_profile storage="<<storage<<" variant="<<variant<<" direction="<<direction
      <<" kernel_invocations=1 elements="<<f.ne<<" scalar_dofs="<<f.size
      <<" blocks="<<(variant=="tensor"?f.ne:(f.size+255)/256)
      <<" threads="<<(variant=="tensor"?64:256)<<'\n';return;
  }
  for(const std::string which:{"forward","reverse","combined"})
  {
    launch(false,which);launch(true,which);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    const Errors error=which=="forward"?Compare(tensor_forward.Download(),dense_forward.Download(),f.size):
      Compare(tensor_back.Download(),dense_back.Download(),f.size);
    const long double tolerance=std::is_same<T,double>::value?2e-12L:3e-7L;
    Require(error.relative_l2<tolerance&&error.relative_linf<4*tolerance,"Actual n512 tensor/dense parity failed.");
    double dense_ms=0,tensor_ms=0;
    for(bool tensor:{false,true})
    {
      const double milliseconds=Time([&]{launch(tensor,which);});
      (tensor?tensor_ms:dense_ms)=milliseconds;
      const int launches=which=="combined"?2:1;
      const std::size_t io_bytes=std::size_t(launches)*f.size*(sizeof(double)+sizeof(T));
      std::cout<<"tensor_nodal_benchmark storage="<<storage<<" tf32_storage="<<std::is_same<T,float>::value
        <<" variant="<<(tensor?"tensor":"dense")<<" direction="<<which
        <<" warmups="<<warmups<<" calls="<<repetitions<<" kernels_per_call="<<launches
        <<" cuda_event_ms_per_call="<<milliseconds<<" blocks_per_kernel="<<(tensor?f.ne:(f.size+255)/256)
        <<" threads_per_block="<<(tensor?64:256)<<" elements="<<f.ne<<" scalar_dofs="<<f.size
        <<" fp64_scalar_products_per_call="<<std::size_t(launches)*f.size*(tensor?12:36)
        <<" field_io_lower_bound_bytes="<<io_bytes
        <<" element_and_scalar_map_bytes="<<(f.elements.size()+f.map.size())*sizeof(int)
        <<" nodal_metadata_bytes="<<(tensor?sizeof(nodal::FactorPacket):f.matrix.size()*sizeof(double))
        <<" relative_l2="<<static_cast<double>(error.relative_l2)
        <<" relative_linf="<<static_cast<double>(error.relative_linf)<<'\n';
    }
    std::cout<<"tensor_nodal_benchmark_comparison storage="<<storage<<" direction="<<which
      <<" dense_ms="<<dense_ms<<" tensor_ms="<<tensor_ms<<" speedup="<<dense_ms/tensor_ms<<'\n';
  }
}
}

int main(int argc,char** argv)
{
  try
  {
    mfem::Mpi::Init(argc,argv);int ranks=0;MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    Require(ranks==1,"Nodal transform tests/benchmark require one MPI rank.");
    std::cout<<std::setprecision(12);
    if(argc==2&&std::string(argv[1])=="--admission-only")
    {CheckAdmission();std::cout<<"tensor_nodal_admission passed\n";return 0;}
    mfem::Device device("cuda");int ordinal=0;cudaDeviceProp properties{};
    CUDA(cudaGetDevice(&ordinal));CUDA(cudaGetDeviceProperties(&properties,ordinal));
    std::cout<<"tensor_nodal_device name="<<properties.name<<" sm="<<properties.major<<properties.minor<<'\n';
    if(argc==3&&std::string(argv[1])=="--benchmark512")
    {
      const auto fixture=ActualSedovFixture(argv[2]);Benchmark<double>(fixture);Benchmark<float>(fixture);return 0;
    }
    if(argc==6&&std::string(argv[1])=="--profile512")
    {
      const std::string variant=argv[3],direction=argv[4],storage=argv[5];
      Require((variant=="dense"||variant=="tensor")&&(direction=="forward"||direction=="reverse")&&
              (storage=="fp64"||storage=="float"),"Invalid --profile512 variant/direction/storage.");
      const auto fixture=ActualSedovFixture(argv[2]);
      if(storage=="fp64")Benchmark<double>(fixture,true,variant,direction);
      else Benchmark<float>(fixture,true,variant,direction);return 0;
    }
    Require(argc==1,"Usage: dg_igr_tensor_nodal_test [--admission-only | --benchmark512 actual-Sedov-mesh | "
                       "--profile512 actual-Sedov-mesh dense|tensor forward|reverse fp64|float]");
    CheckAdmission();
    for(int basis:{mfem::BasisType::GaussLegendre,mfem::BasisType::GaussLobatto})for(bool reordered:{false,true})
    {
      const auto fixture=MakeFixture(basis,reordered);
      CheckDirections<double>(fixture);CheckDirections<float>(fixture);
    }
    std::cout<<"tensor_nodal_test passed independent_dense_cuda=true independent_long_double_dense_oracle=true"
      <<" forward_separate=true reverse_separate=true every_input_basis=true actual_GL_GLL_nodes=true"
      <<" native_permutation=true element_permutation=true scalar_field_permutation=true nonsquare=true"
      <<" incomplete_dense_tail=true inactive_tensor_lanes=true zero=true cancellation=true"
      <<" power2_exponents=-1060,-900,-300,0,300,900,1020 subnormal_input=true divide_before_scalbn=true"
      <<" fp64_intermediates=true final_float_storage_cast=true"
      <<" mixed_dynamic_range=true INT_MAX_poison=true finite_recovery=true zero_recovery=true"
      <<" malformed_packet_dense_fallback=true matrix_mismatch_dense_fallback=true\n";
    return 0;
  }
  catch(const std::exception& error)
  {std::cerr<<"tensor_nodal_test failed: "<<error.what()<<'\n';return 1;}
}

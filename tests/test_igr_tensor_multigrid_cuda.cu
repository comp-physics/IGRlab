#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_tensor_multigrid.hpp"
#include "../src/physics/igr/IGR_device_krylov.hpp"
#include "../src/physics/igr/IGR_tensor_reference.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace {
using Precision=igrlab::igr_internal::TensorMultigridPrecision;
using Smoother=igrlab::igr_internal::TensorMultigridSmoother;
using OperatorMode=igrlab::igr_internal::TensorMultigridOperator;

struct Configuration
{
  Precision precision;
  Smoother smoother;
  int warps,split_k;
  OperatorMode operator_mode=OperatorMode::Sampled;
};

void Assemble(mfem::BilinearForm& form,mfem::Coefficient& reaction,mfem::Coefficient& diffusion)
{
  form.AddDomainIntegrator(new mfem::MassIntegrator(reaction));
  form.AddDomainIntegrator(new mfem::DiffusionIntegrator(diffusion));
  form.AddInteriorFaceIntegrator(new mfem::DGDiffusionIntegrator(diffusion,-1.0,36.0));
  form.Assemble();form.Finalize();
}

double Relative(const mfem::Vector& a,const mfem::Vector& b)
{
  MFEM_VERIFY(a.Size()==b.Size(),"Tensor MG fixture vector size mismatch.");
  const double* aa=a.HostRead();const double* bb=b.HostRead();long double error=0,scale=0;
  for(int i=0;i<a.Size();++i)
  {
    MFEM_VERIFY(std::isfinite(aa[i])&&std::isfinite(bb[i]),"Nonfinite tensor MG fixture result.");
    const long double delta=static_cast<long double>(aa[i])-bb[i];error+=delta*delta;scale+=static_cast<long double>(bb[i])*bb[i];
  }
  return static_cast<double>(std::sqrt(scale>0?error/scale:error));
}

const char* Name(Precision p)
{return p==Precision::FP64?"fp64":p==Precision::FP32?"fp32":"tf32";}
const char* Name(Smoother s)
{return s==Smoother::Separable?"separable":"center";}
const char* Name(OperatorMode mode)
{return mode==OperatorMode::Sampled?"sampled":"shared";}

template<typename T> void CheckSharedResidual(int d,int restricted_d,bool tf32)
{
  using igrlab::igr_internal::ApplyTensorSharedResidual;
  constexpr int nx=5,ny=4,ne=nx*ny,tail=19;
  const int n=d*d,rd=restricted_d?restricted_d:d,nr=rd*rd;
  // Twenty actual elements exercise the final partial WMMA column tile. The
  // five-neighbor map is independently scrambled, including periodic wraps.
  std::array<int,ne> permutation{};
  for(int e=0;e<ne;++e)permutation[e]=(7*e+3)%ne;
  mfem::Array<int> neighbors(5*ne);
  int* neighbor=neighbors.HostWrite();
  for(int y=0;y<ny;++y)for(int x=0;x<nx;++x)
  {
    const int e=permutation[y*nx+x];
    neighbor[5*e]=e;
    neighbor[5*e+1]=permutation[y*nx+(x+nx-1)%nx];
    neighbor[5*e+2]=permutation[y*nx+(x+1)%nx];
    neighbor[5*e+3]=permutation[((y+ny-1)%ny)*nx+x];
    neighbor[5*e+4]=permutation[((y+1)%ny)*nx+x];
  }
  mfem::Array<T> stencil(n*5*n),rhs(ne*n),state(ne*n),output(ne*nr+tail);
  T* a=stencil.HostWrite();T* x=state.HostWrite();T* b=rhs.HostWrite();
  for(int column=0;column<5*n;++column)for(int row=0;row<n;++row)
  {
    const double sign=(column+3*row)%2?-1.:1.;
    a[row+n*column]=static_cast<T>((column==row?1.+.01*row:0.)+
      sign*.004*std::cos(.17*(1+column+2*row)));
  }
  for(int e=0;e<ne;++e)for(int j=0;j<n;++j)
    x[e*n+j]=static_cast<T>(.3+std::sin(.23*(1+e*n+j))-.2*std::cos(.31*j));
  // Almost-canceling b-A*x makes a relative-to-residual check misleading.
  // Certify forward error against independently accumulated |b|+|A|*|x|.
  for(int e=0;e<ne;++e)for(int row=0;row<n;++row)
  {
    long double sum=0;
    for(int slot=0;slot<5;++slot)for(int j=0;j<n;++j)
      sum+=static_cast<long double>(a[row+n*(slot*n+j)])*x[neighbor[5*e+slot]*n+j];
    b[e*n+row]=static_cast<T>(sum+1e-5*std::sin(.41*(row+e)));
  }
  T* values=output.HostWrite();std::fill(values,values+output.Size(),static_cast<T>(12345.));
  if constexpr(std::is_same<T,double>::value)
    ApplyTensorSharedResidual(ne,d,neighbors.Read(),stencil.Read(),rhs.Read(),state.Read(),output.ReadWrite(),restricted_d);
  else
    ApplyTensorSharedResidual(ne,d,neighbors.Read(),stencil.Read(),rhs.Read(),state.Read(),output.ReadWrite(),tf32,restricted_d);
  MFEM_VERIFY(cudaGetLastError()==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess,
              "Shared tensor residual launch failed.");
  const T* result=output.HostRead();long double worst=0;
  for(int e=0;e<ne;++e)for(int row=0;row<nr;++row)
  {
    const int fine_row=row%rd+d*(row/rd);
    long double sum=0,bound=std::abs(static_cast<long double>(b[e*n+fine_row]));
    for(int slot=0;slot<5;++slot)for(int j=0;j<n;++j)
    {
      const long double product=static_cast<long double>(a[fine_row+n*(slot*n+j)])*x[neighbor[5*e+slot]*n+j];
      sum+=product;bound+=std::abs(product);
    }
    const long double expected=static_cast<long double>(b[e*n+fine_row])-sum;
    MFEM_VERIFY(std::isfinite(static_cast<double>(result[e*nr+row])),"Nonfinite shared tensor residual.");
    worst=std::max(worst,std::abs(static_cast<long double>(result[e*nr+row])-expected)/bound);
  }
  for(int i=ne*nr;i<output.Size();++i)
    MFEM_VERIFY(result[i]==static_cast<T>(12345.),"Shared residual wrote beyond real output columns.");
  const double tolerance=std::is_same<T,double>::value?2e-13:tf32?3e-3:5e-6;
  MFEM_VERIFY(worst<tolerance,"Shared tensor residual forward error failed: d="<<d
    <<" restricted_d="<<restricted_d<<" tf32="<<tf32<<" error="<<static_cast<double>(worst));
  std::cout<<"tensor_shared_residual_test d="<<d<<" restricted_d="<<restricted_d
    <<" precision="<<(std::is_same<T,double>::value?"fp64":tf32?"tf32":"fp32")
    <<" real_columns="<<ne<<" cancellation_forward_error="<<static_cast<double>(worst)<<'\n';
}

void CheckSharedResidualAdmission()
{
  using igrlab::igr_internal::TensorSharedResidualLayoutValid;
  using igrlab::igr_internal::TensorSharedStencilEntries;
  MFEM_VERIFY(TensorSharedResidualLayoutValid(512*512,6)&&
              TensorSharedResidualLayoutValid(512*512,6,4)&&
              TensorSharedStencilEntries(6)==std::size_t(5)*36*36,
              "Shared tensor operator must admit the fixed n512 p5 workload.");
  MFEM_VERIFY(!TensorSharedResidualLayoutValid(std::numeric_limits<int>::max(),6)&&
              !TensorSharedResidualLayoutValid(0,6)&&
              !TensorSharedResidualLayoutValid(20,6,3),
              "Shared tensor residual failed invalid/overflow layout admission.");
  for(int d:{6,4})for(int restricted_d:{0,d==6?4:2})
  {
    CheckSharedResidual<double>(d,restricted_d,false);
    CheckSharedResidual<float>(d,restricted_d,false);
    CheckSharedResidual<float>(d,restricted_d,true);
  }
}

void CheckHierarchy(bool variable)
{
  // The variable fixture has 20 real patches, so the production pack/map/add
  // path also sees a partial 16-column tile. Both meshes scramble element
  // order before topology reconstruction; geometric neighbors are not e+/-1.
  mfem::Mesh box=mfem::Mesh::MakeCartesian2D(variable?5:4,4,mfem::Element::QUADRILATERAL,true,1.0,1.0);
  // The periodic variable coefficients are sampled on a translated domain;
  // reference averaging must use physical geometry, including wrapped faces.
  if(variable)for(int v=0;v<box.GetNV();++v)
  {double* vertex=box.GetVertex(v);vertex[0]+=.125;vertex[1]-=.0625;}
  mfem::Array<int> element_order(box.GetNE());
  for(int e=0;e<box.GetNE();++e)element_order[e]=(7*e+3)%box.GetNE();
  box.ReorderElements(element_order,false);
  mfem::Vector dx(2),dy(2);dx=0.0;dy=0.0;dx(0)=1;dy(1)=1;
  std::vector<mfem::Vector> shifts{dx,dy};
  mfem::Mesh mesh=mfem::Mesh::MakePeriodic(box,box.CreatePeriodicVertexMapping(shifts));
  mfem::ParMesh pmesh(MPI_COMM_WORLD,mesh);
  mfem::L2_FECollection collection(5,2,mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&pmesh,&collection);
  const int n=fes.GetVSize();constexpr double pi=3.14159265358979323846;
  mfem::FunctionCoefficient reaction([=](const mfem::Vector& x)
    {return 1.0+(variable?.15*std::cos(2*pi*x(0))*.5*(1+std::sin(2*pi*x(1))):0.0);});
  mfem::FunctionCoefficient diffusion([=](const mfem::Vector& x)
    {return .01*(1.0+(variable?.12*std::sin(2*pi*x(0))*std::cos(2*pi*x(1)):0.0));});
  mfem::FunctionCoefficient changed_reaction([=](const mfem::Vector& x)
    {return 1.1+.07*std::sin(2*pi*x(1));});
  mfem::FunctionCoefficient changed_diffusion([=](const mfem::Vector& x)
    {return .012*(1.0+.08*std::cos(2*pi*x(0)));});
  mfem::BilinearForm current(&fes);Assemble(current,changed_reaction,changed_diffusion);
  mfem::Vector exact(n),rhs(n),x(n),action(n),probe(n),reference(n),result(n),scaled(n),scaled_result(n);
  mfem::Array<int> dofs;
  // Nonconstant polynomials in actual nodal order plus element variations. The
  // constructor independently checks constant and p3/p1 Galerkin actions.
  for(int e=0;e<fes.GetNE();++e)
  {
    fes.GetElementDofs(e,dofs);const auto& points=fes.GetFE(e)->GetNodes();
    for(int i=0;i<36;++i)
    {
      const auto& ip=points.IntPoint(i);
      exact(dofs[i])=.7+.1*std::sin(.73*e)+ip.x*ip.y+.3*ip.x*ip.x-.2*ip.y;
      probe(dofs[i])=.4+std::sin(.31*(1+e*36+i))+.2*std::cos(.13*i);
    }
  }
  exact.UseDevice(true);probe.UseDevice(true);current.SpMat().Mult(exact,rhs);
  std::array<mfem::Vector,3> separable_reference;
  for(const Configuration config:{
      Configuration{Precision::FP64,Smoother::Separable,1,1},
      Configuration{Precision::FP32,Smoother::Separable,1,1},
      Configuration{Precision::TF32,Smoother::Separable,1,1},
      Configuration{Precision::FP64,Smoother::Center,1,1},
      Configuration{Precision::FP32,Smoother::Center,4,2},
      Configuration{Precision::TF32,Smoother::Center,4,4},
      Configuration{Precision::FP64,Smoother::Separable,1,1,OperatorMode::Shared},
      Configuration{Precision::FP64,Smoother::Center,1,1,OperatorMode::Shared},
      Configuration{Precision::FP32,Smoother::Center,4,2,OperatorMode::Shared},
      Configuration{Precision::TF32,Smoother::Center,4,4,OperatorMode::Shared}})
  {
    const Precision precision=config.precision;
    const int precision_index=precision==Precision::FP64?0:precision==Precision::FP32?1:2;
    std::unique_ptr<mfem::Solver> preconditioner;
    {
      mfem::BilinearForm frozen(&fes);Assemble(frozen,reaction,diffusion);
      preconditioner=igrlab::igr_internal::MakeTensorMultigridPreconditioner(
        frozen.SpMat(),fes,precision,config.smoother,config.warps,config.split_k,config.operator_mode);
    } // A0 lifetime ends: the preconditioner must own its coefficient snapshot.
    preconditioner->SetOperator(current.SpMat());
    preconditioner->Mult(probe,result);
    double action_error=0;
    const bool sampled=config.operator_mode==OperatorMode::Sampled;
    const bool equivalent=sampled||!variable;
    if(sampled&&config.smoother==Smoother::Separable)separable_reference[precision_index]=result;
    if(sampled&&config.smoother==Smoother::Separable&&precision==Precision::FP64)reference=result;
    else
    {
      action_error=Relative(result,reference);
      MFEM_VERIFY(!equivalent||action_error<(precision==Precision::FP64?5e-10:precision==Precision::FP32?5e-5:2e-2),
        "Tensor MG real patch batch control mismatch: "<<Name(precision)<<" smoother="<<Name(config.smoother)
        <<" relative="<<action_error);
    }
    const double separable_error=Relative(result,separable_reference[precision_index]);
    MFEM_VERIFY(!equivalent||separable_error<(precision==Precision::FP64?5e-10:precision==Precision::FP32?8e-5:3e-2),
      "Tensor MG center/separable V-cycle mismatch: "<<Name(precision)<<" relative="<<separable_error);
    // Dynamic power-of-two input scaling must preserve tiny and huge useful
    // residuals. Long-double comparison intentionally has no 1e-300 norm floor.
    for(double scale:{1e-200,-1e-200,1e200})
    {
      const double* p=probe.HostRead();double* s=scaled.HostWrite();
      for(int i=0;i<n;++i)s[i]=scale*p[i];scaled.UseDevice(true);
      preconditioner->Mult(scaled,scaled_result);
      double* v=scaled_result.HostReadWrite();for(int i=0;i<n;++i)v[i]/=scale;
      const double error=Relative(scaled_result,result);
      MFEM_VERIFY(error<(precision==Precision::FP64?5e-10:2e-3),
        "Tensor MG scaled residual lost homogeneity: "<<Name(precision)<<" smoother="<<Name(config.smoother)
        <<" scale="<<scale<<" relative="<<error);
    }
    scaled=0.0;preconditioner->Mult(scaled,scaled_result);
    MFEM_VERIFY(scaled_result.Norml2()==0.0,"Tensor MG zero RHS must produce zero correction.");
    auto solver=igrlab::igr_internal::MakeDeviceFGMRESSolver(MPI_COMM_WORLD,50);
    solver->SetOperator(current.SpMat());solver->SetPreconditioner(*preconditioner);
    solver->SetRelTol(1e-10);solver->SetAbsTol(0);solver->SetMaxIter(400);solver->SetPrintLevel(0);
    for(int repeat=0;repeat<2;++repeat)
    {
      if(repeat==0){x=0.0;solver->iterative_mode=false;}
      else
      {
        double* values=exact.HostReadWrite();for(int i=0;i<n;++i)values[i]+=.07*std::cos(.19*i);
        current.SpMat().Mult(exact,rhs);solver->iterative_mode=true;
      }
      solver->Mult(rhs,x);current.SpMat().Mult(x,action);
      const double residual=Relative(action,rhs),error=Relative(x,exact);
      std::cout<<"tensor_mg_current_A_test variable="<<variable<<" precision="<<Name(precision)
        <<" translated_periodic_domain="<<variable
        <<" operator_mode="<<Name(config.operator_mode)<<" sampled_action_equivalence_required="<<equivalent
        <<" smoother="<<Name(config.smoother)<<" center_warps="<<config.warps<<" center_split_k="<<config.split_k
        <<" elements="<<fes.GetNE()
        <<" repeat="<<repeat<<" iterations="<<solver->GetNumIterations()<<" true_residual="<<residual
        <<" solution_error="<<error<<" patch_action_control_error="<<action_error
        <<" separable_action_control_error="<<separable_error<<'\n';
      MFEM_VERIFY(solver->GetConverged()&&residual<=1e-10&&error<1e-7,
        "Tensor MG original-current-A convergence failed: "<<Name(precision)<<" residual="<<residual<<" error="<<error);
    }
  }
}
}

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cuda");
  int ordinal=0;cudaDeviceProp properties{};
  MFEM_VERIFY(cudaGetDevice(&ordinal)==cudaSuccess&&cudaGetDeviceProperties(&properties,ordinal)==cudaSuccess,"CUDA property query failed.");
  if(properties.major<8){std::cout<<"tensor_mg_test skipped: nativeSM80+ required\n";return 77;}
  CheckSharedResidualAdmission();CheckHierarchy(false);CheckHierarchy(true);
  std::cout<<"tensor_mg_test passed multiplicative_vcycle=true hierarchy=5,3,1"
    <<" original_A_certificate=true changed_coefficients=true real_patch_columns=true"
    <<" precision_controls=true center_controls=true center_layouts=1x1,4x2,4x4"
    <<" reordered_topology=true patch_tail=true scaled_rhs=true snapshot_lifetime=true"
    <<" shared_constant_equivalence=true shared_variable_current_A=true shared_residual_tails=true"
    <<" shared_large_layout=true cancellation_forward_error=true\n";
  return 0;
}

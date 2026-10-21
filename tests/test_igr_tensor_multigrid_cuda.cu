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
#include <string>
#include <type_traits>
#include <vector>

namespace {
using Precision=igrlab::igr_internal::TensorMultigridPrecision;
using Smoother=igrlab::igr_internal::TensorMultigridSmoother;
using OperatorMode=igrlab::igr_internal::TensorMultigridOperator;
using ResidualMode=igrlab::igr_internal::TensorMultigridResidual;

struct Configuration
{
  Precision precision;
  Smoother smoother;
  int warps,split_k;
  OperatorMode operator_mode=OperatorMode::Sampled;
  int owned_tile=1;
  ResidualMode residual_mode=ResidualMode::Auto;
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
{return p==Precision::FP64?"fp64":p==Precision::FP32?"fp32":p==Precision::TF32?"tf32":"tf32x3";}
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

void CheckHierarchy(bool variable,bool owned=false)
{
  // The variable fixture has 20 real patches, so the production pack/map/add
  // path also sees a partial 16-column tile. Both meshes scramble element
  // order before topology reconstruction; geometric neighbors are not e+/-1.
  // The larger smoother sees six genuine 4x4 owned blocks on an 8x12 mesh,
  // exercising partial 16-column warp tiles and partial 64-column CTAs.
  mfem::Mesh box=mfem::Mesh::MakeCartesian2D(owned?8:variable?5:4,owned?12:4,
    mfem::Element::QUADRILATERAL,true,1.0,1.0);
  // The periodic variable coefficients are sampled on a translated domain;
  // reference averaging must use physical geometry, including wrapped faces.
  if(variable||owned)for(int v=0;v<box.GetNV();++v)
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
  std::vector<Configuration> configurations={
      Configuration{Precision::FP64,Smoother::Separable,1,1},
      Configuration{Precision::FP32,Smoother::Separable,1,1},
      Configuration{Precision::TF32,Smoother::Separable,1,1},
      Configuration{Precision::FP64,Smoother::Center,1,1},
      Configuration{Precision::FP32,Smoother::Center,4,2},
      Configuration{Precision::TF32,Smoother::Center,4,4},
      Configuration{Precision::FP64,Smoother::Separable,1,1,OperatorMode::Shared},
      Configuration{Precision::FP64,Smoother::Center,1,1,OperatorMode::Shared},
      Configuration{Precision::FP32,Smoother::Center,4,2,OperatorMode::Shared},
      Configuration{Precision::TF32,Smoother::Center,4,4,OperatorMode::Shared},
      Configuration{Precision::TF32,Smoother::Center,1,1,OperatorMode::Shared,1,ResidualMode::FP32}};
  if(owned)configurations={
      Configuration{Precision::FP64,Smoother::Center,1,1,OperatorMode::Shared,4},
      Configuration{Precision::FP32,Smoother::Center,1,1,OperatorMode::Shared,4},
      Configuration{Precision::TF32,Smoother::Center,1,1,OperatorMode::Shared,4},
      Configuration{Precision::TF32,Smoother::Center,1,1,OperatorMode::Shared,4,ResidualMode::FP32}};
  for(const Configuration config:configurations)
  {
    const Precision precision=config.precision;
    const int precision_index=precision==Precision::FP64?0:precision==Precision::FP32?1:2;
    std::unique_ptr<mfem::Solver> preconditioner;
    {
      mfem::BilinearForm frozen(&fes);Assemble(frozen,reaction,diffusion);
      preconditioner=igrlab::igr_internal::MakeTensorMultigridPreconditioner(
        frozen.SpMat(),fes,precision,config.smoother,config.warps,config.split_k,config.operator_mode,
        config.owned_tile,config.residual_mode);
    } // A0 lifetime ends: the preconditioner must own its coefficient snapshot.
    preconditioner->SetOperator(current.SpMat());
    preconditioner->Mult(probe,result);
    double action_error=0;
    const bool sampled=config.operator_mode==OperatorMode::Sampled;
    const bool equivalent=!owned&&(sampled||!variable);
    if(!owned&&sampled&&config.smoother==Smoother::Separable)separable_reference[precision_index]=result;
    if((!owned&&sampled&&config.smoother==Smoother::Separable&&precision==Precision::FP64)||
       (owned&&precision==Precision::FP64))reference=result;
    else
    {
      action_error=Relative(result,reference);
      MFEM_VERIFY(!equivalent||action_error<(precision==Precision::FP64?5e-10:precision==Precision::FP32?5e-5:2e-2),
        "Tensor MG real patch batch control mismatch: "<<Name(precision)<<" smoother="<<Name(config.smoother)
        <<" relative="<<action_error);
    }
    // Changing the owned region changes the mathematical preconditioner. Its
    // tests use a separate staged map oracle and current-A convergence, with
    // no equivalence gate against the old one-element center smoother.
    const double separable_error=owned?0.0:Relative(result,separable_reference[precision_index]);
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
        <<" translated_periodic_domain="<<(variable||owned)
        <<" operator_mode="<<Name(config.operator_mode)<<" sampled_action_equivalence_required="<<equivalent
        <<" owned_tile="<<config.owned_tile<<" owned_real_columns="<<(owned?fes.GetNE()/16:0)
        <<" residual_mode="<<(config.residual_mode==ResidualMode::Auto?"auto":"fp32")
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

void CheckSpectral(bool variable,bool natural=false)
{
  // Both rectangles expose x/y swaps and incorrect GEMM leading dimensions.
  // Unequal physical widths also distinguish x/y face scaling on open edges.
  // Element indices carry no Cartesian ordering.
  constexpr int d=6;const int nx=natural?5:8,ny=natural?7:12;
  const double width_x=natural?1.25:1.0,width_y=natural?.75:1.0;
  constexpr double pi=3.14159265358979323846;
  mfem::Mesh box=mfem::Mesh::MakeCartesian2D(nx,ny,mfem::Element::QUADRILATERAL,true,width_x,width_y);
  for(int v=0;v<box.GetNV();++v)
  {double* vertex=box.GetVertex(v);vertex[0]+=.125;vertex[1]-=.0625;}
  mfem::Array<int> ordering(box.GetNE());
  for(int e=0;e<box.GetNE();++e)ordering[e]=(37*e+11)%box.GetNE();
  box.ReorderElements(ordering,false);
  mfem::Vector dx(2),dy(2);dx=0.0;dy=0.0;dx(0)=1;dy(1)=1;
  const std::vector<mfem::Vector> shifts{dx,dy};
  mfem::Mesh mesh=natural?mfem::Mesh(box):
    mfem::Mesh::MakePeriodic(box,box.CreatePeriodicVertexMapping(shifts));
  mfem::ParMesh pmesh(MPI_COMM_WORLD,mesh);
  mfem::L2_FECollection collection(5,2,mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&pmesh,&collection);
  const int n=fes.GetVSize();
  mfem::FunctionCoefficient reaction([=](const mfem::Vector& x)
    {return 1.0+(variable?.15*std::cos(2*pi*x(0))*.5*(1+std::sin(2*pi*x(1))):0.0);});
  mfem::FunctionCoefficient diffusion([=](const mfem::Vector& x)
    {return .01*(1.0+(variable?.12*std::sin(2*pi*x(0))*std::cos(2*pi*x(1)):0.0));});
  mfem::FunctionCoefficient changed_reaction([=](const mfem::Vector& x)
    {return 1.1+.07*std::sin(2*pi*x(1));});
  mfem::FunctionCoefficient changed_diffusion([=](const mfem::Vector& x)
    {return .012*(1.0+.08*std::cos(2*pi*x(0)));});
  // The oracle is separately assembled and outlives the source operator passed
  // to the factory. A retained pointer into that temporary source is invalid.
  mfem::BilinearForm frozen_oracle(&fes),current(&fes),mass_oracle(&fes);
  Assemble(frozen_oracle,reaction,diffusion);Assemble(current,changed_reaction,changed_diffusion);
  if(natural)
  {
    // With no boundary integrator the diffusion part annihilates a physical
    // constant. This load independently tests the screened natural mode.
    mass_oracle.AddDomainIntegrator(new mfem::MassIntegrator(reaction));
    mass_oracle.Assemble();mass_oracle.Finalize();
  }
  mfem::Vector exact(n),probe(n),rhs(n),x(n),result(n),action(n),scaled(n),scaled_result(n);
  std::array<int,4> corner_dofs{};
  std::array<double,4> corner_distance;corner_distance.fill(std::numeric_limits<double>::infinity());
  mfem::Array<int> dofs;
  for(int e=0;e<fes.GetNE();++e)
  {
    fes.GetElementDofs(e,dofs);const auto& nodes=fes.GetFE(e)->GetNodes();
    auto* tr=fes.GetElementTransformation(e);mfem::Vector point(2);
    for(int i=0;i<d*d;++i)
    {
      const auto& node=nodes.IntPoint(i);tr->Transform(node,point);
      exact(dofs[i])=.7+.13*std::sin(2*pi*point(0))-.09*std::cos(4*pi*point(1))+
        .06*node.x*node.y+.02*std::sin(.53*e);
      probe(dofs[i])=.31+std::sin(.19*(1+e*d*d+i))+.17*std::cos(.37*(2*i+e));
      if(natural)for(int corner=0;corner<4;++corner)
      {
        const double dx=point(0)-(.125+(corner%2?width_x:0.0));
        const double dy=point(1)-(-.0625+(corner/2?width_y:0.0));
        const double distance=dx*dx+dy*dy;
        if(distance<corner_distance[corner])
        {corner_distance[corner]=distance;corner_dofs[corner]=dofs[i];}
      }
    }
  }
  exact.UseDevice(true);probe.UseDevice(true);
  double condition_bound=0,inverse_limit=0;
  if(!variable)
  {
    // For this SIPG operator diffusion is positive semidefinite, so the
    // minimum collocated reaction mass bounds lambda_min(A0) from below.
    // Symmetry gives lambda_max(A0)<=max absolute row sum. Keep the resulting
    // FP64 inverse gate bounded even if the fixture unexpectedly changes.
    double minimum_mass=std::numeric_limits<double>::infinity(),row_norm=0;
    const auto& rule=mfem::IntRules.Get(mfem::Geometry::SQUARE,10);
    for(int e=0;e<fes.GetNE();++e)for(int q=0;q<rule.GetNPoints();++q)
    {
      auto* tr=fes.GetElementTransformation(e);const auto& point=rule.IntPoint(q);
      tr->SetIntPoint(&point);minimum_mass=std::min(minimum_mass,point.weight*tr->Weight());
    }
    const int* rows=frozen_oracle.SpMat().HostReadI();
    const double* values=frozen_oracle.SpMat().HostReadData();
    for(int i=0;i<n;++i)
    {
      double sum=0;for(int j=rows[i];j<rows[i+1];++j)sum+=std::abs(values[j]);
      row_norm=std::max(row_norm,sum);
    }
    MFEM_VERIFY(std::isfinite(minimum_mass)&&minimum_mass>0&&std::isfinite(row_norm)&&row_norm>0,
                "Invalid spectral inverse condition bound.");
    condition_bound=row_norm/minimum_mass;
    inverse_limit=std::min(1e-7,std::max(5e-10,128*std::numeric_limits<double>::epsilon()*condition_bound));
  }
  const std::vector<Precision> precisions=natural?
    std::vector<Precision>{Precision::FP64,Precision::TF32}:
    std::vector<Precision>{Precision::FP64,Precision::FP32,Precision::TF32,Precision::TF32x3};
  for(Precision precision:precisions)
  {
    std::unique_ptr<mfem::Solver> inverse;
    {
      mfem::BilinearForm frozen(&fes);Assemble(frozen,reaction,diffusion);
      inverse=igrlab::igr_internal::MakeTensorMultigridPreconditioner(
        frozen.SpMat(),fes,precision,Smoother::Separable,1,1,OperatorMode::Spectral);
    }
    inverse->SetOperator(current.SpMat());
    if(!variable)
    {
      for(int kind=0;kind<(natural?4:3);++kind)
      {
        if(kind==0)rhs=1.0;
        else if(kind==1)
        {
          rhs=0.0;double* values=rhs.HostReadWrite();
          if(natural)
          {
            // Geometric corners survive element reordering. Unequal signed
            // loads excite both ends of both axes without corner symmetry.
            constexpr double loads[4]={1.0,-.31,.23,-.17};
            for(int corner=0;corner<4;++corner)values[corner_dofs[corner]]=loads[corner];
          }
          else values[n/3+5]=1.0;
        }
        else if(kind==2)rhs=probe;
        else
        {
          scaled=1.0;scaled.UseDevice(true);mass_oracle.SpMat().Mult(scaled,rhs);
          frozen_oracle.SpMat().Mult(scaled,action);
          MFEM_VERIFY(Relative(action,rhs)<=inverse_limit,
            "Natural SIPG diffusion does not annihilate a physical constant.");
        }
        rhs.UseDevice(true);inverse->Mult(rhs,result);
        frozen_oracle.SpMat().Mult(result,action);
        const double residual=Relative(action,rhs);
        const double constant_error=natural&&kind==3?Relative(result,scaled):0.0;
        std::cout<<"tensor_spectral_inverse_test precision="<<Name(precision)<<" kind="<<kind
          <<" boundary="<<(natural?"natural":"periodic")
          <<" modal_nx="<<d*nx<<" modal_ny="<<d*ny<<" original_A0_residual="<<residual
          <<" physical_constant_error="<<constant_error
          <<" condition_bound="<<condition_bound<<" fp64_limit="<<inverse_limit
          <<" exact_inverse_required="<<(precision==Precision::FP64)<<'\n';
        MFEM_VERIFY(precision!=Precision::FP64||
          (residual<=inverse_limit&&constant_error<=inverse_limit),
          "FP64 spectral inverse failed the independent original constant-A0 gate: boundary="
          <<(natural?"natural":"periodic")<<" residual="<<residual<<" constant_error="<<constant_error);
      }
    }
    inverse->Mult(probe,result);
    for(double scale:{1e-200,-1e-200,1e200})
    {
      const double* p=probe.HostRead();double* values=scaled.HostWrite();
      for(int i=0;i<n;++i)values[i]=scale*p[i];scaled.UseDevice(true);
      inverse->Mult(scaled,scaled_result);
      double* values_out=scaled_result.HostReadWrite();for(int i=0;i<n;++i)values_out[i]/=scale;
      const double relative=Relative(scaled_result,result);
      MFEM_VERIFY(relative<(precision==Precision::FP64?5e-10:2e-3),
        "Spectral inverse lost scaled-input homogeneity: "<<Name(precision)<<" scale="<<scale
        <<" relative="<<relative);
    }
    scaled=0.0;inverse->Mult(scaled,scaled_result);
    MFEM_VERIFY(scaled_result.Norml2()==0.0,"Spectral zero RHS produced a nonzero correction.");
    for(double bad:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()})
    {
      scaled=probe;scaled.HostReadWrite()[n/4+7]=bad;scaled.UseDevice(true);
      inverse->Mult(scaled,scaled_result);
      const double* values=scaled_result.HostRead();
      MFEM_VERIFY(std::all_of(values,values+n,[](double value){return std::isnan(value);}),
        "Spectral nonfinite input did not poison the complete correction for original-A rejection.");
    }
    inverse->Mult(probe,scaled_result);
    MFEM_VERIFY(Relative(scaled_result,result)<(precision==Precision::FP64?5e-10:2e-3),
                "Spectral inverse retained nonfinite workspace state after a new finite input.");
    current.SpMat().Mult(exact,rhs);
    auto solver=igrlab::igr_internal::MakeDeviceFGMRESSolver(MPI_COMM_WORLD,50);
    solver->SetOperator(current.SpMat());solver->SetPreconditioner(*inverse);
    solver->SetRelTol(1e-10);solver->SetAbsTol(0);solver->SetMaxIter(400);solver->SetPrintLevel(0);
    for(int repeat=0;repeat<2;++repeat)
    {
      if(repeat==0){x=0.0;solver->iterative_mode=false;}
      else
      {
        double* values=exact.HostReadWrite();for(int i=0;i<n;++i)values[i]+=.043*std::cos(.23*i);
        current.SpMat().Mult(exact,rhs);solver->iterative_mode=true;
      }
      solver->Mult(rhs,x);current.SpMat().Mult(x,action);
      const double residual=Relative(action,rhs),error=Relative(x,exact);
      std::cout<<"tensor_spectral_current_A_test variable="<<variable<<" precision="<<Name(precision)
        <<" boundary="<<(natural?"natural":"periodic")
        <<" modal_nx="<<d*nx<<" modal_ny="<<d*ny<<" repeat="<<repeat
        <<" iterations="<<solver->GetNumIterations()<<" true_residual="<<residual
        <<" solution_error="<<error<<" old_mg_equivalence_required=false\n";
      MFEM_VERIFY(solver->GetConverged()&&residual<=1e-10&&error<1e-7,
        "Spectral inverse did not converge against current original FP64 A: "<<Name(precision)
        <<" residual="<<residual<<" error="<<error);
    }
  }
}

// GLL sigma uses the consistent (noncollocated) reaction mass. Keep this
// separate from the existing GL controls and their diagonal-mass condition
// estimate: the minimum GL quadrature weight is not a GLL mass eigenvalue.
void CheckSpectralGLL(bool natural)
{
  constexpr int d=6,nx=5,ny=7;
  constexpr double width_x=1.25,width_y=.875,pi=3.14159265358979323846;
  // Binary-exact spacings (.25,.125), unequal axes and reordered elements
  // exercise the production Cartesian proof without relying on element IDs.
  mfem::Mesh box=mfem::Mesh::MakeCartesian2D(nx,ny,mfem::Element::QUADRILATERAL,
                                           true,width_x,width_y);
  for(int v=0;v<box.GetNV();++v)
  {double* vertex=box.GetVertex(v);vertex[0]+=.125;vertex[1]-=.0625;}
  mfem::Array<int> ordering(box.GetNE());
  for(int e=0;e<box.GetNE();++e)ordering[e]=(12*e+3)%box.GetNE();
  box.ReorderElements(ordering,false);
  const std::vector<mfem::Vector> shifts{mfem::Vector({width_x,0.0}),
                                         mfem::Vector({0.0,width_y})};
  mfem::Mesh mesh=natural?mfem::Mesh(box):
    mfem::Mesh::MakePeriodic(box,box.CreatePeriodicVertexMapping(shifts));
  mfem::ParMesh pmesh(MPI_COMM_WORLD,mesh);
  mfem::L2_FECollection collection(5,2,mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace fes(&pmesh,&collection);
  const auto* tensor=dynamic_cast<const mfem::TensorBasisElement*>(fes.GetFE(0));
  MFEM_VERIFY(tensor&&tensor->GetBasisType()==mfem::BasisType::GaussLobatto&&
                fes.GetFE(0)->GetDof()==d*d,
              "GLL spectral fixture must use the actual p5 Gauss-Lobatto element.");
  const int n=fes.GetVSize();
  mfem::ConstantCoefficient reaction(1.0),diffusion(.01);
  mfem::BilinearForm frozen_oracle(&fes),mass_oracle(&fes);
  Assemble(frozen_oracle,reaction,diffusion);
  mass_oracle.AddDomainIntegrator(new mfem::MassIntegrator(reaction));
  mass_oracle.Assemble();mass_oracle.Finalize();
  bool consistent_mass_has_offdiagonal=false;
  const int* mass_rows=mass_oracle.SpMat().HostReadI();
  const int* mass_columns=mass_oracle.SpMat().HostReadJ();
  const double* mass_values=mass_oracle.SpMat().HostReadData();
  for(int row=0;row<n;++row)for(int entry=mass_rows[row];entry<mass_rows[row+1];++entry)
    if(mass_columns[entry]!=row&&mass_values[entry]!=0.0)consistent_mass_has_offdiagonal=true;
  MFEM_VERIFY(consistent_mass_has_offdiagonal,
              "GLL spectral oracle must retain the noncollocated consistent mass.");

  std::array<std::unique_ptr<mfem::FunctionCoefficient>,2> current_reaction,current_diffusion;
  std::array<std::unique_ptr<mfem::BilinearForm>,2> current;
  for(int phase=0;phase<2;++phase)
  {
    current_reaction[phase]=std::make_unique<mfem::FunctionCoefficient>(
      [phase](const mfem::Vector& point)
      {return 1.1+.05*phase+.07*std::sin(2*pi*point(1));});
    current_diffusion[phase]=std::make_unique<mfem::FunctionCoefficient>(
      [phase](const mfem::Vector& point)
      {return .012*(1.0+.04*phase+.08*std::cos(2*pi*point(0)));});
    current[phase]=std::make_unique<mfem::BilinearForm>(&fes);
    Assemble(*current[phase],*current_reaction[phase],*current_diffusion[phase]);
  }

  mfem::Vector exact(n),probe(n),rhs(n),x(n),result(n),action(n),ones(n),mass_load(n);
  mfem::Vector retained(n),repeated(n),first_action(n),second_action(n);
  for(auto* vector:{&exact,&probe,&rhs,&x,&result,&action,&ones,&mass_load,
                   &retained,&repeated,&first_action,&second_action})vector->UseDevice(true);
  double* probe_values=probe.HostWrite();
  for(int i=0;i<n;++i)
    probe_values[i]=.31+std::sin(.19*(1+i))+.17*std::cos(.37*(2*i+3));
  // Independently witness that neither current matrix equals the retained A0,
  // and that the second coefficient stage changes the operator again.
  frozen_oracle.SpMat().Mult(probe,first_action);
  current[0]->SpMat().Mult(probe,second_action);
  MFEM_VERIFY(Relative(second_action,first_action)>1e-3,
              "GLL spectral current-A fixture did not change frozen coefficients.");
  first_action=second_action;
  current[1]->SpMat().Mult(probe,second_action);
  MFEM_VERIFY(Relative(second_action,first_action)>1e-3,
              "GLL spectral second coefficient stage did not change current A.");

  ones=1.0;mass_oracle.SpMat().Mult(ones,mass_load);
  frozen_oracle.SpMat().Mult(ones,action);
  // This fixed 5e-10 gate is the strict floor of the old GL inverse gate.
  // It measures the actual assembled consistent-mass residual directly; no
  // collocated-mass lower bound or condition-number inflation is used.
  constexpr double inverse_limit=5e-10;
  MFEM_VERIFY(Relative(action,mass_load)<=inverse_limit,
              "GLL natural/periodic diffusion failed its physical constant mode.");
  mfem::Array<int> endpoint_dofs;fes.GetElementDofs(0,endpoint_dofs);

  for(Precision precision:{Precision::FP64,Precision::TF32})
  {
    std::unique_ptr<mfem::Solver> inverse;
    {
      // A separate oracle remains above after the sampled source is destroyed.
      mfem::BilinearForm frozen(&fes);Assemble(frozen,reaction,diffusion);
      inverse=igrlab::igr_internal::MakeTensorMultigridPreconditioner(
        frozen.SpMat(),fes,precision,Smoother::Separable,1,1,OperatorMode::Spectral);
    }
    if(precision==Precision::FP64)
    {
      for(int kind=0;kind<4;++kind)
      {
        if(kind==0)rhs=1.0;
        else if(kind==1)
        {
          rhs=0.0;double* values=rhs.HostReadWrite();
          values[endpoint_dofs[0]]=1.0;values[endpoint_dofs[d*d-1]]=-.31;
        }
        else if(kind==2)rhs=probe;
        else rhs=mass_load;
        inverse->Mult(rhs,result);frozen_oracle.SpMat().Mult(result,action);
        const double residual=Relative(action,rhs);
        const double constant_error=kind==3?Relative(result,ones):0.0;
        MFEM_VERIFY(residual<=inverse_limit&&constant_error<=inverse_limit,
          "GLL FP64 spectral inverse failed independently assembled consistent A0: "
          <<"boundary="<<(natural?"natural":"periodic")<<" kind="<<kind
          <<" residual="<<residual<<" constant_error="<<constant_error);
        std::cout<<"tensor_spectral_gll_inverse_test boundary="<<(natural?"natural":"periodic")
          <<" kind="<<kind<<" precision=fp64 original_A0_residual="<<residual
          <<" physical_constant_error="<<constant_error<<" fixed_limit="<<inverse_limit
          <<" mass=consistent noncollocated=true\n";
      }
    }
    inverse->Mult(probe,retained);
    auto solver=igrlab::igr_internal::MakeDeviceFGMRESSolver(MPI_COMM_WORLD,50);
    solver->SetRelTol(1e-11);solver->SetAbsTol(0.0);
    solver->SetMaxIter(400);solver->SetPrintLevel(0);
    for(int phase=0;phase<2;++phase)
    {
      inverse->SetOperator(current[phase]->SpMat());
      inverse->Mult(probe,repeated);
      MFEM_VERIFY(Relative(repeated,retained)<1e-13,
                  "GLL spectral SetOperator changed its retained frozen reference.");
      solver->SetOperator(current[phase]->SpMat());
      // DeviceKrylovBase::SetOperator clears a previous preconditioner.
      solver->SetPreconditioner(*inverse);
      for(int warm=0;warm<2;++warm)
      {
        double* values=exact.HostWrite();
        for(int i=0;i<n;++i)
          values[i]=.7+.13*std::sin(.17*(i+1))-.09*std::cos(.23*i)+
                    .04*phase*std::sin(.11*i)+.043*warm*std::cos(.29*i);
        current[phase]->SpMat().Mult(exact,rhs);
        if(!warm){x=0.0;solver->iterative_mode=false;}
        else solver->iterative_mode=true;
        solver->Mult(rhs,x);current[phase]->SpMat().Mult(x,action);
        const double residual=Relative(action,rhs),error=Relative(x,exact);
        MFEM_VERIFY(solver->GetConverged()&&residual<=1e-10&&error<1e-7,
          "GLL spectral preconditioner failed independent current FP64 A: "
          <<"boundary="<<(natural?"natural":"periodic")<<" precision="<<Name(precision)
          <<" phase="<<phase<<" warm="<<warm<<" residual="<<residual<<" error="<<error);
        std::cout<<"tensor_spectral_gll_current_A_test boundary="<<(natural?"natural":"periodic")
          <<" precision="<<Name(precision)<<" coefficient_phase="<<phase<<" warm="<<warm
          <<" modal_nx="<<d*nx<<" modal_ny="<<d*ny<<" iterations="<<solver->GetNumIterations()
          <<" true_residual="<<residual<<" solution_error="<<error
          <<" frozen_reference_retained=true original_current_A=fp64\n";
      }
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
  const bool spectral_gll_only=argc==2&&std::string(argv[1])=="--spectral-gll-only";
  MFEM_VERIFY(argc==1||spectral_gll_only,
              "Usage: dg_igr_tensor_multigrid_test [--spectral-gll-only]");
  if(spectral_gll_only)
  {
    CheckSpectralGLL(false);CheckSpectralGLL(true);
    std::cout<<"tensor_spectral_gll_test passed actual_gll_nodes=true consistent_mass=true"
      <<" periodic=true natural=true fp64_inverse=true tf32_current_A=true"
      <<" changed_coefficients=true cold_warm=true reordered_elements=true snapshot_lifetime=true\n";
    return 0;
  }
  CheckSharedResidualAdmission();CheckHierarchy(false);CheckHierarchy(true);
  CheckHierarchy(false,true);CheckHierarchy(true,true);
  CheckSpectral(false);CheckSpectral(true);CheckSpectral(false,true);
  std::cout<<"tensor_mg_test passed multiplicative_vcycle=true hierarchy=5,3,1"
    <<" original_A_certificate=true changed_coefficients=true real_patch_columns=true"
    <<" precision_controls=true center_controls=true center_layouts=1x1,4x2,4x4"
    <<" reordered_topology=true patch_tail=true scaled_rhs=true snapshot_lifetime=true"
    <<" shared_constant_equivalence=true shared_variable_current_A=true shared_residual_tails=true"
    <<" shared_large_layout=true cancellation_forward_error=true owned_tile_4x4=true"
    <<" owned_changed_current_A=true owned_tail_columns=6 old_smoother_equivalence=false"
    <<" hybrid_residual_tile1=true hybrid_residual_tile4=true spectral_rectangular_48x72=true"
    <<" spectral_constant_inverse=true spectral_variable_current_A=true spectral_lifetime=true"
    <<" spectral_tf32x3=true spectral_nonfinite_poison=true spectral_finite_recovery=true"
    <<" spectral_natural_rectangular_30x42=true spectral_natural_corners=true"
    <<" spectral_natural_constant_mode=true spectral_natural_current_A_fp64_tf32=true\n";
  return 0;
}

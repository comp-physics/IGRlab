#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_tensor_multigrid.hpp"
#include "../src/physics/igr/IGR_device_krylov.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {
using Precision=igrlab::igr_internal::TensorMultigridPrecision;

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

void CheckHierarchy(bool variable)
{
  mfem::Mesh box=mfem::Mesh::MakeCartesian2D(4,4,mfem::Element::QUADRILATERAL,true,1.0,1.0);
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
  for(Precision precision:{Precision::FP64,Precision::FP32,Precision::TF32})
  {
    std::unique_ptr<mfem::Solver> preconditioner;
    {
      mfem::BilinearForm frozen(&fes);Assemble(frozen,reaction,diffusion);
      preconditioner=igrlab::igr_internal::MakeTensorMultigridPreconditioner(frozen.SpMat(),fes,precision);
    } // A0 lifetime ends: the preconditioner must own its coefficient snapshot.
    preconditioner->SetOperator(current.SpMat());
    preconditioner->Mult(probe,result);
    double action_error=0;
    if(precision==Precision::FP64) reference=result;
    else
    {
      action_error=Relative(result,reference);
      MFEM_VERIFY(action_error<(precision==Precision::FP32?5e-5:2e-2),
        "Tensor MG real patch batch control mismatch: "<<Name(precision)<<" relative="<<action_error);
    }
    // Dynamic power-of-two input scaling must preserve tiny and huge useful
    // residuals. Long-double comparison intentionally has no1e-300 norm floor.
    for(double scale:{1e-200,1e200})
    {
      const double* p=probe.HostRead();double* s=scaled.HostWrite();
      for(int i=0;i<n;++i)s[i]=scale*p[i];scaled.UseDevice(true);
      preconditioner->Mult(scaled,scaled_result);
      double* v=scaled_result.HostReadWrite();for(int i=0;i<n;++i)v[i]/=scale;
      const double error=Relative(scaled_result,result);
      MFEM_VERIFY(error<(precision==Precision::FP64?5e-10:2e-3),
        "Tensor MG scaled residual lost homogeneity: "<<Name(precision)<<" scale="<<scale<<" relative="<<error);
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
        <<" repeat="<<repeat<<" iterations="<<solver->GetNumIterations()<<" true_residual="<<residual
        <<" solution_error="<<error<<" patch_action_control_error="<<action_error<<'\n';
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
  CheckHierarchy(false);CheckHierarchy(true);
  std::cout<<"tensor_mg_test passed multiplicative_vcycle=true hierarchy=5,3,1"
    <<" original_A_certificate=true changed_coefficients=true real_patch_columns=true"
    <<" precision_controls=true scaled_rhs=true snapshot_lifetime=true\n";
  return 0;
}

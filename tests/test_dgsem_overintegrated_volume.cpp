#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "../src/numerics/dgsem/dgsem_overintegrated_volume.hpp"
#include "numerics/dgsem/dgsem_device.hpp"
#include "numerics/dgsem/dgsem_basis.hpp"
#include "physics/igr/igr_flux.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace mfem;
using namespace mfem::dgsem;
using namespace mfem::dgsem::overintegration_internal;

void StateValue(const Vector& x,int kind,Vector& u)
{
  u.SetSize(4);
  if(kind==0){u[0]=1.2;u[1]=.4;u[2]=-.1;u[3]=3.;return;}
  const double z=kind==1?std::sin(3.1*x[0]+.7*x[1]):std::tanh(6*(x[0]-.5));
  u[0]=1.1+.55*z;u[1]=.3+.17*std::sin(2*x[0]+x[1]);
  u[2]=-.08+.11*std::cos(3*x[1]-.2*x[0]);u[3]=3.2+.2*std::cos(x[0]+2*x[1]);
}
double SigmaValue(const Vector& x,int kind,double amplitude)
{
  return amplitude*(kind==0?.09:.07*std::cos(4*x[0])+.03*std::sin(2*x[1]));
}

DenseMatrix HostWeak(ParFiniteElementSpace& fes,const ParGridFunction& state,
                     const ParGridFunction& sigma,const IntegrationRule& ir,
                     bool use_sigma)
{
  const int ne=fes.GetNE(),dof=fes.GetFE(0)->GetDof(),rows=ne*dof;
  DenseMatrix out(rows,4);out=0.;DenseMatrix ds(dof,2),flux(4,2);Vector value(4);
  EulerFlux euler(2,1.4);FluxWithIGR igr(euler);igr.SetSigmaField(use_sigma?&sigma:nullptr);
  for(int e=0;e<ne;++e)
  {
    auto* tr=fes.GetElementTransformation(e);const auto& fe=*fes.GetFE(e);
    for(int q=0;q<ir.GetNPoints();++q)
    {
      const auto& ip=ir.IntPoint(q);tr->SetIntPoint(&ip);
      state.GetVectorValue(*tr,ip,value);igr.ComputeFlux(value,*tr,flux);
      fe.CalcPhysDShape(*tr,ds);
      for(int equation=0;equation<4;++equation)for(int i=0;i<dof;++i)for(int d=0;d<2;++d)
        out(e*dof+i,equation)+=ip.weight*tr->Weight()*ds(i,d)*flux(equation,d);
    }
  }
  return out;
}

double RelativeMax(const DenseMatrix& a,const DenseMatrix& b)
{
  MFEM_VERIFY(a.Height()==b.Height()&&a.Width()==b.Width(),"Comparison shape mismatch.");
  const double* x=a.HostRead();const double* y=b.HostRead();double error=0.,scale=1.;
  for(int i=0;i<a.Height()*a.Width();++i){error=std::max(error,std::abs(x[i]-y[i]));scale=std::max(scale,std::abs(y[i]));}
  return error/scale;
}

void CheckFreestream(ParFiniteElementSpace& fes,const ParGridFunction& sigma,
                     bool use_sigma,const DenseMatrix& weak)
{
  const int dof=fes.GetFE(0)->GetDof(),rows=fes.GetNE()*dof,p=fes.GetFE(0)->GetOrder();
  IntegrationRules lgl(0,Quadrature1D::GaussLobatto);
  const auto& face=lgl.Get(Geometry::SEGMENT,2*p-1);DenseMatrix residual(weak);residual.HostReadWrite();
  EulerFlux euler(2,1.4);FluxWithIGR igr(euler);igr.SetSigmaField(use_sigma?&sigma:nullptr);
  Vector state(4),x(2),shape(dof);DenseMatrix flux(4,2);
  for(int e=0;e<fes.GetNE();++e)
  {
    auto* tr=fes.GetElementTransformation(e);const auto& fe=*fes.GetFE(e);
    for(int side=0;side<4;++side)for(int q=0;q<face.GetNPoints();++q)
    {
      const auto& fp=face.IntPoint(q);IntegrationPoint ip;
      if(side<2)ip.Set2(side==0?0.:1.,fp.x);else ip.Set2(fp.x,side==2?0.:1.);
      tr->SetIntPoint(&ip);tr->Transform(ip,x);StateValue(x,0,state);igr.ComputeFlux(state,*tr,flux);
      fe.CalcShape(ip,shape);const auto& jac=tr->Jacobian();
      const double normal[2]={side<2?(side==0?-jac(1,1):jac(1,1)):(side==2?jac(1,0):-jac(1,0)),
                              side<2?(side==0?jac(0,1):-jac(0,1)):(side==2?-jac(0,0):jac(0,0))};
      for(int equation=0;equation<4;++equation)for(int i=0;i<dof;++i)
        residual(e*dof+i,equation)-=fp.weight*shape[i]*(normal[0]*flux(equation,0)+normal[1]*flux(equation,1));
    }
  }
  double maximum=0.;const double* r=residual.HostRead();
  for(int i=0;i<rows*4;++i)maximum=std::max(maximum,std::abs(r[i]));
  MFEM_VERIFY(maximum<2e-11,"Overintegrated volume plus retained LGL physical faces does not preserve a constant state: "<<maximum);
}

int main(int argc,char**argv)
{
  Mpi::Init(argc,argv);Hypre::Init();const std::string backend=argc>1?argv[1]:"cpu";
  MFEM_VERIFY(backend=="cpu"||backend=="cuda","Use cpu or cuda.");Device device(backend.c_str());
  MFEM_VERIFY(backend!="cuda"||Device::Allows(Backend::CUDA),"CUDA backend inactive.");
  double parity=0.,lgl_parity=0.,alias_difference=0.,budget=0.;int cases=0;
  for(int p:{3,5,7,10,12})
  {
    Mesh serial=Mesh::MakeCartesian2D(2,2,Element::QUADRILATERAL,true,1.,1.);
    serial.Transform([](const Vector& x,Vector& y)
    {const double a=x[0]+.2*x[0]*x[0];y.SetSize(2);y[0]=a+.15*x[1];y[1]=.08*a+.9*x[1];});
    ParMesh mesh(MPI_COMM_WORLD,serial);L2_FECollection fec(p,2,BasisType::GaussLobatto),sfec(p,2,BasisType::GaussLegendre);
    ParFiniteElementSpace fes(&mesh,&fec,4,Ordering::byVDIM),sfes(&mesh,&sfec);
    ParGridFunction state(&fes),sigma(&sfes);state.UseDevice(true);sigma.UseDevice(true);
    const int ne=fes.GetNE(),dof=fes.GetFE(0)->GetDof(),rows=ne*dof;
    DenseMatrix states(rows,4),metrics(rows,4);Vector sigma_nodes(rows);Array<int> map(dof),vdofs,sdofs;
    const auto& tensor=dynamic_cast<const TensorBasisElement&>(*fes.GetFE(0));const auto& local=tensor.GetDofMap();
    for(int j=0;j<dof;++j)map[j]=local.Size()?local[j]:j;
    states.GetMemory().UseDevice(true);metrics.GetMemory().UseDevice(true);sigma_nodes.UseDevice(true);
    const auto& basis=GetLGLBasis1D(p);IntegrationRules lgl(0,Quadrature1D::GaussLobatto);
    const auto& lgl_ir=lgl.Get(Geometry::SQUARE,2*p-1);
    for(int kind=0;kind<3;++kind)for(bool use_sigma:{false,true})for(int change=0;change<2;++change)
    {
      double* state_data=state.HostWrite();double* sigma_data=sigma.HostWrite();Vector x(2),u(4);
      for(int e=0;e<ne;++e)
      {
        fes.GetElementVDofs(e,vdofs);sfes.GetElementDofs(e,sdofs);auto* tr=mesh.GetElementTransformation(e);
        for(int j=0;j<dof;++j)
        {
          tr->Transform(fes.GetFE(e)->GetNodes().IntPoint(j),x);StateValue(x,kind,u);
          for(int c=0;c<4;++c)state_data[vdofs[j+dof*c]]=u[c];
          tr->Transform(sfes.GetFE(e)->GetNodes().IntPoint(j),x);
          sigma_data[sdofs[j]]=SigmaValue(x,kind,change?1.7:1.);
        }
      }
      double* input=states.HostWrite();double* sig=sigma_nodes.HostWrite();double* nodal_metric=metrics.HostWrite();
      for(int e=0;e<ne;++e)
      {
        auto* tr=mesh.GetElementTransformation(e);
        for(int j=0;j<dof;++j)
        {
          const auto& ip=fes.GetFE(e)->GetNodes().IntPoint(j);tr->SetIntPoint(&ip);state.GetVectorValue(*tr,ip,u);
          for(int c=0;c<4;++c)input[e*dof+j+rows*c]=u[c];sig[e*dof+j]=use_sigma?sigma.GetValue(*tr,ip):0.;
          const auto& jac=tr->Jacobian();const double adj[4]={jac(1,1),-jac(0,1),-jac(1,0),jac(0,0)};
          for(int k=0;k<4;++k)nodal_metric[e*dof+j+rows*k]=adj[k];
        }
      }
      DenseMatrix native_flux,native_weak;Vector native_speed;
      BuildEulerContravariantFluxDevice(2,1.4,1e-12,states,sigma_nodes,metrics,native_flux,native_speed);
      ApplyCentralVolumeDevice(2,p+1,map,basis.Dsplit,basis.weights,native_flux,native_weak);
      for(int quadrature:{-1,2*p,p>10?28:22})
      {
        const auto& ir=quadrature<0?lgl_ir:IntRules.Get(Geometry::SQUARE,quadrature);
        DenseMatrix b,g,qstate,qmetric(ne*ir.GetNPoints(),4),qflux,actual;Vector qsigma,qspeed;
        BuildVolumeQuadratureMaps2D(*fes.GetFE(0),ir,b,g);qmetric.GetMemory().UseDevice(true);
        double* qm=qmetric.HostWrite();const int qrows=ne*ir.GetNPoints();
        for(int e=0;e<ne;++e)for(int q=0;q<ir.GetNPoints();++q)for(int k=0;k<4;++k)
          qm[e*ir.GetNPoints()+q+qrows*k]=nodal_metric[e*dof+rows*k];
        InterpolateVolumeState2DDevice(ne,dof,b,states,sigma_nodes,qstate,qsigma);
        BuildEulerContravariantFluxDevice(2,1.4,1e-12,qstate,qsigma,qmetric,qflux,qspeed);
        ApplyWeakVolumeProjection2DDevice(ne,dof,g,qflux,actual);
        const auto expected=HostWeak(fes,state,sigma,ir,use_sigma);const double error=RelativeMax(actual,expected);
        parity=std::max(parity,error);MFEM_VERIFY(error<2e-11,"Host full-energy FluxWithIGR weak-volume parity failed: "<<error);
        const double native_delta=RelativeMax(actual,native_weak);
        if(quadrature<0){lgl_parity=std::max(lgl_parity,native_delta);MFEM_VERIFY(native_delta<2e-11,"LGL quadrature does not recover original central volume.");}
        else if(kind)alias_difference=std::max(alias_difference,native_delta);
        const double* values=actual.HostRead();double scale=1.;
        for(int i=0;i<rows*4;++i)scale=std::max(scale,std::abs(values[i]));
        for(int e=0;e<ne;++e)for(int c=0;c<4;++c)
        {
          double sum=0.;for(int j=0;j<dof;++j)sum+=values[e*dof+j+rows*c];
          budget=std::max(budget,std::abs(sum)/scale);
          MFEM_VERIFY(std::abs(sum)/scale<2e-11,"Volume projection changes element conservative means.");
        }
        if(kind==0)CheckFreestream(fes,sigma,use_sigma,actual);
        ++cases;
      }
    }
  }
  MFEM_VERIFY(alias_difference>1e-7,"Nonlinear aliasing control was accidentally degenerate.");
  std::cout<<"PASS overintegrated-volume device="<<backend<<" cases="<<cases
           <<" relative_host_parity="<<parity<<" relative_lgl_parity="<<lgl_parity
           <<" volume_sum_error="<<budget<<" nonlinear_volume_difference="<<alias_difference<<'\n';
}

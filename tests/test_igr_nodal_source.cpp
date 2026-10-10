#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "../src/physics/igr/IGR_nodal_source.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace mfem;
using namespace ::igrlab::igr_internal;

// Independent oracle: assemble vector loads from the current traces, solve
// each local mass, then evaluate the resulting polynomial gradient correction.
// Production instead retains combined geometry matrices and only applies them.
Vector Reference(ParFiniteElementSpace& fes, const QuadratureSpace& qs,
                 const ParGridFunction& state, const Vector& alpha, bool central)
{
  auto& mesh = *fes.GetParMesh();
  const int ne = fes.GetNE(), nd = fes.GetFE(0)->GetDof(), nq = qs.GetIntRule(0).GetNPoints();
  L2_FECollection vc(fes.GetFE(0)->GetOrder(), 2,
    dynamic_cast<const TensorBasisElement&>(*fes.GetFE(0)).GetBasisType());
  ParFiniteElementSpace vs(&mesh, &vc, 2, Ordering::byVDIM);
  ParGridFunction velocity(&vs); Array<int> ud, vd;
  const double* input = state.HostRead(); double* vv = velocity.HostWrite();
  for (int e=0;e<ne;++e)
  {
    fes.GetElementVDofs(e,ud); vs.GetElementVDofs(e,vd);
    for (int j=0;j<nd;++j) for (int v=0;v<2;++v)
      vv[vd[j+nd*v]] = input[ud[j+nd*(1+v)]]/std::max(input[ud[j]],.01);
  }
  std::vector<Vector> load(ne), correction(ne);
  for (int e=0;e<ne;++e) { load[e].SetSize(4*nd); load[e]=0.; correction[e].SetSize(4*nd); correction[e]=0.; }
  const int quad = 4*fes.GetFE(0)->GetOrder()+8;
  if (central)
  {
    const auto& faceir=IntRules.Get(Geometry::SEGMENT,quad);
    Vector s1(nd),s2(nd),normal(2),v1(2),v2(2);
    for (int f=0;f<mesh.GetNumFaces();++f)
    {
      auto* tr=mesh.GetInteriorFaceTransformations(f); if(!tr) continue;
      for (int q=0;q<faceir.GetNPoints();++q)
      {
        const auto& ip=faceir.IntPoint(q); tr->SetAllIntPoints(&ip);
        fes.GetFE(tr->Elem1No)->CalcShape(tr->GetElement1IntPoint(),s1);
        fes.GetFE(tr->Elem2No)->CalcShape(tr->GetElement2IntPoint(),s2);
        velocity.GetVectorValue(*tr->Elem1,tr->GetElement1IntPoint(),v1);
        velocity.GetVectorValue(*tr->Elem2,tr->GetElement2IntPoint(),v2);
        CalcOrtho(tr->Jacobian(),normal);
        for(int v=0;v<2;++v) for(int d=0;d<2;++d) for(int i=0;i<nd;++i)
        {
          const double value=.5*ip.weight*normal(d)*(v2(v)-v1(v));
          load[tr->Elem1No][i+nd*(v+2*d)]+=s1(i)*value;
          load[tr->Elem2No][i+nd*(v+2*d)]+=s2(i)*value;
        }
      }
    }
    for(int e=0;e<ne;++e)
    {
      auto* tr=mesh.GetElementTransformation(e); const auto& fe=*fes.GetFE(e);
      const auto& ir=IntRules.Get(Geometry::SQUARE,quad);
      DenseMatrix mass(nd); mass=0.; Vector shape(nd);
      for(int q=0;q<ir.GetNPoints();++q)
      {
        const auto& ip=ir.IntPoint(q); tr->SetIntPoint(&ip); fe.CalcShape(ip,shape);
        for(int j=0;j<nd;++j) for(int i=0;i<nd;++i)
          mass(i,j)+=ip.weight*tr->Weight()*shape(i)*shape(j);
      }
      DenseMatrixInverse solve(mass,true);
      for(int c=0;c<4;++c)
      {
        Vector b(load[e].GetData()+nd*c,nd),x(correction[e].GetData()+nd*c,nd);
        solve.Mult(b,x);
      }
    }
  }
  Vector result(ne*nq),shape(nd); DenseMatrix grad(2);
  for(int e=0;e<ne;++e)
  {
    auto* tr=mesh.GetElementTransformation(e); const auto& ir=qs.GetIntRule(e);
    for(int q=0;q<nq;++q)
    {
      const auto& ip=ir.IntPoint(q);tr->SetIntPoint(&ip);
      velocity.GetVectorGradient(*tr,grad); fes.GetFE(e)->CalcShape(ip,shape);
      for(int d=0;d<2;++d)for(int v=0;v<2;++v)for(int j=0;j<nd;++j)
        grad(v,d)+=shape(j)*correction[e][j+nd*(v+2*d)];
      result[q+nq*e]=::igrlab::frozen_scalar_igr_algebra::SourceFromVelocityGradient(2,grad.Data(),1,2,alpha[e]);
    }
  }
  return result;
}

void SetState(ParFiniteElementSpace& fes,ParGridFunction& state,int kind,double offset,double scale)
{
  Array<int> vdofs; Vector x(2); double* data=state.HostWrite();
  for(int e=0;e<fes.GetNE();++e)
  {
    fes.GetElementVDofs(e,vdofs); const auto& fe=*fes.GetFE(e);
    auto* tr=fes.GetMesh()->GetElementTransformation(e);const int nd=fe.GetDof();
    for(int j=0;j<nd;++j)
    {
      tr->Transform(fe.GetNodes().IntPoint(j),x);
      const double rho=1.0+.23*x[0]+.14*x[1];
      double u=0.,v=0.;
      if(kind==0){u=-.7*x[1];v=.7*x[0];} // rotation, negative source
      if(kind==1){u=.9*x[1];v=0.;} // shear, zero source
      if(kind==2){u=.1+x[0]*x[0]+.13*x[1];v=.2*x[0]-.3*x[1]*x[1];}
      if(kind==3){u=.2+.19*e;v=-.11*e;} // only face lifting sees these jumps
      u=scale*u+offset;v=scale*v-.37*offset;
      data[vdofs[j]]=rho;data[vdofs[j+nd]]=rho*u;data[vdofs[j+2*nd]]=rho*v;
      data[vdofs[j+3*nd]]=2.5+.5*rho*(u*u+v*v);
    }
  }
}
int main(int argc,char**argv)
{
  Mpi::Init(argc,argv); Hypre::Init();
  const std::string backend=argc>1?argv[1]:"cpu"; Device device(backend.c_str());
  MFEM_VERIFY(backend=="cpu"||backend=="cuda","Use cpu or cuda.");
  MFEM_VERIFY(backend!="cuda"||Device::Allows(Backend::CUDA),"CUDA backend is not active.");
  // Test the same admission predicate used before application QI launches.
  // Dense source maps themselves support larger Q; that does not qualify QI.
  const auto& limits=DeviceDofQuadLimits::Get();
  std::string reason;
  MFEM_VERIFY(SupportsSourceQuadratureTensor(2,limits.MAX_D1D,limits.MAX_Q1D,&reason),
              "Valid 2D quadrature boundary was rejected.");
  MFEM_VERIFY(!SupportsSourceQuadratureTensor(2,8,limits.MAX_Q1D+1,&reason)&&!reason.empty(),
              "Oversized 2D quadrature must be rejected before a QI launch.");
  MFEM_VERIFY(!SupportsSourceQuadratureTensor(2,limits.MAX_D1D+1,4,&reason),
              "Oversized 2D nodal basis must be rejected before a QI launch.");
  MFEM_VERIFY(SupportsSourceQuadratureTensor(3,limits.MAX_INTERP_1D,limits.MAX_INTERP_1D,&reason)&&
              !SupportsSourceQuadratureTensor(3,4,limits.MAX_INTERP_1D+1,&reason),
              "3D quadrature must use MAX_INTERP_1D, not the 2D bound.");
  MFEM_VERIFY(SupportsSourceQuadratureTensor(1,8,65,&reason)&&
              !SupportsSourceQuadratureTensor(0,8,4,&reason),
              "1D dynamic-loop/invalid-dimension admission failed.");
  if(backend=="cuda")
  {
    MFEM_VERIFY(!SupportsSourceQuadratureTensor(2,8,15,&reason),
                "Current CUDA generic 2D QI must reject order28 (15 Gauss points)." );
  }
  else
  {
    MFEM_VERIFY(SupportsSourceQuadratureTensor(2,8,15,&reason)==(limits.MAX_Q1D>=15),
                "CPU quadrature admission must use its own runtime limit.");
  }
  double max_error=0,max_offset_error=0;int cases=0;
  for(bool curved:{false,true})for(int p:{3,7})for(int basis:{BasisType::GaussLegendre,BasisType::GaussLobatto})
  {
    Mesh serial=Mesh::MakeCartesian2D(3,2,Element::QUADRILATERAL,true,1.,1.);
    serial.SetCurvature(2,false,2,Ordering::byVDIM);
    serial.Transform([=](const Vector& x,Vector& y)
    { y.SetSize(2);y[0]=x[0]+.12*x[0]*x[0];y[1]=x[1];
      if(curved){y[0]+=.08*x[0]*x[1];y[1]+=.06*x[0]*(1-x[0]);} });
    ParMesh mesh(MPI_COMM_WORLD,serial);L2_FECollection fec(p,2,basis);
    ParFiniteElementSpace fes(&mesh,&fec,4,Ordering::byVDIM);ParGridFunction state(&fes);state.UseDevice(true);
    const auto& ir=IntRules.Get(Geometry::SQUARE,4*p);QuadratureSpace qs(mesh,ir);
    Vector alpha(fes.GetNE());alpha.UseDevice(true);
    for(int e=0;e<fes.GetNE();++e)alpha[e]=.02*(1.+.07*e);
    for(bool central:{false,true})
    {
      NodalVelocitySourceCache cache;Vector evec,actual,first;
      const auto* restriction=fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
      evec.SetSize(restriction->Height());evec.UseDevice(true);
      for(int kind=0;kind<4;++kind)for(int change=0;change<3;++change)
      {
        const double scale=change==2?1.31:1.;
        SetState(fes,state,kind,change==1?.43:0.,scale);restriction->Mult(state,evec);
        ApplyNodalVelocitySource2D(fes,qs,evec,.02,&alpha,.01,central,cache,actual);
        Vector expected=Reference(fes,qs,state,alpha,central);const double* got=actual.HostRead();
        double error=0,denom=1.;
        for(int i=0;i<actual.Size();++i){error=std::max(error,std::abs(got[i]-expected[i]));denom=std::max(denom,std::abs(expected[i]));}
        max_error=std::max(max_error,error/denom);
        MFEM_VERIFY(error/denom<2e-10,"Nodal source quadrature/face/mass parity failed: "<<error/denom);
        if(change==0)first=actual;
        else
        {
          double delta=0,refscale=1.;const double factor=change==2?scale*scale:1.;
          const double* saved=first.HostRead();
          for(int i=0;i<actual.Size();++i){delta=std::max(delta,std::abs(got[i]-factor*saved[i]));refscale=std::max(refscale,std::abs(got[i]));}
          max_offset_error=std::max(max_offset_error,delta/refscale);
          MFEM_VERIFY(delta/refscale<2e-10,"Velocity-offset invariance/current-state cache update failed.");
        }
        if(kind==0||kind==1)
        {
          for(int e=0;e<fes.GetNE();++e)for(int q=0;q<ir.GetNPoints();++q)
          {
            const double analytic=kind==0?-2*.7*.7*scale*scale*alpha[e]:0.;
            MFEM_VERIFY(std::abs(got[q+ir.GetNPoints()*e]-analytic)<2e-10,
                        "Signed rotation/shear source identity failed.");
          }
        }
        if(kind==3&&!central)MFEM_VERIFY(actual.Normlinf()<2e-20,"Broken derivative must annihilate element constants.");
        ++cases;
      }
      MFEM_VERIFY(cache.rebuilds==1&&cache.hits==11,"Geometry cache incorrectly rebuilt or reused.");
      // Exercise global alpha and alpha zero through the same actual kernel.
      ApplyNodalVelocitySource2D(fes,qs,evec,0.,nullptr,.01,central,cache,actual);
      MFEM_VERIFY(actual.Normlinf()==0.,"Alpha-zero source must vanish exactly.");
    }
  }
  std::cout<<"PASS nodal-source device="<<backend<<" cases="<<cases
           <<" max_relative_parity="<<max_error<<" max_invariance_error="<<max_offset_error<<'\n';
}

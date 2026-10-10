#include "mfem.hpp"
#include "../../src/physics/igr/IGR_nodal_source.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "sim/wiring/bc_wiring.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "sim/context/mesh_build.hpp"
#include "igr_wall_consistency_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <vector>

using namespace mfem;
using namespace ::dg_igr::igr_internal;
namespace
{
// Independent current-trace oracle: assemble four vector loads, solve local
// physical masses, then evaluate. Production applies cached geometry maps.
Vector Reference(ParFiniteElementSpace& fes,const QuadratureSpace& qs,
                 const ParGridFunction& state,const Array<int>& marker,
                 bool zero_tangent=false)
{
  auto& mesh=*fes.GetParMesh(); const int p=fes.GetFE(0)->GetOrder();
  const int ne=fes.GetNE(),nd=fes.GetFE(0)->GetDof(),nq=qs.GetIntRule(0).GetNPoints();
  L2_FECollection fec(p,2,dynamic_cast<const TensorBasisElement&>(*fes.GetFE(0)).GetBasisType());
  ParFiniteElementSpace vfes(&mesh,&fec,2,Ordering::byVDIM);
  ParGridFunction velocity(&vfes);Array<int> ud,vd;
  const double* input=state.HostRead();double* vv=velocity.HostWrite();
  for(int e=0;e<ne;++e)
  {
    fes.GetElementVDofs(e,ud);vfes.GetElementVDofs(e,vd);
    for(int j=0;j<nd;++j)for(int v=0;v<2;++v)
      vv[vd[j+nd*v]]=input[ud[j+nd*(1+v)]]/std::max(input[ud[j]],.01);
  }
  std::vector<Vector> load(ne),correction(ne);
  for(int e=0;e<ne;++e){load[e].SetSize(4*nd);load[e]=0.;correction[e].SetSize(4*nd);correction[e]=0.;}
  const auto& faceir=IntRules.Get(Geometry::SEGMENT,6*p+18);
  Vector s1(nd),s2(nd),normal(2),v1(2),v2(2);
  for(int f=0;f<mesh.GetNumFaces();++f)
  {
    auto* tr=mesh.GetInteriorFaceTransformations(f);if(!tr)continue;
    for(int q=0;q<faceir.GetNPoints();++q)
    {
      const auto& ip=faceir.IntPoint(q);tr->SetAllIntPoints(&ip);
      fes.GetFE(tr->Elem1No)->CalcShape(tr->GetElement1IntPoint(),s1);
      fes.GetFE(tr->Elem2No)->CalcShape(tr->GetElement2IntPoint(),s2);
      velocity.GetVectorValue(*tr->Elem1,tr->GetElement1IntPoint(),v1);
      velocity.GetVectorValue(*tr->Elem2,tr->GetElement2IntPoint(),v2);
      CalcOrtho(tr->Jacobian(),normal);
      for(int v=0;v<2;++v)for(int d=0;d<2;++d)for(int i=0;i<nd;++i)
      {
        const double value=.5*ip.weight*normal[d]*(v2[v]-v1[v]);
        load[tr->Elem1No][i+nd*(v+2*d)]+=s1[i]*value;
        load[tr->Elem2No][i+nd*(v+2*d)]+=s2[i]*value;
      }
    }
  }
  for(int b=0;b<mesh.GetNBE();++b)
  {
    const int attr=mesh.GetBdrAttribute(b);
    if(attr<=0||attr>marker.Size()||!marker[attr-1])continue;
    auto* tr=mesh.GetBdrFaceTransformations(b);
    for(int q=0;q<faceir.GetNPoints();++q)
    {
      const auto& ip=faceir.IntPoint(q);tr->SetAllIntPoints(&ip);
      fes.GetFE(tr->Elem1No)->CalcShape(tr->GetElement1IntPoint(),s1);
      velocity.GetVectorValue(*tr->Elem1,tr->GetElement1IntPoint(),v1);
      CalcOrtho(tr->Jacobian(),normal);const double jac=normal.Norml2();
      const double un=(v1*normal)/jac;
      for(int v=0;v<2;++v)for(int d=0;d<2;++d)for(int i=0;i<nd;++i)
      {
        const double delta=zero_tangent?-v1[v]:-un*normal[v]/jac;
        load[tr->Elem1No][i+nd*(v+2*d)]+=ip.weight*s1[i]*delta*normal[d];
      }
    }
  }
  const auto& massir=IntRules.Get(Geometry::SQUARE,4*p+10);
  for(int e=0;e<ne;++e)
  {
    auto* tr=mesh.GetElementTransformation(e);const auto& fe=*fes.GetFE(e);
    DenseMatrix mass(nd);mass=0.;Vector shape(nd);
    for(int q=0;q<massir.GetNPoints();++q)
    {
      const auto& ip=massir.IntPoint(q);tr->SetIntPoint(&ip);fe.CalcShape(ip,shape);
      for(int j=0;j<nd;++j)for(int i=0;i<nd;++i)
        mass(i,j)+=ip.weight*tr->Weight()*shape[i]*shape[j];
    }
    DenseMatrixInverse solve(mass,true);
    for(int c=0;c<4;++c)
    {
      Vector rhs(load[e].GetData()+nd*c,nd),out(correction[e].GetData()+nd*c,nd);solve.Mult(rhs,out);
    }
  }
  Vector result(ne*nq),shape(nd);DenseMatrix grad(2);
  for(int e=0;e<ne;++e)
  {
    auto* tr=mesh.GetElementTransformation(e);const auto& ir=qs.GetIntRule(e);
    for(int q=0;q<nq;++q)
    {
      const auto& ip=ir.IntPoint(q);tr->SetIntPoint(&ip);
      velocity.GetVectorGradient(*tr,grad);fes.GetFE(e)->CalcShape(ip,shape);
      for(int d=0;d<2;++d)for(int v=0;v<2;++v)for(int j=0;j<nd;++j)
        grad(v,d)+=shape[j]*correction[e][j+nd*(v+2*d)];
      result[q+nq*e]=::dg_igr::frozen_scalar_igr_algebra::SourceFromVelocityGradient(2,grad.Data(),1,2,.0025);
    }
  }
  return result;
}
void SetState(ParFiniteElementSpace& fes,ParGridFunction& state,bool reflected=false)
{
  Array<int> vd;Vector x(2),center(2);double* data=state.HostWrite();
  for(int e=0;e<fes.GetNE();++e)
  {
    fes.GetElementVDofs(e,vd);const auto& fe=*fes.GetFE(e);const int nd=fe.GetDof();
    auto* tr=fes.GetMesh()->GetElementTransformation(e);
    tr->Transform(Geometries.GetCenter(Geometry::SQUARE),center);
    const bool mirror=reflected&&center[0]>1.;
    for(int j=0;j<nd;++j)
    {
      tr->Transform(fe.GetNodes().IntPoint(j),x);
      const double xx=mirror?2.-x[0]:x[0], yy=x[1];
      const double rho=1.+.1*xx+.07*yy;
      double u=.23+.31*xx*xx+.17*yy,v=-.16+.19*xx-.21*yy*yy;
      if(mirror)u=-u;
      data[vd[j]]=rho;data[vd[j+nd]]=rho*u;data[vd[j+2*nd]]=rho*v;
      data[vd[j+3*nd]]=2.5+.5*rho*(u*u+v*v);
    }
  }
}
double Difference(const Vector& a,const Vector& b)
{
  MFEM_VERIFY(a.Size()==b.Size(),"Comparison layout mismatch.");
  const auto* aa=a.HostRead();const auto* bb=b.HostRead();double error=0.,denom=1.;
  for(int i=0;i<a.Size();++i){error=std::max(error,std::abs(aa[i]-bb[i]));denom=std::max(denom,std::abs(bb[i]));}
  return error/denom;
}
Vector Apply(ParFiniteElementSpace& fes,QuadratureSpace& qs,ParGridFunction& state,
             const Array<int>* marker,NodalVelocitySourceCache& cache)
{
  const auto* restriction=fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);
  Vector evec(restriction->Height()),out;evec.UseDevice(true);restriction->Mult(state,evec);
  ApplyNodalVelocitySource2D(fes,qs,evec,.0025,nullptr,.01,true,cache,out,marker);return out;
}
void CheckMarkerPrecedence(ParFiniteElementSpace& fes)
{
  dg_sim::BoundaryMarkerMap markers;Array<int> all(fes.GetMesh()->bdr_attributes.Max());all=1;
  markers.Add(dg_sim::BoundaryRole::Slip,all);markers.Add(dg_sim::BoundaryRole::Outlet,all);
  for(bool slip_first:{false,true})
  {
    EulerFlux euler(2,1.4);RusanovFluxWithSides flux(euler);
    auto form=std::make_unique<HyperbolicFormIntegrator>(flux,2);
    DGHyperbolicConservationLaws op(fes,std::move(form),false);
    dg_sim::EulerBoundaryWiring<RusanovFluxWithSides> wiring(op,flux,2,2,1.4);
    if(slip_first){wiring.AddSlipWall(markers);wiring.AddSupersonicOutlet(markers);}
    else{wiring.AddSupersonicOutlet(markers);wiring.AddSlipWall(markers);}
    const auto marker=wiring.SlipWallMarker();
    MFEM_VERIFY(marker.Size()==all.Size(),"Boundary metadata lost attributes.");
    for(int i=0;i<marker.Size();++i)MFEM_VERIFY(marker[i]==int(slip_first),"Boundary metadata must honor first registered kind.");
  }
}
void ReflectionTest(int p,int basis,double& maximum)
{
  Mesh sh=Mesh::MakeCartesian2D(2,2,Element::QUADRILATERAL,true,1.,1.);
  Mesh sf=Mesh::MakeCartesian2D(4,2,Element::QUADRILATERAL,true,2.,1.);
  ParMesh half(MPI_COMM_WORLD,sh),full(MPI_COMM_WORLD,sf);L2_FECollection fec(p,2,basis);
  ParFiniteElementSpace hf(&half,&fec,4,Ordering::byVDIM),ff(&full,&fec,4,Ordering::byVDIM);
  ParGridFunction uh(&hf),uf(&ff);SetState(hf,uh);SetState(ff,uf,true);
  const auto& ir=IntRules.Get(Geometry::SQUARE,2*p+2);QuadratureSpace hq(half,ir),fq(full,ir);
  Array<int> marker(half.bdr_attributes.Max());marker=0;Vector center(2);
  for(int b=0;b<half.GetNBE();++b)
  {
    half.GetBdrElementTransformation(b)->Transform(Geometries.GetCenter(Geometry::SEGMENT),center);
    if(std::abs(center[0]-1.)<1e-12)marker[half.GetBdrAttribute(b)-1]=1;
  }
  NodalVelocitySourceCache hc,fc;const Vector h=Apply(hf,hq,uh,&marker,hc),f=Apply(ff,fq,uf,nullptr,fc);
  Vector expected(h.Size()),hcpos(2),fcpos(2);const auto* data=f.HostRead();
  for(int he=0;he<hf.GetNE();++he)
  {
    half.GetElementTransformation(he)->Transform(Geometries.GetCenter(Geometry::SQUARE),hcpos);
    int match=-1;
    for(int fe=0;fe<ff.GetNE();++fe)
    {
      full.GetElementTransformation(fe)->Transform(Geometries.GetCenter(Geometry::SQUARE),fcpos);
      fcpos-=hcpos;if(fcpos.Norml2()<1e-12){match=fe;break;}
    }
    MFEM_VERIFY(match>=0,"Reflected mesh has no matching original element.");
    for(int q=0;q<ir.GetNPoints();++q)expected[q+ir.GetNPoints()*he]=data[q+ir.GetNPoints()*match];
  }
  const double error=Difference(h,expected);maximum=std::max(maximum,error);
  MFEM_VERIFY(error<3e-10,"Actual reflected full-domain source differs from mirror wall: "<<error);
}
void AnnulusTest()
{
  std::ostringstream json;json<<std::setprecision(17)<<"{\"schema\":\"igr-wall-annulus-source-v1\",\"alpha\":0.0025,\"omega\":1,\"rows\":[";
  bool first=true;
  for(int geometry_order:{2,3,4})for(int angular:{8,16})for(int p:{5,7})
  {
    if(geometry_order==4&&(p!=7||angular!=8))continue;
    auto serial=igr_wall_consistency::Annulus(2,angular,geometry_order);ParMesh mesh(MPI_COMM_WORLD,serial);
    L2_FECollection fec(p,2,BasisType::GaussLobatto);
    ParFiniteElementSpace fes(&mesh,&fec,4,Ordering::byVDIM);ParGridFunction state(&fes);
    Array<int> vd;Vector x(2),value(4);double* data=state.HostWrite();
    for(int e=0;e<fes.GetNE();++e)
    {
      fes.GetElementVDofs(e,vd);const auto& fe=*fes.GetFE(e);const int nd=fe.GetDof();
      for(int j=0;j<nd;++j)
      {
        mesh.GetElementTransformation(e)->Transform(fe.GetNodes().IntPoint(j),x);
        igr_wall_consistency::Rotation(x,value);
        for(int c=0;c<4;++c)data[vd[j+nd*c]]=value[c];
      }
    }
    const auto& ir=IntRules.Get(Geometry::SQUARE,22);QuadratureSpace qs(mesh,ir);
    Array<int> marker(mesh.bdr_attributes.Max());marker=1;NodalVelocitySourceCache own_cache,mirror_cache;
    const Vector own=Apply(fes,qs,state,nullptr,own_cache),mirror=Apply(fes,qs,state,&marker,mirror_cache);
    const Vector expected=Reference(fes,qs,state,marker);
    const double oracle_error=Difference(mirror,expected);MFEM_VERIFY(oracle_error<3e-10,"Annulus mirror-map oracle mismatch.");
    double own_error=0.,mirror_error=0.,own_squared=0.,mirror_squared=0.,volume=0.;
    double own_mean=0.,mirror_mean=0.;const double* oq=own.HostRead();const double* mq=mirror.HostRead();
    for(int e=0;e<fes.GetNE();++e)for(int q=0;q<ir.GetNPoints();++q)
    {
      const auto& ip=ir.IntPoint(q);auto* tr=mesh.GetElementTransformation(e);tr->SetIntPoint(&ip);
      const double weight=ip.weight*tr->Weight(),eo=oq[q+ir.GetNPoints()*e]+.005,em=mq[q+ir.GetNPoints()*e]+.005;
      own_error=std::max(own_error,std::abs(eo));mirror_error=std::max(mirror_error,std::abs(em));
      own_squared+=weight*eo*eo;mirror_squared+=weight*em*em;volume+=weight;
      own_mean+=weight*oq[q+ir.GetNPoints()*e];mirror_mean+=weight*mq[q+ir.GetNPoints()*e];
    }
    MFEM_VERIFY(own_error<3e-10,"Own-trace rigid rotation must reproduce the signed constant source.");
    double radius_error=0.,normal_error=0.,unmax=0.,un2=0.,wall_length=0.;Vector normal(2);
    const auto& faceir=IntRules.Get(Geometry::SEGMENT,60);
    for(int b=0;b<mesh.GetNBE();++b)
    {
      auto* tr=mesh.GetBdrFaceTransformations(b);const bool inner=mesh.GetBdrAttribute(b)==1;
      for(int q=0;q<faceir.GetNPoints();++q)
      {
        const auto& ip=faceir.IntPoint(q);tr->SetAllIntPoints(&ip);tr->Transform(ip,x);CalcOrtho(tr->Jacobian(),normal);
        const double jac=normal.Norml2(),r=x.Norml2();normal/=jac;
        const double un=-x[1]*normal[0]+x[0]*normal[1];
        radius_error=std::max(radius_error,std::abs(r-(inner?.5:1.)));
        double difference=0.;for(int d=0;d<2;++d)difference+=std::pow(normal[d]-(inner?-1.:1.)*x[d]/r,2);
        normal_error=std::max(normal_error,std::sqrt(difference));unmax=std::max(unmax,std::abs(un));
        un2+=ip.weight*jac*un*un;wall_length+=ip.weight*jac;
      }
    }
    if(!first)json<<',';first=false;
    json<<"{\"radial\":2,\"angular\":"<<angular<<",\"geometry_order\":"<<geometry_order<<",\"p\":"<<p
      <<",\"own_source_max_error\":"<<own_error<<",\"own_source_relative_l2\":"<<std::sqrt(own_squared/volume)/.005
      <<",\"mirror_source_max_error\":"<<mirror_error<<",\"mirror_source_relative_l2\":"<<std::sqrt(mirror_squared/volume)/.005
      <<",\"own_source_mean\":"<<own_mean/volume<<",\"mirror_source_mean\":"<<mirror_mean/volume
      <<",\"mirror_oracle_error\":"<<oracle_error<<",\"radius_max_error\":"<<radius_error
      <<",\"normal_max_error\":"<<normal_error<<",\"max_analytic_wall_un\":"<<unmax
      <<",\"rms_analytic_wall_un\":"<<std::sqrt(un2/wall_length)<<'}';
  }
  json<<"]}\n";std::cout<<json.str();
}
}
int main(int argc,char** argv)
{
  Mpi::Init(argc,argv);Hypre::Init();const std::string backend=argc>1?argv[1]:"cpu";
  MFEM_VERIFY(backend=="cpu"||backend=="cuda","Use cpu or cuda.");Device device(backend.c_str());
  MFEM_VERIFY(backend!="cuda"||Device::Allows(Backend::CUDA),"CUDA backend is not active.");
  if(argc>2&&std::string(argv[2])=="--write-annulus")
  {
    MFEM_VERIFY(argc==6||argc==7,"Use cpu --write-annulus output.mesh radial angular [geometry-order].");
    const int radial=std::stoi(argv[4]),angular=std::stoi(argv[5]),geometry_order=argc==7?std::stoi(argv[6]):2;
    MFEM_VERIFY(geometry_order>=2&&geometry_order<=4,"Use qualified geometry order 2, 3 or 4.");
    auto serial=igr_wall_consistency::Annulus(radial,angular,geometry_order);std::ofstream output(argv[3]);
    MFEM_VERIFY(output.good(),"Cannot write annulus mesh.");output<<std::setprecision(17);serial.Print(output);output.close();
    ParMesh mesh(MPI_COMM_WORLD,serial);
    const auto alpha=dg_sim::ComputeAlphaInfo(mesh,1.,::dg_igr::IGRAlphaLengthScalePolicy::Average);
    std::cout<<std::setprecision(17)<<"{\"radial\":"<<radial<<",\"angular\":"<<angular<<",\"geometry_order\":"<<geometry_order<<",\"havg\":"<<alpha.havg
      <<",\"alpha\":0.0025,\"c_alpha\":"<<.0025/(alpha.havg*alpha.havg)<<"}\n";return 0;
  }
  if(argc>2&&std::string(argv[2])=="--annulus"){MFEM_VERIFY(argc==3,"Use cpu|cuda --annulus.");AnnulusTest();return 0;}
  MFEM_VERIFY(argc<=2,"Unknown wall source probe argument.");
  double max_error=0.,max_reflection=0.,min_wrong=1e100,max_tangent=0.;int cases=0;
  for(bool curved:{false,true})for(int p:{3,7})for(int basis:{BasisType::GaussLegendre,BasisType::GaussLobatto})
  {
    Mesh serial=Mesh::MakeCartesian2D(2,2,Element::QUADRILATERAL,true,1.,1.);
    serial.SetCurvature(2,false,2,Ordering::byVDIM);
    serial.Transform([=](const Vector& x,Vector& y)
    {
      const double xx=x[0]+(curved?.11*x[0]*x[1]:0.);
      const double yy=x[1]+(curved?.08*x[0]*(1.-x[0]):0.);
      y.SetSize(2);y[0]=std::cos(.37)*xx-std::sin(.37)*yy;y[1]=std::sin(.37)*xx+std::cos(.37)*yy;
    });
    ParMesh mesh(MPI_COMM_WORLD,serial);L2_FECollection fec(p,2,basis);
    ParFiniteElementSpace fes(&mesh,&fec,4,Ordering::byVDIM);ParGridFunction state(&fes);SetState(fes,state);
    const auto& ir=IntRules.Get(Geometry::SQUARE,2*p+2);QuadratureSpace qs(mesh,ir);
    Array<int> marker(mesh.bdr_attributes.Max());marker=1;
    NodalVelocitySourceCache cache;const Vector actual=Apply(fes,qs,state,&marker,cache);
    const Vector expected=Reference(fes,qs,state,marker),wrong=Reference(fes,qs,state,marker,true);
    const double error=Difference(actual,expected);max_error=std::max(max_error,error);
    min_wrong=std::min(min_wrong,Difference(wrong,expected));
    MFEM_VERIFY(error<3e-10,"Curved/oblique source oracle differs: "<<error);
    MFEM_VERIFY(Difference(wrong,expected)>1e-3,"Chosen vector field does not distinguish tangential erasure.");
    const Vector repeat=Apply(fes,qs,state,&marker,cache);
    MFEM_VERIFY(Difference(repeat,actual)==0.&&cache.rebuilds==1&&cache.hits==1,"Wall cache did not reuse identical geometry.");
    marker=0;const Vector empty=Apply(fes,qs,state,&marker,cache);
    MFEM_VERIFY(cache.wall_element_count==0&&cache.wall_derivative.Size()==0&&cache.rebuilds==2,"Changed marker did not rebuild compact wall cache.");
    const Vector own=Apply(fes,qs,state,nullptr,cache);
    MFEM_VERIFY(Difference(empty,own)==0.&&cache.rebuilds==3,"Empty wall marker must reproduce own trace.");
    if(!curved)
    {
      // Flat oblique right wall: a constant tangential velocity must produce
      // no gradient correction. Inspect actual cached maps because the source
      // invariant alone also vanishes for an erroneous rank-one pure shear.
      marker=0;marker[1]=1;Apply(fes,qs,state,&marker,cache);
      const double tx=-std::sin(.37),ty=std::cos(.37);
      const double* w=cache.wall_derivative.HostRead();double defect=0.;
      for(int slot=0;slot<cache.wall_element_count;++slot)for(int q=0;q<cache.nq;++q)
      {
        double g[4]={0,0,0,0};
        for(int j=0;j<cache.ndof;++j)
        {
          const int k=j+cache.ndof*4*(q+cache.nq*slot);
          g[0]+=w[k]*tx+w[k+cache.ndof]*ty;
          g[1]+=w[k+cache.ndof]*tx+w[k+2*cache.ndof]*ty;
          g[2]=g[1];g[3]+=w[k+2*cache.ndof]*tx+w[k+3*cache.ndof]*ty;
        }
        for(double value:g)defect=std::max(defect,std::abs(value));
      }
      max_tangent=std::max(max_tangent,defect);
      MFEM_VERIFY(defect<3e-10,"Oblique wall changed a constant tangential velocity.");
      ReflectionTest(p,basis,max_reflection);
      if(p==3&&basis==BasisType::GaussLobatto)CheckMarkerPrecedence(fes);
    }
    ++cases;
  }
  std::cout<<"PASS wall-source device="<<backend<<" cases="<<cases
    <<" max_oracle_error="<<max_error<<" max_reflection_error="<<max_reflection
    <<" max_tangent_gradient="<<max_tangent<<" min_wrong_tangent_source="<<min_wrong<<'\n';
}

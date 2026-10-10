// Test-only stationary slip-wall flux and discrete boundary-work audit.
// Uses the production packed face gather/Rusanov kernels on CPU or CUDA, the shared
// host reflection policy, shared physical sigma flux, and existing DG basis.
// No closure boundary condition or production wall policy is changed.
#define DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY
#include "igr_hp_front_tracking_probe.cpp"
#undef DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY
#include "numerics/dgsem/dgsem_device.hpp"
#include "physics/bc/boundary_state.hpp"
#include "physics/bc/detail/euler_boundary_state.hpp"

namespace
{
using Four=std::array<double,4>;
using Two=std::array<double,2>;
const std::vector<std::string> wall_policies{
  "mirrored-copy", "mirrored-zero", "direct-traction", "direct-traction-llf"};

double Pressure(const Four& a)
{mfem::Vector v(const_cast<double*>(a.data()),4);return mfem::EulerPressureFromConservative(v,2,gamma_euler);}
Four WallPrimitive(double rho,double vn,double vt,double pressure,const Two& unit)
{const double vx=vn*unit[0]-vt*unit[1],vy=vn*unit[1]+vt*unit[0];return {rho,rho*vx,rho*vy,pressure/(gamma_euler-1.)+.5*rho*(vx*vx+vy*vy)};}
void CheckNear(double a,double b,double tolerance,const char* message)
{if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tolerance*std::max({1.,std::abs(a),std::abs(b)}))throw std::runtime_error(message);}
void WriteFour(std::ostream& out,const Four& a)
{out<<'[';for(int c=0;c<4;++c){if(c)out<<',';out<<a[c];}out<<']';}

struct WallFluxes
{
  Four exterior{},total{},baseline{},central_sigma{},sigma_speed{};
  double gathered_sigma=0.,speed=0.,baseline_speed=0.,reflection_error=0.;
};

WallFluxes ProductionWallFlux(const Four& u,const Two& normal,double sigma,const std::string& policy)
{
  // One boundary row exercises the same gather and point-flux implementation
  // used by production DGSEM, on the explicitly selected backend.
  mfem::DenseMatrix packed(1,4),normals(1,2),reference(1,4),left,right;
  for(int c=0;c<4;++c)packed(0,c)=reference(0,c)=u[c];
  for(int d=0;d<2;++d)normals(0,d)=normal[d];
  mfem::Vector packed_sigma(1),neighbor(0),shared_sigma(1),gammas(1),pressure_out(1),density_floor(1),pressure_floor(1),sl,sr;
  packed_sigma[0]=sigma;shared_sigma=0.;gammas=gamma_euler;pressure_out=1.;density_floor=floor_value;pressure_floor=floor_value;
  mfem::Array<int> element_l(1),local_l(1),element_r(1),local_r(1),owner(1),kind(1),shared_vdofs(4);
  element_l=0;local_l=0;element_r=-1;local_r=-1;shared_vdofs=0;
  owner[0]=static_cast<int>(mfem::dgsem::DGSEMDeviceFaceOwnership::Boundary);
  kind[0]=static_cast<int>(mfem::detail::EulerBoundaryStateKind::slip_wall);
  mfem::dgsem::GatherEulerRusanovFacesDevice(2,1,1,packed,packed_sigma,neighbor,shared_vdofs,shared_sigma,normals,
    element_l,local_l,element_r,local_r,owner,kind,reference,gammas,pressure_out,density_floor,pressure_floor,left,right,sl,sr);
  WallFluxes out;out.gathered_sigma=sr.HostRead()[0];
  const double* exterior=right.HostRead();for(int c=0;c<4;++c)out.exterior[c]=exterior[c];
  mfem::Vector uv(const_cast<double*>(u.data()),4),nv(const_cast<double*>(normal.data()),2),host_exterior;
  mfem::FaceElementTransformations unused;mfem::SlipWallState host_wall(2,floor_value,floor_value);
  host_wall.Eval(uv,nv,unused,host_exterior);
  for(int c=0;c<4;++c)out.reflection_error=std::max(out.reflection_error,std::abs(host_exterior[c]-out.exterior[c]));
  CheckNear(out.gathered_sigma,sigma,2.e-13,"production wall gather did not copy sigma");
  CheckNear(out.reflection_error,0.,2.e-13,"host and packed wall reflection disagree");

  mfem::Vector face_weights(1),speeds;face_weights=1.;mfem::DenseMatrix flux,weak;
  mfem::Array<int> offsets(2),ids(1),signs(1);offsets[0]=0;offsets[1]=1;ids[0]=0;signs[0]=-1;
  auto numerical=[&](double a,double b,double& speed)
  {
    sl.HostWrite()[0]=a;sr.HostWrite()[0]=b;
    mfem::dgsem::AssembleEulerRusanovFacesDevice(2,1,1,gamma_euler,floor_value,left,right,normals,sl,sr,face_weights,
      offsets,ids,signs,flux,speeds,weak,true,false);
    Four f{};const double* values=flux.HostRead();for(int c=0;c<4;++c)f[c]=values[c];speed=speeds.HostRead()[0];return f;
  };
  out.baseline=numerical(0.,0.,out.baseline_speed);
  const double exterior_sigma=policy=="mirrored-zero"?0.:sigma;
  out.total=numerical(sigma,exterior_sigma,out.speed);
  Four fl{},fr{};
  igr::AddNormalSigmaFlux(2,u.data(),1,normal.data(),1,sigma,floor_value,fl.data(),1);
  igr::AddNormalSigmaFlux(2,out.exterior.data(),1,normal.data(),1,exterior_sigma,floor_value,fr.data(),1);
  for(int c=0;c<4;++c)out.central_sigma[c]=.5*(fl[c]+fr[c]);
  if(policy=="direct-traction"||policy=="direct-traction-llf")
  {
    const bool stabilized=policy=="direct-traction-llf";
    const double length=std::hypot(normal[0],normal[1]);
    const double vn=(u[1]*normal[0]+u[2]*normal[1])/(u[0]*length);
    out.total={};out.baseline={};out.central_sigma={};
    for(int d=0;d<2;++d)
    {
      out.baseline[d+1]=(Pressure(u)+(stabilized?u[0]*out.baseline_speed*vn:0.))*normal[d];
      out.total[d+1]=(Pressure(u)+sigma+(stabilized?u[0]*out.speed*vn:0.))*normal[d];
      out.central_sigma[d+1]=sigma*normal[d];
    }
  }
  for(int c=0;c<4;++c)out.sigma_speed[c]=out.total[c]-out.baseline[c]-out.central_sigma[c];
  return out;
}

void CheckFlux(const Four& u,const Two& normal,double sigma,const std::string& policy,const WallFluxes& f)
{
  const double length=std::hypot(normal[0],normal[1]);
  const Two unit{normal[0]/length,normal[1]/length};
  const double vn=(u[1]*unit[0]+u[2]*unit[1])/u[0];
  const bool mirror=policy=="mirrored-copy"||policy=="mirrored-zero";
  const double sigma_average=(policy=="mirrored-zero"?.5:1.)*sigma;
  const double expected_traction=Pressure(u)+sigma_average+(mirror?u[0]*vn*vn:0.)+
    (policy=="direct-traction"?0.:u[0]*f.speed*vn);
  CheckNear(f.total[0],0.,3.e-12,"stationary wall mass flux failed");
  CheckNear(f.total[3],policy=="mirrored-zero"?.5*sigma*vn*length:0.,3.e-12,"wall energy flux identity failed");
  CheckNear((f.total[1]*unit[0]+f.total[2]*unit[1])/length,expected_traction,3.e-12,"wall normal traction failed");
  CheckNear((-f.total[1]*unit[1]+f.total[2]*unit[0])/length,0.,3.e-12,"wall tangential traction failed");
  CheckNear(f.sigma_speed[3],0.,3.e-12,"sigma speed increment leaked energy");
  CheckNear(f.sigma_speed[0],0.,3.e-12,"sigma speed increment leaked mass");
  const double expected_speed=policy=="direct-traction"?0.:u[0]*(f.speed-f.baseline_speed)*vn;
  CheckNear((f.sigma_speed[1]*unit[0]+f.sigma_speed[2]*unit[1])/length,expected_speed,3.e-12,"wall sigma speed traction identity failed");
}

void WriteFluxes(std::ostream& out,const WallFluxes& f)
{
  out<<"{\"total\":";WriteFour(out,f.total);out<<",\"policy_euler_baseline\":";WriteFour(out,f.baseline);
  out<<",\"central_sigma_increment\":";WriteFour(out,f.central_sigma);out<<",\"sigma_llf_speed_increment\":";WriteFour(out,f.sigma_speed);
  out<<",\"speed\":"<<f.speed<<",\"baseline_speed\":"<<f.baseline_speed<<",\"production_gathered_exterior_sigma\":"<<f.gathered_sigma
     <<",\"host_packed_reflection_maximum_difference\":"<<f.reflection_error<<'}';
}

struct WallRule{State w;std::vector<State> v;};
struct WallElement:TrackingProblem
{
  WallRule method,audit;
  WallElement(int degree,const std::string& mass_kind):TrackingProblem(MakeOptions(degree))
  {
    if(mass_kind=="collocated")
    {
      mass=0.;mass_inverse=0.;
      for(int i=0;i<n;++i)
      {method.w.push_back(physics.basis.weights[i]);method.v.push_back(Shape(physics.basis.nodes[i],false));
        mass(i,i)=physics.basis.weights[i];mass_inverse(i,i)=1./physics.basis.weights[i];}
    }
    else method={weights,shapes};
    const auto& rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,8*p+50);
    for(int k=0;k<rule.GetNPoints();++k){const auto& ip=rule.IntPoint(k);audit.w.push_back(ip.weight);audit.v.push_back(Shape(ip.x,false));}
  }
  static TrackOptions MakeOptions(int degree)
  {TrackOptions o;o.base.elements=8;o.base.order=degree;o.initial="entropy";o.left=0.;o.right=1.;o.center=.5;o.closure_space="l2";o.quad_extra=2*degree+20;return o;}
  State Fixture(const std::string& kind,double stage_vn)const
  {
    State state(n*4);
    for(int i=0;i<n;++i)
    {
      const double x=physics.basis.nodes[i],pi=std::acos(-1.);
      double rho=1.2,vn=stage_vn,vt=.7,pressure=1.;
      if(kind=="smooth")
      {rho=1.2+.2*x+.1*std::sin(2*pi*x);vn+=.3*(1.-x)*std::cos(3*pi*x);vt+=.15*std::sin(2*pi*x);pressure+=.1*std::sin(pi*x);}
      if(kind=="steep")
      {const double front=.5*(1.+std::tanh((x-.73)/.08));rho+=.6*front;vn+=.4*(1.-x)*front;vt+=.2*front;pressure+=.2*front;}
      const Four value=WallPrimitive(rho,vn,vt,pressure,{1.,0.});for(int c=0;c<4;++c)state[i*4+c]=value[c];
    }
    return state;
  }
  Four At(const State& u,const State& shape)const
  {Four out{};for(int i=0;i<n;++i)for(int c=0;c<4;++c)out[c]+=shape[i]*u[i*4+c];return out;}
};

struct WallRates
{double mass=0.,energy=0.,kinetic=0.,internal=0.,identity=0.,represented_vn=0.,represented_vt=0.,represented_half_speed2=0.;};

WallRates BoundaryRates(const WallElement& pr,const State& state,const Four& flux,const WallRule& rule)
{
  // Outward right wall contributes -e_p*Fhat to the weak residual.
  // Boundary-only rates deliberately exclude volume and other-face terms.
  State rate(pr.n*4,0.);for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)rate[i*4+c]=-pr.mass_inverse(i,pr.p)*flux[c];
  WallRates out;State vn_load(pr.n,0.),vt_load(pr.n,0.),half_load(pr.n,0.);
  for(std::size_t q=0;q<rule.w.size();++q)
  {
    const Four u=pr.At(state,rule.v[q]),du=pr.At(rate,rule.v[q]);
    if(!(u[0]>floor_value)||!(Pressure(u)>floor_value))throw std::runtime_error("wall polynomial fixture lost positivity");
    const double vn=u[1]/u[0],vt=u[2]/u[0],half=.5*(vn*vn+vt*vt),w=rule.w[q];
    out.mass+=w*du[0];out.energy+=w*du[3];out.kinetic+=w*(vn*du[1]+vt*du[2]-half*du[0]);
    for(int i=0;i<pr.n;++i){vn_load[i]+=w*rule.v[q][i]*vn;vt_load[i]+=w*rule.v[q][i]*vt;half_load[i]+=w*rule.v[q][i]*half;}
  }
  for(int j=0;j<pr.n;++j)
  {out.represented_vn+=pr.mass_inverse(pr.p,j)*vn_load[j];out.represented_vt+=pr.mass_inverse(pr.p,j)*vt_load[j];out.represented_half_speed2+=pr.mass_inverse(pr.p,j)*half_load[j];}
  out.internal=out.energy-out.kinetic;
  const double kinetic_identity=-out.represented_vn*flux[1]-out.represented_vt*flux[2]+out.represented_half_speed2*flux[0];
  out.identity=out.kinetic-kinetic_identity;
  CheckNear(out.identity,0.,2.e-10,"discrete wall kinetic work identity failed");
  CheckNear(out.mass,-flux[0],2.e-10,"discrete wall mass budget failed");
  CheckNear(out.energy,-flux[3],2.e-10,"discrete wall energy budget failed");
  return out;
}
void WriteRates(std::ostream& out,const WallRates& r)
{out<<"{\"mass\":"<<r.mass<<",\"energy\":"<<r.energy<<",\"kinetic\":"<<r.kinetic<<",\"internal\":"<<r.internal<<",\"kinetic_identity_defect\":"<<r.identity<<",\"represented_normal_velocity_at_wall\":"<<r.represented_vn<<",\"represented_tangential_velocity_at_wall\":"<<r.represented_vt<<'}';}
void WriteFunctional(std::ostream& out,const WallElement& pr,const State& state,const WallFluxes& f,const WallRule& rule)
{
  out<<"{\"total\":";WriteRates(out,BoundaryRates(pr,state,f.total,rule));
  out<<",\"policy_euler_baseline\":";WriteRates(out,BoundaryRates(pr,state,f.baseline,rule));
  out<<",\"central_sigma_increment\":";WriteRates(out,BoundaryRates(pr,state,f.central_sigma,rule));
  out<<",\"sigma_llf_speed_increment\":";WriteRates(out,BoundaryRates(pr,state,f.sigma_speed,rule));out<<'}';
}

void PointCases(std::ostream& out)
{
  out<<'[';bool first=true;
  for(double angle:{0.,.4,1.7})for(double length:{1.,2.3})for(double vn:{-.12,0.,.12})for(double vt:{0.,.7})for(double sigma:{-.2,0.,.2})
  {
    const Two unit{std::cos(angle),std::sin(angle)},normal{length*unit[0],length*unit[1]};
    const Four state=WallPrimitive(1.2,vn,vt,1.,unit);
    for(const auto& policy:wall_policies)
    {
      const auto f=ProductionWallFlux(state,normal,sigma,policy);CheckFlux(state,normal,sigma,policy,f);
      if(!first)out<<",\n";first=false;
      out<<"{\"policy\":\""<<policy<<"\",\"normal_angle\":"<<angle<<",\"normal_length\":"<<length<<",\"normal_velocity\":"<<vn<<",\"tangential_velocity\":"<<vt<<",\"sigma\":"<<sigma<<",\"fluxes\":";WriteFluxes(out,f);out<<'}';
    }
  }
  out<<']';
}

void SemidiscreteCases(std::ostream& out)
{
  out<<'[';bool first=true;
  for(int degree:{3,7,11})for(const std::string mass_kind:{"exact","collocated"})
  {
    WallElement pr(degree,mass_kind);
    for(const std::string fixture:{"constant","smooth","steep"})for(double vn:{-.12,0.,.12})
    {
      const State state=pr.Fixture(fixture,vn);const Four wall=pr.At(state,pr.Shape(1.,false));
      for(const auto& policy:wall_policies)
      {
        const double sigma=.2;const auto f=ProductionWallFlux(wall,{1.,0.},sigma,policy);CheckFlux(wall,{1.,0.},sigma,policy,f);
        if(!first)out<<",\n";first=false;
        out<<"{\"order\":"<<degree<<",\"mass\":\""<<mass_kind<<"\",\"fixture\":\""<<fixture<<"\",\"normal_velocity_at_wall\":"<<vn<<",\"sigma\":"<<sigma<<",\"policy\":\""<<policy<<"\",\"method_points\":"<<pr.method.w.size()<<",\"common_audit_points\":"<<pr.audit.w.size()<<",\"fluxes\":";WriteFluxes(out,f);
        out<<",\"method_functional\":";WriteFunctional(out,pr,state,f,pr.method);
        out<<",\"common_audit_functional\":";WriteFunctional(out,pr,state,f,pr.audit);out<<'}';
      }
    }
  }
  out<<']';
}

void CurvedManufactured(std::ostream& out)
{
  // A steady rigid rotation v=Omega*(-y,x) in a disk has v.n=0,
  // n.(v.grad)v=-Omega^2 R. Momentum then requires d_n(p+sigma)=rho Omega^2 R.
  // Constant rho and sigma=-2*alpha*rho*Omega^2 also satisfy the shared
  // instantaneous closure; p(r)=1+rho*Omega^2*(r^2-R^2)/2 supplies curvature.
  // Homogeneous sigma Neumann IS compatible with this example.
  const double radius=1.3,omega=.7,rho=1.2,alpha=.0004,sigma=-2.*alpha*rho*omega*omega;
  double max_energy=0.,max_mass=0.,max_traction_error=0.,max_tangent_error=0.;
  double source=0.,maximum_closure_defect=0.;
  for(int i=0;i<32;++i)
  {
    const double angle=2.*std::acos(-1.)*(i+.31)/32.;const Two normal{std::cos(angle),std::sin(angle)};
    const Four state=WallPrimitive(rho,0.,omega*radius,1.,normal);
    double gradient[8]={0.,0.,0.,-rho*omega,rho*omega,0.,
      gamma_euler/(gamma_euler-1.)*rho*omega*omega*radius*normal[0],
      gamma_euler/(gamma_euler-1.)*rho*omega*omega*radius*normal[1]};
    source=igr::SourceFromConservativeStateGradient(2,state.data(),1,gradient,2,1,alpha,floor_value);
    maximum_closure_defect=std::max(maximum_closure_defect,std::abs(sigma/rho-source));
    CheckNear(sigma/rho,source,2.e-13,"rigid-rotation constant-stress closure identity failed");
    const auto f=ProductionWallFlux(state,normal,sigma,"mirrored-copy");CheckFlux(state,normal,sigma,"mirrored-copy",f);
    max_energy=std::max(max_energy,std::abs(f.total[3]));max_mass=std::max(max_mass,std::abs(f.total[0]));
    max_traction_error=std::max(max_traction_error,std::abs(f.total[1]*normal[0]+f.total[2]*normal[1]-(1.+sigma)));
    max_tangent_error=std::max(max_tangent_error,std::abs(-f.total[1]*normal[1]+f.total[2]*normal[0]));
  }
  out<<"{\"scope\":\"Pointwise exact rigid-rotation continuum control; no mesh or 2D evolution\",\"radius\":"<<radius<<",\"angular_velocity\":"<<omega<<",\"density\":"<<rho
     <<",\"alpha\":"<<alpha<<",\"sigma\":"<<sigma<<",\"shared_igr_source\":"<<source<<",\"constant_sigma_closure_maximum_defect\":"<<maximum_closure_defect
     <<",\"homogeneous_sigma_neumann_compatible\":true"
     <<",\"normal_material_acceleration\":"<<-omega*omega*radius<<",\"required_normal_total_pressure_derivative\":"<<rho*omega*omega*radius
     <<",\"homogeneous_total_pressure_neumann_momentum_defect\":"<<rho*omega*omega*radius
     <<",\"maximum_copied_sigma_energy_flux\":"<<max_energy<<",\"maximum_mass_flux\":"<<max_mass<<",\"maximum_normal_traction_error\":"<<max_traction_error<<",\"maximum_tangential_traction\":"<<max_tangent_error<<'}';
}

void HDGTraceAlgebra(std::ostream& out)
{
  // Read-only algebra of BorrowedEulerFaceElement's physical trace row:
  // Uhat=Exterior(Uminus), local flux F(Uhat,sigmahat).n+beta*(Uminus-Uhat).
  // This evaluates the shared physical flux, not the private CUDA residual.
  out<<'[';bool first=true;mfem::EulerFlux euler(2,gamma_euler);
  for(double vn:{-.12,0.,.12})
  {
    const double sigma=.2,beta=2.;const Two normal{1.,0.};const Four u=WallPrimitive(1.2,vn,.7,1.,normal);
    const auto mirrored=ProductionWallFlux(u,normal,sigma,"mirrored-copy");
    mfem::Vector trace(const_cast<double*>(mirrored.exterior.data()),4),n(const_cast<double*>(normal.data()),2),flux(4);mfem::FaceElementTransformations unused;
    euler.ComputeFluxDotN(trace,n,unused,flux);
    igr::AddNormalSigmaFlux(2,trace.GetData(),1,normal.data(),1,sigma,floor_value,flux.GetData(),1);
    Four local{};for(int c=0;c<4;++c)local[c]=flux[c]+beta*(u[c]-trace[c]);
    CheckNear(local[0],-u[0]*vn,2.e-13,"HDG mirrored trace mass algebra failed");
    CheckNear(local[3],-(u[3]+Pressure(u)+sigma)*vn,2.e-13,"HDG mirrored trace energy algebra failed");
    // Alternative trace target: remove only normal momentum. Preserve rho,
    // tangential momentum, AND total E; do not subtract normal kinetic energy.
    Four zero_normal=u;const double normal_momentum=u[1]*normal[0]+u[2]*normal[1];
    for(int d=0;d<2;++d)zero_normal[d+1]-=normal_momentum*normal[d];
    mfem::Vector zero_trace(zero_normal.data(),4);euler.ComputeFluxDotN(zero_trace,n,unused,flux);
    igr::AddNormalSigmaFlux(2,zero_normal.data(),1,normal.data(),1,sigma,floor_value,flux.GetData(),1);
    Four zero_local{},zero_same_speed{};
    for(int c=0;c<4;++c)
    {zero_local[c]=flux[c]+beta*(u[c]-zero_normal[c]);zero_same_speed[c]=flux[c]+mirrored.speed*(u[c]-zero_normal[c]);}
    const double trace_pressure=Pressure(zero_normal),expected_pressure=Pressure(u)+.5*(gamma_euler-1.)*u[0]*vn*vn;
    CheckNear(trace_pressure,expected_pressure,2.e-13,"zero-normal HDG trace pressure failed");
    CheckNear(zero_normal[0],u[0],0.,"zero-normal trace changed density");
    CheckNear(zero_normal[3],u[3],0.,"zero-normal trace changed total energy");
    CheckNear(zero_local[0],0.,2.e-13,"zero-normal HDG trace leaked mass");
    CheckNear(zero_local[3],0.,2.e-13,"zero-normal HDG trace leaked energy");
    CheckNear(zero_local[2],0.,2.e-13,"zero-normal HDG trace has tangential flux");
    CheckNear(zero_local[1],trace_pressure+sigma+beta*u[0]*vn,2.e-13,"zero-normal HDG trace traction failed");
    const double same_speed_difference=zero_same_speed[1]-mirrored.total[1];
    CheckNear(same_speed_difference,.5*(gamma_euler-3.)*u[0]*vn*vn,2.e-13,"matched-stabilization wall traction difference failed");
    if(!first)out<<',';first=false;out<<"{\"normal_velocity\":"<<vn<<",\"sigma\":"<<sigma<<",\"stabilization\":"<<beta<<",\"trace_equals_reflected_state_local_flux\":";WriteFour(out,local);
    out<<",\"dg_reflected_rusanov_flux\":";WriteFour(out,mirrored.total);
    out<<",\"zero_normal_momentum_trace_state\":";WriteFour(out,zero_normal);
    out<<",\"zero_normal_trace_pressure\":"<<trace_pressure<<",\"zero_normal_trace_local_flux\":";WriteFour(out,zero_local);
    out<<",\"zero_normal_trace_with_fixed_dg_stabilization_flux\":";WriteFour(out,zero_same_speed);
    out<<",\"fixed_dg_stabilization\":"<<mirrored.speed<<",\"same_stabilization_normal_traction_difference\":"<<same_speed_difference<<'}';
  }
  out<<']';
}

void RunWallWork(std::ostream& out,const std::string& backend)
{
  const auto runtime=mfem::dgsem::GetDGSEMDeviceRuntimeInfo();
  if(backend=="cuda"&&(!runtime.configured||!runtime.enabled||!runtime.cuda))throw std::runtime_error("requested CUDA wall audit did not select the CUDA production runtime");
  if(backend=="cpu"&&runtime.cuda)throw std::runtime_error("CPU wall audit unexpectedly selected CUDA");
  out<<std::setprecision(17)<<"{\"scope\":\"Stationary slip-wall point flux and boundary-only semidiscrete work; no closure boundary condition, time integration, stability, or performance claim\",\"requested_backend\":\""<<backend
     <<"\",\"production_face_runtime_cuda\":"<<(runtime.cuda?"true":"false")<<",\"production_face_device_id\":"<<runtime.device_id
     <<",\"execution_scope\":\"Only production gather and face kernels use selected backend; basis, functionals and HDG formula calculations run on host\",\"flux_order\":[\"mass\",\"momentum_x\",\"momentum_y\",\"total_energy\"],\"speed_policy\":\"actual production frozen-sigma normal spectral Rusanov speed\",\"point_records\":";
  PointCases(out);out<<",\"semidiscrete_records\":";SemidiscreteCases(out);out<<",\"curved_manufactured\":";CurvedManufactured(out);
  out<<",\"hdg_trace_algebra_scope\":\"Read-only private CUDA formula audit using shared physical flux; private CUDA kernel and coupled solve not executed\",\"hdg_trace_algebra\":";HDGTraceAlgebra(out);out<<"}\n";
}
}

int main(int argc,char** argv)
{
  try
  {
    mfem::Mpi::Init(argc,argv);bool self_test=false;std::string output,backend="cpu";
    for(int i=1;i<argc;++i)
    {const std::string flag=argv[i];if(flag=="--self-test")self_test=true;
      else if(flag=="--output"&&i+1<argc)output=argv[++i];
      else if(flag=="--device"&&i+1<argc)backend=argv[++i];
      else throw std::runtime_error("unknown/missing wall-work option: "+flag);}
    if(backend!="cpu"&&backend!="cuda")throw std::runtime_error("wall-work --device must be cpu or cuda");
    mfem::Device device(backend.c_str());
    if(self_test){std::ostringstream sink;RunWallWork(sink,backend);std::cout<<"Wall flux and discrete work identities passed, production "<<backend<<" gather/flux, p3/p7/p11, exact/collocated, rotated/scaled normals; HDG formulas checked on host\n";}
    else if(output.empty())RunWallWork(std::cout,backend);
    else{std::ofstream out(output);if(!out)throw std::runtime_error("cannot open wall-work output");RunWallWork(out,backend);}
    return 0;
  }
  catch(const std::exception& error){std::cerr<<"DG/IGR wall-work probe failed: "<<error.what()<<'\n';return 2;}
}

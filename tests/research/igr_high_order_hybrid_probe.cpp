// Test-only high-order Euler DG with localized conservative LGL subcell blending.
// Reuses production Euler/IGR algebra and the existing reduced current closure.
#define main igr_high_order_existing_spacetime_main
#include "screened_spacetime_probe.cpp"
#undef main
#include "igr_high_order_front.hpp"
#define DG_IGR_HIGH_ORDER_CONDENSATION_LIBRARY_ONLY
#include "igr_high_order_condensation_probe.cpp"
#undef DG_IGR_HIGH_ORDER_CONDENSATION_LIBRARY_ONLY
#include <fstream>
#include <sstream>

namespace
{
struct HybridOptions
{
  Options base;
  std::string method="blend", output, quadrature="collocated", closure_solver="dense";
  int aggregate=1;
  double width=.012,mach=5.,speed=.4,phase=.13,threshold=1.e-4;
  bool self_test=false,positivity=true;
};
struct HybridWork
{
  Work base;
  CondWork condensed;
  std::uint64_t rhs=0, sensor_visits=0, pure_high=0, blended=0;
  std::uint64_t positivity_visits=0, positivity_limited=0, sample_checks=0;
  double sum_blend=0.,maximum_sensor=0.,minimum_theta=1.;
  double closure_seconds=0.,transport_seconds=0.,limiter_seconds=0.;
  double maximum_closure_residual=0.,maximum_conservation_residual=0.;
};
struct HybridGeometry
{
  mfem::DenseMatrix averages, inverse_averages, modal;
  std::vector<State> positivity_shapes, qshapes, qderivatives;
  State qweights;
  explicit HybridGeometry(const Problem& pr):averages(pr.n),inverse_averages(pr.n),modal(pr.n)
  {
    averages=0.;modal=0.;
    const auto& volume=mfem::IntRules.Get(mfem::Geometry::SEGMENT,2*pr.p+5);
    for(int k=0;k<volume.GetNPoints();++k)
    {
      const auto& ip=volume.IntPoint(k);qshapes.push_back(Shape(pr,ip.x));qweights.push_back(ip.weight);
      State derivative(pr.n,0.);
      for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)if(j!=i)
      {double term=1./(pr.basis.nodes[i]-pr.basis.nodes[j]);
        for(int l=0;l<pr.n;++l)if(l!=i&&l!=j)term*=(ip.x-pr.basis.nodes[l])/(pr.basis.nodes[i]-pr.basis.nodes[l]);
        derivative[i]+=term;}
      qderivatives.push_back(derivative);
    }
    const auto& rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,2*pr.p+3);
    for(int s=0;s<pr.n;++s)for(int k=0;k<rule.GetNPoints();++k)
    {const auto& ip=rule.IntPoint(k);const State shape=Shape(pr,(s+ip.x)/pr.n);
      for(int j=0;j<pr.n;++j)averages(s,j)+=ip.weight*shape[j];}
    mfem::DenseMatrixInverse inv(averages);inv.GetInverseMatrix(inverse_averages);
    for(int i=0;i<pr.n;++i)
    {
      const double x=2*pr.basis.nodes[i]-1.;double prev=1.,cur=x;
      modal(0,i)=pr.basis.weights[i];
      if(pr.p)modal(1,i)=3*pr.basis.weights[i]*x;
      for(int k=2;k<pr.n;++k)
      {const double next=((2*k-1)*x*cur-(k-1)*prev)/k;
        modal(k,i)=(2*k+1)*pr.basis.weights[i]*next;prev=cur;cur=next;}
    }
    // The highest LGL mode has a different discrete norm.
    for(int i=0;i<pr.n;++i)modal(pr.p,i)*=double(pr.p)/(2*pr.p+1);
    const auto& positive=mfem::IntRules.Get(mfem::Geometry::SEGMENT,4*pr.p+3);
    for(int k=0;k<positive.GetNPoints();++k)positivity_shapes.push_back(Shape(pr,positive.IntPoint(k).x));
    for(double x:pr.basis.nodes)positivity_shapes.push_back(Shape(pr,x));
  }
  static State Shape(const Problem& pr,double x)
  {State shape(pr.n,1.);for(int j=0;j<pr.n;++j)for(int k=0;k<pr.n;++k)if(j!=k)
      shape[j]*=(x-pr.basis.nodes[k])/(pr.basis.nodes[j]-pr.basis.nodes[k]);return shape;}
  State Map(const Problem& pr,const State& u,const mfem::DenseMatrix& matrix) const
  {
    State result(u.size(),0.);
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)for(int c=0;c<4;++c)
      result[pr.Index(e,i,c)]+=matrix(i,j)*u[pr.Index(e,j,c)];
    return result;
  }
};
State HybridInitial(const Problem& pr,const HybridOptions& options,double time=0.)
{
  if(pr.o.initial_case=="smooth"||pr.o.initial_case=="compression")return pr.Initial();
  State u(pr.ne*pr.n*4);auto shock=MakeHighOrderPeriodicFront(options.width,options.mach,options.speed,options.phase);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {
    const double x=pr.h*(e+pr.basis.nodes[i]);mfem::Vector v(4);
    if(pr.o.initial_case=="front")shock(x,v);
    else
    {const double rho=1.+.2*std::sin(2*std::acos(-1.)*(x-.2*time));
      v[0]=rho;v[1]=.2*rho;v[2]=0.;v[3]=1./(gamma_euler-1.)+.02*rho;}
    for(int c=0;c<4;++c)u[pr.Index(e,i,c)]=v[c];
  }
  return u;
}
bool HybridAdmitted(const std::array<double,4>& u)
{
  for(double x:u)if(!std::isfinite(x))return false;
  if(!(u[0]>1.e-10))return false;
  const double p=(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
  return p>1.e-10;
}
void HybridLimit(const Problem& pr,const HybridGeometry& geometry,State& u,
                 const HybridOptions& o,HybridWork& work)
{
  const auto start=Clock::now();
  for(int e=0;e<pr.ne;++e)
  {
    ++work.positivity_visits;std::array<double,4> mean{};
    for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)mean[c]+=pr.basis.weights[i]*u[pr.Index(e,i,c)];
    if(!HybridAdmitted(mean))throw std::runtime_error("element mean admission failed");
    double theta=1.;
    // A denser Gauss rule is not a superset of the operator's Gauss rule.
    // Admit the actual nonlinear evaluation points as well as the independent
    // interior audit rule, so reconstruction and operator admission agree.
    for(int rule=0;rule<(o.quadrature=="collocated"?1:2);++rule)
    for(const State& shape:rule==0?geometry.positivity_shapes:geometry.qshapes)
    {
      ++work.sample_checks;std::array<double,4> point{};
      for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)point[c]+=shape[i]*u[pr.Index(e,i,c)];
      if(HybridAdmitted(point))continue;
      if(!o.positivity)throw std::runtime_error("interior polynomial sample admission failed");
      double lo=0.,hi=1.;
      for(int k=0;k<48;++k)
      {const double mid=.5*(lo+hi);std::array<double,4> trial{};
        for(int c=0;c<4;++c)trial[c]=mean[c]+mid*(point[c]-mean[c]);
        if(HybridAdmitted(trial))lo=mid;else hi=mid;}
      theta=std::min(theta,lo*(1.-1.e-12));
    }
    if(theta<1.)
    {++work.positivity_limited;work.minimum_theta=std::min(work.minimum_theta,theta);
      for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)
        u[pr.Index(e,i,c)]=mean[c]+theta*(u[pr.Index(e,i,c)]-mean[c]);}
  }
  work.limiter_seconds+=Wall(start);
}
// Independent local assembly for the exact condensed variant; the original
// global assembly below remains the test oracle. Current density/source and
// quadrature are identical, including the nonlinear conservative quotient.
void HybridElementClosure(const Problem& pr,const HybridGeometry& geometry,const HybridOptions& o,
                          const State& u,int e,mfem::DenseMatrix& a,State& b)
{
  a.SetSize(pr.n);a=0.;b.assign(pr.n,0.);
  const int nq=o.quadrature=="collocated"?pr.n:geometry.qweights.size();
  for(int k=0;k<nq;++k)
  {
    State shape,derivative;double weight=0.;
    if(o.quadrature=="collocated")
    {
      shape.assign(pr.n,0.);shape[k]=1.;derivative.resize(pr.n);weight=pr.basis.weights[k];
      for(int i=0;i<pr.n;++i)derivative[i]=pr.basis.D(k,i);
    }
    else {shape=geometry.qshapes[k];derivative=geometry.qderivatives[k];weight=geometry.qweights[k];}
    double point[4]={},gradient[8]={};
    for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)
    {point[c]+=shape[i]*u[pr.Index(e,i,c)];gradient[2*c]+=derivative[i]*u[pr.Index(e,i,c)]/pr.h;}
    pr.CheckState(point);const double invrho=igr::SafeInverseDensity(point[0],floor_value);
    const double source=igr::SourceFromConservativeStateGradient(2,point,1,gradient,2,1,pr.o.alpha,floor_value);
    if(!std::isfinite(source))throw std::runtime_error("element current source failed");
    for(int i=0;i<pr.n;++i)
    {
      b[i]+=pr.h*weight*shape[i]*source;
      for(int j=0;j<pr.n;++j)a(i,j)+=weight*invrho*(pr.h*shape[i]*shape[j]+pr.o.alpha*derivative[i]*derivative[j]/pr.h);
    }
  }
}
State HybridClosure(const Problem& pr,const HybridGeometry& geometry,const HybridOptions& o,const State& u,HybridWork& work)
{
  const auto start=Clock::now();mfem::DenseMatrix a;State b;
  if(o.closure_solver!="dense")
  {
    const auto assemble=[&](const State& current,int e,mfem::DenseMatrix& matrix,State& rhs)
      {HybridElementClosure(pr,geometry,o,current,e,matrix,rhs);};
    auto result=SolveCondensedFromElements(pr,u,o.aggregate,o.closure_solver=="condensed-band"?"band":"dense",work.condensed,assemble);
    work.maximum_closure_residual=std::max(work.maximum_closure_residual,work.condensed.residual);
    ++work.base.global_factors;++work.base.certificate_calls;
    work.closure_seconds+=Wall(start);return std::move(result.sigma);
  }
  if(o.quadrature=="collocated")pr.Assemble(u,pr.ne,true,a,b,work.base);
  else
  {
    a.SetSize(pr.ne*pr.p);a=0.;b.assign(pr.ne*pr.p,0.);work.base.coefficient_elements+=pr.ne;
    for(int e=0;e<pr.ne;++e)for(std::size_t k=0;k<geometry.qweights.size();++k)
    {
      double point[4]={},gradient[8]={};const auto& shape=geometry.qshapes[k];const auto& derivative=geometry.qderivatives[k];
      for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)
      {point[c]+=shape[i]*u[pr.Index(e,i,c)];gradient[2*c]+=derivative[i]*u[pr.Index(e,i,c)]/pr.h;}
      pr.CheckState(point);const double invrho=igr::SafeInverseDensity(point[0],floor_value);
      const double source=igr::SourceFromConservativeStateGradient(2,point,1,gradient,2,1,pr.o.alpha,floor_value);
      if(!std::isfinite(source))throw std::runtime_error("overintegrated closure source failed");
      for(int i=0;i<pr.n;++i)
      {b[pr.SigmaIndex(e,i)]+=pr.h*geometry.qweights[k]*shape[i]*source;
        for(int j=0;j<pr.n;++j)
          a(pr.SigmaIndex(e,i),pr.SigmaIndex(e,j))+=geometry.qweights[k]*invrho*
            (pr.h*shape[i]*shape[j]+pr.o.alpha*derivative[i]*derivative[j]/pr.h);}
    }
  }
  State sigma(b.size());
  mfem::DenseMatrixInverse inv(a,true);mfem::Vector rhs(b.data(),b.size()),out(sigma.data(),sigma.size());inv.Mult(rhs,out);
  mfem::Vector defect(b.size());a.Mult(out,defect);defect-=rhs;
  const double residual=defect.Norml2()/std::max(rhs.Norml2(),1.e-20);
  // A vanishing entropy-wave closure is checked with an absolute floor.
  const double certification=defect.Norml2()/std::max(rhs.Norml2(),1.e-12);
  work.maximum_closure_residual=std::max(work.maximum_closure_residual,certification);
  if(!std::isfinite(residual)||certification>1.e-9)throw std::runtime_error("current closure residual failed");
  ++work.base.global_factors;++work.base.certificate_calls;
  const double n=b.size();work.base.factor_flops+=n*n*n/3.;work.base.solve_flops+=2*n*n;
  work.closure_seconds+=Wall(start);return sigma;
}
double HybridSensor(const Problem& pr,const HybridGeometry& geometry,const State& u,int e)
{
  double sensor=0.;
  for(int variable=0;variable<2;++variable)
  {
    State values(pr.n),modes(pr.n,0.);
    for(int i=0;i<pr.n;++i)
    {const double* q=&u[pr.Index(e,i,0)];values[i]=variable==0?q[0]:
      (gamma_euler-1.)*(q[3]-.5*(q[1]*q[1]+q[2]*q[2])/q[0]);}
    for(int k=0;k<pr.n;++k)for(int i=0;i<pr.n;++i)modes[k]+=geometry.modal(k,i)*values[i];
    double all=0.,tail=0.;
    for(int k=0;k<pr.n;++k)
    {const double energy=modes[k]*modes[k]/(2*k+1);all+=energy;if(k>=pr.p-1)tail+=energy;}
    sensor=std::max(sensor,tail/std::max(all,1.e-30));
  }
  return sensor;
}
State HybridRhs(const Problem& pr,const HybridGeometry& geometry,const HybridOptions& o,
                const State& u,const State& sigma,HybridWork& work)
{
  const auto start=Clock::now();++work.rhs;
  State high;
  if(o.quadrature!="all")high=pr.Residual(u,sigma,pr.ne,true,0,nullptr,nullptr,work.base);
  else
  {
    high.assign(u.size(),0.);work.base.residual_elements+=pr.ne;
    for(int e=0;e<pr.ne;++e)for(std::size_t k=0;k<geometry.qweights.size();++k)
    {
      double point[4]={},sq=0.,speed=0.;
      for(int i=0;i<pr.n;++i)
      {sq+=geometry.qshapes[k][i]*sigma[pr.SigmaIndex(e,i)];
        for(int c=0;c<4;++c)point[c]+=geometry.qshapes[k][i]*u[pr.Index(e,i,c)];}
      const auto f=pr.Flux(point,sq,speed);
      for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)
        high[pr.Index(e,i,c)]+=geometry.qweights[k]*geometry.qderivatives[k][i]*f[c]/(pr.h*pr.basis.weights[i]);
    }
    for(int face=0;face<pr.ne;++face)
    {
      const double* left=&u[pr.Index(face-1,pr.p,0)],*right=&u[pr.Index(face,0,0)];double al=0.,ar=0.;
      const auto fl=pr.Flux(left,sigma[pr.SigmaIndex(face-1,pr.p)],al),fr=pr.Flux(right,sigma[pr.SigmaIndex(face,0)],ar);
      for(int c=0;c<4;++c)
      {const double star=.5*(fl[c]+fr[c]-pr.FaceSpeed(al,ar)*(right[c]-left[c]));
        high[pr.Index(face-1,pr.p,c)]-=star/(pr.h*pr.basis.weights[pr.p]);
        high[pr.Index(face,0,c)]+=star/(pr.h*pr.basis.weights[0]);}
      ++work.base.face_fluxes;
    }
  }
  State alpha(pr.ne,0.);
  for(int e=0;e<pr.ne;++e)
  {
    ++work.sensor_visits;
    if(o.method=="blend")
    {const double sensor=HybridSensor(pr,geometry,u,e);work.maximum_sensor=std::max(work.maximum_sensor,sensor);
      alpha[e]=std::max(0.,std::min(1.,.5*(std::log10(std::max(sensor,1.e-30)/o.threshold)+1.)));}
    if(o.method=="low")alpha[e]=1.;
    work.sum_blend+=alpha[e];if(alpha[e]>0.)++work.blended;else ++work.pure_high;
  }
  // Internal low-order fluxes use the same sigma-aware physical flux as DG.
  // Boundary fluxes exactly equal DG's common Rusanov face flux, so an
  // elementwise constant blend preserves the native DG element average.
  for(int e=0;e<pr.ne;++e)
  {
    if(alpha[e]==0.)continue;
    std::vector<std::array<double,4>> flux(pr.n+1);
    for(int face=0;face<=pr.n;++face)
    {
      const int le=face==0?e-1:e,li=face==0?pr.p:face-1;
      const int re=face==pr.n?e+1:e,ri=face==pr.n?0:face;
      const double* left=&u[pr.Index(le,li,0)],*right=&u[pr.Index(re,ri,0)];
      double al=0.,ar=0.;const auto fl=pr.Flux(left,sigma[pr.SigmaIndex(le,li)],al);
      const auto fr=pr.Flux(right,sigma[pr.SigmaIndex(re,ri)],ar);
      for(int c=0;c<4;++c)flux[face][c]=.5*(fl[c]+fr[c]-pr.FaceSpeed(al,ar)*(right[c]-left[c]));
      ++work.base.face_fluxes;
    }
    for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)
    {const double low=-(flux[i+1][c]-flux[i][c])/(pr.h*pr.basis.weights[i]);
      double& v=high[pr.Index(e,i,c)];v=(1.-alpha[e])*v+alpha[e]*low;}
  }
  const auto sums=pr.Integrals(high);
  for(double v:sums)work.maximum_conservation_residual=std::max(work.maximum_conservation_residual,std::abs(v));
  work.transport_seconds+=Wall(start);return high;
}
struct HybridResult
{
  State u,sigma;HybridWork work;double time=0.,seconds=0.,setup=0.;
  bool complete=false;std::string failure;
};
HybridResult HybridRun(const Problem& pr,const HybridGeometry& geometry,const HybridOptions& o)
{
  HybridResult r;const auto start=Clock::now();r.u=HybridInitial(pr,o);
  try
  {
    HybridLimit(pr,geometry,r.u,o,r.work);r.sigma=HybridClosure(pr,geometry,o,r.u,r.work);
    while(r.time<pr.o.final_time-1.e-14)
    {
      const double dt=std::min(pr.o.dt,pr.o.final_time-r.time);++r.work.base.attempted_steps;
      State current=r.u,stage=r.u;
      for(int k=0;k<3;++k)
      {
        const State sigma=k==0?r.sigma:HybridClosure(pr,geometry,o,stage,r.work);
        const State residual=HybridRhs(pr,geometry,o,stage,sigma,r.work);
        if(k==0)for(std::size_t i=0;i<stage.size();++i)stage[i]=current[i]+dt*residual[i];
        if(k==1)for(std::size_t i=0;i<stage.size();++i)stage[i]=.75*current[i]+.25*(stage[i]+dt*residual[i]);
        if(k==2)for(std::size_t i=0;i<stage.size();++i)stage[i]=current[i]/3.+(2./3.)*(stage[i]+dt*residual[i]);
        HybridLimit(pr,geometry,stage,o,r.work);
      }
      State sigma=HybridClosure(pr,geometry,o,stage,r.work);
      for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
      {double speed=0.;pr.Flux(&stage[pr.Index(e,i,0)],sigma[pr.SigmaIndex(e,i)],speed);}
      r.u.swap(stage);r.sigma.swap(sigma);r.time+=dt;++r.work.base.accepted_steps;
    }
    r.complete=true;
  }
  catch(const std::exception& error)
  {r.failure=error.what();r.work.base.rejected_steps=r.work.base.attempted_steps-r.work.base.accepted_steps;}
  r.seconds=Wall(start);return r;
}
std::array<double,3> HybridCoordinateTest(const Problem& pr,const HybridGeometry& geometry)
{
  State u=pr.Initial();Work work;const State sigma=pr.ExactClosure(u,work);
  const State rhs=pr.Residual(u,sigma,pr.ne,true,0,nullptr,nullptr,work);
  const State mapped=geometry.Map(pr,u,geometry.averages);
  const State recovered=geometry.Map(pr,mapped,geometry.inverse_averages);
  const State transformed=geometry.Map(pr,rhs,geometry.averages);
  const State recovered_rhs=geometry.Map(pr,transformed,geometry.inverse_averages);
  State difference=u,rd=rhs;for(std::size_t i=0;i<u.size();++i){difference[i]-=recovered[i];rd[i]-=recovered_rhs[i];}
  double mean_error=0.;
  for(int e=0;e<pr.ne;++e)for(int c=0;c<4;++c)
  {double a=0.,b=0.;for(int i=0;i<pr.n;++i){a+=pr.basis.weights[i]*u[pr.Index(e,i,c)];b+=mapped[pr.Index(e,i,c)]/pr.n;}
    mean_error=std::max(mean_error,std::abs(a-b));}
  return {{Norm(difference)/Norm(u),Norm(rd)/Norm(rhs),mean_error}};
}
void HybridPrint(const Problem& pr,const HybridOptions& o,const HybridResult& r,
                 const std::array<double,3>& coordinate,std::ostream& out)
{
  const auto initial=pr.Integrals(HybridInitial(pr,o)),final=pr.Integrals(r.u);
  double conservation=0.,rho=1.e300,pressure=1.e300;
  for(int c=0;c<4;++c)conservation=std::max(conservation,std::abs(initial[c]-final[c]));
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {const double* u=&r.u[pr.Index(e,i,0)];rho=std::min(rho,u[0]);pressure=std::min(pressure,(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]));}
  const auto& w=r.work;
  out<<std::setprecision(17)<<"{\"method\":\""<<o.method<<"\",\"case\":\""<<pr.o.initial_case
    <<"\",\"closure\":\"current-periodic-H1\",\"transport_mass\":\"LGL-lumped\",\"integrator\":\"SSPRK3\",\"elements\":"<<pr.ne<<",\"order\":"<<pr.p
    <<",\"alpha\":"<<pr.o.alpha<<",\"dt\":"<<pr.o.dt<<",\"horizon\":"<<pr.o.final_time
    <<",\"width\":"<<o.width<<",\"mach\":"<<o.mach<<",\"speed\":"<<o.speed<<",\"phase\":"<<o.phase
    <<",\"closure_solver\":\""<<o.closure_solver<<"\",\"aggregate\":"<<o.aggregate
    <<",\"quadrature\":\""<<o.quadrature<<"\",\"sensor_threshold\":"<<o.threshold<<",\"complete\":"<<(r.complete?"true":"false")<<",\"time\":"<<r.time
    <<",\"failure\":\""<<r.failure<<"\",\"setup_seconds\":"<<r.setup<<",\"run_seconds\":"<<r.seconds
    <<",\"maximum_conservative_drift\":"<<conservation<<",\"minimum_nodal_density\":"<<rho<<",\"minimum_nodal_pressure\":"<<pressure
    <<",\"coordinate_state_relative_error\":"<<coordinate[0]<<",\"coordinate_rhs_relative_error\":"<<coordinate[1]<<",\"coordinate_mean_error\":"<<coordinate[2]
    <<",\"entropy_exact_relative_nodal_error\":"<<(pr.o.initial_case=="entropy"?pr.Error(r.u,HybridInitial(pr,o,r.time)):-1.)
    <<",\"work\":{\"rhs_calls\":"<<w.rhs<<",\"sensor_visits\":"<<w.sensor_visits
    <<",\"pure_high_visits\":"<<w.pure_high<<",\"blended_visits\":"<<w.blended<<",\"sum_blend\":"<<w.sum_blend
    <<",\"maximum_sensor\":"<<w.maximum_sensor<<",\"positivity_visits\":"<<w.positivity_visits
    <<",\"positivity_limited_visits\":"<<w.positivity_limited<<",\"minimum_theta\":"<<w.minimum_theta<<",\"sample_checks\":"<<w.sample_checks
    <<",\"closure_calls\":"<<w.base.global_factors<<",\"face_fluxes\":"<<w.base.face_fluxes
    <<",\"factor_flops\":"<<w.base.factor_flops<<",\"solve_flops\":"<<w.base.solve_flops
    <<",\"accepted_steps\":"<<w.base.accepted_steps<<",\"attempted_steps\":"<<w.base.attempted_steps<<",\"rejected_steps\":"<<w.base.rejected_steps
    <<",\"closure_seconds\":"<<w.closure_seconds<<",\"transport_seconds\":"<<w.transport_seconds<<",\"limiter_seconds\":"<<w.limiter_seconds
    <<",\"maximum_current_closure_residual\":"<<w.maximum_closure_residual<<",\"maximum_residual_integral\":"<<w.maximum_conservation_residual
    <<",\"condensed_closure_calls\":"<<w.condensed.calls<<",\"condensed_assembly_seconds\":"<<w.condensed.assembly
    <<",\"condensed_factor_seconds\":"<<w.condensed.factor<<",\"condensed_response_seconds\":"<<w.condensed.responses
    <<",\"condensed_trace_seconds\":"<<w.condensed.trace<<",\"condensed_reconstruct_seconds\":"<<w.condensed.reconstruct
    <<",\"condensed_certificate_seconds\":"<<w.condensed.certificate<<",\"condensed_factor_products\":"<<w.condensed.factor_products
    <<",\"condensed_response_products\":"<<w.condensed.response_products<<",\"condensed_schur_products\":"<<w.condensed.schur_products
    <<",\"condensed_active_source_columns\":"<<w.condensed.active_source_columns
    <<"},\"nodes\":[";
  for(int i=0;i<pr.n;++i){if(i)out<<',';out<<pr.basis.nodes[i];}out<<"],\"state\":[";
  for(std::size_t i=0;i<r.u.size();++i){if(i)out<<',';out<<r.u[i];}out<<"],\"sigma\":[";
  for(std::size_t i=0;i<r.sigma.size();++i){if(i)out<<',';out<<r.sigma[i];}out<<"]}\n";
}
HybridOptions HybridParse(int argc,char** argv)
{
  HybridOptions o;o.base.elements=8;o.base.order=5;o.base.initial_case="front";
  o.base.alpha=.0004;o.base.dt=.00002;o.base.final_time=.005;
  for(int i=1;i<argc;++i)
  {
    const std::string flag=argv[i];if(flag=="--self-test"){o.self_test=true;continue;}
    if(flag=="--no-positivity"){o.positivity=false;continue;}
    if(i+1==argc)throw std::runtime_error("missing option argument");const std::string value=argv[++i];
    if(flag=="--elements")o.base.elements=std::stoi(value);
    else if(flag=="--order")o.base.order=std::stoi(value);
    else if(flag=="--case")o.base.initial_case=value;
    else if(flag=="--method")o.method=value;
    else if(flag=="--quadrature")o.quadrature=value;
    else if(flag=="--closure-solver")o.closure_solver=value;
    else if(flag=="--aggregate")o.aggregate=std::stoi(value);
    else if(flag=="--output")o.output=value;
    else if(flag=="--alpha")o.base.alpha=std::stod(value);
    else if(flag=="--dt")o.base.dt=std::stod(value);
    else if(flag=="--final-time")o.base.final_time=std::stod(value);
    else if(flag=="--threshold")o.threshold=std::stod(value);
    else if(flag=="--width")o.width=std::stod(value);
    else if(flag=="--mach")o.mach=std::stod(value);
    else if(flag=="--speed")o.speed=std::stod(value);
    else if(flag=="--phase")o.phase=std::stod(value);
    else throw std::runtime_error("unknown option "+flag);
  }
  if(o.base.elements<2||(o.base.order!=3&&o.base.order!=5&&o.base.order!=7)||
     o.base.alpha<=0||o.base.dt<=0||o.base.final_time<0||o.width<=0||o.mach<=1||o.threshold<=0||
     (o.method!="dg"&&o.method!="blend"&&o.method!="low")||
     (o.quadrature!="collocated"&&o.quadrature!="closure"&&o.quadrature!="all")||
     (o.closure_solver!="dense"&&o.closure_solver!="condensed-band"&&o.closure_solver!="condensed-dense")||
     o.aggregate<1||o.base.elements%o.aggregate||
     (o.base.initial_case!="front"&&o.base.initial_case!="entropy"&&o.base.initial_case!="smooth"&&o.base.initial_case!="compression"))
    throw std::runtime_error("invalid high-order hybrid options");
  return o;
}
void HybridSelfTest()
{
  for(int p:{3,5,7})
  {
    HybridOptions o;o.base.elements=4;o.base.order=p;o.base.alpha=.0004;o.base.dt=1.e-5;o.base.final_time=1.e-4;o.base.initial_case="entropy";
    Problem pr(o.base);HybridGeometry g(pr);const auto coordinate=HybridCoordinateTest(pr,g);
    if(coordinate[0]>1.e-12||coordinate[1]>1.e-12||coordinate[2]>1.e-12)throw std::runtime_error("coordinate equivalence failed");
    State u=HybridInitial(pr,o);HybridWork wd,wb;const State sigma=HybridClosure(pr,g,o,u,wd);
    HybridOptions dg=o;dg.method="dg";HybridOptions low=o;low.method="low";
    State rh=HybridRhs(pr,g,dg,u,sigma,wd),rl=HybridRhs(pr,g,low,u,sigma,wb);
    for(int e=0;e<pr.ne;++e)for(int c=0;c<4;++c)
    {double difference=0.;for(int i=0;i<pr.n;++i)difference+=pr.basis.weights[i]*(rh[pr.Index(e,i,c)]-rl[pr.Index(e,i,c)]);
      if(std::abs(difference)>1.e-10)throw std::runtime_error("blended residual changes element balance");}
    for(const std::string rule:{"collocated","closure","all"})
    {
      o.quadrature=rule;auto result=HybridRun(pr,g,o);
      if(!result.complete||result.work.maximum_conservation_residual>1.e-10)throw std::runtime_error("advancing high-order check failed");
      HybridWork oracle_work;const State smooth=pr.Initial();const State oracle=HybridClosure(pr,g,o,smooth,oracle_work);
      for(const std::string factor:{"condensed-band","condensed-dense"})for(int group:{1,2,4})
      {
        HybridOptions candidate=o;candidate.closure_solver=factor;candidate.aggregate=group;HybridWork candidate_work;
        State actual=HybridClosure(pr,g,candidate,smooth,candidate_work);
        for(std::size_t i=0;i<actual.size();++i)actual[i]-=oracle[i];
        if(Norm(actual)/std::max(Norm(oracle),1.e-12)>1.e-11)throw std::runtime_error("exact condensed current closure differs from global oracle");
      }
    }
    State bad=HybridInitial(pr,o);bad[pr.Index(0,0,0)]=-.1;bad[pr.Index(0,0,1)]=0.;
    const auto before=pr.Integrals(bad);HybridWork limited;HybridLimit(pr,g,bad,o,limited);const auto after=pr.Integrals(bad);
    if(limited.positivity_limited==0)throw std::runtime_error("active positivity branch was not exercised");
    for(int c=0;c<4;++c)if(std::abs(before[c]-after[c])>1.e-12)throw std::runtime_error("positivity scaling changed conservative integral");
    for(int e=0;e<pr.ne;++e)for(const auto& shape:g.qshapes)
    {
      std::array<double,4> point{};for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)point[c]+=shape[i]*bad[pr.Index(e,i,c)];
      if(!HybridAdmitted(point))throw std::runtime_error("limiter missed an actual operator quadrature point");
    }
  }
  std::cout<<"high-order coordinate equivalence, native element conservation, and current advancing closure passed\n";
}
}
#ifndef DG_IGR_HIGH_ORDER_HYBRID_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    const auto o=HybridParse(argc,argv);if(o.self_test){HybridSelfTest();return 0;}
    const auto start=Clock::now();Problem pr(o.base);HybridGeometry geometry(pr);const double setup=Wall(start);
    // Algebraic oracle is deliberately outside trajectory timing and counted
    // by whole-process timing in campaign records.
    const auto coordinate=HybridCoordinateTest(pr,geometry);auto result=HybridRun(pr,geometry,o);result.setup=setup;
    if(!o.output.empty()){std::ofstream file(o.output);if(!file)throw std::runtime_error("cannot open output");HybridPrint(pr,o,result,coordinate,file);}
    HybridPrint(pr,o,result,coordinate,std::cout);return result.complete?0:2;
  }
  catch(const std::exception& e){std::cerr<<"high-order hybrid probe: "<<e.what()<<'\n';return 1;}
}
#endif

// Test-only high-order DG macroelements with two internally C0 polynomial pieces.
// Shared Euler/IGR point algebra, LGL basis, and periodic full-energy front.
// The internal source is integrated separately on each piece: no averaging of
// the two derivatives before the nonlinear gradient square.
#define main igr_composite_existing_spacetime_main
#include "screened_spacetime_probe.cpp"
#undef main
#include "igr_high_order_front.hpp"
#include <fstream>

namespace
{
struct CompositeOptions
{
  Options base;
  std::string representation="composite",quadrature="closure",output;
  double split=.5,width=.012,mach=5.,speed=.4,phase=.13;
  bool positivity=true,self_test=false;
};
struct CompositeWork
{
  Work shared;
  std::uint64_t rhs=0,piece_source_points=0,positivity_visits=0,limited=0;
  std::uint64_t positivity_points=0,flux_audit_points=0;
  double minimum_theta=1.,closure_seconds=0.,transport_seconds=0.,limiter_seconds=0.;
  double maximum_closure_residual=0.,maximum_rhs_integral=0.;
};
bool CompositeAdmitted(const std::array<double,4>& u)
{
  for(double v:u)if(!std::isfinite(v))return false;
  return u[0]>1.e-10 && (gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0])>1.e-10;
}
struct CompositeProblem
{
  CompositeOptions o;
  Problem physics;
  int ne,p,npieces,n,ns;
  double h;
  mfem::dgsem::LGLBasis1D basis;
  State nodes,mass,breaks;
  mfem::DenseMatrix Q,D;
  std::vector<State> shapes,derivatives,positive_shapes;
  State weights;
  explicit CompositeProblem(const CompositeOptions& options):o(options),physics(o.base),
    ne(o.base.elements*(o.representation=="broken"?2:1)),p(o.base.order),
    npieces(o.representation=="composite"?2:1),n(npieces*p+1),ns(ne*(n-1)),h(1./ne),
    basis(mfem::dgsem::BuildLGLBasis1D(p)),nodes(n),mass(n,0.),Q(n),D(n)
  {
    breaks=npieces==2?State{0.,o.split,1.}:State{0.,1.};Q=0.;D=0.;
    for(int s=0;s<npieces;++s)for(int i=0;i<=p;++i)
    {
      nodes[s*p+i]=breaks[s]+(breaks[s+1]-breaks[s])*basis.nodes[i];
      mass[s*p+i]+=(breaks[s+1]-breaks[s])*basis.weights[i];
      for(int j=0;j<=p;++j)Q(s*p+i,s*p+j)+=basis.weights[i]*basis.D(i,j);
    }
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)D(i,j)=Q(i,j)/mass[i];
    const auto& quadrature=mfem::IntRules.Get(mfem::Geometry::SEGMENT,2*p+5);
    for(int k=0;k<quadrature.GetNPoints();++k)
    {const auto& ip=quadrature.IntPoint(k);shapes.push_back(Shape(ip.x,false));
      derivatives.push_back(Shape(ip.x,true));weights.push_back(ip.weight);}
    const auto& positive=mfem::IntRules.Get(mfem::Geometry::SEGMENT,4*p+3);
    for(int k=0;k<positive.GetNPoints();++k)positive_shapes.push_back(Shape(positive.IntPoint(k).x,false));
    for(double x:basis.nodes)positive_shapes.push_back(Shape(x,false));
    // A denser interlaced Gauss rule is not a superset of the actual operator
    // rule. Include every coefficient/flux point in the admission set.
    for(const State& shape:shapes)positive_shapes.push_back(shape);
  }
  int Mod(int e) const {return (e%ne+ne)%ne;}
  int Index(int e,int i,int c) const {return (Mod(e)*n+i)*4+c;}
  int SigmaIndex(int e,int i) const {return (Mod(e)*(n-1)+i)%ns;}
  State Shape(double x,bool derivative) const
  {
    State v(p+1,derivative?0.:1.);
    for(int i=0;i<=p;++i)
    {
      if(!derivative)
      {
        for(int j=0;j<=p;++j)
          if(i!=j)v[i]*=(x-basis.nodes[j])/(basis.nodes[i]-basis.nodes[j]);
      }
      else
      {
        for(int j=0;j<=p;++j)
        {
          if(i==j)continue;
          double term=1./(basis.nodes[i]-basis.nodes[j]);
          for(int k=0;k<=p;++k)
            if(k!=i&&k!=j)term*=(x-basis.nodes[k])/(basis.nodes[i]-basis.nodes[k]);
          v[i]+=term;
        }
      }
    }
    return v;
  }
  std::array<double,4> Point(const State& u,int e,int s,const State& shape) const
  {
    std::array<double,4> value{};
    for(int i=0;i<=p;++i)for(int c=0;c<4;++c)value[c]+=shape[i]*u[Index(e,s*p+i,c)];
    return value;
  }
  std::array<double,4> Integrals(const State& u) const
  {
    std::array<double,4> value{};
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)for(int c=0;c<4;++c)value[c]+=h*mass[i]*u[Index(e,i,c)];
    return value;
  }
  State Initial(double time=0.) const
  {
    auto shock=MakeHighOrderPeriodicFront(o.width,o.mach,o.speed,o.phase);State u(ne*n*4);
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
    {
      const double x=h*(e+nodes[i]);mfem::Vector v(4);
      if(o.base.initial_case=="front")shock(x,v);
      else if(o.base.initial_case=="entropy")
      {const double rho=1.+.2*std::sin(2*std::acos(-1.)*(x-.2*time));
        v[0]=rho;v[1]=.2*rho;v[2]=0.;v[3]=1./(gamma_euler-1.)+.02*rho;}
      else
      {
        const double a=2*std::acos(-1.)*x,rho=1+.04*std::cos(a),vx=.2+.045*std::sin(a),vy=.02*std::cos(a),pressure=1+.025*std::sin(a+.3);
        v[0]=rho;v[1]=rho*vx;v[2]=rho*vy;v[3]=pressure/(gamma_euler-1.)+.5*rho*(vx*vx+vy*vy);
      }
      for(int c=0;c<4;++c)u[Index(e,i,c)]=v[c];
    }
    return u;
  }
  void Limit(State& u,CompositeWork& work) const
  {
    const auto start=Clock::now();
    for(int e=0;e<ne;++e)
    {
      ++work.positivity_visits;std::array<double,4> mean{};
      for(int i=0;i<n;++i)for(int c=0;c<4;++c)mean[c]+=mass[i]*u[Index(e,i,c)];
      if(!CompositeAdmitted(mean))throw std::runtime_error("macro mean admission failed");
      double theta=1.;
      for(int s=0;s<npieces;++s)for(const State& shape:positive_shapes)
      {
        ++work.positivity_points;const auto value=Point(u,e,s,shape);
        if(CompositeAdmitted(value))continue;
        if(!o.positivity)throw std::runtime_error("piece polynomial positivity sample failed");
        double lo=0.,hi=1.;
        for(int k=0;k<48;++k)
        {double mid=.5*(lo+hi);std::array<double,4> trial{};
          for(int c=0;c<4;++c)trial[c]=mean[c]+mid*(value[c]-mean[c]);
          if(CompositeAdmitted(trial))lo=mid;else hi=mid;}
        theta=std::min(theta,lo*(1.-1.e-12));
      }
      if(theta<1.)
      {++work.limited;work.minimum_theta=std::min(work.minimum_theta,theta);
        for(int i=0;i<n;++i)for(int c=0;c<4;++c)u[Index(e,i,c)]=mean[c]+theta*(u[Index(e,i,c)]-mean[c]);}
    }
    work.limiter_seconds+=Wall(start);
  }
  State Closure(const State& u,CompositeWork& work) const
  {
    const auto start=Clock::now();mfem::DenseMatrix a(ns);a=0.;State b(ns,0.);
    for(int e=0;e<ne;++e)for(int s=0;s<npieces;++s)
    {
      const double hp=h*(breaks[s+1]-breaks[s]);
      const int nq=o.quadrature=="collocated"?p+1:weights.size();
      for(int k=0;k<nq;++k)
      {
        ++work.piece_source_points;State shape,derivative;
        if(o.quadrature=="collocated")
        {shape.assign(p+1,0.);shape[k]=1.;derivative.resize(p+1);for(int i=0;i<=p;++i)derivative[i]=basis.D(k,i);}
        else {shape=shapes[k];derivative=derivatives[k];}
        const double weight=o.quadrature=="collocated"?basis.weights[k]:weights[k];
        auto point=Point(u,e,s,shape);double gradient[8]={};
        for(int i=0;i<=p;++i)for(int c=0;c<4;++c)gradient[2*c]+=derivative[i]*u[Index(e,s*p+i,c)]/hp;
        physics.CheckState(point.data());const double invrho=igr::SafeInverseDensity(point[0],floor_value);
        const double source=igr::SourceFromConservativeStateGradient(2,point.data(),1,gradient,2,1,o.base.alpha,floor_value);
        if(!std::isfinite(source))throw std::runtime_error("piecewise current source is nonfinite");
        for(int i=0;i<=p;++i)
        {
          b[SigmaIndex(e,s*p+i)]+=hp*weight*shape[i]*source;
          for(int j=0;j<=p;++j)a(SigmaIndex(e,s*p+i),SigmaIndex(e,s*p+j))+=weight*invrho*(hp*shape[i]*shape[j]+o.base.alpha*derivative[i]*derivative[j]/hp);
        }
      }
    }
    State sigma(ns);mfem::DenseMatrixInverse inverse(a,true);mfem::Vector bv(b.data(),ns),sv(sigma.data(),ns),defect(ns);
    inverse.Mult(bv,sv);a.Mult(sv,defect);defect-=bv;
    const double certificate=defect.Norml2()/std::max(bv.Norml2(),1.e-12);
    if(!std::isfinite(certificate)||certificate>1.e-9)throw std::runtime_error("current composite closure residual failed");
    work.maximum_closure_residual=std::max(work.maximum_closure_residual,certificate);
    ++work.shared.global_factors;++work.shared.certificate_calls;work.shared.factor_flops+=double(ns)*ns*ns/3.;work.shared.solve_flops+=2.*ns*ns;
    work.closure_seconds+=Wall(start);return sigma;
  }
  State Residual(const State& u,const State& sigma,CompositeWork& work) const
  {
    const auto start=Clock::now();++work.rhs;State r(u.size(),0.);
    std::vector<std::array<double,4>> flux(ne*n);State speed(ne*n);
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)flux[e*n+i]=physics.Flux(&u[Index(e,i,0)],sigma[SigmaIndex(e,i)],speed[e*n+i]);
    if(o.quadrature=="all")
    {
      for(int e=0;e<ne;++e)for(int s=0;s<npieces;++s)for(std::size_t k=0;k<weights.size();++k)
      {
        const auto point=Point(u,e,s,shapes[k]);double sq=0.,unused=0.;
        for(int i=0;i<=p;++i)sq+=shapes[k][i]*sigma[SigmaIndex(e,s*p+i)];
        const auto f=physics.Flux(point.data(),sq,unused);
        for(int i=0;i<=p;++i)for(int c=0;c<4;++c)r[Index(e,s*p+i,c)]+=weights[k]*derivatives[k][i]*f[c]/(h*mass[s*p+i]);
      }
    }
    else for(int e=0;e<ne;++e)for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int c=0;c<4;++c)r[Index(e,i,c)]-=D(i,j)*flux[e*n+j][c]/h;
    for(int face=0;face<ne;++face)
    {
      ++work.shared.face_fluxes;const int le=Mod(face-1),re=face;
      const auto& fl=flux[le*n+n-1];const auto& fr=flux[re*n];
      const double a=physics.FaceSpeed(speed[le*n+n-1],speed[re*n]);
      for(int c=0;c<4;++c)
      {
        const double star=.5*(fl[c]+fr[c]-a*(u[Index(re,0,c)]-u[Index(le,n-1,c)]));
        r[Index(le,n-1,c)]-=(star-(o.quadrature=="all"?0.:fl[c]))/(h*mass[n-1]);
        r[Index(re,0,c)]+=(star-(o.quadrature=="all"?0.:fr[c]))/(h*mass[0]);
      }
    }
    const auto sum=Integrals(r);for(double x:sum)work.maximum_rhs_integral=std::max(work.maximum_rhs_integral,std::abs(x));
    work.transport_seconds+=Wall(start);return r;
  }
  void FluxAudit(const State& u,const State& sigma,CompositeWork& work) const
  {
    for(int e=0;e<ne;++e)for(int s=0;s<npieces;++s)for(const State& shape:positive_shapes)
    {
      ++work.flux_audit_points;const auto point=Point(u,e,s,shape);double sq=0.,speed=0.;
      for(int i=0;i<=p;++i)sq+=shape[i]*sigma[SigmaIndex(e,s*p+i)];physics.Flux(point.data(),sq,speed);
    }
  }
};
struct CompositeResult
{
  State u,sigma;CompositeWork work;double time=0.,seconds=0.;bool complete=false;std::string failure;
};
CompositeResult CompositeRun(const CompositeProblem& pr)
{
  CompositeResult r;const auto start=Clock::now();r.u=pr.Initial();
  try
  {
    pr.Limit(r.u,r.work);r.sigma=pr.Closure(r.u,r.work);pr.FluxAudit(r.u,r.sigma,r.work);
    while(r.time<pr.o.base.final_time-1.e-14)
    {
      const double dt=std::min(pr.o.base.dt,pr.o.base.final_time-r.time);++r.work.shared.attempted_steps;
      const State base=r.u;State stage=r.u;
      for(int k=0;k<3;++k)
      {
        const State sigma=k==0?r.sigma:pr.Closure(stage,r.work);const State rhs=pr.Residual(stage,sigma,r.work);
        for(std::size_t i=0;i<stage.size();++i)
        {
          if(k==0)stage[i]=base[i]+dt*rhs[i];
          if(k==1)stage[i]=.75*base[i]+.25*(stage[i]+dt*rhs[i]);
          if(k==2)stage[i]=base[i]/3.+(2./3.)*(stage[i]+dt*rhs[i]);
        }
        pr.Limit(stage,r.work);
      }
      State sigma=pr.Closure(stage,r.work);pr.FluxAudit(stage,sigma,r.work);
      r.u.swap(stage);r.sigma.swap(sigma);r.time+=dt;++r.work.shared.accepted_steps;
    }
    r.complete=true;
  }
  catch(const std::exception& e){r.failure=e.what();r.work.shared.rejected_steps=r.work.shared.attempted_steps-r.work.shared.accepted_steps;}
  r.seconds=Wall(start);return r;
}
void CompositePrint(const CompositeProblem& pr,const CompositeResult& r,double setup,std::ostream& out)
{
  const auto initial=pr.Integrals(pr.Initial()),final=pr.Integrals(r.u);double drift=0.,rho=INFINITY,pressure=INFINITY;
  for(int c=0;c<4;++c)drift=std::max(drift,std::abs(initial[c]-final[c]));
  for(int e=0;e<pr.ne;++e)for(int s=0;s<pr.npieces;++s)for(const State& shape:pr.positive_shapes)
  {const auto v=pr.Point(r.u,e,s,shape);rho=std::min(rho,v[0]);pressure=std::min(pressure,(gamma_euler-1.)*(v[3]-.5*(v[1]*v[1]+v[2]*v[2])/v[0]));}
  const State exact=pr.Initial(r.time);double error=0.,norm=0.;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)
  {const int j=pr.Index(e,i,c);error+=pr.mass[i]*std::pow(r.u[j]-exact[j],2);norm+=pr.mass[i]*exact[j]*exact[j];}
  const auto& w=r.work;
  out<<std::setprecision(17)<<"{\"schema\":\"igr-high-order-composite-v1\",\"representation\":\""<<pr.o.representation
    <<"\",\"quadrature\":\""<<pr.o.quadrature<<"\",\"case\":\""<<pr.o.base.initial_case
    <<"\",\"closure\":\"current-periodic-piecewise-H1\",\"integrator\":\"SSPRK3\",\"transport_mass\":\"composite-LGL-lumped\",\"requested_macros\":"<<pr.o.base.elements
    <<",\"dg_elements\":"<<pr.ne<<",\"pieces_per_dg_element\":"<<pr.npieces<<",\"order_per_piece\":"<<pr.p
    <<",\"polynomial_pieces\":"<<pr.ne*pr.npieces<<",\"euler_dofs_per_component\":"<<pr.ne*pr.n<<",\"closure_dofs\":"<<pr.ns
    <<",\"split\":"<<pr.o.split<<",\"alpha\":"<<pr.o.base.alpha<<",\"dt\":"<<pr.o.base.dt<<",\"horizon\":"<<pr.o.base.final_time
    <<",\"width\":"<<pr.o.width<<",\"mach\":"<<pr.o.mach<<",\"speed\":"<<pr.o.speed<<",\"phase\":"<<pr.o.phase
    <<",\"complete\":"<<(r.complete?"true":"false")<<",\"time\":"<<r.time<<",\"failure\":\""<<r.failure
    <<"\",\"setup_seconds\":"<<setup<<",\"run_seconds\":"<<r.seconds<<",\"maximum_conservative_drift\":"<<drift
    <<",\"minimum_piece_sampled_density\":"<<rho<<",\"minimum_piece_sampled_pressure\":"<<pressure
    <<",\"entropy_exact_relative_nodal_error\":"<<(pr.o.base.initial_case=="entropy"?std::sqrt(error/norm):-1.)
    <<",\"work\":{\"rhs_calls\":"<<w.rhs<<",\"closure_calls\":"<<w.shared.global_factors
    <<",\"piece_source_points\":"<<w.piece_source_points<<",\"face_fluxes\":"<<w.shared.face_fluxes
    <<",\"positivity_visits\":"<<w.positivity_visits<<",\"limited_visits\":"<<w.limited<<",\"minimum_theta\":"<<w.minimum_theta
    <<",\"positivity_sample_points\":"<<w.positivity_points<<",\"flux_audit_points\":"<<w.flux_audit_points
    <<",\"accepted_steps\":"<<w.shared.accepted_steps<<",\"attempted_steps\":"<<w.shared.attempted_steps<<",\"rejected_steps\":"<<w.shared.rejected_steps
    <<",\"closure_seconds\":"<<w.closure_seconds<<",\"transport_seconds\":"<<w.transport_seconds<<",\"limiter_seconds\":"<<w.limiter_seconds
    <<",\"factor_flops\":"<<w.shared.factor_flops<<",\"solve_flops\":"<<w.shared.solve_flops
    <<",\"maximum_current_closure_residual\":"<<w.maximum_closure_residual<<",\"maximum_rhs_integral\":"<<w.maximum_rhs_integral<<"},\"nodes\":[";
  for(int i=0;i<pr.n;++i){if(i)out<<',';out<<pr.nodes[i];}out<<"],\"mass\":[";
  for(int i=0;i<pr.n;++i){if(i)out<<',';out<<pr.mass[i];}out<<"],\"state\":[";
  for(std::size_t i=0;i<r.u.size();++i){if(i)out<<',';out<<r.u[i];}out<<"],\"sigma\":[";
  for(std::size_t i=0;i<r.sigma.size();++i){if(i)out<<',';out<<r.sigma[i];}out<<"]}\n";
}
CompositeOptions CompositeParse(int argc,char** argv)
{
  CompositeOptions o;o.base.elements=8;o.base.order=5;o.base.alpha=.0004;o.base.dt=.00002;o.base.final_time=.005;o.base.initial_case="front";
  for(int i=1;i<argc;++i)
  {
    const std::string flag=argv[i];if(flag=="--self-test"){o.self_test=true;continue;}if(flag=="--no-positivity"){o.positivity=false;continue;}
    if(i+1==argc)throw std::runtime_error("missing option value");const std::string value=argv[++i];
    if(flag=="--elements")o.base.elements=std::stoi(value);else if(flag=="--order")o.base.order=std::stoi(value);
    else if(flag=="--representation")o.representation=value;else if(flag=="--quadrature")o.quadrature=value;
    else if(flag=="--case")o.base.initial_case=value;else if(flag=="--output")o.output=value;
    else if(flag=="--alpha")o.base.alpha=std::stod(value);else if(flag=="--dt")o.base.dt=std::stod(value);
    else if(flag=="--final-time")o.base.final_time=std::stod(value);else if(flag=="--split")o.split=std::stod(value);
    else if(flag=="--width")o.width=std::stod(value);else if(flag=="--mach")o.mach=std::stod(value);
    else if(flag=="--speed")o.speed=std::stod(value);else if(flag=="--phase")o.phase=std::stod(value);
    else throw std::runtime_error("unknown option "+flag);
  }
  if(o.base.elements<2||o.base.order<2||o.base.order>11||o.base.alpha<=0||o.base.dt<=0||o.base.final_time<0||o.width<=0||o.mach<=1||o.split<=.1||o.split>=.9||
     (o.representation!="single"&&o.representation!="composite"&&o.representation!="broken")||
     (o.quadrature!="collocated"&&o.quadrature!="closure"&&o.quadrature!="all")||
     (o.base.initial_case!="front"&&o.base.initial_case!="entropy"&&o.base.initial_case!="smooth")||
     (o.representation=="broken"&&o.split!=.5))throw std::runtime_error("invalid composite options");
  return o;
}
void CompositeSelfTest()
{
  // The ordinary-DG control must reproduce the existing reduced engine.
  {
    CompositeOptions o;o.representation="single";o.quadrature="collocated";o.base.initial_case="smooth";
    o.base.elements=4;o.base.order=5;o.base.alpha=.0004;CompositeProblem pr(o);
    const State u=pr.Initial();CompositeWork cw;Work existing;
    const State sc=pr.Closure(u,cw),sr=pr.physics.ExactClosure(u,existing);
    const State rc=pr.Residual(u,sc,cw),rr=pr.physics.Residual(u,sr,pr.ne,true,0,nullptr,nullptr,existing);
    double se=0.,re=0.;
    for(std::size_t i=0;i<sc.size();++i)se=std::max(se,std::abs(sc[i]-sr[i]));
    for(std::size_t i=0;i<rc.size();++i)re=std::max(re,std::abs(rc[i]-rr[i]));
    if(se>1.e-12||re>1.e-11)throw std::runtime_error("single-piece existing-engine equivalence failed");
  }
  for(int p:{3,5,7})for(double split:{.35,.5})
  {
    CompositeOptions o;o.base.elements=4;o.base.order=p;o.split=split;o.base.alpha=.0004;o.base.dt=1.e-5;o.base.final_time=1.e-4;o.base.initial_case="entropy";
    CompositeProblem pr(o);double sbp=0.,derivative=0.;
    for(int i=0;i<pr.n;++i)
    {double dx=0.;for(int j=0;j<pr.n;++j)
      {const double boundary=i==j&&i==0?-1.:(i==j&&i==pr.n-1?1.:0.);sbp=std::max(sbp,std::abs(pr.Q(i,j)+pr.Q(j,i)-boundary));dx+=pr.D(i,j)*pr.nodes[j];}
      derivative=std::max(derivative,std::abs(dx-1.));}
    if(sbp>1.e-11||derivative>1.e-11)throw std::runtime_error("composite SBP/polynomial identity failed");
    for(const std::string quad:{"collocated","closure","all"})
    {
      o.quadrature=quad;CompositeProblem test(o);const auto result=CompositeRun(test);
      if(!result.complete||result.work.maximum_rhs_integral>1.e-10)throw std::runtime_error("composite advancing conservative entropy check failed");
    }
  }
  // A piecewise-linear velocity has distinct left/right derivatives at the
  // shared node. Piece-by-piece source must equal direct quadrature, rather
  // than the square of a composite weighted derivative.
  CompositeOptions o;o.base.elements=2;o.base.order=3;o.base.alpha=.0004;o.base.initial_case="entropy";
  CompositeProblem pr(o);State u(pr.ne*pr.n*4);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {const double x=pr.nodes[i],vx=x<=.5?.1*x:.05+.4*(x-.5);u[pr.Index(e,i,0)]=1.;u[pr.Index(e,i,1)]=vx;u[pr.Index(e,i,2)]=0.;u[pr.Index(e,i,3)]=3.+.5*vx*vx;}
  CompositeWork w;const State sigma=pr.Closure(u,w);double integral=0.;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)integral+=pr.h*pr.mass[i]*sigma[pr.SigmaIndex(e,i)];
  const double expected=2*o.base.alpha*(.5*.1*.1+.5*.4*.4)/(pr.h*pr.h);
  if(std::abs(integral-expected)>1.e-11)throw std::runtime_error("piecewise nonlinear source identity failed");
  std::cout<<"composite SBP, piecewise nonlinear source, conservative entropy advancement and current closure passed\n";
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    const auto o=CompositeParse(argc,argv);if(o.self_test){CompositeSelfTest();return 0;}
    const auto start=Clock::now();CompositeProblem pr(o);const double setup=Wall(start);const auto result=CompositeRun(pr);
    if(!o.output.empty()){std::ofstream file(o.output);if(!file)throw std::runtime_error("cannot open output");CompositePrint(pr,result,setup,file);}
    CompositePrint(pr,result,setup,std::cout);return result.complete?0:2;
  }
  catch(const std::exception& e){std::cerr<<"high-order composite probe: "<<e.what()<<'\n';return 1;}
}

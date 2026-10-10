// Test-only causal hp study. Numerical changes are explicit: quadrature,
// velocity representation and optional source projection. No continuum change.
#define DG_IGR_HIGH_ORDER_HYBRID_LIBRARY_ONLY
#include "igr_high_order_hybrid_probe.cpp"
#undef DG_IGR_HIGH_ORDER_HYBRID_LIBRARY_ONLY

namespace
{
struct CausalOptions
{
  HybridOptions hybrid;
  std::string gradient="quotient",mass="lumped",output;
  int closure_points=0,volume_points=0,projection_points=0;
  int initial_base_order=0,source_degree=-1;
  bool self_test=false;
};
struct CausalRule
{
  State x,w;std::vector<State> value,derivative;
  CausalRule(const Problem& pr,int points)
  {
    mfem::IntegrationRule rule;mfem::QuadratureFunctions1D generator;
    generator.GaussLegendre(points,&rule);
    auto& basis=mfem::poly1d.GetBasis(pr.p,mfem::BasisType::GaussLobatto);
    mfem::Vector v(pr.n),d(pr.n);
    for(int q=0;q<rule.GetNPoints();++q)
    {
      const auto& ip=rule.IntPoint(q);basis.Eval(ip.x,v,d);x.push_back(ip.x);w.push_back(ip.weight);
      value.emplace_back(v.GetData(),v.GetData()+pr.n);derivative.emplace_back(d.GetData(),d.GetData()+pr.n);
    }
  }
};
struct CausalGeometry
{
  HybridGeometry volume;
  CausalRule closure,projection,audit;
  mfem::DenseMatrix inverse_mass,inverse_interior_mass,mass,stiffness,inverse_interior_stiffness;
  CausalGeometry(const Problem& pr,const CausalOptions& o):volume(pr),
    closure(pr,o.closure_points?o.closure_points:pr.p+3),
    projection(pr,o.projection_points?o.projection_points:4*pr.p+5),
    audit(pr,8*pr.p+17),inverse_mass(pr.n),inverse_interior_mass(pr.n-2),mass(pr.n),
    stiffness(pr.n),inverse_interior_stiffness(pr.n-2)
  {
    CausalRule vr(pr,o.volume_points?o.volume_points:pr.p+3);
    volume.qshapes=vr.value;volume.qderivatives=vr.derivative;volume.qweights=vr.w;
    // A common admission set across the closure-quadrature/gradient screens.
    // Always include actual volume, projection and audit points, and the usual
    // closure quadrature choices, so a changed point set is not mistaken for
    // a changed numerical representation in an initial-state comparison.
    for(const auto* rule:{&closure,&projection,&audit})for(const auto& shape:rule->value)
      volume.positivity_shapes.push_back(shape);
    for(int count:{pr.p+1,pr.p+3,2*pr.p+3,4*pr.p+5,8*pr.p+9})
    {CausalRule rule(pr,count);for(const auto& shape:rule.value)volume.positivity_shapes.push_back(shape);}
    mass=0.;for(std::size_t q=0;q<projection.w.size();++q)for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)
      mass(i,j)+=projection.w[q]*projection.value[q][i]*projection.value[q][j];
    mfem::DenseMatrixInverse inv(mass,true);inv.GetInverseMatrix(inverse_mass);
    mfem::DenseMatrix interior(pr.n-2);for(int i=1;i<pr.p;++i)for(int j=1;j<pr.p;++j)interior(i-1,j-1)=mass(i,j);
    mfem::DenseMatrixInverse ii(interior,true);ii.GetInverseMatrix(inverse_interior_mass);
    stiffness=0.;for(std::size_t q=0;q<projection.w.size();++q)for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)
      stiffness(i,j)+=projection.w[q]*projection.derivative[q][i]*projection.derivative[q][j];
    for(int i=1;i<pr.p;++i)for(int j=1;j<pr.p;++j)interior(i-1,j-1)=stiffness(i,j);
    mfem::DenseMatrixInverse si(interior,true);si.GetInverseMatrix(inverse_interior_stiffness);
  }
};
struct CausalWork
{
  HybridWork hybrid;
  double closure_seconds=0.,projection_seconds=0.;
  double maximum_source_integral=0.,maximum_current_residual=0.;
  std::uint64_t source_points=0,velocity_projections=0;
};
std::array<double,4> CausalValue(const Problem& pr,const State& u,int e,const State& shape)
{
  std::array<double,4> point{};
  for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)point[c]+=shape[i]*u[pr.Index(e,i,c)];return point;
}
State CausalVelocity(const Problem& pr,const CausalGeometry& g,const CausalOptions& o,
                     const State& u,int e,CausalWork& work)
{
  const auto start=Clock::now();State v(3*pr.n,0.);
  const bool lifted=o.gradient.size()>4&&o.gradient.substr(o.gradient.size()-4)=="-br1";
  const std::string kind=lifted?o.gradient.substr(0,o.gradient.size()-4):o.gradient;
  if(lifted)
  {
    const auto velocity_at=[&](int element,int i)
      {return u[pr.Index(element,i,1)]/u[pr.Index(element,i,0)];};
    const double left=.5*(velocity_at(e-1,pr.p)-velocity_at(e,0));
    const double right=.5*(velocity_at(e+1,0)-velocity_at(e,pr.p));
    for(int i=0;i<pr.n;++i)v[2*pr.n+i]=(-g.inverse_mass(i,0)*left+g.inverse_mass(i,pr.p)*right)/pr.h;
  }
  if(kind=="quotient")return v;
  if(kind=="nodal")
  {for(int c=0;c<2;++c)for(int i=0;i<pr.n;++i)v[c*pr.n+i]=u[pr.Index(e,i,c+1)]/u[pr.Index(e,i,0)];return v;}
  State b(2*pr.n,0.);mfem::DenseMatrix weighted(pr.n);weighted=0.;
  for(std::size_t k=0;k<g.projection.w.size();++k)
  {
    const auto point=CausalValue(pr,u,e,g.projection.value[k]);pr.CheckState(point.data());
    if(kind=="h1")
    {
      std::array<double,3> derivative{};
      for(int i=0;i<pr.n;++i)for(int c=0;c<3;++c)derivative[c]+=g.projection.derivative[k][i]*u[pr.Index(e,i,c)];
      for(int c=0;c<2;++c)
      {
        const double dv=(derivative[c+1]-point[c+1]*derivative[0]/point[0])/point[0];
        for(int i=0;i<pr.n;++i)b[c*pr.n+i]+=g.projection.w[k]*g.projection.derivative[k][i]*dv;
      }
      continue;
    }
    const double rho_weight=kind=="weighted-l2"?point[0]:1.;
    for(int i=0;i<pr.n;++i)
    {
      for(int c=0;c<2;++c)b[c*pr.n+i]+=g.projection.w[k]*rho_weight*g.projection.value[k][i]*point[c+1]/point[0];
      if(kind=="weighted-l2")for(int j=0;j<pr.n;++j)
        weighted(i,j)+=g.projection.w[k]*rho_weight*g.projection.value[k][i]*g.projection.value[k][j];
    }
  }
  if(kind=="weighted-l2")
  {
    mfem::DenseMatrixInverse inv(weighted,true);
    for(int c=0;c<2;++c){mfem::Vector bv(b.data()+c*pr.n,pr.n),vv(v.data()+c*pr.n,pr.n);inv.Mult(bv,vv);}
  }
  else if(kind=="endpoint-l2"||kind=="h1")
  {
    const auto& matrix=kind=="h1"?g.stiffness:g.mass;
    const auto& inverse=kind=="h1"?g.inverse_interior_stiffness:g.inverse_interior_mass;
    for(int c=0;c<2;++c)
    {
      v[c*pr.n]=u[pr.Index(e,0,c+1)]/u[pr.Index(e,0,0)];v[c*pr.n+pr.p]=u[pr.Index(e,pr.p,c+1)]/u[pr.Index(e,pr.p,0)];
      for(int i=1;i<pr.p;++i)b[c*pr.n+i]-=matrix(i,0)*v[c*pr.n]+matrix(i,pr.p)*v[c*pr.n+pr.p];
      for(int i=1;i<pr.p;++i)for(int j=1;j<pr.p;++j)v[c*pr.n+i]+=inverse(i-1,j-1)*b[c*pr.n+j];
    }
  }
  else for(int c=0;c<2;++c)for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)v[c*pr.n+i]+=g.inverse_mass(i,j)*b[c*pr.n+j];
  ++work.velocity_projections;work.projection_seconds+=Wall(start);return v;
}
double CausalGradient(const Problem& pr,const State& u,int e,const State& velocity,
                      const std::string& kind,const State& shape,const State& derivative)
{
  const bool lifted=kind.size()>4&&kind.substr(kind.size()-4)=="-br1";
  const std::string base=lifted?kind.substr(0,kind.size()-4):kind;
  double lifting=0.;if(lifted)for(int i=0;i<pr.n;++i)lifting+=shape[i]*velocity[2*pr.n+i];
  if(base!="quotient")
  {double result=lifting;for(int i=0;i<pr.n;++i)result+=derivative[i]*velocity[i]/pr.h;return result;}
  const auto point=CausalValue(pr,u,e,shape);double rho_x=0.,momentum_x=0.;
  for(int i=0;i<pr.n;++i){rho_x+=derivative[i]*u[pr.Index(e,i,0)]/pr.h;momentum_x+=derivative[i]*u[pr.Index(e,i,1)]/pr.h;}
  return (momentum_x-point[1]*rho_x/point[0])/point[0]+lifting;
}
double CausalExactGradient(const CausalOptions& o,double x)
{
  const double pi=std::acos(-1.);const std::string& name=o.hybrid.base.initial_case;
  if(name=="entropy")return 0.;
  if(name=="smooth")return .045*2*pi*std::cos(2*pi*x);
  const double z=std::sin(2*pi*(x-o.hybrid.phase))/(2*pi),t=std::tanh(z/o.hybrid.width);
  const double m=o.hybrid.mach,rho2=(gamma_euler+1.)/(gamma_euler-1.+2/(m*m));
  const double rho=rho2+(1-rho2)*.5*(1+t);
  const double rho_x=(1-rho2)*.5*(1-t*t)*std::cos(2*pi*(x-o.hybrid.phase))/o.hybrid.width;
  return m*std::sqrt(gamma_euler)*rho_x/(rho*rho);
}
State CausalLegendre(int degree,double x)
{
  State p(degree+1,1.);if(degree>0)p[1]=2*x-1.;
  for(int k=2;k<=degree;++k)p[k]=((2*k-1)*(2*x-1)*p[k-1]-(k-1)*p[k-2])/k;return p;
}
void CausalElement(const Problem& pr,const CausalGeometry& g,const CausalOptions& o,
                    const State& u,int e,mfem::DenseMatrix& a,State& b,CausalWork& work,
                    bool exact_source=false,bool exact_density=false)
{
  const State velocity=CausalVelocity(pr,g,o,u,e,work);a.SetSize(pr.n);a=0.;b.assign(pr.n,0.);
  State source(g.closure.w.size());std::vector<std::array<double,4>> points(source.size());
  auto initial=MakeHighOrderPeriodicFront(o.hybrid.width,o.hybrid.mach,o.hybrid.speed,o.hybrid.phase);
  for(std::size_t k=0;k<source.size();++k)
  {
    points[k]=CausalValue(pr,u,e,g.closure.value[k]);pr.CheckState(points[k].data());
    const double dx=exact_source?CausalExactGradient(o,pr.h*(e+g.closure.x[k])):
      CausalGradient(pr,u,e,velocity,o.gradient,g.closure.value[k],g.closure.derivative[k]);
    const double gradient[4]={dx,0.,0.,0.};source[k]=igr::SourceFromVelocityGradient(2,gradient,2,1,pr.o.alpha);
    if(!std::isfinite(source[k]))throw std::runtime_error("nonfinite causal source");++work.source_points;
    if(exact_density)
    {
      const double x=pr.h*(e+g.closure.x[k]);
      if(o.hybrid.base.initial_case=="front")
      {mfem::Vector exact(4);initial(x,exact);points[k][0]=exact[0];}
      else points[k][0]=o.hybrid.base.initial_case=="smooth"?1.+.04*std::cos(2*std::acos(-1.)*x):1.+.2*std::sin(2*std::acos(-1.)*x);
    }
  }
  if(o.source_degree>=0)
  {
    State coefficient(o.source_degree+1,0.);
    for(std::size_t k=0;k<source.size();++k)
    {const auto polynomial=CausalLegendre(o.source_degree,g.closure.x[k]);
      for(int j=0;j<=o.source_degree;++j)coefficient[j]+=(2*j+1)*g.closure.w[k]*source[k]*polynomial[j];}
    for(std::size_t k=0;k<source.size();++k)
    {source[k]=0.;const auto polynomial=CausalLegendre(o.source_degree,g.closure.x[k]);
      for(int j=0;j<=o.source_degree;++j)source[k]+=coefficient[j]*polynomial[j];}
  }
  for(std::size_t k=0;k<source.size();++k)
  {
    const auto& shape=g.closure.value[k];const auto& derivative=g.closure.derivative[k];
    const double invrho=1./points[k][0],weight=g.closure.w[k];
    for(int i=0;i<pr.n;++i)
    {
      b[i]+=pr.h*weight*shape[i]*source[k];
      for(int j=0;j<pr.n;++j)a(i,j)+=weight*invrho*(pr.h*shape[i]*shape[j]+pr.o.alpha*derivative[i]*derivative[j]/pr.h);
    }
  }
}
State CausalClosure(const Problem& pr,const CausalGeometry& g,const CausalOptions& o,
                    const State& u,CausalWork& work,bool exact_source=false,bool exact_density=false)
{
  const auto start=Clock::now();double integral=0.;
  auto assemble=[&](const State& state,int e,mfem::DenseMatrix& a,State& b)
    {CausalElement(pr,g,o,state,e,a,b,work,exact_source,exact_density);for(double value:b)integral+=value;};
  auto result=SolveCondensedFromElements(pr,u,1,"band",work.hybrid.condensed,assemble);
  work.maximum_source_integral=std::max(work.maximum_source_integral,integral);
  work.maximum_current_residual=std::max(work.maximum_current_residual,work.hybrid.condensed.residual);
  ++work.hybrid.base.global_factors;work.closure_seconds+=Wall(start);return std::move(result.sigma);
}
State CausalInitial(const Problem& pr,const CausalOptions& o)
{
  if(!o.initial_base_order||o.initial_base_order==pr.p)return HybridInitial(pr,o.hybrid);
  Options lower=o.hybrid.base;lower.order=o.initial_base_order;Problem low(lower);const State low_state=HybridInitial(low,o.hybrid);
  State u(pr.ne*pr.n*4);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {const auto shape=HybridGeometry::Shape(low,pr.basis.nodes[i]);
    for(int c=0;c<4;++c)for(int j=0;j<low.n;++j)u[pr.Index(e,i,c)]+=shape[j]*low_state[low.Index(e,j,c)];}
  return u;
}
struct CausalResult
{
  State initial,u,sigma,oracle_source_sigma,oracle_both_sigma;CausalWork work;
  double time=0.,seconds=0.,setup=0.;bool complete=false;std::string failure;
};
CausalResult CausalRun(const Problem& pr,const CausalGeometry& g,const CausalOptions& o)
{
  CausalResult r;const auto start=Clock::now();r.u=CausalInitial(pr,o);
  try
  {
    HybridLimit(pr,g.volume,r.u,o.hybrid,r.work.hybrid);r.initial=r.u;r.sigma=CausalClosure(pr,g,o,r.u,r.work);
    if(pr.o.final_time==0.)
    {
      CausalOptions exact=o;exact.source_degree=-1;CausalWork oracle;
      r.oracle_source_sigma=CausalClosure(pr,g,exact,r.u,oracle,true,false);
      r.oracle_both_sigma=CausalClosure(pr,g,exact,r.u,oracle,true,true);
    }
    while(r.time<pr.o.final_time-1.e-14)
    {
      const double dt=std::min(pr.o.dt,pr.o.final_time-r.time);++r.work.hybrid.base.attempted_steps;
      const State base=r.u;State stage=r.u;
      for(int k=0;k<3;++k)
      {
        State sigma=k==0?r.sigma:CausalClosure(pr,g,o,stage,r.work);
        State rhs=HybridRhs(pr,g.volume,o.hybrid,stage,sigma,r.work.hybrid);
        if(o.mass=="exact")
        {
          const auto mass_start=Clock::now();const State lumped=rhs;std::fill(rhs.begin(),rhs.end(),0.);
          for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)for(int c=0;c<4;++c)
            rhs[pr.Index(e,i,c)]+=g.inverse_mass(i,j)*pr.basis.weights[j]*lumped[pr.Index(e,j,c)];
          r.work.hybrid.transport_seconds+=Wall(mass_start);
        }
        for(std::size_t i=0;i<stage.size();++i)
        {
          if(k==0)stage[i]=base[i]+dt*rhs[i];
          if(k==1)stage[i]=.75*base[i]+.25*(stage[i]+dt*rhs[i]);
          if(k==2)stage[i]=base[i]/3.+(2./3.)*(stage[i]+dt*rhs[i]);
        }
        HybridLimit(pr,g.volume,stage,o.hybrid,r.work.hybrid);
      }
      State sigma=CausalClosure(pr,g,o,stage,r.work);
      for(int e=0;e<pr.ne;++e)for(std::size_t k=0;k<g.volume.qweights.size();++k)
      {const auto point=CausalValue(pr,stage,e,g.volume.qshapes[k]);double s=0.,speed=0.;
        for(int i=0;i<pr.n;++i)s+=g.volume.qshapes[k][i]*sigma[pr.SigmaIndex(e,i)];pr.Flux(point.data(),s,speed);}
      r.u.swap(stage);r.sigma.swap(sigma);r.time+=dt;++r.work.hybrid.base.accepted_steps;
    }
    r.complete=true;
  }
  catch(const std::exception& error){r.failure=error.what();r.work.hybrid.base.rejected_steps=r.work.hybrid.base.attempted_steps-r.work.hybrid.base.accepted_steps;}
  r.seconds=Wall(start);return r;
}
struct CausalAudit
{
  double source_integral=0.,exact_source_integral=0.,gradient_error=0.,gradient_norm=0.;
  double velocity_jump_squared=0.,max_velocity_jump=0.,weighted_sigma_mean=0.;
};
CausalAudit AuditCausal(const Problem& pr,const CausalGeometry& g,const CausalOptions& o,const State& u,const State& sigma)
{
  CausalAudit a;CausalWork work;
  for(int e=0;e<pr.ne;++e)
  {
    const State velocity=CausalVelocity(pr,g,o,u,e,work);
    for(std::size_t k=0;k<g.audit.w.size();++k)
    {
      const double x=pr.h*(e+g.audit.x[k]);const auto point=CausalValue(pr,u,e,g.audit.value[k]);
      const double gradient=CausalGradient(pr,u,e,velocity,o.gradient,g.audit.value[k],g.audit.derivative[k]);
      const double exact=CausalExactGradient(o,x),weight=pr.h*g.audit.w[k];
      a.source_integral+=2*pr.o.alpha*weight*gradient*gradient;
      a.exact_source_integral+=2*pr.o.alpha*weight*exact*exact;
      a.gradient_error+=weight*(gradient-exact)*(gradient-exact);a.gradient_norm+=weight*exact*exact;
      if(!sigma.empty())
      {double sq=0.;for(int i=0;i<pr.n;++i)sq+=g.audit.value[k][i]*sigma[pr.SigmaIndex(e,i)];a.weighted_sigma_mean+=weight*sq/point[0];}
    }
    const double left=u[pr.Index(e,pr.p,1)]/u[pr.Index(e,pr.p,0)],right=u[pr.Index(e+1,0,1)]/u[pr.Index(e+1,0,0)];
    a.velocity_jump_squared+=(left-right)*(left-right);a.max_velocity_jump=std::max(a.max_velocity_jump,std::abs(left-right));
  }
  a.gradient_error=std::sqrt(a.gradient_error);a.gradient_norm=std::sqrt(a.gradient_norm);return a;
}
void CausalPrint(const Problem& pr,const CausalGeometry& g,const CausalOptions& o,const CausalResult& r,std::ostream& out)
{
  // Static exact-gradient comparisons are meaningful only at t=0; advancing
  // audits retain source/gradient energies but never compare to the old IC.
  const auto audit=AuditCausal(pr,g,o,r.u,r.sigma);
  double conservation=0.;if(!r.initial.empty())
  {const auto before=pr.Integrals(r.initial),after=pr.Integrals(r.u);
    for(int c=0;c<4;++c)conservation=std::max(conservation,std::abs(before[c]-after[c]));}
  HybridResult hybrid;hybrid.u=r.u;hybrid.sigma=r.sigma;hybrid.work=r.work.hybrid;hybrid.time=r.time;hybrid.complete=r.complete;hybrid.failure=r.failure;hybrid.setup=r.setup;hybrid.seconds=r.seconds;
  std::ostringstream record;HybridPrint(pr,o.hybrid,hybrid,{{0.,0.,0.}},record);std::string json=record.str();
  const auto end=json.find_last_of('}');json.erase(end);
  out<<std::setprecision(17)<<json<<",\"source_gradient\":\""<<o.gradient<<"\",\"causal_transport_mass\":\""<<o.mass
    <<"\",\"causal_conservative_drift\":"<<conservation<<",\"closure_points\":"<<g.closure.w.size()
    <<",\"volume_points\":"<<g.volume.qweights.size()<<",\"projection_points\":"<<g.projection.w.size()
    <<",\"initial_base_order\":"<<o.initial_base_order<<",\"source_degree\":"<<o.source_degree
    <<",\"source_integral_audit\":"<<audit.source_integral<<",\"weighted_sigma_mean_audit\":"<<audit.weighted_sigma_mean
    <<",\"source_integral_maximum_assembly\":"<<r.work.maximum_source_integral
    <<",\"initial_exact_source_integral\":"<<(r.time==0.?audit.exact_source_integral:-1.)
    <<",\"initial_gradient_relative_error\":"<<(r.time==0.?audit.gradient_error/std::max(audit.gradient_norm,1.e-20):-1.)
    <<",\"velocity_face_jump_l2\":"<<std::sqrt(audit.velocity_jump_squared)<<",\"maximum_velocity_face_jump\":"<<audit.max_velocity_jump
    <<",\"causal_current_residual\":"<<r.work.maximum_current_residual<<",\"causal_closure_seconds\":"<<r.work.closure_seconds
    <<",\"velocity_projection_seconds\":"<<r.work.projection_seconds<<",\"source_evaluation_points\":"<<r.work.source_points
    <<",\"velocity_projections\":"<<r.work.velocity_projections<<",\"initial_state\":[";
  for(std::size_t i=0;i<r.initial.size();++i){if(i)out<<',';out<<r.initial[i];}out<<"],\"oracle_source_sigma\":[";
  for(std::size_t i=0;i<r.oracle_source_sigma.size();++i){if(i)out<<',';out<<r.oracle_source_sigma[i];}out<<"],\"oracle_both_sigma\":[";
  for(std::size_t i=0;i<r.oracle_both_sigma.size();++i){if(i)out<<',';out<<r.oracle_both_sigma[i];}out<<"]}\n";
}
CausalOptions CausalParse(int argc,char** argv)
{
  CausalOptions o;o.hybrid.base.elements=8;o.hybrid.base.order=7;o.hybrid.base.alpha=.0004;
  o.hybrid.base.initial_case="front";o.hybrid.base.final_time=0.;o.hybrid.base.dt=.00002;
  o.hybrid.quadrature="all";o.hybrid.method="dg";o.hybrid.closure_solver="condensed-band";
  for(int i=1;i<argc;++i)
  {
    const std::string flag=argv[i];if(flag=="--self-test"){o.self_test=true;continue;}
    if(i+1==argc)throw std::runtime_error("missing argument");const std::string value=argv[++i];
    if(flag=="--elements")o.hybrid.base.elements=std::stoi(value);else if(flag=="--order")o.hybrid.base.order=std::stoi(value);
    else if(flag=="--gradient")o.gradient=value;else if(flag=="--mass")o.mass=value;else if(flag=="--case")o.hybrid.base.initial_case=value;
    else if(flag=="--method")o.hybrid.method=value;else if(flag=="--threshold")o.hybrid.threshold=std::stod(value);
    else if(flag=="--closure-points")o.closure_points=std::stoi(value);else if(flag=="--volume-points")o.volume_points=std::stoi(value);
    else if(flag=="--projection-points")o.projection_points=std::stoi(value);else if(flag=="--source-degree")o.source_degree=std::stoi(value);
    else if(flag=="--initial-base-order")o.initial_base_order=std::stoi(value);else if(flag=="--alpha")o.hybrid.base.alpha=std::stod(value);
    else if(flag=="--dt")o.hybrid.base.dt=std::stod(value);else if(flag=="--final-time")o.hybrid.base.final_time=std::stod(value);
    else if(flag=="--width")o.hybrid.width=std::stod(value);else if(flag=="--phase")o.hybrid.phase=std::stod(value);
    else if(flag=="--mach")o.hybrid.mach=std::stod(value);else if(flag=="--speed")o.hybrid.speed=std::stod(value);
    else if(flag=="--output")o.output=value;else throw std::runtime_error("unknown argument "+flag);
  }
  const int p=o.hybrid.base.order;
  if(o.hybrid.base.elements<2||p<3||p>15||o.hybrid.base.alpha<0||o.hybrid.base.dt<=0||o.hybrid.base.final_time<0||
     o.closure_points<0||o.volume_points<0||o.projection_points<0||o.initial_base_order<0||o.initial_base_order>p||o.source_degree>p||
     (o.gradient!="quotient"&&o.gradient!="nodal"&&o.gradient!="l2"&&o.gradient!="weighted-l2"&&o.gradient!="endpoint-l2"&&o.gradient!="h1"&&
      o.gradient!="quotient-br1"&&o.gradient!="nodal-br1"&&o.gradient!="h1-br1")||
     (o.mass!="lumped"&&o.mass!="exact")||(o.mass=="exact"&&o.hybrid.method!="dg")||
     (o.hybrid.method!="dg"&&o.hybrid.method!="blend"&&o.hybrid.method!="low")||
     (o.hybrid.base.initial_case!="front"&&o.hybrid.base.initial_case!="smooth"&&o.hybrid.base.initial_case!="entropy"))
    throw std::runtime_error("invalid causal options");
  return o;
}
void CausalSelfTest()
{
  CausalOptions fixture;fixture.hybrid.base.initial_case="front";
  auto front=MakeHighOrderPeriodicFront(fixture.hybrid.width,fixture.hybrid.mach,fixture.hybrid.speed,fixture.hybrid.phase);
  for(double x:{.12,.13,.14,.62,.63,.64})
  {
    mfem::Vector plus(4),minus(4);const double step=1.e-7;front(x+step,plus);front(x-step,minus);
    const double finite_difference=(plus[1]/plus[0]-minus[1]/minus[0])/(2*step),exact=CausalExactGradient(fixture,x);
    if(std::abs(finite_difference-exact)>1.e-7*std::max(1.,std::abs(exact)))
      throw std::runtime_error("shared front analytic velocity derivative mismatch");
  }
  for(int p:{3,7,11,15})
  {
    CausalOptions o;o.hybrid.base.elements=4;o.hybrid.base.order=p;o.hybrid.base.alpha=.0004;o.hybrid.base.initial_case="smooth";
    o.hybrid.quadrature="all";Problem pr(o.hybrid.base);CausalGeometry g(pr,o);const State u=CausalInitial(pr,o);
    CausalWork w;const State sigma=CausalClosure(pr,g,o,u,w);HybridWork hw;const State oracle=HybridClosure(pr,g.volume,o.hybrid,u,hw);
    State error=sigma;for(std::size_t i=0;i<error.size();++i)error[i]-=oracle[i];
    if(Norm(error)>1.e-10*Norm(oracle))throw std::runtime_error("quotient closure differs from inherited current oracle");
    CausalOptions mean=o;mean.source_degree=0;CausalWork mw;CausalClosure(pr,g,mean,u,mw);
    if(std::abs(mw.maximum_source_integral-w.maximum_source_integral)>1.e-12)throw std::runtime_error("source projection changed constant load");
    State linear(u.size());
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {double v=.2+.1*pr.h*(e+pr.basis.nodes[i]);linear[pr.Index(e,i,0)]=1.;linear[pr.Index(e,i,1)]=v;linear[pr.Index(e,i,2)]=0.;linear[pr.Index(e,i,3)]=2.5+.5*v*v;}
    for(const std::string kind:{"quotient","nodal","l2","weighted-l2","endpoint-l2","h1"})
    {
      CausalOptions candidate=o;candidate.gradient=kind;CausalWork cw;const State actual=CausalClosure(pr,g,candidate,linear,cw);
      for(double value:actual)if(std::abs(value-2*o.hybrid.base.alpha*.01)>1.e-11)throw std::runtime_error("linear velocity gradient consistency failed");
    }
    CausalOptions euler=o;euler.hybrid.base.alpha=0.;Problem zero(euler.hybrid.base);CausalGeometry zg(zero,euler);CausalWork zw;
    if(Norm(CausalClosure(zero,zg,euler,u,zw))>1.e-30)throw std::runtime_error("alpha-zero closure not zero");
    // Endpoint-constrained H1 projection has the exact primitive endpoint
    // values and cannot increase the derivative seminorm. The source-energy
    // reduction is projection error, not removal of closure quadrature aliasing.
    CausalOptions hp=o;hp.gradient="h1";CausalWork pw;
    for(int e=0;e<pr.ne;++e)
    {
      const State velocity=CausalVelocity(pr,g,hp,u,e,pw);double projected=0.,original=0.,defect=0.;
      for(int i:{0,pr.p})if(std::abs(velocity[i]-u[pr.Index(e,i,1)]/u[pr.Index(e,i,0)])>1.e-12)
        throw std::runtime_error("H1 projection changed endpoint velocity");
      for(std::size_t q=0;q<g.audit.w.size();++q)
      {
        const double a=CausalGradient(pr,u,e,velocity,"h1",g.audit.value[q],g.audit.derivative[q]);
        const double b=CausalGradient(pr,u,e,velocity,"quotient",g.audit.value[q],g.audit.derivative[q]);
        projected+=g.audit.w[q]*a*a;original+=g.audit.w[q]*b*b;defect+=g.audit.w[q]*(a-b)*(a-b);
      }
      if(std::abs(original-projected-defect)>1.e-9*std::max(original,1.e-15))
        throw std::runtime_error("H1 derivative projection energy identity failed");
    }
    CausalOptions embedded=o;embedded.initial_base_order=3;const State initial=CausalInitial(pr,embedded);
    Options low=o.hybrid.base;low.order=3;Problem lp(low);const State lu=HybridInitial(lp,embedded.hybrid);
    for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<g.audit.w.size();++q)
    {
      const auto actual=CausalValue(pr,initial,e,g.audit.value[q]);
      const auto expected=CausalValue(lp,lu,e,HybridGeometry::Shape(lp,g.audit.x[q]));
      for(int c=0;c<4;++c)if(std::abs(actual[c]-expected[c])>1.e-11)
        throw std::runtime_error("embedded initial polynomial changed");
    }
    for(int j=0;j<pr.n;++j)
    {double weight=0.;for(int i=0;i<pr.n;++i)weight+=pr.basis.weights[i]*g.inverse_mass(i,j)*pr.basis.weights[j];
      if(std::abs(weight-pr.basis.weights[j])>1.e-11)throw std::runtime_error("exact mass correction changed constant test");}
    State discontinuous(u.size());
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {const double v=.2+.1*e;discontinuous[pr.Index(e,i,0)]=1.;discontinuous[pr.Index(e,i,1)]=v;
      discontinuous[pr.Index(e,i,3)]=2.5+.5*v*v;}
    CausalOptions lifted=o;lifted.gradient="nodal-br1";CausalWork lw;double periodic_integral=0.,gradient_energy=0.;
    for(int e=0;e<pr.ne;++e)
    {
      const State velocity=CausalVelocity(pr,g,lifted,discontinuous,e,lw);
      for(std::size_t q=0;q<g.audit.w.size();++q)
      {const double derivative=CausalGradient(pr,discontinuous,e,velocity,lifted.gradient,g.audit.value[q],g.audit.derivative[q]);
        periodic_integral+=pr.h*g.audit.w[q]*derivative;gradient_energy+=pr.h*g.audit.w[q]*derivative*derivative;}
    }
    if(std::abs(periodic_integral)>1.e-11||gradient_energy<.01)throw std::runtime_error("central BR1 lifting lost weak derivative identity");
  }
  std::cout<<"causal hp current-closure parity, primitive consistency, source mean preservation and alpha-zero checks passed\n";
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    const auto o=CausalParse(argc,argv);if(o.self_test){CausalSelfTest();return 0;}
    const auto start=Clock::now();Problem pr(o.hybrid.base);CausalGeometry g(pr,o);const double setup=Wall(start);
    auto result=CausalRun(pr,g,o);result.setup=setup;
    if(!o.output.empty()){std::ofstream file(o.output);if(!file)throw std::runtime_error("cannot open output");CausalPrint(pr,g,o,result,file);}
    CausalPrint(pr,g,o,result,std::cout);return result.complete?0:2;
  }
  catch(const std::exception& e){std::cerr<<"hp causal probe: "<<e.what()<<'\n';return 1;}
}

// Test-only periodic DG/IGR pressure-work compatibility diagnostic.
// It differentiates discrete kinetic/internal energy and physical entropy,
// separates central sigma work from the change in LLF speed, and evaluates
// exact finite-dimensional identities including every periodic face jump.
#define DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY
#include "igr_hp_front_tracking_probe.cpp"
#undef DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY
#include "numerics/dgsem/dgsem_entropy_flux.hpp"

namespace
{
struct WorkOptions
{
  int elements=8,audit_extra=30;std::vector<int> orders{3,7,11};
  double alpha=.0004;std::string state="all",mass="all",source="all",stress="all",space="l2",output;
  bool self_test=false;
};
struct WorkRule{State w;std::vector<State> v,d;};
struct WorkProblem:TrackingProblem
{
  std::string mass_kind;
  WorkRule audit;
  WorkProblem(const WorkOptions& options,int degree,const std::string& mode)
    :TrackingProblem(MakeOptions(options,degree)),mass_kind(mode)
  {
    weights.clear();shapes.clear();derivatives.clear();
    if(mode=="collocated")
      for(int i=0;i<n;++i){weights.push_back(physics.basis.weights[i]);shapes.push_back(Shape(physics.basis.nodes[i],false));derivatives.push_back(Shape(physics.basis.nodes[i],true));}
    else
    {
      const auto& rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,4*p+16);
      for(int i=0;i<rule.GetNPoints();++i){const auto& ip=rule.IntPoint(i);weights.push_back(ip.weight);shapes.push_back(Shape(ip.x,false));derivatives.push_back(Shape(ip.x,true));}
    }
    mass=0.;
    if(mode=="exact")
      for(std::size_t q=0;q<weights.size();++q)for(int i=0;i<n;++i)for(int j=0;j<n;++j)mass(i,j)+=weights[q]*shapes[q][i]*shapes[q][j];
    else for(int i=0;i<n;++i)mass(i,i)=physics.basis.weights[i];
    mfem::DenseMatrixInverse inverse(mass,true);mfem::DenseMatrix identity(n);identity=0.;for(int i=0;i<n;++i)identity(i,i)=1.;inverse.Mult(identity,mass_inverse);
    const auto& high=mfem::IntRules.Get(mfem::Geometry::SEGMENT,8*p+options.audit_extra);
    for(int q=0;q<high.GetNPoints();++q){const auto& ip=high.IntPoint(q);audit.w.push_back(ip.weight);audit.v.push_back(Shape(ip.x,false));audit.d.push_back(Shape(ip.x,true));}
  }
  static TrackOptions MakeOptions(const WorkOptions& options,int degree)
  {TrackOptions t;t.base.elements=options.elements;t.base.order=degree;t.base.alpha=options.alpha;t.initial="entropy";t.closure_space=options.space;t.closure_boundary="natural";t.left=0.;t.right=1.;t.center=.5;t.mode="uniform";return t;}
  WorkRule MethodRule()const{return {weights,shapes,derivatives};}
  double Scalar(const State& coefficients,int e,const State& shape)const
  {double v=0.;for(int i=0;i<n;++i)v+=shape[i]*coefficients[e*n+i];return v;}
  double Stress(const State& sigma,int e,const State& shape)const
  {double v=0.;for(int i=0;i<n;++i)v+=shape[i]*sigma[SigmaIndex(e,i)];return v;}
  double VelocityDerivative(const State& u,int e,const State& shape,const State& derivative)const
  {const auto v=Point(u,e,shape);const auto dv=Point(u,e,derivative);return (dv[1]-v[1]*dv[0]/v[0])/v[0]/(edges[e+1]-edges[e]);}
  State MakeState(const std::string& kind)const
  {
    State u(ne*n*4,0.);const double pi=std::acos(-1.);
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
    {
      const double x=edges[e]+(edges[e+1]-edges[e])*physics.basis.nodes[i];double rho,v,pressure;
      if(kind=="smooth")
      {rho=1.+.2*std::sin(2*pi*x);v=.35+.18*std::cos(2*pi*x)+.05*std::sin(4*pi*x);pressure=1.+.15*std::sin(2*pi*x+.3);}
      else
      {const double front=.5*(std::tanh((x-.30)/.01)-std::tanh((x-.72)/.01));rho=1.+.65*front+.12*(e%2?1.:-1.);v=.5-.6*front+.08*(e%3-1);pressure=1.+.15*std::sin(2*pi*x)+.05*(e%2);}
      const auto point=PrimitiveState(rho,v,pressure);for(int c=0;c<4;++c)u[Id(e,i,c)]=point[c];
    }return u;
  }
  // Riesz representative for the chosen energy functional. This is an L2
  // projection only when its quadrature mass is the transport mass.
  State Represent(const State& u,const WorkRule& q,int quantity)const
  {
    State out(ne*n,0.);for(int e=0;e<ne;++e)
    {
      State load(n,0.);for(std::size_t k=0;k<q.w.size();++k)
      {
        const auto v=Point(u,e,q.v[k]);const double velocity=v[1]/v[0],pressure=(gamma_euler-1.)*(v[3]-.5*(v[1]*v[1]+v[2]*v[2])/v[0]);
        const double value=quantity==0?velocity:(quantity==1?v[0]/pressure:v[1]/pressure);
        for(int i=0;i<n;++i)load[i]+=q.w[k]*q.v[k][i]*value;
      }
      for(int i=0;i<n;++i)for(int j=0;j<n;++j)out[e*n+i]+=mass_inverse(i,j)*load[j];
    }return out;
  }
};
struct ClosureInfo{double residual=0.,source_integral=0.,weighted_mean=0.;};
State PeriodicClosure(const WorkProblem& pr,const State& u,const std::string& source,ClosureInfo& info)
{
  TrackWork ignored;BandMatrix band;State rhs;pr.AssembleClosure(u,0.,ignored,band,rhs);
  if(source=="projected-velocity")
  {
    const State velocity=pr.Represent(u,pr.MethodRule(),0);std::fill(rhs.begin(),rhs.end(),0.);
    for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
    {const double h=pr.edges[e+1]-pr.edges[e],gradient=pr.Scalar(velocity,e,pr.derivatives[q])/h,value=2.*pr.o.base.alpha*gradient*gradient;
      for(int i=0;i<pr.n;++i)rhs[pr.SigmaIndex(e,i)]+=h*pr.weights[q]*pr.shapes[q][i]*value;}
  }
  const bool l2=pr.o.closure_space=="l2";const int size=l2?pr.ns:pr.ns-1;
  auto periodic=[&](int i){return i==size&&!l2?0:i;};
  mfem::DenseMatrix a(size);a=0.;State b(size,0.);
  for(int i=0;i<pr.ns;++i)
  {b[periodic(i)]+=rhs[i];for(int j=std::max(0,i-band.k);j<std::min(pr.ns,i+band.k+1);++j)a(periodic(i),periodic(j))+=band.Get(i,j);}
  if(l2&&pr.o.base.alpha!=0.)
  {
    const int el=pr.ne-1,er=0;const double hl=pr.edges[el+1]-pr.edges[el],hr=pr.edges[1]-pr.edges[0];
    const double al=pr.o.base.alpha/u[pr.Id(el,pr.p,0)],ar=pr.o.base.alpha/u[pr.Id(er,0,0)];
    const double penalty=pr.o.sipg_penalty*pr.n*pr.n*.5*(al/hl+ar/hr);const State dl=pr.Shape(1.,true),dr=pr.Shape(0.,true);
    for(int i=0;i<2*pr.n;++i)for(int j=0;j<2*pr.n;++j)
    {
      const bool il=i<pr.n,jl=j<pr.n;const int ii=i%pr.n,jj=j%pr.n;
      const double ji=il?(ii==pr.p?1.:0.):(ii==0?-1.:0.),jjump=jl?(jj==pr.p?1.:0.):(jj==0?-1.:0.);
      const double fi=.5*(il?al*dl[ii]/hl:ar*dr[ii]/hr),fj=.5*(jl?al*dl[jj]/hl:ar*dr[jj]/hr);
      a(pr.SigmaIndex(il?el:er,ii),pr.SigmaIndex(jl?el:er,jj))+=-ji*fj-fi*jjump+penalty*ji*jjump;
    }
  }
  mfem::DenseMatrixInverse inverse(a,true);State x(size,0.);mfem::Vector bv(b.data(),size),xv(x.data(),size);inverse.Mult(bv,xv);
  State defect(size,0.);for(int i=0;i<size;++i){defect[i]=-b[i];for(int j=0;j<size;++j)defect[i]+=a(i,j)*x[j];}
  info.residual=Norm(defect)/std::max(Norm(b),1.e-20);if(!std::isfinite(info.residual)||info.residual>1.e-8)throw std::runtime_error("periodic current closure residual failed");
  State sigma(pr.ns);for(int i=0;i<pr.ns;++i)sigma[i]=x[periodic(i)];
  for(double value:b)info.source_integral+=value;
  for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
  {const auto v=pr.Point(u,e,pr.shapes[q]);info.weighted_mean+=(pr.edges[e+1]-pr.edges[e])*pr.weights[q]*pr.Stress(sigma,e,pr.shapes[q])/v[0];}
  return sigma;
}
std::array<double,4> SigmaFlux(const std::array<double,4>& u,double sigma)
{std::array<double,4> flux{};const double normal[2]={1.,0.};igr::AddNormalSigmaFlux(2,u.data(),1,normal,1,sigma,floor_value,flux.data(),1);return flux;}
State SigmaRate(const WorkProblem& pr,const State& u,const State& sigma,bool speed_change)
{
  State load(u.size(),0.),rate(u.size(),0.);
  if(!speed_change)for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
  {const auto v=pr.Point(u,e,pr.shapes[q]);const auto flux=SigmaFlux(v,pr.Stress(sigma,e,pr.shapes[q]));for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)load[pr.Id(e,i,c)]+=pr.weights[q]*pr.derivatives[q][i]*flux[c];}
  for(int face=0;face<pr.ne;++face)
  {
    const int el=(face+pr.ne-1)%pr.ne,er=face;std::array<double,4> ul{},ur{},flux{};
    for(int c=0;c<4;++c){ul[c]=u[pr.Id(el,pr.p,c)];ur[c]=u[pr.Id(er,0,c)];}
    const double sl=sigma[pr.SigmaIndex(el,pr.p)],sr=sigma[pr.SigmaIndex(er,0)];
    if(speed_change)
    {
      double al,ar,a0l,a0r;pr.physics.Flux(ul.data(),sl,al);pr.physics.Flux(ur.data(),sr,ar);pr.physics.Flux(ul.data(),0.,a0l);pr.physics.Flux(ur.data(),0.,a0r);
      const double delta=pr.physics.FaceSpeed(al,ar)-pr.physics.FaceSpeed(a0l,a0r);
      for(int c=0;c<4;++c)flux[c]=-.5*delta*(ur[c]-ul[c]);
    }
    else{const auto fl=SigmaFlux(ul,sl),fr=SigmaFlux(ur,sr);for(int c=0;c<4;++c)flux[c]=.5*(fl[c]+fr[c]);}
    for(int c=0;c<4;++c){load[pr.Id(el,pr.p,c)]-=flux[c];load[pr.Id(er,0,c)]+=flux[c];}
  }
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)for(int c=0;c<4;++c)rate[pr.Id(e,i,c)]+=pr.mass_inverse(i,j)*load[pr.Id(e,j,c)]/(pr.edges[e+1]-pr.edges[e]);return rate;
}
struct Rates{double mass=0.,total=0.,kinetic=0.,internal=0.,entropy=0.;};
Rates RatesAt(const WorkProblem& pr,const State& u,const State& rate,const WorkRule& rule)
{
  Rates out;for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<rule.w.size();++q)
  {
    const auto v=pr.Point(u,e,rule.v[q]),dv=pr.Point(rate,e,rule.v[q]);const double rho=v[0],vx=v[1]/rho,vy=v[2]/rho,speed2=vx*vx+vy*vy;
    const double pressure=(gamma_euler-1.)*(v[3]-.5*rho*speed2),beta=rho/pressure,specific=std::log(pressure/std::pow(rho,gamma_euler))/(gamma_euler-1.);
    const double weight=(pr.edges[e+1]-pr.edges[e])*rule.w[q];
    out.mass+=weight*dv[0];out.total+=weight*dv[3];out.kinetic+=weight*(vx*dv[1]+vy*dv[2]-.5*speed2*dv[0]);
    out.entropy+=weight*((specific-gamma_euler/(gamma_euler-1.)+.5*beta*speed2)*dv[0]-beta*(vx*dv[1]+vy*dv[2])+beta*dv[3]);
  }out.internal=out.total-out.kinetic;return out;
}
struct Budget
{
  Rates central,speed;
  double internal_volume=0.,internal_faces=0.,internal_continuum_volume=0.;
  double entropy_volume=0.,entropy_faces=0.,entropy_continuum_volume=0.;
  double internal_identity_defect=0.,entropy_identity_defect=0.;
};
Budget WorkBudget(const WorkProblem& pr,const State& u,const State& sigma,
                  const State& central,const State& speed,const WorkRule& functional)
{
  Budget out;out.central=RatesAt(pr,u,central,functional);out.speed=RatesAt(pr,u,speed,functional);
  const State velocity=pr.Represent(u,functional,0),beta=pr.Represent(u,functional,1),beta_velocity=pr.Represent(u,functional,2);
  for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
  {
    const auto v=pr.Point(u,e,pr.shapes[q]);const double stress=pr.Stress(sigma,e,pr.shapes[q]),weight=pr.weights[q];
    out.internal_volume-=weight*stress*pr.Scalar(velocity,e,pr.derivatives[q]);
    out.entropy_volume+=weight*stress*(v[1]/v[0]*pr.Scalar(beta,e,pr.derivatives[q])-pr.Scalar(beta_velocity,e,pr.derivatives[q]));
  }
  for(int face=0;face<pr.ne;++face)
  {
    const int el=(face+pr.ne-1)%pr.ne,er=face;const double sl=sigma[pr.SigmaIndex(el,pr.p)],sr=sigma[pr.SigmaIndex(er,0)];
    const double ul=u[pr.Id(el,pr.p,1)]/u[pr.Id(el,pr.p,0)],ur=u[pr.Id(er,0,1)]/u[pr.Id(er,0,0)];
    const double jumpv=velocity[el*pr.n+pr.p]-velocity[er*pr.n],jumpb=beta[el*pr.n+pr.p]-beta[er*pr.n],jumpbv=beta_velocity[el*pr.n+pr.p]-beta_velocity[er*pr.n];
    out.internal_faces+=jumpv*.5*(sl+sr);out.entropy_faces+=-jumpb*.5*(sl*ul+sr*ur)+jumpbv*.5*(sl+sr);
  }
  for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<functional.w.size();++q)
  {
    const auto v=pr.Point(u,e,functional.v[q]);const double pressure=(gamma_euler-1.)*(v[3]-.5*(v[1]*v[1]+v[2]*v[2])/v[0]);
    const double term=(pr.edges[e+1]-pr.edges[e])*functional.w[q]*pr.Stress(sigma,e,functional.v[q])*pr.VelocityDerivative(u,e,functional.v[q],functional.d[q]);
    out.internal_continuum_volume-=term;out.entropy_continuum_volume-=v[0]/pressure*term;
  }
  out.internal_identity_defect=out.central.internal-out.internal_volume-out.internal_faces;
  out.entropy_identity_defect=out.central.entropy-out.entropy_volume-out.entropy_faces;return out;
}
void WriteRates(std::ostream& out,const Rates& r)
{out<<"{\"mass\":"<<r.mass<<",\"total_energy\":"<<r.total<<",\"kinetic_energy\":"<<r.kinetic<<",\"internal_energy\":"<<r.internal<<",\"physical_entropy\":"<<r.entropy<<'}';}
void WriteBudget(std::ostream& out,const Budget& b)
{
  out<<"{\"central_stress_rate\":";WriteRates(out,b.central);out<<",\"sigma_speed_llf_rate\":";WriteRates(out,b.speed);
  out<<",\"internal_volume_work\":"<<b.internal_volume<<",\"internal_face_work\":"<<b.internal_faces<<",\"internal_continuum_broken_volume\":"<<b.internal_continuum_volume
     <<",\"internal_volume_product_quadrature_defect\":"<<b.internal_volume-b.internal_continuum_volume<<",\"internal_identity_defect\":"<<b.internal_identity_defect
     <<",\"entropy_volume_work\":"<<b.entropy_volume<<",\"entropy_face_work\":"<<b.entropy_faces<<",\"entropy_continuum_broken_volume\":"<<b.entropy_continuum_volume
     <<",\"entropy_volume_product_quadrature_defect\":"<<b.entropy_volume-b.entropy_continuum_volume<<",\"entropy_identity_defect\":"<<b.entropy_identity_defect<<'}';
}
void WorkCase(const WorkOptions& options,int degree,const std::string& state,const std::string& mass,
              const std::string& source,const std::string& stress,std::ostream& out)
{
  WorkProblem pr(options,degree,mass);const State u=pr.MakeState(state);ClosureInfo info;State sigma;
  if(stress=="closure")sigma=PeriodicClosure(pr,u,source,info);
  else
  {sigma.assign(pr.ns,0.);for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i){const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];sigma[pr.SigmaIndex(e,i)]=.4+.2*std::sin(2.*std::acos(-1.)*x)+.05*std::cos(4.*std::acos(-1.)*x);}}
  double minrho=INFINITY,minpressure=INFINITY,grad_gap=0.,grad_norm=0.,source_conservative=0.,source_projected=0.,jumps=0.;const State velocity=pr.Represent(u,pr.MethodRule(),0);
  for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.audit.w.size();++q)
  {const auto v=pr.Point(u,e,pr.audit.v[q]);minrho=std::min(minrho,v[0]);minpressure=std::min(minpressure,(gamma_euler-1.)*(v[3]-.5*(v[1]*v[1]+v[2]*v[2])/v[0]));}
  if(!(minrho>0.)||!(minpressure>0.))throw std::runtime_error("diagnostic state is not admitted at common audit points");
  for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
  {const double h=pr.edges[e+1]-pr.edges[e],gc=pr.VelocityDerivative(u,e,pr.shapes[q],pr.derivatives[q]),gp=pr.Scalar(velocity,e,pr.derivatives[q])/h,weight=h*pr.weights[q];grad_gap+=weight*(gp-gc)*(gp-gc);grad_norm+=weight*gc*gc;source_conservative+=weight*2.*pr.o.base.alpha*gc*gc;source_projected+=weight*2.*pr.o.base.alpha*gp*gp;}
  for(int face=0;face<pr.ne;++face){const int el=(face+pr.ne-1)%pr.ne,er=face;const double jump=u[pr.Id(el,pr.p,1)]/u[pr.Id(el,pr.p,0)]-u[pr.Id(er,0,1)]/u[pr.Id(er,0,0)];jumps+=jump*jump;}
  const State central=SigmaRate(pr,u,sigma,false),speed=SigmaRate(pr,u,sigma,true);
  const Budget method=WorkBudget(pr,u,sigma,central,speed,pr.MethodRule()),audit=WorkBudget(pr,u,sigma,central,speed,pr.audit);
  for(const Budget* b:{&method,&audit})
  {
    const double scale=std::max({1.,std::abs(b->central.internal),std::abs(b->central.entropy)});
    if(std::abs(b->internal_identity_defect)>3.e-10*scale||std::abs(b->entropy_identity_defect)>3.e-10*scale||std::abs(b->central.total)>3.e-10*scale||std::abs(b->speed.total)>3.e-10*scale)throw std::runtime_error("discrete stress work identity failed");
  }
  if(stress=="closure"&&std::abs(info.weighted_mean-info.source_integral)>1.e-9*std::max(1.,std::abs(info.source_integral)))throw std::runtime_error("periodic closure constant-test balance failed");
  out<<std::setprecision(17)<<"{\"order\":"<<degree<<",\"elements\":"<<options.elements<<",\"state\":\""<<state<<"\",\"mass\":\""<<mass<<"\",\"source_gradient\":\""<<source<<"\",\"stress\":\""<<stress<<"\",\"closure_space\":\""<<options.space<<"\",\"alpha\":"<<options.alpha
     <<",\"method_quadrature_points\":"<<pr.weights.size()<<",\"audit_quadrature_points\":"<<pr.audit.w.size()<<",\"minimum_density\":"<<minrho<<",\"minimum_pressure\":"<<minpressure<<",\"velocity_face_jump_squared_sum\":"<<jumps
     <<",\"relative_velocity_gradient_gap_l2\":"<<std::sqrt(grad_gap/std::max(grad_norm,1.e-30))<<",\"source_integral_conservative\":"<<source_conservative<<",\"source_integral_projected\":"<<source_projected
     <<",\"closure_residual\":"<<info.residual<<",\"closure_source_integral\":"<<info.source_integral<<",\"closure_weighted_sigma_mean\":"<<info.weighted_mean<<",\"method_functional\":";WriteBudget(out,method);out<<",\"common_audit_functional\":";WriteBudget(out,audit);out<<'}';
}
WorkOptions WorkParse(int argc,char** argv)
{
  WorkOptions o;for(int i=1;i<argc;++i)
  {
    const std::string flag=argv[i];if(flag=="--self-test"){o.self_test=true;continue;}
    if(i+1==argc)throw std::runtime_error("missing option value");const std::string v=argv[++i];
    if(flag=="--elements")o.elements=std::stoi(v);else if(flag=="--orders"){o.orders.clear();std::istringstream list(v);std::string value;while(std::getline(list,value,','))o.orders.push_back(std::stoi(value));}
    else if(flag=="--alpha")o.alpha=std::stod(v);else if(flag=="--audit-extra")o.audit_extra=std::stoi(v);else if(flag=="--state")o.state=v;else if(flag=="--mass")o.mass=v;else if(flag=="--source-gradient")o.source=v;else if(flag=="--stress")o.stress=v;else if(flag=="--closure-space")o.space=v;else if(flag=="--output")o.output=v;else throw std::runtime_error("unknown work option "+flag);
  }
  auto valid=[](const std::string& value,std::initializer_list<const char*> choices){for(const char* c:choices)if(value==c)return true;return false;};
  if(!valid(o.state,{"smooth","steep","all"})||!valid(o.mass,{"collocated","exact","overintegrated-lumped","all"})||!valid(o.source,{"conservative","projected-velocity","all"})||!valid(o.stress,{"closure","manufactured","all"})||!valid(o.space,{"h1","l2"})||o.elements<8||o.alpha<0.||o.audit_extra<8||o.orders.empty())throw std::runtime_error("invalid work diagnostic options");
  for(int degree:o.orders)if(degree<1||degree>31)throw std::runtime_error("unsupported work diagnostic degree");return o;
}
std::vector<std::string> Choices(const std::string& selected,std::initializer_list<const char*> all)
{if(selected!="all")return {selected};std::vector<std::string> out;for(const char* value:all)out.push_back(value);return out;}
void EntropyFluxDiagonalAudit(std::ostream& out)
{
  Options options;options.elements=8;options.order=3;Problem physics(options);
  std::array<double,4> state{{1.2,.36,.12,3.}};const double sigma=.2,velocity=state[1]/state[0];
  double unused;const auto physical=physics.Flux(state.data(),sigma,unused),physical_zero=physics.Flux(state.data(),0.,unused);
  mfem::Vector v(state.data(),4),flux,zero;out<<'[';
  for(int kind=0;kind<2;++kind)
  {
    if(kind){mfem::dgsem::RanochaFlux2D(v,v,0,gamma_euler,sigma,sigma,flux);mfem::dgsem::RanochaFlux2D(v,v,0,gamma_euler,0.,0.,zero);}
    else{mfem::dgsem::ChandrashekarFlux2D(v,v,0,gamma_euler,sigma,sigma,flux);mfem::dgsem::ChandrashekarFlux2D(v,v,0,gamma_euler,0.,0.,zero);}
    double maximum=0.,split_maximum=0.;mfem::Vector split=zero;
    // Consistent research control: thermal Euler EC flux at sigma=0, plus
    // separate symmetric IGR terms {sigma} and {sigma}{v}. At equal states
    // these are sigma and sigma*v. This is NOT a coupled entropy proof.
    split[1]+=sigma;split[3]+=sigma*velocity;
    for(int c=0;c<4;++c){maximum=std::max(maximum,std::abs(flux[c]-physical[c]));split_maximum=std::max(split_maximum,std::abs(split[c]-physical[c]));}
    if(kind)out<<',';
    out<<std::setprecision(17)<<"{\"helper\":\""<<(kind?"Ranocha":"Chandrashekar")<<"\",\"sigma\":"<<sigma<<",\"velocity\":"<<velocity
       <<",\"physical_energy_increment\":"<<physical[3]-physical_zero[3]<<",\"helper_energy_increment\":"<<flux[3]-zero[3]
       <<",\"energy_flux_excess\":"<<flux[3]-physical[3]<<",\"maximum_diagonal_flux_error\":"<<maximum
       <<",\"helper_sigma_energy_coefficient\":"<<(flux[3]-zero[3])/(sigma*velocity)
       <<",\"consistent_with_shared_physical_flux\":"<<(maximum<1.e-12?"true":"false")
       <<",\"separate_sigma_control_energy_increment\":"<<split[3]-zero[3]<<",\"separate_sigma_control_maximum_diagonal_error\":"<<split_maximum
       <<",\"separate_sigma_control_consistent\":"<<(split_maximum<1.e-12?"true":"false")
       <<",\"separate_sigma_control_claim\":\"Thermal Euler helper at sigma0 plus separate symmetric IGR pressure/work; no coupled entropy-conservation assertion\"}";
  }out<<']';
}
void WorkRun(const WorkOptions& o,std::ostream& out)
{
  out<<"{\"entropy_flux_diagonal_audit\":";EntropyFluxDiagonalAudit(out);out<<",\"scope\":\"Periodic fixed-mesh semidiscrete instantaneous sigma work; signed rates, no entropy monotonicity assertion or runtime claim\",\"physical_entropy\":\"integral rho*log(p/rho^gamma)/(gamma-1)\",\"source_alternative\":\"Mass-Riesz represented velocity; an L2 projection only when quadrature mass and transport mass match\",\"records\":[\n";bool first=true;
  for(int p:o.orders)for(const auto& state:Choices(o.state,{"smooth","steep"}))for(const auto& mass:Choices(o.mass,{"collocated","exact","overintegrated-lumped"}))for(const auto& source:Choices(o.source,{"conservative","projected-velocity"}))for(const auto& stress:Choices(o.stress,{"closure","manufactured"}))
  {if(!first)out<<",\n";first=false;WorkCase(o,p,state,mass,source,stress,out);}out<<"\n]}\n";
}
}
int main(int argc,char** argv)
{
  try
  {
    mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");const auto options=WorkParse(argc,argv);
    if(options.self_test)
    {std::ostringstream sink;WorkOptions test=options;test.orders={3,7,11};test.state="all";test.mass="all";test.source="all";test.stress="all";for(const std::string space:{"h1","l2"}){test.space=space;WorkRun(test,sink);}std::cout<<"DG/IGR work identities passed: periodic faces, two energy functionals, H1/L2, p3/p7/p11, mass/source controls\n";return 0;}
    if(options.output.empty())WorkRun(options,std::cout);else{std::ofstream out(options.output);if(!out)throw std::runtime_error("cannot open work diagnostic output");WorkRun(options,out);}return 0;
  }
  catch(const std::exception& error){std::cerr<<"DG/IGR work probe failed: "<<error.what()<<'\n';return 2;}
}

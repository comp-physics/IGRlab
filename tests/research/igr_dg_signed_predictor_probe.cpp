// Test-only local signed-tail tangent screen. No time integrator or model change.
// The enriched full Euler, LLF, SIPG and reconstruction come from the companion.
#define DG_IGR_SIGNED_ENRICHMENT_LIBRARY_ONLY
#include "igr_dg_signed_enrichment_probe.cpp"
#undef DG_IGR_SIGNED_ENRICHMENT_LIBRARY_ONLY

namespace {
State PredictorRead(const std::string& path,std::size_t count)
{
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open signed predictor input");State a(count);
  for(double& x:a)if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid signed predictor input");
  in>>std::ws;if(!in.eof())throw std::runtime_error("extra signed predictor input entries");return a;
}
State PredictorAdd(const State& a,const State& b,double s)
{if(a.size()!=b.size())throw std::runtime_error("predictor state shape mismatch");State c=a;for(std::size_t i=0;i<c.size();++i)c[i]+=s*b[i];return c;}
State PredictorSubtract(const State& a,const State& b,double denominator=1.)
{if(a.size()!=b.size()||denominator==0.)throw std::runtime_error("predictor difference shape mismatch");State c(a.size());for(std::size_t i=0;i<c.size();++i)c[i]=(a[i]-b[i])/denominator;return c;}
double PredictorMaximum(const State& a)
{double x=0.;for(double v:a)x=std::max(x,std::abs(v));return x;}
double PredictorRelative(const State& a,const State& b)
{return Norm(PredictorSubtract(a,b))/std::max({Norm(a),Norm(b),1.e-12});}
void PredictorString(std::ostream& out,const std::string& s)
{out<<'"';for(char c:s){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else out<<c;}out<<'"';}
State PredictorLegendre(double x,int order)
{State p(order+1,0.);p[0]=1.;if(order)p[1]=x;for(int k=1;k<order;++k)p[k+1]=((2*k+1)*x*p[k]-k*p[k-1])/(k+1);return p;}
void PredictorGeometry(SignedEnrichmentProblem& pr,const State& edges)
{
  for(double x:edges)if(!std::isfinite(x))throw std::runtime_error("nonfinite predictor edge");SignedGeometry(pr,edges);
}
struct PredictorWork
{
  TrackWork shared;std::uint64_t rhs_calls=0,admission_points=0;double density_min=INFINITY,pressure_min=INFINITY,q_min=INFINITY;
};
struct PredictorCoordinates
{
  const SignedEnrichmentProblem& pr;int p,q,extra,block,size;std::vector<int> components;mfem::DenseMatrix modal;std::vector<State> nodal;
  PredictorCoordinates(const SignedEnrichmentProblem& problem,int base):pr(problem),p(base),q(pr.retained.p),extra(q-p),block(0),size(0),modal(extra,pr.retained.n)
  {
    if(pr.enrich_density)components.push_back(0);components.push_back(1);components.push_back(2);if(pr.enrich_energy)components.push_back(3);block=int(components.size())*extra;size=pr.ne*block;
    if(extra<1||extra>8)throw std::runtime_error("predictor screen requires one to eight extra degrees");modal=0.;nodal.resize(pr.retained.n);
    for(int i=0;i<pr.retained.n;++i)nodal[i]=PredictorLegendre(2*pr.retained.physics.basis.nodes[i]-1.,q);
    for(std::size_t j=0;j<pr.weights.size();++j)
    {
      double x=0.;for(int i=0;i<pr.n;++i)x+=pr.shapes[j][i]*pr.physics.basis.nodes[i];const State modes=PredictorLegendre(2*x-1.,q);
      for(int k=0;k<extra;++k)for(int i=0;i<pr.retained.n;++i)modal(k,i)+=(2*(p+1+k)+1)*pr.weights[j]*modes[p+1+k]*pr.low_shapes[j][i];
    }
  }
  State Extract(const State& high)const
  {
    const State low=pr.state_transfer.Apply(pr.state_transfer.P,high,pr.ne,4);State out(size,0.);
    for(int e=0;e<pr.ne;++e)for(int k=0;k<extra;++k)for(int c=0;c<int(components.size());++c)for(int i=0;i<pr.retained.n;++i)out[e*block+int(components.size())*k+c]+=modal(k,i)*low[(e*pr.retained.n+i)*4+components[c]];return out;
  }
  State Embed(const State& z)const
  {
    if(int(z.size())!=size)throw std::runtime_error("predictor tail size mismatch");State low(pr.ne*pr.retained.n*4,0.);
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.retained.n;++i)for(int k=0;k<extra;++k)for(int c=0;c<int(components.size());++c)low[(e*pr.retained.n+i)*4+components[c]]+=nodal[i][p+1+k]*z[e*block+int(components.size())*k+c];
    return pr.state_transfer.Apply(pr.state_transfer.E,low,pr.ne,4);
  }
  State Coarse(const State& u)const{return PredictorAdd(u,Embed(Extract(u)),-1.);}
  // Exact kinetic tangent metric at frozen rho: integral rho |delta v|^2.
  // No claim that this is a global entropy norm or bounds transient growth.
  State KineticMetric(const State& external)const
  {
    const State physical=pr.Physical(external);const int m=pr.retained.n;State result(pr.ne*block*block,0.);
    for(int e=0;e<pr.ne;++e)
    {
      BandMatrix weighted(m,m-1);State load(extra*m,0.);std::vector<State> modes(pr.weights.size());
      for(std::size_t j=0;j<pr.weights.size();++j)
      {
        const auto u=pr.Point(physical,e,pr.shapes[j]);double x=0.;for(int i=0;i<pr.n;++i)x+=pr.shapes[j][i]*pr.physics.basis.nodes[i];modes[j]=PredictorLegendre(2*x-1.,q);
        for(int i=0;i<m;++i){for(int l=0;l<m;++l)weighted.At(i,l)+=pr.weights[j]*u[0]*pr.low_shapes[j][i]*pr.low_shapes[j][l];for(int k=0;k<extra;++k)load[k*m+i]+=pr.weights[j]*pr.low_shapes[j][i]*modes[j][p+1+k];}
      }
      BandFactor factor(weighted,0,m);State velocity=load;factor.Solve(velocity,extra);const double h=pr.edges[e+1]-pr.edges[e];
      for(int k=0;k<extra;++k)for(int l=0;l<extra;++l){double value=0.;for(int i=0;i<m;++i)value+=h*load[k*m+i]*velocity[l*m+i];for(int c=0;c<2;++c)result[(e*block+2*k+c)*block+2*l+c]=value;}
    }
    return result;
  }
  // Positive physical thermal-Euler entropy Hessian pulled back through the
  // exact nonlinear reconstruction differential. This is a frozen metric,
  // not the possibly indefinite Hessian of the nonlinear manifold functional.
  State ThermalMetric(const State& external)const
  {
    const State physical=pr.Physical(external);const int m=pr.retained.n,nq=int(pr.weights.size()),nc=int(components.size()),thermal=pr.enrich_energy?q:p;State result(pr.ne*block*block,0.);
    std::vector<State> modes(nq);for(int j=0;j<nq;++j){double z=0.;for(int i=0;i<pr.n;++i)z+=pr.shapes[j][i]*pr.physics.basis.nodes[i];modes[j]=PredictorLegendre(2*z-1.,q);}
    for(int e=0;e<pr.ne;++e)
    {
      BandMatrix weighted(m,m-1);State velocity_rate(2*block*m,0.);std::vector<std::array<double,4>> base(nq);
      for(int j=0;j<nq;++j)
      {
        const auto u=pr.Point(physical,e,pr.shapes[j]);const double vx=u[1]/u[0],vy=u[2]/u[0],pressure=(gamma_euler-1.)*(u[3]-.5*u[0]*(vx*vx+vy*vy));base[j]={u[0],vx,vy,pressure};
        if(!(u[0]>floor_value&&pressure>floor_value))throw std::runtime_error("thermal predictor metric requires admitted physical state");
        for(int i=0;i<m;++i)
        {
          for(int l=0;l<m;++l)weighted.At(i,l)+=pr.weights[j]*u[0]*pr.low_shapes[j][i]*pr.low_shapes[j][l];
          for(int a=0;a<block;++a){const int component=components[a%nc],degree=p+1+a/nc;const double shape=modes[j][degree],dr=component==0?shape:0.;for(int c=0;c<2;++c){const double dm=component==c+1?shape:0.;velocity_rate[(2*a+c)*m+i]+=pr.weights[j]*pr.low_shapes[j][i]*(dm-dr*(c?vy:vx));}}
        }
      }
      BandFactor factor(weighted,0,m);factor.Solve(velocity_rate,2*block);State kinetic(block*(thermal+1),0.);std::vector<State> dx(block,State(nq)),dy(block,State(nq));
      for(int a=0;a<block;++a)for(int j=0;j<nq;++j)
      {
        for(int i=0;i<m;++i){dx[a][j]+=pr.low_shapes[j][i]*velocity_rate[2*a*m+i];dy[a][j]+=pr.low_shapes[j][i]*velocity_rate[(2*a+1)*m+i];}
        const auto u=base[j];const double dr=components[a%nc]==0?modes[j][p+1+a/nc]:0.;const double kt=.5*dr*(u[1]*u[1]+u[2]*u[2])+u[0]*(u[1]*dx[a][j]+u[2]*dy[a][j]);
        for(int k=0;k<=thermal;++k)kinetic[a*(thermal+1)+k]+=(2*k+1)*pr.weights[j]*modes[j][k]*kt;
      }
      const double h=pr.edges[e+1]-pr.edges[e];
      for(int j=0;j<nq;++j)
      {
        const auto u=base[j];State dr(block,0.),scaled_thermal(block,0.);
        for(int a=0;a<block;++a){const int component=components[a%nc];const double shape=modes[j][p+1+a/nc];dr[a]=component==0?shape:0.;double K=0.;for(int k=0;k<=thermal;++k)K+=modes[j][k]*kinetic[a*(thermal+1)+k];const double dp=(gamma_euler-1.)*((component==3?shape:0.)-K);scaled_thermal[a]=dp/u[3]-dr[a]/u[0];}
        for(int a=0;a<block;++a)for(int b=0;b<block;++b)result[(e*block+a)*block+b]+=h*pr.weights[j]*(dr[a]*dr[b]/u[0]+u[0]/(gamma_euler-1.)*scaled_thermal[a]*scaled_thermal[b]+u[0]*u[0]/u[3]*(dx[a][j]*dx[b][j]+dy[a][j]*dy[b][j]));
      }
    }
    return result;
  }
  State Metric(const State& external)const
  {return components.size()==2?KineticMetric(external):ThermalMetric(external);}
  std::string MetricKind()const
  {return components.size()==2?"frozen-density kinetic velocity tangent":"physical thermal-Euler entropy metric on reconstruction tangent";}
};
void PredictorAdmit(const SignedEnrichmentProblem& pr,const State& external,const State& sigma,PredictorWork& work)
{
  const State physical=pr.Physical(external);
  for(int e=0;e<pr.ne;++e)for(const State& shape:pr.positive)
  {
    const auto u=pr.Point(physical,e,shape);double s=0.;for(int i=0;i<pr.n;++i)s+=shape[i]*sigma[e*pr.n+i];
    const double pressure=(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]),Q=gamma_euler*pressure+(gamma_euler-1.)*s;
    ++work.admission_points;if(!(u[0]>floor_value&&pressure>floor_value&&Q>floor_value)||!std::isfinite(u[0])||!std::isfinite(pressure)||!std::isfinite(Q))throw std::runtime_error("signed predictor perturbation is inadmissible; no limiter applied");
    work.density_min=std::min(work.density_min,u[0]);work.pressure_min=std::min(work.pressure_min,pressure);work.q_min=std::min(work.q_min,Q);
  }
}
State PredictorRHS(const SignedEnrichmentProblem& pr,const State& u,const State& sigma,double time,PredictorWork& work)
{PredictorAdmit(pr,u,sigma,work);std::array<double,4> budget{};State rhs=pr.Residual(u,sigma,time,work.shared,budget);++work.rhs_calls;return rhs;}
struct PredictorTangent{State full,fixed,stress;double direction_max=0.;};
PredictorTangent PredictorDifferentiate(const SignedEnrichmentProblem& pr,const PredictorCoordinates& coordinates,const State& u,const State& sigma,const State& direction,double time,double epsilon,PredictorWork& work)
{
  PredictorTangent out;out.direction_max=PredictorMaximum(direction);out.full.assign(coordinates.size,0.);out.fixed=out.full;out.stress.assign(pr.ne*pr.stress.n,0.);if(out.direction_max==0.)return out;
  const double step=epsilon/out.direction_max;const State plus=PredictorAdd(u,direction,step),minus=PredictorAdd(u,direction,-step);
  const State sp=pr.Closure(plus,time,work.shared),sm=pr.Closure(minus,time,work.shared);
  const State fp=coordinates.Extract(PredictorRHS(pr,plus,sp,time,work)),fm=coordinates.Extract(PredictorRHS(pr,minus,sm,time,work));
  const State gp=coordinates.Extract(PredictorRHS(pr,plus,sigma,time,work)),gm=coordinates.Extract(PredictorRHS(pr,minus,sigma,time,work));
  out.full=PredictorSubtract(fp,fm,2*step);out.fixed=PredictorSubtract(gp,gm,2*step);out.stress=pr.stress_transfer.Apply(pr.stress_transfer.P,PredictorSubtract(sp,sm,2*step),pr.ne,1);return out;
}
State PredictorMultiply(const State& matrix,const State& vector)
{const int n=int(vector.size());if(matrix.size()!=std::size_t(n*n))throw std::runtime_error("predictor matrix shape mismatch");State a(n,0.);for(int i=0;i<n;++i)for(int j=0;j<n;++j)a[i]+=matrix[i*n+j]*vector[j];return a;}
void PredictorSelfTest()
{
  MemoryOptions mo;auto& b=mo.recovery.boundary;auto& t=b.compatibility.track;mo.recovery.recovery="existing";b.left=b.right="outflow";b.stress="trace";t.initial="entropy";t.mode="uniform";t.left=-1.;t.right=1.;t.base.elements=8;t.base.order=15;t.base.alpha=.04;t.base.final_time=.01;t.quad_extra=19;t.closure_space="l2";t.closure_boundary="natural";mo.retained_order=mo.stress_order=5;
  SignedEnrichmentProblem pr(mo,3,.25,.25);PredictorGeometry(pr,State{-1.,-.85,-.6,-.3,0.,.2,.45,.7,1.});PredictorCoordinates coordinates(pr,3);State z(coordinates.size);for(std::size_t i=0;i<z.size();++i)z[i]=.03*std::sin(double(i+1));const State lift=coordinates.Embed(z),back=coordinates.Extract(lift);const double roundtrip=PredictorRelative(back,z);const auto means=pr.Integrals(lift,pr.edges);double mean=0.;for(double x:means)mean=std::max(mean,std::abs(x));
  // Nonzero signed tails and unequal face states make the evolving-coarse
  // split substantive. A globally affine fixture has exact zero tail rates,
  // so its relative split is only a roundoff/roundoff ratio.
  State low(pr.ne*pr.retained.n*4,0.);for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.retained.n;++i){const double z0=pr.retained.physics.basis.nodes[i],x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*z0,rho=1.2+.03*x,v=.1+.02*x;const State modes=PredictorLegendre(2*z0-1.,5);const int j=(e*pr.retained.n+i)*4;low[j]=rho;low[j+1]=rho*v+.004*(1.+.1*e)*modes[5];low[j+2]=.04*rho+.003*modes[4];low[j+3]=3.+.5*rho*(v*v+.04*.04);}
  State u=pr.state_transfer.Apply(pr.state_transfer.E,low,pr.ne,4);u=pr.RestrictExternal(u);PredictorWork work;const State sigma=pr.Closure(u,0.,work.shared),rhs=PredictorRHS(pr,u,sigma,0.,work);const auto d1=PredictorDifferentiate(pr,coordinates,u,sigma,lift,0.,1.e-5,work),d2=PredictorDifferentiate(pr,coordinates,u,sigma,lift,0.,5.e-6,work);const auto coarse=coordinates.Coarse(rhs),tail=coordinates.Embed(coordinates.Extract(rhs));const auto dt=PredictorDifferentiate(pr,coordinates,u,sigma,rhs,0.,2.e-6,work),dc=PredictorDifferentiate(pr,coordinates,u,sigma,coarse,0.,2.e-6,work),dz=PredictorDifferentiate(pr,coordinates,u,sigma,tail,0.,2.e-6,work);
  const double split=PredictorRelative(dt.full,PredictorAdd(dc.full,dz.full,1.)),epsilon=PredictorRelative(d1.full,d2.full);const State metric=coordinates.KineticMetric(u);double symmetry=0.,diagonal=INFINITY;for(int e=0;e<pr.ne;++e)for(int i=0;i<coordinates.block;++i){diagonal=std::min(diagonal,metric[(e*coordinates.block+i)*coordinates.block+i]);for(int j=0;j<coordinates.block;++j)symmetry=std::max(symmetry,std::abs(metric[(e*coordinates.block+i)*coordinates.block+j]-metric[(e*coordinates.block+j)*coordinates.block+i]));}
  SignedEnrichmentProblem all(mo,3,.25,.25,"original",true,true);PredictorGeometry(all,pr.edges);PredictorCoordinates full_coordinates(all,3);State full_z(full_coordinates.size);for(std::size_t i=0;i<full_z.size();++i)full_z[i]=.02*std::sin(.31*(i+1));const State full_direction=full_coordinates.Embed(full_z),full_metric=full_coordinates.Metric(u);const double all_roundtrip=PredictorRelative(full_coordinates.Extract(full_direction),full_z);double predicted=0.;
  for(int e=0;e<all.ne;++e)for(int i=0;i<full_coordinates.block;++i)for(int j=0;j<full_coordinates.block;++j)predicted+=full_z[e*full_coordinates.block+i]*full_metric[(e*full_coordinates.block+i)*full_coordinates.block+j]*full_z[e*full_coordinates.block+j];
  const double perturbation=1.e-5;const State physical=all.Physical(u),positive=all.Physical(PredictorAdd(u,full_direction,perturbation)),negative=all.Physical(PredictorAdd(u,full_direction,-perturbation));double independent=0.;
  for(int e=0;e<all.ne;++e)for(std::size_t j=0;j<all.weights.size();++j)
  {
    const auto a=all.Point(physical,e,all.shapes[j]),plus=all.Point(positive,e,all.shapes[j]),minus=all.Point(negative,e,all.shapes[j]);std::array<double,4> d{};for(int c=0;c<4;++c)d[c]=(plus[c]-minus[c])/(2*perturbation);const double vx=a[1]/a[0],vy=a[2]/a[0],pressure=(gamma_euler-1.)*(a[3]-.5*a[0]*(vx*vx+vy*vy)),dvx=(d[1]-vx*d[0])/a[0],dvy=(d[2]-vy*d[0])/a[0],dp=(gamma_euler-1.)*(d[3]-vx*d[1]-vy*d[2]+.5*(vx*vx+vy*vy)*d[0]),thermal=dp/pressure-d[0]/a[0];
    independent+=(all.edges[e+1]-all.edges[e])*all.weights[j]*(d[0]*d[0]/a[0]+a[0]/(gamma_euler-1.)*thermal*thermal+a[0]*a[0]/pressure*(dvx*dvx+dvy*dvy));
  }
  const double thermal_metric_error=std::abs(predicted-independent)/std::max(predicted,1.e-12);
  std::cout<<std::setprecision(17)<<"{\"tail_roundtrip\":"<<roundtrip<<",\"tail_integral\":"<<mean<<",\"tangent_epsilon_difference\":"<<epsilon<<",\"evolving_coarse_split\":"<<split<<",\"evolving_coarse_split_absolute\":"<<Norm(PredictorSubtract(dt.full,PredictorAdd(dc.full,dz.full,1.)))<<",\"total_tail_tangent_norm\":"<<Norm(dt.full)<<",\"coarse_tail_tangent_norm\":"<<Norm(dc.full)<<",\"tail_tail_tangent_norm\":"<<Norm(dz.full)<<",\"kinetic_metric_symmetry\":"<<symmetry<<",\"minimum_kinetic_metric_diagonal\":"<<diagonal<<",\"all_field_tail_roundtrip\":"<<all_roundtrip<<",\"physical_thermal_metric_independent_differential\":"<<thermal_metric_error<<",\"physical_thermal_metric_norm_squared\":"<<predicted<<",\"rhs_conservation\":"<<work.shared.max_rhs_defect<<"}\n";
  if(roundtrip>2.e-11||mean>1.e-11||epsilon>2.e-4||split>2.e-4||symmetry>1.e-11||!(diagonal>0.)||work.shared.max_rhs_defect>1.e-8||all_roundtrip>2.e-11||thermal_metric_error>1.e-7||!(predicted>0.))throw std::runtime_error("signed predictor projector/tangent/metric test failed");
}
}
#ifndef DG_IGR_SIGNED_PREDICTOR_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    const auto process=Clock::now();double slope=.25,curvature=.25,time=0.,epsilon=1.e-6;int extra=2;bool enrich_density=false,enrich_energy=false;std::string input,edges,volume="original";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--enrich-density"){enrich_density=true;continue;}if(flag=="--enrich-energy"){enrich_energy=true;continue;}if(flag=="--extra-order"||flag=="--slope-strength"||flag=="--curvature-strength"||flag=="--probe-time"||flag=="--moments-in"||flag=="--external-moments-in"||flag=="--edges-in"||flag=="--epsilon"||flag=="--entropy-volume")
      {if(++i==argc)throw std::runtime_error("missing signed predictor option");const std::string value=argv[i];if(flag=="--extra-order")extra=std::stoi(value);else if(flag=="--slope-strength")slope=std::stod(value);else if(flag=="--curvature-strength")curvature=std::stod(value);else if(flag=="--probe-time")time=std::stod(value);else if(flag=="--moments-in"||flag=="--external-moments-in")input=value;else if(flag=="--edges-in")edges=value;else if(flag=="--epsilon")epsilon=std::stod(value);else volume=value;}
      else common.push_back(argv[i]);
    }
    MemoryOptions mo;mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;if(t.self_test){PredictorSelfTest();return 0;}
    const int p=t.base.order,q=p+extra;mo.retained_order=mo.stress_order=q;t.base.order=3*q;mo.input=input;mo.start_time=time;
    if(extra<1||extra>8||q>35||!(epsilon>0.)||epsilon>1.e-3||!std::isfinite(epsilon)||!std::isfinite(time)||time<0.||time>t.base.final_time||input.empty()||edges.empty())throw std::runtime_error("predictor requires rich external moments, saved edges, and valid epsilon/time");
    if(mo.recovery.boundary.scenario!="application-tanh"||mo.recovery.boundary.left!="nonreflecting"||mo.recovery.boundary.right!="fixed")throw std::runtime_error("predictor time derivative is qualified only for autonomous application-tanh boundaries");
    const auto setup_start=Clock::now();SignedEnrichmentProblem pr(mo,p,slope,curvature,volume,enrich_density,enrich_energy);PredictorGeometry(pr,PredictorRead(edges,pr.ne+1));PredictorCoordinates coordinates(pr,p);const State low=PredictorRead(input,pr.ne*pr.retained.n*4);const State u=pr.state_transfer.Apply(pr.state_transfer.E,low,pr.ne,4);const double restriction=PredictorRelative(u,pr.RestrictExternal(u));if(restriction>2.e-10)throw std::runtime_error("input rho/E contain disallowed high conservative moments");const State metric=coordinates.Metric(u);const double setup=Wall(setup_start);
    PredictorWork work;const State sigma=pr.Closure(u,time,work.shared),rhs=PredictorRHS(pr,u,sigma,time,work),tail_rhs=coordinates.Extract(rhs),tail=coordinates.Extract(u);const int n=coordinates.size;
    State full(n*n,0.),fixed(n*n,0.),stress_tangent(pr.ne*pr.stress.n*n,0.);const auto matrix_start=Clock::now();
    for(int j=0;j<n;++j){State column(n,0.);column[j]=1.;const auto derivative=PredictorDifferentiate(pr,coordinates,u,sigma,coordinates.Embed(column),time,epsilon,work);for(int i=0;i<n;++i){full[i*n+j]=derivative.full[i];fixed[i*n+j]=derivative.fixed[i];}for(int i=0;i<pr.ne*pr.stress.n;++i)stress_tangent[i*n+j]=derivative.stress[i];}
    const double matrix_seconds=Wall(matrix_start);const auto audit_start=Clock::now();const State coarse_rhs=coordinates.Coarse(rhs),tail_direction=coordinates.Embed(tail_rhs);
    const auto dc=PredictorDifferentiate(pr,coordinates,u,sigma,coarse_rhs,time,epsilon,work),dt=PredictorDifferentiate(pr,coordinates,u,sigma,rhs,time,epsilon,work),dz=PredictorDifferentiate(pr,coordinates,u,sigma,tail_direction,time,epsilon,work);
    const auto dc_half=PredictorDifferentiate(pr,coordinates,u,sigma,coarse_rhs,time,.5*epsilon,work),dt_half=PredictorDifferentiate(pr,coordinates,u,sigma,rhs,time,.5*epsilon,work),dz_half=PredictorDifferentiate(pr,coordinates,u,sigma,tail_direction,time,.5*epsilon,work);
    const State matrix_tail_full=PredictorMultiply(full,tail_rhs),matrix_tail_fixed=PredictorMultiply(fixed,tail_rhs);const double audit_seconds=Wall(audit_start);
    const double split_full=PredictorRelative(dt.full,PredictorAdd(dc.full,dz.full,1.)),split_fixed=PredictorRelative(dt.fixed,PredictorAdd(dc.fixed,dz.fixed,1.));
    const bool verified=std::max({split_full,split_fixed,PredictorRelative(matrix_tail_full,dz.full),PredictorRelative(matrix_tail_fixed,dz.fixed),PredictorRelative(dt.full,dt_half.full),PredictorRelative(dt.fixed,dt_half.fixed),PredictorRelative(dc.full,dc_half.full),PredictorRelative(dc.fixed,dc_half.fixed),PredictorRelative(dz.full,dz_half.full),PredictorRelative(dz.fixed,dz_half.fixed)})<=2.e-3;
    std::cout<<std::setprecision(17)<<"{\"schema\":1,\"completed\":true,\"kind\":\"frozen-signed-tail-tangent\",\"continuum\":\"full-energy Euler with instantaneous scalar IGR\",\"arithmetic\":\"FP64 CPU\",\"elements\":"<<pr.ne<<",\"base_order\":"<<p<<",\"rich_order\":"<<q<<",\"physical_order\":"<<pr.p<<",\"tail_block_size\":"<<coordinates.block<<",\"tail_size\":"<<n<<",\"time\":"<<time<<",\"alpha\":"<<t.base.alpha<<",\"epsilon_max_high_coefficient_perturbation\":"<<epsilon<<",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"entropy_volume\":";PredictorString(std::cout,volume);
    std::cout<<",\"enrich_density\":"<<(enrich_density?"true":"false")<<",\"enrich_energy\":"<<(enrich_energy?"true":"false")<<",\"tail_layout\":\"cell, Legendre degrees p+1..q, listed tail_components; reference [-1,1]\",\"matrix_layout\":\"row-major tail_size squared\",\"metric_layout\":\"cell, row, column\",\"metric_kind\":";PredictorString(std::cout,coordinates.MetricKind());std::cout<<",\"input_layout\":\"cell, rich retained node, rho/mx/my/E; only unselected density/energy restricted to base Pp\",\"stress_order\":"<<pr.stress.p<<",\"effective_sipg_factor\":"<<pr.o.sipg_penalty*pr.n*pr.n<<",\"quad_extra\":"<<t.quad_extra<<",\"quad_points\":"<<pr.weights.size()<<",\"source_faces\":\"central\",\"transport_faces\":\"LLF\",\"closure_boundary\":\"natural L2/SIPG\",\"scenario\":";PredictorString(std::cout,mo.recovery.boundary.scenario);
    std::cout<<",\"left_boundary\":";PredictorString(std::cout,mo.recovery.boundary.left);std::cout<<",\"right_boundary\":";PredictorString(std::cout,mo.recovery.boundary.right);
    auto array=[&](const char* name,const State& values){std::cout<<",\""<<name<<"\":";TrackArray(std::cout,values);};
    array("edges",pr.edges);array("rich_nodes",State(pr.retained.physics.basis.nodes.GetData(),pr.retained.physics.basis.nodes.GetData()+pr.retained.n));array("physical_nodes",State(pr.physics.basis.nodes.GetData(),pr.physics.basis.nodes.GetData()+pr.n));array("input_moments",low);array("physical_coefficients",pr.Physical(u));array("sigma",pr.stress_transfer.Apply(pr.stress_transfer.P,sigma,pr.ne,1));array("base_rhs",pr.state_transfer.Apply(pr.state_transfer.P,rhs,pr.ne,4));array("tail",tail);array("tail_rhs",tail_rhs);array("tail_components",State(coordinates.components.begin(),coordinates.components.end()));array("tail_metric",metric);if(coordinates.components.size()==2)array("tail_kinetic_metric",metric);array("full_tail_matrix",full);array("fixed_tail_matrix",fixed);array("stress_tail_matrix",stress_tangent);array("coarse_full_tangent",dc.full);array("coarse_fixed_tangent",dc.fixed);array("total_full_tangent",dt.full);array("total_fixed_tangent",dt.fixed);array("tail_full_tangent",dz.full);array("tail_fixed_tangent",dz.fixed);
    std::cout<<",\"checks_passed\":"<<(verified?"true":"false")<<",\"checks\":{\"input_restriction\":"<<restriction<<",\"full_evolving_coarse_split\":"<<split_full<<",\"fixed_evolving_coarse_split\":"<<split_fixed<<",\"matrix_full_direction\":"<<PredictorRelative(matrix_tail_full,dz.full)<<",\"matrix_fixed_direction\":"<<PredictorRelative(matrix_tail_fixed,dz.fixed)<<",\"coarse_epsilon\":"<<PredictorRelative(dc.full,dc_half.full)<<",\"total_epsilon\":"<<PredictorRelative(dt.full,dt_half.full)<<",\"tail_epsilon\":"<<PredictorRelative(dz.full,dz_half.full)<<",\"coarse_fixed_epsilon\":"<<PredictorRelative(dc.fixed,dc_half.fixed)<<",\"total_fixed_epsilon\":"<<PredictorRelative(dt.fixed,dt_half.fixed)<<",\"tail_fixed_epsilon\":"<<PredictorRelative(dz.fixed,dz_half.fixed)<<",\"maximum_rhs_conservation_defect\":"<<work.shared.max_rhs_defect<<",\"maximum_closure_residual\":"<<work.shared.max_closure_residual<<"}";
    std::cout<<",\"admission\":{\"density_min\":"<<work.density_min<<",\"pressure_min\":"<<work.pressure_min<<",\"Q_min\":"<<work.q_min<<",\"points\":"<<work.admission_points<<",\"limiter_calls\":0},\"work\":{\"setup_seconds\":"<<setup<<",\"matrix_seconds\":"<<matrix_seconds<<",\"audit_seconds\":"<<audit_seconds<<",\"total_before_output_end_seconds\":"<<Wall(process)<<",\"closure_calls\":"<<work.shared.closure_calls<<",\"rhs_calls\":"<<work.rhs_calls<<",\"closure_seconds\":"<<work.shared.closure<<",\"transport_seconds\":"<<work.shared.transport<<",\"flux_points\":"<<work.shared.flux_points<<"}}\n";
    if(!verified){std::cerr<<"predictor finite-difference linearity/convergence is unresolved at selected epsilon\n";return 2;}return 0;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif

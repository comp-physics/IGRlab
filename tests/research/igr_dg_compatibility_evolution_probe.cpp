// Test-only fixed-mesh DG/IGR compatibility controls. No production defaults.
// Alternating LDG G uses the left velocity trace and the opposite stress trace.
// G, its density-weighted adjoint square, and the current quadratic source are
// assembled together. These are discretization changes, not algebraic parity
// claims with SIPG. Euler LLF dissipation remains in the shared base residual.
#define DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY
#include "igr_hp_front_tracking_probe.cpp"
#undef DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY

namespace
{
struct CompatibilityOptions
{
  TrackOptions track;
  std::string formulation="baseline",direction="left",energy_trace="physical";
  // Optional test-only physical-wall traces for the velocity gradient. The
  // opposite pressure gradient retains its natural boundary treatment.
  std::array<bool,2> homogeneous_velocity_boundary{{false,false}};
};
struct CompatibilityProblem:TrackingProblem
{
  CompatibilityOptions compatibility;
  mutable std::uint64_t velocity_projections=0,gradient_points=0;
  mutable double projection_seconds=0.,max_mean_defect=0.;
  explicit CompatibilityProblem(const CompatibilityOptions& input):TrackingProblem(input.track),compatibility(input)
  {
    if(o.mode!="uniform"||o.closure_space!="l2"||o.closure_boundary!="natural")
      throw std::runtime_error("compatibility probe requires uniform L2 natural closure");
  }
  State Velocity(const State& state)const
  {
    const auto start=Clock::now();State result(ns,0.);
    for(int e=0;e<ne;++e)
    {
      State load(n,0.);for(std::size_t q=0;q<weights.size();++q)
      {const auto value=Point(state,e,shapes[q]);physics.CheckState(value.data());
        for(int i=0;i<n;++i)load[i]+=weights[q]*shapes[q][i]*value[1]/value[0];}
      for(int i=0;i<n;++i)for(int j=0;j<n;++j)result[e*n+i]+=mass_inverse(i,j)*load[j];
      ++velocity_projections;
    }
    projection_seconds+=Wall(start);return result;
  }
  void GradientRow(int e,std::size_t q,std::vector<int>& index,State& values,bool opposite=false)const
  {
    const double h=edges[e+1]-edges[e];index.resize(n);values.resize(n);
    for(int i=0;i<n;++i){index[i]=e*n+i;values[i]=derivatives[q][i]/h;}
    const bool left=(compatibility.direction=="left")!=opposite;
    if(!((left&&e==0)||(!left&&e==ne-1)))
    {
      const int endpoint=left?0:p;double lift=0.;
      for(int i=0;i<n;++i)lift+=shapes[q][i]*mass_inverse(i,endpoint)/h;
      const double sign=left?1.:-1.;values[endpoint]+=sign*lift;
      index.push_back(left?e*n-1:(e+1)*n);values.push_back(-sign*lift);
    }
    if(!opposite)for(int side=0;side<2;++side)
    {
      if(!compatibility.homogeneous_velocity_boundary[side]||e!=(side?ne-1:0))continue;
      const int endpoint=side?p:0;double lift=0.;
      for(int i=0;i<n;++i)lift+=shapes[q][i]*mass_inverse(i,endpoint)/h;
      values[endpoint]+=(side?-1.:1.)*lift;
    }
  }
  State Closure(const State& state,double time,TrackWork& work,
                const std::array<double,2>& robin={{0.,0.}},
                const std::array<double,2>& neumann={{0.,0.}})const
  {
    const bool boundary_load=robin[0]!=0.||robin[1]!=0.||neumann[0]!=0.||neumann[1]!=0.;
    if(compatibility.formulation=="baseline")
    {
      if(!boundary_load)return TrackingProblem::Closure(state,time,work);
      const auto start=Clock::now();BandMatrix a;State b;AssembleClosure(state,time,work,a,b);
      return SolveCurrentSystem(state,a,b,work,start,robin,neumann);
    }
    if(compatibility.formulation=="weak-product")
    {if(boundary_load)throw std::runtime_error("weak product boundary-load control not implemented");return WeakProductClosure(state,time,work);}
    const auto start=Clock::now();const State velocity=Velocity(state);
    const bool mixed=compatibility.formulation=="mixed"||compatibility.formulation=="paired";
    BandMatrix a(ns,n);State b(ns,0.);
    if(!mixed){AssembleClosure(state,time,work,a,b);std::fill(b.begin(),b.end(),0.);}
    double source_integral=0.;std::vector<int> index,diffusion_index;State gradient,diffusion_gradient;
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];
      for(std::size_t q=0;q<weights.size();++q)
      {
        const auto value=Point(state,e,shapes[q]);physics.CheckState(value.data());
        const double ri=1./value[0],weight=h*weights[q];
        GradientRow(e,q,index,gradient);double g=0.;
        for(std::size_t j=0;j<index.size();++j)g+=gradient[j]*velocity[index[j]];
        const double source=2.*o.base.alpha*g*g;source_integral+=weight*source;
        ++gradient_points;++work.source_points;
        for(int i=0;i<n;++i)b[e*n+i]+=weight*shapes[q][i]*source;
        if(mixed)
        {
          for(int i=0;i<n;++i)for(int j=0;j<n;++j)a.At(e*n+i,e*n+j)+=weight*ri*shapes[q][i]*shapes[q][j];
          // The stress gradient has the opposite LDG trace. Thus divergence
          // D^- of acceleration -rho^-1 D^+sigma produces the intended
          // positive screened diffusion. Same-direction squaring would still
          // be SPD, but would not pair this source with its pressure force.
          GradientRow(e,q,diffusion_index,diffusion_gradient,true);
          for(std::size_t i=0;i<diffusion_index.size();++i)for(std::size_t j=0;j<diffusion_index.size();++j)
            a.At(diffusion_index[i],diffusion_index[j])+=weight*ri*o.base.alpha*diffusion_gradient[i]*diffusion_gradient[j];
        }
      }
    }
    if(mixed)++work.closure_assemblies;
    work.current_source_integral=source_integral;
    return SolveCurrentSystem(state,a,b,work,start,robin,neumann);
  }
  State SolveCurrentSystem(const State& state,BandMatrix& a,State& b,TrackWork& work,
                           const Clock::time_point& start,const std::array<double,2>& robin,
                           const std::array<double,2>& neumann)const
  {
    for(int side=0;side<2;++side)
    {const int i=side?ns-1:0;a.At(i,i)+=robin[side];b[i]+=neumann[side];}
    BandFactor factor(a,0,ns);State sigma=b;factor.Solve(sigma,1);
    State residual(ns,0.);for(int i=0;i<ns;++i)
    {residual[i]=-b[i];for(int j=std::max(0,i-a.k);j<std::min(ns,i+a.k+1);++j)residual[i]+=a.Get(i,j)*sigma[j];}
    const double relative=Norm(residual)/std::max(Norm(b),1.e-12);
    work.max_closure_residual=std::max(work.max_closure_residual,relative);
    if(!std::isfinite(relative)||relative>1.e-8)throw std::runtime_error("compatibility current closure residual failed");
    double mean=0.;for(int e=0;e<ne;++e)for(std::size_t q=0;q<weights.size();++q)
    {const auto value=Point(state,e,shapes[q]);double sq=0.;for(int i=0;i<n;++i)sq+=shapes[q][i]*sigma[e*n+i];mean+=(edges[e+1]-edges[e])*weights[q]*sq/value[0];}
    const double balance=mean+robin[0]*sigma.front()+robin[1]*sigma.back()-neumann[0]-neumann[1];
    max_mean_defect=std::max(max_mean_defect,std::abs(balance-work.current_source_integral)/
      std::max({std::abs(work.current_source_integral),std::abs(neumann[0])+std::abs(neumann[1]),1.e-12}));
    ++work.closure_calls;work.closure+=Wall(start);return sigma;
  }
  State WeakProductClosure(const State& state,double time,TrackWork& work)const
  {
    const auto start=Clock::now();BandMatrix a;State b;AssembleClosure(state,time,work,a,b);
    const State dl=Shape(1.,true),dr=Shape(0.,true);double face_integral=0.;
    for(int face=1;face<ne;++face)
    {
      const int el=face-1,er=face;const double hl=edges[el+1]-edges[el],hr=edges[er+1]-edges[er];
      const double rl=state[Id(el,p,0)],rr=state[Id(er,0,0)];
      const double vl=state[Id(el,p,1)]/rl,vr=state[Id(er,0,1)]/rr;
      double gl=0.,gr=0.;for(int i=0;i<n;++i)
      {gl+=dl[i]*(state[Id(el,i,1)]-vl*state[Id(el,i,0)])/(hl*rl);
        gr+=dr[i]*(state[Id(er,i,1)]-vr*state[Id(er,i,0)])/(hr*rr);}
      const double jump=vl-vr,average=.5*(gl+gr);
      const double penalty=o.sipg_penalty*n*n*.5*(1./hl+1./hr);
      const double value=o.base.alpha*(2.*penalty*jump*jump-4.*average*jump);
      b[el*n+p]+=.5*value;b[er*n]+=.5*value;face_integral+=value;
      for(int i=0;i<n;++i)
      {b[el*n+i]-=.5*o.base.alpha*jump*jump*dl[i]/hl;
        b[er*n+i]+=.5*o.base.alpha*jump*jump*dr[i]/hr;}
    }
    work.current_source_integral+=face_integral;BandFactor factor(a,0,ns);State sigma=b;factor.Solve(sigma,1);
    State residual(ns,0.);for(int i=0;i<ns;++i)
    {residual[i]=-b[i];for(int j=std::max(0,i-a.k);j<std::min(ns,i+a.k+1);++j)residual[i]+=a.Get(i,j)*sigma[j];}
    const double relative=Norm(residual)/std::max(Norm(b),1.e-12);work.max_closure_residual=std::max(work.max_closure_residual,relative);
    if(!std::isfinite(relative)||relative>1.e-8)throw std::runtime_error("weak product current closure residual failed");
    ++work.closure_calls;work.closure+=Wall(start);return sigma;
  }
  State Residual(const State& state,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    State result=TrackingProblem::Residual(state,sigma,time,work,budget);
    if(compatibility.formulation!="paired")return result;
    const auto start=Clock::now();State projected;
    if(compatibility.energy_trace=="projected")projected=Velocity(state);
    for(int face=1;face<ne;++face)
    {
      const int el=face-1,er=face;const double sl=sigma[el*n+p],sr=sigma[er*n];
      const double vl=state[Id(el,p,1)]/state[Id(el,p,0)],vr=state[Id(er,0,1)]/state[Id(er,0,0)];
      const bool left=compatibility.direction=="left";const double stress=left?sr:sl;
      double velocity=left?vl:vr;
      if(!projected.empty())velocity=left?projected[el*n+p]:projected[er*n];
      const double dm=stress-.5*(sl+sr),de=stress*velocity-.5*(sl*vl+sr*vr);
      const double hl=edges[el+1]-edges[el],hr=edges[er+1]-edges[er];
      for(int i=0;i<n;++i)
      {result[Id(el,i,1)]-=mass_inverse(i,p)*dm/hl;result[Id(er,i,1)]+=mass_inverse(i,0)*dm/hr;
        result[Id(el,i,3)]-=mass_inverse(i,p)*de/hl;result[Id(er,i,3)]+=mass_inverse(i,0)*de/hr;}
    }
    const auto integrals=Integrals(result,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(integrals[c]-budget[c]));
    work.transport+=Wall(start);return result;
  }
};
template<class ProblemType>
TrackResult CompatibilityRun(ProblemType& pr)
{
  TrackResult out;const auto start=Clock::now();
  try
  {
    out.u=pr.Initial();pr.Limit(out.u,out.work);out.initial_integrals=pr.Integrals(out.u,pr.edges);out.sigma=pr.Closure(out.u,0.,out.work);
    while(out.time<pr.o.base.final_time-1.e-14)
    {
      ++out.work.attempted;const double dt=std::min(pr.o.base.dt,pr.o.base.final_time-out.time);const State original=out.u;State stage=original;
      std::array<std::array<double,4>,3> budgets{};const double times[3]={out.time,out.time+dt,out.time+.5*dt};
      for(int k=0;k<3;++k)
      {const State sigma=k?pr.Closure(stage,times[k],out.work):out.sigma;const State rhs=pr.Residual(stage,sigma,times[k],out.work,budgets[k]);
        for(std::size_t j=0;j<stage.size();++j){const double candidate=stage[j]+dt*rhs[j];stage[j]=k==0?candidate:(k==1?.75*original[j]+.25*candidate:original[j]/3.+2.*candidate/3.);}
        pr.Limit(stage,out.work);}
      out.u=std::move(stage);out.time+=dt;++out.work.steps;
      for(int c=0;c<4;++c)out.boundary_budget[c]+=dt*(budgets[0][c]/6.+budgets[1][c]/6.+2.*budgets[2][c]/3.);
      out.sigma=pr.Closure(out.u,out.time,out.work);
    }
  }catch(const std::exception& error){out.failure=error.what();}
  if(!out.u.empty()){const auto totals=pr.Integrals(out.u,pr.edges);for(int c=0;c<4;++c)out.conservation_defect[c]=totals[c]-out.initial_integrals[c]-out.boundary_budget[c];}
  out.wall=Wall(start);return out;
}
CompatibilityOptions CompatibilityParse(int argc,char** argv)
{
  CompatibilityOptions out;std::vector<char*> common{argv[0]};
  for(int i=1;i<argc;++i)
  {const std::string flag=argv[i];if(flag=="--formulation"||flag=="--gradient-direction"||flag=="--energy-trace")
    {if(++i>=argc)throw std::runtime_error("missing compatibility option");if(flag=="--formulation")out.formulation=argv[i];else if(flag=="--gradient-direction")out.direction=argv[i];else out.energy_trace=argv[i];}
    else common.push_back(argv[i]);}
  out.track=TrackParse(int(common.size()),common.data());
  if(out.formulation!="baseline"&&out.formulation!="weak-product"&&out.formulation!="lifted-source"&&out.formulation!="mixed"&&out.formulation!="paired")throw std::runtime_error("invalid compatibility formulation");
  if(out.direction!="left"&&out.direction!="right")throw std::runtime_error("invalid gradient direction");
  if(out.energy_trace!="physical"&&out.energy_trace!="projected")throw std::runtime_error("invalid energy trace");return out;
}
void CompatibilitySelfTest()
{
  for(int degree:{3,7,11})for(const std::string direction:{"left","right"})
  {
    CompatibilityOptions o;o.track.initial="entropy";o.track.mode="uniform";o.track.closure_space="l2";o.track.closure_boundary="natural";
    o.track.base.elements=8;o.track.base.order=degree;o.track.base.final_time=0.;o.direction=direction;o.formulation="paired";CompatibilityProblem pr(o);TrackWork work;
    State state=pr.Initial(),v=pr.Velocity(state);std::vector<int> index;State gradient;
    for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
    {pr.GradientRow(e,q,index,gradient);double sum=0.,value=0.;for(std::size_t j=0;j<index.size();++j){sum+=gradient[j];value+=gradient[j]*v[index[j]];}
      if(std::abs(sum)>2.e-10||std::abs(value)>2.e-10)throw std::runtime_error("alternating gradient constant nullspace failed");}
    // Independently integrate adjoint pairing for discontinuous polynomials.
    State test_v(pr.ns),test_s(pr.ns);for(int i=0;i<pr.ns;++i){test_v[i]=std::sin(.7*i);test_s[i]=std::cos(.3*i);}
    double product=0.;for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
    {double vq=0.,sq=0.,dv=0.,ds=0.;for(int i=0;i<pr.n;++i){vq+=pr.shapes[q][i]*test_v[e*pr.n+i];sq+=pr.shapes[q][i]*test_s[e*pr.n+i];}
      pr.GradientRow(e,q,index,gradient);for(std::size_t j=0;j<index.size();++j)dv+=gradient[j]*test_v[index[j]];
      pr.GradientRow(e,q,index,gradient,true);for(std::size_t j=0;j<index.size();++j)ds+=gradient[j]*test_s[index[j]];
      product+=(pr.edges[e+1]-pr.edges[e])*pr.weights[q]*(sq*dv+vq*ds);}
    const double boundary=test_v.back()*test_s.back()-test_v.front()*test_s.front();
    if(std::abs(product-boundary)>1.e-9)throw std::runtime_error("alternating gradient adjoint face/boundary identity failed");
    const State zero=pr.Closure(state,0.,work);if(Norm(zero)>1.e-10)throw std::runtime_error("constant velocity source failed");
    // Nonzero affine velocity: independent analytic constant source/solution,
    // including natural boundaries and every interior lifting cancellation.
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i],velocity=.2+.1*x;
      state[pr.Id(e,i,0)]=2.;state[pr.Id(e,i,1)]=2.*velocity;state[pr.Id(e,i,2)]=0.;state[pr.Id(e,i,3)]=3.+velocity*velocity;}
    const State sigma=pr.Closure(state,0.,work);const double exact=4.*o.track.base.alpha*.01;
    for(double value:sigma)if(std::abs(value-exact)>1.e-10)throw std::runtime_error("affine velocity mixed closure failed");
    std::array<double,4> budget{};pr.Residual(state,sigma,0.,work,budget);
    if(work.max_rhs_defect>1.e-9||pr.max_mean_defect>1.e-8)throw std::runtime_error("paired conservation/mean test failed");
  }
  CompatibilityOptions o;o.track.initial="simple-wave";o.track.width=.04;o.track.amplitude=.8;o.track.speed=0.;o.track.mode="uniform";o.track.closure_space="l2";o.track.closure_boundary="natural";o.track.base.elements=16;o.track.base.order=3;o.track.base.dt=1.e-5;o.track.base.final_time=.0001;
  TrackingProblem base(o.track);CompatibilityProblem current(o);const auto first=TrackRun(base),second=CompatibilityRun(current);
  if(first.u!=second.u||first.sigma!=second.sigma||!second.failure.empty())throw std::runtime_error("baseline trajectory parity failed");
  o.formulation="weak-product";CompatibilityProblem weak(o);TrackWork wa,wb;const State state=weak.Initial();
  const State sa=weak.TrackingProblem::Closure(state,0.,wa),sb=weak.Closure(state,0.,wb);
  for(std::size_t i=0;i<sa.size();++i)if(std::abs(sa[i]-sb[i])>1.e-10)throw std::runtime_error("continuous trace weak source parity failed");
  std::cout<<"DG compatibility evolution self-test passed\n";
}
}
#ifndef DG_IGR_DG_COMPATIBILITY_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {const auto options=CompatibilityParse(argc,argv);if(options.track.self_test){CompatibilitySelfTest();return 0;}
    const auto setup=Clock::now();CompatibilityProblem pr(options);const double setup_seconds=Wall(setup);auto result=CompatibilityRun(pr);result.work.setup=setup_seconds;
    std::ostringstream original;TrackWrite(pr,result,original);std::string data=original.str();const auto label=data.find("selective_high_order_dg_conservative_rezone");
    if(label!=std::string::npos)data.replace(label,std::string("selective_high_order_dg_conservative_rezone").size(),"fixed_mesh_dg_igr_compatibility");
    const auto closing=data.rfind('}');data.erase(closing);
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"compatibility_formulation\":\""<<options.formulation<<"\",\"gradient_direction\":\""<<options.direction<<"\",\"stress_energy_trace\":\""<<options.energy_trace<<"\",\"velocity_projections\":"<<pr.velocity_projections<<",\"gradient_points\":"<<pr.gradient_points<<",\"projection_seconds\":"<<pr.projection_seconds<<",\"maximum_constant_test_defect\":";
    if(options.formulation=="baseline"||options.formulation=="weak-product")extra<<"null";else extra<<pr.max_mean_defect;
    extra<<",\"stress_pairing_scope\":\"interior faces; existing physical boundary flux retained\"\n}\n";data+=extra.str();
    if(options.track.output.empty())std::cout<<data;else{std::ofstream output(options.track.output);if(!output)throw std::runtime_error("cannot open output");output<<data;}
    return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif

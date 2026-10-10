// Diagnostic temporal Picard with full current-closure Euler residuals.
// No global tangent matrix, local closure claim, limiter, or fallback.
#define DG_IGR_SIGNED_PREDICTOR_LIBRARY_ONLY
#include "igr_dg_signed_predictor_probe.cpp"
#undef DG_IGR_SIGNED_PREDICTOR_LIBRARY_ONLY

namespace {
constexpr double PicardA[2][3]={{5./24.,1./3.,-1./24.},{1./6.,2./3.,1./6.}};
struct PicardData
{PredictorWork work;State iteration_changes,temporal_defects;std::uint64_t sweeps_completed=0;};
struct PicardRate{State rhs;std::array<double,4> budget{};};
PicardRate PicardEvaluate(const SignedEnrichmentProblem& pr,const State& u,const State& sigma,double time,PicardData& data)
{
  PredictorAdmit(pr,u,sigma,data.work);PicardRate result;result.rhs=pr.Residual(u,sigma,time,data.work.shared,result.budget);++data.work.rhs_calls;return result;
}
State PicardStage(const State& u,double dt,int stage,const std::array<PicardRate,3>& rates)
{
  State result=u;for(std::size_t k=0;k<u.size();++k)for(int j=0;j<3;++j)result[k]+=dt*PicardA[stage][j]*rates[j].rhs[k];return result;
}
TrackResult PicardRun(SignedEnrichmentProblem& pr,int subdivisions,int sweeps,PicardData& data)
{
  TrackResult out;const auto start=Clock::now();out.time=pr.memory.start_time;double accepted_source=0.;
  try
  {
    State initial=pr.Initial();State sigma=pr.Closure(initial,out.time,data.work.shared);PredictorAdmit(pr,initial,sigma,data.work);out.u=std::move(initial);out.sigma=std::move(sigma);out.initial_integrals=pr.Integrals(out.u,pr.edges);accepted_source=data.work.shared.current_source_integral;
    const double interval=(pr.o.base.final_time-out.time)/subdivisions;
    for(int part=0;part<subdivisions;++part)
    {
      ++data.work.shared.attempted;const double dt=std::min(interval,pr.o.base.final_time-out.time);const State original=out.u;
      std::array<PicardRate,3> rates;rates[0]=PicardEvaluate(pr,original,out.sigma,out.time,data);
      std::array<State,2> stage{PredictorAdd(original,rates[0].rhs,.5*dt),PredictorAdd(original,rates[0].rhs,dt)};
      for(int iteration=0;iteration<sweeps;++iteration)
      {
        for(int j=0;j<2;++j){const double time=out.time+(j?1.:.5)*dt;const State current=pr.Closure(stage[j],time,data.work.shared);rates[j+1]=PicardEvaluate(pr,stage[j],current,time,data);}
        std::array<State,2> next{PicardStage(original,dt,0,rates),PicardStage(original,dt,1,rates)};double change=0.,increment=0.;
        for(int j=0;j<2;++j){change=std::max(change,Norm(PredictorSubtract(next[j],stage[j])));increment=std::max(increment,Norm(PredictorSubtract(next[j],original)));}
        data.iteration_changes.push_back(change/std::max(increment,1.e-30));stage=std::move(next);++data.sweeps_completed;
      }
      // The accepted candidate and its stress share the same final state/time.
      State candidate_sigma=pr.Closure(stage[1],out.time+dt,data.work.shared);PredictorAdmit(pr,stage[1],candidate_sigma,data.work);const double candidate_source=data.work.shared.current_source_integral;
      // Evaluate the actual collocation defect, independently of the last
      // iteration change. These additional global closures/RHSs are charged.
      std::array<PicardRate,3> current;current[0]=rates[0];
      const State midpoint_sigma=pr.Closure(stage[0],out.time+.5*dt,data.work.shared);current[1]=PicardEvaluate(pr,stage[0],midpoint_sigma,out.time+.5*dt,data);current[2]=PicardEvaluate(pr,stage[1],candidate_sigma,out.time+dt,data);
      double residual=0.,increment=0.;for(int j=0;j<2;++j){residual=std::max(residual,Norm(PredictorSubtract(stage[j],PicardStage(original,dt,j,current))));increment=std::max(increment,Norm(PredictorSubtract(stage[j],original)));}data.temporal_defects.push_back(residual/std::max(increment,1.e-30));
      for(int c=0;c<4;++c)for(int j=0;j<3;++j)out.boundary_budget[c]+=dt*PicardA[1][j]*rates[j].budget[c];
      out.u=std::move(stage[1]);out.sigma=std::move(candidate_sigma);out.time=pr.memory.start_time+(part+1)*interval;++data.work.shared.steps;accepted_source=candidate_source;
    }
  }
  catch(const std::exception& error){out.failure=error.what();}
  data.work.shared.current_source_integral=accepted_source;
  if(!out.u.empty()){const auto totals=pr.Integrals(out.u,pr.edges);for(int c=0;c<4;++c)out.conservation_defect[c]=totals[c]-out.initial_integrals[c]-out.boundary_budget[c];}
  out.work=data.work.shared;out.wall=Wall(start);return out;
}
void PicardSelfTest()
{
  double quadrature=0.;const double nodes[3]={0.,.5,1.};for(int row=0;row<2;++row)for(int power=0;power<=2;++power){double value=0.;for(int j=0;j<3;++j)value+=PicardA[row][j]*std::pow(nodes[j],power);quadrature=std::max(quadrature,std::abs(value-std::pow(nodes[row+1],power+1)/(power+1)));}
  MemoryOptions mo;mo.recovery.recovery="existing";auto& b=mo.recovery.boundary;auto& t=b.compatibility.track;b.left=b.right="outflow";b.stress="trace";t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=15;t.base.final_time=1.e-4;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=21;mo.retained_order=mo.stress_order=5;
  SignedEnrichmentProblem pr(mo,3,.25,.25,"original",true,true);PicardData data;const auto result=PicardRun(pr,2,3,data);double conservation=0.;for(double x:result.conservation_defect)conservation=std::max(conservation,std::abs(x));
  std::cout<<std::setprecision(17)<<"{\"integrated_quadratic_time_basis_defect\":"<<quadrature<<",\"time\":"<<result.time<<",\"conservation\":"<<conservation<<",\"RHS_calls\":"<<data.work.rhs_calls<<",\"limited\":"<<result.work.limited<<",\"failure\":";PredictorString(std::cout,result.failure);std::cout<<"}\n";
  if(quadrature>1.e-14||!result.failure.empty()||std::abs(result.time-1.e-4)>1.e-14||conservation>1.e-8||data.work.rhs_calls!=18||result.work.limited)throw std::runtime_error("global temporal Picard selftest failed");
}
}

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    double time=0.,elapsed=1.e-4,slope=.25,curvature=.25;int extra=2,subdivisions=1,sweeps=3;std::string input,edges,volume="original";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--enrich-density"||flag=="--enrich-energy")continue;
      if(flag=="--extra-order"||flag=="--probe-time"||flag=="--external-moments-in"||flag=="--edges-in"||flag=="--elapsed"||flag=="--subintervals"||flag=="--sweeps"||flag=="--slope-strength"||flag=="--curvature-strength"||flag=="--entropy-volume")
      {if(++i==argc)throw std::runtime_error("missing temporal Picard option");const std::string v=argv[i];if(flag=="--extra-order")extra=std::stoi(v);else if(flag=="--probe-time")time=std::stod(v);else if(flag=="--external-moments-in")input=v;else if(flag=="--edges-in")edges=v;else if(flag=="--elapsed")elapsed=std::stod(v);else if(flag=="--subintervals")subdivisions=std::stoi(v);else if(flag=="--sweeps")sweeps=std::stoi(v);else if(flag=="--slope-strength")slope=std::stod(v);else if(flag=="--curvature-strength")curvature=std::stod(v);else volume=v;}
      else common.push_back(argv[i]);
    }
    MemoryOptions mo;mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;if(t.self_test){PicardSelfTest();return 0;}
    const int p=t.base.order,q=p+extra;mo.retained_order=mo.stress_order=q;t.base.order=3*q;t.base.final_time=time+elapsed;t.base.dt=elapsed/subdivisions;mo.input=input;mo.start_time=time;
    if(extra<1||extra>4||q>35||input.empty()||edges.empty()||!(elapsed>0.)||elapsed>.01||!std::isfinite(time)||time<0.||subdivisions<1||subdivisions>64||sweeps<1||sweeps>8)throw std::runtime_error("invalid temporal Picard options");
    if(mo.recovery.boundary.scenario!="application-tanh"||mo.recovery.boundary.left!="nonreflecting"||mo.recovery.boundary.right!="fixed"||volume!="original")throw std::runtime_error("Picard probe requires original autonomous application-tanh setup");
    const auto setup=Clock::now();SignedEnrichmentProblem pr(mo,p,slope,curvature,volume,true,true);PredictorGeometry(pr,PredictorRead(edges,pr.ne+1));const double setup_seconds=Wall(setup);PicardData data;TrackResult result=PicardRun(pr,subdivisions,sweeps,data);result.work.setup=setup_seconds;State external;
    if(!result.u.empty()){external=pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);result.u=pr.Physical(result.u);}
    std::ostringstream additional;additional<<std::setprecision(17)<<",\"temporal_method\":\"three-node Lobatto temporal Picard\",\"temporal_scope\":\"full-state global current-closure RHS at every evaluation\",\"sweeps\":"<<sweeps<<",\"subintervals\":"<<subdivisions<<",\"sweeps_completed\":"<<data.sweeps_completed<<",\"rhs_calls\":"<<data.work.rhs_calls<<",\"injected_state_time\":"<<time<<",\"base_order\":"<<p<<",\"retained_order\":"<<q<<",\"velocity_order\":"<<q<<",\"density_order\":"<<q<<",\"energy_moment_order\":"<<q<<",\"thermal_pressure_order\":"<<q<<",\"stress_order\":"<<q<<",\"extra_order\":"<<extra<<",\"primitive_source_faces\":\"central\",\"moment_operator\":\"direct\",\"enrich_density\":true,\"enrich_energy\":true,\"allocation\":\"all\",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"entropy_volume\":\"original\",\"actual_euler_dofs_total\":"<<4*pr.retained.ns<<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"minimum_admitted_density\":"<<data.work.density_min<<",\"minimum_admitted_pressure\":"<<data.work.pressure_min<<",\"minimum_admitted_Q\":"<<data.work.q_min<<",\"limiter_calls\":0,\"iteration_change_norm\":\"maximum stage high-coefficient Euclidean difference divided by maximum current stage increment; diagnostic fixed-point measure, not physical metric\",\"iteration_changes\":";TrackArray(additional,data.iteration_changes);additional<<",\"temporal_defects\":";TrackArray(additional,data.temporal_defects);additional<<",\"external_moment_coefficients\":";TrackArray(additional,external);additional<<",\"moment_coefficients\":";TrackArray(additional,external);
    BoundaryWrite(pr,result,additional.str());return result.failure.empty()?0:2;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

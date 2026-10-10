// Test-only autonomous full-RHS Taylor defect; no local closure construction.
#define DG_IGR_SIGNED_PREDICTOR_LIBRARY_ONLY
#include "igr_dg_signed_predictor_probe.cpp"
#undef DG_IGR_SIGNED_PREDICTOR_LIBRARY_ONLY

namespace {
struct CurvatureEvaluation{State rhs,sigma,branches;};
State CurvatureBranches(const SignedEnrichmentProblem& pr,const State& external,const State& sigma)
{
  const State physical=pr.Physical(external);State result;
  // The identical shared ghost-state helper is evaluated diagnostically. The
  // residual itself always comes from the original complete shared operator.
  for(int face=0;face<=pr.ne;++face)
  {
    std::array<double,4> l{},r{};double sl=0.,sr=0.;
    if(face>0){for(int c=0;c<4;++c)l[c]=physical[pr.Id(face-1,pr.p,c)];sl=sigma[pr.SigmaIndex(face-1,pr.p)];}
    if(face<pr.ne){for(int c=0;c<4;++c)r[c]=physical[pr.Id(face,0,c)];sr=sigma[pr.SigmaIndex(face,0)];}
    if(face==0||face==pr.ne)
    {
      const int side=face==pr.ne;auto& inner=side?l:r;auto& outer=side?r:l;
      const auto reference=pr.InitialAt(pr.o.center+(side?40.:-40.)*pr.o.width);const double normal[2]={side?1.:-1.,0.};
      using Kind=mfem::detail::EulerBoundaryStateKind;const std::string kind=side?pr.boundary.right:pr.boundary.left;Kind tag=Kind::supersonic_inlet;
      if(kind=="outflow")tag=Kind::supersonic_outlet;else if(kind=="nonreflecting")tag=Kind::nonreflecting_outlet;else if(kind=="wall")tag=Kind::slip_wall;else if(kind=="characteristic")tag=Kind::far_field;
      if(!mfem::detail::EvaluateEulerBoundaryState(tag,2,4,gamma_euler,1.,reference.data(),normal,inner.data(),outer.data(),floor_value,floor_value))throw std::runtime_error("diagnostic ghost state rejected");
      const bool zero=pr.boundary.stress=="zero"||(pr.boundary.stress=="legacy"&&kind=="fixed");if(side)sr=zero?0.:sl;else sl=zero?0.:sr;
    }
    double al=0.,ar=0.,jump=0.;pr.physics.Flux(l.data(),sl,al);pr.physics.Flux(r.data(),sr,ar);for(int c=0;c<4;++c)jump=std::max(jump,std::abs(l[c]-r[c]));
    result.insert(result.end(),{al-ar,l[1]/l[0],r[1]/r[0],jump,al,ar});
  }
  return result;
}
CurvatureEvaluation CurvatureEvaluate(const SignedEnrichmentProblem& pr,const State& u,double time,PredictorWork& work)
{
  CurvatureEvaluation value;value.sigma=pr.Closure(u,time,work.shared);value.rhs=PredictorRHS(pr,u,value.sigma,time,work);value.branches=CurvatureBranches(pr,u,value.sigma);return value;
}
State CurvatureFirst(const SignedEnrichmentProblem& pr,const State& u,const State& direction,double time,double epsilon,PredictorWork& work)
{
  const double magnitude=PredictorMaximum(direction);if(magnitude==0.)return State(u.size(),0.);const double step=epsilon/magnitude;
  const auto plus=CurvatureEvaluate(pr,PredictorAdd(u,direction,step),time,work),minus=CurvatureEvaluate(pr,PredictorAdd(u,direction,-step),time,work);
  return PredictorSubtract(plus.rhs,minus.rhs,2*step);
}
State CurvatureSecond(const State& plus,const State& center,const State& minus,double step)
{
  if(plus.size()!=center.size()||minus.size()!=center.size()||!(step>0.))throw std::runtime_error("curvature difference shape/step mismatch");State value(center.size());
  for(std::size_t i=0;i<value.size();++i)value[i]=((plus[i]-center[i])+(minus[i]-center[i]))/(step*step);return value;
}
void CurvaturePredictorSelfTest()
{
  MemoryOptions mo;auto& b=mo.recovery.boundary;auto& t=b.compatibility.track;mo.recovery.recovery="existing";b.left=b.right="outflow";b.stress="trace";t.initial="entropy";t.mode="uniform";t.left=-1.;t.right=1.;t.base.elements=8;t.base.order=15;t.base.alpha=.04;t.base.final_time=.01;t.quad_extra=21;t.closure_space="l2";t.closure_boundary="natural";mo.retained_order=mo.stress_order=5;
  SignedEnrichmentProblem pr(mo,3,.25,.25,"original",true,true);PredictorGeometry(pr,State{-1.,-.85,-.6,-.3,0.,.2,.45,.7,1.});PredictorCoordinates coordinates(pr,3);
  State low(pr.ne*pr.retained.n*4,0.),direction=low,exact=low;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.retained.n;++i)
  {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.retained.physics.basis.nodes[i],rho=1.2,v=.2+.1*x,dv=.3+.07*x;const int j=(e*pr.retained.n+i)*4;low[j]=rho;low[j+1]=rho*v;low[j+3]=3.+.5*rho*v*v;direction[j+1]=rho*dv;exact[j+1]=-2*(3-gamma_euler)*rho*dv*.07;}
  const auto embed=[&](const State& a){return pr.state_transfer.Apply(pr.state_transfer.E,a,pr.ne,4);};const State u=embed(low),d=embed(direction);PredictorWork work;const auto base=CurvatureEvaluate(pr,u,0.,work);const double h=.01/PredictorMaximum(d);
  const auto plus=CurvatureEvaluate(pr,PredictorAdd(u,d,h),0.,work),minus=CurvatureEvaluate(pr,PredictorAdd(u,d,-h),0.,work),halfplus=CurvatureEvaluate(pr,PredictorAdd(u,d,.5*h),0.,work),halfminus=CurvatureEvaluate(pr,PredictorAdd(u,d,-.5*h),0.,work);
  const State dd=pr.state_transfer.Apply(pr.state_transfer.P,CurvatureSecond(plus.rhs,base.rhs,minus.rhs,h),pr.ne,4),dh=pr.state_transfer.Apply(pr.state_transfer.P,CurvatureSecond(halfplus.rhs,base.rhs,halfminus.rhs,.5*h),pr.ne,4);double error=0.,epsilon=0.;
  for(std::size_t i=0;i<dd.size();++i)if(i%4==1){error=std::max(error,std::abs(dd[i]-exact[i]));epsilon=std::max(epsilon,std::abs(dd[i]-dh[i]));}
  const double projection=PredictorRelative(PredictorAdd(coordinates.Coarse(base.rhs),coordinates.Embed(coordinates.Extract(base.rhs)),1.),base.rhs);
  std::cout<<std::setprecision(17)<<"{\"affine_velocity_momentum_hessian_maximum_error\":"<<error<<",\"half_step_hessian_maximum_difference\":"<<epsilon<<",\"coarse_tail_split\":"<<projection<<",\"maximum_rhs_conservation_defect\":"<<work.shared.max_rhs_defect<<"}\n";
  if(error>1.e-5||epsilon>2.e-5||projection>1.e-12||work.shared.max_rhs_defect>1.e-8)throw std::runtime_error("full Euler second directional derivative selftest failed");
}
}

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    const auto start=Clock::now();double time=0.,epsilon=1.e-3,tangent_epsilon=1.e-5,slope=.25,curvature=.25;int extra=2;std::string input,edges,volume="original";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--enrich-density"||flag=="--enrich-energy")continue;
      if(flag=="--extra-order"||flag=="--probe-time"||flag=="--external-moments-in"||flag=="--edges-in"||flag=="--epsilon"||flag=="--tangent-epsilon"||flag=="--slope-strength"||flag=="--curvature-strength"||flag=="--entropy-volume")
      {if(++i==argc)throw std::runtime_error("missing predictor curvature option");const std::string v=argv[i];if(flag=="--extra-order")extra=std::stoi(v);else if(flag=="--probe-time")time=std::stod(v);else if(flag=="--external-moments-in")input=v;else if(flag=="--edges-in")edges=v;else if(flag=="--epsilon")epsilon=std::stod(v);else if(flag=="--tangent-epsilon")tangent_epsilon=std::stod(v);else if(flag=="--slope-strength")slope=std::stod(v);else if(flag=="--curvature-strength")curvature=std::stod(v);else volume=v;}
      else common.push_back(argv[i]);
    }
    MemoryOptions mo;mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;if(t.self_test){CurvaturePredictorSelfTest();return 0;}
    const int p=t.base.order,q=p+extra;mo.retained_order=mo.stress_order=q;t.base.order=3*q;mo.input=input;mo.start_time=time;
    if(extra<1||extra>4||q>35||input.empty()||edges.empty()||!(epsilon>0.)||epsilon>.01||!(tangent_epsilon>0.)||tangent_epsilon>.001||!std::isfinite(time)||time<0.||time>t.base.final_time)throw std::runtime_error("invalid curvature input/order/time/perturbation");
    if(mo.recovery.boundary.scenario!="application-tanh"||mo.recovery.boundary.left!="nonreflecting"||mo.recovery.boundary.right!="fixed"||volume!="original")throw std::runtime_error("curvature probe requires original autonomous application-tanh setup");
    SignedEnrichmentProblem pr(mo,p,slope,curvature,volume,true,true);PredictorGeometry(pr,PredictorRead(edges,pr.ne+1));PredictorCoordinates coordinates(pr,p);const State low=PredictorRead(input,pr.ne*pr.retained.n*4),u=pr.state_transfer.Apply(pr.state_transfer.E,low,pr.ne,4);PredictorWork work;
    const auto base=CurvatureEvaluate(pr,u,time,work);const double magnitude=PredictorMaximum(base.rhs);if(!(magnitude>0.))throw std::runtime_error("curvature endpoint has zero residual");
    const State jf=CurvatureFirst(pr,u,base.rhs,time,tangent_epsilon,work),jfhalf=CurvatureFirst(pr,u,base.rhs,time,.5*tangent_epsilon,work);
    const State coarse_acceleration=coordinates.Coarse(jfhalf),missing=coordinates.Extract(CurvatureFirst(pr,u,coarse_acceleration,time,tangent_epsilon,work));
    const State missing_half=coordinates.Extract(CurvatureFirst(pr,u,coordinates.Coarse(jf),time,.5*tangent_epsilon,work));
    const State total_jjf=coordinates.Extract(CurvatureFirst(pr,u,jfhalf,time,tangent_epsilon,work)),tail_jjf=coordinates.Extract(CurvatureFirst(pr,u,coordinates.Embed(coordinates.Extract(jfhalf)),time,tangent_epsilon,work));
    auto array=[](const char* key,const State& a){std::cout<<",\""<<key<<"\":";TrackArray(std::cout,a);};
    std::cout<<std::setprecision(17)<<"{\"schema\":1,\"completed\":true,\"kind\":\"all-field autonomous predictor Taylor curvature\",\"arithmetic\":\"FP64 CPU\",\"closure_scope\":\"full instantaneous global SIPG at every evaluation\",\"elements\":"<<pr.ne<<",\"base_order\":"<<p<<",\"rich_order\":"<<q<<",\"physical_order\":"<<pr.p<<",\"time\":"<<time<<",\"alpha\":"<<t.base.alpha<<",\"effective_sipg_factor\":"<<pr.o.sipg_penalty*pr.n*pr.n<<",\"quad_extra\":"<<t.quad_extra<<",\"quad_points\":"<<pr.weights.size()<<",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"tangent_epsilon\":"<<tangent_epsilon<<",\"second_epsilon\":"<<epsilon<<",\"residual_maximum\":"<<magnitude<<",\"tail_block_size\":"<<coordinates.block<<",\"tail_size\":"<<coordinates.size<<",\"branch_layout\":\"face0..N, speed_left-minus-right, velocity_left, velocity_right, state_jump_max, speed_left, speed_right\"";
    array("edges",pr.edges);array("rich_nodes",State(pr.retained.physics.basis.nodes.GetData(),pr.retained.physics.basis.nodes.GetData()+pr.retained.n));array("input_moments",low);array("base_rhs",pr.state_transfer.Apply(pr.state_transfer.P,base.rhs,pr.ne,4));array("base_sigma",pr.stress_transfer.Apply(pr.stress_transfer.P,base.sigma,pr.ne,1));array("base_branches",base.branches);array("tail_rhs",coordinates.Extract(base.rhs));array("JF",pr.state_transfer.Apply(pr.state_transfer.P,jfhalf,pr.ne,4));array("tail_JF",coordinates.Extract(jfhalf));array("coarse_acceleration",pr.state_transfer.Apply(pr.state_transfer.P,coarse_acceleration,pr.ne,4));array("missing_coarse_term",missing);array("missing_coarse_term_control",missing_half);array("tail_JJF",total_jjf);array("tail_JQJF",tail_jjf);
    std::cout<<",\"second_derivative_ladder\":[";bool first=true;
    for(double factor:State{4.,2.,1.,.5,.25})
    {
      const double amplitude=epsilon*factor,h=amplitude/magnitude;
      const auto plus=CurvatureEvaluate(pr,PredictorAdd(u,base.rhs,h),time,work),minus=CurvatureEvaluate(pr,PredictorAdd(u,base.rhs,-h),time,work),plus2=CurvatureEvaluate(pr,PredictorAdd(u,base.rhs,2*h),time,work),minus2=CurvatureEvaluate(pr,PredictorAdd(u,base.rhs,-2*h),time,work);
      if(!first)std::cout<<',';first=false;std::cout<<"{\"maximum_coefficient_amplitude\":"<<amplitude<<",\"direction_step\":"<<h;
      array("centered",coordinates.Extract(CurvatureSecond(plus.rhs,base.rhs,minus.rhs,h)));array("forward",coordinates.Extract(CurvatureSecond(plus2.rhs,plus.rhs,base.rhs,h)));array("backward",coordinates.Extract(CurvatureSecond(base.rhs,minus.rhs,minus2.rhs,h)));array("first_derivative",coordinates.Extract(PredictorSubtract(plus.rhs,minus.rhs,2*h)));array("plus_branches",plus.branches);array("minus_branches",minus.branches);array("plus2_branches",plus2.branches);array("minus2_branches",minus2.branches);std::cout<<'}';
    }
    std::cout<<"],\"checks\":{\"JF_step_relative_difference\":"<<PredictorRelative(jf,jfhalf)<<",\"missing_coarse_step_relative_difference\":"<<PredictorRelative(missing,missing_half)<<",\"JJF_coarse_tail_split_relative_difference\":"<<PredictorRelative(total_jjf,PredictorAdd(missing,tail_jjf,1.))<<",\"maximum_rhs_conservation_defect\":"<<work.shared.max_rhs_defect<<",\"maximum_closure_residual\":"<<work.shared.max_closure_residual<<"},\"admission\":{\"density_min\":"<<work.density_min<<",\"pressure_min\":"<<work.pressure_min<<",\"Q_min\":"<<work.q_min<<",\"limiter_calls\":0},\"work\":{\"closure_calls\":"<<work.shared.closure_calls<<",\"rhs_calls\":"<<work.rhs_calls<<",\"closure_seconds\":"<<work.shared.closure<<",\"transport_seconds\":"<<work.shared.transport<<",\"total_before_output_end_seconds\":"<<Wall(start)<<"}}\n";
    return 0;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

// Direct combination screen: independent conservative subcell means, the
// existing compact-current-density H1/SIPG closure and dense face condensation,
// plus a local MUSCL-Hancock half-step predictor and conservative corrector.
// Predictor sigma is explicitly frozen at the current START state. A current
// midpoint closure is solved for the predicted means; accepted endpoint closure
// is solved and carried. No derivatives of Rusanov max/abs and no continuum edit.
#define main igr_existing_subcell_evolution_probe_main
#include "igr_subcell_evolution_probe.cpp"
#undef main
#include <memory>

namespace
{
struct HancockWork
{
  std::uint64_t predictor_physical_flux_calls=0, boundary_speed_flux_calls=0;
  std::uint64_t midpoint_closures=0, endpoint_closures=0, predicted_cells=0;
  std::uint64_t midpoint_transport_limited_cells=0;
};
struct TransportEndpoints
{
  State left,right,left_flux,right_flux;
  double speed=0.;
};
TransportEndpoints ReconstructTransport(const Problem& p,const State& u,
    const Problem::Closure& closure,Work& w,HancockWork& hw,bool cache_physical_flux=true)
{
  TransportEndpoints result;result.left.resize(p.N);result.right.resize(p.N);
  if(cache_physical_flux){result.left_flux.resize(p.N);result.right_flux.resize(p.N);}
  for(int i=0;i<p.N;++i)
  {
    Cons slope{};
    if(p.o.transport=="muscl")for(int k=0;k<4;++k)
    {const double l=u[i][k]-p.Neighbor(u,i-1)[k],r=p.Neighbor(u,i+1)[k]-u[i][k];
     slope[k]=Minmod(2*l,.5*(l+r),2*r);}
    else ++w.first_order_cells;
    Poly linear;linear.c[0]=u[i];
    for(int k=0;k<4;++k){linear.c[0][k]-=.5*slope[k];linear.c[1][k]=slope[k];}
    const double theta=p.Limit(linear,u[i],w,true);
    if(theta<1.)++w.transport_limited_cells;
    if(theta<1.e-8)++w.first_order_cells;
    result.left[i]=linear.Value(0.);result.right[i]=linear.Value(1.);
    if(cache_physical_flux)
    {double a=0.,b=0.;hw.predictor_physical_flux_calls+=2;
    result.left_flux[i]=p.Flux(result.left[i],p.SigmaFace(closure,i,false),a);
    result.right_flux[i]=p.Flux(result.right[i],p.SigmaFace(closure,i,true),b);
    result.speed=std::max({result.speed,a,b});}
  }
  if(!p.periodic && cache_physical_flux)
  {
    double a=0.,b=0.;hw.boundary_speed_flux_calls+=2;
    (void)p.Flux(p.left,p.SigmaFace(closure,0,false),a);
    (void)p.Flux(p.right,p.SigmaFace(closure,p.N-1,true),b);
    result.speed=std::max({result.speed,a,b});
  }
  return result;
}

Problem::RHS MidpointCorrector(const Problem& p,const State& left,const State& right,
    const Problem::Closure& current_midpoint,Work& w)
{
  ++w.rhs_calls;Problem::RHS result;result.value.resize(p.N);
  std::vector<Cons> face_flux(p.N+1);
  for(int face=0;face<(p.periodic?p.N:p.N+1);++face)
  {
    const int li=(face+p.N-1)%p.N,ri=face%p.N;
    const Cons& ul=face==0&&!p.periodic?p.left:right[li];
    const Cons& ur=face==p.N&&!p.periodic?p.right:left[ri];
    const double sl=face==0&&!p.periodic?p.SigmaFace(current_midpoint,0,false):p.SigmaFace(current_midpoint,li,true);
    const double sr=face==p.N&&!p.periodic?p.SigmaFace(current_midpoint,p.N-1,true):p.SigmaFace(current_midpoint,ri,false);
    double a=0.,b=0.;const Cons fl=p.Flux(ul,sl,a),fr=p.Flux(ur,sr,b);
    const double speed=std::max(a,b);result.speed=std::max(result.speed,speed);++w.face_fluxes;
    for(int k=0;k<4;++k)face_flux[face][k]=.5*(fl[k]+fr[k]-speed*(ur[k]-ul[k]));
  }
  if(p.periodic)face_flux[p.N]=face_flux[0];
  for(int k=0;k<4;++k)result.boundary[k]=face_flux[p.N][k]-face_flux[0][k];
  for(int i=0;i<p.N;++i)for(int k=0;k<4;++k)
    result.value[i][k]=(face_flux[i][k]-face_flux[i+1][k])/p.dx;
  return result;
}

Result AdvanceHancock(const Problem& p,Work& w,HancockWork& hw,bool heun=false)
{
  Result r;r.state=p.Initial();r.initial_integral=p.Integrals(r.state);
  const auto start=Clock::now();std::unique_ptr<Problem::Closure> current;
  try{current=std::make_unique<Problem::Closure>(p.Assemble(r.state,w));p.Solve(*current,w);}
  catch(const std::exception& e){r.success=false;r.reason=e.what();}
  while(r.success && r.time<p.o.final_time-4*std::numeric_limits<double>::epsilon()*std::max(1.,p.o.final_time))
  {
    TransportEndpoints transport;Problem::RHS first;
    try
    {transport=ReconstructTransport(p,r.state,*current,w,hw,!heun);
     if(heun){first=MidpointCorrector(p,transport.left,transport.right,*current,w);transport.speed=first.speed;}}
    catch(const std::exception& e){r.success=false;r.reason=e.what();break;}
    double dt=std::min({p.o.dt,p.o.final_time-r.time,p.o.cfl*p.dx/std::max(transport.speed,1.e-12)});
    bool accepted=false;
    for(int trial=0;trial<20 && dt>0. && (trial==0 || dt>1.e-10);++trial)
    {
      ++w.attempted_steps;
      try
      {
        State next;Problem::RHS rhs;
        if(heun)
        {
          State one=Stage(r.state,r.state,first.value,0.,dt);
          ++hw.midpoint_closures;const auto second=p.Rhs(one,w);
          next=Stage(r.state,one,second.value,.5,dt);
          for(int k=0;k<4;++k)rhs.boundary[k]=.5*(first.boundary[k]+second.boundary[k]);
          rhs.speed=std::max(first.speed,second.speed);
        }
        else
        {
        State means=r.state,left=transport.left,right=transport.right;
        hw.predicted_cells+=p.N;
        for(int i=0;i<p.N;++i)
        {
          for(int k=0;k<4;++k)
          {
            const double change=-.5*dt*(transport.right_flux[i][k]-transport.left_flux[i][k])/p.dx;
            means[i][k]+=change;left[i][k]+=change;right[i][k]+=change;
          }
          Require(means[i]);
          Poly predicted;predicted.c[0]=left[i];
          for(int k=0;k<4;++k)predicted.c[1][k]=right[i][k]-left[i][k];
          const double theta=p.Limit(predicted,means[i],w,true);
          if(theta<1.){++w.transport_limited_cells;++hw.midpoint_transport_limited_cells;}
          if(theta<1.e-8)++w.first_order_cells;
          left[i]=predicted.Value(0.);right[i]=predicted.Value(1.);
        }
        ++hw.midpoint_closures;auto midpoint=p.Assemble(means,w);p.Solve(midpoint,w);
        rhs=MidpointCorrector(p,left,right,midpoint,w);
        next=Stage(r.state,r.state,rhs.value,0.,dt);
        }
        ++hw.endpoint_closures;
        auto endpoint=std::make_unique<Problem::Closure>(p.Assemble(next,w));p.Solve(*endpoint,w);
        // Assemble/Recover admits the current endpoint representation, as in
        // the baseline's charged endpoint Recover. Flux admission is checked
        // when this carried state supplies the next predictor. Shared output
        // audits density/pressure and closure residual, not endpoint flux speed.
        for(int k=0;k<4;++k)r.boundary[k]+=dt*rhs.boundary[k];
        r.state=std::move(next);current.swap(endpoint);r.time+=dt;
        ++w.accepted_steps;accepted=true;
        w.max_cfl=std::max(w.max_cfl,dt*std::max(transport.speed,rhs.speed)/p.dx);break;
      }
      catch(const std::exception& e){++w.rejected_steps;r.reason=e.what();dt*=.5;}
    }
    if(!accepted){r.success=false;r.reason="step retry limit: "+r.reason;break;}
  }
  r.seconds=Wall(start);const Cons final=p.Integrals(r.state);
  for(int k=0;k<4;++k)r.defect[k]=final[k]+r.boundary[k]-r.initial_integral[k];
  if(r.success)r.reason="target-reached";return r;
}

void Annotate(const Options& o,const std::string& method,const HancockWork& hw)
{
  std::ifstream input(o.output);std::string original((std::istreambuf_iterator<char>(input)),{});
  if(original.empty()||original[0]!='{')throw std::runtime_error("missing shared output for predictor annotation");
  std::ofstream output(o.output);output<<"{\n\"time_method\":";JsonString(output,method);
  output<<",\"predictor_sigma_policy\":";
  JsonString(output,method=="hancock"?"current-start explicit approximation":"not applicable: current explicit RK stages");
  output<<",\"midpoint_closure_policy\":";
  JsonString(output,method=="hancock"?"current predicted means, original-operator residual checked":method=="heun"?"current full-step Euler predictor, original-operator residual checked":"not applicable: current SSPRK3 stages");
  output
        <<",\"hancock_work\":{\"predictor_physical_flux_calls\":"<<hw.predictor_physical_flux_calls
        <<",\"boundary_speed_flux_calls\":"<<hw.boundary_speed_flux_calls
        <<",\"corrector_stage_closures\":"<<hw.midpoint_closures
        <<",\"endpoint_closures\":"<<hw.endpoint_closures
        <<",\"predicted_cells\":"<<hw.predicted_cells
        <<",\"midpoint_transport_limited_cells\":"<<hw.midpoint_transport_limited_cells
        <<"},"<<original.substr(1);
}

void HancockSelfTest()
{
  for(const std::string kind:{"h1","sipg"})
  {
    Options o;o.elements=4;o.subcells=3;o.order=3;o.initial="smooth";o.closure=kind;o.final_time=.004;
    Problem p(o);Work w;HancockWork hw;State u=p.Initial();auto closure=p.Assemble(u,w);p.Solve(closure,w);
    const auto transport=ReconstructTransport(p,u,closure,w,hw);
    const auto coupled=MidpointCorrector(p,transport.left,transport.right,closure,w),baseline=p.Rhs(u,w);
    for(int i=0;i<p.N;++i)for(int k=0;k<4;++k)
      if(std::abs(coupled.value[i][k]-baseline.value[i][k])>1.e-12)
        throw std::runtime_error("Hancock zero-step face flux differs from shared MUSCL RHS");
    for(bool heun:{false,true})
    {Work run_work;HancockWork run_extra;auto r=AdvanceHancock(p,run_work,run_extra,heun);
    if(!r.success||run_work.rejected_steps||run_work.closure_calls!=1+2*run_work.accepted_steps)
      throw std::runtime_error("Hancock closure-count/admission self-test failed");
    for(double x:r.defect)if(std::abs(x)>2.e-12)throw std::runtime_error("Hancock/Heun conservative boundary budget failed");}
  }
  std::cout<<"subcell_predictor_self_test pass\n";
}
} // namespace

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");const auto start=Clock::now();
  try
  {
    std::string method="hancock";std::vector<std::string> args{argv[0]};
    for(int i=1;i<argc;++i)
    {if(std::string(argv[i])=="--method")
      {if(i+1==argc)throw std::runtime_error("missing method value");method=argv[++i];}
     else args.push_back(argv[i]);}
    if(method!="hancock"&&method!="heun"&&method!="ssprk3")throw std::runtime_error("unsupported time method");
    std::vector<char*> pointers;for(auto& s:args)pointers.push_back(&s[0]);
    Options o=Parse(pointers.size(),pointers.data());
    if(o.self_test){HancockSelfTest();return 0;}
    Problem p(o);Work w;HancockWork hw;
    Result result=method=="ssprk3"?Advance(p,w):AdvanceHancock(p,w,hw,method=="heun");
    // Shared output charges one more final closure and dense positivity audit
    // to all methods. Hancock/Heun therefore cost 2N+2 closures on a clean run,
    // versus 3N+1 for SSPRK3, rather than hiding this output-only recomputation.
    Write(o,p,result,w,start);Annotate(o,method,hw);return result.success?0:2;
  }
  catch(const std::exception& e){std::cerr<<"subcell_predictor_error "<<e.what()<<'\n';return 2;}
}

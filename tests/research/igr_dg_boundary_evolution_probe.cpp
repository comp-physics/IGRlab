// Test-only current-closure DG boundary controls. Reuses the advancing core;
// boundary flux replacement and its redundant baseline work are both charged.
#define DG_IGR_DG_COMPATIBILITY_LIBRARY_ONLY
#include "igr_dg_compatibility_evolution_probe.cpp"
#undef DG_IGR_DG_COMPATIBILITY_LIBRARY_ONLY
#include "physics/bc/detail/euler_boundary_state.hpp"
#include "physics/bc/boundary_state.hpp"
#include "../../examples/shock_vortex_interaction/shock_vortex_interaction.hpp"

namespace
{
struct BoundaryOptions
{
  CompatibilityOptions compatibility;
  std::string scenario="existing",left="fixed",right="fixed",stress="trace";
  std::string open_closure="natural",wall_closure="natural",wall_source="interior",wall_flux="mirror";
  int positivity_subdivisions=0;
};
CompatibilityOptions BoundaryBase(const BoundaryOptions& options)
{
  auto result=options.compatibility;
  // Base boundary flux is an interior physical flux. The residual below
  // replaces it, leaving shared volume/interior operators untouched.
  result.track.euler_boundary="outflow";
  result.homogeneous_velocity_boundary={{options.wall_source=="zero"&&options.left=="wall",
                                         options.wall_source=="zero"&&options.right=="wall"}};
  return result;
}
struct BoundaryProblem:CompatibilityProblem
{
  BoundaryOptions boundary;
  std::function<void(const mfem::Vector&,mfem::Vector&)> application_initial;
  mutable double max_wall_mass=0.,max_wall_energy=0.,max_wall_velocity=0.;
  mutable std::uint64_t boundary_calls=0;
  explicit BoundaryProblem(const BoundaryOptions& options):CompatibilityProblem(BoundaryBase(options)),boundary(options)
  {
    if(boundary.scenario=="application-tanh")
    {
      // The application mixes rho, momentum, and pressure, then constructs E.
      // Reuse its initializer, including its right-to-left shock orientation.
      const double base_width=std::min((o.right-o.left)/ne,1.);
      application_initial=mfem::GetShockVortexInoueHattoriInitCustom(
          gamma_euler,ne,1,o.mach,0.,o.width/base_width,
          o.left,o.right,-.5,.5,0.,0.,o.center);
    }
    if(boundary.positivity_subdivisions>0)
      for(int j=0;j<=boundary.positivity_subdivisions;++j)
        positive.push_back(Shape(double(j)/boundary.positivity_subdivisions,false));
  }
  std::array<double,4> InitialAt(double x)const
  {
    if(boundary.scenario=="application-tanh")
    {
      mfem::Vector point(2),value(4);point[0]=x;point[1]=0.;application_initial(point,value);
      value[3]+=o.speed*value[1]+.5*o.speed*o.speed*value[0];value[1]+=o.speed*value[0];
      return {{value[0],value[1],value[2],value[3]}};
    }
    if(boundary.scenario=="existing")
    {const auto v=Reference(x,0.);return {{v[0],v[1],v[2],v[3]}};}
    // Smooth impingement: normal velocity and its first derivative vanish at
    // the stationary right wall initially. No preformed wall discontinuity.
    const double distance=(o.right-x)/o.width;
    const double velocity=o.amplitude*(1.-std::exp(-distance*distance));
    return {{1.,velocity,0.,1./(gamma_euler-1.)+.5*velocity*velocity}};
  }
  State Initial()const
  {
    if(boundary.scenario=="existing")return CompatibilityProblem::Initial();
    State state(ne*n*4,0.);
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
    {const auto value=InitialAt(edges[e]+(edges[e+1]-edges[e])*physics.basis.nodes[i]);
      for(int c=0;c<4;++c)state[Id(e,i,c)]=value[c];}
    return state;
  }
  State Closure(const State& state,double time,TrackWork& work)const
  {
    const auto start=Clock::now();std::array<double,2> robin{},neumann{};
    for(int side=0;side<2;++side)
    {
      const int e=side?ne-1:0,i=side?p:0;const double rho=state[Id(e,i,0)];
      const bool wall=(side?boundary.right:boundary.left)=="wall";
      if(!wall&&boundary.open_closure=="screened")robin[side]=std::sqrt(o.base.alpha)/rho;
      if(wall&&boundary.wall_closure=="pressure-balance")
      {
        const State derivative=Shape(side?1.:0.,true);std::array<double,4> du{};
        for(int j=0;j<n;++j)for(int c=0;c<4;++c)du[c]+=derivative[j]*state[Id(e,j,c)]/(edges[e+1]-edges[e]);
        const double vx=state[Id(e,i,1)]/rho,vy=state[Id(e,i,2)]/rho;
        const double px=(gamma_euler-1.)*(du[3]-vx*du[1]-vy*du[2]+.5*(vx*vx+vy*vy)*du[0]);
        // Test-only total normal pressure-gradient balance for a flat wall.
        neumann[side]=-o.base.alpha/rho*(side?1.:-1.)*px;
      }
    }
    work.closure+=Wall(start);return CompatibilityProblem::Closure(state,time,work,robin,neumann);
  }
  std::array<double,4> BoundaryFlux(const State& state,const State& sigma,int side)const
  {
    using Kind=mfem::detail::EulerBoundaryStateKind;
    const int e=side?ne-1:0,i=side?p:0;std::array<double,4> inner{},outer{};
    for(int c=0;c<4;++c)inner[c]=state[Id(e,i,c)];
    // Application inlet/outlet reference states are exact RH far fields, not
    // the exponentially close finite-endpoint values of the smoothed IC.
    const double reference_x=boundary.scenario=="application-tanh"?
        o.center+(side?40.:-40.)*o.width:(side?o.right:o.left);
    const auto reference=InitialAt(reference_x);const double normal[2]={side?1.:-1.,0.};
    const std::string kind=side?boundary.right:boundary.left;
    Kind tag=Kind::supersonic_inlet;
    if(kind=="outflow")tag=Kind::supersonic_outlet;
    if(kind=="wall")tag=Kind::slip_wall;
    if(kind=="characteristic")tag=Kind::far_field;
    if(kind=="nonreflecting")tag=Kind::nonreflecting_outlet;
    if(!mfem::detail::EvaluateEulerBoundaryState(tag,2,4,gamma_euler,1.,reference.data(),normal,
                                                inner.data(),outer.data(),floor_value,floor_value))
      throw std::runtime_error("shared Euler boundary state rejected");
    const double si=sigma[SigmaIndex(e,i)];
    const bool zero=boundary.stress=="zero"||(boundary.stress=="legacy"&&kind=="fixed");
    const double so=zero?0.:si;double ai=0.,ao=0.;
    const auto fi=physics.Flux(inner.data(),si,ai),fo=physics.Flux(outer.data(),so,ao);
    const double speed=physics.FaceSpeed(ai,ao);std::array<double,4> flux{};
    for(int c=0;c<4;++c)flux[c]=.5*(fi[c]+fo[c]-(side?1.:-1.)*speed*(outer[c]-inner[c]));
    if(kind=="wall"&&boundary.wall_flux=="traction-llf")
    {
      const double velocity=inner[1]/inner[0];
      const double pressure=(gamma_euler-1.)*(inner[3]-.5*(inner[1]*inner[1]+inner[2]*inner[2])/inner[0]);
      flux={{0.,pressure+si+inner[0]*speed*(side?1.:-1.)*velocity,0.,0.}};
    }
    if(kind=="wall")
    {max_wall_mass=std::max(max_wall_mass,std::abs(flux[0]));max_wall_energy=std::max(max_wall_energy,std::abs(flux[3]));
      max_wall_velocity=std::max(max_wall_velocity,std::abs(inner[1]/inner[0]));}
    ++boundary_calls;return flux;
  }
  State Residual(const State& state,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    State result=CompatibilityProblem::Residual(state,sigma,time,work,budget);const auto start=Clock::now();
    for(int side=0;side<2;++side)
    {
      const int e=side?ne-1:0,j=side?p:0;const double sign=side?-1.:1.,h=edges[e+1]-edges[e];
      std::array<double,4> inner{};for(int c=0;c<4;++c)inner[c]=state[Id(e,j,c)];double speed=0.;
      const auto old=physics.Flux(inner.data(),sigma[SigmaIndex(e,j)],speed),next=BoundaryFlux(state,sigma,side);
      for(int c=0;c<4;++c)
      {const double delta=next[c]-old[c];budget[c]+=sign*delta;
        for(int i=0;i<n;++i)result[Id(e,i,c)]+=sign*mass_inverse(i,j)*delta/h;}
    }
    const auto totals=Integrals(result,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    work.transport+=Wall(start);return result;
  }
};
BoundaryOptions BoundaryParse(int argc,char** argv,bool custom_wall_source=false)
{
  BoundaryOptions out;std::vector<char*> common{argv[0]};
  for(int i=1;i<argc;++i)
  {
    const std::string flag=argv[i];
    if(flag=="--scenario"||flag=="--left-boundary"||flag=="--right-boundary"||flag=="--exterior-stress"||
       flag=="--open-closure"||flag=="--wall-closure"||flag=="--wall-source"||flag=="--wall-flux"||flag=="--positivity-subdivisions")
    {
      if(++i>=argc)throw std::runtime_error("missing boundary option");const std::string value=argv[i];
      if(flag=="--scenario")out.scenario=value;else if(flag=="--left-boundary")out.left=value;
      else if(flag=="--right-boundary")out.right=value;else if(flag=="--exterior-stress")out.stress=value;
      else if(flag=="--open-closure")out.open_closure=value;else if(flag=="--wall-closure")out.wall_closure=value;
      else if(flag=="--wall-source")out.wall_source=value;else if(flag=="--wall-flux")out.wall_flux=value;
      else out.positivity_subdivisions=std::stoi(value);
    }
    else common.push_back(argv[i]);
  }
  out.compatibility=CompatibilityParse(int(common.size()),common.data());
  auto admitted=[](const std::string& value,std::initializer_list<const char*> choices)
  {for(const auto* choice:choices)if(value==choice)return true;return false;};
  if(!admitted(out.scenario,{"existing","wall-compression","application-tanh"})||!admitted(out.left,{"fixed","outflow","wall","characteristic","nonreflecting"})||
     !admitted(out.right,{"fixed","outflow","wall","characteristic","nonreflecting"})||!admitted(out.stress,{"legacy","trace","zero"})||
     !admitted(out.open_closure,{"natural","screened"})||!admitted(out.wall_closure,{"natural","pressure-balance"})||
     !admitted(out.wall_source,{"interior","zero"})||!admitted(out.wall_flux,{"mirror","traction-llf"})||out.positivity_subdivisions<0)
    throw std::runtime_error("invalid boundary control");
  if(out.scenario=="wall-compression"&&out.right!="wall")throw std::runtime_error("wall-compression requires a right wall");
  if(out.scenario=="application-tanh")out.compatibility.track.initial="tanh";
  if(out.wall_source=="zero"&&((out.compatibility.formulation=="baseline"&&!custom_wall_source)||out.compatibility.formulation=="weak-product"))
    throw std::runtime_error("wall source lifting requires a lifted formulation");
  if(out.compatibility.formulation=="weak-product"&&(out.open_closure!="natural"||out.wall_closure!="natural"))
    throw std::runtime_error("weak-product does not implement boundary closure loads");
  return out;
}
void BoundarySelfTest()
{
  CompatibilitySelfTest();
  BoundaryOptions o;o.compatibility.track.initial="simple-wave";o.compatibility.track.width=.04;
  o.compatibility.track.mode="uniform";o.compatibility.track.closure_space="l2";o.compatibility.track.closure_boundary="natural";
  o.compatibility.track.base.elements=16;o.compatibility.track.base.order=7;o.compatibility.track.base.dt=1.e-5;o.compatibility.track.base.final_time=.0001;
  for(const std::string formulation:{"baseline","mixed","paired"})
  {
    o.compatibility.formulation=formulation;o.stress="trace";
    CompatibilityProblem old(o.compatibility);BoundaryProblem current(o);const auto a=CompatibilityRun(old),b=CompatibilityRun(current);
    if(!a.failure.empty()||!b.failure.empty()||Norm(a.u)-Norm(b.u)>1.e-10)throw std::runtime_error("natural boundary trajectory failed");
    double difference=0.;for(std::size_t i=0;i<a.u.size();++i)difference=std::max(difference,std::abs(a.u[i]-b.u[i]));
    if(difference>1.e-11)throw std::runtime_error("natural boundary replacement parity failed");
  }
  o.scenario="wall-compression";o.right="wall";o.stress="trace";o.compatibility.formulation="mixed";
  BoundaryProblem wall(o);State state=wall.Initial(),sigma(wall.ns,.7);state[wall.Id(wall.ne-1,wall.p,1)]=.1;
  const auto flux=wall.BoundaryFlux(state,sigma,1);
  if(std::abs(flux[0])>1.e-13||std::abs(flux[3])>1.e-13)throw std::runtime_error("copied stress wall flux is not impermeable/adiabatic");
  // Manufactured constant reaction solution with matching Robin boundary load.
  for(int e=0;e<wall.ne;++e)for(int i=0;i<wall.n;++i)
  {state[wall.Id(e,i,0)]=2.;state[wall.Id(e,i,1)]=0.;state[wall.Id(e,i,2)]=0.;state[wall.Id(e,i,3)]=3.;}
  BandMatrix matrix;State load;TrackWork work;wall.AssembleClosure(state,0.,work,matrix,load);
  const double exact=.3;for(int e=0;e<wall.ne;++e)for(std::size_t q=0;q<wall.weights.size();++q)for(int i=0;i<wall.n;++i)
    load[e*wall.n+i]+=(wall.edges[e+1]-wall.edges[e])*wall.weights[q]*wall.shapes[q][i]*exact/2.;
  work.current_source_integral=(o.compatibility.track.right-o.compatibility.track.left)*exact/2.;
  const std::array<double,2> robin{{.02,.03}},neumann{{.02*exact,.03*exact}};
  const State solution=wall.SolveCurrentSystem(state,matrix,load,work,Clock::now(),robin,neumann);
  for(double value:solution)if(std::abs(value-exact)>1.e-10)throw std::runtime_error("Robin/load sign or constant response failed");
  {
    BoundaryOptions a;a.scenario="application-tanh";a.left="nonreflecting";a.right="fixed";
    auto& t=a.compatibility.track;t.initial="tanh";t.left=-4.;t.right=4.;t.center=0.;
    t.width=.3;t.mach=5.;t.speed=.4;t.base.elements=32;t.base.order=7;
    t.closure_space="l2";t.closure_boundary="natural";
    BoundaryProblem app(a);
    // Independent application caller geometry: its custom mesh does not
    // change the factory's original [-30,10]x[-20,20] width calculation.
    auto shared=mfem::GetShockVortexInoueHattoriInitCustom(
        gamma_euler,32,1,5.,0.,.3/1.25,-30.,10.,-20.,20.,4.,0.,0.);
    for(double x:{-4.,-.31,0.,.17,4.})
    {
      mfem::Vector at(2),expected(4);at[0]=x;at[1]=0.;shared(at,expected);
      expected[3]+=.4*expected[1]+.5*.4*.4*expected[0];expected[1]+=.4*expected[0];
      const auto actual=app.InitialAt(x);
      for(int c=0;c<4;++c)if(std::abs(actual[c]-expected[c])>2.e-13*std::max(1.,std::abs(expected[c])))
        throw std::runtime_error("application shared initializer mismatch");
    }
    const auto midpoint=app.InitialAt(0.);
    if(std::abs(midpoint[0]-3.)>1.e-13||std::abs(midpoint[1]-(-5.*std::sqrt(gamma_euler)+1.2))>1.e-13)
      throw std::runtime_error("application initializer did not preserve momentum blending");
    State u=app.Initial(),s(app.ns,.7);const auto ref=app.InitialAt(-40.*t.width);
    mfem::Vector reference(4),inner(4),normal(2),outer(4);normal=0.;normal[0]=-1.;
    for(int c=0;c<4;++c){reference[c]=ref[c];inner[c]=ref[c];}
    inner[3]+=.12*inner[1]+.5*.12*.12*inner[0];inner[1]+=.12*inner[0];
    for(int c=0;c<4;++c)u[app.Id(0,0,c)]=inner[c];
    mfem::EvaluateEulerExteriorState(mfem::EulerExteriorStateKind::NonReflectingOutlet,
        inner,normal,2,gamma_euler,&reference,1.,outer,floor_value,floor_value);
    double ai=0.,ao=0.;const auto fi=app.physics.Flux(inner.GetData(),.7,ai),fo=app.physics.Flux(outer.GetData(),.7,ao);
    const auto actual=app.BoundaryFlux(u,s,0);const double speed=app.physics.FaceSpeed(ai,ao);
    for(int c=0;c<4;++c)
    {const double expected=.5*(fi[c]+fo[c]+speed*(outer[c]-inner[c]));
      if(std::abs(actual[c]-expected)>2.e-11)throw std::runtime_error("application nonreflecting flux differs from shared host boundary");}
  }
  std::cout<<"DG boundary evolution self-test passed\n";
}
template<class ProblemType>
void BoundaryWrite(const ProblemType& problem,const TrackResult& result,std::string additional={})
{
  const auto& options=problem.boundary;
  std::ostringstream original;TrackWrite(problem,result,original);
    std::string data=original.str();data.erase(data.rfind('}'));std::ostringstream extra;extra<<std::setprecision(17);
    extra<<",\"boundary_scenario\":\""<<options.scenario<<"\",\"left_boundary\":\""<<options.left<<"\",\"right_boundary\":\""<<options.right
      <<"\",\"exterior_stress\":\""<<options.stress<<"\",\"open_closure\":\""<<options.open_closure<<"\",\"wall_closure\":\""<<options.wall_closure
      <<"\",\"wall_source\":\""<<options.wall_source<<"\",\"wall_flux\":\""<<options.wall_flux<<"\",\"compatibility_formulation\":\""<<options.compatibility.formulation
      <<"\",\"gradient_direction\":\""<<options.compatibility.direction<<"\",\"stress_energy_trace\":\""<<options.compatibility.energy_trace
      <<"\",\"maximum_wall_mass_flux\":"<<problem.max_wall_mass<<",\"maximum_wall_energy_flux\":"<<problem.max_wall_energy
      <<",\"maximum_wall_normal_velocity\":"<<problem.max_wall_velocity<<",\"boundary_flux_calls\":"<<problem.boundary_calls
      <<",\"positivity_subdivisions\":"<<options.positivity_subdivisions<<",\"projection_seconds\":"<<problem.projection_seconds
      <<",\"velocity_projections\":"<<problem.velocity_projections<<",\"maximum_boundary_constant_test_defect\":"<<problem.max_mean_defect<<"\n}\n";
    data+=extra.str();
    if(!additional.empty())data.insert(data.rfind('}'),additional);const auto pos=data.find("selective_high_order_dg_conservative_rezone");
    if(pos!=std::string::npos)data.replace(pos,std::string("selective_high_order_dg_conservative_rezone").size(),"fixed_mesh_dg_boundary_control");
    auto replace_once=[&](const std::string& before,const std::string& after)
    {const auto position=data.find(before);if(position!=std::string::npos)data.replace(position,before.size(),after);};
    replace_once("\"euler_boundary\":\"outflow\"","\"euler_boundary\":\"custom\"");
    if(options.open_closure!="natural"||options.wall_closure!="natural")
      replace_once("\"closure_boundary\":\"natural\"","\"closure_boundary\":\"custom\"");
    if(options.scenario=="wall-compression"||options.scenario=="application-tanh")
    {
      replace_once("\"initial\":\""+problem.o.initial+"\"","\"initial\":\""+options.scenario+"\"");
      const std::string key="\"unregularized_euler_breaking_time\":";const auto position=data.find(key);
      if(position!=std::string::npos)
      {const auto end=data.find_first_of(",}\n",position+key.size());data.replace(position+key.size(),end-position-key.size(),"null");}
    }
    if(options.compatibility.track.output.empty())std::cout<<data;
    else{std::ofstream output(options.compatibility.track.output);if(!output)throw std::runtime_error("cannot open boundary output");output<<data;}
}

}
#ifndef DG_IGR_DG_BOUNDARY_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    const auto options=BoundaryParse(argc,argv);if(options.compatibility.track.self_test){BoundarySelfTest();return 0;}
    const auto start=Clock::now();BoundaryProblem problem(options);const double setup=Wall(start);
    auto result=CompatibilityRun(problem);result.work.setup=setup;BoundaryWrite(problem,result);
    return result.failure.empty()?0:2;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

#endif

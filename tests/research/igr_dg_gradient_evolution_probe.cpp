// Same-DOF velocity-gradient/conserved-mean evolution on the shared full-energy
// Euler + instantaneous IGR research adapter. Not a DGSEM implementation.
#define DG_IGR_HP_PATCH_LIBRARY_ONLY
#include "igr_dg_hp_patch_probe.cpp"
#undef DG_IGR_HP_PATCH_LIBRARY_ONLY
namespace {
#include "igr_dg_gradient_evolution_probe.hpp"
State GradientRead(const std::string& path,std::size_t count)
{
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open gradient input");State a(count);for(double& x:a)if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid gradient input");in>>std::ws;if(!in.eof())throw std::runtime_error("extra gradient input");return a;
}
void GradientGeometry(GradientEvolutionProblem& pr,const State& edges)
{
  if(edges.size()!=std::size_t(pr.ne+1)||std::abs(edges.front()-pr.o.left)>1.e-13||std::abs(edges.back()-pr.o.right)>1.e-13)throw std::runtime_error("gradient physical geometry mismatch");
  PatchGeometry g;g.edges=edges;g.left=edges.front();g.right=edges.back();g.inner_elements=pr.ne;g.hmin=INFINITY;
  for(int e=0;e<pr.ne;++e){double h=edges[e+1]-edges[e];if(!(h>0.))throw std::runtime_error("invalid gradient cell width");g.hmin=std::min(g.hmin,h);g.hmax=std::max(g.hmax,h);if(e){double previous=edges[e]-edges[e-1];g.maximum_adjacent_ratio=std::max(g.maximum_adjacent_ratio,std::max(h/previous,previous/h));}}
  SetPatchGeometry(pr,g,g.hmax-g.hmin<1.e-13?"uniform":"saved-nonuniform");pr.translated_coordinates.clear();
}
void GradientSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=15;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=25;mo.retained_order=mo.stress_order=5;
  GradientEvolutionProblem chain(mo,.25,.25,"chain"),candidate(mo,.25,.25,"commuting");CurvatureProblem base(mo,.25,.25);
  State physical(chain.ne*chain.n*4,0.);
  for(int e=0;e<chain.ne;++e)for(int i=0;i<chain.n;++i)
  {
    const double z=chain.physics.basis.nodes[i],rho=1.2+.05*z+.02*z*z,v=.3+.02*e+.03*std::pow(z,5),vy=.04+.015*std::pow(z,4),pressure=1.1+.02*std::pow(z,5);const int j=chain.Id(e,i,0);
    physical[j]=rho;physical[j+1]=rho*v;physical[j+2]=rho*vy;physical[j+3]=pressure/(gamma_euler-1.)+.5*rho*(v*v+vy*vy);
  }
  State standard=chain.state_transfer.Apply(chain.state_transfer.E,chain.state_transfer.Apply(chain.state_transfer.P,physical,chain.ne,4),chain.ne,4);
  State packed=chain.Pack(standard),back=chain.Moments(packed);double roundtrip=0.,physical_error=0.,gradient_error=0.;
  const State rebuilt=chain.PrimitiveMomentProblem::Reconstruct(back),low=chain.state_transfer.Apply(chain.state_transfer.P,packed,chain.ne,4);
  for(std::size_t i=0;i<back.size();++i){roundtrip=std::max(roundtrip,std::abs(back[i]-standard[i]));physical_error=std::max(physical_error,std::abs(rebuilt[i]-physical[i]));}
  for(int e=0;e<chain.ne;++e)for(std::size_t q=0;q<chain.weights.size();++q)
  {
    double z=0.;for(int i=0;i<chain.n;++i)z+=chain.shapes[q][i]*chain.physics.basis.nodes[i];const State P=chain.Legendre(2*z-1.,chain.retained.p-1);double g=0.;for(int k=0;k<chain.retained.p;++k)g+=P[k]*low[(e*chain.retained.n+k+1)*4+1];
    gradient_error=std::max(gradient_error,std::abs(g-.15*std::pow(z,4)/(chain.edges[e+1]-chain.edges[e])));
  }
  TrackWork wb,wc,wn;std::array<double,4> bb{},bc{},bn{};State sigma=base.Closure(standard,0.,wb);const State expected=base.Residual(standard,sigma,0.,wb,bb);
  const State rate=chain.Residual(packed,sigma,0.,wc,bc);double parity=0.;for(std::size_t i=0;i<expected.size();++i)parity=std::max(parity,std::abs(chain.last_standard_rhs[i]-expected[i]));
  State differential_errors;
  for(double eps:{1.e-5,5.e-6,2.5e-6})
  {
    State plus=packed,minus=packed;for(std::size_t i=0;i<packed.size();++i){plus[i]+=eps*rate[i];minus[i]-=eps*rate[i];}
    const State fp=chain.Moments(plus),fm=chain.Moments(minus);double error=0.;for(std::size_t i=0;i<fp.size();++i)error=std::max(error,std::abs((fp[i]-fm[i])/(2*eps)-expected[i]));differential_errors.push_back(error);
  }
  const double differential=differential_errors.back();

  State active=candidate.Pack(standard);candidate.Residual(active,sigma,0.,wn,bn);double fixed_components=0.;for(std::size_t i=0;i<expected.size();++i)if(i%4==0||i%4==3)fixed_components=std::max(fixed_components,std::abs(candidate.last_standard_rhs[i]-expected[i]));
  // With spatially constant rho and v in Pp, s_rho is in P_(p-1).
  // Dropping its projection commutator must then change no discrete RHS.
  GradientEvolutionProblem weighted(mo,.25,.25,"continuum-weighted");State constant_physical=physical;
  for(int e=0;e<chain.ne;++e)for(int i=0;i<chain.n;++i)
  {
    const int j=chain.Id(e,i,0);const double v=physical[j+1]/physical[j],vy=physical[j+2]/physical[j],thermal=physical[j+3]-.5*physical[j]*(v*v+vy*vy);
    constant_physical[j]=1.2;constant_physical[j+1]=1.2*v;constant_physical[j+2]=1.2*vy;constant_physical[j+3]=thermal+.6*(v*v+vy*vy);
  }
  const State constant_standard=chain.state_transfer.Apply(chain.state_transfer.E,chain.state_transfer.Apply(chain.state_transfer.P,constant_physical,chain.ne,4),chain.ne,4);
  TrackWork ww;const State constant_sigma=base.Closure(constant_standard,0.,ww);std::array<double,4> constant_budget{};
  const State constant_expected=base.Residual(constant_standard,constant_sigma,0.,ww,constant_budget);weighted.Residual(weighted.Pack(constant_standard),constant_sigma,0.,ww,constant_budget);
  double weighted_constant_parity=0.;for(std::size_t i=0;i<constant_expected.size();++i)weighted_constant_parity=std::max(weighted_constant_parity,std::abs(constant_expected[i]-weighted.last_standard_rhs[i]));
  // An active limiter operates in ordinary conservative moments and repacks.
  State bad=standard;for(int e=0;e<chain.ne;++e)for(int i=0;i<chain.n;++i)bad[chain.Id(e,i,3)]+=8.*(chain.physics.basis.nodes[i]-.5);
  State bad_packed=chain.Pack(bad);const auto before=chain.Integrals(bad_packed,chain.edges);TrackWork wl;chain.Limit(bad_packed,wl);const auto after=chain.Integrals(bad_packed,chain.edges);double limiter_mean=0.;for(int c=0;c<4;++c)limiter_mean=std::max(limiter_mean,std::abs(after[c]-before[c]));
  std::cout<<std::setprecision(17)<<"{\"moment_roundtrip\":"<<roundtrip<<",\"physical_reconstruction\":"<<physical_error<<",\"gradient_coefficients\":"<<gradient_error<<",\"exact_chain_rhs_parity\":"<<parity<<",\"independent_reconstruction_differential\":"<<differential<<",\"differential_error_eps1e5\":"<<differential_errors[0]<<",\"differential_error_eps5e6\":"<<differential_errors[1]<<",\"rho_energy_unchanged\":"<<fixed_components<<",\"weighted_volume_projection_identity\":"<<candidate.maximum_weighted_projection_defect<<",\"candidate_mean_defect\":"<<candidate.maximum_mean_defect<<",\"candidate_rhs_conservation\":"<<wn.max_rhs_defect<<",\"weighted_continuum_constant_density_parity\":"<<weighted_constant_parity<<",\"active_limiter_mean\":"<<limiter_mean<<"}\n";
  if(roundtrip>2.e-11||physical_error>2.e-11||gradient_error>2.e-11||parity>1.e-9||weighted_constant_parity>1.e-9||differential>2.e-6||differential_errors[1]>.3*differential_errors[0]||differential_errors[2]>.3*differential_errors[1]||fixed_components>1.e-10||candidate.maximum_weighted_projection_defect>1.e-9||candidate.maximum_mean_defect>1.e-10||wn.max_rhs_defect>1.e-8||!wl.limited||limiter_mean>1.e-10)throw std::runtime_error("gradient coordinate reconstruction/chain/mean/limiter check failed");

}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    double slope=.25,curvature=.25,time=0.;bool frozen=false;std::string input,edge_input,method="commuting";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--residual-only"){frozen=true;continue;}
      if(flag=="--gradient-acceleration"||flag=="--slope-strength"||flag=="--curvature-strength"||flag=="--probe-time"||flag=="--moments-in"||flag=="--edges-in")
      {if(++i==argc)throw std::runtime_error("missing gradient option");const std::string value=argv[i];if(flag=="--gradient-acceleration")method=value;else if(flag=="--slope-strength")slope=std::stod(value);else if(flag=="--curvature-strength")curvature=std::stod(value);else if(flag=="--probe-time")time=std::stod(value);else if(flag=="--moments-in")input=value;else edge_input=value;}
      else common.push_back(argv[i]);
    }
    MemoryOptions mo;mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;
    if(t.self_test){GradientSelfTest();return 0;}const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;mo.input=input;mo.start_time=time;
    if(time<0.||time>t.base.final_time||(!input.empty()&&edge_input.empty())||(frozen&&input.empty()))throw std::runtime_error("saved state/time/edges required for frozen or restart mode");
    if(mo.recovery.boundary.scenario!="application-tanh"||mo.recovery.boundary.left!="nonreflecting"||mo.recovery.boundary.right!="fixed")throw std::runtime_error("gradient front qualified only for original application-tanh boundaries");
    const auto start=Clock::now();GradientEvolutionProblem pr(mo,slope,curvature,method);if(!edge_input.empty())GradientGeometry(pr,GradientRead(edge_input,pr.ne+1));const double setup=Wall(start);
    TrackResult result;State rhs;std::array<double,4> budget{};
    if(frozen){result.time=time;result.u=pr.Initial();result.sigma=pr.Closure(result.u,time,result.work);rhs=pr.Residual(result.u,result.sigma,time,result.work,budget);}
    else result=MemoryRun(pr);
    result.work.setup=setup;State coordinates,moments;
    if(!result.u.empty()){coordinates=pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);const State standard=pr.Moments(result.u);moments=pr.state_transfer.Apply(pr.state_transfer.P,standard,pr.ne,4);result.u=pr.PrimitiveMomentProblem::Reconstruct(standard);}
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"joint-density-weighted-primitive\",\"volume_projection\":\"gradient-coordinates\",\"retained_order\":"<<degree
      <<",\"primitive_source_faces\":\"central\",\"moment_operator\":\"direct\",\"stress_order\":"<<degree<<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns
      <<",\"sigma_serialized_order\":"<<pr.p<<",\"injected_state_time\":"<<time<<",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"gradient_acceleration\":\""<<method<<"\",\"gradient_mode\":\""<<(frozen?"frozen":"advance")<<"\""
      <<",\"coordinate_seconds\":"<<pr.coordinate_seconds<<",\"coordinate_conversions\":"<<pr.coordinate_conversions<<",\"acceleration_seconds\":"<<pr.acceleration_seconds<<",\"acceleration_cells\":"<<pr.acceleration_cells
      <<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"reconstruction_cells\":"<<pr.reconstruction_cells<<",\"maximum_coordinate_mean_defect\":"<<pr.maximum_mean_defect<<",\"maximum_weighted_projection_defect\":"<<pr.maximum_weighted_projection_defect
      <<",\"last_added_kinetic_power\":"<<pr.last_added_kinetic_power;
    extra<<",\"moment_coefficients\":";TrackArray(extra,moments);extra<<",\"gradient_coordinates\":";TrackArray(extra,coordinates);
    if(frozen)
    {
      extra<<",\"base_rhs\":";TrackArray(extra,pr.state_transfer.Apply(pr.state_transfer.P,pr.last_standard_rhs,pr.ne,4));
      extra<<",\"native_conservative_rhs\":";TrackArray(extra,pr.state_transfer.Apply(pr.state_transfer.P,pr.last_native_rhs,pr.ne,4));
      extra<<",\"coordinate_rhs\":";TrackArray(extra,pr.state_transfer.Apply(pr.state_transfer.P,rhs,pr.ne,4));
      extra<<",\"velocity_acceleration\":";TrackArray(extra,pr.last_acceleration);extra<<",\"volume_target_projection\":";TrackArray(extra,pr.last_target);
      extra<<",\"boundary_budget\":";TrackArray(extra,State(budget.begin(),budget.end()));
    }
    BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

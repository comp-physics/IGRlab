// Full-energy weak DG: thermal Euler two-point flux plus separate IGR work.
#define DG_IGR_HP_PATCH_LIBRARY_ONLY
#include "igr_dg_hp_patch_probe.cpp"
#undef DG_IGR_HP_PATCH_LIBRARY_ONLY
#include "numerics/dgsem/dgsem_entropy_flux.hpp"
namespace {
#include "igr_dg_entropy_coupled_probe.hpp"
State EntropyCoupledRead(const std::string& path,std::size_t count)
{
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open entropy input");State a(count);for(double& x:a)if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid entropy input");in>>std::ws;if(!in.eof())throw std::runtime_error("extra entropy input");return a;
}
void EntropyCoupledGeometry(EntropyCoupledProblem& pr,const State& edges)
{
  if(edges.size()!=std::size_t(pr.ne+1)||std::abs(edges.front()-pr.o.left)>1.e-13||std::abs(edges.back()-pr.o.right)>1.e-13)throw std::runtime_error("entropy physical geometry mismatch");
  PatchGeometry g;g.edges=edges;g.left=edges.front();g.right=edges.back();g.inner_elements=pr.ne;g.hmin=INFINITY;
  for(int e=0;e<pr.ne;++e){double h=edges[e+1]-edges[e];if(!(h>0.))throw std::runtime_error("invalid entropy cell width");g.hmin=std::min(g.hmin,h);g.hmax=std::max(g.hmax,h);if(e){double previous=edges[e]-edges[e-1];g.maximum_adjacent_ratio=std::max(g.maximum_adjacent_ratio,std::max(h/previous,previous/h));}}
  SetPatchGeometry(pr,g,g.hmax-g.hmin<1.e-13?"uniform":"saved-nonuniform");
}
double EntropyCoupledIntegral(const EntropyCoupledProblem& pr,const State& moments)
{
  const State physical=pr.Reconstruct(moments);double value=0.;for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)value+=(pr.edges[e+1]-pr.edges[e])*pr.weights[q]*EntropyCoupledState(pr.Point(physical,e,pr.shapes[q]),0.).eta;return value;
}
void EntropyCoupledSelfTest()
{
  double diagonal=0.,helper=0.,pair=0.,symmetry=0.,arithmetic_parity=0.,mean=0.,sbp_null=0.,work_identity=0.,entropy_derivative=0.,nontrivial=0.;
  for(int degree:{3,7,11})
  {
    MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=3*degree;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=21;mo.retained_order=mo.stress_order=degree;
    EntropyCoupledProblem base(mo,.25,.25,"original"),ordinary(mo,.25,.25,"arithmetic"),active(mo,.25,.25,"ranocha");
    for(int k=0;k<20;++k)
    {
      auto state=[&](double z){const double r=1.+.3*z,p=1.2+.2*std::sin(z),v=.4+.2*z,vy=-.1+.03*z;return std::array<double,4>{r,r*v,r*vy,p/(gamma_euler-1.)+.5*r*(v*v+vy*vy)};};
      const auto ua=state(.01*k),ub=state(k%3==0?.01*k+.00000001:.02*k+.1);auto a=EntropyCoupledState(ua,.1+.01*k),b=EntropyCoupledState(ub,.2-.003*k);
      const auto fa=EntropyCoupledFlux(a,a,false),fab=EntropyCoupledFlux(a,b,false),fba=EntropyCoupledFlux(b,a,false);double wave=0.;const auto exact=base.physics.Flux(ua.data(),a.sigma,wave);
      mfem::Vector ul(const_cast<double*>(ua.data()),4),ur(const_cast<double*>(ub.data()),4),shared;mfem::dgsem::RanochaFlux2D(ul,ur,0,gamma_euler,0.,0.,shared);
      shared(1)+=.5*(a.sigma+b.sigma);shared(3)+=.25*(a.sigma+b.sigma)*(a.vx+b.vx);
      double contraction=0.;for(int c=0;c<4;++c){diagonal=std::max(diagonal,std::abs(fa[c]-exact[c]));helper=std::max(helper,std::abs(fab[c]-shared(c)));symmetry=std::max(symmetry,std::abs(fab[c]-fba[c]));contraction+=(b.w[c]-a.w[c])*fab[c];}
      const double expected=b.u[1]-a.u[1]+.25*(a.sigma+b.sigma)*(a.beta+b.beta)*(b.vx-a.vx);pair=std::max(pair,std::abs(contraction-expected));
    }
    State physical(base.ne*base.n*4,0.);for(int e=0;e<base.ne;++e)for(int i=0;i<base.n;++i)
    {
      const double z=base.physics.basis.nodes[i],rho=1.1+.04*z*z*z+.01*e,v=.3+.04*z*z*z+.02*e,vy=.04+.01*z*z,p=1.+.03*z+.005*e;
      const int j=base.Id(e,i,0);physical[j]=rho;physical[j+1]=rho*v;physical[j+2]=rho*vy;physical[j+3]=p/(gamma_euler-1.)+.5*rho*(v*v+vy*vy);
    }
    const State moments=base.state_transfer.Apply(base.state_transfer.E,base.state_transfer.Apply(base.state_transfer.P,physical,base.ne,4),base.ne,4);TrackWork wb,wo,wa;const State sigma=base.Closure(moments,0.,wb);std::array<double,4> bb{},bo{},ba{};
    const State rb=base.Residual(moments,sigma,0.,wb,bb),ro=ordinary.Residual(moments,sigma,0.,wo,bo),ra=active.Residual(moments,sigma,0.,wa,ba);
    arithmetic_parity=std::max(arithmetic_parity,PatchRelativeDifference(rb,ro));nontrivial=std::max(nontrivial,PatchRelativeDifference(rb,ra));mean=std::max({mean,ordinary.maximum_entropy_mean,active.maximum_entropy_mean,wa.max_rhs_defect});sbp_null=std::max(sbp_null,active.entropy_operator.sbp_null_defect);work_identity=std::max(work_identity,active.maximum_entropy_pair_work_defect);
    State direction(moments.size());for(std::size_t i=0;i<direction.size();++i)direction[i]=.02*std::sin(.17*i);direction=base.state_transfer.Apply(base.state_transfer.E,base.state_transfer.Apply(base.state_transfer.P,direction,base.ne,4),base.ne,4);
    const double rate=active.EntropyRate(moments,direction),epsilon=2.e-5;State plus=moments,minus=moments;for(std::size_t i=0;i<moments.size();++i){plus[i]+=epsilon*direction[i];minus[i]-=epsilon*direction[i];}
    const double finite=(EntropyCoupledIntegral(active,plus)-EntropyCoupledIntegral(active,minus))/(2.*epsilon);entropy_derivative=std::max(entropy_derivative,std::abs(rate-finite));
  }
  if(diagonal>1.e-11||helper>1.e-7||pair>1.e-11||symmetry>1.e-13||arithmetic_parity>1.e-9||mean>1.e-9||sbp_null>1.e-10||work_identity>1.e-9||entropy_derivative>1.e-8||nontrivial<1.e-9)throw std::runtime_error("coupled entropy flux/SBP/manifold selftest failed");
  std::cout<<std::setprecision(17)<<"{\"physical_diagonal\":"<<diagonal<<",\"thermal_helper_parity\":"<<helper<<",\"pair_entropy_work_identity\":"<<pair<<",\"pair_symmetry\":"<<symmetry<<",\"arithmetic_original_parity\":"<<arithmetic_parity<<",\"conservation\":"<<mean<<",\"sbp_constant\":"<<sbp_null<<",\"sbp_pair_work_identity\":"<<work_identity<<",\"manifold_entropy_derivative\":"<<entropy_derivative<<",\"nontrivial_ranocha_change\":"<<nontrivial<<"}\n";
}
}
#ifndef DG_IGR_ENTROPY_COUPLED_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    double slope=.25,curvature=.25,time=0.;bool frozen=false;std::string input,edge_input,mode="ranocha";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--residual-only"){frozen=true;continue;}
      if(flag=="--entropy-volume"||flag=="--slope-strength"||flag=="--curvature-strength"||flag=="--probe-time"||flag=="--moments-in"||flag=="--edges-in")
      {if(++i==argc)throw std::runtime_error("missing entropy option");const std::string value=argv[i];if(flag=="--entropy-volume")mode=value;else if(flag=="--slope-strength")slope=std::stod(value);else if(flag=="--curvature-strength")curvature=std::stod(value);else if(flag=="--probe-time")time=std::stod(value);else if(flag=="--moments-in")input=value;else edge_input=value;}
      else common.push_back(argv[i]);
    }
    MemoryOptions mo;mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;if(t.self_test){EntropyCoupledSelfTest();return 0;}
    const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;mo.input=input;mo.start_time=time;
    if(time<0.||time>t.base.final_time||(!input.empty()&&edge_input.empty()))throw std::runtime_error("saved state requires compatible time and edges");
    if(mo.recovery.boundary.scenario!="application-tanh"||mo.recovery.boundary.left!="nonreflecting"||mo.recovery.boundary.right!="fixed")throw std::runtime_error("entropy front qualified only for original application-tanh boundaries");
    const auto start=Clock::now();EntropyCoupledProblem pr(mo,slope,curvature,mode);if(!edge_input.empty())EntropyCoupledGeometry(pr,EntropyCoupledRead(edge_input,pr.ne+1));const double setup=Wall(start);
    TrackResult result;State rhs,correction;std::array<double,4> budget{};double entropy_rate=0.,correction_rate=0.;
    if(frozen){result.time=time;result.u=pr.Initial();result.sigma=pr.Closure(result.u,time,result.work);rhs=pr.Residual(result.u,result.sigma,time,result.work,budget);correction=pr.last_entropy_correction;entropy_rate=pr.EntropyRate(result.u,rhs);correction_rate=pr.EntropyRate(result.u,correction);}
    else result=MemoryRun(pr);
    result.work.setup=setup;const State moments=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);if(!result.u.empty())result.u=pr.Reconstruct(result.u);
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"joint-density-weighted-primitive\",\"volume_projection\":\"thermal-euler-plus-mechanical-stress-flux-differencing\",\"entropy_volume\":\""<<mode<<"\",\"entropy_proof_claimed\":false,\"retained_order\":"<<degree
      <<",\"primitive_source_faces\":\"central\",\"moment_operator\":\"direct\",\"stress_order\":"<<degree<<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns<<",\"sigma_serialized_order\":"<<pr.p
      <<",\"injected_state_time\":"<<time<<",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"entropy_mode\":\""<<(frozen?"frozen":"advance")<<"\",\"entropy_seconds\":"<<pr.entropy_seconds<<",\"entropy_cells\":"<<pr.entropy_cells<<",\"entropy_pairs\":"<<pr.entropy_pairs
      <<",\"entropy_quadrature_nodes\":"<<pr.entropy_operator.nq<<",\"entropy_extended_nodes\":"<<pr.entropy_operator.nn<<",\"maximum_entropy_mean\":"<<pr.maximum_entropy_mean<<",\"maximum_entropy_pair_work_defect\":"<<pr.maximum_entropy_pair_work_defect<<",\"sbp_constant_defect\":"<<pr.entropy_operator.sbp_null_defect
      <<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"local_velocity_factors\":"<<pr.local_velocity_factors<<",\"direct_assembly_seconds\":"<<pr.direct_assembly_seconds<<",\"moment_transfer_seconds\":"<<pr.transfer_seconds<<",\"galerkin_seconds\":"<<pr.galerkin_seconds
      <<",\"slope_penalty_seconds\":"<<pr.slope_seconds<<",\"curvature_seconds\":"<<pr.curvature_seconds<<",\"maximum_slope_identity_error\":"<<pr.maximum_identity_error<<",\"maximum_curvature_identity_error\":"<<pr.curvature_identity_error<<",\"moment_coefficients\":";TrackArray(extra,moments);
    if(frozen){extra<<",\"base_rhs\":";TrackArray(extra,pr.state_transfer.Apply(pr.state_transfer.P,rhs,pr.ne,4));extra<<",\"volume_correction\":";TrackArray(extra,pr.state_transfer.Apply(pr.state_transfer.P,correction,pr.ne,4));extra<<",\"boundary_budget\":";TrackArray(extra,State(budget.begin(),budget.end()));extra<<",\"manifold_gas_mathematical_entropy_rate\":"<<entropy_rate<<",\"correction_gas_mathematical_entropy_rate\":"<<correction_rate<<",\"sbp_stress_transfer\":"<<pr.last_entropy_pair_stress;}
    BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif

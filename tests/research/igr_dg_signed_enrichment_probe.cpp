// Signed full-Euler mixed-degree enrichment; test-only weak-DG adapter.
#define DG_IGR_HP_PATCH_LIBRARY_ONLY
#include "igr_dg_hp_patch_probe.cpp"
#undef DG_IGR_HP_PATCH_LIBRARY_ONLY
#include "numerics/dgsem/dgsem_entropy_flux.hpp"
#include <memory>
namespace {
#include "igr_dg_entropy_coupled_probe.hpp"
#include "igr_dg_signed_enrichment_probe.hpp"
State SignedRead(const std::string& path,std::size_t count)
{
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open signed input");State a(count);
  for(double& x:a)if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid signed input");
  in>>std::ws;if(!in.eof())throw std::runtime_error("extra signed input");return a;
}
void SignedGeometry(SignedEnrichmentProblem& pr,const State& edges)
{
  if(edges.size()!=std::size_t(pr.ne+1)||std::abs(edges.front()-pr.o.left)>1.e-13||std::abs(edges.back()-pr.o.right)>1.e-13)throw std::runtime_error("signed physical geometry mismatch");
  PatchGeometry g;g.edges=edges;g.left=edges.front();g.right=edges.back();g.inner_elements=pr.ne;g.hmin=INFINITY;
  for(int e=0;e<pr.ne;++e){double h=edges[e+1]-edges[e];if(!(h>0.))throw std::runtime_error("invalid signed cell width");g.hmin=std::min(g.hmin,h);g.hmax=std::max(g.hmax,h);if(e){double previous=edges[e]-edges[e-1];g.maximum_adjacent_ratio=std::max(g.maximum_adjacent_ratio,std::max(h/previous,previous/h));}}
  const std::string mode=g.hmax-g.hmin<1.e-13?"uniform":"saved-nonuniform";SetPatchGeometry(pr,g,mode);
  pr.coarse.edges=edges;pr.coarse.base_uniform_edges=edges;pr.coarse.initial_graded_hmin=g.hmin;pr.coarse.o.mode=mode;
  if(pr.initial_base)SetPatchGeometry(*pr.initial_base,g,mode);pr.ClearEnrichmentCache();
}
double SignedEntropy(const SignedEnrichmentProblem& pr,const State& external)
{
  const State physical=pr.Physical(external);double value=0.;
  for(int e=0;e<pr.ne;++e)for(std::size_t j=0;j<pr.weights.size();++j)value+=(pr.edges[e+1]-pr.edges[e])*pr.weights[j]*EntropyCoupledState(pr.Point(physical,e,pr.shapes[j]),0.).eta;return value;
}
double SignedLegendre(int p,double z)
{double a=1.,b=z;if(!p)return a;for(int k=1;k<p;++k){const double c=((2*k+1)*z*b-k*a)/(k+1);a=b;b=c;}return b;}
void SignedSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=15;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=23;mo.retained_order=mo.stress_order=5;
  SignedEnrichmentProblem equal(mo,5,.25,.25),rich(mo,3,.25,.25);CurvatureProblem base(mo,.25,.25);
  const State ordinary=base.Initial();TrackWork w0,w1;const State s0=base.Closure(ordinary,0.,w0);std::array<double,4>b0{},b1{};
  const State r0=base.Residual(ordinary,s0,0.,w0,b0),r1=equal.Residual(ordinary,s0,0.,w1,b1);const double equality=PatchRelativeDifference(r0,r1);
  State physical(rich.ne*rich.n*4,0.);for(int e=0;e<rich.ne;++e)for(int i=0;i<rich.n;++i)
  {
    const double z=rich.physics.basis.nodes[i],rho=1.2+.04*z*z*z,v=.3+.03*std::pow(z,5),vy=.01+.02*std::pow(z,4),p=1.1+.01*z*z*z;
    const int j=rich.Id(e,i,0);physical[j]=rho;physical[j+1]=rho*v;physical[j+2]=rho*vy;physical[j+3]=p/(gamma_euler-1.)+.5*rho*(v*v+vy*vy);
  }
  const State projected=rich.state_transfer.Apply(rich.state_transfer.E,rich.state_transfer.Apply(rich.state_transfer.P,physical,rich.ne,4),rich.ne,4);
  const State external=rich.RestrictExternal(projected),rebuilt=rich.Physical(external),internal=rich.InternalMoments(external);
  const double reconstruction=PatchRelativeDifference(physical,rebuilt),moment_map=PatchRelativeDifference(projected,internal);
  State direction(external.size());for(std::size_t i=0;i<direction.size();++i)direction[i]=.04*std::sin(.17*i);
  direction=rich.RestrictExternal(rich.state_transfer.Apply(rich.state_transfer.E,rich.state_transfer.Apply(rich.state_transfer.P,direction,rich.ne,4),rich.ne,4));
  const State derivative=rich.InternalDirection(external,direction);const double epsilon=2.e-5;State plus=external,minus=external;
  for(std::size_t i=0;i<external.size();++i){plus[i]+=epsilon*direction[i];minus[i]-=epsilon*direction[i];}
  const State ip=rich.InternalMoments(plus),im=rich.InternalMoments(minus);double differential=0.;
  for(std::size_t i=0;i<derivative.size();++i)differential=std::max(differential,std::abs((ip[i]-im[i])/(2*epsilon)-derivative[i]));
  const double entropy_rate=rich.EntropyRate(external,direction),entropy_fd=(SignedEntropy(rich,plus)-SignedEntropy(rich,minus))/(2*epsilon),entropy_error=std::abs(entropy_rate-entropy_fd);
  TrackWork wr;const State sr=rich.Closure(external,0.,wr);std::array<double,4>br{};const State rr=rich.Residual(external,sr,0.,wr,br);const auto totals=rich.Integrals(rr,rich.edges);double conservation=0.;for(int c=0;c<4;++c)conservation=std::max(conservation,std::abs(totals[c]-br[c]));
  State tail(rich.ne*rich.retained.n*4,0.);for(int e=0;e<rich.ne;++e)for(int i=0;i<rich.retained.n;++i)tail[(e*rich.retained.n+i)*4+1]=.005*(1.+.13*e)*SignedLegendre(5,2*rich.retained.physics.basis.nodes[i]-1.);
  tail=rich.state_transfer.Apply(rich.state_transfer.E,tail,rich.ne,4);plus=external;minus=external;for(std::size_t i=0;i<plus.size();++i){plus[i]+=tail[i];minus[i]-=tail[i];}
  TrackWork wp,wm;const State sp=rich.Closure(plus,0.,wp),sm=rich.Closure(minus,0.,wm);std::array<double,4>bp{},bm{};const State rp=rich.Residual(plus,sp,0.,wp,bp),rm=rich.Residual(minus,sm,0.,wm,bm);
  double density_feedback=0.,energy_feedback=0.;for(int e=0;e<rich.ne;++e)for(int i=0;i<rich.n;++i){density_feedback=std::max(density_feedback,std::abs(rp[rich.Id(e,i,0)]-rm[rich.Id(e,i,0)]));energy_feedback=std::max(energy_feedback,std::abs(rp[rich.Id(e,i,3)]-rm[rich.Id(e,i,3)]));}
  const double closure_feedback=PatchRelativeDifference(sp,sm);State bad=external;for(std::size_t i=0;i<bad.size();++i)bad[i]+=4000.*tail[i];const auto before=rich.Integrals(bad,rich.edges);TrackWork wl;rich.Limit(bad,wl);const auto after=rich.Integrals(bad,rich.edges);double limiter_mean=0.;for(int c=0;c<4;++c)limiter_mean=std::max(limiter_mean,std::abs(before[c]-after[c]));
  double allocation_reconstruction=0.,allocation_initial=0.,all_order_parity=0.;
  const State common_initial=rich.Physical(rich.Initial());
  for(bool density:{false,true})for(bool energy:{false,true})
  {
    SignedEnrichmentProblem allocation(mo,3,.25,.25,"original",density,energy);
    allocation_initial=std::max(allocation_initial,PatchRelativeDifference(common_initial,allocation.Physical(allocation.Initial())));
    State state(physical.size(),0.);
    for(int e=0;e<rich.ne;++e)for(int i=0;i<rich.n;++i)
    {
      const double z=rich.physics.basis.nodes[i],rho=1.2+.04*std::pow(z,density?5:3),v=.3+.03*std::pow(z,5),vy=.01+.02*std::pow(z,4),p=1.1+.01*std::pow(z,energy?5:3);
      const int j=rich.Id(e,i,0);state[j]=rho;state[j+1]=rho*v;state[j+2]=rho*vy;state[j+3]=p/(gamma_euler-1.)+.5*rho*(v*v+vy*vy);
    }
    const State projected_state=allocation.state_transfer.Apply(allocation.state_transfer.E,allocation.state_transfer.Apply(allocation.state_transfer.P,state,allocation.ne,4),allocation.ne,4);
    const State external_state=allocation.RestrictExternal(projected_state);
    allocation_reconstruction=std::max(allocation_reconstruction,PatchRelativeDifference(state,allocation.Physical(external_state)));
    if(density&&energy)
    {
      TrackWork wa,wb;const State sigma=base.Closure(external_state,0.,wb);std::array<double,4>ba{},bb{};
      const State ra=allocation.Residual(external_state,sigma,0.,wa,ba),rb=base.Residual(external_state,sigma,0.,wb,bb);
      all_order_parity=PatchRelativeDifference(ra,rb);
    }
  }
  std::cout<<std::setprecision(17)<<"{\"equal_order_full_rhs\":"<<equality<<",\"physical_reconstruction\":"<<reconstruction<<",\"internal_moment_map\":"<<moment_map<<",\"internal_chain_differential\":"<<differential<<",\"physical_entropy_differential\":"<<entropy_error<<",\"conservation\":"<<conservation<<",\"signed_density_feedback\":"<<density_feedback<<",\"signed_energy_feedback\":"<<energy_feedback<<",\"signed_closure_feedback\":"<<closure_feedback<<",\"active_limited_cells\":"<<wl.limited<<",\"limiter_conservative_mean\":"<<limiter_mean<<",\"four_allocation_reconstruction\":"<<allocation_reconstruction<<",\"four_allocation_common_initial\":"<<allocation_initial<<",\"all_enriched_ordinary_rhs_parity\":"<<all_order_parity<<"}\n";
  if(equality!=0.||reconstruction>1.e-10||moment_map>1.e-10||differential>1.e-7||entropy_error>1.e-7||conservation>1.e-8||density_feedback<1.e-8||energy_feedback<1.e-8||closure_feedback<1.e-8||wl.limited==0||limiter_mean>1.e-10||allocation_reconstruction>1.e-10||allocation_initial>1.e-10||all_order_parity!=0.)throw std::runtime_error("signed enrichment reconstruction/chain/conservation/feedback/limiter selftest failed");
}
}
#ifndef DG_IGR_SIGNED_ENRICHMENT_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    int extra_order=2;double slope=.25,curvature=.25,time=0.;bool frozen=false,enrich_density=false,enrich_energy=false;std::string input,edge_input,volume="original";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--residual-only"){frozen=true;continue;}if(flag=="--enrich-density"){enrich_density=true;continue;}if(flag=="--enrich-energy"){enrich_energy=true;continue;}
      if(flag=="--extra-order"||flag=="--entropy-volume"||flag=="--slope-strength"||flag=="--curvature-strength"||flag=="--probe-time"||flag=="--external-moments-in"||flag=="--edges-in")
      {if(++i==argc)throw std::runtime_error("missing signed option");const std::string value=argv[i];if(flag=="--extra-order")extra_order=std::stoi(value);else if(flag=="--entropy-volume")volume=value;else if(flag=="--slope-strength")slope=std::stod(value);else if(flag=="--curvature-strength")curvature=std::stod(value);else if(flag=="--probe-time")time=std::stod(value);else if(flag=="--external-moments-in")input=value;else edge_input=value;}
      else common.push_back(argv[i]);
    }
    MemoryOptions mo;mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;if(t.self_test){SignedSelfTest();return 0;}
    const int degree=t.base.order,rich_degree=degree+extra_order;if(extra_order<0)throw std::runtime_error("negative enrichment order");
    mo.retained_order=mo.stress_order=rich_degree;t.base.order=3*rich_degree;mo.input=input;mo.start_time=time;
    if(time<0.||time>t.base.final_time||(!input.empty()&&edge_input.empty())||(frozen&&input.empty()))throw std::runtime_error("saved signed state requires compatible time, moments and edges");
    if(mo.recovery.boundary.scenario!="application-tanh"||mo.recovery.boundary.left!="nonreflecting"||mo.recovery.boundary.right!="fixed")throw std::runtime_error("signed front qualified only for original application-tanh boundaries");
    const auto start=Clock::now();SignedEnrichmentProblem pr(mo,degree,slope,curvature,volume,enrich_density,enrich_energy);if(!edge_input.empty())SignedGeometry(pr,SignedRead(edge_input,pr.ne+1));const double setup=Wall(start);
    TrackResult result;State rhs,internal_rhs;std::array<double,4>budget{};double entropy_rate=0.;
    if(frozen){result.time=time;result.u=pr.Initial();result.sigma=pr.Closure(result.u,time,result.work);rhs=pr.Residual(result.u,result.sigma,time,result.work,budget);internal_rhs=pr.InternalDirection(result.u,rhs);entropy_rate=pr.EntropyRate(result.u,rhs);}
    else result=MemoryRun(pr);
    result.work.setup=setup;State external,moments;if(!result.u.empty()){external=pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);moments=pr.state_transfer.Apply(pr.state_transfer.P,pr.InternalMoments(result.u),pr.ne,4);result.u=pr.Physical(result.u);}
    const int density_degree=enrich_density?rich_degree:degree,energy_degree=enrich_energy?rich_degree:degree;
    const std::string allocation=enrich_density?(enrich_energy?"all":"density-velocity"):(enrich_energy?"thermal-velocity":"velocity");
    std::ostringstream out;out<<std::setprecision(17)<<",\"source_recovery\":\"signed-velocity-enrichment\",\"base_order\":"<<degree<<",\"velocity_order\":"<<rich_degree<<",\"density_order\":"<<density_degree<<",\"thermal_pressure_order\":"<<energy_degree<<",\"energy_moment_order\":"<<energy_degree
      <<",\"retained_order\":"<<rich_degree<<",\"extra_order\":"<<extra_order<<",\"stress_order\":"<<rich_degree<<",\"primitive_source_faces\":\"central\",\"moment_operator\":\"direct\",\"sigma_serialized_order\":"<<pr.p
      <<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns<<",\"actual_euler_dofs_total\":"<<pr.ne*((density_degree+1)+(energy_degree+1)+2*(rich_degree+1))<<",\"actual_euler_dofs_by_field\":["<<pr.ne*(density_degree+1)<<','<<pr.ne*(rich_degree+1)<<','<<pr.ne*(rich_degree+1)<<','<<pr.ne*(energy_degree+1)<<"],\"external_storage_coefficients\":"<<4*pr.retained.ns
      <<",\"enrich_density\":"<<(enrich_density?"true":"false")<<",\"enrich_energy\":"<<(enrich_energy?"true":"false")<<",\"allocation\":\""<<allocation<<"\""<<",\"initial_enrichment\":\""<<(input.empty()?"base-physical-velocity-lift":"external-moment-input")<<"\",\"injected_state_time\":"<<time<<",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"entropy_volume\":\""<<volume<<"\",\"entropy_proof_claimed\":false,\"enrichment_mode\":\""<<(frozen?"frozen":"advance")
      <<"\",\"enrichment_seconds\":"<<pr.enrichment_seconds<<",\"enrichment_cells\":"<<pr.enrichment_cells<<",\"restriction_seconds\":"<<pr.restriction_seconds<<",\"restriction_calls\":"<<pr.restriction_calls<<",\"maximum_restriction_mean\":"<<pr.maximum_restriction_mean
      <<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"local_velocity_factors\":"<<pr.local_velocity_factors<<",\"entropy_seconds\":"<<pr.entropy_seconds<<",\"entropy_pairs\":"<<pr.entropy_pairs<<",\"maximum_entropy_pair_work_defect\":"<<pr.maximum_entropy_pair_work_defect
      <<",\"direct_assembly_seconds\":"<<pr.direct_assembly_seconds<<",\"moment_transfer_seconds\":"<<pr.transfer_seconds<<",\"galerkin_seconds\":"<<pr.galerkin_seconds<<",\"slope_penalty_seconds\":"<<pr.slope_seconds<<",\"curvature_seconds\":"<<pr.curvature_seconds<<",\"external_moment_coefficients\":";TrackArray(out,external);out<<",\"moment_coefficients\":";TrackArray(out,moments);
    if(frozen){out<<",\"external_rhs\":";TrackArray(out,pr.state_transfer.Apply(pr.state_transfer.P,rhs,pr.ne,4));out<<",\"base_rhs\":";TrackArray(out,pr.state_transfer.Apply(pr.state_transfer.P,internal_rhs,pr.ne,4));out<<",\"unrestricted_rhs\":";TrackArray(out,pr.state_transfer.Apply(pr.state_transfer.P,pr.last_unrestricted_rhs,pr.ne,4));out<<",\"boundary_budget\":";TrackArray(out,State(budget.begin(),budget.end()));out<<",\"physical_gas_mathematical_entropy_rate\":"<<entropy_rate;}
    BoundaryWrite(pr,result,out.str());return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif

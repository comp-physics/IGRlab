// Test-only exact-manifold physical gas-entropy-volume work correction.
#define DG_IGR_ENTROPY_COUPLED_LIBRARY_ONLY
#include "igr_dg_entropy_coupled_probe.cpp"
#undef DG_IGR_ENTROPY_COUPLED_LIBRARY_ONLY
namespace {
#include "igr_dg_entropy_coupled_functional.hpp"
void EntropyFunctionalSelfTest()
{
  double none_parity=0.,entropy_identity=0.,mean=0.,energy_fixed=0.,active_residual=0.,nonzero=0.;std::uint64_t active=0;
  for(int degree:{3,5})
  {
    MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=3*degree;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=25;mo.retained_order=mo.stress_order=degree;
    EntropyFunctionalProblem none(mo,.25,.25,"none"),metric(mo,.25,.25,"metric"),energy(mo,.25,.25,"energy");EntropyCoupledProblem original(mo,.25,.25,"original");
    State physical(none.ne*none.n*4,0.);for(int e=0;e<none.ne;++e)for(int i=0;i<none.n;++i)
    {
      const double z=none.physics.basis.nodes[i],rho=1.+.8*z*z*z+.02*e,v=.4+1.3*z*z*z+.03*e,vy=-.1+.2*z*z,p=.5+.6*z*z*z+.01*e;const int j=none.Id(e,i,0);
      physical[j]=rho;physical[j+1]=rho*v;physical[j+2]=rho*vy;physical[j+3]=p/(gamma_euler-1.)+.5*rho*(v*v+vy*vy);
    }
    const State moments=none.state_transfer.Apply(none.state_transfer.E,none.state_transfer.Apply(none.state_transfer.P,physical,none.ne,4),none.ne,4);TrackWork w0,wo,wm,we;const State sigma=none.Closure(moments,0.,w0);std::array<double,4>b0{},bo{},bm{},be{};
    const State r0=none.Residual(moments,sigma,0.,w0,b0),ro=original.Residual(moments,sigma,0.,wo,bo),rm=metric.Residual(moments,sigma,0.,wm,bm),re=energy.Residual(moments,sigma,0.,we,be);none_parity=std::max(none_parity,PatchRelativeDifference(r0,ro));
    for(EntropyFunctionalProblem* pr:{&metric,&energy})
    {
      double predicted=0.;for(int e=0;e<pr->ne;++e)predicted+=pr->last_functional_defects[e]-pr->last_functional_remaining[e];
      const double actual=pr->EntropyRate(moments,pr->last_functional_correction);entropy_identity=std::max(entropy_identity,std::abs(actual-predicted));active+=pr->functional_active_cells;
      mean=std::max({mean,pr->maximum_functional_mean,pr==&metric?wm.max_rhs_defect:we.max_rhs_defect});active_residual=std::max(active_residual,pr->maximum_functional_controlled_residual);
      if(pr->functional_unattainable_cells)throw std::runtime_error("unexpected unattainable entropy selftest budget");
    }
    for(std::size_t i=0;i<rm.size();++i){nonzero=std::max(nonzero,std::abs(rm[i]-r0[i]));if(i%4!=3)energy_fixed=std::max(energy_fixed,std::abs(re[i]-r0[i]));}
  }
  if(none_parity!=0.||entropy_identity>1.e-8||mean>1.e-8||energy_fixed!=0.||active_residual>1.e-8||active==0||nonzero<1.e-6)throw std::runtime_error("functional entropy exact-gradient/physical-work/mean test failed");
  std::cout<<std::setprecision(17)<<"{\"none_original_parity\":"<<none_parity<<",\"actual_manifold_entropy_correction_identity\":"<<entropy_identity<<",\"mean_and_conservation\":"<<mean<<",\"energy_only_other_components_fixed\":"<<energy_fixed<<",\"active_budget_residual\":"<<active_residual<<",\"active_cells\":"<<active<<",\"nontrivial_correction\":"<<nonzero<<"}\n";
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    double slope=.25,curvature=.25,time=0.;bool frozen=false;std::string input,edge_input,mode="metric";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--residual-only"){frozen=true;continue;}
      if(flag=="--functional-entropy"||flag=="--slope-strength"||flag=="--curvature-strength"||flag=="--probe-time"||flag=="--moments-in"||flag=="--edges-in")
      {if(++i==argc)throw std::runtime_error("missing entropy-functional option");const std::string value=argv[i];if(flag=="--functional-entropy")mode=value;else if(flag=="--slope-strength")slope=std::stod(value);else if(flag=="--curvature-strength")curvature=std::stod(value);else if(flag=="--probe-time")time=std::stod(value);else if(flag=="--moments-in")input=value;else edge_input=value;}
      else common.push_back(argv[i]);
    }
    MemoryOptions mo;mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;if(t.self_test){EntropyFunctionalSelfTest();return 0;}
    const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;mo.input=input;mo.start_time=time;
    if(time<0.||time>t.base.final_time||(!input.empty()&&edge_input.empty()))throw std::runtime_error("saved functional entropy state requires compatible time and edges");
    if(mo.recovery.boundary.scenario!="application-tanh"||mo.recovery.boundary.left!="nonreflecting"||mo.recovery.boundary.right!="fixed")throw std::runtime_error("entropy-functional front qualified only for original application-tanh boundaries");
    const auto start=Clock::now();EntropyFunctionalProblem pr(mo,slope,curvature,mode);if(!edge_input.empty())EntropyCoupledGeometry(pr,EntropyCoupledRead(edge_input,pr.ne+1));const double setup=Wall(start);
    TrackResult result;State rhs,correction;std::array<double,4> budget{};double entropy_rate=0.,correction_rate=0.;
    if(frozen){result.time=time;result.u=pr.Initial();result.sigma=pr.Closure(result.u,time,result.work);rhs=pr.Residual(result.u,result.sigma,time,result.work,budget);correction=pr.last_functional_correction;entropy_rate=pr.EntropyRate(result.u,rhs);correction_rate=pr.EntropyRate(result.u,correction);}
    else result=MemoryRun(pr);
    result.work.setup=setup;const State moments=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);if(!result.u.empty())result.u=pr.Reconstruct(result.u);
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"joint-density-weighted-primitive\",\"volume_projection\":\"actual-manifold-physical-entropy-work-control\",\"entropy_volume\":\"original\",\"functional_entropy\":\""<<mode<<"\",\"entropy_proof_claimed\":false,\"retained_order\":"<<degree
      <<",\"primitive_source_faces\":\"central\",\"moment_operator\":\"direct\",\"stress_order\":"<<degree<<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns<<",\"sigma_serialized_order\":"<<pr.p
      <<",\"injected_state_time\":"<<time<<",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"entropy_mode\":\""<<(frozen?"frozen":"advance")<<"\",\"functional_seconds\":"<<pr.functional_seconds<<",\"functional_cells\":"<<pr.functional_cells<<",\"functional_active_cells\":"<<pr.functional_active_cells<<",\"functional_unattainable_cells\":"<<pr.functional_unattainable_cells
      <<",\"functional_budget_tolerance\":"<<pr.budget_tolerance<<",\"maximum_functional_defect\":"<<pr.maximum_functional_defect<<",\"maximum_functional_controlled_residual\":"<<pr.maximum_functional_controlled_residual<<",\"maximum_functional_skipped_defect\":"<<pr.maximum_functional_skipped_defect<<",\"maximum_functional_unattainable_defect\":"<<pr.maximum_functional_unattainable_defect<<",\"maximum_functional_mean\":"<<pr.maximum_functional_mean<<",\"maximum_functional_rhs_ratio\":"<<pr.maximum_functional_rhs_ratio
      <<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"local_velocity_factors\":"<<pr.local_velocity_factors<<",\"direct_assembly_seconds\":"<<pr.direct_assembly_seconds<<",\"moment_transfer_seconds\":"<<pr.transfer_seconds<<",\"galerkin_seconds\":"<<pr.galerkin_seconds
      <<",\"slope_penalty_seconds\":"<<pr.slope_seconds<<",\"curvature_seconds\":"<<pr.curvature_seconds<<",\"maximum_slope_identity_error\":"<<pr.maximum_identity_error<<",\"maximum_curvature_identity_error\":"<<pr.curvature_identity_error<<",\"moment_coefficients\":";TrackArray(extra,moments);
    if(frozen){extra<<",\"base_rhs\":";TrackArray(extra,pr.state_transfer.Apply(pr.state_transfer.P,rhs,pr.ne,4));extra<<",\"volume_correction\":";TrackArray(extra,pr.state_transfer.Apply(pr.state_transfer.P,correction,pr.ne,4));extra<<",\"boundary_budget\":";TrackArray(extra,State(budget.begin(),budget.end()));extra<<",\"manifold_gas_mathematical_entropy_rate\":"<<entropy_rate<<",\"correction_gas_mathematical_entropy_rate\":"<<correction_rate<<",\"functional_defects\":";TrackArray(extra,pr.last_functional_defects);extra<<",\"functional_remaining\":";TrackArray(extra,pr.last_functional_remaining);extra<<",\"functional_denominators\":";TrackArray(extra,pr.last_functional_denominators);extra<<",\"functional_tolerances\":";TrackArray(extra,pr.last_functional_tolerances);}
    BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

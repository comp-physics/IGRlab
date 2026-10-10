// Test-only derivative-compatible volume evolution on the existing primitive
// weak DG + full-energy Euler + instantaneous natural L2/SIPG application.
#define DG_IGR_HP_PATCH_LIBRARY_ONLY
#include "igr_dg_hp_patch_probe.cpp"
#undef DG_IGR_HP_PATCH_LIBRARY_ONLY
#include <memory>
namespace {
#include "igr_dg_commuting_evolution_probe.hpp"
#include "igr_dg_gradient_evolution_probe.hpp" // Independent weighted-coordinate self-test only.
State CommutingRead(const std::string& path,std::size_t count)
{
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open commuting input");State a(count);for(double& x:a)if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid commuting input");in>>std::ws;if(!in.eof())throw std::runtime_error("extra commuting input");return a;
}
void CommutingGeometry(CommutingProblem& pr,const State& edges)
{
  if(edges.size()!=std::size_t(pr.ne+1)||std::abs(edges.front()-pr.o.left)>1.e-13||std::abs(edges.back()-pr.o.right)>1.e-13)throw std::runtime_error("commuting physical geometry mismatch");
  PatchGeometry g;g.edges=edges;g.left=edges.front();g.right=edges.back();g.inner_elements=pr.ne;g.hmin=INFINITY;
  for(int e=0;e<pr.ne;++e){double h=edges[e+1]-edges[e];if(!(h>0.))throw std::runtime_error("invalid commuting cell width");g.hmin=std::min(g.hmin,h);g.hmax=std::max(g.hmax,h);if(e){double previous=edges[e]-edges[e-1];g.maximum_adjacent_ratio=std::max(g.maximum_adjacent_ratio,std::max(h/previous,previous/h));}}
  SetPatchGeometry(pr,g,g.hmax-g.hmin<1.e-13?"uniform":"saved-nonuniform");
}
void CommutingSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=15;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=21;mo.retained_order=mo.stress_order=5;
  double reproduction=0.,orthogonality=0.,means=0.,two_mode_parity=0.;
  for(double eta:{0.,1.,16.,256.,-1.})
  {
    CommutingProblem pr(mo,.25,.25,std::max(0.,eta),eta<0.);const int m=pr.retained.n;
    for(int power:{pr.retained.p+1,pr.retained.p+3})
    {
      std::vector<std::array<double,4>> F(pr.weights.size());std::array<std::array<double,4>,2> face{},derivative{};State target(4*m,0.),load(4*m,0.),derivative_load(4*m,0.);
      for(int c=0;c<4;++c){face[1][c]=c+1.;derivative[1][c]=(c+1.)*power;}
      for(std::size_t q=0;q<pr.weights.size();++q)
      {
        double z=0.;for(int i=0;i<pr.n;++i)z+=pr.shapes[q][i]*pr.physics.basis.nodes[i];
        for(int c=0;c<4;++c)
        {
          F[q][c]=(c+1.)*std::pow(z,power);const double s=-(c+1.)*power*std::pow(z,power-1),ds=-(c+1.)*power*(power-1)*std::pow(z,power-2);
          for(int i=0;i<m;++i){load[c*m+i]+=pr.weights[q]*pr.low_shapes[q][i]*s;derivative_load[c*m+i]+=pr.weights[q]*pr.low_derivatives[q][i]*ds;}
        }
      }
      for(int c=0;c<4;++c)for(int i=0;i<m;++i)for(int j=0;j<m;++j)target[c*m+i]+=pr.retained.mass_inverse(i,j)*load[c*m+j];
      State delta=pr.CorrectCell(F,face,derivative),ibp=pr.CorrectCellIntegrationByParts(F,face,derivative);for(std::size_t i=0;i<delta.size();++i)two_mode_parity=std::max(two_mode_parity,std::abs(delta[i]-ibp[i]));
      for(int c=0;c<4;++c)
      {
        double mean=0.;for(int i=0;i<m;++i)mean+=pr.means[i]*delta[c*m+i];means=std::max(means,std::abs(mean));
        for(int i=0;i<m;++i)
        {
          if(power==pr.retained.p+1)reproduction=std::max(reproduction,std::abs(delta[c*m+i]));
          double actual=0.,wanted=0.;for(int j=0;j<m;++j)actual+=(pr.seminorm?pr.stiffness.Get(i,j):pr.sobolev.Get(i,j))*(target[c*m+j]+delta[c*m+j]);
          wanted=pr.seminorm?derivative_load[c*m+i]:load[c*m+i]+pr.ratio*derivative_load[c*m+i];orthogonality=std::max(orthogonality,std::abs(actual-wanted)/std::max(1.,std::abs(wanted)));
        }
      }
    }
  }
  CommutingProblem zero(mo,.25,.25,0.,false),active(mo,.25,.25,16.,false);CurvatureProblem base(mo,.25,.25);State initial=base.Initial();TrackWork wb,w0,wa;base.Limit(initial,wb);State sigma=base.Closure(initial,0.,wb);std::array<double,4> bb{},b0{},ba{};
  const State rb=base.Residual(initial,sigma,0.,wb,bb),r0=zero.Residual(initial,sigma,0.,w0,b0),ra=active.Residual(initial,sigma,0.,wa,ba);double parity=0.;for(std::size_t i=0;i<rb.size();++i)parity=std::max(parity,std::abs(rb[i]-r0[i]));
  CommutingProblem momentum(mo,.25,.25,16.,false,true);TrackWork wm;std::array<double,4> bm{};const State rm=momentum.Residual(initial,sigma,0.,wm,bm);double fixed_components=0.,selected_parity=0.;
  for(std::size_t i=0;i<rm.size();++i){if(i%4==0||i%4==3)fixed_components=std::max(fixed_components,std::abs(rm[i]-rb[i]));else selected_parity=std::max(selected_parity,std::abs(rm[i]-ra[i]));}
  if(fixed_components!=0.||selected_parity>1.e-12||wm.max_rhs_defect>1.e-8)throw std::runtime_error("commuting component selection/mean failed");
  CommutingProblem product(mo,.25,.25,0.,false,false,true);GradientEvolutionProblem weighted(mo,.25,.25,"continuum-weighted");State physical(product.ne*product.n*4,0.);
  for(int e=0;e<product.ne;++e)for(int i=0;i<product.n;++i)
  {
    const double z=product.physics.basis.nodes[i],rho=1.2+.04*std::pow(z,5),v=.3+.05*std::pow(z,5),vy=.01+.02*std::pow(z,4),pressure=1.1+.01*z;const int j=product.Id(e,i,0);
    physical[j]=rho;physical[j+1]=rho*v;physical[j+2]=rho*vy;physical[j+3]=pressure/(gamma_euler-1.)+.5*rho*(v*v+vy*vy);
  }
  State test_moments=product.state_transfer.Apply(product.state_transfer.E,product.state_transfer.Apply(product.state_transfer.P,physical,product.ne,4),product.ne,4);TrackWork wp,ww;std::array<double,4> bp{},bw{};
  State test_sigma=product.Closure(test_moments,0.,wp);const State rp=product.Residual(test_moments,test_sigma,0.,wp,bp);weighted.Residual(weighted.Pack(test_moments),test_sigma,0.,ww,bw);
  double product_parity=0.;for(std::size_t i=0;i<rp.size();++i)product_parity=std::max(product_parity,std::abs(rp[i]-weighted.last_standard_rhs[i]));
  std::vector<std::array<double,4>> constant_density(product.retained.n);for(int i=0;i<product.retained.n;++i){const double z=product.retained.physics.basis.nodes[i];constant_density[i]={1.2,.3+.05*std::pow(z,5),.01+.02*std::pow(z,4),1.1};}
  const State no_commutator=product.DensityProductCell(constant_density);double product_zero=0.;for(double value:no_commutator)product_zero=std::max(product_zero,std::abs(value));
  if(product_parity>1.e-8||product_zero>1.e-10||product.maximum_commuting_mean>1.e-10||wp.max_rhs_defect>1.e-8)throw std::runtime_error("density product weighted-coordinate/constant-density/mean identity failed");
  if(two_mode_parity>2.e-8||reproduction>2.e-8||orthogonality>2.e-8||means>1.e-9||parity!=0.||wa.max_rhs_defect>1.e-8)throw std::runtime_error("commuting polynomial/Sobolev/mean/eta0 check failed");
  std::cout<<std::setprecision(17)<<"{\"polynomial_reproduction\":"<<reproduction<<",\"independent_sobolev_orthogonality\":"<<orthogonality<<",\"two_mode_vs_integration_by_parts\":"<<two_mode_parity<<",\"correction_mean\":"<<means<<",\"eta0_parity\":"<<parity<<",\"active_rhs_conservation\":"<<wa.max_rhs_defect<<",\"momentum_only_fixed_components\":"<<fixed_components<<",\"selected_momentum_parity\":"<<selected_parity<<",\"density_product_weighted_rhs_parity\":"<<product_parity<<",\"density_product_constant_rho_zero\":"<<product_zero<<",\"density_product_mean\":"<<product.maximum_commuting_mean<<"}\n";
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    double eta=1.,slope=.25,curvature=.25,time=0.;bool seminorm=false,frozen=false,density_product=false,strength_set=false;std::string input,edge_input,components="all";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--sobolev-seminorm"){seminorm=true;continue;}if(flag=="--residual-only"){frozen=true;continue;}if(flag=="--density-product-correction"){density_product=true;continue;}
      if(flag=="--sobolev-components"||flag=="--sobolev-strength"||flag=="--slope-strength"||flag=="--curvature-strength"||flag=="--probe-time"||flag=="--moments-in"||flag=="--edges-in")
      {if(++i==argc)throw std::runtime_error("missing commuting option");const std::string value=argv[i];if(flag=="--sobolev-components")components=value;else if(flag=="--sobolev-strength"){eta=std::stod(value);strength_set=true;}else if(flag=="--slope-strength")slope=std::stod(value);else if(flag=="--curvature-strength")curvature=std::stod(value);else if(flag=="--probe-time")time=std::stod(value);else if(flag=="--moments-in")input=value;else edge_input=value;}
      else common.push_back(argv[i]);
    }
    MemoryOptions mo;mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;
    if(t.self_test){CommutingSelfTest();return 0;}const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;mo.input=input;mo.start_time=time;
    if(time<0.||time>t.base.final_time||(!input.empty()&&edge_input.empty())||(frozen&&input.empty()))throw std::runtime_error("saved state/time/edges required for frozen or restart mode");
    if(mo.recovery.boundary.scenario!="application-tanh"||mo.recovery.boundary.left!="nonreflecting"||mo.recovery.boundary.right!="fixed")throw std::runtime_error("commuting front qualified only for original application-tanh boundaries");
    if(components!="all"&&components!="momentum")throw std::runtime_error("unknown Sobolev component selection");
    if(density_product){if(seminorm||(strength_set&&eta!=0.))throw std::runtime_error("density product and Sobolev correction cannot be combined");eta=0.;}
    const auto start=Clock::now();CommutingProblem pr(mo,slope,curvature,eta,seminorm,components=="momentum",density_product);if(!edge_input.empty())CommutingGeometry(pr,CommutingRead(edge_input,pr.ne+1));const double setup=Wall(start);
    TrackResult result;State rhs,correction;std::array<double,4> budget{};
    if(frozen)
    {
      result.time=time;result.u=pr.Initial();result.sigma=pr.Closure(result.u,time,result.work);rhs=pr.Residual(result.u,result.sigma,time,result.work,budget);correction=pr.state_transfer.Apply(pr.state_transfer.P,pr.last_commuting_correction,pr.ne,4);
    }
    else result=MemoryRun(pr);
    result.work.setup=setup;const State moments=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);if(!result.u.empty())result.u=pr.Reconstruct(result.u);
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"joint-density-weighted-primitive\",\"volume_projection\":\""<<(density_product?"density-product-conservative-moments":"sobolev-volume-conservative-moments")<<"\",\"retained_order\":"<<degree
      <<",\"primitive_source_faces\":\"central\",\"moment_operator\":\"direct\",\"stress_order\":"<<degree<<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns
      <<",\"sigma_serialized_order\":"<<pr.p<<",\"injected_state_time\":"<<time<<",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"sobolev_strength\":"<<eta<<",\"sobolev_seminorm\":"<<(seminorm?"true":"false")
      <<",\"sobolev_components\":\""<<components<<"\""<<",\"density_product_correction\":"<<(density_product?"true":"false")<<",\"commuting_mode\":\""<<(frozen?"frozen":"advance")<<"\",\"commuting_seconds\":"<<pr.commuting_seconds<<",\"commuting_cells\":"<<pr.commuting_cells<<",\"maximum_commuting_mean\":"<<pr.maximum_commuting_mean
      <<",\"maximum_commuting_rhs_ratio\":"<<pr.maximum_commuting_ratio<<",\"last_commuting_kinetic_power\":";TrackArray(extra,State(pr.last_commuting_kinetic_power.begin(),pr.last_commuting_kinetic_power.end()));
    extra<<",\"moment_coefficients\":";TrackArray(extra,moments);
    if(frozen){extra<<",\"base_rhs\":";TrackArray(extra,pr.state_transfer.Apply(pr.state_transfer.P,rhs,pr.ne,4));extra<<",\"volume_correction\":";TrackArray(extra,correction);extra<<",\"boundary_budget\":";TrackArray(extra,State(budget.begin(),budget.end()));}
    BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

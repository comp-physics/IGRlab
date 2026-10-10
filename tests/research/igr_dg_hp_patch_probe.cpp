// Fixed nonuniform high-p mesh control for the existing primitive direct DG.
// Patch bounds use only the declared initial center/width/drift and horizon.
// No moving mesh, oracle state, remap, changed continuum model, or solver copy.
#define DG_IGR_CURVATURE_PENALTY_LIBRARY_ONLY
#include "igr_dg_curvature_penalty_probe.cpp"
#undef DG_IGR_CURVATURE_PENALTY_LIBRARY_ONLY

namespace
{
struct HPPatchOptions
{
  std::string policy="source-patch";
  int inner_elements=20;
  double width_multiple=4.,phase=0.;
};
struct PatchGeometry
{
  State edges;double left=0.,right=0.,hmin=0.,hmax=0.,maximum_adjacent_ratio=1.;
  int left_elements=0,inner_elements=0,right_elements=0;
};
PatchGeometry MakePatch(const TrackOptions& t,const HPPatchOptions& o)
{
  PatchGeometry g;const int ne=t.base.elements;g.edges.resize(ne+1);
  if(o.policy=="uniform")
  {for(int e=0;e<=ne;++e)g.edges[e]=t.left+(t.right-t.left)*e/ne;g.left=t.left;g.right=t.right;g.inner_elements=ne;}
  else
  {
    if(o.policy!="source-patch"||o.inner_elements<2||o.inner_elements>ne-2||!(o.width_multiple>0.)||
       !std::isfinite(o.width_multiple)||!std::isfinite(o.phase)||std::abs(o.phase)>.5||!(t.width>0.))
      throw std::runtime_error("invalid static hp patch parameters");
    g.left_elements=(ne-o.inner_elements)/2;g.right_elements=ne-o.inner_elements-g.left_elements;g.inner_elements=o.inner_elements;
    const double drift=t.speed*t.base.final_time;
    g.left=t.center-o.width_multiple*t.width+std::min(0.,drift);
    g.right=t.center+o.width_multiple*t.width+std::max(0.,drift);
    const double inner_h=(g.right-g.left)/g.inner_elements;
    // A fixed lattice phase control. It moves neither the state nor the mesh
    // during evolution and uses no observed feature position.
    g.left+=o.phase*inner_h;g.right+=o.phase*inner_h;
    if(!(t.left<g.left&&g.left<t.center&&t.center<g.right&&g.right<t.right))
      throw std::runtime_error("initial-data patch must lie inside the fixed domain and contain the initial center");
    int k=0;
    auto segment=[&](double left,double right,int count)
    {for(int i=0;i<count;++i)g.edges[k++]=left+(right-left)*i/count;};
    segment(t.left,g.left,g.left_elements);segment(g.left,g.right,g.inner_elements);segment(g.right,t.right,g.right_elements);g.edges[k]=t.right;
  }
  g.hmin=INFINITY;g.hmax=0.;for(int e=0;e<ne;++e)
  {
    const double h=g.edges[e+1]-g.edges[e];if(!(h>0.)||!std::isfinite(h))throw std::runtime_error("nonpositive static patch cell");
    g.hmin=std::min(g.hmin,h);g.hmax=std::max(g.hmax,h);
    if(e){const double prior=g.edges[e]-g.edges[e-1];g.maximum_adjacent_ratio=std::max(g.maximum_adjacent_ratio,std::max(h/prior,prior/h));}
  }
  return g;
}
void SetPatchGeometry(SlopePenaltyProblem& pr,const PatchGeometry& g,const std::string& mode)
{
  // Reference-element mass, density-weighted reconstruction, and polynomial
  // transfers are unchanged under an affine cell map. Every differential
  // operator must nevertheless use the same physical edges in all spaces.
  for(TrackingProblem* space:{static_cast<TrackingProblem*>(&pr),static_cast<TrackingProblem*>(&pr.retained),static_cast<TrackingProblem*>(&pr.stress)})
  {
    if(space->ne!=pr.ne||g.edges.size()!=std::size_t(pr.ne+1))throw std::runtime_error("hp patch space geometry mismatch");
    space->edges=g.edges;space->base_uniform_edges=g.edges;space->initial_graded_hmin=g.hmin;space->o.mode=mode;
  }
  pr.cached_moments.clear();pr.cached_physical.clear();
}
struct HPPatchProblem:CurvatureProblem
{
  HPPatchOptions patch;PatchGeometry geometry;
  HPPatchProblem(const MemoryOptions& mo,const HPPatchOptions& po,double strength,bool direct=true,double curvature=0.):
      CurvatureProblem(mo,strength,curvature),patch(po),geometry(MakePatch(mo.recovery.boundary.compatibility.track,po))
  {direct_operator=direct;SetPatchGeometry(*this,geometry,po.policy=="uniform"?"uniform":"static-source-patch");}
};
double PatchRelativeDifference(const State& a,const State& b)
{if(a.size()!=b.size())throw std::runtime_error("patch parity size mismatch");double difference=0.,scale=1.;for(std::size_t i=0;i<a.size();++i){difference=std::max(difference,std::abs(a[i]-b[i]));scale=std::max({scale,std::abs(a[i]),std::abs(b[i])});}return difference/scale;}
std::string HPPatchSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& b=mo.recovery.boundary;auto& t=b.compatibility.track;
  b.left=b.right="outflow";b.stress="trace";t.initial="entropy";t.mode="uniform";t.left=-1.;t.right=1.;t.center=0.;t.width=.1;t.speed=0.;t.radius=1;
  t.base.elements=8;t.base.order=9;t.base.alpha=.04;t.base.dt=1.e-5;t.base.final_time=3.e-5;t.quad_extra=19;t.closure_space="l2";t.closure_boundary="natural";
  mo.retained_order=mo.stress_order=3;HPPatchOptions po;po.inner_elements=4;po.phase=.35;
  HPPatchProblem direct(mo,po,.5,true),embedded(mo,po,.5,false);const int m=direct.retained.n;
  if(direct.edges!=direct.retained.edges||direct.edges!=direct.stress.edges||direct.geometry.maximum_adjacent_ratio<1.2)
    throw std::runtime_error("nonuniform three-space geometry synchronization failed");
  const State initial=direct.Initial(),initial_low=direct.state_transfer.Apply(direct.state_transfer.P,initial,direct.ne,4);double initialization=0.;
  for(int e=0;e<direct.ne;++e)for(int i=0;i<m;++i)
  {const double x=direct.edges[e]+(direct.edges[e+1]-direct.edges[e])*direct.retained.physics.basis.nodes[i];const auto expected=direct.retained.InitialAt(x);
    for(int c=0;c<4;++c)initialization=std::max(initialization,std::abs(initial_low[direct.retained.Id(e,i,c)]-expected[c]));}
  State low(direct.ne*m*4,0.);
  auto fill=[&](double slope,bool variable_density)
  {for(int e=0;e<direct.ne;++e)for(int i=0;i<m;++i)
    {const double x=direct.edges[e]+(direct.edges[e+1]-direct.edges[e])*direct.retained.physics.basis.nodes[i],rho=variable_density?1.+.1*x:2.,v=slope*x;const int j=direct.retained.Id(e,i,0);
      low[j]=rho;low[j+1]=rho*v;low[j+2]=0.;low[j+3]=1./(gamma_euler-1.)+.5*rho*v*v;}};
  fill(0.,true);State state=direct.state_transfer.Apply(direct.state_transfer.E,low,direct.ne,4);TrackWork work;State sigma=direct.Closure(state,0.,work);std::array<double,4> budget{};
  State rhs=direct.Residual(state,sigma,0.,work,budget);double rest=0.;for(double value:rhs)rest=std::max(rest,std::abs(value));
  const double a=.13;fill(a,false);state=direct.state_transfer.Apply(direct.state_transfer.E,low,direct.ne,4);sigma=direct.Closure(state,0.,work);rhs=direct.Residual(state,sigma,0.,work,budget);
  const double exact_sigma=4.*t.base.alpha*a*a,P=1.+exact_sigma;double stress=0.,affine=0.;
  for(double value:sigma)stress=std::max(stress,std::abs(value-exact_sigma));
  for(int e=0;e<direct.ne;++e)for(int i=0;i<direct.n;++i)
  {const double x=direct.edges[e]+(direct.edges[e+1]-direct.edges[e])*direct.physics.basis.nodes[i];const std::array<double,4> exact{{-2.*a,-4.*a*a*x,0.,-a*(1./(gamma_euler-1.)+P+3.*a*a*x*x)}};
    for(int c=0;c<4;++c)affine=std::max(affine,std::abs(rhs[direct.Id(e,i,c)]-exact[c]));}
  // Discontinuous polynomial fixture engages central lifts and slope work on
  // unequal neighboring cells, then compares direct vs full embedded assembly.
  for(int e=0;e<direct.ne;++e)for(int i=0;i<m;++i)
  {const double z=direct.retained.physics.basis.nodes[i];const int j=direct.retained.Id(e,i,0);low[j]=1.+.05*z;low[j+1]=.2+.01*e+.07*z*z*z;low[j+2]=.03*z*z;low[j+3]=3.+.04*z*z;}
  state=direct.state_transfer.Apply(direct.state_transfer.E,low,direct.ne,4);const State physical=direct.Reconstruct(state);double mean_error=0.;
  for(int e=0;e<direct.ne;++e)for(int c=0;c<4;++c)
  {double a0=0.,b0=0.;for(int i=0;i<m;++i)a0+=direct.retained.physics.basis.weights[i]*low[direct.retained.Id(e,i,c)];for(int i=0;i<direct.n;++i)b0+=direct.physics.basis.weights[i]*physical[direct.Id(e,i,c)];mean_error=std::max(mean_error,std::abs((direct.edges[e+1]-direct.edges[e])*(a0-b0)));}
  const auto sd=direct.Closure(state,0.,work),se=embedded.Closure(state,0.,work);std::array<double,4> bd{},be{};
  const auto rd=direct.Residual(state,sd,0.,work,bd),re=embedded.Residual(state,se,0.,work,be);
  const double closure_parity=PatchRelativeDifference(sd,se),rhs_parity=PatchRelativeDifference(rd,re);
  HPPatchProblem curved(mo,po,.25,true,.25),first_only(mo,po,.25,true,0.);TrackWork cw;std::array<double,4> cb{},fb{};
  const auto cs=curved.Closure(state,0.,cw);const auto cr=curved.Residual(state,cs,0.,cw,cb),fr=first_only.Residual(state,cs,0.,cw,fb);State delta(cr.size());
  double fixed=0.,active=0.,curvature_power=0.,curvature_mean=0.;
  for(std::size_t j=0;j<cr.size();++j){delta[j]=cr[j]-fr[j];if(j%4==0||j%4==3)fixed=std::max(fixed,std::abs(delta[j]));else active=std::max(active,std::abs(delta[j]));}
  for(int e=0;e<direct.ne;++e)
  {std::array<double,4> mean{};const double h=direct.edges[e+1]-direct.edges[e];
    for(std::size_t q=0;q<direct.weights.size();++q)
    {const auto value=direct.Point(physical,e,direct.shapes[q]),change=direct.Point(delta,e,direct.shapes[q]);
      for(int c=0;c<4;++c)mean[c]+=h*direct.weights[q]*change[c];curvature_power+=h*direct.weights[q]*(value[1]*change[1]+value[2]*change[2])/value[0];}
    for(double value:mean)curvature_mean=std::max(curvature_mean,std::abs(value));}
  const double curvature_power_defect=std::abs(curvature_power+curved.maximum_curvature_dissipation);
  if(fixed!=0.||active<1.e-10||curvature_power>=0.||curvature_power_defect>1.e-9||curvature_mean>1.e-9||curved.curvature_identity_error>1.e-9)
    throw std::runtime_error("nonuniform curvature correction activation/kinetic/mean identity failed");
  HPPatchOptions uniform;uniform.policy="uniform";HPPatchProblem current(mo,uniform,.5);SlopePenaltyProblem old(mo,true,.5,true);
  const auto uc=MemoryRun(current),uo=MemoryRun(old);const double uniform_parity=std::max(PatchRelativeDifference(uc.u,uo.u),PatchRelativeDifference(uc.sigma,uo.sigma));
  const auto evolved=MemoryRun(direct);
  if(initialization>1.e-11||rest>1.e-9||stress>1.e-10||affine>1.e-9||mean_error>1.e-11||closure_parity>1.e-9||rhs_parity>1.e-9||uniform_parity>1.e-12||
     direct.maximum_identity_error>1.e-9||work.max_rhs_defect>1.e-9||!uc.failure.empty()||!uo.failure.empty()||!evolved.failure.empty())
    throw std::runtime_error("static hp patch geometry/mass/affine/closure/conservation/parity self-test failed");
  std::ostringstream f;f<<std::setprecision(17)<<"{\"initialization_max_error\":"<<initialization<<",\"rest_rhs_max\":"<<rest<<",\"affine_sigma_max\":"<<stress<<",\"affine_rhs_max\":"<<affine
    <<",\"physical_mean_max_defect\":"<<mean_error<<",\"direct_embedded_sigma_relative_max\":"<<closure_parity<<",\"direct_embedded_rhs_relative_max\":"<<rhs_parity
    <<",\"uniform_trajectory_parity\":"<<uniform_parity<<",\"slope_identity_defect\":"<<direct.maximum_identity_error<<",\"constant_test_defect\":"<<work.max_rhs_defect<<",\"curvature_nonuniform_power\":"<<curvature_power<<",\"curvature_nonuniform_power_defect\":"<<curvature_power_defect<<",\"curvature_nonuniform_mean_defect\":"<<curvature_mean<<"}";return f.str();
}
}
#ifndef DG_IGR_HP_PATCH_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    MemoryOptions mo;HPPatchOptions po;double strength=.5,curvature=0.;std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--mesh-policy"||flag=="--patch-elements"||flag=="--patch-width-multiple"||flag=="--patch-phase"||flag=="--slope-strength"||flag=="--curvature-strength")
      {if(++i==argc)throw std::runtime_error("missing hp patch option");const std::string value=argv[i];
        if(flag=="--mesh-policy")po.policy=value;else if(flag=="--patch-elements")po.inner_elements=std::stoi(value);else if(flag=="--patch-width-multiple")po.width_multiple=std::stod(value);else if(flag=="--patch-phase")po.phase=std::stod(value);else if(flag=="--curvature-strength")curvature=std::stod(value);else strength=std::stod(value);}
      else common.push_back(argv[i]);
    }
    mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;
    if(t.self_test){std::cout<<HPPatchSelfTest()<<'\n';return 0;}
    const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;
    const auto start=Clock::now();HPPatchProblem pr(mo,po,strength,true,curvature);const double setup=Wall(start);auto result=MemoryRun(pr);result.work.setup=setup;
    const State moment_coefficients=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);if(!result.u.empty())result.u=pr.Reconstruct(result.u);
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"joint-density-weighted-primitive\",\"volume_projection\":\"conservative-moments\",\"retained_order\":"<<degree
      <<",\"primitive_source_faces\":\"central\",\"moment_operator\":\"direct\",\"slope_sensor\":\"constant\",\"direct_assembly_seconds\":"<<pr.direct_assembly_seconds<<",\"slope_strength\":"<<strength<<",\"stress_order\":"<<degree
      <<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns<<",\"sigma_serialized_order\":"<<pr.p
      <<",\"injected_state_time\":0,\"local_velocity_factors\":"<<pr.local_velocity_factors<<",\"reconstruction_cells\":"<<pr.reconstruction_cells
      <<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"moment_transfer_seconds\":"<<pr.transfer_seconds<<",\"galerkin_seconds\":"<<pr.galerkin_seconds
      <<",\"slope_penalty_seconds\":"<<pr.slope_seconds<<",\"slope_penalty_faces\":"<<pr.slope_faces<<",\"maximum_slope_dissipation\":"<<pr.maximum_dissipation
      <<",\"maximum_slope_jump\":"<<pr.maximum_slope_jump<<",\"maximum_slope_identity_error\":"<<pr.maximum_identity_error<<",\"maximum_slope_momentum_rhs_ratio\":"<<pr.maximum_rhs_ratio
      <<",\"curvature_strength\":"<<curvature<<",\"curvature_seconds\":"<<pr.curvature_seconds<<",\"curvature_faces\":"<<pr.curvature_faces<<",\"maximum_curvature_dissipation\":"<<pr.maximum_curvature_dissipation<<",\"curvature_identity_error\":"<<pr.curvature_identity_error
      <<",\"mesh_policy\":\""<<po.policy<<"\",\"patch_width_multiple\":"<<po.width_multiple<<",\"patch_phase\":"<<po.phase<<",\"patch_left\":"<<pr.geometry.left<<",\"patch_right\":"<<pr.geometry.right
      <<",\"patch_left_elements\":"<<pr.geometry.left_elements<<",\"patch_inner_elements\":"<<pr.geometry.inner_elements<<",\"patch_right_elements\":"<<pr.geometry.right_elements
      <<",\"mesh_hmin\":"<<pr.geometry.hmin<<",\"mesh_hmax\":"<<pr.geometry.hmax<<",\"maximum_adjacent_h_ratio\":"<<pr.geometry.maximum_adjacent_ratio
      <<",\"mesh_space_edges_synchronized\":"<<(pr.edges==pr.retained.edges&&pr.edges==pr.stress.edges?"true":"false")
      <<",\"patch_rule\":\"center plus/minus width_multiple*width, extended by declared speed*target_time; optional fixed inner-cell phase\",\"moment_coefficients\":";
    TrackArray(extra,moment_coefficients);BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif

// Test-only conservative derivative-jump stabilization of the joint primitive
// manifold. For its kinetic functional, delta K = -sum tau [v_x]^2 exactly.
// Conservative cell means and the full-energy equation are unchanged.
#define DG_IGR_PRIMITIVE_MOMENTS_LIBRARY_ONLY
#include "igr_dg_primitive_moments_probe.cpp"
#undef DG_IGR_PRIMITIVE_MOMENTS_LIBRARY_ONLY
#include "igr_dg_primitive_direct.hpp"

namespace
{
struct SlopePenaltyProblem:PrimitiveMomentProblem
{
  double theta;bool direct_operator;std::vector<State> primitive_nodes,low_derivatives;std::array<State,2> slope_tests,dual_slopes;
  std::string slope_sensor;
  mutable double direct_assembly_seconds=0.;
  mutable double slope_seconds=0.,maximum_dissipation=0.,maximum_slope_jump=0.,maximum_identity_error=0.,maximum_rhs_ratio=0.;
  mutable std::uint64_t slope_faces=0;
  SlopePenaltyProblem(const MemoryOptions& mo,bool central,double strength,bool direct=false,std::string sensor="constant"):PrimitiveMomentProblem(mo,central),theta(strength),direct_operator(direct),slope_sensor(std::move(sensor))
  {
    if(!(theta>=0.)||!std::isfinite(theta))throw std::runtime_error("invalid slope penalty strength");
    if(slope_sensor!="constant"&&slope_sensor!="relative")throw std::runtime_error("unknown slope sensor");
    for(double z:retained.physics.basis.nodes)primitive_nodes.push_back(Shape(z,false));
    for(std::size_t q=0;q<weights.size();++q)
    {double z=0.;for(int i=0;i<n;++i)z+=shapes[q][i]*physics.basis.nodes[i];low_derivatives.push_back(retained.Shape(z,true));}
    for(int side=0;side<2;++side)
    {
      slope_tests[side]=retained.Shape(side?1.:0.,true);dual_slopes[side].assign(retained.n,0.);
      for(int i=0;i<retained.n;++i)for(int j=0;j<retained.n;++j)dual_slopes[side][i]+=retained.mass_inverse(i,j)*slope_tests[side][j];
    }
  }
  void MixedSystem(const State& physical,double time,TrackWork& work,BandMatrix& a,State& b)const override
  {
    if(!direct_operator){PrimitiveMomentProblem::MixedSystem(physical,time,work,a,b);return;}
    const auto start=Clock::now();PrimitiveDirectSystem(*this,low_derivatives,physical,work,a,b);direct_assembly_seconds+=Wall(start);
  }
  State Residual(const State& moments,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    State rhs=direct_operator?PrimitiveDirectResidual(*this,low_derivatives,Reconstruct(moments),sigma,work,budget):PrimitiveMomentProblem::Residual(moments,sigma,time,work,budget);
    if(theta==0.)return rhs;
    const auto start=Clock::now();const State& physical=Reconstruct(moments);const int m=retained.n;
    State velocity(ne*m*2),addition(ne*m*2,0.);std::vector<std::array<double,2>> slopes(2*ne);
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];
      for(int i=0;i<m;++i)
      {
        const auto u=Point(physical,e,primitive_nodes[i]);
        for(int c=0;c<2;++c)
        {const double v=u[c+1]/u[0];velocity[(e*m+i)*2+c]=v;for(int side=0;side<2;++side)slopes[2*e+side][c]+=slope_tests[side][i]*v/h;}
      }
    }
    double dissipation=0.;const double degree=retained.p;
    const double trace_norm=degree*degree*(degree+1)*(degree+1)*(degree+2)*(degree+2)/3.;
    for(int f=1;f<ne;++f)
    {
      ++slope_faces;const double hl=edges[f]-edges[f-1],hr=edges[f+1]-edges[f],heff=2.*hl*hr/(hl+hr);
      double al=0.,ar=0.;physics.Flux(&physical[Id(f-1,p,0)],sigma[SigmaIndex(f-1,p)],al);physics.Flux(&physical[Id(f,0,0)],sigma[SigmaIndex(f,0)],ar);
      const double rho_l=physical[Id(f-1,p,0)],rho_r=physical[Id(f,0,0)];
      const double spectral_norm=trace_norm*(1./(rho_l*hl*hl*hl)+1./(rho_r*hr*hr*hr));
      const double base_tau=theta*physics.FaceSpeed(al,ar)*(degree+1)*(degree+1)/(heff*spectral_norm);
      for(int c=0;c<2;++c)
      {
        const double gl=slopes[2*(f-1)+1][c],gr=slopes[2*f][c],jump=gl-gr;
        // Prefer a disagreement of slopes over two large agreeing physical
        // gradients. This nonnegative factor is <=2 and retains the same
        // kinetic/mean identities; it is a separate numerical sensor trial.
        const double tiny=1.e-12*physics.FaceSpeed(al,ar)/heff;
        const double activation=slope_sensor=="relative"?jump*jump/(gl*gl+gr*gr+tiny*tiny):1.;
        const double tau=base_tau*activation;maximum_slope_jump=std::max(maximum_slope_jump,std::abs(jump));dissipation+=tau*jump*jump;
        for(int i=0;i<m;++i)
        {addition[((f-1)*m+i)*2+c]-=tau*jump*dual_slopes[1][i]/(hl*hl);addition[(f*m+i)*2+c]+=tau*jump*dual_slopes[0][i]/(hr*hr);}
      }
    }
    double power=0.,correction_norm=0.,rhs_norm=0.;
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];std::array<double,2> mean{};
      for(std::size_t q=0;q<weights.size();++q)for(int c=0;c<2;++c)
      {
        double v=0.,delta=0.,r=0.;for(int i=0;i<m;++i){v+=low_shapes[q][i]*velocity[(e*m+i)*2+c];delta+=low_shapes[q][i]*addition[(e*m+i)*2+c];}
        for(int i=0;i<n;++i)r+=shapes[q][i]*rhs[Id(e,i,c+1)];
        power+=h*weights[q]*v*delta;mean[c]+=h*weights[q]*delta;correction_norm+=h*weights[q]*delta*delta;rhs_norm+=h*weights[q]*r*r;
      }
      for(double value:mean)maximum_identity_error=std::max(maximum_identity_error,std::abs(value));
      for(int i=0;i<n;++i)for(int j=0;j<m;++j)for(int c=0;c<2;++c)rhs[Id(e,i,c+1)]+=state_transfer.E(i,j)*addition[(e*m+j)*2+c];
    }
    maximum_identity_error=std::max(maximum_identity_error,std::abs(power+dissipation));maximum_dissipation=std::max(maximum_dissipation,dissipation);
    maximum_rhs_ratio=std::max(maximum_rhs_ratio,std::sqrt(correction_norm/std::max(rhs_norm,1.e-100)));
    const auto totals=Integrals(rhs,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    const double elapsed=Wall(start);slope_seconds+=elapsed;work.transport+=elapsed;return rhs;
  }
};
void SlopeSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;
  t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=9;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=19;
  mo.retained_order=mo.stress_order=3;SlopePenaltyProblem pr(mo,true,.5),base(mo,true,0.);State low(pr.retained.ns*4);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.retained.n;++i)
  {
    const double z=pr.retained.physics.basis.nodes[i];const int j=pr.retained.Id(e,i,0);
    low[j]=1.1+.03*z;low[j+1]=.4+.08*z*z*z+.01*e;low[j+2]=-.1+.03*z*z;low[j+3]=3.+.07*z*z*z;
  }
  State moments=pr.state_transfer.Apply(pr.state_transfer.E,low,pr.ne,4);TrackWork work;const State sigma=pr.Closure(moments,0.,work);
  std::array<double,4>a{},b{};const auto r=pr.Residual(moments,sigma,0.,work,a),rb=base.Residual(moments,sigma,0.,work,b);
  double fixed=0.,nonzero=0.;for(std::size_t i=0;i<r.size();++i){if(i%4==0||i%4==3)fixed=std::max(fixed,std::abs(r[i]-rb[i]));else nonzero=std::max(nonzero,std::abs(r[i]-rb[i]));}
  if(fixed!=0.||nonzero<1.e-10||pr.maximum_identity_error>1.e-9||work.max_rhs_defect>1.e-9)throw std::runtime_error("slope penalty kinetic/mean identity failed");
  const double p=pr.retained.p;double norm=0.;for(int i=0;i<pr.retained.n;++i)norm+=pr.slope_tests[1][i]*pr.dual_slopes[1][i];
  const double expected=p*p*(p+1)*(p+1)*(p+2)*(p+2)/3.;if(std::abs(norm-expected)>1.e-9)throw std::runtime_error("slope trace norm failed");
  SlopePenaltyProblem direct(mo,true,.5,true);TrackWork direct_work;const State sd=direct.Closure(moments,0.,direct_work);std::array<double,4> bd{};const State rd=direct.Residual(moments,sd,0.,direct_work,bd);
  double parity=0.;for(std::size_t i=0;i<sigma.size();++i)parity=std::max(parity,std::abs(sd[i]-sigma[i]));for(std::size_t i=0;i<r.size();++i)parity=std::max(parity,std::abs(rd[i]-r[i]));
  if(parity>1.e-9)throw std::runtime_error("direct primitive Galerkin/transport parity failed");
  SlopePenaltyProblem sensed(mo,true,1.,true,"relative");TrackWork sensed_work;std::array<double,4> sb{};sensed.Residual(moments,sd,0.,sensed_work,sb);
  if(sensed.maximum_identity_error>1.e-9||sensed_work.max_rhs_defect>1.e-9)throw std::runtime_error("relative slope sensor kinetic/mean identity failed");
  std::cout<<"slope penalty checks passed: dissipation="<<pr.maximum_dissipation<<" power-defect="<<pr.maximum_identity_error<<" trace-norm="<<norm<<'\n';
}
}
#ifndef DG_IGR_SLOPE_PENALTY_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    MemoryOptions mo;std::string source_faces="central",moment_operator="embedded",sensor="constant";double strength=.5;std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--primitive-source-faces"||flag=="--slope-strength"||flag=="--moment-operator"||flag=="--slope-sensor")
      {if(++i==argc)throw std::runtime_error("missing slope option");if(flag=="--primitive-source-faces")source_faces=argv[i];else if(flag=="--moment-operator")moment_operator=argv[i];else if(flag=="--slope-sensor")sensor=argv[i];else strength=std::stod(argv[i]);}else common.push_back(argv[i]);
    }
    mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;
    if(t.self_test){SlopeSelfTest();return 0;}
    if(source_faces!="central"&&source_faces!="broken")throw std::runtime_error("unknown primitive source selector");
    if(moment_operator!="embedded"&&moment_operator!="direct")throw std::runtime_error("unknown primitive operator selector");
    const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;
    const auto start=Clock::now();SlopePenaltyProblem pr(mo,source_faces=="central",strength,moment_operator=="direct",sensor);const double setup=Wall(start);auto result=MemoryRun(pr);result.work.setup=setup;
    const State moment_coefficients=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);if(!result.u.empty())result.u=pr.Reconstruct(result.u);
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"joint-density-weighted-primitive\",\"volume_projection\":\"conservative-moments\",\"retained_order\":"<<degree
      <<",\"primitive_source_faces\":\""<<source_faces<<"\",\"moment_operator\":\""<<moment_operator<<"\",\"slope_sensor\":\""<<sensor<<"\",\"direct_assembly_seconds\":"<<pr.direct_assembly_seconds<<",\"slope_strength\":"<<strength<<",\"stress_order\":"<<degree
      <<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns<<",\"sigma_serialized_order\":"<<pr.p
      <<",\"injected_state_time\":0,\"local_velocity_factors\":"<<pr.local_velocity_factors<<",\"reconstruction_cells\":"<<pr.reconstruction_cells
      <<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"moment_transfer_seconds\":"<<pr.transfer_seconds<<",\"galerkin_seconds\":"<<pr.galerkin_seconds
      <<",\"slope_penalty_seconds\":"<<pr.slope_seconds<<",\"slope_penalty_faces\":"<<pr.slope_faces<<",\"maximum_slope_dissipation\":"<<pr.maximum_dissipation
      <<",\"maximum_slope_jump\":"<<pr.maximum_slope_jump<<",\"maximum_slope_identity_error\":"<<pr.maximum_identity_error<<",\"maximum_slope_momentum_rhs_ratio\":"<<pr.maximum_rhs_ratio<<",\"moment_coefficients\":";
    TrackArray(extra,moment_coefficients);BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif

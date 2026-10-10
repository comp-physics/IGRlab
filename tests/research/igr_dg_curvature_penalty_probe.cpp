// Test-only extension of the kinetic-compatible face correction to jumps of
// second velocity derivatives. No source/model/alpha or time-coupling changes.
#define DG_IGR_SLOPE_PENALTY_LIBRARY_ONLY
#include "igr_dg_slope_penalty_probe.cpp"
#undef DG_IGR_SLOPE_PENALTY_LIBRARY_ONLY
namespace
{
struct CurvatureProblem: SlopePenaltyProblem
{
  double curvature_strength,curvature_trace_norm=0.;
  std::array<State,2> curvature_test,curvature_dual;
  mutable double curvature_seconds=0.,curvature_identity_error=0.,maximum_curvature_dissipation=0.;
  mutable std::uint64_t curvature_faces=0;
  CurvatureProblem(const MemoryOptions& mo,double first,double second):SlopePenaltyProblem(mo,true,first,true),curvature_strength(second)
  {
    if(second<0.||!std::isfinite(second)||retained.p<1||(retained.p<2&&second!=0.))throw std::runtime_error("invalid curvature strength/degree");
    const int m=retained.n;std::vector<State> d;
    for(double z:retained.physics.basis.nodes)d.push_back(retained.Shape(z,true));
    for(int side=0;side<2;++side)
    {
      curvature_test[side].assign(m,0.);curvature_dual[side].assign(m,0.);
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)curvature_test[side][i]+=slope_tests[side][j]*d[j][i];
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)curvature_dual[side][i]+=retained.mass_inverse(i,j)*curvature_test[side][j];
    }
    for(double k=2.;k<=retained.p;++k)
    {const double value=(k-1.)*k*(k+1.)*(k+2.)/2.;curvature_trace_norm+=(2.*k+1.)*value*value;}
  }
  State Residual(const State& moments,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    State rhs=SlopePenaltyProblem::Residual(moments,sigma,time,work,budget);if(curvature_strength==0.)return rhs;
    const auto start=Clock::now();const State& physical=Reconstruct(moments);const int m=retained.n;
    State velocity(ne*m*2,0.),addition(ne*m*2,0.);std::vector<std::array<double,2>> traces(2*ne);
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];
      for(int i=0;i<m;++i)
      {const auto u=Point(physical,e,primitive_nodes[i]);for(int c=0;c<2;++c)
        {const double v=u[c+1]/u[0];velocity[(e*m+i)*2+c]=v;for(int side=0;side<2;++side)traces[2*e+side][c]+=curvature_test[side][i]*v/(h*h);}}
    }
    double dissipation=0.;
    for(int f=1;f<ne;++f)
    {
      ++curvature_faces;const double hl=edges[f]-edges[f-1],hr=edges[f+1]-edges[f],heff=2.*hl*hr/(hl+hr);
      double al=0.,ar=0.;physics.Flux(&physical[Id(f-1,p,0)],sigma[SigmaIndex(f-1,p)],al);physics.Flux(&physical[Id(f,0,0)],sigma[SigmaIndex(f,0)],ar);
      const double norm=curvature_trace_norm*(1./(physical[Id(f-1,p,0)]*std::pow(hl,5))+1./(physical[Id(f,0,0)]*std::pow(hr,5)));
      const double tau=curvature_strength*physics.FaceSpeed(al,ar)*m*m/(heff*norm);
      for(int c=0;c<2;++c)
      {
        const double jump=traces[2*f-1][c]-traces[2*f][c];dissipation+=tau*jump*jump;
        for(int i=0;i<m;++i)
        {addition[((f-1)*m+i)*2+c]-=tau*jump*curvature_dual[1][i]/(hl*hl*hl);addition[(f*m+i)*2+c]+=tau*jump*curvature_dual[0][i]/(hr*hr*hr);}
      }
    }
    double power=0.;
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];std::array<double,2> mean{};
      for(std::size_t q=0;q<weights.size();++q)for(int c=0;c<2;++c)
      {double v=0.,delta=0.;for(int i=0;i<m;++i){v+=low_shapes[q][i]*velocity[(e*m+i)*2+c];delta+=low_shapes[q][i]*addition[(e*m+i)*2+c];}power+=h*weights[q]*v*delta;mean[c]+=h*weights[q]*delta;}
      for(double v:mean)curvature_identity_error=std::max(curvature_identity_error,std::abs(v));
      for(int i=0;i<n;++i)for(int j=0;j<m;++j)for(int c=0;c<2;++c)rhs[Id(e,i,c+1)]+=state_transfer.E(i,j)*addition[(e*m+j)*2+c];
    }
    curvature_identity_error=std::max(curvature_identity_error,std::abs(power+dissipation));maximum_curvature_dissipation=std::max(maximum_curvature_dissipation,dissipation);
    const auto totals=Integrals(rhs,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    const double elapsed=Wall(start);curvature_seconds+=elapsed;work.transport+=elapsed;return rhs;
  }
};
void CurvatureSelfTest()
{
  SlopeSelfTest();MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;
  t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=15;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=21;mo.retained_order=mo.stress_order=5;
  CurvatureProblem pr(mo,.375,.125),base(mo,.375,0.);State low(pr.retained.ns*4);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.retained.n;++i)
  {const double z=pr.retained.physics.basis.nodes[i];const int j=pr.retained.Id(e,i,0);low[j]=1.1+.03*z;low[j+1]=.3+.03*z*z*z+.01*e;low[j+2]=.02*z*z;low[j+3]=3.+.01*z;}
  State moments=pr.state_transfer.Apply(pr.state_transfer.E,low,pr.ne,4);TrackWork w;const State s=pr.Closure(moments,0.,w);std::array<double,4>a{},b{};
  const auto r=pr.Residual(moments,s,0.,w,a),r0=base.Residual(moments,s,0.,w,b);double fixed=0.,norm=0.,nonzero=0.;
  for(std::size_t i=0;i<r.size();++i){if(i%4==0||i%4==3)fixed=std::max(fixed,std::abs(r[i]-r0[i]));else nonzero=std::max(nonzero,std::abs(r[i]-r0[i]));}
  for(int i=0;i<pr.retained.n;++i)norm+=pr.curvature_test[1][i]*pr.curvature_dual[1][i];
  if(std::abs(norm-pr.curvature_trace_norm)>1.e-9*norm||fixed!=0.||nonzero<1.e-10||pr.curvature_identity_error>1.e-9||w.max_rhs_defect>1.e-9)throw std::runtime_error("curvature trace/kinetic/mean check failed");
  // On the reference cell the assembled second derivative must differentiate
  // each monomial of degree <=p, including the constant and affine nullspace.
  double polynomial=0.;for(int k=0;k<=pr.retained.p;++k)for(int side=0;side<2;++side)
  {double actual=0.;for(int i=0;i<pr.retained.n;++i)actual+=pr.curvature_test[side][i]*std::pow(pr.retained.physics.basis.nodes[i],k);const double exact=k<2?0.:k*(k-1)*std::pow(double(side),k-2);polynomial=std::max(polynomial,std::abs(actual-exact));}
  if(polynomial>1.e-10)throw std::runtime_error("curvature polynomial derivative failed");
  std::cout<<"curvature checks passed: identity="<<pr.curvature_identity_error<<" trace-norm="<<norm<<" polynomial="<<polynomial<<'\n';
}
}
#ifndef DG_IGR_CURVATURE_PENALTY_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    double first=.375,second=.125;std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {const std::string flag=argv[i];if(flag=="--slope-strength"||flag=="--curvature-strength")
      {if(++i==argc)throw std::runtime_error("missing curvature option");(flag=="--slope-strength"?first:second)=std::stod(argv[i]);}else common.push_back(argv[i]);}
    MemoryOptions mo;mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;
    if(t.self_test){CurvatureSelfTest();return 0;}const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;
    const auto start=Clock::now();CurvatureProblem pr(mo,first,second);const double setup=Wall(start);auto result=MemoryRun(pr);result.work.setup=setup;
    const State coefficients=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);if(!result.u.empty())result.u=pr.Reconstruct(result.u);
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"joint-density-weighted-primitive\",\"volume_projection\":\"conservative-moments\",\"retained_order\":"<<degree
      <<",\"primitive_source_faces\":\"central\",\"moment_operator\":\"direct\",\"slope_sensor\":\"constant\",\"slope_strength\":"<<first<<",\"curvature_strength\":"<<second<<",\"stress_order\":"<<degree
      <<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns<<",\"sigma_serialized_order\":"<<pr.p<<",\"injected_state_time\":0"
      <<",\"local_velocity_factors\":"<<pr.local_velocity_factors<<",\"reconstruction_cells\":"<<pr.reconstruction_cells<<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds
      <<",\"moment_transfer_seconds\":"<<pr.transfer_seconds<<",\"galerkin_seconds\":"<<pr.galerkin_seconds<<",\"direct_assembly_seconds\":"<<pr.direct_assembly_seconds
      <<",\"slope_penalty_seconds\":"<<pr.slope_seconds<<",\"slope_penalty_faces\":"<<pr.slope_faces<<",\"maximum_slope_identity_error\":"<<pr.maximum_identity_error
      <<",\"curvature_seconds\":"<<pr.curvature_seconds<<",\"curvature_faces\":"<<pr.curvature_faces<<",\"maximum_curvature_identity_error\":"<<pr.curvature_identity_error<<",\"maximum_curvature_dissipation\":"<<pr.maximum_curvature_dissipation<<",\"moment_coefficients\":";
    TrackArray(extra,coefficients);BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif

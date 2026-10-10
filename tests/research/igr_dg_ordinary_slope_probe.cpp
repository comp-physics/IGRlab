// Test-only transfer of the derivative-jump remedy to ordinary conservative DG.
// Project physical velocity with the transport mass before differentiating it:
// this is the gradient of the quadrature-defined ordinary kinetic functional.
#define DG_IGR_MOMENT_MEMORY_LIBRARY_ONLY
#include "igr_dg_moment_memory_probe.cpp"
#undef DG_IGR_MOMENT_MEMORY_LIBRARY_ONLY
namespace
{
struct OrdinarySlopeProblem:MemoryProblem
{
  double theta;bool central_source;std::array<State,2> slope_tests,dual_slopes;
  mutable double slope_seconds=0.,max_slope_power_error=0.,max_slope_dissipation=0.;
  mutable std::uint64_t slope_faces=0,slope_projection_cells=0;
  OrdinarySlopeProblem(const MemoryOptions& mo,double strength,bool central=false):MemoryProblem(mo),theta(strength),central_source(central)
  {
    if(theta<0.||!std::isfinite(theta))throw std::runtime_error("invalid ordinary slope strength");
    if(central&&(boundary.left=="wall"||boundary.right=="wall"))throw std::runtime_error("ordinary central planar control requires open boundaries");
    for(int side=0;side<2;++side)
    {slope_tests[side]=Shape(side?1.:0.,true);dual_slopes[side].assign(n,0.);for(int i=0;i<n;++i)for(int j=0;j<n;++j)dual_slopes[side][i]+=mass_inverse(i,j)*slope_tests[side][j];}
  }
  State Closure(const State& u,double time,TrackWork& work)const
  {
    if(!central_source)return MemoryProblem::Closure(u,time,work);
    const auto start=Clock::now();BandMatrix a;State b;AssembleClosure(u,time,work,a,b);
    // Retain the actual rational quotient derivative and add only the P_p
    // central face lift. Original coefficient/source assembly remains charged.
    std::fill(b.begin(),b.end(),0.);double integral=0.;
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];
      for(std::size_t q=0;q<weights.size();++q)
      {
        const auto value=Point(u,e,shapes[q]);double dr=0.,dm=0.;
        for(int i=0;i<n;++i){dr+=derivatives[q][i]*u[Id(e,i,0)]/h;dm+=derivatives[q][i]*u[Id(e,i,1)]/h;}
        double g=(dm-value[1]/value[0]*dr)/value[0];
        for(int side=0;side<2;++side)
        {
          const int other=e+(side?1:-1);if(other<0||other>=ne)continue;
          const int endpoint=side?p:0,opposite=side?0:p;
          const double jump=u[Id(e,endpoint,1)]/u[Id(e,endpoint,0)]-u[Id(other,opposite,1)]/u[Id(other,opposite,0)];
          double lift=0.;for(int i=0;i<n;++i)lift+=shapes[q][i]*mass_inverse(i,endpoint)/h;
          g+=(side?-.5:.5)*jump*lift;
        }
        const double source=2.*o.base.alpha*g*g,weight=h*weights[q];integral+=weight*source;
        for(int i=0;i<n;++i)b[e*n+i]+=weight*shapes[q][i]*source;
        ++work.source_points;
      }
    }
    work.current_source_integral=integral;return SolveCurrentSystem(u,a,b,work,start,{{0.,0.}},{{0.,0.}});
  }
  State Residual(const State& u,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    State rhs=RecoveryProblem::Residual(u,sigma,time,work,budget);if(theta==0.)return rhs;const auto start=Clock::now();
    State velocity(ne*n*2,0.),addition(ne*n*2,0.);std::vector<std::array<double,2>> slopes(2*ne);
    for(int e=0;e<ne;++e)
    {
      ++slope_projection_cells;State load(n*2,0.);const double h=edges[e+1]-edges[e];
      for(std::size_t q=0;q<weights.size();++q)
      {const auto v=Point(u,e,shapes[q]);for(int i=0;i<n;++i)for(int c=0;c<2;++c)load[i*2+c]+=weights[q]*shapes[q][i]*v[c+1]/v[0];}
      for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int c=0;c<2;++c)velocity[(e*n+i)*2+c]+=mass_inverse(i,j)*load[j*2+c];
      for(int side=0;side<2;++side)for(int i=0;i<n;++i)for(int c=0;c<2;++c)slopes[2*e+side][c]+=slope_tests[side][i]*velocity[(e*n+i)*2+c]/h;
    }
    const double degree=p,trace_norm=degree*degree*(degree+1)*(degree+1)*(degree+2)*(degree+2)/3.;double dissipation=0.;
    for(int f=1;f<ne;++f)
    {
      ++slope_faces;const double hl=edges[f]-edges[f-1],hr=edges[f+1]-edges[f],heff=2.*hl*hr/(hl+hr);double al=0.,ar=0.;
      physics.Flux(&u[Id(f-1,p,0)],sigma[(f-1)*n+p],al);physics.Flux(&u[Id(f,0,0)],sigma[f*n],ar);
      const double norm=trace_norm*(1./(u[Id(f-1,p,0)]*hl*hl*hl)+1./(u[Id(f,0,0)]*hr*hr*hr));
      const double tau=theta*physics.FaceSpeed(al,ar)*(degree+1)*(degree+1)/(heff*norm);
      for(int c=0;c<2;++c)
      {const double jump=slopes[2*f-1][c]-slopes[2*f][c];dissipation+=tau*jump*jump;for(int i=0;i<n;++i)
        {addition[((f-1)*n+i)*2+c]-=tau*jump*dual_slopes[1][i]/(hl*hl);addition[(f*n+i)*2+c]+=tau*jump*dual_slopes[0][i]/(hr*hr);}}
    }
    double power=0.;
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];std::array<double,2> mean{};
      for(std::size_t q=0;q<weights.size();++q)
      {
        const auto v=Point(u,e,shapes[q]);for(int c=0;c<2;++c)
        {double delta=0.;for(int i=0;i<n;++i)delta+=shapes[q][i]*addition[(e*n+i)*2+c];power+=h*weights[q]*v[c+1]/v[0]*delta;mean[c]+=h*weights[q]*delta;}
      }
      for(double value:mean)max_slope_power_error=std::max(max_slope_power_error,std::abs(value));
      for(int i=0;i<n;++i)for(int c=0;c<2;++c)rhs[Id(e,i,c+1)]+=addition[(e*n+i)*2+c];
    }
    max_slope_dissipation=std::max(max_slope_dissipation,dissipation);max_slope_power_error=std::max(max_slope_power_error,std::abs(power+dissipation));
    const auto totals=Integrals(rhs,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    const double elapsed=Wall(start);slope_seconds+=elapsed;work.transport+=elapsed;return rhs;
  }
};
void OrdinarySlopeSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;t.initial="entropy";t.mode="uniform";t.closure_space="l2";t.closure_boundary="natural";t.base.elements=8;t.base.order=5;t.base.alpha=.04;t.quad_extra=25;mo.retained_order=mo.stress_order=5;
  OrdinarySlopeProblem pr(mo,.5),base(mo,0.);State u=pr.Initial();for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {const double z=pr.physics.basis.nodes[i];u[pr.Id(e,i,1)]+=.1*z*z*z+.01*e;u[pr.Id(e,i,2)]+=.03*z*z;}
  TrackWork w;const State s=pr.Closure(u,0.,w);std::array<double,4>a{},b{};const auto r=pr.Residual(u,s,0.,w,a),rb=base.Residual(u,s,0.,w,b);double fixed=0.;for(std::size_t i=0;i<r.size();++i)if(i%4==0||i%4==3)fixed=std::max(fixed,std::abs(r[i]-rb[i]));
  if(fixed!=0.||pr.max_slope_dissipation<1.e-10||pr.max_slope_power_error>1.e-10||w.max_rhs_defect>1.e-9)throw std::runtime_error("ordinary projected-velocity kinetic/mean check failed");
  OrdinarySlopeProblem central(mo,0.,true);State piecewise(u.size());double expected=0.;
  for(int e=0;e<pr.ne;++e)
  {
    const double h=pr.edges[e+1]-pr.edges[e];State face_load(pr.n,0.);
    if(e>0)face_load[0]=.005;if(e+1<pr.ne)face_load[pr.p]=.005;
    for(int i=0;i<pr.n;++i)
    {piecewise[pr.Id(e,i,0)]=1.;piecewise[pr.Id(e,i,1)]=.01*e;piecewise[pr.Id(e,i,2)]=0.;piecewise[pr.Id(e,i,3)]=3.;}
    for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)expected+=2.*pr.o.base.alpha/h*face_load[i]*pr.mass_inverse(i,j)*face_load[j];
  }
  TrackWork cw;central.Closure(piecewise,0.,cw);
  if(std::abs(cw.current_source_integral-expected)>1.e-11||central.max_mean_defect>1.e-10)throw std::runtime_error("ordinary quotient central lift/constant-test check failed");
  std::cout<<"ordinary slope checks passed: kinetic-power="<<pr.max_slope_power_error<<" dissipation="<<pr.max_slope_dissipation<<'\n';
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    MemoryOptions mo;double strength=.5;std::string source_faces="broken";int requested_retained=-1,requested_stress=-1;std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];
      if(flag=="--ordinary-source-faces"){if(++i==argc)throw std::runtime_error("missing ordinary source face selector");source_faces=argv[i];continue;}
      if(flag=="--slope-strength"||flag=="--retained-order"||flag=="--stress-order"||flag=="--memory-time"||flag=="--state-time")
      {
        if(++i==argc)throw std::runtime_error("missing ordinary slope option");
        if(flag=="--slope-strength")strength=std::stod(argv[i]);else if(flag=="--retained-order")requested_retained=std::stoi(argv[i]);else if(flag=="--stress-order")requested_stress=std::stoi(argv[i]);
        else if(std::stod(argv[i])!=0.)throw std::runtime_error("ordinary slope control has no memory reset or shifted input");
      }else common.push_back(argv[i]);
    }
    mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;if(t.self_test){OrdinarySlopeSelfTest();return 0;}
    if(source_faces!="broken"&&source_faces!="central")throw std::runtime_error("unknown ordinary source face selector");
    if((requested_retained!=-1&&requested_retained!=t.base.order)||(requested_stress!=-1&&requested_stress!=t.base.order))throw std::runtime_error("ordinary slope control requires equal polynomial degrees");
    mo.retained_order=mo.stress_order=t.base.order;const auto start=Clock::now();OrdinarySlopeProblem pr(mo,strength,source_faces=="central");const double setup=Wall(start);auto result=MemoryRun(pr);result.work.setup=setup;
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"existing\",\"volume_projection\":\"weak\",\"retained_order\":"<<pr.p<<",\"stress_order\":"<<pr.p
      <<",\"actual_sigma_solve_dofs\":"<<pr.ns<<",\"sigma_serialized_order\":"<<pr.p<<",\"injected_state_time\":0,\"memory_time\":0,\"reset_every_step\":false,\"moment_resets\":0,\"slope_strength\":"<<strength
      <<",\"ordinary_source_faces\":\""<<source_faces<<"\",\"slope_penalty_seconds\":"<<pr.slope_seconds<<",\"slope_penalty_faces\":"<<pr.slope_faces<<",\"slope_projection_cells\":"<<pr.slope_projection_cells<<",\"maximum_slope_dissipation\":"<<pr.max_slope_dissipation<<",\"maximum_slope_identity_error\":"<<pr.max_slope_power_error;
    BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

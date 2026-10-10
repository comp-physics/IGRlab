// Test-only mixed polynomial spaces and retained local moment memory.
// All stages use the shared weak DG, full-energy flux, LLF faces and current
// quotient source. This is not an element-local ADER method or a lagged closure.
#define DG_IGR_DG_RECOVERY_LIBRARY_ONLY
#include "igr_dg_recovery_evolution_probe.cpp"
#undef DG_IGR_DG_RECOVERY_LIBRARY_ONLY

namespace
{
struct MemoryOptions
{
  RecoveryOptions recovery;
  int retained_order=-1,stress_order=-1;
  double start_time=0.,memory_time=0.;
  bool reset_step=false;
  std::string input;
};
RecoveryOptions SpaceOptions(const MemoryOptions& o,int degree,bool high=false)
{
  auto r=o.recovery;auto& t=r.boundary.compatibility.track;
  const int original=t.base.order;
  if(!high)t.quad_extra+=2*(original-degree); // Same physical Gauss rule.
  t.base.order=degree;
  if(high)t.sipg_penalty*=double((o.stress_order+1)*(o.stress_order+1))/double((degree+1)*(degree+1));
  return r;
}
struct LocalTransfer
{
  mfem::DenseMatrix E,P;
  LocalTransfer(const RecoveryProblem& high,const RecoveryProblem& low):E(high.n,low.n),P(low.n,high.n)
  {
    for(int i=0;i<high.n;++i){const State s=low.Shape(high.physics.basis.nodes[i],false);for(int j=0;j<low.n;++j)E(i,j)=s[j];}
    P=0.;
    for(int i=0;i<low.n;++i)for(int j=0;j<high.n;++j)
      for(int k=0;k<low.n;++k)for(int l=0;l<high.n;++l)
        P(i,j)+=low.mass_inverse(i,k)*E(l,k)*high.mass(l,j);
  }
  State Apply(const mfem::DenseMatrix& a,const State& u,int ne,int components)const
  {
    State v(ne*a.Height()*components,0.);
    for(int e=0;e<ne;++e)for(int i=0;i<a.Height();++i)for(int j=0;j<a.Width();++j)for(int c=0;c<components;++c)
      v[(e*a.Height()+i)*components+c]+=a(i,j)*u[(e*a.Width()+j)*components+c];
    return v;
  }
};
struct MemoryProblem:RecoveryProblem
{
  MemoryOptions memory;
  RecoveryProblem retained,stress;
  LocalTransfer state_transfer,stress_transfer;
  mutable double transfer_seconds=0.,galerkin_seconds=0.,max_transfer_mean_defect=0.;
  mutable std::uint64_t resets=0,galerkin_products=0;
  explicit MemoryProblem(const MemoryOptions& o):RecoveryProblem(SpaceOptions(o,o.recovery.boundary.compatibility.track.base.order,true)),
      memory(o),retained(SpaceOptions(o,o.retained_order)),stress(SpaceOptions(o,o.stress_order)),
      state_transfer(*this,retained),stress_transfer(*this,stress)
  {
    if(o.retained_order<1||o.stress_order<1||o.retained_order>p||o.stress_order>p||
       o.recovery.recovery!="existing"||o.recovery.volume_projection!="weak"||
       o.recovery.boundary.compatibility.formulation!="baseline")throw std::runtime_error("memory screen requires nested degrees and existing weak quotient formulation");
    if(o.memory_time<0.||o.start_time<0.||o.start_time>this->o.base.final_time||
       (o.reset_step&&o.memory_time>0.))throw std::runtime_error("invalid memory cadence/time");
    if(o.start_time>0.&&(o.input.empty()||boundary.scenario!="application-tanh"))throw std::runtime_error("shifted input requires autonomous application-tanh");
  }
  State Initial()const
  {
    const auto start=Clock::now();State low=retained.Initial();
    if(!memory.input.empty())
    {
      std::ifstream in(memory.input);for(double& x:low)if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid retained-space input");
      double extra;if(in>>extra)throw std::runtime_error("extra retained-space input entries");
    }
    State out=state_transfer.Apply(state_transfer.E,low,ne,4);transfer_seconds+=Wall(start);return out;
  }
  void Reset(State& u,TrackWork& work)const
  {
    const auto start=Clock::now();const State before=u;
    State low=state_transfer.Apply(state_transfer.P,u,ne,4);
    u=state_transfer.Apply(state_transfer.E,low,ne,4);
    for(int e=0;e<ne;++e)for(int c=0;c<4;++c)
    {
      double delta=0.;for(int i=0;i<n;++i)delta+=physics.basis.weights[i]*(u[Id(e,i,c)]-before[Id(e,i,c)]);
      max_transfer_mean_defect=std::max(max_transfer_mean_defect,std::abs(delta));
    }
    ++resets;transfer_seconds+=Wall(start);Limit(u,work);
  }
  virtual void MixedSystem(const State& u,double time,TrackWork& work,BandMatrix& a,State& b)const
  {
    BandMatrix ah;State bh;AssembleClosure(u,time,work,ah,bh);
    const auto start=Clock::now();const int m=stress.n;
    // A full adjacent block bandwidth safely includes every Galerkin entry.
    a=BandMatrix(ne*m,2*m-1);b.assign(ne*m,0.);const auto& E=stress_transfer.E;
    for(int e=0;e<ne;++e)
    {
      for(int i=0;i<m;++i)for(int j=0;j<n;++j)b[e*m+i]+=E(j,i)*bh[e*n+j];
      for(int f=std::max(0,e-1);f<=std::min(ne-1,e+1);++f)
      {
        mfem::DenseMatrix temp(n,m);temp=0.;
        for(int i=0;i<n;++i)for(int k=0;k<n;++k)for(int j=0;j<m;++j)
          temp(i,j)+=ah.Get(e*n+i,f*n+k)*E(k,j);
        for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<n;++k)
          a.At(e*m+i,f*m+j)+=E(k,i)*temp(k,j);
        galerkin_products+=std::uint64_t(n)*m*(n+m);
      }
    }
    galerkin_seconds+=Wall(start);
  }
  State Closure(const State& u,double time,TrackWork& work)const
  {
    if(stress.p==p)return RecoveryProblem::Closure(u,time,work);
    const auto start=Clock::now();BandMatrix a;State b;MixedSystem(u,time,work,a,b);
    BandFactor factor(a,0,a.n);State s=b;factor.Solve(s,1);State residual(a.n,0.);
    for(int i=0;i<a.n;++i){residual[i]=-b[i];for(int j=std::max(0,i-a.k);j<std::min(a.n,i+a.k+1);++j)residual[i]+=a.Get(i,j)*s[j];}
    const double relative=Norm(residual)/std::max(Norm(b),1.e-12);
    if(!std::isfinite(relative)||relative>1.e-8)throw std::runtime_error("mixed Galerkin closure residual failed");
    work.max_closure_residual=std::max(work.max_closure_residual,relative);
    State sh=stress_transfer.Apply(stress_transfer.E,s,ne,1);double reaction=0.;
    for(int e=0;e<ne;++e)for(std::size_t q=0;q<weights.size();++q)
    {
      double value=0.;for(int i=0;i<n;++i)value+=shapes[q][i]*sh[e*n+i];
      reaction+=(edges[e+1]-edges[e])*weights[q]*value/Point(u,e,shapes[q])[0];
    }
    max_mean_defect=std::max(max_mean_defect,std::abs(reaction-work.current_source_integral));
    ++work.closure_calls;work.closure+=Wall(start);return sh;
  }
};
void MemorySetFixedEdges(MemoryProblem& pr,const State& edges)
{
  if(edges.size()!=std::size_t(pr.ne+1)||edges.front()!=pr.o.left||edges.back()!=pr.o.right)
    throw std::runtime_error("fixed edges must preserve the declared element count and physical domain");
  double hmin=INFINITY;
  for(int e=0;e<pr.ne;++e)
  {
    const double h=edges[e+1]-edges[e];
    if(!std::isfinite(edges[e])||!std::isfinite(edges[e+1])||!(h>0.))
      throw std::runtime_error("fixed edges must be finite and strictly increasing");
    hmin=std::min(hmin,h);
  }
  // All transfers and mass matrices live on the affine reference element.
  // Initial data, derivatives, fluxes and closure coefficients must use the
  // same physical map in every space. The analytic initial center is unchanged.
  for(TrackingProblem* space:{static_cast<TrackingProblem*>(&pr),
      static_cast<TrackingProblem*>(&pr.retained),static_cast<TrackingProblem*>(&pr.stress)})
  {
    if(space->ne!=pr.ne)throw std::runtime_error("fixed edges space mismatch");
    space->edges=edges;space->base_uniform_edges=edges;
    space->initial_graded_hmin=hmin;space->o.mode="saved-fixed";
  }
}
template<class ProblemType>
TrackResult MemoryRun(ProblemType& pr)
{
  TrackResult out;const auto start=Clock::now();out.time=pr.memory.start_time;
  double next_reset=out.time+pr.memory.memory_time,accepted_source_integral=0.;
  try
  {
    State initial=pr.Initial();pr.Limit(initial,out.work);State initial_sigma=pr.Closure(initial,out.time,out.work);
    out.u=std::move(initial);out.sigma=std::move(initial_sigma);out.initial_integrals=pr.Integrals(out.u,pr.edges);
    accepted_source_integral=out.work.current_source_integral;
    while(out.time<pr.o.base.final_time-1.e-14)
    {
      ++out.work.attempted;double dt=std::min(pr.o.base.dt,pr.o.base.final_time-out.time);
      if(pr.memory.memory_time>0.)dt=std::min(dt,next_reset-out.time);
      const State original=out.u;State stage=original;std::array<std::array<double,4>,3> budgets{};
      const double times[3]={out.time,out.time+dt,out.time+.5*dt};
      for(int k=0;k<3;++k)
      {
        const State sigma=k?pr.Closure(stage,times[k],out.work):out.sigma;
        const State rhs=pr.Residual(stage,sigma,times[k],out.work,budgets[k]);
        for(std::size_t j=0;j<stage.size();++j){const double candidate=stage[j]+dt*rhs[j];stage[j]=k==0?candidate:(k==1?.75*original[j]+.25*candidate:original[j]/3.+2.*candidate/3.);}
        pr.Limit(stage,out.work);
      }
      const double candidate_time=out.time+dt;
      const bool reset=pr.memory.reset_step||(pr.memory.memory_time>0.&&candidate_time>=next_reset-1.e-13);
      if(reset)pr.Reset(stage,out.work);
      State candidate_sigma=pr.Closure(stage,candidate_time,out.work);
      // Preserve the last accepted state/stress/time together on rejection.
      out.u=std::move(stage);out.sigma=std::move(candidate_sigma);out.time=candidate_time;++out.work.steps;
      accepted_source_integral=out.work.current_source_integral;
      for(int c=0;c<4;++c)out.boundary_budget[c]+=dt*(budgets[0][c]/6.+budgets[1][c]/6.+2.*budgets[2][c]/3.);
      if(reset&&pr.memory.memory_time>0.)next_reset+=pr.memory.memory_time;
    }
  }catch(const std::exception& error){out.failure=error.what();out.work.current_source_integral=accepted_source_integral;}
  if(!out.u.empty()){const auto totals=pr.Integrals(out.u,pr.edges);for(int c=0;c<4;++c)out.conservation_defect[c]=totals[c]-out.initial_integrals[c]-out.boundary_budget[c];}
  out.wall=Wall(start);return out;
}
void MemorySelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;
  t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=7;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";
  t.quad_extra=25;mo.retained_order=5;mo.stress_order=5;MemoryProblem pr(mo);
  State lo=pr.retained.Initial();for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.retained.n;++i)
  {lo[pr.retained.Id(e,i,0)]+=.01*std::sin(1.7*e+.9*i);lo[pr.retained.Id(e,i,1)]+=.007*std::cos(.7*e+1.3*i);}
  State hi=pr.state_transfer.Apply(pr.state_transfer.E,lo,pr.ne,4),back=pr.state_transfer.Apply(pr.state_transfer.P,hi,pr.ne,4);
  double transfer=0.;for(std::size_t i=0;i<lo.size();++i)transfer=std::max(transfer,std::abs(lo[i]-back[i]));
  TrackWork work;BandMatrix a,b;State al,bl;pr.MixedSystem(hi,0.,work,a,al);pr.stress.AssembleClosure(lo,0.,work,b,bl);
  double matrix=0.,load=0.;for(int i=0;i<a.n;++i){load=std::max(load,std::abs(al[i]-bl[i]));for(int j=0;j<a.n;++j)matrix=std::max(matrix,std::abs(a.Get(i,j)-b.Get(i,j)));}
  const State sh=pr.Closure(hi,0.,work),sl=pr.retained.Closure(lo,0.,work);std::array<double,4> bh{},bc{};
  State rh=pr.Residual(hi,sh,0.,work,bh),rc=pr.retained.Residual(lo,sl,0.,work,bc);
  back=pr.state_transfer.Apply(pr.state_transfer.P,rh,pr.ne,4);double rhs=0.;for(std::size_t i=0;i<back.size();++i)rhs=std::max(rhs,std::abs(back[i]-rc[i]));
  // Exercise actual mode removal, including separate means in every cell.
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {hi[pr.Id(e,i,0)]+=.001*std::sin(1.3*i+.2*e);hi[pr.Id(e,i,3)]+=.002*std::cos(1.7*i+.3*e);}
  pr.Reset(hi,work);
  if(transfer>1.e-11||matrix>1.e-9||load>1.e-10||rhs>2.e-9||pr.max_transfer_mean_defect>1.e-11||pr.max_mean_defect>1.e-9)
    throw std::runtime_error("nested-space transfer/Galerkin/weak residual parity failed");
  mo.retained_order=7;mo.stress_order=7;t.base.final_time=2.e-5;t.base.dt=1.e-5;MemoryProblem equal(mo);RecoveryProblem old(SpaceOptions(mo,7));
  const auto x=MemoryRun(equal),y=CompatibilityRun(old);double step=0.;for(std::size_t i=0;i<x.u.size();++i)step=std::max(step,std::abs(x.u[i]-y.u[i]));
  if(!x.failure.empty()||!y.failure.empty()||step>1.e-12)throw std::runtime_error("equal-space evolution parity failed");
  MemoryProblem supplied(mo);MemorySetFixedEdges(supplied,equal.edges);
  const auto z=MemoryRun(supplied);double supplied_step=0.;
  for(std::size_t i=0;i<x.u.size();++i)supplied_step=std::max(supplied_step,std::abs(x.u[i]-z.u[i]));
  if(!z.failure.empty()||supplied_step>1.e-12)throw std::runtime_error("supplied uniform geometry evolution parity failed");
  const double fractions[9]={0.,.04,.09,.18,.35,.57,.75,.92,1.};State graded(9);
  for(int j=0;j<=8;++j)graded[j]=t.left+(t.right-t.left)*fractions[j];
  MemorySetFixedEdges(supplied,graded);State affine(supplied.ne*supplied.n*4);
  const double density=2.,gradient=.1,pressure=1.,expected_sigma=2.*t.base.alpha*density*gradient*gradient;
  for(int e=0;e<supplied.ne;++e)for(int i=0;i<supplied.n;++i)
  {
    const double coordinate=graded[e]+(graded[e+1]-graded[e])*supplied.physics.basis.nodes[i];
    const double velocity=.2+gradient*coordinate;
    affine[supplied.Id(e,i,0)]=density;affine[supplied.Id(e,i,1)]=density*velocity;
    affine[supplied.Id(e,i,2)]=0.;affine[supplied.Id(e,i,3)]=pressure/(gamma_euler-1.)+.5*density*velocity*velocity;
  }
  TrackWork wg;const State sg=supplied.Closure(affine,0.,wg);std::array<double,4> bg{};
  const State rg=supplied.Residual(affine,sg,0.,wg,bg);double graded_closure=0.,graded_transport=0.;
  for(double value:sg)graded_closure=std::max(graded_closure,std::abs(value-expected_sigma));
  for(int e=1;e+1<supplied.ne;++e)for(int i=0;i<supplied.n;++i)
  {
    const double velocity=affine[supplied.Id(e,i,1)]/density;
    const std::array<double,4> exact{{-density*gradient,-2.*density*velocity*gradient,0.,
      -gradient*(gamma_euler*pressure/(gamma_euler-1.)+expected_sigma+1.5*density*velocity*velocity)}};
    for(int c=0;c<4;++c)graded_transport=std::max(graded_transport,std::abs(rg[supplied.Id(e,i,c)]-exact[c]));
  }
  bool rejected=false;State invalid=graded;invalid[3]=invalid[2];
  try{MemorySetFixedEdges(supplied,invalid);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected||graded_closure>1.e-10||graded_transport>2.e-9)
    throw std::runtime_error("supplied nonuniform geometry analytic closure/transport check failed");
  std::cout<<"moment memory checks passed: transfer="<<transfer<<" matrix="<<matrix<<" load="<<load<<" rhs="<<rhs<<" equal-step="<<step
           <<" supplied-step="<<supplied_step<<" graded-closure="<<graded_closure<<" graded-transport="<<graded_transport<<'\n';
}
}
#ifndef DG_IGR_MOMENT_MEMORY_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    MemoryOptions o;std::string edges_input;std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--reset-every-step"){o.reset_step=true;continue;}
      if(flag=="--edges-in")
      {if(++i==argc)throw std::runtime_error("missing fixed edges input");edges_input=argv[i];continue;}
      if(flag=="--retained-order"||flag=="--stress-order"||flag=="--state-time"||flag=="--memory-time"||flag=="--state-in")
      {if(++i==argc)throw std::runtime_error("missing memory option");if(flag=="--retained-order")o.retained_order=std::stoi(argv[i]);else if(flag=="--stress-order")o.stress_order=std::stoi(argv[i]);else if(flag=="--state-time")o.start_time=std::stod(argv[i]);else if(flag=="--memory-time")o.memory_time=std::stod(argv[i]);else o.input=argv[i];}
      else common.push_back(argv[i]);
    }
    o.recovery=RecoveryParse(int(common.size()),common.data());
    if(o.recovery.boundary.compatibility.track.self_test){MemorySelfTest();return 0;}
    const int p=o.recovery.boundary.compatibility.track.base.order;if(o.retained_order<0)o.retained_order=p;if(o.stress_order<0)o.stress_order=p;
    if(o.retained_order<1||o.stress_order<1||o.retained_order>p||o.stress_order>p||
       !std::isfinite(o.start_time)||!std::isfinite(o.memory_time))throw std::runtime_error("invalid nested degree or time");
    const auto start=Clock::now();MemoryProblem pr(o);
    if(!edges_input.empty())
    {
      if(pr.o.mode!="uniform")throw std::runtime_error("supplied fixed edges require --mode uniform");
      std::ifstream in(edges_input);State edges(pr.ne+1);
      for(double& value:edges)if(!(in>>value))throw std::runtime_error("invalid fixed edges input");
      double extra;if(in>>extra)throw std::runtime_error("extra fixed edges input entries");
      MemorySetFixedEdges(pr,edges);
    }
    const double setup=Wall(start);auto result=MemoryRun(pr);result.work.setup=setup;
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"existing\",\"volume_projection\":\"weak\",\"retained_order\":"<<o.retained_order
      <<",\"stress_order\":"<<o.stress_order<<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"sigma_serialized_order\":"<<pr.p
      <<",\"injected_state_time\":"<<o.start_time<<",\"memory_time\":"<<o.memory_time<<",\"reset_every_step\":"<<(o.reset_step?"true":"false")
      <<",\"moment_resets\":"<<pr.resets<<",\"moment_transfer_seconds\":"<<pr.transfer_seconds<<",\"galerkin_seconds\":"<<pr.galerkin_seconds
      <<",\"galerkin_products\":"<<pr.galerkin_products<<",\"maximum_moment_mean_defect\":"<<pr.max_transfer_mean_defect;
    BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif

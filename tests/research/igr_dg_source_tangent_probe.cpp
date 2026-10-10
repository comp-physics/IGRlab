// Test-only finite-difference tangent of the existing high-order weak-DG RHS.
// A frozen instantaneous Jacobian is a diagnostic, not a stability theorem.
#define DG_IGR_DG_RECOVERY_LIBRARY_ONLY
#include "igr_dg_recovery_evolution_probe.cpp"
#undef DG_IGR_DG_RECOVERY_LIBRARY_ONLY

namespace
{
struct InjectedProblem:RecoveryProblem
{
  State injected;
  explicit InjectedProblem(const RecoveryOptions& o):RecoveryProblem(o){}
  State Initial()const{return injected.empty()?RecoveryProblem::Initial():injected;}
};
State TangentRHS(const RecoveryProblem& pr,const State& u,const State* frozen,double time)
{
  TrackWork work;std::array<double,4> budget{};
  const State sigma=frozen?*frozen:pr.Closure(u,time,work);
  return pr.Residual(u,sigma,time,work,budget);
}
State SplitRHS(const RecoveryProblem& pr,const State& u,const BandFactor& base_factor,
               const State& base_load,bool source_only,double time)
{
  TrackWork work;std::array<double,4> budget{};BandMatrix a;State b;
  pr.AssembleClosure(u,time,work,a,b);
  if(source_only)base_factor.Solve(b,1);
  else{b=base_load;BandFactor current(a,0,pr.ns);current.Solve(b,1);}
  return pr.Residual(u,b,time,work,budget);
}
void TangentSelfTest()
{
  RecoveryOptions o;o.recovery="existing";
  auto& t=o.boundary.compatibility.track;
  t.initial="entropy";t.base.elements=8;t.base.order=5;t.base.alpha=.04;
  t.closure_space="l2";t.closure_boundary="natural";
  RecoveryProblem pr(o);State u=pr.Initial(),direction(u.size(),0.);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {
    const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];
    u[pr.Id(e,i,0)]=2.;u[pr.Id(e,i,1)]=2.*(.2+.1*x);
    u[pr.Id(e,i,2)]=0.;u[pr.Id(e,i,3)]=4.;direction[pr.Id(e,i,1)]=2.*x;
  }
  const double eps=1.e-5;State plus=u,minus=u;
  for(std::size_t j=0;j<u.size();++j){plus[j]+=eps*direction[j];minus[j]-=eps*direction[j];}
  TrackWork work;const auto a=pr.Closure(plus,0.,work),b=pr.Closure(minus,0.,work);
  double error=0.;for(int j=0;j<pr.ns;++j)
    error=std::max(error,std::abs((a[j]-b[j])/(2*eps)-4.*t.base.alpha*2.*.1));
  if(error>2.e-8)throw std::runtime_error("affine closure tangent failed");
  std::cout<<"DG source tangent affine closure check passed, error="<<error<<'\n';
}
}

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    std::string input;double epsilon=1.e-5,time=0.;bool advance=false,split=false,residual_only=false;std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];
      if(flag=="--advance-input"){advance=true;continue;}
      if(flag=="--split-closure-feedback"){split=true;continue;}
      if(flag=="--residual-only"){residual_only=true;continue;}
      if(flag=="--state-in"||flag=="--tangent-epsilon"||flag=="--state-time")
      {
        if(++i==argc)throw std::runtime_error("missing tangent option");
        if(flag=="--state-in")input=argv[i];
        else if(flag=="--tangent-epsilon")epsilon=std::stod(argv[i]);else time=std::stod(argv[i]);
      }
      else common.push_back(argv[i]);
    }
    const auto o=RecoveryParse(int(common.size()),common.data());
    if(o.boundary.compatibility.track.self_test){TangentSelfTest();return 0;}
    if(!(epsilon>0.)||!std::isfinite(epsilon)||!std::isfinite(time))throw std::runtime_error("invalid tangent scale/time");
    if(residual_only&&(advance||split))throw std::runtime_error("residual-only is a separate endpoint diagnostic");
    if(split&&(advance||o.recovery!="existing"||o.boundary.compatibility.formulation!="baseline"||
       o.boundary.open_closure!="natural"||o.boundary.wall_closure!="natural"))
      throw std::runtime_error("split feedback requires baseline natural SIPG diagnostic");
    const auto start=Clock::now();InjectedProblem pr(o);State u=pr.Initial();
    if(!input.empty())
    {
      std::ifstream in(input);for(double& v:u)if(!(in>>v)||!std::isfinite(v))throw std::runtime_error("invalid tangent state");
      double extra;if(in>>extra)throw std::runtime_error("extra tangent state entries");
    }
    if(advance)
    {
      // Only this autonomous IVP is qualified for a shifted starting time.
      if(input.empty()||pr.boundary.scenario!="application-tanh"||
         pr.boundary.left!="nonreflecting"||pr.boundary.right!="fixed")
        throw std::runtime_error("injected advancement requires autonomous application-tanh boundary data");
      pr.injected=u;auto result=CompatibilityRun(pr);
      BoundaryWrite(pr,result,",\"injected_state_time\":"+std::to_string(time)+
          ",\"source_recovery\":\""+o.recovery+"\",\"source_recovery_faces\":\""+o.faces+
          "\",\"source_product\":\""+o.source_product+"\",\"recovery_operator\":\""+o.closure_operator+
          "\",\"volume_projection\":\""+o.volume_projection+"\"\n");
      return result.failure.empty()?0:2;
    }
    if(residual_only)
    {
      TrackWork work;std::array<double,4> budget{};
      const State sigma=pr.Closure(u,time,work),rhs=pr.Residual(u,sigma,time,work,budget);
      const auto left=pr.BoundaryFlux(u,sigma,0),right=pr.BoundaryFlux(u,sigma,1);
      std::ofstream out(pr.o.output);if(!out)throw std::runtime_error("cannot open residual output");
      out<<std::setprecision(17)<<"{\"order\":"<<pr.p<<",\"quad_extra\":"<<pr.o.quad_extra<<",\"time\":"<<time<<",\"alpha\":"<<pr.o.base.alpha
         <<",\"volume_projection\":\""<<o.volume_projection<<"\",\"source_recovery\":\""<<o.recovery
         <<"\",\"source_recovery_faces\":\""<<o.faces<<"\",\"source_product\":\""<<o.source_product
         <<"\",\"recovery_operator\":\""<<o.closure_operator<<"\",\"edges\":";
      TrackArray(out,pr.edges);out<<",\"base_state\":";TrackArray(out,u);
      out<<",\"base_rhs\":";TrackArray(out,rhs);out<<",\"sigma\":";TrackArray(out,sigma);
      out<<",\"boundary_flux_left\":";TrackArray(out,State(left.begin(),left.end()));
      out<<",\"boundary_flux_right\":";TrackArray(out,State(right.begin(),right.end()));
      out<<",\"max_rhs_defect\":"<<work.max_rhs_defect<<",\"max_closure_residual\":"<<work.max_closure_residual<<"}\n";
      return 0;
    }
    // Retain rho, x momentum and full energy, excluding the decoupled planar
    // transverse mode. Column perturbations are scaled in each component.
    std::vector<int> ids;State scale(3,1.);const int components[3]={0,1,3};
    for(int k=0;k<3;++k)for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
      scale[k]=std::max(scale[k],std::abs(u[pr.Id(e,i,components[k])]));
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int c:components)ids.push_back(pr.Id(e,i,c));
    TrackWork work;const State sigma=pr.Closure(u,time,work);const int size=int(ids.size());
    BandMatrix base_matrix;State base_load;
    if(split)pr.AssembleClosure(u,time,work,base_matrix,base_load);
    std::unique_ptr<BandFactor> base_factor;
    if(split)base_factor=std::make_unique<BandFactor>(base_matrix,0,pr.ns);
    std::ofstream out(pr.o.output);if(!out)throw std::runtime_error("cannot open tangent output");
    out<<std::setprecision(17)<<"{\"size\":"<<size<<",\"epsilon\":"<<epsilon<<",\"time\":"<<time
       <<",\"scale\":["<<scale[0]<<','<<scale[1]<<','<<scale[2]<<"],\"sigma\":[";
    for(int j=0;j<pr.ns;++j){if(j)out<<',';out<<sigma[j];}out<<"],\"columns\":[";
    for(int j=0;j<size;++j)
    {
      const double step=epsilon*scale[j%3];State plus=u,minus=u;plus[ids[j]]+=step;minus[ids[j]]-=step;
      const auto ap=TangentRHS(pr,plus,nullptr,time),am=TangentRHS(pr,minus,nullptr,time);
      const auto fp=TangentRHS(pr,plus,&sigma,time),fm=TangentRHS(pr,minus,&sigma,time);
      State sp,sm,cp,cm;
      if(split)
      {
        sp=SplitRHS(pr,plus,*base_factor,base_load,true,time);sm=SplitRHS(pr,minus,*base_factor,base_load,true,time);
        cp=SplitRHS(pr,plus,*base_factor,base_load,false,time);cm=SplitRHS(pr,minus,*base_factor,base_load,false,time);
      }
      if(j)out<<',';out<<'[';
      for(int k=0;k<(split?4:2);++k)
      {
        if(k)out<<',';out<<'[';
        for(int i=0;i<size;++i)
        {
          const State& plus_rhs=k==0?ap:(k==1?fp:(k==2?sp:cp));
          const State& minus_rhs=k==0?am:(k==1?fm:(k==2?sm:cm));
          const double value=(plus_rhs[ids[i]]-minus_rhs[ids[i]])/(2*step);
          if(!std::isfinite(value))throw std::runtime_error("nonfinite tangent");
          if(i)out<<',';out<<value;
        }
        out<<']';
      }
      out<<']';
    }
    out<<"],\"complete_probe_seconds\":"<<Wall(start)<<"}\n";
    std::cout<<"tangent size="<<size<<" elapsed="<<Wall(start)<<'\n';return 0;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

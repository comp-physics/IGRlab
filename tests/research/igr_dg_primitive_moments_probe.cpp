// Test-only nonlinear conservative trial manifold. The evolved unknowns are
// degree-p conservative moments, while density, velocity and thermal pressure
// are degree p in the reconstructed physical state. Shared full-energy weak DG
// and instantaneous SIPG are evaluated through a redundant degree-3p embedding.
#define DG_IGR_MOMENT_MEMORY_LIBRARY_ONLY
#include "igr_dg_moment_memory_probe.cpp"
#undef DG_IGR_MOMENT_MEMORY_LIBRARY_ONLY

namespace
{
struct PrimitiveMomentProblem:MemoryProblem
{
  std::vector<State> low_shapes;
  mutable double reconstruction_seconds=0.;
  mutable std::uint64_t local_velocity_factors=0,reconstruction_cells=0;
  mutable State cached_moments,cached_physical;
  bool central_source=false;
  explicit PrimitiveMomentProblem(const MemoryOptions& o,bool central=false):MemoryProblem(o),central_source(central)
  {
    if(p!=3*retained.p||stress.p!=retained.p||o.reset_step||o.memory_time!=0.)
      throw std::runtime_error("primitive moments require high degree 3p, stress degree p, persistent conservative moments");
    if(central&&(boundary.left=="wall"||boundary.right=="wall"))throw std::runtime_error("primitive central source wall trace is not qualified");
    for(std::size_t q=0;q<weights.size();++q)
    {double x=0.;for(int i=0;i<n;++i)x+=shapes[q][i]*physics.basis.nodes[i];low_shapes.push_back(retained.Shape(x,false));}
  }
  State Cell(const State& low,int e)const
  {
    const auto start=Clock::now();const int m=retained.n;BandMatrix weighted(m,m-1);State velocity(2*m,0.),kinetic_load(m,0.);
    for(std::size_t q=0;q<weights.size();++q)
    {
      const auto& b=low_shapes[q];double rho=0.;for(int i=0;i<m;++i)rho+=b[i]*low[(e*m+i)*4];
      if(!(rho>floor_value))throw std::runtime_error("primitive reconstruction density admission failed");
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)weighted.At(i,j)+=weights[q]*rho*b[i]*b[j];
    }
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<2;++c)
      velocity[c*m+i]+=retained.mass(i,j)*low[(e*m+j)*4+c+1];
    BandFactor factor(weighted,0,m);factor.Solve(velocity,2);++local_velocity_factors;
    for(std::size_t q=0;q<weights.size();++q)
    {
      const auto& b=low_shapes[q];double rho=0.,vx=0.,vy=0.;for(int i=0;i<m;++i)
      {rho+=b[i]*low[(e*m+i)*4];vx+=b[i]*velocity[i];vy+=b[i]*velocity[m+i];}
      for(int i=0;i<m;++i)kinetic_load[i]+=weights[q]*b[i]*.5*rho*(vx*vx+vy*vy);
    }
    State kinetic(m,0.);for(int i=0;i<m;++i)for(int j=0;j<m;++j)kinetic[i]+=retained.mass_inverse(i,j)*kinetic_load[j];
    State out(n*4,0.);for(int i=0;i<n;++i)
    {
      double rho=0.,vx=0.,vy=0.,energy=0.,k=0.;for(int j=0;j<m;++j)
      {const double b=state_transfer.E(i,j);rho+=b*low[(e*m+j)*4];vx+=b*velocity[j];vy+=b*velocity[m+j];energy+=b*low[(e*m+j)*4+3];k+=b*kinetic[j];}
      out[4*i]=rho;out[4*i+1]=rho*vx;out[4*i+2]=rho*vy;out[4*i+3]=energy+.5*rho*(vx*vx+vy*vy)-k;
    }
    ++reconstruction_cells;reconstruction_seconds+=Wall(start);return out;
  }
  const State& Reconstruct(const State& moments)const
  {
    if(moments==cached_moments)return cached_physical;
    const State low=state_transfer.Apply(state_transfer.P,moments,ne,4);State physical(moments.size());
    for(int e=0;e<ne;++e){const State cell=Cell(low,e);std::copy(cell.begin(),cell.end(),physical.begin()+e*n*4);}
    cached_moments=moments;cached_physical=std::move(physical);return cached_physical;
  }
  void Limit(State& moments,TrackWork& work)const
  {
    const auto start=Clock::now();State low=state_transfer.Apply(state_transfer.P,moments,ne,4);const int m=retained.n;
    for(int e=0;e<ne;++e)
    {
      State original(m*4);std::array<double,4> mean{};
      for(int i=0;i<m;++i)for(int c=0;c<4;++c)
      {original[4*i+c]=low[(e*m+i)*4+c];mean[c]+=retained.physics.basis.weights[i]*original[4*i+c];}
      if(!TrackAdmitted(mean))throw std::runtime_error("primitive moment mean admission failed");
      bool accepted=false;double theta=1.;
      for(int attempt=0;attempt<49;++attempt,theta*=.5)
      {
        for(int i=0;i<m;++i)for(int c=0;c<4;++c)low[(e*m+i)*4+c]=mean[c]+theta*(original[4*i+c]-mean[c]);
        State physical;bool admitted=true;
        try{physical=Cell(low,e);}catch(const std::exception&){admitted=false;}
        if(admitted)for(const auto& b:positive)
        {
          std::array<double,4> point{};for(int i=0;i<n;++i)for(int c=0;c<4;++c)point[c]+=b[i]*physical[4*i+c];
          ++work.positivity_points;if(!TrackAdmitted(point)){admitted=false;break;}
        }
        if(admitted){accepted=true;if(attempt){++work.limited;work.min_theta=std::min(work.min_theta,theta);}break;}
        if(!o.positivity)throw std::runtime_error("primitive physical polynomial admission failed");
      }
      if(!accepted)throw std::runtime_error("primitive reconstruction limiting failed");
    }
    moments=state_transfer.Apply(state_transfer.E,low,ne,4);cached_moments.clear();work.limiter+=Wall(start);
  }
  State Closure(const State& moments,double time,TrackWork& work)const
  {return MemoryProblem::Closure(Reconstruct(moments),time,work);}
  void MixedSystem(const State& physical,double time,TrackWork& work,BandMatrix& a,State& b)const override
  {
    MemoryProblem::MixedSystem(physical,time,work,a,b);if(!central_source)return;
    // The auxiliary velocity lives in P_p, so its central lift uses the P_p
    // mass. Using the degree-3p serialization mass would define another method.
    // The redundant original source construction above remains charged.
    std::fill(b.begin(),b.end(),0.);double integral=0.;const int m=retained.n;
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];
      for(std::size_t q=0;q<weights.size();++q)
      {
        const auto u=Point(physical,e,shapes[q]);double dr=0.,dm=0.;
        for(int i=0;i<n;++i){dr+=derivatives[q][i]*physical[Id(e,i,0)]/h;dm+=derivatives[q][i]*physical[Id(e,i,1)]/h;}
        double g=(dm-u[1]/u[0]*dr)/u[0];
        for(int side=0;side<2;++side)
        {
          const int neighbor=e+(side?1:-1);if(neighbor<0||neighbor>=ne)continue;
          const int endpoint=side?p:0,other=side?0:p,low_endpoint=side?retained.p:0;
          const double own=physical[Id(e,endpoint,1)]/physical[Id(e,endpoint,0)];
          const double outside=physical[Id(neighbor,other,1)]/physical[Id(neighbor,other,0)];
          double lift=0.;for(int i=0;i<m;++i)lift+=low_shapes[q][i]*retained.mass_inverse(i,low_endpoint)/h;
          g+=(side?-1.:1.)*.5*(own-outside)*lift;
        }
        const double value=2.*o.base.alpha*g*g,weight=h*weights[q];integral+=weight*value;
        for(int i=0;i<m;++i)b[e*m+i]+=weight*low_shapes[q][i]*value;
        ++work.source_points;
      }
    }
    work.current_source_integral=integral;
  }
  State Residual(const State& moments,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    const State high=RecoveryProblem::Residual(Reconstruct(moments),sigma,time,work,budget);
    const auto start=Clock::now();const State low=state_transfer.Apply(state_transfer.P,high,ne,4);
    State out=state_transfer.Apply(state_transfer.E,low,ne,4);transfer_seconds+=Wall(start);return out;
  }
};
void PrimitiveSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;
  t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=9;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=19;
  mo.retained_order=mo.stress_order=3;PrimitiveMomentProblem pr(mo);State physical(pr.ne*pr.n*4),boosted(physical.size());
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {
    const double z=pr.physics.basis.nodes[i],rho=1.1+.04*z+.01*z*z*z,v=.2+.02*z*z*z+.01*e,vy=-.03+.015*z*z*z,p=1.+.01*z*z*z;
    physical[pr.Id(e,i,0)]=rho;physical[pr.Id(e,i,1)]=rho*v;physical[pr.Id(e,i,2)]=rho*vy;physical[pr.Id(e,i,3)]=p/(gamma_euler-1.)+.5*rho*(v*v+vy*vy);
  }
  auto embed=[&](const State& a){return pr.state_transfer.Apply(pr.state_transfer.E,pr.state_transfer.Apply(pr.state_transfer.P,a,pr.ne,4),pr.ne,4);};
  State moments=embed(physical),actual=pr.Reconstruct(moments);double patch=0.,moment_error=0.;
  for(std::size_t j=0;j<actual.size();++j)patch=std::max(patch,std::abs(actual[j]-physical[j]));
  const State check=embed(actual);for(std::size_t j=0;j<check.size();++j)moment_error=std::max(moment_error,std::abs(check[j]-moments[j]));
  double source_error=0.;
  for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
  {
    const auto u=pr.Point(actual,e,pr.shapes[q]);double du[8]={},z=0.;const double h=pr.edges[e+1]-pr.edges[e];
    for(int i=0;i<pr.n;++i){z+=pr.shapes[q][i]*pr.physics.basis.nodes[i];for(int c=0;c<4;++c)du[2*c]+=pr.derivatives[q][i]*actual[pr.Id(e,i,c)]/h;}
    const double source=igr::SourceFromConservativeStateGradient(2,u.data(),1,du,2,1,t.base.alpha,floor_value);
    source_error=std::max(source_error,std::abs(source-2*t.base.alpha*std::pow(.06*z*z/h,2)));
  }
  boosted=moments;const double boost=.7;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {boosted[pr.Id(e,i,3)]+=boost*moments[pr.Id(e,i,1)]+.5*boost*boost*moments[pr.Id(e,i,0)];boosted[pr.Id(e,i,1)]+=boost*moments[pr.Id(e,i,0)];}
  State transformed=pr.Reconstruct(boosted);double galilean=0.;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {
    const int j=pr.Id(e,i,0);physical[j+3]+=boost*physical[j+1]+.5*boost*boost*physical[j];physical[j+1]+=boost*physical[j];
    for(int c=0;c<4;++c)galilean=std::max(galilean,std::abs(transformed[j+c]-physical[j+c]));
  }
  TrackWork work;pr.Limit(moments,work);const State s=pr.Closure(moments,0.,work);std::array<double,4> budget{};
  const auto rhs=pr.Residual(moments,s,0.,work,budget);const auto totals=pr.Integrals(rhs,pr.edges);double conservation=0.;
  for(int c=0;c<4;++c)conservation=std::max(conservation,std::abs(totals[c]-budget[c]));
  if(patch>1.e-10||moment_error>1.e-10||galilean>1.e-10||source_error>1.e-10||conservation>1.e-9||work.limited)throw std::runtime_error("primitive manifold patch/moment/boost/source/conservation check failed");
  State bad=moments;for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)bad[pr.Id(e,i,3)]+=6.*(pr.physics.basis.nodes[i]-.5);
  const State before=pr.state_transfer.Apply(pr.state_transfer.P,bad,pr.ne,4);pr.Limit(bad,work);
  const State after=pr.state_transfer.Apply(pr.state_transfer.P,bad,pr.ne,4);double limiter_mean=0.;
  for(int e=0;e<pr.ne;++e)for(int c=0;c<4;++c)
  {double change=0.;for(int i=0;i<pr.retained.n;++i)change+=pr.retained.physics.basis.weights[i]*(after[pr.retained.Id(e,i,c)]-before[pr.retained.Id(e,i,c)]);limiter_mean=std::max(limiter_mean,std::abs(change));}
  if(!work.limited||limiter_mean>1.e-11)throw std::runtime_error("primitive active limiter mean check failed");
  PrimitiveMomentProblem central(mo,true);State affine(physical.size());
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i],v=.2+.1*x;const int j=pr.Id(e,i,0);
    affine[j]=2.;affine[j+1]=2.*v;affine[j+2]=0.;affine[j+3]=2.5+v*v;}
  const State affine_moments=embed(affine),central_sigma=central.Closure(affine_moments,0.,work);double central_error=0.;
  for(double value:central_sigma)central_error=std::max(central_error,std::abs(value-4.*t.base.alpha*.01));
  if(central_error>1.e-10)throw std::runtime_error("primitive central gradient affine patch failed");
  std::cout<<"primitive moment checks passed: patch="<<patch<<" moments="<<moment_error<<" boost="<<galilean<<" source="<<source_error<<" conservation="<<conservation<<" limiter-mean="<<limiter_mean<<'\n';
}
}
#ifndef DG_IGR_PRIMITIVE_MOMENTS_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    MemoryOptions mo;std::string source_faces="broken";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {const std::string flag=argv[i];if(flag=="--state-in"||flag=="--state-time"||flag=="--primitive-source-faces")
      {if(++i==argc)throw std::runtime_error("missing primitive moment option");if(flag=="--state-in")mo.input=argv[i];else if(flag=="--primitive-source-faces")source_faces=argv[i];else mo.start_time=std::stod(argv[i]);}else common.push_back(argv[i]);}
    mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;
    if(t.self_test){PrimitiveSelfTest();return 0;}
    if(!std::isfinite(mo.start_time))throw std::runtime_error("nonfinite initial time");
    if(source_faces!="broken"&&source_faces!="central")throw std::runtime_error("unknown primitive source faces");
    const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;
    // Keep the requested extra degree beyond exact high-space mass integration.
    const auto start=Clock::now();PrimitiveMomentProblem pr(mo,source_faces=="central");const double setup=Wall(start);auto result=MemoryRun(pr);result.work.setup=setup;
    const State moment_coefficients=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);
    if(!result.u.empty())result.u=pr.Reconstruct(result.u);
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"joint-density-weighted-primitive\",\"volume_projection\":\"conservative-moments\",\"retained_order\":"<<degree
      <<",\"primitive_source_faces\":\""<<source_faces<<"\",\"stress_order\":"<<degree<<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns
      <<",\"sigma_serialized_order\":"<<pr.p<<",\"injected_state_time\":"<<mo.start_time<<",\"local_velocity_factors\":"<<pr.local_velocity_factors
      <<",\"reconstruction_cells\":"<<pr.reconstruction_cells<<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds
      <<",\"moment_transfer_seconds\":"<<pr.transfer_seconds<<",\"galerkin_seconds\":"<<pr.galerkin_seconds<<",\"moment_coefficients\":";
    TrackArray(extra,moment_coefficients);BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
#endif

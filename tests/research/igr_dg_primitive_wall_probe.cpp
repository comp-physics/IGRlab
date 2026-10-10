// Test-only flat-wall qualification of the conservative primitive manifold.
// Compare a reflected half domain with its equally resolved full-domain image.
// This is discrete reflection parity, not an independent accuracy reference.
#define DG_IGR_PRIMITIVE_MOMENTS_LIBRARY_ONLY
#include "igr_dg_primitive_moments_probe.cpp"
#undef DG_IGR_PRIMITIVE_MOMENTS_LIBRARY_ONLY

namespace
{
struct WallGateOptions
{
  int elements=8,order=3,quad_extra=19;
  double alpha=.0025,amplitude=6.,dt=.0001,final_time=.01;
  bool self_test=false;
  std::string output;
};
MemoryOptions WallSpace(const WallGateOptions& w,bool full)
{
  MemoryOptions o;o.recovery.recovery="existing";auto& b=o.recovery.boundary;
  b.left=full?"outflow":"wall";b.right="outflow";
  b.stress="trace";b.open_closure=b.wall_closure="natural";
  auto& t=b.compatibility.track;t.mode="uniform";t.initial="entropy";
  t.left=full?-1.:0.;t.right=1.;t.center=.5;t.radius=1;
  t.base.elements=(full?2:1)*w.elements;t.base.order=3*w.order;
  t.base.alpha=w.alpha;t.base.dt=w.dt;t.base.final_time=w.final_time;
  t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=w.quad_extra;
  o.retained_order=o.stress_order=w.order;return o;
}
struct PrimitiveWallProblem:PrimitiveMomentProblem
{
  bool full_domain;double amplitude;
  explicit PrimitiveWallProblem(const WallGateOptions& w,bool full):
    PrimitiveMomentProblem(WallSpace(w,full),false),full_domain(full),amplitude(w.amplitude){}
  State Initial()const
  {
    const auto start=Clock::now();State low(ne*retained.n*4,0.);
    for(int e=0;e<ne;++e)for(int i=0;i<retained.n;++i)
    {
      const double x=edges[e]+(edges[e+1]-edges[e])*retained.physics.basis.nodes[i];
      const double rho=1.+.05*std::cos(M_PI*x),v=-amplitude*std::sin(M_PI*x);
      const double pressure=1.+.05*std::cos(2.*M_PI*x);const int j=retained.Id(e,i,0);
      low[j]=rho;low[j+1]=rho*v;low[j+2]=0.;low[j+3]=pressure/(gamma_euler-1.)+.5*rho*v*v;
    }
    State out=state_transfer.Apply(state_transfer.E,low,ne,4);transfer_seconds+=Wall(start);return out;
  }
  void MixedSystem(const State& physical,double time,TrackWork& work,BandMatrix& a,State& b)const override
  {
    MemoryProblem::MixedSystem(physical,time,work,a,b);
    // Redundant original source assembly is retained in work/timing. Complete
    // the central derivative in the actual degree-p velocity space.
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
          const int neighbor=e+(side?1:-1),endpoint=side?p:0,low_endpoint=side?retained.p:0;
          const double own=physical[Id(e,endpoint,1)]/physical[Id(e,endpoint,0)];
          double outside=own;
          if(neighbor>=0&&neighbor<ne)
          {const int other=side?0:p;outside=physical[Id(neighbor,other,1)]/physical[Id(neighbor,other,0)];}
          else if((side?boundary.right:boundary.left)=="wall")outside=-own;
          double lift=0.;for(int i=0;i<m;++i)lift+=low_shapes[q][i]*retained.mass_inverse(i,low_endpoint)/h;
          g+=(side?-1.:1.)*.5*(own-outside)*lift;
        }
        const double source=2.*o.base.alpha*g*g,weight=h*weights[q];integral+=weight*source;
        for(int i=0;i<m;++i)b[e*m+i]+=weight*low_shapes[q][i]*source;
        ++work.source_points;
      }
    }
    work.current_source_integral=integral;
  }
};
State Mirror(const State& half,int cells,int nodes,int components)
{
  State full(2*half.size(),0.);
  for(int e=0;e<cells;++e)for(int i=0;i<nodes;++i)for(int c=0;c<components;++c)
  {
    const double value=half[(e*nodes+i)*components+c];
    full[((cells+e)*nodes+i)*components+c]=value;
    full[((cells-1-e)*nodes+nodes-1-i)*components+c]=(components==4&&c==1)?-value:value;
  }
  return full;
}
double RelativeMax(const State& a,const State& b)
{
  if(a.size()!=b.size())throw std::runtime_error("wall parity vector size mismatch");
  double difference=0.,scale=1.;for(std::size_t i=0;i<a.size();++i)
  {difference=std::max(difference,std::abs(a[i]-b[i]));scale=std::max({scale,std::abs(a[i]),std::abs(b[i])});}
  return difference/scale;
}
struct WallStaticResult
{double matrix=0.,load=0.,sigma=0.,rhs=0.,source=0.,trace=0.;};
WallStaticResult WallStaticCheck(PrimitiveWallProblem& half,PrimitiveWallProblem& full)
{
  State physical(half.ne*half.n*4,0.);
  for(int e=0;e<half.ne;++e)for(int i=0;i<half.n;++i)
  {
    const double z=half.physics.basis.nodes[i],rho=1.+.03*e+.02*z+.005*std::pow(z,half.retained.p);
    const double v=.2+.05*e+.02*z+.007*std::pow(z,half.retained.p),vy=.03+.004*z,pressure=1.+.02*e+.01*z;
    const int j=half.Id(e,i,0);physical[j]=rho;physical[j+1]=rho*v;physical[j+2]=rho*vy;
    physical[j+3]=pressure/(gamma_euler-1.)+.5*rho*(v*v+vy*vy);
  }
  const State low=half.state_transfer.Apply(half.state_transfer.P,physical,half.ne,4);
  const State moments=half.state_transfer.Apply(half.state_transfer.E,low,half.ne,4);
  const State full_moments=Mirror(moments,half.ne,half.n,4);
  physical=half.Reconstruct(moments);const State full_physical=full.Reconstruct(full_moments);
  TrackWork wh,wf;BandMatrix ah,af;State bh,bf;
  half.MixedSystem(physical,0.,wh,ah,bh);full.MixedSystem(full_physical,0.,wf,af,bf);
  const int m=half.stress.n;State restricted(ah.n*ah.n,0.),load(ah.n,0.),reference(ah.n*ah.n,0.);
  auto index=[&](int j){const int e=j/m,i=j%m;return e<half.ne?(half.ne-1-e)*m+m-1-i:(e-half.ne)*m+i;};
  for(int i=0;i<af.n;++i)
  {
    load[index(i)]+=.5*bf[i];
    for(int j=std::max(0,i-af.k);j<std::min(af.n,i+af.k+1);++j)restricted[index(i)*ah.n+index(j)]+=.5*af.Get(i,j);
  }
  for(int i=0;i<ah.n;++i)for(int j=0;j<ah.n;++j)reference[i*ah.n+j]=ah.Get(i,j);
  WallStaticResult out;out.matrix=RelativeMax(restricted,reference);out.load=RelativeMax(load,bh);
  const State sh=half.Closure(moments,0.,wh),sf=full.Closure(full_moments,0.,wf);
  out.sigma=RelativeMax(sf,Mirror(sh,half.ne,half.n,1));
  out.source=std::abs(wf.current_source_integral-2.*wh.current_source_integral)/std::max(1.,std::abs(wf.current_source_integral));
  std::array<double,4> budgeth{},budgetf{};
  const State rh=half.Residual(moments,sh,0.,wh,budgeth),rf=full.Residual(full_moments,sf,0.,wf,budgetf);
  out.rhs=RelativeMax(rf,Mirror(rh,half.ne,half.n,4));out.trace=physical[1]/physical[0];
  if(out.matrix>2.e-10||out.load>2.e-10||out.sigma>2.e-9||out.rhs>2.e-9||out.source>2.e-10||std::abs(out.trace)<.1)
    throw std::runtime_error("nonzero-wall-trace SIPG/source/stress/RHS reflection parity failed");
  return out;
}
std::string WallSerialize(const PrimitiveWallProblem& pr,TrackResult result)
{
  const State moments=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);
  if(!result.u.empty())result.u=pr.Reconstruct(result.u);
  std::ostringstream base;TrackWrite(pr,result,base);std::string text=base.str();text.erase(text.rfind('}'));
  std::ostringstream extra;extra<<std::setprecision(17)<<",\"wall_gate_full_domain\":"<<(pr.full_domain?"true":"false")
    <<",\"boundary_scenario\":\"parity-compatible-wall-sine\",\"left_boundary\":\""<<pr.boundary.left<<"\",\"right_boundary\":\""<<pr.boundary.right
    <<"\",\"exterior_stress\":\"trace\",\"primitive_source_faces\":\"central-mirror\",\"source_recovery\":\"joint-density-weighted-primitive\""
    <<",\"retained_order\":"<<pr.retained.p<<",\"stress_order\":"<<pr.stress.p<<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns
    <<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns<<",\"sigma_serialized_order\":"<<pr.p
    <<",\"wall_amplitude\":"<<pr.amplitude<<",\"maximum_wall_mass_flux\":"<<pr.max_wall_mass
    <<",\"maximum_wall_energy_flux\":"<<pr.max_wall_energy<<",\"maximum_wall_normal_velocity\":"<<pr.max_wall_velocity
    <<",\"reconstruction_cells\":"<<pr.reconstruction_cells<<",\"local_velocity_factors\":"<<pr.local_velocity_factors
    <<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"moment_transfer_seconds\":"<<pr.transfer_seconds
    <<",\"galerkin_seconds\":"<<pr.galerkin_seconds<<",\"moment_coefficients\":";TrackArray(extra,moments);extra<<"}\n";
  return text+extra.str();
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    WallGateOptions o;
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--self-test"){o.self_test=true;continue;}
      if(++i==argc)throw std::runtime_error("missing wall gate option");const std::string value=argv[i];
      if(flag=="--elements")o.elements=std::stoi(value);else if(flag=="--order")o.order=std::stoi(value);
      else if(flag=="--alpha")o.alpha=std::stod(value);else if(flag=="--amplitude")o.amplitude=std::stod(value);
      else if(flag=="--dt")o.dt=std::stod(value);else if(flag=="--final-time")o.final_time=std::stod(value);
      else if(flag=="--quad-extra")o.quad_extra=std::stoi(value);else if(flag=="--output")o.output=value;
      else throw std::runtime_error("unknown wall gate option "+flag);
    }
    if(o.elements<4||o.order<1||o.quad_extra<1||!std::isfinite(o.alpha)||o.alpha<0.||!std::isfinite(o.amplitude)||
       !std::isfinite(o.dt)||!(o.dt>0.)||!std::isfinite(o.final_time)||o.final_time<0.)throw std::runtime_error("invalid wall gate parameters");
    if(o.self_test){o.elements=4;o.order=3;o.dt=1.e-5;o.final_time=3.e-5;}
    const auto total_start=Clock::now(),static_start=Clock::now();PrimitiveWallProblem sh(o,false),sf(o,true);
    const WallStaticResult check=WallStaticCheck(sh,sf);const double static_seconds=Wall(static_start);
    auto setup_start=Clock::now();PrimitiveWallProblem half(o,false);const double half_setup=Wall(setup_start);
    auto h=MemoryRun(half);h.work.setup=half_setup;
    setup_start=Clock::now();PrimitiveWallProblem full(o,true);const double full_setup=Wall(setup_start);
    auto f=MemoryRun(full);f.work.setup=full_setup;
    const bool common=!h.u.empty()&&!f.u.empty()&&std::abs(h.time-f.time)<1.e-13;
    const double state_parity=common?RelativeMax(f.u,Mirror(h.u,half.ne,half.n,4)):-1.;
    const double stress_parity=common?RelativeMax(f.sigma,Mirror(h.sigma,half.ne,half.n,1)):-1.;
    double physical_parity=-1.;if(common)
    {const State hp=half.Reconstruct(h.u),fp=full.Reconstruct(f.u);physical_parity=RelativeMax(fp,Mirror(hp,half.ne,half.n,4));}
    const bool completed=h.failure.empty()&&f.failure.empty();
    if(o.self_test&&(!completed||!common||state_parity>2.e-9||physical_parity>2.e-9||stress_parity>2.e-9||half.max_wall_mass>1.e-11||half.max_wall_energy>1.e-11))
      throw std::runtime_error("advancing primitive wall reflection parity failed");
    const std::string half_text=WallSerialize(half,h),full_text=WallSerialize(full,f);
    std::ostringstream output;output<<std::setprecision(17)<<"{\"scope\":\"same-resolution flat-wall/full-reflection parity, not continuum accuracy or performance\",\"completed\":"<<(completed?"true":"false")
      <<",\"static\":{\"matrix_relative_max\":"<<check.matrix<<",\"load_relative_max\":"<<check.load<<",\"sigma_relative_max\":"<<check.sigma
      <<",\"rhs_relative_max\":"<<check.rhs<<",\"source_integral_relative_defect\":"<<check.source<<",\"nonzero_wall_velocity\":"<<check.trace<<"}"
      <<",\"static_seconds\":"<<static_seconds<<",\"common_endpoint\":"<<(common?"true":"false")<<",\"moment_state_relative_max\":"<<state_parity
      <<",\"physical_state_relative_max\":"<<physical_parity
      <<",\"sigma_relative_max\":"<<stress_parity<<",\"total_seconds_before_output\":"<<Wall(total_start)<<",\"half\":"<<half_text<<",\"full\":"<<full_text<<"}\n";
    if(o.output.empty())std::cout<<output.str();else{std::ofstream file(o.output);if(!file)throw std::runtime_error("cannot open wall gate output");file<<output.str();}
    return completed?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

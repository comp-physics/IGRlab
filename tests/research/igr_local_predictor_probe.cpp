// Test-only local-volume predictor/shared-face corrector and coupled control.
// Reuse the existing reduced problem, production flux, closure algebra, basis,
// RK4 baseline and its option validation without creating a second physics layer.
#define main igr_existing_spacetime_probe_main
#include "screened_spacetime_probe.cpp"
#undef main

namespace
{
struct PredictorOptions
{
  Options base;
  std::string predictor="both";
  int sweeps=3, closure_updates=1, spatial_reference_elements=0;
  bool linear=false, self_test=false;
};
struct PredictorWork
{
  std::uint64_t stage_coefficient_elements=0, closure_apply_elements=0;
  std::uint64_t batch_calls=0, rhs_columns=0, nonzero_rhs_columns=0;
  std::uint64_t predictor_sweeps=0, endpoint_admission_elements=0;
  double maximum_stage_closure_residual=0, maximum_density_ratio_defect=0;
  double last_stage_closure_residual=0, maximum_corrector_closure_residual=0;
};
struct StageClosureData
{
  State inverse_density, rhs;
  double rhs_norm=0;
  StageClosureData(const Problem& pr,const State& u,PredictorWork& w)
  {
    inverse_density.resize(pr.ne*pr.n);rhs.assign(pr.ne*pr.p,0.);
    w.stage_coefficient_elements+=pr.ne;
    for(int e=0;e<pr.ne;++e)for(int k=0;k<pr.n;++k)
    {
      const double* x=&u[pr.Index(e,k,0)];pr.CheckState(x);
      double gradient[8]={};
      for(int c=0;c<4;++c)for(int j=0;j<pr.n;++j)
        gradient[2*c]+=pr.basis.D(k,j)*u[pr.Index(e,j,c)]/pr.h;
      const double source=igr::SourceFromConservativeStateGradient(
          2,x,1,gradient,2,1,pr.o.alpha,floor_value);
      inverse_density[e*pr.n+k]=igr::SafeInverseDensity(x[0],floor_value);
      rhs[pr.SigmaIndex(e,k)]+=pr.h*pr.basis.weights[k]*source;
    }
    rhs_norm=std::max(Norm(rhs),1.e-20);
  }
  State Defect(const Problem& pr,const State& sigma,PredictorWork& w) const
  {
    State result=rhs;w.closure_apply_elements+=pr.ne;
    for(int e=0;e<pr.ne;++e)for(int k=0;k<pr.n;++k)
    {
      const double invrho=inverse_density[e*pr.n+k];
      const double weight=pr.basis.weights[k];const int row=pr.SigmaIndex(e,k);
      result[row]-=pr.h*weight*invrho*sigma[row];
      double sx=0;
      for(int j=0;j<pr.n;++j)sx+=pr.basis.D(k,j)*sigma[pr.SigmaIndex(e,j)]/pr.h;
      for(int i=0;i<pr.n;++i)
        result[pr.SigmaIndex(e,i)]-=pr.o.alpha*invrho*weight*pr.basis.D(k,i)*sx;
    }
    return result;
  }
};

// CURRENT matrices differ at the two stages. Only their preconditioner is
// shared. Two columns are real temporal defects, never duplicated columns.
void SharedStageClosures(const Problem& pr,const CurrentClosure& current,
    const Stages& u,Stages& sigma,int requested_updates,Work& work,PredictorWork& w)
{
  std::array<StageClosureData,2> data{{StageClosureData(pr,u[0],w),StageClosureData(pr,u[1],w)}};
  const int ns=pr.ne*pr.p;
  for(int j=0;j<2;++j)for(int k=0;k<pr.ne*pr.n;++k)
    w.maximum_density_ratio_defect=std::max(w.maximum_density_ratio_defect,
      std::abs(data[j].inverse_density[k]/current.point_cache[k*4]-1.));
  const int max_updates=requested_updates?requested_updates:40;
  mfem::DenseMatrix rhs(ns,2),correction(ns,2);
  for(int update=0;update<=max_updates;++update)
  {
    bool done=true;double maximum=0;
    std::uint64_t nonzero=0;
    for(int stage=0;stage<2;++stage)
    {
      const State residual=data[stage].Defect(pr,sigma[stage],w);
      const double relative=Norm(residual)/data[stage].rhs_norm;
      if(!std::isfinite(relative))throw std::runtime_error("nonfinite stage closure defect");
      maximum=std::max(maximum,relative);
      const bool active=relative>pr.o.closure_tolerance;
      if(active){done=false;++nonzero;}
      for(int k=0;k<ns;++k)rhs(k,stage)=active?residual[k]:0.;
    }
    if(done || update==max_updates)
    {
      w.maximum_stage_closure_residual=std::max(w.maximum_stage_closure_residual,maximum);
      w.last_stage_closure_residual=maximum;
      if(!done && !requested_updates)
        throw std::runtime_error("certified stage closure correction failed");
      return;
    }
    current.inverse->Mult(rhs,correction);++w.batch_calls;
    w.rhs_columns+=2;w.nonzero_rhs_columns+=nonzero;
    work.solve_flops+=4.*ns*ns;
    for(int stage=0;stage<2;++stage)for(int k=0;k<ns;++k)
      sigma[stage][k]+=correction(k,stage);
  }
}

struct CachedFlux
{
  State volume,speed;
  std::vector<std::array<double,4>> flux;
  CachedFlux(const Problem& pr,const State& u,const State& sigma,Work& w)
  {
    volume.assign(u.size(),0.);speed.resize(pr.ne*pr.n);flux.resize(pr.ne*pr.n);
    w.residual_elements+=pr.ne;
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
      flux[e*pr.n+i]=pr.Flux(&u[pr.Index(e,i,0)],sigma[pr.SigmaIndex(e,i)],speed[e*pr.n+i]);
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
      for(int j=0;j<pr.n;++j)for(int c=0;c<4;++c)
        volume[pr.Index(e,i,c)]-=pr.basis.D(i,j)*flux[e*pr.n+j][c]/pr.h;
  }
  State WithFaces(const Problem& pr,const State& u,Work& w) const
  {
    State residual=volume;
    for(int face=0;face<pr.ne;++face)
    {
      ++w.face_fluxes;const int l=pr.Mod(face-1),r=face;
      const double speed_bound=pr.FaceSpeed(speed[l*pr.n+pr.p],speed[r*pr.n]);
      for(int c=0;c<4;++c)
      {
        const double fl=flux[l*pr.n+pr.p][c],fr=flux[r*pr.n][c];
        const double star=.5*(fl+fr-speed_bound*(u[pr.Index(r,0,c)]-u[pr.Index(l,pr.p,c)]));
        residual[pr.Index(l,pr.p,c)]-=(star-fl)/(pr.h*pr.basis.weights[pr.p]);
        residual[pr.Index(r,0,c)]+=(star-fr)/(pr.h*pr.basis.weights[0]);
      }
    }
    return residual;
  }
};

State PredictorStep(const Problem& pr,const State& initial,const CurrentClosure& current,
    double dt,bool local,const PredictorOptions& options,Work& work,PredictorWork& w)
{
  Stages state{{initial,initial}},sigma{{current.sigma,current.sigma}};
  const CachedFlux initial_flux(pr,initial,current.sigma,work);
  const State initial_rhs=local?initial_flux.volume:initial_flux.WithFaces(pr,initial,work);
  Stages residual{{initial_rhs,initial_rhs}};
  std::array<std::unique_ptr<CachedFlux>,2> flux;
  for(int sweep=0;sweep<options.sweeps;++sweep)
  {
    ++w.predictor_sweeps;
    for(int stage=0;stage<2;++stage)for(std::size_t k=0;k<initial.size();++k)
      state[stage][k]=initial[k]+dt*(q[stage][0]*residual[0][k]+q[stage][1]*residual[1][k]);
    SharedStageClosures(pr,current,state,sigma,options.closure_updates,work,w);
    for(int stage=0;stage<2;++stage)
    {
      flux[stage]=std::make_unique<CachedFlux>(pr,state[stage],sigma[stage],work);
      residual[stage]=local?flux[stage]->volume:flux[stage]->WithFaces(pr,state[stage],work);
    }
  }
  w.maximum_corrector_closure_residual=std::max(w.maximum_corrector_closure_residual,
      w.last_stage_closure_residual);
  if(local)for(int stage=0;stage<2;++stage)
    residual[stage]=flux[stage]->WithFaces(pr,state[stage],work);
  State result=initial;
  for(std::size_t k=0;k<initial.size();++k)result[k]+=.5*dt*(residual[0][k]+residual[1][k]);
  return result;
}

struct PredictorResult {Result base;PredictorWork extra;};
PredictorResult RunPredictor(const Problem& pr,const PredictorOptions& options,bool local)
{
  PredictorResult result;auto& r=result.base;auto& w=result.extra;
  r.u=pr.Initial();const auto start=Clock::now();
  try
  {
    auto current=std::make_unique<CurrentClosure>(pr,r.u,r.work);r.sigma=current->sigma;
    while(r.time<pr.o.final_time-1.e-14)
    {
      ++r.work.attempted_steps;
      const double dt=std::min(pr.o.dt,pr.o.final_time-r.time);
      State trial=PredictorStep(pr,r.u,*current,dt,local,options,r.work,w);
      auto endpoint=std::make_unique<CurrentClosure>(pr,trial,r.work);
      w.endpoint_admission_elements+=pr.ne;
      for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
      {double speed=0;pr.Flux(&trial[pr.Index(e,i,0)],endpoint->sigma[pr.SigmaIndex(e,i)],speed);}
      r.u.swap(trial);current.swap(endpoint);r.sigma=current->sigma;
      r.time+=dt;++r.work.accepted_steps;
    }
    r.complete=true;
  }
  catch(const std::exception& error)
  {r.failure=error.what();r.work.rejected_steps=r.work.attempted_steps-r.work.accepted_steps;}
  r.seconds=Wall(start);return result;
}

struct SpatialMetrics
{
  double state_error=-1, sigma_error=-1, minimum_density=INFINITY;
  double minimum_pressure=INFINITY, seconds=0;
};
std::array<double,5> SampleFields(const Problem& pr,const Result& r,double x)
{
  const int e=std::min(pr.ne-1,int(x/pr.h));const double local=x/pr.h-e;
  std::array<double,5> value{};
  for(int i=0;i<pr.n;++i)
  {
    double shape=1;
    for(int j=0;j<pr.n;++j)if(j!=i)
      shape*=(local-pr.basis.nodes[j])/(pr.basis.nodes[i]-pr.basis.nodes[j]);
    for(int c=0;c<4;++c)value[c]+=shape*r.u[pr.Index(e,i,c)];
    value[4]+=shape*r.sigma[pr.SigmaIndex(e,i)];
  }
  return value;
}
SpatialMetrics CompareSpatial(const Problem& pr,const Result& candidate,
    const Problem* reference_problem,const Result* reference)
{
  SpatialMetrics result;const auto start=Clock::now();
  const int panels=reference_problem?reference_problem->ne:pr.ne;
  const auto& rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,2*pr.p+2);
  double un=0,ud=0,sn=0,sd=0;
  for(int panel=0;panel<panels;++panel)for(int i=0;i<rule.GetNPoints();++i)
  {
    const auto& point=rule.IntPoint(i);const double x=(panel+point.x)/panels;
    const auto u=SampleFields(pr,candidate,x);
    result.minimum_density=std::min(result.minimum_density,u[0]);
    const double pressure=(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
    result.minimum_pressure=std::min(result.minimum_pressure,pressure);
    if(reference_problem&&reference&&candidate.complete&&reference->complete)
    {
      const auto exact=SampleFields(*reference_problem,*reference,x);
      const double weight=point.weight/panels;
      for(int c=0;c<4;++c){un+=weight*std::pow(u[c]-exact[c],2);ud+=weight*exact[c]*exact[c];}
      sn+=weight*std::pow(u[4]-exact[4],2);sd+=weight*exact[4]*exact[4];
    }
  }
  if(reference_problem&&reference&&candidate.complete&&reference->complete)
  {result.state_error=std::sqrt(un/std::max(ud,1.e-40));result.sigma_error=std::sqrt(sn/std::max(sd,1.e-40));}
  result.seconds=Wall(start);return result;
}

void PrintPredictor(const Problem& pr,const PredictorOptions& options,const std::string& method,
    const PredictorResult& result,const Result& reference,
    const Problem* spatial_problem=nullptr,const Result* spatial_reference=nullptr)
{
  const Result& r=result.base;const auto& w=result.extra;
  const SpatialMetrics spatial=CompareSpatial(pr,r,spatial_problem,spatial_reference);
  State difference=r.sigma;double sigma_error=-1;
  if(r.complete&&reference.complete)
  {for(std::size_t k=0;k<difference.size();++k)difference[k]-=reference.sigma[k];
   sigma_error=Norm(difference)/std::max(Norm(reference.sigma),1.e-20);}
  const auto initial=pr.Integrals(pr.Initial()),final=pr.Integrals(r.u);
  double conservation=0;for(int c=0;c<4;++c)conservation=std::max(conservation,std::abs(final[c]-initial[c]));
  std::cout<<std::setprecision(16)<<"{\"method\":\""<<method<<"\",\"elements\":"<<pr.ne
    <<",\"order\":"<<pr.p<<",\"case\":\""<<pr.o.initial_case<<"\",\"alpha\":"<<pr.o.alpha
    <<",\"dt\":"<<pr.o.dt<<",\"horizon\":"<<pr.o.final_time
    <<",\"predictor_sweeps_per_step\":"<<(method=="rk4"?0:options.sweeps)
    <<",\"closure_updates_per_sweep\":"<<options.closure_updates
    <<",\"stage_closure_certification_requested\":"<<(options.closure_updates==0?"true":"false")
    <<",\"complete\":"<<(r.complete?"true":"false")<<",\"reached_time\":"<<r.time
    <<",\"wall_seconds\":"<<r.seconds<<",\"reference_dt\":"<<pr.o.reference_dt
    <<",\"reference_seconds\":"<<reference.seconds
    <<",\"state_relative_l2_error\":"<<(r.complete&&reference.complete?pr.Error(r.u,reference.u):-1.)
    <<",\"sigma_relative_nodal_error\":"<<sigma_error
    <<",\"maximum_conservative_integral_drift\":"<<conservation
    <<",\"stage_closure_residual_measured\":"<<(method=="rk4"?"false":"true")
    <<",\"spatial_reference_elements\":"<<(spatial_problem?spatial_problem->ne:0)
    <<",\"spatial_reference_seconds\":"<<(spatial_reference?spatial_reference->seconds:0.)
    <<",\"state_relative_physical_l2_to_fine\":"<<spatial.state_error
    <<",\"sigma_relative_physical_l2_to_fine\":"<<spatial.sigma_error
    <<",\"minimum_final_sampled_density\":"<<spatial.minimum_density
    <<",\"minimum_final_sampled_pressure\":"<<spatial.minimum_pressure
    <<",\"diagnostic_seconds\":"<<spatial.seconds
    <<",\"maximum_stage_closure_residual\":"<<w.maximum_stage_closure_residual
    <<",\"maximum_corrector_closure_residual\":"<<w.maximum_corrector_closure_residual
    <<",\"maximum_density_ratio_defect\":"<<w.maximum_density_ratio_defect
    <<",\"accepted_steps\":"<<r.work.accepted_steps
    <<",\"attempted_steps\":"<<r.work.attempted_steps<<",\"rejected_steps\":"<<r.work.rejected_steps
    <<",\"current_factors\":"<<r.work.global_factors
    <<",\"current_dense_coefficient_elements\":"<<r.work.coefficient_elements
    <<",\"stage_coefficient_elements\":"<<w.stage_coefficient_elements
    <<",\"closure_apply_elements\":"<<w.closure_apply_elements
    <<",\"useful_rhs_per_batch_maximum\":2,\"batch_calls\":"<<w.batch_calls
    <<",\"executed_rhs_columns\":"<<w.rhs_columns
    <<",\"active_rhs_columns\":"<<w.nonzero_rhs_columns
    <<",\"scalar_rhs_solves_total\":"<<r.work.global_factors+w.rhs_columns
    <<",\"residual_flux_elements\":"<<r.work.residual_elements
    <<",\"unique_face_fluxes\":"<<r.work.face_fluxes
    <<",\"endpoint_admission_elements\":"<<w.endpoint_admission_elements
    <<",\"factor_flops_estimate\":"<<r.work.factor_flops<<",\"solve_flops_estimate\":"<<r.work.solve_flops
    <<",\"retained_factor_bytes\":"<<r.work.retained_factor_bytes
    <<",\"failure\":\""<<r.failure<<"\"}\n";
}

// A scalar constant-speed control tests the local predictor independently of
// IGR and nonlinear flux switches. It compares directly to exact translation.
State LinearResidual(const Problem& pr,const State& u,bool faces)
{
  State r(u.size(),0.);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)
    r[e*pr.n+i]-=pr.basis.D(i,j)*u[e*pr.n+j]/pr.h;
  if(faces)for(int e=0;e<pr.ne;++e)
    r[e*pr.n]+=(u[pr.Mod(e-1)*pr.n+pr.p]-u[e*pr.n])/(pr.h*pr.basis.weights[0]);
  return r;
}
State LinearStep(const Problem& pr,const State& u,double dt,int sweeps,bool local,bool rk4)
{
  if(rk4)
  {
    const State a=LinearResidual(pr,u,true);State trial=u;
    for(std::size_t k=0;k<u.size();++k)trial[k]=u[k]+.5*dt*a[k];
    const State b=LinearResidual(pr,trial,true);
    for(std::size_t k=0;k<u.size();++k)trial[k]=u[k]+.5*dt*b[k];
    const State c=LinearResidual(pr,trial,true);
    for(std::size_t k=0;k<u.size();++k)trial[k]=u[k]+dt*c[k];
    const State d=LinearResidual(pr,trial,true);
    for(std::size_t k=0;k<u.size();++k)trial[k]=u[k]+dt*(a[k]+2*b[k]+2*c[k]+d[k])/6.;
    return trial;
  }
  Stages stage{{u,u}},r; r[0]=r[1]=LinearResidual(pr,u,!local);
  for(int sweep=0;sweep<sweeps;++sweep)
  {
    for(int j=0;j<2;++j)for(std::size_t k=0;k<u.size();++k)
      stage[j][k]=u[k]+dt*(q[j][0]*r[0][k]+q[j][1]*r[1][k]);
    for(int j=0;j<2;++j)r[j]=LinearResidual(pr,stage[j],!local);
  }
  if(local)for(int j=0;j<2;++j)r[j]=LinearResidual(pr,stage[j],true);
  State result=u;for(std::size_t k=0;k<u.size();++k)result[k]+=.5*dt*(r[0][k]+r[1][k]);
  return result;
}
void LinearRun(const Problem& pr,const PredictorOptions& options,const std::string& method)
{
  State u(pr.ne*pr.n);const double pi=std::acos(-1.);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    u[e*pr.n+i]=1.+.1*std::sin(2.*pi*pr.h*(e+pr.basis.nodes[i]));
  const auto start=Clock::now();double time=0;
  while(time<pr.o.final_time-1.e-14)
  {const double dt=std::min(pr.o.dt,pr.o.final_time-time);
   u=LinearStep(pr,u,dt,options.sweeps,method=="local",method=="rk4");time+=dt;}
  double num=0,den=0;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {const double exact=1.+.1*std::sin(2.*pi*(pr.h*(e+pr.basis.nodes[i])-time));
   num+=pr.h*pr.basis.weights[i]*std::pow(u[e*pr.n+i]-exact,2);
   den+=pr.h*pr.basis.weights[i]*std::pow(exact-1.,2);}
  const double error=std::sqrt(num/den);
  std::cout<<std::setprecision(16)<<"{\"linear_method\":\""<<method<<"\",\"elements\":"<<pr.ne
    <<",\"order\":"<<pr.p<<",\"dt\":"<<pr.o.dt<<",\"horizon\":"<<time
    <<",\"perturbation_relative_l2_error\":"<<(std::isfinite(error)?error:-1.)
    <<",\"complete_finite\":"<<(std::isfinite(error)?"true":"false")
    <<",\"wall_seconds\":"<<Wall(start)<<"}\n";
}

PredictorOptions ParsePredictor(int argc,char** argv)
{
  PredictorOptions result;std::vector<std::string> base_args{argv[0],"--method","rk4","--halo","0"};
  for(int i=1;i<argc;++i)
  {
    const std::string key=argv[i];
    if(key=="--self-test"){result.self_test=true;continue;}
    if(key=="--linear"){result.linear=true;continue;}
    if(i+1==argc)throw std::runtime_error("missing option value");
    const std::string value=argv[++i];
    if(key=="--predictor")result.predictor=value;
    else if(key=="--predictor-sweeps")result.sweeps=std::stoi(value);
    else if(key=="--closure-updates")result.closure_updates=std::stoi(value);
    else if(key=="--spatial-reference-elements")result.spatial_reference_elements=std::stoi(value);
    else {base_args.push_back(key);base_args.push_back(value);}
  }
  std::vector<char*> base_argv;for(auto& s:base_args)base_argv.push_back(&s[0]);
  result.base=Parse(base_argv.size(),base_argv.data());
  if((result.predictor!="both"&&result.predictor!="local"&&result.predictor!="coupled"&&result.predictor!="rk4")||
     result.sweeps<1||result.sweeps>12||result.closure_updates<0||result.closure_updates>40||
     result.spatial_reference_elements<0||result.spatial_reference_elements>128||
     (result.spatial_reference_elements && (result.spatial_reference_elements<result.base.elements||
      result.spatial_reference_elements%result.base.elements!=0)))
    throw std::runtime_error("unsupported predictor options");
  return result;
}

bool PredictorSelfTest()
{
  Options o;o.elements=5;o.order=3;o.final_time=.008;o.dt=.002;
  Problem pr(o);const State u=pr.Initial();Work w;PredictorWork extra;
  CurrentClosure current(pr,u,w);State sigma=current.sigma;
  for(std::size_t k=0;k<sigma.size();++k)sigma[k]+=.03*std::sin(double(k));
  StageClosureData data(pr,u,extra);const State matrix_free=data.Defect(pr,sigma,extra);
  for(int i=0;i<pr.ne*pr.p;++i)
  {double assembled=current.rhs[i];for(int j=0;j<pr.ne*pr.p;++j)assembled-=current.matrix(i,j)*sigma[j];
   if(std::abs(assembled-matrix_free[i])>1.e-12)return false;}
  const CachedFlux flux(pr,u,current.sigma,w);
  const State cached=flux.WithFaces(pr,u,w),original=pr.Residual(u,current.sigma,pr.ne,true,0,nullptr,nullptr,w);
  for(std::size_t k=0;k<cached.size();++k)if(std::abs(cached[k]-original[k])>1.e-13)return false;
  Stages states{{u,u}},sigmas{{sigma,sigma}};
  for(std::size_t k=0;k<u.size();++k){states[0][k]*=1.01;states[1][k]*=.99;}
  SharedStageClosures(pr,current,states,sigmas,0,w,extra);
  for(int stage=0;stage<2;++stage)
  {State exact=pr.ExactClosure(states[stage],w);for(std::size_t k=0;k<exact.size();++k)exact[k]-=sigmas[stage][k];
   if(Norm(exact)>1.e-10)return false;}
  Result same;same.u=u;same.sigma=current.sigma;same.complete=true;
  const auto same_metric=CompareSpatial(pr,same,&pr,&same);
  if(same_metric.state_error!=0.||same_metric.sigma_error!=0.||same_metric.minimum_pressure<=0.)return false;
  PredictorOptions options;options.base=o;
  const auto reference=Run(pr,"rk4",.00003125);
  const auto local=RunPredictor(pr,options,true),coupled=RunPredictor(pr,options,false);
  if(!local.base.complete||!coupled.base.complete||!reference.complete)return false;
  const auto initial=pr.Integrals(u);
  for(const auto* r:{&local.base,&coupled.base})
  {const auto integral=pr.Integrals(r->u);for(int c=0;c<4;++c)if(std::abs(integral[c]-initial[c])>1.e-12)return false;}
  if(coupled.base.work.global_factors!=coupled.base.work.accepted_steps+1 ||
     coupled.extra.rhs_columns>6*coupled.base.work.accepted_steps)return false;
  return pr.Error(coupled.base.u,reference.u)<1.e-7;
}
} // namespace

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);
  try
  {
    const PredictorOptions options=ParsePredictor(argc,argv);const Problem pr(options.base);
    if(options.self_test)
    {const bool pass=PredictorSelfTest();std::cout<<"local predictor self-test "<<(pass?"passed":"FAILED")<<'\n';return pass?0:1;}
    if(options.linear)
    {for(const std::string method:{"rk4","local","coupled"})
      if(options.predictor=="both"||options.predictor==method)LinearRun(pr,options,method);
     return 0;}
    const Result reference=Run(pr,"rk4",pr.o.reference_dt);
    if(!reference.complete)throw std::runtime_error("reference failed: "+reference.failure);
    std::unique_ptr<Problem> spatial_problem;std::unique_ptr<Result> spatial_reference;
    if(options.spatial_reference_elements)
    {
      Options fine=pr.o;fine.elements=options.spatial_reference_elements;
      fine.alpha_from_h=false; // Hold physical alpha fixed, even for an h/p-selected candidate alpha.
      spatial_problem=std::make_unique<Problem>(fine);
      spatial_reference=std::make_unique<Result>(Run(*spatial_problem,"rk4",pr.o.reference_dt));
      if(!spatial_reference->complete)throw std::runtime_error("spatial reference failed: "+spatial_reference->failure);
    }
    if(options.predictor=="both"||options.predictor=="rk4")
    {
      PredictorResult baseline;baseline.base=Run(pr,"rk4",pr.o.dt);
      baseline.extra.endpoint_admission_elements=pr.ne*baseline.base.work.accepted_steps;
      PrintPredictor(pr,options,"rk4",baseline,reference,spatial_problem.get(),spatial_reference.get());
    }
    for(const std::string method:{"local","coupled"})
      if(options.predictor=="both"||options.predictor==method)
        PrintPredictor(pr,options,method,RunPredictor(pr,options,method=="local"),reference,
                       spatial_problem.get(),spatial_reference.get());
    return 0;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}

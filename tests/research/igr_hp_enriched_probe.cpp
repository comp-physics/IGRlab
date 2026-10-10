// Test-only stationary, locally enriched conservative Euler DG with current IGR.
// Two singular Euler modes replace two polynomial modes on only the two pieces
// adjacent to the prescribed stationary sonic face. H1 sigma remains degree p.
// Every conservative coefficient evolves independently; no profile/source fit
// enters the closure or volume flux. This is an accuracy/cost screen on CPU.
#define DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY
#include "igr_hp_front_tracking_probe.cpp"
#undef DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY

namespace
{
struct EnrichedOptions
{
  TrackOptions track;
  bool enriched=true,self_test=false;
  int pair_points=64,ordinary_points=0;
  double perturbation=0.;
};

struct EnrichedBasis
{
  const TrackingProblem& base;
  int kind,n; // 0 ordinary; 1 cusp at right; 2 cusp at left.
  State x,w,mean,constant,left,right;
  std::vector<State> value,derivative,sigma_value,sigma_derivative,positive;
  mfem::DenseMatrix transform,mass,inverse;

  State Raw(double z,bool differentiate=false,double cusp_distance=-1.)const
  {
    if(kind==0)return base.Shape(z,differentiate);
    State b(n,0.);b[0]=differentiate?-1.:1.-z;b[1]=differentiate?1.:z;
    // Retain all polynomials through degree p-2, plus two singular bubbles.
    for(int j=0;j<n-4;++j)
    {
      const double q=2*z-1.;double P=1.,dP=0.,pm=1.,dm=0.;
      if(j){P=q;dP=2.;for(int k=2;k<=j;++k){const double pn=((2*k-1)*q*P-(k-1)*pm)/k;
        const double dn=((2*k-1)*(2*P+q*dP)-(k-1)*dm)/k;pm=P;dm=dP;P=pn;dP=dn;}}
      b[2+j]=differentiate?(1.-2*z)*P+z*(1.-z)*dP:z*(1.-z)*P;
    }
    // Preserve the small distance on the left: 1-(1-t^7) may round to zero.
    const double s=cusp_distance>=0.?cusp_distance:(kind==1?1.-z:z),orientation=kind==1?-1.:1.;
    for(int j=0;j<2;++j)
    {const double exponent=(6.+2*j)/7.;b[n-2+j]=differentiate?
      orientation*(exponent*std::pow(s,exponent-1.)-1.):std::pow(s,exponent)-s;}
    return b;
  }
  State Apply(const State& raw)const
  {State out(n,0.);for(int i=0;i<n;++i)for(int j=0;j<n;++j)out[j]+=raw[i]*transform(i,j);return out;}
  State At(double z)const{return Apply(Raw(z));}
  explicit EnrichedBasis(const TrackingProblem& pr,int k,int points,bool graded):base(pr),kind(k),n(pr.n),transform(n),mass(n),inverse(n)
  {
    mfem::IntegrationRule q;mfem::QuadratureFunctions1D generator;generator.GaussLegendre(points,&q);
    for(int j=0;j<q.GetNPoints();++j)
    {
      const double t=q.IntPoint(j).x;
      // Apply the same near-sonic quadrature to polynomial controls. The
      // coordinate removes all 6/7 and 8/7 basis powers from weak integrals.
      const double z=graded?(kind==1?1.-std::pow(t,7):std::pow(t,7)):t;
      x.push_back(z);w.push_back(q.IntPoint(j).weight*(graded?7*std::pow(t,6):1.));
      const double distance=graded?std::pow(t,7):-1.;
      value.push_back(Raw(z,false,distance));derivative.push_back(Raw(z,true,distance));
      sigma_value.push_back(base.Shape(z,false));sigma_derivative.push_back(base.Shape(z,true));
    }
    // Twice-reorthogonalized weighted QR avoids an ill-conditioned raw mass
    // without changing the span or introducing constraints among fields.
    transform=0.;for(int j=0;j<n;++j)transform(j,j)=1.;
    std::vector<State> columns(n,State(w.size()));
    for(int j=0;j<n;++j)
    {
      for(std::size_t q=0;q<w.size();++q)columns[j][q]=value[q][j];
      for(int pass=0;pass<2;++pass)for(int i=0;i<j;++i)
      {
        double projection=0.;for(std::size_t q=0;q<w.size();++q)projection+=w[q]*columns[i][q]*columns[j][q];
        for(std::size_t q=0;q<w.size();++q)columns[j][q]-=projection*columns[i][q];
        for(int r=0;r<n;++r)transform(r,j)-=projection*transform(r,i);
      }
      double norm=0.;for(std::size_t q=0;q<w.size();++q)norm+=w[q]*columns[j][q]*columns[j][q];norm=std::sqrt(norm);
      if(!(norm>1.e-12))throw std::runtime_error("enriched basis rank failure");
      for(std::size_t q=0;q<w.size();++q)columns[j][q]/=norm;for(int r=0;r<n;++r)transform(r,j)/=norm;
    }
    mean.assign(n,0.);mass=0.;
    for(std::size_t q=0;q<w.size();++q)
    {value[q]=Apply(value[q]);derivative[q]=Apply(derivative[q]);
      for(int i=0;i<n;++i){mean[i]+=w[q]*value[q][i];for(int j=0;j<n;++j)mass(i,j)+=w[q]*value[q][i]*value[q][j];}}
    mfem::DenseMatrixInverse inv(mass,true);mfem::DenseMatrix identity(n);identity=0.;for(int i=0;i<n;++i)identity(i,i)=1.;inv.Mult(identity,inverse);
    constant.assign(n,0.);for(int i=0;i<n;++i)for(int j=0;j<n;++j)constant[i]+=inverse(i,j)*mean[j];
    left=At(0.);right=At(1.);positive=value;positive.push_back(left);positive.push_back(right);
    for(int i=0;i<=4*base.p+8;++i)positive.push_back(At(double(i)/(4*base.p+8)));
    for(std::size_t q=0;q<w.size();++q)
    {double one=0.,zero=0.;for(int i=0;i<n;++i){one+=value[q][i]*constant[i];zero+=derivative[q][i]*constant[i];}
      if(std::abs(one-1.)>1.e-10||std::abs(zero)>1.e-6)throw std::runtime_error("constant not represented in enriched span");}
  }
};

struct EnrichedWork
{
  double setup=0.,closure=0.,transport=0.,limiter=0.,minimum_theta=1.;
  double maximum_closure_residual=0.,maximum_rhs_defect=0.;
  std::uint64_t closure_calls=0,source_points=0,flux_points=0,faces=0,limited=0,steps=0,attempted=0;
};

struct EnrichedProblem
{
  EnrichedOptions o;TrackingProblem base;
  // Near-sonic polynomial controls use distinct mirrored quadrature instances.
  EnrichedBasis regular,left_pair,right_pair;
  explicit EnrichedProblem(const EnrichedOptions& options):o(options),base(o.track),
    regular(base,0,o.ordinary_points?o.ordinary_points:base.p+4,false),
    left_pair(base,o.enriched?1:0,o.pair_points,true),
    right_pair(base,o.enriched?2:0,o.pair_points,true)
  {
    if(std::abs(base.profile.speed)>1.e-14&&o.track.initial=="profile")throw std::runtime_error("enriched screen requires stationary reference");
    // An ordinary left-pair rule is reflected to match the enriched left rule.
    // For a polynomial this changes no span; use a separately reconstructed
    // reflected data set because source quadrature must resolve the same cusp.
    if(!o.enriched)
    {
      for(std::size_t q=0;q<left_pair.x.size();++q)
      {const double z=1.-left_pair.x[q];left_pair.x[q]=z;
        left_pair.value[q]=left_pair.At(z);left_pair.derivative[q]=left_pair.Apply(left_pair.Raw(z,true));
        left_pair.sigma_value[q]=base.Shape(z,false);left_pair.sigma_derivative[q]=base.Shape(z,true);}
      left_pair.positive=left_pair.value;left_pair.positive.push_back(left_pair.left);left_pair.positive.push_back(left_pair.right);
      for(int i=0;i<=4*base.p+8;++i)left_pair.positive.push_back(left_pair.At(double(i)/(4*base.p+8)));
    }
  }
  const EnrichedBasis& Basis(int e)const{return e==base.patch_index-1?left_pair:(e==base.patch_index?right_pair:regular);}
  std::array<double,4> Point(const State& u,int e,const State& shape)const
  {std::array<double,4> v{};for(int i=0;i<base.n;++i)for(int c=0;c<4;++c)v[c]+=shape[i]*u[base.Id(e,i,c)];return v;}
  std::array<double,4> Integrals(const State& u)const
  {std::array<double,4> integral{};for(int e=0;e<base.ne;++e)for(int i=0;i<base.n;++i)for(int c=0;c<4;++c)integral[c]+=(base.edges[e+1]-base.edges[e])*Basis(e).mean[i]*u[base.Id(e,i,c)];return integral;}
  State Initial()const
  {
    State u(base.ne*base.n*4,0.);
    for(int e=0;e<base.ne;++e)
    {
      const auto& b=Basis(e);State load(base.n*4,0.);
      for(std::size_t q=0;q<b.w.size();++q)
      {
        const double x=base.edges[e]+(base.edges[e+1]-base.edges[e])*b.x[q];auto v=base.Reference(x,0.);
        const double z=(x-base.o.center)/.12;
        const double perturb=std::abs(z)<1.?o.perturbation*std::pow(1.-z*z,4)*std::sin(37*(x-base.o.center)):0.;
        const double direction[4]={1.,.7,.2,10.};for(int c=0;c<4;++c)v[c]+=direction[c]*perturb;
        for(int i=0;i<base.n;++i)for(int c=0;c<4;++c)load[i*4+c]+=b.w[q]*b.value[q][i]*v[c];
      }
      for(int i=0;i<base.n;++i)for(int j=0;j<base.n;++j)for(int c=0;c<4;++c)u[base.Id(e,i,c)]+=b.inverse(i,j)*load[j*4+c];
    }return u;
  }
  void Limit(State& u,EnrichedWork& work)const
  {
    const auto start=Clock::now();for(int e=0;e<base.ne;++e)
    {
      const auto& b=Basis(e);std::array<double,4> mean{};
      for(int i=0;i<base.n;++i)for(int c=0;c<4;++c)mean[c]+=b.mean[i]*u[base.Id(e,i,c)];
      if(!TrackAdmitted(mean))throw std::runtime_error("enriched element mean admission failed");double theta=1.;
      for(const auto& shape:b.positive)
      {
        const auto point=Point(u,e,shape);if(TrackAdmitted(point))continue;
        if(!base.o.positivity)throw std::runtime_error("enriched polynomial admission failed");
        double lo=0.,hi=1.;for(int k=0;k<50;++k){const double mid=.5*(lo+hi);std::array<double,4> value{};
          for(int c=0;c<4;++c)value[c]=mean[c]+mid*(point[c]-mean[c]);if(TrackAdmitted(value))lo=mid;else hi=mid;}
        theta=std::min(theta,lo*(1.-1.e-12));
      }
      if(theta<1.){++work.limited;work.minimum_theta=std::min(work.minimum_theta,theta);
        for(int i=0;i<base.n;++i)for(int c=0;c<4;++c)u[base.Id(e,i,c)]=theta*u[base.Id(e,i,c)]+(1.-theta)*mean[c]*b.constant[i];}
    }work.limiter+=Wall(start);
  }
  State Closure(const State& u,double time,EnrichedWork& work)const
  {
    const auto start=Clock::now();BandMatrix a(base.ns,base.p);State load(base.ns,0.);
    for(int e=0;e<base.ne;++e)
    {
      const auto& b=Basis(e);const double h=base.edges[e+1]-base.edges[e];
      for(std::size_t q=0;q<b.w.size();++q)
      {
        ++work.source_points;const auto v=Point(u,e,b.value[q]);base.physics.CheckState(v.data());double du[8]={};
        for(int i=0;i<base.n;++i)for(int c=0;c<4;++c)du[2*c]+=b.derivative[q][i]*u[base.Id(e,i,c)]/h;
        const double source=igr::SourceFromConservativeStateGradient(2,v.data(),1,du,2,1,o.track.base.alpha,floor_value);
        if(!std::isfinite(source))throw std::runtime_error("nonfinite enriched source");
        for(int i=0;i<base.n;++i)
        {load[e*base.p+i]+=h*b.w[q]*b.sigma_value[q][i]*source;
          for(int j=0;j<base.n;++j)a.At(e*base.p+i,e*base.p+j)+=b.w[q]/v[0]*(h*b.sigma_value[q][i]*b.sigma_value[q][j]+o.track.base.alpha/h*b.sigma_derivative[q][i]*b.sigma_derivative[q][j]);}
      }
    }
    for(int side=0;side<2;++side)
    {
      const int j=side?base.ns-1:0;const double boundary=base.Reference(side?base.o.right:base.o.left,time)[4];
      for(int i=std::max(0,j-base.p);i<std::min(base.ns,j+base.p+1);++i)if(i!=j){load[i]-=a.Get(i,j)*boundary;a.At(i,j)=0.;a.At(j,i)=0.;}
      a.At(j,j)=1.;load[j]=boundary;
    }
    BandFactor factor(a,0,base.ns);State sigma=load;factor.Solve(sigma,1);State defect(base.ns,0.);
    for(int i=0;i<base.ns;++i){defect[i]=-load[i];for(int j=std::max(0,i-base.p);j<std::min(base.ns,i+base.p+1);++j)defect[i]+=a.Get(i,j)*sigma[j];}
    const double residual=Norm(defect)/std::max(1.e-12,Norm(load));work.maximum_closure_residual=std::max(work.maximum_closure_residual,residual);
    if(!std::isfinite(residual)||residual>1.e-8)throw std::runtime_error("enriched current closure residual failed");
    ++work.closure_calls;work.closure+=Wall(start);return sigma;
  }
  State Rhs(const State& u,const State& sigma,double time,EnrichedWork& work,std::array<double,4>& boundary)const
  {
    const auto start=Clock::now();State load(u.size(),0.),rhs(u.size(),0.);
    for(int e=0;e<base.ne;++e)
    {
      const auto& b=Basis(e);
      for(std::size_t q=0;q<b.w.size();++q)
      {
        ++work.flux_points;const auto v=Point(u,e,b.value[q]);double s=0.,speed=0.;for(int i=0;i<base.n;++i)s+=b.sigma_value[q][i]*sigma[e*base.p+i];
        const auto f=base.physics.Flux(v.data(),s,speed);
        for(int i=0;i<base.n;++i)for(int c=0;c<4;++c)load[base.Id(e,i,c)]+=b.w[q]*b.derivative[q][i]*f[c];
      }
    }
    for(int face=0;face<=base.ne;++face)
    {
      ++work.faces;std::array<double,4> vl{},vr{};double sl=0.,sr=0.,al=0.,ar=0.;
      if(face==0){const auto v=base.Reference(base.o.left,time);std::copy_n(v.begin(),4,vl.begin());sl=v[4];}
      else{vl=Point(u,face-1,Basis(face-1).right);sl=sigma[face*base.p];}
      if(face==base.ne){const auto v=base.Reference(base.o.right,time);std::copy_n(v.begin(),4,vr.begin());sr=v[4];}
      else{vr=Point(u,face,Basis(face).left);sr=sigma[face*base.p];}
      const auto fl=base.physics.Flux(vl.data(),sl,al),fr=base.physics.Flux(vr.data(),sr,ar);const double speed=base.physics.FaceSpeed(al,ar);
      for(int c=0;c<4;++c)
      {
        const double f=.5*(fl[c]+fr[c]-speed*(vr[c]-vl[c]));
        if(face)for(int i=0;i<base.n;++i)load[base.Id(face-1,i,c)]-=Basis(face-1).right[i]*f;
        if(face<base.ne)for(int i=0;i<base.n;++i)load[base.Id(face,i,c)]+=Basis(face).left[i]*f;
        if(face==0)boundary[c]=f;if(face==base.ne)boundary[c]-=f;
      }
    }
    for(int e=0;e<base.ne;++e)for(int i=0;i<base.n;++i)for(int j=0;j<base.n;++j)for(int c=0;c<4;++c)rhs[base.Id(e,i,c)]+=Basis(e).inverse(i,j)*load[base.Id(e,j,c)]/(base.edges[e+1]-base.edges[e]);
    const auto integrals=Integrals(rhs);for(int c=0;c<4;++c)work.maximum_rhs_defect=std::max(work.maximum_rhs_defect,std::abs(integrals[c]-boundary[c]));
    work.transport+=Wall(start);return rhs;
  }
  State RawCoefficients(const State& u)const
  {State raw(u.size(),0.);for(int e=0;e<base.ne;++e)for(int i=0;i<base.n;++i)for(int j=0;j<base.n;++j)for(int c=0;c<4;++c)raw[base.Id(e,i,c)]+=Basis(e).transform(i,j)*u[base.Id(e,j,c)];return raw;}
};

struct EnrichedResult
{
  State u,sigma;EnrichedWork work;double time=0.,wall=0.;std::string failure;
  std::array<double,4> initial_integrals{},budget{},conservation{};
};
EnrichedResult AdvanceEnriched(const EnrichedProblem& pr)
{
  EnrichedResult out;const auto start=Clock::now();
  try
  {
    out.u=pr.Initial();pr.Limit(out.u,out.work);out.initial_integrals=pr.Integrals(out.u);out.sigma=pr.Closure(out.u,0.,out.work);
    while(out.time<pr.o.track.base.final_time-1.e-14)
    {
      ++out.work.attempted;const double dt=std::min(pr.o.track.base.dt,pr.o.track.base.final_time-out.time);const State original=out.u;State stage=original;
      std::array<std::array<double,4>,3> budget{};const double times[3]={out.time,out.time+dt,out.time+.5*dt};
      for(int k=0;k<3;++k)
      {
        const State sigma=k?pr.Closure(stage,times[k],out.work):out.sigma;const State rhs=pr.Rhs(stage,sigma,times[k],out.work,budget[k]);
        for(std::size_t i=0;i<stage.size();++i){const double v=stage[i]+dt*rhs[i];stage[i]=k==0?v:(k==1?.75*original[i]+.25*v:original[i]/3.+2.*v/3.);}
        pr.Limit(stage,out.work);
      }
      out.u.swap(stage);out.time+=dt;++out.work.steps;
      for(int c=0;c<4;++c)out.budget[c]+=dt*(budget[0][c]/6.+budget[1][c]/6.+2.*budget[2][c]/3.);
      out.sigma=pr.Closure(out.u,out.time,out.work);
    }
  }
  catch(const std::exception& error){out.failure=error.what();}
  if(!out.u.empty()){const auto now=pr.Integrals(out.u);for(int c=0;c<4;++c)out.conservation[c]=now[c]-out.initial_integrals[c]-out.budget[c];}
  out.wall=Wall(start);return out;
}

void EnrichedSelfTest()
{
  for(int p:{3,5,7})for(bool enriched:{false,true})
  {
    EnrichedOptions o;o.track.initial="entropy";o.track.base.elements=8;o.track.base.order=p;o.track.base.alpha=.0004;o.track.left=0.;o.track.right=1.;o.track.center=.51;o.track.radius=1;o.track.mode="frozen";o.enriched=enriched;
    EnrichedProblem pr(o);State u(pr.base.ne*pr.base.n*4,0.);EnrichedWork work;
    for(int e=0;e<pr.base.ne;++e)for(int i=0;i<pr.base.n;++i)for(int c=0;c<4;++c)u[pr.base.Id(e,i,c)]=pr.Basis(e).constant[i]*(c==0?2.:(c==1?.4:(c==2?0.:4.)));
    const State sigma=pr.Closure(u,0.,work);if(Norm(sigma)>1.e-9)throw std::runtime_error("constant-velocity enriched source failed");
    std::array<double,4> boundary{};pr.Rhs(u,sigma,0.,work,boundary);if(work.maximum_rhs_defect>2.e-8)throw std::runtime_error("enriched conservative flux budget failed");
    const auto mass=pr.Integrals(u);if(std::abs(mass[0]-2.)>2.e-10||std::abs(mass[3]-4.)>2.e-10)throw std::runtime_error("enriched mass integration failed");
  }
  std::cout<<"enriched p3/p5/p7 self-test passed: constants, full mass, current closure, four-field boundary conservation\n";
}

void EnrichedWrite(const EnrichedProblem& pr,const EnrichedResult& r,std::ostream& f)
{
  f<<std::setprecision(17)<<"{\n\"method\":\"local_independent_conservative_enriched_DG\",\n\"completed\":"<<(r.failure.empty()?"true":"false")<<",\"failure\":\""<<r.failure<<"\",\n\"enriched\":"<<(pr.o.enriched?"true":"false")<<",\"elements\":"<<pr.base.ne<<",\"order\":"<<pr.base.p<<",\"euler_dofs_per_field\":"<<pr.base.ne*pr.base.n<<",\"sigma_dofs\":"<<pr.base.ns<<",\"enriched_elements\":"<<(pr.o.enriched?2:0)<<",\"sonic_face\":"<<pr.base.patch_index<<",\n\"alpha\":"<<pr.o.track.base.alpha<<",\"dt\":"<<pr.o.track.base.dt<<",\"target_time\":"<<pr.o.track.base.final_time<<",\"time\":"<<r.time<<",\"pair_points\":"<<pr.o.pair_points<<",\"ordinary_points\":"<<pr.regular.x.size()<<",\"perturbation\":"<<pr.o.perturbation<<",\"wall_seconds\":"<<r.wall<<",\n\"work\":{\"setup_seconds\":"<<r.work.setup<<",\"closure_seconds\":"<<r.work.closure<<",\"transport_seconds\":"<<r.work.transport<<",\"limiter_seconds\":"<<r.work.limiter<<",\"closure_calls\":"<<r.work.closure_calls<<",\"source_points\":"<<r.work.source_points<<",\"flux_points\":"<<r.work.flux_points<<",\"faces\":"<<r.work.faces<<",\"limited\":"<<r.work.limited<<",\"minimum_theta\":"<<r.work.minimum_theta<<",\"steps\":"<<r.work.steps<<",\"attempted_steps\":"<<r.work.attempted<<",\"maximum_closure_residual\":"<<r.work.maximum_closure_residual<<",\"maximum_rhs_defect\":"<<r.work.maximum_rhs_defect<<"},\n\"conservation_defect\":";TrackArray(f,State(r.conservation.begin(),r.conservation.end()));
  f<<",\n\"edges\":";TrackArray(f,pr.base.edges);f<<",\n\"nodes\":";TrackArray(f,State(pr.base.physics.basis.nodes.GetData(),pr.base.physics.basis.nodes.GetData()+pr.base.n));
  f<<",\n\"raw_euler_coefficients\":";TrackArray(f,pr.RawCoefficients(r.u));f<<",\n\"sigma\":";TrackArray(f,r.sigma);f<<"\n}\n";
}
EnrichedOptions EnrichedParse(int argc,char** argv)
{
  EnrichedOptions o;o.track.base.elements=16;o.track.base.order=7;o.track.base.alpha=.0004;o.track.base.dt=1.e-5;o.track.base.final_time=.003;
  o.track.mode="frozen";o.track.radius=1;o.track.center=0.;
  for(int i=1;i<argc;++i)
  {
    const std::string flag=argv[i];if(flag=="--self-test"){o.self_test=true;continue;}if(flag=="--ordinary"){o.enriched=false;continue;}if(flag=="--no-positivity"){o.track.positivity=false;continue;}
    if(i+1==argc)throw std::runtime_error("missing enriched option value");const std::string v=argv[++i];
    if(flag=="--elements")o.track.base.elements=std::stoi(v);else if(flag=="--order")o.track.base.order=std::stoi(v);else if(flag=="--alpha")o.track.base.alpha=std::stod(v);else if(flag=="--dt")o.track.base.dt=std::stod(v);else if(flag=="--final-time")o.track.base.final_time=std::stod(v);else if(flag=="--table")o.track.table=v;else if(flag=="--output")o.track.output=v;else if(flag=="--pair-points")o.pair_points=std::stoi(v);else if(flag=="--ordinary-points")o.ordinary_points=std::stoi(v);else if(flag=="--perturbation")o.perturbation=std::stod(v);else if(flag=="--left")o.track.left=std::stod(v);else if(flag=="--right")o.track.right=std::stod(v);else throw std::runtime_error("unknown enriched option "+flag);
  }
  if(o.track.base.order<3||o.track.base.order>7||o.track.base.order%2!=1||o.track.base.elements<4||o.pair_points<7*o.track.base.order+4||!(o.track.base.dt>0.)||o.track.base.final_time<0.)throw std::runtime_error("invalid enriched dimensions/time");
  return o;
}
}
int main(int argc,char** argv)
{
  try
  {
    mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");const auto o=EnrichedParse(argc,argv);if(o.self_test){EnrichedSelfTest();return 0;}
    const auto setup=Clock::now();EnrichedProblem pr(o);const double setup_time=Wall(setup);auto result=AdvanceEnriched(pr);result.work.setup=setup_time;result.wall+=setup_time;
    if(o.track.output.empty())EnrichedWrite(pr,result,std::cout);else{std::ofstream f(o.track.output);if(!f)throw std::runtime_error("cannot open enriched output");EnrichedWrite(pr,result,f);}
    return result.failure.empty()?0:2;
  }
  catch(const std::exception& error){std::cerr<<"enriched probe failed: "<<error.what()<<'\n';return 2;}
}

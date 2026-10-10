// Exact zero-swirl cylindrical reduction of full-energy Euler + instantaneous IGR.
// Test-only ordinary DG; shared EOS, flux, boundary states, source and band solve.
#define DG_IGR_DG_BOUNDARY_LIBRARY_ONLY
#include "igr_dg_boundary_evolution_probe.cpp"
#undef DG_IGR_DG_BOUNDARY_LIBRARY_ONLY

namespace
{
struct RadialOptions
{
  int elements=8,order=7,quad_extra=19,positivity_subdivisions=128;
  double amplitude=6.,alpha=.0004,dt=.00002,final_time=.04,slope_strength=0.;
  std::string source="central",output;bool self_test=false,positivity=true;
};
BoundaryOptions RadialBase(const RadialOptions& r)
{
  BoundaryOptions b;b.left=b.right="wall";b.stress="trace";
  b.positivity_subdivisions=r.positivity_subdivisions;auto& t=b.compatibility.track;
  t.initial="entropy";t.mode="uniform";t.left=.5;t.right=1.;t.center=.75;t.radius=1;
  t.base.elements=r.elements;t.base.order=r.order;t.base.alpha=r.alpha;
  t.base.dt=r.dt;t.base.final_time=r.final_time;t.quad_extra=r.quad_extra;t.amplitude=r.amplitude;
  t.closure_space="l2";t.closure_boundary="natural";t.output=r.output;t.positivity=r.positivity;return b;
}
struct RadialProblem:BoundaryProblem
{
  RadialOptions radial;double sine_scale=0.;State quadrature_x;
  std::vector<mfem::DenseMatrix> radial_mass,radial_inverse;std::vector<State> moment_weights;
  mutable double max_reaction_defect=0.;mutable std::uint64_t admission_points=0;
  double slope_trace_norm=0.;mutable double slope_projection_seconds=0.,slope_face_seconds=0.;
  mutable double max_slope_power_defect=0.,max_slope_cell_mean_defect=0.,max_slope_rate=0.;
  mutable std::uint64_t slope_projections=0,slope_faces=0,slope_extra_flux_points=0;
  explicit RadialProblem(const RadialOptions& input):BoundaryProblem(RadialBase(input)),radial(input)
  {
    double lo=0.,hi=.5;const double pi=std::acos(-1.);
    for(int i=0;i<64;++i){const double mid=.5*(lo+hi);if(std::tan(pi*mid)-2*pi*(mid+1./3.)>0)hi=mid;else lo=mid;}
    const double peak=.5*(lo+hi);sine_scale=radial.amplitude*std::sqrt(.25+.75*peak)/std::sin(pi*peak);
    const auto endpoint_derivative=Shape(1.,true);
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)slope_trace_norm+=endpoint_derivative[i]*mass_inverse(i,j)*endpoint_derivative[j];
    for(const auto& shape:shapes){double z=0.;for(int i=0;i<n;++i)z+=shape[i]*physics.basis.nodes[i];quadrature_x.push_back(z);}
    for(int e=0;e<ne;++e)
    {
      radial_mass.emplace_back(n);radial_inverse.emplace_back(n);moment_weights.emplace_back(n,0.);
      auto& m=radial_mass.back();m=0.;const double h=edges[e+1]-edges[e];
      for(std::size_t q=0;q<weights.size();++q)
      {const double r=edges[e]+h*quadrature_x[q],w=h*weights[q]*r;
        for(int i=0;i<n;++i){moment_weights[e][i]+=w*shapes[q][i];for(int j=0;j<n;++j)m(i,j)+=w*shapes[q][i]*shapes[q][j];}}
      mfem::DenseMatrixInverse inverse(m,true);mfem::DenseMatrix identity(n);identity=0.;for(int i=0;i<n;++i)identity(i,i)=1.;inverse.Mult(identity,radial_inverse.back());
    }
  }
  std::array<double,4> InitialAt(double r)const
  {const double v=sine_scale*std::sin(std::acos(-1.)*(r*r-.25)/.75)/r;return {{1.,v,0.,1./(gamma_euler-1.)+.5*v*v}};}
  State Initial()const
  {State u(ne*n*4,0.);for(int e=0;e<ne;++e)for(int i=0;i<n;++i){const auto v=InitialAt(edges[e]+(edges[e+1]-edges[e])*physics.basis.nodes[i]);for(int c=0;c<4;++c)u[Id(e,i,c)]=v[c];}return u;}
  std::array<double,4> CellIntegral(const State& u,int e)const
  {std::array<double,4> total{};for(int i=0;i<n;++i)for(int c=0;c<4;++c)total[c]+=moment_weights[e][i]*u[Id(e,i,c)];return total;}
  std::array<double,4> Integrals(const State& u,const State&)const
  {std::array<double,4> total{};for(int e=0;e<ne;++e){const auto cell=CellIntegral(u,e);for(int c=0;c<4;++c)total[c]+=cell[c];}return total;}
  void Limit(State& u,TrackWork& w)const
  {
    const auto start=Clock::now();for(int e=0;e<ne;++e)
    {
      auto mean=CellIntegral(u,e);const double volume=.5*(edges[e+1]*edges[e+1]-edges[e]*edges[e]);for(double& value:mean)value/=volume;
      if(!TrackAdmitted(mean))throw std::runtime_error("radial element mean admission failed");double theta=1.;
      for(const auto& shape:positive)
      {
        ++w.positivity_points;const auto v=Point(u,e,shape);if(TrackAdmitted(v))continue;
        if(!o.positivity)throw std::runtime_error("radial polynomial positivity sample failed");double lo=0.,hi=1.;
        for(int j=0;j<50;++j){const double mid=.5*(lo+hi);std::array<double,4> test{};for(int c=0;c<4;++c)test[c]=mean[c]+mid*(v[c]-mean[c]);if(TrackAdmitted(test))lo=mid;else hi=mid;}
        theta=std::min(theta,lo*(1.-1.e-12));
      }
      if(theta<1.){++w.limited;w.min_theta=std::min(w.min_theta,theta);for(int i=0;i<n;++i)for(int c=0;c<4;++c)u[Id(e,i,c)]=mean[c]+theta*(u[Id(e,i,c)]-mean[c]);}
    }w.limiter+=Wall(start);
  }
  double Source(const State& u,int e,std::size_t q)const
  {
    const double h=edges[e+1]-edges[e],r=edges[e]+h*quadrature_x[q];const auto v=Point(u,e,shapes[q]);physics.CheckState(v.data());
    double du[8]={};for(int i=0;i<n;++i)for(int c=0;c<4;++c)du[2*c]+=derivatives[q][i]*u[Id(e,i,c)]/h;
    if(v[2]!=0.)throw std::runtime_error("radial reduction requires identically zero swirl");
    if(radial.source=="central")for(int side=0;side<2;++side)
    {
      const int endpoint=side?p:0,neighbor=e+(side?1:-1);const double own=u[Id(e,endpoint,1)]/u[Id(e,endpoint,0)];double outside=-own;
      if(neighbor>=0&&neighbor<ne){const int j=side?0:p;outside=u[Id(neighbor,j,1)]/u[Id(neighbor,j,0)];}
      double lift=0.;for(int i=0;i<n;++i)lift+=shapes[q][i]*radial_inverse[e](i,endpoint)*edges[e+side];
      du[2]+=v[0]*(side?-1.:1.)*.5*(own-outside)*lift;
    }
    // At (x=r,y=0), d_y m_y=m_r/r. The shared invariant now has eigenvalues
    // (u_r,u/r), including the cylindrical contribution absent in Cartesian1D.
    du[5]=v[1]/r;
    return igr::SourceFromConservativeStateGradient(2,v.data(),1,du,2,1,o.base.alpha,floor_value);
  }
  void AssembleClosure(const State& u,TrackWork& w,BandMatrix& a,State& b,
      const std::function<double(double)>& manufactured={})const
  {
    a=BandMatrix(ns,n);b.assign(ns,0.);double integral=0.;
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];for(std::size_t q=0;q<weights.size();++q)
      {
        ++w.source_points;const auto v=Point(u,e,shapes[q]);physics.CheckState(v.data());const double r=edges[e]+h*quadrature_x[q],ri=1./v[0];
        const double source=manufactured?manufactured(r):Source(u,e,q);if(!std::isfinite(source))throw std::runtime_error("nonfinite radial source");integral+=h*weights[q]*r*source;
        for(int i=0;i<n;++i){b[e*n+i]+=h*weights[q]*r*shapes[q][i]*source;for(int j=0;j<n;++j)a.At(e*n+i,e*n+j)+=weights[q]*r*ri*(h*shapes[q][i]*shapes[q][j]+o.base.alpha*derivatives[q][i]*derivatives[q][j]/h);}
      }
    }
    const State dl=Shape(1.,true),dr=Shape(0.,true);
    for(int face=1;face<ne&&o.base.alpha!=0.;++face)
    {
      ++w.closure_faces;const int el=face-1,er=face;const double hl=edges[face]-edges[face-1],hr=edges[face+1]-edges[face];
      const double al=o.base.alpha*edges[face]/u[Id(el,p,0)],ar=o.base.alpha*edges[face]/u[Id(er,0,0)];
      const double penalty=o.sipg_penalty*n*n*.5*(al/hl+ar/hr);
      for(int i=0;i<2*n;++i)for(int j=0;j<2*n;++j)
      {
        const bool il=i<n,jl=j<n;const int ii=i%n,jj=j%n;const double ji=il?(ii==p?1.:0.):(ii==0?-1.:0.),jjump=jl?(jj==p?1.:0.):(jj==0?-1.:0.);
        if(ji==0.&&jjump==0.)continue;const double fi=.5*(il?al*dl[ii]/hl:ar*dr[ii]/hr),fj=.5*(jl?al*dl[jj]/hl:ar*dr[jj]/hr);
        a.At((il?el:er)*n+ii,(jl?el:er)*n+jj)+=-ji*fj-fi*jjump+penalty*ji*jjump;
      }
    }
    ++w.closure_assemblies;w.current_source_integral=integral;
  }
  State Closure(const State& u,double,TrackWork& w)const
  {
    const auto start=Clock::now();BandMatrix a;State b;AssembleClosure(u,w,a,b);BandFactor factor(a,0,ns);State sigma=b;factor.Solve(sigma,1);State defect(ns,0.);
    for(int i=0;i<ns;++i){defect[i]=-b[i];for(int j=std::max(0,i-a.k);j<std::min(ns,i+a.k+1);++j)defect[i]+=a.Get(i,j)*sigma[j];}
    const double residual=Norm(defect)/std::max(Norm(b),1.e-12);w.max_closure_residual=std::max(w.max_closure_residual,residual);
    if(!std::isfinite(residual)||residual>1.e-8)throw std::runtime_error("radial closure certificate failed");
    double reaction=0.;for(int e=0;e<ne;++e)for(std::size_t q=0;q<weights.size();++q)
    {double s=0.;for(int i=0;i<n;++i)s+=shapes[q][i]*sigma[e*n+i];reaction+=(edges[e+1]-edges[e])*weights[q]*(edges[e]+(edges[e+1]-edges[e])*quadrature_x[q])*s/Point(u,e,shapes[q])[0];}
    max_reaction_defect=std::max(max_reaction_defect,std::abs(reaction-w.current_source_integral));
    for(int e=0;e<ne;++e)for(const auto& shape:positive)
    {++admission_points;const auto v=Point(u,e,shape);if(!TrackAdmitted(v))throw std::runtime_error("radial endpoint state admission failed");double s=0.;for(int i=0;i<n;++i)s+=shape[i]*sigma[e*n+i];const double pressure=(gamma_euler-1.)*(v[3]-.5*v[1]*v[1]/v[0]);if(!igr::FrozenSigmaFluxIsLocallyStrictlyHyperbolic(pressure,s,gamma_euler))throw std::runtime_error("radial endpoint q_sigma admission failed");}
    ++w.closure_calls;w.closure+=Wall(start);return sigma;
  }
  State ProjectRadialVelocity(const State& u)const
  {
    const auto start=Clock::now();State v(ns,0.);
    for(int e=0;e<ne;++e)
    {
      State load(n,0.);const double h=edges[e+1]-edges[e];
      for(std::size_t q=0;q<weights.size();++q)
      {const auto state=Point(u,e,shapes[q]);physics.CheckState(state.data());const double r=edges[e]+h*quadrature_x[q];
        for(int i=0;i<n;++i)load[i]+=h*weights[q]*r*shapes[q][i]*state[1]/state[0];}
      for(int i=0;i<n;++i)for(int j=0;j<n;++j)v[e*n+i]+=radial_inverse[e](i,j)*load[j];++slope_projections;
    }
    slope_projection_seconds+=Wall(start);return v;
  }
  double AddSlope(const State& u,const State& sigma,State& load,TrackWork& w)const
  {
    if(radial.slope_strength==0.)return 0.;const State v=ProjectRadialVelocity(u);const auto start=Clock::now();
    const State left_d=Shape(1.,true),right_d=Shape(0.,true);State change(ns,0.);double power=0.;
    for(int face=1;face<ne;++face)
    {
      const int el=face-1,er=face;const double hl=edges[face]-edges[face-1],hr=edges[face+1]-edges[face],r=edges[face];double jump=0.;
      for(int i=0;i<n;++i)jump+=left_d[i]*v[el*n+i]/hl-right_d[i]*v[er*n+i]/hr;
      std::array<double,4> ul{},ur{};for(int c=0;c<4;++c){ul[c]=u[Id(el,p,c)];ur[c]=u[Id(er,0,c)];}double al=0.,ar=0.;
      physics.Flux(ul.data(),sigma[el*n+p],al);physics.Flux(ur.data(),sigma[er*n],ar);w.flux_points+=2;slope_extra_flux_points+=2;
      const double effective_h=2.*hl*hr/(hl+hr),rate=radial.slope_strength*physics.FaceSpeed(al,ar)*n*n/effective_h;
      // Endpoint-frozen density/radius estimate of the velocity derivative
      // dual norm in the inertia mass integral(r rho phi_i phi_j).
      const double inertia_trace=slope_trace_norm*(1./(r*ul[0]*hl*hl*hl)+1./(r*ur[0]*hr*hr*hr));
      const double tau=rate/inertia_trace;power-=tau*jump*jump;max_slope_rate=std::max(max_slope_rate,rate);
      for(int i=0;i<n;++i){change[el*n+i]-=tau*jump*left_d[i]/hl;change[er*n+i]+=tau*jump*right_d[i]/hr;}++slope_faces;
    }
    double measured=0.;for(int e=0;e<ne;++e){double mean=0.;for(int i=0;i<n;++i){const double delta=change[e*n+i];load[Id(e,i,1)]+=delta;mean+=delta;measured+=v[e*n+i]*delta;}max_slope_cell_mean_defect=std::max(max_slope_cell_mean_defect,std::abs(mean));}
    max_slope_power_defect=std::max(max_slope_power_defect,std::abs(measured-power)/std::max(1.,std::abs(power)));slope_face_seconds+=Wall(start);return power;
  }
  State Residual(const State& u,const State& sigma,double,TrackWork& w,std::array<double,4>& budget,bool physical_boundary=false)const
  {
    const auto start=Clock::now();State load(u.size(),0.),rhs(u.size(),0.);budget={};
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];for(std::size_t q=0;q<weights.size();++q)
      {
        ++w.flux_points;const auto v=Point(u,e,shapes[q]);double s=0.,speed=0.;for(int j=0;j<n;++j)s+=shapes[q][j]*sigma[e*n+j];const auto f=physics.Flux(v.data(),s,speed);
        const double r=edges[e]+h*quadrature_x[q],pressure=(gamma_euler-1.)*(v[3]-.5*v[1]*v[1]/v[0]),P=pressure+s;
        for(int i=0;i<n;++i){for(int c=0;c<4;++c)load[Id(e,i,c)]+=weights[q]*r*derivatives[q][i]*f[c];load[Id(e,i,1)]+=h*weights[q]*shapes[q][i]*P;}
        budget[1]+=h*weights[q]*P;
      }
    }
    for(int face=0;face<=ne;++face)
    {
      ++w.faces;std::array<double,4> flux{};
      if(face==0||face==ne)
      {
        if(!physical_boundary)flux=BoundaryFlux(u,sigma,face==ne);
        else{const int e=face?ne-1:0,i=face?p:0;const auto value=Point(u,e,Shape(face?1.:0.,false));double speed=0.;flux=physics.Flux(value.data(),sigma[e*n+i],speed);}
      }
      else
      {
        std::array<double,4> left{},right{};for(int c=0;c<4;++c){left[c]=u[Id(face-1,p,c)];right[c]=u[Id(face,0,c)];}double al=0.,ar=0.;
        const auto fl=physics.Flux(left.data(),sigma[(face-1)*n+p],al),fr=physics.Flux(right.data(),sigma[face*n],ar);const double speed=physics.FaceSpeed(al,ar);
        for(int c=0;c<4;++c)flux[c]=.5*(fl[c]+fr[c]-speed*(right[c]-left[c]));
      }
      for(int c=0;c<4;++c){const double f=edges[face]*flux[c];if(face>0)load[Id(face-1,p,c)]-=f;if(face<ne)load[Id(face,0,c)]+=f;if(face==0)budget[c]+=f;if(face==ne)budget[c]-=f;}
    }
    AddSlope(u,sigma,load,w);
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int c=0;c<4;++c)rhs[Id(e,i,c)]+=radial_inverse[e](i,j)*load[Id(e,j,c)];
    const auto totals=Integrals(rhs,edges);for(int c=0;c<4;++c)w.max_rhs_defect=std::max(w.max_rhs_defect,std::abs(totals[c]-budget[c]));w.transport+=Wall(start);return rhs;
  }
};
TrackResult RadialRun(RadialProblem& pr)
{
  TrackResult out;const auto start=Clock::now();double accepted_source=0.;
  try
  {
    State initial=pr.Initial();pr.Limit(initial,out.work);State sigma=pr.Closure(initial,0.,out.work);
    out.u=std::move(initial);out.sigma=std::move(sigma);out.initial_integrals=pr.Integrals(out.u,pr.edges);accepted_source=out.work.current_source_integral;
    while(out.time<pr.o.base.final_time-1.e-14)
    {
      ++out.work.attempted;const double dt=std::min(pr.o.base.dt,pr.o.base.final_time-out.time);const State original=out.u;State stage=original;std::array<std::array<double,4>,3> budgets{};
      const double times[3]={out.time,out.time+dt,out.time+.5*dt};
      for(int k=0;k<3;++k)
      {
        const State s=k?pr.Closure(stage,times[k],out.work):out.sigma;const State rhs=pr.Residual(stage,s,times[k],out.work,budgets[k]);
        for(std::size_t j=0;j<stage.size();++j){const double candidate=stage[j]+dt*rhs[j];stage[j]=k==0?candidate:(k==1?.75*original[j]+.25*candidate:original[j]/3.+2.*candidate/3.);}pr.Limit(stage,out.work);
      }
      State next_sigma=pr.Closure(stage,out.time+dt,out.work);out.u=std::move(stage);out.sigma=std::move(next_sigma);out.time+=dt;++out.work.steps;accepted_source=out.work.current_source_integral;
      for(int c=0;c<4;++c)out.boundary_budget[c]+=dt*(budgets[0][c]/6.+budgets[1][c]/6.+2.*budgets[2][c]/3.);
    }
  }catch(const std::exception& error){out.failure=error.what();out.work.current_source_integral=accepted_source;}
  if(!out.u.empty()){const auto now=pr.Integrals(out.u,pr.edges);for(int c=0;c<4;++c)out.conservation_defect[c]=now[c]-out.initial_integrals[c]-out.boundary_budget[c];}
  out.wall=Wall(start);return out;
}
std::string RadialSelfTest()
{
  RadialOptions o;o.elements=4;o.order=5;o.source="broken";RadialProblem pr(o);TrackWork work;State u(pr.ne*pr.n*4,0.),s(pr.ns,0.);
  auto fill=[&](double a,bool variable_rho){for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i){const double r=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i],rho=variable_rho?1.+.2*r:2.,v=a*r;u[pr.Id(e,i,0)]=rho;u[pr.Id(e,i,1)]=rho*v;u[pr.Id(e,i,2)]=0.;u[pr.Id(e,i,3)]=1./(gamma_euler-1.)+.5*rho*v*v;}};
  fill(0.,true);s=pr.Closure(u,0.,work);std::array<double,4> budget{};const State rest=pr.Residual(u,s,0.,work,budget);double rest_error=0.;for(double v:rest)rest_error=std::max(rest_error,std::abs(v));
  // Constant radial dilation has two velocity-gradient eigenvalues equal a.
  const double a=.3;fill(a,false);s=pr.Closure(u,0.,work);double affine_error=0.,source_error=0.;for(double v:s)affine_error=std::max(affine_error,std::abs(v-12.*o.alpha*a*a));
  for(int e=0;e<pr.ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)source_error=std::max(source_error,std::abs(pr.Source(u,e,q)-6.*o.alpha*a*a));
  const State rhs=pr.Residual(u,s,0.,work,budget,true);double metric_error=0.;const double P=1.+12.*o.alpha*a*a;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
  {const double r=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];const std::array<double,4> exact{{-4.*a,-6.*a*a*r,0.,-2.*a*(1./(gamma_euler-1.)+P)-4.*a*a*a*r*r}};for(int c=0;c<4;++c)metric_error=std::max(metric_error,std::abs(rhs[pr.Id(e,i,c)]-exact[c]));}
  // sigma=r^2, rho=2: source=(r^2-4alpha)/2, with signed weighted conormal.
  fill(0.,false);BandMatrix matrix;State load;pr.AssembleClosure(u,work,matrix,load,[&](double r){return .5*(r*r-4.*o.alpha);});
  load.front()-=o.alpha*pr.edges.front()*pr.edges.front();load.back()+=o.alpha*pr.edges.back()*pr.edges.back();BandFactor factor(matrix,0,pr.ns);State sigma=load;factor.Solve(sigma,1);double quadratic_error=0.;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i){const double r=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];quadratic_error=std::max(quadratic_error,std::abs(sigma[e*pr.n+i]-r*r));}
  fill(0.,false);u[pr.Id(1,pr.p/2,3)]=-1.;const auto before=pr.CellIntegral(u,1);const auto limited=work.limited;pr.Limit(u,work);const auto after=pr.CellIntegral(u,1);double limiter_error=0.;for(int c=0;c<4;++c)limiter_error=std::max(limiter_error,std::abs(after[c]-before[c]));
  if(rest_error>1.e-9||affine_error>1.e-10||source_error>1.e-11||metric_error>1.e-8||quadratic_error>1.e-10||limiter_error>1.e-12||work.limited==limited)throw std::runtime_error("radial metric/source/manufactured/limiter self-test failed");
  RadialOptions so=o;so.slope_strength=.08;RadialProblem slope(so);State perturbed(slope.ne*slope.n*4,0.),zero(slope.ns,0.);
  for(int e=0;e<slope.ne;++e)for(int i=0;i<slope.n;++i)
  {const double z=slope.physics.basis.nodes[i],rho=1.+.04*e+.2*z*z,m=.3+.02*e+.1*z+.003*std::pow(z,slope.p);const int j=slope.Id(e,i,0);perturbed[j]=rho;perturbed[j+1]=m;perturbed[j+2]=0.;perturbed[j+3]=3.+.05*z;}
  State change(perturbed.size(),0.);const double expected_power=slope.AddSlope(perturbed,zero,change,work);State delta(perturbed.size(),0.);
  for(int e=0;e<slope.ne;++e)for(int i=0;i<slope.n;++i)for(int j=0;j<slope.n;++j)delta[slope.Id(e,i,1)]+=slope.radial_inverse[e](i,j)*change[slope.Id(e,j,1)];
  double actual_power=0.,mean_error=0.;for(int e=0;e<slope.ne;++e)
  {const double h=slope.edges[e+1]-slope.edges[e];const auto mean=slope.CellIntegral(delta,e);mean_error=std::max(mean_error,std::abs(mean[1]));
    for(std::size_t q=0;q<slope.weights.size();++q){const auto value=slope.Point(perturbed,e,slope.shapes[q]),rate=slope.Point(delta,e,slope.shapes[q]);actual_power+=h*slope.weights[q]*(slope.edges[e]+h*slope.quadrature_x[q])*value[1]/value[0]*rate[1];}}
  const double power_error=std::abs(actual_power-expected_power)/std::max(1.,std::abs(expected_power));
  if(!(expected_power<-1.e-8)||power_error>1.e-11||mean_error>1.e-11)throw std::runtime_error("radial quotient projection/kinetic power identity failed");
  std::ostringstream f;f<<std::setprecision(17)<<"{\"rest_rhs_max\":"<<rest_error<<",\"affine_sigma_max\":"<<affine_error<<",\"affine_source_max\":"<<source_error<<",\"affine_rhs_max\":"<<metric_error<<",\"quadratic_sigma_max\":"<<quadratic_error<<",\"limiter_weighted_mean_max\":"<<limiter_error<<",\"slope_kinetic_power\":"<<expected_power<<",\"slope_kinetic_power_relative_defect\":"<<power_error<<",\"slope_cell_mean_defect\":"<<mean_error<<"}";return f.str();
}
void RadialWrite(const RadialProblem& pr,const TrackResult& result,const std::string& checks,double total)
{
  std::ostringstream f;TrackWrite(pr,result,f);std::string data=f.str();data.erase(data.rfind('}'));std::ostringstream extra;extra<<std::setprecision(17);
  extra<<",\"radial_geometry\":\"exact cylindrical zero swirl, ideal circular walls\",\"integral_measure\":\"r dr; common 2pi omitted\",\"source_gradient\":\""<<pr.radial.source<<"\",\"initial_profile\":\"flux-sine\",\"sine_scale\":"<<pr.sine_scale
    <<",\"maximum_wall_mass_flux\":"<<pr.max_wall_mass<<",\"maximum_wall_energy_flux\":"<<pr.max_wall_energy<<",\"maximum_wall_normal_velocity\":"<<pr.max_wall_velocity<<",\"admission_points\":"<<pr.admission_points<<",\"maximum_reaction_identity_defect\":"<<pr.max_reaction_defect
    <<",\"positivity_enabled\":"<<(pr.o.positivity?"true":"false")<<",\"positivity_subdivisions\":"<<pr.radial.positivity_subdivisions<<",\"slope_strength\":"<<pr.radial.slope_strength<<",\"slope_velocity_projection\":\"r dr L2 projection of actual conservative quotient\",\"slope_trace_norm_reference\":"<<pr.slope_trace_norm
    <<",\"slope_projection_seconds\":"<<pr.slope_projection_seconds<<",\"slope_face_seconds\":"<<pr.slope_face_seconds<<",\"slope_projections\":"<<pr.slope_projections<<",\"slope_faces\":"<<pr.slope_faces<<",\"slope_extra_flux_points\":"<<pr.slope_extra_flux_points
    <<",\"maximum_slope_power_identity_relative_defect\":"<<pr.max_slope_power_defect<<",\"maximum_slope_cell_mean_defect\":"<<pr.max_slope_cell_mean_defect<<",\"maximum_slope_rate_estimate\":"<<pr.max_slope_rate<<",\"initial_integrals\":";TrackArray(extra,State(result.initial_integrals.begin(),result.initial_integrals.end()));
  extra<<",\"integrated_boundary_and_geometric_budget\":";TrackArray(extra,State(result.boundary_budget.begin(),result.boundary_budget.end()));extra<<",\"static_checks\":"<<checks<<",\"total_seconds_before_output\":"<<total<<"}\n";data+=extra.str();
  auto replace=[&](const std::string& from,const std::string& to){const auto at=data.find(from);if(at!=std::string::npos)data.replace(at,from.size(),to);};
  replace("selective_high_order_dg_conservative_rezone","ordinary_dg_radial_full_energy_igr");replace("\"initial\":\"entropy\"","\"initial\":\"radial-flux-sine\"");replace("\"euler_boundary\":\"outflow\"","\"euler_boundary\":\"reflecting-wall\"");
  if(pr.radial.output.empty())std::cout<<data;else{std::ofstream file(pr.radial.output);if(!file)throw std::runtime_error("cannot open radial output");file<<data;}
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    RadialOptions o;for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--self-test"){o.self_test=true;continue;}if(flag=="--no-positivity"){o.positivity=false;continue;}
      if(++i>=argc)throw std::runtime_error("missing radial option");const std::string value=argv[i];
      if(flag=="--elements")o.elements=std::stoi(value);else if(flag=="--order")o.order=std::stoi(value);else if(flag=="--quad-extra")o.quad_extra=std::stoi(value);else if(flag=="--positivity-subdivisions")o.positivity_subdivisions=std::stoi(value);
      else if(flag=="--slope-strength")o.slope_strength=std::stod(value);else if(flag=="--amplitude")o.amplitude=std::stod(value);else if(flag=="--alpha")o.alpha=std::stod(value);else if(flag=="--dt")o.dt=std::stod(value);else if(flag=="--final-time")o.final_time=std::stod(value);else if(flag=="--source")o.source=value;else if(flag=="--output")o.output=value;else throw std::runtime_error("unknown radial option "+flag);
    }
    if(!std::isfinite(o.slope_strength)||o.slope_strength<0.||o.elements<4||o.order<2||o.quad_extra<1||o.positivity_subdivisions<0||!std::isfinite(o.amplitude)||o.amplitude<0.||!std::isfinite(o.alpha)||o.alpha<0.||!std::isfinite(o.dt)||o.dt<=0.||!std::isfinite(o.final_time)||o.final_time<0.||(o.source!="broken"&&o.source!="central"))throw std::runtime_error("invalid radial parameters");
    const auto start=Clock::now();const std::string checks=RadialSelfTest();
    if(o.self_test){o.elements=4;o.order=5;o.dt=1.e-5;o.final_time=3.e-5;}
    const auto setup=Clock::now();RadialProblem pr(o);const double setup_seconds=Wall(setup);auto result=RadialRun(pr);result.work.setup=setup_seconds;
    if(o.self_test&&(!result.failure.empty()||pr.max_wall_mass>1.e-13||pr.max_wall_energy>1.e-13))throw std::runtime_error("radial advancing wall gate failed");
    RadialWrite(pr,result,checks,Wall(start));return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

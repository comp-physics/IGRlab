// Test-only temporal and algebraic closure controls for moving high-order DG.
// Reuses the full-energy transport, current source, L2/SIPG or H1 assembly,
// limiter, boundary flux, exact conservative remap and serialization.
// Retained factors only precondition the CURRENT matrix. The one-correction
// branch is explicitly an approximate closure, not an equivalent solve.
#define DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY
#include "igr_hp_front_tracking_probe.cpp"
#undef DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY
#include "numerics/timestepping/ssprk_coefficients.hpp"
#include "igr_hp_l2_condensed.hpp"

namespace
{
State MovingDifference(const State& a,const State& b)
{State result=a;for(std::size_t i=0;i<a.size();++i)result[i]-=b[i];return result;}
struct MovingTimeOptions
{
  TrackOptions track;
  std::string method="ssprk3",closure="direct";
  double tolerance=1.e-11,remap_interval=0.;
  int max_corrections=100,reuse_age=0,block_elements=0,predictor_sweeps=1;
};
struct MovingTimeWork
{
  std::uint64_t factors=0,factor_refreshes=0,solves=0,matvecs=0,reductions=0;
  std::uint64_t corrections=0,max_corrections=0,fallbacks=0,reused_closures=0;
  std::uint64_t current_quadrature_actions=0,current_face_actions=0,coefficient_caches=0;
  std::uint64_t local_predictor_calls=0,local_flux_points=0,local_physical_faces=0;
  double local_predictor_seconds=0.;
  std::uint64_t factor_products=0,operator_entries=0,retained_factor_entries=0;
  double assembly=0,factor=0,solve=0,certificate=0,max_approximate_residual=0;
};
struct MovingTimeProblem:TrackingProblem
{
  MovingTimeOptions time_options;mutable MovingTimeWork tw;mutable MovingL2CondensedWork condensed_work;
  mutable State factor_edges,previous_sigma;
  mutable std::vector<std::unique_ptr<BandFactor>> factors;
  mutable std::vector<int> first;
  mutable int factor_age=0;
  explicit MovingTimeProblem(const MovingTimeOptions& o):TrackingProblem(o.track),time_options(o){}
  State Apply(const BandMatrix& a,const State& x)const
  {
    State y(a.n,0.);for(int i=0;i<a.n;++i)for(int j=std::max(0,i-a.k);j<std::min(a.n,i+a.k+1);++j){y[i]+=a.Get(i,j)*x[j];++tw.operator_entries;}
    ++tw.matvecs;return y;
  }
  static double Inner(const State& a,const State& b)
  {double value=0.;for(std::size_t i=0;i<a.size();++i)value+=a[i]*b[i];return value;}
  State Defect(const BandMatrix& a,const State& x,const State& b)const
  {State r=Apply(a,x);for(int i=0;i<a.n;++i)r[i]=b[i]-r[i];return r;}
  void Refresh(const BandMatrix& a)const
  {
    const auto start=Clock::now();factors.clear();first.clear();
    const int block=time_options.closure=="pcg"&&time_options.block_elements>0?
      time_options.block_elements*(o.closure_space=="l2"?n:p):ns;
    std::uint64_t entries=0;
    for(int begin=0;begin<ns;begin+=block)
    {first.push_back(begin);factors.emplace_back(std::make_unique<BandFactor>(a,begin,std::min(block,ns-begin)));++tw.factors;tw.factor_products+=std::uint64_t(factors.back()->products);entries+=factors.back()->l.size();}
    tw.retained_factor_entries=std::max(tw.retained_factor_entries,entries);
    factor_edges=edges;factor_age=0;++tw.factor_refreshes;tw.factor+=Wall(start);
  }
  State Precondition(const State& r)const
  {
    const auto start=Clock::now();State z(r.size());
    for(std::size_t k=0;k<factors.size();++k)
    {const int begin=first[k],size=factors[k]->n;State local(r.begin()+begin,r.begin()+begin+size);factors[k]->Solve(local,1);std::copy(local.begin(),local.end(),z.begin()+begin);++tw.solves;}
    tw.solve+=Wall(start);return z;
  }
  struct CurrentQuadrature
  {State rhoinv,face_left,face_right,b;double source_integral=0.;};
  State ApplyQuadratureRaw(const CurrentQuadrature& cache,const State& x)const
  {
    State y(ns,0.);
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];
      for(std::size_t k=0;k<weights.size();++k)
      {
        double v=0.,dv=0.;for(int i=0;i<n;++i){v+=shapes[k][i]*x[SigmaIndex(e,i)];dv+=derivatives[k][i]*x[SigmaIndex(e,i)];}
        const double coefficient=weights[k]*cache.rhoinv[e*weights.size()+k];
        for(int i=0;i<n;++i)y[SigmaIndex(e,i)]+=coefficient*(h*shapes[k][i]*v+o.base.alpha*derivatives[k][i]*dv/h);
        ++tw.current_quadrature_actions;
      }
    }
    if(o.closure_space=="l2"&&o.base.alpha!=0.)
    {
      const State dl=Shape(1.,true),dr=Shape(0.,true);
      for(int face=1;face<ne;++face)
      {
        const int el=face-1,er=face;const double al=cache.face_left[face-1],ar=cache.face_right[face-1];
        const double hl=edges[el+1]-edges[el],hr=edges[er+1]-edges[er];
        double dlx=0.,drx=0.;for(int i=0;i<n;++i){dlx+=dl[i]*x[SigmaIndex(el,i)];drx+=dr[i]*x[SigmaIndex(er,i)];}
        const double jump=x[SigmaIndex(el,p)]-x[SigmaIndex(er,0)],flux=.5*(al*dlx/hl+ar*drx/hr);
        const double penalty=o.sipg_penalty*n*n*.5*(al/hl+ar/hr);
        for(int i=0;i<n;++i){y[SigmaIndex(el,i)]-=.5*al*dl[i]*jump/hl;y[SigmaIndex(er,i)]-=.5*ar*dr[i]*jump/hr;}
        y[SigmaIndex(el,p)]+=-flux+penalty*jump;y[SigmaIndex(er,0)]+=flux-penalty*jump;++tw.current_face_actions;
      }
    }
    ++tw.matvecs;return y;
  }
  State ApplyQuadrature(const CurrentQuadrature& cache,const State& x)const
  {
    if(o.closure_boundary=="natural")return ApplyQuadratureRaw(cache,x);
    State interior=x;interior.front()=interior.back()=0.;State y=ApplyQuadratureRaw(cache,interior);y.front()=x.front();y.back()=x.back();return y;
  }
  CurrentQuadrature CacheQuadrature(const State& u,double time,TrackWork& w)const
  {
    CurrentQuadrature cache;cache.b.assign(ns,0.);cache.rhoinv.reserve(ne*weights.size());
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];
      for(std::size_t k=0;k<weights.size();++k)
      {
        ++w.source_points;const auto v=Point(u,e,shapes[k]);physics.CheckState(v.data());double du[8]={};
        for(int i=0;i<n;++i)for(int c=0;c<4;++c)du[2*c]+=derivatives[k][i]*u[Id(e,i,c)]/h;
        cache.rhoinv.push_back(igr::SafeInverseDensity(v[0],floor_value));
        const double source=igr::SourceFromConservativeStateGradient(2,v.data(),1,du,2,1,o.base.alpha,floor_value);
        if(!std::isfinite(source))throw std::runtime_error("nonfinite matrix-free current source");
        cache.source_integral+=h*weights[k]*source;
        for(int i=0;i<n;++i)cache.b[SigmaIndex(e,i)]+=h*weights[k]*shapes[k][i]*source;
      }
    }
    if(o.closure_space=="l2"&&o.base.alpha!=0.)for(int face=1;face<ne;++face)
    {cache.face_left.push_back(o.base.alpha*igr::SafeInverseDensity(u[Id(face-1,p,0)],floor_value));cache.face_right.push_back(o.base.alpha*igr::SafeInverseDensity(u[Id(face,0,0)],floor_value));++w.closure_faces;}
    if(o.closure_boundary!="natural")
    {
      State boundary(ns,0.);boundary.front()=Reference(o.left,time)[4];boundary.back()=Reference(o.right,time)[4];
      const State load=ApplyQuadratureRaw(cache,boundary);for(int i=1;i<ns-1;++i)cache.b[i]-=load[i];cache.b.front()=boundary.front();cache.b.back()=boundary.back();
    }
    w.current_source_integral=cache.source_integral;++tw.coefficient_caches;return cache;
  }
  State ClosureMatrixFree(const State& u,double time,TrackWork& w)const
  {
    const auto start=Clock::now(),assembly_start=Clock::now();const CurrentQuadrature cache=CacheQuadrature(u,time,w);tw.assembly+=Wall(assembly_start);
    const bool changed=factor_edges!=edges;
    const bool refresh=factors.empty()||changed||(time_options.reuse_age>0&&factor_age>=time_options.reuse_age);
    if(refresh)
    {const auto as=Clock::now();BandMatrix a;State b;AssembleClosure(u,time,w,a,b);tw.assembly+=Wall(as);Refresh(a);}
    else ++tw.reused_closures;
    State sigma=refresh?Precondition(cache.b):previous_sigma;
    const double denominator=std::max(Norm(cache.b),1.e-12);
    auto defect=[&](const State& s){State r=ApplyQuadrature(cache,s);for(int i=0;i<ns;++i)r[i]=cache.b[i]-r[i];return r;};
    State r=defect(sigma);double relative=Norm(r)/denominator;++tw.reductions;int count=0;
    for(;relative>time_options.tolerance&&count<time_options.max_corrections;++count)
    {const State z=Precondition(r);for(int i=0;i<ns;++i)sigma[i]+=z[i];r=defect(sigma);relative=Norm(r)/denominator;++tw.reductions;}
    tw.corrections+=count;tw.max_corrections=std::max(tw.max_corrections,std::uint64_t(count));
    if(!std::isfinite(relative)||relative>time_options.tolerance)
    {
      ++tw.fallbacks;const auto as=Clock::now();BandMatrix a;State b;AssembleClosure(u,time,w,a,b);tw.assembly+=Wall(as);Refresh(a);sigma=Precondition(b);
      r=defect(sigma);relative=Norm(r)/denominator;++tw.reductions;
      if(!std::isfinite(relative)||relative>time_options.tolerance)throw std::runtime_error("matrix-free current closure certificate failed");
    }
    tw.max_approximate_residual=std::max(tw.max_approximate_residual,relative);w.max_closure_residual=std::max(w.max_closure_residual,relative);
    previous_sigma=sigma;++factor_age;++w.closure_calls;w.closure+=Wall(start);return sigma;
  }
  State LocalPredictorResidual(const State& u,const State& frozen_sigma,TrackWork& w)const
  {
    const auto start=Clock::now();State load(u.size(),0.),r(u.size(),0.);
    // Every element reads only its own conservative polynomial and the own
    // restriction of accepted sigma. There is no neighboring state, Riemann
    // flux, coefficient construction, or closure solve in this predictor.
    for(int e=0;e<ne;++e)
    {
      for(std::size_t k=0;k<weights.size();++k)
      {
        ++w.flux_points;++tw.local_flux_points;const auto v=Point(u,e,shapes[k]);double sq=0.,speed=0.;
        for(int i=0;i<n;++i)sq+=shapes[k][i]*frozen_sigma[SigmaIndex(e,i)];
        const auto flux=physics.Flux(v.data(),sq,speed);
        for(int i=0;i<n;++i)for(int c=0;c<4;++c)load[Id(e,i,c)]+=weights[k]*derivatives[k][i]*flux[c];
      }
      for(int side=0;side<2;++side)
      {
        const int i=side?p:0;std::array<double,4> v{};for(int c=0;c<4;++c)v[c]=u[Id(e,i,c)];double speed=0.;
        const auto flux=physics.Flux(v.data(),frozen_sigma[SigmaIndex(e,i)],speed);
        for(int c=0;c<4;++c)load[Id(e,i,c)]+=(side?-1.:1.)*flux[c];++tw.local_physical_faces;
      }
      for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int c=0;c<4;++c)r[Id(e,i,c)]+=mass_inverse(i,j)*load[Id(e,j,c)]/(edges[e+1]-edges[e]);
    }
    ++tw.local_predictor_calls;const double seconds=Wall(start);tw.local_predictor_seconds+=seconds;w.transport+=seconds;return r;
  }
  State Closure(const State& u,double time,TrackWork& w)const
  {
    if(time_options.closure=="matrixfree")return ClosureMatrixFree(u,time,w);
    const auto start=Clock::now();BandMatrix a(ns,ClosureBandwidth());State b(ns,0.);
    const auto assembly_start=Clock::now();AssembleClosure(u,time,w,a,b);tw.assembly+=Wall(assembly_start);
    if(time_options.closure=="condensed")
    {
      State sigma=SolveMovingL2Condensed(a,b,ne,p,condensed_work);
      if(condensed_work.last_residual>time_options.tolerance)throw std::runtime_error("condensed closure requested tolerance failed");
      w.max_closure_residual=std::max(w.max_closure_residual,condensed_work.last_residual);
      ++w.closure_calls;w.closure+=Wall(start);return sigma;
    }
    const bool direct=time_options.closure=="direct";
    const bool changed=factor_edges!=edges;
    const bool refresh=direct||factors.empty()||changed||(time_options.reuse_age>0&&factor_age>=time_options.reuse_age);
    if(refresh)Refresh(a);else ++tw.reused_closures;
    const double bnorm=std::max(Norm(b),1.e-12);State sigma;
    int corrections=0;
    if(direct||refresh&&time_options.block_elements==0)
      sigma=Precondition(b);
    else
    {
      sigma=previous_sigma.size()==b.size()&&!changed?previous_sigma:State(ns,0.);
      State r=Defect(a,sigma,b);
      double relative=Norm(r)/bnorm;tw.reductions+=2;
      if(time_options.closure=="pcg")
      {
        State z=Precondition(r),direction=z;double rz=Inner(r,z);++tw.reductions;
        for(;corrections<time_options.max_corrections&&relative>time_options.tolerance;++corrections)
        {
          const State ad=Apply(a,direction);const double denominator=Inner(direction,ad);++tw.reductions;
          if(!(denominator>0.)||!std::isfinite(denominator))break;
          const double step=rz/denominator;
          for(int i=0;i<ns;++i){sigma[i]+=step*direction[i];r[i]-=step*ad[i];}
          // Recompute true residual each iteration: no recursively drifting
          // certificate is allowed to silently qualify the current equation.
          r=Defect(a,sigma,b);relative=Norm(r)/bnorm;++tw.reductions;
          if(relative<=time_options.tolerance){++corrections;break;}
          z=Precondition(r);const double next=Inner(r,z);++tw.reductions;
          if(!(rz>0.)||!std::isfinite(next))break;
          for(int i=0;i<ns;++i)direction[i]=z[i]+next/rz*direction[i];rz=next;
        }
      }
      else
      {
        const int cap=time_options.closure=="one-correction"?1:time_options.max_corrections;
        for(;corrections<cap&&relative>time_options.tolerance;++corrections)
        {const State z=Precondition(r);for(int i=0;i<ns;++i)sigma[i]+=z[i];r=Defect(a,sigma,b);relative=Norm(r)/bnorm;++tw.reductions;}
      }
    }
    tw.corrections+=corrections;tw.max_corrections=std::max(tw.max_corrections,std::uint64_t(corrections));
    const auto certificate_start=Clock::now();double relative=Norm(Defect(a,sigma,b))/bnorm;++tw.reductions;
    if(time_options.closure!="one-correction"&&(!std::isfinite(relative)||relative>time_options.tolerance))
    {
      // Charged direct fallback also replaces the full retained factor. For
      // local-block PCG it does not turn subsequent preconditioning global.
      ++tw.fallbacks;const auto fs=Clock::now();BandFactor exact(a,0,ns);++tw.factors;tw.factor_products+=std::uint64_t(exact.products);tw.factor+=Wall(fs);
      const auto ss=Clock::now();sigma=b;exact.Solve(sigma,1);++tw.solves;tw.solve+=Wall(ss);
      relative=Norm(Defect(a,sigma,b))/bnorm;++tw.reductions;
      if(!std::isfinite(relative)||relative>time_options.tolerance)throw std::runtime_error("moving-time current closure certificate failed");
    }
    tw.certificate+=Wall(certificate_start);
    tw.max_approximate_residual=std::max(tw.max_approximate_residual,relative);
    if(!std::isfinite(relative))throw std::runtime_error("moving-time nonfinite approximate closure");
    w.max_closure_residual=std::max(w.max_closure_residual,relative);
    previous_sigma=sigma;++factor_age;++w.closure_calls;w.closure+=Wall(start);return sigma;
  }
};

template<class Coefficients>
State MovingTableauStep(MovingTimeProblem& pr,const State& original,const State& endpoint,
                       double time,double dt,TrackWork& work,std::array<double,4>& integral)
{
  constexpr int stages=int(Coefficients::stage_count);
  std::array<State,stages> rhs;
  for(int k=0;k<stages;++k)
  {
    State stage=original;
    for(int j=0;j<k;++j)for(std::size_t i=0;i<stage.size();++i)stage[i]+=dt*Coefficients::a[k][j]*rhs[j][i];
    if(k)pr.Limit(stage,work);
    const double t=time+dt*Coefficients::c[k];
    const State sigma=k?pr.Closure(stage,t,work):endpoint;
    std::array<double,4> budget{};rhs[k]=pr.Residual(stage,sigma,t,work,budget);
    for(int c=0;c<4;++c)integral[c]+=dt*Coefficients::b[k]*budget[c];
  }
  State next=original;for(int k=0;k<stages;++k)for(std::size_t i=0;i<next.size();++i)next[i]+=dt*Coefficients::b[k]*rhs[k][i];
  pr.Limit(next,work);return next;
}
State MovingMidpointStep(MovingTimeProblem& pr,const State& original,const State& endpoint,
                         double time,double dt,TrackWork& work,std::array<double,4>& integral)
{
  const bool local=pr.time_options.method=="local-midpoint";
  State stage=original;const int sweeps=local?pr.time_options.predictor_sweeps:1;
  for(int sweep=0;sweep<sweeps;++sweep)
  {
    std::array<double,4> discarded{};
    const State rhs=local?pr.LocalPredictorResidual(stage,endpoint,work):pr.Residual(original,endpoint,time,work,discarded);
    for(std::size_t i=0;i<stage.size();++i)stage[i]=original[i]+.5*dt*rhs[i];pr.Limit(stage,work);
  }
  // The globally shared corrector restores conservative coupling and solves
  // the complete current IGR constraint at the predicted midpoint. It does
  // not turn the local predictor into exact semidiscrete RK2 or full ADER.
  const State sigma=pr.Closure(stage,time+.5*dt,work);std::array<double,4> budget{};
  const State rhs=pr.Residual(stage,sigma,time+.5*dt,work,budget);State next=original;
  for(std::size_t i=0;i<next.size();++i)next[i]+=dt*rhs[i];pr.Limit(next,work);
  for(int c=0;c<4;++c)integral[c]=dt*budget[c];return next;
}
TrackResult MovingTimeRun(MovingTimeProblem& pr)
{
  TrackResult out;const auto start=Clock::now();out.initial_center=pr.edges[pr.patch_index];out.last_center=out.initial_center;
  double next_remap=pr.time_options.remap_interval;
  try
  {
    out.u=pr.Initial();pr.Limit(out.u,out.work);out.initial_integrals=pr.Integrals(out.u,pr.edges);out.sigma=pr.Closure(out.u,0.,out.work);
    while(out.time<pr.o.base.final_time-1.e-14)
    {
      ++out.work.attempted;double dt=std::min(pr.o.base.dt,pr.o.base.final_time-out.time);
      if(pr.time_options.remap_interval>0.)dt=std::min(dt,next_remap-out.time);
      if(!(dt>0.))throw std::runtime_error("invalid scheduled remap interval");std::array<double,4> budget{};State next;
      if(pr.time_options.method=="local-midpoint"||pr.time_options.method=="global-midpoint")next=MovingMidpointStep(pr,out.u,out.sigma,out.time,dt,out.work,budget);
      else if(pr.time_options.method=="ssprk3")next=MovingTableauStep<mfem::SSPRK33Coefficients>(pr,out.u,out.sigma,out.time,dt,out.work,budget);
      else if(pr.time_options.method=="rk4")next=MovingTableauStep<mfem::ResearchRK4Coefficients>(pr,out.u,out.sigma,out.time,dt,out.work,budget);
      else next=MovingTableauStep<mfem::SSPRK104Coefficients>(pr,out.u,out.sigma,out.time,dt,out.work,budget);
      out.u=std::move(next);out.time+=dt;++out.work.steps;
      for(int c=0;c<4;++c)out.boundary_budget[c]+=budget[c];
      out.sigma=pr.Closure(out.u,out.time,out.work);
      const bool scheduled=pr.time_options.remap_interval>0.?out.time>=next_remap-1.e-13:out.work.steps%pr.o.remap_every==0;
      if(scheduled&&pr.time_options.remap_interval>0.){do{next_remap+=pr.time_options.remap_interval;}while(next_remap<=out.time+1.e-13);}
      if((pr.o.mode=="oracle"||pr.o.mode=="detected"||pr.o.mode=="roundtrip"||pr.o.mode=="roundtrip-uniform")&&scheduled)
      {
        const double center=pr.o.mode=="oracle"?pr.o.center+pr.profile.speed*out.time:pr.Detect(out.sigma,out.work);
        const State next_mesh=pr.Mesh(center,out.u,out.work);double movement=0.;for(int e=0;e<=pr.ne;++e)movement=std::max(movement,std::abs(next_mesh[e]-pr.edges[e]));
        if(movement>1.e-14)
        {
          const double before=out.work.current_source_integral;const State original_edges=pr.edges;
          out.u=pr.Remap(out.u,next_mesh,out.work);
          if(pr.o.mode=="roundtrip"||pr.o.mode=="roundtrip-uniform")out.u=pr.Remap(out.u,original_edges,out.work);
          pr.Limit(out.u,out.work);out.sigma=pr.Closure(out.u,out.time,out.work);
          const double change=out.work.current_source_integral-before;
          out.work.max_remap_source_relative_change=std::max(out.work.max_remap_source_relative_change,std::abs(change)/std::max(std::abs(before),1.e-20));
          out.work.signed_remap_source_change+=change;out.work.absolute_remap_source_change+=std::abs(change);
        }
        out.last_center=pr.o.mesh_features=="two"?center:pr.edges[pr.patch_index];
      }
    }
  }
  catch(const std::exception& e){out.failure=e.what();}
  if(!out.u.empty()){const auto now=pr.Integrals(out.u,pr.edges);for(int c=0;c<4;++c)out.conservation_defect[c]=now[c]-out.initial_integrals[c]-out.boundary_budget[c];}
  out.wall=Wall(start);return out;
}
MovingTimeOptions MovingTimeParse(int argc,char** argv)
{
  MovingTimeOptions o;std::vector<char*> base{argv[0]};
  for(int i=1;i<argc;++i)
  {
    const std::string flag=argv[i];
    if(flag=="--time-method"||flag=="--closure-method"||flag=="--solve-tolerance"||flag=="--max-corrections"||flag=="--reuse-age"||flag=="--block-elements"||flag=="--remap-interval"||flag=="--predictor-sweeps")
    {
      if(i+1==argc)throw std::runtime_error("missing value for "+flag);const std::string value=argv[++i];
      if(flag=="--time-method")o.method=value;else if(flag=="--closure-method")o.closure=value;else if(flag=="--solve-tolerance")o.tolerance=std::stod(value);else if(flag=="--max-corrections")o.max_corrections=std::stoi(value);else if(flag=="--reuse-age")o.reuse_age=std::stoi(value);else if(flag=="--remap-interval")o.remap_interval=std::stod(value);else if(flag=="--predictor-sweeps")o.predictor_sweeps=std::stoi(value);else o.block_elements=std::stoi(value);
    }
    else base.push_back(argv[i]);
  }
  o.track=TrackParse(int(base.size()),base.data());
  if(o.method!="ssprk3"&&o.method!="rk4"&&o.method!="ssprk104"&&o.method!="local-midpoint"&&o.method!="global-midpoint")throw std::runtime_error("invalid time method");
  if(o.closure!="direct"&&o.closure!="reuse"&&o.closure!="one-correction"&&o.closure!="pcg"&&o.closure!="matrixfree"&&o.closure!="condensed")throw std::runtime_error("invalid closure method");
  if(!(o.tolerance>0.)||o.max_corrections<1||o.reuse_age<0||o.block_elements<0||o.remap_interval<0.||o.predictor_sweeps<1||o.predictor_sweeps>4)throw std::runtime_error("invalid closure controls");
  if(o.closure=="condensed"&&o.track.closure_space!="l2")throw std::runtime_error("condensed closure requires L2");
  if(o.block_elements&&o.closure!="pcg")throw std::runtime_error("local blocks require pcg");return o;
}
void MovingTimeWrite(const MovingTimeProblem& pr,const TrackResult& result,std::ostream& output)
{
  std::ostringstream base;TrackWrite(pr,result,base);std::string text=base.str();const auto last=text.find_last_of('}');text.resize(last);
  const auto& w=pr.tw;output<<text<<std::setprecision(17)<<",\n\"time_method\":\""<<pr.time_options.method<<"\",\"closure_method\":\""<<pr.time_options.closure<<"\",\"closure_equivalence\":\""<<(pr.time_options.closure=="one-correction"?"approximate_current_closure":"certified_current_operator")<<"\",\"solve_tolerance\":"<<pr.time_options.tolerance<<",\"predictor_sweeps\":"<<pr.time_options.predictor_sweeps<<",\"remap_interval\":"<<pr.time_options.remap_interval<<",\"reuse_age\":"<<pr.time_options.reuse_age<<",\"block_elements\":"<<pr.time_options.block_elements<<",\"max_corrections\":"<<pr.time_options.max_corrections;
  output<<",\n\"temporal_work\":{\"factors\":"<<w.factors<<",\"factor_refreshes\":"<<w.factor_refreshes<<",\"solves\":"<<w.solves<<",\"current_matvecs\":"<<w.matvecs<<",\"scalar_reductions\":"<<w.reductions<<",\"corrections\":"<<w.corrections<<",\"maximum_corrections\":"<<w.max_corrections<<",\"fallbacks\":"<<w.fallbacks<<",\"reused_closures\":"<<w.reused_closures<<",\"factor_multiply_subtracts\":"<<w.factor_products<<",\"local_predictor_calls\":"<<w.local_predictor_calls<<",\"local_flux_points\":"<<w.local_flux_points<<",\"local_physical_faces\":"<<w.local_physical_faces<<",\"local_predictor_seconds\":"<<w.local_predictor_seconds<<",\"current_quadrature_actions\":"<<w.current_quadrature_actions<<",\"current_face_actions\":"<<w.current_face_actions<<",\"coefficient_caches\":"<<w.coefficient_caches<<",\"current_matrix_entries_visited\":"<<w.operator_entries<<",\"retained_factor_entries\":"<<w.retained_factor_entries<<",\"assembly_seconds\":"<<w.assembly<<",\"factor_seconds\":"<<w.factor<<",\"solve_seconds\":"<<w.solve<<",\"certificate_seconds\":"<<w.certificate<<",\"maximum_residual\":"<<w.max_approximate_residual<<"},\n";
  const auto& cw=pr.condensed_work;output<<"\"condensed_work\":{\"calls\":"<<cw.calls<<",\"local_factors\":"<<cw.local_factors<<",\"trace_factors\":"<<cw.trace_factors<<",\"face_response_columns\":"<<cw.response_columns<<",\"source_columns\":"<<cw.source_columns<<",\"local_factor_products\":"<<cw.local_factor_products<<",\"response_products\":"<<cw.response_products<<",\"trace_factor_products\":"<<cw.trace_factor_products<<",\"schur_products\":"<<cw.schur_products<<",\"recovery_products\":"<<cw.recovery_products<<",\"peak_factor_entries\":"<<cw.peak_factor_entries<<",\"peak_response_entries\":"<<cw.peak_response_entries<<",\"max_residual\":"<<cw.max_residual<<",\"seconds\":"<<cw.seconds<<"}\n}\n";
}
void MovingTimeSelfTest()
{
  const char* raw_options[]={"moving-time","--remap-interval",".00016","--predictor-sweeps","2","--time-method","local-midpoint"};
  char* parser_options[7];for(int i=0;i<7;++i)parser_options[i]=const_cast<char*>(raw_options[i]);const auto parsed=MovingTimeParse(7,parser_options);
  if(parsed.predictor_sweeps!=2||std::abs(parsed.remap_interval-.00016)>1.e-15)throw std::runtime_error("local predictor and remap options interfere");
  MovingL2CondensedSelfTest();
  for(const std::string space:{"h1","l2"})for(const std::string boundary:{"dirichlet","natural"})
  {
    MovingTimeOptions o;o.track.initial="entropy";o.track.closure_space=space;o.track.closure_boundary=boundary;o.track.base.elements=8;o.track.base.order=3;o.track.left=0.;o.track.right=1.;o.track.center=.51;o.track.base.dt=.0005;o.track.base.final_time=.004;o.closure="reuse";
    MovingTimeProblem pr(o);State u=pr.Initial();
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];const double v=.2+.03*std::sin(6.*x);u[pr.Id(e,i,1)]=u[pr.Id(e,i,0)]*v;u[pr.Id(e,i,3)]=3.+.5*u[pr.Id(e,i,0)]*v*v;}
    State constant(u.size());for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)constant[pr.Id(e,i,c)]=c==0?2.:(c==1?.4:(c==2?0.:3.04));TrackWork local_work;
    if(Norm(pr.LocalPredictorResidual(constant,State(pr.ns,0.),local_work))>1.e-10)throw std::runtime_error("element-local constant predictor failed");
    TrackWork operator_work;const auto cache=pr.CacheQuadrature(u,0.,operator_work);BandMatrix assembled;State assembled_b;pr.AssembleClosure(u,0.,operator_work,assembled,assembled_b);
    State test_sigma(pr.ns);for(int i=0;i<pr.ns;++i)test_sigma[i]=std::sin(.31*i)+.03*i;
    const State explicit_action=pr.Apply(assembled,test_sigma),implicit_action=pr.ApplyQuadrature(cache,test_sigma);
    if(Norm(MovingDifference(explicit_action,implicit_action))/std::max(Norm(explicit_action),1.e-12)>2.e-12||Norm(MovingDifference(cache.b,assembled_b))>1.e-12)throw std::runtime_error("matrix-free current quadrature/face operator mismatch");
    TrackWork work;pr.Closure(u,0.,work);
    for(int trial=0;trial<3;++trial)
    {
      for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)u[pr.Id(e,i,0)]*=1.+.001*(trial+1)*std::sin(double(e+i));
      const State approximate=pr.Closure(u,0.,work);TrackWork independent;const State exact=pr.TrackingProblem::Closure(u,0.,independent);
      if(Norm(MovingDifference(approximate,exact))/std::max(Norm(exact),1.e-12)>2.e-9)throw std::runtime_error("retained current operator mismatch");
    }
    if(pr.tw.factor_refreshes!=1||pr.tw.reused_closures!=3||pr.tw.fallbacks)throw std::runtime_error("factor reuse accounting failed");
    o.closure="matrixfree";MovingTimeProblem matrixfree(o);TrackWork mw;const State ms=matrixfree.Closure(u,0.,mw);TrackWork dw;const State ds=matrixfree.TrackingProblem::Closure(u,0.,dw);
    if(Norm(MovingDifference(ms,ds))/std::max(Norm(ds),1.e-12)>2.e-9)throw std::runtime_error("matrix-free initial closure mismatch");
    o.closure="direct";MovingTimeProblem base(o);TrackingProblem ordinary(o.track);const auto a=MovingTimeRun(base),b=TrackRun(ordinary);
    if(!a.failure.empty()||!b.failure.empty()||Norm(MovingDifference(a.u,b.u))>1.e-11)throw std::runtime_error("SSPRK3 shared transport mismatch");
    // A common physical-time schedule must not remap a shortened final
    // interval; its final accepted horizon remains unchanged.
    o.remap_interval=.0015;o.track.mode="detected";o.track.base.final_time=.004;MovingTimeProblem scheduled(o);const auto scheduled_result=MovingTimeRun(scheduled);
    if(!scheduled_result.failure.empty()||std::abs(scheduled_result.time-.004)>1.e-14||scheduled_result.work.remaps>2)throw std::runtime_error("physical remap schedule failed");
    o.remap_interval=0.;o.track.mode="uniform";
    for(const std::string method:{"rk4","ssprk104","global-midpoint","local-midpoint"}){o.method=method;o.predictor_sweeps=2;MovingTimeProblem higher(o);const auto r=MovingTimeRun(higher);if(!r.failure.empty()||r.work.max_rhs_defect>1.e-9)throw std::runtime_error("temporal conservation/admission failed");for(double defect:r.conservation_defect)if(std::abs(defect)>1.e-10)throw std::runtime_error("temporal accepted flux budget failed");if(method=="local-midpoint"&&higher.tw.local_predictor_calls!=2*r.work.steps)throw std::runtime_error("local Picard sweep count failed");}
  }
  std::cout<<"moving-time self-test passed: H1/L2 current factor correction, SSPRK3 equivalence, RK4/SSPRK104 flux conservation\n";
}
}
int main(int argc,char** argv)
{
  try
  {
    mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");const auto options=MovingTimeParse(argc,argv);
    if(options.track.self_test){MovingTimeSelfTest();return 0;}
    const auto setup=Clock::now();MovingTimeProblem problem(options);const double setup_seconds=Wall(setup);auto result=MovingTimeRun(problem);result.work.setup=setup_seconds;result.wall+=setup_seconds;
    if(options.track.output.empty())MovingTimeWrite(problem,result,std::cout);else{std::ofstream output(options.track.output);if(!output)throw std::runtime_error("cannot open moving-time output");MovingTimeWrite(problem,result,output);}
    return result.failure.empty()?0:2;
  }
  catch(const std::exception& error){std::cerr<<"moving-time probe failed: "<<error.what()<<'\n';return 2;}
}

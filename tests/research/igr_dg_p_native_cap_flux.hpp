#pragma once
// Research-only paired physical-stress cap. Raw stress remains in the source
// and elliptic matrix. This direct weak operator evaluates the nonlinear cap
// at quadrature and each face; it never projects the physical stress first.
struct NativePhysicalCap
{
  double identity=12.,bound=20.;
  mutable std::uint64_t evaluations=0,active=0,residual_calls=0;
  mutable double seconds=0.,minimum_ratio=INFINITY,maximum_ratio=-INFINITY;
  mutable double minimum_physical_ratio=INFINITY,maximum_physical_ratio=-INFINITY;
  mutable double minimum_feedback=INFINITY,minimum_sound_squared=INFINITY;
  struct Value {double pressure,physical,psi,derivative,feedback,sound_squared;};
  Value Evaluate(const double* u,double raw)const
  {
    const double pressure=(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
    if(!(u[0]>floor_value)||!(pressure>floor_value)||!std::isfinite(raw))
      throw std::runtime_error("physical cap requires positive density/thermal pressure and finite raw stress");
    const double ratio=raw/pressure,z=std::abs(ratio);double psi=ratio,derivative=1.;
    if(z>identity)
    {
      const double t=z-identity,b=bound-identity,exponent=std::exp(-b/t),den=1.+t/b*exponent;
      psi=std::copysign(identity+t/den,ratio);derivative=-std::expm1(-b/t)/(den*den);++active;
    }
    const double physical=pressure*psi,feedback=1.+psi-ratio*derivative;
    const double sound_squared=feedback*(gamma_euler*pressure+(gamma_euler-1.)*physical)/u[0];
    if(!std::isfinite(sound_squared)||!(feedback>0.)||!(sound_squared>0.))
      throw std::runtime_error("physical cap lost acoustic/thermal feedback admission");
    ++evaluations;minimum_ratio=std::min(minimum_ratio,ratio);maximum_ratio=std::max(maximum_ratio,ratio);
    minimum_physical_ratio=std::min(minimum_physical_ratio,psi);maximum_physical_ratio=std::max(maximum_physical_ratio,psi);
    minimum_feedback=std::min(minimum_feedback,feedback);minimum_sound_squared=std::min(minimum_sound_squared,sound_squared);
    return {pressure,physical,psi,derivative,feedback,sound_squared};
  }
  template<class Problem>
  std::array<double,4> Flux(const Problem& pr,const double* u,double raw,double& speed)const
  {
    const auto value=Evaluate(u,raw);double old_speed=0.;
    const auto flux=pr.physics.Flux(u,value.physical,old_speed);
    speed=std::abs(u[1]/u[0])+std::sqrt(value.sound_squared);return flux;
  }
  template<class Problem>
  void Admit(const Problem& pr,const State& physical,const State& raw)const
  {
    const auto start=Clock::now();
    for(int e=0;e<pr.ne;++e)for(const State& shape:pr.positive)
    {const auto u=pr.Point(physical,e,shape);double s=0.;for(int i=0;i<pr.n;++i)s+=shape[i]*raw[e*pr.n+i];Evaluate(u.data(),s);}
    seconds+=Wall(start);
  }
  template<class Problem>
  std::array<double,4> BoundaryFlux(const Problem& pr,const State& physical,const State& raw,int side)const
  {
    std::array<double,4> flux{};
    const int e=side?pr.ne-1:0,node=side?pr.p:0;const double normal[2]={side?1.:-1.,0.};
    std::array<double,4> inner{},outer{};for(int c=0;c<4;++c)inner[c]=physical[pr.Id(e,node,c)];
    using Kind=mfem::detail::EulerBoundaryStateKind;
    const bool wall=(side?pr.boundary.right:pr.boundary.left)=="wall";
    if(!mfem::detail::EvaluateEulerBoundaryState(wall?Kind::slip_wall:Kind::supersonic_outlet,
         2,4,gamma_euler,1.,inner.data(),normal,inner.data(),outer.data(),floor_value,floor_value))
      throw std::runtime_error("capped shared Euler wall/outflow ghost rejected");
    const double s=raw[pr.SigmaIndex(e,node)];double ai=0.,ao=0.;
    const auto fi=Flux(pr,inner.data(),s,ai),fo=Flux(pr,outer.data(),s,ao);const double speed=pr.physics.FaceSpeed(ai,ao);
    for(int c=0;c<4;++c)flux[c]=.5*(fi[c]+fo[c]-normal[0]*speed*(outer[c]-inner[c]));
    if(wall)
    {pr.max_wall_mass=std::max(pr.max_wall_mass,std::abs(flux[0]));pr.max_wall_energy=std::max(pr.max_wall_energy,std::abs(flux[3]));
     pr.max_wall_velocity=std::max(pr.max_wall_velocity,std::abs(inner[1]/inner[0]));}
    ++pr.boundary_calls;return flux;
  }
  template<class Problem>
  State Residual(const Problem& pr,const State& physical,const State& raw,TrackWork& work,std::array<double,4>& budget)const
  {
    const auto start=Clock::now();const int m=pr.retained.n,n=pr.n,ne=pr.ne;
    State load(ne*m*4,0.),low(load.size(),0.);
    for(int e=0;e<ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
    {
      const auto u=pr.Point(physical,e,pr.shapes[q]);double s=0.,speed=0.;
      for(int i=0;i<n;++i)s+=pr.shapes[q][i]*raw[e*n+i];
      const auto flux=Flux(pr,u.data(),s,speed);++work.flux_points;
      for(int i=0;i<m;++i)for(int c=0;c<4;++c)load[(e*m+i)*4+c]+=pr.weights[q]*pr.low_derivatives[q][i]*flux[c];
    }
    budget.fill(0.);
    for(int face=0;face<=ne;++face)
    {
      std::array<double,4> flux{};
      if(face==0||face==ne)
      {
        flux=BoundaryFlux(pr,physical,raw,face==ne);work.flux_points+=2;
      }
      else
      {
        const double* ul=&physical[pr.Id(face-1,pr.p,0)],*ur=&physical[pr.Id(face,0,0)];double al=0.,ar=0.;
        const auto fl=Flux(pr,ul,raw[pr.SigmaIndex(face-1,pr.p)],al),fr=Flux(pr,ur,raw[pr.SigmaIndex(face,0)],ar);
        const double speed=pr.physics.FaceSpeed(al,ar);
        for(int c=0;c<4;++c)flux[c]=.5*(fl[c]+fr[c]-speed*(ur[c]-ul[c]));work.flux_points+=2;
      }
      ++work.faces;
      for(int c=0;c<4;++c)
      {if(face>0)load[((face-1)*m+m-1)*4+c]-=flux[c];if(face<ne)load[(face*m)*4+c]+=flux[c];
       if(face==0)budget[c]=flux[c];if(face==ne)budget[c]-=flux[c];}
    }
    for(int e=0;e<ne;++e)for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<4;++c)
      low[(e*m+i)*4+c]+=pr.retained.mass_inverse(i,j)*load[(e*m+j)*4+c]/(pr.edges[e+1]-pr.edges[e]);
    State result=pr.state_transfer.Apply(pr.state_transfer.E,low,ne,4);const auto totals=pr.Integrals(result,pr.edges);
    for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    const double elapsed=Wall(start);seconds+=elapsed;work.transport+=elapsed;++residual_calls;return result;
  }
};

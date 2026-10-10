#pragma once
#include <functional>
// Mixed weak-DG expansive-work limiter. Sigma=S-T(G_c u). The velocity gradient
// uses central traces; the traction trace uses the Pp Riesz projection of T.
// This makes the isolated momentum correction exactly -G_c^* T. Energy uses
// the paired volume u*T and face uhat*That. The BR1 part is semidefinite, not
// coercive; ordinary conservative LLF remains and its speed can increase.
struct NativeWorkCap
{
  double identity=4.,bound=6.,tolerance=1.e-9,diffusion_cfl_limit=1.;
  mutable std::uint64_t calls=0,points=0,intervals=0,maximum_depth=0;
  mutable double seconds=0.,maximum_pairing_error=0.,maximum_energy_budget_error=0.;
  mutable double maximum_diffusion_bound=0.,maximum_diffusion_cfl=0.,maximum_mu=0.;
  mutable double minimum_physical=INFINITY,minimum_pressure=INFINITY,maximum_work_ratio=0.;
  mutable double last_heat=0.,last_rate_bound=0.,maximum_sound=0.;
  mutable State cached_physical,cached_raw,cell_speed,face_traction,face_velocity;
  struct Value{double T,mu,physical,speed;};
  Value Evaluate(double rho,double u,double p,double S,double g,double alpha)const
  {
    if(!(rho>floor_value)||!(p>floor_value)||!(S>=0.)||!(alpha>0.))
      throw std::runtime_error("work-cap positive-rho/p/raw-S branch required");
    const double q=gamma_euler*p+(gamma_euler-1.)*S,tau=std::sqrt(alpha*rho/q),z=tau*S*g/p;
    double sigma=S,mu=0.,ceff=q/rho;
    if(z>identity)
    {
      const double t=z-identity,b=bound-identity,a=std::exp(-b/t),den=1.+t/b*a;
      const double psi=identity+t/den,dpsi=-std::expm1(-b/t)/(den*den);
      const double chi=(psi-z*dpsi)/(tau*g),sp=chi*(1.+gamma_euler*p/(2.*q));
      sigma=p*psi/(tau*g);mu=p*chi/g;
      const double qe=gamma_euler*p+(gamma_euler-1.)*S*dpsi;
      ceff=-p*chi/(2.*rho)+(1.+sp)*qe/rho;
      maximum_work_ratio=std::max(maximum_work_ratio,psi);
    }
    if(!std::isfinite(sigma+mu+ceff)||sigma<0.||mu<0.||!(ceff>0.))
      throw std::runtime_error("work cap physical/principal admission failed");
    minimum_physical=std::min(minimum_physical,sigma);minimum_pressure=std::min(minimum_pressure,p);
    maximum_mu=std::max(maximum_mu,mu);maximum_sound=std::max(maximum_sound,std::sqrt(ceff));
    // Inactive cells must preserve the existing face-only LLF exactly. Only
    // active mixed states request an extra principal-speed envelope.
    return {S-sigma,mu,sigma,z>identity?std::abs(u)+std::sqrt(ceff):0.};
  }
  template<class Problem>
  void Add(const Problem& pr,const State& physical,const State& raw,State& rhs,TrackWork& work,std::array<double,4>& budget)const
  {
    const auto start=Clock::now();const int m=pr.retained.n,ne=pr.ne;
    const auto& M=pr.retained.mass;const auto& MI=pr.retained.mass_inverse;const auto& D=pr.retained.physics.basis.D;
    std::vector<State> primitive(ne,State(5*m)),g(ne,State(m)),load(ne,State(2*m+m*m,0.)),Tpoly(ne,State(m));
    State h(ne),rho_min(ne,INFINITY),speed(ne,0.),local_trace(ne,0.);
    const State lowS=pr.stress_transfer.Apply(pr.stress_transfer.P,raw,ne,1);
    for(int e=0;e<ne;++e)
    {
      h[e]=pr.edges[e+1]-pr.edges[e];
      for(int i=0;i<m;++i)
      {
        const auto u=pr.Point(physical,e,pr.primitive_nodes[i]);
        primitive[e][i]=u[0];primitive[e][m+i]=u[1]/u[0];
        primitive[e][2*m+i]=(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
        primitive[e][3*m+i]=lowS[e*m+i];
      }
      // Same positive quadrature defining the exact polynomial density mass.
      for(const State& b:pr.low_shapes)
      {double rho=0.;for(int i=0;i<m;++i)rho+=b[i]*primitive[e][i];rho_min[e]=std::min(rho_min[e],rho);}
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)g[e][i]+=D(i,j)*primitive[e][m+j]/h[e];
    }
    // Three sparse blocks of G per output element, in local nodal coordinates.
    std::vector<std::array<State,3>> G(ne);
    for(int e=0;e<ne;++e)
    {
      for(auto& block:G[e])block.assign(m*m,0.);
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)G[e][1][i*m+j]=D(i,j)/h[e];
      for(int side=0;side<2;++side)
      {
        const int neighbor=e+(side?1:-1),node=side?m-1:0,other=side?0:m-1;
        const double normal=side?1.:-1.,own=primitive[e][m+node];
        const bool wall=neighbor<0||neighbor>=ne;const double outside=wall?-own:primitive[neighbor][m+other];
        const double factor=wall?1.:.5;
        for(int i=0;i<m;++i)
        {
          const double lift=MI(i,node)/h[e];g[e][i]+=.5*normal*lift*(outside-own);
          G[e][1][i*m+node]-=factor*normal*lift;
          if(!wall)G[e][side?2:0][i*m+other]+=.5*normal*lift;
        }
      }
      auto rule=[&](double left,double right,int nq)
      {
        State out(2*m+m*m,0.);const auto& q=mfem::IntRules.Get(mfem::Geometry::SEGMENT,2*nq-1);
        for(int k=0;k<q.GetNPoints();++k)
        {
          const auto& point=q.IntPoint(k);const double z=left+(right-left)*point.x;
          const State b=pr.retained.Shape(z,false);double rho=0.,u=0.,p=0.,S=0.,gg=0.;
          for(int i=0;i<m;++i){rho+=b[i]*primitive[e][i];u+=b[i]*primitive[e][m+i];p+=b[i]*primitive[e][2*m+i];S+=b[i]*primitive[e][3*m+i];gg+=b[i]*g[e][i];}
          const auto value=Evaluate(rho,u,p,S,gg,pr.o.base.alpha);const double weight=h[e]*(right-left)*point.weight;
          speed[e]=std::max(speed[e],value.speed);++points;
          for(int i=0;i<m;++i)
          {
            out[i]+=weight*b[i]*value.T;out[m+i]+=weight*b[i]*u*value.T;
            for(int j=0;j<m;++j)out[2*m+i*m+j]+=weight*b[i]*b[j]*value.mu;
          }
        }
        return out;
      };
      std::function<State(double,double,int)> integrate=[&](double a,double b,int depth)
      {
        const auto coarse=rule(a,b,8),fine=rule(a,b,16);double error=0.,scale=0.;
        for(std::size_t i=0;i<fine.size();++i){error=std::max(error,std::abs(fine[i]-coarse[i]));scale=std::max(scale,std::abs(fine[i]));}
        ++intervals;maximum_depth=std::max(maximum_depth,std::uint64_t(depth));
        if(error<=tolerance*((b-a)*h[e]+scale))return fine;
        if(depth>=16)throw std::runtime_error("work-cap adaptive correction quadrature did not converge");
        State result=integrate(a,.5*(a+b),depth+1);const State right=integrate(.5*(a+b),b,depth+1);
        for(std::size_t i=0;i<result.size();++i)result[i]+=right[i];return result;
      };
      load[e]=integrate(0.,1.,0);
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)Tpoly[e][i]+=MI(i,j)*load[e][j]/h[e];
      // Upper bound on each patch's H_rho-normalized PSD diffusion matrix.
      // H_rho^{-1} <= M_ref^{-1}/(h*rho_min) in the defining quadrature.
      for(int block=0;block<3;++block)
      {
        const int f=e+block-1;if(f<0||f>=ne)continue;const auto& B=G[e][block];
        State NB(m*m,0.);
        for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)NB[i*m+j]+=load[e][2*m+i*m+k]*B[k*m+j];
        for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)
          local_trace[e]+=MI(i,j)*B[k*m+j]*NB[k*m+i]/(h[f]*rho_min[f]);
      }
    }
    double rate=0.;
    for(int f=0;f<ne;++f)
    {double overlap=0.;for(int e=std::max(0,f-1);e<=std::min(ne-1,f+1);++e)overlap+=local_trace[e];rate=std::max(rate,overlap);}
    last_rate_bound=rate;maximum_diffusion_bound=std::max(maximum_diffusion_bound,rate);
    maximum_diffusion_cfl=std::max(maximum_diffusion_cfl,pr.o.base.dt*rate);
    if(!std::isfinite(rate)||rate<0.||pr.o.base.dt*rate>diffusion_cfl_limit)
      throw std::runtime_error("work-cap monitored diffusion timestep bound exceeded");
    State addition(ne*m*4,0.);std::vector<State> momentum_load(ne,State(m,0.)),energy_load(ne,State(m,0.));
    for(int e=0;e<ne;++e)for(int i=0;i<m;++i)for(int j=0;j<m;++j)
    {momentum_load[e][i]-=D(j,i)*load[e][j]/h[e];energy_load[e][i]-=D(j,i)*load[e][m+j]/h[e];}
    face_traction.assign(ne+1,0.);face_velocity.assign(ne+1,0.);cell_speed=speed;
    for(int f=0;f<=ne;++f)
    {
      const double That=f==0?Tpoly[0][0]:f==ne?Tpoly[ne-1][m-1]:.5*(Tpoly[f-1][m-1]+Tpoly[f][0]);
      const double uhat=f==0||f==ne?0.:.5*(primitive[f-1][2*m-1]+primitive[f][m]);
      face_traction[f]=That;face_velocity[f]=uhat;
      if(f>0){momentum_load[f-1][m-1]+=That;energy_load[f-1][m-1]+=uhat*That;}
      if(f<ne){momentum_load[f][0]-=That;energy_load[f][0]-=uhat*That;}
    }
    double heat=0.,power=0.,energy=0.;
    for(int e=0;e<ne;++e)
    {
      for(int i=0;i<m;++i){heat+=g[e][i]*load[e][i];power+=primitive[e][m+i]*momentum_load[e][i];energy+=energy_load[e][i];}
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)
      {addition[(e*m+i)*4+1]+=MI(i,j)*momentum_load[e][j]/h[e];addition[(e*m+i)*4+3]+=MI(i,j)*energy_load[e][j]/h[e];}
    }
    last_heat=heat;maximum_pairing_error=std::max(maximum_pairing_error,std::abs(power+heat));
    maximum_energy_budget_error=std::max(maximum_energy_budget_error,std::abs(energy));
    budget[1]+=face_traction[ne]-face_traction[0];
    // Retain raw Euler LLF and raise its scalar envelope where the mixed
    // principal first-order coefficient is larger. This is separate from the
    // adjoint viscous correction whose kinetic identity is measured above.
    for(int f=0;f<=ne;++f)
    {
      std::array<double,4> left{},right{};double al=0.,ar=0.;
      if(f==0||f==ne)
      {
        const int e=f==0?0:ne-1,node=f==0?0:pr.p;
        for(int c=0;c<4;++c)left[c]=right[c]=physical[pr.Id(e,node,c)];
        if(f==0)left[1]=-left[1];else right[1]=-right[1];
        pr.physics.Flux(left.data(),raw[pr.SigmaIndex(e,node)],al);pr.physics.Flux(right.data(),raw[pr.SigmaIndex(e,node)],ar);
      }
      else
      {
        for(int c=0;c<4;++c){left[c]=physical[pr.Id(f-1,pr.p,c)];right[c]=physical[pr.Id(f,0,c)];}
        pr.physics.Flux(left.data(),raw[pr.SigmaIndex(f-1,pr.p)],al);pr.physics.Flux(right.data(),raw[pr.SigmaIndex(f,0)],ar);
      }
      const double old=pr.physics.FaceSpeed(al,ar),next=std::max(old,std::max(speed[std::max(0,f-1)],speed[std::min(ne-1,f)]));
      for(int c=0;c<4;++c)
      {
        const double delta=-.5*(next-old)*(right[c]-left[c]);
        if(f>0)for(int i=0;i<m;++i)addition[((f-1)*m+i)*4+c]-=MI(i,m-1)*delta/h[f-1];
        if(f<ne)for(int i=0;i<m;++i)addition[(f*m+i)*4+c]+=MI(i,0)*delta/h[f];
        if(f==0)budget[c]+=delta;if(f==ne)budget[c]-=delta;
      }
      work.flux_points+=2;++work.faces;
    }
    const State high=pr.state_transfer.Apply(pr.state_transfer.E,addition,ne,4);
    for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=high[i];
    const auto totals=pr.Integrals(rhs,pr.edges);
    for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    cached_physical=physical;cached_raw=raw;++calls;const double elapsed=Wall(start);seconds+=elapsed;work.transport+=elapsed;
  }
  template<class Problem>
  std::array<double,4> BoundaryCorrection(const Problem& pr,const State& physical,const State& raw,int side)const
  {
    if(physical!=cached_physical||raw!=cached_raw)throw std::runtime_error("work-cap boundary report requires matching current residual");
    const int e=side?pr.ne-1:0,node=side?pr.p:0;double old=0.;pr.physics.Flux(&physical[pr.Id(e,node,0)],raw[pr.SigmaIndex(e,node)],old);
    const double delta=std::max(0.,cell_speed[e]-old)*(side?1.:-1.)*physical[pr.Id(e,node,1)];
    return {{0.,-face_traction[side?pr.ne:0]+delta,0.,0.}};
  }
};

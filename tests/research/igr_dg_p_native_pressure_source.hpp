#pragma once
// Test-only constitutive-source bridge. The existing DG density/SIPG matrix,
// primitive conservative moment reconstruction, Euler flux and LLF stay shared.
// Fields are frozen only within each instantaneous nonlinear closure solve.
struct NativePressureSource
{
  std::string law="constant-compression",pressure_faces="central";
  bool mirror_wall_source=false;
  double kappa=6.,tolerance=1.e-11;
  int iteration_limit=100;
  mutable std::uint64_t calls=0,factors=0,solves=0,quadrature_points=0;
  mutable int maximum_iterations=0;
  mutable double seconds=0.,maximum_residual=0.,minimum_stress=INFINITY;
  mutable double minimum_pressure=INFINITY,minimum_density=INFINITY,maximum_pressure_gradient=0.;
  mutable State last_load,last_sigma;
  template<class Problem>
  void Update(const Problem& pr,const State& physical,TrackWork& work,const BandMatrix& a,State& b)const
  {
    const auto start=Clock::now();const int m=pr.retained.n,nq=int(pr.weights.size());
    const bool nonlinear=law=="pressure-compression",gated=law!="constant";
    const double alpha=pr.o.base.alpha;
    State rho(pr.ne*nq),pressure(rho.size()),gradient(rho.size()),px(rho.size());
    auto thermal=[](const std::array<double,4>& u)
    {return (gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);};
    for(int e=0;e<pr.ne;++e)for(int q=0;q<nq;++q)
    {
      const int k=e*nq+q;const double h=pr.edges[e+1]-pr.edges[e];
      const auto u=pr.Point(physical,e,pr.shapes[q]);pr.physics.CheckState(u.data());
      std::array<double,4> dx{};
      for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)dx[c]+=pr.derivatives[q][i]*physical[pr.Id(e,i,c)]/h;
      const double vx=u[1]/u[0],vy=u[2]/u[0];rho[k]=u[0];pressure[k]=thermal(u);
      gradient[k]=(dx[1]-vx*dx[0])/u[0];
      px[k]=(gamma_euler-1.)*(dx[3]-vx*dx[1]-vy*dx[2]+.5*(vx*vx+vy*vy)*dx[0]);
      for(int side=0;side<2;++side)
      {
        const int neighbor=e+(side?1:-1);
        const bool exterior=neighbor<0||neighbor>=pr.ne;
        const bool mirror=exterior&&mirror_wall_source&&(side?pr.boundary.right:pr.boundary.left)=="wall";
        if(exterior&&!mirror)continue;
        const int endpoint=side?pr.p:0,other=side?0:pr.p,low_endpoint=side?m-1:0;
        std::array<double,4> own{},outside{};
        for(int c=0;c<4;++c)
        {own[c]=physical[pr.Id(e,endpoint,c)];outside[c]=mirror?own[c]:physical[pr.Id(neighbor,other,c)];}
        if(mirror)outside[1]=-outside[1];
        double lift=0.;for(int i=0;i<m;++i)lift+=pr.low_shapes[q][i]*pr.retained.mass_inverse(i,low_endpoint)/h;
        const double signed_lift=(side?-1.:1.)*.5*lift;
        if(pr.central_source)gradient[k]+=signed_lift*(own[1]/own[0]-outside[1]/outside[0]);
        if(pressure_faces=="central")px[k]+=signed_lift*(thermal(own)-thermal(outside));
      }
      if(gated)gradient[k]=std::min(gradient[k],0.);
      minimum_density=std::min(minimum_density,rho[k]);minimum_pressure=std::min(minimum_pressure,pressure[k]);
      maximum_pressure_gradient=std::max(maximum_pressure_gradient,std::abs(px[k]));
    }
    auto load=[&](const State& sigma)
    {
      b.assign(a.n,0.);double integral=0.;
      for(int e=0;e<pr.ne;++e)for(int q=0;q<nq;++q)
      {
        const int k=e*nq+q;double s=0.;for(int i=0;i<m;++i)s+=pr.low_shapes[q][i]*sigma[e*m+i];
        minimum_stress=std::min(minimum_stress,s);
        double value=alpha*(nonlinear?3.*gamma_euler+1.:kappa)*gradient[k]*gradient[k];
        if(nonlinear)
        {
          const double denominator=gamma_euler*pressure[k]+(gamma_euler-1.)*s;
          const double numerator=2.*gamma_euler*pressure[k]+(2.*gamma_euler-1.)*s;
          if(!(denominator>0.)||!(numerator>=0.))throw std::runtime_error("native pressure source lost positive denominator/numerator; no clipping applied");
          value+=alpha*px[k]*px[k]/rho[k]*numerator/(denominator*denominator);
        }
        if(!std::isfinite(value)||value<0.)throw std::runtime_error("native constitutive source nonfinite/negative");
        const double weight=(pr.edges[e+1]-pr.edges[e])*pr.weights[q];integral+=weight*value;
        for(int i=0;i<m;++i)b[e*m+i]+=weight*pr.low_shapes[q][i]*value;
        ++quadrature_points;++work.source_points;
      }
      work.current_source_integral=integral;
    };
    State sigma(a.n,0.);load(sigma);
    if(nonlinear)
    {
      BandFactor factor(a,0,a.n);++factors;bool converged=false;
      for(int iteration=0;iteration<iteration_limit;++iteration)
      {
        State residual(a.n,0.);
        for(int i=0;i<a.n;++i)
        {residual[i]=-b[i];for(int j=std::max(0,i-a.k);j<std::min(a.n,i+a.k+1);++j)residual[i]+=a.Get(i,j)*sigma[j];}
        const double relative=Norm(residual)/std::max(Norm(b),1.e-12);
        if(!std::isfinite(relative))throw std::runtime_error("nonfinite native pressure source residual");
        if(relative<tolerance)
        {maximum_residual=std::max(maximum_residual,relative);maximum_iterations=std::max(maximum_iterations,iteration);converged=true;break;}
        sigma=b;factor.Solve(sigma,1);++solves;load(sigma);
      }
      if(!converged)throw std::runtime_error("native pressure source Picard iteration limit");
    }
    last_load=b;last_sigma=nonlinear?sigma:State{};++calls;seconds+=Wall(start);
    // MixedSystem's caller performs one additional factor/linear solve. This
    // redundant work preserves the existing dispatch and is deliberately charged.
  }
};

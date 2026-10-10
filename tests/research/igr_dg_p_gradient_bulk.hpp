#pragma once
// Test-only bulk velocity-gradient spectral penalty. The high-pass cutoff
// grows with p. The optional top-strain policy has a rank-one momentum load,
// vanishes for input velocity in P_(p-1), and has quadratic h activation on
// smooth fields. The density-weighted velocity update can mix modes. Both are numerical
// methods, not equivalent IGR rewrites; the old upper-half policy loses h order.
struct FrontBulkGradient
{
  int cutoff,m;double strength;bool top_strain;State stiffness,dual;
  std::vector<State> nodal_shapes,velocity_derivatives;
  mutable double seconds=0.,maximum_mean_defect=0.,maximum_work_defect=0.;
  mutable double maximum_dissipation=0.,maximum_rate_bound=0.;
  mutable double maximum_activity=0.,maximum_strain_ratio_squared=0.;
  mutable std::uint64_t calls=0,local_factors=0;
  FrontBulkGradient(const PrimitiveMomentProblem& pr,const std::vector<State>& derivatives,double value,bool consistent=false):
    cutoff(consistent?pr.retained.p-1:std::max(1,pr.retained.p/2)),m(pr.retained.n),strength(value),top_strain(consistent),stiffness(m*m,0.),dual(m*m,0.),velocity_derivatives(derivatives)
  {
    const int p=pr.retained.p;std::vector<State> moments(p,State(m,0.));
    for(double z:pr.retained.physics.basis.nodes)nodal_shapes.push_back(pr.Shape(z,false));
    for(std::size_t q=0;q<pr.weights.size();++q)
    {
      double z=0.;for(int i=0;i<pr.n;++i)z+=pr.shapes[q][i]*pr.physics.basis.nodes[i];
      State v,d;MinimumGradientRecovery::Legendre(2*z-1.,p,v,d);
      for(int k=cutoff;k<p;++k)for(int i=0;i<m;++i)moments[k][i]+=pr.weights[q]*v[k]*derivatives[q][i];
    }
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=cutoff;k<p;++k)
      stiffness[i*m+j]+=(2*k+1)*moments[k][i]*moments[k][j];
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)
      dual[i*m+j]+=pr.retained.mass_inverse(i,k)*stiffness[k*m+j];
  }
  void Add(const PrimitiveMomentProblem& pr,const State& physical,const State& sigma,
           State& rhs,TrackWork& work)const
  {
    if(strength==0.||cutoff>=pr.retained.p)return;
    const auto start=Clock::now();State addition(pr.ne*m*4,0.);
    for(int e=0;e<pr.ne;++e)
    {
      BandMatrix density_mass(m,m-1);double speed=0.,thermal_sound_squared=0.,reference_gradient_square=0.;State velocity(2*m,0.);
      for(int i=0;i<m;++i)
      {const auto u=pr.Point(physical,e,nodal_shapes[i]);for(int c=0;c<2;++c)velocity[c*m+i]=u[c+1]/u[0];}
      for(std::size_t q=0;q<pr.weights.size();++q)
      {
        const auto u=pr.Point(physical,e,pr.shapes[q]);double s=0.,a=0.;
        for(int i=0;i<pr.n;++i)s+=pr.shapes[q][i]*sigma[e*pr.n+i];
        pr.physics.Flux(u.data(),s,a);speed=std::max(speed,a);++work.flux_points;
        if(top_strain)
        {
          const double pressure=(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
          if(!(pressure>0.))throw std::runtime_error("top-strain sensor needs positive thermal pressure");
          thermal_sound_squared=std::max(thermal_sound_squared,gamma_euler*pressure/u[0]);
          for(int c=0;c<2;++c)
          {double d=0.;for(int i=0;i<m;++i)d+=velocity_derivatives[q][i]*velocity[c*m+i];reference_gradient_square+=pr.weights[q]*d*d;}
        }
        for(int i=0;i<m;++i)for(int j=0;j<m;++j)
          density_mass.At(i,j)+=pr.weights[q]*u[0]*pr.low_shapes[q][i]*pr.low_shapes[q][j];
      }
      // Trace(M_rho^{-1} K) bounds the largest positive generalized
      // eigenvalue. It uses the current density, not an unweighted estimate.
      BandFactor factor(density_mass,0,m);State normalized=stiffness;factor.Solve(normalized,m);++local_factors;
      double trace=0.;for(int i=0;i<m;++i)trace+=normalized[i*m+i];
      if(!std::isfinite(trace)||!(trace>0.))throw std::runtime_error("invalid bulk gradient normalization");
      const double ratio=top_strain?reference_gradient_square/thermal_sound_squared:0.;
      const double activity=top_strain?ratio/(1.+ratio):1.;
      if(!std::isfinite(activity)||activity<0.||activity>1.)throw std::runtime_error("invalid bulk strain activity");
      maximum_activity=std::max(maximum_activity,activity);maximum_strain_ratio_squared=std::max(maximum_strain_ratio_squared,ratio);
      const double h=pr.edges[e+1]-pr.edges[e],rate=strength*activity*speed*m*m/h,coefficient=rate/trace;
      maximum_rate_bound=std::max(maximum_rate_bound,rate);
      double dissipation=0.,power=0.;
      for(int c=0;c<2;++c)
      {
        for(int i=0;i<m;++i)for(int j=0;j<m;++j)
        {
          addition[(e*m+i)*4+c+1]-=coefficient*dual[i*m+j]*velocity[c*m+j];
          dissipation+=h*coefficient*velocity[c*m+i]*stiffness[i*m+j]*velocity[c*m+j];
        }
        double mean=0.;for(int i=0;i<m;++i)
        {
          mean+=h*pr.retained.physics.basis.weights[i]*addition[(e*m+i)*4+c+1];
          for(int j=0;j<m;++j)power+=h*velocity[c*m+i]*pr.retained.mass(i,j)*addition[(e*m+j)*4+c+1];
        }
        maximum_mean_defect=std::max(maximum_mean_defect,std::abs(mean));
      }
      maximum_dissipation=std::max(maximum_dissipation,dissipation);
      maximum_work_defect=std::max(maximum_work_defect,std::abs(power+dissipation));
    }
    const State high=pr.state_transfer.Apply(pr.state_transfer.E,addition,pr.ne,4);
    for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=high[i];
    ++calls;const double elapsed=Wall(start);seconds+=elapsed;work.transport+=elapsed;
  }
};

#pragma once
// Test-only gradient coordinates on the existing degree-p primitive manifold.
// Each momentum field stores its conserved mean and p Legendre coefficients
// of the broken physical velocity derivative. The instantaneous IGR source
// still uses the inherited true-Pp central gradient, including face lifts.
struct GradientEvolutionProblem:CurvatureProblem
{
  std::string acceleration;
  State means;mfem::DenseMatrix gradient_map,integral_map;
  mutable State translated_coordinates,translated_moments;
  mutable State last_standard_rhs,last_native_rhs,last_acceleration,last_target;
  mutable double coordinate_seconds=0.,acceleration_seconds=0.,maximum_mean_defect=0.;
  mutable double maximum_weighted_projection_defect=0.,last_added_kinetic_power=0.;
  mutable std::uint64_t coordinate_conversions=0,acceleration_cells=0;
  static State Legendre(double x,int degree)
  {
    State a(degree+1,1.);if(degree)a[1]=x;
    for(int k=1;k<degree;++k)a[k+1]=((2*k+1)*x*a[k]-k*a[k-1])/(k+1);
    return a;
  }
  GradientEvolutionProblem(const MemoryOptions& mo,double slope,double curvature,std::string method):
    CurvatureProblem(mo,slope,curvature),acceleration(std::move(method)),means(retained.n,0.),
    gradient_map(retained.p,retained.n),integral_map(retained.n,retained.p)
  {
    if(acceleration!="chain"&&acceleration!="commuting"&&acceleration!="continuum"&&acceleration!="continuum-weighted")throw std::runtime_error("unknown gradient acceleration method");
    if(memory.reset_step||memory.memory_time!=0.)throw std::runtime_error("gradient coordinate memory resets are not defined");
    gradient_map=0.;integral_map=0.;
    for(std::size_t q=0;q<weights.size();++q)
    {
      double z=0.;for(int i=0;i<n;++i)z+=shapes[q][i]*physics.basis.nodes[i];const auto P=Legendre(2*z-1.,retained.p);
      for(int i=0;i<retained.n;++i)
      {
        means[i]+=weights[q]*low_shapes[q][i];
        for(int k=0;k<retained.p;++k)gradient_map(k,i)+=weights[q]*(2*k+1)*P[k]*low_derivatives[q][i];
      }
    }
    for(int i=0;i<retained.n;++i)
    {
      const auto P=Legendre(2*retained.physics.basis.nodes[i]-1.,retained.p);
      integral_map(i,0)=.5*P[1];
      for(int k=1;k<retained.p;++k)integral_map(i,k)=.5*(P[k+1]-P[k-1])/(2*k+1);
    }
  }
  State Velocity(const State& low,int e)const
  {
    const int m=retained.n;const double h=edges[e+1]-edges[e];State v(2*m,0.);double mass=0.;
    for(int i=0;i<m;++i)for(int k=0;k<retained.p;++k)for(int c=0;c<2;++c)v[c*m+i]+=h*integral_map(i,k)*low[(e*m+k+1)*4+c+1];
    std::array<double,2> momentum{};
    for(std::size_t q=0;q<weights.size();++q)
    {
      double rho=0.;std::array<double,2> value{};
      for(int i=0;i<m;++i){rho+=low_shapes[q][i]*low[(e*m+i)*4];for(int c=0;c<2;++c)value[c]+=low_shapes[q][i]*v[c*m+i];}
      mass+=weights[q]*rho;for(int c=0;c<2;++c)momentum[c]+=weights[q]*rho*value[c];
    }
    if(!(mass>floor_value)||!std::isfinite(mass))throw std::runtime_error("gradient coordinate density mean admission failed");
    for(int c=0;c<2;++c){const double constant=(low[e*m*4+c+1]-momentum[c])/mass;for(int i=0;i<m;++i)v[c*m+i]+=constant;}
    return v;
  }
  State Pack(const State& moments)const
  {
    const auto start=Clock::now();const int m=retained.n;State low=state_transfer.Apply(state_transfer.P,moments,ne,4),packed=low;
    const State& physical=PrimitiveMomentProblem::Reconstruct(moments);
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];State v(2*m,0.);
      for(int i=0;i<m;++i){const auto u=Point(physical,e,primitive_nodes[i]);for(int c=0;c<2;++c)v[c*m+i]=u[c+1]/u[0];}
      for(int c=0;c<2;++c)
      {
        double mean=0.;for(int i=0;i<m;++i)mean+=means[i]*low[(e*m+i)*4+c+1];packed[e*m*4+c+1]=mean;
        for(int k=0;k<retained.p;++k){double a=0.;for(int i=0;i<m;++i)a+=gradient_map(k,i)*v[c*m+i]/h;packed[(e*m+k+1)*4+c+1]=a;}
      }
    }
    State out=state_transfer.Apply(state_transfer.E,packed,ne,4);coordinate_seconds+=Wall(start);++coordinate_conversions;return out;
  }
  const State& Moments(const State& coordinates)const
  {
    if(coordinates==translated_coordinates)return translated_moments;
    const auto start=Clock::now();const int m=retained.n;const State low=state_transfer.Apply(state_transfer.P,coordinates,ne,4);State out=low;
    for(int e=0;e<ne;++e)
    {
      const State v=Velocity(low,e);State load(2*m,0.);
      for(std::size_t q=0;q<weights.size();++q)
      {
        double rho=0.;std::array<double,2> value{};for(int i=0;i<m;++i){rho+=low_shapes[q][i]*low[(e*m+i)*4];for(int c=0;c<2;++c)value[c]+=low_shapes[q][i]*v[c*m+i];}
        for(int i=0;i<m;++i)for(int c=0;c<2;++c)load[c*m+i]+=weights[q]*low_shapes[q][i]*rho*value[c];
      }
      for(int i=0;i<m;++i)for(int c=0;c<2;++c){double value=0.;for(int j=0;j<m;++j)value+=retained.mass_inverse(i,j)*load[c*m+j];out[(e*m+i)*4+c+1]=value;}
    }
    translated_coordinates=coordinates;translated_moments=state_transfer.Apply(state_transfer.E,out,ne,4);coordinate_seconds+=Wall(start);++coordinate_conversions;return translated_moments;
  }
  State Initial()const{return Pack(MemoryProblem::Initial());}
  void Limit(State& coordinates,TrackWork& work)const
  {
    State standard=Moments(coordinates);PrimitiveMomentProblem::Limit(standard,work);coordinates=Pack(standard);translated_coordinates.clear();
  }
  State Closure(const State& coordinates,double time,TrackWork& work)const
  {return PrimitiveMomentProblem::Closure(Moments(coordinates),time,work);}
  std::array<double,4> Integrals(const State& coordinates,const State& mesh)const
  {
    const State low=state_transfer.Apply(state_transfer.P,coordinates,ne,4);std::array<double,4> totals{};const int m=retained.n;
    for(int e=0;e<ne;++e)
    {
      const double h=mesh[e+1]-mesh[e];for(int c:{0,3})for(int i=0;i<m;++i)totals[c]+=h*means[i]*low[(e*m+i)*4+c];
      for(int c:{1,2})totals[c]+=h*low[e*m*4+c];
    }
    return totals;
  }
  // Exact differential of the coordinate reconstruction; this also provides
  // an independently checkable conservative RHS for frozen error analysis.
  State StandardDirection(const State& coordinates,const State& direction)const
  {
    const int m=retained.n;const State low=state_transfer.Apply(state_transfer.P,coordinates,ne,4),d=state_transfer.Apply(state_transfer.P,direction,ne,4);State out=d;
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];const State v=Velocity(low,e);State a(2*m,0.),load(2*m,0.);double mass=0.;std::array<double,2> change{};
      for(int i=0;i<m;++i)for(int k=0;k<retained.p;++k)for(int c=0;c<2;++c)a[c*m+i]+=h*integral_map(i,k)*d[(e*m+k+1)*4+c+1];
      for(std::size_t q=0;q<weights.size();++q)
      {
        double rho=0.,drho=0.;std::array<double,2> value{},rate{};
        for(int i=0;i<m;++i){rho+=low_shapes[q][i]*low[(e*m+i)*4];drho+=low_shapes[q][i]*d[(e*m+i)*4];for(int c=0;c<2;++c){value[c]+=low_shapes[q][i]*v[c*m+i];rate[c]+=low_shapes[q][i]*a[c*m+i];}}
        mass+=weights[q]*rho;for(int c=0;c<2;++c)change[c]+=weights[q]*(drho*value[c]+rho*rate[c]);
      }
      for(int c=0;c<2;++c){const double constant=(d[e*m*4+c+1]-change[c])/mass;for(int i=0;i<m;++i)a[c*m+i]+=constant;}
      for(std::size_t q=0;q<weights.size();++q)
      {
        double rho=0.,drho=0.;std::array<double,2> value{},rate{};
        for(int i=0;i<m;++i){rho+=low_shapes[q][i]*low[(e*m+i)*4];drho+=low_shapes[q][i]*d[(e*m+i)*4];for(int c=0;c<2;++c){value[c]+=low_shapes[q][i]*v[c*m+i];rate[c]+=low_shapes[q][i]*a[c*m+i];}}
        for(int i=0;i<m;++i)for(int c=0;c<2;++c)load[c*m+i]+=weights[q]*low_shapes[q][i]*(drho*value[c]+rho*rate[c]);
      }
      for(int i=0;i<m;++i)for(int c=0;c<2;++c){double value=0.;for(int j=0;j<m;++j)value+=retained.mass_inverse(i,j)*load[c*m+j];out[(e*m+i)*4+c+1]=value;}
    }
    return state_transfer.Apply(state_transfer.E,out,ne,4);
  }
  State Residual(const State& coordinates,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    const bool continuum_target=acceleration=="continuum"||acceleration=="continuum-weighted";
    const State standard=Moments(coordinates);const State original=CurvatureProblem::Residual(standard,sigma,time,work,budget);
    const auto start=Clock::now();const int m=retained.n;const State low=state_transfer.Apply(state_transfer.P,standard,ne,4),r=state_transfer.Apply(state_transfer.P,original,ne,4);
    const State& physical=PrimitiveMomentProblem::Reconstruct(standard);State packed=r;last_acceleration.assign(ne*m*2,0.);last_target.assign(ne*m*2,0.);
    for(int e=0;e<ne;++e)
    {
      ++acceleration_cells;const double h=edges[e+1]-edges[e];std::vector<std::array<double,4>> primitive(m);State low_sigma(m,0.),velocity(2*m,0.);
      for(int i=0;i<m;++i)
      {
        const auto u=Point(physical,e,primitive_nodes[i]);primitive[i]={u[0],u[1]/u[0],u[2]/u[0],(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0])};
        for(int c=0;c<2;++c)velocity[c*m+i]=primitive[i][c+1];for(int j=0;j<n;++j)low_sigma[i]+=primitive_nodes[i][j]*sigma[SigmaIndex(e,j)];
      }
      BandMatrix weighted(m,m-1);State load(2*m,0.),strong_load(3*m,0.),volume(3*m,0.);std::vector<std::array<double,3>> strong(weights.size());
      auto strong_at=[&](const State& b,const State& db)
      {
        std::array<double,4> a{},da{};double ds=0.;for(int i=0;i<m;++i){for(int c=0;c<4;++c){a[c]+=b[i]*primitive[i][c];da[c]+=db[i]*primitive[i][c]/h;}ds+=db[i]*low_sigma[i]/h;}
        return std::array<double,3>{-(da[0]*a[1]+a[0]*da[1]),-(da[0]*a[1]*a[1]+2*a[0]*a[1]*da[1]+da[3]+ds),-(da[0]*a[1]*a[2]+a[0]*da[1]*a[2]+a[0]*a[1]*da[2])};
      };
      for(std::size_t q=0;q<weights.size();++q)
      {
        const auto& b=low_shapes[q];double rho=0.,drho=0.;std::array<double,2> v{},dm{};
        for(int i=0;i<m;++i){rho+=b[i]*low[(e*m+i)*4];drho+=b[i]*r[(e*m+i)*4];for(int c=0;c<2;++c){v[c]+=b[i]*velocity[c*m+i];dm[c]+=b[i]*r[(e*m+i)*4+c+1];}}
        for(int i=0;i<m;++i){for(int j=0;j<m;++j)weighted.At(i,j)+=weights[q]*rho*b[i]*b[j];for(int c=0;c<2;++c)load[c*m+i]+=weights[q]*b[i]*(dm[c]-v[c]*drho);}
        if(acceleration!="chain")
        {strong[q]=strong_at(b,low_derivatives[q]);for(int c=0;c<3;++c)for(int i=0;i<m;++i)strong_load[c*m+i]+=weights[q]*b[i]*strong[q][c];}
      }
      BandFactor factor(weighted,0,m);factor.Solve(load,2);State actual=load;
      if(acceleration!="chain")
      {
        for(int c=0;c<3;++c)for(int i=0;i<m;++i)for(int j=0;j<m;++j)volume[c*m+i]+=retained.mass_inverse(i,j)*strong_load[c*m+j];
        State acceleration_load(2*m,0.),volume_chain(2*m,0.),target_nodal(2*m,0.),weighted_target(2*m,0.);
        for(std::size_t q=0;q<weights.size();++q)
        {
          const auto& b=low_shapes[q];double rho=0.,drho=0.;std::array<double,2> v{},dm{};
          for(int i=0;i<m;++i){rho+=b[i]*low[(e*m+i)*4];drho+=b[i]*volume[i];for(int c=0;c<2;++c){v[c]+=b[i]*velocity[c*m+i];dm[c]+=b[i]*volume[(c+1)*m+i];}}
          for(int c=0;c<2;++c)
          {
            const double target=(strong[q][c+1]-v[c]*(continuum_target?strong[q][0]:drho))/rho;
            for(int i=0;i<m;++i){acceleration_load[c*m+i]+=weights[q]*b[i]*target;weighted_target[c*m+i]+=weights[q]*b[i]*rho*target;volume_chain[c*m+i]+=weights[q]*b[i]*(dm[c]-v[c]*drho);}
          }
        }
        factor.Solve(volume_chain,2);factor.Solve(weighted_target,2);
        if(acceleration=="commuting")for(std::size_t i=0;i<volume_chain.size();++i)maximum_weighted_projection_defect=std::max(maximum_weighted_projection_defect,std::abs(volume_chain[i]-weighted_target[i]));
        for(int c=0;c<2;++c)for(int i=0;i<m;++i)for(int j=0;j<m;++j)target_nodal[c*m+i]+=retained.mass_inverse(i,j)*acceleration_load[c*m+j];
        const auto sl=strong_at(retained.Shape(0.,false),slope_tests[0]),sr=strong_at(retained.Shape(1.,false),slope_tests[1]);
        for(int c=0;c<2;++c)
        {
          const double left=(sl[c+1]-primitive[0][c+1]*(continuum_target?sl[0]:volume[0]))/primitive[0][0]-target_nodal[c*m];
          const double right=(sr[c+1]-primitive[m-1][c+1]*(continuum_target?sr[0]:volume[m-1]))/primitive[m-1][0]-target_nodal[c*m+m-1];
          const double sign=retained.p%2?-1.:1.,a=.5*(right-sign*left),b=.5*(right+sign*left);
          for(int i=0;i<m;++i)
          {
            const auto P=Legendre(2*retained.physics.basis.nodes[i]-1.,retained.p);target_nodal[c*m+i]+=a*P[retained.p-1]+b*P[retained.p];
            // The weighted ablation drops the density projection commutator
            // while retaining the original density-weighted L2 projector.
            const double chosen=acceleration=="continuum-weighted"?weighted_target[c*m+i]:target_nodal[c*m+i];
            actual[c*m+i]+=chosen-volume_chain[c*m+i];last_target[(e*m+i)*2+c]=chosen;
          }
        }
      }
      double mass=0.;std::array<double,2> wanted{},produced{};
      for(int i=0;i<m;++i)for(int c=0;c<2;++c)wanted[c]+=means[i]*r[(e*m+i)*4+c+1];
      for(std::size_t q=0;q<weights.size();++q)
      {
        double rho=0.,drho=0.;std::array<double,2> v{},a{};
        for(int i=0;i<m;++i){rho+=low_shapes[q][i]*low[(e*m+i)*4];drho+=low_shapes[q][i]*r[(e*m+i)*4];for(int c=0;c<2;++c){v[c]+=low_shapes[q][i]*velocity[c*m+i];a[c]+=low_shapes[q][i]*actual[c*m+i];}}
        mass+=weights[q]*rho;for(int c=0;c<2;++c)produced[c]+=weights[q]*(drho*v[c]+rho*a[c]);
      }
      for(int c=0;c<2;++c)
      {
        const double constant=(wanted[c]-produced[c])/mass;for(int i=0;i<m;++i){actual[c*m+i]+=constant;last_acceleration[(e*m+i)*2+c]=actual[c*m+i];}
        packed[e*m*4+c+1]=wanted[c];
        for(int k=0;k<retained.p;++k){double value=0.;for(int i=0;i<m;++i)value+=gradient_map(k,i)*actual[c*m+i]/h;packed[(e*m+k+1)*4+c+1]=value;}
      }
    }
    State out=state_transfer.Apply(state_transfer.E,packed,ne,4);last_native_rhs=original;last_standard_rhs=StandardDirection(coordinates,out);last_added_kinetic_power=0.;
    const State delta_low=state_transfer.Apply(state_transfer.P,last_standard_rhs,ne,4);
    for(int e=0;e<ne;++e)
    {
      std::array<double,4> difference{};
      for(int i=0;i<m;++i)for(int c=0;c<4;++c)difference[c]+=means[i]*(delta_low[(e*m+i)*4+c]-r[(e*m+i)*4+c]);
      for(double value:difference)maximum_mean_defect=std::max(maximum_mean_defect,std::abs(value));
      for(std::size_t q=0;q<weights.size();++q)
      {
        const auto u=Point(physical,e,shapes[q]);for(int c=0;c<2;++c){double delta=0.;for(int i=0;i<m;++i)delta+=low_shapes[q][i]*(delta_low[(e*m+i)*4+c+1]-r[(e*m+i)*4+c+1]);last_added_kinetic_power+=(edges[e+1]-edges[e])*weights[q]*u[c+1]/u[0]*delta;}
      }
    }
    const auto totals=Integrals(out,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    const double elapsed=Wall(start);acceleration_seconds+=elapsed;work.transport+=elapsed;return out;
  }
};

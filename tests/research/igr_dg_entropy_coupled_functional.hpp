#pragma once
// Exact quadrature gas-entropy gradient of the conservative primitive
// manifold, followed by a mean-preserving physical-volume-work correction.
// Existing numerical faces and CIP contributions are left in the base RHS.
// Matching this signed budget is NOT a global entropy-stability theorem.
struct EntropyFunctionalProblem:EntropyCoupledProblem
{
  std::string functional_mode;
  double budget_tolerance=2.e-12;
  mutable double functional_seconds=0.,maximum_functional_defect=0.,maximum_functional_controlled_residual=0.,maximum_functional_skipped_defect=0.,maximum_functional_unattainable_defect=0.,maximum_functional_mean=0.,maximum_functional_rhs_ratio=0.;
  mutable std::uint64_t functional_cells=0,functional_active_cells=0,functional_unattainable_cells=0;
  mutable State last_functional_correction,last_functional_defects,last_functional_remaining,last_functional_denominators,last_functional_tolerances;
  EntropyFunctionalProblem(const MemoryOptions& mo,double slope,double curvature,std::string mode):EntropyCoupledProblem(mo,slope,curvature,"original"),functional_mode(std::move(mode))
  {if(functional_mode!="none"&&functional_mode!="metric"&&functional_mode!="energy")throw std::runtime_error("unknown functional entropy control");}
  static BandMatrix ThermalHessian(const std::array<double,4>& U)
  {
    const auto a=EntropyCoupledState(U,0.);const double rho=a.rho,p=a.pressure,beta=a.beta,vv=a.vx*a.vx+a.vy*a.vy,gm=gamma_euler-1.;
    const std::array<double,4> dp{gm*.5*vv,-gm*a.vx,-gm*a.vy,gm};BandMatrix H(4,3);
    for(int j=0;j<4;++j)
    {
      const double db=(j==0?1./p:0.)-rho*dp[j]/(p*p),ds=dp[j]/p-(j==0?gamma_euler/rho:0.);
      const double dx=(j==1?1./rho:0.)-(j==0?a.vx/rho:0.),dy=(j==2?1./rho:0.)-(j==0?a.vy/rho:0.);
      H.At(0,j)=-ds/gm-.5*vv*db-beta*(a.vx*dx+a.vy*dy);H.At(1,j)=a.vx*db+beta*dx;H.At(2,j)=a.vy*db+beta*dy;H.At(3,j)=-db;
    }
    for(int i=0;i<4;++i)for(int j=i+1;j<4;++j){const double x=.5*(H.Get(i,j)+H.Get(j,i));H.At(i,j)=H.At(j,i)=x;}return H;
  }
  State Residual(const State& moments,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    State rhs=EntropyCoupledProblem::Residual(moments,sigma,time,work,budget);last_functional_correction.assign(rhs.size(),0.);last_functional_defects.assign(ne,0.);last_functional_remaining.assign(ne,0.);last_functional_denominators.assign(ne,0.);last_functional_tolerances.assign(ne,0.);
    if(functional_mode=="none")return rhs;
    const auto start=Clock::now();const State& physical=Reconstruct(moments);const State low=state_transfer.Apply(state_transfer.P,moments,ne,4),low_rhs=state_transfer.Apply(state_transfer.P,rhs,ne,4);const int m=retained.n,nq=int(weights.size());State all(ne*m*4,0.);double correction_norm=0.,rhs_norm=0.;
    for(int e=0;e<ne;++e)
    {
      ++functional_cells;const double h=edges[e+1]-edges[e];State Bload(m,0.),B(m,0.),zeta(2*m,0.),Z(m*4,0.),direction(m*4,0.);BandMatrix weighted(m,m-1);
      std::vector<EntropyCoupledPoint> point(nq);std::vector<std::array<double,4>> primitive(m),strong(nq);State low_sigma(m,0.);std::array<double,4> meanU{},meanZ{};
      for(int i=0;i<m;++i)
      {
        const auto u=Point(physical,e,primitive_nodes[i]);const auto a=EntropyCoupledState(u,0.);primitive[i]={a.rho,a.vx,a.vy,a.pressure};
        for(int j=0;j<n;++j)low_sigma[i]+=primitive_nodes[i][j]*sigma[SigmaIndex(e,j)];
        double weight=0.;for(int j=0;j<m;++j)weight+=retained.mass(i,j);for(int c=0;c<4;++c)meanU[c]+=weight*low[(e*m+i)*4+c];
      }
      for(int q=0;q<nq;++q)
      {
        double s=0.;for(int i=0;i<n;++i)s+=shapes[q][i]*sigma[SigmaIndex(e,i)];point[q]=EntropyCoupledState(Point(physical,e,shapes[q]),s);const auto a=point[q];
        std::array<double,4> derivative{};double ds=0.;for(int i=0;i<m;++i){for(int c=0;c<4;++c)derivative[c]+=low_derivatives[q][i]*primitive[i][c]/h;ds+=low_derivatives[q][i]*low_sigma[i]/h;}
        const double dr=derivative[0],dv=derivative[1],dvy=derivative[2],dp=derivative[3],vv=a.vx*a.vx+a.vy*a.vy;
        const double total=gamma_euler/(gamma_euler-1.)*a.pressure+.5*a.rho*vv+s,dtotal=gamma_euler/(gamma_euler-1.)*dp+.5*dr*vv+a.rho*(a.vx*dv+a.vy*dvy)+ds;
        strong[q]={-(dr*a.vx+a.rho*dv),-(dr*a.vx*a.vx+2*a.rho*a.vx*dv+dp+ds),-(dr*a.vx*a.vy+a.rho*dv*a.vy+a.rho*a.vx*dvy),-(dtotal*a.vx+total*dv)};
        for(int i=0;i<m;++i){Bload[i]+=weights[q]*low_shapes[q][i]*a.beta;for(int j=0;j<m;++j)weighted.At(i,j)+=weights[q]*a.rho*low_shapes[q][i]*low_shapes[q][j];}
      }
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)B[i]+=retained.mass_inverse(i,j)*Bload[j];
      for(int q=0;q<nq;++q)
      {double b=0.;for(int i=0;i<m;++i)b+=low_shapes[q][i]*B[i];for(int i=0;i<m;++i){const double f=weights[q]*low_shapes[q][i]*point[q].rho*b;zeta[i]+=f*point[q].vx;zeta[m+i]+=f*point[q].vy;}}
      BandFactor weighted_factor(weighted,0,m);weighted_factor.Solve(zeta,2);State rho_load(m,0.);
      for(int q=0;q<nq;++q)
      {
        double b=0.,zx=0.,zy=0.;for(int i=0;i<m;++i){b+=low_shapes[q][i]*B[i];zx+=low_shapes[q][i]*zeta[i];zy+=low_shapes[q][i]*zeta[m+i];}
        const auto a=point[q];const double slog=std::log(a.pressure)-gamma_euler*a.logrho,value=(gamma_euler-slog)/(gamma_euler-1.)+.5*b*(a.vx*a.vx+a.vy*a.vy)-zx*a.vx-zy*a.vy;
        for(int i=0;i<m;++i)rho_load[i]+=weights[q]*low_shapes[q][i]*value;
      }
      for(int i=0;i<m;++i){for(int j=0;j<m;++j)Z[4*i]+=retained.mass_inverse(i,j)*rho_load[j];Z[4*i+1]=zeta[i];Z[4*i+2]=zeta[m+i];Z[4*i+3]=-B[i];double weight=0.;for(int j=0;j<m;++j)weight+=retained.mass(i,j);for(int c=0;c<4;++c)meanZ[c]+=weight*Z[4*i+c];}
      const BandMatrix H=ThermalHessian(meanU);for(int i=0;i<m;++i)for(int c=0;c<4;++c)direction[4*i+c]=functional_mode=="energy"&&c!=3?0.:Z[4*i+c]-meanZ[c];
      if(functional_mode=="metric"){BandFactor factor(H,0,4);factor.Solve(direction,m);}
      double physical_volume=0.,functional_volume=0.,defect=0.,denominator=0.;
      for(int q=0;q<nq;++q)
      {
        std::array<double,4> z{},d{};for(int i=0;i<m;++i)for(int c=0;c<4;++c){z[c]+=low_shapes[q][i]*Z[4*i+c];d[c]+=low_shapes[q][i]*direction[4*i+c];}
        for(int c=0;c<4;++c){physical_volume+=h*weights[q]*point[q].w[c]*strong[q][c];functional_volume+=h*weights[q]*z[c]*strong[q][c];defect+=h*weights[q]*(point[q].w[c]-z[c])*strong[q][c];denominator+=h*weights[q]*(z[c]-meanZ[c])*d[c];}
      }
      const double tolerance=budget_tolerance*std::max(1.,std::abs(physical_volume)+std::abs(functional_volume)),rank_threshold=1.e-24*h;double multiplier=0.;
      maximum_functional_defect=std::max(maximum_functional_defect,std::abs(defect));
      if(std::abs(defect)>tolerance)
      {if(denominator>rank_threshold){multiplier=defect/denominator;++functional_active_cells;}else{++functional_unattainable_cells;maximum_functional_unattainable_defect=std::max(maximum_functional_unattainable_defect,std::abs(defect));}}
      else maximum_functional_skipped_defect=std::max(maximum_functional_skipped_defect,std::abs(defect));
      std::array<double,4> conserved{};double obtained=0.;
      for(int i=0;i<m;++i)for(int c=0;c<4;++c)all[(e*m+i)*4+c]=multiplier*direction[4*i+c];
      for(int q=0;q<nq;++q)
      {
        std::array<double,4> z{},delta{},r{};for(int i=0;i<m;++i)for(int c=0;c<4;++c){z[c]+=low_shapes[q][i]*Z[4*i+c];delta[c]+=low_shapes[q][i]*all[(e*m+i)*4+c];r[c]+=low_shapes[q][i]*low_rhs[(e*m+i)*4+c];}
        for(int c=0;c<4;++c){conserved[c]+=h*weights[q]*delta[c];obtained+=h*weights[q]*z[c]*delta[c];for(int k=0;k<4;++k){correction_norm+=h*weights[q]*delta[c]*H.Get(c,k)*delta[k];rhs_norm+=h*weights[q]*r[c]*H.Get(c,k)*r[k];}}
      }
      for(double x:conserved)maximum_functional_mean=std::max(maximum_functional_mean,std::abs(x));if(multiplier!=0.)maximum_functional_controlled_residual=std::max(maximum_functional_controlled_residual,std::abs(defect-obtained));
      last_functional_defects[e]=defect;last_functional_remaining[e]=defect-obtained;last_functional_denominators[e]=denominator;last_functional_tolerances[e]=tolerance;
    }
    maximum_functional_rhs_ratio=std::max(maximum_functional_rhs_ratio,std::sqrt(std::max(0.,correction_norm)/std::max(rhs_norm,1.e-100)));
    last_functional_correction=state_transfer.Apply(state_transfer.E,all,ne,4);for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=last_functional_correction[i];
    const auto totals=Integrals(rhs,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));const double elapsed=Wall(start);functional_seconds+=elapsed;work.transport+=elapsed;return rhs;
  }
};

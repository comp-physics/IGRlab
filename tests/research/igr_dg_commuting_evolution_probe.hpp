#pragma once
// Elementwise derivative-orthogonal projection of the physical volume RHS.
// The retained primitive manifold, numerical faces, closure, and CIP terms are
// inherited. This correction has no assumed kinetic/entropy sign.
struct CommutingProblem:CurvatureProblem
{
  double eta,ratio;bool seminorm,momentum_only,density_product;BandMatrix stiffness,sobolev;
  std::unique_ptr<BandFactor> sobolev_factor;
  State means;std::vector<State> third;std::array<State,2> second_face;
  mutable double commuting_seconds=0.,maximum_commuting_mean=0.,maximum_commuting_ratio=0.;
  mutable std::uint64_t commuting_cells=0;
  mutable std::array<double,4> last_commuting_kinetic_power{};
  mutable State last_commuting_correction;
  CommutingProblem(const MemoryOptions& mo,double slope,double curvature,double strength,bool h1,bool only_momentum=false,bool product=false):
    CurvatureProblem(mo,slope,curvature),eta(strength),ratio(strength/(retained.p*retained.p*(retained.p+1.)*(retained.p+1.))),seminorm(h1),momentum_only(only_momentum),density_product(product),
    stiffness(retained.n,retained.n-1),sobolev(retained.n,retained.n-1),means(retained.n,0.)
  {
    if(retained.p<2||eta<0.||!std::isfinite(eta))throw std::runtime_error("invalid commuting degree/strength");
    if(product&&(h1||strength!=0.))throw std::runtime_error("density product correction requires eta0 without Sobolev correction");
    const int m=retained.n;std::vector<State> d(m),d2(m,State(m,0.)),d3(m,State(m,0.));
    for(int i=0;i<m;++i)d[i]=retained.Shape(retained.physics.basis.nodes[i],true);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)d2[i][j]+=d[i][k]*d[k][j];
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)d3[i][j]+=d2[i][k]*d[k][j];
    second_face={d2.front(),d2.back()};
    for(std::size_t q=0;q<weights.size();++q)
    {
      State values(m,0.);for(int i=0;i<m;++i)for(int j=0;j<m;++j)values[j]+=low_shapes[q][i]*d3[i][j];third.push_back(values);
      for(int i=0;i<m;++i){means[i]+=weights[q]*low_shapes[q][i];for(int j=0;j<m;++j)stiffness.At(i,j)+=weights[q]*low_derivatives[q][i]*low_derivatives[q][j];}
    }
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)sobolev.At(i,j)=seminorm?stiffness.Get(i,j)+means[i]*means[j]:retained.mass(i,j)+ratio*stiffness.Get(i,j);
    sobolev_factor=std::make_unique<BandFactor>(sobolev,0,m);
  }
  // Reference-cell correction; divide by the actual physical h afterwards.
  // F' is a reference-coordinate derivative of the actual polynomial flux.
  State CorrectCellIntegrationByParts(const std::vector<std::array<double,4>>& flux,const std::array<std::array<double,4>,2>& face,
                    const std::array<std::array<double,4>,2>& derivative)const
  {
    const int m=retained.n;State load(4*m,0.),gradient(4*m,0.),volume(4*m,0.);
    for(int c=0;c<4;++c)for(int i=0;i<m;++i)
    {
      load[c*m+i]=(i==0?face[0][c]:0.)-(i==m-1?face[1][c]:0.);
      gradient[c*m+i]=slope_tests[0][i]*derivative[0][c]-slope_tests[1][i]*derivative[1][c]
                            +second_face[1][i]*face[1][c]-second_face[0][i]*face[0][c];
      for(std::size_t q=0;q<weights.size();++q)
      {load[c*m+i]+=weights[q]*low_derivatives[q][i]*flux[q][c];gradient[c*m+i]-=weights[q]*third[q][i]*flux[q][c];}
    }
    for(int c=0;c<4;++c)for(int i=0;i<m;++i)for(int j=0;j<m;++j)volume[c*m+i]+=retained.mass_inverse(i,j)*load[c*m+j];
    State correction(4*m,0.);
    if(seminorm)
    {
      for(int c=0;c<4;++c){double mean=0.;for(int i=0;i<m;++i)mean+=load[c*m+i];for(int i=0;i<m;++i)correction[c*m+i]=gradient[c*m+i]+means[i]*mean;}
      sobolev_factor->Solve(correction,4);for(std::size_t i=0;i<correction.size();++i)correction[i]-=volume[i];
    }
    else
    {
      for(int c=0;c<4;++c)for(int i=0;i<m;++i)
      {double value=gradient[c*m+i];for(int j=0;j<m;++j)value-=stiffness.Get(i,j)*volume[c*m+j];correction[c*m+i]=ratio*value;}
      sobolev_factor->Solve(correction,4);
    }
    return correction;
  }
  State CorrectCell(const std::vector<std::array<double,4>>& flux,const std::array<std::array<double,4>,2>& face,
                    const std::array<std::array<double,4>,2>& derivative)const
  {
    const int m=retained.n;State load(4*m,0.),volume(4*m,0.),difference(4*m,0.),correction(4*m,0.);
    for(int c=0;c<4;++c)for(int i=0;i<m;++i)
    {
      load[c*m+i]=(i==0?face[0][c]:0.)-(i==m-1?face[1][c]:0.);
      for(std::size_t q=0;q<weights.size();++q)load[c*m+i]+=weights[q]*low_derivatives[q][i]*flux[q][c];
    }
    for(int c=0;c<4;++c)for(int i=0;i<m;++i)for(int j=0;j<m;++j)volume[c*m+i]+=retained.mass_inverse(i,j)*load[c*m+j];
    for(int c=0;c<4;++c)
    {
      const double left=-derivative[0][c]-volume[c*m],right=-derivative[1][c]-volume[c*m+m-1];
      const double sign=retained.p%2?-1.:1.,a=.5*(right-sign*left),b=.5*(right+sign*left);
      for(int i=0;i<m;++i)
      {
        const double z=2.*retained.physics.basis.nodes[i]-1.;double previous=1.,current=z;
        for(int k=1;k<retained.p;++k){const double next=((2*k+1)*z*current-k*previous)/(k+1);previous=current;current=next;}
        difference[c*m+i]=a*previous+b*current;
      }
    }
    if(seminorm)return difference;
    for(int c=0;c<4;++c)for(int i=0;i<m;++i)for(int j=0;j<m;++j)correction[c*m+i]+=ratio*stiffness.Get(i,j)*difference[c*m+j];
    sobolev_factor->Solve(correction,4);return correction;
  }
  // Reference-coordinate ordinary conservative correction. It is exactly the
  // weighted-continuum acceleration method before time discretization, but
  // uses only polynomial products and fixed mass projections. Since v is Pp,
  // its cell mean is zero by L2 orthogonality of s_rho - Pi_p(s_rho).
  State DensityProductCell(const std::vector<std::array<double,4>>& primitive)const
  {
    const int m=retained.n;State strong(weights.size(),0.),load(m,0.),projected(m,0.),correction(4*m,0.);
    for(std::size_t q=0;q<weights.size();++q)
    {
      double rho=0.,v=0.,drho=0.,dv=0.;for(int i=0;i<m;++i)
      {rho+=low_shapes[q][i]*primitive[i][0];v+=low_shapes[q][i]*primitive[i][1];drho+=low_derivatives[q][i]*primitive[i][0];dv+=low_derivatives[q][i]*primitive[i][1];}
      strong[q]=-(drho*v+rho*dv);for(int i=0;i<m;++i)load[i]+=weights[q]*low_shapes[q][i]*strong[q];
    }
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)projected[i]+=retained.mass_inverse(i,j)*load[j];
    State product_load(2*m,0.);
    for(std::size_t q=0;q<weights.size();++q)
    {
      double represented=0.;std::array<double,2> v{};for(int i=0;i<m;++i){represented+=low_shapes[q][i]*projected[i];for(int c=0;c<2;++c)v[c]+=low_shapes[q][i]*primitive[i][c+1];}
      for(int i=0;i<m;++i)for(int c=0;c<2;++c)product_load[c*m+i]-=weights[q]*low_shapes[q][i]*v[c]*(strong[q]-represented);
    }
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<2;++c)correction[(c+1)*m+i]+=retained.mass_inverse(i,j)*product_load[c*m+j];
    return correction;
  }
  State Residual(const State& moments,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    State rhs=CurvatureProblem::Residual(moments,sigma,time,work,budget);last_commuting_kinetic_power={};last_commuting_correction.assign(rhs.size(),0.);
    if(eta==0.&&!seminorm&&!density_product)return rhs;
    const auto start=Clock::now();const State& physical=Reconstruct(moments);const int m=retained.n;double c2=0.,r2=0.;
    for(int e=0;e<ne;++e)
    {
      ++commuting_cells;const double h=edges[e+1]-edges[e];
      std::vector<std::array<double,4>> flux(weights.size()),primitive(m);State low_sigma(m,0.);
      for(int i=0;i<m;++i)
      {
        const auto u=Point(physical,e,primitive_nodes[i]);primitive[i]={u[0],u[1]/u[0],u[2]/u[0],(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0])};
        for(int j=0;j<n;++j)low_sigma[i]+=primitive_nodes[i][j]*sigma[SigmaIndex(e,j)];
      }
      if(!density_product)for(std::size_t q=0;q<weights.size();++q)
      {
        const auto u=Point(physical,e,shapes[q]);double s=0.,speed=0.;for(int j=0;j<n;++j)s+=shapes[q][j]*sigma[SigmaIndex(e,j)];flux[q]=physics.Flux(u.data(),s,speed);++work.flux_points;
      }
      std::array<std::array<double,4>,2> face{},derivative{};
      if(!density_product)for(int side=0;side<2;++side)
      {
        const int i=side?m-1:0;const auto a=primitive[i];std::array<double,4> da{};double ds=0.;
        for(int j=0;j<m;++j){for(int c=0;c<4;++c)da[c]+=slope_tests[side][j]*primitive[j][c];ds+=slope_tests[side][j]*low_sigma[j];}
        const double rho=a[0],v=a[1],vy=a[2],p=a[3],dr=da[0],dv=da[1],dvy=da[2],dp=da[3],speed2=v*v+vy*vy;
        const auto u=Point(physical,e,primitive_nodes[i]);double wave=0.;face[side]=physics.Flux(u.data(),low_sigma[i],wave);
        const double total=gamma_euler/(gamma_euler-1.)*p+.5*rho*speed2+low_sigma[i];
        const double dt=gamma_euler/(gamma_euler-1.)*dp+.5*dr*speed2+rho*(v*dv+vy*dvy)+ds;
        derivative[side]={dr*v+rho*dv,dr*v*v+2*rho*v*dv+dp+ds,dr*v*vy+rho*dv*vy+rho*v*dvy,dt*v+total*dv};
      }
      State correction=density_product?DensityProductCell(primitive):CorrectCell(flux,face,derivative);if(momentum_only)for(int i=0;i<m;++i){correction[i]=0.;correction[3*m+i]=0.;}std::array<double,4> mean{};
      for(int i=0;i<m;++i)for(int c=0;c<4;++c)mean[c]+=means[i]*correction[c*m+i];
      for(double value:mean)maximum_commuting_mean=std::max(maximum_commuting_mean,std::abs(value));
      for(std::size_t q=0;q<weights.size();++q)
      {
        std::array<double,4> delta{},base{};double v=0.,vy=0.;
        for(int i=0;i<m;++i){v+=low_shapes[q][i]*primitive[i][1];vy+=low_shapes[q][i]*primitive[i][2];for(int c=0;c<4;++c)delta[c]+=low_shapes[q][i]*correction[c*m+i]/h;}
        for(int i=0;i<n;++i)for(int c=0;c<4;++c)base[c]+=shapes[q][i]*rhs[Id(e,i,c)];
        const double weight=h*weights[q];last_commuting_kinetic_power[0]-=weight*.5*(v*v+vy*vy)*delta[0];last_commuting_kinetic_power[1]+=weight*v*delta[1];last_commuting_kinetic_power[2]+=weight*vy*delta[2];
        for(int c=0;c<4;++c){c2+=weight*delta[c]*delta[c];r2+=weight*base[c]*base[c];}
      }
      for(int i=0;i<n;++i)for(int j=0;j<m;++j)for(int c=0;c<4;++c)
      {const double value=state_transfer.E(i,j)*correction[c*m+j]/h;rhs[Id(e,i,c)]+=value;last_commuting_correction[Id(e,i,c)]+=value;}
    }
    maximum_commuting_ratio=std::max(maximum_commuting_ratio,std::sqrt(c2/std::max(r2,1.e-100)));
    const auto totals=Integrals(rhs,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    const double elapsed=Wall(start);commuting_seconds+=elapsed;work.transport+=elapsed;return rhs;
  }
};

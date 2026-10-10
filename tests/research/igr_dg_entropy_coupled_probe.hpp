#pragma once
// Test-only thermal-Euler flux differencing plus mechanical IGR stress work.
// Included after the existing primitive/curvature problem definitions.
// The physical states are NOT entropy projected. Therefore pair identities do
// not by themselves imply an entropy theorem on the nonlinear trial manifold.
struct EntropyCoupledPoint
{
  std::array<double,4> u{},w{},flux{};
  double rho=0.,vx=0.,vy=0.,pressure=0.,beta=0.,logrho=0.,logbeta=0.,sigma=0.,eta=0.;
};
inline EntropyCoupledPoint EntropyCoupledState(const std::array<double,4>& u,double sigma)
{
  EntropyCoupledPoint a;a.u=u;a.rho=u[0];a.vx=u[1]/a.rho;a.vy=u[2]/a.rho;
  const double vv=a.vx*a.vx+a.vy*a.vy;
  a.pressure=(gamma_euler-1.)*(u[3]-.5*a.rho*vv);
  if(!(a.rho>floor_value&&a.pressure>floor_value))throw std::runtime_error("entropy flux thermal admission failed");
  a.beta=a.rho/a.pressure;a.logrho=std::log(a.rho);a.logbeta=std::log(a.beta);a.sigma=sigma;
  const double s=std::log(a.pressure)-gamma_euler*a.logrho;
  a.eta=-a.rho*s/(gamma_euler-1.);
  a.w={(gamma_euler-s)/(gamma_euler-1.)-.5*a.beta*vv,a.beta*a.vx,a.beta*a.vy,-a.beta};
  a.flux={u[1],u[1]*a.vx+a.pressure+sigma,u[1]*a.vy,a.vx*(u[3]+a.pressure+sigma)};
  return a;
}
inline double EntropyCoupledLogMean(double a,double b,double la,double lb)
{
  const double mean=.5*(a+b),z=(b-a)/(a+b),z2=z*z;
  // Stable near equal states; omitted terms are <1e-24 for this branch.
  if(std::abs(z)<1.e-3)return mean/(1.+z2*(1./3.+z2*(1./5.+z2/7.)));
  return (b-a)/(lb-la);
}
inline std::array<double,4> EntropyCoupledFlux(const EntropyCoupledPoint& a,const EntropyCoupledPoint& b,bool arithmetic)
{
  std::array<double,4> f{};
  if(arithmetic){for(int c=0;c<4;++c)f[c]=.5*(a.flux[c]+b.flux[c]);return f;}
  const double vx=.5*(a.vx+b.vx),vy=.5*(a.vy+b.vy),pressure=.5*(a.pressure+b.pressure),sigma=.5*(a.sigma+b.sigma);
  const double rho=EntropyCoupledLogMean(a.rho,b.rho,a.logrho,b.logrho);
  const double beta=EntropyCoupledLogMean(a.beta,b.beta,a.logbeta,b.logbeta);
  f[0]=rho*vx;f[1]=f[0]*vx+pressure+sigma;f[2]=f[0]*vy;
  f[3]=f[0]*(.5*(a.vx*b.vx+a.vy*b.vy)+1./((gamma_euler-1.)*beta))
        +.5*(a.pressure*b.vx+b.pressure*a.vx)+sigma*vx;
  return f;
}
struct EntropyCoupledOperator
{
  int m,nq,nn;
  std::vector<State> values,physical_shapes;
  mfem::DenseMatrix skew;
  double sbp_null_defect=0.,sbp_symmetry_defect=0.;
  explicit EntropyCoupledOperator(const SlopePenaltyProblem& pr):m(pr.retained.n),nq(int(pr.weights.size())),nn(nq+2),skew(nn)
  {
    values=pr.low_shapes;physical_shapes=pr.shapes;
    for(int side=0;side<2;++side){values.push_back(pr.retained.Shape(double(side),false));physical_shapes.push_back(pr.Shape(double(side),false));}
    mfem::DenseMatrix project(m,nq),q(nq),trace(2,nq);project=0.;q=0.;trace=0.;skew=0.;
    for(int i=0;i<m;++i)for(int j=0;j<nq;++j)for(int k=0;k<m;++k)project(i,j)+=pr.retained.mass_inverse(i,k)*values[j][k]*pr.weights[j];
    for(int i=0;i<nq;++i)for(int j=0;j<nq;++j)for(int k=0;k<m;++k)q(i,j)+=pr.weights[i]*pr.low_derivatives[i][k]*project(k,j);
    for(int side=0;side<2;++side)for(int j=0;j<nq;++j)for(int k=0;k<m;++k)trace(side,j)+=values[nq+side][k]*project(k,j);
    for(int i=0;i<nq;++i)for(int j=0;j<nq;++j)skew(i,j)=.5*(q(i,j)-q(j,i));
    for(int i=0;i<nq;++i)for(int side=0;side<2;++side){const double x=.5*(side?1.:-1.)*trace(side,i);skew(i,nq+side)=x;skew(nq+side,i)=-x;}
    for(int i=0;i<nn;++i)
    {
      double row=i==nq?-.5:(i==nq+1?.5:0.);for(int j=0;j<nn;++j){row+=skew(i,j);sbp_symmetry_defect=std::max(sbp_symmetry_defect,std::abs(skew(i,j)+skew(j,i)));}
      sbp_null_defect=std::max(sbp_null_defect,std::abs(row));
    }
    if(sbp_null_defect>2.e-9||sbp_symmetry_defect>1.e-14)throw std::runtime_error("entropy generalized SBP construction failed");
  }
  // Returns a reference-cell weak-volume load, before the fixed M^-1/h map.
  // Pair work is the exact entropy-variable contraction on extended SBP nodes.
  State CellLoad(const SlopePenaltyProblem& pr,const State& physical,const State& sigma,int e,bool arithmetic,
                 std::uint64_t& pairs,double& pair_work_defect,double& entropy_transfer)const
  {
    std::vector<EntropyCoupledPoint> a(nn);std::vector<std::array<double,4>> divergence(nn);
    for(int i=0;i<nn;++i){double s=0.;for(int j=0;j<pr.n;++j)s+=physical_shapes[i][j]*sigma[pr.SigmaIndex(e,j)];a[i]=EntropyCoupledState(pr.Point(physical,e,physical_shapes[i]),s);}
    double stress=0.;
    for(int i=0;i<nn;++i)for(int j=i+1;j<nn;++j)
    {
      const double q=2.*skew(i,j);if(q==0.)continue;++pairs;const auto f=EntropyCoupledFlux(a[i],a[j],arithmetic);
      for(int c=0;c<4;++c){divergence[i][c]+=q*f[c];divergence[j][c]-=q*f[c];}
      stress+=q*.5*(a[i].sigma+a[j].sigma)*.5*(a[i].beta+a[j].beta)*(a[i].vx-a[j].vx);
    }
    double contraction=0.;for(int i=0;i<nn;++i)for(int c=0;c<4;++c)contraction+=a[i].w[c]*divergence[i][c];
    // The skew block removes the physical diagonal boundary contribution.
    // Euler's potential is rho*v; its skew contraction is -[rho*v].
    if(!arithmetic)pair_work_defect=std::max(pair_work_defect,std::abs(contraction+(a[nq+1].u[1]-a[nq].u[1])-stress));
    entropy_transfer+=stress;
    State load(m*4,0.);for(int k=0;k<m;++k)for(int i=0;i<nn;++i)for(int c=0;c<4;++c)load[4*k+c]-=values[i][k]*divergence[i][c];
    return load;
  }
};
struct EntropyCoupledProblem:CurvatureProblem
{
  std::string entropy_volume;
  EntropyCoupledOperator entropy_operator;
  mutable std::uint64_t entropy_pairs=0,entropy_cells=0;
  mutable double entropy_seconds=0.,maximum_entropy_mean=0.,maximum_entropy_pair_work_defect=0.;
  mutable State last_entropy_correction;
  mutable double last_entropy_pair_stress=0.;
  EntropyCoupledProblem(const MemoryOptions& mo,double slope,double curvature,std::string mode):CurvatureProblem(mo,slope,curvature),entropy_volume(std::move(mode)),entropy_operator(*this)
  {if(entropy_volume!="original"&&entropy_volume!="arithmetic"&&entropy_volume!="ranocha")throw std::runtime_error("unknown entropy volume mode");}
  State Residual(const State& moments,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    State rhs=CurvatureProblem::Residual(moments,sigma,time,work,budget);last_entropy_correction.assign(rhs.size(),0.);last_entropy_pair_stress=0.;
    if(entropy_volume=="original")return rhs;
    const auto start=Clock::now();const State& physical=Reconstruct(moments);const int m=retained.n;State low(ne*m*4,0.);
    for(int e=0;e<ne;++e)
    {
      ++entropy_cells;const double h=edges[e+1]-edges[e];State load=entropy_operator.CellLoad(*this,physical,sigma,e,entropy_volume=="arithmetic",entropy_pairs,maximum_entropy_pair_work_defect,last_entropy_pair_stress);
      // The inherited ordinary volume computation is intentionally retained
      // and charged in this first composable correction probe.
      for(std::size_t q=0;q<weights.size();++q)
      {
        double s=0.,speed=0.;for(int j=0;j<n;++j)s+=shapes[q][j]*sigma[SigmaIndex(e,j)];const auto u=Point(physical,e,shapes[q]);const auto f=physics.Flux(u.data(),s,speed);++work.flux_points;
        for(int i=0;i<m;++i)for(int c=0;c<4;++c)load[4*i+c]-=weights[q]*low_derivatives[q][i]*f[c];
      }
      for(int c=0;c<4;++c){double sum=0.;for(int i=0;i<m;++i)sum+=load[4*i+c];maximum_entropy_mean=std::max(maximum_entropy_mean,std::abs(sum));}
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<4;++c)low[(e*m+i)*4+c]+=retained.mass_inverse(i,j)*load[4*j+c]/h;
    }
    last_entropy_correction=state_transfer.Apply(state_transfer.E,low,ne,4);for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=last_entropy_correction[i];
    const auto totals=Integrals(rhs,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    const double elapsed=Wall(start);entropy_seconds+=elapsed;work.transport+=elapsed;return rhs;
  }
  // Exact directional derivative of the quadrature gas entropy on the actual
  // conservative primitive manifold, including its weighted velocity solve.
  double EntropyRate(const State& moments,const State& rhs)const
  {
    const State& physical=Reconstruct(moments);const State low=state_transfer.Apply(state_transfer.P,rhs,ne,4);const int m=retained.n;double rate=0.;
    for(int e=0;e<ne;++e)
    {
      BandMatrix weighted(m,m-1);State dv(2*m,0.),drho(weights.size(),0.),denergy(weights.size(),0.),kt(weights.size(),0.),kload(m,0.),kproject(m,0.);std::vector<EntropyCoupledPoint> a(weights.size());
      for(std::size_t q=0;q<weights.size();++q)
      {
        a[q]=EntropyCoupledState(Point(physical,e,shapes[q]),0.);std::array<double,2> dm{};
        for(int i=0;i<m;++i){drho[q]+=low_shapes[q][i]*low[(e*m+i)*4];denergy[q]+=low_shapes[q][i]*low[(e*m+i)*4+3];for(int c=0;c<2;++c)dm[c]+=low_shapes[q][i]*low[(e*m+i)*4+c+1];}
        for(int i=0;i<m;++i){for(int j=0;j<m;++j)weighted.At(i,j)+=weights[q]*a[q].rho*low_shapes[q][i]*low_shapes[q][j];dv[i]+=weights[q]*low_shapes[q][i]*(dm[0]-drho[q]*a[q].vx);dv[m+i]+=weights[q]*low_shapes[q][i]*(dm[1]-drho[q]*a[q].vy);}
      }
      BandFactor factor(weighted,0,m);factor.Solve(dv,2);
      for(std::size_t q=0;q<weights.size();++q)
      {double vx=0.,vy=0.;for(int i=0;i<m;++i){vx+=low_shapes[q][i]*dv[i];vy+=low_shapes[q][i]*dv[m+i];}kt[q]=.5*drho[q]*(a[q].vx*a[q].vx+a[q].vy*a[q].vy)+a[q].rho*(a[q].vx*vx+a[q].vy*vy);for(int i=0;i<m;++i)kload[i]+=weights[q]*low_shapes[q][i]*kt[q];}
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)kproject[i]+=retained.mass_inverse(i,j)*kload[j];
      for(std::size_t q=0;q<weights.size();++q)
      {double represented=0.;for(int i=0;i<m;++i)represented+=low_shapes[q][i]*kproject[i];const double dp=(gamma_euler-1.)*(denergy[q]-represented);const double entropy=std::log(a[q].pressure)-gamma_euler*a[q].logrho;rate+=(edges[e+1]-edges[e])*weights[q]*((gamma_euler-entropy)/(gamma_euler-1.)*drho[q]-a[q].rho/((gamma_euler-1.)*a[q].pressure)*dp);}
    }
    return rate;
  }
};

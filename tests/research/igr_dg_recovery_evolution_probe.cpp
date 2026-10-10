// Test-only local nonlinear-variable recovery for the current DG/IGR source.
// Conservative Euler polynomials and physical fluxes are unchanged. Default
// source-only modes retain SIPG; central-Gram changes the elliptic discretization.
#define DG_IGR_DG_BOUNDARY_LIBRARY_ONLY
#include "igr_dg_boundary_evolution_probe.cpp"
#undef DG_IGR_DG_BOUNDARY_LIBRARY_ONLY

namespace
{
struct RecoveryOptions
{
  BoundaryOptions boundary;
  std::string recovery="nodal",faces="broken",closure_operator="sipg",source_product="square";
  std::string volume_projection="weak";
  int rational_degree=5,volume_gradient_degree=-1;
};
struct RecoveryProblem:BoundaryProblem
{
  RecoveryOptions recovery_options;
  std::vector<State> recovery_derivative,gradient_projection;
  mutable double recovery_seconds=0.;
  mutable std::uint64_t recovered_cells=0,recovery_products=0;
  mutable std::uint64_t chain_rule_points=0;
  mutable double maximum_volume_mean_correction=0.;
  explicit RecoveryProblem(const RecoveryOptions& options):BoundaryProblem(options.boundary),recovery_options(options)
  {
    if(options.volume_projection!="weak"&&options.volume_projection!="chain-mean"&&options.volume_projection!="chain-mean-lgl")
      throw std::runtime_error("unknown volume projection");
    if(options.volume_projection!="weak"&&compatibility.formulation!="baseline")
      throw std::runtime_error("chain-rule volume requires baseline physical stress/energy traces");
    if(options.recovery!="existing"&&compatibility.formulation!="baseline")
      throw std::runtime_error("recovery source controls retain baseline SIPG operator");
    if(options.closure_operator!="sipg"&&(options.closure_operator!="central-gram"||options.recovery!="nodal"||options.faces!="central"))
      throw std::runtime_error("central Gram requires nodal velocity and central source faces");
    if(options.boundary.open_closure!="natural"||options.boundary.wall_closure!="natural")
      throw std::runtime_error("bounded recovery screen requires natural scalar boundaries");
    if(options.recovery=="variational"&&options.faces!="central")
      throw std::runtime_error("variational velocity recovery requires central source faces");
    if(options.source_product!="square"&&
       (options.source_product!="opposite"||options.faces!="central"||options.closure_operator!="sipg"||
        (options.recovery!="nodal"&&options.recovery!="variational")))
      throw std::runtime_error("opposite source product requires nodal/variational central recovery with SIPG");
    recovery_derivative=derivatives;
    if(options.volume_gradient_degree!=-1)
    {
      const int degree=options.volume_gradient_degree;
      if(options.recovery!="nodal"||degree<0||degree>=p)
        throw std::runtime_error("volume gradient degree requires nodal recovery and 0 <= degree < p");
      // Reference-cell L2 projection of the broken nodal derivative only.
      // Face lifts retain their full degree and the SIPG matrix is unchanged.
      std::vector<State> B(weights.size(),State(degree+1));
      for(std::size_t q=0;q<weights.size();++q)
      {
        double z=0.;for(int i=0;i<n;++i)z+=shapes[q][i]*physics.basis.nodes[i];z=2.*z-1.;
        B[q][0]=1.;if(degree>0)B[q][1]=z;
        for(int k=2;k<=degree;++k)B[q][k]=((2*k-1)*z*B[q][k-1]-(k-1)*B[q][k-2])/k;
      }
      for(std::size_t q=0;q<weights.size();++q)for(int i=0;i<n;++i)
      {
        double value=0.;
        for(std::size_t r=0;r<weights.size();++r)for(int k=0;k<=degree;++k)
          value+=(2*k+1)*B[q][k]*B[r][k]*weights[r]*derivatives[r][i];
        recovery_derivative[q][i]=value;
      }
    }
    if(options.recovery=="rational")
    {
      const int degree=options.rational_degree<0?p+options.rational_degree:options.rational_degree;
      if(degree<0||degree>p)throw std::runtime_error("invalid Floater-Hormann degree");
      State bw(n,0.);const auto& nodes=physics.basis.nodes;
      for(int i=0;i<n;++i)for(int k=std::max(0,i-degree);k<=std::min(i,p-degree);++k)
      {double value=k%2?-1.:1.;for(int j=k;j<=k+degree;++j)if(j!=i)value/=nodes[i]-nodes[j];bw[i]+=value;}
      double scale=0.;for(double value:bw)scale=std::max(scale,std::abs(value));for(double& value:bw)value/=scale;
      for(std::size_t q=0;q<weights.size();++q)
      {
        // Recover the Gauss point from its exact polynomial coordinate image.
        double z=0.;for(int i=0;i<n;++i)z+=shapes[q][i]*nodes[i];
        double sum=0.,diff=0.;State terms(n);int hit=-1;
        for(int i=0;i<n;++i)if(std::abs(z-nodes[i])<1.e-13)hit=i;
        if(hit>=0)
        {
          std::fill(recovery_derivative[q].begin(),recovery_derivative[q].end(),0.);
          for(int i=0;i<n;++i)if(i!=hit)
          {const double value=bw[i]/(bw[hit]*(nodes[hit]-nodes[i]));recovery_derivative[q][i]=value;recovery_derivative[q][hit]-=value;}
        }
        else
        {
          for(int i=0;i<n;++i){terms[i]=bw[i]/(z-nodes[i]);sum+=terms[i];diff-=terms[i]/(z-nodes[i]);}
          if(!std::isfinite(sum)||std::abs(sum)<1.e-100)throw std::runtime_error("invalid rational basis denominator");
          for(int i=0;i<n;++i)recovery_derivative[q][i]=-terms[i]/((z-nodes[i])*sum)-terms[i]*diff/(sum*sum);
        }
      }
    }
    if(options.recovery=="gradient")
    {
      // L2 projection of the exact conservative-quotient gradient to P_(p-1).
      // Legendre mass is diagonal on [0,1]; Gauss rule integrates the matrix.
      std::vector<State> B(weights.size(),State(p));
      for(std::size_t q=0;q<weights.size();++q)
      {
        double z=0.;for(int i=0;i<n;++i)z+=shapes[q][i]*physics.basis.nodes[i];z=2.*z-1.;B[q][0]=1.;
        if(p>1)B[q][1]=z;for(int k=2;k<p;++k)B[q][k]=((2*k-1)*z*B[q][k-1]-(k-1)*B[q][k-2])/k;
      }
      gradient_projection.assign(weights.size(),State(weights.size(),0.));
      for(std::size_t q=0;q<weights.size();++q)for(std::size_t r=0;r<weights.size();++r)
        for(int k=0;k<p;++k)gradient_projection[q][r]+=(2*k+1)*B[q][k]*B[r][k]*weights[r];
    }
  }
  State Residual(const State& state,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    if(recovery_options.volume_projection=="weak")return BoundaryProblem::Residual(state,sigma,time,work,budget);
    // A new spatial discretization: interpolate the true nonlinear flux
    // divergence, then change only its constant mode to preserve the flux
    // integral. This is neither D*I_p(F) nor the unavailable fine-state oracle.
    // Reuse the full-energy directional flux derivative and shared face flux.
    const auto start=Clock::now();State result(state.size(),0.),speeds(ne*n);
    std::vector<std::array<double,4>> flux(ne*n);
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];std::array<double,4> mean{};
      for(int i=0;i<n;++i)
      {
        std::array<double,4> du{};double ds=0.;
        for(int j=0;j<n;++j)
        {
          const double d=physics.basis.D(i,j)/h;
          for(int c=0;c<4;++c)du[c]+=d*state[Id(e,j,c)];
          ds+=d*sigma[SigmaIndex(e,j)];
        }
        const auto H=DifferentiateFlux(physics,&state[Id(e,i,0)],du.data(),sigma[SigmaIndex(e,i)],ds);
        flux[e*n+i]=physics.Flux(&state[Id(e,i,0)],sigma[SigmaIndex(e,i)],speeds[e*n+i]);
        ++work.flux_points;++chain_rule_points;
        for(int c=0;c<4;++c)
        {result[Id(e,i,c)]=-H.flux[c];mean[c]+=physics.basis.weights[i]*H.flux[c];}
      }
      for(int c=0;c<4;++c)
      {
        const double correction=mean[c]-(flux[e*n+p][c]-flux[e*n][c])/h;
        maximum_volume_mean_correction=std::max(maximum_volume_mean_correction,std::abs(correction));
        for(int i=0;i<n;++i)result[Id(e,i,c)]+=correction;
      }
    }
    budget.fill(0.);
    for(int face=0;face<=ne;++face)
    {
      std::array<double,4> numerical{};
      if(face==0||face==ne)numerical=BoundaryFlux(state,sigma,face==ne);
      else
      {
        const double speed=physics.FaceSpeed(speeds[(face-1)*n+p],speeds[face*n]);
        for(int c=0;c<4;++c)numerical[c]=.5*(flux[(face-1)*n+p][c]+flux[face*n][c]
            -speed*(state[Id(face,0,c)]-state[Id(face-1,p,c)]));
      }
      ++work.faces;
      for(int side=0;side<2;++side)
      {
        const int e=side?face:face-1,j=side?0:p;const double sign=side?1.:-1.;
        if(e<0||e>=ne)continue;const double h=edges[e+1]-edges[e];
        for(int i=0;i<n;++i)
        {
          const double lift=recovery_options.volume_projection=="chain-mean-lgl"?
              (i==j?1./physics.basis.weights[j]:0.):mass_inverse(i,j);
          for(int c=0;c<4;++c)result[Id(e,i,c)]+=sign*lift*(numerical[c]-flux[e*n+j][c])/h;
        }
      }
      for(int c=0;c<4;++c){if(face==0)budget[c]=numerical[c];if(face==ne)budget[c]-=numerical[c];}
    }
    const auto totals=Integrals(result,edges);
    for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    work.transport+=Wall(start);return result;
  }
  State Closure(const State& state,double time,TrackWork& work)const
  {
    if(recovery_options.recovery=="existing")return BoundaryProblem::Closure(state,time,work);
    const auto start=Clock::now();BandMatrix a;State b;AssembleClosure(state,time,work,a,b);
    if(recovery_options.closure_operator=="central-gram")
    {
      // Symmetric central DG gradient, natural stress boundary. Its weighted
      // Gram is a distinct screened discretization, not a SIPG optimization.
      a=BandMatrix(ns,n+1);
      for(int e=0;e<ne;++e)for(std::size_t q=0;q<weights.size();++q)
      {
        const double h=edges[e+1]-edges[e],weight=h*weights[q]/Point(state,e,shapes[q])[0];
        std::vector<int> idx(n);State g(n);for(int i=0;i<n;++i){idx[i]=e*n+i;g[i]=derivatives[q][i]/h;}
        for(int side=0;side<2;++side)
        {
          const int neighbor=e+(side?1:-1);if(neighbor<0||neighbor>=ne)continue;
          const int endpoint=side?p:0;double lift=0.;for(int i=0;i<n;++i)lift+=shapes[q][i]*mass_inverse(i,endpoint)/h;
          lift*=side?-.5:.5;g[endpoint]+=lift;idx.push_back(neighbor*n+(side?0:p));g.push_back(-lift);
        }
        for(int i=0;i<n;++i)for(int j=0;j<n;++j)a.At(e*n+i,e*n+j)+=weight*shapes[q][i]*shapes[q][j];
        for(std::size_t i=0;i<idx.size();++i)for(std::size_t j=0;j<idx.size();++j)a.At(idx[i],idx[j])+=weight*o.base.alpha*g[i]*g[j];
      }
      ++work.closure_assemblies;
    }
    // The redundant original source computation is deliberately charged.
    std::fill(b.begin(),b.end(),0.);const auto rs=Clock::now();double integral=0.;
    // Local L2 projection, not an H1 projection. It is the variational
    // momentum derivative of the exact-mass discrete kinetic energy. Halo
    // traces must use this same projected field for the central adjoint.
    State projected_velocity;
    if(recovery_options.recovery=="variational")
    {
      projected_velocity=Velocity(state);
      recovery_products+=std::uint64_t(ne)*(5*weights.size()*n+n*n);
    }
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];State v(n),g(weights.size(),0.);
      for(int i=0;i<n;++i)v[i]=projected_velocity.empty()?
          state[Id(e,i,1)]/state[Id(e,i,0)]:projected_velocity[e*n+i];
      if(recovery_options.recovery=="gradient")
      {
        State raw(weights.size(),0.);
        for(std::size_t q=0;q<weights.size();++q)
        {const auto u=Point(state,e,shapes[q]);double dr=0.,dm=0.;for(int i=0;i<n;++i){dr+=derivatives[q][i]*state[Id(e,i,0)];dm+=derivatives[q][i]*state[Id(e,i,1)];}
          raw[q]=(dm-dr*u[1]/u[0])/(h*u[0]);}
        for(std::size_t q=0;q<weights.size();++q)for(std::size_t r=0;r<weights.size();++r)g[q]+=gradient_projection[q][r]*raw[r];
        recovery_products+=weights.size()*weights.size();
      }
      else
      {for(std::size_t q=0;q<weights.size();++q)for(int i=0;i<n;++i)g[q]+=recovery_derivative[q][i]*v[i]/h;
        recovery_products+=weights.size()*n;}
      for(std::size_t q=0;q<weights.size();++q)
      {
        double half_disagreement=0.;
        for(int side=0;side<2;++side)
        {
          const int neighbor=e+(side?1:-1),endpoint=side?p:0;double jump=0.;
          if(neighbor>=0&&neighbor<ne&&recovery_options.faces=="central")
          {const double other=projected_velocity.empty()?
              state[Id(neighbor,side?0:p,1)]/state[Id(neighbor,side?0:p,0)]:projected_velocity[neighbor*n+(side?0:p)];
            jump=.5*(v[endpoint]-other);}
          if((neighbor<0||neighbor>=ne)&&boundary.wall_source=="zero"&&(side?boundary.right:boundary.left)=="wall")jump=v[endpoint];
          if(jump!=0.)
          {
            double lift=0.;for(int i=0;i<n;++i)lift+=shapes[q][i]*mass_inverse(i,endpoint);
            const double correction=(side?-1.:1.)*jump*lift/h;g[q]+=correction;
            // The two opposite derivatives use the same physical boundary
            // trace; only their interior one-sided lifts differ.
            if(recovery_options.source_product=="opposite"&&neighbor>=0&&neighbor<ne)
              half_disagreement+=(side?-correction:correction);
          }
        }
        // g_minus*g_plus = g_central^2 - ((g_minus-g_plus)/2)^2.
        // This signed source is a new discrete control. Never clip it.
        const double value=recovery_options.source_product=="opposite"?
            2.*o.base.alpha*(g[q]*g[q]-half_disagreement*half_disagreement):2.*o.base.alpha*g[q]*g[q];
        const double weight=h*weights[q];integral+=weight*value;
        for(int i=0;i<n;++i)b[e*n+i]+=weight*shapes[q][i]*value;
        ++work.source_points;
      }
      ++recovered_cells;
    }
    recovery_seconds+=Wall(rs);work.current_source_integral=integral;
    return SolveCurrentSystem(state,a,b,work,start,{{0.,0.}},{{0.,0.}});
  }
};
RecoveryOptions RecoveryParse(int argc,char** argv)
{
  RecoveryOptions o;std::vector<char*> common{argv[0]};
  for(int i=1;i<argc;++i)
  {const std::string flag=argv[i];if(flag=="--recovery"||flag=="--source-faces"||flag=="--rational-degree"||flag=="--recovery-operator"||flag=="--volume-gradient-degree"||flag=="--source-product"||flag=="--volume-projection")
    {if(++i==argc)throw std::runtime_error("missing recovery option");if(flag=="--recovery")o.recovery=argv[i];else if(flag=="--source-faces")o.faces=argv[i];else if(flag=="--recovery-operator")o.closure_operator=argv[i];else if(flag=="--volume-gradient-degree")o.volume_gradient_degree=std::stoi(argv[i]);else if(flag=="--source-product")o.source_product=argv[i];else if(flag=="--volume-projection")o.volume_projection=argv[i];else o.rational_degree=std::stoi(argv[i]);}
    else common.push_back(argv[i]);}
  o.boundary=BoundaryParse(int(common.size()),common.data(),o.recovery!="existing");
  if(o.recovery!="existing"&&o.recovery!="nodal"&&o.recovery!="rational"&&o.recovery!="gradient"&&o.recovery!="variational")throw std::runtime_error("unknown recovery");
  if(o.faces!="broken"&&o.faces!="central")throw std::runtime_error("unknown recovery face policy");
  return o;
}
void RecoverySelfTest()
{
  BoundarySelfTest();
  for(const std::string projection:{"chain-mean","chain-mean-lgl"})for(int degree:{3,7,11})
  {
    RecoveryOptions o;o.volume_projection=projection;o.recovery="existing";
    auto& t=o.boundary.compatibility.track;t.initial="simple-wave";t.base.elements=8;t.base.order=degree;
    t.closure_space="l2";t.closure_boundary="natural";
    RecoveryProblem pr(o);State u=pr.Initial(),s(pr.ns);TrackWork work;std::array<double,4> budget{};
    // Discontinuous, nonconstant state exercises both shared faces and the
    // nonlinear mean correction; the independent integral must telescope.
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {
      const double z=pr.physics.basis.nodes[i],rho=1.+.1*z+.01*e,v=.2+.03*z*z+.005*e;
      u[pr.Id(e,i,0)]=rho;u[pr.Id(e,i,1)]=rho*v;u[pr.Id(e,i,2)]=.1*rho;
      u[pr.Id(e,i,3)]=3.+.5*rho*(v*v+.01);s[e*pr.n+i]=.04+.01*z*z;
    }
    pr.Residual(u,s,0.,work,budget);
    if(work.max_rhs_defect>3.e-10)throw std::runtime_error("chain-rule conservative residual failed");
    // Shared flux derivative is checked against centered differentiation of
    // the actual physical flux, including transverse momentum and sigma_x.
    const double du[4]={.12,-.23,.07,.19},ds=-.13,epsilon=1.e-6;
    std::array<double,4> plus{},minus{};for(int c=0;c<4;++c){plus[c]=u[c]+epsilon*du[c];minus[c]=u[c]-epsilon*du[c];}
    double speed=0.;const auto fp=pr.physics.Flux(plus.data(),s[0]+epsilon*ds,speed),fm=pr.physics.Flux(minus.data(),s[0]-epsilon*ds,speed);
    const auto tangent=DifferentiateFlux(pr.physics,u.data(),du,s[0],ds);
    for(int c=0;c<4;++c)if(std::abs((fp[c]-fm[c])/(2*epsilon)-tangent.flux[c])>2.e-9)
      throw std::runtime_error("chain-rule shared flux derivative failed");
  }
  for(const std::string method:{"nodal","rational","gradient","variational"})for(int degree:{3,7,11,15})
  {
    RecoveryOptions o;o.recovery=method;o.rational_degree=-2;o.boundary.compatibility.track.initial="simple-wave";
    if(method=="variational")o.faces="central";
    auto& t=o.boundary.compatibility.track;t.base.elements=8;t.base.order=degree;t.closure_space="l2";t.closure_boundary="natural";
    RecoveryProblem pr(o);TrackWork work;State u=pr.Initial();
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i],v=.2+.1*x;
      u[pr.Id(e,i,0)]=2.;u[pr.Id(e,i,1)]=2.*v;u[pr.Id(e,i,2)]=0.;u[pr.Id(e,i,3)]=3.+v*v;}
    const auto s=pr.Closure(u,0.,work);for(double value:s)if(std::abs(value-4.*t.base.alpha*.01)>1.e-10)throw std::runtime_error("recovered affine gradient closure failed");
  }
  for(int degree:{3,7})
  {
    RecoveryOptions o;o.recovery="nodal";o.faces="central";o.source_product="opposite";
    auto& t=o.boundary.compatibility.track;t.initial="simple-wave";t.base.elements=8;t.base.order=degree;
    t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=17;
    RecoveryProblem pr(o);State u=pr.Initial();TrackWork work;
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {const double v=e>=pr.ne/2?.4:.1;u[pr.Id(e,i,0)]=2.;u[pr.Id(e,i,1)]=2*v;u[pr.Id(e,i,2)]=0.;u[pr.Id(e,i,3)]=3.+v*v;}
    const auto jump_sigma=pr.Closure(u,0.,work);
    for(double value:jump_sigma)if(std::abs(value)>2.e-10)
      throw std::runtime_error("opposite source should vanish for an isolated two-cell constant jump");
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i],v=.2+.1*x;
      u[pr.Id(e,i,1)]=2*v;u[pr.Id(e,i,3)]=3.+v*v;}
    const auto affine_sigma=pr.Closure(u,0.,work);
    for(double value:affine_sigma)if(std::abs(value-4*t.base.alpha*.01)>2.e-10)
      throw std::runtime_error("opposite source affine continuous gradient failed");
  }
  {
    RecoveryOptions o;o.recovery="variational";o.faces="central";
    auto& t=o.boundary.compatibility.track;t.initial="simple-wave";t.base.elements=8;t.base.order=7;
    t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=17;
    RecoveryProblem pr(o);State u=pr.Initial(),sigma(pr.ns),load(pr.ns,0.);
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {
      const double z=pr.physics.basis.nodes[i];
      u[pr.Id(e,i,1)]+=.003*(e%3-1)*u[pr.Id(e,i,0)];
      sigma[e*pr.n+i]=.3+.05*std::legendre(7,2*z-1.)+.01*e;
    }
    const State velocity=pr.Velocity(u);double adjoint=0.,direct=0.;
    for(int e=0;e<pr.ne;++e)
    {
      const double h=pr.edges[e+1]-pr.edges[e];
      for(std::size_t q=0;q<pr.weights.size();++q)
      {
        double s=0.,g=0.;for(int i=0;i<pr.n;++i)
        {s+=pr.shapes[q][i]*sigma[e*pr.n+i];g+=pr.derivatives[q][i]*velocity[e*pr.n+i]/h;}
        for(int i=0;i<pr.n;++i)load[e*pr.n+i]+=pr.weights[q]*pr.derivatives[q][i]*s;
        for(int side=0;side<2;++side)
        {
          const int neighbor=e+(side?1:-1),endpoint=side?pr.p:0;if(neighbor<0||neighbor>=pr.ne)continue;
          const double jump=velocity[e*pr.n+endpoint]-velocity[neighbor*pr.n+(side?0:pr.p)];
          for(int i=0;i<pr.n;++i)g+=(side?-.5:.5)*jump*pr.shapes[q][i]*pr.mass_inverse(i,endpoint)/h;
        }
        adjoint+=h*pr.weights[q]*s*g;
      }
    }
    for(int e=0;e<pr.ne-1;++e)
    {const double trace=.5*(sigma[e*pr.n+pr.p]+sigma[(e+1)*pr.n]);load[e*pr.n+pr.p]-=trace;load[(e+1)*pr.n]+=trace;}
    load[0]+=sigma[0];load.back()-=sigma.back();
    for(int e=0;e<pr.ne;++e)
    {
      State residual(pr.n,0.);for(int i=0;i<pr.n;++i)for(int j=0;j<pr.n;++j)residual[i]+=pr.mass_inverse(i,j)*load[e*pr.n+j];
      for(std::size_t q=0;q<pr.weights.size();++q)
      {const auto value=pr.Point(u,e,pr.shapes[q]);double r=0.;for(int i=0;i<pr.n;++i)r+=pr.shapes[q][i]*residual[i];
        direct+=pr.weights[q]*(value[1]/value[0])*r;}
    }
    const double boundary=velocity.back()*sigma.back()-velocity.front()*sigma.front();
    if(std::abs(direct+boundary-adjoint)>2.e-11*std::max(1.,std::abs(adjoint)))
      throw std::runtime_error("variational central physical stress-work identity failed");
  }
  {
    RecoveryOptions o;o.volume_gradient_degree=4;
    auto& t=o.boundary.compatibility.track;t.initial="simple-wave";t.base.elements=8;t.base.order=7;t.closure_space="l2";t.closure_boundary="natural";
    RecoveryProblem pr(o);State v(pr.n);
    for(int i=0;i<pr.n;++i)v[i]=std::legendre(7,2.*pr.physics.basis.nodes[i]-1.);
    for(std::size_t q=0;q<pr.weights.size();++q)
    {
      double z=0.,actual=0.;for(int i=0;i<pr.n;++i){z+=pr.shapes[q][i]*pr.physics.basis.nodes[i];actual+=pr.recovery_derivative[q][i]*v[i];}
      z=2.*z-1.;const double expected=2.*(9.*std::legendre(4,z)+5.*std::legendre(2,z)+1.);
      if(std::abs(actual-expected)>1.e-9)throw std::runtime_error("volume gradient projection failed independent Legendre identity");
    }
  }
  std::cout<<"DG source recovery self-test passed\n";
}
}
#ifndef DG_IGR_DG_RECOVERY_LIBRARY_ONLY
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    const auto o=RecoveryParse(argc,argv);if(o.boundary.compatibility.track.self_test){RecoverySelfTest();return 0;}
    const auto start=Clock::now();RecoveryProblem pr(o);const double setup=Wall(start);auto result=CompatibilityRun(pr);result.work.setup=setup;
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\""<<o.recovery<<"\",\"recovery_operator\":\""<<o.closure_operator<<"\",\"source_recovery_faces\":\""<<o.faces
      <<"\",\"rational_degree_option\":"<<o.rational_degree<<",\"volume_gradient_degree_option\":"<<o.volume_gradient_degree<<",\"recovery_seconds\":"<<pr.recovery_seconds
      <<",\"recovered_cells\":"<<pr.recovered_cells<<",\"recovery_scalar_products\":"<<pr.recovery_products
      <<",\"source_product\":\""<<o.source_product<<"\",\"volume_projection\":\""<<o.volume_projection
      <<"\",\"chain_rule_points\":"<<pr.chain_rule_points<<",\"maximum_volume_mean_correction\":"<<pr.maximum_volume_mean_correction<<"\n";
    BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif

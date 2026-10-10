#pragma once
// Test-only full-moment, minimum-gradient velocity recovery. At fixed rho_p,
// minimize integral v_x^2 in H1 subject to every Pp moment of rho_p*v.
// The exact minimizer lies in broken P(2p+2); local KKT elimination leaves
// one scalar tridiagonal solve for common face velocities. No p modes removed.
namespace
{
struct MinimumGradientResult
{
  State coefficients,face_values,load;
  double energy=0.,source_integral=0.,minimum_source=INFINITY,minimum_density=INFINITY;
  double moment_defect=0.,value_jump=0.,derivative_jump=0.,boundary_gradient=0.;
  double kkt_residual=0.,face_residual=0.;
};
struct MinimumGradientRecovery
{
  int p,n,q,r,nq;
  State weights;
  std::vector<State> values,derivatives,tests,nodal,nodal_derivatives;
  mfem::DenseMatrix gram,modal;
  static void Legendre(double x,int degree,State& v,State& d)
  {
    v.assign(degree+1,0.);d.assign(degree+1,0.);v[0]=1.;
    if(degree){v[1]=x;d[1]=1.;}
    for(int k=2;k<=degree;++k)
    {v[k]=((2*k-1)*x*v[k-1]-(k-1)*v[k-2])/k;
     d[k]=((2*k-1)*(v[k-1]+x*d[k-1])-(k-1)*d[k-2])/k;}
  }
  explicit MinimumGradientRecovery(const TrackingProblem& retained):
    p(retained.p),n(p+1),q(2*p+2),r(q+1),nq(0),gram(r),modal(n)
  {
    // Integrates rho_p*Pp*P(2p+2), H, and v_x^2*Pp. MFEM segment
    // points/weights live on [0,1]; the Legendre matrices use [-1,1].
    const auto& rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,std::max(5*p+2,2*(2*q+8)-1));
    nq=rule.GetNPoints();gram=0.;modal=0.;
    for(int k=0;k<nq;++k)
    {
      const auto& point=rule.IntPoint(k);if(!(point.weight>0.))throw std::runtime_error("minimum-gradient quadrature is not positive");
      State v,d;Legendre(2*point.x-1.,q,v,d);values.push_back(v);derivatives.push_back(d);
      tests.push_back(State(v.begin(),v.begin()+n));nodal.push_back(retained.Shape(point.x,false));weights.push_back(2*point.weight);
      nodal_derivatives.push_back(retained.Shape(point.x,true));
      for(int i=0;i<r;++i)for(int j=0;j<r;++j)gram(i,j)+=2*point.weight*d[i]*d[j];
      for(int i=0;i<n;++i)for(int j=0;j<n;++j)modal(i,j)+=(2*i+1)*point.weight*v[i]*nodal.back()[j];
    }
    if(2*nq-1<5*p+2)throw std::runtime_error("minimum-gradient source quadrature is insufficient");
  }
  MinimumGradientResult Recover(const State& moments,const State& edges,double alpha)const
  {
    const int ne=int(edges.size())-1,size=r+n+2;
    if(ne<1||moments.size()!=std::size_t(ne*n*4))throw std::runtime_error("minimum-gradient input shape mismatch");
    MinimumGradientResult out;out.coefficients.assign(ne*r,0.);out.load.assign(ne*n,0.);
    State diagonal(ne+1,0.),upper(ne,0.),lower(ne,0.),rhs(ne+1,0.);
    std::vector<mfem::DenseMatrix> responses,constraints;responses.reserve(ne);constraints.reserve(ne);
    State all_moments(ne*n);
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];if(!(h>0.))throw std::runtime_error("minimum-gradient mesh is not ordered");
      mfem::DenseMatrix C(n+2,r),H(r),K(size),target(size,3),solution(size,3),response(r,3);
      C=0.;H=0.;K=0.;target=0.;State scale(r);
      for(int i=0;i<r;++i)
      {
        for(int j=0;j<r;++j)H(i,j)=2.*gram(i,j)/h;
        scale[i]=1./std::sqrt(std::max(H(i,i),1.));C(n,i)=i%2?-1.:1.;C(n+1,i)=1.;
      }
      for(int k=0;k<nq;++k)
      {
        double rho=0.;for(int j=0;j<n;++j)rho+=nodal[k][j]*moments[(e*n+j)*4];
        if(!(rho>floor_value)||!std::isfinite(rho))throw std::runtime_error("minimum-gradient density admission failed");
        out.minimum_density=std::min(out.minimum_density,rho);
        for(int i=0;i<n;++i)for(int j=0;j<r;++j)C(i,j)+=.5*(2*i+1)*weights[k]*rho*tests[k][i]*values[k][j];
      }
      for(int i=0;i<n;++i)
      {double m=0.;for(int j=0;j<n;++j)m+=modal(i,j)*moments[(e*n+j)*4+1];all_moments[e*n+i]=m;target(r+i,0)=m;}
      target(r+n,1)=1.;target(r+n+1,2)=1.;
      for(int i=0;i<r;++i)
      {
        for(int j=0;j<r;++j)K(i,j)=scale[i]*H(i,j)*scale[j];
        for(int j=0;j<n+2;++j)K(i,r+j)=K(r+j,i)=scale[i]*C(j,i);
      }
      // KKT is indefinite: use pivoted LU, never the SPD factorization.
      mfem::DenseMatrixInverse inverse(K,false);inverse.Mult(target,solution);
      for(int i=0;i<r;++i)for(int c=0;c<3;++c)response(i,c)=scale[i]*solution(i,c);
      for(int i=0;i<size;++i)for(int c=0;c<3;++c)
      {
        double residual=-target(i,c);for(int j=0;j<size;++j)residual+=K(i,j)*solution(j,c);
        out.kkt_residual=std::max(out.kkt_residual,std::abs(residual));
      }
      double cell[2][2]{},load[2]{};
      for(int i=0;i<r;++i)for(int j=0;j<r;++j)for(int a=0;a<2;++a)
      {
        load[a]-=response(i,a+1)*H(i,j)*response(j,0);
        for(int b=0;b<2;++b)cell[a][b]+=response(i,a+1)*H(i,j)*response(j,b+1);
      }
      diagonal[e]+=cell[0][0];diagonal[e+1]+=cell[1][1];upper[e]+=cell[0][1];lower[e]+=cell[1][0];rhs[e]+=load[0];rhs[e+1]+=load[1];
      responses.push_back(std::move(response));constraints.push_back(std::move(C));
    }
    const State original_diagonal=diagonal,original_rhs=rhs;out.face_values=rhs;
    for(int e=1;e<=ne;++e)
    {
      if(!(diagonal[e-1]>0.)||!std::isfinite(diagonal[e-1]))throw std::runtime_error("minimum-gradient face pivot failed");
      const double factor=lower[e-1]/diagonal[e-1];diagonal[e]-=factor*upper[e-1];out.face_values[e]-=factor*out.face_values[e-1];
    }
    if(!(diagonal[ne]>0.)||!std::isfinite(diagonal[ne]))throw std::runtime_error("minimum-gradient final face pivot failed");
    out.face_values[ne]/=diagonal[ne];
    for(int e=ne-1;e>=0;--e)out.face_values[e]=(out.face_values[e]-upper[e]*out.face_values[e+1])/diagonal[e];
    State endpoint_value(2*ne),endpoint_derivative(2*ne);
    for(int e=0;e<=ne;++e)
    {
      double residual=original_diagonal[e]*out.face_values[e]-original_rhs[e];
      if(e)residual+=lower[e-1]*out.face_values[e-1];if(e<ne)residual+=upper[e]*out.face_values[e+1];
      out.face_residual=std::max(out.face_residual,std::abs(residual));
    }
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];const auto& response=responses[e];const auto& C=constraints[e];
      for(int i=0;i<r;++i)
      {
        const double a=response(i,0)+response(i,1)*out.face_values[e]+response(i,2)*out.face_values[e+1];
        if(!std::isfinite(a))throw std::runtime_error("nonfinite minimum-gradient coefficient");out.coefficients[e*r+i]=a;
        endpoint_value[2*e]+=(i%2?-1.:1.)*a;endpoint_value[2*e+1]+=a;
        const double slope=i*(i+1)/h;endpoint_derivative[2*e]+=(i%2?1.:-1.)*slope*a;endpoint_derivative[2*e+1]+=slope*a;
      }
      for(int i=0;i<n;++i)
      {double m=-all_moments[e*n+i];for(int j=0;j<r;++j)m+=C(i,j)*out.coefficients[e*r+j];out.moment_defect=std::max(out.moment_defect,std::abs(m));}
      for(int k=0;k<nq;++k)
      {
        double g=0.;for(int j=0;j<r;++j)g+=2./h*derivatives[k][j]*out.coefficients[e*r+j];
        const double source=2.*alpha*g*g,W=.5*h*weights[k];out.energy+=W*g*g;out.source_integral+=W*source;out.minimum_source=std::min(out.minimum_source,source);
        for(int i=0;i<n;++i)out.load[e*n+i]+=W*nodal[k][i]*source;
      }
    }
    for(int e=0;e+1<ne;++e)
    {out.value_jump=std::max(out.value_jump,std::abs(endpoint_value[2*e+1]-endpoint_value[2*e+2]));out.derivative_jump=std::max(out.derivative_jump,std::abs(endpoint_derivative[2*e+1]-endpoint_derivative[2*e+2]));}
    out.boundary_gradient=std::max(std::abs(endpoint_derivative.front()),std::abs(endpoint_derivative.back()));
    if(out.moment_defect>1.e-8||out.value_jump>1.e-8||out.derivative_jump>1.e-6||out.boundary_gradient>1.e-6||out.kkt_residual>1.e-7)
      throw std::runtime_error("minimum-gradient constraint certificate failed");
    return out;
  }
};

struct MinimumGradientCommonState
{
  State pressure,pressure_legendre;
  double moment_defect=0.;
  MinimumGradientCommonState(const TrackingProblem& retained,const MinimumGradientRecovery& recovery,
                             const MinimumGradientResult& velocity,const State& moments)
  {
    const int n=recovery.n,ne=int(moments.size())/(4*n);pressure.assign(ne*n,0.);
    if(2*recovery.nq-1<8*recovery.p+5)throw std::runtime_error("common-state Euler quadrature is insufficient");
    for(std::size_t i=2;i<moments.size();i+=4)
      if(moments[i]!=0.)throw std::runtime_error("minimum-gradient common transport requires exactly zero transverse momentum");
    for(int e=0;e<ne;++e)
    {
      State load(n,0.);
      for(int k=0;k<recovery.nq;++k)
      {
        double rho=0.,v=0.;for(int i=0;i<n;++i)rho+=recovery.nodal[k][i]*moments[(e*n+i)*4];
        for(int i=0;i<recovery.r;++i)v+=recovery.values[k][i]*velocity.coefficients[e*recovery.r+i];
        for(int i=0;i<n;++i)load[i]+=.25*recovery.weights[k]*recovery.nodal[k][i]*rho*v*v;
      }
      for(int i=0;i<n;++i)
      {
        double kinetic=0.;for(int j=0;j<n;++j)kinetic+=retained.mass_inverse(i,j)*load[j];
        pressure[e*n+i]=(gamma_euler-1.)*(moments[(e*n+i)*4+3]-kinetic);
      }
    }
    Certify(recovery,velocity,moments);
  }
  void Certify(const MinimumGradientRecovery& recovery,const MinimumGradientResult& velocity,const State& moments)
  {
    const int n=recovery.n,ne=int(moments.size())/(4*n);moment_defect=0.;
    for(int e=0;e<ne;++e)
    {
      // Independent normalized Legendre moments of the actual common state.
      std::vector<std::array<double,4>> physical;
      for(int k=0;k<recovery.nq;++k)physical.push_back(Point(recovery,velocity,moments,e,recovery.nodal[k],recovery.values[k]));
      for(int i=0;i<n;++i)for(int c=0;c<4;++c)
      {
        double defect=0.;for(int j=0;j<n;++j)defect-=recovery.modal(i,j)*moments[(e*n+j)*4+c];
        for(int k=0;k<recovery.nq;++k)
          defect+=.5*(2*i+1)*recovery.weights[k]*recovery.tests[k][i]*physical[k][c];
        moment_defect=std::max(moment_defect,std::abs(defect));
      }
    }
    if(moment_defect>1.e-8)throw std::runtime_error("minimum-gradient common conservative moment certificate failed");
  }
  std::array<double,4> Point(const MinimumGradientRecovery& recovery,const MinimumGradientResult& velocity,
                            const State& moments,int e,const State& nodal,const State& legendre)const
  {
    double rho=0.,v=0.,pressure_value=0.;
    for(int i=0;i<recovery.n;++i){rho+=nodal[i]*moments[(e*recovery.n+i)*4];if(pressure_legendre.empty())pressure_value+=nodal[i]*pressure[e*recovery.n+i];}
    if(!pressure_legendre.empty())for(int i=0;i<recovery.r;++i)pressure_value+=legendre[i]*pressure_legendre[e*recovery.r+i];
    for(int i=0;i<recovery.r;++i)v+=legendre[i]*velocity.coefficients[e*recovery.r+i];
    return {{rho,rho*v,0.,pressure_value/(gamma_euler-1.)+.5*rho*v*v}};
  }
};
}

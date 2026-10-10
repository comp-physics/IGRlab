#include "igrlab/compat.hpp"
// Reduced advancing full-energy Euler/instantaneous-IGR information study.
// Independent conservative FV means; current compact or ordinary FV closure
// data; shared dissipative transport fluxes. No closure lag or template fit.
#include "mfem.hpp"
#include "numerics/dgsem/dgsem_basis.hpp"
#include "physics/euler/eos.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "../../examples/shock_vortex_interaction/shock_vortex_interaction.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Cons = std::array<double,4>;
using State = std::vector<Cons>;
using Clock = std::chrono::steady_clock;
namespace igr = ::igrlab::frozen_scalar_igr_algebra;
constexpr double gamma_value = 1.4, admission_floor = 1.e-10;
double Wall(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
struct Options
{
  int elements=16, subcells=4, order=3, quadrature=8;
  double length=8., alpha=.04, dt=.001, final_time=.1, cfl=.3;
  double mach=2., speed=.4, front=0., width=.3, penalty=4.;
  std::string initial="shock", recovery="compact", closure="h1", transport="muscl";
  std::string output="/tmp/igr-subcell-evolution.json";
  bool limit_recovery=false, self_test=false;
};
struct Work
{
  std::uint64_t rhs_calls=0, closure_calls=0, source_points=0, coefficient_products=0;
  std::uint64_t local_factors=0, trace_factors=0, useful_local_rhs_columns=0;
  std::uint64_t face_fluxes=0, reconstruction_cells=0, recovery_limited_cells=0;
  std::uint64_t transport_limited_cells=0, first_order_cells=0, positivity_points=0;
  std::uint64_t accepted_steps=0, attempted_steps=0, rejected_steps=0;
  double local_factor_flops=0., trace_factor_flops=0., local_solve_products=0.;
  double schur_products=0., recovery_seconds=0., assembly_seconds=0., solve_seconds=0.;
  double min_theta=1., maximum_closure_residual=0., max_cfl=0.;
  std::size_t maximum_dense_operator_entries=0, maximum_trace_entries=0;
};
struct Poly
{
  std::array<Cons,3> c{};
  Cons Value(double t) const
  { Cons u{}; for(int k=0;k<4;++k)u[k]=c[0][k]+t*(c[1][k]+t*c[2][k]);return u; }
  Cons Derivative(double t,double dx) const
  { Cons u{};for(int k=0;k<4;++k)u[k]=(c[1][k]+2*t*c[2][k])/dx;return u; }
  void Scale(const Cons& mean,double theta)
  {for(int k=0;k<4;++k){c[0][k]=mean[k]+theta*(c[0][k]-mean[k]);c[1][k]*=theta;c[2][k]*=theta;}}
};
double Pressure(const Cons& u)
{mfem::Vector v(const_cast<double*>(u.data()),4);return mfem::EulerPressureFromConservative(v,2,gamma_value);}
bool Admissible(const Cons& u)
{for(double x:u)if(!std::isfinite(x))return false;return u[0]>admission_floor && Pressure(u)>admission_floor;}
void Require(const Cons& u)
{if(!Admissible(u))throw std::runtime_error("conservative mean or reconstructed state admission failed");}
bool BernsteinAdmissible(const Poly& p)
{
  // Sufficient whole-interval admission: a polynomial lies in the convex hull
  // of its Bernstein coefficients. Check rho-rho_floor (degree2) and
  // rho*(E-p_floor/(gamma-1))-(mx²+my²)/2 (degree4), not just nodal pressure.
  const double r0=p.c[0][0],r1=p.c[1][0],r2=p.c[2][0];
  if(!(std::min({r0,r0+.5*r1,r0+r1+r2})>admission_floor))return false;
  double a[5]={};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    a[i+j]+=(p.c[i][3]-(i==0?admission_floor/(gamma_value-1.):0.))*p.c[j][0]
             -.5*(p.c[i][1]*p.c[j][1]+p.c[i][2]*p.c[j][2]);
  const double b[5]={a[0],a[0]+a[1]/4.,a[0]+a[1]/2.+a[2]/6.,
    a[0]+.75*a[1]+.5*a[2]+.25*a[3],a[0]+a[1]+a[2]+a[3]+a[4]};
  for(double x:b)if(!(x>0.)||!std::isfinite(x))return false;
  return true;
}
double Norm(const mfem::Vector& v)
{long double x=0.;for(int i=0;i<v.Size();++i)x+=static_cast<long double>(v[i])*v[i];return std::sqrt(static_cast<double>(x));}
double Minmod(double a,double b,double c)
{if(a*b<=0 || b*c<=0)return 0.;return std::copysign(std::min({std::abs(a),std::abs(b),std::abs(c)}),a);}

class Problem
{
public:
  explicit Problem(const Options& input):o(input),N(o.elements*o.subcells),n(o.order+1),
    H(o.length/o.elements),dx(o.length/N),periodic(o.initial=="smooth"),
    basis(mfem::dgsem::BuildLGLBasis1D(o.order)),euler(2,gamma_value)
  {
    init=mfem::GetShockVortexInoueHattoriInitCustom(gamma_value,N,1,o.mach,0.,
      o.width/dx,-o.length/2,o.length/2,-o.length/2,o.length/2,0.,0.,o.front);
    left=PointInitial(-100*o.length);right=PointInitial(100*o.length);
    ir=&mfem::IntRules.Get(mfem::Geometry::SEGMENT,2*o.quadrature-1);
    for(int s=0;s<o.subcells;++s)for(int q=0;q<ir->GetNPoints();++q)
    {auto v=Shape((s+ir->IntPoint(q).x)/o.subcells);qshape.push_back(v.first);qdshape.push_back(v.second);}
    for(int s=0;s<=o.subcells;++s)face_shape.push_back(Shape(double(s)/o.subcells).first);
  }
  Cons PointInitial(double x) const
  {
    if(periodic)
    {const double z=2*std::acos(-1.)*x/o.length;const double rho=1.+.12*std::cos(z),
      vx=.4+.14*std::sin(z),vy=.025*std::cos(z+.3),p=1.+.08*std::sin(z+.2);
      return {{rho,rho*vx,rho*vy,p/(gamma_value-1)+.5*rho*(vx*vx+vy*vy)}};}
    mfem::Vector at(2),u(4);at[0]=x;at[1]=0.;init(at,u);
    u[3]+=o.speed*u[1]+.5*o.speed*o.speed*u[0];u[1]+=o.speed*u[0];
    return {{u[0],u[1],u[2],u[3]}};
  }
  State Initial() const
  {
    State u(N);const auto& rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,31);
    for(int i=0;i<N;++i)for(int k=0;k<rule.GetNPoints();++k)
    {const auto& p=rule.IntPoint(k);const Cons value=PointInitial(-o.length/2+dx*(i+p.x));
      for(int c=0;c<4;++c)u[i][c]+=p.weight*value[c];}
    return u;
  }
  const Cons& Neighbor(const State& u,int i) const
  {if(periodic)return u[(i%N+N)%N];return u[std::max(0,std::min(N-1,i))];}
  std::pair<std::vector<double>,std::vector<double>> Shape(double x) const
  {
    std::vector<double> v(n,1.),d(n,0.);
    for(int i=0;i<n;++i)
    {for(int j=0;j<n;++j)if(j!=i)v[i]*=(x-basis.nodes[j])/(basis.nodes[i]-basis.nodes[j]);
      for(int k=0;k<n;++k)if(k!=i)
      {double term=1./(basis.nodes[i]-basis.nodes[k]);for(int j=0;j<n;++j)if(j!=i && j!=k)
        term*=(x-basis.nodes[j])/(basis.nodes[i]-basis.nodes[j]);d[i]+=term;}}
    return {v,d};
  }
  double Limit(Poly& p,const Cons& mean,Work& w,bool enabled) const
  {
    // Bernstein bounds cover the whole interval; samples are independent
    // diagnostics. A failed sufficient bound can limit an actually positive
    // polynomial, which is part of the charged numerical-method choice.
    std::vector<double> points;for(int i=0;i<=24;++i)points.push_back(i/24.);
    for(int k=0;k<ir->GetNPoints();++k)points.push_back(ir->IntPoint(k).x);
    bool okay=BernsteinAdmissible(p);for(double x:points){++w.positivity_points;okay=Admissible(p.Value(x))&&okay;}
    if(okay)return 1.;if(!enabled)throw std::runtime_error("unlimited compact reconstruction is inadmissible");
    Require(mean);double lo=0.,hi=1.;
    for(int j=0;j<48;++j)
    {double t=.5*(lo+hi);Poly trial=p;trial.Scale(mean,t);okay=BernsteinAdmissible(trial);
      for(double x:points){++w.positivity_points;okay=Admissible(trial.Value(x))&&okay;}
      if(okay)lo=t;else hi=t;}
    p.Scale(mean,.999999999*lo);w.min_theta=std::min(w.min_theta,lo);return lo;
  }
  std::vector<Poly> Recover(const State& u,Work& w) const
  {
    auto start=Clock::now();std::vector<Poly> poly(N);w.reconstruction_cells+=N;
    for(int i=0;i<N;++i)
    {Require(u[i]);if(o.recovery=="fv-center"){poly[i].c[0]=u[i];continue;}
      const Cons& mm=Neighbor(u,i-2);const Cons& m=Neighbor(u,i-1);
      const Cons& p=Neighbor(u,i+1);const Cons& pp=Neighbor(u,i+2);
      for(int c=0;c<4;++c)
      {const double a=(-mm[c]+7*m[c]+7*u[i][c]-p[c])/12.;
       const double b=(-m[c]+7*u[i][c]+7*p[c]-pp[c])/12.;
       poly[i].c[0][c]=a;poly[i].c[1][c]=6*u[i][c]-4*a-2*b;
       poly[i].c[2][c]=-6*u[i][c]+3*a+3*b;}
      if(Limit(poly[i],u[i],w,o.limit_recovery)<1.)++w.recovery_limited_cells;
    }
    w.recovery_seconds+=Wall(start);return poly;
  }
  int NS() const {return o.closure=="sipg"?o.elements*n:o.elements*o.order+(periodic?0:1);}
  int Index(int e,int i) const
  {return o.closure=="sipg"?e*n+i:(e*o.order+i)%NS();}
  struct Closure
  {mfem::DenseMatrix a;mfem::Vector b,sigma;std::vector<Poly> recovered;double residual=0.;};
  Closure Assemble(const State& u,Work& w) const
  {
    Closure c;c.recovered=Recover(u,w);auto start=Clock::now();
    c.a.SetSize(NS());c.a=0.;c.b.SetSize(NS());c.b=0.;
    ++w.closure_calls;w.maximum_dense_operator_entries=std::max(w.maximum_dense_operator_entries,std::size_t(NS())*NS());
    for(int e=0;e<o.elements;++e)for(int s=0;s<o.subcells;++s)
    {const int cell=e*o.subcells+s;for(int k=0;k<ir->GetNPoints();++k)
      {const auto& qp=ir->IntPoint(k);const auto& phi=qshape[s*ir->GetNPoints()+k];
       const auto& dphi=qdshape[s*ir->GetNPoints()+k];const Cons point=c.recovered[cell].Value(qp.x);
       Require(point);double source_value=0.;
       if(o.recovery=="compact")
       {const Cons derivative=c.recovered[cell].Derivative(qp.x,dx);double gradient[8]={};
        for(int j=0;j<4;++j)gradient[2*j]=derivative[j];
        source_value=igr::SourceFromConservativeStateGradient(2,point.data(),1,gradient,2,1,o.alpha,admission_floor);}
       else
       {const auto& l=Neighbor(u,cell-1);const auto& r=Neighbor(u,cell+1);double gradient[4]={};
        gradient[0]=(r[1]/r[0]-l[1]/l[0])/(2*dx);gradient[2]=(r[2]/r[0]-l[2]/l[0])/(2*dx);
        source_value=igr::SourceFromVelocityGradient(2,gradient,2,1,o.alpha);}
       const double reaction=igr::SafeInverseDensity(point[0],admission_floor),diffusion=igr::AlphaOverDensity(o.alpha,point[0],admission_floor);
       if(!std::isfinite(source_value+reaction+diffusion))throw std::runtime_error("nonfinite current closure coefficient");
       ++w.source_points;w.coefficient_products+=n*n;
       for(int i=0;i<n;++i){c.b[Index(e,i)]+=dx*qp.weight*source_value*phi[i];
        for(int j=0;j<n;++j)c.a(Index(e,i),Index(e,j))+=dx*qp.weight*(reaction*phi[i]*phi[j]+diffusion*dphi[i]*dphi[j]/(H*H));}
      }}
    if(o.closure=="sipg")
    {
      const auto dl=Shape(1.).second,dr=Shape(0.).second;
      for(int face=periodic?0:1;face<o.elements;++face)
      {const int le=(face+o.elements-1)%o.elements,re=face;
       const double kl=o.alpha/c.recovered[(le+1)*o.subcells-1].Value(1.)[0];
       const double kr=o.alpha/c.recovered[re*o.subcells].Value(0.)[0];
       const double eta=o.penalty*n*n*std::max(kl,kr)/H;
       std::vector<int> ids;std::vector<double> jump,flux;
       for(int i=0;i<n;++i){ids.push_back(Index(le,i));jump.push_back(i==o.order?1.:0.);flux.push_back(.5*kl*dl[i]/H);}
       for(int i=0;i<n;++i){ids.push_back(Index(re,i));jump.push_back(i==0?-1.:0.);flux.push_back(.5*kr*dr[i]/H);}
       for(int i=0;i<2*n;++i)for(int j=0;j<2*n;++j)
        c.a(ids[i],ids[j])+=-flux[i]*jump[j]-jump[i]*flux[j]+eta*jump[i]*jump[j];
      }
    }
    w.assembly_seconds+=Wall(start);return c;
  }
  void Solve(Closure& c,Work& w,bool global_oracle=false) const
  {
    auto start=Clock::now();c.sigma.SetSize(NS());c.sigma=0.;
    if(global_oracle){mfem::DenseMatrixInverse inverse(c.a,true);inverse.Mult(c.b,c.sigma);}
    else
    {
      std::vector<int> trace,trace_position(NS(),-1);
      for(int e=0;e<o.elements;++e)for(int j:{0,o.order})
      {const int id=Index(e,j);if(trace_position[id]<0){trace_position[id]=trace.size();trace.push_back(id);}}
      const int nt=trace.size();mfem::DenseMatrix schur(nt);mfem::Vector rhs(nt),answer(nt);
      for(int i=0;i<nt;++i){rhs[i]=c.b[trace[i]];for(int j=0;j<nt;++j)schur(i,j)=c.a(trace[i],trace[j]);}
      w.maximum_trace_entries=std::max(w.maximum_trace_entries,std::size_t(nt)*nt);
      struct Elimination {std::vector<int> ids,active;mfem::DenseMatrix solved;};
      std::vector<Elimination> factors;factors.reserve(o.elements);
      for(int e=0;e<o.elements;++e)
      {
        const int ni=o.order-1;if(!ni)continue;factors.emplace_back();auto& f=factors.back();
        for(int i=1;i<o.order;++i)f.ids.push_back(Index(e,i));
        for(int j=0;j<nt;++j){bool nonzero=false;for(int id:f.ids)nonzero=nonzero||c.a(id,trace[j])!=0.;if(nonzero)f.active.push_back(j);}
        const int nr=f.active.size();mfem::DenseMatrix d(ni),b(ni,nr+1);
        for(int i=0;i<ni;++i){for(int j=0;j<ni;++j)d(i,j)=c.a(f.ids[i],f.ids[j]);
          for(int j=0;j<nr;++j)b(i,j)=c.a(f.ids[i],trace[f.active[j]]);b(i,nr)=c.b[f.ids[i]];}
        mfem::DenseMatrixInverse inverse(d,true);inverse.Mult(b,f.solved);
        ++w.local_factors;w.useful_local_rhs_columns+=nr+1;w.local_factor_flops+=double(ni)*ni*ni/3.;
        w.local_solve_products+=double(ni)*ni*(nr+1);
        for(int i=0;i<nr;++i)
        {const int ti=f.active[i];for(int k=0;k<ni;++k)rhs[ti]-=c.a(trace[ti],f.ids[k])*f.solved(k,nr);
         for(int j=0;j<nr;++j)for(int k=0;k<ni;++k)
          schur(ti,f.active[j])-=c.a(trace[ti],f.ids[k])*f.solved(k,j);}
        w.schur_products+=double(nr)*(nr+1)*ni;
      }
      mfem::DenseMatrixInverse inverse(schur,true);inverse.Mult(rhs,answer);
      ++w.trace_factors;w.trace_factor_flops+=double(nt)*nt*nt/3.;
      for(int i=0;i<nt;++i)c.sigma[trace[i]]=answer[i];
      for(const auto& f:factors)for(int i=0;i<int(f.ids.size());++i)
      {double x=f.solved(i,f.active.size());for(int j=0;j<int(f.active.size());++j)x-=f.solved(i,j)*answer[f.active[j]];c.sigma[f.ids[i]]=x;}
    }
    mfem::Vector action(NS());c.a.Mult(c.sigma,action);action-=c.b;
    c.residual=Norm(action)/std::max(Norm(c.b),1.e-30);w.maximum_closure_residual=std::max(w.maximum_closure_residual,c.residual);
    if(!std::isfinite(c.residual)||c.residual>1.e-9)throw std::runtime_error("current original-operator closure residual failed");
    w.solve_seconds+=Wall(start);
  }
  double SigmaFace(const Closure& c,int cell,bool right_face) const
  {const int e=cell/o.subcells,s=cell%o.subcells+(right_face?1:0);double x=0.;
   for(int i=0;i<n;++i)x+=face_shape[s][i]*c.sigma[Index(e,i)];return x;}
  Cons Flux(const Cons& u,double sigma,double& speed) const
  {
    Require(u);mfem::Vector state(const_cast<double*>(u.data()),4),normal(2),f(4);normal=0.;normal[0]=1.;
    mfem::FaceElementTransformations unused;euler.ComputeFluxDotN(state,normal,unused,f);
    const double p=Pressure(u);if(!igr::FrozenSigmaFluxIsLocallyStrictlyHyperbolic(p,sigma,gamma_value))
      throw std::runtime_error("current sigma flux hyperbolicity admission failed");
    igr::AddNormalSigmaFlux(2,u.data(),1,normal.GetData(),1,sigma,admission_floor,f.GetData(),1);
    speed=igr::FrozenSigmaCharacteristicModulusBound(u[0],p,u[1]/u[0],sigma,gamma_value,admission_floor);
    if(!std::isfinite(speed))throw std::runtime_error("nonfinite face speed");
    return {{f[0],f[1],f[2],f[3]}};
  }
  struct RHS {State value;Cons boundary{};double speed=0.;};
  RHS Rhs(const State& u,Work& w) const
  {
    ++w.rhs_calls;Closure closure=Assemble(u,w);Solve(closure,w);
    std::vector<Cons> left_state(N),right_state(N);
    for(int i=0;i<N;++i)
    {
      Cons slope{};if(o.transport=="muscl")for(int k=0;k<4;++k)
      {double l=u[i][k]-Neighbor(u,i-1)[k],r=Neighbor(u,i+1)[k]-u[i][k];slope[k]=Minmod(2*l,.5*(l+r),2*r);}
      else ++w.first_order_cells;
      Poly linear;linear.c[0]=u[i];for(int k=0;k<4;++k){linear.c[0][k]-=.5*slope[k];linear.c[1][k]=slope[k];}
      const double theta=Limit(linear,u[i],w,true);if(theta<1.)++w.transport_limited_cells;
      if(theta<1.e-8)++w.first_order_cells;left_state[i]=linear.Value(0.);right_state[i]=linear.Value(1.);
    }
    RHS out;out.value.resize(N);std::vector<Cons> flux(N+1);
    for(int face=0;face<(periodic?N:N+1);++face)
    {
      const int li=(face+N-1)%N,ri=face%N;
      const Cons& ul=face==0&&!periodic?left:right_state[li];
      const Cons& ur=face==N&&!periodic?right:left_state[ri];
      const double sl=face==0&&!periodic?SigmaFace(closure,0,false):SigmaFace(closure,li,true);
      const double sr=face==N&&!periodic?SigmaFace(closure,N-1,true):SigmaFace(closure,ri,false);
      double al=0.,ar=0.;const Cons fl=Flux(ul,sl,al),fr=Flux(ur,sr,ar);const double a=std::max(al,ar);
      out.speed=std::max(out.speed,a);++w.face_fluxes;
      for(int k=0;k<4;++k)flux[face][k]=.5*(fl[k]+fr[k]-a*(ur[k]-ul[k]));
    }
    if(periodic)flux[N]=flux[0];
    for(int k=0;k<4;++k)out.boundary[k]=flux[N][k]-flux[0][k];
    for(int i=0;i<N;++i)for(int k=0;k<4;++k)out.value[i][k]=(flux[i][k]-flux[i+1][k])/dx;
    return out;
  }
  Cons Integrals(const State& u) const
  {Cons x{};for(const auto& cell:u)for(int k=0;k<4;++k)x[k]+=dx*cell[k];return x;}
  const Options o;const int N,n;const double H,dx;const bool periodic;
  const mfem::dgsem::LGLBasis1D basis;const mfem::EulerFlux euler;
  const mfem::IntegrationRule* ir=nullptr;Cons left{},right{};
  std::function<void(const mfem::Vector&,mfem::Vector&)> init;
  std::vector<std::vector<double>> qshape,qdshape,face_shape;
};

struct Result
{State state;Cons initial_integral{},boundary{},defect{};double time=0.,seconds=0.;
 bool success=true;std::string reason="target-reached";};
State Stage(const State& base,const State& current,const State& rhs,double a,double dt)
{State next(base.size());for(int i=0;i<int(base.size());++i){for(int k=0;k<4;++k)
 next[i][k]=a*base[i][k]+(1-a)*(current[i][k]+dt*rhs[i][k]);Require(next[i]);}return next;}
Result Advance(const Problem& p,Work& w)
{
  Result r;r.state=p.Initial();r.initial_integral=p.Integrals(r.state);auto start=Clock::now();
  while(r.time<p.o.final_time-4*std::numeric_limits<double>::epsilon()*std::max(1.,p.o.final_time))
  {
    Problem::RHS k1;
    try{k1=p.Rhs(r.state,w);}catch(const std::exception& e){r.success=false;r.reason=e.what();break;}
    double dt=std::min({p.o.dt,p.o.final_time-r.time,p.o.cfl*p.dx/std::max(k1.speed,1.e-12)});
    bool accepted=false;
    // The first attempt may be a roundoff-sized remainder at the horizon.
    // The lower bound guards failed-step retries, not a valid final step.
    for(int trial=0;trial<20 && dt>0. && (trial==0 || dt>1.e-10);++trial)
    {
      ++w.attempted_steps;
      try
      {
        State one=Stage(r.state,r.state,k1.value,0.,dt);auto k2=p.Rhs(one,w);
        State two=Stage(r.state,one,k2.value,.75,dt);auto k3=p.Rhs(two,w);
        State next=Stage(r.state,two,k3.value,1./3.,dt);
        // The RK endpoint is also an admissible representation, even though
        // SSPRK3 does not evaluate its derivative until the following step.
        // This extra reconstruction is charged and makes a failed endpoint
        // retryable instead of silently committing an unusable source state.
        (void)p.Recover(next,w);
        for(int k=0;k<4;++k)r.boundary[k]+=dt*(k1.boundary[k]/6.+k2.boundary[k]/6.+2*k3.boundary[k]/3.);
        r.state=std::move(next);r.time+=dt;++w.accepted_steps;accepted=true;
        w.max_cfl=std::max(w.max_cfl,dt*std::max({k1.speed,k2.speed,k3.speed})/p.dx);break;
      }
      catch(const std::exception& e){++w.rejected_steps;r.reason=e.what();dt*=.5;}
    }
    if(!accepted){r.success=false;r.reason="step retry limit: "+r.reason;break;}
  }
  r.seconds=Wall(start);const Cons final=p.Integrals(r.state);
  for(int k=0;k<4;++k)r.defect[k]=final[k]+r.boundary[k]-r.initial_integral[k];
  if(r.success)r.reason="target-reached";return r;
}
void JsonString(std::ostream& out,const std::string& s)
{out<<'"';for(char c:s){if(c=='"'||c=='\\')out<<'\\';out<<c;}out<<'"';}
template<class Values> void JsonValues(std::ostream& out,const Values& a)
{out<<'[';bool first=true;for(double x:a){if(!first)out<<',';first=false;if(std::isfinite(x))out<<x;else out<<"null";}out<<']';}
void Write(const Options& o,const Problem& p,Result& r,Work& w,Clock::time_point start)
{
  auto final=p.Assemble(r.state,w);p.Solve(final,w);double min_rho=INFINITY,min_pressure=INFINITY;
  for(const auto& poly:final.recovered)for(int i=0;i<=64;++i)
  {Cons u=poly.Value(i/64.);min_rho=std::min(min_rho,u[0]);min_pressure=std::min(min_pressure,Pressure(u));}
  if(!(min_rho>admission_floor && min_pressure>admission_floor))
  {r.success=false;r.reason="final dense positivity audit failed";}
  std::ofstream out(o.output);if(!out)throw std::runtime_error("cannot create output JSON");out<<std::setprecision(17);
  out<<"{\n\"schema\":\"igr-subcell-evolution-v1\",\n\"configuration\":{";
  out<<"\"elements\":"<<o.elements<<",\"subcells\":"<<o.subcells<<",\"closure_order\":"<<o.order
     <<",\"quadrature\":"<<o.quadrature<<",\"length\":"<<o.length<<",\"alpha\":"<<o.alpha
     <<",\"dt\":"<<o.dt<<",\"final_time\":"<<o.final_time<<",\"cfl\":"<<o.cfl
     <<",\"mach\":"<<o.mach<<",\"speed\":"<<o.speed<<",\"front\":"<<o.front<<",\"width\":"<<o.width
     <<",\"penalty\":"<<o.penalty<<",\"limit_recovery\":"<<(o.limit_recovery?"true":"false");
  for(const auto& x:std::vector<std::pair<std::string,std::string>>{{"initial",o.initial},{"recovery",o.recovery},{"closure",o.closure},{"transport",o.transport}})
  {out<<',';JsonString(out,x.first);out<<':';JsonString(out,x.second);}out<<"},\n";
  out<<"\"success\":"<<(r.success?"true":"false")<<",\"reason\":";JsonString(out,r.reason);
  out<<",\"reached_time\":"<<r.time<<",\"advance_seconds\":"<<r.seconds<<",\"before_output_seconds\":"<<Wall(start)
     <<",\"final_current_operator_residual\":"<<final.residual<<",\"minimum_density_sampled\":"<<min_rho
     <<",\"minimum_pressure_sampled\":"<<min_pressure<<",\"mean_scalar_dofs\":"<<4*p.N<<",\"closure_dofs\":"<<p.NS();
  out<<",\"initial_integral\":";JsonValues(out,r.initial_integral);out<<",\"boundary_flux_budget\":";JsonValues(out,r.boundary);
  out<<",\"conservation_defect\":";JsonValues(out,r.defect);out<<",\n\"work\":{";
  bool first=true;auto number=[&](const char* key,double value){if(!first)out<<',';first=false;JsonString(out,key);out<<':'<<value;};
#define WORK_VALUE(name) number(#name,static_cast<double>(w.name))
  WORK_VALUE(rhs_calls);WORK_VALUE(closure_calls);WORK_VALUE(source_points);WORK_VALUE(coefficient_products);
  WORK_VALUE(local_factors);WORK_VALUE(trace_factors);WORK_VALUE(useful_local_rhs_columns);WORK_VALUE(face_fluxes);
  WORK_VALUE(reconstruction_cells);WORK_VALUE(recovery_limited_cells);WORK_VALUE(transport_limited_cells);
  WORK_VALUE(first_order_cells);WORK_VALUE(positivity_points);WORK_VALUE(accepted_steps);WORK_VALUE(attempted_steps);
  WORK_VALUE(rejected_steps);WORK_VALUE(local_factor_flops);WORK_VALUE(trace_factor_flops);WORK_VALUE(local_solve_products);
  WORK_VALUE(schur_products);WORK_VALUE(recovery_seconds);WORK_VALUE(assembly_seconds);WORK_VALUE(solve_seconds);
  WORK_VALUE(min_theta);WORK_VALUE(maximum_closure_residual);WORK_VALUE(max_cfl);
  WORK_VALUE(maximum_dense_operator_entries);WORK_VALUE(maximum_trace_entries);
#undef WORK_VALUE
  out<<"},\n\"means\":[";for(int i=0;i<p.N;++i){if(i)out<<',';JsonValues(out,r.state[i]);}out<<"],\n\"reconstructed_coefficients\":[";
  for(int i=0;i<p.N;++i){if(i)out<<',';out<<'[';for(int k=0;k<3;++k){if(k)out<<',';JsonValues(out,final.recovered[i].c[k]);}out<<']';}
  out<<"],\n\"sigma\":[";for(int i=0;i<final.sigma.Size();++i){if(i)out<<',';out<<final.sigma[i];}out<<"],\n\"closure_nodes\":[";
  for(int i=0;i<p.n;++i){if(i)out<<',';out<<p.basis.nodes[i];}out<<"]\n}\n";
  std::cout<<"subcell_evolution success="<<r.success<<" time="<<r.time<<" steps="<<w.accepted_steps
           <<" rejected="<<w.rejected_steps<<" advance_s="<<r.seconds<<" residual="<<final.residual<<" output="<<o.output<<'\n';
}
void SelfTest()
{
  for(const std::string kind:{"h1","sipg"})for(int order:{1,3,5})
  {Options o;o.elements=3;o.subcells=3;o.order=order;o.initial="smooth";o.closure=kind;o.final_time=.001;
   Problem p(o);Work w;State u=p.Initial();auto a=p.Assemble(u,w),b=p.Assemble(u,w);
   p.Solve(a,w);p.Solve(b,w,true);mfem::Vector error(a.sigma);error-=b.sigma;
   if(Norm(error)>1.e-10*std::max(Norm(b.sigma),1.e-20))throw std::runtime_error("condensation differs from independent global factorization");
   for(int i=0;i<p.N;++i)for(int k=0;k<4;++k)
   {const auto& c=a.recovered[i].c;const double mean=c[0][k]+c[1][k]/2+c[2][k]/3;
    if(std::abs(mean-u[i][k])>1.e-13*std::max(1.,std::abs(u[i][k])))throw std::runtime_error("compact reconstruction lost a component mean");}
   auto rhs=p.Rhs(u,w);Cons integral=p.Integrals(rhs.value);
   for(double x:integral)if(std::abs(x)>1.e-12)throw std::runtime_error("shared periodic flux is not conservative");
   auto r=Advance(p,w);if(!r.success)throw std::runtime_error("advancing self-test failed");
   for(double x:r.defect)if(std::abs(x)>2.e-12)throw std::runtime_error("SSPRK boundary budget is not conservative");
  }
  {Options o;o.elements=2;o.subcells=2;o.initial="smooth";o.final_time=.5;o.dt=.00025;
   Problem p(o);Work w;auto r=Advance(p,w);
   if(!r.success || std::abs(r.time-o.final_time)>4*std::numeric_limits<double>::epsilon())
     throw std::runtime_error("roundoff-sized horizon remainder was not completed");}
  std::cout<<"subcell_evolution_self_test pass\n";
}
Options Parse(int argc,char** argv)
{
  Options o;for(int i=1;i<argc;++i)
  {std::string a=argv[i];if(a=="--self-test"){o.self_test=true;continue;}if(a=="--limit-recovery"){o.limit_recovery=true;continue;}
   if(i+1==argc)throw std::runtime_error("missing option value");std::string v=argv[++i];
   if(a=="--elements")o.elements=std::stoi(v);else if(a=="--subcells")o.subcells=std::stoi(v);
   else if(a=="--order")o.order=std::stoi(v);else if(a=="--quadrature")o.quadrature=std::stoi(v);
   else if(a=="--length")o.length=std::stod(v);else if(a=="--alpha")o.alpha=std::stod(v);
   else if(a=="--dt")o.dt=std::stod(v);else if(a=="--final-time")o.final_time=std::stod(v);
   else if(a=="--cfl")o.cfl=std::stod(v);else if(a=="--mach")o.mach=std::stod(v);
   else if(a=="--speed")o.speed=std::stod(v);else if(a=="--front")o.front=std::stod(v);
   else if(a=="--width")o.width=std::stod(v);else if(a=="--penalty")o.penalty=std::stod(v);
   else if(a=="--case")o.initial=v;else if(a=="--recovery")o.recovery=v;
   else if(a=="--closure")o.closure=v;else if(a=="--transport")o.transport=v;
   else if(a=="--output")o.output=v;else throw std::runtime_error("unknown option: "+a);}
  if(o.elements<2||o.elements>128||o.subcells<1||o.subcells>32||o.order<1||o.order>7||
     o.quadrature<3||o.quadrature>24||!(o.alpha>0)||!(o.dt>0)||!(o.length>0)||!(o.width>0)||
     o.final_time<0||!(o.cfl>0&&o.cfl<=.5)||!(o.mach>1&&o.mach<=10)||
     (o.initial!="shock"&&o.initial!="smooth")||(o.recovery!="compact"&&o.recovery!="fv-center")||
     (o.closure!="h1"&&o.closure!="sipg")||(o.transport!="muscl"&&o.transport!="first-order"))
    throw std::runtime_error("outside bounded research controls");
  return o;
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");auto start=Clock::now();
  try{Options o=Parse(argc,argv);if(o.self_test){SelfTest();return 0;}
      Problem p(o);Work w;Result r=Advance(p,w);Write(o,p,r,w,start);return r.success?0:2;}
  catch(const std::exception& e){std::cerr<<"subcell_evolution_error "<<e.what()<<'\n';return 2;}
}

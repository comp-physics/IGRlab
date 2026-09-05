// Test-only screen-aware space-time Schwarz experiment. This is a reduced
// pseudo-1D full-energy Euler/instantaneous-IGR discretization, not a new app.
// It uses the production LGL basis, MFEM Euler flux and shared IGR point algebra.
// Spatial changes: 1D periodic DGSEM Euler, lumped nodal continuous H1 closure.
// Temporal change: two-node Gauss collocation solved by overlapping local
// space-time predictors. It is not classic element-volume-only ADER. Patch
// boundary waveforms are exchanged only between outer Schwarz sweeps. The
// original CURRENT-density reaction AND diffusion equations are certified.
// Frozen patch factors only precondition multi-RHS defect correction.
#include "numerics/dgsem/dgsem_basis.hpp"
#include "physics/euler/eos.hpp"
#include "physics/igr/frozen_scalar_igr_algebra.hpp"
#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using State = std::vector<double>;
using Stages = std::array<State, 2>;
using Clock = std::chrono::steady_clock;
namespace igr = dg_igr::frozen_scalar_igr_algebra;
constexpr double gamma_euler = 1.4;
constexpr double floor_value = 1.e-12;
const double q[2][2] = {{.25, .25-std::sqrt(3.)/6.},
                        {.25+std::sqrt(3.)/6., .25}};

struct Work
{
  std::uint64_t global_factors=0, patch_factors=0, patch_rhs_solves=0;
  std::uint64_t patch_block_solves=0, coefficient_elements=0;
  std::uint64_t residual_elements=0, face_fluxes=0, outer_exchanges=0;
  std::uint64_t certificate_calls=0, accepted_steps=0;
  std::uint64_t startup_rk4_steps=0;
  std::uint64_t joint_factors=0, joint_jacobian_columns=0;
  std::uint64_t joint_residual_calls=0, joint_iterations=0, joint_line_searches=0;
  std::uint64_t attempted_steps=0, rejected_steps=0;
  double factor_flops=0, solve_flops=0, setup_seconds=0;
  std::size_t retained_factor_bytes=0;
};
struct Options
{
  int elements=12, order=3, halo=1, inner=4, outer_max=80, fixed_outer=0;
  double alpha=.0025, dt=.002, final_time=.02, reference_dt=.0000625;
  double closure_tolerance=1.e-10, state_tolerance=1.e-11;
  std::string initial_case="smooth", method="both";
  std::string sigma_predictor="constant", state_predictor="constant";
  int joint_block_elements=0, joint_block_offset=0, joint_rebuild=0, joint_max=40;
  double alpha_power=0.;
  bool alpha_from_h=false, self_test=false;
};
double Norm(const State& x)
{ long double s=0; for(double v:x){if(!std::isfinite(v))return INFINITY;
    s+=static_cast<long double>(v)*v;} return static_cast<double>(std::sqrt(s)); }
double Wall(Clock::time_point start)
{ return std::chrono::duration<double>(Clock::now()-start).count(); }

class Problem
{
public:
  explicit Problem(const Options& options): o(options), ne(o.elements),
    p(o.order), n(p+1), h(1./ne), basis(mfem::dgsem::BuildLGLBasis1D(p)),
    euler(2,gamma_euler) {}
  int Mod(int e) const {return (e%ne+ne)%ne;}
  int Index(int e,int i,int c) const {return (Mod(e)*n+i)*4+c;}
  int SigmaIndex(int e,int i) const {return (Mod(e)*p+i)%(ne*p);}
  State Initial() const
  {
    State u(ne*n*4);
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
    {
      const double x=h*(e+basis.nodes[i]), a=2.*std::acos(-1.)*x;
      const double front=std::tanh(std::sin(a)/.12);
      const double rho=o.initial_case=="compression"?1.+.18*front:
        1.+.04*std::cos(a);
      const double vx=o.initial_case=="compression"?-.3*front:
        .2+.045*std::sin(a);
      const double vy=.02*std::cos(a);
      const double pressure=o.initial_case=="compression"?1.+.12*front:
        1.+.025*std::sin(a+.3);
      u[Index(e,i,0)]=rho;u[Index(e,i,1)]=rho*vx;
      u[Index(e,i,2)]=rho*vy;
      u[Index(e,i,3)]=pressure/(gamma_euler-1.)+.5*rho*(vx*vx+vy*vy);
    }
    return u;
  }
  void CheckState(const double* x) const
  {
    mfem::Vector v(const_cast<double*>(x),4);
    for(int c=0;c<4;++c)if(!std::isfinite(x[c]))
      throw std::runtime_error("nonfinite state");
    if(!(x[0]>floor_value) ||
       !(mfem::EulerPressureFromConservative(v,2,gamma_euler)>floor_value))
      throw std::runtime_error("density/pressure admission failed");
  }
  // Every call includes nonlinear coefficient reconstruction and BOTH 1/rho
  // reaction and alpha/rho diffusion. Local rows use continuous nodal numbering.
  void Assemble(const State& u,int count,bool periodic,
                mfem::DenseMatrix& a,State& b,Work& work) const
  {
    const int ns=periodic?count*p:count*p+1;
    a.SetSize(ns);a=0.;b.assign(ns,0.);work.coefficient_elements+=count;
    for(int e=0;e<count;++e)for(int k=0;k<n;++k)
    {
      const double* x=&u[(e*n+k)*4];CheckState(x);
      double du[8]={};
      for(int c=0;c<4;++c)for(int j=0;j<n;++j)
        du[c*2]+=basis.D(k,j)*u[(e*n+j)*4+c]/h;
      const double source=igr::SourceFromConservativeStateGradient(
          2,x,1,du,2,1,o.alpha,floor_value);
      const double invrho=igr::SafeInverseDensity(x[0],floor_value);
      const double ck=igr::AlphaOverDensity(o.alpha,x[0],floor_value);
      if(!std::isfinite(source+invrho+ck))
        throw std::runtime_error("nonfinite closure coefficient");
      const int row=(e*p+k)%ns;
      a(row,row)+=h*basis.weights[k]*invrho;
      b[row]+=h*basis.weights[k]*source;
      for(int i=0;i<n;++i)for(int j=0;j<n;++j)
        a((e*p+i)%ns,(e*p+j)%ns)+=
          ck*basis.weights[k]*basis.D(k,i)*basis.D(k,j)/h;
    }
  }
  State ExactClosure(const State& u,Work& work) const
  {
    mfem::DenseMatrix a;State b;Assemble(u,ne,true,a,b,work);
    const int ns=ne*p;State sigma(ns);
    mfem::DenseMatrixInverse inverse(a,true);
    mfem::Vector bv(b.data(),ns),sv(sigma.data(),ns);inverse.Mult(bv,sv);
    ++work.global_factors;work.factor_flops+=double(ns)*ns*ns/3.;
    work.solve_flops+=2.*ns*ns;
    return sigma;
  }
  std::array<double,4> Flux(const double* x,double sigma,double& speed) const
  {
    CheckState(x);mfem::Vector u(const_cast<double*>(x),4),normal(2),f(4);
    normal=0.;normal[0]=1.;mfem::FaceElementTransformations unused;
    const double base=euler.ComputeFluxDotN(u,normal,unused,f);
    const double pressure=mfem::EulerPressureFromConservative(u,2,gamma_euler);
    if(!igr::FrozenSigmaFluxIsLocallyStrictlyHyperbolic(
        pressure,sigma,gamma_euler))throw std::runtime_error("q_sigma admission failed");
    igr::AddNormalSigmaFlux(2,x,1,normal.GetData(),1,sigma,floor_value,
                            f.GetData(),1);
    speed=igr::FrozenSigmaCharacteristicModulusBound(x[0],pressure,
        x[1]/x[0],sigma,gamma_euler,floor_value);
    if(!std::isfinite(base)||!std::isfinite(speed))
      throw std::runtime_error("nonfinite flux speed");
    return {{f[0],f[1],f[2],f[3]}};
  }
  // A SINGLE Rusanov flux is constructed per face and shared by the two
  // residuals. For patch-edge faces the exterior waveform is the previous
  // Schwarz iterate; patch interfaces need not match before certification.
  State Residual(const State& u,const State& sigma,int count,
      bool periodic,int first,const State* exterior,
      const State* exterior_sigma,Work& work) const
  {
    State r(u.size(),0.);std::vector<std::array<double,4>> f(count*n);
    State speed(count*n);work.residual_elements+=count;
    const int ns=static_cast<int>(sigma.size());
    for(int e=0;e<count;++e)for(int i=0;i<n;++i)
      f[e*n+i]=Flux(&u[(e*n+i)*4],sigma[(e*p+i)%ns],speed[e*n+i]);
    for(int e=0;e<count;++e)for(int i=0;i<n;++i)
      for(int j=0;j<n;++j)for(int c=0;c<4;++c)
        r[(e*n+i)*4+c]-=basis.D(i,j)*f[e*n+j][c]/h;
    const int faces=periodic?count:count+1;
    for(int face=0;face<faces;++face)
    {
      ++work.face_fluxes;
      const int le=face-1,re=face;
      const double* left=nullptr;const double* right=nullptr;
      std::array<double,4> fl{},fr{};double al=0,ar=0;
      if(le>=0 || periodic)
      {const int l=(le+count)%count;left=&u[(l*n+p)*4];fl=f[l*n+p];al=speed[l*n+p];}
      else
      {const int e=Mod(first-1);left=&(*exterior)[Index(e,p,0)];
       fl=Flux(left,(*exterior_sigma)[SigmaIndex(e,p)],al);}
      if(re<count)
      {right=&u[(re*n)*4];fr=f[re*n];ar=speed[re*n];}
      else
      {const int e=Mod(first+count);right=&(*exterior)[Index(e,0,0)];
       fr=Flux(right,(*exterior_sigma)[SigmaIndex(e,0)],ar);}
      const double a=std::max(al,ar);
      for(int c=0;c<4;++c)
      {
        const double star=.5*(fl[c]+fr[c]-a*(right[c]-left[c]));
        if(le>=0 || periodic)
        {const int l=(le+count)%count;r[(l*n+p)*4+c]-=(star-fl[c])/(h*basis.weights[p]);}
        if(re<count)r[(re*n)*4+c]+=(star-fr[c])/(h*basis.weights[0]);
      }
    }
    return r;
  }
  State Rhs(const State& u,Work& w) const
  {return Residual(u,ExactClosure(u,w),ne,true,0,nullptr,nullptr,w);}
  std::array<double,4> Integrals(const State& u) const
  {
    std::array<double,4> sum{};
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)for(int c=0;c<4;++c)
      sum[c]+=h*basis.weights[i]*u[Index(e,i,c)];
    return sum;
  }
  double Error(const State& u,const State& v) const
  {
    double num=0,den=0;
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)for(int c=0;c<4;++c)
    {const int k=Index(e,i,c);const double w=h*basis.weights[i];
     num+=w*(u[k]-v[k])*(u[k]-v[k]);den+=w*v[k]*v[k];}
    return std::sqrt(num/den);
  }
  const Options o;const int ne,p,n;const double h;
  const mfem::dgsem::LGLBasis1D basis;const mfem::EulerFlux euler;
};

State RK4Step(const Problem& pr,const State& u,const State& current_sigma,
              double dt,Work& w)
{
  // The accepted endpoint was exactly reclosed. Reuse that same current-state
  // sigma for k1, just as a strong application driver can reuse its endpoint.
  const State a=pr.Residual(u,current_sigma,pr.ne,true,0,nullptr,nullptr,w);
  State v(u.size());
  for(std::size_t i=0;i<u.size();++i)v[i]=u[i]+.5*dt*a[i];
  const State b=pr.Rhs(v,w);
  for(std::size_t i=0;i<u.size();++i)v[i]=u[i]+.5*dt*b[i];
  const State c=pr.Rhs(v,w);
  for(std::size_t i=0;i<u.size();++i)v[i]=u[i]+dt*c[i];
  const State d=pr.Rhs(v,w);
  for(std::size_t i=0;i<u.size();++i)v[i]=u[i]+dt*(a[i]+2*b[i]+2*c[i]+d[i])/6.;
  return v;
}

struct Patch
{
  int first=0,count=0,center=0,ns=0;
  State initial;mfem::DenseMatrix frozen;
  std::unique_ptr<mfem::DenseMatrixInverse> inverse;
};

std::vector<Patch> BuildPatches(const Problem& pr,const State& u,Work& w)
{
  const auto start=Clock::now();std::vector<Patch> patches(pr.ne);
  for(int e=0;e<pr.ne;++e)
  {
    auto& patch=patches[e];patch.first=e-pr.o.halo;
    patch.count=2*pr.o.halo+1;patch.center=pr.o.halo;
    patch.ns=patch.count*pr.p+1;patch.initial.resize(patch.count*pr.n*4);
    for(int l=0;l<patch.count;++l)for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)
      patch.initial[(l*pr.n+i)*4+c]=u[pr.Index(patch.first+l,i,c)];
    mfem::DenseMatrix a;State b;pr.Assemble(patch.initial,patch.count,false,a,b,w);
    const int m=patch.ns-2;patch.frozen.SetSize(m);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)patch.frozen(i,j)=a(i+1,j+1);
    patch.inverse=std::make_unique<mfem::DenseMatrixInverse>(patch.frozen,true);
    ++w.patch_factors;w.factor_flops+=double(m)*m*m/3.;
  }
  const int m=patches.front().ns-2;
  w.retained_factor_bytes=std::max(w.retained_factor_bytes,
      std::size_t(pr.ne)*m*m*sizeof(double));
  w.setup_seconds+=Wall(start);return patches;
}

void PatchClosures(const Problem& pr,const Patch& patch,const Stages& u,
                   Stages& sigma,Work& w)
{
  const int m=patch.ns-2;std::array<mfem::DenseMatrix,2> a;
  Stages b;double bnorm[2];
  for(int j=0;j<2;++j)
  {
    pr.Assemble(u[j],patch.count,false,a[j],b[j],w);
    State effective_rhs(m);
    for(int i=1;i<patch.ns-1;++i)
      effective_rhs[i-1]=b[j][i]-a[j](i,0)*sigma[j][0]-
        a[j](i,patch.ns-1)*sigma[j][patch.ns-1];
    // Screening can drive a patch through nonzero Dirichlet data even when
    // its volume source vanishes. Scale the interior solve by its actual RHS.
    bnorm[j]=std::max(Norm(effective_rhs),1.e-20);
  }
  mfem::DenseMatrix residual(m,2),delta(m,2);
  for(int iteration=0;iteration<40;++iteration)
  {
    bool done=true;
    for(int stage=0;stage<2;++stage)
    {
      double norm=0;
      for(int i=1;i<patch.ns-1;++i)
      {
        double r=b[stage][i];
        for(int k=0;k<patch.ns;++k)r-=a[stage](i,k)*sigma[stage][k];
        residual(i-1,stage)=r;norm+=r*r;
      }
      if(std::sqrt(norm)/bnorm[stage]>.1*pr.o.closure_tolerance)done=false;
    }
    if(done)return;
    // Two genuine temporal RHS use ONE retained factor. No per-node factor.
    patch.inverse->Mult(residual,delta);++w.patch_block_solves;
    w.patch_rhs_solves+=2;w.solve_flops+=4.*m*m;
    for(int j=0;j<2;++j)for(int i=1;i<patch.ns-1;++i)
      sigma[j][i]+=delta(i-1,j);
  }
  throw std::runtime_error("current-density patch defect correction failed");
}

struct Certificate {double closure=INFINITY,state=INFINITY;};
Certificate Certify(const Problem& pr,const State& initial,double dt,
                     const Stages& u,const Stages& sigma,Work& w)
{
  Certificate c{0,0};Stages f;
  for(int stage=0;stage<2;++stage)
  {
    mfem::DenseMatrix a;State b;pr.Assemble(u[stage],pr.ne,true,a,b,w);
    State r(b.size());
    for(int i=0;i<a.Height();++i)
    {r[i]=-b[i];for(int k=0;k<a.Width();++k)r[i]+=a(i,k)*sigma[stage][k];}
    c.closure=std::max(c.closure,Norm(r)/std::max(Norm(b),1.e-20));
    f[stage]=pr.Residual(u[stage],sigma[stage],pr.ne,true,0,nullptr,nullptr,w);
  }
  for(int stage=0;stage<2;++stage)
  {
    State r(initial.size());
    for(std::size_t i=0;i<r.size();++i)
      r[i]=u[stage][i]-initial[i]-dt*(q[stage][0]*f[0][i]+q[stage][1]*f[1][i]);
    c.state=std::max(c.state,Norm(r)/std::max(Norm(initial),1.e-20));
  }
  ++w.certificate_calls;return c;
}

State PatchStep(const Problem& pr,const State& initial,const Stages& seed_u,
                 const Stages& seed_sigma,
                 double dt,Work& w,Certificate& worst,bool& certified)
{
  auto patches=BuildPatches(pr,initial,w);
  Stages u=seed_u,sigma=seed_sigma;
  bool passed=false;Certificate current;
  const int limit=pr.o.fixed_outer?pr.o.fixed_outer:pr.o.outer_max;
  for(int outer=0;outer<limit;++outer)
  {
    Stages next=u,next_sigma=sigma;
    for(int e=0;e<pr.ne;++e)
    {
      const auto& patch=patches[e];Stages local,lsigma;
      for(int stage=0;stage<2;++stage)
      {
        local[stage].resize(patch.initial.size());lsigma[stage].resize(patch.ns);
        for(int l=0;l<patch.count;++l)for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)
          local[stage][(l*pr.n+i)*4+c]=u[stage][pr.Index(patch.first+l,i,c)];
        for(int i=0;i<patch.ns;++i)
          lsigma[stage][i]=sigma[stage][pr.SigmaIndex(patch.first,i)];
      }
      for(int inner=0;inner<pr.o.inner;++inner)
      {
        PatchClosures(pr,patch,local,lsigma,w);Stages f,next_local=local;
        for(int stage=0;stage<2;++stage)
          f[stage]=pr.Residual(local[stage],lsigma[stage],patch.count,false,
                              patch.first,&u[stage],&sigma[stage],w);
        for(int stage=0;stage<2;++stage)for(std::size_t i=0;i<patch.initial.size();++i)
          next_local[stage][i]=patch.initial[i]+dt*(q[stage][0]*f[0][i]+q[stage][1]*f[1][i]);
        local.swap(next_local);
      }
      PatchClosures(pr,patch,local,lsigma,w);
      for(int stage=0;stage<2;++stage)
      {
        for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)
          next[stage][pr.Index(e,i,c)]=local[stage][(patch.center*pr.n+i)*4+c];
        // One owner per continuous node. No averaging hides overlap mismatch.
        for(int i=0;i<pr.p;++i)
          next_sigma[stage][pr.SigmaIndex(e,i)]=lsigma[stage][patch.center*pr.p+i];
      }
    }
    u.swap(next);sigma.swap(next_sigma);++w.outer_exchanges;
    current=Certify(pr,initial,dt,u,sigma,w);
    passed=current.closure<=pr.o.closure_tolerance &&
      (pr.o.inner==0 || current.state<=pr.o.state_tolerance);
    if(passed && !pr.o.fixed_outer)break;
  }
  worst.closure=std::max(worst.closure,current.closure);
  worst.state=std::max(worst.state,current.state);certified=certified&&passed;
  if(!passed && !pr.o.fixed_outer)
    throw std::runtime_error("space-time Schwarz certification failed");
  // Conservative corrector: gather the retained central-cell predictions,
  // evaluate EACH unique periodic face once per time node, then use that same
  // flux for both cells. This is charged in addition to predictor/certificate.
  Stages f;
  for(int j=0;j<2;++j)
    f[j]=pr.Residual(u[j],sigma[j],pr.ne,true,0,nullptr,nullptr,w);
  State result(initial.size());
  for(std::size_t i=0;i<result.size();++i)result[i]=initial[i]+.5*dt*(f[0][i]+f[1][i]);
  return result;
}

// Joint current-equation residual: a coupled two-stage nonlinear solve. Sigma
// is an unknown, so no elliptic factorization occurs inside an evaluation.
// This small CPU screen intentionally charges finite-difference Jacobians.
int JointIndex(const Problem& pr,int stage,int e,int component)
{return (stage*pr.ne+e)*(4*pr.n+pr.p)+component;}
State PackJoint(const Problem& pr,const Stages& u,const Stages& sigma)
{
  State z(2*pr.ne*(4*pr.n+pr.p));
  for(int st=0;st<2;++st)for(int e=0;e<pr.ne;++e)
  {
    for(int i=0;i<4*pr.n;++i)z[JointIndex(pr,st,e,i)]=u[st][e*4*pr.n+i];
    for(int i=0;i<pr.p;++i)z[JointIndex(pr,st,e,4*pr.n+i)]=sigma[st][e*pr.p+i];
  }
  return z;
}
void UnpackJoint(const Problem& pr,const State& z,Stages& u,Stages& sigma)
{
  for(int st=0;st<2;++st)
  {
    u[st].resize(pr.ne*pr.n*4);sigma[st].resize(pr.ne*pr.p);
    for(int e=0;e<pr.ne;++e)
    {
      for(int i=0;i<4*pr.n;++i)u[st][e*4*pr.n+i]=z[JointIndex(pr,st,e,i)];
      for(int i=0;i<pr.p;++i)sigma[st][e*pr.p+i]=z[JointIndex(pr,st,e,4*pr.n+i)];
    }
  }
}
State JointResidual(const Problem& pr,const State& initial,double dt,
                    const State& z,Work& w,Certificate* certificate=nullptr)
{
  ++w.joint_residual_calls; // Count attempted evaluations, including admission failure.
  Stages u,sigma,f,sr;UnpackJoint(pr,z,u,sigma);Certificate c{0,0};
  for(int st=0;st<2;++st)
  {
    mfem::DenseMatrix a;State b;pr.Assemble(u[st],pr.ne,true,a,b,w);
    sr[st]=b;
    for(int i=0;i<a.Height();++i)
    {sr[st][i]=-b[i];for(int j=0;j<a.Width();++j)sr[st][i]+=a(i,j)*sigma[st][j];}
    c.closure=std::max(c.closure,Norm(sr[st])/std::max(Norm(b),1.e-20));
    f[st]=pr.Residual(u[st],sigma[st],pr.ne,true,0,nullptr,nullptr,w);
  }
  Stages ur=u;
  for(int st=0;st<2;++st)
  {
    for(std::size_t i=0;i<initial.size();++i)
      ur[st][i]-=initial[i]+dt*(q[st][0]*f[0][i]+q[st][1]*f[1][i]);
    c.state=std::max(c.state,Norm(ur[st])/std::max(Norm(initial),1.e-20));
  }
  if(certificate)*certificate=c;
  return PackJoint(pr,ur,sr);
}
struct JointBlock
{
  std::vector<int> indices;mfem::DenseMatrix matrix;
  std::unique_ptr<mfem::DenseMatrixInverse> inverse;
};
struct JointCache {std::vector<JointBlock> blocks;int age=0;double dt=-1;};
void BuildJointFactors(const Problem& pr,const State& initial,double dt,
                      const State& z,const State& residual,JointCache& cache,Work& w)
{
  const auto start=Clock::now();cache.blocks.clear();
  const int count=pr.o.joint_block_elements?pr.o.joint_block_elements:pr.ne;
  std::size_t bytes=0;
  for(int first=0;first<pr.ne;first+=count)
  {
    cache.blocks.emplace_back();auto& block=cache.blocks.back();
    for(int st=0;st<2;++st)for(int e=first;e<std::min(first+count,pr.ne);++e)
      for(int i=0;i<4*pr.n+pr.p;++i)
        block.indices.push_back(JointIndex(pr,st,pr.Mod(e+pr.o.joint_block_offset),i));
    const int n=static_cast<int>(block.indices.size());block.matrix.SetSize(n);
    for(int j=0;j<n;++j)
    {
      const int col=block.indices[j];State perturbed=z;
      const double eps=2.e-7*std::max(std::abs(z[col]),.01);
      perturbed[col]+=eps;
      const State r=JointResidual(pr,initial,dt,perturbed,w);
      for(int i=0;i<n;++i)block.matrix(i,j)=(r[block.indices[i]]-residual[block.indices[i]])/eps;
      ++w.joint_jacobian_columns;
    }
    block.inverse=std::make_unique<mfem::DenseMatrixInverse>(block.matrix,false);
    ++w.joint_factors;w.factor_flops+=2.*double(n)*n*n/3.;
    bytes+=std::size_t(n)*n*sizeof(double);
  }
  cache.age=0;cache.dt=dt;w.retained_factor_bytes=std::max(w.retained_factor_bytes,bytes);
  w.setup_seconds+=Wall(start);
}
bool JointPassed(const Problem& pr,const Certificate& c)
{return c.closure<=pr.o.closure_tolerance && c.state<=pr.o.state_tolerance;}
double JointMerit(const Problem& pr,const Certificate& c)
{return std::hypot(c.closure/pr.o.closure_tolerance,c.state/pr.o.state_tolerance);}
State JointStep(const Problem& pr,const State& initial,const Stages& seed_u,
                const Stages& seed_sigma,double dt,JointCache& cache,
                Work& w,Certificate& worst)
{
  State z=PackJoint(pr,seed_u,seed_sigma);Certificate current;
  State residual=JointResidual(pr,initial,dt,z,w,&current);
  const bool rebuild=cache.blocks.empty()||std::abs(cache.dt-dt)>1.e-14||
    (pr.o.joint_rebuild>0&&cache.age>=pr.o.joint_rebuild);
  if(!JointPassed(pr,current)&&rebuild)
    BuildJointFactors(pr,initial,dt,z,residual,cache,w);
  for(int iteration=0;!JointPassed(pr,current)&&iteration<pr.o.joint_max;++iteration)
  {
    State delta(z.size(),0.);
    for(const auto& block:cache.blocks)
    {
      const int n=static_cast<int>(block.indices.size());State rb(n),db(n);
      for(int i=0;i<n;++i)rb[i]=-residual[block.indices[i]];
      mfem::Vector rv(rb.data(),n),dv(db.data(),n);block.inverse->Mult(rv,dv);
      for(int i=0;i<n;++i)delta[block.indices[i]]=db[i];
      w.solve_flops+=2.*double(n)*n;
    }
    bool accepted=false;
    for(int line=0;line<12;++line)
    {
      ++w.joint_line_searches;const double fraction=std::ldexp(1.,-line);
      State trial=z;for(std::size_t i=0;i<z.size();++i)trial[i]+=fraction*delta[i];
      try
      {
        Certificate candidate;State rr=JointResidual(pr,initial,dt,trial,w,&candidate);
        if(JointMerit(pr,candidate)<(1.-1.e-4*fraction)*JointMerit(pr,current)||JointPassed(pr,candidate))
        {z.swap(trial);residual.swap(rr);current=candidate;accepted=true;break;}
      }
      catch(const std::runtime_error&){/* Failed admission is charged, never committed. */}
    }
    ++w.joint_iterations;
    if(!accepted)
    {worst.closure=std::max(worst.closure,current.closure);
     worst.state=std::max(worst.state,current.state);
     throw std::runtime_error("joint chord-Newton line search failed");}
  }
  worst.closure=std::max(worst.closure,current.closure);
  worst.state=std::max(worst.state,current.state);
  if(!JointPassed(pr,current))throw std::runtime_error("joint current-equation certification failed");
  ++w.certificate_calls;++cache.age;
  Stages u,sigma,f;UnpackJoint(pr,z,u,sigma);
  for(int st=0;st<2;++st)f[st]=pr.Residual(u[st],sigma[st],pr.ne,true,0,nullptr,nullptr,w);
  State result(initial.size());
  for(std::size_t i=0;i<result.size();++i)result[i]=initial[i]+.5*dt*(f[0][i]+f[1][i]);
  return result;
}

struct Result
{
  State u,sigma;Work work;Certificate certificate{0,0};
  double time=0,seconds=0;bool complete=false,certified=true;
  std::string failure;
};
struct Anchor {double time;State u,sigma;};
// Only accepted, exactly reclosed endpoints enter this bounded history.
// Actual anchor times handle the shortened final timestep without uniform-dt
// coefficients or extrapolation of rejected trial data.
State PredictHistory(const std::vector<Anchor>& history,double time,bool sigma)
{
  if(history.size()!=4 || !std::isfinite(time))
    throw std::runtime_error("invalid cubic history request");
  const State& first=sigma?history[0].sigma:history[0].u;
  State value(first.size(),0.);
  for(int j=0;j<4;++j)
  {
    if(!std::isfinite(history[j].time) ||
       (j && !(history[j].time>history[j-1].time)))
      throw std::runtime_error("nonmonotone accepted history");
    double coefficient=1.;
    for(int k=0;k<4;++k)if(k!=j)
      coefficient*=(time-history[k].time)/(history[j].time-history[k].time);
    const State& sample=sigma?history[j].sigma:history[j].u;
    if(sample.size()!=value.size() || !std::isfinite(coefficient))
      throw std::runtime_error("invalid accepted history field");
    for(std::size_t i=0;i<value.size();++i)value[i]+=coefficient*sample[i];
  }
  if(!std::isfinite(Norm(value)))throw std::runtime_error("nonfinite history predictor");
  return value;
}
Result Run(const Problem& pr,const std::string& method,double step)
{
  Result r;r.u=pr.Initial();const auto start=Clock::now();
  try
  {
    r.sigma=pr.ExactClosure(r.u,r.work);
    std::vector<Anchor> history{{0.,r.u,r.sigma}};JointCache joint_cache;
    while(r.time<pr.o.final_time-1.e-14)
    {
      const double dt=std::min(step,pr.o.final_time-r.time);
      const bool use_history=pr.o.sigma_predictor=="history3" ||
          pr.o.state_predictor=="history3";
      const bool startup=(method=="patch"||method=="joint") && use_history && history.size()<4;
      ++r.work.attempted_steps;
      State trial;
      if(method=="rk4" || startup)trial=RK4Step(pr,r.u,r.sigma,dt,r.work);
      else
      {
        Stages seed_u{{r.u,r.u}},seed_sigma{{r.sigma,r.sigma}};
        for(int stage=0;stage<2;++stage)
        {
          const double time=r.time+dt*(q[stage][0]+q[stage][1]);
          if(pr.o.sigma_predictor=="history3")
            seed_sigma[stage]=PredictHistory(history,time,true);
          if(pr.o.state_predictor=="history3")
            seed_u[stage]=PredictHistory(history,time,false);
        }
        if(method=="joint")trial=JointStep(pr,r.u,seed_u,seed_sigma,dt,joint_cache,r.work,r.certificate);
        else trial=PatchStep(pr,r.u,seed_u,seed_sigma,dt,r.work,r.certificate,r.certified);
      }
      // Instantaneous endpoint closure is always recomputed, including in
      // uncertified fixed-sweep ablations. A failed attempt is not committed.
      State ss=pr.ExactClosure(trial,r.work);
      for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
      {double speed=0;pr.Flux(&trial[pr.Index(e,i,0)],ss[pr.SigmaIndex(e,i)],speed);}
      r.u.swap(trial);r.sigma.swap(ss);r.time+=dt;++r.work.accepted_steps;
      if(startup)++r.work.startup_rk4_steps;
      history.push_back({r.time,r.u,r.sigma});
      if(history.size()>4)history.erase(history.begin());
    }
    r.complete=true;
  }
  catch(const std::exception& error){r.failure=error.what();
    r.work.rejected_steps=r.work.attempted_steps-r.work.accepted_steps;r.certified=false;}
  r.seconds=Wall(start);return r;
}

void Print(const Problem& pr,const std::string& method,const Result& r,
           const Result& reference)
{
  const auto initial=pr.Integrals(pr.Initial()),final=pr.Integrals(r.u);
  double conservation=0;
  for(int c=0;c<4;++c)conservation=std::max(conservation,std::abs(final[c]-initial[c]));
  double sigma_error=-1;
  if(r.complete&&reference.complete)
  {State difference=r.sigma;for(std::size_t i=0;i<difference.size();++i)difference[i]-=reference.sigma[i];
   sigma_error=Norm(difference)/std::max(Norm(reference.sigma),1.e-20);}
  const Work& w=r.work;
  std::cout<<std::setprecision(16)<<"{\"method\":\""<<method
    <<"\",\"model\":\"pseudo1d-full-energy-Euler-instantaneous-IGR\""
    <<",\"closure_space\":\"periodic-continuous-lumped-LGL\""
    <<",\"case\":\""<<pr.o.initial_case<<"\",\"elements\":"<<pr.ne
    <<",\"order\":"<<pr.p<<",\"alpha\":"<<pr.o.alpha<<",\"dt\":"<<pr.o.dt
    <<",\"halo\":"<<pr.o.halo<<",\"screen_length_over_h\":"<<std::sqrt(pr.o.alpha)/pr.h
    <<",\"inner_sweeps\":"<<pr.o.inner<<",\"fixed_outer\":"<<pr.o.fixed_outer
    <<",\"temporal_method\":\""<<(method=="rk4"?"rk4":
       method=="joint-chord-Newton"?"joint-Gauss-chord-Newton":
       pr.o.inner==0?"cubic-history-Gauss-flux-quadrature":"space-time-Schwarz-collocation")<<"\""
    <<",\"collocation_is_target_equation\":"<<(method=="joint-chord-Newton"||(method!="rk4"&&pr.o.inner>0)?"true":"false")
    <<",\"sigma_predictor\":\""<<pr.o.sigma_predictor
    <<"\",\"state_predictor\":\""<<pr.o.state_predictor<<"\""
    <<",\"complete\":"<<(r.complete?"true":"false")
    <<",\"certified\":"<<(r.certified&&r.complete?"true":"false")
    <<",\"reached_time\":"<<r.time<<",\"wall_seconds\":"<<r.seconds
    <<",\"reference_dt\":"<<pr.o.reference_dt<<",\"reference_seconds\":"<<reference.seconds
    <<",\"state_relative_l2_error\":"<<(r.complete&&reference.complete?pr.Error(r.u,reference.u):-1)
    <<",\"sigma_relative_nodal_error\":"<<sigma_error
    <<",\"maximum_conservative_integral_drift\":"<<conservation
    <<",\"maximum_current_closure_residual\":"<<r.certificate.closure
    <<",\"maximum_collocation_residual\":"<<r.certificate.state
    <<",\"global_factors\":"<<w.global_factors<<",\"patch_factors\":"<<w.patch_factors
    <<",\"patch_rhs_solves\":"<<w.patch_rhs_solves<<",\"patch_block_solves\":"<<w.patch_block_solves
    <<",\"coefficient_elements\":"<<w.coefficient_elements
    <<",\"residual_elements\":"<<w.residual_elements<<",\"unique_or_patch_face_fluxes\":"<<w.face_fluxes
    <<",\"outer_waveform_exchanges\":"<<w.outer_exchanges
    <<",\"global_certificates\":"<<w.certificate_calls<<",\"accepted_steps\":"<<w.accepted_steps
    <<",\"startup_rk4_steps\":"<<w.startup_rk4_steps
    <<",\"attempted_steps\":"<<w.attempted_steps<<",\"rejected_steps\":"<<w.rejected_steps
    <<",\"joint_block_elements\":"<<pr.o.joint_block_elements
    <<",\"joint_block_offset\":"<<pr.o.joint_block_offset
    <<",\"joint_rebuild_interval\":"<<pr.o.joint_rebuild
    <<",\"joint_factors\":"<<w.joint_factors
    <<",\"joint_jacobian_columns\":"<<w.joint_jacobian_columns
    <<",\"joint_residual_calls\":"<<w.joint_residual_calls
    <<",\"joint_iterations\":"<<w.joint_iterations
    <<",\"joint_line_searches\":"<<w.joint_line_searches
    <<",\"alpha_from_h\":"<<(pr.o.alpha_from_h?"true":"false")
    <<",\"alpha_power\":"<<pr.o.alpha_power
    <<",\"dense_factor_flops_estimate\":"<<w.factor_flops
    <<",\"dense_solve_flops_estimate\":"<<w.solve_flops
    <<",\"patch_setup_seconds\":"<<w.setup_seconds
    <<",\"retained_patch_factor_bytes\":"<<w.retained_factor_bytes
    <<",\"retained_patch_matrix_and_factor_bytes\":"<<2*w.retained_factor_bytes
    <<",\"failure\":\""<<r.failure<<"\"}\n";
}

Options Parse(int argc,char** argv)
{
  Options o;
  for(int i=1;i<argc;++i)
  {
    const std::string key=argv[i];if(key=="--self-test"){o.self_test=true;continue;}
    if(i+1==argc)throw std::runtime_error("missing option value");
    const std::string value=argv[++i];
    if(key=="--elements")o.elements=std::stoi(value);
    else if(key=="--order")o.order=std::stoi(value);
    else if(key=="--halo")o.halo=std::stoi(value);
    else if(key=="--inner")o.inner=std::stoi(value);
    else if(key=="--outer-max")o.outer_max=std::stoi(value);
    else if(key=="--fixed-outer")o.fixed_outer=std::stoi(value);
    else if(key=="--alpha")o.alpha=std::stod(value);
    else if(key=="--alpha-h-power"){o.alpha_power=std::stod(value);o.alpha_from_h=true;}
    else if(key=="--joint-block-elements")o.joint_block_elements=std::stoi(value);
    else if(key=="--joint-block-offset")o.joint_block_offset=std::stoi(value);
    else if(key=="--joint-rebuild")o.joint_rebuild=std::stoi(value);
    else if(key=="--joint-max")o.joint_max=std::stoi(value);
    else if(key=="--dt")o.dt=std::stod(value);
    else if(key=="--final-time")o.final_time=std::stod(value);
    else if(key=="--reference-dt")o.reference_dt=std::stod(value);
    else if(key=="--closure-tolerance")o.closure_tolerance=std::stod(value);
    else if(key=="--state-tolerance")o.state_tolerance=std::stod(value);
    else if(key=="--case")o.initial_case=value;
    else if(key=="--method")o.method=value;
    else if(key=="--sigma-predictor")o.sigma_predictor=value;
    else if(key=="--state-predictor")o.state_predictor=value;
    else throw std::runtime_error("unknown option "+key);
  }
  if(o.alpha_from_h)o.alpha=1./(double(o.elements)*o.elements*std::pow(o.order+1.,o.alpha_power));
  if(o.joint_block_elements<0||o.joint_block_elements>o.elements||
     o.joint_block_offset<0||o.joint_block_offset>=o.elements||o.joint_rebuild<0||
     o.joint_max<1||!std::isfinite(o.alpha_power)||o.alpha_power<0||
     o.elements<3||o.order<1||o.order>7||o.halo<0||2*o.halo+1>=o.elements||
     o.inner<0||o.outer_max<1||o.fixed_outer<0||!(o.alpha>0)||!(o.dt>0)||
     !(o.final_time>0)||!(o.reference_dt>0)||!(o.closure_tolerance>0)||
     !(o.state_tolerance>0)||!std::isfinite(o.alpha+o.dt+o.final_time+o.reference_dt+
          o.closure_tolerance+o.state_tolerance)||
     (o.initial_case!="smooth"&&o.initial_case!="compression")||
     (o.sigma_predictor!="constant"&&o.sigma_predictor!="history3")||
     (o.state_predictor!="constant"&&o.state_predictor!="history3")||
     (o.method!="rk4"&&o.method!="patch"&&o.method!="both"&&o.method!="joint"&&o.method!="rk4-joint"))
    throw std::runtime_error("unsupported/nonfinite probe options");
  if(o.inner==0 && (o.fixed_outer<1 || o.sigma_predictor!="history3" ||
                      o.state_predictor!="history3"))
    throw std::runtime_error("inner0 requires fixed outer sweeps and joint cubic history");
  return o;
}

bool SelfTest()
{
  const auto polynomial=[](double t){return 1.+2.*t-3.*t*t+.5*t*t*t;};
  std::vector<Anchor> history;
  for(double t:{0.,.07,.19,.3})history.push_back({t,{polynomial(t)},{polynomial(t)}});
  for(bool sigma:{false,true})
    if(std::abs(PredictHistory(history,.337,sigma)[0]-polynomial(.337))>2.e-14)return false;
  Options o;o.elements=6;Problem pr(o);Work w;const State u=pr.Initial();
  State sigma=pr.ExactClosure(u,w),rhs=pr.Residual(u,sigma,pr.ne,true,0,nullptr,nullptr,w);
  const auto sum=pr.Integrals(rhs);
  for(double s:sum)if(!std::isfinite(s)||std::abs(s)>1.e-11)return false;
  mfem::DenseMatrix a;State b;pr.Assemble(u,pr.ne,true,a,b,w);
  State residual=b;for(int i=0;i<a.Height();++i)
    for(int j=0;j<a.Width();++j)residual[i]-=a(i,j)*sigma[j];
  if(Norm(residual)>1.e-11*Norm(b))return false;
  auto patches=BuildPatches(pr,u,w);const Patch& patch=patches[2];
  Stages local{{patch.initial,patch.initial}},lsigma;
  // Nonconstant changed density perturbs reaction and diffusion together.
  for(int stage=0;stage<2;++stage)
  {
    for(std::size_t i=0;i<local[stage].size();i+=4)
      local[stage][i]*=1.+(.01+.01*stage)*std::sin(double(i+1));
    lsigma[stage].resize(patch.ns);
    for(int i=0;i<patch.ns;++i)lsigma[stage][i]=sigma[pr.SigmaIndex(patch.first,i)];
  }
  PatchClosures(pr,patch,local,lsigma,w);
  for(int stage=0;stage<2;++stage)
  {
    pr.Assemble(local[stage],patch.count,false,a,b,w);State rr(patch.ns-2);
    for(int i=1;i<patch.ns-1;++i)
    {rr[i-1]=b[i];for(int j=0;j<patch.ns;++j)rr[i-1]-=a(i,j)*lsigma[stage][j];}
    if(Norm(rr)>o.closure_tolerance*Norm(b))return false;
  }
  // Boundary-driven screened solve with identically zero volume source.
  // This must not demand a relative residual scaled by that zero source.
  State constant(patch.initial.size());
  for(std::size_t i=0;i<constant.size();i+=4)
  {constant[i]=1.;constant[i+1]=.2;constant[i+2]=0.;constant[i+3]=2.52;}
  local={{constant,constant}};
  for(auto& s:lsigma){std::fill(s.begin(),s.end(),0.);s.front()=.1;s.back()=.2;}
  PatchClosures(pr,patch,local,lsigma,w);
  // A converged joint solve must satisfy the original stage equations and
  // share the same conservative update, including with a retained Jacobian.
  Options jo;jo.elements=4;jo.order=1;jo.halo=0;jo.dt=.002;
  jo.final_time=.004;jo.reference_dt=.0000625;
  Problem jp(jo);const Result reference=Run(jp,"rk4",jo.reference_dt);
  const Result joint=Run(jp,"joint",jo.dt);
  if(!reference.complete||!joint.complete||!joint.certified||
     joint.work.joint_factors!=1||joint.work.joint_iterations<2||
     jp.Error(joint.u,reference.u)>1.e-8)return false;
  const auto jbefore=jp.Integrals(jp.Initial()),jafter=jp.Integrals(joint.u);
  for(int c=0;c<4;++c)if(std::abs(jafter[c]-jbefore[c])>2.e-12)return false;
  jo.joint_block_elements=2;jo.joint_block_offset=1;jo.alpha=.0001;
  Problem shifted(jo);const Result shift_ref=Run(shifted,"rk4",jo.reference_dt);
  const Result shift_joint=Run(shifted,"joint",jo.dt);
  if(!shift_ref.complete||!shift_joint.complete||!shift_joint.certified||
     shift_joint.work.joint_factors!=2||shifted.Error(shift_joint.u,shift_ref.u)>1.e-8)return false;
  std::cout<<"screened space-time conservation, current-density correction, and joint chord-Newton PASS\n";
  return true;
}
} // namespace

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    const Options o=Parse(argc,argv);if(o.self_test)return SelfTest()?0:1;
    Problem pr(o);const Result reference=Run(pr,"rk4",o.reference_dt);
    if(!reference.complete)throw std::runtime_error("reference failed: "+reference.failure);
    if(o.method=="both"||o.method=="rk4"||o.method=="rk4-joint")Print(pr,"rk4",Run(pr,"rk4",o.dt),reference);
    if(o.method=="joint"||o.method=="rk4-joint")
      Print(pr,"joint-chord-Newton",Run(pr,"joint",o.dt),reference);
    if(o.method=="both"||o.method=="patch")
      Print(pr,o.inner==0?"cubic-history-Gauss-quadrature":"space-time-Schwarz",
            Run(pr,"patch",o.dt),reference);
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
  return 0;
}

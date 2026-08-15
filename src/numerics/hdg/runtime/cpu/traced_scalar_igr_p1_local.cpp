#include "numerics/hdg/runtime/cpu/traced_scalar_igr_p1_local.hpp"

#include "physics/igr/frozen_scalar_igr_algebra.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace dg::hdg::runtime::cpu::traced_scalar_igr_p1
{
namespace
{
using V2 = std::array<double, 2>;
using U4 = std::array<double, 4>;
using J45 = std::array<std::array<double, 5>, 4>;

bool Finite(const double x) noexcept { return std::isfinite(x); }
int I(const int component, const int scalar) noexcept { return component * 4 + scalar; }
int T(const int component, const int face_scalar) noexcept { return component * 8 + face_scalar; }
double& M(std::vector<double>& matrix, const int columns, const int row, const int column) noexcept
{ return matrix[static_cast<std::size_t>(row) * columns + column]; }


void Shapes(const double xi, const double eta, std::array<double,4>& n,
            std::array<V2,4>& dr) noexcept
{
  constexpr double a = 0.21132486540518711775;
  constexpr double b = 0.78867513459481288225;
  constexpr double s = 1.0 / (b-a);
  const double lx[2] = {(b-xi)*s, (xi-a)*s};
  const double ly[2] = {(b-eta)*s, (eta-a)*s};
  const double dl[2] = {-s,s};
  for (int y=0; y<2; ++y) for (int x=0; x<2; ++x)
  { const int q=x+2*y; n[q]=lx[x]*ly[y]; dr[q]={{dl[x]*ly[y],lx[x]*dl[y]}}; }
}
void FacePoint(const int face, const double s, double& xi, double& eta, V2& tangent) noexcept
{
  switch(face) { case 0: xi=s; eta=0.; tangent={{1.,0.}}; break; case 1: xi=1.; eta=s; tangent={{0.,1.}}; break; case 2: xi=1.-s; eta=1.; tangent={{-1.,0.}}; break; default: xi=0.; eta=1.-s; tangent={{0.,-1.}}; }
}
bool Admissible(const U4& u, const Parameters& p) noexcept
{
  if (!Finite(u[0]) || !(u[0] > p.density_floor)) return false;
  const double pressure=(p.gamma-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
  return Finite(pressure) && pressure > 0.;
}
bool EulerSigma(const U4& u, const double sigma, const V2& normal,
                const Parameters& p, U4& flux, J45& jac) noexcept
{
  if (!Admissible(u,p) || !Finite(sigma)) return false;
  const double rho=u[0], mx=u[1], my=u[2], e=u[3], gm1=p.gamma-1.;
  const double pressure=gm1*(e-.5*(mx*mx+my*my)/rho), un=(mx*normal[0]+my*normal[1])/rho;
  flux={{mx*normal[0]+my*normal[1], mx*un+pressure*normal[0], my*un+pressure*normal[1], (e+pressure)*un}};
  for(auto& r:jac) r.fill(0.);
  const U4 dp{{.5*gm1*(mx*mx+my*my)/(rho*rho),-gm1*mx/rho,-gm1*my/rho,gm1}};
  const U4 dun{{-un/rho,normal[0]/rho,normal[1]/rho,0.}};
  jac[0][1]=normal[0]; jac[0][2]=normal[1];
  for(int c=0;c<4;++c) { jac[1][c]=(c==1?un:0.)+mx*dun[c]+dp[c]*normal[0]; jac[2][c]=(c==2?un:0.)+my*dun[c]+dp[c]*normal[1]; jac[3][c]=((c==3?1.:0.)+dp[c])*un+(e+pressure)*dun[c]; }
  ::dg_igr::frozen_scalar_igr_algebra::AddNormalSigmaFlux(
      2, u.data(), 1, normal.data(), 1, sigma, p.density_floor,
      flux.data(), 1);
  ::dg_igr::frozen_scalar_igr_algebra::AddNormalSigmaFluxJacobian(
      2, u.data(), 1, normal.data(), 1, sigma, p.density_floor,
      &jac[0][0], 5, 1);
  // Sigma remains an independent mixed unknown; shared algebra is intentionally
  // frozen with respect to U and therefore does not provide this fifth column.
  jac[1][4]=normal[0]; jac[2][4]=normal[1]; jac[3][4]=un;
  return std::all_of(flux.begin(),flux.end(),Finite);
}
U4 Value(const double* x, const std::array<double,4>& n) noexcept
{ U4 r{}; for(int c=0;c<4;++c) for(int q=0;q<4;++q) r[c]+=x[I(c,q)]*n[q]; return r; }
double SValue(const double* x,const int c,const std::array<double,4>& n) noexcept
{ double r=0.; for(int q=0;q<4;++q) r+=x[I(c,q)]*n[q]; return r; }
double Derivative(const double* x,const int c,const std::array<V2,4>& d,const int k) noexcept
{ double r=0.;for(int q=0;q<4;++q)r+=x[I(c,q)]*d[q][k];return r; }
bool Geometry(const AffineQuad& q, std::array<V2,2>& j, double& det) noexcept
{
  for (const V2& vertex : q.vertices)
    if (!Finite(vertex[0]) || !Finite(vertex[1])) return false;
  j[0]={{q.vertices[1][0]-q.vertices[0][0],q.vertices[1][1]-q.vertices[0][1]}};
  j[1]={{q.vertices[3][0]-q.vertices[0][0],q.vertices[3][1]-q.vertices[0][1]}};
  const V2 expected{{q.vertices[1][0]+j[1][0],q.vertices[1][1]+j[1][1]}};
  if (!Finite(j[0][0])||!Finite(j[0][1])||!Finite(j[1][0])||!Finite(j[1][1])||std::abs(expected[0]-q.vertices[2][0])>1e-12||std::abs(expected[1]-q.vertices[2][1])>1e-12) return false;
  det=j[0][0]*j[1][1]-j[1][0]*j[0][1]; return Finite(det)&&det>0.;
}
bool AllFinite(const LocalOperator& o) noexcept
{ const auto f=[](const std::vector<double>& v){return std::all_of(v.begin(),v.end(),Finite);}; return f(o.interior_residual)&&f(o.trace_residual)&&f(o.A)&&f(o.B)&&f(o.C)&&f(o.D); }
}

void LocalOperator::Resize()
{
  interior_residual.assign(interior_unknowns,0.); trace_residual.assign(trace_unknowns,0.);
  A.assign(interior_unknowns*interior_unknowns,0.); B.assign(interior_unknowns*trace_unknowns,0.);
  C.assign(trace_unknowns*interior_unknowns,0.); D.assign(trace_unknowns*trace_unknowns,0.);
  f.assign(interior_unknowns,0.); g.assign(trace_unknowns,0.);
}

Status Assemble(const AffineQuad& quad, const Parameters& p, const ElementState& s, LocalOperator& o) noexcept
{
  if (!s.interior||!s.previous_interior||!s.trace||!s.previous_trace||!Finite(p.gamma)||!(p.gamma>1.)||!Finite(p.alpha)||!(p.alpha>0.)||!Finite(p.beta)||!(p.beta>0.)||!Finite(p.dt)||!(p.dt>0.)||!Finite(p.density_floor)||!(p.density_floor>0.)||p.quadrature_order<2||p.quadrature_order>32) return Status::invalid_parameter;
  std::array<V2,2> j{}; double det=0.; if(!Geometry(quad,j,det)) return Status::invalid_geometry;
  const mfem::IntegrationRule* volume_rule_ptr = nullptr;
  try { volume_rule_ptr = &mfem::IntRules.Get(mfem::Geometry::SQUARE, p.quadrature_order); }
  catch (...) { return Status::invalid_parameter; }
  const mfem::IntegrationRule& volume_rule = *volume_rule_ptr;
  if (volume_rule.GetNPoints() <= 0) return Status::invalid_parameter;
  const std::size_t nq = static_cast<std::size_t>(volume_rule.GetNPoints());
  if(s.static_euler_rhs.data && s.static_euler_rhs.sample_count!=nq) return Status::invalid_layout; if(!s.static_euler_rhs.data&&s.static_euler_rhs.sample_count) return Status::invalid_layout;
  try { o.Resize(); } catch(...) { return Status::invalid_layout; }
  const double inv[2][2]={{j[1][1]/det,-j[1][0]/det},{-j[0][1]/det,j[0][0]/det}};
  for(int point=0;point<volume_rule.GetNPoints();++point)
  {
    const mfem::IntegrationPoint& ip = volume_rule.IntPoint(point);
    std::array<double,4> n{}; std::array<V2,4> dr{}, d{}; Shapes(ip.x,ip.y,n,dr);
    for(int a=0;a<4;++a) for(int k=0;k<2;++k) d[a][k]=dr[a][0]*inv[0][k]+dr[a][1]*inv[1][k];
    const U4 u=Value(s.interior,n), old=Value(s.previous_interior,n); if(!Admissible(u,p)||!Admissible(old,p)) return Status::inadmissible_state;
    const double sigma=SValue(s.interior,4,n); const V2 z{{SValue(s.interior,5,n),SValue(s.interior,6,n)}};
    U4 fx{},fy{}; J45 jx{},jy{}; if(!EulerSigma(u,sigma,{{1.,0.}},p,fx,jx)||!EulerSigma(u,sigma,{{0.,1.}},p,fy,jy)) return Status::inadmissible_state;
    std::array<V2,2> vg{}; for(int v=0;v<2;++v) for(int k=0;k<2;++k){const double du=Derivative(s.interior,0,d,k), dm=Derivative(s.interior,1+v,d,k); vg[v][k]=dm/u[0]-(v==0?u[1]:u[2])*du/(u[0]*u[0]);}
    const double tr=vg[0][0]+vg[1][1], q=vg[0][0]*vg[0][0]+2.*vg[0][1]*vg[1][0]+vg[1][1]*vg[1][1]+tr*tr;
    std::array<V2,2> dqg{}; for(int v=0;v<2;++v)for(int k=0;k<2;++k)dqg[v][k]=2.*vg[k][v]+(v==k?2.*tr:0.);
    const double ww=ip.weight*det;
    for(int r=0;r<4;++r)
    {
      for(int e=0;e<4;++e)
      { const int row=I(e,r); const double rhs=s.static_euler_rhs.data?s.static_euler_rhs.data[point*4+e]:0.; if(!Finite(rhs))return Status::invalid_parameter; o.interior_residual[row]+=ww*(n[r]*(u[e]-old[e])/p.dt-d[r][0]*fx[e]-d[r][1]*fy[e]-n[r]*rhs);
        for(int c=0;c<4;++c)for(int v=0;v<5;++v){double zc=(v==e?n[r]*n[c]/p.dt:0.)-d[r][0]*jx[e][v]*n[c]-d[r][1]*jy[e][v]*n[c];M(o.A,28,row,I(v,c))+=ww*zc;}
      }
      for(int k=0;k<2;++k) { const int row=I(4+k,r); o.interior_residual[row]+=ww*(n[r]*u[0]*z[k]+d[r][k]*sigma); for(int c=0;c<4;++c){M(o.A,28,row,I(0,c))+=ww*n[r]*z[k]*n[c];M(o.A,28,row,I(4,c))+=ww*d[r][k]*n[c];M(o.A,28,row,I(5+k,c))+=ww*n[r]*u[0]*n[c];} }
      const int rs=I(6,r); o.interior_residual[rs]+=ww*(n[r]*sigma/u[0]+p.alpha*(d[r][0]*z[0]+d[r][1]*z[1])-p.alpha*n[r]*q);
      for(int c=0;c<4;++c){M(o.A,28,rs,I(4,c))+=ww*n[r]*n[c]/u[0];for(int k=0;k<2;++k)M(o.A,28,rs,I(5+k,c))+=ww*p.alpha*d[r][k]*n[c];for(int v=0;v<4;++v){double dq=0.;if(v<3)for(int vel=0;vel<2;++vel)for(int k=0;k<2;++k){const double mom=vel==0?u[1]:u[2], dm=Derivative(s.interior,1+vel,d,k), drho=Derivative(s.interior,0,d,k);const double delta_r=v==0?n[c]:0., delta_dr=v==0?d[c][k]:0., delta_m=v==1+vel?n[c]:0., delta_dm=v==1+vel?d[c][k]:0.;const double delta_g=delta_dm/u[0]-dm*delta_r/(u[0]*u[0])-delta_m*drho/(u[0]*u[0])-mom*delta_dr/(u[0]*u[0])+2.*mom*drho*delta_r/(u[0]*u[0]*u[0]);dq+=dqg[vel][k]*delta_g;} double val=-p.alpha*n[r]*dq; if(v==0)val-=n[r]*sigma*n[c]/(u[0]*u[0]);M(o.A,28,rs,I(v,c))+=ww*val;}}
    }
  }
  // Element-oriented counter-clockwise faces: bottom, right, top, left.
  const mfem::IntegrationRule* face_rule_ptr = nullptr;
  try { face_rule_ptr = &mfem::IntRules.Get(mfem::Geometry::SEGMENT, p.quadrature_order); }
  catch (...) { return Status::invalid_parameter; }
  const mfem::IntegrationRule& face_rule = *face_rule_ptr;
  if (face_rule.GetNPoints() <= 0) return Status::invalid_parameter;
  for(int face=0;face<4;++face) for(int qp=0;qp<face_rule.GetNPoints();++qp)
  {
    const mfem::IntegrationPoint& ip = face_rule.IntPoint(qp);
    double xi=0.,eta=0.;V2 tangent{};FacePoint(face,ip.x,xi,eta,tangent); std::array<double,4> n{};std::array<V2,4> dr{},d{};Shapes(xi,eta,n,dr);for(int a=0;a<4;++a)for(int k=0;k<2;++k)d[a][k]=dr[a][0]*inv[0][k]+dr[a][1]*inv[1][k];
    V2 physical_t{{j[0][0]*tangent[0]+j[1][0]*tangent[1],j[0][1]*tangent[0]+j[1][1]*tangent[1]}}; const double length=std::hypot(physical_t[0],physical_t[1]); if(!(length>0.)||!Finite(length))return Status::invalid_geometry; const V2 normal{{physical_t[1]/length,-physical_t[0]/length}}; const double h=det/length;
    std::array<double,2> hshape{{(0.78867513459481288225-ip.x)*1.7320508075688772935,(ip.x-0.21132486540518711775)*1.7320508075688772935}};
    const U4 u=Value(s.interior,n),old=Value(s.previous_interior,n); U4 uh{},oldh{}; for(int c=0;c<4;++c)for(int a=0;a<2;++a){uh[c]+=s.trace[T(c,face*2+a)]*hshape[a];oldh[c]+=s.previous_trace[T(c,face*2+a)]*hshape[a];} if(!Admissible(u,p)||!Admissible(old,p)||!Admissible(uh,p)||!Admissible(oldh,p))return Status::inadmissible_state;
    const double sigma=SValue(s.interior,4,n);const V2 z{{SValue(s.interior,5,n),SValue(s.interior,6,n)}};double sh=0.;for(int a=0;a<2;++a)sh+=s.trace[T(4,face*2+a)]*hshape[a]; U4 flux{};J45 jac{};if(!EulerSigma(uh,sh,normal,p,flux,jac))return Status::inadmissible_state; const double tau=4./(uh[0]*h),jump=sigma-sh,zh=z[0]*normal[0]+z[1]*normal[1]-tau*jump,ww=ip.weight*length; if(!Finite(tau)||!(tau>0.)||!Finite(zh))return Status::nonfinite_result;
    for(int r=0;r<4;++r){for(int e=0;e<4;++e){const int row=I(e,r);const double fh=flux[e]+p.beta*(u[e]-uh[e]);o.interior_residual[row]+=ww*n[r]*fh;for(int c=0;c<4;++c)M(o.A,28,row,I(e,c))+=ww*n[r]*p.beta*n[c];for(int v=0;v<5;++v)for(int a=0;a<2;++a)M(o.B,40,row,T(v,face*2+a))+=ww*n[r]*(jac[e][v]-(v==e?p.beta:0.))*hshape[a];}for(int k=0;k<2;++k){const int row=I(4+k,r);o.interior_residual[row]-=ww*n[r]*normal[k]*sh;for(int a=0;a<2;++a)M(o.B,40,row,T(4,face*2+a))-=ww*n[r]*normal[k]*hshape[a];}const int row=I(6,r);o.interior_residual[row]-=p.alpha*ww*n[r]*zh;for(int c=0;c<4;++c){M(o.A,28,row,I(4,c))+=p.alpha*ww*n[r]*tau*n[c];for(int k=0;k<2;++k)M(o.A,28,row,I(5+k,c))-=p.alpha*ww*n[r]*normal[k]*n[c];}for(int a=0;a<2;++a){M(o.B,40,row,T(4,face*2+a))-=p.alpha*ww*n[r]*tau*hshape[a];M(o.B,40,row,T(0,face*2+a))-=p.alpha*ww*n[r]*tau*jump/uh[0]*hshape[a];}}
    for(int rc=0;rc<5;++rc)for(int r=0;r<2;++r){const int row=T(rc,face*2+r);if(rc<4){const double fh=flux[rc]+p.beta*(u[rc]-uh[rc]);o.trace_residual[row]+=ww*hshape[r]*fh;for(int c=0;c<4;++c)M(o.C,28,row,I(rc,c))+=ww*hshape[r]*p.beta*n[c];for(int v=0;v<5;++v)for(int a=0;a<2;++a)M(o.D,40,row,T(v,face*2+a))+=ww*hshape[r]*(jac[rc][v]-(v==rc?p.beta:0.))*hshape[a];}else{o.trace_residual[row]+=p.alpha*ww*hshape[r]*zh;for(int c=0;c<4;++c){M(o.C,28,row,I(4,c))-=p.alpha*ww*hshape[r]*tau*n[c];for(int k=0;k<2;++k)M(o.C,28,row,I(5+k,c))+=p.alpha*ww*hshape[r]*normal[k]*n[c];}for(int a=0;a<2;++a){M(o.D,40,row,T(4,face*2+a))+=p.alpha*ww*hshape[r]*tau*hshape[a];M(o.D,40,row,T(0,face*2+a))+=p.alpha*ww*hshape[r]*tau*jump/uh[0]*hshape[a];}}}
  }
  if(!AllFinite(o))return Status::nonfinite_result; for(int i=0;i<28;++i)o.f[i]=-o.interior_residual[i];for(int i=0;i<40;++i)o.g[i]=-o.trace_residual[i]; return Status::success;
}

core::LocalSystemBatchView HostReferenceInput(const LocalOperator& o) noexcept
{
  const auto v=[](const double* d,int r,int c){return StridedBlockView<const double>{d,1,r,c,r*c,c,1};};
  return {{4,8,7,5},v(o.A.data(),28,28),v(o.B.data(),28,40),v(o.C.data(),40,28),v(o.D.data(),40,40),v(o.f.data(),28,1),v(o.g.data(),40,1)};
}
} // namespace dg::hdg::runtime::cpu::traced_scalar_igr_p1

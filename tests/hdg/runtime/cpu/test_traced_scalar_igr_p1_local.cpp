#include "numerics/hdg/runtime/cpu/traced_scalar_igr_p1_local.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
namespace op = dg::hdg::runtime::cpu::traced_scalar_igr_p1;

bool Near(const double a, const double b, const double tolerance = 2.e-5)
{ return std::abs(a-b) <= tolerance * std::max({1.,std::abs(a),std::abs(b)}); }

bool Solve(std::vector<double> a, std::vector<double> b, std::vector<double>& x)
{
  const int n=static_cast<int>(b.size()); x=b;
  for(int k=0;k<n;++k){int p=k;for(int r=k+1;r<n;++r)if(std::abs(a[r*n+k])>std::abs(a[p*n+k]))p=r;if(!(std::abs(a[p*n+k])>1.e-13))return false;for(int c=k;c<n;++c)std::swap(a[k*n+c],a[p*n+c]);std::swap(x[k],x[p]);for(int r=k+1;r<n;++r){const double m=a[r*n+k]/a[k*n+k];for(int c=k;c<n;++c)a[r*n+c]-=m*a[k*n+c];x[r]-=m*x[k];}}
  for(int r=n-1;r>=0;--r){for(int c=r+1;c<n;++c)x[r]-=a[r*n+c]*x[c];x[r]/=a[r*n+r];}return true;
}

bool Check()
{
  op::AffineQuad geometry{{{{.2,-.1},{2.2,.2},{2.6,1.5},{.6,1.2}}}};
  op::Parameters p{}; p.alpha=.7;p.beta=1.3;p.dt=.09;p.quadrature_order=12;
  const int n12=mfem::IntRules.Get(mfem::Geometry::SQUARE,12).GetNPoints();
  std::vector<double> u(28),old(28),trace(40),old_trace(40),rhs(static_cast<std::size_t>(n12)*4,0.);
  for(int c=0;c<4;++c)for(int q=0;q<4;++q){u[c*4+q]=(c==0?1.0+.02*q:c==1?.15+.01*q:c==2?-.05+.008*q:2.7+.04*q);old[c*4+q]=u[c*4+q]-.003*(c+1);}
  for(int q=0;q<4;++q){u[16+q]=.9+.03*q;old[16+q]=u[16+q]-.01;u[20+q]=.02+.01*q;u[24+q]=-.01+.004*q;old[20+q]=u[20+q];old[24+q]=u[24+q];}
  for(int c=0;c<5;++c)for(int q=0;q<8;++q){trace[c*8+q]=(c==0?1.01:c==1?.14:c==2?-.04:c==3?2.71:.92)+.002*q;old_trace[c*8+q]=trace[c*8+q]-.001;}
  for(int q=0;q<n12;++q)rhs[static_cast<std::size_t>(q)*4+3]=.03;
  op::ElementState s{u.data(),old.data(),trace.data(),old_trace.data(),{rhs.data(),static_cast<std::size_t>(n12)}};
  op::LocalOperator base; if(op::Assemble(geometry,p,s,base)!=op::Status::success||base.A.size()!=28*28||base.B.size()!=28*40||base.C.size()!=40*28||base.D.size()!=40*40)return false;
  if(!std::any_of(base.A.begin(),base.A.end(),[](double x){return x!=0.;})||!std::any_of(base.B.begin(),base.B.end(),[](double x){return x!=0.;})||!std::any_of(base.C.begin(),base.C.end(),[](double x){return x!=0.;})||!std::any_of(base.D.begin(),base.D.end(),[](double x){return x!=0.;}))return false;
  // The auxiliary Sigma/Sigmahat face penalty has the required [+ -; - +] signs.
  const int sigma_row=24, sigmahat=32;
  if(!(base.A[sigma_row*28+16]>0.&&base.B[sigma_row*40+sigmahat]<0.&&base.C[sigmahat*28+16]<0.&&base.D[sigmahat*40+sigmahat]>0.))return false;
  op::AffineQuad reversed=geometry;std::swap(reversed.vertices[1],reversed.vertices[3]);op::LocalOperator bad;if(op::Assemble(reversed,p,s,bad)!=op::Status::invalid_geometry)return false;
  op::AffineQuad nan_vertex=geometry;nan_vertex.vertices[2][0]=std::numeric_limits<double>::quiet_NaN();if(op::Assemble(nan_vertex,p,s,bad)!=op::Status::invalid_geometry)return false;
  // Independent constant-state equation witness: volume and all four outward
  // face terms cancel exactly when Uhat=U, Sigma=Sigmahat=z=0, and Uold=U.
  std::vector<double> equilibrium(28,0.),equilibrium_trace(40,0.);
  for(int q=0;q<4;++q){equilibrium[q]=1.;equilibrium[4+q]=.1;equilibrium[8+q]=-.02;equilibrium[12+q]=2.6;}
  for(int c=0;c<4;++c)for(int q=0;q<8;++q)equilibrium_trace[c*8+q]=equilibrium[c*4];
  op::ElementState exact{equilibrium.data(),equilibrium.data(),equilibrium_trace.data(),equilibrium_trace.data(),{}};op::LocalOperator exact_op;if(op::Assemble(geometry,p,exact,exact_op)!=op::Status::success)return false;for(double value:exact_op.interior_residual)if(!Near(value,0.,3.e-10))return false;
  // Static Euler RHS changes only R_U and has no Newton contribution.
  op::ElementState zero=s;zero.static_euler_rhs={};op::LocalOperator without; if(op::Assemble(geometry,p,zero,without)!=op::Status::success||without.A!=base.A||without.B!=base.B||without.C!=base.C||without.D!=base.D)return false;
  for(int i=16;i<28;++i)if(!Near(without.interior_residual[i],base.interior_residual[i],1.e-12))return false;
  // Every one of the 68 analytic columns is checked against centered FD.
  std::vector<double> unknown(68);std::copy(u.begin(),u.end(),unknown.begin());std::copy(trace.begin(),trace.end(),unknown.begin()+28);
  for(int column=0;column<68;++column){const double h=2.e-7*std::max(1.,std::abs(unknown[column]));auto plus=unknown,minus=unknown;plus[column]+=h;minus[column]-=h;op::ElementState sp{plus.data(),old.data(),plus.data()+28,old_trace.data(),{rhs.data(),static_cast<std::size_t>(n12)}},sm{minus.data(),old.data(),minus.data()+28,old_trace.data(),{rhs.data(),static_cast<std::size_t>(n12)}};op::LocalOperator lp,lm;if(op::Assemble(geometry,p,sp,lp)!=op::Status::success||op::Assemble(geometry,p,sm,lm)!=op::Status::success)return false;for(int row=0;row<28;++row){const double fd=(lp.interior_residual[row]-lm.interior_residual[row])/(2*h);const double an=column<28?base.A[row*28+column]:base.B[row*40+column-28];if(!Near(fd,an))return false;}for(int row=0;row<40;++row){const double fd=(lp.trace_residual[row]-lm.trace_residual[row])/(2*h);const double an=column<28?base.C[row*28+column]:base.D[row*40+column-28];if(!Near(fd,an))return false;}}
  // A deterministic mixed interior/trace direction checks the assembled 68x68 action.
  auto plus=unknown,minus=unknown;std::vector<double> direction(68);for(int i=0;i<68;++i){direction[i]=(i%5-2)*.07;plus[i]+=1.e-6*direction[i];minus[i]-=1.e-6*direction[i];}op::ElementState sp{plus.data(),old.data(),plus.data()+28,old_trace.data(),{rhs.data(),static_cast<std::size_t>(n12)}},sm{minus.data(),old.data(),minus.data()+28,old_trace.data(),{rhs.data(),static_cast<std::size_t>(n12)}};op::LocalOperator lp,lm;if(op::Assemble(geometry,p,sp,lp)!=op::Status::success||op::Assemble(geometry,p,sm,lm)!=op::Status::success)return false;for(int r=0;r<68;++r){double jv=0.;for(int c=0;c<68;++c){const double entry=r<28?(c<28?base.A[r*28+c]:base.B[r*40+c-28]):(c<28?base.C[(r-28)*28+c]:base.D[(r-28)*40+c-28]);jv+=entry*direction[c];}const double fd=((r<28?lp.interior_residual[r]:lp.trace_residual[r-28])-(r<28?lm.interior_residual[r]:lm.trace_residual[r-28]))/2.e-6;if(!Near(fd,jv))return false;}
  // The HostReference sign adapter must be usable and agrees with dense 68x68 Newton.
  dg::hdg::core::HostReferenceEngine host; std::vector<double> schur(40*40),reduced(40),lambda(40),interior(28);auto view=op::HostReferenceInput(base);auto block=[](double* d,int r,int c){return dg::hdg::StridedBlockView<double>{d,1,r,c,r*c,c,1};};if(host.Condense(view,{block(schur.data(),40,40),block(reduced.data(),40,1)})!=dg::hdg::core::Status::success||!Solve(schur,reduced,lambda)||host.Recover(dg::hdg::StridedBlockView<const double>{lambda.data(),1,40,1,40,1,1},block(interior.data(),28,1))!=dg::hdg::core::Status::success)return false;
  std::vector<double> full(68*68,0.),rr(68);for(int r=0;r<28;++r){rr[r]=-base.interior_residual[r];for(int c=0;c<28;++c)full[r*68+c]=base.A[r*28+c];for(int c=0;c<40;++c)full[r*68+28+c]=base.B[r*40+c];}for(int r=0;r<40;++r){rr[28+r]=-base.trace_residual[r];for(int c=0;c<28;++c)full[(28+r)*68+c]=base.C[r*28+c];for(int c=0;c<40;++c)full[(28+r)*68+28+c]=base.D[r*40+c];}std::vector<double> direct;if(!Solve(full,rr,direct))return false;for(int i=0;i<28;++i)if(!Near(interior[i],direct[i]))return false;for(int i=0;i<40;++i)if(!Near(lambda[i],direct[28+i]))return false;
  const int n14=mfem::IntRules.Get(mfem::Geometry::SQUARE,14).GetNPoints();std::vector<double> rhs14(static_cast<std::size_t>(n14)*4,0.);for(int q=0;q<n14;++q)rhs14[static_cast<std::size_t>(q)*4+3]=.03;
  p.quadrature_order=14;op::ElementState s14=s;s14.static_euler_rhs={rhs14.data(),static_cast<std::size_t>(n14)};op::LocalOperator high;if(op::Assemble(geometry,p,s14,high)!=op::Status::success)return false;for(std::size_t i=0;i<base.interior_residual.size();++i)if(!Near(base.interior_residual[i],high.interior_residual[i],3.e-6))return false;
  return true;
}
}
int main(){return Check()?0:1;}

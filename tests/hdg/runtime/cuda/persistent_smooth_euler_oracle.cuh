#pragma once

#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_types.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mpi.h>

namespace cuda_w8a_test_oracle
{
using namespace dg::hdg::runtime::cuda_w8a;
inline constexpr double tolerance = 3.0e-10;
inline constexpr double gamma = 1.4;
inline constexpr double tau = 2.5;
inline constexpr double pi = 3.141592653589793238462643383279502884;

inline int Nodal(const int scalar, const int count, const int component)
{ return scalar + count * component; }

inline bool Close(const double a, const double b, const double scale = 1.0)
{
  return std::isfinite(a) && std::isfinite(b) &&
      std::abs(a - b) <= tolerance *
          std::max({scale, std::abs(a), std::abs(b)});
}

inline bool Flux(const double* u, const double nx, const double ny,
                 double* flux, double* jacobian)
{
  const double rho=u[0], mx=u[1], my=u[2], energy=u[3];
  if (!(rho>0.0)) return false;
  const double pressure=(gamma-1.0)*(energy-0.5*(mx*mx+my*my)/rho);
  if (!(pressure>0.0)) return false;
  const double q=mx*nx+my*ny, h=energy+pressure;
  flux[0]=q; flux[1]=mx*q/rho+pressure*nx;
  flux[2]=my*q/rho+pressure*ny; flux[3]=h*q/rho;
  const double pr=(gamma-1.0)*0.5*(mx*mx+my*my)/(rho*rho);
  const double pmx=-(gamma-1.0)*mx/rho, pmy=-(gamma-1.0)*my/rho;
  const double pe=gamma-1.0;
  std::fill_n(jacobian,16,0.0); jacobian[1]=nx; jacobian[2]=ny;
  jacobian[4]=-mx*q/(rho*rho)+pr*nx;
  jacobian[5]=q/rho+mx*nx/rho+pmx*nx;
  jacobian[6]=mx*ny/rho+pmy*nx; jacobian[7]=pe*nx;
  jacobian[8]=-my*q/(rho*rho)+pr*ny;
  jacobian[9]=my*nx/rho+pmx*ny;
  jacobian[10]=q/rho+my*ny/rho+pmy*ny; jacobian[11]=pe*ny;
  jacobian[12]=pr*q/rho-h*q/(rho*rho);
  jacobian[13]=pmx*q/rho+h*nx/rho;
  jacobian[14]=pmy*q/rho+h*ny/rho; jacobian[15]=(1.0+pe)*q/rho;
  return true;
}

inline void Manufactured(const double x,const double y,double* u)
{const double sx=std::sin(2*pi*x),sy=std::sin(2*pi*y),cx=std::cos(2*pi*x),cy=std::cos(2*pi*y);const double rho=1+0.1*sx*sy,vx=0.35+0.05*cx*sy,vy=-0.20+0.05*sx*cy,p=1+0.1*cx*cy;u[0]=rho;u[1]=rho*vx;u[2]=rho*vy;u[3]=p/(gamma-1)+0.5*rho*(vx*vx+vy*vy);}

inline bool Source(const double x,const double y,double* source)
{const double sx=std::sin(2*pi*x),sy=std::sin(2*pi*y),cx=std::cos(2*pi*x),cy=std::cos(2*pi*y);double u[4];Manufactured(x,y,u);const double rho=u[0],vx=u[1]/rho,vy=u[2]/rho;const double rx=0.2*pi*cx*sy,ry=0.2*pi*sx*cy,ux=-0.1*pi*sx*sy,uy=0.1*pi*cx*cy,vxx=0.1*pi*cx*cy,vyy=-0.1*pi*sx*sy,px=-0.2*pi*sx*cy,py=-0.2*pi*cx*sy;const double dx[4]={rx,rx*vx+rho*ux,rx*vy+rho*vxx,px/(gamma-1)+0.5*rx*(vx*vx+vy*vy)+rho*(vx*ux+vy*vxx)};const double dy[4]={ry,ry*vx+rho*uy,ry*vy+rho*vyy,py/(gamma-1)+0.5*ry*(vx*vx+vy*vy)+rho*(vx*uy+vy*vyy)};double f[4],jx[16],jy[16];if(!Flux(u,1,0,f,jx)||!Flux(u,0,1,f,jy))return false;for(int r=0;r<4;++r){source[r]=0;for(int c=0;c<4;++c)source[r]+=jx[r*4+c]*dx[c]+jy[r*4+c]*dy[c];}return true;}

inline bool VerifyPhysics(const CudaOraclePod& p)
{
  for(int e=0;e<p.elements;++e)
  {
    std::array<double,interior_dofs_per_element> r{};
    std::array<double,trace_dofs_per_element> g{};
    std::array<double,interior_dofs_per_element*interior_dofs_per_element> a{};
    std::array<double,interior_dofs_per_element*trace_dofs_per_element> b{};
    std::array<double,trace_dofs_per_element*interior_dofs_per_element> c{};
    std::array<double,trace_dofs_per_element*trace_dofs_per_element> d{};
    const int ei=e*16,et=e*32;
    for(int q=0;q<p.volume_points;++q)
    {const int qp=e*p.volume_points+q;double u[4]={},old[4]={};for(int n=0;n<4;++n)for(int k=0;k<4;++k){const double phi=p.volume_shape[qp*4+n];u[k]+=phi*p.current_interior[ei+Nodal(n,4,k)];old[k]+=phi*p.old_interior[ei+Nodal(n,4,k)];}double src[4],fx[4],fy[4],jx[16],jy[16];if(!Source(p.volume_position[qp*2],p.volume_position[qp*2+1],src)||!Flux(u,1,0,fx,jx)||!Flux(u,0,1,fy,jy))return false;const double w=p.volume_weight[qp];for(int rn=0;rn<4;++rn){const double phi=p.volume_shape[qp*4+rn],dx=p.volume_derivative[(qp*4+rn)*2],dy=p.volume_derivative[(qp*4+rn)*2+1];for(int rc=0;rc<4;++rc){const int lr=Nodal(rn,4,rc);r[lr]+=w*(phi*(u[rc]-old[rc])/p.dt-dx*fx[rc]-dy*fy[rc]-phi*src[rc]);for(int cn=0;cn<4;++cn)for(int cc=0;cc<4;++cc){const int lc=Nodal(cn,4,cc);a[lr*16+lc]+=w*((rc==cc?phi*p.volume_shape[qp*4+cn]/p.dt:0)-(dx*jx[rc*4+cc]+dy*jy[rc*4+cc])*p.volume_shape[qp*4+cn]);}}}}
    for(int face=0;face<4;++face)for(int q=0;q<p.face_points;++q)
    {const int qp=(e*4+face)*p.face_points+q,to=face*8;double u[4]={},l[4]={};for(int n=0;n<4;++n)for(int k=0;k<4;++k)u[k]+=p.face_interior_shape[qp*4+n]*p.current_interior[ei+Nodal(n,4,k)];for(int n=0;n<2;++n)for(int k=0;k<4;++k)l[k]+=p.face_trace_shape[qp*2+n]*p.current_local_trace[et+to+Nodal(n,2,k)];double flux[4],jac[16];if(!Flux(l,p.face_normal[qp*2],p.face_normal[qp*2+1],flux,jac))return false;const double w=p.face_weight[qp];for(int rn=0;rn<4;++rn){const double rp=p.face_interior_shape[qp*4+rn];for(int rc=0;rc<4;++rc){const int lr=Nodal(rn,4,rc);r[lr]+=w*rp*(flux[rc]+tau*(u[rc]-l[rc]));for(int cn=0;cn<4;++cn)a[lr*16+Nodal(cn,4,rc)]+=w*rp*tau*p.face_interior_shape[qp*4+cn];for(int cn=0;cn<2;++cn)for(int cc=0;cc<4;++cc)b[lr*32+to+Nodal(cn,2,cc)]+=w*rp*p.face_trace_shape[qp*2+cn]*(jac[rc*4+cc]-(rc==cc?tau:0));}}for(int rn=0;rn<2;++rn){const double rp=p.face_trace_shape[qp*2+rn];for(int rc=0;rc<4;++rc){const int lr=to+Nodal(rn,2,rc);g[lr]+=w*rp*(flux[rc]+tau*(u[rc]-l[rc]));for(int cn=0;cn<4;++cn)c[lr*16+Nodal(cn,4,rc)]+=w*rp*tau*p.face_interior_shape[qp*4+cn];for(int cn=0;cn<2;++cn)for(int cc=0;cc<4;++cc)d[lr*32+to+Nodal(cn,2,cc)]+=w*rp*p.face_trace_shape[qp*2+cn]*(jac[rc*4+cc]-(rc==cc?tau:0));}}}
    for(int i=0;i<16;++i)if(!Close(r[i],p.residual_r[ei+i])||!Close(-r[i],p.f[ei+i]))return false;
    for(int i=0;i<32;++i)if(!Close(g[i],p.residual_g[et+i])||!Close(-g[i],p.g[et+i]))return false;
    const std::size_t ao=static_cast<std::size_t>(e)*256,bo=static_cast<std::size_t>(e)*512,co=static_cast<std::size_t>(e)*512,doff=static_cast<std::size_t>(e)*1024;
    for(int i=0;i<256;++i)if(!Close(a[i],p.a[ao+i]))return false;
    for(int i=0;i<512;++i)if(!Close(b[i],p.b[bo+i])||!Close(c[i],p.c[co+i]))return false;
    for(int i=0;i<1024;++i)if(!Close(d[i],p.d[doff+i]))return false;
  }
  return true;
}

inline bool Solve(std::array<double,16*33>& rhs,std::array<double,256> a)
{for(int k=0;k<16;++k){int pivot=k;for(int r=k+1;r<16;++r)if(std::abs(a[r*16+k])>std::abs(a[pivot*16+k]))pivot=r;if(!(std::abs(a[pivot*16+k])>1e-14))return false;if(pivot!=k){for(int c=0;c<16;++c)std::swap(a[k*16+c],a[pivot*16+c]);for(int c=0;c<33;++c)std::swap(rhs[k*33+c],rhs[pivot*33+c]);}for(int r=k+1;r<16;++r){const double m=a[r*16+k]/a[k*16+k];for(int c=k;c<16;++c)a[r*16+c]-=m*a[k*16+c];for(int c=0;c<33;++c)rhs[r*33+c]-=m*rhs[k*33+c];}}for(int k=15;k>=0;--k)for(int c=0;c<33;++c){for(int j=k+1;j<16;++j)rhs[k*33+c]-=a[k*16+j]*rhs[j*33+c];rhs[k*33+c]/=a[k*16+k];}return true;}

inline bool VerifyCondensationRecovery(const CudaOraclePod& p)
{for(int e=0;e<p.elements;++e){std::array<double,256>a{};std::array<double,16*33>w{};for(int i=0;i<256;++i)a[i]=p.a[e*256+i];for(int r=0;r<16;++r){for(int c=0;c<32;++c)w[r*33+c]=p.b[(e*16+r)*32+c];w[r*33+32]=p.f[e*16+r];}if(!Solve(w,a))return false;for(int r=0;r<32;++r){double rhs=p.g[e*32+r];for(int k=0;k<16;++k)rhs-=p.c[(e*32+r)*16+k]*w[k*33+32];if(!Close(rhs,p.condensed_rhs[e*32+r]))return false;for(int col=0;col<32;++col){double x=p.d[(e*32+r)*32+col];for(int k=0;k<16;++k)x-=p.c[(e*32+r)*16+k]*w[k*33+col];if(!Close(x,p.schur[(e*32+r)*32+col]))return false;}}std::array<double,16*33> recovery_rhs{};for(int r=0;r<16;++r){double x=p.f[e*16+r];for(int c=0;c<32;++c)x-=p.b[(e*16+r)*32+c]*p.local_trace_increment[e*32+c];recovery_rhs[r*33]=x;}if(!Solve(recovery_rhs,a))return false;for(int r=0;r<16;++r)if(!Close(recovery_rhs[r*33],p.recovered_increment[e*16+r]))return false;}return true;}

inline bool VerifySignsAndTrueResidual(const CudaOraclePod& p)
{for(int e=0;e<p.elements;++e)for(int local=0;local<32;++local){const int encoded=p.signed_element_trace[e*32+local];const double sign=encoded>=0?1.0:-1.0;const int dof=encoded>=0?encoded:-1-encoded;if(dof<0||dof>=p.local_trace_count||!Close(p.current_local_trace[e*32+local],sign*p.distributed_local_trace[dof]))return false;}std::array<double,maximum_true_trace_dofs> local_solution{},global_solution{},local_trace{},global_trace{};std::array<int,maximum_true_trace_dofs> local_owner{},global_owner{};for(int i=0;i<p.true_trace_count;++i){const auto global=p.owned_global_true[i];if(global<0||global>=p.global_true_trace_count)return false;local_solution[global]=p.true_solution[i];local_trace[global]=p.current_true_trace[i];local_owner[global]=1;}if(MPI_Allreduce(local_solution.data(),global_solution.data(),p.global_true_trace_count,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD)!=MPI_SUCCESS||MPI_Allreduce(local_trace.data(),global_trace.data(),p.global_true_trace_count,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD)!=MPI_SUCCESS||MPI_Allreduce(local_owner.data(),global_owner.data(),p.global_true_trace_count,MPI_INT,MPI_SUM,MPI_COMM_WORLD)!=MPI_SUCCESS)return false;for(int i=0;i<p.global_true_trace_count;++i)if(global_owner[i]!=1)return false;for(int local=0;local<p.local_trace_count;++local){const auto global=p.local_to_global_true[local];if(global<0||global>=p.global_true_trace_count||!Close(p.distributed_local_trace[local],global_trace[global]))return false;}double residual2=0,rhs2=0;for(int row=0;row<p.true_csr_rows;++row){double value=-p.true_rhs[row];for(int entry=p.true_csr_offsets[row];entry<p.true_csr_offsets[row+1];++entry){const auto global=p.true_csr_columns[entry];if(global<0||global>=p.global_true_trace_count)return false;value+=p.true_csr_values[entry]*global_solution[global];}residual2+=value*value;rhs2+=p.true_rhs[row]*p.true_rhs[row];}double local_norms[2]={residual2,rhs2},global_norms[2]={};if(MPI_Allreduce(local_norms,global_norms,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD)!=MPI_SUCCESS)return false;return std::sqrt(global_norms[0])/std::max(std::sqrt(global_norms[1]),1e-300)<=1e-9;}

inline bool Verify(const CudaOraclePod& p)
{return p.schema==1&&p.dt==fixed_dt&&VerifyPhysics(p)&&VerifyCondensationRecovery(p)&&VerifySignsAndTrueResidual(p);}
} // namespace cuda_w8a_test_oracle

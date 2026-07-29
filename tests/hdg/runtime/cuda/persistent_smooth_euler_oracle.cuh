#pragma once

#include "numerics/hdg/runtime/cuda/persistent_smooth_euler_types.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
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

inline bool AllRanks(const bool local)
{
  const int local_value = local ? 1 : 0;
  int global_value = 0;
  return MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                       MPI_COMM_WORLD) == MPI_SUCCESS && global_value == 1;
}

inline bool ValidShape(const CudaOraclePod& p)
{
  const bool smoke_shape = p.scalar_interior_dofs == 4 &&
      p.scalar_face_dofs == 2 && p.interior_dofs == 16 &&
      p.trace_dofs == 32;
  const bool functional_shape = p.scalar_interior_dofs == 9 &&
      p.scalar_face_dofs == 3 && p.interior_dofs == 36 &&
      p.trace_dofs == 48;
  if (p.schema != 1 || (p.ranks != 1 && p.ranks != 2) || p.rank < 0 ||
      p.rank >= p.ranks || p.elements <= 0 ||
      p.elements > maximum_rank_elements || p.scalar_interior_dofs <= 0 ||
      p.scalar_interior_dofs > maximum_scalar_dofs_per_element ||
      p.scalar_face_dofs <= 0 ||
      p.scalar_face_dofs > maximum_scalar_dofs_per_face ||
      p.interior_dofs != conserved_components * p.scalar_interior_dofs ||
      p.trace_dofs != faces_per_element * conserved_components *
          p.scalar_face_dofs ||
      (!smoke_shape && !functional_shape) ||
      p.interior_dofs > maximum_interior_dofs_per_element ||
      p.trace_dofs > maximum_trace_dofs_per_element ||
      p.interior_count != p.elements * p.interior_dofs ||
      p.local_trace_count <= 0 ||
      p.local_trace_count > maximum_local_trace_dofs ||
      p.true_trace_count <= 0 ||
      p.true_trace_count > maximum_true_trace_dofs ||
      p.global_true_trace_count <= 0 ||
      p.global_true_trace_count > maximum_true_trace_dofs ||
      p.volume_points <= 0 || p.volume_points > maximum_volume_points ||
      p.face_points <= 0 || p.face_points > maximum_face_points ||
      p.true_csr_rows != p.true_trace_count ||
      p.true_csr_nonzeros < 0 ||
      p.true_csr_nonzeros > maximum_true_matrix_entries ||
      p.true_csr_offsets[0] != 0 ||
      p.true_csr_offsets[p.true_csr_rows] != p.true_csr_nonzeros)
  {
    return false;
  }
  for (int row = 0; row < p.true_csr_rows; ++row)
  {
    if (p.true_csr_offsets[row] < 0 ||
        p.true_csr_offsets[row] > p.true_csr_offsets[row + 1] ||
        p.true_csr_offsets[row + 1] > p.true_csr_nonzeros)
    {
      return false;
    }
  }
  return true;
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
    std::array<double,maximum_interior_dofs_per_element> r{};
    std::array<double,maximum_trace_dofs_per_element> g{};
    std::array<double,maximum_interior_dofs_per_element*
                          maximum_interior_dofs_per_element> a{};
    std::array<double,maximum_interior_dofs_per_element*
                          maximum_trace_dofs_per_element> b{};
    std::array<double,maximum_trace_dofs_per_element*
                          maximum_interior_dofs_per_element> c{};
    std::array<double,maximum_trace_dofs_per_element*
                          maximum_trace_dofs_per_element> d{};
    const int si=p.scalar_interior_dofs,sf=p.scalar_face_dofs;
    const int ni=p.interior_dofs,nt=p.trace_dofs;
    const int ei=e*ni,et=e*nt;
    for(int q=0;q<p.volume_points;++q)
    {const int qp=e*p.volume_points+q;double u[4]={},old[4]={};for(int n=0;n<si;++n)for(int k=0;k<4;++k){const double phi=p.volume_shape[qp*si+n];u[k]+=phi*p.current_interior[ei+Nodal(n,si,k)];old[k]+=phi*p.old_interior[ei+Nodal(n,si,k)];}double src[4],fx[4],fy[4],jx[16],jy[16];if(!Source(p.volume_position[qp*2],p.volume_position[qp*2+1],src)||!Flux(u,1,0,fx,jx)||!Flux(u,0,1,fy,jy))return false;const double w=p.volume_weight[qp];for(int rn=0;rn<si;++rn){const double phi=p.volume_shape[qp*si+rn],dx=p.volume_derivative[(qp*si+rn)*2],dy=p.volume_derivative[(qp*si+rn)*2+1];for(int rc=0;rc<4;++rc){const int lr=Nodal(rn,si,rc);r[lr]+=w*(phi*(u[rc]-old[rc])/p.dt-dx*fx[rc]-dy*fy[rc]-phi*src[rc]);for(int cn=0;cn<si;++cn)for(int cc=0;cc<4;++cc){const int lc=Nodal(cn,si,cc);a[lr*ni+lc]+=w*((rc==cc?phi*p.volume_shape[qp*si+cn]/p.dt:0)-(dx*jx[rc*4+cc]+dy*jy[rc*4+cc])*p.volume_shape[qp*si+cn]);}}}}
    for(int face=0;face<4;++face)for(int q=0;q<p.face_points;++q)
    {const int qp=(e*4+face)*p.face_points+q,to=face*4*sf;double u[4]={},l[4]={};for(int n=0;n<si;++n)for(int k=0;k<4;++k)u[k]+=p.face_interior_shape[qp*si+n]*p.current_interior[ei+Nodal(n,si,k)];for(int n=0;n<sf;++n)for(int k=0;k<4;++k)l[k]+=p.face_trace_shape[qp*sf+n]*p.current_local_trace[et+to+Nodal(n,sf,k)];double flux[4],jac[16];if(!Flux(l,p.face_normal[qp*2],p.face_normal[qp*2+1],flux,jac))return false;const double w=p.face_weight[qp];for(int rn=0;rn<si;++rn){const double rp=p.face_interior_shape[qp*si+rn];for(int rc=0;rc<4;++rc){const int lr=Nodal(rn,si,rc);r[lr]+=w*rp*(flux[rc]+tau*(u[rc]-l[rc]));for(int cn=0;cn<si;++cn)a[lr*ni+Nodal(cn,si,rc)]+=w*rp*tau*p.face_interior_shape[qp*si+cn];for(int cn=0;cn<sf;++cn)for(int cc=0;cc<4;++cc)b[lr*nt+to+Nodal(cn,sf,cc)]+=w*rp*p.face_trace_shape[qp*sf+cn]*(jac[rc*4+cc]-(rc==cc?tau:0));}}for(int rn=0;rn<sf;++rn){const double rp=p.face_trace_shape[qp*sf+rn];for(int rc=0;rc<4;++rc){const int lr=to+Nodal(rn,sf,rc);g[lr]+=w*rp*(flux[rc]+tau*(u[rc]-l[rc]));for(int cn=0;cn<si;++cn)c[lr*ni+Nodal(cn,si,rc)]+=w*rp*tau*p.face_interior_shape[qp*si+cn];for(int cn=0;cn<sf;++cn)for(int cc=0;cc<4;++cc)d[lr*nt+to+Nodal(cn,sf,cc)]+=w*rp*p.face_trace_shape[qp*sf+cn]*(jac[rc*4+cc]-(rc==cc?tau:0));}}}
    for(int i=0;i<ni;++i)if(!Close(r[i],p.residual_r[ei+i])||!Close(-r[i],p.f[ei+i]))return false;
    for(int i=0;i<nt;++i)if(!Close(g[i],p.residual_g[et+i])||!Close(-g[i],p.g[et+i]))return false;
    const std::size_t ao=static_cast<std::size_t>(e)*ni*ni,bo=static_cast<std::size_t>(e)*ni*nt,co=static_cast<std::size_t>(e)*nt*ni,doff=static_cast<std::size_t>(e)*nt*nt;
    for(int i=0;i<ni*ni;++i)if(!Close(a[i],p.a[ao+i]))return false;
    for(int i=0;i<ni*nt;++i)if(!Close(b[i],p.b[bo+i]))return false;
    for(int i=0;i<nt*ni;++i)if(!Close(c[i],p.c[co+i]))return false;
    for(int i=0;i<nt*nt;++i)if(!Close(d[i],p.d[doff+i]))return false;
  }
  return true;
}

inline bool Solve(std::array<double,maximum_interior_dofs_per_element*
                                      (maximum_trace_dofs_per_element+1)>& rhs,
                  std::array<double,maximum_interior_dofs_per_element*
                                      maximum_interior_dofs_per_element> a,
                  const int n,const int columns)
{for(int k=0;k<n;++k){int pivot=k;for(int r=k+1;r<n;++r)if(std::abs(a[r*n+k])>std::abs(a[pivot*n+k]))pivot=r;if(!(std::abs(a[pivot*n+k])>1e-14))return false;if(pivot!=k){for(int c=0;c<n;++c)std::swap(a[k*n+c],a[pivot*n+c]);for(int c=0;c<columns;++c)std::swap(rhs[k*columns+c],rhs[pivot*columns+c]);}for(int r=k+1;r<n;++r){const double m=a[r*n+k]/a[k*n+k];for(int c=k;c<n;++c)a[r*n+c]-=m*a[k*n+c];for(int c=0;c<columns;++c)rhs[r*columns+c]-=m*rhs[k*columns+c];}}for(int k=n-1;k>=0;--k)for(int c=0;c<columns;++c){for(int j=k+1;j<n;++j)rhs[k*columns+c]-=a[k*n+j]*rhs[j*columns+c];rhs[k*columns+c]/=a[k*n+k];}return true;}

inline bool VerifyCondensationRecovery(const CudaOraclePod& p)
{const int ni=p.interior_dofs,nt=p.trace_dofs,columns=nt+1;for(int e=0;e<p.elements;++e){std::array<double,maximum_interior_dofs_per_element*maximum_interior_dofs_per_element>a{};std::array<double,maximum_interior_dofs_per_element*(maximum_trace_dofs_per_element+1)>w{};for(int i=0;i<ni*ni;++i)a[i]=p.a[e*ni*ni+i];for(int r=0;r<ni;++r){for(int c=0;c<nt;++c)w[r*columns+c]=p.b[(e*ni+r)*nt+c];w[r*columns+nt]=p.f[e*ni+r];}if(!Solve(w,a,ni,columns))return false;for(int r=0;r<nt;++r){double rhs=p.g[e*nt+r];for(int k=0;k<ni;++k)rhs-=p.c[(e*nt+r)*ni+k]*w[k*columns+nt];if(!Close(rhs,p.condensed_rhs[e*nt+r]))return false;for(int col=0;col<nt;++col){double x=p.d[(e*nt+r)*nt+col];for(int k=0;k<ni;++k)x-=p.c[(e*nt+r)*ni+k]*w[k*columns+col];if(!Close(x,p.schur[(e*nt+r)*nt+col]))return false;}}std::array<double,maximum_interior_dofs_per_element*(maximum_trace_dofs_per_element+1)>recovery_rhs{};for(int r=0;r<ni;++r){double x=p.f[e*ni+r];for(int c=0;c<nt;++c)x-=p.b[(e*ni+r)*nt+c]*p.local_trace_increment[e*nt+c];recovery_rhs[r]=x;}if(!Solve(recovery_rhs,a,ni,1))return false;for(int r=0;r<ni;++r)if(!Close(recovery_rhs[r],p.recovered_increment[e*ni+r]))return false;}return true;}

inline bool SolveTrueDofs(
    const int count,
    std::vector<double>& matrix,
    std::vector<double>& rhs)
{
  for (int column = 0; column < count; ++column)
  {
    int pivot = column;
    for (int row = column + 1; row < count; ++row)
      if (std::abs(matrix[row * count + column]) >
          std::abs(matrix[pivot * count + column]))
        pivot = row;
    if (!(std::abs(matrix[pivot * count + column]) > 1.0e-14))
      return false;
    if (pivot != column)
    {
      for (int entry = 0; entry < count; ++entry)
        std::swap(matrix[column * count + entry],
                  matrix[pivot * count + entry]);
      std::swap(rhs[column], rhs[pivot]);
    }
    for (int row = column + 1; row < count; ++row)
    {
      const double multiplier = matrix[row * count + column] /
          matrix[column * count + column];
      for (int entry = column; entry < count; ++entry)
        matrix[row * count + entry] -=
            multiplier * matrix[column * count + entry];
      rhs[row] -= multiplier * rhs[column];
    }
  }
  for (int row = count - 1; row >= 0; --row)
  {
    for (int column = row + 1; column < count; ++column)
      rhs[row] -= matrix[row * count + column] * rhs[column];
    rhs[row] /= matrix[row * count + row];
  }
  return true;
}

inline bool VerifySignsAndTrueResidual(const CudaOraclePod& p)
{
  const int count = p.global_true_trace_count;
  std::vector<double> local_matrix(static_cast<std::size_t>(count) * count);
  std::vector<double> global_matrix(static_cast<std::size_t>(count) * count);
  std::vector<double> local_rhs(static_cast<std::size_t>(count));
  std::vector<double> global_rhs(static_cast<std::size_t>(count));
  std::vector<double> local_solution(static_cast<std::size_t>(count));
  std::vector<double> global_solution(static_cast<std::size_t>(count));
  std::vector<double> local_trace(static_cast<std::size_t>(count));
  std::vector<double> global_trace(static_cast<std::size_t>(count));
  std::vector<int> local_owner(static_cast<std::size_t>(count));
  std::vector<int> global_owner(static_cast<std::size_t>(count));

  const bool local_assembly_valid = [&]()
  {
    for (int element = 0; element < p.elements; ++element)
    {
      for (int row = 0; row < p.trace_dofs; ++row)
      {
        const int encoded_row =
            p.signed_element_trace[element * p.trace_dofs + row];
        if (encoded_row == std::numeric_limits<int>::min()) { return false; }
        const double row_sign = encoded_row >= 0 ? 1.0 : -1.0;
        const int local_row = encoded_row >= 0
            ? encoded_row : -(encoded_row + 1);
        if (local_row < 0 || local_row >= p.local_trace_count)
          return false;
        const std::int64_t global_row = p.local_to_global_true[local_row];
        if (global_row < 0 || global_row >= count ||
            !Close(p.current_local_trace[
                       element * p.trace_dofs + row],
                   row_sign * p.distributed_local_trace[local_row]))
          return false;
        local_rhs[static_cast<std::size_t>(global_row)] +=
            row_sign * p.condensed_rhs[
                element * p.trace_dofs + row];
        for (int column = 0; column < p.trace_dofs; ++column)
        {
          const int encoded_column = p.signed_element_trace[
              element * p.trace_dofs + column];
          if (encoded_column == std::numeric_limits<int>::min())
            return false;
          const double column_sign = encoded_column >= 0 ? 1.0 : -1.0;
          const int local_column = encoded_column >= 0
              ? encoded_column : -(encoded_column + 1);
          if (local_column < 0 || local_column >= p.local_trace_count)
            return false;
          const std::int64_t global_column =
              p.local_to_global_true[local_column];
          if (global_column < 0 || global_column >= count) { return false; }
          local_matrix[static_cast<std::size_t>(global_row) * count +
                       static_cast<std::size_t>(global_column)] +=
              row_sign * column_sign * p.schur[
                  (element * p.trace_dofs + row) * p.trace_dofs + column];
        }
      }
    }
    for (int local = 0; local < p.true_trace_count; ++local)
    {
      const std::int64_t global = p.owned_global_true[local];
      if (global < 0 || global >= count) { return false; }
      local_solution[static_cast<std::size_t>(global)] = p.true_solution[local];
      local_trace[static_cast<std::size_t>(global)] =
          p.current_true_trace[local];
      ++local_owner[static_cast<std::size_t>(global)];
    }
    return true;
  }();
  if (!AllRanks(local_assembly_valid)) { return false; }

  const bool matrix_reduced = MPI_Allreduce(
      local_matrix.data(), global_matrix.data(), count * count, MPI_DOUBLE,
      MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool rhs_reduced = MPI_Allreduce(
      local_rhs.data(), global_rhs.data(), count, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool solution_reduced = MPI_Allreduce(
      local_solution.data(), global_solution.data(), count, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool trace_reduced = MPI_Allreduce(
      local_trace.data(), global_trace.data(), count, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool owner_reduced = MPI_Allreduce(
      local_owner.data(), global_owner.data(), count, MPI_INT, MPI_SUM,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  if (!AllRanks(matrix_reduced && rhs_reduced && solution_reduced &&
                trace_reduced && owner_reduced))
    return false;

  const bool local_result = [&]()
  {
    for (int global = 0; global < count; ++global)
      if (global_owner[global] != 1) { return false; }
    for (int local = 0; local < p.local_trace_count; ++local)
    {
      const std::int64_t global = p.local_to_global_true[local];
      if (global < 0 || global >= count ||
          !Close(p.distributed_local_trace[local],
                 global_trace[static_cast<std::size_t>(global)]))
        return false;
    }

    std::vector<double> production_row(static_cast<std::size_t>(count));
    for (int local_row = 0; local_row < p.true_csr_rows; ++local_row)
    {
      std::fill(production_row.begin(), production_row.end(), 0.0);
      const std::int64_t global_row = p.owned_global_true[local_row];
      if (global_row < 0 || global_row >= count ||
          !Close(p.true_rhs[local_row],
                 global_rhs[static_cast<std::size_t>(global_row)]))
        return false;
      for (int entry = p.true_csr_offsets[local_row];
           entry < p.true_csr_offsets[local_row + 1]; ++entry)
      {
        const std::int64_t global_column = p.true_csr_columns[entry];
        if (global_column < 0 || global_column >= count) { return false; }
        production_row[static_cast<std::size_t>(global_column)] +=
            p.true_csr_values[entry];
      }
      for (int global_column = 0; global_column < count; ++global_column)
        if (!Close(production_row[global_column],
                   global_matrix[static_cast<std::size_t>(global_row) * count +
                                 global_column]))
          return false;
    }

    auto oracle_matrix = global_matrix;
    auto oracle_solution = global_rhs;
    if (!SolveTrueDofs(count, oracle_matrix, oracle_solution)) { return false; }
    double residual_squared = 0.0;
    double rhs_squared = 0.0;
    for (int row = 0; row < count; ++row)
    {
      if (!Close(oracle_solution[row], global_solution[row], 10.0))
        return false;
      double residual = -global_rhs[row];
      for (int column = 0; column < count; ++column)
        residual += global_matrix[row * count + column] *
            global_solution[column];
      residual_squared += residual * residual;
      rhs_squared += global_rhs[row] * global_rhs[row];
    }
    return std::sqrt(residual_squared) /
        std::max(std::sqrt(rhs_squared), 1.0e-300) <= 1.0e-9;
  }();
  return AllRanks(local_result);
}

inline bool Verify(const CudaOraclePod& p)
{
  int communicator_size = 0;
  const bool size_ready =
      MPI_Comm_size(MPI_COMM_WORLD, &communicator_size) == MPI_SUCCESS;
  int count_min = p.global_true_trace_count;
  int count_max = p.global_true_trace_count;
  const bool count_min_ready = MPI_Allreduce(
      &p.global_true_trace_count, &count_min, 1, MPI_INT, MPI_MIN,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool count_max_ready = MPI_Allreduce(
      &p.global_true_trace_count, &count_max, 1, MPI_INT, MPI_MAX,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  std::uint64_t identity[2] = {p.attempt_sequence, p.product_generation};
  std::uint64_t identity_min[2] = {};
  std::uint64_t identity_max[2] = {};
  const bool identity_min_ready = MPI_Allreduce(
      identity, identity_min, 2, MPI_UINT64_T, MPI_MIN,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool identity_max_ready = MPI_Allreduce(
      identity, identity_max, 2, MPI_UINT64_T, MPI_MAX,
      MPI_COMM_WORLD) == MPI_SUCCESS;
  const bool identity_consistent = size_ready && count_min_ready &&
      count_max_ready && identity_min_ready && identity_max_ready &&
      ValidShape(p) && p.ranks == communicator_size && p.dt == fixed_dt &&
      count_min == count_max && identity_min[0] == identity_max[0] &&
      identity_min[1] == identity_max[1];
  if (!AllRanks(identity_consistent)) { return false; }
  if (!AllRanks(VerifyPhysics(p))) { return false; }
  if (!AllRanks(VerifyCondensationRecovery(p))) { return false; }
  return VerifySignsAndTrueResidual(p);
}
} // namespace cuda_w8a_test_oracle

#pragma once

// Internal Cartesian algebra for the global shared-reference inverse.
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace igrlab { namespace igr_internal {

struct TensorSpectralTopology
{
  int nx=0,ny=0;
  std::vector<int> elements; // Cartesian cell x+nx*y -> actual element.
};

inline TensorSpectralTopology BuildTensorSpectralTopology(
    const std::vector<std::array<int,5>>& neighbors)
{
  if(neighbors.empty() || neighbors.size()>std::size_t(INT_MAX))
    throw std::invalid_argument("Spectral tensor topology must be nonempty and bounded.");
  const int ne=static_cast<int>(neighbors.size());
  for(int e=0;e<ne;++e)
  {
    if(neighbors[e][0]!=e)throw std::invalid_argument("Invalid spectral self slot.");
    for(int next:neighbors[e])if(next<0 || next>=ne)
      throw std::invalid_argument("Invalid spectral neighbor index.");
  }
  std::vector<char> seen(ne,0);
  auto cycle=[&](int slot)
  {
    std::fill(seen.begin(),seen.end(),0);int e=0,length=0;
    do
    {
      if(seen[e])throw std::invalid_argument("Spectral translation does not close at its origin.");
      seen[e]=1;++length;e=neighbors[e][slot];
    }while(e!=0);
    return length;
  };
  TensorSpectralTopology result;result.nx=cycle(2);result.ny=cycle(4);
  const int nx=result.nx,ny=result.ny;
  if(nx<4 || ny<4 || std::size_t(nx)*ny!=neighbors.size())
    throw std::invalid_argument("Spectral tensor topology requires a periodic Cartesian product with nx,ny>=4.");
  result.elements.resize(ne);std::fill(seen.begin(),seen.end(),0);int origin=0;
  for(int y=0;y<ny;++y)
  {
    int e=origin;
    for(int x=0;x<nx;++x)
    {
      if(seen[e])throw std::invalid_argument("Spectral Cartesian product repeats an element.");
      seen[e]=1;result.elements[x+nx*y]=e;e=neighbors[e][2];
    }
    if(e!=origin)throw std::invalid_argument("Spectral Cartesian row does not close.");
    origin=neighbors[origin][4];
  }
  auto element=[&](int x,int y){return result.elements[(x+nx)%nx+nx*((y+ny)%ny)];};
  for(int y=0;y<ny;++y)for(int x=0;x<nx;++x)
  {
    const auto& row=neighbors[element(x,y)];
    if(row[1]!=element(x-1,y) || row[2]!=element(x+1,y) ||
        row[3]!=element(x,y-1) || row[4]!=element(x,y+1))
      throw std::invalid_argument("Spectral Cartesian directions are inconsistent.");
  }
  return result;
}

inline bool TensorSpectralLayoutValid(int nx,int ny)
{
  if(nx<4 || ny<4)return false;
  const std::size_t x=6*std::size_t(nx),y=6*std::size_t(ny),bound=INT_MAX-255;
  return x<=32768 && y<=32768 && x<=bound && y<=bound &&
         x<=bound/x && y<=bound/y && x<=bound/y;
}

struct TensorSpectralReference
{
  // Three column-major d-by-d blocks: center, negative, positive direction.
  std::vector<double> x,y;
  double mean=0,mass=0,relative_defect=0;
};

inline TensorSpectralReference BuildTensorSpectralReference(
    const std::vector<double>& a,int d=6,double tolerance=1e-8)
{
  if(d!=6 || a.size()!=std::size_t(5)*d*d*d*d || !(tolerance>0) || !std::isfinite(tolerance))
    throw std::invalid_argument("Invalid spectral reference dimensions/tolerance.");
  const int n=d*d;double largest=0;
  for(double value:a)
  {
    if(!std::isfinite(value))throw std::invalid_argument("Nonfinite spectral reference.");
    largest=std::max(largest,std::abs(value));
  }
  if(!(largest>0))throw std::invalid_argument("Empty spectral reference.");
  TensorSpectralReference result;result.x.assign(3*d*d,0);result.y.assign(3*d*d,0);
  for(int slot=0;slot<3;++slot)for(int j=0;j<d;++j)for(int i=0;i<d;++i)
  {
    for(int k=0;k<d;++k)
    {
      result.x[(slot*d+j)*d+i]+=a[((slot*n+j+d*k)*n)+i+d*k]/d;
      const int yslot=slot==0?0:slot+2;
      result.y[(slot*d+j)*d+i]+=a[((yslot*n+k+d*j)*n)+k+d*i]/d;
    }
  }
  for(int i=0;i<n;++i)result.mean+=a[i+n*i]/n;
  for(int slot=0;slot<5;++slot)result.mass+=a[slot*n*n];
  if(!(result.mass>0) || !std::isfinite(result.mass))
    throw std::invalid_argument("Spectral reference requires positive screened constant mode.");
  double error=0,scale=0;
  for(int slot=0;slot<5;++slot)for(int j=0;j<n;++j)for(int i=0;i<n;++i)
  {
    const int ix=i%d,iy=i/d,jx=j%d,jy=j/d;double predicted=0;
    if(slot==0)predicted=(iy==jy?result.x[ix+d*jx]:0)+
        (ix==jx?result.y[iy+d*jy]:0)-(i==j?result.mean:0);
    else if(slot<3)predicted=iy==jy?result.x[(slot*d+jx)*d+ix]:0;
    else predicted=ix==jx?result.y[((slot-2)*d+jy)*d+iy]:0;
    const double value=a[(slot*n+j)*n+i]/largest,delta=predicted/largest-value;
    error+=delta*delta;scale+=value*value;
  }
  result.relative_defect=std::sqrt(error/scale);
  if(!std::isfinite(result.relative_defect) || result.relative_defect>tolerance)
    throw std::invalid_argument("Averaged spectral stencil is not sufficiently separable.");
  return result;
}

} }

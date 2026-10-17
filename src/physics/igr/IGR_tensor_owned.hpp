#pragma once

// Internal algebra and topology for the single 4x4-owned / 6x6-input smoother.
// This header deliberately has no MFEM or CUDA dependency.
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace igrlab { namespace igr_internal {

inline bool TensorOwnedLayoutValid(int tiles,int ne,int d)
{
  if(tiles<=0 || ne<=0 || (d!=4 && d!=6) ||
      std::size_t(ne)!=16*std::size_t(tiles))return false;
  return std::size_t(tiles)<=std::size_t(INT_MAX-255)/(36*d*d) &&
         std::size_t(ne)<=std::size_t(INT_MAX-255)/(d*d);
}

struct TensorOwnedPatchMap
{
  int nx=0,ny=0;
  // Cell indices, x fastest within each 6x6 input or 4x4 owned tile.
  // Tile anchors have Cartesian coordinates (4*i,4*j), x fastest.
  std::vector<int> input,owned;
};

inline TensorOwnedPatchMap BuildTensorOwnedPatchMap(
    const std::vector<std::array<int,5>>& neighbors)
{
  if(neighbors.empty() || neighbors.size()>std::size_t(INT_MAX))
    throw std::invalid_argument("Owned tensor patches require bounded nonempty topology.");
  const int ne=static_cast<int>(neighbors.size());
  for(int e=0;e<ne;++e)
  {
    if(neighbors[e][0]!=e)
      throw std::invalid_argument("Owned tensor patch self slot is invalid.");
    for(int next:neighbors[e])if(next<0 || next>=ne)
      throw std::invalid_argument("Owned tensor patch neighbor is out of range.");
  }
  std::vector<char> visited(ne,0);
  auto cycle=[&](int slot)
  {
    std::fill(visited.begin(),visited.end(),0);int length=0,e=0;
    do
    {
      if(visited[e])throw std::invalid_argument("Owned tensor translation is not a cycle.");
      visited[e]=1;++length;e=neighbors[e][slot];
    }while(e!=0);
    return length;
  };
  TensorOwnedPatchMap result;result.nx=cycle(2);result.ny=cycle(4);
  const int nx=result.nx,ny=result.ny;
  if(nx<8 || ny<8 || nx%4 || ny%4 || std::size_t(nx)*ny!=neighbors.size())
    throw std::invalid_argument("Owned tensor patches require a Cartesian product with nx,ny divisible by four and at least eight.");
  std::vector<int> grid(ne);std::fill(visited.begin(),visited.end(),0);
  int origin=0;
  for(int y=0;y<ny;++y)
  {
    int e=origin;
    for(int x=0;x<nx;++x)
    {
      if(visited[e])throw std::invalid_argument("Owned tensor Cartesian product repeats an element.");
      visited[e]=1;grid[x+nx*y]=e;e=neighbors[e][2];
    }
    if(e!=origin)throw std::invalid_argument("Owned tensor Cartesian row does not close.");
    origin=neighbors[origin][4];
  }
  auto element=[&](int x,int y){return grid[(x+nx)%nx+nx*((y+ny)%ny)];};
  for(int y=0;y<ny;++y)for(int x=0;x<nx;++x)
  {
    const auto& row=neighbors[element(x,y)];
    if(row[1]!=element(x-1,y) || row[2]!=element(x+1,y) ||
        row[3]!=element(x,y-1) || row[4]!=element(x,y+1))
      throw std::invalid_argument("Owned tensor topology is not a consistent periodic Cartesian product.");
  }
  const int tiles=ne/16;
  if(std::size_t(tiles)>std::size_t(INT_MAX)/36)
    throw std::invalid_argument("Owned tensor patch indices exceed integer storage.");
  result.input.resize(std::size_t(tiles)*36);result.owned.resize(ne);
  std::fill(visited.begin(),visited.end(),0);
  for(int y=0;y<ny;y+=4)for(int x=0;x<nx;x+=4)
  {
    const int tile=x/4+(nx/4)*(y/4);
    for(int cy=0;cy<6;++cy)for(int cx=0;cx<6;++cx)
      result.input[36*tile+cx+6*cy]=element(x+cx-1,y+cy-1);
    for(int cy=0;cy<4;++cy)for(int cx=0;cx<4;++cx)
    {
      const int e=element(x+cx,y+cy);
      if(visited[e])throw std::invalid_argument("Owned tensor output has multiple owners.");
      visited[e]=1;result.owned[16*tile+cx+4*cy]=e;
    }
  }
  if(std::find(visited.begin(),visited.end(),0)!=visited.end())
    throw std::invalid_argument("Owned tensor outputs do not cover the mesh.");
  return result;
}

// The factors are column-major s-by-s, s=6*d. inverse is indexed (y,x),
// matching the staged separable action. Return a column-major M-by-K map,
// M=(4*d)^2, K=(6*d)^2: rows a+4*d*b are owned tensor coordinates and columns
// u+6*d*v are full-patch tensor coordinates. The owned rows start at (d,d).
inline std::vector<double> ComposeTensorOwnedResponse(int d,
    const std::vector<double>& forward_x,const std::vector<double>& forward_y,
    const std::vector<double>& back_x,const std::vector<double>& back_y,
    const std::vector<double>& inverse)
{
  if(d!=4 && d!=6)throw std::invalid_argument("Owned tensor response requires d=4 or d=6.");
  const int s=6*d,t=4*d,m=t*t,k=s*s;
  for(const auto* a:{&forward_x,&forward_y,&back_x,&back_y,&inverse})
  {
    if(a->size()!=std::size_t(k))throw std::invalid_argument("Owned tensor response factor size mismatch.");
    for(double value:*a)if(!std::isfinite(value))
      throw std::invalid_argument("Nonfinite owned tensor response factor.");
  }
  for(double value:inverse)if(value<=0)
    throw std::invalid_argument("Owned tensor response requires a positive clipped inverse.");
  // Cache the y contraction once per (owned y, x eigen-index, input y).
  // Recomputing it for every owned/input x would cost almost a billion terms
  // at p5. This factorization has about 28 million terms instead.
  std::vector<double> y(std::size_t(t)*s*s),response(std::size_t(m)*k),sum(t);
  for(int v=0;v<s;++v)for(int i=0;i<s;++i)for(int b=0;b<t;++b)
  {
    double value=0;
    for(int j=0;j<s;++j)value+=back_y[d+b+s*j]*inverse[j+s*i]*forward_y[j+s*v];
    if(!std::isfinite(value))throw std::overflow_error("Owned tensor y composition overflow.");
    y[b+t*(i+s*v)]=value;
  }
  for(int v=0;v<s;++v)for(int u=0;u<s;++u)for(int b=0;b<t;++b)
  {
    std::fill(sum.begin(),sum.end(),0.0);
    for(int i=0;i<s;++i)
    {
      const double weight=forward_x[i+s*u]*y[b+t*(i+s*v)];
      for(int a=0;a<t;++a)sum[a]+=back_x[d+a+s*i]*weight;
    }
    for(int a=0;a<t;++a)
    {
      if(!std::isfinite(sum[a]))throw std::overflow_error("Owned tensor response composition overflow.");
      response[a+t*b+m*(u+s*v)]=sum[a];
    }
  }
  return response;
}

// packed is K-by-tiles and response is M-by-K as above. owned contains sixteen
// distinct element indices per tile and must cover all ne=16*tiles elements
// exactly once, as BuildTensorOwnedPatchMap guarantees. Each element's modes
// are x-fast. Fuse x += 0.8*factor[element]*(response*packed) directly into the
// unique owned destination. No input may alias x. Direct calls assume typed
// representable data; the surrounding V-cycle supplies power-of-two scaling.
void ApplyTensorOwnedResponse(int tiles,int ne,int d,const int* owned,
    const double* response,const double* packed,const double* factor,double* x);
void ApplyTensorOwnedResponse(int tiles,int ne,int d,const int* owned,
    const float* response,const float* packed,const float* factor,float* x,bool tf32);

} }

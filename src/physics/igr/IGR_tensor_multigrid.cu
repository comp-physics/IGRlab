#include "igrlab/compat.hpp"
#include "IGR_tensor_multigrid.hpp"
#include "IGR_tensor_center.hpp"
#include "IGR_tensor_reference.hpp"
#include "IGR_tensor_owned.hpp"
#include "IGR_tensor_spectral.hpp"
#include "IGR_device_krylov.hpp"
#include "IGR_shared_patch_cuda.cuh" // Keep the shared/MG periodic topology gate.
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <mma.h>
#include <math_constants.h>
#include <climits>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace igrlab { namespace igr_internal { namespace {
using Stencil = std::vector<double>;
using Topology = std::vector<std::array<int, 5>>;

void CUDA(cudaError_t status)
{ MFEM_VERIFY(status == cudaSuccess, "Tensor multigrid CUDA: " << cudaGetErrorString(status)); }

const char* Name(TensorMultigridPrecision p)
{ return p == TensorMultigridPrecision::FP64 ? "fp64" : p == TensorMultigridPrecision::FP32 ? "fp32" :
    p == TensorMultigridPrecision::TF32x3 ? "tf32x3" : "tf32"; }

const char* SmootherName(TensorMultigridSmoother smoother,int owned_tile=1)
{ return owned_tile==4 ? "restricted-overlap-6x6-owned-4x4" :
    smoother==TensorMultigridSmoother::Center ? "restricted-overlap-3x3-center" : "restricted-overlap-3x3-separable"; }
const char* OperatorName(TensorMultigridOperator mode)
{ return mode==TensorMultigridOperator::Spectral ? "spectral" :
    mode==TensorMultigridOperator::Shared ? "shared" : "sampled"; }
const char* ResidualPolicyName(TensorMultigridResidual mode)
{ return mode==TensorMultigridResidual::FP32 ? "fp32" : "auto"; }

std::size_t Entry(int e, int slot, int i, int j, int n)
{ return ((std::size_t(e) * 5 + slot) * n + j) * n + i; }

// Small deterministic symmetric Jacobi eigensystem. Columns of q are
// orthonormal eigenvectors; no dependency on a platform LAPACK ABI.
void Eigen(std::vector<double> a, int n, std::vector<double>& q,
           std::vector<double>& eigenvalues)
{
  q.assign(n*n, 0.0);
  for (int i=0; i<n; ++i) q[i+n*i]=1.0;
  double scale=0.0;
  for (double v:a) { MFEM_VERIFY(std::isfinite(v), "Nonfinite patch eigensystem."); scale=std::max(scale,std::abs(v)); }
  MFEM_VERIFY(scale>0.0, "Empty patch eigensystem.");
  bool converged=false;
  for (int sweep=0; sweep<100; ++sweep)
  {
    double largest=0.0;
    for (int j=1; j<n; ++j) for (int i=0; i<j; ++i) largest=std::max(largest,std::abs(a[i+n*j]));
    if (largest<=8*std::numeric_limits<double>::epsilon()*scale) { converged=true; break; }
    for (int j=1; j<n; ++j) for (int i=0; i<j; ++i)
    {
      const double off=a[i+n*j];
      if (std::abs(off)<=std::numeric_limits<double>::epsilon()*scale) continue;
      const double tau=(a[j+n*j]-a[i+n*i])/(2*off);
      const double t=std::copysign(1.0,tau)/(std::abs(tau)+std::hypot(1.0,tau));
      const double c=1/std::sqrt(1+t*t), s=t*c;
      const double ai=a[i+n*i], aj=a[j+n*j];
      a[i+n*i]=ai-t*off; a[j+n*j]=aj+t*off;
      a[i+n*j]=a[j+n*i]=0.0;
      for (int k=0;k<n;++k) if(k!=i && k!=j)
      {
        const double x=a[k+n*i],y=a[k+n*j];
        a[k+n*i]=a[i+n*k]=c*x-s*y;
        a[k+n*j]=a[j+n*k]=s*x+c*y;
      }
      for (int k=0;k<n;++k)
      {
        const double x=q[k+n*i],y=q[k+n*j];
        q[k+n*i]=c*x-s*y; q[k+n*j]=s*x+c*y;
      }
    }
  }
  MFEM_VERIFY(converged,"Patch symmetric eigensystem did not converge.");
  eigenvalues.resize(n);
  for(int i=0;i<n;++i) eigenvalues[i]=a[i+n*i];
}

double Legendre(int degree, double x)
{
  const double z=2*x-1;
  if(degree==0) return 1;
  double p0=1,p1=z;
  for(int k=2;k<=degree;++k) { const double p=((2*k-1)*z*p1-(k-1)*p0)/k; p0=p1;p1=p; }
  return std::sqrt(2*degree+1.0)*p1;
}

std::vector<int> Colors(const Topology& topology)
{
  const int ne=static_cast<int>(topology.size());
  std::vector<int> colors(ne,-1),used(ne,-1);
  for(int e=0;e<ne;++e)
  {
    // The reciprocal five-point graph's distance-two neighbors are exactly
    // the conflicts in the former per-element sets. Keep setup storage O(ne).
    for(int neighbor:topology[e]) for(int j:topology[neighbor])
      if(colors[j]>=0) used[colors[j]]=e;
    int c=0;while(used[c]==e)++c;colors[e]=c;
  }
  // The coloring is a reconstruction condition, not a heuristic assumption.
  for(const auto& support:topology) for(int i=0;i<5;++i) for(int j=0;j<i;++j)
    MFEM_VERIFY(colors[support[i]]!=colors[support[j]],"Tensor MG stencil coloring collision.");
  return colors;
}

std::vector<int> PatchMap(const Topology& topology)
{
  const int ne=static_cast<int>(topology.size());
  std::vector<int> patches(ne*9);
  for(int e=0;e<ne;++e)
  {
    for(int y=0;y<3;++y) for(int x=0;x<3;++x)
    {
      const int ys=y==0?3:y==2?4:0, xs=x==0?1:x==2?2:0;
      const int a=topology[topology[e][ys]][xs], b=topology[topology[e][xs]][ys];
      MFEM_VERIFY(a==b,"Tensor patches require commuting Cartesian face neighbors.");
      patches[9*e+3*y+x]=a;
    }
    for(int i=0;i<9;++i) for(int j=0;j<i;++j)
      MFEM_VERIFY(patches[9*e+i]!=patches[9*e+j],"Tensor patches require nine distinct elements.");
    // Check the complete induced stencil, including periodic seams.
    for(int cell=0;cell<9;++cell) for(int slot=1;slot<5;++slot)
    {
      const int x=cell%3,y=cell/3;
      const int xx=x+(slot==1?-1:slot==2?1:0), yy=y+(slot==3?-1:slot==4?1:0);
      const int neighbor=topology[patches[9*e+cell]][slot];
      if(xx>=0 && xx<3 && yy>=0 && yy<3)
      { MFEM_VERIFY(neighbor==patches[9*e+3*yy+xx],"Tensor patch induced face orientation mismatch."); }
      else
      { MFEM_VERIFY(std::find(patches.begin()+9*e,patches.begin()+9*e+9,neighbor)==patches.begin()+9*e+9,
                    "Tensor patch wraps into itself; use at least four elements in each direction."); }
    }
  }
  return patches;
}

void CheckTranslationTopology(const Topology& topology)
{
  const int ne=static_cast<int>(topology.size());
  constexpr int opposite[5]={0,2,1,4,3};
  for(int e=0;e<ne;++e) for(int slot=1;slot<5;++slot)
  {
    MFEM_VERIFY(topology[topology[e][slot]][opposite[slot]]==e,
                "Shared tensor reference requires reciprocal Cartesian translations.");
    for(int other=1;other<5;++other)
      MFEM_VERIFY(topology[topology[e][slot]][other]==topology[topology[e][other]][slot],
                  "Shared tensor reference requires commuting Cartesian translations.");
  }
  std::vector<char> seen(ne,0);std::vector<int> connected;connected.reserve(ne);
  seen[0]=1;connected.push_back(0);
  for(std::size_t i=0;i<connected.size();++i) for(int slot=1;slot<5;++slot)
  {
    const int next=topology[connected[i]][slot];
    if(!seen[next]){seen[next]=1;connected.push_back(next);}
  }
  MFEM_VERIFY(connected.size()==topology.size(),
              "Shared tensor reference requires one connected periodic Cartesian mesh.");
}

// Spectral mode alone admits physical faces. Shared patches and MG continue
// through SharedPatchTopology, whose periodic admission is unchanged.
Topology SpectralFaceTopology(mfem::ParFiniteElementSpace& sfes)
{
  auto& mesh=*sfes.GetMesh();const int ne=sfes.GetNE();bool boundary=false;
  for(int face=0;face<mesh.GetNumFaces();++face)
  {int a,b;mesh.GetFaceElements(face,&a,&b);boundary=boundary || b<0;}
  if(!boundary)return SharedPatchTopology(sfes);
  MFEM_VERIFY(ne>=16 && mesh.Dimension()==2 && mesh.SpaceDimension()==2 &&
      sfes.GetFE(0)->GetOrder()==5 && sfes.GetFE(0)->GetDof()==36,
      "Boundary spectral inverse requires uniform p5 Cartesian quads with at least 16 elements.");
  Topology topology(ne);std::vector<std::array<bool,5>> assigned(ne);
  double hx=0,hy=0;
  for(int e=0;e<ne;++e)
  {
    topology[e].fill(-1);topology[e][0]=e;assigned[e].fill(false);
    auto* transform=mesh.GetElementTransformation(e);
    MFEM_VERIFY(transform->Order()==1,"Boundary spectral inverse requires linear geometry.");
    for(const auto point:std::array<std::array<double,2>,5>{{{0,0},{1,0},{0,1},{1,1},{.5,.5}}})
    {
      mfem::IntegrationPoint ip;ip.Set2(point[0],point[1]);transform->SetIntPoint(&ip);
      const auto& jacobian=transform->Jacobian();
      if(e==0 && point[0]==0 && point[1]==0){hx=jacobian(0,0);hy=jacobian(1,1);}
      const double scale=std::max(hx,hy);
      MFEM_VERIFY(hx>0 && hy>0 && std::abs(jacobian(0,0)-hx)<=1e-11*scale &&
          std::abs(jacobian(1,1)-hy)<=1e-11*scale && std::abs(jacobian(0,1))<=1e-11*scale &&
          std::abs(jacobian(1,0))<=1e-11*scale,
          "Boundary spectral inverse requires identical positive axis-aligned affine Jacobians.");
    }
  }
  mfem::IntegrationPoint midpoint;midpoint.Set1w(.5,1.0);
  for(int face=0;face<mesh.GetNumFaces();++face)
  {
    int a,b;mesh.GetFaceElements(face,&a,&b);
    MFEM_VERIFY(a>=0 && a<ne && b<ne && a!=b,"Invalid spectral Cartesian face adjacency.");
    auto* transform=mesh.GetFaceElementTransformations(face);
    MFEM_VERIFY(transform,"Missing spectral Cartesian face transformation.");
    for(int side=0;side<(b<0?1:2);++side)
    {
      mfem::IntegrationPoint ip;
      (side?transform->Loc2:transform->Loc1).Transform(midpoint,ip);
      const int slot=std::abs(ip.x)<1e-12 && std::abs(ip.y-.5)<1e-12?1:
          std::abs(ip.x-1)<1e-12 && std::abs(ip.y-.5)<1e-12?2:
          std::abs(ip.y)<1e-12 && std::abs(ip.x-.5)<1e-12?3:
          std::abs(ip.y-1)<1e-12 && std::abs(ip.x-.5)<1e-12?4:-1;
      const int e=side?b:a;
      MFEM_VERIFY(slot>0 && !assigned[e][slot],"Invalid/duplicate canonical spectral face slot.");
      assigned[e][slot]=true;topology[e][slot]=side?a:b<0?-1:b;
    }
  }
  for(int e=0;e<ne;++e)for(int slot=1;slot<5;++slot)
    MFEM_VERIFY(assigned[e][slot],"Missing spectral Cartesian physical/interior face.");
  return topology;
}

std::vector<int> BoundarySpectralColors(const Topology& topology)
{
  const int ne=static_cast<int>(topology.size());
  std::vector<int> colors(ne,-1),used(ne,-1);
  for(int e=0;e<ne;++e)
  {
    for(int neighbor:topology[e])if(neighbor>=0)for(int j:topology[neighbor])
      if(j>=0 && colors[j]>=0)used[colors[j]]=e;
    int color=0;while(used[color]==e)++color;colors[e]=color;
  }
  for(const auto& support:topology)for(int i=0;i<5;++i)if(support[i]>=0)
    for(int j=0;j<i;++j)if(support[j]>=0)
      MFEM_VERIFY(colors[support[i]]!=colors[support[j]],"Boundary spectral stencil coloring collision.");
  return colors;
}

void SharedModalData(mfem::ParFiniteElementSpace& sfes,
    std::vector<int>& map,std::vector<double>& nodal)
{
  constexpr int d=6,n=36;
  const int ne=sfes.GetNE(),height=sfes.GetVSize();
  map.resize(height);nodal.resize(n*n);
  const auto& reference=sfes.GetFE(0)->GetNodes();
  MFEM_VERIFY(reference.GetNPoints()==n,"Shared tensor reference requires p5 nodal ordering.");
  for(int i=0;i<n;++i)for(int j=0;j<n;++j)
    nodal[i+n*j]=Legendre(j%d,reference.IntPoint(i).x)*Legendre(j/d,reference.IntPoint(i).y);
  std::vector<char> seen(height,0);mfem::Array<int>dofs;
  for(int e=0;e<ne;++e)
  {
    const auto& fe=*sfes.GetFE(e);
    MFEM_VERIFY(fe.GetDof()==n && fe.GetOrder()==5 && fe.GetGeomType()==mfem::Geometry::SQUARE &&
        fe.GetMapType()==mfem::FiniteElement::VALUE && fe.GetNodes().GetNPoints()==n,
        "Shared tensor reference requires uniform p5 nodal VALUE L2 quads.");
    sfes.GetElementDofs(e,dofs);
    for(int i=0;i<n;++i)
    {
      MFEM_VERIFY(dofs[i]>=0 && dofs[i]<height && !seen[dofs[i]],
                  "Shared tensor reference requires unsigned disjoint L2 maps.");
      const auto& ip=fe.GetNodes().IntPoint(i);const auto& expected=reference.IntPoint(i);
      MFEM_VERIFY(ip.x==expected.x && ip.y==expected.y,
                  "Shared tensor reference requires identical nodal ordering.");
      map[e*n+i]=dofs[i];seen[dofs[i]]=1;
    }
  }
}

__global__ void SharedProbeInput(int ne,int color,int column,const int* colors,
    const int* map,const double* nodal,double* x)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*36)return;
  x[map[row]]=colors[row/36]==color?nodal[row%36+36*column]:0.0;
}

__global__ void SharedProbeProject(int ne,int color,int column,const int* colors,
    const int* map,const double* nodal,const double* y,double* modal,
    double* diagonals,int* invalid)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*36)return;
  const int e=row/36,i=row%36;double value=0;
  for(int j=0;j<36;++j)value+=nodal[j+36*i]*y[map[e*36+j]];
  if(!isfinite(value))atomicExch(invalid,1);
  modal[row]=value;
  if(colors[e]==color && i==column)
  {
    if(!(value>0.0))atomicExch(invalid,1);
    diagonals[e]+=value;
    if(i%6<4 && i/6<4)diagonals[ne+e]+=value;
    if(i%6<2 && i/6<2)diagonals[2*ne+e]+=value;
    diagonals[3*ne+e]=fmax(diagonals[3*ne+e],fabs(value));
  }
}

// Each reduction handles one (slot,modal row), for the current modal column.
// A bounded first stage and a fixed reduction tree avoid global atomics and
// make repeated setup deterministic for the same element ordering.
__global__ void SharedProbePartial(int ne,int color,const int* colors,
    const int* neighbors,const double* modal,double* partial)
{
  __shared__ double sum[256];
  const int entry=blockIdx.x,slot=entry/36,i=entry%36;
  double value=0;
  for(int e=blockIdx.y*256+threadIdx.x;e<ne;e+=256*gridDim.y)
    if(colors[neighbors[5*e+slot]]==color)value+=modal[e*36+i];
  sum[threadIdx.x]=value;__syncthreads();
  for(int step=128;step;step/=2)
  {if(threadIdx.x<step)sum[threadIdx.x]+=sum[threadIdx.x+step];__syncthreads();}
  if(threadIdx.x==0)partial[blockIdx.y*180+entry]=sum[0];
}

__global__ void SharedProbeFinish(int ne,int groups,int column,const double* partial,
    double* average,int* invalid)
{
  __shared__ double sum[256];
  const int entry=blockIdx.x;double value=0;
  for(int group=threadIdx.x;group<groups;group+=256)value+=partial[group*180+entry];
  sum[threadIdx.x]=value;__syncthreads();
  for(int step=128;step;step/=2)
  {if(threadIdx.x<step)sum[threadIdx.x]+=sum[threadIdx.x+step];__syncthreads();}
  if(threadIdx.x==0)
  {
    const int index=((entry/36)*36+column)*36+entry%36;
    const double updated=average[index]+sum[0]/ne;
    average[index]=updated;if(!isfinite(updated))atomicExch(invalid,1);
  }
}

struct SharedSnapshotData
{
  Stencil average;
  std::array<std::vector<double>,3> traces;
  double maximum_diagonal=0,symmetry_error=0;
};

SharedSnapshotData SharedSnapshot(const mfem::Operator& op,
    mfem::ParFiniteElementSpace& sfes,const Topology& topology,
    const mfem::Array<int>& neighbors,std::vector<int>& map,
    std::vector<double>& nodal,int& probes,int& color_count)
{
  const int ne=sfes.GetNE(),height=sfes.GetVSize();
  SharedModalData(sfes,map,nodal);
  const auto colors=Colors(topology);color_count=1+*std::max_element(colors.begin(),colors.end());
  mfem::Array<int> device_colors(ne),device_map(height),invalid(1);
  std::copy(colors.begin(),colors.end(),device_colors.HostWrite());
  std::copy(map.begin(),map.end(),device_map.HostWrite());
  mfem::Vector transform(36*36),input(height),output(height),modal(height),diagonals(4*ne);
  std::copy(nodal.begin(),nodal.end(),transform.HostWrite());
  for(auto* v:{&transform,&input,&output,&modal,&diagonals})v->UseDevice(true);
  const int groups=std::min(1024,(ne+255)/256);
  mfem::Vector partial(groups*180),average(5*36*36);partial.UseDevice(true);average.UseDevice(true);
  CUDA(cudaMemsetAsync(average.Write(),0,std::size_t(average.Size())*sizeof(double)));
  CUDA(cudaMemsetAsync(diagonals.Write(),0,std::size_t(diagonals.Size())*sizeof(double)));
  CUDA(cudaMemsetAsync(invalid.Write(),0,sizeof(int)));
  for(int color=0;color<color_count;++color)for(int j=0;j<36;++j)
  {
    SharedProbeInput<<<(height+255)/256,256>>>(ne,color,j,device_colors.Read(),
        device_map.Read(),transform.Read(),input.Write());
    op.Mult(input,output);++probes;
    SharedProbeProject<<<(height+255)/256,256>>>(ne,color,j,device_colors.Read(),
        device_map.Read(),transform.Read(),output.Read(),modal.Write(),diagonals.ReadWrite(),invalid.ReadWrite());
    SharedProbePartial<<<dim3(180,groups),256>>>(ne,color,device_colors.Read(),
        neighbors.Read(),modal.Read(),partial.Write());
    SharedProbeFinish<<<180,256>>>(ne,groups,j,partial.Read(),average.ReadWrite(),invalid.ReadWrite());
  }
  CUDA(cudaGetLastError());
  MFEM_VERIFY(invalid.HostRead()[0]==0,"Shared tensor reference probe has a nonfinite value or nonpositive diagonal.");
  SharedSnapshotData result;const double* a=average.HostRead();
  result.average.assign(a,a+average.Size());
  const double* diagonal=diagonals.HostRead();
  for(int l=0;l<3;++l)result.traces[l].assign(diagonal+l*ne,diagonal+(l+1)*ne);
  for(int e=0;e<ne;++e)result.maximum_diagonal=std::max(result.maximum_diagonal,diagonal[3*ne+e]);
  // This is the group average of P_t^T A P_t over periodic translations.
  // SPD follows from the admitted screened SIPG A, not from positive diagonals.
  // Repair only a bounded roundoff asymmetry of opposite shared face blocks.
  constexpr int opposite[5]={0,2,1,4,3};double error=0,magnitude=0,coefficient_scale=0;
  for(double value:result.average)coefficient_scale=std::max(coefficient_scale,std::abs(value));
  MFEM_VERIFY(coefficient_scale>0 && std::isfinite(coefficient_scale),"Invalid shared tensor reference scale.");
  for(int slot=0;slot<5;++slot)for(int j=0;j<36;++j)for(int i=0;i<36;++i)
  {
    const double x=result.average[(slot*36+j)*36+i]/coefficient_scale;
    const double y=result.average[(opposite[slot]*36+i)*36+j]/coefficient_scale;
    error+=(x-y)*(x-y);magnitude+=x*x;
  }
  result.symmetry_error=std::sqrt(error/std::max(magnitude,1e-300));
  MFEM_VERIFY(result.symmetry_error<1e-10,"Shared tensor reference is not symmetric: "<<result.symmetry_error);
  for(int slot:{0,1,3})for(int j=0;j<36;++j)for(int i=0;i<36;++i)
  {
    if(slot==0 && i>j)continue;
    double& x=result.average[(slot*36+j)*36+i];
    double& y=result.average[(opposite[slot]*36+i)*36+j];
    const double value=.5*x+.5*y;x=value;y=value;
  }
  return result;
}

// Classes have separate short element lists. First-stage groups cover those
// lists without allocating a dense stencil per element or scanning ne nine times.
__global__ void BoundarySpectralProbePartial(int color,const int* colors,
    const int* neighbors,const int* elements,const int* offsets,const int* group_offsets,
    const double* modal,double* partial)
{
  __shared__ double sum[256];
  const int entry=blockIdx.x,slot=entry/36,i=entry%36,group=blockIdx.y;
  int cls=0;while(group>=group_offsets[cls+1])++cls;
  const int groups=group_offsets[cls+1]-group_offsets[cls];double value=0;
  for(int index=offsets[cls]+(group-group_offsets[cls])*256+threadIdx.x;
      index<offsets[cls+1];index+=256*groups)
  {
    const int e=elements[index],source=neighbors[5*e+slot];
    if(source>=0 && colors[source]==color)value+=modal[e*36+i];
  }
  sum[threadIdx.x]=value;__syncthreads();
  for(int step=128;step;step/=2)
  {if(threadIdx.x<step)sum[threadIdx.x]+=sum[threadIdx.x+step];__syncthreads();}
  if(threadIdx.x==0)partial[group*180+entry]=sum[0];
}

__global__ void BoundarySpectralProbeFinish(int column,const int* offsets,
    const int* group_offsets,const double* partial,double* average,int* invalid)
{
  __shared__ double sum[256];
  const int cls=blockIdx.x/180,entry=blockIdx.x%180;double value=0;
  for(int group=group_offsets[cls]+threadIdx.x;group<group_offsets[cls+1];group+=256)
    value+=partial[group*180+entry];
  sum[threadIdx.x]=value;__syncthreads();
  for(int step=128;step;step/=2)
  {if(threadIdx.x<step)sum[threadIdx.x]+=sum[threadIdx.x+step];__syncthreads();}
  if(threadIdx.x==0)
  {
    const int index=((cls*5+entry/36)*36+column)*36+entry%36;
    const double updated=average[index]+sum[0]/(offsets[cls+1]-offsets[cls]);
    average[index]=updated;if(!isfinite(updated))atomicExch(invalid,1);
  }
}

struct BoundarySpectralSnapshotData
{
  std::array<Stencil,9> classes;
  double maximum_diagonal=0,symmetry_error=0;
};

BoundarySpectralSnapshotData BoundarySpectralSnapshot(const mfem::Operator& op,
    mfem::ParFiniteElementSpace& sfes,const Topology& topology,
    const TensorSpectralTopology& coordinates,const mfem::Array<int>& neighbors,
    std::vector<int>& map,std::vector<double>& nodal,int& probes,int& color_count)
{
  const int ne=sfes.GetNE(),height=sfes.GetVSize();SharedModalData(sfes,map,nodal);
  const auto colors=BoundarySpectralColors(topology);
  color_count=1+*std::max_element(colors.begin(),colors.end());
  std::array<std::vector<int>,9> class_elements;
  for(int y=0;y<coordinates.ny;++y)for(int x=0;x<coordinates.nx;++x)
    class_elements[TensorSpectralBoundaryClass(x,y,coordinates.nx,coordinates.ny)].push_back(
        coordinates.elements[x+coordinates.nx*y]);
  mfem::Array<int> device_colors(ne),device_map(height),elements(ne),offsets(10),group_offsets(10),invalid(1);
  std::copy(colors.begin(),colors.end(),device_colors.HostWrite());
  std::copy(map.begin(),map.end(),device_map.HostWrite());
  int* element=elements.HostWrite();int* offset=offsets.HostWrite();int* group_offset=group_offsets.HostWrite();
  offset[0]=group_offset[0]=0;
  for(int cls=0;cls<9;++cls)
  {
    const int count=static_cast<int>(class_elements[cls].size());
    MFEM_VERIFY(count>0,"Boundary spectral Cartesian class is empty.");
    std::copy(class_elements[cls].begin(),class_elements[cls].end(),element+offset[cls]);
    offset[cls+1]=offset[cls]+count;
    group_offset[cls+1]=group_offset[cls]+std::min(1024,(count+255)/256);
  }
  const int groups=group_offset[9];
  mfem::Vector transform(36*36),input(height),output(height),modal(height),diagonals(4*ne);
  std::copy(nodal.begin(),nodal.end(),transform.HostWrite());
  for(auto* vector:{&transform,&input,&output,&modal,&diagonals})vector->UseDevice(true);
  mfem::Vector partial(groups*180),average(9*5*36*36);partial.UseDevice(true);average.UseDevice(true);
  CUDA(cudaMemsetAsync(average.Write(),0,std::size_t(average.Size())*sizeof(double)));
  CUDA(cudaMemsetAsync(diagonals.Write(),0,std::size_t(diagonals.Size())*sizeof(double)));
  CUDA(cudaMemsetAsync(invalid.Write(),0,sizeof(int)));
  for(int color=0;color<color_count;++color)for(int j=0;j<36;++j)
  {
    SharedProbeInput<<<(height+255)/256,256>>>(ne,color,j,device_colors.Read(),
        device_map.Read(),transform.Read(),input.Write());
    op.Mult(input,output);++probes;
    SharedProbeProject<<<(height+255)/256,256>>>(ne,color,j,device_colors.Read(),
        device_map.Read(),transform.Read(),output.Read(),modal.Write(),diagonals.ReadWrite(),invalid.ReadWrite());
    BoundarySpectralProbePartial<<<dim3(180,groups),256>>>(color,device_colors.Read(),neighbors.Read(),
        elements.Read(),offsets.Read(),group_offsets.Read(),modal.Read(),partial.Write());
    BoundarySpectralProbeFinish<<<9*180,256>>>(j,offsets.Read(),group_offsets.Read(),
        partial.Read(),average.ReadWrite(),invalid.ReadWrite());
  }
  CUDA(cudaGetLastError());
  MFEM_VERIFY(invalid.HostRead()[0]==0,"Boundary spectral probe has a nonfinite value or nonpositive diagonal.");
  BoundarySpectralSnapshotData result;const double* a=average.HostRead();
  for(int cls=0;cls<9;++cls)result.classes[cls].assign(a+cls*5*36*36,a+(cls+1)*5*36*36);
  const double* diagonal=diagonals.HostRead();
  for(int e=0;e<ne;++e)result.maximum_diagonal=std::max(result.maximum_diagonal,diagonal[3*ne+e]);
  double coefficient_scale=0,error=0,magnitude=0;
  for(const auto& stencil:result.classes)for(double value:stencil)
    coefficient_scale=std::max(coefficient_scale,std::abs(value));
  MFEM_VERIFY(coefficient_scale>0 && std::isfinite(coefficient_scale),"Invalid boundary spectral reference scale.");
  constexpr int opposite[5]={0,2,1,4,3};
  for(int cls=0;cls<9;++cls)for(int slot=0;slot<(cls==0?5:1);++slot)
    for(int j=0;j<36;++j)for(int i=0;i<36;++i)
    {
      const double x=result.classes[cls][(slot*36+j)*36+i]/coefficient_scale;
      const double y=result.classes[cls][(opposite[slot]*36+i)*36+j]/coefficient_scale;
      error+=(x-y)*(x-y);magnitude+=x*x;
    }
  result.symmetry_error=std::sqrt(error/std::max(magnitude,1e-300));
  MFEM_VERIFY(result.symmetry_error<1e-10,"Boundary spectral reference is not symmetric: "<<result.symmetry_error);
  // Repair only admitted roundoff. Class separability and the complete open
  // eigensystems are checked afterwards; averaging itself does not prove SPD.
  for(int cls=0;cls<9;++cls)for(int slot:{0,1,3})
  {
    if(cls!=0 && slot!=0)continue;
    for(int j=0;j<36;++j)for(int i=0;i<36;++i)
    {
      if(slot==0 && i>j)continue;
      double& x=result.classes[cls][(slot*36+j)*36+i];
      double& y=result.classes[cls][(opposite[slot]*36+i)*36+j];
      const double value=.5*x+.5*y;x=value;y=value;
    }
  }
  return result;
}

// Exact modal Galerkin snapshot, with every exterior-face diagonal retained.
Stencil Snapshot(const mfem::Operator& op, mfem::ParFiniteElementSpace& sfes,
                 const Topology& topology, std::vector<int>& map,
                 std::vector<double>& nodal, int& probes, int& color_count)
{
  constexpr int d=6,n=36;
  const int ne=sfes.GetNE(),height=sfes.GetVSize();
  MFEM_VERIFY(std::size_t(ne)*5*n*n <= (std::size_t(1)<<26),"Tensor MG frozen stencil exceeds512MiB setup bound.");
  map.resize(height);nodal.resize(n*n);
  std::vector<char> seen(height,0);mfem::Array<int>dofs;
  for(int e=0;e<ne;++e)
  {
    const auto& fe=*sfes.GetFE(e);
    MFEM_VERIFY(fe.GetDof()==n && fe.GetOrder()==5 && fe.GetGeomType()==mfem::Geometry::SQUARE &&
      fe.GetMapType()==mfem::FiniteElement::VALUE && fe.GetNodes().GetNPoints()==n,
      "Tensor MG requires uniform p5 nodal VALUE L2 quads.");
    sfes.GetElementDofs(e,dofs);
    for(int i=0;i<n;++i)
    {
      MFEM_VERIFY(dofs[i]>=0 && dofs[i]<height && !seen[dofs[i]],"Tensor MG requires unsigned disjoint L2 maps.");
      map[e*n+i]=dofs[i];seen[dofs[i]]=1;
      const auto& ip=fe.GetNodes().IntPoint(i);
      for(int j=0;j<n;++j)
      {
        const double value=Legendre(j%d,ip.x)*Legendre(j/d,ip.y);
        if(e==0) nodal[i+n*j]=value;
        else MFEM_VERIFY(value==nodal[i+n*j],"Tensor MG requires identical nodal ordering.");
      }
    }
  }
  const auto colors=Colors(topology);
  color_count=1+*std::max_element(colors.begin(),colors.end());
  Stencil sampled(std::size_t(ne)*5*n*n,0.0);
  mfem::Vector input(height),output(height);input.UseDevice(true);output.UseDevice(true);
  for(int color=0;color<color_count;++color) for(int j=0;j<n;++j)
  {
    double* x=input.HostWrite();std::fill(x,x+height,0.0);
    // Probe modal columns directly, avoiding one of two dense basis changes.
    for(int e=0;e<ne;++e) if(colors[e]==color)
      for(int i=0;i<n;++i) x[map[e*n+i]]=nodal[i+n*j];
    op.Mult(input,output);++probes;
    const double* y=output.HostRead();
    for(int e=0;e<ne;++e) for(int slot=0;slot<5;++slot) if(colors[topology[e][slot]]==color)
      for(int i=0;i<n;++i)
      {
        double value=0;
        for(int k=0;k<n;++k) value+=static_cast<double>(nodal[k+n*i])*y[map[e*n+k]];
        MFEM_VERIFY(std::isfinite(static_cast<double>(value)),"Nonfinite frozen tensor MG stencil.");
        sampled[Entry(e,slot,i,j,n)]=static_cast<double>(value);
      }
  }
  return sampled;
}

Stencil Galerkin(const Stencil& fine,int ne,int d)
{
  const int n=d*d;Stencil result(std::size_t(ne)*5*n*n);
  for(int e=0;e<ne;++e)for(int s=0;s<5;++s)for(int j=0;j<n;++j)for(int i=0;i<n;++i)
    result[Entry(e,s,i,j,n)]=fine[Entry(e,s,i%d+6*(i/d),j%d+6*(j/d),36)];
  return result;
}

// Independent whole-domain probes detect missing graph edges and wrong
// restriction/order. Constant and nonconstant coarse polynomials are checked.
double CheckGalerkin(const mfem::Operator& op,const Topology& topology,
    const std::vector<int>& map,const std::vector<double>& nodal,
    const Stencil& a,int d,int& probes)
{
  const int ne=static_cast<int>(topology.size()),n=d*d;
  mfem::Vector x(ne*36),y(ne*36);x.UseDevice(true);y.UseDevice(true);
  double worst=0;
  for(int sample=0;sample<3;++sample)
  {
    std::vector<double> v(ne*n);
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
      v[e*n+i]=sample==0?(i==0?1.0:0.0):std::sin((sample==1?.17:.31)*(1+e*n+i));
    double* xx=x.HostWrite();
    for(int e=0;e<ne;++e)for(int k=0;k<36;++k)
    {
      double sum=0;for(int j=0;j<n;++j)sum+=static_cast<double>(nodal[k+36*(j%d+6*(j/d))])*v[e*n+j];
      xx[map[e*36+k]]=static_cast<double>(sum);
    }
    op.Mult(x,y);++probes;const double* yy=y.HostRead();
    double error=0,scale=0;
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
    {
      double measured=0,expected=0;
      for(int k=0;k<36;++k)measured+=static_cast<double>(nodal[k+36*(i%d+6*(i/d))])*yy[map[e*36+k]];
      for(int slot=0;slot<5;++slot)for(int j=0;j<n;++j)
        expected+=static_cast<double>(a[Entry(e,slot,i,j,n)])*v[topology[e][slot]*n+j];
      error+=(measured-expected)*(measured-expected);scale+=measured*measured;
    }
    const double relative=static_cast<double>(std::sqrt(error/std::max(scale,1e-300)));
    worst=std::max(worst,relative);
  }
  MFEM_VERIFY(worst<1e-8,"Tensor MG original/Galerkin stencil probe failed: "<<worst);
  return worst;
}

__global__ void InputMaximum(int n,const double* x,double* partial)
{
  __shared__ double values[256];double maximum=0;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
  { const double a=x[i];maximum=!isfinite(a)?CUDART_INF:fmax(maximum,fabs(a)); }
  values[threadIdx.x]=maximum;__syncthreads();
  for(int step=128;step;step/=2){if(threadIdx.x<step)values[threadIdx.x]=fmax(values[threadIdx.x],values[threadIdx.x+step]);__syncthreads();}
  if(threadIdx.x==0)partial[blockIdx.x]=values[0];
}
__global__ void InputExponent(int count,const double* partial,int* exponent)
{
  double maximum=0;for(int i=0;i<count;++i)maximum=fmax(maximum,partial[i]);
  int e=0;if(!isfinite(maximum))e=INT_MAX;else if(maximum>0)frexp(maximum,&e);*exponent=e;
}
template<typename T> __global__ void ToModal(int ne,const int* map,const double* t,
    const double* x,const int* exponent,T* b)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*36)return;
  if(*exponent==INT_MAX){b[row]=static_cast<T>(CUDART_NAN);return;}
  const int e=row/36,i=row%36;double sum=0;
  for(int k=0;k<36;++k)sum+=t[k+36*i]*scalbn(x[map[e*36+k]],-*exponent);
  b[row]=static_cast<T>(sum);
}
template<typename T> __global__ void FromModal(int ne,const int* map,const double* t,
    const T* x,const int* exponent,double scale,double* y)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*36)return;
  const int e=row/36,i=row%36;double sum=0;
  for(int j=0;j<36;++j)sum+=t[i+36*j]*static_cast<double>(x[e*36+j]);
  y[map[row]]=*exponent==INT_MAX?CUDART_NAN:scalbn(sum/scale,*exponent);
}
template<typename T> __global__ void Residual(int n,int ne,const int* neighbors,
    const T* a,const T* b,const T* x,T* r)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*n)return;
  const int e=row/n,i=row%n;T sum=0;
  for(int s=0;s<5;++s)for(int j=0;j<n;++j)
    sum+=a[((e*5+s)*n+j)*n+i]*x[neighbors[e*5+s]*n+j];
  r[row]=b[row]-sum;
}

template<typename T> __global__ void IGRTensorMultigridSharedResidualSIMTKernel(
    int ne,int d,int output_d,const int* neighbors,const T* a,
    const T* b,const T* x,T* r)
{
  const int n=d*d,m=output_d*output_d,row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=ne*m)return;
  const int e=row/m,i=row%m,fine_i=i%output_d+d*(i/output_d);T sum=0;
  for(int slot=0;slot<5;++slot)for(int j=0;j<n;++j)
    sum+=a[(slot*n+j)*n+fine_i]*x[neighbors[5*e+slot]*n+j];
  const T value=b[e*n+fine_i]-sum;
  r[row]=isfinite(value)?value:static_cast<T>(CUDART_NAN);
}

// One shared fixed reference matrix times different, genuine five-neighbor
// columns. The B panels are gathered directly, so there is no global packing
// workspace. Pre-coarse calls select only the coarse tensor rows and fuse
// b-Ax with restriction. All CTA warps participate, including tail columns.
__global__ void IGRTensorMultigridSharedResidualTF32Kernel(
    int ne,int d,int output_d,const int* neighbors,const float* a,
    const float* b,const float* x,float* r)
{
#if __CUDA_ARCH__ >= 800
  constexpr int warps=4;
  const int n=d*d,m=output_d*output_d,k=5*n,tiles_m=(m+15)/16;
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
  const int row0=16*(blockIdx.x%tiles_m);
  const int col0=16*(warps*(blockIdx.x/tiles_m)+warp);
  __shared__ __align__(32) float sa[8*24],sb[warps][16*16],sc[warps][16*24];
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> fa;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> fb;
  wmma::fragment<wmma::accumulator,16,16,8,float> fc;wmma::fill_fragment(fc,0.0f);
  for(int k0=0;k0<k;k0+=8)
  {
    for(int i=threadIdx.x;i<8*24;i+=32*warps)if(i%24<16)
    {
      const int row=row0+i%24,column=k0+i/24;
      const int fine_row=row%output_d+d*(row/output_d);
      sa[i]=wmma::__float_to_tf32(row<m && column<k?a[n*column+fine_row]:0.0f);
    }
    for(int i=lane;i<16*16;i+=32)if(i%16<8)
    {
      const int inner=k0+i%16,e=col0+i/16;
      const float value=e<ne && inner<k?
          x[neighbors[5*e+inner/n]*n+inner%n]:0.0f;
      sb[warp][i]=wmma::__float_to_tf32(value);
    }
    __syncthreads();
    wmma::load_matrix_sync(fa,sa,24);wmma::load_matrix_sync(fb,sb[warp],16);
    wmma::mma_sync(fc,fa,fb,fc);__syncthreads();
  }
  wmma::store_matrix_sync(sc[warp],fc,24,wmma::mem_col_major);__syncwarp();
  for(int i=lane;i<16*24;i+=32)if(i%24<16)
  {
    const int row=row0+i%24,e=col0+i/24;
    if(row<m && e<ne)
    {
      const int fine_row=row%output_d+d*(row/output_d);
      const float value=b[e*n+fine_row]-sc[warp][i];
      r[e*m+row]=isfinite(value)?value:CUDART_NAN_F;
    }
  }
#else
  const int m=output_d*output_d;
  for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<ne*m;row+=blockDim.x*gridDim.x)
    r[row]=CUDART_NAN_F;
#endif
}
template<typename T> __global__ void Restrict(int ne,int df,int dc,const T* r,T* b)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x,n=dc*dc;if(row>=ne*n)return;
  const int e=row/n,i=row%n;b[row]=r[e*df*df+i%dc+df*(i/dc)];
}
template<typename T> __global__ void ProlongAdd(int ne,int df,int dc,const T* c,T* x)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x,n=dc*dc;if(row>=ne*n)return;
  const int e=row/n,i=row%n;x[e*df*df+i%dc+df*(i/dc)]+=c[row];
}
template<typename T> __global__ void PackPatch(int ne,int d,const int* patches,const T* r,T* p)
{
  const int s=3*d,row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*s*s)return;
  const int e=row/(s*s),xy=row%(s*s),x=xy%s,y=xy/s;
  const int element=patches[9*e+3*(y/d)+x/d];p[row]=r[element*d*d+(x%d)+d*(y%d)];
}
template<typename T> __global__ void PackOwnedPatch(int tiles,int d,
    const int* patches,const T* r,T* packed)
{
  const int s=6*d,k=s*s,row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=tiles*k)return;
  const int tile=row/k,i=row%k,x=i%s,y=i/s;
  const int e=patches[36*tile+x/d+6*(y/d)];
  packed[row]=r[e*d*d+x%d+d*(y%d)];
}
template<typename T> __global__ void TransposePatch(int ne,int s,const T* a,T* b)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*s*s)return;
  const int e=row/(s*s),i=row%(s*s);b[e*s*s+(i%s)*s+i/s]=a[row];
}
template<typename T> __global__ void DividePatch(int ne,int s,const T* inverse,T* a)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<ne*s*s)a[row]*=inverse[row%(s*s)];
}
template<typename T> __global__ void AddCenter(int ne,int d,const T* factor,const T* patch,T* x)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x,n=d*d,s=3*d;if(row>=ne*n)return;
  const int e=row/n,i=row%n;x[row]+=T(.8)*factor[e]*patch[e*s*s+(d+i%d)+s*(d+i/d)];
}
template<typename T> __global__ void SharedTransformSIMT(int s,int columns,const T* a,const T* b,T* c)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=s*columns)return;
  const int i=row%s,j=row/s;T sum=0;for(int k=0;k<s;++k)sum+=a[i+s*k]*b[k+s*j];c[row]=sum;
}

// Each warp computes16 distinct real patch-line RHSs, except zero-padded tails.
// No RHS is copied into another column. Four independent warp tiles per CTA.
__global__ void IGRTensorMultigridTF32TransformKernel(int s,int columns,
    const float* a,const float* b,float* c)
{
#if __CUDA_ARCH__ >= 800
  const int warp=threadIdx.x/32,lane=threadIdx.x%32,tiles_m=(s+15)/16;
  const int tile=4*blockIdx.x+warp;if(tile>=tiles_m*((columns+15)/16))return;
  const int row0=16*(tile%tiles_m),col0=16*(tile/tiles_m);
  __shared__ __align__(32) float sa[4][128],sb[4][128],sc[4][256];
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> fa;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> fb;
  wmma::fragment<wmma::accumulator,16,16,8,float> fc;wmma::fill_fragment(fc,0.0f);
  for(int k0=0;k0<s;k0+=8)
  {
    for(int i=lane;i<128;i+=32)
    {
      const int ar=row0+i%16,ak=k0+i/16,bk=k0+i%8,bc=col0+i/8;
      sa[warp][i]=wmma::__float_to_tf32(ar<s && ak<s?a[ar+s*ak]:0.0f);
      sb[warp][i]=wmma::__float_to_tf32(bk<s && bc<columns?b[bk+s*bc]:0.0f);
    }
    __syncwarp();wmma::load_matrix_sync(fa,sa[warp],16);wmma::load_matrix_sync(fb,sb[warp],8);
    wmma::mma_sync(fc,fa,fb,fc);__syncwarp();
  }
  wmma::store_matrix_sync(sc[warp],fc,16,wmma::mem_col_major);__syncwarp();
  for(int i=lane;i<256;i+=32)
  { const int r=row0+i%16,col=col0+i/16;if(r<s && col<columns)c[r+s*col]=sc[warp][i]; }
#else
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<s*columns)c[row]=CUDART_NAN_F;
#endif
}

// Each SIMT thread owns one genuine element-center output, for one K slice.
// Split boundaries coincide with the TF32 kernel's eight-entry K panels.
template<typename T> __global__ void IGRTensorMultigridCenterSIMTKernel(
    int ne,int d,const T* response,const T* packed,T* partial,int split_k)
{
  const int n=d*d,k=9*n,row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=ne*n)return;
  const int part=blockIdx.y,panels=(k+7)/8;
  const int begin=8*((panels*part)/split_k);
  const int end=min(k,8*((panels*(part+1))/split_k));
  const int element=row/n,center=row%n;T sum=0;
  for(int j=begin;j<end;++j)sum+=response[center+n*j]*packed[j+k*element];
  partial[part*ne*n+row]=isfinite(sum)?sum:static_cast<T>(CUDART_NAN);
}

// Warps share one converted response panel and each own 16 distinct real
// patch columns. The skewed leading dimensions (24,16,24) replace the tightly
// packed, independently restaged panels of the separable control. All warps
// participate in CTA barriers, including M/N tails; tails are only zero fill.
// Scalar shared transfers traverse physical padded offsets, skipping padding,
// so active lanes in a warp access distinct banks in each transfer.
template<int Warps> __global__ void IGRTensorMultigridCenterTF32Kernel(
    int ne,int d,const float* response,const float* packed,float* partial,int split_k)
{
#if __CUDA_ARCH__ >= 800
  const int n=d*d,k=9*n,tiles_m=(n+15)/16;
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
  const int row0=16*(blockIdx.x%tiles_m);
  const int col0=16*(Warps*(blockIdx.x/tiles_m)+warp);
  const int part=blockIdx.y,panels=(k+7)/8;
  const int first=(panels*part)/split_k,last=(panels*(part+1))/split_k;
  __shared__ __align__(32) float sa[8*24],sb[Warps][16*16],sc[Warps][16*24];
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> fa;
  wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> fb;
  wmma::fragment<wmma::accumulator,16,16,8,float> fc;wmma::fill_fragment(fc,0.0f);
  for(int panel=first;panel<last;++panel)
  {
    const int k0=8*panel;
    for(int i=threadIdx.x;i<8*24;i+=32*Warps) if(i%24<16)
    {
      const int row=row0+i%24,column=k0+i/24;
      const float value=wmma::__float_to_tf32(row<n && column<k?response[row+n*column]:0.0f);
      sa[i]=isfinite(value)?value:CUDART_NAN_F;
    }
    for(int i=lane;i<16*16;i+=32) if(i%16<8)
    {
      const int row=k0+i%16,column=col0+i/16;
      const float value=wmma::__float_to_tf32(row<k && column<ne?packed[row+k*column]:0.0f);
      sb[warp][i]=isfinite(value)?value:CUDART_NAN_F;
    }
    __syncthreads();
    wmma::load_matrix_sync(fa,sa,24);wmma::load_matrix_sync(fb,sb[warp],16);
    wmma::mma_sync(fc,fa,fb,fc);
    __syncthreads();
  }
  wmma::store_matrix_sync(sc[warp],fc,24,wmma::mem_col_major);__syncwarp();
  for(int i=lane;i<16*24;i+=32) if(i%24<16)
  {
    const int row=row0+i%24,column=col0+i/24;
    if(row<n && column<ne)
    {
      const float value=sc[warp][i];
      partial[(part*ne+column)*n+row]=isfinite(value)?value:CUDART_NAN_F;
    }
  }
#else
  // A TF32 request is never silently replaced by a SIMT implementation.
  const int n=d*d;
  for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<ne*n;row+=gridDim.x*blockDim.x)
    partial[blockIdx.y*ne*n+row]=CUDART_NAN_F;
#endif
}

template<typename T> __global__ void IGRTensorMultigridAddCenterKernel(
    int ne,int d,const T* factor,const T* partial,T* x,int split_k)
{
  const int n=d*d,row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=ne*n)return;
  T sum=0;for(int part=0;part<split_k;++part)sum+=partial[part*ne*n+row];
  const T scale=factor[row/n],updated=x[row]+T(.8)*scale*sum;
  x[row]=isfinite(sum)&&isfinite(scale)&&isfinite(updated)?updated:static_cast<T>(CUDART_NAN);
}

template<typename T> __device__ __forceinline__ void StoreOwnedResponse(
    int tile,int row,int d,const int* owned,const T* factor,T sum,T* x)
{
  const int t=4*d,xx=row%t,yy=row/t;
  const int element=owned[16*tile+xx/d+4*(yy/d)];
  const int target=element*d*d+xx%d+d*(yy%d);
  const T scale=factor[element],updated=x[target]+T(.8)*scale*sum;
  x[target]=isfinite(sum)&&isfinite(scale)&&isfinite(updated)?updated:static_cast<T>(CUDART_NAN);
}

// Matched SIMT control: a 64x64 output tile per CTA, shared A/B panels, and a
// fused uniquely owned scatter. Each thread accumulates sixteen real outputs.
template<typename T> __global__ void IGRTensorMultigridOwnedSIMTKernel(
    int tiles,int d,const int* owned,const T* response,const T* packed,
    const T* factor,T* x)
{
  const int t=4*d,s=6*d,m=t*t,k=s*s,tiles_m=m/64;
  const int row0=64*(blockIdx.x%tiles_m),col0=64*(blockIdx.x/tiles_m);
  const int row=threadIdx.x%64,column_group=threadIdx.x/64;
  __shared__ __align__(32) T sa[8*72],sb[64*16];
  T sums[16];
  #pragma unroll
  for(int j=0;j<16;++j)sums[j]=0;
  for(int k0=0;k0<k;k0+=8)
  {
    for(int i=threadIdx.x;i<8*72;i+=256)if(i%72<64)
      sa[i]=response[row0+i%72+m*(k0+i/72)];
    for(int i=threadIdx.x;i<64*16;i+=256)if(i%16<8)
    {
      const int column=col0+i/16;
      sb[i]=column<tiles?packed[k0+i%16+k*column]:T(0);
    }
    __syncthreads();
    for(int inner=0;inner<8;++inner)
    {
      const T left=sa[row+72*inner];
      #pragma unroll
      for(int j=0;j<16;++j)sums[j]+=left*sb[inner+16*(column_group+4*j)];
    }
    __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<16;++j)
  {
    const int tile=col0+column_group+4*j;
    if(tile<tiles)StoreOwnedResponse(tile,row0+row,d,owned,factor,sums[j],x);
  }
}

// Standard WMMA only. Four warps each own a 32x32 region (four accumulator
// fragments) of a 64x64 output tile. Each staged 64x8 A / 8x64 B panel is
// reused across both M and N warp regions before the CTA advances K.
// Every warp reaches every barrier; incomplete real-column batches zero-fill.
__global__ void IGRTensorMultigridOwnedTF32Kernel(int tiles,int d,
    const int* owned,const float* response,const float* packed,
    const float* factor,float* x)
{
#if __CUDA_ARCH__ >= 800
  const int t=4*d,s=6*d,m=t*t,k=s*s,tiles_m=m/64;
  const int row0=64*(blockIdx.x%tiles_m),col0=64*(blockIdx.x/tiles_m);
  const int warp=threadIdx.x/32;
  const int warp_row=32*(warp%2),warp_column=32*(warp/2);
  __shared__ __align__(32) float sa[8*72],sb[64*16],sc[64*72];
  using namespace nvcuda;
  wmma::fragment<wmma::accumulator,16,16,8,float> accum[2][2];
  #pragma unroll
  for(int i=0;i<2;++i)
  {
    #pragma unroll
    for(int j=0;j<2;++j)wmma::fill_fragment(accum[i][j],0.0f);
  }
  for(int k0=0;k0<k;k0+=8)
  {
    for(int i=threadIdx.x;i<8*72;i+=128)if(i%72<64)
      sa[i]=wmma::__float_to_tf32(response[row0+i%72+m*(k0+i/72)]);
    for(int i=threadIdx.x;i<64*16;i+=128)if(i%16<8)
    {
      const int column=col0+i/16;
      sb[i]=wmma::__float_to_tf32(column<tiles?packed[k0+i%16+k*column]:0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::col_major> a[2];
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::col_major> b[2];
    #pragma unroll
    for(int i=0;i<2;++i)wmma::load_matrix_sync(a[i],sa+warp_row+16*i,72);
    #pragma unroll
    for(int j=0;j<2;++j)wmma::load_matrix_sync(b[j],sb+16*(warp_column+16*j),16);
    #pragma unroll
    for(int i=0;i<2;++i)
    {
      #pragma unroll
      for(int j=0;j<2;++j)wmma::mma_sync(accum[i][j],a[i],b[j],accum[i][j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<2;++i)
  {
    #pragma unroll
    for(int j=0;j<2;++j)
      wmma::store_matrix_sync(sc+warp_row+16*i+72*(warp_column+16*j),
          accum[i][j],72,wmma::mem_col_major);
  }
  __syncthreads();
  for(int i=threadIdx.x;i<64*72;i+=128)if(i%72<64)
  {
    const int tile=col0+i/72;
    if(tile<tiles)StoreOwnedResponse(tile,row0+i%72,d,owned,factor,sc[i],x);
  }
#else
  const int t=4*d,m=t*t;
  for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<tiles*m;row+=gridDim.x*blockDim.x)
    StoreOwnedResponse(row/m,row%m,d,owned,factor,CUDART_NAN_F,x);
#endif
}

void CheckCenterLayout(int ne,int d,int warps,int split_k)
{
  MFEM_VERIFY(ne>0 && (d==4||d==6) && (warps==1||warps==4) &&
    (split_k==1||split_k==2||split_k==4),"Invalid tensor center response layout.");
  MFEM_VERIFY(std::size_t(ne)<=std::size_t(INT_MAX-255)/(9*d*d) &&
              std::size_t(ne)<=std::size_t(INT_MAX-255)/(d*d*split_k),
              "Tensor center response layout exceeds integer indexing.");
}

template<typename From,typename To> __global__ void Convert(int n,const From* x,To* y)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=static_cast<To>(x[i]);}

template<typename T> void Store(mfem::Array<T>& dst,const std::vector<double>& src)
{
  dst.SetSize(static_cast<int>(src.size()));T* out=dst.HostWrite();
  for(std::size_t i=0;i<src.size();++i)
  {out[i]=static_cast<T>(src[i]);MFEM_VERIFY(std::isfinite(static_cast<double>(out[i])),"Tensor MG storage overflow.");}
  (void)dst.Read();
}

struct PatchReference
{
  std::vector<double> forward_x,forward_y,back_x,back_y,inverse,scales;
  int clipped=0;
  double relative_error=0, eigen_error=0;
};

PatchReference Reference(const Stencil& a,int ne,int d,
    const std::vector<double>* traces=nullptr,int patch_elements=3)
{
  MFEM_VERIFY(patch_elements==3 || patch_elements==6,"Invalid tensor reference patch width.");
  const int n=d*d,s=patch_elements*d,np=s*s;PatchReference result;
  std::vector<double> average(5*n*n,0.0);
  for(int slot=0;slot<5;++slot)for(int j=0;j<n;++j)for(int i=0;i<n;++i)
  {
    if(traces)average[(slot*n+j)*n+i]=a[Entry(0,slot,i,j,n)];
    else
    {
      double sum=0;for(int e=0;e<ne;++e)sum+=a[Entry(e,slot,i,j,n)];
      average[(slot*n+j)*n+i]=static_cast<double>(sum/ne);
    }
  }
  std::vector<double> patch(np*np,0.0);
  for(int cell=0;cell<patch_elements*patch_elements;++cell)for(int slot=0;slot<5;++slot)
  {
    const int x=cell%patch_elements,y=cell/patch_elements;
    const int xx=x+(slot==1?-1:slot==2?1:0),yy=y+(slot==3?-1:slot==4?1:0);
    if(xx<0||xx>=patch_elements||yy<0||yy>=patch_elements)continue;
    for(int j=0;j<n;++j)for(int i=0;i<n;++i)
    {
      const int row=x*d+i%d+s*(y*d+i/d),col=xx*d+j%d+s*(yy*d+j/d);
      patch[row+np*col]=average[(slot*n+j)*n+i];
    }
  }
  double asymmetry=0,magnitude=0;
  for(int j=0;j<np;++j)for(int i=0;i<np;++i)
  {const double x=patch[i+np*j],y=patch[j+np*i];asymmetry+=(x-y)*(x-y);magnitude+=x*x;}
  MFEM_VERIFY(std::sqrt(asymmetry/magnitude)<1e-10,"Tensor MG patch reference is not symmetric.");
  for(int j=0;j<np;++j)for(int i=0;i<j;++i)
    patch[i+np*j]=patch[j+np*i]=.5*(patch[i+np*j]+patch[j+np*i]);
  std::vector<double> ax(s*s,0),ay(s*s,0);
  for(int j=0;j<s;++j)for(int i=0;i<s;++i)for(int k=0;k<s;++k)
  {ax[i+s*j]+=patch[(i+s*k)+np*(j+s*k)]/s;ay[i+s*j]+=patch[(k+s*i)+np*(k+s*j)]/s;}
  double mean=0;for(int i=0;i<s;++i)mean+=ax[i+s*i]/s;
  double mass=0;for(int slot=0;slot<5;++slot)mass+=average[slot*n*n];
  MFEM_VERIFY(mass>0 && std::isfinite(mass),"Tensor MG requires positive screened constant mode.");
  std::vector<double> lx,ly;Eigen(ax,s,result.back_x,lx);Eigen(ay,s,result.back_y,ly);
  result.forward_x.resize(s*s);result.forward_y.resize(s*s);result.inverse.resize(s*s);
  for(int j=0;j<s;++j)for(int i=0;i<s;++i)
  {
    result.forward_x[i+s*j]=result.back_x[j+s*i];result.forward_y[i+s*j]=result.back_y[j+s*i];
    const double value=lx[i]+ly[j]-mean;
    if(value<mass)++result.clipped;
    // The exact constant-coefficient Kronecker sum has no clipping. Variable
    // coefficients are approximated only inside this positive patch smoother.
    result.inverse[j+s*i]=1/std::max(mass,value); // transposed (y,x) layout
  }
  for(int dim=0;dim<2;++dim)
  {
    const auto& mat=dim?ay:ax;const auto& q=dim?result.back_y:result.back_x;const auto& eig=dim?ly:lx;
    double error=0,scale=0;
    for(int j=0;j<s;++j)for(int i=0;i<s;++i)
    {double v=0;for(int k=0;k<s;++k)v+=static_cast<double>(mat[i+s*k])*q[k+s*j];
     const double delta=v-q[i+s*j]*eig[j];error+=delta*delta;scale+=v*v;}
    result.eigen_error=std::max(result.eigen_error,static_cast<double>(std::sqrt(error/scale)));
  }
  MFEM_VERIFY(result.eigen_error<1e-12,"Tensor patch eigensystem residual failed.");
  double error=0,scale=0;
  for(int j=0;j<np;++j)for(int i=0;i<np;++i)
  {
    const int ix=i%s,iy=i/s,jx=j%s,jy=j/s;
    const double approx=(iy==jy?ax[ix+s*jx]:0)+(ix==jx?ay[iy+s*jy]:0)-(i==j?mean:0);
    const double delta=approx-patch[i+np*j];error+=delta*delta;scale+=patch[i+np*j]*patch[i+np*j];
  }
  result.relative_error=static_cast<double>(std::sqrt(error/scale));
  double reference_diag=0;for(int i=0;i<n;++i)reference_diag+=average[i+n*i];
  result.scales.resize(ne);
  MFEM_VERIFY(!traces || traces->size()==std::size_t(ne),"Shared tensor reference trace size mismatch.");
  for(int e=0;e<ne;++e)
  {
    double diagonal=traces?(*traces)[e]:0.0;
    if(!traces)for(int i=0;i<n;++i)diagonal+=a[Entry(e,0,i,i,n)];
    const double ratio=reference_diag/diagonal;
    MFEM_VERIFY(std::isfinite(ratio)&&ratio>=1.0/16&&ratio<=16,"Tensor MG patch coefficient contrast exceeds16.");
    result.scales[e]=ratio;
  }
  return result;
}

template<typename T> struct Level
{
  int d=0;
  mfem::Array<T> a,forward_x,forward_y,back_x,back_y,inverse,center_response,scale;
  mutable mfem::Array<T> rhs,x,residual,patch_a,patch_b;
};

void SpectralBLAS(cublasStatus_t status)
{MFEM_VERIFY(status==CUBLAS_STATUS_SUCCESS,"Spectral cuBLAS status="<<status);}
void SpectralSolver(cusolverStatus_t status)
{MFEM_VERIFY(status==CUSOLVER_STATUS_SUCCESS,"Spectral cuSolver status="<<status);}

__global__ void BuildSpectralMatrix(int n,bool periodic,const double* blocks,double* matrix)
{
  const int entry=blockIdx.x*blockDim.x+threadIdx.x;if(entry>=n*n)return;
  const int row=entry%n,col=entry/n,cell=row/6,cells=n/6,neighbor=col/6;
  const int i=row%6,j=col%6;
  double value=neighbor==cell?blocks[i+6*j]:0.0;
  if(!periodic && neighbor==cell)
  {
    if(cell==0)value+=blocks[108+i+6*j];
    if(cell==cells-1)value+=blocks[144+i+6*j];
  }
  if(neighbor==(periodic?(cell+cells-1)%cells:cell-1))value+=blocks[36+i+6*j];
  if(neighbor==(periodic?(cell+1)%cells:cell+1))value+=blocks[72+i+6*j];
  matrix[entry]=value;
}

__global__ void CheckSpectralEigenKernel(int n,bool periodic,const double* blocks,
    const double* q,const double* lambda,double* partial,int* invalid)
{
  __shared__ double errors[256],scales[256];double error=0,scale=0;
  const int cells=n/6;
  for(long long entry=blockIdx.x*blockDim.x+threadIdx.x;entry<static_cast<long long>(n)*n;
      entry+=blockDim.x*gridDim.x)
  {
    const int row=entry%n,column=entry/n,cell=row/6,i=row%6;
    double action=0;
    for(int slot=0;slot<3;++slot)
    {
      const int neighbor=slot==0?cell:slot==1?
          (periodic?(cell+cells-1)%cells:cell-1):(periodic?(cell+1)%cells:cell+1);
      if(neighbor<0 || neighbor>=cells)continue;
      for(int j=0;j<6;++j)
      {
        double coefficient=blocks[slot*36+i+6*j];
        if(!periodic && slot==0)
        {
          if(cell==0)coefficient+=blocks[108+i+6*j];
          if(cell==cells-1)coefficient+=blocks[144+i+6*j];
        }
        action+=coefficient*q[6*neighbor+j+n*column];
      }
    }
    const double value=q[entry],eigenvalue=lambda[column],delta=action-eigenvalue*value;
    if(!isfinite(value)||!isfinite(eigenvalue)||!isfinite(delta)||
        (column>0 && eigenvalue<lambda[column-1]))atomicExch(invalid,1);
    error+=delta*delta;scale+=action*action;
  }
  errors[threadIdx.x]=error;scales[threadIdx.x]=scale;__syncthreads();
  for(int step=128;step;step/=2)
  {
    if(threadIdx.x<step){errors[threadIdx.x]+=errors[threadIdx.x+step];scales[threadIdx.x]+=scales[threadIdx.x+step];}
    __syncthreads();
  }
  if(threadIdx.x==0){partial[2*blockIdx.x]=errors[0];partial[2*blockIdx.x+1]=scales[0];}
}

__global__ void CheckSpectralOrthogonalKernel(int n,const double* gram,double* partial)
{
  __shared__ double values[256];double sum=0;
  for(long long entry=blockIdx.x*blockDim.x+threadIdx.x;entry<static_cast<long long>(n)*n;
      entry+=blockDim.x*gridDim.x)
  {
    const double delta=gram[entry]-(entry%n==entry/n?1.0:0.0);sum+=delta*delta;
  }
  values[threadIdx.x]=sum;__syncthreads();
  for(int step=128;step;step/=2)
  {if(threadIdx.x<step)values[threadIdx.x]+=values[threadIdx.x+step];__syncthreads();}
  if(threadIdx.x==0)partial[blockIdx.x]=values[0];
}

template<typename T> __global__ void SpectralInverseKernel(int nx,int ny,
    const double* lx,const double* ly,double mean,double mass,T* inverse,int* status)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=nx*ny)return;
  const double value=lx[row%nx]+ly[row/nx]-mean;
  if(value<mass)atomicAdd(status+1,1);
  const T inv=static_cast<T>(1.0/fmax(mass,value));inverse[row]=inv;
  if(!isfinite(value)||!(inv>0)||!isfinite(inv))atomicExch(status,1);
}

template<typename T> __global__ void ToSpectralKernel(int nx,int ny,
    const int* elements,const int* map,const double* nodal,const double* x,
    const int* exponent,T* result)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=nx*ny)return;
  if(*exponent==INT_MAX){result[row]=static_cast<T>(CUDART_NAN);return;}
  const int gx=row%nx,gy=row/nx,e=elements[gx/6+(nx/6)*(gy/6)],i=gx%6+6*(gy%6);
  double sum=0;
  for(int j=0;j<36;++j)sum+=nodal[j+36*i]*scalbn(x[map[36*e+j]],-*exponent);
  result[row]=static_cast<T>(sum);
}

template<typename T> __global__ void FromSpectralKernel(int nx,int ny,
    const int* elements,const int* map,const double* nodal,const T* x,
    const int* exponent,double scale,double* result)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=nx*ny)return;
  const int gx=row%nx,gy=row/nx,cx=gx/6,cy=gy/6,e=elements[cx+(nx/6)*cy],i=gx%6+6*(gy%6);
  if(*exponent==INT_MAX){result[map[36*e+i]]=CUDART_NAN;return;}
  double sum=0;
  for(int j=0;j<36;++j)sum+=nodal[i+36*j]*static_cast<double>(x[6*cx+j%6+nx*(6*cy+j/6)]);
  result[map[36*e+i]]=scalbn(sum/scale,*exponent);
}

template<typename T> __global__ void ScaleSpectralKernel(int n,const T* inverse,T* x)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row<n)x[row]*=inverse[row];
}

__global__ void SplitSpectralTF32Kernel(int n,const float* source,float* hi,float* lo)
{
  const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=n)return;
#if __CUDA_ARCH__ >= 800
  const float value=source[row];
  const float high=nvcuda::wmma::__float_to_tf32(value);
  hi[row]=high;lo[row]=isfinite(value)?value-high:CUDART_NAN_F;
#else
  hi[row]=lo[row]=CUDART_NAN_F;
#endif
}

template<typename T> class TensorMultigrid final:public mfem::Solver
{
public:
  TensorMultigrid(const mfem::Operator& op,mfem::ParFiniteElementSpace& sfes,
      TensorMultigridPrecision precision,TensorMultigridSmoother smoother,
      int center_warps,int center_split_k,TensorMultigridOperator operator_mode,int owned_tile,
      TensorMultigridResidual residual_mode)
    :mfem::Solver(op.Height()),ne_(sfes.GetNE()),precision_(precision),smoother_(smoother),
     operator_mode_(operator_mode),residual_mode_(residual_mode),
     center_warps_(center_warps),center_split_k_(center_split_k),owned_tile_(owned_tile)
  {
    const auto start=std::chrono::steady_clock::now();
    CheckCenterLayout(ne_,6,center_warps_,center_split_k_);
    int ranks=0;MPI_Comm_size(sfes.GetComm(),&ranks);
    MFEM_VERIFY(ranks==1 && mfem::Device::Allows(mfem::Backend::CUDA) &&
      dynamic_cast<const mfem::L2_FECollection*>(sfes.FEColl()) && sfes.GetVDim()==1 &&
      sfes.GetMesh()->Dimension()==2 && sfes.GetVSize()==height && sfes.GetTrueVSize()==height &&
      op.Width()==height && std::size_t(height)==std::size_t(ne_)*36,
      "Tensor MG requires rank-one CUDA scalar p5 L2.");
    int device=0;cudaDeviceProp prop{};CUDA(cudaGetDevice(&device));CUDA(cudaGetDeviceProperties(&prop,device));
    if(precision_==TensorMultigridPrecision::TF32)
    {MFEM_VERIFY(prop.major>=8,"Tensor MG TF32 requires nativeSM80+.");}
    const bool shared=operator_mode_==TensorMultigridOperator::Shared;
    const auto topology=SharedPatchTopology(sfes);
    if(shared)CheckTranslationTopology(topology);
    neighbors_.SetSize(ne_*5);
    for(int e=0;e<ne_;++e)for(int s=0;s<5;++s)neighbors_[5*e+s]=topology[e][s];
    if(owned_tile_==4)
    {
      const auto layout=BuildTensorOwnedPatchMap(topology);
      owned_tiles_=ne_/16;owned_nx_=layout.nx;owned_ny_=layout.ny;
      MFEM_VERIFY(TensorOwnedLayoutValid(owned_tiles_,ne_,6),"Owned tensor multigrid layout exceeds integer bounds.");
      owned_inputs_.SetSize(static_cast<int>(layout.input.size()));
      owned_outputs_.SetSize(static_cast<int>(layout.owned.size()));
      std::copy(layout.input.begin(),layout.input.end(),owned_inputs_.HostWrite());
      std::copy(layout.owned.begin(),layout.owned.end(),owned_outputs_.HostWrite());
    }
    else
    {
      const auto patch=PatchMap(topology);patches_.SetSize(ne_*9);
      std::copy(patch.begin(),patch.end(),patches_.HostWrite());
    }
    std::vector<int> map;std::vector<double> nodal;int probes=0,colors=0;
    SharedSnapshotData reference;Stencil fine;
    if(shared)
    {
      reference=SharedSnapshot(op,sfes,topology,neighbors_,map,nodal,probes,colors);
      fine=std::move(reference.average);operator_scale_=reference.maximum_diagonal;
    }
    else fine=Snapshot(op,sfes,topology,map,nodal,probes,colors);
    double probe_error=0;std::array<Stencil,3> matrices;
    for(int l=0;l<3;++l)
    {
      const int d=l==0?6:l==1?4:2;matrices[l]=l==0?fine:Galerkin(fine,shared?1:ne_,d);
      if(!shared)probe_error=std::max(probe_error,CheckGalerkin(op,topology,map,nodal,matrices[l],d,probes));
    }
    if(!shared)
    {
      operator_scale_=0;
      for(int e=0;e<ne_;++e)for(int i=0;i<36;++i)
        operator_scale_=std::max(operator_scale_,std::abs(fine[Entry(e,0,i,i,36)]));
    }
    MFEM_VERIFY(operator_scale_>0 && std::isfinite(operator_scale_),"Invalid tensor MG operator scale.");
    element_map_.SetSize(height);std::copy(map.begin(),map.end(),element_map_.HostWrite());
    nodal_.SetSize(36*36);std::copy(nodal.begin(),nodal.end(),nodal_.HostWrite());nodal_.UseDevice(true);
    std::size_t bytes=0,center_bytes=0,patch_bytes=0;int clipped=0;
    double separable_error=0,eigen_error=0,center_setup_seconds=0;
    for(int l=0;l<3;++l)
    {
      auto& level=levels_[l];level.d=l==0?6:l==1?4:2;const int n=level.d*level.d;
      for(double& value:matrices[l])value/=operator_scale_;
      Store(level.a,matrices[l]);bytes+=matrices[l].size()*sizeof(T);
      level.rhs.SetSize(ne_*n);level.x.SetSize(ne_*n);level.residual.SetSize(ne_*n);
      if(l<2)
      {
        if(shared)for(double& value:reference.traces[l])value/=operator_scale_;
        const auto ref=Reference(matrices[l],ne_,level.d,shared?&reference.traces[l]:nullptr,
            owned_tile_==4?6:3);
        Store(level.scale,ref.scales);
        const int size=owned_tile_==4?owned_tiles_*36*n:ne_*9*n;level.patch_a.SetSize(size);
        if(smoother_==TensorMultigridSmoother::Center)
        {
          const auto center_start=std::chrono::steady_clock::now();
          const auto response=owned_tile_==4?
            ComposeTensorOwnedResponse(level.d,ref.forward_x,ref.forward_y,ref.back_x,ref.back_y,ref.inverse):
            ComposeTensorCenterResponse(level.d,ref.forward_x,ref.forward_y,ref.back_x,ref.back_y,ref.inverse);
          Store(level.center_response,response);
          center_setup_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-center_start).count();
          center_bytes+=response.size()*sizeof(T);
          if(owned_tile_==1)level.patch_b.SetSize(center_split_k_*ne_*n);
        }
        else
        {
          Store(level.forward_x,ref.forward_x);Store(level.forward_y,ref.forward_y);
          Store(level.back_x,ref.back_x);Store(level.back_y,ref.back_y);
          Store(level.inverse,ref.inverse);level.patch_b.SetSize(size);
        }
        patch_bytes+=(std::size_t(level.patch_a.Size())+level.patch_b.Size())*sizeof(T);
        clipped+=ref.clipped;separable_error=std::max(separable_error,ref.relative_error);eigen_error=std::max(eigen_error,ref.eigen_error);
      }
    }
    if(shared)
    {
      // The graph is known: construct sorted CSR directly, avoiding 80*ne
      // linked-list insertions and their transient allocation at large ne.
      const int rows=4*ne_;
      // Allocate through MFEM so its configured HOST/UMPIRE memory policy owns
      // and releases these buffers. Wrapping new[] with the raw CSR constructor
      // incorrectly registers that pointer as the configured host memory type.
      coarse_=std::make_unique<mfem::SparseMatrix>(rows,rows,20);
      int* columns=coarse_->HostWriteJ();
      mfem::real_t* values=coarse_->HostWriteData();
      for(int e=0;e<ne_;++e)
      {
        std::array<int,5> order{{0,1,2,3,4}};
        std::sort(order.begin(),order.end(),[&](int a,int b){return topology[e][a]<topology[e][b];});
        for(int i=0;i<4;++i)for(int block=0;block<5;++block)for(int j=0;j<4;++j)
        {
          const int slot=order[block],entry=20*(4*e+i)+4*block+j;
          columns[entry]=4*topology[e][slot]+j;
          values[entry]=matrices[2][Entry(0,slot,i,j,4)];
        }
      }
    }
    else
    {
      coarse_=std::make_unique<mfem::SparseMatrix>(4*ne_);
      for(int e=0;e<ne_;++e)for(int s=0;s<5;++s)for(int j=0;j<4;++j)for(int i=0;i<4;++i)
        coarse_->Add(4*e+i,4*topology[e][s]+j,matrices[2][Entry(e,s,i,j,4)]);
      coarse_->Finalize();
    }
    mfem::Vector diagonal(4*ne_);coarse_->GetDiag(diagonal);
    // SparseMatrix::GetDiag writes through the device backend; Vector::operator()
    // does not synchronize host validity. Validate the actual computed entries.
    const double* diagonal_values=diagonal.HostRead();
    for(int i=0;i<diagonal.Size();++i)
      MFEM_VERIFY(std::isfinite(diagonal_values[i])&&diagonal_values[i]>0,
                  "Nonpositive tensor p1 diagonal at row "<<i<<": "<<diagonal_values[i]);
    jacobi_=std::make_unique<mfem::OperatorJacobiSmoother>(diagonal,essential_,1.0);
    coarse_solver_=MakeCooperativeCGSolver(sfes.GetComm());cooperative_=bool(coarse_solver_);
    if(!coarse_solver_)coarse_solver_=MakeDeviceCGSolver(sfes.GetComm());
    coarse_solver_->SetOperator(*coarse_);coarse_solver_->SetPreconditioner(*jacobi_);
    coarse_solver_->SetRelTol(1e-2);coarse_solver_->SetAbsTol(0);coarse_solver_->SetMaxIter(100);
    coarse_solver_->SetPrintLevel(0);coarse_solver_->iterative_mode=false;
    coarse_rhs_.SetSize(4*ne_);coarse_x_.SetSize(4*ne_);coarse_rhs_.UseDevice(true);coarse_x_.UseDevice(true);
    norm_blocks_=std::min(4096,(height+255)/256);norm_partial_.SetSize(norm_blocks_);norm_partial_.UseDevice(true);exponent_.SetSize(1);
    (void)neighbors_.Read();
    if(owned_tile_==4){(void)owned_inputs_.Read();(void)owned_outputs_.Read();}
    else (void)patches_.Read();
    (void)element_map_.Read();(void)nodal_.Read();
    CUDA(cudaDeviceSynchronize());
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"[igr-tensor-mg] event=setup precision="<<Name(precision_)<<" hierarchy=5,3,1 cycle=multiplicative-v"
      <<" pre_sweeps=1 post_sweeps=1 smoother="<<SmootherName(smoother_,owned_tile_)<<" damping=0.8"
      <<" operator="<<OperatorName(operator_mode_)
      <<" owned_tile="<<owned_tile_
      <<" residual_precision_policy="<<ResidualPolicyName(residual_mode_)
      <<" residual_arithmetic="<<(precision_==TensorMultigridPrecision::FP64?"fp64-simt":
          shared && precision_==TensorMultigridPrecision::TF32 && residual_mode_==TensorMultigridResidual::Auto?
            "tf32-wmma":"fp32-simt")
      <<" center_warps="<<center_warps_<<" center_split_k="<<center_split_k_
      <<" elements="<<ne_<<" n="<<height<<" coarse_n="<<4*ne_<<" coarse_precision=fp64 coarse_rtol=0.01 coarse_maxiter=100"
      <<" coarse_backend=cuda-fp64 coarse_execution="<<(cooperative_?"cooperative":"host-polled")
      <<" vectors="<<(std::is_same<T,double>::value?"fp64":"fp32")
      <<" real_columns_p5="<<(owned_tile_==4?owned_tiles_:smoother_==TensorMultigridSmoother::Center?ne_:18*ne_)
      <<" real_columns_p3="<<(owned_tile_==4?owned_tiles_:smoother_==TensorMultigridSmoother::Center?ne_:12*ne_)
      <<" operator_probes="<<probes<<" colors="<<colors
      <<" center_map_bytes="<<center_bytes<<" patch_workspace_bytes="<<patch_bytes
      <<" center_setup_seconds="<<center_setup_seconds
      <<" stencil_bytes="<<bytes<<" coarse_nnz="<<coarse_->NumNonZeroElems();
    if(shared)std::cout<<" shared_symmetry_rel="<<reference.symmetry_error
      <<" original_galerkin_check=not-applicable reference_model=translation-averaged-frozen-coefficients"
      <<" reference_diagonal_scaling=per-element-trace residual_pack=direct-gather residual_restrict=fused";
    else std::cout<<" probe_rel="<<probe_error;
    if(owned_tile_==4)std::cout<<" owned_tiles="<<owned_tiles_<<" owned_nx="<<owned_nx_<<" owned_ny="<<owned_ny_
      <<" owned_input_elements=36 owned_output_elements=16 owned_halo=1"
      <<" owned_matrix_m_p5=576 owned_matrix_k_p5=1296 owned_matrix_m_p3=256 owned_matrix_k_p3=576"
      <<" owned_output_cta=64x64 owned_tf32_warps=4 owned_simt_threads=256 owned_scatter=fused owned_split_k=1";
    std::cout<<" separable_reference_rel="<<separable_error<<" eigen_residual="<<eigen_error<<" clipped_eigenvalues="<<clipped
      <<" setup_seconds="<<seconds<<" frozen_operator="<<(shared?"shared-reference-approximate":"sampled-original-galerkin")
      <<" alpha=inherited-fine"
      <<" tensor_execution="<<(precision_==TensorMultigridPrecision::TF32?"requires-profile-proof":"none")<<" fallback=none\n";
  }
  ~TensorMultigrid() override
  {
    std::cout<<"[igr-tensor-mg] event=summary precision="<<Name(precision_)<<" applications="<<applications_
      <<" smoother="<<SmootherName(smoother_,owned_tile_)<<" center_warps="<<center_warps_<<" center_split_k="<<center_split_k_
      <<" operator="<<OperatorName(operator_mode_)
      <<" owned_tile="<<owned_tile_<<" owned_submissions="<<owned_submissions_
      <<" residual_precision_policy="<<ResidualPolicyName(residual_mode_)
      <<" coarse_iterations="<<coarse_iterations_<<" coarse_failures="<<coarse_failures_<<" coarse_cap_hits="<<coarse_cap_hits_
      <<" transform_submissions="<<transforms_<<" center_submissions="<<centers_
      <<" center_pack_submissions="<<center_packs_<<" center_add_submissions="<<center_adds_
      <<" shared_residual_submissions="<<shared_residuals_<<" shared_restricted_residual_submissions="<<shared_restricted_residuals_
      <<" cycle=multiplicative-v fallback=none\n";
  }
  void SetOperator(const mfem::Operator& op) override
  {MFEM_VERIFY(op.Height()==height&&op.Width()==width,"Tensor MG frozen hierarchy dimension changed; rebuild required.");}
  mfem::MemoryClass GetMemoryClass() const override{return mfem::MemoryClass::DEVICE;}
  void Mult(const mfem::Vector& b,mfem::Vector& x) const override
  {
    MFEM_VERIFY(b.Size()==height&&&b!=&x,"Tensor MG invalid/aliased input.");x.SetSize(height);x.UseDevice(true);
    InputMaximum<<<norm_blocks_,256>>>(height,b.Read(),norm_partial_.Write());
    InputExponent<<<1,1>>>(norm_blocks_,norm_partial_.Read(),exponent_.Write());
    ToModal<<<(height+255)/256,256>>>(ne_,element_map_.Read(),nodal_.Read(),b.Read(),exponent_.Read(),levels_[0].rhs.Write());
    Cycle(0);
    FromModal<<<(height+255)/256,256>>>(ne_,element_map_.Read(),nodal_.Read(),levels_[0].x.Read(),exponent_.Read(),operator_scale_,x.Write());
    CUDA(cudaGetLastError());++applications_;
  }
private:
  void Transform(int s,const mfem::Array<T>& a,const mfem::Array<T>& b,mfem::Array<T>& c) const
  {
    const int columns=ne_*s;
    if constexpr(std::is_same<T,float>::value)
    {
      if(precision_==TensorMultigridPrecision::TF32)
      {const int tiles=((s+15)/16)*((columns+15)/16);IGRTensorMultigridTF32TransformKernel<<<(tiles+3)/4,128>>>(s,columns,a.Read(),b.Read(),c.Write());}
      else SharedTransformSIMT<<<(s*columns+255)/256,256>>>(s,columns,a.Read(),b.Read(),c.Write());
    }
    else SharedTransformSIMT<<<(s*columns+255)/256,256>>>(s,columns,a.Read(),b.Read(),c.Write());
    ++transforms_;
  }
  void Smooth(int l,const mfem::Array<T>& r) const
  {
    if(owned_tile_==4)
    {
      const auto& v=levels_[l];const int size=owned_tiles_*36*v.d*v.d;
      PackOwnedPatch<<<(size+255)/256,256>>>(owned_tiles_,v.d,owned_inputs_.Read(),r.Read(),v.patch_a.Write());
      if constexpr(std::is_same<T,float>::value)
        ApplyTensorOwnedResponse(owned_tiles_,ne_,v.d,owned_outputs_.Read(),v.center_response.Read(),
            v.patch_a.Read(),v.scale.Read(),v.x.ReadWrite(),precision_==TensorMultigridPrecision::TF32);
      else
        ApplyTensorOwnedResponse(owned_tiles_,ne_,v.d,owned_outputs_.Read(),v.center_response.Read(),
            v.patch_a.Read(),v.scale.Read(),v.x.ReadWrite());
      ++owned_submissions_;return;
    }
    const auto& v=levels_[l];const int d=v.d,s=3*d,size=ne_*s*s;
    PackPatch<<<(size+255)/256,256>>>(ne_,d,patches_.Read(),r.Read(),v.patch_a.Write());
    if(smoother_==TensorMultigridSmoother::Center)
    {
      if constexpr(std::is_same<T,float>::value)
        ApplyTensorCenterResponse(ne_,d,v.center_response.Read(),v.patch_a.Read(),v.patch_b.Write(),
            precision_==TensorMultigridPrecision::TF32,center_warps_,center_split_k_);
      else
        ApplyTensorCenterResponse(ne_,d,v.center_response.Read(),v.patch_a.Read(),v.patch_b.Write(),
            center_warps_,center_split_k_);
      AddTensorCenterResponse(ne_,d,v.scale.Read(),v.patch_b.Read(),v.x.ReadWrite(),center_split_k_);
      ++centers_;++center_packs_;++center_adds_;return;
    }
    Transform(s,v.forward_x,v.patch_a,v.patch_b);
    TransposePatch<<<(size+255)/256,256>>>(ne_,s,v.patch_b.Read(),v.patch_a.Write());
    Transform(s,v.forward_y,v.patch_a,v.patch_b);
    DividePatch<<<(size+255)/256,256>>>(ne_,s,v.inverse.Read(),v.patch_b.ReadWrite());
    Transform(s,v.back_y,v.patch_b,v.patch_a);
    TransposePatch<<<(size+255)/256,256>>>(ne_,s,v.patch_a.Read(),v.patch_b.Write());
    Transform(s,v.back_x,v.patch_b,v.patch_a);
    AddCenter<<<(ne_*d*d+255)/256,256>>>(ne_,d,v.scale.Read(),v.patch_a.Read(),v.x.ReadWrite());
  }
  void SharedResidual(int l,bool restrict_rows) const
  {
    const auto& v=levels_[l];
    const int restricted_d=restrict_rows?levels_[l+1].d:0;
    T* out=restrict_rows?levels_[l+1].rhs.Write():v.residual.Write();
    if constexpr(std::is_same<T,float>::value)
      ApplyTensorSharedResidual(ne_,v.d,neighbors_.Read(),v.a.Read(),v.rhs.Read(),
          v.x.Read(),out,precision_==TensorMultigridPrecision::TF32 &&
            residual_mode_==TensorMultigridResidual::Auto,restricted_d);
    else
      ApplyTensorSharedResidual(ne_,v.d,neighbors_.Read(),v.a.Read(),v.rhs.Read(),
          v.x.Read(),out,restricted_d);
    ++shared_residuals_;if(restrict_rows)++shared_restricted_residuals_;
  }
  void Cycle(int l) const
  {
    const auto& v=levels_[l];const int n=ne_*v.d*v.d;
    if(l==2)
    {
      Convert<<<(n+255)/256,256>>>(n,v.rhs.Read(),coarse_rhs_.Write());
      coarse_solver_->Mult(coarse_rhs_,coarse_x_);coarse_iterations_+=coarse_solver_->GetNumIterations();
      if(!coarse_solver_->GetConverged())
      {
        // A finite bounded coarse approximation remains valid for flexible
        // outer GMRES. Only a reported convergence has the p1 true-rBr gate;
        // a cap is explicitly counted, never relabeled as certified.
        bool finite=std::isfinite(coarse_solver_->GetFinalNorm()) && coarse_solver_->GetFinalNorm()>=0;
        if(coarse_solver_->GetNumIterations()==100 && finite)
        {
          const double* values=coarse_x_.HostRead();
          for(int i=0;i<n;++i)finite=finite&&std::isfinite(values[i]);
          if(finite)++coarse_cap_hits_;
        }
        else finite=false;
        if(!finite)++coarse_failures_;
        MFEM_VERIFY(finite,"Tensor MG p1 CG breakdown or nonfinite coarse correction.");
      }
      Convert<<<(n+255)/256,256>>>(n,coarse_x_.Read(),v.x.Write());return;
    }
    CUDA(cudaMemsetAsync(v.x.Write(),0,std::size_t(n)*sizeof(T)));
    Smooth(l,v.rhs);
    const auto& c=levels_[l+1];const int nc=ne_*c.d*c.d;
    if(operator_mode_==TensorMultigridOperator::Shared)SharedResidual(l,true);
    else
    {
      Residual<<<(n+255)/256,256>>>(v.d*v.d,ne_,neighbors_.Read(),v.a.Read(),v.rhs.Read(),v.x.Read(),v.residual.Write());
      Restrict<<<(nc+255)/256,256>>>(ne_,v.d,c.d,v.residual.Read(),c.rhs.Write());
    }
    Cycle(l+1);
    ProlongAdd<<<(nc+255)/256,256>>>(ne_,v.d,c.d,c.x.Read(),v.x.ReadWrite());
    if(operator_mode_==TensorMultigridOperator::Shared)SharedResidual(l,false);
    else Residual<<<(n+255)/256,256>>>(v.d*v.d,ne_,neighbors_.Read(),v.a.Read(),v.rhs.Read(),v.x.Read(),v.residual.Write());
    Smooth(l,v.residual);
  }
  int ne_,norm_blocks_=0;TensorMultigridPrecision precision_;TensorMultigridSmoother smoother_;
  TensorMultigridOperator operator_mode_;
  TensorMultigridResidual residual_mode_;
  int center_warps_,center_split_k_,owned_tile_,owned_tiles_=0,owned_nx_=0,owned_ny_=0;
  double operator_scale_=1;
  std::array<Level<T>,3> levels_;
  mfem::Array<int> neighbors_,patches_,owned_inputs_,owned_outputs_,element_map_,essential_;
  mfem::Vector nodal_;
  mutable mfem::Array<int> exponent_;
  mutable mfem::Vector norm_partial_,coarse_rhs_,coarse_x_;
  std::unique_ptr<mfem::SparseMatrix> coarse_;
  std::unique_ptr<mfem::OperatorJacobiSmoother> jacobi_;
  std::unique_ptr<mfem::IterativeSolver> coarse_solver_;
  bool cooperative_=false;
  mutable unsigned long long applications_=0,coarse_iterations_=0,coarse_failures_=0,coarse_cap_hits_=0,transforms_=0;
  mutable unsigned long long centers_=0,center_packs_=0,center_adds_=0;
  mutable unsigned long long shared_residuals_=0,shared_restricted_residuals_=0;
  mutable unsigned long long owned_submissions_=0;
};

template<typename T> class TensorSpectral final:public mfem::Solver
{
public:
  TensorSpectral(const mfem::Operator& op,mfem::ParFiniteElementSpace& sfes,
      TensorMultigridPrecision precision)
    :mfem::Solver(op.Height()),precision_(precision)
  {
    const auto start=std::chrono::steady_clock::now();int ranks=0;
    MPI_Comm_size(sfes.GetComm(),&ranks);
    MFEM_VERIFY(ranks==1 && mfem::Device::Allows(mfem::Backend::CUDA) &&
        dynamic_cast<const mfem::L2_FECollection*>(sfes.FEColl()) && sfes.GetVDim()==1 &&
        sfes.GetMesh()->Dimension()==2 && op.Width()==height && sfes.GetVSize()==height &&
        sfes.GetTrueVSize()==height && std::size_t(sfes.GetNE())*36==std::size_t(height),
        "Spectral tensor inverse requires rank-one CUDA scalar p5 nodal L2.");
    int device=0;cudaDeviceProp properties{};
    CUDA(cudaGetDevice(&device));CUDA(cudaGetDeviceProperties(&properties,device));
    if(precision_==TensorMultigridPrecision::TF32 || precision_==TensorMultigridPrecision::TF32x3)
    {
      MFEM_VERIFY(properties.major>=8,"Spectral TF32 requires native SM80+.");
      const char* override_value=std::getenv("NVIDIA_TF32_OVERRIDE");
      MFEM_VERIFY(!override_value || std::string(override_value)!="0",
                  "NVIDIA_TF32_OVERRIDE=0 disables the requested spectral TF32 execution.");
    }
    const auto topology=SpectralFaceTopology(sfes);
    const auto coordinates=BuildTensorSpectralTopology(topology);
    if(coordinates.periodic)CheckTranslationTopology(topology);
    MFEM_VERIFY(TensorSpectralLayoutValid(coordinates.nx,coordinates.ny),
                "Spectral dimensions exceed cuSolver/int matrix bounds.");
    nx_=6*coordinates.nx;ny_=6*coordinates.ny;
    MFEM_VERIFY(std::size_t(nx_)*ny_==std::size_t(height),"Spectral Cartesian field size mismatch.");
    const std::size_t matrices=std::size_t(nx_)*nx_+std::size_t(ny_)*ny_;
    const std::size_t fields=std::size_t(height)*(precision_==TensorMultigridPrecision::TF32x3?5:3);
    std::size_t free_bytes=0,total_bytes=0;CUDA(cudaMemGetInfo(&free_bytes,&total_bytes));
    const std::size_t map_bytes=(std::size_t(height)+sfes.GetNE())*sizeof(int)+
        (std::size_t(36*36)+4096)*sizeof(double)+sizeof(int);
    MFEM_VERIFY((matrices*(precision_==TensorMultigridPrecision::TF32x3?2:1)+fields)*sizeof(T)+map_bytes<free_bytes,
                "Spectral persistent basis/field storage exceeds free device memory.");
    mfem::Array<int> neighbors(5*sfes.GetNE());
    int* neighbor=neighbors.HostWrite();
    for(int e=0;e<sfes.GetNE();++e)for(int s=0;s<5;++s)neighbor[5*e+s]=topology[e][s];
    std::vector<int> map;std::vector<double> nodal;int probes=0,colors=0;
    TensorSpectralReference reference;double symmetry_error=0;
    if(coordinates.periodic)
    {
      auto sampled=SharedSnapshot(op,sfes,topology,neighbors,map,nodal,probes,colors);
      operator_scale_=sampled.maximum_diagonal;symmetry_error=sampled.symmetry_error;
      MFEM_VERIFY(operator_scale_>0 && std::isfinite(operator_scale_),"Invalid spectral operator scale.");
      for(double& value:sampled.average)value/=operator_scale_;
      reference=BuildTensorSpectralReference(sampled.average);
    }
    else
    {
      auto sampled=BoundarySpectralSnapshot(op,sfes,topology,coordinates,neighbors,map,nodal,probes,colors);
      operator_scale_=sampled.maximum_diagonal;symmetry_error=sampled.symmetry_error;
      MFEM_VERIFY(operator_scale_>0 && std::isfinite(operator_scale_),"Invalid boundary spectral operator scale.");
      for(auto& stencil:sampled.classes)for(double& value:stencil)value/=operator_scale_;
      reference=BuildTensorSpectralReference(sampled.classes);
    }
    elements_.SetSize(sfes.GetNE());element_map_.SetSize(height);
    std::copy(coordinates.elements.begin(),coordinates.elements.end(),elements_.HostWrite());
    std::copy(map.begin(),map.end(),element_map_.HostWrite());
    Store(nodal_,nodal);
    SpectralBLAS(cublasCreate(&blas_));SpectralBLAS(cublasSetStream(blas_,nullptr));
    SpectralBLAS(cublasSetPointerMode(blas_,CUBLAS_POINTER_MODE_HOST));
    SpectralBLAS(cublasSetMathMode(blas_,precision_==TensorMultigridPrecision::FP64 ||
        precision_==TensorMultigridPrecision::FP32?CUBLAS_PEDANTIC_MATH:CUBLAS_DEFAULT_MATH));
    SpectralSolver(cusolverDnCreate(&solver_));SpectralSolver(cusolverDnSetStream(solver_,nullptr));
    mfem::Array<double> lambda_x,lambda_y;
    Eigensystem(reference.x,nx_,coordinates.periodic,qx_,lambda_x);
    Eigensystem(reference.y,ny_,coordinates.periodic,qy_,lambda_y);
    const double* lx=lambda_x.HostRead();const double* ly=lambda_y.HostRead();
    const double raw_min=lx[0]+ly[0]-reference.mean;
    const double raw_max=lx[nx_-1]+ly[ny_-1]-reference.mean;
    const double eigen_scale=std::max({reference.mass,std::abs(reference.mean)+
        std::abs(lx[0])+std::abs(ly[0]),std::abs(reference.mean)+std::abs(lx[nx_-1])+std::abs(ly[ny_-1])});
    const double screen_roundoff=256*std::numeric_limits<double>::epsilon()*std::max(nx_,ny_)*eigen_scale;
    MFEM_VERIFY(std::isfinite(raw_min) && std::isfinite(raw_max) && raw_min>0 && raw_max>=raw_min &&
        std::isfinite(screen_roundoff) && raw_min>=reference.mass-screen_roundoff,
        "Spectral reference loses positive screening: min="<<raw_min<<" mass="<<reference.mass
          <<" roundoff="<<screen_roundoff);
    inverse_.SetSize(height);mfem::Array<int> inverse_status(2);
    CUDA(cudaMemsetAsync(inverse_status.Write(),0,2*sizeof(int)));
    SpectralInverseKernel<<<(height+255)/256,256>>>(nx_,ny_,lambda_x.Read(),lambda_y.Read(),
        reference.mean,reference.mass,inverse_.Write(),inverse_status.ReadWrite());
    CUDA(cudaGetLastError());const int* status=inverse_status.HostRead();
    MFEM_VERIFY(status[0]==0,"Spectral inverse denominator or selected storage is nonfinite/nonpositive.");
    const int clipped=status[1];
    if constexpr(std::is_same<T,float>::value)
    {
      if(precision_==TensorMultigridPrecision::TF32x3)
      {
        qx_low_.SetSize(qx_.Size());qy_low_.SetSize(qy_.Size());
        float* qx=qx_.ReadWrite();float* qy=qy_.ReadWrite();
        SplitSpectralTF32Kernel<<<(qx_.Size()+255)/256,256>>>(qx_.Size(),qx,qx,qx_low_.Write());
        SplitSpectralTF32Kernel<<<(qy_.Size()+255)/256,256>>>(qy_.Size(),qy,qy,qy_low_.Write());
        high_.SetSize(height);low_.SetSize(height);
        (void)high_.Write();(void)low_.Write();
      }
    }
    field_.SetSize(height);temporary_.SetSize(height);
    (void)field_.Write();(void)temporary_.Write();
    norm_blocks_=std::min(4096,(height+255)/256);
    norm_partial_.SetSize(norm_blocks_);norm_partial_.UseDevice(true);exponent_.SetSize(1);
    (void)norm_partial_.Write();(void)exponent_.Write();(void)elements_.Read();(void)element_map_.Read();
    CUDA(cudaDeviceSynchronize());
    const std::size_t basis_bytes=(std::size_t(qx_.Size())+qy_.Size()+qx_low_.Size()+qy_low_.Size())*sizeof(T);
    const std::size_t field_bytes=(std::size_t(field_.Size())+temporary_.Size()+inverse_.Size()+high_.Size()+low_.Size())*sizeof(T);
    std::cout<<"[igr-tensor-spectral] event=setup precision="<<Name(precision_)
      <<" operator=spectral cycle=global-reference-inverse coarse_execution=none"
      <<" elements="<<sfes.GetNE()<<" n="<<height<<" cells_x="<<coordinates.nx<<" cells_y="<<coordinates.ny
      <<" topology="<<(coordinates.periodic?"periodic-cartesian":"natural-boundary-cartesian")
      <<" reference_classes="<<(coordinates.periodic?1:9)
      <<" matrix_rows="<<nx_<<" matrix_columns="<<ny_<<" operator_probes="<<probes<<" colors="<<colors
      <<" basis_bytes="<<basis_bytes<<" field_workspace_bytes="<<field_bytes
      <<" eigen_workspace_bound_bytes="<<eigen_workspace_peak_bytes_
      <<" shared_symmetry_rel="<<symmetry_error<<" separable_reference_rel="<<reference.relative_defect
      <<" eigen_residual="<<eigen_error_<<" eigen_orthogonality="<<orthogonal_error_
      <<" screen_mass="<<reference.mass<<" spectral_mean="<<reference.mean
      <<" raw_eigen_min="<<raw_min<<" raw_eigen_max="<<raw_max
      <<" screen_roundoff_tolerance="<<screen_roundoff<<" inverse_floor=screen-mass-roundoff-only clipped_eigenvalues="<<clipped
      <<" basis_setup=fp64-cusolver-dsyevd basis_orthogonal_check_gemms=2"
      <<" gemms_per_application="<<(precision_==TensorMultigridPrecision::TF32x3?12:4)
      <<" gemm_compute="<<(precision_==TensorMultigridPrecision::FP64?"fp64":
          precision_==TensorMultigridPrecision::FP32?"fp32-pedantic":"tf32-explicit")
      <<" low_low_term="<<(precision_==TensorMultigridPrecision::TF32x3?"omitted":"not-applicable")
      <<" alpha=inherited-fine acceptance=original-current-fp64 fallback=none"
      // cuBLAS can select IEEE FP64 tensor cores even with pedantic arithmetic.
      <<" tensor_execution=requires-profile-proof"
      <<" setup_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';
  }
  ~TensorSpectral() override
  {
    if(solver_)SpectralSolver(cusolverDnDestroy(solver_));
    if(blas_)SpectralBLAS(cublasDestroy(blas_));
    std::cout<<"[igr-tensor-spectral] event=summary precision="<<Name(precision_)
      <<" applications="<<applications_<<" basis_products="<<4*applications_<<" gemm_calls="<<gemm_calls_
      <<" cycle=global-reference-inverse coarse_execution=none fallback=none\n";
  }
  void SetOperator(const mfem::Operator& op) override
  {MFEM_VERIFY(op.Height()==height && op.Width()==width,"Spectral frozen reference dimensions changed; rebuild required.");}
  mfem::MemoryClass GetMemoryClass() const override{return mfem::MemoryClass::DEVICE;}
  void Mult(const mfem::Vector& b,mfem::Vector& x) const override
  {
    MFEM_VERIFY(b.Size()==height && &b!=&x,"Invalid/aliased spectral input.");
    x.SetSize(height);x.UseDevice(true);
    InputMaximum<<<norm_blocks_,256>>>(height,b.Read(),norm_partial_.Write());
    InputExponent<<<1,1>>>(norm_blocks_,norm_partial_.Read(),exponent_.Write());
    ToSpectralKernel<<<(height+255)/256,256>>>(nx_,ny_,elements_.Read(),element_map_.Read(),
        nodal_.Read(),b.Read(),exponent_.Read(),field_.Write());
    Product(true,true,qx_,qx_low_,field_,temporary_);
    Product(false,false,qy_,qy_low_,temporary_,field_);
    ScaleSpectralKernel<<<(height+255)/256,256>>>(height,inverse_.Read(),field_.ReadWrite());
    Product(true,false,qx_,qx_low_,field_,temporary_);
    Product(false,true,qy_,qy_low_,temporary_,field_);
    FromSpectralKernel<<<(height+255)/256,256>>>(nx_,ny_,elements_.Read(),element_map_.Read(),
        nodal_.Read(),field_.Read(),exponent_.Read(),operator_scale_,x.Write());
    CUDA(cudaGetLastError());++applications_;
  }
private:
  void Eigensystem(const std::vector<double>& host_blocks,int n,bool periodic,mfem::Array<T>& q,
      mfem::Array<double>& lambda)
  {
    const int square=n*n;std::size_t available=0,total=0;
    CUDA(cudaMemGetInfo(&available,&total));
    MFEM_VERIFY(std::size_t(square)*(2*sizeof(double)+sizeof(T))<available,
                "Spectral eigensystem basis/validation storage exceeds free device memory.");
    mfem::Array<double> blocks,matrix(square);Store(blocks,host_blocks);lambda.SetSize(n);
    BuildSpectralMatrix<<<(square+255)/256,256>>>(n,periodic,blocks.Read(),matrix.Write());
    CUDA(cudaGetLastError());
    int workspace=0;
    SpectralSolver(cusolverDnDsyevd_bufferSize(solver_,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
        n,matrix.ReadWrite(),n,lambda.Write(),&workspace));
    MFEM_VERIFY(workspace>0,"Invalid spectral cuSolver workspace size.");
    CUDA(cudaMemGetInfo(&available,&total));
    const std::size_t needed=std::size_t(workspace)*sizeof(double)+std::size_t(square)*(sizeof(double)+sizeof(T));
    MFEM_VERIFY(needed<available,"Spectral cuSolver workspace exceeds free device memory.");
    eigen_workspace_peak_bytes_=std::max(eigen_workspace_peak_bytes_,needed+std::size_t(square)*sizeof(double));
    {
      mfem::Array<double> work(workspace);mfem::Array<int> info(1);
      SpectralSolver(cusolverDnDsyevd(solver_,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
          n,matrix.ReadWrite(),n,lambda.Write(),work.Write(),workspace,info.Write()));
      MFEM_VERIFY(info.HostRead()[0]==0,"Spectral cuSolver eigensystem did not converge.");
    }
    const int groups=std::min(4096,(square+255)/256);
    mfem::Array<double> partial(2*groups),gram(square);mfem::Array<int> invalid(1);
    CUDA(cudaMemsetAsync(invalid.Write(),0,sizeof(int)));
    CheckSpectralEigenKernel<<<groups,256>>>(n,periodic,blocks.Read(),matrix.Read(),lambda.Read(),partial.Write(),invalid.ReadWrite());
    CUDA(cudaGetLastError());
    MFEM_VERIFY(invalid.HostRead()[0]==0,"Spectral eigensystem contains a nonfinite value or unordered eigenvalue.");
    const double* residual=partial.HostRead();double error=0,scale=0;
    for(int i=0;i<groups;++i){error+=residual[2*i];scale+=residual[2*i+1];}
    const double eigen=std::sqrt(error/std::max(scale,1e-300));
    MFEM_VERIFY(scale>0 && std::isfinite(eigen) && eigen<1e-10,"Spectral eigen residual failed: "<<eigen);
    const double one=1,zero=0;
    SpectralBLAS(cublasDgemm(blas_,CUBLAS_OP_T,CUBLAS_OP_N,n,n,n,&one,matrix.Read(),n,
        matrix.Read(),n,&zero,gram.Write(),n));
    CheckSpectralOrthogonalKernel<<<groups,256>>>(n,gram.Read(),partial.Write());
    CUDA(cudaGetLastError());
    const double* orthogonal=partial.HostRead();double orthogonal_sum=0;
    for(int i=0;i<groups;++i)orthogonal_sum+=orthogonal[i];
    const double orthogonal_norm=std::sqrt(orthogonal_sum/n);
    MFEM_VERIFY(std::isfinite(orthogonal_norm) && orthogonal_norm<1e-10,
                "Spectral basis orthogonality failed: "<<orthogonal_norm);
    eigen_error_=std::max(eigen_error_,eigen);orthogonal_error_=std::max(orthogonal_error_,orthogonal_norm);
    q.SetSize(square);Convert<<<(square+255)/256,256>>>(square,matrix.Read(),q.Write());
    CUDA(cudaGetLastError());
  }
  void FloatProduct(bool left,bool transpose,const float* q,const float* source,
      float* target,float beta,cublasComputeType_t compute) const
  {
    const float one=1;
    SpectralBLAS(cublasGemmEx(blas_,left && transpose?CUBLAS_OP_T:CUBLAS_OP_N,
        !left && transpose?CUBLAS_OP_T:CUBLAS_OP_N,nx_,ny_,left?nx_:ny_,&one,
        left?q:source,CUDA_R_32F,nx_,left?source:q,CUDA_R_32F,left?nx_:ny_,&beta,
        target,CUDA_R_32F,nx_,compute,CUBLAS_GEMM_DEFAULT));
    ++gemm_calls_;
  }
  void Product(bool left,bool transpose,const mfem::Array<T>& q,const mfem::Array<T>& q_low,
      const mfem::Array<T>& source,mfem::Array<T>& target) const
  {
    if constexpr(std::is_same<T,double>::value)
    {
      const double one=1,zero=0;
      SpectralBLAS(cublasDgemm(blas_,left && transpose?CUBLAS_OP_T:CUBLAS_OP_N,
          !left && transpose?CUBLAS_OP_T:CUBLAS_OP_N,nx_,ny_,left?nx_:ny_,&one,
          left?q.Read():source.Read(),nx_,left?source.Read():q.Read(),left?nx_:ny_,&zero,target.Write(),nx_));
      ++gemm_calls_;
    }
    else if(precision_==TensorMultigridPrecision::TF32x3)
    {
      SplitSpectralTF32Kernel<<<(height+255)/256,256>>>(height,source.Read(),high_.Write(),low_.Write());
      FloatProduct(left,transpose,q.Read(),high_.Read(),target.Write(),0,CUBLAS_COMPUTE_32F_FAST_TF32);
      FloatProduct(left,transpose,q.Read(),low_.Read(),target.ReadWrite(),1,CUBLAS_COMPUTE_32F_FAST_TF32);
      FloatProduct(left,transpose,q_low.Read(),high_.Read(),target.ReadWrite(),1,CUBLAS_COMPUTE_32F_FAST_TF32);
    }
    else FloatProduct(left,transpose,q.Read(),source.Read(),target.Write(),0,
        precision_==TensorMultigridPrecision::FP32?CUBLAS_COMPUTE_32F_PEDANTIC:CUBLAS_COMPUTE_32F_FAST_TF32);
  }
  int nx_=0,ny_=0,norm_blocks_=0;TensorMultigridPrecision precision_;
  double operator_scale_=1,eigen_error_=0,orthogonal_error_=0;
  std::size_t eigen_workspace_peak_bytes_=0;
  cublasHandle_t blas_=nullptr;cusolverDnHandle_t solver_=nullptr;
  mfem::Array<int> elements_,element_map_;
  mfem::Array<double> nodal_;
  mfem::Array<T> qx_,qy_,qx_low_,qy_low_,inverse_;
  mutable mfem::Array<T> field_,temporary_,high_,low_;
  mutable mfem::Vector norm_partial_;
  mutable mfem::Array<int> exponent_;
  mutable unsigned long long applications_=0,gemm_calls_=0;
};
} // namespace

void ApplyTensorOwnedResponse(int tiles,int ne,int d,const int* owned,
    const double* response,const double* packed,const double* factor,double* x)
{
  MFEM_VERIFY(TensorOwnedLayoutValid(tiles,ne,d) && owned && response && packed && factor && x &&
      x!=response && x!=packed && x!=factor,"Invalid owned tensor response layout/storage.");
  const int m=16*d*d,blocks=(m/64)*((tiles+63)/64);
  IGRTensorMultigridOwnedSIMTKernel<<<blocks,256>>>(tiles,d,owned,response,packed,factor,x);
  CUDA(cudaGetLastError());
}

void ApplyTensorOwnedResponse(int tiles,int ne,int d,const int* owned,
    const float* response,const float* packed,const float* factor,float* x,bool tf32)
{
  MFEM_VERIFY(TensorOwnedLayoutValid(tiles,ne,d) && owned && response && packed && factor && x &&
      x!=response && x!=packed && x!=factor,"Invalid owned tensor response layout/storage.");
  const int m=16*d*d,blocks=(m/64)*((tiles+63)/64);
  if(tf32)IGRTensorMultigridOwnedTF32Kernel<<<blocks,128>>>(tiles,d,owned,response,packed,factor,x);
  else IGRTensorMultigridOwnedSIMTKernel<<<blocks,256>>>(tiles,d,owned,response,packed,factor,x);
  CUDA(cudaGetLastError());
}

void ApplyTensorSharedResidual(int ne,int d,const int* neighbors,
    const double* stencil,const double* rhs,const double* x,double* residual,
    int restricted_d)
{
  MFEM_VERIFY(TensorSharedResidualLayoutValid(ne,d,restricted_d) &&
      neighbors && stencil && rhs && x && residual && residual!=rhs && residual!=x && residual!=stencil,
      "Invalid shared tensor residual layout/storage.");
  const int output_d=restricted_d?restricted_d:d,m=output_d*output_d;
  IGRTensorMultigridSharedResidualSIMTKernel<<<(ne*m+255)/256,256>>>(
      ne,d,output_d,neighbors,stencil,rhs,x,residual);
  CUDA(cudaGetLastError());
}

void ApplyTensorSharedResidual(int ne,int d,const int* neighbors,
    const float* stencil,const float* rhs,const float* x,float* residual,
    bool tf32,int restricted_d)
{
  MFEM_VERIFY(TensorSharedResidualLayoutValid(ne,d,restricted_d) &&
      neighbors && stencil && rhs && x && residual && residual!=rhs && residual!=x && residual!=stencil,
      "Invalid shared tensor residual layout/storage.");
  const int output_d=restricted_d?restricted_d:d,m=output_d*output_d;
  if(tf32)
  {
    const int tiles_m=(m+15)/16,groups=(ne+63)/64;
    IGRTensorMultigridSharedResidualTF32Kernel<<<tiles_m*groups,128>>>(
        ne,d,output_d,neighbors,stencil,rhs,x,residual);
  }
  else IGRTensorMultigridSharedResidualSIMTKernel<<<(ne*m+255)/256,256>>>(
      ne,d,output_d,neighbors,stencil,rhs,x,residual);
  CUDA(cudaGetLastError());
}

void ApplyTensorCenterResponse(int ne,int d,const double* response,
    const double* packed,double* partial,int warps,int split_k)
{
  CheckCenterLayout(ne,d,warps,split_k);
  MFEM_VERIFY(response&&packed&&partial,"Null tensor center response storage.");
  const int threads=32*warps;
  IGRTensorMultigridCenterSIMTKernel<<<dim3((ne*d*d+threads-1)/threads,split_k),threads>>>(
      ne,d,response,packed,partial,split_k);
  CUDA(cudaGetLastError());
}

void ApplyTensorCenterResponse(int ne,int d,const float* response,
    const float* packed,float* partial,bool tf32,int warps,int split_k)
{
  CheckCenterLayout(ne,d,warps,split_k);
  MFEM_VERIFY(response&&packed&&partial,"Null tensor center response storage.");
  if(tf32)
  {
    const int tiles_m=(d*d+15)/16,groups=((ne+15)/16+warps-1)/warps;
    const dim3 grid(tiles_m*groups,split_k);
    if(warps==1)IGRTensorMultigridCenterTF32Kernel<1><<<grid,32>>>(ne,d,response,packed,partial,split_k);
    else IGRTensorMultigridCenterTF32Kernel<4><<<grid,128>>>(ne,d,response,packed,partial,split_k);
  }
  else
  {
    const int threads=32*warps;
    IGRTensorMultigridCenterSIMTKernel<<<dim3((ne*d*d+threads-1)/threads,split_k),threads>>>(
        ne,d,response,packed,partial,split_k);
  }
  CUDA(cudaGetLastError());
}

void AddTensorCenterResponse(int ne,int d,const double* factor,
    const double* partial,double* x,int split_k)
{
  CheckCenterLayout(ne,d,1,split_k);
  MFEM_VERIFY(factor&&partial&&x,"Null tensor center add storage.");
  IGRTensorMultigridAddCenterKernel<<<(ne*d*d+255)/256,256>>>(ne,d,factor,partial,x,split_k);
  CUDA(cudaGetLastError());
}

void AddTensorCenterResponse(int ne,int d,const float* factor,
    const float* partial,float* x,int split_k)
{
  CheckCenterLayout(ne,d,1,split_k);
  MFEM_VERIFY(factor&&partial&&x,"Null tensor center add storage.");
  IGRTensorMultigridAddCenterKernel<<<(ne*d*d+255)/256,256>>>(ne,d,factor,partial,x,split_k);
  CUDA(cudaGetLastError());
}

std::unique_ptr<mfem::Solver> MakeTensorMultigridPreconditioner(
    const mfem::Operator& op,mfem::ParFiniteElementSpace& sfes,TensorMultigridPrecision precision,
    TensorMultigridSmoother smoother,int center_warps,int center_split_k,
    TensorMultigridOperator operator_mode,int owned_tile,TensorMultigridResidual residual_mode)
{
  if(operator_mode==TensorMultigridOperator::Spectral)
  {
    MFEM_VERIFY(smoother==TensorMultigridSmoother::Separable && center_warps==1 &&
        center_split_k==1 && owned_tile==1 && residual_mode==TensorMultigridResidual::Auto,
        "Spectral tensor inverse requires default MG-only smoother/layout/residual controls.");
    if(precision==TensorMultigridPrecision::FP64)
      return std::make_unique<TensorSpectral<double>>(op,sfes,precision);
    MFEM_VERIFY(precision==TensorMultigridPrecision::FP32 || precision==TensorMultigridPrecision::TF32 ||
        precision==TensorMultigridPrecision::TF32x3,"Invalid spectral tensor precision.");
    return std::make_unique<TensorSpectral<float>>(op,sfes,precision);
  }
  MFEM_VERIFY((residual_mode==TensorMultigridResidual::Auto || residual_mode==TensorMultigridResidual::FP32) &&
      (residual_mode==TensorMultigridResidual::Auto ||
        (operator_mode==TensorMultigridOperator::Shared && precision==TensorMultigridPrecision::TF32)),
      "Tensor MG FP32 residual placement requires shared TF32 multigrid.");
  MFEM_VERIFY((owned_tile==1 || owned_tile==4) &&
      (owned_tile==1 || (operator_mode==TensorMultigridOperator::Shared &&
        smoother==TensorMultigridSmoother::Center && center_warps==1 && center_split_k==1)),
      "Tensor MG owned_tile=4 requires shared operator, center smoother and default center layout controls.");
  MFEM_VERIFY(operator_mode==TensorMultigridOperator::Sampled||operator_mode==TensorMultigridOperator::Shared,
              "Invalid tensor MG operator mode.");
  MFEM_VERIFY(smoother==TensorMultigridSmoother::Separable||smoother==TensorMultigridSmoother::Center,
              "Invalid tensor MG smoother.");
  CheckCenterLayout(sfes.GetNE(),6,center_warps,center_split_k);
  MFEM_VERIFY(smoother==TensorMultigridSmoother::Center||(center_warps==1&&center_split_k==1),
              "Tensor center layout options require the center smoother.");
  if(precision==TensorMultigridPrecision::FP64)
    return std::make_unique<TensorMultigrid<double>>(op,sfes,precision,smoother,center_warps,center_split_k,operator_mode,owned_tile,residual_mode);
  MFEM_VERIFY(precision==TensorMultigridPrecision::FP32||precision==TensorMultigridPrecision::TF32,
              "Tensor MG precision must be FP64, FP32 or TF32; TF32x3 requires spectral mode.");
  return std::make_unique<TensorMultigrid<float>>(op,sfes,precision,smoother,center_warps,center_split_k,operator_mode,owned_tile,residual_mode);
}
} }

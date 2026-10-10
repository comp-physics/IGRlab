#pragma once
// Controlled transport-only research replacements. The stress/source solver,
// physical state map, application boundary flux and derivative penalties remain
// owned by the existing front probe. No production DGSEM qualification implied.
#include "numerics/dgsem/dgsem_entropy_flux.hpp"

inline std::array<double,4> FrontThermalRanocha(const std::array<double,4>& a,
                                               const std::array<double,4>& b,
                                               double sa,double sb)
{
  mfem::Vector left(const_cast<double*>(a.data()),4),right(const_cast<double*>(b.data()),4),f;
  // Thermal Euler flux must receive zero stress: adding sigma to its pressure
  // changes its equal-state energy flux by u*sigma/(gamma-1).
  mfem::dgsem::RanochaFlux2D(left,right,0,gamma_euler,0.,0.,f);
  const double s=.5*(sa+sb),v=.5*(a[1]/a[0]+b[1]/b[0]);
  return {{f[0],f[1]+s,f[2],f[3]+s*v}};
}

inline State FrontAlternativeTransport(const PrimitiveMomentProblem& pr,
                                      const std::vector<State>& low_derivatives,
                                      const State& physical,const State& sigma,
                                      const std::string& form,TrackWork& work,
                                      std::array<double,4>& budget)
{
  const int m=pr.retained.n,ne=pr.ne;const auto& weights=pr.retained.physics.basis.weights;
  State low(ne*m*4,0.);
  if(form=="weak-lumped")
  {
    const State high=PrimitiveDirectResidual(pr,low_derivatives,physical,sigma,work,budget);
    const State exact=pr.state_transfer.Apply(pr.state_transfer.P,high,ne,4);
    for(int e=0;e<ne;++e)for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<4;++c)
      low[(e*m+i)*4+c]+=pr.retained.mass(i,j)*exact[(e*m+j)*4+c]/weights[i];
  }
  else
  {
    if(form!="dgsem-central"&&form!="dgsem-ranocha")throw std::runtime_error("unknown alternate transport form");
    const auto start=Clock::now();std::vector<State> shape(m),derivative(m);
    for(int i=0;i<m;++i)
    {shape[i]=pr.Shape(pr.retained.physics.basis.nodes[i],false);derivative[i]=pr.retained.Shape(pr.retained.physics.basis.nodes[i],true);}
    std::vector<std::array<double,4>> u(ne*m),f(ne*m);State s(ne*m);
    for(int e=0;e<ne;++e)for(int i=0;i<m;++i)
    {
      const int j=e*m+i;u[j]=pr.Point(physical,e,shape[i]);
      for(int k=0;k<pr.n;++k)s[j]+=shape[i][k]*sigma[e*pr.n+k];
      double speed=0.;f[j]=pr.physics.Flux(u[j].data(),s[j],speed);
    }
    for(int e=0;e<ne;++e)
    {
      const double h=pr.edges[e+1]-pr.edges[e];
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)
      {
        std::array<double,4> pair{};
        if(form=="dgsem-ranocha")pair=FrontThermalRanocha(u[e*m+i],u[e*m+j],s[e*m+i],s[e*m+j]);
        else for(int c=0;c<4;++c)pair[c]=.5*(f[e*m+i][c]+f[e*m+j][c]);
        for(int c=0;c<4;++c)low[(e*m+i)*4+c]-=2.*derivative[i][j]*pair[c]/h;
        ++work.flux_points;
      }
    }
    budget.fill(0.);
    for(int face=0;face<=ne;++face)
    {
      std::array<double,4> hat{};
      if(face==0||face==ne)hat=pr.BoundaryFlux(physical,sigma,face==ne);
      else
      {
        const int jl=face*m-1,jr=face*m;double al=0.,ar=0.;
        pr.physics.Flux(u[jl].data(),s[jl],al);pr.physics.Flux(u[jr].data(),s[jr],ar);
        const double a=pr.physics.FaceSpeed(al,ar);
        for(int c=0;c<4;++c)hat[c]=.5*(f[jl][c]+f[jr][c]-a*(u[jr][c]-u[jl][c]));
      }
      for(int c=0;c<4;++c)
      {
        if(face){const int j=face*m-1;low[4*j+c]+=(f[j][c]-hat[c])/((pr.edges[face]-pr.edges[face-1])*weights[m-1]);}
        if(face<ne){const int j=face*m;low[4*j+c]+=(hat[c]-f[j][c])/((pr.edges[face+1]-pr.edges[face])*weights[0]);}
        if(face==0)budget[c]=hat[c];if(face==ne)budget[c]-=hat[c];
      }
      ++work.faces;
    }
    work.transport+=Wall(start);
  }
  const State result=pr.state_transfer.Apply(pr.state_transfer.E,low,ne,4);
  const auto sums=pr.Integrals(result,pr.edges);
  for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(sums[c]-budget[c]));
  return result;
}

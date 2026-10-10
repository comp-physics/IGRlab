#pragma once
#include "mfem.hpp"
#include <cmath>

// Test-only homothetic polynomial annulus. The nodes lie on concentric circles;
// the interpolated edges generally do not. Keep that geometric error explicit.
namespace igr_wall_consistency
{
inline mfem::Mesh Annulus(int radial,int angular,int geometry_order=2)
{
  MFEM_VERIFY(radial>=1&&angular>=4&&geometry_order>=1,"Invalid annulus resolution.");
  const double pi=std::acos(-1.),delta=2*pi/angular;
  mfem::Mesh mesh(2,(radial+1)*angular,radial*angular,2*angular,2);
  const auto vertex=[=](int i,int j){return i+(radial+1)*(j%angular);};
  for(int j=0;j<angular;++j)for(int i=0;i<=radial;++i)
  {
    const double r=.5+.5*i/radial,theta=delta*j;
    mesh.AddVertex(r*std::cos(theta),r*std::sin(theta));
  }
  for(int j=0;j<angular;++j)for(int i=0;i<radial;++i)
    mesh.AddQuad(vertex(i,j),vertex(i+1,j),vertex(i+1,j+1),vertex(i,j+1),1);
  for(int j=0;j<angular;++j)
  {
    mesh.AddBdrSegment(vertex(0,j+1),vertex(0,j),1);
    mesh.AddBdrSegment(vertex(radial,j),vertex(radial,j+1),2);
  }
  mesh.FinalizeQuadMesh(1,0,true);
  mesh.SetCurvature(geometry_order,false,2,mfem::Ordering::byVDIM);
  mesh.Transform([=](const mfem::Vector& x,mfem::Vector& y)
  {
    double theta=std::atan2(x[1],x[0]);if(theta<0)theta+=2*pi;
    const int sector=std::min(angular-1,int(theta/delta));
    const double factor=std::cos(theta-(sector+.5)*delta)/std::cos(.5*delta);
    y.SetSize(2);y[0]=factor*x[0];y[1]=factor*x[1];
  });
  return mesh;
}

inline void Rotation(const mfem::Vector& x,mfem::Vector& u,double omega=1.,double gamma=1.4)
{
  const double speed2=omega*omega*(x*x),pressure=1.+.5*speed2;
  u.SetSize(4);u[0]=1.;u[1]=-omega*x[1];u[2]=omega*x[0];
  u[3]=pressure/(gamma-1.)+.5*speed2;
}
}

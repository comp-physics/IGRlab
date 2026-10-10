#pragma once

// Shared TEST-ONLY strong-front fixture. This composes the production normal
// shock initializer with a smooth periodic coordinate. There is one compression
// and one reverse front; it is not an exact traveling shock solution of IGR.
#include "../../examples/shock_vortex_interaction/shock_vortex_interaction.hpp"
#include <cmath>
#include <functional>

inline std::function<void(double, mfem::Vector&)>
MakeHighOrderPeriodicFront(double width=0.012, double mach=5.,
                          double speed=0.4, double phase=0.13)
{
  auto physical=mfem::GetShockVortexInoueHattoriInitCustom(
      1.4,1,1,mach,0.,width,-0.5,0.5,-0.5,0.5,0.,0.,0.);
  return [physical,speed,phase](double x,mfem::Vector& u)
  {
    const double pi=std::acos(-1.);
    mfem::Vector at(2);at[0]=std::sin(2*pi*(x-phase))/(2*pi);at[1]=0.;
    physical(at,u);
    u[3]+=speed*u[1]+0.5*speed*speed*u[0];u[1]+=speed*u[0];
  };
}

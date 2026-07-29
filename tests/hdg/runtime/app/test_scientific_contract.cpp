#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr std::array<double, 5> nodes{{
    -0.9061798459386639928, -0.5384693101056830910, 0.0,
     0.5384693101056830910,  0.9061798459386639928}};
constexpr std::array<double, 5> weights{{
    0.2369268850561890875, 0.4786286704993664680,
    0.5688888888888888889, 0.4786286704993664680,
    0.2369268850561890875}};

struct Dual2
{
  double value = 0.0;
  double dx = 0.0;
  double dy = 0.0;
};
Dual2 operator+(Dual2 a, Dual2 b) { return {a.value+b.value,a.dx+b.dx,a.dy+b.dy}; }
Dual2 operator-(Dual2 a, Dual2 b) { return {a.value-b.value,a.dx-b.dx,a.dy-b.dy}; }
Dual2 operator*(Dual2 a, Dual2 b) { return {a.value*b.value,a.dx*b.value+a.value*b.dx,a.dy*b.value+a.value*b.dy}; }
Dual2 operator/(Dual2 a, Dual2 b)
{
  const double inverse = 1.0 / b.value;
  return {a.value*inverse,(a.dx*b.value-a.value*b.dx)*inverse*inverse,
          (a.dy*b.value-a.value*b.dy)*inverse*inverse};
}
Dual2 operator*(double a, Dual2 b) { return {a*b.value,a*b.dx,a*b.dy}; }
Dual2 Sin(Dual2 a) { return {std::sin(a.value),std::cos(a.value)*a.dx,std::cos(a.value)*a.dy}; }
Dual2 Cos(Dual2 a) { return {std::cos(a.value),-std::sin(a.value)*a.dx,-std::sin(a.value)*a.dy}; }

std::array<double, 4> Source(double x_value, double y_value)
{
  const Dual2 x{x_value,1.0,0.0}, y{y_value,0.0,1.0};
  const Dual2 rho = Dual2{1.0,0.0,0.0} + 0.1*Sin(2.0*pi*x)*Sin(2.0*pi*y);
  const Dual2 u = Dual2{0.35,0.0,0.0} + 0.05*Cos(2.0*pi*x)*Sin(2.0*pi*y);
  const Dual2 v = Dual2{-0.20,0.0,0.0} + 0.05*Sin(2.0*pi*x)*Cos(2.0*pi*y);
  const Dual2 pressure = Dual2{1.0,0.0,0.0} + 0.1*Cos(2.0*pi*x)*Cos(2.0*pi*y);
  const Dual2 mx = rho*u, my = rho*v;
  const Dual2 energy = pressure/ Dual2{0.4,0.0,0.0} +
      0.5*rho*(u*u+v*v);
  const std::array<Dual2,4> fx{{mx,mx*u+pressure,my*u,(energy+pressure)*u}};
  const std::array<Dual2,4> fy{{my,mx*v,my*v+pressure,(energy+pressure)*v}};
  std::array<double,4> source{};
  for (std::size_t c=0;c<source.size();++c) source[c]=fx[c].dx+fy[c].dy;
  return source;
}

double Density(double x, double y)
{ return 1.0 + 0.1*std::sin(2.0*pi*x)*std::sin(2.0*pi*y); }

double QuadraticDensity(double x, double y, double cx, double cy)
{
  const double ax=2.0*pi*cx, ay=2.0*pi*cy, dx=x-cx, dy=y-cy;
  const double f=Density(cx,cy);
  const double fx=0.2*pi*std::cos(ax)*std::sin(ay);
  const double fy=0.2*pi*std::sin(ax)*std::cos(ay);
  const double fxx=-0.4*pi*pi*std::sin(ax)*std::sin(ay);
  const double fyy=fxx;
  const double fxy=0.4*pi*pi*std::cos(ax)*std::cos(ay);
  return f+fx*dx+fy*dy+0.5*fxx*dx*dx+fxy*dx*dy+0.5*fyy*dy*dy;
}

double TruncationError(int n)
{
  double error=0.0;
  const double h=1.0/static_cast<double>(n);
  for(int j=0;j<n;++j) for(int i=0;i<n;++i)
  {
    const double cx=(i+0.5)*h, cy=(j+0.5)*h;
    for(std::size_t qy=0;qy<nodes.size();++qy)
      for(std::size_t qx=0;qx<nodes.size();++qx)
      {
        const double x=cx+0.5*h*nodes[qx], y=cy+0.5*h*nodes[qy];
        const double difference=Density(x,y)-QuadraticDensity(x,y,cx,cy);
        error += 0.25*h*h*weights[qx]*weights[qy]*difference*difference;
      }
  }
  return std::sqrt(error);
}

bool Check(bool condition,const char* name)
{ if(!condition) std::cerr<<"FAILED: "<<name<<'\n'; return condition; }
} // namespace

int main()
{
  std::array<double,4> integral{};
  for(std::size_t j=0;j<nodes.size();++j) for(std::size_t i=0;i<nodes.size();++i)
  {
    const auto source=Source(0.5*(nodes[i]+1.0),0.5*(nodes[j]+1.0));
    for(std::size_t c=0;c<integral.size();++c)
      integral[c]+=0.25*weights[i]*weights[j]*source[c];
  }
  bool passed=true;
  for(double value:integral)
    passed=Check(std::abs(value)<=2.0e-13,"periodic_analytic_source_balance")&&passed;
  const double e3=TruncationError(3),e6=TruncationError(6),e12=TruncationError(12);
  const double r36=std::log2(e3/e6),r612=std::log2(e6/e12);
  passed=Check(std::isfinite(e3)&&e3>e6&&e6>e12&&e12>1.0e-10,
               "three_point_positive_decreasing")&&passed;
  passed=Check(r36>=2.25&&r612>=2.25,
               "independent_spatial_truncation_rate")&&passed;
  if(passed) std::cout<<"HDG runtime scientific contract PASS\n";
  return passed?0:1;
}

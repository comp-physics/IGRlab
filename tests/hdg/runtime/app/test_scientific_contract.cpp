#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

namespace
{
constexpr double pi = 3.141592653589793238462643383279502884;
// The p2 functional card freezes the disjoint continuous oracle at qO=26.
// MFEM's order-26 one-dimensional Gauss rule has 14 points; hard-code that
// independent rule here instead of silently using the old five-point smoke.
constexpr std::array<double, 14> nodes{{
    -0.98628380869681233884, -0.92843488366357351734,
    -0.82720131506976499319, -0.68729290481168547015,
    -0.51524863635815409197, -0.31911236892788976044,
    -0.10805494870734366207,  0.10805494870734366207,
     0.31911236892788976044,  0.51524863635815409197,
     0.68729290481168547015,  0.82720131506976499319,
     0.92843488366357351734,  0.98628380869681233884}};
constexpr std::array<double, 14> weights{{
    0.03511946033175186303, 0.08015808715976020981,
    0.12151857068790318469, 0.15720316715819353457,
    0.18553839747793781374, 0.20519846372129560397,
    0.21526385346315779020, 0.21526385346315779020,
    0.20519846372129560397, 0.18553839747793781374,
    0.15720316715819353457, 0.12151857068790318469,
    0.08015808715976020981, 0.03511946033175186303}};
constexpr int source_balance_mesh_n = 6;

struct CompensatedSum
{
  double sum = 0.0;
  double correction = 0.0;
  void Add(const double value) noexcept
  {
    const double adjusted = value - correction;
    const double next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
};

struct Dual2
{
  double value = 0.0;
  double dx = 0.0;
  double dy = 0.0;
};
Dual2 operator+(Dual2 a, Dual2 b) { return {a.value+b.value,a.dx+b.dx,a.dy+b.dy}; }
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
  // qO=26 is applied element-by-element on the frozen p2/N6 functional mesh,
  // not once over an artificial N=1 domain. The latter under-resolves the
  // highest analytic energy-flux harmonic and is not the card's oracle.
  std::array<CompensatedSum,4> integral{};
  const double h=1.0/static_cast<double>(source_balance_mesh_n);
  for(int ey=0;ey<source_balance_mesh_n;++ey)
    for(int ex=0;ex<source_balance_mesh_n;++ex)
      for(std::size_t j=0;j<nodes.size();++j)
        for(std::size_t i=0;i<nodes.size();++i)
  {
    const double x=(static_cast<double>(ex)+0.5*(nodes[i]+1.0))*h;
    const double y=(static_cast<double>(ey)+0.5*(nodes[j]+1.0))*h;
    const auto source=Source(x,y);
    for(std::size_t c=0;c<integral.size();++c)
      integral[c].Add(0.25*h*h*weights[i]*weights[j]*source[c]);
  }
  bool passed=true;
  for(std::size_t component=0;component<integral.size();++component)
  {
    const bool balanced=std::abs(integral[component].sum)<=2.0e-13;
    if(!balanced)
      std::cerr<<"periodic source component="<<component
               <<" integral="<<std::setprecision(17)
               <<integral[component].sum<<" tolerance=2e-13\n";
    passed=Check(balanced,"periodic_analytic_source_balance")&&passed;
  }
  const double e3=TruncationError(3),e6=TruncationError(6),e12=TruncationError(12);
  const double r36=std::log2(e3/e6),r612=std::log2(e6/e12);
  passed=Check(std::isfinite(e3)&&e3>e6&&e6>e12&&e12>1.0e-10,
               "three_point_positive_decreasing")&&passed;
  passed=Check(r36>=2.25&&r612>=2.25,
               "independent_spatial_truncation_rate")&&passed;
  if(passed) std::cout<<"HDG runtime scientific contract PASS\n";
  return passed?0:1;
}

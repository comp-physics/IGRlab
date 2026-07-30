#include "numerics/hdg/runtime/app/application_contract.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

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

bool Check(bool condition,const char* name)
{ if(!condition) std::cerr<<"FAILED: "<<name<<'\n'; return condition; }

struct VtkField
{
  std::size_t points = 0;
  std::size_t cells = 0;
  std::vector<double> coordinates;
  std::array<std::vector<double>, 4> conservative;
};

bool ParsePhysicalVtk(const char* path, VtkField& field)
{
  std::ifstream input(path);
  if(!input) return false;
  std::vector<std::string> tokens;
  for(std::string token; input>>token;) tokens.push_back(std::move(token));
  auto find=[](const std::vector<std::string>& values,
               const std::string& token, std::size_t begin)
  {
    for(std::size_t index=begin;index<values.size();++index)
      if(values[index]==token) return index;
    return values.size();
  };
  try
  {
    const std::size_t points=find(tokens,"POINTS",0);
    const std::size_t cells=find(tokens,"CELLS",0);
    if(points==tokens.size()||cells==tokens.size()||points+3>tokens.size()||
       cells+2>tokens.size()) return false;
    field.points=static_cast<std::size_t>(std::stoull(tokens[points+1]));
    field.cells=static_cast<std::size_t>(std::stoull(tokens[cells+1]));
    field.coordinates.resize(3*field.points);
    for(std::size_t index=0;index<field.coordinates.size();++index)
      field.coordinates[index]=std::stod(tokens[points+3+index]);
    for(std::size_t component=0;component<4;++component)
    {
      const std::string name="conservative"+std::to_string(component);
      std::size_t scalar=find(tokens,"SCALARS",0);
      while(scalar<tokens.size() &&
            (scalar+1>=tokens.size()||tokens[scalar+1]!=name))
        scalar=find(tokens,"SCALARS",scalar+1);
      if(scalar==tokens.size()) return false;
      const std::size_t lookup=find(tokens,"LOOKUP_TABLE",scalar+1);
      if(lookup==tokens.size()||lookup+2+field.points>tokens.size())
        return false;
      auto& values=field.conservative[component];
      values.resize(field.points);
      for(std::size_t index=0;index<field.points;++index)
        values[index]=std::stod(tokens[lookup+2+index]);
    }
  }
  catch(...) { return false; }
  return field.points>0&&field.cells>0;
}

bool ComparePhysicalVtk(const char* left_path,const char* right_path,
                        const double tolerance)
{
  VtkField left,right;
  if(!ParsePhysicalVtk(left_path,left)||!ParsePhysicalVtk(right_path,right)||
     left.points!=right.points||left.cells!=right.cells||
     left.coordinates.size()!=right.coordinates.size()) return false;
  for(std::size_t index=0;index<left.coordinates.size();++index)
    if(std::abs(left.coordinates[index]-right.coordinates[index])>1.0e-14)
      return false;
  for(std::size_t component=0;component<4;++component)
    for(std::size_t index=0;index<left.points;++index)
    {
      const double a=left.conservative[component][index];
      const double b=right.conservative[component][index];
      const double scale=std::max({1.0,std::abs(a),std::abs(b)});
      if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>tolerance*scale)
        return false;
    }
  return true;
}
} // namespace

int main(int argc,char** argv)
{
  if(argc==5&&std::string{argv[1]}=="--compare-vtk")
  {
    double tolerance=0.0;
    try { tolerance=std::stod(argv[4]); }
    catch(...) { return 64; }
    const bool compared=tolerance>0.0&&
        ComparePhysicalVtk(argv[2],argv[3],tolerance);
    if(compared) std::cout<<"HDG runtime CPU-CUDA VTK parity PASS\n";
    return compared?0:1;
  }
  if(argc!=1) return 64;
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
  // Spatial order is admitted only by the separate solver-linked N3/N6/N12
  // leaf, which consumes committed HDG fields.  This leaf owns only the
  // independent analytic periodic-source balance.
  if(passed) std::cout<<"HDG runtime scientific contract PASS\n";
  return passed?0:1;
}

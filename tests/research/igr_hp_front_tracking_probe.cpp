// Test-only selective moving high-order DG partition for full-energy Euler/IGR.
// The continuum model, current-density elliptic coefficients and source algebra
// are shared with the application. All pieces retain degree p and Rusanov faces.
// Mesh motion is conservative L2 rezoning after accepted SSPRK3 steps, NOT ALE.
// Exact reference mass is used in transport and remap; this differs from DGSEM
// lumping. Only a fixed-size patch changes geometry; actual pieces are counted.
#define main igr_front_tracking_existing_screened_main
#include "screened_spacetime_probe.cpp"
#undef main
#define DG_IGR_HIGH_ORDER_CONDENSATION_LIBRARY_ONLY
#include "igr_high_order_condensation_probe.cpp"
#undef DG_IGR_HIGH_ORDER_CONDENSATION_LIBRARY_ONLY
#include <numeric>

namespace
{
struct TrackOptions
{
  Options base;std::string table,output,mode="uniform",initial="profile";
  std::string closure_space="h1",closure_boundary="auto",euler_boundary="constant",patch_motion="anchored",mesh_features="one";
  int radius=3,remap_every=10,quad_extra=5;
  double left=-.25,right=.25,grading=1.,center=.013;
  double width=.05,mach=5.,speed=.4,amplitude=.5,sipg_penalty=4.,feature_prominence=.02;
  bool self_test=false,positivity=true;
};
struct TrackWork
{
  double setup=0,closure=0,transport=0,limiter=0,remap=0,detection=0;
  double max_closure_residual=0,max_remap_defect=0,max_rhs_defect=0,min_theta=1.;
  double current_source_integral=0,max_remap_source_relative_change=0;
  double signed_remap_source_change=0,absolute_remap_source_change=0;
  std::uint64_t closure_calls=0,closure_assemblies=0,closure_faces=0,source_points=0,flux_points=0,faces=0;
  std::uint64_t remaps=0,remap_cells=0,remap_points=0,detector_points=0;
  std::uint64_t positivity_points=0,limited=0,steps=0,attempted=0;
  std::uint64_t feature_detections=0,feature_points=0,feature_meshes=0,feature_clearance_rejections=0;
};
struct ProfileTable
{
  double gamma=1.4,alpha=.0004,speed=.4,sonic=.013;
  State x;std::vector<std::array<double,5>> v;
  void Read(const std::string& path)
  {
    std::ifstream file(path);if(!file)throw std::runtime_error("cannot open profile table");
    std::string line;int count=0;bool header=false;
    while(std::getline(file,line))
    {
      if(line.empty()||line[0]=='#')continue;std::istringstream row(line);
      if(!header){if(!(row>>count>>gamma>>alpha>>speed>>sonic)||count<2)throw std::runtime_error("invalid profile header");header=true;continue;}
      double point;std::array<double,5> values{};if(!(row>>point))continue;
      for(double& a:values)if(!(row>>a))throw std::runtime_error("invalid profile row");
      if(!x.empty()&&!(point>x.back()))throw std::runtime_error("profile coordinates not strictly increasing");
      x.push_back(point);v.push_back(values);
    }
    if(int(x.size())!=count||std::abs(gamma-gamma_euler)>1.e-12)throw std::runtime_error("profile size/gamma mismatch");
  }
  std::array<double,5> At(double point,double time)const
  {
    // Exported row coordinates are relative to the sonic point; the header
    // records its physical placement in the finite-domain experiment.
    const double z=point-sonic-speed*time;
    if(z<x.front()-1.e-12||z>x.back()+1.e-12)throw std::runtime_error("profile table does not cover moving boundary");
    auto it=std::upper_bound(x.begin(),x.end(),z);const int i=std::max(0,std::min(int(x.size())-2,int(it-x.begin())-1));
    const double a=std::max(0.,std::min(1.,(z-x[i])/(x[i+1]-x[i])));std::array<double,5> out{};
    for(int c=0;c<5;++c)out[c]=(1.-a)*v[i][c]+a*v[i+1][c];return out;
  }
};
bool TrackAdmitted(const std::array<double,4>& v)
{
  for(double z:v)if(!std::isfinite(z))return false;if(!(v[0]>floor_value))return false;
  return (gamma_euler-1.)*(v[3]-.5*(v[1]*v[1]+v[2]*v[2])/v[0])>floor_value;
}
struct TrackingProblem
{
  TrackOptions o;Problem physics;ProfileTable profile;
  int ne,p,n,ns,patch_index;double anchor_left,anchor_right;
  double translated_center_min=0.,translated_center_max=0.,initial_graded_hmin=0.;
  mutable bool feature_active=false;mutable double feature_center=0.,feature_relative_prominence=0.;
  State edges,base_uniform_edges,weights;std::vector<State> shapes,derivatives,positive;
  mfem::DenseMatrix mass,mass_inverse;
  explicit TrackingProblem(const TrackOptions& options):o(options),physics(o.base),ne(o.base.elements),p(o.base.order),n(p+1),ns(o.closure_space=="l2"?ne*n:ne*p+1),mass(n),mass_inverse(n)
  {
    if(!o.table.empty())profile.Read(o.table);
    if(o.closure_boundary=="auto")o.closure_boundary=(o.initial=="tanh"||o.initial=="simple-wave")?"natural":"dirichlet";
    if(o.initial=="profile")
    {if(o.table.empty())throw std::runtime_error("profile initial condition requires --table");
      if(std::abs(o.base.alpha-profile.alpha)>1.e-12*std::max(1.,profile.alpha))throw std::runtime_error("profile alpha mismatch");o.center=profile.sonic;}
    edges.resize(ne+1);for(int e=0;e<=ne;++e)edges[e]=o.left+(o.right-o.left)*e/ne;base_uniform_edges=edges;
    if(ne<2*o.radius+2||p<1||o.radius<1||!(o.grading>=1.))throw std::runtime_error("invalid patch dimensions");
    patch_index=std::max(o.radius,std::min(ne-o.radius,int(std::lround((o.center-o.left)/(o.right-o.left)*ne))));
    anchor_left=edges[patch_index-o.radius];anchor_right=edges[patch_index+o.radius];
    if(!(o.center>anchor_left&&o.center<anchor_right))throw std::runtime_error("initial center outside patch");
    // Translation keeps every graded patch width fixed. Outside elements share
    // the remaining length; neither side may become smaller than the smallest
    // original patch cell. Leaving this range is an explicit experimental
    // failure, rather than silently collapsing h while retaining the old dt.
    const double initial_hmin=std::min(o.center-anchor_left,anchor_right-o.center)/std::pow(double(o.radius),o.grading);initial_graded_hmin=initial_hmin;
    translated_center_min=o.left+(o.center-anchor_left)+(patch_index-o.radius)*initial_hmin;
    translated_center_max=o.right-(anchor_right-o.center)-(ne-patch_index-o.radius)*initial_hmin;
    if(o.patch_motion=="translate"&&(patch_index==o.radius||patch_index+o.radius==ne))throw std::runtime_error("translated patch requires an outside element on each side");
    if(o.mesh_features=="two"&&patch_index<=o.radius)throw std::runtime_error("two-feature mesh requires an upstream outer edge");
    const auto& rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,2*p+o.quad_extra);
    for(int k=0;k<rule.GetNPoints();++k)
    {const auto& ip=rule.IntPoint(k);weights.push_back(ip.weight);shapes.push_back(Shape(ip.x,false));derivatives.push_back(Shape(ip.x,true));}
    const auto& audit=mfem::IntRules.Get(mfem::Geometry::SEGMENT,4*p+5);
    for(int k=0;k<audit.GetNPoints();++k)positive.push_back(Shape(audit.IntPoint(k).x,false));
    for(int i=0;i<n;++i)positive.push_back(Shape(physics.basis.nodes[i],false));
    positive.insert(positive.end(),shapes.begin(),shapes.end());
    mass=0.;for(std::size_t k=0;k<weights.size();++k)for(int i=0;i<n;++i)for(int j=0;j<n;++j)mass(i,j)+=weights[k]*shapes[k][i]*shapes[k][j];
    mfem::DenseMatrixInverse inverse(mass,true);mfem::DenseMatrix identity(n);identity=0.;for(int i=0;i<n;++i)identity(i,i)=1.;inverse.Mult(identity,mass_inverse);
    if(o.mode!="uniform"&&o.mode!="roundtrip-uniform")edges=Mesh(o.center);
  }
  int Id(int e,int i,int c)const{return (e*n+i)*4+c;}
  int SigmaIndex(int e,int i)const{return o.closure_space=="l2"?e*n+i:e*p+i;}
  int ClosureBandwidth()const{return o.closure_space=="l2"?n:p;}
  State Shape(double x,bool derivative)const
  {
    State s(n,0.);const auto& nodes=physics.basis.nodes;
    for(int i=0;i<n;++i)
    {
      if(!derivative)
      {double a=1.;for(int j=0;j<n;++j)if(i!=j)a*=(x-nodes[j])/(nodes[i]-nodes[j]);s[i]=a;}
      else
      {for(int k=0;k<n;++k)if(k!=i){double a=1./(nodes[i]-nodes[k]);for(int j=0;j<n;++j)if(j!=i&&j!=k)a*=(x-nodes[j])/(nodes[i]-nodes[j]);s[i]+=a;}}
    }
    return s;
  }
  State Mesh(double center)const
  {
    State out=o.mesh_features=="two"?base_uniform_edges:edges;
    if(o.patch_motion=="translate")
    {
      if(center<translated_center_min||center>translated_center_max)throw std::runtime_error("translated patch center outside permitted range");
      const double left_span=o.center-anchor_left,right_span=anchor_right-o.center;
      for(int j=0;j<=o.radius;++j)
      {const double a=std::pow(double(j)/o.radius,o.grading);out[patch_index-j]=center-left_span*a;out[patch_index+j]=center+right_span*a;}
      const int il=patch_index-o.radius,ir=patch_index+o.radius;
      for(int e=0;e<il;++e)out[e]=o.left+(out[il]-o.left)*e/il;
      for(int e=ir+1;e<=ne;++e)out[e]=out[ir]+(o.right-out[ir])*(e-ir)/(ne-ir);
      for(int e=0;e<ne;++e)if(!(out[e+1]>out[e]))throw std::runtime_error("invalid translated mesh");return out;
    }
    const double margin=1.e-4*(anchor_right-anchor_left);
    center=std::max(anchor_left+margin,std::min(anchor_right-margin,center));
    for(int j=0;j<=o.radius;++j)
    {const double a=std::pow(double(j)/o.radius,o.grading);out[patch_index-j]=center-(center-anchor_left)*a;out[patch_index+j]=center+(anchor_right-center)*a;}
    for(int e=0;e<ne;++e)if(!(out[e+1]>out[e]))throw std::runtime_error("invalid rezoned mesh");return out;
  }
  State Mesh(double center,const State& u,TrackWork& w)const
  {
    State next=Mesh(center);if(o.mesh_features!="two")return next;
    const auto start=Clock::now();++w.feature_detections;feature_active=false;
    const double baseline=std::max(u[Id(0,0,3)],u[Id(ne-1,p,3)]);
    const double lower=center+2.*std::sqrt(o.base.alpha);
    const double upper=o.right-initial_graded_hmin;
    double best=-INFINITY,peak=0.;
    auto energy=[&](int e,double z){++w.feature_points;++w.detector_points;return Point(u,e,Shape(z,false))[3];};
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];
      const double zl=std::max(0.,(lower-edges[e])/h),zr=std::min(1.,(upper-edges[e])/h);
      if(!(zr>zl))continue;
      const int samples=8*p;int winner=0;double maximum=-INFINITY;
      for(int j=0;j<=samples;++j){const double z=zl+(zr-zl)*j/samples,v=energy(e,z);if(v>maximum){maximum=v;winner=j;}}
      double lo=zl+(zr-zl)*std::max(0,winner-1)/samples,hi=zl+(zr-zl)*std::min(samples,winner+1)/samples;
      for(int j=0;j<32;++j){const double a=lo+(hi-lo)/3.,b=hi-(hi-lo)/3.;if(energy(e,a)<energy(e,b))lo=a;else hi=b;}
      const double z=.5*(lo+hi),value=energy(e,z);
      if(value>best){best=value;peak=edges[e]+z*h;}
      for(double endpoint:{zl,zr}){const double v=energy(e,endpoint);if(v>best){best=v;peak=edges[e]+endpoint*h;}}
    }
    feature_center=peak;feature_relative_prominence=std::isfinite(best)?(best-baseline)/std::max(std::abs(baseline),1.e-12):0.;
    if(std::isfinite(best)&&best>(1.+o.feature_prominence)*baseline&&peak>lower&&peak<upper)
    {
      // Exchange one coarse upstream edge for the observed outgoing pulse.
      // Both split cells retain at least the original graded minimum width;
      // absence of clearance simply retains the one-feature partition.
      const int split=int(std::upper_bound(next.begin(),next.end(),peak)-next.begin())-1;
      if(split>=0&&split<ne&&peak-next[split]>=initial_graded_hmin&&next[split+1]-peak>=initial_graded_hmin)
      {
        next.erase(next.begin()+patch_index-o.radius);
        next.insert(std::upper_bound(next.begin(),next.end(),peak),peak);
        if(int(next.size())!=ne+1)throw std::runtime_error("two-feature element count changed");
        for(int e=0;e<ne;++e)if(!(next[e+1]>next[e]))throw std::runtime_error("invalid two-feature partition");
        feature_active=true;++w.feature_meshes;
      }
      else ++w.feature_clearance_rejections;
    }
    w.detection+=Wall(start);return next;
  }
  std::array<double,4> Point(const State& u,int e,const State& shape)const
  {std::array<double,4> v{};for(int i=0;i<n;++i)for(int c=0;c<4;++c)v[c]+=shape[i]*u[Id(e,i,c)];return v;}
  std::array<double,5> PrimitiveState(double rho,double velocity,double pressure)const
  {return {{rho,rho*velocity,0.,pressure/(gamma_euler-1.)+.5*rho*velocity*velocity,0.}};}
  std::array<double,5> SmoothInitial(double x)const
  {
    const double transition=.5*(1.+std::tanh((x-o.center)/o.width));
    if(o.initial=="tanh")
    {
      const double m2=o.mach*o.mach;
      const double ratio=(gamma_euler+1.)*m2/((gamma_euler-1.)*m2+2.);
      const double downstream_pressure=1.+2.*gamma_euler/(gamma_euler+1.)*(m2-1.);
      const double inflow=o.mach*std::sqrt(gamma_euler);
      return PrimitiveState(1.+transition*(ratio-1.),o.speed+inflow*(1.-transition+transition/ratio),1.+transition*(downstream_pressure-1.));
    }
    // Right-running isentropic simple wave: u-2c/(gamma-1) is constant.
    // A smooth monotone compression evolves from two isentropic far fields.
    // The unregularized Euler breaking time is 2*width/((gamma+1)*amplitude).
    const double velocity=o.amplitude*(1.-2.*transition);
    const double c0=std::sqrt(gamma_euler),sound=c0+.5*(gamma_euler-1.)*velocity;
    if(!(sound>0.))throw std::runtime_error("simple wave sound speed is nonpositive");
    const double rho=std::pow(sound/c0,2./(gamma_euler-1.));
    return PrimitiveState(rho,velocity+o.speed,std::pow(rho,gamma_euler));
  }
  std::array<double,5> Reference(double x,double t)const
  {
    if(o.initial=="profile")return profile.At(x,t);
    if(o.initial=="tanh"||o.initial=="simple-wave")return SmoothInitial(x);
    const double rho=1.+.2*std::sin(2.*std::acos(-1.)*(x-.2*t));
    return {{rho,.2*rho,0.,1./(gamma_euler-1.)+.02*rho,0.}};
  }
  std::array<double,4> Integrals(const State& u,const State& mesh)const
  {
    std::array<double,4> total{};for(int e=0;e<ne;++e)for(int i=0;i<n;++i)for(int c=0;c<4;++c)total[c]+=(mesh[e+1]-mesh[e])*physics.basis.weights[i]*u[Id(e,i,c)];return total;
  }
  State Initial()const
  {
    State u(ne*n*4,0.);
    // Interpolate known analytic profile at the actual high-order nodes. The
    // subsequent remap is a separate exact conservative polynomial projection.
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)
    {const auto v=Reference(edges[e]+(edges[e+1]-edges[e])*physics.basis.nodes[i],0.);for(int c=0;c<4;++c)u[Id(e,i,c)]=v[c];}return u;
  }
  void Limit(State& u,TrackWork& w)const
  {
    const auto start=Clock::now();for(int e=0;e<ne;++e)
    {
      std::array<double,4> mean{};for(int i=0;i<n;++i)for(int c=0;c<4;++c)mean[c]+=physics.basis.weights[i]*u[Id(e,i,c)];
      if(!TrackAdmitted(mean))throw std::runtime_error("element mean admission failed");double theta=1.;
      for(const auto& shape:positive)
      {
        ++w.positivity_points;const auto v=Point(u,e,shape);if(TrackAdmitted(v))continue;
        if(!o.positivity)throw std::runtime_error("polynomial positivity sample failed");double lo=0.,hi=1.;
        for(int j=0;j<50;++j){const double mid=.5*(lo+hi);std::array<double,4> test{};for(int c=0;c<4;++c)test[c]=mean[c]+mid*(v[c]-mean[c]);if(TrackAdmitted(test))lo=mid;else hi=mid;}
        theta=std::min(theta,lo*(1.-1.e-12));
      }
      if(theta<1.){++w.limited;w.min_theta=std::min(w.min_theta,theta);for(int i=0;i<n;++i)for(int c=0;c<4;++c)u[Id(e,i,c)]=mean[c]+theta*(u[Id(e,i,c)]-mean[c]);}
    }w.limiter+=Wall(start);
  }
  void ApplyClosureBoundary(BandMatrix& a,State& b,double time)const
  {
    if(o.closure_boundary=="natural")return;
    // The exact-profile finite-domain comparison uses the same strong scalar
    // trace constraints in H1 and L2. Other initial cases default to natural
    // Neumann closure, matching the application's current boundary policy.
    const std::array<double,2> boundary{{Reference(o.left,time)[4],Reference(o.right,time)[4]}};
    for(int side=0;side<2;++side)
    {
      const int j=side?ns-1:0;
      for(int i=std::max(0,j-a.k);i<std::min(ns,j+a.k+1);++i)if(i!=j)
      {b[i]-=a.Get(i,j)*boundary[side];a.At(i,j)=0.;a.At(j,i)=0.;}
      a.At(j,j)=1.;b[j]=boundary[side];
    }
  }
  void AssembleClosure(const State& u,double time,TrackWork& w,BandMatrix& a,State& b)const
  {
    a=BandMatrix(ns,ClosureBandwidth());b.assign(ns,0.);double source_integral=0.;
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];for(std::size_t k=0;k<weights.size();++k)
      {
        ++w.source_points;const auto v=Point(u,e,shapes[k]);physics.CheckState(v.data());double du[8]={};
        for(int i=0;i<n;++i)for(int c=0;c<4;++c)du[2*c]+=derivatives[k][i]*u[Id(e,i,c)]/h;
        const double ri=igr::SafeInverseDensity(v[0],floor_value);
        const double source=igr::SourceFromConservativeStateGradient(2,v.data(),1,du,2,1,o.base.alpha,floor_value);
        if(!std::isfinite(source))throw std::runtime_error("nonfinite current source");
        source_integral+=h*weights[k]*source;
        for(int i=0;i<n;++i)
        {b[SigmaIndex(e,i)]+=h*weights[k]*shapes[k][i]*source;for(int j=0;j<n;++j)a.At(SigmaIndex(e,i),SigmaIndex(e,j))+=weights[k]*ri*(h*shapes[k][i]*shapes[k][j]+o.base.alpha*derivatives[k][i]*derivatives[k][j]/h);}
      }
    }
    if(o.closure_space=="l2"&&o.base.alpha!=0.)
    {
      const State dl=Shape(1.,true),dr=Shape(0.,true);
      for(int face=1;face<ne;++face)
      {
        ++w.closure_faces;const int el=face-1,er=face;
        const double hl=edges[el+1]-edges[el],hr=edges[er+1]-edges[er];
        const double al=o.base.alpha*igr::SafeInverseDensity(u[Id(el,p,0)],floor_value);
        const double ar=o.base.alpha*igr::SafeInverseDensity(u[Id(er,0,0)],floor_value);
        // Symmetric interior penalty for -(alpha/rho sigma_x)_x. The
        // arithmetic face coefficient and each side's physical h match the
        // default application scaling. p enters the penalty, not alpha.
        const double penalty=o.sipg_penalty*n*n*.5*(al/hl+ar/hr);
        for(int i=0;i<2*n;++i)for(int j=0;j<2*n;++j)
        {
          const bool il=i<n,jl=j<n;const int ii=i%n,jj=j%n;
          const double jumpi=il?(ii==p?1.:0.):(ii==0?-1.:0.);
          const double jumpj=jl?(jj==p?1.:0.):(jj==0?-1.:0.);
          if(jumpi==0.&&jumpj==0.)continue;
          const double fluxi=.5*(il?al*dl[ii]/hl:ar*dr[ii]/hr);
          const double fluxj=.5*(jl?al*dl[jj]/hl:ar*dr[jj]/hr);
          a.At(SigmaIndex(il?el:er,ii),SigmaIndex(jl?el:er,jj))+=-jumpi*fluxj-fluxi*jumpj+penalty*jumpi*jumpj;
        }
      }
    }
    ApplyClosureBoundary(a,b,time);w.current_source_integral=source_integral;++w.closure_assemblies;
  }
  State Closure(const State& u,double time,TrackWork& w)const
  {
    const auto start=Clock::now();BandMatrix a;State b;AssembleClosure(u,time,w,a,b);
    BandFactor factor(a,0,ns);State sigma=b;factor.Solve(sigma,1);State defect(ns,0.);
    for(int i=0;i<ns;++i){defect[i]=-b[i];for(int j=std::max(0,i-a.k);j<std::min(ns,i+a.k+1);++j)defect[i]+=a.Get(i,j)*sigma[j];}
    const double residual=Norm(defect)/std::max(Norm(b),1.e-12);w.max_closure_residual=std::max(w.max_closure_residual,residual);
    if(!std::isfinite(residual)||residual>1.e-8)throw std::runtime_error("current banded closure certificate failed");
    ++w.closure_calls;w.closure+=Wall(start);return sigma;
  }
  State Residual(const State& u,const State& sigma,double time,TrackWork& w,std::array<double,4>& boundary_budget)const
  {
    const auto start=Clock::now();State load(u.size(),0.),r(u.size(),0.);
    for(int e=0;e<ne;++e)for(std::size_t k=0;k<weights.size();++k)
    {
      ++w.flux_points;const auto v=Point(u,e,shapes[k]);double sq=0.,speed=0.;for(int j=0;j<n;++j)sq+=shapes[k][j]*sigma[SigmaIndex(e,j)];const auto f=physics.Flux(v.data(),sq,speed);
      for(int i=0;i<n;++i)for(int c=0;c<4;++c)load[Id(e,i,c)]+=weights[k]*derivatives[k][i]*f[c];
    }
    // Natural scalar boundaries extend the interior stress into the Euler ghost,
    // matching production DG/DGSEM. Dirichlet profile experiments retain their datum.
    for(int face=0;face<=ne;++face)
    {
      ++w.faces;std::array<double,4> vl{},vr{};double sl=0.,sr=0.,al=0.,ar=0.;
      if(face==0){if(o.euler_boundary=="outflow"){for(int c=0;c<4;++c)vl[c]=u[Id(0,0,c)];sl=sigma[SigmaIndex(0,0)];}else{const auto v=Reference(o.left,time);std::copy_n(v.begin(),4,vl.begin());sl=o.closure_boundary=="natural"?sigma[SigmaIndex(0,0)]:v[4];}}
      else{for(int c=0;c<4;++c)vl[c]=u[Id(face-1,p,c)];sl=sigma[SigmaIndex(face-1,p)];}
      if(face==ne){if(o.euler_boundary=="outflow"){for(int c=0;c<4;++c)vr[c]=u[Id(ne-1,p,c)];sr=sigma[SigmaIndex(ne-1,p)];}else{const auto v=Reference(o.right,time);std::copy_n(v.begin(),4,vr.begin());sr=o.closure_boundary=="natural"?sigma[SigmaIndex(ne-1,p)]:v[4];}}
      else{for(int c=0;c<4;++c)vr[c]=u[Id(face,0,c)];sr=sigma[SigmaIndex(face,0)];}
      const auto fl=physics.Flux(vl.data(),sl,al),fr=physics.Flux(vr.data(),sr,ar);const double speed=physics.FaceSpeed(al,ar);
      for(int c=0;c<4;++c)
      {const double f=.5*(fl[c]+fr[c]-speed*(vr[c]-vl[c]));if(face>0)load[Id(face-1,p,c)]-=f;if(face<ne)load[Id(face,0,c)]+=f;if(face==0)boundary_budget[c]=f;if(face==ne)boundary_budget[c]-=f;}
    }
    for(int e=0;e<ne;++e)for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int c=0;c<4;++c)r[Id(e,i,c)]+=mass_inverse(i,j)*load[Id(e,j,c)]/(edges[e+1]-edges[e]);
    const auto integrals=Integrals(r,edges);for(int c=0;c<4;++c)w.max_rhs_defect=std::max(w.max_rhs_defect,std::abs(integrals[c]-boundary_budget[c]));w.transport+=Wall(start);return r;
  }
  double DetectPhysical(const State& sigma,TrackWork& w)const
  {
    const auto start=Clock::now();double best=-INFINITY,center=o.center;
    auto value=[&](int e,double z){++w.detector_points;const State shape=Shape(z,false);double v=0.;for(int i=0;i<n;++i)v+=shape[i]*sigma[SigmaIndex(e,i)];return v;};
    for(int e=0;e<ne;++e)
    {
      const double h=edges[e+1]-edges[e];const double zl=std::max(0.,(anchor_left-edges[e])/h),zr=std::min(1.,(anchor_right-edges[e])/h);
      if(!(zr>zl))continue;const int samples=8*p;int winner=0;double maximum=-INFINITY;
      for(int j=0;j<=samples;++j){const double z=zl+(zr-zl)*j/samples,v=value(e,z);if(v>maximum){maximum=v;winner=j;}}
      double lo=zl+(zr-zl)*std::max(0,winner-1)/samples,hi=zl+(zr-zl)*std::min(samples,winner+1)/samples;
      for(int j=0;j<32;++j){const double a=lo+(hi-lo)/3.,b=hi-(hi-lo)/3.;if(value(e,a)<value(e,b))lo=a;else hi=b;}
      const double z=.5*(lo+hi),v=value(e,z);if(v>best){best=v;center=edges[e]+z*h;}
      for(double endpoint:{zl,zr}){const double v0=value(e,endpoint);if(v0>best){best=v0;center=edges[e]+endpoint*h;}}
    }
    w.detection+=Wall(start);return center;
  }
  double Detect(const State& sigma,TrackWork& w)const
  {
    if(o.mesh_features=="two")return DetectPhysical(sigma,w);
    const auto start=Clock::now();double best=-INFINITY,center=edges[patch_index];
    auto value=[&](int e,double x){++w.detector_points;const State s=Shape(x,false);double v=0.;for(int i=0;i<n;++i)v+=s[i]*sigma[SigmaIndex(e,i)];return v;};
    // Find all candidate maxima on a deterministic fine polynomial sample, then
    // refine each winning bracket. Search stays inside the selected patch.
    for(int e=patch_index-o.radius;e<patch_index+o.radius;++e)
    {
      const int samples=8*p;int winner=0;double vmax=-INFINITY;for(int j=0;j<=samples;++j){const double v=value(e,double(j)/samples);if(v>vmax){vmax=v;winner=j;}}
      double lo=std::max(0.,double(winner-1)/samples),hi=std::min(1.,double(winner+1)/samples);
      for(int j=0;j<32;++j){const double a=lo+(hi-lo)/3.,b=hi-(hi-lo)/3.;if(value(e,a)<value(e,b))lo=a;else hi=b;}
      const double z=.5*(lo+hi),v=value(e,z);if(v>best){best=v;center=edges[e]+z*(edges[e+1]-edges[e]);}
      // Include endpoints exactly, since a true sonic maximum may be a trace.
      for(double z0:{0.,1.}){const double v0=value(e,z0);if(v0>best){best=v0;center=edges[e]+z0*(edges[e+1]-edges[e]);}}
    }
    w.detection+=Wall(start);return center;
  }
  State Remap(const State& u,const State& next,TrackWork& w)
  {
    const auto start=Clock::now();State out(u.size(),0.);const auto before=Integrals(u,edges);
    const auto& rule=mfem::IntRules.Get(mfem::Geometry::SEGMENT,2*p+1);
    for(int e=0;e<ne;++e)
    {
      if(next[e]==edges[e]&&next[e+1]==edges[e+1]){for(int i=0;i<n;++i)for(int c=0;c<4;++c)out[Id(e,i,c)]=u[Id(e,i,c)];continue;}
      ++w.remap_cells;State load(n*4,0.);const double h=next[e+1]-next[e];
      int old=std::max(0,int(std::upper_bound(edges.begin(),edges.end(),next[e])-edges.begin())-1);
      for(;old<ne&&edges[old]<next[e+1];++old)
      {
        const double left=std::max(edges[old],next[e]),right=std::min(edges[old+1],next[e+1]);if(!(right>left))continue;
        for(int k=0;k<rule.GetNPoints();++k)
        {++w.remap_points;const auto& ip=rule.IntPoint(k);const double x=left+(right-left)*ip.x;const auto s=Shape((x-next[e])/h,false);const auto v=Point(u,old,Shape((x-edges[old])/(edges[old+1]-edges[old]),false));
          for(int i=0;i<n;++i)for(int c=0;c<4;++c)load[i*4+c]+=(right-left)*ip.weight*s[i]*v[c]/h;}
      }
      for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int c=0;c<4;++c)out[Id(e,i,c)]+=mass_inverse(i,j)*load[j*4+c];
    }
    const auto after=Integrals(out,next);for(int c=0;c<4;++c)w.max_remap_defect=std::max(w.max_remap_defect,std::abs(after[c]-before[c]));edges=next;++w.remaps;w.remap+=Wall(start);return out;
  }
};

struct TrackResult
{
  State u,sigma;TrackWork work;std::string failure;double time=0,wall=0,initial_center=0,last_center=0;
  std::array<double,4> initial_integrals{},boundary_budget{},conservation_defect{};
};
TrackResult TrackRun(TrackingProblem& pr)
{
  TrackResult out;const auto start=Clock::now();out.initial_center=pr.edges[pr.patch_index];out.last_center=out.initial_center;
  try
  {
    out.u=pr.Initial();pr.Limit(out.u,out.work);out.initial_integrals=pr.Integrals(out.u,pr.edges);out.sigma=pr.Closure(out.u,0.,out.work);
    while(out.time<pr.o.base.final_time-1.e-14)
    {
      ++out.work.attempted;const double dt=std::min(pr.o.base.dt,pr.o.base.final_time-out.time);const State original=out.u;
      State stage=original;std::array<std::array<double,4>,3> budget{};
      const double times[3]={out.time,out.time+dt,out.time+.5*dt};
      for(int k=0;k<3;++k)
      {
        const State sigma=k?pr.Closure(stage,times[k],out.work):out.sigma;
        const State rhs=pr.Residual(stage,sigma,times[k],out.work,budget[k]);
        for(std::size_t i=0;i<stage.size();++i)
        {const double candidate=stage[i]+dt*rhs[i];stage[i]=k==0?candidate:(k==1?.75*original[i]+.25*candidate:original[i]/3.+2.*candidate/3.);}
        pr.Limit(stage,out.work);
      }
      // An accepted PDE step precedes any remap. All accepted boundary fluxes
      // are counted even if subsequent rezoning or closure fails.
      out.u=std::move(stage);out.time+=dt;++out.work.steps;
      for(int c=0;c<4;++c)out.boundary_budget[c]+=dt*(budget[0][c]/6.+budget[1][c]/6.+2.*budget[2][c]/3.);
      out.sigma=pr.Closure(out.u,out.time,out.work);
      if((pr.o.mode=="oracle"||pr.o.mode=="detected"||pr.o.mode=="roundtrip"||pr.o.mode=="roundtrip-uniform")&&out.work.steps%pr.o.remap_every==0)
      {
        const double center=pr.o.mode=="oracle"?pr.o.center+pr.profile.speed*out.time:pr.Detect(out.sigma,out.work);
        const State next=pr.Mesh(center,out.u,out.work);double movement=0.;for(int e=0;e<=pr.ne;++e)movement=std::max(movement,std::abs(next[e]-pr.edges[e]));
        if(movement>1.e-14)
        {
          const double before=out.work.current_source_integral;
          const State original_edges=pr.edges;
          out.u=pr.Remap(out.u,next,out.work);
          // Causal control: retain both conservative projection filters while
          // restoring the original fixed space before any physics evaluation.
          // The intermediate state is only a polynomial transfer, not a PDE
          // stage; no intermediate limiter or closure is introduced.
          if(pr.o.mode=="roundtrip"||pr.o.mode=="roundtrip-uniform")out.u=pr.Remap(out.u,original_edges,out.work);
          pr.Limit(out.u,out.work);out.sigma=pr.Closure(out.u,out.time,out.work);
          const double change=out.work.current_source_integral-before;
          out.work.max_remap_source_relative_change=std::max(out.work.max_remap_source_relative_change,std::abs(change)/std::max(std::abs(before),1.e-20));
          out.work.signed_remap_source_change+=change;out.work.absolute_remap_source_change+=std::abs(change);
        }
        out.last_center=pr.o.mesh_features=="two"?center:pr.edges[pr.patch_index];
      }
    }
  }
  catch(const std::exception& e){out.failure=e.what();}
  if(!out.u.empty()){const auto now=pr.Integrals(out.u,pr.edges);for(int c=0;c<4;++c)out.conservation_defect[c]=now[c]-out.initial_integrals[c]-out.boundary_budget[c];}
  out.wall=Wall(start);return out;
}
void TrackArray(std::ostream& f,const State& a)
{f<<'[';for(std::size_t i=0;i<a.size();++i){if(i)f<<',';if(std::isfinite(a[i]))f<<a[i];else f<<"null";}f<<']';}
void TrackWrite(const TrackingProblem& pr,const TrackResult& r,std::ostream& f)
{
  f<<std::setprecision(17)<<"{\n\"method\":\"selective_high_order_dg_conservative_rezone\",\n\"mode\":\""<<pr.o.mode<<"\",\n\"completed\":"<<(r.failure.empty()?"true":"false")<<",\n\"failure\":\""<<r.failure<<"\",\n\"elements\":"<<pr.ne<<",\"order\":"<<pr.p<<",\"euler_dofs_per_field\":"<<pr.ne*pr.n<<",\"sigma_dofs\":"<<pr.ns<<",\n\"alpha\":"<<pr.o.base.alpha<<",\"dt\":"<<pr.o.base.dt<<",\"target_time\":"<<pr.o.base.final_time<<",\"time\":"<<r.time<<",\"grading\":"<<pr.o.grading<<",\"patch_radius\":"<<pr.o.radius<<",\"remap_every\":"<<pr.o.remap_every<<",\"quad_extra\":"<<pr.o.quad_extra<<",\"wall_seconds\":"<<r.wall<<",\n\"initial_center\":"<<r.initial_center<<",\"last_center\":"<<r.last_center<<",\n\"work\":{\"steps\":"<<r.work.steps<<",\"attempted\":"<<r.work.attempted<<",\"closure_calls\":"<<r.work.closure_calls<<",\"source_points\":"<<r.work.source_points<<",\"flux_points\":"<<r.work.flux_points<<",\"faces\":"<<r.work.faces<<",\"remaps\":"<<r.work.remaps<<",\"remap_cells\":"<<r.work.remap_cells<<",\"remap_points\":"<<r.work.remap_points<<",\"detector_points\":"<<r.work.detector_points<<",\"positivity_points\":"<<r.work.positivity_points<<",\"limited\":"<<r.work.limited<<",\"minimum_theta\":"<<r.work.min_theta<<",\"max_closure_residual\":"<<r.work.max_closure_residual<<",\"max_remap_defect\":"<<r.work.max_remap_defect<<",\"max_rhs_defect\":"<<r.work.max_rhs_defect<<",\"closure_seconds\":"<<r.work.closure<<",\"transport_seconds\":"<<r.work.transport<<",\"limiter_seconds\":"<<r.work.limiter<<",\"remap_seconds\":"<<r.work.remap<<",\"detection_seconds\":"<<r.work.detection<<"},\n\"conservation_defect\":";TrackArray(f,State(r.conservation_defect.begin(),r.conservation_defect.end()));
  f<<",\n\"setup_seconds\":"<<r.work.setup<<",\"source_integral\":"<<r.work.current_source_integral<<",\"max_remap_source_relative_change\":"<<r.work.max_remap_source_relative_change<<",\"signed_remap_source_change\":"<<r.work.signed_remap_source_change<<",\"absolute_remap_source_change\":"<<r.work.absolute_remap_source_change;
  f<<",\n\"initial\":\""<<pr.o.initial<<"\",\"closure_space\":\""<<pr.o.closure_space<<"\",\"sigma_layout\":\""<<(pr.o.closure_space=="l2"?"l2_element_nodes":"h1_shared_nodes")<<"\",\"closure_boundary\":\""<<pr.o.closure_boundary<<"\",\"euler_boundary\":\""<<pr.o.euler_boundary<<"\",\"width\":"<<pr.o.width<<",\"physical_center\":"<<pr.o.center<<",\"mach\":"<<pr.o.mach<<",\"speed\":"<<pr.o.speed<<",\"amplitude\":"<<pr.o.amplitude<<",\"sipg_penalty_multiplier\":"<<pr.o.sipg_penalty<<",\"closure_assemblies\":"<<r.work.closure_assemblies<<",\"closure_faces\":"<<r.work.closure_faces;
  f<<",\"patch_motion\":\""<<pr.o.patch_motion<<"\",\"translated_center_min\":"<<pr.translated_center_min<<",\"translated_center_max\":"<<pr.translated_center_max;
  f<<",\"mesh_features\":\""<<pr.o.mesh_features<<"\",\"feature_prominence_threshold\":"<<pr.o.feature_prominence<<",\"feature_active\":"<<(pr.feature_active?"true":"false")<<",\"feature_center\":"<<pr.feature_center<<",\"feature_relative_prominence\":"<<pr.feature_relative_prominence<<",\"feature_detections\":"<<r.work.feature_detections<<",\"feature_points\":"<<r.work.feature_points<<",\"feature_meshes\":"<<r.work.feature_meshes<<",\"feature_clearance_rejections\":"<<r.work.feature_clearance_rejections;
  if(pr.o.initial=="simple-wave")f<<",\"unregularized_euler_breaking_time\":"<<2.*pr.o.width/((gamma_euler+1.)*pr.o.amplitude);
  f<<",\n\"edges\":";TrackArray(f,pr.edges);f<<",\n\"u\":";TrackArray(f,r.u);f<<",\n\"sigma\":";TrackArray(f,r.sigma);f<<"\n}\n";
}
TrackOptions TrackParse(int argc,char** argv)
{
  TrackOptions o;o.base.elements=16;o.base.order=3;o.base.alpha=.0004;o.base.dt=2.e-6;o.base.final_time=.003;
  for(int i=1;i<argc;++i)
  {
    const std::string flag=argv[i];if(flag=="--self-test"){o.self_test=true;continue;}if(flag=="--no-positivity"){o.positivity=false;continue;}
    if(i+1==argc)throw std::runtime_error("missing argument for "+flag);const std::string v=argv[++i];
    if(flag=="--elements")o.base.elements=std::stoi(v);else if(flag=="--order")o.base.order=std::stoi(v);else if(flag=="--alpha")o.base.alpha=std::stod(v);else if(flag=="--dt")o.base.dt=std::stod(v);else if(flag=="--final-time")o.base.final_time=std::stod(v);else if(flag=="--left")o.left=std::stod(v);else if(flag=="--right")o.right=std::stod(v);else if(flag=="--center")o.center=std::stod(v);else if(flag=="--grading")o.grading=std::stod(v);else if(flag=="--radius")o.radius=std::stoi(v);else if(flag=="--remap-every")o.remap_every=std::stoi(v);else if(flag=="--quad-extra")o.quad_extra=std::stoi(v);else if(flag=="--mode")o.mode=v;else if(flag=="--table")o.table=v;else if(flag=="--output")o.output=v;else if(flag=="--initial")o.initial=v;else if(flag=="--mesh-features")o.mesh_features=v;else if(flag=="--feature-prominence")o.feature_prominence=std::stod(v);else if(flag=="--patch-motion")o.patch_motion=v;else if(flag=="--closure-space")o.closure_space=v;else if(flag=="--closure-boundary")o.closure_boundary=v;else if(flag=="--euler-boundary")o.euler_boundary=v;else if(flag=="--width")o.width=std::stod(v);else if(flag=="--mach")o.mach=std::stod(v);else if(flag=="--speed")o.speed=std::stod(v);else if(flag=="--amplitude")o.amplitude=std::stod(v);else if(flag=="--sipg-penalty")o.sipg_penalty=std::stod(v);else throw std::runtime_error("unknown option "+flag);
  }
  if(o.mode!="uniform"&&o.mode!="frozen"&&o.mode!="oracle"&&o.mode!="detected"&&o.mode!="roundtrip"&&o.mode!="roundtrip-uniform")throw std::runtime_error("invalid mode");
  if(o.initial!="profile"&&o.initial!="entropy"&&o.initial!="tanh"&&o.initial!="simple-wave")throw std::runtime_error("invalid initial case");
  if(o.mesh_features!="one"&&o.mesh_features!="two")throw std::runtime_error("invalid mesh feature count");
  if(o.mesh_features=="two"&&(o.mode!="detected"||o.patch_motion!="anchored"))throw std::runtime_error("two features require detected anchored mesh motion");
  if(!(o.feature_prominence>0.))throw std::runtime_error("invalid feature prominence");
  if(o.patch_motion!="anchored"&&o.patch_motion!="translate")throw std::runtime_error("invalid patch motion");
  if(o.closure_space!="h1"&&o.closure_space!="l2")throw std::runtime_error("invalid closure space");
  if(o.closure_boundary!="auto"&&o.closure_boundary!="dirichlet"&&o.closure_boundary!="natural")throw std::runtime_error("invalid closure boundary");
  if(o.euler_boundary!="constant"&&o.euler_boundary!="outflow")throw std::runtime_error("invalid Euler boundary");
  if(o.mode=="oracle"&&o.initial!="profile")throw std::runtime_error("oracle motion is defined only for the exact traveling profile");
  if(!(o.width>0.)||!(o.mach>1.)||!(o.amplitude>0.)||!(o.sipg_penalty>0.)||o.base.alpha<0.)throw std::runtime_error("invalid smooth profile/closure parameters");
  if(!(o.base.dt>0.)||o.base.final_time<0.||!(o.left<o.right)||o.remap_every<1||o.quad_extra<1)throw std::runtime_error("invalid time/domain/quadrature");return o;
}
void TrackSelfTest()
{
  // Nonzero endpoint stress must enter a fixed Euler ghost consistently with
  // the scalar boundary condition. Unequal stresses avoid cancellation in the
  // integrated boundary budget; this exercises both scalar layouts.
  for(const std::string space:{"h1","l2"})for(const std::string bc:{"natural","dirichlet"})
  {
    TrackOptions o;o.initial="entropy";o.closure_space=space;o.closure_boundary=bc;
    o.base.elements=8;o.base.order=3;TrackingProblem pr(o);TrackWork work;
    const State u=pr.Initial();State sigma(pr.ns,.2);sigma.back()=.7;
    std::array<double,4> budget{},expected{};pr.Residual(u,sigma,0.,work,budget);
    for(int side=0;side<2;++side)
    {
      const auto reference=pr.Reference(side?o.right:o.left,0.);
      const double si=side?.7:.2,so=bc=="natural"?si:reference[4];double ai=0.,ao=0.;
      const auto fi=pr.physics.Flux(reference.data(),si,ai),fo=pr.physics.Flux(reference.data(),so,ao);
      for(int c=0;c<4;++c)expected[c]+=(side?-1.:1.)*.5*(fi[c]+fo[c]);
    }
    for(int c=0;c<4;++c)if(std::abs(budget[c]-expected[c])>1.e-12)
      throw std::runtime_error("fixed Euler boundary stress datum failed");
  }
  for(int p:{3,5,7})
  {
    TrackOptions o;o.initial="entropy";o.base.elements=12;o.base.order=p;o.base.alpha=.0004;o.left=0.;o.right=1.;o.center=.51;o.mode="frozen";o.grading=2.;TrackingProblem pr(o);TrackWork w;
    State constant(pr.ne*pr.n*4);for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)constant[pr.Id(e,i,c)]=c==0?2.:(c==1?.4:(c==2?.7:4.));
    const State before=constant;auto after=pr.Remap(constant,pr.Mesh(.53),w);
    for(std::size_t i=0;i<after.size();++i)if(std::abs(after[i]-before[i])>2.e-12)throw std::runtime_error("constant rezone/GCL failed");
    const State fixed=pr.edges;auto roundtrip=pr.Remap(after,pr.Mesh(.49),w);roundtrip=pr.Remap(roundtrip,fixed,w);
    for(std::size_t i=0;i<roundtrip.size();++i)if(std::abs(roundtrip[i]-before[i])>4.e-12)throw std::runtime_error("constant roundtrip/GCL failed");
    if(pr.edges!=fixed||w.max_remap_defect>3.e-12)throw std::runtime_error("roundtrip fixed geometry/conservation failed");
    // Independent piece polynomials ensure intersections really must be split.
    State u(pr.ne*pr.n*4);for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c)u[pr.Id(e,i,c)]=2.+.01*(e+1)*(c+1)*std::pow(pr.physics.basis.nodes[i],p);
    auto remapped=pr.Remap(u,pr.Mesh(.493),w);if(w.max_remap_defect>3.e-12)throw std::runtime_error("all-component conservative remap failed");
    // Global degree-p polynomials must be represented exactly after rezoning.
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c){const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];u[pr.Id(e,i,c)]=1.+(c+1)*std::pow(x,p);}
    remapped=pr.Remap(u,pr.Mesh(.55),w);for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)for(int c=0;c<4;++c){const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];if(std::abs(remapped[pr.Id(e,i,c)]-(1.+(c+1)*std::pow(x,p)))>2.e-11)throw std::runtime_error("degree-p remap recovery failed");}
    const State initial=pr.Initial();const State sigma=pr.Closure(initial,0.,w);std::array<double,4> budget{};const State rhs=pr.Residual(initial,sigma,0.,w,budget);(void)rhs;
    if(w.max_rhs_defect>2.e-10||Norm(sigma)>1.e-10)throw std::runtime_error("flux budget/constant velocity closure failed");
    // Nonzero affine velocity verifies physical derivatives and source scales.
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i){const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i],v=.2+.01*x;constant[pr.Id(e,i,0)]=2.;constant[pr.Id(e,i,1)]=2.*v;constant[pr.Id(e,i,2)]=0.;constant[pr.Id(e,i,3)]=3.+v*v;}
    const State driven=pr.Closure(constant,0.,w);if(!(Norm(driven)>1.e-8)||w.max_closure_residual>1.e-9)throw std::runtime_error("nonzero current source closure failed");
  }
  {
    TrackOptions o;o.initial="entropy";o.base.elements=16;o.base.order=7;o.left=-.25;o.right=.25;o.center=0.;o.mode="detected";o.grading=3.;o.patch_motion="translate";
    TrackingProblem pr(o);TrackWork w;const State first=pr.edges,next=pr.Mesh(.10);State u(pr.ne*pr.n*4,0.);
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {u[pr.Id(e,i,0)]=2.;u[pr.Id(e,i,1)]=.4;u[pr.Id(e,i,3)]=3.;}
    for(int e=pr.patch_index-o.radius;e<pr.patch_index+o.radius;++e)
      if(std::abs((next[e+1]-next[e])-(first[e+1]-first[e]))>2.e-16)throw std::runtime_error("translated patch width changed");
    const State after=pr.Remap(u,next,w);for(std::size_t i=0;i<u.size();++i)if(std::abs(after[i]-u[i])>3.e-12)throw std::runtime_error("translated patch constant preservation failed");
    if(w.max_remap_defect>3.e-12)throw std::runtime_error("translated patch conservation failed");
    bool rejected=false;try{pr.Mesh(pr.translated_center_max+1.e-4);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("translated patch range guard failed");
  }
  {
    TrackOptions o;o.initial="tanh";o.width=.01;o.base.alpha=.0004;o.base.elements=8;o.base.order=7;o.mode="detected";o.grading=3.;o.mesh_features="two";
    TrackingProblem pr(o);TrackWork w;const State first=pr.edges;pr.Mesh(o.center,pr.Initial(),w);
    if(pr.feature_active)throw std::runtime_error("two-feature sensor activated on monotone initial tanh");
    State u(pr.ne*pr.n*4,0.);for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];u[pr.Id(e,i,0)]=2.;u[pr.Id(e,i,1)]=.4;u[pr.Id(e,i,3)]=10.-50.*(x-.15)*(x-.15);}
    const State next=pr.Mesh(o.center,u,w);
    if(!pr.feature_active||std::abs(pr.feature_center-.15)>2.e-8||int(next.size())!=pr.ne+1)throw std::runtime_error("two-feature current polynomial sensor failed");
    if(std::find(next.begin(),next.end(),first[pr.patch_index-o.radius])!=next.end())throw std::runtime_error("two-feature upstream edge exchange failed");
    const State remapped=pr.Remap(u,next,w);if(w.max_remap_defect>3.e-12)throw std::runtime_error("two-feature conservative remap failed");
    State sigma(pr.ns,0.);for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];sigma[pr.SigmaIndex(e,i)]=1.-(x-o.center)*(x-o.center);}
    if(std::abs(pr.Detect(sigma,w)-o.center)>2.e-8)throw std::runtime_error("two-feature physical stress detector failed");
    const State repeated=pr.Mesh(o.center,remapped,w);
    for(int i=0;i<=pr.ne;++i)if(std::abs(repeated[i]-next[i])>3.e-8)throw std::runtime_error("two-feature deterministic mesh reconstruction failed");
  }
  for(int p:{3,5,7})for(const std::string space:{"h1","l2"})
  {
    TrackOptions o;o.initial="entropy";o.closure_space=space;o.closure_boundary="dirichlet";
    o.base.elements=12;o.base.order=p;o.base.alpha=.0004;o.left=0.;o.right=1.;o.center=.51;o.mode="frozen";o.grading=2.;
    TrackingProblem pr(o);TrackWork w;State u(pr.ne*pr.n*4,0.);
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {u[pr.Id(e,i,0)]=2.;u[pr.Id(e,i,3)]=3.;}
    BandMatrix a;State b;pr.AssembleClosure(u,0.,w,a,b);
    for(int i=0;i<pr.ns;++i)for(int j=std::max(0,i-a.k);j<std::min(pr.ns,i+a.k+1);++j)
      if(std::abs(a.Get(i,j)-a.Get(j,i))>1.e-12)throw std::runtime_error("closure matrix symmetry failed");
    // Independent manufactured sigma=x(1-x), rho=2 gives f=(sigma+2alpha)/2.
    // Exact consistency requires the SIPG flux/jump signs to cancel all
    // interior boundary terms, including on the graded nonuniform partition.
    std::fill(b.begin(),b.end(),0.);State exact(pr.ns,0.);
    for(int e=0;e<pr.ne;++e)
    {
      const double h=pr.edges[e+1]-pr.edges[e];
      for(int i=0;i<pr.n;++i){const double x=pr.edges[e]+h*pr.physics.basis.nodes[i];exact[pr.SigmaIndex(e,i)]=x*(1.-x);}
      for(std::size_t k=0;k<pr.weights.size();++k)
      {
        // Recover the quadrature coordinate from the exact degree-one map.
        double z=0.;for(int i=0;i<pr.n;++i)z+=pr.shapes[k][i]*pr.physics.basis.nodes[i];
        const double x=pr.edges[e]+h*z,source=.5*(x*(1.-x)+2.*o.base.alpha);
        for(int i=0;i<pr.n;++i)b[pr.SigmaIndex(e,i)]+=h*pr.weights[k]*pr.shapes[k][i]*source;
      }
    }
    b.front()=b.back()=0.;State residual(pr.ns,0.);
    for(int i=0;i<pr.ns;++i){residual[i]=-b[i];for(int j=std::max(0,i-a.k);j<std::min(pr.ns,i+a.k+1);++j)residual[i]+=a.Get(i,j)*exact[j];}
    if(Norm(residual)/Norm(b)>2.e-10)throw std::runtime_error("manufactured H1/SIPG consistency failed");
    BandFactor factor(a,0,pr.ns);factor.Solve(b,1);
    for(int i=0;i<pr.ns;++i)if(std::abs(b[i]-exact[i])>2.e-9)throw std::runtime_error("manufactured H1/SIPG recovery failed");
    // The constant test must retain the complete broken nonlinear source mean
    // under natural boundary conditions, with no separate source projection.
    pr.o.closure_boundary="natural";
    for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
    {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.physics.basis.nodes[i];const double v=.1+.02*x;
      u[pr.Id(e,i,1)]=2.*v;u[pr.Id(e,i,3)]=3.+v*v;}
    const State sigma=pr.Closure(u,0.,w);double mean=0.;
    for(int e=0;e<pr.ne;++e)for(std::size_t k=0;k<pr.weights.size();++k)
    {double value=0.;for(int i=0;i<pr.n;++i)value+=pr.shapes[k][i]*sigma[pr.SigmaIndex(e,i)];mean+=(pr.edges[e+1]-pr.edges[e])*pr.weights[k]*value/2.;}
    if(std::abs(mean-w.current_source_integral)>2.e-13)throw std::runtime_error("natural closure source mean balance failed");
    pr.o.base.alpha=0.;if(Norm(pr.Closure(u,0.,w))>1.e-13)throw std::runtime_error("alpha-zero closure failed");
  }
  std::cout<<"hp front tracking self-test passed: p3/p5/p7 constants, degree-p recovery, four conserved integrals, boundary budget, current closure; H1/SIPG symmetry/SPD/manufactured consistency, natural source balance, alpha zero\n";
}
}
#ifndef DG_IGR_HP_FRONT_TRACKING_LIBRARY_ONLY
int main(int argc,char** argv)
{
  try
  {
    mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");const auto o=TrackParse(argc,argv);if(o.self_test){TrackSelfTest();return 0;}
    const auto setup=Clock::now();TrackingProblem pr(o);const double setup_seconds=Wall(setup);auto result=TrackRun(pr);result.work.setup=setup_seconds;result.wall+=setup_seconds;
    if(o.output.empty())TrackWrite(pr,result,std::cout);else{std::ofstream file(o.output);if(!file)throw std::runtime_error("cannot open output");TrackWrite(pr,result,file);}
    return result.failure.empty()?0:2;
  }
  catch(const std::exception& e){std::cerr<<"hp front tracking failed: "<<e.what()<<'\n';return 2;}
}
#endif

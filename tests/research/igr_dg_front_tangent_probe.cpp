// Test-only retained-moment directional derivatives at an actual saved front.
// Reuses the complete primitive DG, LLF, instantaneous SIPG and face corrections.
// Frozen-stress/source/coefficient branches are diagnostics, not new time schemes.
#define DG_IGR_HP_PATCH_LIBRARY_ONLY
#include "igr_dg_hp_patch_probe.cpp"
#undef DG_IGR_HP_PATCH_LIBRARY_ONLY
#include "igr_dg_p_variational_native.hpp"
#include "igr_dg_p_dgsem_native.hpp"
#include "igr_dg_p_modal_transport.hpp"
#include "igr_dg_p_gradient_bulk.hpp"
#include "igr_dg_p_flux_split.hpp"
#include "igr_dg_p_native_pressure_source.hpp"
#include "igr_dg_p_native_cap_flux.hpp"
#include "igr_dg_p_workcap_native_operator.hpp"
#include "igr_dg_p_obstacle_native_operator.hpp"

namespace
{
struct FrontWork
{
  TrackWork shared;
  std::uint64_t states=0,factors=0,solves=0,rhs_calls=0,admission_points=0,tangent_velocity_factors=0,tangent_points=0;
  double assembly_seconds=0.,factor_seconds=0.,solve_seconds=0.,certificate_seconds=0.,analytic_seconds=0.;
  double maximum_solve_residual=0.;
};
void FrontString(std::ostream& out,const std::string& value)
{
  out<<'"';for(unsigned char c:value)
  {if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else if(c=='\r')out<<"\\r";else if(c=='\t')out<<"\\t";else if(c<32)out<<'?';else out<<c;}out<<'"';
}
State FrontRead(const std::string& path,std::size_t count)
{
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open front input: "+path);State out(count);
  for(double& value:out)if(!(in>>value)||!std::isfinite(value))throw std::runtime_error("invalid front input entries: "+path);
  in>>std::ws;if(!in.eof())throw std::runtime_error("extra front input entries: "+path);return out;
}
State FrontDifference(const State& a,const State& b,double denominator=1.)
{
  if(a.size()!=b.size()||denominator==0.)throw std::runtime_error("front difference shape/scale mismatch");State out(a.size());
  for(std::size_t i=0;i<a.size();++i){out[i]=(a[i]-b[i])/denominator;if(!std::isfinite(out[i]))throw std::runtime_error("nonfinite front difference");}return out;
}
State FrontSum(const State& a,const State& b,double scale=1.)
{if(a.size()!=b.size())throw std::runtime_error("front sum shape mismatch");State out=a;for(std::size_t i=0;i<a.size();++i)out[i]+=scale*b[i];return out;}
double FrontRelative(const State& a,const State& b)
{return Norm(FrontDifference(a,b))/std::max({Norm(a),Norm(b),1.e-12});}
State FrontApply(const BandMatrix& a,const State& u)
{
  if(int(u.size())!=a.n)throw std::runtime_error("front band action size mismatch");State out(a.n,0.);
  for(int i=0;i<a.n;++i)for(int j=std::max(0,i-a.k);j<std::min(a.n,i+a.k+1);++j)out[i]+=a.Get(i,j)*u[j];return out;
}
BandFactor FrontFactor(const BandMatrix& a,FrontWork& work)
{const auto start=Clock::now();BandFactor factor(a,0,a.n);++work.factors;work.factor_seconds+=Wall(start);return factor;}
State FrontSolve(const BandMatrix& a,const BandFactor& factor,const State& b,FrontWork& work)
{
  const auto start=Clock::now();State u=b;factor.Solve(u,1);++work.solves;work.solve_seconds+=Wall(start);
  const auto certificate=Clock::now();const double defect=Norm(FrontDifference(FrontApply(a,u),b))/std::max(Norm(b),1.e-12);
  if(!std::isfinite(defect)||defect>1.e-8)throw std::runtime_error("front closure solve residual failed");
  work.maximum_solve_residual=std::max(work.maximum_solve_residual,defect);work.certificate_seconds+=Wall(certificate);return u;
}
PatchGeometry FrontGeometry(const State& edges,int ne,double left,double right)
{
  if(edges.size()!=std::size_t(ne+1)||std::abs(edges.front()-left)>1.e-13||std::abs(edges.back()-right)>1.e-13)
    throw std::runtime_error("saved mesh does not match the declared domain/elements");
  PatchGeometry g;g.edges=edges;g.left=left;g.right=right;g.inner_elements=ne;g.hmin=INFINITY;
  for(int e=0;e<ne;++e){const double h=edges[e+1]-edges[e];if(!(h>0.)||!std::isfinite(h))throw std::runtime_error("invalid saved physical edge");g.hmin=std::min(g.hmin,h);g.hmax=std::max(g.hmax,h);
    if(e){const double previous=edges[e]-edges[e-1];g.maximum_adjacent_ratio=std::max(g.maximum_adjacent_ratio,std::max(h/previous,previous/h));}}
  return g;
}
MemoryOptions FrontConstructionOptions(const MemoryOptions& supplied,bool wall_study)
{
  auto out=supplied;
  if(wall_study)
  {
    // The inherited central-source constructor only qualifies interior traces.
    // Build that unchanged interior operator, then install the boundary tags
    // before any evaluation. NativePressureSource supplies the separately
    // checked reflected wall derivative; no inherited source path is used.
    auto& b=out.recovery.boundary;
    if(b.wall_source!="interior")throw std::runtime_error("wall input construction requires interior base-source setting");
    if(b.left=="wall")b.left="outflow";
    if(b.right=="wall")b.right="outflow";
  }
  return out;
}
struct FrontProblem:CurvatureProblem
{
  PatchGeometry geometry;
  State rhs_offset,source_offset_load;
  bool replace_source=false;
  bool retained_transport=false;
  bool input_boundary_study=false;
  std::string transport_form="weak-exact";
  double top_transport_factor=1.,top_rhs_factor=1.;
  double top_central_factor=1.,top_llf_factor=1.;
  double interior_llf_multiplier=1.;
  int top_component_mask=15;
  std::unique_ptr<FrontBulkGradient> bulk_gradient;
  std::unique_ptr<NativePressureSource> constitutive_source;
  std::unique_ptr<NativePhysicalCap> physical_cap;
  std::unique_ptr<NativeWorkCap> work_cap;
  std::unique_ptr<NativeObstacleStress> obstacle_stress;
  double bulk_gradient_setup_seconds=0.;
  bool common_transport=false;
  bool minimum_gradient_pressure=false;
  bool retained_common_jump=false;
  bool retained_common_flux=false;
  mutable std::array<double,3> retained_face_minima{{INFINITY,INFINITY,INFINITY}};
  mutable std::uint64_t retained_face_admission_points=0;
  std::unique_ptr<MinimumGradientRecovery> minimum_gradient;
  mutable std::unique_ptr<MinimumGradientCommonState> common_state;
  mutable double common_state_seconds=0.,common_max_moment_defect=0.;
  mutable std::array<double,3> common_minima{{INFINITY,INFINITY,INFINITY}};
  mutable MinimumGradientResult minimum_gradient_result;
  mutable MinimumGradientResult common_pressure_result;
  mutable std::uint64_t common_pressure_calls=0;
  mutable double common_pressure_seconds=0.;
  mutable State minimum_gradient_moments;
  mutable double minimum_gradient_seconds=0.,minimum_gradient_max_moment=0.,minimum_gradient_max_value_jump=0.,minimum_gradient_max_derivative_jump=0.;
  mutable std::uint64_t minimum_gradient_calls=0;
  double minimum_gradient_setup_seconds=0.;
  double source_offset_integral=0.;
  mutable double offset_seconds=0.,source_offset_seconds=0.,maximum_negative_source_cell_mean=0.;
  mutable std::uint64_t offset_calls=0,source_offset_calls=0;
  FrontProblem(const MemoryOptions& mo,const State& edges,double slope,double curvature,bool boundary_study=false):CurvatureProblem(FrontConstructionOptions(mo,boundary_study),slope,curvature),
    geometry(FrontGeometry(edges,ne,o.left,o.right))
  {
    if(boundary_study)
      for(BoundaryProblem* space:{static_cast<BoundaryProblem*>(this),static_cast<BoundaryProblem*>(&retained),static_cast<BoundaryProblem*>(&stress)})
      {space->boundary.left=mo.recovery.boundary.left;space->boundary.right=mo.recovery.boundary.right;}
    SetPatchGeometry(*this,geometry,geometry.hmax-geometry.hmin<1.e-13?"uniform":"saved-nonuniform");
  }
  // A frozen, test-only conservative residual correction. Its construction
  // and any outer refitting cadence belong to the experiment, not the closure.
  void SetOffset(const State& low)
  {
    if(low.size()!=std::size_t(retained.ns*4))throw std::runtime_error("front offset size mismatch");
    for(double x:low)if(!std::isfinite(x))throw std::runtime_error("nonfinite front offset");
    const auto mean=retained.Integrals(low,edges);
    for(double x:mean)if(std::abs(x)>1.e-8)throw std::runtime_error("front offset changes the physical boundary budget");
    rhs_offset=state_transfer.Apply(state_transfer.E,low,ne,4);
  }
  // Values of a signed source correction at retained LGL nodes. Integrate
  // with the exact physical mass matrix; this is a volume source, not a flux.
  void SetSourceOffset(const State& values)
  {
    if(values.size()!=std::size_t(stress.ns))throw std::runtime_error("front source offset size mismatch");
    for(double x:values)if(!std::isfinite(x))throw std::runtime_error("nonfinite front source offset");
    source_offset_load.assign(values.size(),0.);source_offset_integral=0.;const int m=stress.n;
    for(int e=0;e<ne;++e)for(int i=0;i<m;++i)for(int j=0;j<m;++j)
      source_offset_load[e*m+i]+=(edges[e+1]-edges[e])*stress.mass(i,j)*values[e*m+j];
    for(double x:source_offset_load)source_offset_integral+=x;
  }
  void EnableMinimumGradient()
  {const auto start=Clock::now();minimum_gradient=std::make_unique<MinimumGradientRecovery>(retained);minimum_gradient_setup_seconds=Wall(start);}
  void EnableBulkGradient(double strength,bool top_strain=false)
  {const auto start=Clock::now();bulk_gradient=std::make_unique<FrontBulkGradient>(*this,low_derivatives,strength,top_strain);bulk_gradient_setup_seconds=Wall(start);}
  void RecoverMinimumGradient(const State& low)const
  {
    const auto start=Clock::now();minimum_gradient_result=minimum_gradient->Recover(low,edges,o.base.alpha);minimum_gradient_moments=low;
    minimum_gradient_max_moment=std::max(minimum_gradient_max_moment,minimum_gradient_result.moment_defect);
    minimum_gradient_max_value_jump=std::max(minimum_gradient_max_value_jump,minimum_gradient_result.value_jump);
    minimum_gradient_max_derivative_jump=std::max(minimum_gradient_max_derivative_jump,minimum_gradient_result.derivative_jump);
    ++minimum_gradient_calls;minimum_gradient_seconds+=Wall(start);
    if(common_transport)
    {
      const auto common_start=Clock::now();common_state=std::make_unique<MinimumGradientCommonState>(retained,*minimum_gradient,minimum_gradient_result,low);
      if(minimum_gradient_pressure)
      {
        // Recover the physical pressure, not specific internal energy: at
        // constant rho this pairs the exact same unweighted R with velocity.
        const auto pressure_start=Clock::now();State input(low.size(),0.);
        for(int i=0;i<retained.ns;++i){input[4*i]=1.;input[4*i+1]=common_state->pressure[i];}
        common_pressure_result=minimum_gradient->Recover(input,edges,0.);++common_pressure_calls;
        common_state->pressure_legendre=common_pressure_result.coefficients;
        common_state->Certify(*minimum_gradient,minimum_gradient_result,low);common_pressure_seconds+=Wall(pressure_start);
      }
      common_max_moment_defect=std::max(common_max_moment_defect,common_state->moment_defect);common_state_seconds+=Wall(common_start);
      CommonAdmission(State(stress.ns,0.));
    }
  }
  std::array<double,4> CommonPoint(int e,const State& nodal,const State& legendre)const
  {return common_state->Point(*minimum_gradient,minimum_gradient_result,minimum_gradient_moments,e,nodal,legendre);}
  std::array<double,4> CommonEndpoint(int e,int side)const
  {
    State v,d;MinimumGradientRecovery::Legendre(side?1.:-1.,minimum_gradient->q,v,d);
    return CommonPoint(e,retained.Shape(side?1.:0.,false),v);
  }
  std::array<double,4> RetainedFaceState(const State& low,int node,double sigma)const
  {
    std::array<double,4> u{};for(int c=0;c<4;++c)u[c]=low[4*node+c];
    const double pressure=(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
    const double q=gamma_euler*pressure+(gamma_euler-1.)*sigma;
    if(!std::isfinite(u[0])||!std::isfinite(pressure)||!std::isfinite(q)||!(u[0]>floor_value)||!(pressure>floor_value)||!(q>floor_value))
      throw std::runtime_error("retained interior Euler face state or Q admission failed; no fallback applied");
    retained_face_minima[0]=std::min(retained_face_minima[0],u[0]);retained_face_minima[1]=std::min(retained_face_minima[1],pressure);retained_face_minima[2]=std::min(retained_face_minima[2],q);++retained_face_admission_points;
    return u;
  }
  State CommonBoundaryAdapter()const
  {
    // BoundaryFlux reads only the two exterior endpoint states. This adapter
    // shares the existing application BC, without interpolating a volume state.
    State adapter(ns*4,0.);
    for(int side=0;side<2;++side)
    {const int e=side?ne-1:0,j=side?p:0;const auto u=CommonEndpoint(e,side);for(int c=0;c<4;++c)adapter[Id(e,j,c)]=u[c];}
    return adapter;
  }
  std::array<double,3> CommonAdmission(const State& low_sigma)const
  {
    const auto& recovery=*minimum_gradient;std::array<double,3> out{{INFINITY,INFINITY,INFINITY}};
    auto check=[&](const std::array<double,4>& u,double s)
    {
      const double pressure=(gamma_euler-1.)*(u[3]-.5*u[1]*u[1]/u[0]),q=gamma_euler*pressure+(gamma_euler-1.)*s;
      if(!std::isfinite(u[0])||!std::isfinite(pressure)||!std::isfinite(q)||!(u[0]>floor_value)||!(pressure>floor_value)||!(q>floor_value))
        throw std::runtime_error("minimum-gradient common state or Q admission failed; no filtering or fallback applied");
      out[0]=std::min(out[0],u[0]);out[1]=std::min(out[1],pressure);out[2]=std::min(out[2],q);
    };
    for(int e=0;e<ne;++e)
    {
      for(int k=0;k<recovery.nq;++k)
      {double s=0.;for(int i=0;i<retained.n;++i)s+=recovery.nodal[k][i]*low_sigma[e*retained.n+i];check(CommonPoint(e,recovery.nodal[k],recovery.values[k]),s);}
      for(int side=0;side<2;++side)check(CommonEndpoint(e,side),low_sigma[e*retained.n+(side?retained.p:0)]);
    }
    for(int c=0;c<3;++c)common_minima[c]=std::min(common_minima[c],out[c]);return out;
  }
  void Limit(State& moments,TrackWork& work)const
  {
    if(!common_transport){CurvatureProblem::Limit(moments,work);return;}
    // The nonlinear common state is admitted at every following closure.
    // No limiter based on the former primitive state may edit its moments.
    for(double x:moments)if(!std::isfinite(x))throw std::runtime_error("nonfinite common-state conservative moments");
  }
  State Closure(const State& moments,double time,TrackWork& work)const
  {
    State result=CurvatureProblem::Closure(moments,time,work);
    if(physical_cap)physical_cap->Admit(*this,Reconstruct(moments),result);
    if(common_transport)CommonAdmission(stress_transfer.Apply(stress_transfer.P,result,ne,1));return result;
  }
  std::array<double,4> BoundaryFlux(const State& physical,const State& raw,int side)const
  {
    auto result=physical_cap?physical_cap->BoundaryFlux(*this,physical,raw,side):CurvatureProblem::BoundaryFlux(physical,raw,side);
    if(work_cap){const auto extra=work_cap->BoundaryCorrection(*this,physical,raw,side);for(int c=0;c<4;++c)result[c]+=extra[c];}
    return result;
  }
  State CommonResidual(const State& moments,const State& sigma,TrackWork& work,std::array<double,4>& budget)const
  {
    const auto start=Clock::now();const auto& recovery=*minimum_gradient;const int m=retained.n;
    const State low=state_transfer.Apply(state_transfer.P,moments,ne,4);
    if(low!=minimum_gradient_moments)throw std::runtime_error("common transport and instantaneous source do not share the same current moments");
    const State low_sigma=stress_transfer.Apply(stress_transfer.P,sigma,ne,1);CommonAdmission(low_sigma);
    State load(ne*m*4,0.),rhs(ne*m*4,0.);
    for(int e=0;e<ne;++e)for(int k=0;k<recovery.nq;++k)
    {
      const auto u=CommonPoint(e,recovery.nodal[k],recovery.values[k]);double s=0.,speed=0.;for(int j=0;j<m;++j)s+=recovery.nodal[k][j]*low_sigma[e*m+j];
      const auto flux=physics.Flux(u.data(),s,speed);++work.flux_points;
      for(int i=0;i<m;++i)for(int c=0;c<4;++c)load[(e*m+i)*4+c]+=.5*recovery.weights[k]*recovery.nodal_derivatives[k][i]*flux[c];
    }
    const State adapter=CommonBoundaryAdapter();budget.fill(0.);
    for(int face=0;face<=ne;++face)
    {
      std::array<double,4> flux{};
      if(face==0||face==ne)flux=BoundaryFlux(adapter,sigma,face==ne);
      else
      {
        const auto ul=retained_common_flux?RetainedFaceState(low,face*m-1,low_sigma[face*m-1]):CommonEndpoint(face-1,1);
        const auto ur=retained_common_flux?RetainedFaceState(low,face*m,low_sigma[face*m]):CommonEndpoint(face,0);double al=0.,ar=0.;
        const auto fl=physics.Flux(ul.data(),low_sigma[(face-1)*m+m-1],al),fr=physics.Flux(ur.data(),low_sigma[face*m],ar);const double speed=physics.FaceSpeed(al,ar);
        for(int c=0;c<4;++c)
        {
          // In jump-only mode the retained vector need not be admissible.
          // Full retained-flux mode instead admits both retained traces above
          // and uses their own physical flux and IGR-adjusted wave speed.
          const double jump=retained_common_jump?low[(face*m)*4+c]-low[((face-1)*m+m-1)*4+c]:ur[c]-ul[c];
          flux[c]=.5*(fl[c]+fr[c]-speed*jump);
        }
      }
      ++work.faces;
      for(int c=0;c<4;++c)
      {if(face)load[((face-1)*m+m-1)*4+c]-=flux[c];if(face<ne)load[face*m*4+c]+=flux[c];if(!face)budget[c]=flux[c];if(face==ne)budget[c]-=flux[c];}
    }
    for(int e=0;e<ne;++e)for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<4;++c)
      rhs[(e*m+i)*4+c]+=retained.mass_inverse(i,j)*load[(e*m+j)*4+c]/(edges[e+1]-edges[e]);
    const auto totals=retained.Integrals(rhs,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    work.transport+=Wall(start);return state_transfer.Apply(state_transfer.E,rhs,ne,4);
  }
  void MixedSystem(const State& physical,double time,TrackWork& work,BandMatrix& a,State& b)const override
  {
    if(common_transport)
    {
      // This existing SIPG assembler depends only on rho. Supplying an
      // admissible zero-velocity carrier prevents the former primitive thermal
      // pressure from deciding admission of the actual common state.
      const auto start=Clock::now();State coefficient_state=physical;
      for(std::size_t i=0;i<coefficient_state.size();i+=4)
      {coefficient_state[i+1]=coefficient_state[i+2]=0.;coefficient_state[i+3]=1./(gamma_euler-1.);}
      PrimitiveDirectSystem(*this,low_derivatives,coefficient_state,work,a,b);direct_assembly_seconds+=Wall(start);
    }
    else CurvatureProblem::MixedSystem(physical,time,work,a,b);
    if(obstacle_stress){obstacle_stress->Update(*this,physical,work,a,b);return;}
    if(constitutive_source)
    {constitutive_source->Update(*this,physical,work,a,b);return;}
    if(minimum_gradient)
    {
      // Closure dispatch receives the current primitive physical reconstruction.
      // Its cached input contains the exact retained conservative moments,
      // avoiding a second nonlinear physical-to-moment projection.
      if(physical!=cached_physical)throw std::runtime_error("minimum-gradient source requires the current reconstructed state");
      RecoverMinimumGradient(state_transfer.Apply(state_transfer.P,cached_moments,ne,4));
      b=minimum_gradient_result.load;work.current_source_integral=minimum_gradient_result.source_integral;
      work.source_points+=std::uint64_t(ne)*minimum_gradient->nq;
      return;
    }
    if(source_offset_load.empty())return;
    const auto start=Clock::now();
    if(replace_source){b=source_offset_load;work.current_source_integral=source_offset_integral;}
    else{for(std::size_t i=0;i<b.size();++i)b[i]+=source_offset_load[i];work.current_source_integral+=source_offset_integral;}
    for(int e=0;e<ne;++e)
    {double total=0.;for(int i=0;i<stress.n;++i)total+=b[e*stress.n+i];maximum_negative_source_cell_mean=std::max(maximum_negative_source_cell_mean,-total/(edges[e+1]-edges[e]));}
    ++source_offset_calls;source_offset_seconds+=Wall(start);
  }
  State Residual(const State& moments,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    if(physical_cap)return physical_cap->Residual(*this,Reconstruct(moments),sigma,work,budget);
    State rhs=common_transport?CommonResidual(moments,sigma,work,budget):CurvatureProblem::Residual(moments,sigma,time,work,budget);
    if(work_cap){work_cap->Add(*this,Reconstruct(moments),sigma,rhs,work,budget);return rhs;}
    if(retained_transport)
    {
      // Causal research switch: replace only ordinary volume/LLF transport.
      // Keep the primitive-source closure and both existing derivative penalties.
      // The three transport evaluations below are deliberate redundant work.
      const auto start=Clock::now();
      for(int e=0;e<ne;++e)for(const State& shape:positive)
      {
        const auto u=Point(moments,e,shape);double s=0.;
        for(int i=0;i<n;++i)s+=shape[i]*sigma[e*n+i];
        const double pressure=(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
        const double q=gamma_euler*pressure+(gamma_euler-1.)*s;
        if(!std::isfinite(u[0])||!std::isfinite(pressure)||!std::isfinite(q)||
           !(u[0]>floor_value)||!(pressure>floor_value)||!(q>floor_value))
          throw std::runtime_error("retained conservative transport admission failed");
      }
      work.transport+=Wall(start);
      std::array<double,4> old_budget{},new_budget{};
      const State old_rhs=PrimitiveDirectResidual(*this,low_derivatives,Reconstruct(moments),sigma,work,old_budget);
      const State new_rhs=PrimitiveDirectResidual(*this,low_derivatives,moments,sigma,work,new_budget);
      for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=new_rhs[i]-old_rhs[i];
      for(int c=0;c<4;++c)budget[c]+=new_budget[c]-old_budget[c];
      const auto totals=Integrals(rhs,edges);
      for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    }
    if(transport_form!="weak-exact")
    {
      if(common_transport)throw std::runtime_error("alternative transport form has no common-state qualification");
      const State& physical=retained_transport?moments:Reconstruct(moments);
      std::array<double,4> old_budget{},new_budget{};
      const State old=PrimitiveDirectResidual(*this,low_derivatives,physical,sigma,work,old_budget);
      const State replacement=FrontAlternativeTransport(*this,low_derivatives,physical,sigma,transport_form,work,new_budget);
      for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=replacement[i]-old[i];
      for(int c=0;c<4;++c)budget[c]+=new_budget[c]-old_budget[c];
    }
    if(bulk_gradient)bulk_gradient->Add(*this,Reconstruct(moments),sigma,rhs,work);
    if(top_central_factor!=1.||top_llf_factor!=1.||interior_llf_multiplier!=1.)
    {
      const State& physical=retained_transport?moments:Reconstruct(moments);
      const State llf=FrontInteriorLLFResidual(*this,physical,sigma,work);
      State change(rhs.size(),0.);
      if(interior_llf_multiplier!=1.)
        for(std::size_t i=0;i<change.size();++i)change[i]=(interior_llf_multiplier-1.)*llf[i];
      if(top_central_factor!=1.)
      {
        std::array<double,4> ignored{};
        State central=PrimitiveDirectResidual(*this,low_derivatives,physical,sigma,work,ignored);
        const auto start=Clock::now();
        for(std::size_t i=0;i<central.size();++i)central[i]-=llf[i];
        change=FrontTopModeDelta(*this,central,top_central_factor,15);
        work.transport+=Wall(start);
      }
      const auto start=Clock::now();
      if(top_llf_factor!=1.)
      {
        const State delta=FrontTopModeDelta(*this,llf,top_llf_factor,15);
        for(std::size_t i=0;i<change.size();++i)change[i]+=delta[i];
      }
      for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=change[i];
      work.transport+=Wall(start);
    }
    if(top_transport_factor!=1.)
    {
      std::array<double,4> ignored{};
      const State& physical=retained_transport?moments:Reconstruct(moments);
      const State transport=PrimitiveDirectResidual(*this,low_derivatives,physical,sigma,work,ignored);
      const auto start=Clock::now();
      const State change=FrontTopModeDelta(*this,transport,top_transport_factor,top_component_mask);
      for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=change[i];
      work.transport+=Wall(start);
    }
    if(top_rhs_factor!=1.)
    {
      const auto start=Clock::now();const State change=FrontTopModeDelta(*this,rhs,top_rhs_factor,top_component_mask);
      for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=change[i];
      work.transport+=Wall(start);
    }
    if(top_transport_factor!=1.||top_rhs_factor!=1.||top_central_factor!=1.||top_llf_factor!=1.||interior_llf_multiplier!=1.)
    {
      const auto totals=Integrals(rhs,edges);
      for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    }
    if(rhs_offset.empty())return rhs;
    const auto start=Clock::now();
    for(std::size_t i=0;i<rhs.size();++i)rhs[i]+=rhs_offset[i];
    const auto totals=Integrals(rhs,edges);
    for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    ++offset_calls;const double elapsed=Wall(start);offset_seconds+=elapsed;work.transport+=elapsed;
    return rhs;
  }
};
void FrontWriteBulkGradient(std::ostream& out,const FrontProblem& pr)
{
  out<<",\"bulk_gradient_strength\":"<<(pr.bulk_gradient?pr.bulk_gradient->strength:0.);
  if(!pr.bulk_gradient)return;const auto& b=*pr.bulk_gradient;
  out<<",\"bulk_gradient_policy\":\""<<(b.top_strain?"top-strain":"upper-half")<<"\",\"bulk_gradient_maximum_activity\":"<<b.maximum_activity
     <<",\"bulk_gradient_maximum_strain_ratio_squared\":"<<b.maximum_strain_ratio_squared;
  out<<",\"bulk_gradient_cutoff\":"<<b.cutoff<<",\"bulk_gradient_calls\":"<<b.calls
     <<",\"bulk_gradient_local_factors\":"<<b.local_factors<<",\"bulk_gradient_seconds\":"<<b.seconds
     <<",\"bulk_gradient_setup_seconds\":"<<pr.bulk_gradient_setup_seconds
     <<",\"bulk_gradient_mean_defect\":"<<b.maximum_mean_defect<<",\"bulk_gradient_work_defect\":"<<b.maximum_work_defect
     <<",\"bulk_gradient_maximum_dissipation\":"<<b.maximum_dissipation<<",\"bulk_gradient_maximum_rate_bound\":"<<b.maximum_rate_bound;
}
void FrontWriteConstitutiveSource(std::ostream& out,const FrontProblem& pr)
{
  if(pr.obstacle_stress)
  {
    const auto& s=*pr.obstacle_stress;
    out<<",\"igr_source_law\":\"pressure-compression\",\"obstacle_stress\":true,\"obstacle_tau\":"<<s.tau<<",\"obstacle_bound\":"<<s.bound<<",\"obstacle_margin\":"<<s.margin
       <<",\"obstacle_unconstrained_trials\":"<<s.unconstrained_trials<<",\"obstacle_unconstrained_accepts\":"<<s.unconstrained_accepts
       <<",\"obstacle_tolerance\":"<<s.tolerance<<",\"obstacle_calls\":"<<s.calls<<",\"obstacle_factors\":"<<s.factors<<",\"obstacle_qp_solves\":"<<s.qp_solves<<",\"obstacle_source_points\":"<<s.source_points
       <<",\"obstacle_maximum_qp_iterations\":"<<s.maximum_qp_iterations<<",\"obstacle_maximum_cuts\":"<<s.maximum_cuts<<",\"obstacle_maximum_picard\":"<<s.maximum_picard<<",\"obstacle_seconds\":"<<s.seconds
       <<",\"obstacle_maximum_kkt\":"<<s.maximum_kkt<<",\"obstacle_maximum_gap\":"<<s.maximum_gap<<",\"obstacle_maximum_nonlinear\":"<<s.maximum_nonlinear
       <<",\"obstacle_minimum_stress\":"<<s.minimum_stress<<",\"obstacle_minimum_work_slack\":"<<s.minimum_work<<",\"obstacle_minimum_pressure\":"<<s.minimum_pressure
       <<",\"obstacle_physical_source_integral\":"<<s.physical_source_integral<<",\"obstacle_effective_source_integral\":"<<s.effective_source_integral<<",\"obstacle_energy_bound_ratio\":"<<s.energy_bound_ratio
       <<",\"obstacle_policy\":\"research changed continuum; current-stress nonlinear VI in full retained Pp SIPG; tightened sampled constraints with polynomial extrema certificate of physical bounds; same constrained stress in shared full-energy Euler and LLF; effective A Sigma adapter, physical source and reaction separate; all redundant factors charged\""
       <<",\"obstacle_last_stress\":";TrackArray(out,s.last_sigma);out<<",\"obstacle_last_physical_load\":";TrackArray(out,s.last_physical_load);
    out<<",\"obstacle_last_effective_load\":";TrackArray(out,s.last_effective_load);out<<",\"obstacle_last_reaction\":";TrackArray(out,s.last_reaction);
    out<<",\"obstacle_last_gradient\":";TrackArray(out,s.last_gradient);out<<",\"obstacle_last_pressure\":";TrackArray(out,s.last_pressure);return;
  }

  if(pr.work_cap)
  {
    const auto& c=*pr.work_cap;
    out<<",\"work_stress_cap\":\"expansive-work-mixed-BR1\",\"workcap_identity\":"<<c.identity<<",\"workcap_bound\":"<<c.bound
       <<",\"workcap_tolerance\":"<<c.tolerance<<",\"workcap_diffusion_cfl_limit\":"<<c.diffusion_cfl_limit
       <<",\"workcap_calls\":"<<c.calls<<",\"workcap_points\":"<<c.points<<",\"workcap_intervals\":"<<c.intervals<<",\"workcap_maximum_depth\":"<<c.maximum_depth
       <<",\"workcap_seconds\":"<<c.seconds<<",\"workcap_pairing_error\":"<<c.maximum_pairing_error<<",\"workcap_energy_budget_error\":"<<c.maximum_energy_budget_error
       <<",\"workcap_maximum_diffusion_bound\":"<<c.maximum_diffusion_bound<<",\"workcap_maximum_diffusion_cfl\":"<<c.maximum_diffusion_cfl
       <<",\"workcap_maximum_mu\":"<<c.maximum_mu<<",\"workcap_minimum_physical\":"<<c.minimum_physical<<",\"workcap_minimum_pressure\":"<<c.minimum_pressure
       <<",\"workcap_maximum_work_ratio\":"<<c.maximum_work_ratio<<",\"workcap_maximum_sound\":"<<c.maximum_sound
       <<",\"workcap_last_heat\":"<<c.last_heat<<",\"workcap_last_rate_bound\":"<<c.last_rate_bound
       <<",\"workcap_policy\":\"raw closure unchanged; central lifted velocity gradient; adjoint Riesz traction and paired conservative energy; adaptive correction quadrature; active principal LLF envelope; frozen diffusion bound is not a full coupled stability theorem\""
       <<",\"workcap_cell_speed\":";TrackArray(out,c.cell_speed);
    out<<",\"workcap_face_traction\":";TrackArray(out,c.face_traction);
    out<<",\"workcap_face_velocity\":";TrackArray(out,c.face_velocity);
  }
  if(pr.physical_cap)
  {
    const auto& c=*pr.physical_cap;
    out<<",\"physical_stress_cap\":\"identity-smooth\",\"physical_stress_cap_identity\":"<<c.identity<<",\"physical_stress_cap_bound\":"<<c.bound
       <<",\"physical_stress_policy\":\"raw S remains in closure; Sigma=p*psi(S/p) evaluated independently at volume and each face in both momentum and energy; capped principal acoustic LLF speed\""
       <<",\"cap_evaluations\":"<<c.evaluations<<",\"cap_active_evaluations\":"<<c.active<<",\"cap_residual_calls\":"<<c.residual_calls<<",\"cap_seconds\":"<<c.seconds
       <<",\"cap_raw_ratio_min\":"<<c.minimum_ratio<<",\"cap_raw_ratio_max\":"<<c.maximum_ratio
       <<",\"cap_physical_ratio_min\":"<<c.minimum_physical_ratio<<",\"cap_physical_ratio_max\":"<<c.maximum_physical_ratio
       <<",\"cap_feedback_min\":"<<c.minimum_feedback<<",\"cap_sound_squared_min\":"<<c.minimum_sound_squared;
  }
  if(pr.input_boundary_study)
    out<<",\"input_boundary_study\":true,\"input_boundary_scope\":\"injected autonomous data; shared outflow/slip-wall Euler flux; natural raw stress; no supplied reference in source\"";
  out<<",\"igr_source_law\":\""<<(pr.constitutive_source?pr.constitutive_source->law:"original")<<'"';
  if(!pr.constitutive_source)return;const auto& s=*pr.constitutive_source;
  out<<",\"igr_source_model_scope\":\""<<(!pr.physical_cap&&!pr.work_cap&&pr.input_boundary_study&&s.law=="constant"&&s.kappa==2.?
       "original continuum with explicit reflected DG wall source; full-energy Euler and LLF shared":
       (pr.work_cap?"research changed continuum with expansive-work stress and paired mixed-DG traction/energy":pr.physical_cap?"research changed continuum with paired physical stress cap and capped acoustic LLF":"research changed continuum; full-energy Euler and existing DG/LLF unchanged"))<<'"'
     <<",\"igr_source_pressure_faces\":\""<<s.pressure_faces<<"\",\"igr_source_kappa\":"<<s.kappa
     <<",\"igr_source_mirror_wall\":"<<(s.mirror_wall_source?"true":"false")
     <<",\"igr_source_tolerance\":"<<s.tolerance<<",\"igr_source_calls\":"<<s.calls
     <<",\"igr_source_internal_factors\":"<<s.factors<<",\"igr_source_internal_solves\":"<<s.solves
     <<",\"igr_source_maximum_iterations\":"<<s.maximum_iterations<<",\"igr_source_seconds\":"<<s.seconds
     <<",\"igr_source_quadrature_points\":"<<s.quadrature_points<<",\"igr_source_maximum_residual\":"<<s.maximum_residual
     <<",\"igr_source_minimum_iteration_stress\":"<<s.minimum_stress<<",\"igr_source_minimum_pressure\":"<<s.minimum_pressure
     <<",\"igr_source_minimum_density\":"<<s.minimum_density<<",\"igr_source_maximum_pressure_gradient\":"<<s.maximum_pressure_gradient
     <<",\"igr_source_cost_policy\":\"internal Picard factor/solves are additional to the existing outer factor/solve and original source assembly; all elapsed work retained\""
     <<",\"igr_source_last_call_load\":";TrackArray(out,s.last_load);
  out<<",\"igr_source_last_call_internal_stress\":";TrackArray(out,s.last_sigma);
}
void FrontWriteMinimumGradient(std::ostream& out,const FrontProblem& pr)
{
  out<<",\"source_velocity\":\""<<(pr.minimum_gradient?"minimum-gradient":"primitive")<<'"';
  if(!pr.minimum_gradient)return;const auto& r=pr.minimum_gradient_result;
  out<<",\"minimum_gradient_degree\":"<<pr.minimum_gradient->q<<",\"minimum_gradient_quadrature_points\":"<<pr.minimum_gradient->nq
     <<",\"minimum_gradient_policy\":\"full density-weighted conservative Pp moments; H1 minimum-gradient recovery each closure; complete positive square against every Pp stress test; natural velocity derivative at outer endpoints\""
     <<",\"minimum_gradient_coefficient_layout\":\"cell, normalized Legendre mode 0..2p+2 on [-1,1]\""
     <<",\"minimum_gradient_calls\":"<<pr.minimum_gradient_calls<<",\"minimum_gradient_seconds\":"<<pr.minimum_gradient_seconds
     <<",\"minimum_gradient_setup_seconds\":"<<pr.minimum_gradient_setup_seconds
     <<",\"minimum_gradient_moment_defect\":"<<r.moment_defect<<",\"minimum_gradient_value_jump\":"<<r.value_jump
     <<",\"minimum_gradient_derivative_jump\":"<<r.derivative_jump<<",\"minimum_gradient_boundary_derivative\":"<<r.boundary_gradient
     <<",\"minimum_gradient_max_moment_defect\":"<<pr.minimum_gradient_max_moment
     <<",\"minimum_gradient_max_value_jump\":"<<pr.minimum_gradient_max_value_jump
     <<",\"minimum_gradient_max_derivative_jump\":"<<pr.minimum_gradient_max_derivative_jump
     <<",\"minimum_gradient_source_minimum\":"<<r.minimum_source<<",\"minimum_gradient_density_minimum\":"<<r.minimum_density
     <<",\"minimum_gradient_energy\":"<<r.energy<<",\"minimum_gradient_source_integral\":"<<r.source_integral
     <<",\"minimum_gradient_kkt_residual\":"<<r.kkt_residual<<",\"minimum_gradient_face_residual\":"<<r.face_residual;
  out<<",\"minimum_gradient_velocity_coefficients\":";TrackArray(out,r.coefficients);
  out<<",\"minimum_gradient_face_values\":";TrackArray(out,r.face_values);
  out<<",\"minimum_gradient_source_load\":";TrackArray(out,r.load);
  if(pr.common_transport)
  {
    out<<",\"common_state_policy\":\"rho=rho_p, M=rho_p*v_R, E=p_physical/(gamma-1)+.5*rho_p*v_R^2; Pi_p(p_physical)=(gamma-1)(E_p-Pi_p(.5*rho_p*v_R^2)); common v_R in Euler and IGR; all conservative Pp moments preserved\""
       <<",\"common_state_degree\":"<<5*pr.retained.p+4<<",\"common_state_weak_flux_exactness_required\":"<<8*pr.retained.p+5
       <<",\"common_state_quadrature_exactness\":"<<2*pr.minimum_gradient->nq-1
       <<",\"common_coefficient_assembly_policy\":\"existing SIPG assembled from current rho with an admissible zero-velocity carrier; common-state thermal admission only\""
       <<",\"common_state_moment_defect\":"<<pr.common_state->moment_defect<<",\"common_state_max_moment_defect\":"<<pr.common_max_moment_defect
       <<",\"common_state_seconds\":"<<pr.common_state_seconds<<",\"common_state_density_minimum\":"<<pr.common_minima[0]
       <<",\"common_state_pressure_minimum\":"<<pr.common_minima[1]<<",\"common_state_q_minimum\":"<<pr.common_minima[2]
       <<",\"common_pressure_layout\":\"cell, retained LGL node; full physical state must be evaluated from rho_p, recovered velocity coefficients and this pressure polynomial\""
       <<",\"physical_serialization_policy\":\"legacy primitive diagnostic serialization only; installed common state has degree 5p+4 and is specified by retained rho, recovered velocity and common pressure\""
       <<",\"common_pressure_coefficients\":";TrackArray(out,pr.common_state->pressure);
    out<<",\"common_pressure_policy\":\""<<(pr.minimum_gradient_pressure?"minimum-gradient":"polynomial")<<'"';
    out<<",\"common_interior_dissipative_jump\":\""<<((pr.retained_common_jump||pr.retained_common_flux)?"retained-conservative":"physical-common")<<'"'
       <<",\"common_boundary_dissipative_jump\":\"existing physical common-state application BC\"";
    out<<",\"common_interior_numerical_flux\":\""<<(pr.retained_common_flux?"retained-conservative-state":"physical-common-state")<<'"';
    if(pr.retained_common_flux)
      out<<",\"retained_face_admission_points\":"<<pr.retained_face_admission_points
         <<",\"retained_face_density_minimum\":"<<pr.retained_face_minima[0]<<",\"retained_face_pressure_minimum\":"<<pr.retained_face_minima[1]
         <<",\"retained_face_q_minimum\":"<<pr.retained_face_minima[2]
         <<",\"retained_face_flux_policy\":\"entire interior LLF flux and IGR wave speed from explicitly admissible retained conservative traces; full common volume/source and physical application boundary flux unchanged\"";
    if(pr.minimum_gradient_pressure)
    {
      const auto& p=pr.common_pressure_result;
      out<<",\"common_pressure_physical_policy\":\"unweighted H1 minimum-gradient recovery of every Pp physical-pressure moment, with natural outer derivative; U_E=p_R/(gamma-1)+.5*rho_p*v_R^2\""
         <<",\"common_pressure_exact_degree\":"<<pr.retained.p+2<<",\"common_pressure_serialized_degree\":"<<pr.minimum_gradient->q
         <<",\"common_pressure_moment_defect\":"<<p.moment_defect<<",\"common_pressure_value_jump\":"<<p.value_jump
         <<",\"common_pressure_derivative_jump\":"<<p.derivative_jump<<",\"common_pressure_boundary_derivative\":"<<p.boundary_gradient
         <<",\"common_pressure_gradient_energy\":"<<p.energy<<",\"common_pressure_calls\":"<<pr.common_pressure_calls<<",\"common_pressure_seconds\":"<<pr.common_pressure_seconds
         <<",\"common_pressure_legendre_coefficients\":";TrackArray(out,p.coefficients);
    }
  }
}
struct FrontAdmission
{double density=INFINITY,pressure=INFINITY,q_sigma=INFINITY;};
FrontAdmission FrontAdmit(const FrontProblem& pr,const State& physical,const State& sigma,FrontWork& work)
{
  FrontAdmission out;
  if(pr.common_transport)
  {
    const auto a=pr.CommonAdmission(pr.stress_transfer.Apply(pr.stress_transfer.P,sigma,pr.ne,1));
    out.density=a[0];out.pressure=a[1];out.q_sigma=a[2];work.admission_points+=pr.ne*(pr.minimum_gradient->nq+2);return out;
  }
  for(int e=0;e<pr.ne;++e)for(const State& shape:pr.positive)
  {
    const auto u=pr.Point(physical,e,shape);double s=0.;for(int i=0;i<pr.n;++i)s+=shape[i]*sigma[e*pr.n+i];
    const double pressure=(gamma_euler-1.)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
    const double q=pr.physical_cap?u[0]*pr.physical_cap->Evaluate(u.data(),s).sound_squared:gamma_euler*pressure+(gamma_euler-1.)*s;++work.admission_points;
    if(!std::isfinite(u[0])||!std::isfinite(pressure)||!std::isfinite(q)||!(u[0]>floor_value)||!(pressure>floor_value)||!(q>floor_value))
      throw std::runtime_error("perturbed reconstructed state or Q is inadmissible; no limiter applied");
    out.density=std::min(out.density,u[0]);out.pressure=std::min(out.pressure,pressure);out.q_sigma=std::min(out.q_sigma,q);
  }
  return out;
}
void FrontWriteAdmission(std::ostream& out,const FrontAdmission& a)
{out<<"{\"density_min\":"<<a.density<<",\"pressure_min\":"<<a.pressure<<",\"q_sigma_min\":"<<a.q_sigma<<'}';}
State FrontVelocity(const FrontProblem& pr,const State& physical)
{
  const int m=pr.retained.n;State v(pr.ne*m*2);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<m;++i){const auto u=pr.Point(physical,e,pr.primitive_nodes[i]);for(int c=0;c<2;++c)v[(e*m+i)*2+c]=u[c+1]/u[0];}return v;
}
State FrontGradient(const FrontProblem& pr,const State& velocity)
{
  const int m=pr.retained.n;State g(pr.ne*m,0.);
  for(int e=0;e<pr.ne;++e)
  {
    const double h=pr.edges[e+1]-pr.edges[e];
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)g[e*m+i]+=pr.retained.physics.basis.D(i,j)*velocity[(e*m+j)*2]/h;
    for(int side=0;side<2;++side)
    {
      const int neighbor=e+(side?1:-1);if(neighbor<0||neighbor>=pr.ne)continue;const int endpoint=side?m-1:0,other=side?0:m-1;
      const double jump=velocity[(e*m+endpoint)*2]-velocity[(neighbor*m+other)*2];
      for(int i=0;i<m;++i)g[e*m+i]+=(side?-1.:1.)*.5*jump*pr.retained.mass_inverse(i,endpoint)/h;
    }
  }
  return g;
}
struct FrontState
{
  State moments,high,physical,load,sigma;BandMatrix matrix;double source_integral=0.;
};
FrontState FrontAssemble(const FrontProblem& pr,const State& low,double time,FrontWork& work)
{
  FrontState f;f.moments=low;f.high=pr.state_transfer.Apply(pr.state_transfer.E,low,pr.ne,4);f.physical=pr.Reconstruct(f.high);++work.states;
  const auto start=Clock::now();pr.MixedSystem(f.physical,time,work.shared,f.matrix,f.load);f.source_integral=work.shared.current_source_integral;work.assembly_seconds+=Wall(start);return f;
}
State FrontRHS(const FrontProblem& pr,const FrontState& state,const State& low_sigma,double time,FrontWork& work,
               std::array<double,4>& budget,FrontAdmission& admission)
{
  const State sigma=pr.stress_transfer.Apply(pr.stress_transfer.E,low_sigma,pr.ne,1);admission=FrontAdmit(pr,state.physical,sigma,work);
  const State high=pr.Residual(state.high,sigma,time,work.shared,budget);++work.rhs_calls;
  return pr.state_transfer.Apply(pr.state_transfer.P,high,pr.ne,4);
}
struct FrontAnalytic
{
  State velocity,pressure,g,load,coefficient,coefficient_volume,coefficient_faces,sigma_source,sigma_coeff,sigma;
  double base_source_relative_defect=0.,closure_identity_relative_defect=0.,constant_test_defect=0.;
};
FrontAnalytic FrontDifferentiate(const FrontProblem& pr,const FrontState& base,const BandFactor& factor,
                               const State& direction,FrontWork& work)
{
  const auto start=Clock::now();FrontAnalytic out;const int m=pr.retained.n,ne=pr.ne;const double alpha=pr.o.base.alpha;
  const State v=FrontVelocity(pr,base.physical),g=FrontGradient(pr,v);out.velocity.assign(ne*m*2,0.);out.pressure.assign(ne*m,0.);
  for(int e=0;e<ne;++e)
  {
    BandMatrix mass(m,m-1);State rhs(2*m,0.);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<2;++c)rhs[c*m+i]+=pr.retained.mass(i,j)*direction[(e*m+j)*4+c+1];
    for(std::size_t q=0;q<pr.weights.size();++q)
    {
      const auto& b=pr.low_shapes[q];double rho=0.,drho=0.;std::array<double,2> velocity{};
      for(int i=0;i<m;++i){rho+=b[i]*base.moments[(e*m+i)*4];drho+=b[i]*direction[(e*m+i)*4];for(int c=0;c<2;++c)velocity[c]+=b[i]*v[(e*m+i)*2+c];}
      for(int i=0;i<m;++i){for(int j=0;j<m;++j)mass.At(i,j)+=pr.weights[q]*rho*b[i]*b[j];for(int c=0;c<2;++c)rhs[c*m+i]-=pr.weights[q]*b[i]*drho*velocity[c];}
      ++work.tangent_points;
    }
    BandFactor local(mass,0,m);local.Solve(rhs,2);++work.tangent_velocity_factors;
    for(int i=0;i<m;++i)for(int c=0;c<2;++c)out.velocity[(e*m+i)*2+c]=rhs[c*m+i];
    State kinetic_load(m,0.);
    for(std::size_t q=0;q<pr.weights.size();++q)
    {
      const auto& b=pr.low_shapes[q];double rho=0.,drho=0.;std::array<double,2> velocity{},dv{};
      for(int i=0;i<m;++i){rho+=b[i]*base.moments[(e*m+i)*4];drho+=b[i]*direction[(e*m+i)*4];for(int c=0;c<2;++c){velocity[c]+=b[i]*v[(e*m+i)*2+c];dv[c]+=b[i]*out.velocity[(e*m+i)*2+c];}}
      const double dk=.5*drho*(velocity[0]*velocity[0]+velocity[1]*velocity[1])+rho*(velocity[0]*dv[0]+velocity[1]*dv[1]);
      for(int i=0;i<m;++i)kinetic_load[i]+=pr.weights[q]*b[i]*dk;++work.tangent_points;
    }
    for(int i=0;i<m;++i){double dk=0.;for(int j=0;j<m;++j)dk+=pr.retained.mass_inverse(i,j)*kinetic_load[j];out.pressure[e*m+i]=(gamma_euler-1.)*(direction[(e*m+i)*4+3]-dk);}
  }
  out.g=FrontGradient(pr,out.velocity);out.load.assign(ne*m,0.);out.coefficient_volume.assign(ne*m,0.);out.coefficient_faces.assign(ne*m,0.);State base_load(ne*m,0.);
  for(int e=0;e<ne;++e)
  {
    const double h=pr.edges[e+1]-pr.edges[e];
    for(std::size_t q=0;q<pr.weights.size();++q)
    {
      const auto& b=pr.low_shapes[q];const auto& d=pr.low_derivatives[q];double rho=0.,drho=0.,value=0.,derivative=0.,gradient=0.,dg=0.;
      for(int i=0;i<m;++i){rho+=b[i]*base.moments[(e*m+i)*4];drho+=b[i]*direction[(e*m+i)*4];value+=b[i]*base.sigma[e*m+i];derivative+=d[i]*base.sigma[e*m+i];gradient+=b[i]*g[e*m+i];dg+=b[i]*out.g[e*m+i];}
      const double dri=-drho/(rho*rho),weight=pr.weights[q];
      for(int i=0;i<m;++i){base_load[e*m+i]+=h*weight*b[i]*2.*alpha*gradient*gradient;out.load[e*m+i]+=h*weight*b[i]*4.*alpha*gradient*dg;
        out.coefficient_volume[e*m+i]+=weight*dri*(h*b[i]*value+alpha*d[i]*derivative/h);}
      ++work.tangent_points;
    }
  }
  const State left_derivative=pr.retained.Shape(1.,true),right_derivative=pr.retained.Shape(0.,true);
  for(int face=1;face<ne;++face)
  {
    const double hl=pr.edges[face]-pr.edges[face-1],hr=pr.edges[face+1]-pr.edges[face];
    const double rl=base.moments[((face-1)*m+m-1)*4],rr=base.moments[(face*m)*4];
    const double dal=-alpha*direction[((face-1)*m+m-1)*4]/(rl*rl),dar=-alpha*direction[(face*m)*4]/(rr*rr);
    const double penalty=pr.o.sipg_penalty*pr.n*pr.n*.5*(dal/hl+dar/hr),jump=base.sigma[(face-1)*m+m-1]-base.sigma[face*m];
    double sl=0.,sr=0.;for(int i=0;i<m;++i){sl+=left_derivative[i]*base.sigma[(face-1)*m+i]/hl;sr+=right_derivative[i]*base.sigma[face*m+i]/hr;}
    const double flux=.5*(dal*sl+dar*sr);
    for(int side=0;side<2;++side)for(int i=0;i<m;++i)
    {
      const double ji=side?(i==0?-1.:0.):(i==m-1?1.:0.);
      const double fi=.5*(side?dar*right_derivative[i]/hr:dal*left_derivative[i]/hl);
      out.coefficient_faces[(side?face:face-1)*m+i]+=-ji*flux-fi*jump+penalty*ji*jump;
    }
  }
  out.coefficient=FrontSum(out.coefficient_volume,out.coefficient_faces);out.sigma_source=FrontSolve(base.matrix,factor,out.load,work);
  State negative=out.coefficient;for(double& x:negative)x=-x;out.sigma_coeff=FrontSolve(base.matrix,factor,negative,work);out.sigma=FrontSum(out.sigma_source,out.sigma_coeff);
  const State tangent_load=FrontDifference(out.load,out.coefficient);out.closure_identity_relative_defect=Norm(FrontDifference(FrontApply(base.matrix,out.sigma),tangent_load))/std::max(Norm(tangent_load),1.e-12);
  out.base_source_relative_defect=FrontRelative(base_load,base.load);double reaction=0.;
  for(int e=0;e<ne;++e)for(std::size_t q=0;q<pr.weights.size();++q)
  {
    double rho=0.,drho=0.,s=0.,ds=0.;for(int i=0;i<m;++i){const double b=pr.low_shapes[q][i];rho+=b*base.moments[(e*m+i)*4];drho+=b*direction[(e*m+i)*4];s+=b*base.sigma[e*m+i];ds+=b*out.sigma[e*m+i];}
    reaction+=(pr.edges[e+1]-pr.edges[e])*pr.weights[q]*(ds/rho-s*drho/(rho*rho));++work.tangent_points;
  }
  out.constant_test_defect=std::abs(reaction-std::accumulate(out.load.begin(),out.load.end(),0.));work.analytic_seconds+=Wall(start);return out;
}
void FrontWriteAnalytic(std::ostream& out,const FrontAnalytic& a)
{
  out<<'{';bool comma=false;auto array=[&](const char* name,const State& x){if(comma)out<<',';comma=true;FrontString(out,name);out<<':';TrackArray(out,x);};
  array("delta_velocity",a.velocity);array("delta_pressure",a.pressure);array("delta_g",a.g);array("delta_b",a.load);array("delta_A_sigma",a.coefficient);
  array("delta_A_sigma_volume",a.coefficient_volume);array("delta_A_sigma_faces",a.coefficient_faces);array("delta_sigma_source",a.sigma_source);array("delta_sigma_coeff",a.sigma_coeff);array("delta_sigma",a.sigma);
  out<<",\"base_source_relative_defect\":"<<a.base_source_relative_defect<<",\"closure_identity_relative_defect\":"<<a.closure_identity_relative_defect<<",\"constant_test_defect\":"<<a.constant_test_defect<<'}';
}
struct FrontEvaluation
{
  FrontState state;std::array<State,4> rhs;std::array<std::array<double,4>,4> budget{};std::array<FrontAdmission,4> admission;
};
FrontEvaluation FrontEvaluate(const FrontProblem& pr,const State& low,const FrontState& base,const BandFactor& base_factor,double time,FrontWork& work)
{
  FrontEvaluation out;out.state=FrontAssemble(pr,low,time,work);const BandFactor factor=FrontFactor(out.state.matrix,work);
  out.state.sigma=FrontSolve(out.state.matrix,factor,out.state.load,work);
  const std::array<State,4> stresses{{out.state.sigma,base.sigma,FrontSolve(base.matrix,base_factor,out.state.load,work),FrontSolve(out.state.matrix,factor,base.load,work)}};
  for(int mode=0;mode<4;++mode)out.rhs[mode]=FrontRHS(pr,out.state,stresses[mode],time,work,out.budget[mode],out.admission[mode]);return out;
}
struct FrontPair
{
  std::array<State,4> jvp,budget;std::array<std::array<FrontAdmission,4>,2> admission;State load,coefficient,sigma;
  double split_absolute=0.,split_relative=0.,closure_fd_relative=0.,load_fd_relative=0.,coefficient_fd_relative=0.,conservation_defect=0.;
};
FrontPair FrontCentral(const FrontProblem& pr,const FrontState& base,const BandFactor& factor,const State& direction,const FrontAnalytic& analytic,double time,double epsilon,FrontWork& work)
{
  const auto plus=FrontEvaluate(pr,FrontSum(base.moments,direction,epsilon),base,factor,time,work);
  const auto minus=FrontEvaluate(pr,FrontSum(base.moments,direction,-epsilon),base,factor,time,work);FrontPair out;out.admission={{plus.admission,minus.admission}};
  for(int mode=0;mode<4;++mode)
  {
    out.jvp[mode]=FrontDifference(plus.rhs[mode],minus.rhs[mode],2.*epsilon);
    out.budget[mode]=FrontDifference(State(plus.budget[mode].begin(),plus.budget[mode].end()),State(minus.budget[mode].begin(),minus.budget[mode].end()),2.*epsilon);
    const auto totals=pr.retained.Integrals(out.jvp[mode],pr.edges);for(int c=0;c<4;++c)out.conservation_defect=std::max(out.conservation_defect,std::abs(totals[c]-out.budget[mode][c]));
  }
  out.load=FrontDifference(plus.state.load,minus.state.load,2.*epsilon);
  out.coefficient=FrontDifference(FrontApply(plus.state.matrix,base.sigma),FrontApply(minus.state.matrix,base.sigma),2.*epsilon);
  out.sigma=FrontDifference(plus.state.sigma,minus.state.sigma,2.*epsilon);
  const State split=FrontDifference(FrontSum(out.jvp[0],out.jvp[1]),FrontSum(out.jvp[2],out.jvp[3]));out.split_absolute=Norm(split);
  out.split_relative=Norm(split)/std::max(Norm(FrontDifference(out.jvp[0],out.jvp[1])),1.e-12);
  out.closure_fd_relative=FrontRelative(out.sigma,analytic.sigma);out.load_fd_relative=FrontRelative(out.load,analytic.load);out.coefficient_fd_relative=FrontRelative(out.coefficient,analytic.coefficient);return out;
}
void FrontWritePair(std::ostream& out,const FrontPair& p)
{
  const char* names[4]={"jvp_full","jvp_fixed","jvp_source","jvp_coeff"};
  for(int mode=0;mode<4;++mode){out<<',';FrontString(out,names[mode]);out<<':';TrackArray(out,p.jvp[mode]);}
  out<<",\"boundary_budget_jvp\":[";for(int mode=0;mode<4;++mode){if(mode)out<<',';TrackArray(out,p.budget[mode]);}out<<']';
  out<<",\"admission_plus_minus\":[";for(int side=0;side<2;++side){if(side)out<<',';out<<'[';for(int mode=0;mode<4;++mode){if(mode)out<<',';FrontWriteAdmission(out,p.admission[side][mode]);}out<<']';}out<<']';
  out<<",\"fd_delta_b\":";TrackArray(out,p.load);out<<",\"fd_delta_A_sigma\":";TrackArray(out,p.coefficient);out<<",\"fd_delta_sigma\":";TrackArray(out,p.sigma);
  out<<",\"rhs_split_absolute\":"<<p.split_absolute<<",\"rhs_split_relative_to_feedback\":"<<p.split_relative
    <<",\"closure_fd_relative_difference\":"<<p.closure_fd_relative<<",\"load_fd_relative_difference\":"<<p.load_fd_relative
    <<",\"coefficient_fd_relative_difference\":"<<p.coefficient_fd_relative<<",\"boundary_conservation_absolute_defect\":"<<p.conservation_defect;
}
std::array<double,3> FrontMinimumGradientSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& b=mo.recovery.boundary;auto& t=b.compatibility.track;
  b.left=b.right="outflow";b.stress="trace";t.initial="entropy";t.mode="uniform";t.left=-1.;t.right=1.;t.center=0.;t.width=.1;t.speed=0.;t.radius=1;
  t.base.elements=8;t.base.order=15;t.base.alpha=.04;t.base.dt=1.e-5;t.base.final_time=.5;t.quad_extra=19;t.closure_space="l2";t.closure_boundary="natural";
  mo.retained_order=mo.stress_order=5;HPPatchOptions patch;patch.inner_elements=4;patch.phase=.3;
  FrontProblem pr(mo,MakePatch(t,patch).edges,.25,.25);pr.EnableMinimumGradient();State low(pr.retained.ns*4);FrontWork work;
  auto value=[](double x){return x*x*x*x/60.+299.*x*x*x/900.-x*x/30.-299.*x/300.;};
  auto derivative=[](double x){return x*x*x/15.+299.*x*x/300.-x/15.-299./300.;};
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.retained.n;++i)
  {
    const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.retained.physics.basis.nodes[i],rho=2.+.2*x;
    const int j=pr.retained.Id(e,i,0);low[j]=rho;low[j+1]=rho*value(x);low[j+2]=0.;low[j+3]=20.;
  }
  const auto base=FrontAssemble(pr,low,.1,work);const auto& recovery=*pr.minimum_gradient;const auto& result=pr.minimum_gradient_result;
  State expected(base.load.size(),0.);double value_error=0.;
  for(int e=0;e<pr.ne;++e)for(int k=0;k<recovery.nq;++k)
  {
    const double h=pr.edges[e+1]-pr.edges[e],z=recovery.values[k][1],x=pr.edges[e]+.5*h*(z+1.);
    double v=0.;for(int j=0;j<recovery.r;++j)v+=recovery.values[k][j]*result.coefficients[e*recovery.r+j];
    value_error=std::max(value_error,std::abs(v-value(x)));
    const double source=2.*t.base.alpha*derivative(x)*derivative(x),W=.5*h*recovery.weights[k];
    for(int i=0;i<pr.retained.n;++i)expected[e*pr.retained.n+i]+=W*recovery.nodal[k][i]*source;
  }
  const double load_error=FrontRelative(base.load,expected);const auto previous=base.load;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.retained.n;++i)low[pr.retained.Id(e,i,1)]*=1.01;
  const auto changed=FrontAssemble(pr,low,.1,work);const double scaling_error=FrontRelative(changed.load,FrontSum(previous,previous,1.0201-1.));
  if(value_error>1.e-9||load_error>1.e-9||scaling_error>1.e-9||result.moment_defect>1.e-9||result.value_jump>1.e-9||result.derivative_jump>1.e-8||result.boundary_gradient>1.e-8)
    throw std::runtime_error("minimum-gradient weighted polynomial/source/each-stage scaling self-test failed");
  return {{value_error,load_error,scaling_error}};
}
std::array<double,5> FrontCommonStateSelfTest(bool pressure_recovery=false,bool retained_jump=false,bool retained_flux=false)
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& b=mo.recovery.boundary;auto& t=b.compatibility.track;
  b.left=b.right="outflow";b.stress="trace";t.initial="entropy";t.mode="uniform";t.left=-1.;t.right=1.;t.center=0.;t.width=.1;t.speed=0.;t.radius=1;
  t.base.elements=8;t.base.order=9;t.base.alpha=.04;t.base.dt=1.e-5;t.base.final_time=.5;t.quad_extra=19;t.closure_space="l2";t.closure_boundary="natural";
  mo.retained_order=mo.stress_order=3;HPPatchOptions patch;patch.inner_elements=4;patch.phase=.3;
  FrontProblem pr(mo,MakePatch(t,patch).edges,0.,0.);pr.common_transport=true;pr.minimum_gradient_pressure=pressure_recovery;pr.retained_common_jump=retained_jump;pr.retained_common_flux=retained_flux;pr.EnableMinimumGradient();const int m=pr.retained.n;State low(pr.retained.ns*4,0.);
  // rho is cubic, v''=rho*(x^3-c), c=(integral rho*x^3)/(integral rho).
  // v'(-1)=v'(1)=0; thus this degree-8 v is the exact full-moment minimizer
  // although only P3 momentum is retained. The common energy has degree19.
  const double c=19./1400.;State second{{-2*c,-.1*c,0.,2.-.05*c,.1,0.,.05}},v(9,0.);v[0]=.1;
  double endpoint=0.;for(int j=0;j<int(second.size());++j){v[j+2]=second[j]/((j+1.)*(j+2.));endpoint+=second[j]*std::pow(-1.,j+1)/(j+1.);}v[1]=-endpoint;
  auto velocity=[&](double x){double value=0.;for(int j=int(v.size())-1;j>=0;--j)value=value*x+v[j];return value;};
  auto rho=[](double x){return 2.+.1*x+.05*x*x*x;};
  auto pressure=[](double x){return 10.+.03*(x*x-1.)*(x*x-1.);};
  const auto& independent=mfem::IntRules.Get(mfem::Geometry::SEGMENT,127);
  for(int e=0;e<pr.ne;++e)
  {
    State load(m,0.),energy(m,0.);const double h=pr.edges[e+1]-pr.edges[e];
    for(int k=0;k<independent.GetNPoints();++k)
    {const auto& q=independent.IntPoint(k);const auto shape=pr.retained.Shape(q.x,false);const double x=pr.edges[e]+h*q.x;for(int i=0;i<m;++i)
      {load[i]+=q.weight*shape[i]*rho(x)*velocity(x);energy[i]+=q.weight*shape[i]*(pressure(x)/(gamma_euler-1.)+.5*rho(x)*velocity(x)*velocity(x));}}
    for(int i=0;i<m;++i)
    {const double x=pr.edges[e]+h*pr.retained.physics.basis.nodes[i];low[(e*m+i)*4]=rho(x);low[(e*m+i)*4+3]=pressure_recovery?0.:20.;for(int j=0;j<m;++j)
      {low[(e*m+i)*4+1]+=pr.retained.mass_inverse(i,j)*load[j];if(pressure_recovery)low[(e*m+i)*4+3]+=pr.retained.mass_inverse(i,j)*energy[j];}}
  }
  FrontWork work;auto base=FrontAssemble(pr,low,.1,work);const auto factor=FrontFactor(base.matrix,work);base.sigma=FrontSolve(base.matrix,factor,base.load,work);
  std::array<double,4> budget{};FrontAdmission admission;const State actual=FrontRHS(pr,base,base.sigma,.1,work,budget,admission);
  State independent_load(low.size(),0.),expected(low.size(),0.);double value_error=0.,moment_error=0.,pressure_error=0.;
  for(int e=0;e<pr.ne;++e)
  {
    State projected(low.size()/pr.ne,0.);const double h=pr.edges[e+1]-pr.edges[e];
    for(int k=0;k<independent.GetNPoints();++k)
    {
      const auto& q=independent.IntPoint(k);const auto shape=pr.retained.Shape(q.x,false),derivative=pr.retained.Shape(q.x,true);State l,d;MinimumGradientRecovery::Legendre(2*q.x-1.,pr.minimum_gradient->q,l,d);
      const auto u=pr.CommonPoint(e,shape,l);const double x=pr.edges[e]+h*q.x;value_error=std::max(value_error,std::abs(u[1]/u[0]-velocity(x)));
      double s=0.;for(int i=0;i<m;++i)s+=shape[i]*base.sigma[e*m+i];const double pvalue=(gamma_euler-1.)*(u[3]-.5*u[1]*u[1]/u[0]);
      if(pressure_recovery)pressure_error=std::max(pressure_error,std::abs(pvalue-pressure(x)));
      const std::array<double,4> flux{{u[1],u[1]*u[1]/u[0]+pvalue+s,0.,(u[3]+pvalue+s)*u[1]/u[0]}};
      for(int i=0;i<m;++i)for(int j=0;j<4;++j)
      {projected[4*i+j]+=q.weight*shape[i]*u[j];independent_load[(e*m+i)*4+j]+=q.weight*derivative[i]*flux[j];}
    }
    for(int i=0;i<m;++i)for(int c=0;c<4;++c)
    {double difference=-low[(e*m+i)*4+c];for(int j=0;j<m;++j)difference+=pr.retained.mass_inverse(i,j)*projected[j*4+c];moment_error=std::max(moment_error,std::abs(difference));}
  }
  // Independently integrate the polynomial volume flux; reuse only the tested
  // shared application LLF/BC face semantics to complete the weak residual.
  const State high_sigma=pr.stress_transfer.Apply(pr.stress_transfer.E,base.sigma,pr.ne,1),adapter=pr.CommonBoundaryAdapter();
  for(int face=0;face<=pr.ne;++face)
  {
    std::array<double,4> flux{};
    if(!face||face==pr.ne)flux=pr.BoundaryFlux(adapter,high_sigma,face==pr.ne);
    else
    {
      auto ul=pr.CommonEndpoint(face-1,1),ur=pr.CommonEndpoint(face,0);
      if(retained_flux)for(int c=0;c<4;++c){ul[c]=low[(face*m-1)*4+c];ur[c]=low[face*m*4+c];}
      double al=0.,ar=0.;const auto fl=pr.physics.Flux(ul.data(),base.sigma[face*m-1],al),fr=pr.physics.Flux(ur.data(),base.sigma[face*m],ar);
      for(int c=0;c<4;++c)
      {const double jump=retained_jump?low[(face*m)*4+c]-low[(face*m-1)*4+c]:ur[c]-ul[c];flux[c]=.5*(fl[c]+fr[c]-pr.physics.FaceSpeed(al,ar)*jump);}
    }
    for(int c=0;c<4;++c){if(face)independent_load[(face*m-1)*4+c]-=flux[c];if(face<pr.ne)independent_load[face*m*4+c]+=flux[c];}
  }
  for(int e=0;e<pr.ne;++e)for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int c=0;c<4;++c)
    expected[(e*m+i)*4+c]+=pr.retained.mass_inverse(i,j)*independent_load[(e*m+j)*4+c]/(pr.edges[e+1]-pr.edges[e]);
  const double rhs_error=FrontRelative(actual,expected);const auto totals=pr.retained.Integrals(actual,pr.edges);double conservation=0.;
  for(int c=0;c<4;++c)conservation=std::max(conservation,std::abs(totals[c]-budget[c]));
  bool transverse_rejected=false;low[2]=1.e-12;try{pr.RecoverMinimumGradient(low);}catch(const std::exception&){transverse_rejected=true;}
  if(value_error>1.e-9||moment_error>1.e-9||rhs_error>1.e-9||conservation>1.e-9||pressure_error>1.e-9||!transverse_rejected)
    throw std::runtime_error("common-state full-degree recovery/moment/flux/conservation/guard self-test failed");
  if(retained_flux)
  {
    State invalid=low;invalid[4*m+3]=-1.;bool rejected=false;
    try{pr.RetainedFaceState(invalid,m,0.);}catch(const std::runtime_error&){rejected=true;}
    if(!rejected)throw std::runtime_error("inadmissible retained Euler face was not rejected");
  }
  return {{value_error,moment_error,rhs_error,conservation,pressure_error}};
}
void FrontSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& b=mo.recovery.boundary;auto& t=b.compatibility.track;
  b.left=b.right="outflow";b.stress="trace";t.initial="entropy";t.mode="uniform";t.left=-1.;t.right=1.;t.center=0.;t.width=.1;t.speed=0.;t.radius=1;
  t.base.elements=8;t.base.order=9;t.base.alpha=.04;t.base.dt=1.e-5;t.base.final_time=.5;t.quad_extra=19;t.closure_space="l2";t.closure_boundary="natural";
  mo.retained_order=mo.stress_order=3;HPPatchOptions patch;patch.inner_elements=4;patch.phase=.3;FrontProblem pr(mo,MakePatch(t,patch).edges,.25,.25);FrontWork work;
  const int m=pr.retained.n;const double slope=.13;State low(pr.retained.ns*4),velocity(low.size(),0.),density(low.size(),0.),pressure(low.size(),0.);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<m;++i)
  {
    const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.retained.physics.basis.nodes[i],v=.2+slope*x;const int j=pr.retained.Id(e,i,0);
    low[j]=2.;low[j+1]=2.*v;low[j+2]=0.;low[j+3]=2.5+v*v;
    velocity[j+1]=2.*x;velocity[j+3]=2.*v*x;density[j]=1.;density[j+1]=v;density[j+3]=.5*v*v;pressure[j+3]=1.;
  }
  auto base=FrontAssemble(pr,low,.37,work);const auto factor=FrontFactor(base.matrix,work);base.sigma=FrontSolve(base.matrix,factor,base.load,work);
  const auto av=FrontDifferentiate(pr,base,factor,velocity,work),ar=FrontDifferentiate(pr,base,factor,density,work),ap=FrontDifferentiate(pr,base,factor,pressure,work);
  double affine=0.;for(double x:av.sigma)affine=std::max(affine,std::abs(x-8.*t.base.alpha*slope));
  const double isolated=std::max({Norm(av.coefficient),Norm(ar.load),Norm(ap.sigma),Norm(ap.velocity),Norm(av.pressure),Norm(ar.pressure)});
  const auto combo=FrontDifferentiate(pr,base,factor,FrontSum(velocity,density,.3),work);const double linearity=FrontRelative(combo.sigma,FrontSum(av.sigma,ar.sigma,.3));
  if(affine>1.e-9||isolated>1.e-9||linearity>1.e-10)throw std::runtime_error("front affine/primitive direction/analytic linearity check failed");
  // Independent full-energy derivative, not assembled from the JVP pieces:
  // rho=2, v=.2+b*x, p=1 gives R=(-2b,-4vb,0,-b(3.5+sigma+3v²)).
  // The retained velocity direction has delta v=x and delta p=delta rho=0.
  const auto affine_pair=FrontCentral(pr,base,factor,velocity,av,.37,1.e-5,work);double affine_rhs_error=0.;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<m;++i)
  {
    const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.retained.physics.basis.nodes[i],v=.2+slope*x,sigma=4.*t.base.alpha*slope*slope;
    const std::array<double,4> exact{{-2.,-4.*(slope*x+v),0.,-(3.5+sigma+3.*v*v)-slope*(8.*t.base.alpha*slope+6.*v*x)}};
    for(int c=0;c<4;++c)affine_rhs_error=std::max(affine_rhs_error,std::abs(affine_pair.jvp[0][pr.retained.Id(e,i,c)]-exact[c]));
  }
  if(affine_rhs_error>5.e-7)throw std::runtime_error("front independent full-energy affine JVP failed");
  for(int e=0;e<pr.ne;++e)for(int i=0;i<m;++i)
  {
    const double z=pr.retained.physics.basis.nodes[i];const int j=pr.retained.Id(e,i,0);
    low[j]=1.2+.02*z+.003*e;low[j+1]=.3+.07*z*z*z+.01*e;low[j+2]=.02*z*z;low[j+3]=3.+.05*z*z;
    density[j]=.1+.02*z+.003*e;density[j+1]=.2*z*z-.01*e;density[j+2]=.03*z;density[j+3]=.07*z*z;
  }
  base=FrontAssemble(pr,low,.37,work);const auto varied_factor=FrontFactor(base.matrix,work);base.sigma=FrontSolve(base.matrix,varied_factor,base.load,work);
  const auto analytic=FrontDifferentiate(pr,base,varied_factor,density,work);
  const auto coarse=FrontCentral(pr,base,varied_factor,density,analytic,.37,.02,work),fine=FrontCentral(pr,base,varied_factor,density,analytic,.37,.01,work);
  if(fine.closure_fd_relative>2.e-4||fine.closure_fd_relative>.4*coarse.closure_fd_relative+1.e-9||fine.split_relative>2.e-3||fine.conservation_defect>1.e-8||analytic.base_source_relative_defect>1.e-10||analytic.constant_test_defect>1.e-9)
    throw std::runtime_error("front nonlinear epsilon/closure/split/conservation check failed");
  std::array<double,4> original_budget{},corrected_budget{};FrontAdmission original_admission,corrected_admission;
  const State original_rhs=FrontRHS(pr,base,base.sigma,.37,work,original_budget,original_admission);
  State offset(low.size(),0.);
  // Opposite physical cell-average momentum flux contributions across one
  // internal face: unequal widths must still give exactly zero global load.
  for(int i=0;i<m;++i){offset[pr.retained.Id(2,i,1)]=.003/(pr.edges[3]-pr.edges[2]);offset[pr.retained.Id(3,i,1)]=-.003/(pr.edges[4]-pr.edges[3]);}
  pr.SetOffset(offset);const State corrected_rhs=FrontRHS(pr,base,base.sigma,.37,work,corrected_budget,corrected_admission);
  const double offset_error=FrontRelative(FrontDifference(corrected_rhs,original_rhs),offset);
  const auto corrected_totals=pr.retained.Integrals(corrected_rhs,pr.edges);double offset_budget_error=0.;
  for(int c=0;c<4;++c)offset_budget_error=std::max({offset_budget_error,std::abs(corrected_totals[c]-original_budget[c]),std::abs(corrected_budget[c]-original_budget[c])});
  bool bad_offset_rejected=false;State bad_offset=offset;for(std::size_t i=0;i<bad_offset.size();i+=4)bad_offset[i]=1.;
  try{pr.SetOffset(bad_offset);}catch(const std::runtime_error&){bad_offset_rejected=true;}
  if(offset_error>1.e-10||offset_budget_error>1.e-9||!bad_offset_rejected)throw std::runtime_error("front offset conservation/rejection check failed");
  // Independent constant-density manufactured response: a constant extra
  // source ds gives constant extra stress rho*ds on the natural SIPG domain.
  State uniform=low;for(int e=0;e<pr.ne;++e)for(int i=0;i<m;++i)
  {const int j=pr.retained.Id(e,i,0);uniform[j]=2.;uniform[j+1]=.4;uniform[j+2]=0.;uniform[j+3]=3.;}
  State source_values(pr.stress.ns,.037);pr.SetSourceOffset(source_values);
  auto shifted=FrontAssemble(pr,uniform,.37,work);const auto sf=FrontFactor(shifted.matrix,work);
  const auto ss=FrontSolve(shifted.matrix,sf,shifted.load,work);double source_error=0.;
  for(double x:ss)source_error=std::max(source_error,std::abs(x-.074));
  const double source_mean_error=std::abs(shifted.source_integral-.037*(pr.edges.back()-pr.edges.front()));
  const auto shared_source_sigma=pr.stress_transfer.Apply(pr.stress_transfer.P,pr.Closure(shifted.high,.37,work.shared),pr.ne,1);
  const double source_dispatch=FrontRelative(shared_source_sigma,ss);
  if(source_error>1.e-10||source_mean_error>1.e-11||source_dispatch>1.e-10)
    throw std::runtime_error("front source offset manufactured/dispatch check failed");
  // Replacement must ignore a nonzero current raw source while retaining
  // the current density operator and the prescribed positive load.
  pr.replace_source=true;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<m;++i)
  {const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.retained.physics.basis.nodes[i];uniform[pr.retained.Id(e,i,1)]=.4+.1*x;}
  auto replaced=FrontAssemble(pr,uniform,.37,work);const auto rf=FrontFactor(replaced.matrix,work);
  const auto rs=FrontSolve(replaced.matrix,rf,replaced.load,work);double replacement_error=0.;
  for(double x:rs)replacement_error=std::max(replacement_error,std::abs(x-.074));
  if(replacement_error>1.e-10||std::abs(replaced.source_integral-.037*(pr.edges.back()-pr.edges.front()))>1.e-11)
    throw std::runtime_error("front source replacement manufactured check failed");
  pr.rhs_offset.clear();pr.source_offset_load.clear();pr.replace_source=false;
  for(int e=0;e<pr.ne;++e)for(int i=0;i<m;++i)
  {
    const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.retained.physics.basis.nodes[i];
    const double v=.2+.1*x;const int j=pr.retained.Id(e,i,0);
    uniform[j]=2.;uniform[j+1]=2.*v;uniform[j+2]=0.;uniform[j+3]=2.5+v*v;
  }
  auto transport_state=FrontAssemble(pr,uniform,.37,work);
  const auto tf=FrontFactor(transport_state.matrix,work);
  transport_state.sigma=FrontSolve(transport_state.matrix,tf,transport_state.load,work);
  pr.retained_transport=true;std::array<double,4> transport_budget{};FrontAdmission transport_admission;
  const auto transport_rhs=FrontRHS(pr,transport_state,transport_state.sigma,.37,work,transport_budget,transport_admission);
  double transport_error=0.;
  for(int e=1;e+1<pr.ne;++e)for(int i=0;i<m;++i)
  {
    const double x=pr.edges[e]+(pr.edges[e+1]-pr.edges[e])*pr.retained.physics.basis.nodes[i];
    const double v=.2+.1*x,s=4.*t.base.alpha*.01;
    const std::array<double,4> expected{{-.2,-.4*v,0.,-.1*(3.5+s+3.*v*v)}};
    for(int c=0;c<4;++c)transport_error=std::max(transport_error,std::abs(transport_rhs[pr.retained.Id(e,i,c)]-expected[c]));
  }
  if(transport_error>1.e-8)throw std::runtime_error("retained-transport independent affine solution check failed");
  for(const std::string form:{"weak-lumped","dgsem-central","dgsem-ranocha"})
  {
    pr.transport_form=form;
    const State r=FrontRHS(pr,transport_state,transport_state.sigma,.37,work,transport_budget,transport_admission);
    double error=0.;for(int e=1;e+1<pr.ne;++e)for(int i=0;i<m;++i)for(int c=0;c<4;++c)
      error=std::max(error,std::abs(r[pr.retained.Id(e,i,c)]-transport_rhs[pr.retained.Id(e,i,c)]));
    if(error>1.e-8)throw std::runtime_error("alternative transport independent affine solution failed");
  }
  pr.transport_form="weak-exact";
  // Degree one has no curvature term. It must still admit the ordinary
  // closure/transport path with that term disabled, and reject a nonzero
  // coefficient whose trace normalization would vanish.
  {
    MemoryOptions linear=mo;linear.retained_order=linear.stress_order=1;
    linear.recovery.boundary.compatibility.track.base.order=3;
    FrontProblem lp(linear,pr.edges,.25,0.);FrontWork lw;
    State constant(lp.retained.ns*4,0.);
    for(int j=0;j<lp.retained.ns;++j){constant[4*j]=1.;constant[4*j+1]=.2;constant[4*j+3]=2.52;}
    auto ls=FrontAssemble(lp,constant,.37,lw);const auto lf=FrontFactor(ls.matrix,lw);
    ls.sigma=FrontSolve(ls.matrix,lf,ls.load,lw);std::array<double,4> lb{};FrontAdmission la;
    const auto lr=FrontRHS(lp,ls,ls.sigma,.37,lw,lb,la);
    if(Norm(lr)>1.e-9||Norm(ls.sigma)>1.e-10||lp.curvature_trace_norm!=0.)throw std::runtime_error("degree-one zero-curvature uniform state failed");
    bool rejected=false;try{FrontProblem invalid(linear,pr.edges,.25,.25);}catch(const std::runtime_error&){rejected=true;}
    if(!rejected)throw std::runtime_error("degree-one nonzero-curvature guard failed");
    for(const std::string form:{"weak-lumped","dgsem-central","dgsem-ranocha"})
    {
      lp.transport_form=form;const State r=FrontRHS(lp,ls,ls.sigma,.37,lw,lb,la);
      if(Norm(r)>1.e-9)throw std::runtime_error("alternative transport constant state failed");
    }
  }
  {
    const std::array<double,4> u{{1.,.2,.03,2.52045}};double speed=0.;
    const auto physical=pr.physics.Flux(u.data(),.4,speed),pair=FrontThermalRanocha(u,u,.4,.4);
    for(int c=0;c<4;++c)if(std::abs(physical[c]-pair[c])>1.e-12)throw std::runtime_error("full-energy IGR split diagonal failed");
  }
  {
    State polynomial(pr.retained.ns*4),expected(polynomial.size());
    const double factor=.37;const int mask=9;
    for(int e=0;e<pr.ne;++e)for(int i=0;i<m;++i)
    {
      State v,d;MinimumGradientRecovery::Legendre(2.*pr.retained.physics.basis.nodes[i]-1.,pr.retained.p,v,d);
      for(int c=0;c<4;++c)
      {const int j=(e*m+i)*4+c;polynomial[j]=1.+c+.2*v[0]+.3*v.back();expected[j]=(mask&(1<<c))?(factor-1.)*.3*v.back():0.;}
    }
    const State high=pr.state_transfer.Apply(pr.state_transfer.E,polynomial,pr.ne,4);
    const State delta=FrontTopModeDelta(pr,high,factor,mask);
    const State low=pr.state_transfer.Apply(pr.state_transfer.P,delta,pr.ne,4);
    if(FrontRelative(low,expected)>1.e-10)throw std::runtime_error("rank-one top-mode independent polynomial/mask check failed");
    for(double x:pr.Integrals(delta,pr.edges))if(std::abs(x)>1.e-11)throw std::runtime_error("rank-one top-mode changes cell-mean budget");
  }
  const auto variational=FrontMinimumGradientSelfTest();
  {
    FrontBulkGradient bulk(pr,pr.low_derivatives,.1),consistent(pr,pr.low_derivatives,.1,true);State physical(pr.ns*4,0.),sigma(pr.ns,0.);
    for(int degree:{1,3})
    {
      for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.n;++i)
      {
        const double z=pr.physics.basis.nodes[i],rho=1.+.2*z,v=.2+.03*z+(degree==3?.02*z*z*z:0.);
        const int j=pr.Id(e,i,0);physical[j]=rho;physical[j+1]=rho*v;physical[j+3]=3.+.5*rho*v*v;
      }
      State result(physical.size(),0.);TrackWork work;bulk.Add(pr,physical,sigma,result,work);
      if(degree==1&&Norm(result)>1.e-9)throw std::runtime_error("bulk-gradient affine velocity nullspace failed");
      if(degree==3&&(Norm(result)<1.e-9||bulk.maximum_dissipation<=0.))throw std::runtime_error("bulk-gradient nonzero high-gradient control failed");
      for(std::size_t i=0;i<result.size();++i)if((i%4==0||i%4==3)&&result[i]!=0.)throw std::runtime_error("bulk-gradient changes density/energy equation");
      if(bulk.maximum_mean_defect>1.e-10||bulk.maximum_work_defect>1.e-10)throw std::runtime_error("bulk-gradient conservative kinetic work identity failed");
      State top_result(physical.size(),0.);consistent.Add(pr,physical,sigma,top_result,work);
      if(degree==1&&Norm(top_result)>1.e-9)throw std::runtime_error("top-strain lower polynomial nullspace failed");
      if(degree==3&&(Norm(top_result)<1.e-12||consistent.maximum_activity<=0.||consistent.maximum_activity>=1.))
        throw std::runtime_error("top-strain nonzero bounded activation failed");
      if(consistent.maximum_mean_defect>1.e-10||consistent.maximum_work_defect>1.e-10)throw std::runtime_error("top-strain isolated work identity failed");
    }
  }
  const auto common_state=FrontCommonStateSelfTest();
  const auto common_pressure=FrontCommonStateSelfTest(true);
  const auto common_jump=FrontCommonStateSelfTest(true,true);
  const auto common_flux=FrontCommonStateSelfTest(true,false,true);
  double constitutive_parity=0.,constitutive_affine=0.,constitutive_contact=0.,constitutive_nonlinear_residual=0.;
  {
    FrontProblem source_pr(mo,pr.edges,0.,0.);const int mm=source_pr.retained.n;
    for(int example=0;example<4;++example)
    {
      State low(source_pr.retained.ns*4,0.);
      const double g=example==0?-.13:example==1?.13:0.;
      for(int e=0;e<source_pr.ne;++e)for(int i=0;i<mm;++i)
      {
        const double x=source_pr.edges[e]+(source_pr.edges[e+1]-source_pr.edges[e])*source_pr.retained.physics.basis.nodes[i];
        const double rho=example==2?2.+.03*x:2.,v=.2+g*x,pth=example==3?1.+.05*x:1.;
        const int j=source_pr.retained.Id(e,i,0);low[j]=rho;low[j+1]=rho*v;low[j+3]=pth/(gamma_euler-1.)+.5*rho*v*v;
      }
      const State moments=source_pr.state_transfer.Apply(source_pr.state_transfer.E,low,source_pr.ne,4);
      const State physical=source_pr.Reconstruct(moments);TrackWork w;BandMatrix a;State original;
      source_pr.MixedSystem(physical,0.,w,a,original);NativePressureSource source;source.law="constant";source.kappa=2.;State b;
      source.Update(source_pr,physical,w,a,b);constitutive_parity=std::max(constitutive_parity,Norm(FrontDifference(b,original)));
      for(const std::string law:{"constant-compression","pressure-compression"})
      {
        source.law=law;source.kappa=6.;source.Update(source_pr,physical,w,a,b);
        BandFactor f(a,0,a.n);State sigma=b;f.Solve(sigma,1);
        if(example<2)for(double s:sigma)
          constitutive_affine=std::max(constitutive_affine,std::abs(s-(law=="pressure-compression"?3.*gamma_euler+1.:6.)*t.base.alpha*2.*std::pow(std::min(g,0.),2)));
        if(example==2)constitutive_contact=std::max(constitutive_contact,Norm(sigma));
        if(example==3&&law=="pressure-compression")
        {if(Norm(sigma)<1.e-6||source.solves==0)throw std::runtime_error("pressure-only nonlinear source test was inactive");constitutive_nonlinear_residual=source.maximum_residual;}
      }
    }
    if(constitutive_parity>1.e-10||constitutive_affine>1.e-10||constitutive_contact>1.e-12||constitutive_nonlinear_residual>1.e-10)
      throw std::runtime_error("native constitutive source parity/affine/contact/nonlinear checks failed");
  }
  std::cout<<std::setprecision(17)<<"{\"affine_closure_error\":"<<affine<<",\"independent_full_energy_affine_jvp_error\":"<<affine_rhs_error<<",\"primitive_direction_isolation\":"<<isolated<<",\"analytic_linearity\":"<<linearity
    <<",\"coarse_closure_fd_relative\":"<<coarse.closure_fd_relative<<",\"fine_closure_fd_relative\":"<<fine.closure_fd_relative<<",\"rhs_split_relative\":"<<fine.split_relative
    <<",\"nonuniform_conservation_defect\":"<<fine.conservation_defect<<",\"offset_relative_error\":"<<offset_error<<",\"offset_boundary_budget_error\":"<<offset_budget_error
    <<",\"source_offset_manufactured_error\":"<<source_error<<",\"source_offset_constant_test_error\":"<<source_mean_error<<",\"source_offset_shared_dispatch_error\":"<<source_dispatch<<",\"source_replacement_manufactured_error\":"<<replacement_error<<",\"retained_transport_affine_error\":"<<transport_error<<",\"minimum_gradient_weighted_polynomial_error\":"<<variational[0]<<",\"minimum_gradient_source_load_error\":"<<variational[1]<<",\"minimum_gradient_each_stage_scaling_error\":"<<variational[2]
    <<",\"common_state_degree_eight_velocity_error\":"<<common_state[0]<<",\"common_state_degree_nineteen_moment_error\":"<<common_state[1]<<",\"common_state_independent_weak_flux_error\":"<<common_state[2]<<",\"common_state_conservation_error\":"<<common_state[3]
    <<",\"common_recovered_pressure_moment_error\":"<<common_pressure[1]<<",\"common_recovered_pressure_independent_weak_flux_error\":"<<common_pressure[2]<<",\"common_recovered_pressure_quartic_error\":"<<common_pressure[4]
    <<",\"common_retained_flux_independent_flux_error\":"<<common_flux[2]<<",\"common_retained_flux_conservation_error\":"<<common_flux[3]
    <<",\"common_retained_jump_independent_flux_error\":"<<common_jump[2]<<",\"common_retained_jump_conservation_error\":"<<common_jump[3]<<",\"maximum_linear_solve_residual\":"<<work.maximum_solve_residual
    <<",\"constitutive_original_source_parity\":"<<constitutive_parity<<",\"constitutive_affine_error\":"<<constitutive_affine
    <<",\"constitutive_contact_stress\":"<<constitutive_contact<<",\"constitutive_nonlinear_residual\":"<<constitutive_nonlinear_residual<<"}\n";
}
}

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    MemoryOptions mo;std::string input,edge_input,direction_input,offset_input,source_input,replacement_input,transport_state="primitive",transport_form="weak-exact",source_velocity="primitive",common_pressure="polynomial",common_jump="physical",common_flux="physical",epsilon_text="0.0001,0.00003,0.00001";int direction_count=0,top_mask=15;double time=0.,slope=.5,curvature=0.,top_transport=1.,top_rhs=1.,top_central=1.,top_llf=1.,bulk_strength=0.;bool advance=false,rhs_only=false;
    std::string bulk_policy="upper-half",igr_source_law="original",igr_pressure_faces="central",cap_kind="none";double cap_identity=12.,cap_bound=20.;
    bool obstacle=false;double obstacle_tau=.5,obstacle_bound=6.,obstacle_margin=.001,obstacle_tolerance=2.e-9;
    bool workcap=false;double workcap_identity=4.,workcap_bound=6.,workcap_tolerance=1.e-9,workcap_cfl=1.;
    double llf_multiplier=1.,igr_source_kappa=6.,igr_source_tolerance=1.e-11;bool input_boundary_study=false,mirror_wall_source=false;std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--advance-input"){advance=true;continue;}if(flag=="--rhs-only"){rhs_only=true;continue;}
      if(flag=="--input-boundary-study"){input_boundary_study=true;continue;}
      if(flag=="--mirror-wall-source"){mirror_wall_source=true;continue;}
      if(flag=="--obstacle-stress"){obstacle=true;continue;}
      if(flag=="--obstacle-tau"||flag=="--obstacle-bound"||flag=="--obstacle-margin"||flag=="--obstacle-tolerance")
      {if(++i==argc)throw std::runtime_error("missing obstacle option");const double v=std::stod(argv[i]);if(flag=="--obstacle-tau")obstacle_tau=v;else if(flag=="--obstacle-bound")obstacle_bound=v;else if(flag=="--obstacle-margin")obstacle_margin=v;else obstacle_tolerance=v;continue;}
      if(flag=="--work-stress-cap"){workcap=true;continue;}
      if(flag=="--workcap-identity"||flag=="--workcap-bound"||flag=="--workcap-tolerance"||flag=="--workcap-diffusion-cfl-limit")
      {if(++i==argc)throw std::runtime_error("missing work-cap option");const double value=std::stod(argv[i]);if(flag=="--workcap-identity")workcap_identity=value;else if(flag=="--workcap-bound")workcap_bound=value;else if(flag=="--workcap-tolerance")workcap_tolerance=value;else workcap_cfl=value;continue;}
      if(flag=="--physical-stress-cap"||flag=="--cap-identity"||flag=="--cap-bound")
      {if(++i==argc)throw std::runtime_error("missing physical-cap option");if(flag=="--physical-stress-cap")cap_kind=argv[i];else if(flag=="--cap-identity")cap_identity=std::stod(argv[i]);else cap_bound=std::stod(argv[i]);continue;}
      if(flag=="--igr-source-law"||flag=="--igr-pressure-faces"||flag=="--igr-source-kappa"||flag=="--igr-source-tolerance")
      {
        if(++i==argc)throw std::runtime_error("missing research constitutive-source option");
        if(flag=="--igr-source-law")igr_source_law=argv[i];else if(flag=="--igr-pressure-faces")igr_pressure_faces=argv[i];
        else if(flag=="--igr-source-kappa")igr_source_kappa=std::stod(argv[i]);else igr_source_tolerance=std::stod(argv[i]);continue;
      }
      if(flag=="--transport-form"){if(++i==argc)throw std::runtime_error("missing transport form");transport_form=argv[i];continue;}
      if(flag=="--bulk-gradient-strength"){if(++i==argc)throw std::runtime_error("missing bulk gradient strength");bulk_strength=std::stod(argv[i]);continue;}
      if(flag=="--bulk-gradient-policy"){if(++i==argc)throw std::runtime_error("missing bulk gradient policy");bulk_policy=argv[i];continue;}
      if(flag=="--interior-llf-multiplier"){if(++i==argc)throw std::runtime_error("missing interior LLF multiplier");llf_multiplier=std::stod(argv[i]);continue;}
      if(flag=="--top-central-factor"||flag=="--top-llf-factor")
      {if(++i==argc)throw std::runtime_error("missing central/LLF factor");(flag=="--top-central-factor"?top_central:top_llf)=std::stod(argv[i]);continue;}
      if(flag=="--top-transport-factor"||flag=="--top-rhs-factor"||flag=="--top-component-mask")
      {
        if(++i==argc)throw std::runtime_error("missing top-mode factor/mask");
        if(flag=="--top-component-mask")top_mask=std::stoi(argv[i]);
        else if(flag=="--top-transport-factor")top_transport=std::stod(argv[i]);else top_rhs=std::stod(argv[i]);
        continue;
      }
      if(flag=="--moments-in"||flag=="--edges-in"||flag=="--directions-in"||flag=="--rhs-offset-in"||flag=="--source-offset-in"||flag=="--source-replacement-in"||flag=="--transport-state"||flag=="--source-velocity"||flag=="--common-pressure"||flag=="--common-face-jump"||flag=="--common-face-flux"||flag=="--direction-count"||flag=="--probe-time"||flag=="--epsilons"||flag=="--slope-strength"||flag=="--curvature-strength")
      {
        if(++i==argc)throw std::runtime_error("missing front probe option");const std::string value=argv[i];
        if(flag=="--moments-in")input=value;else if(flag=="--edges-in")edge_input=value;else if(flag=="--directions-in")direction_input=value;else if(flag=="--rhs-offset-in")offset_input=value;else if(flag=="--source-offset-in")source_input=value;else if(flag=="--source-replacement-in")replacement_input=value;
        else if(flag=="--direction-count")direction_count=std::stoi(value);else if(flag=="--probe-time")time=std::stod(value);else if(flag=="--epsilons")epsilon_text=value;
        else if(flag=="--transport-state")transport_state=value;else if(flag=="--source-velocity")source_velocity=value;else if(flag=="--common-pressure")common_pressure=value;else if(flag=="--common-face-jump")common_jump=value;else if(flag=="--common-face-flux")common_flux=value;else if(flag=="--slope-strength")slope=std::stod(value);else curvature=std::stod(value);
      }
      else common.push_back(argv[i]);
    }
    mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;
    if(t.self_test){FrontSelfTest();return 0;}
    if(obstacle&&(!std::isfinite(obstacle_tau)||!std::isfinite(obstacle_bound)||!std::isfinite(obstacle_margin)||!std::isfinite(obstacle_tolerance)||
       obstacle_tau<=0.||obstacle_bound<=0.||obstacle_margin<=0.||obstacle_margin>=.01||obstacle_tolerance<=0.||obstacle_tolerance>1.e-7||
       workcap||cap_kind!="none"||!input_boundary_study||(!advance&&!rhs_only)||transport_state!="primitive"||transport_form!="weak-exact"||source_velocity!="primitive"||
       slope!=0.||curvature!=0.||bulk_strength!=0.||top_transport!=1.||top_rhs!=1.||top_central!=1.||top_llf!=1.||llf_multiplier!=1.||
       !offset_input.empty()||!source_input.empty()||!replacement_input.empty()||mo.recovery.boundary.wall_flux!="mirror"||
       mo.recovery.boundary.left!="wall"||mo.recovery.boundary.right!="wall"||igr_source_law!="pressure-compression"||igr_pressure_faces!="central"||t.base.alpha<=0.))
      throw std::runtime_error("obstacle requires positive-parameter current pressure-source primitive weak-DG mirrored walls, no other cap/penalty/offset");
    if(workcap&&(!std::isfinite(workcap_identity)||!std::isfinite(workcap_bound)||!std::isfinite(workcap_tolerance)||!std::isfinite(workcap_cfl)||
       workcap_identity<=0.||workcap_bound<=workcap_identity||workcap_tolerance<=0.||workcap_tolerance>1.e-6||workcap_cfl<=0.||
       cap_kind!="none"||!input_boundary_study||(!advance&&!rhs_only)||transport_state!="primitive"||transport_form!="weak-exact"||source_velocity!="primitive"||
       slope!=0.||curvature!=0.||bulk_strength!=0.||top_transport!=1.||top_rhs!=1.||top_central!=1.||top_llf!=1.||llf_multiplier!=1.||
       !offset_input.empty()||!source_input.empty()||!replacement_input.empty()||mo.recovery.boundary.wall_flux!="mirror"||
       mo.recovery.boundary.left!="wall"||mo.recovery.boundary.right!="wall"||igr_source_law!="pressure-compression"||t.base.alpha<=0.))
      throw std::runtime_error("work cap requires plain positive-alpha pressure-source primitive weak-DG wall RHS/advance, no other cap/penalty/offset, and valid parameters");
    if(cap_kind!="none"&&cap_kind!="identity-smooth")throw std::runtime_error("invalid physical stress cap");
    if(cap_kind!="none"&&(!std::isfinite(cap_identity)||!std::isfinite(cap_bound)||cap_identity<0.||cap_bound<=cap_identity||
       !input_boundary_study||(!advance&&!rhs_only)||transport_state!="primitive"||transport_form!="weak-exact"||source_velocity!="primitive"||
       slope!=0.||curvature!=0.||bulk_strength!=0.||top_transport!=1.||top_rhs!=1.||top_central!=1.||top_llf!=1.||llf_multiplier!=1.||
       !offset_input.empty()||!source_input.empty()||!replacement_input.empty()||mo.recovery.boundary.wall_flux!="mirror"))
      throw std::runtime_error("physical cap requires plain primitive wall/outflow weak-DG RHS/advance, mirrored Euler flux, no penalties/offsets, and valid identity<bound");
    if(igr_source_law!="original"&&igr_source_law!="constant"&&igr_source_law!="constant-compression"&&igr_source_law!="pressure-compression")
      throw std::runtime_error("invalid research IGR source law");
    if(igr_pressure_faces!="central"&&igr_pressure_faces!="broken")throw std::runtime_error("invalid pressure-source face derivative");
    if(!std::isfinite(igr_source_kappa)||igr_source_kappa<0.||!std::isfinite(igr_source_tolerance)||igr_source_tolerance<=0.||igr_source_tolerance>1.e-8)
      throw std::runtime_error("invalid source coefficient/tolerance");
    if(igr_source_law!="original"&&(!advance&&!rhs_only||transport_state!="primitive"||transport_form!="weak-exact"||source_velocity!="primitive"||
       !offset_input.empty()||!source_input.empty()||!replacement_input.empty()||top_transport!=1.||top_rhs!=1.||top_central!=1.||top_llf!=1.||llf_multiplier!=1.||bulk_strength!=0.))
      throw std::runtime_error("new constitutive source requires plain primitive weak-DG rhs/advance with current source and unmodified LLF/mass");
    if(!std::isfinite(bulk_strength)||bulk_strength<0.||bulk_strength>0.&&(!advance&&!rhs_only||transport_state=="minimum-gradient"))
      throw std::runtime_error("bulk gradient study needs nonnegative strength and primitive/retained operator or advance");
    if(bulk_policy!="upper-half"&&bulk_policy!="top-strain")throw std::runtime_error("invalid bulk gradient policy");
    if(!std::isfinite(top_transport)||!std::isfinite(top_rhs)||top_transport<=0.||top_rhs<=0.||top_mask<1||top_mask>15)
      throw std::runtime_error("top-mode factors must be finite positive; component mask in1..15");
    if((top_transport!=1.||top_rhs!=1.)&&(!advance&&!rhs_only||transport_state=="minimum-gradient"||transport_form!="weak-exact"||!offset_input.empty()))
      throw std::runtime_error("top-mode study needs plain weak-exact primitive/retained operator or advance, without frozen RHS correction");
    if(!std::isfinite(top_central)||!std::isfinite(top_llf)||top_central<=0.||top_llf<=0.)
      throw std::runtime_error("central/LLF factors must be finite positive");
    if((top_central!=1.||top_llf!=1.)&&(!advance&&!rhs_only||transport_state=="minimum-gradient"||transport_form!="weak-exact"||!offset_input.empty()||top_transport!=1.||top_rhs!=1.||top_mask!=15))
      throw std::runtime_error("central/LLF study needs weak-exact primitive/retained all-component operator or advance, without other mass factors or frozen correction");
    if(!std::isfinite(llf_multiplier)||llf_multiplier<=0.)throw std::runtime_error("interior LLF multiplier must be finite positive");
    if(llf_multiplier!=1.&&(!advance&&!rhs_only||transport_state=="minimum-gradient"||transport_form!="weak-exact"||!offset_input.empty()||top_transport!=1.||top_rhs!=1.||top_central!=1.||top_llf!=1.||top_mask!=15))
      throw std::runtime_error("whole LLF discriminator requires plain weak-exact all-component primitive/retained operator or advance");
    if(transport_form!="weak-exact"&&transport_form!="weak-lumped"&&transport_form!="dgsem-central"&&transport_form!="dgsem-ranocha")throw std::runtime_error("invalid transport form");
    if(transport_form!="weak-exact"&&(!advance&&!rhs_only||transport_state=="minimum-gradient"))throw std::runtime_error("alternative transport needs an operator/advance and primitive or retained state");
    if(input.empty()||edge_input.empty()||!std::isfinite(time)||time<0.||time>t.base.final_time)throw std::runtime_error("saved retained moments/mesh and valid physical time required");
    if(input_boundary_study)
    {
      const auto& bc=mo.recovery.boundary;
      const auto permitted=[](const std::string& kind){return kind=="wall"||kind=="outflow";};
      if((!advance&&!rhs_only)||bc.scenario!="existing"||!permitted(bc.left)||!permitted(bc.right)||
         bc.stress!="trace"||bc.open_closure!="natural"||bc.wall_closure!="natural"||
         transport_state!="primitive"||transport_form!="weak-exact"||source_velocity!="primitive"||
         !offset_input.empty()||!source_input.empty()||!replacement_input.empty()||(advance&&time!=0.))
        throw std::runtime_error("boundary input study requires a single time-zero autonomous primitive weak-DG advance or RHS, natural stress, copied wall/outflow stress, and no supplied offsets");
      if((bc.left=="wall"||bc.right=="wall")&&!mirror_wall_source)
        throw std::runtime_error("boundary input study requires explicit reflected wall source traces");
    }
    else if(mo.recovery.boundary.scenario!="application-tanh"||mo.recovery.boundary.left!="nonreflecting"||mo.recovery.boundary.right!="fixed")
      throw std::runtime_error("front input is qualified only for autonomous application-tanh inlet/outlet data");
    if(mirror_wall_source&&!input_boundary_study)throw std::runtime_error("reflected source requires explicit boundary input study");
    if((advance||rhs_only)&&(direction_count||!direction_input.empty()))throw std::runtime_error("advancement/operator evaluation and directional diagnosis are separate modes");
    if(advance&&rhs_only)throw std::runtime_error("choose advancement or operator evaluation");
    if(!advance&&!rhs_only&&(direction_count<1||direction_count>64||direction_input.empty()))throw std::runtime_error("one to64 explicit retained directions required");
    if((!offset_input.empty()||!source_input.empty()||!replacement_input.empty())&&!advance&&!rhs_only)throw std::runtime_error("frozen source/RHS inputs are not a qualified directional-derivative mode");
    if(!source_input.empty()&&!replacement_input.empty())throw std::runtime_error("choose additive source or source replacement");
    if(source_velocity!="primitive"&&source_velocity!="minimum-gradient")throw std::runtime_error("invalid source velocity policy");
    if(source_velocity=="minimum-gradient"&&(!source_input.empty()||!replacement_input.empty()))throw std::runtime_error("minimum-gradient source cannot be combined with supplied source offsets or replacements");
    if(source_velocity=="minimum-gradient"&&!advance&&!rhs_only)throw std::runtime_error("minimum-gradient source has no qualified analytic tangent mode");
    if(transport_state!="primitive"&&transport_state!="retained"&&transport_state!="minimum-gradient")throw std::runtime_error("invalid transport state");
    if(transport_state=="minimum-gradient"&&(source_velocity!="minimum-gradient"||slope!=0.||curvature!=0.))
      throw std::runtime_error("minimum-gradient common transport requires the same source velocity and zero slope/curvature penalties");
    if(common_pressure!="polynomial"&&common_pressure!="minimum-gradient")throw std::runtime_error("invalid common physical-pressure recovery");
    if(common_pressure=="minimum-gradient"&&transport_state!="minimum-gradient")throw std::runtime_error("physical pressure recovery requires minimum-gradient common transport");
    if(common_jump!="physical"&&common_jump!="retained")throw std::runtime_error("invalid common dissipative face jump");
    if(common_jump=="retained"&&transport_state!="minimum-gradient")throw std::runtime_error("retained common jump requires minimum-gradient common transport");
    if(common_flux!="physical"&&common_flux!="retained")throw std::runtime_error("invalid common interior numerical flux");
    if(common_flux=="retained"&&(transport_state!="minimum-gradient"||common_jump!="physical"))
      throw std::runtime_error("retained full interior flux requires minimum-gradient common transport and cannot combine with separate jump override");
    if(transport_state!="primitive"&&!advance&&!rhs_only)throw std::runtime_error("retained transport has no qualified analytic tangent mode");
    State epsilons;std::istringstream scales(epsilon_text);std::string piece;
    while(std::getline(scales,piece,',')){std::size_t consumed=0;const double value=std::stod(piece,&consumed);if(consumed!=piece.size()||!std::isfinite(value)||!(value>0.))throw std::runtime_error("invalid finite-difference epsilon");epsilons.push_back(value);}
    if(epsilons.empty()||epsilons.size()>16)throw std::runtime_error("one to16 epsilon values required");
    const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;mo.input=input;if(advance)mo.start_time=time;
    const auto start=Clock::now();const State edges=FrontRead(edge_input,t.base.elements+1);FrontProblem pr(mo,edges,slope,curvature,input_boundary_study);const double setup=Wall(start);
    pr.retained_transport=transport_state=="retained";
    pr.input_boundary_study=input_boundary_study;
    if(cap_kind!="none"){pr.physical_cap=std::make_unique<NativePhysicalCap>();pr.physical_cap->identity=cap_identity;pr.physical_cap->bound=cap_bound;}
    if(obstacle){pr.obstacle_stress=std::make_unique<NativeObstacleStress>();pr.obstacle_stress->tau=obstacle_tau;pr.obstacle_stress->bound=obstacle_bound;pr.obstacle_stress->margin=obstacle_margin;pr.obstacle_stress->tolerance=obstacle_tolerance;}
    if(workcap){pr.work_cap=std::make_unique<NativeWorkCap>();pr.work_cap->identity=workcap_identity;pr.work_cap->bound=workcap_bound;pr.work_cap->tolerance=workcap_tolerance;pr.work_cap->diffusion_cfl_limit=workcap_cfl;}
    pr.transport_form=transport_form;
    pr.top_transport_factor=top_transport;pr.top_rhs_factor=top_rhs;pr.top_component_mask=top_mask;
    pr.top_central_factor=top_central;pr.top_llf_factor=top_llf;
    pr.interior_llf_multiplier=llf_multiplier;
    pr.common_transport=transport_state=="minimum-gradient";
    pr.minimum_gradient_pressure=common_pressure=="minimum-gradient";
    pr.retained_common_jump=common_jump=="retained";
    pr.retained_common_flux=common_flux=="retained";
    if(igr_source_law!="original"||mirror_wall_source)
    {pr.constitutive_source=std::make_unique<NativePressureSource>();pr.constitutive_source->law=igr_source_law=="original"?"constant":igr_source_law;pr.constitutive_source->pressure_faces=igr_pressure_faces;pr.constitutive_source->kappa=igr_source_law=="original"?2.:igr_source_kappa;pr.constitutive_source->tolerance=igr_source_tolerance;pr.constitutive_source->mirror_wall_source=mirror_wall_source;}
    if(source_velocity=="minimum-gradient")pr.EnableMinimumGradient();
    if(bulk_strength>0.)pr.EnableBulkGradient(bulk_strength,bulk_policy=="top-strain");
    const State moments=FrontRead(input,pr.retained.ns*4);
    if(!offset_input.empty())pr.SetOffset(FrontRead(offset_input,moments.size()));
    if(!source_input.empty())pr.SetSourceOffset(FrontRead(source_input,pr.stress.ns));
    if(!replacement_input.empty()){pr.SetSourceOffset(FrontRead(replacement_input,pr.stress.ns));pr.replace_source=true;}
    if(advance)
    {
      auto result=MemoryRun(pr);result.work.setup=setup;const State coefficients=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);if(!result.u.empty())result.u=pr.Reconstruct(result.u);
      // On failure the last recovery can belong to a rejected stage; restore
      // the accepted snapshot before serializing its source certificate.
      if(pr.minimum_gradient&&!coefficients.empty()&&coefficients!=pr.minimum_gradient_moments)pr.RecoverMinimumGradient(coefficients);
      std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\""<<(pr.minimum_gradient?"minimum-gradient-full-moment":"joint-density-weighted-primitive")<<"\",\"volume_projection\":\"conservative-moments\",\"retained_order\":"<<degree<<",\"primitive_source_faces\":\""<<(pr.minimum_gradient?"minimum-gradient-continuous":"central")<<"\",\"moment_operator\":\"direct\",\"stress_order\":"<<degree
        <<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns<<",\"sigma_serialized_order\":"<<pr.p<<",\"injected_state_time\":"<<time
        <<",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"mesh_source\":\"explicit saved physical edges\",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"direct_assembly_seconds\":"<<pr.direct_assembly_seconds
        <<",\"slope_penalty_seconds\":"<<pr.slope_seconds<<",\"curvature_seconds\":"<<pr.curvature_seconds<<",\"local_velocity_factors\":"<<pr.local_velocity_factors<<",\"moment_coefficients\":";
      TrackArray(extra,coefficients);extra<<",\"rhs_offset_policy\":\""<<(offset_input.empty()?"none":"supplied frozen conservative RHS correction")<<"\",\"rhs_offset_calls\":"<<pr.offset_calls<<",\"rhs_offset_seconds\":"<<pr.offset_seconds;
      extra<<",\"source_offset_policy\":\""<<(pr.replace_source?"frozen reconstructed source load replaces current raw source; current density operator recomputed every stage":source_input.empty()?"none":"supplied frozen additive Pp volume source; current native source and density operator recomputed every stage")<<"\",\"source_offset_integral\":"<<(pr.replace_source?0.:pr.source_offset_integral)<<",\"source_offset_calls\":"<<pr.source_offset_calls<<",\"source_offset_seconds\":"<<pr.source_offset_seconds<<",\"maximum_negative_source_cell_mean\":"<<pr.maximum_negative_source_cell_mean<<",\"source_offset_load\":";TrackArray(extra,pr.replace_source?State{}:pr.source_offset_load);
      extra<<",\"source_replacement_integral\":"<<(pr.replace_source?pr.source_offset_integral:0.)<<",\"source_replacement_load\":";TrackArray(extra,pr.replace_source?pr.source_offset_load:State{});
      extra<<",\"source_mode\":\""<<(pr.replace_source?"replacement":source_input.empty()?"native":"additive")<<'"';
      extra<<",\"transport_state\":\""<<transport_state<<"\",\"transport_source_policy\":\""<<(pr.common_transport?"same current recovered velocity in full conservative-moment Euler state and complete IGR source; LLF; no derivative penalties or filtering":pr.minimum_gradient?"current minimum-gradient full-moment source and current density; existing primitive derivative penalties":"current primitive-central source and current density; existing primitive derivative penalties")<<"\"";
      extra<<",\"transport_form\":\""<<transport_form<<'"';
      extra<<",\"top_transport_factor\":"<<top_transport<<",\"top_rhs_factor\":"<<top_rhs<<",\"top_component_mask\":"<<top_mask;
      extra<<",\"top_central_factor\":"<<top_central<<",\"top_llf_factor\":"<<top_llf<<",\"split_boundary_policy\":\"unchanged physical exterior flux included in central complement\"";
      extra<<",\"interior_llf_multiplier\":"<<llf_multiplier;
      FrontWriteMinimumGradient(extra,pr);
      FrontWriteBulkGradient(extra,pr);
      FrontWriteConstitutiveSource(extra,pr);
      BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
    }
    const State directions=rhs_only?State{}:FrontRead(direction_input,std::size_t(direction_count)*moments.size());FrontWork work;
    auto base=FrontAssemble(pr,moments,time,work);const auto factor=FrontFactor(base.matrix,work);base.sigma=FrontSolve(base.matrix,factor,base.load,work);
    std::array<double,4> budget{};FrontAdmission admission;const State rhs=FrontRHS(pr,base,base.sigma,time,work,budget,admission);
    if(rhs_only)
    {
      std::ofstream out(pr.o.output);if(!out)throw std::runtime_error("cannot open front operator output");
      out<<std::setprecision(17)<<"{\"schema\":\"igr-dg-front-operator-v1\",\"completed\":true,\"precision\":\"FP64\",\"retained_order\":"<<degree<<",\"elements\":"<<pr.ne<<",\"time\":"<<time<<",\"alpha\":"<<t.base.alpha<<",\"gamma\":"<<gamma_euler;
      auto array=[&](const char* key,const State& x){out<<',';FrontString(out,key);out<<':';TrackArray(out,x);};
      array("edges",edges);array("base_moments",moments);array("base_stress",base.sigma);array("base_rhs",rhs);array("base_boundary_budget",State(budget.begin(),budget.end()));
      array("base_load",base.load);array("source_offset_load",pr.replace_source?State{}:pr.source_offset_load);array("source_replacement_load",pr.replace_source?pr.source_offset_load:State{});
      const State full_sigma=pr.stress_transfer.Apply(pr.stress_transfer.E,base.sigma,pr.ne,1);
      const State transport_physical=pr.common_transport?pr.CommonBoundaryAdapter():pr.retained_transport?base.high:base.physical;
      const auto left_flux=pr.BoundaryFlux(transport_physical,full_sigma,0),right_flux=pr.BoundaryFlux(transport_physical,full_sigma,1);
      for(int c=0;c<4;++c)if(std::abs(left_flux[c]-right_flux[c]-budget[c])>1.e-8)throw std::runtime_error("front exterior flux budget mismatch");
      array("base_boundary_flux_left",State(left_flux.begin(),left_flux.end()));array("base_boundary_flux_right",State(right_flux.begin(),right_flux.end()));
      out<<",\"base_admission\":";FrontWriteAdmission(out,admission);
      out<<",\"transport_state\":\""<<transport_state<<'"';
      out<<",\"top_transport_factor\":"<<top_transport<<",\"top_rhs_factor\":"<<top_rhs<<",\"top_component_mask\":"<<top_mask
         <<",\"top_central_factor\":"<<top_central<<",\"top_llf_factor\":"<<top_llf;
      out<<",\"interior_llf_multiplier\":"<<llf_multiplier;
      FrontWriteMinimumGradient(out,pr);
      FrontWriteBulkGradient(out,pr);
      FrontWriteConstitutiveSource(out,pr);
      out<<",\"base_source_integral\":"<<base.source_integral<<",\"setup_seconds\":"<<setup<<",\"maximum_solve_residual\":"<<work.maximum_solve_residual<<",\"maximum_rhs_defect\":"<<work.shared.max_rhs_defect<<",\"rhs_offset_calls\":"<<pr.offset_calls<<",\"source_offset_calls\":"<<pr.source_offset_calls<<",\"source_offset_integral\":"<<(pr.replace_source?0.:pr.source_offset_integral)<<",\"source_replacement_integral\":"<<(pr.replace_source?pr.source_offset_integral:0.)<<",\"source_mode\":\""<<(pr.replace_source?"replacement":source_input.empty()?"native":"additive")<<"\",\"maximum_negative_source_cell_mean\":"<<pr.maximum_negative_source_cell_mean<<",\"wall_seconds_before_final_output\":"<<Wall(start)<<"}\n";
      return 0;
    }
    const State shared_stress=pr.stress_transfer.Apply(pr.stress_transfer.P,pr.Closure(base.high,time,work.shared),pr.ne,1);const double closure_parity=FrontRelative(shared_stress,base.sigma);
    if(closure_parity>1.e-10)throw std::runtime_error("explicit closure differs from shared instantaneous path");
    const State velocity=FrontVelocity(pr,base.physical),gradient=FrontGradient(pr,velocity);const auto base_time_tangent=FrontDifferentiate(pr,base,factor,rhs,work);
    std::ofstream out(pr.o.output);if(!out)throw std::runtime_error("cannot open front diagnostic output");
    out<<std::setprecision(17)<<"{\"schema\":\"igr-dg-front-tangent-v1\",\"precision\":\"FP64\",\"retained_order\":"<<degree<<",\"serialized_order\":"<<pr.p<<",\"elements\":"<<pr.ne<<",\"time\":"<<time<<",\"alpha\":"<<t.base.alpha<<",\"gamma\":"<<gamma_euler
      <<",\"quad_extra\":"<<t.quad_extra<<",\"effective_sipg_factor\":"<<pr.o.sipg_penalty*pr.n*pr.n<<",\"slope_strength\":"<<slope<<",\"curvature_strength\":"<<curvature<<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns
      <<",\"source_faces\":\"central\",\"closure_boundary\":\"natural\",\"closure_space\":\"l2\",\"state_layout\":\"element,node,component(rho,mx,my,E); retained Pp coefficients\",\"stress_layout\":\"element,node; retained Pp\""
      <<",\"whole_rhs_modes\":[\"full instantaneous\",\"fixed base stress\",\"current source/base matrix\",\"current matrix/base source\"],\"perturbation_rule\":\"base_moments plus/minus epsilon*direction; no automatic normalization or limiting\",\"analytic_scope\":\"exact retained-polynomial Galerkin closure derivative, including central lift and both SIPG consistency terms; whole RHS uses central finite differences\"";
    auto array=[&](const char* name,const State& x){out<<',';FrontString(out,name);out<<':';TrackArray(out,x);};
    array("edges",edges);array("base_moments",moments);array("base_physical_state",base.physical);array("base_stress",base.sigma);array("base_rhs",rhs);array("base_load",base.load);array("base_velocity",velocity);array("base_g",gradient);array("base_boundary_budget",State(budget.begin(),budget.end()));
    State points(pr.weights.size()),nodes(pr.retained.n);for(int i=0;i<pr.retained.n;++i)nodes[i]=pr.retained.physics.basis.nodes[i];for(std::size_t q=0;q<points.size();++q)for(int i=0;i<pr.n;++i)points[q]+=pr.shapes[q][i]*pr.physics.basis.nodes[i];
    array("nodes",nodes);array("quadrature_points",points);array("quadrature_weights",pr.weights);array("epsilons",epsilons);
    out<<",\"base_matrix\":{\"size\":"<<base.matrix.n<<",\"half_bandwidth\":"<<base.matrix.k<<",\"layout\":\"row i stores columns i-k through i+k, padded outside matrix\",\"values\":";TrackArray(out,base.matrix.a);out<<'}';
    out<<",\"base_admission\":";FrontWriteAdmission(out,admission);out<<",\"base_source_integral\":"<<base.source_integral<<",\"shared_closure_relative_parity\":"<<closure_parity<<",\"base_rhs_primitive_and_closure_tangent\":";FrontWriteAnalytic(out,base_time_tangent);
    bool complete=true;out<<",\"directions\":[";
    for(int index=0;index<direction_count;++index)
    {
      const State direction(directions.begin()+index*moments.size(),directions.begin()+(index+1)*moments.size());const auto analytic=FrontDifferentiate(pr,base,factor,direction,work);
      if(index)out<<',';out<<"{\"index\":"<<index<<",\"direction\":";TrackArray(out,direction);out<<",\"direction_coefficient_norm\":"<<Norm(direction)<<",\"analytic\":";FrontWriteAnalytic(out,analytic);out<<",\"epsilon_results\":[";
      State previous;double previous_epsilon=0.;
      for(std::size_t k=0;k<epsilons.size();++k)
      {
        if(k)out<<',';out<<"{\"epsilon\":"<<epsilons[k];
        try
        {
          const auto result=FrontCentral(pr,base,factor,direction,analytic,time,epsilons[k],work);out<<",\"admitted\":true";FrontWritePair(out,result);
          if(!previous.empty())out<<",\"previous_epsilon\":"<<previous_epsilon<<",\"full_jvp_relative_change\":"<<FrontRelative(result.jvp[0],previous);
          previous=result.jvp[0];previous_epsilon=epsilons[k];
        }
        catch(const std::exception& error){complete=false;out<<",\"admitted\":false,\"failure\":";FrontString(out,error.what());}
        out<<'}';out.flush();
      }
      out<<"]}";
    }
    out<<"],\"completed\":"<<(complete?"true":"false")<<",\"setup_seconds\":"<<setup<<",\"wall_seconds_before_final_output\":"<<Wall(start)
      <<",\"work\":{\"assembled_states\":"<<work.states<<",\"matrix_factorizations\":"<<work.factors<<",\"matrix_solves\":"<<work.solves<<",\"rhs_calls\":"<<work.rhs_calls<<",\"admission_points\":"<<work.admission_points
      <<",\"tangent_velocity_factors\":"<<work.tangent_velocity_factors<<",\"tangent_quadrature_points\":"<<work.tangent_points<<",\"assembly_seconds\":"<<work.assembly_seconds<<",\"factor_seconds\":"<<work.factor_seconds<<",\"solve_seconds\":"<<work.solve_seconds<<",\"certificate_seconds\":"<<work.certificate_seconds<<",\"analytic_seconds\":"<<work.analytic_seconds
      <<",\"maximum_solve_residual\":"<<work.maximum_solve_residual<<",\"source_points\":"<<work.shared.source_points<<",\"flux_points\":"<<work.shared.flux_points<<",\"faces\":"<<work.shared.faces<<",\"closure_faces\":"<<work.shared.closure_faces<<",\"maximum_rhs_defect\":"<<work.shared.max_rhs_defect
      <<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"reconstruction_velocity_factors\":"<<pr.local_velocity_factors<<",\"slope_seconds\":"<<pr.slope_seconds<<",\"curvature_seconds\":"<<pr.curvature_seconds<<"}}\n";
    return complete?0:2;
  }
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

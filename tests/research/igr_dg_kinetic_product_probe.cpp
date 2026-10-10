// Test-only conservative correction of the primitive manifold's projected
// kinetic mass-test defect. The target keeps the actual physical LLF faces.
#define DG_IGR_PRIMITIVE_MOMENTS_LIBRARY_ONLY
#include "igr_dg_primitive_moments_probe.cpp"
#undef DG_IGR_PRIMITIVE_MOMENTS_LIBRARY_ONLY

namespace
{
struct KineticProductProblem:PrimitiveMomentProblem
{
  bool correct;
  std::vector<State> product_derivatives,primitive_nodes;
  mutable double product_seconds=0.,max_product_power=0.,max_identity_error=0.,max_correction_ratio=0.;
  mutable std::uint64_t product_cells=0,constant_cells=0,product_face_fluxes=0;
  KineticProductProblem(const MemoryOptions& mo,bool central,bool enabled):PrimitiveMomentProblem(mo,central),correct(enabled)
  {
    for(std::size_t q=0;q<weights.size();++q)
    {double z=0.;for(int i=0;i<n;++i)z+=shapes[q][i]*physics.basis.nodes[i];product_derivatives.push_back(retained.Shape(z,true));}
    for(double z:retained.physics.basis.nodes)primitive_nodes.push_back(Shape(z,false));
  }
  State Residual(const State& moments,const State& sigma,double time,TrackWork& work,std::array<double,4>& budget)const
  {
    State rhs=PrimitiveMomentProblem::Residual(moments,sigma,time,work,budget);
    const auto start=Clock::now();const State& physical=Reconstruct(moments);const int m=retained.n;
    State mass_flux(ne+1,0.);
    for(int face=0;face<=ne;++face)
    {
      if(face==0||face==ne)mass_flux[face]=BoundaryFlux(physical,sigma,face==ne)[0];
      else
      {
        double al=0.,ar=0.;const auto fl=physics.Flux(&physical[Id(face-1,p,0)],sigma[SigmaIndex(face-1,p)],al);
        const auto fr=physics.Flux(&physical[Id(face,0,0)],sigma[SigmaIndex(face,0)],ar);
        mass_flux[face]=.5*(fl[0]+fr[0]-physics.FaceSpeed(al,ar)*(physical[Id(face,0,0)]-physical[Id(face-1,p,0)]));
      }
      ++product_face_fluxes;
    }
    double correction_norm=0.,rhs_norm=0.;
    for(int e=0;e<ne;++e)
    {
      ++product_cells;const double h=edges[e+1]-edges[e];State velocity(2*m),qload(m,0.),qpoly(m,0.);
      std::array<double,2> mean{};
      for(int i=0;i<m;++i)
      {
        const auto value=Point(physical,e,primitive_nodes[i]);
        for(int c=0;c<2;++c){velocity[c*m+i]=value[c+1]/value[0];mean[c]+=retained.physics.basis.weights[i]*velocity[c*m+i];}
      }
      // e=|v|^2-Pi_p|v|^2 is invariant under a constant velocity shift.
      // Center before squaring so nearly constant cells do not subtract large
      // numbers before division by their small velocity variance.
      for(int i=0;i<m;++i)for(int c=0;c<2;++c)velocity[c*m+i]-=mean[c];
      for(std::size_t q=0;q<weights.size();++q)
      {
        double speed2=0.;for(int c=0;c<2;++c)
        {double v=0.;for(int i=0;i<m;++i)v+=low_shapes[q][i]*velocity[c*m+i];speed2+=v*v;}
        for(int i=0;i<m;++i)qload[i]+=weights[q]*low_shapes[q][i]*speed2;
      }
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)qpoly[i]+=retained.mass_inverse(i,j)*qload[j];
      double defect=0.,denominator=0.,velocity_norm=0.;
      for(std::size_t q=0;q<weights.size();++q)
      {
        const State& d=product_derivatives[q];
        double dq=0.;for(int i=0;i<m;++i)dq+=d[i]*qpoly[i]/h;
        for(int c=0;c<2;++c)
        {
          double v=0.,dv=0.;for(int i=0;i<m;++i){v+=low_shapes[q][i]*velocity[c*m+i];dv+=d[i]*velocity[c*m+i]/h;}
          dq-=2.*v*dv;denominator+=h*weights[q]*v*v;velocity_norm+=h*weights[q]*(v+mean[c])*(v+mean[c]);
        }
        const auto u=Point(physical,e,shapes[q]);defect-=.5*h*weights[q]*dq*u[1];
        const auto r=Point(rhs,e,shapes[q]);for(double value:r)rhs_norm+=h*weights[q]*value*value;
      }
      for(int side=0;side<2;++side)
      {
        const int i=side?m-1:0;double error=-qpoly[i];for(int c=0;c<2;++c)error+=velocity[c*m+i]*velocity[c*m+i];
        defect-=(side?1.:-1.)*.5*error*mass_flux[e+side];
      }
      max_product_power=std::max(max_product_power,std::abs(defect));
      if(denominator<=1.e-26*std::max(velocity_norm,1.))
      {++constant_cells;continue;}
      if(!correct)continue;
      const double scale=-defect/denominator;State change(n*2,0.);std::array<double,2> mean_change{};double power=0.;
      for(int i=0;i<n;++i)for(int c=0;c<2;++c)
      {for(int j=0;j<m;++j)change[2*i+c]+=scale*state_transfer.E(i,j)*velocity[c*m+j];rhs[Id(e,i,c+1)]+=change[2*i+c];}
      for(std::size_t q=0;q<weights.size();++q)for(int c=0;c<2;++c)
      {
        double v=0.,delta=0.;for(int i=0;i<m;++i)v+=low_shapes[q][i]*velocity[c*m+i];for(int i=0;i<n;++i)delta+=shapes[q][i]*change[2*i+c];
        power+=h*weights[q]*v*delta;mean_change[c]+=h*weights[q]*delta;correction_norm+=h*weights[q]*delta*delta;
      }
      max_identity_error=std::max(max_identity_error,std::abs(power+defect));
      for(double value:mean_change)max_identity_error=std::max(max_identity_error,std::abs(value));
    }
    max_correction_ratio=std::max(max_correction_ratio,std::sqrt(correction_norm/std::max(rhs_norm,1.e-100)));
    const auto totals=Integrals(rhs,edges);for(int c=0;c<4;++c)work.max_rhs_defect=std::max(work.max_rhs_defect,std::abs(totals[c]-budget[c]));
    const double elapsed=Wall(start);product_seconds+=elapsed;work.transport+=elapsed;return rhs;
  }
};
void KineticProductSelfTest()
{
  MemoryOptions mo;mo.recovery.recovery="existing";auto& t=mo.recovery.boundary.compatibility.track;
  t.initial="entropy";t.mode="uniform";t.base.elements=8;t.base.order=9;t.base.alpha=.04;t.closure_space="l2";t.closure_boundary="natural";t.quad_extra=19;
  mo.retained_order=mo.stress_order=3;KineticProductProblem pr(mo,true,true),base(mo,true,false);
  State low(pr.retained.ns*4);
  for(int e=0;e<pr.ne;++e)for(int i=0;i<pr.retained.n;++i)
  {
    const double z=pr.retained.physics.basis.nodes[i];const int j=(e*pr.retained.n+i)*4;
    low[j]=1.1+.05*z+.01*e;low[j+1]=.4+.09*z*z*z+.01*e;low[j+2]=-.1+.03*z*z;low[j+3]=3.+.07*z*z*z;
  }
  const State moments=pr.state_transfer.Apply(pr.state_transfer.E,low,pr.ne,4);TrackWork w;const State sigma=pr.Closure(moments,0.,w);
  std::array<double,4>a{},b{};const State r=pr.Residual(moments,sigma,0.,w,a),rb=base.Residual(moments,sigma,0.,w,b);
  double fixed=0.,nonzero=0.;for(std::size_t i=0;i<r.size();++i){if(i%4==0||i%4==3)fixed=std::max(fixed,std::abs(r[i]-rb[i]));else nonzero=std::max(nonzero,std::abs(r[i]-rb[i]));}
  if(fixed!=0.||nonzero<1.e-10||pr.max_identity_error>1.e-10||w.max_rhs_defect>1.e-9)throw std::runtime_error("conservative kinetic product identity check failed");
  std::cout<<"kinetic product checks passed: power="<<pr.max_product_power<<" identity="<<pr.max_identity_error<<" correction="<<nonzero<<'\n';
}
}
int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cpu");
  try
  {
    MemoryOptions mo;std::string source_faces="central",correction="product";std::vector<char*> common{argv[0]};
    for(int i=1;i<argc;++i)
    {
      const std::string flag=argv[i];if(flag=="--primitive-source-faces"||flag=="--kinetic-correction")
      {if(++i==argc)throw std::runtime_error("missing product option");if(flag=="--primitive-source-faces")source_faces=argv[i];else correction=argv[i];}else common.push_back(argv[i]);
    }
    mo.recovery=RecoveryParse(int(common.size()),common.data());auto& t=mo.recovery.boundary.compatibility.track;
    if(t.self_test){KineticProductSelfTest();return 0;}
    if((source_faces!="central"&&source_faces!="broken")||(correction!="product"&&correction!="none"))throw std::runtime_error("unknown product/source selector");
    const int degree=t.base.order;mo.retained_order=mo.stress_order=degree;t.base.order=3*degree;
    const auto start=Clock::now();KineticProductProblem pr(mo,source_faces=="central",correction=="product");const double setup=Wall(start);auto result=MemoryRun(pr);result.work.setup=setup;
    const State moment_coefficients=result.u.empty()?State{}:pr.state_transfer.Apply(pr.state_transfer.P,result.u,pr.ne,4);
    if(!result.u.empty())result.u=pr.Reconstruct(result.u);
    std::ostringstream extra;extra<<std::setprecision(17)<<",\"source_recovery\":\"joint-density-weighted-primitive\",\"volume_projection\":\"conservative-moments\",\"retained_order\":"<<degree
      <<",\"primitive_source_faces\":\""<<source_faces<<"\",\"kinetic_correction\":\""<<correction<<"\",\"stress_order\":"<<degree
      <<",\"actual_sigma_solve_dofs\":"<<pr.stress.ns<<",\"actual_euler_dofs_per_field\":"<<pr.retained.ns<<",\"sigma_serialized_order\":"<<pr.p
      <<",\"injected_state_time\":0,\"local_velocity_factors\":"<<pr.local_velocity_factors<<",\"reconstruction_cells\":"<<pr.reconstruction_cells
      <<",\"reconstruction_seconds\":"<<pr.reconstruction_seconds<<",\"moment_transfer_seconds\":"<<pr.transfer_seconds<<",\"galerkin_seconds\":"<<pr.galerkin_seconds
      <<",\"kinetic_product_seconds\":"<<pr.product_seconds<<",\"kinetic_product_cells\":"<<pr.product_cells<<",\"kinetic_constant_cells\":"<<pr.constant_cells
      <<",\"kinetic_product_face_fluxes\":"<<pr.product_face_fluxes<<",\"maximum_product_power\":"<<pr.max_product_power
      <<",\"maximum_product_identity_error\":"<<pr.max_identity_error<<",\"maximum_correction_rhs_ratio\":"<<pr.max_correction_ratio<<",\"moment_coefficients\":";
    TrackArray(extra,moment_coefficients);BoundaryWrite(pr,result,extra.str());return result.failure.empty()?0:2;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

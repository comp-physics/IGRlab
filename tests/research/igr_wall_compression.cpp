#include "igrlab/compat.hpp"
#include "mfem.hpp"
#include "sim/igr_sim.hpp"
#include "igr_wall_consistency_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace
{
constexpr double gamma_gas=1.4;
struct WallOptions
{
  double compression_x=2.,compression_y=1.,rotation_rate=1.;
  double radial_amplitude=6.,radial_width=.06;
  const char* scenario="compression";const char* radial_profile="flux-sine";
};

// Test-only initial/final exports. Preserve the native polynomial coefficients,
// reference nodes and physical nodal coordinates, rather than a display grid.
void WriteNative(const char* path,const dg_sim::ExampleContext& ctx,
                 const dg_sim::EulerStateBundle& state,bool include_sigma)
{
  int ranks=0;MPI_Comm_size(ctx.Mesh().GetComm(),&ranks);
  MFEM_VERIFY(ranks==1,"The bounded wall research export requires one rank.");
  auto* fes=state.sol.FESpace();auto& mesh=*fes->GetMesh();
  const auto* sigma=include_sigma&&state.field_list.igr?&state.field_list.igr->GetSigma():nullptr;
  std::ofstream out(path);MFEM_VERIFY(out.good(),"Cannot write wall research native export.");
  out<<std::setprecision(17)<<"{\"schema\":\"igr-wall-native-v1\",\"dimension\":2,\"gamma\":"<<gamma_gas
     <<",\"phase\":\""<<(include_sigma?"accepted-final":"projected-initial")<<"\",\"requested_horizon\":"<<ctx.run_cfg.opts.t_final
     <<",\"sigma_current\":"<<(sigma?"true":"false")<<",\"boundary_attributes\":[";
  for(int b=0;b<mesh.GetNBE();++b){if(b)out<<',';out<<mesh.GetBdrAttribute(b);}out<<"],\"boundary_faces\":[";
  for(int b=0;b<mesh.GetNBE();++b)
  {
    if(b)out<<',';auto* tr=mesh.GetBdrFaceTransformations(b);
    out<<"{\"attribute\":"<<mesh.GetBdrAttribute(b)<<",\"element\":"<<tr->Elem1No<<",\"reference_endpoints\":[";
    for(int end=0;end<2;++end)
    {
      mfem::IntegrationPoint ip;ip.Set1w(double(end),1.);tr->SetAllIntPoints(&ip);
      const auto& local=tr->GetElement1IntPoint();if(end)out<<',';out<<'['<<local.x<<','<<local.y<<']';
    }
    out<<"]}";
  }
  out<<"],\"elements\":[";
  mfem::Array<int> vd,sd;mfem::Vector x(2);const double* data=state.sol.HostRead();const double* sig=sigma?sigma->HostRead():nullptr;
  for(int e=0;e<fes->GetNE();++e)
  {
    if(e)out<<',';const auto& fe=*fes->GetFE(e);const int nd=fe.GetDof();fes->GetElementVDofs(e,vd);
    out<<"{\"order\":"<<fe.GetOrder()<<",\"reference_nodes\":[";
    for(int j=0;j<nd;++j){if(j)out<<',';const auto& ip=fe.GetNodes().IntPoint(j);out<<'['<<ip.x<<','<<ip.y<<']';}
    out<<"],\"physical_nodes\":[";
    for(int j=0;j<nd;++j){if(j)out<<',';mesh.GetElementTransformation(e)->Transform(fe.GetNodes().IntPoint(j),x);out<<'['<<x[0]<<','<<x[1]<<']';}
    out<<"],\"state\":[";
    for(int v=0;v<4;++v){if(v)out<<',';out<<'[';for(int j=0;j<nd;++j){if(j)out<<',';out<<data[vd[j+nd*v]];}out<<']';}
    out<<']';
    if(sigma)
    {
      auto* sfes=sigma->FESpace();sfes->GetElementDofs(e,sd);const auto& sfe=*sfes->GetFE(e);
      out<<",\"sigma_reference_nodes\":[";
      for(int j=0;j<sd.Size();++j){if(j)out<<',';const auto& ip=sfe.GetNodes().IntPoint(j);out<<'['<<ip.x<<','<<ip.y<<']';}
      out<<"],\"sigma\":[";for(int j=0;j<sd.Size();++j){if(j)out<<',';out<<sig[sd[j]];}out<<']';
    }
    out<<'}';
  }
  out<<"]}\n";
}
}
int main(int argc,char** argv)
{
  dg_sim::CaseSpec<WallOptions,mfem::DGSEMSurfaceFlux> spec;
  spec.case_name="igr_wall_compression";spec.case_file=__FILE__;
  // The campaign supplies a unit Cartesian box using --mesh. Never create or
  // overwrite a source-tree mesh as a side effect of a research invocation.
  spec.default_mesh_path="";spec.gamma=gamma_gas;spec.dg_backend_default=dg_sim::DGBackend::DGSEM;
  spec.need_h1=false;spec.use_igr=true;spec.igr_defaults.space="L2";spec.igr_defaults.grad="L2";
  spec.glvis.precision=17;spec.vtk.enabled=true;spec.vtk.print_dataset_name=true;
  spec.diagnostics.h1_diagnostics=false;spec.diagnostics.l2_diagnostics=true;
  spec.configure_case=[](dg_sim::CaseConfig& cfg)
  {
    cfg.defaults.t_final=.08;cfg.defaults.par_ref_levels=0;cfg.rk3_config.limit_initial=false;
    cfg.igr_config.solve_mode=igrlab::IGRSolveMode::Krylov;
    cfg.igr_config.lin_solver_type=igrlab::IGRLinearSolverType::Auto;
    cfg.igr_config.prec_type=igrlab::IGRPrecondType::Auto;cfg.igr_config.ip_kind=igrlab::IGRIPKind::SIPG;
    cfg.igr_config.lin_rtol=1e-8;cfg.igr_config.lin_atol=0.;cfg.igr_config.lin_maxit=1000;cfg.igr_config.lin_print=0;
    cfg.igr_config.sigma_cadence=igrlab::IGRSigmaUpdateCadence::EveryStep;
    cfg.igr_config.sigma_update_interval=1;cfg.igr_config.stage_rtol_scale=1.;
  };
  spec.add_options=[](mfem::OptionsParser& args,WallOptions& opts)
  {
    args.AddOption(&opts.compression_x,"-compression-x","--compression-x","Smooth x velocity amplitude in -A sin(2 pi x).");
    args.AddOption(&opts.compression_y,"-compression-y","--compression-y","Smooth y velocity amplitude in -B sin(2 pi y).");
    args.AddOption(&opts.scenario,"-wall-scenario","--wall-scenario","Research initial condition: compression, rotation or radial-compression.");
    args.AddOption(&opts.rotation_rate,"-rotation-rate","--rotation-rate","Rigid rotation rate for the annulus equilibrium control.");
    args.AddOption(&opts.radial_amplitude,"-radial-amplitude","--radial-amplitude","Maximum intended radial velocity for smooth wall approach.");
    args.AddOption(&opts.radial_width,"-radial-width","--radial-width","Approach width for Gaussian/flat radial profiles; unused by flux-sine.");
    args.AddOption(&opts.radial_profile,"-radial-profile","--radial-profile","Radial approach profile: flux-sine, gaussian or flat.");
  };
  spec.make_ic=[](const WallOptions& opts,const dg_sim::ExampleContext& ctx,double gamma)
  {
    MFEM_VERIFY(std::isfinite(opts.compression_x)&&std::isfinite(opts.compression_y),"Compression amplitudes must be finite.");
    auto& mesh=ctx.Mesh();MFEM_VERIFY(mesh.Dimension()==2&&mesh.SpaceDimension()==2,"Wall case requires a 2D mesh.");
    const std::string scenario=opts.scenario;
    MFEM_VERIFY(scenario=="compression"||scenario=="rotation"||scenario=="radial-compression","Use compression, rotation or radial-compression.");
    if(scenario=="rotation"||scenario=="radial-compression")
    {
      MFEM_VERIFY(std::isfinite(opts.rotation_rate),"Rotation rate must be finite.");
      for(int i=0;i<mesh.GetNV();++i)
      {
        const double* v=mesh.GetVertex(i);const double r=std::hypot(v[0],v[1]);
        MFEM_VERIFY(r>=.5-1e-12&&r<=1.+1e-12,"Rotation control requires the generated .5-to-1 annulus.");
      }
      if(scenario=="rotation")
      {
        const double omega=opts.rotation_rate;
        return dg_sim::MakeVectorFunctionIC(4,[=](const mfem::Vector& x,mfem::Vector& u)
          {igr_wall_consistency::Rotation(x,u,omega,gamma);});
      }
      const std::string profile=opts.radial_profile;const double amplitude=opts.radial_amplitude,width=opts.radial_width,pi=std::acos(-1.);
      MFEM_VERIFY(std::isfinite(amplitude)&&amplitude>0&&std::isfinite(width)&&width>0,"Invalid radial approach amplitude/width.");
      MFEM_VERIFY(profile=="flux-sine"||profile=="gaussian"||profile=="flat","Use flux-sine, gaussian or flat radial profile.");
      double lo=0.,hi=.5;
      for(int i=0;i<64;++i)
      {
        const double mid=.5*(lo+hi);
        if(std::tan(pi*mid)-2*pi*(mid+1./3.)>0)hi=mid;else lo=mid;
      }
      const double peak=.5*(lo+hi),sine_scale=amplitude*std::sqrt(.25+.75*peak)/std::sin(pi*peak);
      return dg_sim::MakeVectorFunctionIC(4,[=](const mfem::Vector& x,mfem::Vector& u)
      {
        const double r=x.Norml2(),left=r-.5,right=1.-r;double speed=0.;
        if(profile=="flux-sine")speed=sine_scale*std::sin(pi*(r*r-.25)/.75)/r;
        else if(left>0&&right>0)
        {
          if(profile=="flat")speed=amplitude*std::exp(2*std::pow(width/.25,2)-std::pow(width/left,2)-std::pow(width/right,2));
          else speed=amplitude*(-std::expm1(-std::pow(left/width,2)))*(-std::expm1(-std::pow(right/width,2)))
            /std::pow(-std::expm1(-std::pow(.25/width,2)),2);
        }
        u.SetSize(4);u[0]=1.;u[1]=speed*x[0]/r;u[2]=speed*x[1]/r;u[3]=1./(gamma-1.)+.5*speed*speed;
      });
    }
    double xmin=1e100,xmax=-1e100,ymin=1e100,ymax=-1e100;
    for(int i=0;i<mesh.GetNV();++i){const double* v=mesh.GetVertex(i);xmin=std::min(xmin,v[0]);xmax=std::max(xmax,v[0]);ymin=std::min(ymin,v[1]);ymax=std::max(ymax,v[1]);}
    MFEM_VERIFY(std::abs(xmin)<1e-12&&std::abs(ymin)<1e-12&&std::abs(xmax-1.)<1e-12&&std::abs(ymax-1.)<1e-12,
      "This initial condition is qualified for the unit box; supply --mesh with a unit Cartesian mesh.");
    const double a=opts.compression_x,b=opts.compression_y;
    return dg_sim::MakeVectorFunctionIC(4,[=](const mfem::Vector& x,mfem::Vector& u)
    {
      const double pi=std::acos(-1.);const double vx=-a*std::sin(2.*pi*x[0]),vy=-b*std::sin(2.*pi*x[1]);
      u.SetSize(4);u[0]=1.;u[1]=vx;u[2]=vy;u[3]=1./(gamma-1.)+.5*(vx*vx+vy*vy);
    });
  };
  spec.build_markers=[](dg_sim::ExampleContext& ctx)
  {
    const auto& mesh=ctx.Mesh();mfem::Array<int> marker(mesh.bdr_attributes.Max());marker=0;
    for(int b=0;b<mesh.GetNBE();++b)marker[mesh.GetBdrAttribute(b)-1]=1;
    dg_sim::BoundaryMarkerMap markers;markers.Add(dg_sim::BoundaryRole::Slip,std::move(marker));return markers;
  };
  spec.apply_bc=[](const WallOptions&,const dg_sim::ExampleContext&,auto& bc,const auto& markers){bc.AddSlipWall(markers);};
  spec.after_init=[](const WallOptions&,dg_sim::ExampleContext& ctx,dg_sim::EulerStateBundle& state)
    {WriteNative("igr-wall-initial.json",ctx,state,false);};
  spec.postprocess=[](const WallOptions& opts,const dg_sim::ExampleContext& ctx,const dg_sim::EulerStateBundle& state,mfem::VectorCoefficient&)
  {
    WriteNative("igr-wall-final.json",ctx,state,true);
    std::cout<<"[igr-wall-research] initial="<<(std::string(opts.scenario)=="rotation"?"rigid-rotation-annulus":(std::string(opts.scenario)=="radial-compression"?"smooth-radial-wall-approach":"smooth-sinusoidal-box"))
      <<" rotation_rate="<<opts.rotation_rate<<" compression_x="<<opts.compression_x
      <<" compression_y="<<opts.compression_y<<" radial_profile="<<opts.radial_profile
      <<" radial_amplitude="<<opts.radial_amplitude<<" radial_width="<<opts.radial_width
      <<" natural_scalar_boundary=1 native_export=igr-wall-final.json\n";
    return 0;
  };
  return dg_sim::Solver::solve(spec,argc,argv);
}

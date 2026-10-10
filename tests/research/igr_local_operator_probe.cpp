#include "igrlab/compat.hpp"
// Bounded test-only frozen-operator export. Uses production physics and faces;
// no application recording or replay interface is introduced.
#include "mfem.hpp"
#include "physics/euler/conversions.hpp"
#include "physics/igr/IGR.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point t)
{ return std::chrono::duration<double>(Clock::now()-t).count(); }
class InverseDensity final : public mfem::Coefficient
{
  mfem::ParGridFunction& state_;
public:
  explicit InverseDensity(mfem::ParGridFunction& state) : state_(state) {}
  mfem::real_t Eval(mfem::ElementTransformation& tr,
                   const mfem::IntegrationPoint& ip) override
  {
    mfem::Vector u; state_.GetVectorValue(tr, ip, u);
    MFEM_VERIFY(std::isfinite(u(0)) && u(0)>1e-12, "Invalid quadrature density.");
    return 1.0/u(0);
  }
};
template<class T> void Array(std::ostream& out, const std::vector<T>& values)
{
  out << '[';
  for (std::size_t i=0;i<values.size();++i) { if(i)out<<','; out<<values[i]; }
  out << ']';
}
void Matrix(std::ostream& out, const mfem::DenseMatrix& a)
{
  out<<'[';
  for(int i=0;i<a.Height();++i)
  { if(i)out<<','; out<<'['; for(int j=0;j<a.Width();++j){if(j)out<<',';out<<a(i,j);}out<<']'; }
  out<<']';
}
}

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv); mfem::Hypre::Init();
  MFEM_VERIFY(mfem::Mpi::WorldSize()==1,"One rank required.");
  mfem::Device device("cpu");
  const char *mesh_path="", *state_path="", *output="";
  int nx=8,ny=4,p=3;
  mfem::real_t c_alpha=1.0,width=.5,shift=0.0,perturb=0.0;
  mfem::OptionsParser args(argc,argv);
  args.AddOption(&mesh_path,"-m","--mesh","Existing quad mesh; default square-cell strip.");
  args.AddOption(&state_path,"-u","--nodal-state","Test nodal input: ne, points/cell, then x y rho mx my E, cell by cell.");
  args.AddOption(&output,"-o","--output","Required bounded research JSON output.");
  args.AddOption(&nx,"-nx","--nx","Cartesian cells in x.");
  args.AddOption(&ny,"-ny","--ny","Cartesian cells in y.");
  args.AddOption(&p,"-p","--order","Target Euler and GL sigma order.");
  args.AddOption(&c_alpha,"-ca","--c-alpha","Physical alpha_e=C*h_e^2.");
  args.AddOption(&width,"-w","--width-cells","Prescribed tanh width/h.");
  args.AddOption(&shift,"-sh","--shift-cells","Prescribed tanh translation/h.");
  args.AddOption(&perturb,"-dp","--density-perturbation","Multiply all conserved fields by 1+dp*sin(3x)*cos(2y).");
  args.Parse(); if(!args.Good()){args.PrintUsage(std::cout);return 2;}
  MFEM_VERIFY(output[0] && p>=1 && p<=8 && nx>=2 && ny>=1 && c_alpha>0 &&
    std::isfinite(c_alpha) && width>0 && std::isfinite(width) &&
    std::isfinite(shift) && std::isfinite(perturb) && std::abs(perturb)<.1,
    "Invalid bounded probe parameters.");
  auto start=Clock::now();
  mfem::Mesh serial = mesh_path[0] ? mfem::Mesh(mesh_path,1,1,true) :
    mfem::Mesh::MakeCartesian2D(nx,ny,mfem::Element::QUADRILATERAL,true,1.0,double(ny)/nx);
  mfem::ParMesh mesh(MPI_COMM_WORLD,serial);
  MFEM_VERIFY(mesh.Dimension()==2 && !mesh.Nonconforming(),"Conforming 2D mesh required.");
  mfem::L2_FECollection ufe(p,2,mfem::BasisType::GaussLobatto);
  mfem::L2_FECollection sfe(p,2,mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace ufes(&mesh,&ufe,4,mfem::Ordering::byNODES),sfes(&mesh,&sfe);
  mfem::ParGridFunction state(&ufes);
  const int ne=mesh.GetNE(),q=sfes.GetFE(0)->GetDof(),n=sfes.GetVSize();
  MFEM_VERIFY(n<=50000 && ne*q==n,"Probe scalar DOF cap exceeded.");
  std::ifstream input;
  int source_q=0,source_p=-1;
  std::unique_ptr<mfem::L2_FECollection> source_collection;
  if(state_path[0])
  {
    input.open(state_path);int input_ne=0;input>>input_ne>>source_q;
    source_p=int(std::llround(std::sqrt(double(source_q))))-1;
    MFEM_VERIFY(input && input_ne==ne && source_p>=1 && source_p<=8 &&
      (source_p+1)*(source_p+1)==source_q && p>=source_p,
      "Invalid nodal input shape or target order below input sampling degree.");
    // MFEM VTK samples are equally spaced; they need not be LGL nodes.
    source_collection=std::make_unique<mfem::L2_FECollection>(source_p,2,mfem::BasisType::ClosedUniform);
  }
  mfem::Vector x(2),y(2),primitive(4),conserved(4),shape;
  mfem::Array<int> vdofs;
  double match_error=0.0;
  for(int e=0;e<ne;++e)
  {
    MFEM_VERIFY(ufes.GetFE(e)->GetGeomType()==mfem::Geometry::SQUARE,"Uniform quads required.");
    auto* tr=mesh.GetElementTransformation(e);
    const auto& nodes=ufes.GetFE(e)->GetNodes(); ufes.GetElementVDofs(e,vdofs);
    mfem::DenseMatrix input_values(4,source_q);
    if(input.is_open())
    {
      const auto* source_fe=source_collection->FiniteElementForGeometry(mfem::Geometry::SQUARE);
      const auto& source_nodes=source_fe->GetNodes();std::vector<int> used(source_q,0);
      for(int k=0;k<source_q;++k)
      {
        input>>x(0)>>x(1); for(int c=0;c<4;++c)input>>conserved(c);
        MFEM_VERIFY(input && std::isfinite(x(0)) && std::isfinite(x(1)),"Truncated/nonfinite nodal input.");
        int best=-1;double distance=INFINITY;
        for(int j=0;j<source_q;++j)if(!used[j])
        {tr->Transform(source_nodes.IntPoint(j),y);y-=x;double d=y.Norml2();if(d<distance){distance=d;best=j;}}
        MFEM_VERIFY(best>=0 && distance<1e-10,"Input element/node geometry mismatch.");
        match_error=std::max(match_error,distance);used[best]=1;
        for(int c=0;c<4;++c)
        {MFEM_VERIFY(std::isfinite(conserved(c)),"Nonfinite conserved input.");input_values(c,best)=conserved(c);}
      }
    }
    for(int j=0;j<q;++j)
    {
      tr->Transform(nodes.IntPoint(j),x);
      if(input.is_open())
      {
        shape.SetSize(source_q);
        source_collection->FiniteElementForGeometry(mfem::Geometry::SQUARE)->CalcShape(nodes.IntPoint(j),shape);
        input_values.Mult(shape,conserved);
      }
      else
      {
        double mix=.5*(1+std::tanh((x(0)-(.5+shift/nx))/(width/nx)));
        primitive(0)=1+mix*(8.0/3.0-1); primitive(1)=2*std::sqrt(1.4)/primitive(0);
        primitive(2)=0;primitive(3)=1+mix*3.5;
        mfem::EulerPrimToConservative(primitive,2,1.4,conserved);
      }
      conserved*=1+perturb*std::sin(3*x(0))*std::cos(2*x(1));
      MFEM_VERIFY(conserved(0)>1e-12 && conserved(3)-.5*(conserved(1)*conserved(1)+conserved(2)*conserved(2))/conserved(0)>0,
        "Interpolated state is inadmissible.");
      for(int c=0;c<4;++c)state(vdofs[c*q+j])=conserved(c);
    }
  }
  if(input.is_open()){std::string extra;MFEM_VERIFY(!(input>>extra),"Unexpected trailing nodal input.");}
  igrlab::IGRSolver igr(sfes,ufes,2,4,c_alpha);
  igr.SetAlphaLengthScalePolicy(igrlab::IGRAlphaLengthScalePolicy::Local);
  igr.SetGradBackend(igrlab::IGRGradBackend::BrokenDG);igr.SetExactDGFaceOperator(true);
  igr.SetSolveMode(igrlab::IGRSolveMode::SmootherSweeps);igr.SetSweeps(0,0);
  const auto form_start=Clock::now();igr.UpdateFromState(state,false);double form_s=Seconds(form_start);
  const auto* op=igr.GetLinearOperator();
  MFEM_VERIFY(op && op->Height()==n && igr.GetLastExactFaceEffective(),"Production face operator unavailable.");
  std::vector<std::vector<int>> dofs(ne);
  mfem::Array<int> ed;
  for(int e=0;e<ne;++e){sfes.GetElementDofs(e,ed);dofs[e].assign(ed.begin(),ed.end());}
  std::vector<std::set<int>> support(ne),conflict(ne);
  for(int e=0;e<ne;++e)support[e].insert(e);
  for(int f=0;f<mesh.GetNumFaces();++f)
  {int a,b;mesh.GetFaceElements(f,&a,&b);if(a>=0 && b>=0){support[a].insert(b);support[b].insert(a);}}
  for(const auto& row:support)for(int a:row)for(int b:row)if(a!=b)conflict[a].insert(b);
  std::vector<int> colors(ne,-1),used(ne,-1),offset(ne+1,0);int nc=0;
  for(int e=0;e<ne;++e)
  {for(int j:conflict[e])if(colors[j]>=0)used[colors[j]]=e;int c=0;while(used[c]==e)++c;
   colors[e]=c;nc=std::max(nc,c+1);offset[e+1]=offset[e]+support[e].size();}
  const std::size_t entries=std::size_t(offset.back())*q*q;
  MFEM_VERIFY(entries<=16000000,"Sampled block storage cap exceeded.");
  std::vector<double> blocks(entries);mfem::Vector unit(n),action(n),check(n);
  const auto sample_start=Clock::now();
  for(int c=0;c<nc;++c)for(int j=0;j<q;++j)
  {
    unit=0.;for(int e=0;e<ne;++e)if(colors[e]==c)unit(dofs[e][j])=1.;op->Mult(unit,action);
    for(int e=0;e<ne;++e){int slot=offset[e];for(int other:support[e])
      {if(colors[other]==c)for(int i=0;i<q;++i)blocks[(std::size_t(slot)*q+j)*q+i]=action(dofs[e][i]);++slot;}}
  }
  double sampling_s=Seconds(sample_start),graph_error=0.;
  for(int k=0;k<3;++k)
  {
    for(int i=0;i<n;++i)unit(i)=std::sin((.23+.13*k)*(i+1))+.17*std::cos(.51*(i+1));
    op->Mult(unit,action);check=0.;
    for(int e=0;e<ne;++e){int slot=offset[e];for(int other:support[e])
      {for(int i=0;i<q;++i)for(int j=0;j<q;++j)check(dofs[e][i])+=blocks[(std::size_t(slot)*q+j)*q+i]*unit(dofs[other][j]);++slot;}}
    check-=action;graph_error=std::max(graph_error,check.Norml2()/action.Norml2());
  }
  MFEM_VERIFY(graph_error<1e-11,"Face graph reconstruction failed.");
  std::vector<int> rows,cols;std::vector<double> values;
  for(int e=0;e<ne;++e){int slot=offset[e];for(int other:support[e])
    {for(int i=0;i<q;++i)for(int j=0;j<q;++j){double a=blocks[(std::size_t(slot)*q+j)*q+i];
      if(a!=0.){MFEM_VERIFY(std::isfinite(a),"Nonfinite operator sample.");rows.push_back(dofs[e][i]);cols.push_back(dofs[other][j]);values.push_back(a);}}++slot;}}
  InverseDensity invrho(state);mfem::MassIntegrator mass,react(invrho);
  std::vector<mfem::DenseMatrix> masses(ne),reactions(ne),volumes(ne);
  for(int e=0;e<ne;++e)
  {auto* tr=mesh.GetElementTransformation(e);mass.AssembleElementMatrix(*sfes.GetFE(e),*tr,masses[e]);
   react.AssembleElementMatrix(*sfes.GetFE(e),*tr,reactions[e]);volumes[e].SetSize(q);}
  const auto* base_op=igr.GetLinearBaseOperator();
  MFEM_VERIFY(base_op && base_op->Height()==n,"Production volume operator unavailable.");
  for(int j=0;j<q;++j)
  {unit=0.;for(int e=0;e<ne;++e)unit(dofs[e][j])=1.;base_op->Mult(unit,action);
   for(int e=0;e<ne;++e)for(int i=0;i<q;++i)volumes[e](i,j)=action(dofs[e][i]);}
  for(int i=0;i<n;++i)unit(i)=std::sin(.39*(i+1));
  base_op->Mult(unit,action);check=0.;
  for(int e=0;e<ne;++e)for(int i=0;i<q;++i)for(int j=0;j<q;++j)
    check(dofs[e][i])+=volumes[e](i,j)*unit(dofs[e][j]);
  check-=action;const double volume_error=check.Norml2()/action.Norml2();
  MFEM_VERIFY(volume_error<1e-11,"Volume block reconstruction failed.");
  std::ofstream out(output);MFEM_VERIFY(out,"Cannot open export.");out<<std::setprecision(17);
  out<<"{\"schema\":\"igr-frozen-local-operator-v1\",\"n\":"<<n<<",\"ne\":"<<ne<<",\"order\":"<<p;
  out<<",\"source_order\":"<<source_p<<",\"c_alpha\":"<<c_alpha<<",\"density_perturbation\":"<<perturb;
  out<<",\"source_type\":\""<<(state_path[0]?"saved-polynomial-state":"prescribed-tanh")<<"\"";
  out<<",\"source_match_error\":"<<match_error<<",\"operator_form_seconds\":"<<form_s<<",\"sampling_seconds\":"<<sampling_s;
  out<<",\"operator_probes\":"<<nc*q+3<<",\"volume_probes\":"<<q+1<<",\"volume_error\":"<<volume_error;
  out<<",\"graph_error\":"<<graph_error<<",\"element_dofs\":[";
  for(int e=0;e<ne;++e){if(e)out<<',';Array(out,dofs[e]);}out<<"],\"neighbors\":[";
  for(int e=0;e<ne;++e){if(e)out<<',';std::vector<int> v;for(int j:support[e])if(j!=e)v.push_back(j);Array(out,v);}out<<']';
  out<<",\"element_vertices\":[";
  for(int e=0;e<ne;++e){if(e)out<<',';mesh.GetElementVertices(e,ed);std::vector<int> v(ed.begin(),ed.end());Array(out,v);}out<<']';
  out<<",\"reference_nodes\":[";const auto& nodes=sfes.GetFE(0)->GetNodes();
  for(int i=0;i<q;++i){if(i)out<<',';out<<'['<<nodes.IntPoint(i).x<<','<<nodes.IntPoint(i).y<<']';}out<<']';
  std::vector<std::vector<double>> coords(n),centers(ne);std::vector<double> alpha(ne);
  for(int e=0;e<ne;++e)
  {mesh.GetElementCenter(e,x);centers[e]={x(0),x(1)};double h=mesh.GetElementSize(e,1);alpha[e]=c_alpha*h*h;
   for(int i=0;i<q;++i){mesh.GetElementTransformation(e)->Transform(nodes.IntPoint(i),x);coords[dofs[e][i]]={x(0),x(1)};}}
  out<<",\"dof_coordinates\":[";for(int i=0;i<n;++i){if(i)out<<',';Array(out,coords[i]);}out<<']';
  out<<",\"element_centers\":[";for(int e=0;e<ne;++e){if(e)out<<',';Array(out,centers[e]);}out<<']';
  out<<",\"element_alpha\":";Array(out,alpha);
  out<<",\"A\":{\"rows\":";Array(out,rows);out<<",\"cols\":";Array(out,cols);out<<",\"values\":";Array(out,values);out<<'}';
  out<<",\"mass_blocks\":[";for(int e=0;e<ne;++e){if(e)out<<',';Matrix(out,masses[e]);}out<<']';
  out<<",\"reaction_blocks\":[";for(int e=0;e<ne;++e){if(e)out<<',';Matrix(out,reactions[e]);}out<<']';
  out<<",\"volume_blocks\":[";for(int e=0;e<ne;++e){if(e)out<<',';Matrix(out,volumes[e]);}out<<']';
  out<<",\"b\":[";const auto& b=igr.GetLinearRHS();for(int i=0;i<n;++i){if(i)out<<',';out<<b(i);}out<<"]}\n";
  out.close();MFEM_VERIFY(out,"Export write failed.");
  std::cout<<"[igr-local-operator] n="<<n<<" ne="<<ne<<" p="<<p<<" source_p="<<source_p
    <<" nnz="<<values.size()<<" graph_error="<<graph_error<<" total_host_seconds="<<Seconds(start)<<'\n';
  return 0;
}

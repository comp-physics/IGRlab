#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_tensor_volume.hpp"
#include "mfem/general/forall.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

namespace
{
using namespace mfem;
using ::igrlab::igr_internal::MakeTensorVolumeOperator;
using ::igrlab::igr_internal::RefreshTensorVolumeCoefficients;

double Relative(const Vector& actual, const Vector& expected)
{
  const double* a=actual.HostRead(); const double* b=expected.HostRead();
  long double error=0.0L, norm=0.0L;
  for (int i=0;i<actual.Size();++i)
  {
    MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite tensor-volume comparison.");
    const long double delta=static_cast<long double>(a[i])-b[i];
    error+=delta*delta; norm+=static_cast<long double>(b[i])*b[i];
  }
  MFEM_VERIFY(norm>0.0L, "Tensor-volume oracle must be nonzero.");
  return std::sqrt(static_cast<double>(error/norm));
}

// Updates only the device copy: a stale coefficient snapshot must fail phases
// 1--3. The two independent QF strides deliberately need not be equal.
void FillCoefficients(QuadratureFunction& mass, QuadratureFunction& diffusion, int phase)
{
  double* m=mass.Write(); double* c=diffusion.Write();
  mfem::forall(mass.Size(), [=] MFEM_HOST_DEVICE(int i)
  {
    m[i]=phase==1 ? 0.0 : 1.2+0.13*sin(0.31*i+0.7*phase);
  });
  mfem::forall(diffusion.Size(), [=] MFEM_HOST_DEVICE(int i)
  {
    c[i]=phase==0 ? 0.0 : 0.03+0.01*cos(0.21*i-0.8*phase);
  });
}

void DenseOracle(ParFiniteElementSpace& fes, QuadratureFunction& mass,
                 QuadratureFunction& diffusion, const Vector& input, Vector& output)
{
  QuadratureFunctionCoefficient mc(mass), dc(diffusion);
  MassIntegrator mi(mc); DiffusionIntegrator di(dc);
  mi.SetIntRule(&mass.GetSpace()->GetIntRule(0));
  di.SetIntRule(&diffusion.GetSpace()->GetIntRule(0));
  const double* x=input.HostRead(); output.SetSize(input.Size());
  double* y=output.HostWrite();
  for (int e=0;e<fes.GetNE();++e)
  {
    Array<int> dofs; fes.GetElementDofs(e,dofs);
    DenseMatrix m,k;
    mi.AssembleElementMatrix(*fes.GetFE(e),*fes.GetElementTransformation(e),m);
    di.AssembleElementMatrix(*fes.GetFE(e),*fes.GetElementTransformation(e),k);
    for (int i=0;i<dofs.Size();++i)
    {
      long double sum=0.0L;
      for (int j=0;j<dofs.Size();++j)
      {
        const int index=dofs[j]>=0?dofs[j]:-1-dofs[j];
        const double value=dofs[j]>=0?x[index]:-x[index];
        sum+=(static_cast<long double>(m(i,j))+k(i,j))*value;
      }
      const int index=dofs[i]>=0?dofs[i]:-1-dofs[i];
      y[index]=static_cast<double>(dofs[i]>=0?sum:-sum);
    }
  }
}

void CheckCase(int order, int basis, bool curved, bool unequal)
{
  // Nine elements exercises the final partial four-warp CTA. Rotating one
  // element also prevents accidental reliance on an axis-aligned Jacobian.
  Mesh serial=Mesh::MakeCartesian2D(3,3,Element::QUADRILATERAL,false,1.0,1.0,false);
  if (curved)
  {
    int* vertices=serial.GetElement(0)->GetVertices();
    const int first=vertices[0];
    vertices[0]=vertices[1];vertices[1]=vertices[2];vertices[2]=vertices[3];vertices[3]=first;
    serial.FinalizeTopology(); serial.Finalize(false,false);
    serial.SetCurvature(2,false,2,Ordering::byVDIM);
    serial.Transform([](const Vector& x,Vector& y)
    {
      y.SetSize(2); y[0]=x[0]+0.08*x[0]*x[1];
      y[1]=x[1]+0.06*x[0]*(1.0-x[0]);
    });
  }
  ParMesh mesh(MPI_COMM_SELF,serial);
  L2_FECollection fec(order,2,basis);
  ParFiniteElementSpace fes(&mesh,&fec);
  const auto& fe=*fes.GetFE(0);
  // Same default rule selection as IGRSolver::EnsureDeviceQuadrature_. The
  // explicit case exercises Qmass=7, Qdiffusion=8 independently of p5 D=6.
  const auto& mass_rule=unequal ? IntRules.Get(Geometry::SQUARE,13) :
    MassIntegrator::GetRule(fe,fe,*mesh.GetTypicalElementTransformation());
  const auto& diff_rule=unequal ? IntRules.Get(Geometry::SQUARE,15) :
    DiffusionIntegrator::GetRule(fe,fe);
  QuadratureSpace mass_space(mesh,mass_rule), diffusion_space(mesh,diff_rule);
  QuadratureFunction mass(mass_space),diffusion(diffusion_space);
  FillCoefficients(mass,diffusion,0);
  auto simt=MakeTensorVolumeOperator(fes,mass,diffusion,"simt");
  auto wmma=MakeTensorVolumeOperator(fes,mass,diffusion,"wmma");
  Vector x(fes.GetVSize()), ys(fes.GetVSize()), yw(fes.GetVSize()), yp(fes.GetVSize()), yd;
  ys.UseDevice(true);yw.UseDevice(true);
  const auto* simt_address=ys.Write();const auto* wmma_address=yw.Write();
  double worst_pa=0.0,worst_dense=0.0,worst_modes=0.0;
  for (int phase=0;phase<4;++phase)
  {
    FillCoefficients(mass,diffusion,phase);
    if (phase) { RefreshTensorVolumeCoefficients(*simt,mass,diffusion); RefreshTensorVolumeCoefficients(*wmma,mass,diffusion); }
    double* values=x.HostWrite();
    for(int i=0;i<x.Size();++i) { values[i]=phase==3?1.0:0.8+0.2*sin(0.37*i+phase); }
    QuadratureFunctionCoefficient mc(mass),dc(diffusion);
    ParBilinearForm original(&fes); original.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    auto* mi=new MassIntegrator(mc);mi->SetIntRule(&mass_rule);original.AddDomainIntegrator(mi);
    auto* di=new DiffusionIntegrator(dc);di->SetIntRule(&diff_rule);original.AddDomainIntegrator(di);
    original.Assemble(); original.Mult(x,yp);
    simt->Mult(x,ys);wmma->Mult(x,yw);
    DenseOracle(fes,mass,diffusion,x,yd);
    const double pa=std::max(Relative(ys,yp),Relative(yw,yp));
    const double dense=std::max(Relative(ys,yd),Relative(yw,yd));
    const double modes=Relative(yw,ys);
    worst_pa=std::max(worst_pa,pa);worst_dense=std::max(worst_dense,dense);worst_modes=std::max(worst_modes,modes);
    MFEM_VERIFY(pa<=2e-12 && dense<=2e-12 && modes<=2e-12,
                "IGR tensor volume parity failed: p="<<order<<" phase="<<phase
                <<" curved="<<curved<<" unequal="<<unequal<<" pa="<<pa<<" dense="<<dense<<" modes="<<modes);
    MFEM_VERIFY(simt_address==ys.Read() && wmma_address==yw.Read(), "IGR tensor Mult resized preallocated outputs.");
  }
  // Native current-A authority must see a bad coefficient; neither mode may
  // hide it by reusing a previously valid coefficient snapshot.
  diffusion.HostReadWrite()[0]=std::numeric_limits<double>::quiet_NaN();
  for (Operator* op:{simt.get(),wmma.get()})
  {
    RefreshTensorVolumeCoefficients(*op,mass,diffusion);op->Mult(x,yw);
    const double* values=yw.HostRead();bool nonfinite=false;
    for(int i=0;i<yw.Size();++i) { nonfinite=nonfinite||!std::isfinite(values[i]); }
    MFEM_VERIFY(nonfinite,"IGR tensor volume hid a nonfinite current coefficient.");
  }
  std::cout<<"igr_tensor_volume_case p="<<order<<" basis="<<basis<<" curved="<<curved
           <<" mass_nq="<<mass_rule.GetNPoints()<<" diffusion_nq="<<diff_rule.GetNPoints()
           <<" max_pa_relative="<<worst_pa<<" max_dense_relative="<<worst_dense
           <<" max_simt_wmma_relative="<<worst_modes<<" passed\n";
}
}

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);mfem::Device device("cuda");
  for(int basis:{mfem::BasisType::GaussLegendre,mfem::BasisType::GaussLobatto})
  {
    for(int order:{1,3,5,7}) { CheckCase(order,basis,false,false); }
    CheckCase(5,basis,true,false);
    CheckCase(5,basis,false,true);
  }
  std::cout<<"igr_tensor_volume_test passed original_pa=true independent_dense=true "
              "current_coefficients=true separate_quadratures=true curved_geometry=true "
              "real_columns=true partial_warp_batch=true fp64_simt=true fp64_wmma=true nonfinite_exposed=true\n";
  return 0;
}

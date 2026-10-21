#include "igrlab/compat.hpp"
#include "../src/physics/igr/IGR_tensor_volume.hpp"
#include "mfem/general/forall.hpp"
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
#include "mfem/general/cuda.hpp"
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

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

// The opt-in large benchmark uses the original mass and diffusion rules,
// independently indexed current QFs, and one fixed input across generations.
// Coefficient generation is outside every measured apply/refresh interval.
void FillBenchmarkCoefficients(QuadratureFunction& mass,
                               QuadratureFunction& diffusion, int phase)
{
  constexpr double alpha = 6.357828776041666e-7;
  double* m = mass.Write();
  double* c = diffusion.Write();
  mfem::forall(mass.Size(), [=] MFEM_HOST_DEVICE(int i)
  {
    const double rho = 0.9 + 0.08*sin(0.031*(i+1) + 0.37*phase) +
      0.03*cos(0.017*(i+1) - 0.23*phase);
    m[i] = 1.0/rho;
  });
  mfem::forall(diffusion.Size(), [=] MFEM_HOST_DEVICE(int i)
  {
    const double rho = 0.9 + 0.08*sin(0.031*(i+1) + 0.37*phase) +
      0.03*cos(0.017*(i+1) - 0.23*phase);
    c[i] = alpha/rho;
  });
}

struct BenchmarkError
{
  double l2 = 0.0, linf = 0.0;
};

BenchmarkError CheckBenchmarkParity(const Vector& actual, const Vector& reference)
{
  MFEM_VERIFY(actual.Size() == reference.Size(), "Large volume output size changed.");
  const double* a = actual.HostRead();
  const double* b = reference.HostRead();
  long double error = 0.0L, norm = 0.0L;
  double max_error = 0.0, max_reference = 0.0;
  for (int i = 0; i < actual.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]),
                "Large volume apply did not produce finite, complete output.");
    const long double delta = static_cast<long double>(a[i])-b[i];
    error += delta*delta;
    norm += static_cast<long double>(b[i])*b[i];
    max_error = std::max(max_error, static_cast<double>(std::abs(delta)));
    max_reference = std::max(max_reference, std::abs(b[i]));
  }
  MFEM_VERIFY(norm > 0.0L && max_reference > 0.0,
              "Large volume PA reference must have a nonzero action.");
  const BenchmarkError result{std::sqrt(static_cast<double>(error/norm)),
                              max_error/max_reference};
  MFEM_VERIFY(std::isfinite(result.l2) && std::isfinite(result.linf) &&
                result.l2 <= 2e-12 && result.linf <= 2e-12,
              "Large volume parity exceeds the existing FP64 budget: relative_l2="
                << result.l2 << " relative_linf=" << result.linf);
  return result;
}

void Benchmark512(const std::string& mesh_path)
{
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  constexpr int n = 512, order = 5, warmups = 4, repeats = 64;
  const unsigned long alternatives = Backend::OMP_MASK | Backend::RAJA_MASK |
    Backend::OCCA_MASK | Backend::CEED_MASK | Backend::HIP_MASK | Backend::DEBUG_DEVICE;
  MFEM_VERIFY(Mpi::WorldSize() == 1 && Device::Allows(Backend::CUDA) &&
                !Device::Allows(alternatives),
              "The large volume benchmark requires one rank with ordinary native CUDA.");
  // Match the application's -rs 0 -rp 4 mesh path and element ordering.
  Mesh serial(mesh_path);
  MFEM_VERIFY(serial.Dimension() == 2 && serial.SpaceDimension() == 2 &&
                serial.GetNE() == 32*32,
              "The large volume benchmark requires the original 32-square Sedov mesh.");
  ParMesh mesh(MPI_COMM_SELF, serial);
  for (int level = 0; level < 4; ++level) { mesh.UniformRefinement(); }
  MFEM_VERIFY(mesh.GetNE() == n*n, "Four parallel refinements must produce 512-square cells.");
  double lower[2] = {1.0, 1.0}, upper[2] = {0.0, 0.0};
  for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
  {
    const double* point = mesh.GetVertex(vertex);
    for (int axis = 0; axis < 2; ++axis)
    {
      lower[axis] = std::min(lower[axis], point[axis]);
      upper[axis] = std::max(upper[axis], point[axis]);
    }
  }
  MFEM_VERIFY(std::abs(lower[0]) < 1e-13 && std::abs(lower[1]) < 1e-13 &&
                std::abs(upper[0]-1.0) < 1e-13 && std::abs(upper[1]-1.0) < 1e-13,
              "The large volume benchmark requires the production unit-square domain.");
  L2_FECollection fec(order, 2, BasisType::GaussLegendre);
  ParFiniteElementSpace fes(&mesh, &fec, 1, Ordering::byNODES);
  MFEM_VERIFY(fes.GetVSize() == 9437184 && fes.GetTrueVSize() == fes.GetVSize() &&
                fes.GetFE(0)->GetDof() == 36,
              "The fixed p5/GL Sedov benchmark must have 9,437,184 scalar DOFs.");
  const auto& fe = *fes.GetFE(0);
  const auto& mass_rule = MassIntegrator::GetRule(fe, fe, *mesh.GetTypicalElementTransformation());
  const auto& diffusion_rule = DiffusionIntegrator::GetRule(fe, fe);
  const int qm = static_cast<int>(std::lround(std::sqrt(double(mass_rule.GetNPoints()))));
  const int qd = static_cast<int>(std::lround(std::sqrt(double(diffusion_rule.GetNPoints()))));
  std::cout << std::setprecision(17)
            << "IGRTensorVolumeBenchmark512 event=configuration cells=" << mesh.GetNE()
            << " scalar_dofs=" << fes.GetVSize() << " order=" << order << " d1d=6"
            << " mass_q1d=" << qm << " diffusion_q1d=" << qd
            << " mass_nq=" << mass_rule.GetNPoints() << " diffusion_nq=" << diffusion_rule.GetNPoints()
            << " mass_rule_order=" << mass_rule.GetOrder()
            << " diffusion_rule_order=" << diffusion_rule.GetOrder()
            << " mesh_path=" << std::quoted(mesh_path)
            << " serial_refinements=0 parallel_refinements=4"
            << " coefficient_fixture=nonuniform-positive-device-updated"
            << " actual_basis_maps=true mass_basis_identity_assumed=false"
            << " baseline_route=public-ParBilinearForm-Mult restriction_scatter_included=true"
            << " warmups=" << warmups << " repeats=" << repeats << std::endl;
  MFEM_VERIFY(qm == 6 && qd == 6 && mass_rule.GetNPoints() == qm*qm &&
                diffusion_rule.GetNPoints() == qd*qd,
              "The fixed case must retain the actual separate default 6x6 mass/diffusion rules.");
  QuadratureSpace mass_space(mesh, mass_rule), diffusion_space(mesh, diffusion_rule);
  QuadratureFunction mass(mass_space), diffusion(diffusion_space);
  mass.UseDevice(true); diffusion.UseDevice(true);
  FillBenchmarkCoefficients(mass, diffusion, 0);
  Vector x(fes.GetVSize()), yp(fes.GetVSize()), ys(fes.GetVSize()), yw(fes.GetVSize());
  for (Vector* value : {&x, &yp, &ys, &yw}) { value->UseDevice(true); }
  double* input = x.HostWrite();
  for (int i = 0; i < x.Size(); ++i)
  { input[i] = 0.01*std::sin(0.013*(i+1)) + 0.002*(i%7); }
  const double* input_address = x.Read();
  const double* pa_address = yp.Write();
  const double* simt_address = ys.Write();
  const double* wmma_address = yw.Write();
  MFEM_VERIFY(x.OwnsData() && yp.OwnsData() && ys.OwnsData() && yw.OwnsData() &&
                input_address != pa_address && input_address != simt_address &&
                input_address != wmma_address && pa_address != simt_address &&
                pa_address != wmma_address && simt_address != wmma_address,
              "Large volume input and outputs must be distinct owning device vectors.");

  using Clock = std::chrono::steady_clock;
  auto setup = [&](auto construct)
  {
    MFEM_GPU_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    construct();
    MFEM_GPU_CHECK(cudaDeviceSynchronize());
    return std::chrono::duration<double>(Clock::now()-start).count();
  };
  QuadratureFunctionCoefficient mc(mass), dc(diffusion);
  ParBilinearForm pa(&fes);
  const double pa_setup = setup([&]()
  {
    pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    auto* mi = new MassIntegrator(mc); mi->SetIntRule(&mass_rule); pa.AddDomainIntegrator(mi);
    auto* di = new DiffusionIntegrator(dc); di->SetIntRule(&diffusion_rule); pa.AddDomainIntegrator(di);
    pa.Assemble();
  });
  std::unique_ptr<Operator> simt, wmma;
  const double simt_setup = setup([&]() { simt = MakeTensorVolumeOperator(fes, mass, diffusion, "simt"); });
  const double wmma_setup = setup([&]() { wmma = MakeTensorVolumeOperator(fes, mass, diffusion, "wmma"); });
  // These are source-derived persistent array footprints for one fused mode,
  // not allocation-inventory measurements or whole-process memory savings.
  const std::size_t geometry_bytes = std::size_t(mesh.GetNE())*4*64*sizeof(double);
  const std::size_t current_bytes = geometry_bytes;
  const std::size_t map_bytes = std::size_t(fes.GetVSize())*sizeof(int);
  const std::size_t basis_bytes = 6u*64u*sizeof(double);
  std::size_t free_bytes = 0, total_bytes = 0;
  MFEM_GPU_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::cout << "IGRTensorVolumeBenchmark512 event=setup pa_seconds=" << pa_setup
            << " simt_seconds=" << simt_setup << " wmma_seconds=" << wmma_setup
            << " setup_scope=operator-setup-and-initial-coefficients first_apply_included=false"
            << " mesh_and_vector_setup_included=false"
            << " per_fused_geometry_bytes=" << geometry_bytes
            << " per_fused_current_coefficient_bytes=" << current_bytes
            << " per_fused_dof_map_bytes=" << map_bytes << " per_fused_basis_map_bytes=" << basis_bytes
            << " per_fused_array_bytes=" << geometry_bytes+current_bytes+map_bytes+basis_bytes
            << " array_bytes_scope=source-derived-persistent-device-arrays"
            << " original_pa_retained=true fused_instances_live=2"
            << " cuda_free_bytes=" << free_bytes << " cuda_total_bytes=" << total_bytes
            << " process_memory_saving_claim=false" << std::endl;

  cudaEvent_t start{}, stop{};
  MFEM_GPU_CHECK(cudaEventCreate(&start)); MFEM_GPU_CHECK(cudaEventCreate(&stop));
  auto measure = [&](auto launch)
  {
    for (int i = 0; i < warmups; ++i) { launch(); }
    MFEM_GPU_CHECK(cudaGetLastError()); MFEM_GPU_CHECK(cudaDeviceSynchronize());
    MFEM_GPU_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) { launch(); }
    MFEM_GPU_CHECK(cudaEventRecord(stop)); MFEM_GPU_CHECK(cudaEventSynchronize(stop));
    MFEM_GPU_CHECK(cudaGetLastError());
    float elapsed_ms = 0.0f;
    MFEM_GPU_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double result = double(elapsed_ms)/repeats;
    MFEM_VERIFY(std::isfinite(result) && result > 0.0,
                "Large volume CUDA-event timing must be finite and positive.");
    return result;
  };
  Vector previous;
  for (int phase = 0; phase < 2; ++phase)
  {
    if (phase)
    {
      FillBenchmarkCoefficients(mass, diffusion, phase);
      pa.Update(); pa.Assemble();
      RefreshTensorVolumeCoefficients(*simt, mass, diffusion);
      RefreshTensorVolumeCoefficients(*wmma, mass, diffusion);
    }
    yp = std::numeric_limits<double>::quiet_NaN();
    ys = std::numeric_limits<double>::quiet_NaN();
    yw = std::numeric_limits<double>::quiet_NaN();
    pa.Mult(x, yp); simt->Mult(x, ys); wmma->Mult(x, yw);
    const BenchmarkError simt_error = CheckBenchmarkParity(ys, yp);
    const BenchmarkError wmma_error = CheckBenchmarkParity(yw, yp);
    if (phase)
    {
      const double* now = yp.HostRead(); const double* old = previous.HostRead();
      bool changed = false;
      for (int i = 0; i < yp.Size(); ++i) { changed = changed || now[i] != old[i]; }
      MFEM_VERIFY(changed, "Large volume apply must consume the changed coefficient generation.");
    }
    else
    {
      previous.SetSize(yp.Size());
      std::copy(yp.HostRead(), yp.HostRead()+yp.Size(), previous.HostWrite());
    }
    const double pa_apply_ms = measure([&]() { pa.Mult(x, yp); });
    const double simt_apply_ms = measure([&]() { simt->Mult(x, ys); });
    const double wmma_apply_ms = measure([&]() { wmma->Mult(x, yw); });
    // Production continues to assemble original PA data for its independent
    // current-equation certificates. Fused Refresh is additional work.
    const double pa_refresh_ms = measure([&]() { pa.Update(); pa.Assemble(); });
    const double simt_refresh_ms = measure([&]() { RefreshTensorVolumeCoefficients(*simt, mass, diffusion); });
    const double wmma_refresh_ms = measure([&]() { RefreshTensorVolumeCoefficients(*wmma, mass, diffusion); });
    pa.Mult(x, yp); simt->Mult(x, ys); wmma->Mult(x, yw);
    const BenchmarkError timed_simt = CheckBenchmarkParity(ys, yp);
    const BenchmarkError timed_wmma = CheckBenchmarkParity(yw, yp);
    MFEM_VERIFY(input_address == x.Read() && pa_address == yp.Read() &&
                  simt_address == ys.Read() && wmma_address == yw.Read(),
                "Measured public Mult must reuse the preallocated input/output buffers.");
    auto report = [&](const char* variant, double apply_ms, double refresh_ms,
                      const BenchmarkError& initial, const BenchmarkError& timed)
    {
      std::cout << "IGRTensorVolumeBenchmark512 variant=" << variant << " phase=" << phase
                << " parity=passed coefficient_refresh_parity=passed cells=" << mesh.GetNE()
                << " scalar_dofs=" << fes.GetVSize() << " d1d=6 mass_q1d=" << qm << " diffusion_q1d=" << qd
                << " warmups=" << warmups << " repeats=" << repeats
                << " mean_apply_ms=" << apply_ms << " apply_speedup_vs_pa=" << pa_apply_ms/apply_ms
                << " mean_refresh_ms=" << refresh_ms
                << " mean_common_pa_refresh_ms=" << pa_refresh_ms
                << " refresh_scope=" << (std::string(variant) == "pa" ? "original-pa-common" : "additional-fused-only")
                << " original_pa_refresh_retained=true"
                << " relative_l2=" << std::max(initial.l2, timed.l2)
                << " relative_linf=" << std::max(initial.linf, timed.linf)
                << " apply_includes_public_mult=true coefficient_generation_timed=false"
                << " timing_clock=cuda-events refresh_includes_host_submission_gaps=true"
                << " setup_timed_separately=true oracle_downloads_timed=false"
                << " faces_included=false solves_included=false" << std::endl;
    };
    report("pa", pa_apply_ms, pa_refresh_ms, {}, {});
    report("simt", simt_apply_ms, simt_refresh_ms, simt_error, timed_simt);
    report("wmma", wmma_apply_ms, wmma_refresh_ms, wmma_error, timed_wmma);
  }
  MFEM_GPU_CHECK(cudaEventDestroy(start)); MFEM_GPU_CHECK(cudaEventDestroy(stop));
  std::cout << "IGRTensorVolumeBenchmark512 parity=passed coefficient_generations=2"
            << " variants=3 large_oracle=public-pa normal_suite_executed=false"
            << " baseline_route=public-ParBilinearForm-Mult restriction_scatter_included=true"
            << " full_dense_assembly=false full_solver_speedup_claim=false" << std::endl;
#else
  (void)mesh_path;
  MFEM_ABORT("The large volume benchmark must be compiled as native CUDA.");
#endif
}
}

int main(int argc,char** argv)
{
  mfem::Mpi::Init(argc,argv);
  const bool benchmark = argc == 3 && std::string(argv[1]) == "--benchmark512";
  MFEM_VERIFY(argc == 1 || benchmark, "Expected no arguments or --benchmark512 <Sedov mesh>.");
  mfem::Device device("cuda");
  if (benchmark) { Benchmark512(argv[2]); return 0; }
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

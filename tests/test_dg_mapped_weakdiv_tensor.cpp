#include "igrlab/environment.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/igr/igr_flux.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

// The production CUDA operator is linked, but the MFEM assembly and analytic
// nodal-flux oracle in this translation unit must remain ordinary host CXX.
#ifdef __CUDACC__
#error "Mapped weak-divergence oracle must be compiled as CXX."
#endif

namespace
{
using mfem::Vector;
using mfem::DenseMatrix;
using dg_igr_tests::TestContext;

class Environment
{
  std::string name_;
  std::optional<std::string> old_;
public:
  Environment(const char* name, const char* value) : name_(name)
  {
    if (const char* old = std::getenv(name)) { old_ = old; }
    if (value) { setenv(name, value, 1); }
    else { unsetenv(name); }
  }
  ~Environment()
  {
    if (old_) { setenv(name_.c_str(), old_->c_str(), 1); }
    else { unsetenv(name_.c_str()); }
  }
};

struct Capture
{
  std::ostringstream text;
  std::streambuf* previous = std::cout.rdbuf(text.rdbuf());
  ~Capture() { std::cout.rdbuf(previous); }
};

bool Record(const std::string& text, const std::string& prefix, const std::string& token)
{
  const auto begin = text.find(prefix);
  if (begin == std::string::npos) { return false; }
  return text.substr(begin, text.find('\n', begin) - begin).find(token) != std::string::npos;
}

void Near(TestContext& ctx, const Vector& actual, const Vector& expected, const std::string& label)
{
  double difference = 0.0, scale = 1.0;
  bool finite = actual.Size() == expected.Size();
  const auto* a = actual.HostRead(); const auto* b = expected.HostRead();
  for (int i = 0; finite && i < actual.Size(); ++i)
  {
    finite = std::isfinite(a[i]) && std::isfinite(b[i]);
    difference = std::max(difference, std::abs(a[i] - b[i]));
    scale = std::max(scale, std::abs(b[i]));
  }
  ctx.Check(finite && difference <= 2.e-11 * scale,
            label + " independent full-RHS error=" + dg_igr_tests::FormatSci(difference));
}

mfem::Mesh Mesh(int geometry_order)
{
  auto box = mfem::Mesh::MakeCartesian2D(3, 3, mfem::Element::QUADRILATERAL,
                                        false, 1.0, 1.0, false);
  const std::vector<Vector> shifts{Vector({1.0, 0.0}), Vector({0.0, 1.0})};
  auto mesh = mfem::Mesh::MakePeriodic(box, box.CreatePeriodicVertexMapping(shifts));
  if (geometry_order > 1)
  {
    mesh.SetCurvature(geometry_order, true, 2, mfem::Ordering::byVDIM);
    mesh.Transform([](const Vector& x, Vector& y)
    {
      const double warp = std::sin(2 * std::acos(-1.0) * x[0]) *
                          std::sin(2 * std::acos(-1.0) * x[1]);
      y.SetSize(2); y[0] = x[0] + .035 * warp; y[1] = x[1] + .025 * warp;
    });
  }
  return mesh;
}

void Fill(mfem::ParFiniteElementSpace& fes, Vector& state,
          mfem::ParGridFunction& sigma, int phase)
{
  auto* values = state.HostWrite();
  mfem::Array<int> dofs;
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementDofs(e, dofs);
    const auto& nodes = fes.GetFE(e)->GetNodes();
    for (int j = 0; j < dofs.Size(); ++j)
    {
      const auto& ip = nodes.IntPoint(j);
      const double rho = phase == 0 ? 1.1 : 1.0 + .02*e + .09*std::sin(3.1*ip.x+1.3*ip.y);
      const double vx = phase == 0 ? .21 : .21 + .08*ip.x - .003*e;
      const double vy = phase == 0 ? -.13 : -.13 + .09*ip.y + .004*e;
      const double pressure = phase == 0 ? 1.2 : 1.2 + .2*ip.x*ip.y + .03*e;
      const double u[] = {rho, rho*vx, rho*vy, pressure/(1.4-1.0) + .5*rho*(vx*vx+vy*vy)};
      for (int c = 0; c < 4; ++c) { values[fes.DofToVDof(dofs[j], c)] = u[c]; }
    }
  }
  auto& scalar = *sigma.ParFESpace();
  auto* s = sigma.HostWrite();
  for (int e = 0; e < scalar.GetNE(); ++e)
  {
    scalar.GetElementDofs(e, dofs);
    const auto& nodes = scalar.GetFE(e)->GetNodes();
    for (int j = 0; j < dofs.Size(); ++j)
    {
      const auto& ip = nodes.IntPoint(j);
      s[dofs[j]] = phase == 0 ? .035 : (phase == 3 ? 1.7 : 1.0) *
        (.031 + .017*e + .07*ip.x - .04*ip.y*ip.y);
    }
  }
}

class Oracle
{
  mfem::ParFiniteElementSpace& fes_;
  mfem::HyperbolicFormIntegrator integrator_;
  mfem::ParNonlinearForm faces_;
  std::vector<DenseMatrix> weak_, inverse_;
public:
  std::vector<Vector> cell_mass;
  Oracle(mfem::ParFiniteElementSpace& fes, const mfem::NumericalFlux& flux,
         int face_offset = 2)
    : fes_(fes), integrator_(flux, face_offset), faces_(&fes),
      weak_(fes.GetNE()), inverse_(fes.GetNE()), cell_mass(fes.GetNE())
  {
    faces_.AddInteriorFaceIntegrator(&integrator_); faces_.UseExternalIntegrators();
    mfem::TransposeIntegrator weak(new mfem::GradientIntegrator());
    mfem::InverseIntegrator inverse(new mfem::MassIntegrator());
    mfem::MassIntegrator mass;
    for (int e = 0; e < fes.GetNE(); ++e)
    {
      const auto& fe = *fes.GetFE(e); auto& tr = *fes.GetElementTransformation(e);
      weak.AssembleElementMatrix2(fe, fe, tr, weak_[e]);
      inverse.AssembleElementMatrix(fe, tr, inverse_[e]);
      DenseMatrix m; mass.AssembleElementMatrix(fe, tr, m);
      Vector one(fe.GetDof()); one = 1.0; cell_mass[e].SetSize(fe.GetDof()); m.Mult(one, cell_mass[e]);
    }
  }
  void Apply(const Vector& state, const mfem::ParGridFunction* sigma, Vector& rhs)
  {
    // Only MFEM's original host face assembly is shared with the face policy.
    // Volume fluxes below never call production packed helpers or ComputeFlux.
    Vector face(fes_.GetVSize()); faces_.Mult(state, face);
    const auto* input = state.HostRead(); const auto* boundary = face.HostRead();
    auto* output = rhs.HostWrite(); mfem::Array<int> dofs;
    for (int e = 0; e < fes_.GetNE(); ++e)
    {
      const auto& fe = *fes_.GetFE(e); auto& tr = *fes_.GetElementTransformation(e);
      fes_.GetElementDofs(e, dofs); const int n = fe.GetDof();
      DenseMatrix nodal(2*n, 4), residual(n, 4);
      for (int j = 0; j < n; ++j)
      {
        double u[4]; for (int c = 0; c < 4; ++c) { u[c] = input[fes_.DofToVDof(dofs[j], c)]; }
        const auto& ip = fe.GetNodes().IntPoint(j); tr.SetIntPoint(&ip);
        const double s = sigma ? sigma->GetValue(tr, ip) : 0.0;
        const double p = (1.4-1.0)*(u[3] - .5*(u[1]*u[1]+u[2]*u[2])/u[0]);
        nodal(j, 0) = u[1]; nodal(n+j, 0) = u[2];
        nodal(j, 1) = u[1]*u[1]/u[0]+p+s; nodal(n+j, 1) = u[1]*u[2]/u[0];
        nodal(j, 2) = u[2]*u[1]/u[0]; nodal(n+j, 2) = u[2]*u[2]/u[0]+p+s;
        nodal(j, 3) = (u[3]+p+s)*u[1]/u[0]; nodal(n+j, 3) = (u[3]+p+s)*u[2]/u[0];
      }
      for (int c = 0; c < 4; ++c) for (int i = 0; i < n; ++i)
      {
        long double value = boundary[fes_.DofToVDof(dofs[i], c)];
        for (int j = 0; j < 2*n; ++j) { value += static_cast<long double>(weak_[e](i, j))*nodal(j, c); }
        residual(i, c) = static_cast<double>(value);
      }
      for (int c = 0; c < 4; ++c) for (int i = 0; i < n; ++i)
      {
        long double value = 0.0;
        for (int j = 0; j < n; ++j) { value += static_cast<long double>(inverse_[e](i, j))*residual(j, c); }
        output[fes_.DofToVDof(dofs[i], c)] = static_cast<double>(value);
      }
    }
  }
};

void Conservation(TestContext& ctx, mfem::ParFiniteElementSpace& fes,
                  const std::vector<Vector>& mass, const Vector& rhs,
                  const std::string& label)
{
  std::array<long double, 4> total{}, bound{};
  const auto* data = rhs.HostRead(); mfem::Array<int> dofs;
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementDofs(e, dofs);
    for (int c = 0; c < 4; ++c) for (int j = 0; j < dofs.Size(); ++j)
    {
      const long double term = static_cast<long double>(mass[e][j]) * data[fes.DofToVDof(dofs[j], c)];
      total[c] += term; bound[c] += std::abs(term);
    }
  }
  bool conserved = true;
  for (int c = 0; c < 4; ++c)
    conserved = conserved && std::isfinite(total[c]) && std::abs(total[c]) <= 5.e-11L * std::max(1.0L, bound[c]);
  ctx.Check(conserved, label+" conserves periodic physical mass-weighted moments");
}

std::unique_ptr<mfem::DGHyperbolicConservationLaws> Make(
  mfem::ParFiniteElementSpace& fes, const mfem::NumericalFlux& flux,
  bool tensor, std::string& log, const char* dense = "rows", bool alias = false,
  int face_offset = 2)
{
  Environment canonical("IGRLAB_DG_MAPPED_WEAKDIV_TENSOR", alias ? nullptr : tensor ? "1" : "0");
  Environment legacy("DG_IGR_DG_MAPPED_WEAKDIV_TENSOR", alias && tensor ? "1" : nullptr);
  Environment execution("IGRLAB_DG_PACKED_DENSE_EXECUTION", dense);
  Capture capture;
  auto op = std::make_unique<mfem::DGHyperbolicConservationLaws>(fes, flux, face_offset, true);
  op->SetDGSEMGamma(1.4); op->SetDGSEMDiagnosticStateScan(false);
  log = capture.text.str(); return op;
}

void Apply(mfem::DGHyperbolicConservationLaws& op, const Vector& state, Vector& rhs,
           bool tensor, const char* dense = "rows")
{
  Environment selected("IGRLAB_DG_MAPPED_WEAKDIV_TENSOR", tensor ? "1" : "0");
  Environment execution("IGRLAB_DG_PACKED_DENSE_EXECUTION", dense);
  rhs = std::numeric_limits<double>::quiet_NaN(); op.Mult(state, rhs);
}

void Cohort(TestContext& ctx, int order, int geometry_order, int ordering,
            bool interpolated_sigma, bool supplementary, bool combined_faces = false)
{
  Environment face_mode("IGRLAB_DG_TENSOR_FACE", combined_faces ? "fp64" : "off");
  Environment face_reuse("IGRLAB_DG_TENSOR_FACE_ROW_REUSE", combined_faces ? "1" : "0");
  Environment mass_reuse("IGRLAB_DG_INVMASS_ROW_REUSE", combined_faces ? "1" : "0");
  auto serial = Mesh(geometry_order); mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  mfem::L2_FECollection fec(order, 2, mfem::BasisType::GaussLegendre);
  mfem::L2_FECollection sfec(interpolated_sigma ? 1 : order, 2,
    interpolated_sigma ? mfem::BasisType::GaussLobatto : mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, 4, ordering), sfes(&mesh, &sfec);
  mfem::ParGridFunction sigma(&sfes); mfem::EulerFlux euler(2, 1.4);
  mfem::FluxWithIGR physical(euler, 1.4); physical.SetSigmaField(&sigma);
  physical.SetInvariantDomainFlux(false); physical.SetSigmaAwareCFL(false); physical.SetSigmaAwareRusanov(false);
  mfem::RusanovFluxWithSides flux(physical);
  std::string old_log, tensor_log;
  const int face_offset = combined_faces ? 1 : 2;
  auto old = Make(fes, flux, false, old_log, "rows", false, face_offset);
  auto tensor = Make(fes, flux, true, tensor_log, "rows", false, face_offset);
  const bool admitted = mfem::Device::Allows(mfem::Backend::CUDA_MASK) && geometry_order <= 2;
  ctx.Check(Record(tensor_log, "[dg-mapped-weakdiv] event=setup", admitted ? "selected=1" : "selected=0"),
            "mapped tensor setup admission/fallback is explicit");
  if (admitted)
  {
    ctx.Check(Record(tensor_log, "[dg-mapped-weakdiv] event=setup", "dense_matrix_entries=0"),
              "admitted mapped tensor owns no dense weak-divergence matrices");
    if (order == 5 && geometry_order == 2)
      ctx.Check(Record(tensor_log, "[dg-mapped-weakdiv] event=setup", "q1d=7"),
                "Q2 p5 retains the original GL7 weak quadrature");
  }
  Oracle oracle(fes, flux, face_offset);
  Vector state(fes.GetVSize()), expected(fes.GetVSize()), baseline(fes.GetVSize()), candidate(fes.GetVSize());
  baseline.UseDevice(true); candidate.UseDevice(true);
  const std::string label = "p="+std::to_string(order)+" Q="+std::to_string(geometry_order)+" ordering="+std::to_string(ordering);
  for (int phase = 0; phase < 4; ++phase)
  {
    Fill(fes, state, sigma, phase); physical.SetSigmaField(phase == 1 ? nullptr : &sigma);
    Apply(*old, state, baseline, false);
    {
      Capture capture;
      Apply(*tensor, state, candidate, true);
      tensor_log += capture.text.str();
    }
    if (admitted)
    {
      ctx.Check(Record(tensor_log, "[dg-device] event=face-domain-path", "selected=packed-mfem-forall-euler-rusanov"),
                label+" mapped tensor storage retains actual packed face dispatch");
    }
    if (combined_faces && admitted)
    {
      ctx.Check(Record(tensor_log, "[dg-tensor-face] event=setup", "selected=fp64") &&
                Record(tensor_log, "[dg-tensor-face] event=setup", "row_reuse=1"),
                label+" tensor volume composes with actual reused tensor faces");
      ctx.Check(Record(tensor_log, "[dg-device] event=weakdiv-volume-path", "dense_execution=tensor-mapped"),
                label+" combined path applies mapped tensor volume");
    }
    oracle.Apply(state, physical.GetSigmaField(), expected);
    Near(ctx, baseline, expected, label+" dense oracle phase="+std::to_string(phase));
    Near(ctx, candidate, expected, label+" tensor oracle phase="+std::to_string(phase));
    Conservation(ctx, fes, oracle.cell_mass, candidate, label);
    ctx.Check(std::isfinite(old->GetMaxCharSpeed()) && std::isfinite(tensor->GetMaxCharSpeed()) &&
                old->GetMaxCharSpeed() == tensor->GetMaxCharSpeed(), label+" preserves face/CFL characteristic speeds");
  }
  if (!supplementary) { return; }
  Vector alias(state); alias.UseDevice(true);
  {
    Environment selected("IGRLAB_DG_MAPPED_WEAKDIV_TENSOR", "1");
    tensor->Mult(alias, alias);
  }
  Near(ctx, alias, expected, label+" in-place state/output alias");
  auto fallback = Make(fes, flux, true, tensor_log, "element");
  ctx.Check(Record(tensor_log, "[dg-mapped-weakdiv] event=setup", "selected=0"),
            label+" explicit element request retains the original storage");
  Apply(*fallback, state, candidate, true, "element");
  Near(ctx, candidate, expected, label+" explicit element fallback");
  auto legacy = Make(fes, flux, true, tensor_log, "rows", true);
  ctx.Check(Record(tensor_log, "[dg-mapped-weakdiv] event=setup", admitted ? "selected=1" : "selected=0"),
            label+" legacy selector reaches actual tensor admission");
  Apply(*legacy, state, candidate, true); Near(ctx, candidate, expected, label+" legacy selector alias");
  // Rebuild both the matrix-free representation and the physical dense inverse
  // after node motion. No separable or collocated mass assumption is made.
  mesh.Transform([](const Vector& x, Vector& y)
  {
    y = x; y[0] += .002*std::sin(2*std::acos(-1.0)*x[0])*std::sin(2*std::acos(-1.0)*x[1]);
  });
  {
    Environment selected("IGRLAB_DG_MAPPED_WEAKDIV_TENSOR", "1");
    Capture capture; tensor->Update(); tensor_log = capture.text.str();
  }
  ctx.Check(Record(tensor_log, "[dg-mapped-weakdiv] event=setup", admitted ? "selected=1" : "selected=0"),
            label+" geometry update rebuilds the selected weak-divergence storage");
  Oracle moved(fes, flux, face_offset); moved.Apply(state, &sigma, expected);
  Apply(*tensor, state, candidate, true); Near(ctx, candidate, expected, label+" updated mapped geometry");
  {
    Environment selected("IGRLAB_DG_MAPPED_WEAKDIV_TENSOR", "0"); tensor->Update();
  }
  Apply(*tensor, state, candidate, false); Near(ctx, candidate, expected, label+" rebuilt dense fallback");
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv); mfem::Device device(argc > 1 ? argv[1] : "cpu");
  Environment precision("IGRLAB_SHARED_OPERATOR_PRECISION", "off");
  Environment layout("IGRLAB_SHARED_EULER_LAYOUT", "dense");
  Environment faces("IGRLAB_DG_TENSOR_FACE", "off");
  Environment row_reuse("IGRLAB_DG_WEAKDIV_ROW_REUSE", "1");
  Environment mass("IGRLAB_DG_INVMASS_ROW_REUSE", "0");
  Environment native_mass("IGRLAB_DG_NATIVE_MASS_INV", "0");
  Environment packed_mass("IGRLAB_DG_PACKED_INVMASS_DEVICE", "1");
  Environment sigma("IGRLAB_WEAKDIV_SIGMA_STORAGE", "auto");
  TestContext ctx;
  Cohort(ctx, 5, 2, mfem::Ordering::byNODES, false, true);
  Cohort(ctx, 2, 2, mfem::Ordering::byVDIM, true, false);
  Cohort(ctx, 1, 2, mfem::Ordering::byVDIM, false, false);
  Cohort(ctx, 7, 2, mfem::Ordering::byNODES, false, false);
  Cohort(ctx, 5, 1, mfem::Ordering::byVDIM, false, false);
  Cohort(ctx, 5, 3, mfem::Ordering::byNODES, false, false);
  Cohort(ctx, 5, 2, mfem::Ordering::byNODES, false, false, true);
  return ctx.failures == 0 ? 0 : 1;
}

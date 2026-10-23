#include "igrlab/environment.hpp"
#include "numerics/dg/BdrHyperbolicFluxIntegrator.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/riemann/rusanov.hpp"
#include "physics/igr/igr_flux.hpp"
#include "physics/bc/boundary_state.hpp"
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

namespace
{
using mfem::DenseMatrix;
using mfem::Vector;
using dg_igr_tests::TestContext;
constexpr int kDimension = 2;
constexpr int kEquations = 4;
constexpr double kGamma = 1.4;

class ScopedEnvironment
{
  std::string name_;
  std::optional<std::string> previous_;
public:
  ScopedEnvironment(const char* name, const char* value) : name_(name)
  {
    if (const char* previous = std::getenv(name)) { previous_ = previous; }
    if (value) { setenv(name, value, 1); }
    else { unsetenv(name); }
  }
  ~ScopedEnvironment()
  {
    if (previous_) { setenv(name_.c_str(), previous_->c_str(), 1); }
    else { unsetenv(name_.c_str()); }
  }
};

double Difference(const Vector& actual, const Vector& expected)
{
  const auto* a = actual.HostRead();
  const auto* b = expected.HostRead();
  double error = 0.0;
  for (int i = 0; i < actual.Size(); ++i)
  {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
    {
      return std::numeric_limits<double>::infinity();
    }
    error = std::max(error, std::abs(a[i] - b[i]));
  }
  return error;
}

void Compare(TestContext& ctx, const Vector& actual, const Vector& expected,
             double tolerance, const std::string& label)
{
  const double error = Difference(actual, expected);
  const double scale = std::max(1.0, expected.Normlinf());
  ctx.Check(std::isfinite(error) && error <= tolerance * scale,
            label + " max error=" + dg_igr_tests::FormatSci(error));
}

void CompareRepeat(TestContext& ctx, const Vector& actual, const Vector& expected,
                    double allowance, const std::string& label)
{
  const double error = Difference(actual, expected);
  ctx.Check(std::isfinite(error) && std::isfinite(allowance) &&
              allowance >= 0.0 && error <= allowance,
            label + " max error=" + dg_igr_tests::FormatSci(error) +
              " roundoff bound=" + dg_igr_tests::FormatSci(allowance));
}

bool LogHas(const std::string& log, const char* event, const std::string& token)
{
  const auto begin = log.find(event);
  if (begin == std::string::npos) { return false; }
  const auto end = log.find('\n', begin);
  return log.substr(begin, end - begin).find(token) != std::string::npos;
}

std::string Apply(mfem::DGHyperbolicConservationLaws& op,
                  const Vector& state, Vector& rhs, const char* execution,
                  bool legacy_alias = false, const char* row_reuse = nullptr,
                  const char* legacy_row_reuse = nullptr, bool poison_output = true)
{
  ScopedEnvironment canonical("IGRLAB_DG_PACKED_DENSE_EXECUTION",
                              legacy_alias ? nullptr : execution);
  ScopedEnvironment legacy("DG_IGR_DG_PACKED_DENSE_EXECUTION",
                           legacy_alias ? execution : nullptr);
  ScopedEnvironment reuse("IGRLAB_DG_WEAKDIV_ROW_REUSE", row_reuse);
  ScopedEnvironment legacy_reuse("DG_IGR_DG_WEAKDIV_ROW_REUSE", legacy_row_reuse);
  // Every output must be owned and overwritten, including a final partial
  // block. This is test-only poisoning, not a production diagnostic scan.
  if (poison_output) { rhs = std::numeric_limits<double>::quiet_NaN(); }
  std::ostringstream output;
  {
    struct RestoreOutput
    {
      std::streambuf* previous;
      ~RestoreOutput() { std::cout.rdbuf(previous); }
    } restore{std::cout.rdbuf(output.rdbuf())};
    op.Mult(state, rhs);
  }
  return output.str();
}

std::string ApplyReuseSelection(mfem::DGHyperbolicConservationLaws& op,
                                const Vector& state, Vector& rhs,
                                const char* execution, const char* weakdiv_reuse,
                                const char* mass_reuse, const char* face_reuse = "0",
                                const char* legacy_mass_reuse = nullptr,
                                bool poison_output = true)
{
  ScopedEnvironment mass("IGRLAB_DG_INVMASS_ROW_REUSE", mass_reuse);
  ScopedEnvironment legacy_mass("DG_IGR_DG_INVMASS_ROW_REUSE", legacy_mass_reuse);
  ScopedEnvironment face("IGRLAB_DG_TENSOR_FACE_ROW_REUSE", face_reuse);
  ScopedEnvironment legacy_face("DG_IGR_DG_TENSOR_FACE_ROW_REUSE", nullptr);
  return Apply(op, state, rhs, execution, false, weakdiv_reuse, nullptr, poison_output);
}

enum class Geometry { Straight, Shear, CurvedQ2, CurvedQ3 };

struct Fixture
{
  Geometry geometry;
  int order;
  int ordering;
  int sigma_order;
  int sigma_basis;
  const char* name;
};

mfem::Mesh MakeMesh(Geometry geometry)
{
  // Nine cells and runtime dof counts exercise partially occupied CUDA blocks.
  mfem::Mesh box = mfem::Mesh::MakeCartesian2D(
    3, 3, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  const std::vector<Vector> translations = {
    Vector({1.0, 0.0}), Vector({0.0, 1.0})};
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
    box, box.CreatePeriodicVertexMapping(translations));
  if (geometry == Geometry::CurvedQ2 || geometry == Geometry::CurvedQ3)
  {
    // Keep the discontinuous coordinate space used by MFEM periodic meshes.
    // The warp agrees on opposite boundaries, so the periodic face pairing
    // remains geometric and constant-state/global conservation checks apply.
    periodic.SetCurvature(geometry == Geometry::CurvedQ2 ? 2 : 3,
                          true, kDimension, mfem::Ordering::byVDIM);
    periodic.Transform([](const Vector& point, Vector& mapped)
    {
      const double pi = std::acos(-1.0);
      const double warp = std::sin(2.0 * pi * point[0]) *
                          std::sin(2.0 * pi * point[1]);
      mapped.SetSize(kDimension);
      mapped[0] = point[0] + 0.035 * warp;
      mapped[1] = point[1] + 0.025 * warp;
    });
  }
  else if (geometry == Geometry::Shear)
  {
    periodic.Transform([](const Vector& point, Vector& mapped)
    {
      mapped.SetSize(kDimension);
      mapped[0] = 0.85 * point[0] + 0.21 * point[1];
      mapped[1] = 1.15 * point[1];
    });
  }
  return periodic;
}

void FillState(mfem::ParFiniteElementSpace& fes, Vector& state, bool constant)
{
  auto* values = state.HostWrite();
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    const auto& nodes = fes.GetFE(e)->GetNodes();
    for (int j = 0; j < dofs.Size(); ++j)
    {
      const auto& ip = nodes.IntPoint(j);
      const double rho = constant ? 1.1 : 1.0 + 0.007 * e + 0.018 * ip.x * ip.y;
      const double vx = constant ? 0.21 : 0.22 + 0.025 * ip.x - 0.002 * e;
      const double vy = constant ? -0.13 : -0.09 + 0.015 * ip.y + 0.001 * e;
      const double pressure = constant ? 1.0 :
        1.05 + 0.01 * ip.x + 0.004 * e + 0.015 * ip.x * ip.y;
      const double u[kEquations] = {rho, rho * vx, rho * vy,
        pressure / (kGamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)};
      for (int c = 0; c < kEquations; ++c)
      {
        values[fes.DofToVDof(dofs[j], c)] = u[c];
      }
    }
  }
}

void FillSigma(mfem::ParGridFunction& sigma, bool constant, double scale)
{
  auto& fes = *sigma.ParFESpace();
  auto* values = sigma.HostWrite();
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    const auto& nodes = fes.GetFE(e)->GetNodes();
    for (int j = 0; j < dofs.Size(); ++j)
    {
      const auto& ip = nodes.IntPoint(j);
      values[dofs[j]] = constant ? 0.03 :
        scale * (0.03 + 0.004 * e + 0.017 * ip.x + 0.011 * ip.x * ip.y);
    }
  }
}

// This reference never calls the DG operator's weak-divergence or inverse-mass
// implementation. MFEM independently assembles physical dense matrices with
// its default rules; faces use its host NonlinearForm assembly. Tensor fluxes
// are evaluated below from the Euler + frozen-sigma formulas, independently of
// the packed raw flux helper and of FluxWithIGR::ComputeFlux.
class HostDenseRHS
{
  mfem::ParFiniteElementSpace& fes_;
  const mfem::NumericalFlux& flux_;
  int face_order_offset_;
  mfem::HyperbolicFormIntegrator face_integrator_;
  mfem::Array<int> boundary_marker_;
  std::unique_ptr<mfem::BdrHyperbolicFluxIntegrator> boundary_integrator_;
  mfem::ParNonlinearForm faces_;
  std::vector<DenseMatrix> weak_, inverse_;
  double repeat_allowance_ = 0.0;

  void AbsoluteFaceContributions(const Vector& state, Vector& absolute)
  {
    absolute.SetSize(fes_.GetVSize()); absolute = 0.0;
    auto* sums = absolute.HostWrite();
    const auto* input = state.HostRead();
    mfem::HyperbolicFormIntegrator integrator(flux_, face_order_offset_);
    mfem::IntegrationRule single_point(1);
    integrator.SetIntRule(&single_point);
    auto& mesh = *fes_.GetMesh();
    for (int f = 0; f < mesh.GetNumFaces(); ++f)
    {
      auto* tr = mesh.GetInteriorFaceTransformations(f);
      if (!tr) { continue; }
      const auto& left = *fes_.GetFE(tr->Elem1No);
      const auto& right = *fes_.GetFE(tr->Elem2No);
      mfem::Array<int> left_dofs, right_dofs;
      fes_.GetElementVDofs(tr->Elem1No, left_dofs);
      fes_.GetElementVDofs(tr->Elem2No, right_dofs);
      Vector local(left_dofs.Size() + right_dofs.Size()), contribution;
      for (int i = 0; i < left_dofs.Size(); ++i) { local[i] = input[left_dofs[i]]; }
      for (int i = 0; i < right_dofs.Size(); ++i)
        local[left_dofs.Size() + i] = input[right_dofs[i]];
      const auto& rule = mfem::IntRules.Get(tr->GetGeometryType(),
        2 * std::max(left.GetOrder(), right.GetOrder()) + face_order_offset_);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        single_point.IntPoint(0) = rule.IntPoint(q);
        integrator.AssembleFaceVector(left, right, *tr, local, contribution);
        for (int i = 0; i < left_dofs.Size(); ++i)
          sums[left_dofs[i]] += std::abs(contribution[i]);
        for (int i = 0; i < right_dofs.Size(); ++i)
          sums[right_dofs[i]] += std::abs(contribution[left_dofs.Size() + i]);
      }
    }
  }

public:
  std::vector<Vector> cell_mass;
  HostDenseRHS(TestContext& ctx, mfem::ParFiniteElementSpace& fes,
               const mfem::NumericalFlux& flux, const Fixture& fixture,
               int face_order_offset = 2, bool slip_walls = false)
    : fes_(fes), flux_(flux), face_order_offset_(face_order_offset),
      face_integrator_(flux, face_order_offset), faces_(&fes),
      weak_(fes.GetNE()), inverse_(fes.GetNE()), cell_mass(fes.GetNE())
  {
    faces_.AddInteriorFaceIntegrator(&face_integrator_);
    if (slip_walls)
    {
      boundary_marker_.SetSize(fes.GetMesh()->bdr_attributes.Max());
      boundary_marker_ = 1;
      boundary_integrator_ = std::make_unique<mfem::BdrHyperbolicFluxIntegrator>(
        flux, std::make_unique<mfem::SlipWallState>(kDimension, 1.e-12, 1.e-12),
        face_order_offset);
      faces_.AddBdrFaceIntegrator(boundary_integrator_.get(), boundary_marker_);
    }
    faces_.UseExternalIntegrators();
    mfem::TransposeIntegrator weak(new mfem::GradientIntegrator());
    mfem::InverseIntegrator inverse(new mfem::MassIntegrator());
    mfem::MassIntegrator mass;
    double largest_off_diagonal = 0.0, largest_diagonal = 0.0;
    double weak_difference = 0.0;
    bool positive_geometry = true, quadratic_geometry = true;
    for (int e = 0; e < fes.GetNE(); ++e)
    {
      const auto& fe = *fes.GetFE(e);
      auto& tr = *fes.GetElementTransformation(e);
      weak.AssembleElementMatrix2(fe, fe, tr, weak_[e]);
      inverse.AssembleElementMatrix(fe, tr, inverse_[e]);
      DenseMatrix matrix;
      mass.AssembleElementMatrix(fe, tr, matrix);
      Vector one(fe.GetDof()); one = 1.0;
      cell_mass[e].SetSize(fe.GetDof());
      matrix.Mult(one, cell_mass[e]);
      const auto& rule = mfem::MassIntegrator::GetRule(fe, fe, tr);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        tr.SetIntPoint(&rule.IntPoint(q));
        positive_geometry = positive_geometry &&
          std::isfinite(tr.Weight()) && tr.Weight() > 0.0;
      }
      quadratic_geometry = quadratic_geometry && tr.Order() == 2;
      if (fixture.geometry == Geometry::CurvedQ2 && e == 0)
      {
        const int q1d = fixture.order + 2;
        ctx.Check(rule.GetNPoints() == q1d * q1d &&
                    rule.GetNPoints() != fe.GetDof(),
                  std::string(fixture.name) + " physical mass uses " +
                    std::to_string(q1d) + "x" + std::to_string(q1d) +
                    " quadrature for " + std::to_string(fe.GetDof()) + " state nodes");
      }
      for (int j = 0; j < fe.GetDof(); ++j)
        for (int i = 0; i < fe.GetDof(); ++i)
        {
          if (i == j)
          {
            largest_diagonal = std::max(largest_diagonal, std::abs(inverse_[e](i, j)));
          }
          else
          {
            largest_off_diagonal = std::max(largest_off_diagonal,
                                             std::abs(inverse_[e](i, j)));
          }
        }
      for (int j = 0; j < weak_[e].Width(); ++j)
        for (int i = 0; i < weak_[e].Height(); ++i)
          weak_difference = std::max(weak_difference,
            std::abs(weak_[e](i, j) - weak_[0](i, j)));
    }
    ctx.Check(positive_geometry, std::string(fixture.name) + " has positive physical Jacobians");
    if (fixture.geometry == Geometry::CurvedQ2)
    {
      ctx.Check(quadratic_geometry && largest_off_diagonal > 1.e-6 * largest_diagonal,
                std::string(fixture.name) + " exercises off-diagonal Q2 inverse mass entries");
      ctx.Check(weak_difference > 1.e-6,
                std::string(fixture.name) + " requires distinct mapped weak-divergence blocks");
    }
  }

  double RepeatAllowance() const { return repeat_allowance_; }

  Vector SlipBoundaryFlux(const Vector& state, double& normal_variation) const
  {
    Vector integral(kEquations); integral = 0.0;
    normal_variation = 0.0;
    const auto* input = state.HostRead();
    auto& mesh = *fes_.GetMesh();
    for (int b = 0; b < mesh.GetNBE(); ++b)
    {
      auto* tr = mesh.GetBdrFaceTransformations(b);
      if (!tr) { continue; }
      const auto& fe = *fes_.GetFE(tr->Elem1No);
      mfem::Array<int> dofs;
      fes_.GetElementDofs(tr->Elem1No, dofs);
      Vector shape(fe.GetDof()), minus(kEquations), plus(kEquations);
      Vector normal(kDimension), first_normal(kDimension), flux(kEquations);
      const auto& rule = mfem::IntRules.Get(tr->GetGeometryType(),
        2 * fe.GetOrder() + face_order_offset_);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        const auto& point = rule.IntPoint(q);
        tr->SetAllIntPoints(&point);
        fe.CalcShape(tr->GetElement1IntPoint(), shape);
        minus = 0.0;
        for (int j = 0; j < dofs.Size(); ++j)
          for (int c = 0; c < kEquations; ++c)
            minus[c] += shape[j] * input[fes_.DofToVDof(dofs[j], c)];
        mfem::CalcOrtho(tr->Jacobian(), normal);
        const double length = normal.Norml2();
        MFEM_VERIFY(std::isfinite(length) && length > 0.0,
                    "Curved boundary oracle requires finite physical normals.");
        if (q == 0) { first_normal = normal; first_normal /= length; }
        normal_variation = std::max(normal_variation,
          std::hypot(normal[0] / length - first_normal[0],
                     normal[1] / length - first_normal[1]));
        // Reflect momentum analytically, independently of SlipWallState and
        // the packed exterior-state helper. Density and energy stay fixed.
        plus = minus;
        const double reflected = 2.0 *
          (minus[1] * normal[0] + minus[2] * normal[1]) / (length * length);
        plus[1] -= reflected * normal[0];
        plus[2] -= reflected * normal[1];
        flux_.Eval(minus, plus, normal, *tr, flux);
        integral.Add(point.weight, flux);
      }
    }
    return integral;
  }

  double Apply(const Vector& state, const mfem::ParGridFunction* sigma,
                Vector& rhs, const mfem::FluxWithIGR* tracked_flux = nullptr)
  {
    face_integrator_.ResetMaxCharSpeed();
    if (boundary_integrator_) { boundary_integrator_->ResetMaxCharSpeed(); }
    Vector face_rhs(fes_.GetVSize());
    faces_.Mult(state, face_rhs);
    double maximum_speed = face_integrator_.GetMaxCharSpeed();
    if (boundary_integrator_)
      maximum_speed = std::max(maximum_speed,
        static_cast<double>(boundary_integrator_->GetMaxCharSpeed()));
    Vector absolute_face;
    // Exact repeats are only checked for the original periodic cohorts.
    const bool atomic_faces = !boundary_integrator_ &&
      mfem::Device::Allows(mfem::Backend::CUDA_MASK);
    if (atomic_faces) { AbsoluteFaceContributions(state, absolute_face); }
    const auto* face_bound = atomic_faces ? absolute_face.HostRead() : nullptr;
    repeat_allowance_ = 0.0;
    const auto* input = state.HostRead();
    const auto* face = face_rhs.HostRead();
    auto* result = rhs.HostWrite();
    for (int e = 0; e < fes_.GetNE(); ++e)
    {
      const auto& fe = *fes_.GetFE(e);
      const int dof = fe.GetDof();
      auto& tr = *fes_.GetElementTransformation(e);
      mfem::Array<int> dofs;
      fes_.GetElementDofs(e, dofs);
      DenseMatrix nodal(dof * kDimension, kEquations), residual(dof, kEquations);
      DenseMatrix absolute_residual(dof, kEquations);
      for (int j = 0; j < dof; ++j)
      {
        const auto& ip = fe.GetNodes().IntPoint(j);
        tr.SetIntPoint(&ip);
        double u[kEquations];
        for (int c = 0; c < kEquations; ++c)
          u[c] = input[fes_.DofToVDof(dofs[j], c)];
        const double vx = u[1] / u[0], vy = u[2] / u[0];
        const double pressure = (kGamma - 1.0) *
          (u[3] - 0.5 * (u[1] * u[1] + u[2] * u[2]) / u[0]);
        const double s = sigma ? sigma->GetValue(tr, ip) : 0.0;
        nodal(j, 0) = u[1];
        nodal(j, 1) = u[1] * vx + pressure + s;
        nodal(j, 2) = u[2] * vx;
        nodal(j, 3) = (u[3] + pressure + s) * vx;
        nodal(dof + j, 0) = u[2];
        nodal(dof + j, 1) = u[1] * vy;
        nodal(dof + j, 2) = u[2] * vy + pressure + s;
        nodal(dof + j, 3) = (u[3] + pressure + s) * vy;
        if (tracked_flux)
        {
          // Only the supplementary invariant-speed case requests node speeds;
          // the primary campaign fixtures keep volume speed tracking off.
          Vector value(u, kEquations);
          DenseMatrix unused(kEquations, kDimension);
          maximum_speed = std::max(maximum_speed,
            static_cast<double>(tracked_flux->ComputeFlux(value, tr, unused)));
        }
      }
      for (int c = 0; c < kEquations; ++c)
        for (int i = 0; i < dof; ++i)
        {
          long double sum = face[fes_.DofToVDof(dofs[i], c)];
          long double absolute_sum = atomic_faces ?
            face_bound[fes_.DofToVDof(dofs[i], c)] : 0.0L;
          for (int j = 0; j < dof * kDimension; ++j)
          {
            sum += static_cast<long double>(weak_[e](i, j)) * nodal(j, c);
            absolute_sum += std::abs(static_cast<long double>(weak_[e](i, j)) * nodal(j, c));
          }
          residual(i, c) = static_cast<double>(sum);
          absolute_residual(i, c) = static_cast<double>(absolute_sum);
        }
      // The unchanged generic Rusanov kernels atomically add four faces times
      // nq contributions into each element coefficient. Bound two evaluations
      // with gamma_n = n*eps/(1-n*eps), propagated through the absolute dense
      // weak-divergence and inverse-mass contractions. This uses physical
      // matrix/flux magnitudes, including cancellation in a constant RHS.
      const int nq = mfem::IntRules.Get(mfem::Geometry::SEGMENT,
        2 * fe.GetOrder() + face_order_offset_).GetNPoints();
      const double operations = 4 * nq + 2 * kDimension * dof + 2 * dof + 8;
      const double accumulated_epsilon = operations * std::numeric_limits<double>::epsilon();
      const double gamma_n = accumulated_epsilon / (1.0 - accumulated_epsilon);
      for (int c = 0; c < kEquations; ++c)
        for (int i = 0; i < dof; ++i)
        {
          long double sum = 0.0L, absolute_sum = 0.0L;
          for (int j = 0; j < dof; ++j)
          {
            sum += static_cast<long double>(inverse_[e](i, j)) * residual(j, c);
            absolute_sum += std::abs(static_cast<long double>(inverse_[e](i, j))) *
                            absolute_residual(j, c);
          }
          result[fes_.DofToVDof(dofs[i], c)] = static_cast<double>(sum);
          if (atomic_faces)
            repeat_allowance_ = std::max(repeat_allowance_,
              static_cast<double>(2.0L * gamma_n * absolute_sum));
        }
    }
    return maximum_speed;
  }
};

void CheckConservation(TestContext& ctx, mfem::ParFiniteElementSpace& fes,
                        const std::vector<Vector>& cell_mass, const Vector& rhs,
                        bool nonconstant, const std::string& label)
{
  std::array<long double, kEquations> total{}, bound{};
  long double largest_cell_moment = 0.0L;
  const auto* values = rhs.HostRead();
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    for (int c = 0; c < kEquations; ++c)
    {
      long double cell = 0.0L;
      for (int i = 0; i < dofs.Size(); ++i)
      {
        const long double term = static_cast<long double>(cell_mass[e][i]) *
          values[fes.DofToVDof(dofs[i], c)];
        cell += term;
        bound[c] += std::abs(term);
      }
      total[c] += cell;
      largest_cell_moment = std::max(largest_cell_moment, std::abs(cell));
    }
  }
  bool conserved = true;
  for (int c = 0; c < kEquations; ++c)
    conserved = conserved && std::isfinite(total[c]) &&
      std::abs(total[c]) <= 5.e-11L * std::max(1.0L, bound[c]);
  ctx.Check(conserved, label + " conserves periodic physical mass-weighted moments");
  if (nonconstant)
    ctx.Check(largest_cell_moment > 1.e-5L,
              label + " has nonzero face-coupled cell moments");
}

std::unique_ptr<mfem::DGHyperbolicConservationLaws> MakeOperator(
  mfem::ParFiniteElementSpace& fes, const mfem::NumericalFlux& flux,
  const char* execution, int face_order_offset = 2)
{
  ScopedEnvironment selected("IGRLAB_DG_PACKED_DENSE_EXECUTION", execution);
  auto op = std::make_unique<mfem::DGHyperbolicConservationLaws>(
    fes, flux, face_order_offset, true);
  op->SetDGSEMGamma(kGamma);
  op->SetDGSEMDiagnosticStateScan(false);
  return op;
}

void CheckFixture(TestContext& ctx, const Fixture& fixture, bool supplementary)
{
  mfem::Mesh serial = MakeMesh(fixture.geometry);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  mfem::L2_FECollection fec(fixture.order, kDimension, mfem::BasisType::GaussLegendre);
  mfem::L2_FECollection sfec(fixture.sigma_order, kDimension, fixture.sigma_basis);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, kEquations, fixture.ordering);
  mfem::ParFiniteElementSpace sfes(&mesh, &sfec);
  mfem::ParGridFunction sigma(&sfes);
  mfem::EulerFlux euler(kDimension, kGamma);
  mfem::FluxWithIGR physical(euler, kGamma);
  physical.SetSigmaField(&sigma);
  physical.SetInvariantDomainFlux(false);
  physical.SetSigmaAwareCFL(false);
  physical.SetSigmaAwareRusanov(false);
  mfem::RusanovFluxWithSides flux(physical);
  auto rows = MakeOperator(fes, flux, "rows");
  auto element = MakeOperator(fes, flux, "element");
  auto reuse = MakeOperator(fes, flux, "rows");
  auto mass_reuse = MakeOperator(fes, flux, "rows");
  auto combined_reuse = MakeOperator(fes, flux, "rows");
  HostDenseRHS oracle(ctx, fes, flux, fixture);
  Vector state(fes.GetVSize()), actual(fes.GetVSize()), legacy(fes.GetVSize());
  Vector expected(fes.GetVSize()), repeat(fes.GetVSize()), legacy_repeat(fes.GetVSize());
  Vector previous_sigma(fes.GetVSize());
  Vector reused(fes.GetVSize()), reused_repeat(fes.GetVSize());
  Vector mass_reused(fes.GetVSize()), combined_reused(fes.GetVSize());
  actual.UseDevice(true); legacy.UseDevice(true); repeat.UseDevice(true); legacy_repeat.UseDevice(true);
  reused.UseDevice(true); reused_repeat.UseDevice(true);
  mass_reused.UseDevice(true); combined_reused.UseDevice(true);
  const std::string cohort(fixture.name);
  ctx.Check(!rows->GetDGSEMDiagnosticStateScan() &&
              !element->GetDGSEMDiagnosticStateScan() &&
              !rows->GetInvariantDomainVolumeSpeedTracking() &&
              !element->GetInvariantDomainVolumeSpeedTracking(),
            cohort + " keeps optional diagnostic scans and volume speed tracking off");
  // Constant Euler+sigma; nonconstant Euler; then the same state with frozen
  // sigma and changed current sigma values. No sigma solve is part of this RHS.
  for (int phase = 0; phase < 4; ++phase)
  {
    FillState(fes, state, phase == 0);
    FillSigma(sigma, phase == 0, phase == 3 ? 1.8 : 1.0);
    physical.SetSigmaField(phase == 1 ? nullptr : &sigma);
    const std::string label = cohort + " phase=" + std::to_string(phase);
    const std::string row_log = Apply(*rows, state, actual, "rows");
    const std::string element_log = ApplyReuseSelection(
      *element, state, legacy, "element", "1", "1");
    const std::string reuse_log = Apply(*reuse, state, reused, "rows", false, "1");
    const std::string mass_log = ApplyReuseSelection(
      *mass_reuse, state, mass_reused, "rows", "0", "1");
    const std::string combined_log = ApplyReuseSelection(
      *combined_reuse, state, combined_reused, "rows", "1", "1");
    const double expected_speed = oracle.Apply(state, physical.GetSigmaField(), expected);
    Compare(ctx, actual, legacy, 3.e-12, label + " rows versus legacy element full RHS");
    Compare(ctx, actual, expected, 2.e-11, label + " full RHS versus independent host dense reference");
    Compare(ctx, reused, actual, 3.e-12, label + " row reuse versus original rows full RHS");
    Compare(ctx, reused, expected, 2.e-11, label + " row reuse versus independent host dense reference");
    Compare(ctx, mass_reused, expected, 2.e-11,
            label + " inverse-mass row reuse versus independent host dense reference");
    Compare(ctx, combined_reused, expected, 2.e-11,
            label + " combined weak-divergence and inverse-mass reuse versus independent reference");
    Compare(ctx, mass_reused, actual, 3.e-12, label + " inverse-mass reuse versus original rows");
    Compare(ctx, combined_reused, reused, 3.e-12, label + " combined reuse versus weak-divergence reuse");
    const double speed = rows->GetMaxCharSpeed();
    ctx.Check(std::isfinite(speed) && std::isfinite(expected_speed) &&
                std::abs(speed - expected_speed) <= 2.e-12 * std::max(1.0, expected_speed) &&
                speed == element->GetMaxCharSpeed(),
              label + " preserves the face/CFL characteristic speed policy");
    ctx.Check(reuse->GetMaxCharSpeed() == speed,
              label + " row reuse preserves the characteristic speed");
    ctx.Check(mass_reuse->GetMaxCharSpeed() == speed &&
                combined_reuse->GetMaxCharSpeed() == speed,
              label + " inverse-mass and combined reuse preserve the characteristic speed");
    Apply(*rows, state, repeat, "rows");
    Apply(*element, state, legacy_repeat, "element");
    Apply(*reuse, state, reused_repeat, "rows", false, "1");
    CompareRepeat(ctx, legacy_repeat, legacy, oracle.RepeatAllowance(),
                    label + " legacy repeat controls atomic face roundoff");
    CompareRepeat(ctx, repeat, actual, oracle.RepeatAllowance(),
                    label + " rows repeat with finite overwritten outputs");
    CompareRepeat(ctx, reused_repeat, reused, oracle.RepeatAllowance(),
                    label + " row reuse repeats with finite overwritten outputs");
    ApplyReuseSelection(*mass_reuse, state, reused_repeat, "rows", "0", "1");
    CompareRepeat(ctx, reused_repeat, mass_reused, oracle.RepeatAllowance(),
                    label + " inverse-mass reuse repeats with finite overwritten outputs");
    ApplyReuseSelection(*combined_reuse, state, reused_repeat, "rows", "1", "1");
    CompareRepeat(ctx, reused_repeat, combined_reused, oracle.RepeatAllowance(),
                    label + " combined reuse repeats with finite overwritten outputs");
    CheckConservation(ctx, fes, oracle.cell_mass, actual, phase != 0, label);
    CheckConservation(ctx, fes, oracle.cell_mass, reused, phase != 0, label + " row reuse");
    CheckConservation(ctx, fes, oracle.cell_mass, mass_reused, phase != 0, label + " inverse-mass reuse");
    CheckConservation(ctx, fes, oracle.cell_mass, combined_reused, phase != 0, label + " combined reuse");
    if (phase == 0)
    {
      ctx.Check(actual.Normlinf() < 8.e-11, cohort + " preserves a constant state");
      ctx.Check(reused.Normlinf() < 8.e-11, cohort + " row reuse preserves a constant state");
      if (mfem::Device::Allows(mfem::Backend::CUDA_MASK))
      {
        const bool dense = fixture.geometry == Geometry::CurvedQ2 ||
                           fixture.geometry == Geometry::CurvedQ3;
        ctx.Check(LogHas(row_log, "event=weakdiv-volume-path", "dense_execution=rows") &&
                    LogHas(element_log, "event=weakdiv-volume-path", "dense_execution=element"),
                  cohort + " selects both packed dense weak-divergence implementations");
        ctx.Check(LogHas(row_log, "event=weakdiv-volume-path", "weakdiv_row_reuse=0") &&
                    LogHas(reuse_log, "event=weakdiv-volume-path", "weakdiv_row_reuse=1") &&
                    LogHas(reuse_log, "event=weakdiv-volume-path", "weakdiv_row_reuse_requested=1") &&
                    LogHas(element_log, "event=weakdiv-volume-path", "weakdiv_row_reuse=0") &&
                    LogHas(element_log, "event=weakdiv-volume-path", "weakdiv_row_reuse_requested=1"),
                  cohort + " selects row reuse and retains requested element fallback");
        ctx.Check(LogHas(row_log, "event=invmass-path", dense ? "dense_execution=rows" : "dense_execution=diagonal") &&
                    LogHas(element_log, "event=invmass-path", dense ? "dense_execution=element" : "dense_execution=diagonal"),
                  cohort + " selects the expected physical inverse-mass representation");
        const std::string mass_token = dense ? "invmass_row_reuse=1" : "invmass_row_reuse=0";
        ctx.Check(LogHas(row_log, "event=invmass-path", "invmass_row_reuse=0") &&
                    LogHas(mass_log, "event=invmass-path", mass_token) &&
                    LogHas(combined_log, "event=invmass-path", mass_token) &&
                    LogHas(mass_log, "event=invmass-path", "invmass_row_reuse_requested=1") &&
                    LogHas(element_log, "event=invmass-path", "invmass_row_reuse_requested=1") &&
                    LogHas(element_log, "event=invmass-path", "invmass_row_reuse=0"),
                  cohort + " selects inverse-mass reuse or the original diagonal/element fallback");
        const bool exact = fixture.sigma_order == fixture.order &&
          fixture.sigma_basis == mfem::BasisType::GaussLegendre;
        ctx.Check(row_log.find(exact ? "storage=exact-nodes" : "storage=shared-reference") != std::string::npos,
                  cohort + " exercises the requested sigma node/interpolation route");
      }
      else
        ctx.Check(LogHas(mass_log, "event=invmass-path", "selected=host-fallback") &&
                    LogHas(mass_log, "event=invmass-path", "backend=cpu"),
                  cohort + " retains the CPU inverse-mass path when reuse is requested");
    }
    if (phase == 2) { previous_sigma = actual; }
    if (phase == 3)
      ctx.Check(Difference(actual, previous_sigma) > 1.e-4,
                cohort + " consumes changed sigma values without rebuilding the operator");
  }
  if (supplementary || fixture.geometry == Geometry::Straight)
  {
    // Exercise the public full-RHS alias contract against the independently
    // assembled mass/weak/face reference. Aliased outputs cannot be poisoned
    // before the apply because they contain the input state.
    Vector in_place(state); in_place.UseDevice(true);
    ApplyReuseSelection(*combined_reuse, in_place, in_place,
                         "rows", "1", "1", "0", nullptr, false);
    Compare(ctx, in_place, expected, 2.e-11,
            cohort + " combined reuse preserves full-RHS in-place aliasing");
    for (int shift : {-3, 3})
    {
      Vector storage(state.Size() + std::abs(shift)); storage.UseDevice(true);
      storage = -777.0;
      Vector input_alias, output_alias;
      input_alias.MakeRef(storage, shift < 0 ? -shift : 0, state.Size());
      output_alias.MakeRef(storage, shift > 0 ? shift : 0, state.Size());
      input_alias.UseDevice(true); output_alias.UseDevice(true);
      input_alias = state;
      ApplyReuseSelection(*combined_reuse, input_alias, output_alias,
                           "rows", "1", "1", "0", nullptr, false);
      Compare(ctx, output_alias, expected, 2.e-11,
              cohort + " combined reuse preserves full-RHS offset alias shift=" +
                std::to_string(shift));
    }
  }
  if (!supplementary) { return; }
  Apply(*rows, state, repeat, nullptr);
  CompareRepeat(ctx, repeat, actual, oracle.RepeatAllowance(),
                  cohort + " default execution agrees with explicit rows");
  Apply(*rows, state, repeat, "rows", true);
  CompareRepeat(ctx, repeat, actual, oracle.RepeatAllowance(),
                  cohort + " legacy selector alias agrees with explicit rows");
  auto legacy_reuse = MakeOperator(fes, flux, "rows");
  const std::string alias_log = Apply(*legacy_reuse, state, reused_repeat,
                                      "rows", false, nullptr, "1");
  CompareRepeat(ctx, reused_repeat, reused, oracle.RepeatAllowance(),
                  cohort + " legacy row-reuse alias agrees with canonical selection");
  auto precedence = MakeOperator(fes, flux, "rows");
  const std::string precedence_log = Apply(*precedence, state, reused_repeat,
                                           "rows", false, "0", "1");
  CompareRepeat(ctx, reused_repeat, actual, oracle.RepeatAllowance(),
                  cohort + " canonical row-reuse off overrides the legacy request");
  if (mfem::Device::Allows(mfem::Backend::CUDA_MASK))
    ctx.Check(LogHas(alias_log, "event=weakdiv-volume-path", "weakdiv_row_reuse=1") &&
                LogHas(precedence_log, "event=weakdiv-volume-path", "weakdiv_row_reuse=0") &&
                LogHas(precedence_log, "event=weakdiv-volume-path", "weakdiv_row_reuse_requested=0"),
              cohort + " verifies row-reuse alias selection and canonical precedence");
  auto mass_alias = MakeOperator(fes, flux, "rows");
  const std::string mass_alias_log = ApplyReuseSelection(
    *mass_alias, state, reused_repeat, "rows", "0", nullptr, "0", "1");
  CompareRepeat(ctx, reused_repeat, mass_reused, oracle.RepeatAllowance(),
                  cohort + " inverse-mass legacy alias agrees with canonical selection");
  auto mass_precedence = MakeOperator(fes, flux, "rows");
  const std::string mass_precedence_log = ApplyReuseSelection(
    *mass_precedence, state, reused_repeat, "rows", "0", "0", "0", "1");
  CompareRepeat(ctx, reused_repeat, actual, oracle.RepeatAllowance(),
                  cohort + " canonical inverse-mass off overrides the legacy request");
  if (mfem::Device::Allows(mfem::Backend::CUDA_MASK))
    ctx.Check(LogHas(mass_alias_log, "event=invmass-path", "invmass_row_reuse=1") &&
                LogHas(mass_precedence_log, "event=invmass-path", "invmass_row_reuse=0") &&
                LogHas(mass_precedence_log, "event=invmass-path", "invmass_row_reuse_requested=0"),
              cohort + " verifies inverse-mass alias selection and canonical precedence");

  // Perturb only one cell. Its neighbors must change through shared faces,
  // while independent full-RHS comparison still covers every output row.
  mfem::Array<int> perturbed_dofs;
  fes.GetElementDofs(0, perturbed_dofs);
  auto* modified = state.HostReadWrite();
  for (int j = 0; j < perturbed_dofs.Size(); ++j)
    modified[fes.DofToVDof(perturbed_dofs[j], 3)] += 0.004;
  Apply(*rows, state, repeat, "rows");
  Apply(*reuse, state, reused_repeat, "rows", false, "1");
  ApplyReuseSelection(*combined_reuse, state, combined_reused, "rows", "1", "1");
  oracle.Apply(state, &sigma, expected);
  Compare(ctx, repeat, expected, 2.e-11, cohort + " cell perturbation versus independent full RHS");
  Compare(ctx, reused_repeat, expected, 2.e-11,
            cohort + " row reuse reads changed state versus independent full RHS");
  Compare(ctx, combined_reused, expected, 2.e-11,
            cohort + " inverse-mass reuse reads changed state versus independent full RHS");
  double neighbor_change = 0.0;
  const auto* perturbed = repeat.HostRead(); const auto* before = actual.HostRead();
  for (int e = 1; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    for (int j = 0; j < dofs.Size(); ++j)
      for (int c = 0; c < kEquations; ++c)
      {
        const int index = fes.DofToVDof(dofs[j], c);
        neighbor_change = std::max(neighbor_change, std::abs(perturbed[index] - before[index]));
      }
  }
  ctx.Check(neighbor_change > 1.e-4, cohort + " transmits a cell perturbation through global face coupling");
  CheckConservation(ctx, fes, oracle.cell_mass, repeat, true, cohort + " perturbed cell");

  // Reuse the curved mesh with fresh operators: face telemetry allocation is
  // fixed when the invariant-speed policy first builds the face metadata.
  physical.SetInvariantDomainFlux(true);
  rows = MakeOperator(fes, flux, "rows");
  element = MakeOperator(fes, flux, "element");
  reuse = MakeOperator(fes, flux, "rows");
  rows->SetInvariantDomainVolumeSpeedTracking(true);
  element->SetInvariantDomainVolumeSpeedTracking(true);
  reuse->SetInvariantDomainVolumeSpeedTracking(true);
  Apply(*rows, state, actual, "rows");
  Apply(*element, state, legacy, "element");
  Apply(*reuse, state, reused, "rows", false, "1");
  const double expected_speed = oracle.Apply(state, &sigma, expected, &physical);
  Compare(ctx, actual, legacy, 3.e-12, cohort + " invariant-speed rows versus element");
  Compare(ctx, actual, expected, 2.e-11, cohort + " invariant-speed independent full RHS");
  Compare(ctx, reused, expected, 2.e-11,
            cohort + " invariant-speed row reuse versus independent full RHS");
  const double speed = rows->GetMaxCharSpeed();
  ctx.Check(std::isfinite(speed) && std::isfinite(expected_speed) &&
              std::abs(speed - expected_speed) <= 2.e-12 * std::max(1.0, expected_speed) &&
              speed == element->GetMaxCharSpeed() && speed == reuse->GetMaxCharSpeed(),
            cohort + " invariant-speed node maximum agrees with host reference");
}

void CheckWallBalance(TestContext& ctx, mfem::ParFiniteElementSpace& fes,
                       const std::vector<Vector>& cell_mass, const Vector& rhs,
                       const Vector& outward, const std::string& label)
{
  const auto* values = rhs.HostRead();
  const auto* flux = outward.HostRead();
  std::array<long double, kEquations> total{}, bound{};
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    for (int c = 0; c < kEquations; ++c)
      for (int i = 0; i < dofs.Size(); ++i)
      {
        const long double term = static_cast<long double>(cell_mass[e][i]) *
          values[fes.DofToVDof(dofs[i], c)];
        total[c] += term; bound[c] += std::abs(term);
      }
  }
  bool balanced = true;
  for (int c = 0; c < kEquations; ++c)
    balanced = balanced && std::isfinite(total[c]) && std::isfinite(flux[c]) &&
      std::abs(total[c] + flux[c]) <=
        5.e-11L * std::max(1.0L, bound[c] + std::abs(flux[c]));
  ctx.Check(balanced, label + " closes the independently integrated physical wall flux balance");
}

void CheckCurvedTensorFaces(TestContext& ctx)
{
  struct Configuration
  {
    int geometry_order;
    int face_order_offset;
    bool mismatched_sigma;
    const char* name;
    const char* fallback;
  };
  const std::array<Configuration, 4> configurations{{
    {2, 1, false, "Q2-p5-curved-slip-wall", nullptr},
    {2, 2, false, "Q2-p5-curved-slip-wall-higher-quadrature", "quadrature"},
    {2, 1, true, "Q2-p5-curved-slip-wall-interpolated-sigma", "sigma-layout"},
    {3, 1, false, "Q3-p5-curved-slip-wall", "geometry"}
  }};
  struct Execution
  {
    const char* face_mode;
    const char* weakdiv_reuse;
    const char* mass_reuse;
    const char* face_reuse;
    const char* name;
  };
  const std::array<Execution, 7> executions{{
    {"off", "0", "0", "0", "generic-faces"},
    {"fp64", "0", "0", "0", "original-tensor-faces"},
    {"fp64", "1", "0", "0", "weakdiv-reuse"},
    {"fp64", "0", "1", "0", "mass-reuse"},
    {"fp64", "1", "1", "0", "weakdiv-mass-reuse"},
    {"fp64", "0", "0", "1", "face-reuse"},
    {"fp64", "1", "1", "1", "all-reuse"}
  }};
  for (const auto& configuration : configurations)
  {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      2, 2, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
    serial.SetCurvature(configuration.geometry_order, false,
                        kDimension, mfem::Ordering::byVDIM);
    serial.Transform([&configuration](const Vector& point, Vector& mapped)
    {
      const double x = point[0], y = point[1];
      mapped.SetSize(kDimension);
      mapped[0] = x + 0.06 * y * (1.0 - y) + 0.04 * x * y;
      mapped[1] = y + 0.07 * x * (1.0 - x) + 0.03 * x * y;
      if (configuration.geometry_order == 3) { mapped[0] += 0.02 * x * x * x * y; }
    });
    mfem::ParMesh mesh(MPI_COMM_SELF, serial);
    mfem::L2_FECollection fec(5, kDimension, mfem::BasisType::GaussLegendre);
    mfem::L2_FECollection sfec(5, kDimension, configuration.mismatched_sigma
      ? mfem::BasisType::GaussLobatto : mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace fes(&mesh, &fec, kEquations, mfem::Ordering::byNODES);
    mfem::ParFiniteElementSpace sfes(&mesh, &sfec);
    mfem::ParGridFunction sigma(&sfes);
    Vector state(fes.GetVSize());
    FillState(fes, state, false);
    FillSigma(sigma, false, 1.0);
    mfem::EulerFlux euler(kDimension, kGamma);
    mfem::FluxWithIGR physical(euler, kGamma);
    physical.SetSigmaField(&sigma);
    physical.SetInvariantDomainFlux(false);
    physical.SetSigmaAwareCFL(false);
    physical.SetSigmaAwareRusanov(false);
    mfem::RusanovFluxWithSides flux(physical);
    const Fixture fixture{configuration.geometry_order == 2
      ? Geometry::CurvedQ2 : Geometry::CurvedQ3, 5, mfem::Ordering::byNODES, 5,
      configuration.mismatched_sigma ? mfem::BasisType::GaussLobatto
                                    : mfem::BasisType::GaussLegendre,
      configuration.name};
    HostDenseRHS oracle(ctx, fes, flux, fixture,
                         configuration.face_order_offset, true);
    mfem::Array<int> boundaries(mesh.bdr_attributes.Max()); boundaries = 1;
    std::array<std::unique_ptr<mfem::DGHyperbolicConservationLaws>, 7> operators;
    std::array<Vector, 7> residual, captured;
    for (int variant = 0; variant < static_cast<int>(executions.size()); ++variant)
    {
      ScopedEnvironment face_mode("IGRLAB_DG_TENSOR_FACE", executions[variant].face_mode);
      operators[variant] = MakeOperator(fes, flux, "rows", configuration.face_order_offset);
      operators[variant]->AddBdrFaceIntegrator(new mfem::BdrHyperbolicFluxIntegrator(
        flux, std::make_unique<mfem::SlipWallState>(kDimension, 1.e-12, 1.e-12),
        configuration.face_order_offset), boundaries);
      residual[variant].SetSize(fes.GetVSize()); residual[variant].UseDevice(true);
    }
    // Only the admitted fixture repeats with changed current sigma. Each
    // fallback configuration needs one complete RHS and boundary comparison.
    const int phases = configuration.fallback ? 1 : 2;
    Vector expected(fes.GetVSize()), first_sigma(fes.GetVSize()), repeat(fes.GetVSize());
    repeat.UseDevice(true);
    for (int phase = 0; phase < phases; ++phase)
    {
      FillSigma(sigma, false, phase ? 1.8 : 1.0);
      const std::string label = std::string(configuration.name) + " phase=" + std::to_string(phase);
      const double expected_speed = oracle.Apply(state, &sigma, expected);
      double normal_variation = 0.0;
      const Vector boundary_reference = oracle.SlipBoundaryFlux(state, normal_variation);
      ctx.Check(normal_variation > 1.e-3,
                label + " has varying physical normal directions along boundary edges");
      ctx.Check(boundary_reference.Normlinf() > 1.e-4 &&
                  std::abs(boundary_reference[0]) < 2.e-12 &&
                  std::abs(boundary_reference[3]) < 2.e-12,
                label + " has nontrivial wall force and zero mass/energy leakage");
      for (int variant = 0; variant < static_cast<int>(executions.size()); ++variant)
      {
        const auto& execution = executions[variant];
        const char* mode = execution.face_mode;
        const std::string case_label = label + " variant=" + execution.name;
        ScopedEnvironment face_mode("IGRLAB_DG_TENSOR_FACE", mode);
        auto& op = *operators[variant];
        op.ArmPhysicalBoundaryFluxIntegral();
        const std::string log = ApplyReuseSelection(
          op, state, residual[variant], "rows", execution.weakdiv_reuse,
          execution.mass_reuse, execution.face_reuse);
        if (phase == 0 && mfem::Device::Allows(mfem::Backend::CUDA_MASK))
        {
          ctx.Check(LogHas(log, "event=weakdiv-volume-path",
                            std::string("weakdiv_row_reuse=") + execution.weakdiv_reuse) &&
                      LogHas(log, "event=invmass-path",
                              std::string("invmass_row_reuse=") + execution.mass_reuse),
                    case_label + " selects the requested weak-divergence and inverse-mass paths");
        }
        ctx.Check(op.HasLastPhysicalBoundaryFluxIntegral(),
                  case_label + " publishes the requested physical boundary integral mode=" + mode);
        if (op.HasLastPhysicalBoundaryFluxIntegral())
        {
          captured[variant] = op.GetLastPhysicalBoundaryFluxIntegral();
          Compare(ctx, captured[variant], boundary_reference, 2.e-12,
                    case_label + " boundary flux versus independent reflection/quadrature mode=" + mode);
        }
        Compare(ctx, residual[variant], expected, 2.e-11,
                  case_label + " full RHS versus independent host reference mode=" + mode);
        CheckWallBalance(ctx, fes, oracle.cell_mass, residual[variant], boundary_reference,
                           case_label + " mode=" + mode);
        const double speed = op.GetMaxCharSpeed();
        ctx.Check(std::isfinite(speed) && std::isfinite(expected_speed) &&
                    std::abs(speed - expected_speed) <= 2.e-12 * std::max(1.0, expected_speed),
                  case_label + " preserves physical boundary/interior characteristic speeds mode=" + mode);
        if (!variant && phase == 0 && mfem::Device::Allows(mfem::Backend::CUDA_MASK))
          ctx.Check(log.find("[dg-tensor-face]") == std::string::npos &&
                      LogHas(log, "event=face-domain-path", "selected=packed-mfem-forall-euler-rusanov"),
                    case_label + " keeps the off control on original generic faces");
        if (variant && phase == 0 && mfem::Device::Allows(mfem::Backend::CUDA_MASK))
        {
          const char* face_event = configuration.fallback
            ? "[dg-tensor-face] event=fallback" : "[dg-tensor-face] event=setup";
          const char* selected_face_reuse = configuration.fallback ? "0" : execution.face_reuse;
          ctx.Check(LogHas(log, face_event,
                            std::string("row_reuse_requested=") + execution.face_reuse) &&
                      LogHas(log, face_event, std::string("row_reuse=") + selected_face_reuse),
                    case_label + " selects face reuse or its admitted fallback explicitly");
          if (configuration.fallback)
          {
            ctx.Check(LogHas(log, "[dg-tensor-face] event=fallback",
                              std::string("reason=") + configuration.fallback) &&
                        log.find("[dg-tensor-face] event=setup selected=fp64") == std::string::npos &&
                        LogHas(log, "event=face-domain-path", "selected=packed-mfem-forall-euler-rusanov"),
                      case_label + " retains original generic faces for the incompatible configuration");
          }
          else
          {
            ctx.Check(LogHas(log, "[dg-tensor-face] event=setup", "selected=fp64") &&
                        LogHas(log, "[dg-tensor-face] event=setup", "geometry_order_max=2") &&
                        LogHas(log, "[dg-tensor-face] event=setup", "jacobian_validation=sampled-q2"),
                      case_label + " admits sampled Q2 geometry to the exact tensor face path");
          }
        }
        if (variant && !configuration.fallback)
        {
          ApplyReuseSelection(op, state, repeat, "rows", execution.weakdiv_reuse,
                               execution.mass_reuse, execution.face_reuse);
          CompareRepeat(ctx, repeat, residual[variant], 0.0,
                          case_label + " tensor faces repeat with every output row owned");
          if (variant == 6 && phase == 0)
          {
            Vector in_place(state); in_place.UseDevice(true);
            ApplyReuseSelection(op, in_place, in_place, "rows", "1", "1", "1", nullptr, false);
            Compare(ctx, in_place, expected, 2.e-11,
                    case_label + " preserves the physical-boundary RHS with in-place output");
          }
        }
      }
      Compare(ctx, residual[1], residual[0], 3.e-12,
                label + " tensor/requested fallback versus generic full RHS");
      // Admitted tensor faces own each output, so these complete RHSs also
      // provide an exact execution check without atomic face-order variation.
      for (int variant = 2; variant < static_cast<int>(executions.size()); ++variant)
        Compare(ctx, residual[variant], residual[1], configuration.fallback ? 3.e-12 : 0.0,
                  label + " reuse preserves the complete physical boundary RHS variant=" +
                    executions[variant].name);
      if (phase == 0) { first_sigma = residual[1]; }
      else
        ctx.Check(Difference(residual[1], first_sigma) > 1.e-4,
                  label + " reads changed sigma on curved physical boundaries");
    }
  }
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device(argc > 1 ? argv[1] : "cpu");
  ScopedEnvironment execution("IGRLAB_DG_PACKED_DENSE_EXECUTION", "rows");
  ScopedEnvironment precision("IGRLAB_SHARED_OPERATOR_PRECISION", "off");
  ScopedEnvironment layout("IGRLAB_SHARED_EULER_LAYOUT", "dense");
  ScopedEnvironment tensor_face("IGRLAB_DG_TENSOR_FACE", "off");
  ScopedEnvironment force_faces("IGRLAB_DG_FORCE_NLFORM_FACE", "0");
  ScopedEnvironment packed_mass("IGRLAB_DG_PACKED_INVMASS_DEVICE", "1");
  ScopedEnvironment sigma_storage("IGRLAB_WEAKDIV_SIGMA_STORAGE", "auto");
  ScopedEnvironment inverse_mass_reuse("IGRLAB_DG_INVMASS_ROW_REUSE", "0");
  ScopedEnvironment inverse_mass_reuse_alias("DG_IGR_DG_INVMASS_ROW_REUSE", nullptr);
  ScopedEnvironment tensor_face_reuse("IGRLAB_DG_TENSOR_FACE_ROW_REUSE", "0");
  ScopedEnvironment tensor_face_reuse_alias("DG_IGR_DG_TENSOR_FACE_ROW_REUSE", nullptr);
  TestContext ctx;
  const std::array<Fixture, 6> fixtures{{
    {Geometry::CurvedQ2, 5, mfem::Ordering::byNODES, 5,
      mfem::BasisType::GaussLegendre, "Q2-p5-byNODES-exact-sigma"},
    {Geometry::CurvedQ2, 2, mfem::Ordering::byVDIM, 2,
      mfem::BasisType::GaussLobatto, "Q2-p2-byVDIM-interpolated-sigma"},
    {Geometry::Straight, 5, mfem::Ordering::byVDIM, 5,
      mfem::BasisType::GaussLegendre, "straight-p5-byVDIM-exact-sigma"},
    {Geometry::Shear, 2, mfem::Ordering::byNODES, 1,
      mfem::BasisType::GaussLobatto, "shear-p2-byNODES-interpolated-sigma"},
    {Geometry::CurvedQ2, 1, mfem::Ordering::byVDIM, 1,
      mfem::BasisType::GaussLegendre, "Q2-p1-byVDIM-exact-sigma"},
    {Geometry::CurvedQ3, 7, mfem::Ordering::byNODES, 7,
      mfem::BasisType::GaussLegendre, "Q3-p7-byNODES-exact-sigma"}
  }};
  for (std::size_t i = 0; i < fixtures.size(); ++i)
    CheckFixture(ctx, fixtures[i], i == 0);
  CheckCurvedTensorFaces(ctx);
  return ctx.failures == 0 ? 0 : 1;
}

#include "igrlab/compat.hpp"
#include "igrlab/environment.hpp"
#include "mfem.hpp"
#include "numerics/limiters/zhang_shu.hpp"
#include "physics/euler/eos.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace
{
constexpr int kDim = 2;
constexpr int kNumEq = 4;
constexpr int kOrder = 5;
constexpr int kQuadratureOrder = 2 * kOrder + 2;
constexpr int kBisectionIterations = 12;
constexpr double kGamma = 1.4;
constexpr double kRhoMin = 5.0e-7;
constexpr double kPressureMin = 1.0e-6;

enum class Case { Noop, Density, Pressure, Sequential, NearFloors,
                  InvalidMean, NonfinitePressure };

struct LimiterVariant
{
  bool cooperative;
  bool fused_face;
  bool reference_dedup;
};

// Cycle each representation off/on/off on one FES, then enable both. The last
// case checks that the serial selector ignores both representation requests.
constexpr LimiterVariant kLimiterVariants[] = {
  {false, false, false}, {true, false, false}, {true, true, false},
  {true, false, false}, {true, false, true}, {true, false, false},
  {true, true, true}, {false, true, true}};

void SelectVariant(const LimiterVariant& variant)
{
  MFEM_VERIFY(setenv("IGRLAB_LIMITER_COOPERATIVE",
                      variant.cooperative ? "1" : "0", 1) == 0 &&
                setenv("IGRLAB_LIMITER_FUSED_FACE",
                       variant.fused_face ? "1" : "0", 1) == 0 &&
                setenv("IGRLAB_LIMITER_REFERENCE_DEDUP",
                       variant.reference_dedup ? "1" : "0", 1) == 0,
              "Cannot select p5 limiter implementation.");
}

const char* CaseName(Case kind)
{
  switch (kind)
  {
  case Case::Noop: return "noop";
  case Case::Density: return "density";
  case Case::Pressure: return "pressure";
  case Case::Sequential: return "sequential";
  case Case::NearFloors: return "near-floors";
  case Case::InvalidMean: return "invalid-mean";
  case Case::NonfinitePressure: return "nonfinite-pressure";
  }
  return "unknown";
}

Case ParseCase(const std::string& value)
{
  for (Case kind : {Case::Noop, Case::Density, Case::Pressure, Case::Sequential})
  {
    if (value == CaseName(kind)) { return kind; }
  }
  MFEM_ABORT("Benchmark case must be noop, density, pressure, or sequential.");
  return Case::Noop;
}

double Pressure(const mfem::Vector& value)
{
  return mfem::EulerPressureFromConservative(value, kDim, kGamma, 1.0e-16);
}

void MapQ2Mesh(mfem::Mesh& mesh)
{
  mesh.SetCurvature(2, false, kDim, mfem::Ordering::byNODES);
  mfem::GridFunction& nodes = *mesh.GetNodes();
  const int ndofs = nodes.FESpace()->GetNDofs();
  double* coordinates = nodes.HostReadWrite();
  for (int d = 0; d < ndofs; ++d)
  {
    const double x = coordinates[d];
    const double y = coordinates[d + ndofs];
    coordinates[d] = x + 0.2 * x * (1.0 - x) * (2.0 * y - 1.0);
    coordinates[d + ndofs] =
      y - 0.15 * y * (1.0 - y) * (2.0 * x - 1.0);
  }
}

void RotateQuadrilateralVertices(mfem::Mesh& mesh)
{
  // Match the permutation construction in test_dgsem_operator.cpp: rotate
  // neighboring cells differently while preserving their positive orientation.
  for (int e = 0; e < mesh.GetNE(); ++e)
  {
    int* vertices = mesh.GetElement(e)->GetVertices();
    int original[4];
    std::copy_n(vertices, 4, original);
    const int shift = 1 + e % 3;
    for (int v = 0; v < 4; ++v) { vertices[v] = original[(v + shift) % 4]; }
  }
  mesh.FinalizeTopology();
  mesh.Finalize(/*refine=*/false, /*fix_orientation=*/false);
}

void SetInitial(mfem::ParFiniteElementSpace& fes,
                mfem::ParGridFunction& state,
                Case kind)
{
  state = 0.0;
  mfem::Array<int> vdofs;
  double* data = state.HostReadWrite();
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementVDofs(e, vdofs);
    const mfem::IntegrationRule& nodes = fes.GetFE(e)->GetNodes();
    const int dof = fes.GetFE(e)->GetDof();
    // Rejection fixtures keep all other ranks/cells actively limitable.  This
    // checks that no good cell commits before the collective reject decision.
    const bool reject = (kind == Case::InvalidMean ||
                         kind == Case::NonfinitePressure) &&
                        mfem::Mpi::WorldRank() == 0 && e == 0;
    const Case local_kind = (kind == Case::InvalidMean ||
                             kind == Case::NonfinitePressure) && !reject
                              ? Case::Sequential : kind;
    for (int d = 0; d < dof; ++d)
    {
      const double x = nodes.IntPoint(d).x - 0.5;
      double value[kNumEq] = {1.0, 0.0, 0.0, 2.5};
      if (local_kind == Case::Density || local_kind == Case::Sequential)
      {
        value[0] = 1.0 + 3.0 * x;
      }
      if (local_kind == Case::Pressure) { value[1] = 6.0 * x; }
      if (local_kind == Case::Sequential) { value[1] = -3.0 * x; }
      if (local_kind == Case::NearFloors)
      {
        value[0] = kRhoMin * (1.0 + 1.0e-8);
        value[3] = kPressureMin * (1.0 + 1.0e-8) / (kGamma - 1.0);
      }
      if (local_kind == Case::InvalidMean) { value[0] = -1.0; }
      if (local_kind == Case::NonfinitePressure)
      {
        const double maximum = std::numeric_limits<double>::max();
        value[1] = 4.0 * std::sqrt(maximum) * x;
        value[3] = 0.1 * maximum;
      }
      for (int c = 0; c < kNumEq; ++c)
      {
        data[vdofs[d + dof * c]] = value[c];
      }
    }
  }
  state.UseDevice(true);
}

void Reset(const mfem::Vector& initial, mfem::Vector& state)
{
  const int size = initial.Size();
  const double* source = initial.Read();
  double* target = state.Write();
  mfem::forall(size, [=] MFEM_HOST_DEVICE(int i) { target[i] = source[i]; });
  MFEM_DEVICE_SYNC;
}

void Apply(mfem::ParFiniteElementSpace& fes,
           mfem::ParGridFunction& state,
           mfem::ZhangShuLimiterElementDiagnostics* diagnostics,
           bool& modified,
           mfem::ZhangShuPointSet point_set = mfem::ZhangShuPointSet::kConsumer,
           bool vtk_points = false)
{
  mfem::ApplyZhangShuPositivityLimiter(
    fes, state, kDim, kGamma, kRhoMin, kPressureMin, kQuadratureOrder,
    {2 * kOrder + 1}, kBisectionIterations, point_set, diagnostics,
    vtk_points, &modified);
}

mfem::Vector ElementValues(mfem::ParFiniteElementSpace& fes,
                           const mfem::ParGridFunction& state,
                           int e)
{
  mfem::Array<int> vdofs;
  mfem::Vector values;
  fes.GetElementVDofs(e, vdofs);
  state.GetSubVector(vdofs, values);
  return values;
}

mfem::Vector Mean(mfem::ParFiniteElementSpace& fes,
                   const mfem::ParGridFunction& state,
                   int e,
                   bool gll)
{
  const mfem::FiniteElement& fe = *fes.GetFE(e);
  mfem::ElementTransformation& transformation = *fes.GetElementTransformation(e);
  static mfem::IntegrationRules gll_rules(0, mfem::Quadrature1D::GaussLobatto);
  const mfem::IntegrationRule& rule = gll
    ? gll_rules.Get(fe.GetGeomType(), 2 * kOrder - 1)
    : mfem::IntRules.Get(fe.GetGeomType(), kQuadratureOrder);
  mfem::Vector values = ElementValues(fes, state, e);
  mfem::DenseMatrix matrix(values.GetData(), fe.GetDof(), kNumEq);
  mfem::Vector shape(fe.GetDof()), value(kNumEq), mean(kNumEq);
  mean = 0.0;
  double measure = 0.0;
  for (int q = 0; q < rule.GetNPoints(); ++q)
  {
    const mfem::IntegrationPoint& point = rule.IntPoint(q);
    transformation.SetIntPoint(&point);
    fe.CalcShape(point, shape);
    matrix.MultTranspose(shape, value);
    const double weight = point.weight * transformation.Weight();
    MFEM_VERIFY(weight > 0.0 && std::isfinite(weight),
                "Mapped p5 fixture has an invalid quadrature measure.");
    mean.Add(weight, value);
    measure += weight;
  }
  mean /= measure;
  return mean;
}

void CheckFinitePoints(mfem::ParFiniteElementSpace& fes,
                       const mfem::ParGridFunction& state,
                       int e,
                       bool vtk_points,
                       bool enforce_nodes = true)
{
  const mfem::FiniteElement& fe = *fes.GetFE(e);
  mfem::Vector values = ElementValues(fes, state, e);
  mfem::DenseMatrix matrix(values.GetData(), fe.GetDof(), kNumEq);
  mfem::Vector shape(fe.GetDof()), value(kNumEq);
  const auto check = [&](const mfem::IntegrationPoint& point)
  {
    fe.CalcShape(point, shape);
    matrix.MultTranspose(shape, value);
    const double pressure = Pressure(value);
    for (int c = 0; c < kNumEq; ++c)
    {
      MFEM_VERIFY(std::isfinite(value(c)), "Non-finite limited p5 point.");
    }
    MFEM_VERIFY(std::isfinite(pressure) &&
                  value(0) >= kRhoMin * (1.0 - 1.0e-7) &&
                  pressure >= kPressureMin * (1.0 - 1.0e-7),
                "Limited p5 point does not satisfy the production floors.");
  };
  const mfem::IntegrationRule& volume =
    mfem::IntRules.Get(fe.GetGeomType(), kQuadratureOrder);
  for (int q = 0; q < volume.GetNPoints(); ++q) { check(volume.IntPoint(q)); }
  for (int order : {kQuadratureOrder, 2 * kOrder + 1})
  {
    const mfem::IntegrationRule& face =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, order);
    for (int q = 0; q < face.GetNPoints(); ++q)
    {
      const double x = face.IntPoint(q).x;
      mfem::IntegrationPoint point;
      point.Set2(x, 0.0); check(point);
      point.Set2(x, 1.0); check(point);
      point.Set2(0.0, x); check(point);
      point.Set2(1.0, x); check(point);
    }
  }
  if (enforce_nodes)
  {
    const mfem::IntegrationRule& nodes = fe.GetNodes();
    for (int q = 0; q < nodes.GetNPoints(); ++q) { check(nodes.IntPoint(q)); }
  }
  if (vtk_points)
  {
    const mfem::RefinedGeometry* refined =
      mfem::GlobGeometryRefiner.Refine(fe.GetGeomType(), kOrder, 1);
    for (int q = 0; q < refined->RefPts.GetNPoints(); ++q)
    {
      check(refined->RefPts.IntPoint(q));
    }
  }
}

void CheckTransformedFacePoints(mfem::ParFiniteElementSpace& fes,
                                const mfem::ParGridFunction& state,
                                int e)
{
  mfem::Mesh& mesh = *fes.GetMesh();
  const mfem::FiniteElement& fe = *fes.GetFE(e);
  mfem::Vector values = ElementValues(fes, state, e);
  mfem::DenseMatrix matrix(values.GetData(), fe.GetDof(), kNumEq);
  mfem::Vector shape(fe.GetDof()), value(kNumEq);
  // Evaluate actual transformed element-side points, including hanging-face
  // subintervals. Do not replace them with an assumed reference face rule.
  for (int face = 0; face < mesh.GetNumFaces(); ++face)
  {
    int e1 = -1, e2 = -1;
    mesh.GetFaceElements(face, &e1, &e2);
    if (e1 != e && e2 != e) { continue; }
    auto* transformation = mesh.GetFaceElementTransformations(face);
    if (transformation == nullptr) { continue; }
    for (int order : {kQuadratureOrder, 2 * kOrder + 1})
    {
      const auto& rule = mfem::IntRules.Get(transformation->GetGeometryType(), order);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
        transformation->SetAllIntPoints(&rule.IntPoint(q));
        const auto& point = e1 == e ? transformation->GetElement1IntPoint()
                                    : transformation->GetElement2IntPoint();
        fe.CalcShape(point, shape);
        matrix.MultTranspose(shape, value);
        for (int c = 0; c < kNumEq; ++c)
        {
          MFEM_VERIFY(std::isfinite(value(c)), "Non-finite transformed face state.");
        }
        MFEM_VERIFY(value(0) >= kRhoMin * (1.0 - 1.0e-7) &&
                      std::isfinite(Pressure(value)) &&
                      Pressure(value) >= kPressureMin * (1.0 - 1.0e-7),
                    "Transformed p5 face point violates a production floor.");
      }
    }
  }
}

int ActiveCount(const mfem::ZhangShuLimiterElementDiagnostics& diagnostics)
{
  int active = 0;
  const double* values = diagnostics.active.HostRead();
  for (int e = 0; e < diagnostics.active.Size(); ++e)
  {
    active += values[e] != 0.0 ? 1 : 0;
  }
  return active;
}

void CheckAccepted(mfem::ParFiniteElementSpace& fes,
                    const mfem::ParGridFunction& initial,
                    const mfem::ParGridFunction& state,
                    const mfem::ZhangShuLimiterElementDiagnostics& diagnostics,
                    bool modified,
                    Case kind,
                    bool gll,
                    bool vtk_points)
{
  const bool active = kind != Case::Noop && kind != Case::NearFloors;
  MFEM_VERIFY(modified == active &&
                ActiveCount(diagnostics) == (active ? fes.GetNE() : 0),
              "Limiter mutation flag/active diagnostics disagree with fixture.");
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    const mfem::Vector before = Mean(fes, initial, e, gll);
    const mfem::Vector after = Mean(fes, state, e, gll);
    MFEM_VERIFY(before(0) > kRhoMin && std::isfinite(Pressure(before)) &&
                  Pressure(before) > kPressureMin,
                "Active fixture requires an admissible physical cell mean.");
    for (int c = 0; c < kNumEq; ++c)
    {
      MFEM_VERIFY(std::abs(after(c) - before(c)) <=
                    4.0e-12 * std::max(1.0, std::abs(before(c))),
                  "p5 limiter changed a physical/discrete conserved mean.");
    }
    const double rho_theta = diagnostics.theta_rho.HostRead()[e];
    const double pressure_theta = diagnostics.theta_p.HostRead()[e];
    const bool density_active = kind == Case::Density || kind == Case::Sequential;
    const bool pressure_active = kind == Case::Pressure || kind == Case::Sequential;
    MFEM_VERIFY(rho_theta >= 0.0 && rho_theta <= 1.0 &&
                  pressure_theta >= 0.0 && pressure_theta <= 1.0 &&
                  (rho_theta < 1.0) == density_active &&
                  (pressure_theta < 1.0) == pressure_active,
                "p5 fixture did not activate the intended contractions.");
    MFEM_VERIFY(diagnostics.inadmissible_average.HostRead()[e] == 0.0 &&
                  diagnostics.collapsed_average.HostRead()[e] == 0.0,
                "Accepted p5 mean diagnostics are inconsistent.");
    const mfem::Vector original = ElementValues(fes, initial, e);
    const mfem::Vector limited = ElementValues(fes, state, e);
    const int dof = fes.GetFE(e)->GetDof();
    for (int c = 0; c < kNumEq; ++c)
    {
      MFEM_VERIFY(diagnostics.average_repair.HostRead()[e + fes.GetNE() * c] == 0.0,
                  "Strict p5 limiter unexpectedly repaired a mean.");
      for (int d = 0; d < dof; ++d)
      {
        double expected = original[d + dof * c];
        if (rho_theta < 1.0)
        {
          expected = before(c) + rho_theta * (expected - before(c));
        }
        if (pressure_theta < 1.0)
        {
          expected = before(c) + pressure_theta * (expected - before(c));
        }
        MFEM_VERIFY(std::abs(expected - limited[d + dof * c]) <=
                      6.0e-12 * std::max(1.0, std::abs(expected)),
                    "p5 nodal state is not the sequential mean contraction.");
      }
    }
    CheckFinitePoints(fes, state, e, vtk_points);
  }
  if (!active)
  {
    MFEM_VERIFY(std::memcmp(initial.HostRead(), state.HostRead(),
                            initial.Size() * sizeof(double)) == 0,
                "Inactive p5 limiter changed state bits.");
  }
}

void CompareVector(const mfem::Vector& reference, const mfem::Vector& candidate)
{
  MFEM_VERIFY(reference.Size() == candidate.Size(), "p5 comparison size mismatch.");
  const double* a = reference.HostRead();
  const double* b = candidate.HostRead();
  for (int i = 0; i < reference.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]) &&
                  std::abs(a[i] - b[i]) <=
                    2.0e-12 * std::max({1.0, std::abs(a[i]), std::abs(b[i])}),
                "Serial/cooperative/fused-face p5 limiter values differ.");
  }
}

void CompareDiagnostics(
  const mfem::ZhangShuLimiterElementDiagnostics& reference,
  const mfem::ZhangShuLimiterElementDiagnostics& candidate)
{
  CompareVector(reference.active, candidate.active);
  CompareVector(reference.theta_rho, candidate.theta_rho);
  CompareVector(reference.theta_p, candidate.theta_p);
  CompareVector(reference.inadmissible_average, candidate.inadmissible_average);
  CompareVector(reference.collapsed_average, candidate.collapsed_average);
  CompareVector(reference.average_repair, candidate.average_repair);
}

void PointPolicyParity(mfem::ParFiniteElementSpace& fes,
                        bool gll,
                        const char* topology)
{
  mfem::ParGridFunction initial(&fes), state(&fes), reference(&fes);
  initial = 0.0;
  double* data = initial.HostReadWrite();
  mfem::Array<int> vdofs;
  double target = 0.0;
  const mfem::IntegrationRule& reference_nodes = fes.GetFE(0)->GetNodes();
  for (int q = 0; q < reference_nodes.GetNPoints(); ++q)
  {
    const double x = reference_nodes.IntPoint(q).x;
    if (x > 0.2 && x < 0.5) { target = x; break; }
  }
  MFEM_VERIFY(target > 0.2, "p5 point-policy fixture has no target node.");
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    fes.GetElementVDofs(e, vdofs);
    const mfem::IntegrationRule& nodes = fes.GetFE(e)->GetNodes();
    const int dof = fes.GetFE(e)->GetDof();
    for (int d = 0; d < dof; ++d)
    {
      const double x = nodes.IntPoint(d).x - target;
      const double y = nodes.IntPoint(d).y - target;
      data[vdofs[d]] = 1.0;
      data[vdofs[d + 3 * dof]] = (x * x + y * y - 0.001) / (kGamma - 1.0);
    }
  }
  initial.UseDevice(true);
  for (int policy = 0; policy < 3; ++policy)
  {
    const bool consumer = policy == 2;
    const bool vtk = policy == 1;
    const bool expected_active = gll || consumer || vtk;
    mfem::ZhangShuLimiterElementDiagnostics reference_diagnostics;
    bool have_reference = false;
    for (const auto& variant : kLimiterVariants)
    {
      SelectVariant(variant);
      Reset(initial, state);
      bool modified = false;
      mfem::ZhangShuLimiterElementDiagnostics diagnostics;
      Apply(fes, state, &diagnostics, modified,
            consumer ? mfem::ZhangShuPointSet::kConsumer
                     : mfem::ZhangShuPointSet::kQuadrature, vtk);
      MFEM_VERIFY(modified == expected_active &&
                    ActiveCount(diagnostics) == (expected_active ? fes.GetNE() : 0),
                  "p5 GL/GLL/VTK positivity-point policy changed.");
      for (int e = 0; e < fes.GetNE(); ++e)
      {
        CheckFinitePoints(fes, state, e, vtk, gll || consumer);
        CheckTransformedFacePoints(fes, state, e);
      }
      if (!expected_active)
      {
        MFEM_VERIFY(std::memcmp(initial.HostRead(), state.HostRead(),
                                initial.Size() * sizeof(double)) == 0,
                    "Quadrature-only GL p5 policy unexpectedly changed FE nodes.");
      }
      if (!have_reference)
      {
        reference = state;
        reference_diagnostics = diagnostics;
        have_reference = true;
      }
      else
      {
        CompareVector(reference, state);
        CompareDiagnostics(reference_diagnostics, diagnostics);
      }
    }
  }
  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] p5 mapped-Q2 " << (gll ? "GLL" : "GL")
              << " topology=" << topology
              << " quadrature/consumer/VTK point-policy parity\n";
  }
}

void Correctness(bool gll, bool permuted = false, bool nonconforming = false)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    permuted ? 3 : 4, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  if (permuted) { RotateQuadrilateralVertices(serial); }
  if (nonconforming)
  {
    mfem::Array<int> refine(1);
    refine[0] = 0;
    serial.GeneralRefinement(refine, /*nonconforming=*/1);
    MFEM_VERIFY(serial.Nonconforming(), "Expected a hanging-face GL fixture.");
  }
  MapQ2Mesh(serial);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::DG_FECollection fec(kOrder, kDim,
    gll ? mfem::BasisType::GaussLobatto : mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, kNumEq, mfem::Ordering::byNODES);
  MFEM_VERIFY(!gll || mfem::FaceQuadratureInterpolator::SupportsFESpace(fes),
              "p5 GLL face parity fixture must admit MFEM face interpolation.");
  const char* topology = nonconforming ? "rotated-nc"
                         : (permuted ? "rotated" : "standard");
  PointPolicyParity(fes, gll, topology);
  mfem::ParGridFunction initial(&fes), state(&fes), reference(&fes);
  for (Case kind : {Case::Noop, Case::Density, Case::Pressure, Case::Sequential,
                    Case::NearFloors, Case::InvalidMean, Case::NonfinitePressure})
  {
    SetInitial(fes, initial, kind);
    mfem::ZhangShuLimiterElementDiagnostics reference_diagnostics;
    const bool rejected = kind == Case::InvalidMean || kind == Case::NonfinitePressure;
    for (bool vtk_points : {false, true})
    {
      bool have_reference = false;
      for (const auto& variant : kLimiterVariants)
      {
        SelectVariant(variant);
        Reset(initial, state);
        mfem::ZhangShuLimiterElementDiagnostics diagnostics;
        bool modified = true;
        int rejected_count = 0;
        try { Apply(fes, state, &diagnostics, modified,
                    mfem::ZhangShuPointSet::kConsumer, vtk_points); }
        catch (const mfem::RecoverableLimiterFailure& failure)
        {
          rejected_count = failure.GlobalRejectedElementCount();
        }
        if (rejected)
        {
          MFEM_VERIFY(rejected_count == 1 && !modified &&
                        std::memcmp(initial.HostRead(), state.HostRead(),
                                    initial.Size() * sizeof(double)) == 0,
                      "p5 rejection was not collective and transactional.");
          if (kind == Case::NonfinitePressure && mfem::Mpi::WorldRank() == 0)
          {
            const mfem::Vector mean = Mean(fes, initial, 0, gll);
            MFEM_VERIFY(std::isfinite(mean(0)) && mean(0) > kRhoMin &&
                          std::isfinite(Pressure(mean)) && Pressure(mean) > kPressureMin &&
                          diagnostics.inadmissible_average.HostRead()[0] == 0.0,
                        "Nonfinite p5 fixture must reject its consumer points, not its mean.");
          }
        }
        else
        {
          MFEM_VERIFY(rejected_count == 0, "Admissible p5 fixture was rejected.");
          CheckAccepted(fes, initial, state, diagnostics, modified, kind, gll, vtk_points);
          for (int e = 0; e < fes.GetNE(); ++e)
          {
            CheckTransformedFacePoints(fes, state, e);
          }
        }
        if (!have_reference)
        {
          reference = state;
          reference_diagnostics = diagnostics;
          have_reference = true;
        }
        else
        {
          if (!rejected) { CompareVector(reference, state); }
          CompareDiagnostics(reference_diagnostics, diagnostics);
        }
      }
    }
    if (mfem::Mpi::Root())
    {
      std::cout << "[PASS] p5 mapped-Q2 " << (gll ? "GLL" : "GL")
                << " topology=" << topology << " case=" << CaseName(kind)
                << " serial/cooperative/fused-face/dedup native points, mean, mutation and rejection\n";
    }
  }
}

struct Sample { double wall_ms = 0.0; double event_ms = 0.0; };

void Benchmark(bool gll, Case kind, int nx, int calls)
{
  MFEM_VERIFY(mfem::Mpi::WorldSize() == 1,
              "The bounded throughput benchmark expects one MPI rank.");
  MFEM_VERIFY(nx > 0 && nx <= 512 && calls >= 20 && calls <= 100,
              "Benchmark bounds are nx in [1,512], calls in [20,100].");
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    nx, nx, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  MapQ2Mesh(serial);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial);
  mfem::DG_FECollection fec(kOrder, kDim,
    gll ? mfem::BasisType::GaussLobatto : mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, kNumEq, mfem::Ordering::byNODES);
  mfem::ParGridFunction initial(&fes), state(&fes);
  SetInitial(fes, initial, kind);
  state.UseDevice(true);
  const bool cuda = mfem::Device::Allows(mfem::Backend::CUDA_MASK);
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  cudaEvent_t begin{}, end{};
  if (cuda)
  {
    MFEM_GPU_CHECK(cudaEventCreate(&begin));
    MFEM_GPU_CHECK(cudaEventCreate(&end));
  }
#else
  MFEM_VERIFY(!cuda, "CUDA benchmark needs a CUDA translation unit.");
#endif
  bool modified = false;
  const auto sample = [&]()
  {
    Reset(initial, state);
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
    if (cuda) { MFEM_GPU_CHECK(cudaEventRecord(begin)); }
#endif
    const auto start = std::chrono::steady_clock::now();
    Apply(fes, state, nullptr, modified);
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
    if (cuda)
    {
      MFEM_GPU_CHECK(cudaEventRecord(end));
      MFEM_GPU_CHECK(cudaEventSynchronize(end));
    }
#endif
    MFEM_DEVICE_SYNC;
    Sample result;
    result.wall_ms = 1.0e3 * std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
    if (cuda)
    {
      float elapsed = 0.0f;
      MFEM_GPU_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
      result.event_ms = elapsed;
    }
#endif
    MFEM_VERIFY(modified == (kind != Case::Noop),
                "Benchmark fixture's actual mutation schedule is incorrect.");
    return result;
  };
  const Sample cold = sample();
  constexpr int warmups = 4;
  for (int i = 0; i < warmups; ++i) { sample(); }
  std::vector<double> wall, events;
  for (int i = 0; i < calls; ++i)
  {
    const Sample result = sample();
    wall.push_back(result.wall_ms);
    events.push_back(result.event_ms);
  }
  // Diagnostic capture and native representative-cell conservation are outside
  // all timed intervals. Timed calls request the production mutation result.
  Reset(initial, state);
  mfem::ZhangShuLimiterElementDiagnostics diagnostics;
  Apply(fes, state, &diagnostics, modified);
  const int active = ActiveCount(diagnostics);
  MFEM_VERIFY(active == (kind == Case::Noop ? 0 : fes.GetNE()),
              "Benchmark fixture active-element count is incorrect.");
  for (int e : {0, fes.GetNE() / 2, fes.GetNE() - 1})
  {
    const mfem::Vector before = Mean(fes, initial, e, gll);
    const mfem::Vector after = Mean(fes, state, e, gll);
    MFEM_VERIFY(before(0) > kRhoMin && std::isfinite(Pressure(before)) &&
                  Pressure(before) > kPressureMin,
                "Benchmark fixture must have an admissible physical mean.");
    for (int c = 0; c < kNumEq; ++c)
    {
      MFEM_VERIFY(std::abs(before(c) - after(c)) <=
                    4.0e-12 * std::max(1.0, std::abs(before(c))),
                  "Benchmark limiter changed a conserved mean.");
    }
    CheckFinitePoints(fes, state, e, false);
  }
  const double wall_mean = std::accumulate(wall.begin(), wall.end(), 0.0) / calls;
  const double event_mean = std::accumulate(events.begin(), events.end(), 0.0) / calls;
  std::sort(wall.begin(), wall.end());
  std::sort(events.begin(), events.end());
  const char* variant = ::igrlab::GetEnv("DG_IGR_LIMITER_COOPERATIVE");
  const char* fused_face = ::igrlab::GetEnv("DG_IGR_LIMITER_FUSED_FACE");
  const char* reference_dedup = ::igrlab::GetEnv("DG_IGR_LIMITER_REFERENCE_DEDUP");
  std::cout << std::setprecision(17)
            << "[limiter-throughput] backend=" << (cuda ? "cuda" : "cpu")
            << " basis=" << (gll ? "GLL" : "GL") << " order=" << kOrder
            << " mesh=mapped-Q2 nx=" << nx << " ny=" << nx
            << " elements=" << fes.GetNE() << " case=" << CaseName(kind)
            << " cooperative=" << (variant ? variant : "default")
            << " fused_face_requested=" << (fused_face ? fused_face : "default")
            << " reference_dedup_requested="
            << (reference_dedup ? reference_dedup : "default")
            << " rho_min=" << kRhoMin << " p_min=" << kPressureMin
            << " bisections=" << kBisectionIterations
            << " volume_order=" << kQuadratureOrder << " face_orders=12,11"
            << " point_set=consumer vtk_points=0 warmups=" << warmups
            << " calls=" << calls << " active_elements=" << active
            << " cold_wall_ms=" << cold.wall_ms << " cold_event_ms=" << cold.event_ms
            << " mean_wall_ms=" << wall_mean << " median_wall_ms=" << wall[calls / 2]
            << " mean_event_ms=" << event_mean << " median_event_ms=" << events[calls / 2]
            << " reset_in_timing=0 diagnostic_capture_in_timing=0"
            << " scope=complete-production-limiter-including-admission-and-mutation-result\n";
#if defined(MFEM_USE_CUDA) && defined(__CUDACC__)
  if (cuda)
  {
    MFEM_GPU_CHECK(cudaEventDestroy(begin));
    MFEM_GPU_CHECK(cudaEventDestroy(end));
  }
#endif
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const std::string backend = argc > 1 ? argv[1] : "cpu";
  MFEM_VERIFY(backend == "cpu" || backend == "cuda", "Expected cpu or cuda.");
  mfem::Device device(backend.c_str());
  if (argc > 2)
  {
    MFEM_VERIFY(std::string(argv[2]) == "--benchmark" && argc >= 5 && argc <= 7,
                "Usage: [cpu|cuda] --benchmark [gl|gll] [noop|density|pressure|sequential] [nx=512] [calls=30].");
    const std::string basis = argv[3];
    MFEM_VERIFY(basis == "gl" || basis == "gll", "Expected gl or gll.");
    Benchmark(basis == "gll", ParseCase(argv[4]),
              argc > 5 ? std::stoi(argv[5]) : 512,
              argc > 6 ? std::stoi(argv[6]) : 30);
    return 0;
  }
  // Correctness must actually select the admitted GLL face path regardless of
  // an inherited optimization-off environment. Benchmark settings are retained.
  MFEM_VERIFY(setenv("IGRLAB_LIMITER_MFEM_FACE_QI", "1", 1) == 0,
              "Cannot enable MFEM face interpolation for parity checks.");
  Correctness(false);
  Correctness(true);
  Correctness(false, /*permuted=*/true);
  Correctness(false, /*permuted=*/true, /*nonconforming=*/true);
  return 0;
}

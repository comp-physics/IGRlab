#include "igrlab/compat.hpp"
#include "numerics/shared_reference_map.hpp"
#include "numerics/dg/DGHyperbolicConservationLaws.hpp"
#include "numerics/dgsem/dgsem_basis.hpp"
#include "numerics/dgsem/dgsem_device.hpp"
#include "numerics/riemann/rusanov.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using mfem::DenseMatrix;
using mfem::Vector;
using igrlab::shared_reference::Map;
using igrlab::shared_reference::Precision;

double AlgebraTolerance(Precision precision)
{
  return precision == Precision::FP64 ? 2e-13 :
         precision == Precision::FP32 ? 3e-6 : 2e-3;
}

void MatrixOracle(int m, int k, int columns, Precision precision)
{
  DenseMatrix matrix(m, k), saved(m, k);
  for (int j = 0; j < k; ++j)
  {
    for (int i = 0; i < m; ++i)
    {
      matrix(i, j) = std::sin(0.31 * (i + 1) + 0.19 * j) / (j + 1);
    }
  }
  saved = matrix;
  Map map;
  map.Configure(matrix, columns, precision, "fixture-matrix");
  matrix = 17.0; // the fixed map must own its immutable matrix snapshot
  const auto* input_address = map.Input().Read();
  const auto* output_address = map.Output().Read();
  for (int phase = 0; phase < 3; ++phase)
  {
    const double scale = std::ldexp(1.0, phase == 0 ? 0 : phase == 1 ? -80 : 80);
    double* b = map.Input().HostWrite();
    for (int c = 0; c < columns; ++c)
    {
      for (int j = 0; j < k; ++j)
      {
        b[j + k * c] = scale * std::cos(0.13 * (c + 1) - 0.17 * j + phase);
      }
    }
    map.Apply();
    const double* result = map.Output().HostRead();
    long double max_error = 0.0L, max_bound = 0.0L, maximum = 0.0L;
    for (int c = 0; c < columns; ++c)
    {
      for (int i = 0; i < m; ++i)
      {
        long double oracle = 0.0L, bound = 0.0L;
        for (int j = 0; j < k; ++j)
        {
          const long double product = static_cast<long double>(saved(i, j)) * b[j + k * c];
          oracle += product; bound += std::abs(product);
        }
        MFEM_VERIFY(std::isfinite(result[i + m * c]), "Nonfinite shared matrix result.");
        max_error = std::max(max_error, std::abs(result[i + m * c] - oracle));
        max_bound = std::max(max_bound, bound);
        maximum = std::max(maximum, std::abs(oracle));
      }
    }
    MFEM_VERIFY(maximum > 0.01L * scale && max_bound > 0.0L &&
                  max_error <= AlgebraTolerance(precision) * max_bound,
                "Shared matrix failed independent long-double oracle: m=" << m
                << " k=" << k << " n=" << columns << " phase=" << phase
                << " relative_error=" << static_cast<double>(max_error / max_bound));
    MFEM_VERIFY(input_address == map.Input().Read() && output_address == map.Output().Read(),
                "Shared map resized a live workspace.");
  }
  double* b = map.Input().HostWrite();
  std::fill(b, b + k * columns, 0.0);
  map.Apply();
  const double* result = map.Output().HostRead();
  for (int i = 0; i < m * columns; ++i)
  {
    MFEM_VERIFY(result[i] == 0.0, "Zero shared matrix input did not produce exact zero.");
  }
  b = map.Input().HostWrite();
  b[0] = precision == Precision::FP64 ? std::numeric_limits<double>::quiet_NaN() : 1e100;
  map.Apply();
  result = map.Output().HostRead();
  bool nonfinite = false;
  for (int i = 0; i < m; ++i) { nonfinite = nonfinite || !std::isfinite(result[i]); }
  MFEM_VERIFY(nonfinite, "Shared map hid nonfinite/FP32-overflow input instead of exposing failure.");
}

void TensorLineOracle(int order, Precision precision, bool sheared)
{
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(
    1,1,mfem::Element::QUADRILATERAL,false,1.0,1.0,false);
  if (sheared)
    for (int v = 0; v < mesh.GetNV(); ++v)
    {
      double* point = mesh.GetVertex(v);
      point[0] = 0.75*point[0]+0.25*point[1]; point[1] *= 1.25;
    }
  mfem::L2_FECollection fec(order,2,mfem::BasisType::GaussLegendre);
  const auto& fe = *fec.FiniteElementForGeometry(mfem::Geometry::SQUARE);
  auto* transformation = mesh.GetElementTransformation(0);
  mfem::TransposeIntegrator weak(new mfem::GradientIntegrator());
  DenseMatrix assembled;
  weak.AssembleElementMatrix2(fe,fe,*transformation,assembled);
  const int dof = fe.GetDof(), k = 2*dof, columns = k+3;
  DenseMatrix matrix(dof,k), saved;
  for (int i = 0; i < dof; ++i) for (int j = 0; j < dof; ++j)
    for (int d = 0; d < 2; ++d) { matrix(i,2*j+d) = assembled(i,d*dof+j); }
  saved = matrix;
  transformation->SetIntPoint(&mfem::Geometries.GetCenter(mfem::Geometry::SQUARE));
  DenseMatrix adjugate(2); mfem::CalcAdjugate(transformation->Jacobian(),adjugate);
  Map map;
  map.ConfigureDGTensorLines(matrix,columns,precision,fe,adjugate);
  matrix = 17.0;
  const auto* input_address = map.Input().Read();
  const auto* output_address = map.Output().Read();
  for (int phase = 0; phase < 3; ++phase)
  {
    const double scale = std::ldexp(1.0,phase == 0 ? 0 : phase == 1 ? -80 : 80);
    double* input = map.Input().HostWrite();
    for (int column = 0; column < columns; ++column) for (int j = 0; j < k; ++j)
      input[j+k*column] = phase == 0 && column < k ? (column == j ? scale : 0.0) :
        scale*std::cos(0.31*j-0.17*column+phase);
    map.Apply();
    const double* output = map.Output().HostRead();
    long double error = 0.0L, bound = 0.0L;
    for (int column = 0; column < columns; ++column) for (int i = 0; i < dof; ++i)
    {
      long double expected = 0.0L, row_bound = 0.0L;
      for (int j = 0; j < k; ++j)
      {
        const long double product = static_cast<long double>(saved(i,j))*input[j+k*column];
        expected += product; row_bound += std::abs(product);
      }
      MFEM_VERIFY(std::isfinite(output[i+dof*column]),"Nonfinite DG tensor-line result.");
      error = std::max(error,std::abs(output[i+dof*column]-expected));
      bound = std::max(bound,row_bound);
    }
    const double tolerance = precision == Precision::FP64 ? 2e-12 : AlgebraTolerance(precision);
    MFEM_VERIFY(bound > 0.0L && error <= tolerance*bound,
      "DG tensor lines failed independent assembled-matrix oracle: p=" << order
      << " phase=" << phase << " sheared=" << sheared << " relative="
      << static_cast<double>(error/bound));
    MFEM_VERIFY(input_address == map.Input().Read() && output_address == map.Output().Read(),
                "DG tensor lines resized a live workspace.");
  }
  double* input = map.Input().HostWrite(); std::fill(input,input+k*columns,0.0);
  map.Apply();
  const double* output = map.Output().HostRead();
  for (int i = 0; i < dof*columns; ++i)
    MFEM_VERIFY(output[i] == 0.0,"Zero DG tensor-line input did not produce exact zero.");
  input = map.Input().HostWrite();
  input[0] = precision == Precision::FP64 ? std::numeric_limits<double>::quiet_NaN() : 1e100;
  map.Apply(); output = map.Output().HostRead();
  bool nonfinite = false;
  for (int i = 0; i < dof; ++i) { nonfinite = nonfinite || !std::isfinite(output[i]); }
  MFEM_VERIFY(nonfinite,"DG tensor lines hid nonfinite/FP32-overflow input.");
}

void CentralLineOracle(int order, Precision precision)
{
  const int n = order + 1, dof = n * n, ne = 5, rows = ne * dof;
  const auto& basis = mfem::dgsem::GetLGLBasis1D(order);
  mfem::Array<int> local(dof);
  for (int i = 0; i < dof; ++i) { local[i] = dof - 1 - i; }
  DenseMatrix flux(rows, 8), expected, actual;
  Vector row_sums;
  Map map;
  for (int phase = 0; phase < 3; ++phase)
  {
    double* f = flux.HostWrite();
    for (int c = 0; c < 8; ++c)
    {
      for (int e = 0; e < ne; ++e)
      {
        for (int lex = 0; lex < dof; ++lex)
        {
          // Phase zero is constant along each line: all reduced-precision
          // inputs are exactly zero, but the restored split form is nonzero.
          f[e * dof + local[lex] + rows * c] = 4.0 + 0.25 * c + 0.125 * e +
            (phase ? 0.02 * std::sin(0.17 * lex + 0.21 * c + phase) : 0.0);
        }
      }
    }
    mfem::dgsem::ApplyCentralVolumeDevice(2, n, local, basis.Dsplit,
        basis.weights, flux, expected);
    mfem::dgsem::ApplySharedCentralVolumeDevice(n, local, basis.Dsplit,
        basis.weights, flux, actual, map, row_sums, precision);
    const double* a = actual.HostRead(); const double* b = expected.HostRead();
    double error = 0.0, scale = 0.0;
    for (int i = 0; i < rows * 4; ++i)
    {
      MFEM_VERIFY(std::isfinite(a[i]), "Nonfinite shared central line result.");
      error = std::max(error, std::abs(a[i] - b[i]));
      scale = std::max(scale, std::abs(b[i]));
    }
    const double tolerance = phase == 0 ? 2e-13 : AlgebraTolerance(precision);
    MFEM_VERIFY(scale > 0.0 && error <= tolerance * scale,
                "Shared DGSEM lines differ from original symmetric split flux: p="
                << order << " phase=" << phase << " error=" << error / scale);
  }
}

void FillEuler(mfem::ParFiniteElementSpace& fes, Vector& state,
               mfem::ParGridFunction& sigma, int phase)
{
  double* u = state.HostWrite(); double* s = sigma.HostWrite();
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs, sdofs;
    fes.GetElementDofs(e, dofs); sigma.ParFESpace()->GetElementDofs(e, sdofs);
    const auto& nodes = fes.GetFE(e)->GetNodes();
    for (int j = 0; j < dofs.Size(); ++j)
    {
      const auto& ip = nodes.IntPoint(j);
      const double rho = 1.0 + 0.01 * phase * (e + ip.x * ip.y);
      const double vx = 0.23 + 0.02 * phase * ip.x;
      const double vy = -0.11 + 0.01 * phase * ip.y;
      const double p = 1.0 + 0.02 * phase * ip.x;
      const double values[4] = {rho, rho * vx, rho * vy,
        p / 0.4 + 0.5 * rho * (vx * vx + vy * vy)};
      for (int c = 0; c < 4; ++c) { u[fes.DofToVDof(dofs[j], c)] = values[c]; }
      s[sdofs[j]] = 0.02 + 0.01 * phase * (ip.x + ip.y);
    }
  }
}

void DGProductionOracle(int order, Precision precision, const char* layout = "dense",
                        bool sheared = false)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    4, 4, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<Vector> translations = {Vector({1.0, 0.0}), Vector({0.0, 1.0})};
  if (sheared)
  {
    for (int v = 0; v < serial.GetNV(); ++v)
    {
      double* point = serial.GetVertex(v);
      point[0] = 0.75*point[0]+0.25*point[1]; point[1] *= 1.25;
    }
    translations = {Vector({0.75,0.0}),Vector({0.25,1.25})};
  }
  mfem::Mesh periodic = mfem::Mesh::MakePeriodic(
    serial, serial.CreatePeriodicVertexMapping(translations));
  mfem::ParMesh mesh(MPI_COMM_SELF, periodic);
  mfem::L2_FECollection fec(order, 2, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::ParFiniteElementSpace sfes(&mesh, &fec);
  mfem::ParGridFunction sigma(&sfes);
  mfem::EulerFlux euler(2, 1.4);
  mfem::FluxWithIGR physical(euler, 1.4);
  physical.SetSigmaField(&sigma);
  physical.SetInvariantDomainFlux(true);
  physical.SetSigmaAwareCFL(true); physical.SetSigmaAwareRusanov(true);
  mfem::RusanovFluxWithSides flux(physical);
  setenv("IGRLAB_SHARED_EULER_LAYOUT", "dense", 1);
  setenv("IGRLAB_SHARED_OPERATOR_PRECISION", "off", 1);
  mfem::DGHyperbolicConservationLaws baseline(fes, flux, 2, true);
  const bool tensor_lines = std::string(layout) == "tensor-lines";
  // Build an independent full-storage control before enabling the constructor
  // opt-in. Both controls subsequently execute the identical tensor-line map.
  std::unique_ptr<mfem::DGHyperbolicConservationLaws> full_setup;
  if (tensor_lines)
    full_setup = std::make_unique<mfem::DGHyperbolicConservationLaws>(fes,flux,2,true);
  setenv("IGRLAB_SHARED_EULER_LAYOUT", layout, 1);
  setenv("IGRLAB_SHARED_OPERATOR_PRECISION", igrlab::shared_reference::PrecisionName(precision), 1);
  mfem::DGHyperbolicConservationLaws candidate(fes, flux, 2, true);
  baseline.SetDGSEMGamma(1.4); candidate.SetDGSEMGamma(1.4);
  baseline.SetInvariantDomainVolumeSpeedTracking(true);
  candidate.SetInvariantDomainVolumeSpeedTracking(true);
  if (full_setup)
  {
    full_setup->SetDGSEMGamma(1.4);
    full_setup->SetInvariantDomainVolumeSpeedTracking(true);
  }
  Vector state(fes.GetVSize()), expected(fes.GetVSize()), actual(fes.GetVSize());
  Vector full_result(fes.GetVSize());
  // Assemble the physical mass independently of the production inverse and
  // its cached correction weights. The conserved cell moment is (M 1)^T du/dt,
  // not an unweighted sum of the strong nodal residual.
  std::vector<Vector> cell_mass(mesh.GetNE());
  mfem::MassIntegrator mass_integrator;
  for (int e = 0; e < mesh.GetNE(); ++e)
  {
    DenseMatrix mass;
    const auto& fe = *fes.GetFE(e);
    mass_integrator.AssembleElementMatrix(fe, *fes.GetElementTransformation(e), mass);
    Vector one(fe.GetDof());
    one = 1.0;
    cell_mass[e].SetSize(fe.GetDof());
    mass.Mult(one, cell_mass[e]);
    for (int i = 0; i < fe.GetDof(); ++i)
    {
      MFEM_VERIFY(std::isfinite(cell_mass[e][i]) && cell_mass[e][i] > 0.0,
                  "Invalid independently assembled cell mass.");
    }
  }
  for (int phase = 0; phase < 3; ++phase)
  {
    FillEuler(fes, state, sigma, phase);
    setenv("IGRLAB_SHARED_EULER_LAYOUT", "dense", 1);
    setenv("IGRLAB_SHARED_OPERATOR_PRECISION", "off", 1);
    baseline.Mult(state, expected);
    if (phase == 2 && full_setup)
    {
      full_setup->Update(); // still dense/off: preserve its per-element setup
    }
    setenv("IGRLAB_SHARED_EULER_LAYOUT", layout, 1);
    setenv("IGRLAB_SHARED_OPERATOR_PRECISION", igrlab::shared_reference::PrecisionName(precision), 1);
    if (phase == 2 && full_setup) { candidate.Update(); }
    if (full_setup) { full_setup->Mult(state, full_result); }
    candidate.Mult(state, actual);
    MFEM_VERIFY(std::isfinite(candidate.GetMaxCharSpeed()) &&
                  candidate.GetMaxCharSpeed() == baseline.GetMaxCharSpeed(),
                "Shared derivative changed the FP64 characteristic-speed policy.");
    const double* a = actual.HostRead(); const double* b = expected.HostRead();
    const double* full = full_setup ? full_result.HostRead() : nullptr;
    double error = 0.0, scale = 0.0;
    double setup_error = 0.0, setup_scale = 0.0;
    for (int i = 0; i < state.Size(); ++i)
    {
      MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite DG production oracle.");
      error = std::max(error, std::abs(a[i] - b[i]));
      scale = std::max(scale, std::abs(b[i]));
      if (full)
      {
        MFEM_VERIFY(std::isfinite(full[i]), "Nonfinite full-storage tensor-line control.");
        setup_error = std::max(setup_error,std::abs(a[i]-full[i]));
        setup_scale = std::max(setup_scale,std::abs(full[i]));
      }
    }
    MFEM_VERIFY(!full || setup_error <= 2e-11*std::max(1.0,setup_scale),
                "Reference-only setup changed the matched tensor-line RHS: p=" << order
                << " phase=" << phase << " sheared=" << sheared
                << " error=" << setup_error << " scale=" << setup_scale);
    if (phase == 0)
    {
      MFEM_VERIFY(error <= 5e-11, "Shared DG changed the constant-state residual.");
    }
    else
    {
      // Match the existing full-DG device/oracle FP64 tolerance: applying
      // inverse mass amplifies reassociation roundoff in weak-volume sums.
      const double tolerance = precision == Precision::FP64 ? 2e-11 : AlgebraTolerance(precision);
      MFEM_VERIFY(scale > 1e-3 && error <= tolerance * scale,
                  "Shared DG full RHS failed original operator comparison: p=" << order
                  << " phase=" << phase << " relative_error=" << error / scale);
    }
    long double largest_cell_moment = 0.0L;
    for (int e = 0; e < mesh.GetNE(); ++e)
    {
      mfem::Array<int> dofs;
      fes.GetElementDofs(e, dofs);
      for (int equation = 0; equation < 4; ++equation)
      {
        long double reference_moment = 0.0L, candidate_moment = 0.0L, bound = 0.0L;
        long double full_moment = 0.0L;
        for (int i = 0; i < dofs.Size(); ++i)
        {
          const int index = fes.DofToVDof(dofs[i], equation);
          const long double weight = cell_mass[e][i];
          reference_moment += weight * b[index];
          candidate_moment += weight * a[index];
          bound += weight * (std::abs(a[index]) + std::abs(b[index]));
          if (full) { full_moment += weight*full[index]; }
        }
        largest_cell_moment = std::max(largest_cell_moment, std::abs(reference_moment));
        const long double moment_tolerance =
          128.0L * std::numeric_limits<double>::epsilon() * dofs.Size() *
          std::max(1.0L, bound);
        MFEM_VERIFY(std::abs(candidate_moment - reference_moment) <= moment_tolerance,
                    "Shared DG changed the original conserved cell moment: p=" << order
                    << " phase=" << phase << " element=" << e << " equation=" << equation
                    << " error=" << static_cast<double>(candidate_moment - reference_moment)
                    << " tolerance=" << static_cast<double>(moment_tolerance));
        MFEM_VERIFY(!full || std::abs(candidate_moment-full_moment) <= moment_tolerance,
                    "Reference-only setup changed the matched tensor-line cell moment.");
      }
    }
    if (phase > 0)
    {
      // The surface flux yields nonzero individual cell moments even on this
      // periodic mesh. A blanket zero-moment projection of the full RHS fails.
      MFEM_VERIFY(largest_cell_moment > 1e-4L,
                  "Cell-moment regression lacks a nontrivial physical moment.");
    }
  }
  setenv("IGRLAB_SHARED_EULER_LAYOUT", "dense", 1);
  setenv("IGRLAB_SHARED_OPERATOR_PRECISION", "off", 1);
}

// Reverse the scalar local basis, including its node rule, so matching nodes
// require a nonidentity permutation. This remains a native nodal L2 VALUE FE.
class PermutedSigmaElement : public mfem::L2_QuadrilateralElement
{
  mfem::L2_QuadrilateralElement reference;
public:
  PermutedSigmaElement(int order, int basis)
    : mfem::L2_QuadrilateralElement(order, basis), reference(order, basis)
  {
    for (int i = 0; i < GetDof(); ++i)
      Nodes.IntPoint(i) = reference.GetNodes().IntPoint(GetDof() - 1 - i);
  }
  void CalcShape(const mfem::IntegrationPoint& point, Vector& shape) const override
  {
    Vector original(GetDof()); reference.CalcShape(point, original);
    for (int i = 0; i < GetDof(); ++i) { shape[i] = original[GetDof() - 1 - i]; }
  }
  void CalcDShape(const mfem::IntegrationPoint& point, DenseMatrix& shape) const override
  {
    DenseMatrix original(GetDof(), 2); reference.CalcDShape(point, original);
    for (int i = 0; i < GetDof(); ++i) for (int d = 0; d < 2; ++d)
      shape(i, d) = original(GetDof() - 1 - i, d);
  }
};

class PermutedSigmaCollection : public mfem::L2_FECollection
{
  PermutedSigmaElement square;
public:
  PermutedSigmaCollection(int order, int basis)
    : mfem::L2_FECollection(order, 2, basis), square(order, basis) {}
  const mfem::FiniteElement* FiniteElementForGeometry(mfem::Geometry::Type geometry) const override
  {
    return geometry == mfem::Geometry::SQUARE ? &square :
      mfem::L2_FECollection::FiniteElementForGeometry(geometry);
  }
};

// Keep the sigma and Euler fills separate: a warm operator must observe new
// scalar coefficients even when the Euler state and both FE spaces are fixed.
void FillVolumeMemoryEuler(mfem::ParFiniteElementSpace& fes, Vector& state)
{
  double* values = state.HostWrite();
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    const auto& nodes = fes.GetFE(e)->GetNodes();
    for (int i = 0; i < dofs.Size(); ++i)
    {
      const auto& ip = nodes.IntPoint(i);
      const double rho = 1.0 + 0.013 * e + 0.05 * ip.x + 0.02 * ip.x * ip.y;
      const double vx = 0.23 + 0.08 * ip.x - 0.004 * e;
      const double vy = -0.11 + 0.06 * ip.y + 0.003 * e;
      const double pressure = 1.0 + 0.04 * ip.x + 0.02 * ip.y + 0.01 * e;
      const double u[4] = {rho, rho * vx, rho * vy,
        pressure / 0.4 + 0.5 * rho * (vx * vx + vy * vy)};
      for (int c = 0; c < 4; ++c) { values[fes.DofToVDof(dofs[i], c)] = u[c]; }
    }
  }
}

void FillVolumeMemorySigma(mfem::ParGridFunction& sigma, int phase)
{
  auto& fes = *sigma.ParFESpace();
  double* values = sigma.HostWrite();
  const double amplitude = phase % 2 ? -(phase + 1.0) : phase + 1.0;
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    const auto& nodes = fes.GetFE(e)->GetNodes();
    for (int i = 0; i < dofs.Size(); ++i)
    {
      const auto& ip = nodes.IntPoint(i);
      values[dofs[i]] = amplitude * (0.025 + 0.005 * e +
        0.021 * std::sin(0.17 * (e + 1) + 2.3 * ip.x + 0.9 * ip.y) +
        0.012 * ip.x * ip.y);
    }
  }
}

void VolumeMemoryMode(const char* storage, const char* execution, bool shared)
{
  setenv("IGRLAB_WEAKDIV_SIGMA_STORAGE", storage, 1);
  setenv("IGRLAB_SHARED_EULER_FLUX_EXECUTION", execution, 1);
  setenv("IGRLAB_SHARED_EULER_LAYOUT", "dense", 1);
  setenv("IGRLAB_SHARED_OPERATOR_PRECISION", shared ? "fp64" : "off", 1);
}

struct VolumeMemoryControl
{
  const char* storage;
  const char* execution;
  bool shared;
  mfem::EulerFlux euler;
  mfem::FluxWithIGR physical;
  mfem::RusanovFluxWithSides numerical;
  std::unique_ptr<mfem::DGHyperbolicConservationLaws> op;

  VolumeMemoryControl(mfem::ParFiniteElementSpace& fes,
                      const mfem::ParGridFunction* sigma,
                      const char* storage_, const char* execution_, bool shared_)
    : storage(storage_), execution(execution_), shared(shared_), euler(2, 1.4),
      physical(euler, 1.4), numerical(physical)
  {
    physical.SetSigmaField(sigma);
    physical.SetInvariantDomainFlux(true);
    physical.SetInvariantDomainFloors(1e-12, 1e-12);
    VolumeMemoryMode(storage, execution, shared);
    op = std::make_unique<mfem::DGHyperbolicConservationLaws>(fes, numerical, 2, true);
    op->SetDGSEMGamma(1.4);
  }

  void Select() const { VolumeMemoryMode(storage, execution, shared); }
};

void CheckVolumeMemoryVector(const Vector& actual, const Vector& expected,
                             const std::string& label)
{
  const double* a = actual.HostRead(); const double* b = expected.HostRead();
  double error = 0.0, scale = 0.0;
  MFEM_VERIFY(actual.Size() == expected.Size(), "Mismatched DG oracle sizes.");
  for (int i = 0; i < actual.Size(); ++i)
  {
    MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]),
                "Nonfinite DG volume-memory oracle: " << label);
    error = std::max(error, std::abs(a[i] - b[i]));
    scale = std::max(scale, std::abs(b[i]));
  }
  MFEM_VERIFY(error <= 2e-11 * std::max(1.0, scale),
              "DG volume-memory residual differs: " << label
              << " error=" << error << " scale=" << scale);
}

void CheckVolumeMemoryTelemetry(const mfem::IGRFrozenSigmaTelemetry& actual,
                                const mfem::IGRFrozenSigmaTelemetry& expected,
                                const std::string& label)
{
  MFEM_VERIFY(actual.enabled == expected.enabled &&
                actual.sample_count == expected.sample_count &&
                actual.nonpositive_count == expected.nonpositive_count &&
                actual.nonfinite_count == expected.nonfinite_count,
              "DG volume-memory changed telemetry counts: " << label
              << " actual=" << actual.sample_count << " expected=" << expected.sample_count);
  if (expected.sample_count == 0) { return; }
  MFEM_VERIFY(std::isfinite(actual.minimum_raw_q_sigma) &&
                std::isfinite(actual.minimum_q_sigma_over_gamma_p) &&
                std::abs(actual.minimum_raw_q_sigma - expected.minimum_raw_q_sigma) <=
                  4e-13 * std::max(1.0, std::abs(expected.minimum_raw_q_sigma)) &&
                std::abs(actual.minimum_q_sigma_over_gamma_p -
                         expected.minimum_q_sigma_over_gamma_p) <=
                  4e-13 * std::max(1.0, std::abs(expected.minimum_q_sigma_over_gamma_p)),
              "DG volume-memory changed telemetry minima: " << label
              << " actual_q_sigma=" << actual.minimum_raw_q_sigma
              << " expected_q_sigma=" << expected.minimum_raw_q_sigma
              << " actual_ratio=" << actual.minimum_q_sigma_over_gamma_p
              << " expected_ratio=" << expected.minimum_q_sigma_over_gamma_p);
}

// A single cell with no boundary integrator has no surface contribution.
// This is an independent nodal-flux oracle: interpolate sigma with CalcShape,
// evaluate Euler+sigma fluxes in FP64, then apply assembled gradient and mass
// matrices. It never reads production metadata or calls a packed flux helper.
mfem::IGRFrozenSigmaTelemetry AssembleVolumeMemoryReference(
  mfem::ParFiniteElementSpace& fes, const Vector& state,
  const mfem::ParGridFunction* sigma, Vector& expected, double& speed)
{
  expected = 0.0;
  speed = 0.0;
  mfem::IGRFrozenSigmaTelemetry telemetry;
  telemetry.enabled = sigma != nullptr;
  const double* values = state.HostRead();
  const double* scalar = sigma ? sigma->HostRead() : nullptr;
  mfem::TransposeIntegrator gradient(new mfem::GradientIntegrator());
  mfem::IntegrationRules lobatto_rules(0, mfem::Quadrature1D::GaussLobatto);
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    const auto& fe = *fes.GetFE(e);
    const int dof = fe.GetDof();
    auto& transformation = *fes.GetElementTransformation(e);
    mfem::Array<int> dofs, sdofs;
    fes.GetElementDofs(e, dofs);
    if (sigma) { sigma->ParFESpace()->GetElementDofs(e, sdofs); }
    Vector shape(sigma ? sdofs.Size() : 0);
    DenseMatrix weak, mass, nodal_flux(2 * dof, 4);
    gradient.AssembleElementMatrix2(fe, fe, transformation, weak);
    mfem::MassIntegrator mass_integrator;
    const auto* tensor = dynamic_cast<const mfem::TensorBasisElement*>(&fe);
    if (tensor && tensor->GetBasisType() == mfem::BasisType::GaussLobatto)
      mass_integrator.SetIntRule(&lobatto_rules.Get(
        fe.GetGeomType(), std::max(1, 2 * fe.GetOrder() - 1)));
    mass_integrator.AssembleElementMatrix(fe, transformation, mass);
    mfem::DenseMatrixInverse inverse_mass(mass);
    for (int j = 0; j < dof; ++j)
    {
      double sampled_sigma = 0.0;
      if (sigma)
      {
        sigma->ParFESpace()->GetFE(e)->CalcShape(fe.GetNodes().IntPoint(j), shape);
        long double interpolated = 0.0L;
        for (int k = 0; k < sdofs.Size(); ++k)
          interpolated += static_cast<long double>(shape[k]) * scalar[sdofs[k]];
        sampled_sigma = static_cast<double>(interpolated);
      }
      const double rho = values[fes.DofToVDof(dofs[j], 0)];
      const double mx = values[fes.DofToVDof(dofs[j], 1)];
      const double my = values[fes.DofToVDof(dofs[j], 2)];
      const double energy = values[fes.DofToVDof(dofs[j], 3)];
      const double pressure = 0.4 * (energy - 0.5 * (mx * mx + my * my) / rho);
      const double enthalpy_density = energy + pressure + sampled_sigma;
      nodal_flux(j, 0) = mx;
      nodal_flux(dof + j, 0) = my;
      nodal_flux(j, 1) = mx * mx / rho + pressure + sampled_sigma;
      nodal_flux(dof + j, 1) = mx * my / rho;
      nodal_flux(j, 2) = my * mx / rho;
      nodal_flux(dof + j, 2) = my * my / rho + pressure + sampled_sigma;
      nodal_flux(j, 3) = mx * enthalpy_density / rho;
      nodal_flux(dof + j, 3) = my * enthalpy_density / rho;
      if (sigma)
      {
        const double transport = std::sqrt(mx * mx + my * my) / rho;
        speed = std::max(speed, mfem::IGRFluxInvariantDomainSpeed(
          transport + std::sqrt(1.4 * pressure / rho), rho, pressure,
          transport, sampled_sigma, 1.4, 1e-12, 1e-12));
        const double q_sigma = 1.4 * pressure + 0.4 * sampled_sigma;
        ++telemetry.sample_count;
        telemetry.minimum_raw_q_sigma = std::min(telemetry.minimum_raw_q_sigma, q_sigma);
        telemetry.minimum_q_sigma_over_gamma_p = std::min(
          telemetry.minimum_q_sigma_over_gamma_p, q_sigma / (1.4 * pressure));
      }
    }
    Vector weak_rhs(dof), strong_rhs(dof);
    double* result = expected.HostReadWrite();
    for (int c = 0; c < 4; ++c)
    {
      for (int i = 0; i < dof; ++i)
      {
        long double sum = 0.0L;
        for (int j = 0; j < 2 * dof; ++j)
          sum += static_cast<long double>(weak(i, j)) * nodal_flux(j, c);
        weak_rhs[i] = static_cast<double>(sum);
      }
      inverse_mass.Mult(weak_rhs, strong_rhs);
      for (int i = 0; i < dof; ++i)
        result[fes.DofToVDof(dofs[i], c)] = strong_rhs[i];
    }
  }
  return telemetry;
}

void DGVolumeMemoryOracle(int state_basis, int sigma_basis, int sigma_order,
                          bool periodic = false, bool with_sigma = true,
                          bool h1_sigma = false, bool permuted_sigma = false)
{
  constexpr int order = 3;
  const int cells = periodic ? 4 : 1;
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    cells, cells, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  for (int v = 0; v < serial.GetNV(); ++v)
  {
    double* point = serial.GetVertex(v);
    point[0] = 0.75 * point[0] + 0.25 * point[1];
    point[1] *= 1.25;
  }
  mfem::Mesh topology;
  if (periodic)
  {
    const std::vector<Vector> translations = {Vector({0.75, 0.0}), Vector({0.25, 1.25})};
    topology = mfem::Mesh::MakePeriodic(serial, serial.CreatePeriodicVertexMapping(translations));
  }
  else { topology = std::move(serial); }
  mfem::ParMesh mesh(MPI_COMM_SELF, topology);
  mfem::L2_FECollection fec(order, 2, state_basis);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, 4, mfem::Ordering::byNODES);
  std::unique_ptr<mfem::FiniteElementCollection> sigma_fec;
  if (permuted_sigma)
    sigma_fec = std::make_unique<PermutedSigmaCollection>(sigma_order, sigma_basis);
  else if (h1_sigma)
    sigma_fec = std::make_unique<mfem::H1_FECollection>(sigma_order, 2, sigma_basis);
  else
    sigma_fec = std::make_unique<mfem::L2_FECollection>(sigma_order, 2, sigma_basis);
  mfem::ParFiniteElementSpace sfes(&mesh, sigma_fec.get());
  mfem::ParGridFunction sigma(&sfes);
  if (sigma_basis == mfem::BasisType::IntegratedGLL)
  {
    const auto& state_nodes = fes.GetFE(0)->GetNodes();
    const auto& scalar_fe = *sfes.GetFE(0);
    const auto& scalar_nodes = scalar_fe.GetNodes();
    MFEM_VERIFY(state_nodes.Size() == scalar_nodes.Size(),
                "IntegratedGLL regression requires matching coordinate tables.");
    Vector shape(scalar_fe.GetDof());
    bool nonnodal = false;
    for (int i = 0; i < state_nodes.Size(); ++i)
    {
      const auto& state_node = state_nodes.IntPoint(i);
      const auto& scalar_node = scalar_nodes.IntPoint(i);
      MFEM_VERIFY(state_node.x == scalar_node.x && state_node.y == scalar_node.y,
                  "IntegratedGLL regression requires identical GL coordinates.");
      scalar_fe.CalcShape(state_node, shape);
      for (int j = 0; j < shape.Size(); ++j)
        nonnodal = nonnodal || std::abs(shape[j] - (i == j ? 1.0 : 0.0)) > 1e-6;
    }
    MFEM_VERIFY(nonnodal, "IntegratedGLL regression must exercise histopolant coefficients.");
  }
  mfem::L2_FECollection replacement_fec(2, 2, mfem::BasisType::GaussLobatto);
  mfem::ParFiniteElementSpace replacement_fes(&mesh, &replacement_fec);
  mfem::ParGridFunction replacement(&replacement_fes);
  Vector state(fes.GetVSize()), assembled(fes.GetVSize()), first_result(fes.GetVSize());
  FillVolumeMemoryEuler(fes, state);
  std::vector<std::unique_ptr<VolumeMemoryControl>> controls;
  const auto add_control = [&](const char* storage, const char* execution, bool shared)
  {
    controls.emplace_back(std::make_unique<VolumeMemoryControl>(
      fes, with_sigma ? &sigma : nullptr, storage, execution, shared));
  };
  add_control("dense", "element", false);
  add_control("auto", "element", false);
  // The shared-reference mass policy admits GL state nodes. GLL still tests
  // matching nodes and its ordinary packed interpolation/fallback path.
  if (state_basis == mfem::BasisType::GaussLegendre)
  {
    add_control("dense", "element", true);
    add_control("auto", "element", true);
    add_control("auto", "pointwise", true);
  }
  const std::string cohort = "state_basis=" + std::to_string(state_basis) +
    " sigma_basis=" + std::to_string(sigma_basis) + " sigma_order=" +
    std::to_string(sigma_order) + " periodic=" + std::to_string(periodic) +
    " h1_sigma=" + std::to_string(h1_sigma) + " permuted_sigma=" +
    std::to_string(permuted_sigma) + " with_sigma=" + std::to_string(with_sigma);
  const int phases = periodic ? 5 : with_sigma ? 2 : 1;
  for (int phase = 0; phase < phases; ++phase)
  {
    FillVolumeMemorySigma(sigma, phase);
    FillVolumeMemorySigma(replacement, phase + 1);
    const mfem::ParGridFunction* current_sigma = with_sigma ? &sigma : nullptr;
    if (periodic && phase == 2) { current_sigma = &replacement; }
    if (periodic && phase == 3) { current_sigma = nullptr; }
    double assembled_speed = 0.0;
    mfem::IGRFrozenSigmaTelemetry volume_samples;
    if (!periodic)
      volume_samples = AssembleVolumeMemoryReference(
        fes, state, current_sigma, assembled, assembled_speed);
    std::vector<Vector> without_tracking(controls.size());
    for (bool tracking : {false, true})
    {
      Vector legacy(fes.GetVSize());
      double legacy_speed = 0.0;
      mfem::IGRFrozenSigmaTelemetry legacy_telemetry;
      for (std::size_t index = 0; index < controls.size(); ++index)
      {
        auto& control = *controls[index];
        control.Select();
        control.physical.SetSigmaField(current_sigma);
        if (periodic && phase == 4 && !tracking) { control.op->Update(); }
        control.op->SetInvariantDomainVolumeSpeedTracking(tracking);
        const auto before = control.op->GetGlobalFrozenSigmaTelemetry();
        Vector actual(fes.GetVSize());
        std::ostringstream output;
        {
          struct RestoreOutput
          {
            std::streambuf* previous;
            ~RestoreOutput() { std::cout.rdbuf(previous); }
          } restore{std::cout.rdbuf(output.rdbuf())};
          control.op->Mult(state, actual);
        }
        const std::string log = output.str();
        std::cout << log;
        const auto after = control.op->GetGlobalFrozenSigmaTelemetry();
        const double speed = control.op->GetMaxCharSpeed();
        const std::string label = cohort + " phase=" + std::to_string(phase) +
          " storage=" + control.storage + " execution=" + control.execution +
          " shared=" + std::to_string(control.shared) + " tracking=" + std::to_string(tracking);
        if (phase == 0 && !tracking)
        {
          const std::string execution = control.shared ? control.execution : "element";
          MFEM_VERIFY(log.find("flux_execution=" + execution) != std::string::npos,
                      "DG flux fixture did not execute its selected mode: " << label);
          if (with_sigma && !h1_sigma && state_basis == sigma_basis &&
              sigma_order == order && std::string(control.storage) == "auto")
            MFEM_VERIFY(log.find("policy=auto storage=exact-nodes") != std::string::npos,
                        "Matching DG sigma nodes did not select exact gathering: " << label);
          if (sigma_basis == mfem::BasisType::IntegratedGLL &&
              std::string(control.storage) == "auto")
            MFEM_VERIFY(log.find("policy=auto storage=shared-reference") != std::string::npos,
                        "Histopolant sigma must use shape interpolation: " << label);
        }
        if (index == 0)
        {
          legacy = actual; legacy_speed = speed; legacy_telemetry = after;
          if (phase == 0 && !tracking) { first_result = actual; }
          if (phase == 1 && !tracking)
          {
            Vector sigma_change(actual); sigma_change -= first_result;
            MFEM_VERIFY(sigma_change.Normlinf() > 1e-4,
                        "DG sigma mutation fixture has no residual effect: " << label);
          }
        }
        CheckVolumeMemoryVector(actual, legacy, label + " legacy parity");
        MFEM_VERIFY(std::isfinite(speed) &&
                      std::abs(speed - legacy_speed) <= 2e-12 * std::max(1.0, legacy_speed),
                    "DG volume-memory changed characteristic speed: " << label);
        CheckVolumeMemoryTelemetry(after, legacy_telemetry, label + " legacy parity");
        if (!periodic)
        {
          CheckVolumeMemoryVector(actual, assembled, label + " assembled nonlinear flux");
          const double expected_speed = tracking ? assembled_speed : 0.0;
          MFEM_VERIFY(std::abs(speed - expected_speed) <= 2e-12 * std::max(1.0, expected_speed),
                      "DG volume-memory speed differs from nodal oracle: " << label);
          auto expected_telemetry = before;
          // The public getter reports empty minima as NaN. Restore the
          // accumulator identity before adding the first independent samples.
          if (before.sample_count == 0)
          {
            expected_telemetry.minimum_raw_q_sigma = std::numeric_limits<double>::infinity();
            expected_telemetry.minimum_q_sigma_over_gamma_p =
              std::numeric_limits<double>::infinity();
          }
          expected_telemetry.Merge(volume_samples);
          CheckVolumeMemoryTelemetry(after, expected_telemetry, label + " nodal samples");
        }
        if (!tracking) { without_tracking[index] = actual; }
        else
          CheckVolumeMemoryVector(actual, without_tracking[index], label + " tracking residual");
      }
    }
  }
  std::cout << "dg_volume_memory_oracle " << cohort << " passed\n";
}

void DGVolumeMemoryOracles()
{
  const int gl = mfem::BasisType::GaussLegendre;
  const int gll = mfem::BasisType::GaussLobatto;
  DGVolumeMemoryOracle(gl, gl, 3);
  DGVolumeMemoryOracle(gl, gl, 3, false, true, false, true);
  DGVolumeMemoryOracle(gl, gll, 3);
  // IntegratedGLL exposes GL coordinates but its coefficients are histopolant.
  DGVolumeMemoryOracle(gl, mfem::BasisType::IntegratedGLL, 3);
  DGVolumeMemoryOracle(gl, gl, 2);
  DGVolumeMemoryOracle(gll, gll, 3);
  // H1 GLL orders its shared vertex/edge nodes differently from L2 GLL.
  // It exercises the preserved unsupported-space fallback against CalcShape.
  DGVolumeMemoryOracle(gll, gll, 3, false, true, true);
  DGVolumeMemoryOracle(gl, gl, 3, false, false);
  DGVolumeMemoryOracle(gl, gl, 3, true);
  VolumeMemoryMode("dense", "element", false);
  std::cout << "dg_volume_memory_test passed matching_gl=true matching_gll=true "
               "node_permutation=true basis_interpolation=true histopolant_interpolation=true "
               "order_interpolation=true h1_fallback=true "
               "current_sigma=true nonlinear_volume_oracle=true pointwise_rhs=true "
               "no_sigma=true volume_speed=true telemetry_counts=true "
               "sigma_space_rebind=true update=true\n";
}

mfem::Mesh TensorFaceMesh(int cells, bool sheared, bool rotated)
{
  mfem::Mesh mesh(2, (cells + 1) * (cells + 1), cells * cells, 4 * cells, 2);
  for (int j = 0; j <= cells; ++j) for (int i = 0; i <= cells; ++i)
  {
    const double x = double(i) / cells, y = double(j) / cells;
    const double point[2] = {sheared ? 0.75 * x + 0.25 * y : x, sheared ? 1.25 * y : y};
    mesh.AddVertex(point);
  }
  for (int j = 0; j < cells; ++j) for (int i = 0; i < cells; ++i)
  {
    const int v[4] = {i + j * (cells + 1), i + 1 + j * (cells + 1),
      i + 1 + (j + 1) * (cells + 1), i + (j + 1) * (cells + 1)};
    const int shift = rotated ? (i + 2 * j) % 4 : 0;
    int element[4];
    for (int k = 0; k < 4; ++k) { element[k] = v[(k + shift) % 4]; }
    mesh.AddQuad(element);
  }
  for (int i = 0; i < cells; ++i)
  {
    mesh.AddBdrSegment(i, i + 1, 1);
    mesh.AddBdrSegment(cells + i * (cells + 1), cells + (i + 1) * (cells + 1), 2);
    mesh.AddBdrSegment(i + 1 + cells * (cells + 1), i + cells * (cells + 1), 3);
    mesh.AddBdrSegment((i + 1) * (cells + 1), i * (cells + 1), 4);
  }
  mesh.FinalizeQuadMesh(1, 0, true);
  return mesh;
}

void FillTensorFaceBlast(mfem::ParFiniteElementSpace& fes, Vector& state)
{
  double* values = state.HostWrite();
  Vector point(2);
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    const auto& nodes = fes.GetFE(e)->GetNodes();
    auto& transformation = *fes.GetElementTransformation(e);
    for (int i = 0; i < dofs.Size(); ++i)
    {
      transformation.Transform(nodes.IntPoint(i), point);
      const double x = point[0], y = point[1];
      const double rho = 1.0 + 0.03 * x;
      const double vx = 0.18 * (x - 0.4), vy = 0.12 * (y - 0.45);
      const double radius2 = (x - 0.35) * (x - 0.35) + (y - 0.4) * (y - 0.4);
      const double pressure = 0.12 + 0.9 * std::exp(-18.0 * radius2) + 0.02 * x + 0.03 * y;
      const double u[4] = {rho, rho * vx, rho * vy,
        pressure / 0.4 + 0.5 * rho * (vx * vx + vy * vy)};
      for (int c = 0; c < 4; ++c) { values[fes.DofToVDof(dofs[i], c)] = u[c]; }
    }
  }
}

struct TensorFaceControl
{
  mfem::EulerFlux euler;
  mfem::FluxWithIGR physical;
  mfem::RusanovFluxWithSides numerical;
  std::unique_ptr<mfem::DGHyperbolicConservationLaws> op;

  TensorFaceControl(mfem::ParFiniteElementSpace& fes, const mfem::ParGridFunction& sigma,
                    int face_order_offset)
    : euler(2, 1.4), physical(euler, 1.4), numerical(physical)
  {
    physical.SetSigmaField(&sigma);
    physical.SetInvariantDomainFlux(true);
    physical.SetInvariantDomainFloors(1e-12, 1e-12);
    Select();
    // Offset 1 is the Sedov GL(p+1) rule. The offset-2 control must retain
    // its original higher quadrature through the legacy fallback.
    op = std::make_unique<mfem::DGHyperbolicConservationLaws>(
      fes, numerical, face_order_offset, true);
    op->SetDGSEMGamma(1.4);
  }
  void Select() const { VolumeMemoryMode("dense", "element", false); }
};

std::string ApplyTensorFace(TensorFaceControl& control, const char* mode,
                            const Vector& state, Vector& residual)
{
  control.Select();
  setenv("IGRLAB_DG_TENSOR_FACE", mode, 1);
  std::ostringstream output;
  {
    struct RestoreOutput
    {
      std::streambuf* previous;
      ~RestoreOutput() { std::cout.rdbuf(previous); }
    } restore{std::cout.rdbuf(output.rdbuf())};
    control.op->Mult(state, residual);
  }
  std::cout << output.str();
  return output.str();
}

void CheckTensorFaceMoments(mfem::ParFiniteElementSpace& fes,
                            const std::vector<Vector>& cell_mass,
                            const Vector& actual, const Vector& legacy,
                            const Vector& boundary_flux, const std::string& label)
{
  const double* a = actual.HostRead(); const double* b = legacy.HostRead();
  const double* outward = boundary_flux.HostRead();
  long double total[4] = {}, total_bound[4] = {};
  long double largest_cell_moment = 0.0L;
  for (int e = 0; e < fes.GetNE(); ++e)
  {
    mfem::Array<int> dofs;
    fes.GetElementDofs(e, dofs);
    for (int c = 0; c < 4; ++c)
    {
      long double moment = 0.0L, reference = 0.0L, bound = 0.0L;
      for (int i = 0; i < dofs.Size(); ++i)
      {
        const int index = fes.DofToVDof(dofs[i], c);
        const long double weight = cell_mass[e][i];
        moment += weight * a[index];
        reference += weight * b[index];
        bound += weight * (std::abs(a[index]) + std::abs(b[index]));
      }
      const long double tolerance = 256.0L * std::numeric_limits<double>::epsilon() *
        dofs.Size() * std::max(1.0L, bound);
      MFEM_VERIFY(std::abs(moment - reference) <= tolerance,
                  "Tensor face changed a conserved cell moment: " << label
                  << " element=" << e << " equation=" << c
                  << " error=" << static_cast<double>(moment - reference));
      total[c] += moment;
      total_bound[c] += bound;
      largest_cell_moment = std::max(largest_cell_moment, std::abs(reference));
    }
  }
  MFEM_VERIFY(largest_cell_moment > 1e-4L,
              "Tensor face fixture lacks nontrivial cell moments: " << label);
  for (int c = 0; c < 4; ++c)
  {
    const long double tolerance = 512.0L * std::numeric_limits<double>::epsilon() *
      fes.GetNDofs() * std::max(1.0L, total_bound[c] + std::abs(outward[c]));
    MFEM_VERIFY(std::abs(total[c] + outward[c]) <= tolerance,
                "Tensor face residual does not close the boundary-flux balance: " << label
                << " equation=" << c << " residual=" << static_cast<double>(total[c])
                << " outward_flux=" << outward[c] << " tolerance=" << static_cast<double>(tolerance));
  }
}

void DGTensorFaceOracle(bool periodic, bool sheared, bool rotated,
                        bool mismatched_sigma = false, int face_order_offset = 1)
{
  constexpr int order = 5;
  mfem::Mesh serial = TensorFaceMesh(periodic ? 4 : 2, sheared, rotated);
  mfem::Mesh topology;
  if (periodic)
  {
    const std::vector<Vector> translations = sheared
      ? std::vector<Vector>{Vector({0.75, 0.0}), Vector({0.25, 1.25})}
      : std::vector<Vector>{Vector({1.0, 0.0}), Vector({0.0, 1.0})};
    topology = mfem::Mesh::MakePeriodic(serial, serial.CreatePeriodicVertexMapping(translations));
  }
  else { topology = std::move(serial); }
  mfem::ParMesh mesh(MPI_COMM_SELF, topology);
  mfem::L2_FECollection fec(order, 2, mfem::BasisType::GaussLegendre);
  mfem::L2_FECollection sfec(order, 2, mismatched_sigma
    ? mfem::BasisType::GaussLobatto : mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(&mesh, &fec, 4, mfem::Ordering::byNODES);
  mfem::ParFiniteElementSpace sfes(&mesh, &sfec);
  mfem::ParGridFunction sigma(&sfes);
  Vector state(fes.GetVSize()), first_legacy(fes.GetVSize());
  if (periodic && face_order_offset == 1) { FillVolumeMemoryEuler(fes, state); }
  else { FillTensorFaceBlast(fes, state); }
  // Independent M*1 weights convert the strong RHS to conserved cell moments.
  std::vector<Vector> cell_mass(mesh.GetNE());
  mfem::MassIntegrator mass_integrator;
  for (int e = 0; e < mesh.GetNE(); ++e)
  {
    DenseMatrix mass;
    const auto& fe = *fes.GetFE(e);
    mass_integrator.AssembleElementMatrix(fe, *fes.GetElementTransformation(e), mass);
    Vector one(fe.GetDof()); one = 1.0;
    cell_mass[e].SetSize(fe.GetDof());
    mass.Mult(one, cell_mass[e]);
  }
  std::vector<std::unique_ptr<TensorFaceControl>> controls;
  for (const char* mode : {"off", "fp64"})
  {
    setenv("IGRLAB_DG_TENSOR_FACE", mode, 1);
    controls.emplace_back(std::make_unique<TensorFaceControl>(fes, sigma, face_order_offset));
    if (!periodic)
    {
      mfem::Array<int> boundaries(mesh.bdr_attributes.Max()); boundaries = 1;
      controls.back()->op->AddBdrFaceIntegrator(
        new mfem::BdrHyperbolicFluxIntegrator(controls.back()->numerical,
          std::make_unique<mfem::SlipWallState>(2, 1e-12, 1e-12), face_order_offset), boundaries);
    }
  }
  const std::string cohort = "periodic=" + std::to_string(periodic) +
    " sheared=" + std::to_string(sheared) + " rotated=" + std::to_string(rotated) +
    " mismatched_sigma=" + std::to_string(mismatched_sigma) +
    " face_order_offset=" + std::to_string(face_order_offset);
  // Euler coefficients remain fixed; only the current sigma values change.
  for (int phase = 0; phase < 2; ++phase)
  {
    FillVolumeMemorySigma(sigma, phase);
    Vector legacy(fes.GetVSize()), legacy_boundary_flux;
    double legacy_speed = 0.0;
    mfem::IGRFrozenSigmaTelemetry legacy_telemetry;
    for (std::size_t index = 0; index < controls.size(); ++index)
    {
      auto& control = *controls[index];
      const char* mode = index == 0 ? "off" : "fp64";
      const std::string label = cohort + " phase=" + std::to_string(phase) + " mode=" + mode;
      MFEM_VERIFY(!control.op->GetInvariantDomainVolumeSpeedTracking(),
                  "Tensor face fixture must compare face-only characteristic speeds.");
      Vector actual(fes.GetVSize()), without_capture;
      if (phase == 0)
      {
        const std::string log = ApplyTensorFace(control, mode, state, actual);
        MFEM_VERIFY(!control.op->HasLastPhysicalBoundaryFluxIntegral(),
                    "Unarmed tensor face application manufactured boundary capture: " << label);
        if (index == 0)
        {
          MFEM_VERIFY(log.find("[dg-tensor-face] event=setup selected=fp64") == std::string::npos,
                      "Tensor face off control selected fp64.");
        }
        else if (mismatched_sigma)
        {
          MFEM_VERIFY(log.find("[dg-tensor-face] event=fallback reason=sigma-layout") != std::string::npos,
                      "Nonmatching sigma space did not dispatch to the legacy face path.");
        }
        else if (face_order_offset != 1)
        {
          MFEM_VERIFY(log.find("[dg-tensor-face] event=fallback reason=quadrature") != std::string::npos &&
                        log.find("[dg-tensor-face] event=setup selected=fp64") == std::string::npos,
                      "Tensor faces must preserve the original offset-2 quadrature through fallback.");
        }
        else
        {
          MFEM_VERIFY(log.find("[dg-tensor-face] event=setup selected=fp64") != std::string::npos &&
                        log.find("dense_shape_entries=0") != std::string::npos,
                      "Tensor face fixture did not select compact FP64 faces: " << label);
        }
        without_capture = actual;
      }
      control.op->ArmPhysicalBoundaryFluxIntegral();
      ApplyTensorFace(control, mode, state, actual);
      MFEM_VERIFY(control.op->HasLastPhysicalBoundaryFluxIntegral(),
                  "Armed tensor face application did not publish boundary capture: " << label);
      const Vector boundary_flux(control.op->GetLastPhysicalBoundaryFluxIntegral());
      MFEM_VERIFY(boundary_flux.Size() == 4, "Tensor face capture has an invalid equation count.");
      if (phase == 0)
        CheckVolumeMemoryVector(actual, without_capture, label + " capture preserves RHS");
      const double speed = control.op->GetMaxCharSpeed();
      const auto telemetry = control.op->GetGlobalFrozenSigmaTelemetry();
      if (index == 0)
      {
        legacy = actual; legacy_boundary_flux = boundary_flux;
        legacy_speed = speed; legacy_telemetry = telemetry;
        if (phase == 0) { first_legacy = actual; }
        else
        {
          Vector change(actual); change -= first_legacy;
          MFEM_VERIFY(change.Normlinf() > 1e-4,
                      "Tensor face sigma mutation fixture has no residual effect: " << label);
        }
      }
      CheckVolumeMemoryVector(actual, legacy, label + " full RHS versus legacy");
      CheckVolumeMemoryVector(boundary_flux, legacy_boundary_flux, label + " boundary capture versus legacy");
      MFEM_VERIFY(std::isfinite(speed) && std::abs(speed - legacy_speed) <=
                    2e-12 * std::max(1.0, legacy_speed),
                  "Tensor faces changed the characteristic-speed policy: " << label);
      CheckVolumeMemoryTelemetry(telemetry, legacy_telemetry, label + " telemetry versus legacy");
      CheckTensorFaceMoments(fes, cell_mass, actual, legacy, boundary_flux, label);
      if (periodic)
      {
        MFEM_VERIFY(boundary_flux.Normlinf() == 0.0,
                    "Periodic tensor face capture contains physical boundary flux.");
      }
      else
      {
        const double* flux = boundary_flux.HostRead();
        MFEM_VERIFY(boundary_flux.Normlinf() > 1e-4 &&
                      std::abs(flux[0]) <= 2e-12 && std::abs(flux[3]) <= 2e-12,
                    "Slip-wall blast must have nontrivial momentum flux and zero mass/energy leakage.");
      }
    }
  }
  std::cout << "dg_tensor_face_oracle " << cohort << " passed\n";
}

void DGTensorFaceOracles()
{
  DGTensorFaceOracle(true, false, false);
  DGTensorFaceOracle(true, true, true);
  DGTensorFaceOracle(false, false, false);
  DGTensorFaceOracle(true, false, false, true);
  DGTensorFaceOracle(true, false, false, false, 2);
  setenv("IGRLAB_DG_TENSOR_FACE", "off", 1);
  VolumeMemoryMode("dense", "element", false);
  std::cout << "dg_tensor_face_test passed periodic=true sheared_oriented=true "
               "slip_blast=true current_sigma=true full_rhs=true cell_moments=true "
               "boundary_capture=true canonical_conservation=true "
               "face_speed_telemetry=true sigma_layout_fallback=true quadrature_fallback=true\n";
}

void ReferenceSetupRejection()
{
#ifdef MFEM_USE_EXCEPTIONS
  for (bool nonaffine : {false,true})
  {
    mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
      2,2,mfem::Element::QUADRILATERAL,false,1.0,1.0,false);
    for (int v = 0; v < serial.GetNV(); ++v)
    {
      double* point = serial.GetVertex(v);
      if (point[0] == 0.5 && (!nonaffine || point[1] == 0.5)) { point[0] += 0.0625; }
    }
    mfem::ParMesh mesh(MPI_COMM_SELF,serial);
    mfem::L2_FECollection fec(5,2,mfem::BasisType::GaussLegendre);
    mfem::ParFiniteElementSpace fes(&mesh,&fec,4,mfem::Ordering::byNODES);
    mfem::EulerFlux euler(2,1.4); mfem::RusanovFluxWithSides flux(euler);
    setenv("IGRLAB_SHARED_EULER_LAYOUT", "tensor-lines", 1);
    setenv("IGRLAB_SHARED_OPERATOR_PRECISION", "fp64", 1);
    const auto previous = mfem::get_error_action();
    mfem::set_error_action(mfem::MFEM_ERROR_THROW);
    bool rejected = false;
    try { mfem::DGHyperbolicConservationLaws candidate(fes,flux,2,true); }
    catch (const mfem::ErrorException&) { rejected = true; }
    mfem::set_error_action(previous);
    MFEM_VERIFY(rejected,"Reference-only DG setup admitted nonuniform/non-affine cells.");
  }
  std::cout << "shared_reference_setup_rejection nonuniform=true nonaffine=true passed\n";
#else
  std::cout << "shared_reference_setup_rejection skipped=no-mfem-exceptions\n";
#endif
  setenv("IGRLAB_SHARED_EULER_LAYOUT", "dense", 1);
  setenv("IGRLAB_SHARED_OPERATOR_PRECISION", "off", 1);
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  unsetenv("DG_IGR_SHARED_OPERATOR_PRECISION");
  unsetenv("DG_IGR_SHARED_EULER_LAYOUT");
  setenv("IGRLAB_SHARED_EULER_LAYOUT", "dense", 1);
  setenv("IGRLAB_SHARED_OPERATOR_PRECISION", "off", 1);
  unsetenv("DG_IGR_WEAKDIV_SIGMA_STORAGE");
  unsetenv("DG_IGR_SHARED_EULER_FLUX_EXECUTION");
  unsetenv("DG_IGR_DG_TENSOR_FACE");
  setenv("IGRLAB_DG_TENSOR_FACE", "off", 1);
  if (argc > 1 && std::string(argv[1]) == "--dg-tensor-face-only")
  {
    DGTensorFaceOracles();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "--dg-volume-memory-only")
  {
    DGVolumeMemoryOracles();
    return 0;
  }
  DGVolumeMemoryOracles();
  ReferenceSetupRejection();
  for (Precision precision : {Precision::FP64, Precision::FP32, Precision::TF32})
  {
    MatrixOracle(6, 6, 37, precision);
    MatrixOracle(36, 98, 71, precision);
    MatrixOracle(17, 13, 19, precision);
    for (int order = 1; order <= 7; ++order)
      for (bool sheared : {false,true}) { TensorLineOracle(order,precision,sheared); }
    for (int order : {1, 3, 5, 7})
    {
      CentralLineOracle(order, precision);
      DGProductionOracle(order, precision);
      DGProductionOracle(order, precision, "tensor-lines");
      DGProductionOracle(order, precision, "tensor-lines", true);
    }
  }
  std::cout << "shared_reference_test passed real_columns=true padded_tails=true "
               "immutable_matrix=true reused_workspace=true centered_constant=true "
               "dg_original_rhs=true dg_original_cell_moment=true "
               "dg_tensor_lines_original_matrix=true dg_tensor_lines_shear=true "
               "dg_reference_setup_rhs=true dg_reference_setup_moment=true dg_reference_setup_update=true "
               "dgsem_original_split=true nonfinite_exposed=true\n";
  return 0;
}

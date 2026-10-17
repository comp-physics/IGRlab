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

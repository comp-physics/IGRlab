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

void DGProductionOracle(int order, Precision precision)
{
  mfem::Mesh serial = mfem::Mesh::MakeCartesian2D(
    4, 4, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false);
  std::vector<Vector> translations = {Vector({1.0, 0.0}), Vector({0.0, 1.0})};
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
  mfem::DGHyperbolicConservationLaws baseline(fes, flux, 2, true);
  mfem::DGHyperbolicConservationLaws candidate(fes, flux, 2, true);
  baseline.SetDGSEMGamma(1.4); candidate.SetDGSEMGamma(1.4);
  baseline.SetInvariantDomainVolumeSpeedTracking(true);
  candidate.SetInvariantDomainVolumeSpeedTracking(true);
  Vector state(fes.GetVSize()), expected(fes.GetVSize()), actual(fes.GetVSize());
  for (int phase = 0; phase < 3; ++phase)
  {
    FillEuler(fes, state, sigma, phase);
    setenv("IGRLAB_SHARED_OPERATOR_PRECISION", "off", 1);
    baseline.Mult(state, expected);
    setenv("IGRLAB_SHARED_OPERATOR_PRECISION", igrlab::shared_reference::PrecisionName(precision), 1);
    candidate.Mult(state, actual);
    MFEM_VERIFY(std::isfinite(candidate.GetMaxCharSpeed()) &&
                  candidate.GetMaxCharSpeed() == baseline.GetMaxCharSpeed(),
                "Shared derivative changed the FP64 characteristic-speed policy.");
    const double* a = actual.HostRead(); const double* b = expected.HostRead();
    double error = 0.0, scale = 0.0;
    for (int i = 0; i < state.Size(); ++i)
    {
      MFEM_VERIFY(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite DG production oracle.");
      error = std::max(error, std::abs(a[i] - b[i]));
      scale = std::max(scale, std::abs(b[i]));
    }
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
  }
  setenv("IGRLAB_SHARED_OPERATOR_PRECISION", "off", 1);
}
} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  mfem::Device device("cuda");
  unsetenv("DG_IGR_SHARED_OPERATOR_PRECISION");
  setenv("IGRLAB_SHARED_OPERATOR_PRECISION", "off", 1);
  for (Precision precision : {Precision::FP64, Precision::FP32, Precision::TF32})
  {
    MatrixOracle(6, 6, 37, precision);
    MatrixOracle(36, 98, 71, precision);
    MatrixOracle(17, 13, 19, precision);
    for (int order : {1, 3, 5, 7})
    {
      CentralLineOracle(order, precision);
      DGProductionOracle(order, precision);
    }
  }
  std::cout << "shared_reference_test passed real_columns=true padded_tails=true "
               "immutable_matrix=true reused_workspace=true centered_constant=true "
               "dg_original_rhs=true dgsem_original_split=true nonfinite_exposed=true\n";
  return 0;
}

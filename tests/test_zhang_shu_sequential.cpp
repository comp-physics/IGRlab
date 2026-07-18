#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include "numerics/limiters/zhang_shu.hpp"
#include "physics/euler/eos.hpp"
#include "../src/numerics/limiters/limiters_common.hpp"

namespace
{

constexpr int kDim = 2;
constexpr int kNumEq = kDim + 2;
constexpr double kGamma = 1.4;
constexpr double kRhoMin = 0.1;
constexpr double kPressureMin = 0.1;
constexpr int kQuadratureOrder = 4;
constexpr int kBisectionIterations = 40;

double Pressure(const mfem::Vector& state)
{
  return mfem::EulerPressureFromConservative(
    state, kDim, kGamma, 1.0e-16);
}

void CurveMesh(mfem::Mesh& mesh)
{
  mesh.SetCurvature(2, false, kDim, mfem::Ordering::byNODES);
  mfem::GridFunction* nodes = mesh.GetNodes();
  MFEM_VERIFY(nodes != nullptr, "Curved limiter test requires mesh nodes.");
  const int ndofs = nodes->FESpace()->GetNDofs();
  double* coordinates = nodes->HostReadWrite();
  for (int i = 0; i < ndofs; ++i)
  {
    const double x = coordinates[i];
    const double y = coordinates[i + ndofs];
    const double bubble = std::sin(M_PI * x) * std::sin(M_PI * y);
    coordinates[i] = x + 0.08 * bubble;
    coordinates[i + ndofs] = y - 0.05 * bubble;
  }
}

mfem::Vector ElementAverage(mfem::ParFiniteElementSpace& fes,
                            const mfem::ParGridFunction& state,
                            int element)
{
  mfem::Array<int> vdofs;
  mfem::Vector element_state;
  fes.GetElementVDofs(element, vdofs);
  state.GetSubVector(vdofs, element_state);
  const int dof = fes.GetFE(element)->GetDof();
  mfem::DenseMatrix values(element_state.GetData(), dof, kNumEq);
  mfem::Vector average;
  mfem::dg_igr::limiters_detail::ElementAverage(
    *fes.GetFE(element),
    *fes.GetElementTransformation(element),
    values,
    kNumEq,
    kQuadratureOrder,
    average);
  return average;
}

mfem::Vector ElementValue(mfem::ParFiniteElementSpace& fes,
                          const mfem::ParGridFunction& state,
                          int element,
                          const mfem::IntegrationPoint& point)
{
  mfem::Array<int> vdofs;
  mfem::Vector element_state;
  fes.GetElementVDofs(element, vdofs);
  state.GetSubVector(vdofs, element_state);
  const int dof = fes.GetFE(element)->GetDof();
  mfem::DenseMatrix values(element_state.GetData(), dof, kNumEq);
  mfem::Vector shape(dof), value(kNumEq);
  fes.GetFE(element)->CalcShape(point, shape);
  values.MultTranspose(shape, value);
  return value;
}

double PressureFactor(const mfem::Vector& average,
                      const mfem::Vector& density_limited_point)
{
  mfem::Vector candidate(kNumEq);
  auto pressure_at = [&](double factor)
  {
    candidate = average;
    candidate.Add(factor, density_limited_point);
    candidate.Add(-factor, average);
    return Pressure(candidate);
  };
  if (pressure_at(1.0) >= kPressureMin)
  {
    return 1.0;
  }
  double lo = 0.0;
  double hi = 1.0;
  for (int iteration = 0; iteration < 80; ++iteration)
  {
    const double mid = 0.5 * (lo + hi);
    if (pressure_at(mid) >= kPressureMin)
    {
      lo = mid;
    }
    else
    {
      hi = mid;
    }
  }
  return lo;
}

bool Near(double actual, double expected, double tolerance)
{
  return std::abs(actual - expected) <=
         tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

enum class ManufacturedCase
{
  DensityOnly,
  PressureOnly,
  Sequential
};

const char* CaseName(ManufacturedCase kind)
{
  switch (kind)
  {
  case ManufacturedCase::DensityOnly:
    return "density-only";
  case ManufacturedCase::PressureOnly:
    return "pressure-only";
  case ManufacturedCase::Sequential:
    return "density-and-pressure";
  }
  return "unknown";
}

void SetManufacturedValue(ManufacturedCase kind,
                          double coordinate,
                          mfem::Vector& value)
{
  value = 0.0;
  value(0) = (kind == ManufacturedCase::PressureOnly)
               ? 1.0
               : 1.0 + 1.5 * coordinate;
  value(1) = (kind == ManufacturedCase::DensityOnly)
               ? 0.0
               : -1.5 * coordinate;
  if (kind == ManufacturedCase::PressureOnly)
  {
    value(1) = -3.0 * coordinate;
  }
  value(3) = 2.5;
}

int ManufacturedLimiterCase(ManufacturedCase kind, bool curved)
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  if (curved)
  {
    CurveMesh(serial_mesh);
  }
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::DG_FECollection fec(1, kDim, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(
    &mesh, &fec, kNumEq, mfem::Ordering::byNODES);
  mfem::ParGridFunction state(&fes);

  mfem::Array<int> vdofs;
  for (int element = 0; element < fes.GetNE(); ++element)
  {
    fes.GetElementVDofs(element, vdofs);
    mfem::Vector element_state(vdofs.Size());
    const int dof = fes.GetFE(element)->GetDof();
    mfem::DenseMatrix values(element_state.GetData(), dof, kNumEq);
    const mfem::IntegrationRule& nodes = fes.GetFE(element)->GetNodes();
    for (int node = 0; node < dof; ++node)
    {
      const double coordinate = 2.0 * nodes.IntPoint(node).x - 1.0;
      mfem::Vector value(kNumEq);
      SetManufacturedValue(kind, coordinate, value);
      values.SetRow(node, value);
    }
    state.SetSubVector(vdofs, element_state);
  }

  mfem::IntegrationPoint left;
  left.Set2(0.0, 0.5);
  const mfem::Vector average_before = ElementAverage(fes, state, 0);
  const mfem::Vector point_before = ElementValue(fes, state, 0, left);
  const double expected_theta_rho = point_before(0) < kRhoMin
                                      ? (average_before(0) - kRhoMin) /
                                          (average_before(0) - point_before(0))
                                      : 1.0;
  mfem::Vector density_limited_point(average_before);
  density_limited_point.Add(expected_theta_rho, point_before);
  density_limited_point.Add(-expected_theta_rho, average_before);
  const double expected_theta_pressure =
    PressureFactor(average_before, density_limited_point);
  mfem::Vector expected_point(average_before);
  expected_point.Add(expected_theta_pressure, density_limited_point);
  expected_point.Add(-expected_theta_pressure, average_before);

  const bool exercises_density = expected_theta_rho < 1.0;
  const bool exercises_pressure = expected_theta_pressure < 1.0;
  const bool expected_density = kind != ManufacturedCase::PressureOnly;
  const bool expected_pressure = kind != ManufacturedCase::DensityOnly;
  if (exercises_density != expected_density ||
      exercises_pressure != expected_pressure ||
      !(expected_theta_rho > 0.0) || !(expected_theta_pressure > 0.0))
  {
    std::cerr << "[FAIL] manufactured " << CaseName(kind)
              << " case does not exercise the intended limiter stages"
              << " curved=" << curved
              << " theta_rho=" << expected_theta_rho
              << " theta_p=" << expected_theta_pressure << "\n";
    return 1;
  }

  mfem::ZhangShuLimiterElementDiagnostics diagnostics;
  mfem::ApplyZhangShuPositivityLimiter(
    fes,
    state,
    kDim,
    kGamma,
    kRhoMin,
    kPressureMin,
    kQuadratureOrder,
    {kQuadratureOrder},
    kBisectionIterations,
    mfem::ZhangShuPointSet::kConsumer,
    &diagnostics);

  const mfem::Vector average_after = ElementAverage(fes, state, 0);
  const mfem::Vector point_after = ElementValue(fes, state, 0, left);
  mfem::Vector average_error(average_after);
  average_error -= average_before;
  mfem::Vector point_error(point_after);
  point_error -= expected_point;

  bool ok = true;
  ok = Near(diagnostics.theta_rho.HostRead()[0], expected_theta_rho, 2.0e-12) &&
       ok;
  ok = Near(diagnostics.theta_p.HostRead()[0],
            expected_theta_pressure,
            2.0e-10) && ok;
  ok = diagnostics.collapsed_average.HostRead()[0] == 0.0 && ok;
  ok = average_error.Normlinf() <= 2.0e-12 && ok;
  ok = point_error.Normlinf() <= 2.0e-10 && ok;
  ok = point_after(0) >= 0.99 * kRhoMin && ok;
  ok = Pressure(point_after) >= 0.99 * kPressureMin && ok;

  mfem::ParGridFunction once_limited(&fes);
  once_limited = state;
  mfem::ApplyZhangShuPositivityLimiter(
    fes,
    state,
    kDim,
    kGamma,
    kRhoMin,
    kPressureMin,
    kQuadratureOrder,
    {kQuadratureOrder},
    kBisectionIterations,
    mfem::ZhangShuPointSet::kConsumer,
    nullptr);
  state -= once_limited;
  ok = state.Normlinf() <= 2.0e-12 && ok;

  if (!ok)
  {
    std::cerr << "[FAIL] Zhang-Shu manufactured reference"
              << " case=" << CaseName(kind)
              << " curved=" << curved
              << " expected_theta_rho=" << expected_theta_rho
              << " actual_theta_rho=" << diagnostics.theta_rho.HostRead()[0]
              << " expected_theta_p=" << expected_theta_pressure
              << " actual_theta_p=" << diagnostics.theta_p.HostRead()[0]
              << " average_error=" << average_error.Normlinf()
              << " point_error=" << point_error.Normlinf()
              << " reapply_error=" << state.Normlinf() << "\n";
    return 2;
  }
  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] Zhang-Shu manufactured reference"
              << " case=" << CaseName(kind)
              << " curved=" << curved
              << " theta_rho=" << expected_theta_rho
              << " theta_p=" << expected_theta_pressure << "\n";
  }
  return 0;
}

int InvalidAverageCase()
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::DG_FECollection fec(0, kDim);
  mfem::ParFiniteElementSpace fes(
    &mesh, &fec, kNumEq, mfem::Ordering::byNODES);
  mfem::ParGridFunction state(&fes);
  state = 0.0;
  const int ndofs = fes.GetNDofs();
  double* data = state.HostReadWrite();
  for (int dof = 0; dof < ndofs; ++dof)
  {
    data[dof] = -1.0;
    data[dof + 3 * ndofs] = 0.1;
  }
  const mfem::Vector average_before = ElementAverage(fes, state, 0);

  mfem::ZhangShuLimiterElementDiagnostics diagnostics;
  mfem::ApplyZhangShuPositivityLimiter(
    fes,
    state,
    kDim,
    kGamma,
    kRhoMin,
    kPressureMin,
    kQuadratureOrder,
    {},
    kBisectionIterations,
    mfem::ZhangShuPointSet::kConsumer,
    &diagnostics);
  const mfem::Vector average_after = ElementAverage(fes, state, 0);

  bool ok = diagnostics.collapsed_average.HostRead()[0] == 1.0;
  for (int component = 0; component < kNumEq; ++component)
  {
    ok = Near(diagnostics.average_repair.HostRead()[component],
              average_after(component) - average_before(component),
              2.0e-12) && ok;
  }
  ok = average_after(0) >= kRhoMin && ok;
  ok = Pressure(average_after) >= 0.99 * kPressureMin && ok;
  if (!ok)
  {
    std::cerr << "[FAIL] invalid-average collapse accounting\n";
    return 3;
  }
  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] invalid-average collapse accounting"
              << " delta_rho=" << diagnostics.average_repair.HostRead()[0]
              << " delta_energy="
              << diagnostics.average_repair.HostRead()[3] << "\n";
  }
  return 0;
}

int InvalidPressureAverageCase()
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::DG_FECollection fec(0, kDim);
  mfem::ParFiniteElementSpace fes(
    &mesh, &fec, kNumEq, mfem::Ordering::byNODES);
  mfem::ParGridFunction state(&fes);
  state = 0.0;
  const int ndofs = fes.GetNDofs();
  double* data = state.HostReadWrite();
  for (int dof = 0; dof < ndofs; ++dof)
  {
    data[dof] = 1.0;
    data[dof + ndofs] = 2.0;
    data[dof + 3 * ndofs] = 0.1;
  }
  const mfem::Vector average_before = ElementAverage(fes, state, 0);
  MFEM_VERIFY(Pressure(average_before) < kPressureMin,
              "Invalid-pressure test average must be inadmissible.");

  mfem::ZhangShuLimiterElementDiagnostics diagnostics;
  mfem::ApplyZhangShuPositivityLimiter(
    fes,
    state,
    kDim,
    kGamma,
    kRhoMin,
    kPressureMin,
    kQuadratureOrder,
    {},
    kBisectionIterations,
    mfem::ZhangShuPointSet::kConsumer,
    &diagnostics);
  const mfem::Vector average_after = ElementAverage(fes, state, 0);

  bool ok = diagnostics.collapsed_average.HostRead()[0] == 1.0;
  for (int component = 0; component < kNumEq; ++component)
  {
    ok = Near(diagnostics.average_repair.HostRead()[component],
              average_after(component) - average_before(component),
              2.0e-12) && ok;
  }
  ok = Near(average_after(0), average_before(0), 2.0e-12) && ok;
  ok = Near(average_after(1), average_before(1), 2.0e-12) && ok;
  ok = Pressure(average_after) >= 0.99 * kPressureMin && ok;
  if (!ok)
  {
    std::cerr << "[FAIL] invalid-pressure-average collapse accounting\n";
    return 4;
  }
  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] invalid-pressure-average collapse accounting"
              << " delta_energy="
              << diagnostics.average_repair.HostRead()[3] << "\n";
  }
  return 0;
}

} // namespace

int main(int argc, char** argv)
{
  mfem::Mpi::Init(argc, argv);
  const char* device_config = argc > 1 ? argv[1] : "cpu";
  mfem::Device device(device_config);
  for (const bool curved : {false, true})
  {
    for (const ManufacturedCase kind : {ManufacturedCase::DensityOnly,
                                        ManufacturedCase::PressureOnly,
                                        ManufacturedCase::Sequential})
    {
      const int result = ManufacturedLimiterCase(kind, curved);
      if (result != 0) { return result; }
    }
  }
  const int invalid_density = InvalidAverageCase();
  if (invalid_density != 0) { return invalid_density; }
  const int invalid_pressure = InvalidPressureAverageCase();
  if (invalid_pressure != 0) { return invalid_pressure; }
  return 0;
}

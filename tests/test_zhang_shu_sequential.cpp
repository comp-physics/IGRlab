#include "mfem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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
                            int element,
                            int quadrature_order = kQuadratureOrder)
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
    quadrature_order,
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
  const mfem::Vector state_before(state);

  mfem::ZhangShuLimiterElementDiagnostics diagnostics;
  int rejected_elements = 0;
  try
  {
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
  }
  catch (const mfem::RecoverableLimiterFailure& failure)
  {
    rejected_elements = failure.GlobalRejectedElementCount();
  }
  const mfem::Vector average_after = ElementAverage(fes, state, 0);
  mfem::Vector state_error(state);
  state_error -= state_before;

  bool ok = rejected_elements == 1;
  ok = diagnostics.inadmissible_average.HostRead()[0] == 1.0 && ok;
  ok = diagnostics.collapsed_average.HostRead()[0] == 0.0 && ok;
  for (int component = 0; component < kNumEq; ++component)
  {
    ok = diagnostics.average_repair.HostRead()[component] == 0.0 && ok;
    ok = Near(average_after(component), average_before(component), 2.0e-12) &&
         ok;
  }
  ok = state_error.Normlinf() == 0.0 && ok;
  ok = average_after(0) < kRhoMin && ok;
  if (!ok)
  {
    std::cerr << "[FAIL] invalid-density average was not rejected transactionally\n";
    return 3;
  }
  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] invalid-density average rejected without repair\n";
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
  const mfem::Vector state_before(state);

  mfem::ZhangShuLimiterElementDiagnostics diagnostics;
  int rejected_elements = 0;
  try
  {
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
  }
  catch (const mfem::RecoverableLimiterFailure& failure)
  {
    rejected_elements = failure.GlobalRejectedElementCount();
  }
  const mfem::Vector average_after = ElementAverage(fes, state, 0);
  mfem::Vector state_error(state);
  state_error -= state_before;

  bool ok = rejected_elements == 1;
  ok = diagnostics.inadmissible_average.HostRead()[0] == 1.0 && ok;
  ok = diagnostics.collapsed_average.HostRead()[0] == 0.0 && ok;
  for (int component = 0; component < kNumEq; ++component)
  {
    ok = diagnostics.average_repair.HostRead()[component] == 0.0 && ok;
    ok = Near(average_after(component), average_before(component), 2.0e-12) &&
         ok;
  }
  ok = state_error.Normlinf() == 0.0 && ok;
  ok = Pressure(average_after) < kPressureMin && ok;
  if (!ok)
  {
    std::cerr << "[FAIL] invalid-pressure average was not rejected transactionally\n";
    return 4;
  }
  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] invalid-pressure average rejected without repair\n";
  }
  return 0;
}

int NonFiniteConsumerPointCases()
{
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(
    1, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0);
  mfem::ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  mfem::DG_FECollection fec(1, kDim, mfem::BasisType::GaussLegendre);
  mfem::ParFiniteElementSpace fes(
    &mesh, &fec, kNumEq, mfem::Ordering::byNODES);
  const mfem::IntegrationRule& nodes = fes.GetFE(0)->GetNodes();
  const double finite_max = std::numeric_limits<double>::max();

  auto set_state = [&](mfem::ParGridFunction& state,
                       bool overflow_component)
  {
    state = 0.0;
    mfem::Array<int> vdofs;
    fes.GetElementVDofs(0, vdofs);
    mfem::Vector element_state(vdofs.Size());
    mfem::DenseMatrix values(element_state.GetData(),
                             fes.GetFE(0)->GetDof(),
                             kNumEq);
    for (int node = 0; node < nodes.GetNPoints(); ++node)
    {
      mfem::Vector value(kNumEq);
      value = 0.0;
      if (overflow_component)
      {
        value(0) = nodes.IntPoint(node).x < 0.5
                     ? 0.9 * finite_max
                     : 0.4 * finite_max;
      }
      else
      {
        value(0) = 1.0;
        const double momentum_amplitude = 2.0 * std::sqrt(finite_max);
        value(1) = nodes.IntPoint(node).x < 0.5
                     ? momentum_amplitude
                     : -momentum_amplitude;
      }
      value(3) = 0.2 * finite_max;
      values.SetRow(node, value);
    }
    state.SetSubVector(vdofs, element_state);
  };

  auto reject_transactionally = [&](mfem::ParGridFunction& state,
                                    int quadrature_order,
                                    const std::string& label)
  {
    const mfem::Vector before(state);
    int rejected_elements = 0;
    bool generic_failure_report = false;
    try
    {
      mfem::ApplyZhangShuPositivityLimiter(
        fes,
        state,
        kDim,
        kGamma,
        kRhoMin,
        kPressureMin,
        quadrature_order,
        {quadrature_order},
        kBisectionIterations,
        mfem::ZhangShuPointSet::kConsumer,
        nullptr);
    }
    catch (const mfem::RecoverableLimiterFailure& failure)
    {
      rejected_elements = failure.GlobalRejectedElementCount();
      generic_failure_report =
        std::string(failure.what()).find("failed limiter admission") !=
        std::string::npos;
    }
    mfem::Vector error(state);
    error -= before;
    if (rejected_elements != 1 || !generic_failure_report ||
        error.Normlinf() != 0.0)
    {
      std::cerr << "[FAIL] " << label
                << " was not rejected collectively and transactionally\n";
      return false;
    }
    return true;
  };

  mfem::ParGridFunction component_overflow(&fes);
  set_state(component_overflow, /*overflow_component=*/true);
  const mfem::Vector component_average =
    ElementAverage(fes, component_overflow, 0, /*quadrature_order=*/2);
  mfem::IntegrationPoint left_face_midpoint;
  left_face_midpoint.Set2(0.0, 0.5);
  const mfem::Vector component_face_value =
    ElementValue(fes, component_overflow, 0, left_face_midpoint);
  bool ok = std::isfinite(component_average(0)) &&
            std::isfinite(Pressure(component_average)) &&
            !std::isfinite(component_face_value(0));
  ok = reject_transactionally(component_overflow,
                              /*quadrature_order=*/2,
                              "non-finite consumer density") &&
       ok;

  mfem::ParGridFunction pressure_overflow(&fes);
  set_state(pressure_overflow, /*overflow_component=*/false);
  const mfem::Vector pressure_average =
    ElementAverage(fes, pressure_overflow, 0);
  const mfem::Vector pressure_node =
    ElementValue(fes, pressure_overflow, 0, nodes.IntPoint(0));
  ok = std::isfinite(pressure_average(0)) &&
       std::isfinite(Pressure(pressure_average)) &&
       !std::isfinite(Pressure(pressure_node)) && ok;
  ok = reject_transactionally(pressure_overflow,
                              kQuadratureOrder,
                              "non-finite consumer pressure") &&
       ok;

  if (!ok)
  {
    std::cerr << "[FAIL] non-finite consumer-point test setup or rejection\n";
    return 5;
  }
  if (mfem::Mpi::Root())
  {
    std::cout << "[PASS] non-finite consumer density and pressure rejected "
                 "without relying on the optional NaN scan\n";
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
  const int nonfinite_consumer = NonFiniteConsumerPointCases();
  if (nonfinite_consumer != 0) { return nonfinite_consumer; }
  return 0;
}

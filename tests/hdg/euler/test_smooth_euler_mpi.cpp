#include "numerics/hdg/euler/smooth_euler_mpi.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{

using dg::hdg::euler::SmoothEulerMPICondensation;
using dg::hdg::euler::SmoothEulerMPIEvidence;
using dg::hdg::euler::SmoothEulerMPIExperiment;
using dg::hdg::euler::SmoothEulerMPIFailureInjection;
using dg::hdg::euler::SmoothEulerMPIFailurePoint;
using dg::hdg::euler::SmoothEulerMPILinearization;
using dg::hdg::euler::SmoothEulerMPIStatus;
using dg::hdg::euler::ExperimentStatus;
using dg::hdg::euler::ConvergenceEvidence;
using dg::hdg::euler::SmoothEulerExperiment;
using dg::hdg::euler::SolvedMeshEvidence;
using dg::hdg::euler::TrialAuditSnapshot;
using dg::hdg::Ownership;
using dg::hdg::trace::TraceMapEntry;

constexpr int dimension = 2;
constexpr int components = 4;
constexpr double gamma_value = 1.4;
constexpr double tau_value = 2.5;
constexpr double dt_value = 0.05;
constexpr double oracle_pi = 3.141592653589793238462643383279502884;
constexpr double local_tolerance = 5.0e-11;
constexpr double state_tolerance = 5.0e-9;
constexpr double true_residual_tolerance = 1.0e-10;

using State = std::array<double, components>;
using Jacobian = std::array<std::array<double, components>, components>;

int Index(const int scalar, const int scalar_count, const int component)
{
  return scalar + scalar_count * component;
}

int Decode(const int signed_dof, double& sign)
{
  sign = signed_dof < 0 ? -1.0 : 1.0;
  return signed_dof < 0 ? -1 - signed_dof : signed_dof;
}

bool Finite(const double value) { return std::isfinite(value); }

bool Finite(const std::vector<double>& values)
{
  return std::all_of(values.begin(), values.end(),
                     [](const double value) { return Finite(value); });
}

bool GlobalAll(MPI_Comm communicator, const bool local)
{
  int local_value = local ? 1 : 0;
  int global_value = 0;
  MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                communicator);
  return global_value == 1;
}

double Pressure(const State& state)
{
  return (gamma_value - 1.0) *
      (state[3] - 0.5 * (state[1] * state[1] + state[2] * state[2]) /
                        state[0]);
}

State PrimitiveToConservative(const State& primitive)
{
  const double density = primitive[0];
  const double u = primitive[1];
  const double v = primitive[2];
  return {{density, density * u, density * v,
           primitive[3] / (gamma_value - 1.0) +
               0.5 * density * (u * u + v * v)}};
}

State ManufacturedPrimitive(const double x, const double y)
{
  return {{
      1.0 + 0.1 * std::sin(2.0 * oracle_pi * x) *
                    std::sin(2.0 * oracle_pi * y),
      0.35 + 0.05 * std::cos(2.0 * oracle_pi * x) *
                     std::sin(2.0 * oracle_pi * y),
      -0.20 + 0.05 * std::sin(2.0 * oracle_pi * x) *
                      std::cos(2.0 * oracle_pi * y),
      1.0 + 0.1 * std::cos(2.0 * oracle_pi * x) *
                    std::cos(2.0 * oracle_pi * y)}};
}

State ManufacturedDerivative(const double x, const double y,
                             const int direction)
{
  const double sx = std::sin(2.0 * oracle_pi * x);
  const double sy = std::sin(2.0 * oracle_pi * y);
  const double cx = std::cos(2.0 * oracle_pi * x);
  const double cy = std::cos(2.0 * oracle_pi * y);
  const State primitive = ManufacturedPrimitive(x, y);
  const double density_derivative = direction == 0 ?
      0.2 * oracle_pi * cx * sy : 0.2 * oracle_pi * sx * cy;
  const double u_derivative = direction == 0 ?
      -0.1 * oracle_pi * sx * sy : 0.1 * oracle_pi * cx * cy;
  const double v_derivative = direction == 0 ?
      0.1 * oracle_pi * cx * cy : -0.1 * oracle_pi * sx * sy;
  const double pressure_derivative = direction == 0 ?
      -0.2 * oracle_pi * sx * cy : -0.2 * oracle_pi * cx * sy;
  const double density = primitive[0];
  const double u = primitive[1];
  const double v = primitive[2];
  const double energy_derivative =
      pressure_derivative / (gamma_value - 1.0) +
      0.5 * density_derivative * (u * u + v * v) +
      density * (u * u_derivative + v * v_derivative);
  return {{density_derivative,
           density_derivative * u + density * u_derivative,
           density_derivative * v + density * v_derivative,
           energy_derivative}};
}

State NormalFlux(const State& state,
                 const std::array<double, dimension>& normal)
{
  const double pressure = Pressure(state);
  const double normal_momentum =
      state[1] * normal[0] + state[2] * normal[1];
  return {{normal_momentum,
           state[1] * normal_momentum / state[0] + pressure * normal[0],
           state[2] * normal_momentum / state[0] + pressure * normal[1],
           (state[3] + pressure) * normal_momentum / state[0]}};
}

Jacobian NormalJacobian(const State& state,
                        const std::array<double, dimension>& normal)
{
  const double density = state[0];
  const double mx = state[1];
  const double my = state[2];
  const double energy = state[3];
  const double normal_momentum = mx * normal[0] + my * normal[1];
  const double pressure = Pressure(state);
  const double p_density = (gamma_value - 1.0) * 0.5 *
      (mx * mx + my * my) / (density * density);
  const double p_mx = -(gamma_value - 1.0) * mx / density;
  const double p_my = -(gamma_value - 1.0) * my / density;
  const double p_energy = gamma_value - 1.0;
  const double enthalpy = energy + pressure;
  return {{
      {{0.0, normal[0], normal[1], 0.0}},
      {{-mx * normal_momentum / (density * density) +
            p_density * normal[0],
        normal_momentum / density + mx * normal[0] / density +
            p_mx * normal[0],
        mx * normal[1] / density + p_my * normal[0],
        p_energy * normal[0]}},
      {{-my * normal_momentum / (density * density) +
            p_density * normal[1],
        my * normal[0] / density + p_mx * normal[1],
        normal_momentum / density + my * normal[1] / density +
            p_my * normal[1],
        p_energy * normal[1]}},
      {{p_density * normal_momentum / density -
            enthalpy * normal_momentum / (density * density),
        p_mx * normal_momentum / density + enthalpy * normal[0] / density,
        p_my * normal_momentum / density + enthalpy * normal[1] / density,
        (1.0 + p_energy) * normal_momentum / density}}}};
}

State ManufacturedSource(const double x, const double y)
{
  const State conservative =
      PrimitiveToConservative(ManufacturedPrimitive(x, y));
  const Jacobian x_jacobian = NormalJacobian(conservative, {{1.0, 0.0}});
  const Jacobian y_jacobian = NormalJacobian(conservative, {{0.0, 1.0}});
  const State x_derivative = ManufacturedDerivative(x, y, 0);
  const State y_derivative = ManufacturedDerivative(x, y, 1);
  State source{};
  for (int row = 0; row < components; ++row)
    for (int column = 0; column < components; ++column)
    {
      source[row] += x_jacobian[row][column] * x_derivative[column] +
          y_jacobian[row][column] * y_derivative[column];
    }
  return source;
}

bool Interpolate(const mfem::Vector& field,
                 const mfem::Array<int>& signed_vdofs,
                 const mfem::Vector& shape, State& state)
{
  if (signed_vdofs.Size() != components * shape.Size()) { return false; }
  const double* const data = field.HostRead();
  if (data == nullptr) { return false; }
  state = {};
  for (int scalar = 0; scalar < shape.Size(); ++scalar)
    for (int component = 0; component < components; ++component)
    {
      double sign = 1.0;
      const int dof = Decode(
          signed_vdofs[Index(scalar, shape.Size(), component)], sign);
      if (dof < 0 || dof >= field.Size()) { return false; }
      state[component] += shape[scalar] * sign * data[dof];
    }
  return true;
}

struct OracleSystem
{
  int elements = 0;
  int ni = 0;
  int nt = 0;
  std::vector<double> residual_interior;
  std::vector<double> residual_trace;
  std::vector<double> a, b, c, d, f, g;
  std::vector<int> signed_trace_dofs;
};

bool AssembleOracle(mfem::Mesh& mesh,
                    mfem::FiniteElementSpace& interior_space,
                    mfem::FiniteElementSpace& trace_space,
                    const mfem::GridFunction& old_interior,
                    const mfem::GridFunction& current_interior,
                    const mfem::GridFunction& current_trace,
                    const int polynomial_order, OracleSystem& oracle)
{
  OracleSystem next;
  next.elements = mesh.GetNE();
  const mfem::FiniteElement* const first = interior_space.GetFE(0);
  if (first == nullptr || next.elements <= 0) { return false; }
  next.ni = components * first->GetDof();
  mfem::Array<int> first_faces, first_orientations;
  mesh.GetElementEdges(0, first_faces, first_orientations);
  for (int local_face = 0; local_face < first_faces.Size(); ++local_face)
  {
    mfem::Array<int> vdofs;
    trace_space.GetFaceVDofs(first_faces[local_face], vdofs);
    next.nt += vdofs.Size();
  }
  if (next.ni <= 0 || next.nt <= 0) { return false; }
  const std::size_t batches = static_cast<std::size_t>(next.elements);
  const std::size_t ni = static_cast<std::size_t>(next.ni);
  const std::size_t nt = static_cast<std::size_t>(next.nt);
  next.residual_interior.assign(batches * ni, 0.0);
  next.residual_trace.assign(batches * nt, 0.0);
  next.a.assign(batches * ni * ni, 0.0);
  next.b.assign(batches * ni * nt, 0.0);
  next.c.assign(batches * nt * ni, 0.0);
  next.d.assign(batches * nt * nt, 0.0);
  next.signed_trace_dofs.assign(batches * nt, 0);
  const int quadrature_order = 3 * polynomial_order + 8;

  for (int element = 0; element < next.elements; ++element)
  {
    const mfem::FiniteElement* const finite_element =
        interior_space.GetFE(element);
    mfem::ElementTransformation* const transformation =
        mesh.GetElementTransformation(element);
    if (finite_element == nullptr || transformation == nullptr) { return false; }
    const int scalar_dofs = finite_element->GetDof();
    mfem::Array<int> interior_vdofs;
    interior_space.GetElementVDofs(element, interior_vdofs);
    if (interior_vdofs.Size() != next.ni) { return false; }
    const mfem::IntegrationRule& rule = mfem::IntRules.Get(
        finite_element->GetGeomType(), quadrature_order);
    for (int point_index = 0; point_index < rule.GetNPoints(); ++point_index)
    {
      const mfem::IntegrationPoint& point = rule.IntPoint(point_index);
      transformation->SetIntPoint(&point);
      mfem::Vector physical(dimension), shape(scalar_dofs);
      mfem::DenseMatrix derivative(scalar_dofs, dimension);
      transformation->Transform(point, physical);
      finite_element->CalcShape(point, shape);
      finite_element->CalcPhysDShape(*transformation, derivative);
      State state{}, old_state{};
      if (!Interpolate(current_interior, interior_vdofs, shape, state) ||
          !Interpolate(old_interior, interior_vdofs, shape, old_state))
      {
        return false;
      }
      const State source = ManufacturedSource(physical[0], physical[1]);
      const State flux_x = NormalFlux(state, {{1.0, 0.0}});
      const State flux_y = NormalFlux(state, {{0.0, 1.0}});
      const Jacobian jacobian_x = NormalJacobian(state, {{1.0, 0.0}});
      const Jacobian jacobian_y = NormalJacobian(state, {{0.0, 1.0}});
      const double weight = point.weight * transformation->Weight();
      for (int row = 0; row < scalar_dofs; ++row)
      {
        for (int row_component = 0; row_component < components;
             ++row_component)
        {
          const int local_row = Index(row, scalar_dofs, row_component);
          next.residual_interior[
              static_cast<std::size_t>(element) * ni + local_row] +=
              weight *
              (shape[row] * (state[row_component] - old_state[row_component]) /
                   dt_value -
               derivative(row, 0) * flux_x[row_component] -
               derivative(row, 1) * flux_y[row_component] -
               shape[row] * source[row_component]);
        }
        for (int column = 0; column < scalar_dofs; ++column)
          for (int row_component = 0; row_component < components;
               ++row_component)
            for (int column_component = 0; column_component < components;
                 ++column_component)
            {
              const int local_row = Index(row, scalar_dofs, row_component);
              const int local_column =
                  Index(column, scalar_dofs, column_component);
              const double mass = row_component == column_component ?
                  shape[row] * shape[column] / dt_value : 0.0;
              const double volume =
                  -(derivative(row, 0) *
                        jacobian_x[row_component][column_component] +
                    derivative(row, 1) *
                        jacobian_y[row_component][column_component]) *
                  shape[column];
              next.a[static_cast<std::size_t>(element) * ni * ni +
                     static_cast<std::size_t>(local_row) * ni +
                     local_column] += weight * (mass + volume);
            }
      }
    }

    mfem::Array<int> faces, orientations;
    mesh.GetElementEdges(element, faces, orientations);
    int trace_offset = 0;
    for (int local_face = 0; local_face < faces.Size(); ++local_face)
    {
      const int face = faces[local_face];
      mfem::FaceElementTransformations* const face_transformation =
          mesh.GetFaceElementTransformations(face);
      const mfem::FiniteElement* const trace_element =
          trace_space.GetFaceElement(face);
      mfem::Array<int> trace_vdofs;
      trace_space.GetFaceVDofs(face, trace_vdofs);
      if (face_transformation == nullptr ||
          face_transformation->Face == nullptr || trace_element == nullptr ||
          trace_vdofs.Size() != components * trace_element->GetDof())
      {
        return false;
      }
      std::copy(trace_vdofs.begin(), trace_vdofs.end(),
                next.signed_trace_dofs.begin() +
                    static_cast<std::size_t>(element) * nt + trace_offset);
      const bool element_one = face_transformation->Elem1No == element;
      if (!element_one && face_transformation->Elem2No != element)
      {
        return false;
      }
      const int trace_scalar_dofs = trace_element->GetDof();
      const mfem::IntegrationRule& face_rule = mfem::IntRules.Get(
          trace_element->GetGeomType(), quadrature_order);
      mfem::Vector interior_shape(scalar_dofs),
          trace_shape(trace_scalar_dofs);
      for (int point_index = 0; point_index < face_rule.GetNPoints();
           ++point_index)
      {
        const mfem::IntegrationPoint& face_point =
            face_rule.IntPoint(point_index);
        face_transformation->SetAllIntPoints(&face_point);
        const mfem::IntegrationPoint& element_point = element_one ?
            face_transformation->GetElement1IntPoint() :
            face_transformation->GetElement2IntPoint();
        mfem::Vector scaled_normal(dimension);
        mfem::CalcOrtho(face_transformation->Face->Jacobian(), scaled_normal);
        const double measure = scaled_normal.Norml2();
        if (!(measure > 0.0) || !Finite(measure)) { return false; }
        const double side = element_one ? 1.0 : -1.0;
        const std::array<double, dimension> normal{{
            side * scaled_normal[0] / measure,
            side * scaled_normal[1] / measure}};
        finite_element->CalcShape(element_point, interior_shape);
        trace_element->CalcShape(face_point, trace_shape);
        State interior_state{}, trace_state{};
        if (!Interpolate(current_interior, interior_vdofs, interior_shape,
                         interior_state) ||
            !Interpolate(current_trace, trace_vdofs, trace_shape, trace_state))
        {
          return false;
        }
        const State trace_flux = NormalFlux(trace_state, normal);
        const Jacobian trace_jacobian = NormalJacobian(trace_state, normal);
        const double weight = face_point.weight * measure;
        for (int row = 0; row < scalar_dofs; ++row)
          for (int component = 0; component < components; ++component)
          {
            const int local_row = Index(row, scalar_dofs, component);
            next.residual_interior[
                static_cast<std::size_t>(element) * ni + local_row] +=
                weight * interior_shape[row] *
                (trace_flux[component] + tau_value *
                    (interior_state[component] - trace_state[component]));
          }
        for (int row = 0; row < trace_scalar_dofs; ++row)
          for (int component = 0; component < components; ++component)
          {
            const int local_row = trace_offset +
                Index(row, trace_scalar_dofs, component);
            next.residual_trace[
                static_cast<std::size_t>(element) * nt + local_row] +=
                weight * trace_shape[row] *
                (trace_flux[component] + tau_value *
                    (interior_state[component] - trace_state[component]));
          }
        for (int row = 0; row < scalar_dofs; ++row)
          for (int column = 0; column < scalar_dofs; ++column)
            for (int component = 0; component < components; ++component)
            {
              const int local_row = Index(row, scalar_dofs, component);
              const int local_column = Index(column, scalar_dofs, component);
              next.a[static_cast<std::size_t>(element) * ni * ni +
                     static_cast<std::size_t>(local_row) * ni +
                     local_column] += weight * interior_shape[row] *
                  tau_value * interior_shape[column];
            }
        for (int row = 0; row < scalar_dofs; ++row)
          for (int column = 0; column < trace_scalar_dofs; ++column)
            for (int row_component = 0; row_component < components;
                 ++row_component)
              for (int column_component = 0; column_component < components;
                   ++column_component)
              {
                const int local_row = Index(row, scalar_dofs, row_component);
                const int local_column = trace_offset +
                    Index(column, trace_scalar_dofs, column_component);
                next.b[static_cast<std::size_t>(element) * ni * nt +
                       static_cast<std::size_t>(local_row) * nt +
                       local_column] += weight * interior_shape[row] *
                    trace_shape[column] *
                    (trace_jacobian[row_component][column_component] -
                     (row_component == column_component ? tau_value : 0.0));
              }
        for (int row = 0; row < trace_scalar_dofs; ++row)
          for (int column = 0; column < scalar_dofs; ++column)
            for (int component = 0; component < components; ++component)
            {
              const int local_row = trace_offset +
                  Index(row, trace_scalar_dofs, component);
              const int local_column = Index(column, scalar_dofs, component);
              next.c[static_cast<std::size_t>(element) * nt * ni +
                     static_cast<std::size_t>(local_row) * ni +
                     local_column] += weight * trace_shape[row] * tau_value *
                  interior_shape[column];
            }
        for (int row = 0; row < trace_scalar_dofs; ++row)
          for (int column = 0; column < trace_scalar_dofs; ++column)
            for (int row_component = 0; row_component < components;
                 ++row_component)
              for (int column_component = 0; column_component < components;
                   ++column_component)
              {
                const int local_row = trace_offset +
                    Index(row, trace_scalar_dofs, row_component);
                const int local_column = trace_offset +
                    Index(column, trace_scalar_dofs, column_component);
                next.d[static_cast<std::size_t>(element) * nt * nt +
                       static_cast<std::size_t>(local_row) * nt +
                       local_column] += weight * trace_shape[row] *
                    trace_shape[column] *
                    (trace_jacobian[row_component][column_component] -
                     (row_component == column_component ? tau_value : 0.0));
              }
      }
      trace_offset += trace_vdofs.Size();
    }
    if (trace_offset != next.nt) { return false; }
  }
  next.f.resize(next.residual_interior.size());
  next.g.resize(next.residual_trace.size());
  std::transform(next.residual_interior.begin(),
                 next.residual_interior.end(), next.f.begin(),
                 [](const double value) { return -value; });
  std::transform(next.residual_trace.begin(), next.residual_trace.end(),
                 next.g.begin(),
                 [](const double value) { return -value; });
  if (!Finite(next.residual_interior) || !Finite(next.residual_trace) ||
      !Finite(next.a) || !Finite(next.b) || !Finite(next.c) ||
      !Finite(next.d))
  {
    return false;
  }
  oracle = std::move(next);
  return true;
}

double ScaledError(const std::vector<double>& actual,
                   const std::vector<double>& expected)
{
  if (actual.size() != expected.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  double difference = 0.0;
  double scale = 1.0;
  for (std::size_t index = 0; index < actual.size(); ++index)
  {
    if (!Finite(actual[index]) || !Finite(expected[index]))
    {
      return std::numeric_limits<double>::infinity();
    }
    difference = std::max(difference,
                          std::abs(actual[index] - expected[index]));
    scale = std::max(scale, std::abs(expected[index]));
  }
  return difference / scale;
}

struct LUFactorization
{
  int size = 0;
  std::vector<double> lu;
  std::vector<int> pivots;

  bool Factor(const double* matrix, const int matrix_size)
  {
    size = matrix_size;
    if (matrix == nullptr || size <= 0) { return false; }
    lu.assign(matrix, matrix + static_cast<std::size_t>(size) * size);
    pivots.resize(static_cast<std::size_t>(size));
    for (int pivot = 0; pivot < size; ++pivot)
    {
      int selected = pivot;
      double magnitude = std::abs(lu[
          static_cast<std::size_t>(pivot) * size + pivot]);
      for (int row = pivot + 1; row < size; ++row)
      {
        const double candidate = std::abs(lu[
            static_cast<std::size_t>(row) * size + pivot]);
        if (candidate > magnitude)
        {
          magnitude = candidate;
          selected = row;
        }
      }
      if (!Finite(magnitude) || magnitude <=
          128.0 * std::numeric_limits<double>::epsilon())
      {
        return false;
      }
      pivots[static_cast<std::size_t>(pivot)] = selected;
      if (selected != pivot)
      {
        for (int column = 0; column < size; ++column)
        {
          std::swap(lu[static_cast<std::size_t>(pivot) * size + column],
                    lu[static_cast<std::size_t>(selected) * size + column]);
        }
      }
      for (int row = pivot + 1; row < size; ++row)
      {
        double& multiplier =
            lu[static_cast<std::size_t>(row) * size + pivot];
        multiplier /= lu[static_cast<std::size_t>(pivot) * size + pivot];
        for (int column = pivot + 1; column < size; ++column)
        {
          lu[static_cast<std::size_t>(row) * size + column] -=
              multiplier *
              lu[static_cast<std::size_t>(pivot) * size + column];
        }
      }
    }
    return Finite(lu);
  }

  bool Solve(const std::vector<double>& right_hand_side,
             std::vector<double>& solution) const
  {
    if (size <= 0 || right_hand_side.size() !=
            static_cast<std::size_t>(size))
    {
      return false;
    }
    solution = right_hand_side;
    for (int pivot = 0; pivot < size; ++pivot)
    {
      const int selected = pivots[static_cast<std::size_t>(pivot)];
      if (selected != pivot)
      {
        std::swap(solution[static_cast<std::size_t>(pivot)],
                  solution[static_cast<std::size_t>(selected)]);
      }
    }
    for (int row = 0; row < size; ++row)
      for (int column = 0; column < row; ++column)
      {
        solution[static_cast<std::size_t>(row)] -=
            lu[static_cast<std::size_t>(row) * size + column] *
            solution[static_cast<std::size_t>(column)];
      }
    for (int row = size - 1; row >= 0; --row)
    {
      for (int column = row + 1; column < size; ++column)
      {
        solution[static_cast<std::size_t>(row)] -=
            lu[static_cast<std::size_t>(row) * size + column] *
            solution[static_cast<std::size_t>(column)];
      }
      solution[static_cast<std::size_t>(row)] /=
          lu[static_cast<std::size_t>(row) * size + row];
    }
    return Finite(solution);
  }
};

bool CondenseOracle(const OracleSystem& system,
                    std::vector<double>& schur,
                    std::vector<double>& right_hand_side,
                    std::vector<LUFactorization>* factorizations = nullptr)
{
  schur.assign(static_cast<std::size_t>(system.elements) * system.nt *
                   system.nt,
               0.0);
  right_hand_side.assign(
      static_cast<std::size_t>(system.elements) * system.nt, 0.0);
  if (factorizations != nullptr)
  {
    factorizations->clear();
    factorizations->reserve(static_cast<std::size_t>(system.elements));
  }
  for (int element = 0; element < system.elements; ++element)
  {
    LUFactorization factorization;
    const double* const a = system.a.data() +
        static_cast<std::size_t>(element) * system.ni * system.ni;
    if (!factorization.Factor(a, system.ni)) { return false; }
    std::vector<double> f(
        system.f.begin() + static_cast<std::size_t>(element) * system.ni,
        system.f.begin() + static_cast<std::size_t>(element + 1) * system.ni);
    std::vector<double> inverse_f;
    if (!factorization.Solve(f, inverse_f)) { return false; }
    std::vector<double> inverse_b(
        static_cast<std::size_t>(system.ni) * system.nt, 0.0);
    for (int trace_column = 0; trace_column < system.nt; ++trace_column)
    {
      std::vector<double> column(static_cast<std::size_t>(system.ni), 0.0);
      for (int row = 0; row < system.ni; ++row)
      {
        column[static_cast<std::size_t>(row)] = system.b[
            static_cast<std::size_t>(element) * system.ni * system.nt +
            static_cast<std::size_t>(row) * system.nt + trace_column];
      }
      std::vector<double> solved;
      if (!factorization.Solve(column, solved)) { return false; }
      for (int row = 0; row < system.ni; ++row)
      {
        inverse_b[static_cast<std::size_t>(row) * system.nt +
                  trace_column] = solved[static_cast<std::size_t>(row)];
      }
    }
    for (int row = 0; row < system.nt; ++row)
    {
      double reduced = system.g[
          static_cast<std::size_t>(element) * system.nt + row];
      for (int interior = 0; interior < system.ni; ++interior)
      {
        const double c = system.c[
            static_cast<std::size_t>(element) * system.nt * system.ni +
            static_cast<std::size_t>(row) * system.ni + interior];
        reduced -= c * inverse_f[static_cast<std::size_t>(interior)];
      }
      right_hand_side[
          static_cast<std::size_t>(element) * system.nt + row] = reduced;
      for (int column = 0; column < system.nt; ++column)
      {
        double value = system.d[
            static_cast<std::size_t>(element) * system.nt * system.nt +
            static_cast<std::size_t>(row) * system.nt + column];
        for (int interior = 0; interior < system.ni; ++interior)
        {
          value -= system.c[
              static_cast<std::size_t>(element) * system.nt * system.ni +
              static_cast<std::size_t>(row) * system.ni + interior] *
              inverse_b[static_cast<std::size_t>(interior) * system.nt +
                        column];
        }
        schur[static_cast<std::size_t>(element) * system.nt * system.nt +
               static_cast<std::size_t>(row) * system.nt + column] = value;
      }
    }
    if (factorizations != nullptr)
    {
      factorizations->push_back(std::move(factorization));
    }
  }
  return Finite(schur) && Finite(right_hand_side);
}

struct GlobalOracle
{
  int size = 0;
  std::vector<double> matrix;
  std::vector<double> right_hand_side;
  std::vector<double> solution;
  std::vector<double> probe;
  std::vector<double> probe_action;
};

bool AssembleGlobalOracle(
    MPI_Comm communicator, const OracleSystem& local,
    const dg::hdg::trace::DGInterfaceTraceMap& map,
    const std::vector<double>& schur,
    const std::vector<double>& reduced_rhs, GlobalOracle& oracle)
{
  GlobalOracle next;
  const long long local_global_size = static_cast<long long>(
      map.dofs.global_true_dof_count);
  long long minimum_global_size = 0;
  long long maximum_global_size = 0;
  MPI_Allreduce(&local_global_size, &minimum_global_size, 1,
                MPI_LONG_LONG_INT, MPI_MIN, communicator);
  MPI_Allreduce(&local_global_size, &maximum_global_size, 1,
                MPI_LONG_LONG_INT, MPI_MAX, communicator);
  bool local_ok = local_global_size > 0 &&
      local_global_size <= std::numeric_limits<int>::max() &&
      minimum_global_size == maximum_global_size && local.elements > 0 &&
      local.nt > 0 &&
      local.signed_trace_dofs.size() ==
          static_cast<std::size_t>(local.elements) * local.nt &&
      schur.size() == static_cast<std::size_t>(local.elements) * local.nt *
          local.nt &&
      reduced_rhs.size() ==
          static_cast<std::size_t>(local.elements) * local.nt &&
      map.dofs.entries.size() > 0;
  if (!GlobalAll(communicator, local_ok))
  {
    return false;
  }
  next.size = static_cast<int>(local_global_size);
  std::vector<double> local_matrix;
  std::vector<double> local_rhs;
  try
  {
    local_matrix.assign(
        static_cast<std::size_t>(next.size) * next.size, 0.0);
    local_rhs.assign(static_cast<std::size_t>(next.size), 0.0);
    for (int element = 0; element < local.elements; ++element)
      for (int row = 0; row < local.nt; ++row)
      {
        double row_sign = 1.0;
        const int local_row = Decode(local.signed_trace_dofs[
            static_cast<std::size_t>(element) * local.nt + row], row_sign);
        local_ok = local_row >= 0 &&
            static_cast<std::size_t>(local_row) < map.dofs.entries.size();
        if (!local_ok) { break; }
        const std::int64_t global_row = map.dofs.entries[
            static_cast<std::size_t>(local_row)].global_true_dof;
        local_ok = global_row >= 0 && global_row < next.size;
        if (!local_ok) { break; }
        local_rhs[static_cast<std::size_t>(global_row)] += row_sign *
            reduced_rhs[static_cast<std::size_t>(element) * local.nt + row];
        for (int column = 0; column < local.nt; ++column)
        {
          double column_sign = 1.0;
          const int local_column = Decode(local.signed_trace_dofs[
              static_cast<std::size_t>(element) * local.nt + column],
              column_sign);
          local_ok = local_column >= 0 &&
              static_cast<std::size_t>(local_column) <
                  map.dofs.entries.size();
          if (!local_ok) { break; }
          const std::int64_t global_column = map.dofs.entries[
              static_cast<std::size_t>(local_column)].global_true_dof;
          local_ok = global_column >= 0 && global_column < next.size;
          if (!local_ok) { break; }
          local_matrix[static_cast<std::size_t>(global_row) * next.size +
                       global_column] += row_sign * column_sign * schur[
              static_cast<std::size_t>(element) * local.nt * local.nt +
              static_cast<std::size_t>(row) * local.nt + column];
        }
        if (!local_ok) { break; }
      }
    next.matrix.resize(local_matrix.size());
    next.right_hand_side.resize(local_rhs.size());
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!GlobalAll(communicator, local_ok)) { return false; }

  MPI_Allreduce(local_matrix.data(), next.matrix.data(),
                static_cast<int>(local_matrix.size()), MPI_DOUBLE, MPI_SUM,
                communicator);
  MPI_Allreduce(local_rhs.data(), next.right_hand_side.data(), next.size,
                MPI_DOUBLE, MPI_SUM, communicator);
  try
  {
    LUFactorization factorization;
    local_ok = factorization.Factor(next.matrix.data(), next.size) &&
        factorization.Solve(next.right_hand_side, next.solution);
    next.probe.resize(static_cast<std::size_t>(next.size));
    next.probe_action.assign(static_cast<std::size_t>(next.size), 0.0);
    for (int row = 0; row < next.size; ++row)
    {
      next.probe[static_cast<std::size_t>(row)] =
          0.25 + std::sin(0.01 * static_cast<double>(row + 1));
    }
    for (int row = 0; row < next.size; ++row)
      for (int column = 0; column < next.size; ++column)
      {
        next.probe_action[static_cast<std::size_t>(row)] +=
            next.matrix[static_cast<std::size_t>(row) * next.size + column] *
            next.probe[static_cast<std::size_t>(column)];
      }
    local_ok = local_ok && Finite(next.matrix) &&
        Finite(next.right_hand_side) && Finite(next.solution) &&
        Finite(next.probe) && Finite(next.probe_action);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!GlobalAll(communicator, local_ok)) { return false; }
  oracle = std::move(next);
  return true;
}

double DenseRelativeResidual(const GlobalOracle& oracle)
{
  double residual_squared = 0.0;
  double rhs_squared = 0.0;
  for (int row = 0; row < oracle.size; ++row)
  {
    double action = 0.0;
    for (int column = 0; column < oracle.size; ++column)
    {
      action += oracle.matrix[static_cast<std::size_t>(row) * oracle.size +
                              column] *
          oracle.solution[static_cast<std::size_t>(column)];
    }
    const double residual =
        action - oracle.right_hand_side[static_cast<std::size_t>(row)];
    residual_squared += residual * residual;
    const double rhs = oracle.right_hand_side[static_cast<std::size_t>(row)];
    rhs_squared += rhs * rhs;
  }
  return std::sqrt(residual_squared) /
      std::max(std::sqrt(rhs_squared), 1.0e-300);
}

bool Assign(mfem::Vector& vector, const std::vector<double>& values)
{
  if (vector.Size() != static_cast<int>(values.size())) { return false; }
  double* const data = vector.HostWrite();
  if (data == nullptr && !values.empty()) { return false; }
  if (!values.empty())
  {
    std::memcpy(data, values.data(), values.size() * sizeof(double));
  }
  return true;
}

std::vector<double> Values(const mfem::Vector& vector)
{
  const double* const data = vector.HostRead();
  return data == nullptr ? std::vector<double>{} :
      std::vector<double>(data, data + vector.Size());
}

void HashBytes(std::uint64_t& hash, const void* bytes, const std::size_t count)
{
  const auto* const data = static_cast<const unsigned char*>(bytes);
  for (std::size_t index = 0; index < count; ++index)
  {
    hash ^= data[index];
    hash *= UINT64_C(1099511628211);
  }
}

template <typename T>
void HashValue(std::uint64_t& hash, const T& value)
{
  HashBytes(hash, &value, sizeof(value));
}

template <typename T>
void HashVector(std::uint64_t& hash, const std::vector<T>& values)
{
  HashValue(hash, values.size());
  if (!values.empty())
  {
    HashBytes(hash, values.data(), values.size() * sizeof(T));
  }
}

std::uint64_t EvidenceHash(const SmoothEulerMPIEvidence& evidence)
{
  std::uint64_t hash = UINT64_C(1469598103934665603);
  HashValue(hash, evidence.status);
  HashValue(hash, evidence.rank);
  HashValue(hash, evidence.ranks);
  HashValue(hash, evidence.polynomial_order);
  HashValue(hash, evidence.elements_per_direction);
  HashValue(hash, evidence.local_elements);
  HashValue(hash, evidence.global_elements);
  HashValue(hash, evidence.local_trace_dofs);
  HashValue(hash, evidence.local_true_trace_dofs);
  HashValue(hash, evidence.global_true_trace_dofs);
  HashValue(hash, evidence.owned_trace_dofs);
  HashValue(hash, evidence.ghost_trace_dofs);
  HashValue(hash, evidence.global_ghost_trace_dofs);
  HashValue(hash, evidence.shared_faces);
  HashValue(hash, evidence.global_shared_faces);
  HashValue(hash, evidence.negative_signed_trace_dofs);
  HashValue(hash, evidence.global_negative_signed_trace_dofs);
  HashValue(hash, evidence.gmres_iterations);
  HashValue(hash, evidence.gmres_reported_relative_residual);
  HashValue(hash, evidence.true_relative_residual);
  HashValue(hash, evidence.damping);
  HashValue(hash, evidence.initial_nonlinear_residual);
  HashValue(hash, evidence.candidate_nonlinear_residual);
  HashValue(hash, evidence.candidate_minimum_density);
  HashValue(hash, evidence.candidate_minimum_pressure);
  HashValue(hash, evidence.entry_generation);
  HashValue(hash, evidence.committed_generation);
  HashValue(hash, evidence.linearization.local_elements);
  HashValue(hash, evidence.linearization.interior_dofs);
  HashValue(hash, evidence.linearization.trace_dofs);
  HashVector(hash, evidence.linearization.residual_interior);
  HashVector(hash, evidence.linearization.residual_trace);
  HashVector(hash, evidence.linearization.a);
  HashVector(hash, evidence.linearization.b);
  HashVector(hash, evidence.linearization.c);
  HashVector(hash, evidence.linearization.d);
  HashVector(hash, evidence.linearization.signed_trace_dofs);
  HashVector(hash, evidence.condensation.schur);
  HashVector(hash, evidence.condensation.right_hand_side);
  HashVector(hash, evidence.condensation.local_assembled_right_hand_side);
  HashVector(hash, evidence.entry_interior);
  HashVector(hash, evidence.entry_trace);
  HashVector(hash, evidence.final_interior);
  HashVector(hash, evidence.final_trace);
  HashVector(hash, evidence.true_right_hand_side);
  HashVector(hash, evidence.true_trace_increment);
  HashVector(hash, evidence.distributed_trace_increment);
  HashVector(hash, evidence.recovered_interior_increment);
  HashVector(hash, evidence.true_probe);
  HashVector(hash, evidence.true_probe_action);
  HashVector(hash, evidence.true_basis_actions);
  HashValue(hash, evidence.passed);
  return hash;
}

double GlobalNorm(MPI_Comm communicator, const OracleSystem& system,
                  const dg::hdg::trace::DGInterfaceTraceMap& map)
{
  const long long local_global_size = static_cast<long long>(
      map.dofs.global_true_dof_count);
  long long minimum_global_size = 0;
  long long maximum_global_size = 0;
  MPI_Allreduce(&local_global_size, &minimum_global_size, 1,
                MPI_LONG_LONG_INT, MPI_MIN, communicator);
  MPI_Allreduce(&local_global_size, &maximum_global_size, 1,
                MPI_LONG_LONG_INT, MPI_MAX, communicator);
  bool local_ok = local_global_size > 0 &&
      local_global_size <= std::numeric_limits<int>::max() &&
      minimum_global_size == maximum_global_size &&
      !map.dofs.entries.empty() &&
      system.residual_trace.size() == system.signed_trace_dofs.size();
  if (!GlobalAll(communicator, local_ok))
  {
    return std::numeric_limits<double>::infinity();
  }
  const int global_size = static_cast<int>(local_global_size);
  std::vector<double> local_trace;
  std::vector<double> local_global_trace;
  std::vector<double> global_trace;
  try
  {
    local_trace.assign(map.dofs.entries.size(), 0.0);
    local_global_trace.assign(static_cast<std::size_t>(global_size), 0.0);
    global_trace.assign(local_global_trace.size(), 0.0);
    for (std::size_t local = 0; local < system.residual_trace.size(); ++local)
    {
      double sign = 1.0;
      const int dof = Decode(system.signed_trace_dofs[local], sign);
      local_ok = dof >= 0 &&
          static_cast<std::size_t>(dof) < local_trace.size();
      if (!local_ok) { break; }
      local_trace[static_cast<std::size_t>(dof)] +=
          sign * system.residual_trace[local];
    }
    for (const TraceMapEntry& entry : map.dofs.entries)
    {
      local_ok = entry.local_dof >= 0 &&
          static_cast<std::size_t>(entry.local_dof) < local_trace.size() &&
          entry.global_true_dof >= 0 &&
          entry.global_true_dof < global_size;
      if (!local_ok) { break; }
      local_global_trace[static_cast<std::size_t>(entry.global_true_dof)] +=
          local_trace[static_cast<std::size_t>(entry.local_dof)];
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!GlobalAll(communicator, local_ok))
  {
    return std::numeric_limits<double>::infinity();
  }
  MPI_Allreduce(local_global_trace.data(), global_trace.data(), global_size,
                MPI_DOUBLE, MPI_SUM, communicator);
  double local = 0.0;
  for (const double value : system.residual_interior) local += value * value;
  for (const TraceMapEntry& entry : map.dofs.entries)
    if (entry.ownership == Ownership::owned)
    {
      const double value =
          global_trace[static_cast<std::size_t>(entry.global_true_dof)];
      local += value * value;
    }
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, communicator);
  return std::sqrt(global);
}

constexpr long long coordinate_scale = 1000000000000LL;

long long CoordinateBin(const double coordinate)
{
  double periodic = std::fmod(coordinate, 1.0);
  if (periodic < 0.0) { periodic += 1.0; }
  if (std::abs(periodic) <= 1.0e-11 ||
      std::abs(periodic - 1.0) <= 1.0e-11)
  {
    periodic = 0.0;
  }
  return std::llround(periodic * static_cast<double>(coordinate_scale));
}

struct CoordinateKey
{
  long long x = -1;
  long long y = -1;
  int component = -1;

  bool operator<(const CoordinateKey& other) const noexcept
  {
    return std::tie(x, y, component) <
        std::tie(other.x, other.y, other.component);
  }

  bool operator==(const CoordinateKey& other) const noexcept
  {
    return x == other.x && y == other.y && component == other.component;
  }
};

struct ElementDofKey
{
  long long element_x = -1;
  long long element_y = -1;
  CoordinateKey dof{};

  bool operator<(const ElementDofKey& other) const noexcept
  {
    return std::tie(element_x, element_y, dof.x, dof.y, dof.component) <
        std::tie(other.element_x, other.element_y, other.dof.x, other.dof.y,
                 other.dof.component);
  }
};

bool ElementCoordinate(mfem::Mesh& mesh, const int element,
                       long long& x, long long& y)
{
  mfem::ElementTransformation* const transformation =
      mesh.GetElementTransformation(element);
  if (transformation == nullptr) { return false; }
  mfem::IntegrationPoint center;
  center.x = 0.5;
  center.y = 0.5;
  mfem::Vector physical(dimension);
  transformation->Transform(center, physical);
  x = CoordinateBin(physical[0]);
  y = CoordinateBin(physical[1]);
  return true;
}

bool BuildElementDofKeys(
    mfem::Mesh& mesh, mfem::FiniteElementSpace& interior_space,
    mfem::FiniteElementSpace& trace_space,
    std::vector<std::vector<ElementDofKey>>& interior_keys,
    std::vector<std::vector<ElementDofKey>>& trace_keys)
{
  interior_keys.clear();
  trace_keys.clear();
  try
  {
    interior_keys.resize(static_cast<std::size_t>(mesh.GetNE()));
    trace_keys.resize(static_cast<std::size_t>(mesh.GetNE()));
    for (int element = 0; element < mesh.GetNE(); ++element)
    {
      long long element_x = -1;
      long long element_y = -1;
      if (!ElementCoordinate(mesh, element, element_x, element_y))
      {
        return false;
      }
      const mfem::FiniteElement* const interior_element =
          interior_space.GetFE(element);
      mfem::ElementTransformation* const element_transformation =
          mesh.GetElementTransformation(element);
      if (interior_element == nullptr || element_transformation == nullptr)
      {
        return false;
      }
      const int interior_scalar_dofs = interior_element->GetDof();
      auto& element_interior =
          interior_keys[static_cast<std::size_t>(element)];
      element_interior.resize(
          static_cast<std::size_t>(components) * interior_scalar_dofs);
      const mfem::IntegrationRule& interior_nodes =
          interior_element->GetNodes();
      for (int scalar = 0; scalar < interior_scalar_dofs; ++scalar)
      {
        mfem::Vector physical(dimension);
        element_transformation->Transform(interior_nodes.IntPoint(scalar),
                                          physical);
        for (int component = 0; component < components; ++component)
        {
          element_interior[static_cast<std::size_t>(
              Index(scalar, interior_scalar_dofs, component))] =
              {element_x, element_y,
               {CoordinateBin(physical[0]), CoordinateBin(physical[1]),
                component}};
        }
      }

      mfem::Array<int> faces, orientations;
      mesh.GetElementEdges(element, faces, orientations);
      auto& element_trace = trace_keys[static_cast<std::size_t>(element)];
      for (int local_face = 0; local_face < faces.Size(); ++local_face)
      {
        const int face = faces[local_face];
        const mfem::FiniteElement* const trace_element =
            trace_space.GetFaceElement(face);
        mfem::ElementTransformation* const face_transformation =
            mesh.GetFaceTransformation(face);
        if (trace_element == nullptr || face_transformation == nullptr)
        {
          return false;
        }
        const int trace_scalar_dofs = trace_element->GetDof();
        const mfem::IntegrationRule& trace_nodes = trace_element->GetNodes();
        const std::size_t face_offset = element_trace.size();
        element_trace.resize(face_offset +
            static_cast<std::size_t>(components) * trace_scalar_dofs);
        for (int scalar = 0; scalar < trace_scalar_dofs; ++scalar)
        {
          mfem::Vector physical(dimension);
          face_transformation->Transform(trace_nodes.IntPoint(scalar),
                                         physical);
          for (int component = 0; component < components; ++component)
          {
            element_trace[face_offset + static_cast<std::size_t>(
                Index(scalar, trace_scalar_dofs, component))] =
                {element_x, element_y,
                 {CoordinateBin(physical[0]), CoordinateBin(physical[1]),
                  component}};
          }
        }
      }
    }
  }
  catch (const std::exception&)
  {
    return false;
  }
  return true;
}

bool BuildGridDofKeys(mfem::Mesh& mesh, mfem::FiniteElementSpace& space,
                      const bool trace_space,
                      std::vector<CoordinateKey>& keys)
{
  keys.assign(static_cast<std::size_t>(space.GetVSize()), {});
  std::vector<int> census(keys.size(), 0);
  try
  {
    if (trace_space)
    {
      for (int face = 0; face < mesh.GetNumFaces(); ++face)
      {
        const mfem::FiniteElement* const finite_element =
            space.GetFaceElement(face);
        mfem::ElementTransformation* const transformation =
            mesh.GetFaceTransformation(face);
        if (finite_element == nullptr || transformation == nullptr)
        {
          return false;
        }
        const int scalar_dofs = finite_element->GetDof();
        const mfem::IntegrationRule& nodes = finite_element->GetNodes();
        mfem::Array<int> vdofs;
        space.GetFaceVDofs(face, vdofs);
        if (vdofs.Size() != components * scalar_dofs) { return false; }
        for (int scalar = 0; scalar < scalar_dofs; ++scalar)
        {
          mfem::Vector physical(dimension);
          transformation->Transform(nodes.IntPoint(scalar), physical);
          for (int component = 0; component < components; ++component)
          {
            double sign = 1.0;
            const int dof = Decode(
                vdofs[Index(scalar, scalar_dofs, component)], sign);
            if (dof < 0 || static_cast<std::size_t>(dof) >= keys.size())
            {
              return false;
            }
            const CoordinateKey key{CoordinateBin(physical[0]),
                                    CoordinateBin(physical[1]), component};
            if (census[static_cast<std::size_t>(dof)] > 0 &&
                !(keys[static_cast<std::size_t>(dof)] == key))
            {
              return false;
            }
            keys[static_cast<std::size_t>(dof)] = key;
            ++census[static_cast<std::size_t>(dof)];
          }
        }
      }
    }
    else
    {
      for (int element = 0; element < mesh.GetNE(); ++element)
      {
        const mfem::FiniteElement* const finite_element =
            space.GetFE(element);
        mfem::ElementTransformation* const transformation =
            mesh.GetElementTransformation(element);
        if (finite_element == nullptr || transformation == nullptr)
        {
          return false;
        }
        const int scalar_dofs = finite_element->GetDof();
        const mfem::IntegrationRule& nodes = finite_element->GetNodes();
        mfem::Array<int> vdofs;
        space.GetElementVDofs(element, vdofs);
        if (vdofs.Size() != components * scalar_dofs) { return false; }
        for (int scalar = 0; scalar < scalar_dofs; ++scalar)
        {
          mfem::Vector physical(dimension);
          transformation->Transform(nodes.IntPoint(scalar), physical);
          for (int component = 0; component < components; ++component)
          {
            double sign = 1.0;
            const int dof = Decode(
                vdofs[Index(scalar, scalar_dofs, component)], sign);
            if (dof < 0 || static_cast<std::size_t>(dof) >= keys.size())
            {
              return false;
            }
            const CoordinateKey key{CoordinateBin(physical[0]),
                                    CoordinateBin(physical[1]), component};
            keys[static_cast<std::size_t>(dof)] = key;
            ++census[static_cast<std::size_t>(dof)];
          }
        }
      }
    }
  }
  catch (const std::exception&)
  {
    return false;
  }
  return std::all_of(census.begin(), census.end(),
                     [](const int count) { return count > 0; });
}

double EntryError(const double actual, const double expected)
{
  if (!Finite(actual) || !Finite(expected))
  {
    return std::numeric_limits<double>::infinity();
  }
  return std::abs(actual - expected) / std::max(1.0, std::abs(expected));
}

State PerturbedConservative(const double x, const double y)
{
  State primitive = ManufacturedPrimitive(x, y);
  const double psi_one = std::sin(2.0 * oracle_pi * x) *
      std::cos(2.0 * oracle_pi * y);
  const double psi_two = std::cos(2.0 * oracle_pi * x) *
      std::sin(2.0 * oracle_pi * y);
  const double psi_three = std::sin(2.0 * oracle_pi * (x + y));
  primitive[0] *= 1.0 + 0.04 * psi_one;
  primitive[1] += 0.03 * psi_two;
  primitive[2] -= 0.03 * psi_one;
  primitive[3] *= 1.0 + 0.04 * psi_three;
  return PrimitiveToConservative(primitive);
}

struct OwnerRefreshResult
{
  bool passed = false;
  bool staging_ok = false;
  bool exact_owner_bytes = false;
  bool exact_ghost_bytes = false;
  bool coordinate_values_ok = false;
  int owner_values = 0;
  int ghost_values = 0;
  double owner_value_error = std::numeric_limits<double>::infinity();
  double ghost_value_error = std::numeric_limits<double>::infinity();
  double coordinate_value_error = std::numeric_limits<double>::infinity();
};

OwnerRefreshResult CheckOwnerRefreshedInitialTrace(
    MPI_Comm communicator, mfem::ParMesh& mesh,
    mfem::ParFiniteElementSpace& trace_space,
    const dg::hdg::trace::DGInterfaceTraceMap& map,
    const SmoothEulerMPIEvidence& evidence)
{
  OwnerRefreshResult result;
  std::unique_ptr<mfem::ParGridFunction> projected;
  std::unique_ptr<mfem::ParGridFunction> redistributed;
  mfem::Vector owned_true;
  std::vector<CoordinateKey> trace_keys;
  bool local_ok = false;
  try
  {
    projected = std::make_unique<mfem::ParGridFunction>(&trace_space);
    redistributed = std::make_unique<mfem::ParGridFunction>(&trace_space);
    local_ok = Assign(*projected, evidence.entry_trace) &&
        BuildGridDofKeys(mesh, trace_space, true, trace_keys) &&
        map.dofs.entries.size() == trace_keys.size();
    if (local_ok)
    {
      owned_true.SetSize(trace_space.GetTrueVSize());
      projected->ParallelProject(owned_true);
      *redistributed = std::numeric_limits<double>::quiet_NaN();
      local_ok = Finite(Values(owned_true));
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  result.staging_ok = local_ok;
  if (!GlobalAll(communicator, local_ok)) { return result; }

  try
  {
    redistributed->Distribute(owned_true);
    local_ok = Finite(Values(*redistributed));
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!GlobalAll(communicator, local_ok)) { return result; }

  bool owner_bytes = true;
  bool ghost_bytes = true;
  double owner_error = 0.0;
  double ghost_error = 0.0;
  double coordinate_error = 0.0;
  try
  {
    const double* const projected_values = projected->HostRead();
    const double* const redistributed_values = redistributed->HostRead();
    const double* const true_values = owned_true.HostRead();
    local_ok = projected_values != nullptr && redistributed_values != nullptr &&
        (true_values != nullptr || owned_true.Size() == 0);
    for (const TraceMapEntry& entry : map.dofs.entries)
    {
      if (!local_ok || entry.local_dof < 0 ||
          static_cast<std::size_t>(entry.local_dof) >= trace_keys.size())
      {
        local_ok = false;
        break;
      }
      const int local_dof = entry.local_dof;
      const double projected_value = projected_values[local_dof];
      const double redistributed_value = redistributed_values[local_dof];
      const bool byte_equal = std::memcmp(&projected_value,
                                          &redistributed_value,
                                          sizeof(double)) == 0;
      const double exchange_error =
          EntryError(redistributed_value, projected_value);
      if (entry.ownership == Ownership::owned)
      {
        ++result.owner_values;
        owner_bytes = owner_bytes && byte_equal &&
            entry.local_true_dof >= 0 &&
            entry.local_true_dof < owned_true.Size();
        if (entry.local_true_dof >= 0 &&
            entry.local_true_dof < owned_true.Size())
        {
          owner_bytes = owner_bytes &&
              std::memcmp(&projected_value,
                          true_values + entry.local_true_dof,
                          sizeof(double)) == 0;
        }
        owner_error = std::max(owner_error, exchange_error);
      }
      else
      {
        ++result.ghost_values;
        ghost_bytes = ghost_bytes && byte_equal;
        ghost_error = std::max(ghost_error, exchange_error);
      }
      const CoordinateKey& key =
          trace_keys[static_cast<std::size_t>(local_dof)];
      const double x = static_cast<double>(key.x) /
          static_cast<double>(coordinate_scale);
      const double y = static_cast<double>(key.y) /
          static_cast<double>(coordinate_scale);
      const State expected = PerturbedConservative(x, y);
      local_ok = key.component >= 0 && key.component < components && local_ok;
      if (key.component >= 0 && key.component < components)
      {
        coordinate_error = std::max(
            coordinate_error,
            EntryError(projected_value, expected[key.component]));
      }
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  result.exact_owner_bytes = owner_bytes;
  result.exact_ghost_bytes = ghost_bytes;
  result.owner_value_error = owner_error;
  result.ghost_value_error = ghost_error;
  result.coordinate_value_error = coordinate_error;
  result.coordinate_values_ok = coordinate_error <= state_tolerance;
  result.passed = local_ok && result.staging_ok && result.exact_owner_bytes &&
      result.exact_ghost_bytes && result.coordinate_values_ok &&
      result.owner_values == trace_space.GetTrueVSize();
  return result;
}

struct SerialParityResult
{
  bool passed = false;
  bool serial_truth_ok = false;
  bool coordinate_keys_ok = false;
  bool assembly_ok = false;
  bool condensation_ok = false;
  bool action_rhs_ok = false;
  bool trace_solution_ok = false;
  bool recovery_ok = false;
  bool state_ok = false;
  bool status_ok = false;
  int matched_elements = 0;
  double residual_error = std::numeric_limits<double>::infinity();
  double jacobian_error = std::numeric_limits<double>::infinity();
  double condensation_error = std::numeric_limits<double>::infinity();
  double matrix_error = std::numeric_limits<double>::infinity();
  double action_error = std::numeric_limits<double>::infinity();
  double right_hand_side_error = std::numeric_limits<double>::infinity();
  double trace_solution_error = std::numeric_limits<double>::infinity();
  double recovery_error = std::numeric_limits<double>::infinity();
  double state_error = std::numeric_limits<double>::infinity();
};

double CoordinateProbe(const CoordinateKey& key)
{
  const double x = static_cast<double>(key.x) /
      static_cast<double>(coordinate_scale);
  const double y = static_cast<double>(key.y) /
      static_cast<double>(coordinate_scale);
  return 0.31 + 0.17 * x + 0.23 * y + 0.07 * key.component +
      0.013 * x * y;
}

SerialParityResult CheckSerialParity(
    MPI_Comm communicator, const int polynomial_order,
    mfem::ParMesh& parallel_mesh,
    mfem::ParFiniteElementSpace& parallel_interior_space,
    mfem::ParFiniteElementSpace& parallel_trace_space,
    const dg::hdg::trace::DGInterfaceTraceMap& parallel_map,
    const SmoothEulerMPIEvidence& mpi_evidence)
{
  SerialParityResult result;
  ConvergenceEvidence serial_ladder;
  std::unique_ptr<SmoothEulerExperiment> serial_truth;
  std::unique_ptr<SmoothEulerExperiment> serial_graph;
  const SolvedMeshEvidence* serial_solved = nullptr;
  const TrialAuditSnapshot* accepted_trial = nullptr;
  mfem::Mesh* serial_mesh = nullptr;
  mfem::FiniteElementSpace* serial_interior_space = nullptr;
  mfem::FiniteElementSpace* serial_trace_space = nullptr;
  bool local_ok = false;
  try
  {
    serial_truth = std::make_unique<SmoothEulerExperiment>(polynomial_order);
    local_ok = serial_truth->RunConvergenceLadder(serial_ladder) &&
        serial_ladder.status == ExperimentStatus::Success &&
        serial_ladder.solves[0].elements_per_direction == 3 &&
        serial_ladder.solves[0].status == ExperimentStatus::Success &&
        serial_ladder.solves[0].has_linear_audit;
    if (local_ok)
    {
      serial_solved = &serial_ladder.solves[0];
      for (const TrialAuditSnapshot& audit : serial_solved->trial_audits)
        if (audit.nonlinear_iteration == 0 && audit.accepted)
        {
          accepted_trial = &audit;
          break;
        }
      local_ok = accepted_trial != nullptr;
    }
    serial_graph = std::make_unique<SmoothEulerExperiment>(polynomial_order);
    local_ok = serial_graph->BuildPeriodicMesh(3) &&
        serial_graph->ProjectOldStage() && local_ok;
    if (local_ok)
    {
      serial_trace_space = const_cast<mfem::FiniteElementSpace*>(
          serial_graph->trace_space());
      serial_interior_space = const_cast<mfem::FiniteElementSpace*>(
          serial_graph->current_interior()->FESpace());
      serial_mesh = serial_trace_space == nullptr ? nullptr :
          serial_trace_space->GetMesh();
      local_ok = serial_mesh != nullptr && serial_interior_space != nullptr &&
          serial_trace_space != nullptr;
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  result.serial_truth_ok = local_ok;
  if (!GlobalAll(communicator, local_ok)) { return result; }

  std::vector<std::vector<ElementDofKey>> parallel_interior_element_keys;
  std::vector<std::vector<ElementDofKey>> parallel_trace_element_keys;
  std::vector<std::vector<ElementDofKey>> serial_interior_element_keys;
  std::vector<std::vector<ElementDofKey>> serial_trace_element_keys;
  std::vector<CoordinateKey> parallel_interior_keys;
  std::vector<CoordinateKey> parallel_trace_keys;
  std::vector<CoordinateKey> serial_interior_keys;
  std::vector<CoordinateKey> serial_trace_keys;
  try
  {
    local_ok = BuildElementDofKeys(
        parallel_mesh, parallel_interior_space, parallel_trace_space,
        parallel_interior_element_keys, parallel_trace_element_keys) &&
        BuildElementDofKeys(
            *serial_mesh, *serial_interior_space, *serial_trace_space,
            serial_interior_element_keys, serial_trace_element_keys) &&
        BuildGridDofKeys(parallel_mesh, parallel_interior_space, false,
                         parallel_interior_keys) &&
        BuildGridDofKeys(parallel_mesh, parallel_trace_space, true,
                         parallel_trace_keys) &&
        BuildGridDofKeys(*serial_mesh, *serial_interior_space, false,
                         serial_interior_keys) &&
        BuildGridDofKeys(*serial_mesh, *serial_trace_space, true,
                         serial_trace_keys);
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  result.coordinate_keys_ok = local_ok;
  if (!GlobalAll(communicator, local_ok)) { return result; }

  std::map<ElementDofKey, std::size_t> serial_interior_slots;
  std::map<ElementDofKey, std::size_t> serial_trace_slots;
  OracleSystem serial_all_elements;
  std::vector<double> serial_schur;
  std::vector<double> serial_reduced_rhs;
  double residual_error = 0.0;
  double jacobian_error = 0.0;
  double condensation_error = 0.0;
  int local_matched_elements = 0;
  try
  {
    const int serial_elements = static_cast<int>(
        serial_interior_element_keys.size());
    const int serial_ni = mpi_evidence.linearization.interior_dofs;
    const int serial_nt = mpi_evidence.linearization.trace_dofs;
    mfem::GridFunction serial_current_interior(serial_interior_space);
    mfem::GridFunction serial_current_trace(serial_trace_space);
    std::vector<double> serial_interior_values(serial_interior_keys.size(),
                                               0.0);
    std::vector<double> serial_trace_values(serial_trace_keys.size(), 0.0);
    for (std::size_t dof = 0; dof < serial_interior_keys.size(); ++dof)
    {
      const CoordinateKey& key = serial_interior_keys[dof];
      const State state = PerturbedConservative(
          static_cast<double>(key.x) / coordinate_scale,
          static_cast<double>(key.y) / coordinate_scale);
      local_ok = key.component >= 0 && key.component < components;
      if (!local_ok) { break; }
      serial_interior_values[dof] = state[key.component];
    }
    for (std::size_t dof = 0;
         dof < serial_trace_keys.size() && local_ok; ++dof)
    {
      const CoordinateKey& key = serial_trace_keys[dof];
      const State state = PerturbedConservative(
          static_cast<double>(key.x) / coordinate_scale,
          static_cast<double>(key.y) / coordinate_scale);
      local_ok = key.component >= 0 && key.component < components;
      if (!local_ok) { break; }
      serial_trace_values[dof] = state[key.component];
    }
    local_ok = local_ok && Assign(serial_current_interior,
                                  serial_interior_values) &&
        Assign(serial_current_trace, serial_trace_values) &&
        AssembleOracle(*serial_mesh, *serial_interior_space,
                       *serial_trace_space, *serial_graph->old_interior(),
                       serial_current_interior, serial_current_trace,
                       polynomial_order, serial_all_elements) &&
        CondenseOracle(serial_all_elements, serial_schur,
                       serial_reduced_rhs);
    local_ok = local_ok && serial_elements == 9 && serial_ni > 0 &&
        serial_nt > 0 && serial_all_elements.elements == serial_elements &&
        serial_all_elements.ni == serial_ni &&
        serial_all_elements.nt == serial_nt;
    for (int element = 0; element < serial_elements && local_ok; ++element)
    {
      local_ok = serial_interior_element_keys[
                     static_cast<std::size_t>(element)].size() ==
              static_cast<std::size_t>(serial_ni) &&
          serial_trace_element_keys[
              static_cast<std::size_t>(element)].size() ==
              static_cast<std::size_t>(serial_nt);
      for (int local = 0; local < serial_ni && local_ok; ++local)
      {
        local_ok = serial_interior_slots.emplace(
            serial_interior_element_keys[static_cast<std::size_t>(element)]
                                        [static_cast<std::size_t>(local)],
            static_cast<std::size_t>(element) * serial_ni + local).second;
      }
      for (int local = 0; local < serial_nt && local_ok; ++local)
      {
        local_ok = serial_trace_slots.emplace(
            serial_trace_element_keys[static_cast<std::size_t>(element)]
                                     [static_cast<std::size_t>(local)],
            static_cast<std::size_t>(element) * serial_nt + local).second;
      }
    }

    const int local_elements = mpi_evidence.linearization.local_elements;
    local_ok = local_ok && local_elements == parallel_mesh.GetNE() &&
        mpi_evidence.linearization.residual_interior.size() ==
            static_cast<std::size_t>(local_elements) * serial_ni &&
        mpi_evidence.linearization.residual_trace.size() ==
            static_cast<std::size_t>(local_elements) * serial_nt;
    for (int element = 0; element < local_elements && local_ok; ++element)
    {
      const auto& interior =
          parallel_interior_element_keys[static_cast<std::size_t>(element)];
      const auto& trace =
          parallel_trace_element_keys[static_cast<std::size_t>(element)];
      local_ok = interior.size() == static_cast<std::size_t>(serial_ni) &&
          trace.size() == static_cast<std::size_t>(serial_nt);
      std::vector<int> serial_interior_local(
          static_cast<std::size_t>(serial_ni), -1);
      std::vector<int> serial_trace_local(
          static_cast<std::size_t>(serial_nt), -1);
      int serial_element = -1;
      for (int row = 0; row < serial_ni && local_ok; ++row)
      {
        const auto found = serial_interior_slots.find(
            interior[static_cast<std::size_t>(row)]);
        local_ok = found != serial_interior_slots.end();
        if (local_ok)
        {
          const int found_element =
              static_cast<int>(found->second) / serial_ni;
          serial_element = serial_element < 0 ? found_element : serial_element;
          local_ok = found_element == serial_element;
          serial_interior_local[static_cast<std::size_t>(row)] =
              static_cast<int>(found->second) % serial_ni;
          residual_error = std::max(
              residual_error,
              EntryError(
                  mpi_evidence.linearization.residual_interior[
                      static_cast<std::size_t>(element) * serial_ni + row],
                  serial_all_elements.residual_interior[found->second]));
        }
      }
      for (int row = 0; row < serial_nt && local_ok; ++row)
      {
        const auto found = serial_trace_slots.find(
            trace[static_cast<std::size_t>(row)]);
        local_ok = found != serial_trace_slots.end();
        if (local_ok)
        {
          const int found_element =
              static_cast<int>(found->second) / serial_nt;
          serial_element = serial_element < 0 ? found_element : serial_element;
          local_ok = found_element == serial_element;
          serial_trace_local[static_cast<std::size_t>(row)] =
              static_cast<int>(found->second) % serial_nt;
          residual_error = std::max(
              residual_error,
              EntryError(
                  mpi_evidence.linearization.residual_trace[
                      static_cast<std::size_t>(element) * serial_nt + row],
                  serial_all_elements.residual_trace[found->second]));
        }
      }
      if (local_ok)
      {
        ++local_matched_elements;
        for (int row = 0; row < serial_ni && local_ok; ++row)
          for (int column = 0; column < serial_ni; ++column)
          {
            const int serial_row =
                serial_interior_local[static_cast<std::size_t>(row)];
            const int serial_column =
                serial_interior_local[static_cast<std::size_t>(column)];
            jacobian_error = std::max(
                jacobian_error,
                EntryError(
                    mpi_evidence.linearization.a[
                        static_cast<std::size_t>(element) * serial_ni *
                            serial_ni +
                        static_cast<std::size_t>(row) * serial_ni + column],
                    serial_all_elements.a[
                        static_cast<std::size_t>(serial_element) * serial_ni *
                            serial_ni +
                        static_cast<std::size_t>(serial_row) * serial_ni +
                        serial_column]));
          }
        for (int row = 0; row < serial_ni && local_ok; ++row)
          for (int column = 0; column < serial_nt; ++column)
          {
            const int serial_row =
                serial_interior_local[static_cast<std::size_t>(row)];
            const int serial_column =
                serial_trace_local[static_cast<std::size_t>(column)];
            jacobian_error = std::max(
                jacobian_error,
                EntryError(
                    mpi_evidence.linearization.b[
                        static_cast<std::size_t>(element) * serial_ni *
                            serial_nt +
                        static_cast<std::size_t>(row) * serial_nt + column],
                    serial_all_elements.b[
                        static_cast<std::size_t>(serial_element) * serial_ni *
                            serial_nt +
                        static_cast<std::size_t>(serial_row) * serial_nt +
                        serial_column]));
          }
        for (int row = 0; row < serial_nt && local_ok; ++row)
          for (int column = 0; column < serial_ni; ++column)
          {
            const int serial_row =
                serial_trace_local[static_cast<std::size_t>(row)];
            const int serial_column =
                serial_interior_local[static_cast<std::size_t>(column)];
            jacobian_error = std::max(
                jacobian_error,
                EntryError(
                    mpi_evidence.linearization.c[
                        static_cast<std::size_t>(element) * serial_nt *
                            serial_ni +
                        static_cast<std::size_t>(row) * serial_ni + column],
                    serial_all_elements.c[
                        static_cast<std::size_t>(serial_element) * serial_nt *
                            serial_ni +
                        static_cast<std::size_t>(serial_row) * serial_ni +
                        serial_column]));
          }
        for (int row = 0; row < serial_nt && local_ok; ++row)
          for (int column = 0; column < serial_nt; ++column)
          {
            const int serial_row =
                serial_trace_local[static_cast<std::size_t>(row)];
            const int serial_column =
                serial_trace_local[static_cast<std::size_t>(column)];
            jacobian_error = std::max(
                jacobian_error,
                EntryError(
                    mpi_evidence.linearization.d[
                        static_cast<std::size_t>(element) * serial_nt *
                            serial_nt +
                        static_cast<std::size_t>(row) * serial_nt + column],
                    serial_all_elements.d[
                        static_cast<std::size_t>(serial_element) * serial_nt *
                            serial_nt +
                        static_cast<std::size_t>(serial_row) * serial_nt +
                        serial_column]));
          }
        for (int row = 0; row < serial_nt && local_ok; ++row)
        {
          const int serial_row =
              serial_trace_local[static_cast<std::size_t>(row)];
          condensation_error = std::max(
              condensation_error,
              EntryError(
                  mpi_evidence.condensation.right_hand_side[
                      static_cast<std::size_t>(element) * serial_nt + row],
                  serial_reduced_rhs[
                      static_cast<std::size_t>(serial_element) * serial_nt +
                      serial_row]));
          for (int column = 0; column < serial_nt; ++column)
          {
            const int serial_column =
                serial_trace_local[static_cast<std::size_t>(column)];
            condensation_error = std::max(
                condensation_error,
                EntryError(
                    mpi_evidence.condensation.schur[
                        static_cast<std::size_t>(element) * serial_nt *
                            serial_nt +
                        static_cast<std::size_t>(row) * serial_nt + column],
                    serial_schur[
                        static_cast<std::size_t>(serial_element) * serial_nt *
                            serial_nt +
                        static_cast<std::size_t>(serial_row) * serial_nt +
                        serial_column]));
          }
        }
      }
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  int global_matched_elements = 0;
  MPI_Allreduce(&local_matched_elements, &global_matched_elements,
                1, MPI_INT, MPI_SUM, communicator);
  result.matched_elements = global_matched_elements;
  result.residual_error = residual_error;
  result.jacobian_error = jacobian_error;
  result.condensation_error = condensation_error;
  result.assembly_ok = local_ok && residual_error <= local_tolerance &&
      jacobian_error <= local_tolerance && global_matched_elements == 9;
  result.condensation_ok = local_ok &&
      condensation_error <= local_tolerance &&
      global_matched_elements == 9;
  if (!GlobalAll(communicator, local_ok)) { return result; }

  const int global_trace_size = static_cast<int>(
      parallel_map.dofs.global_true_dof_count);
  std::vector<long long> local_global_x;
  std::vector<long long> local_global_y;
  std::vector<int> local_global_component;
  std::vector<long long> global_x;
  std::vector<long long> global_y;
  std::vector<int> global_component;
  try
  {
    local_ok = global_trace_size > 0 &&
        serial_solved->linear_audit.rows == global_trace_size &&
        mpi_evidence.local_true_trace_dofs ==
            parallel_trace_space.GetTrueVSize();
    local_global_x.assign(static_cast<std::size_t>(global_trace_size), -1);
    local_global_y.assign(static_cast<std::size_t>(global_trace_size), -1);
    local_global_component.assign(
        static_cast<std::size_t>(global_trace_size), -1);
    global_x.resize(local_global_x.size());
    global_y.resize(local_global_y.size());
    global_component.resize(local_global_component.size());
    for (const TraceMapEntry& entry : parallel_map.dofs.entries)
      if (entry.ownership == Ownership::owned)
      {
        local_ok = entry.local_dof >= 0 &&
            static_cast<std::size_t>(entry.local_dof) <
                parallel_trace_keys.size() && entry.global_true_dof >= 0 &&
            entry.global_true_dof < global_trace_size && local_ok;
        if (local_ok)
        {
          const CoordinateKey& key = parallel_trace_keys[
              static_cast<std::size_t>(entry.local_dof)];
          local_global_x[static_cast<std::size_t>(entry.global_true_dof)] =
              key.x;
          local_global_y[static_cast<std::size_t>(entry.global_true_dof)] =
              key.y;
          local_global_component[
              static_cast<std::size_t>(entry.global_true_dof)] = key.component;
        }
      }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  if (!GlobalAll(communicator, local_ok)) { return result; }
  MPI_Allreduce(local_global_x.data(), global_x.data(), global_trace_size,
                MPI_LONG_LONG_INT, MPI_MAX, communicator);
  MPI_Allreduce(local_global_y.data(), global_y.data(), global_trace_size,
                MPI_LONG_LONG_INT, MPI_MAX, communicator);
  MPI_Allreduce(local_global_component.data(), global_component.data(),
                global_trace_size, MPI_INT, MPI_MAX, communicator);

  std::map<CoordinateKey, int> serial_trace_lookup;
  std::map<CoordinateKey, int> serial_interior_lookup;
  std::vector<int> serial_column_for_global(
      static_cast<std::size_t>(global_trace_size), -1);
  std::vector<double> serial_dense;
  double matrix_error = 0.0;
  double action_error = 0.0;
  double rhs_error = 0.0;
  double solution_error = 0.0;
  double recovery_error = 0.0;
  double state_error = 0.0;
  try
  {
    for (int dof = 0; dof < static_cast<int>(serial_trace_keys.size()); ++dof)
    {
      local_ok = serial_trace_lookup.emplace(
          serial_trace_keys[static_cast<std::size_t>(dof)], dof).second &&
          local_ok;
    }
    for (int dof = 0;
         dof < static_cast<int>(serial_interior_keys.size()); ++dof)
    {
      local_ok = serial_interior_lookup.emplace(
          serial_interior_keys[static_cast<std::size_t>(dof)], dof).second &&
          local_ok;
    }
    for (int global = 0; global < global_trace_size; ++global)
    {
      const CoordinateKey key{
          global_x[static_cast<std::size_t>(global)],
          global_y[static_cast<std::size_t>(global)],
          global_component[static_cast<std::size_t>(global)]};
      const auto found = serial_trace_lookup.find(key);
      local_ok = found != serial_trace_lookup.end() && local_ok;
      if (found != serial_trace_lookup.end())
      {
        serial_column_for_global[static_cast<std::size_t>(global)] =
            found->second;
      }
    }

    const auto& linear = serial_solved->linear_audit;
    local_ok = linear.row_offsets.size() ==
            static_cast<std::size_t>(linear.rows + 1) &&
        linear.right_hand_side.size() == static_cast<std::size_t>(linear.rows) &&
        linear.trace_increment.size() == static_cast<std::size_t>(linear.rows) &&
        local_ok;
    serial_dense.assign(
        static_cast<std::size_t>(linear.rows) * linear.rows, 0.0);
    for (int row = 0; row < linear.rows && local_ok; ++row)
    {
      local_ok = linear.row_offsets[static_cast<std::size_t>(row)] >= 0 &&
          linear.row_offsets[static_cast<std::size_t>(row)] <=
              linear.row_offsets[static_cast<std::size_t>(row + 1)] &&
          linear.row_offsets[static_cast<std::size_t>(row + 1)] <=
              static_cast<int>(linear.values.size()) &&
          linear.values.size() == linear.columns.size();
      for (int entry = linear.row_offsets[static_cast<std::size_t>(row)];
           entry < linear.row_offsets[static_cast<std::size_t>(row + 1)] &&
           local_ok; ++entry)
      {
        const int column = linear.columns[static_cast<std::size_t>(entry)];
        local_ok = column >= 0 && column < linear.rows;
        if (local_ok)
        {
          serial_dense[static_cast<std::size_t>(row) * linear.rows + column] =
              linear.values[static_cast<std::size_t>(entry)];
        }
      }
    }

    for (const TraceMapEntry& entry : parallel_map.dofs.entries)
      if (entry.ownership == Ownership::owned && local_ok)
      {
        const int row_global = static_cast<int>(entry.global_true_dof);
        const int serial_row =
            serial_column_for_global[static_cast<std::size_t>(row_global)];
        local_ok = entry.local_true_dof >= 0 &&
            entry.local_true_dof < parallel_trace_space.GetTrueVSize() &&
            serial_row >= 0 && serial_row < linear.rows;
        if (!local_ok) { break; }
        rhs_error = std::max(
            rhs_error,
            EntryError(
                mpi_evidence.true_right_hand_side[
                    static_cast<std::size_t>(entry.local_true_dof)],
                linear.right_hand_side[static_cast<std::size_t>(serial_row)]));
        solution_error = std::max(
            solution_error,
            EntryError(
                mpi_evidence.true_trace_increment[
                    static_cast<std::size_t>(entry.local_true_dof)],
                linear.trace_increment[static_cast<std::size_t>(serial_row)]));
        double mpi_action = 0.0;
        double serial_action = 0.0;
        for (int column_global = 0; column_global < global_trace_size;
             ++column_global)
        {
          const int serial_column = serial_column_for_global[
              static_cast<std::size_t>(column_global)];
          const CoordinateKey column_key{
              global_x[static_cast<std::size_t>(column_global)],
              global_y[static_cast<std::size_t>(column_global)],
              global_component[static_cast<std::size_t>(column_global)]};
          const double probe = CoordinateProbe(column_key);
          const double mpi_entry = mpi_evidence.true_basis_actions[
              static_cast<std::size_t>(column_global) *
                  parallel_trace_space.GetTrueVSize() +
              entry.local_true_dof];
          const double serial_entry = serial_dense[
              static_cast<std::size_t>(serial_row) * linear.rows +
              serial_column];
          matrix_error = std::max(
              matrix_error, EntryError(mpi_entry, serial_entry));
          mpi_action += mpi_entry * probe;
          serial_action += serial_entry * probe;
        }
        action_error = std::max(
            action_error, EntryError(mpi_action, serial_action));
      }

    std::vector<double> parallel_recovery_grid(
        static_cast<std::size_t>(parallel_interior_space.GetVSize()), 0.0);
    std::vector<int> recovery_census(parallel_recovery_grid.size(), 0);
    for (int element = 0; element < parallel_mesh.GetNE() && local_ok;
         ++element)
    {
      mfem::Array<int> vdofs;
      parallel_interior_space.GetElementVDofs(element, vdofs);
      local_ok = vdofs.Size() == mpi_evidence.linearization.interior_dofs;
      for (int local = 0; local < vdofs.Size() && local_ok; ++local)
      {
        double sign = 1.0;
        const int dof = Decode(vdofs[local], sign);
        local_ok = dof >= 0 &&
            static_cast<std::size_t>(dof) < parallel_recovery_grid.size();
        if (local_ok)
        {
          parallel_recovery_grid[static_cast<std::size_t>(dof)] = sign *
              mpi_evidence.recovered_interior_increment[
                  static_cast<std::size_t>(element) * vdofs.Size() + local];
          ++recovery_census[static_cast<std::size_t>(dof)];
        }
      }
    }
    local_ok = std::all_of(
        recovery_census.begin(), recovery_census.end(),
        [](const int count) { return count == 1; }) && local_ok;
    for (int local_dof = 0;
         local_dof < parallel_interior_space.GetVSize() && local_ok;
         ++local_dof)
    {
      const auto found = serial_interior_lookup.find(
          parallel_interior_keys[static_cast<std::size_t>(local_dof)]);
      local_ok = found != serial_interior_lookup.end();
      if (!local_ok) { break; }
      const std::size_t serial_dof = static_cast<std::size_t>(found->second);
      recovery_error = std::max(
          recovery_error,
          EntryError(parallel_recovery_grid[static_cast<std::size_t>(local_dof)],
                     accepted_trial->delta_interior[serial_dof]));
      state_error = std::max(
          state_error,
          EntryError(mpi_evidence.entry_interior[
                         static_cast<std::size_t>(local_dof)],
                     accepted_trial->base_interior[serial_dof]));
      state_error = std::max(
          state_error,
          EntryError(mpi_evidence.final_interior[
                         static_cast<std::size_t>(local_dof)],
                     accepted_trial->trial_interior[serial_dof]));
    }
    for (int local_dof = 0;
         local_dof < parallel_trace_space.GetVSize() && local_ok;
         ++local_dof)
    {
      const auto found = serial_trace_lookup.find(
          parallel_trace_keys[static_cast<std::size_t>(local_dof)]);
      local_ok = found != serial_trace_lookup.end();
      if (!local_ok) { break; }
      const std::size_t serial_dof = static_cast<std::size_t>(found->second);
      recovery_error = std::max(
          recovery_error,
          EntryError(mpi_evidence.distributed_trace_increment[
                         static_cast<std::size_t>(local_dof)],
                     accepted_trial->delta_trace[serial_dof]));
      state_error = std::max(
          state_error,
          EntryError(mpi_evidence.entry_trace[
                         static_cast<std::size_t>(local_dof)],
                     accepted_trial->base_trace[serial_dof]));
      state_error = std::max(
          state_error,
          EntryError(mpi_evidence.final_trace[
                         static_cast<std::size_t>(local_dof)],
                     accepted_trial->trial_trace[serial_dof]));
    }
  }
  catch (const std::exception&)
  {
    local_ok = false;
  }
  result.matrix_error = matrix_error;
  result.action_error = action_error;
  result.right_hand_side_error = rhs_error;
  result.trace_solution_error = solution_error;
  result.recovery_error = recovery_error;
  result.state_error = state_error;
  result.action_rhs_ok = local_ok && matrix_error <= local_tolerance &&
      action_error <= local_tolerance && rhs_error <= local_tolerance;
  result.trace_solution_ok = local_ok &&
      solution_error <= state_tolerance;
  result.recovery_ok = local_ok && recovery_error <= state_tolerance;
  result.state_ok = local_ok && state_error <= state_tolerance &&
      EntryError(mpi_evidence.damping, accepted_trial->alpha) == 0.0;
  result.status_ok = local_ok &&
      serial_solved->status == ExperimentStatus::Success &&
      serial_solved->newton.status == ExperimentStatus::Success &&
      accepted_trial->accepted && accepted_trial->generation_published &&
      mpi_evidence.status == SmoothEulerMPIStatus::Success &&
      mpi_evidence.passed;
  result.passed = result.serial_truth_ok && result.coordinate_keys_ok &&
      result.assembly_ok && result.condensation_ok && result.action_rhs_ok &&
      result.trace_solution_ok && result.recovery_ok && result.state_ok &&
      result.status_ok;
  return result;
}

struct OrientationOracleResult
{
  bool passed = false;
  bool finite_element_permutation_ok = false;
  bool actual_shared_reversal_ok = false;
  bool wrong_permutation_rejected = false;
  int shared_faces = 0;
  int reversed_dofs = 0;
  int actual_reversed_face_pairs = 0;
  double value_error = std::numeric_limits<double>::infinity();
  double action_error = std::numeric_limits<double>::infinity();
  double wrong_value_error = 0.0;
  double wrong_action_error = 0.0;
};

OrientationOracleResult CheckOrientationOracle(
    MPI_Comm communicator, mfem::ParMesh& mesh,
    mfem::ParFiniteElementSpace& trace_space,
    const dg::hdg::trace::DGInterfaceTraceMap& map,
    const int rank, const int polynomial_order, const int rank_count)
{
  OrientationOracleResult result;
  const mfem::FiniteElementCollection* const collection = trace_space.FEColl();
  const mfem::FiniteElement* const finite_element =
      collection == nullptr ? nullptr :
          collection->FiniteElementForGeometry(mfem::Geometry::SEGMENT);
  const int scalar_dofs = polynomial_order + 1;
  const int* const permutation = collection == nullptr ? nullptr :
      collection->DofOrderForOrientation(mfem::Geometry::SEGMENT, -1);
  bool valid = finite_element != nullptr && permutation != nullptr &&
      (finite_element == nullptr || finite_element->GetDof() == scalar_dofs);
  for (int destination = 0; destination < scalar_dofs && valid; ++destination)
  {
    valid = permutation[destination] == scalar_dofs - 1 - destination &&
        permutation[destination] >= 0 && valid;
    result.reversed_dofs += permutation[destination] != destination ? 1 : 0;
  }
  result.finite_element_permutation_ok =
      valid && result.reversed_dofs > 0;
  result.shared_faces = mesh.GetNSharedFaces();
  std::vector<CoordinateKey> trace_keys;
  std::vector<long long> local_records;
  const int record_width = 3 + scalar_dofs;
  try
  {
    valid = BuildGridDofKeys(mesh, trace_space, true, trace_keys) &&
        map.dofs.entries.size() == trace_keys.size() && valid;
    for (int shared = 0; shared < result.shared_faces && valid; ++shared)
    {
      const int face = mesh.GetSharedFace(shared);
      const mfem::FiniteElement* const face_element =
          trace_space.GetFaceElement(face);
      mfem::ElementTransformation* const transformation =
          mesh.GetFaceTransformation(face);
      mfem::Array<int> vdofs;
      trace_space.GetFaceVDofs(face, vdofs);
      valid = face_element != nullptr && transformation != nullptr &&
          face_element->GetDof() == scalar_dofs &&
          vdofs.Size() == components * scalar_dofs && valid;
      if (!valid) { break; }
      mfem::IntegrationPoint center;
      center.x = 0.5;
      mfem::Vector physical(dimension);
      transformation->Transform(center, physical);
      local_records.push_back(rank);
      local_records.push_back(CoordinateBin(physical[0]));
      local_records.push_back(CoordinateBin(physical[1]));
      for (int scalar = 0; scalar < scalar_dofs; ++scalar)
      {
        double sign = 1.0;
        const int local_dof = Decode(
            vdofs[Index(scalar, scalar_dofs, 0)], sign);
        valid = local_dof >= 0 &&
            static_cast<std::size_t>(local_dof) < map.dofs.entries.size() &&
            valid;
        if (!valid) { break; }
        local_records.push_back(map.dofs.entries[
            static_cast<std::size_t>(local_dof)].global_true_dof);
      }
    }
    valid = valid && local_records.size() ==
        static_cast<std::size_t>(result.shared_faces * record_width);
  }
  catch (const std::exception&)
  {
    valid = false;
  }
  if (!GlobalAll(communicator, valid)) { return result; }

  const int local_record_count = result.shared_faces;
  std::vector<int> record_counts;
  try
  {
    record_counts.assign(static_cast<std::size_t>(rank_count), 0);
  }
  catch (const std::exception&)
  {
    valid = false;
  }
  if (!GlobalAll(communicator, valid)) { return result; }
  MPI_Allgather(&local_record_count, 1, MPI_INT, record_counts.data(), 1,
                MPI_INT, communicator);
  std::vector<int> value_counts;
  std::vector<int> displacements;
  std::vector<long long> records;
  int total_records = 0;
  try
  {
    value_counts.assign(record_counts.size(), 0);
    displacements.assign(record_counts.size(), 0);
    for (int process = 0; process < rank_count; ++process)
    {
      displacements[static_cast<std::size_t>(process)] =
          total_records * record_width;
      value_counts[static_cast<std::size_t>(process)] =
          record_counts[static_cast<std::size_t>(process)] * record_width;
      total_records += record_counts[static_cast<std::size_t>(process)];
    }
    records.assign(static_cast<std::size_t>(total_records) * record_width,
                   -1);
  }
  catch (const std::exception&)
  {
    valid = false;
  }
  if (!GlobalAll(communicator, valid)) { return result; }
  MPI_Allgatherv(local_records.data(),
                 static_cast<int>(local_records.size()), MPI_LONG_LONG_INT,
                 records.data(), value_counts.data(), displacements.data(),
                 MPI_LONG_LONG_INT, communicator);

  const long long local_global_size = static_cast<long long>(
      map.dofs.global_true_dof_count);
  long long minimum_global_size = 0;
  long long maximum_global_size = 0;
  MPI_Allreduce(&local_global_size, &minimum_global_size, 1,
                MPI_LONG_LONG_INT, MPI_MIN, communicator);
  MPI_Allreduce(&local_global_size, &maximum_global_size, 1,
                MPI_LONG_LONG_INT, MPI_MAX, communicator);
  valid = local_global_size > 0 &&
      local_global_size <= std::numeric_limits<int>::max() &&
      minimum_global_size == maximum_global_size && valid;
  if (!GlobalAll(communicator, valid)) { return result; }
  const int global_size = static_cast<int>(local_global_size);

  result.value_error = 0.0;
  result.action_error = 0.0;
  for (int first = 0; first < total_records; ++first)
    for (int second = first + 1; second < total_records; ++second)
    {
      const long long* const a = records.data() +
          static_cast<std::size_t>(first) * record_width;
      const long long* const b = records.data() +
          static_cast<std::size_t>(second) * record_width;
      if (a[0] == b[0] || a[1] != b[1] || a[2] != b[2]) { continue; }
      bool reversed = true;
      bool identical = true;
      for (int scalar = 0; scalar < scalar_dofs; ++scalar)
      {
        reversed = a[3 + scalar] ==
            b[3 + scalar_dofs - 1 - scalar] && reversed;
        identical = a[3 + scalar] == b[3 + scalar] && identical;
      }
      if (!reversed || identical) { continue; }
      ++result.actual_reversed_face_pairs;
      std::vector<int> a_for_peer(static_cast<std::size_t>(scalar_dofs), -1);
      std::vector<int> peer_for_a(static_cast<std::size_t>(scalar_dofs), -1);
      for (int a_index = 0; a_index < scalar_dofs; ++a_index)
      {
        valid = a[3 + a_index] >= 0 &&
            a[3 + a_index] < global_size && valid;
        if (!valid) { break; }
        const auto peer = std::find(b + 3, b + 3 + scalar_dofs,
                                    a[3 + a_index]);
        valid = peer != b + 3 + scalar_dofs && valid;
        if (!valid) { break; }
        const int peer_index = static_cast<int>(peer - (b + 3));
        peer_for_a[static_cast<std::size_t>(a_index)] = peer_index;
        a_for_peer[static_cast<std::size_t>(peer_index)] = a_index;
      }
      std::vector<double> a_value(static_cast<std::size_t>(scalar_dofs), 0.0);
      std::vector<double> peer_value(a_value.size(), 0.0);
      std::vector<double> a_action(a_value.size(), 0.0);
      std::vector<double> peer_action(a_value.size(), 0.0);
      std::vector<double> wrong_action(a_value.size(), 0.0);
      for (int index = 0; index < scalar_dofs && valid; ++index)
      {
        a_value[static_cast<std::size_t>(index)] =
            0.27 + 0.09 * index + 0.017 * index * index;
      }
      for (int peer_index = 0; peer_index < scalar_dofs && valid;
           ++peer_index)
      {
        const int a_index =
            a_for_peer[static_cast<std::size_t>(peer_index)];
        valid = a_index >= 0 && a_index < scalar_dofs;
        if (valid)
        {
          peer_value[static_cast<std::size_t>(peer_index)] =
              a_value[static_cast<std::size_t>(a_index)];
        }
      }
      for (int row = 0; row < scalar_dofs && valid; ++row)
        for (int column = 0; column < scalar_dofs; ++column)
        {
          const double entry = 0.41 + 0.13 * (row + 1) +
              0.019 * (column + 1) * (column + 1) +
              0.007 * (row + 1) * (column + 1);
          a_action[static_cast<std::size_t>(row)] +=
              entry * a_value[static_cast<std::size_t>(column)];
        }
      for (int peer_row = 0; peer_row < scalar_dofs && valid; ++peer_row)
        for (int peer_column = 0; peer_column < scalar_dofs; ++peer_column)
        {
          const int a_row =
              a_for_peer[static_cast<std::size_t>(peer_row)];
          const int a_column =
              a_for_peer[static_cast<std::size_t>(peer_column)];
          valid = a_row >= 0 && a_column >= 0;
          if (!valid) { break; }
          const double transformed_entry =
              0.41 + 0.13 * (a_row + 1) +
              0.019 * (a_column + 1) * (a_column + 1) +
              0.007 * (a_row + 1) * (a_column + 1);
          const double wrong_entry = 0.41 + 0.13 * (peer_row + 1) +
              0.019 * (peer_column + 1) * (peer_column + 1) +
              0.007 * (peer_row + 1) * (peer_column + 1);
          peer_action[static_cast<std::size_t>(peer_row)] +=
              transformed_entry *
              peer_value[static_cast<std::size_t>(peer_column)];
          wrong_action[static_cast<std::size_t>(peer_row)] +=
              wrong_entry * peer_value[static_cast<std::size_t>(peer_column)];
        }
      for (int a_index = 0; a_index < scalar_dofs && valid; ++a_index)
      {
        const int peer_index =
            peer_for_a[static_cast<std::size_t>(a_index)];
        result.value_error = std::max(
            result.value_error,
            EntryError(a_value[static_cast<std::size_t>(a_index)],
                       peer_value[static_cast<std::size_t>(peer_index)]));
        result.action_error = std::max(
            result.action_error,
            EntryError(a_action[static_cast<std::size_t>(a_index)],
                       peer_action[static_cast<std::size_t>(peer_index)]));
        result.wrong_value_error = std::max(
            result.wrong_value_error,
            EntryError(a_value[static_cast<std::size_t>(peer_index)],
                       peer_value[static_cast<std::size_t>(peer_index)]));
        result.wrong_action_error = std::max(
            result.wrong_action_error,
            EntryError(a_action[static_cast<std::size_t>(a_index)],
                       wrong_action[static_cast<std::size_t>(peer_index)]));
      }
    }
  result.actual_shared_reversal_ok = rank_count == 1 ?
      result.shared_faces == 0 : result.actual_reversed_face_pairs > 0;
  result.wrong_permutation_rejected = rank_count == 1 ||
      (result.wrong_value_error > local_tolerance &&
       result.wrong_action_error > local_tolerance);
  result.passed = valid && result.finite_element_permutation_ok &&
      result.actual_shared_reversal_ok &&
      result.wrong_permutation_rejected &&
      result.value_error <= local_tolerance &&
      result.action_error <= local_tolerance;
  return result;
}

const char* FailurePointName(const SmoothEulerMPIFailurePoint point)
{
  switch (point)
  {
    case SmoothEulerMPIFailurePoint::None: return "None";
    case SmoothEulerMPIFailurePoint::Assembly: return "Assembly";
    case SmoothEulerMPIFailurePoint::Condensation: return "Condensation";
    case SmoothEulerMPIFailurePoint::TrueAssembly: return "TrueAssembly";
    case SmoothEulerMPIFailurePoint::TraceSolve: return "TraceSolve";
    case SmoothEulerMPIFailurePoint::Distribution: return "Distribution";
    case SmoothEulerMPIFailurePoint::Recovery: return "Recovery";
    case SmoothEulerMPIFailurePoint::CandidateValidation:
      return "CandidateValidation";
    case SmoothEulerMPIFailurePoint::Commit: return "Commit";
  }
  return "Invalid";
}

bool CheckOneOrder(MPI_Comm communicator, const int rank,
                   const int rank_count, const int polynomial_order)
{
  SmoothEulerMPIExperiment experiment(communicator, polynomial_order);
  const bool setup_ok = experiment.Initialize(3) &&
      experiment.status() == SmoothEulerMPIStatus::Ready &&
      experiment.generation() == 1 && experiment.Execute();
  bool passed = setup_ok;
  if (!GlobalAll(communicator, passed)) { return false; }
  const SmoothEulerMPIEvidence& evidence = experiment.evidence();
  mfem::ParMesh* const mesh = const_cast<mfem::ParMesh*>(experiment.mesh());
  mfem::ParFiniteElementSpace* const interior_space =
      const_cast<mfem::ParFiniteElementSpace*>(experiment.interior_space());
  mfem::ParFiniteElementSpace* const trace_space =
      const_cast<mfem::ParFiniteElementSpace*>(experiment.trace_space());
  const auto* const map = experiment.trace_map();
  const mfem::ParGridFunction* const old_interior = experiment.old_interior();
  const bool graph_requirements_ok = mesh != nullptr &&
      interior_space != nullptr && trace_space != nullptr && map != nullptr &&
      old_interior != nullptr;
  if (!GlobalAll(communicator, graph_requirements_ok))
  {
    std::cerr << "FAIL requirement=graph rank=" << rank
              << " ranks=" << rank_count << " p=" << polynomial_order
              << " local_ok=" << graph_requirements_ok << '\n';
    return false;
  }
  std::unique_ptr<mfem::ParGridFunction> entry_interior;
  std::unique_ptr<mfem::ParGridFunction> entry_trace;
  bool entry_assignment_ok = false;
  try
  {
    entry_interior =
        std::make_unique<mfem::ParGridFunction>(interior_space);
    entry_trace = std::make_unique<mfem::ParGridFunction>(trace_space);
    entry_assignment_ok = Assign(*entry_interior, evidence.entry_interior) &&
        Assign(*entry_trace, evidence.entry_trace);
  }
  catch (const std::exception&)
  {
    entry_assignment_ok = false;
  }
  passed = entry_assignment_ok && passed;
  if (!GlobalAll(communicator, entry_assignment_ok))
  {
    std::cerr << "FAIL requirement=entry_assignment rank=" << rank
              << " ranks=" << rank_count << " p=" << polynomial_order
              << " local_ok=" << entry_assignment_ok << '\n';
    return false;
  }
  const OwnerRefreshResult owner_refresh =
      CheckOwnerRefreshedInitialTrace(
          communicator, *mesh, *trace_space, *map, evidence);
  const bool owner_refresh_ok = owner_refresh.passed;
  passed = owner_refresh_ok && passed;
  OracleSystem local;
  const bool local_oracle_ok = AssembleOracle(
      *mesh, *interior_space, *trace_space, *old_interior, *entry_interior,
      *entry_trace, polynomial_order, local);
  passed = local_oracle_ok && passed;
  if (!GlobalAll(communicator, local_oracle_ok))
  {
    std::cerr << "FAIL requirement=local_oracle_prerequisite rank=" << rank
              << " ranks=" << rank_count << " p=" << polynomial_order
              << " local_ok=" << local_oracle_ok << '\n';
    return false;
  }
  const double initial_residual = GlobalNorm(communicator, local, *map);
  const SmoothEulerMPILinearization& production = evidence.linearization;
  const bool linearization_ok =
      production.local_elements == local.elements &&
      production.interior_dofs == local.ni &&
      production.trace_dofs == local.nt &&
      production.signed_trace_dofs == local.signed_trace_dofs &&
      ScaledError(production.residual_interior,
                  local.residual_interior) <= local_tolerance &&
      ScaledError(production.residual_trace,
                  local.residual_trace) <= local_tolerance &&
      ScaledError(production.a, local.a) <= local_tolerance &&
      ScaledError(production.b, local.b) <= local_tolerance &&
      ScaledError(production.c, local.c) <= local_tolerance &&
      ScaledError(production.d, local.d) <= local_tolerance;
  passed = linearization_ok && passed;

  std::vector<double> schur, reduced_rhs;
  std::vector<LUFactorization> factorizations;
  const bool condensation_ok =
      CondenseOracle(local, schur, reduced_rhs, &factorizations) &&
      ScaledError(evidence.condensation.schur, schur) <= local_tolerance &&
      ScaledError(evidence.condensation.right_hand_side, reduced_rhs) <=
          local_tolerance;
  passed = condensation_ok && passed;
  if (!GlobalAll(communicator, condensation_ok))
  {
    std::cerr << "FAIL requirement=condensation_prerequisite rank=" << rank
              << " ranks=" << rank_count << " p=" << polynomial_order
              << " local_ok=" << condensation_ok << '\n';
    return false;
  }

  GlobalOracle global;
  const bool global_oracle_ok = AssembleGlobalOracle(
      communicator, local, *map, schur, reduced_rhs, global);
  passed = global_oracle_ok && passed;
  if (!GlobalAll(communicator, global_oracle_ok))
  {
    std::cerr << "FAIL requirement=global_oracle_prerequisite rank=" << rank
              << " ranks=" << rank_count << " p=" << polynomial_order
              << " local_ok=" << global_oracle_ok << '\n';
    return false;
  }
  std::vector<double> expected_local_rhs(
      static_cast<std::size_t>(trace_space->GetVSize()), 0.0);
  for (int element = 0; element < local.elements; ++element)
    for (int row = 0; row < local.nt; ++row)
    {
      double sign = 1.0;
      const int dof = Decode(local.signed_trace_dofs[
          static_cast<std::size_t>(element) * local.nt + row], sign);
      expected_local_rhs[static_cast<std::size_t>(dof)] += sign *
          reduced_rhs[static_cast<std::size_t>(element) * local.nt + row];
    }
  const bool local_rhs_ok = ScaledError(
      evidence.condensation.local_assembled_right_hand_side,
      expected_local_rhs) <= local_tolerance;
  passed = local_rhs_ok && passed;

  std::vector<double> expected_true_rhs, expected_true_solution,
      expected_probe, expected_probe_action, expected_basis_actions;
  expected_true_rhs.assign(
      static_cast<std::size_t>(trace_space->GetTrueVSize()), 0.0);
  expected_true_solution = expected_true_rhs;
  expected_probe = expected_true_rhs;
  expected_probe_action = expected_true_rhs;
  expected_basis_actions.assign(
      static_cast<std::size_t>(global.size) * trace_space->GetTrueVSize(),
      0.0);
  for (const TraceMapEntry& entry : map->dofs.entries)
    if (entry.ownership == Ownership::owned)
    {
      expected_true_rhs[static_cast<std::size_t>(entry.local_true_dof)] =
          global.right_hand_side[
              static_cast<std::size_t>(entry.global_true_dof)];
      expected_true_solution[static_cast<std::size_t>(entry.local_true_dof)] =
          global.solution[static_cast<std::size_t>(entry.global_true_dof)];
      expected_probe[static_cast<std::size_t>(entry.local_true_dof)] =
          global.probe[static_cast<std::size_t>(entry.global_true_dof)];
      expected_probe_action[
          static_cast<std::size_t>(entry.local_true_dof)] =
          global.probe_action[
              static_cast<std::size_t>(entry.global_true_dof)];
      for (int column = 0; column < global.size; ++column)
      {
        expected_basis_actions[
            static_cast<std::size_t>(column) * trace_space->GetTrueVSize() +
            entry.local_true_dof] = global.matrix[
                static_cast<std::size_t>(entry.global_true_dof) *
                    global.size +
                column];
      }
    }
  const bool true_system_ok =
      ScaledError(evidence.true_right_hand_side, expected_true_rhs) <=
          local_tolerance &&
      ScaledError(evidence.true_trace_increment, expected_true_solution) <=
          state_tolerance &&
      ScaledError(evidence.true_probe, expected_probe) <= local_tolerance &&
      ScaledError(evidence.true_probe_action, expected_probe_action) <=
          local_tolerance &&
      ScaledError(evidence.true_basis_actions, expected_basis_actions) <=
          local_tolerance &&
      evidence.true_relative_residual <= true_residual_tolerance &&
      DenseRelativeResidual(global) <= true_residual_tolerance;
  passed = true_system_ok && passed;

  std::vector<double> expected_distributed(
      static_cast<std::size_t>(trace_space->GetVSize()), 0.0);
  for (const TraceMapEntry& entry : map->dofs.entries)
  {
    expected_distributed[static_cast<std::size_t>(entry.local_dof)] =
        global.solution[static_cast<std::size_t>(entry.global_true_dof)];
  }
  const bool distribution_ok = ScaledError(
      evidence.distributed_trace_increment,
      expected_distributed) <= state_tolerance;
  passed = distribution_ok && passed;

  std::vector<double> expected_recovered(
      static_cast<std::size_t>(local.elements) * local.ni, 0.0);
  bool recovery_solve_ok = true;
  for (int element = 0; element < local.elements; ++element)
  {
    std::vector<double> local_trace(static_cast<std::size_t>(local.nt), 0.0);
    for (int column = 0; column < local.nt; ++column)
    {
      double sign = 1.0;
      const int dof = Decode(local.signed_trace_dofs[
          static_cast<std::size_t>(element) * local.nt + column], sign);
      local_trace[static_cast<std::size_t>(column)] = sign *
          expected_distributed[static_cast<std::size_t>(dof)];
    }
    std::vector<double> recovery_rhs(
        local.f.begin() + static_cast<std::size_t>(element) * local.ni,
        local.f.begin() + static_cast<std::size_t>(element + 1) * local.ni);
    for (int row = 0; row < local.ni; ++row)
      for (int column = 0; column < local.nt; ++column)
      {
        recovery_rhs[static_cast<std::size_t>(row)] -= local.b[
            static_cast<std::size_t>(element) * local.ni * local.nt +
            static_cast<std::size_t>(row) * local.nt + column] *
            local_trace[static_cast<std::size_t>(column)];
      }
    std::vector<double> recovered;
    recovery_solve_ok =
        factorizations[static_cast<std::size_t>(element)].Solve(
            recovery_rhs, recovered) && recovery_solve_ok;
    std::copy(recovered.begin(), recovered.end(),
              expected_recovered.begin() +
                  static_cast<std::size_t>(element) * local.ni);
  }
  const bool recovery_ok = recovery_solve_ok &&
      ScaledError(evidence.recovered_interior_increment,
                  expected_recovered) <= state_tolerance;
  passed = recovery_ok && passed;

  std::vector<double> expected_final_interior = evidence.entry_interior;
  std::vector<double> expected_final_trace = evidence.entry_trace;
  std::vector<double> grid_recovered(expected_final_interior.size(), 0.0);
  std::vector<int> interior_census(grid_recovered.size(), 0);
  bool interior_scatter_ok = true;
  for (int element = 0; element < local.elements; ++element)
  {
    mfem::Array<int> interior_vdofs;
    interior_space->GetElementVDofs(element, interior_vdofs);
    interior_scatter_ok =
        interior_vdofs.Size() == local.ni && interior_scatter_ok;
    for (int local_dof = 0; local_dof < interior_vdofs.Size(); ++local_dof)
    {
      double sign = 1.0;
      const int vdof = Decode(interior_vdofs[local_dof], sign);
      interior_scatter_ok = vdof >= 0 &&
          static_cast<std::size_t>(vdof) < grid_recovered.size() &&
          interior_scatter_ok;
      if (vdof >= 0 && static_cast<std::size_t>(vdof) < grid_recovered.size())
      {
        grid_recovered[static_cast<std::size_t>(vdof)] = sign *
            expected_recovered[static_cast<std::size_t>(element) * local.ni +
                               local_dof];
        ++interior_census[static_cast<std::size_t>(vdof)];
      }
    }
  }
  interior_scatter_ok =
      std::all_of(interior_census.begin(), interior_census.end(),
                  [](const int count) { return count == 1; }) &&
      interior_scatter_ok;
  for (std::size_t index = 0; index < expected_final_interior.size(); ++index)
  {
    expected_final_interior[index] +=
        evidence.damping * grid_recovered[index];
  }
  for (std::size_t index = 0; index < expected_final_trace.size(); ++index)
  {
    expected_final_trace[index] +=
        evidence.damping * expected_distributed[index];
  }
  const bool state_publication_ok = interior_scatter_ok &&
      ScaledError(evidence.final_interior,
                  expected_final_interior) <= state_tolerance &&
      ScaledError(evidence.final_trace, expected_final_trace) <=
          state_tolerance &&
      ScaledError(Values(*experiment.current_interior()),
                  evidence.final_interior) == 0.0 &&
      ScaledError(Values(*experiment.current_trace()),
                  evidence.final_trace) == 0.0;
  passed = state_publication_ok && passed;

  const SerialParityResult serial_parity = CheckSerialParity(
      communicator, polynomial_order, *mesh, *interior_space, *trace_space,
      *map, evidence);
  const bool serial_parity_ok = serial_parity.passed;
  passed = serial_parity_ok && passed;

  std::unique_ptr<mfem::ParGridFunction> final_interior;
  std::unique_ptr<mfem::ParGridFunction> final_trace;
  OracleSystem final_system;
  bool final_oracle_ok = false;
  try
  {
    final_interior =
        std::make_unique<mfem::ParGridFunction>(interior_space);
    final_trace = std::make_unique<mfem::ParGridFunction>(trace_space);
    final_oracle_ok = Assign(*final_interior, evidence.final_interior) &&
        Assign(*final_trace, evidence.final_trace) &&
        AssembleOracle(*mesh, *interior_space, *trace_space, *old_interior,
                       *final_interior, *final_trace, polynomial_order,
                       final_system);
  }
  catch (const std::exception&)
  {
    final_oracle_ok = false;
  }
  passed = final_oracle_ok && passed;
  if (!GlobalAll(communicator, final_oracle_ok))
  {
    std::cerr << "FAIL requirement=final_oracle_prerequisite rank=" << rank
              << " ranks=" << rank_count << " p=" << polynomial_order
              << " local_ok=" << final_oracle_ok << '\n';
    return false;
  }
  const double final_residual = GlobalNorm(communicator, final_system, *map);
  const int local_negative_signed_trace_dofs =
      static_cast<int>(std::count_if(
          local.signed_trace_dofs.begin(), local.signed_trace_dofs.end(),
          [](const int dof) { return dof < 0; }));
  int global_negative_signed_trace_dofs = 0;
  MPI_Allreduce(&local_negative_signed_trace_dofs,
                &global_negative_signed_trace_dofs, 1, MPI_INT, MPI_SUM,
                communicator);
  const OrientationOracleResult orientation = CheckOrientationOracle(
      communicator, *mesh, *trace_space, *map, rank, polynomial_order,
      rank_count);
  const bool nonlinear_residual_ok =
      std::abs(final_residual - evidence.candidate_nonlinear_residual) /
          std::max(1.0, final_residual) <= local_tolerance &&
      std::abs(initial_residual - evidence.initial_nonlinear_residual) /
              std::max(1.0, initial_residual) <= local_tolerance &&
      final_residual < evidence.initial_nonlinear_residual;
  const bool status_publication_ok =
      evidence.status == SmoothEulerMPIStatus::Success && evidence.passed &&
      experiment.status() == SmoothEulerMPIStatus::Success;
  const bool generation_publication_ok = evidence.entry_generation == 1 &&
      evidence.committed_generation == 2 && experiment.generation() == 2;
  const bool evidence_identity_ok =
      experiment.status() == SmoothEulerMPIStatus::Success &&
      evidence.polynomial_order == polynomial_order &&
      evidence.elements_per_direction == 3 &&
      evidence.ranks == rank_count && evidence.rank == rank &&
      evidence.global_elements == 9 &&
      evidence.owned_trace_dofs ==
          static_cast<std::size_t>(trace_space->GetTrueVSize()) &&
      evidence.global_true_trace_dofs == map->dofs.global_true_dof_count &&
      evidence.negative_signed_trace_dofs ==
          local_negative_signed_trace_dofs &&
      evidence.global_negative_signed_trace_dofs ==
          global_negative_signed_trace_dofs;
  const bool orientation_requirements_ok = orientation.passed;
  passed = nonlinear_residual_ok && status_publication_ok &&
      generation_publication_ok && evidence_identity_ok &&
      orientation_requirements_ok && passed;

  std::vector<int> local_owners;
  std::vector<int> global_owners;
  bool owner_census_staging_ok = true;
  try
  {
    local_owners.assign(static_cast<std::size_t>(global.size), 0);
    global_owners.assign(local_owners.size(), 0);
    for (const TraceMapEntry& entry : map->dofs.entries)
      if (entry.ownership == Ownership::owned)
      {
        owner_census_staging_ok = entry.global_true_dof >= 0 &&
            entry.global_true_dof < global.size && owner_census_staging_ok;
        if (owner_census_staging_ok)
        {
          ++local_owners[static_cast<std::size_t>(entry.global_true_dof)];
        }
      }
  }
  catch (const std::exception&)
  {
    owner_census_staging_ok = false;
  }
  if (!GlobalAll(communicator, owner_census_staging_ok))
  {
    std::cerr << "FAIL requirement=ownership_census_staging rank=" << rank
              << " ranks=" << rank_count << " p=" << polynomial_order
              << " local_ok=" << owner_census_staging_ok << '\n';
    return false;
  }
  MPI_Allreduce(local_owners.data(), global_owners.data(), global.size,
                MPI_INT, MPI_SUM, communicator);
  const bool single_owner_requirement_ok =
      std::all_of(global_owners.begin(), global_owners.end(),
                  [](const int count) { return count == 1; });
  const bool ghost_census_requirement_ok = rank_count == 1 ?
      evidence.global_ghost_trace_dofs == 0 :
      evidence.global_ghost_trace_dofs > 0;
  const bool shared_face_census_requirement_ok = rank_count == 1 ?
      evidence.global_shared_faces == 0 :
      evidence.global_shared_faces > 0;
  passed = single_owner_requirement_ok && ghost_census_requirement_ok &&
      shared_face_census_requirement_ok && owner_refresh_ok && passed;

  std::vector<double> rollback_interior;
  std::vector<double> rollback_trace;
  const std::uint64_t rollback_generation = experiment.generation();
  const SmoothEulerMPIStatus rollback_status = experiment.status();
  std::uint64_t rollback_evidence_hash = 0;
  bool rollback_staging_ok = true;
  try
  {
    rollback_interior = Values(*experiment.current_interior());
    rollback_trace = Values(*experiment.current_trace());
    rollback_evidence_hash = EvidenceHash(experiment.evidence());
  }
  catch (const std::exception&)
  {
    rollback_staging_ok = false;
  }
  if (!GlobalAll(communicator, rollback_staging_ok))
  {
    std::cerr << "FAIL requirement=rollback_staging rank=" << rank
              << " ranks=" << rank_count << " p=" << polynomial_order
              << " local_ok=" << rollback_staging_ok << '\n';
    return false;
  }
  bool rollback_ok = true;
  int rollback_cases_exercised = 0;
  if (polynomial_order == 1)
  {
    const std::array<SmoothEulerMPIFailurePoint, 8> failure_points{{
        SmoothEulerMPIFailurePoint::Assembly,
        SmoothEulerMPIFailurePoint::Condensation,
        SmoothEulerMPIFailurePoint::TrueAssembly,
        SmoothEulerMPIFailurePoint::TraceSolve,
        SmoothEulerMPIFailurePoint::Distribution,
        SmoothEulerMPIFailurePoint::Recovery,
        SmoothEulerMPIFailurePoint::CandidateValidation,
        SmoothEulerMPIFailurePoint::Commit}};
    for (const SmoothEulerMPIFailurePoint point : failure_points)
      for (int failing_rank = 0; failing_rank < rank_count; ++failing_rank)
      {
        ++rollback_cases_exercised;
        const bool return_requirement_ok =
            !experiment.Execute({point, failing_rank});
        const bool phase_requirement_ok =
            experiment.last_failure_point_reached() == point;
        const bool generation_rollback_ok =
            experiment.generation() == rollback_generation;
        const bool status_rollback_ok =
            experiment.status() == rollback_status;
        bool evidence_byte_rollback_ok = false;
        bool interior_byte_rollback_ok = false;
        bool trace_byte_rollback_ok = false;
        bool rollback_observation_ok = true;
        try
        {
          evidence_byte_rollback_ok =
              EvidenceHash(experiment.evidence()) == rollback_evidence_hash;
          interior_byte_rollback_ok =
              Values(*experiment.current_interior()) == rollback_interior;
          trace_byte_rollback_ok =
              Values(*experiment.current_trace()) == rollback_trace;
        }
        catch (const std::exception&)
        {
          rollback_observation_ok = false;
        }
        rollback_observation_ok =
            GlobalAll(communicator, rollback_observation_ok);
        const bool rollback_case_ok = return_requirement_ok &&
            phase_requirement_ok && generation_rollback_ok &&
            status_rollback_ok && evidence_byte_rollback_ok &&
            interior_byte_rollback_ok && trace_byte_rollback_ok &&
            rollback_observation_ok;
        if (!rollback_case_ok)
        {
          std::cerr << "FAIL requirement=rollback rank=" << rank
                    << " ranks=" << rank_count << " p=" << polynomial_order
                    << " phase=" << FailurePointName(point)
                    << " injected_rank=" << failing_rank
                    << " return_ok=" << return_requirement_ok
                    << " phase_reached_ok=" << phase_requirement_ok
                    << " generation_ok=" << generation_rollback_ok
                    << " status_ok=" << status_rollback_ok
                    << " observation_ok=" << rollback_observation_ok
                    << " evidence_bytes_ok=" << evidence_byte_rollback_ok
                    << " interior_bytes_ok=" << interior_byte_rollback_ok
                    << " trace_bytes_ok=" << trace_byte_rollback_ok << '\n';
        }
        rollback_ok = rollback_case_ok && rollback_ok;
      }
  }
  passed = rollback_ok && passed;

  const double minimum_damping = std::ldexp(1.0, -12);
  const bool damping_ok = evidence.damping >= minimum_damping &&
      evidence.damping <= 1.0 &&
      std::abs(std::log2(evidence.damping) -
               std::round(std::log2(evidence.damping))) == 0.0;
  passed = damping_ok && passed;

  if (!passed)
  {
    const auto report_requirement = [&](const char* name, const bool ok)
    {
      if (!ok)
      {
        std::cerr << "FAIL requirement=" << name << " rank=" << rank
                  << " ranks=" << rank_count << " p=" << polynomial_order
                  << '\n';
      }
    };
    report_requirement("setup", setup_ok);
    report_requirement("entry_assignment", entry_assignment_ok);
    report_requirement("owner_refresh_staging", owner_refresh.staging_ok);
    report_requirement("owner_refresh_owner_bytes",
                       owner_refresh.exact_owner_bytes);
    report_requirement("owner_refresh_ghost_bytes",
                       owner_refresh.exact_ghost_bytes);
    report_requirement("owner_refresh_coordinate_values",
                       owner_refresh.coordinate_values_ok);
    report_requirement("owner_refresh", owner_refresh_ok);
    report_requirement("local_oracle", local_oracle_ok);
    report_requirement("linearization", linearization_ok);
    report_requirement("condensation", condensation_ok);
    report_requirement("global_oracle", global_oracle_ok);
    report_requirement("local_rhs", local_rhs_ok);
    report_requirement("true_system", true_system_ok);
    report_requirement("distribution", distribution_ok);
    report_requirement("recovery", recovery_ok);
    report_requirement("state_publication", state_publication_ok);
    report_requirement("serial_parity_truth", serial_parity.serial_truth_ok);
    report_requirement("serial_parity_coordinate_keys",
                       serial_parity.coordinate_keys_ok);
    report_requirement("serial_parity_assembly", serial_parity.assembly_ok);
    report_requirement("serial_parity_condensation",
                       serial_parity.condensation_ok);
    report_requirement("serial_parity_action_rhs",
                       serial_parity.action_rhs_ok);
    report_requirement("serial_parity_trace_solution",
                       serial_parity.trace_solution_ok);
    report_requirement("serial_parity_recovery", serial_parity.recovery_ok);
    report_requirement("serial_parity_state", serial_parity.state_ok);
    report_requirement("serial_parity_status", serial_parity.status_ok);
    report_requirement("serial_parity", serial_parity_ok);
    report_requirement("final_oracle", final_oracle_ok);
    report_requirement("nonlinear_residual", nonlinear_residual_ok);
    report_requirement("status_publication", status_publication_ok);
    report_requirement("generation_publication", generation_publication_ok);
    report_requirement("evidence_identity", evidence_identity_ok);
    report_requirement("orientation_finite_element_permutation",
                       orientation.finite_element_permutation_ok);
    report_requirement("orientation_actual_shared_reversal",
                       orientation.actual_shared_reversal_ok);
    report_requirement("orientation_wrong_permutation_rejected",
                       orientation.wrong_permutation_rejected);
    report_requirement("orientation_values_actions",
                       orientation_requirements_ok);
    report_requirement("ownership_single_owner", single_owner_requirement_ok);
    report_requirement("ownership_ghost_census",
                       ghost_census_requirement_ok);
    report_requirement("ownership_shared_face_census",
                       shared_face_census_requirement_ok);
    report_requirement("rollback", rollback_ok);
    report_requirement("damping", damping_ok);
    std::cerr << "FAIL metrics rank=" << rank << " ranks=" << rank_count
              << " p=" << polynomial_order << " local_error_R="
              << ScaledError(production.residual_interior,
                             local.residual_interior)
              << " local_error_G="
              << ScaledError(production.residual_trace,
                             local.residual_trace)
              << " local_error_A=" << ScaledError(production.a, local.a)
              << " local_error_B=" << ScaledError(production.b, local.b)
              << " local_error_C=" << ScaledError(production.c, local.c)
              << " local_error_D=" << ScaledError(production.d, local.d)
              << " schur_error="
              << ScaledError(evidence.condensation.schur, schur)
              << " condensed_rhs_error="
              << ScaledError(evidence.condensation.right_hand_side,
                             reduced_rhs)
              << " true_rhs_error="
              << ScaledError(evidence.true_right_hand_side,
                             expected_true_rhs)
              << " probe_error="
              << ScaledError(evidence.true_probe, expected_probe)
              << " probe_action_error="
              << ScaledError(evidence.true_probe_action,
                             expected_probe_action)
              << " trace_error="
              << ScaledError(evidence.true_trace_increment,
                             expected_true_solution)
              << " basis_action_error="
              << ScaledError(evidence.true_basis_actions,
                             expected_basis_actions)
              << " recovery_error="
              << ScaledError(evidence.recovered_interior_increment,
                             expected_recovered)
              << " final_interior_error="
              << ScaledError(evidence.final_interior,
                             expected_final_interior)
              << " final_trace_error="
              << ScaledError(evidence.final_trace, expected_final_trace)
              << " initial_residual=" << initial_residual
              << " published_initial_residual="
              << evidence.initial_nonlinear_residual
              << " final_residual=" << final_residual
              << " published_final_residual="
              << evidence.candidate_nonlinear_residual
              << " negative_signed_local="
              << local_negative_signed_trace_dofs
              << " negative_signed_global="
              << global_negative_signed_trace_dofs
              << " orientation_shared_faces=" << orientation.shared_faces
              << " orientation_reversed_dofs=" << orientation.reversed_dofs
              << " orientation_actual_pairs="
              << orientation.actual_reversed_face_pairs
              << " orientation_value_error=" << orientation.value_error
              << " orientation_action_error=" << orientation.action_error
              << " orientation_wrong_value_error="
              << orientation.wrong_value_error
              << " orientation_wrong_action_error="
              << orientation.wrong_action_error
              << " owner_values=" << owner_refresh.owner_values
              << " ghost_values=" << owner_refresh.ghost_values
              << " owner_value_error=" << owner_refresh.owner_value_error
              << " ghost_value_error=" << owner_refresh.ghost_value_error
              << " owner_coordinate_error="
              << owner_refresh.coordinate_value_error
              << " serial_matched_elements="
              << serial_parity.matched_elements
              << " serial_residual_error=" << serial_parity.residual_error
              << " serial_jacobian_error=" << serial_parity.jacobian_error
              << " serial_condensation_error="
              << serial_parity.condensation_error
              << " serial_matrix_error=" << serial_parity.matrix_error
              << " serial_action_error=" << serial_parity.action_error
              << " serial_rhs_error="
              << serial_parity.right_hand_side_error
              << " serial_trace_solution_error="
              << serial_parity.trace_solution_error
              << " serial_recovery_error=" << serial_parity.recovery_error
              << " serial_state_error=" << serial_parity.state_error
              << " rollback_cases=" << rollback_cases_exercised
              << " damping=" << evidence.damping
              << " true_residual=" << evidence.true_relative_residual
              << " dense_residual=" << DenseRelativeResidual(global)
              << '\n';
  }
  return GlobalAll(communicator, passed);
}

bool CheckParameterConsensus(MPI_Comm communicator, const int rank,
                             const int rank_count)
{
  bool constructor_rejected = false;
  try
  {
    const int mismatched_order = rank_count == 2 ? rank + 1 : 0;
    SmoothEulerMPIExperiment mismatched(communicator, mismatched_order);
    (void)mismatched;
  }
  catch (const std::invalid_argument&)
  {
    constructor_rejected = true;
  }
  catch (const std::exception&)
  {
    constructor_rejected = false;
  }
  bool passed = GlobalAll(communicator, constructor_rejected);

  SmoothEulerMPIExperiment experiment(communicator, 1);
  const int mismatched_elements = rank_count == 2 && rank == 0 ? 3 : 2;
  const bool initialize_rejected =
      !experiment.Initialize(mismatched_elements);
  passed = GlobalAll(communicator, initialize_rejected) &&
      experiment.status() == SmoothEulerMPIStatus::Uninitialized &&
      experiment.generation() == 0 && experiment.mesh() == nullptr && passed;
  passed = experiment.Initialize(3) &&
      experiment.status() == SmoothEulerMPIStatus::Ready &&
      experiment.generation() == 1 && experiment.mesh() != nullptr && passed;
  return GlobalAll(communicator, passed);
}

} // namespace

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int rank = -1;
  int rank_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
  if (rank_count != 1 && rank_count != 2)
  {
    MPI_Finalize();
    return 75;
  }

  bool passed = CheckParameterConsensus(MPI_COMM_WORLD, rank, rank_count);
  for (int polynomial_order = 1; polynomial_order <= 4; ++polynomial_order)
  {
    passed = CheckOneOrder(MPI_COMM_WORLD, rank, rank_count,
                           polynomial_order) && passed;
  }
  const bool global_passed = GlobalAll(MPI_COMM_WORLD, passed);
  MPI_Finalize();
  return global_passed ? 0 : 1;
}

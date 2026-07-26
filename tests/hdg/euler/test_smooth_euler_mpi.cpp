#include "numerics/hdg/euler/smooth_euler_mpi.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

bool AssembleOracle(mfem::ParMesh& mesh,
                    mfem::ParFiniteElementSpace& interior_space,
                    mfem::ParFiniteElementSpace& trace_space,
                    const mfem::ParGridFunction& old_interior,
                    const mfem::ParGridFunction& current_interior,
                    const mfem::ParGridFunction& current_trace,
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
  if (map.dofs.global_true_dof_count <= 0 ||
      map.dofs.global_true_dof_count > std::numeric_limits<int>::max())
  {
    return false;
  }
  next.size = static_cast<int>(map.dofs.global_true_dof_count);
  std::vector<double> local_matrix(
      static_cast<std::size_t>(next.size) * next.size, 0.0);
  std::vector<double> local_rhs(static_cast<std::size_t>(next.size), 0.0);
  for (int element = 0; element < local.elements; ++element)
    for (int row = 0; row < local.nt; ++row)
    {
      double row_sign = 1.0;
      const int local_row = Decode(local.signed_trace_dofs[
          static_cast<std::size_t>(element) * local.nt + row], row_sign);
      if (local_row < 0 ||
          static_cast<std::size_t>(local_row) >= map.dofs.entries.size())
      {
        return false;
      }
      const std::int64_t global_row =
          map.dofs.entries[static_cast<std::size_t>(local_row)].global_true_dof;
      local_rhs[static_cast<std::size_t>(global_row)] += row_sign *
          reduced_rhs[static_cast<std::size_t>(element) * local.nt + row];
      for (int column = 0; column < local.nt; ++column)
      {
        double column_sign = 1.0;
        const int local_column = Decode(local.signed_trace_dofs[
            static_cast<std::size_t>(element) * local.nt + column],
            column_sign);
        if (local_column < 0 ||
            static_cast<std::size_t>(local_column) >= map.dofs.entries.size())
        {
          return false;
        }
        const std::int64_t global_column = map.dofs.entries[
            static_cast<std::size_t>(local_column)].global_true_dof;
        local_matrix[static_cast<std::size_t>(global_row) * next.size +
                     global_column] += row_sign * column_sign * schur[
            static_cast<std::size_t>(element) * local.nt * local.nt +
            static_cast<std::size_t>(row) * local.nt + column];
      }
    }
  next.matrix.resize(local_matrix.size());
  next.right_hand_side.resize(local_rhs.size());
  MPI_Allreduce(local_matrix.data(), next.matrix.data(),
                static_cast<int>(local_matrix.size()), MPI_DOUBLE, MPI_SUM,
                communicator);
  MPI_Allreduce(local_rhs.data(), next.right_hand_side.data(), next.size,
                MPI_DOUBLE, MPI_SUM, communicator);
  LUFactorization factorization;
  if (!factorization.Factor(next.matrix.data(), next.size) ||
      !factorization.Solve(next.right_hand_side, next.solution))
  {
    return false;
  }
  next.probe.resize(static_cast<std::size_t>(next.size));
  next.probe_action.assign(static_cast<std::size_t>(next.size), 0.0);
  for (int row = 0; row < next.size; ++row)
  {
    next.probe[static_cast<std::size_t>(row)] =
        0.25 + std::sin(0.01 * static_cast<double>(row + 1));
  }
  for (int row = 0; row < next.size; ++row)
  {
    for (int column = 0; column < next.size; ++column)
    {
      next.probe_action[static_cast<std::size_t>(row)] +=
          next.matrix[static_cast<std::size_t>(row) * next.size + column] *
          next.probe[static_cast<std::size_t>(column)];
    }
  }
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
  if (map.dofs.global_true_dof_count <= 0 ||
      map.dofs.global_true_dof_count > std::numeric_limits<int>::max() ||
      map.dofs.entries.empty() ||
      system.residual_trace.size() != system.signed_trace_dofs.size())
  {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<double> local_trace(map.dofs.entries.size(), 0.0);
  for (std::size_t local = 0; local < system.residual_trace.size(); ++local)
  {
    double sign = 1.0;
    const int dof = Decode(system.signed_trace_dofs[local], sign);
    if (dof < 0 || static_cast<std::size_t>(dof) >= local_trace.size())
    {
      return std::numeric_limits<double>::infinity();
    }
    local_trace[static_cast<std::size_t>(dof)] +=
        sign * system.residual_trace[local];
  }
  const int global_size =
      static_cast<int>(map.dofs.global_true_dof_count);
  std::vector<double> local_global_trace(
      static_cast<std::size_t>(global_size), 0.0);
  for (const TraceMapEntry& entry : map.dofs.entries)
  {
    if (entry.local_dof < 0 ||
        static_cast<std::size_t>(entry.local_dof) >= local_trace.size() ||
        entry.global_true_dof < 0 || entry.global_true_dof >= global_size)
    {
      return std::numeric_limits<double>::infinity();
    }
    local_global_trace[static_cast<std::size_t>(entry.global_true_dof)] +=
        local_trace[static_cast<std::size_t>(entry.local_dof)];
  }
  std::vector<double> global_trace(local_global_trace.size(), 0.0);
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

struct OrientationOracleResult
{
  bool passed = false;
  int shared_faces = 0;
  int reversed_dofs = 0;
  double value_error = std::numeric_limits<double>::infinity();
  double action_error = std::numeric_limits<double>::infinity();
};

OrientationOracleResult CheckOrientationOracle(
    mfem::ParMesh& mesh, mfem::ParFiniteElementSpace& trace_space,
    const dg::hdg::trace::DGInterfaceTraceMap& map,
    const int polynomial_order, const int rank_count)
{
  OrientationOracleResult result;
  const mfem::FiniteElementCollection* const collection = trace_space.FEColl();
  const mfem::FiniteElement* const finite_element =
      collection == nullptr ? nullptr :
          collection->FiniteElementForGeometry(mfem::Geometry::SEGMENT);
  const int scalar_dofs = polynomial_order + 1;
  const int* const permutation = collection == nullptr ? nullptr :
      collection->DofOrderForOrientation(mfem::Geometry::SEGMENT, -1);
  if (finite_element == nullptr || permutation == nullptr ||
      finite_element->GetDof() != scalar_dofs)
  {
    return result;
  }

  bool valid = true;
  for (int destination = 0; destination < scalar_dofs; ++destination)
  {
    valid = permutation[destination] == scalar_dofs - 1 - destination &&
        permutation[destination] >= 0 && valid;
    result.reversed_dofs += permutation[destination] != destination ? 1 : 0;
  }
  result.shared_faces = mesh.GetNSharedFaces();
  valid = result.reversed_dofs > 0 &&
      (rank_count == 1 ? result.shared_faces == 0 : result.shared_faces > 0) &&
      valid;

  std::vector<int> faces;
  if (rank_count == 1)
  {
    if (mesh.GetNumFaces() > 0) faces.push_back(0);
  }
  else
  {
    for (int shared = 0; shared < result.shared_faces; ++shared)
    {
      faces.push_back(mesh.GetSharedFace(shared));
    }
  }
  valid = !faces.empty() && valid;
  result.value_error = 0.0;
  result.action_error = 0.0;
  for (const int face : faces)
  {
    mfem::Array<int> direct;
    trace_space.GetFaceVDofs(face, direct);
    const auto face_map = std::find_if(
        map.faces.begin(), map.faces.end(),
        [face](const dg::hdg::trace::DGInterfaceFaceMap& entry)
        { return entry.local_face == face; });
    valid = direct.Size() == components * scalar_dofs &&
        face_map != map.faces.end() &&
        (face_map == map.faces.end() ||
         std::equal(direct.begin(), direct.end(),
                    face_map->signed_local_vdofs.begin(),
                    face_map->signed_local_vdofs.end())) && valid;

    for (const double point : {0.07, 0.23, 0.61, 0.91})
    {
      mfem::IntegrationPoint forward_point;
      mfem::IntegrationPoint reverse_point;
      forward_point.x = point;
      reverse_point.x = 1.0 - point;
      mfem::Vector forward(scalar_dofs), reverse(scalar_dofs);
      finite_element->CalcShape(forward_point, forward);
      finite_element->CalcShape(reverse_point, reverse);
      for (int component = 0; component < components; ++component)
      {
        std::vector<double> value(static_cast<std::size_t>(scalar_dofs), 0.0);
        std::vector<double> reversed_value(value.size(), 0.0);
        for (int dof = 0; dof < scalar_dofs; ++dof)
        {
          value[static_cast<std::size_t>(dof)] =
              0.25 + 0.03 * component + 0.07 * dof;
        }
        for (int dof = 0; dof < scalar_dofs; ++dof)
        {
          reversed_value[static_cast<std::size_t>(dof)] =
              value[static_cast<std::size_t>(permutation[dof])];
        }
        double forward_action = 0.0;
        double reverse_action = 0.0;
        for (int dof = 0; dof < scalar_dofs; ++dof)
        {
          forward_action += forward[dof] * reversed_value[
              static_cast<std::size_t>(dof)];
          reverse_action +=
              reverse[dof] * value[static_cast<std::size_t>(dof)];
        }
        result.value_error = std::max(
            result.value_error, std::abs(forward_action - reverse_action));

        std::vector<double> action(value.size(), 0.0);
        std::vector<double> reversed_action(value.size(), 0.0);
        for (int row = 0; row < scalar_dofs; ++row)
          for (int column = 0; column < scalar_dofs; ++column)
          {
            const double entry =
                (row == column ? 2.0 : 0.0) +
                0.01 * (row + 1) * (column + 2) + 0.001 * component;
            action[static_cast<std::size_t>(row)] +=
                entry * value[static_cast<std::size_t>(column)];
            const double reversed_entry =
                (permutation[row] == permutation[column] ? 2.0 : 0.0) +
                0.01 * (permutation[row] + 1) *
                    (permutation[column] + 2) + 0.001 * component;
            reversed_action[static_cast<std::size_t>(row)] +=
                reversed_entry *
                reversed_value[static_cast<std::size_t>(column)];
          }
        for (int row = 0; row < scalar_dofs; ++row)
        {
          result.action_error = std::max(
              result.action_error,
              std::abs(reversed_action[static_cast<std::size_t>(row)] -
                       action[static_cast<std::size_t>(permutation[row])]));
        }
      }
    }
  }
  result.passed = valid && result.value_error <= local_tolerance &&
      result.action_error <= local_tolerance;
  return result;
}

bool GlobalAll(MPI_Comm communicator, const bool local)
{
  int local_value = local ? 1 : 0;
  int global_value = 0;
  MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                communicator);
  return global_value == 1;
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
  if (mesh == nullptr || interior_space == nullptr || trace_space == nullptr ||
      map == nullptr || old_interior == nullptr)
  {
    return false;
  }
  mfem::ParGridFunction entry_interior(interior_space);
  mfem::ParGridFunction entry_trace(trace_space);
  const bool entry_assignment_ok =
      Assign(entry_interior, evidence.entry_interior) &&
      Assign(entry_trace, evidence.entry_trace);
  passed = entry_assignment_ok && passed;
  OracleSystem local;
  const bool local_oracle_ok = AssembleOracle(
      *mesh, *interior_space, *trace_space, *old_interior, entry_interior,
      entry_trace, polynomial_order, local);
  passed = local_oracle_ok && passed;
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

  GlobalOracle global;
  const bool global_oracle_ok = AssembleGlobalOracle(
      communicator, local, *map, schur, reduced_rhs, global);
  passed = global_oracle_ok && passed;
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

  mfem::ParGridFunction final_interior(interior_space);
  mfem::ParGridFunction final_trace(trace_space);
  OracleSystem final_system;
  const bool final_oracle_ok =
      Assign(final_interior, evidence.final_interior) &&
      Assign(final_trace, evidence.final_trace) &&
      AssembleOracle(*mesh, *interior_space, *trace_space, *old_interior,
                     final_interior, final_trace, polynomial_order,
                     final_system);
  passed = final_oracle_ok && passed;
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
      *mesh, *trace_space, *map, polynomial_order, rank_count);
  const bool nonlinear_and_evidence_ok =
      std::abs(final_residual - evidence.candidate_nonlinear_residual) /
          std::max(1.0, final_residual) <= local_tolerance &&
      std::abs(initial_residual - evidence.initial_nonlinear_residual) /
              std::max(1.0, initial_residual) <= local_tolerance &&
      final_residual < evidence.initial_nonlinear_residual &&
      evidence.status == SmoothEulerMPIStatus::Success && evidence.passed &&
      evidence.entry_generation == 1 && evidence.committed_generation == 2 &&
      experiment.status() == SmoothEulerMPIStatus::Success &&
      experiment.generation() == 2 && evidence.polynomial_order ==
          polynomial_order && evidence.elements_per_direction == 3 &&
      evidence.ranks == rank_count && evidence.rank == rank &&
      evidence.global_elements == 9 &&
      evidence.owned_trace_dofs ==
          static_cast<std::size_t>(trace_space->GetTrueVSize()) &&
      evidence.global_true_trace_dofs == map->dofs.global_true_dof_count &&
      evidence.negative_signed_trace_dofs ==
          local_negative_signed_trace_dofs &&
      evidence.global_negative_signed_trace_dofs ==
          global_negative_signed_trace_dofs &&
      orientation.passed;
  passed = nonlinear_and_evidence_ok && passed;

  std::vector<int> local_owners(static_cast<std::size_t>(global.size), 0);
  for (const TraceMapEntry& entry : map->dofs.entries)
    if (entry.ownership == Ownership::owned)
    {
      ++local_owners[static_cast<std::size_t>(entry.global_true_dof)];
    }
  std::vector<int> global_owners(local_owners.size(), 0);
  MPI_Allreduce(local_owners.data(), global_owners.data(), global.size,
                MPI_INT, MPI_SUM, communicator);
  const bool ownership_ok =
      std::all_of(global_owners.begin(), global_owners.end(),
                  [](const int count) { return count == 1; }) &&
      (rank_count == 1 ? evidence.global_ghost_trace_dofs == 0 :
                         evidence.global_ghost_trace_dofs > 0) &&
      (rank_count == 1 ? evidence.global_shared_faces == 0 :
                         evidence.global_shared_faces > 0);
  passed = ownership_ok && passed;

  const std::vector<double> rollback_interior =
      Values(*experiment.current_interior());
  const std::vector<double> rollback_trace = Values(*experiment.current_trace());
  const std::uint64_t rollback_generation = experiment.generation();
  const SmoothEulerMPIStatus rollback_status = experiment.status();
  const std::uint64_t rollback_evidence_hash = EvidenceHash(experiment.evidence());
  bool rollback_ok = true;
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
        const bool failed =
            !experiment.Execute({point, failing_rank});
        rollback_ok = failed &&
            experiment.last_failure_point_reached() == point &&
            experiment.generation() == rollback_generation &&
            experiment.status() == rollback_status &&
            EvidenceHash(experiment.evidence()) == rollback_evidence_hash &&
            Values(*experiment.current_interior()) == rollback_interior &&
            Values(*experiment.current_trace()) == rollback_trace &&
            rollback_ok;
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
    std::cerr << "FAIL rank=" << rank << " ranks=" << rank_count
              << " p=" << polynomial_order
              << " setup_ok=" << setup_ok
              << " entry_assignment_ok=" << entry_assignment_ok
              << " local_oracle_ok=" << local_oracle_ok
              << " linearization_ok=" << linearization_ok
              << " condensation_ok=" << condensation_ok
              << " global_oracle_ok=" << global_oracle_ok
              << " local_rhs_ok=" << local_rhs_ok
              << " true_system_ok=" << true_system_ok
              << " distribution_ok=" << distribution_ok
              << " recovery_ok=" << recovery_ok
              << " state_publication_ok=" << state_publication_ok
              << " final_oracle_ok=" << final_oracle_ok
              << " nonlinear_and_evidence_ok="
              << nonlinear_and_evidence_ok
              << " ownership_ok=" << ownership_ok
              << " rollback_ok=" << rollback_ok
              << " damping_ok=" << damping_ok
              << " local_error_R="
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
              << " orientation_value_error=" << orientation.value_error
              << " orientation_action_error=" << orientation.action_error
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
